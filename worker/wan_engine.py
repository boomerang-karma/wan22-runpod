"""Wan 2.2 image-to-video (A14B MoE) + Lightning 4-step LoRA, for a RunPod serverless worker.

Two responsibilities:
  prepare()   one-time: download the public Diffusers checkpoint (~126 GB, fp32 experts), cast to bf16, fuse the
              Lightning step-distill LoRA into each expert, and write a compact (~69 GB) model to the network volume.
  WanEngine   per-worker: load the prepared model once, then generate() a clip per request.

Design notes (why it is shaped this way):
  * The LoRA is fused at prepare time, so the hot path has no PEFT layers and no per-cold-start fusing.
  * Memory mode "resident" (default, 80 GB GPUs): both 14B experts (~2 x 28.6 GB bf16) and the VAE stay on the GPU;
    the 5.7B UMT5 text encoder (~11.4 GB) is parked in host RAM and moved to the GPU only to encode the prompt.
    Because components then live on different devices, the pipeline's execution device is pinned explicitly.
  * Memory mode "offload" (48 GB GPUs, needs ~75 GB host RAM): diffusers model CPU offload moves each component in
    and out per request. Slower, but fits smaller cards.
  * Sampler matches the reference ComfyUI Lightning workflow: Euler flow-matching, shift 5.0, 4 steps, CFG 1.0,
    high-noise expert for the first 2 steps and low-noise expert for the last 2 (boundary_ratio 0.9 gives this split).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from typing import Callable, Optional

import numpy as np
import torch
from diffusers import (AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline,
                       WanTransformer3DModel)
from transformers import AutoTokenizer, UMT5EncoderModel

log = logging.getLogger("wan_engine")

SRC_REPO = os.getenv("WAN_REPO", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
SRC_REVISION = os.getenv("WAN_REVISION") or None
LORA_REPO = os.getenv("LORA_REPO", "lightx2v/Wan2.2-Lightning")
LORA_REVISION = os.getenv("LORA_REVISION") or None
LORA_HIGH = os.getenv("LORA_HIGH", "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors")
LORA_LOW = os.getenv("LORA_LOW", "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors")
LORA_STRENGTH_HIGH = float(os.getenv("LORA_STRENGTH_HIGH", "1.0"))
LORA_STRENGTH_LOW = float(os.getenv("LORA_STRENGTH_LOW", "1.0"))
MARKER = "prepared.json"
LORA_TARGETS_PER_BLOCK = 10     # 4 self-attn + 4 cross-attn + 2 FFN linears -> 400 for the 40-block A14B experts

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _free(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- LoRA fusing
def fuse_lightning_lora(model: WanTransformer3DModel, lora: str | dict, strength: float) -> dict:
    """Fuse a Lightning LoRA (original Wan key format) into `model` in place and verify it actually changed it.

    Returns stats. Raises if the LoRA matched too few modules or left the probe weight unchanged, so a key-format
    mismatch fails loudly at prepare time instead of silently producing an un-distilled (blurry 4-step) model.
    """
    state = WanImageToVideoPipeline.lora_state_dict(lora)          # converts diffusion_model.* keys -> transformer.*
    if isinstance(state, tuple):                                      # (state_dict, metadata) on some versions
        state = state[0]
    n_keys = len(state)
    probe = model.blocks[0].attn1.to_q.weight
    before = probe.detach().float().clone()
    model.load_lora_adapter(state, prefix="transformer", adapter_name="lightning")
    n_modules = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A") and "lightning" in getattr(m, "lora_A", {}))
    expected = LORA_TARGETS_PER_BLOCK * model.config.num_layers
    if n_modules < 0.9 * expected:
        raise RuntimeError(f"Lightning LoRA matched only {n_modules} modules (expected ~{expected}); key format mismatch?")
    model.fuse_lora(lora_scale=strength)
    model.unload_lora()
    after = model.blocks[0].attn1.to_q.weight.detach().float()
    delta = (after - before.to(after.device)).abs().max().item()
    if delta == 0.0:
        raise RuntimeError("LoRA fuse left the probe weight unchanged; refusing to save an un-fused model")
    del state, before
    return {"lora_keys": n_keys, "lora_modules": n_modules, "strength": strength, "probe_max_abs_delta": delta}


# ----------------------------------------------------------------------------------------------- one-time prepare
def prepare(out_dir: str, work_dir: str, device: str = "cuda", progress: Progress = _noop, force: bool = False) -> dict:
    """Build the fused bf16 model at `out_dir` (on the network volume). Idempotent; safe to re-run."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if os.path.exists(os.path.join(out_dir, MARKER)) and not force:
        with open(os.path.join(out_dir, MARKER)) as f:
            return {"status": "already_prepared", **json.load(f)}

    t_start = time.time()
    partial = out_dir.rstrip("/") + ".partial"
    src = os.path.join(work_dir, "wan_src")
    for d in (partial, src):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(partial, exist_ok=True)
    os.makedirs(src, exist_ok=True)
    stats: dict = {"source": SRC_REPO, "source_revision": SRC_REVISION, "lora_repo": LORA_REPO,
                   "lora_high": LORA_HIGH, "lora_low": LORA_LOW}

    def fetch(patterns: list[str]) -> None:
        t = time.time()
        snapshot_download(SRC_REPO, revision=SRC_REVISION, local_dir=src, allow_patterns=patterns)
        progress(f"downloaded {patterns} in {time.time() - t:.0f}s")

    # 1. small components are copied as-is (VAE stays fp32 for decode quality)
    fetch(["model_index.json", "scheduler/*", "tokenizer/*", "vae/*"])
    shutil.copy2(os.path.join(src, "model_index.json"), partial)
    for sub in ("tokenizer", "vae", "scheduler"):          # scheduler config is informational (engine builds its own)
        if sub == "scheduler" and not os.path.isdir(os.path.join(src, sub)):
            continue
        shutil.copytree(os.path.join(src, sub), os.path.join(partial, sub))

    # 2. text encoder -> bf16
    fetch(["text_encoder/*"])
    te = UMT5EncoderModel.from_pretrained(src, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    te.save_pretrained(os.path.join(partial, "text_encoder"), max_shard_size="5GB")
    del te
    shutil.rmtree(os.path.join(src, "text_encoder"), ignore_errors=True)
    _free(device)
    progress("text encoder saved (bf16)")

    # 3. each expert: download -> bf16 -> fuse its Lightning LoRA on the GPU -> save -> free disk and memory
    for sub, lora_file, strength in (("transformer", LORA_HIGH, LORA_STRENGTH_HIGH),
                                     ("transformer_2", LORA_LOW, LORA_STRENGTH_LOW)):
        fetch([f"{sub}/*"])
        lora_path = hf_hub_download(LORA_REPO, lora_file, revision=LORA_REVISION, cache_dir=os.path.join(work_dir, "hf"))
        t = time.time()
        model = WanTransformer3DModel.from_pretrained(src, subfolder=sub, torch_dtype=torch.bfloat16)
        model.to(device)
        stats[sub] = fuse_lightning_lora(model, lora_path, strength)
        model.to("cpu")                                   # serialize from host memory (safe for any safetensors build)
        model.save_pretrained(os.path.join(partial, sub), max_shard_size="10GB")
        del model
        _free(device)
        shutil.rmtree(os.path.join(src, sub), ignore_errors=True)
        progress(f"{sub}: fused {stats[sub]['lora_modules']} LoRA modules and saved bf16 in {time.time() - t:.0f}s")

    import diffusers
    import transformers
    stats.update({"prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dtype": "bfloat16",
                  "diffusers": diffusers.__version__, "transformers": transformers.__version__,
                  "torch": torch.__version__, "seconds": round(time.time() - t_start, 1)})
    with open(os.path.join(partial, MARKER), "w") as f:
        json.dump(stats, f, indent=2)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(partial, out_dir)                      # atomic on the same filesystem: never a half-written model dir
    shutil.rmtree(src, ignore_errors=True)
    return {"status": "prepared", **stats}


# ----------------------------------------------------------------------------------------------- runtime engine
class _PinnedDevicePipeline(WanImageToVideoPipeline):
    """In "resident" mode the text encoder sits on the CPU while everything else is on the GPU. diffusers infers the
    execution device from the first module it finds, which could be the CPU-parked encoder, so pin it."""

    @property
    def _execution_device(self):
        forced = self.__dict__.get("_forced_device")
        return torch.device(forced) if forced is not None else super()._execution_device


def _load_model(cls, path: str, subfolder: str, dtype, device: Optional[str]):
    """Load straight onto `device` when possible (skips a host-RAM copy of 28 GB experts), else load then move."""
    if device is None:
        return cls.from_pretrained(path, subfolder=subfolder, torch_dtype=dtype)
    try:
        return cls.from_pretrained(path, subfolder=subfolder, torch_dtype=dtype, device_map={"": device})
    except (TypeError, ValueError, NotImplementedError) as e:      # older/newer loaders may reject device_map
        log.warning("device_map load failed for %s (%s); falling back to load-then-move", subfolder, e)
        return cls.from_pretrained(path, subfolder=subfolder, torch_dtype=dtype).to(device)


class WanEngine:
    def __init__(self, model_dir: str, device: str = "cuda", memory_mode: str = "resident",
                 vae_tiling: bool = False, dtype: torch.dtype = torch.bfloat16):
        if memory_mode not in ("resident", "offload"):
            raise ValueError("memory_mode must be 'resident' or 'offload'")
        t0 = time.time()
        self.device, self.memory_mode, self.dtype = device, memory_mode, dtype
        with open(os.path.join(model_dir, MARKER)) as f:
            self.meta = json.load(f)
        with open(os.path.join(model_dir, "model_index.json")) as f:
            boundary = json.load(f).get("boundary_ratio", 0.9)
        on = device if memory_mode == "resident" else None
        tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(model_dir, subfolder="text_encoder", torch_dtype=dtype)
        vae = _load_model(AutoencoderKLWan, model_dir, "vae", torch.float32, on)
        t_high = _load_model(WanTransformer3DModel, model_dir, "transformer", dtype, on)
        t_low = _load_model(WanTransformer3DModel, model_dir, "transformer_2", dtype, on)
        self.pipe = _PinnedDevicePipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
                                          scheduler=FlowMatchEulerDiscreteScheduler(shift=5.0),
                                          transformer=t_high, transformer_2=t_low, boundary_ratio=boundary)
        if memory_mode == "resident":
            self.pipe.__dict__["_forced_device"] = device
        else:
            self.pipe.enable_model_cpu_offload(device=device)
        if vae_tiling:
            vae.enable_tiling()
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.load_seconds = round(time.time() - t0, 1)
        self.gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else device

    @torch.inference_mode()
    def generate(self, image, prompt: str, width: int, height: int, num_frames: int = 81, steps: int = 4,
                 guidance: float = 1.0, guidance_2: Optional[float] = None, shift: float = 5.0, seed: int = 0,
                 negative_prompt: str = "", progress: Progress = _noop) -> tuple[np.ndarray, dict]:
        """Returns (frames uint8 [F,H,W,3], timings)."""
        timings: dict = {}
        g2 = guidance if guidance_2 is None else guidance_2
        cfg = guidance > 1.0 or g2 > 1.0                 # Lightning is CFG-distilled: 1.0 = no negative pass (2x faster)
        pipe = self.pipe
        pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)

        t = time.time()
        if self.memory_mode == "resident":
            pipe.text_encoder.to(self.device)
            prompt_embeds, negative_embeds = pipe.encode_prompt(
                prompt, negative_prompt if cfg else None, do_classifier_free_guidance=cfg, num_videos_per_prompt=1,
                max_sequence_length=512, device=torch.device(self.device), dtype=self.dtype)
            pipe.text_encoder.to("cpu")
            _free(self.device)
            call_text = {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_embeds}
        else:
            call_text = {"prompt": prompt, "negative_prompt": negative_prompt if cfg else None}
        timings["encode_prompt_s"] = round(time.time() - t, 2)

        t = time.time()
        step_times: list[float] = []

        def on_step(_pipe, i, _t, kwargs):
            step_times.append(time.time())
            progress(f"denoise step {i + 1}/{steps}")
            return kwargs

        out = pipe(image=image, height=height, width=width, num_frames=num_frames, num_inference_steps=steps,
                   guidance_scale=guidance, guidance_scale_2=g2, generator=torch.Generator("cpu").manual_seed(seed),
                   output_type="np", max_sequence_length=512, callback_on_step_end=on_step, **call_text)
        frames = out.frames[0]
        timings["denoise_and_decode_s"] = round(time.time() - t, 2)
        if step_times:
            timings["denoise_s"] = round(step_times[-1] - t, 2)
            timings["vae_decode_s"] = round(time.time() - step_times[-1], 2)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            timings["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            torch.cuda.reset_peak_memory_stats()
        frames = np.asarray(frames)
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return frames, timings
