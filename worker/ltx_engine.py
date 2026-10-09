"""LTX-2.5 image-to-video with synchronized audio (distilled), for a RunPod serverless worker.

Two responsibilities:
  prepare()   one-time: download the gated Diffusers checkpoint (Lightricks/LTX-2.5-Diffusers, needs an HF token
              whose account accepted the license), one component at a time, and write ~71 GB (bf16) to the network
              volume. Some folders ship the same weights twice with different sharding, so only the shards named
              by each component's *.index.json are fetched.
  LTXEngine   per-worker: load the prepared model once, then generate() a clip + audio track per request.

Design notes (why it is shaped this way):
  * Distilled recipe from the model card: explicit sigma schedules, every guidance knob neutral (one forward pass
    per step, no negative prompt). Two-stage by default: 8 steps at half resolution -> x2 latent upsampler ->
    3 refinement steps at full resolution. Single-stage (8 steps at full resolution) is the cheap draft path.
  * Memory mode "resident" (default, 80 GB GPUs): transformer (~38 GB bf16), connectors, VAEs, vocoder and
    upsampler stay on the GPU (~48 GB); the 12B Gemma text encoder (~24 GB) is parked in host RAM and visits the
    GPU only to encode the prompt. Components then live on different devices, so the execution device is pinned.
  * Memory mode "offload": diffusers model CPU offload moves each component in and out per request. Needs ~80 GB
    of host RAM; slower, but fits 48 GB cards.
  * Only the convolutional VAE decoder is used. The optional diffusion decoder needs extra kernels and memory.
"""
from __future__ import annotations

import gc
import importlib
import json
import logging
import os
import shutil
import time
from typing import Callable, Optional

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

log = logging.getLogger("ltx_engine")

SRC_REPO = os.getenv("LTX_REPO", "Lightricks/LTX-2.5-Diffusers")
SRC_REVISION = os.getenv("LTX_REVISION", "a97165959b05f5eb52a9f40f19c55957e334abe9") or None   # pinned 2026-10-08
MARKER = "prepared.json"
COMPONENTS = ("scheduler", "tokenizer", "vae", "audio_vae", "vocoder", "duration_head", "latent_upsampler",
              "connectors", "text_encoder", "transformer")
REQUIRED = ("scheduler", "tokenizer", "text_encoder", "connectors", "transformer", "vae", "audio_vae", "vocoder",
            "latent_upsampler")
UNGUIDED = {"guidance_scale": 1.0, "audio_guidance_scale": 1.0, "stg_scale": 0.0, "audio_stg_scale": 0.0,
            "modality_scale": 1.0, "audio_modality_scale": 1.0}
MAX_SEQ_LEN = 1024

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _free(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _component_class(index: dict, name: str):
    """Resolve a model_index.json entry such as ["diffusers", "LTX2VideoTransformer3DModel"] or
    ["ltx2", "LTX2TextConnectors"] (pipeline-local classes are listed by their pipeline module)."""
    lib, cls = index[name][:2]
    try:
        mod = importlib.import_module(lib)
        return getattr(mod, cls)
    except (ImportError, AttributeError):
        return getattr(importlib.import_module(f"diffusers.pipelines.{lib}"), cls)


def _load(index: dict, path: str, name: str, dtype, device: Optional[str] = None):
    cls = _component_class(index, name)
    if not issubclass(cls, torch.nn.Module):                     # tokenizer, scheduler
        return cls.from_pretrained(path, subfolder=name)
    key = "dtype" if index[name][0] == "transformers" else "torch_dtype"
    if device is None:
        return cls.from_pretrained(path, subfolder=name, **{key: dtype})
    try:                                                          # straight onto the GPU: no host-RAM copy
        return cls.from_pretrained(path, subfolder=name, device_map={"": device}, **{key: dtype})
    except (TypeError, ValueError, NotImplementedError) as e:
        log.warning("device_map load failed for %s (%s); falling back to load-then-move", name, e)
        return cls.from_pretrained(path, subfolder=name, **{key: dtype}).to(device)


# ----------------------------------------------------------------------------------------------- one-time prepare
def prepare(out_dir: str, work_dir: str, device: str = "cuda", progress: Progress = _noop, force: bool = False,
            token: Optional[str] = None) -> dict:
    """Build the bf16 model at `out_dir` (on the network volume). Idempotent; safe to re-run."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if os.path.exists(os.path.join(out_dir, MARKER)) and not force:
        with open(os.path.join(out_dir, MARKER)) as f:
            return {"status": "already_prepared", **json.load(f)}

    t_start = time.time()
    partial = out_dir.rstrip("/") + ".partial"
    src = os.path.join(work_dir, "ltx_src")
    for d in (partial, src):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(partial, exist_ok=True)
    os.makedirs(src, exist_ok=True)
    stats: dict = {"source": SRC_REPO, "source_revision": SRC_REVISION}

    def fetch(patterns: list[str]) -> None:
        t = time.time()
        try:
            snapshot_download(SRC_REPO, revision=SRC_REVISION, local_dir=src, allow_patterns=patterns, token=token)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise RuntimeError(
                f"Hugging Face refused access to {SRC_REPO} ({type(e).__name__}). Accept the license at "
                f"https://huggingface.co/{SRC_REPO} with the account that owns the token, and send the token "
                f"(export HF_TOKEN=... before `wanctl.py prepare`).") from None
        progress(f"downloaded {patterns} in {time.time() - t:.0f}s")

    fetch(["model_index.json"])
    shutil.copy2(os.path.join(src, "model_index.json"), partial)
    with open(os.path.join(src, "model_index.json")) as f:
        index = json.load(f)

    # one component at a time, so the container disk never holds more than the biggest one (~38 GB transformer)
    for sub in COMPONENTS:
        if sub not in index and sub != "latent_upsampler":        # the upsampler is not a pipeline component
            continue
        fetch([f"{sub}/*.json"])
        indexes = [f for f in os.listdir(os.path.join(src, sub)) if f.endswith(".index.json")] \
            if os.path.isdir(os.path.join(src, sub)) else []
        if indexes:                                                # only the shards the index points at
            with open(os.path.join(src, sub, indexes[0])) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
            fetch([f"{sub}/{s}" for s in shards])
            stats[sub] = {"shards": len(shards)}
        else:
            fetch([f"{sub}/*"])
        if not os.path.isdir(os.path.join(src, sub)):
            if sub in REQUIRED:
                raise RuntimeError(f"{SRC_REPO}@{SRC_REVISION} has no {sub}/ folder; the repo layout changed")
            continue
        shutil.move(os.path.join(src, sub), os.path.join(partial, sub))
        progress(f"{sub}: saved")

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
class _PinnedDevicePipeline(LTX2ImageToVideoPipeline):
    """In "resident" mode the text encoder sits on the CPU while everything else is on the GPU. diffusers infers the
    execution device from the first module it finds, which could be the CPU-parked encoder, so pin it."""

    @property
    def _execution_device(self):
        forced = self.__dict__.get("_forced_device")
        return torch.device(forced) if forced is not None else super()._execution_device


class LTXEngine:
    def __init__(self, model_dir: str, device: str = "cuda", memory_mode: str = "resident",
                 vae_tiling: bool = True, dtype: torch.dtype = torch.bfloat16):
        if memory_mode not in ("resident", "offload"):
            raise ValueError("memory_mode must be 'resident' or 'offload'")
        t0 = time.time()
        self.device, self.memory_mode, self.dtype = device, memory_mode, dtype
        with open(os.path.join(model_dir, MARKER)) as f:
            self.meta = json.load(f)
        with open(os.path.join(model_dir, "model_index.json")) as f:
            index = json.load(f)
        on = device if memory_mode == "resident" else None
        parts = {name: _load(index, model_dir, name, dtype, on)
                 for name in ("transformer", "connectors", "vae", "audio_vae", "vocoder")}
        parts["text_encoder"] = _load(index, model_dir, "text_encoder", dtype)          # host RAM until needed
        parts["tokenizer"] = _load(index, model_dir, "tokenizer", dtype)
        if "duration_head" in index and os.path.isdir(os.path.join(model_dir, "duration_head")):
            parts["duration_head"] = _load(index, model_dir, "duration_head", dtype, on)
        self.pipe = _PinnedDevicePipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler"), **parts)
        upsampler = LTX2LatentUpsamplerModel.from_pretrained(model_dir, subfolder="latent_upsampler", torch_dtype=dtype)
        self.upsample = LTX2LatentUpsamplePipeline(vae=parts["vae"], latent_upsampler=upsampler.to(device))
        if memory_mode == "resident":
            self.pipe.__dict__["_forced_device"] = device
        else:
            self.pipe.enable_model_cpu_offload(device=device)
        if vae_tiling:
            parts["vae"].enable_tiling()
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.sample_rate = int(self.pipe.vocoder.config.output_sampling_rate)
        self.load_seconds = round(time.time() - t0, 1)
        self.gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else device

    def _encode(self, prompt: str) -> dict:
        pipe = self.pipe
        if self.memory_mode != "resident":
            return {"prompt": prompt}
        def encode():
            pipe.text_encoder.to(self.device)
            try:
                return pipe.encode_prompt(prompt, None, do_classifier_free_guidance=False,
                                          max_sequence_length=MAX_SEQ_LEN, device=torch.device(self.device))[:2]
            finally:
                pipe.text_encoder.to("cpu")
                _free(self.device)

        try:
            embeds, mask = encode()
        except torch.OutOfMemoryError:
            # ~48 GB resident + ~24 GB encoder is close to an 80 GB card's limit: park the transformer for a moment
            log.warning("prompt encode hit CUDA OOM; retrying with the transformer moved to host RAM")
            pipe.transformer.to("cpu")
            _free(self.device)
            try:
                embeds, mask = encode()
            finally:
                pipe.transformer.to(self.device)
        return {"prompt_embeds": embeds, "prompt_attention_mask": mask}

    @torch.inference_mode()
    def generate(self, image, prompt: str, width: int, height: int, num_frames: int = 121, frame_rate: float = 24.0,
                 seed: int = 0, two_stage: bool = True, progress: Progress = _noop):
        """Returns (frames uint8 [F,H,W,3], audio float32 [channels, samples], sample_rate, timings)."""
        timings: dict = {}
        total = len(DISTILLED_SIGMA_VALUES) + (len(STAGE_2_DISTILLED_SIGMA_VALUES) if two_stage else 0)
        done = [0]

        def on_step(_pipe, i, _t, kwargs):
            done[0] += 1
            progress(f"denoise step {done[0]}/{total}")
            return kwargs

        t = time.time()
        text = self._encode(prompt)
        timings["encode_prompt_s"] = round(time.time() - t, 2)
        generator = torch.Generator("cpu").manual_seed(seed)
        common = dict(image=image, num_frames=num_frames, frame_rate=frame_rate, generator=generator,
                      max_sequence_length=MAX_SEQ_LEN, callback_on_step_end=on_step, return_dict=False,
                      **UNGUIDED, **text)

        if two_stage:
            t = time.time()
            latents, audio_latents = self.pipe(height=height // 2, width=width // 2, sigmas=DISTILLED_SIGMA_VALUES,
                                               output_type="latent", **common)
            timings["stage1_s"] = round(time.time() - t, 2)
            t = time.time()
            latents = self.upsample(latents=latents, output_type="latent", return_dict=False)[0]
            timings["upsample_s"] = round(time.time() - t, 2)
            t = time.time()
            video, audio = self.pipe(height=height, width=width, sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                                     latents=latents, audio_latents=audio_latents,
                                     noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], output_type="np", **common)
            timings["stage2_and_decode_s"] = round(time.time() - t, 2)
        else:
            t = time.time()
            video, audio = self.pipe(height=height, width=width, sigmas=DISTILLED_SIGMA_VALUES, output_type="np",
                                     **common)
            timings["denoise_and_decode_s"] = round(time.time() - t, 2)

        if self.device.startswith("cuda") and torch.cuda.is_available():
            timings["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            torch.cuda.reset_peak_memory_stats()
        frames = np.asarray(video[0])
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        wav = audio[0].float().cpu().numpy() if audio is not None else None
        if wav is not None and wav.ndim == 1:
            wav = wav[None]
        return frames, wav, self.sample_rate, timings
