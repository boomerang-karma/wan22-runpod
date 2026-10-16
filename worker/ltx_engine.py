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
  * Quality "full": stage 1 (motion, faces, at half size) runs the full / SFT transformer for ~30 steps with real
    guidance (CFG + STG + modality, negative prompt); stage 2 is the distilled 3-step refine at full size. Only one
    22B transformer fits next to everything else on an 80 GB card, so the other one waits in pinned host RAM and
    they swap by pointer (~2 x 38 GB over PCIe, a few seconds) around stage 2.
  * Optional end frame: with `end_image`, LTX2ConditionPipeline pins the photo (or another image) as the LAST frame
    too, so the clip has to return to it. This keeps identities and poses anchored over long clips.
  * The parked text encoder lives in pinned host memory and its weights are swapped onto the GPU by pointer, so a
    visit costs ~1-2 s of PCIe copy instead of tens of seconds; recent prompt embeddings are cached.
  * Only the convolutional VAE decoder is used. The optional diffusion decoder needs extra kernels and memory.
"""
from __future__ import annotations

import gc
import importlib
from collections import OrderedDict
import json
import logging
import os
import shutil
import time
from typing import Callable, Optional

import numpy as np
import torch
from diffusers import (FlowMatchEulerDiscreteScheduler, LTX2ConditionPipeline, LTX2ImageToVideoPipeline,
                       LTX2LatentUpsamplePipeline)
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import (DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES,
                                           STAGE_2_DISTILLED_SIGMA_VALUES)

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
PROMPT_CACHE = 8
FULL = "transformer_full"
FULL_STEPS = 30

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
def _fetch_fn(src: str, token: Optional[str], progress: Progress):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

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
    return fetch


def _fetch_component(fetch, src: str, sub: str) -> Optional[int]:
    """Download one component folder; when it has a *.index.json, only the shards that index names."""
    fetch([f"{sub}/*.json"])
    folder = os.path.join(src, sub)
    indexes = [f for f in os.listdir(folder) if f.endswith(".index.json")] if os.path.isdir(folder) else []
    if not indexes:
        fetch([f"{sub}/*"])
        return None
    with open(os.path.join(folder, indexes[0])) as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    fetch([f"{sub}/{s}" for s in shards])
    return len(shards)


def prepare_full(out_dir: str, work_dir: str, progress: Progress = _noop, token: Optional[str] = None) -> dict:
    """Add the full / SFT transformer (~38 GB) next to an already prepared model. Idempotent."""
    with open(os.path.join(out_dir, MARKER)) as f:
        meta = json.load(f)
    if os.path.isdir(os.path.join(out_dir, FULL)) and meta.get(FULL):
        return {"status": "already_prepared", **meta}
    src = os.path.join(work_dir, "ltx_src_full")
    shutil.rmtree(src, ignore_errors=True)
    os.makedirs(src, exist_ok=True)
    t = time.time()
    shards = _fetch_component(_fetch_fn(src, token, progress), src, FULL)
    if not os.path.isdir(os.path.join(src, FULL)):
        raise RuntimeError(f"{SRC_REPO}@{SRC_REVISION} has no {FULL}/ folder")
    tmp = os.path.join(out_dir, FULL + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.move(os.path.join(src, FULL), tmp)
    os.replace(tmp, os.path.join(out_dir, FULL))
    shutil.rmtree(src, ignore_errors=True)
    meta[FULL] = {"shards": shards, "seconds": round(time.time() - t, 1)}
    with open(os.path.join(out_dir, MARKER + ".tmp"), "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(os.path.join(out_dir, MARKER + ".tmp"), os.path.join(out_dir, MARKER))
    progress(f"{FULL}: saved")
    return {"status": "prepared", **meta}


def prepare(out_dir: str, work_dir: str, device: str = "cuda", progress: Progress = _noop, force: bool = False,
            token: Optional[str] = None, full: bool = False) -> dict:
    """Build the bf16 model at `out_dir` (on the network volume). Idempotent; safe to re-run.
    `full` also adds the full / SFT transformer used by quality "full"."""
    if os.path.exists(os.path.join(out_dir, MARKER)) and not force:
        if full:
            return prepare_full(out_dir, work_dir, progress, token)
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
    fetch = _fetch_fn(src, token, progress)

    fetch(["model_index.json"])
    shutil.copy2(os.path.join(src, "model_index.json"), partial)
    with open(os.path.join(src, "model_index.json")) as f:
        index = json.load(f)

    # one component at a time, so the container disk never holds more than the biggest one (~38 GB transformer)
    for sub in COMPONENTS:
        if sub not in index and sub != "latent_upsampler":        # the upsampler is not a pipeline component
            continue
        shards = _fetch_component(fetch, src, sub)
        if shards:
            stats[sub] = {"shards": shards}
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
    if full:
        return prepare_full(out_dir, work_dir, progress, token)
    return {"status": "prepared", **stats}


# ----------------------------------------------------------------------------------------------- runtime engine
class _PinnedDeviceMixin:
    """In "resident" mode the text encoder sits on the CPU while everything else is on the GPU. diffusers infers the
    execution device from the first module it finds, which could be the CPU-parked encoder, so pin it."""

    @property
    def _execution_device(self):
        forced = self.__dict__.get("_forced_device")
        return torch.device(forced) if forced is not None else super()._execution_device


class _PinnedDevicePipeline(_PinnedDeviceMixin, LTX2ImageToVideoPipeline):
    pass


class _PinnedDeviceConditionPipeline(_PinnedDeviceMixin, LTX2ConditionPipeline):
    pass


class _HostParked:
    """Keeps a module's weights in pinned (page-locked) host memory and points them at a GPU copy only while needed.
    `module.to("cpu")` would re-allocate ~24 GB of pageable memory on every visit; swapping `.data` pointers does not,
    and pinned memory copies to the GPU at full PCIe speed."""

    def __init__(self, module: torch.nn.Module, device: str):
        """`module` may start on the CPU or the GPU; afterwards it is parked (on the host) until `up()`."""
        self.module, self.device = module, device
        pin = device.startswith("cuda") and torch.cuda.is_available()
        self.host = {}
        for key, t in self._tensors():
            host = t.data.to("cpu")
            t.data = host.pin_memory() if pin else host
            self.host[key] = t.data
        _free(device)

    def _tensors(self):
        for name, p in self.module.named_parameters():
            yield "p:" + name, p
        for name, b in self.module.named_buffers():
            yield "b:" + name, b

    def up(self) -> None:
        for key, t in self._tensors():
            t.data = self.host[key].to(self.device, non_blocking=True)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def down(self) -> None:
        for key, t in self._tensors():
            t.data = self.host[key]


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
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
        self.pipe = _PinnedDevicePipeline(scheduler=scheduler, **parts)
        self.cond_pipe = _PinnedDeviceConditionPipeline(scheduler=scheduler, **parts)    # same modules, end frames
        upsampler = LTX2LatentUpsamplerModel.from_pretrained(model_dir, subfolder="latent_upsampler", torch_dtype=dtype)
        self.upsample = LTX2LatentUpsamplePipeline(vae=parts["vae"], latent_upsampler=upsampler.to(device))
        self._prompts: OrderedDict = OrderedDict()
        self.model_dir, self._index = model_dir, index
        self.has_full = os.path.isdir(os.path.join(model_dir, FULL))
        self._distilled, self._full, self._full_parked, self._distilled_parked = parts["transformer"], None, None, None
        self._distilled_scheduler = scheduler
        self._full_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            scheduler.config, use_dynamic_shifting=True, shift_terminal=0.1)
        if memory_mode == "resident":
            self.pipe.__dict__["_forced_device"] = device
            self.cond_pipe.__dict__["_forced_device"] = device
            self._parked = _HostParked(parts["text_encoder"], device)
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

    def _encode(self, prompt: str, negative: Optional[str] = None) -> dict:
        """Prompt embeddings (+ negative ones when `negative` is not None, for guided sampling)."""
        pipe = self.pipe
        keys = ("prompt_embeds", "prompt_attention_mask", "negative_prompt_embeds", "negative_prompt_attention_mask")
        if self.memory_mode != "resident":
            return {"prompt": prompt, **({"negative_prompt": negative} if negative is not None else {})}
        key = (prompt, negative)
        if key in self._prompts:                                     # re-rolls of the same shot skip the encoder
            self._prompts.move_to_end(key)
            return {k: v.to(self.device) for k, v in zip(keys, self._prompts[key]) if v is not None}

        def encode():
            self._parked.up()
            try:
                out = pipe.encode_prompt(prompt, negative, do_classifier_free_guidance=negative is not None,
                                         max_sequence_length=MAX_SEQ_LEN, device=torch.device(self.device))
                return out if negative is not None else (out[0], out[1], None, None)
            finally:
                self._parked.down()
                _free(self.device)

        try:
            enc = encode()
        except torch.OutOfMemoryError:
            # ~48 GB resident + ~24 GB encoder is close to an 80 GB card's limit: park the transformer for a moment
            log.warning("prompt encode hit CUDA OOM; retrying with the transformer moved to host RAM")
            pipe.transformer.to("cpu")
            _free(self.device)
            try:
                enc = encode()
            finally:
                pipe.transformer.to(self.device)
        self._prompts[key] = tuple(v.to("cpu") if v is not None else None for v in enc)
        while len(self._prompts) > PROMPT_CACHE:
            self._prompts.popitem(last=False)
        return {k: v for k, v in zip(keys, enc) if v is not None}

    def _use_transformer(self, which: str) -> None:
        """Put the distilled or the full transformer on the GPU (the other one waits in pinned host RAM)."""
        want = self._full if which == "full" else self._distilled
        if self.pipe.transformer is want:
            return
        if which == "full":
            if self._full is None:                                   # first full request on this worker
                index = {**self._index, FULL: self._index["transformer"]}        # same class, other folder
                self._full = _load(index, self.model_dir, FULL, self.dtype)
                self._full_parked = _HostParked(self._full, self.device)
            if self._distilled_parked is None:
                self._distilled_parked = _HostParked(self._distilled, self.device)
            else:
                self._distilled_parked.down()
            _free(self.device)
            self._full_parked.up()
            want = self._full
        else:
            self._full_parked.down()
            _free(self.device)
            self._distilled_parked.up()
        for p in (self.pipe, self.cond_pipe):
            p.register_modules(transformer=want)

    @torch.inference_mode()
    def generate(self, image, prompt: str, width: int, height: int, num_frames: int = 121, frame_rate: float = 24.0,
                 seed: int = 0, two_stage: bool = True, end_image=None, quality: str = "fast",
                 steps: int = FULL_STEPS, negative_prompt: str = "", mid_strength: float = 0.0,
                 progress: Progress = _noop):
        """Returns (frames uint8 [F,H,W,3], audio float32 [channels, samples], sample_rate, timings).
        `end_image` (same size as `image`) is pinned as the last frame; the clip must return to it.
        `quality` "full" runs stage 1 on the full transformer for `steps` guided steps (needs `prepare --full`).
        `mid_strength` > 0 also pins `image` at the middle frame with that strength (holds people in place)."""
        if quality not in ("fast", "full"):
            raise ValueError("quality must be 'fast' or 'full'")
        full = quality == "full"
        if full and not self.has_full:
            raise ValueError("quality 'full' needs the full transformer on the volume: run `wanctl.py prepare --full`")
        if full and self.memory_mode != "resident":
            raise ValueError("quality 'full' needs memory_mode 'resident'")
        timings: dict = {}
        stage1 = steps if full else len(DISTILLED_SIGMA_VALUES)
        total = stage1 + (len(STAGE_2_DISTILLED_SIGMA_VALUES) if two_stage else 0)
        done = [0]

        def on_step(_pipe, i, _t, kwargs):
            done[0] += 1
            progress(f"denoise step {done[0]}/{total}")
            return kwargs

        t = time.time()
        text = self._encode(prompt)
        guided = self._encode(prompt, ", ".join(x for x in (negative_prompt, DEFAULT_NEGATIVE_PROMPT) if x)) \
            if full else None
        timings["encode_prompt_s"] = round(time.time() - t, 2)
        generator = torch.Generator("cpu").manual_seed(seed)
        common = dict(num_frames=num_frames, frame_rate=frame_rate, generator=generator,
                      max_sequence_length=MAX_SEQ_LEN, callback_on_step_end=on_step, return_dict=False)
        if end_image is None and mid_strength <= 0:
            pipe = self.pipe
            common["image"] = image
        else:
            if self.memory_mode != "resident":
                raise ValueError("keyframes need memory_mode 'resident'")
            pipe = self.cond_pipe
            conds = [LTX2VideoCondition(frames=image, index=0, strength=1.0)]
            if mid_strength > 0:
                mid = max(8, (num_frames - 1) // 2 // 8 * 8)            # on the VAE's 8-frame grid
                conds.append(LTX2VideoCondition(frames=image, index=mid, strength=float(mid_strength)))
            if end_image is not None:
                conds.append(LTX2VideoCondition(frames=end_image, index=-1, strength=1.0))
            common["conditions"] = conds
        fast = {**UNGUIDED, **text}

        if full:                                                     # stage 1: full transformer, guided
            t = time.time()
            self._use_transformer("full")
            pipe.scheduler = self._full_scheduler
            timings["swap_s"] = round(time.time() - t, 2)
            h1, w1 = (height // 2, width // 2) if two_stage else (height, width)
            t = time.time()
            try:
                out = pipe(height=h1, width=w1, num_inference_steps=steps, output_type="latent" if two_stage else "np",
                           **common, **guided)
            finally:
                pipe.scheduler = self._distilled_scheduler
                t2 = time.time()
                self._use_transformer("distilled")
                timings["swap_s"] += round(time.time() - t2, 2)
            timings["stage1_s"] = round(time.time() - t - (time.time() - t2), 2)
            if two_stage:
                latents, audio_latents = out
            else:
                video, audio = out
        if two_stage:
            if not full:
                t = time.time()
                latents, audio_latents = pipe(height=height // 2, width=width // 2, sigmas=DISTILLED_SIGMA_VALUES,
                                              output_type="latent", **common, **fast)
                timings["stage1_s"] = round(time.time() - t, 2)
            t = time.time()
            latents = self.upsample(latents=latents, output_type="latent", return_dict=False)[0]
            timings["upsample_s"] = round(time.time() - t, 2)
            t = time.time()
            video, audio = pipe(height=height, width=width, sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                                latents=latents, audio_latents=audio_latents,
                                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], output_type="np", **common, **fast)
            timings["stage2_and_decode_s"] = round(time.time() - t, 2)
        elif not full:
            t = time.time()
            video, audio = pipe(height=height, width=width, sigmas=DISTILLED_SIGMA_VALUES, output_type="np",
                                **common, **fast)
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
