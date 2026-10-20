"""LTX-2.5 image-to-video (+ audio) on Modal: the same worker code as RunPod (worker/handler.py, worker/ltx_engine.py),
with the weights on a Modal Volume and GPUs from Modal's whole pool (no region pinning, scale to zero).

  modal deploy deploy/modal/ltx_app.py                    create / update the app (no GPU time)
  modal run deploy/modal/ltx_app.py::prepare              once: download ~155 GB of weights onto the Volume (CPU only)
  modal run deploy/modal/ltx_app.py::info                 start a GPU container and report what it loaded (billed)
  modal run deploy/modal/ltx_app.py::generate --image photo.jpg --prompt "..."   one clip -> outputs/

The Hugging Face token (the repo is gated) is read locally from $HF_TOKEN or config.yaml and sent only in the prepare
call; it is not stored on Modal. Idle cost: $0 compute once scaled to zero; the Volume fits Modal's free 1 TiB.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import modal

APP_NAME = "ltx-memes"
_HERE = Path(__file__).resolve()
# repo root locally; inside a Modal container this file is /root/ltx_app.py and the repo isn't there (not needed)
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
WORKER = ROOT / "worker"
WEIGHTS = "/weights"
MODEL_SUBDIR = "ltx25-i2v-distilled-bf16"

# Ordered by preference = price: Modal tries the first type, then falls back. All have >= 80 GB, which the
# "resident" memory mode needs. "H100" may be upgraded to an H200 at the H100 price.
GPUS = ["A100-80GB", "H100", "RTX-PRO-6000", "H200", "B200"]
CPU = 4.0                       # physical cores
MEMORY = (65536, 131072)        # MiB request / limit: the 24 GB text encoder waits in pinned host RAM;
                                # "full" quality also parks the 38 GB distilled transformer there
SCALEDOWN_S = 120               # a warm container waits this long for the next meme (billed), then scales to zero

WORKER_ENV = {
    "WORKER_MODE": "real",
    "VOLUME_PATH": WEIGHTS,
    "MODEL_SUBDIR": MODEL_SUBDIR,
    "MEMORY_MODE": "resident",
    "VAE_TILING": "1",
    "INLINE_LIMIT_MB": "200",           # Modal returns large results directly; no 10 MB cap like RunPod's /run
    "WORK_DIR": "/tmp/ltx_work",        # prepare stages each component on local disk, then moves it to the Volume
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "ca-certificates")
    .pip_install("torch==2.14.1")                                  # CUDA 13.0 wheels; Modal hosts run CUDA 13.3
    .pip_install_from_requirements(str(WORKER / "requirements.txt"))
    .env(WORKER_ENV)
    .add_local_file(WORKER / "handler.py", "/app/handler.py")
    .add_local_file(WORKER / "ltx_engine.py", "/app/ltx_engine.py")
    .add_local_file(WORKER / "mock_engine.py", "/app/mock_engine.py")
)

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("ltx25-weights", create_if_missing=True)
progress_board = modal.Dict.from_name("ltx-memes-progress", create_if_missing=True)   # call id -> latest stage


@app.function(image=image, volumes={WEIGHTS: weights}, cpu=4.0, memory=16384, ephemeral_disk=512 * 1024,
              timeout=3 * 3600)
def prepare_weights(hf_token: str, full: bool = True, force: bool = False) -> dict:
    """CPU only: download the components the engine loads (and the full transformer when `full`) to the Volume."""
    sys.path.insert(0, "/app")
    import ltx_engine
    os.makedirs(WORKER_ENV["WORK_DIR"], exist_ok=True)
    res = ltx_engine.prepare(f"{WEIGHTS}/{MODEL_SUBDIR}", WORKER_ENV["WORK_DIR"], device="cpu", force=force,
                             progress=lambda m: print(m, flush=True), token=hf_token or None, full=full)
    weights.commit()
    return res


@app.cls(image=image, gpu=GPUS, volumes={WEIGHTS: weights}, cpu=CPU, memory=MEMORY, timeout=900,
         scaledown_window=SCALEDOWN_S, max_containers=1)
class LTX:
    @modal.enter()
    def load(self):
        sys.path.insert(0, "/app")
        import handler                      # loads the model at import, exactly as the RunPod worker does
        self.handler = handler
        handler._progress = self._progress  # stages go to a Modal Dict the portal can read

    def _progress(self, job: dict, msg: str) -> None:
        self.handler.log.info(msg)
        try:
            progress_board.put(job["id"], msg)
        except Exception:                   # progress is cosmetic; never fail a job over it
            pass

    @modal.method()
    def run(self, inp: dict) -> dict:
        """Same input and output as the RunPod worker: action generate (default) | info."""
        if inp.get("action") == "prepare":
            return {"error": "use the prepare_weights function on Modal"}
        call_id = modal.current_function_call_id() or f"local-{time.time():.0f}"
        out = self.handler.handler({"id": call_id, "input": inp})
        if isinstance(out, dict) and out.get("refresh_worker"):
            # CUDA OOM or a broken GPU: answer this input, then let Modal replace the container
            modal.experimental.stop_fetching_inputs()
            out = out.get("job_results") or {}
        try:
            progress_board.pop(call_id)
        except Exception:
            pass
        return out


# ----------------------------------------------------------------------------------------------- local commands
def _hf_token() -> str:
    tok = os.getenv("HF_TOKEN", "").strip()
    if not tok and (ROOT / "config.yaml").exists():
        import yaml
        tok = ((yaml.safe_load((ROOT / "config.yaml").read_text()) or {}).get("huggingface") or {}).get("token", "")
    if not tok:
        raise SystemExit("no Hugging Face token: set HF_TOKEN or config.yaml -> huggingface.token "
                         "(and accept the license at https://huggingface.co/Lightricks/LTX-2.5-Diffusers)")
    return tok.strip()


@app.local_entrypoint()
def prepare(full: bool = True, force: bool = False):
    print("downloading the weights onto the Modal Volume (CPU container; about 20-40 minutes, a few cents)...")
    print(prepare_weights.remote(_hf_token(), full=full, force=force))


@app.local_entrypoint()
def info():
    import json
    print(json.dumps(LTX().run.remote({"action": "info"}), indent=2))


@app.local_entrypoint()
def generate(image: str, prompt: str, resolution: str = "540p", num_frames: int = 121, seed: int = -1,
             quality: str = "fast", out: str = ""):
    import base64
    import json
    inp = {"image_base64": base64.b64encode(Path(image).read_bytes()).decode(), "prompt": prompt,
           "resolution": resolution, "num_frames": num_frames, "quality": quality}
    if seed >= 0:
        inp["seed"] = seed
    t0 = time.time()
    res = LTX().run.remote(inp)
    if "video_base64" not in res:
        raise SystemExit(f"failed: {res.get('error')}")
    dst = Path(out or ROOT / "outputs" / f"modal_{time.strftime('%Y%m%d_%H%M%S')}_s{res['seed']}.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(base64.b64decode(res.pop("video_base64")))
    print(f"saved {dst} in {time.time() - t0:.0f}s")
    print(json.dumps(res, indent=2))
