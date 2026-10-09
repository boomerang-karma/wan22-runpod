"""RunPod serverless entry point for LTX-2.5 image-to-video with synchronized audio (distilled).

Actions (job["input"]["action"]):
  generate (default)  image + prompt -> mp4 with an AAC audio track (inline base64, or a presigned URL when a bucket
                      is configured)
  prepare             one-time weight preparation onto the network volume, then the worker restarts itself
  info                cheap health/metadata check (no generation)

The model is loaded at import time, outside the handler, so the load is paid once per worker (cold start) and is
captured by FlashBoot. If the weights are not prepared yet, the worker still starts and only accepts `prepare`/`info`.
"""
from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import random
import subprocess
import tempfile
import time
import traceback
import urllib.request

import runpod
from PIL import Image, ImageOps

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("handler")

PROCESS_START = time.time()
MODE = os.getenv("WORKER_MODE", "real")                    # "real" | "mock" (CPU, for local tests)
VOLUME = os.getenv("VOLUME_PATH", "/runpod-volume")
MODEL_DIR = os.path.join(VOLUME, os.getenv("MODEL_SUBDIR", "ltx25-i2v-distilled-bf16"))
WORK_DIR = os.getenv("WORK_DIR", "/tmp/ltx_work")         # container disk; needs ~80 GB free during prepare
MEMORY_MODE = os.getenv("MEMORY_MODE", "resident")
VAE_TILING = os.getenv("VAE_TILING", "1") == "1"
INLINE_LIMIT_MB = float(os.getenv("INLINE_LIMIT_MB", "8"))   # base64 size; /run payloads are capped at 10 MB
MAX_IMAGE_MB = 25
# (short, long, two_stage). Two-stage runs at half size then upsamples x2, so its sides must be multiples of 64.
RESOLUTIONS = {"540p": (544, 960, False), "720p": (704, 1280, True), "1080p": (1088, 1920, True)}
MAX_SECONDS = 20.0

if MODE == "mock":
    import mock_engine as engine_mod
else:
    import ltx_engine as engine_mod

ENGINE = None
LOAD_ERROR: str | None = None
COLD_START_S: float | None = None
_FIRST_JOB = True


def _load_engine() -> None:
    global ENGINE, LOAD_ERROR, COLD_START_S
    if not os.path.exists(os.path.join(MODEL_DIR, engine_mod.MARKER)):
        LOAD_ERROR = f"model not prepared at {MODEL_DIR}; run the 'prepare' action once (python wanctl.py prepare)"
        log.warning(LOAD_ERROR)
        return
    try:
        if MODE == "mock":
            ENGINE = engine_mod.MockEngine()
        else:
            ENGINE = engine_mod.LTXEngine(MODEL_DIR, memory_mode=MEMORY_MODE, vae_tiling=VAE_TILING)
        COLD_START_S = round(time.time() - PROCESS_START, 1)
        log.info("engine ready on %s in %.1fs (memory_mode=%s)", ENGINE.gpu, COLD_START_S, MEMORY_MODE)
    except Exception as e:                                     # keep the worker alive so `info` can report it
        LOAD_ERROR = f"engine load failed: {e!r}"
        log.error("%s\n%s", LOAD_ERROR, traceback.format_exc())


_load_engine()


# ----------------------------------------------------------------------------------------------- input handling
class BadInput(ValueError):
    pass


def _int(inp: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = inp.get(key, default)
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise BadInput(f"'{key}' must be an integer")
    if not lo <= v <= hi:
        raise BadInput(f"'{key}' must be between {lo} and {hi}")
    return v


def _read_image(inp: dict) -> Image.Image:
    if inp.get("image_base64"):
        data = inp["image_base64"]
        if data.startswith("data:"):
            data = data.split(",", 1)[-1]
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise BadInput("'image_base64' is not valid base64")
    elif inp.get("image_url"):
        url = inp["image_url"]
        if not url.startswith(("https://", "http://")):
            raise BadInput("'image_url' must be http(s)")
        req = urllib.request.Request(url, headers={"User-Agent": "ltx25-runpod-worker"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read(MAX_IMAGE_MB * 2**20 + 1)
    else:
        raise BadInput("provide 'image_base64' or 'image_url'")
    if len(raw) > MAX_IMAGE_MB * 2**20:
        raise BadInput(f"image larger than {MAX_IMAGE_MB} MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise BadInput("could not decode the image")
    return img


def _bool(inp: dict, key: str, default: bool) -> bool:
    v = inp.get(key, default)
    if not isinstance(v, bool):
        raise BadInput(f"'{key}' must be true or false")
    return v


def _target_size(inp: dict, img: Image.Image) -> tuple[int, int, bool]:
    if inp.get("width") or inp.get("height"):
        two_stage = _bool(inp, "two_stage", True)
        w, h = _int(inp, "width", 0, 256, 1920), _int(inp, "height", 0, 256, 1920)
        mult = 64 if two_stage else 32
        if w % mult or h % mult:
            raise BadInput(f"'width' and 'height' must be multiples of {mult}" + (" (two-stage)" if two_stage else ""))
        return w, h, two_stage
    res = inp.get("resolution", "720p")
    if res not in RESOLUTIONS:
        raise BadInput(f"'resolution' must be one of {sorted(RESOLUTIONS)}")
    short, long, two_stage = RESOLUTIONS[res]
    return ((short, long) if img.height >= img.width else (long, short)) + (two_stage,)


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to cover w x h and centre-crop (no distortion, no padding)."""
    return ImageOps.fit(img, (w, h), method=Image.LANCZOS, centering=(0.5, 0.5))


def parse(inp: dict) -> dict:
    prompt = (inp.get("prompt") or "").strip()
    if not prompt:
        raise BadInput("'prompt' is required")
    if len(prompt) > 4000:
        raise BadInput("'prompt' is longer than 4000 characters")
    img = _read_image(inp)
    w, h, two_stage = _target_size(inp, img)
    fps = _int(inp, "fps", 24, 12, 50)
    nf = _int(inp, "num_frames", 121, 9, 1001)
    if (nf - 1) % 8:
        raise BadInput("'num_frames' must be 8k+1 (e.g. 97, 121, 241)")
    if nf / fps > MAX_SECONDS:
        raise BadInput(f"clip is {nf / fps:.1f} s; the model supports up to {MAX_SECONDS:.0f} s (num_frames / fps)")
    seed = inp.get("seed")
    seed = random.randint(0, 2**31 - 1) if seed is None else _int(inp, "seed", 0, 0, 2**31 - 1)
    return {
        "image": _cover(img, w, h), "prompt": prompt, "width": w, "height": h, "two_stage": two_stage,
        "num_frames": nf, "fps": fps, "seed": seed, "audio": _bool(inp, "audio", True),
        "crf": _int(inp, "crf", 19, 10, 35),
    }


# ----------------------------------------------------------------------------------------------- output handling
def encode_mp4(frames, fps: int, crf: int, audio=None, sample_rate: int = 0) -> bytes:
    """H.264 video from uint8 frames [F,H,W,3]; plus AAC from float audio [channels, samples] when given."""
    n, h, w, _ = frames.shape
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.mp4")
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "-"]
        if audio is not None:
            raw = os.path.join(d, "audio.f32")
            audio.T.astype("<f4").tofile(raw)                      # interleaved little-endian float32
            cmd += ["-f", "f32le", "-ar", str(sample_rate), "-ac", str(audio.shape[0]), "-i", raw,
                    "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", out]
        p = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
        if p.returncode:
            raise RuntimeError(f"ffmpeg failed: {p.stderr.decode()[-800:]}")
        with open(out, "rb") as f:
            return f.read()


def _bucket_configured() -> bool:
    return all(os.getenv(k) for k in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY",
                                       "BUCKET_NAME"))


def upload(data: bytes, key: str) -> str:
    import boto3
    s3 = boto3.client("s3", endpoint_url=os.environ["BUCKET_ENDPOINT_URL"],
                      aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
                      region_name=os.getenv("BUCKET_REGION", "auto"))
    s3.put_object(Bucket=os.environ["BUCKET_NAME"], Key=key, Body=data, ContentType="video/mp4")
    return s3.generate_presigned_url("get_object", Params={"Bucket": os.environ["BUCKET_NAME"], "Key": key},
                                     ExpiresIn=int(os.getenv("BUCKET_URL_TTL_S", "604800")))


# ----------------------------------------------------------------------------------------------- actions
def _progress(job: dict, msg: str) -> None:
    log.info(msg)
    if os.getenv("RUNPOD_ENDPOINT_ID"):          # only inside a real RunPod worker (local tests have no job server)
        runpod.serverless.progress_update(job, msg)


def do_info() -> dict:
    meta = getattr(ENGINE, "meta", None)
    return {"ready": ENGINE is not None, "error": LOAD_ERROR, "mode": MODE, "model": "LTX-2.5 distilled (I2V + audio)",
            "memory_mode": MEMORY_MODE,
            "model_dir": MODEL_DIR, "gpu": getattr(ENGINE, "gpu", None), "cold_start_s": COLD_START_S,
            "prepared": meta, "inline_limit_mb": INLINE_LIMIT_MB, "bucket": _bucket_configured()}


def do_prepare(job: dict, inp: dict) -> dict:
    device = "cpu" if MODE == "mock" else "cuda"
    token = inp.get("hf_token") or os.getenv("HF_TOKEN") or None       # gated repo; never echoed back
    res = engine_mod.prepare(MODEL_DIR, WORK_DIR, device=device, force=bool(inp.get("force")),
                             progress=lambda m: _progress(job, m), token=token)
    # restart this worker so the next job loads the freshly prepared model at import time
    return {"refresh_worker": True, "job_results": res}


def do_generate(job: dict, inp: dict) -> dict:
    global _FIRST_JOB
    if ENGINE is None:
        return {"error": LOAD_ERROR or "engine not loaded"}
    t0 = time.time()
    try:
        p = parse(inp)
    except BadInput as e:
        return {"error": f"bad input: {e}"}
    t_parse = time.time() - t0

    frames, audio, sample_rate, timings = ENGINE.generate(
        p["image"], p["prompt"], p["width"], p["height"], num_frames=p["num_frames"], frame_rate=float(p["fps"]),
        seed=p["seed"], two_stage=p["two_stage"], progress=lambda m: _progress(job, m))

    t = time.time()
    video = encode_mp4(frames, p["fps"], p["crf"], audio if p["audio"] else None, sample_rate)
    timings.update({"input_s": round(t_parse, 2), "encode_mp4_s": round(time.time() - t, 2),
                    "handler_total_s": round(time.time() - t0, 2)})
    if _FIRST_JOB:
        timings["cold_start_s"] = COLD_START_S       # model load billed once per worker start
        _FIRST_JOB = False

    out = {"seed": p["seed"], "width": p["width"], "height": p["height"], "num_frames": p["num_frames"],
           "fps": p["fps"], "two_stage": p["two_stage"], "audio": p["audio"] and audio is not None,
           "gpu": ENGINE.gpu, "memory_mode": ENGINE.memory_mode, "video_bytes": len(video), "timings": timings}
    b64_mb = len(video) * 4 / 3 / 2**20
    if b64_mb <= INLINE_LIMIT_MB:
        out["video_base64"] = base64.b64encode(video).decode()
    elif _bucket_configured():
        out["video_url"] = upload(video, f"ltx25/{job.get('id', 'job')}.mp4")
    else:
        out["error"] = (f"video is {b64_mb:.1f} MB as base64 (> INLINE_LIMIT_MB={INLINE_LIMIT_MB}); raise 'crf', "
                        f"lower 'resolution', or configure a bucket (BUCKET_* env)")
    return out


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    action = inp.get("action", "generate")
    try:
        if action == "info":
            return do_info()
        if action == "prepare":
            return do_prepare(job, inp)
        if action == "generate":
            return do_generate(job, inp)
        return {"error": f"unknown action '{action}'"}
    except Exception as e:                       # report, don't hide: the client prints this
        log.error("job failed: %s\n%s", e, traceback.format_exc())
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
