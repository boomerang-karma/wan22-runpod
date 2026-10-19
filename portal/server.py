"""Meme portal: a small web app in front of the LTX-2.5 RunPod endpoint.

Upload a photo, pick a meme preset (motion + sound), add top/bottom captions, get an mp4 with audio. The RunPod key
stays on this server; the browser only talks to /api/*. Captions are burned in here with ffmpeg after the GPU job, so
editing a caption later is free (no new GPU job).

  python portal/server.py                       real mode: uses config.yaml + .runpod_state.json like wanctl.py
  PORTAL_MOCK=1 python portal/server.py         no GPU: fakes the job with an ffmpeg zoom of the photo

Environment:
  PORTAL_PASSWORD           if set, visitors must enter it once (cookie). Set it before sharing the URL.
  PORTAL_DAILY_BUDGET_USD   stop accepting jobs once today's estimated spend reaches this (default 5)
  PORTAL_MAX_ACTIVE         max queued + running jobs at once (default 3)
  PORTAL_DATA               where uploads, videos and the gallery live (default ./portal_data)
  PORTAL_HOST / PORTAL_PORT bind address (default 127.0.0.1:8000)
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import io
import json
import logging
import os
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Optional

import requests
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import presets  # noqa: E402
import wanctl  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("portal")

MOCK = os.getenv("PORTAL_MOCK") == "1"
MOCK_DELAY = float(os.getenv("PORTAL_MOCK_DELAY", "1.5"))          # seconds per fake stage
DATA = os.path.abspath(os.getenv("PORTAL_DATA", os.path.join(ROOT, "portal_data")))
CONFIG = os.getenv("PORTAL_CONFIG", os.path.join(ROOT, "config.yaml"))
PASSWORD = os.getenv("PORTAL_PASSWORD", "")
SECRET = (os.getenv("PORTAL_SECRET") or secrets.token_hex(16)).encode()
BUDGET_USD = float(os.getenv("PORTAL_DAILY_BUDGET_USD", "5"))
MAX_ACTIVE = int(os.getenv("PORTAL_MAX_ACTIVE", "3"))
POLL_S = 3.0
MAX_UPLOAD_MB = 20

# Explicit sizes so every aspect works. Draft is single-stage (multiples of 32), HD two-stage (multiples of 64).
SIZES = {"draft": {"9:16": (544, 960), "16:9": (960, 544), "1:1": (736, 736)},
         "hd": {"9:16": (704, 1280), "16:9": (1280, 704), "1:1": (960, 960)}}
FRAMES = {3: 73, 5: 121, 8: 193}                                     # 8k+1 frames at 24 fps
FPS = 24
CRF = 23                                                            # keeps 8 s HD under the 8 MB inline limit
# Warm-worker estimate per second of clip (README cost table); the real cost is computed from billed time after.
EST_USD_PER_CLIP_S = {"draft": 0.006, "hd": 0.011}
COLD_START_USD = 0.25
ACTIVE = ("queued", "running")
FFMPEG = shutil.which("ffmpeg")
FONTS = ("/System/Library/Fonts/Supplemental/Impact.ttf", "/Library/Fonts/Impact.ttf",
         "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf", "C:/Windows/Fonts/impact.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")


# ----------------------------------------------------------------------------------------------- storage
class Store:
    """Jobs live in DATA/jobs/<id>/job.json next to their files; spend per day in DATA/spend.json."""

    def __init__(self, root: str):
        self.root = root
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        os.makedirs(os.path.join(root, "jobs"), exist_ok=True)
        for d in os.listdir(os.path.join(root, "jobs")):
            p = os.path.join(root, "jobs", d, "job.json")
            if os.path.exists(p):
                with open(p) as f:
                    self.jobs[d] = json.load(f)

    def dir(self, jid: str) -> str:
        return os.path.join(self.root, "jobs", jid)

    def save(self, job: dict) -> None:
        with self.lock:
            self.jobs[job["id"]] = job
            os.makedirs(self.dir(job["id"]), exist_ok=True)
            tmp = os.path.join(self.dir(job["id"]), "job.json.tmp")
            with open(tmp, "w") as f:
                json.dump(job, f, indent=2)
            os.replace(tmp, os.path.join(self.dir(job["id"]), "job.json"))

    def update(self, jid: str, **fields) -> dict:
        with self.lock:
            job = dict(self.jobs[jid], **fields)
            self.save(job)
            return job

    def get(self, jid: str) -> dict:
        with self.lock:
            if jid not in self.jobs:
                raise HTTPException(404, "no such meme")
            return dict(self.jobs[jid])

    def delete(self, jid: str) -> None:
        with self.lock:
            self.jobs.pop(jid, None)
            shutil.rmtree(self.dir(jid), ignore_errors=True)

    def active(self) -> int:
        with self.lock:
            return sum(j["status"] in ACTIVE for j in self.jobs.values())

    # spend ledger (estimated USD per local day)
    def spent_today(self) -> float:
        return self._ledger().get(datetime.date.today().isoformat(), 0.0)

    def add_spend(self, usd: float) -> None:
        with self.lock:
            led = self._ledger()
            day = datetime.date.today().isoformat()
            led[day] = round(led.get(day, 0.0) + usd, 4)
            with open(os.path.join(self.root, "spend.json"), "w") as f:
                json.dump(led, f, indent=2)

    def _ledger(self) -> dict:
        p = os.path.join(self.root, "spend.json")
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f)


# ----------------------------------------------------------------------------------------------- images & video
def pick_aspect(img: Image.Image, aspect: str) -> str:
    if aspect in ("9:16", "16:9", "1:1"):
        return aspect
    r = img.width / img.height
    return "1:1" if 0.85 <= r <= 1.18 else ("16:9" if r > 1 else "9:16")


def crop_to(src: str, dst: str, size: tuple[int, int]) -> None:
    with Image.open(src) as img:
        ImageOps.fit(img.convert("RGB"), size, method=Image.LANCZOS, centering=(0.5, 0.45)).save(dst, "JPEG", quality=95)


def _font(size: int):
    for path in FONTS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def caption_png(w: int, h: int, top: str, bottom: str) -> Image.Image:
    """Classic meme captions: uppercase, white, black outline, shrunk until each block fits in 3 lines."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for text, at_top in ((top, True), (bottom, False)):
        text = " ".join(text.upper().split())
        if not text:
            continue
        size = int(min(w, h) * 0.13)
        while True:
            font = _font(size)
            lines = _wrap(d, text, font, int(w * 0.92))
            line_h = int(size * 1.08)
            fits = all(d.textlength(ln, font=font) <= w * 0.92 for ln in lines)
            if (len(lines) <= 3 and fits and line_h * len(lines) <= h * 0.3) or size <= 14:
                break
            size = int(size * 0.9)
        stroke = max(2, size // 13)
        y = int(h * 0.035) if at_top else h - int(h * 0.045) - line_h * len(lines)
        for ln in lines:
            x = (w - d.textlength(ln, font=font)) / 2
            d.text((x, y), ln, font=font, fill="white", stroke_width=stroke, stroke_fill="black")
            y += line_h
    return im


def ffmpeg(*args: str) -> None:
    if not FFMPEG:
        raise RuntimeError("ffmpeg is not installed (brew install ffmpeg)")
    p = subprocess.run([FFMPEG, "-loglevel", "error", "-y", *args], capture_output=True)
    if p.returncode:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.decode()[-600:]}")


def burn_captions(d: str, w: int, h: int, top: str, bottom: str) -> str:
    """raw.mp4 -> meme.mp4 with captions; without captions the raw video is the meme. Returns the file name."""
    meme = os.path.join(d, "meme.mp4")
    if not (top.strip() or bottom.strip()):
        if os.path.exists(meme):
            os.remove(meme)
        return "raw.mp4"
    with tempfile.TemporaryDirectory() as t:
        png = os.path.join(t, "cap.png")
        caption_png(w, h, top, bottom).save(png)
        out = os.path.join(t, "meme.mp4")
        ffmpeg("-i", os.path.join(d, "raw.mp4"), "-i", png, "-filter_complex",
               f"[1:v]scale={w}:{h}[c];[0:v][c]overlay=0:0,format=yuv420p", "-map", "0:a?", "-c:a", "copy",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", out)
        shutil.move(out, meme)
    return "meme.mp4"


def mock_video(d: str, w: int, h: int, frames: int) -> None:
    """Stand-in for the GPU: slow zoom into the photo plus a tone, same size and length as the real job."""
    secs = frames / FPS
    ffmpeg("-i", os.path.join(d, "input.jpg"), "-f", "lavfi", "-i", f"sine=frequency=330:duration={secs}",
           "-filter_complex",
           f"[0:v]scale={w * 2}:{h * 2},zoompan=z='min(zoom+0.003,1.35)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
           f":d={frames}:s={w}x{h}:fps={FPS},format=yuv420p[v]",
           "-map", "[v]", "-map", "1:a", "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast",
           "-c:a", "aac", "-shortest", os.path.join(d, "raw.mp4"))


# ----------------------------------------------------------------------------------------------- RunPod side
class Backend:
    def __init__(self, store: Store):
        self.store = store
        self.cfg: dict = {}
        self.endpoint = ""
        self._health: tuple[float, dict] = (0.0, {})
        if not MOCK:
            self.cfg = wanctl.load_config(CONFIG)
            self.key = wanctl.api_key(self.cfg)
            self.endpoint = wanctl._endpoint_id(self.cfg)

    def rp(self) -> wanctl.RunPod:
        return wanctl.RunPod(self.key)

    def health(self) -> dict:
        if MOCK:
            busy = self.store.active() > 0
            return {"state": "busy" if busy else "warm", "label": "Mock GPU", "detail": "PORTAL_MOCK=1: no RunPod calls"}
        t, h = self._health
        if time.time() - t > 5:
            try:
                h = self.rp().sls("GET", self.endpoint, "/health")
            except Exception as e:                                    # keep the page alive if RunPod blips
                h = {"error": str(e)}
            self._health = (time.time(), h)
        return describe_health(h, self.cfg.get("runpod", {}).get("data_center", ""))

    # one thread per job
    def start(self, jid: str) -> None:
        threading.Thread(target=self._run, args=(jid,), daemon=True, name=f"job-{jid}").start()

    def _run(self, jid: str) -> None:
        try:
            (self._run_mock if MOCK else self._run_runpod)(jid)
        except Exception as e:
            log.exception("job %s failed", jid)
            self.store.update(jid, status="failed", error=str(e), finished=time.time())

    def _stage(self, jid: str, status: str, stage: str) -> bool:
        """Record progress; returns False when the user cancelled."""
        job = self.store.get(jid)
        if job.get("cancel"):
            return False
        if job["status"] != status or job.get("stage") != stage:
            self.store.update(jid, status=status, stage=stage, **({"started": time.time()} if status == "running"
                                                                   and not job.get("started") else {}))
        return True

    def _run_mock(self, jid: str) -> None:
        stages = [("queued", "Waking up a GPU…"), ("running", "Loading model"), ("running", "stage 1: 8 steps"),
                  ("running", "upsampling x2"), ("running", "stage 2: 3 steps"), ("running", "decoding video + audio")]
        for status, stage in stages:
            if not self._stage(jid, status, stage):
                self.store.update(jid, status="cancelled", stage="", finished=time.time())
                return
            time.sleep(MOCK_DELAY)
        job = self.store.get(jid)
        mock_video(self.store.dir(jid), job["width"], job["height"], job["num_frames"])
        self._finish(jid, {"gpu": "mock", "seed": job["seed"]}, cost=0.0, exec_ms=None, delay_ms=None)

    def _run_runpod(self, jid: str) -> None:
        rp, job = self.rp(), self.store.get(jid)
        if not job.get("runpod_id"):
            with open(os.path.join(self.store.dir(jid), "input.jpg"), "rb") as f:
                img = base64.b64encode(f.read()).decode()
            inp = {"action": "generate", "image_base64": img, "prompt": job["prompt"], "width": job["width"],
                   "height": job["height"], "two_stage": job["quality"] == "hd", "num_frames": job["num_frames"],
                   "fps": FPS, "crf": CRF, "audio": job["audio"], "seed": job["seed"]}
            rid = rp.sls("POST", self.endpoint, "/run", {"input": inp, "policy": {"executionTimeout": 900_000}})["id"]
            job = self.store.update(jid, runpod_id=rid)
            log.info("job %s -> runpod %s", jid, rid)
        rid = job["runpod_id"]
        cancelled = False
        while True:
            if self.store.get(jid).get("cancel") and not cancelled:
                rp.sls("POST", self.endpoint, f"/cancel/{rid}")
                cancelled = True
            try:
                st = rp.sls("GET", self.endpoint, f"/status/{rid}")
            except wanctl.CtlError as e:
                if "404" in str(e):
                    raise RuntimeError("RunPod no longer has this job (results expire after 30 min)")
                log.warning("status poll for %s failed: %s", jid, e)
                time.sleep(POLL_S * 3)
                continue
            status = st.get("status")
            if status == "IN_QUEUE":
                h = self.health()
                self._stage(jid, "queued", "Waking up a GPU (first meme after a break takes 2–5 min)"
                            if h["state"] in ("asleep", "waking") else h["label"] if h["state"] == "no_gpu"
                            else "In the queue")
            elif status == "IN_PROGRESS":
                note = st.get("output") if isinstance(st.get("output"), str) else ""
                self._stage(jid, "running", note or "Generating")
            elif status == "COMPLETED":
                out = st.get("output") or {}
                if out.get("error"):
                    raise RuntimeError(out["error"])
                d = self.store.dir(jid)
                if "video_base64" in out:
                    with open(os.path.join(d, "raw.mp4"), "wb") as f:
                        f.write(base64.b64decode(out.pop("video_base64")))
                elif "video_url" in out:
                    with requests.get(out["video_url"], stream=True, timeout=120) as r, \
                            open(os.path.join(d, "raw.mp4"), "wb") as f:
                        r.raise_for_status()
                        shutil.copyfileobj(r.raw, f)
                else:
                    raise RuntimeError("the worker returned no video")
                cost = wanctl.estimate_cost(self.cfg, out, st.get("executionTime"))
                self._finish(jid, out, cost=cost.get("usd_est") or 0.0, exec_ms=st.get("executionTime"),
                             delay_ms=st.get("delayTime"))
                return
            elif status in ("CANCELLED", "TIMED_OUT", "FAILED"):
                # a cancelled job may still have started a worker; count the execution time it reports
                cost = wanctl.estimate_cost(self.cfg, {"gpu": ""}, st.get("executionTime"))
                self.store.add_spend(cost.get("usd_est") or 0.0)
                self.store.update(jid, status="cancelled" if status == "CANCELLED" else "failed", stage="",
                                  error=None if status == "CANCELLED" else (st.get("error") or status),
                                  finished=time.time())
                return
            time.sleep(POLL_S)

    def _finish(self, jid: str, out: dict, cost: float, exec_ms, delay_ms) -> None:
        job = self.store.get(jid)
        self._stage(jid, "running", "adding captions")
        video = burn_captions(self.store.dir(jid), job["width"], job["height"], job["caption_top"], job["caption_bottom"])
        self.store.add_spend(cost)
        self.store.update(jid, status="completed", stage="", video=video, cost_usd=cost, gpu=out.get("gpu"),
                          execution_ms=exec_ms, delay_ms=delay_ms, finished=time.time(),
                          cold_start_s=(out.get("timings") or {}).get("cold_start_s"))
        log.info("job %s done (%s, $%.3f)", jid, out.get("gpu"), cost)

    def resume(self) -> None:
        """After a restart: keep polling RunPod jobs still in flight; local mock jobs cannot resume."""
        for job in list(self.store.jobs.values()):
            if job["status"] in ACTIVE:
                if job.get("runpod_id") and not MOCK:
                    self.start(job["id"])
                else:
                    self.store.update(job["id"], status="failed", error="the portal restarted before this job was sent")


def describe_health(h: dict, dc: str) -> dict:
    if "error" in h or not h:
        return {"state": "unknown", "label": "Can't reach RunPod", "detail": h.get("error", "")}
    w, j = h.get("workers") or {}, h.get("jobs") or {}
    queued = j.get("inQueue", 0)
    if w.get("running"):
        return {"state": "busy", "label": "GPU busy", "detail": f"{queued} waiting"}
    if w.get("idle") or w.get("ready"):
        return {"state": "warm", "label": "GPU warm", "detail": "next meme starts right away"}
    if w.get("initializing"):
        return {"state": "waking", "label": "GPU waking up", "detail": "loading the model, 2–5 min"}
    if w.get("throttled") and queued:
        return {"state": "no_gpu", "label": f"Waiting for a free GPU in {dc}",
                "detail": "all matching GPUs in the data center are taken; the job runs when one frees up"}
    return {"state": "asleep", "label": "GPU asleep", "detail": "first meme wakes it up (2–5 min), then it's fast"}


# ----------------------------------------------------------------------------------------------- app
store = Store(DATA)
backend = Backend(store)
app = FastAPI(title="Meme Studio", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
TOKEN = hmac.new(SECRET, b"meme-portal", hashlib.sha256).hexdigest()


def authed(request: Request) -> bool:
    return not PASSWORD or hmac.compare_digest(request.cookies.get("portal_auth", ""), TOKEN)


@app.middleware("http")
async def gate(request: Request, call_next):
    path = request.url.path
    if (path.startswith("/api/") or path.startswith("/media/")) and path not in ("/api/login", "/api/status") \
            and not authed(request):
        return JSONResponse({"detail": "enter the password first"}, status_code=401)
    response = await call_next(request)
    if path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"           # revalidate so UI updates show up on reload
    return response


def public(job: dict) -> dict:
    keys = ("id", "created", "status", "stage", "error", "preset", "preset_name", "emoji", "subject", "line",
            "caption_top", "caption_bottom", "quality", "aspect", "duration", "width", "height", "seed", "audio",
            "prompt", "started", "finished", "cost_usd", "gpu", "execution_ms", "delay_ms", "cold_start_s")
    out = {k: job.get(k) for k in keys}
    out["poster"] = f"/media/{job['id']}/input.jpg"
    if job["status"] == "completed":
        v = job.get("video", "raw.mp4")
        mtime = int(os.path.getmtime(os.path.join(store.dir(job["id"]), v))) if os.path.exists(
            os.path.join(store.dir(job["id"]), v)) else 0
        out["video_url"] = f"/media/{job['id']}/{v}?v={mtime}"
    return out


def estimate(quality: str, duration: int, gpu_state: str) -> float:
    usd = EST_USD_PER_CLIP_S[quality] * duration
    return round(usd + (COLD_START_USD if gpu_state in ("asleep", "waking", "no_gpu", "unknown") else 0), 3)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"), headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def status(request: Request):
    spent = store.spent_today()
    return {"mock": MOCK, "auth_required": bool(PASSWORD), "authed": authed(request),
            "budget_usd": BUDGET_USD, "spent_today_usd": round(spent, 3), "gpu": backend.health(),
            "active_jobs": store.active(), "max_active": MAX_ACTIVE, "captions": bool(FFMPEG),
            "est_usd_per_clip_s": EST_USD_PER_CLIP_S, "cold_start_usd": COLD_START_USD}


@app.post("/api/login")
def login(response: Response, password: str = Body(..., embed=True)):
    if not PASSWORD:
        return {"ok": True}
    if not hmac.compare_digest(password.encode(), PASSWORD.encode()):
        time.sleep(0.7)
        raise HTTPException(401, "wrong password")
    response.set_cookie("portal_auth", TOKEN, httponly=True, samesite="lax", max_age=30 * 86400)
    return {"ok": True}


@app.get("/api/presets")
def list_presets():
    return presets.public()


@app.get("/api/jobs")
def list_jobs():
    with store.lock:
        jobs = sorted(store.jobs.values(), key=lambda j: j["created"], reverse=True)
    return [public(j) for j in jobs]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return public(store.get(jid))


def _check_capacity(quality: str, duration: int) -> None:
    if store.active() >= MAX_ACTIVE:
        raise HTTPException(429, f"{MAX_ACTIVE} memes are already cooking — wait for one to finish")
    est = estimate(quality, duration, backend.health()["state"])
    if store.spent_today() + est > BUDGET_USD:
        raise HTTPException(402, f"today's budget (${BUDGET_USD:.2f}) is used up; "
                                 f"spent ${store.spent_today():.2f}. Raise PORTAL_DAILY_BUDGET_USD to continue.")


def _new_job(src_dir: str, *, preset: str, subject: str, line: str, custom: str, caption_top: str,
             caption_bottom: str, quality: str, aspect: str, duration: int, audio: bool, seed: Optional[int]) -> dict:
    if quality not in SIZES:
        raise HTTPException(400, "quality must be draft or hd")
    if duration not in FRAMES:
        raise HTTPException(400, f"duration must be one of {sorted(FRAMES)} seconds")
    try:
        prompt = presets.build_prompt(preset, subject=subject, line=line, custom=custom)
    except presets.PresetError as e:
        raise HTTPException(400, str(e))
    _check_capacity(quality, duration)
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = store.dir(jid)
    os.makedirs(d)
    shutil.copy(os.path.join(src_dir, "source.jpg"), os.path.join(d, "source.jpg"))
    with Image.open(os.path.join(d, "source.jpg")) as img:
        aspect = pick_aspect(img, aspect)
    w, h = SIZES[quality][aspect]
    crop_to(os.path.join(d, "source.jpg"), os.path.join(d, "input.jpg"), (w, h))
    p = presets.BY_ID[preset]
    job = {"id": jid, "created": time.time(), "status": "queued", "stage": "Sending to the GPU",
           "preset": preset, "preset_name": p["name"], "emoji": p["emoji"], "subject": subject, "line": line,
           "custom": custom, "caption_top": caption_top[:200], "caption_bottom": caption_bottom[:200],
           "quality": quality, "aspect": aspect, "duration": duration, "num_frames": FRAMES[duration],
           "width": w, "height": h, "audio": audio,
           "seed": seed if seed is not None else random.randint(0, 2**31 - 1), "prompt": prompt}
    store.save(job)
    backend.start(jid)
    return public(job)


@app.post("/api/generate")
async def generate(image: UploadFile = File(...), preset: str = Form(...), subject: str = Form(""),
                   line: str = Form(""), custom: str = Form(""), caption_top: str = Form(""),
                   caption_bottom: str = Form(""), quality: str = Form("draft"), aspect: str = Form("auto"),
                   duration: int = Form(5), audio: bool = Form(True), seed: Optional[int] = Form(None)):
    raw = await image.read(MAX_UPLOAD_MB * 2**20 + 1)
    if len(raw) > MAX_UPLOAD_MB * 2**20:
        raise HTTPException(413, f"image is larger than {MAX_UPLOAD_MB} MB")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "that file isn't an image I can read")
    if min(img.size) < 256:
        raise HTTPException(400, "image is too small (needs at least 256 px on the short side)")
    img.thumbnail((2048, 2048), Image.LANCZOS)
    with tempfile.TemporaryDirectory(dir=DATA) as t:
        img.save(os.path.join(t, "source.jpg"), "JPEG", quality=95)
        return _new_job(t, preset=preset, subject=subject, line=line, custom=custom, caption_top=caption_top,
                        caption_bottom=caption_bottom, quality=quality, aspect=aspect, duration=duration,
                        audio=audio, seed=seed)


@app.post("/api/jobs/{jid}/remix")
def remix(jid: str, body: dict = Body(default={})):
    """Same photo and settings; new seed unless keep_seed; optional overrides (e.g. quality=hd for the final)."""
    old = store.get(jid)
    fields = {k: old.get(k) for k in ("preset", "subject", "line", "custom", "caption_top", "caption_bottom",
                                       "quality", "aspect", "duration", "audio")}
    fields.update({k: v for k, v in body.items() if k in fields})
    seed = old["seed"] if body.get("keep_seed") else None
    return _new_job(store.dir(jid), seed=seed, **fields)


@app.post("/api/jobs/{jid}/captions")
def recaption(jid: str, caption_top: str = Body(""), caption_bottom: str = Body("")):
    """Re-burn captions on the finished video: no GPU, no cost."""
    job = store.get(jid)
    if job["status"] != "completed":
        raise HTTPException(409, "the meme isn't finished yet")
    video = burn_captions(store.dir(jid), job["width"], job["height"], caption_top[:200], caption_bottom[:200])
    return public(store.update(jid, caption_top=caption_top[:200], caption_bottom=caption_bottom[:200], video=video))


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    job = store.get(jid)
    if job["status"] not in ACTIVE:
        raise HTTPException(409, "not running")
    return public(store.update(jid, cancel=True, stage="Cancelling"))


@app.delete("/api/jobs/{jid}")
def delete(jid: str):
    if store.get(jid)["status"] in ACTIVE:
        raise HTTPException(409, "cancel it first")
    store.delete(jid)
    return {"ok": True}


@app.get("/media/{jid}/{name}")
def media(jid: str, name: str, dl: int = 0):
    if name not in ("input.jpg", "raw.mp4", "meme.mp4"):
        raise HTTPException(404)
    job = store.get(jid)
    path = os.path.join(store.dir(jid), name)
    if not os.path.exists(path):
        raise HTTPException(404)
    fname = f"meme-{job['preset']}-{job['seed']}{os.path.splitext(name)[1]}" if dl else None
    return FileResponse(path, filename=fname, headers={"Cache-Control": "private, max-age=3600"})


backend.resume()

if __name__ == "__main__":
    import uvicorn
    host, port = os.getenv("PORTAL_HOST", "127.0.0.1"), int(os.getenv("PORTAL_PORT", "8000"))
    if host not in ("127.0.0.1", "localhost") and not PASSWORD:
        log.warning("listening on %s without PORTAL_PASSWORD: anyone who can reach it can spend GPU money", host)
    log.info("Meme Studio on http://%s:%d (%s)", host, port, "MOCK mode" if MOCK else f"endpoint {backend.endpoint}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
