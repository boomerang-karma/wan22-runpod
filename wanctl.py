#!/usr/bin/env python3
"""wanctl: deploy and drive the LTX-2.5 image-to-video (+ audio) RunPod serverless endpoint.

  python wanctl.py init                  create config.yaml from the example (chmod 600) — then add your API key
  python wanctl.py check                 validate config and that the API key works (read-only)
  python wanctl.py deploy [--dry-run]    create/update network volume, template, endpoint (idempotent)
  python wanctl.py prepare               one-time: download weights onto the volume (needs HF_TOKEN; ~20-40 min)
  python wanctl.py info                  ask a worker what it has loaded (spins up a worker: billed)
  python wanctl.py generate --image IMG --prompt-file P [--seed N] [--resolution 540p|720p|1080p] [--out F]
  python wanctl.py health                queue / worker counts (free)
  python wanctl.py costs                 sum estimated cost of jobs saved in outputs/
  python wanctl.py teardown [--volume] [--yes]

Only depends on `requests`, `pyyaml` and `pillow`. The API key is read from $RUNPOD_API_KEY or config.yaml and is
never printed.
"""
from __future__ import annotations

import argparse
import base64
import copy
import glob
import io
import json
import os
import shutil
import stat
import sys
import time
from typing import Any, Optional

import requests
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REST = "https://rest.runpod.io/v1"
SERVERLESS = "https://api.runpod.ai/v2"
STATE_FILE = os.path.join(HERE, ".runpod_state.json")
RESOLUTIONS = {"540p": (544, 960), "720p": (704, 1280), "1080p": (1088, 1920)}   # must match worker/handler.py
SECRET_ENV = ("BUCKET_SECRET_ACCESS_KEY", "BUCKET_ACCESS_KEY_ID", "HF_TOKEN")
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")


class CtlError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- config & state
def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise CtlError(f"{path} not found — run `python wanctl.py init` and fill it in")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    mode = os.stat(path).st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(f"warning: {path} is readable by other users; run `chmod 600 {path}`", file=sys.stderr)
    return cfg


def api_key(cfg: dict) -> str:
    key = os.getenv("RUNPOD_API_KEY") or (cfg.get("runpod") or {}).get("api_key") or ""
    if not key.strip():
        raise CtlError("no API key: set runpod.api_key in config.yaml or export RUNPOD_API_KEY")
    return key.strip()


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    os.chmod(STATE_FILE, 0o600)


def masked(d: dict) -> dict:
    out = copy.deepcopy(d)
    for k in list(out.get("env", {})):
        if k in SECRET_ENV and out["env"][k]:
            out["env"][k] = "***"
    return out


# ----------------------------------------------------------------------------------------------- HTTP
class RunPod:
    def __init__(self, key: str, dry_run: bool = False):
        self.dry_run = dry_run
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    def _call(self, method: str, url: str, body: Optional[dict] = None, ok404: bool = False) -> Any:
        if self.dry_run and method != "GET":
            print(f"[dry-run] {method} {url}\n{json.dumps(masked(body or {}), indent=2)}")
            return {"id": f"dry-run-{url.rsplit('/', 1)[-1]}"}
        for attempt in range(5):
            try:
                r = self.s.request(method, url, json=body, timeout=60)
            except requests.RequestException as e:
                if attempt == 4:
                    raise CtlError(f"{method} {url}: {e}")
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404 and ok404:
                return None
            if r.status_code == 401:
                raise CtlError("RunPod rejected the API key (401)")
            if not r.ok:
                raise CtlError(f"{method} {url} -> {r.status_code}: {r.text[:600]}")
            return r.json() if r.text.strip() else {}
        raise CtlError(f"{method} {url}: retries exhausted")

    def rest(self, method: str, path: str, body: Optional[dict] = None, ok404: bool = False) -> Any:
        return self._call(method, f"{REST}{path}", body, ok404)

    def sls(self, method: str, endpoint_id: str, path: str, body: Optional[dict] = None) -> Any:
        return self._call(method, f"{SERVERLESS}/{endpoint_id}{path}", body)


# ----------------------------------------------------------------------------------------------- payloads
def template_payload(cfg: dict) -> dict:
    env = {k: str(v) for k, v in (cfg.get("worker_env") or {}).items() if v not in (None, "")}
    env.update({"VOLUME_PATH": "/runpod-volume", "WORKER_MODE": "real"})
    body = {"name": f"{cfg['endpoint']['name']}-tpl", "imageName": cfg["image"]["name"], "isServerless": True,
            "containerDiskInGb": int(cfg["endpoint"].get("container_disk_gb", 100)), "volumeInGb": 0,
            "env": env, "ports": []}
    if cfg["image"].get("registry_auth_id"):
        body["containerRegistryAuthId"] = cfg["image"]["registry_auth_id"]
    return body


def endpoint_payload(cfg: dict, template_id: str, volume_id: str) -> dict:
    e = cfg["endpoint"]
    body = {"name": e["name"], "templateId": template_id, "computeType": "GPU", "gpuTypeIds": list(e["gpu_types"]),
            "gpuCount": 1, "networkVolumeId": volume_id, "dataCenterIds": [cfg["runpod"]["data_center"]],
            "workersMin": int(e.get("workers_min", 0)), "workersMax": int(e.get("workers_max", 1)),
            "idleTimeout": int(e.get("idle_timeout_s", 5)),
            "executionTimeoutMs": int(e.get("execution_timeout_s", 900)) * 1000,
            "flashboot": bool(e.get("flashboot", True)), "scalerType": "QUEUE_DELAY", "scalerValue": 4}
    if e.get("min_cuda_version"):
        body["minCudaVersion"] = str(e["min_cuda_version"])
    return body


def validate(cfg: dict, need_image: bool = True) -> None:
    problems = []
    if not (cfg.get("runpod") or {}).get("data_center"):
        problems.append("runpod.data_center is empty")
    if need_image and not (cfg.get("image") or {}).get("name"):
        problems.append("image.name is empty (build and push worker/ first)")
    if not (cfg.get("endpoint") or {}).get("gpu_types"):
        problems.append("endpoint.gpu_types is empty")
    if problems:
        raise CtlError("config.yaml: " + "; ".join(problems))


# ----------------------------------------------------------------------------------------------- commands
def cmd_init(args) -> None:
    dst = args.config
    if os.path.exists(dst):
        print(f"{dst} already exists; not overwriting")
        return
    shutil.copy(os.path.join(HERE, "config.example.yaml"), dst)
    os.chmod(dst, 0o600)
    print(f"created {dst} (mode 600). Fill in runpod.api_key, runpod.data_center and image.name.")


def cmd_check(args) -> None:
    cfg = load_config(args.config)
    rp = RunPod(api_key(cfg))
    eps = rp.rest("GET", "/endpoints")
    vols = rp.rest("GET", "/networkvolumes")
    print(f"API key OK — {len(eps or [])} endpoint(s), {len(vols or [])} network volume(s) on the account")
    try:
        validate(cfg)
        print("config OK for deploy")
    except CtlError as e:
        print(e)
    st = load_state()
    if st:
        print("state:", json.dumps(st, indent=2))


def cmd_deploy(args) -> None:
    cfg = load_config(args.config)
    validate(cfg)
    rp = RunPod(api_key(cfg), dry_run=args.dry_run)
    st = load_state()

    # 1. network volume
    vol_id = cfg["network_volume"].get("id") or st.get("volume_id")
    if vol_id and not args.dry_run:
        vol = rp.rest("GET", f"/networkvolumes/{vol_id}", ok404=True)
        if not vol:
            raise CtlError(f"network volume {vol_id} not found")
        if vol.get("dataCenterId") != cfg["runpod"]["data_center"]:
            raise CtlError(f"volume {vol_id} is in {vol.get('dataCenterId')}, config says {cfg['runpod']['data_center']}")
        print(f"volume: reusing {vol_id} ({vol.get('size')} GB, {vol.get('dataCenterId')})")
    if not vol_id:
        vol = rp.rest("POST", "/networkvolumes", {"name": cfg["network_volume"]["name"],
                                                  "size": int(cfg["network_volume"]["size_gb"]),
                                                  "dataCenterId": cfg["runpod"]["data_center"]})
        vol_id = vol["id"]
        print(f"volume: created {vol_id}")

    # 2. template (PATCH triggers a rolling release of the endpoint's workers)
    tpl = template_payload(cfg)
    tpl_id = st.get("template_id")
    if tpl_id:
        rp.rest("PATCH", f"/templates/{tpl_id}", {k: v for k, v in tpl.items() if k not in ("isServerless",)})
        print(f"template: updated {tpl_id} -> {tpl['imageName']}")
    else:
        tpl_id = rp.rest("POST", "/templates", tpl)["id"]
        print(f"template: created {tpl_id}")

    # 3. endpoint
    ep = endpoint_payload(cfg, tpl_id, vol_id)
    ep_id = st.get("endpoint_id")
    if ep_id:
        rp.rest("PATCH", f"/endpoints/{ep_id}", {k: v for k, v in ep.items() if k != "computeType"})
        print(f"endpoint: updated {ep_id}")
    else:
        ep_id = rp.rest("POST", "/endpoints", ep)["id"]
        print(f"endpoint: created {ep_id}")

    if not args.dry_run:
        st.update({"volume_id": vol_id, "template_id": tpl_id, "endpoint_id": ep_id,
                   "data_center": cfg["runpod"]["data_center"], "image": cfg["image"]["name"]})
        save_state(st)
        print(f"\nsaved {os.path.basename(STATE_FILE)}. Next: python wanctl.py prepare")


def _endpoint_id(cfg: dict) -> str:
    ep = load_state().get("endpoint_id") or (cfg.get("endpoint") or {}).get("id")
    if not ep:
        raise CtlError("no endpoint id — run `python wanctl.py deploy` first")
    return ep


def run_job(rp: RunPod, ep: str, payload: dict, poll_s: float, quiet: bool = False) -> dict:
    job = rp.sls("POST", ep, "/run", payload)
    jid = job["id"]
    print(f"job {jid} submitted")
    last, t0 = None, time.time()
    while True:
        st = rp.sls("GET", ep, f"/status/{jid}")
        status = st.get("status")
        note = st.get("output") if status == "IN_PROGRESS" and isinstance(st.get("output"), (str, dict)) else None
        line = f"{status}" + (f" — {note}" if note else "")
        if line != last and not quiet:
            print(f"  [{time.time() - t0:6.0f}s] {line}")
            last = line
        if status in TERMINAL:
            st["_wall_s"] = round(time.time() - t0, 1)
            return st
        time.sleep(poll_s)


def hf_token(cfg: dict) -> str:
    token = os.getenv("HF_TOKEN") or (cfg.get("huggingface") or {}).get("token") or ""
    if not token.strip():
        raise CtlError("no Hugging Face token. The LTX-2.5 weights are gated:\n"
                       "  1. sign in at https://huggingface.co/Lightricks/LTX-2.5-Diffusers and accept the license\n"
                       "  2. create a read token at https://huggingface.co/settings/tokens\n"
                       "  3. put it in config.yaml -> huggingface.token (or export HF_TOKEN=hf_...) and run prepare again")
    return token.strip()


def cmd_prepare(args) -> None:
    cfg = load_config(args.config)
    rp, ep = RunPod(api_key(cfg)), _endpoint_id(cfg)
    token = hf_token(cfg)
    print("Preparing weights on the network volume. This downloads ~71 GB from Hugging Face on a GPU worker\n"
          "(about 8 minutes of billed GPU time once, plus any wait for a free GPU). Safe to re-run (skips if prepared).")
    st = run_job(rp, ep, {"input": {"action": "prepare", "force": args.force, "hf_token": token},
                          "policy": {"executionTimeout": int(args.timeout_min) * 60 * 1000}}, poll_s=20)
    print(json.dumps(st.get("output") or st.get("error"), indent=2))
    if st.get("status") != "COMPLETED":
        raise CtlError(f"prepare ended with {st.get('status')}")


def cmd_info(args) -> None:
    cfg = load_config(args.config)
    rp, ep = RunPod(api_key(cfg)), _endpoint_id(cfg)
    st = run_job(rp, ep, {"input": {"action": "info"}}, poll_s=5)
    print(json.dumps(st.get("output") or st.get("error"), indent=2))


def cmd_health(args) -> None:
    cfg = load_config(args.config)
    print(json.dumps(RunPod(api_key(cfg)).sls("GET", _endpoint_id(cfg), "/health"), indent=2))


def target_size(w: int, h: int, resolution: str) -> tuple[int, int]:
    short, long = RESOLUTIONS[resolution]
    return (short, long) if h >= w else (long, short)


def encode_image(path: str, resolution: str) -> tuple[str, tuple[int, int]]:
    """Cover-crop to the exact generation size client-side so the request stays small (worker does the same)."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    size = target_size(img.width, img.height, resolution)
    img = ImageOps.fit(img, size, method=Image.LANCZOS, centering=(0.5, 0.5))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode(), size


def estimate_cost(cfg: dict, out: dict, exec_ms: Optional[float]) -> dict:
    gpu = out.get("gpu") or ""
    price = next((p for k, p in (cfg.get("pricing_per_hour") or {}).items() if k in gpu), None)
    cold = (out.get("timings") or {}).get("cold_start_s") or 0.0
    idle = float((cfg.get("endpoint") or {}).get("idle_timeout_s", 5))
    billed = (exec_ms or 0) / 1000.0 + idle + cold
    return {"gpu": gpu, "usd_per_hour": price, "billed_s_est": round(billed, 1), "cold_start_s": cold,
            "usd_est": round(billed * price / 3600.0, 4) if price else None,
            "note": "estimate = cold start (first job on a fresh worker) + execution + idle timeout"}


def cmd_generate(args) -> None:
    cfg = load_config(args.config)
    d = cfg.get("generate_defaults") or {}
    rp, ep = RunPod(api_key(cfg)), _endpoint_id(cfg)
    prompt = args.prompt or (open(args.prompt_file).read().strip() if args.prompt_file else "")
    if not prompt:
        raise CtlError("give --prompt or --prompt-file")
    resolution = args.resolution or d.get("resolution", "720p")
    img_b64, (w, h) = encode_image(args.image, resolution)
    inp = {"action": "generate", "image_base64": img_b64, "prompt": prompt, "resolution": resolution,
           "num_frames": args.num_frames or d.get("num_frames", 121), "fps": args.fps or d.get("fps", 24),
           "crf": args.crf or d.get("crf", 19), "audio": not args.no_audio}
    if args.seed is not None:
        inp["seed"] = args.seed
    timeout_ms = int(args.timeout_s or d.get("timeout_s", 900)) * 1000
    print(f"generating {w}x{h}, {inp['num_frames']} frames @ {inp['fps']} fps "
          f"({inp['num_frames'] / inp['fps']:.1f} s{', with audio' if inp['audio'] else ''}) from {os.path.basename(args.image)}")
    st = run_job(rp, ep, {"input": inp, "policy": {"executionTimeout": timeout_ms}}, poll_s=4)
    out = st.get("output") or {}
    if st.get("status") != "COMPLETED" or "error" in out:
        raise CtlError(f"job {st.get('id')} {st.get('status')}: {out.get('error') or st.get('error')}")

    os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)
    stem = args.out or os.path.join(cfg.get("output_dir", "outputs"), f"ltx_{time.strftime('%Y%m%d_%H%M%S')}_s{out['seed']}.mp4")
    if "video_base64" in out:
        with open(stem, "wb") as f:
            f.write(base64.b64decode(out.pop("video_base64")))
    elif "video_url" in out:
        with requests.get(out["video_url"], stream=True, timeout=120) as r, open(stem, "wb") as f:
            r.raise_for_status()
            shutil.copyfileobj(r.raw, f)
    else:
        raise CtlError("job returned no video")
    cost = estimate_cost(cfg, out, st.get("executionTime"))
    meta = {"job_id": st.get("id"), "delay_ms": st.get("delayTime"), "execution_ms": st.get("executionTime"),
            "wall_s": st.get("_wall_s"), "prompt": prompt, "image": os.path.abspath(args.image),
            "result": out, "cost": cost}
    with open(os.path.splitext(stem)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    t = out.get("timings", {})
    print(f"\nsaved {stem}\n  seed {out['seed']} | gpu {out.get('gpu')} | queue+start {st.get('delayTime', 0) / 1000:.1f}s | "
          f"execution {st.get('executionTime', 0) / 1000:.1f}s")
    if cost["usd_est"] is not None:
        print(f"  est. cost ${cost['usd_est']:.3f} ({cost['billed_s_est']}s @ ${cost['usd_per_hour']}/h"
              f"{', incl. cold start' if cost['cold_start_s'] else ''})")


def cmd_costs(args) -> None:
    cfg = load_config(args.config)
    rows = []
    for p in sorted(glob.glob(os.path.join(cfg.get("output_dir", "outputs"), "*.json"))):
        with open(p) as f:
            m = json.load(f)
        rows.append((os.path.basename(p), (m.get("cost") or {}).get("usd_est") or 0.0, (m.get("cost") or {}).get("billed_s_est")))
    for name, usd, s in rows:
        print(f"{name:48s} ${usd:7.3f}  {s}s")
    print(f"{'TOTAL (' + str(len(rows)) + ' jobs)':48s} ${sum(r[1] for r in rows):7.3f}")


def cmd_teardown(args) -> None:
    cfg = load_config(args.config)
    rp, st = RunPod(api_key(cfg)), load_state()
    what = [f"endpoint {st.get('endpoint_id')}", f"template {st.get('template_id')}"]
    if args.volume:
        what.append(f"network volume {st.get('volume_id')} (deletes the prepared ~72 GB model; re-prepare costs GPU time)")
    if not args.yes:
        print("will delete:\n  " + "\n  ".join(what))
        if input("type 'delete' to confirm: ").strip() != "delete":
            print("aborted")
            return
    if st.get("endpoint_id"):
        rp.rest("PATCH", f"/endpoints/{st['endpoint_id']}", {"workersMin": 0, "workersMax": 0})
        rp.rest("DELETE", f"/endpoints/{st['endpoint_id']}", ok404=True)
        print("endpoint deleted")
        st.pop("endpoint_id")
    if st.get("template_id"):
        rp.rest("DELETE", f"/templates/{st['template_id']}", ok404=True)
        print("template deleted")
        st.pop("template_id")
    if args.volume and st.get("volume_id"):
        rp.rest("DELETE", f"/networkvolumes/{st['volume_id']}", ok404=True)
        print("network volume deleted")
        st.pop("volume_id")
    save_state(st)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    p = sub.add_parser("deploy")
    p.add_argument("--dry-run", action="store_true", help="print the API payloads; create nothing")
    p.set_defaults(fn=cmd_deploy)
    p = sub.add_parser("prepare")
    p.add_argument("--force", action="store_true", help="rebuild even if already prepared")
    p.add_argument("--timeout-min", type=int, default=120)
    p.set_defaults(fn=cmd_prepare)
    sub.add_parser("info").set_defaults(fn=cmd_info)
    sub.add_parser("health").set_defaults(fn=cmd_health)
    p = sub.add_parser("generate")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt")
    p.add_argument("--prompt-file")
    p.add_argument("--seed", type=int)
    p.add_argument("--resolution", choices=sorted(RESOLUTIONS), help="540p = single-stage draft; 720p/1080p two-stage")
    p.add_argument("--num-frames", type=int, help="8k+1, e.g. 121 = 5 s at 24 fps; up to 20 s")
    p.add_argument("--fps", type=int)
    p.add_argument("--no-audio", action="store_true", help="return a silent mp4")
    p.add_argument("--crf", type=int)
    p.add_argument("--timeout-s", type=int)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_generate)
    sub.add_parser("costs").set_defaults(fn=cmd_costs)
    p = sub.add_parser("teardown")
    p.add_argument("--volume", action="store_true", help="also delete the network volume (prepared weights)")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_teardown)
    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except CtlError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
