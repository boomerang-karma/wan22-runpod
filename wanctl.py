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
# Data centers where RunPod serverless workers may run (POST /endpoints enum, Oct 2026). Network volumes exist in more
# places (e.g. US-NE-1, US-CO-1) but an endpoint cannot use those.
SERVERLESS_DCS = {"AP-IN-1", "AP-JP-1", "CA-MTL-1", "CA-MTL-2", "CA-MTL-3", "EU-CZ-1", "EU-FR-1", "EU-NL-1", "EU-RO-1",
                  "EU-SE-1", "EUR-IS-1", "EUR-IS-2", "EUR-IS-3", "EUR-NO-1", "OC-AU-1", "US-CA-2", "US-DE-1", "US-GA-1",
                  "US-GA-2", "US-IL-1", "US-KS-2", "US-KS-3", "US-MD-1", "US-MO-1", "US-MO-2", "US-NC-1", "US-TX-1",
                  "US-TX-3", "US-TX-4", "US-WA-1"}
# Cards used only for the one-time download in each region (no model is loaded; any small GPU works).
PREPARE_GPUS = ["NVIDIA RTX A5000", "NVIDIA RTX A4000", "NVIDIA L4", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000",
                "NVIDIA A40", "NVIDIA L40S", "NVIDIA GeForce RTX 3090", "NVIDIA RTX 4000 Ada Generation"]


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


def data_centers(cfg: dict) -> list[str]:
    """runpod.data_centers (list, first = primary) or the older single runpod.data_center."""
    r = cfg.get("runpod") or {}
    dcs = list(r.get("data_centers") or ([r["data_center"]] if r.get("data_center") else []))
    return list(dict.fromkeys(dcs))


def endpoint_payload(cfg: dict, template_id: str, volumes: dict[str, str]) -> dict:
    """`volumes` maps data center -> network volume id; workers may start in any of these data centers."""
    e = cfg["endpoint"]
    dcs = list(volumes)
    body = {"name": e["name"], "templateId": template_id, "computeType": "GPU", "gpuTypeIds": list(e["gpu_types"]),
            "gpuCount": 1, "networkVolumeId": volumes[dcs[0]] if dcs else None, "dataCenterIds": dcs,
            "workersMin": int(e.get("workers_min", 0)), "workersMax": int(e.get("workers_max", 1)),
            "idleTimeout": int(e.get("idle_timeout_s", 5)),
            "executionTimeoutMs": int(e.get("execution_timeout_s", 900)) * 1000,
            "flashboot": bool(e.get("flashboot", True)), "scalerType": "QUEUE_DELAY", "scalerValue": 4}
    # always send the list: a PATCH that only sets networkVolumeId leaves an older multi-volume list in place
    body["networkVolumeIds"] = [volumes[dc] for dc in dcs]
    if e.get("min_cuda_version"):
        body["minCudaVersion"] = str(e["min_cuda_version"])
    return body


# ----------------------------------------------------------------------------------------------- model cache
# RunPod model caching: the endpoint's Model field makes RunPod keep the Hugging Face repo on its hosts and mount it
# at /runpod-volume/huggingface-cache/hub. No network volume, so workers may start in any serverless data center.
# RunPod's API has no field for it yet: the Model field is set once in the console (deploy prints how).
CACHE_REPO = "Lightricks/LTX-2.5-Diffusers"


def cache_cfg(cfg: dict) -> dict:
    return cfg.get("model_cache") or {}


def cache_enabled(cfg: dict) -> bool:
    return bool(cache_cfg(cfg).get("enabled"))


def cache_endpoint_name(cfg: dict) -> str:
    return cache_cfg(cfg).get("endpoint_name") or f"{cfg['endpoint']['name']}-cached"


def cache_template_payload(cfg: dict) -> dict:
    body = template_payload(cfg)
    body["name"] = f"{cache_endpoint_name(cfg)}-tpl"
    body["containerDiskInGb"] = int(cache_cfg(cfg).get("container_disk_gb", 40))     # no prepare: nothing to stage
    body["env"].update({"MODEL_SOURCE": "cache", "LTX_REPO": CACHE_REPO, "HF_HUB_OFFLINE": "1"})
    return body


def cache_endpoint_payload(cfg: dict, template_id: str) -> dict:
    body = endpoint_payload(cfg, template_id, {})
    body.update({"name": cache_endpoint_name(cfg), "dataCenterIds": sorted(SERVERLESS_DCS), "networkVolumeIds": []})
    body.pop("networkVolumeId")
    return body


def prepare_endpoint_payload(cfg: dict, template_id: str, dc: str, volume_id: str) -> dict:
    """Temporary single-region endpoint used only to fill one volume. Downloading needs no big GPU, so it takes
    whatever small card is free (cheaper, and much easier to get than an 80 GB one)."""
    body = endpoint_payload(cfg, template_id, {dc: volume_id})
    body.update({"name": f"{cfg['endpoint']['name']}-prep-{dc}", "workersMax": 1, "idleTimeout": 5,
                 "gpuTypeIds": list(cfg["endpoint"].get("prepare_gpu_types") or PREPARE_GPUS),
                 "executionTimeoutMs": 2 * 3600 * 1000, "flashboot": False})
    return body


def validate(cfg: dict, need_image: bool = True) -> None:
    problems = []
    if not data_centers(cfg):
        problems.append("runpod.data_centers is empty")
    bad = [dc for dc in data_centers(cfg) if dc not in SERVERLESS_DCS]
    if bad:
        problems.append(f"{', '.join(bad)} cannot run serverless workers (allowed: {', '.join(sorted(SERVERLESS_DCS))})")
    if need_image and not (cfg.get("image") or {}).get("name"):
        problems.append("image.name is empty (build and push worker/ first)")
    if not (cfg.get("endpoint") or {}).get("gpu_types"):
        problems.append("endpoint.gpu_types is empty")
    if problems:
        raise CtlError("config.yaml: " + "; ".join(problems))


def _volumes_state(st: dict) -> dict:
    """Migrate the single-volume state ({volume_id, data_center}) to {volumes: {dc: id}, prepared: [dc]}."""
    if "volumes" not in st:
        st["volumes"] = {st["data_center"]: st["volume_id"]} if st.get("volume_id") and st.get("data_center") else {}
        st["prepared"] = list(st["volumes"])         # the old single-region flow served (and prepared) only this one
    st.pop("volume_id", None)
    return st


def serving_volumes(cfg: dict, st: dict) -> dict:
    """Data centers the main endpoint may use: only prepared ones, so no worker starts next to an empty volume.
    Before anything is prepared (first deploy), the primary data center alone."""
    vols = st.get("volumes") or {}
    # once any region holds the full model, serve only such regions so --quality full never lands without it
    done = st.get("prepared_full") or st.get("prepared") or []
    ready = {dc: vols[dc] for dc in data_centers(cfg) if dc in vols and dc in done}
    if ready:
        return ready
    first = data_centers(cfg)[0]
    return {first: vols[first]} if first in vols else {}


# ----------------------------------------------------------------------------------------------- commands
def cmd_init(args) -> None:
    dst = args.config
    if os.path.exists(dst):
        print(f"{dst} already exists; not overwriting")
        return
    shutil.copy(os.path.join(HERE, "config.example.yaml"), dst)
    os.chmod(dst, 0o600)
    print(f"created {dst} (mode 600). Fill in runpod.api_key, runpod.data_centers and image.name.")


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
    st = _volumes_state(load_state())

    # 1. one network volume per data center (a volume, and the model on it, can only serve its own data center)
    for dc in data_centers(cfg):
        vol_id = st["volumes"].get(dc) or (cfg["network_volume"].get("id") if dc == data_centers(cfg)[0] else None)
        if vol_id and not args.dry_run:
            vol = rp.rest("GET", f"/networkvolumes/{vol_id}", ok404=True)
            if not vol:
                raise CtlError(f"network volume {vol_id} ({dc}) not found")
            if vol.get("dataCenterId") != dc:
                raise CtlError(f"volume {vol_id} is in {vol.get('dataCenterId')}, expected {dc}")
            want = int(cfg["network_volume"]["size_gb"])
            if int(vol.get("size") or 0) < want:                     # volumes can grow, never shrink
                rp.rest("PATCH", f"/networkvolumes/{vol_id}", {"size": want})
                print(f"volume {dc}: grew {vol_id} {vol.get('size')} -> {want} GB")
            print(f"volume {dc}: reusing {vol_id} ({max(want, int(vol.get('size') or 0))} GB)"
                  + ("" if dc in st.get("prepared", []) else " — not prepared yet"))
        if not vol_id:
            vol = rp.rest("POST", "/networkvolumes", {"name": f"{cfg['network_volume']['name']}-{dc.lower()}",
                                                      "size": int(cfg["network_volume"]["size_gb"]),
                                                      "dataCenterId": dc})
            vol_id = vol["id"]
            print(f"volume {dc}: created {vol_id} — run `python wanctl.py prepare` to fill it")
        st["volumes"][dc] = vol_id

    # 2. template (PATCH triggers a rolling release of the endpoint's workers)
    tpl = template_payload(cfg)
    tpl_id = st.get("template_id")
    if tpl_id:
        rp.rest("PATCH", f"/templates/{tpl_id}", {k: v for k, v in tpl.items() if k not in ("isServerless",)})
        print(f"template: updated {tpl_id} -> {tpl['imageName']}")
    else:
        tpl_id = rp.rest("POST", "/templates", tpl)["id"]
        print(f"template: created {tpl_id}")
    st["template_id"] = tpl_id

    # 3. endpoint, limited to data centers whose volume holds the model
    ep_id = _update_endpoint(rp, cfg, st)
    # 4. optional: a second endpoint that loads the model from RunPod's model cache (any data center)
    if cache_enabled(cfg):
        _deploy_cache(rp, cfg, st)
    if not args.dry_run:
        st.update({"endpoint_id": ep_id, "image": cfg["image"]["name"]})
        save_state(st)
        todo = [dc for dc in data_centers(cfg) if dc not in st.get("prepared", [])]
        print(f"\nsaved {os.path.basename(STATE_FILE)}." + (f" Next: python wanctl.py prepare  ({', '.join(todo)})"
                                                             if todo else ""))


def _update_endpoint(rp: RunPod, cfg: dict, st: dict) -> str:
    ep = endpoint_payload(cfg, st["template_id"], serving_volumes(cfg, st))
    ep_id = st.get("endpoint_id")
    if ep_id:
        rp.rest("PATCH", f"/endpoints/{ep_id}", {k: v for k, v in ep.items() if k != "computeType"})
        print(f"endpoint: updated {ep_id} (data centers: {', '.join(ep['dataCenterIds'])})")
    else:
        ep_id = rp.rest("POST", "/endpoints", ep)["id"]
        print(f"endpoint: created {ep_id} (data centers: {', '.join(ep['dataCenterIds'])})")
    return ep_id


def _deploy_cache(rp: RunPod, cfg: dict, st: dict) -> None:
    tpl = cache_template_payload(cfg)
    if st.get("cache_template_id"):
        rp.rest("PATCH", f"/templates/{st['cache_template_id']}", {k: v for k, v in tpl.items() if k != "isServerless"})
        print(f"cache template: updated {st['cache_template_id']}")
    else:
        st["cache_template_id"] = rp.rest("POST", "/templates", tpl)["id"]
        print(f"cache template: created {st['cache_template_id']}")
    ep = cache_endpoint_payload(cfg, st["cache_template_id"])
    if st.get("cache_endpoint_id"):
        rp.rest("PATCH", f"/endpoints/{st['cache_endpoint_id']}", {k: v for k, v in ep.items() if k != "computeType"})
        print(f"cache endpoint: updated {st['cache_endpoint_id']} (any of {len(ep['dataCenterIds'])} data centers)")
    else:
        st["cache_endpoint_id"] = rp.rest("POST", "/endpoints", ep)["id"]
        print(f"cache endpoint: created {st['cache_endpoint_id']} (any of {len(ep['dataCenterIds'])} data centers)")
    if not st.get("cache_verified"):
        print(f"""
  One manual step (RunPod has no API for it yet): console -> Serverless -> {ep['name']} -> Manage -> Edit Endpoint
    Model: {CACHE_REPO}    and your Hugging Face token (the repo is gated). Save.
  Then run `python wanctl.py info`. Once a worker reports the cached model, jobs move to this endpoint;
  until then they keep using the volume endpoint.""")


def serving(cfg: dict) -> str:
    """'cache' once the cache endpoint has been verified by `info`, else 'volume'."""
    st = load_state()
    return "cache" if cache_enabled(cfg) and st.get("cache_endpoint_id") and st.get("cache_verified") else "volume"


def _endpoint_id(cfg: dict) -> str:
    if serving(cfg) == "cache":
        return load_state()["cache_endpoint_id"]
    ep = load_state().get("endpoint_id") or (cfg.get("endpoint") or {}).get("id")
    if not ep:
        raise CtlError("no endpoint id — run `python wanctl.py deploy` first")
    return ep


def run_job(rp: RunPod, ep: str, payload: dict, poll_s: float, quiet: bool = False, label: str = "") -> dict:
    job = rp.sls("POST", ep, "/run", payload)
    jid = job["id"]
    print(f"{label}job {jid} submitted", flush=True)
    last, t0 = None, time.time()
    while True:
        st = rp.sls("GET", ep, f"/status/{jid}")
        status = st.get("status")
        note = st.get("output") if status == "IN_PROGRESS" and isinstance(st.get("output"), (str, dict)) else None
        line = f"{status}" + (f" — {note}" if note else "")
        if line != last and not quiet:
            print(f"{label}  [{time.time() - t0:6.0f}s] {line}", flush=True)
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


def _prepare_one(rp: RunPod, cfg: dict, st: dict, dc: str, token: str, args) -> dict:
    """Fill one region's volume through a temporary endpoint pinned to it, then delete that endpoint."""
    label = f"[{dc}] "
    ep_id = rp.rest("POST", "/endpoints", prepare_endpoint_payload(cfg, st["template_id"], dc, st["volumes"][dc]))["id"]
    print(f"{label}temporary prepare endpoint {ep_id}", flush=True)
    try:
        return run_job(rp, ep_id, {"input": {"action": "prepare", "force": args.force, "hf_token": token,
                                             "full": bool(args.full)},
                                   "policy": {"executionTimeout": int(args.timeout_min) * 60 * 1000}},
                       poll_s=20, label=label)
    finally:
        rp.rest("PATCH", f"/endpoints/{ep_id}", {"workersMin": 0, "workersMax": 0})
        rp.rest("DELETE", f"/endpoints/{ep_id}", ok404=True)
        print(f"{label}temporary endpoint deleted", flush=True)


def cmd_prepare(args) -> None:
    import concurrent.futures
    cfg = load_config(args.config)
    rp, st = RunPod(api_key(cfg)), _volumes_state(load_state())
    token = hf_token(cfg)
    if not st.get("template_id") or not st["volumes"]:
        raise CtlError("nothing deployed yet — run `python wanctl.py deploy` first")
    done = st.get("prepared_full", []) if args.full else st.get("prepared", [])
    todo = [dc for dc in data_centers(cfg) if dc in st["volumes"] and (args.force or dc not in done)]
    if args.region:
        todo = [dc for dc in todo if dc in args.region]
    if not todo:
        print("every region is already prepared: " + ", ".join(st.get("prepared", [])))
        return
    print(f"Preparing {', '.join(todo)}: each downloads ~71 GB from Hugging Face onto its region's volume using a\n"
          "small GPU (about 8-15 minutes and ~$0.10 per region once, plus any wait for a free GPU).", flush=True)
    failed = []
    with concurrent.futures.ThreadPoolExecutor(len(todo)) as pool:
        jobs = {pool.submit(_prepare_one, rp, cfg, st, dc, token, args): dc for dc in todo}
        for fut in concurrent.futures.as_completed(jobs):
            dc = jobs[fut]
            try:
                res = fut.result()
            except CtlError as e:
                res = {"status": "ERROR", "error": str(e)}
            out = res.get("output") or {}
            ok = res.get("status") == "COMPLETED" and (out.get("job_results") or out).get("status") in ("prepared", "already_prepared")
            print(f"[{dc}] {'prepared' if ok else 'FAILED: ' + str(out.get('error') or res.get('error') or res.get('status'))}",
                  flush=True)
            if ok:
                st["prepared"] = sorted(set(st.get("prepared", [])) | {dc})
                if args.full:
                    st["prepared_full"] = sorted(set(st.get("prepared_full", [])) | {dc})
                save_state(st)
            else:
                failed.append(dc)
    _update_endpoint(rp, cfg, st)                    # start serving from the newly prepared regions
    save_state(st)
    if failed:
        raise CtlError(f"prepare failed in {', '.join(failed)}; the endpoint keeps using {', '.join(st.get('prepared', []))}")


def cmd_info(args) -> None:
    cfg = load_config(args.config)
    state = load_state()
    rp, ep = RunPod(api_key(cfg)), _endpoint_id(cfg)
    checking_cache = cache_enabled(cfg) and state.get("cache_endpoint_id") and not state.get("cache_verified")
    if checking_cache:
        ep = state["cache_endpoint_id"]
        print(f"checking the model-cache endpoint {ep} (jobs still go to the volume endpoint until it passes)")
    st = run_job(rp, ep, {"input": {"action": "info"}}, poll_s=5)
    out = st.get("output") or st.get("error")
    print(json.dumps(out, indent=2))
    if checking_cache:
        if isinstance(out, dict) and out.get("ready") and out.get("model_source") == "cache":
            state["cache_verified"] = True
            save_state(state)
            print(f"\nmodel cache verified on {out.get('gpu')}: jobs now go to {ep} (any data center)")
        else:
            print("\nthe cache endpoint isn't serving yet; jobs stay on the volume endpoint")


def cmd_health(args) -> None:
    cfg = load_config(args.config)
    print(json.dumps(RunPod(api_key(cfg)).sls("GET", _endpoint_id(cfg), "/health"), indent=2))


def target_size(w: int, h: int, resolution: str) -> tuple[int, int]:
    short, long = RESOLUTIONS[resolution]
    return (short, long) if h >= w else (long, short)


def parse_size(text: str, two_stage: bool) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise CtlError(f"--size must look like 736x672, got {text!r}")
    mult = 64 if two_stage else 32
    if w % mult or h % mult or not (256 <= w <= 1920 and 256 <= h <= 1920):
        raise CtlError(f"--size sides must be 256-1920 and multiples of {mult}"
                       + ("" if not two_stage else " (or add --single-stage for multiples of 32)"))
    return w, h


def encode_image(path: str, resolution: str, size: Optional[tuple[int, int]] = None) -> tuple[str, tuple[int, int]]:
    """Cover-crop to the exact generation size client-side so the request stays small (worker does the same)."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    size = size or target_size(img.width, img.height, resolution)
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
    size = parse_size(args.size, not args.single_stage) if args.size else None
    img_b64, (w, h) = encode_image(args.image, resolution, size)
    inp = {"action": "generate", "image_base64": img_b64, "prompt": prompt,
           **({"width": w, "height": h, "two_stage": not args.single_stage} if size else {"resolution": resolution}),
           "num_frames": args.num_frames or d.get("num_frames", 121), "fps": args.fps or d.get("fps", 24),
           "crf": args.crf or d.get("crf", 19), "audio": not args.no_audio}
    if args.quality == "full":
        inp.update({"quality": "full", "steps": args.steps or d.get("full_steps", 30)})
        neg = args.negative or (open(args.negative_file).read().strip() if args.negative_file else "")
        if neg:
            inp["negative_prompt"] = neg
    if args.mid_strength:
        inp["mid_strength"] = args.mid_strength
    if args.loop:
        inp["loop"] = True
    elif args.end_image:
        inp["end_image_base64"] = encode_image(args.end_image, resolution, (w, h))[0]
    if args.seed is not None:
        inp["seed"] = args.seed
    timeout_ms = int(args.timeout_s or d.get("timeout_s", 900)) * 1000
    print(f"generating {w}x{h}, {inp['num_frames']} frames @ {inp['fps']} fps "
          f"({inp['num_frames'] / inp['fps']:.1f} s{', with audio' if inp['audio'] else ''}) from {os.path.basename(args.image)}")
    st = run_job(rp, ep, {"input": inp, "policy": {"executionTimeout": timeout_ms}}, poll_s=4)
    out = st.get("output") or {}
    out = out.get("job_results", out) if isinstance(out, dict) else out     # worker asked RunPod to restart it
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
    rp, st = RunPod(api_key(cfg)), _volumes_state(load_state())
    what = [f"endpoint {st.get('endpoint_id')}", f"template {st.get('template_id')}"]
    if st.get("cache_endpoint_id"):
        what.append(f"model-cache endpoint {st['cache_endpoint_id']} + template {st.get('cache_template_id')}")
    if args.volume:
        what.append(f"network volumes {st.get('volumes')} (deletes the prepared ~71 GB model in each region)")
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
    if st.get("cache_endpoint_id"):
        rp.rest("PATCH", f"/endpoints/{st['cache_endpoint_id']}", {"workersMin": 0, "workersMax": 0})
        rp.rest("DELETE", f"/endpoints/{st['cache_endpoint_id']}", ok404=True)
        print("cache endpoint deleted")
        st.pop("cache_endpoint_id")
        st.pop("cache_verified", None)
    if st.get("cache_template_id"):
        rp.rest("DELETE", f"/templates/{st['cache_template_id']}", ok404=True)
        print("cache template deleted")
        st.pop("cache_template_id")
    if args.volume:
        for dc, vol in list(st.get("volumes", {}).items()):
            rp.rest("DELETE", f"/networkvolumes/{vol}", ok404=True)
            print(f"network volume {dc} deleted")
            st["volumes"].pop(dc)
        st["prepared"] = []
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
    p.add_argument("--region", action="append", help="only prepare this data center (repeatable)")
    p.add_argument("--full", action="store_true", help="also download the full model (~38 GB) for --quality full")
    p.set_defaults(fn=cmd_prepare)
    sub.add_parser("info").set_defaults(fn=cmd_info)
    sub.add_parser("health").set_defaults(fn=cmd_health)
    p = sub.add_parser("generate")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt")
    p.add_argument("--prompt-file")
    p.add_argument("--seed", type=int)
    p.add_argument("--resolution", choices=sorted(RESOLUTIONS), help="540p = single-stage draft; 720p/1080p two-stage")
    p.add_argument("--size", help="exact WxH instead of --resolution, e.g. 1280x1152 for a near-square photo")
    p.add_argument("--single-stage", action="store_true", help="with --size: one pass at full size (cheaper draft)")
    p.add_argument("--num-frames", type=int, help="8k+1, e.g. 121 = 5 s at 24 fps; up to 20 s")
    p.add_argument("--fps", type=int)
    p.add_argument("--no-audio", action="store_true", help="return a silent mp4")
    p.add_argument("--loop", action="store_true", help="pin the input image as the LAST frame too (returns to it)")
    p.add_argument("--quality", choices=("fast", "full"), default="fast",
                   help="full = full LTX-2.5 model for the motion pass (guided, ~30 steps; ~5x the GPU time)")
    p.add_argument("--steps", type=int, help="steps for --quality full (10-60, default 30)")
    p.add_argument("--negative", help="negative prompt for --quality full")
    p.add_argument("--negative-file")
    p.add_argument("--mid-strength", type=float, help="also pin the input image at the middle frame (0-1)")
    p.add_argument("--end-image", help="pin this image as the last frame (cover-cropped to the same size)")
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
