# wan22-runpod — Wan 2.2 image-to-video on RunPod Serverless

Self-hosted image→video endpoint: **Wan 2.2 I2V A14B** (Apache-2.0) with the **Lightning 4-step** distill LoRA
(lightx2v, Apache-2.0), packaged as a scale-to-zero RunPod serverless worker plus a small CLI (`wanctl.py`).

## Status — what is verified and what is not

| Verified here (CPU, `make test`, 27 tests) | Not verified (needs your first GPU run) |
|---|---|
| Engine code runs end to end on a tiny random Wan 2.2 (load → encode → 2-expert denoise → decode) | Real 14B weights on a real GPU: speed, peak VRAM, quality |
| 4 steps split 2 high-noise / 2 low-noise, as in the reference ComfyUI Lightning workflow | Docker image build (no Docker daemon in the build environment) |
| `prepare()` orchestration: per-component download, bf16 cast, LoRA fuse, atomic save, idempotent re-run | Network-volume read speed → cold-start time |
| LoRA key mapping onto all 400 target linears of the **full-size** A14B expert (meta device) | The exact Lightning LoRA file format (prepare fails loudly if it does not match) |
| Handler request path through the RunPod SDK test runner (mock engine) → real H.264 mp4 | RunPod REST calls against the live API (payloads follow the published schema) |
| Client payloads, dry-run deploy, key masking, cost maths | |

The first `prepare` + one `generate` on RunPod is the integration test. Everything that can fail there reports
a specific error rather than producing a silently bad model.

## How it works

```
your Mac ── wanctl.py ──REST──▶ RunPod: network volume (1 DC) + template (image, env) + endpoint (GPU pool)
                    └──/run──▶ worker (scale-to-zero)
                                 ├─ cold start: load fused bf16 model from /runpod-volume (~69 GB)
                                 └─ job: image+prompt → 4 Euler steps → VAE decode → H.264 mp4 → base64 / S3 URL
```

* **prepare (once):** downloads `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (126 GB, fp32 experts) one component at a time
  through the container disk, casts to bf16, **fuses** the Lightning LoRA into each expert, and writes ~69 GB to the
  network volume. Fusing once means no LoRA work on any cold start or request.
* **generate:** both experts + VAE stay on the GPU ("resident", 80 GB cards); the 5.7B text encoder waits in host
  RAM and visits the GPU only to encode the prompt. CFG = 1 (Lightning is CFG-distilled → one pass per step).
* Output: 81 frames @ 16 fps (5.06 s), 720×1280 portrait by default (480×832 for cheap drafts).

## Cost — estimates, not measurements

From FLOP counts (≈6.8 PFLOP per 720p forward pass) and RunPod's listed serverless rates. Your first runs replace
these numbers: every `generate` prints its billed seconds and estimated cost, and `wanctl.py costs` sums them.

| Item | A100 80GB ($2.72/h) | H100 80GB ($4.79/h) |
|---|---|---|
| 720p 5 s clip, warm worker | ~2.5–3.5 min ≈ **$0.12–0.16** | ~1.3–2 min ≈ **$0.10–0.16** |
| 480p 5 s clip, warm worker | ≈ $0.03–0.05 | ≈ $0.03–0.05 |
| Cold start (load ~69 GB from the volume) | +1–4 min per fresh worker | same |
| `prepare`, once | ~30 min ≈ $1.4 | ≈ $2.4 |
| Network volume 100 GB | $7 / month while it exists | |

Reference point: Higgsfield Kling 3.0 std ≈ 6.25 credits per 5 s ≈ $0.18–0.47 depending on plan. Self-hosting is
cheaper per clip only when the worker is warm (batch your shots) and you keep using it; quality is open-model level.

## Setup

Prereqs: Python 3.10+, a RunPod account with credit, and somewhere to build the image (GitHub Actions or Docker
buildx).

```bash
pip install -r requirements-client.txt
python wanctl.py init                 # creates config.yaml (chmod 600, git-ignored)
# edit config.yaml: runpod.api_key, runpod.data_center, image.name
python wanctl.py check                # validates the key (read-only)
```

**1. Build the worker image** (either):
* push this repo to GitHub → Actions → *build-worker* → Run (pushes `ghcr.io/<you>/wan22-runpod-worker:<tag>`), then
  make the package public or add a RunPod registry auth (`image.registry_auth_id`); or
* `make build` with Docker buildx (cross-builds linux/amd64 on Apple Silicon; slow but works).

**2. Deploy** — creates the network volume, template and endpoint, and records their IDs in `.runpod_state.json`:
```bash
python wanctl.py deploy --dry-run     # see the exact payloads first
python wanctl.py deploy
```

**3. Prepare the weights (once):**
```bash
python wanctl.py prepare              # ~20-40 min; prints progress; safe to re-run
python wanctl.py info                 # worker reports GPU, cold-start time, what it loaded
```

**4. Generate:**
```bash
python wanctl.py generate --image path/to/photo.png \
  --prompt-file prompts/shot1_squad_arrives.txt --seed 7 --resolution 480p      # cheap draft
python wanctl.py generate --image path/to/photo.png \
  --prompt-file prompts/shot1_squad_arrives.txt --seed 7                        # final at 720p, same seed
```
Each run writes `outputs/<name>.mp4` plus a `.json` with the prompt, seed, timings, and estimated cost. Submit several
shots back to back so they share one warm worker (idle timeout is 5 s by default; raise `endpoint.idle_timeout_s`
to ~60 while iterating, then lower it again).

**5. Tear down** when finished:
```bash
python wanctl.py teardown             # endpoint + template; keeps the prepared volume ($7/month)
python wanctl.py teardown --volume    # also deletes the weights (re-prepare costs ~$1.5-2.5)
```

## Knobs

| Where | Setting | Effect |
|---|---|---|
| `generate` | `--resolution 480p` | ~4× cheaper drafts; use the same seed for the 720p final |
| `generate` | `--flow-shift 8` | try if motion looks mushy (reference workflow uses 5) |
| `generate` | `--guidance 3.5` + `--negative-file` | enables a negative-prompt pass; 2× slower; Lightning is tuned for 1.0 |
| `generate` | `--num-frames 49` | shorter clip (must be 4k+1); 81 is the trained length |
| `worker_env` | `MEMORY_MODE: offload` | lets 48 GB GPUs (L40S/A6000) work if the host has ~75 GB RAM; slower |
| `worker_env` | `VAE_TILING: "1"` | lower decode VRAM |
| `worker_env` | `LORA_STRENGTH_*` | baked in at prepare time → re-run `prepare --force` after changing |
| `worker_env` | `BUCKET_*` | upload videos > 8 MB (base64) to S3/R2 and return a link instead of inline |

## Troubleshooting

* **Jobs sit IN_QUEUE / no workers start:** no listed GPU free in your data center. Add GPU types or move to a DC
  with availability (a network volume cannot move; create a new one there and re-prepare).
* **Worker fails to start with a CUDA error:** host driver older than the image's CUDA. Keep
  `endpoint.min_cuda_version` matched to the torch build (see `worker/Dockerfile`).
* **CUDA out of memory:** use H100/A100 80 GB, set `VAE_TILING: "1"`, or `MEMORY_MODE: offload`.
* **`prepare` says the LoRA matched too few modules:** the LoRA file layout changed upstream; pin `LORA_REVISION` /
  `WAN_REVISION` in `worker_env` to known commits.
* **Faces drift:** expected with any image-to-video model over 5 s. Keep faces still and small in frame, or cut away.

## Security

* `config.yaml` and `.runpod_state.json` are git-ignored and created with mode 600. `RUNPOD_API_KEY` in the
  environment overrides the file — prefer it in CI.
* The key is sent only to `rest.runpod.io` and `api.runpod.ai`, and it is never printed. `--dry-run` masks bucket secrets.
* Bucket credentials go into the RunPod template env, which RunPod stores. Use a bucket-scoped key.
* Input images are sent to RunPod as base64 in the job payload. Async job results stay retrievable for 30 minutes.

## Layout

```
wanctl.py                CLI: init | check | deploy | prepare | info | generate | health | costs | teardown
config.example.yaml      everything configurable; copy to config.yaml
worker/handler.py        RunPod entry: actions, input validation, mp4 encode, inline/S3 output
worker/wan_engine.py     prepare() + WanEngine (diffusers)
worker/mock_engine.py    CPU stand-in for local tests
worker/Dockerfile        python:3.12-slim + torch 2.14.1 (CUDA 13.0) + pinned diffusers stack
prompts/                 the two Birthday Squad shots + negative prompt
tests/                   CPU test suite (make test)
```
