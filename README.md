# wan22-runpod — LTX-2.5 image-to-video (with audio) on RunPod Serverless

Self-hosted image→video endpoint running **LTX-2.5** (Lightricks, Aug 2026; distilled checkpoint) as a scale-to-zero
RunPod serverless worker, plus a small CLI (`wanctl.py`). Each clip comes with a **synchronized audio track**
(48 kHz stereo, muxed into the mp4). The repo and CLI keep their original `wan22` names; the model is LTX-2.5.

**License:** LTX-2.x Community License — free for commercial use by organisations under $10M annual revenue (see
[the license](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md)). The weights are gated on Hugging Face.

## Status — what is verified and what is not

| Verified here (CPU, `make test`, 85 tests) | Not verified (needs your first GPU run) |
|---|---|
| Engine runs end to end on a tiny random LTX-2 (load → encode → 8-step stage 1 → x2 latent upsample → 3-step stage 2 → video + audio decode) | Real 22B weights on a real GPU: speed, peak VRAM, quality |
| Single-stage draft path, seed determinism, prompt pre-encoding, OOM fallback | (Checked against the real `model_index.json` and shard indexes, 2026-10-09) |
| `prepare()`: per-component download of indexed shards only, atomic save, idempotent re-run, clear error without HF access | Network-volume read speed → cold-start time |
| Handler through the RunPod SDK test runner (mock engine) → real H.264 + AAC mp4 | |
| Client payloads, dry-run deploy, key and token hygiene, cost maths | |

## How it works

```
your Mac ── wanctl.py ──REST──▶ RunPod: network volume (1 DC) + template (image, env) + endpoint (GPU pool)
                    └──/run──▶ worker (scale-to-zero)
                                 ├─ cold start: load bf16 model from /runpod-volume (~72 GB)
                                 └─ job: image+prompt → 8 steps @ half size → x2 upsample → 3 steps → video+audio → mp4
```

* **prepare (once):** downloads `Lightricks/LTX-2.5-Diffusers` (pinned revision) one component at a time through
  the container disk (only the shards each component's index uses; the repo ships some weights twice) and writes
  ~71 GB to the network volume. Needs a Hugging Face token (below); the token is sent in the prepare job only and never stored.
* **generate:** the distilled recipe from the model card: fixed sigma schedules, no guidance (one pass per step,
  no negative prompt). Transformer, VAEs, vocoder and upsampler stay on the GPU; the 12B Gemma text encoder waits in
  host RAM and visits the GPU only to encode the prompt.
* Output: 121 frames @ 24 fps (5 s) by default, up to 20 s. Sizes (portrait shown; landscape inputs flip them):

| `--resolution` | Size | Pipeline |
|---|---|---|
| `540p` | 544×960 | single-stage, 8 steps (cheap draft) |
| `720p` (default) | 704×1280 | two-stage |
| `1080p` | 1088×1920 | two-stage |

## Cost — estimates, not measurements

From FLOP counts for a ~19B-parameter transformer and RunPod's listed serverless rates (A100 80GB $2.72/h, H100
$4.79/h). Your first runs replace these: every `generate` prints billed seconds and estimated cost, and
`wanctl.py costs` sums them.

| Item | A100 80GB | H100 80GB |
|---|---|---|
| 540p 5 s draft, warm worker | ~20–40 s ≈ **$0.02–0.03** | ≈ $0.02–0.03 |
| 720p 5 s clip, warm worker | ~45–75 s ≈ **$0.03–0.06** | ≈ $0.03–0.05 |
| 1080p 5 s clip, warm worker | ~1.5–2.5 min ≈ **$0.07–0.11** | ≈ $0.05–0.09 |
| Cold start (load ~72 GB from the volume) | +2–5 min per fresh worker ≈ $0.10–0.25 | same time, ≈ $0.15–0.40 |
| `prepare`, once | measured 7.7 min ≈ $0.35 | ≈ $0.60 |
| Network volume 100 GB | $7 / month while it exists | |

Longer clips cost more than linearly (attention). Batch shots back to back so they share one warm worker.

## Setup

Prereqs: Python 3.10+, a RunPod account with credit, a free Hugging Face account, and somewhere to build the image
(GitHub Actions or Docker buildx).

```bash
pip install -r requirements-client.txt
python wanctl.py init                 # creates config.yaml (chmod 600, git-ignored)
# edit config.yaml: runpod.data_center, image.name   (API key: export RUNPOD_API_KEY=... instead)
python wanctl.py check                # validates the key (read-only)
```

**Hugging Face access (once):** sign in, open https://huggingface.co/Lightricks/LTX-2.5-Diffusers and click
*Agree and Access*; then create a **read** token at https://huggingface.co/settings/tokens and put it in
`config.yaml` → `huggingface.token` (or `export HF_TOKEN=...`).

**1. Build the worker image:** GitHub → Actions → *build-worker* → Run with a new tag (e.g. `0.2.0`) → put
`ghcr.io/<you>/wan22-runpod-worker:<tag>` in `image.name`. Or `make build` with Docker buildx.

**2. Deploy** (creates or updates the volume, template and endpoint; IDs go to `.runpod_state.json`):
```bash
python wanctl.py deploy --dry-run
python wanctl.py deploy
```

**3. Prepare the weights (once):**
```bash
python wanctl.py prepare              # ~8 min of GPU time; prints progress; safe to re-run
python wanctl.py info                 # worker reports GPU, cold-start time, what it loaded
```

**4. Generate:**
```bash
python wanctl.py generate --image path/to/photo.png \
  --prompt-file prompts/shot1_squad_arrives.txt --seed 7 --resolution 540p      # cheap draft
python wanctl.py generate --image path/to/photo.png \
  --prompt-file prompts/shot1_squad_arrives.txt --seed 7                        # 720p final
```
Each run writes `outputs/<name>.mp4` plus a `.json` with the prompt, seed, timings and estimated cost. A draft and
the final with the same seed are similar but not identical (different pipelines). Raise `endpoint.idle_timeout_s`
to ~60 while iterating so consecutive shots reuse the warm worker, then lower it again.

**Prompting:** LTX-2.5 was trained on long single-paragraph captions that describe the shot, motion, light **and
sound**. Add what should be heard (laughter, wind, a birthday song hummed, footsteps); short prompts degrade quality.

**5. Tear down** when finished:
```bash
python wanctl.py teardown             # endpoint + template; keeps the prepared volume ($7/month)
python wanctl.py teardown --volume    # also deletes the weights (re-prepare costs ~$1-3)
```

## Meme Studio (web portal)

A Higgsfield-style web app in front of the endpoint: drop a photo, pick an effect, add top/bottom text, get an mp4
with sound. The RunPod key stays on the server; the browser only talks to `/api/*`.

```bash
python portal/server.py                                  # http://127.0.0.1:8000, uses config.yaml + .runpod_state.json
PORTAL_MOCK=1 python portal/server.py                    # no GPU, no cost: fakes jobs with an ffmpeg zoom (UI work)
```

* **12 meme effects + Custom** (`portal/presets.py`): Dramatic Zoom, Vibing, Sitcom Laugh, Side Eye, Sad Violin,
  Explosion Behind, Make Them Say It (your line, lip-synced), Victory Dance, Mind Blown, Slow-mo Hero, Shook, Smolder.
  Each becomes a long LTX-2.5 prompt describing motion, camera **and sound**. "Who's in it?" fills in the subject.
* **Draft / HD** (540p single-stage / 720p two-stage), **3 / 5 / 8 s**, **Auto / 9:16 / 1:1 / 16:9**, sound on/off.
* **Captions** are burned in locally with ffmpeg after the GPU job (Impact, white with black outline), so editing
  them later is free. **New take** re-rolls the seed; **Make it HD** re-renders a draft at 720p with the same seed.
* The header shows the GPU state (asleep / waking / warm / busy / waiting for a free GPU) and today's estimated spend.
* Jobs, photos and videos live in `portal_data/` (git-ignored). In-flight jobs resume after a server restart.

**Who can get in** (first match wins):

| Mode | Set | Galleries and limits |
|---|---|---|
| Auth0 login | `AUTH0_DOMAIN`, `AUTH0_CLIENT_ID`, `AUTH0_CLIENT_SECRET`, `PORTAL_BASE_URL`, `SESSION_SECRET` | one per user, plus per-user budget and job limits |
| Shared password | `PORTAL_PASSWORD` | one shared gallery |
| Open | nothing | local use only |

| Env var | Default | |
|---|---|---|
| `PORTAL_ALLOWED_EMAILS` | — | with Auth0: optional guest list, e.g. `me@x.com,@mycompany.com` (verified emails only) |
| `PORTAL_DAILY_BUDGET_USD` | `5` | refuses new jobs once today's estimated spend (whole portal) would pass this |
| `PORTAL_USER_DAILY_BUDGET_USD` | the global one | the same per signed-in user |
| `PORTAL_MAX_ACTIVE` / `PORTAL_USER_MAX_ACTIVE` | `3` / the global one | max queued + running jobs, overall / per user |
| `RUNPOD_API_KEY` + `RUNPOD_ENDPOINT_ID` | — | use these instead of `config.yaml` / `.runpod_state.json` (containers) |
| `PORTAL_HOST` / `PORTAL_PORT` | `127.0.0.1` / `8000` | use `0.0.0.0` to serve your LAN |
| `PORTAL_DATA` | `./portal_data` | gallery storage |

To try Auth0 locally, add `http://localhost:8000/auth/callback` to the Auth0 app's callback URLs and run with
`PORTAL_BASE_URL=http://localhost:8000` plus the three `AUTH0_*` variables.

With `workers_max: 1`, memes run one at a time; raise `endpoint.idle_timeout_s` (e.g. 60) during a busy session so
back-to-back memes skip the 2–5 min cold start.

## Run the model on Modal instead of RunPod

`deploy/modal/ltx_app.py` runs the same worker code (`worker/handler.py`, `worker/ltx_engine.py`) on Modal. The
weights sit on a Modal Volume and GPUs come from Modal's whole pool, so no region is pinned. Modal tries the cheapest
80 GB card first: A100-80GB, then H100, RTX PRO 6000, H200, B200.

```bash
python -m modal setup                                   # once: sign in (writes ~/.modal.toml)
make modal-deploy                                       # create/update the app: no GPU time
make modal-prepare                                      # once: ~155 GB of weights onto the Volume (CPU, ~20-40 min)
make modal-info                                         # start a GPU container, report what it loaded (billed)
python -m modal run deploy/modal/ltx_app.py::generate --image photo.jpg --prompt "..."
```

With `backend: modal` in `config.yaml` (or `PORTAL_BACKEND=modal`), the portal sends memes to Modal. Progress shows up
live, and the cost of each meme is estimated from Modal's prices.

**Cost (Modal list prices, Oct 2026):**
* **Idle:** $0 once scaled to zero. The Volume fits Modal's free 1 TiB, and the Starter plan includes $30 of credits a month.
* **Running:** A100-80GB $0.000694/s, H100 $0.001097/s, plus about $0.0002/s for the 4 cores and 64 GiB.
* **After each job:** a container stays warm for 120 s (`SCALEDOWN_S`, billed).

| Item | Estimate |
|---|---|
| Draft meme on a warm container | ≈ $0.05 |
| Model load after a break (2–4 min) | ≈ $0.10–0.20 |
| One-time weights download (CPU) | ≈ $0.20–0.35 |

The Hugging Face token is sent only in the prepare call; it isn't stored on Modal.

## Deploy Meme Studio to AWS (with Auth0 login)

```
browser ──HTTPS──▶ CloudFront ──(secret header)──▶ ALB ──▶ Fargate task (ARM64, 1) ──▶ RunPod endpoint
                   *.cloudfront.net or your domain  only CloudFront   │  Auth0 login, captions (ffmpeg)
                                                                     └─ EFS: photos, videos, jobs, spend
CodeBuild builds portal/Dockerfile ──▶ ECR          Secrets Manager: RunPod key, Auth0 secret, session key
```

Everything is one CloudFormation stack (`deploy/aws/template.yaml`). `deploy/aws/deploy.sh` needs only the AWS
CLI: it uploads the portal source to S3 and CodeBuild builds the image, so no local Docker is needed.

**1. Auth0 (once):** in the Auth0 dashboard, *Applications → Create Application → Regular Web Applications*. From
its *Settings*, copy the Domain, Client ID and Client Secret. Optional: under *Authentication → Database* turn off
sign-ups and invite people, or use `ALLOWED_EMAILS` below.

**2. Settings:**
```bash
cp deploy/aws/deploy.env.example deploy/aws/deploy.env      # git-ignored
# fill in AUTH0_DOMAIN, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET (or export it), optionally ALLOWED_EMAILS
```
The RunPod endpoint and key are read from `config.yaml` / `.runpod_state.json`. The secrets go straight to
Secrets Manager and are never stack parameters.

**3. Deploy** (first run about 15 minutes, mostly CloudFront; later runs about 5):
```bash
deploy/aws/deploy.sh
```
It prints the app URL and the three Auth0 URLs to paste into the app's *Settings* (Allowed Callback URLs,
Allowed Logout URLs, Allowed Web Origins). Save them in Auth0, then sign in.

* **Update the app:** run `deploy/aws/deploy.sh` again (it deploys the working tree, committed or not). The
  single task restarts, so there's about a minute of downtime; memes in flight resume.
* **Change settings or rotate a secret:** edit `deploy.env`, then run `deploy/aws/deploy.sh --infra`.
* **Custom domain:** set `DOMAIN_NAME` and `CERTIFICATE_ARN` (ACM, us-east-1), deploy, then CNAME the domain to the
  `CloudFrontDomain` output and update the Auth0 URLs.
* **Logs:** `aws logs tail /ecs/meme-studio --follow`
* **Remove:** `deploy/aws/destroy.sh`. The gallery's EFS file system is kept; the script prints the delete command.

**AWS cost (us-east-1 list prices, always on; estimates):** ALB ≈ $17, public IPv4 addresses ≈ $11,
Fargate 0.5 vCPU / 1 GB ARM ≈ $15, EFS / Secrets Manager / logs / ECR ≈ $2. That's about **$45 a month** plus
CloudFront traffic beyond the free tier, and RunPod GPU time for the memes (capped by the daily budgets).

## Knobs

| Where | Setting | Effect |
|---|---|---|
| `generate` | `--resolution 540p\|720p\|1080p` | draft / default / high quality |
| `generate` | `--num-frames 241` | 10 s at 24 fps; must be 8k+1, max 20 s |
| `generate` | `--no-audio` | silent mp4 |
| `generate` | `--fps 25` | frame rate the model generates for (12–50) |
| `worker_env` | `MEMORY_MODE: offload` | lets 48 GB GPUs work if the host has ~80 GB RAM; slower |
| `worker_env` | `VAE_TILING: "0"` | slightly faster decode at 540p/720p; keep "1" for 1080p |
| `worker_env` | `LTX_REVISION` | pin a different Hugging Face commit (then `prepare --force`) |
| `worker_env` | `BUCKET_*` | upload videos > 8 MB (base64) to S3/R2 and return a link instead of inline |

## Troubleshooting

* **`prepare` says Hugging Face refused access:** accept the license on the model page with the same account that
  created the token, then `export HF_TOKEN=...` and re-run.
* **Jobs sit IN_QUEUE / no workers start:** no listed GPU free in your data center. Add GPU types or move to a DC
  with availability (a network volume cannot move; create a new one there and re-prepare).
* **CUDA out of memory:** use an H200 (in the GPU list) or set `MEMORY_MODE: offload`. Prompt encoding already
  retries with the transformer parked in host RAM.
* **Worker fails to start with a CUDA error:** host driver older than the image's CUDA. Keep
  `endpoint.min_cuda_version` matched to the torch build (see `worker/Dockerfile`).
* **Faces drift:** expected with any image-to-video model; keep faces still and small in frame, or cut away.

## Security

* `config.yaml` and `.runpod_state.json` are git-ignored and created with mode 600. `RUNPOD_API_KEY` and `HF_TOKEN`
  in the environment are preferred over the file.
* The RunPod key is sent only to `rest.runpod.io` and `api.runpod.ai`; the HF token only inside the prepare job.
  Neither is printed or written to the volume. `--dry-run` masks bucket secrets.
* Input images are sent to RunPod as base64 in the job payload. Async job results stay retrievable for 30 minutes.
* Meme Studio on AWS: the RunPod key, Auth0 client secret and session key live in Secrets Manager and reach the task
  as environment variables. The load balancer only accepts CloudFront (prefix list plus a secret header), the task only
  the load balancer, and EFS only the task. Sessions are signed, `httponly`, `Secure` and `SameSite=Lax` cookies
  holding only the verified ID-token claims. The image is built from an allowlist (`portal/Dockerfile.dockerignore`),
  so `config.yaml` can't end up in it.

## Layout

```
portal/server.py         Meme Studio web app (FastAPI): jobs, captions, budgets, login, per-user galleries
portal/auth.py           Auth0 login (OIDC authorization code flow, ID-token verification, guest list)
portal/presets.py        meme effects -> LTX-2.5 prompts
portal/static/           the single-page UI (vanilla JS)
portal/Dockerfile        the portal image (ffmpeg + Anton caption font); built by CodeBuild
deploy/aws/              CloudFormation stack + deploy.sh / destroy.sh for AWS
deploy/modal/ltx_app.py  the same worker on Modal: GPU class, weights Volume, prepare / info / generate
wanctl.py                CLI: init | check | deploy | prepare | info | generate | health | costs | teardown
config.example.yaml      everything configurable; copy to config.yaml
worker/handler.py        RunPod entry: actions, input validation, mp4 + AAC encode, inline/S3 output
worker/ltx_engine.py     prepare() + LTXEngine (diffusers LTX2ImageToVideoPipeline, two-stage distilled)
worker/mock_engine.py    CPU stand-in for local tests
worker/Dockerfile        python:3.12-slim + torch 2.14.1 (CUDA 13.0) + pinned diffusers stack
prompts/                 the two Birthday Squad shots + negative prompt (unused by the distilled model)
tests/                   CPU test suite (make test)
```
