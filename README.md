# wan22-runpod — LTX-2.5 image-to-video (with audio) on RunPod Serverless

Self-hosted image→video endpoint running **LTX-2.5** (Lightricks, Aug 2026; distilled checkpoint) as a scale-to-zero
RunPod serverless worker, plus a small CLI (`wanctl.py`). Each clip comes with a **synchronized audio track**
(48 kHz stereo, muxed into the mp4). The repo and CLI keep their original `wan22` names; the model is LTX-2.5.

**License:** LTX-2.x Community License — free for commercial use by organisations under $10M annual revenue (see
[the license](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md)). The weights are gated on Hugging Face.

## Status — what is verified and what is not

| Verified here (CPU, `make test`, 29 tests) | Not verified (needs your first GPU run) |
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

## Layout

```
wanctl.py                CLI: init | check | deploy | prepare | info | generate | health | costs | teardown
config.example.yaml      everything configurable; copy to config.yaml
worker/handler.py        RunPod entry: actions, input validation, mp4 + AAC encode, inline/S3 output
worker/ltx_engine.py     prepare() + LTXEngine (diffusers LTX2ImageToVideoPipeline, two-stage distilled)
worker/mock_engine.py    CPU stand-in for local tests
worker/Dockerfile        python:3.12-slim + torch 2.14.1 (CUDA 13.0) + pinned diffusers stack
prompts/                 the two Birthday Squad shots + negative prompt (unused by the distilled model)
tests/                   CPU test suite (make test)
```
