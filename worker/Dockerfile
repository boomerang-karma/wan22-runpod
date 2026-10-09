# RunPod serverless worker: LTX-2.5 image-to-video with synchronized audio (distilled).
# Weights are NOT baked in (they live on the network volume after `wanctl.py prepare`), so the image stays ~8-10 GB.
#
# torch comes from PyPI by default (currently CUDA 13.0 wheels -> needs hosts with CUDA >= 13.0 drivers; the
# endpoint's minCudaVersion in config.yaml must match). To target older drivers, build with e.g.
#   --build-arg TORCH_SPEC="torch==<version>" --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
# and set endpoint.min_cuda_version: "12.8".
FROM python:3.12-slim

ARG TORCH_SPEC="torch==2.14.1"
ARG TORCH_INDEX=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN if [ -n "$TORCH_INDEX" ]; then pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"; \
    else pip install "$TORCH_SPEC"; fi

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY handler.py ltx_engine.py mock_engine.py ./

# Fail the build early if the pinned stack cannot import the pieces the engine needs.
RUN python -c "from diffusers import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline; from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES; from transformers import Gemma4UnifiedForConditionalGeneration; import av, runpod; print('imports ok')"

CMD ["python", "-u", "handler.py"]
