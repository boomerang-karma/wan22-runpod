# syntax=docker/dockerfile:1.7
# Meme Studio web app. Build from the repo root (the CodeBuild project in deploy/aws does this):
#   docker build -f portal/Dockerfile -t meme-studio .
# Only wanctl.py, config.example.yaml and portal/ are sent to the build (portal/Dockerfile.dockerignore), so
# config.yaml, .runpod_state.json and local galleries can never end up in the image.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Meme captions want a heavy condensed face like Impact (not redistributable); Anton (SIL OFL 1.1) stands in.
# Pinned to a google/fonts commit and checksummed.
ADD --checksum=sha256:a4ba3a92350ebb031da0cb47630ac49eb265082ca1bc0450442f4a83ab947cab --chmod=644 \
    https://raw.githubusercontent.com/google/fonts/e0a8124cf36bb7c32ca68e5d46d6acdbc3df866a/ofl/anton/Anton-Regular.ttf \
    /usr/share/fonts/truetype/anton/Anton-Regular.ttf

WORKDIR /app
COPY portal/requirements.txt portal/requirements.txt
RUN pip install --no-cache-dir -r portal/requirements.txt
COPY wanctl.py config.example.yaml ./
COPY portal/ portal/

# uid 1000 matches the EFS access point that holds /data
RUN useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin app && mkdir -p /data && chown app /data
USER 1000
ENV PORTAL_HOST=0.0.0.0 PORTAL_PORT=8000 PORTAL_DATA=/data PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "portal/server.py"]
