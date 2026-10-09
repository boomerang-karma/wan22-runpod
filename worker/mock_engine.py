"""CPU stand-in for LTXEngine so the whole request path (decode, validate, mux, return) can be tested without a GPU.
It animates the input image with a slow push-in over a quiet two-tone audio track; it is not a model."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from PIL import Image

MARKER = "prepared.json"


def prepare(out_dir: str, work_dir: str, device: str = "cpu", progress=lambda m: None, force: bool = False,
            token=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    meta = {"mock": True, "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_dir, MARKER), "w") as f:
        json.dump(meta, f)
    progress("mock weights prepared")
    return {"status": "prepared", **meta}


class MockEngine:
    memory_mode = "mock"
    sample_rate = 48000

    def __init__(self, *_, **__):
        self.meta, self.gpu, self.load_seconds = {"mock": True}, "cpu (mock)", 0.0

    def generate(self, image: Image.Image, prompt: str, width: int, height: int, num_frames: int = 121,
                 frame_rate: float = 24.0, seed: int = 0, two_stage: bool = True, progress=lambda m: None):
        t = time.time()
        base = image.convert("RGB").resize((width, height), Image.BICUBIC)
        frames = []
        for i in range(num_frames):
            z = 1.0 + 0.08 * i / max(1, num_frames - 1)
            cw, ch = width / z, height / z
            x0, y0 = (width - cw) / 2, (height - ch) / 2
            frames.append(np.asarray(base.transform((width, height), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch),
                                                    Image.BICUBIC)))
        steps = 11 if two_stage else 8
        for s in range(steps):
            progress(f"denoise step {s + 1}/{steps}")
        n = int(round(num_frames / frame_rate * self.sample_rate))
        tt = np.arange(n) / self.sample_rate
        audio = 0.1 * np.stack([np.sin(2 * np.pi * 440 * tt), np.sin(2 * np.pi * 660 * tt)]).astype(np.float32)
        return (np.stack(frames).astype(np.uint8), audio, self.sample_rate,
                {"denoise_and_decode_s": round(time.time() - t, 2)})
