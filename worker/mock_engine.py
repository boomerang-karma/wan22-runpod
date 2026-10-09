"""CPU stand-in for WanEngine so the whole request path (decode, validate, encode, return) can be tested without a
GPU. It animates the input image with a slow push-in; it is not a model."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from PIL import Image

MARKER = "prepared.json"


def prepare(out_dir: str, work_dir: str, device: str = "cpu", progress=lambda m: None, force: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    meta = {"mock": True, "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_dir, MARKER), "w") as f:
        json.dump(meta, f)
    progress("mock weights prepared")
    return {"status": "prepared", **meta}


class MockEngine:
    memory_mode = "mock"

    def __init__(self, *_, **__):
        self.meta, self.gpu, self.load_seconds = {"mock": True}, "cpu (mock)", 0.0

    def generate(self, image: Image.Image, prompt: str, width: int, height: int, num_frames: int = 81, steps: int = 4,
                 guidance: float = 1.0, guidance_2=None, shift: float = 5.0, seed: int = 0, negative_prompt: str = "",
                 progress=lambda m: None):
        t = time.time()
        base = image.convert("RGB").resize((width, height), Image.BICUBIC)
        frames = []
        for i in range(num_frames):
            z = 1.0 + 0.08 * i / max(1, num_frames - 1)
            cw, ch = width / z, height / z
            x0, y0 = (width - cw) / 2, (height - ch) / 2
            frames.append(np.asarray(base.transform((width, height), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch),
                                                    Image.BICUBIC)))
        for s in range(steps):
            progress(f"denoise step {s + 1}/{steps}")
        return np.stack(frames).astype(np.uint8), {"denoise_and_decode_s": round(time.time() - t, 2)}
