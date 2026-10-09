"""Run the real engine code (load, LoRA fuse, encode, two-expert denoise, decode) on a tiny random Wan on CPU."""
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
sys.path.insert(0, os.path.dirname(__file__))

import wan_engine  # noqa: E402
from tiny_wan import LAYERS, build_tiny_model_dir, fake_lightning_lora, tiny_transformer  # noqa: E402


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return build_tiny_model_dir(str(tmp_path_factory.mktemp("tiny_wan")))


@pytest.mark.parametrize("style", ["down_up_alpha", "down_up", "A_B"])
def test_fuse_lightning_lora_changes_weights_and_removes_adapters(style):
    model = tiny_transformer(1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    stats = wan_engine.fuse_lightning_lora(model, fake_lightning_lora(style=style), strength=1.0)
    assert stats["lora_modules"] == 10 * LAYERS
    assert stats["probe_max_abs_delta"] > 0
    after = model.state_dict()
    assert set(after) == set(before), "fused model must have the original keys only (no lora_A/lora_B left)"
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert len(changed) == 10 * LAYERS                     # exactly the targeted linear weights moved
    assert not any(hasattr(m, "lora_A") for m in model.modules())


def test_fuse_rejects_lora_that_matches_nothing():
    bad = {k.replace("blocks.", "nonexistent_blocks."): v for k, v in fake_lightning_lora().items()}
    with pytest.raises(Exception):
        wan_engine.fuse_lightning_lora(tiny_transformer(1), bad, 1.0)


def test_engine_generates_uint8_frames_resident_mode_on_cpu(model_dir):
    eng = wan_engine.WanEngine(model_dir, device="cpu", memory_mode="resident", dtype=torch.float32)
    img = Image.fromarray((np.random.default_rng(0).random((48, 32, 3)) * 255).astype(np.uint8))
    steps_seen = []
    frames, timings = eng.generate(img, "a cartoon hero runs with cake", width=32, height=48, num_frames=9, steps=4,
                                   seed=7, progress=steps_seen.append)
    assert frames.dtype == np.uint8 and frames.shape == (9, 48, 32, 3)
    assert steps_seen == [f"denoise step {i}/4" for i in range(1, 5)]
    assert {"encode_prompt_s", "denoise_s", "vae_decode_s"} <= set(timings)
    # deterministic for a fixed seed (CPU generator) and different for another seed
    again, _ = eng.generate(img, "a cartoon hero runs with cake", width=32, height=48, num_frames=9, steps=4, seed=7)
    other, _ = eng.generate(img, "a cartoon hero runs with cake", width=32, height=48, num_frames=9, steps=4, seed=8)
    assert np.array_equal(frames, again) and not np.array_equal(frames, other)


def test_both_experts_are_used_with_4_steps(model_dir, monkeypatch):
    """boundary_ratio 0.9 + shift 5 + 4 steps must give the 2 high-noise / 2 low-noise split of the reference
    ComfyUI Lightning workflow."""
    eng = wan_engine.WanEngine(model_dir, device="cpu", memory_mode="resident", dtype=torch.float32)
    calls = {"high": 0, "low": 0}
    hi, lo = eng.pipe.transformer, eng.pipe.transformer_2
    orig_hi, orig_lo = hi.forward, lo.forward
    monkeypatch.setattr(hi, "forward", lambda *a, **k: (calls.__setitem__("high", calls["high"] + 1), orig_hi(*a, **k))[1])
    monkeypatch.setattr(lo, "forward", lambda *a, **k: (calls.__setitem__("low", calls["low"] + 1), orig_lo(*a, **k))[1])
    img = Image.new("RGB", (32, 48), (200, 120, 60))
    eng.generate(img, "photo comes to life", width=32, height=48, num_frames=5, steps=4, seed=1)
    assert calls == {"high": 2, "low": 2}, calls
