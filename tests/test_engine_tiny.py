"""Run the real engine code (load, prompt encode, distilled sigmas, two-stage upsample, video + audio decode) on a
tiny random LTX-2.x checkpoint on CPU."""
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
sys.path.insert(0, os.path.dirname(__file__))

import ltx_engine  # noqa: E402
from tiny_ltx import build_tiny_model_dir  # noqa: E402


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    model_dir = build_tiny_model_dir(str(tmp_path_factory.mktemp("tiny_ltx")))
    return ltx_engine.LTXEngine(model_dir, device="cpu", memory_mode="resident", vae_tiling=False,
                                dtype=torch.float32)


def _img():
    return Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8))


def test_two_stage_generates_full_size_frames_and_audio(engine):
    steps = []
    frames, audio, sr, timings = engine.generate(_img(), "a cartoon hero runs with cake", width=64, height=64,
                                                 num_frames=5, frame_rate=24.0, seed=7, progress=steps.append)
    assert frames.dtype == np.uint8 and frames.shape == (5, 64, 64, 3)
    assert audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 0 and audio.dtype == np.float32
    assert sr == 16000
    n = len(ltx_engine.DISTILLED_SIGMA_VALUES) + len(ltx_engine.STAGE_2_DISTILLED_SIGMA_VALUES)
    assert steps == [f"denoise step {i}/{n}" for i in range(1, n + 1)]          # 8 at half size + 3 at full size
    assert {"encode_prompt_s", "stage1_s", "upsample_s", "stage2_and_decode_s"} <= set(timings)


def test_single_stage_draft_and_seed_determinism(engine):
    args = dict(width=64, height=64, num_frames=5, two_stage=False)
    a, wav_a, _, timings = engine.generate(_img(), "balloons on a sunny patio", seed=3, **args)
    b, wav_b, _, _ = engine.generate(_img(), "balloons on a sunny patio", seed=3, **args)
    c, _, _, _ = engine.generate(_img(), "balloons on a sunny patio", seed=4, **args)
    assert a.shape == (5, 64, 64, 3) and "denoise_and_decode_s" in timings
    assert np.array_equal(a, b) and np.array_equal(wav_a, wav_b) and not np.array_equal(a, c)


def test_prompt_encode_retries_with_transformer_parked_after_oom(engine, monkeypatch):
    real, calls, moves = engine.pipe.encode_prompt, [], []
    real_to = engine.pipe.transformer.to

    def flaky(*a, **k):
        if not (a and a[0]) and not k.get("prompt"):          # the pipeline's own call with pre-encoded embeds
            return real(*a, **k)
        calls.append(1)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(*a, **k)

    monkeypatch.setattr(engine.pipe, "encode_prompt", flaky)
    monkeypatch.setattr(engine.pipe.transformer, "to", lambda d: (moves.append(str(d)), real_to(d))[1])
    frames, _, _, _ = engine.generate(_img(), "photo comes to life", width=64, height=64, num_frames=5, seed=1,
                                      two_stage=False)
    assert len(calls) == 2 and moves == ["cpu", "cpu"] and frames.shape == (5, 64, 64, 3)   # parked, then restored


def test_prompt_is_pre_encoded_so_text_encoder_can_stay_off_gpu(engine, monkeypatch):
    """Resident mode encodes before the pipeline call; the pipeline must not run the text encoder again."""
    calls = []
    orig = engine.pipe.text_encoder.forward
    monkeypatch.setattr(engine.pipe.text_encoder, "forward", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    engine.generate(_img(), "photo comes to life", width=64, height=64, num_frames=5, seed=1)
    assert len(calls) == 1
