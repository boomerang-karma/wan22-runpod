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
    engine.generate(_img(), "a prompt seen for the first time", width=64, height=64, num_frames=5, seed=1)
    assert len(calls) == 1


def test_repeated_prompt_reuses_cached_embeddings(engine, monkeypatch):
    calls = []
    orig = engine.pipe.text_encoder.forward
    monkeypatch.setattr(engine.pipe.text_encoder, "forward", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    args = dict(width=64, height=64, num_frames=5, two_stage=False)
    a, _, _, _ = engine.generate(_img(), "cache me", seed=5, **args)
    b, _, _, t = engine.generate(_img(), "cache me", seed=5, **args)
    assert len(calls) == 1 and np.array_equal(a, b)                 # second run: no encoder pass, same result


def test_text_encoder_swaps_by_pointer_and_returns_to_host(engine):
    te = engine.pipe.text_encoder
    p = next(te.parameters())
    host = p.data
    engine._parked.up()
    engine._parked.down()
    assert p.data.data_ptr() == host.data_ptr()                      # back on the same host buffer, no re-allocation


@pytest.mark.parametrize("two_stage", [False, True])
def test_end_frame_conditions_the_last_frame(engine, two_stage):
    start = Image.new("RGB", (64, 64), (230, 40, 40))
    end = Image.new("RGB", (64, 64), (40, 40, 230))
    frames, audio, _, _ = engine.generate(start, "statues come alive", width=64, height=64, num_frames=9, seed=2,
                                          two_stage=two_stage, end_image=end)
    assert frames.shape == (9, 64, 64, 3) and audio.shape[0] == 2
    plain, _, _, _ = engine.generate(start, "statues come alive", width=64, height=64, num_frames=9, seed=2,
                                     two_stage=two_stage)
    assert not np.array_equal(frames, plain)                         # the end condition actually changed the video


def test_full_quality_swaps_transformers_and_runs_guided_steps(engine):
    distilled = engine.pipe.transformer
    steps = []
    frames, audio, _, timings = engine.generate(_img(), "statues come alive", width=64, height=64, num_frames=9,
                                                seed=4, quality="full", steps=4, negative_prompt="blurry",
                                                end_image=_img(), progress=steps.append)
    assert frames.shape == (9, 64, 64, 3) and audio.shape[0] == 2
    assert steps[-1] == "denoise step 7/7"                           # 4 guided full steps + 3 distilled refine
    assert engine.pipe.transformer is distilled and engine.cond_pipe.transformer is distilled   # swapped back
    assert engine._full is not None and engine._full is not distilled and "swap_s" in timings
    fast, _, _, _ = engine.generate(_img(), "statues come alive", width=64, height=64, num_frames=9, seed=4,
                                    end_image=_img())
    assert not np.array_equal(frames, fast)


def test_mid_keyframe_changes_the_clip(engine):
    args = dict(width=64, height=64, num_frames=17, seed=6, two_stage=False)
    plain, _, _, _ = engine.generate(_img(), "hold still", **args)
    held, _, _, _ = engine.generate(_img(), "hold still", mid_strength=0.6, **args)
    assert held.shape == plain.shape and not np.array_equal(held, plain)


def test_full_quality_without_weights_is_a_clear_error(engine, monkeypatch):
    monkeypatch.setattr(engine, "has_full", False)
    with pytest.raises(ValueError, match="prepare --full"):
        engine.generate(_img(), "x", width=64, height=64, num_frames=9, quality="full")


def _as_hf_cache(snapshot: str, blobs: str) -> None:
    """Turn a saved model folder into Hugging Face cache layout: every file a symlink into blobs/."""
    os.makedirs(blobs, exist_ok=True)
    for i, (d, _, names) in enumerate(os.walk(snapshot)):
        for j, n in enumerate(names):
            p = os.path.join(d, n)
            blob = os.path.join(blobs, f"{i}-{j}")
            os.replace(p, blob)
            os.symlink(os.path.relpath(blob, d), p)


def test_loads_from_runpod_model_cache_without_prepare(tmp_path):
    """RunPod model caching mounts the repo's Hugging Face cache; the engine loads the snapshot as is."""
    repo = tmp_path / "huggingface-cache" / "hub" / "models--Lightricks--LTX-2.5-Diffusers"
    snap = repo / "snapshots" / "da0b690960"
    build_tiny_model_dir(str(snap), marker=False)
    _as_hf_cache(str(snap), str(repo / "blobs"))
    e = ltx_engine.LTXEngine(str(snap), device="cpu", memory_mode="resident", vae_tiling=False, dtype=torch.float32)
    assert e.meta == {"source": ltx_engine.SRC_REPO, "source_revision": "da0b690960", "from": "runpod model cache"}
    assert e.has_full
    frames, wav, _, _ = e.generate(_img(), "a cat laughing", width=64, height=64, num_frames=5, two_stage=False,
                                   seed=1)
    assert frames.shape == (5, 64, 64, 3) and wav is not None
