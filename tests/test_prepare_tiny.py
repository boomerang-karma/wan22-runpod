"""prepare(): per-component download -> bf16 re-save of the big parts -> marker -> atomic rename, against a fake hub
that serves the tiny checkpoint; then load the result with LTXEngine and generate."""
import fnmatch
import json
import os
import shutil
import sys

import httpx
import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
sys.path.insert(0, os.path.dirname(__file__))

import huggingface_hub  # noqa: E402
import ltx_engine  # noqa: E402
from huggingface_hub.errors import GatedRepoError  # noqa: E402
from tiny_ltx import build_tiny_model_dir  # noqa: E402


def _tensors(folder):
    sd = {}
    for f in os.listdir(folder):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(folder, f)))
    return sd


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    repo = build_tiny_model_dir(str(tmp_path / "hub_repo"), marker=False)
    for sub in ("transformer", "connectors"):                  # ship them fp32, like the real distilled transformer
        folder = os.path.join(repo, sub)
        for f in os.listdir(folder):
            if f.endswith(".safetensors"):
                sd = load_file(os.path.join(folder, f))
                save_file({k: v.float() for k, v in sd.items()}, os.path.join(folder, f))
    calls = []

    def snapshot_download(repo_id, revision=None, local_dir=None, allow_patterns=None, token=None, **_):
        calls.append((tuple(allow_patterns), token))
        if token != "hf_test":
            raise GatedRepoError("401 gated", response=httpx.Response(401, request=httpx.Request("GET", "https://hf.co")))
        for root, _, files in os.walk(repo):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), repo)
                if any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                    os.makedirs(os.path.dirname(os.path.join(local_dir, rel)), exist_ok=True)
                    shutil.copy2(os.path.join(root, fn), os.path.join(local_dir, rel))
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return repo, calls


def test_prepare_builds_a_loadable_bf16_model(fake_hub, tmp_path):
    repo, calls = fake_hub
    out = str(tmp_path / "volume" / "model")
    msgs = []
    res = ltx_engine.prepare(out, str(tmp_path / "work"), device="cpu", progress=msgs.append, token="hf_test")
    assert res["status"] == "prepared"
    assert not os.path.exists(out + ".partial") and not os.path.exists(str(tmp_path / "work" / "ltx_src"))
    patterns = [p for p, _ in calls]
    assert all(len(p) == 1 for p in patterns)                                  # one component at a time
    assert ("transformer/*",) in patterns and ("latent_upsampler/*",) in patterns
    for sub in ("transformer", "connectors"):
        sd = _tensors(os.path.join(out, sub))
        floats = [v for v in sd.values() if v.is_floating_point()]
        assert floats and any(v.dtype == torch.bfloat16 for v in floats), sub
        assert res[sub]["dtype"] == "bfloat16"
    for sub in ("vae", "audio_vae", "vocoder", "text_encoder", "tokenizer", "scheduler", "latent_upsampler"):
        assert os.listdir(os.path.join(out, sub)), sub
    meta = json.load(open(os.path.join(out, "prepared.json")))
    assert meta["dtype"] == "bfloat16" and meta["source"] == "Lightricks/LTX-2.5-Diffusers" and "seconds" in meta
    assert "hf_test" not in json.dumps(meta) and "hf_test" not in " ".join(msgs)       # token never persisted

    again = ltx_engine.prepare(out, str(tmp_path / "work"), device="cpu")             # idempotent, no download
    assert again["status"] == "already_prepared"

    eng = ltx_engine.LTXEngine(out, device="cpu", vae_tiling=False, dtype=torch.float32)
    frames, audio, _, _ = eng.generate(Image.new("RGB", (64, 64), (90, 160, 220)), "a hero with balloons",
                                       width=64, height=64, num_frames=5, seed=3)
    assert frames.shape == (5, 64, 64, 3) and frames.dtype == np.uint8 and audio.shape[0] == 2


def test_prepare_without_access_explains_how_to_fix_it(fake_hub, tmp_path):
    with pytest.raises(RuntimeError, match="Accept the license"):
        ltx_engine.prepare(str(tmp_path / "m"), str(tmp_path / "w"), device="cpu", token=None)
    assert not os.path.exists(str(tmp_path / "m"))
