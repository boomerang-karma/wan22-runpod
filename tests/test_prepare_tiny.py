"""prepare(): download -> bf16 -> fuse LoRA per expert -> save -> marker -> atomic rename, against a fake hub that
serves the tiny checkpoint, then load the result with WanEngine and generate."""
import fnmatch
import json
import os
import shutil
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
sys.path.insert(0, os.path.dirname(__file__))

import huggingface_hub  # noqa: E402
import wan_engine  # noqa: E402
from tiny_wan import build_tiny_model_dir, fake_lightning_lora  # noqa: E402


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    repo = build_tiny_model_dir(str(tmp_path / "hub_repo"))
    os.remove(os.path.join(repo, "prepared.json"))                       # a real hub repo has no marker
    lora_dir = tmp_path / "hub_lora"
    lora_dir.mkdir()
    save_file(fake_lightning_lora(style="down_up_alpha"), str(lora_dir / "high.safetensors"))
    save_file(fake_lightning_lora(style="down_up_alpha"), str(lora_dir / "low.safetensors"))
    downloads = []

    def snapshot_download(repo_id, revision=None, local_dir=None, allow_patterns=None, **_):
        downloads.append(tuple(allow_patterns))
        for root, _, files in os.walk(repo):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), repo)
                if any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                    os.makedirs(os.path.dirname(os.path.join(local_dir, rel)), exist_ok=True)
                    shutil.copy2(os.path.join(root, fn), os.path.join(local_dir, rel))
        return local_dir

    def hf_hub_download(repo_id, filename, revision=None, cache_dir=None, **_):
        return str(lora_dir / ("high.safetensors" if "high" in filename else "low.safetensors"))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    return repo, downloads


def test_prepare_builds_a_loadable_fused_bf16_model(fake_hub, tmp_path):
    repo, downloads = fake_hub
    out = str(tmp_path / "volume" / "model")
    msgs = []
    res = wan_engine.prepare(out, str(tmp_path / "work"), device="cpu", progress=msgs.append)
    assert res["status"] == "prepared"
    assert res["transformer"]["lora_modules"] == 20 and res["transformer_2"]["lora_modules"] == 20
    assert not os.path.exists(out + ".partial") and not os.path.exists(str(tmp_path / "work" / "wan_src"))
    assert downloads[-2:] == [("transformer/*",), ("transformer_2/*",)]            # one expert at a time
    for sub in ("transformer", "transformer_2"):                                       # saved bf16, fused, no LoRA keys
        files = [f for f in os.listdir(os.path.join(out, sub)) if f.endswith(".safetensors")]
        sd = {}
        for f in files:
            sd.update(load_file(os.path.join(out, sub, f)))
        keep32 = tuple(wan_engine.WanTransformer3DModel._keep_in_fp32_modules)     # diffusers keeps norms/time emb fp32
        big = {k: v for k, v in sd.items() if not any(m in k for m in keep32)}
        assert big and all(v.dtype == torch.bfloat16 for v in big.values()), {k: v.dtype for k, v in big.items()}
        assert not any("lora" in k for k in sd)
        src = {}
        for f in os.listdir(os.path.join(repo, sub)):
            if f.endswith(".safetensors"):
                src.update(load_file(os.path.join(repo, sub, f)))
        k = "blocks.0.attn1.to_q.weight"
        assert (sd[k].float() - src[k].to(torch.bfloat16).float()).abs().max() > 1e-3     # the LoRA really landed
    te = {}
    for f in os.listdir(os.path.join(out, "text_encoder")):
        if f.endswith(".safetensors"):
            te.update(load_file(os.path.join(out, "text_encoder", f)))
    assert te and all(v.dtype == torch.bfloat16 for v in te.values() if v.is_floating_point())   # 11 GB, not 22 GB
    meta = json.load(open(os.path.join(out, "prepared.json")))
    assert meta["dtype"] == "bfloat16" and "seconds" in meta

    again = wan_engine.prepare(out, str(tmp_path / "work"), device="cpu")               # idempotent
    assert again["status"] == "already_prepared"

    eng = wan_engine.WanEngine(out, device="cpu", dtype=torch.float32)
    frames, _ = eng.generate(Image.new("RGB", (32, 48), (90, 160, 220)), "a hero with balloons", width=32, height=48,
                             num_frames=5, steps=4, seed=3)
    assert frames.shape == (5, 48, 32, 3) and frames.dtype == np.uint8
