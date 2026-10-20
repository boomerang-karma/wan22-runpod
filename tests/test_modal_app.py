"""deploy/modal/ltx_app.py without Modal: the app definition imports, and the container-side class drives the real
worker handler (mock engine) with the same input and output as RunPod."""
import base64
import importlib.util
import io
import os
import sys

import pytest
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "worker"))


def _load_app():
    spec = importlib.util.spec_from_file_location("ltx_app", os.path.join(ROOT, "deploy", "modal", "ltx_app.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def container(tmp_path, monkeypatch):
    """The LTX class as it runs inside a Modal container, on a prepared mock volume."""
    m = _load_app()
    monkeypatch.setenv("WORKER_MODE", "mock")
    monkeypatch.setenv("VOLUME_PATH", str(tmp_path))
    monkeypatch.setenv("MODEL_SOURCE", "volume")
    import mock_engine
    mock_engine.prepare(str(tmp_path / "ltx25-i2v-distilled-bf16"), str(tmp_path / "work"))
    board, stopped = {}, []
    monkeypatch.setattr(m, "progress_board", board)
    monkeypatch.setattr(m.modal, "current_function_call_id", lambda: "fc-1")
    monkeypatch.setattr(m.modal.experimental, "stop_fetching_inputs", lambda: stopped.append(True))
    sys.modules.pop("handler", None)                   # load() imports it fresh, like a new container
    obj = m.LTX._get_user_cls()()
    obj.load()
    return obj, board, stopped


def _img_b64():
    buf = io.BytesIO()
    Image.new("RGB", (1080, 1920), (200, 100, 50)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_app_defines_gpu_class_volume_and_commands():
    m = _load_app()
    assert m.app.name == "ltx-memes" and m.GPUS[0] == "A100-80GB" and m.SCALEDOWN_S == 120
    assert m.WORKER_ENV["VOLUME_PATH"] == m.WEIGHTS and m.WORKER_ENV["INLINE_LIMIT_MB"] == "200"


def test_container_generates_like_the_runpod_worker(container):
    obj, board, _ = container
    run = obj.run
    assert run({"action": "info"})["ready"] is True
    out = run({"prompt": "a cat laughing", "image_base64": _img_b64(), "resolution": "540p", "num_frames": 25,
               "seed": 3})
    assert "error" not in out and out["seed"] == 3 and (out["width"], out["height"]) == (544, 960)
    assert base64.b64decode(out["video_base64"])[4:8] == b"ftyp"
    assert "fc-1" not in board                           # progress entry cleaned up after the job
    assert "use the prepare_weights" in run({"action": "prepare"})["error"]


def test_container_asks_modal_for_a_fresh_container_after_oom(container, monkeypatch):
    obj, _, stopped = container
    import handler

    def oom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.28 GiB")
    monkeypatch.setattr(handler.ENGINE, "generate", oom)
    out = obj.run({"prompt": "x", "image_base64": _img_b64(), "resolution": "540p", "num_frames": 9})
    assert "out of memory" in out["error"] and "refresh_worker" not in out and stopped == [True]
