"""End-to-end request path in mock mode: prepare -> restart -> generate -> real mp4 via ffmpeg."""
import base64
import importlib
import io
import json
import os
import subprocess
import sys

import pytest
from PIL import Image

WORKER = os.path.join(os.path.dirname(__file__), "..", "worker")
sys.path.insert(0, WORKER)


def _fresh_handler(volume):
    os.environ.update({"WORKER_MODE": "mock", "VOLUME_PATH": str(volume), "WORK_DIR": str(volume / "work")})
    import handler
    return importlib.reload(handler)


def _img_b64(w=1080, h=1920):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (230, 150, 60)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_unprepared_worker_reports_and_refuses_generate(tmp_path):
    h = _fresh_handler(tmp_path)
    info = h.handler({"id": "j0", "input": {"action": "info"}})
    assert info["ready"] is False and "not prepared" in info["error"]
    out = h.handler({"id": "j1", "input": {"prompt": "x", "image_base64": _img_b64()}})
    assert "not prepared" in out["error"]


def test_prepare_then_generate_returns_playable_mp4(tmp_path):
    h = _fresh_handler(tmp_path)
    res = h.handler({"id": "p", "input": {"action": "prepare"}})
    assert res["refresh_worker"] is True and res["job_results"]["status"] == "prepared"
    h = _fresh_handler(tmp_path)                           # simulates the worker restart
    out = h.handler({"id": "g", "input": {"prompt": "cake and balloons", "image_base64": _img_b64(),
                                          "resolution": "480p", "num_frames": 17, "seed": 5}})
    assert "error" not in out, out.get("error")
    assert (out["width"], out["height"], out["seed"]) == (480, 832, 5)
    assert out["timings"]["cold_start_s"] is not None      # first job on the worker carries the cold start
    mp4 = tmp_path / "o.mp4"
    mp4.write_bytes(base64.b64decode(out["video_base64"]))
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_read_frames,r_frame_rate,pix_fmt", "-of", "json", str(mp4)]))["streams"][0]
    assert (probe["width"], probe["height"], int(probe["nb_read_frames"])) == (480, 832, 17)
    assert probe["r_frame_rate"] == "16/1" and probe["pix_fmt"] == "yuv420p"
    second = h.handler({"id": "g2", "input": {"prompt": "again", "image_base64": _img_b64(), "resolution": "480p",
                                              "num_frames": 5}})
    assert "cold_start_s" not in second["timings"]          # only the first job pays it


@pytest.mark.parametrize("bad,msg", [
    ({"num_frames": 80}, "4k+1"), ({"width": 500, "height": 832}, "multiples of 16"),
    ({"resolution": "4k"}, "resolution"), ({"prompt": ""}, "prompt"), ({"image_base64": "%%%"}, "base64"),
    ({"steps": 0}, "steps"), ({"image_url": "file:///etc/passwd"}, "http"),
])
def test_bad_inputs_are_rejected_with_a_reason(tmp_path, bad, msg):
    h = _fresh_handler(tmp_path)
    h.handler({"id": "p", "input": {"action": "prepare"}})
    h = _fresh_handler(tmp_path)
    inp = {"prompt": "ok", "image_base64": _img_b64(64, 64), **bad}
    if "image_url" in bad:
        inp.pop("image_base64")
    out = h.handler({"id": "b", "input": inp})
    assert "error" in out and msg in out["error"], out


def test_landscape_input_gets_landscape_output(tmp_path):
    h = _fresh_handler(tmp_path)
    h.handler({"id": "p", "input": {"action": "prepare"}})
    h = _fresh_handler(tmp_path)
    out = h.handler({"id": "g", "input": {"prompt": "x", "image_base64": _img_b64(1920, 1080), "num_frames": 5}})
    assert (out["width"], out["height"]) == (1280, 720)


def test_oversized_output_without_bucket_is_an_error_not_a_truncated_payload(tmp_path, monkeypatch):
    h = _fresh_handler(tmp_path)
    h.handler({"id": "p", "input": {"action": "prepare"}})
    h = _fresh_handler(tmp_path)
    monkeypatch.setattr(h, "INLINE_LIMIT_MB", 0.001)
    out = h.handler({"id": "g", "input": {"prompt": "x", "image_base64": _img_b64(), "resolution": "480p",
                                          "num_frames": 5}})
    assert "video_base64" not in out and "bucket" in out["error"]
