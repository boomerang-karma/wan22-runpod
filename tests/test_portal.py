"""Meme portal in mock mode: presets -> prompt, captions, the full job flow through the HTTP API, and the guards
(password, budget, capacity). Needs ffmpeg for the video steps."""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pytest
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "portal"))
os.environ.update(PORTAL_MOCK="1", PORTAL_MOCK_DELAY="0", PORTAL_DATA=tempfile.mkdtemp(prefix="portal_test_"))

import presets  # noqa: E402
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
client = TestClient(server.app)


def photo(w=900, h=1200) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 120, 60)).save(buf, "JPEG")
    return buf.getvalue()


def submit(_size=(900, 1200), **form):
    data = {"preset": "dramatic_zoom", "duration": "3", **form}
    return client.post("/api/generate", files={"image": ("p.jpg", photo(*_size), "image/jpeg")}, data=data)


def wait_done(jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{jid}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height", "-of", "csv=p=0",
                          path], capture_output=True, text=True).stdout.split()
    return out


# ------------------------------------------------------------------ presets
def test_every_preset_builds_a_long_prompt_with_sound():
    for p in presets.PRESETS:
        prompt = presets.build_prompt(p["id"], subject="a grumpy cat", line="hello there",
                                      custom="The cat jumps onto the table while a trumpet plays loudly.")
        assert len(prompt) > 150 and "{" not in prompt and prompt[0].isupper()


def test_say_it_needs_a_line_and_quotes_are_neutralised():
    with pytest.raises(presets.PresetError):
        presets.build_prompt("say_it")
    assert '"I said \'no\'"' in presets.build_prompt("say_it", line='I said "no"')


def test_custom_needs_a_real_description_and_unknown_preset_fails():
    with pytest.raises(presets.PresetError):
        presets.build_prompt("custom", custom="dance")
    with pytest.raises(presets.PresetError):
        presets.build_prompt("nope")


def test_public_presets_hide_prompt_text():
    assert all("prompt" not in p for p in client.get("/api/presets").json())


# ------------------------------------------------------------------ captions & sizes
def test_caption_png_draws_top_and_bottom_only():
    im = server.caption_png(544, 960, "top text", "a much longer bottom caption that has to wrap onto lines")
    alpha = im.getchannel("A")
    assert im.size == (544, 960)
    assert alpha.crop((0, 0, 544, 200)).getbbox() and alpha.crop((0, 760, 544, 960)).getbbox()
    assert alpha.crop((0, 380, 544, 580)).getbbox() is None


def test_sizes_respect_pipeline_multiples():
    for q, mult in (("draft", 32), ("hd", 64)):
        for w, h in server.SIZES[q].values():
            assert w % mult == 0 and h % mult == 0
    assert all((n - 1) % 8 == 0 for n in server.FRAMES.values())


def test_auto_aspect():
    assert server.pick_aspect(Image.new("RGB", (900, 1600)), "auto") == "9:16"
    assert server.pick_aspect(Image.new("RGB", (1600, 900)), "auto") == "16:9"
    assert server.pick_aspect(Image.new("RGB", (1000, 1000)), "auto") == "1:1"


# ------------------------------------------------------------------ job flow
@needs_ffmpeg
def test_generate_caption_remix_delete():
    r = submit(caption_top="when the build passes", caption_bottom="first try", subject="a cat")
    assert r.status_code == 200, r.text
    job = wait_done(r.json()["id"])
    assert job["status"] == "completed" and (job["width"], job["height"]) == (544, 960)
    assert job["video_url"].startswith(f"/media/{job['id']}/meme.mp4")
    path = os.path.join(server.store.dir(job["id"]), "meme.mp4")
    assert probe(path) == ["video,544,960", "audio"]
    assert client.get(job["video_url"]).status_code == 200

    # captions removed -> raw video is served; free, no new job
    n = len(client.get("/api/jobs").json())
    j2 = client.post(f"/api/jobs/{job['id']}/captions", json={"caption_top": "", "caption_bottom": ""}).json()
    assert "/raw.mp4" in j2["video_url"] and len(client.get("/api/jobs").json()) == n

    # HD remix with the same seed re-crops the original photo at the HD size
    hd = client.post(f"/api/jobs/{job['id']}/remix", json={"quality": "hd", "keep_seed": True}).json()
    hd = wait_done(hd["id"])
    assert hd["seed"] == job["seed"] and (hd["width"], hd["height"]) == (704, 1280) and hd["status"] == "completed"

    assert client.delete(f"/api/jobs/{job['id']}").json() == {"ok": True}
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


@needs_ffmpeg
def test_landscape_square_and_cancel(monkeypatch):
    sq = wait_done(submit(_size=(1000, 1000), aspect="auto").json()["id"])
    assert (sq["width"], sq["height"]) == (736, 736)
    monkeypatch.setattr(server, "MOCK_DELAY", 0.5)
    j = submit(aspect="16:9").json()
    assert (j["width"], j["height"]) == (960, 544)
    client.post(f"/api/jobs/{j['id']}/cancel")
    assert wait_done(j["id"])["status"] == "cancelled"


def test_bad_inputs():
    assert submit(preset="say_it").status_code == 400
    assert submit(quality="4k").status_code == 400
    assert submit(duration="4").status_code == 400
    r = client.post("/api/generate", files={"image": ("x.jpg", b"not an image", "image/jpeg")},
                    data={"preset": "vibing"})
    assert r.status_code == 400
    assert client.get("/media/whatever/../../config.yaml").status_code == 404


def test_budget_and_capacity_guards(monkeypatch):
    monkeypatch.setattr(server, "BUDGET_USD", 0.0)
    assert submit().status_code == 402
    monkeypatch.setattr(server, "BUDGET_USD", 100.0)
    monkeypatch.setattr(server, "MAX_ACTIVE", 0)
    assert submit().status_code == 429


def test_password_gate(monkeypatch):
    monkeypatch.setattr(server, "PASSWORD", "hunter2")
    c = TestClient(server.app)
    assert c.get("/api/status").json()["authed"] is False
    assert c.get("/api/jobs").status_code == 401
    assert c.post("/api/login", json={"password": "wrong"}).status_code == 401
    assert c.post("/api/login", json={"password": "hunter2"}).status_code == 200
    assert c.get("/api/jobs").status_code == 200


def test_describe_health_states():
    d = server.describe_health
    assert d({"workers": {"running": 1}, "jobs": {}}, "X")["state"] == "busy"
    assert d({"workers": {"idle": 1}, "jobs": {}}, "X")["state"] == "warm"
    assert d({"workers": {"throttled": 1}, "jobs": {"inQueue": 1}}, "US-MD-1")["state"] == "no_gpu"
    assert d({"workers": {"throttled": 1}, "jobs": {}}, "X")["state"] == "asleep"
    assert d({"error": "boom"}, "X")["state"] == "unknown"
