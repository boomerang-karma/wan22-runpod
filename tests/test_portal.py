"""Meme portal in mock mode: presets -> prompt, captions, the full job flow through the HTTP API, and the guards
(password, budget, capacity). Needs ffmpeg for the video steps."""
import base64
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
    monkeypatch.setattr(server, "AUTH_MODE", "password")
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


# ------------------------------------------------------------------ Auth0 login (fake tenant, real RS256 tokens)
import json  # noqa: E402
import urllib.parse  # noqa: E402

import auth  # noqa: E402
import jwt as pyjwt  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
JWK = dict(json.loads(pyjwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key())), kid="k1", use="sig", alg="RS256")


def id_token(nonce, **over):
    now = int(time.time())
    claims = {"iss": "https://tenant.auth0.com/", "aud": "cid", "sub": "auth0|alice", "email": "alice@example.com",
              "email_verified": True, "name": "Alice", "iat": now, "exp": now + 600, "nonce": nonce, **over}
    return pyjwt.encode(claims, KEY, algorithm="RS256", headers={"kid": "k1"})


@pytest.fixture
def tenant(monkeypatch):
    a = auth.Auth0("tenant.auth0.com", "cid", "s3cret", "https://memes.example.com")
    monkeypatch.setattr(server, "AUTH0", a)
    monkeypatch.setattr(server, "AUTH_MODE", "auth0")
    monkeypatch.setattr(auth, "_get_json", lambda url: {"keys": [JWK]})
    issued = {}

    def token_endpoint(url, data):
        assert url == "https://tenant.auth0.com/oauth/token" and data["client_secret"] == "s3cret"
        assert data["code"] == "good-code" and data["redirect_uri"] == "https://memes.example.com/auth/callback"
        return {"id_token": issued["make"](issued["nonce"])}

    monkeypatch.setattr(auth, "_post_form", token_endpoint)
    return a, issued


def sign_in(c, issued, make=id_token, code="good-code", state=None):
    r = c.get("/login", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("https://tenant.auth0.com/authorize?")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(r.headers["location"]).query)
    assert q["redirect_uri"] == ["https://memes.example.com/auth/callback"] and q["scope"] == ["openid profile email"]
    issued.update(nonce=q["nonce"][0], make=make)
    return c.get(f"/auth/callback?code={code}&state={state or q['state'][0]}", follow_redirects=False)


def as_bob(nonce):
    return id_token(nonce, sub="auth0|bob", email="bob@example.com", name="Bob")


@needs_ffmpeg
def test_auth0_login_and_private_galleries(tenant):
    _, issued = tenant
    alice, bob = TestClient(server.app), TestClient(server.app)
    assert alice.get("/api/status").json()["authed"] is False
    assert alice.get("/api/jobs").status_code == 401 and alice.get("/healthz").json() == {"ok": True}

    r = sign_in(alice, issued)
    assert r.status_code == 303 and r.headers["location"] == "/"
    st = alice.get("/api/status").json()
    assert st["authed"] and st["user"]["email"] == "alice@example.com" and st["auth_mode"] == "auth0"

    j = alice.post("/api/generate", files={"image": ("p.jpg", photo(), "image/jpeg")},
                   data={"preset": "vibing", "duration": "3"}).json()
    t0 = time.time()
    while alice.get(f"/api/jobs/{j['id']}").json()["status"] in ("queued", "running") and time.time() - t0 < 60:
        time.sleep(0.2)
    assert server.store.get(j["id"])["owner"] == "auth0|alice"

    sign_in(bob, issued, make=as_bob)
    assert bob.get("/api/status").json()["user"]["name"] == "Bob"
    assert all(x["id"] != j["id"] for x in bob.get("/api/jobs").json())
    assert bob.get(f"/api/jobs/{j['id']}").status_code == 404
    assert bob.get(f"/media/{j['id']}/input.jpg").status_code == 404
    assert bob.delete(f"/api/jobs/{j['id']}").status_code == 404
    assert any(x["id"] == j["id"] for x in alice.get("/api/jobs").json())
    assert alice.get(f"/media/{j['id']}/input.jpg").status_code == 200

    r = alice.get("/logout", follow_redirects=False)
    assert r.headers["location"].startswith("https://tenant.auth0.com/v2/logout?client_id=cid")
    assert alice.get("/api/jobs").status_code == 401


@pytest.mark.parametrize("make,state,why", [
    (id_token, "forged", "login expired"),
    (lambda n: id_token("other-nonce"), None, "nonce"),
    (lambda n: id_token(n, aud="someone-else"), None, "audience"),
    (lambda n: id_token(n, iss="https://evil.example/"), None, "issuer"),
    (lambda n: id_token(n, exp=int(time.time()) - 3600, iat=int(time.time()) - 7200), None, "expired"),
])
def test_auth0_rejects_bad_callbacks(tenant, make, state, why):
    _, issued = tenant
    c = TestClient(server.app)
    r = sign_in(c, issued, make=make, state=state)
    assert r.status_code == 303 and r.headers["location"].startswith("/?login_error=")
    assert why in urllib.parse.unquote(r.headers["location"]).lower()
    assert c.get("/api/jobs").status_code == 401


def test_auth0_rejects_token_from_another_key(tenant):
    _, issued = tenant
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = lambda n: pyjwt.encode({"iss": "https://tenant.auth0.com/", "aud": "cid", "sub": "auth0|mallory",
                                     "iat": int(time.time()), "exp": int(time.time()) + 600, "nonce": n},
                                    other, algorithm="RS256", headers={"kid": "k1"})
    r = sign_in(TestClient(server.app), issued, make=forged)
    assert "signature" in urllib.parse.unquote(r.headers["location"]).lower()


def test_guest_list():
    a = auth.Auth0("t.auth0.com", "c", "s", "https://x", allowed="boss@corp.com, @example.com")
    assert a.is_allowed({"email": "Alice@Example.com", "email_verified": True})
    assert a.is_allowed({"email": "boss@corp.com", "email_verified": True})
    assert not a.is_allowed({"email": "alice@example.com", "email_verified": False})
    assert not a.is_allowed({"email": "eve@corp.com", "email_verified": True})
    assert not a.is_allowed({"email": "x@notexample.com", "email_verified": True})
    assert auth.Auth0("t", "c", "s", "https://x").is_allowed({"sub": "anyone"})


def test_guest_list_blocks_login(tenant, monkeypatch):
    a, issued = tenant
    monkeypatch.setattr(a, "allowed", ["@corp.com"])
    r = sign_in(TestClient(server.app), issued)
    assert "guest list" in urllib.parse.unquote(r.headers["location"])


def test_per_user_limits(tenant, monkeypatch):
    _, issued = tenant
    c = TestClient(server.app)
    sign_in(c, issued, make=as_bob)
    send = lambda: c.post("/api/generate", files={"image": ("p.jpg", photo(), "image/jpeg")},
                          data={"preset": "vibing", "duration": "3"})
    monkeypatch.setattr(server, "USER_MAX_ACTIVE", 0)
    r = send()
    assert r.status_code == 429 and "you have" in r.json()["detail"]
    monkeypatch.setattr(server, "USER_MAX_ACTIVE", 5)
    monkeypatch.setattr(server, "USER_BUDGET_USD", 0.0)
    r = send()
    assert r.status_code == 402 and "allowance" in r.json()["detail"]


def test_spend_ledger_per_user_and_old_format(tmp_path):
    st = server.Store(str(tmp_path))
    import datetime
    (tmp_path / "spend.json").write_text(json.dumps({datetime.date.today().isoformat(): 0.5}))
    assert st.spent_today() == 0.5 and st.spent_today("a") == 0.0
    st.add_spend(0.25, "a")
    assert st.spent_today() == 0.75 and st.spent_today("a") == 0.25 and st.spent_today("b") == 0.0


# ------------------------------------------------------------------ Modal backend (fake Modal client, no network)
class _FakeCall:
    def __init__(self, result, polls=2):
        self.object_id, self.result, self.polls, self.cancelled = "fc-123", result, polls, False

    def get(self, timeout=None):
        if self.cancelled:
            raise RuntimeError("cancelled")
        if self.polls:
            self.polls -= 1
            raise TimeoutError()
        return self.result

    def cancel(self, terminate_containers=False):
        self.cancelled = True


def _modal_backend(monkeypatch, tmp_path, call):
    import modal
    import types
    sent = []

    class Run:
        def spawn(self, inp):
            sent.append(inp)
            return call

        def get_current_stats(self):
            return types.SimpleNamespace(num_total_runners=0, num_running_inputs=0, backlog=0)

    board = {"fc-123": "denoise step 4/8"}
    monkeypatch.setattr(modal.Cls, "from_name", lambda app, name: (lambda: types.SimpleNamespace(run=Run())))
    monkeypatch.setattr(modal.Dict, "from_name", lambda name, create_if_missing=False: board)
    monkeypatch.setattr(modal.FunctionCall, "from_id", lambda cid: call)
    monkeypatch.setattr(server, "POLL_S", 0.01)
    monkeypatch.setattr(server, "MOCK", False)
    monkeypatch.setenv("PORTAL_BACKEND", "modal")
    st = server.Store(str(tmp_path))
    b = server.Backend(st)
    return b, st, sent


def _queued_job(st, tmp_path):
    jid = "j1"
    os.makedirs(st.dir(jid))
    Image.new("RGB", (544, 960), (1, 2, 3)).save(os.path.join(st.dir(jid), "input.jpg"))
    st.save({"id": jid, "created": time.time(), "status": "queued", "owner": "local", "prompt": "p", "width": 544,
             "height": 960, "quality": "draft", "num_frames": 73, "audio": True, "seed": 7, "caption_top": "",
             "caption_bottom": "", "preset": "vibing"})
    return jid


@needs_ffmpeg
def test_modal_backend_runs_a_job_and_estimates_cost(monkeypatch, tmp_path):
    d = tmp_path / "v"
    d.mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=544x960:d=1", "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", str(d / "a.mp4")], check=True)
    video = base64.b64encode((d / "a.mp4").read_bytes()).decode()
    call = _FakeCall({"video_base64": video, "seed": 7, "gpu": "NVIDIA A100-SXM4-80GB",
                      "timings": {"handler_total_s": 60, "cold_start_s": 100}})
    b, st, sent = _modal_backend(monkeypatch, tmp_path, call)
    assert b.endpoint == "modal:ltx-memes" and b.health()["state"] == "asleep"
    jid = _queued_job(st, tmp_path)
    b._run(jid)
    job = st.get(jid)
    assert job["status"] == "completed", job.get("error")
    assert job["modal_call_id"] == "fc-123" and sent[0]["width"] == 544 and sent[0]["seed"] == 7
    rate = server.MODAL_GPU_USD_S["A100"] + server.MODAL_HOST_USD_S
    assert job["cost_usd"] == round((60 + 100 + server.MODAL_SCALEDOWN_S) * rate, 4)
    assert st.spent_today("local") == job["cost_usd"]


def test_modal_backend_reports_worker_errors_and_cancels(monkeypatch, tmp_path):
    b, st, _ = _modal_backend(monkeypatch, tmp_path, _FakeCall({"error": "model not prepared at /weights/x"}, 0))
    jid = _queued_job(st, tmp_path)
    b._run(jid)
    assert st.get(jid)["status"] == "failed" and "not prepared" in st.get(jid)["error"]

    call = _FakeCall({}, polls=1000)
    b, st, _ = _modal_backend(monkeypatch, tmp_path / "c", call)
    jid = _queued_job(st, tmp_path)
    st.update(jid, cancel=True)
    b._run(jid)
    assert call.cancelled and st.get(jid)["status"] == "cancelled"


def test_modal_cost_falls_back_to_h100_rate_for_unknown_gpus():
    out = {"gpu": "Mystery GPU", "timings": {"handler_total_s": 10}}
    rate = server.MODAL_GPU_USD_S["H100"] + server.MODAL_HOST_USD_S
    assert server.modal_cost(out) == round((10 + server.MODAL_SCALEDOWN_S) * rate, 4)


# ------------------------------------------------------------------ group photos
def test_group_prompt_keeps_everyone_else_in_place():
    p = presets.build_prompt("victory_dance", group=True)
    assert p.startswith("The person in the centre of the photo") and "Everyone else in the photo stays" in p
    assert "nobody new appears" in p
    assert "Everyone else" not in presets.build_prompt("victory_dance")             # single-subject unchanged


def test_group_size_keeps_photo_shape_on_the_64px_grid():
    for w, h in ((1072, 976), (4000, 3000), (900, 1600), (5000, 1000)):
        gw, gh = server.group_size(Image.new("RGB", (w, h)))
        assert gw % 64 == 0 and gh % 64 == 0 and max(gw, gh) <= 1280 + 32
        assert abs(gw / gh - min(max(w / h, 9 / 16), 16 / 9)) < 0.08


@needs_ffmpeg
def test_group_job_uses_full_model_pinned_at_both_ends():
    r = submit(_size=(1072, 976), quality="group")
    assert r.status_code == 200, r.text
    job = wait_done(r.json()["id"])
    assert job["quality"] == "group" and (job["width"], job["height"]) == (1280, 1152)
    inp = server.backend._job_input(server.store.jobs[job["id"]])
    assert inp["quality"] == "full" and inp["loop"] is True and inp["two_stage"] is True and inp["steps"] == 30
    assert "Everyone else in the photo stays" in inp["prompt"]


def test_analyze_counts_faces_and_suggests_group(monkeypatch):
    monkeypatch.setattr(server, "count_faces", lambda img: 4)
    a = client.post("/api/analyze", files={"image": ("p.jpg", photo(), "image/jpeg")}).json()
    assert a == {"faces": 4, "suggest_group": True}
    monkeypatch.setattr(server, "count_faces", lambda img: 1)
    assert client.post("/api/analyze", files={"image": ("p.jpg", photo(), "image/jpeg")}).json()["suggest_group"] is False
    assert client.post("/api/analyze", files={"image": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 400


def test_count_faces_runs_on_a_blank_photo():
    assert server.count_faces(Image.new("RGB", (800, 600), (128, 128, 128))) == 0
