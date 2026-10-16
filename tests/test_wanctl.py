"""Client: payload shapes, dry-run deploy (no network), image prep, cost estimate, key hygiene."""
import base64
import io
import os
import sys

import yaml
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import wanctl  # noqa: E402


def cfg(**over):
    with open(os.path.join(ROOT, "config.example.yaml")) as f:
        c = yaml.safe_load(f)
    c["runpod"].update(api_key="rpa_TESTKEY_should_never_print", data_center="EU-RO-1")
    c["image"]["name"] = "ghcr.io/me/wan22-runpod-worker:0.2.0"
    c["worker_env"]["BUCKET_SECRET_ACCESS_KEY"] = "s3cr3t"
    c.update(over)
    return c


def test_endpoint_payload_matches_rest_schema_fields():
    body = wanctl.endpoint_payload(cfg(), "tpl1", {"EU-RO-1": "vol1"})
    assert body["templateId"] == "tpl1" and body["networkVolumeId"] == "vol1" and "networkVolumeIds" not in body
    assert body["dataCenterIds"] == ["EU-RO-1"] and body["gpuTypeIds"][0] == "NVIDIA A100 80GB PCIe"
    assert body["executionTimeoutMs"] == 900_000 and body["idleTimeout"] == 5 and body["workersMin"] == 0
    assert body["minCudaVersion"] == "13.0" and body["scalerType"] == "QUEUE_DELAY"


def multi(**over):
    c = cfg(**over)
    c["runpod"]["data_centers"] = ["US-MD-1", "US-MO-1", "EU-FR-1"]
    return c


def test_multi_region_endpoint_only_serves_prepared_regions():
    st = {"volumes": {"US-MD-1": "v1", "US-MO-1": "v2", "EU-FR-1": "v3"}, "prepared": ["US-MD-1", "EU-FR-1"]}
    body = wanctl.endpoint_payload(multi(), "tpl", wanctl.serving_volumes(multi(), st))
    assert body["dataCenterIds"] == ["US-MD-1", "EU-FR-1"] and body["networkVolumeIds"] == ["v1", "v3"]
    assert body["networkVolumeId"] == "v1"
    # nothing prepared yet: serve from the primary only, so prepare/first run behave like a single region
    first = wanctl.serving_volumes(multi(), {"volumes": {"US-MD-1": "v1", "US-MO-1": "v2"}, "prepared": []})
    assert first == {"US-MD-1": "v1"}


def test_deploy_rejects_regions_without_serverless():
    c = multi()
    c["runpod"]["data_centers"].append("US-NE-1")
    try:
        wanctl.validate(c)
        raise AssertionError("expected CtlError")
    except wanctl.CtlError as e:
        assert "US-NE-1 cannot run serverless" in str(e)


def test_old_single_region_state_migrates_as_prepared():
    st = wanctl._volumes_state({"volume_id": "v1", "data_center": "US-MD-1", "endpoint_id": "e"})
    assert st["volumes"] == {"US-MD-1": "v1"} and st["prepared"] == ["US-MD-1"] and "volume_id" not in st


def test_prepare_endpoint_is_pinned_to_one_region_with_small_gpus():
    body = wanctl.prepare_endpoint_payload(multi(), "tpl", "US-MO-1", "v2")
    assert body["dataCenterIds"] == ["US-MO-1"] and body["networkVolumeId"] == "v2" and "networkVolumeIds" not in body
    assert body["name"].endswith("-prep-US-MO-1") and "NVIDIA RTX A5000" in body["gpuTypeIds"]
    assert not any("80GB" in g for g in body["gpuTypeIds"]) and body["workersMax"] == 1


def test_prepare_fills_each_unprepared_region_then_widens_the_endpoint(tmp_path, monkeypatch, capsys):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(multi(huggingface={"token": "hf_x"})))
    os.chmod(p, 0o600)
    monkeypatch.setattr(wanctl, "STATE_FILE", str(tmp_path / "state.json"))
    wanctl.save_state({"template_id": "tpl", "endpoint_id": "main",
                       "volumes": {"US-MD-1": "v1", "US-MO-1": "v2", "EU-FR-1": "v3"}, "prepared": ["US-MD-1"]})
    calls = []

    def rest(self, method, path, body=None, ok404=False):
        calls.append((method, path, body))
        return {"id": f"tmp-{body['dataCenterIds'][0]}"} if method == "POST" else {}

    def run_job(rp, ep, payload, poll_s, quiet=False, label=""):
        ok = ep == "tmp-US-MO-1"                                    # EU-FR-1 fails
        return {"status": "COMPLETED" if ok else "FAILED", "output": {"job_results": {"status": "prepared"}} if ok
                else {"error": "boom"}}

    monkeypatch.setattr(wanctl.RunPod, "rest", rest)
    monkeypatch.setattr(wanctl, "run_job", run_job)
    assert wanctl.main(["--config", str(p), "prepare"]) == 1           # one region failed -> non-zero, clear message
    st = wanctl.load_state()
    assert st["prepared"] == ["US-MD-1", "US-MO-1"]
    deleted = sorted(path for m, path, _ in calls if m == "DELETE")
    assert deleted == ["/endpoints/tmp-EU-FR-1", "/endpoints/tmp-US-MO-1"]  # temporary endpoints always removed
    main_patch = [b for m, path, b in calls if m == "PATCH" and path == "/endpoints/main"][-1]
    assert main_patch["dataCenterIds"] == ["US-MD-1", "US-MO-1"] and main_patch["networkVolumeIds"] == ["v1", "v2"]
    assert "EU-FR-1" in capsys.readouterr().err


def test_template_payload_drops_empty_env_and_sets_volume_path():
    body = wanctl.template_payload(cfg())
    assert body["isServerless"] is True and body["containerDiskInGb"] == 120
    env = body["env"]
    assert env["VOLUME_PATH"] == "/runpod-volume" and env["MEMORY_MODE"] == "resident"
    assert "BUCKET_NAME" not in env                        # empty values are not sent


def test_dry_run_deploy_prints_masked_payloads_and_never_the_key(tmp_path, capsys, monkeypatch):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg()))
    os.chmod(p, 0o600)
    monkeypatch.setattr(wanctl, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert wanctl.main(["--config", str(p), "deploy", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "POST https://rest.runpod.io/v1/networkvolumes" in out
    assert "POST https://rest.runpod.io/v1/templates" in out and "POST https://rest.runpod.io/v1/endpoints" in out
    assert "rpa_TESTKEY" not in out and "s3cr3t" not in out and '"***"' in out
    assert not (tmp_path / "state.json").exists()          # dry run writes no state


def test_env_api_key_wins_over_file(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "from_env")
    assert wanctl.api_key(cfg()) == "from_env"


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    c = cfg()
    c["runpod"]["api_key"] = ""
    try:
        wanctl.api_key(c)
        raise AssertionError("expected CtlError")
    except wanctl.CtlError as e:
        assert "API key" in str(e)


def test_encode_image_cover_crops_to_generation_size(tmp_path):
    p = tmp_path / "frame.png"
    Image.new("RGB", (1080, 1920), (10, 20, 30)).save(p)
    b64, size = wanctl.encode_image(str(p), "720p")
    assert size == (704, 1280)                              # two-stage sizes are multiples of 64
    assert Image.open(io.BytesIO(base64.b64decode(b64))).size == (704, 1280)
    assert len(b64) < 1_000_000


def test_custom_size_keeps_photo_shape_and_validates_multiples(tmp_path):
    p = tmp_path / "square.jpg"
    Image.new("RGB", (1072, 976), (10, 20, 30)).save(p)
    _, size = wanctl.encode_image(str(p), "720p", wanctl.parse_size("736x672", two_stage=False))
    assert size == (736, 672)
    assert wanctl.parse_size("1280x1152", two_stage=True) == (1280, 1152)
    for bad, two in (("736x672", True), ("700x672", False), ("12x12", False), ("big", False)):
        try:
            wanctl.parse_size(bad, two)
            raise AssertionError(bad)
        except wanctl.CtlError:
            pass


def test_cost_estimate_includes_cold_start_only_when_reported():
    c = cfg()
    warm = wanctl.estimate_cost(c, {"gpu": "NVIDIA A100 80GB PCIe", "timings": {}}, 120_000)
    cold = wanctl.estimate_cost(c, {"gpu": "NVIDIA A100 80GB PCIe", "timings": {"cold_start_s": 180}}, 120_000)
    assert warm["billed_s_est"] == 125.0 and warm["usd_est"] == round(125 * 2.72 / 3600, 4)
    assert cold["billed_s_est"] == 305.0
    assert wanctl.estimate_cost(c, {"gpu": "Mystery GPU"}, 1000)["usd_est"] is None


def test_prepare_needs_an_hf_token_and_sends_it_only_in_the_job(tmp_path, monkeypatch, capsys):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg()))
    os.chmod(p, 0o600)
    monkeypatch.setattr(wanctl, "STATE_FILE", str(tmp_path / "state.json"))
    (tmp_path / "state.json").write_text('{"endpoint_id": "ep1", "template_id": "t", "volumes": {"EU-RO-1": "v"}, "prepared": []}')
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert wanctl.main(["--config", str(p), "prepare"]) == 1
    assert "huggingface.co/Lightricks/LTX-2.5-Diffusers" in capsys.readouterr().err

    sent = []
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET")
    monkeypatch.setattr(wanctl, "run_job", lambda rp, ep, payload, poll_s, quiet=False, label="": (
        sent.append(payload), {"status": "COMPLETED", "output": {"status": "prepared"}})[1])
    monkeypatch.setattr(wanctl.RunPod, "rest", lambda self, m, path, body=None, ok404=False: {"id": "tmp"})
    assert wanctl.main(["--config", str(p), "prepare"]) == 0
    assert sent[0]["input"]["hf_token"] == "hf_SECRET" and "hf_SECRET" not in capsys.readouterr().out
    monkeypatch.delenv("HF_TOKEN")
    p.write_text(yaml.safe_dump(cfg(huggingface={"token": "hf_FROMFILE"})))
    assert wanctl.main(["--config", str(p), "prepare", "--force"]) == 0
    assert sent[1]["input"]["hf_token"] == "hf_FROMFILE"
    assert "hf_FROMFILE" not in str(wanctl.template_payload(cfg(huggingface={"token": "hf_FROMFILE"})))


def test_gitignore_protects_secrets():
    ignored = open(os.path.join(ROOT, ".gitignore")).read().split()
    assert {"config.yaml", ".runpod_state.json", ".env"} <= set(ignored)
