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
    c["image"]["name"] = "ghcr.io/me/wan22-runpod-worker:0.1.0"
    c["worker_env"]["BUCKET_SECRET_ACCESS_KEY"] = "s3cr3t"
    c.update(over)
    return c


def test_endpoint_payload_matches_rest_schema_fields():
    body = wanctl.endpoint_payload(cfg(), "tpl1", "vol1")
    assert body["templateId"] == "tpl1" and body["networkVolumeId"] == "vol1"
    assert body["dataCenterIds"] == ["EU-RO-1"] and body["gpuTypeIds"][0] == "NVIDIA A100 80GB PCIe"
    assert body["executionTimeoutMs"] == 900_000 and body["idleTimeout"] == 5 and body["workersMin"] == 0
    assert body["minCudaVersion"] == "13.0" and body["scalerType"] == "QUEUE_DELAY"


def test_template_payload_drops_empty_env_and_sets_volume_path():
    body = wanctl.template_payload(cfg())
    assert body["isServerless"] is True and body["containerDiskInGb"] == 100
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
    assert size == (720, 1280)
    assert Image.open(io.BytesIO(base64.b64decode(b64))).size == (720, 1280)
    assert len(b64) < 1_000_000


def test_cost_estimate_includes_cold_start_only_when_reported():
    c = cfg()
    warm = wanctl.estimate_cost(c, {"gpu": "NVIDIA A100 80GB PCIe", "timings": {}}, 120_000)
    cold = wanctl.estimate_cost(c, {"gpu": "NVIDIA A100 80GB PCIe", "timings": {"cold_start_s": 180}}, 120_000)
    assert warm["billed_s_est"] == 125.0 and warm["usd_est"] == round(125 * 2.72 / 3600, 4)
    assert cold["billed_s_est"] == 305.0
    assert wanctl.estimate_cost(c, {"gpu": "Mystery GPU"}, 1000)["usd_est"] is None


def test_gitignore_protects_secrets():
    ignored = open(os.path.join(ROOT, ".gitignore")).read().split()
    assert {"config.yaml", ".runpod_state.json", ".env"} <= set(ignored)
