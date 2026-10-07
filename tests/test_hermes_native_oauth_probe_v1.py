"""Offline controls for the one-shot native OAuth compatibility gate."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from benchmarks import codex_subscription_staged_v6 as staged
from benchmarks import hermes_lme_oauth_v1 as bridge
from hymem.extraction.llm import LLMRequest
from tools.diagnostics import hermes_native_oauth_probe_v1 as probe


ROOT = Path(__file__).resolve().parents[1]


class MetadataSession:
    calls = []
    quota_denied = False

    def __init__(self, binary, cwd, timeout):
        self.created_at = time.monotonic()
        self.closed = False

    def set_deadline(self, deadline):
        assert deadline > time.monotonic()

    def send(self, method, params, notification=False):
        self.calls.append(method)
        assert method == "initialized" and notification

    def rpc(self, method, params):
        self.calls.append(method)
        if method == "account/read":
            return {"account": {"type": "chatgpt", "planType": "pro",
                                "email": "invented@example.invalid"}}
        if method == "model/list":
            return {"data": [{"model": "gpt-6-luna", "supportedReasoningEfforts":
                [{"reasoningEffort": "low"}]}], "nextCursor": None}
        if method == "account/rateLimits/read":
            return {"rateLimits": {"planType": "pro", "primary": {"usedPercent": 99},
                "spendControlReached": self.quota_denied,
                "credits": {"hasCredits": True, "unlimited": False, "balance": "2.5"}}}
        assert method == "initialize"
        return {}

    def close(self):
        self.closed = True


def _auth(path):
    claims = {"exp": time.time() + 3600, "email": "invented@example.invalid",
              "https://api.openai.com/auth": {"chatgpt_account_id": "invented-account"}}
    token = "header." + base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=") + ".sig"
    path.write_text(json.dumps({"auth_mode": "chatgpt", "OPENAI_API_KEY": None,
                                "tokens": {"access_token": token, "account_id": "invented-account"}}))
    path.chmod(0o600)


def _loaded():
    return {"warm": staged.warm, "staged": staged, "candidate": ROOT,
            "binary": Path("/unused"), "request_type": LLMRequest}


def _stage_response(batch, *, correct=True):
    return json.dumps({"schema": staged.staged.ORIGINAL_SCHEMA,
        "batch_sha256": batch.batch_sha256 if correct else "0" * 64,
        "complete": True, "originals": [{"index": 0,
            "original": {"state": "not_established", "support": None}}]})


def _wire(tmp_path, monkeypatch, *, denied=False, incorrect=False):
    auth = tmp_path / "auth.json"
    _auth(auth)
    MetadataSession.calls = []
    MetadataSession.quota_denied = denied
    loaded = _loaded()
    _, _, batch = probe.build_fixture(loaded)
    calls = []

    def transport(credentials, system, user, schema, *, timeout):
        calls.append({"schema": schema, "timeout": timeout})
        text = "invented ready" if schema is None else _stage_response(batch, correct=not incorrect)
        return bridge.native.Completed(text, 9, 4, 13, 3, 2)

    fake_bridge = SimpleNamespace(
        AdmissionBroker=lambda binary, path, **kwargs: bridge.AdmissionBroker(
            binary, str(auth), session_factory=MetadataSession, **kwargs),
        NativeLMEClient=lambda broker, budget, key, limits: bridge.NativeLMEClient(
            broker, budget, key, limits, transport=transport))
    monkeypatch.setattr(probe, "_live_containment", lambda receipt: True)
    monkeypatch.setattr(probe, "_resources", lambda receipt: (0, 0))
    return loaded, fake_bridge, calls


def test_fixture_is_original_staged_v6_and_bound_to_invented_source():
    ordinary, request, batch = probe.build_fixture(_loaded())
    assert ordinary.response_format == "text"
    assert batch.sources[0].content == probe.SOURCE_TEXT
    assert staged.staged.validate_original_request(request, batch) is None
    schema = staged.staged.build_original_output_schema(batch)
    assert schema["properties"]["batch_sha256"]["enum"] == [batch.batch_sha256]
    assert staged.staged.parse_original_response(_stage_response(batch), batch).batch_sha256 == batch.batch_sha256


def test_two_sequential_calls_reconcile_and_never_export_text(tmp_path, monkeypatch):
    loaded, fake_bridge, calls = _wire(tmp_path, monkeypatch)
    result = probe.run_probe(loaded, fake_bridge, tmp_path, {})
    assert probe.validate_result(result)
    assert result["status"] == "transport_verified"
    assert result["turns"] == 2 and result["known_tokens"] == 26
    assert result["reserved"] == result["in_flight"] == 0
    assert result["native_summary"]["successes"] == 2
    assert calls[0]["schema"] is None and calls[1]["schema"] is not None
    assert len(calls) == 2 and all(0 < call["timeout"] <= 120 for call in calls)
    assert not any(name.startswith("thread/") or name.startswith("turn/") for name in MetadataSession.calls)
    serialized = json.dumps(result)
    assert "invented ready" not in serialized and probe.SOURCE_TEXT not in serialized


def test_quota_denial_stops_before_http_and_preserves_native_failure(tmp_path, monkeypatch):
    loaded, fake_bridge, calls = _wire(tmp_path, monkeypatch, denied=True)
    result = probe.run_probe(loaded, fake_bridge, tmp_path, {})
    assert probe.validate_result(result)
    assert result["status"] != "transport_verified"
    assert result["turns"] == 0 and calls == []
    assert result["first_failure"]["code"] == "quota_unverified"
    assert result["first_failure"]["phase"] == "admission"


def test_structured_binding_failure_is_terminal_after_two_settled_calls(tmp_path, monkeypatch):
    loaded, fake_bridge, calls = _wire(tmp_path, monkeypatch, incorrect=True)
    result = probe.run_probe(loaded, fake_bridge, tmp_path, {})
    assert probe.validate_result(result)
    assert result["status"] == "failed" and result["turns"] == 2
    assert result["known_tokens"] == 26 and len(calls) == 2
    assert result["staged_completed"] and not result["staged_response_valid"]
    assert result["first_failure"] == {"code": "stage_response_invalid", "phase": "structured_output"}


def test_result_validator_rejects_forged_success_and_nonfinite_summary(tmp_path, monkeypatch):
    loaded, fake_bridge, _ = _wire(tmp_path, monkeypatch)
    result = probe.run_probe(loaded, fake_bridge, tmp_path, {})
    assert probe.validate_result(result)
    bad = dict(result, turns=1)
    assert not probe.validate_result(bad)
    bad = dict(result, first_failure={"code": "secret", "phase": "http"})
    assert not probe.validate_result(bad)
    bad_summary = dict(result["native_summary"], timing_seconds={"total": float("nan"),
        "admission": 0.0, "http": 0.0})
    assert not probe.validate_result(dict(result, native_summary=bad_summary))


def test_command_is_one_shot_bounded_unit_with_distinct_name(monkeypatch):
    root = Path("/home/atta/.hymem-lme-diagnostic-invented123")
    monkeypatch.setattr(probe, "_root", lambda value: value)
    unit = probe.unit_for(root)
    cmd = probe.command(root, {"unit": unit}, "a" * 64)
    assert unit.startswith("hymem-luna-native-oauth-probe-")
    assert "--property=Restart=no" in cmd
    assert "--property=KillMode=control-group" in cmd
    assert "--property=RuntimeMaxSec=330s" in cmd
    assert "--property=TimeoutStopSec=10s" in cmd
    assert "--property=MemoryMax=4294967296" in cmd
    assert "--property=CPUQuota=200%" in cmd
    assert "--property=TasksMax=256" in cmd
    assert "--property=OOMPolicy=kill" in cmd
    assert cmd[-5:] == [str(root / probe.TOOL), "--run-root", str(root),
                        "--receipt-sha256", "a" * 64]


def test_execution_marker_is_exclusive(tmp_path):
    path = tmp_path / "marker.json"
    probe._write_once(path, {"one_shot": True})
    with pytest.raises(FileExistsError):
        probe._write_once(path, {"one_shot": True})
