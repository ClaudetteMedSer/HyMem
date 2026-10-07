"""Offline v2 integration checks using invented requests and responses."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from benchmarks import hermes_codex_responses_v1 as native_v1
from benchmarks import hermes_lme_oauth_v2 as bridge
from hymem.extraction.llm import LLMRequest
from tools.diagnostics import hermes_native_oauth_probe_v2 as probe


REPO = Path(__file__).resolve().parents[1]


def _offline_probe(tmp_path, monkeypatch, failure=None):
    loaded = {"warm": bridge.warm, "staged": bridge.staged_v6,
              "request_type": LLMRequest, "binary": Path("/unused"), "candidate": REPO}
    _, _, batch = probe.build_fixture(loaded)
    quota = bridge.warm.quota_metadata({"rateLimits": {
        "planType": "pro", "primary": {"usedPercent": 90},
        "credits": {"hasCredits": True, "unlimited": False, "balance": "1"}}})
    calls = []
    brokers = []

    class Broker:
        def __init__(self, *args, **kwargs):
            self.closed = False
            brokers.append(self)

        def admit(self, deadline):
            return bridge.native.Credentials("invented-token", "invented-account"), {
                "auth": "chatgpt", "model": "gpt-6-luna", "quota_windows": quota,
                "config_isolation_admitted": True, "inference_enabled": False}

        def close(self):
            self.closed = True

    def transport(credentials, system, user, schema, *, timeout):
        calls.append(schema)
        if failure is not None:
            raise bridge.native.TransportError(failure)
        text = "invented ordinary reply"
        if schema is not None:
            text = json.dumps({"schema": bridge.staged_v6.staged.ORIGINAL_SCHEMA,
                "batch_sha256": batch.batch_sha256, "complete": True,
                "originals": [{"index": 0, "original": {
                    "state": "not_established", "support": None}}]})
        return bridge.native.Completed(text, 9, 4, 13, 0, 0)

    fake = SimpleNamespace(AdmissionBroker=Broker,
        NativeLMEClient=lambda *args, **kwargs: bridge.NativeLMEClient(
            *args, **kwargs, transport=transport))
    monkeypatch.setattr(probe, "_live_containment", lambda receipt: True)
    monkeypatch.setattr(probe, "_resources", lambda receipt: (0, 0))
    return probe.run_probe(loaded, fake, tmp_path, {}), calls, brokers


def test_real_bridge_budget_and_staged_contract(tmp_path, monkeypatch):
    result, calls, brokers = _offline_probe(tmp_path, monkeypatch)
    assert probe.validate_result(result)
    assert result["status"] == "transport_verified"
    assert result["turns"] == 2 and result["known_tokens"] == 26
    assert result["usage_complete"] and result["reserved"] == result["in_flight"] == 0
    assert result["schema_acknowledged"] and result["staged_response_valid"]
    assert calls[0] is None and type(calls[1]) is dict
    assert all(b.closed for b in brokers)
    assert "invented ordinary reply" not in json.dumps(result)


@pytest.mark.parametrize("code", ["html_response", "access_challenge",
    "unsupported_media_type", "invalid_json"])
def test_v2_fault_is_finite_terminal_and_usage_unknown(tmp_path, monkeypatch, code):
    result, calls, brokers = _offline_probe(tmp_path, monkeypatch, failure=code)
    assert probe.validate_result(result)
    assert result["status"] == "accounting_unverified"
    assert len(calls) == 1 and all(b.closed for b in brokers)
    assert result["turns"] == 1 and result["known_tokens"] == 0
    assert not result["usage_complete"]
    assert result["first_failure"] == {"code": code, "phase": "http",
        "turn_admitted": True, "known_usage": False}
    assert result["native_summary"]["first_failure"] == result["first_failure"]


def test_receipt_pins_both_native_sources_and_v2_bridge(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-lme-diagnostic-invented123"
    monkeypatch.setattr(probe, "_root", lambda path: path)
    monkeypatch.setattr(probe, "_tool", lambda path: REPO / "tools/diagnostics/hermes_native_oauth_probe_v2.py")
    runner = SimpleNamespace(PINS={}, DIAGNOSTIC_HELPER_SHA256="a" * 64,
        ACCEPTED_MAP_SHA256="b" * 64, DATASET_SHA256="c" * 64)
    launcher = SimpleNamespace(_load_runner=lambda path: runner)
    receipt = probe.receipt_for(root, launcher)
    pins = receipt["source_sha256"]
    assert receipt["schema"] == probe.SCHEMA
    assert receipt["transport"] == "legacy_codex_direct_responses_v2"
    assert pins["benchmarks/hermes_codex_responses_v1.py"] == probe.NATIVE_V1_SHA
    assert pins["benchmarks/hermes_codex_responses_v2.py"] == probe.NATIVE_SHA
    assert pins["benchmarks/hermes_lme_oauth_v2.py"] == probe.BRIDGE_SHA
    assert pins[probe.TOOL] == probe._sha(REPO / "tools/diagnostics/hermes_native_oauth_probe_v2.py")
    assert Path(bridge.native.v1.__file__).resolve() == REPO / "benchmarks/hermes_codex_responses_v1.py"
    assert bridge.native.v1 is native_v1


def test_one_shot_service_policy_and_caps_stay_bounded(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-lme-diagnostic-invented123"
    monkeypatch.setattr(probe, "_root", lambda path: path)
    command = probe.command(root, {"unit": probe.unit_for(root)}, "a" * 64)
    for property_name in ("Restart=no", "KillMode=control-group", "OOMPolicy=kill",
        "RuntimeMaxSec=330s", "TimeoutStopSec=10s", "MemoryMax=4294967296",
        "CPUQuota=200%", "TasksMax=256"):
        assert "--property=" + property_name in command
    assert command[-5:] == [str(root / probe.TOOL), "--run-root", str(root),
                            "--receipt-sha256", "a" * 64]
    assert probe.LIMITS == (2, 160_000, 300)
    marker = tmp_path / "one-shot-marker.json"
    probe._write_once(marker, {"one_shot": True})
    with pytest.raises(FileExistsError):
        probe._write_once(marker, {"one_shot": True})


@pytest.mark.parametrize("tamper", ["benchmarks/hermes_codex_responses_v1.py",
    "benchmarks/hermes_codex_responses_v2.py", "benchmarks/hermes_lme_oauth_v2.py"])
def test_source_drift_fails_closed_before_import(tmp_path, monkeypatch, tamper):
    root = tmp_path / ".hymem-lme-diagnostic-invented123"
    code = root / "code/benchmarks"
    code.mkdir(parents=True)
    for name in ("hermes_codex_responses_v1.py", "hermes_codex_responses_v2.py",
                 "hermes_lme_oauth_v2.py"):
        shutil.copyfile(REPO / "benchmarks" / name, code / name)
    with (root / "code" / tamper).open("ab") as target:
        target.write(b"\n# invented drift\n")
    monkeypatch.setattr(probe, "_root", lambda path: path)
    monkeypatch.setattr(probe, "_tool", lambda path: REPO / "tools/diagnostics/hermes_native_oauth_probe_v2.py")
    warm = bridge.warm
    loaded = {"warm": warm, "staged": bridge.staged_v6}
    fake_launcher = SimpleNamespace(HOST_ROOT=probe.HOST_ROOT, RUNNER_SHA256=probe.RUNNER_SHA,
        INVENTORY_SHA256=probe.INVENTORY_SHA, BINARY_SHA256=probe.BINARY_SHA,
        verify_sources=lambda path: (SimpleNamespace(__file__=str(
            root / "code/tools/diagnostics/luna_lme_diagnostic_v10.py")), loaded))
    monkeypatch.setattr(probe, "_launcher", lambda path: fake_launcher)
    with pytest.raises(ValueError, match="^native_source_drift$"):
        probe.verify_sources(root)
