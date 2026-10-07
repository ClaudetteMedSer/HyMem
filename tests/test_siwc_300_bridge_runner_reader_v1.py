"""Offline source and metadata checks for the approved 300-second SIWC chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import chatgpt_plan_lme_v5 as bridge
from benchmarks import chatgpt_plan_responses_v10 as transport
from benchmarks import chatgpt_plan_responses_v6 as transport_v6
from tools.diagnostics import siwc_lme_diagnostic_v7 as runner
from tools.diagnostics import siwc_lme_diagnostic_progress_v9 as reader


ROOT = Path(__file__).resolve().parents[1]


def digest(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def observation() -> dict:
    return {"child_phase": "stream_read", "parent_timeout_site": "result_wait",
        "last_progress_elapsed_ms": 1, "parent_elapsed_ms": 999,
        "elapsed_saturated": False, "snapshot_valid": True,
        "timeout_allowance_ms": 1000, "wire_bytes": 90, "event_count": 1,
        "completion_seen": True, "result_ready": False,
        "result_ipc_started": False, "child_alive_when_sampled": True}


def test_bridge_default_is_actual_v10_complete_with_immutable_origins():
    assert bridge.transport is transport
    assert bridge.transport_v6 is transport_v6
    assert transport._v6 is transport_v6
    assert bridge.SIWCLMEClient.__init__.__kwdefaults__["response_call"] is transport.complete
    assert transport.MAX_WALL_SECONDS == bridge.MAX_INVOCATION == 300.0
    for relative in ("benchmarks/chatgpt_plan_responses_v10.py",
                     "benchmarks/chatgpt_plan_responses_v6.py",
                     "benchmarks/chatgpt_plan_lme_v5.py"):
        assert runner.SIWC_PINS[relative] == digest(relative)
    assert "benchmarks/chatgpt_plan_responses_v8.py" not in runner.SIWC_PINS
    assert "benchmarks/chatgpt_plan_responses_v7.py" not in runner.SIWC_PINS
    assert "benchmarks/chatgpt_plan_responses_v9.py" not in runner.SIWC_PINS


def test_runner_reader_source_identity_and_policy_are_exact():
    assert runner.SCHEMA == reader.RUN_SCHEMA == "siwc-lme-semantic-diagnostic-v7"
    assert reader.RUNNER_RELATIVE == runner.RUNNER_RELATIVE
    assert reader.RUNNER_SHA256 == digest(runner.RUNNER_RELATIVE)
    assert reader.LAUNCHER_NAME == "siwc_lme_diagnostic_launch_v6.py"
    assert reader.LAUNCHER_SHA256 == digest("tools/diagnostics/siwc_lme_diagnostic_launch_v6.py")
    assert reader.source_constants(ROOT / runner.RUNNER_RELATIVE)["SIWC_PINS"] == runner.SIWC_PINS
    assert runner.ACCEPTED_MAP_SHA256 == reader.MAP_SHA256
    assert runner.ACCEPTED_INVENTORY_SHA256 == reader.INVENTORY_SHA256
    assert runner.MAX_LIMITS == {"campaign": (8012, 48_160_000, 14_400),
        "question": (2000, 12_000_000, 12_600), "canary": (12, 160_000, 600)}
    assert runner.ACCEPTED_FILES == 514
    assert reader.LIMITS["campaign"] == list(runner.MAX_LIMITS["campaign"])
    assert len(runner.PINS) + len(runner.SIWC_PINS) + 2 == 26


def test_effective_producer_reports_v10_and_direct_v6_origin():
    declaration = runner._memory_client({}, object()).phase1_producer_declaration()
    effective = declaration.effective_request
    assert effective["transport_sha256"] == runner.SIWC_PINS["benchmarks/chatgpt_plan_responses_v10.py"]
    assert effective["transport_v6_sha256"] == runner.SIWC_PINS["benchmarks/chatgpt_plan_responses_v6.py"]
    assert effective["bridge_sha256"] == runner.SIWC_PINS["benchmarks/chatgpt_plan_lme_v5.py"]
    assert "transport_v7_sha256" not in effective


def test_manifest_serializes_only_pinned_direct_transport_closure():
    strictness = SimpleNamespace(content_hash=lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    loaded = {"strictness": strictness,
              "diagnostic": SimpleNamespace(MODE="semantic_diagnostic_v1")}
    questions = [{"question_id": f"invented-{i}"} for i in range(4)]
    limits = {name: dict(zip(("turns", "known_tokens", "seconds"), value))
              for name, value in runner.MAX_LIMITS.items()}
    limits.update(indexing_seconds=10_800, workers=4)
    manifest = runner._identity_manifest(loaded, questions, limits,
        runner.DIAGNOSTIC_HELPER_SHA256)
    assert manifest["schema"] == reader.RUN_SCHEMA
    assert manifest["transport_sha256"] == runner.SIWC_PINS["benchmarks/chatgpt_plan_responses_v10.py"]
    assert manifest["transport_v6_sha256"] == runner.SIWC_PINS["benchmarks/chatgpt_plan_responses_v6.py"]
    assert manifest["bridge_sha256"] == runner.SIWC_PINS["benchmarks/chatgpt_plan_lme_v5.py"]
    assert "transport_v7_sha256" not in manifest
    assert manifest["expected_count"] == 4 and manifest["rerolls"] == 0
    assert manifest["limits"]["campaign"] == {"turns": 8012,
        "known_tokens": 48_160_000, "seconds": 14_400}


def test_timeout_first_fault_retains_unknown_usage_and_finite_snapshot():
    exc = transport.TransportError("timeout", timeout_observation=observation())
    projected = bridge._safe_observation(exc, "timeout")
    assert projected == {"timeout_observation": observation()}
    fault = {"code": "timeout", "phase": "http", "turn_admitted": True,
             "unknown_usage": True, **projected}
    assert bridge.project_first_failure(fault) == fault
    assert reader.first_failure(fault) == fault
    invalid = {**observation(), "wire_bytes": True}
    with pytest.raises(ValueError, match="timeout_observation_invalid"):
        reader.timeout_observation(invalid)


def test_approved_elapsed_range_is_finite_and_reader_matches_transport():
    sample = {**observation(), "last_progress_elapsed_ms": 250_000,
              "parent_elapsed_ms": 250_100, "timeout_allowance_ms": 300_000}
    assert transport.sanitize_timeout_observation(sample) == sample
    assert reader.timeout_observation(sample) is None
    for field, invalid in (("last_progress_elapsed_ms", 301_001),
                           ("parent_elapsed_ms", 301_001),
                           ("timeout_allowance_ms", 300_001)):
        changed = {**sample, field: invalid}
        assert transport.sanitize_timeout_observation(changed) is None
        with pytest.raises(ValueError, match="timeout_observation_invalid"):
            reader.timeout_observation(changed)
