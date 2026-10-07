"""Offline source and metadata checks for the SIWC reaper integration."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from benchmarks import chatgpt_plan_lme_v3 as bridge
from benchmarks import chatgpt_plan_responses_v8 as transport
from benchmarks import chatgpt_plan_responses_v7 as transport_v7
from benchmarks import chatgpt_plan_responses_v6 as transport_v6
from tools.diagnostics import siwc_lme_diagnostic_v5 as runner
from tools.diagnostics import siwc_lme_diagnostic_progress_v7 as reader


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


def test_bridge_default_is_actual_v8_complete_with_immutable_origins():
    assert bridge.transport is transport
    assert bridge.transport_v7 is transport_v7
    assert bridge.transport_v6 is transport_v6
    assert transport._v7 is transport_v7
    assert transport_v7._v6 is transport_v6
    assert bridge.SIWCLMEClient.__init__.__kwdefaults__["response_call"] is transport.complete
    assert transport.complete is not transport_v7.complete
    for relative in ("benchmarks/chatgpt_plan_responses_v8.py",
                     "benchmarks/chatgpt_plan_responses_v7.py",
                     "benchmarks/chatgpt_plan_responses_v6.py",
                     "benchmarks/chatgpt_plan_lme_v3.py"):
        assert runner.SIWC_PINS[relative] == digest(relative)


def test_runner_reader_source_identity_and_policy_are_exact():
    assert runner.SCHEMA == reader.RUN_SCHEMA == "siwc-lme-semantic-diagnostic-v5"
    assert reader.RUNNER_RELATIVE == runner.RUNNER_RELATIVE
    assert reader.RUNNER_SHA256 == digest(runner.RUNNER_RELATIVE)
    assert reader.source_constants(ROOT / runner.RUNNER_RELATIVE)["SIWC_PINS"] == runner.SIWC_PINS
    assert runner.ACCEPTED_MAP_SHA256 == reader.MAP_SHA256
    assert runner.ACCEPTED_INVENTORY_SHA256 == reader.INVENTORY_SHA256
    assert runner.MAX_LIMITS == {"campaign": (8012, 48_160_000, 14_400),
        "question": (2000, 12_000_000, 12_600), "canary": (12, 160_000, 600)}
    assert runner.ACCEPTED_FILES == 514
    assert reader.LIMITS["campaign"] == list(runner.MAX_LIMITS["campaign"])


def test_effective_producer_reports_v8_and_delegated_origins():
    declaration = runner._memory_client({}, object()).phase1_producer_declaration()
    effective = declaration.effective_request
    assert effective["transport_sha256"] == runner.SIWC_PINS["benchmarks/chatgpt_plan_responses_v8.py"]
    assert effective["transport_v7_sha256"] == runner.SIWC_PINS["benchmarks/chatgpt_plan_responses_v7.py"]
    assert effective["transport_v6_sha256"] == runner.SIWC_PINS["benchmarks/chatgpt_plan_responses_v6.py"]
    assert effective["bridge_sha256"] == runner.SIWC_PINS["benchmarks/chatgpt_plan_lme_v3.py"]


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
