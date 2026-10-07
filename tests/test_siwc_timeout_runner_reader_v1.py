"""Offline runner and metadata-reader controls for the finite SIWC timeout path."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from benchmarks import chatgpt_plan_lme_v2 as bridge
from benchmarks import chatgpt_plan_responses_v7 as transport
from tools.diagnostics import siwc_lme_diagnostic_v4 as runner
from tools.diagnostics import siwc_lme_diagnostic_progress_v6 as reader


ROOT = Path(__file__).resolve().parents[1]


def observation():
    return {"child_phase": "stream_read", "parent_timeout_site": "result_wait",
        "last_progress_elapsed_ms": 1, "parent_elapsed_ms": 999,
        "elapsed_saturated": False, "snapshot_valid": True,
        "timeout_allowance_ms": 1000, "wire_bytes": 90, "event_count": 1,
        "completion_seen": True, "result_ready": False,
        "result_ipc_started": False, "child_alive_when_sampled": True}


def failure(**changes):
    value = {"code": "timeout", "phase": "http", "turn_admitted": True,
        "unknown_usage": True, "timeout_observation": observation()}
    value.update(changes)
    return value


def test_new_source_binding_retains_parser_and_candidate_pins():
    assert runner.RUNNER_RELATIVE.endswith("siwc_lme_diagnostic_v4.py")
    assert reader.RUNNER_RELATIVE == runner.RUNNER_RELATIVE
    assert reader.RUNNER_SHA256 == hashlib.sha256(
        (ROOT / runner.RUNNER_RELATIVE).read_bytes()).hexdigest()
    assert runner.SIWC_PINS["benchmarks/chatgpt_plan_responses_v7.py"] == hashlib.sha256(
        (ROOT / "benchmarks/chatgpt_plan_responses_v7.py").read_bytes()).hexdigest()
    assert runner.SIWC_PINS["benchmarks/chatgpt_plan_lme_v2.py"] == hashlib.sha256(
        (ROOT / "benchmarks/chatgpt_plan_lme_v2.py").read_bytes()).hexdigest()
    assert runner.SIWC_PINS["benchmarks/chatgpt_plan_responses_v6.py"] == hashlib.sha256(
        (ROOT / "benchmarks/chatgpt_plan_responses_v6.py").read_bytes()).hexdigest()
    assert runner.CANDIDATE_PINS and runner.ACCEPTED_FILES == 514
    assert runner.MAX_LIMITS == {"campaign": (8012, 48_160_000, 14_400),
        "question": (2000, 12_000_000, 12_600), "canary": (12, 160_000, 600)}


def test_transport_to_bridge_to_reader_boundary_propagates_snapshot():
    exc = transport.TransportError("timeout", timeout_observation=observation())
    projected = bridge._safe_observation(exc, "timeout")
    assert projected == {"timeout_observation": observation()}
    candidate = failure(**projected)
    assert bridge.project_first_failure(candidate) == candidate
    assert reader.first_failure(candidate) == candidate


@pytest.mark.parametrize("mutation", [
    {"child_phase": "private_content"}, {"wire_bytes": True},
    {"wire_bytes": 16_000_001}, {"event_count": 16_000_000 // 9 + 2},
    {"snapshot_valid": False}, {"result_ipc_started": True},
    {"parent_timeout_site": "none"},
])
def test_reader_rejects_every_snapshot_rejected_by_transport(mutation):
    item = observation()
    item.update(mutation)
    if mutation == {"parent_timeout_site": "none"}:
        assert item["child_alive_when_sampled"] is True
    assert transport.sanitize_timeout_observation(item) is None
    with pytest.raises(ValueError, match="timeout_observation_invalid"):
        reader.timeout_observation(item)


@pytest.mark.parametrize("changes", [
    {"phase": "admission", "turn_admitted": False, "unknown_usage": False},
    {"code": "quota_failure"},
    {"timeout_observation": {**observation(), "request_body": "secret"}},
    {"timeout_observation": None},
])
def test_reader_rejects_malformed_or_misplaced_timeout(changes):
    with pytest.raises(ValueError):
        reader.first_failure(failure(**changes))


def test_underlying_resource_timeout_wrapper_is_accepted_only_when_finite():
    wrapped = failure(code="resource_task_denial", underlying_code="timeout",
        resource_observation={"current": 1, "peak": 1, "limit": 256, "denials": 1})
    assert bridge.project_first_failure(wrapped) == wrapped
    assert reader.first_failure(wrapped) == wrapped
    broken = copy.deepcopy(wrapped)
    broken["timeout_observation"]["wire_bytes"] = -1
    with pytest.raises(ValueError):
        reader.first_failure(broken)


def test_summary_revalidates_nested_timeout():
    item = {"schema": "siwc_lme_summary_v2", "calls": 1, "successes": 0,
        "failures": 1, "internal_http_attempts": 1, "provider_internal_retries_known": False,
        "admitted_turns": 1, "known_tokens": 0, "usage_complete": False,
        "timing_seconds": {"total": 1.0, "admission": 0.0, "http": 1.0},
        "timing_saturated": False, "first_failure": failure(),
        "last_failure_code": "timeout"}
    assert reader.summary(item) == item
    item["first_failure"]["timeout_observation"]["event_count"] = True
    with pytest.raises(ValueError):
        reader.summary(item)


def test_pilot_views_revalidate_nested_timeout():
    empty = {"schema": "siwc_lme_summary_v2", "calls": 0, "successes": 0,
        "failures": 0, "internal_http_attempts": 0,
        "provider_internal_retries_known": False, "admitted_turns": 0,
        "known_tokens": 0, "usage_complete": True,
        "timing_seconds": {"total": 0.0, "admission": 0.0, "http": 0.0},
        "timing_saturated": False, "first_failure": None, "last_failure_code": None}
    ids = ["canary", *(f"q-{index:04d}" for index in range(4))]
    rows = [{"question_id": qid, "ordinary": copy.deepcopy(empty),
        "structured": copy.deepcopy(empty),
        "ledger": {"admitted_turns": 0, "known_tokens": 0,
                   "usage_complete": True}} for qid in ids]
    projection = {"schema": "siwc_lme_pilot_projection_v2", "canary": rows[0],
        "questions": rows[1:], "aggregate": {key: 0 for key in
            ("calls", "successes", "failures", "internal_http_attempts",
             "admitted_turns", "known_tokens")}}
    observations = {f"{prefix}.{route}": {"status": "observed", "summary": view[route]}
        for prefix, view in [("canary", rows[0]), *(
            (f"question.{index}", row) for index, row in enumerate(rows[1:]))]
        for route in ("ordinary", "structured")}
    budget = {"turns": 0, "known_tokens": 0,
        "questions": {qid: {"turns": 0, "known_tokens": 0} for qid in ids}}
    assert reader.pilot(projection, observations, budget) == projection
    broken = copy.deepcopy(projection)
    broken["questions"][1]["structured"]["first_failure"] = failure()
    with pytest.raises(ValueError):
        reader.pilot(broken, observations, budget)
