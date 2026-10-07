"""Root-owned real fake-wire controls for the long-lived LME observer."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from benchmarks import codex_subscription_timeout_v3 as observer
from hymem.extraction.llm import LLMRequest


def wire(monkeypatch, *, fail_at=None, turns=64, tokens=100000):
    path = Path(__file__).with_name("test_luna_warm_sequence_root.py")
    spec = importlib.util.spec_from_file_location("root_lme_observer_wire", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    monkeypatch.setattr(fixture, "observer", observer)
    factory, clients, processes, called = fixture.setup(monkeypatch, fail_at=fail_at)
    cap = observer.BudgetLimits(turns, tokens, 600)
    budget = observer.SharedBudget(cap)
    client = factory("unused", budget, "q", cap, max_requests=16, max_age_seconds=300)
    request = LLMRequest(system="invented system", user=fixture.PRIVATE,
                         temperature=0.0, max_tokens=32, response_format="text")
    return fixture, client, budget, processes, called, request


def test_thirty_three_calls_rotate_without_whole_client_cap(monkeypatch):
    fixture, client, budget, processes, calls, request = wire(monkeypatch)
    try:
        for _ in range(33):
            assert client.complete(request) == fixture.PRIVATE
        summary = client.diagnostic_summary()
        assert summary["calls"] == summary["successes"] == 33
        assert summary["failures"] == 0 and summary["first_failure"] is None
        assert summary["counts_saturated"] is False
        assert summary["last_record"]["known_usage"] is True
        assert summary["last_record"]["precleanup"] is None
        assert summary["last_record"]["deadline_monotonic"] > summary["last_record"]["invocation_start_monotonic"]
        assert len(client.diagnostic_records()) == 16
        assert len(processes) == 3 and client.rotations == 2
        assert budget.snapshot()["turns"] == 33
        assert budget.snapshot()["known_tokens"] == 33 * 19
        assert budget.snapshot()["usage_complete"] is True
        assert len(calls) == 33
        assert fixture.PRIVATE not in json.dumps(summary)
        assert "invented-thread" not in json.dumps(summary)
        assert len(json.dumps(summary)) < 10000
        summary["last_record"]["phase_seconds"]["events"] = -1
        assert client.diagnostic_summary()["last_record"]["phase_seconds"]["events"] >= 0
    finally:
        client.close()
    assert client.session is None and client.directory is None
    assert budget.snapshot()["in_flight"] == budget.snapshot()["reserved"] == 0


def test_timeout_seventeen_survives_tail_eviction_and_keeps_usage_unknown(monkeypatch):
    fixture, client, budget, processes, calls, request = wire(monkeypatch, fail_at=17)
    try:
        for _ in range(16):
            client.complete(request)
        with pytest.raises(observer.ConcurrentStop, match="timeout"):
            client.complete(request)
        original = client.diagnostic_summary()["first_failure"]
        assert original["call_index"] == 17
        assert original["record"]["failure_code"] == "timeout"
        assert original["record"]["known_usage"] is False
        assert original["record"]["precleanup"] is not None
        assert original["record"]["deadline_monotonic"] > original["record"]["invocation_start_monotonic"]
        for _ in range(18):
            with pytest.raises(observer.ConcurrentStop):
                client.complete(request)
        summary = client.diagnostic_summary()
        assert summary["first_failure"] == original
        assert summary["calls"] == 35 and summary["failures"] == 19
        assert len(client.diagnostic_records()) == 16 and len(calls) == 17
        snapshot = budget.snapshot()
        assert snapshot["turns"] == 17 and snapshot["known_tokens"] == 16 * 19
        assert snapshot["usage_complete"] is False
        assert snapshot["first_failure"]["code"] == "timeout"
        assert len(processes) == 2
        original["record"]["failure_code"] = "private-data"
        assert client.diagnostic_summary()["first_failure"]["record"]["failure_code"] == "timeout"
        assert fixture.PRIVATE not in json.dumps(summary)
    finally:
        client.close()


def test_budget_still_stops_before_seventeenth_admission(monkeypatch):
    _, client, budget, processes, calls, request = wire(monkeypatch, turns=16)
    try:
        for _ in range(16):
            client.complete(request)
        with pytest.raises(observer.ConcurrentStop):
            client.complete(request)
        assert len(calls) == budget.snapshot()["turns"] == 16
        assert len(processes) == 1
        assert budget.snapshot()["usage_complete"] is True
        assert client.diagnostic_summary()["calls"] == 17
        rejected = client.diagnostic_summary()["last_record"]
        assert rejected["deadline_monotonic"] is None and rejected["observed"] is None
    finally:
        client.close()


@pytest.mark.parametrize("field,bad", [("calls", True), ("successes", 1.0),
    ("failures", -1), ("counts_saturated", 0), ("timing_saturated", 1)])
def test_summary_fails_closed_on_noncanonical_types(monkeypatch, field, bad):
    _, client, _, _, _, request = wire(monkeypatch)
    try:
        client.complete(request)
        summary = deepcopy(client.diagnostic_summary())
        summary[field] = bad
        assert observer._project_summary(summary) is None
        summary = deepcopy(client.diagnostic_summary())
        summary["private"] = "PRIVATE"
        assert observer._project_summary(summary) is None
    finally:
        client.close()


def test_summary_cannot_use_saturation_to_hide_impossible_counts(monkeypatch):
    _, client, _, _, _, request = wire(monkeypatch)
    try:
        client.complete(request)
        valid = client.diagnostic_summary()
        for field in ("counts_saturated", "timing_saturated"):
            forged = deepcopy(valid)
            forged[field] = True
            assert observer._project_summary(forged) is None
        forged = deepcopy(valid)
        forged["timing_seconds"]["total"] = float("nan")
        assert observer._project_summary(forged) is None
        forged = deepcopy(valid)
        forged["last_record"]["prompt"] = "PRIVATE"
        assert observer._project_summary(forged) is None
    finally:
        client.close()
