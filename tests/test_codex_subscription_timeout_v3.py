"""Offline controls for the bounded long-lived timeout observer."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from benchmarks import codex_subscription_timeout_v3 as observer
from hymem.extraction.llm import LLMRequest


def _wire(monkeypatch, *, turns=64, fail_at=None, rotate_after=None):
    path = Path(__file__).with_name("test_luna_warm_sequence_root.py")
    spec = importlib.util.spec_from_file_location("timeout_v3_sequence_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fixture)
    monkeypatch.setattr(fixture, "observer", observer)
    factory, _, processes, called = fixture.setup(
        monkeypatch, fail_at=fail_at, rotate_after=rotate_after)
    limits = observer.BudgetLimits(turns, 100_000, 600)
    budget = observer.SharedBudget(limits)
    client = factory("unused", budget, "q", limits)
    request = LLMRequest(system="invented system", user=fixture.PRIVATE,
                         temperature=0.0, max_tokens=32, response_format="text")
    return fixture, client, budget, processes, called, request


def test_tail_rotates_and_age_rotation_never_copies_old_invocation(monkeypatch):
    fixture, client, budget, processes, called, request = _wire(monkeypatch, rotate_after=1)
    try:
        for _ in range(18):
            assert client.complete(request) == fixture.PRIVATE
        records = client.diagnostic_records()
        assert len(records) == observer.MAX_RECORDS
        assert client.diagnostic_summary()["calls"] == 18
        assert client.diagnostic_summary()["successes"] == 18
        assert len(processes) >= 2 and called == list(range(18))
        assert all(record["precleanup"] is None for record in records)
        assert all(record["deadline_monotonic"] >= record["invocation_start_monotonic"]
                   for record in records)
        assert budget.snapshot()["turns"] == 18
        assert fixture.PRIVATE not in json.dumps(client.diagnostic_summary())
    finally:
        client.close()


def test_first_failure_retained_after_eviction_with_unknown_usage(monkeypatch):
    fixture, client, budget, _, _, request = _wire(monkeypatch, fail_at=17)
    try:
        for _ in range(16):
            client.complete(request)
        with pytest.raises(observer.ConcurrentStop, match="timeout"):
            client.complete(request)
        first = deepcopy(client.diagnostic_summary()["first_failure"])
        assert first["call_index"] == 17
        assert first["record"]["known_usage"] is False
        assert first["record"]["precleanup"] is not None
        for _ in range(17):
            with pytest.raises(observer.ConcurrentStop):
                client.complete(request)
        assert len(client.diagnostic_records()) == 16
        assert client.diagnostic_summary()["first_failure"] == first
        assert budget.snapshot()["usage_complete"] is False
        assert fixture.PRIVATE not in json.dumps(client.diagnostic_summary())
    finally:
        client.close()


def test_summary_projection_is_strict_detached_and_finite(monkeypatch):
    _, client, _, _, _, request = _wire(monkeypatch)
    try:
        client.complete(request)
        clean = client.diagnostic_summary()
        changed = deepcopy(clean)
        changed["last_record"]["observed"]["private"] = "secret"
        assert observer._project_summary(changed) is None
        assert "private" not in json.dumps(client.diagnostic_summary())
        cases = [
            dict(clean, calls=True),
            dict(clean, successes=1.0),
            dict(clean, counts_saturated=True),
            dict(clean, timing_saturated=True),
            dict(clean, private="secret"),
        ]
        for case in cases:
            assert observer._project_summary(case) is None
        changed = deepcopy(clean)
        changed["timing_seconds"]["total"] = float("nan")
        assert observer._project_summary(changed) is None
        changed = deepcopy(clean)
        changed["timing_seconds"]["phases"]["events"] = True
        assert observer._project_summary(changed) is None
        changed = deepcopy(clean)
        changed["timing_seconds"]["phases"]["extra"] = 1
        assert observer._project_summary(changed) is None
    finally:
        client.close()


def test_saturating_counters_and_timing_do_not_admit_or_reject(monkeypatch):
    _, client, budget, _, _, request = _wire(monkeypatch)
    try:
        client.complete(request)
        cap = observer.MAX_SUMMARY_COUNT
        client._calls = client._successes = cap
        client._timing_seconds["total"] = observer.MAX_SUMMARY_SECONDS
        client.complete(request)
        summary = client.diagnostic_summary()
        assert summary["calls"] == summary["successes"] == cap
        assert summary["counts_saturated"] is True
        assert summary["timing_seconds"]["total"] == observer.MAX_SUMMARY_SECONDS
        assert summary["timing_saturated"] is True
        assert budget.snapshot()["turns"] == 2
        assert budget.snapshot()["usage_complete"] is True
    finally:
        client.close()


def test_observed_session_hook_defaults_to_exact_graph_session(monkeypatch):
    _, client, _, _, _, request = _wire(monkeypatch)
    try:
        assert client._observed_session() is None
        client.complete(request)
        assert type(client._observed_session()) is observer.TimeoutSession
        original = client.session
        client.session = object()
        assert client._observed_session() is None
        client.session = original
    finally:
        client.close()
