"""Root-owned no-provider controls for one-process sequence coverage."""
from copy import deepcopy
import io
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_timeout_v2 as observer
from hymem.extraction.llm import LLMRequest


PRIVATE = "PRIVATE-SYNTHETIC-DO-NOT-EXPORT"
RAW = {"rateLimits": {"planType": "pro", "credits": {
    "hasCredits": True, "unlimited": False, "balance": "2.0"},
    "primary": {"usedPercent": 99, "windowDurationMins": 300,
                "resetsAt": 1790784000}}}


def attested(*args):
    return {"containment": True, "denials": 0, "oom": 0}


def test_sequence_changes_allocation_not_fixtures_transport_or_total_limits():
    from tools.diagnostics import luna_timeout_probe_v2 as prior
    from tools.diagnostics import luna_timeout_probe_v3 as probe
    assert probe.USERS == prior.USERS and probe.SYSTEM == prior.SYSTEM
    assert probe.FIXTURE_SHA256 == prior.FIXTURE_SHA256
    assert probe.POLICY == prior.POLICY
    assert probe.PENDING_HOST == prior.PENDING_HOST
    assert probe.LIMITS == {**prior.LIMITS, "workers": 1, "calls_per_worker": 16}
    assert probe.SOURCE_PINS == prior.SOURCE_PINS
    root = Path(__file__).resolve().parents[1]
    for relative, pin in probe.SOURCE_PINS.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == pin


def setup(monkeypatch, *, fail_at=None, rotate_after=None, replace_after=None, tokens=19):
    monkeypatch.setattr(observer.base.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout="codex-cli 0.158.0"))
    processes, clients, called = [], [], []

    def process(*args, **kwargs):
        value = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(),
                                poll=lambda: None, pid=9000 + len(processes))
        processes.append(value)
        return value

    monkeypatch.setattr(observer.base.subprocess, "Popen", process)
    monkeypatch.setattr(observer.warm.WarmSession, "_read", lambda self: None)
    monkeypatch.setattr(observer.TimeoutSession, "close", lambda self: setattr(self, "closed", True))

    def ready(session, **kwargs):
        index = len(called)
        called.append(index)
        session.root_should_timeout = index + 1 == fail_at
        thread = "invented-thread-" + str(session.next_id)
        session.active_thread = thread

        def event(method, **fields):
            return {"method": method, "params": {"threadId": thread,
                    "turnId": "invented-turn", **fields}}

        for value in [
            {"id": session.next_id + 1, "result": {
                "turn": {"id": "invented-turn", "status": "inProgress"}}},
            event("item/started", item={"id": "item", "type": "agentMessage"}),
            event("item/completed", item={"id": "item", "type": "agentMessage",
                  "phase": "final_answer", "text": PRIVATE}),
            event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": tokens}}),
            event("turn/completed", turn={"id": "invented-turn", "status": "completed"}),
            {"id": session.next_id + 2, "result": {"status": "unsubscribed"}},
        ]:
            session.events.put(value)
        return {"auth": "chatgpt", "model": "gpt-6-luna", "inference_enabled": False,
                "config_isolation_admitted": True, "_thread_id": thread,
                "quota_windows": observer.base.quota_metadata(deepcopy(RAW))}

    monkeypatch.setattr(observer.base, "inspect_preflight", ready)
    original_next = observer.TimeoutSession.next_event

    def next_event(session):
        if session.root_should_timeout:
            observer.base._fail("timeout")
        return original_next(session)

    monkeypatch.setattr(observer.TimeoutSession, "next_event", next_event)

    def factory(*args, **kwargs):
        client = observer.TimeoutSubscriptionClient(*args, **kwargs)
        original_complete = client.complete

        def complete(request):
            result = original_complete(request)
            if len(called) == rotate_after:
                client.session.created_at -= 301
            if len(called) == replace_after:
                client.session.process = process()
            return result

        client.complete = complete
        clients.append(client)
        return client

    return factory, clients, processes, called


def run(monkeypatch, **kwargs):
    from tools.diagnostics import luna_timeout_probe_v3 as probe
    factory, clients, processes, called = setup(monkeypatch, **kwargs)
    result = probe.run_probe(observer, LLMRequest, "unused",
                             attest=attested, client_factory=factory)
    assert probe.validate_result(result, observer)
    assert PRIVATE not in json.dumps(result)
    assert "invented-thread" not in json.dumps(result)
    assert result["reserved"] == result["in_flight"] == 0
    assert result["client_cleanup_verified"] is True
    assert all(client.session is None and client.directory is None for client in clients)
    return result, clients, processes, called


def test_real_observer_client_parser_and_budget_cover_sixteen_on_one_process(monkeypatch):
    result, clients, processes, called = run(monkeypatch)
    assert result["status"] == "observed_success"
    assert result["turns"] == result["returned"] == 16
    assert result["failed"] == result["not_attempted"] == 0
    assert result["known_tokens"] == 16 * 19 and result["usage_complete"] is True
    assert len(clients) == len(processes) == 1 and called == list(range(16))
    assert clients[0].processes_started == 1 and clients[0].rotations == 0
    assert clients[0].cold_calls == 1 and clients[0].warm_calls == 15
    assert clients[0].requests_on_process == 16
    assert result["lme_readiness_proved"] is False
    assert result["historical_timeout_cause_proved"] is False


def test_timeout_at_fifteen_retains_primary_fault_and_unknown_usage(monkeypatch):
    result, clients, processes, called = run(monkeypatch, fail_at=15)
    assert result["status"] == "incomplete_or_failed"
    assert result["returned"] == 14 and result["turns"] == result["attempted"] == 15
    assert result["failed"] == result["not_attempted"] == 1
    assert result["known_tokens"] == 14 * 19 and result["usage_complete"] is False
    assert result["first_failure"]["code"] == "timeout"
    assert result["first_failure"]["request_index"] == 15
    assert result["first_failure"]["process_index"] == 1
    assert len(processes) == 1 and len(called) == 15


def test_legitimate_age_rotation_cannot_claim_same_process_coverage(monkeypatch):
    result, clients, processes, called = run(monkeypatch, rotate_after=4)
    assert result["status"] == "coverage_not_reached"
    assert clients[0].rotations == 1 and len(processes) == 2
    assert result["usage_complete"] is True
    assert len(called) < 16
    assert result["first_failure"] is None


def test_process_object_substitution_cannot_pass_even_with_unchanged_counters(monkeypatch):
    result, clients, processes, called = run(monkeypatch, replace_after=4)
    assert result["status"] == "incomplete_or_failed"
    assert result["lifecycle"]["coverage_reason"] == "identity_mismatch"
    assert clients[0].rotations == 0 and clients[0].processes_started == 1
    assert result["usage_complete"] is True and len(called) < 16
    assert result["first_failure"] is None


def test_one_admitted_call_overshoot_retains_usage_and_stops_before_next(monkeypatch):
    result, _, _, called = run(monkeypatch, tokens=200_001)
    assert result["status"] == "incomplete_or_failed"
    assert result["turns"] == result["returned"] == len(called) == 1
    assert result["known_tokens"] == 200_001 and result["usage_complete"] is True
    # The unchanged client may enter complete() once more, but its reservation
    # rejects before preflight or turn admission. This is not a second model call.
    assert result["known_token_overshoot"] is True and result["not_attempted"] == 14
    assert result["attempted"] == 2 and result["failed"] == 1


@pytest.mark.parametrize("field,bad", [
    ("processes_started", 2), ("processes_started", True),
    ("rotations", 1), ("cold_calls", 2), ("warm_calls", 14),
    ("requests_on_process", 15), ("identity_checks", 15),
    ("coverage_complete", False), ("coverage_complete", 1),
    ("successful_positions", [1] * 16),
    ("successful_positions", [True, *range(2, 17)]),
    ("successful_positions", [1.0, *range(2, 17)]),
])
def test_validator_rejects_success_with_inconsistent_lifecycle(monkeypatch, field, bad):
    from tools.diagnostics import luna_timeout_probe_v3 as probe
    result, _, _, _ = run(monkeypatch)
    result["lifecycle"][field] = bad
    assert not probe.validate_result(result, observer)


@pytest.mark.parametrize("group,key,value", [
    ("limits", "workers", True), ("limits", "turns", 16.0),
    ("policy", "rerolls", False),
])
def test_fixed_contract_does_not_equate_booleans_or_floats_with_ints(
        monkeypatch, group, key, value):
    from tools.diagnostics import luna_timeout_probe_v3 as probe
    result, _, _, _ = run(monkeypatch)
    result[group][key] = value
    assert not probe.validate_result(result, observer)


@pytest.mark.parametrize("change", ["position", "identity", "private", "worker"])
def test_validator_requires_per_request_evidence_without_extra_fields(monkeypatch, change):
    from tools.diagnostics import luna_timeout_probe_v3 as probe
    result, _, _, _ = run(monkeypatch)
    slot = result["records"][14]
    if change == "position":
        slot["sequence_position"] = 1
    elif change == "identity":
        slot["identity_checked"] = False
    elif change == "private":
        slot["pid"] = 9000
    else:
        slot["worker"] = 1
    assert not probe.validate_result(result, observer)
