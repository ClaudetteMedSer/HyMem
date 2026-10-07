"""Offline transport and finite timeout observation controls."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_timeout_v2 as diagnostic


SECRET = "PRIVATE-INVENTED-CONTENT"


class NoThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


def notice(method, **params):
    return {"method": method, "params": {"threadId": "thread", "turnId": "turn", **params}}


def final_sequence():
    return [notice("item/started", item={"id": "item", "type": "agentMessage"}),
            notice("item/completed", item={"id": "item", "type": "agentMessage",
                                      "phase": "final_answer", "text": SECRET}),
            notice("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 19}}),
            notice("turn/completed", turn={"id": "turn", "status": "completed"})]


def make_client(monkeypatch, events, *, budget_seconds=120):
    monkeypatch.setattr(diagnostic.base.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout=f"codex-cli {diagnostic.base.VERSION}"))
    monkeypatch.setattr(diagnostic.base.subprocess, "Popen", lambda *a, **k:
                        SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda: None))
    monkeypatch.setattr(diagnostic.base.threading, "Thread", NoThread)
    monkeypatch.setattr(diagnostic.TimeoutSession, "close", lambda self: None)
    monkeypatch.setattr(diagnostic.base, "inspect_preflight", lambda session, **k:
                        {"auth": "chatgpt", "model": diagnostic.base.MODEL,
                         "config_isolation_admitted": True, "inference_enabled": False,
                         "quota_windows": [{"remaining_percent": 80}], "_thread_id": "thread"})

    sessions = []
    def factory(*args, **kwargs):
        session = diagnostic.TimeoutSession(*args, **kwargs)
        session.active_thread = "thread"
        session.events.put({"id": 1, "result": {"turn": {"id": "turn", "status": "inProgress"}}})
        for value in events:
            session.events.put(copy.deepcopy(value))
        sessions.append(session)
        return session

    limits = diagnostic.BudgetLimits(4, 1000, budget_seconds)
    budget = diagnostic.SharedBudget(limits)
    client = diagnostic.TimeoutSubscriptionClient("unused", budget, "q", limits,
        session_factory=factory)
    request = SimpleNamespace(system="invented-system", user=SECRET,
                              temperature=0.0, max_tokens=32, response_format="text")
    return client, request, sessions


def test_complete_usage_and_final_shape_are_observed_without_text(monkeypatch):
    client, request, sessions = make_client(monkeypatch, final_sequence())
    monkeypatch.setattr(diagnostic.TimeoutSession, "unsubscribe", lambda self, tid: None)
    assert client.complete(request) == SECRET
    record, = client.diagnostic_records()
    assert record["status"] == "success" and record["known_usage"] is True
    assert record["observed"]["final_seen"] is True
    assert record["observed"]["completed_seen"] is True
    assert record["observed"]["usage_state"] == "positive"
    assert record["remaining_at_turn_start_seconds"] <= 120
    assert record["phase_seconds"]["preflight"] is not None
    assert record["phase_seconds"]["events"] is not None
    assert record["precleanup"] is None
    assert SECRET not in json.dumps(record)
    assert record["deadline_monotonic"] == pytest.approx(sessions[0].deadline, abs=0.000001)


@pytest.mark.parametrize("events,final,usage", [
    ([], False, "absent"),
    ([notice("item/started", item={"id": "item", "type": "agentMessage"})], False, "absent"),
    (final_sequence()[:-1], True, "positive"),
])
def test_timeout_preserves_partial_shape_and_precleanup_queue(monkeypatch, events, final, usage):
    client, request, sessions = make_client(monkeypatch, events, budget_seconds=0.04)
    with pytest.raises(diagnostic.ConcurrentStop, match="timeout"):
        client.complete(request)
    record, = client.diagnostic_records()
    assert record["status"] == "failure" and record["failure_code"] == "timeout"
    assert record["known_usage"] is False
    assert record["observed"]["final_seen"] is final
    assert record["observed"]["usage_state"] == usage
    assert record["observed"]["completed_seen"] is False
    assert record["precleanup"]["queue_depth"] == 0
    assert record["phase_seconds"]["cleanup"] is not None
    assert client.budget.snapshot()["usage_complete"] is False
    assert SECRET not in json.dumps(record)
    assert record["deadline_monotonic"] == pytest.approx(sessions[0].deadline, abs=0.000001)


@pytest.mark.parametrize("bad", [
    notice("item/completed", item={"id": "item", "type": "agentMessage", "phase": "final_answer", "text": SECRET}),
    notice("thread/tokenUsage/updated", turnId="foreign", tokenUsage={"total": {"totalTokens": 19}}),
    notice("item/agentMessage/delta", itemId="item", delta=SECRET),
    notice("error", willRetry=False, error={"message": SECRET, "codexErrorInfo": "unauthorized"}),
])
def test_foreign_malformed_optout_and_error_never_succeed(monkeypatch, bad):
    client, request, _ = make_client(monkeypatch, [bad, *final_sequence()])
    with pytest.raises(diagnostic.ConcurrentStop):
        client.complete(request)
    record, = client.diagnostic_records()
    assert record["status"] == "failure"
    assert record["known_usage"] is False
    assert SECRET not in json.dumps(record)


def test_queue_records_local_enqueue_and_dequeue_without_payload(monkeypatch):
    client, request, sessions = make_client(monkeypatch, final_sequence())
    session = sessions[0] if sessions else None
    assert session is None
    monkeypatch.setattr(diagnostic.TimeoutSession, "unsubscribe", lambda self, tid: None)
    assert client.complete(request) == SECRET
    record, = client.diagnostic_records()
    # The synthetic queue was filled before set_deadline, so the per-call
    # enqueue counter is zero while local dequeues occur during the call.
    assert record["reader_enqueued"] == 0 and record["reader_dequeued"] >= 5
    assert record["queue_depth"] == 0
    assert record["last_queue_delay_seconds"] is not None


def test_public_projection_rejects_injected_fields_and_is_detached(monkeypatch):
    client, request, _ = make_client(monkeypatch, final_sequence())
    monkeypatch.setattr(diagnostic.TimeoutSession, "unsubscribe", lambda self, tid: None)
    client.complete(request)
    one, = client.diagnostic_records()
    one["observed"]["extra"] = SECRET
    two, = client.diagnostic_records()
    assert "extra" not in two["observed"]
    injected = dict(two, prompt=SECRET)
    assert diagnostic._project_record(injected) is None
    assert SECRET not in json.dumps(two)


def test_source_identity_and_default_120_second_budget():
    path = Path(diagnostic.warm.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == diagnostic.PINNED_WARM_V9_SHA256
    assert diagnostic.warm.WarmSession is diagnostic.warm.v8.WarmSession
    assert diagnostic.warm.WarmSession.__mro__[1] is diagnostic.warm.v8.v7.WarmSession
    limits = diagnostic.BudgetLimits(1, 1000, 600)
    budget = diagnostic.SharedBudget(limits)
    budget.register("q", limits)
    assert budget.reserve("q") == 120.0
