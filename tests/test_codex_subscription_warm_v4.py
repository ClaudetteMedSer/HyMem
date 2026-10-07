"""Offline controls for consumed-event diagnostics against the frozen parser."""
from __future__ import annotations

from collections import deque
import json
import time
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_warm_v4 as warm


THREAD = "thread-a"
TURN = "turn-a"
SECRET = "private@example.com /private/hidden prompt"


def event(method, **fields):
    return {"method": method, "params": {"threadId": THREAD, "turnId": TURN, **fields}}


START = event("item/started", item={"id": "message-a", "type": "agentMessage"})
FINAL = event("item/completed", item={"id": "message-a", "type": "agentMessage",
                                 "phase": "final_answer", "text": SECRET})
USAGE = event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 42}})
DONE = event("turn/completed", turn={"id": TURN, "status": "completed"})
ZERO = event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 0}})


def session(monkeypatch, values):
    s = object.__new__(warm.WarmSession)
    s.pending = deque(values)
    s.active_thread = THREAD
    s.active_turn = None
    s.stage = "turn/start"
    s.retired_threads = set()
    s.last_event = None
    s.rpc_error = None
    s.turn_observation = None
    s._observation_thread = None
    s._observation_turn = None
    def rpc(self, method, params, *, preserve_notifications=False):
        assert method == "turn/start" and params["threadId"] == THREAD
        return {"turn": {"id": TURN, "status": "inProgress"}}
    monkeypatch.setattr(warm.v3.v2.WarmSession, "rpc", rpc)
    return s


def run(monkeypatch, values):
    s = session(monkeypatch, values)
    try:
        answer = warm.base._run_turn(s, THREAD, "synthetic user")
        failure = None
    except warm.base.SubscriptionTransportError as exc:
        answer = None
        failure = exc.args
    return s, answer, failure


@pytest.mark.parametrize("values,expected", [
    ([START, FINAL, DONE], (3, True, 1, 0, "absent", "turn_completed")),
    ([USAGE, DONE], (2, True, 0, 1, "positive", "turn_completed")),
    ([START, FINAL, ZERO, DONE], (4, True, 1, 1, "zero", "turn_completed")),
    ([event("thread/status/changed")] * warm.base.MAX_EVENTS,
     (warm.base.MAX_EVENTS, False, 0, 0, "absent", "thread_status_changed")),
])
def test_four_incomplete_arms_keep_frozen_exception_and_distinct_observation(monkeypatch, values, expected):
    s, answer, failure = run(monkeypatch, values)
    assert answer is None and failure == ("incomplete_turn_or_usage",)
    observation = warm._project_observation(s.turn_observation)
    assert observation is not None
    _, _, finals, _, _, _ = expected
    assert (observation["events_consumed"], observation["completed_seen"],
            observation["final_count"], observation["usage_update_count"],
            observation["usage_state"], observation["last_event_family"]) == expected
    assert observation["final_seen"] == bool(finals)
    assert SECRET not in json.dumps(observation)


def test_success_and_post_completion_usage_do_not_change_parser_decision(monkeypatch):
    s, answer, failure = run(monkeypatch, [START, FINAL, USAGE, DONE])
    assert answer == (SECRET, 42) and failure is None
    assert s.turn_observation["events_consumed"] == 4
    s, answer, failure = run(monkeypatch, [START, FINAL, DONE, USAGE])
    assert answer is None and failure == ("incomplete_turn_or_usage",)
    assert s.turn_observation["events_consumed"] == 3
    assert s.turn_observation["usage_state"] == "absent"
    assert len(s.pending) == 1


@pytest.mark.parametrize("method,extra,family", [
    ("item/reasoning/summaryPartAdded", {"summaryIndex": 0},
     "item_reasoning_summaryPartAdded"),
    ("item/reasoning/summaryTextDelta", {"summaryIndex": 0, "delta": SECRET},
     "item_reasoning_summaryTextDelta"),
])
def test_accepted_reasoning_family_at_actual_parser_ceiling(monkeypatch, method, extra, family):
    reasoning_start = event("item/started", item={"id": "reason-a", "type": "reasoning"})
    reasoning_event = event(method, itemId="reason-a", **extra)
    values = [reasoning_start] + [event("thread/status/changed")] * (warm.base.MAX_EVENTS - 2)
    values.append(reasoning_event)
    s, answer, failure = run(monkeypatch, values)
    assert answer is None and failure == ("incomplete_turn_or_usage",)
    assert s.turn_observation["events_consumed"] == warm.base.MAX_EVENTS
    assert s.turn_observation["last_event_family"] == family
    assert warm._project_observation(s.turn_observation) is not None
    assert SECRET not in json.dumps(s.turn_observation)


def test_fresh_turn_resets_and_wrong_identity_is_not_attributed(monkeypatch):
    s = session(monkeypatch, [START, FINAL, USAGE, DONE])
    assert warm.base._run_turn(s, THREAD, "first") == (SECRET, 42)
    s.pending = deque([event("thread/tokenUsage/updated", turnId="other",
                             tokenUsage={"total": {"totalTokens": 99}})])
    with pytest.raises(warm.base.SubscriptionTransportError, match="usage_identity_mismatch"):
        warm.base._run_turn(s, THREAD, "second")
    assert s.turn_observation["events_consumed"] == 1
    assert s.turn_observation["usage_update_count"] == 0
    assert s.turn_observation["final_count"] == 0


@pytest.mark.parametrize("values,code,usage_count,usage_state", [
    ([USAGE, USAGE, event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 20}})],
     "usage_regressed", 3, "positive"),
    ([event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": True}})],
     "usage_invalid", 0, "absent"),
    ([START, FINAL, FINAL], "final_message_invalid", 0, "absent"),
    ([event("item/started", item={"id": "tool-a", "type": "functionCall"})],
     "tool_or_extra_item", 0, "absent"),
    ([event("model/rerouted")], "unexpected_notification:model/rerouted", 0, "absent"),
])
def test_parser_rejections_remain_exact_and_observation_never_claims_acceptance(
        monkeypatch, values, code, usage_count, usage_state):
    s, answer, failure = run(monkeypatch, values)
    assert answer is None and failure == (code,)
    assert s.turn_observation["usage_update_count"] == usage_count
    assert s.turn_observation["usage_state"] == usage_state
    assert s.turn_observation["basis"] == "consumed_observed_shape"


def test_failure_snapshot_is_taken_before_cleanup_and_contains_no_private_text(monkeypatch):
    class Budget:
        def __init__(self):
            self.failure = None
        def snapshot(self):
            return {"known_tokens": 0, "usage_complete": False}
        def record_first_failure(self, code, metadata):
            self.failure = {"code": code, **metadata}
    s = session(monkeypatch, [START, FINAL, DONE])
    with pytest.raises(warm.base.SubscriptionTransportError):
        warm.base._run_turn(s, THREAD, SECRET)
    s.created_at = warm.time.monotonic()
    s.starting_events = []
    budget = Budget()
    client = object.__new__(warm.WarmSubscriptionClient)
    client.session = s
    client.budget = budget
    client.processes_started = 1
    client.requests_on_process = 0
    client._record_failure("incomplete_turn_or_usage", "run", True, False)
    assert budget.failure["turn_observation"]["final_seen"] is True
    assert budget.failure["known_usage"] is False
    assert budget.failure["usage_complete"] is False
    s.reset_turn_observation()
    assert budget.failure["turn_observation"]["events_consumed"] == 3
    projected = warm.serialize_failure(budget.failure)
    assert projected["turn_observation"] == budget.failure["turn_observation"]
    assert SECRET not in json.dumps(projected)
    assert THREAD not in json.dumps(projected) and TURN not in json.dumps(projected)


def test_projection_rejects_fabricated_or_private_observation():
    payload = {"code": "incomplete_turn_or_usage", "phase": "run", "rpc": "turn/events",
        "turn_observation": {
        "basis": "consumed_observed_shape", "events_consumed": True,
        "completed_seen": True, "final_seen": True, "final_count": 1,
        "usage_update_count": 0, "usage_state": "absent",
        "last_event_family": "turn_completed", "secret": SECRET}}
    assert "turn_observation" not in warm.serialize_failure(payload)
    payload["turn_observation"]["events_consumed"] = 2
    assert SECRET not in json.dumps(warm.serialize_failure(payload))


@pytest.mark.parametrize("field,value", [
    ("basis", {"private": SECRET}), ("events_consumed", []),
    ("completed_seen", 1), ("final_count", {"private": SECRET}),
    ("usage_state", [SECRET]), ("last_event_family", {"private": SECRET}),
    ("events_consumed", warm.base.MAX_EVENTS + 1),
])
def test_projection_is_total_and_finite_for_malformed_json(field, value):
    observation = {"basis": "consumed_observed_shape", "events_consumed": 1,
                   "completed_seen": False, "final_seen": False, "final_count": 0,
                   "usage_update_count": 0, "usage_state": "absent",
                   "last_event_family": "warning"}
    observation[field] = value
    payload = {"code": "incomplete_turn_or_usage", "phase": "run", "rpc": "turn/events",
               "turn_observation": observation}
    assert "turn_observation" not in warm.serialize_failure(payload)


def test_empty_snapshot_has_no_last_family():
    value = {"basis": "consumed_observed_shape", "events_consumed": 0,
             "completed_seen": False, "final_seen": False, "final_count": 0,
             "usage_update_count": 0, "usage_state": "absent",
             "last_event_family": None}
    assert warm._project_observation(value) == value


def test_queued_turn_events_bind_only_after_rpc_response(monkeypatch):
    s = session(monkeypatch, [])
    monkeypatch.undo()
    incoming = deque([START, {"id": 7, "result": {"turn": {"id": TURN, "status": "inProgress"}}}])
    s.send = lambda *args, **kwargs: 7
    s.receive = lambda: incoming.popleft()
    result = s.rpc("turn/start", {"threadId": THREAD}, preserve_notifications=True)
    assert result["turn"]["id"] == TURN
    assert s.turn_observation["events_consumed"] == 0
    assert len(s.pending) == 1
    assert s.next_event() == START
    assert s.turn_observation["events_consumed"] == 1


def test_full_client_records_before_cleanup_and_keeps_ledger_incomplete(monkeypatch):
    s = session(monkeypatch, [START, FINAL, DONE])
    s.created_at = time.monotonic()
    s.starting_events = []
    s.closed = False
    s.set_deadline = lambda deadline: None
    def close():
        assert budget.snapshot()["first_failure"]["turn_observation"]["events_consumed"] == 3
        s.closed = True
    s.close = close
    monkeypatch.setattr(warm.base, "inspect_preflight", lambda *args, **kwargs: {
        "auth": "chatgpt", "model": warm.base.MODEL,
        "config_isolation_admitted": True, "inference_enabled": False,
        "quota_windows": [{"remaining_percent": 75}], "_thread_id": THREAD})
    budget = warm.SharedBudget(warm.BudgetLimits(100, 100000, 1000))
    client = warm.WarmSubscriptionClient("unused", budget, "q",
        warm.BudgetLimits(100, 100000, 1000), session_factory=lambda *args, **kwargs: s)
    request = SimpleNamespace(system="system", user=SECRET, temperature=0,
                              max_tokens=32, response_format="json")
    with pytest.raises(warm.ConcurrentStop, match="incomplete_turn_or_usage"):
        client.complete(request)
    first = budget.snapshot()["first_failure"]
    assert s.closed is True and first["turn_observation"]["usage_state"] == "absent"
    assert first["known_usage"] is False and first["usage_complete"] is False
    assert SECRET not in json.dumps(first)


def test_staged_bound_session_forwards_observation_without_copying_identity(monkeypatch):
    from benchmarks.codex_subscription_staged_v1 import _BoundSession
    s = session(monkeypatch, [])
    s.turn_observation = {"basis": "consumed_observed_shape", "events_consumed": 0,
                          "completed_seen": False, "final_seen": False,
                          "final_count": 0, "usage_update_count": 0,
                          "usage_state": "absent", "last_event_family": None}
    wrapped = _BoundSession(s, lambda *args: None)
    assert wrapped.turn_observation is s.turn_observation
    wrapped.reset_turn_observation()
    assert s.turn_observation is None
