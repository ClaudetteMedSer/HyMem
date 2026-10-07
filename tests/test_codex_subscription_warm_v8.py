"""Offline controls for the source-pinned delta notification opt-out."""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import io
import json
from pathlib import Path
import queue
from types import SimpleNamespace
import time

import pytest

from benchmarks import codex_subscription_warm_v7 as previous
from benchmarks import codex_subscription_warm_v8 as warm
from tests.test_codex_subscription_warm_v5_root import event, sequence, session


def delta(**fields):
    return event("item/agentMessage/delta", itemId="item", delta="secret fragment", **fields)


def test_initialize_wire_is_exact_copy_and_cached_once():
    current = object.__new__(warm.WarmSession)
    current.deadline = time.monotonic() + 30
    current.next_id = 0
    current.request_id = None
    current.rpc_error = None
    current.stage = "startup"
    current.initialized_result = None
    current.initialized_sent = False
    current.active_thread = None
    current.process = SimpleNamespace(stdin=io.StringIO())
    current.events = queue.Queue()
    current.events.put({"id": 1, "result": {"serverInfo": {"name": "mock"}}})
    params = {"clientInfo": {"name": "hymem_luna_pilot", "version": "0.1"},
              "capabilities": {"experimentalApi": True}}
    original = copy.deepcopy(params)
    assert current.rpc("initialize", params) == {"serverInfo": {"name": "mock"}}
    assert current.rpc("initialize", params) == {"serverInfo": {"name": "mock"}}
    current.send("initialized", {}, notification=True)
    current.send("initialized", {}, notification=True)
    wire = [json.loads(line) for line in current.process.stdin.getvalue().splitlines()]
    assert wire == [
        {"method": "initialize", "params": {"clientInfo": original["clientInfo"],
         "capabilities": {"experimentalApi": True,
                          "optOutNotificationMethods": ["item/agentMessage/delta"]}}, "id": 1},
        {"method": "initialized", "params": {}},
    ]
    assert params == original
    assert warm.NOTIFICATION_POLICY == "agent_message_delta_optout_v1"
    assert warm.OPT_OUT_NOTIFICATION_METHODS == ("item/agentMessage/delta",)
    assert warm.BILLING_POLICY == previous.BILLING_POLICY


def test_delta_during_initialize_fails_before_initialize_is_cached():
    current = object.__new__(warm.WarmSession)
    current.deadline = time.monotonic() + 30
    current.next_id = 0
    current.request_id = None
    current.rpc_error = None
    current.stage = "startup"
    current.initialized_result = None
    current.initialized_sent = False
    current.process = SimpleNamespace(stdin=io.StringIO())
    current.events = queue.Queue()
    current.events.put(delta())
    current.events.put({"id": 1, "result": {}})
    with pytest.raises(warm.base.SubscriptionTransportError, match="^notification_optout_unverified$"):
        current.rpc("initialize", {"capabilities": {"experimentalApi": True}})
    assert current.next_id == 1 and current.initialized_result is None


def test_client_default_selects_v8_session_and_keeps_private_sink():
    client = warm.WarmSubscriptionClient("unused", warm.SharedBudget(warm.BudgetLimits(100, 100000, 1000)),
        "q", warm.BudgetLimits(100, 100000, 1000), private_failure_sink=None)
    assert client.session_factory is warm.WarmSession
    assert client.private_failure_sink is None
    assert issubclass(warm.WarmSession, warm.v7.WarmSession)


def test_actual_client_records_distinct_failure_and_closes_process(monkeypatch):
    current = session(warm, monkeypatch, [delta(), *sequence()])
    current.reset_private_errors()
    current.set_deadline = lambda value: None
    current.closed = False
    current.close = lambda: setattr(current, "closed", True)
    current.unsubscribe = lambda thread_id: None
    monkeypatch.setattr(warm.base, "inspect_preflight", lambda *args, **kwargs: {
        "auth": "chatgpt", "model": warm.base.MODEL,
        "config_isolation_admitted": True, "inference_enabled": False,
        "quota_windows": [{"remaining_percent": 75}], "_thread_id": "thread"})
    budget = warm.SharedBudget(warm.BudgetLimits(100, 100000, 1000))
    client = warm.WarmSubscriptionClient("unused", budget, "q",
        warm.BudgetLimits(100, 100000, 1000), session_factory=lambda *args, **kwargs: current)
    request = SimpleNamespace(system="system", user="invented", temperature=0,
                              max_tokens=32, response_format="json")
    with pytest.raises(warm.ConcurrentStop, match="^notification_optout_unverified$"):
        client.complete(request)
    assert current.closed
    assert budget.snapshot()["first_failure"]["code"] == warm.OPT_OUT_FAILURE_CODE
    assert warm.serialize_failure(budget.snapshot()["first_failure"])["code"] == warm.OPT_OUT_FAILURE_CODE


@pytest.mark.parametrize("placement", ["before_ack", "after_ack", "pending"])
@pytest.mark.parametrize("notice", [delta(), event("item/agentMessage/delta", itemId="wrong", delta="x"),
                                    {"method": "item/agentMessage/delta", "params": None}])
def test_any_delta_fails_immediately_even_if_early_foreign_or_malformed(monkeypatch, placement, notice):
    current = session(warm, monkeypatch, sequence(), early=[notice] if placement == "before_ack" else None)
    current.reset_private_errors()
    if placement == "after_ack":
        current.events.queue.insert(1, notice)
    elif placement == "pending":
        current.pending.append(notice)
    deadline = current.deadline
    with pytest.raises(warm.base.SubscriptionTransportError, match="^notification_optout_unverified$"):
        warm.base._run_turn(current, "thread", "invented")
    assert current.next_id == 1 and current.deadline == deadline
    assert current.turn_observation is None or current.turn_observation["events_consumed"] <= 1


def test_one_delta_in_5000_fragments_fails_before_cap(monkeypatch):
    text = "x" * 5000
    events = [event("item/started", item={"id": "item", "type": "agentMessage"}),
        *(event("item/agentMessage/delta", itemId="item", delta=char) for char in text),
        event("item/completed", item={"id": "item", "type": "agentMessage",
             "phase": "final_answer", "text": text}),
        event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 5008}}),
        event("turn/completed", turn={"id": "turn", "status": "completed"})]
    current = session(warm, monkeypatch, events)
    current.reset_private_errors()
    with pytest.raises(warm.base.SubscriptionTransportError, match="^notification_optout_unverified$"):
        warm.base._run_turn(current, "thread", "invented")
    assert current.turn_observation["events_consumed"] == 1


def test_authoritative_final_usage_and_completion_still_succeed(monkeypatch):
    current = session(warm, monkeypatch, sequence())
    current.reset_private_errors()
    assert warm.base._run_turn(current, "thread", "invented") == (
        "SYNTHETIC-PRIVATE-NOT-FOR-METADATA", 19)


def test_separate_http_400_error_is_still_visible(monkeypatch):
    error = event("error", willRetry=False, error={"message": "private", "misalignment": None,
        "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 400}}})
    current = session(warm, monkeypatch, [error, *sequence()])
    current.reset_private_errors()
    with pytest.raises(warm.base.SubscriptionTransportError, match="^unexpected_notification:error$"):
        warm.base._run_turn(current, "thread", "invented")
    assert current.private_error_count == 1
    assert current.private_error_last["http_status_code"] == 400


@pytest.mark.parametrize("mutation", [
    lambda xs: xs.pop(1),
    lambda xs: xs.pop(2),
    lambda xs: xs[1]["params"]["item"].update(phase="commentary"),
    lambda xs: xs[2]["params"].update(turnId="wrong"),
    lambda xs: xs[0]["params"]["item"].update(type="commandExecution"),
])
def test_existing_final_usage_identity_and_tool_gates_remain(monkeypatch, mutation):
    events = sequence()
    mutation(events)
    current = session(warm, monkeypatch, events)
    current.reset_private_errors()
    with pytest.raises(warm.base.SubscriptionTransportError):
        warm.base._run_turn(current, "thread", "invented")


def test_retry_limit_and_event_limit_stay_frozen(monkeypatch):
    retry = event("error", willRetry=True, error={"message": "private", "misalignment": None,
        "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 503}}})
    for count, expected in ((8, True), (9, False)):
        current = session(warm, monkeypatch, [retry] * count + sequence())
        current.reset_private_errors()
        deadline = current.deadline
        if expected:
            assert warm.base._run_turn(current, "thread", "invented")[1] == 19
        else:
            with pytest.raises(warm.base.SubscriptionTransportError, match="^unexpected_notification:error$"):
                warm.base._run_turn(current, "thread", "invented")
        assert current.deadline == deadline and current.next_id == 1
    notices = [event("turn/started", turn={"id": "turn"})] * 4093
    current = session(warm, monkeypatch, [sequence()[0], *notices, *sequence()[1:]])
    current.reset_private_errors()
    with pytest.raises(warm.base.SubscriptionTransportError, match="^incomplete_turn_or_usage$"):
        warm.base._run_turn(current, "thread", "invented")
    assert current.turn_observation["events_consumed"] == warm.base.MAX_EVENTS == 4096


def test_new_failure_code_is_finite_and_prior_sources_stay_unchanged():
    projected = warm.serialize_failure({"code": warm.OPT_OUT_FAILURE_CODE,
        "phase": "run", "rpc": "turn/events", "message": "private"})
    assert projected["code"] == warm.OPT_OUT_FAILURE_CODE
    assert "private" not in json.dumps(projected)
    assert previous.serialize_failure({"code": warm.OPT_OUT_FAILURE_CODE,
        "phase": "run", "rpc": "turn/events"})["code"] == "fixed_other"
    assert hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() == warm.PINNED_WARM_V7_SHA256
