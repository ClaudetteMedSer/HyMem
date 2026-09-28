"""Offline controls for warm process isolation and accounting."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import time

import pytest

from benchmarks import codex_subscription_warm as warm


def request():
    return SimpleNamespace(system="synthetic system", user="synthetic user",
                           temperature=0, max_tokens=32, response_format="json")


class FakeSession:
    instances = []
    def __init__(self, binary, cwd, timeout=120):
        self.created_at = time.monotonic()
        self.deadline = self.created_at + timeout
        self.ids = []
        self.closed = False
        self.unsubscribed = []
        self.__class__.instances.append(self)
    def set_deadline(self, deadline):
        self.deadline = deadline
    def unsubscribe(self, thread_id):
        self.unsubscribed.append(thread_id)
    def close(self):
        self.closed = True


def admission(session, **kwargs):
    tid = f"thread-{len(session.ids) + 1}"
    session.ids.append(tid)
    return {"auth": "chatgpt", "model": warm.base.MODEL,
            "config_isolation_admitted": True, "inference_enabled": False,
            "quota_windows": [{"remaining_percent": 75}], "_thread_id": tid}


def client(*, max_requests=16, max_age_seconds=300):
    budget = warm.SharedBudget(warm.BudgetLimits(10, 100, 30))
    return warm.WarmSubscriptionClient("unused", budget, "q",
        warm.BudgetLimits(10, 100, 30), session_factory=FakeSession,
        max_requests=max_requests, max_age_seconds=max_age_seconds)


def test_fresh_thread_same_process_and_close(monkeypatch):
    FakeSession.instances.clear()
    monkeypatch.setattr(warm.base, "inspect_preflight", admission)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 5))
    item = client()
    assert item.complete(request()) == "{}"
    assert item.complete(request()) == "{}"
    assert len(FakeSession.instances) == 1
    assert FakeSession.instances[0].ids == ["thread-1", "thread-2"]
    assert FakeSession.instances[0].unsubscribed == ["thread-1", "thread-2"]
    assert not FakeSession.instances[0].closed
    item.close()
    assert FakeSession.instances[0].closed
    assert item.observed_turns == 2 and item.observed_tokens == 10


def test_rotation_after_bound_and_age(monkeypatch):
    FakeSession.instances.clear()
    monkeypatch.setattr(warm.base, "inspect_preflight", admission)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 5))
    item = client(max_requests=1)
    item.complete(request())
    item.complete(request())
    assert len(FakeSession.instances) == 2
    assert FakeSession.instances[0].closed
    assert item.rotations == 1
    item.close()


def test_quota_drop_blocks_second_turn_and_reaps(monkeypatch):
    FakeSession.instances.clear()
    counter = 0
    def changing(session, **kwargs):
        nonlocal counter
        counter += 1
        value = admission(session)
        value["quota_windows"][0]["remaining_percent"] = 75 if counter == 1 else 24
        return value
    monkeypatch.setattr(warm.base, "inspect_preflight", changing)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 5))
    item = client()
    item.complete(request())
    with pytest.raises(warm.ConcurrentStop, match="quota_unverified"):
        item.complete(request())
    assert FakeSession.instances[0].closed
    assert item.observed_turns == 1 and item.observed_tokens == 5


def test_unsubscribe_failure_preserves_known_usage(monkeypatch):
    FakeSession.instances.clear()
    monkeypatch.setattr(warm.base, "inspect_preflight", admission)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 7))
    def fail(self, tid):
        warm.base._fail("unsubscribe_unverified")
    monkeypatch.setattr(FakeSession, "unsubscribe", fail)
    item = client()
    with pytest.raises(warm.ConcurrentStop, match="transport_or_admission_failure"):
        item.complete(request())
    assert item.observed_turns == 1 and item.observed_tokens == 7
    assert item.usage_complete
    assert FakeSession.instances[0].closed


def test_turn_failure_marks_usage_unknown_and_reaps(monkeypatch):
    FakeSession.instances.clear()
    monkeypatch.setattr(warm.base, "inspect_preflight", admission)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: warm.base._fail("process_exit"))
    item = client()
    with pytest.raises(warm.ConcurrentStop):
        item.complete(request())
    assert item.observed_turns == 1 and not item.usage_complete
    assert FakeSession.instances[0].closed


def session_with_events(*events):
    session = object.__new__(warm.WarmSession)
    session.events_for_test = deque(events)
    session.next_id = 0
    session.pending = deque()
    session.initialized_result = {"cached": True}
    session.initialized_sent = True
    session.active_thread = None
    session.starting_threads = []
    session.starting_events = []
    session.started_notifications = set()
    session.seen_threads = set()
    session.retired_threads = set()
    session.warning_targets = []
    session.bound_thread_id = None
    session.send = lambda *args, **kwargs: 1
    session.receive = lambda: session.events_for_test.popleft()
    return session


def test_start_notification_before_response_and_reused_id_rejected():
    event = {"method": "thread/started", "params": {"thread": {"id": "t1"}}}
    result = {"id": 1, "result": {"thread": {"id": "t1"}}}
    session = session_with_events(event, result)
    assert session.rpc("thread/start", {}) == result["result"]
    assert session.active_thread == "t1"
    session.retired_threads.add("t1")
    session.active_thread = None
    session.events_for_test.append(result)
    with pytest.raises(warm.base.SubscriptionTransportError, match="thread_identity_mismatch"):
        session.rpc("thread/start", {})


def test_retired_tail_status_allowed_content_and_unknown_rejected():
    session = session_with_events()
    session.retired_threads.add("old")
    session._route({"method": "thread/status/changed", "params":
        {"threadId": "old", "status": {"type": "idle"}}}, "account/read", False)
    session._route({"method": "thread/closed", "params": {"threadId": "old"}},
        "account/read", False)
    with pytest.raises(warm.base.SubscriptionTransportError):
        session._route({"method": "item/started", "params":
            {"threadId": "old", "turnId": "turn", "item": {"type": "agentMessage"}}},
            "account/read", False)
    with pytest.raises(warm.base.SubscriptionTransportError):
        session._route({"method": "thread/closed", "params": {"threadId": "unknown"}},
            "account/read", False)


def test_cached_initialize_only():
    session = session_with_events()
    assert session.rpc("initialize", {}) == {"cached": True}
    assert not session.events_for_test


def test_status_before_start_response_is_bound_to_response_id():
    status = {"method": "thread/status/changed", "params":
        {"threadId": "t1", "status": {"type": "idle"}}}
    matching = session_with_events(status, {"id": 1, "result": {"thread": {"id": "t1"}}})
    matching.rpc("thread/start", {})
    assert matching.active_thread == "t1"
    mismatched = session_with_events(status, {"id": 1, "result": {"thread": {"id": "t2"}}})
    with pytest.raises(warm.base.SubscriptionTransportError, match="thread_identity_mismatch"):
        mismatched.rpc("thread/start", {})


def test_expired_deadline_prevents_rpc_write():
    session = session_with_events()
    session.deadline = time.monotonic() - 1
    writes = []
    session.send = warm.WarmSession.send.__get__(session, warm.WarmSession)
    session.process = SimpleNamespace(stdin=SimpleNamespace(write=writes.append,
        flush=lambda: None))
    with pytest.raises(warm.base.SubscriptionTransportError, match="timeout"):
        session.send("turn/start", {"threadId": "t1"})
    assert writes == []


def test_failed_reap_keeps_handle_for_final_retry(monkeypatch):
    FakeSession.instances.clear()
    monkeypatch.setattr(warm.base, "inspect_preflight", admission)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 7))
    monkeypatch.setattr(FakeSession, "unsubscribe", lambda *args:
        warm.base._fail("unsubscribe_unverified"))
    attempts = 0
    def transient_close(self):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient reap failure")
        self.closed = True
    monkeypatch.setattr(FakeSession, "close", transient_close)
    item = client()
    with pytest.raises(warm.ConcurrentStop, match="cleanup_failure"):
        item.complete(request())
    assert item.session is FakeSession.instances[0]
    assert item.budget.snapshot()["stopped"]
    item.close()
    assert attempts == 2 and FakeSession.instances[0].closed
    assert item.session is None and item.closed
    assert item.observed_tokens == 7
