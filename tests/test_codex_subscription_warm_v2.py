"""Offline first-failure and unchanged warm-lifecycle controls."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_warm_v2 as warm


def request():
    return SimpleNamespace(system="synthetic system", user="synthetic user",
                           temperature=0, max_tokens=32, response_format="json")


class FakeSession:
    instances = []
    now = staticmethod(warm.time.monotonic)
    fail_unsubscribe = None
    fail_close = None

    def __init__(self, binary, cwd, timeout=120):
        self.created_at = self.now()
        self.deadline = self.created_at + timeout
        self.stage = "initialize"
        self.ids = []
        self.pending = deque()
        self.starting_events = []
        self.retired_threads = set()
        self.closed = False
        self.__class__.instances.append(self)

    def set_deadline(self, deadline):
        self.deadline = deadline

    def unsubscribe(self, thread_id):
        self.stage = "thread/unsubscribe"
        if self.fail_unsubscribe is not None:
            raise self.fail_unsubscribe
        self.retired_threads.add(thread_id)

    def close(self):
        if self.fail_close is not None:
            raise self.fail_close
        self.closed = True


def admission(session, **kwargs):
    session.stage = "thread/start"
    tid = f"thread-{len(session.ids) + 1}"
    session.ids.append(tid)
    return {"auth": "chatgpt", "model": warm.base.MODEL,
            "config_isolation_admitted": True, "inference_enabled": False,
            "quota_windows": [{"remaining_percent": 75}], "_thread_id": tid}


def client(*, budget=None, question_id="q", max_requests=16, max_age_seconds=300):
    budget = budget or warm.SharedBudget(warm.BudgetLimits(100, 10000, 1000))
    return warm.WarmSubscriptionClient("unused", budget, question_id,
        warm.BudgetLimits(100, 10000, 1000), session_factory=FakeSession,
        max_requests=max_requests, max_age_seconds=max_age_seconds)


@pytest.fixture(autouse=True)
def reset_session():
    FakeSession.instances.clear()
    FakeSession.fail_unsubscribe = None
    FakeSession.fail_close = None
    FakeSession.now = staticmethod(warm.time.monotonic)


def test_preflight_specific_first_code_and_bounded_metadata(monkeypatch):
    monkeypatch.setattr(warm.base, "inspect_preflight", lambda *args, **kwargs:
        warm.base._fail("quota_floor"))
    item = client()
    with pytest.raises(warm.ConcurrentStop, match="quota_floor"):
        item.complete(request())
    state = item.budget.snapshot()
    assert state["stop_code"] == "quota_floor"
    assert state["first_failure"] == {
        "code": "quota_floor", "phase": "preflight", "rpc": "initialize",
        "process_age_seconds": pytest.approx(0, abs=1), "process_index": 1,
        "request_index": 1, "retired_count": 0, "queue_count": 0,
        "turn_admitted": False, "known_usage": True, "known_tokens": 0,
        "usage_complete": False,
    }
    assert state["turns"] == 0 and FakeSession.instances[0].closed


@pytest.mark.parametrize("phase,setup,expected,known", [
    ("run", "run", "process_exit:turn/start", False),
    ("unsubscribe", "unsubscribe", "unsubscribe_unverified", True),
])
def test_run_and_unsubscribe_attribution(monkeypatch, phase, setup, expected, known):
    monkeypatch.setattr(warm.base, "inspect_preflight", admission)
    if setup == "run":
        def fail_run(session, *args):
            session.stage = "turn/start"
            warm.base._fail(expected)
        monkeypatch.setattr(warm.base, "_run_turn", fail_run)
    else:
        monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 7))
        FakeSession.fail_unsubscribe = warm.base.SubscriptionTransportError(expected)
    item = client()
    with pytest.raises(warm.ConcurrentStop, match=expected):
        item.complete(request())
    state = item.budget.snapshot()
    assert state["first_failure"]["phase"] == phase
    assert state["first_failure"]["code"] == expected
    assert state["first_failure"]["turn_admitted"] is True
    assert state["first_failure"]["known_usage"] is known
    assert state["known_tokens"] == (7 if known else 0)
    assert state["usage_complete"] is known
    assert state["turns"] == 1 and FakeSession.instances[0].closed


def test_unknown_exception_code_and_metadata_never_leak(monkeypatch):
    secret = "thread/secret account@example.com /private/key provider payload"
    def fail(session, **kwargs):
        session.stage = secret
        session.pending.append({"prompt": secret})
        warm.base._fail("unexpected_notification:" + secret)
    monkeypatch.setattr(warm.base, "inspect_preflight", fail)
    item = client()
    with pytest.raises(warm.ConcurrentStop, match="fixed_other") as caught:
        item.complete(request())
    state = item.budget.snapshot()
    assert state["first_failure"]["rpc"] is None
    assert state["first_failure"]["queue_count"] == 1
    assert secret not in str(caught.value)
    assert secret not in repr(state)


def test_first_fault_survives_cleanup_failure(monkeypatch):
    monkeypatch.setattr(warm.base, "inspect_preflight", admission)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 7))
    FakeSession.fail_unsubscribe = warm.base.SubscriptionTransportError("unsubscribe_unverified")
    FakeSession.fail_close = RuntimeError("secret cleanup path")
    item = client()
    with pytest.raises(warm.ConcurrentStop, match="cleanup_failure"):
        item.complete(request())
    state = item.budget.snapshot()
    assert state["stop_code"] == "unsubscribe_unverified"
    assert state["first_failure"]["code"] == "unsubscribe_unverified"
    assert item.session is FakeSession.instances[0]
    FakeSession.fail_close = None
    item.close()
    assert item.session is None and FakeSession.instances[0].closed


def test_cleanup_as_first_fault_fails_closed(monkeypatch):
    monkeypatch.setattr(warm.base, "inspect_preflight", admission)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 7))
    item = client(max_requests=1)
    item.complete(request())
    FakeSession.fail_close = RuntimeError("secret cleanup path")
    with pytest.raises(warm.ConcurrentStop, match="cleanup_failure"):
        item.complete(request())
    state = item.budget.snapshot()
    assert state["first_failure"]["code"] == "cleanup_failure"
    assert state["first_failure"]["phase"] == "rotation_cleanup"
    assert state["turns"] == 1 and len(FakeSession.instances) == 1
    FakeSession.fail_close = None
    item.close()


def test_final_close_failure_is_first_fault():
    item = client()
    item.session = FakeSession("unused", "unused")
    FakeSession.fail_close = RuntimeError("secret cleanup path")
    with pytest.raises(warm.ConcurrentStop, match="cleanup_failure"):
        item.close()
    state = item.budget.snapshot()
    assert state["stop_code"] == "cleanup_failure"
    assert state["first_failure"]["code"] == "cleanup_failure"
    assert state["first_failure"]["phase"] == "cleanup"
    assert item.session is FakeSession.instances[0]
    FakeSession.fail_close = None
    item.close()
    assert FakeSession.instances[0].closed


def test_successful_close_does_not_claim_peer_cleanup_failure():
    budget = warm.SharedBudget(warm.BudgetLimits(100, 10000, 1000))
    peer = client(budget=budget, question_id="peer")
    peer.session = FakeSession("unused", "unused")
    budget.halt("cleanup_failure")
    peer.close()
    assert FakeSession.instances[0].closed
    assert budget.snapshot()["first_failure"] is None


def test_peer_stop_remains_peer_stop(monkeypatch):
    monkeypatch.setattr(warm.base, "inspect_preflight", admission)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 7))
    budget = warm.SharedBudget(warm.BudgetLimits(100, 10000, 1000))
    first = client(budget=budget, question_id="first")
    peer = client(budget=budget, question_id="peer")
    budget.record_first_failure("quota_floor", {"phase": "preflight"})
    with pytest.raises(warm.ConcurrentStop, match="campaign_stopped"):
        peer.complete(request())
    assert budget.snapshot()["first_failure"] == {"code": "quota_floor", "phase": "preflight"}
    assert not FakeSession.instances
    first.close()
    peer.close()


def test_runner_budget_factory_uses_attributed_type():
    budget = warm.concurrent.SharedBudget(warm.BudgetLimits(2, 100, 10))
    assert type(budget) is warm.SharedBudget
    assert warm.ConcurrentStop is warm.concurrent.ConcurrentStop
    assert budget.snapshot()["first_failure"] is None


def test_actual_sixteen_seventeen_rotation_and_age_boundary(monkeypatch):
    now = [10.0]
    FakeSession.now = staticmethod(lambda: now[0])
    monkeypatch.setattr(warm.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(warm.base, "inspect_preflight", admission)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 5))
    item = client()
    for _ in range(16):
        assert item.complete(request()) == "{}"
    assert len(FakeSession.instances) == 1 and not FakeSession.instances[0].closed
    item.complete(request())
    assert len(FakeSession.instances) == 2 and FakeSession.instances[0].closed
    assert item.rotations == 1 and item.observed_turns == 17
    item.close()

    FakeSession.instances.clear()
    item = client()
    item.complete(request())
    now[0] = 310.0
    item.complete(request())
    assert len(FakeSession.instances) == 2 and FakeSession.instances[0].closed
    assert item.rotations == 1
    item.close()


def test_delayed_unknown_event_still_fails_closed():
    session = object.__new__(warm.WarmSession)
    session.active_thread = None
    session.retired_threads = {"old"}
    session.pending = deque()
    session.starting_events = []
    with pytest.raises(warm.base.SubscriptionTransportError, match="unexpected_notification"):
        session._route({"method": "item/started", "params": {"threadId": "old"}},
                       "account/read", False)
