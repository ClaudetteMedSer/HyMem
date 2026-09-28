"""Offline race and accounting controls for the separate Luna client."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_concurrent as concurrent
from hymem.extraction import chunk


def request():
    return SimpleNamespace(system="synthetic system", user="synthetic user",
                           temperature=0, max_tokens=32, response_format="json")


def admission(*, quota=75):
    return {"auth": "chatgpt", "model": concurrent.base.MODEL,
            "config_isolation_admitted": True, "inference_enabled": False,
            "quota_windows": [{"remaining_percent": quota}], "_thread_id": "thread-1"}


class Session:
    def __init__(self, binary, cwd, timeout=120):
        self.closed = False
        self.timeout = timeout

    def close(self):
        self.closed = True


def test_limits_validate_finite_positive():
    for value in (0, -1, True, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            concurrent.BudgetLimits(1, 1, value)
    with pytest.raises(ValueError):
        concurrent.BudgetLimits(0, 1, 1)
    with pytest.raises(ValueError):
        concurrent.BudgetLimits(1, True, 1)
    with pytest.raises(ValueError):
        concurrent.SharedBudget(concurrent.BudgetLimits(2, 10, 10), max_in_flight=3)


def test_two_real_overlapping_turns_share_one_aggregate(monkeypatch):
    barrier = threading.Barrier(2, timeout=3)
    sessions = []
    def factory(*args, **kwargs):
        instance = Session(*args, **kwargs)
        sessions.append(instance)
        return instance
    monkeypatch.setattr(concurrent.base, "inspect_preflight", lambda *a, **kw: admission())
    def run_turn(*args):
        barrier.wait()
        return "{}", 7
    monkeypatch.setattr(concurrent.base, "_run_turn", run_turn)
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(2, 20, 30))
    clients = [concurrent.ConcurrentSubscriptionClient("unused", budget, str(i),
               concurrent.BudgetLimits(1, 10, 30), session_factory=factory)
               for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda client: client.complete(request()), clients)) == ["{}", "{}"]
    state = budget.snapshot()
    assert state["turns"] == 2 and state["known_tokens"] == 14
    assert state["in_flight"] == state["reserved"] == 0
    assert state["usage_complete"] is True
    assert len(sessions) == 2 and all(s.closed for s in sessions)
    assert all(amount >= 0 for amount in state["timings"].values())


def test_question_usage_is_not_masked_by_sibling_inflight(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(concurrent.base, "inspect_preflight", lambda *a, **kw: admission())
    def run_turn(session, *args):
        if threading.current_thread().name == "slow":
            entered.set()
            assert release.wait(3)
        return "{}", 5
    monkeypatch.setattr(concurrent.base, "_run_turn", run_turn)
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(3, 100, 30))
    fast = concurrent.ConcurrentSubscriptionClient("unused", budget, "fast",
        concurrent.BudgetLimits(2, 50, 30), session_factory=Session)
    slow = concurrent.ConcurrentSubscriptionClient("unused", budget, "slow",
        concurrent.BudgetLimits(2, 50, 30), session_factory=Session)
    with ThreadPoolExecutor(max_workers=2) as pool:
        def call_slow():
            threading.current_thread().name = "slow"
            return slow.complete(request())
        future = pool.submit(call_slow)
        assert entered.wait(3)
        assert fast.complete(request()) == "{}"
        assert fast.usage_complete is True
        assert budget.snapshot()["usage_complete"] is False
        release.set()
        assert future.result() == "{}"


def test_aggregate_cap_not_multiplied_by_worker(monkeypatch):
    monkeypatch.setattr(concurrent.base, "inspect_preflight", lambda *a, **kw: admission())
    monkeypatch.setattr(concurrent.base, "_run_turn", lambda *a: ("{}", 11))
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(1, 20, 30))
    a = concurrent.ConcurrentSubscriptionClient("unused", budget, "a",
        concurrent.BudgetLimits(5, 100, 30), session_factory=Session)
    b = concurrent.ConcurrentSubscriptionClient("unused", budget, "b",
        concurrent.BudgetLimits(5, 100, 30), session_factory=Session)
    assert a.complete(request()) == "{}"
    with pytest.raises(concurrent.ConcurrentStop, match="campaign_budget_exhausted"):
        b.complete(request())
    assert budget.snapshot()["turns"] == 1


def test_stop_between_preflight_and_turn_blocks_sibling(monkeypatch):
    b_in_preflight = threading.Event()
    release_b = threading.Event()
    turn_calls = []
    def factory(binary, cwd, timeout=120):
        session = Session(binary, cwd, timeout)
        session.label = threading.current_thread().name
        return session
    def inspect(session, **kw):
        if session.label == "B":
            b_in_preflight.set()
            assert release_b.wait(3)
            return admission()
        assert b_in_preflight.wait(3)
        return admission(quota=24)
    monkeypatch.setattr(concurrent.base, "inspect_preflight", inspect)
    monkeypatch.setattr(concurrent.base, "_run_turn", lambda *a: turn_calls.append(1))
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(3, 100, 30))
    a = concurrent.ConcurrentSubscriptionClient("unused", budget, "a",
        concurrent.BudgetLimits(2, 100, 30), session_factory=factory)
    b = concurrent.ConcurrentSubscriptionClient("unused", budget, "b",
        concurrent.BudgetLimits(2, 100, 30), session_factory=factory)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="worker") as pool:
        # Explicit names avoid dependence on executor naming conventions.
        def call(client, name):
            threading.current_thread().name = name
            return client.complete(request())
        fb = pool.submit(call, b, "B")
        assert b_in_preflight.wait(3)
        fa = pool.submit(call, a, "A")
        with pytest.raises(concurrent.ConcurrentStop):
            fa.result()
        release_b.set()
        with pytest.raises(concurrent.ConcurrentStop):
            fb.result()
    state = budget.snapshot()
    assert state["stopped"] is True and state["turns"] == 0
    assert state["in_flight"] == state["reserved"] == 0
    assert turn_calls == []


def test_fatal_preflight_halts_before_slow_cleanup(monkeypatch):
    entered_close = threading.Event()
    release_close = threading.Event()
    class SlowClose(Session):
        def close(self):
            entered_close.set()
            assert release_close.wait(3)
            self.closed = True
    def fail_preflight(*args, **kwargs):
        raise concurrent.base.SubscriptionTransportError("private auth detail")
    monkeypatch.setattr(concurrent.base, "inspect_preflight", fail_preflight)
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(3, 100, 30))
    a = concurrent.ConcurrentSubscriptionClient("unused", budget, "a",
        concurrent.BudgetLimits(2, 50, 30), session_factory=SlowClose)
    b = concurrent.ConcurrentSubscriptionClient("unused", budget, "b",
        concurrent.BudgetLimits(2, 50, 30), session_factory=Session)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(a.complete, request())
        assert entered_close.wait(3)
        assert budget.snapshot()["stopped"] is True
        with pytest.raises(concurrent.ConcurrentStop, match="campaign_stopped"):
            b.complete(request())
        release_close.set()
        with pytest.raises(concurrent.ConcurrentStop):
            future.result()
    assert budget.snapshot()["in_flight"] == 0


def test_unknown_usage_sticky_preserves_prior_known_tokens(monkeypatch):
    monkeypatch.setattr(concurrent.base, "inspect_preflight", lambda *a, **kw: admission())
    calls = 0
    def run(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise concurrent.base.SubscriptionTransportError("private provider text")
        return "{}", 9
    monkeypatch.setattr(concurrent.base, "_run_turn", run)
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(3, 100, 30))
    client = concurrent.ConcurrentSubscriptionClient("unused", budget, "q",
        concurrent.BudgetLimits(3, 100, 30), session_factory=Session)
    client.complete(request())
    with pytest.raises(concurrent.ConcurrentStop) as error:
        client.complete(request())
    assert "private provider text" not in str(error.value)
    state = budget.snapshot()
    assert state["known_tokens"] == 9 and state["turns"] == 2
    assert state["usage_complete"] is False and state["stopped"] is True
    with pytest.raises(concurrent.ConcurrentStop):
        client.complete(request())
    assert calls == 2


def test_cleanup_failure_stops_globally_and_retains_known_usage(monkeypatch):
    class BrokenClose(Session):
        def close(self):
            self.closed = True
            raise OSError("private cleanup detail")
    monkeypatch.setattr(concurrent.base, "inspect_preflight", lambda *a, **kw: admission())
    monkeypatch.setattr(concurrent.base, "_run_turn", lambda *a: ("{}", 5))
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(3, 100, 30))
    client = concurrent.ConcurrentSubscriptionClient("unused", budget, "q",
        concurrent.BudgetLimits(3, 100, 30), session_factory=BrokenClose)
    with pytest.raises(concurrent.ConcurrentStop, match="cleanup_failure"):
        client.complete(request())
    state = budget.snapshot()
    assert state["known_tokens"] == 5 and state["turns"] == 1
    assert state["stopped"] is True and state["in_flight"] == 0


def test_per_question_budget_stops_only_that_question(monkeypatch):
    monkeypatch.setattr(concurrent.base, "inspect_preflight", lambda *a, **kw: admission())
    monkeypatch.setattr(concurrent.base, "_run_turn", lambda *a: ("{}", 5))
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(4, 100, 30))
    a = concurrent.ConcurrentSubscriptionClient("unused", budget, "a",
        concurrent.BudgetLimits(1, 10, 30), session_factory=Session)
    b = concurrent.ConcurrentSubscriptionClient("unused", budget, "b",
        concurrent.BudgetLimits(2, 10, 30), session_factory=Session)
    a.complete(request())
    with pytest.raises(concurrent.ConcurrentStop, match="question_budget_exhausted"):
        a.complete(request())
    assert b.complete(request()) == "{}"
    assert budget.snapshot()["stopped"] is False


def test_wall_limit_and_quota_failure(monkeypatch):
    now = [0.0]
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(3, 100, 10), clock=lambda: now[0])
    budget.register("a", concurrent.BudgetLimits(2, 100, 2))
    now[0] = 3.0
    with pytest.raises(concurrent.ConcurrentStop, match="wall_limit"):
        budget.reserve("a")
    assert budget.snapshot()["stopped"] is False
    monkeypatch.setattr(concurrent.base, "inspect_preflight", lambda *a, **kw: admission(quota=24))
    other = concurrent.ConcurrentSubscriptionClient("unused", budget, "b",
        concurrent.BudgetLimits(2, 100, 10), session_factory=Session)
    with pytest.raises(concurrent.ConcurrentStop):
        other.complete(request())
    assert budget.snapshot()["stopped"] is True


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_preflight_base_exception_stops_and_settles(monkeypatch, interrupt):
    def broken(*args, **kwargs):
        raise interrupt()
    monkeypatch.setattr(concurrent.base, "inspect_preflight", broken)
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(3, 100, 30))
    client = concurrent.ConcurrentSubscriptionClient("unused", budget, "q",
        concurrent.BudgetLimits(2, 50, 30), session_factory=Session)
    with pytest.raises(interrupt):
        client.complete(request())
    state = budget.snapshot()
    assert state["stopped"] is True and state["stop_code"] == "interrupted_invocation"
    assert state["reserved"] == state["in_flight"] == state["turns"] == 0


def test_mutated_helper_rejected_before_execution(tmp_path):
    module = Path(concurrent.__file__)
    (tmp_path / module.name).write_bytes(module.read_bytes())
    (tmp_path / "codex_subscription.py").write_text("raise RuntimeError('must never execute')\n")
    spec = importlib.util.spec_from_file_location("untrusted_concurrent", tmp_path / module.name)
    candidate = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="pinned_transport_source_mismatch"):
        spec.loader.exec_module(candidate)


def test_control_stop_escapes_real_frozen_extractor_without_retry():
    class StoppedClient:
        calls = 0

        def complete(self, request):
            self.calls += 1
            raise concurrent.ConcurrentStop("campaign_budget_exhausted")
    client = StoppedClient()
    with pytest.raises(concurrent.ConcurrentStop, match="campaign_budget_exhausted"):
        chunk.extract_chunk(client, "synthetic source content")
    assert client.calls == 1


def test_ledger_rejects_double_settlement_and_turn_without_reservation():
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(3, 100, 30))
    budget.register("q", concurrent.BudgetLimits(2, 50, 30))
    with pytest.raises(concurrent.ConcurrentStop, match="ledger_protocol_violation"):
        budget.before_turn("q", admission())
    assert budget.snapshot()["stopped"] is True

    second = concurrent.SharedBudget(concurrent.BudgetLimits(3, 100, 30))
    second.register("q", concurrent.BudgetLimits(2, 50, 30))
    second.reserve("q")
    second.before_turn("q", admission())
    second.settle("q", used=4, turn_started=True)
    with pytest.raises(concurrent.ConcurrentStop, match="ledger_protocol_violation"):
        second.settle("q", used=4, turn_started=True)
    assert second.snapshot()["known_tokens"] == 4
    assert second.snapshot()["in_flight"] == 0
