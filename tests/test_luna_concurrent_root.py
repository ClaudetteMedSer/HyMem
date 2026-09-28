"""Independent root controls: no model calls and no subscription process."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from benchmarks import codex_subscription_concurrent as subject
from hymem.extraction.llm import LLMRequest


def admission():
    return {"auth": "chatgpt", "model": "gpt-6-luna",
            "config_isolation_admitted": True, "inference_enabled": False,
            "quota_windows": [{"remaining_percent": 80}], "_thread_id": "fresh"}


class Session:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def close(self):
        self.closed = True


def setup_clients(monkeypatch, *, turns=20, tokens=10000):
    monkeypatch.setattr(subject.base, "inspect_preflight", lambda *a, **k: admission())
    budget = subject.SharedBudget(subject.BudgetLimits(turns, tokens, 600))
    clients = [subject.ConcurrentSubscriptionClient(
        "unused", budget, key, subject.BudgetLimits(10, 5000, 300),
        session_factory=Session) for key in ("q0", "q1")]
    return budget, clients


def test_completed_peer_usage_does_not_depend_on_other_inflight(monkeypatch):
    budget, clients = setup_clients(monkeypatch)
    in_second = threading.Event()
    release_second = threading.Event()

    def infer(_session, _thread, user):
        if user == "second":
            in_second.set()
            assert release_second.wait(3)
        return "answer", 50

    monkeypatch.setattr(subject.base, "_run_turn", infer)
    with ThreadPoolExecutor(2) as pool:
        second = pool.submit(clients[1].complete, LLMRequest("s", "second"))
        assert in_second.wait(3)
        try:
            assert clients[0].complete(LLMRequest("s", "first")) == "answer"
            assert clients[0].usage_complete is True
            assert clients[0].observed_tokens == 50
            assert budget.snapshot()["usage_complete"] is False
        finally:
            release_second.set()
        assert second.result(3) == "answer"
    assert budget.snapshot()["known_tokens"] == 100
    assert budget.snapshot()["usage_complete"] is True


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_preflight_base_exception_globally_stops(monkeypatch, exception):
    budget, clients = setup_clients(monkeypatch)
    def rejected(*args, **kwargs):
        raise exception()
    monkeypatch.setattr(subject.base, "inspect_preflight", rejected)
    with pytest.raises(BaseException):
        clients[0].complete(LLMRequest("s", "u"))
    state = budget.snapshot()
    assert state["stopped"] is True
    assert state["turns"] == state["in_flight"] == state["reserved"] == 0
    with pytest.raises(subject.ConcurrentStop):
        clients[1].complete(LLMRequest("s", "u"))


def test_cleanup_error_retains_reported_usage(monkeypatch):
    budget, _clients = setup_clients(monkeypatch)
    monkeypatch.setattr(subject.base, "_run_turn", lambda *a: ("answer", 31))
    class BadCleanup(Session):
        def close(self):
            raise RuntimeError("private exception")
    client = subject.ConcurrentSubscriptionClient(
        "unused", budget, "q2", subject.BudgetLimits(10, 1000, 100),
        session_factory=BadCleanup)
    with pytest.raises(subject.ConcurrentStop):
        client.complete(LLMRequest("s", "u"))
    state = budget.snapshot()
    assert state["known_tokens"] == 31
    assert state["turns"] == 1
    assert state["in_flight"] == 0
    assert state["stopped"] is True
    assert "private" not in str(state)


def test_aggregate_last_slot_is_not_duplicated(monkeypatch):
    budget, clients = setup_clients(monkeypatch, turns=1)
    entered = threading.Event()
    release = threading.Event()
    def infer(*args):
        entered.set()
        assert release.wait(3)
        return "answer", 30
    monkeypatch.setattr(subject.base, "_run_turn", infer)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(clients[0].complete, LLMRequest("s", "first"))
        assert entered.wait(3)
        try:
            with pytest.raises(subject.ConcurrentStop):
                clients[1].complete(LLMRequest("s", "second"))
        finally:
            release.set()
        assert first.result(3) == "answer"
    assert budget.snapshot()["turns"] == 1
    assert budget.snapshot()["known_tokens"] == 30


def test_preflight_peer_must_recheck_sticky_stop(monkeypatch):
    budget, clients = setup_clients(monkeypatch)
    preflight_done = threading.Event()
    release = threading.Event()
    started = []
    def preflight(*args, **kwargs):
        preflight_done.set()
        assert release.wait(3)
        return admission()
    monkeypatch.setattr(subject.base, "inspect_preflight", preflight)
    monkeypatch.setattr(subject.base, "_run_turn", lambda *a: started.append(True))
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(clients[0].complete, LLMRequest("s", "u"))
        assert preflight_done.wait(3)
        budget.halt("peer_isolation_failure")
        release.set()
        with pytest.raises(subject.ConcurrentStop):
            future.result(3)
    assert started == []
    assert budget.snapshot()["reserved"] == budget.snapshot()["in_flight"] == 0
