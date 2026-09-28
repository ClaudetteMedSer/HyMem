"""Independent notification and absolute-deadline controls for process reuse."""
from collections import deque
import io
import time
import threading
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_warm as warm
from tools.diagnostics import luna_subscription_lme_warm as runner


def session(*events):
    s = object.__new__(warm.WarmSession)
    s.pending = deque()
    s.next_id = 0
    s.initialized_result = None
    s.initialized_sent = False
    s.active_thread = None
    s.bound_thread_id = None
    s.starting_threads = []
    s.starting_events = []
    s.started_notifications = set()
    s.seen_threads = set()
    s.retired_threads = set()
    s.warning_targets = []
    queue = deque(events)
    s.receive = queue.popleft
    s.send = lambda *args, **kwargs: 1
    return s


def test_delayed_started_notification_owned_once_even_during_turn_events():
    started = {"method": "thread/started", "params": {"thread": {"id": "fresh"}}}
    reply = {"id": 1, "result": {"thread": {"id": "fresh"}}}
    turn = {"id": 1, "result": {"turn": {"id": "turn", "status": "inProgress"}}}
    s = session(reply, started, turn, started)
    s.rpc("thread/start", {})
    s.rpc("turn/start", {}, preserve_notifications=True)
    with pytest.raises(warm.base.SubscriptionTransportError):
        s.next_event()


@pytest.mark.parametrize("late", ["thread/tokenUsage/updated", "item/completed", "turn/completed"])
def test_pending_post_completion_content_is_not_silently_dropped(late):
    s = session()
    s.active_thread = "old"
    s.pending.append({"method": late, "params": {"threadId": "old"}})
    with pytest.raises(warm.base.SubscriptionTransportError):
        s.unsubscribe("old")


def test_lifecycle_pending_tail_and_later_closed_do_not_authorize_id_reuse():
    s = session({"id": 1, "result": {"status": "unsubscribed"}},
        {"method": "thread/closed", "params": {"threadId": "old"}},
        {"id": 1, "result": {"thread": {"id": "old"}}})
    s.active_thread = s.bound_thread_id = "old"
    s.seen_threads.add("old")
    s.pending.append({"method": "thread/status/changed", "params":
                      {"threadId": "old", "status": {"type": "idle"}}})
    s.unsubscribe("old")
    assert not s.pending and s.active_thread is None
    with pytest.raises(warm.base.SubscriptionTransportError):
        s.rpc("thread/start", {})


def test_preflight_does_not_grant_new_turn_deadline(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(warm.time, "monotonic", lambda: now[0])
    instances = []
    class Fake:
        def __init__(self, *args, **kwargs):
            self.created_at = now[0]
            self.closed = False
            instances.append(self)
        def set_deadline(self, deadline):
            self.deadline = deadline
        def unsubscribe(self, thread):
            pass
        def close(self):
            self.closed = True
    budget = warm.SharedBudget(warm.BudgetLimits(3, 100, 1000), clock=lambda: now[0])
    client = warm.WarmSubscriptionClient("unused", budget, "q", warm.BudgetLimits(3, 100, 900),
                                         session_factory=Fake)
    def preflight(s, **kwargs):
        now[0] += 100
        return {"auth": "chatgpt", "model": "gpt-6-luna", "inference_enabled": False,
                "config_isolation_admitted": True, "_thread_id": "fresh",
                "quota_windows": [{"remaining_percent": 80}]}
    def turn(s, *args):
        assert s.deadline == 220.0
        assert s.deadline - now[0] == 20
        now[0] += 21
        warm.base._fail("timeout")
    monkeypatch.setattr(warm.base, "inspect_preflight", preflight)
    monkeypatch.setattr(warm.base, "_run_turn", turn)
    request = SimpleNamespace(system="invented", user="invented", temperature=0,
                              max_tokens=32, response_format="text")
    with pytest.raises(warm.ConcurrentStop):
        client.complete(request)
    assert not client.usage_complete and client.observed_turns == 1
    assert instances[0].closed
    assert budget.snapshot()["stopped"]
    with pytest.raises(warm.ConcurrentStop):
        client.complete(request)
    assert len(instances) == 1
    client.close()


def test_expired_invocation_sends_no_rpc():
    s = object.__new__(warm.WarmSession)
    s.next_id = 0
    s.deadline = time.monotonic() - 1
    s.process = SimpleNamespace(stdin=io.StringIO())
    with pytest.raises(warm.base.SubscriptionTransportError, match="timeout"):
        s.send("turn/start", {"threadId": "invented"})
    assert s.process.stdin.getvalue() == ""


def test_failed_close_keeps_handle_for_final_cleanup():
    budget = warm.SharedBudget(warm.BudgetLimits(2, 50, 30))
    client = warm.WarmSubscriptionClient("unused", budget, "q", warm.BudgetLimits(2, 50, 30))
    class TransientCloseFailure:
        calls = 0
        def close(self):
            self.calls += 1
            if self.calls == 1:
                raise OSError("invented transient cleanup failure")
    process = TransientCloseFailure()
    client.session = process
    with pytest.raises(warm.ConcurrentStop, match="cleanup_failure"):
        client._close_process()
    assert client.session is process
    assert budget.snapshot()["stopped"]
    client.close()
    assert process.calls == 2
    assert client.session is None
    assert budget.snapshot()["stopped"]  # cleanup does not turn a failed run into success


def test_five_questions_never_retain_more_than_four_clients(tmp_path, monkeypatch):
    active, closed, lock = set(), [], threading.Lock()
    barrier = threading.Barrier(4, timeout=3)
    class Client:
        def __init__(self, key, limits, budget):
            self.key, self.budget = key, budget
            budget.register(key, limits)
            with lock:
                assert len(active) < 4
                active.add(key)
        @property
        def observed_turns(self):
            return self.budget.snapshot()["questions"][self.key]["turns"]
        @property
        def observed_tokens(self):
            return self.budget.snapshot()["questions"][self.key]["known_tokens"]
        @property
        def usage_complete(self):
            return self.budget.snapshot()["questions"][self.key]["usage_complete"]
        def complete(self, request):
            self.budget.reserve(self.key)
            self.budget.before_turn(self.key, {"auth": "chatgpt", "model": "gpt-6-luna",
                "inference_enabled": False, "config_isolation_admitted": True,
                "quota_windows": [{"remaining_percent": 80}]})
            if self.key in {"q-0000", "q-0001", "q-0002", "q-0003"}:
                barrier.wait()
            self.budget.settle(self.key, used=5, turn_started=True)
            return "{}"
        def close(self):
            with lock:
                active.remove(self.key)
                closed.append(self.key)
    class Adapter:
        def __init__(self, path, **kwargs):
            self.path = path
            self.last_indexing_summary = None
        def open(self):
            pass
        def close(self):
            pass
    def canary(_a, _b, client, **kwargs):
        client.complete(SimpleNamespace())
        return {"passed": True}
    def evaluate(reader, judge, adapter, question, **kwargs):
        assert "canary" in closed
        reader.chat([{"role": "user", "content": "invented"}])
        indexing = {"outcome": "success", "healthy": True, "summary_healthy": True}
        adapter.last_indexing_summary = indexing
        return {"indexing": indexing, "correct": question["index"] != 1,
                "benchmark_failure": None, "judge_error": False, "judge_parse_valid": True}
    monkeypatch.setattr(runner.old, "experimental_canary", canary)
    monkeypatch.setattr(runner.old, "make_adapter_class", lambda *a: Adapter)
    monkeypatch.setattr(runner, "terminalize_owned_dream_run", lambda *a, **kw: {"open_runs_after": 0})
    result = runner.run_campaign(concurrent=warm.concurrent, request_type=lambda **kw: SimpleNamespace(**kw),
        canary=None, chunk=None,
        lme=SimpleNamespace(evaluate_question=evaluate, IndexingConvergenceError=ValueError,
                            DEFAULT_MAX_INPUT_TOKENS=100, DEFAULT_MAX_INPUT_BYTES=1000),
        protocol=SimpleNamespace(_validate_versioned_indexing=lambda *a, **kw: True),
        binary="unused", questions=[{"index": i} for i in range(5)], output=tmp_path,
        campaign_limits=warm.BudgetLimits(10, 100, 60), question_limits=warm.BudgetLimits(1, 10, 50),
        canary_limits=warm.BudgetLimits(1, 10, 20), indexing_timeout_s=40, workers=4,
        client_factory=lambda key, limits, budget: Client(key, limits, budget))
    assert result["campaign_stop"] is None and result["usage_complete_now"]
    assert result["budget"]["turns"] == 6 and result["budget"]["known_tokens"] == 30
    assert [q["index"] for q in result["questions"]] == list(range(5))
    assert all(q["question_completed"] and q["cleanup_ok"] for q in result["questions"])
    assert result["questions"][1]["correct"] is False
    assert not active and len(closed) == 6
