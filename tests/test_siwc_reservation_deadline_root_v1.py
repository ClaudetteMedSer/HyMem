"""Independent offline regression controls for actual SIWC deadline propagation."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from benchmarks import chatgpt_plan_lme_v5 as old
from benchmarks import chatgpt_plan_lme_v6 as bridge


def broker_for(module, monkeypatch, acquire=None):
    broker = object.__new__(module.owner.CredentialBroker)
    broker.identity_digest = "a" * 64
    monkeypatch.setattr(module.owner.CredentialBroker, "acquire", acquire or
        (lambda self, **kw: module.owner.CredentialLease("invented-not-a-token", int(time.time()) + 900)))
    return broker


@pytest.mark.parametrize("module,expected", [(old, 120), (bridge, 300)])
def test_old_failure_and_new_actual_forwarded_deadline(module, expected, monkeypatch):
    seen = []
    def response(*args, timeout):
        seen.append(timeout)
        return module.transport.Completed("invented", 2, 1, 3, 0, 0)
    budget = module.SharedBudget(module.warm.BudgetLimits(8012, 48160000, 25200))
    client = module.SIWCLMEClient(broker_for(module, monkeypatch), budget, "q",
        module.warm.BudgetLimits(2000, 12000000, 23400), response_call=response)
    assert client.complete(module.LLMRequest("invented system", "invented user")) == "invented"
    assert expected - 1 < seen[0] <= expected
    snap = budget.snapshot()
    assert snap["turns"] == 1 and snap["known_tokens"] == 3
    assert snap["usage_complete"] and snap["in_flight"] == snap["reserved"] == 0
    client.close()


def _assert_deadline_child(send, slots, credentials, request, timeout):
    """Use the actual parent/spawn/parser; replace only external I/O with invention."""
    from tests.test_chatgpt_plan_responses_root_v1 import terminal
    def deny(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise AssertionError("network_forbidden")
    sys.addaudithook(deny)
    assert 299 < timeout <= 300
    assert credentials.access_token == "invented-not-a-token"
    assert request["model"] == "gpt-5.6-luna"
    assert request["reasoning"] == {"effort": "low"}
    assert request["store"] is False and request["stream"] is True
    progress = bridge.transport._Progress(slots, timeout)
    progress.mark("child_entry")
    value = bridge.transport.parse_stream_events([terminal()])
    progress.mark("result_ipc", ready=True, ipc=True)
    send.send(("ok", value))
    send.close()


def test_four_default_transport_calls_receive_300_and_recursively_settle(monkeypatch):
    before = {p.pid for p in multiprocessing.active_children()}
    broker = broker_for(bridge, monkeypatch)
    monkeypatch.setattr(bridge.transport, "_child", _assert_deadline_child)
    budget = bridge.SharedBudget(bridge.warm.BudgetLimits(8012, 48160000, 25200))
    clients = [bridge.SIWCLMEClient(broker, budget, str(i),
        bridge.warm.BudgetLimits(2000, 12000000, 23400)) for i in range(4)]
    assert all(c.response_call is bridge.transport.complete for c in clients)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(lambda c: c.complete(bridge.LLMRequest("invented s", "invented u")), clients))
        assert values == [" invented\n"] * 4
        snap = budget.snapshot()
        assert snap["turns"] == 4 and snap["known_tokens"] > 0
        assert snap["in_flight"] == snap["reserved"] == 0 and snap["usage_complete"]
        for c in clients:
            summary = bridge.validate_summary_projection(c.diagnostic_summary())
            assert summary["calls"] == summary["successes"] == 1
            assert summary["failures"] == 0 and summary["usage_complete"]
    finally:
        for c in clients:
            c.close()
    assert {p.pid for p in multiprocessing.active_children()} == before


@pytest.mark.parametrize("campaign,question,expected", [(25200, 23400, 293), (37, 900, 30), (900, 37, 30)])
def test_admission_consumes_same_deadline_and_outer_walls_clip(campaign, question, expected, monkeypatch):
    clock = [100.0]
    fake = SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 2000.0)
    monkeypatch.setattr(bridge, "time", fake)
    def acquire(self, *, caller_deadline):
        assert caller_deadline == 100 + min(300, campaign, question)
        clock[0] += 7
        return bridge.owner.CredentialLease("invented-not-a-token", 3000)
    broker = broker_for(bridge, monkeypatch, acquire)
    seen = []
    def response(*args, timeout):
        seen.append(timeout)
        return bridge.transport.Completed("invented", 2, 1, 3, 0, 0)
    budget = bridge.SharedBudget(bridge.warm.BudgetLimits(30, 10000, campaign), clock=fake.monotonic)
    client = bridge.SIWCLMEClient(broker, budget, "q", bridge.warm.BudgetLimits(20, 9000, question), response_call=response)
    assert client.complete(bridge.LLMRequest("invented s", "invented u")) == "invented"
    assert seen == [expected]
    assert budget.snapshot()["turns"] == 1 and budget.snapshot()["in_flight"] == 0


def test_original_codex_ledger_remains_120():
    budget = bridge.warm.concurrent.SharedBudget(bridge.warm.BudgetLimits(100, 10000, 2000), clock=lambda: 0)
    budget.register("q", bridge.warm.BudgetLimits(50, 5000, 1000))
    assert budget.reserve("q") == 120


def test_only_new_reserve_method_changes_bridge_source():
    import ast
    before = ast.parse(Path(old.__file__).read_text())
    after = ast.parse(Path(bridge.__file__).read_text())
    cls = next(node for node in after.body if isinstance(node, ast.ClassDef) and node.name == "SharedBudget")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "reserve"]
    assert len(methods) == 1
    cls.body.remove(methods[0])
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)


@pytest.mark.parametrize("case", ["campaign_stop", "question_stop", "question_busy",
    "campaign_wall", "question_wall", "campaign_turns", "campaign_tokens",
    "question_turns", "question_tokens", "concurrency"])
def test_reservation_guards_and_mutations_equal_historical_ledger(case):
    states = []
    codes = []
    for module in (old, bridge):
        budget = module.SharedBudget(module.warm.BudgetLimits(100, 10000, 2000), clock=lambda: 0)
        budget.register("q", module.warm.BudgetLimits(50, 5000, 1000))
        q = budget._questions["q"]
        if case == "campaign_stop": budget.stopped = True
        elif case == "question_stop": q.stopped = True
        elif case == "question_busy": q.in_flight = 1
        elif case == "campaign_wall": budget.started_at = -2001
        elif case == "question_wall": q.started_at = -1001
        elif case == "campaign_turns": budget.turns, budget.reserved = 99, 1
        elif case == "campaign_tokens": budget.known_tokens = 10000
        elif case == "question_turns": q.turns, q.reserved = 49, 1
        elif case == "question_tokens": q.known_tokens = 5000
        elif case == "concurrency": budget.in_flight = 4
        with pytest.raises(module.warm.ConcurrentStop) as error:
            budget.reserve("q")
        codes.append(error.value.args)
        states.append(budget.snapshot())
    assert codes[0] == codes[1] and states[0] == states[1]
