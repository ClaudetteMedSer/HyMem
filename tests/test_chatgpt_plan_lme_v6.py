"""Offline SIWC 300-second reservation boundary; no provider or auth access."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from benchmarks import chatgpt_plan_lme_v6 as bridge
from benchmarks import codex_subscription_concurrent_v2 as codex_ledger
from hymem.extraction.llm import LLMRequest
from tools.diagnostics.siwc_lme_diagnostic_v8 import RegistrationAlias


def graph(monkeypatch, *, campaign=(8012, 48_160_000, 25_200),
          question=(2000, 12_000_000, 23_400), key="q"):
    broker = object.__new__(bridge.owner.CredentialBroker)
    broker.identity_digest = "a" * 64
    admitted = []
    def acquire(self, *, caller_deadline):
        admitted.append(caller_deadline)
        return bridge.owner.CredentialLease("invented-token", int(time.time()) + 3600)
    monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire", acquire)
    budget = bridge.SharedBudget(bridge.warm.BudgetLimits(*campaign), max_in_flight=4)
    limits = bridge.warm.BudgetLimits(*question)
    forwarded = []
    def response(credentials, system, user, schema, *, timeout):
        assert credentials.access_token == "invented-token"
        forwarded.append((time.monotonic(), timeout, schema))
        return bridge.transport.Completed("invented-answer", 9, 4, 13, 0, 0)
    client = bridge.SIWCLMEClient(broker, budget, key, limits, response_call=response)
    return broker, budget, limits, client, admitted, forwarded, response


def test_real_ordinary_and_structured_forward_300_minus_admission(monkeypatch):
    broker, budget, limits, ordinary, admitted, forwarded, response = graph(monkeypatch)
    structured = bridge.SIWCLMEClient(broker, RegistrationAlias(budget, "q", limits),
        "q", limits, response_call=response)
    request = LLMRequest("invented system", "invented user")
    assert ordinary.complete(request) == "invented-answer"
    source = bridge.staged_v6.classification.GroundingSource(
        7, "Mira uses CairnDB.", source_role="user", source_peer_id="invented",
        source_created_at="2026-09-30")
    triple = bridge.staged_v6.classification.Triple(
        "Mira", "uses", "CairnDB", 1, source_message_id=7)
    staged, batch = bridge.staged_v6.staged.build_original_request((triple,), (source,))
    assert structured.complete_stage(staged, batch, "original", False) == "invented-answer"
    assert len(admitted) == len(forwarded) == 2
    assert forwarded[0][2] is None
    assert forwarded[1][2] == bridge.staged_v6.staged.build_original_output_schema(batch)
    for absolute, (sent_at, allowance, _) in zip(admitted, forwarded):
        assert 299 < allowance <= bridge.MAX_INVOCATION
        assert abs((absolute - sent_at) - allowance) < 0.05
    assert budget.snapshot()["turns"] == 2
    assert budget.snapshot()["known_tokens"] == 26
    assert ordinary.diagnostic_summary()["successes"] == 1
    assert structured.diagnostic_summary()["successes"] == 1


@pytest.mark.parametrize("wall", ["campaign", "question"])
def test_near_wall_clamps_forwarded_timeout(monkeypatch, wall):
    _, budget, _, client, admitted, forwarded, _ = graph(monkeypatch)
    # The opposite wall stays distant. Only the selected outer bound is near.
    if wall == "campaign":
        budget.started_at -= 25_200 - 37
    else:
        budget._questions["q"].started_at -= 23_400 - 37
    assert client.complete(LLMRequest("invented system", "invented user")) == "invented-answer"
    assert 35 < forwarded[0][1] < 38
    assert admitted[0] - forwarded[0][0] == pytest.approx(forwarded[0][1], abs=0.05)
    assert budget.snapshot()["turns"] == 1


def test_codex_reserve_remains_120_and_siwc_reserve_is_300():
    caps = codex_ledger.BudgetLimits(8012, 48_160_000, 25_200)
    qcaps = codex_ledger.BudgetLimits(2000, 12_000_000, 23_400)
    old = codex_ledger.SharedBudget(caps, max_in_flight=4)
    old.register("q", qcaps)
    assert old.reserve("q") <= 120
    new = bridge.SharedBudget(caps, max_in_flight=4)
    new.register("q", qcaps)
    assert 299 < new.reserve("q") <= 300
    assert old.snapshot()["reserved"] == 1
    assert new.snapshot()["reserved"] == 1


def test_clock_failure_before_super_never_leaves_reservation():
    calls = [0]
    def clock():
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError("invented clock fault")
        return 1000.0
    limits = bridge.warm.BudgetLimits(10, 1000, 1000)
    budget = bridge.SharedBudget(limits, max_in_flight=4, clock=clock)
    budget.register("q", limits)
    with pytest.raises(RuntimeError, match="invented clock fault"):
        budget.reserve("q")
    state = budget.snapshot()
    assert state["reserved"] == state["in_flight"] == 0

    calls = [0]
    def fail_only_on_sixth_call():
        calls[0] += 1
        if calls[0] == 6:
            raise RuntimeError("extra post-mutation clock read")
        return 1000.0
    safe = bridge.SharedBudget(limits, max_in_flight=4, clock=fail_only_on_sixth_call)
    safe.register("q", limits)
    assert safe.reserve("q") == 300
    assert calls[0] == 5
    assert safe.snapshot()["reserved"] == 1


def test_super_wall_check_still_rejects_cliff_after_early_sample():
    ticks = iter((0.0, 0.0, 999.0, 1000.1, 1000.1))
    limits = bridge.warm.BudgetLimits(10, 1000, 1000)
    budget = bridge.SharedBudget(limits, max_in_flight=4, clock=lambda: next(ticks))
    budget.register("q", limits)
    with pytest.raises(bridge.warm.ConcurrentStop, match="wall_limit"):
        budget.reserve("q")
    state = budget.snapshot()
    assert state["reserved"] == state["in_flight"] == 0
    assert state["stop_code"] == "campaign_wall_limit"


def test_turn_token_and_concurrency_rejections_preserved(monkeypatch):
    _, turn_budget, _, turn_client, _, turn_forwarded, _ = graph(monkeypatch,
        campaign=(1, 1000, 25_200))
    assert turn_client.complete(LLMRequest("invented", "turn")) == "invented-answer"
    with pytest.raises(bridge.warm.ConcurrentStop, match="campaign_budget_exhausted"):
        turn_client.complete(LLMRequest("invented", "turn"))
    assert len(turn_forwarded) == turn_budget.snapshot()["turns"] == 1

    _, token_budget, _, token_client, _, token_forwarded, _ = graph(monkeypatch,
        campaign=(10, 10, 25_200), key="token")
    assert token_client.complete(LLMRequest("invented", "token")) == "invented-answer"
    with pytest.raises(bridge.warm.ConcurrentStop, match="campaign_budget_exhausted"):
        token_client.complete(LLMRequest("invented", "token"))
    assert len(token_forwarded) == token_budget.snapshot()["turns"] == 1

    _, busy_budget, _, busy_client, _, busy_forwarded, _ = graph(monkeypatch, key="busy")
    busy_budget.reserve("busy")
    with pytest.raises(bridge.warm.ConcurrentStop, match="question_concurrent_invocation"):
        busy_client.complete(LLMRequest("invented", "busy"))
    assert not busy_forwarded
    assert busy_budget.snapshot()["questions"]["busy"]["in_flight"] == 1
    busy_budget.settle("busy", used=None, turn_started=False, failure=None)


def test_four_workers_settle_without_serializing_reservations(monkeypatch):
    broker, budget, _, first, _, forwarded, _ = graph(monkeypatch, key="q0")
    barrier = threading.Barrier(4)
    def response(credentials, system, user, schema, *, timeout):
        forwarded.append((time.monotonic(), timeout, schema))
        barrier.wait(timeout=5)
        return bridge.transport.Completed("invented-answer", 9, 4, 13, 0, 0)
    first.response_call = response
    clients = [first]
    limits = bridge.warm.BudgetLimits(2000, 12_000_000, 23_400)
    for i in range(1, 4):
        clients.append(bridge.SIWCLMEClient(broker, budget, f"q{i}", limits,
                                         response_call=response))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda client: client.complete(
            LLMRequest("invented system", "invented user")), clients))
    assert results == ["invented-answer"] * 4
    assert len(forwarded) == 4
    assert all(299 < timeout <= 300 for _, timeout, _ in forwarded)
    state = budget.snapshot()
    assert state["turns"] == 4 and state["known_tokens"] == 52
    assert state["in_flight"] == 0 and state["reserved"] == 0
