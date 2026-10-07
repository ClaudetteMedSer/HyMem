"""Independent root checks for the public SIWC ledger bridge; no live I/O."""
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from benchmarks import chatgpt_plan_lme_v1 as b
from hymem.extraction.llm import LLMRequest


def rig(monkeypatch, *, response=None, limits=None, budget_class=None):
    owner = object.__new__(b.owner.CredentialBroker)
    owner.identity_digest = "a" * 64
    monkeypatch.setattr(b.owner.CredentialBroker, "acquire", lambda self, **kw:
                        b.owner.CredentialLease("invented-not-a-token", int(time.time()) + 900))
    limits = limits or b.warm.BudgetLimits(30, 10000, 300)
    budget = (budget_class or b.SharedBudget)(limits, max_in_flight=4)
    calls = []

    def invoke(*args, **kw):
        calls.append(kw["timeout"])
        if response:
            return response(*args, **kw)
        return b.transport.Completed("invented", 10, 2, 12, 0, 0)

    client = b.SIWCLMEClient(owner, budget, "q", limits, response_call=invoke)
    return owner, budget, client, calls


def test_settled_token_threshold_blocks_next_call(monkeypatch):
    _, budget, client, calls = rig(monkeypatch, limits=b.warm.BudgetLimits(5, 10, 300))
    assert client.complete(LLMRequest("s", "u")) == "invented"
    with pytest.raises(b.warm.ConcurrentStop):
        client.complete(LLMRequest("s", "u"))
    state = budget.snapshot()
    assert state["known_tokens"] == 12  # Observed threshold is not a monetary cap.
    assert state["turns"] == 1 and state["reserved"] == state["in_flight"] == 0
    assert len(calls) == 1


def test_expired_deadline_after_owner_does_not_dispatch(monkeypatch):
    _, budget, client, calls = rig(monkeypatch, limits=b.warm.BudgetLimits(5, 1000, .02))
    def acquire(self, **kw):
        time.sleep(.04)
        return b.owner.CredentialLease("invented", int(time.time()) + 900)
    monkeypatch.setattr(b.owner.CredentialBroker, "acquire", acquire)
    with pytest.raises(b.BridgeError):
        client.complete(LLMRequest("s", "u"))
    state = budget.snapshot()
    assert not calls and state["turns"] == state["in_flight"] == state["reserved"] == 0
    assert state["usage_complete"] and state["stopped"]


def test_unknown_failure_does_not_export_exception_text(monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError("secret body token and private model output")
    _, budget, client, _ = rig(monkeypatch, response=failure)
    with pytest.raises(b.BridgeError, match="^bridge_failure$"):
        client.complete(LLMRequest("private prompt", "private prompt"))
    summary = b.validate_summary_projection(client.diagnostic_summary())
    assert "secret" not in repr(summary) and "private" not in repr(summary)
    assert summary["first_failure"]["unknown_usage"] is True
    state = budget.snapshot()
    assert state["turns"] == 1 and state["known_tokens"] == 0 and not state["usage_complete"]
    assert state["stopped"] and state["reserved"] == state["in_flight"] == 0


def test_owner_binding_is_immutable_even_with_valid_new_digest(monkeypatch):
    owner, budget, client, calls = rig(monkeypatch)
    owner.identity_digest = "b" * 64
    with pytest.raises(b.BridgeError, match="^admission_rejected$"):
        client.complete(LLMRequest("s", "u"))
    assert not calls and budget.snapshot()["turns"] == 0


def test_resource_observed_subclass_is_preserved(monkeypatch):
    class Observed(b.SharedBudget):
        def before_turn(self, qid, proof):
            raise b.warm.ConcurrentStop("resource_task_denial")
    _, budget, client, calls = rig(monkeypatch, budget_class=Observed)
    with pytest.raises(b.BridgeError, match="^resource_task_denial$"):
        client.complete(LLMRequest("s", "u"))
    assert not calls and budget.snapshot()["stop_code"] == "resource_task_denial"


def test_real_four_way_overlap_and_clean_settlement(monkeypatch):
    entered = Event()
    release = Event()
    lock = Lock()
    active = 0
    def response(*args, **kw):
        nonlocal active
        with lock:
            active += 1
            if active == 4:
                entered.set()
        assert release.wait(2)
        return b.transport.Completed("invented", 10, 2, 12, 0, 0)
    owner, budget, first, _ = rig(monkeypatch, response=response)
    clients = [first] + [b.SIWCLMEClient(owner, budget, f"q{i}", budget.limits,
                      response_call=response) for i in range(1, 4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(c.complete, LLMRequest("s", "u")) for c in clients]
        try:
            assert entered.wait(2)
            state = budget.snapshot()
            assert state["in_flight"] == 4 and state["turns"] == 4
            assert not state["usage_complete"]
        finally:
            release.set()
        assert [f.result() for f in pending] == ["invented"] * 4
    state = budget.snapshot()
    assert state["in_flight"] == state["reserved"] == 0
    assert state["known_tokens"] == 48 and state["usage_complete"]


@pytest.mark.parametrize("value", [True, False, -1, float("inf"), float("nan"), "1", 10**100])
def test_projection_numeric_fields_are_strict_and_bounded(monkeypatch, value):
    _, _, client, _ = rig(monkeypatch)
    summary = client.diagnostic_summary()
    for field in ("calls", "successes", "failures", "internal_http_attempts", "admitted_turns", "known_tokens"):
        changed = copy.deepcopy(summary)
        changed[field] = value
        with pytest.raises(ValueError):
            b.validate_summary_projection(changed)


def test_first_failure_is_immutable_across_later_calls(monkeypatch):
    def response(*args, **kw):
        raise b.transport.TransportError("http_failure", 403, "json", "json")
    _, budget, client, calls = rig(monkeypatch, response=response)
    with pytest.raises(b.BridgeError):
        client.complete(LLMRequest("s", "u"))
    original = copy.deepcopy(client.diagnostic_summary()["first_failure"])
    with pytest.raises(b.warm.ConcurrentStop):
        client.complete(LLMRequest("s", "u"))
    assert client.diagnostic_summary()["first_failure"] == original
    assert len(calls) == 1 and budget.snapshot()["stopped"]


@pytest.mark.parametrize("field,value", [("phase", []), ("code", []),
    ("wire_observation", None), ("stream_observation", None), ("http_status", True),
    ("resource_observation", {"current": 1, "peak": 1, "limit": 256, "denials": False})])
def test_first_failure_rejects_malformed_types(field, value):
    failure = {"code": "http_failure", "phase": "http", "turn_admitted": True,
               "unknown_usage": True}
    failure[field] = value
    with pytest.raises(ValueError):
        b.project_first_failure(failure)


def test_invalid_request_is_observed_without_authentication(monkeypatch):
    _, budget, client, calls = rig(monkeypatch)
    def disallow(*args, **kwargs):
        pytest.fail("invalid request reached owner")
    monkeypatch.setattr(b.owner.CredentialBroker, "acquire", disallow)
    with pytest.raises(b.BridgeError):
        client.complete(LLMRequest(None, "invented"))
    summary = b.validate_summary_projection(client.diagnostic_summary())
    assert summary["failures"] == summary["calls"] == 1
    assert summary["first_failure"]["turn_admitted"] is False
    state = budget.snapshot()
    assert state["stopped"] and state["turns"] == state["in_flight"] == state["reserved"] == 0
    assert state["usage_complete"] and not calls
