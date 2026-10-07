"""Offline identity, binding, and setup-cleanup controls for staged observer v6."""
from __future__ import annotations

import copy
import inspect
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import codex_subscription_staged_v6 as transport


def sample():
    source = transport.classification.GroundingSource(
        7, "Mira uses CairnDB.", source_role="user",
        source_peer_id="invented", source_created_at="2026-09-30")
    triple = transport.classification.Triple(
        "Mira", "uses", "CairnDB", 1, source_message_id=7)
    return transport.staged.build_original_request((triple,), (source,))


def client(*, session_factory):
    limits = transport.warm.BudgetLimits(4, 1000, 1000)
    budget = transport.warm.SharedBudget(limits)
    return transport.StagedSubscriptionClient(
        "unused", budget, "q", limits, session_factory=session_factory)


def test_one_observer_graph_and_exact_wrapper_hook():
    observer = transport.observer
    assert transport.warm is observer.warm
    assert transport.warm.SharedBudget is observer.SharedBudget
    assert transport.StagedSubscriptionClient.__mro__[1] is observer.TimeoutSubscriptionClient
    assert inspect.signature(transport.StagedSubscriptionClient).parameters[
        "session_factory"].default is observer.TimeoutSession

    item = object.__new__(transport.StagedSubscriptionClient)
    raw = object.__new__(observer.TimeoutSession)
    bound = transport._BoundSession(raw, lambda *_: None)
    item.session = bound
    assert item._observed_session() is raw
    assert item.session is bound
    item.session = object()
    assert item._observed_session() is None
    item.session = transport._BoundSession(object(), lambda *_: None)
    assert item._observed_session() is None
    class ForeignWrapper(transport._BoundSession):
        pass
    item.session = ForeignWrapper(raw, lambda *_: None)
    assert item._observed_session() is None


def test_bound_session_adds_only_trusted_schema_to_fresh_thread():
    class Raw:
        calls = []

        def rpc(self, method, params, *, preserve_notifications=False):
            self.calls.append((method, copy.deepcopy(params)))
            return {"thread": {"id": "fresh"}} if method == "thread/start" else {}

    raw = Raw()
    bound = transport._BoundSession(raw, lambda *_: None)
    schema = {"type": "object", "properties": {"schema": {"enum": ["trusted"]}}}
    bound.bind("trusted-system", "trusted-user", schema)
    schema["properties"]["schema"]["enum"] = ["mutated"]
    bound.rpc("thread/start", {"baseInstructions": "trusted-system"})
    with pytest.raises(transport.warm.base.SubscriptionTransportError):
        bound.rpc("turn/start", {"threadId": "fresh", "input": [
            {"type": "text", "text": "trusted-user"}], "outputSchema": {}})
    assert len(raw.calls) == 1
    bound.rpc("turn/start", {"threadId": "fresh", "input": [
        {"type": "text", "text": "trusted-user"}]})
    assert raw.calls[-1][1]["outputSchema"] == {
        "type": "object", "properties": {"schema": {"enum": ["trusted"]}}}
    with pytest.raises(transport.warm.base.SubscriptionTransportError):
        bound.rpc("turn/start", {"threadId": "fresh", "input": [
            {"type": "text", "text": "trusted-user"}]})
    bound.clear_binding()
    with pytest.raises(transport.warm.base.SubscriptionTransportError):
        bound.rpc("thread/start", {"baseInstructions": "trusted-system"})


def test_new_wrapper_bind_failure_closes_raw_and_records_no_admission(monkeypatch):
    class Raw:
        closed = False

        def close(self):
            self.closed = True

    raw = Raw()
    item = client(session_factory=lambda *args, **kwargs: raw)
    request, batch = sample()
    def reject_bind(*args):
        raise ValueError("private-sentinel")
    monkeypatch.setattr(transport._BoundSession, "bind", reject_bind)
    try:
        with pytest.raises(transport.warm.ConcurrentStop) as caught:
            item.complete_stage(request, batch, "original", False)
        assert "private-sentinel" not in str(caught.value)
        assert raw.closed and item._pending_raw is None and item.session is None
        assert item._active_binding is None
        assert item.budget.snapshot()["turns"] == 0
        assert item.diagnostic_summary()["calls"] == 1
    finally:
        item.close()


@pytest.mark.parametrize("mutation", [
    "fake=types.ModuleType('benchmarks.codex_subscription_timeout_v3');"
    "fake.__file__='/private/foreign.py';sys.modules[fake.__name__]=fake",
    "from benchmarks import codex_subscription_timeout_v3 as observer;"
    "observer.SharedBudget=object",
    "from benchmarks import codex_subscription_timeout_v3 as observer;"
    "observer.TimeoutSession=type('ForeignSession',(observer.TimeoutSession,),{})",
])
def test_foreign_cached_observer_or_graph_rejected_before_use(mutation):
    repo = Path(__file__).resolve().parents[1]
    script = (f"import sys,types;sys.path.insert(0,{str(repo)!r});"
              + mutation + ";from benchmarks import codex_subscription_staged_v6")
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "pinned_observer_v3_import_mismatch" in result.stderr
