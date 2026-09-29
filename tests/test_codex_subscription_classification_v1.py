"""Offline wire and lifecycle checks for the inactive classification adapter."""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import replace

import pytest

from benchmarks import codex_subscription_classification_v1 as transport


def sample():
    c = transport.classification
    source = c.GroundingSource(7, "Mira uses CairnDB.", source_role="user",
                               source_peer_id="mira", source_created_at="2026-09-29")
    triple = c.Triple("Mira", "uses", "CairnDB", 1, source_message_id=7)
    return c.build_grounding_request((triple,), (source,))


def events(text="raw-classification"):
    return deque([
        {"method": "turn/started", "params": {"threadId": "thread-1", "turn": {"id": "turn-1"}}},
        {"method": "thread/tokenUsage/updated", "params": {"threadId": "thread-1", "turnId": "turn-1",
            "tokenUsage": {"total": {"totalTokens": 7}}}},
        {"method": "item/started", "params": {"threadId": "thread-1", "turnId": "turn-1",
            "item": {"id": "item-1", "type": "agentMessage"}}},
        {"method": "item/completed", "params": {"threadId": "thread-1", "turnId": "turn-1",
            "item": {"id": "item-1", "type": "agentMessage", "phase": "final_answer", "text": text}}},
        {"method": "turn/completed", "params": {"threadId": "thread-1", "turn": {
            "id": "turn-1", "status": "completed"}}},
    ])


class FakeSession:
    instances = []
    fail_turn = False
    fail_close = False
    unknown_usage = False
    response_text = "raw-classification"

    def __init__(self, binary, cwd, timeout=120):
        self.created_at = transport.warm.time.monotonic()
        self.stage = "initialize"
        self.calls = []
        self.pending = deque()
        self.starting_events = []
        self.retired_threads = set()
        self.last_event = None
        self.rpc_error = None
        self.closed = False
        self.__class__.instances.append(self)

    def set_deadline(self, deadline):
        self.deadline = deadline

    def rpc(self, method, params, *, preserve_notifications=False):
        self.stage = method
        self.calls.append((method, copy.deepcopy(params), preserve_notifications))
        if method == "initialize":
            return {}
        if method == "account/read":
            return {"account": {"type": "chatgpt", "planType": "plus"}}
        if method == "model/list":
            return {"nextCursor": None, "data": [{"model": transport.warm.base.MODEL,
                "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]}
        if method == "account/rateLimits/read":
            return {"rateLimits": {"primary": {"usedPercent": 25}}}
        if method == "config/read":
            return {"config": {"forced_login_method": "chatgpt", "model_provider": "openai",
                "web_search": "disabled", "model": transport.warm.base.MODEL,
                "project_doc_max_bytes": 0, "memories": {"use_memories": False,
                    "generate_memories": False},
                "features": {**{key: False for key in transport.warm.base.DISABLED_FEATURES},
                             "skip_host_skill_discovery": True}, "mcp_servers": {}}}
        if method == "thread/start":
            return {"thread": {"id": "thread-1", "ephemeral": True, "environments": [],
                                "turns": [], "path": None},
                "model": transport.warm.base.MODEL, "modelProvider": "openai",
                "runtimeWorkspaceRoots": [], "instructionSources": [],
                "approvalPolicy": "never", "reasoningEffort": "low",
                "sandbox": {"type": "readOnly", "networkAccess": False}}
        if method == "turn/start":
            if self.fail_turn:
                transport.warm.base._fail("rpc_failure:turn/start")
            self.pending = events(self.response_text)
            if self.unknown_usage:
                self.pending = deque(e for e in self.pending if e["method"] != "thread/tokenUsage/updated")
            return {"turn": {"id": "turn-1", "status": "inProgress"}}
        if method == "thread/unsubscribe":
            return {"status": "unsubscribed"}
        raise AssertionError(method)

    def send(self, method, params, *, notification=False):
        self.calls.append((method, copy.deepcopy(params), notification))

    def next_event(self):
        self.stage = "turn/events"
        return self.pending.popleft()

    def unsubscribe(self, thread_id):
        self.rpc("thread/unsubscribe", {"threadId": thread_id})

    def close(self):
        if self.fail_close:
            raise RuntimeError("private cleanup detail")
        self.closed = True


def client(*, max_requests=16):
    limits = transport.warm.BudgetLimits(20, 1000, 1000)
    budget = transport.warm.SharedBudget(limits)
    return transport.ClassificationSubscriptionClient("unused", budget, "q", limits,
        session_factory=FakeSession, max_requests=max_requests)


def test_actual_preflight_and_turn_parser_receive_exact_schema_and_raw_text():
    FakeSession.instances.clear()
    request, batch = sample()
    item = client()
    assert item.complete_grounding(request, batch) == "raw-classification"
    raw = FakeSession.instances[0]
    turns = [params for method, params, _ in raw.calls if method == "turn/start"]
    assert len(turns) == 1
    assert turns[0]["outputSchema"] == transport.classification.build_output_schema(batch)
    assert turns[0]["input"] == [{"type": "text", "text": request.user}]
    assert turns[0]["model"] == transport.warm.base.MODEL
    assert item.requested_controls[-1]["output_schema_sent"] is True
    assert item.requested_controls[-1]["output_schema_acknowledged"] is True
    assert item.requested_controls[-1]["response_format_effective"] is None
    assert item.observed_tokens == 7
    item.close()


def test_warm_reuse_rebinds_schema_without_mutating_prior_wire_copy():
    FakeSession.instances.clear()
    request, batch = sample()
    item = client()
    item.complete_grounding(request, batch)
    prior = [params for method, params, _ in FakeSession.instances[0].calls if method == "turn/start"][0]
    prior["outputSchema"]["properties"]["batch_sha256"]["enum"] = ["tampered"]
    item.complete_grounding(request, batch)
    assert len(FakeSession.instances) == 1
    turns = [params for method, params, _ in FakeSession.instances[0].calls if method == "turn/start"]
    assert turns[1]["outputSchema"] == transport.classification.build_output_schema(batch)
    assert item.warm_calls == 1
    item.close()


def test_rotation_closes_old_session_and_binds_only_new_one():
    FakeSession.instances.clear()
    request, batch = sample()
    item = client(max_requests=1)
    item.complete_grounding(request, batch)
    item.complete_grounding(request, batch)
    assert len(FakeSession.instances) == 2
    assert FakeSession.instances[0].closed
    for raw in FakeSession.instances:
        turns = [params for method, params, _ in raw.calls if method == "turn/start"]
        assert len(turns) == 1
        assert turns[0]["outputSchema"] == transport.classification.build_output_schema(batch)
    assert item.rotations == 1
    item.close()


def test_raw_response_is_not_mistaken_for_validated_classification():
    FakeSession.instances.clear()
    request, batch = sample()
    item = client()
    FakeSession.response_text = "{malformed"
    try:
        # The real parser returns exactly the completed final item; the caller
        # retains responsibility for schema and semantic validation.
        assert item.complete_grounding(request, batch) == "{malformed"
        with pytest.raises(transport.classification.GroundingContractError):
            transport.classification.parse_grounding_response("{malformed", batch)
        assert item.requested_controls[-1]["output_schema_acknowledged"] is True
        assert item.requested_controls[-1]["response_format_effective"] is None
    finally:
        FakeSession.response_text = "raw-classification"
        item.close()


@pytest.mark.parametrize("change", [
    {"system": "other"}, {"user": "other"}, {"max_tokens": 4097},
    {"temperature": False}, {"response_format": "text"},
])
def test_bad_request_binding_is_rejected_before_session_or_budget(change):
    FakeSession.instances.clear()
    request, batch = sample()
    item = client()
    with pytest.raises(transport.classification.GroundingContractError):
        item.complete_grounding(replace(request, **change), batch)
    assert not FakeSession.instances and not item.requested_controls
    item.close()


def test_invalid_warm_request_closes_owned_session_without_rewriting_prior_metadata():
    FakeSession.instances.clear()
    request, batch = sample()
    item = client()
    item.complete_grounding(request, batch)
    prior = copy.deepcopy(item.requested_controls)
    with pytest.raises(transport.classification.GroundingContractError):
        item.complete_grounding(replace(request, user="wrong"), batch)
    assert item.session is None and FakeSession.instances[0].closed
    assert item.requested_controls == prior
    assert item._active_binding is None
    item.close()


def test_new_session_factory_binding_failure_closes_created_raw(monkeypatch):
    FakeSession.instances.clear()
    request, batch = sample()
    item = client()
    def reject_bind(self, system, user, schema):
        raise RuntimeError("synthetic binding failure")
    monkeypatch.setattr(transport._BoundSession, "bind", reject_bind)
    with pytest.raises(transport.warm.ConcurrentStop):
        item.complete_grounding(request, batch)
    assert FakeSession.instances[0].closed
    assert item.session is None and item._active_binding is None


def test_factory_bind_and_raw_close_failure_retains_owner_for_later_close(monkeypatch):
    FakeSession.instances.clear()
    request, batch = sample()
    item = client()
    def reject_bind(self, system, user, schema):
        raise RuntimeError("private binding detail")
    monkeypatch.setattr(transport._BoundSession, "bind", reject_bind)
    FakeSession.fail_close = True
    try:
        with pytest.raises(transport.warm.ConcurrentStop) as caught:
            item.complete_grounding(request, batch)
        assert "private" not in str(caught.value)
        assert item.session is None and item._pending_raw is FakeSession.instances[0]
        assert not FakeSession.instances[0].closed
    finally:
        FakeSession.fail_close = False
    item.close()
    assert FakeSession.instances[0].closed and item._pending_raw is None


def test_warm_session_binding_failure_closes_owned_raw(monkeypatch):
    FakeSession.instances.clear()
    request, batch = sample()
    item = client()
    item.complete_grounding(request, batch)
    prior = copy.deepcopy(item.requested_controls)
    def reject_bind(self, system, user, schema):
        raise RuntimeError("synthetic binding failure")
    monkeypatch.setattr(transport._BoundSession, "bind", reject_bind)
    with pytest.raises(transport.warm.base.SubscriptionTransportError, match="invalid_request"):
        item.complete_grounding(request, batch)
    assert FakeSession.instances[0].closed and item.session is None
    assert item._active_binding is None
    assert item.requested_controls == prior
    item.close()


def test_wrong_thread_user_duplicate_turn_and_schema_drift_are_rejected():
    request, batch = sample()
    raw = FakeSession("unused", "unused")
    session = transport._BoundSession(raw, lambda *args: None)
    session.bind(request.system, request.user, transport.classification.build_output_schema(batch))
    with pytest.raises(transport.warm.base.SubscriptionTransportError):
        session.rpc("thread/start", {"baseInstructions": "wrong"})
    session.rpc("thread/start", {"baseInstructions": request.system})
    with pytest.raises(transport.warm.base.SubscriptionTransportError):
        session.rpc("turn/start", {"threadId": "foreign", "input": [{"type": "text", "text": request.user}]})
    with pytest.raises(transport.warm.base.SubscriptionTransportError):
        session.rpc("turn/start", {"threadId": "thread-1", "input": [{"type": "text", "text": "wrong"}]})
    with pytest.raises(transport.warm.base.SubscriptionTransportError):
        session.rpc("turn/start", {"threadId": "thread-1", "input": [{"type": "text", "text": request.user}],
                                   "outputSchema": {}})
    session.rpc("turn/start", {"threadId": "thread-1", "input": [{"type": "text", "text": request.user}]})
    with pytest.raises(transport.warm.base.SubscriptionTransportError):
        session.rpc("turn/start", {"threadId": "thread-1", "input": [{"type": "text", "text": request.user}]})


@pytest.mark.parametrize("failure", ["rpc", "usage"])
def test_failed_dispatch_or_unknown_usage_stops_and_cleans(failure):
    FakeSession.instances.clear()
    FakeSession.fail_turn = failure == "rpc"
    FakeSession.unknown_usage = failure == "usage"
    try:
        request, batch = sample()
        item = client()
        with pytest.raises(transport.warm.ConcurrentStop):
            item.complete_grounding(request, batch)
        assert item.session is None and FakeSession.instances[0].closed
        assert item.budget.snapshot()["usage_complete"] is False
        assert item.requested_controls[-1]["output_schema_sent"] is True
        assert item.requested_controls[-1]["output_schema_acknowledged"] is (failure == "usage")
        assert len([method for method, _, _ in FakeSession.instances[0].calls
                    if method == "turn/start"]) == 1
        assert item._active_binding is None
    finally:
        FakeSession.fail_turn = False
        FakeSession.unknown_usage = False


def test_general_interfaces_are_disabled():
    item = client()
    with pytest.raises(ValueError, match="classification_only"):
        item.complete(object())
    with pytest.raises(ValueError, match="classification_only"):
        item.chat([])
    item.close()


def test_concurrent_call_rejected_before_binding_is_changed():
    request, batch = sample()
    item = client()
    sentinel = ("system", "user", {})
    item._active_binding = sentinel
    item._flight.acquire()
    try:
        with pytest.raises(transport.warm.ConcurrentStop, match="concurrent_completion_rejected"):
            item.complete_grounding(request, batch)
        assert item._active_binding is sentinel
        assert item.session is None
    finally:
        item._flight.release()
    item._active_binding = None
    item.close()
