"""Offline fake-wire checks for the inactive staged subscription adapter."""
from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import replace

import pytest

from benchmarks import codex_subscription_staged_v1 as transport


def sample():
    c = transport.classification
    source = c.GroundingSource(7, "Mira uses CairnDB.", source_role="user",
                               source_peer_id="mira", source_created_at="2026-09-29")
    triple = c.Triple("Mira", "uses", "CairnDB", 1, source_message_id=7)
    return transport.staged.build_original_request((triple,), (source,))


def alternatives(batch):
    original = json.dumps({"schema": transport.staged.ORIGINAL_SCHEMA,
                           "batch_sha256": batch.batch_sha256, "complete": True,
                           "originals": [{"index": 0, "original": {
                               "state": "not_established", "support": None}}]})
    return transport.staged.build_alternatives_request(batch, original)


def events(text):
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
    rate_used_percent = 25
    response_text = "raw-stage"

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
            return {"rateLimits": {"primary": {"usedPercent": self.rate_used_percent}}}
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


@pytest.fixture(autouse=True)
def reset_fake():
    FakeSession.instances.clear()
    FakeSession.fail_turn = False
    FakeSession.fail_close = False
    FakeSession.unknown_usage = False
    FakeSession.rate_used_percent = 25
    FakeSession.response_text = "raw-stage"
    yield


def client(*, max_requests=16, turns=20):
    limits = transport.warm.BudgetLimits(turns, 1000, 1000)
    budget = transport.warm.SharedBudget(limits)
    return transport.StagedSubscriptionClient("unused", budget, "q", limits,
        session_factory=FakeSession, max_requests=max_requests)


def wires(raw):
    return [params for method, params, _ in raw.calls if method == "turn/start"]


def test_original_alternatives_recheck_use_separate_exact_schemas_and_one_budget():
    request, batch = sample()
    alt_request, alt_batch = alternatives(batch)
    item = client()
    try:
        assert item.complete_stage(request, batch, "original", False) == "raw-stage"
        assert item.complete_stage(alt_request, alt_batch, "alternatives", False) == "raw-stage"
        assert item.complete_stage(request, batch, "original", True) == "raw-stage"
        assert len(FakeSession.instances) == 1
        turns = wires(FakeSession.instances[0])
        assert len(turns) == 3
        assert [p["outputSchema"] for p in turns] == [
            transport.staged.build_original_output_schema(batch),
            transport.staged.build_alternatives_output_schema(alt_batch),
            transport.staged.build_original_output_schema(batch)]
        assert [p["input"] for p in turns] == [
            [{"type": "text", "text": request.user}],
            [{"type": "text", "text": alt_request.user}],
            [{"type": "text", "text": request.user}]]
        assert item.observed_turns == 3 and item.observed_tokens == 21
        assert all(x["output_schema_sent"] and x["output_schema_acknowledged"]
                   for x in item.requested_controls)
    finally:
        item.close()
    assert FakeSession.instances[0].closed


@pytest.mark.parametrize("stage,recheck", [("unknown", False), ("original", 0),
                                          ("alternatives", True), (False, False)])
def test_invalid_stage_or_recheck_rejected_before_admission(stage, recheck):
    request, batch = sample()
    item = client()
    try:
        with pytest.raises(transport.staged.GroundingContractError):
            item.complete_stage(request, batch, stage, recheck)
        assert not FakeSession.instances and not item.requested_controls
    finally:
        item.close()


def test_request_and_batch_cross_binding_rejected_and_warm_session_closed():
    request, batch = sample()
    alt_request, alt_batch = alternatives(batch)
    item = client()
    try:
        item.complete_stage(request, batch, "original", False)
        prior = copy.deepcopy(item.requested_controls)
        with pytest.raises(transport.staged.GroundingContractError):
            item.complete_stage(alt_request, batch, "original", False)
        assert item.requested_controls == prior
        assert item.session is None and FakeSession.instances[0].closed
        assert item._active_binding is None
        with pytest.raises(transport.staged.GroundingContractError):
            item.complete_stage(request, alt_batch, "alternatives", False)
    finally:
        item.close()


def test_forged_alternative_prior_binding_and_old_v4_request_rejected():
    request, batch = sample()
    alt_request, alt_batch = alternatives(batch)
    item = client()
    try:
        with pytest.raises(transport.staged.GroundingContractError):
            item.complete_stage(alt_request, replace(alt_batch, original_response_sha256="0" * 64),
                                "alternatives", False)
        v4_request, v4_batch = transport.classification.build_grounding_request(batch.triples, batch.sources)
        with pytest.raises(transport.staged.GroundingContractError):
            item.complete_stage(v4_request, v4_batch, "original", False)
        assert not FakeSession.instances and not item.requested_controls
    finally:
        item.close()


def test_rotation_and_no_generic_fallback():
    request, batch = sample()
    item = client(max_requests=1)
    try:
        for method in (lambda: item.complete(request), lambda: item.chat([]),
                       lambda: item.complete_grounding(request, batch)):
            with pytest.raises(ValueError, match="staged_only"):
                method()
        item.complete_stage(request, batch, "original", False)
        item.complete_stage(request, batch, "original", False)
        assert len(FakeSession.instances) == 2 and FakeSession.instances[0].closed
        assert item.rotations == 1 and item.observed_turns == 2
    finally:
        item.close()
    assert all(s.closed for s in FakeSession.instances)


def test_stages_share_a_hard_three_turn_ceiling():
    request, batch = sample()
    alt_request, alt_batch = alternatives(batch)
    item = client(turns=3)
    try:
        item.complete_stage(request, batch, "original", False)
        item.complete_stage(alt_request, alt_batch, "alternatives", False)
        item.complete_stage(request, batch, "original", True)
        with pytest.raises(transport.warm.ConcurrentStop):
            item.complete_stage(request, batch, "original", False)
        assert item.observed_turns == 3
        assert len(wires(FakeSession.instances[0])) == 3
    finally:
        item.close()


def test_quota_floor_rejects_before_first_turn():
    request, batch = sample()
    FakeSession.rate_used_percent = 100
    item = client()
    try:
        with pytest.raises(transport.warm.ConcurrentStop):
            item.complete_stage(request, batch, "original", False)
        assert not any(wires(raw) for raw in FakeSession.instances)
    finally:
        item.close()


def test_single_flight_rejection_halts_shared_budget():
    request, batch = sample()
    item = client()
    try:
        assert item._flight.acquire(blocking=False)
        with pytest.raises(transport.warm.ConcurrentStop):
            item.complete_stage(request, batch, "original", False)
        assert item.budget.stop_code == "concurrent_completion_rejected"
        assert not FakeSession.instances
    finally:
        item._flight.release()
        item.close()


def test_prior_wire_schema_mutation_does_not_cross_bind_next_turn():
    request, batch = sample()
    alt_request, alt_batch = alternatives(batch)
    item = client()
    try:
        item.complete_stage(request, batch, "original", False)
        turns = wires(FakeSession.instances[0])
        turns[0]["outputSchema"]["properties"]["schema"]["enum"] = ["tampered"]
        item.complete_stage(alt_request, alt_batch, "alternatives", False)
        assert wires(FakeSession.instances[0])[1]["outputSchema"] == (
            transport.staged.build_alternatives_output_schema(alt_batch))
    finally:
        item.close()


def test_failed_turn_start_preserves_unknown_usage_and_clears_binding():
    request, batch = sample()
    item = client()
    FakeSession.fail_turn = True
    try:
        with pytest.raises(transport.warm.ConcurrentStop):
            item.complete_stage(request, batch, "original", False)
        assert item._active_binding is None and item.session is None
        assert FakeSession.instances[0].closed
        assert item.requested_controls[-1]["output_schema_sent"] is True
        assert item.requested_controls[-1]["output_schema_acknowledged"] is False
        assert item.usage_complete is False
    finally:
        item.close()
