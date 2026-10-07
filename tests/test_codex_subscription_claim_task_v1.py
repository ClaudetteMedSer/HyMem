"""Offline fake-protocol checks for the inactive two-arm transport."""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import codex_subscription_claim_task_v1 as transport


def sample(arm="A", object_="CairnDB"):
    c = transport.adapter.classification
    source = c.GroundingSource(7, f"Mira uses {object_}.", source_role="user",
                               source_peer_id="mira", source_created_at="2026-09-29")
    triple = c.Triple("Mira", "uses", object_, 1, source_message_id=7)
    return transport.contract.build_arm_request(arm, (triple,), (source,))


def events(number, text="raw"):
    thread, turn = f"thread-{number}", f"turn-{number}"
    return deque([
        {"method": "turn/started", "params": {"threadId": thread, "turn": {"id": turn}}},
        {"method": "thread/tokenUsage/updated", "params": {"threadId": thread,
            "turnId": turn, "tokenUsage": {"total": {"totalTokens": 7}}}},
        {"method": "item/started", "params": {"threadId": thread, "turnId": turn,
            "item": {"id": "item-1", "type": "agentMessage"}}},
        {"method": "item/completed", "params": {"threadId": thread, "turnId": turn,
            "item": {"id": "item-1", "type": "agentMessage", "phase": "final_answer", "text": text}}},
        {"method": "turn/completed", "params": {"threadId": thread, "turn": {
            "id": turn, "status": "completed"}}},
    ])


class FakeSession:
    instances = []
    fail_turn = False
    fail_close = False
    missing_usage = False
    rate_used_percent = 25

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
        self.next_number = 0
        self.__class__.instances.append(self)

    def set_deadline(self, deadline):
        self.deadline = deadline

    def rpc(self, method, params, *, preserve_notifications=False):
        self.stage = method
        self.calls.append((method, copy.deepcopy(params)))
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
            self.next_number += 1
            return {"thread": {"id": f"thread-{self.next_number}", "ephemeral": True,
                "environments": [], "turns": [], "path": None},
                "model": transport.warm.base.MODEL, "modelProvider": "openai",
                "runtimeWorkspaceRoots": [], "instructionSources": [],
                "approvalPolicy": "never", "reasoningEffort": "low",
                "sandbox": {"type": "readOnly", "networkAccess": False}}
        if method == "turn/start":
            if self.fail_turn:
                transport.warm.base._fail("rpc_failure:turn/start")
            self.pending = events(self.next_number)
            if self.missing_usage:
                self.pending = deque(e for e in self.pending
                                     if e["method"] != "thread/tokenUsage/updated")
            return {"turn": {"id": f"turn-{self.next_number}", "status": "inProgress"}}
        if method == "thread/unsubscribe":
            return {"status": "unsubscribed"}
        raise AssertionError(method)

    def send(self, method, params, *, notification=False):
        self.calls.append((method, copy.deepcopy(params)))

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
    FakeSession.missing_usage = False
    FakeSession.rate_used_percent = 25
    yield
    FakeSession.fail_close = False


def client(max_requests=16):
    limits = transport.warm.BudgetLimits(20, 1000, 1000)
    budget = transport.warm.SharedBudget(limits)
    return transport.ClaimTaskSubscriptionClient("unused", budget, "q", limits,
        session_factory=FakeSession, max_requests=max_requests)


def test_multiarm_warm_reuse_and_exact_turn_bindings():
    item = client()
    try:
        for arm in ("A", "B", "A"):
            req, batch = sample(arm)
            assert item.complete_arm(arm, req, batch) == "raw"
        assert len(FakeSession.instances) == 1
        raw = FakeSession.instances[0]
        turns = [p for m, p in raw.calls if m == "turn/start"]
        starts = [p for m, p in raw.calls if m == "thread/start"]
        assert len(turns) == len(starts) == 3
        for index, arm in enumerate(("A", "B", "A")):
            req, batch = sample(arm)
            assert starts[index]["baseInstructions"] == req.system
            assert turns[index]["input"] == [{"type": "text", "text": req.user}]
            assert turns[index]["outputSchema"] == transport.contract.build_arm_output_schema(arm, batch)
        assert turns[0]["outputSchema"] == turns[2]["outputSchema"]
        assert turns[0]["outputSchema"] != turns[1]["outputSchema"]
        assert item.observed_turns == 3 and item.observed_tokens == 21
        assert all(x["output_schema_sent"] and x["output_schema_acknowledged"]
                   for x in item.requested_controls)
    finally:
        item.close()
    assert raw.closed


@pytest.mark.parametrize("arm,change", [
    ("C", {}), ("B", {"system": "foreign"}), ("A", {"user": "foreign"}),
    ("B", {"temperature": 0.5}), ("B", {"max_tokens": 1}),
])
def test_bad_arm_or_request_rejected_before_admission(arm, change):
    item = client()
    req, batch = sample("B" if arm == "B" else "A")
    with pytest.raises(transport.adapter.classification.GroundingContractError):
        item.complete_arm(arm, replace(req, **change), batch)
    assert not FakeSession.instances and not item.requested_controls
    item.close()


def test_cross_arm_and_batch_tamper_rejected_without_turn():
    item = client()
    req, batch = sample("A")
    with pytest.raises(transport.adapter.classification.GroundingContractError):
        item.complete_arm("B", req, batch)
    foreign_batch = sample("A", "OtherDB")[1]
    with pytest.raises(transport.adapter.classification.GroundingContractError):
        item.complete_arm("A", req, foreign_batch)
    assert not FakeSession.instances
    item.close()


def test_public_bypass_methods_rejected():
    item = client()
    req, batch = sample()
    for call in (lambda: item.complete(req), lambda: item.chat([]),
                 lambda: item.complete_grounding(req, batch)):
        with pytest.raises(ValueError):
            call()
    assert not FakeSession.instances
    item.close()


def test_failed_turn_has_unknown_usage_and_cleanup():
    FakeSession.fail_turn = True
    item = client()
    with pytest.raises(transport.warm.ConcurrentStop):
        item.complete_arm("A", *sample("A"))
    assert item.observed_turns == 1
    assert not item.usage_complete
    assert FakeSession.instances[0].closed
    assert item.requested_controls[-1]["output_schema_sent"]
    assert not item.requested_controls[-1]["output_schema_acknowledged"]
    item.close()


def test_missing_usage_stops_and_cleanup():
    FakeSession.missing_usage = True
    item = client()
    with pytest.raises(transport.warm.ConcurrentStop):
        item.complete_arm("B", *sample("B"))
    assert item.observed_turns == 1 and not item.usage_complete
    assert FakeSession.instances[0].closed
    item.close()


def test_quota_floor_blocks_dispatch_and_closes():
    FakeSession.rate_used_percent = 76
    item = client()
    with pytest.raises(transport.warm.ConcurrentStop):
        item.complete_arm("A", *sample("A"))
    assert item.observed_turns == 0
    assert not any(m == "turn/start" for m, _ in FakeSession.instances[0].calls)
    assert FakeSession.instances[0].closed
    item.close()


def test_rotation_between_arms_closes_previous_process():
    item = client(max_requests=1)
    try:
        item.complete_arm("A", *sample("A"))
        item.complete_arm("B", *sample("B"))
        assert len(FakeSession.instances) == 2
        assert FakeSession.instances[0].closed
        assert item.rotations == 1
        assert item.observed_turns == 2 and item.observed_tokens == 14
    finally:
        item.close()
    assert FakeSession.instances[1].closed


def test_cleanup_failure_stops_campaign():
    item = client()
    item.complete_arm("A", *sample("A"))
    FakeSession.fail_close = True
    with pytest.raises(transport.warm.ConcurrentStop, match="cleanup_failure"):
        item.close()
    assert item.budget.snapshot()["stopped"]
    FakeSession.fail_close = False
    item.close()


def test_concurrent_call_cannot_replace_binding():
    item = client()
    sentinel = ("system", "user", {})
    item._active_binding = sentinel
    item._flight.acquire()
    try:
        with pytest.raises(transport.warm.ConcurrentStop, match="concurrent_completion_rejected"):
            item.complete_arm("B", *sample("B"))
        assert item._active_binding is sentinel
        assert not FakeSession.instances
    finally:
        item._flight.release()
    item._active_binding = None
    item.close()


@pytest.mark.parametrize("name,marker", [
    ("codex_subscription_classification_v3.py", "pinned_classification_adapter_source_mismatch"),
    ("luna_claim_task_contract_v1.py", "pinned_claim_task_contract_source_mismatch"),
])
def test_source_drift_rejected_before_execution(name, marker):
    repo = Path(__file__).resolve().parents[1]
    code = (
        "import pathlib,runpy,sys;"
        f"sys.path.insert(0,{str(repo)!r});"
        "original=pathlib.Path.read_bytes;"
        f"pathlib.Path.read_bytes=lambda self: original(self)+b' ' if self.name=={name!r} else original(self);"
        f"runpy.run_path({str(repo / 'benchmarks/codex_subscription_claim_task_v1.py')!r})"
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0 and marker in result.stderr
