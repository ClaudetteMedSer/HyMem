"""Root-owned actual framed-session controls for observed structured requests."""
import io
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_staged_v6 as staged


def wire(monkeypatch, *, fail_at=None):
    observed = staged.observer
    base, warm = observed.base, staged.warm
    processes, calls = [], []
    monkeypatch.setattr(base.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="codex-cli 0.158.0"))
    def process(*args, **kwargs):
        child = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda: None)
        processes.append(child)
        return child
    monkeypatch.setattr(base.subprocess, "Popen", process)
    monkeypatch.setattr(warm.WarmSession, "_read", lambda self: None)
    monkeypatch.setattr(observed.TimeoutSession, "close", lambda self: setattr(self, "closed", True))
    def inspect(session, *, base_instructions):
        calls.append(True)
        thread, turn = "private-thread-" + str(len(calls)), "private-turn"
        session.root_timeout = len(calls) == fail_at
        offset = session.next_id
        def notice(method, **fields):
            return {"method": method, "params": {"threadId": thread, "turnId": turn, **fields}}
        events = [
            {"id": offset+1, "result": {"thread": {"id": thread}}},
            {"id": offset+2, "result": {"turn": {"id": turn, "status": "inProgress"}}},
            notice("item/started", item={"id": "private-item", "type": "agentMessage"}),
            notice("item/completed", item={"id": "private-item", "type": "agentMessage", "phase": "final_answer", "text": "{}"}),
            notice("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 27}}),
            notice("turn/completed", turn={"id": turn, "status": "completed"}),
            {"id": offset+3, "result": {"status": "unsubscribed"}},
        ]
        for value in events:
            session.events.put(value)
        session.rpc("thread/start", {"baseInstructions": base_instructions})
        session.active_thread = thread
        return {"auth": "chatgpt", "model": "gpt-6-luna", "inference_enabled": False,
            "config_isolation_admitted": True, "_thread_id": thread,
            "quota_windows": base.quota_metadata({"rateLimits": {"planType": "pro",
                "credits": {"hasCredits": True, "unlimited": False, "balance": "2.0"},
                "primary": {"usedPercent": 99, "windowDurationMins": 300, "resetsAt": 1790784000}}})}
    monkeypatch.setattr(base, "inspect_preflight", inspect)
    original = observed.TimeoutSession.next_event
    def next_event(session):
        if session.root_timeout:
            base._fail("timeout")
        return original(session)
    monkeypatch.setattr(observed.TimeoutSession, "next_event", next_event)
    cap = warm.BudgetLimits(40, 100000, 600)
    budget = warm.SharedBudget(cap)
    client = staged.StagedSubscriptionClient("unused", budget, "q", cap,
        max_requests=16, max_age_seconds=300)
    source = staged.classification.GroundingSource(7, "Mira uses CairnDB.",
        source_role="user", source_peer_id="invented", source_created_at="2026-09-30")
    triple = staged.classification.Triple("Mira", "uses", "CairnDB", 1, source_message_id=7)
    request, batch = staged.staged.build_original_request((triple,), (source,))
    return client, budget, processes, calls, request, batch


def test_observed_schema_binding_survives_eighteen_calls_and_rotation(monkeypatch):
    client, budget, processes, calls, request, batch = wire(monkeypatch)
    try:
        for _ in range(18):
            assert client.complete_stage(request, batch, "original", False) == "{}"
            assert type(client.session) is staged._BoundSession
            assert type(client.session._raw) is staged.observer.TimeoutSession
            assert client.session._binding is None
            record = client.diagnostic_summary()["last_record"]
            assert record["observed"]["completed_seen"] is True
            assert record["precleanup"] is None
            assert record["deadline_monotonic"] > record["invocation_start_monotonic"]
        schema = staged.staged.build_original_output_schema(batch)
        messages = [json.loads(line) for child in processes for line in child.stdin.getvalue().splitlines()]
        starts = [msg for msg in messages if msg.get("method") == "turn/start"]
        threads = [msg for msg in messages if msg.get("method") == "thread/start"]
        assert len(starts) == len(threads) == len(calls) == 18
        assert len(processes) == 2 and client.rotations == 1
        assert all(msg["params"]["outputSchema"] == schema for msg in starts)
        assert all(msg["params"]["input"] == [{"type": "text", "text": request.user}] for msg in starts)
        assert all(msg["params"]["baseInstructions"] == request.system for msg in threads)
        assert client.diagnostic_summary()["calls"] == 18
        assert len(client.diagnostic_records()) == 16
        assert budget.snapshot()["turns"] == 18 and budget.snapshot()["known_tokens"] == 18*27
        assert "private-thread" not in json.dumps(client.diagnostic_summary())
        assert staged.warm is staged.observer.warm
        assert staged.observer.SharedBudget is staged.warm.SharedBudget
    finally:
        client.close()
    assert client.session is None and client.directory is None


def test_structured_timeout_seventeen_keeps_current_invocation_evidence(monkeypatch):
    client, budget, _, calls, request, batch = wire(monkeypatch, fail_at=17)
    try:
        for _ in range(16):
            client.complete_stage(request, batch, "original", False)
        with pytest.raises(staged.warm.ConcurrentStop, match="timeout"):
            client.complete_stage(request, batch, "original", False)
        summary = client.diagnostic_summary()
        assert summary["first_failure"]["call_index"] == 17
        record = summary["first_failure"]["record"]
        assert record["failure_code"] == "timeout" and record["known_usage"] is False
        assert record["precleanup"] is not None and record["events_returned"] == 0
        assert record["deadline_monotonic"] > record["invocation_start_monotonic"]
        assert len(calls) == 17 and budget.snapshot()["usage_complete"] is False
        assert budget.snapshot()["known_tokens"] == 16*27
        assert client._active_binding is None and client.session is None
    finally:
        client.close()


def test_original_schema_binding_and_stage_validation_are_unchanged():
    source = Path(staged.__file__)
    def nodes(path):
        tree = ast.parse(path.read_text())
        wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_BoundSession")
        client = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StagedSubscriptionClient")
        stage = next(node for node in client.body if isinstance(node, ast.FunctionDef) and node.name == "complete_stage")
        return ast.dump(wrapper, include_attributes=False), ast.dump(stage, include_attributes=False)
    assert nodes(source) == nodes(source.with_name("codex_subscription_staged_v5.py"))
