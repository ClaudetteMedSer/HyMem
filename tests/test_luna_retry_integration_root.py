"""Root-owned ordinary/staged retry, capture and accounting controls."""
from collections import deque
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import time
import sys

import pytest

from benchmarks import codex_subscription_staged_v3 as staged
from tools.diagnostics import luna_lme_diagnostic_v5 as runner


def test_integration_preserves_limits_and_frozen_candidate():
    from tools.diagnostics import luna_lme_diagnostic_v4 as old
    from tools.diagnostics import luna_lme_diagnostic_launch_v4 as old_launch
    from tools.diagnostics import luna_lme_diagnostic_launch_v5 as launch
    assert runner.MAX_LIMITS == old.MAX_LIMITS
    assert runner.CANDIDATE_PINS == old.CANDIDATE_PINS
    assert runner.DATASET_SHA256 == old.DATASET_SHA256
    assert runner.ACCEPTED_MAP_SHA256 == old.ACCEPTED_MAP_SHA256
    assert runner.DIAGNOSTIC_HELPER_SHA256 == old.DIAGNOSTIC_HELPER_SHA256
    root = Path("/invented")
    receipt = {"unit": "invented.service"}
    old_command = old_launch.command(root, receipt, "a" * 64)
    command = launch.command(root, receipt, "a" * 64)
    assert command == [part.replace("luna_lme_diagnostic_v4.py", "luna_lme_diagnostic_v5.py")
                       for part in old_command]


def test_actual_bundle_passes_remote_archive_checks_before_side_effects(monkeypatch):
    from tools.diagnostics import luna_lme_diagnostic_host_preflight_v5 as preflight
    source = Path("/private/tmp/hymem-lme-retry-offline-assembly-v1")
    if not source.is_dir():
        pytest.skip("Root's source-only verification assembly is not present on this host")
    payload = preflight.archive_bytes(source)
    nodes = []
    for node in ast.parse(preflight.REMOTE).body:
        statement = ast.get_source_segment(preflight.REMOTE, node) or ""
        if statement.startswith("need(os.getuid()"):
            continue
        if statement.startswith("need(regular(DATASET)"):
            break
        nodes.append(node)
    else:
        pytest.fail("No remote side-effect boundary found")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<root-archive-test>", "exec"), namespace)
    assert len(namespace["manifest"]) == 527
    assert sum(name.startswith("code/") for name in namespace["manifest"]) == 12
    assert sum(name.startswith("candidate/") for name in namespace["manifest"]) == 514
    assert set(namespace["content"]) == set(namespace["manifest"])


@pytest.mark.parametrize("route", ["ordinary", "staged"])
@pytest.mark.parametrize("outcome", ["success", "terminal", "missing_usage"])
def test_actual_runner_routes_retry_and_capture_without_second_turn(monkeypatch, tmp_path, route, outcome):
    warm = staged.warm
    thread, turn = "invented-thread", "invented-turn"
    private = "INVENTED_UPSTREAM_PRIVATE_REASON"
    def event(method, **kwargs):
        return {"method": method, "params": {"threadId": thread, "turnId": turn, **kwargs}}
    retry = event("error", willRetry=True, error={"message": private,
        "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 403}}})
    events = [retry]
    if outcome == "terminal":
        events.append(event("error", willRetry=False, error={"message": private,
            "codexErrorInfo": "unauthorized"}))
    else:
        events.extend([
            event("item/started", item={"id": "answer", "type": "agentMessage"}),
            event("item/completed", item={"id": "answer", "type": "agentMessage",
                  "phase": "final_answer", "text": "{}"}),
        ])
        if outcome == "success":
            events.append(event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 42}}))
        events.append(event("turn/completed", turn={"id": turn, "status": "completed"}))
    limits = warm.BudgetLimits(12, 160000, 600)
    budget = warm.SharedBudget(limits)
    starts, closes = [], []
    directory = tmp_path.resolve() / "private"
    directory.mkdir(mode=0o700)

    def factory(_binary, _cwd, timeout=120):
        session = object.__new__(warm.WarmSession)
        session.created_at = time.monotonic()
        session.pending = deque(events)
        session.starting_events = []
        session.retired_threads = set()
        session.active_thread = session.active_turn = None
        session.stage = "startup"
        session.last_event = session.rpc_error = session.turn_observation = None
        session._observation_thread = session._observation_turn = None
        session.set_deadline = lambda _deadline: None
        session.unsubscribe = lambda _thread: None
        session.reset_private_errors()
        session.close = lambda: closes.append(True)
        return session

    def rpc(session, method, params, *, preserve_notifications=False):
        session.stage = method
        if method == "thread/start":
            session.active_thread = thread
            return {"thread": {"id": thread}}
        assert method == "turn/start"
        starts.append(params)
        return {"turn": {"id": turn, "status": "inProgress"}}

    def next_event(session):
        session.stage = "turn/events"
        return session.pending.popleft()

    def inspect(session, *, base_instructions):
        session.rpc("thread/start", {"baseInstructions": base_instructions})
        return {"auth": "chatgpt", "model": warm.base.MODEL,
            "config_isolation_admitted": True, "inference_enabled": False,
            "quota_windows": [{"remaining_percent": 75}], "_thread_id": thread}

    monkeypatch.setattr(warm.v5.v4.v3.WarmSession, "rpc", rpc)
    monkeypatch.setattr(warm.v5.v4.v3.WarmSession, "next_event", next_event)
    monkeypatch.setattr(warm.base, "inspect_preflight", inspect)
    ordinary_type, staged_type = warm.WarmSubscriptionClient, staged.StagedSubscriptionClient
    monkeypatch.setattr(warm, "WarmSubscriptionClient",
        lambda *args, **kwargs: ordinary_type(*args, session_factory=factory, **kwargs))
    monkeypatch.setattr(staged, "StagedSubscriptionClient",
        lambda *args, **kwargs: staged_type(*args, session_factory=factory, **kwargs))
    dual = runner.make_dual({"warm": warm, "staged": staged, "binary": Path("/unused")},
        budget, "q", limits, directory)
    assert dual.ordinary.private_failure_sink is dual.staged.private_failure_sink

    def complete():
        if route == "ordinary":
            return dual.complete(SimpleNamespace(system="system", user="invented probe",
                temperature=0, max_tokens=32, response_format="json"))
        source = staged.classification.GroundingSource(7, "Mira uses CairnDB.",
            source_role="user", source_peer_id="invented", source_created_at="2026-09-30")
        triple = staged.classification.Triple("Mira", "uses", "CairnDB", 1, source_message_id=7)
        request, batch = staged.staged.build_original_request((triple,), (source,))
        return dual.complete_stage(request, batch, "original", False)

    try:
        if outcome == "success":
            assert complete() == "{}"
        else:
            with pytest.raises(warm.ConcurrentStop):
                complete()
        assert len(starts) == 1
        assert ("outputSchema" in starts[0]) is (route == "staged")
        snapshot = budget.snapshot()
        assert snapshot["turns"] == 1
        assert snapshot["known_tokens"] == (42 if outcome == "success" else 0)
        assert snapshot["usage_complete"] is (outcome == "success")
        assert private not in json.dumps(snapshot)
        files = list(directory.glob("warm-private-failure-*.json"))
        assert len(files) == (0 if outcome == "success" else 1)
        if files:
            capture = json.loads(files[0].read_text())
            assert capture["first"]["message"] == private
            assert capture["first"]["http_status_code"] == 403
            assert "invented-turn" not in files[0].read_text()
            assert files[0].stat().st_mode & 0o777 == 0o600
    finally:
        dual.close()
    assert closes == [True]
