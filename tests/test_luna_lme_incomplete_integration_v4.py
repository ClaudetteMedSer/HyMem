"""Source-bound, no-inference checks for the versioned turn observer integration."""
from __future__ import annotations

import hashlib
import ast
from collections import deque
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

from benchmarks import codex_subscription_staged_v2 as staged
from tools.diagnostics import luna_lme_diagnostic_bundle_v4 as bundle
from tools.diagnostics import luna_lme_diagnostic_host_preflight_v4 as preflight
from tools.diagnostics import luna_lme_diagnostic_progress_v5 as reader
from tools.diagnostics import luna_lme_diagnostic_v4 as runner


ROOT = Path(__file__).resolve().parents[1]


def test_runner_and_bundle_pin_actual_new_sources():
    assert runner.PINS["benchmarks/codex_subscription_warm_v4.py"] == (
        hashlib.sha256((ROOT / "benchmarks/codex_subscription_warm_v4.py").read_bytes()).hexdigest())
    assert runner.PINS["benchmarks/codex_subscription_staged_v2.py"] == (
        hashlib.sha256((ROOT / "benchmarks/codex_subscription_staged_v2.py").read_bytes()).hexdigest())
    assert bundle.RUNNER_SHA256 == hashlib.sha256(
        (ROOT / bundle.RUNNER_RELATIVE).read_bytes()).hexdigest()
    assert reader.PINS["tools/diagnostics/luna_lme_diagnostic_v4.py"] == bundle.RUNNER_SHA256
    assert staged.warm.WarmSubscriptionClient.__module__ == staged.warm.__name__
    assert staged.warm.v3 is not staged.warm


def test_embedded_remote_validates_actual_archive_before_host_side_effects(monkeypatch):
    source = Path("/private/tmp/hymem-lme-diagnostic-offline-assembly-v5")
    if not source.is_dir():
        pytest.skip("accepted offline source assembly is not present on this host")
    payload = preflight.archive_bytes(source)
    tree = ast.parse(preflight.REMOTE)
    selected = []
    for node in tree.body:
        statement = ast.get_source_segment(preflight.REMOTE, node) or ""
        if statement.startswith("need(os.getuid()"):
            continue
        if statement.startswith("need(regular(DATASET)"):
            break
        selected.append(node)
    else:
        pytest.fail("remote validation boundary not found")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
    namespace = {}
    prefix = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(prefix, "<remote-archive-validation>", "exec"), namespace)
    assert len(namespace["manifest"]) == 525
    assert sum(name.startswith("candidate/") for name in namespace["manifest"]) == 514
    assert sum(name.startswith("code/") for name in namespace["manifest"]) == 10
    assert set(namespace["content"]) == set(namespace["manifest"])


def test_make_dual_uses_new_observer_on_both_paths_and_rejects_unknown_stage():
    limits = staged.warm.BudgetLimits(12, 160_000, 600)
    budget = staged.warm.SharedBudget(limits)
    loaded = {"warm": staged.warm, "staged": staged, "binary": Path("/unused")}
    dual = runner.make_dual(loaded, budget, "q", limits)
    try:
        assert isinstance(dual.ordinary, staged.warm.WarmSubscriptionClient)
        assert isinstance(dual.staged, staged.StagedSubscriptionClient)
        assert isinstance(dual.staged, staged.warm.WarmSubscriptionClient)
        with pytest.raises(staged.staged.GroundingContractError):
            dual.complete_stage(None, None, "unknown", False)
        assert budget.snapshot()["turns"] == 0
    finally:
        dual.close()


def test_new_reader_rejects_old_receipt_before_interpreting_result(monkeypatch, tmp_path):
    root = tmp_path / ".hymem-lme-diagnostic-reader-v5"
    root.mkdir(mode=0o700)
    unit = "hymem-luna-lme-diagnostic-reader-v5.service"
    receipt = {
        "schema": "luna-lme-diagnostic-launch-v3", "root": str(root),
        "unit": unit,
        "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "source_sha256": reader.PINS,
        "candidate_map_sha256": reader.MAP_SHA256,
        "inventory_sha256": reader.INVENTORY_SHA256,
        "dataset_sha256": reader.DATASET_SHA256,
        "binary_sha256": reader.BINARY_SHA256,
        "selected_count": 4, "workers": 4, "indexing_seconds": 10_800,
        "output_dir": str(root / "run"), "limits": reader.LIMITS,
    }
    monkeypatch.setattr(reader, "_read", lambda *_args: receipt)
    monkeypatch.setattr(reader, "_sha", lambda *_args: "a" * 64)
    with pytest.raises(ValueError, match="receipt_identity_invalid"):
        reader._receipt(root, "a" * 64)


def _fault(code: str = "incomplete_turn_or_usage") -> dict:
    return {
        "code": code, "phase": "run", "rpc": "turn/events",
        "resource_observation": {"current": 4, "peak": 4, "limit": 256, "denials": 0},
        "process_index": 1, "request_index": 1, "retired_count": 0, "queue_count": 0,
        "turn_admitted": True, "known_usage": False, "usage_complete": False,
        "turn_observation": {
            "basis": "consumed_observed_shape", "events_consumed": 3,
            "completed_seen": True, "final_seen": True, "final_count": 1,
            "usage_update_count": 0, "usage_state": "absent",
            "last_event_family": "turn_completed",
        },
    }


def test_reader_accepts_only_valid_resource_override_of_incomplete_turn():
    fault = _fault("resource_task_denial")
    fault["underlying_code"] = "incomplete_turn_or_usage"
    fault["resource_observation"]["denials"] = 1
    assert reader._failure_projection({"first_failure": fault})["turn_observation"] == (
        fault["turn_observation"])
    fault["resource_observation"]["denials"] = 0
    with pytest.raises(ValueError, match="resource_denial_evidence_missing"):
        reader._failure_projection({"first_failure": fault})
    fault["resource_observation"]["denials"] = 1
    fault["underlying_code"] = "usage_invalid"
    with pytest.raises(ValueError, match="turn_observation_scope_invalid"):
        reader._failure_projection({"first_failure": fault})


def test_reader_rejects_missing_observation_and_fabricated_success():
    fault = _fault()
    fault.pop("turn_observation")
    with pytest.raises(ValueError, match="turn_observation_missing"):
        reader._failure_projection({"first_failure": fault})
    fault = _fault()
    fault["turn_observation"]["usage_state"] = "positive"
    fault["turn_observation"]["usage_update_count"] = 1
    fault["turn_observation"]["events_consumed"] = 4
    with pytest.raises(ValueError, match="turn_observation_success_contradiction"):
        reader._failure_projection({"first_failure": fault})


@pytest.mark.parametrize("route", ["ordinary", "staged"])
def test_actual_runner_dual_path_copies_incomplete_snapshot_before_cleanup(monkeypatch, route):
    warm = staged.warm
    thread, turn = "private-thread", "private-turn"
    events = deque([
        {"method": "item/started", "params": {"threadId": thread, "turnId": turn,
            "item": {"id": "message", "type": "agentMessage"}}},
        {"method": "item/completed", "params": {"threadId": thread, "turnId": turn,
            "item": {"id": "message", "type": "agentMessage",
                     "phase": "final_answer", "text": "private response"}}},
        {"method": "turn/completed", "params": {"threadId": thread,
            "turn": {"id": turn, "status": "completed"}}},
    ])
    limits = warm.BudgetLimits(12, 160_000, 600)
    budget = warm.SharedBudget(limits)
    closes = []

    def factory(_binary, _cwd, timeout=120):
        session = object.__new__(warm.WarmSession)
        session.created_at = time.monotonic()
        session.pending = events
        session.starting_events = []
        session.retired_threads = set()
        session.active_thread = None
        session.active_turn = None
        session.stage = "startup"
        session.last_event = None
        session.rpc_error = None
        session.turn_observation = None
        session._observation_thread = None
        session._observation_turn = None
        session.set_deadline = lambda _deadline: None
        session.unsubscribe = lambda _thread: None
        def close():
            first = budget.snapshot()["first_failure"]
            assert first["turn_observation"]["events_consumed"] == 3
            closes.append(True)
        session.close = close
        return session

    def rpc(session, method, params, *, preserve_notifications=False):
        session.stage = method
        if method == "thread/start":
            session.active_thread = thread
            return {"thread": {"id": thread}}
        if method == "turn/start":
            return {"turn": {"id": turn, "status": "inProgress"}}
        raise AssertionError(method)

    def next_event(session):
        session.stage = "turn/events"
        return session.pending.popleft()

    def inspect(session, *, base_instructions):
        session.rpc("thread/start", {"baseInstructions": base_instructions})
        return {"auth": "chatgpt", "model": warm.base.MODEL,
                "config_isolation_admitted": True, "inference_enabled": False,
                "quota_windows": [{"remaining_percent": 75}], "_thread_id": thread}

    monkeypatch.setattr(warm.v3.WarmSession, "rpc", rpc)
    monkeypatch.setattr(warm.v3.WarmSession, "next_event", next_event)
    monkeypatch.setattr(warm.base, "inspect_preflight", inspect)
    ordinary_type = warm.WarmSubscriptionClient
    staged_type = staged.StagedSubscriptionClient
    monkeypatch.setattr(warm, "WarmSubscriptionClient",
        lambda *args, **kwargs: ordinary_type(*args, session_factory=factory, **kwargs))
    monkeypatch.setattr(staged, "StagedSubscriptionClient",
        lambda *args, **kwargs: staged_type(*args, session_factory=factory, **kwargs))
    dual = runner.make_dual({"warm": warm, "staged": staged,
        "binary": Path("/unused")}, budget, "q", limits)
    try:
        with pytest.raises(warm.ConcurrentStop, match="incomplete_turn_or_usage"):
            if route == "ordinary":
                dual.complete(SimpleNamespace(system="system", user="private prompt",
                    temperature=0, max_tokens=32, response_format="json"))
            else:
                source = staged.classification.GroundingSource(
                    7, "private source", source_role="user",
                    source_peer_id="private", source_created_at="2026-09-29")
                triple = staged.classification.Triple(
                    "Mira", "uses", "CairnDB", 1, source_message_id=7)
                request, batch = staged.staged.build_original_request((triple,), (source,))
                dual.complete_stage(request, batch, "original", False)
        first = budget.snapshot()["first_failure"]
        assert first["turn_observation"]["usage_state"] == "absent"
        assert first["known_usage"] is False
        assert budget.snapshot()["usage_complete"] is False
        assert closes == [True]
    finally:
        dual.close()
