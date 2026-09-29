"""Fail-closed checks for the source-free four-question observer."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


PATH = Path(__file__).resolve().parents[1] / "luna_subscription_profiled_progress.py"
SPEC = importlib.util.spec_from_file_location("profiled_progress_under_test", PATH)
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def _slot(turns=1, tokens=100):
    return {"attempted_calls": turns, "admitted_turns": turns,
        "known_tokens": tokens, "invocation_wall_seconds": 1.25,
        "returned_calls": turns, "rejected_calls": 0, "usage_complete": True}


def _warm():
    return {"processes_started": 1, "rotations": 0, "cold_calls": 1, "warm_calls": 0,
        "startup_seconds": 0.2, "unsubscribe_seconds": 0.1,
        "rotation_cleanup_seconds": 0.0, "final_cleanup_seconds": 0.1}


def _fixture(tmp_path, monkeypatch, scores=(True, False, True, False)):
    root = tmp_path / "run-root"
    output = root / "run"
    output.mkdir(parents=True)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    protocol_path = candidate / "benchmarks" / "lme_protocol.py"
    protocol_path.parent.mkdir()
    protocol_path.write_text("# synthetic pinned protocol path\n")
    protocol = types.ModuleType("benchmarks.lme_protocol")
    protocol.__file__ = str(protocol_path)
    protocol._validate_versioned_indexing = lambda *args, **kwargs: True
    benchmarks = types.ModuleType("benchmarks")
    benchmarks.__path__ = [str(protocol_path.parent)]
    benchmarks.lme_protocol = protocol
    monkeypatch.setitem(sys.modules, "benchmarks", benchmarks)
    monkeypatch.setitem(sys.modules, "benchmarks.lme_protocol", protocol)
    monkeypatch.setattr(observer, "bounded_json", lambda path: source[str(path)])
    source = {}
    entries = []
    stages = {"canary": {"reader": _slot()}}
    budget_questions = {"canary": {"turns": 1, "known_tokens": 100}}
    for index, score in enumerate(scores):
        house = {"open_runs_before": 0, "terminalized": 0,
            "open_runs_after": 0, "active_leases_after": 0}
        entry = {"index": index, "question_started": True, "question_completed": True,
            "cleanup_ok": True, "stop_code": None, "correct": score,
            "dream_run_housekeeping": house, "transport": _warm()}
        qdir = output / f"q-{index:04d}"
        qdir.mkdir()
        row = {"correct": score, "benchmark_failure": None, "judge_error": False,
            "judge_parse_valid": True,
            "indexing": {"outcome": "success", "healthy": True, "summary_healthy": True}}
        source[str(qdir / "private-result.json")] = entry
        source[str(qdir / "private-row.json")] = row
        entries.append(entry)
        stages[f"q-{index:04d}"] = {"reader": _slot()}
        budget_questions[f"q-{index:04d}"] = {"turns": 1, "known_tokens": 100}
    budget = {"questions": budget_questions, "turns": 5, "known_tokens": 500,
        "usage_complete": True, "in_flight": 0, "reserved": 0,
        "known_tokens_scope": "completed_turns_only_failed_turn_usage_unknown"}
    result = {"schema": "luna-subscription-lme-profiled-v1", "questions": entries,
        "budget": budget, "canary": {"passed": True}, "canary_transport": _warm(),
        "campaign_stop": None, "usage_complete_now": True, "active_invocations": 0,
        "stage_accounting": stages, "stage_accounting_reconciled": True,
        "stage_accounting_schema": "source-free-stages-v1",
        "stage_collector_sha256": observer.COLLECTOR}
    safe = {"schema": "luna-subscription-lme-profiled-v1", "ok": True,
        "stop_code": None, "questions_selected": 4, "questions_completed": 4,
        "source_files_verified": 508, "dataset_sha256": observer.DATASET,
        "candidate_source_map_sha256": observer.SOURCE_MAP,
        "pilot_helper_sha256": observer.PILOT, "runner_sha256": observer.PROFILED,
        "base_transport_sha256": observer.BASE,
        "concurrent_transport_sha256": observer.CONCURRENT,
        "warm_transport_sha256": observer.WARM, "model": "gpt-6-luna",
        "transport_kind": "warm_process_fresh_ephemeral_thread",
        "warm_max_requests": 16, "warm_max_age_seconds": 300,
        "canary_passed": True, "usage_complete": True, "in_flight": 0,
        "active_invocations": 0, "failed_count": 0, "not_started_count": 0,
        "turns": 5, "known_tokens": 500,
        "known_tokens_scope": budget["known_tokens_scope"],
        "correct_count": sum(scores), "incorrect_count": 4 - sum(scores)}
    return root, {"candidate": str(candidate)}, safe, result, source


def test_valid_terminal_can_score_incorrect_answers(tmp_path, monkeypatch):
    root, receipt, safe, result, _ = _fixture(tmp_path, monkeypatch)
    verdict = observer.verify_terminal(root, receipt, safe, result)
    assert verdict["validated"] is True
    assert [q["correct"] for q in verdict["questions"]] == [True, False, True, False]
    assert verdict["stage_accounting_reconciled"] is True


def test_terminal_rejects_protocol_score_and_cleanup_tamper(tmp_path, monkeypatch):
    root, receipt, safe, result, source = _fixture(tmp_path, monkeypatch)
    qdir = root / "run" / "q-0001"
    source[str(qdir / "private-row.json")]["correct"] = True
    assert observer.verify_terminal(root, receipt, safe, result)["validated"] is False
    source[str(qdir / "private-row.json")]["correct"] = False
    result["questions"][1]["dream_run_housekeeping"]["active_leases_after"] = 1
    assert observer.verify_terminal(root, receipt, safe, result)["validated"] is False


def test_stage_reconciliation_and_usage_fail_closed(tmp_path, monkeypatch):
    root, receipt, safe, result, _ = _fixture(tmp_path, monkeypatch)
    result["stage_accounting"]["q-0000"]["reader"]["known_tokens"] += 1
    assert observer.verify_terminal(root, receipt, safe, result)["validated"] is False
    result["stage_accounting"]["q-0000"]["reader"]["known_tokens"] -= 1
    result["stage_accounting"]["q-0000"]["reader"]["usage_complete"] = False
    assert observer.verify_terminal(root, receipt, safe, result)["validated"] is False
    result["stage_accounting"]["q-0000"]["reader"]["usage_complete"] = True
    result["stage_accounting"]["q-0003"]["reader"]["usage_complete"] = "yes"
    assert observer.verify_terminal(root, receipt, safe, result)["validated"] is False
    result["stage_accounting"]["q-0003"]["reader"]["usage_complete"] = True
    result["stage_accounting"]["q-0004"] = {"reader": _slot()}
    assert observer.verify_terminal(root, receipt, safe, result)["validated"] is False


def test_stage_and_warm_summaries_never_copy_text():
    stages = {"canary": {"reader": dict(_slot(), prompt="secret text")}}
    summary = observer.stage_summary(stages)
    assert summary is not None
    assert "prompt" not in summary["canary"]["reader"]
    assert observer.stage_summary({"canary": {"raw prompt": _slot()}}) is None
    assert observer.stage_summary({"canary": {"reader": dict(_slot(), invocation_wall_seconds=float("nan"))}}) is None
    assert observer.warm_summary(dict(_warm(), response="secret text")) == _warm()


def test_missing_progress_is_unknown_and_foreign_cgroup_is_not_clean():
    assert observer.progress_summary(None)["usage_complete"] is None
    unit = {"available": True, "active_state": "active", "sub_state": "exited",
        "result": "success", "main_pid": 0, "restarts": 0, "exit_status": 0,
        "exit_code_kind": 1, "policy_ok": True, "cgroup_header_matches": False,
        "cgroup_header_empty": False, "expected_cgroup_processes": 0}
    assert observer.completed_and_clean({"validated": True}, unit) is False
    unit["cgroup_header_matches"] = True
    assert observer.completed_and_clean({"validated": True}, unit) is True
    unit["policy_ok"] = False
    assert observer.completed_and_clean({"validated": True}, unit) is False


def test_prepared_unit_suffix_and_source_pin_failure(tmp_path):
    root_name = ".hymem-luna-lme-profiled-4_yhy8b9"
    unit = "hymem-luna-lme-profiled-4_yhy8b9.service"
    assert observer.UNIT.fullmatch(unit)
    assert not observer.UNIT.fullmatch("hymem-luna-lme-multi-4_yhy8b9.service")
    root = tmp_path / root_name
    root.mkdir()
    source = root / "luna_subscription_lme_profiled.py"
    source.write_bytes(b"tampered")
    receipt = {"source_sha256": {source.name: observer.PROFILED},
        "dataset": str(tmp_path / "dataset.json")}
    assert observer.verify_live_pins(root, receipt) == {
        "source_pins_verified": False, "dataset_pin_verified": False,
        "inventory_verified": False}


def test_safe_terminal_failure_is_categorical():
    assert observer.safe_terminal_summary(None)["available"] is False
    assert observer.safe_terminal_summary({"ok": False, "stop_code": "usage_incomplete"})[
        "failure_category"] == "usage_incomplete"
    arbitrary = "private prompt or provider text"
    summary = observer.safe_terminal_summary({"ok": False, "stop_code": arbitrary})
    assert summary["failure_category"] == "other"
    assert arbitrary not in str(summary)
