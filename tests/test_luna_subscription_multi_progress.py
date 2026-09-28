"""Offline safety tests for the private two-question progress observer."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

from tools.diagnostics import luna_subscription_multi_progress as monitor


def test_inflight_progress_is_unknown_and_no_raw_text_export():
    source = {"canary": {"passed": True, "private_response": "secret"},
              "budget": {"turns": 17, "known_tokens": 4_100_000, "in_flight": 1,
                         "known_tokens_scope": "completed_turns_only_failed_turn_usage_unknown"},
              "active_invocations": 1, "usage_complete_now": True,
              "questions": [{"question_started": True, "question_completed": True,
                             "correct": False, "answer": "secret"}, None]}
    report = monitor.progress_summary(source)
    assert report["known_tokens"] == 4_100_000
    assert report["usage_complete"] is False
    assert report["questions"][0]["correct"] is False
    assert report["questions"][1]["correct"] is None
    assert report["last_cycle_stale_count"] == 0
    assert "secret" not in json.dumps(report)
    assert monitor.progress_summary(None)["known_tokens"] is None


def test_bounded_json_and_file_stat_do_not_emit_contents(tmp_path):
    path = tmp_path / "private-progress.json"
    path.write_bytes(b"x" * (monitor.MAX_JSON + 1))
    assert monitor.bounded_json(path) is None
    path.write_text('{"a":1}')
    assert monitor.bounded_json(path) == {"a": 1}
    link = tmp_path / "link"
    link.symlink_to(path)
    assert monitor.bounded_json(link) is None
    assert set(monitor.file_meta(path)) == {"present", "bytes", "mtime_ns"}


def test_systemd_blank_header_still_checks_fixed_cgroup(monkeypatch):
    expected = "/user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-multi-x.service"
    def run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=(
            "ActiveState=active\nSubState=exited\nResult=success\nMainPID=0\n"
            "ControlGroup=\nNRestarts=0\nExecMainStatus=0\nExecMainCode=1\nRestart=no\n"
            "KillMode=control-group\nType=exec\nRemainAfterExit=yes\n"
            "RuntimeMaxUSec=4h 2min 10s\nTimeoutStopUSec=10s\nMemoryMax=4294967296\n"
            "CPUQuotaPerSecUSec=2s\nTasksMax=128\nOOMPolicy=kill\n"))
    monkeypatch.setattr(monitor.subprocess, "run", run)
    seen = []
    monkeypatch.setattr(monitor, "count_processes", lambda path: seen.append(path) or 0)
    state = monitor.systemd_state("hymem-luna-lme-multi-x.service", expected)
    assert seen == [expected]
    assert state["cgroup_header_empty"] is True
    assert state["cgroup_header_matches"] is False
    assert state["expected_cgroup_processes"] == 0
    assert state["policy_ok"] is True
    assert state["exit_code_kind"] == 1
    assert monitor.completed_and_clean({"validated": True}, state) is True
    wrong = dict(state, cgroup_header_empty=False, cgroup_header_matches=False)
    assert monitor.completed_and_clean({"validated": True}, wrong) is False
    wrong_code = dict(state, exit_code_kind=2)
    assert monitor.completed_and_clean({"validated": True}, wrong_code) is False


def test_systemd_resource_policy_mismatch_rejected(monkeypatch):
    def run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=(
            "ActiveState=active\nSubState=exited\nResult=success\nMainPID=0\n"
            "ControlGroup=\nNRestarts=0\nExecMainStatus=0\nExecMainCode=1\nRestart=no\n"
            "KillMode=control-group\nType=exec\nRemainAfterExit=yes\n"
            "RuntimeMaxUSec=4h 2min 10s\nTimeoutStopUSec=10s\nMemoryMax=4294967296\n"
            "CPUQuotaPerSecUSec=1s\nTasksMax=128\nOOMPolicy=kill\n"))
    monkeypatch.setattr(monitor.subprocess, "run", run)
    monkeypatch.setattr(monitor, "count_processes", lambda *_: 0)
    state = monitor.systemd_state("hymem-luna-lme-multi-x.service",
        "/user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-multi-x.service")
    assert state["policy_ok"] is False
    assert monitor.completed_and_clean({"validated": True}, state) is False


def test_terminal_validation_accepts_false_score_and_rejects_bad_health(tmp_path, monkeypatch):
    root = tmp_path
    for name in ("luna_subscription_lme_multi.py", "luna_subscription_pilot.py",
                 "codex_subscription.py", "codex_subscription_concurrent.py",
                 "headless-source-map.json"):
        (root / name).write_text("pinned")
    dataset = root / "dataset.json"
    dataset.write_text("private dataset")
    expected = {"luna_subscription_lme_multi.py": monitor.RUNNER_SHA256,
                "luna_subscription_pilot.py": monitor.PILOT_SHA256,
                "codex_subscription.py": monitor.BASE_SHA256,
                "codex_subscription_concurrent.py": monitor.CONCURRENT_SHA256,
                "headless-source-map.json": "a" * 64,
                "dataset.json": monitor.DATASET_SHA256}
    monkeypatch.setattr(monitor, "digest", lambda path: expected[path.name])
    # The pinned helper is executed from verified bytes. The fake helper only
    # supplies inventory verification, never candidate/model behavior.
    pilot_bytes = b"def verify_inventory(*args):\n    return 508\n"
    (root / "luna_subscription_pilot.py").write_bytes(pilot_bytes)
    monkeypatch.setattr(monitor, "PILOT_SHA256", hashlib.sha256(pilot_bytes).hexdigest())
    expected["luna_subscription_pilot.py"] = monitor.PILOT_SHA256
    monkeypatch.setitem(sys.modules, "benchmarks", SimpleNamespace(
        lme_protocol=SimpleNamespace(__file__=str(root / "benchmarks" / "lme_protocol.py"),
            _validate_versioned_indexing=lambda *a, **k: True)))
    rows = []
    entries = []
    for index, correct in enumerate((False, True)):
        qdir = root / "run" / f"q-{index:04d}"
        qdir.mkdir(parents=True)
        row = {"correct": correct, "benchmark_failure": None,
               "judge_error": False, "judge_parse_valid": True,
               "indexing": {"outcome": "success", "healthy": True,
                            "summary_healthy": True}, "raw_answer": "secret"}
        entry = {"index": index, "correct": correct, "question_completed": True,
                 "cleanup_ok": True, "stop_code": None,
                 "dream_run_housekeeping": {"open_runs_before": 0,
                    "terminalized": 0, "open_runs_after": 0,
                    "active_leases_after": 0}}
        (qdir / "private-row.json").write_text(json.dumps(row))
        (qdir / "private-result.json").write_text(json.dumps(entry))
        rows.append(row)
        entries.append(entry)
    receipt = {"inventory_sha256": "a" * 64, "dataset": str(dataset),
               "candidate": str(root)}
    safe = {"ok": True, "questions_selected": 2, "questions_completed": 2,
            "canary_passed": True, "turns": 10, "known_tokens": 100,
            "usage_complete": True, "in_flight": 0,
            "source_files_verified": 508, "dataset_sha256": monitor.DATASET_SHA256,
            "runner_sha256": monitor.RUNNER_SHA256,
            "pilot_helper_sha256": monitor.PILOT_SHA256,
            "base_transport_sha256": monitor.BASE_SHA256,
            "concurrent_transport_sha256": monitor.CONCURRENT_SHA256,
            "correct_count": 1, "incorrect_count": 1,
            "failed_count": 0, "not_started_count": 0}
    result = {"schema": "luna-subscription-lme-multi-v1", "questions": entries,
              "campaign_stop": None, "canary": {"passed": True},
              "usage_complete_now": True, "active_invocations": 0,
              "budget": {"usage_complete": True, "in_flight": 0,
                         "reserved": 0, "turns": 10, "known_tokens": 100}}
    verdict = monitor.verify_terminal(root=root, receipt=receipt, safe=safe, result=result)
    assert verdict["validated"] is True
    assert [q["correct"] for q in verdict["questions"]] == [False, True]
    assert "secret" not in json.dumps(verdict)
    rows[1]["indexing"]["summary_healthy"] = False
    qdir = root / "run" / "q-0001"
    (qdir / "private-row.json").write_text(json.dumps(rows[1]))
    assert monitor.verify_terminal(root=root, receipt=receipt, safe=safe,
                                   result=result)["validated"] is False
