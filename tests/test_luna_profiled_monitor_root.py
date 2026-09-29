"""Independent root controls for the four-worker metadata boundary."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/diagnostics/luna_subscription_profiled_progress.py"
spec = importlib.util.spec_from_file_location("root_profiled_monitor", SOURCE)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


@pytest.mark.parametrize("change", ["CPUQuotaPerSecUSec=3s", "MemoryMax=8589934592",
    "RuntimeMaxUSec=infinity", "OOMPolicy=continue", "TasksMax=256",
    "KillMode=process", "Restart=on-failure"])
def test_actual_systemd_parser_rejects_each_relaxed_policy(monkeypatch, change):
    fields = dict(line.split("=", 1) for line in (
        "ActiveState=active\nSubState=exited\nResult=success\nMainPID=0\n"
        "ControlGroup=\nNRestarts=0\nExecMainStatus=0\nExecMainCode=1\n"
        "Restart=no\nKillMode=control-group\nType=exec\nRemainAfterExit=yes\n"
        "RuntimeMaxUSec=4h 2min 10s\nTimeoutStopUSec=10s\nMemoryMax=4294967296\n"
        "CPUQuotaPerSecUSec=2s\nTasksMax=128\nOOMPolicy=kill").splitlines())
    monkeypatch.setattr(monitor, "count_processes", lambda _: 0)
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout="\n".join(k + "=" + v for k, v in fields.items())))
    unit = "hymem-luna-lme-profiled-4_yhy8b9.service"
    cgroup = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit
    assert monitor.completed_and_clean({"validated": True}, monitor.systemd_state(unit, cgroup))
    key, value = change.split("=", 1)
    fields[key] = value
    assert not monitor.completed_and_clean({"validated": True}, monitor.systemd_state(unit, cgroup))


def test_four_inflight_calls_do_not_claim_complete_usage_or_health():
    result = monitor.progress_summary({"budget": {"turns": 100,
        "known_tokens": 10000, "in_flight": 4}, "active_invocations": 4,
        "usage_complete_now": True, "questions": [{"question_started": True}] * 4})
    assert result["known_tokens"] == 10000
    assert result["usage_complete"] is False
    assert all(q["correct"] is None and q["indexing_healthy"] is None
               and q["summary_healthy"] is None for q in result["questions"])


def test_independently_bound_cgroup_residue_blocks_clean_status(monkeypatch):
    unit = {"available": True, "active_state": "active", "sub_state": "exited",
        "result": "success", "main_pid": 0, "restarts": 0, "exit_status": 0,
        "exit_code_kind": 1, "policy_ok": True, "cgroup_header_matches": False,
        "cgroup_header_empty": True, "expected_cgroup_processes": 1}
    assert not monitor.completed_and_clean({"validated": True}, unit)
    unit["expected_cgroup_processes"] = None
    assert not monitor.completed_and_clean({"validated": True}, unit)
