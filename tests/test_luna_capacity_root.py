"""Offline controls for the capacity-only four-question Luna envelope."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools/diagnostics"


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, TOOLS / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = _module("luna_capacity_launch_test", "luna_subscription_capacity_launch.py")
progress = _module("luna_capacity_progress_test", "luna_subscription_capacity_progress.py")


def test_exact_capacity_launch_envelope_and_unchanged_runner(monkeypatch):
    monkeypatch.setattr(launch, "sha", lambda _: "a" * 64)
    root = Path("/home/atta/.hymem-luna-lme-capacity-abcd1234")
    receipt = launch.receipt_for(root)
    cmd = launch.command(root, receipt)
    assert receipt["schema"] == "luna-capacity-launch-v1"
    assert receipt["tasks_max"] == 256
    assert receipt["unit"] == "hymem-luna-lme-capacity-abcd1234.service"
    assert receipt["source_sha256"] == launch.PINS
    assert receipt["runner_sha256"] == launch.PINS["luna_subscription_lme_profiled_v2.py"]
    for setting in ("RuntimeMaxSec=14530s", "TimeoutStopSec=10s",
                    "MemoryMax=4294967296", "CPUQuota=200%", "TasksMax=256",
                    "OOMPolicy=kill", "KillMode=control-group", "Restart=no"):
        assert "--property=" + setting in cmd
    assert "--property=TasksMax=128" not in cmd
    assert str(root / "luna_subscription_lme_profiled_v2.py") in cmd
    assert "--model" not in cmd and "--resume" not in cmd
    assert cmd[cmd.index("--output-dir") + 1] == str(root / "run")
    for key, value in launch.LIMITS.items():
        assert cmd[cmd.index("--" + key) + 1] == str(value)


def test_receipt_rejects_old_ceiling_and_any_other_policy_change(monkeypatch):
    monkeypatch.setattr(launch, "sha", lambda _: "a" * 64)
    root = Path("/home/atta/.hymem-luna-lme-capacity-abcd1234")
    receipt = launch.receipt_for(root)
    unit, cgroup = receipt["unit"], receipt["expected_cgroup"]
    monkeypatch.setattr(Path, "is_dir", lambda self: self == root)
    assert progress.root_identity_valid(root, unit, cgroup)
    assert progress.receipt_valid(receipt, root, unit, cgroup)
    for key, bad in (("tasks_max", 128), ("tasks_max", 257),
                     ("runtime_max_seconds", 14531), ("memory_max_bytes", 1),
                     ("cpu_quota_percent", 100), ("source_sha256", {}),
                     ("limits", {}), ("schema", "luna-profiled-launch-v2")):
        assert not progress.receipt_valid(dict(receipt, **{key: bad}), root, unit, cgroup)
    assert not progress.root_identity_valid(root, unit.replace("capacity", "attributed"), cgroup)
    assert not progress.root_identity_valid(root, unit, cgroup + "/foreign")


@pytest.mark.parametrize("bad", ["128", "257", "infinity"])
def test_systemd_policy_rejects_wrong_tasks_max(monkeypatch, bad):
    fields = ("ActiveState=active\nSubState=exited\nResult=success\nMainPID=0\n"
        "ControlGroup=\nNRestarts=0\nExecMainStatus=0\nExecMainCode=1\n"
        "Restart=no\nKillMode=control-group\nType=exec\nRemainAfterExit=yes\n"
        "RuntimeMaxUSec=4h 2min 10s\nTimeoutStopUSec=10s\nMemoryMax=4294967296\n"
        "CPUQuotaPerSecUSec=2s\nTasksMax=256\nOOMPolicy=kill")
    monkeypatch.setattr(progress, "count_processes", lambda _: 0)
    monkeypatch.setattr(progress, "cgroup_task_counters", lambda _: {})
    output = {"value": fields}
    monkeypatch.setattr(progress.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout=output["value"]))
    unit = "hymem-luna-lme-capacity-abcd1234.service"
    cgroup = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit
    assert progress.completed_and_clean({"validated": True}, progress.systemd_state(unit, cgroup))
    output["value"] = fields.replace("TasksMax=256", "TasksMax=" + bad)
    assert not progress.completed_and_clean({"validated": True}, progress.systemd_state(unit, cgroup))


def test_live_task_counters_and_terminal_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "CGROUP_ROOT", tmp_path)
    cgroup = "/user.slice/unit.service"
    assert progress.cgroup_task_counters(cgroup) == dict.fromkeys(
        ("pids_current", "pids_peak", "pids_max", "pids_events_max"))
    base = tmp_path / "user.slice" / "unit.service"
    base.mkdir(parents=True)
    for file, content in (("pids.current", "128\n"), ("pids.peak", "141\n"),
                          ("pids.max", "256\n"), ("pids.events", "max 3\n")):
        (base / file).write_text(content)
    assert progress.cgroup_task_counters(cgroup) == {
        "pids_current": 128, "pids_peak": 141, "pids_max": 256, "pids_events_max": 3}
    (base / "pids.events").write_text("max provider-secret\n")
    assert progress.cgroup_task_counters(cgroup)["pids_events_max"] is None
    (base / "pids.peak").unlink()
    assert progress.cgroup_task_counters(cgroup)["pids_peak"] is None


def test_one_shot_marker_is_exclusive(tmp_path):
    path = tmp_path / "launch-attempt.json"
    launch.write_once(path, {"one_shot": True})
    with pytest.raises(FileExistsError):
        launch.write_once(path, {"one_shot": False})
    assert json.loads(path.read_text()) == {"one_shot": True}
    assert path.stat().st_mode & 0o777 == 0o600


def test_launch_marks_attempt_before_dispatch_and_refuses_retry(tmp_path, monkeypatch, capsys):
    root = tmp_path / ".hymem-luna-lme-capacity-abcd1234"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(launch, "HOST_ROOT", tmp_path)
    monkeypatch.setattr(launch, "HOST_UID", root.stat().st_uid)
    monkeypatch.setattr(launch, "host_admission", lambda: None)
    monkeypatch.setattr(launch, "verify_sources", lambda _: None)
    monkeypatch.setattr(launch, "receipt_for", lambda _: {"unit": launch.unit_for(root)})
    monkeypatch.setattr(launch, "sha", lambda _: "a" * 64)
    (root / "launch-receipt.json").write_text(json.dumps({"unit": launch.unit_for(root)}))
    observed = []

    def dispatch(*args, **kwargs):
        observed.append((root / "launch-attempt.json").is_file())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, "run", dispatch)
    args = ["--launch-root", str(root), "--receipt-sha256", "a" * 64]
    assert launch.main(args) == 0
    assert observed == [True]
    assert launch.main(args) == 1
    assert observed == [True]
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["never_retry_launch"] is True


def test_progress_privacy_and_inflight_usage():
    value = {"budget": {"turns": 5, "known_tokens": 1000, "in_flight": 4,
                         "first_failure": {"code": "private error text"}},
             "active_invocations": 4, "usage_complete_now": True,
             "questions": [{"question_started": True, "stop_code": "private text"}]}
    summary = progress.progress_summary(value)
    assert summary["usage_complete"] is False
    assert summary["first_failure"] is None
    assert summary["questions"][0]["stop_code"] == "other"
    assert "private" not in str(summary)
