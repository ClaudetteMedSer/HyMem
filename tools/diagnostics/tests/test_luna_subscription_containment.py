"""Offline tests for the harmless systemd containment probe."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / "luna_subscription_containment.py"
SPEC = importlib.util.spec_from_file_location("luna_containment_subject", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_controller_uses_transient_no_restart_control_group(monkeypatch, tmp_path):
    output = tmp_path / "private"
    output.mkdir(mode=0o700)
    monkeypatch.setattr(probe.tempfile, "mkdtemp", lambda **kwargs: str(output))
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "/usr/bin/systemd-run":
            receipt = Path(command[-1])
            receipt.write_text(json.dumps({"parent_pid": 10, "parent_starttime": "1",
                                           "child_pid": 11, "child_starttime": "2"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    monkeypatch.setattr(probe, "_show", lambda unit: {"Restart": "no", "KillMode": "control-group",
        "RuntimeMaxUSec": "3s", "TimeoutStopUSec": "2s", "NRestarts": "0",
        "ActiveState": "inactive"})
    monkeypatch.setattr(probe, "_same_process", lambda pid, started: False)
    ticks = iter(range(20))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    report = probe._controller()
    assert report["ok"] and report["parent_gone"] and report["new_session_child_gone"]
    assert report["runtime_expiry_observed"]
    argv = commands[0]
    assert "--property=Type=exec" in argv
    assert "--property=KillMode=control-group" in argv
    assert "--property=Restart=no" in argv
    assert "--property=RuntimeMaxSec=3s" in argv
    assert "--property=TimeoutStopSec=2s" in argv
    assert not any("codex" in part or "luna_subscription_pilot" in part for part in argv)
    assert len(commands) == 1


def test_process_identity_requires_same_starttime(monkeypatch):
    monkeypatch.setattr(probe, "_starttime", lambda pid: "100" if pid == 10 else None)
    assert probe._same_process(10, "100")
    assert not probe._same_process(10, "101")
    assert not probe._same_process(11, "100")


def test_unverified_policy_stops_only_owned_unit(monkeypatch, tmp_path):
    output = tmp_path / "private"
    output.mkdir(mode=0o700)
    monkeypatch.setattr(probe.tempfile, "mkdtemp", lambda **kwargs: str(output))
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "/usr/bin/systemd-run":
            Path(command[-1]).write_text(json.dumps({
                "parent_pid": 10, "parent_starttime": "1",
                "child_pid": 11, "child_starttime": "2"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    monkeypatch.setattr(probe, "_show", lambda unit: {"KillMode": "process"})
    report = probe._controller()
    assert report["ok"] is False
    assert report["stop_code"] == "unit_policy_unverified"
    assert commands[-1] == ["/usr/bin/systemctl", "--user", "stop", report["unit"]]
    assert report["unit"].startswith(probe.UNIT_PREFIX)
