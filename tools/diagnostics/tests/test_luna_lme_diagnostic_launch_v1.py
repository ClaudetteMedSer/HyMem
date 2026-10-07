"""Offline controls for the one-shot diagnostic launcher; no host dispatch."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_lme_diagnostic_launch_v1 as launch


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(launch, "HOST_ROOT", tmp_path)
    monkeypatch.setattr(launch, "HOST_UID", os.getuid())
    root = tmp_path / ".hymem-lme-diagnostic-test_1234"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    runner = SimpleNamespace(
        PINS={"benchmarks/something.py": "a" * 64},
        DIAGNOSTIC_HELPER_SHA256="b" * 64,
        ACCEPTED_MAP_SHA256="c" * 64,
        DATASET_SHA256="d" * 64,
        MAX_LIMITS={"turns": (1, 29)},
    )
    monkeypatch.setattr(launch, "host_admission", lambda: None)
    monkeypatch.setattr(launch, "verify_sources", lambda path: runner)
    return root, runner


def test_prepare_exact_receipt_and_no_dispatch(staged, monkeypatch):
    root, runner = staged
    monkeypatch.setattr(launch.subprocess, "run",
        lambda *a, **kw: pytest.fail("prepare attempted subprocess dispatch"))
    result = launch.prepare(root)
    receipt = json.loads((root / "launch-receipt.json").read_text())
    assert receipt == launch.receipt_for(root, runner)
    assert receipt["selected_count"] == receipt["workers"] == 4
    assert receipt["indexing_seconds"] == 10_800
    assert receipt["unit"].startswith("hymem-luna-lme-diagnostic-")
    assert result["receipt_sha256"] == launch._sha(root / "launch-receipt.json")
    assert result["model_calls"] == 0
    assert not (root / "launch-attempt.json").exists()
    assert not (root / "run").exists()
    with pytest.raises(ValueError, match="root_already_prepared"):
        launch.prepare(root)


def test_exact_bounded_command_and_bus_context(staged):
    root, runner = staged
    receipt = launch.receipt_for(root, runner)
    command = launch.command(root, receipt, "f" * 64)
    assert command[:5] == ["/usr/bin/systemd-run", "--user", "--quiet",
                           "--unit", receipt["unit"]]
    assert "--property=RuntimeMaxSec=14530s" in command
    assert "--property=TimeoutStopSec=10s" in command
    assert "--property=MemoryMax=4294967296" in command
    assert "--property=CPUQuota=200%" in command
    assert "--property=TasksMax=128" in command
    assert "--property=Restart=no" in command
    assert "--property=KillMode=control-group" in command
    assert command[command.index("/usr/bin/env") + 1] == "-i"
    assert "XDG_RUNTIME_DIR=/run/user/1000" in command
    assert "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus" in command
    assert command[-1] == "--run"
    assert command[command.index("--questions") + 1] == "4"
    assert command[command.index("--workers") + 1] == "4"
    assert command[command.index("--receipt-sha256") + 1] == "f" * 64
    assert "--preflight-only" not in command


def test_marker_precedes_only_dispatch_even_when_dispatch_fails(staged, monkeypatch):
    root, _ = staged
    digest = launch.prepare(root)["receipt_sha256"]
    monkeypatch.setattr(launch, "_bus_env", lambda: {})
    calls = []

    def failed_dispatch(cmd, **kwargs):
        calls.append(cmd)
        assert json.loads((root / "launch-attempt.json").read_text()) == {
            "receipt_sha256": digest, "one_shot": True}
        assert not (root / "run").exists()
        raise subprocess.TimeoutExpired(cmd, 20)

    monkeypatch.setattr(launch.subprocess, "run", failed_dispatch)
    with pytest.raises(subprocess.TimeoutExpired):
        launch.launch(root, digest)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="launch_already_attempted"):
        launch.launch(root, digest)
    assert len(calls) == 1


def test_changed_source_and_failed_admission_precede_marker(staged, monkeypatch):
    root, _ = staged
    digest = launch.prepare(root)["receipt_sha256"]
    monkeypatch.setattr(launch, "verify_sources",
        lambda path: (_ for _ in ()).throw(ValueError("code_source_drift")))
    with pytest.raises(ValueError, match="code_source_drift"):
        launch.launch(root, digest)
    assert not (root / "launch-attempt.json").exists()
    monkeypatch.setattr(launch, "host_admission",
        lambda: (_ for _ in ()).throw(ValueError("old_deepseek_running_or_unverified")))
    with pytest.raises(ValueError, match="old_deepseek_running_or_unverified"):
        launch.launch(root, digest)
    assert not (root / "launch-attempt.json").exists()


def test_runner_source_hash_drift_rejected_before_import(staged):
    root, _ = staged
    source = root / "code/tools/diagnostics/luna_lme_diagnostic_v1.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise AssertionError('must not import changed runner')\n")
    with pytest.raises(ValueError, match="runner_source_drift"):
        launch._load_runner(root)


def test_bus_requires_private_runtime_and_owned_socket(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    runtime.chmod(0o700)
    bus = runtime / "bus"
    monkeypatch.setattr(launch, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(launch, "BUS", bus)
    monkeypatch.setattr(launch, "HOST_UID", os.getuid())
    original_lstat = Path.lstat
    monkeypatch.setattr(Path, "lstat", lambda path: (
        SimpleNamespace(st_mode=stat.S_IFSOCK | 0o600, st_uid=os.getuid())
        if path == bus else original_lstat(path)))
    assert launch._bus_env()["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=" + str(bus)
    runtime.chmod(0o777)
    with pytest.raises(ValueError, match="user_runtime_invalid"):
        launch._bus_env()
