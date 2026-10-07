"""Independent launcher controls. Never call systemd or model providers."""
import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_timeout_launch_v1 as launch


def configure(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "HOST_UID", os.getuid())
    receipts = {}
    def receipt_for(root, digest, mode):
        return {"mode": mode, "unit": "synthetic-" + mode + ".service", "one_shot": True}
    def verify_receipt(root, digest, mode):
        name = "launch-receipt.json" if mode == "probe" else "containment-receipt.json"
        raw = (root / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        return json.loads(raw)
    fake = SimpleNamespace(receipt_for=receipt_for, verify_receipt=verify_receipt,
        verify_sources=lambda root: None, unit_for=lambda root, mode: "synthetic-" + mode + ".service")
    monkeypatch.setattr(launch, "_host", lambda root: fake)
    monkeypatch.setattr(launch, "host_admission", lambda *args: None)
    monkeypatch.setattr(launch, "_bus_env", lambda: {})
    return fake


def test_root_containment_then_probe_preparation_is_possible(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    first = launch.prepare(tmp_path, "containment")
    launch._write_once(tmp_path / "containment-attempt.json", {
        "receipt_sha256": first["receipt_sha256"], "one_shot": True})
    launch._write_once(tmp_path / "containment-execution-marker.json", {
        "receipt_sha256": first["receipt_sha256"], "execution_started": True})
    verified = []
    monkeypatch.setattr(launch, "_containment_verified", lambda *args: verified.append(True) or True)
    result = launch.prepare(tmp_path, "probe")
    assert result["mode"] == "probe" and result["model_calls"] == 0
    assert verified == [True]
    assert (tmp_path / "containment-attempt.json").exists()


def test_root_ambiguous_dispatch_is_consumed_and_never_retried(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    prepared = launch.prepare(tmp_path, "containment")
    calls = []
    syncs = []
    original_fsync = os.fsync
    def observed_fsync(fd):
        syncs.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        return original_fsync(fd)
    monkeypatch.setattr(launch.os, "fsync", observed_fsync)
    def ambiguous(*args, **kwargs):
        assert (tmp_path / "containment-attempt.json").exists()
        assert syncs and syncs[-1] is True
        calls.append(True)
        raise TimeoutError("PRIVATE-SYSTEMD-ERROR")
    monkeypatch.setattr(launch.subprocess, "run", ambiguous)
    result = launch.launch(tmp_path, prepared["receipt_sha256"], "containment")
    assert result["never_retry"] is True and result["command_returncode"] is None
    assert "PRIVATE" not in json.dumps(result)
    with pytest.raises(ValueError):
        launch.launch(tmp_path, prepared["receipt_sha256"], "containment")
    assert calls == [True]


def test_root_probe_without_containment_cannot_reach_dispatch(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    def denied(*args):
        raise ValueError("containment_cleanup_unverified")
    monkeypatch.setattr(launch, "_containment_verified", denied)
    with pytest.raises(ValueError, match="containment_cleanup_unverified"):
        launch.prepare(tmp_path, "probe")
    assert not (tmp_path / "launch-receipt.json").exists()
    assert not (tmp_path / "launch-attempt.json").exists()


@pytest.mark.parametrize("artifact", ["probe-result.json", "probe-execution-marker.json", "private-probe-stdout.log"])
def test_root_inconsistent_prior_state_is_still_consumed(tmp_path, monkeypatch, artifact):
    configure(tmp_path, monkeypatch)
    monkeypatch.setattr(launch, "_containment_verified", lambda *args: True)
    prepared = launch.prepare(tmp_path, "probe")
    (tmp_path / artifact).write_text("existing state")
    def forbidden(*args, **kwargs):
        pytest.fail("dispatch despite previous execution artifact")
    monkeypatch.setattr(launch.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="already_attempted"):
        launch.launch(tmp_path, prepared["receipt_sha256"], "probe")
    assert not (tmp_path / "launch-attempt.json").exists()


def test_root_generated_command_preserves_finite_policy(tmp_path):
    command = launch.command(tmp_path, {"unit": "synthetic-probe.service"}, "a" * 64, "probe")
    for expected in ("--property=RuntimeMaxSec=730s", "--property=TimeoutStopSec=10s",
        "--property=TasksMax=256", "--property=MemoryMax=4294967296",
        "--property=CPUQuota=200%", "--property=Restart=no", "--property=KillMode=control-group",
        "--property=RemainAfterExit=yes", "--property=UMask=0077", "--run-once", "-I", "-B"):
        assert expected in command
    assert "--collect" not in command
    assert command[command.index("/usr/bin/env") + 1] == "-i"
