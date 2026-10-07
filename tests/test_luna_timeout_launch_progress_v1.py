"""Offline controls for one-shot dispatch and finite terminal projection."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


BASE = Path(__file__).resolve().parents[1] / "tools/diagnostics"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + ".py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = _module("luna_timeout_launch_v1")
progress = _module("luna_timeout_progress_v1")


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def test_command_has_bounded_service_policy_and_no_collect(tmp_path):
    root = tmp_path / ".hymem-luna-timeout-abcdefgh"
    receipt = {"unit": "hymem-luna-timeout-containment-abcdefgh.service"}
    command = launch.command(root, receipt, "0" * 64, "containment")
    assert "--property=Type=exec" in command
    assert "--property=RemainAfterExit=yes" in command
    assert "--property=Restart=no" in command
    assert "--property=KillMode=control-group" in command
    assert "--property=RuntimeMaxSec=730s" in command
    assert "--property=TimeoutStopSec=10s" in command
    assert "--property=TasksMax=256" in command
    assert "--property=MemoryMax=4294967296" in command
    assert "--property=CPUQuota=200%" in command
    assert "--property=OOMPolicy=kill" in command
    assert "--property=UMask=0077" in command
    assert "--collect" not in command
    assert command.count("--containment-only") == 1
    assert "--run-once" not in command
    assert [x for x in command if x.startswith("HOME=")] == ["HOME=/home/atta"]


def test_probe_preparation_accepts_consumed_clean_containment(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-timeout-abcdefgh"
    root.mkdir()
    (root / "containment-attempt.json").write_bytes(b"old")
    (root / "containment-execution-marker.json").write_bytes(b"old")
    host = SimpleNamespace(verify_sources=lambda _: None,
        receipt_for=lambda *_: {"mode": "probe"},
        unit_for=lambda *_: "hymem-luna-timeout-probe-abcdefgh.service")
    monkeypatch.setattr(launch, "_host", lambda _: host)
    monkeypatch.setattr(launch, "host_admission", lambda *_: None)
    monkeypatch.setattr(launch, "_workdirs", lambda _: None)
    monkeypatch.setattr(launch, "_containment_verified", lambda *_: True)
    result = launch.prepare(root, "probe")
    assert result["mode"] == "probe" and result["prepared"] is True
    assert (root / "launch-receipt.json").read_bytes() == _bytes({"mode": "probe"})


def test_ambiguous_dispatch_consumes_attempt_before_systemd(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-timeout-abcdefgh"
    root.mkdir()
    receipt_path = root / "containment-receipt.json"
    receipt_path.write_bytes(b"receipt")
    digest = hashlib.sha256(b"receipt").hexdigest()
    host = SimpleNamespace(verify_receipt=lambda *_: {"unit": "unit.service"},
                           verify_sources=lambda _: None)
    monkeypatch.setattr(launch, "_host", lambda _: host)
    monkeypatch.setattr(launch, "host_admission", lambda *_: None)
    monkeypatch.setattr(launch, "_workdirs", lambda _: None)
    monkeypatch.setattr(launch, "_bus_env", lambda: {})
    def uncertain(*_, **__):
        assert (root / "containment-attempt.json").read_bytes() == _bytes(
            {"receipt_sha256": digest, "one_shot": True})
        raise TimeoutError("ambiguous")
    monkeypatch.setattr(launch.subprocess, "run", uncertain)
    result = launch.launch(root, digest, "containment")
    assert result["attempted"] is True and result["command_returncode"] is None
    with pytest.raises(ValueError, match="already_attempted"):
        launch.launch(root, digest, "containment")


def test_orphan_result_blocks_dispatch(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-timeout-abcdefgh"
    root.mkdir()
    receipt_path = root / "containment-receipt.json"
    receipt_path.write_bytes(b"receipt")
    digest = hashlib.sha256(b"receipt").hexdigest()
    (root / "containment-result.json").write_bytes(b"orphan")
    host = SimpleNamespace(verify_receipt=lambda *_: {"unit": "unit.service"},
                           verify_sources=lambda _: None)
    monkeypatch.setattr(launch, "_host", lambda _: host)
    monkeypatch.setattr(launch, "host_admission", lambda *_: None)
    monkeypatch.setattr(launch, "_workdirs", lambda _: None)
    with pytest.raises(ValueError, match="already_attempted"):
        launch.launch(root, digest, "containment")
    assert not (root / "containment-attempt.json").exists()


def test_containment_requires_execution_marker(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-timeout-abcdefgh"
    root.mkdir()
    receipt_path = root / "containment-receipt.json"
    receipt_path.write_bytes(b"receipt")
    digest = hashlib.sha256(b"receipt").hexdigest()
    (root / "containment-attempt.json").write_bytes(_bytes(
        {"receipt_sha256": digest, "one_shot": True}))
    (root / "containment-result.json").write_bytes(b"{}")
    monkeypatch.setattr(launch, "_regular", lambda path, *_: path.exists())
    host = SimpleNamespace(verify_receipt=lambda *_: {},
        validate_host_result=lambda *_: True,
        terminal_runtime=lambda *_: {})
    with pytest.raises(ValueError, match="file_missing_or_invalid|result_missing_or_invalid"):
        launch._containment_verified(host, root)


def test_terminal_missing_probe_result_keeps_calls_unknown(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-timeout-abcdefgh"
    root.mkdir()
    host = SimpleNamespace(PROBE_SHA256="a" * 64, OBSERVER_SHA256="b" * 64,
        PREPARATION_SHA256="c" * 64,
        verify_sources=lambda _: (object(), (object(), object())),
        terminal_runtime=lambda _: {"policy_verified": True, "unit_stopped": True,
            "group_matched": True, "recursive_cleanup_verified": True,
            "runtime_exit": "failure"})
    monkeypatch.setattr(progress, "_host", lambda _: host)
    monkeypatch.setattr(progress, "_receipt", lambda *_: {"unit": "unit.service"})
    monkeypatch.setattr(progress, "_attempt", lambda *_: True)
    monkeypatch.setattr(progress, "_execution", lambda *_: True)
    result = progress.inspect(root, "0" * 64, "probe")
    assert result["status"] == "result_missing"
    assert result["model_calls"] is None and result["probe"] is None
    assert result["completed_and_clean"] is False


def test_reader_rejects_duplicate_and_private_json(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    monkeypatch.setattr(progress, "_regular", lambda *_: True)
    path.write_bytes(b'{"schema":"x","schema":"x"}')
    with pytest.raises(ValueError, match="duplicate_json_key"):
        progress._read(path, 1000)
    path.write_bytes(b'{"schema":"x","private_text":"secret"}')
    value = progress._read(path, 1000)
    host = SimpleNamespace(validate_host_result=lambda *_args, **_kwargs: False)
    assert host.validate_host_result(value) is False
