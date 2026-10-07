"""Offline fault checks for the staged startup derivation and sidecar."""
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.diagnostics import luna_staged_startup_bundle_v1 as builder


def adapter():
    module = ModuleType("staged_adapter_tested")
    module.__file__ = "/synthetic/adapter-run-staged-v1.py"
    exec(compile(builder.derive_adapter(), module.__file__, "exec"), module.__dict__)
    return module


def test_derivation_preserves_every_accepted_code_byte(tmp_path):
    target = tmp_path / "derived"
    result = builder.prepare(target)
    receipt = json.loads((target / builder.RECEIPT).read_bytes())
    original = builder.base_inventory()
    assert len(original) == result["code_files"] == 37
    assert receipt["base_derivation_receipt_sha256"] == builder.BASE_RECEIPT_SHA
    assert receipt["old_adapter_sha256"] == builder.OLD_ADAPTER_SHA
    for name, raw in original.items():
        assert (target / name).read_bytes() == raw
        assert receipt["output_sha256"][name] == builder.sha(raw)
    assert builder.sha((target / builder.ADAPTER).read_bytes()) == result["adapter_sha256"]
    assert result["model_calls"] == 0 and result["launched"] is False
    with pytest.raises(ValueError, match="output_boundary_invalid"):
        builder.prepare(target)


def test_base_and_old_adapter_pins_fail_closed(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    (base / "derivation-receipt.json").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="source_pin_invalid"):
        builder.base_inventory(base)
    old = tmp_path / "old.py"
    old.write_bytes(b"wrong")
    monkeypatch.setattr(builder, "OLD_ADAPTER", old)
    with pytest.raises(ValueError, match="source_pin_invalid"):
        builder.derive_adapter()


def _sealed(root, a, mode="smoke"):
    (root / "launch-receipt.json").write_bytes(b"receipt")
    (root / a.ADAPTER).write_bytes(b"adapter")
    receipt_sha = a.digest(root / "launch-receipt.json")
    adapter_sha = a.digest(root / a.ADAPTER)
    sidecar = {"schema": "luna-staged-probe-adapter-staged-v1",
               "base_receipt_sha256": receipt_sha, "adapter_sha256": adapter_sha,
               "entry": a.ADAPTER, "entry_action": mode,
               "bus_scope": "systemctl-and-systemd-run-only",
               "startup_failure_schema": "luna-staged-probe-startup-failure-staged-v1"}
    (root / a.SIDECAR).write_text(json.dumps(sidecar))
    return receipt_sha, adapter_sha, a.digest(root / a.SIDECAR)


def test_sidecar_mode_entry_and_hash_pins(tmp_path):
    a = adapter()
    receipt_sha, adapter_sha, sidecar_sha = _sealed(tmp_path, a)
    assert a.verify(tmp_path, receipt_sha, adapter_sha, "smoke", sidecar_sha)["entry_action"] == "smoke"
    with pytest.raises(ValueError, match="adapter_pin_invalid"):
        a.verify(tmp_path, receipt_sha, adapter_sha, "inference", sidecar_sha)
    sidecar = json.loads((tmp_path / a.SIDECAR).read_text())
    sidecar["entry"] = "wrong.py"
    (tmp_path / a.SIDECAR).write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="adapter_pin_invalid"):
        a.verify(tmp_path, receipt_sha, adapter_sha, "smoke", a.digest(tmp_path / a.SIDECAR))
    with pytest.raises(ValueError, match="adapter_pin_invalid"):
        a.verify(tmp_path, receipt_sha, "0" * 64, "smoke", sidecar_sha)


def test_containment_exact_staged_runner_command_and_restoration(monkeypatch):
    a = adapter()
    calls = []
    original = SimpleNamespace(run=lambda command, **kw: calls.append((command, kw)))
    props = ("ActiveState,SubState,MainPID,ControlGroup,NRestarts,MemoryMax,TasksMax,"
             "CPUQuotaPerSecUSec,KillMode,Restart,RemainAfterExit,OOMPolicy,"
             "RuntimeMaxUSec,TimeoutStopUSec")
    command = ["/usr/bin/systemctl", "--user", "show", "unit.service",
               "--property=" + props, "--no-pager"]
    def verify(_root, _receipt):
        run.subprocess.run(command, check=True)
        with pytest.raises(ValueError, match="containment_command_invalid"):
            run.subprocess.run(command[:-2] + ["--property=MainPID", "--no-pager"])
    run = SimpleNamespace(subprocess=original, verify_live_containment=verify)
    monkeypatch.setattr(a, "bus_environment", lambda: {"XDG_RUNTIME_DIR": "/run/user/1000"})
    a.contained(Path("/unused"), {"unit": "unit.service"}, run)
    assert run.subprocess is original
    assert calls == [(command, {"env": {"XDG_RUNTIME_DIR": "/run/user/1000"}, "check": True})]


def test_host_control_bus_does_not_reach_docker_or_children(monkeypatch):
    a = adapter()
    calls = []
    original = SimpleNamespace(run=lambda command, **kw: calls.append((command, kw)))
    host = SimpleNamespace(subprocess=original, OLD_DEEPSEEK="old")
    monkeypatch.setattr(a, "bus_environment", lambda: {"DBUS_SESSION_BUS_ADDRESS": "fixed"})
    listed = ["/usr/bin/systemctl", "--user", "list-units", "hymem-luna*",
              "--all", "--plain", "--no-legend", "--no-pager"]
    docker = ["/usr/bin/docker", "inspect", "--format", "{{.State.Running}}", "old"]
    def action():
        host.subprocess.run(listed)
        host.subprocess.run(docker)
        with pytest.raises(ValueError, match="host_command_invalid"):
            host.subprocess.run(["/usr/bin/python3", "child"])
    a.control_plane(host, action)
    assert host.subprocess is original
    assert calls[0][1]["env"] == {"DBUS_SESSION_BUS_ADDRESS": "fixed"}
    assert "env" not in calls[1][1]


def test_launch_one_dispatch_and_no_child_bus(monkeypatch, tmp_path):
    a = adapter()
    seen = []
    root = tmp_path
    old = str(root / "code/tools/diagnostics/luna_staged_run_v1.py")
    def launch(_root, _sha, *, dispatch):
        command = ["/usr/bin/systemd-run", "--user", "/usr/bin/env", "-i",
                   "HOME=/home/atta", "TMPDIR=" + str(root / "tmp"), old]
        dispatch(command, timeout=20)
        return {"never_retry": True}
    host = SimpleNamespace(subprocess=SimpleNamespace(run=lambda *x, **k: None),
                           OLD_DEEPSEEK="old", launch=launch)
    monkeypatch.setattr(a, "verify", lambda *x: {"entry_action": "smoke"})
    monkeypatch.setattr(a, "bus_environment", lambda: {"DBUS_SESSION_BUS_ADDRESS": "fixed"})
    monkeypatch.setattr(a.subprocess, "run", lambda c, **k: seen.append((c, k)))
    assert a.launch(root, "receipt", "adapter", "sidecar", host, "smoke")["never_retry"]
    assert len(seen) == 1
    assert seen[0][0].count(str(root / a.ADAPTER)) == 1
    assert seen[0][0][seen[0][0].index(str(root / a.ADAPTER)) + 1] == "smoke"
    assert seen[0][1]["env"] == {"DBUS_SESSION_BUS_ADDRESS": "fixed"}
    assert not any("DBUS_SESSION_BUS_ADDRESS=" in part for part in seen[0][0])


def test_zero_failure_requires_verified_marker_and_no_run(monkeypatch, tmp_path):
    a = adapter()
    a.__file__ = str(tmp_path / a.ADAPTER)
    receipt_sha, adapter_sha, sidecar_sha = _sealed(tmp_path, a, "inference")
    receipt = {"source_sha256": {}, "binary_sha256": "bin"}
    (tmp_path / "launch-receipt.json").write_text(json.dumps(receipt))
    receipt_sha = a.digest(tmp_path / "launch-receipt.json")
    sidecar = json.loads((tmp_path / a.SIDECAR).read_text())
    sidecar["base_receipt_sha256"] = receipt_sha
    (tmp_path / a.SIDECAR).write_text(json.dumps(sidecar))
    sidecar_sha = a.digest(tmp_path / a.SIDECAR)
    (tmp_path / "launch-attempt.json").write_text(json.dumps(
        {"receipt_sha256": receipt_sha, "one_shot": True}))
    monkeypatch.setattr(a, "verify", lambda *x: {})
    monkeypatch.setattr(a, "digest", lambda p: adapter_sha if p == Path(a.__file__) else hashlib.sha256(p.read_bytes()).hexdigest())
    run = SimpleNamespace(execute=lambda *x, **k: (_ for _ in ()).throw(ValueError("preflight")))
    def write_once(path, value):
        path.write_text(json.dumps(value))
    host = SimpleNamespace(root_valid=lambda r: True, regular=lambda p: p.is_file(),
        strict_equal=lambda x, y: x == y, receipt_for=lambda *x: receipt,
        verify_bundle=lambda *x, **k: None, write_once=write_once)
    monkeypatch.setattr(a, "load_source", lambda _p, _n, h: run if h == a.RUN_SHA else host)
    assert a.entry(tmp_path, receipt_sha, "inference", adapter_sha, sidecar_sha) == 1
    terminal = json.loads((tmp_path / "safe-terminal.json").read_text())
    assert terminal["model_calls"] == terminal["paid_turns"] == 0
    (tmp_path / "safe-terminal.json").unlink()
    (tmp_path / "run").mkdir()
    assert a.entry(tmp_path, receipt_sha, "inference", adapter_sha, sidecar_sha) == 1
    assert not (tmp_path / "safe-terminal.json").exists()


def test_inference_observer_delegates_staged_reader_and_replay(monkeypatch, tmp_path):
    a = adapter()
    root = tmp_path
    receipt = {"unit": "unit.service", "expected_cgroup": "/group",
               "source_sha256": {"tools/diagnostics/luna_classification_progress_reference_v3.py": "a" * 64},
               "binary_sha256": "bin"}
    (root / "launch-receipt.json").write_text(json.dumps(receipt))
    (root / "launch-attempt.json").write_text(json.dumps(
        {"receipt_sha256": "receipt", "one_shot": True}))
    (root / "launch-admission.json").write_text(json.dumps({
        "receipt_sha256": "receipt", "old_luna_stopped": True,
        "old_deepseek_stopped": True, "memory_floor_bytes": 6 * 1024**3,
        "disk_floor_bytes": 20 * 1024**3, "memory_floor_met": True,
        "disk_floor_met": True}))
    (root / "run").mkdir()
    (root / "run/private-result.json").write_text("{}")
    policy = {"available": True, "clean": True, "active_state": "inactive"}
    calls = []
    reference = SimpleNamespace(subprocess=SimpleNamespace(run=lambda *x, **k: None))
    progress = SimpleNamespace(subprocess=SimpleNamespace(run=lambda *x, **k: None),
        _systemd=lambda unit, group, ref: calls.append((unit, group, ref)) or policy,
        observe=lambda *x, **k: calls.append((x, k)) or {"replay_validated": True})
    host = SimpleNamespace(root_valid=lambda r: True, strict_equal=lambda x, y: x == y,
        receipt_for=lambda *x: receipt, verify_bundle=lambda *x, **k: None,
        regular=lambda p: p.is_file())
    monkeypatch.setattr(a, "verify", lambda *x: {"entry_action": "inference"})
    monkeypatch.setattr(a, "load_source", lambda p, n, h: reference if "reference" in str(p) else host)
    result = a.observe(root, "receipt", "adapter", "sidecar", progress)
    assert calls[0] == ("unit.service", "/group", reference)
    assert calls[1][0][0:2] == (root, "receipt")
    assert calls[1][1]["systemd"]("unit.service", "/group") is policy
    assert result["replay_validated"] is True
    assert result["zero_admission_verified"] is False
