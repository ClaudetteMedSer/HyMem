"""Root-owned checks for the new host boundary; no host or model calls."""
from pathlib import Path
import hashlib
import json
import os

import pytest

from tools.diagnostics import luna_timeout_host_v1 as host


def policy():
    return {
        "ActiveState": "active", "SubState": "running", "MainPID": "4321",
        "ControlGroup": "/user.slice/probe.service", "NRestarts": "0",
        "Type": "exec", "MemoryMax": "4294967296", "TasksMax": "256",
        "CPUQuotaPerSecUSec": "2s", "KillMode": "control-group",
        "Restart": "no", "OOMPolicy": "kill", "RuntimeMaxUSec": "12min 10s",
        "TimeoutStopUSec": "10s", "UMask": "0077", "RemainAfterExit": "yes",
        "Result": "success", "ExecMainStatus": "0",
    }


def empty_group(path):
    path.mkdir(parents=True)
    for name, contents in {
        "cgroup.procs": "", "cgroup.threads": "", "cgroup.events": "populated 0\nfrozen 0\n",
        "memory.max": "4294967296\n", "pids.max": "256\n", "cpu.max": "200000 100000\n",
    }.items():
        (path / name).write_text(contents)
    return path


@pytest.mark.parametrize("field,value", [
    ("UMask", "0007"), ("RemainAfterExit", "no"), ("KillMode", "process"),
    ("Restart", "always"), ("RuntimeMaxUSec", "infinity"), ("TasksMax", "infinity"),
    ("MemoryMax", "8589934592"), ("CPUQuotaPerSecUSec", "4s"),
])
def test_root_actual_unit_policy_rejects_drift(field, value):
    values = policy()
    assert host._policy_ok(values)
    values[field] = value
    assert not host._policy_ok(values)


def test_root_terminal_main_exit_does_not_hide_nested_child(tmp_path, monkeypatch):
    group = empty_group(tmp_path / "probe.service")
    child = empty_group(group / "nested")
    (child / "cgroup.procs").write_text("98765\n")
    (child / "cgroup.events").write_text("populated 1\n")
    values = {**policy(), "SubState": "exited", "MainPID": "0"}
    monkeypatch.setattr(host, "_unit_values", lambda unit: values)
    monkeypatch.setattr(host, "_group", lambda receipt: group)
    result = host.terminal_runtime({"unit": "probe.service", "expected_cgroup": values["ControlGroup"]})
    assert result["unit_stopped"] is True
    assert result["runtime_exit"] == "success"
    assert result["recursive_cleanup_verified"] is False


@pytest.mark.parametrize("mutation", ["running", "policy", "group_policy"])
def test_root_empty_group_does_not_hide_wrong_terminal_policy(tmp_path, monkeypatch, mutation):
    group = empty_group(tmp_path / "probe.service")
    values = {**policy(), "SubState": "exited", "MainPID": "0"}
    if mutation == "running":
        values["SubState"], values["MainPID"] = "running", "4321"
    elif mutation == "policy":
        values["Restart"] = "always"
    else:
        (group / "pids.max").write_text("512\n")
    monkeypatch.setattr(host, "_unit_values", lambda unit: values)
    monkeypatch.setattr(host, "_group", lambda receipt: group)
    result = host.terminal_runtime({"unit": "probe.service", "expected_cgroup": values["ControlGroup"]})
    assert result["recursive_cleanup_verified"] is False


def test_root_absent_group_and_stopped_unit_can_verify_cleanup(tmp_path, monkeypatch):
    values = {**policy(), "SubState": "exited", "MainPID": "0", "ControlGroup": ""}
    monkeypatch.setattr(host, "_unit_values", lambda unit: values)
    monkeypatch.setattr(host, "_group", lambda receipt: tmp_path / "absent")
    result = host.terminal_runtime({"unit": "probe.service", "expected_cgroup": "/user.slice/probe.service"})
    assert result["recursive_cleanup_verified"] is True
    assert result["runtime_exit"] == "success"


@pytest.mark.parametrize("contents", ["max 0\nmax 1\n", "max -1\n", "max nope\n", "max 0 extra\n"])
def test_root_counter_parse_rejects_duplicate_and_malformed_values(tmp_path, contents):
    path = tmp_path / "pids.events"
    path.write_text(contents)
    with pytest.raises((ValueError, KeyError)):
        host._counter(path, "max")


def test_root_dangling_group_link_is_not_proven_absent(tmp_path):
    path = tmp_path / "linked"
    path.symlink_to(tmp_path / "missing")
    assert host.recursive_empty(path) is False


def receipt_fixture(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-timeout-rootchecks"
    root.mkdir(mode=0o700)
    script = root / "timeout-host-v1.py"
    script.write_bytes(b"# synthetic host source\n")
    script.chmod(0o600)
    monkeypatch.setattr(host, "HOST_HOME", tmp_path)
    monkeypatch.setattr(host, "HOST_UID", os.getuid())
    monkeypatch.setattr(host, "__file__", str(script))
    receipt = host.receipt_for(root, hashlib.sha256(script.read_bytes()).hexdigest(), "probe")
    return root, script, receipt


def save_receipt(root, receipt, raw=None):
    data = raw or json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    path = root / "launch-receipt.json"
    path.write_bytes(data)
    path.chmod(0o600)
    return hashlib.sha256(data).hexdigest()


def test_root_receipt_binds_host_origin_not_just_equal_bytes(tmp_path, monkeypatch):
    root, script, receipt = receipt_fixture(tmp_path, monkeypatch)
    digest = save_receipt(root, receipt)
    assert host.verify_receipt(root, digest, "probe") == receipt
    outside = tmp_path / "same-bytes.py"
    outside.write_bytes(script.read_bytes())
    outside.chmod(0o600)
    monkeypatch.setattr(host, "__file__", str(outside))
    with pytest.raises(ValueError):
        host.verify_receipt(root, digest, "probe")


@pytest.mark.parametrize("mutation", ["float", "bool", "duplicate"])
def test_root_rehashed_noncanonical_receipt_is_rejected(tmp_path, monkeypatch, mutation):
    root, _, receipt = receipt_fixture(tmp_path, monkeypatch)
    if mutation == "float":
        receipt["turns"] = 16.0
    elif mutation == "bool":
        receipt["one_shot"] = 1
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("ascii")
    if mutation == "duplicate":
        raw = raw.replace(b'"one_shot":true', b'"one_shot":false,"one_shot":true')
    digest = save_receipt(root, receipt, raw)
    with pytest.raises(ValueError):
        host.verify_receipt(root, digest, "probe")


def test_root_containment_only_never_loads_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "_base", lambda *args: {})
    counters = dict(pids_denials=0, memory_oom=0, memory_oom_kill=0,
                    pids_current=1, memory_current=100, memory_peak=100)
    monkeypatch.setattr(host, "live_attestation", lambda *args: (
        {"containment": True, "denials": 0, "oom": 0}, counters))
    def prohibited(*args):
        raise AssertionError("no source execution in containment mode")
    monkeypatch.setattr(host, "verify_sources", prohibited)
    result = host.run_containment_only(tmp_path, "a" * 64)
    assert result["verified"] is True and result["model_calls"] == 0
    assert (tmp_path / "containment-result.json").stat().st_mode & 0o777 == 0o600


def test_root_invalid_probe_result_is_not_preserved(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(host, "_base", lambda *args: {})
    counters = dict(pids_denials=0, memory_oom=0, memory_oom_kill=0,
                    pids_current=1, memory_current=100, memory_peak=100)
    monkeypatch.setattr(host, "live_attestation", lambda *args: (
        {"containment": True, "denials": 0, "oom": 0}, counters))
    monkeypatch.setattr(host, "_regular", lambda path: True)
    monkeypatch.setattr(host, "_sha", lambda path: host.BINARY_SHA256)
    probe = SimpleNamespace(run_probe=lambda *a, **kw: {"private": "DO-NOT-EXPORT"},
                            validate_result=lambda *args: False)
    monkeypatch.setattr(host, "verify_sources", lambda root: (probe, (object(), object())))
    result = host.run_once(tmp_path, "b" * 64)
    assert result["probe_result"] is None
    assert result["failure_code"] == "probe_result_invalid"
    assert "DO-NOT-EXPORT" not in (tmp_path / "probe-result.json").read_text()


@pytest.mark.parametrize("mutation", [None, "pid", "group", "membership", "kernel_policy", "oom"])
def test_root_live_boundary_uses_real_identity_not_only_policy(tmp_path, monkeypatch, mutation):
    values = {**policy(), "MainPID": str(os.getpid())}
    expected = values["ControlGroup"]
    group = empty_group(tmp_path / "live.service")
    (group / "cgroup.procs").write_text(str(os.getpid()) + "\n")
    (group / "pids.events").write_text("max 0\n")
    (group / "memory.events").write_text("oom 0\noom_kill 0\n")
    for name in ("pids.current", "memory.current", "memory.peak"):
        (group / name).write_text("1\n")
    if mutation == "pid":
        values["MainPID"] = "99999999"
    elif mutation == "group":
        values["ControlGroup"] = "/user.slice/other.service"
    elif mutation == "membership":
        (group / "cgroup.procs").write_text("99999999\n")
    elif mutation == "kernel_policy":
        (group / "pids.max").write_text("512\n")
    elif mutation == "oom":
        (group / "memory.events").write_text("oom 1\noom_kill 0\n")
    monkeypatch.setattr(host, "_unit_values", lambda unit: values)
    monkeypatch.setattr(host, "_group", lambda receipt: group)
    original = Path.read_text
    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return "0::" + expected + "\n"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)
    gate, _ = host.live_attestation({"unit": "probe.service", "expected_cgroup": expected}, "initial", None)
    if mutation is None:
        assert gate == {"containment": True, "denials": 0, "oom": 0}
    elif mutation == "oom":
        assert gate == {"containment": True, "denials": 0, "oom": 1}
    else:
        assert gate == {"containment": False, "denials": None, "oom": None}
