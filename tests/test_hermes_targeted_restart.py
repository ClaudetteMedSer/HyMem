"""No-signal fixture checks for the targeted Hermes child restart."""

import hashlib
import os
import signal
import sqlite3
import stat
import subprocess

import pytest

from tools.diagnostics import hermes_targeted_restart as operator


def _records():
    old_honcho = operator.quiesce.Process(21, "hymem-honcho", 123)
    old_mcp = operator.quiesce.Process(22, "hymem-server", 456)
    new_honcho = operator.quiesce.Process(31, "hymem-honcho", 789)
    new_mcp = operator.quiesce.Process(32, "hymem-server", 987)
    return old_honcho, old_mcp, new_honcho, new_mcp


def _hook_block():
    return (
        operator.BLOCK_START
        + b"      HYMEM_FIXTURE=redacted \\\n" * 15
        + operator.BLOCK_END
    )


def _files(tmp_path, monkeypatch):
    hook = tmp_path / "post-restart.sh"
    wrapper = tmp_path / "wrapper"
    private = tmp_path / ".env"
    hook.write_bytes(b"#!/bin/bash\n" + operator.ENV_SOURCE + b"\n" + _hook_block())
    wrapper.write_bytes(b"#!/bin/bash\nexec fixture\n")
    private.write_bytes(b"DEEPSEEK_API_KEY=fixture-only\n")
    hook.chmod(0o755)
    wrapper.chmod(0o755)
    private.chmod(0o600)
    target = tmp_path / "code"
    (target / "hymem").mkdir(parents=True)
    module = target / "hymem" / "bootstrap.py"
    module.write_bytes(b"new fixture code\n")
    module.chmod(0o644)
    monkeypatch.setattr(operator, "CAP_HASHES", {
        16: {
            "honcho": hashlib.sha256(hook.read_bytes()).hexdigest(),
            "mcp": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        }
    })
    monkeypatch.setattr(operator.modules, "FILES", {
        "hymem/bootstrap.py": ("old", hashlib.sha256(module.read_bytes()).hexdigest())
    })
    db = tmp_path / "hymem.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE run_lock (name TEXT)")
    return hook, wrapper, private, target, db


def test_preflight_pins_cap_scripts_code_and_private_env_without_reading_env(
    tmp_path, monkeypatch,
):
    hook, wrapper, private, target, db = _files(tmp_path, monkeypatch)
    original_read = operator.Path.read_bytes

    def no_secret_read(path):
        if path == private:
            raise AssertionError("preflight read secret file")
        return original_read(path)

    monkeypatch.setattr(operator.Path, "read_bytes", no_secret_read)
    block = operator._preflight(
        16, honcho_hook=hook, mcp_wrapper=wrapper, target=target,
        env_file=private, db_path=db,
    )
    assert block == _hook_block()
    wrapper.write_bytes(wrapper.read_bytes() + b"# drift\n")
    with pytest.raises(operator.RestartStopped, match="launch script differs"):
        operator._preflight(
            16, honcho_hook=hook, mcp_wrapper=wrapper, target=target,
            env_file=private, db_path=db,
        )


def test_preflight_rejects_missing_or_ambiguous_launch_block(tmp_path):
    hook = tmp_path / "hook"
    hook.write_bytes(b"#!/bin/bash\n" + operator.ENV_SOURCE + b"\n" + _hook_block())
    assert operator._block(hook) == _hook_block()
    hook.write_bytes(hook.read_bytes() + _hook_block())
    with pytest.raises(operator.RestartStopped, match="launch block"):
        operator._block(hook)


def test_preview_reports_only_exact_process_ids_and_health(monkeypatch):
    old_honcho, old_mcp, *_ = _records()
    monkeypatch.setattr(operator, "_preflight", lambda *_a, **_k: _hook_block())
    monkeypatch.setattr(operator.quiesce, "_processes", lambda _p: (old_honcho, old_mcp))
    monkeypatch.setattr(operator, "_health", lambda *_a: True)
    monkeypatch.setattr(operator, "_terminate_exact", lambda *_a: pytest.fail("signaled"))
    result = operator.preview(16)
    assert result["ready"] and not result["restarted"]
    assert result["honcho"] == {"pid": 21, "starttime": 123}
    assert result["mcp"] == {"pid": 22, "starttime": 456}


def test_mismatched_pid_starttime_refuses_before_signal(monkeypatch):
    old_honcho, old_mcp, *_ = _records()
    monkeypatch.setattr(operator, "_preflight", lambda *_a, **_k: _hook_block())
    monkeypatch.setattr(operator.quiesce, "_processes", lambda _p: (old_honcho, old_mcp))
    monkeypatch.setattr(operator, "_health", lambda *_a: True)
    monkeypatch.setattr(operator, "_terminate_exact", lambda *_a: pytest.fail("signaled"))
    changed = operator.quiesce.Process(21, "hymem-honcho", 124)
    with pytest.raises(operator.RestartStopped, match="PID/starttime"):
        operator.restart(16, changed, old_mcp)


def test_restart_targets_only_old_children_and_waits_for_new_cap(monkeypatch):
    old_honcho, old_mcp, new_honcho, new_mcp = _records()
    events = []
    terminated = set()
    monkeypatch.setattr(operator, "_preflight", lambda *_a, **_k: _hook_block())
    monkeypatch.setattr(operator.quiesce, "_processes", lambda _p: (old_honcho, old_mcp))
    monkeypatch.setattr(operator.quiesce, "_dream_idle", lambda _p: events.append(("lease", "idle")))
    monkeypatch.setattr(operator.quiesce, "_same_process",
                        lambda record, _p: record not in terminated)
    monkeypatch.setattr(operator, "_health", lambda *_a: True)
    monkeypatch.setattr(operator, "_port_closed", lambda _port: True)
    monkeypatch.setattr(operator, "_exited", lambda record, _p: record in terminated)
    gateway = operator.Gateway(120, 998)
    monkeypatch.setattr(operator, "_gateway_parent", lambda _record, _p: gateway)
    monkeypatch.setattr(operator, "_mcp_protocol_probe",
                        lambda: events.append(("probe", "mcp")))
    monkeypatch.setattr(operator, "_observe_gateway_child",
                        lambda *_a: events.append(("observe", "mcp")))

    def terminate(record, _proc):
        events.append(("terminate", record.role))
        terminated.add(record)

    def recover(_block, old, _cap, *, env_file, target, proc_root):
        events.append(("launch", "hymem-honcho"))
        assert old == old_honcho
        return new_honcho

    def role_processes(role, _proc):
        if role == "hymem-honcho":
            return [new_honcho] if old_honcho in terminated else [old_honcho]
        return [new_mcp] if old_mcp in terminated else [old_mcp]

    monkeypatch.setattr(operator, "_terminate_exact", terminate)
    monkeypatch.setattr(operator, "_recover_honcho", recover)
    monkeypatch.setattr(operator, "_role_processes", role_processes)
    monkeypatch.setattr(operator, "_cap_in_process", lambda *_a: True)
    result = operator.restart(16, old_honcho, old_mcp)
    assert events == [
        ("lease", "idle"),
        ("terminate", "hymem-honcho"),
        ("launch", "hymem-honcho"),
        ("lease", "idle"),
        ("terminate", "hymem-server"),
        ("probe", "mcp"),
        ("observe", "mcp"),
    ]
    assert result["restarted"]
    assert result["honcho"] == {"old_pid": 21, "new_pid": 31}
    assert result["mcp"] == {"old_pid": 22, "new_pid": 32}


def test_honcho_shell_runs_only_reviewed_source_and_block_without_output(
    tmp_path, monkeypatch,
):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(operator.subprocess, "run", fake_run)
    operator._launch_honcho(_hook_block(), env_file=tmp_path / ".env", cwd=tmp_path)
    args, options = calls[0]
    assert args == ["/bin/bash", "-s"]
    assert options["input"] == (
        b"set -euo pipefail\nset -a; source "
        + str(tmp_path / ".env").encode()
        + b"; set +a\n" + _hook_block()
    )
    assert options["stdout"] is subprocess.DEVNULL
    assert options["stderr"] is subprocess.DEVNULL
    assert "DEEPSEEK_API_KEY" not in options["env"]


def test_pidfd_signal_targets_only_rechecked_process(monkeypatch):
    old_honcho, *_ = _records()
    calls = []
    monkeypatch.setattr(operator.quiesce, "_same_process", lambda *_a: True)
    monkeypatch.setattr(operator.os, "pidfd_open", lambda pid: calls.append(("open", pid)) or 42)
    monkeypatch.setattr(operator.signal, "pidfd_send_signal",
                        lambda fd, sig: calls.append(("signal", fd, sig)))
    monkeypatch.setattr(operator.os, "close", lambda fd: calls.append(("close", fd)))
    operator._terminate_exact(old_honcho, operator.PROC)
    assert calls == [("open", 21), ("signal", 42, signal.SIGTERM), ("close", 42)]
    calls.clear()
    monkeypatch.setattr(operator.quiesce, "_same_process", lambda *_a: False)
    with pytest.raises(operator.RestartStopped, match="changed"):
        operator._terminate_exact(old_honcho, operator.PROC)
    assert calls == []


def test_cap_check_selects_only_requested_nonsecret_variable(tmp_path, monkeypatch):
    old_honcho, *_ = _records()
    process = tmp_path / str(old_honcho.pid)
    process.mkdir()
    (process / "environ").write_bytes(
        b"UNRELATED_SECRET=fixture-secret\0"
        b"HYMEM_DREAM_MAX_FACTS_PER_SESSION=16\0"
    )
    monkeypatch.setattr(operator.quiesce, "_same_process", lambda *_a: True)
    assert operator._cap_in_process(old_honcho, 16, tmp_path)
    assert not operator._cap_in_process(old_honcho, 32, tmp_path)


def test_zombie_counts_as_exited_for_bounded_wait(tmp_path, monkeypatch):
    old_honcho, *_ = _records()
    proc = tmp_path / str(old_honcho.pid)
    proc.mkdir()
    (proc / "stat").write_text(
        "21 (hymem-honcho) Z " + "0 " * 18 + "123 0\n"
    )
    monkeypatch.setattr(operator.quiesce, "_same_process", lambda *_a: True)
    assert operator._exited(old_honcho, tmp_path)


def test_gateway_parent_requires_exact_gateway_identity(tmp_path, monkeypatch):
    _, old_mcp, *_ = _records()
    child = tmp_path / str(old_mcp.pid)
    parent = tmp_path / "120"
    child.mkdir()
    parent.mkdir()
    (child / "stat").write_text(
        "22 (hymem-server) S 120 " + "0 " * 17 + "456 0\n"
    )
    (parent / "stat").write_text(
        "120 (python3) S " + "0 " * 18 + "998 0\n"
    )
    (parent / "cmdline").write_bytes(
        b"python3\0/home/node/.local/bin/hermes\0gateway\0run\0--replace\0"
    )
    monkeypatch.setattr(operator.quiesce, "_same_process", lambda *_a: True)
    assert operator._gateway_parent(old_mcp, tmp_path) == operator.Gateway(120, 998)
    (parent / "cmdline").write_bytes(b"python3\0unrelated\0gateway\0run\0")
    with pytest.raises(operator.RestartStopped, match="not the reviewed Hermes gateway"):
        operator._gateway_parent(old_mcp, tmp_path)


def test_honcho_launch_failure_retries_boundedly_and_recovers(monkeypatch):
    old_honcho, _, new_honcho, _ = _records()
    attempts = []

    def launch(*_a, **_k):
        attempts.append("launch")
        if len(attempts) == 1:
            raise OSError("injected first launch failure")

    def wait(predicate, _seconds, reason):
        if not predicate():
            raise operator.RestartStopped(reason)

    monkeypatch.setattr(operator, "_launch_honcho", launch)
    monkeypatch.setattr(operator, "_wait_until", wait)
    monkeypatch.setattr(operator, "_port_closed", lambda _port: True)
    monkeypatch.setattr(operator, "_new_process", lambda *_a: new_honcho if len(attempts) > 1 else None)
    monkeypatch.setattr(operator, "_role_processes", lambda *_a: [])
    monkeypatch.setattr(operator, "_health", lambda *_a: True)
    recovered = operator._recover_honcho(
        _hook_block(), old_honcho, 16,
        env_file=operator.HERMES_ENV, target=operator.TARGET,
        proc_root=operator.PROC,
    )
    assert recovered == new_honcho
    assert len(attempts) == 2


def test_honcho_recovery_stops_after_three_failed_launches(monkeypatch):
    old_honcho, *_ = _records()
    attempts = []
    monkeypatch.setattr(operator, "_launch_honcho",
                        lambda *_a, **_k: attempts.append(1))
    monkeypatch.setattr(operator, "_wait_until",
                        lambda predicate, _seconds, reason:
                        None if predicate() else
                        (_ for _ in ()).throw(operator.RestartStopped(reason)))
    monkeypatch.setattr(operator, "_port_closed", lambda _port: True)
    monkeypatch.setattr(operator, "_new_process", lambda *_a: None)
    monkeypatch.setattr(operator, "_role_processes", lambda *_a: [])
    with pytest.raises(operator.RestartStopped, match="three reviewed"):
        operator._recover_honcho(
            _hook_block(), old_honcho, 16,
            env_file=operator.HERMES_ENV, target=operator.TARGET,
            proc_root=operator.PROC,
        )
    assert len(attempts) == 3


def test_honcho_recovery_never_terminates_unproven_replacement(monkeypatch):
    old_honcho, _, new_honcho, _ = _records()
    monkeypatch.setattr(operator, "_launch_honcho", lambda *_a, **_k: None)
    monkeypatch.setattr(operator, "_wait_until",
                        lambda predicate, _seconds, reason:
                        None if predicate() else
                        (_ for _ in ()).throw(operator.RestartStopped(reason)))
    monkeypatch.setattr(operator, "_port_closed", lambda _port: True)
    monkeypatch.setattr(operator, "_new_process", lambda *_a: new_honcho)
    monkeypatch.setattr(operator, "_health", lambda *_a: False)
    monkeypatch.setattr(operator, "_role_processes", lambda *_a: [new_honcho])
    monkeypatch.setattr(operator, "_terminate_exact",
                        lambda *_a: pytest.fail("terminated an unproven child"))
    with pytest.raises(operator.RestartStopped, match="manual recovery"):
        operator._recover_honcho(
            _hook_block(), old_honcho, 16,
            env_file=operator.HERMES_ENV, target=operator.TARGET,
            proc_root=operator.PROC,
        )


def test_mcp_protocol_probe_only_lists_tools_with_suppressed_output(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(operator.subprocess, "run", fake_run)
    operator._mcp_protocol_probe()
    args, options = calls[0]
    assert args == ["/home/node/.local/bin/hermes", "mcp", "test", "hymem"]
    assert options["stdin"] is subprocess.DEVNULL
    assert options["stdout"] is subprocess.DEVNULL
    assert options["stderr"] is subprocess.DEVNULL
    assert options["timeout"] == 45
    monkeypatch.setattr(operator.subprocess, "run",
                        lambda *_a, **_k: subprocess.CompletedProcess(args, 1))
    with pytest.raises(operator.RestartStopped, match="protocol probe failed"):
        operator._mcp_protocol_probe()


def test_mcp_stability_requires_same_child_parent_and_cap(monkeypatch):
    _, _, _, new_mcp = _records()
    gateway = operator.Gateway(120, 998)
    monkeypatch.setattr(operator, "MCP_STABLE_SECONDS", 0)
    monkeypatch.setattr(operator, "_role_processes", lambda *_a: [new_mcp])
    monkeypatch.setattr(operator, "_gateway_parent", lambda *_a: gateway)
    monkeypatch.setattr(operator, "_cap_in_process", lambda *_a: True)
    operator._observe_gateway_child(
        new_mcp, 16, gateway, operator.PROC, operator.time.monotonic()
    )
    monkeypatch.setattr(operator, "_gateway_parent",
                        lambda *_a: operator.Gateway(120, 999))
    with pytest.raises(operator.RestartStopped, match="changed"):
        operator._observe_gateway_child(
            new_mcp, 16, gateway, operator.PROC, operator.time.monotonic()
        )
