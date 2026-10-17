"""Fixture rollback/recovery checks; no production signals or provider calls."""

import hashlib
import os
import signal
import sqlite3
import stat
import sys
import time

import pytest

from tools.diagnostics import hermes_delayed_rollback as operator


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _private(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _proc(root, pid, role, ppid, start, *, cap=16):
    directory = root / str(pid)
    directory.mkdir()
    argv = (
        b"python3\0/home/node/.local/bin/hermes\0gateway\0run\0--replace\0"
        if role == "gateway" else
        f"/home/node/hymem-env/bin/{role}".encode() + b"\0"
    )
    (directory / "cmdline").write_bytes(argv)
    (directory / "stat").write_text(
        f"{pid} ({role}) S {ppid} " + "0 " * 17 + f"{start} 0\n"
    )
    (directory / "environ").write_bytes(
        b"UNRELATED_SECRET=fixture-secret\0"
        + (f"HYMEM_DREAM_MAX_FACTS_PER_SESSION={cap}".encode() + b"\0"
           if cap is not None else b"")
    )


def _fixture(tmp_path, monkeypatch, *, honcho=True, mcp=True):
    target = tmp_path / "code"
    _private(target)
    module_backup = tmp_path / "module-backups"
    cap_backup = tmp_path / "cap-backups"
    _private(module_backup)
    _private(cap_backup)
    manifest = {}
    for name in ("hymem/bootstrap.py", "hymem/dreaming/facts.py",
                 "hymem/dreaming/evidence.py", "hymem/dreaming/phase1.py"):
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("new:" + name).encode())
        path.chmod(0o664)
        saved = module_backup / name
        _private(saved.parent)
        saved.write_bytes(("old:" + name).encode())
        saved.chmod(0o600)
        manifest[name] = (_sha(saved.read_bytes()), _sha(path.read_bytes()))
    monkeypatch.setattr(operator.modules, "FILES", manifest)
    hook = tmp_path / "post-restart.sh"
    mcp_script = tmp_path / "wrapper"
    hook_old = (
        b"#!/bin/bash\n" + operator.restart.ENV_SOURCE + b"\n"
        + operator.restart.BLOCK_START
        + b"      HYMEM_FIXTURE=redacted \\\n" * 14
        + operator.restart.BLOCK_END
    )
    mcp_old = (
        b"#!/bin/bash\nexport HYMEM_AGGREGATION_NODES_ENABLED=1\n"
        b"exec hymem-server\n"
    )
    old = {"honcho": hook_old, "mcp": mcp_old}
    for role, path in (("honcho", hook), ("mcp", mcp_script)):
        path.write_bytes(operator.cap_tool._edit(old[role], 16, role))
        path.chmod(0o755)
        saved = cap_backup / f"{role}.before"
        saved.write_bytes(old[role])
        saved.chmod(0o600)
    monkeypatch.setattr(operator.cap_tool, "EXPECTED_OLD",
                        {role: _sha(payload) for role, payload in old.items()})
    monkeypatch.setattr(operator.restart, "CAP_HASHES", {
        16: {"honcho": _sha(hook.read_bytes()), "mcp": _sha(mcp_script.read_bytes())}
    })
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"DEEPSEEK_API_KEY=fixture-only\n")
    env_file.chmod(0o600)
    db = tmp_path / "hymem.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE run_lock (name TEXT)")
    proc = tmp_path / "proc"
    proc.mkdir()
    _proc(proc, 120, "gateway", 1, 998, cap=None)
    if honcho:
        _proc(proc, 21, "hymem-honcho", 97, 123)
    if mcp:
        _proc(proc, 22, "hymem-server", 120, 456)
    return {
        "target": target, "mcp_wrapper": mcp_script, "honcho_hook": hook,
        "module_backup": module_backup, "cap_backup": cap_backup,
        "env_file": env_file, "db_path": db, "proc_root": proc,
    }


@pytest.mark.parametrize("honcho,mcp", [(True, True), (False, True),
                                       (True, False), (False, False)])
def test_preview_accepts_exact_reviewed_partial_restart_states(
    tmp_path, monkeypatch, honcho, mcp,
):
    paths = _fixture(tmp_path, monkeypatch, honcho=honcho, mcp=mcp)
    result = operator.preview(16, **paths)
    assert result["ready"] and not result["restored"]
    assert result["files"] == 6
    assert result["gateway"] == {"pid": 120, "starttime": 998}
    assert (result["honcho"] is not None) == honcho
    assert (result["mcp"] is not None) == mcp


def test_exact_six_file_restore_preserves_modes_and_backup_bytes(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    planned = operator._files(16, **{key: paths[key] for key in (
        "target", "mcp_wrapper", "honcho_hook", "module_backup", "cap_backup"
    )})
    operator._restore_files(planned, deadline=operator.time.monotonic() + 10)
    assert len(planned) == 6
    for item in planned:
        assert item.path.read_bytes() == item.saved
        assert stat.S_IMODE(item.path.stat().st_mode) == item.mode
    for name in operator.modules.FILES:
        assert (paths["module_backup"] / name).read_bytes() == (
            "old:" + name
        ).encode()


def test_second_restore_failure_returns_every_file_to_new_version(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    planned = operator._files(16, **{key: paths[key] for key in (
        "target", "mcp_wrapper", "honcho_hook", "module_backup", "cap_backup"
    )})
    actual_replace = operator.os.replace
    calls = []

    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("injected second module restore failure")
        return actual_replace(source, destination)

    monkeypatch.setattr(operator.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        operator._restore_files(planned, deadline=operator.time.monotonic() + 10)
    assert len(calls) == 3
    assert all(item.path.read_bytes() == item.current for item in planned)


def test_slow_staging_expires_before_any_file_swap(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    planned = operator._files(16, **{key: paths[key] for key in (
        "target", "mcp_wrapper", "honcho_hook", "module_backup", "cap_backup"
    )})
    actual_stage = operator.modules._stage_replacement
    stages = []

    def slow_first_stage(*args):
        staged = actual_stage(*args)
        stages.append(staged)
        time.sleep(0.15)
        return staged

    monkeypatch.setattr(operator.modules, "_stage_replacement", slow_first_stage)
    monkeypatch.setattr(operator.os, "replace", lambda *_a: pytest.fail("file swapped"))
    with pytest.raises(operator.RollbackStopped, match="deadline elapsed"):
        operator._restore_files(planned, deadline=time.monotonic() + 0.1)
    assert len(stages) == 1
    assert not stages[0].exists()
    assert all(item.path.read_bytes() == item.current for item in planned)


def test_rollback_deadline_starts_before_watchdog_ack(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    gateway = operator.restart.Gateway(120, 998)
    children = (operator.Pin(21, "hymem-honcho", 123),
                operator.Pin(22, "hymem-server", 456))
    seen = {}

    class Watchdog:
        def terminate(self):
            pass

        def wait(self, *, timeout):
            return 0

    def slow_watchdog(_pins):
        time.sleep(0.05)
        seen["ack_at"] = time.monotonic()
        return Watchdog()

    def refuse_restore(_planned, *, deadline):
        seen["deadline"] = deadline
        raise operator.RollbackStopped("test stop before restore")

    monkeypatch.setattr(operator, "_watchdog", slow_watchdog)
    monkeypatch.setattr(operator, "_stop", lambda *_a: None)
    monkeypatch.setattr(operator, "_resume", lambda *_a: None)
    monkeypatch.setattr(operator, "_restore_files", refuse_restore)
    with pytest.raises(operator.RollbackStopped, match="test stop before restore"):
        operator.rollback(16, gateway, children, **paths)
    assert seen["deadline"] < seen["ack_at"] + operator.RESTORE_DEADLINE_SECONDS - 0.04


def test_unreviewed_process_cap_refuses_before_any_signal(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    (paths["proc_root"] / "21" / "environ").write_bytes(
        b"HYMEM_DREAM_MAX_FACTS_PER_SESSION=64\0"
    )
    with pytest.raises(operator.RollbackStopped, match="unreviewed fact cap"):
        operator.preview(16, **paths)


def test_rollback_pause_failure_resumes_every_stopped_pid(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    gateway = operator.restart.Gateway(120, 998)
    children = (operator.Pin(21, "hymem-honcho", 123),
                operator.Pin(22, "hymem-server", 456))
    stopped, resumed = [], []

    class Watchdog:
        terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, *, timeout):
            return 0

    watchdog = Watchdog()
    monkeypatch.setattr(operator, "_watchdog", lambda _pins: watchdog)

    def fail_second(pin, _proc):
        stopped.append(pin)
        if len(stopped) == 2:
            raise RuntimeError("injected pause failure")

    monkeypatch.setattr(operator, "_stop", fail_second)
    monkeypatch.setattr(operator, "_resume", lambda pin, _proc: resumed.append(pin))
    monkeypatch.setattr(operator, "_restore_files", lambda *_a, **_k: pytest.fail("restored"))
    with pytest.raises(RuntimeError, match="injected"):
        operator.rollback(16, gateway, children, **paths)
    assert resumed == list(reversed(stopped))
    assert watchdog.terminated


def test_pidfd_pause_resume_recheck_prevents_pid_reuse(monkeypatch):
    pin = operator.Pin(120, "gateway", 998)
    calls = []
    monkeypatch.setattr(operator, "_same", lambda *_a: True)
    monkeypatch.setattr(operator.os, "pidfd_open",
                        lambda pid: calls.append(("open", pid)) or 43)
    monkeypatch.setattr(operator.signal, "pidfd_send_signal",
                        lambda fd, sig: calls.append(("signal", fd, sig)))
    monkeypatch.setattr(operator.os, "close", lambda fd: calls.append(("close", fd)))
    monkeypatch.setattr(operator.quiesce, "_stat_state_starttime",
                        lambda _path: ("T", 998))
    operator._stop(pin, operator.restart.PROC)
    operator._resume(pin, operator.restart.PROC)
    assert calls == [
        ("open", 120), ("signal", 43, signal.SIGSTOP), ("close", 43),
        ("open", 120), ("signal", 43, signal.SIGCONT), ("close", 43),
    ]
    calls.clear()
    checks = iter((True, False))
    monkeypatch.setattr(operator, "_same", lambda *_a: next(checks))
    with pytest.raises(operator.RollbackStopped, match="identity changed"):
        operator._stop(pin, operator.restart.PROC)
    assert calls == [("open", 120), ("close", 43)]


def test_watchdog_cli_starts_without_cap_argument(monkeypatch):
    observed = []
    monkeypatch.setattr(operator, "_watchdog_main",
                        lambda pins, fd, proc: observed.append((pins, fd, proc)))
    monkeypatch.setattr(sys, "argv", [
        "rollback", "--watchdog", "--ack-fd", "4",
        "120:gateway:998", "22:hymem-server:456",
    ])
    assert operator.main() == 0
    assert observed[0][0] == (
        operator.Pin(120, "gateway", 998),
        operator.Pin(22, "hymem-server", 456),
    )
    assert observed[0][1] == 4


def test_new_child_during_pause_refuses_restore_and_resumes(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch, honcho=False, mcp=True)
    gateway = operator.restart.Gateway(120, 998)
    mcp = operator.Pin(22, "hymem-server", 456)
    surprise = operator.Pin(31, "hymem-honcho", 789)
    observed = [(gateway, (mcp,)), (gateway, (surprise, mcp))]
    stopped, resumed = [], []

    class Watchdog:
        terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, *, timeout):
            return 0

    watchdog = Watchdog()
    monkeypatch.setattr(operator, "_processes", lambda *_a: observed.pop(0))
    monkeypatch.setattr(operator, "_watchdog", lambda _pins: watchdog)
    monkeypatch.setattr(operator, "_stop", lambda pin, _proc: stopped.append(pin))
    monkeypatch.setattr(operator, "_resume", lambda pin, _proc: resumed.append(pin))
    monkeypatch.setattr(operator, "_restore_files", lambda *_a, **_k: pytest.fail("restored"))
    with pytest.raises(operator.RollbackStopped, match="process set changed"):
        operator.rollback(16, gateway, (mcp,), **paths)
    assert stopped == [operator.Pin(120, "gateway", 998), mcp]
    assert resumed == list(reversed(stopped))
    assert watchdog.terminated


def test_honcho_absent_relaunch_retries_without_signaling_unproven_child(monkeypatch):
    gateway = operator.restart.Gateway(120, 998)
    new_honcho = operator.quiesce.Process(31, "hymem-honcho", 789)
    new_mcp = operator.quiesce.Process(32, "hymem-server", 987)
    attempts = []
    monkeypatch.setattr(operator.restart, "_port_closed", lambda _port: True)
    monkeypatch.setattr(operator.restart, "_block", lambda _p: b"reviewed block")
    monkeypatch.setattr(operator.restart, "_launch_honcho",
                        lambda *_a, **_k: attempts.append(1))
    monkeypatch.setattr(operator.restart, "_wait_until",
                        lambda predicate, _seconds, reason:
                        None if predicate() else
                        (_ for _ in ()).throw(operator.restart.RestartStopped(reason)))
    monkeypatch.setattr(operator.restart, "_role_processes",
                        lambda role, _proc:
                        ([new_honcho] if len(attempts) >= 2 else [])
                        if role == "hymem-honcho" else [new_mcp])
    monkeypatch.setattr(operator, "_cap_absent", lambda *_a: True)
    monkeypatch.setattr(operator.restart, "_health", lambda *_a: True)
    monkeypatch.setattr(operator.restart, "_gateway_parent", lambda *_a: gateway)
    monkeypatch.setattr(operator.restart, "_mcp_protocol_probe", lambda: None)
    monkeypatch.setattr(operator.restart, "MCP_STABLE_SECONDS", 0)
    monkeypatch.setattr(operator.quiesce, "_same_process", lambda *_a: True)
    monkeypatch.setattr(operator.restart, "_terminate_exact",
                        lambda *_a: pytest.fail("signaled an unproven child"))
    result = operator._recover_services(
        gateway, (), honcho_hook=operator.cap_tool.HONCHO,
        env_file=operator.restart.HERMES_ENV, target=operator.modules.TARGET,
        db_path=operator.restart.DATABASE, proc_root=operator.restart.PROC,
    )
    assert len(attempts) == 2
    assert result["honcho_pid"] == 31 and result["mcp_pid"] == 32


def test_mcp_old_child_is_only_signaled_child_during_recovery(monkeypatch):
    gateway = operator.restart.Gateway(120, 998)
    old_mcp = operator.Pin(22, "hymem-server", 456)
    new_honcho = operator.quiesce.Process(31, "hymem-honcho", 789)
    new_mcp = operator.quiesce.Process(32, "hymem-server", 987)
    terminated = []
    monkeypatch.setattr(operator.restart, "_port_closed", lambda _port: True)
    monkeypatch.setattr(operator.restart, "_block", lambda _p: b"reviewed block")
    monkeypatch.setattr(operator.restart, "_launch_honcho", lambda *_a, **_k: None)
    monkeypatch.setattr(operator.restart, "_wait_until",
                        lambda predicate, _seconds, reason:
                        None if predicate() else
                        (_ for _ in ()).throw(operator.restart.RestartStopped(reason)))
    monkeypatch.setattr(operator.restart, "_role_processes",
                        lambda role, _proc: [new_honcho] if role == "hymem-honcho"
                        else ([new_mcp] if terminated else [
                            operator.quiesce.Process(22, "hymem-server", 456)
                        ]))
    monkeypatch.setattr(operator, "_cap_absent", lambda *_a: True)
    monkeypatch.setattr(operator.restart, "_health", lambda *_a: True)
    monkeypatch.setattr(operator.restart, "_gateway_parent", lambda *_a: gateway)
    monkeypatch.setattr(operator.restart, "_mcp_protocol_probe", lambda: None)
    monkeypatch.setattr(operator.restart, "MCP_STABLE_SECONDS", 0)
    monkeypatch.setattr(operator.quiesce, "_same_process", lambda *_a: True)
    monkeypatch.setattr(operator.quiesce, "_dream_idle", lambda _p: None)
    monkeypatch.setattr(operator.restart, "_terminate_exact",
                        lambda record, _p: terminated.append(record))
    monkeypatch.setattr(operator.restart, "_exited", lambda *_a: bool(terminated))
    result = operator._recover_services(
        gateway, (old_mcp,), honcho_hook=operator.cap_tool.HONCHO,
        env_file=operator.restart.HERMES_ENV, target=operator.modules.TARGET,
        db_path=operator.restart.DATABASE, proc_root=operator.restart.PROC,
    )
    assert terminated == [operator.quiesce.Process(22, "hymem-server", 456)]
    assert result["mcp_pid"] == 32
