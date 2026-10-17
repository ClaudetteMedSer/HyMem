"""Isolated, no-service fixtures for the cap-64/token-6144 cutover."""

import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tools.diagnostics import hermes_fact_cap64_cutover as operator


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path, monkeypatch):
    stage = tmp_path / "stage"
    target = tmp_path / "target"
    backup_parent = tmp_path / "backups"
    for directory in (stage, target, backup_parent, stage / "hymem", target / "hymem"):
        directory.mkdir(mode=0o700)
    old_module = b"VALUE = 32\n"
    new_module = b"VALUE = 64\n"
    (target / "hymem/config.py").write_bytes(old_module)
    (target / "hymem/config.py").chmod(0o664)
    (stage / "hymem/config.py").write_bytes(new_module)
    (stage / "hymem/config.py").chmod(0o600)
    mcp = tmp_path / "mcp"
    honcho = tmp_path / "honcho"
    old_mcp = b"export HYMEM_DREAM_MAX_FACTS_PER_SESSION=32\n"
    old_honcho = b"      HYMEM_DREAM_MAX_FACTS_PER_SESSION=32 \\\n"
    new_mcp = (b"export HYMEM_DREAM_MAX_FACTS_PER_SESSION=64\n"
               b"export HYMEM_FACTS_EXTRACTION_MAX_TOKENS=6144\n")
    new_honcho = (b"      HYMEM_DREAM_MAX_FACTS_PER_SESSION=64 "
                  b"HYMEM_FACTS_EXTRACTION_MAX_TOKENS=6144 \\\n")
    mcp.write_bytes(old_mcp)
    honcho.write_bytes(old_honcho)
    mcp.chmod(0o755)
    honcho.chmod(0o755)
    monkeypatch.setattr(operator, "MODULES", {"hymem/config.py": (_sha(old_module), _sha(new_module))})
    monkeypatch.setattr(operator, "SCRIPTS", {
        "mcp": (mcp, _sha(old_mcp), _sha(new_mcp), old_mcp, new_mcp),
        "honcho": (honcho, _sha(old_honcho), _sha(new_honcho), old_honcho, new_honcho),
    })
    monkeypatch.setattr(operator, "_prior_backups_ready", lambda: None)
    pins = (operator.quiesce.Process(21, "hymem-honcho", 121),
            operator.quiesce.Process(22, "hymem-server", 122),
            operator.restart.Gateway(23, 123))
    monkeypatch.setattr(operator, "_pins", lambda: pins)
    monkeypatch.setattr(operator, "_same_idle_processes", lambda _pins: None)
    monkeypatch.setattr(operator, "_same_stopped_processes", lambda _pins: None)
    monkeypatch.setattr(operator, "_stop_exact", lambda _record: None)
    monkeypatch.setattr(operator, "_resume_exact", lambda _record: None)
    class Watchdog:
        def terminate(self):
            pass
        def wait(self, timeout):
            assert timeout == 2.0
    monkeypatch.setattr(operator, "_start_watchdog", lambda _pins: (Watchdog(), time.monotonic()))
    return stage, target, backup_parent / "cap32", pins, old_mcp, old_honcho


def test_preview_is_read_only_and_reports_exact_pins(tmp_path, monkeypatch):
    stage, target, backup, pins, old_mcp, old_honcho = _fixture(tmp_path, monkeypatch)
    result = operator.cutover(stage=stage, target=target, backup=backup)
    assert result["applied"] is False and result["new_fact_tokens"] == 6144
    assert result["gateway"] == {"pid": 23, "starttime": 123}
    assert not backup.exists()
    assert (tmp_path / "mcp").read_bytes() == old_mcp
    assert (tmp_path / "honcho").read_bytes() == old_honcho


def test_current_pins_validate_running_cap32_children_not_prior_cap16_preview(monkeypatch):
    honcho = operator.quiesce.Process(21, "hymem-honcho", 121)
    mcp = operator.quiesce.Process(22, "hymem-server", 122)
    gateway = operator.restart.Gateway(23, 123)
    events = []
    monkeypatch.setattr(operator.quiesce, "_processes", lambda _proc: (honcho, mcp))
    monkeypatch.setattr(operator.restart, "_gateway_parent",
                        lambda _mcp, _proc: gateway)
    monkeypatch.setattr(operator, "_same_idle_processes",
                        lambda pins, **_kw: events.append(pins))
    assert operator._pins() == (honcho, mcp, gateway)
    assert events == [(honcho, mcp, gateway)]


def test_apply_requires_exact_pins_before_backup(tmp_path, monkeypatch):
    stage, target, backup, pins, *_ = _fixture(tmp_path, monkeypatch)
    wrong = (pins[0], pins[1], operator.restart.Gateway(23, 124))
    with pytest.raises(operator.Cap64CutoverStopped, match="PID/starttime"):
        operator.cutover(stage=stage, target=target, backup=backup,
                         apply=True, expected=wrong)
    assert not backup.exists()


def test_apply_keeps_private_backup_and_byte_minimal_edits(tmp_path, monkeypatch):
    stage, target, backup, pins, old_mcp, old_honcho = _fixture(tmp_path, monkeypatch)
    result = operator.cutover(stage=stage, target=target, backup=backup,
                              apply=True, expected=pins)
    assert result["applied"] is True
    assert backup.stat().st_mode & 0o777 == 0o700
    for name, old in (("mcp", old_mcp), ("honcho", old_honcho)):
        assert (backup / name).read_bytes() == old
        assert (backup / name).stat().st_mode & 0o777 == 0o600
    assert (backup / "hymem/config.py").read_bytes() == b"VALUE = 32\n"
    assert (target / "hymem/config.py").read_bytes() == b"VALUE = 64\n"
    assert (tmp_path / "mcp").read_bytes().count(operator.TOKEN_PREFIX) == 1
    assert (tmp_path / "honcho").read_bytes().count(b"\n") == old_honcho.count(b"\n")


def test_install_failure_restores_already_replaced_files(tmp_path, monkeypatch):
    stage, target, backup, pins, old_mcp, old_honcho = _fixture(tmp_path, monkeypatch)
    real_replace = operator.os.replace
    calls = 0
    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected atomic rename failure")
        real_replace(source, destination)
    monkeypatch.setattr(operator.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        operator.cutover(stage=stage, target=target, backup=backup,
                         apply=True, expected=pins)
    assert (target / "hymem/config.py").read_bytes() == b"VALUE = 32\n"
    assert (tmp_path / "mcp").read_bytes() == old_mcp
    assert (tmp_path / "honcho").read_bytes() == old_honcho


def test_failed_rollback_keeps_all_stopped_until_watchdog_restores(tmp_path, monkeypatch):
    stage, target, backup, pins, old_mcp, old_honcho = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "TARGET", target)
    events = []
    class Watchdog:
        def terminate(self):
            events.append("watchdog-canceled")
        def wait(self, timeout):
            events.append("watchdog-waited")
    monkeypatch.setattr(operator, "_start_watchdog",
                        lambda _pins: (Watchdog(), time.monotonic()))
    monkeypatch.setattr(operator, "_resume_exact",
                        lambda record: events.append(("resume", record.pid)))
    real_replace = operator.os.replace
    calls = 0
    def fail_install_and_first_rollback(source, destination):
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise OSError("injected replace failure")
        real_replace(source, destination)
    monkeypatch.setattr(operator.os, "replace", fail_install_and_first_rollback)
    with pytest.raises(operator.Cap64CutoverStopped, match="watchdog restore"):
        operator.cutover(stage=stage, target=target, backup=backup,
                         apply=True, expected=pins)
    assert events == []  # No SIGCONT or watchdog cancel over mixed code.
    assert (target / "hymem/config.py").read_bytes() == b"VALUE = 64\n"
    assert (tmp_path / "mcp").read_bytes() == old_mcp
    monkeypatch.setattr(operator, "BACKUP", backup)
    monkeypatch.setattr(operator.time, "sleep", lambda _seconds: None)
    read_fd, write_fd = os.pipe()
    try:
        operator._watchdog_main(pins, write_fd)
        assert os.read(read_fd, 1) == b"R"
    finally:
        os.close(read_fd)
    assert (target / "hymem/config.py").read_bytes() == b"VALUE = 32\n"
    assert (tmp_path / "honcho").read_bytes() == old_honcho
    assert events == [("resume", 21), ("resume", 22), ("resume", 23)]


def test_changed_staged_code_or_symlink_refuses_before_backup(tmp_path, monkeypatch):
    stage, target, backup, pins, *_ = _fixture(tmp_path, monkeypatch)
    (stage / "hymem/config.py").write_bytes(b"VALUE = 65\n")
    with pytest.raises(operator.Cap64CutoverStopped, match="reviewed hash"):
        operator.cutover(stage=stage, target=target, backup=backup,
                         apply=True, expected=pins)
    assert not backup.exists()
    (stage / "hymem/config.py").unlink()
    (stage / "hymem/config.py").symlink_to(target / "hymem/config.py")
    with pytest.raises(OSError):
        operator.cutover(stage=stage, target=target, backup=backup)


def test_held_idle_guard_refuses_without_backup_or_edit(tmp_path, monkeypatch):
    stage, target, backup, pins, old_mcp, *_ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "_same_idle_processes",
                        lambda _pins: (_ for _ in ()).throw(
                            operator.Cap64CutoverStopped("dreaming lease is held")))
    with pytest.raises(operator.Cap64CutoverStopped, match="lease"):
        operator.cutover(stage=stage, target=target, backup=backup,
                         apply=True, expected=pins)
    assert not backup.exists()
    assert (tmp_path / "mcp").read_bytes() == old_mcp


def test_gateway_and_children_are_stopped_and_resumed_in_safe_order(tmp_path, monkeypatch):
    stage, target, backup, pins, *_ = _fixture(tmp_path, monkeypatch)
    events = []
    class Watchdog:
        def terminate(self):
            events.append("watchdog-cancel")
        def wait(self, timeout):
            assert timeout == 2.0
    monkeypatch.setattr(operator, "_start_watchdog",
                        lambda _pins: events.append("watchdog-start") or
                        (Watchdog(), time.monotonic()))
    monkeypatch.setattr(operator, "_stop_exact",
                        lambda record: events.append(("stop", record.pid)))
    monkeypatch.setattr(operator, "_resume_exact",
                        lambda record: events.append(("resume", record.pid)))
    monkeypatch.setattr(operator, "_same_stopped_processes",
                        lambda _pins: events.append("all-stopped"))
    operator.cutover(stage=stage, target=target, backup=backup,
                     apply=True, expected=pins)
    assert events[0:5] == ["watchdog-start", ("stop", 23), ("stop", 22),
                           ("stop", 21), "all-stopped"]
    assert events[-4:] == [("resume", 21), ("resume", 22),
                           ("resume", 23), "watchdog-cancel"]


def test_exact_pidfd_signal_checks_identity_on_both_sides(tmp_path, monkeypatch):
    record = operator.restart.Gateway(23, 123)
    events = []
    monkeypatch.setattr(operator, "_identity", lambda *_a: events.append("identity") or True)
    monkeypatch.setattr(operator.os, "pidfd_open",
                        lambda pid: events.append(("open", pid)) or 77)
    monkeypatch.setattr(operator.signal, "pidfd_send_signal",
                        lambda fd, kind: events.append(("signal", fd, kind)))
    monkeypatch.setattr(operator.os, "close", lambda fd: events.append(("close", fd)))
    operator._signal_exact(record, signal.SIGSTOP, tmp_path)
    assert events == ["identity", ("open", 23), "identity",
                      ("signal", 77, signal.SIGSTOP), ("close", 77)]


def test_missing_exact_process_is_a_hard_resume_error(tmp_path, monkeypatch):
    record = operator.restart.Gateway(23, 123)
    monkeypatch.setattr(operator, "_identity", lambda *_a: False)
    monkeypatch.setattr(operator, "_signal_exact", lambda *_a: pytest.fail("unproven signal"))
    with pytest.raises(operator.Cap64CutoverStopped, match="disappeared"):
        operator._resume_exact(record, tmp_path)


def test_zombie_after_sigcont_is_not_reported_as_resumed(tmp_path, monkeypatch):
    record = operator.quiesce.Process(21, "hymem-honcho", 121)
    monkeypatch.setattr(operator, "_identity", lambda *_a: True)
    monkeypatch.setattr(operator, "_signal_exact", lambda *_a: None)
    monkeypatch.setattr(operator, "_process_state", lambda *_a: "Z")
    with pytest.raises(operator.Cap64CutoverStopped, match="exited during resume"):
        operator._resume_exact(record, tmp_path)


def test_resume_failure_keeps_watchdog_armed(tmp_path, monkeypatch):
    stage, target, backup, pins, *_ = _fixture(tmp_path, monkeypatch)
    events = []
    class Watchdog:
        def terminate(self):
            events.append("canceled")
        def wait(self, timeout):
            events.append("waited")
    monkeypatch.setattr(operator, "_start_watchdog",
                        lambda _pins: (Watchdog(), time.monotonic()))
    def resume(record):
        if record == pins[0]:
            raise operator.Cap64CutoverStopped("process disappeared")
        events.append(("resumed", record.pid))
    monkeypatch.setattr(operator, "_resume_exact", resume)
    with pytest.raises(operator.Cap64CutoverStopped, match="watchdog remains armed"):
        operator.cutover(stage=stage, target=target, backup=backup,
                         apply=True, expected=pins)
    assert "canceled" not in events and "waited" not in events
    assert (backup / ".uniform-state").read_bytes() == b"new"


def test_watchdog_cancellation_reaps_after_terminate_or_kill():
    events = []
    class Stubborn:
        def terminate(self):
            events.append("terminate")
        def kill(self):
            events.append("kill")
        def wait(self, timeout):
            events.append(("wait", timeout))
            if events.count(("wait", 2.0)) == 1:
                raise subprocess.TimeoutExpired("watchdog", timeout)
    operator._cancel_watchdog(Stubborn())
    assert events == ["terminate", ("wait", 2.0), "kill", ("wait", 2.0)]
    class Unreapable(Stubborn):
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("watchdog", timeout)
    with pytest.raises(operator.Cap64CutoverStopped, match="exit cannot be proven"):
        operator._cancel_watchdog(Unreapable())


def test_watchdog_restores_partial_install_before_resuming(tmp_path, monkeypatch):
    stage, target, backup, pins, old_mcp, old_honcho = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "TARGET", target)
    plan = operator._plan(stage, target)
    operator._backup(plan, backup)
    (target / "hymem/config.py").write_bytes(b"VALUE = 64\n")
    (tmp_path / "mcp").write_bytes(
        b"export HYMEM_DREAM_MAX_FACTS_PER_SESSION=64\n"
        b"export HYMEM_FACTS_EXTRACTION_MAX_TOKENS=6144\n")
    events = []
    monkeypatch.setattr(operator, "BACKUP", backup)
    monkeypatch.setattr(operator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(operator, "_resume_exact",
                        lambda record: events.append(("resume", record.pid)))
    read_fd, write_fd = os.pipe()
    try:
        operator._watchdog_main(pins, write_fd)
        assert os.read(read_fd, 1) == b"R"
    finally:
        os.close(read_fd)
    assert (target / "hymem/config.py").read_bytes() == b"VALUE = 32\n"
    assert (tmp_path / "mcp").read_bytes() == old_mcp
    assert (tmp_path / "honcho").read_bytes() == old_honcho
    assert events == [("resume", 21), ("resume", 22), ("resume", 23)]


def test_watchdog_leaves_processes_stopped_for_unknown_live_bytes(tmp_path, monkeypatch):
    stage, target, backup, pins, *_ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "TARGET", target)
    operator._backup(operator._plan(stage, target), backup)
    (target / "hymem/config.py").write_bytes(b"UNKNOWN = 1\n")
    monkeypatch.setattr(operator, "BACKUP", backup)
    monkeypatch.setattr(operator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(operator, "_resume_exact", lambda _record: pytest.fail("resumed"))
    read_fd, write_fd = os.pipe()
    try:
        operator._watchdog_main(pins, write_fd)
        assert os.read(read_fd, 1) == b"R"
    finally:
        os.close(read_fd)


def test_watchdog_refuses_restore_if_old_processes_are_not_stopped(tmp_path, monkeypatch):
    stage, target, backup, pins, *_ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "TARGET", target)
    operator._backup(operator._plan(stage, target), backup)
    (target / "hymem/config.py").write_bytes(b"VALUE = 64\n")
    monkeypatch.setattr(operator, "BACKUP", backup)
    monkeypatch.setattr(operator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(operator, "_same_stopped_processes",
                        lambda _pins: (_ for _ in ()).throw(
                            operator.Cap64CutoverStopped("consumer resumed early")))
    monkeypatch.setattr(operator, "_resume_exact", lambda _record: pytest.fail("resumed"))
    read_fd, write_fd = os.pipe()
    try:
        operator._watchdog_main(pins, write_fd)
        assert os.read(read_fd, 1) == b"R"
    finally:
        os.close(read_fd)
    assert (target / "hymem/config.py").read_bytes() == b"VALUE = 64\n"


def test_absolute_watchdog_ack_deadline_refuses_before_first_stop(tmp_path, monkeypatch):
    stage, target, backup, pins, *_ = _fixture(tmp_path, monkeypatch)
    class Watchdog:
        def terminate(self):
            pass
        def wait(self, timeout):
            assert timeout == 2.0
    monkeypatch.setattr(operator, "_start_watchdog",
                        lambda _pins: (Watchdog(), time.monotonic() - 31))
    monkeypatch.setattr(operator, "_stop_exact", lambda _record: pytest.fail("signaled"))
    with pytest.raises(operator.Cap64CutoverStopped, match="absolute deadline"):
        operator.cutover(stage=stage, target=target, backup=backup,
                         apply=True, expected=pins)
