"""Guards and failure cleanup for the two-process Hermes cutover."""

import os
import signal

import pytest

from tools.diagnostics import hermes_quiesced_cutover as operator


def _fake_process(root, pid, role, starttime):
    directory = root / str(pid)
    directory.mkdir()
    (directory / "cmdline").write_bytes(
        f"/home/node/hymem-env/bin/{role}".encode() + b"\0"
    )
    (directory / "stat").write_text(
        f"{pid} ({role}) S " + "0 " * 18 + f"{starttime} 0\n"
    )


def _cap_files(tmp_path, cap):
    honcho, mcp = tmp_path / "post-restart.sh", tmp_path / "wrapper"
    honcho.write_text(f"      HYMEM_DREAM_MAX_FACTS_PER_SESSION={cap} \\\n")
    mcp.write_text(f"export HYMEM_DREAM_MAX_FACTS_PER_SESSION={cap}\n")
    return honcho, mcp


def test_exact_process_identity_rejects_pid_reuse(tmp_path):
    _fake_process(tmp_path, 21, "hymem-honcho", 123)
    _fake_process(tmp_path, 22, "hymem-server", 456)
    records = operator._processes(tmp_path)
    assert {row.role for row in records} == {"hymem-honcho", "hymem-server"}
    assert operator._same_process(records[0], tmp_path)
    (tmp_path / "21" / "stat").write_text(
        "21 (hymem-honcho) S " + "0 " * 18 + "999 0\n"
    )
    assert not operator._same_process(records[0], tmp_path)


def test_stop_and_watchdog_resume_only_matching_pid(tmp_path, monkeypatch):
    _fake_process(tmp_path, 21, "hymem-honcho", 123)
    record = operator.Process(21, "hymem-honcho", 123)
    (tmp_path / "21" / "stat").write_text(
        "21 (hymem-honcho) T " + "0 " * 18 + "123 0\n"
    )
    signals = []
    monkeypatch.setattr(operator.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    operator._stop(record, tmp_path)
    operator._resume(record, tmp_path)
    assert signals == [(21, signal.SIGSTOP), (21, signal.SIGCONT)]
    (tmp_path / "21" / "stat").write_text(
        "21 (hymem-honcho) T " + "0 " * 18 + "999 0\n"
    )
    operator._resume(record, tmp_path)
    assert len(signals) == 2


def test_watchdog_acknowledges_before_delayed_resume(tmp_path, monkeypatch):
    records = (
        operator.Process(21, "hymem-honcho", 123),
        operator.Process(22, "hymem-server", 456),
    )
    resumed = []
    monkeypatch.setattr(operator.time, "sleep", lambda _: None)
    monkeypatch.setattr(operator, "_resume", lambda record, _: resumed.append(record))
    read_fd, write_fd = os.pipe()
    try:
        operator._watchdog_main(records, write_fd, tmp_path)
        assert os.read(read_fd, 1) == b"R"
        assert resumed == list(records)
    finally:
        os.close(read_fd)


def test_cap_must_be_in_both_exact_launch_lines(tmp_path):
    honcho, mcp = _cap_files(tmp_path, 16)
    operator._cap_scripts_match(16, honcho, mcp)
    with pytest.raises(operator.QuiesceStopped, match="selected cap"):
        operator._cap_scripts_match(32, honcho, mcp)
    mcp.write_text(mcp.read_text() * 2)
    with pytest.raises(operator.QuiesceStopped, match="selected cap"):
        operator._cap_scripts_match(16, honcho, mcp)
    mcp.write_text(
        "export HYMEM_DREAM_MAX_FACTS_PER_SESSION=16\n"
        "export HYMEM_DREAM_MAX_FACTS_PER_SESSION=32\n"
    )
    with pytest.raises(operator.QuiesceStopped, match="selected cap"):
        operator._cap_scripts_match(16, honcho, mcp)


def test_cutover_failure_resumes_every_paused_process_and_stops_watchdog(
    tmp_path, monkeypatch,
):
    honcho, mcp = _cap_files(tmp_path, 16)
    records = (
        operator.Process(21, "hymem-honcho", 123),
        operator.Process(22, "hymem-server", 456),
    )
    stopped, resumed, calls = [], [], []

    class Watchdog:
        terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, *, timeout):
            return 0

    watchdog = Watchdog()
    monkeypatch.setattr(operator, "_dream_idle", lambda _: None)
    monkeypatch.setattr(operator, "_processes", lambda _: records)
    monkeypatch.setattr(operator, "_start_watchdog", lambda _: watchdog)
    monkeypatch.setattr(operator, "_stop", lambda row, _: stopped.append(row))
    monkeypatch.setattr(operator, "_resume", lambda row, _: resumed.append(row))

    def fail_apply(_stage, _target, _backup, *, apply, deadline=None):
        calls.append(apply)
        if apply:
            raise RuntimeError("injected cutover failure")
        return {"applied": False}

    monkeypatch.setattr(operator.modules, "cutover", fail_apply)
    with pytest.raises(RuntimeError, match="injected"):
        operator.quiesced_cutover(
            16, stage=tmp_path / "stage", target=tmp_path / "target",
            backup=tmp_path / "backup", honcho=honcho, mcp=mcp,
            db_path=tmp_path / "db", proc_root=tmp_path / "proc",
        )
    assert calls == [False, True]
    assert stopped == list(records)
    assert resumed == list(reversed(records))
    assert watchdog.terminated
