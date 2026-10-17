"""Quiesce exactly two Hermes1 code consumers for the reviewed code cutover.

Apply reads no private memory text and makes no provider call. A detached
watchdog resumes stopped processes after 45 seconds even if this operator is
killed. Do not run until the chosen cap is present in both launch scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from tools.diagnostics import hermes_four_module_cutover as modules
else:
    import hermes_four_module_cutover as modules


HONCHO_WRAPPER = Path("/home/node/.agent37/hooks/post-restart.sh")
MCP_WRAPPER = Path("/home/node/.hermes/bin/hymem-server-wrapper")
DATABASE = Path("/home/node/.hermes/hymem.sqlite")
PROC = Path("/proc")
EXPECTED_STAGE = Path("/tmp/hymem-cutover-20261017")
WATCHDOG_SECONDS = 45


class QuiesceStopped(RuntimeError):
    pass


@dataclass(frozen=True)
class Process:
    pid: int
    role: str
    starttime: int


def _command_role(argv: bytes) -> str | None:
    tokens = {item.rsplit(b"/", 1)[-1] for item in argv.split(b"\0")}
    if tokens & {b"hymem-honcho", b"hymem.honcho.app"}:
        return "hymem-honcho"
    if tokens & {b"hymem-server", b"hymem.server"}:
        return "hymem-server"
    return None


def _stat_state_starttime(path: Path) -> tuple[str, int]:
    # /proc/pid/stat field 2 is parenthesized and can itself contain spaces.
    fields = path.read_text().rsplit(") ", 1)[1].split()
    return fields[0], int(fields[19])


def _processes(proc_root: Path) -> tuple[Process, Process]:
    found: dict[str, list[Process]] = {"hymem-honcho": [], "hymem-server": []}
    for directory in proc_root.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            role = _command_role((directory / "cmdline").read_bytes())
            if role is None:
                continue
            _state, starttime = _stat_state_starttime(directory / "stat")
        except (OSError, ValueError, IndexError):
            continue
        found[role].append(Process(int(directory.name), role, starttime))
    if any(len(items) != 1 for items in found.values()):
        raise QuiesceStopped("expected exactly one Honcho and one MCP process")
    return found["hymem-honcho"][0], found["hymem-server"][0]


def _same_process(record: Process, proc_root: Path) -> bool:
    directory = proc_root / str(record.pid)
    try:
        role = _command_role((directory / "cmdline").read_bytes())
        _state, starttime = _stat_state_starttime(directory / "stat")
    except (OSError, ValueError, IndexError):
        return False
    return role == record.role and starttime == record.starttime


def _resume(record: Process, proc_root: Path) -> None:
    if _same_process(record, proc_root):
        try:
            os.kill(record.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


def _stop(record: Process, proc_root: Path) -> None:
    if not _same_process(record, proc_root):
        raise QuiesceStopped("code consumer changed before pause")
    os.kill(record.pid, signal.SIGSTOP)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _same_process(record, proc_root):
            raise QuiesceStopped("code consumer exited during pause")
        state, _start = _stat_state_starttime(proc_root / str(record.pid) / "stat")
        if state in {"T", "t"}:
            return
        time.sleep(0.05)
    raise QuiesceStopped("code consumer did not pause within two seconds")


def _cap_scripts_match(cap: int, honcho: Path, mcp: Path) -> None:
    if cap not in (16, 32):
        raise QuiesceStopped("unreviewed fact cap")
    expected = {
        mcp: f"export HYMEM_DREAM_MAX_FACTS_PER_SESSION={cap}",
        honcho: f"      HYMEM_DREAM_MAX_FACTS_PER_SESSION={cap} \\",
    }
    for path, line in expected.items():
        if not path.is_file() or path.is_symlink():
            raise QuiesceStopped("selected cap is absent from a Hermes launch script")
        cap_lines = [item for item in path.read_text().splitlines()
                     if "HYMEM_DREAM_MAX_FACTS_PER_SESSION" in item]
        if cap_lines != [line]:
            raise QuiesceStopped("selected cap is absent from a Hermes launch script")


def _dream_idle(db_path: Path) -> None:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        if conn.execute(
            "SELECT 1 FROM run_lock WHERE name='dreaming'"
        ).fetchone() is not None:
            raise QuiesceStopped("dreaming lease is held")
    finally:
        conn.close()


def _watchdog_main(records: tuple[Process, Process], ack_fd: int,
                   proc_root: Path) -> None:
    os.write(ack_fd, b"R")
    os.close(ack_fd)
    time.sleep(WATCHDOG_SECONDS)
    for record in records:
        _resume(record, proc_root)


def _start_watchdog(records: tuple[Process, Process]) -> subprocess.Popen:
    read_fd, write_fd = os.pipe()
    args = [
        sys.executable, str(Path(__file__).resolve()), "--watchdog",
        "--ack-fd", str(write_fd),
        *(f"{record.pid}:{record.role}:{record.starttime}" for record in records),
    ]
    try:
        child = subprocess.Popen(
            args, pass_fds=(write_fd,), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
        )
        os.close(write_fd)
        ready = select.select([read_fd], [], [], 3.0)[0]
        if not ready or os.read(read_fd, 1) != b"R":
            child.terminate()
            raise QuiesceStopped("independent resume watchdog did not start")
        return child
    finally:
        os.close(read_fd)
        try:
            os.close(write_fd)
        except OSError:
            pass


def quiesced_cutover(cap: int, *, stage: Path = modules.STAGE,
                     target: Path = modules.TARGET, backup: Path = modules.BACKUP,
                     honcho: Path = HONCHO_WRAPPER, mcp: Path = MCP_WRAPPER,
                     db_path: Path = DATABASE, proc_root: Path = PROC) -> dict:
    _cap_scripts_match(cap, honcho, mcp)
    modules.cutover(stage, target, backup, apply=False)
    _dream_idle(db_path)
    records = _processes(proc_root)
    watchdog = _start_watchdog(records)
    stopped: list[Process] = []
    try:
        for record in records:
            stopped.append(record)
            _stop(record, proc_root)
        _dream_idle(db_path)
        _cap_scripts_match(cap, honcho, mcp)
        result = modules.cutover(
            stage, target, backup, apply=True,
            deadline=time.monotonic() + 30.0,
        )
        result["resumed_processes"] = len(records)
        return result
    finally:
        resume_failed = False
        for record in reversed(stopped):
            try:
                _resume(record, proc_root)
            except OSError:
                resume_failed = True
        if not resume_failed:
            watchdog.terminate()
            try:
                watchdog.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        else:
            raise QuiesceStopped("a process could not be resumed; watchdog remains armed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, choices=(16, 32))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live-write", action="store_true")
    parser.add_argument("--watchdog", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ack-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("records", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.watchdog:
        if args.ack_fd is None or len(args.records) != 2:
            return 2
        try:
            records = tuple(Process(int(pid), role, int(start)) for pid, role, start
                            in (item.split(":", 2) for item in args.records))
            _watchdog_main(records, args.ack_fd, PROC)
            return 0
        except (OSError, ValueError):
            return 2
    if not args.apply or not args.allow_live_write or args.cap is None:
        parser.error("cutover requires --cap 16|32 --apply --allow-live-write")
    if os.geteuid() != 1000 or os.getegid() != 1000:
        parser.error("cutover must run as the Hermes node user (uid/gid 1000)")
    if (modules.STAGE != EXPECTED_STAGE
            or Path(__file__).resolve() != EXPECTED_STAGE / Path(__file__).name
            or Path(modules.__file__).resolve()
            != EXPECTED_STAGE / "hermes_four_module_cutover.py"):
        parser.error("quiesced operator must run from the private reviewed stage")
    try:
        print(json.dumps(quiesced_cutover(args.cap), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"stopped": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, QuiesceStopped)
                          else "unexpected cutover failure; inspect private backup"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
