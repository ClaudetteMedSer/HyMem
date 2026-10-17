"""Guarded Hermes1 rollback after a successful code/cap cutover and failed restart.

Preview is read-only. Apply pins the gateway and any child processes, pauses
them with an independent 45-second resume watchdog, restores all four modules
and two launch scripts from exact private backups, then relaunches the old
Honcho and MCP services. No DB or private memory text is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from tools.diagnostics import hermes_fact_cap_cutover as cap_tool
    from tools.diagnostics import hermes_four_module_cutover as modules
    from tools.diagnostics import hermes_quiesced_cutover as quiesce
    from tools.diagnostics import hermes_targeted_restart as restart
else:
    import hermes_fact_cap_cutover as cap_tool
    import hermes_four_module_cutover as modules
    import hermes_quiesced_cutover as quiesce
    import hermes_targeted_restart as restart


STAGE = Path("/tmp/hymem-cutover-20261017")
WATCHDOG_SECONDS = 45
RESTORE_DEADLINE_SECONDS = 30


class RollbackStopped(RuntimeError):
    pass


@dataclass(frozen=True)
class Pin:
    pid: int
    role: str
    starttime: int


@dataclass(frozen=True)
class Replacement:
    path: Path
    current: bytes
    saved: bytes
    mode: int


def _physical_private(path: Path) -> None:
    modules._private_dir(path)


def _private_backup(path: Path, expected_sha: str) -> bytes:
    modules._private_dir(path.parent)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.geteuid() or info.st_gid != os.getegid()
            or stat.S_IMODE(info.st_mode) != 0o600):
        raise RollbackStopped("reviewed private backup is missing or changed")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise RollbackStopped("reviewed private backup hash differs")
    return payload


def _files(cap: int, *, target: Path, mcp_wrapper: Path, honcho_hook: Path,
           module_backup: Path, cap_backup: Path) -> list[Replacement]:
    if cap not in (16, 32):
        raise RollbackStopped("unreviewed fact cap")
    _physical_private(module_backup)
    _physical_private(cap_backup)
    planned = []
    for name, (old_sha, new_sha) in modules.FILES.items():
        path = target / name
        info = modules._regular_owned(path)
        current = path.read_bytes()
        if hashlib.sha256(current).hexdigest() != new_sha:
            raise RollbackStopped("deployed module differs from reviewed new code")
        saved = _private_backup(module_backup / name, old_sha)
        planned.append(Replacement(path, current, saved, stat.S_IMODE(info.st_mode)))
    for role, path in (("mcp", mcp_wrapper), ("honcho", honcho_hook)):
        current, mode = cap_tool._script(path)
        if mode != 0o755 or hashlib.sha256(current).hexdigest() != restart.CAP_HASHES[cap][role]:
            raise RollbackStopped("launch script differs from selected cap")
        saved = _private_backup(cap_backup / f"{role}.before", cap_tool.EXPECTED_OLD[role])
        planned.append(Replacement(path, current, saved, mode))
    return planned


def _restore_files(planned: list[Replacement], *, deadline: float) -> None:
    staged: list[tuple[Path, Path]] = []
    undo: dict[Path, Path] = {}
    installed: list[Replacement] = []

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise RollbackStopped("bounded file restore deadline elapsed")

    try:
        for item in planned:
            check_deadline()
            saved = modules._stage_replacement(item.path, item.saved, item.mode)
            staged.append((item.path, saved))
            check_deadline()
            current = modules._stage_replacement(item.path, item.current, item.mode)
            undo[item.path] = current
            check_deadline()
        for item in planned:
            check_deadline()
            info = modules._regular_owned(item.path)
            if stat.S_IMODE(info.st_mode) != item.mode:
                raise RollbackStopped("deployed file mode changed after rollback preview")
            if hashlib.sha256(item.path.read_bytes()).digest() != hashlib.sha256(item.current).digest():
                raise RollbackStopped("deployed file changed after rollback preview")
            check_deadline()
            staged_path = next(temp for path, temp in staged if path == item.path)
            os.replace(staged_path, item.path)
            installed.append(item)
            modules._fsync_dir(item.path.parent)
            check_deadline()
        for item in planned:
            check_deadline()
            if hashlib.sha256(item.path.read_bytes()).digest() != hashlib.sha256(item.saved).digest():
                raise RollbackStopped("restored file verification failed")
            check_deadline()
    except BaseException:
        for item in reversed(installed):
            os.replace(undo[item.path], item.path)
            modules._fsync_dir(item.path.parent)
        raise
    finally:
        for _path, temporary in staged:
            temporary.unlink(missing_ok=True)
        for temporary in undo.values():
            temporary.unlink(missing_ok=True)


def _gateway(proc_root: Path) -> restart.Gateway:
    found = []
    for directory in proc_root.iterdir():
        if directory.name.isdecimal():
            try:
                found.append(restart._gateway_identity(int(directory.name), proc_root))
            except restart.RestartStopped:
                pass
    if len(found) != 1:
        raise RollbackStopped("expected exactly one Hermes gateway")
    return found[0]


def _same(pin: Pin, proc_root: Path) -> bool:
    if pin.role == "gateway":
        try:
            return restart._gateway_identity(pin.pid, proc_root) == restart.Gateway(
                pin.pid, pin.starttime
            )
        except restart.RestartStopped:
            return False
    return quiesce._same_process(
        quiesce.Process(pin.pid, pin.role, pin.starttime), proc_root
    )


def _processes(cap: int, proc_root: Path) -> tuple[restart.Gateway, tuple[Pin, ...]]:
    gateway = _gateway(proc_root)
    children = []
    for role in ("hymem-honcho", "hymem-server"):
        rows = restart._role_processes(role, proc_root)
        if len(rows) > 1:
            raise RollbackStopped("multiple HyMem child processes exist")
        if rows:
            record = rows[0]
            if not (_cap_absent(record, proc_root)
                    or restart._cap_in_process(record, cap, proc_root)):
                raise RollbackStopped("HyMem child has an unreviewed fact cap")
            if role == "hymem-server" and restart._gateway_parent(record, proc_root) != gateway:
                raise RollbackStopped("MCP child has an unexpected gateway parent")
            children.append(Pin(record.pid, record.role, record.starttime))
    return gateway, tuple(children)


def _stop(pin: Pin, proc_root: Path) -> None:
    if not _same(pin, proc_root):
        raise RollbackStopped("process identity changed before rollback pause")
    fd = os.pidfd_open(pin.pid)
    try:
        if not _same(pin, proc_root):
            raise RollbackStopped("process identity changed before rollback pause")
        signal.pidfd_send_signal(fd, signal.SIGSTOP)
    finally:
        os.close(fd)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _same(pin, proc_root):
            raise RollbackStopped("process changed during rollback pause")
        state, _start = quiesce._stat_state_starttime(
            proc_root / str(pin.pid) / "stat"
        )
        if state in {"T", "t"}:
            return
        time.sleep(0.05)
    raise RollbackStopped("process did not pause within two seconds")


def _resume(pin: Pin, proc_root: Path) -> None:
    if not _same(pin, proc_root):
        return
    try:
        fd = os.pidfd_open(pin.pid)
    except ProcessLookupError:
        return
    try:
        if _same(pin, proc_root):
            try:
                signal.pidfd_send_signal(fd, signal.SIGCONT)
            except ProcessLookupError:
                pass
    finally:
        os.close(fd)


def _watchdog_main(pins: tuple[Pin, ...], ack_fd: int, proc_root: Path) -> None:
    os.write(ack_fd, b"R")
    os.close(ack_fd)
    time.sleep(WATCHDOG_SECONDS)
    for pin in pins:
        _resume(pin, proc_root)


def _watchdog(pins: tuple[Pin, ...]) -> subprocess.Popen:
    read_fd, write_fd = os.pipe()
    command = [sys.executable, str(Path(__file__).resolve()),
               "--watchdog", "--ack-fd", str(write_fd),
               *(f"{p.pid}:{p.role}:{p.starttime}" for p in pins)]
    try:
        child = subprocess.Popen(
            command, pass_fds=(write_fd,), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
        )
        os.close(write_fd)
        if not select.select([read_fd], [], [], 3.0)[0] or os.read(read_fd, 1) != b"R":
            child.terminate()
            raise RollbackStopped("independent rollback watchdog did not start")
        return child
    finally:
        os.close(read_fd)
        try:
            os.close(write_fd)
        except OSError:
            pass


def _cap_absent(record: quiesce.Process, proc_root: Path) -> bool:
    if not quiesce._same_process(record, proc_root):
        return False
    try:
        entries = (proc_root / str(record.pid) / "environ").read_bytes().split(b"\0")
    except OSError:
        return False
    return not any(item.startswith(b"HYMEM_DREAM_MAX_FACTS_PER_SESSION=")
                   for item in entries)


def _recover_services(gateway: restart.Gateway, children: tuple[Pin, ...], *,
                      honcho_hook: Path, env_file: Path, target: Path,
                      db_path: Path, proc_root: Path) -> dict:
    old = {item.role: item for item in children}
    old_honcho = old.get("hymem-honcho")
    old_mcp = old.get("hymem-server")
    if old_honcho is not None:
        record = quiesce.Process(old_honcho.pid, old_honcho.role, old_honcho.starttime)
        if restart._role_processes("hymem-honcho", proc_root) != [record]:
            raise RollbackStopped("Honcho changed before recovery")
        quiesce._dream_idle(db_path)
        restart._terminate_exact(record, proc_root)
        restart._wait_until(lambda: restart._exited(record, proc_root), 10.0,
                            "old Honcho did not exit")
    restart._wait_until(lambda: restart._port_closed(8765), 10.0,
                        "Honcho port remained occupied")
    block = restart._block(honcho_hook)
    new_honcho: quiesce.Process | None = None
    for _attempt in range(3):
        try:
            restart._launch_honcho(block, env_file=env_file, cwd=target)
        except (OSError, subprocess.SubprocessError):
            pass

        def honcho_ready() -> bool:
            nonlocal new_honcho
            rows = restart._role_processes("hymem-honcho", proc_root)
            new_honcho = None
            if (len(rows) == 1 and _cap_absent(rows[0], proc_root)
                    and restart._health("127.0.0.1", 8765)):
                new_honcho = rows[0]
            return new_honcho is not None

        try:
            restart._wait_until(honcho_ready, 10.0,
                                "restored Honcho did not become healthy")
            break
        except restart.RestartStopped:
            if restart._role_processes("hymem-honcho", proc_root):
                raise RollbackStopped("unhealthy Honcho replacement requires manual recovery")
            restart._wait_until(lambda: restart._port_closed(8765), 10.0,
                                "Honcho port remained occupied during recovery")
    if new_honcho is None:
        raise RollbackStopped("three restored Honcho launch attempts failed")
    if old_mcp is not None:
        record = quiesce.Process(old_mcp.pid, old_mcp.role, old_mcp.starttime)
        if (restart._role_processes("hymem-server", proc_root) != [record]
                or restart._gateway_parent(record, proc_root) != gateway):
            raise RollbackStopped("MCP child changed before recovery")
        quiesce._dream_idle(db_path)
        restart._terminate_exact(record, proc_root)
        restart._wait_until(lambda: restart._exited(record, proc_root), 10.0,
                            "old MCP child did not exit")
    new_mcp: quiesce.Process | None = None

    def mcp_started() -> bool:
        nonlocal new_mcp
        rows = restart._role_processes("hymem-server", proc_root)
        if (len(rows) == 1 and _cap_absent(rows[0], proc_root)
                and restart._gateway_parent(rows[0], proc_root) == gateway
                and (old_mcp is None or rows[0] != quiesce.Process(
                    old_mcp.pid, old_mcp.role, old_mcp.starttime))):
            new_mcp = rows[0]
            return True
        return False

    restart._wait_until(mcp_started, 30.0,
                        "restored gateway MCP child did not appear")
    assert new_mcp is not None
    first_seen = time.monotonic()
    restart._mcp_protocol_probe()
    deadline = first_seen + restart.MCP_STABLE_SECONDS
    while True:
        if (restart._role_processes("hymem-server", proc_root) != [new_mcp]
                or restart._gateway_parent(new_mcp, proc_root) != gateway
                or not _cap_absent(new_mcp, proc_root)):
            raise RollbackStopped("restored gateway MCP child changed during observation")
        if time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    if (not restart._health("127.0.0.1", 8765)
            or not restart._health("embedding-server", 8766)
            or not quiesce._same_process(new_honcho, proc_root)):
        raise RollbackStopped("restored service postflight health failed")
    return {"honcho_health": True, "embedding_health": True,
            "mcp_protocol_probe_passed": True, "gateway_child_stable": True,
            "honcho_pid": new_honcho.pid, "mcp_pid": new_mcp.pid}


def preview(cap: int, *, target: Path = modules.TARGET,
            mcp_wrapper: Path = cap_tool.MCP, honcho_hook: Path = cap_tool.HONCHO,
            module_backup: Path = modules.BACKUP, cap_backup: Path = cap_tool.BACKUP,
            env_file: Path = restart.HERMES_ENV,
            db_path: Path = restart.DATABASE, proc_root: Path = restart.PROC) -> dict:
    planned = _files(cap, target=target, mcp_wrapper=mcp_wrapper,
                     honcho_hook=honcho_hook, module_backup=module_backup,
                     cap_backup=cap_backup)
    restart._regular_owned(env_file, 0o600)
    quiesce._dream_idle(db_path)
    gateway, children = _processes(cap, proc_root)
    roles = {item.role: item for item in children}
    return {
        "cap": cap, "ready": True, "restored": False,
        "files": len(planned),
        "gateway": {"pid": gateway.pid, "starttime": gateway.starttime},
        "honcho": (None if "hymem-honcho" not in roles else
                   {"pid": roles["hymem-honcho"].pid,
                    "starttime": roles["hymem-honcho"].starttime}),
        "mcp": (None if "hymem-server" not in roles else
                {"pid": roles["hymem-server"].pid,
                 "starttime": roles["hymem-server"].starttime}),
    }


def rollback(cap: int, expected_gateway: restart.Gateway,
             expected_children: tuple[Pin, ...], *, target: Path = modules.TARGET,
             mcp_wrapper: Path = cap_tool.MCP, honcho_hook: Path = cap_tool.HONCHO,
             module_backup: Path = modules.BACKUP, cap_backup: Path = cap_tool.BACKUP,
             env_file: Path = restart.HERMES_ENV,
             db_path: Path = restart.DATABASE, proc_root: Path = restart.PROC) -> dict:
    planned = _files(cap, target=target, mcp_wrapper=mcp_wrapper,
                     honcho_hook=honcho_hook, module_backup=module_backup,
                     cap_backup=cap_backup)
    restart._regular_owned(env_file, 0o600)
    quiesce._dream_idle(db_path)
    gateway, children = _processes(cap, proc_root)
    if gateway != expected_gateway or children != expected_children:
        raise RollbackStopped("process identities differ from rollback preview")
    pins = (Pin(gateway.pid, "gateway", gateway.starttime), *children)
    # The file transaction's deadline starts before watchdog startup and leaves
    # fifteen seconds for a bounded undo before its independent SIGCONT.
    restore_deadline = time.monotonic() + RESTORE_DEADLINE_SECONDS
    watchdog = _watchdog(pins)
    stopped: list[Pin] = []
    try:
        for pin in pins:
            stopped.append(pin)
            _stop(pin, proc_root)
        if _processes(cap, proc_root) != (gateway, children):
            raise RollbackStopped("process set changed during rollback pause")
        quiesce._dream_idle(db_path)
        _restore_files(planned, deadline=restore_deadline)
    finally:
        resume_failed = False
        for pin in reversed(stopped):
            try:
                _resume(pin, proc_root)
            except OSError:
                resume_failed = True
        if not resume_failed:
            watchdog.terminate()
            try:
                watchdog.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        else:
            raise RollbackStopped("process resume failed; watchdog remains armed")
    # The six files are now the exact old reviewed versions; subsequent service
    # recovery may fail independently, but never leaves mixed on-disk versions.
    try:
        result = _recover_services(
            gateway, children, honcho_hook=honcho_hook, env_file=env_file,
            target=target, db_path=db_path, proc_root=proc_root,
        )
    except (RollbackStopped, restart.RestartStopped) as exc:
        raise RollbackStopped(f"reviewed files restored; service recovery incomplete: {exc}") from None
    return {"cap": cap, "restored": True, "files": len(planned), **result}


def _parse_pair(value: str, role: str) -> Pin:
    try:
        pid, start = (int(part) for part in value.split(":", 1))
        if pid < 1 or start < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("expected PID:STARTTIME or none") from None
    return Pin(pid, role, start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, choices=(16, 32))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live-rollback", action="store_true")
    parser.add_argument("--expected-gateway")
    parser.add_argument("--expected-honcho")
    parser.add_argument("--expected-mcp")
    parser.add_argument("--watchdog", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ack-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("records", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.watchdog:
        if args.ack_fd is None or not 1 <= len(args.records) <= 3:
            return 2
        try:
            pins = tuple(Pin(int(pid), role, int(start)) for pid, role, start
                         in (value.split(":", 2) for value in args.records))
            _watchdog_main(pins, args.ack_fd, restart.PROC)
            return 0
        except (OSError, ValueError):
            return 2
    if os.geteuid() != 1000 or os.getegid() != 1000:
        parser.error("rollback must run as Hermes node user (uid/gid 1000)")
    if args.cap is None:
        parser.error("rollback preview/apply requires --cap 16|32")
    siblings = ("hermes_fact_cap_cutover.py", "hermes_four_module_cutover.py",
                "hermes_quiesced_cutover.py", "hermes_targeted_restart.py")
    if (Path(__file__).resolve() != STAGE / Path(__file__).name
            or any(Path(module.__file__).resolve() != STAGE / name
                   for module, name in zip((cap_tool, modules, quiesce, restart), siblings))):
        parser.error("rollback operator must run from the private reviewed stage")
    try:
        modules._private_dir(STAGE)
    except (OSError, modules.CutoverStopped):
        parser.error("rollback stage must be an owned private directory")
    if args.apply and (not args.allow_live_rollback
                       or not args.expected_gateway
                       or args.expected_honcho is None
                       or args.expected_mcp is None):
        parser.error("rollback requires explicit flag and previewed process identities")
    try:
        if args.apply:
            gateway = _parse_pair(args.expected_gateway, "gateway")
            children = tuple(
                _parse_pair(value, role)
                for role, value in (("hymem-honcho", args.expected_honcho),
                                    ("hymem-server", args.expected_mcp))
                if value != "none"
            )
            result = rollback(
                args.cap, restart.Gateway(gateway.pid, gateway.starttime), children
            )
        else:
            result = preview(args.cap)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"stopped": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, RollbackStopped)
                          else "unexpected rollback failure; inspect private backups and service health"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
