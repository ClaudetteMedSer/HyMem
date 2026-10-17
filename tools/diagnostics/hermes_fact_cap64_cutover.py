"""One-use, five-file Hermes1 fact cap 64 / output 6144 cutover.

Preview is read-only. Apply requires the exact idle Honcho, MCP, and gateway
PID/starttime pairs; saves the cap-32 state privately; atomically installs the
three reviewed source modules and two byte-minimal launch-script edits; and
restores already installed files on failure. It never calls a provider. Apply
pauses the exact three code consumers with a private backup-lock watchdog;
if the operator dies during replacement, the watchdog restores cap-32 files
before resuming them. A locked or unprovable restore leaves them stopped for
manual recovery. The separate guarded gateway restart activates new files.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import select
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__:
    from tools.diagnostics import hermes_cap32_services_restart as prior
    from tools.diagnostics import hermes_targeted_restart as restart
    from tools.diagnostics import hermes_quiesced_cutover as quiesce
else:
    import hermes_cap32_services_restart as prior
    import hermes_targeted_restart as restart
    import hermes_quiesced_cutover as quiesce


STAGE = Path("/tmp/hymem-cap64-cutover-20261017")
TARGET = Path("/home/node/HyMem")
BACKUP = Path("/home/node/.hermes/backups/hymem-cap32-to64-20261017")
PRIOR_CODE_BACKUP = Path("/home/node/.hermes/backups/hymem-code-20261017")
MODULES = {
    "hymem/config.py": (
        "ca98f1c636a80b4da9c728dd456a8a8bbfce662c2c1dbf5889b382c929d4bbdb",
        "3c48a46e0433a6bfc516ef3c3d11b9458421d557a98498bf4b4c6322d409d4ab",
    ),
    "hymem/bootstrap.py": (
        "581568e8a4000f628e9fb6c17cfc5c0e33278016a4c287fc3ba0a6857a833cfd",
        "d8ee1ff55b48a975d308dca93bdf2eeec1a702e0265f828ad7f0668f7ba8e7bf",
    ),
    "hymem/dreaming/facts.py": (
        "36a11d562ec199c4657bd657f8781c9e6463be2ef90169f810ba77e0c516c4fb",
        "841bab44537aedf9a9a0295b3d4b4827d30f0a4827127d025d5d11221196d4ec",
    ),
}
SCRIPTS = {
    "mcp": (
        restart.MCP_WRAPPER,
        restart.CAP_HASHES[32]["mcp"],
        "31b36d272f5d0fd00574cc63d4d7f5c2407292b7e2025c9be24863c302058f36",
        b"export HYMEM_DREAM_MAX_FACTS_PER_SESSION=32\n",
        b"export HYMEM_DREAM_MAX_FACTS_PER_SESSION=64\n"
        b"export HYMEM_FACTS_EXTRACTION_MAX_TOKENS=6144\n",
    ),
    "honcho": (
        restart.HONCHO_HOOK,
        restart.CAP_HASHES[32]["honcho"],
        "c16a7df4f9dc15613511df06024dc1af07475e9f1db91c2c56871409268901bf",
        b"      HYMEM_DREAM_MAX_FACTS_PER_SESSION=32 \\\n",
        b"      HYMEM_DREAM_MAX_FACTS_PER_SESSION=64 "
        b"HYMEM_FACTS_EXTRACTION_MAX_TOKENS=6144 \\\n",
    ),
}
TOKEN_PREFIX = b"HYMEM_FACTS_EXTRACTION_MAX_TOKENS="
WATCHDOG_SECONDS = 90
PAUSE_DEADLINE_SECONDS = 30
WATCHDOG_LOCK_WAIT_SECONDS = 15


class Cap64CutoverStopped(RuntimeError):
    pass


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _owned_bytes(path: Path, mode: int) -> bytes:
    restart.modules._physical_directory_chain(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != mode
                or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
            raise Cap64CutoverStopped("cutover file ownership or mode changed")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def _prior_backups_ready(code_backup: Path = PRIOR_CODE_BACKUP) -> None:
    prior._backups_ready(prior.BACKUP)
    restart.modules._private_dir(code_backup)
    for name in ("hymem/bootstrap.py", "hymem/dreaming/facts.py"):
        saved = code_backup / name
        if _sha(_owned_bytes(saved, 0o600)) != restart.modules.FILES[name][0]:
            raise Cap64CutoverStopped("earlier private code backup changed")


def _no_token_override(record: quiesce.Process, proc_root: Path) -> None:
    if not quiesce._same_process(record, proc_root):
        raise Cap64CutoverStopped("reviewed process identity changed")
    entries = (proc_root / str(record.pid) / "environ").read_bytes().split(b"\0")
    if any(item.startswith(TOKEN_PREFIX) for item in entries):
        raise Cap64CutoverStopped("old producer already has a fact token override")


def _pins(proc_root: Path = prior.PROC) -> tuple[quiesce.Process,
                                                quiesce.Process, restart.Gateway]:
    # The earlier cap-32 *activation* preview required old cap-16 children.
    # These children now run cap 32, so inspect them directly at this stage.
    honcho, mcp = quiesce._processes(proc_root)
    gateway = restart._gateway_parent(mcp, proc_root)
    pins = (honcho, mcp, gateway)
    _same_idle_processes(pins, proc_root=proc_root)
    return pins


def _same_idle_processes(pins: tuple[quiesce.Process, quiesce.Process,
                                    restart.Gateway], *,
                         proc_root: Path = prior.PROC,
                         db_path: Path = restart.DATABASE) -> None:
    honcho, mcp, gateway = pins
    quiesce._dream_idle(db_path)
    if (quiesce._processes(proc_root) != (honcho, mcp)
            or restart._gateway_parent(mcp, proc_root) != gateway
            or prior._gateways(proc_root) != [gateway]
            or not restart._cap_in_process(honcho, 32, proc_root)
            or not restart._cap_in_process(mcp, 32, proc_root)):
        raise Cap64CutoverStopped("old cap-32 process identities changed")
    prior._gateway_entrypoint(gateway, proc_root)
    _no_token_override(honcho, proc_root)
    _no_token_override(mcp, proc_root)
    prior._status_idle(gateway, prior.STATUS)
    prior._no_notice_markers(prior.HOME)
    prior._keepalive_config_ready(prior.MCP_CONFIG)
    if not (restart._health("127.0.0.1", 8765)
            and restart._health("embedding-server", 8766)):
        raise Cap64CutoverStopped("local service health changed")


def _identity(record: quiesce.Process | restart.Gateway,
              proc_root: Path) -> bool:
    if isinstance(record, restart.Gateway):
        try:
            prior._gateway_entrypoint(record, proc_root)
            return True
        except prior.Cap32RestartStopped:
            return False
    return quiesce._same_process(record, proc_root)


def _process_state(record: quiesce.Process | restart.Gateway,
                   proc_root: Path) -> str:
    if not _identity(record, proc_root):
        raise Cap64CutoverStopped("paused process identity changed")
    state, start = quiesce._stat_state_starttime(proc_root / str(record.pid) / "stat")
    if start != record.starttime:
        raise Cap64CutoverStopped("paused process starttime changed")
    return state


def _signal_exact(record: quiesce.Process | restart.Gateway,
                  signal_number: int, proc_root: Path) -> None:
    if not _identity(record, proc_root):
        raise Cap64CutoverStopped("process identity changed before signal")
    descriptor = os.pidfd_open(record.pid)
    try:
        if not _identity(record, proc_root):
            raise Cap64CutoverStopped("process identity changed after pidfd open")
        signal.pidfd_send_signal(descriptor, signal_number)
    finally:
        os.close(descriptor)


def _stop_exact(record: quiesce.Process | restart.Gateway,
                proc_root: Path = prior.PROC) -> None:
    _signal_exact(record, signal.SIGSTOP, proc_root)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if _process_state(record, proc_root) in {"T", "t"}:
            return
        time.sleep(0.05)
    raise Cap64CutoverStopped("reviewed process did not stop within two seconds")


def _resume_exact(record: quiesce.Process | restart.Gateway,
                  proc_root: Path = prior.PROC) -> None:
    if not _identity(record, proc_root):
        raise Cap64CutoverStopped("exact process disappeared before resume")
    _signal_exact(record, signal.SIGCONT, proc_root)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        state = _process_state(record, proc_root)
        if state in {"Z", "X", "x"}:
            raise Cap64CutoverStopped("exact process exited during resume")
        if state not in {"T", "t"}:
            return
        time.sleep(0.05)
    raise Cap64CutoverStopped("exact process remained stopped after SIGCONT")


def _same_stopped_processes(pins: tuple[quiesce.Process, quiesce.Process,
                                       restart.Gateway], *,
                            proc_root: Path = prior.PROC,
                            db_path: Path = restart.DATABASE) -> None:
    honcho, mcp, gateway = pins
    quiesce._dream_idle(db_path)
    if (quiesce._processes(proc_root) != (honcho, mcp)
            or prior._gateways(proc_root) != [gateway]
            or restart._gateway_parent(mcp, proc_root) != gateway):
        raise Cap64CutoverStopped("paused process identities changed")
    for record in (gateway, mcp, honcho):
        if _process_state(record, proc_root) not in {"T", "t"}:
            raise Cap64CutoverStopped("one reviewed process is not stopped")
    prior._no_notice_markers(prior.HOME)
    prior._keepalive_config_ready(prior.MCP_CONFIG)


def _watchdog_main(pins: tuple[quiesce.Process, quiesce.Process,
                                restart.Gateway], ack_fd: int) -> None:
    os.write(ack_fd, b"R")
    os.close(ack_fd)
    time.sleep(WATCHDOG_SECONDS)
    # The operator holds this lock through rollback and resume. A hung parent
    # retains it: leave services stopped rather than expose mixed source code.
    lock_fd = _backup_lock(BACKUP)
    try:
        deadline = time.monotonic() + WATCHDOG_LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return
                time.sleep(0.2)
        marker = BACKUP / ".uniform-state"
        state = None
        if marker.exists() and not marker.is_symlink():
            declared = _owned_bytes(marker, 0o600)
            if declared in {b"old", b"new"}:
                try:
                    state = _watchdog_uniform_state(BACKUP)
                except (OSError, Cap64CutoverStopped):
                    state = None
                if state != declared.decode("ascii"):
                    state = None
        if state is None:
            _same_stopped_processes(pins)
            _restore_cap32_backup(BACKUP)
    except (OSError, Cap64CutoverStopped):
        return  # fail closed: no SIGCONT if complete old bytes are unproven
    finally:
        os.close(lock_fd)
    for record in (pins[0], pins[1], pins[2]):
        try:
            _resume_exact(record)
        except (OSError, Cap64CutoverStopped):
            pass


def _start_watchdog(pins: tuple[quiesce.Process, quiesce.Process,
                                restart.Gateway]) -> tuple[subprocess.Popen, float]:
    read_fd, write_fd = os.pipe()
    args = [
        sys.executable, str(Path(__file__).resolve()), "--watchdog",
        "--ack-fd", str(write_fd),
        f"{pins[0].pid}:{pins[0].starttime}",
        f"{pins[1].pid}:{pins[1].starttime}",
        f"{pins[2].pid}:{pins[2].starttime}",
    ]
    try:
        child = subprocess.Popen(
            args, pass_fds=(write_fd,), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
            env={"HOME": "/home/node", "PATH": "/usr/bin:/bin"},
        )
        os.close(write_fd)
        ready = select.select([read_fd], [], [], 3.0)[0]
        if not ready or os.read(read_fd, 1) != b"R":
            child.terminate()
            raise Cap64CutoverStopped("independent resume watchdog did not start")
        return child, time.monotonic()
    finally:
        os.close(read_fd)
        try:
            os.close(write_fd)
        except OSError:
            pass


def _plan(stage: Path, target: Path) -> list[tuple[str, Path, bytes, bytes, int]]:
    restart.modules._private_dir(stage)
    planned = []
    for name, (old_sha, new_sha) in MODULES.items():
        path = target / name
        old = _owned_bytes(path, 0o664)
        new = _owned_bytes(stage / name, 0o600)
        if _sha(old) != old_sha or _sha(new) != new_sha:
            raise Cap64CutoverStopped("source module differs from reviewed hash")
        compile(new, str(path), "exec")
        planned.append((name, path, old, new, 0o664))
    for role, (path, old_sha, new_sha, before, after) in SCRIPTS.items():
        old = _owned_bytes(path, 0o755)
        if (_sha(old) != old_sha or old.count(before) != 1
                or TOKEN_PREFIX in old):
            raise Cap64CutoverStopped("launch script differs from reviewed cap-32 bytes")
        new = old.replace(before, after, 1)
        if _sha(new) != new_sha:
            raise Cap64CutoverStopped("proposed launch script differs from reviewed hash")
        planned.append((role, path, old, new, 0o755))
    return planned


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _temp(path: Path, payload: bytes, mode: int) -> Path:
    fd, name = tempfile.mkstemp(prefix=".hymem-cap64-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        return Path(name)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _backup(plan: list[tuple[str, Path, bytes, bytes, int]], backup: Path) -> None:
    backup.mkdir(mode=0o700, exist_ok=False)
    restart.modules._private_dir(backup)
    for name, _path, old, _new, _mode in plan:
        destination = backup / name
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        restart.modules._private_dir(destination.parent)
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(old)
            stream.flush()
            os.fsync(stream.fileno())
        if _owned_bytes(destination, 0o600) != old:
            raise Cap64CutoverStopped("private backup verification failed")
        _fsync_dir(destination.parent)
    _fsync_dir(backup)
    lock_path = backup / ".install.lock"
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                      0o600)
    try:
        os.fsync(lock_fd)
    finally:
        os.close(lock_fd)
    _fsync_dir(backup)
    _fsync_dir(backup.parent)


def _backup_lock(backup: Path) -> int:
    restart.modules._private_dir(backup)
    path = backup / ".install.lock"
    restart.modules._physical_directory_chain(path.parent)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    info = os.fstat(descriptor)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
        os.close(descriptor)
        raise Cap64CutoverStopped("private backup lock changed")
    return descriptor


def _restore_cap32_backup(backup: Path) -> None:
    """Watchdog recovery before resume; refuse unknown live bytes."""
    restart.modules._private_dir(backup)
    planned = []
    for name, path, old_sha, new_sha, mode in _manifest():
        saved = _owned_bytes(backup / name, 0o600)
        current = _owned_bytes(path, mode)
        if _sha(saved) != old_sha or _sha(current) not in {old_sha, new_sha}:
            raise Cap64CutoverStopped("watchdog cannot prove a safe cap-32 restore")
        planned.append((path, saved, mode))
    for path, saved, mode in planned:
        if _owned_bytes(path, mode) != saved:
            restored = _temp(path, saved, mode)
            os.replace(restored, path)
            _fsync_dir(path.parent)
    for path, saved, mode in planned:
        if _owned_bytes(path, mode) != saved:
            raise Cap64CutoverStopped("watchdog cap-32 restore did not verify")


def _manifest() -> list[tuple[str, Path, str, str, int]]:
    entries = [(name, TARGET / name, old_sha, new_sha, 0o664)
               for name, (old_sha, new_sha) in MODULES.items()]
    entries.extend((role, path, old_sha, new_sha, 0o755)
                   for role, (path, old_sha, new_sha, _before, _after)
                   in SCRIPTS.items())
    return entries


def _watchdog_uniform_state(backup: Path) -> str:
    states = []
    for name, path, old_sha, new_sha, mode in _manifest():
        if _sha(_owned_bytes(backup / name, 0o600)) != old_sha:
            raise Cap64CutoverStopped("watchdog backup differs from reviewed cap 32")
        current = _sha(_owned_bytes(path, mode))
        states.append("old" if current == old_sha else "new" if current == new_sha
                      else "unknown")
    if len(set(states)) == 1 and states[0] in {"old", "new"}:
        return states[0]
    raise Cap64CutoverStopped("watchdog sees mixed or unknown files")


def _write_uniform_marker(backup: Path, state: str) -> None:
    if state not in {"old", "new"}:
        raise Cap64CutoverStopped("unreviewed uniform file state")
    path = backup / ".uniform-state"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(state.encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_dir(backup)


def _cancel_watchdog(watchdog: subprocess.Popen) -> None:
    watchdog.terminate()
    try:
        watchdog.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        watchdog.kill()
        try:
            watchdog.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            raise Cap64CutoverStopped(
                "watchdog exit cannot be proven; backup marker prevents restore"
            ) from None


def _files_uniform_state(plan: list[tuple[str, Path, bytes, bytes, int]]) -> str:
    """Return only fully old or fully new; any partial set is unsafe to resume."""
    matches = []
    for _name, path, old, new, mode in plan:
        current = _owned_bytes(path, mode)
        matches.append("old" if current == old else "new" if current == new else "unknown")
    if len(set(matches)) == 1 and matches[0] in {"old", "new"}:
        return matches[0]
    raise Cap64CutoverStopped("five-file state is mixed; watchdog restore required")


def _install(plan: list[tuple[str, Path, bytes, bytes, int]],
             pins: tuple[quiesce.Process, quiesce.Process, restart.Gateway],
             deadline: float) -> None:
    prepared = []
    installed = []
    try:
        for _name, path, _old, new, mode in plan:
            if time.monotonic() >= deadline:
                raise Cap64CutoverStopped("five-file cutover exceeded pause deadline")
            prepared.append((path, _temp(path, new, mode)))
        _same_stopped_processes(pins)
        for name, path, old, new, mode in plan:
            if time.monotonic() >= deadline:
                raise Cap64CutoverStopped("five-file cutover exceeded pause deadline")
            if _owned_bytes(path, mode) != old:
                raise Cap64CutoverStopped("file changed after backup")
            staged = next(temp for destination, temp in prepared if destination == path)
            os.replace(staged, path)
            installed.append((path, old, mode))
            _fsync_dir(path.parent)
            if _owned_bytes(path, mode) != new:
                raise Cap64CutoverStopped("installed file verification failed")
        if time.monotonic() >= deadline:
            raise Cap64CutoverStopped("five-file cutover exceeded pause deadline")
        _same_stopped_processes(pins)
    except BaseException:
        for path, old, mode in reversed(installed):
            restored = _temp(path, old, mode)
            os.replace(restored, path)
            _fsync_dir(path.parent)
        raise
    finally:
        for _path, staged in prepared:
            staged.unlink(missing_ok=True)


def cutover(*, stage: Path = STAGE, target: Path = TARGET,
            backup: Path = BACKUP, apply: bool = False,
            expected: tuple[quiesce.Process, quiesce.Process,
                            restart.Gateway] | None = None) -> dict:
    restart.modules._private_dir(backup.parent)
    if backup.exists() or backup.is_symlink():
        raise Cap64CutoverStopped("one-use cap-32 backup directory already exists")
    _prior_backups_ready()
    pins = _pins()
    plan = _plan(stage, target)
    result = {
        "applied": False, "old_cap": 32, "new_cap": 64,
        "new_fact_tokens": 6144,
        "old_sha256": {name: _sha(old) for name, _path, old, _new, _mode in plan},
        "new_sha256": {name: _sha(new) for name, _path, _old, new, _mode in plan},
        "honcho": {"pid": pins[0].pid, "starttime": pins[0].starttime},
        "mcp": {"pid": pins[1].pid, "starttime": pins[1].starttime},
        "gateway": {"pid": pins[2].pid, "starttime": pins[2].starttime},
    }
    if not apply:
        return result
    if pins != expected:
        raise Cap64CutoverStopped("process PID/starttime differs from reviewed preview")
    _same_idle_processes(pins)
    _backup(plan, backup)
    _same_idle_processes(pins)
    lock_fd = _backup_lock(backup)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    stopped: list[quiesce.Process | restart.Gateway] = []
    try:
        watchdog, acknowledged_at = _start_watchdog(pins)
        deadline = acknowledged_at + PAUSE_DEADLINE_SECONDS
        installed_complete = False
        # Freeze the gateway first so it cannot create an MCP child while
        # the existing one and Honcho are paused for the multi-file replace.
        try:
            for record in (pins[2], pins[1], pins[0]):
                if time.monotonic() >= deadline:
                    raise Cap64CutoverStopped("process pause exceeded absolute deadline")
                stopped.append(record)
                _stop_exact(record)
            _same_stopped_processes(pins)
            _install(plan, pins, deadline)
            installed_complete = True
        finally:
            try:
                state = _files_uniform_state(plan)
                safe_to_resume = state == ("new" if installed_complete else "old")
            except (OSError, Cap64CutoverStopped):
                safe_to_resume = False
            if not safe_to_resume:
                # The watchdog acquires the released lock, restores all five
                # exact cap-32 bytes, and only then issues SIGCONT.
                raise Cap64CutoverStopped(
                    "five-file state is unverified; watchdog restore remains armed"
                )
            _write_uniform_marker(backup, state)
            resume_failed = False
            for record in reversed(stopped):
                try:
                    _resume_exact(record)
                except (OSError, Cap64CutoverStopped):
                    resume_failed = True
            if not resume_failed:
                _cancel_watchdog(watchdog)
            else:
                raise Cap64CutoverStopped("process resume failed; watchdog remains armed")
    finally:
        os.close(lock_fd)
    result["applied"] = True
    result["resumed_processes"] = len(stopped)
    return result


def _pair(value: str) -> tuple[int, int]:
    try:
        pid, start = (int(item) for item in value.split(":", 1))
        if pid < 1 or start < 1:
            raise ValueError
        return pid, start
    except ValueError:
        raise argparse.ArgumentTypeError("expected PID:STARTTIME") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live-write", action="store_true")
    parser.add_argument("--expected-honcho")
    parser.add_argument("--expected-mcp")
    parser.add_argument("--expected-gateway")
    parser.add_argument("--watchdog", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ack-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("records", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.watchdog:
        if args.ack_fd is None or len(args.records) != 3:
            return 2
        try:
            hp, hs = _pair(args.records[0])
            mp, ms = _pair(args.records[1])
            gp, gs = _pair(args.records[2])
            pins = (quiesce.Process(hp, "hymem-honcho", hs),
                    quiesce.Process(mp, "hymem-server", ms), restart.Gateway(gp, gs))
            restart.modules._private_dir(STAGE)
            if Path(__file__).resolve() != STAGE / Path(__file__).name:
                return 2
            _watchdog_main(pins, args.ack_fd)
            return 0
        except (OSError, ValueError, Cap64CutoverStopped,
                restart.modules.CutoverStopped):
            return 2
    if os.geteuid() != 1000 or os.getegid() != 1000:
        parser.error("run as Hermes node user (uid/gid 1000)")
    if (Path(__file__).resolve() != STAGE / Path(__file__).name
            or Path(prior.__file__).resolve() != STAGE / "hermes_cap32_services_restart.py"
            or Path(restart.__file__).resolve() != STAGE / "hermes_targeted_restart.py"
            or Path(quiesce.__file__).resolve() != STAGE / "hermes_quiesced_cutover.py"):
        parser.error("operator must run from private reviewed stage")
    if args.apply and (not args.allow_live_write or not args.expected_honcho
                       or not args.expected_mcp or not args.expected_gateway):
        parser.error("apply requires live-write flag and all three PID:STARTTIME pairs")
    try:
        expected = None
        if args.apply:
            hp, hs = _pair(args.expected_honcho)
            mp, ms = _pair(args.expected_mcp)
            gp, gs = _pair(args.expected_gateway)
            expected = (quiesce.Process(hp, "hymem-honcho", hs),
                        quiesce.Process(mp, "hymem-server", ms), restart.Gateway(gp, gs))
        print(json.dumps(cutover(apply=args.apply, expected=expected), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"stopped": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, (
                              Cap64CutoverStopped, prior.Cap32RestartStopped,
                              restart.RestartStopped, quiesce.QuiesceStopped,
                              restart.modules.CutoverStopped))
                          else "unexpected cutover failure; inspect private backup"},
                         sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
