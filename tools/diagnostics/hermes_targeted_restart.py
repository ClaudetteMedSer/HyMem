"""Guarded restart of exactly the Hermes1 Honcho and MCP child processes.

Run only after the reviewed cap and code cutovers. Preview reports process
identities without signals. Apply requires those exact PID/starttime pairs,
rechecks the deployed file hashes, and never prints credentials or source text.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shlex
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from tools.diagnostics import hermes_four_module_cutover as modules
    from tools.diagnostics import hermes_quiesced_cutover as quiesce
else:
    import hermes_four_module_cutover as modules
    import hermes_quiesced_cutover as quiesce


STAGE = Path("/tmp/hymem-cutover-20261017")
HONCHO_HOOK = Path("/home/node/.agent37/hooks/post-restart.sh")
MCP_WRAPPER = Path("/home/node/.hermes/bin/hymem-server-wrapper")
HERMES_ENV = Path("/home/node/.hermes/.env")
DATABASE = Path("/home/node/.hermes/hymem.sqlite")
PROC = Path("/proc")
TARGET = Path("/home/node/HyMem")
CAP_HASHES = {
    16: {
        "honcho": "f8c71749393a9d7735daa85bb413601433af0a1c4299b56771568874d6d0dbd7",
        "mcp": "0a65d39216f9e3aa18f1c1760e2082066ac643640d882340f00cdf992dd9f335",
    },
    32: {
        "honcho": "3bd9dd44611613208c8351559a28e3fdc2462cf745609e61f4b9c5dcab38204d",
        "mcp": "98d2b20b26e2f2700d11e66013ecb960c8292e55259ca036bb484d018cb9779a",
    },
}
BLOCK_START = b"  (cd /home/node/HyMem && nohup env \\\n"
BLOCK_END = (
    b"      /home/node/hymem-env/bin/hymem-honcho "
    b">> /tmp/dev-8765.log 2>&1 &)\n"
)
ENV_SOURCE = b"set -a; source /home/node/.hermes/.env; set +a"
MCP_HANDSHAKE_SECONDS = 60
MCP_STABLE_SECONDS = MCP_HANDSHAKE_SECONDS + 5
MCP_PROBE_TIMEOUT_SECONDS = 45


class RestartStopped(RuntimeError):
    pass


@dataclass(frozen=True)
class Gateway:
    pid: int
    starttime: int


def _regular_owned(path: Path, mode: int) -> None:
    modules._physical_directory_chain(path.parent)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != mode
            or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
        raise RestartStopped("reviewed launch file is not an owned regular file")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _block(hook: Path) -> bytes:
    content = hook.read_bytes()
    lines = content.splitlines(keepends=True)
    if lines.count(BLOCK_START) != 1 or lines.count(BLOCK_END) != 1:
        raise RestartStopped("reviewed Honcho launch block is absent")
    first, last = lines.index(BLOCK_START), lines.index(BLOCK_END)
    if last <= first or last - first not in (15, 16):
        raise RestartStopped("reviewed Honcho launch block shape changed")
    if content.count(ENV_SOURCE) != 1:
        raise RestartStopped("reviewed environment source command is absent")
    return b"".join(lines[first:last + 1])


def _preflight(cap: int, *, honcho_hook: Path, mcp_wrapper: Path,
               target: Path, env_file: Path, db_path: Path) -> bytes:
    if cap not in CAP_HASHES:
        raise RestartStopped("unreviewed fact cap")
    for role, path in (("honcho", honcho_hook), ("mcp", mcp_wrapper)):
        _regular_owned(path, 0o755)
        if _digest(path) != CAP_HASHES[cap][role]:
            raise RestartStopped("launch script differs from selected cap")
    _regular_owned(env_file, 0o600)  # Existence and mode only; never read here.
    for name, (_old_sha, new_sha) in modules.FILES.items():
        path = target / name
        modules._regular_owned(path)
        if _digest(path) != new_sha:
            raise RestartStopped("HyMem code differs from reviewed cutover")
    quiesce._dream_idle(db_path)
    block = _block(honcho_hook)
    try:
        result = subprocess.run(
            ["/bin/bash", "-n"], input=_honcho_command(block, env_file),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=3.0, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise RestartStopped("reviewed Honcho syntax check failed") from None
    if result.returncode != 0:
        raise RestartStopped("reviewed Honcho syntax check failed")
    return block


def _role_processes(role: str, proc_root: Path) -> list[quiesce.Process]:
    found = []
    for directory in proc_root.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            seen_role = quiesce._command_role((directory / "cmdline").read_bytes())
            if seen_role != role:
                continue
            state, start = quiesce._stat_state_starttime(directory / "stat")
            if state in {"Z", "X", "x"}:
                continue
            found.append(quiesce.Process(int(directory.name), role, start))
        except (OSError, ValueError, IndexError):
            continue
    return found


def _gateway_identity(pid: int, proc_root: Path) -> Gateway:
    directory = proc_root / str(pid)
    try:
        argv = [token for token in (directory / "cmdline").read_bytes().split(b"\0") if token]
        state, start = quiesce._stat_state_starttime(directory / "stat")
    except (OSError, ValueError, IndexError):
        raise RestartStopped("gateway parent identity is unavailable") from None
    if (len(argv) < 4 or argv[1].rsplit(b"/", 1)[-1] != b"hermes"
            or argv[2:4] != [b"gateway", b"run"]
            or state in {"Z", "X", "x"}):
        raise RestartStopped("MCP parent is not the reviewed Hermes gateway")
    return Gateway(pid, start)


def _gateway_parent(record: quiesce.Process, proc_root: Path) -> Gateway:
    directory = proc_root / str(record.pid)
    if not quiesce._same_process(record, proc_root):
        raise RestartStopped("MCP child identity changed")
    try:
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        parent_pid = int(fields[1])
    except (OSError, ValueError, IndexError):
        raise RestartStopped("MCP parent PID is unavailable") from None
    return _gateway_identity(parent_pid, proc_root)


def _cap_in_process(record: quiesce.Process, cap: int, proc_root: Path) -> bool:
    if not quiesce._same_process(record, proc_root):
        return False
    try:
        entries = (proc_root / str(record.pid) / "environ").read_bytes().split(b"\0")
    except OSError:
        return False
    prefix = b"HYMEM_DREAM_MAX_FACTS_PER_SESSION="
    return [item for item in entries if item.startswith(prefix)] == [
        prefix + str(cap).encode()
    ]


def _health(host: str, port: int) -> bool:
    connection = http.client.HTTPConnection(host, port, timeout=2.0)
    try:
        connection.request("GET", "/health")
        return connection.getresponse().status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _port_closed(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.5)
        return client.connect_ex(("127.0.0.1", port)) != 0


def _wait_until(predicate, seconds: float, reason: str) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise RestartStopped(reason)


def _terminate_exact(record: quiesce.Process, proc_root: Path) -> None:
    if not quiesce._same_process(record, proc_root):
        raise RestartStopped("reviewed child process changed before termination")
    fd = os.pidfd_open(record.pid)
    try:
        if not quiesce._same_process(record, proc_root):
            raise RestartStopped("reviewed child process changed before termination")
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)


def _exited(record: quiesce.Process, proc_root: Path) -> bool:
    if not quiesce._same_process(record, proc_root):
        return True
    try:
        state, _start = quiesce._stat_state_starttime(
            proc_root / str(record.pid) / "stat"
        )
    except (OSError, ValueError, IndexError):
        return True
    return state in {"Z", "X", "x"}


def _new_process(role: str, old: quiesce.Process, cap: int,
                 proc_root: Path) -> quiesce.Process | None:
    records = _role_processes(role, proc_root)
    if len(records) != 1 or records[0] == old:
        return None
    return records[0] if _cap_in_process(records[0], cap, proc_root) else None


def _recover_honcho(block: bytes, old: quiesce.Process, cap: int, *,
                    env_file: Path, target: Path, proc_root: Path) -> quiesce.Process:
    """Retry the pinned launch only while no replacement process exists."""
    for _attempt in range(3):
        if not _port_closed(8765):
            raise RestartStopped("Honcho port is occupied before recovery")
        try:
            _launch_honcho(block, env_file=env_file, cwd=target)
        except (OSError, subprocess.SubprocessError):
            pass
        candidate: quiesce.Process | None = None

        def ready() -> bool:
            nonlocal candidate
            candidate = _new_process("hymem-honcho", old, cap, proc_root)
            return candidate is not None and _health("127.0.0.1", 8765)

        try:
            _wait_until(ready, 10.0, "Honcho attempt did not become healthy")
            assert candidate is not None
            return candidate
        except RestartStopped:
            found = _role_processes("hymem-honcho", proc_root)
            if found:
                raise RestartStopped(
                    "Honcho replacement exists but is unhealthy; "
                    "manual recovery required"
                )
            _wait_until(lambda: _port_closed(8765), 10.0,
                        "Honcho port remained occupied during recovery")
    raise RestartStopped("three reviewed Honcho launch attempts failed")


def _mcp_protocol_probe() -> None:
    """A separate configured MCP initialize/list-tools probe; no tool is invoked."""
    try:
        result = subprocess.run(
            ["/home/node/.local/bin/hermes", "mcp", "test", "hymem"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=MCP_PROBE_TIMEOUT_SECONDS,
            check=False, cwd="/home/node",
            env={"HOME": "/home/node", "USER": "node",
                 "HERMES_HOME": "/home/node/.hermes",
                 "PATH": "/home/node/.local/bin:/usr/local/bin:/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError):
        raise RestartStopped("bounded MCP protocol probe failed") from None
    if result.returncode != 0:
        raise RestartStopped("bounded MCP protocol probe failed")


def _observe_gateway_child(record: quiesce.Process, cap: int,
                           gateway: Gateway, proc_root: Path,
                           first_seen: float) -> None:
    """Require one gateway-owned child to survive beyond initialize's 60s bound."""
    deadline = first_seen + MCP_STABLE_SECONDS
    while True:
        if (_role_processes("hymem-server", proc_root) != [record]
                or _gateway_parent(record, proc_root) != gateway
                or not _cap_in_process(record, cap, proc_root)):
            raise RestartStopped("gateway MCP child changed during stability check")
        if time.monotonic() >= deadline:
            return
        time.sleep(0.5)


def _honcho_command(block: bytes, env_file: Path) -> bytes:
    source = f"set -a; source {shlex.quote(str(env_file))}; set +a".encode()
    return b"set -euo pipefail\n" + source + b"\n" + block


def _launch_honcho(block: bytes, *, env_file: Path, cwd: Path) -> None:
    # The reviewed hook sources its own private env file. Python never reads it.
    command = _honcho_command(block, env_file)
    subprocess.run(
        ["/bin/bash", "-s"], input=command,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=cwd, timeout=8.0, check=True,
        env={"HOME": "/home/node", "USER": "node",
             "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
    )


def preview(cap: int, *, honcho_hook: Path = HONCHO_HOOK,
            mcp_wrapper: Path = MCP_WRAPPER, target: Path = TARGET,
            env_file: Path = HERMES_ENV, db_path: Path = DATABASE,
            proc_root: Path = PROC) -> dict:
    _preflight(cap, honcho_hook=honcho_hook, mcp_wrapper=mcp_wrapper,
               target=target, env_file=env_file, db_path=db_path)
    honcho, mcp = quiesce._processes(proc_root)
    if not _health("127.0.0.1", 8765) or not _health("embedding-server", 8766):
        raise RestartStopped("one of the two local health checks failed")
    return {
        "cap": cap, "ready": True, "restarted": False,
        "honcho": {"pid": honcho.pid, "starttime": honcho.starttime},
        "mcp": {"pid": mcp.pid, "starttime": mcp.starttime},
    }


def restart(cap: int, expected_honcho: quiesce.Process,
            expected_mcp: quiesce.Process, *, honcho_hook: Path = HONCHO_HOOK,
            mcp_wrapper: Path = MCP_WRAPPER, target: Path = TARGET,
            env_file: Path = HERMES_ENV, db_path: Path = DATABASE,
            proc_root: Path = PROC) -> dict:
    block = _preflight(cap, honcho_hook=honcho_hook, mcp_wrapper=mcp_wrapper,
                       target=target, env_file=env_file, db_path=db_path)
    old_honcho, old_mcp = quiesce._processes(proc_root)
    if (old_honcho, old_mcp) != (expected_honcho, expected_mcp):
        raise RestartStopped("child PID/starttime differs from approved preview")
    gateway = _gateway_parent(old_mcp, proc_root)
    if not _health("127.0.0.1", 8765) or not _health("embedding-server", 8766):
        raise RestartStopped("one of the two local health checks failed")
    quiesce._dream_idle(db_path)
    _terminate_exact(old_honcho, proc_root)
    _wait_until(lambda: _exited(old_honcho, proc_root), 10.0,
                "old Honcho did not exit")
    _wait_until(lambda: _port_closed(8765), 10.0,
                "Honcho port remained occupied")
    new_honcho = _recover_honcho(
        block, old_honcho, cap, env_file=env_file,
        target=target, proc_root=proc_root,
    )
    if not quiesce._same_process(old_mcp, proc_root):
        raise RestartStopped("reviewed MCP child changed before termination")
    if _gateway_parent(old_mcp, proc_root) != gateway:
        raise RestartStopped("gateway parent changed before MCP termination")
    quiesce._dream_idle(db_path)
    _terminate_exact(old_mcp, proc_root)
    _wait_until(lambda: _exited(old_mcp, proc_root), 10.0,
                "old MCP child did not exit")
    new_mcp: quiesce.Process | None = None

    def mcp_ready() -> bool:
        nonlocal new_mcp
        new_mcp = _new_process("hymem-server", old_mcp, cap, proc_root)
        return (new_mcp is not None
                and _gateway_parent(new_mcp, proc_root) == gateway)

    _wait_until(mcp_ready, 30.0, "gateway MCP child/cap did not appear")
    assert new_mcp is not None
    first_seen = time.monotonic()
    _mcp_protocol_probe()
    _observe_gateway_child(new_mcp, cap, gateway, proc_root, first_seen)
    if (new_honcho is None or new_mcp is None
            or not quiesce._same_process(new_honcho, proc_root)
            or not _health("127.0.0.1", 8765)
            or not _health("embedding-server", 8766)):
        raise RestartStopped("post-restart process or health check failed")
    return {
        "cap": cap, "restarted": True, "honcho_health": True,
        "embedding_health": True, "mcp_protocol_probe_passed": True,
        "gateway_child_stable": True,
        "honcho": {"old_pid": old_honcho.pid, "new_pid": new_honcho.pid},
        "mcp": {"old_pid": old_mcp.pid, "new_pid": new_mcp.pid},
    }


def _expected(value: str, role: str) -> quiesce.Process:
    try:
        pid, start = (int(part) for part in value.split(":", 1))
        if pid < 1 or start < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("expected PID:STARTTIME") from None
    return quiesce.Process(pid, role, start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, choices=(16, 32), required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live-signals", action="store_true")
    parser.add_argument("--expected-honcho")
    parser.add_argument("--expected-mcp")
    args = parser.parse_args()
    if os.geteuid() != 1000 or os.getegid() != 1000:
        parser.error("restart must run as Hermes node user (uid/gid 1000)")
    if (Path(__file__).resolve() != STAGE / Path(__file__).name
            or Path(modules.__file__).resolve()
            != STAGE / "hermes_four_module_cutover.py"
            or Path(quiesce.__file__).resolve()
            != STAGE / "hermes_quiesced_cutover.py"):
        parser.error("restart operator must run from the private reviewed stage")
    if args.apply and not args.allow_live_signals:
        parser.error("restart requires --allow-live-signals")
    if args.apply and (not args.expected_honcho or not args.expected_mcp):
        parser.error("restart requires previewed PID:STARTTIME pairs")
    try:
        if args.apply:
            result = restart(
                args.cap,
                _expected(args.expected_honcho, "hymem-honcho"),
                _expected(args.expected_mcp, "hymem-server"),
            )
        else:
            result = preview(args.cap)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"stopped": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, RestartStopped)
                          else "unexpected restart failure; inspect service health"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
