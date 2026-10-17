"""Guarded Hermes1 cap-32 Honcho and gateway restart; preview has no signals.

The gateway, not an individual MCP child, owns stdio MCP reconnection. A
SIGTERM of the exact idle gateway lets its entrypoint relaunch it, without
creating the planned-restart home-channel notification marker. No chat or MCP
memory tool is invoked; the optional protocol probe only initializes/lists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

if __package__:
    from tools.diagnostics import hermes_targeted_restart as restart
    from tools.diagnostics import hermes_quiesced_cutover as quiesce
else:
    import hermes_targeted_restart as restart
    import hermes_quiesced_cutover as quiesce


STAGE = Path("/tmp/hymem-cutover-20261017")
BACKUP = Path("/home/node/.hermes/backups/hymem-cap16-to32-20261017")
STATUS = Path("/home/node/.hermes/gateway_state.json")
HOME = Path("/home/node/.hermes")
MCP_CONFIG = HOME / "config.yaml"
EXPECTED_MCP_CONFIG_SHA256 = "1c693c3a21d1c171895f735fac215c536c8ae689fa7ff07fcc863911eabd33e3"
PROC = Path("/proc")
CAP = 32


class Cap32RestartStopped(RuntimeError):
    pass


def _backups_ready(backup: Path) -> None:
    restart.modules._private_dir(backup)
    for role in ("mcp", "honcho"):
        saved = backup / f"{role}.cap16"
        info = saved.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.geteuid() or info.st_gid != os.getegid()
                or restart._digest(saved) != restart.CAP_HASHES[16][role]):
            raise Cap32RestartStopped("private cap-16 backup is absent or changed")


def _gateway_entrypoint(gateway: restart.Gateway, proc_root: Path) -> None:
    try:
        if restart._gateway_identity(gateway.pid, proc_root) != gateway:
            raise Cap32RestartStopped("gateway PID/starttime changed")
        fields = (proc_root / str(gateway.pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        parent = int(fields[1])
        argv = [x for x in (proc_root / str(parent) / "cmdline").read_bytes().split(b"\0") if x]
    except (OSError, ValueError, IndexError, restart.RestartStopped):
        raise Cap32RestartStopped("gateway identity is unavailable") from None
    if parent != 7 or argv != [b"bash", b"/usr/local/bin/entrypoint.sh"]:
        raise Cap32RestartStopped("gateway is not owned by the reviewed entrypoint PID 7")


def _gateways(proc_root: Path) -> list[restart.Gateway]:
    found = []
    for directory in proc_root.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            candidate = restart._gateway_identity(int(directory.name), proc_root)
            _gateway_entrypoint(candidate, proc_root)
            found.append(candidate)
        except (restart.RestartStopped, Cap32RestartStopped):
            continue
    return found


def _status_idle(gateway: restart.Gateway, status_path: Path) -> None:
    try:
        value = json.loads(status_path.read_text())
        stamp = datetime.fromisoformat(value["updated_at"])
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
    except (OSError, ValueError, KeyError, TypeError):
        raise Cap32RestartStopped("gateway runtime status is unavailable") from None
    if (value.get("pid") != gateway.pid or value.get("gateway_state") != "running"
            or value.get("active_agents") != 0 or value.get("restart_requested") is not False
            or not -10 <= age <= 120):
        raise Cap32RestartStopped("gateway is not freshly idle and running")


def _no_notice_markers(home: Path) -> None:
    if any((home / name).exists() or (home / name).is_symlink()
           for name in (".restart_notify.json", ".restart_pending.json")):
        raise Cap32RestartStopped("a restart notification marker is already pending")


def _keepalive_config_ready(path: Path) -> None:
    restart._regular_owned(path, 0o600)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_MCP_CONFIG_SHA256:
        raise Cap32RestartStopped("MCP config differs from reviewed keepalive edit")
    try:
        value = yaml.safe_load(payload)
        setting = value["mcp_servers"]["hymem"]["keepalive_interval"]
    except (yaml.YAMLError, KeyError, TypeError):
        raise Cap32RestartStopped("HyMem MCP keepalive config is invalid") from None
    if type(setting) is not int or setting != 60:
        raise Cap32RestartStopped("HyMem MCP keepalive is not 60 seconds")


def _preflight(*, backup: Path, status_path: Path, home: Path,
               proc_root: Path, db_path: Path, mcp_config: Path,
               honcho_hook: Path, mcp_wrapper: Path,
               target: Path, env_file: Path) -> tuple[bytes, quiesce.Process,
                                                       quiesce.Process, restart.Gateway]:
    block = restart._preflight(
        CAP, honcho_hook=honcho_hook, mcp_wrapper=mcp_wrapper,
        target=target, env_file=env_file, db_path=db_path,
    )
    _keepalive_config_ready(mcp_config)
    _backups_ready(backup)
    honcho, mcp = quiesce._processes(proc_root)
    gateway = restart._gateway_parent(mcp, proc_root)
    _gateway_entrypoint(gateway, proc_root)
    if _gateways(proc_root) != [gateway]:
        raise Cap32RestartStopped("expected exactly one reviewed gateway")
    if (not restart._cap_in_process(honcho, 16, proc_root)
            or not restart._cap_in_process(mcp, 16, proc_root)):
        raise Cap32RestartStopped("old child processes do not both run cap 16")
    _status_idle(gateway, status_path)
    _no_notice_markers(home)
    if (not restart._health("127.0.0.1", 8765)
            or not restart._health("embedding-server", 8766)):
        raise Cap32RestartStopped("one of the two local services is unhealthy")
    return block, honcho, mcp, gateway


def preview(*, backup: Path = BACKUP, status_path: Path = STATUS,
            home: Path = HOME, proc_root: Path = PROC,
            db_path: Path = restart.DATABASE,
            mcp_config: Path = MCP_CONFIG,
            honcho_hook: Path = restart.HONCHO_HOOK,
            mcp_wrapper: Path = restart.MCP_WRAPPER,
            target: Path = restart.TARGET,
            env_file: Path = restart.HERMES_ENV) -> dict:
    _block, honcho, mcp, gateway = _preflight(
        backup=backup, status_path=status_path, home=home,
        proc_root=proc_root, db_path=db_path, mcp_config=mcp_config, honcho_hook=honcho_hook,
        mcp_wrapper=mcp_wrapper, target=target, env_file=env_file,
    )
    return {"cap": CAP, "restarted": False,
            "honcho": {"pid": honcho.pid, "starttime": honcho.starttime},
            "mcp": {"pid": mcp.pid, "starttime": mcp.starttime},
            "gateway": {"pid": gateway.pid, "starttime": gateway.starttime}}


def _terminate_gateway(gateway: restart.Gateway, proc_root: Path) -> None:
    _gateway_entrypoint(gateway, proc_root)
    descriptor = os.pidfd_open(gateway.pid)
    try:
        _gateway_entrypoint(gateway, proc_root)
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    finally:
        os.close(descriptor)


def _gateway_exited(gateway: restart.Gateway, proc_root: Path) -> bool:
    try:
        return restart._gateway_identity(gateway.pid, proc_root) != gateway
    except restart.RestartStopped:
        return True


def service_restart(expected_honcho: quiesce.Process, expected_mcp: quiesce.Process,
                    expected_gateway: restart.Gateway, *,
                    backup: Path = BACKUP, status_path: Path = STATUS,
                    home: Path = HOME, proc_root: Path = PROC,
                    db_path: Path = restart.DATABASE,
                    mcp_config: Path = MCP_CONFIG,
                    honcho_hook: Path = restart.HONCHO_HOOK,
                    mcp_wrapper: Path = restart.MCP_WRAPPER,
                    target: Path = restart.TARGET,
                    env_file: Path = restart.HERMES_ENV) -> dict:
    block, honcho, mcp, gateway = _preflight(
        backup=backup, status_path=status_path, home=home,
        proc_root=proc_root, db_path=db_path, mcp_config=mcp_config, honcho_hook=honcho_hook,
        mcp_wrapper=mcp_wrapper, target=target, env_file=env_file,
    )
    if (honcho, mcp, gateway) != (expected_honcho, expected_mcp, expected_gateway):
        raise Cap32RestartStopped("process PID/starttime differs from reviewed preview")
    quiesce._dream_idle(db_path)
    _keepalive_config_ready(mcp_config)
    restart._terminate_exact(honcho, proc_root)
    restart._wait_until(lambda: restart._exited(honcho, proc_root), 10.0,
                        "old Honcho did not exit")
    restart._wait_until(lambda: restart._port_closed(8765), 10.0,
                        "Honcho port remained occupied")
    new_honcho = restart._recover_honcho(
        block, honcho, CAP, env_file=env_file, target=target, proc_root=proc_root)

    # Rebind every condition before the second signal. A failed Honcho launch
    # leaves the healthy old gateway alone for manual recovery.
    quiesce._dream_idle(db_path)
    _keepalive_config_ready(mcp_config)
    _gateway_entrypoint(gateway, proc_root)
    if (_gateways(proc_root) != [gateway]
            or restart._role_processes("hymem-server", proc_root) != [mcp]
            or restart._gateway_parent(mcp, proc_root) != gateway
            or not restart._cap_in_process(mcp, 16, proc_root)
            or not restart._cap_in_process(new_honcho, CAP, proc_root)):
        raise Cap32RestartStopped("processes changed before gateway restart")
    _status_idle(gateway, status_path)
    _no_notice_markers(home)
    if (not restart._health("127.0.0.1", 8765)
            or not restart._health("embedding-server", 8766)):
        raise Cap32RestartStopped("local service became unhealthy before gateway restart")
    _terminate_gateway(gateway, proc_root)
    restart._wait_until(lambda: _gateway_exited(gateway, proc_root), 45.0,
                        "old gateway did not exit")

    new_gateway = None
    def gateway_ready() -> bool:
        nonlocal new_gateway
        found = _gateways(proc_root)
        new_gateway = found[0] if len(found) == 1 and found[0] != gateway else None
        return new_gateway is not None
    restart._wait_until(gateway_ready, 90.0, "entrypoint gateway replacement did not appear")
    assert new_gateway is not None

    new_mcp = None
    def mcp_ready() -> bool:
        nonlocal new_mcp
        found = restart._role_processes("hymem-server", proc_root)
        new_mcp = None
        if len(found) != 1 or found[0] == mcp:
            return False
        try:
            if (restart._gateway_parent(found[0], proc_root) == new_gateway
                    and restart._cap_in_process(found[0], CAP, proc_root)):
                new_mcp = found[0]
        except restart.RestartStopped:
            return False  # child is still starting; bounded wait retries it
        return new_mcp is not None
    restart._wait_until(mcp_ready, 90.0, "gateway-owned cap-32 MCP child did not appear")
    assert new_mcp is not None
    first_seen = time.monotonic()
    restart._mcp_protocol_probe()
    restart._observe_gateway_child(new_mcp, CAP, new_gateway, proc_root, first_seen)
    quiesce._dream_idle(db_path)
    _keepalive_config_ready(mcp_config)
    _status_idle(new_gateway, status_path)
    _no_notice_markers(home)
    if (restart._role_processes("hymem-honcho", proc_root) != [new_honcho]
            or not restart._cap_in_process(new_honcho, CAP, proc_root)
            or not restart._health("127.0.0.1", 8765)
            or not restart._health("embedding-server", 8766)):
        raise Cap32RestartStopped("post-restart local health or Honcho process check failed")
    return {"cap": CAP, "restarted": True,
            "honcho": {"old_pid": honcho.pid, "new_pid": new_honcho.pid},
            "gateway": {"old_pid": gateway.pid, "new_pid": new_gateway.pid},
            "mcp": {"old_pid": mcp.pid, "new_pid": new_mcp.pid},
            "mcp_protocol_probe_passed": True, "gateway_child_stable": True,
            "honcho_health": True, "embedding_health": True}


def _pair(value: str) -> tuple[int, int]:
    try:
        pid, start = (int(part) for part in value.split(":", 1))
        if pid < 1 or start < 1:
            raise ValueError
        return pid, start
    except ValueError:
        raise argparse.ArgumentTypeError("expected PID:STARTTIME") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live-signals", action="store_true")
    parser.add_argument("--expected-honcho")
    parser.add_argument("--expected-mcp")
    parser.add_argument("--expected-gateway")
    args = parser.parse_args()
    if os.geteuid() != 1000 or os.getegid() != 1000:
        parser.error("restart must run as Hermes node user (uid/gid 1000)")
    try:
        restart.modules._private_dir(STAGE)
    except (OSError, restart.modules.CutoverStopped):
        parser.error("restart operator stage must be owned and private")
    if (Path(__file__).resolve() != STAGE / Path(__file__).name
            or Path(restart.__file__).resolve() != STAGE / "hermes_targeted_restart.py"
            or Path(quiesce.__file__).resolve() != STAGE / "hermes_quiesced_cutover.py"):
        parser.error("restart operator must run from the private reviewed stage")
    if args.apply and (not args.allow_live_signals or not args.expected_honcho
                       or not args.expected_mcp or not args.expected_gateway):
        parser.error("apply requires signal flag and all three previewed PID:STARTTIME pairs")
    try:
        if args.apply:
            hp, hs = _pair(args.expected_honcho)
            mp, ms = _pair(args.expected_mcp)
            gp, gs = _pair(args.expected_gateway)
            result = service_restart(
                quiesce.Process(hp, "hymem-honcho", hs),
                quiesce.Process(mp, "hymem-server", ms), restart.Gateway(gp, gs))
        else:
            result = preview()
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"stopped": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, (
                              Cap32RestartStopped, restart.RestartStopped,
                              quiesce.QuiesceStopped))
                          else "unexpected restart failure; inspect service health"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
