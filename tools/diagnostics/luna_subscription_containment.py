"""No-model systemd user-service containment probe for Afrodite (Linux)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time


RUNTIME_SECONDS = 3
STOP_SECONDS = 2
WAIT_SECONDS = 12
UNIT_PREFIX = "hymem-luna-containment-"


def _starttime(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        return stat.rsplit(") ", 1)[1].split()[19]
    except (OSError, IndexError, ValueError):
        return None


def _same_process(pid: int, starttime: str) -> bool:
    return _starttime(pid) == starttime


def _show(unit: str) -> dict[str, str]:
    keys = ("ActiveState", "SubState", "Restart", "KillMode", "RuntimeMaxUSec",
            "TimeoutStopUSec", "NRestarts")
    result = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit, "--no-pager",
                             *[f"--property={key}" for key in keys]],
                            capture_output=True, text=True, timeout=5, check=False)
    if result.returncode != 0:
        return {}
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def _worker(receipt: Path) -> int:
    child = subprocess.Popen(["/usr/bin/sleep", "30"], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    parent_start, child_start = _starttime(os.getpid()), _starttime(child.pid)
    if not parent_start or not child_start:
        return 2
    payload = {"parent_pid": os.getpid(), "parent_starttime": parent_start,
               "child_pid": child.pid, "child_starttime": child_start}
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as output:
        json.dump(payload, output)
        output.flush()
        os.fsync(output.fileno())
    time.sleep(30)  # systemd must terminate both processes well before this ends
    return 3


def _controller() -> dict:
    directory = Path(tempfile.mkdtemp(prefix="hymem-luna-containment-"))
    os.chmod(directory, 0o700)
    receipt = directory / "owned-processes.json"
    unit = UNIT_PREFIX + secrets.token_hex(8) + ".service"
    report = {"ok": False, "stop_code": None, "unit": unit,
              "private_receipt_dir": str(directory), "parent_gone": False,
              "new_session_child_gone": False, "restart_disabled": False,
              "control_group_kill": False, "runtime_limit_verified": False,
              "unit_inactive": False, "runtime_expiry_observed": False}
    launched = False
    try:
        launched_at = time.monotonic()
        command = ["/usr/bin/systemd-run", "--user", "--collect", "--unit", unit,
                   "--property=Type=exec", f"--property=RuntimeMaxSec={RUNTIME_SECONDS}s",
                   f"--property=TimeoutStopSec={STOP_SECONDS}s",
                   "--property=KillMode=control-group", "--property=Restart=no",
                   sys.executable, str(Path(__file__).resolve()), "--worker", str(receipt)]
        started = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        if started.returncode != 0:
            report["stop_code"] = "unit_start_failed"
            return report
        launched = True
        config = {}
        identity = None
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            if not config:
                config = _show(unit)
            if receipt.is_file():
                identity = json.loads(receipt.read_text(encoding="ascii"))
                break
            time.sleep(0.05)
        if identity is None:
            report["stop_code"] = "receipt_missing"
            return report
        report["restart_disabled"] = config.get("Restart") == "no"
        report["control_group_kill"] = config.get("KillMode") == "control-group"
        report["runtime_limit_verified"] = config.get("RuntimeMaxUSec") in {
            f"{RUNTIME_SECONDS}s", f"{RUNTIME_SECONDS * 1_000_000}us"}
        stop_limit_verified = config.get("TimeoutStopUSec") in {
            f"{STOP_SECONDS}s", f"{STOP_SECONDS * 1_000_000}us"}
        no_prior_restart = config.get("NRestarts") == "0"
        if not all((report["restart_disabled"], report["control_group_kill"],
                    report["runtime_limit_verified"], stop_limit_verified,
                    no_prior_restart)):
            report["stop_code"] = "unit_policy_unverified"
            return report
        while time.monotonic() < deadline:
            report["parent_gone"] = not _same_process(identity["parent_pid"], identity["parent_starttime"])
            report["new_session_child_gone"] = not _same_process(identity["child_pid"], identity["child_starttime"])
            state = _show(unit).get("ActiveState")
            report["unit_inactive"] = state in {"inactive", "failed"} or not state
            if all((report["parent_gone"], report["new_session_child_gone"], report["unit_inactive"])):
                break
            time.sleep(0.05)
        report["ok"] = all((report["parent_gone"], report["new_session_child_gone"],
                            report["unit_inactive"]))
        report["runtime_expiry_observed"] = time.monotonic() - launched_at >= RUNTIME_SECONDS - 0.5
        report["ok"] = report["ok"] and report["runtime_expiry_observed"]
        if not report["ok"]:
            report["stop_code"] = "containment_unverified"
        return report
    except (OSError, ValueError, subprocess.SubprocessError):
        report["stop_code"] = "probe_failure"
        return report
    finally:
        if launched and not report["ok"]:
            # Only this uniquely named, probe-owned unit is stopped.
            try:
                subprocess.run(["/usr/bin/systemctl", "--user", "stop", unit],
                               capture_output=True, timeout=5, check=False)
            except (OSError, subprocess.SubprocessError):
                pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args(argv)
    if args.worker is not None:
        if not args.worker.is_absolute() or args.worker.parent.stat().st_mode & 0o077:
            return 2
        return _worker(args.worker)
    if sys.platform != "linux":
        print(json.dumps({"ok": False, "stop_code": "linux_required"}))
        return 1
    report = _controller()
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
