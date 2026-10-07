"""One-shot host launcher for the accepted Luna timeout diagnostic.

This file runs only on the prepared UID-1000 host.  Preparing makes no model
request.  Containment and probe are distinct, irreversible dispatch attempts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import types
from typing import Any


HOST_SHA256 = "5a136d0a6d64018836982a9411338c2ae8adca3a5e3fa7b1d7e37e457ddc2dbe"
HOST_HOME = Path("/home/atta")
HOST_UID = 1000
RUNTIME = Path("/run/user/1000")
OLD_DEEPSEEK = "523469c234e48ff01d944de22a149251d53f0477aa16133b438bfee2f7030588"
HEX = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = "luna-application-fault-launch-v1"


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _regular(path: Path, mode: int | None = None) -> bool:
    try:
        meta = path.lstat()
        return (stat.S_ISREG(meta.st_mode) and meta.st_uid == HOST_UID and
                (mode is None or stat.S_IMODE(meta.st_mode) == mode))
    except OSError:
        return False


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _write_once(path: Path, value: dict[str, Any]) -> str:
    data = _json(value)
    _need(len(data) <= 8192, "write_too_large")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(data).hexdigest()


def _host(root: Path) -> Any:
    _need(sys.platform == "linux" and os.getuid() == HOST_UID and
          os.geteuid() == HOST_UID, "host_user_invalid")
    path = root / "application-fault-host-v1.py"
    _need(_regular(path, 0o600), "host_origin_invalid")
    source = path.read_bytes()
    _need(hashlib.sha256(source).hexdigest() == HOST_SHA256, "host_drift")
    module = types.ModuleType("pinned_luna_timeout_host")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    module._root(root)
    return module


def _bus_env() -> dict[str, str]:
    runtime = RUNTIME.lstat()
    bus = (RUNTIME / "bus").lstat()
    _need(stat.S_ISDIR(runtime.st_mode) and runtime.st_uid == HOST_UID and
          not runtime.st_mode & 0o077, "runtime_invalid")
    _need(stat.S_ISSOCK(bus.st_mode) and bus.st_uid == HOST_UID, "bus_invalid")
    return {"HOME": str(HOST_HOME), "PATH": "/usr/bin:/bin",
            "XDG_RUNTIME_DIR": str(RUNTIME),
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(RUNTIME / "bus")}


def _unit_state(unit: str, env: dict[str, str]) -> dict[str, str]:
    keys = ("ActiveState", "SubState", "MainPID", "ControlGroup")
    result = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit,
        "--property=" + ",".join(keys), "--no-pager"],
        capture_output=True, text=True, timeout=10, check=True, env=env)
    _need(len(result.stdout) <= 8192, "unit_report_invalid")
    pairs = [line.split("=", 1) for line in result.stdout.splitlines()]
    _need(len(pairs) == len(keys) and all(len(pair) == 2 for pair in pairs),
          "unit_report_invalid")
    values = dict(pairs)
    _need(set(values) == set(keys), "unit_report_invalid")
    return values


def host_admission(host: Any, root: Path) -> None:
    """No-inference current admission, repeated just before each dispatch."""
    _need(shutil.disk_usage(HOST_HOME).free >= 20 * 1024**3, "disk_floor")
    memory = dict(line.split(":", 1) for line in
                  Path("/proc/meminfo").read_text().splitlines())
    _need(int(memory["MemAvailable"].split()[0]) * 1024 >= 6 * 1024**3,
          "memory_floor")
    _need(_regular(Path(host.BINARY)) and _sha(Path(host.BINARY)) ==
          host.BINARY_SHA256, "binary_drift")
    env = _bus_env()
    for pattern in ("hymem-luna*", "hymem-lme*", "hymem-deepseek*"):
        found = subprocess.run(["/usr/bin/systemctl", "--user", "list-units",
            pattern, "--all", "--no-pager", "--plain", "--no-legend"],
            capture_output=True, text=True, timeout=10, check=True, env=env)
        _need(len(found.stdout) <= 65536, "unit_list_invalid")
        units: set[str] = set()
        for line in found.stdout.splitlines():
            fields = line.split()
            _need(len(fields) >= 4 and fields[0].startswith("hymem-") and
                  fields[0].endswith(".service") and fields[0] not in units,
                  "unit_list_invalid")
            units.add(fields[0])
            state = _unit_state(fields[0], env)
            if fields[0] == host.unit_for(root, "containment") and (
                    root / "containment-attempt.json").exists():
                containment_receipt = root / "containment-receipt.json"
                _need(_regular(containment_receipt, 0o600), "containment_receipt_missing")
                receipt = host.verify_receipt(root, _sha(containment_receipt), "containment")
                terminal = host.terminal_runtime(receipt)
                _need(terminal["policy_verified"] is True and
                      terminal["unit_stopped"] is True and
                      terminal["recursive_cleanup_verified"] is True,
                      "prior_benchmark_unit_running")
                continue
            _need(state["MainPID"] == "0" and state["ControlGroup"] == "" and
                  (state["ActiveState"], state["SubState"]) in
                  {("inactive", "dead"), ("failed", "failed")},
                  "prior_benchmark_unit_running")
    old = subprocess.run(["/usr/bin/docker", "inspect", "--format",
        "{{.State.Running}}", OLD_DEEPSEEK], capture_output=True, text=True,
        timeout=10, check=False, env=env)
    _need(old.returncode == 0 and old.stdout.strip() == "false",
          "old_deepseek_unverified")


def _workdirs(root: Path) -> None:
    for name in ("empty", "tmp"):
        path = root / name
        if not path.exists() and not path.is_symlink():
            path.mkdir(mode=0o700)
        _need(path.is_dir() and not path.is_symlink() and
              path.stat().st_uid == HOST_UID and
              stat.S_IMODE(path.stat().st_mode) == 0o700 and
              not any(path.iterdir()), "workdir_invalid")


def _read_json(path: Path, limit: int) -> Any:
    _need(_regular(path, 0o600) and path.stat().st_size <= limit,
          "result_missing_or_invalid")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            _need(key not in value, "duplicate_json_key")
            value[key] = item
        return value
    def invalid(_: str) -> None:
        raise ValueError("nonfinite_json")
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=invalid)


def _containment_verified(host: Any, root: Path) -> bool:
    path = root / "containment-receipt.json"
    _need(_regular(path, 0o600), "containment_receipt_missing")
    receipt = host.verify_receipt(root, _sha(path), "containment")
    _read_json(root / "containment-attempt.json", 512)
    _need((root / "containment-attempt.json").read_bytes() ==
          _json({"receipt_sha256": _sha(path), "one_shot": True}),
          "containment_attempt_invalid")
    marker = root / "containment-execution-marker.json"
    _read_json(marker, 512)
    _need(marker.read_bytes() == _json({"receipt_sha256": _sha(path),
          "execution_started": True}), "containment_execution_invalid")
    result = _read_json(root / "containment-result.json", 1_000_000)
    counters = result.get("resources") if type(result) is dict else None
    _need(type(result) is dict and set(result) == {"schema", "mode", "verified", "model_calls", "resources"}
          and result["schema"] == host.SCHEMA and result["mode"] == "containment"
          and result["verified"] is True and result["model_calls"] == 0
          and host._valid_counters(counters)
          and counters["pids_denials"] == counters["memory_oom"] == counters["memory_oom_kill"] == 0,
          "containment_result_invalid")
    terminal = host.terminal_runtime(receipt)
    _need(terminal == {"policy_verified": True, "unit_stopped": True,
          "group_matched": True, "recursive_cleanup_verified": True,
          "runtime_exit": "success"}, "containment_cleanup_unverified")
    return True


def prepare(root: Path, mode: str) -> dict[str, Any]:
    _need(mode in {"containment", "probe"}, "mode_invalid")
    host = _host(root)
    host_admission(host, root)
    _workdirs(root)
    consumed = (("containment-attempt.json", "containment-execution-marker.json",
                 "containment-result.json", "launch-receipt.json", "host-result.json")
                if mode == "containment" else
                ("launch-attempt.json", "probe-execution-marker.json", "host-result.json", "private-probe"))
    _need(not any((root / name).exists() or (root / name).is_symlink()
                  for name in consumed),
          "root_already_consumed")
    # The accepted closure is checked even for the no-inference containment mode.
    host.verify_sources(root)
    if mode == "probe":
        _containment_verified(host, root)
    receipt_name = "containment-receipt.json" if mode == "containment" else "launch-receipt.json"
    _need(not (root / receipt_name).exists(), "receipt_already_exists")
    row_sha = host.selected_digest(root)
    digest = _write_once(root / receipt_name, host.receipt_for(root, HOST_SHA256, mode, row_sha))
    return {"schema": SCHEMA, "mode": mode, "prepared": True,
            "model_calls": 0, "root": str(root),
            "unit": host.unit_for(root, mode), "receipt_sha256": digest}


def command(root: Path, receipt: dict[str, Any], digest: str, mode: str) -> list[str]:
    _need(mode in {"containment", "probe"}, "mode_invalid")
    return ["/usr/bin/systemd-run", "--user", "--quiet", "--unit", receipt["unit"],
        "--property=Type=exec", "--property=Restart=no",
        "--property=RemainAfterExit=yes", "--property=KillMode=control-group",
        "--property=RuntimeMaxSec=730s", "--property=TimeoutStopSec=10s",
        "--property=TasksMax=256", "--property=MemoryMax=4294967296",
        "--property=CPUQuota=200%", "--property=OOMPolicy=kill",
        "--property=UMask=0077", "--property=WorkingDirectory=" + str(root / "empty"),
        "--property=StandardOutput=file:" + str(root / ("private-" + mode + "-stdout.log")),
        "--property=StandardError=file:" + str(root / ("private-" + mode + "-stderr.log")),
        "/usr/bin/env", "-i", "HOME=/home/atta", "PATH=/usr/bin:/bin",
        "XDG_RUNTIME_DIR=" + str(RUNTIME),
        "DBUS_SESSION_BUS_ADDRESS=unix:path=" + str(RUNTIME / "bus"),
        "TMPDIR=" + str(root / "tmp"), "/usr/bin/python3", "-I", "-B",
        str(root / "application-fault-host-v1.py"),
        "--containment-only" if mode == "containment" else "--run-once",
        "--root", str(root), "--receipt-sha256", digest]


def launch(root: Path, digest: str, mode: str) -> dict[str, Any]:
    _need(mode in {"containment", "probe"} and type(digest) is str and
          HEX.fullmatch(digest) is not None, "launch_arguments_invalid")
    host = _host(root)
    path = root / ("containment-receipt.json" if mode == "containment" else
                   "launch-receipt.json")
    receipt = host.verify_receipt(root, digest, mode)
    _need(_sha(path) == digest, "receipt_drift")
    if mode == "probe":
        _containment_verified(host, root)
    host_admission(host, root)
    _workdirs(root)
    host.verify_sources(root)
    attempt = root / ("containment-attempt.json" if mode == "containment" else
                      "launch-attempt.json")
    result_name = "containment-result.json" if mode == "containment" else "host-result.json"
    execution_name = ("containment-execution-marker.json" if mode == "containment" else
                      "probe-execution-marker.json")
    output_names = ("private-" + mode + "-stdout.log",
                    "private-" + mode + "-stderr.log")
    _need(not any((root / name).exists() or (root / name).is_symlink()
                  for name in (attempt.name, result_name, execution_name, *output_names, "private-probe")),
          "already_attempted")
    for name in output_names:
        fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
    _write_once(attempt, {"receipt_sha256": digest, "one_shot": True})
    # The marker is durable before systemd-run.  Any exception/timeout is an
    # ambiguous, consumed attempt and must never be retried.
    try:
        started = subprocess.run(command(root, receipt, digest, mode),
            capture_output=True, timeout=20, check=False, env=_bus_env())
        code = started.returncode
    except BaseException:
        code = None
    return {"schema": SCHEMA, "mode": mode, "attempted": True,
            "never_retry": True, "command_returncode": code,
            "root": str(root), "unit": receipt["unit"], "receipt_sha256": digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-root")
    action.add_argument("--launch-root")
    parser.add_argument("--mode", choices=("containment", "probe"), required=True)
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    try:
        root = Path(args.prepare_root or args.launch_root)
        if args.prepare_root:
            _need(args.receipt_sha256 is None, "unexpected_digest")
            result = prepare(root, args.mode)
        else:
            result = launch(root, args.receipt_sha256, args.mode)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0 if result.get("command_returncode", 0) == 0 else 1
    except BaseException:
        print(json.dumps({"schema": SCHEMA, "status": "unverified",
                          "mode": args.mode}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
