"""One-shot host launcher for the repaired four-question Luna LME diagnostic.

Prepare performs a real, no-inference source/dataset/binary preflight on an
already staged private root. Launch consumes a one-shot marker before dispatch;
an ambiguous dispatch is never retried. This module never starts a service at
import or during prepare.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys


HOST_ROOT = Path("/home/atta")
HOST_UID = 1000
DATASET = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/"
    "lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json")
BINARY = Path("/home/atta/.codex/packages/standalone/releases/"
    "0.158.0-x86_64-unknown-linux-musl/bin/codex")
BINARY_SHA256 = "167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9"
RUNNER_SHA256 = "1b83e3d3ecfa29cdaab5f5ad5aa0b8c4da844723c280de29b952449ef6902048"
INVENTORY_SHA256 = "1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd"
ROOT_NAME = re.compile(r"\.hymem-lme-diagnostic-[a-z0-9_-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
UNIT_PREFIX = "hymem-luna-lme-diagnostic-"
OLD_DEEPSEEK = "523469c234e48ff01d944de22a149251d53f0477aa16133b438bfee2f7030588"
RUNTIME_DIR = Path("/run/user/1000")
BUS = RUNTIME_DIR / "bus"
CGROUP_ROOT = Path("/sys/fs/cgroup")
PRIOR_UNIT = re.compile(r"hymem-[A-Za-z0-9_.@-]+\.service\Z")


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _write_once(path: Path, value: dict) -> None:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")
    _require(len(data) <= 32_768, "metadata_too_large")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _root(root: Path) -> Path:
    _require(root.is_absolute() and root.parent == HOST_ROOT
        and ROOT_NAME.fullmatch(root.name) is not None
        and root.is_dir() and not root.is_symlink(), "root_invalid")
    metadata = root.stat()
    _require(metadata.st_uid == HOST_UID and not metadata.st_mode & 0o077,
        "root_permission_invalid")
    return root


def unit_for(root: Path) -> str:
    _root(root)
    return UNIT_PREFIX + root.name.removeprefix(".hymem-lme-diagnostic-") + ".service"


def _load_runner(root: Path):
    path = root / "code/tools/diagnostics/luna_lme_diagnostic_v10.py"
    _require(_regular(path) and _sha(path) == RUNNER_SHA256, "runner_source_drift")
    spec = importlib.util.spec_from_file_location("pinned_luna_lme_diagnostic_runner", path)
    _require(spec is not None and spec.loader is not None, "runner_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _require(Path(module.__file__).resolve() == path.resolve(), "runner_origin_invalid")
    return module


def verify_sources(root: Path):
    """Recheck every runner input and import the real accepted candidate."""
    root = _root(root)
    _require(_regular(root / "source-map.json")
        and _sha(root / "source-map.json") == INVENTORY_SHA256,
        "inventory_drift")
    runner = _load_runner(root)
    for relative, digest in {**runner.PINS,
            "benchmarks/lme_diagnostic.py": runner.DIAGNOSTIC_HELPER_SHA256,
            "tools/diagnostics/luna_lme_diagnostic_v10.py": RUNNER_SHA256}.items():
        path = root / "code" / relative
        _require(_regular(path) and _sha(path) == digest, "code_source_drift")
    _require(_regular(DATASET) and _sha(DATASET) == runner.DATASET_SHA256,
        "dataset_drift")
    _require(_regular(BINARY) and _sha(BINARY) == BINARY_SHA256,
        "binary_drift")
    loaded = runner.load_verified(root, root / "source-map.json",
        INVENTORY_SHA256, DATASET, BINARY, BINARY_SHA256,
        runner.DIAGNOSTIC_HELPER_SHA256)
    _require(len(loaded["questions"]) == 4, "selected_denominator_invalid")
    runner.receipt_for(root, loaded)
    return runner, loaded


def _bus_env() -> dict[str, str]:
    runtime = RUNTIME_DIR.lstat()
    bus = BUS.lstat()
    _require(stat.S_ISDIR(runtime.st_mode) and runtime.st_uid == HOST_UID
        and not runtime.st_mode & 0o077, "user_runtime_invalid")
    _require(stat.S_ISSOCK(bus.st_mode) and bus.st_uid == HOST_UID,
        "user_bus_invalid")
    return {"HOME": str(HOST_ROOT), "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": str(RUNTIME_DIR),
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(BUS)}


def _unit_state(unit: str) -> tuple[str, str, str, str]:
    fields = ("ActiveState", "SubState", "MainPID", "ControlGroup")
    completed = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit,
        "--property=ActiveState,SubState,MainPID,ControlGroup", "--no-pager"],
        capture_output=True, text=True, timeout=10, check=True, env=_bus_env())
    _require(len(completed.stdout) <= 8192, "unit_state_invalid")
    pairs = [line.split("=", 1) for line in completed.stdout.splitlines()]
    _require(len(pairs) == len(fields) and all(len(pair) == 2 for pair in pairs),
             "unit_state_invalid")
    values = dict(pairs)
    _require(set(values) == set(fields),
        "unit_state_invalid")
    return tuple(values[key] for key in fields)


def recursive_empty(group: Path) -> bool:
    """Check every extant descendant, including threads and populated events."""
    try:
        if group.is_symlink():
            return False
        if not group.exists():
            return True
        if not group.is_dir():
            return False
        pending = [group]
        visited = 0
        while pending:
            node = pending.pop()
            visited += 1
            if visited > 256 or node.is_symlink():
                return False
            if ((node / "cgroup.procs").read_text().strip()
                    or (node / "cgroup.threads").read_text().strip()):
                return False
            events = (node / "cgroup.events").read_text().splitlines()
            pairs = [line.split() for line in events]
            if (len(pairs) > 32 or any(len(pair) != 2 for pair in pairs)
                    or len(dict(pairs)) != len(pairs)
                    or dict(pairs).get("populated") != "0"):
                return False
            for child in node.iterdir():
                if child.is_symlink():
                    return False
                if child.is_dir():
                    pending.append(child)
        return True
    except (OSError, ValueError):
        return False


def _prior_unit_stopped(unit: str, state: tuple[str, str, str, str]) -> bool:
    if (type(unit) is not str or PRIOR_UNIT.fullmatch(unit) is None or ".." in unit
            or type(state) is not tuple or len(state) != 4
            or state[2] != "0"
            or state[:2] not in {("inactive", "dead"), ("failed", "failed"),
                                 ("active", "exited")}):
        return False
    expected = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit
    if state[3] not in {"", expected}:
        return False
    group = CGROUP_ROOT / expected.lstrip("/")
    try:
        if CGROUP_ROOT.is_symlink() or not group.resolve().is_relative_to(
                CGROUP_ROOT.resolve(strict=True)):
            return False
        node = group
        while node != CGROUP_ROOT:
            if node.is_symlink():
                return False
            node = node.parent
        return recursive_empty(group)
    except (OSError, ValueError):
        return False


def host_admission() -> None:
    """Keep old Luna/DeepSeek benchmark workers stopped and resources bounded."""
    _require(sys.platform == "linux" and os.getuid() == HOST_UID
        and os.geteuid() == HOST_UID,
        "wrong_host_user")
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    _require(int(memory["MemAvailable"].split()[0]) * 1024 >= 6 * 1024**3,
        "memory_floor")
    _require(shutil.disk_usage(HOST_ROOT).free >= 20 * 1024**3,
        "disk_floor")
    bus_env = _bus_env()
    for pattern in ("hymem-luna*", "hymem-lme*", "hymem-deepseek*"):
        units = subprocess.run(["/usr/bin/systemctl", "--user", "list-units",
            pattern, "--all", "--no-pager", "--plain", "--no-legend"],
            capture_output=True, text=True, timeout=10, check=True,
            env=bus_env)
        _require(len(units.stdout) <= 65_536, "unit_list_invalid")
        seen: set[str] = set()
        for line in units.stdout.splitlines():
            fields = line.split()
            _require(len(fields) >= 4 and PRIOR_UNIT.fullmatch(fields[0]) is not None
                and ".." not in fields[0] and fields[0] not in seen,
                "unit_list_invalid")
            seen.add(fields[0])
            _require(_prior_unit_stopped(fields[0], _unit_state(fields[0])),
                "prior_benchmark_unit_running")
    old = subprocess.run(["/usr/bin/docker", "inspect", "--format",
        "{{.State.Running}}", OLD_DEEPSEEK], capture_output=True, text=True,
        timeout=10, check=False, env=bus_env)
    _require(old.returncode == 0 and old.stdout.strip() == "false",
        "old_deepseek_running_or_unverified")


def receipt_for(root: Path, runner, loaded: dict) -> dict:
    receipt = runner.receipt_for(root, loaded)
    _require(receipt["unit"] == unit_for(root), "receipt_unit_invalid")
    return receipt


def command(root: Path, receipt: dict, receipt_sha256: str) -> list[str]:
    """A single bounded dispatch; constructing it performs no subprocess call."""
    return ["/usr/bin/systemd-run", "--user", "--quiet", "--unit", receipt["unit"],
        "--property=Type=exec", "--property=Restart=no",
        "--property=KillMode=control-group", "--property=RemainAfterExit=yes",
        "--property=RuntimeMaxSec=14530s", "--property=TimeoutStopSec=10s",
        "--property=MemoryMax=4294967296", "--property=CPUQuota=200%",
        "--property=TasksMax=256", "--property=OOMPolicy=kill",
        "--property=UMask=0077", "--property=WorkingDirectory=" + str(root / "empty"),
        "--property=StandardOutput=file:" + str(root / "safe-terminal.json"),
        "--property=StandardError=file:" + str(root / "private-launch-stderr.log"),
        "/usr/bin/env", "-i", "HOME=/home/atta", "PATH=/usr/local/bin:/usr/bin:/bin",
        "XDG_RUNTIME_DIR=" + str(RUNTIME_DIR),
        "DBUS_SESSION_BUS_ADDRESS=unix:path=" + str(BUS),
        "TMPDIR=" + str(root / "tmp"), "/usr/bin/python3", "-I", "-B",
        str(root / "code/tools/diagnostics/luna_lme_diagnostic_v10.py"),
        "--root", str(root), "--inventory", str(root / "source-map.json"),
        "--inventory-sha256", INVENTORY_SHA256,
        "--dataset", str(DATASET), "--binary", str(BINARY),
        "--binary-sha256", BINARY_SHA256,
        "--questions", "4", "--workers", "4",
        "--receipt-sha256", receipt_sha256,
        "--output-dir", str(root / "run"), "--run"]


def prepare(root: Path) -> dict:
    root = _root(root)
    host_admission()
    _require(not any((root / name).exists() or (root / name).is_symlink() for name in
        ("launch-receipt.json", "launch-attempt.json", "run",
         "private-diagnostic-run.log", "launch-command-result.json",
         "diagnostic-execution-marker.json", "safe-terminal.json",
         "private-launch-stderr.log")),
        "root_already_prepared")
    runner, loaded = verify_sources(root)
    for name in ("empty", "tmp"):
        path = root / name
        if path.exists():
            _require(path.is_dir() and not path.is_symlink()
                and not any(path.iterdir()), "workdir_not_empty")
        else:
            path.mkdir(mode=0o700)
    receipt = receipt_for(root, runner, loaded)
    _write_once(root / "launch-receipt.json", receipt)
    return {"prepared": True, "launched": False, "root": str(root),
        "unit": receipt["unit"],
        "expected_cgroup": receipt["expected_cgroup"],
        "receipt_sha256": _sha(root / "launch-receipt.json"),
        "model_calls": 0}


def launch(root: Path, receipt_sha256: str) -> dict:
    root = _root(root)
    _require(HEX.fullmatch(receipt_sha256) is not None,
        "receipt_sha_invalid")
    host_admission()
    path = root / "launch-receipt.json"
    _require(_regular(path) and _sha(path) == receipt_sha256,
        "receipt_pin_invalid")
    runner, loaded = verify_sources(root)
    receipt = receipt_for(root, runner, loaded)
    expected_bytes = json.dumps(receipt, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")
    _require(path.stat().st_uid == HOST_UID and stat.S_IMODE(path.stat().st_mode) == 0o600
        and path.stat().st_size <= 8192 and path.read_bytes() == expected_bytes,
        "receipt_invalid")
    _require(not any((root / name).exists() or (root / name).is_symlink()
        for name in ("run", "launch-attempt.json", "private-diagnostic-run.log",
                     "launch-command-result.json", runner.EXECUTION_MARKER,
                     "safe-terminal.json", "private-launch-stderr.log")),
        "launch_already_attempted")
    for name in ("empty", "tmp"):
        work = root / name
        _require(work.is_dir() and not work.is_symlink()
            and not any(work.iterdir()), "workdir_not_empty")
    for name in ("safe-terminal.json", "private-launch-stderr.log"):
        fd = os.open(root / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.close(fd)
    _write_once(root / "launch-attempt.json",
        {"receipt_sha256": receipt_sha256, "one_shot": True})
    started = subprocess.run(command(root, receipt, receipt_sha256),
        capture_output=True, timeout=20, check=False, env=_bus_env())
    _write_once(root / "launch-command-result.json",
        {"returncode": started.returncode})
    return {"launch_command_returncode": started.returncode,
        "root": str(root), "unit": receipt["unit"],
        "receipt_sha256": receipt_sha256, "never_retry": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-root")
    action.add_argument("--launch-root")
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    root = Path(args.prepare_root or args.launch_root)
    try:
        if args.prepare_root:
            _require(args.receipt_sha256 is None, "unexpected_receipt")
            result = prepare(root)
        else:
            _require(args.receipt_sha256 is not None, "receipt_required")
            result = launch(root, args.receipt_sha256)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("launch_command_returncode", 0) == 0 else 1
    except Exception:
        print(json.dumps({"ok": False,
            "stage": "prepare" if args.prepare_root else "launch",
            "root": str(root), "reason": "admission_or_launch_failed",
            "never_retry_launch": bool(args.launch_root)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
