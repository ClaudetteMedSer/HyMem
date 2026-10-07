"""One-shot private SIWC LME host launcher; prepare makes no model call."""
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
DATASET = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json")
RUNNER_RELATIVE = "tools/diagnostics/siwc_lme_diagnostic_v5.py"
RUNNER_SHA256 = "57f42178d3f0fad8ac4775fa01c36eb8dd5fd37cbf1391af04a539631dfc1094"
INVENTORY_SHA256 = "b87ce83ca123fb2a51f1716ca3d3d71fd3b73799a30469b34a8f033babaa90b6"
ROOT_NAME = re.compile(r"\.hymem-siwc-lme-diagnostic-[a-z0-9_-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
UNIT_PREFIX = "hymem-siwc-lme-diagnostic-"
OLD_DEEPSEEK = "523469c234e48ff01d944de22a149251d53f0477aa16133b438bfee2f7030588"
RUNTIME_DIR = Path("/run/user/1000")
BUS = RUNTIME_DIR / "bus"
CGROUP_ROOT = Path("/sys/fs/cgroup")
PRIOR_UNIT = re.compile(r"hymem-[A-Za-z0-9_.@-]+\.service\Z")


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        return False


def write_once(path: Path, value: dict) -> None:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    require(len(data) <= 8192, "metadata_too_large")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def checked_root(root: Path) -> Path:
    require(root.is_absolute() and root.parent == HOST_ROOT
        and ROOT_NAME.fullmatch(root.name) is not None
        and root.is_dir() and not root.is_symlink(), "root_invalid")
    metadata = root.stat()
    require(metadata.st_uid == HOST_UID and not metadata.st_mode & 0o077,
        "root_permission_invalid")
    return root


def unit_for(root: Path) -> str:
    checked_root(root)
    return root.name.removeprefix(".") + ".service"


def load_runner(root: Path):
    path = root / "code" / RUNNER_RELATIVE
    require(regular(path) and sha(path) == RUNNER_SHA256, "runner_source_drift")
    spec = importlib.util.spec_from_file_location("pinned_siwc_lme_runner", path)
    require(spec is not None and spec.loader is not None, "runner_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path.resolve(), "runner_origin_invalid")
    return module


def verify_sources(root: Path):
    root = checked_root(root)
    require(regular(root / "source-map.json")
        and sha(root / "source-map.json") == INVENTORY_SHA256, "inventory_drift")
    runner = load_runner(root)
    for relative, digest in {**runner.PINS, **runner.SIWC_PINS,
            "benchmarks/lme_diagnostic.py": runner.DIAGNOSTIC_HELPER_SHA256,
            RUNNER_RELATIVE: RUNNER_SHA256}.items():
        path = root / "code" / relative
        require(regular(path) and sha(path) == digest, "code_source_drift")
    require(regular(DATASET) and sha(DATASET) == runner.DATASET_SHA256, "dataset_drift")
    runner.verify_runtime_identity()
    loaded = runner.load_verified(root, root / "source-map.json",
        INVENTORY_SHA256, DATASET, runner.DIAGNOSTIC_HELPER_SHA256)
    require(len(loaded["questions"]) == 4, "selected_denominator_invalid")
    runner.verify_owner_identity(loaded["siwc"])
    runner.receipt_for(root, loaded)
    return runner, loaded


def bus_env() -> dict[str, str]:
    runtime, bus = RUNTIME_DIR.lstat(), BUS.lstat()
    require(stat.S_ISDIR(runtime.st_mode) and runtime.st_uid == HOST_UID
        and not runtime.st_mode & 0o077, "user_runtime_invalid")
    require(stat.S_ISSOCK(bus.st_mode) and bus.st_uid == HOST_UID, "user_bus_invalid")
    return {"HOME": str(HOST_ROOT), "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": str(RUNTIME_DIR),
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(BUS)}


def unit_state(unit: str) -> tuple[str, str, str, str]:
    fields = ("ActiveState", "SubState", "MainPID", "ControlGroup")
    done = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit,
        "--property=" + ",".join(fields), "--no-pager"],
        capture_output=True, text=True, timeout=10, check=True, env=bus_env())
    require(len(done.stdout) <= 8192, "unit_state_invalid")
    pairs = [line.split("=", 1) for line in done.stdout.splitlines()]
    require(len(pairs) == len(fields) and all(len(pair) == 2 for pair in pairs)
        and set(dict(pairs)) == set(fields) and len(dict(pairs)) == len(fields),
        "unit_state_invalid")
    values = dict(pairs)
    return tuple(values[key] for key in fields)


def recursive_empty(group: Path) -> bool:
    try:
        if group.is_symlink():
            return False
        if not group.exists():
            return True
        if not group.is_dir():
            return False
        pending, visited = [group], 0
        while pending:
            node = pending.pop()
            visited += 1
            if visited > 256 or node.is_symlink():
                return False
            if (node / "cgroup.procs").read_text().strip() or (node / "cgroup.threads").read_text().strip():
                return False
            pairs = [line.split() for line in (node / "cgroup.events").read_text().splitlines()]
            if (len(pairs) > 32 or any(len(pair) != 2 for pair in pairs)
                    or len(dict(pairs)) != len(pairs) or dict(pairs).get("populated") != "0"):
                return False
            for child in node.iterdir():
                if child.is_symlink():
                    return False
                if child.is_dir():
                    pending.append(child)
        return True
    except (OSError, ValueError):
        return False


def prior_unit_stopped(unit: str, state: tuple[str, str, str, str]) -> bool:
    if (type(unit) is not str or PRIOR_UNIT.fullmatch(unit) is None or ".." in unit
            or type(state) is not tuple or len(state) != 4 or state[2] != "0"
            or state[:2] not in {("inactive", "dead"), ("failed", "failed"),
                                 ("active", "exited")}):
        return False
    expected = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit
    if state[3] not in {"", expected}:
        return False
    group = CGROUP_ROOT / expected.lstrip("/")
    try:
        if CGROUP_ROOT.is_symlink() or not group.resolve().is_relative_to(CGROUP_ROOT.resolve(strict=True)):
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
    require(sys.platform == "linux" and os.getuid() == HOST_UID
        and os.geteuid() == HOST_UID, "wrong_host_user")
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    require(int(memory["MemAvailable"].split()[0]) * 1024 >= 6 * 1024**3, "memory_floor")
    require(shutil.disk_usage(HOST_ROOT).free >= 20 * 1024**3, "disk_floor")
    environment = bus_env()
    for pattern in ("hymem-luna*", "hymem-lme*", "hymem-deepseek*", "hymem-siwc-lme-diagnostic-*"):
        units = subprocess.run(["/usr/bin/systemctl", "--user", "list-units", pattern,
            "--all", "--no-pager", "--plain", "--no-legend"], capture_output=True,
            text=True, timeout=10, check=True, env=environment)
        require(len(units.stdout) <= 65_536, "unit_list_invalid")
        seen: set[str] = set()
        for line in units.stdout.splitlines():
            fields = line.split()
            require(len(fields) >= 4 and PRIOR_UNIT.fullmatch(fields[0]) is not None
                and ".." not in fields[0] and fields[0] not in seen, "unit_list_invalid")
            seen.add(fields[0])
            require(prior_unit_stopped(fields[0], unit_state(fields[0])),
                "prior_benchmark_unit_running")
    old = subprocess.run(["/usr/bin/docker", "inspect", "--format", "{{.State.Running}}",
        OLD_DEEPSEEK], capture_output=True, text=True, timeout=10, check=False, env=environment)
    require(old.returncode == 0 and old.stdout.strip() == "false", "old_deepseek_running_or_unverified")


def command(root: Path, receipt: dict, receipt_sha256: str) -> list[str]:
    return ["/usr/bin/systemd-run", "--user", "--quiet", "--unit", receipt["unit"],
        "--property=Type=exec", "--property=Restart=no", "--property=KillMode=control-group",
        "--property=RemainAfterExit=yes", "--property=RuntimeMaxSec=14530s",
        "--property=TimeoutStopSec=10s", "--property=MemoryMax=4294967296",
        "--property=CPUQuota=200%", "--property=TasksMax=256", "--property=OOMPolicy=kill",
        "--property=UMask=0077", "--property=WorkingDirectory=" + str(root / "empty"),
        "--property=StandardOutput=file:" + str(root / "safe-terminal.json"),
        "--property=StandardError=file:" + str(root / "private-launch-stderr.log"),
        "/usr/bin/env", "-i", "HOME=/home/atta", "PATH=/usr/local/bin:/usr/bin:/bin",
        "XDG_RUNTIME_DIR=" + str(RUNTIME_DIR), "DBUS_SESSION_BUS_ADDRESS=unix:path=" + str(BUS),
        "TMPDIR=" + str(root / "tmp"), str(receipt["runtime_path"]), "-I", "-B",
        str(root / "code" / RUNNER_RELATIVE), "--root", str(root),
        "--inventory", str(root / "source-map.json"), "--inventory-sha256", INVENTORY_SHA256,
        "--dataset", str(DATASET), "--questions", "4", "--workers", "4",
        "--receipt-sha256", receipt_sha256, "--output-dir", str(root / "run"), "--run"]


def prepare(root: Path) -> dict:
    root = checked_root(root)
    host_admission()
    require(not any((root / name).exists() or (root / name).is_symlink() for name in
        ("launch-receipt.json", "launch-attempt.json", "run", "private-diagnostic-run.log",
         "launch-command-result.json", "diagnostic-execution-marker.json", "safe-terminal.json",
         "private-launch-stderr.log")), "root_already_prepared")
    runner, loaded = verify_sources(root)
    for name in ("empty", "tmp"):
        path = root / name
        if path.exists():
            require(path.is_dir() and not path.is_symlink() and not any(path.iterdir()),
                "workdir_not_empty")
        else:
            path.mkdir(mode=0o700)
    receipt = runner.receipt_for(root, loaded)
    require(receipt["unit"] == unit_for(root), "receipt_unit_invalid")
    write_once(root / "launch-receipt.json", receipt)
    return {"prepared": True, "launched": False, "root": str(root),
        "unit": receipt["unit"], "expected_cgroup": receipt["expected_cgroup"],
        "receipt_sha256": sha(root / "launch-receipt.json"), "model_calls": 0}


def launch(root: Path, receipt_sha256: str) -> dict:
    root = checked_root(root)
    require(HEX.fullmatch(receipt_sha256) is not None, "receipt_sha_invalid")
    host_admission()
    receipt_path = root / "launch-receipt.json"
    require(regular(receipt_path) and sha(receipt_path) == receipt_sha256, "receipt_pin_invalid")
    runner, loaded = verify_sources(root)
    receipt = runner.receipt_for(root, loaded)
    expected = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    require(receipt_path.stat().st_uid == HOST_UID
        and stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
        and len(expected) <= 8192 and receipt_path.read_bytes() == expected, "receipt_invalid")
    require(not any((root / name).exists() or (root / name).is_symlink() for name in
        ("run", "launch-attempt.json", "private-diagnostic-run.log", "launch-command-result.json",
         runner.EXECUTION_MARKER, "safe-terminal.json", "private-launch-stderr.log")),
        "launch_already_attempted")
    for name in ("empty", "tmp"):
        work = root / name
        require(work.is_dir() and not work.is_symlink() and not any(work.iterdir()), "workdir_not_empty")
    for name in ("safe-terminal.json", "private-launch-stderr.log"):
        fd = os.open(root / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.close(fd)
    write_once(root / "launch-attempt.json", {"receipt_sha256": receipt_sha256, "one_shot": True})
    started = subprocess.run(command(root, receipt, receipt_sha256),
        capture_output=True, timeout=20, check=False, env=bus_env())
    write_once(root / "launch-command-result.json", {"returncode": started.returncode})
    return {"launch_command_returncode": started.returncode, "root": str(root),
        "unit": receipt["unit"], "receipt_sha256": receipt_sha256, "never_retry": True}


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
            require(args.receipt_sha256 is None, "unexpected_receipt")
            result = prepare(root)
        else:
            require(args.receipt_sha256 is not None, "receipt_required")
            result = launch(root, args.receipt_sha256)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("launch_command_returncode", 0) == 0 else 1
    except Exception:
        print(json.dumps({"ok": False, "stage": "prepare" if args.prepare_root else "launch",
            "root": str(root), "reason": "admission_or_launch_failed",
            "never_retry_launch": bool(args.launch_root)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
