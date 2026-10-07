"""One-shot host launcher for the accepted noncanonical Luna LME diagnostic.

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
RUNNER_SHA256 = "de6ad419c4aa5c5b0db7e0fdbed4385a672325942381af83d45923b86573b737"
INVENTORY_SHA256 = "228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf"
ROOT_NAME = re.compile(r"\.hymem-lme-diagnostic-[a-z0-9_-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
UNIT_PREFIX = "hymem-luna-lme-diagnostic-"
OLD_DEEPSEEK = "523469c234e48ff01d944de22a149251d53f0477aa16133b438bfee2f7030588"
RUNTIME_DIR = Path("/run/user/1000")
BUS = RUNTIME_DIR / "bus"


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
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as output:
        json.dump(value, output, sort_keys=True, separators=(",", ":"))
        output.flush()
        os.fsync(output.fileno())


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
    path = root / "code/tools/diagnostics/luna_lme_diagnostic_v7.py"
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
            "tools/diagnostics/luna_lme_diagnostic_v7.py": RUNNER_SHA256}.items():
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
    return runner


def _bus_env() -> dict[str, str]:
    runtime = RUNTIME_DIR.lstat()
    bus = BUS.lstat()
    _require(stat.S_ISDIR(runtime.st_mode) and runtime.st_uid == HOST_UID
        and not runtime.st_mode & 0o077, "user_runtime_invalid")
    _require(stat.S_ISSOCK(bus.st_mode) and bus.st_uid == HOST_UID,
        "user_bus_invalid")
    return {**os.environ, "XDG_RUNTIME_DIR": str(RUNTIME_DIR),
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(BUS)}


def _unit_state(unit: str) -> tuple[str, str, str, str]:
    completed = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit,
        "--property=ActiveState,SubState,MainPID,ControlGroup", "--no-pager"],
        capture_output=True, text=True, timeout=10, check=True, env=_bus_env())
    fields = dict(line.split("=", 1) for line in completed.stdout.splitlines()
                  if "=" in line)
    _require(set(fields) == {"ActiveState", "SubState", "MainPID", "ControlGroup"},
        "unit_state_invalid")
    return fields["ActiveState"], fields["SubState"], fields["MainPID"], fields["ControlGroup"]


def host_admission() -> None:
    """Keep old Luna/DeepSeek benchmark workers stopped and resources bounded."""
    _require(sys.platform == "linux" and os.getuid() == HOST_UID,
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
        for line in units.stdout.splitlines():
            fields = line.split()
            _require(len(fields) >= 4 and fields[0].startswith("hymem-")
                and fields[0].endswith(".service"), "unit_list_invalid")
            active, sub, pid, group = _unit_state(fields[0])
            _require((active == "active" and sub == "exited" and pid == "0" and group == "")
                or (active in {"inactive", "failed"} and sub not in
                    {"running", "start", "stop"} and pid == "0" and group == ""),
                "prior_benchmark_unit_running")
    old = subprocess.run(["/usr/bin/docker", "inspect", "--format",
        "{{.State.Running}}", OLD_DEEPSEEK], capture_output=True, text=True,
        timeout=10, check=False)
    _require(old.returncode == 0 and old.stdout.strip() == "false",
        "old_deepseek_running_or_unverified")


def receipt_for(root: Path, runner) -> dict:
    unit = unit_for(root)
    return {"schema": "luna-lme-diagnostic-launch-v7", "root": str(root),
        "billing_policy": runner.BILLING_POLICY,
        "notification_policy": runner.NOTIFICATION_POLICY,
        "unit": unit,
        "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "source_sha256": {**runner.PINS,
            "benchmarks/lme_diagnostic.py": runner.DIAGNOSTIC_HELPER_SHA256,
            "tools/diagnostics/luna_lme_diagnostic_v7.py": RUNNER_SHA256},
        "candidate_map_sha256": runner.ACCEPTED_MAP_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "dataset_sha256": runner.DATASET_SHA256,
        "binary_sha256": BINARY_SHA256,
        "selected_count": 4, "workers": 4, "indexing_seconds": 10_800,
        "output_dir": str(root / "run"),
        "limits": {key: list(value) for key, value in runner.MAX_LIMITS.items()}}


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
        str(root / "code/tools/diagnostics/luna_lme_diagnostic_v7.py"),
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
    _require(not any((root / name).exists() for name in
        ("launch-receipt.json", "launch-attempt.json", "run",
         "private-diagnostic-run.log", "launch-command-result.json")),
        "root_already_prepared")
    runner = verify_sources(root)
    for name in ("empty", "tmp"):
        path = root / name
        if path.exists():
            _require(path.is_dir() and not path.is_symlink()
                and not any(path.iterdir()), "workdir_not_empty")
        else:
            path.mkdir(mode=0o700)
    receipt = receipt_for(root, runner)
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
    runner = verify_sources(root)
    receipt = json.loads(path.read_text(encoding="ascii"))
    _require(receipt == receipt_for(root, runner), "receipt_invalid")
    _require(not (root / "run").exists()
        and not (root / "launch-attempt.json").exists()
        and not (root / "private-diagnostic-run.log").exists(),
        "launch_already_attempted")
    for name in ("empty", "tmp"):
        work = root / name
        _require(work.is_dir() and not work.is_symlink()
            and not any(work.iterdir()), "workdir_not_empty")
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
