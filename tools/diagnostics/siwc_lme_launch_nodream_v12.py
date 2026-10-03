"""One-shot SIWC Luna no-dream LME launcher; preparation makes no model call."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
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
SOURCE_ROOT = Path(__file__).resolve().parents[2]
HOST_UID = 1000
DATASET = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json")
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
RUNNER_RELATIVE = "tools/diagnostics/siwc_lme_nodream_v13.py"
RUNNER_SHA256 = "787925aed73b8e4d4a82979400d3650770d249d5e19bd2bc44b31b77198a4ad0"
INVENTORY_SHA256 = "022b1e60f68afd1eac10fc48b76feb02d7c2ffbf4734d84bbd526229a61a2f17"
ROOT_NAME = re.compile(r"\.hymem-siwc-lme-nodream-[a-z0-9-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
UNIT_PREFIX = "hymem-siwc-lme-nodream-"
RUNTIME_DIR = Path("/run/user/1000")
BUS = RUNTIME_DIR / "bus"
CGROUP_ROOT = Path("/sys/fs/cgroup")
PRIOR_UNIT = re.compile(r"hymem-[A-Za-z0-9_.@_-]+\.service\Z")
SELECTION_LOCK = HOST_ROOT / ".hymem-siwc-lme-selection.lock"


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


@contextmanager
def selection_lock():
    """Serialize overlap selection and live admission across new launchers."""
    fd = os.open(SELECTION_LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        metadata = os.fstat(fd)
        require(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == HOST_UID
            and stat.S_IMODE(metadata.st_mode) == 0o600 and metadata.st_nlink == 1,
            "selection_lock_invalid")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def checked_root(root: Path) -> Path:
    require(root.is_absolute() and root.parent == HOST_ROOT
        and ROOT_NAME.fullmatch(root.name) is not None
        and root.is_dir() and not root.is_symlink(), "root_invalid")
    metadata = root.stat()
    require(metadata.st_uid == HOST_UID and not metadata.st_mode & 0o077,
        "root_permission_invalid")
    return root


def launched_batches() -> list[dict]:
    """Inspect prior one-shot receipts; failed windows still count as attempted."""
    batches = []
    for root in HOST_ROOT.iterdir():
        if ROOT_NAME.fullmatch(root.name) is None or not root.is_dir() or root.is_symlink():
            continue
        receipt_path = root / "launch-receipt.json"
        attempt_path = root / "launch-attempt.json"
        if (not regular(receipt_path) or receipt_path.stat().st_size > 8192
                or not regular(attempt_path)):
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            offset = receipt.get("source_offset", 0)
            count = receipt.get("selected_count")
            if (receipt.get("dataset_sha256") != DATASET_SHA256
                    or type(offset) is not int or type(count) is not int
                    or not 0 <= offset <= 500 - count or not 1 <= count <= 4):
                continue
            identity_fields = {key: receipt.get(key) for key in (
                "dataset_sha256", "candidate_map_sha256", "billing_policy", "model")}
            identity_fields["source_sha256"] = receipt.get("source_sha256")
            identity = hashlib.sha256(json.dumps(identity_fields, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest()
            status = "attempted"
            correct_count = None
            result_path = root / "run" / "diagnostic-result.json"
            if regular(result_path) and result_path.stat().st_size <= 128_000:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if (result.get("diagnostic_complete") is True
                        and result.get("selected_denominator") == count
                        and result.get("scored_count") == count):
                    status = "complete"
                    if (type(result.get("correct_count")) is int
                            and 0 <= result["correct_count"] <= count):
                        correct_count = result["correct_count"]
                else:
                    status = "incomplete"
            batches.append({"root": str(root), "offset": offset,
                "count": count, "status": status,
                "correct_count": correct_count,
                "run_identity_sha256": identity})
        except (OSError, ValueError, TypeError, UnicodeError):
            continue
    return sorted(batches, key=lambda item: (item["offset"], item["root"]))


def launched_window_overlap(offset: int, count: int) -> bool:
    return any(offset < item["offset"] + item["count"]
        and item["offset"] < offset + count for item in launched_batches())


def status() -> dict:
    batches = launched_batches()
    complete = [item for item in batches if item["status"] == "complete"]
    covered = {index for item in complete
        for index in range(item["offset"], item["offset"] + item["count"])}
    overlap = sum(item["count"] for item in complete) != len(covered)
    mixed = len({item["run_identity_sha256"] for item in complete}) > 1
    accurate = not overlap and not mixed and all(type(item["correct_count"]) is int
        for item in complete)
    correct = sum(item["correct_count"] for item in complete) if accurate else None
    next_offset = next((index for index in range(500) if index not in covered), None)
    return {"schema": "siwc-lme-batch-status-v1", "dataset_sha256": DATASET_SHA256,
        "rows_total": 500, "rows_completed": len(covered),
        "duplicate_completed_windows": overlap,
        "mixed_completed_run_identity": mixed,
        "diagnostic_correct_completed": correct,
        "diagnostic_accuracy_completed": (correct / len(covered)
            if accurate and covered else None),
        "canonical_r9_artifact": False, "official_model_score": False,
        "next_incomplete_offset": next_offset, "batches": batches,
        "model_calls": 0}


def assemble(root: Path, *, questions: int = 1, workers: int = 1,
             offset: int = 0, repeat_window_authorized: bool = False) -> dict:
    """Create a fresh pinned capsule, then prepare its no-call launch receipt."""
    require(root.is_absolute() and root.parent == HOST_ROOT
        and ROOT_NAME.fullmatch(root.name) is not None
        and not root.exists() and not root.is_symlink(), "root_invalid")
    require(repeat_window_authorized or not launched_window_overlap(offset, questions),
        "window_already_launched")
    runner = load_runner_path(SOURCE_ROOT / RUNNER_RELATIVE)
    inventory_path = SOURCE_ROOT / "source-map.json"
    require(regular(inventory_path), "inventory_missing")
    inventory = json.loads(inventory_path.read_text(encoding="ascii"))
    template = inventory.get("source_sha256")
    require(type(template) is dict and len(template) == runner.ACCEPTED_FILES,
        "inventory_drift")
    files = {}
    for relative in template:
        source = SOURCE_ROOT / relative
        require(regular(source), "candidate_source_missing")
        files[relative] = sha(source)
    inventory_bytes = json.dumps({"source_sha256": files}, sort_keys=True,
        separators=(",", ":")).encode("ascii")
    require(hashlib.sha256(inventory_bytes).hexdigest() == INVENTORY_SHA256,
        "inventory_pin_mismatch")
    code_files = {**runner.PINS, **runner.SIWC_PINS,
        "benchmarks/lme_diagnostic.py": runner.DIAGNOSTIC_HELPER_SHA256,
        RUNNER_RELATIVE: RUNNER_SHA256}
    root.mkdir(mode=0o700)
    for destination, manifest in ((root / "candidate", files),
                                  (root / "code", code_files)):
        destination.mkdir(mode=0o700)
        for relative, digest in manifest.items():
            rel = Path(relative)
            require(type(relative) is str and rel.parts
                and not rel.is_absolute() and all(part not in (".", "..") for part in rel.parts)
                and HEX.fullmatch(digest) is not None, "source_path_invalid")
            source = SOURCE_ROOT / rel
            require(regular(source) and sha(source) == digest, "source_drift")
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, target)
            target.chmod(0o600)
    with (root / "source-map.json").open("xb") as stream:
        stream.write(inventory_bytes)
    (root / "source-map.json").chmod(0o600)
    return _prepare(root, questions=questions, workers=workers, offset=offset,
        repeat_window_authorized=repeat_window_authorized)


def unit_for(root: Path) -> str:
    checked_root(root)
    return root.name.removeprefix(".") + ".service"


def load_runner_path(path: Path):
    require(regular(path) and sha(path) == RUNNER_SHA256, "runner_source_drift")
    spec = importlib.util.spec_from_file_location("pinned_siwc_lme_runner", path)
    require(spec is not None and spec.loader is not None, "runner_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path.resolve(), "runner_origin_invalid")
    return module


def load_runner(root: Path):
    return load_runner_path(root / "code" / RUNNER_RELATIVE)


def verify_sources(root: Path, questions: int, offset: int):
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
        INVENTORY_SHA256, DATASET, runner.DIAGNOSTIC_HELPER_SHA256, questions,
        offset)
    require(len(loaded["questions"]) == questions, "selected_denominator_invalid")
    runner.verify_owner_identity(loaded["siwc"])
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
    for pattern in ("hymem-luna*", "hymem-siwc-lme-diagnostic-*",
                    "hymem-siwc-lme-nodream-*"):
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


def command(root: Path, receipt: dict, receipt_sha256: str) -> list[str]:
    return ["/usr/bin/systemd-run", "--user", "--quiet", "--unit", receipt["unit"],
        "--property=Type=exec", "--property=Restart=no", "--property=KillMode=control-group",
        "--property=RemainAfterExit=yes", "--property=RuntimeMaxSec=25330s",
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
        "--dataset", str(DATASET), "--questions", str(receipt["selected_count"]),
        "--workers", str(receipt["workers"]), "--offset", str(receipt["source_offset"]),
        *(["--repeat-window-authorized"] if receipt["repeat_window_authorized"] else []),
        "--receipt-sha256", receipt_sha256, "--output-dir", str(root / "run"), "--run"]


def _prepare(root: Path, *, questions: int = 1, workers: int = 1,
             offset: int = 0, repeat_window_authorized: bool = False) -> dict:
    root = checked_root(root)
    require(type(questions) is int and 1 <= questions <= 4
        and type(workers) is int and 1 <= workers <= 4
        and type(offset) is int and 0 <= offset <= 500 - questions
        and type(repeat_window_authorized) is bool, "selection_invalid")
    require(repeat_window_authorized or not launched_window_overlap(offset, questions),
        "window_already_launched")
    require(not any((root / name).exists() or (root / name).is_symlink() for name in
        ("launch-receipt.json", "launch-attempt.json", "run", "private-diagnostic-run.log",
         "launch-command-result.json", "diagnostic-execution-marker.json", "safe-terminal.json",
         "private-launch-stderr.log")), "root_already_prepared")
    runner, loaded = verify_sources(root, questions, offset)
    for name in ("empty", "tmp"):
        path = root / name
        if path.exists():
            require(path.is_dir() and not path.is_symlink() and not any(path.iterdir()),
                "workdir_not_empty")
        else:
            path.mkdir(mode=0o700)
    receipt = runner.receipt_for(root, loaded, workers=workers,
        repeat_window_authorized=repeat_window_authorized)
    require(receipt["unit"] == unit_for(root), "receipt_unit_invalid")
    write_once(root / "launch-receipt.json", receipt)
    return {"prepared": True, "launched": False, "root": str(root),
        "unit": receipt["unit"], "expected_cgroup": receipt["expected_cgroup"],
        "receipt_sha256": sha(root / "launch-receipt.json"), "model_calls": 0,
        "selected_count": questions, "workers": workers,
        "source_offset": offset,
        "api_fallback_allowed": False}


def prepare(root: Path, *, questions: int = 1, workers: int = 1,
            offset: int = 0, repeat_window_authorized: bool = False) -> dict:
    with selection_lock():
        return _prepare(root, questions=questions, workers=workers,
            offset=offset, repeat_window_authorized=repeat_window_authorized)


def preflight(root: Path) -> dict:
    """Recheck a prepared receipt and source graph without a model call."""
    root = checked_root(root)
    receipt_path = root / "launch-receipt.json"
    require(regular(receipt_path) and receipt_path.stat().st_size <= 8192,
        "receipt_missing")
    recorded = json.loads(receipt_path.read_text(encoding="ascii"))
    questions, workers = recorded.get("selected_count"), recorded.get("workers")
    offset = recorded.get("source_offset")
    repeat = recorded.get("repeat_window_authorized")
    require(type(questions) is int and 1 <= questions <= 4
        and type(workers) is int and 1 <= workers <= 4
        and type(offset) is int and 0 <= offset <= 500 - questions
        and type(repeat) is bool, "selection_invalid")
    runner, loaded = verify_sources(root, questions, offset)
    expected = runner.receipt_for(root, loaded, workers=workers,
        repeat_window_authorized=repeat)
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")
    require(receipt_path.read_bytes() == encoded, "receipt_drift")
    return {"preflight_verified": True, "model_calls": 0,
        "root": str(root), "receipt_sha256": sha(receipt_path),
        "source_offset": offset, "selected_count": questions,
        "workers": workers, "api_fallback_allowed": False}


def _launch(root: Path, receipt_sha256: str) -> dict:
    root = checked_root(root)
    require(HEX.fullmatch(receipt_sha256) is not None, "receipt_sha_invalid")
    host_admission()
    receipt_path = root / "launch-receipt.json"
    require(regular(receipt_path) and sha(receipt_path) == receipt_sha256, "receipt_pin_invalid")
    preliminary = json.loads(receipt_path.read_text(encoding="ascii"))
    questions, workers = preliminary.get("selected_count"), preliminary.get("workers")
    offset = preliminary.get("source_offset")
    repeat_window_authorized = preliminary.get("repeat_window_authorized")
    require(type(questions) is int and 1 <= questions <= 4
        and type(workers) is int and 1 <= workers <= 4
        and type(offset) is int and 0 <= offset <= 500 - questions
        and type(repeat_window_authorized) is bool, "selection_invalid")
    require(repeat_window_authorized or not launched_window_overlap(offset, questions),
        "window_already_launched")
    runner, loaded = verify_sources(root, questions, offset)
    receipt = runner.receipt_for(root, loaded, workers=workers,
        repeat_window_authorized=repeat_window_authorized)
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


def launch(root: Path, receipt_sha256: str) -> dict:
    with selection_lock():
        return _launch(root, receipt_sha256)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--assemble-root")
    action.add_argument("--prepare-root")
    action.add_argument("--preflight-root")
    action.add_argument("--launch-root")
    action.add_argument("--status", action="store_true")
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--questions", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--allow-repeat-window", action="store_true")
    parser.add_argument("--multi-question-batch", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.assemble_root or args.prepare_root or args.preflight_root
        or args.launch_root or HOST_ROOT)
    try:
        if args.status:
            require(args.receipt_sha256 is None, "unexpected_receipt")
            result = status()
        elif args.preflight_root:
            require(args.receipt_sha256 is None, "unexpected_receipt")
            result = preflight(root)
        elif args.assemble_root or args.prepare_root:
            require(args.receipt_sha256 is None, "unexpected_receipt")
            require((args.questions == 1 or args.multi_question_batch)
                and 0 <= args.offset <= 500 - args.questions,
                "explicit_batch_selection_required")
            options = {"questions": args.questions, "workers": args.workers,
                "offset": args.offset,
                "repeat_window_authorized": args.allow_repeat_window}
            if args.assemble_root:
                with selection_lock():
                    result = assemble(root, **options)
            else:
                result = prepare(root, **options)
        else:
            require(args.receipt_sha256 is not None, "receipt_required")
            require(args.questions == args.workers == 1 and args.offset == 0
                and not args.allow_repeat_window and not args.multi_question_batch,
                "launch_uses_receipt_selection")
            result = launch(root, args.receipt_sha256)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("launch_command_returncode", 0) == 0 else 1
    except Exception:
        stage = ("status" if args.status else "assemble" if args.assemble_root
            else "prepare" if args.prepare_root else "preflight"
            if args.preflight_root else "launch")
        print(json.dumps({"ok": False, "stage": stage,
            "root": str(root), "reason": "admission_or_launch_failed",
            "never_retry_launch": bool(args.launch_root)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
