"""Read-only, metadata-only observer for one pinned Luna LME-S pilot unit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True


PILOT_SHA256 = "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0"
TRANSPORT_SHA256 = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
MAX_JSON_BYTES = 16 * 1024 * 1024
UNIT = re.compile(r"hymem-luna-lme-v2-[a-z0-9-]{1,40}\.service\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
SAFE_STATE = frozenset({"active", "activating", "deactivating", "failed", "inactive"})


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict | None:
    if not path.is_file() or path.is_symlink():
        return None
    if path.stat().st_size > MAX_JSON_BYTES:
        return None
    with path.open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        return None
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError):
        return None
    return value if type(value) is dict else None


def _safe_count(value):
    return value if type(value) is int and 0 <= value <= 100_000_000 else None


def _safe_turns(value):
    return value if type(value) is int and 0 <= value <= 1200 else None


def _safe_bool(value):
    return value if type(value) is bool else None


def summarize_progress(value: dict | None) -> dict:
    if value is None:
        return {"available": False, "usage_complete": None, "known_tokens": None}
    flight = _safe_bool(value.get("invocation_in_flight"))
    usage = _safe_bool(value.get("usage_complete"))
    return {
        "available": True,
        "canary_passed": _safe_bool((value.get("canary") or {}).get("passed"))
            if type(value.get("canary")) is dict else None,
        "question_started": _safe_bool(value.get("question_started")),
        "question_completed": _safe_bool(value.get("question_completed")),
        "observed_turns": _safe_turns(value.get("observed_turns")),
        "known_tokens": _safe_count(value.get("known_tokens")),
        "usage_complete": False if flight is True else usage,
        "invocation_in_flight": flight,
        "indexing_outcome": value.get("indexing_outcome")
            if value.get("indexing_outcome") in ("success", "failure", None) else None,
        "cleanup_ok": _safe_bool(value.get("cleanup_ok")),
    }


def file_stat(path: Path) -> dict:
    try:
        stat = path.stat(follow_symlinks=False)
        return {"present": path.is_file() and not path.is_symlink(),
                "bytes": stat.st_size if stat.st_size >= 0 else None,
                "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"present": False, "bytes": None, "mtime_ns": None}


def unit_state(unit: str, expected_cgroup: str) -> dict:
    keys = ("ActiveState", "SubState", "Result", "MainPID", "ControlGroup",
            "NRestarts", "ExecMainStatus", "RuntimeMaxUSec", "KillMode", "Restart")
    result = subprocess.run(["systemctl", "--user", "show", unit,
                             "--property=" + ",".join(keys)],
                            capture_output=True, text=True, timeout=5, check=False)
    if result.returncode != 0 or len(result.stdout) > 8192:
        return {"available": False, "exact_cgroup": False}
    props = dict(line.split("=", 1) for line in result.stdout.splitlines()
                 if "=" in line and len(line) < 1024)
    active = props.get("ActiveState")
    actual_group = props.get("ControlGroup", "")
    pid = props.get("MainPID", "")
    restarts = props.get("NRestarts", "")
    return {"available": True,
            "active_state": active if active in SAFE_STATE else "unknown",
            "sub_state": props.get("SubState") if re.fullmatch(r"[a-z-]{1,40}", props.get("SubState", "")) else "unknown",
            "result": props.get("Result") if re.fullmatch(r"[a-z-]{1,40}", props.get("Result", "")) else "unknown",
            "main_pid": int(pid) if pid.isdecimal() else None,
            "restarts": int(restarts) if restarts.isdecimal() else None,
            "exit_status": int(props["ExecMainStatus"]) if props.get("ExecMainStatus", "").isdecimal() else None,
            "runtime_limit_ok": props.get("RuntimeMaxUSec") in
                {"5390000000", "1h 29min 50s"},
            "restart_disabled": props.get("Restart") == "no",
            "kill_mode_control_group": props.get("KillMode") == "control-group",
            "exact_cgroup": actual_group == expected_cgroup,
            "cgroup_header_empty": actual_group == "",
            "expected_cgroup_processes": cgroup_process_count(expected_cgroup)}


def cgroup_process_count(expected_cgroup: str) -> int | None:
    """Inspect only the caller-pinned unit subtree, never the cgroup root."""
    base = Path("/sys/fs/cgroup") / expected_cgroup.lstrip("/")
    if not base.exists():
        parent = base.parent
        return 0 if parent.is_dir() and not parent.is_symlink() else None
    if not base.is_dir() or base.is_symlink():
        return None
    count = 0
    visited = 0
    try:
        for current, dirs, files in os.walk(base, followlinks=False):
            visited += 1
            if visited > 1000:
                return None
            dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
            if "cgroup.procs" not in files:
                continue
            path = Path(current) / "cgroup.procs"
            if path.is_symlink() or path.stat().st_size > 1024 * 1024:
                return None
            with path.open("rb") as stream:
                count += sum(1 for _ in stream)
            if count > 100_000:
                return None
    except OSError:
        return None
    return count


def terminal_check(*, safe: dict | None, result: dict | None, row: dict | None,
                   pilot_path: Path, transport_path: Path, candidate: Path,
                   inventory_stamp: Path, inventory_sha256: str,
                   dataset: Path) -> dict:
    verdict = {"available": safe is not None, "validated": False,
               "source_pins_verified": False, "dataset_pin_verified": False,
               "indexing_valid": False, "correct": None}
    if safe is None or result is None or row is None:
        return verdict
    if (digest(pilot_path) != PILOT_SHA256 or digest(transport_path) != TRANSPORT_SHA256
            or digest(dataset) != DATASET_SHA256):
        return verdict
    verdict["dataset_pin_verified"] = True
    spec = importlib.util.spec_from_file_location("pinned_progress_pilot", pilot_path)
    if spec is None or spec.loader is None:
        return verdict
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    try:
        count = pilot.verify_inventory(candidate, inventory_stamp, inventory_sha256)
    except Exception:
        return verdict
    verdict["source_pins_verified"] = count == 508
    if not verdict["source_pins_verified"]:
        return verdict
    sys.path.insert(0, str(candidate))
    from benchmarks import lme_protocol
    indexing = row.get("indexing")
    try:
        verdict["indexing_valid"] = (type(indexing) is dict and
            lme_protocol._validate_versioned_indexing(indexing,
                require_healthy=True, allow_failure=False) is True)
    except Exception:
        return verdict
    summary = result.get("summary") if type(result.get("summary")) is dict else None
    result_row = result.get("row") if type(result.get("row")) is dict else None
    verdict["correct"] = row.get("correct") if type(row.get("correct")) is bool else None
    verdict["validated"] = bool(
        safe.get("ok") is True and safe.get("question_completed") is True
        and safe.get("canary_passed") is True
        and summary is not None and result_row is not None
        and summary.get("question_completed") is True
        and summary.get("cleanup_ok") is True
        and summary.get("usage_complete") is True
        and summary.get("invocation_in_flight") is False
        and _safe_turns(summary.get("observed_turns")) is not None
        and _safe_count(summary.get("known_tokens")) is not None
        and safe.get("observed_turns") == summary.get("observed_turns")
        and safe.get("known_tokens") == summary.get("known_tokens")
        and safe.get("usage_complete") is True
        and safe.get("invocation_in_flight") is False
        and safe.get("correct") == row.get("correct")
        and summary.get("correct") == row.get("correct")
        and summary.get("benchmark_failure") is None
        and row.get("benchmark_failure") is None
        and row.get("judge_parse_valid") is True
        and row.get("judge_error") is False
        and type(row.get("correct")) is bool
        and indexing.get("outcome") == "success"
        and indexing.get("healthy") is True
        and indexing.get("summary_healthy") is True
        and verdict["indexing_valid"]
        and result_row == row)
    if not verdict["validated"]:
        verdict["correct"] = None
    return verdict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for key in ("root", "unit", "expected-cgroup", "pilot", "transport", "candidate",
                "inventory-stamp", "inventory-sha256", "dataset", "launch-receipt",
                "launch-receipt-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    report = {"schema": "luna-lme-progress-v1", "read_only": True,
              "question_text_exported": False, "model_output_exported": False}
    try:
        root = Path(args.root)
        paths = [root, Path(args.pilot), Path(args.transport), Path(args.candidate),
                 Path(args.inventory_stamp), Path(args.dataset), Path(args.launch_receipt)]
        if (not all(path.is_absolute() for path in paths)
                or not UNIT.fullmatch(args.unit)
                or args.expected_cgroup !=
                    "/user.slice/user-1000.slice/user@1000.service/app.slice/" + args.unit
                or not HEX.fullmatch(args.inventory_sha256)
                or not HEX.fullmatch(args.launch_receipt_sha256)
                or digest(Path(args.launch_receipt)) != args.launch_receipt_sha256):
            raise ValueError("input_invalid")
        if (Path(args.pilot).parent != root or Path(args.transport).parent != root
                or Path(args.inventory_stamp).parent != root
                or Path(args.launch_receipt).parent != root
                or Path(args.pilot).name != "luna_subscription_pilot.py"
                or Path(args.transport).name != "codex_subscription.py"
                or Path(args.inventory_stamp).name != "headless-source-map.json"
                or digest(Path(args.pilot)) != PILOT_SHA256
                or digest(Path(args.transport)) != TRANSPORT_SHA256
                or digest(Path(args.inventory_stamp)) != args.inventory_sha256):
            raise ValueError("source_pin_invalid")
        receipt = read_json(Path(args.launch_receipt))
        sources = receipt.get("source_sha256") if type(receipt) is dict else None
        if (type(receipt) is not dict or type(sources) is not dict
                or receipt.get("schema") != "luna-lme-launch-v2"
                or receipt.get("unit") != args.unit
                or receipt.get("cgroup") != args.expected_cgroup
                or receipt.get("candidate") != str(Path(args.candidate))
                or receipt.get("dataset") != str(Path(args.dataset))
                or receipt.get("output") != str(root / "run-v2")
                or receipt.get("dataset_sha256") != DATASET_SHA256
                or receipt.get("model") != "gpt-6-luna"
                or receipt.get("question_limit") != 1
                or receipt.get("source_question_index") != 0
                or receipt.get("max_turns") != 1200
                or receipt.get("max_observed_tokens") != 4_000_000
                or receipt.get("max_total_seconds") != 5400
                or receipt.get("automatic_retry") is not False
                or receipt.get("production_changed") is not False
                or sources.get("luna_subscription_pilot.py") != PILOT_SHA256
                or sources.get("codex_subscription.py") != TRANSPORT_SHA256
                or sources.get("headless-source-map.json") != args.inventory_sha256):
            raise ValueError("launch_receipt_invalid")
        output = root / "run-v2"
        if not root.is_dir() or root.is_symlink() or output.is_symlink():
            raise ValueError("root_invalid")
        report["unit"] = unit_state(args.unit, args.expected_cgroup)
        progress_path = output / "private-progress.json"
        safe_path = root / "safe-terminal.json"
        report["progress"] = summarize_progress(read_json(progress_path))
        report["files"] = {name: file_stat(output / name) for name in
            ("private-progress.json", "private-result.json",
             "private-row.json", "private-run.log")}
        report["files"]["safe-terminal.json"] = file_stat(safe_path)
        stat = os.statvfs(root)
        report["disk_free_gib"] = round(stat.f_bavail * stat.f_frsize / 1024**3, 3)
        report["disk_floor_ok"] = report["disk_free_gib"] >= 20
        safe = read_json(safe_path)
        report["terminal"] = terminal_check(
            safe=safe,
            result=read_json(output / "private-result.json") if safe else None,
            row=read_json(output / "private-row.json") if safe else None,
            pilot_path=Path(args.pilot), transport_path=Path(args.transport),
            candidate=Path(args.candidate), inventory_stamp=Path(args.inventory_stamp),
            inventory_sha256=args.inventory_sha256, dataset=Path(args.dataset))
        unit = report["unit"]
        report["completed_and_clean"] = bool(
            report["terminal"]["validated"]
            and unit.get("available") is True
            and unit.get("active_state") == "active"
            and unit.get("sub_state") == "exited"
            and unit.get("result") == "success"
            and unit.get("main_pid") == 0
            and unit.get("restarts") == 0
            and unit.get("exit_status") == 0
            and unit.get("kill_mode_control_group") is True
            and unit.get("restart_disabled") is True
            and unit.get("runtime_limit_ok") is True
            and unit.get("expected_cgroup_processes") == 0)
        report["launch_receipt_pin_verified"] = True
    except Exception:
        report["error"] = "metadata_unavailable_or_invalid"
    print(json.dumps(report, sort_keys=True))
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
