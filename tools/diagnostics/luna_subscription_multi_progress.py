"""Metadata-only, read-only observer for one pinned two-question Luna run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

sys.dont_write_bytecode = True

RUNNER_SHA256 = "83fb27a9ee86c7f1d25ab7c6775dec31e9540321347971609de47a065734eef5"
PILOT_SHA256 = "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0"
BASE_SHA256 = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
CONCURRENT_SHA256 = "e5f1eacbb02f809ee6246449b672dfcc839069da226230572913612acece069e"
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
SOURCE_MAP_SHA256 = "35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51"
BINARY = Path("/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex")
LIMITS = {"campaign-turns": 4012, "campaign-known-tokens": 24160000,
          "campaign-seconds": 14400, "question-turns": 2000,
          "question-known-tokens": 12000000, "question-seconds": 12600,
          "canary-turns": 12, "canary-known-tokens": 160000,
          "canary-seconds": 600, "indexing-seconds": 10800,
          "questions": 2, "workers": 2}
MAX_JSON = 16 * 1024 * 1024
UNIT = re.compile(r"hymem-luna-lme-multi-[a-z0-9-]{1,48}\.service\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def bounded_json(path: Path) -> dict | None:
    try:
        if not regular(path) or path.stat().st_size > MAX_JSON:
            return None
        with path.open("rb") as stream:
            raw = stream.read(MAX_JSON + 1)
        if len(raw) > MAX_JSON:
            return None
        value = json.loads(raw)
        return value if type(value) is dict else None
    except (OSError, ValueError, UnicodeError):
        return None


def file_meta(path: Path) -> dict:
    try:
        state = path.stat(follow_symlinks=False)
        return {"present": regular(path), "bytes": state.st_size,
                "mtime_ns": state.st_mtime_ns}
    except OSError:
        return {"present": False, "bytes": None, "mtime_ns": None}


def count_processes(cgroup: str) -> int | None:
    """Never traverse the cgroup root or substitute a blank header path."""
    base = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    if not base.exists():
        parent = base.parent
        return 0 if parent.is_dir() and not parent.is_symlink() else None
    if not base.is_dir() or base.is_symlink():
        return None
    count = seen = 0
    try:
        for current, dirs, files in os.walk(base, followlinks=False):
            seen += 1
            if seen > 1000:
                return None
            dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
            if "cgroup.procs" in files:
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


def systemd_state(unit: str, cgroup: str) -> dict:
    keys = ("ActiveState", "SubState", "Result", "MainPID", "ControlGroup",
            "NRestarts", "ExecMainStatus", "ExecMainCode", "Restart", "KillMode", "Type",
            "RemainAfterExit", "RuntimeMaxUSec", "TimeoutStopUSec", "MemoryMax",
            "CPUQuotaPerSecUSec", "TasksMax", "OOMPolicy")
    try:
        call = subprocess.run(["systemctl", "--user", "show", unit,
            "--property=" + ",".join(keys)], capture_output=True, text=True,
            timeout=5, check=False)
        if call.returncode != 0 or len(call.stdout) > 8192:
            return {"available": False, "expected_cgroup_processes": count_processes(cgroup)}
        fields = dict(line.split("=", 1) for line in call.stdout.splitlines()
                      if "=" in line and len(line) < 1024)
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "expected_cgroup_processes": count_processes(cgroup)}
    def number(name):
        value = fields.get(name, "")
        return int(value) if value.isdecimal() else None
    actual = fields.get("ControlGroup", "")
    return {"available": True,
            "active_state": fields.get("ActiveState") if fields.get("ActiveState") in
                {"active", "activating", "deactivating", "inactive", "failed"} else "unknown",
            "sub_state": fields.get("SubState") if re.fullmatch(r"[a-z-]{1,40}", fields.get("SubState", "")) else "unknown",
            "result": fields.get("Result") if re.fullmatch(r"[a-z-]{1,40}", fields.get("Result", "")) else "unknown",
            "main_pid": number("MainPID"), "restarts": number("NRestarts"),
            "exit_status": number("ExecMainStatus"),
            "exit_code_kind": number("ExecMainCode"),
            "cgroup_header_matches": actual == cgroup,
            "cgroup_header_empty": actual == "",
            "expected_cgroup_processes": count_processes(cgroup),
            "policy_ok": (fields.get("Restart") == "no" and
                fields.get("KillMode") == "control-group" and
                fields.get("Type") == "exec" and
                fields.get("RemainAfterExit") == "yes" and
                fields.get("RuntimeMaxUSec") in {"14530000000", "4h 2min 10s"} and
                fields.get("TimeoutStopUSec") in {"10000000", "10s"} and
                fields.get("MemoryMax") in {"4294967296", "4G"} and
                fields.get("CPUQuotaPerSecUSec") in {"2000000", "2s"} and
                fields.get("TasksMax") == "128" and
                fields.get("OOMPolicy") == "kill")}


def safe_count(value, maximum=100_000_000):
    return value if type(value) is int and 0 <= value <= maximum else None


def progress_summary(value: dict | None) -> dict:
    if value is None:
        return {"available": False, "usage_complete": None, "known_tokens": None}
    budget = value.get("budget") if type(value.get("budget")) is dict else {}
    questions = value.get("questions") if type(value.get("questions")) is list else []
    entries = []
    for index in range(2):
        q = questions[index] if index < len(questions) and type(questions[index]) is dict else {}
        health = q.get("indexing_health") if type(q.get("indexing_health")) is dict else {}
        entries.append({"index": index, "started": q.get("question_started") is True,
            "completed": q.get("question_completed") is True,
            "correct": q.get("correct") if type(q.get("correct")) is bool and
                q.get("question_completed") is True else None,
            "cleanup_ok": q.get("cleanup_ok") if type(q.get("cleanup_ok")) is bool else None,
            "indexing_healthy": health.get("healthy") if type(health.get("healthy")) is bool else None,
            "summary_healthy": health.get("summary_healthy") if type(health.get("summary_healthy")) is bool else None,
            "last_cycle_stale": q.get("indexing_last_cycle_stale") if type(q.get("indexing_last_cycle_stale")) is bool else None,
            "pending_last_cycle": safe_count(health.get("pending")),
            "failed": q.get("stop_code") is not None if q else None})
    in_flight = safe_count(budget.get("in_flight"), 2)
    active = safe_count(value.get("active_invocations"), 2)
    usage = value.get("usage_complete_now") if type(value.get("usage_complete_now")) is bool else None
    return {"available": True, "canary_passed": (value.get("canary") or {}).get("passed") is True
            if type(value.get("canary")) is dict else None,
            "questions": entries, "turns": safe_count(budget.get("turns"), 4012),
            "indexing_healthy_count": sum(q["indexing_healthy"] is True for q in entries),
            "summary_healthy_count": sum(q["summary_healthy"] is True for q in entries),
            "last_cycle_stale_count": sum(q["last_cycle_stale"] is True for q in entries),
            "known_tokens": safe_count(budget.get("known_tokens")),
            "in_flight": in_flight, "active_invocations": active,
            "usage_complete": False if (in_flight or active) else usage,
            "campaign_stopped": value.get("campaign_stop") is not None,
            "known_tokens_scope": budget.get("known_tokens_scope") if
                budget.get("known_tokens_scope") ==
                "completed_turns_only_failed_turn_usage_unknown" else None}


def verify_terminal(*, root: Path, receipt: dict, safe: dict | None,
                    result: dict | None) -> dict:
    verdict = {"available": safe is not None, "validated": False,
               "pins_verified": False, "dataset_verified": False,
               "questions": [{"index": i, "validated": False, "correct": None}
                             for i in range(2)]}
    if safe is None or result is None:
        return verdict
    source = {"luna_subscription_lme_multi.py": RUNNER_SHA256,
              "luna_subscription_pilot.py": PILOT_SHA256,
              "codex_subscription.py": BASE_SHA256,
              "codex_subscription_concurrent.py": CONCURRENT_SHA256,
              "headless-source-map.json": receipt["inventory_sha256"]}
    if any(not regular(root / name) or digest(root / name) != expected
           for name, expected in source.items()):
        return verdict
    if digest(Path(receipt["dataset"])) != DATASET_SHA256:
        return verdict
    verdict["dataset_verified"] = True
    # Load verified pilot bytes without importing the candidate first.
    path = root / "luna_subscription_pilot.py"
    code = path.read_bytes()
    if hashlib.sha256(code).hexdigest() != PILOT_SHA256:
        return verdict
    pilot = types.ModuleType("pinned_multi_progress_pilot")
    pilot.__file__ = str(path)
    exec(compile(code, str(path), "exec"), pilot.__dict__)
    try:
        count = pilot.verify_inventory(Path(receipt["candidate"]),
            root / "headless-source-map.json", receipt["inventory_sha256"])
    except Exception:
        return verdict
    verdict["pins_verified"] = count == 508
    if not verdict["pins_verified"]:
        return verdict
    sys.path.insert(0, receipt["candidate"])
    from benchmarks import lme_protocol
    if not Path(lme_protocol.__file__).resolve().is_relative_to(
            Path(receipt["candidate"]).resolve()):
        return verdict
    entries = result.get("questions")
    if type(entries) is not list or len(entries) != 2:
        return verdict
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            continue
        qdir = root / "run" / f"q-{index:04d}"
        if qdir.is_symlink():
            continue
        private = bounded_json(qdir / "private-result.json")
        row = bounded_json(qdir / "private-row.json")
        if private is None or row is None or private != entry:
            continue
        indexing = row.get("indexing")
        if type(indexing) is not dict:
            continue
        try:
            valid_indexing = lme_protocol._validate_versioned_indexing(
                indexing, require_healthy=True, allow_failure=False) is True
        except Exception:
            continue
        house = entry.get("dream_run_housekeeping")
        valid = (entry.get("index") == index
            and entry.get("question_completed") is True
            and entry.get("cleanup_ok") is True
            and entry.get("stop_code") is None
            and type(entry.get("correct")) is bool
            and row.get("correct") == entry.get("correct")
            and row.get("benchmark_failure") is None
            and row.get("judge_error") is False
            and row.get("judge_parse_valid") is True
            and indexing.get("outcome") == "success"
            and indexing.get("healthy") is True
            and indexing.get("summary_healthy") is True
            and valid_indexing
            and type(house) is dict
            and house.get("open_runs_before") == 0
            and house.get("terminalized") == 0
            and house.get("open_runs_after") == 0
            and house.get("active_leases_after") == 0)
        verdict["questions"][index] = {"index": index, "validated": bool(valid),
            "correct": entry["correct"] if valid else None}
    budget = result.get("budget") if type(result.get("budget")) is dict else {}
    verdict["validated"] = bool(
        safe.get("ok") is True and safe.get("questions_selected") == 2
        and safe.get("questions_completed") == 2
        and safe.get("source_files_verified") == 508
        and safe.get("dataset_sha256") == DATASET_SHA256
        and safe.get("runner_sha256") == RUNNER_SHA256
        and safe.get("pilot_helper_sha256") == PILOT_SHA256
        and safe.get("base_transport_sha256") == BASE_SHA256
        and safe.get("concurrent_transport_sha256") == CONCURRENT_SHA256
        and safe.get("canary_passed") is True
        and result.get("schema") == "luna-subscription-lme-multi-v1"
        and result.get("campaign_stop") is None
        and (result.get("canary") or {}).get("passed") is True
        and result.get("usage_complete_now") is True
        and budget.get("usage_complete") is True
        and budget.get("in_flight") == budget.get("reserved") == 0
        and result.get("active_invocations") == 0
        and safe.get("turns") == budget.get("turns")
        and safe.get("known_tokens") == budget.get("known_tokens")
        and safe.get("usage_complete") is True
        and safe.get("in_flight") == 0
        and safe.get("correct_count") == sum(q["correct"] is True for q in verdict["questions"])
        and safe.get("incorrect_count") == sum(q["correct"] is False for q in verdict["questions"])
        and safe.get("failed_count") == 0
        and safe.get("not_started_count") == 0
        and all(q["validated"] for q in verdict["questions"]))
    return verdict


def verify_live_pins(root: Path, receipt: dict) -> dict:
    """Check all frozen source/dataset bytes before reporting a live status."""
    source = receipt["source_sha256"]
    if any(not regular(root / name) or digest(root / name) != expected
           for name, expected in source.items()):
        return {"source_pins_verified": False, "dataset_pin_verified": False,
                "inventory_verified": False}
    dataset = Path(receipt["dataset"])
    if not regular(dataset) or digest(dataset) != DATASET_SHA256:
        return {"source_pins_verified": True, "dataset_pin_verified": False,
                "inventory_verified": False}
    path = root / "luna_subscription_pilot.py"
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != PILOT_SHA256:
        return {"source_pins_verified": False, "dataset_pin_verified": True,
                "inventory_verified": False}
    pilot = types.ModuleType("pinned_multi_live_pilot")
    pilot.__file__ = str(path)
    exec(compile(content, str(path), "exec"), pilot.__dict__)
    try:
        count = pilot.verify_inventory(Path(receipt["candidate"]),
            root / "headless-source-map.json", receipt["inventory_sha256"])
    except Exception:
        count = 0
    return {"source_pins_verified": True, "dataset_pin_verified": True,
            "inventory_verified": count == 508}


def completed_and_clean(terminal: dict, unit: dict) -> bool:
    return bool(terminal.get("validated") is True
        and unit.get("available") is True
        and unit.get("active_state") == "active"
        and unit.get("sub_state") == "exited"
        and unit.get("result") == "success"
        and unit.get("main_pid") == 0
        and unit.get("restarts") == 0
        and unit.get("exit_status") == 0
        and unit.get("exit_code_kind") == 1
        and unit.get("policy_ok") is True
        and (unit.get("cgroup_header_matches") is True or
             unit.get("cgroup_header_empty") is True)
        and unit.get("expected_cgroup_processes") == 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("root", "unit", "expected-cgroup", "receipt-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    report = {"schema": "luna-lme-multi-progress-v1", "read_only": True,
              "raw_text_exported": False, "completed_and_clean": False}
    try:
        root = Path(args.root)
        unit = args.unit
        cgroup = args.expected_cgroup
        if (not root.is_absolute() or root.parent != Path("/home/atta")
                or not root.name.startswith(".hymem-luna-lme-multi-")
                or root.is_symlink() or not root.is_dir()
                or not UNIT.fullmatch(unit)
                or cgroup != "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit
                or not HEX.fullmatch(args.receipt_sha256)):
            raise ValueError("input_invalid")
        receipt_path = root / "launch-receipt.json"
        if not regular(receipt_path) or digest(receipt_path) != args.receipt_sha256:
            raise ValueError("receipt_pin_invalid")
        receipt = bounded_json(receipt_path)
        if type(receipt) is not dict or type(receipt.get("source_sha256")) is not dict:
            raise ValueError("receipt_invalid")
        expected_sources = {"luna_subscription_lme_multi.py": RUNNER_SHA256,
            "luna_subscription_pilot.py": PILOT_SHA256,
            "codex_subscription.py": BASE_SHA256,
            "codex_subscription_concurrent.py": CONCURRENT_SHA256,
            "headless-source-map.json": receipt.get("inventory_sha256")}
        if (receipt.get("root") != str(root) or receipt.get("unit") != unit
                or receipt.get("schema") != "luna-multi-launch-v1"
                or receipt.get("expected_cgroup") != cgroup
                or receipt.get("output") != "run"
                or receipt.get("runner_sha256") != RUNNER_SHA256
                or receipt.get("source_sha256") != expected_sources
                or receipt.get("dataset_sha256") != DATASET_SHA256
                or receipt.get("runtime_max_seconds") != 14530
                or receipt.get("timeout_stop_seconds") != 10
                or receipt.get("memory_max_bytes") != 4294967296
                or receipt.get("cpu_quota_percent") != 200
                or receipt.get("tasks_max") != 128
                or receipt.get("model") != "gpt-6-luna"
                or receipt.get("subscription_only") is not True
                or receipt.get("limits") != LIMITS
                or receipt.get("binary") != str(BINARY)
                or not HEX.fullmatch(str(receipt.get("binary_sha256", "")))
                or not regular(BINARY)
                or digest(BINARY) != receipt.get("binary_sha256")
                or not HEX.fullmatch(str(receipt.get("inventory_sha256", "")))
                or not Path(str(receipt.get("candidate", ""))).is_absolute()
                or not Path(str(receipt.get("dataset", ""))).is_absolute()
                or receipt.get("inventory_stamp") != str(root / "headless-source-map.json")):
            raise ValueError("receipt_invalid")
        if any(not regular(root / name) or digest(root / name) != expected
               for name, expected in expected_sources.items()):
            raise ValueError("source_pin_invalid")
        report["live_pins"] = verify_live_pins(root, receipt)
        if not all(report["live_pins"].values()):
            raise ValueError("live_pin_invalid")
        output = root / "run"
        if output.is_symlink():
            raise ValueError("output_path_invalid")
        report["unit"] = systemd_state(unit, cgroup)
        progress_path = output / "private-progress.json"
        safe_path = root / "safe-terminal.json"
        report["progress"] = progress_summary(bounded_json(progress_path))
        report["files"] = {name: file_meta(output / name) for name in
            ("private-progress.json", "private-result.json", "private-run.log")}
        report["files"]["safe-terminal.json"] = file_meta(safe_path)
        for index in range(2):
            report["files"][f"q-{index:04d}/private-result.json"] = file_meta(
                output / f"q-{index:04d}" / "private-result.json")
            report["files"][f"q-{index:04d}/private-row.json"] = file_meta(
                output / f"q-{index:04d}" / "private-row.json")
        disk = os.statvfs(root)
        report["disk_free_gib"] = round(disk.f_bavail * disk.f_frsize / 1024**3, 3)
        report["disk_floor_ok"] = report["disk_free_gib"] >= 20
        safe = bounded_json(safe_path)
        result = bounded_json(output / "private-result.json") if safe else None
        report["terminal"] = verify_terminal(root=root, receipt=receipt,
            safe=safe, result=result)
        report["completed_and_clean"] = completed_and_clean(
            report["terminal"], report["unit"])
        report["receipt_pin_verified"] = True
    except Exception:
        report["error"] = "metadata_unavailable_or_invalid"
    print(json.dumps(report, sort_keys=True))
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
