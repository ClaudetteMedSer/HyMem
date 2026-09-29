"""Read-only, source-free observer for the capacity-adjusted Luna run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

sys.dont_write_bytecode = True

PROFILED = "53628ac7e9c6107bb68d1cd4ebdf42d5a129b96c4d96360d4e7a6525be7bc739"
WARM_RUNNER = "3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567"
COLLECTOR = "800ef9baedc9d68093b3160cc324ef17528dd0b89f7a734f220217b07323fba2"
WARM = "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593"
CONCURRENT = "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0"
PILOT = "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0"
BASE = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
DATASET = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
SOURCE_MAP = "35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51"
BINARY = Path("/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex")
CGROUP_ROOT = Path("/sys/fs/cgroup")
LIMITS = {"campaign-turns": 8012, "campaign-known-tokens": 48160000,
          "campaign-seconds": 14400, "question-turns": 2000,
          "question-known-tokens": 12000000, "question-seconds": 12600,
          "canary-turns": 12, "canary-known-tokens": 160000,
          "canary-seconds": 600, "indexing-seconds": 10800,
          "questions": 4, "workers": 4,
          "warm-max-requests": 16, "warm-max-age-seconds": 300}
INVENTORY_SHA256 = "852758cfa63902048669f78f2377db4354b2e196a16ccbb98a3e6cadca2590eb"
STAGES = frozenset(("extraction_primary_or_other", "extraction_contract_repair",
    "extraction_terminal_retry", "extraction_empty_verifier",
    "extraction_omission_verifier", "digest_primary", "digest_summary_repair",
    "reader", "judge", "unclassified"))
COUNTS = ("attempted_calls", "admitted_turns", "known_tokens", "returned_calls", "rejected_calls")
TIMINGS = ("invocation_wall_seconds",)
WARM_COUNTS = ("processes_started", "rotations", "cold_calls", "warm_calls")
WARM_TIMINGS = ("startup_seconds", "unsubscribe_seconds", "rotation_cleanup_seconds", "final_cleanup_seconds")
HEX = re.compile(r"[0-9a-f]{64}\Z")
UNIT = re.compile(r"hymem-luna-lme-capacity-[A-Za-z0-9_-]{8,}\.service\Z")
MAX_JSON = 16 * 1024 * 1024
STOP_CODES = frozenset(("questions_incomplete", "campaign_failure", "canary_failed",
    "canary_runtime_failure", "canary_cleanup_failure", "canary_wall_limit",
    "question_runtime_failure", "indexing_convergence_failure", "indexing_unhealthy",
    "question_unscored", "usage_incomplete", "wall_limit_after_score",
    "cleanup_failure", "warm_client_cleanup_failure", "dream_run_housekeeping_failure",
    "worker_interrupted", "controller_interrupted", "stage_accounting_failure",
    "stage_accounting_mismatch", "artifact_write_failure", "dataset_stream_failure",
    "private_progress_unavailable"))
FIXED_FAILURE_CODES = frozenset((
    "account_changed", "account_notification_invalid", "active_thread_closed",
    "binary_version_mismatch", "binary_version_unverified",
    "capability_isolation_unverified", "credit_balance_present", "duplicate_item",
    "event_limit", "exact_model_unavailable", "final_message_invalid",
    "incomplete_model_catalog", "incomplete_turn_or_usage", "instruction_isolation_unverified",
    "invalid_quota", "invalid_request", "item_id_invalid", "item_identity_mismatch",
    "item_lifecycle_invalid", "mcp_isolation_unverified", "message_delta_invalid",
    "model_or_search_unverified", "notification_invalid", "output_limit",
    "pilot_budget_exhausted", "pilot_stopped", "process_not_reaped", "protocol_failure",
    "provider_endpoint_override", "quota_exhausted", "quota_floor",
    "reasoning_delta_invalid", "remote_control_enabled", "retired_thread_closed_invalid",
    "retired_thread_status_invalid", "routing_unverified", "runtime_isolation_not_accepted",
    "service_tier_unverified", "startup_failed", "subscription_auth_required",
    "subscription_plan_unverified", "thread_id_missing", "thread_identity_mismatch",
    "thread_isolation_unverified", "thread_start_status_invalid", "timeout",
    "tool_or_extra_item", "turn_failed", "turn_identity_mismatch", "turn_start_invalid",
    "unexpected_response", "unexpected_server_request", "unknown_quota",
    "unsubscribe_thread_mismatch", "unsubscribe_unverified", "unsupported_chat_control",
    "unsupported_chat_history", "usage_identity_mismatch", "usage_invalid",
    "usage_regressed", "warning_limit", "transport_failure", "warning_thread_unverified",
    "warning_unapproved", "concurrent_completion_rejected", "fixed_other",
    "ledger_protocol_violation", "wall_limit", "campaign_wall_limit",
    "campaign_budget_exhausted", "question_budget_exhausted",
    "budget_exhausted_before_turn", "admission_rejected", "quota_unverified",
    "cleanup_failure",
))
RPC_METHODS = frozenset(("initialize", "config/read", "model/list", "account/read",
    "account/rateLimits/read", "thread/start", "turn/start", "thread/unsubscribe",
    "startup", "turn/events"))
EVENT_METHODS = frozenset(("thread/started", "thread/status/changed", "thread/closed",
    "turn/started", "turn/completed", "turn/failed", "item/started", "item/completed",
    "item/agentMessage/delta", "item/reasoning/textDelta", "thread/tokenUsage/updated",
    "account/updated", "account/rateLimits/updated", "remoteControl/status/changed",
    "warning", "invalid_method"))
PHASES = frozenset(("startup", "rotation_cleanup", "preflight", "run",
    "unsubscribe", "cleanup"))
QUESTION_STOP_CODES = STOP_CODES | FIXED_FAILURE_CODES | frozenset((
    "campaign_stopped", "campaign_stopped_before_question", "transport_or_admission_failure",
    "interrupted_invocation", "incomplete_answer", "invalid_request_or_closed",
    "close_during_invocation", "question_stopped", "question_concurrent_invocation",
    "concurrency_limit", "budget_stopped_before_turn", "worker_unhandled_failure",
    "worker_unhandled_stop", "open_dream_run_unresolved", "active_dream_lease",
    "owned_store_identity_invalid", "output_not_fresh", "question_count_invalid",
    "dataset_item_invalid", "dataset_item_limit", "dataset_shape_invalid",
    "dataset_short", "selected_row_drift", "warm_lifetime_invalid",
))


def fixed_code(value):
    if type(value) is not str or len(value) > 100:
        return None
    if value in FIXED_FAILURE_CODES:
        return value
    parts = value.split(":", 1)
    if len(parts) != 2 or parts[0] not in {
            "process_exit", "protocol_failure", "rpc_failure", "unexpected_notification"}:
        return None
    allowed = EVENT_METHODS if parts[0] == "unexpected_notification" else RPC_METHODS
    return value if parts[1] in allowed else None


def first_failure_summary(value):
    """Copy only bounded atoms defined by the attributed transport."""
    if type(value) is not dict or fixed_code(value.get("code")) is None:
        return None
    phase, rpc = value.get("phase"), value.get("rpc")
    if type(phase) is not str or phase not in PHASES or (
            rpc is not None and (type(rpc) is not str or rpc not in RPC_METHODS)):
        return None
    result = {"code": value["code"], "phase": phase, "rpc": rpc}
    for key in ("process_index", "request_index", "retired_count", "queue_count", "known_tokens"):
        item = value.get(key)
        if type(item) is not int or not 0 <= item <= 10**12:
            return None
        result[key] = item
    age = value.get("process_age_seconds")
    if age is not None and (type(age) not in (int, float) or
            not math.isfinite(age) or not 0 <= age <= 1_000_000):
        return None
    result["process_age_seconds"] = age
    for key in ("turn_admitted", "known_usage", "usage_complete"):
        if type(value.get(key)) is not bool:
            return None
        result[key] = value[key]
    return result


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


def safe_int(value, limit=100_000_000):
    return value if type(value) is int and 0 <= value <= limit else None


def safe_float(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def warm_summary(value):
    if type(value) is not dict:
        return None
    out = {key: safe_int(value.get(key)) for key in WARM_COUNTS}
    out.update({key: safe_float(value.get(key)) for key in WARM_TIMINGS})
    return out if all(item is not None for item in out.values()) else None


def stage_summary(value):
    """Whitelist fixed labels and finite numeric aggregates; reject malformed slots."""
    if type(value) is not dict or set(value) - {"canary", *(f"q-{i:04d}" for i in range(4))}:
        return None
    out = {}
    for question, stages in value.items():
        if type(stages) is not dict or set(stages) - STAGES:
            return None
        out[question] = {}
        for stage, slot in stages.items():
            if type(slot) is not dict:
                return None
            values = {key: safe_int(slot.get(key)) for key in COUNTS}
            values.update({key: safe_float(slot.get(key)) for key in TIMINGS})
            values["usage_complete"] = slot.get("usage_complete") if type(slot.get("usage_complete")) is bool else None
            if (any(item is None for item in values.values()) or
                    values["returned_calls"] + values["rejected_calls"] != values["attempted_calls"] or
                    values["admitted_turns"] > values["attempted_calls"]):
                return None
            out[question][stage] = values
    return out


def stage_totals(stages):
    if stages is None:
        return None
    totals = {key: 0 for key in COUNTS}
    totals["invocation_wall_seconds"] = 0.0
    totals["usage_complete"] = True
    for group in stages.values():
        for slot in group.values():
            for key in COUNTS + TIMINGS:
                totals[key] += slot[key]
            totals["usage_complete"] &= slot["usage_complete"]
    return totals


def stage_reconciled(stages, budget):
    if stages is None or type(budget) is not dict or type(budget.get("questions")) is not dict:
        return False
    totals = stage_totals(stages)
    if totals["admitted_turns"] != budget.get("turns") or totals["known_tokens"] != budget.get("known_tokens"):
        return False
    if set(stages) - set(budget["questions"]):
        return False
    for key, state in budget["questions"].items():
        if key not in {"canary", *(f"q-{i:04d}" for i in range(4))} or type(state) is not dict:
            return False
        group = stages.get(key, {})
        if (sum(slot["admitted_turns"] for slot in group.values()) != state.get("turns") or
                sum(slot["known_tokens"] for slot in group.values()) != state.get("known_tokens")):
            return False
    return True


def progress_summary(value):
    if value is None:
        return {"available": False, "questions": None, "usage_complete": None,
                "stage_accounting": None, "stage_totals": None, "first_failure": None}
    budget = value.get("budget") if type(value.get("budget")) is dict else {}
    question_budget = budget.get("questions") if type(budget.get("questions")) is dict else {}
    questions = value.get("questions") if type(value.get("questions")) is list else []
    entries = []
    for index in range(4):
        q = questions[index] if index < len(questions) and type(questions[index]) is dict else {}
        health = q.get("indexing_health") if type(q.get("indexing_health")) is dict else {}
        key = f"q-{index:04d}"
        q_budget = question_budget.get(key) if type(question_budget.get(key)) is dict else {}
        warm = warm_summary(q.get("transport"))
        admitted = safe_int(q_budget.get("turns"), 8012)
        attempts = warm["cold_calls"] + warm["warm_calls"] if warm is not None else None
        raw_stop = q.get("stop_code")
        stop = (raw_stop if raw_stop in QUESTION_STOP_CODES else
            (fixed_code(raw_stop) if fixed_code(raw_stop) is not None else
             ("other" if raw_stop is not None else None))) if type(raw_stop) is str or raw_stop is None else "other"
        entries.append({"index": index, "started": q.get("question_started") is True,
            "completed": q.get("question_completed") is True,
            "correct": q.get("correct") if type(q.get("correct")) is bool and q.get("question_completed") is True else None,
            "cleanup_ok": q.get("cleanup_ok") if type(q.get("cleanup_ok")) is bool else None,
            "indexing_healthy": health.get("healthy") if type(health.get("healthy")) is bool else None,
            "summary_healthy": health.get("summary_healthy") if type(health.get("summary_healthy")) is bool else None,
            "indexing_cycles": safe_int(health.get("cycles"), 100),
            "pending_last_cycle": safe_int(health.get("pending")),
            "malformed_last_cycle": safe_int(health.get("malformed")),
            "quarantined_last_cycle": safe_int(health.get("quarantined")),
            "terminal_loss_chunks": safe_int(health.get("terminal_loss_chunks")),
            "coverage_integrity_failures": safe_int(health.get("coverage_integrity_failures")),
            "last_cycle_stale": q.get("indexing_last_cycle_stale") if type(q.get("indexing_last_cycle_stale")) is bool else None,
            "failed": raw_stop is not None if q else None, "stop_code": stop,
            "admitted_turns": admitted, "transport_attempts": attempts,
            "attempts_minus_admitted": attempts - admitted if attempts is not None and admitted is not None else None,
            "warm_transport": warm})
    stages = stage_summary(value.get("stage_accounting"))
    canary_warm = warm_summary(value.get("canary_transport"))
    canary_budget = question_budget.get("canary") if type(question_budget.get("canary")) is dict else {}
    canary_admitted = safe_int(canary_budget.get("turns"), 8012)
    canary_attempts = (canary_warm["cold_calls"] + canary_warm["warm_calls"]
                      if canary_warm is not None else None)
    in_flight = safe_int(budget.get("in_flight"), 4)
    active = safe_int(value.get("active_invocations"), 4)
    usage = value.get("usage_complete_now") if type(value.get("usage_complete_now")) is bool else None
    return {"available": True, "questions": entries,
        "canary_passed": value.get("canary", {}).get("passed") is True if type(value.get("canary")) is dict else None,
        "canary_warm_transport": canary_warm,
        "canary_admitted_turns": canary_admitted,
        "canary_transport_attempts": canary_attempts,
        "canary_attempts_minus_admitted": (canary_attempts - canary_admitted
            if canary_attempts is not None and canary_admitted is not None else None),
        "turns": safe_int(budget.get("turns"), 8012), "known_tokens": safe_int(budget.get("known_tokens")),
        "in_flight": in_flight, "active_invocations": active,
        "usage_complete": False if (in_flight or active) else usage,
        "campaign_stopped": value.get("campaign_stop") is not None,
        "first_failure": first_failure_summary(budget.get("first_failure")),
        "first_failure_valid": budget.get("first_failure") is None or first_failure_summary(budget.get("first_failure")) is not None,
        "completed_count": sum(item["completed"] for item in entries),
        "correct_count": sum(item["correct"] is True for item in entries),
        "incorrect_count": sum(item["correct"] is False for item in entries),
        "known_tokens_scope": budget.get("known_tokens_scope") if budget.get("known_tokens_scope") ==
            "completed_turns_only_failed_turn_usage_unknown" else None,
        "stage_accounting_available": stages is not None,
        "stage_accounting": stages, "stage_totals": stage_totals(stages),
        "stage_reconciled_now": stage_reconciled(stages, budget) if stages is not None else None}


def count_processes(cgroup):
    base = CGROUP_ROOT / cgroup.lstrip("/")
    if not base.exists():
        return 0 if base.parent.is_dir() and not base.parent.is_symlink() else None
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
                count += sum(1 for _ in path.open("rb"))
                if count > 100000:
                    return None
    except OSError:
        return None
    return count


def cgroup_task_counters(cgroup):
    """Live cgroup v2 counters; absent terminal cgroups have unknown history."""
    keys = ("pids_current", "pids_peak", "pids_max", "pids_events_max")
    unknown = dict.fromkeys(keys)
    base = CGROUP_ROOT / cgroup.lstrip("/")
    if not base.is_dir() or base.is_symlink():
        return unknown

    def number(path):
        try:
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 4096:
                return None
            raw = path.read_text(encoding="ascii").strip()
            return int(raw) if raw.isdecimal() and len(raw) <= 12 else None
        except (OSError, UnicodeError):
            return None

    events = None
    event_path = base / "pids.events"
    try:
        if event_path.is_file() and not event_path.is_symlink() and event_path.stat().st_size <= 4096:
            lines = event_path.read_text(encoding="ascii").splitlines()
            entries = [line.split() for line in lines]
            if all(len(entry) == 2 for entry in entries) and sum(entry[0] == "max" for entry in entries) == 1:
                raw = next(entry[1] for entry in entries if entry[0] == "max")
                events = int(raw) if raw.isdecimal() and len(raw) <= 12 else None
    except (OSError, UnicodeError):
        pass
    return {"pids_current": number(base / "pids.current"),
            "pids_peak": number(base / "pids.peak"),
            "pids_max": number(base / "pids.max"),
            "pids_events_max": events}


def systemd_state(unit, cgroup):
    keys = ("ActiveState", "SubState", "Result", "MainPID", "ControlGroup", "NRestarts",
        "ExecMainStatus", "ExecMainCode", "Restart", "KillMode", "Type", "RemainAfterExit",
        "RuntimeMaxUSec", "TimeoutStopUSec", "MemoryMax", "CPUQuotaPerSecUSec", "TasksMax", "OOMPolicy",
        "ExecMainStartTimestampMonotonic", "ExecMainExitTimestampMonotonic")
    try:
        call = subprocess.run(["systemctl", "--user", "show", unit,
            "--property=" + ",".join(keys)], capture_output=True, text=True, timeout=5, check=False)
        if call.returncode != 0 or len(call.stdout) > 8192:
            return {"available": False, "expected_cgroup_processes": count_processes(cgroup),
                    "cgroup_tasks": cgroup_task_counters(cgroup)}
        fields = dict(line.split("=", 1) for line in call.stdout.splitlines() if "=" in line and len(line) < 1024)
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "expected_cgroup_processes": count_processes(cgroup),
                "cgroup_tasks": cgroup_task_counters(cgroup)}
    def number(key):
        raw = fields.get(key, "")
        return int(raw) if raw.isdecimal() else None
    actual = fields.get("ControlGroup", "")
    start = number("ExecMainStartTimestampMonotonic")
    end = number("ExecMainExitTimestampMonotonic")
    if start is not None and start > 0:
        elapsed = ((end if end is not None and end >= start else time.monotonic_ns() // 1000) - start) / 1_000_000
        elapsed = safe_float(elapsed)
    else:
        elapsed = None
    return {"available": True,
        "active_state": fields.get("ActiveState") if fields.get("ActiveState") in
            {"active", "activating", "deactivating", "inactive", "failed"} else "unknown",
        "sub_state": fields.get("SubState") if re.fullmatch(r"[a-z-]{1,40}", fields.get("SubState", "")) else "unknown",
        "result": fields.get("Result") if re.fullmatch(r"[a-z-]{1,40}", fields.get("Result", "")) else "unknown",
        "main_pid": number("MainPID"), "restarts": number("NRestarts"),
        "exit_status": number("ExecMainStatus"), "exit_code_kind": number("ExecMainCode"),
        "cgroup_header_matches": actual == cgroup, "cgroup_header_empty": actual == "",
        "expected_cgroup_processes": count_processes(cgroup),
        "cgroup_tasks": cgroup_task_counters(cgroup),
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        "policy_ok": (fields.get("Restart") == "no" and fields.get("KillMode") == "control-group" and
            fields.get("Type") == "exec" and fields.get("RemainAfterExit") == "yes" and
            fields.get("RuntimeMaxUSec") in {"14530000000", "4h 2min 10s"} and
            fields.get("TimeoutStopUSec") in {"10000000", "10s"} and
            fields.get("MemoryMax") in {"4294967296", "4G"} and
            fields.get("CPUQuotaPerSecUSec") in {"2000000", "2s"} and
            fields.get("TasksMax") == "256" and fields.get("OOMPolicy") == "kill")}


def verify_live_pins(root, receipt):
    source = receipt["source_sha256"]
    if any(not regular(root / name) or digest(root / name) != expected for name, expected in source.items()):
        return {"source_pins_verified": False, "dataset_pin_verified": False, "inventory_verified": False}
    dataset = Path(receipt["dataset"])
    if not regular(dataset) or digest(dataset) != DATASET:
        return {"source_pins_verified": True, "dataset_pin_verified": False, "inventory_verified": False}
    path = root / "luna_subscription_pilot.py"
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != PILOT:
        return {"source_pins_verified": False, "dataset_pin_verified": True, "inventory_verified": False}
    pilot = types.ModuleType("pinned_profiled_live_pilot")
    pilot.__file__ = str(path)
    exec(compile(content, str(path), "exec"), pilot.__dict__)
    try:
        count = pilot.verify_inventory(Path(receipt["candidate"]), root / "headless-source-map.json", receipt["inventory_sha256"])
    except Exception:
        count = 0
    return {"source_pins_verified": True, "dataset_pin_verified": True, "inventory_verified": count == 508}


def verify_terminal(root, receipt, safe, result):
    verdict = {"available": safe is not None, "validated": False,
        "questions": [{"index": index, "validated": False, "correct": None} for index in range(4)],
        "stage_accounting_reconciled": False, "warm_metrics_valid": False}
    if safe is None or result is None:
        return verdict
    entries = result.get("questions")
    if type(entries) is not list or len(entries) != 4:
        return verdict
    candidate = Path(receipt["candidate"])
    sys.path.insert(0, str(candidate))
    try:
        from benchmarks import lme_protocol
        if not Path(lme_protocol.__file__).resolve().is_relative_to(candidate.resolve()):
            return verdict
    except (ImportError, OSError, AttributeError):
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
        transport = warm_summary(entry.get("transport"))
        budget = result.get("budget") if type(result.get("budget")) is dict else {}
        budget_questions = budget.get("questions") if type(budget.get("questions")) is dict else {}
        q_budget = budget_questions.get(f"q-{index:04d}")
        valid = (entry.get("index") == index and entry.get("question_started") is True
            and entry.get("question_completed") is True and entry.get("cleanup_ok") is True
            and entry.get("stop_code") is None and type(entry.get("correct")) is bool
            and type(row.get("correct")) is bool and row.get("correct") == entry.get("correct")
            and row.get("benchmark_failure") is None and row.get("judge_error") is False
            and row.get("judge_parse_valid") is True and indexing.get("outcome") == "success"
            and indexing.get("healthy") is True and indexing.get("summary_healthy") is True
            and valid_indexing and type(house) is dict
            and house.get("open_runs_before") == 0 and house.get("terminalized") == 0
            and house.get("open_runs_after") == 0 and house.get("active_leases_after") == 0
            and transport is not None and transport["processes_started"] >= 1
            and transport["cold_calls"] >= 1
            and type(q_budget) is dict
            and transport["cold_calls"] + transport["warm_calls"] == q_budget.get("turns")
            and transport["processes_started"] <= transport["cold_calls"])
        verdict["questions"][index] = {"index": index, "validated": bool(valid),
            "correct": entry["correct"] if valid else None}
    stages = stage_summary(result.get("stage_accounting"))
    budget = result.get("budget") if type(result.get("budget")) is dict else {}
    totals = stage_totals(stages)
    verdict["stage_accounting_reconciled"] = bool(
        result.get("stage_accounting_reconciled") is True and
        stage_reconciled(stages, budget) and totals is not None and totals["usage_complete"] is True)
    canary_warm = warm_summary(result.get("canary_transport"))
    budget_questions = budget.get("questions") if type(budget.get("questions")) is dict else {}
    canary_budget = budget_questions.get("canary")
    verdict["warm_metrics_valid"] = bool(canary_warm is not None and
        canary_warm["processes_started"] >= 1 and canary_warm["cold_calls"] >= 1 and
        canary_warm["processes_started"] <= canary_warm["cold_calls"] and
        type(canary_budget) is dict and
        canary_warm["cold_calls"] + canary_warm["warm_calls"] == canary_budget.get("turns") and
        all(item["validated"] for item in verdict["questions"]))
    verdict["validated"] = bool(
        safe.get("schema") == "luna-subscription-lme-profiled-v2"
        and safe.get("ok") is True and safe.get("stop_code") is None
        and safe.get("questions_selected") == safe.get("questions_completed") == 4
        and safe.get("source_files_verified") == 508
        and safe.get("dataset_sha256") == DATASET
        and safe.get("candidate_source_map_sha256") == SOURCE_MAP
        and safe.get("pilot_helper_sha256") == PILOT
        and safe.get("runner_sha256") == PROFILED
        and safe.get("base_transport_sha256") == BASE
        and safe.get("concurrent_transport_sha256") == CONCURRENT
        and safe.get("warm_transport_sha256") == WARM
        and safe.get("model") == "gpt-6-luna"
        and safe.get("transport_kind") == "warm_process_fresh_ephemeral_thread"
        and safe.get("warm_max_requests") == 16 and safe.get("warm_max_age_seconds") == 300
        and safe.get("canary_passed") is True
        and safe.get("usage_complete") is True
        and safe.get("in_flight") == safe.get("active_invocations") == 0
        and safe.get("failed_count") == safe.get("not_started_count") == 0
        and result.get("schema") == "luna-subscription-lme-profiled-v2"
        and result.get("campaign_stop") is None
        and safe.get("first_failure") is None
        and budget.get("first_failure") is None
        and type(result.get("canary")) is dict and result["canary"].get("passed") is True
        and result.get("usage_complete_now") is True
        and budget.get("usage_complete") is True
        and budget.get("in_flight") == budget.get("reserved") == 0
        and result.get("active_invocations") == 0
        and safe.get("turns") == budget.get("turns")
        and safe.get("known_tokens") == budget.get("known_tokens")
        and safe.get("known_tokens_scope") == budget.get("known_tokens_scope")
        and safe.get("correct_count") == sum(q["correct"] is True for q in verdict["questions"])
        and safe.get("incorrect_count") == sum(q["correct"] is False for q in verdict["questions"])
        and result.get("stage_accounting_schema") == "source-free-stages-v1"
        and result.get("stage_collector_sha256") == COLLECTOR
        and verdict["stage_accounting_reconciled"] and verdict["warm_metrics_valid"])
    return verdict


def completed_and_clean(terminal, unit):
    return bool(terminal.get("validated") is True and unit.get("available") is True
        and unit.get("active_state") == "active" and unit.get("sub_state") == "exited"
        and unit.get("result") == "success" and unit.get("main_pid") == 0
        and unit.get("restarts") == 0 and unit.get("exit_status") == 0
        and unit.get("exit_code_kind") == 1 and unit.get("policy_ok") is True
        and (unit.get("cgroup_header_matches") is True or unit.get("cgroup_header_empty") is True)
        and unit.get("expected_cgroup_processes") == 0)


def file_meta(path):
    try:
        state = path.stat(follow_symlinks=False)
        return {"present": regular(path), "bytes": state.st_size, "mtime_ns": state.st_mtime_ns}
    except OSError:
        return {"present": False, "bytes": None, "mtime_ns": None}


def safe_terminal_summary(value):
    if value is None:
        return {"available": False, "ok": None, "failure_category": None,
                "first_failure": None}
    code = value.get("stop_code")
    code = code if type(code) is str else ("other" if code is not None else None)
    return {"available": True,
        "ok": value.get("ok") if type(value.get("ok")) is bool else None,
        "failure_category": code if code in QUESTION_STOP_CODES or fixed_code(code) is not None
            else ("other" if code is not None else None),
        "first_failure": first_failure_summary(value.get("first_failure")),
        "first_failure_valid": value.get("first_failure") is None or first_failure_summary(value.get("first_failure")) is not None,
        "questions_completed": safe_int(value.get("questions_completed"), 4),
        "correct_count": safe_int(value.get("correct_count"), 4),
        "incorrect_count": safe_int(value.get("incorrect_count"), 4),
        "turns": safe_int(value.get("turns"), 8012),
        "known_tokens": safe_int(value.get("known_tokens")),
        "usage_complete": value.get("usage_complete") if type(value.get("usage_complete")) is bool else None}


def root_identity_valid(root, unit, cgroup):
    match = re.fullmatch(r"\.hymem-luna-lme-capacity-([A-Za-z0-9_-]{8,})", root.name)
    return bool(root.is_absolute() and root.parent == Path("/home/atta") and match is not None
        and not root.is_symlink() and root.is_dir()
        and unit == "hymem-luna-lme-capacity-" + match.group(1) + ".service"
        and UNIT.fullmatch(unit)
        and cgroup == "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit)


def receipt_valid(receipt, root, unit, cgroup):
    if type(receipt) is not dict:
        return False
    expected_sources = {
        "luna_subscription_lme_profiled_v2.py": PROFILED,
        "luna_subscription_lme_warm_v2.py": WARM_RUNNER,
        "luna_stage_accounting.py": COLLECTOR,
        "codex_subscription_warm_v2.py": WARM,
        "codex_subscription_concurrent_v2.py": CONCURRENT,
        "luna_subscription_pilot.py": PILOT,
        "codex_subscription.py": BASE,
        "headless-source-map.json": INVENTORY_SHA256}
    return bool(receipt.get("schema") == "luna-capacity-launch-v1"
        and receipt.get("root") == str(root) and receipt.get("unit") == unit
        and receipt.get("expected_cgroup") == cgroup and receipt.get("output") == "run"
        and receipt.get("source_sha256") == expected_sources
        and receipt.get("runner_sha256") == PROFILED
        and receipt.get("inventory_sha256") == INVENTORY_SHA256
        and receipt.get("inventory_stamp") == str(root / "headless-source-map.json")
        and receipt.get("dataset_sha256") == DATASET
        and receipt.get("binary") == str(BINARY)
        and HEX.fullmatch(str(receipt.get("binary_sha256", "")))
        and receipt.get("runtime_max_seconds") == 14530
        and receipt.get("timeout_stop_seconds") == 10
        and receipt.get("memory_max_bytes") == 4294967296
        and receipt.get("cpu_quota_percent") == 200 and receipt.get("tasks_max") == 256
        and receipt.get("oom_policy") == "kill" and receipt.get("kill_mode") == "control-group"
        and receipt.get("restart") == "no" and receipt.get("remain_after_exit") is True
        and receipt.get("model") == "gpt-6-luna" and receipt.get("subscription_only") is True
        and receipt.get("reported_quota_floor_percent") == 25
        and receipt.get("limits") == LIMITS
        and Path(str(receipt.get("candidate", ""))).is_absolute()
        and Path(str(receipt.get("dataset", ""))).is_absolute())


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("root", "unit", "expected-cgroup", "receipt-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    report = {"schema": "luna-lme-capacity-progress-v1", "read_only": True,
        "raw_text_exported": False, "completed_and_clean": False}
    try:
        root = Path(args.root)
        unit = args.unit
        cgroup = args.expected_cgroup
        if not root_identity_valid(root, unit, cgroup) or not HEX.fullmatch(args.receipt_sha256):
            raise ValueError("input_invalid")
        receipt_path = root / "launch-receipt.json"
        if not regular(receipt_path) or digest(receipt_path) != args.receipt_sha256:
            raise ValueError("receipt_pin_invalid")
        receipt = bounded_json(receipt_path)
        if not receipt_valid(receipt, root, unit, cgroup):
            raise ValueError("receipt_invalid")
        if not regular(BINARY) or digest(BINARY) != receipt["binary_sha256"]:
            raise ValueError("binary_pin_invalid")
        pins = verify_live_pins(root, receipt)
        report["live_pins"] = pins
        if not all(pins.values()):
            raise ValueError("live_pin_invalid")
        output = root / "run"
        if output.is_symlink():
            raise ValueError("output_path_invalid")
        report["unit"] = systemd_state(unit, cgroup)
        max_events = report["unit"].get("cgroup_tasks", {}).get("pids_events_max")
        report["task_limit_denial_observed"] = max_events > 0 if max_events is not None else None
        progress = bounded_json(output / "private-progress.json")
        if progress is not None and (progress.get("stage_accounting_schema") != "source-free-stages-v1"
                or progress.get("stage_collector_sha256") != COLLECTOR
                or stage_summary(progress.get("stage_accounting")) is None):
            raise ValueError("progress_stage_invalid")
        report["progress"] = progress_summary(progress)
        report["files"] = {name: file_meta(output / name) for name in
            ("private-progress.json", "private-result.json", "private-run.log")}
        report["files"]["safe-terminal.json"] = file_meta(root / "safe-terminal.json")
        for index in range(4):
            for name in ("private-result.json", "private-row.json"):
                key = f"q-{index:04d}/{name}"
                report["files"][key] = file_meta(output / key)
        disk = os.statvfs(root)
        report["disk_free_gib"] = round(disk.f_bavail * disk.f_frsize / 1024**3, 3)
        report["disk_floor_ok"] = report["disk_free_gib"] >= 20
        safe = bounded_json(root / "safe-terminal.json")
        report["safe_terminal"] = safe_terminal_summary(safe)
        result = bounded_json(output / "private-result.json") if safe else None
        report["terminal"] = verify_terminal(root, receipt, safe, result)
        report["completed_and_clean"] = completed_and_clean(report["terminal"], report["unit"])
        report["receipt_pin_verified"] = True
    except Exception:
        report["error"] = "metadata_unavailable_or_invalid"
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
