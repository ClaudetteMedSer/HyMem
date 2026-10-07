"""Read-only metadata status for the accepted noncanonical Luna LME diagnostic.

This reader never imports benchmark code, opens a memory store, reads private
question rows or logs, starts a service, or makes a model request. It accepts
only the pinned source-only host root and safe receipt/checkpoint/result JSON.
Version 2 recognizes finite transport stop codes and reports failed-exit cleanup
separately from diagnostic completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


SCHEMA = "luna-lme-diagnostic-progress-v2"
ROOT_PARENT = Path("/home/atta")
ROOT_UID = 1000
ROOT_NAME = re.compile(r"\.hymem-lme-diagnostic-[a-z0-9_-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
RUNNER_SHA256 = "a8f12c305825806fe4ce2b2f63025098632250f33f224dbee5b1fb6f104d5515"
INVENTORY_SHA256 = "228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf"
MAP_SHA256 = "9e3fe4e495585f63bb560b123fbc0edb6441396ec28ae3f8fa6bfb6ba5b8cfae"
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
BINARY_SHA256 = "167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9"
MODE = "semantic_diagnostic_v1"
RUN_SCHEMA = "luna-lme-semantic-diagnostic-v1"
CHECKPOINT_SCHEMA = "hymem-benchmark-checkpoint-v1"
PINS = {
    "benchmarks/codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
    "benchmarks/codex_subscription_concurrent_v2.py": "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0",
    "benchmarks/codex_subscription_warm_v2.py": "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593",
    "benchmarks/codex_subscription_warm_v3.py": "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d",
    "benchmarks/codex_subscription_staged_v1.py": "c483975d53ca3523708cb589a68a8b0523664312ea259057e48ee01e7dad6ca8",
    "tools/diagnostics/luna_subscription_pilot.py": "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0",
    "tools/diagnostics/luna_subscription_lme_warm_v2.py": "3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567",
    "benchmarks/lme_diagnostic.py": "b07cdb4d26ad2f95ab236fe8af4a1665d19c376ffaf077e90a561ce500ac1181",
    "tools/diagnostics/luna_lme_diagnostic_v1.py": RUNNER_SHA256,
}
LIMITS = {"campaign": [8012, 48_160_000, 14_400],
          "question": [2000, 12_000_000, 12_600],
          "canary": [12, 160_000, 600]}
RECEIPT_FIELDS = frozenset({
    "schema", "root", "unit", "expected_cgroup", "source_sha256",
    "candidate_map_sha256", "inventory_sha256", "dataset_sha256",
    "binary_sha256", "selected_count", "workers", "indexing_seconds",
    "output_dir", "limits",
})
COUNT_FIELDS = frozenset({"expected", "attempted", "unique_attempted",
                          "total_attempts", "completed", "failed", "missing"})
RUNTIME = Path("/run/user/1000")
CGROUP_ROOT = Path("/sys/fs/cgroup")

# Source-derived bounded stop vocabulary (warm_v2/v3, concurrent_v2, and the
# accepted diagnostic runner). No free-form provider text is admitted here.
_RPC_METHODS = frozenset({"initialize", "config/read", "model/list", "account/read",
    "account/rateLimits/read", "thread/start", "turn/start", "thread/unsubscribe",
    "startup", "turn/events"})
_EVENT_METHODS = frozenset({"thread/started", "thread/status/changed",
    "thread/closed", "turn/started", "turn/completed", "turn/failed",
    "item/started", "item/completed", "item/agentMessage/delta",
    "item/reasoning/textDelta", "thread/tokenUsage/updated", "account/updated",
    "account/rateLimits/updated", "remoteControl/status/changed", "warning",
    "invalid_method"})
_FIXED_STOPS = frozenset({
    "account_changed", "account_notification_invalid", "active_thread_closed",
    "binary_version_mismatch", "binary_version_unverified",
    "capability_isolation_unverified", "credit_balance_present", "duplicate_item",
    "event_limit", "exact_model_unavailable", "final_message_invalid",
    "incomplete_model_catalog", "incomplete_turn_or_usage",
    "instruction_isolation_unverified", "invalid_quota", "invalid_request",
    "item_id_invalid", "item_identity_mismatch", "item_lifecycle_invalid",
    "mcp_isolation_unverified", "message_delta_invalid", "model_or_search_unverified",
    "notification_invalid", "output_limit", "pilot_budget_exhausted", "pilot_stopped",
    "process_not_reaped", "protocol_failure", "provider_endpoint_override",
    "quota_exhausted", "quota_floor", "reasoning_delta_invalid",
    "remote_control_enabled", "retired_thread_closed_invalid",
    "retired_thread_status_invalid", "routing_unverified",
    "runtime_isolation_not_accepted", "service_tier_unverified", "startup_failed",
    "subscription_auth_required", "subscription_plan_unverified", "thread_id_missing",
    "thread_identity_mismatch", "thread_isolation_unverified",
    "thread_start_status_invalid", "timeout", "tool_or_extra_item", "turn_failed",
    "turn_identity_mismatch", "turn_start_invalid", "unexpected_response",
    "unexpected_server_request", "unknown_quota", "unsubscribe_thread_mismatch",
    "unsubscribe_unverified", "unsupported_chat_control", "unsupported_chat_history",
    "usage_identity_mismatch", "usage_invalid", "usage_regressed", "warning_limit",
    "transport_failure", "warning_thread_unverified", "warning_unapproved",
    "concurrent_completion_rejected", "ledger_protocol_violation", "wall_limit",
    "campaign_wall_limit", "campaign_budget_exhausted", "question_budget_exhausted",
    "budget_exhausted_before_turn", "admission_rejected", "quota_unverified",
    "cleanup_failure", "stage_accounting_failure", "interrupted", "question_failure",
    "adapter_cleanup_failure", "client_cleanup_failure", "worker_runtime_failure",
    "campaign_failure", "close_during_invocation", "invalid_request_or_closed",
    "transport_or_admission_failure", "interrupted_invocation", "campaign_stopped",
    "fixed_other", "unexpected_notification_unknown", "finite_other", "usage_unknown",
})
_RUNNER_STOPS = frozenset({"RuntimeError", "ValueError", "KeyboardInterrupt",
    "finite_other", "final_accounting_or_dataset_failure", "incomplete_questions"})
_PHASES = frozenset({"startup", "preflight", "run", "unsubscribe",
    "rotation_cleanup", "cleanup"})
_APP_ERROR_CLASSES = frozenset({"contextWindowExceeded", "sessionBudgetExceeded",
    "usageLimitExceeded", "rateLimitExceeded", "flexUnavailable", "serverOverloaded",
    "cyberPolicy", "misalignmentPolicyViolation", "internalServerError",
    "unauthorized", "badRequest", "threadRollbackFailed", "sandboxError", "other",
    "httpConnectionFailed", "responseStreamConnectionFailed",
    "responseStreamDisconnected", "responseTooManyFailedAttempts",
    "activeTurnNotSteerable", "invalid", "unspecified"})


def _safe_stop(value: Any) -> bool:
    if type(value) is not str:
        return False
    if value in _FIXED_STOPS:
        return True
    if ":" not in value:
        return False
    family, suffix = value.split(":", 1)
    return ((family in {"process_exit", "protocol_failure", "rpc_failure"}
             and suffix in _RPC_METHODS)
            or (family == "unexpected_notification" and suffix in _EVENT_METHODS))


def _failure_projection(budget: dict[str, Any]) -> dict[str, Any] | None:
    fault = budget.get("first_failure")
    if fault is None:
        return None
    _valid(type(fault) is dict and _safe_stop(fault.get("code")),
           "first_failure_invalid")
    phase, rpc = fault.get("phase"), fault.get("rpc")
    _valid(type(phase) is str and phase in _PHASES
           and (rpc is None or (type(rpc) is str and rpc in _RPC_METHODS)),
           "first_failure_phase_invalid")
    out: dict[str, Any] = {"code": fault["code"], "phase": phase, "rpc": rpc}
    for key in ("process_index", "request_index", "retired_count", "queue_count"):
        number = fault.get(key)
        _valid(_int(number) and number <= 1_000_000_000,
               "first_failure_counter_invalid")
        out[key] = number
    for key in ("turn_admitted", "known_usage", "usage_complete"):
        flag = fault.get(key)
        _valid(type(flag) is bool, "first_failure_flag_invalid")
        out[key] = flag
    rpc_error = fault.get("rpc_error")
    if rpc_error is not None:
        _valid(type(rpc_error) is dict and type(rpc_error.get("category")) is str
               and rpc_error.get("category") in {
            "parse_error", "invalid_request", "method_not_found", "invalid_params",
            "internal_error", "other_numeric", "invalid"}, "first_failure_rpc_invalid")
        number = rpc_error.get("code")
        _valid((number is None and rpc_error["category"] == "invalid")
               or (type(number) is int and -2147483648 <= number <= 2147483647),
               "first_failure_rpc_code_invalid")
        if number is not None:
            out["rpc_error_code"] = number
        out["rpc_error_category"] = rpc_error["category"]
    app_error = fault.get("app_server_error")
    if app_error is not None:
        _valid(type(app_error) is dict and type(app_error.get("identity")) is str
               and app_error.get("identity") in {
            "invalid", "unbound", "mismatch", "matched"}, "first_failure_app_invalid")
        projected = {"identity": app_error["identity"]}
        if app_error["identity"] == "matched":
            retry = app_error.get("will_retry")
            if retry is not None:
                _valid(type(retry) is bool or retry == "invalid",
                       "first_failure_app_retry_invalid")
                projected["will_retry"] = retry
            error_class = app_error.get("error_class")
            if error_class is not None:
                _valid(type(error_class) is str and error_class in _APP_ERROR_CLASSES,
                       "first_failure_app_class_invalid")
                projected["error_class"] = error_class
            if "http_status_code" in app_error:
                status = app_error["http_status_code"]
                _valid(status is None or status == "invalid"
                       or (type(status) is int and 0 <= status <= 65535),
                       "first_failure_app_status_invalid")
                projected["http_status_code"] = status
        out["app_server_error"] = projected
    return out


def _valid(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(actual) == set(expected)
                and all(_strict_equal(actual[key], item)
                        for key, item in expected.items()))
    if type(expected) is list:
        return (len(actual) == len(expected)
                and all(_strict_equal(left, right)
                        for left, right in zip(actual, expected, strict=True)))
    return actual == expected


def _file(path: Path, root: Path, cap: int) -> bool:
    try:
        mode = path.lstat().st_mode
        return (stat.S_ISREG(mode) and path.stat().st_size <= cap
                and path.resolve().is_relative_to(root))
    except (OSError, ValueError):
        return False


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_metadata_field")
        value[key] = item
    return value


def _read(path: Path, root: Path, cap: int) -> Any:
    _valid(_file(path, root, cap), "metadata_file_invalid")
    handle = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        details = os.fstat(handle)
        _valid(stat.S_ISREG(details.st_mode) and details.st_size <= cap,
               "metadata_file_invalid")
        with os.fdopen(handle, "rb", closefd=False) as stream:
            encoded = stream.read(cap + 1)
        _valid(len(encoded) <= cap, "metadata_size_invalid")
    finally:
        os.close(handle)
    return json.loads(encoded.decode("utf-8"), object_pairs_hook=_object_pairs,
                      parse_constant=lambda _value: (_ for _ in ()).throw(
                          ValueError("nonfinite_metadata")))


def _root(path: Path) -> Path:
    _valid(path.is_absolute() and path.parent == ROOT_PARENT
           and ROOT_NAME.fullmatch(path.name) is not None
           and path.is_dir() and not path.is_symlink(), "root_invalid")
    info = path.stat()
    _valid(info.st_uid == ROOT_UID and not info.st_mode & 0o077,
           "root_permission_invalid")
    return path


def _receipt(root: Path, receipt_sha256: str) -> dict[str, Any]:
    _valid(HEX.fullmatch(receipt_sha256) is not None, "receipt_sha_invalid")
    path = root / "launch-receipt.json"
    receipt = _read(path, root, 32_768)
    _valid(_sha(path) == receipt_sha256 and type(receipt) is dict
           and set(receipt) == RECEIPT_FIELDS, "receipt_invalid")
    unit = "hymem-luna-lme-diagnostic-" + root.name.removeprefix(
        ".hymem-lme-diagnostic-") + ".service"
    expected = {
        "schema": "luna-lme-diagnostic-launch-v1", "root": str(root),
        "unit": unit, "expected_cgroup":
            "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "source_sha256": PINS, "candidate_map_sha256": MAP_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "dataset_sha256": DATASET_SHA256, "binary_sha256": BINARY_SHA256,
        "selected_count": 4, "workers": 4, "indexing_seconds": 10_800,
        "output_dir": str(root / "run"), "limits": LIMITS,
    }
    _valid(_strict_equal(receipt, expected), "receipt_identity_invalid")
    source_map = root / "source-map.json"
    _valid(_file(source_map, root, 1_000_000)
           and _sha(source_map) == INVENTORY_SHA256, "inventory_drift")
    inventory = _read(source_map, root, 1_000_000)
    _valid(type(inventory) is dict and type(inventory.get("source_sha256")) is dict,
           "inventory_shape_invalid")
    expected_files = inventory["source_sha256"]
    _valid(len(expected_files) == 514 and all(
        type(relative) is str and relative and not relative.startswith("/")
        and ".." not in Path(relative).parts and type(digest) is str
        and HEX.fullmatch(digest) is not None
        for relative, digest in expected_files.items()), "inventory_entries_invalid")
    encoded = json.dumps(expected_files, sort_keys=True, separators=(",", ":")).encode()
    _valid(hashlib.sha256(encoded).hexdigest() == MAP_SHA256, "inventory_map_drift")
    candidate = root / "candidate"
    _valid(candidate.is_dir() and not candidate.is_symlink(), "candidate_missing")
    actual: dict[str, str] = {}
    for source in candidate.rglob("*"):
        relative = source.relative_to(candidate)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if source.suffix in {".pyc", ".pyo"}:
            continue
        _valid(not source.is_symlink(), "candidate_symlink")
        if source.is_dir():
            continue
        _valid(source.is_file() and source.resolve().is_relative_to(candidate),
               "candidate_special_file")
        actual[relative.as_posix()] = _sha(source)
    _valid(actual == expected_files, "candidate_source_drift")
    for relative, digest in PINS.items():
        source = root / "code" / relative
        _valid(_file(source, root, 2_000_000) and _sha(source) == digest,
               "pinned_code_drift")
    return receipt


def _int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and minimum <= value <= (1 << 63) - 1


def _checkpoint(value: Any, receipt: dict[str, Any]) -> dict[str, Any]:
    _valid(type(value) is dict and value.get("schema") == CHECKPOINT_SCHEMA
           and value.get("status") in {"running", "complete"}
           and value.get("scored") is True and value.get("verdict_key") == "correct"
           and type(value.get("manifest")) is dict
           and type(value.get("expected_ids")) is list
           and type(value.get("entries")) is dict, "checkpoint_invalid")
    ids = value["expected_ids"]
    _valid(len(ids) == receipt["selected_count"]
           and all(type(item) is str and 0 < len(item) <= 128 for item in ids)
           and len(set(ids)) == len(ids),
           "checkpoint_ids_invalid")
    manifest = value["manifest"]
    run_id = manifest.get("run_id")
    _valid(type(run_id) is str and run_id == value.get("run_id")
           and run_id == _canonical_hash({key: item for key, item in manifest.items()
                                          if key != "run_id"})
           and manifest.get("schema") == RUN_SCHEMA
           and manifest.get("mode") == MODE
           and manifest.get("canonical_r9_artifact") is False
           and manifest.get("official_model_score") is False
           and manifest.get("candidate_map_sha256") == MAP_SHA256
           and manifest.get("dataset_sha256") == DATASET_SHA256
           and manifest.get("selected_source_order") == "first_n"
           and manifest.get("expected_count") == len(ids)
           and manifest.get("expected_ids_hash") == _canonical_hash(ids)
           and manifest.get("scored_run") is True
           and manifest.get("diagnostic_helper_sha256") == PINS["benchmarks/lme_diagnostic.py"]
           and manifest.get("runner_sha256") == RUNNER_SHA256
           and manifest.get("transport_sha256") == PINS["benchmarks/codex_subscription_staged_v1.py"]
           and manifest.get("ordinary_transport_sha256") == PINS["benchmarks/codex_subscription_warm_v3.py"]
           and manifest.get("rerolls") == 0,
           "checkpoint_identity_invalid")
    selected_rows = manifest.get("selected_row_sha256")
    _valid(type(selected_rows) is list and len(selected_rows) == len(ids)
           and all(type(item) is str and HEX.fullmatch(item) is not None
                   for item in selected_rows), "checkpoint_row_identity_invalid")
    limits = manifest.get("limits")
    _valid(type(limits) is dict and _strict_equal(limits.get("campaign"), dict(zip(
        ("turns", "known_tokens", "seconds"), LIMITS["campaign"]))
        ) and _strict_equal(limits.get("canary"), dict(zip(
            ("turns", "known_tokens", "seconds"), LIMITS["canary"]))
        ) and _strict_equal(limits.get("question"), dict(zip(
            ("turns", "known_tokens", "seconds"), LIMITS["question"]))
        ) and type(limits.get("indexing_seconds")) in (int, float)
        and limits.get("indexing_seconds") == 10_800
        and type(limits.get("workers")) is int and limits.get("workers") == 4,
        "checkpoint_limits_invalid")
    _valid(set(value["entries"]).issubset(ids), "checkpoint_entries_invalid")
    return value


def _counts(checkpoint: dict[str, Any]) -> tuple[dict[str, int] | None, int, int, int]:
    entries = checkpoint["entries"]
    complete = failed = strict_unhealthy = summary_degraded = 0
    for qid, entry in entries.items():
        _valid(type(entry) is dict and entry.get("status") in {"completed", "failed"}
               and entry.get("attempts") == 1 and type(entry.get("row")) is dict
               and entry["row"].get("question_id") == qid,
               "checkpoint_entry_invalid")
        if entry["status"] == "completed":
            row = entry["row"]
            _valid(type(row.get("correct")) is bool
                   and row.get("benchmark_failure") is None
                   and row.get("diagnostic_kind") in {
                       "strict_healthy", "summary_degradation", "semantic_quarantine"}
                   and type(row.get("strict_indexing_healthy")) is bool
                   and ((row["diagnostic_kind"] == "semantic_quarantine")
                        is (row["strict_indexing_healthy"] is False))
                   and _int(row.get("quarantined_chunks"))
                   and _int(row.get("summary_degraded_sessions"))
                   and type(row.get("context_sha")) is str
                   and HEX.fullmatch(row["context_sha"]) is not None,
                   "checkpoint_projection_invalid")
            _valid((row["diagnostic_kind"] == "semantic_quarantine")
                   == (row["quarantined_chunks"] > 0)
                   and (row["diagnostic_kind"] != "summary_degradation"
                        or row["summary_degraded_sessions"] > 0),
                   "checkpoint_projection_inconsistent")
            complete += 1
            strict_unhealthy += not row["strict_indexing_healthy"]
            summary_degraded += row["summary_degraded_sessions"]
        else:
            failed += 1
    if checkpoint["status"] != "complete":
        return None, complete, strict_unhealthy, summary_degraded
    counts = checkpoint.get("counts")
    _valid(type(counts) is dict and set(counts) == COUNT_FIELDS
           and all(_int(number) for number in counts.values())
           and counts["expected"] == len(checkpoint["expected_ids"])
           and counts["attempted"] == len(entries)
           and counts["unique_attempted"] == len(entries)
           and counts["total_attempts"] == len(entries)
           and counts["completed"] == complete
           and counts["failed"] == failed
           and counts["missing"] == len(checkpoint["expected_ids"]) - len(entries),
           "checkpoint_counts_invalid")
    return counts, complete, strict_unhealthy, summary_degraded


def _terminal(value: Any, checkpoint: dict[str, Any], receipt: dict[str, Any],
              counts: dict[str, int], strict_unhealthy: int) -> dict[str, Any]:
    _valid(type(value) is dict and value.get("schema") == RUN_SCHEMA
           and value.get("run_id") == checkpoint["run_id"]
           and value.get("canonical_r9_artifact") is False
           and value.get("official_model_score") is False
           and _int(value.get("selected_denominator"))
           and value["selected_denominator"] == receipt["selected_count"]
           and _int(value.get("scored_count"))
           and value["scored_count"] == counts["completed"]
           and _int(value.get("failed_or_unscored_count"))
           and value["failed_or_unscored_count"] == counts["failed"] + counts["missing"]
           and _int(value.get("strict_unhealthy_count"))
           and value["strict_unhealthy_count"] == strict_unhealthy
           and value.get("checkpoint_counts") == counts,
           "terminal_identity_invalid")
    scored = value["scored_count"]
    correct, incorrect = value.get("correct_count"), value.get("incorrect_count")
    _valid(_int(correct) and _int(incorrect) and correct + incorrect == scored,
           "terminal_score_counts_invalid")
    verified_correct = sum(entry["row"]["correct"]
        for entry in checkpoint["entries"].values() if entry["status"] == "completed")
    _valid(correct == verified_correct, "terminal_score_mismatch")
    full = value.get("quality_accuracy_full_selected")
    if scored == receipt["selected_count"]:
        _valid(type(full) is float and full == correct / scored,
               "terminal_accuracy_invalid")
    else:
        _valid(full is None, "terminal_partial_accuracy_invalid")
    canary = value.get("canary")
    _valid(canary is None or (type(canary) is dict
           and type(canary.get("structural_valid")) is bool
           and type(canary.get("model_gold_match")) is bool),
           "terminal_canary_invalid")
    budget = value.get("budget")
    _valid(type(budget) is dict and all(_int(budget.get(key)) for key in
        ("turns", "known_tokens", "reserved", "in_flight"))
        and type(budget.get("usage_complete")) is bool
        and type(budget.get("stopped")) is bool
        and (budget.get("stop_code") is None or _safe_stop(budget.get("stop_code")))
        and (value.get("campaign_stop") is None
             or (type(value.get("campaign_stop")) is str
                 and value.get("campaign_stop") in _RUNNER_STOPS)
             or _safe_stop(value.get("campaign_stop"))),
        "terminal_budget_invalid")
    projected = _failure_projection(budget)
    if projected is not None:
        _valid(budget["stopped"] is True and budget["stop_code"] == projected["code"],
               "first_failure_stop_mismatch")
    return value


def _cgroup_policy(group: Path) -> bool:
    try:
        if (not group.is_dir() or not group.resolve().is_relative_to(CGROUP_ROOT)
                or (group / "memory.max").read_text().strip() != "4294967296"
                or (group / "pids.max").read_text().strip() != "128"):
            return False
        cpu = (group / "cpu.max").read_text().split()
        return (len(cpu) == 2 and all(value.isdecimal() for value in cpu)
                and int(cpu[1]) > 0 and int(cpu[0]) == 2 * int(cpu[1]))
    except (OSError, ValueError):
        return False


def _cgroup_empty(group: Path) -> bool:
    try:
        if any((group / name).read_text().strip()
               for name in ("cgroup.procs", "cgroup.threads")):
            return False
        events = dict(line.split(None, 1) for line in
                      (group / "cgroup.events").read_text().splitlines()
                      if len(line.split(None, 1)) == 2)
        return events.get("populated") == "0"
    except (OSError, ValueError):
        return False


def _runtime(receipt: dict[str, Any]) -> str:
    """Verify exact live policy or successful exit with an empty cgroup."""
    if sys.platform != "linux" or os.getuid() != ROOT_UID:
        return "unverified"
    bus = RUNTIME / "bus"
    try:
        runtime = RUNTIME.lstat()
        socket = bus.lstat()
        if (not stat.S_ISDIR(runtime.st_mode) or runtime.st_uid != ROOT_UID
                or runtime.st_mode & 0o077
                or not stat.S_ISSOCK(socket.st_mode) or socket.st_uid != ROOT_UID):
            return "unverified"
        env = {**os.environ, "XDG_RUNTIME_DIR": str(RUNTIME),
               "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(bus)}
        fields = ("ActiveState", "SubState", "MainPID", "ControlGroup",
                  "NRestarts", "Result", "ExecMainStatus", "MemoryMax",
                  "TasksMax", "CPUQuotaPerSecUSec", "KillMode", "Restart",
                  "RemainAfterExit", "OOMPolicy", "RuntimeMaxUSec",
                  "TimeoutStopUSec")
        command = ["/usr/bin/systemctl", "--user", "show", receipt["unit"],
                   "--property=" + ",".join(fields), "--no-pager"]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=10, check=True, env=env)
        values = dict(line.split("=", 1) for line in completed.stdout.splitlines()
                      if "=" in line)
        if (set(values) != set(fields) or values["NRestarts"] != "0"
                or values["MemoryMax"] != "4294967296"
                or values["TasksMax"] != "128"
                or values["CPUQuotaPerSecUSec"] not in {"2s", "2.000s"}
                or values["KillMode"] != "control-group"
                or values["Restart"] != "no"
                or values["RemainAfterExit"] != "yes"
                or values["OOMPolicy"] != "kill"
                or values["RuntimeMaxUSec"] not in {
                    "4h 2min 10s", "4h 2min 10.000s", "14530s", "14530.000s"}
                or values["TimeoutStopUSec"] not in {"10s", "10.000s"}):
            return "unverified"
        group = CGROUP_ROOT / receipt["expected_cgroup"].lstrip("/")
        if not group.resolve().is_relative_to(CGROUP_ROOT):
            return "unverified"
        if (values["ActiveState"] == "active" and values["SubState"] == "running"
                and values["ControlGroup"] == receipt["expected_cgroup"]
                and values["MainPID"].isdecimal()
                and int(values["MainPID"]) > 0 and _cgroup_policy(group)
                and values["MainPID"] in (group / "cgroup.procs").read_text().splitlines()):
            return "running_verified"
        if ((values["ActiveState"], values["SubState"]) not in {
                ("active", "exited"), ("inactive", "dead")}
                or values["MainPID"] != "0" or values["Result"] != "success"
                or values["ExecMainStatus"] != "0"
                or values["ControlGroup"] not in {"", receipt["expected_cgroup"]}):
            return "unverified"
        if group.exists():
            if not _cgroup_policy(group) or not _cgroup_empty(group):
                return "unverified"
        return "clean_exit"
    except (OSError, ValueError, subprocess.SubprocessError):
        return "unverified"


def _failed_exit_cleanup(receipt: dict[str, Any]) -> bool:
    """Prove no tasks remain after a failed unit, without calling it success."""
    if sys.platform != "linux" or os.getuid() != ROOT_UID:
        return False
    try:
        runtime, bus = RUNTIME.lstat(), (RUNTIME / "bus").lstat()
        if (not stat.S_ISDIR(runtime.st_mode) or runtime.st_uid != ROOT_UID
                or runtime.st_mode & 0o077 or not stat.S_ISSOCK(bus.st_mode)
                or bus.st_uid != ROOT_UID):
            return False
        fields = ("ActiveState", "SubState", "MainPID", "ControlGroup",
                  "NRestarts", "Result", "ExecMainStatus", "MemoryMax",
                  "TasksMax", "CPUQuotaPerSecUSec", "KillMode", "Restart",
                  "RemainAfterExit", "OOMPolicy", "RuntimeMaxUSec",
                  "TimeoutStopUSec")
        env = {**os.environ, "XDG_RUNTIME_DIR": str(RUNTIME),
               "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(RUNTIME / "bus")}
        completed = subprocess.run(["/usr/bin/systemctl", "--user", "show",
            receipt["unit"], "--property=" + ",".join(fields), "--no-pager"],
            capture_output=True, text=True, timeout=10, check=True, env=env)
        values = dict(line.split("=", 1) for line in completed.stdout.splitlines()
                      if "=" in line)
        if (set(values) != set(fields) or values["NRestarts"] != "0"
                or values["MemoryMax"] != "4294967296" or values["TasksMax"] != "128"
                or values["CPUQuotaPerSecUSec"] not in {"2s", "2.000s"}
                or values["KillMode"] != "control-group" or values["Restart"] != "no"
                or values["RemainAfterExit"] != "yes" or values["OOMPolicy"] != "kill"
                or values["RuntimeMaxUSec"] not in {"4h 2min 10s", "4h 2min 10.000s",
                    "14530s", "14530.000s"}
                or values["TimeoutStopUSec"] not in {"10s", "10.000s"}
                or (values["ActiveState"], values["SubState"]) not in {
                    ("active", "exited"), ("inactive", "dead"), ("failed", "failed")}
                or values["MainPID"] != "0"
                or values["Result"] not in {"exit-code", "signal", "core-dump", "oom-kill"}
                or not values["ExecMainStatus"].isdecimal()
                or int(values["ExecMainStatus"]) == 0
                or values["ControlGroup"] not in {"", receipt["expected_cgroup"]}):
            return False
        group = CGROUP_ROOT / receipt["expected_cgroup"].lstrip("/")
        return (group.resolve().is_relative_to(CGROUP_ROOT)
                and (not group.exists() or (_cgroup_policy(group) and _cgroup_empty(group))))
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def inspect(root: Path, receipt_sha256: str) -> dict[str, Any]:
    """Return safe metadata only; unknown/partial state never claims completion."""
    root = _root(root)
    receipt = _receipt(root, receipt_sha256)
    output: dict[str, Any] = {"schema": SCHEMA, "status": "prepared_not_launched",
        "completed_diagnostic_and_clean": False, "runtime_cleanup_verified": False,
        "selected_denominator": receipt["selected_count"],
        "scored_count": None, "correct_count": None,
        "strict_indexing_healthy_for_all": None,
        "summary_degraded_sessions_total": None,
        "canary_model_gold_match": None,
        "canary_structural_valid": None,
        "campaign_stop": None, "budget_stop_code": None,
        "first_failure": None, "failed_count": None,
        "known_turns": None, "known_tokens": None,
        "usage_complete": None, "stage_timing_available": False}
    marker = root / "launch-attempt.json"
    run = root / "run"
    if not marker.exists():
        _valid(not run.exists() and not (root / "launch-command-result.json").exists(),
               "unlaunched_state_drift")
        return output
    _valid(_read(marker, root, 4096) == {"receipt_sha256": receipt_sha256,
           "one_shot": True}, "launch_marker_invalid")
    output["status"] = "launch_attempted"
    command_result = root / "launch-command-result.json"
    if command_result.exists():
        command = _read(command_result, root, 4096)
        _valid(type(command) is dict and set(command) == {"returncode"}
               and type(command["returncode"]) is int,
               "launch_command_result_invalid")
        if command["returncode"] != 0:
            output["status"] = "launch_dispatch_failed"
            return output
    runtime_state = _runtime(receipt)
    checkpoint_path = run / "diagnostic-checkpoint.json"
    result_path = run / "diagnostic-result.json"
    if not checkpoint_path.exists():
        _valid(not result_path.exists(), "terminal_without_checkpoint")
        output["status"] = ("launch_running" if runtime_state == "running_verified"
                            else "launch_attempted_runtime_unverified")
        return output
    checkpoint = _checkpoint(_read(checkpoint_path, root, 2_000_000), receipt)
    counts, complete, strict_unhealthy, summary_degraded = _counts(checkpoint)
    output["scored_count"] = complete
    output["summary_degraded_sessions_total"] = summary_degraded
    output["strict_indexing_healthy_for_all"] = (
        strict_unhealthy == 0 if counts is not None and complete == receipt["selected_count"]
        else None)
    if counts is None:
        _valid(not result_path.exists(), "terminal_before_checkpoint_complete")
        output["status"] = ("checkpoint_running" if runtime_state == "running_verified"
                            else "checkpoint_running_runtime_unverified")
        return output
    output["status"] = ("checkpoint_complete_result_pending" if runtime_state == "running_verified"
                        else "checkpoint_complete_runtime_unverified")
    if not result_path.exists():
        return output
    terminal = _terminal(_read(result_path, root, 128_000), checkpoint,
                         receipt, counts, strict_unhealthy)
    output["scored_count"] = terminal["scored_count"]
    output["correct_count"] = terminal["correct_count"]
    output["canary_model_gold_match"] = (terminal["canary"].get("model_gold_match")
        if terminal["canary"] is not None else None)
    output["canary_structural_valid"] = (terminal["canary"].get("structural_valid")
        if terminal["canary"] is not None else None)
    output["campaign_stop"] = terminal.get("campaign_stop")
    budget = terminal["budget"]
    output["budget_stop_code"] = budget["stop_code"]
    output["first_failure"] = _failure_projection(budget)
    output["failed_count"] = counts["failed"]
    output["known_turns"] = budget["turns"]
    output["known_tokens"] = budget["known_tokens"]
    output["usage_complete"] = budget["usage_complete"]
    semantic_complete = (counts["completed"] == receipt["selected_count"]
        and counts["failed"] == counts["missing"] == 0
        and terminal["campaign_stop"] is None
        and terminal["canary"] is not None
        and terminal["canary"].get("structural_valid") is True
        and budget["reserved"] == budget["in_flight"] == 0
        and budget["usage_complete"] is True and budget["stopped"] is False
        and budget["stop_code"] is None)
    runtime_clean = runtime_state == "clean_exit"
    output["runtime_cleanup_verified"] = bool(runtime_clean or _failed_exit_cleanup(receipt))
    output["completed_diagnostic_and_clean"] = bool(semantic_complete and runtime_clean)
    output["status"] = ("completed_diagnostic_and_clean"
        if output["completed_diagnostic_and_clean"] else "terminal_incomplete_or_unclean")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        report = inspect(Path(args.root), args.receipt_sha256)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception:
        print(json.dumps({"schema": SCHEMA, "status": "metadata_unverified",
                          "completed_diagnostic_and_clean": False},
                         sort_keys=True, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
