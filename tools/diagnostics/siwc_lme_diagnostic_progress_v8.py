"""Strict metadata-only reader for the one-shot SIWC four-question diagnostic.

This source never imports the runner, bridge, owner or transport, opens auth
state, reads question rows or logs, or makes a model request.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


SCHEMA = "siwc-lme-diagnostic-progress-v8"
RUNTIME_SITE_MAX_FILE_BYTES = 11_507_864
ROOT_PARENT = Path("/home/atta")
ROOT_UID = 1000
ROOT_NAME = re.compile(r"\.hymem-siwc-lme-diagnostic-[a-z0-9_-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
RUNNER_RELATIVE = "tools/diagnostics/siwc_lme_diagnostic_v6.py"
RUNNER_SHA256 = "c073b69dff5e8052b8fa5fe46bd92ea50749e54f9e5828011e436f6e314546b3"
LAUNCHER_NAME = "siwc_lme_diagnostic_launch_v5.py"
LAUNCHER_SHA256 = "101c5f74b3744d2f0014dd3ce3b9f9b2045fad17ef43f5c1aee90a03161945aa"
INVENTORY_SHA256 = "b87ce83ca123fb2a51f1716ca3d3d71fd3b73799a30469b34a8f033babaa90b6"
MAP_SHA256 = "94d7b2204ed749d1b3de29c24579aeccb2730c6aa9370e2b8ebab87a7835267e"
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
RUNTIME_SHA256 = "17b78e0a93175e86f9ac03141924fd7a7f0c0c52e66b34bfa0de20ffef989df1"
RUNTIME_SITE_SHA256 = "2bed78ec3df853e3efe5052b30d514a2765a2097e183f4b3c32a3d53ef54806d"
GRANT_IDENTITY_SHA256 = "5f91fe05fb7d3b0552b7247d6a81f3fd29aae894dbcf45828b1556a0b11633dc"
HELPER_SHA256 = "b07cdb4d26ad2f95ab236fe8af4a1665d19c376ffaf077e90a561ce500ac1181"
BILLING_POLICY = "siwc_server_enforced_plan_or_existing_credits_v1"
DATASET = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json")
RUNTIME_PATH = Path("/home/atta/.hymem-siwc-runtime-v1/bin/python")
OWNER_STATE = Path("/home/atta/.hymem-chatgpt-plan-lme")
CGROUP_ROOT = Path("/sys/fs/cgroup")
USER_RUNTIME = Path("/run/user/1000")
EXECUTION_MARKER = "diagnostic-execution-marker.json"
RUN_SCHEMA = "siwc-lme-semantic-diagnostic-v6"
CHECKPOINT_SCHEMA = "hymem-benchmark-checkpoint-v1"
MODE = "semantic_diagnostic_v1"
LIMITS = {"campaign": [8012, 48_160_000, 14_400],
          "question": [2000, 12_000_000, 12_600],
          "canary": [12, 160_000, 600]}
SLOTS = ("canary.ordinary", "canary.structured", *(
    f"question.{index}.{route}" for index in range(4)
    for route in ("ordinary", "structured")))
COUNT_FIELDS = {"expected", "attempted", "unique_attempted", "total_attempts",
                "completed", "failed", "missing"}
RECEIPT_FIELDS = {"schema", "root", "unit", "expected_cgroup", "source_sha256",
    "candidate_map_sha256", "inventory_sha256", "dataset_sha256", "runtime_path",
    "runtime_sha256", "runtime_site_sha256", "runtime_site_files", "owner_state",
    "grant_identity_sha256", "selected_source_order", "selected_row_sha256",
    "selected_count", "workers", "indexing_seconds", "output_dir", "limits",
    "invocation_seconds", "model", "reasoning", "auth", "endpoint", "store",
    "stream", "billing_policy", "automatic_topup_user_attested_off",
    "reload_allowed", "api_fallback_allowed", "one_shot"}
SUMMARY_FIELDS = {"schema", "calls", "successes", "failures", "internal_http_attempts",
    "provider_internal_retries_known", "admitted_turns", "known_tokens", "usage_complete",
    "timing_seconds", "timing_saturated", "first_failure", "last_failure_code"}
STAGES = {"grounding_original_initial", "grounding_original_recheck",
    "grounding_alternatives_initial", "grounding_alternatives_recheck",
    "extraction", "digest", "profile", "facts", "rerank", "reader", "judge"}
LOCAL_CODES = {"invalid_credentials", "invalid_request", "request_limit", "invalid_timeout",
    "invalid_event", "invalid_usage", "missing_usage", "incomplete_response",
    "invalid_output", "unsupported_output", "output_limit", "event_limit",
    "event_after_completion", "response_failure", "missing_completion", "wire_limit",
    "truncated_stream", "http_failure", "invalid_content_type", "auth_failure",
    "access_failure", "quota_failure", "model_mismatch", "transport_failure",
    "timeout", "cleanup_failure", "admission_rejected", "invalid_request_or_closed",
    "concurrent_completion_rejected", "client_busy", "bridge_failure", "campaign_stopped",
    "question_stopped", "question_concurrent_invocation", "wall_limit", "campaign_wall_limit",
    "campaign_budget_exhausted", "question_budget_exhausted", "concurrency_limit",
    "budget_stopped_before_turn", "budget_exhausted_before_turn", "ledger_protocol_violation",
    "budget_failure", "resource_observer_unverified", "resource_task_denial"}
PROVIDER_CODES = {"subscription_sharing_user_not_eligible",
    "subscription_sharing_usage_limit_exceeded", "subscription_sharing_usage_unavailable",
    "subscription_sharing_unsupported_capability", "subscription_sharing_route_not_supported",
    "subscription_sharing_invalid_user", "subscription_sharing_user_unavailable",
    "chatpass_v2_scope_not_authorized", "chatpass_v2_invalid_authorization_context"}
OWNER_CODES = {"source_mismatch", "source_unavailable", "state_invalid", "binding_invalid",
    "credential_invalid", "transfer_consumed", "transfer_unavailable", "remote_invalid",
    "remote_existing", "remote_unavailable", "owner_running", "deadline_exceeded",
    "refresh_unknown", "refresh_denied", "refresh_failed", "internal_error"}
STOP_CODES = LOCAL_CODES | PROVIDER_CODES | OWNER_CODES | {"question_failure", "worker_runtime_failure",
    "campaign_failure", "owner_or_preflight_failure", "owner_cleanup_failure", "interrupted",
    "final_accounting_or_dataset_failure", "incomplete_questions", "canary_structural_failure",
    "canary_nonsemantic_failure", "canary_budget_failure", "transport_failure",
    "stage_accounting_failure", "resource_preflight_invalid", "not_started_after_campaign_stop",
    "indexing_rejected", "question_accounting_invalid", "unspecified_failure",
    "adapter_cleanup_failure",
    "client_cleanup_failure", "question_cleanup_failure"}

# Checkpoint failures have a smaller, finite vocabulary than campaign/transport
# stops. Frozen AtomicCheckpoint normalizes the coordinator's two fallback
# strings to unspecified_failure before writing either entry.
INDEXING_CODES = frozenset({
    "coverage_integrity_failure", "malformed_aggregation_failure_report",
    "malformed_cycle_failure_report", "malformed_coverage_integrity_state",
    "malformed_durable_state", "malformed_pending_backlog",
    "malformed_quarantine_state", "malformed_status_shape",
    "malformed_terminal_loss_state", "max_cycles_exhausted",
    "quarantined_extraction", "terminal_extraction_source_loss",
    "timeout_after_cycle", "timeout_before_cycle", "timeout_during_cycle",
    "cycle_exception",
})
WORKER_EXCEPTION_TYPES = frozenset({
    "AssertionError", "AttributeError", "BenchmarkCleanupError",
    "BenchmarkIntegrityError", "Exception", "FileNotFoundError", "IndexError",
    "KeyError", "OSError", "PermissionError", "RuntimeError", "TimeoutError",
    "TypeError", "ValueError", "IndexingConvergenceError",
})
QUESTION_FAILURE_CODES = (
    {"unspecified_failure"}
    | {f"indexing_failure:{code}" for code in INDEXING_CODES}
    | {f"worker_failure:{name}" for name in WORKER_EXCEPTION_TYPES}
)


def valid(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def integer(value: Any, ceiling: int = (1 << 63) - 1) -> bool:
    return type(value) is int and 0 <= value <= ceiling


def finite(value: Any, ceiling: float = 1_000_000) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= ceiling


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(typed_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def file_at(path: Path, root: Path, cap: int) -> bool:
    try:
        info = path.lstat()
        return (stat.S_ISREG(info.st_mode) and info.st_size <= cap
                and path.resolve().is_relative_to(root))
    except (OSError, ValueError):
        return False


def unique_fields(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate_metadata_field")
        output[key] = value
    return output


def read(path: Path, root: Path, cap: int) -> Any:
    valid(file_at(path, root, cap), "metadata_file_invalid")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        valid(stat.S_ISREG(info.st_mode) and info.st_size <= cap, "metadata_file_invalid")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            encoded = handle.read(cap + 1)
        valid(len(encoded) <= cap, "metadata_size_invalid")
    finally:
        os.close(fd)
    return json.loads(encoded.decode("utf-8"), object_pairs_hook=unique_fields,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite_metadata")))


def checked_root(root: Path) -> Path:
    valid(root.is_absolute() and root.parent == ROOT_PARENT
        and ROOT_NAME.fullmatch(root.name) is not None and root.is_dir()
        and not root.is_symlink(), "root_invalid")
    info = root.stat()
    valid(info.st_uid == ROOT_UID and not info.st_mode & 0o077, "root_permission_invalid")
    return root


def source_constants(runner: Path) -> dict:
    wanted = {"PINS", "SIWC_PINS", "ACCEPTED_FILES", "ACCEPTED_MAP_SHA256",
              "DIAGNOSTIC_HELPER_SHA256"}
    values = {}
    for node in ast.parse(runner.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                values[name] = ast.literal_eval(node.value)
    valid(set(values) == wanted, "runner_constants_invalid")
    return values


def receipt(root: Path, receipt_sha256: str) -> dict:
    valid(type(receipt_sha256) is str and HEX.fullmatch(receipt_sha256) is not None,
        "receipt_sha_invalid")
    receipt_path = root / "launch-receipt.json"
    value = read(receipt_path, root, 8192)
    valid(sha(receipt_path) == receipt_sha256 and type(value) is dict
        and set(value) == RECEIPT_FIELDS, "receipt_invalid")
    runner = root / "code" / RUNNER_RELATIVE
    valid(file_at(runner, root, 2_000_000) and sha(runner) == RUNNER_SHA256,
        "runner_drift")
    launcher = root / LAUNCHER_NAME
    valid(file_at(launcher, root, 2_000_000) and sha(launcher) == LAUNCHER_SHA256,
        "launcher_drift")
    constants = source_constants(runner)
    source_map = root / "source-map.json"
    valid(file_at(source_map, root, 1_000_000) and sha(source_map) == INVENTORY_SHA256,
        "inventory_drift")
    inventory = read(source_map, root, 1_000_000)
    entries = inventory.get("source_sha256") if type(inventory) is dict else None
    valid(type(entries) is dict and len(entries) == 514
        and len(entries) == constants["ACCEPTED_FILES"], "inventory_shape_invalid")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    valid(hashlib.sha256(encoded).hexdigest() == MAP_SHA256
        and constants["ACCEPTED_MAP_SHA256"] == MAP_SHA256,
        "inventory_map_drift")
    candidate = root / "candidate"
    valid(candidate.is_dir() and not candidate.is_symlink(), "candidate_missing")
    actual = {}
    for path in candidate.rglob("*"):
        relative = path.relative_to(candidate)
        if (any(part in {"__pycache__", ".pytest_cache", ".git"} for part in relative.parts)
                or path.suffix in {".pyc", ".pyo"}):
            continue
        valid(not path.is_symlink(), "candidate_symlink")
        if path.is_dir():
            continue
        valid(file_at(path, root, 2_000_000), "candidate_special_file")
        actual[relative.as_posix()] = sha(path)
    valid(actual == entries, "candidate_source_drift")
    sources = {**constants["PINS"], **constants["SIWC_PINS"],
        "benchmarks/lme_diagnostic.py": HELPER_SHA256, RUNNER_RELATIVE: RUNNER_SHA256}
    valid(constants["DIAGNOSTIC_HELPER_SHA256"] == HELPER_SHA256,
        "helper_pin_invalid")
    for relative, digest in sources.items():
        path = root / "code" / relative
        valid(file_at(path, root, 2_000_000) and sha(path) == digest, "pinned_source_drift")
    rows = value["selected_row_sha256"]
    valid(type(rows) is list and len(rows) == 4 and all(
        type(item) is str and HEX.fullmatch(item) is not None for item in rows),
        "receipt_rows_invalid")
    unit = root.name.removeprefix(".") + ".service"
    expected = {"schema": "siwc-lme-diagnostic-launch-v1", "root": str(root),
        "unit": unit, "expected_cgroup":
            "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "source_sha256": sources, "candidate_map_sha256": MAP_SHA256,
        "inventory_sha256": INVENTORY_SHA256, "dataset_sha256": DATASET_SHA256,
        "runtime_path": str(RUNTIME_PATH), "runtime_sha256": RUNTIME_SHA256,
        "runtime_site_sha256": RUNTIME_SITE_SHA256, "runtime_site_files": 309,
        "owner_state": str(OWNER_STATE), "grant_identity_sha256": GRANT_IDENTITY_SHA256,
        "selected_source_order": "first_n", "selected_row_sha256": rows,
        "selected_count": 4, "workers": 4, "indexing_seconds": 10_800,
        "output_dir": str(root / "run"), "limits": LIMITS, "invocation_seconds": 120.0,
        "model": "gpt-5.6-luna", "reasoning": "low", "auth": "siwc_oauth",
        "endpoint": "https://api.openai.com/v1/responses", "store": False,
        "stream": True, "billing_policy": BILLING_POLICY,
        "automatic_topup_user_attested_off": True, "reload_allowed": False,
        "api_fallback_allowed": False, "one_shot": True}
    valid(typed_equal(value, expected), "receipt_identity_invalid")
    valid(receipt_path.read_bytes() == json.dumps(expected, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("ascii"), "receipt_canonical_invalid")
    valid(DATASET.is_file() and not DATASET.is_symlink() and sha(DATASET) == DATASET_SHA256,
        "dataset_drift")
    valid(RUNTIME_PATH.is_file() and not RUNTIME_PATH.is_symlink()
        and sha(RUNTIME_PATH) == RUNTIME_SHA256, "runtime_drift")
    site = RUNTIME_PATH.parent.parent / "lib/python3.13/site-packages"
    valid(site.is_dir() and not site.is_symlink(), "runtime_site_drift")
    files = {}
    for path in site.rglob("*"):
        valid(not path.is_symlink(), "runtime_site_drift")
        if path.suffix == ".pyc":
            continue
        valid(path.is_dir() or file_at(path, site, RUNTIME_SITE_MAX_FILE_BYTES), "runtime_site_drift")
        if path.is_file():
            files[str(path.relative_to(site))] = sha(path)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    valid(len(files) == 309 and digest == RUNTIME_SITE_SHA256, "runtime_site_drift")
    return value


def checkpoint(value: Any, receipt: dict) -> tuple[dict | None, int, int, int, dict]:
    valid(type(value) is dict and value.get("schema") == CHECKPOINT_SCHEMA
        and value.get("status") in {"running", "complete"}
        and value.get("scored") is True and value.get("verdict_key") == "correct"
        and type(value.get("manifest")) is dict and type(value.get("expected_ids")) is list
        and type(value.get("entries")) is dict, "checkpoint_invalid")
    ids, manifest, entries = value["expected_ids"], value["manifest"], value["entries"]
    valid(len(ids) == 4 and all(type(item) is str and 0 < len(item) <= 128 for item in ids)
        and len(set(ids)) == 4 and set(entries).issubset(ids), "checkpoint_ids_invalid")
    run_id = manifest.get("run_id")
    valid(type(run_id) is str and run_id == value.get("run_id")
        and run_id == canonical_hash({key: item for key, item in manifest.items() if key != "run_id"})
        and manifest.get("schema") == RUN_SCHEMA and manifest.get("mode") == MODE
        and manifest.get("canonical_r9_artifact") is False
        and manifest.get("official_model_score") is False
        and manifest.get("candidate_map_sha256") == MAP_SHA256
        and manifest.get("dataset_sha256") == DATASET_SHA256
        and manifest.get("selected_row_sha256") == receipt["selected_row_sha256"]
        and manifest.get("selected_source_order") == "first_n"
        and manifest.get("expected_count") == 4
        and manifest.get("expected_ids_hash") == canonical_hash(ids)
        and manifest.get("scored_run") is True
        and manifest.get("diagnostic_helper_sha256") == HELPER_SHA256
        and manifest.get("billing_policy") == BILLING_POLICY
        and manifest.get("runner_sha256") == RUNNER_SHA256
        and manifest.get("transport_sha256") == receipt["source_sha256"]["benchmarks/chatgpt_plan_responses_v9.py"]
        and manifest.get("transport_v7_sha256") == receipt["source_sha256"]["benchmarks/chatgpt_plan_responses_v7.py"]
        and manifest.get("transport_v6_sha256") == receipt["source_sha256"]["benchmarks/chatgpt_plan_responses_v6.py"]
        and manifest.get("bridge_sha256") == receipt["source_sha256"]["benchmarks/chatgpt_plan_lme_v4.py"]
        and manifest.get("grant_identity_sha256") == GRANT_IDENTITY_SHA256
        and type(manifest.get("rerolls")) is int and manifest["rerolls"] == 0,
        "checkpoint_identity_invalid")
    limits = manifest.get("limits")
    valid(type(limits) is dict and all(typed_equal(limits.get(name), dict(zip(
        ("turns", "known_tokens", "seconds"), LIMITS[name]))) for name in LIMITS)
        and type(limits.get("indexing_seconds")) in (int, float)
        and limits["indexing_seconds"] == 10_800
        and type(limits.get("workers")) is int and limits["workers"] == 4,
        "checkpoint_limits_invalid")
    completed = failed = unhealthy = degraded = 0
    for qid, entry in entries.items():
        valid(type(entry) is dict and entry.get("status") in {"completed", "failed"}
            and type(entry.get("attempts")) is int and entry["attempts"] == 1
            and type(entry.get("row")) is dict
            and entry["row"].get("question_id") == qid, "checkpoint_entry_invalid")
        row = entry["row"]
        if entry["status"] == "completed":
            valid(type(row.get("correct")) is bool and row.get("benchmark_failure") is None
                and row.get("diagnostic_kind") in {"strict_healthy", "summary_degradation", "semantic_quarantine"}
                and type(row.get("strict_indexing_healthy")) is bool
                and (row["diagnostic_kind"] == "semantic_quarantine") is (row["strict_indexing_healthy"] is False)
                and integer(row.get("quarantined_chunks"))
                and integer(row.get("summary_degraded_sessions"))
                and type(row.get("context_sha")) is str
                and HEX.fullmatch(row["context_sha"]) is not None,
                "checkpoint_projection_invalid")
            valid((row["diagnostic_kind"] == "semantic_quarantine") == (row["quarantined_chunks"] > 0)
                and (row["diagnostic_kind"] != "summary_degradation"
                     or row["summary_degraded_sessions"] > 0),
                "checkpoint_projection_inconsistent")
            completed += 1
            unhealthy += not row["strict_indexing_healthy"]
            degraded += row["summary_degraded_sessions"]
        else:
            valid(type(entry.get("failure")) is str and entry["failure"] in QUESTION_FAILURE_CODES
                and row.get("benchmark_failure") == entry["failure"]
                and row.get("correct") is False,
                "checkpoint_failure_invalid")
            failed += 1
    if value["status"] != "complete":
        return None, completed, unhealthy, degraded, value
    counts = value.get("counts")
    valid(type(counts) is dict and set(counts) == COUNT_FIELDS
        and all(integer(number) for number in counts.values())
        and counts["expected"] == 4 and counts["attempted"] == len(entries)
        and counts["unique_attempted"] == len(entries)
        and counts["total_attempts"] == len(entries)
        and counts["completed"] == completed and counts["failed"] == failed
        and counts["missing"] == 4 - len(entries), "checkpoint_counts_invalid")
    return counts, completed, unhealthy, degraded, value


def resource(value: Any, *, allow_none: bool = False) -> dict | None:
    if value is None and allow_none:
        return None
    valid(type(value) is dict and set(value) == {"current", "peak", "limit", "denials"}
        and all(integer(value[key], 1_000_000) for key in value)
        and value["limit"] == 256 and value["current"] <= value["peak"] <= 256,
        "resource_observation_invalid")
    return value


def wire_observation(value: Any) -> None:
    keys = {"header_defect", "parsed_header_count", "transfer_encoding",
        "content_encoding", "body_prefix", "body_bytes", "body_truncated",
        "sse_validation"}
    valid(type(value) is dict and set(value) == keys, "wire_observation_invalid")
    count, size, truncated = value["parsed_header_count"], value["body_bytes"], value["body_truncated"]
    prefix, sse = value["body_prefix"], value["sse_validation"]
    valid(value["header_defect"] in {"none", "missing_separator", "first_continuation", "other", "unknown"}
        and (count is None or integer(count, 100))
        and value["transfer_encoding"] in {"missing", "chunked", "identity", "other", "invalid"}
        and value["content_encoding"] in {"missing", "identity", "gzip", "deflate", "br", "other", "invalid"}
        and prefix in {"empty", "sse_prefix", "html_prefix", "http_prefix", "json_prefix", "text", "binary", "unknown"}
        and integer(size, 65_537) and type(truncated) is bool
        and sse in {"not_checked", "validated_completion", "invalid", "truncated"}
        and truncated == (size > 65_536) and (prefix == "empty") == (size == 0)
        and (prefix == "sse_prefix" or sse == "not_checked")
        and (prefix != "sse_prefix" or sse in ({"truncated"} if truncated else {"validated_completion", "invalid"})),
        "wire_observation_invalid")


def stream_observation(value: Any) -> None:
    keys = {"event_type", "terminal_status", "terminal_model_matches",
        "terminal_output_kind", "terminal_channel", "terminal_content_kind",
        "terminal_output_state", "finalized_item_count", "output_reconstructed"}
    events = {"response.created", "response.in_progress", "response.queued",
        "response.output_item.added", "response.output_item.done",
        "response.content_part.added", "response.content_part.done",
        "response.output_text.delta", "response.output_text.done",
        "response.reasoning_text.delta", "response.reasoning_text.done",
        "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
        "response.refusal.delta", "response.refusal.done",
        "response.function_call_arguments.delta", "response.function_call_arguments.done",
        "response.output_text.annotation.added", "response.completed", "response.failed",
        "response.incomplete", "error", "unknown"}
    valid(type(value) is dict and set(value) == keys
        and value["event_type"] in events
        and value["terminal_status"] in {"missing", "completed", "incomplete", "failed", "other"}
        and (value["terminal_model_matches"] is None or type(value["terminal_model_matches"]) is bool)
        and value["terminal_output_kind"] in {"missing", "message", "reasoning", "mixed", "other"}
        and value["terminal_channel"] in {"missing", "final", "final_answer", "analysis", "commentary", "mixed", "other"}
        and value["terminal_content_kind"] in {"missing", "output_text", "refusal", "reasoning_text", "mixed", "other"}
        and value["terminal_output_state"] in {"unseen", "missing", "null", "empty", "list", "invalid"}
        and integer(value["finalized_item_count"], 4096)
        and type(value["output_reconstructed"]) is bool
        and (not value["output_reconstructed"] or
             (value["terminal_output_state"] in {"missing", "null", "empty"}
              and value["finalized_item_count"] > 0)), "stream_observation_invalid")


def timeout_observation(value: Any) -> None:
    """Mirror transport v7's closed snapshot sanitizer without importing it."""
    phases = {"unknown", "child_entry", "request_send", "headers_wait",
        "response_check", "stream_read", "parse", "response_close", "result_ipc"}
    sites = {"none", "result_wait", "result_recv", "deadline_after_recv"}
    child_keys = {"last_progress_elapsed_ms", "wire_bytes", "event_count",
        "completion_seen", "result_ready", "result_ipc_started"}
    keys = child_keys | {"child_phase", "parent_timeout_site", "parent_elapsed_ms",
        "elapsed_saturated", "snapshot_valid", "timeout_allowance_ms",
        "child_alive_when_sampled"}
    valid(type(value) is dict and set(value) == keys
        and type(value["child_phase"]) is str and value["child_phase"] in phases
        and type(value["parent_timeout_site"]) is str and value["parent_timeout_site"] in sites,
        "timeout_observation_invalid")
    for key, maximum in (("last_progress_elapsed_ms", 121_000),
                         ("parent_elapsed_ms", 121_000),
                         ("timeout_allowance_ms", 120_000),
                         ("wire_bytes", 16_000_000),
                         ("event_count", 16_000_000 // 9 + 1)):
        number = value[key]
        valid((number is None and key not in {"parent_elapsed_ms", "timeout_allowance_ms"})
            or integer(number, maximum), "timeout_observation_invalid")
    valid(all(type(value[key]) is bool for key in ("elapsed_saturated", "snapshot_valid"))
        and all(value[key] is None or type(value[key]) is bool for key in
            ("completion_seen", "result_ready", "result_ipc_started", "child_alive_when_sampled")),
        "timeout_observation_invalid")
    if value["snapshot_valid"]:
        valid(value["child_phase"] != "unknown"
            and all(value[key] is not None for key in child_keys)
            and not (value["completion_seen"] and value["event_count"] == 0)
            and not (value["result_ipc_started"] and not value["result_ready"])
            and (value["child_phase"] != "result_ipc" or
                 (value["result_ready"] and value["result_ipc_started"])),
            "timeout_observation_invalid")
    else:
        valid(value["child_phase"] == "unknown"
            and all(value[key] is None for key in child_keys), "timeout_observation_invalid")
    valid(value["parent_timeout_site"] != "none" or
        value["child_alive_when_sampled"] is None, "timeout_observation_invalid")


def first_failure(value: Any) -> dict | None:
    if value is None:
        return None
    valid(type(value) is dict and {"code", "phase", "turn_admitted", "unknown_usage"} <= set(value)
        and set(value) <= {"code", "phase", "turn_admitted", "unknown_usage", "http_status",
            "body_shape", "media_type_class", "wire_observation", "stream_observation",
            "resource_observation", "underlying_code", "timeout_observation"}
        and type(value["code"]) is str and value["code"] in STOP_CODES
        and type(value["phase"]) is str and value["phase"] in {"admission", "http"}
        and type(value["turn_admitted"]) is bool and type(value["unknown_usage"]) is bool
        and value["turn_admitted"] == (value["phase"] == "http")
        and value["unknown_usage"] == value["turn_admitted"],
        "first_failure_invalid")
    # A failed HTTP turn has unknown usage unless the transport returned complete
    # reconciled usage. The bridge's first failure always records this truthfully.
    if "http_status" in value:
        valid(type(value["http_status"]) is int and 100 <= value["http_status"] <= 599,
            "first_failure_http_invalid")
    if "body_shape" in value:
        valid(type(value["body_shape"]) is str and value["body_shape"] in {"empty", "error_object", "detail", "other_json", "non_json", "oversized", "sse_event"},
            "first_failure_body_invalid")
    if "media_type_class" in value:
        valid(type(value["media_type_class"]) is str and value["media_type_class"] in {"missing", "sse", "json", "html", "text", "other", "invalid"},
            "first_failure_media_invalid")
    if "resource_observation" in value:
        resource(value["resource_observation"], allow_none=True)
    if "timeout_observation" in value:
        valid(value["phase"] == "http" and value["turn_admitted"] is True
            and value["unknown_usage"] is True
            and (value["code"] == "timeout" or value.get("underlying_code") == "timeout"),
            "first_failure_timeout_invalid")
        timeout_observation(value["timeout_observation"])
    if "underlying_code" in value:
        valid(value["code"] in {"resource_observer_unverified", "resource_task_denial"}
            and type(value["underlying_code"]) is str
            and value["underlying_code"] in STOP_CODES, "first_failure_underlying_invalid")
    if "wire_observation" in value:
        wire_observation(value["wire_observation"])
    if "stream_observation" in value:
        stream_observation(value["stream_observation"])
    return value


def summary(value: Any) -> dict:
    valid(type(value) is dict and set(value) == SUMMARY_FIELDS
        and value.get("schema") == "siwc_lme_summary_v2", "siwc_summary_invalid")
    for key in ("calls", "successes", "failures", "internal_http_attempts", "admitted_turns", "known_tokens"):
        valid(integer(value[key], 1_000_000_000_000 if key == "known_tokens" else 1_000_000),
            "siwc_summary_invalid")
    valid(value["calls"] == value["successes"] + value["failures"]
        and value["internal_http_attempts"] <= value["admitted_turns"]
        and value["successes"] <= value["internal_http_attempts"]
        and value["provider_internal_retries_known"] is False
        and type(value["usage_complete"]) is bool
        and type(value["timing_saturated"]) is bool,
        "siwc_summary_invalid")
    timing = value["timing_seconds"]
    valid(type(timing) is dict and set(timing) == {"total", "admission", "http"}
        and all(finite(timing[key]) for key in timing), "siwc_summary_timing_invalid")
    first_failure(value["first_failure"])
    valid((value["failures"] == 0) == (value["first_failure"] is None)
        and (value["failures"] == 0) == (value["last_failure_code"] is None)
        and (value["last_failure_code"] is None or value["last_failure_code"] in STOP_CODES),
        "siwc_summary_failure_invalid")
    return value


def pilot(value: Any, observations: dict, budget: dict) -> dict | None:
    if value is None:
        return None
    valid(type(value) is dict and set(value) == {"schema", "canary", "questions", "aggregate"}
        and value["schema"] == "siwc_lme_pilot_projection_v2"
        and type(value["canary"]) is dict and type(value["questions"]) is list
        and len(value["questions"]) == 4, "siwc_pilot_invalid")
    rows = [value["canary"], *value["questions"]]
    totals = {key: 0 for key in ("calls", "successes", "failures", "internal_http_attempts",
        "admitted_turns", "known_tokens")}
    for index, row in enumerate(rows):
        prefix = "canary" if index == 0 else f"question.{index-1}"
        valid(type(row) is dict and set(row) == {"question_id", "ordinary", "structured", "ledger"}
            and row["question_id"] == ("canary" if index == 0 else f"q-{index-1:04d}"),
            "siwc_pilot_row_invalid")
        a, b = summary(row["ordinary"]), summary(row["structured"])
        valid(a == observations[prefix + ".ordinary"]["summary"]
            and b == observations[prefix + ".structured"]["summary"], "siwc_pilot_view_invalid")
        ledger = row["ledger"]
        valid(type(ledger) is dict and set(ledger) == {"admitted_turns", "known_tokens", "usage_complete"}
            and integer(ledger["admitted_turns"]) and integer(ledger["known_tokens"])
            and type(ledger["usage_complete"]) is bool
            and all(view[key] == ledger[key] for view in (a, b)
                    for key in ("admitted_turns", "known_tokens", "usage_complete"))
            and a["internal_http_attempts"] + b["internal_http_attempts"] <= ledger["admitted_turns"]
            and budget["questions"][row["question_id"]]["turns"] == ledger["admitted_turns"]
            and budget["questions"][row["question_id"]]["known_tokens"] == ledger["known_tokens"],
            "siwc_pilot_ledger_invalid")
        for key in ("calls", "successes", "failures", "internal_http_attempts"):
            totals[key] += a[key] + b[key]
        totals["admitted_turns"] += ledger["admitted_turns"]
        totals["known_tokens"] += ledger["known_tokens"]
    valid(typed_equal(value["aggregate"], totals)
        and all(integer(value["aggregate"][key],
            1_000_000_000_000 if key == "known_tokens" else 1_000_000)
            for key in totals)
        and totals["admitted_turns"] == budget["turns"]
        and totals["known_tokens"] == budget["known_tokens"], "siwc_pilot_aggregate_invalid")
    return value


def stage_accounting(stages: Any, checkpoint: dict, budget: dict, clean: bool) -> None:
    valid(type(stages) is dict and set(stages) == set(checkpoint["expected_ids"])
        and type(clean) is bool, "terminal_stage_invalid")
    expected_clean = True
    for index, qid in enumerate(checkpoint["expected_ids"]):
        entry = checkpoint["entries"].get(qid)
        item = stages[qid]
        if entry is None or entry["status"] != "completed":
            valid(item is None or type(item) is dict, "terminal_stage_invalid")
            if item is None:
                expected_clean = False
                continue
        else:
            valid(type(item) is dict, "terminal_stage_invalid")
        valid(set(item).issubset(STAGES), "terminal_stage_label_invalid")
        totals = {"turns": 0, "known_tokens": 0}
        for value in item.values():
            valid(type(value) is dict and set(value) == {"attempts", "returned", "turns", "known_tokens"}
                and all(integer(number, 1_000_000_000_000 if key == "known_tokens" else 1_000_000)
                        for key, number in value.items())
                and value["attempts"] >= value["turns"] >= value["returned"],
                "terminal_stage_counts_invalid")
            totals["turns"] += value["turns"]
            totals["known_tokens"] += value["known_tokens"]
        if entry is not None and entry["status"] == "completed":
            q = budget["questions"].get(f"q-{index:04d}")
            valid(type(q) is dict and totals["turns"] == q["turns"]
                and totals["known_tokens"] == q["known_tokens"],
                "terminal_stage_ledger_invalid")
    valid(clean is (expected_clean and
        sum(entry["status"] == "completed" for entry in checkpoint["entries"].values()) == 4),
        "terminal_stage_clean_invalid")


def terminal(value: Any, checkpoint: dict, receipt: dict, counts: dict,
             completed: int, unhealthy: int) -> dict:
    fields = {"schema", "run_id", "canonical_r9_artifact", "official_model_score",
        "selected_denominator", "scored_count", "correct_count", "incorrect_count",
        "quality_accuracy_full_selected", "failed_or_unscored_count", "strict_unhealthy_count",
        "canary", "campaign_stop", "owner_failure", "stage_accounting",
        "stage_accounting_clean", "checkpoint_counts", "budget", "siwc_observations",
        "siwc_pilot_projection", "diagnostic_complete"}
    valid(type(value) is dict and set(value) == fields and value["schema"] == RUN_SCHEMA
        and value["run_id"] == checkpoint["run_id"]
        and value["canonical_r9_artifact"] is False
        and value["official_model_score"] is False
        and type(value["selected_denominator"]) is int and value["selected_denominator"] == 4
        and type(value["scored_count"]) is int and value["scored_count"] == completed
        and type(value["failed_or_unscored_count"]) is int
        and value["failed_or_unscored_count"] == 4-completed
        and type(value["strict_unhealthy_count"]) is int
        and value["strict_unhealthy_count"] == unhealthy
        and typed_equal(value["checkpoint_counts"], counts)
        and type(value["diagnostic_complete"]) is bool,
        "terminal_identity_invalid")
    correct, incorrect = value["correct_count"], value["incorrect_count"]
    valid(integer(correct, 4) and integer(incorrect, 4) and correct + incorrect == completed
        and correct == sum(entry["row"]["correct"] for entry in checkpoint["entries"].values()
                           if entry["status"] == "completed"), "terminal_score_invalid")
    accuracy = value["quality_accuracy_full_selected"]
    valid((type(accuracy) is float and accuracy == correct/4) if completed == 4
        else accuracy is None, "terminal_accuracy_invalid")
    canary = value["canary"]
    valid(canary is None or (type(canary) is dict
        and type(canary.get("structural_valid")) is bool
        and type(canary.get("model_gold_match")) is bool
        and integer(canary.get("completion_calls"))), "terminal_canary_invalid")
    stop = value["campaign_stop"]
    valid(stop is None or (type(stop) is str and stop in STOP_CODES), "terminal_stop_invalid")
    owner = value["owner_failure"]
    valid(owner is None or (type(owner) is dict and set(owner) == {"phase", "code"}
        and owner["phase"] == "owner_open" and owner["code"] in OWNER_CODES),
        "terminal_owner_failure_invalid")
    stages = value["stage_accounting"]
    budget = value["budget"]
    valid(type(budget) is dict and all(integer(budget.get(key)) for key in
        ("turns", "known_tokens", "reserved", "in_flight"))
        and type(budget.get("usage_complete")) is bool
        and type(budget.get("stopped")) is bool
        and (budget.get("stop_code") is None or budget["stop_code"] in STOP_CODES)
        and type(budget.get("questions")) is dict
        and budget.get("known_tokens_scope") == "completed_turns_only_failed_turn_usage_unknown"
        and budget.get("token_cap_kind") == "stop_before_next_observed_usage"
        and set(budget["questions"]).issubset({"canary", *(f"q-{i:04d}" for i in range(4))}),
        "terminal_budget_invalid")
    valid(budget["turns"] <= LIMITS["campaign"][0]
        and type(budget.get("timings")) is dict
        and set(budget["timings"]) == {"preflight_seconds", "model_seconds", "cleanup_seconds"}
        and all(finite(item) for item in budget["timings"].values()),
        "terminal_budget_timing_invalid")
    for q in budget["questions"].values():
        valid(type(q) is dict and all(integer(q.get(key)) for key in
            ("turns", "known_tokens", "in_flight"))
            and type(q.get("usage_complete")) is bool and type(q.get("stopped")) is bool,
            "terminal_question_budget_invalid")
    valid(sum(q["turns"] for q in budget["questions"].values()) == budget["turns"]
        and sum(q["known_tokens"] for q in budget["questions"].values()) == budget["known_tokens"]
        and sum(q["in_flight"] for q in budget["questions"].values()) == budget["in_flight"],
        "terminal_budget_ledger_invalid")
    stage_accounting(stages, checkpoint, budget, value["stage_accounting_clean"])
    fault = budget.get("first_failure")
    first_failure(fault)
    if fault is not None:
        valid(budget["stopped"] is True and budget["stop_code"] == fault["code"],
            "first_failure_stop_mismatch")
    resource_fault = budget.get("resource_fault")
    valid(resource_fault is None or resource_fault in {"resource_observer_unverified", "resource_task_denial"},
        "resource_fault_invalid")
    sample = resource(budget.get("resource_observation"),
        allow_none=resource_fault == "resource_observer_unverified")
    valid(sample is not None or resource_fault == "resource_observer_unverified",
        "resource_terminal_unverified")
    if sample is not None:
        valid(sample["denials"] == 0 or resource_fault == "resource_task_denial",
            "resource_denial_stop_mismatch")
    observations = value["siwc_observations"]
    valid(type(observations) is dict and set(observations) == set(SLOTS),
        "siwc_observation_slots_invalid")
    observed = []
    for slot in SLOTS:
        item = observations[slot]
        valid(type(item) is dict and item.get("status") in {"unknown", "observed"}
            and set(item) == ({"status"} if item["status"] == "unknown" else {"status", "summary"}),
            "siwc_observation_invalid")
        if item["status"] == "observed":
            observed.append(summary(item["summary"]))
    projection = value["siwc_pilot_projection"]
    if projection is not None:
        valid(len(observed) == 10, "siwc_pilot_incomplete_views")
        pilot(projection, observations, budget)
    clean_observations = (len(observed) == 10 and projection is not None
        and all(item["failures"] == 0 and not item["timing_saturated"]
                and item["usage_complete"] for item in observed)
        and projection["aggregate"]["successes"] == budget["turns"]
        and projection["aggregate"]["known_tokens"] == budget["known_tokens"]
        and projection["aggregate"]["internal_http_attempts"] == budget["turns"]
        and canary is not None
        and sum(observations[f"canary.{route}"]["summary"]["successes"]
                for route in ("ordinary", "structured")) == canary["completion_calls"])
    semantic_clean = (completed == 4 and counts["completed"] == 4
        and counts["failed"] == counts["missing"] == 0
        and stop is None and owner is None and canary is not None
        and canary["structural_valid"] is True
        and budget["reserved"] == budget["in_flight"] == 0
        and budget["usage_complete"] is True and budget["stopped"] is False
        and budget["stop_code"] is None and fault is None and resource_fault is None
        and sample is not None and sample["denials"] == 0
        and set(budget["questions"]) == {"canary", *(f"q-{i:04d}" for i in range(4))}
        and all(q["turns"] > 0 and q["usage_complete"] is True
                and q["in_flight"] == 0 and q["stopped"] is False
                for q in budget["questions"].values())
        and value["stage_accounting_clean"] is True and clean_observations)
    valid(value["diagnostic_complete"] is semantic_clean, "terminal_complete_mismatch")
    return value


def group_path_safe(group: Path) -> bool:
    try:
        root = CGROUP_ROOT.resolve(strict=True)
        if (CGROUP_ROOT.is_symlink() or not group.is_relative_to(CGROUP_ROOT)
                or not group.resolve().is_relative_to(root)):
            return False
        node = group
        while node != CGROUP_ROOT:
            if node.is_symlink():
                return False
            node = node.parent
        return True
    except (OSError, ValueError):
        return False


def cgroup_policy(group: Path) -> bool:
    try:
        if (not group_path_safe(group) or not group.is_dir()
                or (group / "memory.max").read_text().strip() != "4294967296"
                or (group / "pids.max").read_text().strip() != "256"):
            return False
        cpu = (group / "cpu.max").read_text().split()
        return len(cpu) == 2 and all(item.isdecimal() for item in cpu) and int(cpu[1]) > 0 and int(cpu[0]) == 2*int(cpu[1])
    except (OSError, ValueError):
        return False


def recursive_empty(group: Path) -> bool:
    try:
        if not group_path_safe(group):
            return False
        if not group.exists():
            return True
        if not group.is_dir():
            return False
        pending, seen = [group], 0
        while pending:
            node = pending.pop()
            seen += 1
            if seen > 256 or node.is_symlink():
                return False
            if any((node / name).read_text().strip() for name in ("cgroup.procs", "cgroup.threads")):
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


def live_resource(group: Path) -> dict:
    valid(group_path_safe(group) and cgroup_policy(group), "resource_cgroup_invalid")
    sample = {}
    for key, name in (("current", "pids.current"), ("peak", "pids.peak"), ("limit", "pids.max")):
        raw = (group / name).read_text().strip()
        valid(raw.isdecimal(), "resource_counter_unverified")
        sample[key] = int(raw)
    pairs = [line.split() for line in (group / "pids.events").read_text().splitlines()]
    valid(len(pairs) <= 32 and all(len(pair) == 2 for pair in pairs)
        and len(dict(pairs)) == len(pairs), "resource_counter_unverified")
    raw = dict(pairs).get("max")
    valid(type(raw) is str and raw.isdecimal(), "resource_counter_unverified")
    sample["denials"] = int(raw)
    return resource(sample)


def runtime(receipt: dict) -> tuple[str, bool]:
    """Return live/clean state and independent failed-exit cleanup evidence."""
    if sys.platform != "linux" or os.getuid() != ROOT_UID:
        return "unverified", False
    try:
        runtime_info, bus_info = USER_RUNTIME.lstat(), (USER_RUNTIME / "bus").lstat()
        if (not stat.S_ISDIR(runtime_info.st_mode) or runtime_info.st_uid != ROOT_UID
                or runtime_info.st_mode & 0o077 or not stat.S_ISSOCK(bus_info.st_mode)
                or bus_info.st_uid != ROOT_UID):
            return "unverified", False
        fields = ("ActiveState", "SubState", "MainPID", "ControlGroup", "NRestarts",
            "Result", "ExecMainStatus", "MemoryMax", "TasksMax", "CPUQuotaPerSecUSec",
            "KillMode", "Restart", "RemainAfterExit", "OOMPolicy", "RuntimeMaxUSec",
            "TimeoutStopUSec")
        env = {"HOME": str(ROOT_PARENT), "PATH": "/usr/bin:/bin",
            "XDG_RUNTIME_DIR": str(USER_RUNTIME),
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(USER_RUNTIME / "bus")}
        done = subprocess.run(["/usr/bin/systemctl", "--user", "show", receipt["unit"],
            "--property=" + ",".join(fields), "--no-pager"], capture_output=True,
            text=True, timeout=10, check=True, env=env)
        if len(done.stdout) > 8192:
            return "unverified", False
        pairs = [line.split("=", 1) for line in done.stdout.splitlines()]
        if len(pairs) != len(fields) or any(len(pair) != 2 for pair in pairs):
            return "unverified", False
        values = dict(pairs)
        if (set(values) != set(fields) or values["NRestarts"] != "0"
                or values["MemoryMax"] != "4294967296" or values["TasksMax"] != "256"
                or values["CPUQuotaPerSecUSec"] not in {"2s", "2.000s"}
                or values["KillMode"] != "control-group" or values["Restart"] != "no"
                or values["RemainAfterExit"] != "yes" or values["OOMPolicy"] != "kill"
                or values["RuntimeMaxUSec"] not in {"4h 2min 10s", "4h 2min 10.000s", "14530s", "14530.000s"}
                or values["TimeoutStopUSec"] not in {"10s", "10.000s"}):
            return "unverified", False
        group = CGROUP_ROOT / receipt["expected_cgroup"].lstrip("/")
        if not group_path_safe(group):
            return "unverified", False
        if (values["ActiveState"] == "active" and values["SubState"] == "running"
                and values["ControlGroup"] == receipt["expected_cgroup"]
                and values["MainPID"].isdecimal() and int(values["MainPID"]) > 0
                and cgroup_policy(group)
                and values["MainPID"] in (group / "cgroup.procs").read_text().splitlines()):
            return "running_verified", False
        terminal_state = (values["ActiveState"], values["SubState"])
        if (terminal_state not in {("active", "exited"), ("inactive", "dead"), ("failed", "failed")}
                or values["MainPID"] != "0"
                or values["ControlGroup"] not in {"", receipt["expected_cgroup"]}
                or (group.exists() and (not cgroup_policy(group) or not recursive_empty(group)))):
            return "unverified", False
        if values["Result"] == "success" and values["ExecMainStatus"] == "0":
            return "clean_exit", True
        if (values["Result"] in {"exit-code", "signal", "core-dump", "oom-kill", "timeout"}
                and values["ExecMainStatus"].isdecimal()):
            return "failed_exit_cleaned", True
        return "unverified", False
    except (OSError, ValueError, subprocess.SubprocessError):
        return "unverified", False


def inspect(root: Path, receipt_sha256: str) -> dict:
    root = checked_root(root)
    pinned = receipt(root, receipt_sha256)
    output = {"schema": SCHEMA, "status": "prepared_not_launched",
        "completed_diagnostic_and_clean": False, "runtime_cleanup_verified": False,
        "selected_denominator": 4, "scored_count": None, "correct_count": None,
        "strict_indexing_healthy_for_all": None, "summary_degraded_sessions_total": None,
        "canary_model_gold_match": None, "canary_structural_valid": None,
        "campaign_stop": None, "owner_failure": None, "budget_stop_code": None,
        "first_failure": None, "failed_count": None,
        "question_failure_codes": [None, None, None, None], "known_turns": None,
        "known_tokens": None, "usage_complete": None, "stage_timing_available": False,
        "resource_observation": None, "resource_fault": None,
        "siwc_observations": None, "siwc_pilot_projection": None}
    attempt_path = root / "launch-attempt.json"
    run = root / "run"
    if not attempt_path.exists() and not attempt_path.is_symlink():
        valid(not run.exists() and not run.is_symlink()
            and not (root / EXECUTION_MARKER).exists()
            and not (root / "launch-command-result.json").exists(),
            "unlaunched_state_drift")
        return output
    valid(read(attempt_path, root, 512) == {"receipt_sha256": receipt_sha256,
        "one_shot": True}, "launch_attempt_invalid")
    output["status"] = "launch_attempted"
    execution = root / EXECUTION_MARKER
    started = execution.exists() or execution.is_symlink()
    if started:
        expected = {"receipt_sha256": receipt_sha256, "execution_started": True}
        valid(read(execution, root, 512) == expected and execution.read_bytes() ==
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("ascii"),
            "execution_marker_invalid")
    command_path = root / "launch-command-result.json"
    command = None
    if command_path.exists() or command_path.is_symlink():
        command = read(command_path, root, 512)
        valid(type(command) is dict and set(command) == {"returncode"}
            and type(command["returncode"]) is int, "launch_command_invalid")
    runtime_state, cleaned = runtime(pinned)
    output["runtime_cleanup_verified"] = cleaned
    if command is not None and command["returncode"] != 0 and not started:
        output["status"] = "launch_dispatch_failed"
        return output
    if runtime_state == "running_verified":
        output["resource_observation"] = live_resource(
            CGROUP_ROOT / pinned["expected_cgroup"].lstrip("/"))
        if output["resource_observation"]["denials"]:
            output["status"] = "resource_task_denial_observed"
            return output
    checkpoint_path = run / "diagnostic-checkpoint.json"
    result_path = run / "diagnostic-result.json"
    if not checkpoint_path.exists() and not checkpoint_path.is_symlink():
        valid(not result_path.exists() and not result_path.is_symlink(),
            "terminal_without_checkpoint")
        output["status"] = ("launch_running" if runtime_state == "running_verified"
            else "terminal_failure_without_result" if cleaned
            else "launch_attempted_runtime_unverified")
        return output
    valid(started, "checkpoint_without_execution_marker")
    counts, completed, unhealthy, degraded, check = checkpoint(
        read(checkpoint_path, root, 2_000_000), pinned)
    output["question_failure_codes"] = [
        check["entries"][qid]["failure"]
        if qid in check["entries"] and check["entries"][qid]["status"] == "failed"
        else None for qid in check["expected_ids"]]
    output["scored_count"] = completed
    output["summary_degraded_sessions_total"] = degraded
    if counts is not None and completed == 4:
        output["strict_indexing_healthy_for_all"] = unhealthy == 0
    if counts is None:
        valid(not result_path.exists() and not result_path.is_symlink(),
            "terminal_before_checkpoint_complete")
        output["status"] = ("checkpoint_running" if runtime_state == "running_verified"
            else "terminal_failure_without_result" if cleaned
            else "checkpoint_running_runtime_unverified")
        return output
    output["status"] = ("checkpoint_complete_result_pending" if runtime_state == "running_verified"
        else "terminal_failure_without_result" if cleaned
        else "checkpoint_complete_runtime_unverified")
    if not result_path.exists() and not result_path.is_symlink():
        return output
    result = terminal(read(result_path, root, 128_000), check, pinned, counts,
        completed, unhealthy)
    budget = result["budget"]
    output.update(scored_count=result["scored_count"], correct_count=result["correct_count"],
        canary_model_gold_match=(result["canary"].get("model_gold_match") if result["canary"] else None),
        canary_structural_valid=(result["canary"].get("structural_valid") if result["canary"] else None),
        campaign_stop=result["campaign_stop"], owner_failure=result["owner_failure"],
        budget_stop_code=budget["stop_code"], first_failure=budget.get("first_failure"),
        failed_count=counts["failed"], known_turns=budget["turns"],
        known_tokens=budget["known_tokens"], usage_complete=budget["usage_complete"],
        resource_observation=budget.get("resource_observation"),
        resource_fault=budget.get("resource_fault"),
        siwc_observations=result["siwc_observations"],
        siwc_pilot_projection=result["siwc_pilot_projection"])
    output["completed_diagnostic_and_clean"] = bool(result["diagnostic_complete"]
        and runtime_state == "clean_exit" and started)
    output["status"] = "completed_diagnostic_and_clean" if output["completed_diagnostic_and_clean"] else "terminal_incomplete_or_unclean"
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = inspect(Path(args.root), args.receipt_sha256)
        encoded = json.dumps(result, sort_keys=True, allow_nan=False,
            separators=(",", ":"))
        valid(len(encoded.encode("utf-8")) <= 32_768, "metadata_projection_too_large")
        print(encoded)
        return 0
    except BaseException:
        print(json.dumps({"schema": SCHEMA, "status": "unverified",
            "completed_diagnostic_and_clean": False, "runtime_cleanup_verified": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
