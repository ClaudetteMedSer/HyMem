#!/usr/bin/env python3
"""
HyMem Multi-Session Chat (MSC) Benchmark Adapter
================================================
Runs the MSC benchmark ("Beyond Goldfish Memory", Xu et al., ACL 2022) against
HyMem's Python SDK — the cross-session complement to `longmemeval_adapter.py`.

WHY MSC (and not just LME). LongMemEval is single-shot / star-topology: within a
question's haystack a fact appears once and never recurs across sessions. MSC is
genuinely multi-session — the SAME two speakers reconvene over up to 5 sessions
(hours-to-days apart) and their personas accumulate and get restated. That is
exactly the structure LME can't provide, and it's what the Idea-B repetition
signal, the `suggest_rules()` `session_count`, and Track-A multi-hop all need.

DATA (verified 2026-07-28 against the MemGPT/MSC-Self-Instruct HF dataset — the
concrete, downloadable QA derivative with clean labels):

    example = {
      "previous_dialogs": [ {"dialog": [{"text": str}, ...],   # the multi-session
                             "personas": [[str], [str]],       #   history to ingest
                             "time_num": int, "time_unit": str}, ... ],
      "self_instruct": {"B": <question>, "A": <gold answer>},  # the recall probe
      "personas": [[str], [str]], "init_personas": ..., "personas_update1/2": ...,
      "metadata": {"initial_data_id": str, "session_id": int},
    }

Two probe modes:

  --probe-mode recall   (the headline, LME-comparable number)
      Ingest `previous_dialogs` as sessions, dream, then ask `self_instruct.B`
      and judge against `self_instruct.A`. Measures cross-session fact recall,
      with an E1 accuracy-by-session-distance breakdown LME structurally can't
      produce. Reuses the LME answer/judge machinery verbatim (frozen posture).

  --probe-mode recurrence   (produces the E3 input for the existing engine)
      Ingest + dream, then DUMP the extracted behavioral markers with their
      HyMem session_id and an is_rule label derived from MSC's own persona
      annotations (a marker is "durable" iff it matches an annotated persona
      fact — MSC's ground truth, NOT the session_count we're validating, so no
      circularity). Feed the dump to `rule_extraction_experiment.py --labels ...
      --policy-from-canonical`: the honest retry of the corpus-artifact result,
      now on REAL cross-session recurrence. Note: MSC content is preference/fact
      shaped, so many markers are `preference`-kind (profile-tier, not rules) —
      the dump makes that empirically visible rather than assuming it.

Usage:
  python msc_adapter.py --data msc.json --probe-mode recall --sample 50
  python msc_adapter.py --data msc.json --probe-mode recurrence --out markers_msc.json
  python msc_adapter.py --sim            # offline: loader + labeling mechanics, no API
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import math
import os
import random
import re
import sys
import tempfile
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling benchmark imports

from benchmarks.extraction_canary import (
    ExtractionCanaryError,
    extraction_canary_client_policy,
    extraction_canary_policy,
    print_extraction_canary,
    run_configured_extraction_canary,
    skipped_extraction_canary,
    validate_extraction_canary_config_binding,
    validate_extraction_canary_report,
)
from benchmarks.strictness import (
    AtomicCheckpoint,
    BENCHMARK_INDEXING_STATUS_VERSION,
    BenchmarkCleanupError,
    BenchmarkIntegrityError,
    IndexingConvergenceError,
    OwnedResourceScope,
    PythonSourceSlice,
    add_strict_run_arguments,
    aggregate_embedding_usage_snapshots,
    aggregate_usage_snapshots,
    benchmark_hymem_source_paths,
    bounded_exception_type,
    bounded_failure_text,
    build_manifest,
    code_hash,
    content_hash,
    converge_indexing,
    durable_indexing_status,
    embedding_usage_snapshot,
    effective_hymem_config_identity,
    file_hash,
    freeze_calibration,
    load_calibration,
    prepare_checkpoint_artifact,
    publish_prepared_artifact_after_cleanup,
    python_file_imported_symbols,
    python_slice_imported_symbols,
    resolve_checkpoint_path,
    run_cleanup_actions,
    sanitize_for_artifact,
    select_protocol_ids,
    strict_accuracy,
    usage_snapshot,
    validate_ids,
    write_immutable_artifact,
    write_latest_pointer,
)
from benchmarks.store_attestation import (
    MATERIAL_STORE_ATTESTATION_VERSION,
    MaterialStoreAttestationError,
    material_state_mismatch_tables,
    material_store_state as compute_material_store_state,
)
from hymem.contrib.endpoint_policy import validate_http_endpoint
from hymem.contrib.model_policy import (
    DeprecatedModelAliasError,
    require_active_model,
)
from hymem.core.vectors import decode_vector
from hymem.dreaming.lossless import COVERAGE_INTEGRITY_CONFIG_VERSION
from hymem.dreaming.status import (
    DREAM_STATUS_AGGREGATION_AUTHORITY_FIELDS,
    DREAM_STATUS_AGGREGATION_MATERIAL_AUTHORITY_FIELDS,
    DREAM_STATUS_SCHEMA_VERSION,
    DREAM_STATUS_PHASE1_AUTHORITY_FIELDS,
    DURABLE_MALFORMED_FIELDS,
    DURABLE_PENDING_FIELDS,
)
from hymem.extraction.producer import phase1_generation_binding

# Reuse the LME machinery unchanged — same answer/judge clients and scoring keep
# MSC and LME numbers in ONE comparability frame (frozen posture). Imported
# lazily inside functions to keep --sim import-light and API-free.
_ANSWER_MODEL = "deepseek-v4-flash"
_JUDGE_MODEL = "deepseek-v4-flash"
_HYMEM_MODEL = "deepseek-v4-flash"   # NOT the deprecated deepseek-chat (2026-07-24)
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_MSC_APERTURE = {
    "message_fts_top_k": 15,
    "fts_top_k": 10,
    "graph_top_k": 10,
}
DEFAULT_INDEXING_MAX_CYCLES = 100
DEFAULT_INDEXING_TIMEOUT_S = 3600.0
# v4 additionally binds the exact producer-authority status used to interpret
# the Phase-1 pending count. A producer-unavailable zero can never certify a
# completed store. Its fixed report totals, terminal cycle gates, and flag
# counts continue to reconcile to the root totals.
INDEXING_PROVENANCE_VERSION = "hymem-benchmark-indexing-v4"
# v7 carries v4 convergence evidence in addition to the exact Phase-1
# producer/effective request.  A
# screening-model receipt therefore cannot certify a target-model store.
STORE_BUILD_RECEIPT_VERSION = "hymem-benchmark-store-build-v7"
STORE_INDEXING_ATTESTATION_VERSION = (
    "hymem-benchmark-store-indexing-attestation-v2"
)
STORE_BUILD_RECEIPT_NAME = ".hymem-benchmark-store-build.json"
EMBEDDING_STORE_ATTESTATION_VERSION = (
    "hymem-benchmark-embedding-store-attestation-v1"
)
MATERIAL_HYMEM_CODE_IDENTITY_VERSION = "hymem-material-code-closure-v3"
INGESTION_MAPPING_VERSION = "msc-ingestion-mapping-v2"
_INGESTION_MAPPING_SURFACE = (
    "MSCAdapter.open",
    "MSCAdapter.ingest",
    "MSCAdapter.dream",
    "MSCAdapter._durable_status",
)

# Only material/write-side settings belong here. Retrieval aperture, reader,
# judge, and answer-prompt levers are intentionally absent so one proven store
# can serve honest read-side A/B runs.
_STORE_WRITE_CONFIG_FIELDS = (
    "salience_min_chars",
    "redact_secrets",
    "max_message_chars",
    "aggregation_nodes_enabled",
    "aggregation_emb_threshold",
    "aggregation_ent_threshold",
    "aggregation_max_cluster_size",
    "aggregation_blocking_top_k",
    "aggregation_min_sessions",
    "aggregation_min_members",
    "aggregation_max_members",
    "aggregation_digest_enabled",
    "aggregation_digest_max_leaves",
    "aggregation_digest_anchor_facts",
    "profile_extraction_enabled",
    "profile_max_items_per_session",
    "profile_extraction_max_attempts",
    "rules_extraction_enabled",
    "rules_extraction_mode",
    "rules_extraction_confidence_min",
    "rules_extraction_batch_size",
    "facts_extraction_enabled",
    "dream_max_facts_per_session",
    "facts_extraction_max_attempts",
    "episode_granularity_enabled",
    "dream_max_episodes_per_session",
    "coref_enabled",
    "coref_max_turns",
    "coref_llm_enabled",
    "evidence_role_weights",
    "triple_dedup_enabled",
    "triple_dedup_cosine_threshold",
    "triple_dedup_lexical_ratio",
    "decay_window_days",
    "decay_factor",
    "retract_threshold",
    "predicate_half_life_days",
    "zombie_neg_threshold",
    "value_supersession_enabled",
    "reinforce_window_days",
    "profile_max_entries",
    "insights_max_entries",
    "prompt_version",
    "dream_budget",
    "dream_extraction_provider_attempt_budget",
    "dream_baseline_budget",
    "chunk_extraction_max_attempts",
    "dream_digest_max_tokens",
    "dream_digest_max_chars",
    "digest_extraction_max_attempts",
    "max_chunks",
    "retention_days",
    "episode_retention_days",
    "message_retention_days",
    "tombstone_retention_days",
    "vacuum_after_prune",
    "vacuum_min_pruned",
)

_EMBEDDING_PENDING_FIELDS = (
    "pending_chunk_embeddings",
    "pending_message_embeddings",
    "pending_edge_embeddings",
    "pending_episode_embeddings",
    "pending_fact_embeddings",
)
_DURABLE_QUARANTINE_FIELDS = (
    "quarantined_chunks",
    "quarantined_digests",
    "quarantined_profiles",
    "quarantined_facts",
    "quarantined_facts_malformed",
)
_CURRENT_DREAM_REPORT_FAILURE_FIELDS = (
    "chunk_extraction_failures",
    "coverage_integrity_failures",
    "digest_failures",
    "digest_quarantined",
    "profile_failures",
    "fact_failures",
    "aggregation_fusion_failures",
    "aggregation_build_exceptions",
)
_CURRENT_DREAM_REPORT_BOOLEAN_FIELDS = (
    "budget_exhausted",
    "extraction_provider_attempt_budget_exhausted",
    "skipped_locked",
)
# MSC/LoCoMo pin aggregation off, so nullable aggregation-debug counters are
# deliberately absent from the fixed receipt totals.  Every ordinary numeric
# DreamReport field remains visible and exact.
_DREAM_REPORT_TOTAL_FIELDS = (
    "sessions_processed",
    "chunks_seen",
    "chunks_processed",
    "chunk_extraction_failures",
    "chunk_extraction_completion_calls",
    "chunk_extraction_provider_attempts",
    "coverage_integrity_failures",
    "triples_extracted",
    "markers_extracted",
    "rules_extracted",
    "chunks_embedded",
    "chunks_embedded_from_cache",
    "messages_embedded",
    "messages_embedded_from_cache",
    "edges_embedded",
    "edges_embedded_from_cache",
    "episodes_embedded",
    "episodes_embedded_from_cache",
    "aggregation_nodes_built",
    "aggregation_nodes_reused",
    "aggregation_fusion_failures",
    "aggregation_build_exceptions",
    "aggregation_input_episodes",
    "digest_failures",
    "digest_quarantined",
    "episodes_created",
    "facts_extracted",
    "fact_failures",
    "facts_embedded",
    "facts_embedded_from_cache",
    "profile_items_extracted",
    "profile_failures",
)
_DREAM_REPORT_NULLABLE_COUNT_FIELDS = (
    "aggregation_level0_missed",
    "aggregation_leaf_changed",
    "aggregation_predicted_rebuild",
    "aggregation_keying_residual",
    "aggregation_rebuilt_level0",
    "aggregation_rebuilt_rollup",
    "aggregation_rebuilt_root",
    "aggregation_leaf_added",
    "aggregation_leaf_removed",
    "aggregation_facts_rekey",
)
_DREAM_REPORT_FIELDS = (
    *_DREAM_REPORT_TOTAL_FIELDS,
    *_DREAM_REPORT_NULLABLE_COUNT_FIELDS,
    "aggregation_blocking",
    *_CURRENT_DREAM_REPORT_BOOLEAN_FIELDS,
)
_INDEXING_USAGE_FIELDS = (
    "calls",
    "calls_available",
    "request_attempts",
    "request_attempts_available",
    "successful_responses",
    "successful_responses_available",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_s",
    "cost_usd",
    "token_usage_available",
    "latency_available",
    "cost_available",
)
_INDEXING_SETTINGS_FIELDS = (
    "max_cycles_per_convergence",
    "timeout_s_per_convergence",
    "require_healthy",
)
_INDEXING_RUN_FIELDS = (
    "trigger",
    "cycles",
    "report_count",
    "complete",
    "healthy",
    "failure_reason",
    "elapsed_s",
    "dream_report_totals",
    "budget_exhausted_cycles",
    "extraction_provider_attempt_budget_exhausted_cycles",
    "skipped_locked_cycles",
    "final_cycle",
    "final_status",
)
_INDEXING_PROVENANCE_FIELDS = (
    "protocol",
    "scope_id",
    "mode",
    "comparable",
    "complete",
    "healthy",
    "convergence_count",
    "cycles",
    "settings",
    "runs",
    "dream_report_totals",
    "budget_exhausted_cycles",
    "extraction_provider_attempt_budget_exhausted_cycles",
    "skipped_locked_cycles",
    "final_status",
    "pipeline_usage",
)
_INDEXING_ATTESTATION_FIELDS = (
    "schema",
    *_INDEXING_PROVENANCE_FIELDS,
)
_FINAL_STATUS_HEALTH_FIELDS = (
    "dream_status_schema",
    "benchmark_indexing_status_schema",
    *DURABLE_PENDING_FIELDS,
    *DREAM_STATUS_PHASE1_AUTHORITY_FIELDS,
    *DREAM_STATUS_AGGREGATION_AUTHORITY_FIELDS,
    *DREAM_STATUS_AGGREGATION_MATERIAL_AUTHORITY_FIELDS,
    *_EMBEDDING_PENDING_FIELDS,
    *DURABLE_MALFORMED_FIELDS,
    *_DURABLE_QUARANTINE_FIELDS,
    "terminal_loss_chunks",
    "terminal_loss_reasons",
    "coverage_integrity_failures",
    "coverage_integrity_failure_reasons",
    "coverage_integrity_failure_details",
    "coverage_integrity_failure_details_truncated",
    "coverage_integrity_config_version",
    "aggregation_enabled",
    "aggregation_publication_generation",
    "aggregation_material_binding",
    "aggregation_material_revision",
    "in_progress",
)
_STORE_BUILD_RECEIPT_FIELDS = (
    "version",
    "status",
    "identity_sha256",
    "identity",
    "embedding_state",
    "material_state",
    "indexing_sha256",
    "indexing",
)
_MAX_INDEXING_COUNT = (1 << 63) - 1
_MAX_INDEXING_CONVERGENCES = 1_000_000
_MAX_INDEXING_TIMEOUT_S = 31_536_000.0

_EMBEDDING_MIRROR_TABLES = (
    "aggregation_node_embeddings",
    "chunk_embeddings",
    "edge_embeddings",
    "embedding_cache",
    "episode_embeddings",
    "message_embeddings",
    "narrative_fact_embeddings",
)
_EMBEDDING_VEC_TABLES = frozenset({
    "vec_chunks", "vec_edges", "vec_episodes", "vec_facts", "vec_messages",
})


def _known_zero_pipeline_usage() -> dict:
    """Exact usage for paths that intentionally made no pipeline calls."""

    return {
        "calls": 0,
        "calls_available": True,
        "request_attempts": 0,
        "request_attempts_available": True,
        "successful_responses": 0,
        "successful_responses_available": True,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_s": 0.0,
        "cost_usd": 0.0,
        "token_usage_available": True,
        "latency_available": True,
        "cost_available": True,
    }


def _exact_object(value: object, fields: tuple[str, ...], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise BenchmarkIntegrityError(f"{label} has a malformed exact schema")
    return value


def _bounded_count(value: object, label: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < int(positive)
        or value > _MAX_INDEXING_COUNT
    ):
        raise BenchmarkIntegrityError(f"{label} is malformed")
    return value


def _bounded_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise BenchmarkIntegrityError(f"{label} is malformed")
    return float(value)


def _fixed_report_evidence(
    reports: object,
) -> tuple[dict[str, int], int, int, int, dict[str, int | bool]]:
    """Project full DreamReports onto the exact receipt-relevant contract."""

    if (
        not isinstance(reports, list)
        or not reports
        or len(reports) > _MAX_INDEXING_COUNT
        or any(
            not isinstance(report, dict)
            or set(report) != set(_DREAM_REPORT_FIELDS)
            for report in reports
        )
    ):
        raise BenchmarkIntegrityError("indexing reports are malformed")
    totals = {field: 0 for field in _DREAM_REPORT_TOTAL_FIELDS}
    for report in reports:
        for field in _DREAM_REPORT_TOTAL_FIELDS:
            value = _bounded_count(
                report.get(field, 0), f"indexing report {field}"
            )
            total = totals[field] + value
            totals[field] = _bounded_count(
                total, f"indexing report total {field}"
            )
        for field in _CURRENT_DREAM_REPORT_BOOLEAN_FIELDS:
            if not isinstance(report.get(field), bool):
                raise BenchmarkIntegrityError(
                    f"indexing report {field} is malformed"
                )
        for field in _DREAM_REPORT_NULLABLE_COUNT_FIELDS:
            if report[field] is not None:
                _bounded_count(report[field], f"indexing report {field}")
        if (
            not isinstance(report["aggregation_blocking"], str)
            or len(report["aggregation_blocking"]) > 64
            or re.fullmatch(r"[a-z0-9_:.-]*", report["aggregation_blocking"])
            is None
        ):
            raise BenchmarkIntegrityError(
                "indexing report aggregation blocking is malformed"
            )
    final_report = reports[-1]
    final_cycle: dict[str, int | bool] = {
        field: _bounded_count(
            final_report.get(field), f"indexing final cycle {field}"
        )
        for field in _CURRENT_DREAM_REPORT_FAILURE_FIELDS
    }
    final_cycle.update({
        field: final_report[field]
        for field in _CURRENT_DREAM_REPORT_BOOLEAN_FIELDS
    })
    return (
        totals,
        sum(report["budget_exhausted"] is True for report in reports),
        sum(
            report["extraction_provider_attempt_budget_exhausted"] is True
            for report in reports
        ),
        sum(report["skipped_locked"] is True for report in reports),
        final_cycle,
    )


def _validate_indexing_usage(value: object) -> None:
    usage = _exact_object(value, _INDEXING_USAGE_FIELDS, "indexing usage")
    count_pairs = (
        ("calls", "calls_available"),
        ("request_attempts", "request_attempts_available"),
        ("successful_responses", "successful_responses_available"),
    )
    for field, available_field in count_pairs:
        available = usage[available_field]
        if not isinstance(available, bool):
            raise BenchmarkIntegrityError("indexing usage availability is malformed")
        if available:
            _bounded_count(usage[field], f"indexing usage {field}")
        elif usage[field] is not None:
            raise BenchmarkIntegrityError(f"indexing usage {field} is unavailable")
    if (
        usage["calls_available"] is True
        and usage["successful_responses_available"] is True
        and usage["successful_responses"] > usage["calls"]
    ):
        raise BenchmarkIntegrityError("indexing usage successful call arithmetic drifted")
    if (
        usage["request_attempts_available"] is True
        and usage["successful_responses_available"] is True
        and usage["request_attempts"] < usage["successful_responses"]
    ):
        raise BenchmarkIntegrityError("indexing usage attempt arithmetic drifted")
    if (
        usage["request_attempts_available"] is True
        and usage["calls_available"] is True
        and usage["request_attempts"] < usage["calls"]
    ):
        raise BenchmarkIntegrityError("indexing usage call arithmetic drifted")

    if not isinstance(usage["token_usage_available"], bool):
        raise BenchmarkIntegrityError("indexing token availability is malformed")
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if usage["token_usage_available"]:
        tokens = [
            _bounded_count(usage[field], f"indexing usage {field}")
            for field in token_fields
        ]
        if tokens[0] + tokens[1] != tokens[2]:
            raise BenchmarkIntegrityError("indexing token arithmetic drifted")
    elif any(usage[field] is not None for field in token_fields):
        raise BenchmarkIntegrityError("unavailable indexing tokens are populated")

    for field, available_field in (
        ("latency_s", "latency_available"),
        ("cost_usd", "cost_available"),
    ):
        available = usage[available_field]
        if not isinstance(available, bool):
            raise BenchmarkIntegrityError("indexing usage availability is malformed")
        if available:
            _bounded_number(usage[field], f"indexing usage {field}")
        elif usage[field] is not None:
            raise BenchmarkIntegrityError(f"indexing usage {field} is unavailable")


def _scope_matches_item(scope_id: object, item: dict | None) -> bool:
    if (
        not isinstance(scope_id, str)
        or len(scope_id) > 512
        or re.fullmatch(r"(?:msc|locomo):[^\x00-\x20]{1,500}", scope_id) is None
    ):
        return False
    if item is None:
        return True
    item_id = item.get("id") if isinstance(item, dict) else None
    return bool(
        isinstance(item_id, str)
        and scope_id in {f"msc:{item_id}", f"locomo:{item_id}"}
    )


def _final_health_projection(value: object, *, exact: bool) -> dict:
    if not isinstance(value, dict):
        raise BenchmarkIntegrityError("indexing final status is malformed")
    required = set(_FINAL_STATUS_HEALTH_FIELDS)
    if exact and set(value) != required:
        raise BenchmarkIntegrityError(
            "indexing final status attestation has a malformed exact schema"
        )
    if not required.issubset(value):
        raise BenchmarkIntegrityError("indexing final status is incomplete")
    recognized_prefixed = required
    unknown_health = {
        key for key in value
        if isinstance(key, str)
        and (
            key.startswith((
                "pending_", "malformed_", "terminal_loss_",
                "coverage_integrity_",
            ))
            or "quarantined" in key
        )
        and key not in recognized_prefixed
    }
    if unknown_health:
        raise BenchmarkIntegrityError("indexing final status schema is ambiguous")
    if value["dream_status_schema"] != DREAM_STATUS_SCHEMA_VERSION:
        raise BenchmarkIntegrityError("indexing durable status schema is incompatible")
    if (
        value["benchmark_indexing_status_schema"]
        != BENCHMARK_INDEXING_STATUS_VERSION
    ):
        raise BenchmarkIntegrityError("indexing benchmark status schema is incompatible")
    count_fields = (
        *DURABLE_PENDING_FIELDS,
        *_EMBEDDING_PENDING_FIELDS,
        *DURABLE_MALFORMED_FIELDS,
        *_DURABLE_QUARANTINE_FIELDS,
        "terminal_loss_chunks",
        "coverage_integrity_failures",
    )
    for field in count_fields:
        if _bounded_count(value[field], f"indexing final status {field}") != 0:
            raise BenchmarkIntegrityError("indexing final status is not clean")
    if (
        value["terminal_loss_reasons"] != {}
        or value["coverage_integrity_failure_reasons"] != {}
        or value["coverage_integrity_failure_details"] != []
        or value["coverage_integrity_failure_details_truncated"] is not False
        or value["coverage_integrity_config_version"]
        != COVERAGE_INTEGRITY_CONFIG_VERSION
    ):
        raise BenchmarkIntegrityError(
            "indexing final coverage or terminal-loss evidence is malformed"
        )
    if value["in_progress"] is not False:
        raise BenchmarkIntegrityError("indexing final status is still in progress")
    if (
        value["phase1_backlog_status"] != "current_producer"
        or value["pending_chunks_authoritative"] is not True
        or not isinstance(value["phase1_generation_key"], str)
        or not value["phase1_generation_key"]
    ):
        raise BenchmarkIntegrityError(
            "indexing final status lacks exact Phase-1 producer authority"
        )
    # Share the canonical v57 generation/material certificate validator with
    # LME.  MSC/LoCoMo may project fewer diagnostics, but they may not certify
    # a clean aggregation counter without retaining the exact producer,
    # material epoch, revision, and stored binding that were checked.
    try:
        from benchmarks.lme_protocol import _canonical_final_indexing_status

        _canonical_final_indexing_status(value)
    except BenchmarkIntegrityError:
        raise
    except Exception as exc:
        raise BenchmarkIntegrityError(
            "indexing final aggregation authority is malformed"
        ) from exc
    return {field: copy.deepcopy(value[field]) for field in _FINAL_STATUS_HEALTH_FIELDS}


def _validate_indexing_provenance(
    value: object,
    *,
    item: dict | None = None,
    attestation: bool = False,
) -> dict:
    fields = (
        _INDEXING_ATTESTATION_FIELDS
        if attestation else _INDEXING_PROVENANCE_FIELDS
    )
    indexing = _exact_object(value, fields, "indexing provenance")
    if attestation and indexing["schema"] != STORE_INDEXING_ATTESTATION_VERSION:
        raise BenchmarkIntegrityError("indexing attestation schema is incompatible")
    if indexing["protocol"] != INDEXING_PROVENANCE_VERSION:
        raise BenchmarkIntegrityError("indexing provenance protocol is incompatible")
    if not _scope_matches_item(indexing["scope_id"], item):
        raise BenchmarkIntegrityError("indexing scope does not match its source item")
    if (
        indexing["mode"] != "converged"
        or indexing["comparable"] is not True
        or indexing["complete"] is not True
        or indexing["healthy"] is not True
    ):
        raise BenchmarkIntegrityError("indexing provenance is not complete and healthy")

    convergence_count = _bounded_count(
        indexing["convergence_count"],
        "indexing convergence count",
        positive=True,
    )
    if convergence_count > _MAX_INDEXING_CONVERGENCES:
        raise BenchmarkIntegrityError("indexing convergence count is too large")
    cycles = _bounded_count(indexing["cycles"], "indexing cycle count", positive=True)
    settings = _exact_object(
        indexing["settings"], _INDEXING_SETTINGS_FIELDS, "indexing settings"
    )
    max_cycles = _bounded_count(
        settings["max_cycles_per_convergence"],
        "indexing max cycles",
        positive=True,
    )
    if max_cycles > _MAX_INDEXING_CONVERGENCES:
        raise BenchmarkIntegrityError("indexing max cycles is too large")
    timeout_s = _bounded_number(
        settings["timeout_s_per_convergence"], "indexing timeout"
    )
    if timeout_s <= 0 or timeout_s > _MAX_INDEXING_TIMEOUT_S:
        raise BenchmarkIntegrityError("indexing timeout is out of bounds")
    if settings["require_healthy"] is not True:
        raise BenchmarkIntegrityError("indexing did not require healthy convergence")

    runs = indexing["runs"]
    if not isinstance(runs, list) or len(runs) != convergence_count:
        raise BenchmarkIntegrityError("indexing run count is inconsistent")
    summed_cycles = 0
    summed_totals = {field: 0 for field in _DREAM_REPORT_TOTAL_FIELDS}
    summed_flags = {
        "budget_exhausted_cycles": 0,
        "extraction_provider_attempt_budget_exhausted_cycles": 0,
        "skipped_locked_cycles": 0,
    }
    latest_run_health: dict | None = None
    for run in runs:
        run = _exact_object(run, _INDEXING_RUN_FIELDS, "indexing run")
        trigger = run["trigger"]
        if (
            not isinstance(trigger, str)
            or (
                trigger not in {"end_of_history", "reused_store_validation"}
                and re.fullmatch(r"after_session:(?:0|[1-9][0-9]{0,8})", trigger)
                is None
            )
        ):
            raise BenchmarkIntegrityError("indexing run trigger is malformed")
        run_cycles = _bounded_count(
            run["cycles"], "indexing run cycle count", positive=True
        )
        if run_cycles > max_cycles:
            raise BenchmarkIntegrityError("indexing run exceeded its cycle bound")
        if _bounded_count(
            run["report_count"], "indexing run report count", positive=True
        ) != run_cycles:
            raise BenchmarkIntegrityError("indexing run report count is inconsistent")
        summed_cycles = _bounded_count(
            summed_cycles + run_cycles, "indexing summed cycle count"
        )
        if (
            run["complete"] is not True
            or run["healthy"] is not True
            or run["failure_reason"] is not None
        ):
            raise BenchmarkIntegrityError("indexing run is not complete and healthy")
        if _bounded_number(run["elapsed_s"], "indexing run elapsed time") > timeout_s:
            raise BenchmarkIntegrityError("indexing run exceeded its timeout")
        run_totals = _exact_object(
            run["dream_report_totals"],
            _DREAM_REPORT_TOTAL_FIELDS,
            "indexing run report totals",
        )
        for field in _DREAM_REPORT_TOTAL_FIELDS:
            run_total = _bounded_count(
                run_totals[field], f"indexing run report total {field}"
            )
            summed_totals[field] = _bounded_count(
                summed_totals[field] + run_total,
                f"indexing summed report total {field}",
            )
        for field in summed_flags:
            flag_count = _bounded_count(
                run[field], f"indexing run {field}"
            )
            if flag_count > run_cycles:
                raise BenchmarkIntegrityError("indexing run flag count is inconsistent")
            summed_flags[field] = _bounded_count(
                summed_flags[field] + flag_count,
                f"indexing summed {field}",
            )
        final_cycle = _exact_object(
            run["final_cycle"],
            (*_CURRENT_DREAM_REPORT_FAILURE_FIELDS,
             *_CURRENT_DREAM_REPORT_BOOLEAN_FIELDS),
            "indexing final cycle",
        )
        for field in _CURRENT_DREAM_REPORT_FAILURE_FIELDS:
            if _bounded_count(
                final_cycle[field], f"indexing final cycle {field}"
            ) != 0:
                raise BenchmarkIntegrityError("indexing final cycle has failures")
        for field in _CURRENT_DREAM_REPORT_BOOLEAN_FIELDS:
            if final_cycle[field] is not False:
                raise BenchmarkIntegrityError("indexing final cycle is incomplete")
        latest_run_health = _final_health_projection(
            run["final_status"], exact=True
        )

    if summed_cycles != cycles:
        raise BenchmarkIntegrityError("indexing cycle arithmetic drifted")
    root_totals = _exact_object(
        indexing["dream_report_totals"],
        _DREAM_REPORT_TOTAL_FIELDS,
        "indexing report totals",
    )
    for field in _DREAM_REPORT_TOTAL_FIELDS:
        if _bounded_count(
            root_totals[field], f"indexing report total {field}"
        ) != summed_totals[field]:
            raise BenchmarkIntegrityError("indexing report total arithmetic drifted")
    for field, expected in summed_flags.items():
        if _bounded_count(indexing[field], f"indexing {field}") != expected:
            raise BenchmarkIntegrityError("indexing cycle flag arithmetic drifted")

    health = _final_health_projection(
        indexing["final_status"], exact=attestation
    )
    if latest_run_health != health:
        raise BenchmarkIntegrityError("indexing final status disagrees with its run")
    _validate_indexing_usage(indexing["pipeline_usage"])
    return health


def _canonical_indexing_attestation(indexing: object, *, item: dict) -> dict:
    """Return the exact secret-safe indexing proof stored in receipt v5."""

    health = _validate_indexing_provenance(indexing, item=item)
    assert isinstance(indexing, dict)
    projected = {
        "schema": STORE_INDEXING_ATTESTATION_VERSION,
        **{
            field: copy.deepcopy(indexing[field])
            for field in _INDEXING_PROVENANCE_FIELDS
            if field != "final_status"
        },
        "final_status": health,
    }
    sanitized = sanitize_for_artifact(projected)
    if (
        not isinstance(sanitized, dict)
        or sanitize_for_artifact(sanitized) != sanitized
    ):
        raise BenchmarkIntegrityError("indexing attestation sanitization is unstable")
    _validate_indexing_provenance(sanitized, item=item, attestation=True)
    return sanitized


def _identity_mismatch_fields(
    expected: object,
    actual: object,
    *,
    prefix: str = "identity",
    limit: int = 50,
) -> tuple[list[str], bool]:
    """Return bounded structural paths only, never source/config values."""

    mismatches: list[str] = []
    truncated = False

    def safe_component(value: object) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "?", str(value)[:96])

    def visit(want: object, got: object, path: str) -> None:
        nonlocal truncated
        if len(mismatches) >= limit:
            truncated = True
            return
        if isinstance(want, dict) and isinstance(got, dict):
            for key in sorted(set(want) | set(got), key=str):
                if len(mismatches) >= limit:
                    truncated = True
                    return
                child = f"{path}.{safe_component(key)}"
                if key not in want or key not in got:
                    mismatches.append(child)
                else:
                    visit(want[key], got[key], child)
            return
        if want != got:
            mismatches.append(path)

    visit(expected, actual, prefix)
    return mismatches, truncated


def _embedding_attribute(client: object, name: str) -> object:
    """Read one benchmark capability with a deliberate bounded failure."""

    try:
        return getattr(client, name)
    except Exception as exc:
        raise BenchmarkIntegrityError(
            f"benchmark embedding client does not expose a usable {name}"
        ) from exc


def _benchmark_embedding_identity(client: object | None) -> dict:
    """Return an immutable, fail-closed benchmark vector-space identity.

    ``EmbeddingClient`` intentionally has only ``model``/``dim`` and permits
    adaptive implementations.  Reusable benchmark stores need a stronger
    capability: an immutable configured dimension, a pinned provider policy,
    and a latched integrity result.  Older/fake clients are therefore rejected
    explicitly instead of being accepted based on a mutable ``dim`` snapshot.
    """

    from hymem.contrib.endpoint_policy import (
        TRANSPORT_SECURITY_NONE,
        validate_recorded_endpoint_origin,
    )
    from hymem.dreaming.aggregation_material import (
        embedding_execution_identity,
        public_embedding_identity,
    )

    try:
        binding, _model, current_dimension = embedding_execution_identity(client)
    except Exception as exc:
        raise BenchmarkIntegrityError(
            "benchmark embedding client identity is invalid"
        ) from exc
    declaration = binding.get("declaration")
    kind = declaration.get("kind") if isinstance(declaration, dict) else None
    if kind == "disabled":
        return public_embedding_identity(
            binding, None, fallback_policy="none", fallback_reason=None,
            transport_security=TRANSPORT_SECURITY_NONE,
        )
    if binding.get("identity_exact") is not True:
        raise BenchmarkIntegrityError(
            "reusable benchmark stores require a durable exact embedding producer"
        )
    if kind != "openai_compatible":
        raise BenchmarkIntegrityError(
            "MSC/LoCoMo semantic runs require an exact OpenAI-compatible producer"
        )
    configured_dimension = declaration.get("configured_dimension")
    integrity = _embedding_attribute(client, "dimension_integrity_ok")
    if (
        current_dimension != configured_dimension
        or declaration.get("dimension_policy") != "pinned"
        or integrity is not True
    ):
        raise BenchmarkIntegrityError(
            "benchmark embedding client differs from its pinned vector space"
        )
    try:
        _endpoint, transport_security = validate_recorded_endpoint_origin(
            declaration.get("endpoint_origin"), label="benchmark embedding",
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkIntegrityError(
            "benchmark embedding endpoint identity is invalid"
        ) from exc
    return public_embedding_identity(
        binding, current_dimension,
        fallback_policy="fail-closed", fallback_reason=None,
        transport_security=transport_security,
    )


def _validated_stored_vector(raw: object, *, expected_dimension: int) -> bool:
    """Validate one durable mirror vector without returning/logging its data."""

    if not isinstance(raw, (str, bytes, bytearray)):
        return False
    try:
        decoded = decode_vector(raw)
    except (AttributeError, TypeError, UnicodeError, ValueError):
        return False
    if not isinstance(decoded, list) or len(decoded) != expected_dimension:
        return False
    try:
        values = [
            float(value)
            for value in decoded
            if not isinstance(value, bool)
        ]
    except (TypeError, ValueError, OverflowError):
        return False
    if len(values) != expected_dimension or not all(
        math.isfinite(value) for value in values
    ):
        return False
    norm = math.sqrt(sum(value * value for value in values))
    return math.isfinite(norm) and norm > 0.0


@lru_cache(maxsize=16)
def _material_hymem_code_hash(
    *,
    adapter_path: Path | None = None,
    hymem_path: Path | None = None,
    root: Path | None = None,
) -> str:
    """Freeze the transitive HyMem implementation used to build a store.

    This deliberately differs from :func:`msc_code_hash`: the full benchmark
    identity includes readers, judges, reporting and checkpoint machinery,
    while a reusable material store must change only with its write/index
    implementation.  Starting from ``MSCAdapter`` discovers constructor and
    function-local imports without a hand-maintained inventory, then follows
    every local HyMem import and adds the schema/migrations loaded as runtime
    resources.  Generated SQLite/receipt files and unrelated server code never
    enter the closure.
    """

    root_path = Path(root or _repo_root).resolve()
    adapter = Path(adapter_path or __file__).resolve()
    package = Path(hymem_path or root_path / "hymem").resolve()
    paths = benchmark_hymem_source_paths(
        package,
        root=root_path,
        dependency_sources=(PythonSourceSlice(adapter, ("MSCAdapter",)),),
    )
    # The dedicated framing version keeps material-store identity semantically
    # separate from the broader benchmark-code protocol even though both share
    # the same portable path/content hashing implementation.
    return code_hash(
        paths,
        root=root_path,
        identity_version=MATERIAL_HYMEM_CODE_IDENTITY_VERSION,
    )


@lru_cache(maxsize=16)
def _ingestion_mapping_identity(*material_callables) -> dict:
    """Version/hash only the adapter's material source-to-log mapping.

    Hashing the whole benchmark adapter would couple a reusable store to reader,
    judge and reporting edits that cannot change its contents.  The live method
    body is normalized as an AST so comments and indentation are irrelevant,
    while a changed/overridden mapping necessarily changes the build identity.
    """

    try:
        if len(material_callables) != len(_INGESTION_MAPPING_SURFACE):
            raise ValueError("incomplete material callable surface")
        normalized: dict[str, str] = {}
        for label, material_callable in zip(
            _INGESTION_MAPPING_SURFACE, material_callables
        ):
            target = inspect.unwrap(material_callable)
            source = textwrap.dedent(inspect.getsource(target))
            parsed = ast.parse(source)
            nodes = [
                node for node in ast.walk(parsed)
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                )
            ]
            if not nodes:
                raise ValueError("no callable AST")
            normalized[label] = ast.dump(
                nodes[0], annotate_fields=True, include_attributes=False
            )
    except Exception as exc:
        raise BenchmarkIntegrityError(
            "MSC ingestion mapping source is unavailable for store identity"
        ) from exc
    return {
        "version": INGESTION_MAPPING_VERSION,
        "surface": sorted(normalized),
        "sha256": content_hash({
            "version": INGESTION_MAPPING_VERSION,
            "asts": normalized,
        }),
    }


# ── lexical helpers (persona-fact matching, gold-turn location) ─────────────
# MSC recurrence ground truth and the E1 session-distance breakdown both need to
# ask "does this short fact appear in this turn/session?" — a content-word
# Jaccard, no embeddings required (an --embeddings cosine is a future upgrade).

_STOP = frozenset("a an the of to in on at for and or but is are was were be been "
                  "i you he she it we they my your his her our their me him them "
                  "this that these those with as by from about into over".split())


def _content_tokens(s: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (s or "").lower())
    return {t for t in toks if len(t) > 2 and t not in _STOP}


def _lex_match(fact: str, text: str, tau: float = 0.5) -> bool:
    """True if `fact`'s content words are largely present in `text` (asymmetric
    Jaccard: |fact ∩ text| / |fact|). Asymmetric because a fact is 'stated' in a
    turn when the turn CONTAINS it, even if the turn says much more."""
    f = _content_tokens(fact)
    if not f:
        return False
    return len(f & _content_tokens(text)) / len(f) >= tau


# ── dataset loader ──────────────────────────────────────────────────────────

_TIME_UNIT_DAYS = {"hour": 1 / 24, "hours": 1 / 24, "day": 1.0, "days": 1.0,
                   "week": 7.0, "weeks": 7.0, "month": 30.0, "months": 30.0,
                   "year": 365.0, "years": 365.0}


def _gap_days(pd: dict, default: float) -> float:
    """Parse a previous_dialog's inter-session gap (time_num/time_unit) into days,
    falling back to `default` when the fields are missing/unparseable."""
    try:
        n = float(pd.get("time_num"))
        unit = str(pd.get("time_unit", "")).lower().strip()
        return max(n * _TIME_UNIT_DAYS.get(unit, 1.0), 0.01)
    except (TypeError, ValueError):
        return default


def _session_turns(pd: dict, start_role: str) -> list[dict]:
    """A previous_dialog's `dialog` (list of {text}) as alternating role turns.
    Speakers alternate; `start_role` says whether turn 0 is the user or the
    assistant. (Attribution matters little for retrieval — the answer-bearing
    turn is found by text either way — but keeping it stable lets HyMem's
    user-centric extraction treat one side consistently as 'the user'.)"""
    other = "assistant" if start_role == "user" else "user"
    turns = []
    for i, t in enumerate(pd.get("dialog") or []):
        content = (t.get("text") if isinstance(t, dict) else str(t)) or ""
        if content.strip():
            turns.append({"role": start_role if i % 2 == 0 else other,
                          "content": content.strip()})
    return turns


def _persona_facts(ex: dict) -> list[str]:
    """Every annotated persona line for the example, de-duplicated — MSC's
    'important personal points', used as the durability ground truth."""
    facts: list[str] = []
    def _add(x):
        if isinstance(x, str) and x.strip():
            facts.append(x.strip())
        elif isinstance(x, list):
            for y in x:
                _add(y)
    for key in ("personas", "init_personas", "personas_update1", "personas_update2"):
        _add(ex.get(key))
    for pd in ex.get("previous_dialogs") or []:
        _add(pd.get("personas"))
    seen, out = set(), []
    for f in facts:
        if f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out


def load_msc_data(path: str | None, sample: int, seed: int, *,
                  start_role: str = "user", gap_days: float = 1.0,
                  base_date: str = "2023-01-01") -> list[dict]:
    """Load + normalize MSC examples. Accepts a JSON array or JSONL of the
    MemGPT/MSC-Self-Instruct shape. Returns normalized examples:
        {id, sessions: [[{role,content}]], session_dates: [str], question, answer,
         persona_facts: [str], n_sessions}
    Session dates are SYNTHESIZED (MSC carries relative gaps, not timestamps):
    monotonic from `base_date`, spaced by each session's parsed gap — real
    temporal separation for the recency/supersession mechanics, not ground-truth
    dates. `path=None` returns the built-in synthetic fixture (for --sim)."""
    if not path:
        raw = _SIM_FIXTURE
    else:
        text = Path(path).read_text(encoding="utf-8")
        raw = ([json.loads(l) for l in text.splitlines() if l.strip()]
               if path.endswith(".jsonl") else json.loads(text))
        if not isinstance(raw, list):
            raise ValueError("MSC data must be a JSON array (or JSONL) of examples")

    out = []
    base = datetime.strptime(base_date, "%Y-%m-%d")
    for i, ex in enumerate(raw):
        prev = ex.get("previous_dialogs") or []
        sessions, dates, cursor = [], [], base
        for pd in prev:
            turns = _session_turns(pd, start_role)
            if not turns:
                continue
            sessions.append(turns)
            dates.append(cursor.strftime("%Y-%m-%d %H:%M"))
            cursor += timedelta(days=_gap_days(pd, gap_days))
        if not sessions:
            continue
        si = ex.get("self_instruct") or {}
        out.append({
            "id": str((ex.get("metadata") or {}).get("initial_data_id") or f"msc_{i}"),
            "sessions": sessions,
            "session_dates": dates,
            "question": (si.get("B") or "").strip() or None,
            "answer": (si.get("A") or "").strip() or None,
            "persona_facts": _persona_facts(ex),
            "n_sessions": len(sessions),
        })
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:sample] if sample else out


# ── the HyMem driver (self-contained; configurable model) ───────────────────

class MSCAdapter:
    """A HyMem instance over one example's isolated temp DB. Mirrors
    `HyMemAdapter` but is self-contained so it controls the dream model (the LME
    adapter now shares the same pinned v4-flash default) and carries only the
    levers MSC needs."""

    # Retrieval aperture. These three were sized for MSC (histories of 20-60
    # turns, so 15 raw-turn slots is a ~25% aperture). LoCoMo histories are
    # 369-689 turns — the SAME constants there surface ~2% of the history, and
    # `message_hits` is the only channel that can carry a gold *turn* to the
    # reader. Kept as the defaults (MSC's 84.0 baseline is frozen against them),
    # but now overridable per-run so the aperture can be A/B'd per corpus.
    APERTURE = dict(_MSC_APERTURE)

    def __init__(self, db_path: Path, *, api_key: str = "", sim: bool = False,
                 hymem_model: str = _HYMEM_MODEL, hymem_base_url: str = _DEEPSEEK_BASE_URL,
                 hymem_thinking: str = "auto",
                 embeddings: bool = False, rules_extraction: bool | None = None,
                 graph_multihop: bool = False, aperture: dict | None = None,
                 facts_enabled: bool | None = None,
                 facts_extraction: bool | None = None):
        self.db_path = Path(db_path)
        self.api_key = api_key
        self.sim = sim
        self.hymem_model = hymem_model
        try:
            self.hymem_base_url = validate_http_endpoint(
                hymem_base_url, label="memory pipeline"
            ).url
        except ValueError as exc:
            raise BenchmarkIntegrityError(str(exc)) from exc
        self.hymem_thinking = hymem_thinking
        self.embeddings = embeddings
        self.rules_extraction = rules_extraction
        self.graph_multihop = graph_multihop
        self.facts_enabled = facts_enabled
        self.facts_extraction = facts_extraction
        # None-valued entries mean "unset" so callers can pass argparse results
        # straight through without reimplementing the defaults.
        self.aperture = {**self.APERTURE,
                         **{k: v for k, v in (aperture or {}).items() if v is not None}}
        self.hy = None
        self.pipeline_llm = None
        self.embedding_client = None
        self.last_indexing_summary = None
        self.indexing_runs: list[dict] = []
        self.indexing_skip: dict | None = None
        self._owned_resources = OwnedResourceScope("MSC memory-adapter resources")

    def open(self):
        from hymem import HyMem, HyMemConfig
        overrides: dict = {}
        if self.rules_extraction is not None:
            overrides["rules_extraction_enabled"] = self.rules_extraction
        if self.graph_multihop:
            overrides["graph_multihop_enabled"] = True
        # E1 narrative facts (schema v26). None = config default (both ON).
        # --no-facts is the read-side control arm on the same store;
        # --no-facts-extraction changes what is stored, so it needs a rebuild.
        if self.facts_enabled is not None:
            overrides["facts_enabled"] = self.facts_enabled
        if self.facts_extraction is not None:
            overrides["facts_extraction_enabled"] = self.facts_extraction
        overrides.update(self.aperture)
        # RAPTOR aggregation layer: pinned OFF explicitly. The config default
        # flipped False -> True on 2026-08-26 (G-FLIP PASS); this benchmark was
        # a default-config consumer, so without the pin the flip would silently
        # switch the layer + digest ON and the canonical baseline would stop
        # being comparable to every run behind it. Moving a benchmark onto the
        # shipped config is a pre-registered scored decision, not a side effect
        # of a default change. Written AFTER the aperture merge on purpose:
        # this is an invariant, not a tunable, so a caller-supplied aperture
        # dict must not be able to clobber it.
        overrides["aggregation_nodes_enabled"] = False
        cfg = HyMemConfig(root=self.db_path.parent, **overrides)
        # HyMemConfig chooses one fixed filename under its store root. An
        # arbitrary caller filename is not an alternate physical database:
        # reject it before allocating clients or opening/mutating any store.
        if self.db_path != cfg.db_path:
            raise BenchmarkIntegrityError(
                "MSC adapter database path must be its store root's hymem.sqlite"
            )
        embedding_client = None
        try:
            if self.sim:
                from hymem.extraction.llm import StubLLMClient
                llm = StubLLMClient(default="[]")
            else:
                from hymem.contrib.openai_client import OpenAICompatibleClient
                llm = OpenAICompatibleClient(
                    api_key=self.api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
                    base_url=self.hymem_base_url, model=self.hymem_model,
                    thinking=self.hymem_thinking)
                self._owned_resources.own(
                    llm, label="memory pipeline client"
                )
                if self.embeddings:
                    from hymem.contrib.openai_embedding_client import (
                        OpenAICompatibleEmbeddingClient,
                        is_loopback_embedding_url,
                        is_official_openai_embedding_url,
                    )
                    # Fallback constants imported from the LME adapter so --embeddings
                    # means the SAME local embed server/model in both benchmarks
                    # (comparability frame); env vars still override.
                    from longmemeval_adapter import (LOCAL_EMBED_API_KEY, LOCAL_EMBED_BASE_URL,
                                                     LOCAL_EMBED_DIM, LOCAL_EMBED_MODEL)
                    env = os.environ.get
                    embedding_base_url = env("HYMEM_EMBEDDING_BASE_URL") or LOCAL_EMBED_BASE_URL
                    embedding_api_key = env("HYMEM_EMBEDDING_API_KEY")
                    if not embedding_api_key and is_loopback_embedding_url(embedding_base_url):
                        embedding_api_key = LOCAL_EMBED_API_KEY
                    if (
                        not embedding_api_key
                        and is_official_openai_embedding_url(embedding_base_url)
                    ):
                        embedding_api_key = env("OPENAI_API_KEY")
                    embedding_client = OpenAICompatibleEmbeddingClient(
                        api_key=embedding_api_key,
                        base_url=embedding_base_url,
                        model=env("HYMEM_EMBEDDING_MODEL") or LOCAL_EMBED_MODEL,
                        dim=int(env("HYMEM_EMBEDDING_DIM") or LOCAL_EMBED_DIM),
                        # Reusable MSC/LoCoMo stores are comparable only when
                        # the requested vector space is immutable.  A provider
                        # returning any other dimension latches the build as
                        # invalid even if hot-ingest catches that call and a
                        # later retry happens to recover.
                        pin_dimension=True,
                        deployment_revision=env(
                            "HYMEM_EMBEDDING_DEPLOYMENT_REVISION"
                        ),
                        deployment_tenant=env(
                            "HYMEM_EMBEDDING_DEPLOYMENT_TENANT"
                        ),
                    )
                    self._owned_resources.own(
                        embedding_client, label="embedding client"
                    )
            self.embedding_client = embedding_client
            self.pipeline_llm = llm
            self.hy = HyMem(cfg, llm=llm, embedding_client=embedding_client)
            self._owned_resources.own(self.hy, label="memory store")
            return self
        except BaseException as exc:
            self._owned_resources.close(primary_exception=exc)
            raise

    def close(self, *, primary_exception: BaseException | None = None):
        try:
            return self._owned_resources.close(
                primary_exception=primary_exception
            )
        finally:
            self.hy = None

    def ingest(
        self,
        ex: dict,
        *,
        dream_each: bool = False,
        indexing_max_cycles: int = DEFAULT_INDEXING_MAX_CYCLES,
        indexing_timeout_s: float = DEFAULT_INDEXING_TIMEOUT_S,
    ) -> None:
        """One HyMem session per MSC session — NEVER merged (session_count is the
        whole point of the corpus), each stamped with its synthesized date.
        `dream_each` dreams after EVERY session (the live-store posture): profile
        and episode evidence accumulates across dreams instead of arriving in one
        end-of-history batch."""
        for i, (turns, date) in enumerate(zip(ex["sessions"], ex["session_dates"])):
            self.hy.log_messages(
                f'{ex["id"]}_s{i}',
                [(t["role"], t["content"], date) for t in turns])
            if dream_each:
                # This flag models a live store deliberately: each newly
                # appended session reaches a complete, healthy fixed point
                # before the next session arrives. It is more expensive than
                # end-of-history convergence, but it never means "one cycle".
                self.dream(
                    max_cycles=indexing_max_cycles,
                    timeout_s=indexing_timeout_s,
                    trigger=f"after_session:{i}",
                )

    def _durable_status(self, dh) -> dict:
        """Shared benchmark completion status for MSC and LoCoMo."""

        return durable_indexing_status(
            dh, getattr(self, "embedding_client", None),
        )

    def dream(
        self,
        *,
        max_cycles: int = DEFAULT_INDEXING_MAX_CYCLES,
        timeout_s: float = DEFAULT_INDEXING_TIMEOUT_S,
        trigger: str = "end_of_history",
    ) -> dict:
        """Converge one indexing wave; never treat one dream as completion."""

        if self.hy is None:
            raise BenchmarkIntegrityError("MSC adapter must be open before indexing")
        dh = self.hy.fork()
        summary: dict | None = None
        try:
            initial_status = self._durable_status(dh)
            try:
                raw_summary = converge_indexing(
                    dh.dream,
                    status=lambda: self._durable_status(dh),
                    max_cycles=max_cycles,
                    timeout_s=timeout_s,
                    require_healthy=True,
                )
                summary = {
                    **raw_summary,
                    "protocol": INDEXING_PROVENANCE_VERSION,
                    "trigger": trigger,
                    "initial_status": initial_status,
                    "cleanup_errors": [],
                }
            except IndexingConvergenceError as exc:
                summary = {
                    **dict(exc.summary),
                    "protocol": INDEXING_PROVENANCE_VERSION,
                    "trigger": trigger,
                    "initial_status": initial_status,
                    "cleanup_errors": [],
                }
                self.last_indexing_summary = summary
                self.indexing_runs.append(summary)
                raise IndexingConvergenceError(str(exc), summary) from exc
        finally:
            cleanup_sink = (
                summary["cleanup_errors"] if summary is not None else None
            )
            run_cleanup_actions(
                [
                    ("dream_fork_close", dh.close),
                    ("query_cache_invalidation", self.hy.invalidate_query_caches),
                ],
                primary_exception=sys.exc_info()[1],
                evidence_sink=cleanup_sink,
            )

        assert summary is not None
        self.last_indexing_summary = summary
        self.indexing_runs.append(summary)
        return summary

    def mark_indexing_skipped(
        self,
        reason: str,
        *,
        max_cycles: int = DEFAULT_INDEXING_MAX_CYCLES,
        timeout_s: float = DEFAULT_INDEXING_TIMEOUT_S,
    ) -> None:
        """Record an explicit non-comparable path without invoking an LLM."""

        if reason not in {"no_dream", "simulation"}:
            raise ValueError("unknown indexing skip reason")
        if self.indexing_runs:
            raise BenchmarkIntegrityError("cannot skip indexing after it has run")
        self.indexing_skip = {
            "reason": reason,
            "settings": {
                "max_cycles_per_convergence": max_cycles,
                "timeout_s_per_convergence": float(timeout_s),
                "require_healthy": True,
            },
            "observed_status": (
                self._durable_status(self.hy) if self.hy is not None else None
            ),
        }

    def indexing_provenance(self, *, scope_id: str) -> dict:
        """One additive, conversation-scoped indexing and usage receipt."""

        if self.indexing_skip is not None:
            return {
                "protocol": INDEXING_PROVENANCE_VERSION,
                "scope_id": scope_id,
                "mode": "skipped_non_comparable",
                "skip_reason": self.indexing_skip["reason"],
                "comparable": False,
                "complete": False,
                "healthy": False,
                "convergence_count": 0,
                "cycles": 0,
                "settings": self.indexing_skip["settings"],
                "settings_applied": False,
                "observed_status": self.indexing_skip["observed_status"],
                "pipeline_usage": _known_zero_pipeline_usage(),
            }
        if not self.indexing_runs:
            raise BenchmarkIntegrityError(
                "indexing provenance requested before convergence"
            )

        run_receipts = []
        totals = {field: 0 for field in _DREAM_REPORT_TOTAL_FIELDS}
        budget_exhausted_cycles = 0
        provider_budget_exhausted_cycles = 0
        skipped_locked_cycles = 0
        for run in self.indexing_runs:
            (
                run_totals,
                run_budget_exhausted,
                run_provider_budget_exhausted,
                run_skipped_locked,
                final_cycle,
            ) = _fixed_report_evidence(run.get("reports"))
            run_reports = run["reports"]
            for field, value in run_totals.items():
                totals[field] = _bounded_count(
                    totals[field] + value,
                    f"indexing report total {field}",
                )
            budget_exhausted_cycles += run_budget_exhausted
            provider_budget_exhausted_cycles += run_provider_budget_exhausted
            skipped_locked_cycles += run_skipped_locked
            run_receipts.append({
                "trigger": run.get("trigger"),
                "cycles": run.get("cycles"),
                "report_count": len(run_reports),
                "complete": run.get("complete"),
                "healthy": run.get("healthy"),
                "failure_reason": run.get("failure_reason"),
                "elapsed_s": run.get("elapsed_s"),
                "dream_report_totals": run_totals,
                "budget_exhausted_cycles": run_budget_exhausted,
                "extraction_provider_attempt_budget_exhausted_cycles": (
                    run_provider_budget_exhausted
                ),
                "skipped_locked_cycles": run_skipped_locked,
                "final_cycle": final_cycle,
                "final_status": _final_health_projection(
                    run.get("final_status"), exact=False
                ),
            })
        latest = self.indexing_runs[-1]
        result = {
            "protocol": INDEXING_PROVENANCE_VERSION,
            "scope_id": scope_id,
            "mode": "converged",
            "comparable": True,
            "complete": all(run.get("complete") is True for run in self.indexing_runs),
            "healthy": all(run.get("healthy") is True for run in self.indexing_runs),
            "convergence_count": len(self.indexing_runs),
            "cycles": sum(int(run.get("cycles", 0)) for run in self.indexing_runs),
            "settings": {
                "max_cycles_per_convergence": latest.get("max_cycles"),
                "timeout_s_per_convergence": latest.get("timeout_s"),
                "require_healthy": True,
            },
            "runs": run_receipts,
            "dream_report_totals": totals,
            "budget_exhausted_cycles": budget_exhausted_cycles,
            "extraction_provider_attempt_budget_exhausted_cycles": (
                provider_budget_exhausted_cycles
            ),
            "skipped_locked_cycles": skipped_locked_cycles,
            "final_status": latest.get("final_status", {}),
            # The client is one per MSC example / LoCoMo conversation. This
            # cumulative block therefore covers every convergence cycle once.
            "pipeline_usage": usage_snapshot(self.pipeline_llm),
        }
        _validate_indexing_provenance(result)
        return result

    @property
    def store_build_receipt_path(self) -> Path:
        return self.db_path.parent / STORE_BUILD_RECEIPT_NAME

    def material_store_state(self) -> dict:
        """Canonical secret-free digest summary of the current SQLite state."""

        if self.hy is None:
            raise BenchmarkIntegrityError(
                "MSC adapter must be open before attesting material store state"
            )
        if self.db_path != self.hy.config.db_path:
            raise BenchmarkIntegrityError(
                "MSC adapter database path differs from its open memory store"
            )
        return compute_material_store_state(self.db_path)

    def embedding_store_state(self) -> dict:
        """Attest that every durable vector mirror uses the configured pin.

        The canonical material digest proves exact rows and vec/FTS logical
        contents.  This additional bounded summary makes the vector-space
        invariant explicit and validates it before a receipt can bless the
        store: all authoritative vector mirrors must contain finite, non-zero
        vectors with the endpoint-namespaced model and configured dimension.
        """

        embedding_identity = _benchmark_embedding_identity(
            self.embedding_client
        )
        if embedding_identity["configured"] is False:
            payload = {
                "version": EMBEDDING_STORE_ATTESTATION_VERSION,
                "enabled": False,
                "vector_space_sha256": None,
                "dimension": None,
                "total_vectors": 0,
                "tables": {},
                "vec_acceleration": False,
            }
            return {**payload, "sha256": content_hash(payload)}

        if _embedding_attribute(
            self.embedding_client, "dimension_integrity_ok"
        ) is not True:
            raise BenchmarkIntegrityError(
                "embedding provider contradicted the pinned benchmark dimension"
            )
        expected_model = embedding_identity["vector_space_key"]
        expected_dimension = embedding_identity["dimension"]
        conn = self.hy.read_conn
        table_counts: dict[str, int] = {}
        total_vectors = 0
        try:
            for table in _EMBEDDING_MIRROR_TABLES:
                count = 0
                # Table names are closed policy constants, never input.
                rows = conn.execute(
                    f'SELECT vector_json, model, dim FROM "{table}"'
                )
                for row in rows:
                    count += 1
                    stored_dim = row["dim"]
                    if (
                        row["model"] != expected_model
                        or isinstance(stored_dim, bool)
                        or not isinstance(stored_dim, int)
                        or stored_dim != expected_dimension
                        or not _validated_stored_vector(
                            row["vector_json"],
                            expected_dimension=expected_dimension,
                        )
                    ):
                        raise BenchmarkIntegrityError(
                            "stored embedding vector does not match the pinned "
                            "benchmark vector space"
                        )
                table_counts[table] = count
                total_vectors += count

            placeholders = ",".join("?" for _ in _EMBEDDING_VEC_TABLES)
            vec_names = {
                str(row[0]) for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    f"AND name IN ({placeholders})",
                    tuple(sorted(_EMBEDDING_VEC_TABLES)),
                )
            }
            metadata = {
                str(row[0]): row[1]
                for row in conn.execute(
                    "SELECT key, value FROM schema_meta "
                    "WHERE key IN ('vec_dim','vec_model')"
                )
            }
        except BenchmarkIntegrityError:
            raise
        except Exception as exc:
            raise BenchmarkIntegrityError(
                "benchmark embedding store state is unavailable"
            ) from exc

        if total_vectors <= 0:
            raise BenchmarkIntegrityError(
                "embedding-enabled benchmark store contains no durable vectors"
            )
        if vec_names and vec_names != _EMBEDDING_VEC_TABLES:
            raise BenchmarkIntegrityError(
                "benchmark sqlite-vec table set is incomplete"
            )
        if vec_names:
            try:
                metadata_dim = int(metadata.get("vec_dim"))
            except (TypeError, ValueError, OverflowError):
                metadata_dim = None
            if (
                metadata_dim != expected_dimension
                or metadata.get("vec_model") != expected_model
            ):
                raise BenchmarkIntegrityError(
                    "benchmark sqlite-vec metadata does not match the pinned "
                    "vector space"
                )
        elif metadata:
            raise BenchmarkIntegrityError(
                "benchmark sqlite-vec metadata exists without its virtual tables"
            )

        payload = {
            "version": EMBEDDING_STORE_ATTESTATION_VERSION,
            "enabled": True,
            "vector_space_sha256": content_hash({
                "model": expected_model,
                "dimension": expected_dimension,
            }),
            "dimension": expected_dimension,
            "total_vectors": total_vectors,
            "tables": table_counts,
            "vec_acceleration": bool(vec_names),
        }
        return {**payload, "sha256": content_hash(payload)}

    def store_build_identity(self, item: dict) -> dict:
        """Secret-free identity of the material memory representation."""

        if self.hy is None:
            raise BenchmarkIntegrityError(
                "MSC adapter must be open before building store identity"
            )
        cfg = self.hy.config
        missing = [
            name for name in _STORE_WRITE_CONFIG_FIELDS
            if not hasattr(cfg, name)
        ]
        if missing:
            raise BenchmarkIntegrityError(
                f"store identity config fields are missing: {missing}"
            )
        schema_row = self.hy.read_conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        if schema_row is None:
            raise BenchmarkIntegrityError("store schema version is unavailable")
        try:
            schema_version = int(schema_row[0])
        except (TypeError, ValueError):
            raise BenchmarkIntegrityError(
                "store schema version is malformed"
            ) from None
        try:
            pipeline = phase1_generation_binding(
                cfg.prompt_version, self.pipeline_llm
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkIntegrityError(
                "benchmark pipeline producer declaration is invalid"
            ) from exc
        producer = pipeline.get("producer")
        if (
            not isinstance(producer, dict)
            or producer.get("identity_exact") is not True
            or producer.get("reuse_scope") != "durable"
        ):
            raise BenchmarkIntegrityError(
                "benchmark pipeline has no exact durable Phase-1 producer identity"
            )
        pipeline = sanitize_for_artifact(pipeline)
        embedding_identity = sanitize_for_artifact(
            _benchmark_embedding_identity(self.embedding_client)
        )
        identity = {
            "source_sha256": content_hash({
                "id": item.get("id"),
                "sessions": item.get("sessions"),
                "session_dates": item.get("session_dates"),
            }),
            "hymem_code_sha256": _material_hymem_code_hash(),
            "ingestion_mapping": _ingestion_mapping_identity(
                type(self).open,
                type(self).ingest,
                type(self).dream,
                type(self)._durable_status,
            ),
            "schema_version": schema_version,
            "write_config": {
                name: getattr(cfg, name) for name in _STORE_WRITE_CONFIG_FIELDS
            },
            "pipeline": pipeline,
            "embedding": embedding_identity,
        }
        sanitized = sanitize_for_artifact(identity)
        if not isinstance(sanitized, dict):
            raise BenchmarkIntegrityError("store identity sanitization failed")
        return sanitized
    def validate_store_build_receipt(self, item: dict) -> dict:
        """Require exact build identity and actual store-state attestation."""

        path = self.store_build_receipt_path
        expected: dict | None = None
        expected_sha: str | None = None

        def safe_digest(value: object) -> str | None:
            return (
                value if isinstance(value, str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                else None
            )

        def fail(reason: str, **details) -> None:
            summary = {
                "protocol": INDEXING_PROVENANCE_VERSION,
                "status": "store_build_receipt_rejected",
                "failure_reason": reason,
                "receipt_file": STORE_BUILD_RECEIPT_NAME,
                "expected_identity_sha256": expected_sha,
                "remediation": "rebuild this conversation store with --fresh",
                **details,
            }
            raise IndexingConvergenceError(
                f"reused benchmark store has no compatible complete build "
                f"receipt ({reason}); rerun with --fresh",
                summary,
            ) from None

        try:
            expected = self.store_build_identity(item)
            expected_sha = content_hash(expected)
        except Exception:
            fail("store_build_identity_unavailable")

        if not path.is_file():
            fail("missing_store_build_receipt")
        try:
            if path.stat().st_size > 1_000_000:
                fail("oversized_store_build_receipt")
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            fail("malformed_store_build_receipt")
        if not isinstance(receipt, dict):
            fail("malformed_store_build_receipt")
        if receipt.get("version") != STORE_BUILD_RECEIPT_VERSION:
            fail(
                "incompatible_store_build_receipt_version",
                expected_receipt_version=STORE_BUILD_RECEIPT_VERSION,
                recorded_receipt_version=(
                    receipt.get("version")
                    if isinstance(receipt.get("version"), str)
                    and re.fullmatch(
                        r"[A-Za-z0-9_.-]{1,96}", receipt.get("version")
                    )
                    else None
                ),
            )
        if set(receipt) != set(_STORE_BUILD_RECEIPT_FIELDS):
            fail("malformed_store_build_receipt")
        if sanitize_for_artifact(receipt) != receipt:
            fail("malformed_store_build_receipt")
        material_state = receipt.get("material_state")
        embedding_state = receipt.get("embedding_state")
        if (
            receipt.get("status") != "complete_healthy"
            or not isinstance(receipt.get("identity"), dict)
            or not isinstance(receipt.get("indexing"), dict)
            or not isinstance(material_state, dict)
            or not isinstance(embedding_state, dict)
            or material_state.get("version") != MATERIAL_STORE_ATTESTATION_VERSION
            or not isinstance(material_state.get("sha256"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", material_state["sha256"])
            is None
            or not isinstance(material_state.get("tables"), dict)
            or embedding_state.get("version")
            != EMBEDDING_STORE_ATTESTATION_VERSION
            or not isinstance(embedding_state.get("enabled"), bool)
            or not isinstance(embedding_state.get("sha256"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", embedding_state["sha256"])
            is None
            or not isinstance(embedding_state.get("tables"), dict)
            or safe_digest(receipt.get("indexing_sha256")) is None
        ):
            fail("malformed_store_build_receipt")
        indexing = receipt["indexing"]
        actual_indexing_sha = content_hash(indexing)
        if receipt["indexing_sha256"] != actual_indexing_sha:
            fail(
                "corrupt_store_build_receipt",
                recorded_indexing_sha256=safe_digest(
                    receipt.get("indexing_sha256")
                ),
                actual_indexing_sha256=actual_indexing_sha,
            )
        try:
            _validate_indexing_provenance(
                indexing, item=item, attestation=True
            )
        except BenchmarkIntegrityError:
            fail("malformed_store_build_receipt")
        embedding_payload = {
            key: value for key, value in embedding_state.items()
            if key != "sha256"
        }
        if embedding_state["sha256"] != content_hash(embedding_payload):
            fail(
                "corrupt_store_build_receipt",
                recorded_embedding_state_sha256=safe_digest(
                    embedding_state.get("sha256")
                ),
                actual_embedding_state_sha256=content_hash(embedding_payload),
            )
        actual = receipt["identity"]
        actual_sha = content_hash(actual)
        if receipt.get("identity_sha256") != actual_sha:
            fail(
                "corrupt_store_build_receipt",
                recorded_identity_sha256=safe_digest(
                    receipt.get("identity_sha256")
                ),
                actual_identity_sha256=actual_sha,
            )
        if actual != expected:
            fields, truncated = _identity_mismatch_fields(expected, actual)
            fail(
                "store_build_identity_mismatch",
                actual_identity_sha256=actual_sha,
                mismatch_fields=fields,
                mismatch_fields_truncated=truncated,
            )
        try:
            current_state = self.material_store_state()
        except MaterialStoreAttestationError as exc:
            fail("store_material_attestation_failed", **exc.safe_details())
        except Exception:
            fail("store_material_attestation_failed")
        if current_state != material_state:
            tables, truncated = material_state_mismatch_tables(
                material_state, current_state
            )
            fail(
                "store_material_state_mismatch",
                recorded_material_state_sha256=material_state.get("sha256"),
                current_material_state_sha256=current_state.get("sha256"),
                mismatch_tables=tables,
                mismatch_tables_truncated=truncated,
            )
        try:
            current_embedding_state = self.embedding_store_state()
        except BenchmarkIntegrityError:
            fail("store_embedding_attestation_failed")
        except Exception:
            fail("store_embedding_attestation_failed")
        if current_embedding_state != embedding_state:
            fail(
                "store_embedding_state_mismatch",
                recorded_embedding_state_sha256=embedding_state.get("sha256"),
                current_embedding_state_sha256=current_embedding_state.get(
                    "sha256"
                ),
            )
        return receipt

    def publish_store_build_receipt(
        self, item: dict, indexing: dict
    ) -> dict:
        """Atomically publish proof only after complete healthy convergence."""

        indexing_attestation = _canonical_indexing_attestation(
            indexing, item=item
        )
        identity = self.store_build_identity(item)
        embedding_state = self.embedding_store_state()
        material_state = self.material_store_state()
        receipt = {
            "version": STORE_BUILD_RECEIPT_VERSION,
            "status": "complete_healthy",
            "identity_sha256": content_hash(identity),
            "identity": identity,
            "embedding_state": embedding_state,
            "material_state": material_state,
            "indexing_sha256": content_hash(indexing_attestation),
            "indexing": indexing_attestation,
        }
        receipt = sanitize_for_artifact(receipt)
        if (
            not isinstance(receipt, dict)
            or sanitize_for_artifact(receipt) != receipt
            or set(receipt) != set(_STORE_BUILD_RECEIPT_FIELDS)
        ):
            raise BenchmarkIntegrityError(
                "store build receipt sanitization is unstable"
            )
        if receipt["indexing_sha256"] != content_hash(receipt["indexing"]):
            raise BenchmarkIntegrityError("store indexing receipt digest drifted")
        _validate_indexing_provenance(
            receipt["indexing"], item=item, attestation=True
        )
        try:
            write_immutable_artifact(self.store_build_receipt_path, receipt)
        except FileExistsError as exc:
            raise BenchmarkIntegrityError(
                "refusing to overwrite an existing store build receipt; "
                "rerun with --fresh"
            ) from exc
        return receipt

    def search(self, query: str, top_k: int = 10) -> tuple[list[dict], dict]:
        """LME-parity retrieval: the same tiers and message-first ordering as
        `HyMemAdapter.search` (raw turns + procedures lead; episodes/chunks/graph
        facts confidence-ranked behind), same `[:top_k]` cut — the caller passes
        `top_k * 3` exactly like the LME pipeline does. Two MSC additions, both
        ADDITIVE (never consuming a raw-turn slot): the P4 `user_profile` tier is
        prepended ahead of the cut result (MSC content is profile-shaped and the
        tier is distance-invariant by construction — neither adapter consumed it
        before, which on MSC discarded the tier built for exactly this content),
        and `info` carries the FULL pre-truncation pool so misses can be split
        into retrieval loss vs ranking loss (the LME recall-ceiling discipline)."""
        embedding_identity = None
        if self.embedding_client is not None:
            embedding_identity = _benchmark_embedding_identity(
                self.embedding_client
            )
            if _embedding_attribute(
                self.embedding_client, "dimension_integrity_ok"
            ) is not True:
                raise BenchmarkIntegrityError(
                    "configured benchmark embedding retrieval was unavailable "
                    "or contradicted its pinned vector space"
                )
        result = self.hy.augment(query)
        if self.embedding_client is not None:
            semantic = getattr(result, "semantic_status", None)
            if (
                _embedding_attribute(
                    self.embedding_client, "dimension_integrity_ok"
                ) is not True
                or semantic is None
                or getattr(semantic, "configured", None) is not True
                or getattr(semantic, "attempted", None) is not True
                or getattr(semantic, "available", None) is not True
                or getattr(semantic, "model", None)
                != embedding_identity["vector_space_key"]
                or getattr(semantic, "dim", None)
                != embedding_identity["dimension"]
            ):
                # HyMem deliberately retains lexical retrieval on semantic
                # provider failure. That is desirable runtime resilience but
                # would silently change the treatment arm of an embedding
                # benchmark, so MSC/LoCoMo must fail before reader scoring.
                raise BenchmarkIntegrityError(
                    "configured benchmark embedding retrieval was unavailable "
                    "or contradicted its pinned vector space"
                )

        message_hits = []
        for hit in getattr(result, "message_hits", None) or []:
            text = (getattr(hit, "text", "") or "")[:600]
            if text.strip():
                message_hits.append({"content": f'[{getattr(hit, "role", "?")}] {text}',
                                     "type": "message_hit", "confidence": 0.7,
                                     "created_at": getattr(hit, "created_at", "") or ""})
        fts_hits = []
        for hit in getattr(result, "fts_hits", None) or []:
            text = (getattr(hit, "text", "") or "")[:600]
            if text.strip():
                fts_hits.append({"content": text, "type": "fts_hit", "confidence": 0.6})
        procedure_hits = []
        for proc in getattr(result, "procedures", None) or []:
            name = getattr(proc, "name", "")
            desc = (getattr(proc, "description", "") or "")[:400]
            content = f"Procedure: {name}: {desc}" if name else desc
            if content.strip():
                procedure_hits.append({"content": content[:600], "type": "procedure",
                                       "confidence": 0.75})
        episode_hits = []
        for ep in getattr(result, "episodes", None) or []:
            title = getattr(ep, "title", "")
            summary = (getattr(ep, "summary", "") or "")[:500]
            content = f"{title}: {summary}" if title else summary
            if content.strip():
                episode_hits.append({"content": content, "type": "episode",
                                     "confidence": 0.8})
        graph_facts = []
        for fact in getattr(result, "graph_facts", None) or []:
            graph_facts.append({"content": f"{fact.subject} {fact.predicate} {fact.object}",
                                "type": "graph_fact",
                                "confidence": getattr(fact, "confidence", 0.5)})

        memories = message_hits + procedure_hits
        rest = episode_hits + fts_hits + graph_facts
        rest.sort(key=lambda m: -m.get("confidence", 0))
        memories = (memories + rest)[:top_k]

        profile = []
        for p in getattr(result, "user_profile", None) or []:
            key = f" ({p.slot_key})" if getattr(p, "slot_key", None) else ""
            since = (getattr(p, "valid_at", "") or "")[:10]
            profile.append({"content": f"Known user profile — {p.slot}{key}: {p.value}"
                                       + (f" (since {since})" if since else ""),
                            "type": "profile",
                            "confidence": getattr(p, "confidence", 0.9)})
        memories = profile + memories

        aggregation_nodes = []
        for node in getattr(result, "aggregation_nodes", None) or []:
            title = getattr(node, "title", "")
            summary = (getattr(node, "summary", "") or "")[:600]
            content = f"{title}: {summary}" if title else summary
            if content.strip():
                aggregation_nodes.append(content)

        # E1 narrative facts — carried in `info`, NOT in `memories`, so the tier
        # renders as its own block and can never take a raw-turn slot (the same
        # non-competing contract aggregation_nodes and user_profile follow).
        narrative_facts = []
        for nf in getattr(result, "facts", None) or []:
            text = (getattr(nf, "text", "") or "")[:600]
            if text.strip():
                date = getattr(nf, "fact_date", None) or ""
                narrative_facts.append(f"[{date}] {text}" if date else text)

        info = {
            "total_matches": getattr(result, "total_message_matches", 0),
            "graph_count": getattr(result, "graph_count", None),
            "temporal_events": getattr(result, "temporal_events", []) or [],
            "aggregation_nodes": aggregation_nodes,
            "narrative_facts": narrative_facts,
            "n_profile": len(profile),
            "pool": [m["content"] for m in message_hits + fts_hits + episode_hits],
        }
        return memories, info

    def dump_markers(self, ex: dict) -> list[dict]:
        """Extracted behavioral markers with HyMem session_id + an is_rule label
        from MSC's persona annotations. Read post-dream (rows persist even after
        consolidation); the JOIN gives the per-MSC-session provenance recurrence
        needs."""
        generation_key = phase1_generation_binding(
            self.hy.config.prompt_version, self.pipeline_llm
        )["generation_key"]
        rows = self.hy.conn.execute(
            "SELECT bm.kind AS kind,bm.statement AS statement,"
            "c.session_id AS session_id "
            "FROM behavioral_markers bm JOIN chunks c ON bm.chunk_id=c.id "
            "JOIN current_phase1_publications publication "
            "ON publication.chunk_id=bm.chunk_id "
            "AND publication.phase1_generation_key=bm.phase1_generation_key "
            "WHERE bm.phase1_generation_key=? "
            "ORDER BY bm.id",
            (generation_key,),
        ).fetchall()
        facts = ex["persona_facts"]
        out = []
        for r in rows:
            stmt = r["statement"]
            is_rule = any(_lex_match(f, stmt) or _lex_match(stmt, f) for f in facts)
            out.append({"kind": r["kind"], "statement": stmt,
                        "session_id": r["session_id"], "is_rule": bool(is_rule)})
        return out


# MSC deixis: self_instruct questions are asked BY the conversation partner TO
# the user ("B asks A"). With the default --start-role user (turn 0 = speaker A
# = [user]) that means: "you"/"your" in the question is the USER; "I"/"me"/"my"
# is the PARTNER ([assistant] turns). The LME prompt never needed this — its
# questions are third-person — and the parity-run audit showed the reader
# resolving it wrong in BOTH directions: rejecting a gold [assistant] fact as
# "no memory of YOUR parents teaching you an instrument" (msc_431), and
# answering AS the partner persona with the partner's fact when asked about the
# user's (msc_108: "my favorite food is Italian" vs the user's stated Mexican).
# NOTE: the mapping inverts under --start-role assistant.
MSC_PERSPECTIVE_CLAUSE = (
    "\nThe memories are turns from past conversations between the user ([user] turns) "
    "and their conversation partner ([assistant] turns). The QUESTION is asked by the "
    "conversation partner TO the user. So in the question, 'you'/'your' refers to the "
    "USER — answer from [user] turns and user-profile facts; 'I'/'me'/'my' refers to "
    "the PARTNER asking the question — answer from [assistant] turns. Attribute each "
    "fact to the speaker who actually said it before answering."
)

# Overrides the base LME prompt's abstention permission ("say I don't have
# enough information"), which is load-bearing on LME (_abs questions score
# abstention as correct) but a pure miss generator here: every MSC
# self_instruct question is answerable by construction. v3 residual audit
# (2026-07-28) found 5/21 misses were abstentions WITH the gold visible in
# context, plus one hedge (msc_187: stated the gold fact, then talked itself
# out of it).
MSC_ANSWERABILITY_CLAUSE = (
    "\nEvery question in this benchmark has an answer stated somewhere in the "
    "memories — never reply that you don't have enough information. If no memory "
    "states the answer outright, commit to the single best-supported answer from "
    "what the memories do say. When you find the fact, state it directly and "
    "confidently; do not add disclaimers or talk yourself out of it."
)


def model_identity_fields(args, answer_llm, judge_llm, pipeline_llm) -> dict:
    """Identity fields for the triad's bare-list benchmark artifacts."""

    if getattr(args, "sim", False):
        return {
            "answer_model": None,
            "answer_base_url": None,
            "answer_extra_body": None,
            "judge_model": None,
            "judge_base_url": None,
            "judge_extra_body": None,
            "hymem_model": None,
            "hymem_base_url": None,
            "hymem_thinking": None,
            "hymem_extra_body": {},
        }
    return {
        "answer_model": args.answer_model,
        "answer_base_url": validate_http_endpoint(
            args.answer_base_url, label="reader"
        ).url,
        "answer_extra_body": (
            copy.deepcopy(answer_llm.extra_body)
            if answer_llm is not None else None
        ),
        "judge_model": args.judge_model,
        "judge_base_url": _DEEPSEEK_BASE_URL,
        "judge_extra_body": (
            copy.deepcopy(judge_llm.extra_body)
            if judge_llm is not None else None
        ),
        "hymem_model": args.hymem_model,
        "hymem_base_url": validate_http_endpoint(
            args.hymem_base_url, label="memory pipeline"
        ).url,
        "hymem_thinking": getattr(pipeline_llm, "thinking_mode", None),
        "hymem_extra_body": copy.deepcopy(getattr(
            pipeline_llm, "effective_extra_body", {}
        )),
    }


def _indexing_limits(args) -> tuple[int, float]:
    """Resolve direct-call fixtures and CLI runs onto the same safe defaults."""

    max_cycles = getattr(args, "indexing_max_cycles", DEFAULT_INDEXING_MAX_CYCLES)
    timeout_s = getattr(args, "indexing_timeout_s", DEFAULT_INDEXING_TIMEOUT_S)
    if (
        isinstance(max_cycles, bool)
        or not isinstance(max_cycles, int)
        or max_cycles <= 0
    ):
        raise BenchmarkIntegrityError("indexing max_cycles must be positive")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0
    ):
        raise BenchmarkIntegrityError(
            "indexing timeout_s must be positive and finite"
        )
    return max_cycles, float(timeout_s)


def prepare_indexing(
    adapter: MSCAdapter,
    item: dict,
    args,
    *,
    scope_id: str,
    reuse: bool = False,
) -> dict:
    """Ingest if needed, then converge or record an explicit skipped path.

    Reused LoCoMo stores still run convergence. File existence proves neither
    that the old one-cycle builder drained its backlog nor that a newer prompt
    or schema left no work. The convergence wave starts with a read-only status
    snapshot and then checks durable state after every bounded dream cycle.
    """

    max_cycles, timeout_s = _indexing_limits(args)
    skip_reason = (
        "simulation" if getattr(args, "sim", False)
        else "no_dream" if getattr(args, "no_dream", False)
        else None
    )
    dream_each = bool(
        getattr(args, "dream_per_session", False) and skip_reason is None
    )
    validated_receipt = None
    try:
        if reuse and skip_reason is not None:
            raise IndexingConvergenceError(
                "a skipped-indexing run cannot reuse a store that may already "
                "contain dreamed state; rerun with --fresh",
                {
                    "protocol": INDEXING_PROVENANCE_VERSION,
                    "status": "store_reuse_rejected",
                    "failure_reason": "skipped_indexing_reused_store",
                    "remediation": "use --fresh or omit --db-dir",
                },
            )
        if reuse:
            # A SQLite file is not a cache identity. Missing legacy receipts
            # and every material identity mismatch require an explicit rebuild.
            validated_receipt = adapter.validate_store_build_receipt(item)
        if not reuse:
            adapter.ingest(
                item,
                dream_each=dream_each,
                indexing_max_cycles=max_cycles,
                indexing_timeout_s=timeout_s,
            )
        if skip_reason is not None:
            adapter.mark_indexing_skipped(
                skip_reason, max_cycles=max_cycles, timeout_s=timeout_s
            )
        elif reuse:
            adapter.dream(
                max_cycles=max_cycles,
                timeout_s=timeout_s,
                trigger="reused_store_validation",
            )
        elif not dream_each:
            adapter.dream(
                max_cycles=max_cycles,
                timeout_s=timeout_s,
                trigger="end_of_history",
            )
        indexing = adapter.indexing_provenance(scope_id=scope_id)
        if skip_reason is None:
            # Reuse does not republish a receipt. Validate the newly observed
            # convergence evidence itself before any question can be scored.
            _validate_indexing_provenance(indexing, item=item)
        if reuse:
            # Re-attest after the convergence wave and its provenance snapshot,
            # immediately before returning control to scoring.  A supposedly
            # no-op dream that altered source/derived/retrieval state is not a
            # compatible reuse of the immutable build receipt.
            post_convergence_receipt = adapter.validate_store_build_receipt(item)
            if (
                post_convergence_receipt.get("identity_sha256")
                != validated_receipt.get("identity_sha256")
                or post_convergence_receipt.get("material_state")
                != validated_receipt.get("material_state")
                or post_convergence_receipt.get("embedding_state")
                != validated_receipt.get("embedding_state")
                or post_convergence_receipt.get("indexing_sha256")
                != validated_receipt.get("indexing_sha256")
            ):
                raise IndexingConvergenceError(
                    "reused benchmark store receipt changed during convergence; "
                    "rerun with --fresh",
                    {
                        "protocol": INDEXING_PROVENANCE_VERSION,
                        "status": "store_build_receipt_rejected",
                        "failure_reason": "store_build_receipt_changed_during_reuse",
                        "remediation": "rebuild this conversation store with --fresh",
                    },
                )
    except IndexingConvergenceError as exc:
        failed = {
            **dict(exc.summary),
            "scope_id": scope_id,
            "status": "failed_before_scoring",
            "pipeline_usage": usage_snapshot(adapter.pipeline_llm),
        }
        reason = failed.get("failure_reason") or "unknown"
        raise IndexingConvergenceError(
            f"indexing scope {scope_id!r} failed before scoring "
            f"(reason={reason}, cycles={failed.get('cycles')})",
            failed,
        ) from exc
    if reuse:
        indexing["store_build_receipt"] = {
            "version": STORE_BUILD_RECEIPT_VERSION,
            "status": "validated",
            "identity_sha256": post_convergence_receipt["identity_sha256"],
            "indexing_sha256": post_convergence_receipt["indexing_sha256"],
            "material_state_sha256": post_convergence_receipt[
                "material_state"
            ]["sha256"],
            "file": STORE_BUILD_RECEIPT_NAME,
        }
    elif skip_reason is not None:
        indexing["store_build_receipt"] = {
            "version": STORE_BUILD_RECEIPT_VERSION,
            "status": "not_published_non_comparable",
            "identity_sha256": None,
            "indexing_sha256": None,
            "material_state_sha256": None,
            "file": STORE_BUILD_RECEIPT_NAME,
        }
    else:
        try:
            receipt = adapter.publish_store_build_receipt(item, indexing)
            # Close the publication/digest seam before any question is scored.
            receipt = adapter.validate_store_build_receipt(item)
        except IndexingConvergenceError as exc:
            failed = {
                **dict(exc.summary),
                "scope_id": scope_id,
                "status": "failed_before_scoring",
                "pipeline_usage": usage_snapshot(adapter.pipeline_llm),
            }
            raise IndexingConvergenceError(
                "fresh benchmark store failed post-publication attestation; "
                "rerun with --fresh",
                failed,
            ) from exc
        except Exception as exc:
            failed = {
                "protocol": INDEXING_PROVENANCE_VERSION,
                "scope_id": scope_id,
                "status": "failed_before_scoring",
                "failure_reason": "store_build_receipt_publication_failed",
                "indexing": indexing,
                "remediation": "rerun this store with --fresh",
                "exception_type": bounded_exception_type(exc),
            }
            raise IndexingConvergenceError(
                "indexing converged but its immutable store receipt could not "
                "be published; rerun with --fresh",
                failed,
            ) from exc
        indexing["store_build_receipt"] = {
            "version": STORE_BUILD_RECEIPT_VERSION,
            "status": "published",
            "identity_sha256": receipt["identity_sha256"],
            "indexing_sha256": receipt["indexing_sha256"],
            "material_state_sha256": receipt["material_state"]["sha256"],
            "file": STORE_BUILD_RECEIPT_NAME,
        }
    return indexing


def run_or_record_indexing_failure(
    work,
    *,
    benchmark: str,
    out_path: str | None,
    extraction_canary: dict,
):
    """Persist bounded fail-before-score evidence, then preserve the error."""

    try:
        return work()
    except IndexingConvergenceError as exc:
        artifact = {
            "version": INDEXING_PROVENANCE_VERSION,
            "benchmark": benchmark,
            "status": "failed_before_scoring",
            "indexing": dict(exc.summary),
            "extraction_canary": extraction_canary,
        }
        sanitized = sanitize_for_artifact(artifact)
        print(
            "INDEXING FAILURE (no affected questions were scored): "
            + json.dumps(sanitized, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        if out_path:
            failure_path = Path(f"{out_path}.indexing-failure.json")
            try:
                write_immutable_artifact(failure_path, artifact)
            except Exception as receipt_exc:
                print(
                    f"Could not publish {failure_path}: "
                    f"{type(receipt_exc).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"  indexing failure provenance → {failure_path}",
                    file=sys.stderr,
                    flush=True,
                )
        raise


# ── probes ──────────────────────────────────────────────────────────────────

def _gold_session_index(ex: dict) -> int:
    """Which session the gold answer was stated in (best lexical match), or -1.
    Drives the E1 accuracy-by-session-distance breakdown."""
    ans = ex.get("answer") or ""
    best = -1
    for i, turns in enumerate(ex["sessions"]):
        if any(_lex_match(ans, t["content"], tau=0.6) for t in turns):
            best = i
    return best


def _recall_runtime_snapshot(
    adapter: MSCAdapter,
    *,
    scope_id: str,
    indexing: dict | None,
    indexing_failure: dict | None = None,
) -> dict[str, Any]:
    """Capture one isolated MSC store's work before its owner is closed."""

    return {
        "scope_id": scope_id,
        "memory_pipeline_usage": (
            _known_zero_pipeline_usage()
            if adapter.sim else usage_snapshot(adapter.pipeline_llm)
        ),
        "embedding_usage": embedding_usage_snapshot(
            adapter.embedding_client,
            configured=bool(adapter.embeddings and not adapter.sim),
        ),
        "indexing": dict(indexing) if isinstance(indexing, dict) else None,
        "indexing_failure": (
            sanitize_for_artifact(
                dict(indexing_failure), _preserve_evidence_text=False
            )
            if isinstance(indexing_failure, dict) else None
        ),
    }


def _recall_failure_row(ex: dict, exc: Exception) -> dict[str, Any]:
    """Return a denominator-preserving row with only a bounded failure code."""

    if isinstance(exc, IndexingConvergenceError):
        raw_reason = exc.summary.get("failure_reason")
        reason = bounded_failure_text(
            f"indexing_failure:{raw_reason or 'unspecified_failure'}"
        )
    else:
        reason = bounded_failure_text(
            f"probe_failure:{bounded_exception_type(exc)}"
        )
    return {
        "id": ex["id"],
        "question_id": ex["id"],
        "question_type": "recall",
        "correct": False,
        "benchmark_failure": reason,
        "judge_raw": "",
        "judge_error": False,
    }


def _pre_scoring_indexing_failure(exc: Exception) -> dict[str, Any]:
    """Bounded evidence when a question never produced an indexing receipt."""

    return {
        "status": "failed_before_scoring",
        "complete": False,
        "healthy": False,
        "comparable": False,
        "failure_reason": (
            f"pre_scoring_failure:{bounded_exception_type(exc)}"
        ),
    }


def run_recall(
    ex: dict, args, answer_llm, judge_llm, *, on_checkpoint=None,
) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="msc_"))
    scope_id = f"msc:{ex['id']}"
    indexing: dict | None = None
    callback_emitted = False
    adapter = MSCAdapter(tmp / "hymem.sqlite", api_key=args.api_key, sim=args.sim,
                         hymem_model=args.hymem_model, hymem_base_url=args.hymem_base_url,
                         hymem_thinking=args.hymem_thinking,
                         embeddings=args.embeddings, rules_extraction=args.rules_extraction,
                         graph_multihop=args.graph_multihop,
                         facts_enabled=args.facts,
                         facts_extraction=args.facts_extraction)
    try:
        adapter.open()
        indexing = prepare_indexing(
            adapter, ex, args, scope_id=scope_id
        )
        # top_k * 3 at the pipeline layer, EXACTLY like the LME driver — the
        # silently-missing ×3 was the whole BEAM June regression, and here it
        # additionally starved the consolidated tiers out of memories[:top_k].
        memories, info = adapter.search(ex["question"], top_k=args.top_k * 3)
        # Step-0 diagnostics (lexical, tau=0.6 — a signal, not ground truth):
        # gold_in_context = the reader COULD have answered (miss ⇒ synthesis/judge);
        # gold_in_pool = retrievable pre-truncation (in pool but not context ⇒
        # ranking/cut loss; in neither ⇒ retrieval loss).
        joined = " ".join(m["content"] for m in memories)
        gold_in_context = _lex_match(ex["answer"], joined, tau=0.6)
        gold_in_pool = gold_in_context or _lex_match(
            ex["answer"], " ".join(info["pool"]), tau=0.6)
        judge_raw = ""        # only the judge branch below can set this
        if args.sim:
            # Offline: no answer/judge LLM. Test the thing --sim CAN test — did
            # retrieval surface the gold answer?
            ai = memories[0]["content"] if memories else ""
            correct = gold_in_context
        else:
            from longmemeval_adapter import answer_question, judge_scored
            question_date = ex["session_dates"][-1] if ex["session_dates"] else ""
            ai = answer_question(answer_llm, memories, ex["question"],
                                 total_matches=info["total_matches"],
                                 graph_count=info["graph_count"],
                                 temporal_events=info["temporal_events"],
                                 aggregation_nodes=info["aggregation_nodes"],
                                 narrative_facts=info["narrative_facts"],
                                 question_date=question_date,
                                 extra_system=MSC_PERSPECTIVE_CLAUSE
                                              + MSC_ANSWERABILITY_CLAUSE)
            correct, judge_raw = judge_scored(judge_llm, "single-session-user",
                                              ex["question"], ex["answer"], ai)
        gi = _gold_session_index(ex)
        rec = {"id": ex["id"], "question_id": ex["id"],
               "question_type": "recall",
               # NOT bool(): that coerced a judge outage into a wrong answer,
               # which is the deflation D3 names. None = UNSCORED.
               "correct": (None if correct is None else bool(correct)),
               "judge_raw": judge_raw,
               "judge_error": bool(judge_raw) and correct is None,
               "question": ex["question"], "answer": ex["answer"], "ai_answer": ai,
               "n_sessions": ex["n_sessions"], "gold_session": gi,
               "gold_distance": (ex["n_sessions"] - gi) if gi >= 0 else -1,
               "gold_in_context": bool(gold_in_context),
               "gold_in_pool": bool(gold_in_pool),
               "n_memories": len(memories), "n_profile": info["n_profile"],
               # E1 mechanism read (do this BEFORE the score): all-zero n_facts
               # means the tier never reached the reader, so a flat score is a
               # no-op by construction rather than a null result.
               "n_facts": len(info["narrative_facts"]),
               "indexing_scope_id": indexing["scope_id"],
               "indexing": indexing,
               "indexing_comparable": bool(indexing["comparable"]),
               "benchmark_comparable": bool(indexing["comparable"]),
               "non_comparable_reason": indexing.get("skip_reason"),
               # Bare-list MSC artifacts have no top-level config envelope, so
               # bind each scored row to the effective request identities.
               **model_identity_fields(
                   args, answer_llm, judge_llm, adapter.pipeline_llm
               )}
        if args.dump_context:
            # Re-render the IDENTICAL context the answerer saw (same function,
            # same char caps — the LME distillation dry-run trick) so synthesis
            # misses can be audited from the results JSON alone.
            from longmemeval_adapter import _render_answer_context
            rec["context"] = _render_answer_context(
                memories, None, info["total_matches"], info["graph_count"],
                info["temporal_events"], info["aggregation_nodes"],
                narrative_facts=info["narrative_facts"])
        if on_checkpoint is not None:
            callback_emitted = True
            on_checkpoint(
                rec,
                _recall_runtime_snapshot(
                    adapter, scope_id=scope_id, indexing=indexing
                ),
            )
        return rec
    except Exception as exc:
        # A provider/indexing failure is still an attempted benchmark item.  Its
        # bounded row must become durable before the stack unwinds; otherwise a
        # crash at this boundary silently loses paid work and shrinks the
        # denominator.  Callback failures themselves are never retried here.
        if on_checkpoint is not None and not callback_emitted:
            callback_emitted = True
            indexing_failure = (
                dict(exc.summary)
                if isinstance(exc, IndexingConvergenceError) else
                _pre_scoring_indexing_failure(exc)
                if indexing is None else None
            )
            on_checkpoint(
                _recall_failure_row(ex, exc),
                _recall_runtime_snapshot(
                    adapter,
                    scope_id=scope_id,
                    indexing=indexing,
                    indexing_failure=indexing_failure,
                ),
            )
        raise
    finally:
        cleanup_actions = [("adapter_close", adapter.close)]
        if not args.keep_db:
            import shutil
            cleanup_actions.append((
                "temporary_store_cleanup",
                lambda: shutil.rmtree(tmp, ignore_errors=False),
            ))
        primary = sys.exc_info()[1]
        cleanup_failures: list[dict[str, str]] = []
        run_cleanup_actions(
            cleanup_actions,
            primary_exception=primary,
            evidence_sink=cleanup_failures,
        )
        if (
            cleanup_failures
            and on_checkpoint is not None
            and isinstance(primary, Exception)
        ):
            # The original provider/indexing failure is already represented by
            # its bounded durable row. A failed item-owner teardown is a run-
            # level integrity failure and must still prevent publication.
            cleanup_error = BenchmarkCleanupError(
                "MSC item cleanup failed after a recorded item failure"
            )
            cleanup_error.cleanup_errors = tuple(
                dict(item) for item in cleanup_failures
            )
            raise cleanup_error from primary


def run_recurrence_dump(ex: dict, args) -> tuple[list[dict], dict]:
    tmp = Path(tempfile.mkdtemp(prefix="msc_"))
    adapter = MSCAdapter(tmp / "hymem.sqlite", api_key=args.api_key, sim=args.sim,
                         hymem_model=args.hymem_model, hymem_base_url=args.hymem_base_url,
                         hymem_thinking=args.hymem_thinking,
                         rules_extraction=args.rules_extraction)
    try:
        adapter.open()
        indexing = prepare_indexing(
            adapter, ex, args, scope_id=f"msc:{ex['id']}"
        )
        markers = adapter.dump_markers(ex)
        pipeline_identity = {
            "hymem_model": args.hymem_model,
            "hymem_base_url": args.hymem_base_url,
            "hymem_thinking": getattr(adapter.pipeline_llm, "thinking_mode", None),
            "hymem_extra_body": copy.deepcopy(getattr(
                adapter.pipeline_llm, "effective_extra_body", {}
            )),
        }
        stamped = []
        for marker in markers:
            stamped.append({
                **marker,
                **pipeline_identity,
                "indexing_scope_id": indexing["scope_id"],
                "indexing_ref": indexing["scope_id"],
            })
        return stamped, indexing
    finally:
        cleanup_actions = [("adapter_close", adapter.close)]
        if not args.keep_db:
            import shutil
            cleanup_actions.append((
                "temporary_store_cleanup",
                lambda: shutil.rmtree(tmp, ignore_errors=False),
            ))
        run_cleanup_actions(
            cleanup_actions, primary_exception=sys.exc_info()[1],
        )


# ── reporting ───────────────────────────────────────────────────────────────

def _print_recall_report(results: list[dict]) -> None:
    from longmemeval_adapter import compute_scores, judge_error_note
    # UNSCORED rows (the judge errored — D3) are dropped ONCE, here, so the
    # distance table and the miss decomposition below cannot silently count an
    # outage as a reader failure. MSC has no --diag-only branch, so every
    # correct=None reaching this function is a judge error.
    print(f"\n  {judge_error_note(results)}")
    results = [r for r in results if r.get("correct") is not None]
    scores = compute_scores(results)
    ov = scores["OVERALL"]
    print(f"\n=== MSC recall — n={ov['count']} ===")
    print(f"  overall accuracy: {ov['accuracy']*100:.1f}%\n")
    # E1: accuracy by how many sessions back the gold fact was stated, with the
    # gold-surface diagnostics alongside (lexical tau=0.6 — signal, not truth):
    # in-ctx = gold visible to the reader; in-pool = retrievable pre-truncation.
    from collections import defaultdict
    by_dist: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_dist[r.get("gold_distance", -1)].append(r)
    print("  ── E1: recall vs session distance (+ gold-surface diagnostics) ──")
    print(f"  {'distance':>9} {'acc':>7} {'in-ctx':>7} {'in-pool':>8} {'n':>5}")
    for d in sorted(by_dist):
        rs = by_dist[d]
        acc = sum(r["correct"] for r in rs) / len(rs)
        ctx = sum(r.get("gold_in_context", False) for r in rs) / len(rs)
        pool = sum(r.get("gold_in_pool", False) for r in rs) / len(rs)
        label = "unknown" if d < 0 else f"{d} back"
        print(f"  {label:>9} {acc*100:>6.1f}% {ctx*100:>6.1f}% {pool*100:>7.1f}% {len(rs):>5}")

    # Miss decomposition: where do the failures actually sit? This is the Step-0
    # split that decides whether the next lever is retrieval, ranking, or the
    # reader — the same discipline as the LME reader-parity P0.
    misses = [r for r in results if not r["correct"]]
    if misses:
        retrieval = sum(not r.get("gold_in_pool", False) for r in misses)
        ranking = sum(r.get("gold_in_pool", False)
                      and not r.get("gold_in_context", False) for r in misses)
        synthesis = sum(r.get("gold_in_context", False) for r in misses)
        n = len(misses)
        print(f"\n  ── miss decomposition ({n} misses; lexical tau=0.6) ──")
        print(f"  retrieval loss   (gold in neither pool nor ctx): {retrieval:>3}  ({retrieval/n*100:.0f}%)")
        print(f"  ranking/cut loss (gold in pool, not in ctx):     {ranking:>3}  ({ranking/n*100:.0f}%)")
        print(f"  synthesis/judge  (gold IN ctx, still wrong):     {synthesis:>3}  ({synthesis/n*100:.0f}%)")
    n_prof = [r.get("n_profile", 0) for r in results]
    if n_prof:
        print(f"\n  profile tier: {sum(n_prof)/len(n_prof):.1f} entries/question avg "
              f"({sum(1 for p in n_prof if p == 0)} questions saw zero)")


def _print_recurrence_summary(markers: list[dict], out_path: str | None) -> None:
    from collections import Counter
    from hymem.rules import is_rule_eligible_kind
    kinds = Counter(m["kind"] for m in markers)
    eligible = sum(is_rule_eligible_kind(m["kind"]) for m in markers)
    rules = sum(m["is_rule"] for m in markers)
    sessions = {m["session_id"] for m in markers}
    print(f"\n=== MSC recurrence dump — {len(markers)} markers ===")
    print(f"  kinds: {dict(kinds)}")
    print(f"  rule-eligible (rejection/style/correction): {eligible}  "
          f"[preference→profile, not rules: {len(markers)-eligible}]")
    print(f"  is_rule (matches an annotated persona fact): {rules}")
    print(f"  distinct sessions represented: {len(sessions)}")
    if out_path:
        Path(out_path).write_text(
            json.dumps(sanitize_for_artifact(markers), indent=2),
            encoding="utf-8",
        )
        print(f"\n  markers → {out_path}")
        print("  next: python benchmarks/rule_extraction_experiment.py \\")
        print(f"           --labels {out_path} --answer-model <tagger> --policy-from-canonical")
    else:
        print("\n  (pass --out markers_msc.json to feed rule_extraction_experiment.py)")
    # Honest read: if rule-eligible ≈ 0, MSC exercises the PROFILE tier, not rules
    # — which is itself the finding (MSC content is preference/fact shaped).
    if markers and eligible == 0:
        print("\n  NOTE: 0 rule-eligible markers — MSC content is preference/fact shaped,")
        print("        so it drives the profile tier, not the rules recurrence signal.")


# ── main ────────────────────────────────────────────────────────────────────

def _build_llm(model, base_url, api_key, extra_body):
    """Build a raw benchmark client with LME's endpoint-safe body defaults."""
    from longmemeval_adapter import LLMClient
    return LLMClient(model=model, api_key=api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
                     base_url=base_url, extra_body=extra_body)


def parse_extra_body_arg(raw: str | None, role: str) -> dict | None:
    """Parse an optional CLI body while preserving absent versus explicit ``{}``."""

    if raw is None:
        return None
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--{role}-extra-body is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"--{role}-extra-body must be a JSON object")
    return body


def _effective_pipeline_body(args) -> dict[str, Any]:
    """Mirror the memory client's pure DeepSeek thinking-body decision."""

    from urllib.parse import urlsplit

    mode = str(args.hymem_thinking).strip().lower()
    if mode not in {"auto", "disabled", "off", "enabled"}:
        raise BenchmarkIntegrityError("memory-pipeline thinking mode is invalid")
    host = (urlsplit(args.hymem_base_url).hostname or "").casefold()
    sends_disabled = mode == "disabled" or (
        mode == "auto"
        and ("deepseek" in host or "deepseek" in args.hymem_model.casefold())
    )
    return {"thinking": {"type": "disabled"}} if sends_disabled else {}


def _effective_hymem_config(args):
    """Return the adapter's exact config object and public serialized identity."""
    from hymem import HyMemConfig

    overrides: dict[str, Any] = {
        **_MSC_APERTURE,
        # The historical MSC posture is pinned independently of the shipped
        # default; silently inheriting a future default would invalidate an A/B.
        "aggregation_nodes_enabled": False,
    }
    if args.rules_extraction is not None:
        overrides["rules_extraction_enabled"] = args.rules_extraction
    if args.graph_multihop:
        overrides["graph_multihop_enabled"] = True
    if args.facts is not None:
        overrides["facts_enabled"] = args.facts
    if args.facts_extraction is not None:
        overrides["facts_extraction_enabled"] = args.facts_extraction
    cfg = HyMemConfig(root=Path("/benchmark-identity"), **overrides)
    effective_cfg = effective_hymem_config_identity(cfg)
    effective_cfg["content_redaction_enabled"] = effective_cfg.pop(
        "redact_secrets"
    )
    return cfg, effective_cfg


def _strict_identity(args) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the exact scored identity without constructing paid clients."""

    from longmemeval_adapter import _provider_for_url, resolve_embedding_identity

    cfg, effective_cfg = _effective_hymem_config(args)

    embedding = resolve_embedding_identity(SimpleNamespace(
        embeddings=bool(args.embeddings and not args.sim),
        embedding_base_url=None,
        embedding_model=None,
        embedding_dim=None,
    ))
    if args.sim:
        models = {
            "reader": {
                "configured": False,
                "client_class": None,
                "provider": "none",
                "model": None,
                "base_url": None,
            },
            "judge": {
                "configured": False,
                "client_class": None,
                "provider": "none",
                "model": None,
                "base_url": None,
            },
            "memory_pipeline": {
                "configured": True,
                "client_class": "hymem.extraction.llm.StubLLMClient",
                "provider": "local_stub",
                "model": None,
                "base_url": None,
                "response_policy": "constant-empty-json-list-v1",
            },
            "embedding": embedding,
        }
    else:
        models = {
            "reader": {
                "configured": True,
                "client_class": "longmemeval_adapter.LLMClient",
                "provider": _provider_for_url(args.answer_base_url),
                "model": args.answer_model,
                "base_url": args.answer_base_url,
                "temperature": 0.0,
                "max_tokens": 1024,
                "extra_body": copy.deepcopy(args.answer_extra_body_obj),
            },
            "judge": {
                "configured": True,
                "client_class": "longmemeval_adapter.LLMClient",
                "provider": _provider_for_url(_DEEPSEEK_BASE_URL),
                "model": args.judge_model,
                "base_url": _DEEPSEEK_BASE_URL,
                "temperature": 0.0,
                "max_tokens": 10,
                "extra_body": copy.deepcopy(args.judge_extra_body_obj),
                "protocol": "longmemeval-local-judge",
            },
            "memory_pipeline": {
                "configured": True,
                "client_class": (
                    "hymem.contrib.openai_client.OpenAICompatibleClient"
                ),
                "provider": _provider_for_url(args.hymem_base_url),
                "model": args.hymem_model,
                "base_url": args.hymem_base_url,
                "thinking_mode": args.hymem_thinking,
                "effective_extra_body": _effective_pipeline_body(args),
            },
            "embedding": embedding,
        }
    config = {
        "probe_mode": "recall",
        "sample": args.sample,
        "sample_strategy": (
            "seeded-shuffle-prefix-v1" if args.sample else "seeded-shuffle-all-v1"
        ),
        "seed": args.seed,
        "workers": args.workers,
        "top_k": args.top_k,
        "pipeline_search_multiplier": 3,
        "start_role": args.start_role,
        "session_gap_days": float(args.session_gap_days),
        "synthetic_base_date": "2023-01-01",
        "embeddings": bool(args.embeddings),
        "rules_extraction": args.rules_extraction,
        "facts": args.facts,
        "facts_extraction": args.facts_extraction,
        "graph_multihop": bool(args.graph_multihop),
        "no_dream": bool(args.no_dream),
        "dream_per_session": bool(args.dream_per_session),
        "indexing_max_cycles": args.indexing_max_cycles,
        "indexing_timeout_s": float(args.indexing_timeout_s),
        "dump_context": bool(args.dump_context),
        "sim": bool(args.sim),
        "effective_hymem_config": effective_cfg,
        "extraction_canary": extraction_canary_policy(
            prompt_version=cfg.prompt_version
        ),
        "label_free_answer_path": True,
        "scored_run": not args.sim,
        "exploratory_label_steering": False,
        "exploratory_non_comparable": bool(
            args.sample or args.sim or args.no_dream
        ),
    }
    validate_extraction_canary_config_binding(
        config["extraction_canary"], effective_cfg
    )
    return config, models


def msc_code_hash(
    *,
    adapter_path: Path | None = None,
    strictness_path: Path | None = None,
    archive_evidence_path: Path | None = None,
    lme_adapter_path: Path | None = None,
    lme_protocol_path: Path | None = None,
    extraction_canary_path: Path | None = None,
    store_attestation_path: Path | None = None,
    registry_path: Path | None = None,
    hymem_path: Path | None = None,
    root: Path | None = None,
) -> str:
    """Hash exact executable dependencies reached by strict MSC recall."""

    root_path = Path(root or _repo_root).resolve()
    benchmark_dir = Path(__file__).resolve().parent
    adapter = Path(adapter_path or __file__)
    lme_adapter = Path(
        lme_adapter_path or benchmark_dir / "longmemeval_adapter.py"
    )
    lme_symbols = python_file_imported_symbols(
        adapter,
        module_names=("benchmarks.longmemeval_adapter", "longmemeval_adapter"),
    )
    dependency_slices: list[PythonSourceSlice] = []
    if lme_symbols:
        lme_slice = PythonSourceSlice(lme_adapter, lme_symbols)
        dependency_slices.append(lme_slice)
        protocol_symbols = python_slice_imported_symbols(
            lme_slice,
            module_names=("benchmarks.lme_protocol", "lme_protocol"),
        )
        if protocol_symbols:
            dependency_slices.append(PythonSourceSlice(
                Path(lme_protocol_path or benchmark_dir / "lme_protocol.py"),
                protocol_symbols,
            ))
    for path, modules in (
        (
            Path(extraction_canary_path or benchmark_dir / "extraction_canary.py"),
            ("benchmarks.extraction_canary", "extraction_canary"),
        ),
        (
            Path(store_attestation_path or benchmark_dir / "store_attestation.py"),
            ("benchmarks.store_attestation", "store_attestation"),
        ),
        (
            Path(registry_path or benchmark_dir / "msc_registry.py"),
            ("benchmarks.msc_registry", "msc_registry"),
        ),
    ):
        symbols = python_file_imported_symbols(adapter, module_names=modules)
        if symbols:
            dependency_slices.append(PythonSourceSlice(path, symbols))
    strictness = Path(strictness_path or benchmark_dir / "strictness.py")
    strictness_modules = ("benchmarks.strictness", "strictness")
    strictness_symbols = set(python_file_imported_symbols(
        adapter, module_names=strictness_modules
    ))
    for source_slice in dependency_slices:
        strictness_symbols.update(python_slice_imported_symbols(
            source_slice, module_names=strictness_modules
        ))
    if not strictness_symbols:
        raise BenchmarkIntegrityError("MSC code identity lacks strictness imports")
    dependency_slices.append(PythonSourceSlice(
        strictness, tuple(strictness_symbols)
    ))
    archive_symbols: set[str] = set()
    for source_slice in dependency_slices:
        archive_symbols.update(python_slice_imported_symbols(
            source_slice, module_names=("benchmarks.archive_evidence", "archive_evidence"),
        ))
    if archive_symbols:
        dependency_slices.append(PythonSourceSlice(
            Path(archive_evidence_path or benchmark_dir / "archive_evidence.py"),
            tuple(archive_symbols),
        ))
    dependency_sources: list[Path | PythonSourceSlice] = [
        adapter, *dependency_slices,
    ]
    inputs: list[Path | PythonSourceSlice] = [
        adapter,
        *dependency_slices,
        *benchmark_hymem_source_paths(
            Path(hymem_path or root_path / "hymem"),
            root=root_path,
            dependency_sources=dependency_sources,
        ),
    ]
    return code_hash(inputs, root=root_path)


def _select_examples_by_id(
    examples: list[dict], selected_ids: tuple[str, ...],
) -> list[dict]:
    """Project normalized examples onto one exact protocol sequence."""

    by_id: dict[str, dict] = {}
    for example in examples:
        item_id = example.get("id")
        if not isinstance(item_id, str) or item_id in by_id:
            raise BenchmarkIntegrityError("MSC example identity is ambiguous")
        by_id[item_id] = example
    try:
        result = [by_id[item_id] for item_id in selected_ids]
    except KeyError as exc:
        raise BenchmarkIntegrityError(
            "selected MSC id is absent from the normalized dataset"
        ) from exc
    if tuple(example["id"] for example in result) != selected_ids:
        raise BenchmarkIntegrityError("selected MSC id order drifted")
    return result


def _strict_recall_scores(
    rows: list[dict], *, scored: bool,
) -> dict[str, dict[str, int | float]] | None:
    """Small total scorer safe to run before authoritative publication."""

    if not scored:
        return None
    verdicts = []
    for index, row in enumerate(rows):
        verdict = row.get("correct")
        if not isinstance(verdict, bool):
            raise BenchmarkIntegrityError(
                f"reconciled MSC row {index} has no boolean verdict"
            )
        verdicts.append(verdict)
    accuracy = sum(verdicts) / len(verdicts) if verdicts else 0.0
    return {
        "recall": {"accuracy": accuracy, "count": len(verdicts)},
        "OVERALL": {"accuracy": accuracy, "count": len(verdicts)},
    }


_CANARY_USAGE_EVIDENCE_FIELDS = (
    "calls", "calls_available", "request_attempts",
    "request_attempts_available", "successful_responses",
    "successful_responses_available", "prompt_tokens", "completion_tokens",
    "total_tokens", "latency_s", "cost_usd", "token_usage_available",
    "latency_available", "cost_available",
)


def _invalid_canary_evidence(value: object) -> dict[str, Any]:
    """Project an untrusted canary return onto one small durable sentinel."""

    report = value if isinstance(value, dict) else {}
    status = report.get("status")
    if status not in {
        "passed", "failed", "pending", "skipped_non_comparable",
        "not_run_no_pending",
    }:
        status = "unknown"

    def counter(name: str) -> int | None:
        raw = report.get(name)
        return raw if type(raw) is int and raw >= 0 else None

    raw_usage = report.get("usage")
    projected_usage = None
    if isinstance(raw_usage, dict):
        projected_usage = {}
        for field in _CANARY_USAGE_EVIDENCE_FIELDS:
            raw = raw_usage.get(field)
            if isinstance(raw, bool) or raw is None:
                projected_usage[field] = raw
            elif (
                isinstance(raw, (int, float))
                and math.isfinite(float(raw))
                and raw >= 0
            ):
                projected_usage[field] = raw
            else:
                projected_usage[field] = None
    closed = report.get("client_closed")
    return {
        "status": "invalid_report",
        "failure_reason": "internal_validation_failure",
        "reported_status": status,
        "completion_calls": counter("completion_calls"),
        "provider_attempts": counter("provider_attempts"),
        "client_closed": closed if isinstance(closed, bool) else None,
        "usage": projected_usage,
    }


def _run_main(owned_clients: OwnedResourceScope) -> None:
    ap = argparse.ArgumentParser(description="HyMem MSC benchmark adapter.")
    ap.add_argument("--data", default=None, help="MSC JSON/JSONL (MemGPT/MSC-Self-Instruct shape)")
    ap.add_argument("--probe-mode", choices=["recall", "recurrence"], default="recall")
    ap.add_argument("--sample", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=10,
                    help="base K; the pipeline searches top_k*3 like the LME driver")
    ap.add_argument("--start-role", choices=["user", "assistant"], default="user")
    ap.add_argument("--session-gap-days", type=float, default=1.0,
                    help="fallback inter-session spacing when MSC gaps are missing")
    ap.add_argument("--answer-model", default=_ANSWER_MODEL)
    ap.add_argument("--answer-base-url", default=_DEEPSEEK_BASE_URL)
    ap.add_argument("--answer-api-key", default=None)
    ap.add_argument("--answer-extra-body", default=None, metavar="JSON",
                    help="optional provider body; omitted DeepSeek v4-flash "
                         "requests disable thinking automatically")
    ap.add_argument("--judge-model", default=_JUDGE_MODEL)
    ap.add_argument(
        "--judge-api-key", default=None,
        help="judge-specific API key (never inherited from --answer-api-key)",
    )
    ap.add_argument("--judge-extra-body", default=None, metavar="JSON",
                    help="optional provider body; omitted DeepSeek v4-flash "
                         "requests disable thinking automatically")
    ap.add_argument("--hymem-model", default=_HYMEM_MODEL, help="HyMem's dream LLM")
    ap.add_argument("--hymem-base-url", default=_DEEPSEEK_BASE_URL)
    ap.add_argument("--hymem-thinking", choices=("auto", "disabled", "off", "enabled"),
                    default="auto", help="memory-pipeline thinking policy")
    ap.add_argument("--api-key", default="", help="HyMem dream LLM key")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--rules-extraction", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--facts", action=argparse.BooleanOptionalAction, default=None,
                    help="E1 narrative-facts READ side (cfg.facts_enabled). None = "
                         "config default (ON); --no-facts is the paired control arm. "
                         "Check n_facts in the per-question rows before reading the score.")
    ap.add_argument("--facts-extraction", action=argparse.BooleanOptionalAction, default=None,
                    help="E1 WRITE side (cfg.facts_extraction_enabled). None = config "
                         "default (ON). Changes what is STORED, so it only differs on a "
                         "rebuild — not a read-side A/B knob.")
    ap.add_argument("--graph-multihop", action="store_true")
    ap.add_argument(
        "--no-dream", action="store_true",
        help="skip indexing (explicit non-comparable message-only development path)",
    )
    ap.add_argument(
        "--indexing-max-cycles", type=int,
        default=DEFAULT_INDEXING_MAX_CYCLES,
        help="per-wave dream-cycle safety cap before failing closed (default 100)",
    )
    ap.add_argument(
        "--indexing-timeout-s", type=float,
        default=DEFAULT_INDEXING_TIMEOUT_S,
        help="per-wave wall-clock convergence bound in seconds (default 3600)",
    )
    ap.add_argument("--dream-per-session", action="store_true",
                    help="fully converge after EACH session (live-store posture: "
                         "profile/episode evidence accumulates across waves; expensive)")
    ap.add_argument("--keep-db", action="store_true")
    ap.add_argument(
        "--results-dir", default=None,
        help=("recall: strict immutable archive/checkpoint directory (default: "
              "the --out directory when supplied, otherwise ./msc_results)"),
    )
    ap.add_argument(
        "--out", default=None,
        help=("recall: mutable legacy bare-row sidecar written only after the "
              "authoritative strict archive; recurrence: the marker dump"),
    )
    ap.add_argument("--dump-context", action="store_true",
                    help="recall: include the exact rendered answer context in "
                         "each result (synthesis-miss audits)")
    ap.add_argument("--sim", action="store_true", help="offline: StubLLM, no API")
    ap.add_argument("--json", action="store_true")
    add_strict_run_arguments(ap)
    args = ap.parse_args()

    try:
        _indexing_limits(args)
        if isinstance(args.sample, bool) or args.sample < 0:
            raise BenchmarkIntegrityError("sample must be a non-negative integer")
        if isinstance(args.workers, bool) or args.workers <= 0:
            raise BenchmarkIntegrityError("workers must be a positive integer")
        if isinstance(args.top_k, bool) or args.top_k <= 0:
            raise BenchmarkIntegrityError("top-k must be a positive integer")
        if (
            isinstance(args.session_gap_days, bool)
            or not isinstance(args.session_gap_days, (int, float))
            or not math.isfinite(float(args.session_gap_days))
            or args.session_gap_days <= 0
        ):
            raise BenchmarkIntegrityError(
                "session-gap-days must be positive and finite"
            )
        args.answer_base_url = validate_http_endpoint(
            args.answer_base_url, label="reader"
        ).url
        args.hymem_base_url = validate_http_endpoint(
            args.hymem_base_url, label="memory pipeline"
        ).url
    except BenchmarkIntegrityError as exc:
        ap.error(str(exc))
    except ValueError as exc:
        ap.error(str(exc))

    strict_controls = bool(
        args.checkpoint or args.resume_from or args.retry_failures
        or args.calibration_receipt or args.freeze_calibration
        or args.protocol_split != "full" or args.results_dir
    )
    if args.sim and args.embeddings:
        ap.error("--sim does not construct an embedding client; drop --embeddings")
    if args.no_dream and args.dream_per_session:
        ap.error("--dream-per-session cannot be combined with --no-dream")
    if args.probe_mode == "recurrence" and strict_controls:
        ap.error(
            "checkpoint/calibration/protocol/results-dir flags apply only to "
            "strict MSC recall, not the unscored recurrence marker dump"
        )
    if args.retry_failures and not args.resume_from:
        ap.error("--retry-failures requires --resume-from")
    if args.calibration_receipt and args.protocol_split == "full":
        ap.error(
            "--calibration-receipt requires --protocol-split dev or holdout"
        )
    if args.freeze_calibration and (args.checkpoint or args.resume_from):
        ap.error("--freeze-calibration cannot create or resume a checkpoint")
    if args.freeze_calibration and args.calibration_receipt:
        ap.error(
            "--freeze-calibration cannot consume --calibration-receipt"
        )
    if args.freeze_calibration and args.protocol_split != "full":
        ap.error(
            "--freeze-calibration creates both splits; leave --protocol-split full"
        )
    if (args.freeze_calibration or args.protocol_split != "full") and args.sample:
        ap.error("--freeze-calibration and dev/holdout runs require --sample 0")

    if not args.sim:
        active_models = [("MSC memory pipeline", args.hymem_model)]
        if args.probe_mode == "recall":
            active_models.extend((
                ("MSC reader", args.answer_model),
                ("MSC judge", args.judge_model),
            ))
        try:
            for role, active_model in active_models:
                require_active_model(active_model, role=role)
        except DeprecatedModelAliasError as exc:
            ap.error(str(exc))

    extraction_cfg, extraction_effective_cfg = _effective_hymem_config(args)
    extraction_prompt_version = extraction_cfg.prompt_version
    runtime_extraction_binding = validate_extraction_canary_config_binding(
        extraction_canary_policy(
            prompt_version=extraction_prompt_version
        ),
        extraction_effective_cfg,
    )

    def extraction_preflight() -> dict:
        """Run once after mode-specific validation, before any store opens."""
        if args.sim:
            report = skipped_extraction_canary(
                "simulation", prompt_version=extraction_prompt_version
            )
        elif args.no_dream:
            report = skipped_extraction_canary(
                "no_dream", prompt_version=extraction_prompt_version
            )
        else:
            report = run_configured_extraction_canary(
                api_key=args.api_key,
                base_url=args.hymem_base_url,
                model=args.hymem_model,
                thinking=args.hymem_thinking,
                prompt_version=extraction_prompt_version,
            )
        validate_extraction_canary_report(
            report,
            expected_mode=extraction_canary_mode,
            expected_client=extraction_canary_expected_client,
            require_client_closed=extraction_canary_mode == "required",
            expected_prompt_version=extraction_prompt_version,
        )
        print_extraction_canary(report)
        return report

    if args.probe_mode == "recurrence":
        examples = load_msc_data(
            args.data, args.sample, args.seed,
            start_role=args.start_role, gap_days=args.session_gap_days,
        )
        if not examples:
            print("No MSC examples loaded.")
            sys.exit(1)
        print(f"Loaded {len(examples)} MSC examples "
              f"(sessions/ex: {sum(e['n_sessions'] for e in examples)/len(examples):.1f} avg)"
              f"{'  [SIM]' if args.sim else ''}", flush=True)
        extraction_canary_mode = (
            "simulation" if args.sim else
            "no_dream" if args.no_dream else "required"
        )
        extraction_canary_expected_client = (
            extraction_canary_client_policy(
                base_url=args.hymem_base_url,
                model=args.hymem_model,
                thinking=args.hymem_thinking,
            ) if extraction_canary_mode == "required" else None
        )
        extraction_canary_report = extraction_preflight()
        all_markers: list[dict] = []
        indexing_receipts: list[dict] = []
        for k, e in enumerate(examples, 1):
            markers, indexing = run_or_record_indexing_failure(
                lambda e=e: run_recurrence_dump(e, args),
                benchmark="msc_recurrence",
                out_path=args.out,
                extraction_canary=extraction_canary_report,
            )
            all_markers.extend(markers)
            indexing_receipts.append(indexing)
            print(f"  [{k}/{len(examples)}]", end="\r", flush=True)
        owned_clients.close()
        print(" " * 30, end="\r")
        _print_recurrence_summary(all_markers, args.out)
        if args.out:
            validate_extraction_canary_report(
                extraction_canary_report,
                expected_mode=extraction_canary_mode,
                expected_client=extraction_canary_expected_client,
                require_client_closed=extraction_canary_mode == "required",
                expected_prompt_version=extraction_prompt_version,
            )
            canary_path = Path(f"{args.out}.extraction-canary.json")
            canary_path.write_text(
                json.dumps(
                    sanitize_for_artifact(extraction_canary_report), indent=2
                ),
                encoding="utf-8",
            )
            print(f"  extraction-canary provenance → {canary_path}")
            indexing_path = Path(f"{args.out}.indexing.json")
            indexing_path.write_text(
                json.dumps(sanitize_for_artifact({
                        "protocol": INDEXING_PROVENANCE_VERSION,
                        "extraction_canary": extraction_canary_report,
                        "indexing_scopes": indexing_receipts,
                    }), indent=2),
                encoding="utf-8",
            )
            print(f"  indexing provenance → {indexing_path}")
        return

    # From here onward MSC recall always uses the strict lifecycle. Resolve
    # score-affecting request bodies before hashing the identity, but do not
    # construct a provider transport.
    try:
        from longmemeval_adapter import resolve_model_extra_body

        parsed_answer_body = parse_extra_body_arg(
            args.answer_extra_body, "answer"
        )
        parsed_judge_body = parse_extra_body_arg(args.judge_extra_body, "judge")
        args.answer_extra_body_obj, _ = resolve_model_extra_body(
            args.answer_model, args.answer_base_url, parsed_answer_body
        )
        args.judge_extra_body_obj, _ = resolve_model_extra_body(
            args.judge_model, _DEEPSEEK_BASE_URL, parsed_judge_body
        )
    except (BenchmarkIntegrityError, ValueError) as exc:
        ap.error(str(exc))

    source_examples = load_msc_data(
        args.data, 0, args.seed,
        start_role=args.start_role, gap_days=args.session_gap_days,
    )
    eligible = [
        example for example in source_examples
        if example.get("question") and example.get("answer")
    ]
    if not eligible:
        print("No examples carry a self_instruct QA pair — recall mode needs them.")
        sys.exit(1)
    source_ids = validate_ids(
        (example["id"] for example in eligible), label="MSC eligible dataset"
    )
    if args.sample > len(source_ids):
        ap.error(
            f"--sample {args.sample} exceeds the eligible dataset size "
            f"({len(source_ids)})"
        )
    examples = eligible[:args.sample] if args.sample else list(eligible)
    selected_source_ids = validate_ids(
        (example["id"] for example in examples), label="MSC selected dataset"
    )
    data_sha = file_hash(args.data) if args.data else content_hash(_SIM_FIXTURE)
    strict_config, strict_models = _strict_identity(args)

    if args.freeze_calibration:
        receipt = freeze_calibration(
            args.freeze_calibration,
            benchmark="MSC",
            dataset_hash=data_sha,
            ids=source_ids,
            config=strict_config,
            models=strict_models,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
        )
        print(f"Frozen MSC calibration: dev={len(receipt['dev_ids'])}, "
              f"holdout={len(receipt['holdout_ids'])}")
        return

    calibration = None
    if args.calibration_receipt:
        calibration = load_calibration(
            args.calibration_receipt,
            benchmark="MSC",
            dataset_hash=data_sha,
            config=strict_config,
            models=strict_models,
            ids=source_ids,
        )
    selected_ids = select_protocol_ids(
        selected_source_ids, split=args.protocol_split, receipt=calibration
    )
    examples = _select_examples_by_id(examples, selected_ids)
    manifest = build_manifest(
        benchmark="MSC",
        code_sha256=msc_code_hash(),
        data_sha256=data_sha,
        config=strict_config,
        models=strict_models,
        seed=args.seed,
        expected_ids=selected_ids,
        protocol_split=args.protocol_split,
        calibration=calibration,
    )
    manifest_extraction_binding = validate_extraction_canary_config_binding(
        manifest["config"].get("extraction_canary"),
        manifest["config"].get("effective_hymem_config"),
    )
    if manifest_extraction_binding != runtime_extraction_binding:
        raise BenchmarkIntegrityError(
            "MSC manifest extraction contract differs from runtime config"
        )
    extraction_prompt_version = manifest_extraction_binding["prompt_version"]

    results_dir = Path(
        args.results_dir
        or (Path(args.out).parent if args.out else Path.cwd() / "msc_results")
    )
    latest_path = results_dir / "msc-latest.json"
    if args.out:
        out_resolved = Path(args.out).resolve(strict=False)
        if out_resolved == results_dir.resolve(strict=False):
            ap.error("--out must not alias --results-dir")
        if out_resolved == latest_path.resolve(strict=False):
            ap.error("--out must not overwrite the strict latest pointer")
        for label, raw in (
            ("--checkpoint", args.checkpoint),
            ("--resume-from", args.resume_from),
        ):
            if raw and Path(raw).resolve(strict=False) == out_resolved:
                ap.error(f"--out must not overwrite {label}")
    for label, raw in (
        ("--checkpoint", args.checkpoint),
        ("--resume-from", args.resume_from),
    ):
        if raw and Path(raw).resolve(strict=False) == latest_path.resolve(strict=False):
            ap.error(f"{label} must not alias the strict latest pointer")
        if raw and Path(raw).resolve(strict=False) == results_dir.resolve(strict=False):
            ap.error(f"{label} must not alias --results-dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint_path, is_resume = resolve_checkpoint_path(
            checkpoint=args.checkpoint,
            resume_from=args.resume_from,
            base_dir=results_dir,
            benchmark="msc",
            run_id=manifest["run_id"],
        )
    except BenchmarkIntegrityError as exc:
        ap.error(str(exc))
    if checkpoint_path.resolve(strict=False) == latest_path.resolve(strict=False):
        ap.error("checkpoint path must not alias the strict latest pointer")
    if args.out and checkpoint_path.resolve(strict=False) == Path(args.out).resolve(
        strict=False
    ):
        ap.error("--out must not overwrite the checkpoint")

    ledger: AtomicCheckpoint | None = None
    try:
        # This is the first run-state owner. On resume it verifies the full
        # immutable manifest and selected-id order before canary/client/store work.
        ledger = AtomicCheckpoint(
            checkpoint_path,
            manifest=manifest,
            expected_ids=selected_ids,
            resume=is_resume,
            retry_failures=args.retry_failures,
            scored=not args.sim,
        )
        pending = set(ledger.pending_ids)
        work_examples = [
            example for example in examples if example["id"] in pending
        ]
        print(f"Loaded {len(examples)} MSC examples "
              f"(sessions/ex: {sum(e['n_sessions'] for e in examples)/len(examples):.1f} avg)"
              f"{'  [SIM]' if args.sim else ''}", flush=True)
        print(f"  Strict checkpoint: {checkpoint_path} "
              f"({len(pending)} pending / {len(selected_ids)} expected)")

        start_time = time.time()
        segment_id = (
            f"process-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{os.getpid()}"
        )
        answer_llm = judge_llm = None
        attempted = 0
        attempted_ids: set[str] = set()
        failed_attempt_ids: set[str] = set()
        runtime_lock = threading.RLock()
        pipeline_by_scope: dict[str, dict[str, Any]] = {}
        embedding_by_scope: dict[str, dict[str, Any]] = {}
        indexing_by_scope: dict[str, dict[str, Any]] = {}
        indexing_failures: dict[str, dict[str, Any]] = {}
        extraction_canary_report: dict[str, Any] = (
            skipped_extraction_canary(
                "no_pending_work", prompt_version=extraction_prompt_version
            )
            if not pending else
            skipped_extraction_canary(
                "simulation", prompt_version=extraction_prompt_version
            )
            if args.sim else
            skipped_extraction_canary(
                "no_dream", prompt_version=extraction_prompt_version
            )
            if args.no_dream else
            {
                **extraction_canary_policy(
                    prompt_version=extraction_prompt_version
                ),
                "status": "pending",
            }
        )

        def _zero_embedding_usage() -> dict[str, Any]:
            identity = manifest["models"]["embedding"]
            return {
                "configured": identity["configured"],
                "backend": identity["backend"],
                "quality": identity["quality"],
                "network_free": identity["network_free"],
                "model": identity["vector_space_key"],
                "dimension": identity["dimension"],
                "identity_available": True,
                "identity_exact": identity["identity_exact"],
                "reuse_scope": identity["reuse_scope"],
                "calls": 0, "calls_available": True,
                "request_attempts": 0,
                "request_attempts_available": True,
                "successful_responses": 0,
                "successful_responses_available": True,
                "input_count": 0, "input_count_available": True,
                "input_characters": 0,
                "input_characters_available": True,
                "prompt_tokens": None, "total_tokens": None,
                "provider_token_usage_available": False,
                "latency_s": 0.0, "latency_available": True,
                "cost_usd": None, "cost_available": False,
            }

        def _segment(status: str) -> dict[str, Any]:
            with runtime_lock:
                if embedding_by_scope:
                    embedding_usage = aggregate_embedding_usage_snapshots(
                        embedding_by_scope.values()
                    )
                elif pending:
                    embedding_usage = embedding_usage_snapshot(
                        None, configured=bool(args.embeddings)
                    )
                else:
                    embedding_usage = _zero_embedding_usage()
                zero_llm = _known_zero_pipeline_usage()
                return {
                    "segment_id": segment_id,
                    "status": status,
                    "elapsed_s": time.time() - start_time,
                    "attempted_attempts": attempted,
                    "model_identities": manifest["models"],
                    "reader_usage": (
                        usage_snapshot(answer_llm)
                        if answer_llm is not None else dict(zero_llm)
                    ),
                    "judge_usage": (
                        usage_snapshot(judge_llm)
                        if judge_llm is not None else dict(zero_llm)
                    ),
                    "memory_pipeline_usage": (
                        aggregate_usage_snapshots(pipeline_by_scope.values())
                        if pipeline_by_scope else dict(zero_llm)
                    ),
                    "embedding_usage": embedding_usage,
                    "indexing_runs": [
                        {"scope_id": scope, "summary": dict(summary)}
                        for scope, summary in sorted(indexing_by_scope.items())
                    ],
                    "indexing_failures": [
                        {"scope_id": scope, "summary": dict(summary)}
                        for scope, summary in sorted(indexing_failures.items())
                    ],
                    "extraction_canary": dict(extraction_canary_report),
                }

        class _CheckpointPersistenceAbort(BaseException):
            """Never reinterpret durable-checkpoint failure as item failure."""

        def _persist(
            row: dict, runtime: dict[str, Any] | None = None,
        ) -> None:
            nonlocal attempted
            safe_row = dict(row)
            question_id = safe_row.get("question_id")
            legacy_id = safe_row.get("id")
            item_id = question_id if question_id is not None else legacy_id
            if not isinstance(item_id, str) or not item_id:
                raise _CheckpointPersistenceAbort() from BenchmarkIntegrityError(
                    "MSC emitted a row without its exact question id"
                )
            if (
                question_id is not None and question_id != item_id
                or legacy_id is not None and legacy_id != item_id
            ):
                raise _CheckpointPersistenceAbort() from BenchmarkIntegrityError(
                    "MSC emitted inconsistent row identity fields"
                )
            question_type = safe_row.get("question_type")
            if question_type not in {None, "recall"}:
                raise _CheckpointPersistenceAbort() from BenchmarkIntegrityError(
                    "MSC emitted a non-recall result row"
                )
            expected_scope = f"msc:{item_id}"
            row_scope = safe_row.get("indexing_scope_id")
            if row_scope not in {None, expected_scope}:
                raise _CheckpointPersistenceAbort() from BenchmarkIntegrityError(
                    "MSC row indexing scope differs from its question id"
                )
            if not isinstance(runtime, dict):
                raise _CheckpointPersistenceAbort() from BenchmarkIntegrityError(
                    "MSC durable callback lacks question-local runtime evidence"
                )
            runtime_scope = runtime.get("scope_id")
            if runtime_scope != expected_scope:
                raise _CheckpointPersistenceAbort() from BenchmarkIntegrityError(
                    "MSC runtime scope differs from its question id"
                )
            runtime_indexing = runtime.get("indexing")
            runtime_indexing_failure = runtime.get("indexing_failure")
            if (
                isinstance(runtime_indexing, dict)
                == isinstance(runtime_indexing_failure, dict)
            ):
                raise _CheckpointPersistenceAbort() from BenchmarkIntegrityError(
                    "MSC runtime must carry exactly one indexing outcome"
                )
            safe_row["question_id"] = item_id
            safe_row["id"] = item_id
            safe_row["question_type"] = "recall"
            safe_row["indexing_scope_id"] = expected_scope
            if safe_row.get("correct") is None and not safe_row.get(
                "benchmark_failure"
            ):
                safe_row["benchmark_failure"] = (
                    "judge_or_reader_returned_no_valid_verdict"
                )
            safe_row["extraction_canary"] = dict(extraction_canary_report)
            with runtime_lock:
                if item_id in attempted_ids:
                    duplicate = BenchmarkIntegrityError(
                        "MSC emitted one question more than once in an "
                        "execution segment"
                    )
                    raise _CheckpointPersistenceAbort() from duplicate
                attempted_ids.add(item_id)
                if safe_row.get("benchmark_failure"):
                    failed_attempt_ids.add(item_id)
                if runtime is not None:
                    scope = str(runtime["scope_id"])
                    pipeline = runtime.get("memory_pipeline_usage")
                    embedding = runtime.get("embedding_usage")
                    indexing = runtime.get("indexing")
                    indexing_failure = runtime.get("indexing_failure")
                    if isinstance(pipeline, dict):
                        pipeline_by_scope[scope] = dict(pipeline)
                    if isinstance(embedding, dict):
                        embedding_by_scope[scope] = dict(embedding)
                    if isinstance(indexing, dict):
                        indexing_by_scope[scope] = dict(indexing)
                    if isinstance(indexing_failure, dict):
                        indexing_failures[scope] = dict(indexing_failure)
                attempted += 1
                try:
                    ledger.record(
                        item_id,
                        row=safe_row,
                        execution_segment=_segment("running"),
                    )
                except BaseException as exc:
                    raise _CheckpointPersistenceAbort() from exc

        def _record_returned(row: object, example: dict) -> None:
            item_id = example["id"]
            if item_id in attempted_ids:
                return
            if not isinstance(row, dict):
                raise _CheckpointPersistenceAbort() from BenchmarkIntegrityError(
                    "MSC example returned a malformed result row"
                )
            returned_id = row.get("question_id", row.get("id", item_id))
            if returned_id != item_id:
                raise _CheckpointPersistenceAbort() from BenchmarkIntegrityError(
                    "MSC result id differs from its selected dataset id"
                )
            # Production ``run_recall`` always performs its durable callback
            # before returning. A return without that callback has lost its
            # question-local pipeline/embedding/indexing evidence, so it is an
            # integrity abort and is never converted into a scored failure row.
            raise _CheckpointPersistenceAbort() from BenchmarkIntegrityError(
                "MSC result returned without its durable runtime callback"
            )

        def _record_example_failure(example: dict, exc: Exception) -> None:
            if isinstance(exc, BenchmarkCleanupError):
                raise exc
            item_id = example["id"]
            if item_id in attempted_ids:
                # run_recall records ordinary failures before re-raising. A
                # successful callback followed by an exception is a crash/
                # cleanup boundary and must abort publication, not be ignored.
                if item_id in failed_attempt_ids:
                    return
                raise exc
            scope = f"msc:{item_id}"
            if isinstance(exc, IndexingConvergenceError):
                summary = sanitize_for_artifact(
                    dict(exc.summary), _preserve_evidence_text=False
                )
            else:
                summary = _pre_scoring_indexing_failure(exc)
            runtime = {
                "scope_id": scope,
                "indexing_failure": summary,
            }
            _persist(_recall_failure_row(example, exc), runtime)

        if pending:
            ledger.update_execution_segment(segment_id, _segment("running"))
            if args.sim:
                extraction_canary_mode = "simulation"
            elif args.no_dream:
                extraction_canary_mode = "no_dream"
            else:
                extraction_canary_mode = "required"
                try:
                    returned_canary = run_configured_extraction_canary(
                        api_key=args.api_key,
                        base_url=args.hymem_base_url,
                        model=args.hymem_model,
                        thinking=args.hymem_thinking,
                        prompt_version=extraction_prompt_version,
                    )
                    # Persist a bounded projection of the returned counters and
                    # usage before treating the provider-shaped object as
                    # trustworthy. On validation success it is atomically
                    # replaced by the canonical report; on failure this exact
                    # sentinel remains as recovery evidence.
                    extraction_canary_report = _invalid_canary_evidence(
                        returned_canary
                    )
                    ledger.update_execution_segment(
                        segment_id, _segment("running")
                    )
                    validate_extraction_canary_report(
                        returned_canary,
                        expected_mode="required",
                        expected_client=extraction_canary_client_policy(
                            base_url=args.hymem_base_url,
                            model=args.hymem_model,
                            thinking=args.hymem_thinking,
                        ),
                        require_client_closed=True,
                        expected_prompt_version=extraction_prompt_version,
                    )
                    extraction_canary_report = dict(returned_canary)
                    ledger.update_execution_segment(
                        segment_id, _segment("running")
                    )
                except ExtractionCanaryError as exc:
                    extraction_canary_report = dict(exc.report)
                    ledger.update_execution_segment(
                        segment_id, _segment("running")
                    )
                    raise
            validate_extraction_canary_report(
                extraction_canary_report,
                expected_mode=extraction_canary_mode,
                expected_client=(
                    extraction_canary_client_policy(
                        base_url=args.hymem_base_url,
                        model=args.hymem_model,
                        thinking=args.hymem_thinking,
                    ) if extraction_canary_mode == "required" else None
                ),
                require_client_closed=extraction_canary_mode == "required",
                expected_prompt_version=extraction_prompt_version,
            )
            ledger.update_execution_segment(segment_id, _segment("running"))
            print_extraction_canary(extraction_canary_report)

            if not args.sim:
                answer_llm = owned_clients.own(
                    _build_llm(
                        args.answer_model, args.answer_base_url,
                        args.answer_api_key, args.answer_extra_body_obj,
                    ),
                    label="reader client",
                )
                judge_llm = owned_clients.own(
                    _build_llm(
                        args.judge_model, _DEEPSEEK_BASE_URL,
                        args.judge_api_key, args.judge_extra_body_obj,
                    ),
                    label="judge client",
                )
                ledger.update_execution_segment(segment_id, _segment("running"))

            if args.workers > 1:
                with ThreadPoolExecutor(
                    max_workers=min(args.workers, len(work_examples))
                ) as pool:
                    futures = {
                        pool.submit(
                            run_recall, example, args, answer_llm, judge_llm,
                            on_checkpoint=_persist,
                        ): example
                        for example in work_examples
                    }
                    for completed, future in enumerate(as_completed(futures), 1):
                        example = futures[future]
                        try:
                            _record_returned(future.result(), example)
                        except _CheckpointPersistenceAbort:
                            raise
                        except Exception as exc:
                            _record_example_failure(example, exc)
                        print(f"  [{completed}/{len(work_examples)}]", end="\r", flush=True)
            else:
                for completed, example in enumerate(work_examples, 1):
                    try:
                        row = run_recall(
                            example, args, answer_llm, judge_llm,
                            on_checkpoint=_persist,
                        )
                        _record_returned(row, example)
                    except _CheckpointPersistenceAbort:
                        raise
                    except Exception as exc:
                        _record_example_failure(example, exc)
                    print(f"  [{completed}/{len(work_examples)}]", end="\r", flush=True)
            ledger.update_execution_segment(segment_id, _segment("complete"))
        elif is_resume:
            # A terminal resume is an export/recovery operation: no canary,
            # provider construction, store opening, indexing, or paid work.
            try:
                ledger.update_execution_segment(
                    segment_id, _segment("complete")
                )
            except BenchmarkIntegrityError as exc:
                if "cannot mutate a finalized checkpoint" not in str(exc):
                    raise

        results = list(ledger.reconcile().rows)
        elapsed = time.time() - start_time
        scored = not args.sim
        payload = {
            "benchmark": "MSC",
            "version": "strict-v1",
            "date": datetime.now(timezone.utc).isoformat(),
            "scores": _strict_recall_scores(results, scored=scored),
            "strict_accuracy": strict_accuracy(results) if scored else None,
            "result_digest": content_hash(sanitize_for_artifact(results)),
            "legacy_bare_out": bool(args.out),
        }
        archive_now = datetime.now(timezone.utc)
        archive_path = results_dir / (
            f"msc-{archive_now.strftime('%Y%m%dT%H%M%SZ')}-"
            f"{archive_now.strftime('%f')}-seed{args.seed}-strict-"
            f"{manifest['run_id'].removeprefix('sha256:')[:12]}.json"
        )
        artifact = prepare_checkpoint_artifact(ledger, payload=payload)
        # The authoritative producer and consumer share one fail-closed
        # contract. Detect any adapter/registry drift while the recoverable
        # checkpoint is still leased and before cleanup-gated publication.
        from benchmarks.msc_registry import validate_msc_artifact
        validate_msc_artifact(artifact)
        publish_prepared_artifact_after_cleanup(
            archive_path,
            artifact,
            cleanup_actions=[
                ("resource_close", owned_clients.close),
                ("checkpoint_close", ledger.close),
            ],
        )
        write_latest_pointer(
            latest_path,
            archive=archive_path,
            run_id=manifest["run_id"],
            artifact_digest=content_hash(artifact),
        )
        print(f"  done in {elapsed:.0f}s{' '*20}")
        print(f"  strict archive → {archive_path}")

        # Historical consumers still receive a bare list, but only after the
        # immutable artifact and its pointer are safely published.
        if args.out:
            Path(args.out).write_text(
                json.dumps(sanitize_for_artifact(results), indent=2),
                encoding="utf-8",
            )
            print(f"\n  legacy per-question sidecar → {args.out}")
        if args.json:
            print(json.dumps(results, indent=2))
        elif scored:
            _print_recall_report(results)
        else:
            print("  simulation is unscored (retrieval diagnostics only)")
    finally:
        if ledger is not None:
            run_cleanup_actions(
                [("checkpoint_close", ledger.close)],
                primary_exception=sys.exc_info()[1],
            )


def main() -> None:
    """CLI entry point owning shared clients through worker completion."""

    with OwnedResourceScope("MSC shared provider clients") as owned_clients:
        return _run_main(owned_clients)


# A tiny in-schema fixture so --sim exercises the loader + labeling with no API.
_SIM_FIXTURE = [{
    "metadata": {"initial_data_id": "sim_0", "session_id": 3},
    "previous_dialogs": [
        {"time_num": 2, "time_unit": "days",
         "personas": [["I have two dogs"], ["I love hiking"]],
         "dialog": [{"text": "I just adopted two dogs, a lab and a beagle."},
                    {"text": "Nice! I love hiking with my dog on weekends."}]},
        {"time_num": 5, "time_unit": "hours",
         "personas": [["I have two dogs"], ["I work as a nurse"]],
         "dialog": [{"text": "The dogs kept me busy, but work as a nurse is hectic too."},
                    {"text": "Hiking clears my head after long shifts."}]},
    ],
    "personas": [["I have two dogs"], ["I love hiking", "I work as a nurse"]],
    "init_personas": [["I have two dogs"], ["I love hiking"]],
    "self_instruct": {"B": "How many dogs did you say you have?", "A": "Two dogs."},
}]


if __name__ == "__main__":
    main()
