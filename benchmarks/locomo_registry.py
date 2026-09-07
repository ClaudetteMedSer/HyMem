#!/usr/bin/env python3
"""Run registry for LoCoMo benchmark executions.

Same model as lme_registry.py / beam_registry.py: one row per run file,
date, scores, flags (recorded vs analyst-set, never guessed).  DB is
restart-safe: /home/node/.hermes/benchmarks/locomo_runs.db.

LoCoMo run files are HARSHER than the others:
- The adapter writes a BARE LIST of per-question rows to --out (no date,
  no config block, no flags, no scores).  Scores must be computed from
  the rows: overall (correct/n), answerable (cats 1-4), abstention
  (cat 5), and per-category rates.
- Diag-only files (all rows correct=null, e.g. locomo_conv26_diag.json)
  get overall=NULL and kind='diag' — recorded, not fabricated.
- Recovery-probe artifacts are probes, not runs, and are excluded by
  default (kind='probe' if explicitly passed).
- Documented runs whose JSON was lost (the canonical n=800 run of
  2026-07-29 lives only in locomo_adapter_spec.md) can be entered with
  `record-doc` — provenance is 'analyst:doc=...' so it can never be
  confused with a recorded run file.

Usage:
  locomo_registry.py ingest [FILE ...] [--set k=v ...]
  locomo_registry.py record-doc --archive NAME [--set k=v ...]
  locomo_registry.py list [--limit N] [--flag COL]
  locomo_registry.py query "SQL"
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

try:  # package import (tests): benchmarks.locomo_registry
    from . import run_registry as rr
    from .extraction_canary import (
        validate_extraction_canary_config_binding,
        validate_extraction_canary_report,
    )
    from .strictness import CHECKPOINT_VERSION, STRICT_PROTOCOL_VERSION, content_hash
except (ImportError, ValueError):  # direct CLI: python benchmarks/locomo_registry.py
    import run_registry as rr
    from extraction_canary import (  # type: ignore
        validate_extraction_canary_config_binding,
        validate_extraction_canary_report,
    )
    from strictness import (  # type: ignore
        CHECKPOINT_VERSION, STRICT_PROTOCOL_VERSION, content_hash,
    )

DB_ENV = "LOCOMO_REGISTRY_DB"

LOCOMO_COLUMNS = [
    ("archive", "TEXT"), ("kind", "TEXT"), ("run_date", "TEXT"),
    ("source_date", "TEXT"),
    # flags — never recorded by the adapter; NULL unless --set
    ("sample", "INTEGER"), ("seed", "INTEGER"), ("workers", "INTEGER"),
    ("top_k", "INTEGER"), ("message_fts_top_k", "INTEGER"),
    ("rerank_top_k", "INTEGER"), ("fts_top_k", "INTEGER"),
    ("graph_top_k", "INTEGER"), ("max_context_chars", "INTEGER"),
    ("embeddings", "INTEGER"), ("no_dream", "INTEGER"),
    ("dream_per_session", "INTEGER"), ("facts", "INTEGER"),
    ("facts_extraction", "INTEGER"), ("rules_extraction", "INTEGER"),
    ("graph_multihop", "INTEGER"),
    ("user_speaker", "TEXT"), ("name_prefix", "INTEGER"),
    ("answerable_clause", "INTEGER"),
    # models
    ("answer_model", "TEXT"), ("judge_model", "TEXT"),
    # scores (percent 0-100; NULL for diag/probe)
    ("overall", "REAL"), ("answerable", "REAL"), ("abstention", "REAL"),
    ("cat_1", "REAL"), ("cat_2", "REAL"), ("cat_3", "REAL"),
    ("cat_4", "REAL"), ("cat_5", "REAL"),
    ("count", "INTEGER"), ("answer_calls", "INTEGER"),
    ("judge_calls", "INTEGER"),
]

LOCOMO_OVERRIDES = {
    "sample", "seed", "workers", "top_k", "message_fts_top_k",
    "rerank_top_k", "fts_top_k", "graph_top_k", "max_context_chars",
    "embeddings", "no_dream", "dream_per_session", "facts",
    "facts_extraction", "rules_extraction", "graph_multihop",
    "user_speaker", "name_prefix", "answerable_clause",
    "answer_model", "judge_model",
}

# record-doc may also carry the scores themselves (they come from the
# documented run, not from a file that computes them).
DOC_OVERRIDES = LOCOMO_OVERRIDES | {
    "overall", "answerable", "abstention",
    "cat_1", "cat_2", "cat_3", "cat_4", "cat_5",
    "count", "answer_calls", "judge_calls", "run_date",
}

SPEC = {
    "db_file": "locomo_runs.db",
    "columns": LOCOMO_COLUMNS,
    "overrides": LOCOMO_OVERRIDES,
    "patterns": ("locomo*.json", "locomo-*.json"),
    "excludes": ("recovery_probe_", "planD_", "locomo_stores", "latest"),
    # §6 stamp policy: locomo stems carry no \\d{8}T\\d{6}Z stamp
    # (locomo_conv26_diag.json) -> NULL is the domain truth, not a defect.
    "stamp_policy": "optional",
    "gap_label": "flags (the adapter records none of them in --out files)",
    "gap_note": (
        "SELECT COUNT(*) FROM runs WHERE sample IS NULL AND top_k IS NULL "
        "AND embeddings IS NULL AND facts IS NULL"
    ),
}

PROBE_PREFIX = "recovery_probe_"


def _locomo_kind(name: str) -> str:
    if name.startswith(PROBE_PREFIX):
        return "probe"
    if "diag" in name:
        return "diag"
    if "rejudged" in name:
        return "rejudge"
    return "archive"


def _validate_row_canaries(rows: object) -> None:
    """Validate new bare-row provenance while retaining genuine legacy rows."""

    if not isinstance(rows, list) or not rows:
        return
    reports = [
        row.get("extraction_canary") if isinstance(row, dict) else None
        for row in rows
    ]
    present = [report is not None for report in reports]
    if not any(present):
        return
    if not all(present):
        raise ValueError("LoCoMo extraction canary row coverage is incomplete")
    first = reports[0]
    for report in reports:
        if report != first or not isinstance(report, dict):
            raise ValueError("LoCoMo extraction canary rows disagree")
        status = report.get("status")
        if status == "passed":
            mode = "required"
        elif status == "skipped_non_comparable" and report.get(
            "skip_reason"
        ) in {"simulation", "no_dream"}:
            mode = report["skip_reason"]
        else:
            raise ValueError("LoCoMo scored rows carry an invalid canary state")
        try:
            validate_extraction_canary_report(
                report,
                expected_mode=mode,
                # Rows without a canary remain the explicitly supported legacy
                # format above. A present current live pass must prove that its
                # dedicated provider transport actually closed.
                require_client_closed=mode == "required",
            )
        except Exception as exc:
            raise ValueError("LoCoMo extraction canary evidence is invalid") from exc


def _nonnegative_number(value: object, *, integer: bool = False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        or (integer and type(value) is not int)
    ):
        raise ValueError("strict LoCoMo usage value is malformed")
    return int(value) if integer else value


def _available_value(
    snapshot: Mapping, field: str, flag: str, *, integer: bool = False,
):
    available = snapshot.get(flag)
    if not isinstance(available, bool):
        raise ValueError("strict LoCoMo usage availability is malformed")
    value = snapshot.get(field)
    if not available:
        if value is not None:
            raise ValueError("strict LoCoMo unavailable usage has a value")
        return None
    return _nonnegative_number(value, integer=integer)


def _validate_llm_usage(snapshot: object) -> int | None:
    if not isinstance(snapshot, Mapping):
        raise ValueError("strict LoCoMo LLM usage is absent")
    calls = _available_value(
        snapshot, "calls", "calls_available", integer=True
    )
    attempts = _available_value(
        snapshot, "request_attempts", "request_attempts_available", integer=True
    )
    successes = _available_value(
        snapshot, "successful_responses", "successful_responses_available",
        integer=True,
    )
    if calls is not None and successes is not None and calls != successes:
        raise ValueError("strict LoCoMo successful call counts do not reconcile")
    if attempts is not None and successes is not None and attempts < successes:
        raise ValueError("strict LoCoMo provider attempts are below successes")
    token_available = snapshot.get("token_usage_available")
    if not isinstance(token_available, bool):
        raise ValueError("strict LoCoMo token availability is malformed")
    tokens = []
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = snapshot.get(field)
        if token_available:
            tokens.append(_nonnegative_number(value, integer=True))
        elif value is not None:
            raise ValueError("strict LoCoMo unavailable token usage has a value")
    if token_available and tokens[2] != tokens[0] + tokens[1]:
        raise ValueError("strict LoCoMo token totals do not reconcile")
    _available_value(snapshot, "latency_s", "latency_available")
    _available_value(snapshot, "cost_usd", "cost_available")
    return calls


def _validate_embedding_usage(
    snapshot: object, *, identity: Mapping, status: str, attempted: int,
) -> None:
    if not isinstance(snapshot, Mapping):
        raise ValueError("strict LoCoMo embedding usage is absent")
    configured = identity.get("configured")
    if not isinstance(configured, bool):
        raise ValueError("strict LoCoMo embedding identity is malformed")
    if snapshot.get("configured") is not configured:
        raise ValueError("strict LoCoMo embedding configured state drifted")
    for key in ("backend", "quality"):
        if not isinstance(snapshot.get(key), str) or not snapshot[key]:
            raise ValueError("strict LoCoMo embedding identity is malformed")
    if snapshot.get("network_free") not in {True, False, None}:
        raise ValueError("strict LoCoMo embedding network posture is malformed")
    identity_marker = (
        snapshot.get("identity_consistent")
        if "identity_consistent" in snapshot
        else snapshot.get("identity_available")
    )
    if not isinstance(identity_marker, bool):
        raise ValueError("strict LoCoMo embedding identity availability is malformed")
    dimension = snapshot.get("dimension")
    if dimension is not None:
        _nonnegative_number(dimension, integer=True)
        if dimension <= 0:
            raise ValueError("strict LoCoMo embedding dimension is malformed")
    require_identity = status == "complete" or attempted > 0
    unavailable_running = bool(
        not require_identity
        and configured
        and snapshot.get("backend") == "unavailable"
        and snapshot.get("quality") == "none"
        and snapshot.get("network_free") is None
        and snapshot.get("model") is None
        and snapshot.get("dimension") is None
        and identity_marker is False
    )
    if not unavailable_running:
        expected = {
            "backend": identity.get("backend"),
            "quality": identity.get("quality"),
            "network_free": identity.get("network_free"),
            "model": identity.get("vector_space_key"),
            "dimension": identity.get("dimension"),
            "identity_exact": identity.get("identity_exact"),
            "reuse_scope": identity.get("reuse_scope"),
        }
        for field, expected_value in expected.items():
            if snapshot.get(field) != expected_value:
                raise ValueError("strict LoCoMo embedding runtime identity drifted")
    if require_identity and identity_marker is not True:
        raise ValueError("strict LoCoMo embedding runtime identity is unavailable")
    for field, flag in (
        ("calls", "calls_available"),
        ("request_attempts", "request_attempts_available"),
        ("successful_responses", "successful_responses_available"),
        ("input_count", "input_count_available"),
        ("input_characters", "input_characters_available"),
    ):
        _available_value(snapshot, field, flag, integer=True)
    _available_value(snapshot, "latency_s", "latency_available")
    _available_value(snapshot, "cost_usd", "cost_available")
    provider_tokens = snapshot.get("provider_token_usage_available")
    if not isinstance(provider_tokens, bool):
        raise ValueError("strict LoCoMo embedding token availability is malformed")
    for field in ("prompt_tokens", "total_tokens"):
        value = snapshot.get(field)
        if provider_tokens:
            _nonnegative_number(value, integer=True)
        elif value is not None:
            raise ValueError("strict LoCoMo unavailable embedding tokens have a value")


def _strict_scores(rows: list[dict]) -> dict[str, dict[str, int | float]]:
    by_type: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        qtype = str(row.get("question_type") or "unknown").replace("_abs", "")
        by_type[qtype].append(row["correct"])
    scores = {
        qtype: {"accuracy": sum(values) / len(values), "count": len(values)}
        for qtype, values in by_type.items()
    }
    all_values = [value for values in by_type.values() for value in values]
    scores["OVERALL"] = {
        "accuracy": sum(all_values) / len(all_values) if all_values else 0.0,
        "count": len(all_values),
    }
    return scores


def _validate_strict_locomo(data: dict) -> tuple[list[dict], dict, dict]:
    """Fail closed on the strict envelope before registry score extraction."""

    manifest = data.get("manifest")
    execution = data.get("execution")
    config = data.get("config")
    models = data.get("models")
    rows = data.get("per_question")
    if not all(isinstance(value, dict) for value in (
        manifest, execution, config, models,
    )) or not isinstance(rows, list):
        raise ValueError("strict LoCoMo artifact envelope is malformed")
    if data.get("benchmark") != "LoCoMo" or data.get("version") != "strict-v1":
        raise ValueError("strict LoCoMo top-level benchmark/version is invalid")
    if manifest.get("schema") != STRICT_PROTOCOL_VERSION:
        raise ValueError("strict LoCoMo manifest schema is unsupported")
    if manifest.get("benchmark") != "LoCoMo":
        raise ValueError("strict LoCoMo benchmark identity is invalid")
    expected_run_id = content_hash({
        key: value for key, value in manifest.items() if key != "run_id"
    })
    if manifest.get("run_id") != expected_run_id:
        raise ValueError("strict LoCoMo manifest run identity is invalid")
    if config != manifest.get("config") or models != manifest.get("models"):
        raise ValueError("strict LoCoMo top-level identity differs from manifest")
    if manifest.get("config_hash") != content_hash(config):
        raise ValueError("strict LoCoMo config hash is invalid")
    if manifest.get("model_hash") != content_hash(models):
        raise ValueError("strict LoCoMo model hash is invalid")
    for field in ("code_hash", "data_hash", "expected_ids_hash"):
        if re.fullmatch(r"sha256:[0-9a-f]{64}", str(manifest.get(field))) is None:
            raise ValueError(f"strict LoCoMo {field} is malformed")
    if manifest.get("protocol_split") not in {"full", "dev", "holdout"}:
        raise ValueError("strict LoCoMo protocol split is invalid")
    if isinstance(manifest.get("seed"), bool) or not isinstance(
        manifest.get("seed"), int
    ):
        raise ValueError("strict LoCoMo seed is malformed")
    for field in (
        "development_only", "official_split", "official_comparable",
        "label_free_answer_path", "exploratory_label_steering",
        "exploratory_non_comparable", "scored_run",
    ):
        if not isinstance(manifest.get(field), bool):
            raise ValueError(f"strict LoCoMo manifest {field} is malformed")
    for field in (
        "label_free_answer_path", "exploratory_label_steering",
        "exploratory_non_comparable", "scored_run",
    ):
        if config.get(field) is not manifest.get(field):
            raise ValueError(f"strict LoCoMo manifest/config {field} drifted")
    effective = config.get("effective_hymem_config")
    if not isinstance(effective, dict):
        raise ValueError("strict LoCoMo effective HyMem config is absent")
    try:
        validate_extraction_canary_config_binding(
            config.get("extraction_canary"), effective
        )
    except Exception as exc:
        raise ValueError(
            "strict LoCoMo extraction canary policy is invalid"
        ) from exc
    for field in (
        "message_fts_top_k", "rerank_top_k", "fts_top_k", "graph_top_k",
    ):
        value = effective.get(field)
        if type(value) is not int or value < 0:
            raise ValueError("strict LoCoMo effective aperture is malformed")
    for field in (
        "facts_enabled", "facts_extraction_enabled",
        "rules_extraction_enabled", "graph_multihop_enabled",
        "content_redaction_enabled",
    ):
        if not isinstance(effective.get(field), bool):
            raise ValueError("strict LoCoMo effective architecture flags are malformed")
    for requested, resolved in (
        ("facts", "facts_enabled"),
        ("facts_extraction", "facts_extraction_enabled"),
        ("rules_extraction", "rules_extraction_enabled"),
        ("graph_multihop", "graph_multihop_enabled"),
    ):
        raw = config.get(requested)
        if raw is not None and (
            not isinstance(raw, bool) or raw is not effective[resolved]
        ):
            raise ValueError("strict LoCoMo requested/effective config drifted")

    expected_count = manifest.get("expected_count")
    if (
        isinstance(expected_count, bool) or not isinstance(expected_count, int)
        or expected_count <= 0 or expected_count != len(rows)
    ):
        raise ValueError("strict LoCoMo denominator is incomplete")
    ids: list[str] = []
    failure_count = missing_count = completed_count = 0
    canary_reports = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"strict LoCoMo row {index} is not an object")
        item_id = row.get("question_id")
        if (
            not isinstance(item_id, str) or not item_id.strip()
            or item_id != item_id.strip() or item_id in ids
        ):
            raise ValueError(f"strict LoCoMo row {index} has malformed id")
        ids.append(item_id)
        if not isinstance(row.get("correct"), bool):
            raise ValueError(f"strict LoCoMo row {index} has malformed verdict")
        strict_failure = row.get("strict_failure")
        failure = row.get("benchmark_failure")
        if strict_failure is None and manifest["scored_run"] is False:
            # Diagnostic/simulation ledgers preserve their original row shape;
            # their entry status is still reflected by benchmark_failure.
            strict_failure = bool(failure)
        elif not isinstance(strict_failure, bool):
            raise ValueError(f"strict LoCoMo row {index} lacks failure posture")
        if strict_failure:
            if not isinstance(failure, str) or not failure or row["correct"]:
                raise ValueError("strict LoCoMo failed row is inconsistent")
            failure_count += 1
            missing_count += failure == "missing_prediction"
        elif failure:
            raise ValueError("strict LoCoMo completed row carries a failure")
        else:
            completed_count += 1
        report = row.get("extraction_canary")
        if not isinstance(report, dict):
            raise ValueError("strict LoCoMo row canary evidence is absent")
        canary_reports.append(report)
    if manifest.get("expected_ids_hash") != content_hash(ids):
        raise ValueError("strict LoCoMo id order/hash is invalid")

    counts = execution.get("counts")
    segments = execution.get("segments")
    if not isinstance(counts, dict) or not isinstance(segments, list) or not segments:
        raise ValueError("strict LoCoMo execution evidence is incomplete")
    checkpoint = execution.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"schema", "state_sha256"}
        or checkpoint.get("schema") != CHECKPOINT_VERSION
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(checkpoint.get("state_sha256"))
        ) is None
    ):
        raise ValueError("strict LoCoMo checkpoint digest evidence is malformed")
    expected_counts = {
        "expected": expected_count,
        "attempted": expected_count - missing_count,
        "unique_attempted": expected_count - missing_count,
        "completed": completed_count,
        "failed": failure_count,
        "missing": missing_count,
    }
    for field, expected in expected_counts.items():
        if type(counts.get(field)) is not int or counts[field] != expected:
            raise ValueError("strict LoCoMo execution counts do not reconcile")
    total_attempts = counts.get("total_attempts")
    if (
        type(total_attempts) is not int
        or total_attempts < counts["attempted"]
    ):
        raise ValueError("strict LoCoMo total attempt count is invalid")

    pipeline = models.get("memory_pipeline")
    embedding = models.get("embedding")
    if not isinstance(pipeline, dict) or not isinstance(embedding, dict):
        raise ValueError("strict LoCoMo provider identity is incomplete")
    configured_embedding = embedding.get("configured")
    if not isinstance(configured_embedding, bool):
        raise ValueError("strict LoCoMo embedding identity is malformed")
    segment_ids: set[str] = set()
    segment_attempts = 0
    all_segment_canaries = []
    all_complete = True
    answer_calls = judge_calls = 0
    calls_exact = True
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("strict LoCoMo execution segment is malformed")
        segment_id = segment.get("segment_id")
        if (
            not isinstance(segment_id, str) or not segment_id
            or segment_id in segment_ids
        ):
            raise ValueError("strict LoCoMo execution segment id is invalid")
        segment_ids.add(segment_id)
        status = segment.get("status")
        if status not in {"running", "complete"}:
            raise ValueError("strict LoCoMo execution segment status is invalid")
        all_complete = all_complete and status == "complete"
        attempted = segment.get("attempted_attempts")
        if type(attempted) is not int or attempted < 0:
            raise ValueError("strict LoCoMo segment attempt count is invalid")
        segment_attempts += attempted
        elapsed = segment.get("elapsed_s")
        _nonnegative_number(elapsed)
        if segment.get("model_identities") != models:
            raise ValueError("strict LoCoMo segment model identity drifted")
        report = segment.get("extraction_canary")
        if not isinstance(report, dict):
            raise ValueError("strict LoCoMo extraction canary is absent")
        if report.get("status") == "passed":
            mode = "required"
        elif report.get("status") == "pending":
            mode = "pending"
        elif report.get("status") == "failed":
            mode = "failed"
        elif report.get("status") == "skipped_non_comparable":
            mode = report.get("skip_reason")
        else:
            raise ValueError("strict LoCoMo extraction canary is invalid")
        try:
            validate_extraction_canary_report(
                report, expected_mode=mode,
                expected_client=(pipeline if mode in {"required", "failed"} else None),
                require_client_closed=mode in {"required", "failed"},
                expected_prompt_version=effective["prompt_version"],
            )
        except Exception as exc:
            raise ValueError(
                "strict LoCoMo extraction canary evidence is invalid"
            ) from exc
        all_segment_canaries.append(report)
        reader_calls = _validate_llm_usage(segment.get("reader_usage"))
        segment_judge_calls = _validate_llm_usage(segment.get("judge_usage"))
        _validate_llm_usage(segment.get("memory_pipeline_usage"))
        _validate_embedding_usage(
            segment.get("embedding_usage"), identity=embedding,
            status=status, attempted=attempted,
        )
        if reader_calls is None or segment_judge_calls is None or status != "complete":
            calls_exact = False
        else:
            answer_calls += reader_calls
            judge_calls += segment_judge_calls
        for key in ("indexing_runs", "indexing_failures"):
            if not isinstance(segment.get(key), list):
                raise ValueError(f"strict LoCoMo {key} evidence is malformed")
    if segment_attempts != total_attempts:
        raise ValueError("strict LoCoMo segment attempts do not reconcile")
    if not all(any(report == candidate for candidate in all_segment_canaries)
               for report in canary_reports):
        raise ValueError("strict LoCoMo row canary is not owned by a segment")

    declared_scores = data.get("scores")
    recomputed_scores = _strict_scores(rows)
    if declared_scores != recomputed_scores:
        raise ValueError("strict LoCoMo declared scores do not reconcile")
    expected_accuracy = (
        sum(row["correct"] for row in rows) / len(rows)
        if manifest["scored_run"] else None
    )
    if data.get("strict_accuracy") != expected_accuracy:
        raise ValueError("strict LoCoMo strict accuracy does not reconcile")
    if data.get("result_digest") != content_hash(rows):
        raise ValueError("strict LoCoMo result digest is invalid")
    return rows, {
        "answer_calls": answer_calls if calls_exact and all_complete else None,
        "judge_calls": judge_calls if calls_exact and all_complete else None,
    }, config


def _locomo_row(data, path: Path) -> dict:
    if (
        isinstance(data, dict)
        and set(data) <= {"archive", "run_id", "artifact_digest"}
        and "archive" in data
    ):
        raise ValueError(
            "locomo-latest.json is a mutable pointer, not a result artifact"
        )
    row = {c: None for c, _ in LOCOMO_COLUMNS}
    row["archive"] = path.name
    row["kind"] = _locomo_kind(path.name)
    # §6: stamp-derived source date (NULL when the stem carries none).
    row["source_date"] = rr.stem_source_date(
        path.name, SPEC.get("stamp_policy", "optional"))
    strict = bool(
        isinstance(data, dict)
        and (
            str(data.get("version", "")).startswith("strict-")
            or {"manifest", "execution", "per_question"} <= set(data)
        )
    )
    config = {}
    models = {}
    if isinstance(data, dict):  # probe artifacts / metadata wrappers
        row["run_date"] = rr.iso_ts(data.get("date") or data.get("created_at"))
        if strict:
            rows, strict_usage, config = _validate_strict_locomo(data)
            models = data["models"]
            row["answer_calls"] = strict_usage["answer_calls"]
            row["judge_calls"] = strict_usage["judge_calls"]
            if data["manifest"]["scored_run"] is False:
                row["kind"] = (
                    "simulation" if config.get("sim") is True
                    else "diagnostic"
                )
        else:
            rows = data.get("results") or data.get("rows") or []
        if not rows and "correct" not in data:
            rows = []
    else:
        rows = data if isinstance(data, list) else []
        row["run_date"] = None  # §6.5: bare lists record absence as NULL

    if not strict:
        _validate_row_canaries(rows)

    scoreable = not strict or data["manifest"]["scored_run"] is True
    scored = [
        r for r in rows if scoreable and isinstance(r.get("correct"), bool)
    ]
    n = len(scored)
    if n:
        n_correct = sum(1 for r in scored if r["correct"])
        row["overall"] = round(n_correct / n * 100, 3)
        ans = [r for r in scored if r.get("category") != 5]
        abst = [r for r in scored if r.get("category") == 5]
        if ans:
            row["answerable"] = round(
                sum(1 for r in ans if r["correct"]) / len(ans) * 100, 3)
        if abst:
            row["abstention"] = round(
                sum(1 for r in abst if r["correct"]) / len(abst) * 100, 3)
        for c in range(1, 6):
            cc = [r for r in scored if r.get("category") == c]
            if cc:
                row[f"cat_{c}"] = round(
                    sum(1 for r in cc if r["correct"]) / len(cc) * 100, 3)
    row["count"] = n or (len(rows) if rows else None)
    if strict:
        for key in (
            "sample", "seed", "workers", "top_k", "message_fts_top_k",
            "rerank_top_k", "fts_top_k", "graph_top_k",
            "max_context_chars", "embeddings", "no_dream",
            "dream_per_session", "facts", "facts_extraction",
            "rules_extraction", "graph_multihop", "user_speaker",
            "name_prefix", "answerable_clause",
        ):
            if key in config:
                row[key] = config.get(key)
        effective = config["effective_hymem_config"]
        for key in (
            "message_fts_top_k", "rerank_top_k", "fts_top_k", "graph_top_k",
        ):
            row[key] = effective[key]
        row["facts"] = effective["facts_enabled"]
        row["facts_extraction"] = effective["facts_extraction_enabled"]
        row["rules_extraction"] = effective["rules_extraction_enabled"]
        row["graph_multihop"] = effective["graph_multihop_enabled"]
        reader = models.get("reader") if isinstance(models.get("reader"), dict) else {}
        judge = models.get("judge") if isinstance(models.get("judge"), dict) else {}
        row["answer_model"] = reader.get("model")
        row["judge_model"] = judge.get("model")
    row["extras"] = json.dumps({
        "n_rows": len(rows) if isinstance(rows, list) else 0,
        "n_scored": n,
        "raw_is_list": isinstance(data, list),
        "keys": sorted(rows[0].keys()) if isinstance(rows, list) and rows else [],
        "strict": strict,
        "manifest": data.get("manifest") if strict else None,
        "execution": data.get("execution") if strict else None,
    }, default=str)
    return row


def _backfill(db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _locomo_row
    return rr.cmd_backfill(spec, db_path=db_path)


def _ingest(files, overrides=None, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _locomo_row
    return rr.cmd_ingest(spec, files or None, overrides, db_path=db_path)


def _record_doc(archive, overrides=None, db_path=None):
    """Enter a run whose run-file is lost but is documented elsewhere.

    Provenance starts with 'analyst:doc=' — NOT 'recorded'.  Only
    whitelisted columns can be set.  The archive string is the doc
    reference, e.g. 'locomo_adapter_spec.md:2026-07-29 n=800'.
    """
    overrides = dict(overrides or {})
    con = rr.connect(SPEC, db_path)
    ex = con.execute("SELECT id FROM runs WHERE archive=?", (archive,)).fetchone()
    if ex:
        return "skipped"
    row = {c: None for c, _ in LOCOMO_COLUMNS}
    applied = {}
    for k, v in overrides.items():
        if k not in DOC_OVERRIDES:
            continue
        typ = dict(SPEC["columns"]).get(k, "TEXT")
        # §6.5: same canonicalisation as beam._record_doc -- both doc entry
        # points must agree, or a locomo doc row entered without a
        # subsequent backfill sits at width 10 in the sort column.
        row[k] = rr.iso_ts(v) if k == "run_date" else rr._coerce(v, typ)
        applied[k] = row[k]
    row["archive"] = archive
    row["kind"] = "doc"
    row["source_date"] = "DOC"
    names = [c for c, _ in LOCOMO_COLUMNS] + ["flags_provenance", "extras"]
    vals = [row.get(c) for c in names]
    prov = "analyst:doc=" + archive
    if applied:
        prov += "; " + "; ".join(f"analyst:{k}={v}" for k, v in applied.items())
    vals[-2] = prov
    vals[-1] = json.dumps({"doc_row": True, "analyst_set": applied}, default=str)
    con.execute(
        f"INSERT INTO runs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
        vals)
    con.commit()
    print(f"DB: {db_path or rr.DEFAULT_REGISTRY_DIR / SPEC['db_file']}  row added (kind=doc, prov='{prov}')")


def _list(limit=30, flag=None, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _locomo_row
    return rr.cmd_list(spec, limit, flag, db_path)


def _query(sql, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _locomo_row
    return rr.cmd_query(spec, sql, db_path)


def _parse_set(args):
    ov = {}
    out = []
    i = 0
    while i < len(args):
        if args[i] == "--set" and i + 1 < len(args):
            k, v = args[i + 1].split("=", 1)
            ov[k] = v
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out, ov


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "ingest":
        files, ov = _parse_set(sys.argv[2:])
        _ingest(files or None, ov)
    elif cmd == "backfill":
        if _backfill():
            sys.exit(1)   # §6.5: unreachable rows were not migrated
    elif cmd == "record-doc":
        files, ov = _parse_set(sys.argv[2:])
        if not files:
            print("record-doc --archive NAME [--set k=v ...]")
            sys.exit(1)
        _record_doc(files[0], ov)
    elif cmd == "list":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--limit", type=int, default=30)
        p.add_argument("--flag")
        a = p.parse_args(sys.argv[2:])
        _list(a.limit, a.flag)
    elif cmd == "query":
        _query(" ".join(sys.argv[2:]))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
