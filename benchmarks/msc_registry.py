#!/usr/bin/env python3
"""Fail-closed consumer for authoritative strict MSC recall artifacts.

MSC historically emitted only mutable bare-row JSON, so it never had a run
registry.  The strict adapter now publishes immutable envelopes.  This module
is intentionally a small validator/discovery layer rather than a second score
implementation or a migration of legacy files: a bare list is still useful to
old analysis scripts, but it is not admissible benchmark evidence here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from hymem.contrib.endpoint_policy import (
    TRANSPORT_SECURITY_NONE,
    validate_http_endpoint,
    validate_recorded_embedding_endpoint,
)
from hymem.contrib.model_policy import require_active_model

try:  # package imports in tests
    from .extraction_canary import (
        validate_extraction_canary_config_binding,
        validate_extraction_canary_report,
    )
    from .strictness import (
        CHECKPOINT_VERSION,
        STRICT_PROTOCOL_VERSION,
        BenchmarkIntegrityError,
        content_hash,
        effective_hymem_config_identity,
        read_artifact_or_pointer,
    )
except (ImportError, ValueError):  # direct CLI
    from extraction_canary import (  # type: ignore
        validate_extraction_canary_config_binding,
        validate_extraction_canary_report,
    )
    from strictness import (  # type: ignore
        CHECKPOINT_VERSION,
        STRICT_PROTOCOL_VERSION,
        BenchmarkIntegrityError,
        content_hash,
        effective_hymem_config_identity,
        read_artifact_or_pointer,
    )


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ARCHIVE = re.compile(
    r"msc-\d{8}T\d{6}Z-\d{6}-seed-?\d+-strict-[0-9a-f]{12}\.json\Z"
)
_ROOT_FIELDS = {
    "benchmark", "version", "date", "scores", "strict_accuracy",
    "result_digest", "legacy_bare_out", "created_at", "manifest", "config",
    "models", "execution", "per_question",
}
_MANIFEST_FIELDS = {
    "schema", "benchmark", "code_hash", "config_hash", "model_hash",
    "data_hash", "expected_ids_hash", "expected_count", "seed",
    "protocol_split", "development_only", "official_split",
    "official_comparable", "label_free_answer_path",
    "exploratory_label_steering", "exploratory_non_comparable", "scored_run",
    "protocol_limitation", "calibration_receipt_hash", "config", "models",
    "run_id",
}


def _fail(message: str) -> BenchmarkIntegrityError:
    return BenchmarkIntegrityError(f"strict MSC artifact {message}")


def _nonnegative(value: object, *, integer: bool = False) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        or (integer and type(value) is not int)
    ):
        raise _fail("contains malformed non-negative usage")
    return int(value) if integer else value


def _require_aware_timestamp(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise _fail(f"{field} timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail(f"{field} timestamp is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _fail(f"{field} timestamp lacks a timezone")


def _available(
    snapshot: Mapping[str, Any], field: str, availability: str, *, integer=False,
) -> int | float | None:
    marker = snapshot.get(availability)
    if not isinstance(marker, bool):
        raise _fail("contains malformed usage availability")
    value = snapshot.get(field)
    if not marker:
        if value is not None:
            raise _fail("claims unavailable usage with a value")
        return None
    return _nonnegative(value, integer=integer)


def _validate_llm_usage(snapshot: object) -> int | None:
    if not isinstance(snapshot, Mapping):
        raise _fail("lacks LLM usage evidence")
    calls = _available(snapshot, "calls", "calls_available", integer=True)
    attempts = _available(
        snapshot, "request_attempts", "request_attempts_available", integer=True
    )
    successes = _available(
        snapshot, "successful_responses", "successful_responses_available",
        integer=True,
    )
    if calls is not None and successes is not None and calls != successes:
        raise _fail("LLM successful response counts do not reconcile")
    if attempts is not None and successes is not None and attempts < successes:
        raise _fail("LLM request attempts are below successful responses")
    token_available = snapshot.get("token_usage_available")
    if not isinstance(token_available, bool):
        raise _fail("contains malformed token availability")
    tokens: list[int] = []
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = snapshot.get(field)
        if token_available:
            tokens.append(_nonnegative(value, integer=True))
        elif value is not None:
            raise _fail("claims unavailable tokens with a value")
    if token_available and tokens[2] != tokens[0] + tokens[1]:
        raise _fail("token totals do not reconcile")
    _available(snapshot, "latency_s", "latency_available")
    _available(snapshot, "cost_usd", "cost_available")
    return calls


def _validate_invalid_canary_sentinel(report: Mapping[str, Any]) -> None:
    expected = {
        "status", "failure_reason", "reported_status", "completion_calls",
        "provider_attempts", "client_closed", "usage",
    }
    if (
        set(report) != expected
        or report.get("status") != "invalid_report"
        or report.get("failure_reason") != "internal_validation_failure"
        or report.get("reported_status") not in {
            "passed", "failed", "pending", "skipped_non_comparable",
            "not_run_no_pending", "unknown",
        }
        or report.get("client_closed") not in {True, False, None}
    ):
        raise _fail("invalid-canary sentinel is malformed")
    for field in ("completion_calls", "provider_attempts"):
        value = report.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise _fail("invalid-canary counters are malformed")
    usage = report.get("usage")
    if usage is not None:
        expected_usage = {
            "calls", "calls_available", "request_attempts",
            "request_attempts_available", "successful_responses",
            "successful_responses_available", "prompt_tokens",
            "completion_tokens", "total_tokens", "latency_s", "cost_usd",
            "token_usage_available", "latency_available", "cost_available",
        }
        if not isinstance(usage, Mapping) or set(usage) != expected_usage:
            raise _fail("invalid-canary usage projection is malformed")
        for value in usage.values():
            if value is None or isinstance(value, bool):
                continue
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise _fail("invalid-canary usage projection is unsafe")


def _require_zero_llm_work(snapshot: Mapping[str, Any]) -> None:
    for field, available in (
        ("calls", "calls_available"),
        ("request_attempts", "request_attempts_available"),
        ("successful_responses", "successful_responses_available"),
    ):
        if snapshot.get(available) is not True or snapshot.get(field) != 0:
            raise _fail("historical preflight segment contains benchmark work")


def _require_zero_embedding_work(snapshot: Mapping[str, Any]) -> None:
    for field, available in (
        ("calls", "calls_available"),
        ("request_attempts", "request_attempts_available"),
        ("successful_responses", "successful_responses_available"),
        ("input_count", "input_count_available"),
        ("input_characters", "input_characters_available"),
    ):
        if snapshot.get(available) is not True or snapshot.get(field) != 0:
            raise _fail("terminal segment contains embedding work")


def _validate_embedding_usage(
    snapshot: object, *, identity: Mapping[str, Any], attempted: int,
) -> None:
    if not isinstance(snapshot, Mapping):
        raise _fail("lacks embedding usage evidence")
    configured = identity.get("configured")
    if not isinstance(configured, bool) or snapshot.get("configured") is not configured:
        raise _fail("embedding configured state drifted")
    identity_marker = snapshot.get(
        "identity_consistent",
        snapshot.get("identity_available"),
    )
    if not isinstance(identity_marker, bool):
        raise _fail("embedding identity availability is malformed")
    unavailable_before_work = bool(
        attempted == 0
        and configured
        and snapshot.get("backend") == "unavailable"
        and identity_marker is False
    )
    if not unavailable_before_work:
        for field, identity_field in (
            ("backend", "backend"), ("quality", "quality"),
            ("network_free", "network_free"),
            ("model", "vector_space_key"), ("dimension", "dimension"),
            ("identity_exact", "identity_exact"),
            ("reuse_scope", "reuse_scope"),
        ):
            if snapshot.get(field) != identity.get(identity_field):
                raise _fail("embedding runtime identity drifted")
    if attempted > 0 and identity_marker is not True:
        raise _fail("embedding identity is unavailable after work")
    for field, flag in (
        ("calls", "calls_available"),
        ("request_attempts", "request_attempts_available"),
        ("successful_responses", "successful_responses_available"),
        ("input_count", "input_count_available"),
        ("input_characters", "input_characters_available"),
    ):
        _available(snapshot, field, flag, integer=True)
    _available(snapshot, "latency_s", "latency_available")
    _available(snapshot, "cost_usd", "cost_available")
    token_available = snapshot.get("provider_token_usage_available")
    if not isinstance(token_available, bool):
        raise _fail("embedding token availability is malformed")
    for field in ("prompt_tokens", "total_tokens"):
        value = snapshot.get(field)
        if token_available:
            _nonnegative(value, integer=True)
        elif value is not None:
            raise _fail("claims unavailable embedding tokens with a value")


def _validate_model_identity(
    models: Mapping[str, Any], *, scored: bool, simulation: bool,
) -> None:
    if set(models) != {"reader", "judge", "memory_pipeline", "embedding"}:
        raise _fail("model identity is incomplete")
    for role in ("reader", "judge", "memory_pipeline", "embedding"):
        if not isinstance(models.get(role), Mapping):
            raise _fail("model identity contains a malformed role")
    reader = models["reader"]
    judge = models["judge"]
    pipeline = models["memory_pipeline"]
    embedding = models["embedding"]
    try:
        from hymem.dreaming.aggregation_material import (
            validate_public_embedding_identity,
        )

        validated_embedding = validate_public_embedding_identity(embedding)
    except (TypeError, ValueError) as exc:
        raise _fail("embedding identity is malformed or obsolete") from exc
    if dict(embedding) != validated_embedding:
        raise _fail("embedding identity is not canonical")
    if simulation:
        if scored:
            raise _fail("simulation is marked scored")
        absent_role = {
            "configured": False,
            "client_class": None,
            "provider": "none",
            "model": None,
            "base_url": None,
        }
        expected_pipeline = {
            "configured": True,
            "client_class": "hymem.extraction.llm.StubLLMClient",
            "provider": "local_stub",
            "model": None,
            "base_url": None,
            "response_policy": "constant-empty-json-list-v1",
        }
        if (
            dict(reader) != absent_role
            or dict(judge) != absent_role
            or dict(pipeline) != expected_pipeline
            or embedding.get("configured") is not False
        ):
            raise _fail("simulation stub/no-client identity is inaccurate")
        return

    if not scored:
        raise _fail("live MSC recall is unexpectedly unscored")
    expected_role_fields = {
        "configured", "client_class", "provider", "model", "base_url",
        "temperature", "max_tokens", "extra_body",
    }
    if set(reader) != expected_role_fields or set(judge) != (
        expected_role_fields | {"protocol"}
    ) or set(pipeline) != {
        "configured", "client_class", "provider", "model", "base_url",
        "thinking_mode", "effective_extra_body",
    }:
        raise _fail("live model role shape is invalid")

    def provider_endpoint(role: Mapping[str, Any], *, label: str):
        if role.get("configured") is not True:
            raise _fail("live provider identity is not configured")
        model = role.get("model")
        if (
            not isinstance(model, str) or not model
            or model != model.strip()
        ):
            raise _fail("live provider model identity is absent")
        try:
            require_active_model(model, role=f"MSC {label}")
            endpoint = validate_http_endpoint(role.get("base_url"), label=label)
        except (TypeError, ValueError) as exc:
            raise _fail("live provider identity is unsafe") from exc
        expected_provider = endpoint.official_provider or "openai-compatible"
        if (
            role.get("base_url") != endpoint.url
            or role.get("provider") != expected_provider
        ):
            raise _fail("live provider endpoint identity is inconsistent")
        return endpoint

    provider_endpoint(reader, label="reader")
    judge_endpoint = provider_endpoint(judge, label="judge")
    pipeline_endpoint = provider_endpoint(pipeline, label="memory pipeline")
    try:
        try:
            from .longmemeval_adapter import resolve_model_extra_body
        except (ImportError, ValueError):  # direct CLI
            from longmemeval_adapter import resolve_model_extra_body  # type: ignore

        reader_body, _ = resolve_model_extra_body(
            reader["model"], reader["base_url"], reader.get("extra_body")
        )
        judge_body, _ = resolve_model_extra_body(
            judge["model"], judge["base_url"], judge.get("extra_body")
        )
    except (TypeError, ValueError) as exc:
        raise _fail("reader/judge request body is invalid") from exc
    if (
        reader.get("client_class") != "longmemeval_adapter.LLMClient"
        or isinstance(reader.get("temperature"), bool)
        or reader.get("temperature") != 0.0
        or type(reader.get("max_tokens")) is not int
        or reader.get("max_tokens") != 1024
        or reader.get("extra_body") is not None
        and not isinstance(reader.get("extra_body"), Mapping)
        or reader.get("extra_body") != reader_body
    ):
        raise _fail("reader request identity is inconsistent")
    if (
        judge.get("client_class") != "longmemeval_adapter.LLMClient"
        or judge_endpoint.official_provider != "deepseek"
        or judge.get("base_url") != "https://api.deepseek.com"
        or isinstance(judge.get("temperature"), bool)
        or judge.get("temperature") != 0.0
        or type(judge.get("max_tokens")) is not int
        or judge.get("max_tokens") != 10
        or judge.get("extra_body") is not None
        and not isinstance(judge.get("extra_body"), Mapping)
        or judge.get("extra_body") != judge_body
        or judge.get("protocol") != "longmemeval-local-judge"
    ):
        raise _fail("judge request identity is inconsistent")
    thinking = pipeline.get("thinking_mode")
    if thinking not in {"auto", "disabled", "off", "enabled"}:
        raise _fail("memory-pipeline thinking identity is invalid")
    expected_pipeline_body = (
        {"thinking": {"type": "disabled"}}
        if thinking == "disabled" or (
            thinking == "auto"
            and (
                "deepseek" in pipeline_endpoint.hostname.casefold()
                or "deepseek" in pipeline["model"].casefold()
            )
        ) else {}
    )
    if (
        pipeline.get("client_class")
        != "hymem.contrib.openai_client.OpenAICompatibleClient"
        or pipeline.get("effective_extra_body") != expected_pipeline_body
    ):
        raise _fail("memory-pipeline request identity is inconsistent")

    if embedding.get("configured") is False:
        return
    if embedding.get("backend") != "openai_compatible":
        raise _fail("configured embedding backend is unsupported")


def _expected_effective_hymem_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct strict-v1's exact resolved HyMem dataclass identity."""

    from hymem import HyMemConfig

    overrides: dict[str, Any] = {
        "message_fts_top_k": 15,
        "fts_top_k": 10,
        "graph_top_k": 10,
        "aggregation_nodes_enabled": False,
    }
    if config.get("rules_extraction") is not None:
        overrides["rules_extraction_enabled"] = config["rules_extraction"]
    if config.get("graph_multihop"):
        overrides["graph_multihop_enabled"] = True
    if config.get("facts") is not None:
        overrides["facts_enabled"] = config["facts"]
    if config.get("facts_extraction") is not None:
        overrides["facts_extraction_enabled"] = config["facts_extraction"]
    expected = effective_hymem_config_identity(
        HyMemConfig(root=Path("/benchmark-identity"), **overrides)
    )
    expected["content_redaction_enabled"] = expected.pop("redact_secrets")
    return expected


def _validate_config(
    config: Mapping[str, Any], *, models: Mapping[str, Any],
) -> None:
    for field in (
        "embeddings", "graph_multihop", "no_dream", "dream_per_session",
        "dump_context", "sim", "label_free_answer_path", "scored_run",
        "exploratory_label_steering", "exploratory_non_comparable",
    ):
        if not isinstance(config.get(field), bool):
            raise _fail(f"config {field} is malformed")
    if config.get("probe_mode") != "recall":
        raise _fail("probe mode is not recall")
    if config.get("start_role") not in {"user", "assistant"}:
        raise _fail("start-role is malformed")
    for field in (
        "sample", "seed", "workers", "top_k", "pipeline_search_multiplier",
        "indexing_max_cycles",
    ):
        if type(config.get(field)) is not int:
            raise _fail(f"config {field} is malformed")
    if (
        config["sample"] < 0
        or config["workers"] <= 0
        or config["top_k"] <= 0
        or config["indexing_max_cycles"] <= 0
    ):
        raise _fail("config bounds are invalid")
    if config["pipeline_search_multiplier"] != 3:
        raise _fail("retrieval multiplier drifted")
    expected_sample_strategy = (
        "seeded-shuffle-prefix-v1"
        if config["sample"] else "seeded-shuffle-all-v1"
    )
    if config.get("sample_strategy") != expected_sample_strategy:
        raise _fail("sample strategy is inconsistent")
    _nonnegative(config.get("session_gap_days"))
    if config["session_gap_days"] <= 0:
        raise _fail("session gap is invalid")
    indexing_timeout = _nonnegative(config.get("indexing_timeout_s"))
    if indexing_timeout <= 0:
        raise _fail("indexing timeout is invalid")
    if config.get("synthetic_base_date") != "2023-01-01":
        raise _fail("synthetic date policy drifted")
    for field in ("facts", "facts_extraction", "rules_extraction"):
        if config.get(field) is not None and not isinstance(
            config.get(field), bool
        ):
            raise _fail(f"config {field} is malformed")
    if config["no_dream"] and config["dream_per_session"]:
        raise _fail("dream policy is contradictory")
    try:
        validate_extraction_canary_config_binding(
            config.get("extraction_canary"),
            config.get("effective_hymem_config"),
        )
    except Exception as exc:
        raise _fail("extraction-canary policy drifted") from exc
    if config["scored_run"] is config["sim"]:
        raise _fail("simulation/scoring posture is inconsistent")
    if config["label_free_answer_path"] is not True:
        raise _fail("answer path is not label-free")
    if config["exploratory_label_steering"] is not False:
        raise _fail("label steering posture drifted")
    if config["exploratory_non_comparable"] is not bool(
        config["sample"] or config["sim"] or config["no_dream"]
    ):
        raise _fail("exploratory comparability posture is inconsistent")
    embedding = models.get("embedding")
    if (
        not isinstance(embedding, Mapping)
        or config["embeddings"] is not embedding.get("configured")
        or (config["sim"] and config["embeddings"])
    ):
        raise _fail("embedding configuration is inconsistent")
    effective = config.get("effective_hymem_config")
    if not isinstance(effective, Mapping):
        raise _fail("effective HyMem config is absent")
    expected_aperture = {
        "message_fts_top_k": 15,
        "fts_top_k": 10,
        "graph_top_k": 10,
    }
    for field, expected in expected_aperture.items():
        if effective.get(field) != expected:
            raise _fail("effective MSC retrieval aperture drifted")
    if effective.get("aggregation_nodes_enabled") is not False:
        raise _fail("historical aggregation pin drifted")
    for field in (
        "facts_enabled", "facts_extraction_enabled",
        "rules_extraction_enabled", "graph_multihop_enabled",
        "content_redaction_enabled",
    ):
        if not isinstance(effective.get(field), bool):
            raise _fail("effective HyMem flag is malformed")
    for requested, resolved in (
        ("facts", "facts_enabled"),
        ("facts_extraction", "facts_extraction_enabled"),
        ("rules_extraction", "rules_extraction_enabled"),
        ("graph_multihop", "graph_multihop_enabled"),
    ):
        value = config.get(requested)
        if value is not None and (
            not isinstance(value, bool) or effective.get(resolved) is not value
        ):
            raise _fail("requested/effective HyMem configuration drifted")
    try:
        expected_effective = _expected_effective_hymem_config(config)
    except (TypeError, ValueError) as exc:
        raise _fail("effective HyMem identity cannot be reconstructed") from exc
    if dict(effective) != expected_effective:
        raise _fail("effective HyMem identity drifted")


def validate_msc_artifact(value: object) -> dict[str, Any]:
    """Validate and return one strict MSC envelope; reject all legacy shapes."""

    if not isinstance(value, dict) or set(value) != _ROOT_FIELDS:
        raise _fail("envelope is malformed")
    data = dict(value)
    if data.get("benchmark") != "MSC" or data.get("version") != "strict-v1":
        raise _fail("benchmark/version identity is invalid")
    _require_aware_timestamp(data.get("date"), field="run")
    _require_aware_timestamp(data.get("created_at"), field="publication")
    manifest = data.get("manifest")
    config = data.get("config")
    models = data.get("models")
    execution = data.get("execution")
    rows = data.get("per_question")
    if not all(isinstance(item, dict) for item in (
        manifest, config, models, execution,
    )) or not isinstance(rows, list):
        raise _fail("envelope components are malformed")
    if set(manifest) != _MANIFEST_FIELDS:
        raise _fail("manifest shape is malformed")
    if (
        manifest.get("schema") != STRICT_PROTOCOL_VERSION
        or manifest.get("benchmark") != "MSC"
    ):
        raise _fail("manifest identity is invalid")
    if manifest.get("run_id") != content_hash({
        key: item for key, item in manifest.items() if key != "run_id"
    }):
        raise _fail("manifest run identity is invalid")
    if config != manifest.get("config") or models != manifest.get("models"):
        raise _fail("top-level identity differs from manifest")
    if manifest.get("config_hash") != content_hash(config):
        raise _fail("config hash is invalid")
    if manifest.get("model_hash") != content_hash(models):
        raise _fail("model hash is invalid")
    for field in ("code_hash", "data_hash", "expected_ids_hash", "run_id"):
        if not isinstance(manifest.get(field), str) or not _SHA256.fullmatch(
            manifest[field]
        ):
            raise _fail(f"manifest {field} is malformed")
    if manifest.get("protocol_split") not in {"full", "dev", "holdout"}:
        raise _fail("protocol split is invalid")
    protocol_split = manifest["protocol_split"]
    if type(manifest.get("seed")) is not int:
        raise _fail("seed is malformed")
    for field in (
        "development_only", "official_split", "official_comparable",
        "label_free_answer_path", "exploratory_label_steering",
        "exploratory_non_comparable", "scored_run",
    ):
        if not isinstance(manifest.get(field), bool):
            raise _fail(f"manifest {field} is malformed")
    for field in (
        "label_free_answer_path", "exploratory_label_steering",
        "exploratory_non_comparable", "scored_run",
    ):
        if config.get(field) is not manifest[field]:
            raise _fail(f"manifest/config {field} drifted")
    if config.get("seed") != manifest["seed"]:
        raise _fail("manifest/config seed drifted")
    if manifest["official_split"] is not False or manifest[
        "official_comparable"
    ] is not False:
        raise _fail("MSC incorrectly claims an official split")
    expected_limitation = (
        "internal deterministic split, not an official benchmark split"
        if protocol_split in {"dev", "holdout"}
        else "full-set development evidence; may be test-contaminated"
    )
    if manifest.get("protocol_limitation") != expected_limitation:
        raise _fail("protocol limitation is inconsistent")
    receipt_hash = manifest.get("calibration_receipt_hash")
    if protocol_split == "full":
        if receipt_hash is not None:
            raise _fail("full split unexpectedly claims a calibration receipt")
    elif (
        not isinstance(receipt_hash, str)
        or not _SHA256.fullmatch(receipt_hash)
        or config.get("sample") != 0
    ):
        raise _fail("internal split lacks an exact calibration binding")
    expected_development_only = bool(
        protocol_split != "holdout"
        or not manifest["label_free_answer_path"]
        or manifest["exploratory_non_comparable"]
        or manifest["exploratory_label_steering"]
        or not manifest["scored_run"]
    )
    if manifest["development_only"] is not expected_development_only:
        raise _fail("development-only posture is inconsistent")
    _validate_config(config, models=models)
    scored = manifest["scored_run"]
    simulation = config["sim"]
    _validate_model_identity(models, scored=scored, simulation=simulation)
    required_canary = not simulation and not config["no_dream"]

    expected_count = manifest.get("expected_count")
    if (
        type(expected_count) is not int
        or expected_count <= 0
        or expected_count != len(rows)
    ):
        raise _fail("denominator is incomplete")
    ids: list[str] = []
    failed = missing = completed = 0
    row_canaries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise _fail(f"row {index} is malformed")
        item_id = row.get("question_id")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id != item_id.strip()
            or item_id in ids
        ):
            raise _fail(f"row {index} id is malformed")
        if row.get("id") != item_id:
            raise _fail(f"row {index} id/question_id drifted")
        if row.get("question_type") != "recall":
            raise _fail(f"row {index} question type is invalid")
        if row.get("indexing_scope_id") != f"msc:{item_id}":
            raise _fail(f"row {index} indexing scope is invalid")
        ids.append(item_id)
        if not isinstance(row.get("correct"), bool):
            raise _fail(f"row {index} verdict is malformed")
        strict_failure = row.get("strict_failure")
        failure = row.get("benchmark_failure")
        if strict_failure is None and not scored:
            strict_failure = bool(failure)
        elif not isinstance(strict_failure, bool):
            raise _fail(f"row {index} failure posture is absent")
        if strict_failure:
            if not isinstance(failure, str) or not failure or row["correct"]:
                raise _fail("failed row is inconsistent")
            failed += 1
            missing += failure == "missing_prediction"
        elif failure:
            raise _fail("completed row carries a failure")
        else:
            completed += 1
        report = row.get("extraction_canary")
        if not isinstance(report, dict):
            raise _fail("row canary evidence is absent")
        row_canaries.append(report)
    if manifest.get("expected_ids_hash") != content_hash(ids):
        raise _fail("row id order/hash is invalid")

    if set(execution) != {"counts", "segments", "checkpoint"}:
        raise _fail("execution envelope is malformed")
    counts = execution["counts"]
    segments = execution["segments"]
    checkpoint = execution["checkpoint"]
    if not isinstance(counts, dict) or not isinstance(segments, list) or not segments:
        raise _fail("execution evidence is incomplete")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"schema", "state_sha256"}
        or checkpoint.get("schema") != CHECKPOINT_VERSION
        or not isinstance(checkpoint.get("state_sha256"), str)
        or not _SHA256.fullmatch(checkpoint["state_sha256"])
    ):
        raise _fail("checkpoint digest evidence is malformed")
    expected_counts = {
        "expected": expected_count,
        "attempted": expected_count - missing,
        "unique_attempted": expected_count - missing,
        "completed": completed,
        "failed": failed,
        "missing": missing,
    }
    for field, expected in expected_counts.items():
        if type(counts.get(field)) is not int or counts[field] != expected:
            raise _fail("execution counts do not reconcile")
    total_attempts = counts.get("total_attempts")
    if type(total_attempts) is not int or total_attempts < counts["attempted"]:
        raise _fail("total attempt count is invalid")

    segment_ids: set[str] = set()
    segment_attempts = 0
    reader_successes = 0
    judge_successes = 0
    reader_successes_available = True
    judge_successes_available = True
    all_indexing_scopes: set[str] = set()
    segment_canaries: list[dict[str, Any]] = []
    saw_complete = False
    for segment in segments:
        if not isinstance(segment, dict):
            raise _fail("execution segment is malformed")
        segment_id = segment.get("segment_id")
        if (
            not isinstance(segment_id, str)
            or not segment_id
            or segment_id in segment_ids
        ):
            raise _fail("execution segment id is invalid")
        segment_ids.add(segment_id)
        status = segment.get("status")
        if status not in {"running", "complete"}:
            raise _fail("execution segment status is invalid")
        saw_complete = saw_complete or status == "complete"
        attempted = segment.get("attempted_attempts")
        if type(attempted) is not int or attempted < 0:
            raise _fail("segment attempt count is malformed")
        segment_attempts += attempted
        _nonnegative(segment.get("elapsed_s"))
        if segment.get("model_identities") != models:
            raise _fail("segment model identity drifted")
        report = segment.get("extraction_canary")
        if not isinstance(report, dict):
            raise _fail("segment extraction canary is absent")
        historical_preflight = False
        terminal_zero_work = False
        if report.get("status") == "passed":
            mode = "required"
            if not required_canary:
                raise _fail("segment canary mode conflicts with configuration")
        elif report.get("status") == "failed":
            mode = "failed"
            historical_preflight = True
            if not required_canary:
                raise _fail("segment canary mode conflicts with configuration")
        elif report.get("status") == "pending":
            mode = "pending"
            historical_preflight = True
            if not required_canary:
                raise _fail("segment canary mode conflicts with configuration")
        elif report.get("status") == "invalid_report":
            mode = "invalid_report"
            historical_preflight = True
            if not required_canary:
                raise _fail("segment canary mode conflicts with configuration")
        elif report.get("status") == "not_run_no_pending":
            mode = "no_pending_work"
            terminal_zero_work = True
        elif report.get("status") == "skipped_non_comparable":
            mode = report.get("skip_reason")
            expected_skip = "simulation" if simulation else "no_dream"
            if required_canary or mode != expected_skip:
                raise _fail("segment canary mode conflicts with configuration")
        else:
            raise _fail("segment extraction canary state is invalid")
        try:
            if mode == "invalid_report":
                _validate_invalid_canary_sentinel(report)
            else:
                validate_extraction_canary_report(
                    report,
                    expected_mode=mode,
                    expected_client=(
                        models["memory_pipeline"]
                        if mode in {"required", "failed"} else None
                    ),
                    require_client_closed=mode in {"required", "failed"},
                    expected_prompt_version=config[
                        "effective_hymem_config"
                    ]["prompt_version"],
                )
        except Exception as exc:
            raise _fail("segment extraction canary is invalid") from exc
        segment_canaries.append(report)
        reader_usage = segment.get("reader_usage")
        judge_usage = segment.get("judge_usage")
        pipeline_usage = segment.get("memory_pipeline_usage")
        reader_calls = _validate_llm_usage(reader_usage)
        judge_calls = _validate_llm_usage(judge_usage)
        _validate_llm_usage(pipeline_usage)
        if reader_calls is None:
            reader_successes_available = False
        else:
            reader_successes += reader_calls
        if judge_calls is None:
            judge_successes_available = False
        else:
            judge_successes += judge_calls
        _validate_embedding_usage(
            segment.get("embedding_usage"),
            identity=models["embedding"],
            attempted=attempted,
        )
        indexing_scopes: set[str] = set()
        for field in ("indexing_runs", "indexing_failures"):
            evidence = segment.get(field)
            if not isinstance(evidence, list):
                raise _fail(f"segment {field} is malformed")
            for item in evidence:
                if not isinstance(item, Mapping) or set(item) != {
                    "scope_id", "summary",
                }:
                    raise _fail("segment indexing evidence is malformed")
                scope_id = item.get("scope_id")
                if (
                    not isinstance(scope_id, str)
                    or not scope_id.startswith("msc:")
                    or not scope_id.removeprefix("msc:")
                    or scope_id.removeprefix("msc:") not in ids
                    or scope_id in indexing_scopes
                    or not isinstance(item.get("summary"), Mapping)
                    or not item["summary"]
                ):
                    raise _fail("segment indexing evidence is inconsistent")
                indexing_scopes.add(scope_id)
                all_indexing_scopes.add(scope_id)
        if len(indexing_scopes) != attempted:
            raise _fail("segment indexing evidence does not cover its attempts")
        if historical_preflight:
            if status != "running" or attempted != 0:
                raise _fail("historical preflight segment claims benchmark attempts")
            _require_zero_llm_work(reader_usage)
            _require_zero_llm_work(judge_usage)
            _require_zero_llm_work(pipeline_usage)
            if segment["indexing_runs"] or segment["indexing_failures"]:
                raise _fail("historical preflight segment claims indexing work")
        if terminal_zero_work:
            if status != "complete" or attempted != 0:
                raise _fail("terminal segment claims benchmark attempts")
            _require_zero_llm_work(reader_usage)
            _require_zero_llm_work(judge_usage)
            _require_zero_llm_work(pipeline_usage)
            _require_zero_embedding_work(segment["embedding_usage"])
    if not saw_complete or segment_attempts != total_attempts:
        raise _fail("execution segments do not reconcile")
    if scored and (
        not reader_successes_available
        or not judge_successes_available
        or reader_successes < completed
        or judge_successes < completed
    ):
        raise _fail("live model calls are below completed row count")
    if all_indexing_scopes != {f"msc:{item_id}" for item_id in ids}:
        raise _fail("indexing scopes do not reconcile to the result ledger")
    if not all(any(report == candidate for candidate in segment_canaries)
               for report in row_canaries):
        raise _fail("row canary is not owned by an execution segment")
    for report in row_canaries:
        if required_canary and report.get("status") == "passed":
            continue
        if (
            report.get("status") == "skipped_non_comparable"
            and report.get("skip_reason") == (
                "simulation" if simulation else "no_dream"
            )
            and not required_canary
        ):
            continue
        raise _fail("row is bound to a non-successful canary segment")

    expected_scores = None
    expected_accuracy = None
    if scored:
        accuracy = sum(row["correct"] for row in rows) / len(rows)
        expected_scores = {
            "recall": {"accuracy": accuracy, "count": len(rows)},
            "OVERALL": {"accuracy": accuracy, "count": len(rows)},
        }
        expected_accuracy = accuracy
    if data.get("scores") != expected_scores:
        raise _fail("declared scores do not reconcile")
    if data.get("strict_accuracy") != expected_accuracy:
        raise _fail("strict accuracy does not reconcile")
    if data.get("result_digest") != content_hash(rows):
        raise _fail("result digest is invalid")
    if not isinstance(data.get("legacy_bare_out"), bool):
        raise _fail("legacy sidecar declaration is malformed")
    return data


def load_msc_artifact(path: str | Path) -> dict[str, Any]:
    """Read an archive or a verified sibling latest pointer, then validate it."""

    return validate_msc_artifact(read_artifact_or_pointer(path))


def discover_msc_archives(directory: str | Path) -> list[Path]:
    """Return only authoritative archives; never mutable pointers/sidecars."""

    root = Path(directory)
    if not root.is_dir():
        raise BenchmarkIntegrityError("MSC artifact directory does not exist")
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and _ARCHIVE.fullmatch(path.name)
    )


def scan_msc_archives(directory: str | Path) -> list[dict[str, Any]]:
    """Validate every default-discovered archive and return its envelopes."""

    return [load_msc_artifact(path) for path in discover_msc_archives(directory)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate authoritative strict MSC recall artifacts"
    )
    parser.add_argument("files", nargs="*", help="archive or msc-latest.json")
    parser.add_argument(
        "--directory", default="msc_results",
        help="default archive directory when no files are supplied",
    )
    args = parser.parse_args()
    paths = [Path(item) for item in args.files]
    if not paths:
        paths = discover_msc_archives(args.directory)
    results = []
    for path in paths:
        artifact = load_msc_artifact(path)
        results.append({
            "archive": path.name,
            "run_id": artifact["manifest"]["run_id"],
            "count": artifact["manifest"]["expected_count"],
            "scored": artifact["manifest"]["scored_run"],
            "accuracy": artifact["strict_accuracy"],
        })
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
