#!/usr/bin/env python3
"""Run registry for BEAM benchmark executions.

Same model as lme_registry.py: one row per run JSON, recording date,
scores, and the effective flags (recorded vs analyst-set, never guessed).
DB is restart-safe: /home/node/.hermes/benchmarks/beam_runs.db.

BEAM run JSONs come in three dialects:
  A. hardcoded saves: {date, config{scales,sample,top_k,answer_model,
     judge_model}, scores{ABILITY: pct}}               (v13/v14)
  B. same shape but NO date                             (v15; date: None)
  C. adapter-native: {metadata{date,answer_model,...}, summary{scale:
     {ABILITY: frac}}}                                  (v16+)
Current strict artifacts record immutable model/config identities, effective
HyMem levers, durable rows, and execution segments.  Legacy dialects predate
that evidence; fields they did not record remain NULL unless analyst-supplied
via ``--set``.

Usage:
  beam_registry.py ingest [FILE ...] [--set k=v ...]
  beam_registry.py list [--limit N] [--flag COL]
  beam_registry.py query "SQL"
"""

from __future__ import annotations

import os
import sys
import math
import re
import json
from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

try:  # package import (tests): benchmarks.run_registry
    from . import run_registry as rr
    from .strictness import content_hash
    from .run_registry import (
        DEFAULT_REGISTRY_DIR,
        _coerce,
        cmd_ingest,
        cmd_list,
        cmd_query,
        connect,
    )
except (ImportError, ValueError):  # direct CLI: python benchmarks/beam_registry.py
    import run_registry as rr
    from strictness import content_hash
    from run_registry import (
        DEFAULT_REGISTRY_DIR,
        _coerce,
        cmd_ingest,
        cmd_list,
        cmd_query,
        connect,
    )

DB_ENV = "BEAM_REGISTRY_DB"

BEAM_COLUMNS = [
    ("archive", "TEXT"), ("kind", "TEXT"), ("run_date", "TEXT"),
    ("source_date", "TEXT"), ("scale", "TEXT"),
    ("run_id", "TEXT"), ("protocol_split", "TEXT"),
    ("development_only", "INTEGER"),
    ("exploratory_non_comparable", "INTEGER"),
    ("label_free_answer_path", "INTEGER"),
    ("judge_protocol", "TEXT"),
    ("official_judge_protocol_match", "INTEGER"),
    ("dataset_revisions_complete", "INTEGER"),
    ("sample", "INTEGER"), ("top_k", "INTEGER"),
    ("context_memories", "INTEGER"),
    ("answer_model", "TEXT"), ("judge_model", "TEXT"),
    # Strict artifacts bind these levers to the effective HyMem config.  Older
    # dialects retain NULL unless the value was actually recorded/overridden.
    ("embeddings", "INTEGER"), ("facts", "INTEGER"),
    ("facts_extraction", "INTEGER"), ("graph_multihop", "INTEGER"),
    ("no_dream", "INTEGER"), ("distill", "INTEGER"),
    ("episode_granularity_enabled", "INTEGER"),
    ("aggregation_nodes_enabled", "INTEGER"),
    ("value_supersession_enabled", "INTEGER"),
    # scores (percent 0-100; dialect C converted from fraction)
    ("overall", "REAL"),
    ("ability_abs", "REAL"), ("ability_cr", "REAL"), ("ability_eo", "REAL"),
    ("ability_ie", "REAL"), ("ability_if", "REAL"), ("ability_ku", "REAL"),
    ("ability_mr", "REAL"), ("ability_pf", "REAL"), ("ability_sum", "REAL"),
    ("ability_tr", "REAL"),
    ("count", "INTEGER"), ("answer_calls", "INTEGER"),
    ("judge_calls", "INTEGER"), ("total_tokens", "INTEGER"),
    ("elapsed_s", "REAL"),
]

BEAM_ABILITIES = ("ABS", "CR", "EO", "IE", "IF", "KU", "MR", "PF", "SUM", "TR")
VALID_BEAM_SCALES = ("100K", "500K", "1M", "10M")
OFFICIAL_BEAM_DENOMINATORS = {
    "100K": {"conversations": 20, "questions": 400},
    "500K": {"conversations": 35, "questions": 700},
    "1M": {"conversations": 35, "questions": 700},
    "10M": {"conversations": 10, "questions": 200},
}
BEAM_REPO = "Mohammadta/BEAM"
BEAM_REPO_10M = "Mohammadta/BEAM-10M"
OFFICIAL_JUDGE_MODEL = "gpt-4.1-mini"
OFFICIAL_JUDGE_BASE_URL = "https://api.openai.com/v1"
BEAM_UPSTREAM_COMMIT = "b2da22eac88bb0874c64665f13457eb99835774a"
BEAM_OFFICIAL_JUDGE_PROMPT_HASH = (
    "sha256:593373c642a288a7b590577d8a8fc92c3f9a2b70e2f64ad6e59a040a6c56b7f5"
)
BEAM_OFFICIAL_EVALUATOR_URL = (
    "https://github.com/mohammadtavakoli78/BEAM/blob/"
    f"{BEAM_UPSTREAM_COMMIT}/src/evaluation/compute_metrics.py"
)
BEAM_OFFICIAL_PROMPT_URL = (
    "https://github.com/mohammadtavakoli78/BEAM/blob/"
    f"{BEAM_UPSTREAM_COMMIT}/src/prompts.py"
)
RESERVED_CHAT_BODY_KEYS = frozenset({
    "model", "messages", "temperature", "max_tokens",
})
BEAM_PROTOCOL_LIMITATIONS = {
    "full": "full-set development evidence; may be test-contaminated",
    "dev": "internal deterministic split, not an official benchmark split",
    "holdout": "internal deterministic split, not an official benchmark split",
}
STRICT_HYMEM_BOOLEAN_FIELDS = (
    "facts_enabled",
    "facts_extraction_enabled",
    "graph_multihop_enabled",
    "episode_granularity_enabled",
    "aggregation_nodes_enabled",
    "value_supersession_enabled",
)

BEAM_OVERRIDES = {
    "sample", "top_k", "context_memories", "answer_model", "judge_model",
    "embeddings", "facts", "facts_extraction", "graph_multihop", "no_dream",
    "distill", "episode_granularity_enabled", "aggregation_nodes_enabled",
    "value_supersession_enabled",
}

# record-doc may also carry scores/counts (they come from a documented
# run, not from a file we parsed).
BEAM_DOC_OVERRIDES = BEAM_OVERRIDES | {
    "overall", "ability_abs", "ability_cr", "ability_eo", "ability_ie",
    "ability_if", "ability_ku", "ability_mr", "ability_pf", "ability_sum",
    "ability_tr", "count", "answer_calls", "judge_calls", "run_date",
}

# §6.5: beam artifacts are split across two dirs -- the v13-v16 archives
# sit beside the registry, the current results_*.json land in the beam
# run output dir.  Override with BEAM_ARTIFACT_DIRS (os.pathsep-separated).
ARTIFACT_DIRS = tuple(
    Path(p) for p in os.environ["BEAM_ARTIFACT_DIRS"].split(os.pathsep)
) if os.environ.get("BEAM_ARTIFACT_DIRS") else (
    DEFAULT_REGISTRY_DIR,
    Path.home() / "hymem_beam",
)

SPEC = {
    "db_file": "beam_runs.db",
    "artifact_dirs": ARTIFACT_DIRS,
    "columns": BEAM_COLUMNS,
    "overrides": BEAM_OVERRIDES,
    "patterns": ("beam-v*.json", "beam-*.json", "results_*.json"),
    "excludes": ("latest", "comparison"),
    "kind_class": "beam",
    # §6 stamp policy: beam stems carry no \\d{8}T\\d{6}Z stamp
    # (beam-v14-preference-fix.json) -> recording NULL is the domain
    # truth, not a defect.  Rejudge artifacts carry stamps (source +
    # exec) and read the source date from their own rejudged_from.
    "stamp_policy": "optional",
    "gap_label": "levers (facts/facts_extraction/embeddings/no_dream/aggregation)",
    "gap_note": (
        "SELECT COUNT(*) FROM runs WHERE facts IS NULL AND facts_extraction IS NULL "
        "AND embeddings IS NULL AND no_dream IS NULL "
        "AND aggregation_nodes_enabled IS NULL"
    ),
}


def _beam_kind(name: str) -> str:
    if "rejudge" in name:
        return "rejudge"
    if name.startswith("beam-v") or (
        name.startswith("results_") and "-strict-" in name
    ):
        return "archive"
    return "variant"


def _number(value, *, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        or (integer and not float(value).is_integer())
    ):
        return None
    return int(value) if integer else float(value)


def _normalize_scales(value, *, label: str) -> list[str]:
    """Normalize scalar/list scale declarations without string slicing.

    Older artifacts used both ``"100K"`` and ``["100K"]``.  Treating the
    scalar as a sequence produced the plausible-looking but false scale
    ``"1"``.  Mixed, blank, duplicate, and non-string declarations are
    integrity errors rather than values the registry can safely guess.
    """

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"{label} must be a scale string or list of strings")
    if (
        not values
        or any(
            not isinstance(scale, str)
            or not scale.strip()
            or scale != scale.strip()
            for scale in values
        )
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{label} contains malformed or duplicate scales")
    return values


def _legacy_percent_scores(raw) -> tuple[dict[str, float | None], dict]:
    """Validate legacy already-percent scores; never coerce bad data to 0."""

    if not isinstance(raw, dict):
        raise ValueError("legacy BEAM scores must be an object")
    scores: dict[str, float | None] = {}
    invalid: list[str] = []
    for metric, value in raw.items():
        if not isinstance(metric, str) or not metric:
            raise ValueError("legacy BEAM score metric is malformed")
        if value is None:
            scores[metric] = None
            continue
        number = _number(value)
        if number is None or number > 100:
            invalid.append(metric)
            scores[metric] = None
        else:
            scores[metric] = number
    return scores, {
        "score_protocol": "legacy_percent",
        "invalid_score_metrics": invalid,
    }


def _legacy_fraction_summary(
    raw_summary, raw_counts, declared_scales,
) -> tuple[list[str], dict[str, float | None], dict]:
    """Read dialect-C fractions without silently selecting one scale.

    A single scale can be projected directly.  Multiple scales are aggregated
    only when every scale supplies the same metric set and a trustworthy,
    positive integral count for each metric.  Otherwise aggregate score
    columns remain NULL and the complete per-scale evidence stays in extras.
    """

    if not isinstance(raw_summary, dict) or not raw_summary:
        raise ValueError("legacy BEAM summary must be a non-empty object")
    scales = _normalize_scales(list(raw_summary), label="legacy summary scales")
    if declared_scales is not None:
        declared = _normalize_scales(
            declared_scales, label="legacy declared scales"
        )
        if declared != scales:
            raise ValueError(
                "legacy BEAM declared scales differ from summary scale order"
            )
    normalized: dict[str, dict[str, float | None]] = {}
    invalid: list[str] = []
    for scale in scales:
        block = raw_summary.get(scale)
        if not isinstance(block, dict) or not block:
            raise ValueError(f"legacy BEAM summary for {scale} is malformed")
        normalized[scale] = {}
        for metric, value in block.items():
            if not isinstance(metric, str) or not metric:
                raise ValueError("legacy BEAM summary metric is malformed")
            if value is None:
                normalized[scale][metric] = None
                continue
            number = _number(value)
            if number is None or number > 1:
                invalid.append(f"{scale}/{metric}")
                normalized[scale][metric] = None
            else:
                normalized[scale][metric] = number

    disclosure = {
        "score_protocol": "legacy_fraction",
        "per_scale_summary": raw_summary,
        "per_scale_counts": raw_counts,
        "invalid_score_metrics": invalid,
        "aggregate_available": False,
        "aggregate_reason": None,
    }
    if invalid:
        disclosure["aggregate_reason"] = "malformed_score_values"
        return scales, {}, disclosure
    if len(scales) == 1:
        disclosure["aggregate_available"] = True
        disclosure["aggregate_reason"] = "single_scale"
        return scales, {
            metric: value * 100 if value is not None else None
            for metric, value in normalized[scales[0]].items()
        }, disclosure

    metric_sets = [set(normalized[scale]) for scale in scales]
    if any(metrics != metric_sets[0] for metrics in metric_sets[1:]):
        disclosure["aggregate_reason"] = "partial_metric_coverage"
        return scales, {}, disclosure
    if not isinstance(raw_counts, dict) or set(raw_counts) != set(scales):
        disclosure["aggregate_reason"] = "counts_unavailable"
        return scales, {}, disclosure

    output: dict[str, float | None] = {}
    for metric in sorted(metric_sets[0]):
        numerator = 0.0
        denominator = 0
        for scale in scales:
            counts = raw_counts.get(scale)
            if not isinstance(counts, dict) or set(counts) != metric_sets[0]:
                disclosure["aggregate_reason"] = "partial_count_coverage"
                return scales, {}, disclosure
            count = _number(counts.get(metric), integer=True)
            if count is None or count <= 0:
                disclosure["aggregate_reason"] = "malformed_counts"
                return scales, {}, disclosure
            score = normalized[scale][metric]
            if score is None:
                disclosure["aggregate_reason"] = "null_score"
                return scales, {}, disclosure
            numerator += score * count
            denominator += count
        output[metric] = numerator / denominator * 100
    disclosure["aggregate_available"] = True
    disclosure["aggregate_reason"] = "weighted_by_recorded_counts"
    return scales, output, disclosure


def _manifested_embedding_execution_identity(identity: dict) -> tuple:
    """Translate public backend configuration to its metered client identity."""

    backend = identity.get("backend")
    model = identity.get("model")
    dimension = identity.get("dimension")
    if backend == "none":
        return "none", None, None
    if backend == "local-hash":
        return "local_feature_hash", model, dimension
    if backend == "openai-compatible":
        try:
            from hymem.contrib.openai_embedding_client import (
                openai_compatible_embedding_identity,
            )
            observed_model = openai_compatible_embedding_identity(
                identity.get("base_url"), model
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("strict BEAM embedding manifest identity is malformed") from exc
        return "openai_compatible", observed_model, dimension
    raise ValueError("strict BEAM embedding backend is malformed")


def _official_judge_identity_matches(config: dict, models: dict) -> bool:
    """Recompute official protocol identity instead of trusting its flag."""

    judge = models.get("judge")
    if not isinstance(judge, dict):
        return False
    temperature = judge.get("temperature")
    return bool(
        config.get("judge_protocol") == "official"
        and judge.get("protocol") == "official"
        and judge.get("provider") == "openai"
        and judge.get("model") == OFFICIAL_JUDGE_MODEL
        and isinstance(judge.get("base_url"), str)
        and judge["base_url"].rstrip("/") == OFFICIAL_JUDGE_BASE_URL
        and isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and math.isfinite(float(temperature))
        and float(temperature) == 0.0
        and judge.get("max_tokens") is None
        and judge.get("extra_body") == {}
        and judge.get("upstream_commit") == BEAM_UPSTREAM_COMMIT
        and judge.get("prompt_hash") == BEAM_OFFICIAL_JUDGE_PROMPT_HASH
        and config.get("official_judge_upstream_commit") == BEAM_UPSTREAM_COMMIT
        and config.get("official_judge_prompt_hash")
        == BEAM_OFFICIAL_JUDGE_PROMPT_HASH
    )


def _parse_beam_judge_json(raw: str) -> tuple[dict | None, str]:
    """Mirror BEAM adapter extraction so raw and parsed evidence stay bound."""

    if not isinstance(raw, str) or not raw:
        return None, "unreadable"
    flat = raw.replace("\n", " ")
    naive = re.search(r"\{[^}]+\}", flat)
    if naive:
        try:
            value = json.loads(naive.group())
            return value if isinstance(value, dict) else None, "ok"
        except (TypeError, ValueError):
            pass
    depth = 0
    start = None
    in_string = False
    escaped = False
    for index, character in enumerate(flat):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    value = json.loads(flat[start:index + 1])
                    return value if isinstance(value, dict) else None, "recovered"
                except (TypeError, ValueError):
                    start = None
    return None, "unreadable"


def _validate_safe_endpoint(value, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"strict BEAM {label} endpoint is malformed")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"strict BEAM {label} endpoint is unsafe or ambiguous")


def _validate_model_identity(value, *, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"strict BEAM {label} model identity is malformed")
    for field in ("provider", "model", "base_url"):
        field_value = value.get(field)
        if (
            not isinstance(field_value, str)
            or not field_value.strip()
            or field_value != field_value.strip()
        ):
            raise ValueError(
                f"strict BEAM {label} {field} identity is malformed"
            )
    _validate_safe_endpoint(value["base_url"], label=label)


def _validate_prereg(value, *, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError("strict BEAM comparable run lacks pre-registration")
        return
    if not isinstance(value, dict):
        raise ValueError("strict BEAM pre-registration receipt is malformed")
    if set(value) != {"path", "commit", "blob", "committed_at", "code_commit"}:
        raise ValueError("strict BEAM pre-registration receipt is incomplete")
    path = value.get("path")
    if (
        not isinstance(path, str)
        or not path.strip()
        or path != path.strip()
        or "\\" in path
    ):
        raise ValueError("strict BEAM pre-registration path is malformed")
    parsed_path = PurePosixPath(path)
    if parsed_path.is_absolute() or any(part in {"", ".", ".."} for part in parsed_path.parts):
        raise ValueError("strict BEAM pre-registration path is not safe and relative")
    for field in ("commit", "blob", "code_commit"):
        digest = value.get(field)
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{40,64}", digest) is None
        ):
            raise ValueError(
                f"strict BEAM pre-registration {field} is malformed"
            )
    committed_at = value.get("committed_at")
    if not isinstance(committed_at, str) or not committed_at.strip():
        raise ValueError("strict BEAM pre-registration timestamp is malformed")
    try:
        parsed_time = datetime.fromisoformat(committed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("strict BEAM pre-registration timestamp is malformed") from exc
    if parsed_time.tzinfo is None:
        raise ValueError("strict BEAM pre-registration timestamp lacks timezone")


def _validate_embedding_config(identity: dict) -> None:
    expected_fields = {
        "configured", "backend", "model", "base_url", "dimension", "quality",
        "network_free", "fallback_policy", "fallback_reason",
    }
    if not isinstance(identity, dict) or set(identity) != expected_fields:
        raise ValueError("strict BEAM embedding configuration is malformed")
    configured = identity.get("configured")
    network_free = identity.get("network_free")
    if not isinstance(configured, bool) or not isinstance(network_free, bool):
        raise ValueError("strict BEAM embedding booleans are malformed")
    backend = identity.get("backend")
    if backend == "none":
        expected = {
            "configured": False, "backend": "none", "model": None,
            "base_url": None, "dimension": None, "quality": "none",
            "network_free": True, "fallback_policy": "none",
            "fallback_reason": None,
        }
        if identity != expected:
            raise ValueError("strict BEAM disabled embedding identity is inconsistent")
        return
    dimension = identity.get("dimension")
    if (
        configured is not True
        or isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        or not isinstance(identity.get("model"), str)
        or not identity["model"].strip()
        or identity["model"] != identity["model"].strip()
        or identity.get("fallback_reason") is not None
    ):
        raise ValueError("strict BEAM enabled embedding identity is malformed")
    if backend == "local-hash":
        if (
            identity.get("base_url") != "local://feature-hash"
            or identity.get("quality") != "lexical-feature-hash"
            or network_free is not True
            or identity.get("fallback_policy") != "none"
        ):
            raise ValueError("strict BEAM local embedding identity is inconsistent")
        return
    if backend == "openai-compatible":
        if (
            identity.get("quality") != "semantic"
            or network_free is not False
            or identity.get("fallback_policy") != "fail-closed"
        ):
            raise ValueError("strict BEAM remote embedding identity is inconsistent")
        _validate_safe_endpoint(identity.get("base_url"), label="embedding")
        return
    raise ValueError("strict BEAM embedding backend is unsupported")


def _validate_effective_hymem_config(config: dict) -> dict:
    """Return the typed effective HyMem levers used for registry evidence.

    Requested CLI overrides are not sufficient run identity: defaults can
    change between revisions.  Strict archives therefore have to retain the
    effective values that actually configured the store, and requested facts
    overrides must agree with those values when explicitly supplied.
    """

    effective = config.get("effective_hymem_config")
    if not isinstance(effective, dict):
        raise ValueError("strict BEAM effective HyMem config is absent")
    for field in STRICT_HYMEM_BOOLEAN_FIELDS:
        if not isinstance(effective.get(field), bool):
            raise ValueError(
                f"strict BEAM effective HyMem config field {field!r} "
                "must be boolean"
            )
    for requested, resolved in (
        ("facts", "facts_enabled"),
        ("facts_extraction", "facts_extraction_enabled"),
        ("graph_multihop", "graph_multihop_enabled"),
    ):
        if requested not in config or config.get(requested) is None:
            continue
        value = config.get(requested)
        if not isinstance(value, bool) or value is not effective[resolved]:
            raise ValueError(
                f"strict BEAM requested {requested} differs from effective "
                f"HyMem {resolved}"
            )
    for fixed in ("no_dream", "distill"):
        if fixed in config and config.get(fixed) is not False:
            raise ValueError(
                f"strict BEAM fixed protocol field {fixed} must be false"
            )
    return effective


def _validate_chat_extra_body(value, *, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"strict BEAM {label} extra_body must be an object")
    reserved = sorted(RESERVED_CHAT_BODY_KEYS & set(value))
    if reserved:
        raise ValueError(
            f"strict BEAM {label} extra_body overrides reserved fields: "
            f"{reserved}"
        )
    return value


def _validate_memory_pipeline_identity(value) -> None:
    _validate_model_identity(value, label="memory pipeline")
    if value.get("provider") != "openai-compatible":
        raise ValueError("strict BEAM memory pipeline provider is inconsistent")
    mode = value.get("thinking_mode")
    if mode not in {"auto", "disabled", "off", "enabled"}:
        raise ValueError("strict BEAM memory pipeline thinking mode is malformed")
    parsed = urlsplit(value["base_url"])
    sends_thinking = bool(
        mode == "disabled"
        or (
            mode == "auto"
            and (
                "deepseek" in (parsed.hostname or "").casefold()
                or "deepseek" in value["model"].casefold()
            )
        )
    )
    expected_extra = (
        {"thinking": {"type": "disabled"}} if sends_thinking else {}
    )
    if value.get("effective_extra_body") != expected_extra:
        raise ValueError(
            "strict BEAM memory pipeline effective extra_body is inconsistent"
        )


def _nonnegative_number(value, *, integer: bool = False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        or (integer and not float(value).is_integer())
    ):
        return None
    return int(value) if integer else float(value)


def _validate_available_value(
    snapshot: dict, field: str, availability: str, *, label: str,
    integer: bool = False,
) -> int | float | None:
    available = snapshot.get(availability)
    if not isinstance(available, bool):
        raise ValueError(f"strict BEAM {label} {availability} is malformed")
    value = snapshot.get(field)
    if not available:
        if value is not None:
            raise ValueError(
                f"strict BEAM {label} {field} claims an unavailable value"
            )
        return None
    normalized = _nonnegative_number(value, integer=integer)
    if normalized is None:
        raise ValueError(f"strict BEAM {label} {field} is malformed")
    return normalized


def _validate_usage_snapshot(
    snapshot, *, label: str, complete: bool,
) -> dict[str, int | float | None]:
    """Validate an adapter-emitted reader/judge/pipeline usage snapshot."""

    if not isinstance(snapshot, dict):
        raise ValueError(f"strict BEAM {label} usage is absent")
    calls = _validate_available_value(
        snapshot, "calls", "calls_available", label=label, integer=True,
    )
    attempts = _validate_available_value(
        snapshot, "request_attempts", "request_attempts_available",
        label=label, integer=True,
    )
    successes = _validate_available_value(
        snapshot, "successful_responses", "successful_responses_available",
        label=label, integer=True,
    )
    token_available = snapshot.get("token_usage_available")
    if not isinstance(token_available, bool):
        raise ValueError(f"strict BEAM {label} token availability is malformed")
    token_values = []
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = snapshot.get(field)
        if token_available:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"strict BEAM {label} {field} is malformed")
            token_values.append(value)
        elif value is not None:
            raise ValueError(
                f"strict BEAM {label} {field} claims unavailable token usage"
            )
    if token_available and token_values[2] != token_values[0] + token_values[1]:
        raise ValueError(f"strict BEAM {label} token totals do not reconcile")
    latency = _validate_available_value(
        snapshot, "latency_s", "latency_available", label=label,
    )
    _validate_available_value(
        snapshot, "cost_usd", "cost_available", label=label,
    )
    if complete and any(value is None for value in (calls, attempts, successes, latency)):
        raise ValueError(
            f"strict BEAM complete {label} usage lacks measured counters"
        )
    if calls is not None and successes is not None and calls != successes:
        raise ValueError(
            f"strict BEAM {label} calls differ from successful responses"
        )
    if attempts is not None and successes is not None and attempts < successes:
        raise ValueError(
            f"strict BEAM {label} attempts are below successful responses"
        )
    return {
        "calls": calls,
        "attempts": attempts,
        "successes": successes,
        "latency_s": latency,
    }


def _validate_embedding_usage(
    usage, *, identity: dict, status: str, require_identity: bool,
) -> None:
    if not isinstance(usage, dict):
        raise ValueError("strict BEAM execution lacks embedding usage")
    complete = status == "complete"
    configured = usage.get("configured")
    if not isinstance(configured, bool) or configured is not identity["configured"]:
        raise ValueError("strict BEAM embedding configured state drifted")
    expected_backend, expected_model, expected_dimension = (
        _manifested_embedding_execution_identity(identity)
    )
    expected_quality = {
        "none": "none",
        "local-hash": "lexical",
        "openai-compatible": "semantic",
    }[identity["backend"]]
    expected_network_free = identity["backend"] != "openai-compatible"
    unavailable_running = bool(
        not require_identity
        and identity["configured"]
        and usage.get("backend") == "unavailable"
        and usage.get("quality") == "none"
        and usage.get("network_free") is None
        and usage.get("model") is None
        and usage.get("dimension") is None
        and usage.get("identity_available") is False
    )
    if not unavailable_running and (
        usage.get("backend") != expected_backend
        or usage.get("quality") != expected_quality
        or usage.get("network_free") is not expected_network_free
        or usage.get("model") != expected_model
        or usage.get("dimension") != expected_dimension
        or isinstance(usage.get("dimension"), bool)
    ):
        raise ValueError("strict BEAM embedding execution identity drifted")
    for field in ("identity_available", "identity_consistent"):
        if field in usage and not isinstance(usage.get(field), bool):
            raise ValueError(f"strict BEAM embedding {field} is malformed")
    if require_identity and not (
        usage.get("identity_consistent") is True
        or usage.get("identity_available") is True
    ):
        raise ValueError("strict BEAM embedding identity is unavailable")
    if identity["configured"] and require_identity and usage.get("identity_consistent") is not True:
        raise ValueError("strict BEAM embedding identity is inconsistent")
    if "instances" in usage:
        instances = usage.get("instances")
        if isinstance(instances, bool) or not isinstance(instances, int) or instances <= 0:
            raise ValueError("strict BEAM embedding instance count is malformed")
    calls = _validate_available_value(
        usage, "calls", "calls_available", label="embedding", integer=True,
    )
    attempts = _validate_available_value(
        usage, "request_attempts", "request_attempts_available",
        label="embedding", integer=True,
    )
    successes = _validate_available_value(
        usage, "successful_responses", "successful_responses_available",
        label="embedding", integer=True,
    )
    _validate_available_value(
        usage, "input_count", "input_count_available",
        label="embedding", integer=True,
    )
    _validate_available_value(
        usage, "input_characters", "input_characters_available",
        label="embedding", integer=True,
    )
    latency = _validate_available_value(
        usage, "latency_s", "latency_available", label="embedding",
    )
    tokens_available = usage.get("provider_token_usage_available")
    if not isinstance(tokens_available, bool):
        raise ValueError("strict BEAM embedding token availability is malformed")
    for field in ("prompt_tokens", "total_tokens"):
        value = usage.get(field)
        if tokens_available:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"strict BEAM embedding {field} is malformed")
        elif value is not None:
            raise ValueError(
                f"strict BEAM embedding {field} claims unavailable token usage"
            )
    if tokens_available and usage["total_tokens"] != usage["prompt_tokens"]:
        raise ValueError("strict BEAM embedding token totals do not reconcile")
    _validate_available_value(
        usage, "cost_usd", "cost_available", label="embedding",
    )
    if complete and any(value is None for value in (calls, attempts, successes, latency)):
        raise ValueError("strict BEAM complete embedding usage lacks measured counters")
    if identity["backend"] == "openai-compatible":
        if calls is not None and successes is not None and calls != successes:
            raise ValueError("strict BEAM embedding calls differ from successes")
        if attempts is not None and successes is not None and attempts < successes:
            raise ValueError("strict BEAM embedding attempts are below successes")
    elif tokens_available:
        raise ValueError("strict BEAM network-free embedding claims provider tokens")


def _validate_official_row_evidence(row: dict, *, index: int) -> None:
    """Verify the criterion-by-criterion evidence emitted by the adapter."""

    rubric = row.get("rubric")
    scores = row.get("scores")
    criteria = row.get("judge_criterion_results")
    if (
        not isinstance(rubric, list)
        or not rubric
        or any(
            not isinstance(item, str) or not item.strip()
            for item in rubric
        )
        or not isinstance(scores, list)
        or len(scores) != len(rubric)
        or not isinstance(criteria, list)
        or len(criteria) != len(rubric)
    ):
        raise ValueError(
            f"strict BEAM official row {index} has incomplete rubric evidence"
        )
    normalized_scores: list[float] = []
    for criterion_index, (rubric_item, raw_score, criterion) in enumerate(
        zip(rubric, scores, criteria)
    ):
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(float(raw_score))
            or float(raw_score) not in {0.0, 0.5, 1.0}
        ):
            raise ValueError(
                f"strict BEAM official row {index} has invalid criterion score"
            )
        score = float(raw_score)
        normalized_scores.append(score)
        if not isinstance(criterion, dict):
            raise ValueError(
                f"strict BEAM official row {index} criterion evidence is malformed"
            )
        criterion_score = criterion.get("score")
        criterion_raw = criterion.get("raw")
        if isinstance(criterion_raw, str) and criterion_raw.startswith("[LLM_ERROR"):
            raise ValueError(
                f"strict BEAM official row {index} criterion has transport error"
            )
        parsed_raw, parsed_kind = _parse_beam_judge_json(criterion_raw)
        parsed_score = parsed_raw.get("score") if isinstance(parsed_raw, dict) else None
        parsed_reason = parsed_raw.get("reason") if isinstance(parsed_raw, dict) else None
        if (
            criterion.get("criterion_index") != criterion_index
            or isinstance(criterion.get("criterion_index"), bool)
            or criterion.get("rubric_item") != rubric_item
            or isinstance(criterion_score, bool)
            or not isinstance(criterion_score, (int, float))
            or not math.isfinite(float(criterion_score))
            or float(criterion_score) != score
            or criterion.get("parse") not in {"ok", "recovered"}
            or parsed_kind != criterion.get("parse")
            or isinstance(parsed_score, bool)
            or not isinstance(parsed_score, (int, float))
            or not math.isfinite(float(parsed_score))
            or float(parsed_score) != score
            or not isinstance(criterion.get("reason"), str)
            or not criterion["reason"].strip()
            or parsed_reason != criterion.get("reason")
        ):
            raise ValueError(
                f"strict BEAM official row {index} criterion evidence disagrees"
            )
    mean_score = sum(normalized_scores) / len(normalized_scores)
    if not math.isclose(
        float(row["score"]), mean_score, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"strict BEAM official row {index} score is not the criterion mean"
        )


def _strict_execution(data: dict) -> tuple[dict, dict]:
    """Extract only complete, explicitly available cumulative measurements."""

    execution = data.get("execution")
    if not isinstance(execution, dict):
        return {}, {"strict_execution": False}
    segments = execution.get("segments")
    disclosure = {
        "strict_execution": True,
        "segments_present": isinstance(segments, list),
        "segments_complete": False,
        "segment_count": len(segments) if isinstance(segments, list) else None,
    }
    if not isinstance(segments, list) or not segments:
        return {}, disclosure
    complete = all(
        isinstance(segment, dict) and segment.get("status") == "complete"
        for segment in segments
    )
    disclosure["segments_complete"] = complete

    def usage_total(key: str, field: str, available: str):
        if not complete:
            return None
        values = []
        for segment in segments:
            usage = segment.get(key)
            if not isinstance(usage, dict) or usage.get(available) is not True:
                return None
            value = _number(usage.get(field), integer=field != "latency_s")
            if value is None:
                return None
            values.append(value)
        return sum(values)

    elapsed = None
    if complete:
        elapsed_values = [_number(segment.get("elapsed_s")) for segment in segments]
        if all(value is not None for value in elapsed_values):
            elapsed = sum(elapsed_values)

    # Total provider tokens include every network/model component represented
    # by the adapter. Local/disabled embeddings consume no provider tokens;
    # semantic embeddings must explicitly report provider usage or the total
    # remains unavailable.
    total_tokens = 0
    tokens_available = complete
    for segment in segments:
        if not tokens_available:
            break
        for key in ("reader_usage", "judge_usage", "memory_pipeline_usage"):
            usage = segment.get(key)
            if (
                not isinstance(usage, dict)
                or usage.get("token_usage_available") is not True
                or _number(usage.get("total_tokens"), integer=True) is None
            ):
                tokens_available = False
                break
            total_tokens += int(usage["total_tokens"])
        embedding = segment.get("embedding_usage")
        if not tokens_available or not isinstance(embedding, dict):
            if not isinstance(embedding, dict):
                tokens_available = False
            continue
        if embedding.get("configured") and embedding.get("network_free") is False:
            provider_tokens = _number(embedding.get("total_tokens"), integer=True)
            if (
                embedding.get("provider_token_usage_available") is not True
                or provider_tokens is None
            ):
                tokens_available = False
            else:
                total_tokens += int(provider_tokens)

    disclosure["total_tokens_available"] = tokens_available
    counts = execution.get("counts") if isinstance(execution.get("counts"), dict) else {}
    return {
        "count": _number(counts.get("expected"), integer=True),
        "answer_calls": usage_total(
            "reader_usage", "calls", "calls_available"
        ),
        "judge_calls": usage_total(
            "judge_usage", "calls", "calls_available"
        ),
        "total_tokens": total_tokens if tokens_available else None,
        "elapsed_s": elapsed,
    }, disclosure


def _strict_summary(data: dict, scales: list[str]) -> dict[str, float | None]:
    """Weight a row-verified strict summary across every selected scale."""

    summary = data.get("summary")
    counts = data.get("summary_counts")
    if not isinstance(summary, dict) or not summary:
        return {}
    if set(summary) != set(scales):
        raise ValueError(
            "strict BEAM summary scales do not match manifested scales"
        )
    if not isinstance(counts, dict) or set(counts) != set(scales):
        raise ValueError(
            "strict BEAM summaries require per-scale metric counts"
        )
    expected_metrics = set(BEAM_ABILITIES) | {"OVERALL"}
    for scale in scales:
        scale_summary = summary.get(scale)
        scale_counts = counts.get(scale)
        if (
            not isinstance(scale_summary, dict)
            or set(scale_summary) != expected_metrics
            or not isinstance(scale_counts, dict)
            or set(scale_counts) != expected_metrics
        ):
            raise ValueError(
                f"strict BEAM metric/count coverage is incomplete for {scale}"
            )
    output = {}
    for metric in expected_metrics:
        weighted = 0.0
        denominator = 0
        for scale in scales:
            scale_summary = summary.get(scale)
            score = _number(scale_summary[metric])
            if score is None or score > 1:
                raise ValueError(
                    f"strict BEAM score {scale}/{metric} is malformed"
                )
            scale_counts = counts.get(scale)
            weight = _number(scale_counts.get(metric), integer=True)
            if weight is None or weight <= 0:
                raise ValueError(
                    f"strict BEAM summary has malformed count for {scale}/{metric}"
                )
            weighted += score * weight
            denominator += weight
        output[metric] = weighted / denominator * 100
    return output


def _validate_strict_envelope(data: dict) -> tuple[list[dict], dict, dict]:
    """Validate strict identity/denominator and recompute all score evidence."""

    manifest = data.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("strict BEAM artifact lacks a manifest")
    run_id = manifest.get("run_id")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", run_id) is None
        or run_id != content_hash({
            key: value for key, value in manifest.items() if key != "run_id"
        })
    ):
        raise ValueError("strict BEAM manifest run_id is invalid")
    if manifest.get("schema") != "hymem-benchmark-strict-v1":
        raise ValueError("strict BEAM manifest schema is unsupported")
    if manifest.get("benchmark") != "BEAM":
        raise ValueError("strict BEAM manifest benchmark is invalid")
    if data.get("benchmark") not in (None, "BEAM"):
        raise ValueError("strict BEAM top-level benchmark is invalid")
    if "version" in data and data.get("version") != "strict-v1":
        raise ValueError("strict BEAM top-level version is unsupported")
    config = data.get("config")
    models = data.get("models")
    if not isinstance(config, dict) or config != manifest.get("config"):
        raise ValueError("strict BEAM top-level config differs from manifest")
    if not isinstance(models, dict) or models != manifest.get("models"):
        raise ValueError("strict BEAM top-level models differ from manifest")
    if manifest.get("config_hash") != content_hash(config):
        raise ValueError("strict BEAM manifest config hash is invalid")
    if manifest.get("model_hash") != content_hash(models):
        raise ValueError("strict BEAM manifest model hash is invalid")
    for field in ("code_hash", "data_hash"):
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
        ):
            raise ValueError(f"strict BEAM manifest {field} is malformed")
    if (
        manifest.get("official_split") is not False
        or manifest.get("official_comparable") is not False
    ):
        raise ValueError("strict BEAM official split/comparability posture is invalid")
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("strict BEAM manifest seed is malformed")

    protocol_split = manifest.get("protocol_split")
    if protocol_split not in {"full", "dev", "holdout"}:
        raise ValueError("strict BEAM protocol split is invalid")
    if manifest.get("protocol_limitation") != BEAM_PROTOCOL_LIMITATIONS[protocol_split]:
        raise ValueError("strict BEAM protocol limitation is inconsistent")
    calibration_hash = manifest.get("calibration_receipt_hash")
    if protocol_split == "full":
        if calibration_hash is not None:
            raise ValueError(
                "strict BEAM full split cannot carry a calibration receipt"
            )
    elif (
        not isinstance(calibration_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", calibration_hash) is None
    ):
        raise ValueError(
            "strict BEAM internal split lacks a valid calibration receipt hash"
        )
    posture_fields = (
        "development_only", "exploratory_non_comparable",
        "label_free_answer_path", "exploratory_label_steering", "scored_run",
    )
    if any(not isinstance(manifest.get(field), bool) for field in posture_fields):
        raise ValueError("strict BEAM manifest protocol posture is malformed")
    for field in (
        "exploratory_non_comparable", "label_free_answer_path",
        "exploratory_label_steering", "scored_run",
    ):
        if config.get(field) is not manifest.get(field):
            raise ValueError(
                f"strict BEAM manifest/config posture differs for {field}"
            )
    expected_development_only = bool(
        protocol_split != "holdout"
        or not manifest["label_free_answer_path"]
        or manifest["exploratory_non_comparable"]
        or manifest["exploratory_label_steering"]
        or not manifest["scored_run"]
    )
    if manifest["development_only"] is not expected_development_only:
        raise ValueError("strict BEAM development-only posture is inconsistent")
    if manifest["scored_run"] is not True:
        raise ValueError("strict BEAM adapter artifacts must be scored runs")
    judge_protocol = config.get("judge_protocol")
    claimed_official_judge_match = config.get(
        "official_judge_protocol_match"
    )
    revision_complete = config.get("dataset_revision_provenance_complete")
    if (
        judge_protocol not in {"official", "legacy-custom"}
        or not isinstance(claimed_official_judge_match, bool)
        or not isinstance(revision_complete, bool)
    ):
        raise ValueError("strict BEAM protocol/revision disclosure is malformed")
    judge_identity = models.get("judge")
    if (
        not isinstance(judge_identity, dict)
        or judge_identity.get("protocol") != judge_protocol
    ):
        raise ValueError("strict BEAM judge identity differs from protocol")
    actual_official_judge_match = _official_judge_identity_matches(config, models)
    if claimed_official_judge_match is not actual_official_judge_match:
        raise ValueError("strict BEAM official judge disclosure is inconsistent")
    _validate_model_identity(models.get("reader"), label="reader")
    _validate_model_identity(judge_identity, label="judge")
    _validate_memory_pipeline_identity(models.get("memory_pipeline"))
    _validate_effective_hymem_config(config)

    if config.get("indexing_require_healthy") is not True:
        raise ValueError("strict BEAM indexing_require_healthy must be true")
    if config.get("official_judge_evaluator_url") != BEAM_OFFICIAL_EVALUATOR_URL:
        raise ValueError("strict BEAM official evaluator URL is inconsistent")
    if config.get("official_judge_prompt_url") != BEAM_OFFICIAL_PROMPT_URL:
        raise ValueError("strict BEAM official prompt URL is inconsistent")
    answer_extra = _validate_chat_extra_body(
        config.get("answer_extra_body"), label="answer",
    )
    judge_extra = _validate_chat_extra_body(
        config.get("judge_extra_body"), label="judge",
    )
    if judge_identity.get("extra_body") != judge_extra:
        raise ValueError("strict BEAM judge extra_body differs from model identity")

    for field in ("top_k", "max_input_tokens", "indexing_max_cycles"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"strict BEAM config {field} must be a positive integer")
    indexing_timeout = config.get("indexing_timeout_s")
    if (
        isinstance(indexing_timeout, bool)
        or not isinstance(indexing_timeout, (int, float))
        or not math.isfinite(float(indexing_timeout))
        or indexing_timeout <= 0
    ):
        raise ValueError("strict BEAM config indexing_timeout_s is malformed")

    official_protocol_aligned = config.get("official_protocol_aligned")
    if not isinstance(official_protocol_aligned, bool):
        raise ValueError("strict BEAM official protocol alignment is malformed")
    expected_protocol_alignment = bool(
        actual_official_judge_match
        and manifest["label_free_answer_path"]
        and protocol_split == "full"
    )
    if official_protocol_aligned is not expected_protocol_alignment:
        raise ValueError("strict BEAM official protocol alignment is inconsistent")

    scales = _normalize_scales(config.get("scales"), label="strict BEAM scales")
    unsupported_scales = [scale for scale in scales if scale not in VALID_BEAM_SCALES]
    if unsupported_scales:
        raise ValueError(f"strict BEAM artifact has unsupported scales: {unsupported_scales}")
    expected_repos = {
        BEAM_REPO_10M if scale == "10M" else BEAM_REPO for scale in scales
    }
    revisions = config.get("dataset_revisions")
    if not isinstance(revisions, dict) or set(revisions) != expected_repos:
        raise ValueError("strict BEAM dataset revision repository coverage is invalid")
    if revision_complete:
        if any(
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is None
            for revision in revisions.values()
        ):
            raise ValueError("strict BEAM complete dataset revisions are malformed")
    elif not manifest["exploratory_non_comparable"]:
        raise ValueError("strict BEAM unresolved dataset cannot claim comparable posture")
    oracle_ability = config.get("oracle_ability")
    judge_gold = config.get("judge_gold")
    if not isinstance(oracle_ability, bool) or not isinstance(judge_gold, bool):
        raise ValueError("strict BEAM routing/judge-gold posture is malformed")
    if manifest["label_free_answer_path"] is oracle_ability:
        raise ValueError("strict BEAM routing label-free posture is inconsistent")
    requires_exploratory = bool(
        oracle_ability
        or not judge_gold
        or config.get("prereg") is None
        or not revision_complete
        or not actual_official_judge_match
        or protocol_split != "full"
        or config.get("subset_run") is True
    )
    if requires_exploratory and not manifest["exploratory_non_comparable"]:
        raise ValueError("strict BEAM non-comparable protocol is marked comparable")
    _validate_prereg(
        config.get("prereg"), required=not manifest["exploratory_non_comparable"]
    )

    embedding_identity = models.get("embedding")
    config_embedding = config.get("embedding")
    if (
        not isinstance(embedding_identity, dict)
        or not isinstance(config_embedding, dict)
        or embedding_identity != config_embedding
    ):
        raise ValueError("strict BEAM embedding identity differs from config")
    _validate_embedding_config(embedding_identity)

    expected_count = manifest.get("expected_count")
    if (
        isinstance(expected_count, bool) or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise ValueError("strict BEAM expected count is invalid")
    rows = data.get("per_question")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("strict BEAM per-question denominator is incomplete")
    ids = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"strict BEAM row {index} is not an object")
        item_id = row.get("question_id")
        if (
            not isinstance(item_id, str) or not item_id.strip()
            or item_id != item_id.strip()
        ):
            raise ValueError(f"strict BEAM row {index} has malformed id")
        if item_id in seen:
            raise ValueError(f"strict BEAM artifact has duplicate id {item_id!r}")
        seen.add(item_id)
        ids.append(item_id)
    if manifest.get("expected_ids_hash") != content_hash(ids):
        raise ValueError("strict BEAM expected id order/hash is invalid")

    execution = data.get("execution")
    counts = execution.get("counts") if isinstance(execution, dict) else None
    if not isinstance(counts, dict):
        raise ValueError("strict BEAM execution counts are absent")
    segments = execution.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("strict BEAM execution segments are absent")
    complete_embedding_identities = set()
    segment_ids: set[str] = set()
    segment_attempts = 0
    successful_calls = {"reader": 0, "judge": 0, "memory_pipeline": 0}
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("strict BEAM execution segment is malformed")
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise ValueError("strict BEAM execution segment id is malformed")
        if segment_id in segment_ids:
            raise ValueError("strict BEAM execution segment ids are duplicated")
        segment_ids.add(segment_id)
        status = segment.get("status")
        if status not in {"running", "complete"}:
            raise ValueError("strict BEAM execution segment status is invalid")
        elapsed_s = segment.get("elapsed_s")
        if elapsed_s is not None and _nonnegative_number(elapsed_s) is None:
            raise ValueError("strict BEAM execution segment elapsed_s is malformed")
        if status == "complete" and elapsed_s is None:
            raise ValueError("strict BEAM complete execution segment lacks elapsed_s")
        attempted_attempts = segment.get("attempted_attempts")
        if (
            isinstance(attempted_attempts, bool)
            or not isinstance(attempted_attempts, int)
            or attempted_attempts < 0
        ):
            raise ValueError("strict BEAM segment attempted_attempts is invalid")
        segment_attempts += attempted_attempts
        for usage_key, label in (
            ("reader_usage", "reader"),
            ("judge_usage", "judge"),
            ("memory_pipeline_usage", "memory_pipeline"),
        ):
            validated_usage = _validate_usage_snapshot(
                segment.get(usage_key), label=label, complete=status == "complete",
            )
            calls = validated_usage["calls"]
            if calls is not None:
                successful_calls[label] += int(calls)
        embedding_usage = segment.get("embedding_usage")
        _validate_embedding_usage(
            embedding_usage, identity=embedding_identity, status=status,
            require_identity=(status == "complete" or attempted_attempts > 0),
        )
        if status == "complete" or attempted_attempts > 0:
            observed_identity = (
                embedding_usage.get("backend"), embedding_usage.get("quality"),
                embedding_usage.get("network_free"), embedding_usage.get("model"),
                embedding_usage.get("dimension"),
            )
            complete_embedding_identities.add(observed_identity)
    if len(complete_embedding_identities) > 1:
        raise ValueError("strict BEAM embedding identity changed across segments")
    required_counts = (
        "expected", "attempted", "unique_attempted", "total_attempts",
        "completed", "failed", "missing",
    )
    normalized = {}
    for key in required_counts:
        value = counts.get(key)
        if (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"strict BEAM execution count {key!r} is invalid")
        normalized[key] = value
    if (
        normalized["expected"] != expected_count
        or normalized["unique_attempted"] != normalized["attempted"]
        or normalized["attempted"] + normalized["missing"] != expected_count
        or normalized["completed"] + normalized["failed"] != expected_count
        or normalized["failed"] < normalized["missing"]
        or normalized["total_attempts"] < normalized["attempted"]
    ):
        raise ValueError("strict BEAM execution counts do not reconcile")
    if segment_attempts != normalized["total_attempts"]:
        raise ValueError("strict BEAM segment attempts differ from total attempts")

    values: dict[str, dict[str, list[float]]] = {
        scale: {} for scale in scales
    }
    conversation_abilities: dict[tuple[str, str], dict[str, int]] = {}
    completed_rows = 0
    failed_rows = 0
    missing_rows = 0
    minimum_reader_calls = 0
    minimum_judge_calls = 0
    for row_index, row in enumerate(rows):
        scale = row.get("scale")
        ability = row.get("ability")
        valid = row.get("result_valid")
        if scale not in values:
            raise ValueError(f"strict BEAM row has unknown scale {scale!r}")
        if ability not in BEAM_ABILITIES:
            raise ValueError(f"strict BEAM row has unknown ability {ability!r}")
        conv_id = row.get("conv_id")
        if not isinstance(conv_id, str) or not conv_id.strip():
            raise ValueError("strict BEAM row has malformed conversation id")
        if not row["question_id"].startswith(f"beam:{scale}:{conv_id}:"):
            raise ValueError("strict BEAM question id is not bound to scale/conversation")
        if "oracle_ability" not in row or row.get("oracle_ability") != ability:
            raise ValueError("strict BEAM row oracle ability differs from ability")
        detected_ability = row.get("detected_ability")
        ability_used = row.get("ability_used")
        if valid is True and (
            "detected_ability" not in row or "ability_used" not in row
        ):
            raise ValueError("strict BEAM valid row lacks routing evidence")
        if detected_ability not in {None, "MR", "TR"}:
            raise ValueError("strict BEAM row detected ability is malformed")
        expected_ability_used = ability if oracle_ability else detected_ability
        if valid is True:
            if ability_used != expected_ability_used:
                raise ValueError("strict BEAM valid row routing evidence is inconsistent")
        elif ability_used is not None and ability_used != expected_ability_used:
            raise ValueError("strict BEAM failed row routing evidence is inconsistent")
        conversation_abilities.setdefault((scale, conv_id), {}).setdefault(
            ability, 0
        )
        conversation_abilities[(scale, conv_id)][ability] += 1
        if not isinstance(valid, bool):
            raise ValueError("strict BEAM row has malformed result_valid")
        score = _number(row.get("score"))
        llm_judge_score = _number(row.get("llm_judge_score"))
        if score is None or score > 1 or llm_judge_score != score:
            raise ValueError("strict BEAM row judge score fields disagree")
        if row.get("judge_protocol") != judge_protocol:
            raise ValueError("strict BEAM row judge protocol differs from manifest")
        expected_correct = bool(valid and score == 1.0)
        if row.get("correct") is not expected_correct:
            raise ValueError("strict BEAM row correctness disagrees with score")
        if valid:
            if row.get("benchmark_failure"):
                raise ValueError("strict BEAM valid row carries a benchmark failure")
            question = row.get("question")
            answer = row.get("answer")
            rubric = row.get("rubric")
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(answer, str)
                or not answer.strip()
                or not isinstance(rubric, list)
                or not rubric
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in rubric
                )
            ):
                raise ValueError("strict BEAM successful row payload is malformed")
            if judge_protocol == "official":
                if row.get("judge_parse") != "ok":
                    raise ValueError("strict BEAM official row lacks a valid judge parse")
                _validate_official_row_evidence(row, index=row_index)
                minimum_judge_calls += len(row["judge_criterion_results"])
            minimum_reader_calls += 1
            completed_rows += 1
        else:
            if score != 0.0:
                raise ValueError("strict BEAM failed row must carry score zero")
            if row.get("scores") not in (None, []):
                raise ValueError("strict BEAM failed row carries rubric scores")
            failure = row.get("benchmark_failure")
            if not isinstance(failure, str) or not failure.strip():
                raise ValueError("strict BEAM failed row lacks failure metadata")
            if failure.startswith("judge_"):
                minimum_reader_calls += 1
            score = 0.0
            failed_rows += 1
            if failure == "missing_prediction":
                missing_rows += 1
        values[scale].setdefault(ability, []).append(score)
        values[scale].setdefault("OVERALL", []).append(score)
    if (
        normalized["completed"] != completed_rows
        or normalized["failed"] != failed_rows
        or normalized["missing"] != missing_rows
        or normalized["attempted"] != expected_count - missing_rows
    ):
        raise ValueError("strict BEAM execution counts differ from durable rows")
    if successful_calls["reader"] < minimum_reader_calls:
        raise ValueError(
            "strict BEAM reader usage is below durable post-reader rows"
        )
    if successful_calls["judge"] < minimum_judge_calls:
        raise ValueError(
            "strict BEAM judge usage is below durable official criteria"
        )

    subset_run = config.get("subset_run")
    sample = config.get("sample")
    sample_strategy = config.get("sample_strategy")
    denominator_validated = config.get("official_denominator_validated")
    if not isinstance(subset_run, bool) or not isinstance(
        denominator_validated, bool
    ):
        raise ValueError("strict BEAM subset/denominator posture is malformed")
    if subset_run:
        if (
            isinstance(sample, bool)
            or not isinstance(sample, int)
            or sample <= 0
            or sample_strategy != "seeded-label-blind-hash-v1"
            or not manifest["exploratory_non_comparable"]
        ):
            raise ValueError("strict BEAM subset sample posture is inconsistent")
    elif sample is not None or sample_strategy != "all":
        raise ValueError("strict BEAM full-set sample posture is inconsistent")
    expected_denominator_claim = bool(
        not subset_run
        and protocol_split == "full"
        and official_protocol_aligned
    )
    if denominator_validated is not expected_denominator_claim:
        raise ValueError("strict BEAM official denominator claim is inconsistent")
    if denominator_validated:
        if (
            subset_run
            or sample is not None
            or protocol_split != "full"
            or not official_protocol_aligned
        ):
            raise ValueError("strict BEAM official denominator claim is inconsistent")
        for scale in scales:
            expected = OFFICIAL_BEAM_DENOMINATORS[scale]
            scale_rows = [row for row in rows if row.get("scale") == scale]
            scale_conversations = {
                conv_id for row_scale, conv_id in conversation_abilities
                if row_scale == scale
            }
            if (
                len(scale_rows) != expected["questions"]
                or len(scale_conversations) != expected["conversations"]
            ):
                raise ValueError(
                    f"strict BEAM official denominator differs for {scale}"
                )
    if protocol_split == "full":
        expected_distribution = {ability: 2 for ability in BEAM_ABILITIES}
        for (scale, conv_id), ability_counts in conversation_abilities.items():
            if ability_counts != expected_distribution:
                raise ValueError(
                    "strict BEAM full-split conversation ability distribution "
                    f"differs for {scale}/{conv_id}"
                )
        if subset_run:
            for scale in scales:
                conversation_count = sum(
                    1 for row_scale, _conv_id in conversation_abilities
                    if row_scale == scale
                )
                expected_conversations = min(
                    sample, OFFICIAL_BEAM_DENOMINATORS[scale]["conversations"]
                )
                scale_rows = sum(
                    1 for row in rows if row.get("scale") == scale
                )
                if (
                    conversation_count != expected_conversations
                    or scale_rows != expected_conversations * 20
                ):
                    raise ValueError(
                        f"strict BEAM subset denominator differs for {scale}"
                    )

    expected_metrics = set(BEAM_ABILITIES) | {"OVERALL"}
    recomputed_summary = {}
    recomputed_counts = {}
    for scale in scales:
        if set(values[scale]) != expected_metrics:
            raise ValueError(
                f"strict BEAM scale {scale!r} lacks complete ability coverage"
            )
        recomputed_summary[scale] = {
            metric: sum(scores) / len(scores)
            for metric, scores in values[scale].items()
        }
        recomputed_counts[scale] = {
            metric: len(scores) for metric, scores in values[scale].items()
        }

    summary_present = "summary" in data
    counts_present = "summary_counts" in data
    stored_summary = data.get("summary")
    stored_counts = data.get("summary_counts")
    summary_disclosure = {
        "source": "recomputed_from_durable_rows",
        "stored_summary_present": summary_present,
        "stored_summary_validated": False,
    }
    if summary_present or counts_present:
        if (
            not summary_present
            or not counts_present
            or not isinstance(stored_summary, dict)
            or not stored_summary
            or not isinstance(stored_counts, dict)
            or not stored_counts
        ):
            raise ValueError("strict BEAM stored summary/counts are partial")
        if set(stored_summary) != set(recomputed_summary) or set(stored_counts) != set(
            recomputed_counts
        ):
            raise ValueError("strict BEAM stored summary scale coverage is incomplete")
        for scale in scales:
            if (
                not isinstance(stored_summary.get(scale), dict)
                or not isinstance(stored_counts.get(scale), dict)
                or set(stored_summary[scale]) != set(recomputed_summary[scale])
            ):
                raise ValueError(
                    f"strict BEAM stored summary coverage/counts differ for {scale}"
                )
            for metric, expected_count_value in recomputed_counts[scale].items():
                observed_count = stored_counts[scale].get(metric)
                if (
                    isinstance(observed_count, bool)
                    or not isinstance(observed_count, int)
                    or observed_count <= 0
                    or observed_count != expected_count_value
                ):
                    raise ValueError(
                        "strict BEAM stored summary coverage/counts differ for "
                        f"{scale}/{metric}"
                    )
            for metric, expected in recomputed_summary[scale].items():
                observed = _number(stored_summary[scale].get(metric))
                if observed is None or not math.isclose(
                    observed, expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"strict BEAM stored summary differs for {scale}/{metric}"
                    )
        summary_disclosure["stored_summary_validated"] = True
    return rows, {
        "summary": recomputed_summary,
        "summary_counts": recomputed_counts,
    }, summary_disclosure


def _beam_row(data: dict, path: Path) -> dict:
    if (
        isinstance(data, dict)
        and set(data) <= {"archive", "run_id"}
        and "archive" in data
    ):
        raise ValueError(
            "results_latest.json is a mutable pointer, not a BEAM result artifact"
        )
    cfg = dict(data.get("config") or {})
    meta = dict(data.get("metadata") or {})
    raw_scores = data.get("scores") or {}
    summary = data.get("summary") or {}
    strict_filename = bool(
        path.name.startswith("results_") and "-strict-" in path.name
    )
    strict_shape = bool(
        isinstance(data, dict)
        and {"manifest", "execution", "per_question", "models"} & set(data)
    )
    version = data.get("version")
    strict_version = isinstance(version, str) and version.startswith("strict-")
    strict = bool(
        strict_version
        or strict_filename
        or strict_shape
    )
    if strict and (
        not isinstance(data.get("manifest"), dict)
        or not isinstance(data.get("execution"), dict)
    ):
        raise ValueError("strict BEAM artifact lacks manifest or execution state")
    strict_execution = {}
    strict_execution_disclosure = {}
    strict_summary_disclosure = {}
    legacy_score_disclosure = {}

    if strict:
        scales = _normalize_scales(cfg.get("scales"), label="strict BEAM scales")
        _rows, recomputed, strict_summary_disclosure = _validate_strict_envelope(data)
        scale = ",".join(scales)
        scores = _strict_summary(recomputed, scales)
        strict_execution, strict_execution_disclosure = _strict_execution(data)
        eff_cfg = cfg
    elif summary:  # Dialect C: {scale: {ABILITY: frac}}
        scales, scores, legacy_score_disclosure = _legacy_fraction_summary(
            summary,
            data.get("summary_counts") or meta.get("summary_counts"),
            meta.get("scales", cfg.get("scales")),
        )
        scale = ",".join(scales)
        eff_cfg = {**meta, **cfg}
    else:  # Dialect A/B
        scales = _normalize_scales(
            cfg.get("scales"), label="legacy BEAM config scales"
        )
        scale = ",".join(scales)
        scores, legacy_score_disclosure = _legacy_percent_scores(raw_scores)
        eff_cfg = cfg

    kind = _beam_kind(path.name)
    # §6: stamp-derived dates.  Rejudge: source_date from rejudged_from,
    # run_date = last stamp (rejudge exec); stats NULL (inherited-but-wrong
    # is worse than missing).  Archive/variant: first stem stamp or NULL.
    if kind == "rejudge":
        src_date, exec_date = rr.rejudge_dates(
            path.name, meta.get("rejudged_from") or "",
            SPEC.get("stamp_policy", "optional"))
        run_date = rr.iso_ts(exec_date or data.get("date") or meta.get("date"))
        source_date = src_date
        total_tokens = None
        elapsed_s = None
    else:
        run_date = rr.iso_ts(
            data.get("date") or data.get("created_at") or meta.get("date")
        )
        source_date = rr.stem_source_date(
            path.name, SPEC.get("stamp_policy", "optional"))
        total_tokens = (
            strict_execution.get("total_tokens")
            if strict else cfg.get("total_tokens")
        )
        elapsed_s = (
            strict_execution.get("elapsed_s")
            if strict else (meta.get("elapsed_s") or cfg.get("elapsed_s"))
        )

    row = {c: None for c, _ in BEAM_COLUMNS}
    row["archive"] = path.name
    row["kind"] = kind
    row["run_date"] = run_date
    row["source_date"] = source_date
    row["scale"] = scale
    manifest = data.get("manifest") if strict else {}
    row["run_id"] = manifest.get("run_id") if strict else None
    row["protocol_split"] = manifest.get("protocol_split") if strict else None
    row["development_only"] = _coerce(
        manifest.get("development_only") if strict else None, "INTEGER"
    )
    row["exploratory_non_comparable"] = _coerce(
        manifest.get("exploratory_non_comparable") if strict else None,
        "INTEGER",
    )
    row["label_free_answer_path"] = _coerce(
        manifest.get("label_free_answer_path") if strict else None, "INTEGER"
    )
    row["judge_protocol"] = cfg.get("judge_protocol") if strict else None
    row["official_judge_protocol_match"] = _coerce(
        cfg.get("official_judge_protocol_match") if strict else None,
        "INTEGER",
    )
    row["dataset_revisions_complete"] = _coerce(
        cfg.get("dataset_revision_provenance_complete") if strict else None,
        "INTEGER",
    )
    row["sample"] = eff_cfg.get("sample")
    row["top_k"] = eff_cfg.get("top_k")
    row["context_memories"] = eff_cfg.get("context_memories")
    models = data.get("models") if isinstance(data.get("models"), dict) else {}
    reader_model = models.get("reader") if isinstance(models.get("reader"), dict) else {}
    judge_model = models.get("judge") if isinstance(models.get("judge"), dict) else {}
    row["answer_model"] = (
        reader_model.get("model") if strict else eff_cfg.get("answer_model")
    )
    row["judge_model"] = (
        judge_model.get("model") if strict else eff_cfg.get("judge_model")
    )
    effective_hymem = (
        cfg.get("effective_hymem_config")
        if isinstance(cfg.get("effective_hymem_config"), dict) else {}
    )
    if strict and isinstance(cfg.get("embedding"), dict):
        eff_cfg["embeddings"] = cfg["embedding"].get("configured")
    if strict:
        # These are effective values, not guesses from optional CLI flags.
        # BEAM's fixed adapter protocol always completes dreaming and does no
        # separate distillation stage, so those two controls are recorded as
        # false rather than left as an ambiguous NULL.
        eff_cfg.update({
            "facts": effective_hymem["facts_enabled"],
            "facts_extraction": effective_hymem["facts_extraction_enabled"],
            "graph_multihop": effective_hymem["graph_multihop_enabled"],
            "episode_granularity_enabled": (
                effective_hymem["episode_granularity_enabled"]
            ),
            "aggregation_nodes_enabled": (
                effective_hymem["aggregation_nodes_enabled"]
            ),
            "value_supersession_enabled": (
                effective_hymem["value_supersession_enabled"]
            ),
            "no_dream": False,
            "distill": False,
        })
    for key in ("embeddings", "facts", "facts_extraction", "graph_multihop",
                "no_dream", "distill", "episode_granularity_enabled",
                "aggregation_nodes_enabled", "value_supersession_enabled"):
        row[key] = _coerce(eff_cfg.get(key), "INTEGER")
    for ab in BEAM_ABILITIES:
        v = scores.get(ab)
        row[f"ability_{ab.lower()}"] = (
            round(v, 3) if isinstance(v, (int, float)) else None)
    row["overall"] = (
        round(scores.get("OVERALL"), 3)
        if isinstance(scores.get("OVERALL"), (int, float)) else None)
    row["count"] = (
        strict_execution.get("count") if strict
        else (cfg.get("count") or (meta.get("count") if meta else None))
    )
    row["answer_calls"] = (
        strict_execution.get("answer_calls") if strict
        else (cfg.get("answer_calls") or (meta.get("answer_calls") if meta else None))
    )
    row["judge_calls"] = (
        strict_execution.get("judge_calls") if strict
        else (cfg.get("judge_calls") or (meta.get("judge_calls") if meta else None))
    )
    row["total_tokens"] = total_tokens
    row["elapsed_s"] = elapsed_s
    row["extras"] = json_dumps({
        "config": data.get("config"),
        "manifest": data.get("manifest") if strict else None,
        "metadata": data.get("metadata"),
        "scores": scores,
        "strict_summary": data.get("summary") if strict else None,
        "strict_summary_counts": data.get("summary_counts") if strict else None,
        "models": data.get("models") if strict else None,
        "execution": data.get("execution") if strict else None,
        "execution_disclosure": strict_execution_disclosure if strict else None,
        "summary_disclosure": strict_summary_disclosure if strict else None,
        "legacy_score_disclosure": (
            legacy_score_disclosure if not strict else None
        ),
        "raw_json_keys": sorted(data.keys()),
    })
    return row


def json_dumps(obj):
    return json.dumps(obj, default=str)


_ROW_BUILDER = _beam_row
_KIND = _beam_kind


def _backfill(db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _beam_row
    return rr.cmd_backfill(spec, db_path=db_path)


def _ingest(files, overrides=None, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _beam_row
    return cmd_ingest(spec, files or None, overrides, db_path=db_path)


def _record_doc(archive, overrides=None, db_path=None):
    """Enter a run whose run-file is lost but is documented elsewhere
    (e.g. beam-results-history.md).  Provenance starts with
    'analyst:doc=' — never 'recorded'."""
    import json
    import sqlite3
    overrides = dict(overrides or {})
    con = connect(spec=dict(SPEC), db_path=db_path)
    ex = con.execute("SELECT id FROM runs WHERE archive=?", (archive,)).fetchone()
    if ex:
        return "skipped"
    row = {c: None for c, _ in BEAM_COLUMNS}
    applied = {}
    for k, v in overrides.items():
        if k not in BEAM_DOC_OVERRIDES:
            continue
        typ = dict(BEAM_COLUMNS).get(k, "TEXT")
        # §6.5: doc rows are analyst-set and carry bare dates.  cmd_backfill
        # canonicalises them in place too, but a row entered without a
        # later backfill would otherwise sit at width 10 in the sort
        # column -- so both entry points canonicalise on write.
        row[k] = rr.iso_ts(v) if k == "run_date" else _coerce(v, typ)
        applied[k] = row[k]
    row["archive"] = archive
    row["kind"] = "doc"
    row["source_date"] = "DOC"
    names = [c for c, _ in BEAM_COLUMNS] + ["flags_provenance", "extras"]
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
    print(f"row added (kind=doc, prov='{prov}')")


def _list(limit=30, flag=None, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _beam_row
    return cmd_list(spec, limit, flag, db_path)


def _query(sql, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _beam_row
    return cmd_query(spec, sql, db_path)


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
