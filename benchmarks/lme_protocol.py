"""Pinned LongMemEval protocol identities and fail-closed artifact validation.

This module is intentionally provider-free.  Registry ingestion and official
prediction export must be able to validate a completed run without importing an
SDK, reading a credential, or constructing a client.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from hymem.contrib.endpoint_policy import (
    TRANSPORT_SECURITY_NONE,
    validate_recorded_embedding_endpoint,
    validate_http_endpoint,
    secret_free_endpoint_identity,
)
from hymem.dreaming.lossless import (
    COVERAGE_INTEGRITY_CONFIG_VERSION,
    COVERAGE_INTEGRITY_FAILURE_REASONS,
    MAX_COVERAGE_INTEGRITY_OCCURRENCES,
)
from hymem.dreaming.status import (
    DREAM_STATUS_AGGREGATION_AUTHORITY_FIELDS,
    DREAM_STATUS_AGGREGATION_MATERIAL_AUTHORITY_FIELDS,
    DREAM_STATUS_PHASE1_AUTHORITY_FIELDS,
    DREAM_STATUS_SCHEMA_VERSION,
    DURABLE_MALFORMED_FIELDS,
)
from hymem.dreaming.aggregation_generation import (
    validate_aggregation_generation_binding,
)
from hymem.dreaming.aggregation_material import (
    validate_aggregation_material_binding,
    validate_public_embedding_identity,
)
from hymem.extraction.producer import validate_aggregation_producer_binding

try:
    from .extraction_canary import (
        validate_extraction_canary_config_binding,
        validate_extraction_canary_report,
    )
    from .strictness import (
        BENCHMARK_INDEXING_STATUS_VERSION,
        BenchmarkIntegrityError,
        STRICT_PROTOCOL_VERSION,
        content_hash,
        validate_ids,
        write_immutable_artifact,
    )
except (ImportError, ValueError):  # direct benchmark-script import
    from extraction_canary import (  # type: ignore
        validate_extraction_canary_config_binding,
        validate_extraction_canary_report,
    )
    from strictness import (  # type: ignore
        BENCHMARK_INDEXING_STATUS_VERSION,
        BenchmarkIntegrityError,
        STRICT_PROTOCOL_VERSION,
        content_hash,
        validate_ids,
        write_immutable_artifact,
    )


LME_EVALUATOR_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
LME_EVALUATOR_SHA256 = (
    "sha256:ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"
)
LME_EVALUATOR_URL = (
    "https://github.com/xiaowu0162/LongMemEval/blob/"
    f"{LME_EVALUATOR_COMMIT}/src/evaluation/evaluate_qa.py"
)
LME_S_DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LME_S_DATASET_SHA256 = (
    "sha256:d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
LME_S_EXPECTED_COUNT = 500
# ``content_hash`` of the 500 question IDs in the exact order carried by the
# pinned S JSON.  This is deliberately independent of an artifact's own
# self-asserted ``source_order_validated`` flag: official export can bind its
# rows to a known source-order commitment without opening a provider client.
LME_S_SOURCE_IDS_HASH = (
    "sha256:a4849b8afda6b6ed31ead4fc28d00784d2d5fef945be87642f5ce3ab710b21c4"
)
LME_S_QTYPE_COUNTS = {
    "single-session-user": 70,
    "single-session-assistant": 56,
    "multi-session": 133,
    "temporal-reasoning": 133,
    "knowledge-update": 78,
    "single-session-preference": 30,
}
LME_S_DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/blob/"
    f"{LME_S_DATASET_REVISION}/longmemeval_s_cleaned.json"
)
LME_OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"
LME_OFFICIAL_JUDGE_BASE_URL = "https://api.openai.com/v1"
LME_OFFICIAL_JUDGE_TEMPERATURE = 0.0
LME_OFFICIAL_JUDGE_MAX_TOKENS = 10
LME_OFFICIAL_VERDICT_PARSER = "substring-yes-in-lower-v1"
LME_UPSTREAM_RETRY_POLICY = "unbounded-openai-backoff-v1"
LME_LOCAL_RETRY_POLICY = "bounded-three-attempt-backoff-v1"
LME_INDEXING_SUMMARY_VERSION = "hymem-lme-indexing-summary-v4"
LME_INDEXING_COVERAGE_DETAIL_LIMIT = 100
LME_HISTORICAL_LOCAL_JUDGE_PROMPTS_EXACT_OFFICIAL = False
# Compatibility alias for older imports.  Strict evidence uses the longer,
# unambiguous field name above: the separately selected official prompt path is
# byte-pinned even though the historical local prompt suite is not.
LME_LOCAL_PROMPTS_EXACT_OFFICIAL = (
    LME_HISTORICAL_LOCAL_JUDGE_PROMPTS_EXACT_OFFICIAL
)

LME_BASE_QUESTION_TYPES = frozenset({
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
})
LME_SUPPORTED_SCALES = frozenset({"S", "M"})
LME_ABILITY_BY_TYPE = {
    "single-session-user": "IE",
    "single-session-assistant": "IE",
    "multi-session": "MR",
    "temporal-reasoning": "TR",
    "knowledge-update": "KU",
    "single-session-preference": "PF",
}
RESERVED_CHAT_BODY_KEYS = frozenset({
    "model", "messages", "temperature", "max_tokens", "n",
})

_DATE_FORMATS = (
    "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M",
    "%Y/%m/%d", "%Y-%m-%d",
)
_YES_WORD = re.compile(r"\byes\b")
_NO_WORD = re.compile(r"\bno\b")
_NEGATED_YES = re.compile(
    r"\b(?:not|never|isn'?t|wasn'?t|aren'?t|ain'?t)\s+"
    r"(?:really\s+|quite\s+|exactly\s+|an?\s+)?yes\b"
)

_INDEXING_FAILURE_CODES = frozenset({
    "timeout_before_cycle",
    "timeout_during_cycle",
    "cycle_exception",
    "malformed_status_shape",
    "malformed_pending_backlog",
    "malformed_quarantine_state",
    "malformed_terminal_loss_state",
    "malformed_coverage_integrity_state",
    "malformed_aggregation_failure_report",
    "malformed_cycle_failure_report",
    "coverage_integrity_failure",
    "malformed_durable_state",
    "terminal_extraction_source_loss",
    "quarantined_extraction",
    "timeout_after_cycle",
    "max_cycles_exhausted",
})
_MECHANICALLY_COMPLETE_FAILURE_CODES = frozenset({
    "coverage_integrity_failure",
    "malformed_durable_state",
    "terminal_extraction_source_loss",
    "quarantined_extraction",
})
_INDEXING_REPORT_FIELDS = (
    "sessions_processed",
    "chunks_seen",
    "chunks_processed",
    "chunk_extraction_failures",
    "chunk_extraction_completion_calls",
    "chunk_extraction_provider_attempts",
    "extraction_provider_attempt_budget_exhausted",
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
    "aggregation_blocking",
    "digest_failures",
    "digest_quarantined",
    "episodes_created",
    "facts_extracted",
    "fact_failures",
    "facts_embedded",
    "facts_embedded_from_cache",
    "profile_items_extracted",
    "profile_failures",
    "budget_exhausted",
    "skipped_locked",
)
_INDEXING_REPORT_BOOLEAN_FIELDS = frozenset({
    "budget_exhausted",
    "skipped_locked",
    "extraction_provider_attempt_budget_exhausted",
})
_INDEXING_REPORT_OPTIONAL_COUNT_FIELDS = frozenset({
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
})
_INDEXING_AGGREGATION_BLOCKING_VALUES = frozenset({
    "", "exact", "exact:disabled", "exact:no_vec_extension",
    "exact:no_vec_table", "knn",
})
_INDEXING_SUMMARY_COMMON_FIELDS = frozenset({
    "schema", "outcome", "cycles", "max_cycles", "timeout_s", "elapsed_s",
    "complete", "healthy", "reports", "final_status", "cleanup_errors",
})
_INDEXING_PENDING_FIELDS = frozenset({
    "pending_source_materialization",
    "pending_chunks",
    "pending_digests",
    "pending_profiles",
    "pending_facts",
    "pending_aggregation",
    "pending_chunk_embeddings",
    "pending_message_embeddings",
    "pending_edge_embeddings",
    "pending_episode_embeddings",
    "pending_fact_embeddings",
})
_INDEXING_MALFORMED_FIELDS = frozenset(DURABLE_MALFORMED_FIELDS)
_INDEXING_QUARANTINE_FIELDS = frozenset({
    "quarantined_chunks",
    "quarantined_digests",
    "quarantined_profiles",
    "quarantined_facts",
    "quarantined_facts_malformed",
})
_INDEXING_CYCLE_FAILURE_FIELDS = frozenset({
    "chunk_extraction_failures",
    "coverage_integrity_failures",
    "digest_failures",
    "digest_quarantined",
    "profile_failures",
    "fact_failures",
    "aggregation_fusion_failures",
    "aggregation_build_exceptions",
})
_SAFE_EXCEPTION_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,127}")
_SAFE_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def _valid_status_timestamp(value: object) -> bool:
    if not isinstance(value, str) or _SAFE_TIMESTAMP.fullmatch(value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return True


def is_official_abstention_id(question_id: object) -> bool:
    """Mirror upstream exactly: substring membership, not a suffix rule."""

    return isinstance(question_id, str) and "_abs" in question_id


def normalize_lme_date(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BenchmarkIntegrityError(f"{label} must be a non-empty date string")
    cleaned = re.sub(r"\s*\([^)]*\)", "", value).strip()
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BenchmarkIntegrityError(f"{label} has an unsupported date format") from exc
    return parsed.isoformat()


def validate_lme_dataset(
    rows: Iterable[Mapping[str, Any]], *, scale: str,
) -> tuple[dict[str, Any], ...]:
    """Validate the complete source dataset before any provider is contacted."""

    scale_norm = str(scale).upper()
    if scale_norm not in LME_SUPPORTED_SCALES:
        raise BenchmarkIntegrityError(
            f"unsupported LongMemEval scale {scale!r}; expected S or M"
        )
    validated: list[dict[str, Any]] = []
    ids: list[str] = []
    for row_index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise BenchmarkIntegrityError(f"LongMemEval row {row_index} is not an object")
        row = dict(raw)
        qid = row.get("question_id")
        qtype = row.get("question_type")
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(qid, str) or not qid.strip() or qid != qid.strip():
            raise BenchmarkIntegrityError(f"LongMemEval row {row_index} has an invalid question_id")
        if qtype not in LME_BASE_QUESTION_TYPES:
            raise BenchmarkIntegrityError(
                f"LongMemEval row {qid!r} has unknown question_type {qtype!r}"
            )
        if not isinstance(question, str) or not question.strip():
            raise BenchmarkIntegrityError(f"LongMemEval row {qid!r} has an invalid question")
        if not (
            (isinstance(answer, str) and bool(answer.strip()))
            or (isinstance(answer, int) and not isinstance(answer, bool))
        ):
            raise BenchmarkIntegrityError(f"LongMemEval row {qid!r} has an invalid answer")

        sessions = row.get("haystack_sessions")
        session_ids = row.get("haystack_session_ids")
        dates = row.get("haystack_dates")
        answer_ids = row.get("answer_session_ids")
        if not isinstance(sessions, list) or not sessions:
            raise BenchmarkIntegrityError(f"LongMemEval row {qid!r} lacks haystack sessions")
        if not isinstance(session_ids, list) or not isinstance(dates, list):
            raise BenchmarkIntegrityError(f"LongMemEval row {qid!r} has malformed session metadata")
        if len(sessions) != len(session_ids) or len(sessions) != len(dates):
            raise BenchmarkIntegrityError(
                f"LongMemEval row {qid!r} session/id/date lengths differ"
            )
        if not isinstance(answer_ids, list) or not answer_ids or any(
            not isinstance(item, str) or not item.strip() or item != item.strip()
            for item in answer_ids
        ):
            raise BenchmarkIntegrityError(f"LongMemEval row {qid!r} has invalid answer_session_ids")
        valid_session_ids: list[str] = []
        for session_index, (session_id, date, messages) in enumerate(
            zip(session_ids, dates, sessions, strict=True)
        ):
            if (
                not isinstance(session_id, str) or not session_id.strip()
                or session_id != session_id.strip()
            ):
                raise BenchmarkIntegrityError(
                    f"LongMemEval row {qid!r} has an invalid session id"
                )
            # The pinned source contains repeated session IDs in 13 rows.  They
            # are distinct ordered haystack occurrences, not a uniqueness key.
            valid_session_ids.append(session_id)
            normalize_lme_date(date, label=f"{qid} haystack date {session_index}")
            if not isinstance(messages, list) or not messages:
                raise BenchmarkIntegrityError(
                    f"LongMemEval row {qid!r} session {session_id!r} has no messages"
                )
            for message_index, message in enumerate(messages):
                if not isinstance(message, Mapping):
                    raise BenchmarkIntegrityError(
                        f"LongMemEval row {qid!r} message {message_index} is not an object"
                    )
                if message.get("role") not in {"user", "assistant"}:
                    raise BenchmarkIntegrityError(
                        f"LongMemEval row {qid!r} has unsupported message role"
                    )
                content = message.get("content")
                # The pinned S source contains 12 explicit empty-string turns.
                # Their type is valid; ingestion reports (and skips) them rather
                # than mutating the official source or rejecting its hash.
                if not isinstance(content, str):
                    raise BenchmarkIntegrityError(
                        f"LongMemEval row {qid!r} has a non-string message"
                    )
                if "has_answer" in message and not isinstance(message["has_answer"], bool):
                    raise BenchmarkIntegrityError(
                        f"LongMemEval row {qid!r} has malformed has_answer"
                    )
        if any(item not in valid_session_ids for item in answer_ids):
            raise BenchmarkIntegrityError(
                f"LongMemEval row {qid!r} names an unknown answer session"
            )
        if "question_date" not in row:
            raise BenchmarkIntegrityError(f"LongMemEval row {qid!r} lacks question_date")
        normalize_lme_date(row["question_date"], label=f"{qid} question_date")
        ids.append(qid)
        validated.append(row)
    if not validated:
        raise BenchmarkIntegrityError("LongMemEval dataset is empty")
    validate_ids(ids, label="LongMemEval dataset")
    return tuple(validated)


def validate_safe_endpoint(value: object, *, label: str) -> str:
    try:
        return validate_http_endpoint(value, label=label).url
    except (TypeError, ValueError) as exc:
        raise BenchmarkIntegrityError(
            f"{label} endpoint is unsafe or ambiguous"
        ) from exc


def normalize_extra_body(value: object, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BenchmarkIntegrityError(f"{label} extra_body must be an object")
    if any(not isinstance(key, str) for key in value):
        raise BenchmarkIntegrityError(f"{label} extra_body keys must be strings")
    collisions = RESERVED_CHAT_BODY_KEYS & set(value)
    if collisions:
        raise BenchmarkIntegrityError(
            f"{label} extra_body cannot override core field(s): {sorted(collisions)}"
        )
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise BenchmarkIntegrityError(f"{label} extra_body is not canonical JSON") from exc
    return normalized


def validate_prereg(value: object, *, required: bool) -> None:
    if value is None:
        if required:
            raise BenchmarkIntegrityError("comparable LongMemEval run lacks pre-registration")
        return
    if not isinstance(value, Mapping) or set(value) != {
        "path", "commit", "blob", "committed_at", "code_commit",
    }:
        raise BenchmarkIntegrityError("LongMemEval pre-registration receipt is malformed")
    path = value.get("path")
    if not isinstance(path, str) or not path.strip() or path != path.strip() or "\\" in path:
        raise BenchmarkIntegrityError("LongMemEval pre-registration path is malformed")
    parsed_path = PurePosixPath(path)
    if parsed_path.is_absolute() or any(part in {"", ".", ".."} for part in parsed_path.parts):
        raise BenchmarkIntegrityError("LongMemEval pre-registration path is unsafe")
    for field in ("commit", "blob", "code_commit"):
        if not isinstance(value.get(field), str) or re.fullmatch(
            r"[0-9a-fA-F]{40,64}", value[field]
        ) is None:
            raise BenchmarkIntegrityError(f"LongMemEval pre-registration {field} is malformed")
    stamp = value.get("committed_at")
    if not isinstance(stamp, str) or not stamp.strip():
        raise BenchmarkIntegrityError("LongMemEval pre-registration timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BenchmarkIntegrityError("LongMemEval pre-registration timestamp is malformed") from exc
    if parsed.tzinfo is None:
        raise BenchmarkIntegrityError("LongMemEval pre-registration timestamp lacks timezone")


def official_judge_match(config: Mapping[str, Any], models: Mapping[str, Any]) -> bool:
    judge = models.get("judge")
    official_endpoint = secret_free_endpoint_identity(
        LME_OFFICIAL_JUDGE_BASE_URL, label="official judge"
    )
    return bool(
        isinstance(judge, Mapping)
        and config.get("judge_protocol") == "official"
        and judge.get("protocol") == "official"
        and judge.get("provider") == "openai"
        and judge.get("model") == LME_OFFICIAL_JUDGE_MODEL
        and judge.get("endpoint_origin") == official_endpoint["endpoint_origin"]
        and judge.get("endpoint_sha256") == official_endpoint["endpoint_sha256"]
        and config.get("official_judge_endpoint_origin")
        == official_endpoint["endpoint_origin"]
        and config.get("official_judge_endpoint_sha256")
        == official_endpoint["endpoint_sha256"]
        and _finite_number(judge.get("temperature")) == 0.0
        and judge.get("max_tokens") == LME_OFFICIAL_JUDGE_MAX_TOKENS
        and judge.get("n") == 1
        and judge.get("extra_body") == {}
        and judge.get("evaluator_commit") == LME_EVALUATOR_COMMIT
        and judge.get("evaluator_sha256") == LME_EVALUATOR_SHA256
        and judge.get("verdict_parser") == LME_OFFICIAL_VERDICT_PARSER
        and judge.get("prompt_exact_official") is True
    )


def parse_official_verdict(raw: str) -> bool:
    """The pinned evaluator's literal decision rule."""

    return "yes" in (raw or "").lower()


def parse_legacy_verdict(raw: str) -> tuple[bool | None, bool]:
    if not isinstance(raw, str) or not raw or raw.startswith("[LLM_ERROR"):
        return None, False
    low = raw.lower()
    yes = _YES_WORD.search(low)
    no = _NO_WORD.search(low)
    parseable = bool(yes) != bool(no)
    if not parseable:
        return None, False
    if _NEGATED_YES.search(low):
        return False, True
    return bool(yes), True


def strict_intent(data: object, path: Path | None = None) -> bool:
    if not isinstance(data, Mapping):
        return False
    filename = path.name if path is not None else ""
    version = data.get("version")
    return bool(
        (isinstance(version, str) and version.startswith("strict-"))
        or "-strict-" in filename
        or {"manifest", "execution", "models"} & set(data)
    )


def _finite_number(value: object, *, integer: bool = False) -> int | float | None:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or value < 0
        or (integer and not float(value).is_integer())
    ):
        return None
    return int(value) if integer else float(value)


def _bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkIntegrityError(f"{label} must be boolean")
    return value


def _usage(snapshot: object, *, label: str) -> dict[str, int | float | None]:
    if not isinstance(snapshot, Mapping):
        raise BenchmarkIntegrityError(f"LongMemEval {label} usage is absent")

    def available(field: str, marker: str, *, integer: bool = False):
        flag = snapshot.get(marker)
        if not isinstance(flag, bool):
            raise BenchmarkIntegrityError(f"LongMemEval {label} {marker} is malformed")
        value = snapshot.get(field)
        if not flag:
            if value is not None:
                raise BenchmarkIntegrityError(
                    f"LongMemEval {label} {field} claims unavailable precision"
                )
            return None
        normalized = _finite_number(value, integer=integer)
        if normalized is None:
            raise BenchmarkIntegrityError(f"LongMemEval {label} {field} is malformed")
        return normalized

    calls = available("calls", "calls_available", integer=True)
    attempts = available("request_attempts", "request_attempts_available", integer=True)
    successes = available(
        "successful_responses", "successful_responses_available", integer=True
    )
    latency = available("latency_s", "latency_available")
    available("cost_usd", "cost_available")
    token_available = snapshot.get("token_usage_available")
    if not isinstance(token_available, bool):
        raise BenchmarkIntegrityError(f"LongMemEval {label} token availability is malformed")
    token_values: list[int] = []
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = snapshot.get(field)
        if token_available:
            normalized = _finite_number(value, integer=True)
            if normalized is None:
                raise BenchmarkIntegrityError(f"LongMemEval {label} {field} is malformed")
            token_values.append(normalized)
        elif value is not None:
            raise BenchmarkIntegrityError(
                f"LongMemEval {label} {field} claims unavailable precision"
            )
    if token_available and token_values[2] != token_values[0] + token_values[1]:
        raise BenchmarkIntegrityError(f"LongMemEval {label} token totals do not reconcile")
    if calls is not None and successes is not None and successes != calls:
        raise BenchmarkIntegrityError(f"LongMemEval {label} successful call count disagrees")
    if attempts is not None and calls is not None and attempts < calls:
        raise BenchmarkIntegrityError(f"LongMemEval {label} attempts are below calls")
    return {
        "calls": calls, "attempts": attempts, "successes": successes,
        "total_tokens": token_values[2] if token_available else None,
        "token_usage_available": token_available,
        "latency_s": latency,
    }


def _embedding_usage(
    snapshot: object, identity: Mapping[str, Any], *, allow_unavailable: bool = False,
) -> dict[str, int | bool | None]:
    if not isinstance(snapshot, Mapping):
        raise BenchmarkIntegrityError("LongMemEval embedding usage is absent")
    configured = identity.get("configured")
    if not isinstance(configured, bool) or snapshot.get("configured") is not configured:
        raise BenchmarkIntegrityError("LongMemEval embedding configured state drifted")
    expected = (
        identity.get("backend"), identity.get("quality"), identity.get("network_free"),
        identity.get("vector_space_key"), identity.get("dimension"),
        identity.get("identity_exact"), identity.get("reuse_scope"),
    )
    observed = (
        snapshot.get("backend"), snapshot.get("quality"), snapshot.get("network_free"),
        snapshot.get("model"), snapshot.get("dimension"),
        snapshot.get("identity_exact"), snapshot.get("reuse_scope"),
    )
    for marker in ("identity_available", "identity_consistent"):
        if marker in snapshot and not isinstance(snapshot.get(marker), bool):
            raise BenchmarkIntegrityError(
                f"LongMemEval embedding {marker} is malformed"
            )
    identity_available = bool(
        snapshot.get("identity_consistent", snapshot.get("identity_available"))
    )
    unavailable_identity = bool(
        configured and allow_unavailable and not identity_available
        and snapshot.get("backend") in {"unavailable", "mixed"}
        and snapshot.get("model") is None
        and snapshot.get("dimension") is None
    )
    if observed != expected and not unavailable_identity:
        raise BenchmarkIntegrityError("LongMemEval embedding execution identity drifted")
    measured: dict[str, int | float | None] = {}
    for field, marker in (
        ("calls", "calls_available"),
        ("request_attempts", "request_attempts_available"),
        ("successful_responses", "successful_responses_available"),
        ("input_count", "input_count_available"),
        ("input_characters", "input_characters_available"),
        ("latency_s", "latency_available"),
    ):
        flag = snapshot.get(marker)
        if not isinstance(flag, bool):
            raise BenchmarkIntegrityError(f"LongMemEval embedding {marker} is malformed")
        value = snapshot.get(field)
        normalized = (
            _finite_number(value, integer=field != "latency_s") if flag else None
        )
        if flag and normalized is None:
            raise BenchmarkIntegrityError(f"LongMemEval embedding {field} is malformed")
        if not flag and value is not None:
            raise BenchmarkIntegrityError(
                f"LongMemEval embedding {field} claims unavailable precision"
            )
        measured[field] = normalized
    cost_available = snapshot.get("cost_available")
    if not isinstance(cost_available, bool):
        raise BenchmarkIntegrityError("LongMemEval embedding cost availability is malformed")
    cost = snapshot.get("cost_usd")
    if cost_available and _finite_number(cost) is None:
        raise BenchmarkIntegrityError("LongMemEval embedding cost is malformed")
    if not cost_available and cost is not None:
        raise BenchmarkIntegrityError(
            "LongMemEval embedding cost claims unavailable precision"
        )
    tokens_available = snapshot.get("provider_token_usage_available")
    if not isinstance(tokens_available, bool):
        raise BenchmarkIntegrityError(
            "LongMemEval embedding token availability is malformed"
        )
    token_values: list[int] = []
    for field in ("prompt_tokens", "total_tokens"):
        value = snapshot.get(field)
        if tokens_available:
            normalized = _finite_number(value, integer=True)
            if normalized is None:
                raise BenchmarkIntegrityError(
                    f"LongMemEval embedding {field} is malformed"
                )
            token_values.append(normalized)
        elif value is not None:
            raise BenchmarkIntegrityError(
                f"LongMemEval embedding {field} claims unavailable precision"
            )
    if tokens_available and token_values[1] != token_values[0]:
        raise BenchmarkIntegrityError(
            "LongMemEval embedding token totals do not reconcile"
        )
    calls = measured["calls"]
    attempts = measured["request_attempts"]
    successes = measured["successful_responses"]
    if calls is not None and successes is not None and calls != successes:
        raise BenchmarkIntegrityError(
            "LongMemEval embedding successful call count disagrees"
        )
    if attempts is not None and successes is not None and attempts < successes:
        raise BenchmarkIntegrityError(
            "LongMemEval embedding attempts are below successes"
        )
    instances = snapshot.get("instances")
    if instances is not None and (
        isinstance(instances, bool) or not isinstance(instances, int)
        or instances <= 0
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval embedding instance count is malformed"
        )
    if not configured and (
        snapshot.get("calls") != 0 or snapshot.get("input_count") != 0
        or snapshot.get("backend") != "none" or snapshot.get("network_free") is not True
    ):
        raise BenchmarkIntegrityError("disabled LongMemEval embeddings report work")
    if identity.get("network_free") is True and tokens_available:
        raise BenchmarkIntegrityError(
            "network-free LongMemEval embedding claims provider tokens"
        )
    return {
        "identity_available": not unavailable_identity,
        "provider_tokens_available": tokens_available,
        "provider_tokens": token_values[1] if tokens_available else None,
    }


def _validate_model(
    value: object, *, label: str, secret_free_endpoint: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkIntegrityError(f"LongMemEval {label} model identity is malformed")
    required = (
        ("provider", "model", "endpoint_origin", "endpoint_sha256")
        if secret_free_endpoint else ("provider", "model", "base_url")
    )
    for field in required:
        item = value.get(field)
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise BenchmarkIntegrityError(f"LongMemEval {label} {field} is malformed")
    endpoint_field = "endpoint_origin" if secret_free_endpoint else "base_url"
    endpoint = validate_safe_endpoint(value[endpoint_field], label=label)
    if secret_free_endpoint:
        canonical = secret_free_endpoint_identity(endpoint, label=label)
        if (
            value["endpoint_origin"] != canonical["endpoint_origin"]
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value["endpoint_sha256"])
            is None
        ):
            raise BenchmarkIntegrityError(
                f"LongMemEval {label} endpoint identity is malformed"
            )
    normalize_extra_body(
        value.get("extra_body", value.get("effective_extra_body", {})), label=label
    )
    return value


def _validate_segment_extraction_canary(
    segment: Mapping[str, Any], *, pipeline: Mapping[str, Any], no_dream: bool,
    prompt_version: str,
) -> None:
    """Bind every segment probe to its actual work and pipeline posture."""

    attempted = segment.get("attempted_attempts")
    indexing_runs = segment.get("indexing_runs")
    has_question_work = type(attempted) is int and attempted > 0
    has_indexing_work = isinstance(indexing_runs, list) and bool(indexing_runs)
    report = segment.get("extraction_canary")
    if has_indexing_work or (has_question_work and not no_dream):
        mode = "required"
    elif has_question_work and no_dream:
        mode = "no_dream"
    elif segment.get("status") == "complete":
        mode = "no_pending_work"
    elif isinstance(report, Mapping):
        report_status = report.get("status")
        if report_status == "pending":
            mode = "pending"
        elif report_status == "failed":
            mode = "failed"
        elif report_status == "passed":
            mode = "required"
        elif report_status == "skipped_non_comparable" and no_dream:
            mode = "no_dream"
        else:
            raise BenchmarkIntegrityError(
                "LongMemEval zero-work extraction canary state is invalid"
            )
    else:
        raise BenchmarkIntegrityError(
            "LongMemEval extraction canary report is absent"
        )
    if isinstance(report, Mapping) and isinstance(report.get("client"), Mapping):
        client_fields = set(report["client"])
        if "base_url" in client_fields or not {
            "endpoint_origin", "endpoint_sha256",
        } <= client_fields:
            raise BenchmarkIntegrityError(
                "LongMemEval extraction canary endpoint identity is not secret-free"
            )
    try:
        validate_extraction_canary_report(
            report,
            expected_mode=mode,
            expected_client=(
                pipeline if mode in {"required", "failed"} else None
            ),
            require_client_closed=mode in {"required", "failed"},
            expected_prompt_version=prompt_version,
        )
    except BenchmarkIntegrityError as exc:
        raise BenchmarkIntegrityError(
            "LongMemEval extraction canary evidence is invalid"
        ) from exc


def _validate_embedding_identity(value: object) -> Mapping[str, Any]:
    try:
        return validate_public_embedding_identity(value)
    except (TypeError, ValueError) as exc:
        # Legacy identities contain the full request route/model and cannot be
        # safely restamped into the v57 producer space.  Reject them instead
        # of treating an unverifiable historical route as current evidence.
        raise BenchmarkIntegrityError(
            "LongMemEval embedding identity is malformed or obsolete"
        ) from exc


def _bounded_count(value: object, *, label: str, positive: bool = False) -> int:
    result = _finite_number(value, integer=True)
    if (
        result is None or result < (1 if positive else 0)
        or result > 2_147_483_647
    ):
        raise BenchmarkIntegrityError(f"LongMemEval {label} is malformed")
    return int(result)


def _canonical_indexing_report(
    value: object, *, require_current_failures: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkIntegrityError("LongMemEval indexing cycle report is malformed")
    if require_current_failures:
        required = _INDEXING_CYCLE_FAILURE_FIELDS | _INDEXING_REPORT_BOOLEAN_FIELDS
        if required - set(value):
            raise BenchmarkIntegrityError(
                "LongMemEval indexing cycle report lacks current failure fields"
            )
    result: dict[str, Any] = {}
    for field in _INDEXING_REPORT_FIELDS:
        default: object = (
            False if field in _INDEXING_REPORT_BOOLEAN_FIELDS
            else None if field in _INDEXING_REPORT_OPTIONAL_COUNT_FIELDS
            else "" if field == "aggregation_blocking"
            else 0
        )
        raw = value.get(field, default)
        if field in _INDEXING_REPORT_BOOLEAN_FIELDS:
            if not isinstance(raw, bool):
                raise BenchmarkIntegrityError(
                    f"LongMemEval indexing report {field!r} is malformed"
                )
            result[field] = raw
        elif field in _INDEXING_REPORT_OPTIONAL_COUNT_FIELDS:
            result[field] = (
                None if raw is None else _bounded_count(
                    raw, label=f"indexing report {field!r}"
                )
            )
        elif field == "aggregation_blocking":
            if raw not in _INDEXING_AGGREGATION_BLOCKING_VALUES:
                raise BenchmarkIntegrityError(
                    "LongMemEval indexing aggregation blocking mode is malformed"
                )
            result[field] = raw
        else:
            result[field] = _bounded_count(
                raw, label=f"indexing report {field!r}"
            )
    return result


def _canonical_reason_counts(
    value: object, *, label: str, allowed: frozenset[str],
) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BenchmarkIntegrityError(f"LongMemEval {label} is malformed")
    result: dict[str, int] = {}
    for reason, count in value.items():
        if reason not in allowed:
            raise BenchmarkIntegrityError(f"LongMemEval {label} has an unknown reason")
        result[str(reason)] = _bounded_count(
            count, label=f"{label} count", positive=True,
        )
    return dict(sorted(result.items()))


def _canonical_coverage_evidence(final: Mapping[str, Any]) -> dict[str, Any]:
    failures = _bounded_count(
        final.get("coverage_integrity_failures", 0),
        label="coverage-integrity failure count",
    )
    reasons = _canonical_reason_counts(
        final.get("coverage_integrity_failure_reasons", {}),
        label="coverage-integrity reason summary",
        allowed=frozenset(COVERAGE_INTEGRITY_FAILURE_REASONS),
    )
    if sum(reasons.values()) != failures:
        raise BenchmarkIntegrityError(
            "LongMemEval coverage-integrity reason summary is inconsistent"
        )
    config_version = final.get(
        "coverage_integrity_config_version", COVERAGE_INTEGRITY_CONFIG_VERSION,
    )
    if config_version != COVERAGE_INTEGRITY_CONFIG_VERSION:
        raise BenchmarkIntegrityError(
            "LongMemEval coverage-integrity config identity differs"
        )
    raw_details = final.get("coverage_integrity_failure_details", [])
    if not isinstance(raw_details, list) or len(raw_details) > LME_INDEXING_COVERAGE_DETAIL_LIMIT:
        raise BenchmarkIntegrityError(
            "LongMemEval coverage-integrity details are malformed/unbounded"
        )
    details: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    for item in raw_details:
        if not isinstance(item, Mapping) or set(item) != {
            "session_id", "config_version", "failure_reason", "occurrences",
            "first_detected_at", "last_detected_at",
        }:
            raise BenchmarkIntegrityError(
                "LongMemEval coverage-integrity detail shape is malformed"
            )
        session_id = item.get("session_id")
        if (
            not isinstance(session_id, str) or not session_id
            or len(session_id.encode("utf-8")) > 4096
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval coverage-integrity session identity is malformed"
            )
        session_hash = "sha256:" + hashlib.sha256(
            session_id.encode("utf-8")
        ).hexdigest()
        if session_hash in seen_sessions:
            raise BenchmarkIntegrityError(
                "LongMemEval coverage-integrity detail session is duplicated"
            )
        seen_sessions.add(session_hash)
        reason = item.get("failure_reason")
        if (
            item.get("config_version") != config_version
            or reason not in COVERAGE_INTEGRITY_FAILURE_REASONS
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval coverage-integrity detail identity differs"
            )
        first = item.get("first_detected_at")
        last = item.get("last_detected_at")
        if (
            not _valid_status_timestamp(first)
            or not _valid_status_timestamp(last)
            or first > last
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval coverage-integrity detail timestamps are malformed"
            )
        details.append({
            "session_id_hash": session_hash,
            "config_version": config_version,
            "failure_reason": reason,
            "occurrences": _bounded_count(
                item.get("occurrences"),
                label="coverage-integrity occurrence count", positive=True,
            ),
            "first_detected_at": first,
            "last_detected_at": last,
        })
        if details[-1]["occurrences"] > MAX_COVERAGE_INTEGRITY_OCCURRENCES:
            raise BenchmarkIntegrityError(
                "LongMemEval coverage-integrity occurrence count is unbounded"
            )
    truncated = final.get("coverage_integrity_failure_details_truncated", False)
    if not isinstance(truncated, bool):
        raise BenchmarkIntegrityError(
            "LongMemEval coverage-integrity truncation state is malformed"
        )
    detail_counts = Counter(item["failure_reason"] for item in details)
    if failures == 0 and (details or truncated):
        raise BenchmarkIntegrityError(
            "LongMemEval clean coverage state carries failure details"
        )
    if truncated:
        if len(details) != LME_INDEXING_COVERAGE_DETAIL_LIMIT or failures <= len(details):
            raise BenchmarkIntegrityError(
                "LongMemEval truncated coverage details have invalid bounds"
            )
        if any(detail_counts[key] > reasons.get(key, 0) for key in detail_counts):
            raise BenchmarkIntegrityError(
                "LongMemEval coverage detail counts exceed their summary"
            )
    elif len(details) != failures or dict(sorted(detail_counts.items())) != reasons:
        raise BenchmarkIntegrityError(
            "LongMemEval coverage details differ from their reason summary"
        )
    return {
        "failures": failures,
        "reasons": reasons,
        "details": details,
        "details_truncated": truncated,
        "config_version": config_version,
    }


def _canonical_final_indexing_status(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkIntegrityError("LongMemEval indexing final status is absent")
    if value.get("dream_status_schema") != DREAM_STATUS_SCHEMA_VERSION:
        raise BenchmarkIntegrityError(
            "LongMemEval indexing final status has an unsupported dream schema"
        )
    if value.get(
        "benchmark_indexing_status_schema"
    ) != BENCHMARK_INDEXING_STATUS_VERSION:
        raise BenchmarkIntegrityError(
            "LongMemEval indexing final status has an unsupported benchmark schema"
        )

    known_terminal = {"terminal_loss_chunks", "terminal_loss_reasons"}
    known_coverage = {
        "coverage_integrity_failures",
        "coverage_integrity_failure_reasons",
        "coverage_integrity_failure_details",
        "coverage_integrity_failure_details_truncated",
        "coverage_integrity_config_version",
    }
    for key in value:
        if not isinstance(key, str):
            continue
        unknown_health_field = (
            (
                key.startswith("pending_")
                and key not in _INDEXING_PENDING_FIELDS
                and key not in DREAM_STATUS_PHASE1_AUTHORITY_FIELDS
            )
            or (key.startswith("malformed_") and key not in _INDEXING_MALFORMED_FIELDS)
            or ("quarantined" in key and key not in _INDEXING_QUARANTINE_FIELDS)
            or (key.startswith("terminal_loss_") and key not in known_terminal)
            or (key.startswith("coverage_integrity_") and key not in known_coverage)
        )
        if unknown_health_field:
            raise BenchmarkIntegrityError(
                "LongMemEval indexing final status has an unknown health field"
            )
    missing_pending = _INDEXING_PENDING_FIELDS - set(value)
    missing_malformed = _INDEXING_MALFORMED_FIELDS - set(value)
    missing_quarantined = _INDEXING_QUARANTINE_FIELDS - set(value)
    missing_terminal = known_terminal - set(value)
    missing_coverage = known_coverage - set(value)
    missing_aggregation = {
        "aggregation_enabled", "aggregation_publication_generation",
        "aggregation_material_binding", "aggregation_material_revision",
        *DREAM_STATUS_AGGREGATION_AUTHORITY_FIELDS,
        *DREAM_STATUS_AGGREGATION_MATERIAL_AUTHORITY_FIELDS,
    } - set(value)
    if (
        missing_pending or missing_malformed or missing_quarantined
        or missing_terminal or missing_coverage or missing_aggregation
        or "in_progress" not in value
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval indexing final status lacks required health counters"
        )
    pending = {
        key: _bounded_count(
            value[key], label=f"indexing final status {key!r}"
        )
        for key in sorted(_INDEXING_PENDING_FIELDS)
    }
    malformed = {
        key: _bounded_count(
            value[key], label=f"indexing final status {key!r}"
        )
        for key in sorted(_INDEXING_MALFORMED_FIELDS)
    }
    quarantined = {
        key: _bounded_count(
            value[key], label=f"indexing final status {key!r}"
        )
        for key in sorted(_INDEXING_QUARANTINE_FIELDS)
    }
    terminal_reasons = _canonical_reason_counts(
        value.get("terminal_loss_reasons", {}),
        label="terminal-loss reason summary",
        allowed=frozenset({"source_manifest_unrecoverable"}),
    )
    terminal_chunks = _bounded_count(
        value.get("terminal_loss_chunks", 0), label="terminal-loss count"
    )
    if sum(terminal_reasons.values()) != terminal_chunks:
        raise BenchmarkIntegrityError(
            "LongMemEval terminal-loss reason summary is inconsistent"
        )
    in_progress = value["in_progress"]
    if not isinstance(in_progress, bool):
        raise BenchmarkIntegrityError(
            "LongMemEval indexing in-progress state is malformed"
        )
    aggregation_enabled = value["aggregation_enabled"]
    if not isinstance(aggregation_enabled, bool):
        raise BenchmarkIntegrityError(
            "LongMemEval aggregation enabled state is malformed"
        )
    if aggregation_enabled:
        generation_keys = tuple(
            value[field] for field in DREAM_STATUS_AGGREGATION_AUTHORITY_FIELDS
        )
        try:
            binding = validate_aggregation_generation_binding(
                value["aggregation_publication_generation"]
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkIntegrityError(
                "LongMemEval aggregation generation proof is malformed"
            ) from exc
        producer = binding["producer"]
        if (
            any(not isinstance(key, str) or not key for key in generation_keys)
            or len(set(generation_keys)) != 1
            or binding["generation_key"] != generation_keys[0]
            or producer["identity_exact"] is not True
            or producer["reuse_scope"] != "durable"
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval aggregation generation is not current and exact"
            )
        aggregation_generation = {
            "enabled": True,
            "generation_key": generation_keys[0],
            "identity_exact": True,
            "producer_identity_sha256": producer["identity_sha256"],
            "generation_binding": binding,
        }
        material_keys = tuple(
            value[field]
            for field in DREAM_STATUS_AGGREGATION_MATERIAL_AUTHORITY_FIELDS
        )
        try:
            material_binding = validate_aggregation_material_binding(
                value["aggregation_material_binding"]
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkIntegrityError(
                "LongMemEval aggregation material proof is malformed"
            ) from exc
        material_producer = material_binding["embedding_producer"]
        material_revision = value["aggregation_material_revision"]
        if (
            any(not isinstance(key, str) or not key for key in material_keys)
            or len(set(material_keys)) != 1
            or material_binding["material_epoch_key"] != material_keys[0]
            or isinstance(material_revision, bool)
            or not isinstance(material_revision, int)
            or material_revision < 0
            or material_binding["material_revision"] != material_revision
            or material_producer["identity_exact"] is not True
            or material_producer["reuse_scope"] != "durable"
            or material_binding["phase1_scope_identity_exact"] is not True
            or material_binding["phase1_scope_reuse_scope"] != "durable"
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval aggregation material is not current and exact"
            )
        aggregation_material = {
            "enabled": True,
            "material_epoch_key": material_keys[0],
            "material_revision": material_revision,
            "identity_exact": True,
            "material_binding": material_binding,
        }
    else:
        if (
            any(value[field] is not None
                for field in DREAM_STATUS_AGGREGATION_AUTHORITY_FIELDS)
            or any(value[field] is not None for field in
                   DREAM_STATUS_AGGREGATION_MATERIAL_AUTHORITY_FIELDS)
            or value["aggregation_publication_generation"] is not None
            or value["aggregation_material_binding"] is not None
            or value["aggregation_material_revision"] is not None
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval disabled aggregation state is not inert"
            )
        aggregation_generation = {
            "enabled": False,
            "generation_key": None,
            "identity_exact": None,
            "producer_identity_sha256": None,
            "generation_binding": None,
        }
        aggregation_material = {
            "enabled": False,
            "material_epoch_key": None,
            "material_revision": None,
            "identity_exact": None,
            "material_binding": None,
        }
    return {
        "dream_status_schema": DREAM_STATUS_SCHEMA_VERSION,
        "benchmark_indexing_status_schema": BENCHMARK_INDEXING_STATUS_VERSION,
        "pending": pending,
        "malformed": malformed,
        "quarantined": quarantined,
        "terminal_loss": {
            "chunks": terminal_chunks,
            "reasons": terminal_reasons,
        },
        "coverage_integrity": _canonical_coverage_evidence(value),
        "in_progress": in_progress,
        "aggregation_generation": aggregation_generation,
        "aggregation_material": aggregation_material,
    }


def _indexing_failure(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise BenchmarkIntegrityError("LongMemEval failed indexing lacks a reason")
    code = value
    exception_type: str | None = None
    if value.startswith("cycle_exception:"):
        code = "cycle_exception"
        match = re.match(r"cycle_exception:\s*([^:]+)", value)
        if match is not None and _SAFE_EXCEPTION_TYPE.fullmatch(match.group(1).strip()):
            exception_type = match.group(1).strip()
    if code not in _INDEXING_FAILURE_CODES:
        raise BenchmarkIntegrityError("LongMemEval indexing failure reason is unknown")
    return {"code": code, "exception_type": exception_type}


def canonicalize_lme_indexing_summary(summary: object) -> dict[str, Any]:
    """Project a convergence result into bounded, source-free LME evidence."""

    if not isinstance(summary, Mapping):
        raise BenchmarkIntegrityError("LongMemEval indexing summary is absent")
    complete = summary.get("complete")
    healthy = summary.get("healthy")
    if not isinstance(complete, bool) or not isinstance(healthy, bool):
        raise BenchmarkIntegrityError("LongMemEval indexing completion state is malformed")
    if healthy and summary.get("failure_reason") is not None:
        raise BenchmarkIntegrityError(
            "LongMemEval healthy indexing claims a failure reason"
        )
    failure_reason = summary.get("failure_reason")
    if not healthy and failure_reason is None:
        final_raw = summary.get("final_status")
        if isinstance(final_raw, Mapping) and _finite_number(
            final_raw.get("coverage_integrity_failures", 0), integer=True
        ) not in {None, 0}:
            failure_reason = "coverage_integrity_failure"
        elif isinstance(final_raw, Mapping) and _finite_number(
            final_raw.get("terminal_loss_chunks", 0), integer=True
        ) not in {None, 0}:
            failure_reason = "terminal_extraction_source_loss"
        elif isinstance(final_raw, Mapping) and any(
            isinstance(key, str) and "quarantined" in key
            and _finite_number(value, integer=True) not in {None, 0}
            for key, value in final_raw.items()
        ):
            failure_reason = "quarantined_extraction"
        elif isinstance(final_raw, Mapping) and any(
            key in _INDEXING_MALFORMED_FIELDS
            and _finite_number(value, integer=True) not in {None, 0}
            for key, value in final_raw.items()
        ):
            failure_reason = "malformed_durable_state"
    failure = None if healthy else _indexing_failure(failure_reason)
    reports_raw = summary.get("reports")
    if not isinstance(reports_raw, list):
        raise BenchmarkIntegrityError("LongMemEval indexing cycle reports are malformed")
    reports = [
        _canonical_indexing_report(
            report,
            require_current_failures=(
                failure is None
                or failure["code"] != "malformed_cycle_failure_report"
            ),
        )
        for report in reports_raw
    ]
    try:
        final_status: dict[str, Any] | None = _canonical_final_indexing_status(
            summary.get("final_status")
        )
    except BenchmarkIntegrityError:
        if failure is None or failure["code"] not in {
            "cycle_exception", "timeout_before_cycle", "timeout_during_cycle",
            "timeout_after_cycle",
            "malformed_status_shape",
            "malformed_pending_backlog", "malformed_quarantine_state",
            "malformed_terminal_loss_state",
            "malformed_coverage_integrity_state",
            "malformed_cycle_failure_report", "malformed_durable_state",
        }:
            raise
        final_status = None
    cycles = _bounded_count(summary.get("cycles"), label="indexing cycle count")
    result: dict[str, Any] = {
        "schema": LME_INDEXING_SUMMARY_VERSION,
        "outcome": "success" if failure is None else "failure",
        "cycles": cycles,
        "max_cycles": _bounded_count(
            summary.get("max_cycles"), label="indexing max cycle count", positive=True,
        ),
        "timeout_s": summary.get("timeout_s"),
        "elapsed_s": summary.get("elapsed_s"),
        "complete": complete,
        "healthy": healthy,
        "reports": reports,
        "final_status": final_status,
        "cleanup_errors": [],
    }
    if failure is not None:
        result["failure"] = failure
    _validate_versioned_indexing(result, allow_failure=True, require_healthy=True)
    return result


def _validate_canonical_final_status(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "dream_status_schema", "benchmark_indexing_status_schema",
        "pending", "malformed", "quarantined", "terminal_loss",
        "coverage_integrity", "in_progress", "aggregation_generation",
        "aggregation_material",
    }:
        raise BenchmarkIntegrityError("LongMemEval indexing final status is malformed")
    if (
        value.get("dream_status_schema") != DREAM_STATUS_SCHEMA_VERSION
        or value.get("benchmark_indexing_status_schema")
        != BENCHMARK_INDEXING_STATUS_VERSION
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval indexing final status schema identity differs"
        )
    for label in ("pending", "malformed", "quarantined"):
        counts = value.get(label)
        expected_count_fields = (
            _INDEXING_PENDING_FIELDS
            if label == "pending"
            else _INDEXING_MALFORMED_FIELDS
            if label == "malformed"
            else _INDEXING_QUARANTINE_FIELDS
        )
        if not isinstance(counts, Mapping) or set(counts) != expected_count_fields:
            raise BenchmarkIntegrityError(
                f"LongMemEval indexing {label} counters are malformed"
            )
        for key, count in counts.items():
            _bounded_count(count, label=f"indexing {label} counter")
    terminal = value.get("terminal_loss")
    if not isinstance(terminal, Mapping) or set(terminal) != {"chunks", "reasons"}:
        raise BenchmarkIntegrityError("LongMemEval terminal-loss evidence is malformed")
    terminal_count = _bounded_count(
        terminal.get("chunks"), label="terminal-loss count"
    )
    terminal_reasons = _canonical_reason_counts(
        terminal.get("reasons"), label="terminal-loss reason summary",
        allowed=frozenset({"source_manifest_unrecoverable"}),
    )
    if terminal_count != sum(terminal_reasons.values()):
        raise BenchmarkIntegrityError(
            "LongMemEval terminal-loss evidence is inconsistent"
        )
    coverage = value.get("coverage_integrity")
    if not isinstance(coverage, Mapping) or set(coverage) != {
        "failures", "reasons", "details", "details_truncated", "config_version",
    }:
        raise BenchmarkIntegrityError(
            "LongMemEval coverage-integrity evidence is malformed"
        )
    failures = _bounded_count(
        coverage.get("failures"), label="coverage-integrity failure count"
    )
    reasons = _canonical_reason_counts(
        coverage.get("reasons"), label="coverage-integrity reason summary",
        allowed=frozenset(COVERAGE_INTEGRITY_FAILURE_REASONS),
    )
    if failures != sum(reasons.values()) or coverage.get(
        "config_version"
    ) != COVERAGE_INTEGRITY_CONFIG_VERSION:
        raise BenchmarkIntegrityError(
            "LongMemEval coverage-integrity summary is inconsistent"
        )
    details = coverage.get("details")
    truncated = coverage.get("details_truncated")
    if (
        not isinstance(details, list)
        or len(details) > LME_INDEXING_COVERAGE_DETAIL_LIMIT
        or not isinstance(truncated, bool)
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval coverage-integrity details are malformed/unbounded"
        )
    detail_counts: Counter[str] = Counter()
    seen_sessions: set[str] = set()
    for detail in details:
        if not isinstance(detail, Mapping) or set(detail) != {
            "session_id_hash", "config_version", "failure_reason", "occurrences",
            "first_detected_at", "last_detected_at",
        }:
            raise BenchmarkIntegrityError(
                "LongMemEval coverage-integrity detail shape is malformed"
            )
        session_hash = detail.get("session_id_hash")
        if (
            not isinstance(session_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", session_hash) is None
            or session_hash in seen_sessions
            or detail.get("config_version") != COVERAGE_INTEGRITY_CONFIG_VERSION
            or detail.get("failure_reason") not in COVERAGE_INTEGRITY_FAILURE_REASONS
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval coverage-integrity detail identity is malformed"
            )
        seen_sessions.add(session_hash)
        _bounded_count(
            detail.get("occurrences"),
            label="coverage-integrity occurrence count", positive=True,
        )
        first = detail.get("first_detected_at")
        last = detail.get("last_detected_at")
        if (
            not _valid_status_timestamp(first)
            or not _valid_status_timestamp(last)
            or first > last
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval coverage-integrity detail timestamps are malformed"
            )
        detail_counts[str(detail["failure_reason"])] += 1
    if failures == 0 and (details or truncated):
        raise BenchmarkIntegrityError(
            "LongMemEval clean coverage state carries failure details"
        )
    if truncated:
        if len(details) != LME_INDEXING_COVERAGE_DETAIL_LIMIT or failures <= len(details):
            raise BenchmarkIntegrityError(
                "LongMemEval truncated coverage details have invalid bounds"
            )
        if any(detail_counts[key] > reasons.get(key, 0) for key in detail_counts):
            raise BenchmarkIntegrityError(
                "LongMemEval coverage details exceed their summary"
            )
    elif len(details) != failures or dict(sorted(detail_counts.items())) != reasons:
        raise BenchmarkIntegrityError(
            "LongMemEval coverage details differ from their summary"
        )
    if not isinstance(value.get("in_progress"), bool):
        raise BenchmarkIntegrityError(
            "LongMemEval indexing in-progress state is malformed"
        )
    aggregation = value.get("aggregation_generation")
    if not isinstance(aggregation, Mapping) or set(aggregation) != {
        "enabled", "generation_key", "identity_exact",
        "producer_identity_sha256", "generation_binding",
    }:
        raise BenchmarkIntegrityError(
            "LongMemEval aggregation generation certificate is malformed"
        )
    if not isinstance(aggregation.get("enabled"), bool):
        raise BenchmarkIntegrityError(
            "LongMemEval aggregation generation enabled state is malformed"
        )
    if aggregation["enabled"]:
        try:
            generation_binding = validate_aggregation_generation_binding(
                aggregation.get("generation_binding")
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkIntegrityError(
                "LongMemEval aggregation producer proof is malformed"
            ) from exc
        producer = generation_binding["producer"]
        if (
            not isinstance(aggregation.get("generation_key"), str)
            or re.fullmatch(
                r"hymem-aggregation-generation-v1:[0-9a-f]{64}",
                aggregation["generation_key"],
            ) is None
            or aggregation.get("identity_exact") is not True
            or not isinstance(aggregation.get("producer_identity_sha256"), str)
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                aggregation["producer_identity_sha256"],
            ) is None
            or producer["identity_exact"] is not True
            or producer["reuse_scope"] != "durable"
            or producer["identity_sha256"]
            != aggregation["producer_identity_sha256"]
            or generation_binding["generation_key"]
            != aggregation["generation_key"]
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval aggregation generation certificate is inexact"
            )
    elif any(
        aggregation.get(field) is not None
        for field in (
            "generation_key", "identity_exact", "producer_identity_sha256",
            "generation_binding",
        )
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval disabled aggregation generation is not inert"
        )
    material = value.get("aggregation_material")
    if not isinstance(material, Mapping) or set(material) != {
        "enabled", "material_epoch_key", "material_revision",
        "identity_exact", "material_binding",
    } or material.get("enabled") is not aggregation.get("enabled"):
        raise BenchmarkIntegrityError(
            "LongMemEval aggregation material certificate is malformed"
        )
    if material["enabled"]:
        try:
            material_binding = validate_aggregation_material_binding(
                material.get("material_binding")
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkIntegrityError(
                "LongMemEval aggregation material proof is malformed"
            ) from exc
        producer = material_binding["embedding_producer"]
        if (
            material.get("material_epoch_key")
            != material_binding["material_epoch_key"]
            or material.get("material_revision")
            != material_binding["material_revision"]
            or material.get("identity_exact") is not True
            or producer["identity_exact"] is not True
            or producer["reuse_scope"] != "durable"
            or material_binding["phase1_scope_identity_exact"] is not True
            or material_binding["phase1_scope_reuse_scope"] != "durable"
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval aggregation material certificate is inexact"
            )
    elif any(
        material.get(field) is not None for field in (
            "material_epoch_key", "material_revision", "identity_exact",
            "material_binding",
        )
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval disabled aggregation material is not inert"
        )
    return value


def _validated_pipeline_aggregation_producer(
    pipeline: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the manifested producer even when no final status exists."""

    try:
        expected_producer = validate_aggregation_producer_binding(
            pipeline.get("aggregation_producer")
        )
        expected_declaration = expected_producer.get("declaration")
        if not isinstance(expected_declaration, Mapping):
            raise ValueError("aggregation declaration is absent")
        from hymem.contrib.openai_client import (
            openai_compatible_producer_declaration,
        )
        from hymem.extraction.producer import (
            producer_binding_from_typed_declaration,
        )
        derived_producer = producer_binding_from_typed_declaration(
            openai_compatible_producer_declaration(
                model=pipeline.get("model"),
                endpoint=pipeline.get("endpoint_origin"),
                thinking_mode=pipeline.get("thinking_mode"),
                effective_extra_body=pipeline.get("effective_extra_body"),
                transport_package_version=pipeline.get(
                    "transport_package_version"
                ),
                request_timeout_seconds=pipeline.get(
                    "request_timeout_seconds"
                ),
                deployment_revision_sha256=pipeline.get(
                    "deployment_revision_sha256"
                ),
                deployment_tenant_sha256=pipeline.get(
                    "deployment_tenant_sha256"
                ),
                require_consistent_thinking=True,
            ),
            declaration_hook="aggregation_producer_declaration",
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkIntegrityError(
            "LongMemEval memory pipeline aggregation identity is malformed"
        ) from exc
    derived_declaration = derived_producer.get("declaration")
    if (
        not isinstance(expected_declaration, Mapping)
        or not isinstance(derived_declaration, Mapping)
        or expected_declaration.get("endpoint_origin")
        != pipeline.get("endpoint_origin")
        or expected_declaration.get("endpoint_sha256")
        != pipeline.get("endpoint_sha256")
        or {
            key: value for key, value in expected_declaration.items()
            if key != "endpoint_sha256"
        } != {
            key: value for key, value in derived_declaration.items()
            if key != "endpoint_sha256"
        }
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval memory pipeline aggregation identity is malformed"
        )
    return expected_producer


def _validate_aggregation_pipeline_binding(
    final_status: Mapping[str, Any] | None,
    pipeline: Mapping[str, Any],
) -> None:
    """Cross-check an enabled aggregation producer against the run target."""

    expected_producer = _validated_pipeline_aggregation_producer(pipeline)
    if final_status is None:
        return
    certificate = final_status.get("aggregation_generation")
    if not isinstance(certificate, Mapping) or not certificate.get("enabled"):
        return
    binding = validate_aggregation_generation_binding(
        certificate.get("generation_binding")
    )
    if binding["producer"] != expected_producer:
        raise BenchmarkIntegrityError(
            "LongMemEval aggregation producer differs from memory pipeline target"
        )


def _validate_versioned_indexing(
    summary: Mapping[str, Any], *, require_healthy: bool, allow_failure: bool,
) -> bool:
    outcome = summary.get("outcome")
    expected_fields = set(_INDEXING_SUMMARY_COMMON_FIELDS)
    if outcome == "failure":
        expected_fields.add("failure")
    elif outcome != "success":
        raise BenchmarkIntegrityError("LongMemEval indexing outcome is malformed")
    if set(summary) != expected_fields:
        raise BenchmarkIntegrityError("LongMemEval indexing summary fields differ")
    if summary.get("schema") != LME_INDEXING_SUMMARY_VERSION:
        raise BenchmarkIntegrityError("LongMemEval indexing summary schema differs")
    complete = summary.get("complete")
    healthy = summary.get("healthy")
    if not isinstance(complete, bool) or not isinstance(healthy, bool):
        raise BenchmarkIntegrityError("LongMemEval indexing completion state is malformed")
    cycles = _bounded_count(summary.get("cycles"), label="indexing cycle count")
    max_cycles = _bounded_count(
        summary.get("max_cycles"), label="indexing max cycle count", positive=True,
    )
    elapsed = _finite_number(summary.get("elapsed_s"))
    timeout = _finite_number(summary.get("timeout_s"))
    if (
        cycles > max_cycles or elapsed is None or elapsed < 0
        or timeout is None or timeout <= 0
    ):
        raise BenchmarkIntegrityError("LongMemEval indexing bounds are inconsistent")
    reports = summary.get("reports")
    if not isinstance(reports, list) or len(reports) != cycles:
        raise BenchmarkIntegrityError("LongMemEval indexing cycle reports are malformed")
    for report in reports:
        canonical = _canonical_indexing_report(report)
        if not isinstance(report, Mapping) or dict(report) != canonical:
            raise BenchmarkIntegrityError("LongMemEval indexing cycle report fields differ")
    cleanup = summary.get("cleanup_errors")
    if not isinstance(cleanup, list) or len(cleanup) > 2:
        raise BenchmarkIntegrityError("LongMemEval indexing cleanup evidence is malformed")
    for item in cleanup:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"stage", "exception_type"}
            or item.get("stage") not in {
                "dream_fork_close", "query_cache_invalidation",
            }
            or not isinstance(item.get("exception_type"), str)
            or _SAFE_EXCEPTION_TYPE.fullmatch(item["exception_type"]) is None
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval indexing cleanup evidence is malformed"
            )
    if outcome == "success" and cleanup:
        raise BenchmarkIntegrityError(
            "LongMemEval successful indexing cannot contain cleanup failures"
        )
    final = summary.get("final_status")
    if final is not None:
        final = _validate_canonical_final_status(final)
    if outcome == "success":
        if not complete or not healthy or final is None or cycles <= 0 or elapsed > timeout:
            raise BenchmarkIntegrityError("LongMemEval successful indexing state is inconsistent")
        if any(reports[-1][key] for key in _INDEXING_REPORT_BOOLEAN_FIELDS) or any(
            reports[-1][key] != 0 for key in _INDEXING_CYCLE_FAILURE_FIELDS
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval successful indexing final cycle is not clean"
            )
        pending = sum(final["pending"].values())
        malformed = sum(final["malformed"].values())
        quarantined = sum(final["quarantined"].values())
        terminal = final["terminal_loss"]["chunks"]
        coverage = final["coverage_integrity"]["failures"]
        if (
            pending or malformed or quarantined or terminal or coverage
            or final["in_progress"]
        ):
            raise BenchmarkIntegrityError("LongMemEval successful indexing is not healthy")
        return True
    if not allow_failure:
        raise BenchmarkIntegrityError("LongMemEval indexing did not complete healthy")
    failure = summary.get("failure")
    if not isinstance(failure, Mapping) or set(failure) != {"code", "exception_type"}:
        raise BenchmarkIntegrityError("LongMemEval indexing failure evidence is malformed")
    code = failure.get("code")
    exception_type = failure.get("exception_type")
    if code not in _INDEXING_FAILURE_CODES or (
        exception_type is not None and (
            not isinstance(exception_type, str)
            or _SAFE_EXCEPTION_TYPE.fullmatch(exception_type) is None
        )
    ) or (code == "cycle_exception") is not (exception_type is not None):
        raise BenchmarkIntegrityError("LongMemEval indexing failure evidence is malformed")
    if healthy or (complete and code not in _MECHANICALLY_COMPLETE_FAILURE_CODES):
        raise BenchmarkIntegrityError("LongMemEval failed indexing claims usable completion")
    if final is None and code not in {
        "cycle_exception", "timeout_before_cycle", "timeout_during_cycle",
        "timeout_after_cycle",
        "malformed_status_shape",
        "malformed_pending_backlog",
        "malformed_quarantine_state", "malformed_terminal_loss_state",
        "malformed_coverage_integrity_state",
        "malformed_cycle_failure_report", "malformed_durable_state",
    }:
        raise BenchmarkIntegrityError("LongMemEval indexing failure lacks final status")
    if code == "coverage_integrity_failure" and (
        final is None or final["coverage_integrity"]["failures"] <= 0
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval coverage failure lacks durable coverage evidence"
        )
    if code == "quarantined_extraction" and (
        final is None or not any(final["quarantined"].values())
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval quarantine failure lacks durable quarantine evidence"
        )
    if code == "terminal_extraction_source_loss" and (
        final is None or final["terminal_loss"]["chunks"] <= 0
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval terminal-loss failure lacks durable loss evidence"
        )
    if code == "malformed_durable_state" and (
        final is None or not any(final["malformed"].values())
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval malformed-state failure lacks durable evidence"
        )
    if code in {
        "quarantined_extraction", "terminal_extraction_source_loss",
        "malformed_durable_state",
    } and not complete:
        raise BenchmarkIntegrityError(
            "LongMemEval extraction-loss failure lacks mechanical completion"
        )
    if complete:
        if (
            final is None or cycles <= 0 or sum(final["pending"].values()) != 0
            or final["in_progress"]
            or any(reports[-1][key] for key in _INDEXING_REPORT_BOOLEAN_FIELDS)
            or any(
                reports[-1][key] != 0
                for key in _INDEXING_CYCLE_FAILURE_FIELDS
            )
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval mechanically complete failure has blocking work"
            )
    if code == "max_cycles_exhausted" and cycles != max_cycles:
        raise BenchmarkIntegrityError(
            "LongMemEval max-cycle failure did not exhaust its cycle bound"
        )
    if code == "timeout_before_cycle" and elapsed < timeout:
        raise BenchmarkIntegrityError(
            "LongMemEval timeout failure precedes its recorded bound"
        )
    if code == "timeout_during_cycle" and elapsed < timeout:
        raise BenchmarkIntegrityError(
            "LongMemEval in-cycle timeout precedes its recorded bound"
        )
    if code == "timeout_after_cycle" and elapsed < timeout:
        raise BenchmarkIntegrityError(
            "LongMemEval after-cycle timeout precedes its recorded bound"
        )
    return False


def _validate_legacy_indexing(
    summary: Mapping[str, Any], *, require_healthy: bool, allow_failure: bool,
) -> bool:
    """Read pre-v2 evidence conservatively; new executions never emit it."""

    allowed = {
        "question_id", "cycles", "max_cycles", "timeout_s", "elapsed_s",
        "complete", "healthy", "failure_reason", "reports", "final_status",
        "quarantined", "cleanup_errors",
    }
    if not set(summary) <= allowed:
        raise BenchmarkIntegrityError("legacy LongMemEval indexing summary has extra fields")
    complete = summary.get("complete")
    healthy = summary.get("healthy")
    if not isinstance(complete, bool) or not isinstance(healthy, bool):
        raise BenchmarkIntegrityError("LongMemEval indexing completion state is malformed")
    cycles = _bounded_count(summary.get("cycles"), label="indexing cycle count")
    max_cycles = _bounded_count(
        summary.get("max_cycles"), label="indexing max cycle count", positive=True,
    )
    elapsed = _finite_number(summary.get("elapsed_s"))
    timeout = _finite_number(summary.get("timeout_s"))
    if cycles > max_cycles or elapsed is None or elapsed < 0 or timeout is None or timeout <= 0:
        raise BenchmarkIntegrityError("LongMemEval indexing bounds are inconsistent")
    reports = summary.get("reports")
    if not isinstance(reports, list) or len(reports) != cycles or any(
        not isinstance(report, Mapping)
        or not isinstance(report.get("budget_exhausted"), bool)
        or not isinstance(report.get("skipped_locked"), bool)
        for report in reports
    ):
        raise BenchmarkIntegrityError("LongMemEval indexing cycle reports are malformed")
    reason = summary.get("failure_reason")
    failure = None if healthy else _indexing_failure(reason)
    final_raw = summary.get("final_status")
    try:
        canonical_final: Mapping[str, Any] | None = (
            _canonical_final_indexing_status(final_raw)
        )
    except BenchmarkIntegrityError:
        if failure is None or failure["code"] not in {
            "cycle_exception", "timeout_before_cycle", "timeout_during_cycle",
            "timeout_after_cycle",
            "malformed_status_shape",
            "malformed_pending_backlog", "malformed_quarantine_state",
            "malformed_terminal_loss_state", "malformed_coverage_integrity_state",
            "malformed_cycle_failure_report", "malformed_durable_state",
        }:
            raise
        canonical_final = None
    expected_quarantined = {
        key: value for key, value in final_raw.items()
        if isinstance(key, str) and "quarantined" in key
        and _finite_number(value, integer=True) is not None and int(value) > 0
    } if isinstance(final_raw, Mapping) else {}
    quarantined = summary.get("quarantined")
    if not isinstance(quarantined, Mapping) or dict(quarantined) != expected_quarantined:
        raise BenchmarkIntegrityError(
            "LongMemEval indexing quarantine summary differs from final status"
        )
    computed_healthy = bool(
        complete and canonical_final is not None
        and not sum(canonical_final["pending"].values())
        and not sum(canonical_final["malformed"].values())
        and not sum(canonical_final["quarantined"].values())
        and canonical_final["terminal_loss"]["chunks"] == 0
        and canonical_final["coverage_integrity"]["failures"] == 0
        and not canonical_final["in_progress"]
    )
    if healthy is not computed_healthy:
        raise BenchmarkIntegrityError("LongMemEval indexing health flag is inconsistent")
    if healthy:
        if reason is not None or cycles <= 0 or elapsed > timeout:
            raise BenchmarkIntegrityError("LongMemEval successful indexing state is inconsistent")
        if (
            reports[-1].get("budget_exhausted")
            or reports[-1].get("skipped_locked")
            or _finite_number(
                reports[-1].get("aggregation_fusion_failures", 0), integer=True
            ) != 0
            or _finite_number(
                reports[-1].get("aggregation_build_exceptions", 0), integer=True
            ) != 0
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval successful indexing final cycle is not clean"
            )
        return True
    if not allow_failure:
        raise BenchmarkIntegrityError("LongMemEval indexing did not complete healthy")
    if failure["code"] == "cycle_exception":
        if not isinstance(reason, str) or re.fullmatch(
            r"cycle_exception:\s*[A-Za-z_][A-Za-z0-9_.]{0,127}", reason
        ) is None:
            raise BenchmarkIntegrityError(
                "legacy LongMemEval cycle failure contains unbounded detail"
            )
    elif reason != failure["code"]:
        raise BenchmarkIntegrityError(
            "legacy LongMemEval indexing failure reason is malformed"
        )
    if complete and failure["code"] not in _MECHANICALLY_COMPLETE_FAILURE_CODES:
        raise BenchmarkIntegrityError("LongMemEval failed indexing claims usable completion")
    if failure["code"] == "coverage_integrity_failure" and (
        canonical_final is None
        or canonical_final["coverage_integrity"]["failures"] <= 0
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval coverage failure lacks durable coverage evidence"
        )
    if failure["code"] in {
        "quarantined_extraction", "terminal_extraction_source_loss",
    } and not complete:
        raise BenchmarkIntegrityError(
            "LongMemEval extraction-loss failure lacks mechanical completion"
        )
    if complete and (
        canonical_final is None
        or cycles <= 0 or sum(canonical_final["pending"].values()) != 0
        or sum(canonical_final["malformed"].values()) != 0
        or canonical_final["in_progress"]
        or reports[-1].get("budget_exhausted") is not False
        or reports[-1].get("skipped_locked") is not False
        or _finite_number(
            reports[-1].get("aggregation_fusion_failures", 0), integer=True
        ) != 0
        or _finite_number(
            reports[-1].get("aggregation_build_exceptions", 0), integer=True
        ) != 0
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval mechanically complete failure has blocking work"
        )
    if failure["code"] == "max_cycles_exhausted" and cycles != max_cycles:
        raise BenchmarkIntegrityError(
            "LongMemEval max-cycle failure did not exhaust its cycle bound"
        )
    if failure["code"] in {
        "timeout_before_cycle", "timeout_during_cycle", "timeout_after_cycle",
    } and elapsed < timeout:
        raise BenchmarkIntegrityError(
            "LongMemEval timeout failure precedes its recorded bound"
        )
    cleanup = summary.get("cleanup_errors", [])
    if not isinstance(cleanup, list) or len(cleanup) > 2 or any(
        not isinstance(item, str) or re.fullmatch(
            r"(?:dream_fork_close|query_cache_invalidation):\s*"
            r"[A-Za-z_][A-Za-z0-9_.]{0,127}", item
        ) is None
        for item in cleanup
    ):
        raise BenchmarkIntegrityError("LongMemEval indexing cleanup evidence is malformed")
    return False


def _validate_indexing(
    summary: object, *, require_healthy: bool = True,
    allow_incomplete: bool = False,
) -> bool:
    """Validate exact current indexing evidence or a conservative legacy record."""

    if not isinstance(summary, Mapping):
        raise BenchmarkIntegrityError("LongMemEval indexing summary is absent")
    if summary.get("schema") == LME_INDEXING_SUMMARY_VERSION:
        return _validate_versioned_indexing(
            summary, require_healthy=require_healthy,
            allow_failure=allow_incomplete,
        )
    return _validate_legacy_indexing(
        summary, require_healthy=require_healthy,
        allow_failure=allow_incomplete,
    )


def _scores_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        qtype = row["question_type"]
        buckets.setdefault(qtype, []).append(bool(row["correct"]))
    result: dict[str, dict[str, float | int]] = {}
    all_values: list[bool] = []
    for qtype in sorted(buckets):
        values = buckets[qtype]
        all_values.extend(values)
        result[qtype] = {
            "accuracy": sum(values) / len(values) * 100.0,
            "count": len(values),
        }
    result["OVERALL"] = {
        "accuracy": sum(all_values) / len(all_values) * 100.0 if all_values else 0.0,
        "count": len(all_values),
    }
    return result


def _abstention_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def failed(row: Mapping[str, Any]) -> bool:
        return bool(row.get("judge_error") or row.get("benchmark_failure"))

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [item for item in items if not failed(item)]
        return {
            "accuracy": (
                sum(bool(item.get("correct")) for item in items) / len(items)
                if items else None
            ),
            "count": len(items),
            "benchmark_failures": sum(failed(item) for item in items),
            "conditional_valid_accuracy": (
                sum(bool(item.get("correct")) for item in valid) / len(valid)
                if valid else None
            ),
            "conditional_valid_count": len(valid),
        }

    answerable = [
        row for row in rows
        if not is_official_abstention_id(row["question_id"])
    ]
    abstention = [
        row for row in rows if is_official_abstention_id(row["question_id"])
    ]
    by_category: dict[str, dict[str, Any]] = {}
    for qtype in sorted({row["question_type"] for row in rows}):
        category = [row for row in rows if row["question_type"] == qtype]
        by_category[qtype] = {
            "answerable": stats([
                row for row in category
                if not is_official_abstention_id(row["question_id"])
            ]),
            "abstention": stats([
                row for row in category
                if is_official_abstention_id(row["question_id"])
            ]),
        }
    return {
        "answerable": stats(answerable),
        "abstention": stats(abstention),
        "by_category": by_category,
    }


def _recall_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the adapter's post-answer recall diagnostic from rows only."""

    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row["question_type"].replace("_abs", ""), []).append(row)
    tiers: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    result: dict[str, Any] = {}
    for qtype in sorted(by_type):
        category = by_type[qtype]
        infrastructure = [
            row for row in category
            if row.get("judge_error") or row.get("benchmark_failure")
        ]
        causal = [
            row for row in category
            if not (row.get("judge_error") or row.get("benchmark_failure"))
        ]
        known = [row for row in causal if row.get("recall_ceiling") is not None]
        hits = [row for row in known if row.get("recall_ceiling") is True]
        misses = [row for row in causal if not row.get("correct")]
        for row in known:
            tiers[row.get("recall_tier", "none")] += 1
        for row in causal:
            modes[row.get("gold_mode", "none")] += 1
        result[qtype] = {
            "known": len(known),
            "unknown": len(causal) - len(known),
            "ceiling_rate": len(hits) / len(known) if known else None,
            "misses": len(misses),
            "miss_retrieval": sum(
                row.get("recall_ceiling") is False for row in misses
            ),
            "miss_ranking": sum(
                row.get("recall_ceiling") is True for row in misses
            ),
            "miss_unknown": sum(
                row.get("recall_ceiling") is None for row in misses
            ),
            "benchmark_failures_excluded": len(infrastructure),
        }
    result["_tiers"] = dict(tiers)
    result["_gold_mode"] = dict(modes)
    return result


def _router_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the label-free router diagnostic from durable route evidence."""

    labels = ("MR", "TR", "NONE")

    def normalized(value: object) -> str:
        return value if value in {"MR", "TR"} else "NONE"

    confusion = {target: Counter() for target in labels}
    for row in rows:
        confusion[normalized(row.get("oracle_ability"))][
            normalized(row.get("detected_ability"))
        ] += 1
    per_intent: dict[str, Any] = {}
    for intent in ("MR", "TR"):
        true_positive = confusion[intent][intent]
        actual = sum(confusion[intent].values())
        predicted = sum(confusion[target][intent] for target in labels)
        per_intent[intent] = {
            "recall": true_positive / actual if actual else None,
            "precision": true_positive / predicted if predicted else None,
            "actual": actual,
            "predicted": predicted,
            "tp": true_positive,
        }
    none_total = sum(confusion["NONE"].values())
    abstain_ok = confusion["NONE"]["NONE"]
    return {
        "confusion": {target: dict(confusion[target]) for target in labels},
        "per_intent": per_intent,
        "abstain_accuracy": abstain_ok / none_total if none_total else None,
        "false_positives": none_total - abstain_ok,
        "none_total": none_total,
    }


def validate_strict_artifact(
    data: object, *, path: Path | None = None,
    require_scored: bool | None = None,
) -> dict[str, Any]:
    """Validate a strict LME evidence envelope and recompute every score."""

    if not isinstance(data, Mapping):
        raise BenchmarkIntegrityError("LongMemEval artifact root must be an object")
    if data.get("version") != "strict-v1":
        raise BenchmarkIntegrityError("LongMemEval strict version is unsupported")
    if data.get("benchmark") != "LongMemEval":
        raise BenchmarkIntegrityError("LongMemEval benchmark identity differs")
    manifest = data.get("manifest")
    config = data.get("config")
    models = data.get("models")
    execution = data.get("execution")
    rows_raw = data.get("per_question")
    if not all(isinstance(value, Mapping) for value in (manifest, config, models, execution)):
        raise BenchmarkIntegrityError("LongMemEval strict envelope is incomplete")
    if not isinstance(rows_raw, list):
        raise BenchmarkIntegrityError("LongMemEval strict rows are absent")
    result_digest = data.get("result_digest")
    if (
        not isinstance(result_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", result_digest) is None
        or result_digest != content_hash(rows_raw)
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval ordered per-question result digest differs"
        )
    if manifest.get("schema") != STRICT_PROTOCOL_VERSION or manifest.get("benchmark") != "LongMemEval":
        raise BenchmarkIntegrityError("LongMemEval manifest schema/benchmark differs")
    if manifest.get("run_id") != content_hash({
        key: value for key, value in manifest.items() if key != "run_id"
    }):
        raise BenchmarkIntegrityError("LongMemEval manifest run_id is invalid")
    if manifest.get("config") != config or manifest.get("models") != models:
        raise BenchmarkIntegrityError("LongMemEval top-level identity differs from manifest")
    if manifest.get("config_hash") != content_hash(config):
        raise BenchmarkIntegrityError("LongMemEval manifest config hash is invalid")
    if manifest.get("model_hash") != content_hash(models):
        raise BenchmarkIntegrityError("LongMemEval manifest model hash is invalid")
    for field in ("code_hash", "data_hash", "expected_ids_hash"):
        value = manifest.get(field)
        if not isinstance(value, str) or re.fullmatch(
            r"sha256:[0-9a-f]{64}", value
        ) is None:
            raise BenchmarkIntegrityError(f"LongMemEval manifest {field} is malformed")
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BenchmarkIntegrityError("LongMemEval manifest seed is malformed")
    if config.get("seed") != seed:
        raise BenchmarkIntegrityError("LongMemEval config/manifest seed differs")

    scale = config.get("scales", config.get("scale"))
    if scale not in LME_SUPPORTED_SCALES:
        raise BenchmarkIntegrityError("LongMemEval strict scale is unsupported")
    for key in (
        "label_free_answer_path", "scored_run", "exploratory_label_steering",
        "exploratory_non_comparable", "subset_run", "official_denominator_validated",
        "source_order_validated", "indexing_require_healthy",
        "historical_local_judge_prompts_exact_official", "official_judge_match",
        "auto_ability", "no_dream", "embeddings", "retrieval_only",
        "graph_facts_first", "permissive_default", "distill",
        "aggregation_nodes", "aggregation_broad", "episode_granularity",
        "value_supersession", "graph_multihop",
        "official_transport_exact",
    ):
        _bool(config.get(key), label=f"LongMemEval config {key}")
    for key in (
        "label_free_answer_path", "scored_run", "exploratory_label_steering",
        "exploratory_non_comparable",
    ):
        if manifest.get(key) is not config.get(key):
            raise BenchmarkIntegrityError(
                f"LongMemEval manifest/config posture differs for {key}"
            )
    scored_run = config["scored_run"]
    if require_scored is True and scored_run is not True:
        raise BenchmarkIntegrityError("LongMemEval artifact is retrieval-only diagnostic evidence")
    if require_scored is False and scored_run is not False:
        raise BenchmarkIntegrityError("LongMemEval artifact is not retrieval-only diagnostic evidence")
    retrieval_only = config.get("retrieval_only")
    if not isinstance(retrieval_only, bool) or retrieval_only is scored_run:
        raise BenchmarkIntegrityError("LongMemEval scored/retrieval posture is inconsistent")
    if config["label_free_answer_path"] is config["exploratory_label_steering"]:
        raise BenchmarkIntegrityError("LongMemEval routing posture is inconsistent")
    if config["auto_ability"] is not config["label_free_answer_path"]:
        raise BenchmarkIntegrityError("LongMemEval auto-routing disclosure is inconsistent")
    if config["historical_local_judge_prompts_exact_official"] is not (
        LME_HISTORICAL_LOCAL_JUDGE_PROMPTS_EXACT_OFFICIAL
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval historical local judge-prompt identity is false"
        )
    if config.get("evaluator_commit") != LME_EVALUATOR_COMMIT:
        raise BenchmarkIntegrityError("LongMemEval evaluator commit differs")
    if config.get("evaluator_sha256") != LME_EVALUATOR_SHA256:
        raise BenchmarkIntegrityError("LongMemEval evaluator hash differs")
    if config.get("evaluator_url") != LME_EVALUATOR_URL:
        raise BenchmarkIntegrityError("LongMemEval evaluator URL differs")
    dataset_revision = config.get("dataset_revision")
    if (
        not isinstance(dataset_revision, str) or not dataset_revision.strip()
        or dataset_revision != dataset_revision.strip()
    ):
        raise BenchmarkIntegrityError("LongMemEval dataset revision is malformed")
    if config.get("dataset_sha256") != manifest.get("data_hash"):
        raise BenchmarkIntegrityError("LongMemEval config/manifest data hash differs")
    dataset_expected_count = config.get("dataset_expected_count")
    if (
        isinstance(dataset_expected_count, bool)
        or not isinstance(dataset_expected_count, int)
        or dataset_expected_count <= 0
    ):
        raise BenchmarkIntegrityError("LongMemEval source dataset count is malformed")
    source_ids_hash = config.get("source_ids_hash")
    if not isinstance(source_ids_hash, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", source_ids_hash
    ) is None:
        raise BenchmarkIntegrityError("LongMemEval source ID hash is malformed")
    source_qtype_counts = config.get("source_qtype_counts")
    if (
        not isinstance(source_qtype_counts, Mapping)
        or not source_qtype_counts
        or any(key not in LME_BASE_QUESTION_TYPES for key in source_qtype_counts)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in source_qtype_counts.values()
        )
        or sum(source_qtype_counts.values()) != dataset_expected_count
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval source question-type distribution is malformed"
        )
    if config["source_order_validated"] and (
        scale != "S"
        or config.get("dataset_revision") != LME_S_DATASET_REVISION
        or manifest.get("data_hash") != LME_S_DATASET_SHA256
        or config.get("dataset_url") != LME_S_DATASET_URL
        or config.get("source_ids_hash") != LME_S_SOURCE_IDS_HASH
        or config.get("source_qtype_counts") != LME_S_QTYPE_COUNTS
        or config.get("dataset_expected_count") != LME_S_EXPECTED_COUNT
    ):
        raise BenchmarkIntegrityError("LongMemEval pinned source identity is inconsistent")
    if config["official_denominator_validated"] and (
        scale != "S" or dataset_revision != LME_S_DATASET_REVISION
        or manifest.get("data_hash") != LME_S_DATASET_SHA256
        or config.get("dataset_url") != LME_S_DATASET_URL
        or manifest.get("protocol_split") != "full"
        or manifest.get("expected_count") != LME_S_EXPECTED_COUNT
        or config["subset_run"] or config.get("sample") not in (0, None)
        or config.get("sample_strategy") != "all-source-order"
        or config["source_order_validated"] is not True
        or config.get("source_ids_hash") != LME_S_SOURCE_IDS_HASH
        or config.get("source_qtype_counts") != LME_S_QTYPE_COUNTS
    ):
        raise BenchmarkIntegrityError("LongMemEval official denominator claim is inconsistent")
    sample = config.get("sample")
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
        raise BenchmarkIntegrityError("LongMemEval sample is malformed")
    if sample > dataset_expected_count:
        raise BenchmarkIntegrityError("LongMemEval sample exceeds its source dataset")
    if config["subset_run"] != (sample > 0):
        raise BenchmarkIntegrityError("LongMemEval subset posture differs from sample")
    if config["subset_run"] and config.get("sample_strategy") != (
        "sha256-seed-source-index-preserve-order-v1"
    ):
        raise BenchmarkIntegrityError("LongMemEval sample strategy is inconsistent")
    if config["subset_run"] and not config["exploratory_non_comparable"]:
        raise BenchmarkIntegrityError("LongMemEval sampled run claims comparability")
    if not config["subset_run"] and config.get("sample_strategy") != "all-source-order":
        raise BenchmarkIntegrityError("LongMemEval full-run sample strategy is inconsistent")
    if config.get("distill_prompt_version") not in {
        "v1", "v2",
    }:
        raise BenchmarkIntegrityError("LongMemEval distillation prompt identity is unsupported")
    expected_retrieval_owner = (
        "separate-retrieval-meter"
        if config["retrieval_only"] and config["distill"]
        else "reader" if config["distill"] else "none"
    )
    if config.get("retrieval_usage_owner") != expected_retrieval_owner:
        raise BenchmarkIntegrityError(
            "LongMemEval retrieval usage ownership is inconsistent"
        )
    rerank_model = config.get("rerank_model")
    if rerank_model not in {None, "llm", "cross-encoder"}:
        raise BenchmarkIntegrityError("LongMemEval rerank model is unsupported")
    for field in (
        "rerank_message_hits", "rules", "rules_extraction",
        "facts", "facts_extraction",
    ):
        value = config.get(field)
        if value is not None and not isinstance(value, bool):
            raise BenchmarkIntegrityError(f"LongMemEval config {field} is malformed")
    for field in ("top_k", "workers", "indexing_max_cycles"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BenchmarkIntegrityError(f"LongMemEval config {field} is malformed")
    timeout = _finite_number(config.get("indexing_timeout_s"))
    if timeout is None or timeout <= 0:
        raise BenchmarkIntegrityError("LongMemEval indexing timeout is malformed")
    rerank_top_k = config.get("rerank_top_k")
    if rerank_top_k is not None and (
        isinstance(rerank_top_k, bool) or not isinstance(rerank_top_k, int)
        or rerank_top_k <= 0
    ):
        raise BenchmarkIntegrityError("LongMemEval rerank_top_k is malformed")
    max_hops = config.get("graph_multihop_max_hops")
    if max_hops is not None and (
        isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops <= 0
    ):
        raise BenchmarkIntegrityError("LongMemEval graph max hops is malformed")
    decay = config.get("graph_multihop_decay")
    if decay is not None and (
        _finite_number(decay) is None or not 0 < float(decay) <= 1
    ):
        raise BenchmarkIntegrityError("LongMemEval graph decay is malformed")
    min_score = config.get("graph_multihop_min_score")
    if min_score is not None and (
        _finite_number(min_score) is None or not 0 <= float(min_score) <= 1
    ):
        raise BenchmarkIntegrityError("LongMemEval graph min score is malformed")
    if not config["graph_multihop"] and any(
        config.get(field) is not None for field in (
            "graph_multihop_max_hops", "graph_multihop_decay",
            "graph_multihop_min_score",
        )
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval graph parameters are set while graph multihop is disabled"
        )
    if config["aggregation_broad"] and not config["aggregation_nodes"]:
        raise BenchmarkIntegrityError(
            "LongMemEval broad aggregation is set while aggregation is disabled"
        )
    context_policy = config.get("context_policy")
    expected_context_keys = {
        "name", "budget_unit", "max_input_tokens", "max_input_bytes",
        "provider_context_window_tokens", "reserved_output_tokens",
        "reserved_transport_overhead_tokens",
        "tokenizer", "tokenizer_failure_policy", "source_boundaries",
        "raw_evidence_reserve_fraction", "min_semantic_excerpt_alnum",
        "min_semantic_excerpt_chars", "gold_access",
    }
    if not isinstance(context_policy, Mapping) or set(context_policy) != expected_context_keys:
        raise BenchmarkIntegrityError("LongMemEval context packing policy is malformed")
    max_input_bytes = context_policy.get("max_input_bytes")
    provider_ceiling = context_policy.get("provider_context_window_tokens")
    output_reserve = context_policy.get("reserved_output_tokens")
    transport_overhead = context_policy.get("reserved_transport_overhead_tokens")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (provider_ceiling, output_reserve, transport_overhead)
    ):
        raise BenchmarkIntegrityError("LongMemEval byte/context capacity is malformed")
    if (
        config.get("max_input_bytes") != max_input_bytes
        or config.get("provider_context_tokens") != provider_ceiling
        or context_policy.get("source_boundaries") != [
            "head", "query-window", "tail",
        ]
        or context_policy.get("raw_evidence_reserve_fraction") != 0.60
        or context_policy.get("min_semantic_excerpt_alnum") != 8
        or context_policy.get("min_semantic_excerpt_chars") != 12
        or context_policy.get("gold_access") is not False
    ):
        raise BenchmarkIntegrityError("LongMemEval context packing policy is malformed")
    tokenizer = context_policy.get("tokenizer")
    if tokenizer is None:
        if (
            context_policy.get("name") != "conservative-utf8-byte-query-head-tail-v2"
            or context_policy.get("budget_unit") != "utf8_bytes"
            or isinstance(max_input_bytes, bool)
            or not isinstance(max_input_bytes, int) or max_input_bytes <= 0
            or max_input_bytes + output_reserve + transport_overhead > provider_ceiling
            or context_policy.get("max_input_tokens") is not None
            or config.get("max_input_tokens") is not None
            or context_policy.get("tokenizer_failure_policy") != "not-applicable"
        ):
            raise BenchmarkIntegrityError("LongMemEval byte packing identity is malformed")
    else:
        token_budget = context_policy.get("max_input_tokens")
        if (
            not isinstance(tokenizer, Mapping)
            or set(tokenizer) != {
                "configured", "backend", "bound_model", "file_sha256", "local_only",
            }
            or tokenizer.get("configured") is not True
            or tokenizer.get("backend") != "huggingface-tokenizers-json"
            or tokenizer.get("bound_model") != config.get("answer_model")
            or not isinstance(tokenizer.get("file_sha256"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", tokenizer["file_sha256"]) is None
            or tokenizer.get("local_only") is not True
            or context_policy.get("name") != "model-bound-tokenizer-query-head-tail-v2"
            or context_policy.get("budget_unit") != "model_tokens"
            or context_policy.get("tokenizer_failure_policy") != "fail-closed"
            or max_input_bytes is not None
            or isinstance(token_budget, bool) or not isinstance(token_budget, int)
            or token_budget <= 0
            or token_budget + output_reserve + transport_overhead > provider_ceiling
            or config.get("max_input_tokens") != token_budget
        ):
            raise BenchmarkIntegrityError("LongMemEval model tokenizer identity is malformed")
    validate_prereg(
        config.get("prereg"), required=not config["exploratory_non_comparable"]
    )
    requires_exploratory = bool(
        config.get("prereg") is None
        or config["subset_run"]
        or not config["label_free_answer_path"]
        or not config["indexing_require_healthy"]
        or config.get("no_dream") is True
        or not config["source_order_validated"]
        or not scored_run
    )
    if requires_exploratory and not config["exploratory_non_comparable"]:
        raise BenchmarkIntegrityError("LongMemEval non-comparable posture is understated")
    protocol_split = manifest.get("protocol_split")
    calibration_hash = manifest.get("calibration_receipt_hash")
    if protocol_split == "full":
        if calibration_hash is not None:
            raise BenchmarkIntegrityError("LongMemEval full split carries calibration receipt")
    elif protocol_split in {"dev", "holdout"}:
        if not isinstance(calibration_hash, str) or re.fullmatch(
            r"sha256:[0-9a-f]{64}", calibration_hash
        ) is None:
            raise BenchmarkIntegrityError("LongMemEval internal split lacks frozen receipt")
    else:
        raise BenchmarkIntegrityError("LongMemEval protocol split is malformed")
    if config.get("judge_protocol") not in {"legacy-custom", "official"}:
        raise BenchmarkIntegrityError("LongMemEval judge protocol is unsupported")
    legacy_endpoint_fields = {
        "answer_base_url", "judge_base_url", "hymem_base_url",
        "official_judge_base_url",
    }
    if legacy_endpoint_fields & set(config):
        raise BenchmarkIntegrityError(
            "LongMemEval endpoint identity is not secret-free"
        )
    official_endpoint = secret_free_endpoint_identity(
        LME_OFFICIAL_JUDGE_BASE_URL, label="official judge"
    )
    if (
        config.get("official_judge_endpoint_origin")
        != official_endpoint["endpoint_origin"]
        or config.get("official_judge_endpoint_sha256")
        != official_endpoint["endpoint_sha256"]
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval official judge endpoint identity differs"
        )
    if (
        config.get("judge_transport_retry_policy") != LME_LOCAL_RETRY_POLICY
        or config.get("official_transport_retry_policy") != LME_UPSTREAM_RETRY_POLICY
        or config.get("official_transport_exact") is not False
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval judge transport-policy disclosure differs"
        )
    observed_official = official_judge_match(config, models)
    if config["official_judge_match"] is not observed_official:
        raise BenchmarkIntegrityError("LongMemEval official judge identity flag is false")
    if config["judge_protocol"] == "legacy-custom" and observed_official:
        raise BenchmarkIntegrityError("LongMemEval legacy judge claims official identity")
    if config["judge_protocol"] == "official" and not observed_official:
        raise BenchmarkIntegrityError(
            "LongMemEval official protocol lacks the exact pinned judge identity"
        )

    if set(models) != {"reader", "judge", "memory_pipeline", "embedding"}:
        raise BenchmarkIntegrityError("LongMemEval model identity coverage differs")
    reader = _validate_model(
        models.get("reader"), label="reader", secret_free_endpoint=True,
    )
    judge = _validate_model(
        models.get("judge"), label="judge", secret_free_endpoint=True,
    )
    pipeline = _validate_model(
        models.get("memory_pipeline"), label="memory pipeline",
        secret_free_endpoint=True,
    )
    try:
        validate_extraction_canary_config_binding(
            config.get("extraction_canary"),
            config.get("effective_hymem_config"),
        )
    except BenchmarkIntegrityError as exc:
        raise BenchmarkIntegrityError(
            "LongMemEval extraction canary policy is invalid"
        ) from exc
    expected_model_fields = {
        "reader": {
            "provider", "model", "endpoint_origin", "endpoint_sha256",
            "temperature", "max_tokens", "extra_body",
        },
        "judge": {
            "provider", "model", "endpoint_origin", "endpoint_sha256",
            "temperature", "max_tokens", "n", "extra_body", "protocol",
            "evaluator_commit", "evaluator_sha256", "verdict_parser",
            "prompt_exact_official", "retry_policy",
        },
        "memory pipeline": {
            "provider", "model", "endpoint_origin", "endpoint_sha256",
            "thinking_mode",
            "effective_extra_body", "aggregation_producer",
            "deployment_revision_sha256", "deployment_tenant_sha256",
            "transport_package_version", "request_timeout_seconds",
        },
    }
    for label, identity in (
        ("reader", reader), ("judge", judge), ("memory pipeline", pipeline),
    ):
        if set(identity) != expected_model_fields[label]:
            raise BenchmarkIntegrityError(
                f"LongMemEval {label} model identity fields differ"
            )
    _validated_pipeline_aggregation_producer(pipeline)
    def expected_provider(endpoint: str) -> str:
        official = validate_http_endpoint(
            endpoint, label="model identity"
        ).official_provider
        return official if official is not None else "openai-compatible"
    for identity, prefix in ((reader, "answer"), (judge, "judge")):
        if (
            identity.get("model") != config.get(f"{prefix}_model")
            or identity.get("endpoint_origin")
            != config.get(f"{prefix}_endpoint_origin")
            or identity.get("endpoint_sha256")
            != config.get(f"{prefix}_endpoint_sha256")
            or identity.get("provider")
            != expected_provider(identity["endpoint_origin"])
            or _finite_number(identity.get("temperature")) != 0.0
            or identity.get("max_tokens") != (1024 if prefix == "answer" else 10)
            or (
                prefix == "judge"
                and identity.get("n") != (
                    1 if config.get("judge_protocol") == "official" else None
                )
            )
            or identity.get("extra_body") != config.get(f"{prefix}_extra_body_obj")
        ):
            raise BenchmarkIntegrityError(
                f"LongMemEval {prefix} effective identity differs from config"
            )
    if (
        pipeline.get("provider") != expected_provider(pipeline["endpoint_origin"])
        or pipeline.get("model") != config.get("hymem_model")
        or pipeline.get("endpoint_origin") != config.get("hymem_endpoint_origin")
        or pipeline.get("endpoint_sha256") != config.get("hymem_endpoint_sha256")
        or pipeline.get("thinking_mode") != config.get("hymem_thinking")
    ):
        raise BenchmarkIntegrityError("LongMemEval memory pipeline identity differs from config")
    if reader.get("extra_body") != normalize_extra_body(reader.get("extra_body"), label="reader"):
        raise BenchmarkIntegrityError("LongMemEval reader extra_body is not normalized")
    if judge.get("extra_body") != normalize_extra_body(judge.get("extra_body"), label="judge"):
        raise BenchmarkIntegrityError("LongMemEval judge extra_body is not normalized")
    expected_prompt_exact = config["judge_protocol"] == "official"
    if judge.get("prompt_exact_official") is not expected_prompt_exact:
        raise BenchmarkIntegrityError(
            "LongMemEval selected judge prompt identity differs from its protocol"
        )
    if judge.get("retry_policy") != config.get("judge_transport_retry_policy"):
        raise BenchmarkIntegrityError(
            "LongMemEval judge retry identity differs from config"
        )
    mode = pipeline.get("thinking_mode")
    if mode not in {"auto", "disabled", "off", "enabled"}:
        raise BenchmarkIntegrityError("LongMemEval memory pipeline thinking mode is malformed")
    host = (urlsplit(pipeline["endpoint_origin"]).hostname or "").casefold()
    sends_thinking = mode == "disabled" or (
        mode == "auto" and ("deepseek" in host or "deepseek" in pipeline["model"].casefold())
    )
    expected_pipeline_extra = {"thinking": {"type": "disabled"}} if sends_thinking else {}
    if pipeline.get("effective_extra_body") != expected_pipeline_extra:
        raise BenchmarkIntegrityError("LongMemEval memory pipeline request identity differs")
    embedding = _validate_embedding_identity(models.get("embedding"))
    if config.get("embedding_runtime") != embedding:
        raise BenchmarkIntegrityError("LongMemEval embedding config/model identity differs")
    if config.get("embeddings") is not embedding.get("configured"):
        raise BenchmarkIntegrityError("LongMemEval embedding enablement differs")
    effective = config.get("effective_hymem_config")
    if not isinstance(effective, Mapping) or not effective:
        raise BenchmarkIntegrityError("LongMemEval effective HyMem config is absent")
    expected_effective = {
        "message_fts_top_k": 15, "fts_top_k": 10, "graph_top_k": 10,
        "aggregation_nodes_enabled": config.get("aggregation_nodes"),
        "aggregation_inject_abilities": (
            [] if config.get("aggregation_broad") else ["TR"]
        ),
        "episode_granularity_enabled": config.get("episode_granularity"),
        "value_supersession_enabled": config.get("value_supersession"),
        "graph_multihop_enabled": config.get("graph_multihop"),
        "rerank_top_k": (
            config.get("rerank_top_k")
            if config.get("rerank_top_k") is not None else 20
        ),
        "rerank_model": config.get("rerank_model") or "llm",
        "rerank_message_hits": (
            config.get("rerank_message_hits")
            if config.get("rerank_message_hits") is not None else True
        ),
        "graph_multihop_max_hops": (
            config.get("graph_multihop_max_hops")
            if config.get("graph_multihop_max_hops") is not None else 2
        ),
        "graph_multihop_decay": (
            config.get("graph_multihop_decay")
            if config.get("graph_multihop_decay") is not None else 0.5
        ),
        "graph_multihop_min_score": (
            config.get("graph_multihop_min_score")
            if config.get("graph_multihop_min_score") is not None else 0.05
        ),
        "rules_enabled": (
            config.get("rules") if config.get("rules") is not None else True
        ),
        "rules_extraction_enabled": (
            config.get("rules_extraction")
            if config.get("rules_extraction") is not None else False
        ),
        "facts_enabled": (
            config.get("facts") if config.get("facts") is not None else True
        ),
        "facts_extraction_enabled": (
            config.get("facts_extraction")
            if config.get("facts_extraction") is not None else True
        ),
    }
    for field, expected in expected_effective.items():
        if effective.get(field) != expected:
            raise BenchmarkIntegrityError(
                f"LongMemEval effective HyMem lever {field} differs"
            )
    expected_count = manifest.get("expected_count")
    if (
        isinstance(expected_count, bool) or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise BenchmarkIntegrityError("LongMemEval manifest expected_count is malformed")
    if expected_count > dataset_expected_count:
        raise BenchmarkIntegrityError(
            "LongMemEval run denominator exceeds its source dataset"
        )
    if protocol_split == "full" and expected_count != (
        sample if sample > 0 else dataset_expected_count
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval full-run denominator differs from its sample declaration"
        )
    if config["official_denominator_validated"] and (
        manifest.get("expected_ids_hash") != LME_S_SOURCE_IDS_HASH
    ):
        raise BenchmarkIntegrityError("LongMemEval official source ID order differs")

    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    completed = failed = missing = 0
    min_reader_calls = min_judge_calls = 0
    row_distill_calls = 0
    successful_indexed_ids: set[str] = set()
    row_indexing_summaries: dict[str, dict[str, Any]] = {}
    failed_row_indexing_summaries: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows_raw):
        if not isinstance(raw, Mapping):
            raise BenchmarkIntegrityError(f"LongMemEval strict row {index} is malformed")
        row = dict(raw)
        qid = row.get("question_id")
        qtype = row.get("question_type")
        if not isinstance(qid, str) or not qid.strip() or qid != qid.strip():
            raise BenchmarkIntegrityError("LongMemEval strict row id is malformed")
        if qtype not in LME_BASE_QUESTION_TYPES:
            raise BenchmarkIntegrityError("LongMemEval strict row question type is malformed")
        ids.append(qid)
        failure = row.get("benchmark_failure")
        if failure is not None and (
            not isinstance(failure, str) or not failure.strip()
        ):
            raise BenchmarkIntegrityError("LongMemEval failure evidence is malformed")
        verdict = row.get("correct")
        if scored_run:
            if row.get("retrieval_only") is True or not isinstance(verdict, bool):
                raise BenchmarkIntegrityError("LongMemEval scored row verdict posture is malformed")
        elif row.get("retrieval_only") is not True or (
            verdict is not None and not (failure and verdict is False)
        ):
            raise BenchmarkIntegrityError("LongMemEval retrieval row verdict posture is malformed")

        row_indexing = row.get("indexing")
        row_indexing_complete: bool | None = None
        if config.get("no_dream") is True:
            if row_indexing is not None:
                raise BenchmarkIntegrityError(
                    "LongMemEval no-dream row carries indexing evidence"
                )
        elif row_indexing is not None:
            row_indexing_complete = _validate_indexing(
                row_indexing,
                require_healthy=config["indexing_require_healthy"],
                allow_incomplete=True,
            )
            _validate_aggregation_pipeline_binding(
                row_indexing.get("final_status"), pipeline,
            )
            normalized_row_indexing = dict(row_indexing)
            if normalized_row_indexing.get("schema") != LME_INDEXING_SUMMARY_VERSION:
                normalized_row_indexing.pop("question_id", None)
            if row_indexing_complete:
                successful_indexed_ids.add(qid)
                row_indexing_summaries[qid] = normalized_row_indexing
            else:
                failed_row_indexing_summaries[qid] = normalized_row_indexing

        oracle = row.get("oracle_ability")
        detected = row.get("detected_ability")
        ability = row.get("ability_used")
        missing_row = failure == "missing_prediction"
        distill_fired = row.get("distill_fired")
        distill_calls = row.get("distill_calls")
        if missing_row and distill_fired is None and distill_calls is None:
            distill_fired, distill_calls = False, 0
        if (
            not isinstance(distill_fired, bool)
            or isinstance(distill_calls, bool)
            or not isinstance(distill_calls, int)
            or distill_calls < 0
            or (distill_calls > 0 and not distill_fired)
            or (
                not config["distill"]
                and (distill_fired or distill_calls != 0)
            )
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval row distillation usage is malformed"
            )
        row_distill_calls += distill_calls
        if not missing_row:
            if oracle != LME_ABILITY_BY_TYPE[qtype] or detected not in {None, "MR", "TR"}:
                raise BenchmarkIntegrityError("LongMemEval row routing evidence is malformed")
            expected_ability = detected if config["label_free_answer_path"] else oracle
            if ability != expected_ability:
                raise BenchmarkIntegrityError("LongMemEval row routing evidence is inconsistent")

        if failure:
            if scored_run and verdict is not False:
                raise BenchmarkIntegrityError("LongMemEval failed scored row is inconsistent")
            failed += 1
            if missing_row:
                missing += 1
            indexing_failure = re.fullmatch(
                r"indexing_failure:([a-z][a-z0-9_]*)", failure
            )
            if indexing_failure is not None:
                code = indexing_failure.group(1)
                if (
                    row_indexing_complete is not False
                    or not isinstance(row_indexing, Mapping)
                    or row_indexing.get("schema") != LME_INDEXING_SUMMARY_VERSION
                    or row_indexing.get("outcome") != "failure"
                    or not isinstance(row_indexing.get("failure"), Mapping)
                    or row_indexing["failure"].get("code") != code
                ):
                    raise BenchmarkIntegrityError(
                        "LongMemEval indexing failure row/summary differs"
                    )
                forbidden_scoring = {
                    "question", "answer", "hypothesis", "context_sha",
                    "judge_raw", "judge_protocol", "judge_error",
                    "judge_parse_valid", "recall_ceiling", "recall_tier",
                    "gold_mode", "gold_turns", "gold_turns_in_pool",
                    "gold_turn_tiers", "n_episodes", "n_agg_nodes",
                    "n_procedures", "n_facts", "gold_in_episodes",
                    "gold_in_facts",
                }
                if forbidden_scoring & set(row):
                    raise BenchmarkIntegrityError(
                        "LongMemEval indexing failure carries scoring/retrieval evidence"
                    )
                allowed_failure_fields = {
                    "question_id", "question_type", "correct",
                    "benchmark_failure", "retrieval_only", "oracle_ability",
                    "detected_ability", "ability_used", "distill_fired",
                    "distill_calls", "indexing", "memory_pipeline_usage",
                    "embedding_usage", "lifecycle_errors",
                }
                if not set(row) <= allowed_failure_fields:
                    raise BenchmarkIntegrityError(
                        "LongMemEval indexing failure row fields differ"
                    )
            elif row_indexing_complete is False:
                raise BenchmarkIntegrityError(
                    "LongMemEval failed indexing summary lacks its failure code"
                )
            if failure == "reader_transport_or_empty_response":
                hypothesis = row.get("hypothesis")
                if (
                    not isinstance(hypothesis, str)
                    or (hypothesis.strip() and not hypothesis.startswith("[LLM_ERROR"))
                    or row.get("judge_raw") != ""
                    or row.get("judge_error") is not False
                    or row.get("judge_parse_valid") is not None
                    or row.get("judge_protocol") != config["judge_protocol"]
                ):
                    raise BenchmarkIntegrityError(
                        "LongMemEval reader failure reached or fabricated judge evidence"
                    )
            if failure.startswith("judge_"):
                if row.get("judge_error") is not True:
                    raise BenchmarkIntegrityError(
                        "LongMemEval judge failure is not marked in row evidence"
                    )
                raw_judge = row.get("judge_raw")
                if (
                    not isinstance(raw_judge, str)
                    or row.get("judge_protocol") != config["judge_protocol"]
                    or row.get("judge_parse_valid") is not False
                ):
                    raise BenchmarkIntegrityError(
                        "LongMemEval judge failure lacks raw/parser evidence"
                    )
                if config["judge_protocol"] == "official":
                    if raw_judge.strip() and not raw_judge.startswith("[LLM_ERROR"):
                        raise BenchmarkIntegrityError(
                            "LongMemEval official judge failure carries an actual response"
                        )
                else:
                    _legacy_verdict, legacy_valid = parse_legacy_verdict(raw_judge)
                    if legacy_valid:
                        raise BenchmarkIntegrityError(
                            "LongMemEval legacy judge failure carries a parseable response"
                        )
                min_reader_calls += 1
            rows.append(row)
            continue

        completed += 1
        if not isinstance(row.get("context_sha"), str) or re.fullmatch(
            r"[0-9a-f]{64}", row["context_sha"]
        ) is None:
            raise BenchmarkIntegrityError("LongMemEval successful row context hash is malformed")
        if not scored_run:
            if config.get("no_dream") is False:
                if row_indexing_complete is not True:
                    raise BenchmarkIntegrityError(
                        "LongMemEval successful row lacks healthy indexing"
                    )
            rows.append(row)
            continue

        for field in ("question", "answer", "hypothesis", "judge_raw"):
            if not isinstance(row.get(field), str):
                raise BenchmarkIntegrityError(f"LongMemEval successful row lacks {field}")
        if (
            not row["question"].strip() or not row["answer"].strip()
            or not row["hypothesis"].strip() or not row["judge_raw"].strip()
            or row["hypothesis"].startswith("[LLM_ERROR")
            or row["judge_raw"].startswith("[LLM_ERROR")
        ):
            raise BenchmarkIntegrityError("LongMemEval successful row payload is blank/error")
        if row.get("judge_protocol") != config["judge_protocol"]:
            raise BenchmarkIntegrityError("LongMemEval row judge protocol differs")
        if row.get("judge_error") not in {False, None}:
            raise BenchmarkIntegrityError("LongMemEval successful row claims judge error")
        if config["judge_protocol"] == "official":
            expected_verdict = parse_official_verdict(row["judge_raw"])
            if row.get("judge_parse_valid") is not True:
                raise BenchmarkIntegrityError("LongMemEval official row has invalid judge transport")
        else:
            expected_verdict, parse_valid = parse_legacy_verdict(row["judge_raw"])
            if row.get("judge_parse_valid") is not parse_valid or expected_verdict is None:
                raise BenchmarkIntegrityError("LongMemEval legacy judge evidence is unparseable")
        if verdict is not expected_verdict:
            raise BenchmarkIntegrityError("LongMemEval verdict differs from raw judge evidence")
        if config.get("no_dream") is False:
            if row_indexing_complete is not True:
                raise BenchmarkIntegrityError(
                    "LongMemEval successful row lacks healthy indexing"
                )
        min_reader_calls += 1
        min_judge_calls += 1
        rows.append(row)

    ordered_ids = validate_ids(ids, label="LongMemEval strict row")
    ordered_id_set = set(ordered_ids)
    if (
        len(rows) != expected_count
        or manifest.get("expected_ids_hash") != content_hash(list(ordered_ids))
    ):
        raise BenchmarkIntegrityError("LongMemEval strict row denominator/order differs")
    observed_qtype_counts = dict(Counter(row["question_type"] for row in rows))
    if config["official_denominator_validated"] and observed_qtype_counts != LME_S_QTYPE_COUNTS:
        raise BenchmarkIntegrityError("LongMemEval official qtype distribution differs")

    counts = execution.get("counts")
    segments = execution.get("segments")
    if not isinstance(counts, Mapping) or not isinstance(segments, list) or not segments:
        raise BenchmarkIntegrityError("LongMemEval execution evidence is absent")
    required_counts = (
        "expected", "attempted", "unique_attempted", "total_attempts",
        "completed", "failed", "missing",
    )
    normalized_counts: dict[str, int] = {}
    for key in required_counts:
        value = _finite_number(counts.get(key), integer=True)
        if value is None:
            raise BenchmarkIntegrityError(f"LongMemEval execution count {key} is malformed")
        normalized_counts[key] = value
    if (
        normalized_counts["expected"] != expected_count
        or normalized_counts["completed"] != completed
        or normalized_counts["failed"] != failed
        or normalized_counts["missing"] != missing
        or normalized_counts["attempted"] != expected_count - missing
        or normalized_counts["unique_attempted"] != normalized_counts["attempted"]
        or normalized_counts["completed"] + normalized_counts["failed"] != expected_count
        or normalized_counts["total_attempts"] < normalized_counts["attempted"]
    ):
        raise BenchmarkIntegrityError("LongMemEval execution counts do not reconcile")

    call_totals = {
        "reader": 0, "judge": 0, "retrieval": 0, "memory pipeline": 0,
    }
    calls_available = {key: True for key in call_totals}
    retrieval_attempts = 0
    retrieval_attempts_available = True
    total_tokens = 0
    all_tokens_available = True
    elapsed_s = 0.0
    elapsed_available = True
    segment_attempts = 0
    segment_ids: set[str] = set()
    any_running = False
    indexed_ids: set[str] = set()
    indexed_summaries: dict[str, list[dict[str, Any]]] = {}
    indexing_history: dict[str, list[tuple[bool, dict[str, Any]]]] = {}
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise BenchmarkIntegrityError("LongMemEval execution segment is malformed")
        status = segment.get("status")
        if status not in {"running", "complete"}:
            raise BenchmarkIntegrityError("LongMemEval execution segment status is invalid")
        any_running = any_running or status == "running"
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id.strip() or segment_id in segment_ids:
            raise BenchmarkIntegrityError("LongMemEval execution segment id is malformed/duplicate")
        segment_ids.add(segment_id)
        attempted = _finite_number(segment.get("attempted_attempts"), integer=True)
        elapsed = _finite_number(segment.get("elapsed_s"))
        if attempted is None or (status == "complete" and elapsed is None):
            raise BenchmarkIntegrityError("LongMemEval execution segment counters are malformed")
        segment_attempts += attempted
        _validate_segment_extraction_canary(
            segment, pipeline=pipeline, no_dream=config["no_dream"],
            prompt_version=config["effective_hymem_config"]["prompt_version"],
        )
        if segment.get("model_identities") != models:
            raise BenchmarkIntegrityError(
                "LongMemEval execution segment model identity drifted"
            )
        instrumentation_errors = segment.get("instrumentation_errors", [])
        if not isinstance(instrumentation_errors, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in instrumentation_errors
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval execution instrumentation errors are malformed"
            )
        if elapsed is None:
            elapsed_available = False
        else:
            elapsed_s += elapsed
        measured_segment_usage: dict[str, dict[str, Any]] = {}
        for key, label in (
            ("reader_usage", "reader"), ("judge_usage", "judge"),
            ("retrieval_usage", "retrieval"),
            ("memory_pipeline_usage", "memory pipeline"),
        ):
            measured = _usage(segment.get(key), label=label)
            measured_segment_usage[label] = measured
            if not scored_run and label in {"reader", "judge"} and any(
                measured[field] != 0 for field in (
                    "calls", "attempts", "successes",
                )
            ):
                raise BenchmarkIntegrityError(
                    "LongMemEval retrieval-only execution reached a reader/judge client"
                )
            if label == "retrieval":
                owner = config["retrieval_usage_owner"]
                availability_exact = all(
                    measured[field] is not None
                    for field in ("calls", "attempts", "successes")
                )
                if status == "complete" and not instrumentation_errors and not availability_exact:
                    raise BenchmarkIntegrityError(
                        "LongMemEval complete retrieval usage is unavailable"
                    )
                # In a scored distillation run, extraction uses the reader
                # client and is already inside reader_usage. In no-distill
                # runs retrieval has no LLM client. Both postures therefore
                # require an exact zero retrieval meter, preventing double count.
                if owner != "separate-retrieval-meter" and availability_exact:
                    if any(
                        measured[field] != 0
                        for field in ("calls", "attempts", "successes")
                    ) or (
                        measured["token_usage_available"]
                        and measured["total_tokens"] != 0
                    ):
                        raise BenchmarkIntegrityError(
                            "LongMemEval retrieval usage is double-counted"
                        )
                if measured["attempts"] is None:
                    retrieval_attempts_available = False
                else:
                    retrieval_attempts += int(measured["attempts"])
            calls = measured["calls"]
            if calls is None:
                calls_available[label] = False
            else:
                call_totals[label] += int(calls)
            if measured["total_tokens"] is None:
                all_tokens_available = False
            else:
                total_tokens += int(measured["total_tokens"])
        embedding_measured = _embedding_usage(
            segment.get("embedding_usage"), embedding,
            # A provider/client construction or meter failure must not erase
            # durable benchmark rows.  Complete segments may retain an
            # explicitly unavailable identity only alongside durable
            # instrumentation-error evidence; silent identity drift still
            # fails closed.
            allow_unavailable=(status == "running" or bool(instrumentation_errors)),
        )
        if embedding.get("configured"):
            provider_tokens = embedding_measured["provider_tokens"]
            if embedding_measured["provider_tokens_available"] is not True:
                all_tokens_available = False
            elif provider_tokens is not None:
                total_tokens += int(provider_tokens)
        indexing_runs = segment.get("indexing_runs")
        if not isinstance(indexing_runs, list):
            raise BenchmarkIntegrityError("LongMemEval segment indexing_runs is absent")
        latest_indexing = segment.get("latest_indexing")
        has_versioned_indexing = any(
            isinstance(item, Mapping)
            and isinstance(item.get("summary"), Mapping)
            and item["summary"].get("schema") == LME_INDEXING_SUMMARY_VERSION
            for item in indexing_runs
        )
        if has_versioned_indexing and (
            not indexing_runs or latest_indexing != indexing_runs[-1]
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval versioned indexing latest summary is absent/drifted"
            )
        if latest_indexing is not None and (
            not indexing_runs or latest_indexing != indexing_runs[-1]
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval segment latest indexing summary drifted"
            )
        if config.get("no_dream") is False:
            segment_indexing_outcomes: list[bool] = []
            segment_indexing_ids: set[str] = set()
            for recorded in indexing_runs:
                if (
                    isinstance(recorded, Mapping)
                    and "summary" in recorded
                ):
                    if set(recorded) != {"question_id", "summary"}:
                        raise BenchmarkIntegrityError(
                            "LongMemEval indexing run wrapper fields differ"
                        )
                    summary_qid = recorded.get("question_id")
                    summary = recorded.get("summary")
                else:
                    # Deliberate reader compatibility for pre-v2 flat records.
                    summary_qid = recorded.get("question_id") if isinstance(
                        recorded, Mapping
                    ) else None
                    summary = {
                        key: value for key, value in dict(recorded).items()
                        if key != "question_id"
                    } if isinstance(recorded, Mapping) else recorded
                summary_complete = _validate_indexing(
                    summary,
                    require_healthy=config["indexing_require_healthy"],
                    allow_incomplete=True,
                )
                if summary_qid not in ordered_id_set:
                    raise BenchmarkIntegrityError(
                        "LongMemEval indexing summary has unknown/missing question id"
                    )
                normalized_summary = dict(summary)
                segment_indexing_outcomes.append(summary_complete)
                segment_indexing_ids.add(summary_qid)
                indexing_history.setdefault(summary_qid, []).append((
                    summary_complete, normalized_summary,
                ))
                if summary_complete:
                    indexed_ids.add(summary_qid)
                    indexed_summaries.setdefault(summary_qid, []).append(
                        normalized_summary
                    )
            if (
                segment_indexing_outcomes
                and not any(segment_indexing_outcomes)
                and len(segment_indexing_ids) == attempted
            ):
                for label in ("reader", "judge"):
                    measured = measured_segment_usage[label]
                    if any(
                        measured[field] != 0
                        for field in ("calls", "attempts", "successes")
                    ) or (
                        measured["token_usage_available"]
                        and measured["total_tokens"] != 0
                    ):
                        raise BenchmarkIntegrityError(
                            "LongMemEval fail-before-score indexing segment "
                            "reached a reader/judge client"
                        )
        elif indexing_runs or latest_indexing is not None:
            raise BenchmarkIntegrityError(
                "LongMemEval no-dream execution carries indexing evidence"
            )
    if segment_attempts != normalized_counts["total_attempts"]:
        raise BenchmarkIntegrityError("LongMemEval segment attempts differ from run attempts")
    if (
        config["retrieval_usage_owner"] == "separate-retrieval-meter"
        and retrieval_attempts_available
        and retrieval_attempts < row_distill_calls
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval retrieval attempts are below durable distillation calls"
        )
    # A crash leaves the previous segment marked running. Its last persisted
    # counters are useful lower bounds, but not exact process totals; preserve
    # recovery while nulling aggregate usage rather than pretending precision.
    exact_usage = not any_running and all(calls_available.values())
    if exact_usage and scored_run and (
        call_totals["reader"] < min_reader_calls
        or call_totals["judge"] < min_judge_calls
    ):
        raise BenchmarkIntegrityError("LongMemEval metered calls are below durable post-call rows")
    retrieval_cost = data.get("retrieval_cost")
    if retrieval_cost is not None:
        expected_retrieval_cost = {
            "usage_owner": config["retrieval_usage_owner"],
            "llm_calls": call_totals["retrieval"] if exact_usage else None,
            "answer_calls": 0 if not scored_run else call_totals["reader"],
            "judge_calls": 0 if not scored_run else call_totals["judge"],
            "distill_calls": row_distill_calls,
        }
        if not isinstance(retrieval_cost, Mapping) or dict(retrieval_cost) != expected_retrieval_cost:
            raise BenchmarkIntegrityError(
                "LongMemEval retrieval cost summary differs from durable usage"
            )
    if config.get("no_dream") is False and not successful_indexed_ids <= indexed_ids:
        raise BenchmarkIntegrityError("LongMemEval indexing summaries are below successful rows")
    if config.get("no_dream") is False and any(
        row_indexing_summaries[qid] not in indexed_summaries.get(qid, [])
        for qid in successful_indexed_ids
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval row/segment indexing summaries disagree"
        )
    if config.get("no_dream") is False:
        for qid, failed_summary in failed_row_indexing_summaries.items():
            history = indexing_history.get(qid, [])
            if not history or history[-1] != (False, failed_summary):
                raise BenchmarkIntegrityError(
                    "LongMemEval current indexing failure differs from segment history"
                )
        for qid, history in indexing_history.items():
            for index, (complete, _summary) in enumerate(history):
                if complete:
                    continue
                superseded = any(later_complete for later_complete, _ in history[index + 1:])
                if not superseded and qid not in failed_row_indexing_summaries:
                    raise BenchmarkIntegrityError(
                        "LongMemEval indexing failure is neither current nor resumed"
                    )

    diagnostic_errors_raw = data.get("diagnostic_errors", {})
    if (
        not isinstance(diagnostic_errors_raw, Mapping)
        or any(key not in {"scores", "abstention", "recall", "router"}
               for key in diagnostic_errors_raw)
        or any(not isinstance(value, str) or not value.strip()
               for value in diagnostic_errors_raw.values())
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval diagnostic-error disclosure is malformed"
        )
    diagnostic_errors = dict(diagnostic_errors_raw)

    recomputed = _scores_from_rows(rows) if scored_run else {}
    stored = data.get("scores")
    if "scores" in diagnostic_errors:
        fallback = (
            {"OVERALL": {
                "accuracy": round(float(recomputed["OVERALL"]["accuracy"]), 1),
                "count": recomputed["OVERALL"]["count"],
            }} if scored_run else {}
        )
        if stored != fallback:
            raise BenchmarkIntegrityError(
                "LongMemEval failed score diagnostic differs from its safe fallback"
            )
    else:
        if not isinstance(stored, Mapping) or set(stored) != set(recomputed):
            raise BenchmarkIntegrityError("LongMemEval stored score coverage differs")
        for category, expected in recomputed.items():
            observed = stored.get(category)
            if not isinstance(observed, Mapping) or observed.get("count") != expected["count"]:
                raise BenchmarkIntegrityError(f"LongMemEval stored count differs for {category}")
            accuracy = _finite_number(observed.get("accuracy"))
            if accuracy is None or not math.isclose(
                accuracy, round(float(expected["accuracy"]), 1), rel_tol=0.0, abs_tol=1e-12
            ):
                raise BenchmarkIntegrityError(f"LongMemEval stored score differs for {category}")
    recomputed_abstention = _abstention_from_rows(rows) if scored_run else {}
    stored_abstention = data.get("abstention_diagnostics")
    if "abstention_diagnostics" in data:
        expected_abstention = (
            {} if "abstention" in diagnostic_errors else recomputed_abstention
        )
        if stored_abstention != expected_abstention:
            raise BenchmarkIntegrityError("LongMemEval stored abstention summary differs")
    recomputed_recall = _recall_from_rows(rows) if scored_run else {}
    if "recall_diagnostics" in data:
        expected_recall = {} if "recall" in diagnostic_errors else recomputed_recall
        if data.get("recall_diagnostics") != expected_recall:
            raise BenchmarkIntegrityError("LongMemEval stored recall summary differs")
    recomputed_router = _router_from_rows(rows)
    if "router_diagnostics" in data:
        expected_router = {} if "router" in diagnostic_errors else recomputed_router
        if data.get("router_diagnostics") != expected_router:
            raise BenchmarkIntegrityError("LongMemEval stored router summary differs")
    valid_judged = [row for row in rows if not row.get("benchmark_failure")]
    expected_conditional = {
        "accuracy": (
            sum(bool(row.get("correct")) for row in valid_judged) / len(valid_judged)
            if scored_run and valid_judged else None
        ),
        "count": len(valid_judged) if scored_run else 0,
    }
    if (
        "conditional_judged_only" in data
        and data.get("conditional_judged_only") != expected_conditional
    ):
        raise BenchmarkIntegrityError("LongMemEval conditional score summary differs")

    protocol_split = manifest.get("protocol_split")
    if protocol_split not in {"full", "dev", "holdout"}:
        raise BenchmarkIntegrityError("LongMemEval protocol split is malformed")
    # This repository's architecture and prompts were developed against the S
    # benchmark.  A later local receipt is useful experimental discipline, but
    # cannot retroactively turn any S-derived campaign artifact into clean test
    # evidence (including an internal "holdout").
    expected_development = True
    if manifest.get("development_only") is not True:
        raise BenchmarkIntegrityError("LongMemEval manifest development posture is inconsistent")
    if manifest.get("official_split") is not False or manifest.get("official_comparable") is not False:
        raise BenchmarkIntegrityError("LongMemEval official split/comparability posture is invalid")
    expected_limitation = (
        "internal deterministic split, not an official benchmark split"
        if protocol_split in {"dev", "holdout"}
        else "full-set development evidence; may be test-contaminated"
    )
    if manifest.get("protocol_limitation") != expected_limitation:
        raise BenchmarkIntegrityError("LongMemEval protocol limitation differs")
    official_scoring_semantics_aligned = bool(
        config["official_denominator_validated"]
        and config["label_free_answer_path"]
        and config["official_judge_match"]
        and scored_run
        and protocol_split == "full"
        and not config["subset_run"]
        and not config["exploratory_label_steering"]
        and segments[-1].get("status") == "complete"
    )
    return {
        "rows": rows,
        "scores": recomputed,
        "counts": normalized_counts,
        "answer_calls": call_totals["reader"] if exact_usage else None,
        "judge_calls": call_totals["judge"] if exact_usage else None,
        "retrieval_calls": call_totals["retrieval"] if exact_usage else None,
        "pipeline_calls": call_totals["memory pipeline"] if exact_usage else None,
        "total_tokens": (
            total_tokens if exact_usage and all_tokens_available else None
        ),
        "elapsed_s": elapsed_s if elapsed_available and not any_running else None,
        "run_id": manifest["run_id"],
        "judge_protocol": config["judge_protocol"],
        "official_comparable": False,
        "official_scoring_semantics_aligned": (
            official_scoring_semantics_aligned
        ),
        # The successful-response scorer matches upstream, but our bounded
        # three-attempt transport intentionally differs from its unbounded
        # backoff. Do not broaden this into a full-protocol equivalence claim.
        "official_protocol_aligned": bool(
            official_scoring_semantics_aligned
            and config["official_transport_exact"]
        ),
        "development_only": expected_development,
        "official_denominator_validated": config["official_denominator_validated"],
        "abstention_count": sum(
            1 for row in rows if is_official_abstention_id(row["question_id"])
        ),
        "abstention_accuracy": (
            None if not scored_run or not recomputed_abstention["abstention"]["count"]
            else recomputed_abstention["abstention"]["accuracy"] * 100.0
        ),
        "answerable_accuracy": (
            None if not scored_run or not recomputed_abstention["answerable"]["count"]
            else recomputed_abstention["answerable"]["accuracy"] * 100.0
        ),
    }


def export_official_predictions(
    artifact: Mapping[str, Any] | str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Write the exact upstream JSONL hypothesis schema, entirely offline."""

    if isinstance(artifact, (str, Path)):
        source = Path(artifact)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkIntegrityError(f"cannot read LongMemEval artifact {source}: {exc}") from exc
    else:
        data = dict(artifact)
    validated = validate_strict_artifact(data, require_scored=True)
    config = data["config"]
    manifest = data["manifest"]
    segments = data["execution"].get("segments")
    if (
        config.get("scored_run") is not True
        or config.get("retrieval_only") is True
        or not isinstance(segments, list)
        or not segments
        or not isinstance(segments[-1], Mapping)
        or segments[-1].get("status") != "complete"
        or not validated["official_denominator_validated"]
        or manifest.get("expected_count") != LME_S_EXPECTED_COUNT
        or len(validated["rows"]) != LME_S_EXPECTED_COUNT
        or config.get("dataset_revision") != LME_S_DATASET_REVISION
        or manifest.get("data_hash") != LME_S_DATASET_SHA256
        or config.get("source_order_validated") is not True
        or config.get("source_ids_hash") != LME_S_SOURCE_IDS_HASH
        or manifest.get("expected_ids_hash") != LME_S_SOURCE_IDS_HASH
        or config.get("source_qtype_counts") != LME_S_QTYPE_COUNTS
        or config.get("label_free_answer_path") is not True
        or config.get("indexing_require_healthy") is not True
        or config.get("no_dream") is not False
    ):
        raise BenchmarkIntegrityError(
            "official export requires a completed strict full-S source-order run"
        )
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in validated["rows"]:
        hypothesis = row.get("hypothesis")
        if row.get("benchmark_failure") or not isinstance(hypothesis, str):
            hypothesis = ""
        lines.append(json.dumps(
            {"question_id": row["question_id"], "hypothesis": hypothesis},
            ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, dest)
        except FileExistsError as exc:
            raise BenchmarkIntegrityError(
                f"official export already exists: {dest}"
            ) from exc
        try:
            directory_fd = os.open(dest.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return {
        "path": str(dest), "count": len(lines), "run_id": validated["run_id"],
        "schema": ["question_id", "hypothesis"],
        "evaluator_commit": LME_EVALUATOR_COMMIT,
        "evaluator_sha256": LME_EVALUATOR_SHA256,
        "judge_model": LME_OFFICIAL_JUDGE_MODEL,
        "verdict_parser": LME_OFFICIAL_VERDICT_PARSER,
    }
