"""Pinned LongMemEval protocol identities and fail-closed artifact validation.

This module is intentionally provider-free.  Registry ingestion and official
prediction export must be able to validate a completed run without importing an
SDK, reading a credential, or constructing a client.
"""

from __future__ import annotations

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

try:
    from .strictness import (
        BenchmarkIntegrityError,
        STRICT_PROTOCOL_VERSION,
        content_hash,
        validate_ids,
        write_immutable_artifact,
    )
except (ImportError, ValueError):  # direct benchmark-script import
    from strictness import (  # type: ignore
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
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BenchmarkIntegrityError(f"{label} endpoint is malformed")
    if "\\" in value or any(character.isspace() or ord(character) < 32 for character in value):
        raise BenchmarkIntegrityError(f"{label} endpoint is unsafe or ambiguous")
    try:
        parsed = urlsplit(value)
        # Accessing ``port`` is itself validation: urllib deliberately defers
        # rejecting an out-of-range/non-numeric port until this property read.
        parsed.port
    except (TypeError, ValueError) as exc:
        raise BenchmarkIntegrityError(
            f"{label} endpoint is unsafe or ambiguous"
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise BenchmarkIntegrityError(f"{label} endpoint is unsafe or ambiguous")
    if parsed.scheme == "http" and (parsed.hostname or "").casefold() not in {
        "localhost", "127.0.0.1", "::1",
    }:
        raise BenchmarkIntegrityError(f"{label} plaintext endpoint is not loopback")
    return value.rstrip("/")


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
    return bool(
        isinstance(judge, Mapping)
        and config.get("judge_protocol") == "official"
        and judge.get("protocol") == "official"
        and judge.get("provider") == "openai"
        and judge.get("model") == LME_OFFICIAL_JUDGE_MODEL
        and isinstance(judge.get("base_url"), str)
        and judge["base_url"].rstrip("/") == LME_OFFICIAL_JUDGE_BASE_URL
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
        identity.get("model"), identity.get("dimension"),
    )
    observed = (
        snapshot.get("backend"), snapshot.get("quality"), snapshot.get("network_free"),
        snapshot.get("model"), snapshot.get("dimension"),
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


def _validate_model(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkIntegrityError(f"LongMemEval {label} model identity is malformed")
    for field in ("provider", "model", "base_url"):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise BenchmarkIntegrityError(f"LongMemEval {label} {field} is malformed")
    validate_safe_endpoint(value["base_url"], label=label)
    normalize_extra_body(
        value.get("extra_body", value.get("effective_extra_body", {})), label=label
    )
    return value


def _validate_embedding_identity(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkIntegrityError("LongMemEval embedding identity is malformed")
    required = {
        "configured", "backend", "quality", "network_free", "model",
        "base_url", "dimension", "fallback_policy",
    }
    configured_value = value.get("configured")
    expected_keys = required | ({"request_model"} if configured_value is True else set())
    if set(value) != expected_keys:
        raise BenchmarkIntegrityError("LongMemEval embedding identity is incomplete")
    configured = _bool(value.get("configured"), label="embedding configured")
    _bool(value.get("network_free"), label="embedding network_free")
    if not configured:
        expected = {
            "configured": False, "backend": "none", "quality": "none",
            "network_free": True, "model": None, "base_url": None,
            "dimension": None, "fallback_policy": "none",
        }
        if any(value.get(key) != expected[key] for key in expected):
            raise BenchmarkIntegrityError("disabled LongMemEval embedding identity disagrees")
        return value
    if value.get("backend") != "openai_compatible" or value.get("quality") != "semantic":
        raise BenchmarkIntegrityError("LongMemEval embedding backend/quality is unsupported")
    if value.get("network_free") is not False or value.get("fallback_policy") != "fail-closed":
        raise BenchmarkIntegrityError("LongMemEval embedding fallback/network posture is unsafe")
    if not isinstance(value.get("model"), str) or not value["model"].strip():
        raise BenchmarkIntegrityError("LongMemEval embedding model is malformed")
    request_model = value.get("request_model")
    if (
        not isinstance(request_model, str) or not request_model.strip()
        or request_model != request_model.strip()
    ):
        raise BenchmarkIntegrityError("LongMemEval embedding request model is malformed")
    dimension = value.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise BenchmarkIntegrityError("LongMemEval embedding dimension is malformed")
    base_url = validate_safe_endpoint(value.get("base_url"), label="embedding")
    if value.get("base_url") != base_url or value["model"] != (
        f"openai-compatible:{base_url}::{request_model}"
    ):
        raise BenchmarkIntegrityError("LongMemEval embedding vector-space identity differs")
    return value


def _validate_indexing(
    summary: object, *, require_healthy: bool = True,
    allow_incomplete: bool = False,
) -> bool:
    """Validate a durable convergence record and return usable completion.

    Historical execution segments may contain a bounded failed attempt followed
    by a successful resume.  Such a failure is evidence, not corruption; only a
    final successful row is required to have at least one complete acceptable
    summary.
    """

    if not isinstance(summary, Mapping):
        raise BenchmarkIntegrityError("LongMemEval indexing summary is absent")
    complete = summary.get("complete")
    healthy = summary.get("healthy")
    if not isinstance(complete, bool) or not isinstance(healthy, bool):
        raise BenchmarkIntegrityError("LongMemEval indexing completion state is malformed")
    if not complete and not allow_incomplete:
        raise BenchmarkIntegrityError("LongMemEval indexing did not complete")
    if not complete and healthy:
        raise BenchmarkIntegrityError("LongMemEval failed indexing claims health")
    if complete and require_healthy and not healthy:
        raise BenchmarkIntegrityError("LongMemEval indexing did not complete healthy")
    cycles = _finite_number(summary.get("cycles"), integer=True)
    max_cycles = _finite_number(summary.get("max_cycles"), integer=True)
    elapsed = _finite_number(summary.get("elapsed_s"))
    timeout = _finite_number(summary.get("timeout_s"))
    if (
        cycles is None or (complete and cycles <= 0)
        or max_cycles is None or max_cycles <= 0
    ):
        raise BenchmarkIntegrityError("LongMemEval indexing cycle counts are malformed")
    if (
        cycles > max_cycles or elapsed is None or timeout is None or timeout <= 0
        or (complete and elapsed > timeout)
    ):
        raise BenchmarkIntegrityError("LongMemEval indexing bounds are inconsistent")
    reports = summary.get("reports")
    if not isinstance(reports, list) or len(reports) != cycles or any(
        not isinstance(report, Mapping) for report in reports
    ):
        raise BenchmarkIntegrityError("LongMemEval indexing cycle reports are malformed")
    if any(
        not isinstance(report.get(field), bool)
        for report in reports
        for field in ("budget_exhausted", "skipped_locked")
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval indexing reports lack bounded-work state"
        )
    final = summary.get("final_status")
    if not isinstance(final, Mapping):
        raise BenchmarkIntegrityError("LongMemEval indexing final status is absent")
    if complete and any(
        _finite_number(final.get(field), integer=True) is None
        for field in ("pending_chunks", "quarantined_chunks")
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval completed indexing lacks durable backlog health"
        )
    quarantined = summary.get("quarantined")
    if not isinstance(quarantined, Mapping) or any(
        _finite_number(value, integer=True) is None
        for value in quarantined.values()
    ):
        raise BenchmarkIntegrityError("LongMemEval indexing quarantine summary is malformed")
    expected_quarantined = {
        key: value for key, value in final.items()
        if "quarantined" in key
        and _finite_number(value, integer=True) is not None
        and int(value) > 0
    }
    if dict(quarantined) != expected_quarantined:
        raise BenchmarkIntegrityError(
            "LongMemEval indexing quarantine summary differs from final status"
        )
    if healthy is not bool(complete and not expected_quarantined):
        raise BenchmarkIntegrityError("LongMemEval indexing health flag is inconsistent")
    for key, value in final.items():
        if key.startswith("pending_") or "quarantined" in key:
            normalized = _finite_number(value, integer=True)
            if normalized is None or (
                complete and (
                    key.startswith("pending_")
                    or (require_healthy and "quarantined" in key)
                ) and normalized != 0
            ):
                raise BenchmarkIntegrityError(
                    f"LongMemEval indexing final status {key!r} is not clean"
                )
    if complete and final.get("in_progress") is True:
        raise BenchmarkIntegrityError("LongMemEval completed indexing is still in progress")
    if complete and reports and (
        reports[-1].get("budget_exhausted") is not False
        or reports[-1].get("skipped_locked") is not False
    ):
        raise BenchmarkIntegrityError(
            "LongMemEval completed indexing final cycle is not exhausted cleanly"
        )
    if complete and require_healthy and any(
        int(value) != 0 for value in quarantined.values()
    ):
        raise BenchmarkIntegrityError("LongMemEval indexing quarantine summary is not clean")
    reason = summary.get("failure_reason")
    if complete and reason is not None:
        raise BenchmarkIntegrityError("LongMemEval completed indexing claims a failure")
    if not complete and (not isinstance(reason, str) or not reason.strip()):
        raise BenchmarkIntegrityError("LongMemEval failed indexing lacks a reason")
    cleanup = summary.get("cleanup_errors", [])
    if not isinstance(cleanup, list) or any(
        not isinstance(item, str) or not item.strip() for item in cleanup
    ):
        raise BenchmarkIntegrityError("LongMemEval indexing cleanup evidence is malformed")
    return bool(complete and (healthy or not require_healthy))


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
    reader = _validate_model(models.get("reader"), label="reader")
    judge = _validate_model(models.get("judge"), label="judge")
    pipeline = _validate_model(models.get("memory_pipeline"), label="memory pipeline")
    expected_model_fields = {
        "reader": {
            "provider", "model", "base_url", "temperature", "max_tokens",
            "extra_body",
        },
        "judge": {
            "provider", "model", "base_url", "temperature", "max_tokens", "n",
            "extra_body", "protocol", "evaluator_commit", "evaluator_sha256",
            "verdict_parser", "prompt_exact_official", "retry_policy",
        },
        "memory pipeline": {
            "provider", "model", "base_url", "thinking_mode",
            "effective_extra_body",
        },
    }
    for label, identity in (
        ("reader", reader), ("judge", judge), ("memory pipeline", pipeline),
    ):
        if set(identity) != expected_model_fields[label]:
            raise BenchmarkIntegrityError(
                f"LongMemEval {label} model identity fields differ"
            )
    expected_provider = lambda endpoint: (
        "deepseek" if endpoint.rstrip("/") == "https://api.deepseek.com"
        else "openai" if endpoint.rstrip("/") == LME_OFFICIAL_JUDGE_BASE_URL
        else "openai-compatible"
    )
    for identity, prefix in ((reader, "answer"), (judge, "judge")):
        if (
            identity.get("model") != config.get(f"{prefix}_model")
            or identity.get("base_url") != str(config.get(f"{prefix}_base_url", "")).rstrip("/")
            or identity.get("provider") != expected_provider(identity["base_url"])
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
        pipeline.get("provider") != expected_provider(pipeline["base_url"])
        or pipeline.get("model") != config.get("hymem_model")
        or pipeline.get("base_url") != str(config.get("hymem_base_url", "")).rstrip("/")
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
    host = (urlsplit(pipeline["base_url"]).hostname or "").casefold()
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
                _validate_indexing(
                    row.get("indexing"),
                    require_healthy=config["indexing_require_healthy"],
                )
                successful_indexed_ids.add(qid)
                row_indexing_summaries[qid] = {
                    key: value for key, value in dict(row["indexing"]).items()
                    if key != "question_id"
                }
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
            _validate_indexing(
                row.get("indexing"),
                require_healthy=config["indexing_require_healthy"],
            )
            successful_indexed_ids.add(qid)
            row_indexing_summaries[qid] = {
                key: value for key, value in dict(row["indexing"]).items()
                if key != "question_id"
            }
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
        for key, label in (
            ("reader_usage", "reader"), ("judge_usage", "judge"),
            ("retrieval_usage", "retrieval"),
            ("memory_pipeline_usage", "memory pipeline"),
        ):
            measured = _usage(segment.get(key), label=label)
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
        if latest_indexing is not None and (
            not indexing_runs or latest_indexing != indexing_runs[-1]
        ):
            raise BenchmarkIntegrityError(
                "LongMemEval segment latest indexing summary drifted"
            )
        if config.get("no_dream") is False:
            for summary in indexing_runs:
                summary_complete = _validate_indexing(
                    summary,
                    require_healthy=config["indexing_require_healthy"],
                    allow_incomplete=True,
                )
                summary_qid = summary.get("question_id") if isinstance(
                    summary, Mapping
                ) else None
                if summary_qid not in ordered_id_set:
                    raise BenchmarkIntegrityError(
                        "LongMemEval indexing summary has unknown/missing question id"
                    )
                if summary_complete:
                    indexed_ids.add(summary_qid)
                    indexed_summaries.setdefault(summary_qid, []).append({
                        key: value for key, value in dict(summary).items()
                        if key != "question_id"
                    })
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
