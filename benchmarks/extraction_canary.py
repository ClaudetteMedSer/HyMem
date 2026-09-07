"""Fail-fast Phase-1 extraction canary shared by scored memory benchmarks.

The canary deliberately calls :func:`hymem.extraction.chunk.extract_chunk`
instead of maintaining a benchmark-only parser or prompt.  Consequently it
exercises the same prompt, empty and non-empty omission verification passes,
source-id contract, item validators, and bounded retry/splitting behavior used
by dream indexing.

It is run with a dedicated client carrying the same endpoint/model/thinking
configuration as the memory pipeline.  That keeps its small provider spend
separate from scored pipeline usage and, because no ``HyMem``/database object
is constructed here, makes store contamination impossible.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from typing import Any, Mapping
from urllib.parse import urlsplit

try:
    from .strictness import (
        BenchmarkIntegrityError,
        run_cleanup_actions,
        sanitize_for_artifact,
        usage_snapshot,
    )
except (ImportError, ValueError):  # direct benchmark-script import
    from strictness import (  # type: ignore
        BenchmarkIntegrityError,
        run_cleanup_actions,
        sanitize_for_artifact,
        usage_snapshot,
    )
from hymem.extraction import chunk as chunk_extraction
from hymem.contrib.endpoint_policy import secret_free_endpoint_identity
from hymem.extraction.contract import (
    ACTIVE_EXTRACTION_PROMPT_VERSION,
    extraction_contract_binding,
    validate_effective_config_extraction_contract,
)
from hymem.extraction.jsonio import loads_exact_or_fenced
from hymem.extraction.llm import LLMClient
from hymem.extraction.retry import DEFAULT_RETRY_ATTEMPTS
from hymem.extraction.triples import normalize_combined_triple_item


# V17 replaces the output-only dense-table probe with a behavioral fixture. A
# pass now proves that provider requests carried the exact table-header/prelude
# and adjacent-prose contexts needed by two deliberately non-self-contained
# claims, while list and fence controls remained atomic.
EXTRACTION_CANARY_VERSION = "hymem-phase1-extraction-canary-v17"
EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID = 9_271_604_311
EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID = 9_271_604_312
EXTRACTION_CANARY_SOURCE_MESSAGE_IDS = (
    EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID,
    EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID,
)
# Backward-compatible alias for callers that used the old single-source probe.
# New policy/report evidence always carries both exact source ids.
EXTRACTION_CANARY_SOURCE_MESSAGE_ID = (
    EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID
)
# Backward-compatible public alias.  The value has one owner in the extraction
# contract module; reports additionally carry its mechanically derived binding.
EXTRACTION_CANARY_PROMPT_VERSION = ACTIVE_EXTRACTION_PROMPT_VERSION
EXTRACTION_CANARY_FIXTURE_VERSION = (
    "hymem-phase1-context-paths-four-leaf-v7"
)
EXTRACTION_CANARY_STRUCTURAL_PROBE_VERSION = (
    "hymem-phase1-markdown-atom-split-probe-v1"
)
EXTRACTION_CANARY_USAGE_ACCOUNTING = (
    "excluded_from_scored_usage_dedicated_memory_pipeline_client"
)
# Each of the two source records deterministically prepartitions into two
# leaves. The healthy path therefore costs eight calls: four primary passes,
# two clean-empty checks, and two non-empty omission checks. Recovery remains
# available inside a deliberately small 24-call / 72-provider-attempt envelope
# instead of inheriting the production chunk's 96-call ceiling.
EXTRACTION_CANARY_EXPECTED_PREPARTITION_LEAVES = 4
EXTRACTION_CANARY_NORMAL_PASS_COMPLETION_CALLS = 8
EXTRACTION_CANARY_MIN_PASS_COMPLETION_CALLS = 8
EXTRACTION_CANARY_MAX_COMPLETION_CALLS = 24
EXTRACTION_CANARY_MAX_PROVIDER_ATTEMPTS = (
    EXTRACTION_CANARY_MAX_COMPLETION_CALLS * DEFAULT_RETRY_ATTEMPTS
)

_TABLE_PRELUDE = "## Service deployment platform matrix\n"
_TABLE_HEADER_CONTEXT = (
    "| service | deployment platform | notes |\n"
    "| --- | --- | --- |\n"
)
_TABLE_CLAIM_ROW = (
    "| HyMem Canary Relay | Fly.io | supported canary assertion |"
)
_TABLE_PADDING_ROWS = [
    f"|  |  | filler-{index:03d}-"
    + hashlib.sha256(f"hymem-canary-table-padding-{index:03d}".encode()).hexdigest()[
        :56
    ]
    + " |"
    for index in range(64)
]
_TABLE_PADDING_ROWS.insert(52, _TABLE_CLAIM_ROW)
_TABLE_CONTENT = (
    _TABLE_PRELUDE
    + "\n"
    + _TABLE_HEADER_CONTEXT
    + "\n".join(_TABLE_PADDING_ROWS)
)

_LIST_CONTROL = (
    "- Inert parser fixture, not a factual assertion: subject token "
    "'List Canary Decoy'; relation-label fragments 'u' + 'ses'; object token "
    "'Redis'.\n  "
    + hashlib.sha256(b"hymem-canary-list-control").hexdigest() * 4
    + " This sentence is inert test data. It asserts no relationship. "
    + hashlib.sha256(b"hymem-canary-list-control").hexdigest() * 4
)
_FENCED_CODE_CONTROL = (
    "```text\n"
    "NON-FACTUAL PARSER FIXTURE; DO NOT EXTRACT\n"
    'subject = "Fence Canary Decoy"\n'
    'relation_label = "deploys_" + "to"\n'
    'object = "Decoy Cloud"\n'
    + hashlib.sha256(b"hymem-canary-fence-control").hexdigest() * 4
    + "\nThis sentence is inert test data. It asserts no relationship.\n"
    + hashlib.sha256(b"hymem-canary-fence-control").hexdigest() * 4
    + "\n```"
)
_PROSE_BOUNDARY_LEFT = (
    "Avery Boundary Canary's preferred database is identified in the "
    "following paragraph."
)
_PROSE_BOUNDARY_RIGHT = "PostgreSQL is that database."
_PROSE_PREFIX_PADDING = (
    "control-" + hashlib.sha256(b"hymem-canary-prose-prefix").hexdigest() * 20
)
_PROSE_SUFFIX_PADDING = (
    "control-" + hashlib.sha256(b"hymem-canary-prose-suffix").hexdigest() * 35
)
_PROSE_CONTENT = "\n\n".join([
    (
        "Markdown structural controls follow; only the next list and fenced "
        "block are non-factual fixtures."
    ),
    _LIST_CONTROL,
    _FENCED_CODE_CONTROL,
    _PROSE_PREFIX_PADDING + "\n" + _PROSE_BOUNDARY_LEFT,
    _PROSE_BOUNDARY_RIGHT + "\n" + _PROSE_SUFFIX_PADDING,
])

_CANARY_SOURCE_CONTENTS = (_TABLE_CONTENT, _PROSE_CONTENT)
# Retained as a private convenience for diagnostics/tests; extraction receives
# the two separately attributable records below, never this joined string as a
# source record.
_CANARY_CONTENT = "\n".join(_CANARY_SOURCE_CONTENTS)
EXTRACTION_CANARY_SOURCE_CONTENT_CHARS = sum(
    len(content) for content in _CANARY_SOURCE_CONTENTS
)
EXTRACTION_CANARY_TABLE_CONTINUATION_CLAIM_SHA256 = hashlib.sha256(
    _TABLE_CLAIM_ROW.encode("utf-8")
).hexdigest()
EXTRACTION_CANARY_PROSE_BOUNDARY_CLAIM_SHA256 = hashlib.sha256(
    (_PROSE_BOUNDARY_LEFT + "\n\n" + _PROSE_BOUNDARY_RIGHT).encode("utf-8")
).hexdigest()
EXTRACTION_CANARY_LIST_CONTROL_SHA256 = hashlib.sha256(
    _LIST_CONTROL.encode("utf-8")
).hexdigest()
EXTRACTION_CANARY_FENCED_CODE_CONTROL_SHA256 = hashlib.sha256(
    _FENCED_CODE_CONTROL.encode("utf-8")
).hexdigest()
EXTRACTION_CANARY_FIXTURE_SHA256 = hashlib.sha256(json.dumps(
    [
        {"content": content, "source_message_id": source_message_id}
        for source_message_id, content in zip(
            EXTRACTION_CANARY_SOURCE_MESSAGE_IDS,
            _CANARY_SOURCE_CONTENTS,
            strict=True,
        )
    ],
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()

# Import-time construction checks prevent a padding-only edit under the same
# fixture id from quietly moving either claim back into a self-contained leaf.
_TABLE_BOUNDARIES = chunk_extraction._markdown_table_boundary_points(
    _TABLE_CONTENT
)
_TABLE_INITIAL_CUT = chunk_extraction._semantic_split_point(_TABLE_CONTENT)
_PROSE_INITIAL_CUT = chunk_extraction._semantic_split_point(_PROSE_CONTENT)
_PROSE_RIGHT_START = _PROSE_CONTENT.index(_PROSE_BOUNDARY_RIGHT)
if (
    not _TABLE_BOUNDARIES
    or _TABLE_INITIAL_CUT not in _TABLE_BOUNDARIES
    or _TABLE_CONTENT.index(_TABLE_CLAIM_ROW) <= _TABLE_INITIAL_CUT
    or _PROSE_INITIAL_CUT != _PROSE_RIGHT_START
):  # pragma: no cover - deterministic module invariant
    raise RuntimeError("extraction canary no longer exercises both context cuts")

_TABLE_HEADER_START = len(_TABLE_PRELUDE) + 1
_EXPECTED_TABLE_CONTEXT = {
    "version": chunk_extraction.SOURCE_FRAGMENT_CONTEXT_VERSION,
    "kind": "introduced_canonical_markdown_table_header",
    "content": _TABLE_HEADER_CONTEXT,
    "source_content_start": _TABLE_HEADER_START,
    "source_content_end": _TABLE_HEADER_START + len(_TABLE_HEADER_CONTEXT),
    "applies_through_source_content_end": len(_TABLE_CONTENT),
    "prelude_kind": "atx_heading",
    "prelude_content": _TABLE_PRELUDE,
    "prelude_source_content_start": 0,
    "prelude_source_content_end": len(_TABLE_PRELUDE),
}
_PROSE_CONTEXT_START = max(
    0,
    _PROSE_RIGHT_START - chunk_extraction._MAX_SOURCE_BOUNDARY_CONTEXT_CHARS,
)
_EXPECTED_PROSE_BOUNDARY_CONTEXT = {
    "version": chunk_extraction.SOURCE_BOUNDARY_CONTEXT_VERSION,
    "kind": "preceding_adjacent_prose",
    "content": _PROSE_CONTENT[_PROSE_CONTEXT_START:_PROSE_RIGHT_START],
    "source_content_start": _PROSE_CONTEXT_START,
    "source_content_end": _PROSE_RIGHT_START,
    "applies_through_source_content_end": _PROSE_RIGHT_START + min(
        chunk_extraction._MAX_SOURCE_BOUNDARY_CONTEXT_CHARS,
        len(_PROSE_CONTENT) - _PROSE_RIGHT_START,
    ),
}


def _span(content: str, item: str) -> tuple[int, int]:
    start = content.index(item)
    return start, start + len(item)


_LIST_CONTROL_SPAN = _span(_PROSE_CONTENT, _LIST_CONTROL)
_FENCED_CODE_CONTROL_SPAN = _span(_PROSE_CONTENT, _FENCED_CODE_CONTROL)
_CANARY_EXPECTED_CLAIMS = (
    (
        "HyMem Canary Relay", "service", "deploys_to", "Fly.io", "platform",
        1, EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID,
    ),
    (
        "Avery Boundary Canary", "person", "prefers", "PostgreSQL", "database",
        1, EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID,
    ),
)
_NORMAL_EXECUTION_PATH = {
    "primary_requests": 4,
    "empty_verification_requests": 2,
    "omission_verification_requests": 2,
    "parsed_source_records": 8,
    "source_record_parse_failures": 0,
    "source_message_ids_seen": list(EXTRACTION_CANARY_SOURCE_MESSAGE_IDS),
    "table_claim_requests": 2,
    "table_claim_exact_context_requests": 2,
    "table_claim_self_contained_requests": 0,
    "table_claim_exact_context_emissions": 1,
    "table_claim_wrong_context_emissions": 0,
    "prose_claim_requests": 2,
    "prose_claim_exact_context_requests": 2,
    "prose_claim_self_contained_requests": 0,
    "prose_claim_exact_context_emissions": 1,
    "prose_claim_wrong_context_emissions": 0,
    "list_control_atomic_requests": 2,
    "fenced_code_control_atomic_requests": 2,
    "protected_control_split_boundaries": 0,
    "list_control_probe_atomic": True,
    "fenced_code_control_probe_atomic": True,
}
_CANARY_OPTIONAL_TRIPLE_FIELDS = (
    "value_text", "value_numeric", "value_unit", "temporal_scope",
)

_POLICY_KEYS = frozenset({
    "version", "required_before_indexing", "scope", "prompt_version",
    "extraction_contract",
    "fixture_version", "fixture_sha256", "source_content_chars",
    "source_split_policy_version", "clean_empty_recovery_policy_version",
    "source_message_ids", "table_continuation_claim_sha256",
    "prose_boundary_claim_sha256", "list_control_sha256",
    "fenced_code_control_sha256", "normal_pass_completion_calls",
    "structural_control_probe_version",
    "normal_execution_path",
    "expected_prepartition_leaves", "expected_claims", "usage_accounting",
    "store_writes",
    "minimum_pass_completion_calls", "max_completion_calls",
    "max_provider_attempts",
})
_CLIENT_KEYS = frozenset({
    "client_class", "model", "base_url", "thinking_mode",
    "effective_extra_body",
})
_SECRET_FREE_CLIENT_KEYS = frozenset({
    "client_class", "model", "endpoint_origin", "endpoint_sha256",
    "thinking_mode", "effective_extra_body",
})
_USAGE_KEYS = frozenset({
    "calls", "calls_available", "request_attempts",
    "request_attempts_available", "successful_responses",
    "successful_responses_available", "prompt_tokens", "completion_tokens",
    "total_tokens", "latency_s", "cost_usd", "token_usage_available",
    "latency_available", "cost_available",
})
_EVIDENCE_KEYS = frozenset({
    "expected_claim_index", "subject", "subject_type", "predicate", "object",
    "object_type", "polarity", "source_message_id", "value_text",
    "value_numeric", "value_unit", "temporal_scope", "subject_properties",
    "object_properties",
})
_EXECUTION_PATH_KEYS = frozenset({
    "primary_requests",
    "empty_verification_requests",
    "omission_verification_requests",
    "parsed_source_records",
    "source_record_parse_failures",
    "source_message_ids_seen",
    "table_claim_requests",
    "table_claim_exact_context_requests",
    "table_claim_self_contained_requests",
    "table_claim_exact_context_emissions",
    "table_claim_wrong_context_emissions",
    "prose_claim_requests",
    "prose_claim_exact_context_requests",
    "prose_claim_self_contained_requests",
    "prose_claim_exact_context_emissions",
    "prose_claim_wrong_context_emissions",
    "list_control_atomic_requests",
    "fenced_code_control_atomic_requests",
    "protected_control_split_boundaries",
    "list_control_probe_atomic",
    "fenced_code_control_probe_atomic",
})
_LIVE_COMMON_KEYS = frozenset({
    "status", "client", "client_closed", "completion_calls",
    "provider_attempts", "initial_prepartition_leaves", "usage",
    "duplicate_triples_collapsed", "matched_supported_claims",
    "missing_expected_claim_indexes", "valid_triples_returned",
    "valid_markers_returned", "claim_evidence", "execution_path",
})
_FAILED_EXTRA_KEYS = frozenset({"failure_reason", "failure_details"})
_SKIP_EXTRA_KEYS = frozenset({
    "status", "skip_reason", "completion_calls", "provider_attempts", "usage",
})
_FAILURE_REASONS = frozenset({
    "branch_incomplete", "call_failure", "clean_empty", "contract_failure",
    "incomplete_response", "input_contract_failure",
    "internal_validation_failure", "item_validation_failure",
    "output_limit_exceeded", "parse_failure", "resource_limit",
    "response_conflict", "shape_failure", "source_coverage_failure",
    "supported_claim_evidence_missing", "supported_claim_missing",
    "unexpected_canary_output", "unspecified_failure",
})
_SAFE_FAILURE_DETAIL = re.compile(r"^[a-z0-9_.\[\]-]+:[a-z0-9_]+$")
_EXPECTED_MODES = frozenset({
    "required", "failed", "pending", "simulation", "no_dream",
    "no_pending_work",
})


def extraction_canary_policy(
    *, prompt_version: str = ACTIVE_EXTRACTION_PROMPT_VERSION,
) -> dict[str, Any]:
    """Stable manifest identity for the mandatory extraction preflight."""

    return {
        "version": EXTRACTION_CANARY_VERSION,
        "required_before_indexing": True,
        "scope": "once_per_pending_run_configuration",
        "prompt_version": prompt_version,
        "extraction_contract": extraction_contract_binding(prompt_version),
        "fixture_version": EXTRACTION_CANARY_FIXTURE_VERSION,
        "fixture_sha256": EXTRACTION_CANARY_FIXTURE_SHA256,
        "source_content_chars": EXTRACTION_CANARY_SOURCE_CONTENT_CHARS,
        "source_split_policy_version": (
            chunk_extraction.SOURCE_RECORD_SPLIT_POLICY_VERSION
        ),
        "clean_empty_recovery_policy_version": (
            chunk_extraction.CLEAN_EMPTY_RECOVERY_POLICY_VERSION
        ),
        "source_message_ids": list(EXTRACTION_CANARY_SOURCE_MESSAGE_IDS),
        "table_continuation_claim_sha256": (
            EXTRACTION_CANARY_TABLE_CONTINUATION_CLAIM_SHA256
        ),
        "prose_boundary_claim_sha256": (
            EXTRACTION_CANARY_PROSE_BOUNDARY_CLAIM_SHA256
        ),
        "list_control_sha256": EXTRACTION_CANARY_LIST_CONTROL_SHA256,
        "fenced_code_control_sha256": (
            EXTRACTION_CANARY_FENCED_CODE_CONTROL_SHA256
        ),
        "normal_pass_completion_calls": (
            EXTRACTION_CANARY_NORMAL_PASS_COMPLETION_CALLS
        ),
        "structural_control_probe_version": (
            EXTRACTION_CANARY_STRUCTURAL_PROBE_VERSION
        ),
        "normal_execution_path": {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in _NORMAL_EXECUTION_PATH.items()
        },
        "expected_prepartition_leaves": (
            EXTRACTION_CANARY_EXPECTED_PREPARTITION_LEAVES
        ),
        "expected_claims": [
            {
                "subject": subject,
                "subject_type": subject_type,
                "predicate": predicate,
                "object": object_,
                "object_type": object_type,
                "polarity": polarity,
                "source_message_id": source_message_id,
                "value_text": None,
                "value_numeric": None,
                "value_unit": None,
                "temporal_scope": None,
                "subject_properties": {},
                "object_properties": {},
            }
            for (
                subject, subject_type, predicate, object_, object_type, polarity,
                source_message_id,
            ) in _CANARY_EXPECTED_CLAIMS
        ],
        "usage_accounting": EXTRACTION_CANARY_USAGE_ACCOUNTING,
        "store_writes": 0,
        "minimum_pass_completion_calls": (
            EXTRACTION_CANARY_MIN_PASS_COMPLETION_CALLS
        ),
        "max_completion_calls": EXTRACTION_CANARY_MAX_COMPLETION_CALLS,
        "max_provider_attempts": EXTRACTION_CANARY_MAX_PROVIDER_ATTEMPTS,
    }


def skipped_extraction_canary(
    reason: str,
    *,
    prompt_version: str = ACTIVE_EXTRACTION_PROMPT_VERSION,
) -> dict[str, Any]:
    """Exact zero-work evidence for an intentionally unattempted canary."""

    if reason not in {"simulation", "no_dream", "no_pending_work"}:
        raise ValueError("unknown extraction-canary skip reason")
    report = {
        **extraction_canary_policy(prompt_version=prompt_version),
        "status": (
            "not_run_no_pending"
            if reason == "no_pending_work" else "skipped_non_comparable"
        ),
        "skip_reason": reason,
        "completion_calls": 0,
        "provider_attempts": 0,
        "usage": None,
    }
    validate_extraction_canary_report(
        report, expected_mode=reason,
        expected_prompt_version=prompt_version,
    )
    return report


def _source_records() -> tuple[tuple[int, str], ...]:
    records: list[tuple[int, str]] = []
    for source_message_id, content, peer_suffix in zip(
        EXTRACTION_CANARY_SOURCE_MESSAGE_IDS,
        _CANARY_SOURCE_CONTENTS,
        ("table", "prose"),
        strict=True,
    ):
        payload = {
            "content": content,
            "source_created_at": "2000-01-01T00:00:00Z",
            "source_message_id": source_message_id,
            "source_peer_id": (
                f"benchmark-extraction-canary-{peer_suffix}-peer"
            ),
            "source_record_version": "hymem-claim-source-v2",
            "source_role": "user",
            "source_session_id": "benchmark-extraction-canary-session",
            "source_workspace_id": "benchmark-extraction-canary-workspace",
        }
        records.append((source_message_id, json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )))
    return tuple(records)


def _strict_equal(left: object, right: object) -> bool:
    """JSON-shape equality that does not equate ``True`` with ``1``."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float) and not math.isfinite(left):
        return False
    return left == right


def _integrity(message: str) -> BenchmarkIntegrityError:
    # Callers may validate untrusted artifacts. Keep every diagnostic structural
    # and never interpolate source, prompt, endpoint, model, or provider text.
    return BenchmarkIntegrityError(f"extraction canary {message}")


def validate_extraction_canary_policy(
    value: object,
    *,
    expected_prompt_version: str = ACTIVE_EXTRACTION_PROMPT_VERSION,
) -> dict[str, Any]:
    """Require the one exact supported policy, including both typed claims."""

    expected = extraction_canary_policy(
        prompt_version=expected_prompt_version
    )
    if not isinstance(value, Mapping) or set(value) != _POLICY_KEYS:
        raise _integrity("policy shape is invalid")
    actual = dict(value)
    if not _strict_equal(actual, expected):
        raise _integrity("policy identity is unsupported")
    return actual


def validate_extraction_canary_config_binding(
    policy: object,
    effective_hymem_config: object,
) -> dict[str, Any]:
    """Require a canary policy and effective HyMem config to share one contract."""

    try:
        binding = validate_effective_config_extraction_contract(
            effective_hymem_config
        )
    except (TypeError, ValueError) as exc:
        raise _integrity("effective config contract is invalid") from exc
    return validate_extraction_canary_policy(
        policy,
        expected_prompt_version=binding["prompt_version"],
    )


def extraction_canary_client_policy(
    *, base_url: str, model: str, thinking: str,
) -> dict[str, Any]:
    """Build the request-relevant identity expected from a pipeline client."""

    if (
        not isinstance(base_url, str) or not base_url
        or not isinstance(model, str) or not model
        or thinking not in {"auto", "disabled", "off", "enabled"}
    ):
        raise _integrity("expected client identity is malformed")
    parsed = urlsplit(base_url)
    host = (parsed.hostname or "").casefold()
    sends_thinking = thinking == "disabled" or (
        thinking == "auto"
        and ("deepseek" in host or "deepseek" in model.casefold())
    )
    return {
        "model": model,
        "base_url": base_url.rstrip("/"),
        "thinking_mode": thinking,
        "effective_extra_body": (
            {"thinking": {"type": "disabled"}} if sends_thinking else {}
        ),
    }


def _nonnegative_number(value: object, *, integer: bool = False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        or (integer and type(value) is not int)
    ):
        return None
    return value


def _validate_usage(
    value: object, *, completion_calls: int, provider_attempts: int,
    passed: bool,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _USAGE_KEYS:
        raise _integrity("usage shape is invalid")
    usage = dict(value)

    def available(field: str, flag: str, *, integer: bool = False):
        marker = usage.get(flag)
        if not isinstance(marker, bool):
            raise _integrity("usage availability is invalid")
        item = usage.get(field)
        if marker is False:
            if item is not None:
                raise _integrity("usage availability is inconsistent")
            return None
        normalized = _nonnegative_number(item, integer=integer)
        if normalized is None:
            raise _integrity("usage value is invalid")
        return normalized

    calls = available("calls", "calls_available", integer=True)
    attempts = available(
        "request_attempts", "request_attempts_available", integer=True,
    )
    successes = available(
        "successful_responses", "successful_responses_available", integer=True,
    )
    latency = available("latency_s", "latency_available")
    available("cost_usd", "cost_available")
    if any(item is None for item in (calls, attempts, successes, latency)):
        raise _integrity("usage lacks exact attempt accounting")
    if (
        calls != successes
        or attempts != provider_attempts
        or calls > completion_calls
        or attempts < completion_calls
        or (passed and calls != completion_calls)
    ):
        raise _integrity("usage counters do not reconcile")

    token_available = usage.get("token_usage_available")
    if not isinstance(token_available, bool):
        raise _integrity("token availability is invalid")
    tokens: list[int] = []
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = usage.get(field)
        if not token_available:
            if item is not None:
                raise _integrity("token availability is inconsistent")
            continue
        normalized = _nonnegative_number(item, integer=True)
        if normalized is None:
            raise _integrity("token usage is invalid")
        tokens.append(normalized)
    if token_available and tokens[2] != tokens[0] + tokens[1]:
        raise _integrity("token totals do not reconcile")


def _validate_execution_path(
    value: object,
    *,
    completion_calls: int,
    initial_leaves: int,
    passed: bool,
) -> None:
    """Validate bounded evidence derived from the actual provider requests."""

    if not isinstance(value, Mapping) or set(value) != _EXECUTION_PATH_KEYS:
        raise _integrity("execution path shape is invalid")
    path = dict(value)
    source_ids = path.pop("source_message_ids_seen")
    list_probe_atomic = path.pop("list_control_probe_atomic")
    fence_probe_atomic = path.pop("fenced_code_control_probe_atomic")
    if not isinstance(list_probe_atomic, bool) or not isinstance(
        fence_probe_atomic, bool
    ):
        raise _integrity("execution path structural probes are invalid")
    if (
        not isinstance(source_ids, list)
        or len(source_ids) > len(EXTRACTION_CANARY_SOURCE_MESSAGE_IDS)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in source_ids
        )
        or source_ids != sorted(set(source_ids))
    ):
        raise _integrity("execution path source ids are invalid")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in path.values()
    ):
        raise _integrity("execution path counters are invalid")

    primary = path["primary_requests"]
    empty = path["empty_verification_requests"]
    omission = path["omission_verification_requests"]
    if primary + empty + omission != completion_calls:
        raise _integrity("execution path request counts do not reconcile")
    if any(
        path[key] > completion_calls
        for key in (
            "parsed_source_records",
            "source_record_parse_failures",
            "table_claim_requests",
            "table_claim_exact_context_requests",
            "table_claim_self_contained_requests",
            "table_claim_exact_context_emissions",
            "table_claim_wrong_context_emissions",
            "prose_claim_requests",
            "prose_claim_exact_context_requests",
            "prose_claim_self_contained_requests",
            "prose_claim_exact_context_emissions",
            "prose_claim_wrong_context_emissions",
            "list_control_atomic_requests",
            "fenced_code_control_atomic_requests",
        )
    ):
        raise _integrity("execution path counters exceed requests")
    if (
        path["table_claim_exact_context_requests"]
        > path["table_claim_requests"]
        or path["table_claim_self_contained_requests"]
        > path["table_claim_requests"]
        or path["prose_claim_exact_context_requests"]
        > path["prose_claim_requests"]
        or path["prose_claim_self_contained_requests"]
        > path["prose_claim_requests"]
        or path["table_claim_exact_context_emissions"]
        + path["table_claim_wrong_context_emissions"] > completion_calls
        or path["prose_claim_exact_context_emissions"]
        + path["prose_claim_wrong_context_emissions"] > completion_calls
        or path["protected_control_split_boundaries"] > 4
    ):
        raise _integrity("execution path relationships are inconsistent")

    if passed and (
        primary < initial_leaves
        or empty < 2
        or omission < 2
        or path["parsed_source_records"] != completion_calls
        or path["source_record_parse_failures"] != 0
        or source_ids != list(EXTRACTION_CANARY_SOURCE_MESSAGE_IDS)
        or path["table_claim_requests"] < 2
        or path["table_claim_exact_context_requests"]
        != path["table_claim_requests"]
        or path["table_claim_self_contained_requests"] != 0
        or path["table_claim_exact_context_emissions"] < 1
        or path["table_claim_wrong_context_emissions"] != 0
        or path["prose_claim_requests"] < 2
        or path["prose_claim_exact_context_requests"]
        != path["prose_claim_requests"]
        or path["prose_claim_self_contained_requests"] != 0
        or path["prose_claim_exact_context_emissions"] < 1
        or path["prose_claim_wrong_context_emissions"] != 0
        or path["list_control_atomic_requests"] < 2
        or path["fenced_code_control_atomic_requests"] < 2
        or path["protected_control_split_boundaries"] != 0
        or not list_probe_atomic
        or not fence_probe_atomic
    ):
        raise _integrity("passed report did not exercise exact context paths")
    if (
        passed
        and completion_calls == EXTRACTION_CANARY_NORMAL_PASS_COMPLETION_CALLS
        and not _strict_equal(value, _NORMAL_EXECUTION_PATH)
    ):
        raise _integrity("normal execution path differs from exact policy")


class _RecordingClient:
    """Transparent completion recorder; identity/telemetry stay on delegate."""

    def __init__(self, delegate: LLMClient):
        self.delegate = delegate
        self.requests: list[Any] = []
        self.responses: list[tuple[Any, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def complete(self, request: Any) -> str:
        self.requests.append(request)
        response = self.delegate.complete(request)
        self.responses.append((request, response))
        return response


def _request_source_payloads(request: object) -> tuple[list[dict], int]:
    """Read only the exact source-record excerpt from one canary request."""

    system = getattr(request, "system", None)
    user = getattr(request, "user", None)
    if not isinstance(system, str) or not isinstance(user, str):
        return [], 1
    if "OMISSION VERIFICATION PASS" in system:
        prefix = (
            "Excerpt (the exact same source records as the primary pass):\n"
            '"""\n'
        )
    else:
        prefix = 'Excerpt:\n"""\n'
    suffix = '\n"""\n\n'
    if not user.startswith(prefix):
        return [], 1
    end = user.find(suffix, len(prefix))
    if end < 0:
        return [], 1
    encoded_records = user[len(prefix):end]
    payloads: list[dict] = []
    failures = 0
    for encoded in encoded_records.splitlines():
        try:
            payload = json.loads(encoded)
        except (TypeError, ValueError, json.JSONDecodeError):
            failures += 1
            continue
        if not isinstance(payload, dict):
            failures += 1
            continue
        payloads.append(payload)
    if not payloads and not failures:
        failures = 1
    return payloads, failures


def _payload_source_range(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    source_message_id = payload.get("source_message_id")
    try:
        source_index = EXTRACTION_CANARY_SOURCE_MESSAGE_IDS.index(
            source_message_id
        )
    except ValueError:
        return None
    content = payload.get("content")
    if not isinstance(content, str):
        return None
    if payload.get("source_record_version") == "hymem-claim-source-v2":
        return 0, len(_CANARY_SOURCE_CONTENTS[source_index])
    start = payload.get("source_content_start")
    end = payload.get("source_content_end")
    if (
        isinstance(start, bool) or not isinstance(start, int)
        or isinstance(end, bool) or not isinstance(end, int)
        or start < 0 or end != start + len(content)
    ):
        return None
    return start, end


def _response_expected_claim_indexes(response: object) -> list[int]:
    """Return exact canary claims present in one contract-valid item stream."""

    if not isinstance(response, str):
        return []
    data = loads_exact_or_fenced(response)
    if not isinstance(data, dict) or not isinstance(data.get("triples"), list):
        return []
    indexes: list[int] = []
    for item in data["triples"]:
        normalized, errors, _normalizations = normalize_combined_triple_item(
            item, require_source_message_id=True,
        )
        if normalized is None or errors:
            continue
        for index, (
            subject, subject_type, predicate, object_, object_type, polarity,
            source_message_id,
        ) in enumerate(_CANARY_EXPECTED_CLAIMS):
            expected = {
                "subject": subject,
                "subject_type": subject_type,
                "predicate": predicate,
                "object": object_,
                "object_type": object_type,
                "polarity": polarity,
                "source_message_id": source_message_id,
            }
            if all(
                key in normalized and _strict_equal(normalized[key], value)
                for key, value in expected.items()
            ):
                indexes.append(index)
    return indexes


def _control_probe_is_atomic(content: str) -> bool:
    """Exercise real splitting on one exact Markdown atom without a provider."""

    payload = json.loads(_source_records()[1][1])
    payload["content"] = content
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    unit = chunk_extraction._ExtractionUnit(
        text=encoded,
        source_records=((EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID, encoded),),
    )
    return chunk_extraction._split_unit(unit) is None


def _request_execution_path(
    requests: list[Any], responses: list[tuple[Any, Any]],
) -> dict[str, Any]:
    """Summarize context/atomicity evidence without retaining fixture text."""

    path: dict[str, Any] = {
        key: 0 for key in _EXECUTION_PATH_KEYS
        if key not in {
            "source_message_ids_seen",
            "list_control_probe_atomic",
            "fenced_code_control_probe_atomic",
        }
    }
    source_ids: set[int] = set()
    exact_context_by_request: dict[int, set[int]] = {}
    protected_split_boundaries: set[tuple[str, int]] = set()
    for request in requests:
        exact_context_indexes: set[int] = set()
        system = getattr(request, "system", "")
        if "OMISSION VERIFICATION PASS" in system:
            path["omission_verification_requests"] += 1
        elif "EMPTY VERIFICATION PASS" in system:
            path["empty_verification_requests"] += 1
        else:
            path["primary_requests"] += 1
        payloads, failures = _request_source_payloads(request)
        path["source_record_parse_failures"] += failures
        path["parsed_source_records"] += len(payloads)
        for payload in payloads:
            source_message_id = payload.get("source_message_id")
            if (
                isinstance(source_message_id, int)
                and not isinstance(source_message_id, bool)
                and source_message_id > 0
            ):
                source_ids.add(source_message_id)
            content = payload.get("content")
            if not isinstance(content, str):
                continue
            if source_message_id == EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID:
                if _TABLE_CLAIM_ROW in content:
                    path["table_claim_requests"] += 1
                    if _strict_equal(
                        payload.get("source_fragment_context"),
                        _EXPECTED_TABLE_CONTEXT,
                    ):
                        path["table_claim_exact_context_requests"] += 1
                        exact_context_indexes.add(0)
                    if (
                        _TABLE_PRELUDE in content
                        and _TABLE_HEADER_CONTEXT in content
                    ):
                        path["table_claim_self_contained_requests"] += 1
            elif source_message_id == EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID:
                if _PROSE_BOUNDARY_RIGHT in content:
                    path["prose_claim_requests"] += 1
                    if _strict_equal(
                        payload.get("source_boundary_context"),
                        _EXPECTED_PROSE_BOUNDARY_CONTEXT,
                    ):
                        path["prose_claim_exact_context_requests"] += 1
                        exact_context_indexes.add(1)
                    if _PROSE_BOUNDARY_LEFT in content:
                        path["prose_claim_self_contained_requests"] += 1
                if _LIST_CONTROL in content:
                    path["list_control_atomic_requests"] += 1
                if _FENCED_CODE_CONTROL in content:
                    path["fenced_code_control_atomic_requests"] += 1
                source_range = _payload_source_range(payload)
                if source_range is not None:
                    for label, span in (
                        ("list", _LIST_CONTROL_SPAN),
                        ("fence", _FENCED_CODE_CONTROL_SPAN),
                    ):
                        for boundary in source_range:
                            if span[0] < boundary < span[1]:
                                protected_split_boundaries.add((label, boundary))
        exact_context_by_request[id(request)] = exact_context_indexes
    for request, response in responses:
        exact_context_indexes = exact_context_by_request.get(id(request), set())
        for claim_index in _response_expected_claim_indexes(response):
            prefix = "table" if claim_index == 0 else "prose"
            context = "exact_context" if claim_index in exact_context_indexes else (
                "wrong_context"
            )
            path[f"{prefix}_claim_{context}_emissions"] += 1
    path["source_message_ids_seen"] = sorted(source_ids)
    path["protected_control_split_boundaries"] = len(
        protected_split_boundaries
    )
    path["list_control_probe_atomic"] = _control_probe_is_atomic(
        _LIST_CONTROL
    )
    path["fenced_code_control_probe_atomic"] = _control_probe_is_atomic(
        _FENCED_CODE_CONTROL
    )
    return path


def _validate_client(
    value: object, *, expected_client: Mapping[str, Any] | None,
) -> None:
    keys = frozenset(value) if isinstance(value, Mapping) else frozenset()
    if not isinstance(value, Mapping) or keys not in {
        _CLIENT_KEYS, _SECRET_FREE_CLIENT_KEYS,
    }:
        raise _integrity("client identity shape is invalid")
    client = dict(value)
    secret_free = frozenset(client) == _SECRET_FREE_CLIENT_KEYS
    client_class = client.get("client_class")
    model = client.get("model")
    endpoint_origin = (
        client.get("endpoint_origin") if secret_free else client.get("base_url")
    )
    endpoint_sha256 = client.get("endpoint_sha256") if secret_free else None
    thinking = client.get("thinking_mode")
    if (
        not isinstance(client_class, str) or not client_class.strip()
        or client_class != client_class.strip() or len(client_class) > 300
        or not isinstance(model, str) or not model.strip()
        or model != model.strip() or len(model) > 300
        or not isinstance(endpoint_origin, str) or not endpoint_origin
        or endpoint_origin != endpoint_origin.strip() or len(endpoint_origin) > 2048
        or (
            secret_free and (
                not isinstance(endpoint_sha256, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", endpoint_sha256) is None
            )
        )
        or thinking not in {"auto", "disabled", "off", "enabled"}
    ):
        raise _integrity("client identity is malformed")
    try:
        parsed = urlsplit(endpoint_origin)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise _integrity("client endpoint is malformed") from exc
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname or parsed.username is not None
        or parsed.password is not None or parsed.query or parsed.fragment
        or port is not None and port <= 0
        or (secret_free and parsed.path not in {"", "/"})
        or endpoint_origin.endswith("/")
    ):
        raise _integrity("client endpoint is malformed")
    canonical_request = extraction_canary_client_policy(
        base_url=endpoint_origin, model=model, thinking=thinking,
    )
    if not _strict_equal(
        client.get("effective_extra_body"),
        canonical_request["effective_extra_body"],
    ):
        raise _integrity("client request body is inconsistent")
    if expected_client is not None:
        try:
            if secret_free:
                if "endpoint_sha256" in expected_client:
                    expected = {
                        key: expected_client[key]
                        for key in (
                            "model", "endpoint_origin", "endpoint_sha256",
                            "thinking_mode", "effective_extra_body",
                        )
                    }
                else:
                    expected = extraction_canary_client_policy(
                        base_url=expected_client["base_url"],
                        model=expected_client["model"],
                        thinking=expected_client["thinking_mode"],
                    )
                    expected = {
                        "model": expected["model"],
                        **secret_free_endpoint_identity(
                            expected["base_url"], label="extraction canary"
                        ),
                        "thinking_mode": expected["thinking_mode"],
                        "effective_extra_body": expected["effective_extra_body"],
                    }
            else:
                expected = extraction_canary_client_policy(
                    base_url=expected_client["base_url"],
                    model=expected_client["model"],
                    thinking=expected_client["thinking_mode"],
                )
        except (KeyError, TypeError) as exc:
            raise _integrity("expected client identity is malformed") from exc
        observed = {key: client.get(key) for key in expected}
        if not _strict_equal(observed, expected):
            raise _integrity("client differs from memory pipeline")


def _validate_claim_evidence(value: object) -> list[int]:
    if not isinstance(value, list) or len(value) > len(_CANARY_EXPECTED_CLAIMS):
        raise _integrity("claim evidence shape is invalid")
    indexes: list[int] = []
    for evidence in value:
        if not isinstance(evidence, Mapping) or set(evidence) != _EVIDENCE_KEYS:
            raise _integrity("claim evidence item shape is invalid")
        index = evidence.get("expected_claim_index")
        if (
            isinstance(index, bool) or not isinstance(index, int)
            or index < 0 or index >= len(_CANARY_EXPECTED_CLAIMS)
            or index in indexes
        ):
            raise _integrity("claim evidence indexes are invalid")
        expected = {
            "expected_claim_index": index,
            **extraction_canary_policy()["expected_claims"][index],
        }
        if not _strict_equal(dict(evidence), expected):
            raise _integrity("claim evidence does not match the canary contract")
        indexes.append(index)
    if indexes != sorted(indexes):
        raise _integrity("claim evidence order is invalid")
    return indexes


def validate_extraction_canary_report(
    value: object,
    *,
    expected_mode: str,
    expected_client: Mapping[str, Any] | None = None,
    require_client_closed: bool = False,
    expected_prompt_version: str = ACTIVE_EXTRACTION_PROMPT_VERSION,
) -> dict[str, Any]:
    """Validate one report at a live, checkpoint, artifact, or registry edge.

    ``expected_mode`` is execution context, not a claim read from the report.
    This prevents a forged status from changing whether the canary was required.
    The function intentionally emits structural errors only.
    """

    if expected_mode not in _EXPECTED_MODES:
        raise _integrity("validation mode is unsupported")
    if not isinstance(value, Mapping):
        raise _integrity("report is absent")
    report = dict(value)
    validate_extraction_canary_policy(
        {key: report[key] for key in _POLICY_KEYS if key in report},
        expected_prompt_version=expected_prompt_version,
    )

    if expected_mode == "pending":
        if set(report) != _POLICY_KEYS | {"status"} or report.get("status") != "pending":
            raise _integrity("pending report shape is invalid")
        return report

    if expected_mode in {"simulation", "no_dream", "no_pending_work"}:
        expected_status = (
            "not_run_no_pending"
            if expected_mode == "no_pending_work" else "skipped_non_comparable"
        )
        if (
            set(report) != _POLICY_KEYS | _SKIP_EXTRA_KEYS
            or report.get("status") != expected_status
            or report.get("skip_reason") != expected_mode
            or type(report.get("completion_calls")) is not int
            or report.get("completion_calls") != 0
            or type(report.get("provider_attempts")) is not int
            or report.get("provider_attempts") != 0
            or report.get("usage") is not None
            or expected_client is not None
        ):
            raise _integrity("zero-work report is inconsistent")
        return report

    failed = expected_mode == "failed"
    expected_keys = _POLICY_KEYS | _LIVE_COMMON_KEYS | (
        _FAILED_EXTRA_KEYS if failed else frozenset()
    )
    if set(report) != expected_keys:
        raise _integrity("live report shape is invalid")
    if report.get("status") != ("failed" if failed else "passed"):
        raise _integrity("live report status is inconsistent")
    completion_calls = report.get("completion_calls")
    provider_attempts = report.get("provider_attempts")
    initial_leaves = report.get("initial_prepartition_leaves")
    if (
        type(completion_calls) is not int
        or completion_calls < (0 if failed else 1)
        or completion_calls > EXTRACTION_CANARY_MAX_COMPLETION_CALLS
        or type(provider_attempts) is not int
        or provider_attempts < completion_calls
        or provider_attempts > EXTRACTION_CANARY_MAX_PROVIDER_ATTEMPTS
        or type(initial_leaves) is not int
        or initial_leaves < 0
        or initial_leaves > EXTRACTION_CANARY_EXPECTED_PREPARTITION_LEAVES
    ):
        raise _integrity("attempt counters are invalid")
    if (
        not failed
        and completion_calls < EXTRACTION_CANARY_MIN_PASS_COMPLETION_CALLS
    ):
        raise _integrity("passed report lacks omission verification")
    _validate_usage(
        report.get("usage"), completion_calls=completion_calls,
        provider_attempts=provider_attempts, passed=not failed,
    )
    _validate_execution_path(
        report.get("execution_path"),
        completion_calls=completion_calls,
        initial_leaves=initial_leaves,
        passed=not failed,
    )
    _validate_client(report.get("client"), expected_client=expected_client)
    closed = report.get("client_closed")
    if closed is not None and not isinstance(closed, bool):
        raise _integrity("client lifecycle evidence is invalid")
    if require_client_closed and closed is not True:
        raise _integrity("configured client was not closed successfully")

    evidence_indexes = _validate_claim_evidence(report.get("claim_evidence"))
    matched = report.get("matched_supported_claims")
    missing = report.get("missing_expected_claim_indexes")
    triple_count = report.get("valid_triples_returned")
    marker_count = report.get("valid_markers_returned")
    duplicate_count = report.get("duplicate_triples_collapsed")
    expected_missing = [
        index for index in range(len(_CANARY_EXPECTED_CLAIMS))
        if index not in evidence_indexes
    ]
    if (
        type(matched) is not int or matched != len(evidence_indexes)
        or missing != expected_missing
        or type(triple_count) is not int or triple_count < len(evidence_indexes)
        or type(marker_count) is not int or marker_count < 0
        or type(duplicate_count) is not int or duplicate_count < 0
    ):
        raise _integrity("claim evidence counts do not reconcile")
    if not failed and (
        evidence_indexes != list(range(len(_CANARY_EXPECTED_CLAIMS)))
        or matched != len(_CANARY_EXPECTED_CLAIMS) or missing
        or triple_count != len(_CANARY_EXPECTED_CLAIMS)
        or marker_count != 0
        or duplicate_count != 0
        or initial_leaves != EXTRACTION_CANARY_EXPECTED_PREPARTITION_LEAVES
    ):
        raise _integrity("passed report is not the exact behavioral canary result")
    if failed:
        reason = report.get("failure_reason")
        details = report.get("failure_details")
        if reason not in _FAILURE_REASONS:
            raise _integrity("failure reason is unsupported")
        if (
            not isinstance(details, list) or len(details) > 32
            or any(
                not isinstance(detail, str) or len(detail) > 160
                or _SAFE_FAILURE_DETAIL.fullmatch(detail) is None
                for detail in details
            )
        ):
            raise _integrity("failure details are invalid")
        if reason == "clean_empty" and (
            triple_count != 0 or marker_count != 0 or evidence_indexes
        ):
            raise _integrity("clean-empty failure carries extracted evidence")
        if reason in {
            "supported_claim_missing", "supported_claim_evidence_missing",
        } and not expected_missing:
            raise _integrity("claim-missing failure has complete evidence")
    return report


def _safe_client_identity(client: object) -> dict[str, Any]:
    return sanitize_for_artifact({
        "client_class": f"{type(client).__module__}.{type(client).__qualname__}",
        "model": getattr(client, "model", None),
        "base_url": getattr(client, "base_url", None),
        "thinking_mode": getattr(client, "thinking_mode", None),
        "effective_extra_body": getattr(client, "effective_extra_body", None),
    })


def secret_free_extraction_canary_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a validated runtime canary into route-free durable evidence."""

    projected = dict(report)
    client = projected.get("client")
    if not isinstance(client, Mapping):
        return projected
    base_url = client.get("base_url")
    endpoint = secret_free_endpoint_identity(
        base_url, label="extraction canary"
    )
    projected["client"] = {
        key: value for key, value in client.items() if key != "base_url"
    }
    projected["client"].update(endpoint)
    return projected


def _usage_snapshot(client: object) -> dict[str, Any]:
    # Keep the common canary independent of benchmark adapter client classes,
    # while using the same strict missing-data semantics as scored artifacts.
    return usage_snapshot(client)


class ExtractionCanaryError(BenchmarkIntegrityError):
    """The configured memory pipeline cannot satisfy Phase-1's contract."""

    def __init__(self, message: str, report: Mapping[str, Any]):
        super().__init__(message)
        self.report = dict(report)


def run_extraction_canary(
    client: LLMClient,
    *,
    usage_accounting: str = EXTRACTION_CANARY_USAGE_ACCOUNTING,
    prompt_version: str = ACTIVE_EXTRACTION_PROMPT_VERSION,
) -> dict[str, Any]:
    """Exercise the real Phase-1 contract and require all supported claims.

    No raw provider content is retained.  Failure reports contain only the
    extraction layer's bounded, allowlisted diagnostics.
    """

    source_records = _source_records()
    recording_client = _RecordingClient(client)
    result = chunk_extraction.extract_chunk(
        recording_client,
        _CANARY_CONTENT,
        source_records=source_records,
        completion_call_limit=EXTRACTION_CANARY_MAX_COMPLETION_CALLS,
    )
    base = {
        **extraction_canary_policy(prompt_version=prompt_version),
        "usage_accounting": usage_accounting,
        "client": _safe_client_identity(client),
        # A direct caller owns its client. ``run_configured_*`` replaces this
        # with an exact close outcome after releasing its dedicated transport.
        "client_closed": None,
        "completion_calls": result.completion_calls,
        "provider_attempts": result.provider_attempts,
        "initial_prepartition_leaves": result.initial_prepartition_leaves,
        "duplicate_triples_collapsed": result.duplicate_triples_collapsed,
        "usage": _usage_snapshot(client),
        "execution_path": _request_execution_path(
            recording_client.requests, recording_client.responses,
        ),
    }
    if result.failed:
        report = {
            **base,
            "status": "failed",
            "failure_reason": result.failure_reason or "unspecified_failure",
            "failure_details": list(result.failure_details),
            "matched_supported_claims": 0,
            "missing_expected_claim_indexes": list(
                range(len(_CANARY_EXPECTED_CLAIMS))
            ),
            "valid_triples_returned": len(result.triples),
            "valid_markers_returned": len(result.markers),
            "claim_evidence": [],
        }
        validate_extraction_canary_report(
            report, expected_mode="failed",
            expected_prompt_version=prompt_version,
        )
        raise ExtractionCanaryError(
            "memory Phase-1 extraction canary failed: "
            f"{report['failure_reason']}",
            report,
        )

    types_by_entity: dict[str, set[str]] = {}
    for entity, entity_type in result.entity_type_hints.items():
        types_by_entity.setdefault(entity, set()).add(entity_type)
    expected_types_by_entity = {
        entity: {entity_type}
        for (
            subject, subject_type, _predicate, object_, object_type, _polarity,
            _source_message_id,
        )
        in _CANARY_EXPECTED_CLAIMS
        for entity, entity_type in (
            (subject, subject_type), (object_, object_type),
        )
    }
    properties_by_entity: dict[str, list[dict[str, str]]] = {}
    for entity, properties in result.entity_property_hints.items():
        properties_by_entity.setdefault(entity, []).append(properties)

    # Only structural, allowlisted field codes may cross the artifact boundary.
    # In particular, never retain a hallucinated value or property payload.
    unexpected_details: list[str] = []

    def note_unexpected(detail: str) -> None:
        if detail not in unexpected_details:
            unexpected_details.append(detail)

    if any(
        expected_types_by_entity.get(entity) != entity_types
        for entity, entity_types in types_by_entity.items()
    ):
        note_unexpected("entity_type_hints:unexpected")
    if result.entity_property_hints != {}:
        note_unexpected("entity_property_hints:unexpected")

    evidence: list[dict[str, Any]] = []
    for index, (
        subject, subject_type, predicate, object_, object_type, polarity,
        source_message_id,
    ) in enumerate(_CANARY_EXPECTED_CLAIMS):
        matches = [
            triple for triple in result.triples
            if (
                triple.subject == subject
                and triple.predicate == predicate
                and triple.object == object_
                and triple.polarity == polarity
                and triple.source_message_id == source_message_id
            )
        ]
        if len(matches) != 1:
            continue
        match = matches[0]
        exact_claim = True
        for field_name in _CANARY_OPTIONAL_TRIPLE_FIELDS:
            if getattr(match, field_name) is not None:
                note_unexpected(
                    f"expected_claims[{index}].{field_name}:unexpected"
                )
                exact_claim = False
        for side, entity, entity_type in (
            ("subject", match.subject, subject_type),
            ("object", match.object, object_type),
        ):
            if types_by_entity.get(entity) != {entity_type}:
                note_unexpected(
                    f"expected_claims[{index}].{side}_type:unexpected"
                )
                exact_claim = False
            if any(properties_by_entity.get(entity, ())):
                note_unexpected(
                    f"expected_claims[{index}].{side}_properties:unexpected"
                )
                exact_claim = False
        if exact_claim:
            # Evidence is the fixed fixture contract, never provider-supplied
            # display strings or enrichment values. Core matching above is
            # literal after the production contract's normalization.
            evidence.append({
                "expected_claim_index": index,
                **base["expected_claims"][index],
            })
    evidence_indexes = {
        item["expected_claim_index"] for item in evidence
    }
    missing_expected_claim_indexes = [
        index for index in range(len(_CANARY_EXPECTED_CLAIMS))
        if index not in evidence_indexes
    ]
    exact_result = (
        not missing_expected_claim_indexes
        and not unexpected_details
        and len(result.triples) == len(_CANARY_EXPECTED_CLAIMS)
        and not result.markers
        and result.duplicate_triples_collapsed == 0
    )
    if not exact_result:
        matched_count = len(evidence)
        has_unexpected_output = (
            len(result.triples) > len(_CANARY_EXPECTED_CLAIMS)
            or bool(result.markers)
            or result.duplicate_triples_collapsed > 0
            or bool(unexpected_details)
        )
        failure_reason = (
            "clean_empty" if not result.triples and not result.markers
            else "unexpected_canary_output" if has_unexpected_output
            else "supported_claim_evidence_missing"
        )
        report = {
            **base,
            "status": "failed",
            "failure_reason": failure_reason,
            "failure_details": (
                unexpected_details
                if failure_reason == "unexpected_canary_output" else []
            ),
            "matched_supported_claims": matched_count,
            "missing_expected_claim_indexes": missing_expected_claim_indexes,
            "valid_triples_returned": len(result.triples),
            "valid_markers_returned": len(result.markers),
            "claim_evidence": evidence,
        }
        validate_extraction_canary_report(
            report, expected_mode="failed",
            expected_prompt_version=prompt_version,
        )
        raise ExtractionCanaryError(
            "memory Phase-1 extraction canary did not return exactly the two "
            "typed context-dependent supported claims and zero markers "
            "attributable to the canary sources",
            report,
        )

    report = {
        **base,
        "status": "passed",
        "matched_supported_claims": len(_CANARY_EXPECTED_CLAIMS),
        "missing_expected_claim_indexes": [],
        "valid_triples_returned": len(result.triples),
        "valid_markers_returned": len(result.markers),
        "claim_evidence": evidence,
    }
    validate_extraction_canary_report(
        report, expected_mode="required",
        expected_prompt_version=prompt_version,
    )
    return report


def run_configured_extraction_canary(
    *, api_key: str, base_url: str, model: str, thinking: str,
    prompt_version: str = ACTIVE_EXTRACTION_PROMPT_VERSION,
) -> dict[str, Any]:
    """Run one dedicated probe with the indexing client's exact constructor."""

    from hymem.contrib.openai_client import OpenAICompatibleClient

    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        thinking=thinking,
    )
    report: dict[str, Any] | None = None
    canary_error: ExtractionCanaryError | None = None
    cleanup_failures: tuple[dict[str, str], ...] = ()

    def close_configured_client() -> None:
        close = getattr(client, "close", None)
        if not callable(close):
            raise BenchmarkIntegrityError(
                "configured extraction canary client has no close operation"
            )
        close()

    try:
        report = run_extraction_canary(
            client, prompt_version=prompt_version
        )
    except ExtractionCanaryError as exc:
        report = exc.report
        canary_error = exc
    finally:
        # False is explicit attempted-but-unsuccessful evidence.  Only flip it
        # after close returns normally; a configured pass may never claim a
        # lifecycle success merely because the extraction itself succeeded.
        if report is not None:
            report["client_closed"] = False
        active_primary = sys.exc_info()[1] or canary_error
        cleanup_failures = run_cleanup_actions(
            [("resource_close", close_configured_client)],
            primary_exception=active_primary,
        )
        if report is not None and not cleanup_failures:
            report["client_closed"] = True
    if report is None:  # pragma: no cover - unexpected exceptions propagate
        raise AssertionError("configured extraction canary produced no report")
    expected_client = extraction_canary_client_policy(
        base_url=base_url, model=model, thinking=thinking,
    )
    if canary_error is not None:
        # When extraction is already the primary failure, teardown evidence is
        # attached to that exact exception by ``run_cleanup_actions``. Validate
        # the extraction report without letting unsuccessful cleanup replace
        # the more useful primary.
        validate_extraction_canary_report(
            report,
            expected_mode="failed",
            expected_client=expected_client,
            require_client_closed=not cleanup_failures,
            expected_prompt_version=prompt_version,
        )
        raise canary_error
    validate_extraction_canary_report(
        report,
        expected_mode="required",
        expected_client=expected_client,
        require_client_closed=True,
        expected_prompt_version=prompt_version,
    )
    return report


def print_extraction_canary(report: Mapping[str, Any]) -> None:
    """Concise, explicit console disclosure shared by non-strict adapters."""

    status = report.get("status")
    if status == "passed":
        print(
            "  Extraction canary: PASS "
            f"({report.get('provider_attempts')} provider attempt(s); "
            "excluded from scored memory-pipeline usage)",
            flush=True,
        )
    elif status == "not_run_no_pending":
        print(
            "  Extraction canary: not run (no pending indexing work; zero calls)",
            flush=True,
        )
    else:
        print(
            "  [NON-COMPARABLE] Extraction canary not run: "
            f"{report.get('skip_reason', status)}",
            flush=True,
        )
