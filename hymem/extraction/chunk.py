"""Bounded, completeness-aware Phase-1 chunk extraction.

Phase-1 is a publication boundary: a successful result authorizes a durable
``processed_chunks`` marker. This module distinguishes an authoritative empty
from an incomplete answer, recursively reduces dense inputs without breaking
source records, and never returns a partial branch as success.
"""
from __future__ import annotations

from hymem.contrib.implementation_identity import import_time_source_sha256

EXTRACTION_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)

import json
import logging
import re
from dataclasses import dataclass, field, replace

from hymem.extraction.jsonio import (
    is_ceiling_cut,
    loads_exact_or_fenced,
    loads_strict_json,
)
from hymem.extraction.llm import (
    LLMClient,
    LLMRequest,
    measure_provider_attempts,
)
from hymem.extraction.markers import (
    Marker,
    markers_from_list,
    normalize_combined_marker_item,
)
from hymem.extraction.prompts import (
    CHUNK_EXTRACTION_USER_TEMPLATE,
    CHUNK_OMISSION_VERIFICATION_USER_TEMPLATE,
    build_chunk_empty_verification_system,
    build_chunk_extraction_system,
    build_chunk_omission_verification_system,
)
from hymem.extraction.retry import DEFAULT_RETRY_ATTEMPTS
from hymem.extraction.triples import (
    Triple,
    normalize_combined_triple_item,
    triples_from_list,
)

log = logging.getLogger("hymem.extraction.chunk")

_CHUNK_IMPORTED_EXECUTION_GUARD = (
    is_ceiling_cut, loads_exact_or_fenced, loads_strict_json,
    LLMRequest, measure_provider_attempts,
    Marker, markers_from_list, normalize_combined_marker_item,
    build_chunk_empty_verification_system, build_chunk_extraction_system,
    build_chunk_omission_verification_system,
    Triple, normalize_combined_triple_item, triples_from_list,
)


def chunk_extraction_support_integrity() -> bool:
    return _CHUNK_IMPORTED_EXECUTION_GUARD == (
        is_ceiling_cut, loads_exact_or_fenced, loads_strict_json,
        LLMRequest, measure_provider_attempts,
        Marker, markers_from_list, normalize_combined_marker_item,
        build_chunk_empty_verification_system, build_chunk_extraction_system,
        build_chunk_omission_verification_system,
        Triple, normalize_combined_triple_item, triples_from_list,
    )


_CHUNK_EXTRACTION_INTEGRITY_FUNCTION = chunk_extraction_support_integrity


# Large payloads are partitioned before the first call, avoiding the former
# predictable 8192-token failure. These limits fail the whole chunk rather than
# silently dropping input.
_MAX_LEAF_INPUT_CHARS = 4000
_MAX_PREPARTITION_LEAVES = 32
_MAX_SPLIT_DEPTH = 8
# Recursive subdivision is hard-bounded in logical ``LLMClient.complete`` calls.
# A single full binary tree with 32 terminal leaves costs 31 internal primary
# attempts plus 32 primary/omission-verification pairs = 95 calls. Initial
# prepartition leaves may each recover further, so this global counter -- not
# the initial-leaf cap -- is the authoritative bound and atomically rejects any
# less regular shape that would require call 97.
# The shipped OpenAI-compatible client owns one explicit three-attempt retry
# layer and disables SDK retries, so each counted attempt is one HTTP attempt
# and the corresponding per-chunk request envelope is exactly 288.
MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK = 96
SHIPPED_MAX_EXTRACTION_PROVIDER_ATTEMPTS_PER_CHUNK = (
    MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK * DEFAULT_RETRY_ATTEMPTS
)
_MIN_FRAGMENT_CONTENT_CHARS = 96
# Source records are split only at deterministic semantic/structural
# boundaries.  V9 keeps fenced code, list items/continuations, heading/first-
# body pairs, and defensible colon-intro/introduced-block pairs atomic. A large
# canonical table is the one exception: it may split at a proven body-row edge,
# and a continuation carries exact, separately labelled same-source heading or
# colon prelude (when present) together with its canonical header. Canonical
# Markdown-table row provenance still descends only from a validated header +
# delimiter block. An inherited context remains authoritative through its exact
# applicability range: body rows that resemble a header + delimiter cannot
# replace it during recursion. Visual alignment alone never promotes a
# headerless pipe grid into trusted rows. A right-hand prose continuation also
# receives one bounded, exact preceding source window so a relationship that
# crosses an otherwise valid sentence/paragraph boundary remains visible. The
# repeated window is context only and never independently owns an item.
# This identifier is part of the benchmark canary policy so a scored run cannot
# silently reuse evidence produced under an older splitter.
SOURCE_RECORD_SPLIT_POLICY_VERSION = "hymem-source-semantic-split-v10"
# Right-hand canonical-table fragments retain exact original ``content`` and
# offsets. This separately labelled context carries the table's original
# header+delimiter bytes and, for an introduced table, its exact heading/colon
# prelude so row meaning is not lost or misrepresented as child evidence.
SOURCE_FRAGMENT_CONTEXT_VERSION = (
    "hymem-canonical-markdown-table-fragment-context-v2"
)
# Unlike the table header contract, this context is an exact suffix immediately
# preceding a prose continuation. Its bounded applicability on both sides of
# the cut prevents a recursively split record from accumulating unbounded
# prompt material or joining distant statements.
SOURCE_BOUNDARY_CONTEXT_VERSION = (
    "hymem-adjacent-prose-boundary-context-v1"
)
_MAX_SOURCE_BOUNDARY_CONTEXT_CHARS = 320
# Cross-record continuations need the preceding conversational turn, not just
# same-message prose. Context records never join the owned citation set. Both
# semantic bytes and encoded overhead are bounded and counted in leaf inputs.
SOURCE_CONVERSATION_CONTEXT_VERSION = "hymem-claim-context-v1"
_MAX_CONVERSATION_CONTEXT_RECORDS = 2
_MAX_CONVERSATION_CONTEXT_CHARS = 640
_MAX_CONVERSATION_CONTEXT_ENCODED_CHARS = 1400
_MAX_CONVERSATION_CONTEXT_APPLICABILITY_CHARS = 320
# A terminal primary-empty result is never authoritative until that exact
# source unit has received one direct empty-verification call.  Keeping this
# identity separate from prompt wording lets benchmark artifacts reject the
# former cue-split policy, which suppressed verification on child leaves.
CLEAN_EMPTY_RECOVERY_POLICY_VERSION = (
    "hymem-terminal-clean-empty-verification-v2"
)
_NEAR_BALANCED_BOUNDARY_CHARS = 512
_MAX_TRIPLES_PER_RESPONSE = 24
_MAX_MARKERS_PER_RESPONSE = 12
_MAX_DIAGNOSTIC_DETAILS = 32

_FAILURE_REASONS = frozenset({
    "branch_incomplete",
    "call_failure",
    "contract_failure",
    "incomplete_response",
    "input_contract_failure",
    "internal_validation_failure",
    "item_validation_failure",
    "output_limit_exceeded",
    "parse_failure",
    "resource_limit",
    "response_conflict",
    "shape_failure",
    "source_coverage_failure",
    "unspecified_failure",
})
_SAFE_DIAGNOSTIC_RE = re.compile(r"^[a-z0-9_.\[\]-]+:[a-z0-9_]+$")


# A match selects the stronger split-based second look for an input whose
# wording is likely to carry extractable claims.  Every other clean empty still
# receives one whole-unit verification pass: a language-specific regex is not
# an admissible completeness boundary.
_EXPLICIT_EXTRACTION_CUE = re.compile(
    r"\b(?:"
    r"i\s+(?:prefer|like|want|use|rely\s+on|depend\s+on|own|drive|live\s+in|play)"
    r"|(?:i\s+am|i['’]?m)\s+(?:building|working\s+on)"
    r"|we\s+(?:prefer|use|chose|adopted|rely\s+on|depend\s+on|replaced|switched)"
    r"|(?:do\s+not|don['’]?t|no\s+longer|stopped)\s+(?:use|using|want)"
    r"|(?:no|actually|nee)(?=,)|that['’]?s\s+wrong|not\s+correct|incorrect"
    r"|(?:uses|depends\s+on|prefers|rejects|avoids|replaces|deploys\s+to|"
    r"configured\s+with|requires\s+version|runs\s+on|connects\s+to)"
    r"|(?:please|always|never)\s+(?:answer|respond|write|format|use)"
    r"|ik\s+(?:prefereer|wil|gebruik|woon|rij|speel)"
    r"|(?:we|wij)\s+(?:gebruiken|kozen|vertrouwen\s+op)"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class ChunkResult:
    """One whole-chunk extraction outcome.

    Failure diagnostics are bounded field/reason codes. Raw provider output is
    intentionally absent so this object is safe to persist operationally.
    """

    triples: list[Triple] = field(default_factory=list)
    entity_type_hints: dict[str, str] = field(default_factory=dict)
    entity_property_hints: dict[str, dict[str, str]] = field(default_factory=dict)
    markers: list[Marker] = field(default_factory=list)
    failed: bool = False
    failure_reason: str | None = None
    failure_details: tuple[str, ...] = ()
    # Logical extraction calls and underlying provider attempts for this whole
    # chunk. The latter prefers invocation-scoped telemetry (safe for a shared
    # concurrent client), retains legacy serial cumulative-counter support,
    # and conservatively falls back to one attempt per logical call.
    completion_calls: int = 0
    provider_attempts: int = 0
    # Number of leaves produced by the initial, source-safe >4k prepartition.
    # This is execution evidence rather than a tuning input: callers such as
    # the scored-benchmark canary can prove they exercised the dense path
    # instead of inferring it from provider-call counts.
    initial_prepartition_leaves: int = 0
    # Exact core duplicates are harmless to normal indexing and remain
    # coalesced, but the count lets strict callers reject a provider that did
    # not follow an exact-output probe contract.
    duplicate_triples_collapsed: int = 0


@dataclass(frozen=True)
class _ExtractionUnit:
    text: str
    source_records: tuple[tuple[int, str], ...] | None = None
    # Canonical-table row boundaries relative to ``text`` for legacy units or
    # to the sole source record's semantic ``content``.  They are discovered
    # only while the header + delimiter are present, then translated into each
    # child so a recursively split continuation never needs to be reclassified
    # as a generic headerless grid.
    trusted_table_boundaries: tuple[int, ...] = ()
    # Exact, separately labelled preceding source slices for interpretation;
    # these are deliberately absent from source_records and allowed_ids.
    context_records: tuple[tuple[int, str], ...] = ()

    @property
    def allowed_ids(self) -> frozenset[int] | None:
        if self.source_records is None:
            return None
        return frozenset(mid for mid, _record in self.source_records)


@dataclass
class _CallBudget:
    completion_calls: int = 0
    provider_attempts: int = 0


def _bounded_details(*groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    unique: list[str] = []
    for group in groups:
        for detail in group:
            if detail not in unique:
                unique.append(detail)
            if len(unique) == _MAX_DIAGNOSTIC_DETAILS:
                return tuple([*unique[:-1], "diagnostics:truncated"])
    return tuple(unique)


def _failure(reason: str, *details: str) -> ChunkResult:
    return ChunkResult(
        failed=True,
        failure_reason=reason,
        failure_details=_bounded_details(list(details)),
    )


def safe_failure_diagnostics(
    reason: str | None, details: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """Return bounded allowlisted diagnostics suitable for durable storage."""
    safe_reason = reason if reason in _FAILURE_REASONS else "unspecified_failure"
    safe_details: list[str] = []
    for detail in details[:_MAX_DIAGNOSTIC_DETAILS]:
        if (
            isinstance(detail, str)
            and len(detail) <= 160
            and _SAFE_DIAGNOSTIC_RE.fullmatch(detail)
        ):
            safe_details.append(detail)
        else:
            safe_details.append("diagnostic:invalid")
    return safe_reason, _bounded_details(safe_details)


def _chunk_max_tokens(text: str) -> int:
    """Bound a leaf at 1536--4096 output tokens.

    Explicit item and completion contracts make the old 8192-token ceiling
    unnecessary: a leaf either fits or returns the small incomplete signal.
    """
    return min(4096, max(1536, 1024 + len(text)))


def _normalized_identity(value: str) -> str:
    return " ".join(value.casefold().split())


def _merge_duplicate_triples(first: Triple, second: Triple) -> Triple:
    """Merge equal core claims, retaining only unambiguous optional values."""
    winner = first if repr(first) <= repr(second) else second

    def agreed(field_name: str):
        left = getattr(first, field_name)
        right = getattr(second, field_name)
        if left is None:
            return right
        if right is None or left == right:
            return left
        return None

    return replace(
        winner,
        value_text=agreed("value_text"),
        value_numeric=agreed("value_numeric"),
        value_unit=agreed("value_unit"),
        temporal_scope=agreed("temporal_scope"),
    )


def _merge_type_hints(
    left: dict[str, str], right: dict[str, str]
) -> dict[str, str]:
    candidates: dict[str, tuple[str, set[str]]] = {}
    for source in (left, right):
        for entity, entity_type in source.items():
            identity = _normalized_identity(entity)
            display, values = candidates.setdefault(identity, (entity, set()))
            if entity < display:
                display = entity
                candidates[identity] = (display, values)
            values.add(entity_type)
    return {
        display: next(iter(values))
        for _identity, (display, values) in sorted(candidates.items())
        if len(values) == 1
    }


def _merge_property_hints(
    left: dict[str, dict[str, str]], right: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    candidates: dict[str, tuple[str, dict[str, tuple[str, set[str]]]]] = {}
    for source in (left, right):
        for entity, properties in source.items():
            identity = _normalized_identity(entity)
            display, bucket = candidates.setdefault(identity, (entity, {}))
            if entity < display:
                display = entity
                candidates[identity] = (display, bucket)
            for key, value in properties.items():
                key_identity = key.casefold().strip()
                key_display, values = bucket.setdefault(key_identity, (key, set()))
                if key < key_display:
                    key_display = key
                    bucket[key_identity] = (key_display, values)
                values.add(value)
    merged: dict[str, dict[str, str]] = {}
    for _identity, (display, bucket) in sorted(candidates.items()):
        properties = {
            key_display: next(iter(values))
            for _key_identity, (key_display, values) in sorted(bucket.items())
            if len(values) == 1
        }
        if properties:
            merged[display] = properties
    return merged


def _merge_results(a: ChunkResult, b: ChunkResult) -> ChunkResult:
    """Merge two disjoint/overlapping branches atomically."""
    polarities: dict[tuple[str, str, str, int | None], int] = {}
    unique: dict[tuple[str, str, str, int, int | None], Triple] = {}
    conflicts: list[str] = []
    duplicate_triples_collapsed = (
        a.duplicate_triples_collapsed + b.duplicate_triples_collapsed
    )
    for triple in [*a.triples, *b.triples]:
        claim = (
            _normalized_identity(triple.subject),
            triple.predicate,
            _normalized_identity(triple.object),
            triple.source_message_id,
        )
        prior_polarity = polarities.get(claim)
        if prior_polarity is not None and prior_polarity != triple.polarity:
            conflicts.append("triples:polarity_conflict")
            continue
        polarities[claim] = triple.polarity
        identity = (*claim[:3], triple.polarity, triple.source_message_id)
        prior = unique.get(identity)
        if prior is not None:
            duplicate_triples_collapsed += 1
        unique[identity] = (
            triple if prior is None else _merge_duplicate_triples(prior, triple)
        )

    markers: dict[tuple[str, str], Marker] = {}
    for marker in [*a.markers, *b.markers]:
        identity = (marker.kind, _normalized_identity(marker.statement))
        prior = markers.get(identity)
        if prior is None or repr(marker) < repr(prior):
            markers[identity] = marker

    failed = a.failed or b.failed or bool(conflicts)
    reason = (
        "response_conflict" if conflicts
        else "branch_incomplete" if a.failed or b.failed
        else None
    )
    branch_details: list[str] = []
    if a.failed:
        branch_details.append(f"left:{a.failure_reason or 'unspecified_failure'}")
        branch_details.extend(f"left.{detail}" for detail in a.failure_details)
    if b.failed:
        branch_details.append(f"right:{b.failure_reason or 'unspecified_failure'}")
        branch_details.extend(f"right.{detail}" for detail in b.failure_details)
    return ChunkResult(
        triples=list(unique.values()),
        entity_type_hints=_merge_type_hints(
            a.entity_type_hints, b.entity_type_hints
        ),
        entity_property_hints=_merge_property_hints(
            a.entity_property_hints, b.entity_property_hints
        ),
        markers=list(markers.values()),
        failed=failed,
        failure_reason=reason,
        failure_details=_bounded_details(branch_details, conflicts),
        duplicate_triples_collapsed=duplicate_triples_collapsed,
    )


def _failed_split_after_left(left: ChunkResult) -> ChunkResult:
    """Fail the whole split without evaluating or exposing its right branch.

    Once the left child fails, atomic whole-unit publication is impossible.
    Preserve the same parent-level ``branch_incomplete`` shape that a full
    two-child merge would have produced, but discard every partial value and
    avoid inventing diagnostics for a right child that was never attempted.
    """

    reason, details = safe_failure_diagnostics(
        left.failure_reason, left.failure_details
    )
    return ChunkResult(
        failed=True,
        failure_reason="branch_incomplete",
        failure_details=_bounded_details(
            [f"left:{reason}"],
            [f"left.{detail}" for detail in details],
        ),
    )


_SOURCE_FRAGMENT_CONTEXT_BASE_FIELDS = frozenset({
    "version",
    "kind",
    "content",
    "source_content_start",
    "source_content_end",
    "applies_through_source_content_end",
})
_SOURCE_FRAGMENT_CONTEXT_PRELUDE_FIELDS = frozenset({
    "prelude_kind",
    "prelude_content",
    "prelude_source_content_start",
    "prelude_source_content_end",
})
_SOURCE_FRAGMENT_CONTEXT_FIELDS = (
    _SOURCE_FRAGMENT_CONTEXT_BASE_FIELDS
    | _SOURCE_FRAGMENT_CONTEXT_PRELUDE_FIELDS
)
_SOURCE_FRAGMENT_PRELUDE_KINDS = frozenset({
    "atx_heading",
    "setext_heading",
    "colon_led_paragraph",
})
_SOURCE_BOUNDARY_CONTEXT_FIELDS = frozenset({
    "version",
    "kind",
    "content",
    "source_content_start",
    "source_content_end",
    "applies_through_source_content_end",
})
_CONTEXT_UNSET = object()


def _valid_source_fragment_prelude(kind: object, content: object) -> bool:
    """Validate the conservative prelude subset encoded by the splitter.

    Offsets and public provenance are checked by the surrounding context
    validator. This shape check makes the label meaningful even for private
    fragment helpers: arbitrary prose cannot claim heading/colon authority.
    """

    if kind not in _SOURCE_FRAGMENT_PRELUDE_KINDS or not isinstance(content, str):
        return False
    lines = _markdown_lines(content)
    if (
        not content
        or not content.endswith("\n")
        or not lines
        or lines[0].start != 0
        or lines[-1].end != len(content)
        or any(not _has_supported_markdown_line_ending(content, line) for line in lines)
    ):
        return False
    if kind == "atx_heading":
        return len(lines) == 1 and _is_atx_heading(lines[0].text)
    if kind == "setext_heading":
        if (
            len(lines) < 2
            or _MARKDOWN_SETEXT_UNDERLINE_RE.fullmatch(lines[-1].text) is None
            or lines[-1].text.strip(" \t") == "-"
        ):
            return False
        return _heading_line_ranges(lines, ()) == ((0, len(lines)),)
    return (
        all(line.text.strip(" \t") for line in lines)
        and lines[-1].text.rstrip(" \t").endswith(":")
        and all(
            not _is_atx_heading(line.text)
            and _list_marker_indent(line.text) is None
            and _strict_markdown_table_cells(line.text) is None
            and _MARKDOWN_SETEXT_UNDERLINE_RE.fullmatch(line.text) is None
            and not line.text.lstrip(" \t").startswith((">", "<"))
            for line in lines
        )
    )


def _valid_source_fragment_context(
    value: object, *, fragment_start: int,
) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    kind = value.get("kind")
    if kind == "canonical_markdown_table_header":
        if keys != _SOURCE_FRAGMENT_CONTEXT_BASE_FIELDS:
            return False
    elif kind == "introduced_canonical_markdown_table_header":
        if keys != _SOURCE_FRAGMENT_CONTEXT_FIELDS:
            return False
    else:
        return False
    start = value.get("source_content_start")
    end = value.get("source_content_end")
    applies_through = value.get("applies_through_source_content_end")
    content = value.get("content")
    base_valid = (
        value.get("version") == SOURCE_FRAGMENT_CONTEXT_VERSION
        and isinstance(content, str)
        and bool(content)
        and not isinstance(start, bool)
        and isinstance(start, int)
        and not isinstance(end, bool)
        and isinstance(end, int)
        and not isinstance(applies_through, bool)
        and isinstance(applies_through, int)
        and 0 <= start < end == start + len(content)
        and end <= fragment_start < applies_through
        and _is_canonical_table_header_context(content)
    )
    if not base_valid or kind == "canonical_markdown_table_header":
        return base_valid

    prelude_start = value.get("prelude_source_content_start")
    prelude_end = value.get("prelude_source_content_end")
    prelude_content = value.get("prelude_content")
    return (
        not isinstance(prelude_start, bool)
        and isinstance(prelude_start, int)
        and not isinstance(prelude_end, bool)
        and isinstance(prelude_end, int)
        and isinstance(prelude_content, str)
        and 0 <= prelude_start < prelude_end
        and prelude_end == prelude_start + len(prelude_content)
        and prelude_end <= start
        and _valid_source_fragment_prelude(
            value.get("prelude_kind"), prelude_content
        )
    )


def _valid_source_boundary_context(
    value: object,
    *,
    fragment_content: str,
    fragment_start: int,
) -> bool:
    """Validate one deterministic preceding-source context window.

    Public callers cannot submit fragments at all; this validator additionally
    pins the private representation to the full bounded suffix, an exact
    adjacent offset, a proven prose boundary, and a bounded right-side
    applicability range.
    """

    if not isinstance(value, dict) or set(value) != _SOURCE_BOUNDARY_CONTEXT_FIELDS:
        return False
    start = value.get("source_content_start")
    end = value.get("source_content_end")
    applies_through = value.get("applies_through_source_content_end")
    content = value.get("content")
    if not (
        value.get("version") == SOURCE_BOUNDARY_CONTEXT_VERSION
        and value.get("kind") == "preceding_adjacent_prose"
        and isinstance(content, str)
        and bool(content)
        and not isinstance(start, bool)
        and isinstance(start, int)
        and not isinstance(end, bool)
        and isinstance(end, int)
        and not isinstance(applies_through, bool)
        and isinstance(applies_through, int)
        and 0 <= start < end == fragment_start
        and start == max(0, end - _MAX_SOURCE_BOUNDARY_CONTEXT_CHARS)
        and len(content) == end - start
        and fragment_start < applies_through
        <= fragment_start + _MAX_SOURCE_BOUNDARY_CONTEXT_CHARS
    ):
        return False

    # Validate the claimed boundary against exact context plus the available
    # authoritative prefix. Sentence validation needs the first character on
    # the right to distinguish terminal punctuation from abbreviations/URLs.
    combined = content + fragment_content[:_MAX_SOURCE_BOUNDARY_CONTEXT_CHARS]
    cut = len(content)
    return (
        cut in {
            match.end() for match in _PARAGRAPH_BOUNDARY_RE.finditer(combined)
        }
        or cut in _sentence_boundary_points(combined)
    )


def _source_payload(record: tuple[int, str]) -> dict | None:
    message_id, encoded = record
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id < 1:
        return None
    try:
        payload = loads_strict_json(encoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if any(key in payload for key in (
        "source_context_only", "context_for_source_message_id",
        "applies_through_source_content_end",
    )):
        return None
    payload_message_id = payload.get("source_message_id")
    if (
        isinstance(payload_message_id, bool)
        or not isinstance(payload_message_id, int)
        or payload_message_id != message_id
        or not isinstance(payload.get("content"), str)
    ):
        return None
    record_version = payload.get("source_record_version")
    if record_version == "hymem-claim-source-fragment-v2":
        start = payload.get("source_content_start")
        end = payload.get("source_content_end")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end != start + len(payload["content"])
        ):
            return None
        context = payload.get("source_fragment_context")
        if context is not None and not _valid_source_fragment_context(
            context, fragment_start=start
        ):
            return None
        boundary_context = payload.get("source_boundary_context")
        if boundary_context is not None and not _valid_source_boundary_context(
            boundary_context,
            fragment_content=payload["content"],
            fragment_start=start,
        ):
            return None
    elif record_version != "hymem-claim-source-v2":
        return None
    elif (
        "source_fragment_context" in payload
        or "source_boundary_context" in payload
    ):
        # Only the splitter may add context, and only to an offset-bearing
        # fragment. A caller-supplied full source cannot smuggle extra evidence.
        return None
    return payload


def _fragment_record(
    record: tuple[int, str],
    *,
    start: int,
    end: int,
    source_fragment_context: dict | None = None,
    source_boundary_context: dict | None | object = _CONTEXT_UNSET,
) -> tuple[int, str] | None:
    payload = _source_payload(record)
    if payload is None:
        return None
    content = payload["content"]
    base_start = payload.get("source_content_start", 0)
    if isinstance(base_start, bool) or not isinstance(base_start, int) or base_start < 0:
        return None
    if not (0 <= start < end <= len(content)):
        return None
    fragment = dict(payload)
    fragment["content"] = content[start:end]
    fragment["source_record_version"] = "hymem-claim-source-fragment-v2"
    absolute_start = base_start + start
    fragment["source_content_start"] = absolute_start
    fragment["source_content_end"] = base_start + end
    context = (
        source_fragment_context
        if source_fragment_context is not None
        else payload.get("source_fragment_context")
    )
    if (
        source_fragment_context is not None
        and not _valid_source_fragment_context(
            source_fragment_context, fragment_start=absolute_start
        )
    ):
        # A locally proven table boundary must never degrade into a headerless
        # child merely because context construction and validation diverged.
        return None
    if _valid_source_fragment_context(context, fragment_start=absolute_start):
        fragment["source_fragment_context"] = context
    else:
        fragment.pop("source_fragment_context", None)

    boundary_context = (
        payload.get("source_boundary_context")
        if source_boundary_context is _CONTEXT_UNSET
        else source_boundary_context
    )
    if (
        source_boundary_context is not _CONTEXT_UNSET
        and source_boundary_context is not None
        and not _valid_source_boundary_context(
            source_boundary_context,
            fragment_content=fragment["content"],
            fragment_start=absolute_start,
        )
    ):
        return None
    if _valid_source_boundary_context(
        boundary_context,
        fragment_content=fragment["content"],
        fragment_start=absolute_start,
    ):
        fragment["source_boundary_context"] = boundary_context
    else:
        fragment.pop("source_boundary_context", None)
    if "source_fragment_context" in fragment or "source_boundary_context" in fragment:
        # Present the validated preceding context before its continuation.
        # Only top-level owned content moves: every nested context/metadata
        # value retains its canonical encoding, and context-free fragments
        # retain their original byte representation below.
        keys = sorted(key for key in fragment if key != "content") + ["content"]
        fields = [
            json.dumps(key, ensure_ascii=False) + ":" + json.dumps(
                fragment[key],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for key in keys
        ]
        return record[0], "{" + ",".join(fields) + "}"
    return record[0], json.dumps(
        fragment,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


_PARAGRAPH_BOUNDARY_RE = re.compile(r"(?:\r?\n)[ \t]*(?:\r?\n)+")
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?:"
    r"[.!?](?:[\"'\)\]\}\u2019\u201d]*)(?:[ \t]+|\r?\n)"
    r"|[\u3002\uff01\uff1f](?:[\"'\)\]\}\u2019\u201d]*)[ \t]*"
    r")"
)
_MARKDOWN_TABLE_DELIMITER_CELL_RE = re.compile(r"^:?-{3,}:?$")
_MARKDOWN_LIST_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:[-+*]|[0-9]{1,9}[.)])(?:[ \t]+.*|)$"
)
_MARKDOWN_SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")


@dataclass(frozen=True)
class _MarkdownLine:
    """One exact source line with its terminator retained by the offsets."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _MarkdownTableBlock:
    start_line: int
    end_line: int
    header_end: int
    table_end: int
    row_boundaries: tuple[int, ...]


@dataclass(frozen=True)
class _MarkdownTablePrelude:
    """Exact heading/colon source span that semantically introduces a table."""

    table_start_line: int
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class _MarkdownListBlock:
    start_line: int
    end_line: int
    item_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _MarkdownBlockAnalysis:
    protected_spans: tuple[tuple[int, int], ...]
    structural_boundaries: tuple[int, ...]
    # Proven table-row cuts that may cross heading/colon relationship
    # protection because the splitter can carry both parts as labelled context.
    contextual_table_boundaries: tuple[int, ...]


def _markdown_lines(text: str) -> tuple[_MarkdownLine, ...]:
    """Return LF/CRLF-neutral line metadata without normalizing source text."""

    lines: list[_MarkdownLine] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        if raw_line.endswith("\r\n"):
            content = raw_line[:-2]
        elif raw_line.endswith(("\n", "\r")):
            content = raw_line[:-1]
        else:
            content = raw_line
        end = offset + len(raw_line)
        lines.append(_MarkdownLine(content, offset, end))
        offset = end
    return tuple(lines)


def _leading_indent_width(line: str) -> int:
    """Return a conservative visual width for a Markdown indentation prefix."""

    width = 0
    for char in line:
        if char == " ":
            width += 1
        elif char == "\t":
            width += 4 - (width % 4)
        else:
            break
    return width


def _fence_opener(line: str) -> tuple[str, int] | None:
    """Recognize the unambiguous CommonMark backtick/tilde fence subset."""

    leading = len(line) - len(line.lstrip(" "))
    if leading > 3:
        return None
    remainder = line[leading:]
    if not remainder or remainder[0] not in "`~":
        return None
    fence_char = remainder[0]
    run = len(remainder) - len(remainder.lstrip(fence_char))
    if run < 3:
        return None
    info = remainder[run:]
    if fence_char == "`" and "`" in info:
        return None
    return fence_char, run


def _fence_closer(line: str, *, fence_char: str, minimum: int) -> bool:
    leading = len(line) - len(line.lstrip(" "))
    if leading > 3:
        return False
    remainder = line[leading:]
    run = len(remainder) - len(remainder.lstrip(fence_char))
    return run >= minimum and remainder[run:].strip(" \t") == ""


def _fenced_code_spans(
    text: str, lines: tuple[_MarkdownLine, ...] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return atomic fence spans; an unmatched opener protects through EOF."""

    source_lines = _markdown_lines(text) if lines is None else lines
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(source_lines):
        opener = _fence_opener(source_lines[index].text)
        if opener is None:
            index += 1
            continue
        fence_char, minimum = opener
        start = source_lines[index].start
        index += 1
        while index < len(source_lines) and not _fence_closer(
            source_lines[index].text,
            fence_char=fence_char,
            minimum=minimum,
        ):
            index += 1
        if index == len(source_lines):
            spans.append((start, len(text)))
            break
        spans.append((start, source_lines[index].end))
        index += 1
    return tuple(spans)


def _normalized_protected_spans(
    spans: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Sort and union overlaps while preserving safe boundaries at adjacency."""

    merged: list[tuple[int, int]] = []
    for start, end in sorted(set(spans)):
        if start >= end:
            continue
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _last_span_starting_at_or_before(
    offset: int, spans: tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    """Binary-search normalized, non-overlapping spans by their start."""

    lower = 0
    upper = len(spans)
    while lower < upper:
        middle = (lower + upper) // 2
        if spans[middle][0] <= offset:
            lower = middle + 1
        else:
            upper = middle
    return spans[lower - 1] if lower else None


def _offset_in_spans(
    offset: int, spans: tuple[tuple[int, int], ...],
) -> bool:
    span = _last_span_starting_at_or_before(offset, spans)
    return span is not None and span[0] <= offset < span[1]


def _cut_inside_spans(
    cut: int, spans: tuple[tuple[int, int], ...],
) -> bool:
    # Exact block edges remain useful boundaries; only an interior cut can
    # detach or bisect the protected construct.
    span = _last_span_starting_at_or_before(cut, spans)
    return span is not None and span[0] < cut < span[1]

# A period followed by whitespace is not sufficient evidence of a sentence
# boundary.  In particular, titles/initials, dotted abbreviations and
# human-formatted decimal/version components all satisfy that lexical shape.
# Splitting there can place the two halves of one assertion in disjoint source
# fragments and let independently clean-empty replies certify a false empty.
#
# This list is intentionally conservative.  Rejecting an ambiguous boundary
# merely makes an oversized unit try another boundary or fail held/unprocessed;
# accepting a false boundary can permanently erase its claim at the Phase-1
# publication boundary.
_AMBIGUOUS_PERIOD_TOKENS = frozenset({
    "adj", "adm", "approx", "apr", "assn", "aug", "ave", "bros",
    "capt", "cf", "ch", "cmdr", "co", "col", "corp", "dec", "dept",
    "dr", "e.g", "ed", "eds", "eq", "esp", "est", "etc", "feb",
    "fig", "figs", "fri", "gen", "gov", "hon", "i.e", "inc", "jan",
    "jr", "jul", "jun", "lat", "ltd", "lt", "mar", "max", "min",
    "misc", "mon", "mr", "mrs", "ms", "mt", "no", "nos", "nov",
    "oct", "p.m", "ph.d", "pp", "prof", "ref", "refs", "rep", "rev",
    "sat", "sec", "sen", "sep", "sept", "sgt", "sr", "st", "sun",
    "thu", "tue", "u.s", "ver", "viz", "vol", "vols", "vs", "wed",
})
_DOTTED_TOKEN_BEFORE_PERIOD_RE = re.compile(
    r"(?:\b[^\W\d_]{1,4}\.)+[^\W\d_]{1,4}$", re.UNICODE
)
_OPENING_SENTENCE_DELIMITERS = frozenset("\"'([{\u2018\u201c")
_CLOSING_SENTENCE_DELIMITERS = frozenset("\"')]}\u2019\u201d")


def _next_nonspace_after_boundary(text: str, cut: int) -> str | None:
    """Return the first content character after boundary whitespace/quotes."""

    index = cut
    while index < len(text) and (
        text[index].isspace() or text[index] in _OPENING_SENTENCE_DELIMITERS
    ):
        index += 1
    return text[index] if index < len(text) else None


def _period_is_safe_sentence_boundary(
    text: str, *, punctuation_index: int, cut: int
) -> bool:
    """Conservatively distinguish a terminal period from internal punctuation.

    This is a safety predicate, not a general sentence tokenizer.  Ambiguity is
    resolved toward holding an oversized source for retry/manual repair rather
    than authorizing independently empty fragments.
    """

    if punctuation_index <= 0:
        return False
    following = _next_nonspace_after_boundary(text, cut)
    if following is None:
        return False
    if following == "|" and not any(
        newline in text[punctuation_index + 1:cut]
        for newline in ("\r", "\n")
    ):
        # A terminal-looking cell value is not the end of its source row.
        return False

    lexical_end = punctuation_index
    while (
        lexical_end > 0
        and text[lexical_end - 1] in _CLOSING_SENTENCE_DELIMITERS
    ):
        lexical_end -= 1
    if lexical_end == 0:
        return False
    previous = text[lexical_end - 1]
    if previous == ".":  # ellipsis or a dotted token's preceding component
        return False
    if previous.isdigit() and following.isdigit():
        return False

    token_start = lexical_end
    while token_start > 0 and (
        text[token_start - 1].isalpha()
        or text[token_start - 1] == "."
    ):
        token_start -= 1
    lexical_token = text[token_start:lexical_end]
    token = lexical_token.casefold().strip(".")
    if (
        (not token and not previous.isdigit())
        or len(token) == 1
        or token in _AMBIGUOUS_PERIOD_TOKENS
        or _DOTTED_TOKEN_BEFORE_PERIOD_RE.fullmatch(lexical_token) is not None
    ):
        return False

    # A short unknown token followed by lower-case prose still has the lexical
    # shape of an abbreviation.  Longer ordinary words remain admissible so
    # chat-style lower-case sentence starts and repeated prose can still be
    # partitioned; known longer abbreviations were rejected by the table above.
    if following.islower() and len(token) <= 4:
        return False
    return True


def _question_or_exclamation_is_safe_sentence_boundary(
    text: str, *, punctuation_index: int, cut: int
) -> bool:
    """Reject ``?``/``!`` shapes that commonly occur inside one assertion."""

    following = _next_nonspace_after_boundary(text, cut)
    if following is None or following.islower():
        return False
    if following == "|" and not any(
        newline in text[punctuation_index + 1:cut]
        for newline in ("\r", "\n")
    ):
        return False

    # A terminal immediately enclosed by a quote/bracket may be a label or a
    # quoted fragment rather than the end of the containing sentence.  Holding
    # it is safer than certifying the two sides independently empty.
    suffix = text[punctuation_index + 1:cut]
    if any(char in _CLOSING_SENTENCE_DELIMITERS for char in suffix):
        return False

    token_start = punctuation_index
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    lexical_context = text[token_start:punctuation_index]
    if any(char in _OPENING_SENTENCE_DELIMITERS for char in lexical_context):
        return False

    # Query delimiters in a path/URL are punctuation, not sentence terminals.
    if any(marker in lexical_context for marker in ("/", "=", "&", "#")):
        return False
    return True


def _sentence_boundary_points(text: str) -> list[int]:
    """Return only sentence terminals safe enough to fragment durably."""

    table_row_spans: list[tuple[int, int]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        if _strict_markdown_table_cells(line) is not None:
            table_row_spans.append((offset, offset + len(line)))
        offset += len(raw_line)

    points: list[int] = []
    row_index = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        punctuation_index = match.start()
        while (
            row_index < len(table_row_spans)
            and table_row_spans[row_index][1] <= punctuation_index
        ):
            row_index += 1
        if (
            row_index < len(table_row_spans)
            and table_row_spans[row_index][0]
            <= punctuation_index
            < table_row_spans[row_index][1]
        ):
            # Even a linguistically safe terminal is not a source-row
            # boundary.  Table-row cuts are considered separately and must
            # retain every cell (including trailing same-cell text) intact.
            continue
        punctuation = text[punctuation_index]
        if punctuation == "." and not _period_is_safe_sentence_boundary(
            text, punctuation_index=punctuation_index, cut=match.end()
        ):
            continue
        if punctuation in "?!" and not (
            _question_or_exclamation_is_safe_sentence_boundary(
                text, punctuation_index=punctuation_index, cut=match.end()
            )
        ):
            continue
        points.append(match.end())
    return points


def _is_prose_boundary_cut(text: str, cut: int) -> bool:
    """Whether ``cut`` is one of the splitter's proven prose boundaries."""

    return (
        cut in {match.end() for match in _PARAGRAPH_BOUNDARY_RE.finditer(text)}
        or cut in _sentence_boundary_points(text)
    )


def _source_boundary_context_for_cut(
    payload: dict,
    *,
    content: str,
    cut: int,
    base_start: int,
) -> dict | None:
    """Build the exact bounded suffix preceding a right-hand prose child.

    A deeply recursive cut can occur within the bounded prefix of an
    already-fragmented parent. In that case the parent's validated preceding
    context supplies the missing prefix; otherwise the cut is held rather than
    manufacturing a shorter or discontinuous window.
    """

    if not (0 < cut < len(content)) or not _is_prose_boundary_cut(content, cut):
        return None
    absolute_cut = base_start + cut
    available_start = base_start
    available = content[:cut]
    inherited = payload.get("source_boundary_context")
    if _valid_source_boundary_context(
        inherited,
        fragment_content=content,
        fragment_start=base_start,
    ):
        available_start = inherited["source_content_start"]
        available = inherited["content"] + available

    context_start = max(0, absolute_cut - _MAX_SOURCE_BOUNDARY_CONTEXT_CHARS)
    if context_start < available_start:
        return None
    relative_start = context_start - available_start
    context_content = available[relative_start:]
    if len(context_content) != absolute_cut - context_start:
        return None
    context = {
        "version": SOURCE_BOUNDARY_CONTEXT_VERSION,
        "kind": "preceding_adjacent_prose",
        "content": context_content,
        "source_content_start": context_start,
        "source_content_end": absolute_cut,
        "applies_through_source_content_end": absolute_cut + min(
            _MAX_SOURCE_BOUNDARY_CONTEXT_CHARS,
            len(content) - cut,
        ),
    }
    if not _valid_source_boundary_context(
        context,
        fragment_content=content[cut:],
        fragment_start=absolute_cut,
    ):
        return None
    return context


def _strict_markdown_table_cells(line: str) -> tuple[str, ...] | None:
    """Parse only the unambiguous, outer-pipe Markdown row subset.

    Markdown permits several compact/escaped forms whose apparent pipes may be
    prose or inline-code content.  They are deliberately not accepted here:
    declining an uncertain split holds the source for retry, whereas treating
    an arbitrary soft newline as structural could certify two incomplete
    fragments independently.  Empty body cells remain valid and harmless.
    """

    stripped = line.strip(" \t")
    if (
        not stripped.startswith("|")
        or not stripped.endswith("|")
        or stripped.count("|") < 2
        or "\\|" in stripped
        or "`" in stripped
    ):
        return None
    return tuple(cell.strip() for cell in stripped[1:-1].split("|"))


def _is_canonical_table_header_context(content: str) -> bool:
    """Require exactly one canonical header and delimiter, with exact EOLs."""

    lines = _markdown_lines(content)
    if (
        len(lines) != 2
        or lines[0].start != 0
        or lines[1].end != len(content)
        or not content.endswith("\n")
    ):
        return False
    header = _strict_markdown_table_cells(lines[0].text)
    delimiter = _strict_markdown_table_cells(lines[1].text)
    return (
        header is not None
        and delimiter is not None
        and bool(header)
        and all(header)
        and len(delimiter) == len(header)
        and all(
            _MARKDOWN_TABLE_DELIMITER_CELL_RE.fullmatch(cell) is not None
            for cell in delimiter
        )
    )


def _has_supported_markdown_line_ending(
    text: str, line: _MarkdownLine,
) -> bool:
    """Accept exact LF/CRLF (or no final EOL), never a lone-CR separator."""

    raw_line = text[line.start:line.end]
    return raw_line.endswith("\n") or not raw_line.endswith("\r")


def _markdown_table_blocks(
    text: str,
    *,
    lines: tuple[_MarkdownLine, ...] | None = None,
    fenced_spans: tuple[tuple[int, int], ...] | None = None,
) -> tuple[_MarkdownTableBlock, ...]:
    """Classify canonical tables while treating fenced source as opaque."""

    source_lines = _markdown_lines(text) if lines is None else lines
    fences = (
        _fenced_code_spans(text, source_lines)
        if fenced_spans is None else fenced_spans
    )
    blocks: list[_MarkdownTableBlock] = []
    index = 0
    while index + 2 < len(source_lines):
        if _offset_in_spans(source_lines[index].start, fences):
            index += 1
            continue
        header = _strict_markdown_table_cells(source_lines[index].text)
        delimiter = _strict_markdown_table_cells(source_lines[index + 1].text)
        if (
            header is None
            or delimiter is None
            or _offset_in_spans(source_lines[index + 1].start, fences)
            or not _has_supported_markdown_line_ending(
                text, source_lines[index]
            )
            or not _has_supported_markdown_line_ending(
                text, source_lines[index + 1]
            )
            or not header
            or any(not cell for cell in header)
            or len(delimiter) != len(header)
            or any(
                _MARKDOWN_TABLE_DELIMITER_CELL_RE.fullmatch(cell) is None
                for cell in delimiter
            )
        ):
            index += 1
            continue

        end = index + 2
        unsupported_eol = False
        while end < len(source_lines):
            if _offset_in_spans(source_lines[end].start, fences):
                break
            body = _strict_markdown_table_cells(source_lines[end].text)
            if body is None or len(body) != len(header):
                break
            if not _has_supported_markdown_line_ending(
                text, source_lines[end]
            ):
                unsupported_eol = True
                break
            end += 1
        if (
            unsupported_eol
            or end == index + 2
        ):  # header + delimiter without a body is not enough
            index += 1
            continue

        # Every selected cut follows a body row proven to belong to this
        # complete block.  Never detach a header from its delimiter/first body,
        # and exclude the final row because it would not partition the block.
        # A large right-side continuation is handled by provenance carried on
        # the extraction unit; it is never reclassified from appearance alone.
        blocks.append(_MarkdownTableBlock(
            start_line=index,
            end_line=end,
            header_end=source_lines[index + 1].end,
            table_end=source_lines[end - 1].end,
            row_boundaries=tuple(
                source_lines[row].end for row in range(index + 2, end - 1)
            ),
        ))
        index = end
    return tuple(blocks)


def _markdown_table_boundary_points(text: str) -> list[int]:
    """Return cuts proven by canonical, non-fenced Markdown table structure.

    A table block needs a non-empty header, the same number of canonical
    delimiter cells, and at least one column-consistent body row.  Boundaries
    are then safe because Markdown rows cannot continue across them.  Exact
    line endings are included in the left fragment, so concatenating the two
    slices reconstructs the source byte-for-byte (at Python string granularity).

    Headerless pipe grids are deliberately not admitted.  Same-width non-empty
    cells do not prove record independence: adjacent rows can be soft-wrapped
    prose whose relationship crosses the newline.  Table-looking lines inside
    a valid or unmatched code fence are source text, never structural proof.
    """

    return sorted({
        point
        for block in _markdown_table_blocks(text)
        for point in block.row_boundaries
    })


def _canonical_table_context_for_cut(
    text: str,
    *,
    cut: int,
    base_start: int,
    inherited_context: dict | None = None,
    local_discovery_start: int = 0,
) -> dict | None:
    """Build exact, labelled header context for a proven table-row cut.

    A fragment may begin in a table whose canonical header is no longer in its
    ``content``. While that inherited context applies, it is immutable: a pair
    of body rows that happens to look like a header and delimiter cannot be
    promoted during recursive descent. Once the inherited table ends, local
    discovery resumes on the untouched suffix so genuinely later tables can
    establish their own context.
    """

    if inherited_context is not None and cut < local_discovery_start:
        return dict(inherited_context)

    suffix = text[local_discovery_start:]
    lines = _markdown_lines(suffix)
    fenced_spans = _fenced_code_spans(suffix, lines)
    list_blocks = _list_blocks(lines, fenced_spans)
    opaque_spans = _normalized_protected_spans([
        *fenced_spans,
        *(span for block in list_blocks for span in block.item_spans),
    ])
    heading_ranges = _heading_line_ranges(lines, opaque_spans)
    table_blocks = _markdown_table_blocks(
        suffix,
        lines=lines,
        fenced_spans=fenced_spans,
    )
    preludes = {
        prelude.table_start_line: prelude
        for prelude in _introduced_table_preludes(
            suffix,
            lines=lines,
            opaque_spans=opaque_spans,
            heading_ranges=heading_ranges,
            table_blocks=table_blocks,
        )
    }
    local_cut = cut - local_discovery_start
    for block in table_blocks:
        if local_cut not in block.row_boundaries:
            continue
        local_header_start = lines[block.start_line].start
        header_start = local_discovery_start + local_header_start
        header_end = local_discovery_start + block.header_end
        table_end = local_discovery_start + block.table_end
        content = text[header_start:header_end]
        context = {
            "version": SOURCE_FRAGMENT_CONTEXT_VERSION,
            "kind": "canonical_markdown_table_header",
            "content": content,
            "source_content_start": base_start + header_start,
            "source_content_end": base_start + header_end,
            "applies_through_source_content_end": base_start + table_end,
        }
        prelude = preludes.get(block.start_line)
        if prelude is not None:
            context.update({
                "kind": "introduced_canonical_markdown_table_header",
                "prelude_kind": prelude.kind,
                "prelude_content": suffix[prelude.start:prelude.end],
                "prelude_source_content_start": (
                    base_start + local_discovery_start + prelude.start
                ),
                "prelude_source_content_end": (
                    base_start + local_discovery_start + prelude.end
                ),
            })
        return context
    return None


def _trusted_table_boundary_points(
    text: str,
    inherited: tuple[int, ...] = (),
    *,
    local_discovery_start: int = 0,
) -> tuple[int, ...]:
    """Combine canonical ancestor cuts with safe local table discovery.

    ``local_discovery_start`` excludes the prefix governed by an inherited
    table header. Rediscovering tables in that prefix would allow ordinary body
    rows to replace immutable ancestor provenance. The suffix starts at the
    ancestor context's exact applicability end and is therefore eligible to
    introduce an independent later table.
    """

    discovery_start = min(max(local_discovery_start, 0), len(text))
    local = (
        discovery_start + cut
        for cut in _markdown_table_boundary_points(text[discovery_start:])
    )
    return tuple(sorted({
        cut
        for cut in (*inherited, *local)
        if 0 < cut < len(text)
    }))


def _inherited_table_context_local_end(
    payload: dict, *, base_start: int, content_length: int,
) -> int:
    """Return the local end of an already-validated inherited table context."""

    context = payload.get("source_fragment_context")
    if context is None:
        return 0
    applies_through = context["applies_through_source_content_end"]
    return min(content_length, max(0, applies_through - base_start))


def _is_atx_heading(line: str) -> bool:
    leading = len(line) - len(line.lstrip(" "))
    if leading > 3:
        return False
    remainder = line[leading:]
    hashes = len(remainder) - len(remainder.lstrip("#"))
    return (
        1 <= hashes <= 6
        and (len(remainder) == hashes or remainder[hashes] in " \t")
    )


def _list_marker_indent(line: str) -> int | None:
    match = _MARKDOWN_LIST_ITEM_RE.fullmatch(line)
    if match is None:
        return None
    return _leading_indent_width(match.group("indent"))


def _list_blocks(
    lines: tuple[_MarkdownLine, ...],
    fenced_spans: tuple[tuple[int, int], ...],
) -> tuple[_MarkdownListBlock, ...]:
    """Return conservative top-level lists and atomic root-item spans.

    A root item owns nested markers, indented continuation blocks, blank lines,
    and unblanked lazy continuation text.  Only another marker at the same
    indentation proves a sibling boundary.  Ambiguity therefore expands an
    atomic item instead of manufacturing a soft-line split.
    """

    blocks: list[_MarkdownListBlock] = []
    index = 0
    while index < len(lines):
        if _offset_in_spans(lines[index].start, fenced_spans):
            index += 1
            continue
        base_indent = _list_marker_indent(lines[index].text)
        if base_indent is None or base_indent > 3:
            index += 1
            continue

        block_start = index
        item_start = lines[index].start
        item_spans: list[tuple[int, int]] = []
        cursor = index + 1
        saw_blank = False
        while cursor < len(lines):
            line = lines[cursor]
            if not line.text.strip(" \t"):
                saw_blank = True
                cursor += 1
                continue

            marker_indent = (
                None
                if _offset_in_spans(line.start, fenced_spans)
                else _list_marker_indent(line.text)
            )
            if marker_indent == base_indent:
                item_spans.append((item_start, line.start))
                item_start = line.start
                saw_blank = False
                cursor += 1
                continue
            if marker_indent is not None and marker_indent < base_indent:
                break

            indentation = _leading_indent_width(line.text)
            if saw_blank and indentation <= base_indent:
                break
            if (
                not saw_blank
                and indentation <= base_indent
                and _is_atx_heading(line.text)
            ):
                # An unindented heading interrupts a list item even without a
                # blank line; keeping it in the item would erase a useful,
                # independently delimited section boundary.
                break
            saw_blank = False
            cursor += 1

        item_end = lines[cursor].start if cursor < len(lines) else lines[-1].end
        item_spans.append((item_start, item_end))
        blocks.append(_MarkdownListBlock(
            start_line=block_start,
            end_line=cursor,
            item_spans=tuple(item_spans),
        ))
        index = cursor
    return tuple(blocks)


def _heading_line_ranges(
    lines: tuple[_MarkdownLine, ...],
    opaque_spans: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Recognize ATX and fail-closed CommonMark-shaped Setext headings."""

    ranges: list[tuple[int, int]] = []
    consumed: set[int] = set()
    for index, line in enumerate(lines):
        if (
            index > 0
            and _MARKDOWN_SETEXT_UNDERLINE_RE.fullmatch(line.text) is not None
            and line.text.strip(" \t") != "-"
        ):
            # Setext content is the complete contiguous paragraph before the
            # underline, not merely its final line. A hyphen underline may
            # also resemble a thematic break (and a one-character underline
            # an empty list item); treating the whole possible heading as
            # attached is the fail-closed interpretation because it removes
            # unsafe sentence/paragraph cuts rather than authorizing one.
            heading_start = index
            cursor = index - 1
            while cursor >= 0:
                candidate = lines[cursor]
                if (
                    not candidate.text.strip(" \t")
                    or _offset_in_spans(candidate.start, opaque_spans)
                    or _leading_indent_width(candidate.text) > 3
                    or _is_atx_heading(candidate.text)
                    or _list_marker_indent(candidate.text) is not None
                    or _strict_markdown_table_cells(candidate.text) is not None
                    or _MARKDOWN_SETEXT_UNDERLINE_RE.fullmatch(
                        candidate.text
                    ) is not None
                    or candidate.text.lstrip(" \t").startswith((">", "<"))
                ):
                    break
                heading_start = cursor
                cursor -= 1
            if heading_start < index:
                ranges.append((heading_start, index + 1))
                consumed.update(range(heading_start, index + 1))
                continue

        if (
            index not in consumed
            and not _offset_in_spans(line.start, opaque_spans)
            and _is_atx_heading(line.text)
        ):
            ranges.append((index, index + 1))
            consumed.add(index)
    return tuple(sorted(ranges))


def _colon_intro_line_range(
    lines: tuple[_MarkdownLine, ...],
    opaque_spans: tuple[tuple[int, int], ...],
    *,
    block_start_line: int,
) -> tuple[int, int] | None:
    """Return the exact plain-paragraph range introducing a structural block."""

    cursor = block_start_line - 1
    blank_lines = 0
    while cursor >= 0 and not lines[cursor].text.strip(" \t"):
        blank_lines += 1
        cursor -= 1
    if cursor < 0 or blank_lines > 1:
        return None
    intro_line = lines[cursor]
    if (
        not intro_line.text.rstrip(" \t").endswith(":")
        or _offset_in_spans(intro_line.start, opaque_spans)
        or _is_atx_heading(intro_line.text)
        or _list_marker_indent(intro_line.text) is not None
        or _strict_markdown_table_cells(intro_line.text) is not None
    ):
        return None
    intro_start = cursor
    while intro_start > 0:
        previous = lines[intro_start - 1]
        if (
            not previous.text.strip(" \t")
            or _offset_in_spans(previous.start, opaque_spans)
            or _is_atx_heading(previous.text)
            or _list_marker_indent(previous.text) is not None
            or _strict_markdown_table_cells(previous.text) is not None
        ):
            break
        intro_start -= 1
    return intro_start, cursor + 1


def _introduced_table_preludes(
    text: str,
    *,
    lines: tuple[_MarkdownLine, ...],
    opaque_spans: tuple[tuple[int, int], ...],
    heading_ranges: tuple[tuple[int, int], ...],
    table_blocks: tuple[_MarkdownTableBlock, ...],
) -> tuple[_MarkdownTablePrelude, ...]:
    """Find table preludes the fragment contract can reproduce exactly.

    A heading may have blank separator lines before its first body block, as in
    the existing heading/body protection policy. A colon-led paragraph allows
    at most one such blank line. Separator bytes remain authoritative source
    bytes in exactly one fragment; only the labelled prelude and header slices
    are repeated as interpretation context.
    """

    table_by_start = {block.start_line: block for block in table_blocks}
    preludes: dict[int, _MarkdownTablePrelude] = {}
    for heading_start, heading_end in heading_ranges:
        body_start = heading_end
        while body_start < len(lines) and not lines[body_start].text.strip(" \t"):
            body_start += 1
        if body_start not in table_by_start:
            continue
        kind = (
            "atx_heading"
            if heading_end == heading_start + 1
            and _is_atx_heading(lines[heading_start].text)
            else "setext_heading"
        )
        start = lines[heading_start].start
        end = lines[heading_end - 1].end
        content = text[start:end]
        if not _valid_source_fragment_prelude(kind, content):
            continue
        preludes[body_start] = _MarkdownTablePrelude(
            table_start_line=body_start,
            kind=kind,
            start=start,
            end=end,
        )

    for block in table_blocks:
        if block.start_line in preludes:
            continue
        intro_range = _colon_intro_line_range(
            lines,
            opaque_spans,
            block_start_line=block.start_line,
        )
        if intro_range is None:
            continue
        intro_start, intro_end = intro_range
        start = lines[intro_start].start
        end = lines[intro_end - 1].end
        content = text[start:end]
        kind = "colon_led_paragraph"
        if not _valid_source_fragment_prelude(kind, content):
            continue
        preludes[block.start_line] = _MarkdownTablePrelude(
            table_start_line=block.start_line,
            kind=kind,
            start=start,
            end=end,
        )
    return tuple(preludes[index] for index in sorted(preludes))


def _markdown_block_analysis(text: str) -> _MarkdownBlockAnalysis:
    """Compute protected atoms and explicitly structural source boundaries.

    The invariant is fail-closed: a candidate cut is admissible only when it
    lies outside the strict interior of every recognized block relationship.
    This parser intentionally recognizes a conservative Markdown subset; it
    never turns a generic newline into evidence of semantic independence.
    """

    lines = _markdown_lines(text)
    if not lines:
        return _MarkdownBlockAnalysis((), (), ())
    fenced_spans = _fenced_code_spans(text, lines)
    list_blocks = _list_blocks(lines, fenced_spans)
    list_item_spans = tuple(
        span for block in list_blocks for span in block.item_spans
    )
    opaque_spans = _normalized_protected_spans([
        *fenced_spans, *list_item_spans,
    ])
    heading_ranges = _heading_line_ranges(lines, opaque_spans)
    heading_starts = {start for start, _end in heading_ranges}
    table_blocks = _markdown_table_blocks(
        text, lines=lines, fenced_spans=fenced_spans
    )
    introduced_table_preludes = _introduced_table_preludes(
        text,
        lines=lines,
        opaque_spans=opaque_spans,
        heading_ranges=heading_ranges,
        table_blocks=table_blocks,
    )
    introduced_table_start_lines = {
        prelude.table_start_line for prelude in introduced_table_preludes
    }
    introduced_table_prelude_by_start = {
        prelude.table_start_line: prelude
        for prelude in introduced_table_preludes
    }

    list_by_start = {block.start_line: block for block in list_blocks}
    table_by_start = {block.start_line: block for block in table_blocks}
    line_index_by_start = {line.start: index for index, line in enumerate(lines)}
    fence_by_start = {
        line_index_by_start[span[0]]: span
        for span in fenced_spans
        if span[0] in line_index_by_start
    }

    protected: list[tuple[int, int]] = [*opaque_spans]
    structural: set[int] = set()

    for heading_start, heading_end in heading_ranges:
        structural.add(lines[heading_start].start)
        body_start = heading_end
        while (
            body_start < len(lines)
            and not lines[body_start].text.strip(" \t")
        ):
            body_start += 1

        heading_only_end = lines[heading_end - 1].end
        if body_start >= len(lines) or body_start in heading_starts:
            body_end = heading_only_end
        elif body_start in fence_by_start:
            body_end = fence_by_start[body_start][1]
        elif body_start in list_by_start:
            block = list_by_start[body_start]
            body_end = (
                lines[block.end_line].start
                if block.end_line < len(lines) else len(text)
            )
        elif body_start in table_by_start:
            block = table_by_start[body_start]
            body_end = block.table_end
        else:
            cursor = body_start + 1
            while cursor < len(lines):
                if not lines[cursor].text.strip(" \t"):
                    break
                if (
                    cursor in heading_starts
                    or cursor in fence_by_start
                    or cursor in list_by_start
                    or cursor in table_by_start
                ):
                    break
                cursor += 1
            body_end = lines[cursor - 1].end
        protected.append((lines[heading_start].start, body_end))
        structural.add(body_end)

    # A plain paragraph ending in a colon is defensible introduction evidence
    # when a list, fence, or canonical table follows immediately (with at most
    # one blank separator). Bind the paragraph to the entire introduced block:
    # later list items/table rows still depend on the shared predicate/context.
    def bind_colon_intro(block_start_line: int, block_end: int) -> None:
        intro_range = _colon_intro_line_range(
            lines,
            opaque_spans,
            block_start_line=block_start_line,
        )
        if intro_range is None:
            return
        intro_start, _intro_end = intro_range
        protected.append((
            lines[intro_start].start,
            block_end,
        ))

    for block in list_blocks:
        bind_colon_intro(block.start_line, block.item_spans[-1][1])
    for start_line, span in fence_by_start.items():
        bind_colon_intro(start_line, span[1])
    for block in table_blocks:
        bind_colon_intro(block.start_line, block.table_end)

    for start, end in fenced_spans:
        if 0 < start < len(text):
            structural.add(start)
        if 0 < end < len(text):
            structural.add(end)
    for block in list_blocks:
        structural.update(start for start, _end in block.item_spans[1:])
        block_end = block.item_spans[-1][1]
        if 0 < block_end < len(text):
            structural.add(block_end)
    for block in table_blocks:
        block_end = block.table_end
        if 0 < block_end < len(text):
            structural.add(block_end)

    spans = _normalized_protected_spans([
        (start, end)
        for start, end in protected
        if 0 <= start < end <= len(text)
    ])
    boundaries = tuple(sorted(
        cut
        for cut in structural
        if 0 < cut < len(text) and not _cut_inside_spans(cut, spans)
    ))
    contextual_table_boundaries = tuple(sorted({
        cut
        for block in table_blocks
        if block.start_line in introduced_table_start_lines
        for cut in block.row_boundaries
        if 0 < cut < len(text)
        and all(
            not (span_start < cut < span_end)
            or introduced_table_prelude_by_start[block.start_line].start
            <= span_start
            for span_start, span_end in spans
        )
    }))
    return _MarkdownBlockAnalysis(
        spans,
        boundaries,
        contextual_table_boundaries,
    )


def _semantic_split_point(
    text: str,
    *,
    trusted_table_boundaries: tuple[int, ...] = (),
    table_discovery_start: int = 0,
) -> int | None:
    """Choose a deterministic boundary without cutting a semantic unit.

    Prefer paragraph, explicit block edge, sentence-terminal, then validated
    Markdown-table row boundaries when they are reasonably close to the
    balanced cut. Candidates inside a protected block relationship are
    discarded. A soft line wrap is not a semantic boundary: splitting
    ``depends\non`` can erase the relation from both authoritative children.
    If no boundary class is close, use the closest admissible boundary of any
    class; resource/depth limits still decide whether that necessarily-
    unbalanced tree is safe to execute. The cut consumes separator whitespace
    on the left and preserves every source character exactly once across the
    two fragments.
    """

    if len(text) < 2 * _MIN_FRAGMENT_CONTENT_CHARS:
        return None
    midpoint = len(text) // 2
    lower = _MIN_FRAGMENT_CONTENT_CHARS
    upper = len(text) - _MIN_FRAGMENT_CONTENT_CHARS
    block_analysis = _markdown_block_analysis(text)

    def admissible(cut: int) -> bool:
        return (
            lower <= cut <= upper
            and not _cut_inside_spans(cut, block_analysis.protected_spans)
        )

    def table_admissible(cut: int) -> bool:
        # A heading/colon and its first table remain one semantic unit. The
        # only safe interior escape hatch is a body-row edge for which the same
        # parser proved that exact prelude can accompany the table header in a
        # labelled source context. Every other protected relationship remains
        # atomic, including tables nested in list items.
        return (
            lower <= cut <= upper
            and (
                not _cut_inside_spans(cut, block_analysis.protected_spans)
                or cut in block_analysis.contextual_table_boundaries
            )
        )

    tiers: list[list[int]] = []
    tiers.append(sorted({
        match.end()
        for match in _PARAGRAPH_BOUNDARY_RE.finditer(text)
        if admissible(match.end())
    }))
    tiers.append([
        cut for cut in block_analysis.structural_boundaries
        if admissible(cut)
    ])
    tiers.append([
        cut for cut in _sentence_boundary_points(text)
        if admissible(cut)
    ])
    tiers.append([
        cut for cut in _trusted_table_boundary_points(
            text,
            trusted_table_boundaries,
            local_discovery_start=table_discovery_start,
        )
        if table_admissible(cut)
    ])

    near = min(_NEAR_BALANCED_BOUNDARY_CHARS, max(lower, len(text) // 4))
    for candidates in tiers:
        nearby = [cut for cut in candidates if abs(cut - midpoint) <= near]
        if nearby:
            return min(nearby, key=lambda cut: (abs(cut - midpoint), cut))

    ranked = [
        (abs(cut - midpoint), priority, cut)
        for priority, candidates in enumerate(tiers)
        for cut in candidates
    ]
    return min(ranked)[2] if ranked else None


def _conversation_context_suffix(content: str, limit: int) -> str | None:
    """Retain an exact suffix without cutting negation, lists or code blocks."""

    if len(content) <= limit:
        return content
    blocks = _markdown_block_analysis(content)
    candidates = {
        *(match.end() for match in _PARAGRAPH_BOUNDARY_RE.finditer(content)),
        *blocks.structural_boundaries,
        *_sentence_boundary_points(content),
    }
    cuts = [
        cut for cut in candidates
        if len(content) - limit <= cut < len(content)
        and not _cut_inside_spans(cut, blocks.protected_spans)
    ]
    return content[min(cuts):] if cuts else None


def _conversation_context_slice(
    payload: dict, limit: int,
) -> tuple[str, dict | None] | None:
    """Keep a prose suffix or a proven table tail with its exact semantics."""

    content = payload["content"]
    inherited = payload.get("source_fragment_context")
    suffix = _conversation_context_suffix(content, limit)
    if suffix is not None:
        return suffix, inherited
    base_start = (
        0 if payload.get("source_record_version") == "hymem-claim-source-v2"
        else payload["source_content_start"]
    )
    discovery_start = _inherited_table_context_local_end(
        payload, base_start=base_start, content_length=len(content),
    )
    blocks = _markdown_block_analysis(content)
    for cut in _trusted_table_boundary_points(
        content, (), local_discovery_start=discovery_start,
    ):
        if not (
            len(content) - limit <= cut < len(content)
            and (
                not _cut_inside_spans(cut, blocks.protected_spans)
                or cut in blocks.contextual_table_boundaries
            )
        ):
            continue
        table_context = _canonical_table_context_for_cut(
            content, cut=cut, base_start=base_start,
            inherited_context=inherited, local_discovery_start=discovery_start,
        )
        if table_context is not None:
            return content[cut:], table_context
    return None


def _preceding_conversation_context(
    unit: _ExtractionUnit,
    left_records: tuple[tuple[int, str], ...],
    right_record: tuple[int, str],
) -> tuple[tuple[int, str], ...] | None:
    """Build a bounded adjacent window; None holds an unsafe truncation.

    Stop at session/workspace/distinct-user boundaries. Never skip an unrepresentable nearest
    record to attach more distant context, and never truncate inside a semantic
    unit merely to fit it into the context allowance.
    """

    target = _source_payload(right_record)
    if target is None:
        return None
    scope = (target.get("source_session_id"), target.get("source_workspace_id"))
    candidates = unit.context_records + left_records
    remaining = _MAX_CONVERSATION_CONTEXT_CHARS
    encoded_remaining = _MAX_CONVERSATION_CONTEXT_ENCODED_CHARS
    chosen: list[tuple[int, str]] = []
    for message_id, encoded in reversed(candidates):
        if len(chosen) >= _MAX_CONVERSATION_CONTEXT_RECORDS or remaining == 0:
            break
        payload = loads_strict_json(encoded)
        if (payload.get("source_session_id"), payload.get("source_workspace_id")) != scope:
            break
        if (
            payload.get("source_role") == target.get("source_role") == "user"
            and payload.get("source_peer_id") is not None
            and target.get("source_peer_id") is not None
            and payload["source_peer_id"] != target["source_peer_id"]
        ):
            # User-local confirmation cannot adopt another user's statements.
            # Assistant turns may have their own peer ID and remain eligible.
            break
        sliced = _conversation_context_slice(payload, remaining)
        if sliced is None:
            # The nearest record cannot be represented faithfully. A more
            # distant optional record may be omitted once adjacency is kept.
            if not chosen:
                return None
            break
        content, table_context = sliced
        end = (
            len(payload["content"])
            if payload.get("source_record_version") == "hymem-claim-source-v2"
            else payload["source_content_end"]
        )
        context = {
            key: payload.get(key) for key in (
                "source_message_id", "source_role", "source_peer_id",
                "source_session_id", "source_workspace_id", "source_created_at",
            )
        }
        context.update({
            "content": content,
            "source_record_version": SOURCE_CONVERSATION_CONTEXT_VERSION,
            "source_context_only": True,
            "source_content_start": end - len(content),
            "source_content_end": end,
            "context_for_source_message_id": right_record[0],
            "applies_through_source_content_end": min(
                _MAX_CONVERSATION_CONTEXT_APPLICABILITY_CHARS,
                len(target["content"])
                if target.get("source_record_version") == "hymem-claim-source-v2"
                else target["source_content_end"],
            ),
        })
        while True:
            if table_context is not None:
                context["source_fragment_context"] = table_context
            else:
                context.pop("source_fragment_context", None)
            encoded_context = json.dumps(context, sort_keys=True, separators=(",", ":"))
            excess = len(encoded_context) + 1 - encoded_remaining
            if excess <= 0:
                break
            # Header/prelude metadata also consumes the hard encoded allowance.
            # Shorten only at another proven boundary, never an arbitrary byte.
            sliced = _conversation_context_slice(payload, max(0, len(content) - excess))
            if sliced is None or len(sliced[0]) >= len(content):
                break
            content, table_context = sliced
            context["content"] = content
            context["source_content_start"] = end - len(content)
        if excess > 0:
            if not chosen:
                return None
            break
        chosen.append((message_id, encoded_context))
        remaining -= len(content)
        encoded_remaining -= len(encoded_context) + 1
    return tuple(reversed(chosen))


def _source_unit(
    records: tuple[tuple[int, str], ...],
    *,
    context_records: tuple[tuple[int, str], ...] = (),
    trusted_table_boundaries: tuple[int, ...] = (),
) -> _ExtractionUnit:
    """Render context and owned records, retaining one-sided applicability."""

    first_payload = _source_payload(records[0])
    assert first_payload is not None
    first_start = (
        0 if first_payload.get("source_record_version") == "hymem-claim-source-v2"
        else first_payload["source_content_start"]
    )
    retained = tuple(
        record for record in context_records
        if (context := loads_strict_json(record[1]))["context_for_source_message_id"]
        == records[0][0]
        and first_start < context["applies_through_source_content_end"]
    )
    return _ExtractionUnit(
        text="\n".join(encoded for _mid, encoded in retained + records),
        source_records=records,
        context_records=retained,
        trusted_table_boundaries=trusted_table_boundaries,
    )


def _split_unit(unit: _ExtractionUnit) -> tuple[_ExtractionUnit, _ExtractionUnit] | None:
    records = unit.source_records
    table_boundaries: tuple[int, ...] = ()
    right_context = unit.context_records
    if records is not None:
        if len(records) > 1:
            lengths = [len(encoded) + 1 for _mid, encoded in records]
            total = sum(lengths)
            running = 0
            cut = 1
            best = total
            for index in range(1, len(records)):
                running += lengths[index - 1]
                distance = abs(total - 2 * running)
                if distance < best:
                    best = distance
                    cut = index
            left_records = records[:cut]
            right_records = records[cut:]
            right_context = _preceding_conversation_context(
                unit, left_records, right_records[0],
            )
            if right_context is None:
                return None
        elif len(records) == 1:
            payload = _source_payload(records[0])
            if payload is None:
                return None
            content = payload["content"]
            base_start = payload.get("source_content_start", 0)
            if isinstance(base_start, bool) or not isinstance(base_start, int):
                return None
            inherited_context = payload.get("source_fragment_context")
            local_discovery_start = _inherited_table_context_local_end(
                payload,
                base_start=base_start,
                content_length=len(content),
            )
            table_boundaries = _trusted_table_boundary_points(
                content,
                unit.trusted_table_boundaries,
                local_discovery_start=local_discovery_start,
            )
            cut = _semantic_split_point(
                content,
                trusted_table_boundaries=table_boundaries,
                table_discovery_start=local_discovery_start,
            )
            if cut is None:
                return None
            prose_boundary = _is_prose_boundary_cut(content, cut)
            right_boundary_context = _source_boundary_context_for_cut(
                payload,
                content=content,
                cut=cut,
                base_start=base_start,
            )
            if prose_boundary and right_boundary_context is None:
                # Never fall back to independently authoritative prose leaves
                # when the exact bounded adjacent window is unavailable.
                return None
            right_table_context = _canonical_table_context_for_cut(
                content,
                cut=cut,
                base_start=base_start,
                inherited_context=inherited_context,
                local_discovery_start=local_discovery_start,
            )
            if (
                cut in table_boundaries
                and right_table_context is None
            ):
                return None
            left_record = _fragment_record(records[0], start=0, end=cut)
            right_record = _fragment_record(
                records[0],
                start=cut,
                end=len(content),
                source_fragment_context=right_table_context,
                source_boundary_context=right_boundary_context,
            )
            if left_record is None or right_record is None:
                return None
            left_records = (left_record,)
            right_records = (right_record,)
        else:
            return None
        return (
            _source_unit(
                left_records,
                context_records=unit.context_records,
                trusted_table_boundaries=(
                    tuple(point for point in table_boundaries if point < cut)
                    if len(records) == 1 else ()
                ),
            ),
            _source_unit(
                right_records,
                context_records=right_context,
                trusted_table_boundaries=(
                    tuple(
                        point - cut
                        for point in table_boundaries
                        if point > cut
                    )
                    if len(records) == 1 else ()
                ),
            ),
        )

    table_boundaries = _trusted_table_boundary_points(
        unit.text, unit.trusted_table_boundaries
    )
    semantic_cut = _semantic_split_point(
        unit.text, trusted_table_boundaries=table_boundaries
    )
    if semantic_cut is None:
        return None
    if semantic_cut in table_boundaries:
        # Legacy source-less callers have nowhere to carry explicitly labelled
        # header context. Holding is safer than presenting a header-dependent
        # row continuation as a self-contained excerpt.
        return None
    return (
        _ExtractionUnit(
            unit.text[:semantic_cut],
            trusted_table_boundaries=tuple(
                point for point in table_boundaries if point < semantic_cut
            ),
        ),
        _ExtractionUnit(
            unit.text[semantic_cut:],
            trusted_table_boundaries=tuple(
                point - semantic_cut
                for point in table_boundaries
                if point > semantic_cut
            ),
        ),
    )


def _prepartition(
    unit: _ExtractionUnit,
) -> tuple[list[tuple[_ExtractionUnit, int]] | None, ChunkResult | None]:
    leaves: list[tuple[_ExtractionUnit, int]] = []
    failure_details: list[str] = []

    def visit(current: _ExtractionUnit, depth: int) -> bool:
        if len(current.text) <= _MAX_LEAF_INPUT_CHARS:
            leaves.append((current, depth))
            if len(leaves) <= _MAX_PREPARTITION_LEAVES:
                return True
            failure_details.append("input:prepartition_limit_exceeded")
            return False
        if depth >= _MAX_SPLIT_DEPTH:
            failure_details.append("split:max_depth_reached")
            return False
        split = _split_unit(current)
        if split is None:
            failure_details.append("split:no_admissible_semantic_boundary")
            return False
        return visit(split[0], depth + 1) and visit(split[1], depth + 1)

    if not visit(unit, 0):
        return None, _failure(
            "resource_limit",
            *(failure_details or ["input:prepartition_limit_exceeded"]),
        )
    return leaves, None


def _unit_has_explicit_cue(unit: _ExtractionUnit) -> bool:
    if unit.source_records is None:
        return _EXPLICIT_EXTRACTION_CUE.search(unit.text) is not None
    for record in unit.source_records:
        payload = _source_payload(record)
        if payload is not None and _EXPLICIT_EXTRACTION_CUE.search(
            payload["content"]
        ):
            return True
    return False


def _unit_semantic_content_chars(unit: _ExtractionUnit) -> int:
    """Underlying content length, excluding source-record JSON metadata."""

    if unit.source_records is None:
        return len(unit.text)
    total = 0
    for record in unit.source_records:
        payload = _source_payload(record)
        if payload is None:  # validated at the public boundary
            return len(unit.text)
        total += len(payload["content"])
    return total


def _consistent_unique_triples(
    items: list[dict],
    allowed_source_message_ids: frozenset[int] | None = None,
) -> tuple[list[dict] | None, tuple[str, ...], int]:
    """Validate citations/core consistency and merge exact claim duplicates."""
    polarities: dict[tuple[str, str, str, int | None], int] = {}
    grouped: dict[tuple[str, str, str, int, int | None], list[dict]] = {}
    details: list[str] = []
    for index, item in enumerate(items):
        source_message_id = item.get("source_message_id")
        if (
            allowed_source_message_ids is not None
            and source_message_id not in allowed_source_message_ids
        ):
            details.append(f"triples[{index}].source_message_id:not_in_input")
            continue
        claim = (
            _normalized_identity(item["subject"]),
            item["predicate"],
            _normalized_identity(item["object"]),
            source_message_id,
        )
        polarity = item["polarity"]
        prior = polarities.get(claim)
        if prior is not None and prior != polarity:
            details.append(f"triples[{index}].polarity:response_conflict")
            continue
        polarities[claim] = polarity
        grouped.setdefault((*claim[:3], polarity, source_message_id), []).append(item)
    if details:
        return None, _bounded_details(details), 0

    merged: list[dict] = []
    optional_fields = (
        "value_text", "value_numeric", "value_unit", "temporal_scope",
        "subject_type", "object_type", "subject_properties", "object_properties",
    )
    for _identity, group in grouped.items():
        winner = min(
            group,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
        combined = {
            key: winner[key]
            for key in ("subject", "predicate", "object", "polarity")
        }
        if "source_message_id" in winner:
            combined["source_message_id"] = winner["source_message_id"]
        for field_name in optional_fields:
            values = {
                json.dumps(item[field_name], ensure_ascii=False, sort_keys=True)
                for item in group
                if field_name in item
            }
            if len(values) == 1:
                combined[field_name] = next(
                    item[field_name] for item in group if field_name in item
                )
        merged.append(combined)
    return merged, (), sum(len(group) - 1 for group in grouped.values())


def _entity_hints(
    items: list[dict],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    types: dict[str, dict[str, object]] = {}
    properties: dict[str, dict[str, object]] = {}
    for item in items:
        for entity_field, type_field, properties_field in (
            ("subject", "subject_type", "subject_properties"),
            ("object", "object_type", "object_properties"),
        ):
            entity = item[entity_field]
            identity = _normalized_identity(entity)
            type_bucket = types.setdefault(
                identity, {"display": entity, "values": set()}
            )
            type_bucket["display"] = min(str(type_bucket["display"]), entity)
            if type_field in item:
                type_bucket["values"].add(item[type_field])

            property_bucket = properties.setdefault(
                identity, {"display": entity, "values": {}}
            )
            property_bucket["display"] = min(
                str(property_bucket["display"]), entity
            )
            for key, value in item.get(properties_field, {}).items():
                property_bucket["values"].setdefault(key, set()).add(value)

    type_hints = {
        str(bucket["display"]): next(iter(bucket["values"]))
        for _identity, bucket in sorted(types.items())
        if len(bucket["values"]) == 1
    }
    property_hints: dict[str, dict[str, str]] = {}
    for _identity, bucket in sorted(properties.items()):
        unambiguous = {
            key: next(iter(values))
            for key, values in sorted(bucket["values"].items())
            if len(values) == 1
        }
        if unambiguous:
            property_hints[str(bucket["display"])] = unambiguous
    return type_hints, property_hints


def _accepted_items_context(result: ChunkResult) -> str:
    """Serialize bounded claim identities for the one-shot omission prompt.

    The source excerpt remains byte-for-byte identical to the primary unit.
    This compact block is comparison context only; it intentionally excludes
    enrichment hints because they do not define whether a claim was omitted.
    """
    triples: list[dict[str, object]] = []
    for triple in result.triples:
        item: dict[str, object] = {
            "subject": triple.subject,
            "predicate": triple.predicate,
            "object": triple.object,
            "polarity": triple.polarity,
        }
        if triple.source_message_id is not None:
            item["source_message_id"] = triple.source_message_id
        triples.append(item)
    payload = {
        "triples": triples,
        "markers": [
            {"kind": marker.kind, "statement": marker.statement}
            for marker in result.markers
        ],
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _merge_verified_result(
    primary: ChunkResult, verification: ChunkResult
) -> ChunkResult:
    """Merge one omission pass without exposing either partial on failure."""
    merged = _merge_results(primary, verification)
    if not merged.failed:
        return merged
    return ChunkResult(
        failed=True,
        failure_reason=merged.failure_reason,
        failure_details=merged.failure_details,
    )


def extract_chunk(
    client: LLMClient,
    text: str,
    *,
    source_records: tuple[tuple[int, str], ...] | None = None,
    completion_call_limit: int | None = None,
) -> ChunkResult:
    """Extract a complete chunk through bounded source-safe subdivision."""
    if completion_call_limit is None:
        completion_call_limit = MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK
    if (
        isinstance(completion_call_limit, bool)
        or not isinstance(completion_call_limit, int)
        or completion_call_limit < 1
        or completion_call_limit > MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK
    ):
        raise ValueError(
            "completion_call_limit must be an integer between 1 and "
            f"{MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK}"
        )
    system = build_chunk_extraction_system()
    empty_verification_system = build_chunk_empty_verification_system()
    omission_verification_system = build_chunk_omission_verification_system()
    budget = _CallBudget()
    initial_prepartition_leaves = 0

    def finish(result: ChunkResult) -> ChunkResult:
        result.completion_calls = budget.completion_calls
        result.provider_attempts = budget.provider_attempts
        result.initial_prepartition_leaves = initial_prepartition_leaves
        return result

    if source_records is not None:
        source_payloads = tuple(
            _source_payload(record) for record in source_records
        )
        if (
            not source_records
            or len({mid for mid, _record in source_records}) != len(source_records)
            or any(
                payload is None
                or payload.get("source_record_version")
                != "hymem-claim-source-v2"
                for payload in source_payloads
            )
        ):
            return finish(
                _failure("input_contract_failure", "source_records:invalid")
            )
        unit = _ExtractionUnit(
            text="\n".join(record for _mid, record in source_records),
            source_records=source_records,
        )
    else:
        unit = _ExtractionUnit(text=text)

    def attempt(
        current: _ExtractionUnit,
        *,
        verification: str | None = None,
        accepted: ChunkResult | None = None,
    ) -> tuple[ChunkResult, str | None]:
        if (
            budget.completion_calls
            >= completion_call_limit
        ):
            return _failure("resource_limit", "calls:max_exceeded"), None
        budget.completion_calls += 1
        if verification == "omission":
            if accepted is None:
                return _failure(
                    "internal_validation_failure", "verification:missing_context"
                ), None
            request_system = omission_verification_system
            request_user = CHUNK_OMISSION_VERIFICATION_USER_TEMPLATE.format(
                text=current.text,
                accepted=_accepted_items_context(accepted),
            )
        else:
            request_system = empty_verification_system if verification else system
            request_user = CHUNK_EXTRACTION_USER_TEMPLATE.format(text=current.text)
        request = LLMRequest(
            system=request_system,
            user=request_user,
            response_format="json",
            max_tokens=_chunk_max_tokens(current.text),
        )
        attempt_measurement = None
        try:
            with measure_provider_attempts(client) as attempt_measurement:
                raw = client.complete(request)
        except Exception as exc:
            log.warning(
                "chunk_extraction.call_failure error_type=%s", type(exc).__name__
            )
            return _failure("call_failure", "provider:call_failed"), None
        finally:
            # The helper populates on context exit, including when complete()
            # raises. It always yields before the provider call, so ``None`` is
            # only a defensive guard against an instrumentation regression.
            budget.provider_attempts += (
                attempt_measurement.attempts
                if attempt_measurement is not None
                else 1
            )

        data = loads_exact_or_fenced(raw)
        if data is None:
            cut = isinstance(raw, str) and is_ceiling_cut(raw)
            detail = "response:truncated_json" if cut else "response:invalid_json"
            log.warning(
                "chunk_extraction.parse_failure reason=%s raw_len=%d",
                detail,
                len(raw) if isinstance(raw, str) else -1,
            )
            return _failure("parse_failure", detail), raw
        if not isinstance(data, dict):
            log.warning("chunk_extraction.shape_failure type=%s", type(data).__name__)
            return _failure("shape_failure", "response:not_object"), raw

        required = {"triples", "markers", "complete"}
        missing = sorted(required - set(data))
        extra = set(data) - required
        if missing or extra:
            details = [f"top.{key}:missing" for key in missing]
            if extra:
                details.append("top:unexpected_keys")
            event = (
                "chunk_extraction.missing_keys"
                if missing else "chunk_extraction.object_keys_failure"
            )
            log.warning(
                "%s details=%s",
                event,
                ",".join(details),
            )
            return ChunkResult(
                failed=True,
                failure_reason="contract_failure",
                failure_details=_bounded_details(details),
            ), raw
        if not isinstance(data["complete"], bool):
            log.warning("chunk_extraction.completion_shape_failure")
            return _failure("contract_failure", "top.complete:not_boolean"), raw
        triples_raw = data["triples"]
        markers_raw = data["markers"]
        if not isinstance(triples_raw, list) or not isinstance(markers_raw, list):
            details: list[str] = []
            if not isinstance(triples_raw, list):
                details.append("top.triples:not_array")
            if not isinstance(markers_raw, list):
                details.append("top.markers:not_array")
            log.warning(
                "chunk_extraction.array_shape_failure details=%s",
                ",".join(details),
            )
            return ChunkResult(
                failed=True,
                failure_reason="contract_failure",
                failure_details=_bounded_details(details),
            ), raw
        if not data["complete"]:
            log.warning("chunk_extraction.incomplete_response")
            return _failure("incomplete_response", "top.complete:false"), raw
        # The declared maxima are saturation sentinels, not publishable counts.
        # A model can incorrectly claim complete=true after filling an array to
        # its requested boundary. Force source subdivision at the boundary so
        # an omitted 25th/13th item can never be silently certified complete.
        if (
            len(triples_raw) >= _MAX_TRIPLES_PER_RESPONSE
            or len(markers_raw) >= _MAX_MARKERS_PER_RESPONSE
        ):
            details: list[str] = []
            if len(triples_raw) >= _MAX_TRIPLES_PER_RESPONSE:
                details.append("top.triples:item_cap_reached")
            if len(markers_raw) >= _MAX_MARKERS_PER_RESPONSE:
                details.append("top.markers:item_cap_reached")
            log.warning(
                "chunk_extraction.output_limit_exceeded details=%s",
                ",".join(details),
            )
            return ChunkResult(
                failed=True,
                failure_reason="output_limit_exceeded",
                failure_details=_bounded_details(details),
            ), raw

        normalized_triples: list[dict] = []
        validation_details: list[str] = []
        normalization_details: list[str] = []
        for index, item in enumerate(triples_raw):
            normalized, errors, normalizations = normalize_combined_triple_item(
                item,
                require_source_message_id=current.allowed_ids is not None,
            )
            validation_details.extend(
                f"triples[{index}].{detail}" for detail in errors
            )
            normalization_details.extend(
                f"triples[{index}].{detail}" for detail in normalizations
            )
            if normalized is not None:
                normalized_triples.append(normalized)

        normalized_markers: list[dict] = []
        for index, item in enumerate(markers_raw):
            normalized, errors = normalize_combined_marker_item(item)
            validation_details.extend(
                f"markers[{index}].{detail}" for detail in errors
            )
            if normalized is not None:
                normalized_markers.append(normalized)

        if validation_details:
            details = _bounded_details(validation_details)
            log.warning(
                "chunk_extraction.item_validation_failure details=%s",
                ",".join(details),
            )
            return ChunkResult(
                failed=True,
                failure_reason="item_validation_failure",
                failure_details=details,
            ), raw
        if normalization_details:
            log.info(
                "chunk_extraction.optional_metadata_normalized details=%s",
                ",".join(_bounded_details(normalization_details)),
            )

        (
            unique_raw, consistency_details, duplicate_triples_collapsed,
        ) = _consistent_unique_triples(
            normalized_triples, current.allowed_ids
        )
        if unique_raw is None:
            log.warning(
                "chunk_extraction.response_conflict details=%s",
                ",".join(consistency_details),
            )
            return ChunkResult(
                failed=True,
                failure_reason="response_conflict",
                failure_details=consistency_details,
            ), raw
        triples = triples_from_list(unique_raw)
        markers = markers_from_list(normalized_markers)
        if len(triples) != len(unique_raw) or len(markers) != len(normalized_markers):
            log.error("chunk_extraction.internal_validation_mismatch")
            return _failure(
                "internal_validation_failure", "constructors:count_mismatch"
            ), raw
        type_hints, property_hints = _entity_hints(unique_raw)
        return ChunkResult(
            triples=triples,
            entity_type_hints=type_hints,
            entity_property_hints=property_hints,
            markers=markers,
            duplicate_triples_collapsed=duplicate_triples_collapsed,
        ), raw

    split_recoverable = {
        "parse_failure",
        "shape_failure",
        "incomplete_response",
        "output_limit_exceeded",
        "item_validation_failure",
        "response_conflict",
    }

    def verify_nonempty(
        current: _ExtractionUnit, accepted: ChunkResult
    ) -> tuple[ChunkResult, bool]:
        """Certify one non-empty candidate exactly once and merge atomically.

        This helper is the sole publication path for a successful non-empty
        terminal response, including candidates recovered by the empty-check
        and terminal-retry branches.  It deliberately calls ``attempt``
        directly rather than ``recover`` so verifier output can never trigger
        a verifier-of-verifier loop.

        The boolean tells ``recover`` whether subdivision can repair the
        verifier failure.  Resource/provider/contract failures remain atomic
        terminal failures, while the existing source-safe split recovery stays
        available for truncation, saturation, incompleteness, bad items, and
        conflicts.
        """
        verified, _verified_raw = attempt(
            current, verification="omission", accepted=accepted
        )
        merged = _merge_verified_result(accepted, verified)
        verifier_split_recoverable = (
            verified.failure_reason in split_recoverable
            or merged.failure_reason == "response_conflict"
        )
        return merged, verifier_split_recoverable

    def recover(current: _ExtractionUnit, depth: int) -> ChunkResult:
        # A recovery split can add context metadata to its children. Apply the
        # same hard input ceiling as initial prepartitioning before any call.
        if len(current.text) > _MAX_LEAF_INPUT_CHARS:
            split = _split_unit(current) if depth < _MAX_SPLIT_DEPTH else None
            if split is None:
                return _failure("resource_limit", "input:context_leaf_limit_exceeded")
            left = recover(split[0], depth + 1)
            if left.failed:
                return _failed_split_after_left(left)
            return _merge_results(left, recover(split[1], depth + 1))
        verifier_failed = False
        verifier_split_recoverable = False
        result, _raw = attempt(current)
        if not result.failed:
            if not result.triples and not result.markers:
                unresolved_cue_detail: str | None = None
                if _unit_has_explicit_cue(current):
                    split = (
                        _split_unit(current)
                        if depth < _MAX_SPLIT_DEPTH else None
                    )
                    if split is not None:
                        # Each child is a fresh primary extraction unit.  It
                        # therefore retains the same terminal empty-check
                        # obligation as the parent; suppressing that obligation
                        # was able to turn two missed child primaries into one
                        # authoritative whole-chunk empty.
                        left = recover(split[0], depth + 1)
                        if left.failed:
                            return _failed_split_after_left(left)
                        right = recover(split[1], depth + 1)
                        return _merge_results(left, right)
                    if (
                        _unit_semantic_content_chars(current)
                        >= 2 * _MIN_FRAGMENT_CONTENT_CHARS
                    ):
                        unresolved_cue_detail = (
                            "split:max_depth_reached"
                            if depth >= _MAX_SPLIT_DEPTH
                            else "split:no_admissible_semantic_boundary"
                        )
                # This is a direct, one-shot call rather than a recursive
                # ``recover`` invocation.  Its output can receive the standard
                # omission pass when non-empty, but can never trigger an
                # EMPTY-of-EMPTY loop.  Even an unsplittable suspicious unit
                # gets this genuine second look before being held fail-closed.
                result, _verified_raw = attempt(current, verification="empty")
                if result.failed:
                    if unresolved_cue_detail is not None:
                        result.failure_details = _bounded_details(
                            result.failure_details, [unresolved_cue_detail]
                        )
                    return result
                if not result.triples and not result.markers:
                    if unresolved_cue_detail is not None:
                        return _failure("resource_limit", unresolved_cue_detail)
                    return result
            if result.triples or result.markers:
                # A non-empty complete=true response is not, by itself, a
                # trustworthy completeness boundary: the model may have
                # silently returned only one of several supported claims.
                # Verify exactly once over the identical source unit, validate
                # through the same contract, and merge atomically. A non-empty
                # EMPTY result reaches this one OMISSION call; an OMISSION
                # response never re-enters either verification path.
                merged, verifier_split_recoverable = verify_nonempty(
                    current, result
                )
                if not merged.failed:
                    return merged
                # Preserve the primary recovery contract for a verifier that
                # signals truncation, saturation, incompleteness, bad items, or
                # a polarity conflict: subdivide the original source unit and
                # establish fresh primary+verification pairs on its children.
                # A provider/contract/resource failure stays failed; splitting
                # cannot repair it. In no case is the partial primary returned.
                verifier_failed = True
                result = merged
            else:
                return result

        split_recovery_required = (
            verifier_split_recoverable
            if verifier_failed
            else result.failure_reason in split_recoverable
        )
        can_split = split_recovery_required and depth < _MAX_SPLIT_DEPTH
        split = _split_unit(current) if can_split else None
        split_unavailable = split_recovery_required and split is None
        if split is not None:
            left = recover(split[0], depth + 1)
            if left.failed:
                return _failed_split_after_left(left)
            right = recover(split[1], depth + 1)
            return _merge_results(left, right)

        terminal_retry = (
            not verifier_failed
            and result.failure_reason in split_recoverable
            and (
                result.failure_reason != "parse_failure"
                or "response:truncated_json" in result.failure_details
            )
            and budget.completion_calls
            < completion_call_limit
        )
        if terminal_retry:
            retried, _retry_raw = attempt(current, verification="empty")
            if retried.failed:
                if split_unavailable:
                    retried.failure_details = _bounded_details(
                        retried.failure_details,
                        [
                            (
                                "split:max_depth_reached"
                                if depth >= _MAX_SPLIT_DEPTH
                                else "split:no_admissible_semantic_boundary"
                            )
                        ],
                    )
                return retried
            if not retried.triples and not retried.markers:
                if split_unavailable:
                    result.failure_details = _bounded_details(
                        result.failure_details,
                        [
                            (
                                "split:max_depth_reached"
                                if depth >= _MAX_SPLIT_DEPTH
                                else "split:no_admissible_semantic_boundary"
                            )
                        ],
                    )
                    return result
                return retried
            # A retry can recover a well-formed but still incomplete-looking
            # non-empty response.  It crosses the same publication boundary as
            # a primary response and therefore requires the same one-shot
            # omission certification.  Budget exhaustion in that certification
            # returns only an atomic failure, never ``retried``'s partials.
            retried, _ = verify_nonempty(current, retried)
            return retried
        if split_unavailable:
            result.failure_details = _bounded_details(
                result.failure_details,
                [
                    (
                        "split:max_depth_reached"
                        if depth >= _MAX_SPLIT_DEPTH
                        else "split:no_admissible_semantic_boundary"
                    )
                ],
            )
        if split_recovery_required and depth >= _MAX_SPLIT_DEPTH:
            result.failure_details = _bounded_details(
                result.failure_details, ["split:max_depth_reached"]
            )
        return result

    leaves, preflight_failure = _prepartition(unit)
    if preflight_failure is not None:
        return finish(preflight_failure)
    assert leaves is not None
    initial_prepartition_leaves = len(leaves)
    complete: ChunkResult | None = None
    for leaf_index, (leaf, depth) in enumerate(leaves):
        result = recover(leaf, depth)
        complete = result if complete is None else _merge_results(complete, result)
        if complete.failed:
            complete.failure_details = _bounded_details(
                [f"prepartition_leaf:{leaf_index}"], complete.failure_details
            )
            break
    return finish(complete or ChunkResult())
