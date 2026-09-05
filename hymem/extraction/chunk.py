"""Merged: retry ladder (2026-08-07) + failed-flag hold-for-retry (2026-08-08).

The ladder (payload-sized ceiling, structural-cut re-roll, split terminating
step) prevents truncation holes. The failed flag makes the runner HOLD a
chunk whose extraction did not succeed instead of marking it done — parse
failure, wrong shape, and (the ladder's swallowed) transport failure all land
in failed=True.  The only authoritative empty is the exact response contract:
an object carrying both correctly typed ``triples`` and ``markers`` arrays.
"""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field

from hymem.extraction.jsonio import is_ceiling_cut, loads_exact_or_fenced
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.markers import (
    Marker,
    combined_marker_item_is_valid,
    markers_from_list,
)
from hymem.extraction.prompts import (
    CHUNK_EXTRACTION_USER_TEMPLATE,
    build_chunk_extraction_system,
)
from hymem.extraction.triples import (
    entity_properties_from_list,
    entity_types_from_list,
    combined_triple_item_is_valid,
    triples_from_list,
)

log = logging.getLogger("hymem.extraction.chunk")


@dataclass
class ChunkResult:
    """Outcome of one chunk extraction attempt.

    ``failed`` distinguishes "the model answered and there is nothing here"
    (a REAL empty — marked done) from "the extraction did not succeed"
    (parse failure, wrong shape, or transport error — the caller must hold
    the chunk for retry rather than burn the one-shot gate).
    """

    triples: list = field(default_factory=list)
    entity_type_hints: dict = field(default_factory=dict)
    entity_property_hints: dict = field(default_factory=dict)
    markers: list[Marker] = field(default_factory=list)
    failed: bool = False


def _chunk_max_tokens(text: str) -> int:
    """Payload-sized ceiling with headroom: chunk extraction output is roughly
    proportional to input (measured 0.3x-4.8x; a 5.6KB chunk emitted ~8K
    tokens on a rambling draw). The retry ladder (re-roll on cut, then split)
    covers anything beyond the cap."""
    return min(8192, 2048 + len(text) * 2)


# Below this input length the terminating split is not attempted — the
# output of a tiny chunk cannot plausibly exceed the ceiling, and a split
# would produce degenerate halves.
_MIN_SPLIT_CHARS = 200
_SPLIT_OVERLAP_CHARS = 64


def _normalized_identity(value: str) -> str:
    return " ".join(value.casefold().split())


def _consistent_unique_triples(
    items: list[dict],
    allowed_source_message_ids: frozenset[int] | None = None,
) -> list[dict] | None:
    """Reject response-internal contradictions and dedupe repeated claims.

    Downstream type/property writers collapse by entity identity while graph
    evidence collapses by normalized triple identity. Letting one response
    disagree with itself makes list order authoritative at a cursor-burning
    boundary. Equivalent duplicate claims are retained once, selected by a
    stable JSON ordering so reversing the model's list cannot change output.
    """
    types: dict[str, str] = {}
    properties: dict[tuple[str, str], str] = {}
    polarities: dict[tuple[str, str, str, int | None], int] = {}
    unique: dict[tuple[str, str, str, int, int | None], dict] = {}
    for item in items:
        for entity_field, type_field in (
            ("subject", "subject_type"), ("object", "object_type")
        ):
            entity_type = item.get(type_field)
            if entity_type is None:
                continue
            entity = _normalized_identity(item[entity_field])
            prior_type = types.get(entity)
            if prior_type is not None and prior_type != entity_type:
                return None
            types[entity] = entity_type
        for entity_field, props_field in (
            ("subject", "subject_properties"),
            ("object", "object_properties"),
        ):
            entity = _normalized_identity(item[entity_field])
            for key, value in item.get(props_field, {}).items():
                identity = (entity, key.casefold().strip())
                normalized_value = _normalized_identity(value)
                prior_value = properties.get(identity)
                if prior_value is not None and prior_value != normalized_value:
                    return None
                properties[identity] = normalized_value
        source_message_id = item.get("source_message_id")
        if allowed_source_message_ids is not None and source_message_id not in (
            allowed_source_message_ids
        ):
            return None
        claim = (
            _normalized_identity(item["subject"]),
            item["predicate"].casefold(),
            _normalized_identity(item["object"]),
            source_message_id,
        )
        polarity = item["polarity"]
        prior_polarity = polarities.get(claim)
        if prior_polarity is not None and prior_polarity != polarity:
            return None
        polarities[claim] = polarity
        identity = (*claim[:3], polarity, source_message_id)
        prior = unique.get(identity)
        # In source-aware mode the model must produce one unambiguous citation,
        # not a repeated list whose arbitrary metadata wins. Split-attempt
        # overlap is deduplicated later, after each response passed this check.
        if prior is not None and allowed_source_message_ids is not None:
            return None
        if prior is None or json.dumps(
            item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) < json.dumps(
            prior, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ):
            unique[identity] = item
    return list(unique.values())


def _merge_results(a: ChunkResult, b: ChunkResult) -> ChunkResult:
    """Merge two ChunkResults (the two halves of a split). Partial recovery
    is strictly better than a one-shot permanent hole.

    ``failed`` propagates with OR semantics: if EITHER half failed, the chunk
    is held for retry, so a one-half failure never becomes a permanent
    partial hole. This is safe since the marker write became idempotent
    (INSERT ... SELECT ... WHERE NOT EXISTS, 2026-08-08): the re-extracted
    good half costs one LLM call and writes nothing new. Only a chunk whose
    halves ALL succeed is marked done.
    """
    type_hints = dict(a.entity_type_hints)
    normalized_types = {
        _normalized_identity(entity): entity_type
        for entity, entity_type in type_hints.items()
    }
    metadata_conflict = False
    for entity, entity_type in b.entity_type_hints.items():
        identity = _normalized_identity(entity)
        prior = normalized_types.get(identity)
        if prior is not None and prior != entity_type:
            metadata_conflict = True
        else:
            normalized_types[identity] = entity_type
            type_hints.setdefault(entity, entity_type)
    property_hints = {
        entity: dict(values)
        for entity, values in a.entity_property_hints.items()
    }
    for entity, values in b.entity_property_hints.items():
        canonical_entity = next(
            (
                prior_entity for prior_entity in property_hints
                if _normalized_identity(prior_entity)
                == _normalized_identity(entity)
            ),
            entity,
        )
        bucket = property_hints.setdefault(canonical_entity, {})
        for key, value in values.items():
            prior_key = next(
                (prior_key for prior_key in bucket if prior_key.casefold() == key.casefold()),
                None,
            )
            prior = bucket.get(prior_key) if prior_key is not None else None
            if prior is not None and _normalized_identity(prior) != _normalized_identity(value):
                metadata_conflict = True
            else:
                bucket.setdefault(key, value)

    polarities: dict[tuple[str, str, str, int | None], int] = {}
    unique_triples: dict[tuple[str, str, str, int, int | None], object] = {}
    for triple in [*a.triples, *b.triples]:
        claim = (
            _normalized_identity(triple.subject),
            triple.predicate.casefold(),
            _normalized_identity(triple.object),
            triple.source_message_id,
        )
        prior_polarity = polarities.get(claim)
        if prior_polarity is not None and prior_polarity != triple.polarity:
            metadata_conflict = True
            continue
        polarities[claim] = triple.polarity
        identity = (*claim[:3], triple.polarity, triple.source_message_id)
        prior = unique_triples.get(identity)
        if prior is None or repr(triple) < repr(prior):
            unique_triples[identity] = triple
    return ChunkResult(
        triples=list(unique_triples.values()),
        entity_type_hints=type_hints,
        entity_property_hints=property_hints,
        markers=list(a.markers) + list(b.markers),
        failed=a.failed or b.failed or metadata_conflict,
    )


def extract_chunk(
    client: LLMClient,
    text: str,
    negative_examples: str = "",
    *,
    source_records: tuple[tuple[int, str], ...] | None = None,
) -> ChunkResult:
    """Run the merged triples+markers prompt in a SINGLE LLM call and validate.

    The response is a JSON object ``{"triples": [...], "markers": [...]}``. The
    sub-arrays are fed through the exact same validators the separate
    ``extract_triples``/``extract_markers`` paths use, so behavior is identical;
    only the call count changes (one instead of two per chunk).

    Tolerates malformed output exactly like the digest path: invalid JSON or a
    wrong-shaped payload never raises.  It instead sets ``failed`` so the
    caller can hold the chunk for retry instead of marking it done. A
    transport error is caught and ALSO lands in ``failed`` — the runner's
    per-chunk ``except`` used to hold by raising; the ladder swallows the
    exception, so the exhausted call-failure branch must surface as failed.
    A clean ``{"triples": [], "markers": []}`` is a successful empty. Bare
    arrays, missing keys, and non-array values are failures: accepting any of
    them would permanently turn a provider/schema error into "nothing found".

    RETRY LADDER (2026-08-07): ingest is one-shot — nothing downstream retries
    a failed chunk, so a lost extraction is a permanent hole. The ceiling is
    payload-sized (see `_chunk_max_tokens`); when the reply still fails to
    parse and the structural cut detector fires (opens '{', unterminated),
    the SAME input is re-rolled ONCE — licensed empirically by deepseek-v4
    -flash output variance at temperature=0.0 (measured 0.3x-4.8x spread);
    NOT a Protocol guarantee, a deterministic backend makes it a wasted call.
    If still unparseable, the terminating step splits the input in half and
    extracts each half (merging partial results) — a smaller prompt converges
    deterministically and cannot loop; this path has no cache key, so the
    split is safe (unlike rollup fusion, where node_id = sha1(member_ids)).
    """
    system = build_chunk_extraction_system(negative_examples)

    def attempt(
        txt: str, allowed_ids: frozenset[int] | None = None
    ) -> tuple[ChunkResult, str | None]:
        """One pass: call, strict parse, validate. Returns (result, raw) —
        `raw` is None on call failure; `result.failed` is the ladder's
        trigger and the hold-for-retry signal."""
        request = LLMRequest(
            system=system,
            user=CHUNK_EXTRACTION_USER_TEMPLATE.format(text=txt),
            response_format="json",
            max_tokens=_chunk_max_tokens(txt),
        )
        try:
            raw = client.complete(request)
        except Exception:
            log.exception("chunk_extraction.call_failure")
            return ChunkResult(failed=True), None
        data = loads_exact_or_fenced(raw)
        if data is None:
            log.warning("chunk_extraction.parse_failure raw_len=%d",
                        len(raw) if isinstance(raw, str) else -1)
            return ChunkResult(failed=True), raw
        # A bare array (including StubLLMClient's historical ``[]`` default)
        # is not the object contract and therefore cannot authorize the
        # one-shot processed marker.
        if not isinstance(data, dict):
            log.warning("chunk_extraction.shape_failure type=%s",
                        type(data).__name__)
            return ChunkResult(failed=True), raw

        required_keys = {"triples", "markers"}
        missing = sorted(required_keys - set(data))
        extra = sorted(set(data) - required_keys)
        if missing or extra:
            event = (
                "chunk_extraction.missing_keys"
                if missing else "chunk_extraction.object_keys_failure"
            )
            log.warning("%s missing=%s extra=%s", event,
                        ",".join(missing), ",".join(extra))
            return ChunkResult(failed=True), raw
        triples_raw = data["triples"]
        markers_raw = data["markers"]
        if not isinstance(triples_raw, list) or not isinstance(markers_raw, list):
            log.warning(
                "chunk_extraction.array_shape_failure triples=%s markers=%s",
                type(triples_raw).__name__,
                type(markers_raw).__name__,
            )
            return ChunkResult(failed=True), raw
        if not all(
            combined_triple_item_is_valid(
                item, require_source_message_id=allowed_ids is not None
            )
            for item in triples_raw
        ) or not all(
            combined_marker_item_is_valid(item) for item in markers_raw
        ):
            log.warning(
                "chunk_extraction.item_validation_failure triples_returned=%d "
                "markers_returned=%d",
                len(triples_raw), len(markers_raw),
            )
            return ChunkResult(failed=True), raw
        unique_triples_raw = _consistent_unique_triples(triples_raw, allowed_ids)
        if unique_triples_raw is None:
            log.warning(
                "chunk_extraction.response_conflict triples_returned=%d",
                len(triples_raw),
            )
            return ChunkResult(failed=True), raw
        triples = triples_from_list(unique_triples_raw)
        markers = markers_from_list(markers_raw)
        if len(triples) != len(unique_triples_raw) or len(markers) != len(markers_raw):
            log.warning(
                "chunk_extraction.item_validation_failure "
                "triples_returned=%d triples_valid=%d "
                "markers_returned=%d markers_valid=%d",
                len(unique_triples_raw),
                len(triples),
                len(markers_raw),
                len(markers),
            )
            return ChunkResult(failed=True), raw
        return ChunkResult(
            triples=triples,
            entity_type_hints=entity_types_from_list(triples_raw),
            entity_property_hints=entity_properties_from_list(triples_raw),
            markers=markers,
        ), raw

    if source_records is not None:
        if not source_records or len({mid for mid, _ in source_records}) != len(
            source_records
        ):
            return ChunkResult(failed=True)
        text = "\n".join(record for _mid, record in source_records)
        full_allowed = frozenset(mid for mid, _record in source_records)
    else:
        full_allowed = None

    result, raw = attempt(text, full_allowed)
    if result.failed and raw is not None and is_ceiling_cut(raw):
        # One re-roll of the SAME input — empirical license, see docstring.
        result, raw = attempt(text, full_allowed)
    if result.failed and len(text) >= _MIN_SPLIT_CHARS:
        # Source-aware requests split only complete tagged records.  For one
        # oversized source, rebuild two valid records carrying the same id and
        # overlapping content fragments. Arbitrary character slicing could
        # truncate a tag and let the model cite a source absent from its input.
        if source_records is not None:
            if len(source_records) > 1:
                mid_record = len(source_records) // 2
                left_records = source_records[:mid_record]
                right_records = source_records[mid_record:]
            else:
                source_id, encoded = source_records[0]
                try:
                    payload = json.loads(encoded)
                    content = payload["content"]
                    if not isinstance(content, str):
                        raise ValueError
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    return ChunkResult(failed=True)
                mid_char = len(content) // 2
                left_payload = dict(payload)
                right_payload = dict(payload)
                left_end = min(len(content), mid_char + _SPLIT_OVERLAP_CHARS)
                right_start = max(0, mid_char - _SPLIT_OVERLAP_CHARS)
                left_payload["content"] = content[:left_end]
                right_payload["content"] = content[right_start:]
                for fragment, start, end in (
                    (left_payload, 0, left_end),
                    (right_payload, right_start, len(content)),
                ):
                    fragment["source_record_version"] = (
                        "hymem-claim-source-fragment-v1"
                    )
                    fragment["source_content_start"] = start
                    fragment["source_content_end"] = end
                left_records = ((source_id, json.dumps(
                    left_payload, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                )),)
                right_records = ((source_id, json.dumps(
                    right_payload, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                )),)
            left_text = "\n".join(record for _mid, record in left_records)
            right_text = "\n".join(record for _mid, record in right_records)
            left, _ = attempt(
                left_text, frozenset(mid for mid, _record in left_records)
            )
            right, _ = attempt(
                right_text, frozenset(mid for mid, _record in right_records)
            )
        else:
            mid = len(text) // 2
            left, _ = attempt(
                text[: min(len(text), mid + _SPLIT_OVERLAP_CHARS)]
            )
            right, _ = attempt(text[max(0, mid - _SPLIT_OVERLAP_CHARS) :])
        result = _merge_results(left, right)
    return result
