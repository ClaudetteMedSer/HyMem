"""Inactive atomic gate for the claim-first grounding classification v3 contract.

The gate reconstructs trusted source and context records, then binds each
callback to its exact bounded classification batch.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from hymem.contrib.implementation_identity import import_time_source_sha256
from hymem.extraction import grounding_gate as _source_gate
from hymem.extraction.grounding_classification_v3 import (
    ClassificationBatch,
    GroundingContractError,
    build_grounding_request,
    parse_grounding_response,
    validate_request,
)
from hymem.extraction.llm import LLMRequest
from hymem.extraction.grounding_v2 import GroundingContext, GroundingSource
from hymem.extraction.triples import Triple

EXTRACTION_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)
GROUNDING_GATE_VERSION = "hymem-source-grounding-classification-gate-v3"
GroundingGateError = _source_gate.GroundingGateError

_IMPORTED_GUARD = (
    ClassificationBatch, GroundingContractError, build_grounding_request,
    parse_grounding_response, validate_request, LLMRequest, Triple, replace,
    GroundingContext, GroundingSource,
    _source_gate.GroundingGateError, _source_gate._source,
    _source_gate._batches, _source_gate._check_corrections,
)


def grounding_gate_support_integrity() -> bool:
    return _IMPORTED_GUARD == (
        ClassificationBatch, GroundingContractError, build_grounding_request,
        parse_grounding_response, validate_request, LLMRequest, Triple, replace,
        GroundingContext, GroundingSource,
        _source_gate.GroundingGateError, _source_gate._source,
        _source_gate._batches, _source_gate._check_corrections,
    )


_GROUNDING_GATE_INTEGRITY_FUNCTION = grounding_gate_support_integrity


def _fail(code: str) -> None:
    raise GroundingGateError(code)


def _v2_source(source: _source_gate.GroundingSource) -> GroundingSource:
    """Carry every reconstructed field across the v1/v2 dataclass boundary."""
    contexts = tuple(GroundingContext(
        context.region, context.content, context.owned_prefix_chars,
        context.source_role, context.source_peer_id, context.source_created_at,
        context.source_message_id, context.applies_to_region,
        context.applies_to_prefix_chars,
    ) for context in source.contexts)
    return GroundingSource(
        source.source_message_id, source.content, contexts, source.source_role,
        source.source_peer_id, source.source_created_at,
    )


def ground_triples(
    triples: list[Triple],
    source_records: tuple[tuple[int, str], ...] | None,
    context_records: tuple[tuple[int, str], ...],
    legacy_text: str,
    invoke: Callable[[LLMRequest, ClassificationBatch, bool], str],
) -> list[Triple]:
    """Return the whole accepted list, with one full correction recheck if needed.

    A rejected batch rejects the entire result. The callback owns transport and
    the caller's existing call budget; this gate makes no retries or extra calls.
    """
    if not _GROUNDING_GATE_INTEGRITY_FUNCTION() or not grounding_gate_support_integrity():
        _fail("support:integrity")
    if not triples:
        return []
    if source_records is None:
        sources = {None: GroundingSource(None, legacy_text)}
    else:
        try:
            sources = {sid: _v2_source(_source_gate._source((sid, record), context_records))
                       for sid, record in source_records}
        except GroundingGateError:
            raise
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise GroundingGateError("source:invalid") from exc

    current = list(triples)
    corrected_any = False
    for recheck in (False, True):
        proposed: list[Triple] = []
        for group, group_sources in _source_gate._batches(current, sources):
            try:
                request, batch = build_grounding_request(group, group_sources)
                validate_request(request, batch)
            except GroundingContractError as exc:
                _fail("contract:" + exc.code.replace(":", "_"))
            raw = invoke(request, batch, recheck)
            try:
                review = parse_grounding_response(
                    raw, batch, allow_corrections=not recheck,
                )
            except GroundingContractError as exc:
                _fail("contract:" + exc.code.replace(":", "_"))
            for triple, verdict in zip(group, review.verdicts, strict=True):
                if verdict.status in {"unsupported", "uncertain"}:
                    _fail("verdict:" + verdict.status)
                if verdict.status == "replace_predicate":
                    corrected_any = True
                    proposed.append(replace(triple, predicate=verdict.predicate))
                else:
                    proposed.append(triple)
        if recheck:
            return current
        if not corrected_any:
            return current
        _source_gate._check_corrections(current, proposed)
        current = proposed
    raise AssertionError("unreachable")
