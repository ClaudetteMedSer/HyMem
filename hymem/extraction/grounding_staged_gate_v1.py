"""Inactive atomic gate for the staged, source-bound grounding contract.

``invoke(request, batch, stage, recheck)`` owns transport and the caller's
existing budget. ``stage`` is ``"original"`` with a v4 ClassificationBatch or
``"alternatives"`` with a staged AlternativesBatch; ``recheck`` is true only
for original requests in the single whole-list correction recheck. The gate
makes no retries and propagates callback exceptions unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Literal

from hymem.contrib.implementation_identity import import_time_source_sha256
from hymem.extraction import grounding_gate as _source_gate
from hymem.extraction import grounding_staged_v1 as _staged
from hymem.extraction.grounding_classification_v4 import ClassificationBatch
from hymem.extraction.grounding_v2 import GroundingContext, GroundingContractError, GroundingSource
from hymem.extraction.llm import LLMRequest
from hymem.extraction.triples import Triple

EXTRACTION_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)
GROUNDING_GATE_VERSION = "hymem-source-grounding-staged-gate-v1"
GroundingGateError = _source_gate.GroundingGateError
Stage = Literal["original", "alternatives"]
StageBatch = ClassificationBatch | _staged.AlternativesBatch
StageCallback = Callable[[LLMRequest, StageBatch, Stage, bool], str]

_IMPORTED_GUARD = (
    replace, ClassificationBatch, GroundingContext, GroundingContractError,
    GroundingSource, LLMRequest, Triple, _source_gate.GroundingGateError,
    _source_gate._source, _source_gate._batches, _source_gate._check_corrections,
    _staged.AlternativesBatch, _staged.build_original_request,
    _staged.validate_original_request, _staged.parse_original_response,
    _staged.build_alternatives_request, _staged.validate_alternatives_request,
    _staged.parse_staged_responses,
)


def grounding_gate_support_integrity() -> bool:
    return _IMPORTED_GUARD == (
        replace, ClassificationBatch, GroundingContext, GroundingContractError,
        GroundingSource, LLMRequest, Triple, _source_gate.GroundingGateError,
        _source_gate._source, _source_gate._batches, _source_gate._check_corrections,
        _staged.AlternativesBatch, _staged.build_original_request,
        _staged.validate_original_request, _staged.parse_original_response,
        _staged.build_alternatives_request, _staged.validate_alternatives_request,
        _staged.parse_staged_responses,
    )


_GROUNDING_GATE_INTEGRITY_FUNCTION = grounding_gate_support_integrity


def _fail(code: str) -> None:
    raise GroundingGateError(code)


def _v2_source(source: _source_gate.GroundingSource) -> GroundingSource:
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


def _contract(call: Callable[[], object]) -> object:
    try:
        return call()
    except GroundingContractError as exc:
        _fail("contract:" + exc.code.replace(":", "_"))


def ground_triples(
    triples: list[Triple],
    source_records: tuple[tuple[int, str], ...] | None,
    context_records: tuple[tuple[int, str], ...],
    legacy_text: str,
    invoke: StageCallback,
) -> list[Triple]:
    """Accept all claims atomically, allowing one predicate-only correction pass.

    Each batch's originals are judged before that batch's alternatives.
    Ambiguity rejects immediately; a validated negative alone enables the
    alternatives request for that batch.
    Any correction causes one original-only recheck of the entire corrected
    list, including batches with no correction. No recheck correction is allowed.
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
    for recheck in (False, True):
        proposed: list[Triple] = []
        corrected_any = False
        for group, group_sources in _source_gate._batches(current, sources):
            request, batch = _contract(lambda: _staged.build_original_request(group, group_sources))
            _contract(lambda: _staged.validate_original_request(request, batch))
            original_raw = invoke(request, batch, "original", recheck)
            original = _contract(lambda: _staged.parse_original_response(original_raw, batch))
            # Ambiguity does not authorize an alternatives request, even when
            # another row in this batch is negative.
            if "ambiguous" in original.states:
                _fail("verdict:uncertain")
            if recheck and original.negative_indices:
                _fail("verdict:unsupported")
            alternative_raw = None
            if original.negative_indices:
                alternative_request, alternative_batch = _contract(
                    lambda: _staged.build_alternatives_request(batch, original_raw))
                _contract(lambda: _staged.validate_alternatives_request(
                    alternative_request, alternative_batch))
                alternative_raw = invoke(alternative_request, alternative_batch,
                                         "alternatives", False)
            review = _contract(lambda: _staged.parse_staged_responses(
                batch, original_raw, alternative_raw, allow_corrections=not recheck))
            for triple, verdict in zip(group, review.verdicts, strict=True):
                if verdict.status in {"unsupported", "uncertain"}:
                    _fail("verdict:" + verdict.status)
                if verdict.status == "replace_predicate":
                    if recheck:
                        _fail("contract:verdict_correction")
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
