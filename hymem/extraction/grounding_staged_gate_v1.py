"""Source-bound gate for the staged claim grounding contract.

``invoke(request, batch, stage, recheck)`` owns transport and the caller's
existing budget. ``stage`` is ``"original"`` with a v4 ClassificationBatch or
``"alternatives"`` with a staged AlternativesBatch; ``recheck`` is true only
for original requests in the single whole-list correction recheck. The default
gate is fail-atomic. Diagnostic recovery makes bounded, source-preserving
evidence rechecks and rejects claims that still cannot attest their evidence.
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
_DIAGNOSTIC_ATTESTATION_CODES = frozenset({
    "evidence:quote_missing",
    "evidence:context_missing",
    "support:unreferenced_evidence",
})

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
    *,
    diagnostic_grounding_recovery: bool = False,
    rejection_sink: Callable[[str], None] | None = None,
) -> list[Triple]:
    """Accept only source-grounded claims under an explicit diagnostic policy.

    Each batch's originals are judged before that batch's alternatives.
    Canonical runs fail atomically on negative, ambiguous, or invalid verdicts.
    Diagnostic runs omit fully assessed negative/ambiguous claims. Evidence
    attestation errors trigger source-preserving batch isolation and one retry
    for a singleton; a still-invalid singleton is explicitly rejected. Other
    contract, provider, and budget failures always remain atomic.
    """
    if not _GROUNDING_GATE_INTEGRITY_FUNCTION() or not grounding_gate_support_integrity():
        _fail("support:integrity")
    if type(diagnostic_grounding_recovery) is not bool:
        raise TypeError("diagnostic_grounding_recovery must be bool")
    if rejection_sink is not None and not callable(rejection_sink):
        raise TypeError("rejection_sink must be callable")
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

    rejected: list[str] = []

    def assess_group(
        group: list[Triple], group_sources: tuple[GroundingSource, ...],
        *, recheck: bool, singleton_retried: bool = False,
    ) -> list[tuple[Triple, Triple]]:
        def attestation_failure(exc: GroundingContractError) -> list[tuple[Triple, Triple]]:
            if not (diagnostic_grounding_recovery
                    and exc.code in _DIAGNOSTIC_ATTESTATION_CODES):
                _fail("contract:" + exc.code.replace(":", "_"))
            if len(group) == 1:
                if not singleton_retried:
                    return assess_group(group, group_sources, recheck=recheck,
                                        singleton_retried=True)
                rejected.append("invalid_" + exc.code.replace(":", "_"))
                return []
            midpoint = len(group) // 2
            retained: list[tuple[Triple, Triple]] = []
            for part in (group[:midpoint], group[midpoint:]):
                source_ids = {item.source_message_id for item in part}
                part_sources = tuple(source for source in group_sources
                                     if source.source_message_id in source_ids)
                retained.extend(assess_group(part, part_sources, recheck=recheck))
            return retained

        request, batch = _contract(
            lambda: _staged.build_original_request(group, group_sources))
        _contract(lambda: _staged.validate_original_request(request, batch))
        original_raw = invoke(request, batch, "original", recheck)
        try:
            original = _staged.parse_original_response(original_raw, batch)
        except GroundingContractError as exc:
            return attestation_failure(exc)
        if recheck:
            retained = []
            for triple, state in zip(group, original.states, strict=True):
                if state == "supported":
                    retained.append((triple, triple))
                elif diagnostic_grounding_recovery:
                    rejected.append("recheck_" + state)
                else:
                    _fail("verdict:" + ("unsupported" if state == "not_established"
                                        else "uncertain"))
            return retained
        # Preserve the canonical fail-atomic shortcut. A diagnostic batch must
        # still assess its negative rows before projecting the supported ones.
        if not diagnostic_grounding_recovery and "ambiguous" in original.states:
            _fail("verdict:uncertain")
        alternative_raw = None
        if original.negative_indices:
            alternative_request, alternative_batch = _contract(
                lambda: _staged.build_alternatives_request(batch, original_raw))
            _contract(lambda: _staged.validate_alternatives_request(
                alternative_request, alternative_batch))
            alternative_raw = invoke(alternative_request, alternative_batch,
                                     "alternatives", False)
        try:
            review = _staged.parse_staged_responses(
                batch, original_raw, alternative_raw, allow_corrections=True)
        except GroundingContractError as exc:
            return attestation_failure(exc)
        retained = []
        for triple, verdict in zip(group, review.verdicts, strict=True):
            if verdict.status in {"unsupported", "uncertain"}:
                if diagnostic_grounding_recovery:
                    rejected.append("verdict_" + verdict.status)
                    continue
                _fail("verdict:" + verdict.status)
            corrected = (replace(triple, predicate=verdict.predicate)
                         if verdict.status == "replace_predicate" else triple)
            retained.append((triple, corrected))
        return retained

    current = list(triples)
    for recheck in (False, True):
        pairs = []
        for group, group_sources in _source_gate._batches(current, sources):
            pairs.extend(assess_group(group, group_sources, recheck=recheck))
        proposed = [after for _before, after in pairs]
        if recheck or not any(before != after for before, after in pairs):
            if rejection_sink is not None:
                for reason in rejected:
                    rejection_sink(reason)
            return proposed
        _source_gate._check_corrections(
            [before for before, _after in pairs], proposed)
        current = proposed
    raise AssertionError("unreachable")
