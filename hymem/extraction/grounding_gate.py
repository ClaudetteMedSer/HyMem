"""Atomic source-grounding gate for the frozen Phase-1 extraction runtime."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from hymem.contrib.implementation_identity import import_time_source_sha256
from hymem.extraction.grounding import (
    GroundingContext, GroundingContractError, GroundingSource,
    MAX_TOTAL_SOURCE_CHARS, MAX_TRIPLES, build_grounding_request,
    parse_grounding_response,
)
from hymem.extraction.jsonio import loads_strict_json
from hymem.extraction.llm import LLMRequest
from hymem.extraction.triples import Triple

EXTRACTION_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)
GROUNDING_GATE_VERSION = "hymem-source-grounding-gate-v1"
_IMPORTED_GUARD = (
    GroundingContext, GroundingContractError, GroundingSource,
    build_grounding_request, parse_grounding_response, loads_strict_json,
    LLMRequest, Triple, replace,
)


def grounding_gate_support_integrity() -> bool:
    return _IMPORTED_GUARD == (
        GroundingContext, GroundingContractError, GroundingSource,
        build_grounding_request, parse_grounding_response, loads_strict_json,
        LLMRequest, Triple, replace,
    )


_GROUNDING_GATE_INTEGRITY_FUNCTION = grounding_gate_support_integrity


class GroundingGateError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise GroundingGateError(code)


def _prefix(context: dict, start: int, content: str) -> int:
    limit = context.get("applies_through_source_content_end")
    if type(limit) is not int or not start < limit:
        _fail("context:scope")
    return min(len(content), limit - start)


def _source(record: tuple[int, str], contexts: tuple[tuple[int, str], ...]) -> GroundingSource:
    sid, encoded = record
    payload = loads_strict_json(encoded)
    content = payload["content"]
    start = payload.get("source_content_start", 0)
    if type(start) is not int:
        _fail("source:offset")
    regions: list[GroundingContext] = []

    def add(region: str, context: dict, text: str, owned_limit: int,
            origin: dict | None = None, parent_region: str | None = None,
            parent_limit: int | None = None) -> None:
        origin = origin or payload
        if not text or not 0 < owned_limit <= len(content):
            _fail("context:scope")
        regions.append(GroundingContext(
            region, text, owned_limit, origin.get("source_role"),
            origin.get("source_peer_id"), origin.get("source_created_at"),
            origin.get("source_message_id"), parent_region, parent_limit,
        ))

    def attached(context: dict, header_region: str, prelude_region: str,
                 *, outer_limit: int | None = None, origin: dict | None = None,
                 parent_region: str | None = None,
                 parent_limit: int | None = None) -> None:
        limit = outer_limit if outer_limit is not None else _prefix(context, start, content)
        add(header_region, context, context["content"], limit, origin,
            parent_region, parent_limit)
        if "prelude_content" in context:
            add(prelude_region, context, context["prelude_content"], limit, origin,
                parent_region, parent_limit)

    fragment = payload.get("source_fragment_context")
    if fragment is not None:
        attached(fragment, "header", "prelude")
    boundary = payload.get("source_boundary_context")
    if boundary is not None:
        add("boundary", boundary, boundary["content"], _prefix(boundary, start, content))

    for index, (_, encoded_context) in enumerate(contexts):
        prior = loads_strict_json(encoded_context)
        if prior["context_for_source_message_id"] != sid:
            continue
        outer = _prefix(prior, start, content)
        add(f"conversation_{index}", prior, prior["content"], outer, prior)
        nested = prior.get("source_fragment_context")
        if nested is not None:
            # A table header in the prior turn must apply to its quoted tail
            # and that turn must still apply to this owned prefix.
            prior_start = prior["source_content_start"]
            nested_parent_limit = min(
                len(prior["content"]),
                nested["applies_through_source_content_end"] - prior_start,
            )
            if nested_parent_limit < 1:
                _fail("context:nested_scope")
            attached(nested, f"conversation_{index}_header",
                     f"conversation_{index}_prelude", outer_limit=outer,
                     origin=prior, parent_region=f"conversation_{index}",
                     parent_limit=nested_parent_limit)
    return GroundingSource(
        sid, content, tuple(regions), payload.get("source_role"),
        payload.get("source_peer_id"), payload.get("source_created_at"),
    )


def _batches(triples: list[Triple], sources: dict[int | None, GroundingSource]):
    start = 0
    while start < len(triples):
        end = start
        ids: list[int | None] = []
        size = 0
        while end < len(triples) and end - start < MAX_TRIPLES:
            sid = triples[end].source_message_id
            if sid not in sources:
                _fail("triple:source")
            if sid not in ids:
                source = sources[sid]
                length = len(source.content) + sum(len(c.content) for c in source.contexts)
                if size + length > MAX_TOTAL_SOURCE_CHARS:
                    break
                ids.append(sid)
                size += length
            end += 1
        if end == start:
            _fail("sources:total_bounds")
        yield triples[start:end], tuple(sources[sid] for sid in ids)
        start = end


def _canonical_key(triple: Triple) -> tuple[str, str, str, int | None]:
    return (" ".join(triple.subject.casefold().split()), triple.predicate,
            " ".join(triple.object.casefold().split()), triple.source_message_id)


def _check_corrections(original: list[Triple], corrected: list[Triple]) -> None:
    seen: dict[tuple[str, str, str, int | None], int] = {}
    for old, new in zip(original, corrected, strict=True):
        key = _canonical_key(new)
        prior = seen.get(key)
        if prior is not None:
            _fail("correction:collision" if prior == new.polarity else "correction:conflict")
        seen[key] = new.polarity
        if old != new and any(
            getattr(old, field) != getattr(new, field)
            for field in ("subject", "object", "polarity", "source_message_id",
                          "value_text", "value_numeric", "value_unit", "temporal_scope")
        ):
            _fail("correction:mutation")


def ground_triples(
    triples: list[Triple], source_records: tuple[tuple[int, str], ...] | None,
    context_records: tuple[tuple[int, str], ...], legacy_text: str,
    invoke: Callable[[LLMRequest, bool], str],
) -> list[Triple]:
    """Judge all triples, then recheck the full corrected list once if needed."""
    if not triples:
        return []
    if source_records is None:
        sources = {None: GroundingSource(None, legacy_text)}
    else:
        try:
            sources = {sid: _source((sid, record), context_records)
                       for sid, record in source_records}
        except GroundingGateError:
            raise
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise GroundingGateError("source:invalid") from exc
    current = list(triples)
    corrected_any = False
    for recheck in (False, True):
        proposed: list[Triple] = []
        for group, group_sources in _batches(current, sources):
            try:
                request, batch = build_grounding_request(group, group_sources)
                if recheck:
                    raw = invoke(request, True)
                else:
                    raw = invoke(request, False)
                review = parse_grounding_response(raw, batch, allow_corrections=not recheck)
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
        _check_corrections(current, proposed)
        current = proposed
    raise AssertionError("unreachable")
