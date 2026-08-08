"""Merged: retry ladder (2026-08-07) + failed-flag hold-for-retry (2026-08-08).

The ladder (payload-sized ceiling, structural-cut re-roll, split terminating
step) prevents truncation holes. The failed flag makes the runner HOLD a
chunk whose extraction did not succeed instead of marking it done — parse
failure, wrong shape, and (the ladder's swallowed) transport failure all land
in failed=True; only a clean empty (or ``[]``, the documented stub default)
is a real empty and takes the mark.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hymem.extraction.jsonio import is_ceiling_cut, loads_lenient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.markers import Marker, markers_from_list
from hymem.extraction.prompts import (
    CHUNK_EXTRACTION_USER_TEMPLATE,
    build_chunk_extraction_system,
)
from hymem.extraction.triples import (
    entity_properties_from_list,
    entity_types_from_list,
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
    return ChunkResult(
        triples=list(a.triples) + list(b.triples),
        entity_type_hints={**a.entity_type_hints, **b.entity_type_hints},
        entity_property_hints={**a.entity_property_hints, **b.entity_property_hints},
        markers=list(a.markers) + list(b.markers),
        failed=a.failed or b.failed,
    )


def extract_chunk(
    client: LLMClient,
    text: str,
    negative_examples: str = "",
) -> ChunkResult:
    """Run the merged triples+markers prompt in a SINGLE LLM call and validate.

    The response is a JSON object ``{"triples": [...], "markers": [...]}``. The
    sub-arrays are fed through the exact same validators the separate
    ``extract_triples``/``extract_markers`` paths use, so behavior is identical;
    only the call count changes (one instead of two per chunk).

    Tolerates malformed output exactly like the digest path: invalid JSON, a
    non-object payload, or missing keys yields empty lists/dicts, never raises.
    An unparseable or wrong-shaped reply additionally sets ``failed`` so the
    caller can hold the chunk for retry instead of marking it done. A
    transport error is caught and ALSO lands in ``failed`` — the runner's
    per-chunk ``except`` used to hold by raising; the ladder swallows the
    exception, so the exhausted call-failure branch must surface as failed.
    ``[]`` (StubLLMClient's documented default) and a clean empty object are
    NOT failed — real empties, marked done.

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

    def attempt(txt: str) -> tuple[ChunkResult, str | None]:
        """One pass: call, lenient parse, validate. Returns (result, raw) —
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
        data = loads_lenient(raw, expect="object")
        if data is None:
            log.warning("chunk_extraction.parse_failure raw_len=%d",
                        len(raw) if isinstance(raw, str) else -1)
            return ChunkResult(failed=True), raw
        # A bare array (e.g. a stub LLM's "[]" default) or any non-object
        # payload yields an empty result rather than crashing.
        if not isinstance(data, dict):
            # An empty array is that documented stub default and a routine
            # "nothing here", so it stays quiet. Any OTHER shape is a real
            # reply we dropped — failed, so the chunk is held for retry.
            if data != []:
                log.warning("chunk_extraction.shape_failure type=%s",
                            type(data).__name__)
                return ChunkResult(failed=True), raw
            return ChunkResult(), raw

        # MEASURE-ONLY (pending flip): a dict carrying NEITHER key is never a
        # compliant reply — the prompt asks for both, and a chunk that genuinely
        # holds nothing comes back as {"triples": [], "markers": []}, which is the
        # real empty and stays quiet here. Neither-key is a wrong-schema reply, a
        # JSON-wrapped refusal, or a bare {} — always a lost extraction, i.e. the
        # same permanent-hole class parse/shape failures used to be. ONE key is
        # benign (a model that found triples and omitted an empty markers array)
        # and must not trip this.
        #
        # It is logged rather than flagged because the rate is unmeasured, and
        # turning on a second unknown-rate retry class while the re-extraction
        # queue drains would mix its abandonment counts into that verdict. Flip to
        # `return ChunkResult(failed=True)` once the rate is read off a few dreams.
        if "triples" not in data and "markers" not in data:
            log.warning("chunk_extraction.missing_keys keys=%d", len(data))
        triples_raw = data.get("triples", [])
        markers_raw = data.get("markers", [])
        return ChunkResult(
            triples=triples_from_list(triples_raw),
            entity_type_hints=entity_types_from_list(triples_raw),
            entity_property_hints=entity_properties_from_list(triples_raw),
            markers=markers_from_list(markers_raw),
        ), raw

    result, raw = attempt(text)
    if result.failed and raw is not None and is_ceiling_cut(raw):
        # One re-roll of the SAME input — empirical license, see docstring.
        result, raw = attempt(text)
    if result.failed and len(text) >= _MIN_SPLIT_CHARS:
        # Terminating step: split, extract each half, merge partials.
        mid = len(text) // 2
        left, _ = attempt(text[:mid])
        right, _ = attempt(text[mid:])
        result = _merge_results(left, right)
    return result
