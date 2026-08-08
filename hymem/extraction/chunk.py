from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hymem.extraction.jsonio import loads_lenient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.markers import Marker, markers_from_list
from hymem.extraction.prompts import (
    CHUNK_EXTRACTION_USER_TEMPLATE,
    build_chunk_extraction_system,
)
from hymem.extraction.triples import (
    Triple,
    entity_properties_from_list,
    entity_types_from_list,
    triples_from_list,
)

log = logging.getLogger("hymem.extraction.chunk")


@dataclass(frozen=True)
class ChunkResult:
    """Everything one combined extraction call yields for a chunk.

    ``failed`` separates "the extraction did not complete" from "the extraction
    completed and this chunk genuinely holds nothing". Both look like empty
    lists here, but they must NOT be treated alike downstream: the caller marks
    a chunk done in ``processed_chunks``, and marking a failure done makes it a
    permanent hole (nothing re-reads a chunk under the same prompt version).
    Only a clean parse that yielded nothing is a real empty.
    """
    triples: list[Triple] = field(default_factory=list)
    entity_type_hints: dict[str, str] = field(default_factory=dict)
    entity_property_hints: dict[str, dict[str, str]] = field(default_factory=dict)
    markers: list[Marker] = field(default_factory=list)
    failed: bool = False


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
    caller can hold the chunk for retry instead of marking it done (see
    ``ChunkResult``). A transport error still RAISES — the runner's per-chunk
    ``except`` skips persistence entirely, which holds the chunk the same way.
    """
    system = build_chunk_extraction_system(negative_examples)
    request = LLMRequest(
        system=system,
        user=CHUNK_EXTRACTION_USER_TEMPLATE.format(text=text),
        response_format="json",
    )
    raw = client.complete(request)

    # The combined prompt asks for a top-level JSON object; fences/prose around
    # it are tolerated (dream 1013 — json_object mode is a request, not a
    # contract).
    data = loads_lenient(raw, expect="object")
    if data is None:
        log.warning("chunk_extraction.parse_failure raw_len=%d",
                    len(raw) if isinstance(raw, str) else -1)
        return ChunkResult(failed=True)

    # A bare array (e.g. a stub LLM's "[]" default) or any non-object payload
    # yields an empty result rather than crashing.
    if not isinstance(data, dict):
        # An empty array is that documented stub default and a routine "nothing
        # here", so it stays quiet — and it is a real empty, not a failure.
        # Any OTHER shape is a real reply we dropped.
        if data != []:
            log.warning("chunk_extraction.shape_failure type=%s",
                        type(data).__name__)
            return ChunkResult(failed=True)
        return ChunkResult()

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
    )
