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
    """Everything one combined extraction call yields for a chunk."""
    triples: list[Triple] = field(default_factory=list)
    entity_type_hints: dict[str, str] = field(default_factory=dict)
    entity_property_hints: dict[str, dict[str, str]] = field(default_factory=dict)
    markers: list[Marker] = field(default_factory=list)


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
        return ChunkResult()

    # A bare array (e.g. a stub LLM's "[]" default) or any non-object payload
    # yields an empty result rather than crashing.
    if not isinstance(data, dict):
        # An empty array is that documented stub default and a routine "nothing
        # here", so it stays quiet. Any OTHER shape is a real reply we dropped,
        # and ingest is one-shot — nothing will retry it, so say so.
        if data != []:
            log.warning("chunk_extraction.shape_failure type=%s",
                        type(data).__name__)
        return ChunkResult()

    triples_raw = data.get("triples", [])
    markers_raw = data.get("markers", [])
    return ChunkResult(
        triples=triples_from_list(triples_raw),
        entity_type_hints=entity_types_from_list(triples_raw),
        entity_property_hints=entity_properties_from_list(triples_raw),
        markers=markers_from_list(markers_raw),
    )
