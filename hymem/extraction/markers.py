from __future__ import annotations

import logging
from dataclasses import dataclass

from hymem.extraction.jsonio import loads_lenient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import MARKER_SYSTEM, MARKER_USER_TEMPLATE

log = logging.getLogger("hymem.extraction.markers")

_ALLOWED_KINDS = ("correction", "preference", "rejection", "style")


@dataclass(frozen=True)
class Marker:
    kind: str
    statement: str


def extract_markers(client: LLMClient, text: str) -> list[Marker]:
    request = LLMRequest(
        system=MARKER_SYSTEM,
        user=MARKER_USER_TEMPLATE.format(text=text),
        response_format="json",
    )
    raw = client.complete(request)
    return _parse(raw)


def _parse(raw: str) -> list[Marker]:
    # MARKER_SYSTEM asks for a bare JSON array; fences/prose around it are
    # tolerated (dream 1013 — json_object mode is a request, not a contract).
    data = loads_lenient(raw, expect="array")
    if data is None:
        log.warning("markers.parse_failure raw_len=%d",
                    len(raw) if isinstance(raw, str) else -1)
        return []
    if not isinstance(data, list):
        # markers_from_list() would absorb this and return [], which is the
        # behavior we want but not the silence: a wrong-shaped reply is a
        # dropped extraction, and ingest is one-shot with nothing to retry it.
        log.warning("markers.shape_failure type=%s", type(data).__name__)
        return []
    return markers_from_list(data)


def markers_from_list(data: list) -> list[Marker]:
    """Validate an already-decoded markers array into Marker objects."""
    if not isinstance(data, list):
        return []

    markers: list[Marker] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        statement = item.get("statement")
        if kind not in _ALLOWED_KINDS:
            continue
        if not isinstance(statement, str) or not statement.strip():
            continue
        markers.append(Marker(kind=kind, statement=statement.strip()))
    return markers
