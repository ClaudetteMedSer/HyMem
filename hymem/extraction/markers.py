from __future__ import annotations

import json
from dataclasses import dataclass

from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import MARKER_SYSTEM, MARKER_USER_TEMPLATE

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
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
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
