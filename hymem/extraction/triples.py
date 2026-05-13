from __future__ import annotations

import json
from dataclasses import dataclass

from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import (
    ALLOWED_PREDICATES,
    build_triple_system,
    TRIPLE_USER_TEMPLATE,
)


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    polarity: int  # +1 or -1
    value_text: str | None = None
    value_numeric: float | None = None
    value_unit: str | None = None
    temporal_scope: str | None = None


def extract_triples(
    client: LLMClient, 
    text: str, 
    negative_examples: str = "",
) -> tuple[list[Triple], dict[str, str]]:
    """Run the locked-vocabulary triple prompt and validate the output.

    Returns parsed triples and any entity type hints extracted from the same
    LLM response.

    Anything malformed or off-vocabulary is silently dropped — the LLM is allowed
    to be wrong, but we never propagate garbage into the graph.
    """
    system = build_triple_system(negative_examples)
    request = LLMRequest(
        system=system,
        user=TRIPLE_USER_TEMPLATE.format(text=text),
        response_format="json",
    )
    raw = client.complete(request)
    return _parse(raw), extract_entity_types(raw)


def extract_entity_types(raw: str) -> dict[str, str]:
    """Extract entity type hints from the same LLM response."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, list):
        return {}

    types: dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        obj = item.get("object")
        subj_type = item.get("subject_type")
        obj_type = item.get("object_type")
        valid_types = {
            "language", "framework", "database", "service", "tool", "library",
            "file", "environment", "protocol", "container", "package_manager",
            "api", "platform", "config_file", "testing_framework", "ci_tool",
            "monitoring_tool", "identity_provider", "message_broker", "or_other_tool"
        }
        if isinstance(subject, str) and isinstance(subj_type, str) and subj_type in valid_types:
            types[subject.strip()] = subj_type
        if isinstance(obj, str) and isinstance(obj_type, str) and obj_type in valid_types:
            types[obj.strip()] = obj_type
    return types


def _parse(raw: str) -> list[Triple]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    triples: list[Triple] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")
        polarity = item.get("polarity", 1)
        if not (isinstance(subject, str) and isinstance(predicate, str) and isinstance(obj, str)):
            continue
        if predicate not in ALLOWED_PREDICATES:
            continue
        if polarity not in (1, -1):
            continue
        if not subject.strip() or not obj.strip():
            continue
        value_text = item.get("value_text")
        value_numeric = item.get("value_numeric")
        value_unit = item.get("value_unit")
        temporal_scope = item.get("temporal_scope")
        triples.append(
            Triple(
                subject=subject.strip(),
                predicate=predicate,
                object=obj.strip(),
                polarity=polarity,
                value_text=value_text.strip() if isinstance(value_text, str) and value_text.strip() else None,
                value_numeric=float(value_numeric) if isinstance(value_numeric, (int, float)) else None,
                value_unit=value_unit.strip() if isinstance(value_unit, str) and value_unit.strip() else None,
                temporal_scope=temporal_scope.strip() if isinstance(temporal_scope, str) and temporal_scope.strip() else None,
            )
        )
    return triples
