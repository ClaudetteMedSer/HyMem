from __future__ import annotations

import logging
from dataclasses import dataclass

from hymem.extraction.jsonio import loads_lenient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import (
    ALLOWED_PREDICATES,
    build_triple_system,
    TRIPLE_USER_TEMPLATE,
)

log = logging.getLogger("hymem.extraction.triples")


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
) -> tuple[list[Triple], dict[str, str], dict[str, dict[str, str]]]:
    """Run the locked-vocabulary triple prompt and validate the output.

    Returns parsed triples, any entity type hints, and any entity property
    hints (key/value pairs) extracted from the same LLM response.

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
    # Decode ONCE and fan the same list out to the three validators: identical
    # to calling `_parse`/`extract_entity_types`/`extract_entity_properties` on
    # `raw` separately (each decoded it itself), minus two duplicate
    # parse-failure warnings for a single bad reply.
    items = _as_item_list(raw)
    return (
        triples_from_list(items),
        entity_types_from_list(items),
        entity_properties_from_list(items),
    )


_VALID_TYPES = frozenset({
    "language", "framework", "database", "service", "tool", "library",
    "file", "environment", "protocol", "container", "package_manager",
    "api", "platform", "config_file", "testing_framework", "ci_tool",
    "monitoring_tool", "identity_provider", "message_broker",
    "person", "team", "project", "codebase", "or_other_tool",
})


# Caps on per-entity property metadata. The LLM is occasionally chatty;
# truncating here keeps a single bad response from bloating the table.
_MAX_PROPS_PER_ENTITY = 6
_MAX_PROP_KEY_LEN = 32
_MAX_PROP_VALUE_LEN = 64


def _as_item_list(raw: str) -> list:
    """Decode a raw LLM response into a list of items, tolerating garbage.

    Returns [] for invalid JSON or any non-array payload. The combined-extraction
    path bypasses this and feeds an already-decoded sub-array into the
    ``*_from_list`` helpers directly.

    The triple prompt asks for a bare JSON array; fences/prose around it are
    tolerated (dream 1013 — json_object mode is a request, not a contract).
    """
    data = loads_lenient(raw, expect="array")
    if data is None:
        log.warning("triples.parse_failure raw_len=%d",
                    len(raw) if isinstance(raw, str) else -1)
        return []
    if not isinstance(data, list):
        log.warning("triples.shape_failure type=%s", type(data).__name__)
        return []
    return data


def extract_entity_types(raw: str) -> dict[str, str]:
    """Extract entity type hints from the same LLM response."""
    return entity_types_from_list(_as_item_list(raw))


def entity_types_from_list(data: list) -> dict[str, str]:
    """Entity type hints from an already-decoded triples array."""
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
        if isinstance(subject, str) and isinstance(subj_type, str) and subj_type in _VALID_TYPES:
            types[subject.strip()] = subj_type
        if isinstance(obj, str) and isinstance(obj_type, str) and obj_type in _VALID_TYPES:
            types[obj.strip()] = obj_type
    return types


def extract_entity_properties(raw: str) -> dict[str, dict[str, str]]:
    """Pull `{entity: {key: value, ...}}` property hints out of the response.

    Recognises two shapes the LLM may emit per triple item:
      - ``subject_properties``/``object_properties``: an object of key/value
        strings attached to the subject/object of the triple.
      - ``properties``: a top-level mapping ``{entity_name: {key: value}}``
        keyed by the entity's surface form, useful when the same entity
        appears in multiple triples in the response.
    """
    return entity_properties_from_list(_as_item_list(raw))


def entity_properties_from_list(data: list) -> dict[str, dict[str, str]]:
    """Entity property hints from an already-decoded triples array."""
    if not isinstance(data, list):
        return {}

    props: dict[str, dict[str, str]] = {}

    def _merge(entity: object, payload: object) -> None:
        if not isinstance(entity, str) or not isinstance(payload, dict):
            return
        key_name = entity.strip()
        if not key_name:
            return
        bucket = props.setdefault(key_name, {})
        for k, v in payload.items():
            if len(bucket) >= _MAX_PROPS_PER_ENTITY:
                break
            if not isinstance(k, str) or not isinstance(v, (str, int, float, bool)):
                continue
            k_clean = k.strip().lower()
            v_clean = str(v).strip()
            if not k_clean or not v_clean:
                continue
            if len(k_clean) > _MAX_PROP_KEY_LEN or len(v_clean) > _MAX_PROP_VALUE_LEN:
                continue
            # First write wins for a given (entity, key) pair in the response;
            # later persistence does INSERT OR REPLACE so re-extractions still
            # update the row.
            bucket.setdefault(k_clean, v_clean)

    for item in data:
        if not isinstance(item, dict):
            continue
        _merge(item.get("subject"), item.get("subject_properties"))
        _merge(item.get("object"), item.get("object_properties"))
        top_level = item.get("properties")
        if isinstance(top_level, dict):
            for entity, payload in top_level.items():
                _merge(entity, payload)

    return props


def _parse(raw: str) -> list[Triple]:
    """Triples only, from a raw LLM reply. `extract_triples` decodes once and
    fans out instead; this stays for callers that only want the triples."""
    return triples_from_list(_as_item_list(raw))


def triples_from_list(data: list) -> list[Triple]:
    """Validate an already-decoded triples array into Triple objects."""
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
