from __future__ import annotations

from hymem.contrib.implementation_identity import import_time_source_sha256

EXTRACTION_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)

import logging
import math
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
    # Exact message cited by the extractor.  Standalone/legacy parsing leaves
    # this unset; the production combined extractor requires and validates it
    # against the tagged records actually present in that request.
    source_message_id: int | None = None


def extract_triples(
    client: LLMClient,
    text: str,
) -> tuple[list[Triple], dict[str, str], dict[str, dict[str, str]]]:
    """Run the locked-vocabulary triple prompt and validate the output.

    Returns parsed triples, any entity type hints, and any entity property
    hints (key/value pairs) extracted from the same LLM response.

    Anything malformed or off-vocabulary is silently dropped — the LLM is allowed
    to be wrong, but we never propagate garbage into the graph.
    """
    system = build_triple_system()
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
    "person", "team", "project", "codebase", "place", "organization",
    "product", "vehicle", "activity", "event", "document",
    "or_other_entity",
})

_COMBINED_REQUIRED_KEYS = frozenset({"subject", "predicate", "object", "polarity"})
_COMBINED_OPTIONAL_KEYS = frozenset({
    "value_text", "value_numeric", "value_unit", "temporal_scope",
    "subject_type", "object_type", "subject_properties", "object_properties",
    "source_message_id",
})

_COMBINED_MAX_PROPERTIES = 4


def normalize_combined_triple_item(
    item: object, *, require_source_message_id: bool = False
) -> tuple[dict | None, tuple[str, ...], tuple[str, ...]]:
    """Validate one combined-extractor item and normalize harmless drift.

    The return value is ``(item, errors, normalizations)``.  Core claim
    semantics (subject/predicate/object/polarity) and the source citation are
    fail-closed.  Optional enrichment is not allowed to destroy a valid claim:
    malformed type/property/value hints are removed, and harmless surrounding
    whitespace/case drift is normalized deterministically.  Error and
    normalization codes contain field names only -- never model-supplied
    values -- so callers may safely persist them as diagnostics.
    """
    if not isinstance(item, dict):
        return None, ("item:not_object",), ()

    errors: list[str] = []
    normalizations: list[str] = []
    normalized: dict = {}
    keys = set(item)
    for key in sorted(_COMBINED_REQUIRED_KEYS - keys):
        errors.append(f"{key}:missing")
    if keys - _COMBINED_REQUIRED_KEYS - _COMBINED_OPTIONAL_KEYS:
        errors.append("item:unexpected_keys")

    for key in ("subject", "predicate", "object"):
        value = item.get(key)
        if not isinstance(value, str):
            errors.append(f"{key}:not_string")
            continue
        stripped = value.strip()
        if not stripped:
            errors.append(f"{key}:empty")
            continue
        normalized[key] = stripped
    predicate = normalized.get("predicate")
    if predicate is not None:
        canonical_predicate = predicate.casefold()
        if canonical_predicate not in ALLOWED_PREDICATES:
            errors.append("predicate:not_allowed")
        else:
            if canonical_predicate != predicate:
                normalizations.append("predicate:normalized_case")
            normalized["predicate"] = canonical_predicate

    polarity = item.get("polarity")
    if (
        isinstance(polarity, bool)
        or not isinstance(polarity, int)
        or polarity not in (1, -1)
    ):
        errors.append("polarity:not_plus_or_minus_one")
    else:
        normalized["polarity"] = polarity

    source_message_id = item.get("source_message_id")
    if isinstance(source_message_id, str):
        # Some OpenAI-compatible providers serialize an integer schema field as
        # a JSON string. Only the canonical positive-decimal spelling is
        # losslessly equivalent: no sign, whitespace, leading zero, decimal,
        # exponent, or Unicode digit normalization is accepted here.
        if (
            source_message_id
            and len(source_message_id) <= 19
            and source_message_id[0] in "123456789"
            and all(character in "0123456789" for character in source_message_id)
        ):
            parsed_source_message_id = int(source_message_id)
            if str(parsed_source_message_id) == source_message_id:
                source_message_id = parsed_source_message_id
                normalizations.append(
                    "source_message_id:normalized_decimal_string"
                )
    if require_source_message_id and "source_message_id" not in item:
        errors.append("source_message_id:missing")
    elif source_message_id is not None and (
        isinstance(source_message_id, bool)
        or not isinstance(source_message_id, int)
        or source_message_id < 1
    ):
        errors.append("source_message_id:not_positive_integer")
    elif "source_message_id" in item:
        normalized["source_message_id"] = source_message_id

    if errors:
        return None, tuple(errors), ()

    for key in ("value_text", "value_unit", "temporal_scope"):
        if key not in item:
            continue
        value = item[key]
        if not isinstance(value, str) or not value.strip():
            normalizations.append(f"{key}:removed_invalid_optional")
            continue
        normalized[key] = value.strip()

    if "value_numeric" in item:
        number = item["value_numeric"]
        finite = False
        if not isinstance(number, bool) and isinstance(number, (int, float)):
            try:
                finite = math.isfinite(float(number))
            except (OverflowError, ValueError):
                finite = False
        if finite:
            normalized["value_numeric"] = number
        else:
            normalizations.append("value_numeric:removed_invalid_optional")

    for key in ("subject_type", "object_type"):
        if key not in item:
            continue
        value = item[key]
        candidate = value.strip().casefold() if isinstance(value, str) else ""
        if candidate in _VALID_TYPES:
            normalized[key] = candidate
        else:
            normalizations.append(f"{key}:removed_invalid_optional")

    for key in ("subject_properties", "object_properties"):
        if key not in item:
            continue
        properties = item[key]
        if not isinstance(properties, dict):
            normalizations.append(f"{key}:removed_invalid_optional")
            continue
        candidates: dict[str, str] = {}
        conflicts: set[str] = set()
        removed = False
        for prop_key, prop_value in properties.items():
            if not isinstance(prop_key, str) or not isinstance(prop_value, str):
                removed = True
                continue
            normalized_key = prop_key.strip().lower()
            normalized_value = prop_value.strip()
            if (
                not normalized_key
                or not normalized_value
                or len(normalized_key) > _MAX_PROP_KEY_LEN
                or len(normalized_value) > _MAX_PROP_VALUE_LEN
            ):
                removed = True
                continue
            prior = candidates.get(normalized_key)
            if prior is not None and prior != normalized_value:
                conflicts.add(normalized_key)
                removed = True
                continue
            candidates[normalized_key] = normalized_value
        for conflict in conflicts:
            candidates.pop(conflict, None)
        ordered = sorted(candidates.items())
        if len(ordered) > _COMBINED_MAX_PROPERTIES:
            ordered = ordered[:_COMBINED_MAX_PROPERTIES]
            removed = True
        if ordered:
            normalized[key] = dict(ordered)
        if removed or not ordered:
            normalizations.append(f"{key}:normalized_optional")

    return normalized, (), tuple(normalizations)


def combined_triple_item_is_valid(
    item: object, *, require_source_message_id: bool = False
) -> bool:
    """Whether the item has valid core semantics after safe hint cleanup."""
    normalized, errors, _normalizations = normalize_combined_triple_item(
        item, require_source_message_id=require_source_message_id
    )
    return normalized is not None and not errors


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
        if isinstance(polarity, bool) or polarity not in (1, -1):
            continue
        if not subject.strip() or not obj.strip():
            continue
        value_text = item.get("value_text")
        value_numeric = item.get("value_numeric")
        value_unit = item.get("value_unit")
        temporal_scope = item.get("temporal_scope")
        source_message_id = item.get("source_message_id")
        if value_text is not None and not isinstance(value_text, str):
            continue
        if value_numeric is not None and (
            isinstance(value_numeric, bool)
            or not isinstance(value_numeric, (int, float))
        ):
            continue
        if value_unit is not None and not isinstance(value_unit, str):
            continue
        if temporal_scope is not None and not isinstance(temporal_scope, str):
            continue
        if source_message_id is not None and (
            isinstance(source_message_id, bool)
            or not isinstance(source_message_id, int)
            or source_message_id < 1
        ):
            continue
        triples.append(
            Triple(
                subject=subject.strip(),
                predicate=predicate,
                object=obj.strip(),
                polarity=polarity,
                value_text=value_text.strip() if isinstance(value_text, str) and value_text.strip() else None,
                value_numeric=(
                    float(value_numeric)
                    if isinstance(value_numeric, (int, float))
                    and not isinstance(value_numeric, bool)
                    else None
                ),
                value_unit=value_unit.strip() if isinstance(value_unit, str) and value_unit.strip() else None,
                temporal_scope=temporal_scope.strip() if isinstance(temporal_scope, str) and temporal_scope.strip() else None,
                source_message_id=source_message_id,
            )
        )
    return triples
