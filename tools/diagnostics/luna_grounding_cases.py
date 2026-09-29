"""Fixed, invented semantic controls for extraction grounding.

These cases are independent of the extraction prompt and require no model call.
The runner supplies each case's ``source_records`` to ``extract_chunk`` and
passes the resulting ChunkResult to ``grade_case``. Only core graph claims are
graded; optional entity hints and markers do not change the claim verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CoreTriple:
    subject: str
    predicate: str
    object: str
    polarity: int
    source_message_id: int


@dataclass(frozen=True)
class GroundingCase:
    case_id: str
    source_records: tuple[tuple[int, str], ...]
    expected: tuple[CoreTriple, ...]

    @property
    def text(self) -> str:
        return "\n".join(encoded for _, encoded in self.source_records)


def _record(message_id: int, content: str) -> tuple[int, str]:
    payload = {
        "content": content,
        "source_created_at": "2026-09-28T00:00:00.000Z",
        "source_message_id": message_id,
        "source_peer_id": "fixture-author",
        "source_record_version": "hymem-claim-source-v2",
        "source_role": "user",
        "source_session_id": "luna-grounding-controls-v1",
        "source_workspace_id": None,
    }
    return message_id, json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _triple(subject: str, predicate: str, object_: str, polarity: int, source_id: int) -> CoreTriple:
    return CoreTriple(subject, predicate, object_, polarity, source_id)


# Preserve ordering, IDs, and wording: this is a versioned semantic control set.
CASES: tuple[GroundingCase, ...] = (
    GroundingCase("preference_only", (_record(101, "Mira prefers PostgreSQL."),),
                  (_triple("Mira", "prefers", "PostgreSQL", 1, 101),)),
    GroundingCase("actual_use_only", (_record(102, "Owen currently uses SQLite."),),
                  (_triple("Owen", "uses", "SQLite", 1, 102),)),
    GroundingCase("preference_and_use", (_record(103, "Nila prefers PostgreSQL and currently uses PostgreSQL."),),
                  (_triple("Nila", "prefers", "PostgreSQL", 1, 103),
                   _triple("Nila", "uses", "PostgreSQL", 1, 103))),
    GroundingCase("preference_not_using", (_record(104, "Arun prefers Redis but does not use Redis."),),
                  (_triple("Arun", "prefers", "Redis", 1, 104),
                   _triple("Arun", "uses", "Redis", -1, 104))),
    GroundingCase("prefers_a_uses_b", (_record(105, "Bea prefers PostgreSQL but currently uses SQLite."),),
                  (_triple("Bea", "prefers", "PostgreSQL", 1, 105),
                   _triple("Bea", "uses", "SQLite", 1, 105))),
    GroundingCase("considering_recommending", (_record(106, "Cleo prefers PostgreSQL. She is considering SQLite and recommends that Dan consider SQLite."),),
                  (_triple("Cleo", "prefers", "PostgreSQL", 1, 106),)),
    GroundingCase("stopped_use", (_record(107, "Eli no longer uses MongoDB."),),
                  (_triple("Eli", "uses", "MongoDB", -1, 107),)),
    GroundingCase("table_preference_use", (_record(108, "| Person | Preference | Current use |\n| --- | --- | --- |\n| Faye | PostgreSQL | SQLite |"),),
                  (_triple("Faye", "prefers", "PostgreSQL", 1, 108),
                   _triple("Faye", "uses", "SQLite", 1, 108))),
    GroundingCase("cross_paragraph_pronoun", (_record(109, "Gita prefers PostgreSQL.\n\nShe currently uses SQLite."),),
                  (_triple("Gita", "prefers", "PostgreSQL", 1, 109),
                   _triple("Gita", "uses", "SQLite", 1, 109))),
    GroundingCase("use_and_negative_preference", (_record(110, "Hana currently uses SQLite. Hana explicitly does not prefer SQLite."),),
                  (_triple("Hana", "uses", "SQLite", 1, 110),
                   _triple("Hana", "prefers", "SQLite", -1, 110))),
)

CASE_IDS = tuple(case.case_id for case in CASES)
_CASE_BY_ID = {case.case_id: case for case in CASES}


def _field(item: Any, name: str) -> Any:
    return item.get(name) if type(item) is dict else getattr(item, name, None)


def _parse_core(item: Any) -> CoreTriple | None:
    values = tuple(_field(item, field) for field in CoreTriple.__dataclass_fields__)
    subject, predicate, object_, polarity, source_id = values
    if not (type(subject) is str and subject.strip()
            and type(predicate) is str and predicate.strip()
            and type(object_) is str and object_.strip()
            and type(polarity) is int and polarity in (-1, 1)
            and type(source_id) is int and source_id > 0):
        return None
    # Match the extractor's identity comparison: case and repeated whitespace
    # in endpoints are cosmetic. Underscores and aliases remain significant.
    endpoint = lambda value: " ".join(value.casefold().split())
    return CoreTriple(endpoint(subject), predicate, endpoint(object_), polarity, source_id)


def grade_case(case_id: str, result: Any) -> dict[str, Any]:
    """Grade one final extraction result, returning only bounded metadata.

    Unknown case IDs are programming errors. Malformed extraction output fails
    closed; boolean source IDs and polarities never pass as integers.
    """
    case = _CASE_BY_ID[case_id]
    failed = _field(result, "failed")
    raw_triples = _field(result, "triples")
    markers = _field(result, "markers")
    type_hints = _field(result, "entity_type_hints")
    property_hints = _field(result, "entity_property_hints")
    valid_container = type(raw_triples) in (list, tuple)
    parsed = [_parse_core(item) for item in raw_triples] if valid_container else []
    invalid_count = sum(item is None for item in parsed) + (not valid_container)
    actual = [item for item in parsed if item is not None]
    actual_set = set(actual)
    expected_set = {CoreTriple(" ".join(t.subject.casefold().split()), t.predicate,
                               " ".join(t.object.casefold().split()),
                               t.polarity, t.source_message_id) for t in case.expected}
    missing_count = len(expected_set - actual_set)
    extra_count = len(actual_set - expected_set)
    duplicate_count = len(actual) - len(actual_set)
    passed = (failed is False and invalid_count == 0 and missing_count == 0
              and extra_count == 0 and duplicate_count == 0
              and len(actual) == len(case.expected))
    return {
        "case_id": case.case_id,
        "passed": passed,
        "extraction_failed": failed is not False,
        "expected_count": len(case.expected),
        "actual_count": len(actual),
        "missing_count": missing_count,
        "extra_count": extra_count,
        "duplicate_count": duplicate_count,
        "invalid_count": int(invalid_count),
        "markers_count": len(markers) if type(markers) in (list, tuple) else 0,
        "type_hint_count": len(type_hints) if type(type_hints) is dict else 0,
        "property_hint_count": len(property_hints) if type(property_hints) is dict else 0,
    }
