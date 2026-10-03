"""Non-authoritative original extraction occurrences, never confidence signals.

Hashes detect corruption, not authenticity. The complete original set belongs
to an evidence revision; its reduced representative must never be recaptured.
Timestamp text is deliberately opaque, including malformed historical clocks.
"""
from __future__ import annotations

import hashlib
import json
import math

DOMAIN = "hymem-evidence-extraction-audit"
VERSION = 1
FIELDS = (
    "chunk_id", "polarity", "surface_subject", "surface_object", "value_text",
    "value_numeric", "value_unit", "temporal_scope", "source_role",
    "source_peer_id", "source_workspace_id", "evidence_kind", "evidence_weight",
    "weight_source", "extraction_prompt_version", "extracted_at",
    "source_message_id", "source_session_id", "source_created_at", "source_event_at",
    "source_coverage_chunk_id", "source_coverage_version", "provenance_status",
    "interpretation_key",
)
PROJECTIONS = (
    "chunk_id", "source_message_id", "source_session_id",
    "source_coverage_chunk_id", "source_coverage_version",
)
COLUMNS = ("evidence_id", "occurrence_hash", "payload_json", *PROJECTIONS)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate extraction audit JSON key")
        result[key] = value
    return result


def validate(payload):
    if not isinstance(payload, dict) or set(payload) != {
        "domain", "version", "original_edge", "extraction"
    }:
        raise ValueError("invalid extraction audit payload fields")
    if payload["domain"] != DOMAIN or type(payload["version"]) is not int or payload["version"] != VERSION:
        raise ValueError("unsupported extraction audit domain/version")
    edge = payload["original_edge"]
    if type(edge) is not list or len(edge) != 3 or any(type(v) is not str for v in edge):
        raise ValueError("invalid extraction audit original edge")
    extraction = payload["extraction"]
    if not isinstance(extraction, dict) or set(extraction) != set(FIELDS):
        raise ValueError("invalid extraction audit extraction fields")
    required_text = {"chunk_id", "evidence_kind", "weight_source", "provenance_status", "interpretation_key"}
    integers = {"polarity", "evidence_weight", "source_message_id"}
    for key, value in extraction.items():
        if key in integers:
            if value is None and key == "source_message_id":
                continue
            if type(value) is not int or not -(2**63) <= value < 2**63:
                raise ValueError("invalid extraction audit integer")
        elif key == "value_numeric":
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("invalid extraction audit number")
        elif type(value) is not str and (value is not None or key in required_text):
            raise ValueError("invalid extraction audit text")
    coverage = [extraction[k] for k in ("source_message_id", "source_coverage_chunk_id", "source_coverage_version")]
    if any(v is not None for v in coverage[1:]) and not all(v is not None for v in coverage):
        raise ValueError("incomplete extraction audit coverage reference")
    return payload


def encode(payload):
    return json.dumps(validate(payload), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def decode(text):
    if type(text) is not str:
        raise ValueError("extraction audit payload must be JSON text")
    payload = json.loads(text, object_pairs_hook=_object)
    if encode(payload) != text:
        raise ValueError("extraction audit JSON is not canonical")
    return payload


def record(evidence_id, payload):
    text = encode(payload)
    return dict(zip(COLUMNS, (
        evidence_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        *(payload["extraction"][k] for k in PROJECTIONS),
    )))


def validate_record(row):
    if set(row) != set(COLUMNS) or type(row["evidence_id"]) is not int or row["evidence_id"] <= 0:
        raise ValueError("invalid extraction audit owner/fields")
    payload = decode(row["payload_json"])
    expected = record(row["evidence_id"], payload)
    if any(type(row[k]) is not type(expected[k]) or row[k] != expected[k] for k in COLUMNS):
        raise ValueError("extraction audit hash/projection mismatch")
    return payload


def sql_valid(*values):
    try:
        validate_record(dict(zip(COLUMNS, values)))
        return 1
    except (ValueError, TypeError, OverflowError, RecursionError):
        return 0


def original(edge, extraction):
    return {"domain": DOMAIN, "version": VERSION, "original_edge": list(edge),
            "extraction": {k: extraction[k] for k in FIELDS}}


def carrier_record(conn, evidence_id):
    row = conn.execute("SELECT * FROM kg_evidence WHERE id=?", (evidence_id,)).fetchone()
    edge = conn.execute("SELECT subject_canonical,predicate,object_canonical FROM knowledge_graph WHERE id=?",
                        (row["edge_id"],)).fetchone()
    return record(evidence_id, original(edge, row))


def insert(conn, row):
    validate_record(row)
    return conn.execute(
        "INSERT INTO kg_evidence_extraction_audit(" + ",".join(COLUMNS)
        + ") VALUES (" + ",".join("?" for _ in COLUMNS) + ") "
        "ON CONFLICT(evidence_id,occurrence_hash) DO NOTHING",
        tuple(row[k] for k in COLUMNS),
    ).rowcount


def capture(conn, evidence_id):
    if not conn.execute("SELECT 1 FROM kg_evidence_extraction_audit WHERE evidence_id=?",
                        (evidence_id,)).fetchone():
        return insert(conn, carrier_record(conn, evidence_id))
    return 0


def capture_edge(conn, edge_id):
    from hymem.core.db import evidence_history_mutation
    with evidence_history_mutation(conn):
        for row in conn.execute("SELECT id FROM kg_evidence WHERE edge_id=?", (edge_id,)).fetchall():
            capture(conn, row["id"])


def union(conn, keeper, donor):
    capture(conn, keeper)
    capture(conn, donor)
    for row in conn.execute("SELECT * FROM kg_evidence_extraction_audit WHERE evidence_id=?", (donor,)).fetchall():
        insert(conn, {**dict(row), "evidence_id": keeper})


def merge_incoming(conn, evidence_id, incoming, *, preserve_local=True):
    """Union pre-reduction originals, leaving an unchanged bare carrier bare."""
    stored = conn.execute("SELECT 1 FROM kg_evidence_extraction_audit WHERE evidence_id=?", (evidence_id,)).fetchone()
    inserted = 0
    if not stored:
        carrier = carrier_record(conn, evidence_id)
        if all(row["payload_json"] == carrier["payload_json"] for row in incoming):
            return 0
        if preserve_local:
            inserted = capture(conn, evidence_id)
    return inserted + sum(insert(conn, {**row, "evidence_id": evidence_id}) for row in incoming)
