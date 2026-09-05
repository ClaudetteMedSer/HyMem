"""Memory export / import (improv item G).

Emits the canonical HyMem state as JSON Lines — one record per line, each
``{"type": <kind>, "record": {...}}`` — preceded by a ``_meta`` header. The
format is stable and human-inspectable, suitable for backups, project-to-
project migration, and feeding external tooling. Embedded-first: this runs
in-process against the SQLite file, independent of the optional MCP/Honcho
server adapters.

Import is additive for disjoint identities and exactly idempotent for an
unchanged snapshot. A deterministic identity that already exists with
different canonical state fails closed; it is never silently ignored or used
to rebind imported provenance. Best used against a fresh database. The autoincrement
ids of knowledge_graph / profile_entries are dropped on import so they don't
collide with rows already in the target — those tables dedupe on their natural
unique keys ((s,p,o) and text).
"""

from __future__ import annotations

import json
import logging
import hashlib
import math
import os
import re
import sqlite3
import tempfile
import contextlib
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from hymem import redaction
from hymem.core import db as core_db
from hymem.core.message_records import (
    MESSAGE_PROVENANCE_RECORD_VERSION,
    MESSAGE_RECORD_VERSION,
    canonical_message_record,
    encode_message_record,
    encode_provenance_message_record,
)
from hymem.core.time import (
    earliest_timestamp_spelling,
    event_clock_is_valid,
    normalize_iso_timestamp,
    validate_timestamp_order,
)
from hymem.dreaming.digest import (
    digest_generation_is_recognized,
    digest_retry_state_is_valid,
)
from hymem.dreaming import canonicalize
from hymem.dreaming import evidence as evidence_ledger
from hymem.dreaming.aggregation_provenance import (
    BoundSourceOccurrence,
    EPISODE_SOURCE_MANIFEST_VERSION,
    combine_source_occurrences,
    load_episode_source_manifest,
    source_manifest_hash,
)
from hymem.dreaming.lossless import (
    covered_messages_after,
    validate_message_coverage_artifact,
)
from hymem.dreaming.facts import (
    FACT_MAX_ACTIVE_ITEMS_PER_OUTCOME,
    FACT_MAX_HISTORY_ITEMS_PER_OUTCOME,
    FACT_MAX_LIFECYCLE_EVENTS_PER_OUTCOME,
    FACT_MAX_REVISIONS_PER_OUTCOME,
    FACT_MAX_SOURCES_PER_OUTCOME,
    FACT_SOURCE_MANIFEST_VERSION,
    canonical_fact_items,
    fact_input_hash,
    fact_item_key,
    fact_publication_version_is_recognized,
    fact_result_hash,
    fact_slice_key,
    facts_generation_is_recognized,
    facts_retry_state_is_valid,
    load_fact_outcome_source_manifest,
    validate_fact_items,
)
from hymem.dreaming.message_coverage import LOSSLESS_READ_VERSIONS
from hymem.extraction.jsonio import loads_strict_json
from hymem.extraction.prompts import ALLOWED_PREDICATES
from hymem.dreaming.user_profile import (
    ProfileExtraction,
    SINGLE_VALUED_SLOTS,
    _interval_timestamp,
    _redact_profile_key,
    enforce_profile_redaction_policy,
    persist_user_profile,
    profile_generation_is_recognized,
    profile_retry_state_is_valid,
    reconcile_profile_intervals,
    validate_profile_items,
)

log = logging.getLogger("hymem.portability")

# v2 (schema v15): edge records gained valid_at / invalid_at. v3 (schema v37)
# carries durable message-retention proofs after their raw source is gone. v4
# preserves v38 artifact kind, rolling-summary provenance, and digest cursors;
# v5 adds the atomic episode-publication generation marker. v6 carries the
# typed profile itself and durable profile provenance. v7 carries the complete
# claim-source ledger, extraction authority, and lifecycle history. v8 adds
# workspace-qualified external peer/session provenance without changing v7.
# v9 carries exact episode source manifests. v10 carries authoritative fact
# extraction outcomes, exact source occurrences, all revisions/lifecycle
# events, and the current projection. Fact embeddings and FTS shadows remain
# rebuildable local caches. Aggregation nodes remain a
# reproducible cache and are rebuilt after import from those portable episodes.
# Incomplete profile
# staging is deliberately omitted and its cursor is rewound for safe replay.
# Import stays backward-compatible: older exports simply omit newer record
# kinds/columns.
EXPORT_VERSION = 10
_MAX_SQLITE_ROWID = 2**63 - 1
_ROWID_RESERVE_HEADROOM = 1_000_000

# (kind, table, columns) in export order. Import re-orders so a row's
# referenced session always lands first.
_V6_EXPORT_SPEC: list[tuple[str, str, list[str]]] = [
    ("session", "sessions", [
        "id", "started_at", "ended_at", "summary", "summary_source",
        "auto_summary", "auto_summary_message_id",
        "auto_summary_partial_message_id", "auto_summary_message_offset",
        "coverage_message_id", "digest_cursor_message_id",
        "digest_cursor_partial_message_id", "digest_cursor_offset",
        "digest_cursor_prompt_version", "digest_published_generation",
        "digest_retry_count", "digest_retry_config_version",
        "digest_quarantined",
        "digested_prompt_version", "facts_message_id",
        "profile_prompt_version", "profile_cursor_message_id",
        "profile_cursor_partial_message_id", "profile_cursor_offset",
        "profile_cursor_prompt_version", "profile_published_generation",
        "profile_retry_count", "profile_retry_config_version",
        "profile_quarantined",
        "episodes_prompt_version",
    ]),
    ("chunk", "chunks", [
        "id", "session_id", "start_message_id", "end_message_id",
        "salience_reason", "text", "chunk_kind", "created_at",
    ]),
    ("message_retention_coverage", "message_retention_coverage", [
        "message_id", "source_session_id", "source_role", "source_created_at",
        "chunk_id", "message_content_hash", "hash_version", "record_version",
        "coverage_version", "created_at",
    ]),
    ("user_profile_fact", "user_profile", [
        "slot", "slot_key", "value", "evidence_message_id", "confidence",
        "valid_at", "invalid_at", "created_at", "source_message_id",
        "source_session_id", "source_created_at",
    ]),
    ("episode", "episodes", [
        "id", "session_id", "title", "summary", "participants",
        "start_message_id", "end_message_id", "outcome", "key_entities",
        "digest_slice_key", "digest_generation", "created_at",
    ]),
    ("procedure", "procedures", [
        "id", "session_id", "name", "description", "steps", "triggers",
        "entities_involved", "confidence", "status", "created_at",
    ]),
    ("edge", "knowledge_graph", [
        "id", "subject_canonical", "predicate", "object_canonical",
        "pos_evidence", "neg_evidence", "first_seen", "last_seen",
        "last_reinforced", "valid_at", "invalid_at", "status", "derived",
    ]),
    ("profile_entry", "profile_entries", [
        "id", "kind", "text", "pos_evidence", "neg_evidence",
        "first_seen", "last_updated",
    ]),
]

_V7_CLAIM_SPEC: list[tuple[str, str, list[str]]] = [
    ("entity_alias", "entity_aliases", ["alias", "canonical"]),
    ("chunk_source_manifest", "chunks", [
        "id", "source_manifest_version", "source_manifest_count",
    ]),
    ("chunk_message_source", "chunk_message_sources", [
        "chunk_id", "ordinal", "source_message_id", "source_session_id",
        "source_coverage_chunk_id", "source_coverage_version",
    ]),
    ("claim_extraction_outcome", "kg_claim_extraction_outcomes", [
        "chunk_id", "prompt_version", "prompt_generation", "result_hash",
        "succeeded_at",
    ]),
    ("edge_evidence", "kg_evidence", [
        "id", "edge_id", "chunk_id", "polarity", "surface_subject",
        "surface_object", "value_text", "value_numeric", "value_unit",
        "temporal_scope", "source_role", "evidence_kind",
        "evidence_weight", "weight_source", "extraction_prompt_version",
        "extracted_at", "published_at", "source_message_id", "source_session_id",
        "source_created_at", "source_event_at", "source_coverage_chunk_id",
        "source_coverage_version", "provenance_status", "interpretation_key",
        "revision", "is_current", "superseded_at", "superseded_reason",
    ]),
    ("edge_evidence_signal", "kg_evidence_signals", [
        "id", "edge_id", "signal_key", "signal_kind", "polarity",
        "evidence_weight", "counts_toward_confidence", "details", "created_at",
    ]),
    ("claim_observation", "kg_claim_observations", [
        "chunk_id", "edge_id", "source_session_id", "source_message_id",
        "evidence_kind", "polarity", "prompt_version", "prompt_generation",
        "evidence_id", "interpretation_key", "observed_at",
    ]),
    ("edge_lifecycle", "kg_edge_lifecycle", [
        "id", "edge_id", "event_key", "event_kind", "direction", "event_at",
        "source_evidence_id", "dependency_count", "details", "created_at",
    ]),
    ("lifecycle_dependency", "kg_lifecycle_dependencies", [
        "lifecycle_id", "evidence_id",
    ]),
]

_V7_EXPORT_SPEC = [*_V6_EXPORT_SPEC, *_V7_CLAIM_SPEC]


def _v8_columns(kind: str, columns: list[str]) -> list[str]:
    """Extend frozen v7 records without weakening old exact-key validation."""
    result = list(columns)
    if kind == "session":
        result.insert(result.index("id") + 1, "source_workspace_id")
    elif kind == "message_retention_coverage":
        index = result.index("source_role") + 1
        result[index:index] = ["source_peer_id", "source_workspace_id"]
    elif kind == "edge_evidence":
        index = result.index("source_role") + 1
        result[index:index] = ["source_peer_id", "source_workspace_id"]
    return result


_V8_EXPORT_SPEC: list[tuple[str, str, list[str]]] = []
for _kind, _table, _columns in _V7_EXPORT_SPEC:
    _V8_EXPORT_SPEC.append((_kind, _table, _v8_columns(_kind, _columns)))
    if _kind == "session":
        _V8_EXPORT_SPEC.extend([
            ("peer", "peers", [
                "id", "workspace_id", "role", "metadata", "registered_at",
            ]),
            ("session_peer", "session_peers", [
                "session_id", "workspace_id", "peer_id", "configuration",
                "added_at",
            ]),
        ])


def _v9_columns(kind: str, columns: list[str]) -> list[str]:
    """Extend frozen v8 episode records with publication metadata."""

    result = list(columns)
    if kind == "episode":
        index = result.index("created_at")
        result[index:index] = [
            "source_manifest_version", "source_manifest_count",
            "source_manifest_hash", "source_manifest_complete",
        ]
    return result


_V9_EXPORT_SPEC: list[tuple[str, str, list[str]]] = []
for _kind, _table, _columns in _V8_EXPORT_SPEC:
    _V9_EXPORT_SPEC.append((_kind, _table, _v9_columns(_kind, _columns)))
    if _kind == "episode":
        _V9_EXPORT_SPEC.append((
            "episode_source_occurrence",
            "episode_source_occurrences",
            [
                "episode_id", "ordinal", "source_message_id",
                "source_session_id", "source_role", "source_peer_id",
                "source_workspace_id", "source_created_at",
                "source_coverage_chunk_id", "source_coverage_version",
                "source_content_hash",
            ],
        ))


def _v10_columns(kind: str, columns: list[str]) -> list[str]:
    """Extend sessions with the lossless fact cursor/retry contract."""

    result = list(columns)
    if kind == "session":
        index = result.index("facts_message_id") + 1
        result[index:index] = [
            "facts_cursor_message_id", "facts_cursor_partial_message_id",
            "facts_cursor_offset", "facts_cursor_prompt_version",
            "facts_retry_count", "facts_retry_config_version",
            "facts_quarantined",
        ]
    return result


_V10_EXPORT_SPEC: list[tuple[str, str, list[str]]] = [
    (kind, table, _v10_columns(kind, columns))
    for kind, table, columns in _V9_EXPORT_SPEC
]
_V10_EXPORT_SPEC.extend([
    ("fact_extraction_outcome", "fact_extraction_outcomes", [
        "slice_key", "session_id", "prompt_version", "input_hash",
        "cursor_before_message_id", "cursor_before_partial_message_id",
        "cursor_before_offset", "cursor_after_message_id",
        "cursor_after_partial_message_id", "cursor_after_offset",
        "generation", "outcome_status", "result_hash",
        "source_manifest_version", "source_manifest_count",
        "source_manifest_hash", "source_manifest_complete", "succeeded_at",
    ]),
    ("fact_extraction_source_occurrence", "fact_extraction_source_occurrences", [
        "slice_key", "ordinal", "source_message_id", "source_session_id",
        "source_role", "source_peer_id", "source_workspace_id",
        "source_created_at", "source_coverage_chunk_id",
        "source_coverage_version", "source_content_hash",
    ]),
    ("fact_extraction_revision", "fact_extraction_revisions", [
        "slice_key", "generation", "prompt_version", "outcome_status",
        "result_hash", "succeeded_at",
    ]),
    ("narrative_fact", "narrative_facts", [
        "session_id", "start_message_id", "end_message_id", "text",
        "fact_date", "entities", "prompt_version", "valid_at", "invalid_at",
        "source_outcome_key", "fact_key", "current_generation",
        "lifecycle_status", "created_at",
    ]),
    # The on-wire lifecycle uses the stable fact authority key, never a local
    # INTEGER PRIMARY KEY. Collection/import special-case this joined shape.
    ("narrative_fact_lifecycle", "narrative_fact_lifecycle", [
        "source_outcome_key", "fact_key", "generation", "direction",
        "event_at", "prompt_version", "result_hash", "recorded_at",
    ]),
])

_EXPORT_SPEC = _V10_EXPORT_SPEC
_V6_TABLE_BY_KIND = {kind: table for kind, table, _ in _V6_EXPORT_SPEC}
_V6_COLS_BY_KIND = {kind: tuple(cols) for kind, _table, cols in _V6_EXPORT_SPEC}
_V7_TABLE_BY_KIND = {kind: table for kind, table, _ in _V7_EXPORT_SPEC}
_V7_COLS_BY_KIND = {kind: tuple(cols) for kind, _table, cols in _V7_EXPORT_SPEC}
_V8_TABLE_BY_KIND = {kind: table for kind, table, _ in _V8_EXPORT_SPEC}
_V8_COLS_BY_KIND = {kind: tuple(cols) for kind, _table, cols in _V8_EXPORT_SPEC}
_V9_TABLE_BY_KIND = {kind: table for kind, table, _ in _V9_EXPORT_SPEC}
_V9_COLS_BY_KIND = {kind: tuple(cols) for kind, _table, cols in _V9_EXPORT_SPEC}
_V10_TABLE_BY_KIND = {kind: table for kind, table, _ in _V10_EXPORT_SPEC}
_V10_COLS_BY_KIND = {kind: tuple(cols) for kind, _table, cols in _V10_EXPORT_SPEC}
_TABLE_BY_KIND = _V10_TABLE_BY_KIND
_COLS_BY_KIND = _V10_COLS_BY_KIND
# Sessions must import before rows that FK-reference them.
_IMPORT_ORDER = [
    "session", "peer", "session_peer", "chunk",
    "message_retention_coverage", "user_profile_fact",
    "episode", "episode_source_occurrence", "procedure", "edge", "profile_entry",
]
# Autoincrement-id tables: drop the id on import so it can't collide with rows
# already present; they dedupe on their natural unique key instead.
_DROP_ID_ON_IMPORT = {"edge", "profile_entry"}
_SESSION_FACT_FIELDS = {
    "facts_message_id", "facts_cursor_message_id",
    "facts_cursor_partial_message_id", "facts_cursor_offset",
    "facts_cursor_prompt_version", "facts_retry_count",
    "facts_retry_config_version", "facts_quarantined",
}

_PROFILE_SLOTS = frozenset({
    "role", "name", "employer", "location", "language", "relationship",
    "possession", "age_birthday", "health_condition", "recurring_activity",
})
_NORMALIZED_EVENT_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z$"
)
_CLAIM_RESULT_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_LEGACY_INTERPRETATION_RE = re.compile(r"^legacy-row:[0-9]+$")
_LOCAL_LEGACY_SIGNAL_RE = re.compile(r"^legacy:(positive|negative):[0-9]+$")
_LOCAL_RUNTIME_SIGNAL_RE = re.compile(r"^edge:[0-9]+:polarity:(-?1)$")


def _normalized_wire_event(
    value: object, *, allow_legacy_unknown: bool = False
) -> str:
    """Use the shared timestamp policy for portable ordering and validation."""
    try:
        return normalize_iso_timestamp(value, context="portable timestamp")
    except ValueError:
        if allow_legacy_unknown:
            return "0001-01-01T00:00:00.000Z"
        raise


def _portable_source_event(value: object) -> str:
    """Canonical source coordinate, with explicit ancient legacy-unknown.

    Exact raw NULL/malformed message metadata can survive from direct SQL and
    pre-v40 stores. It is portable only as the conservative year-one sentinel;
    this helper is deliberately not used for transaction timestamps.
    """
    try:
        return normalize_iso_timestamp(value, context="portable claim source")
    except ValueError:
        return "0001-01-01T00:00:00.000Z"


def _semantic_interpretation_key(record: dict) -> str:
    return evidence_ledger._interpretation_key(
        polarity=int(record["polarity"]),
        evidence_weight=int(record["evidence_weight"]),
        weight_source=record["weight_source"],
        source_role=record.get("source_role"),
        surface_subject=record.get("surface_subject"),
        surface_object=record.get("surface_object"),
        value_text=record.get("value_text"),
        value_numeric=record.get("value_numeric"),
        value_unit=record.get("value_unit"),
        temporal_scope=record.get("temporal_scope"),
    )


def _portable_interpretation_key(record: dict) -> str:
    value = record.get("interpretation_key")
    if (
        record.get("provenance_status") == "legacy_unattributed"
        and isinstance(value, str)
        and (
            _LOCAL_LEGACY_INTERPRETATION_RE.fullmatch(value)
            or value in {"legacy-migrated-v1", "legacy-unspecified"}
        )
    ):
        return _semantic_interpretation_key(record)
    return value


def _portable_signal_key(record: dict) -> str:
    value = record.get("signal_key")
    if not isinstance(value, str):
        return value
    match = _LOCAL_LEGACY_SIGNAL_RE.fullmatch(value)
    if match:
        return f"legacy:{match.group(1)}"
    match = _LOCAL_RUNTIME_SIGNAL_RE.fullmatch(value)
    if match:
        return f"runtime-unattributed:polarity:{match.group(1)}"
    return value


def _wire_claim_result_hash(
    observations: list[dict], edge_by_wire: dict[int, dict]
) -> str:
    """Hash one wire chunk's observations without local numeric handles."""
    return evidence_ledger.claim_result_hash(
        (
            edge_by_wire[int(record["edge_id"])]["subject_canonical"],
            edge_by_wire[int(record["edge_id"])]["predicate"],
            edge_by_wire[int(record["edge_id"])]["object_canonical"],
            record["source_session_id"],
            int(record["source_message_id"]),
            record["evidence_kind"],
            int(record["polarity"]),
            record["interpretation_key"],
        )
        for record in observations
    )


def _portable_record_sort_key(record: dict) -> str:
    """Return a total, rowid-independent order for one wire record."""
    return json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _collect_v10_records(conn) -> dict[str, list[dict]]:
    """Collect one self-consistent current snapshot and canonicalize keys."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for kind, table, cols in _V10_EXPORT_SPEC:
        if kind == "chunk_source_manifest":
            where = " WHERE source_manifest_version IS NOT NULL"
        elif kind == "narrative_fact":
            # v26 rows have only a numeric range and are deliberately not
            # promoted or exported as authoritative memory.
            where = " WHERE source_outcome_key IS NOT NULL"
        elif kind == "edge":
            # Inference products are reproducible cache, not durable memory.
            # Exporting them without their complete path provenance lets a
            # forged derived row become queryable after restore.
            where = " WHERE derived = 0"
        else:
            where = ""
        if kind == "narrative_fact_lifecycle":
            rows = conn.execute(
                "SELECT f.source_outcome_key,f.fact_key,l.generation,"
                "l.direction,l.event_at,l.prompt_version,l.result_hash,"
                "l.recorded_at FROM narrative_fact_lifecycle l "
                "JOIN narrative_facts f ON f.id=l.fact_id "
                "WHERE f.source_outcome_key IS NOT NULL "
                "ORDER BY f.source_outcome_key,f.fact_key,l.generation"
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {', '.join(cols)} FROM {table}{where} ORDER BY rowid"
            ).fetchall()
        grouped[kind] = [{column: row[column] for column in cols} for row in rows]

    interpretation_by_wire_id: dict[int, str] = {}
    for record in grouped["edge_evidence"]:
        record["interpretation_key"] = _portable_interpretation_key(record)
        interpretation_by_wire_id[int(record["id"])] = record["interpretation_key"]
    for record in grouped["claim_observation"]:
        mapped = interpretation_by_wire_id.get(int(record["evidence_id"]))
        if mapped is not None:
            record["interpretation_key"] = mapped
    for record in grouped["edge_evidence_signal"]:
        record["signal_key"] = _portable_signal_key(record)

    # Table rowids reflect local insertion/import order and are not semantic.
    # Kind order remains the FK-safe wire contract, while records within each
    # kind use a canonical content order so converged stores produce reproducible
    # backups and checksums.
    for records in grouped.values():
        records.sort(key=_portable_record_sort_key)

    return grouped


def _wire_int(value: object, *, minimum: int | None = None) -> bool:
    return bool(
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _wire_number(value: object, *, minimum: float, maximum: float) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(number) and minimum <= number <= maximum


def _wire_text(value: object, *, nullable: bool = False, nonempty: bool = False) -> bool:
    if value is None:
        return nullable
    return isinstance(value, str) and (not nonempty or bool(value.strip()))


def _wire_json_array(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        decoded = loads_strict_json(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(decoded, list)


def _invalid_wire(kind: str, field: str) -> ValueError:
    return ValueError(f"portable {kind} record has invalid {field}")


def _upgrade_pre_v8_peer_fields(grouped: dict[str, list[dict]]) -> None:
    """Map old wire records to explicit unknown external provenance.

    This runs only after the old version's exact key sets were checked. A v7
    export therefore cannot smuggle peer fields, and absence is never inferred
    from a role or a coincidentally registered peer in the destination.
    """
    for record in grouped.get("session", []):
        record["source_workspace_id"] = None
    for kind in ("message_retention_coverage", "edge_evidence"):
        for record in grouped.get(kind, []):
            record["source_peer_id"] = None
            record["source_workspace_id"] = None


def _upgrade_pre_v9_episode_fields(grouped: dict[str, list[dict]]) -> None:
    """Quarantine legacy episodes without inferring from numeric ranges."""

    for record in grouped.get("episode", []):
        record["source_manifest_version"] = None
        record["source_manifest_count"] = 0
        record["source_manifest_hash"] = None
        record["source_manifest_complete"] = 0


def _upgrade_pre_v10_fact_fields(grouped: dict[str, list[dict]]) -> None:
    """Rewind old unproven fact watermarks without range inference.

    v9 and older exported ``facts_message_id`` but no facts, outcomes, or
    occurrence manifests. Carrying that watermark into a restored store would
    permanently skip source bytes while exposing no facts, so absence of v10
    authority always means a conservative replay from the beginning.
    """

    for record in grouped.get("session", []):
        record["facts_message_id"] = None
        record["facts_cursor_message_id"] = None
        record["facts_cursor_partial_message_id"] = None
        record["facts_cursor_offset"] = 0
        record["facts_cursor_prompt_version"] = None
        record["facts_retry_count"] = 0
        record["facts_retry_config_version"] = None
        record["facts_quarantined"] = 0


def _validate_v9_records(grouped: dict[str, list[dict]]) -> None:
    """Validate exact episode source publication entirely before import."""

    episodes: dict[str, dict] = {}
    for record in grouped.get("episode", []):
        episode_id = record.get("id")
        if not _wire_text(episode_id, nonempty=True) or episode_id in episodes:
            raise ValueError("portable episode contains a duplicate identity")
        episodes[str(episode_id)] = record
        complete = record.get("source_manifest_complete")
        count = record.get("source_manifest_count")
        version = record.get("source_manifest_version")
        digest = record.get("source_manifest_hash")
        if (
            not _wire_int(complete, minimum=0)
            or complete not in (0, 1)
            or not _wire_int(count, minimum=0)
        ):
            raise ValueError("portable episode has invalid manifest completeness")
        if complete == 0:
            if count != 0 or version is not None or digest is not None:
                raise ValueError("portable episode has an invalid incomplete manifest")
        elif complete == 1:
            if (
                not _wire_int(count, minimum=1)
                or version != EPISODE_SOURCE_MANIFEST_VERSION
                or not isinstance(digest, str)
                or _CLAIM_RESULT_HASH_RE.fullmatch(digest) is None
            ):
                raise ValueError("portable episode has an invalid manifest header")

    coverage_by_key: dict[tuple[object, object, object], dict] = {}
    for proof in grouped.get("message_retention_coverage", []):
        key = (
            proof.get("message_id"), proof.get("chunk_id"),
            proof.get("coverage_version"),
        )
        if key in coverage_by_key:
            raise ValueError("portable coverage contains a duplicate identity")
        coverage_by_key[key] = proof

    children: dict[str, list[tuple[int, BoundSourceOccurrence]]] = defaultdict(list)
    source_identities: set[tuple[str, str, int]] = set()
    ordinal_identities: set[tuple[str, int]] = set()
    for record in grouped.get("episode_source_occurrence", []):
        episode_id = record.get("episode_id")
        ordinal = record.get("ordinal")
        message_id = record.get("source_message_id")
        if (
            not _wire_text(episode_id, nonempty=True)
            or not _wire_int(ordinal, minimum=0)
            or not _wire_int(message_id, minimum=1)
            or not _wire_text(record.get("source_session_id"), nonempty=True)
            or record.get("source_role")
            not in ("user", "assistant", "system", "tool")
            or not _wire_text(record.get("source_peer_id"), nullable=True, nonempty=True)
            or not _wire_text(
                record.get("source_workspace_id"), nullable=True, nonempty=True
            )
            or not _wire_text(record.get("source_created_at"), nullable=True)
            or not _wire_text(
                record.get("source_coverage_chunk_id"), nonempty=True
            )
            or not _wire_text(
                record.get("source_coverage_version"), nonempty=True
            )
            or not isinstance(record.get("source_content_hash"), str)
            or _RAW_SHA256_RE.fullmatch(record["source_content_hash"])
            is None
        ):
            raise ValueError("portable episode source occurrence is malformed")
        if (record.get("source_peer_id") is None) != (
            record.get("source_workspace_id") is None
        ):
            raise ValueError("portable episode source has partial peer ownership")
        episode = episodes.get(str(episode_id))
        if episode is None or episode["session_id"] != record["source_session_id"]:
            raise ValueError("portable episode source crosses its session")
        ordinal_key = (str(episode_id), int(ordinal))
        source_key = (
            str(episode_id), str(record["source_session_id"]), int(message_id)
        )
        if ordinal_key in ordinal_identities or source_key in source_identities:
            raise ValueError("portable episode source contains a duplicate identity")
        ordinal_identities.add(ordinal_key)
        source_identities.add(source_key)
        proof = coverage_by_key.get((
            message_id,
            record["source_coverage_chunk_id"],
            record["source_coverage_version"],
        ))
        if proof is None or any(
            proof.get(proof_field) != record.get(source_field)
            for proof_field, source_field in (
                ("source_session_id", "source_session_id"),
                ("source_role", "source_role"),
                ("source_peer_id", "source_peer_id"),
                ("source_workspace_id", "source_workspace_id"),
                ("source_created_at", "source_created_at"),
                ("message_content_hash", "source_content_hash"),
            )
        ):
            raise ValueError("portable episode source mismatches coverage")
        occurrence = BoundSourceOccurrence(
            message_id=int(message_id),
            session_id=str(record["source_session_id"]),
            role=str(record["source_role"]),
            source_peer_id=record["source_peer_id"],
            source_workspace_id=record["source_workspace_id"],
            source_created_at=record["source_created_at"],
            coverage_chunk_id=str(record["source_coverage_chunk_id"]),
            coverage_version=str(record["source_coverage_version"]),
            content_hash=str(record["source_content_hash"]),
        )
        children[str(episode_id)].append((int(ordinal), occurrence))

    for episode_id, episode in episodes.items():
        ordered = sorted(children.get(episode_id, []), key=lambda item: item[0])
        if episode["source_manifest_complete"] == 0:
            if ordered:
                raise ValueError("portable incomplete episode has source rows")
            continue
        count = int(episode["source_manifest_count"])
        sources = tuple(item[1] for item in ordered)
        if (
            len(ordered) != count
            or any(item[0] != expected for expected, item in enumerate(ordered))
            or combine_source_occurrences((sources,)) != sources
            or episode["source_manifest_hash"]
            != source_manifest_hash(EPISODE_SOURCE_MANIFEST_VERSION, sources)
        ):
            raise ValueError("portable episode source manifest is corrupt")


def _portable_lossless_fact_sources(
    grouped: dict[str, list[dict]],
) -> dict[tuple[str, int], tuple[BoundSourceOccurrence, str]]:
    """Decode exact canonical lossless artifacts without destination SQL."""

    chunks = {record["id"]: record for record in grouped.get("chunk", [])}
    result: dict[tuple[str, int], tuple[BoundSourceOccurrence, str]] = {}
    for proof in grouped.get("message_retention_coverage", []):
        if proof.get("coverage_version") != "dream-lossless-message-v1":
            continue
        chunk = chunks.get(proof.get("chunk_id"))
        try:
            payload = loads_strict_json(chunk["text"] if chunk else None)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("portable fact source artifact is not canonical") from None
        message_id = proof.get("message_id")
        if (
            not isinstance(payload, dict)
            or payload.get("id") != message_id
            or payload.get("role") != proof.get("source_role")
            or not isinstance(payload.get("content"), str)
            or chunk is None
            or chunk.get("session_id") != proof.get("source_session_id")
            or chunk.get("start_message_id") != message_id
            or chunk.get("end_message_id") != message_id
            or chunk.get("chunk_kind") != "coverage"
        ):
            raise ValueError("portable fact source artifact is not lossless")
        try:
            canonical, content_hash, hash_version, record_version = (
                canonical_message_record(
                    message_id=int(message_id),
                    session_id=proof.get("source_session_id"),
                    role=proof.get("source_role"),
                    content=payload["content"],
                    source_created_at=proof.get("source_created_at"),
                    source_peer_id=proof.get("source_peer_id"),
                    source_workspace_id=proof.get("source_workspace_id"),
                )
            )
        except (TypeError, ValueError):
            raise ValueError("portable fact source metadata is invalid") from None
        if (
            chunk["text"] != canonical
            or proof.get("message_content_hash") != content_hash
            or proof.get("hash_version") != hash_version
            or proof.get("record_version") != record_version
        ):
            raise ValueError("portable fact source proof mismatches its bytes")
        key = (str(proof["source_session_id"]), int(message_id))
        if key in result:
            raise ValueError("portable fact stream has ambiguous source bytes")
        result[key] = (
            BoundSourceOccurrence(
                message_id=int(message_id),
                session_id=str(proof["source_session_id"]),
                role=str(proof["source_role"]),
                source_peer_id=proof.get("source_peer_id"),
                source_workspace_id=proof.get("source_workspace_id"),
                source_created_at=proof.get("source_created_at"),
                coverage_chunk_id=str(proof["chunk_id"]),
                coverage_version=str(proof["coverage_version"]),
                content_hash=str(proof["message_content_hash"]),
            ),
            str(payload["content"]),
        )
    return result


def _portable_fact_initial_valid_at(
    item: dict,
    occurrences: tuple[BoundSourceOccurrence, ...],
) -> str:
    if item.get("date"):
        return normalize_iso_timestamp(
            f"{item['date']}T00:00:00.000Z", context="portable fact date"
        )
    source_times: list[str] = []
    for occurrence in occurrences:
        try:
            source_times.append(normalize_iso_timestamp(
                occurrence.source_created_at, context="portable fact source"
            ))
        except ValueError:
            continue
    return max(source_times) if source_times else "0001-01-01T00:00:00.000Z"


def _validate_v10_fact_records(grouped: dict[str, list[dict]]) -> None:
    """Validate the complete fact authority ledger before target mutation."""

    sessions: dict[str, dict] = {}
    for record in grouped.get("session", []):
        session_id = record.get("id")
        if not _wire_text(session_id, nonempty=True) or session_id in sessions:
            raise ValueError("portable fact session identity is invalid")
        sessions[str(session_id)] = record
        if not facts_retry_state_is_valid(
            record.get("facts_retry_count"),
            record.get("facts_retry_config_version"),
            record.get("facts_quarantined"),
        ):
            raise ValueError("portable session has invalid facts retry state")
        for field in (
            "facts_message_id", "facts_cursor_message_id",
            "facts_cursor_partial_message_id",
        ):
            value = record.get(field)
            if value is not None and not _wire_int(value, minimum=1):
                raise _invalid_wire("session", field)
        if not _wire_int(record.get("facts_cursor_offset"), minimum=0):
            raise _invalid_wire("session", "facts_cursor_offset")
        cursor_generation = record.get("facts_cursor_prompt_version")
        if cursor_generation is not None and not facts_generation_is_recognized(
            cursor_generation
        ):
            raise _invalid_wire("session", "facts_cursor_prompt_version")
        if (record.get("facts_cursor_partial_message_id") is None) != (
            record.get("facts_cursor_offset") == 0
        ):
            raise ValueError("portable session has an incoherent facts cursor")
        if record.get("facts_message_id") != record.get("facts_cursor_message_id"):
            raise ValueError("portable session fact watermark disagrees with cursor")

    source_stream = _portable_lossless_fact_sources(grouped)
    coverage_publication_by_key: dict[tuple[int, str, str], str] = {}
    for proof in grouped.get("message_retention_coverage", []):
        if proof.get("coverage_version") != "dream-lossless-message-v1":
            continue
        try:
            publication = normalize_iso_timestamp(
                proof.get("created_at"), context="portable coverage publication"
            )
            key = (
                int(proof["message_id"]), str(proof["chunk_id"]),
                str(proof["coverage_version"]),
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "portable fact source has an invalid publication time"
            ) from None
        previous = coverage_publication_by_key.get(key)
        if previous is not None and previous != publication:
            raise ValueError("portable fact source publication is ambiguous")
        coverage_publication_by_key[key] = publication
    source_ids_by_session: dict[str, list[int]] = defaultdict(list)
    for (source_session_id, source_message_id), (source, _content) in source_stream.items():
        if source.role in {"user", "assistant"}:
            source_ids_by_session[source_session_id].append(source_message_id)
    for source_ids in source_ids_by_session.values():
        source_ids.sort()

    outcomes: dict[str, dict] = {}
    for record in grouped.get("fact_extraction_outcome", []):
        slice_key = record.get("slice_key")
        if not _wire_text(slice_key, nonempty=True) or slice_key in outcomes:
            raise ValueError("portable fact outcome contains a duplicate identity")
        for field in ("input_hash", "result_hash", "source_manifest_hash"):
            if (
                not isinstance(record.get(field), str)
                or _CLAIM_RESULT_HASH_RE.fullmatch(record[field]) is None
            ):
                raise _invalid_wire("fact_extraction_outcome", field)
        if (
            record.get("session_id") not in sessions
            or not fact_publication_version_is_recognized(
                record.get("prompt_version")
            )
            or not _wire_int(record.get("generation"), minimum=1)
            or int(record.get("generation")) > FACT_MAX_REVISIONS_PER_OUTCOME
            or record.get("outcome_status") not in {"success", "empty"}
            or record.get("source_manifest_version")
            != FACT_SOURCE_MANIFEST_VERSION
            or not _wire_int(record.get("source_manifest_count"), minimum=1)
            or int(record.get("source_manifest_count"))
            > FACT_MAX_SOURCES_PER_OUTCOME
            or not _wire_int(record.get("source_manifest_complete"), minimum=0)
            or record.get("source_manifest_complete") != 1
        ):
            raise ValueError("portable fact outcome has invalid authority state")
        for field in (
            "cursor_before_message_id", "cursor_before_partial_message_id",
            "cursor_after_message_id", "cursor_after_partial_message_id",
        ):
            value = record.get(field)
            if value is not None and not _wire_int(value, minimum=1):
                raise _invalid_wire("fact_extraction_outcome", field)
        for field in ("cursor_before_offset", "cursor_after_offset"):
            if not _wire_int(record.get(field), minimum=0):
                raise _invalid_wire("fact_extraction_outcome", field)
        if (
            (record.get("cursor_before_partial_message_id") is None)
            != (record.get("cursor_before_offset") == 0)
            or (record.get("cursor_after_partial_message_id") is None)
            != (record.get("cursor_after_offset") == 0)
        ):
            raise ValueError("portable fact outcome has incoherent cursor offsets")
        try:
            succeeded_at = normalize_iso_timestamp(
                record.get("succeeded_at"), context="portable fact publication"
            )
        except ValueError:
            raise ValueError("portable fact outcome has invalid publication time") from None
        if succeeded_at != record.get("succeeded_at"):
            raise ValueError("portable fact outcome time is not normalized")
        outcomes[str(slice_key)] = record

    source_rows: dict[str, list[tuple[int, BoundSourceOccurrence, str]]] = defaultdict(list)
    source_publications: dict[str, list[str]] = defaultdict(list)
    source_identity: set[tuple[str, int]] = set()
    source_message_identity: set[tuple[str, str, int]] = set()
    for record in grouped.get("fact_extraction_source_occurrence", []):
        slice_key = record.get("slice_key")
        ordinal = record.get("ordinal")
        message_id = record.get("source_message_id")
        if (
            slice_key not in outcomes
            or not _wire_int(ordinal, minimum=0)
            or not _wire_int(message_id, minimum=1)
            or record.get("source_role") not in {"user", "assistant"}
            or not _wire_text(record.get("source_session_id"), nonempty=True)
            or not _wire_text(record.get("source_peer_id"), nullable=True, nonempty=True)
            or not _wire_text(
                record.get("source_workspace_id"), nullable=True, nonempty=True
            )
            or not _wire_text(record.get("source_created_at"), nullable=True)
            or not _wire_text(record.get("source_coverage_chunk_id"), nonempty=True)
            or record.get("source_coverage_version")
            != "dream-lossless-message-v1"
            or not isinstance(record.get("source_content_hash"), str)
            or _RAW_SHA256_RE.fullmatch(record["source_content_hash"]) is None
        ):
            raise ValueError("portable fact source occurrence is malformed")
        if (record.get("source_peer_id") is None) != (
            record.get("source_workspace_id") is None
        ):
            raise ValueError("portable fact source has partial peer ownership")
        ordinal_key = (str(slice_key), int(ordinal))
        message_key = (
            str(slice_key), str(record["source_session_id"]), int(message_id)
        )
        if ordinal_key in source_identity or message_key in source_message_identity:
            raise ValueError("portable fact source contains a duplicate identity")
        source_identity.add(ordinal_key)
        source_message_identity.add(message_key)
        proof_pair = source_stream.get((str(record["source_session_id"]), int(message_id)))
        if proof_pair is None:
            raise ValueError("portable fact source lacks a lossless proof")
        proof, content = proof_pair
        occurrence = BoundSourceOccurrence(
            message_id=int(message_id),
            session_id=str(record["source_session_id"]),
            role=str(record["source_role"]),
            source_peer_id=record.get("source_peer_id"),
            source_workspace_id=record.get("source_workspace_id"),
            source_created_at=record.get("source_created_at"),
            coverage_chunk_id=str(record["source_coverage_chunk_id"]),
            coverage_version=str(record["source_coverage_version"]),
            content_hash=str(record["source_content_hash"]),
        )
        if occurrence != proof:
            raise ValueError("portable fact source mismatches durable coverage")
        coverage_publication = coverage_publication_by_key.get((
            int(message_id), str(record["source_coverage_chunk_id"]),
            str(record["source_coverage_version"]),
        ))
        if coverage_publication is None:
            raise ValueError("portable fact source lacks a publication clock")
        source_rows[str(slice_key)].append((int(ordinal), occurrence, content))
        source_publications[str(slice_key)].append(coverage_publication)

    occurrences_by_slice: dict[str, tuple[BoundSourceOccurrence, ...]] = {}
    for slice_key, outcome in outcomes.items():
        ordered = sorted(source_rows.get(slice_key, []), key=lambda item: item[0])
        count = int(outcome["source_manifest_count"])
        if count > FACT_MAX_SOURCES_PER_OUTCOME or len(ordered) != count or any(
            item[0] != expected for expected, item in enumerate(ordered)
        ):
            raise ValueError("portable fact source ordinals are incomplete")
        occurrences = tuple(item[1] for item in ordered)
        if (
            len(occurrences) != count
            or combine_source_occurrences((occurrences,)) != occurrences
            or any(source.session_id != outcome["session_id"] for source in occurrences)
            or outcome["source_manifest_hash"] != source_manifest_hash(
                FACT_SOURCE_MANIFEST_VERSION, occurrences
            )
        ):
            raise ValueError("portable fact source manifest is corrupt")
        before = outcome.get("cursor_before_message_id")
        before_partial = outcome.get("cursor_before_partial_message_id")
        before_offset = int(outcome["cursor_before_offset"])
        after = outcome.get("cursor_after_message_id")
        after_partial = outcome.get("cursor_after_partial_message_id")
        after_offset = int(outcome["cursor_after_offset"])
        end_id = after_partial if after_partial is not None else after
        if end_id is None:
            raise ValueError("portable fact outcome does not advance")
        session = sessions[str(outcome["session_id"])]
        frontier = session.get("coverage_message_id")
        source_ids = source_ids_by_session.get(str(outcome["session_id"]), [])
        lower = int(before) if before is not None else -1
        upper = min(int(end_id), int(frontier)) if frontier is not None else -1
        eligible = source_ids[
            bisect_right(source_ids, lower):bisect_right(source_ids, upper)
        ]
        if eligible != [source.message_id for source in occurrences]:
            raise ValueError("portable fact cursor skips an eligible occurrence")
        contents = [item[2] for item in ordered]
        if before_partial is not None and (
            occurrences[0].message_id != before_partial
            or not 0 < before_offset < len(contents[0])
        ):
            raise ValueError("portable fact cursor has an invalid partial start")
        if after_partial is None:
            if after != occurrences[-1].message_id:
                raise ValueError("portable fact cursor has an invalid full boundary")
        elif (
            occurrences[-1].message_id != after_partial
            or not 0 < after_offset < len(contents[-1])
            or after != (
                occurrences[-2].message_id if len(occurrences) > 1 else before
            )
        ):
            raise ValueError("portable fact cursor has an invalid partial end")
        rendered_lines: list[str] = []
        for index, (source, content) in enumerate(zip(occurrences, contents)):
            start = before_offset if source.message_id == before_partial else 0
            end = after_offset if source.message_id == after_partial else len(content)
            empty_whole = not content and start == 0 and end == 0
            if start < 0 or end > len(content) or (end <= start and not empty_whole):
                raise ValueError("portable fact outcome has invalid source offsets")
            rendered_lines.append(f"{source.role}: {content[start:end]}")
        rendered = "\n".join(rendered_lines)
        if (
            fact_slice_key(
                str(outcome["session_id"]),
                cursor_before_message_id=before,
                cursor_before_partial_message_id=before_partial,
                cursor_before_offset=before_offset,
                cursor_after_message_id=after,
                cursor_after_partial_message_id=after_partial,
                cursor_after_offset=after_offset,
                occurrences=occurrences,
            ) != slice_key
            or fact_input_hash(rendered) != outcome["input_hash"]
        ):
            raise ValueError("portable fact outcome source identity is forged")
        occurrences_by_slice[slice_key] = occurrences

    revisions_by_slice: dict[str, list[dict]] = defaultdict(list)
    revision_identity: set[tuple[str, int]] = set()
    for revision in grouped.get("fact_extraction_revision", []):
        slice_key = revision.get("slice_key")
        generation = revision.get("generation")
        if (
            slice_key not in outcomes
            or not _wire_int(generation, minimum=1)
            or int(generation) > FACT_MAX_REVISIONS_PER_OUTCOME
            or (str(slice_key), int(generation)) in revision_identity
            or not fact_publication_version_is_recognized(
                revision.get("prompt_version")
            )
            or revision.get("outcome_status") not in {"success", "empty"}
            or not isinstance(revision.get("result_hash"), str)
            or _CLAIM_RESULT_HASH_RE.fullmatch(revision["result_hash"]) is None
        ):
            raise ValueError("portable fact revision is malformed")
        try:
            succeeded = normalize_iso_timestamp(
                revision.get("succeeded_at"), context="portable fact revision"
            )
        except ValueError:
            raise ValueError("portable fact revision time is invalid") from None
        if succeeded != revision.get("succeeded_at"):
            raise ValueError("portable fact revision time is not normalized")
        revision_identity.add((str(slice_key), int(generation)))
        revisions_by_slice[str(slice_key)].append(revision)
        if (
            len(revisions_by_slice[str(slice_key)])
            > FACT_MAX_REVISIONS_PER_OUTCOME
        ):
            raise ValueError("portable fact outcome exceeds its revision limit")

    facts: dict[tuple[str, str], tuple[dict, dict]] = {}
    fact_items_by_slice: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in grouped.get("narrative_fact", []):
        slice_key = record.get("source_outcome_key")
        if slice_key not in outcomes:
            raise ValueError("portable narrative fact has no outcome")
        try:
            entities = loads_strict_json(record.get("entities"))
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("portable narrative fact has malformed entities") from None
        item = {
            "text": record.get("text"), "date": record.get("fact_date"),
            "entities": entities,
        }
        clean = validate_fact_items([item], max_items=1)
        if clean is None or canonical_fact_items(clean) != [item]:
            raise ValueError("portable narrative fact payload is not canonical")
        key = (str(slice_key), str(record.get("fact_key")))
        occurrences = occurrences_by_slice[str(slice_key)]
        if (
            key in facts
            or record.get("fact_key") != fact_item_key(item)
            or record.get("session_id") != outcomes[str(slice_key)]["session_id"]
            or not _wire_int(record.get("start_message_id"), minimum=1)
            or not _wire_int(record.get("end_message_id"), minimum=1)
            or record.get("start_message_id")
            != min(source.message_id for source in occurrences)
            or record.get("end_message_id")
            != max(source.message_id for source in occurrences)
            or not fact_publication_version_is_recognized(
                record.get("prompt_version")
            )
            or not _wire_int(record.get("current_generation"), minimum=1)
            or int(record.get("current_generation"))
            > FACT_MAX_REVISIONS_PER_OUTCOME
            or record.get("lifecycle_status") not in {"active", "retracted"}
        ):
            raise ValueError("portable narrative fact identity is invalid")
        for field in ("valid_at", "created_at"):
            try:
                normalized = normalize_iso_timestamp(
                    record.get(field), context=f"portable narrative fact {field}"
                )
            except ValueError:
                raise _invalid_wire("narrative_fact", field) from None
            if normalized != record.get(field):
                raise ValueError("portable narrative fact time is not normalized")
        if record.get("invalid_at") is not None:
            try:
                invalid_at = normalize_iso_timestamp(
                    record["invalid_at"], context="portable fact invalid time"
                )
            except ValueError:
                raise _invalid_wire("narrative_fact", "invalid_at") from None
            if invalid_at != record["invalid_at"]:
                raise ValueError("portable narrative fact time is not normalized")
        facts[key] = (record, item)
        fact_items_by_slice[str(slice_key)][str(record["fact_key"])] = item
        if (
            len(fact_items_by_slice[str(slice_key)])
            > FACT_MAX_HISTORY_ITEMS_PER_OUTCOME
        ):
            raise ValueError("portable fact outcome exceeds its item limit")

    events_by_fact: dict[tuple[str, str], list[dict]] = defaultdict(list)
    event_count_by_slice: dict[str, int] = defaultdict(int)
    event_identity: set[tuple[str, str, int]] = set()
    for event in grouped.get("narrative_fact_lifecycle", []):
        fact_key = (str(event.get("source_outcome_key")), str(event.get("fact_key")))
        generation = event.get("generation")
        identity = (*fact_key, generation)
        if (
            fact_key not in facts
            or not _wire_int(generation, minimum=1)
            or int(generation) > FACT_MAX_REVISIONS_PER_OUTCOME
            or identity in event_identity
            or not _wire_int(event.get("direction"))
            or event.get("direction") not in {-1, 1}
            or not fact_publication_version_is_recognized(
                event.get("prompt_version")
            )
            or not isinstance(event.get("result_hash"), str)
            or _CLAIM_RESULT_HASH_RE.fullmatch(event["result_hash"]) is None
        ):
            raise ValueError("portable narrative fact lifecycle is malformed")
        for field in ("event_at", "recorded_at"):
            try:
                normalized = normalize_iso_timestamp(
                    event.get(field), context=f"portable fact lifecycle {field}"
                )
            except ValueError:
                raise _invalid_wire("narrative_fact_lifecycle", field) from None
            if normalized != event.get(field):
                raise ValueError("portable fact lifecycle time is not normalized")
        event_identity.add(identity)
        events_by_fact[fact_key].append(event)
        event_count_by_slice[fact_key[0]] += 1
        if (
            event_count_by_slice[fact_key[0]]
            > FACT_MAX_LIFECYCLE_EVENTS_PER_OUTCOME
        ):
            raise ValueError("portable fact outcome exceeds its lifecycle limit")

    revision_by_slice_generation = {
        slice_key: {
            int(revision["generation"]): revision for revision in revisions
        }
        for slice_key, revisions in revisions_by_slice.items()
    }
    active_by_slice: dict[str, set[str]] = defaultdict(set)
    events_by_slice_generation: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for key, (record, item) in facts.items():
        slice_key, item_key = key
        events = sorted(events_by_fact.get(key, []), key=lambda event: event["generation"])
        revisions = revision_by_slice_generation[slice_key]
        expected_valid = _portable_fact_initial_valid_at(
            item, occurrences_by_slice[slice_key]
        )
        if (
            not events or events[0]["direction"] != 1
            or any(
                left["generation"] >= right["generation"]
                or left["direction"] == right["direction"]
                for left, right in zip(events, events[1:])
            )
            or any(event["event_at"] != expected_valid for event in events)
            or record["valid_at"] != expected_valid
            or record["prompt_version"] != events[0]["prompt_version"]
            or record["created_at"] != events[0]["recorded_at"]
        ):
            raise ValueError("portable narrative fact lifecycle is inconsistent")
        for event in events:
            revision = revisions.get(event["generation"])
            if (
                revision is None
                or event["prompt_version"] != revision["prompt_version"]
                or event["result_hash"] != revision["result_hash"]
                or event["recorded_at"] != revision["succeeded_at"]
            ):
                raise ValueError("portable fact lifecycle is not revision-bound")
            events_by_slice_generation[(slice_key, int(event["generation"]))].append(
                (item_key, int(event["direction"]))
            )
        latest = events[-1]
        active = record["lifecycle_status"] == "active"
        if (
            latest["generation"] != record["current_generation"]
            or active != (latest["direction"] == 1)
            or (active and record["invalid_at"] is not None)
            or (
                not active
                and record["invalid_at"] != expected_valid
            )
        ):
            raise ValueError("portable narrative fact projection is stale")
        if active:
            active_by_slice[slice_key].add(item_key)

    for slice_key, outcome in outcomes.items():
        revisions = sorted(
            revisions_by_slice.get(slice_key, []),
            key=lambda revision: revision["generation"],
        )
        if (
            len(revisions) != int(outcome["generation"])
            or not revisions
            or any(
                revision["generation"] != expected
                for expected, revision in enumerate(revisions, start=1)
            )
        ):
            raise ValueError("portable fact revision history is incomplete")
        if revisions[0]["succeeded_at"] < max(source_publications[slice_key]):
            raise ValueError("portable fact predates its exact source publication")
        previous_time: str | None = None
        folded: set[str] = set()
        result_hash_cache: dict[frozenset[str], str] = {}
        item_by_key = fact_items_by_slice.get(slice_key, {})
        for revision in revisions:
            if previous_time is not None and revision["succeeded_at"] < previous_time:
                raise ValueError("portable fact revisions are out of order")
            previous_time = revision["succeeded_at"]
            for item_key, direction in sorted(
                events_by_slice_generation.get(
                    (slice_key, int(revision["generation"])), []
                )
            ):
                if direction == 1:
                    if item_key in folded:
                        raise ValueError("portable fact lifecycle double-asserts")
                    folded.add(item_key)
                else:
                    if item_key not in folded:
                        raise ValueError("portable fact lifecycle retracts absent state")
                    folded.remove(item_key)
            if len(folded) > FACT_MAX_ACTIVE_ITEMS_PER_OUTCOME:
                raise ValueError("portable fact revision exceeds its active item limit")
            state_key = frozenset(folded)
            result_hash = result_hash_cache.get(state_key)
            if result_hash is None:
                result_hash = fact_result_hash(
                    item_by_key[item_key] for item_key in folded
                )
                result_hash_cache[state_key] = result_hash
            if (
                revision["outcome_status"]
                != ("success" if folded else "empty")
                or revision["result_hash"] != result_hash
            ):
                raise ValueError("portable fact revision result is forged")
        latest = revisions[-1]
        if any(
            latest[field] != outcome[field]
            for field in (
                "generation", "prompt_version", "outcome_status", "result_hash",
                "succeeded_at",
            )
        ) or folded != active_by_slice.get(slice_key, set()):
            raise ValueError("portable fact outcome projection is inconsistent")

    # Every published unit belongs to the one cursor-committed chain. This
    # rejects disconnected but individually source-valid outcomes that were
    # never authorized by the session's extraction walk.
    outcomes_by_before: dict[tuple[str, object, object, int], dict] = {}
    for outcome in outcomes.values():
        key = (
            str(outcome["session_id"]), outcome["cursor_before_message_id"],
            outcome["cursor_before_partial_message_id"],
            int(outcome["cursor_before_offset"]),
        )
        previous = outcomes_by_before.get(key)
        if previous is not None and previous["slice_key"] != outcome["slice_key"]:
            raise ValueError("portable fact outcomes fork a source cursor")
        outcomes_by_before[key] = outcome
    for session_id, session in sessions.items():
        target = (
            session.get("facts_cursor_message_id"),
            session.get("facts_cursor_partial_message_id"),
            int(session.get("facts_cursor_offset") or 0),
        )
        coordinate: tuple[object, object, int] = (None, None, 0)
        seen: set[str] = set()
        while True:
            outcome = outcomes_by_before.get((session_id, *coordinate))
            if outcome is None:
                break
            if outcome["slice_key"] in seen:
                raise ValueError("portable fact cursor chain is cyclic")
            seen.add(str(outcome["slice_key"]))
            coordinate = (
                outcome["cursor_after_message_id"],
                outcome["cursor_after_partial_message_id"],
                int(outcome["cursor_after_offset"]),
            )
        owned = {
            str(outcome["slice_key"]) for outcome in outcomes.values()
            if outcome["session_id"] == session_id
        }
        if coordinate != target or seen != owned:
            raise ValueError("portable fact cursor is not the terminal outcome")
        if seen and not facts_generation_is_recognized(
            session.get("facts_cursor_prompt_version")
        ):
            raise ValueError("portable fact cursor lacks a recognized generation")


def _validate_v6_record_scalars(grouped: dict[str, list[dict]]) -> None:
    """Validate v6 values before the first destination write.

    Exact key sets prevent column loss; this layer prevents SQLite's
    ``INSERT OR IGNORE``/affinity rules from turning malformed canonical
    records into a partial successful restore.
    """
    timestamp_fields = {
        "session": {"started_at", "ended_at"},
        "peer": {"registered_at"},
        "session_peer": {"added_at"},
        "chunk": {"created_at"},
        "message_retention_coverage": {"source_created_at", "created_at"},
        "user_profile_fact": {
            "valid_at", "invalid_at", "created_at", "source_created_at",
        },
        "episode": {"created_at"},
        "procedure": {"created_at"},
        "edge": {"first_seen", "last_seen", "last_reinforced", "valid_at", "invalid_at"},
        "profile_entry": {"first_seen", "last_updated"},
    }
    for kind, fields in timestamp_fields.items():
        for record in grouped.get(kind, []):
            for field in fields:
                if not _wire_text(record[field], nullable=True):
                    raise _invalid_wire(kind, field)

    for record in grouped.get("session", []):
        if not _wire_text(record["id"], nonempty=True):
            raise _invalid_wire("session", "id")
        if not _wire_text(record["started_at"], nonempty=True):
            raise _invalid_wire("session", "started_at")
        if not _wire_text(record.get("source_workspace_id"), nullable=True):
            raise _invalid_wire("session", "source_workspace_id")
        for field in (
            "summary", "auto_summary", "digest_cursor_prompt_version",
            "digest_published_generation", "digested_prompt_version",
            "profile_prompt_version", "profile_cursor_prompt_version",
            "profile_published_generation", "episodes_prompt_version",
            "digest_retry_config_version", "profile_retry_config_version",
            "facts_cursor_prompt_version", "facts_retry_config_version",
        ):
            if not _wire_text(record[field], nullable=True):
                raise _invalid_wire("session", field)
        if record["summary_source"] not in (None, "auto", "operator", "legacy"):
            raise _invalid_wire("session", "summary_source")
        for field in (
            "auto_summary_message_id", "auto_summary_partial_message_id",
            "coverage_message_id", "digest_cursor_message_id",
            "digest_cursor_partial_message_id", "facts_message_id",
            "facts_cursor_message_id", "facts_cursor_partial_message_id",
            "profile_cursor_message_id", "profile_cursor_partial_message_id",
        ):
            value = record[field]
            if value is not None and not _wire_int(value, minimum=1):
                raise _invalid_wire("session", field)
        for field in (
            "auto_summary_message_offset", "digest_cursor_offset",
            "digest_retry_count", "profile_cursor_offset", "profile_retry_count",
            "facts_cursor_offset", "facts_retry_count",
        ):
            if not _wire_int(record[field], minimum=0):
                raise _invalid_wire("session", field)
        for field in (
            "digest_quarantined", "profile_quarantined", "facts_quarantined",
        ):
            if not _wire_int(record[field], minimum=0) or record[field] not in (0, 1):
                raise _invalid_wire("session", field)

    for record in grouped.get("chunk", []):
        for field in ("id", "session_id", "salience_reason"):
            if not _wire_text(record[field], nonempty=True):
                raise _invalid_wire("chunk", field)
        if not isinstance(record["text"], str):
            raise _invalid_wire("chunk", "text")
        if record["chunk_kind"] not in ("extraction", "coverage"):
            raise _invalid_wire("chunk", "chunk_kind")
        if not _wire_int(record["start_message_id"], minimum=1) or not _wire_int(
            record["end_message_id"], minimum=1
        ) or record["start_message_id"] > record["end_message_id"]:
            raise _invalid_wire("chunk", "message range")

    for record in grouped.get("message_retention_coverage", []):
        if not _wire_int(record["message_id"], minimum=1):
            raise _invalid_wire("message_retention_coverage", "message_id")
        for field in (
            "source_session_id", "chunk_id", "message_content_hash",
            "hash_version", "record_version", "coverage_version",
        ):
            if not _wire_text(record[field], nonempty=True):
                raise _invalid_wire("message_retention_coverage", field)
        if record["source_role"] not in ("user", "assistant", "system", "tool"):
            raise _invalid_wire("message_retention_coverage", "source_role")
        if (record.get("source_peer_id") is None) != (
            record.get("source_workspace_id") is None
        ):
            raise _invalid_wire(
                "message_retention_coverage", "external peer provenance"
            )
        for field in ("source_peer_id", "source_workspace_id"):
            if not _wire_text(record.get(field), nullable=True, nonempty=True):
                raise _invalid_wire("message_retention_coverage", field)

    for record in grouped.get("peer", []):
        for field in ("id", "workspace_id"):
            if not _wire_text(record[field], nonempty=True):
                raise _invalid_wire("peer", field)
        if record["role"] not in ("user", "assistant", "system", "tool"):
            raise _invalid_wire("peer", "role")
        try:
            metadata = loads_strict_json(record["metadata"])
        except (TypeError, ValueError, json.JSONDecodeError):
            raise _invalid_wire("peer", "metadata") from None
        if not isinstance(metadata, dict):
            raise _invalid_wire("peer", "metadata")

    for record in grouped.get("session_peer", []):
        for field in ("session_id", "workspace_id", "peer_id"):
            if not _wire_text(record[field], nonempty=True):
                raise _invalid_wire("session_peer", field)
        try:
            configuration = loads_strict_json(record["configuration"])
        except (TypeError, ValueError, json.JSONDecodeError):
            raise _invalid_wire("session_peer", "configuration") from None
        if not isinstance(configuration, dict):
            raise _invalid_wire("session_peer", "configuration")

    for record in grouped.get("user_profile_fact", []):
        if record["slot"] not in _PROFILE_SLOTS:
            raise _invalid_wire("user_profile_fact", "slot")
        if not _wire_text(record["value"], nonempty=True):
            raise _invalid_wire("user_profile_fact", "value")
        if record["slot"] == "relationship":
            if not _wire_text(record["slot_key"], nonempty=True):
                raise _invalid_wire("user_profile_fact", "slot_key")
        elif record["slot_key"] is not None:
            raise _invalid_wire("user_profile_fact", "slot_key")
        if not _wire_number(record["confidence"], minimum=0.0, maximum=1.0):
            raise _invalid_wire("user_profile_fact", "confidence")
        for field in ("evidence_message_id", "source_message_id"):
            if record[field] is not None and not _wire_int(record[field], minimum=1):
                if field == "source_message_id":
                    raise ValueError(
                        "portable profile fact has invalid provenance "
                        "source_message_id"
                    )
                raise _invalid_wire("user_profile_fact", field)
        if not _wire_text(record["source_session_id"], nullable=True):
            raise _invalid_wire("user_profile_fact", "source_session_id")

    for record in grouped.get("episode", []):
        for field in ("id", "session_id", "title"):
            if not _wire_text(record[field], nonempty=True):
                raise _invalid_wire("episode", field)
        if not isinstance(record["summary"], str):
            raise _invalid_wire("episode", "summary")
        for field in ("participants", "key_entities"):
            if not _wire_json_array(record[field]):
                raise _invalid_wire("episode", field)
        for field in ("start_message_id", "end_message_id"):
            if record[field] is not None and not _wire_int(record[field], minimum=1):
                raise _invalid_wire("episode", field)
        if (
            record["start_message_id"] is not None
            and record["end_message_id"] is not None
            and record["start_message_id"] > record["end_message_id"]
        ):
            raise _invalid_wire("episode", "message range")
        if record["outcome"] not in (
            None, "resolved", "blocked", "deferred", "informational"
        ):
            raise _invalid_wire("episode", "outcome")
        for field in ("digest_slice_key", "digest_generation"):
            if not _wire_text(record[field], nullable=True):
                raise _invalid_wire("episode", field)

    for record in grouped.get("procedure", []):
        for field in ("id", "session_id", "name"):
            if not _wire_text(record[field], nonempty=True):
                raise _invalid_wire("procedure", field)
        if not _wire_text(record["description"], nullable=True):
            raise _invalid_wire("procedure", "description")
        for field in ("steps", "triggers", "entities_involved"):
            if not _wire_json_array(record[field]):
                raise _invalid_wire("procedure", field)
        if not _wire_number(record["confidence"], minimum=0.0, maximum=1.0):
            raise _invalid_wire("procedure", "confidence")
        if record["status"] not in ("active", "stale"):
            raise _invalid_wire("procedure", "status")

    for record in grouped.get("edge", []):
        if not _wire_int(record["id"], minimum=1):
            raise _invalid_wire("edge", "id")
        for field in ("subject_canonical", "object_canonical"):
            if not _wire_text(record[field], nonempty=True):
                raise _invalid_wire("edge", field)
        if record["predicate"] not in ALLOWED_PREDICATES:
            raise _invalid_wire("edge", "predicate")
        for field in ("pos_evidence", "neg_evidence"):
            if not _wire_int(record[field], minimum=0):
                raise _invalid_wire("edge", field)
        if record["status"] not in ("active", "stale", "retracted"):
            raise _invalid_wire("edge", "status")
        if not _wire_int(record["derived"], minimum=0) or record["derived"] not in (0, 1):
            raise _invalid_wire("edge", "derived")

    for record in grouped.get("profile_entry", []):
        if not _wire_int(record["id"], minimum=1):
            raise _invalid_wire("profile_entry", "id")
        if record["kind"] not in ("preference", "avoidance", "style", "context"):
            raise _invalid_wire("profile_entry", "kind")
        if not _wire_text(record["text"], nonempty=True):
            raise _invalid_wire("profile_entry", "text")
        for field in ("pos_evidence", "neg_evidence"):
            if not _wire_int(record[field], minimum=0):
                raise _invalid_wire("profile_entry", field)


def _validate_v7_records(grouped: dict[str, list[dict]]) -> None:
    """Validate the complete claim ledger and every portable reference.

    Numeric ids in v7 are wire-local handles only.  Validation resolves them
    to natural edge/evidence identities; import never writes those ids into the
    destination database.
    """
    def unique_map(kind: str, records: list[dict], key_fn):
        result = {}
        for record in records:
            key = key_fn(record)
            if key in result:
                raise ValueError(f"portable {kind} contains a duplicate identity")
            result[key] = record
        return result

    sessions = unique_map(
        "session", grouped.get("session", []), lambda row: row["id"]
    )
    peers = unique_map(
        "peer", grouped.get("peer", []),
        lambda row: (row["workspace_id"], row["id"]),
    )
    session_peers = unique_map(
        "session peer", grouped.get("session_peer", []),
        lambda row: (row["session_id"], row["workspace_id"], row["peer_id"]),
    )
    for member in session_peers.values():
        session = sessions.get(member["session_id"])
        peer = peers.get((member["workspace_id"], member["peer_id"]))
        if (
            session is None
            or peer is None
            or session.get("source_workspace_id") != member["workspace_id"]
        ):
            raise ValueError("portable session peer crosses an identity boundary")
    chunks = unique_map(
        "chunk", grouped.get("chunk", []), lambda row: row["id"]
    )
    edge_by_wire = unique_map(
        "edge id", grouped.get("edge", []), lambda row: row["id"]
    )
    if any(record["derived"] != 0 for record in edge_by_wire.values()):
        raise ValueError("portable v7 must contain direct graph edges only")
    unique_map(
        "edge", grouped.get("edge", []),
        lambda row: (
            row["subject_canonical"], row["predicate"], row["object_canonical"],
        ),
    )
    aliases = unique_map(
        "entity alias", grouped.get("entity_alias", []), lambda row: row["alias"]
    )
    for alias in aliases.values():
        if not _wire_text(alias["alias"], nonempty=True) or not _wire_text(
            alias["canonical"], nonempty=True
        ):
            raise ValueError("portable entity alias has invalid state")
        if (
            canonicalize.normalize(alias["alias"]) != alias["alias"]
            or canonicalize.normalize(alias["canonical"]) != alias["canonical"]
        ):
            raise ValueError("portable entity alias is not canonicalized")
    for alias in aliases.values():
        target = aliases.get(alias["canonical"])
        if target is not None and target["canonical"] != alias["canonical"]:
            raise ValueError("portable entity aliases are cyclic or chained")
    for edge in edge_by_wire.values():
        for field in ("subject_canonical", "object_canonical"):
            endpoint = edge[field]
            if canonicalize.normalize(endpoint) != endpoint:
                raise ValueError("portable edge endpoint is not canonicalized")
            alias = aliases.get(endpoint)
            if alias is not None and alias["canonical"] != endpoint:
                raise ValueError("portable edge endpoint resolves through an alias")
    for chunk in chunks.values():
        if chunk["session_id"] not in sessions:
            raise ValueError("portable chunk references an absent session")
        validate_timestamp_order(
            sessions[chunk["session_id"]]["started_at"],
            chunk["created_at"],
            context="portable session/chunk transaction",
        )

    coverage = unique_map(
        "message coverage", grouped.get("message_retention_coverage", []),
        lambda row: (row["message_id"], row["chunk_id"], row["coverage_version"]),
    )
    for proof in coverage.values():
        if proof["source_session_id"] not in sessions or proof["chunk_id"] not in chunks:
            raise ValueError("portable coverage references an absent parent")
        proof_chunk = chunks[proof["chunk_id"]]
        if proof_chunk["session_id"] != proof["source_session_id"]:
            raise ValueError("portable coverage crosses session boundaries")
        peer_id = proof.get("source_peer_id")
        workspace_id = proof.get("source_workspace_id")
        session = sessions[proof["source_session_id"]]
        if peer_id is None:
            if session.get("source_workspace_id") is not None:
                raise ValueError(
                    "portable bound-session coverage lacks exact peer provenance"
                )
        else:
            try:
                normalize_iso_timestamp(
                    proof["source_created_at"],
                    context="portable external message source",
                )
            except ValueError:
                raise ValueError(
                    "portable external coverage has an invalid source timestamp"
                ) from None
            peer = peers.get((workspace_id, peer_id))
            if (
                session.get("source_workspace_id") != workspace_id
                or peer is None
                or peer["role"] != proof["source_role"]
                or (
                    proof["source_session_id"], workspace_id, peer_id
                ) not in session_peers
            ):
                raise ValueError(
                    "portable coverage has invalid external peer provenance"
                )
        validate_timestamp_order(
            proof_chunk["created_at"],
            proof["created_at"],
            context="portable coverage publication transaction",
        )
        try:
            normalize_iso_timestamp(
                proof["source_created_at"], context="portable message source"
            )
        except ValueError:
            # Explicit native/v1 legacy-unknown metadata has no schedulable
            # occurrence coordinate. External/v2 rows were rejected above.
            pass
        else:
            if not event_clock_is_valid(
                proof["source_created_at"], proof["created_at"]
            ):
                raise ValueError(
                    "portable canonical evidence valid time / message source "
                    "follows its coverage proof"
                )

    manifests = unique_map(
        "chunk source manifest", grouped.get("chunk_source_manifest", []),
        lambda row: row["id"],
    )
    members = unique_map(
        "chunk source member", grouped.get("chunk_message_source", []),
        lambda row: (row["chunk_id"], row["ordinal"]),
    )
    unique_map(
        "chunk source message", grouped.get("chunk_message_source", []),
        lambda row: (row["chunk_id"], row["source_message_id"]),
    )
    members_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for member in members.values():
        for field in ("chunk_id", "source_session_id", "source_coverage_chunk_id",
                      "source_coverage_version"):
            if not _wire_text(member[field], nonempty=True):
                raise _invalid_wire("chunk_message_source", field)
        if not _wire_int(member["ordinal"], minimum=0) or not _wire_int(
            member["source_message_id"], minimum=1
        ):
            raise _invalid_wire("chunk_message_source", "ordinal/source_message_id")
        chunk = chunks.get(member["chunk_id"])
        if chunk is None or member["chunk_id"] not in manifests:
            raise ValueError("portable manifest member has no published chunk")
        if (
            chunk["chunk_kind"] != "extraction"
            or chunk["session_id"] != member["source_session_id"]
            or member["source_coverage_version"] != "dream-lossless-message-v1"
        ):
            raise ValueError("portable manifest member has invalid extraction scope")
        proof = coverage.get((
            member["source_message_id"], member["source_coverage_chunk_id"],
            member["source_coverage_version"],
        ))
        if proof is None or proof["source_session_id"] != member["source_session_id"]:
            raise ValueError("portable manifest member has no exact coverage proof")
        members_by_chunk[member["chunk_id"]].append(member)
    for manifest in manifests.values():
        if (
            not _wire_text(manifest["id"], nonempty=True)
            or manifest["source_manifest_version"] != "claim-source-manifest-v1"
            or not _wire_int(manifest["source_manifest_count"], minimum=1)
        ):
            raise _invalid_wire("chunk_source_manifest", "header")
        chunk = chunks.get(manifest["id"])
        ordered = sorted(
            members_by_chunk.get(manifest["id"], []), key=lambda row: row["ordinal"]
        )
        if chunk is None or len(ordered) != manifest["source_manifest_count"]:
            raise ValueError("portable chunk source manifest is incomplete")
        if any(row["ordinal"] != expected for expected, row in enumerate(ordered)):
            raise ValueError("portable chunk source manifest ordinals are not contiguous")
        source_ids = [row["source_message_id"] for row in ordered]
        if source_ids != sorted(source_ids) or len(set(source_ids)) != len(source_ids):
            raise ValueError("portable chunk source manifest order is invalid")
        if (
            chunk["start_message_id"] != source_ids[0]
            or chunk["end_message_id"] != source_ids[-1]
        ):
            raise ValueError("portable chunk source manifest range is invalid")

    outcomes = unique_map(
        "claim extraction outcome",
        grouped.get("claim_extraction_outcome", []),
        lambda row: row["chunk_id"],
    )
    for outcome in outcomes.values():
        if (
            not _wire_text(outcome["chunk_id"], nonempty=True)
            or outcome["chunk_id"] not in manifests
            or not _wire_text(outcome["prompt_version"], nonempty=True)
            or not _wire_int(outcome["prompt_generation"], minimum=0)
            or evidence_ledger.prompt_generation(outcome["prompt_version"])
            != outcome["prompt_generation"]
            or not isinstance(outcome["result_hash"], str)
            or _CLAIM_RESULT_HASH_RE.fullmatch(outcome["result_hash"]) is None
            or not _wire_text(outcome["succeeded_at"], nonempty=True)
        ):
            raise ValueError("portable claim extraction outcome has invalid state")
        validate_timestamp_order(
            chunks[outcome["chunk_id"]]["created_at"],
            outcome["succeeded_at"],
            context="portable outcome artifact transaction",
        )

    evidence_by_wire: dict[int, dict] = {}
    evidence_by_natural: dict[tuple[object, ...], dict] = {}
    evidence_revisions: set[tuple[object, ...]] = set()
    current_identities: set[tuple[object, ...]] = set()
    for record in grouped.get("edge_evidence", []):
        for field in ("id", "edge_id", "revision", "evidence_weight"):
            if not _wire_int(record[field], minimum=1):
                raise _invalid_wire("edge_evidence", field)
        if record["id"] > _MAX_SQLITE_ROWID or record["edge_id"] not in edge_by_wire:
            raise ValueError("portable evidence references an absent edge")
        if not _wire_text(record["chunk_id"], nonempty=True) or record["chunk_id"] not in chunks:
            raise ValueError("portable evidence references an absent chunk")
        if not _wire_int(record["polarity"]) or record["polarity"] not in (-1, 1):
            raise _invalid_wire("edge_evidence", "polarity")
        for field in (
            "surface_subject", "surface_object", "value_text", "value_unit",
            "temporal_scope", "extraction_prompt_version", "extracted_at",
            "published_at",
            "superseded_at", "superseded_reason", "source_created_at",
        ):
            if not _wire_text(record[field], nullable=True):
                raise _invalid_wire("edge_evidence", field)
        numeric = record["value_numeric"]
        if numeric is not None and (
            not isinstance(numeric, (int, float)) or isinstance(numeric, bool)
            or not math.isfinite(float(numeric))
        ):
            raise _invalid_wire("edge_evidence", "value_numeric")
        if (
            not _wire_text(record["evidence_kind"], nonempty=True)
            or not _wire_text(record["weight_source"], nonempty=True)
            or not _wire_text(record["interpretation_key"], nonempty=True)
            or not _wire_int(record["is_current"], minimum=0)
            or record["is_current"] not in (0, 1)
        ):
            raise _invalid_wire("edge_evidence", "semantic metadata")
        if record["interpretation_key"] != _semantic_interpretation_key(record):
            raise ValueError("portable evidence interpretation key is forged")
        if bool(record["is_current"]) != (
            record["superseded_at"] is None and record["superseded_reason"] is None
        ):
            raise _invalid_wire("edge_evidence", "revision state")
        if record["superseded_at"] is not None:
            validate_timestamp_order(
                record["extracted_at"],
                record["superseded_at"],
                context="portable evidence retirement",
            )
        if record["provenance_status"] == "canonical":
            if (
                not _wire_int(record["source_message_id"], minimum=1)
                or not _wire_text(record["source_session_id"], nonempty=True)
                or record["source_role"] not in {"user", "assistant", "system", "tool"}
                or not _wire_text(record["source_event_at"], nonempty=True)
                or not _NORMALIZED_EVENT_RE.fullmatch(record["source_event_at"])
                or record["source_event_at"]
                != _portable_source_event(record["source_created_at"])
                or not _wire_text(record["published_at"], nonempty=True)
                or not _NORMALIZED_EVENT_RE.fullmatch(record["published_at"])
                or not _wire_text(record["source_coverage_chunk_id"], nonempty=True)
                or record["source_coverage_version"] != "dream-lossless-message-v1"
                or (record.get("source_peer_id") is None)
                != (record.get("source_workspace_id") is None)
                or not _wire_text(
                    record.get("source_peer_id"), nullable=True, nonempty=True
                )
                or not _wire_text(
                    record.get("source_workspace_id"), nullable=True,
                    nonempty=True,
                )
            ):
                raise ValueError("portable canonical evidence has invalid provenance")
            validate_timestamp_order(
                record["extracted_at"],
                record["published_at"],
                context="portable evidence publication transaction",
            )
            if record["superseded_at"] is not None:
                validate_timestamp_order(
                    record["published_at"],
                    record["superseded_at"],
                    context="portable published evidence retirement",
                )
            proof = coverage.get((
                record["source_message_id"], record["source_coverage_chunk_id"],
                record["source_coverage_version"],
            ))
            member = next((
                item for item in members_by_chunk.get(record["chunk_id"], [])
                if item["source_message_id"] == record["source_message_id"]
                and item["source_session_id"] == record["source_session_id"]
                and item["source_coverage_chunk_id"] == record["source_coverage_chunk_id"]
                and item["source_coverage_version"] == record["source_coverage_version"]
            ), None)
            if (
                proof is None or member is None
                or proof["source_session_id"] != record["source_session_id"]
                or proof["source_role"] != record["source_role"]
                or proof["source_created_at"] != record["source_created_at"]
                or proof.get("source_peer_id") != record.get("source_peer_id")
                or proof.get("source_workspace_id")
                != record.get("source_workspace_id")
            ):
                raise ValueError("portable canonical evidence source proof mismatches")
            validate_timestamp_order(
                chunks[record["chunk_id"]]["created_at"],
                record["extracted_at"],
                context="portable extraction artifact transaction",
            )
            validate_timestamp_order(
                proof["created_at"],
                record["extracted_at"],
                context="portable coverage/evidence transaction",
            )
            base_identity = (
                record["edge_id"], "canonical", record["source_session_id"],
                record["source_message_id"], record["evidence_kind"],
            )
        elif record["provenance_status"] == "legacy_unattributed":
            if any(record[field] is not None for field in (
                "source_message_id", "source_session_id", "source_role",
                "source_created_at", "source_event_at",
                "source_coverage_chunk_id", "source_coverage_version",
                "published_at", "source_peer_id", "source_workspace_id",
            )):
                raise ValueError("portable legacy evidence fabricates source provenance")
            base_identity = (
                record["edge_id"], "legacy", record["chunk_id"],
                record["evidence_kind"],
            )
        else:
            raise _invalid_wire("edge_evidence", "provenance_status")
        revision_identity = (*base_identity, record["revision"])
        natural = (*revision_identity, record["interpretation_key"])
        if (
            record["id"] in evidence_by_wire
            or revision_identity in evidence_revisions
            or natural in evidence_by_natural
        ):
            raise ValueError("portable edge evidence contains a duplicate identity")
        if record["is_current"]:
            if base_identity in current_identities:
                raise ValueError("portable evidence has multiple current revisions")
            current_identities.add(base_identity)
        evidence_by_wire[record["id"]] = record
        evidence_revisions.add(revision_identity)
        evidence_by_natural[natural] = record

    signals_by_wire: dict[int, dict] = {}
    signal_natural: set[tuple[object, ...]] = set()
    for record in grouped.get("edge_evidence_signal", []):
        if (
            not _wire_int(record["id"], minimum=1)
            or not _wire_int(record["edge_id"], minimum=1)
            or record["edge_id"] not in edge_by_wire
            or not _wire_text(record["signal_key"], nonempty=True)
            or not _wire_text(record["signal_kind"], nonempty=True)
            or not _wire_int(record["polarity"])
            or record["polarity"] not in (-1, 1)
            or not _wire_int(record["evidence_weight"], minimum=1)
            or not _wire_int(record["counts_toward_confidence"], minimum=0)
            or record["counts_toward_confidence"] not in (0, 1)
            or not _wire_text(record["details"], nullable=True)
            or not _wire_text(record["created_at"], nullable=True)
            or (record["signal_kind"] == "manual_retraction" and record["polarity"] != -1)
        ):
            raise ValueError("portable evidence signal has invalid state")
        natural = (record["edge_id"], record["signal_kind"], record["signal_key"])
        if record["id"] in signals_by_wire or natural in signal_natural:
            raise ValueError("portable evidence signal contains a duplicate identity")
        signals_by_wire[record["id"]] = record
        signal_natural.add(natural)

    observation_identity: set[tuple[object, ...]] = set()
    generation_semantics: dict[tuple[object, ...], tuple[object, ...]] = {}
    observations_by_base: dict[
        tuple[object, ...], list[dict]
    ] = defaultdict(list)
    observations_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for record in grouped.get("claim_observation", []):
        if (
            not _wire_text(record["chunk_id"], nonempty=True)
            or record["chunk_id"] not in manifests
            or not _wire_int(record["edge_id"], minimum=1)
            or record["edge_id"] not in edge_by_wire
            or not _wire_text(record["source_session_id"], nonempty=True)
            or not _wire_int(record["source_message_id"], minimum=1)
            or not _wire_text(record["evidence_kind"], nonempty=True)
            or not _wire_int(record["polarity"])
            or record["polarity"] not in (-1, 1)
            or not _wire_text(record["prompt_version"], nonempty=True)
            or not _wire_int(record["prompt_generation"], minimum=0)
            or not _wire_int(record["evidence_id"], minimum=1)
            or not _wire_text(record["interpretation_key"], nonempty=True)
            or not _wire_text(record["observed_at"], nullable=True)
        ):
            raise ValueError("portable claim observation has invalid state")
        ev = evidence_by_wire.get(record["evidence_id"])
        if ev is None or not (
            ev["provenance_status"] == "canonical"
            and ev["edge_id"] == record["edge_id"]
            and ev["source_session_id"] == record["source_session_id"]
            and ev["source_message_id"] == record["source_message_id"]
            and ev["evidence_kind"] == record["evidence_kind"]
            and ev["polarity"] == record["polarity"]
            and ev["interpretation_key"] == record["interpretation_key"]
        ):
            raise ValueError("portable claim observation mismatches its evidence")
        outcome = outcomes.get(record["chunk_id"])
        if outcome is not None:
            validate_timestamp_order(
                chunks[record["chunk_id"]]["created_at"],
                record["observed_at"],
                context="portable observation artifact transaction",
            )
            validate_timestamp_order(
                ev["extracted_at"],
                record["observed_at"],
                context="portable claim observation",
            )
            validate_timestamp_order(
                record["observed_at"],
                outcome["succeeded_at"],
                context="portable claim outcome publication",
                maximum_gap_seconds=300,
            )
            validate_timestamp_order(
                ev["published_at"],
                outcome["succeeded_at"],
                context="portable evidence publication authority",
            )
        if evidence_ledger.prompt_generation(record["prompt_version"]) != record["prompt_generation"]:
            raise ValueError("portable claim observation has invalid prompt generation")
        if not any(
            member["source_message_id"] == record["source_message_id"]
            and member["source_session_id"] == record["source_session_id"]
            for member in members_by_chunk[record["chunk_id"]]
        ):
            raise ValueError("portable claim observation cites a nonmember source")
        identity = (
            record["chunk_id"], record["edge_id"], record["source_session_id"],
            record["source_message_id"], record["evidence_kind"],
        )
        if identity in observation_identity:
            raise ValueError("portable claim observation contains a duplicate identity")
        observation_identity.add(identity)
        generation_key = (
            record["edge_id"], record["source_session_id"],
            record["source_message_id"], record["evidence_kind"],
            record["prompt_generation"],
        )
        semantic = (record["polarity"], record["interpretation_key"])
        previous = generation_semantics.setdefault(generation_key, semantic)
        if previous != semantic:
            raise ValueError("portable observations diverge at one prompt generation")
        observations_by_base[(
            record["edge_id"], record["source_session_id"],
            record["source_message_id"], record["evidence_kind"],
        )].append(record)
        observations_by_chunk[record["chunk_id"]].append(record)

    if set(observations_by_chunk) - set(outcomes):
        raise ValueError(
            "portable claim observations have no whole-chunk outcome authority"
        )
    for chunk_id, outcome in outcomes.items():
        observations = observations_by_chunk.get(chunk_id, [])
        if any(
            record["prompt_version"] != outcome["prompt_version"]
            or int(record["prompt_generation"])
            != int(outcome["prompt_generation"])
            for record in observations
        ):
            raise ValueError(
                "portable claim outcome disagrees with its observation authority"
            )
        if outcome["result_hash"] != _wire_claim_result_hash(
            observations, edge_by_wire
        ):
            raise ValueError("portable claim extraction outcome hash mismatches")

    canonical_by_base: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for evidence in evidence_by_wire.values():
        if evidence["provenance_status"] == "canonical":
            canonical_by_base[(
                evidence["edge_id"], evidence["source_session_id"],
                evidence["source_message_id"], evidence["evidence_kind"],
            )].append(evidence)
    for base, revisions in canonical_by_base.items():
        observations = observations_by_base.get(base, [])
        currents = [record for record in revisions if record["is_current"]]
        if not observations:
            if currents:
                raise ValueError("portable orphaned canonical evidence is current")
            continue
        winning_generation = max(
            int(record["prompt_generation"]) for record in observations
        )
        winning_semantics = {
            (int(record["polarity"]), record["interpretation_key"])
            for record in observations
            if int(record["prompt_generation"]) == winning_generation
        }
        if len(winning_semantics) != 1 or len(currents) != 1:
            raise ValueError("portable claim prompt authority is inconsistent")
        if (
            int(currents[0]["polarity"]), currents[0]["interpretation_key"]
        ) != next(iter(winning_semantics)):
            raise ValueError("portable current evidence violates prompt authority")

    lifecycle_by_wire: dict[int, dict] = {}
    lifecycle_natural: set[tuple[object, ...]] = set()
    for record in grouped.get("edge_lifecycle", []):
        if (
            not _wire_int(record["id"], minimum=1)
            or not _wire_int(record["edge_id"], minimum=1)
            or record["edge_id"] not in edge_by_wire
            or not _wire_text(record["event_key"], nonempty=True)
            or record["event_kind"] not in {
                "claim_assertion", "manual_retraction", "phase3_retraction",
                "value_supersession", "legacy_state",
            }
            or not _wire_int(record["direction"])
            or record["direction"] not in (-1, 1)
            or not _wire_text(record["event_at"], nonempty=True)
            or not _NORMALIZED_EVENT_RE.fullmatch(record["event_at"])
            or not _wire_int(record["dependency_count"], minimum=0)
            or not _wire_text(record["details"], nullable=True)
            or not _wire_text(record["created_at"], nullable=True)
        ):
            raise ValueError("portable lifecycle event has invalid state")
        source_id = record["source_evidence_id"]
        if source_id is not None and not _wire_int(source_id, minimum=1):
            raise _invalid_wire("edge_lifecycle", "source_evidence_id")
        if record["event_kind"] == "claim_assertion":
            source = evidence_by_wire.get(source_id)
            if source is None or not (
                record["direction"] == 1 and record["dependency_count"] == 0
                and source["edge_id"] == record["edge_id"]
                and source["provenance_status"] == "canonical"
                and source["polarity"] == 1
                and source["source_event_at"] == record["event_at"]
                and record["details"] is None
            ):
                raise ValueError("portable claim lifecycle source mismatches")
            expected_key = evidence_ledger.claim_assertion_event_key(
                source["source_session_id"], source["source_message_id"],
                source["evidence_kind"], source["revision"],
            )
            if record["event_key"] != expected_key:
                raise ValueError("portable claim lifecycle key is forged")
        elif record["event_kind"] == "manual_retraction":
            if source_id is not None or record["direction"] != -1 or record["dependency_count"]:
                raise ValueError("portable manual lifecycle event is invalid")
        elif record["event_kind"] in {"phase3_retraction", "value_supersession"}:
            expected_details = (
                "confidence_or_negative_dominance"
                if record["event_kind"] == "phase3_retraction"
                else "newer typed value superseded this edge"
            )
            if (
                source_id is not None or record["direction"] != -1
                or record["dependency_count"] < 1
                or record["details"] != expected_details
            ):
                raise ValueError("portable decision lifecycle event is invalid")
        elif (
            source_id is not None
            or record["dependency_count"] != 0
            or record["event_key"] not in {
                "legacy-state", "portable-v6-legacy-state",
                "portable-v6-legacy-0-open", "portable-v6-legacy-1-close",
            }
        ):
            raise ValueError("portable legacy lifecycle event is invalid")
        natural = (record["edge_id"], record["event_key"])
        if record["id"] in lifecycle_by_wire or natural in lifecycle_natural:
            raise ValueError("portable lifecycle contains a duplicate identity")
        lifecycle_by_wire[record["id"]] = record
        lifecycle_natural.add(natural)

    deps_by_lifecycle: dict[int, list[dict]] = defaultdict(list)
    dependency_pairs: set[tuple[int, int]] = set()
    for record in grouped.get("lifecycle_dependency", []):
        if not _wire_int(record["lifecycle_id"], minimum=1) or not _wire_int(
            record["evidence_id"], minimum=1
        ):
            raise ValueError("portable lifecycle dependency has invalid handles")
        pair = (record["lifecycle_id"], record["evidence_id"])
        if pair in dependency_pairs:
            raise ValueError("portable lifecycle dependency is duplicated")
        dependency_pairs.add(pair)
        lifecycle = lifecycle_by_wire.get(record["lifecycle_id"])
        cause = evidence_by_wire.get(record["evidence_id"])
        if lifecycle is None or cause is None:
            raise ValueError("portable lifecycle dependency has an absent parent")
        owner = edge_by_wire[lifecycle["edge_id"]]
        cause_edge = edge_by_wire[cause["edge_id"]]
        if lifecycle["event_kind"] == "phase3_retraction":
            valid = cause["edge_id"] == lifecycle["edge_id"] and cause["polarity"] == -1
        elif lifecycle["event_kind"] == "value_supersession":
            valid = (
                cause["polarity"] == 1 and owner["derived"] == 0
                and cause_edge["derived"] == 0
                and owner["subject_canonical"] == cause_edge["subject_canonical"]
                and owner["predicate"] == cause_edge["predicate"]
                and owner["object_canonical"] != cause_edge["object_canonical"]
            )
        else:
            valid = False
        if not valid:
            raise ValueError("portable lifecycle dependency has invalid semantics")
        deps_by_lifecycle[record["lifecycle_id"]].append(cause)
    for lifecycle_id, lifecycle in lifecycle_by_wire.items():
        causes = deps_by_lifecycle.get(lifecycle_id, [])
        if len(causes) != lifecycle["dependency_count"]:
            raise ValueError("portable lifecycle dependency count mismatches")
        if causes and lifecycle["event_at"] != max(
            _evidence_wire_event(cause)
            for cause in causes
        ):
            raise ValueError("portable lifecycle decision time mismatches its causes")
        if lifecycle["event_kind"] == "claim_assertion":
            source = evidence_by_wire[int(lifecycle["source_evidence_id"])]
            validate_timestamp_order(
                source["extracted_at"],
                lifecycle["created_at"],
                context="portable claim lifecycle transaction",
            )
            validate_timestamp_order(
                lifecycle["created_at"],
                source["published_at"],
                context="portable claim lifecycle publication transaction",
            )
        elif lifecycle["event_kind"] in {
            "phase3_retraction", "value_supersession"
        }:
            for cause in causes:
                validate_timestamp_order(
                    (
                        cause["published_at"]
                        if cause["provenance_status"] == "canonical"
                        else cause["extracted_at"]
                    ),
                    lifecycle["created_at"],
                    context="portable decision lifecycle transaction",
                )
        if lifecycle["event_kind"] == "phase3_retraction":
            if lifecycle["event_key"] != _wire_phase3_event_key(causes):
                raise ValueError("portable phase3 lifecycle key is forged")
        elif lifecycle["event_kind"] == "value_supersession":
            if len(causes) != 1 or lifecycle["event_key"] != _wire_value_event_key(
                lifecycle, causes[0], edge_by_wire
            ):
                raise ValueError("portable value lifecycle key is forged")

    expected_assertion_sources = {
        int(evidence_id)
        for evidence_id, evidence in evidence_by_wire.items()
        if evidence["provenance_status"] == "canonical"
        and int(evidence["polarity"]) == 1
    }
    actual_assertion_sources = {
        int(lifecycle["source_evidence_id"])
        for lifecycle in lifecycle_by_wire.values()
        if lifecycle["event_kind"] == "claim_assertion"
    }
    if actual_assertion_sources != expected_assertion_sources:
        raise ValueError(
            "portable positive evidence/lifecycle assertion coverage mismatches"
        )

    manual_signals = {
        (
            int(signal["edge_id"]),
            evidence_ledger.manual_retraction_event_key(signal["signal_key"]),
        ): signal
        for signal in signals_by_wire.values()
        if signal["signal_kind"] == "manual_retraction"
    }
    manual_lifecycle = {
        (int(event["edge_id"]), event["event_key"]): event
        for event in lifecycle_by_wire.values()
        if event["event_kind"] == "manual_retraction"
    }
    if set(manual_signals) != set(manual_lifecycle):
        raise ValueError("portable manual signal/lifecycle coupling is incomplete")
    for identity, signal in manual_signals.items():
        event = manual_lifecycle[identity]
        validate_timestamp_order(
            signal["created_at"],
            event["created_at"],
            context="portable manual lifecycle transaction",
        )
        signal_at = _normalized_wire_event(signal["created_at"])
        open_coordinates = [
            candidate["event_at"]
            for candidate in lifecycle_by_wire.values()
            if candidate["edge_id"] == signal["edge_id"]
            and candidate["direction"] == 1
            and _normalized_wire_event(candidate["created_at"]) <= signal_at
        ]
        expected_at = max([signal_at, *open_coordinates])
        if (
            event["details"] != signal["details"]
            or event["event_at"] != expected_at
        ):
            raise ValueError("portable manual signal/lifecycle semantics mismatch")

    def evidence_order(record: dict) -> tuple[object, ...]:
        if record["provenance_status"] == "canonical":
            return (
                1, record["source_session_id"], int(record["source_message_id"]),
                record["evidence_kind"], int(record["revision"]),
                record["interpretation_key"],
            )
        return (
            0, record["chunk_id"], record["evidence_kind"],
            int(record["revision"]), record["interpretation_key"],
        )

    # Validate the current materialized interval against the exact same active
    # event policy as bitemporal._ordered_events/recompute_edge_interval. This
    # prevents a checksummed lifecycle-A/edge-B envelope from changing on first
    # import and then colliding on exact replay.
    lifecycle_by_edge: dict[int, list[dict]] = defaultdict(list)
    canonical_edges = {
        record["edge_id"] for record in evidence_by_wire.values()
        if record["provenance_status"] == "canonical" and record["is_current"]
    }
    for wire_id, lifecycle in lifecycle_by_wire.items():
        source = (
            evidence_by_wire.get(lifecycle["source_evidence_id"])
            if lifecycle["source_evidence_id"] is not None else None
        )
        if source is not None and not source["is_current"]:
            continue
        if lifecycle["event_kind"] == "legacy_state" and lifecycle["edge_id"] in canonical_edges:
            continue
        causes = deps_by_lifecycle.get(wire_id, [])
        if lifecycle["dependency_count"] and not all(cause["is_current"] for cause in causes):
            continue
        if source is not None:
            causal = evidence_order(source)
        elif lifecycle["event_kind"] in {"phase3_retraction", "value_supersession"}:
            causal = max((evidence_order(cause) for cause in causes), default=(-2, "missing"))
        elif lifecycle["event_kind"] == "manual_retraction":
            causal = (2, lifecycle["event_key"])
        else:
            causal = (-1, lifecycle["event_key"])
        lifecycle_by_edge[lifecycle["edge_id"]].append({
            **lifecycle,
            "_sort": (
                lifecycle["event_at"], causal,
                0 if lifecycle["event_kind"] == "claim_assertion" else 1,
                lifecycle["event_key"],
            ),
        })
    for edge_id, edge in edge_by_wire.items():
        events = lifecycle_by_edge.get(edge_id, [])
        if not events:
            edge_evidence = [
                evidence for evidence in evidence_by_wire.values()
                if int(evidence["edge_id"]) == int(edge_id)
            ]
            current_positive = any(
                int(evidence["is_current"]) == 1
                and int(evidence["polarity"]) == 1
                for evidence in edge_evidence
            ) or any(
                int(signal["edge_id"]) == int(edge_id)
                and int(signal["polarity"]) == 1
                and int(signal["counts_toward_confidence"]) == 1
                for signal in signals_by_wire.values()
            )
            # Successful whole-chunk replay can retire the last assertion (or
            # the cause of a value/phase decision), leaving only inactive
            # append-only lifecycle history. Runtime reconciliation then uses
            # this conservative terminal fallback. A provenance-empty direct
            # row is still rejected: it has no portable authority at all.
            if not edge_evidence or current_positive:
                raise ValueError("portable direct edge has no lifecycle authority")
            fallback = max(
                (
                    evidence["source_event_at"]
                    for evidence in edge_evidence
                    if evidence["provenance_status"] == "canonical"
                ),
                default=(
                    _normalized_wire_event(edge["valid_at"])
                    if edge["valid_at"] is not None
                    else "0001-01-01T00:00:00.000Z"
                ),
            )
            valid_at = max(
                (
                    evidence["source_event_at"]
                    for evidence in edge_evidence
                    if evidence["provenance_status"] == "canonical"
                    and int(evidence["polarity"]) == 1
                ),
                default=(
                    _normalized_wire_event(edge["valid_at"])
                    if edge["valid_at"] is not None
                    else "0001-01-01T00:00:00.000Z"
                ),
            )
            expected_state = (
                "retracted", valid_at,
                max(valid_at or fallback, fallback),
            )
            actual_state = (
                edge["status"],
                _normalized_wire_event(edge["valid_at"])
                if edge["valid_at"] is not None else None,
                _normalized_wire_event(edge["invalid_at"])
                if edge["invalid_at"] is not None else None,
            )
            if actual_state != expected_state:
                raise ValueError(
                    "portable edge interval disagrees with lifecycle ledger"
                )
            edge["valid_at"] = expected_state[1]
            edge["invalid_at"] = expected_state[2]
            continue
        ordered = sorted(events, key=lambda row: row["_sort"])
        open_start = None
        last_closed_start = None
        invalid_at = None
        state_open = False
        for event in ordered:
            if event["direction"] == 1:
                if not state_open:
                    open_start = event["event_at"]
                state_open = True
                invalid_at = None
            else:
                if state_open:
                    last_closed_start = open_start
                    invalid_at = event["event_at"]
                state_open = False
        if state_open:
            expected_state = ("active", open_start, None)
        else:
            valid_at = _normalized_wire_event(
                last_closed_start or open_start or edge["first_seen"]
            )
            closed_at = invalid_at or valid_at
            if valid_at is not None and closed_at is not None:
                closed_at = max(str(valid_at), str(closed_at))
            expected_state = ("retracted", valid_at, closed_at)
        actual_state = (
            edge["status"],
            _normalized_wire_event(edge["valid_at"])
            if edge["valid_at"] is not None else None,
            _normalized_wire_event(edge["invalid_at"])
            if edge["invalid_at"] is not None else None,
        )
        if actual_state != expected_state:
            raise ValueError("portable edge interval disagrees with lifecycle ledger")
        # Canonicalize only a semantically equivalent spelling in the in-memory
        # envelope. A different status/instant fails above instead of being
        # silently healed during export.
        edge["valid_at"] = expected_state[1]
        edge["invalid_at"] = expected_state[2]

    totals: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for record in evidence_by_wire.values():
        if record["is_current"]:
            totals[record["edge_id"]][0 if record["polarity"] == 1 else 1] += int(
                record["evidence_weight"]
            )
    for record in signals_by_wire.values():
        if record["counts_toward_confidence"]:
            totals[record["edge_id"]][0 if record["polarity"] == 1 else 1] += int(
                record["evidence_weight"]
            )
    claim_edges = {
        record["edge_id"] for record in evidence_by_wire.values()
    } | {record["edge_id"] for record in signals_by_wire.values()} | {
        record["edge_id"] for record in grouped.get("claim_observation", [])
    } | {record["edge_id"] for record in lifecycle_by_wire.values()}
    for edge_id, edge in edge_by_wire.items():
        if edge["derived"]:
            if edge_id in claim_edges:
                raise ValueError("portable derived edge carries observed provenance")
            continue
        if (edge["pos_evidence"], edge["neg_evidence"]) != tuple(totals[edge_id]):
            raise ValueError("portable edge cache disagrees with its evidence ledger")


def _v6_existing_row_is_identical(conn, kind: str, record: dict) -> bool:
    """Return whether an exact idempotent v6 target row already exists.

    A deterministic key collision is never an invitation to reinterpret the
    target's row as the imported source. In particular, coverage/profile
    provenance is an authority boundary and an ``OR IGNORE`` collision must
    not bind a claim to unrelated local bytes.
    """
    key_sql: str
    key_params: tuple[object, ...]
    compare_cols = list(_COLS_BY_KIND[kind])
    if kind == "session":
        key_sql, key_params = "id = ?", (record["id"],)
        # Fact cursor/retry state is a projection of the v10 history chain and
        # merges monotonically in _import_v10_fact_state. Pre-v10 imports carry
        # only a conservative rewind and must never regress a newer target.
        compare_cols = [
            column for column in compare_cols
            if column not in _SESSION_FACT_FIELDS
        ]
    elif kind == "peer":
        key_sql = "id = ? AND workspace_id = ?"
        key_params = (record["id"], record["workspace_id"])
        # Metadata/registration time are registry presentation state, not
        # evidence identity. Existing role must agree; local metadata wins.
        compare_cols = ["id", "workspace_id", "role"]
    elif kind == "session_peer":
        key_sql = "session_id = ? AND workspace_id = ? AND peer_id = ?"
        key_params = (
            record["session_id"], record["workspace_id"], record["peer_id"],
        )
        compare_cols = [
            "session_id", "workspace_id", "peer_id", "configuration",
        ]
    elif kind == "episode":
        compare_cols = list(_COLS_BY_KIND[kind])
        source_columns = {
            "source_manifest_version", "source_manifest_count",
            "source_manifest_hash", "source_manifest_complete",
        }
        base_columns = [
            column for column in compare_cols if column not in source_columns
        ]
        row = conn.execute(
            f"SELECT {', '.join(compare_cols)} FROM episodes WHERE id=?",
            (record["id"],),
        ).fetchone()
        if row is None:
            return False
        if any(row[column] != record[column] for column in base_columns):
            raise ValueError("portable episode collides with different target state")
        target_complete = row["source_manifest_complete"]
        imported_complete = record["source_manifest_complete"]
        if target_complete not in (0, 1):
            raise ValueError("portable episode collides with invalid target provenance")
        if target_complete == 1:
            if load_episode_source_manifest(conn, record["id"]) is None:
                raise ValueError(
                    "portable episode collides with corrupt target provenance"
                )
            if imported_complete == 1 and any(
                row[column] != record[column] for column in source_columns
            ):
                raise ValueError(
                    "portable episode collides with different complete provenance"
                )
            # A legacy/incomplete import can never erase stronger local proof.
            return True
        if not (
            row["source_manifest_version"] is None
            and row["source_manifest_count"] == 0
            and row["source_manifest_hash"] is None
        ):
            raise ValueError("portable episode collides with invalid target provenance")
        # Same bytes may monotonically upgrade from incomplete to imported
        # complete proof. Skip the base INSERT; child rows and publication run
        # later in the same transaction.
        return True
    elif kind in {"chunk", "procedure"}:
        key_sql, key_params = "id = ?", (record["id"],)
    elif kind == "episode_source_occurrence":
        key_sql = "episode_id = ? AND ordinal = ?"
        key_params = (record["episode_id"], record["ordinal"])
    elif kind == "message_retention_coverage":
        key_sql = "message_id = ? AND chunk_id = ? AND coverage_version = ?"
        key_params = (
            record["message_id"], record["chunk_id"], record["coverage_version"],
        )
    elif kind == "edge":
        key_sql = (
            "subject_canonical = ? AND predicate = ? AND object_canonical = ?"
        )
        key_params = (
            record["subject_canonical"], record["predicate"],
            record["object_canonical"],
        )
        compare_cols.remove("id")
    elif kind == "profile_entry":
        key_sql, key_params = "text = ?", (record["text"],)
        compare_cols.remove("id")
    else:
        return False
    table = _TABLE_BY_KIND[kind]
    row = conn.execute(
        f"SELECT {', '.join(compare_cols)} FROM {table} WHERE {key_sql}",
        key_params,
    ).fetchone()
    if row is None:
        return False
    if kind == "session_peer":
        try:
            existing_configuration = loads_strict_json(row["configuration"])
            imported_configuration = loads_strict_json(record["configuration"])
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError(
                "portable session_peer collides with invalid target state"
            ) from None
        if (
            tuple(row[column] for column in compare_cols[:-1])
            != tuple(record[column] for column in compare_cols[:-1])
            or existing_configuration != imported_configuration
        ):
            raise ValueError(
                "portable session_peer collides with different target state"
            )
        return True
    if any(row[column] != record[column] for column in compare_cols):
        raise ValueError(f"portable {kind} collides with different target state")
    return True


def _preflight_v6_target_collisions(
    conn, grouped: dict[str, list[dict]], *, merge_v7_edges: bool = False
) -> None:
    for kind in (
        "session", "peer", "session_peer", "chunk",
        "message_retention_coverage", "episode",
        "episode_source_occurrence",
        "procedure", "edge", "profile_entry",
    ):
        for record in grouped.get(kind, []):
            if merge_v7_edges and kind == "edge":
                # A v7 edge row is a materialized view over the history that
                # follows it. Same-natural direct edges merge those histories;
                # their counters/intervals therefore need not match before the
                # union is applied. Derived rows are removed as rebuildable
                # cache before this check.
                existing = conn.execute(
                    "SELECT derived FROM knowledge_graph WHERE "
                    "subject_canonical=? AND predicate=? AND object_canonical=?",
                    _edge_natural(record),
                ).fetchone()
                if existing is not None and int(existing["derived"]) != 0:
                    raise ValueError("portable direct edge collides with derived target")
                continue
            _v6_existing_row_is_identical(conn, kind, record)


def _preflight_v7_target_aliases(conn, grouped: dict[str, list[dict]]) -> None:
    """Reject an import whose edge identities would resolve differently.

    Re-keying a graph endpoint also requires rewriting every evidence and
    lifecycle natural identity. Until that is explicitly requested, fail
    closed instead of creating a split edge that normal queries cannot reach.
    """
    mappings = {
        str(row["alias"]): str(row["canonical"])
        for row in conn.execute(
            "SELECT alias,canonical FROM entity_aliases ORDER BY alias"
        ).fetchall()
    }
    for record in grouped.get("entity_alias", []):
        alias = str(record["alias"])
        canonical = str(record["canonical"])
        previous = mappings.get(alias)
        if previous is not None and previous != canonical:
            raise ValueError("portable entity alias collides with target")
        mappings[alias] = canonical
    for alias, canonical in mappings.items():
        if (
            canonicalize.normalize(alias) != alias
            or canonicalize.normalize(canonical) != canonical
        ):
            raise ValueError("combined entity aliases are not canonicalized")
        next_value = mappings.get(canonical)
        if next_value is not None and next_value != canonical:
            raise ValueError("combined entity aliases are cyclic or chained")
    for edge in grouped.get("edge", []):
        for endpoint in (
            edge["subject_canonical"], edge["object_canonical"],
        ):
            if mappings.get(endpoint, endpoint) != endpoint:
                raise ValueError(
                    "portable edge endpoint resolves differently in target aliases"
                )


_FACT_WIRE_KINDS = (
    "fact_extraction_outcome", "fact_extraction_source_occurrence",
    "fact_extraction_revision", "narrative_fact",
    "narrative_fact_lifecycle",
)


def _target_fact_wire_state(conn, slice_key: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for kind in _FACT_WIRE_KINDS:
        columns = list(_V10_COLS_BY_KIND[kind])
        if kind == "fact_extraction_outcome":
            rows = conn.execute(
                f"SELECT {', '.join(columns)} FROM fact_extraction_outcomes "
                "WHERE slice_key=?", (slice_key,),
            ).fetchall()
        elif kind in {
            "fact_extraction_source_occurrence", "fact_extraction_revision",
        }:
            rows = conn.execute(
                f"SELECT {', '.join(columns)} FROM {_V10_TABLE_BY_KIND[kind]} "
                "WHERE slice_key=?", (slice_key,),
            ).fetchall()
        elif kind == "narrative_fact":
            rows = conn.execute(
                f"SELECT {', '.join(columns)} FROM narrative_facts "
                "WHERE source_outcome_key=?", (slice_key,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT f.source_outcome_key,f.fact_key,l.generation,"
                "l.direction,l.event_at,l.prompt_version,l.result_hash,"
                "l.recorded_at FROM narrative_fact_lifecycle l "
                "JOIN narrative_facts f ON f.id=l.fact_id "
                "WHERE f.source_outcome_key=?", (slice_key,),
            ).fetchall()
        records = [{column: row[column] for column in columns} for row in rows]
        records.sort(key=_portable_record_sort_key)
        result[kind] = records
    return result


def _fact_wire_by_slice(
    grouped: dict[str, list[dict]]
) -> dict[str, dict[str, list[dict]]]:
    imported_by_slice: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for kind in _FACT_WIRE_KINDS:
        owner_field = (
            "source_outcome_key"
            if kind in {"narrative_fact", "narrative_fact_lifecycle"}
            else "slice_key"
        )
        for record in grouped.get(kind, []):
            imported_by_slice[str(record[owner_field])][kind].append(record)
    for state in imported_by_slice.values():
        for records in state.values():
            records.sort(key=_portable_record_sort_key)
    return imported_by_slice


def _fact_slice_prefix_relation(
    imported: dict[str, list[dict]], target: dict[str, list[dict]]
) -> int:
    """Return -1/0/1 when imported history is older/equal/newer."""

    imported_outcome = imported["fact_extraction_outcome"][0]
    target_outcome = target["fact_extraction_outcome"][0]
    mutable_outcome = {
        "prompt_version", "generation", "outcome_status", "result_hash",
        "succeeded_at",
    }
    if any(
        imported_outcome[column] != target_outcome[column]
        for column in _V10_COLS_BY_KIND["fact_extraction_outcome"]
        if column not in mutable_outcome
    ) or imported["fact_extraction_source_occurrence"] != target[
        "fact_extraction_source_occurrence"
    ]:
        raise ValueError("portable fact source unit diverges from target")

    imported_revisions = sorted(
        imported["fact_extraction_revision"], key=lambda row: row["generation"]
    )
    target_revisions = sorted(
        target["fact_extraction_revision"], key=lambda row: row["generation"]
    )
    shared = min(len(imported_revisions), len(target_revisions))
    if imported_revisions[:shared] != target_revisions[:shared]:
        raise ValueError("portable fact revision history diverges from target")
    relation = (len(imported_revisions) > len(target_revisions)) - (
        len(imported_revisions) < len(target_revisions)
    )
    shorter_generation = min(
        int(imported_outcome["generation"]), int(target_outcome["generation"])
    )
    imported_facts = {
        str(record["fact_key"]): record for record in imported["narrative_fact"]
    }
    target_facts = {
        str(record["fact_key"]): record for record in target["narrative_fact"]
    }
    immutable_fact_columns = {
        "session_id", "start_message_id", "end_message_id", "text",
        "fact_date", "entities", "prompt_version", "valid_at",
        "source_outcome_key", "fact_key", "created_at",
    }
    imported_events: dict[str, list[dict]] = defaultdict(list)
    target_events: dict[str, list[dict]] = defaultdict(list)
    for event in imported["narrative_fact_lifecycle"]:
        imported_events[str(event["fact_key"])].append(event)
    for event in target["narrative_fact_lifecycle"]:
        target_events[str(event["fact_key"])].append(event)
    all_keys = set(imported_facts) | set(target_facts)
    for fact_key in all_keys:
        left = imported_facts.get(fact_key)
        right = target_facts.get(fact_key)
        left_events = sorted(
            imported_events.get(fact_key, []), key=lambda row: row["generation"]
        )
        right_events = sorted(
            target_events.get(fact_key, []), key=lambda row: row["generation"]
        )
        if left is not None and right is not None:
            if any(left[column] != right[column] for column in immutable_fact_columns):
                raise ValueError("portable narrative fact bytes diverge from target")
            prefix = min(len(left_events), len(right_events))
            if left_events[:prefix] != right_events[:prefix]:
                raise ValueError("portable fact lifecycle diverges from target")
            expected_event_relation = (len(left_events) > len(right_events)) - (
                len(left_events) < len(right_events)
            )
            if expected_event_relation and relation and expected_event_relation != relation:
                raise ValueError("portable fact histories are not monotonic prefixes")
        elif left is not None:
            if not left_events or int(left_events[0]["generation"]) <= shorter_generation:
                raise ValueError("portable target omits historical fact payload")
            if relation < 0:
                raise ValueError("older portable history contains a future fact")
        else:
            if not right_events or int(right_events[0]["generation"]) <= shorter_generation:
                raise ValueError("portable snapshot omits historical fact payload")
            if relation > 0:
                raise ValueError("newer portable history omits a target fact")
    if relation == 0:
        for kind in (
            "fact_extraction_outcome", "narrative_fact",
            "narrative_fact_lifecycle",
        ):
            if imported[kind] != target[kind]:
                raise ValueError("equal-generation fact state diverges from target")
    return relation


def _portable_fact_chain(
    grouped: dict[str, list[dict]],
    session_id: str,
    *,
    terminal: tuple[object, object, int] | None = None,
    owned_outcomes: list[dict] | None = None,
) -> list[str]:
    outcomes = owned_outcomes if owned_outcomes is not None else [
        record for record in grouped.get("fact_extraction_outcome", [])
        if record["session_id"] == session_id
    ]
    by_before: dict[tuple[object, object, int], dict] = {}
    for record in outcomes:
        key = (
            record["cursor_before_message_id"],
            record["cursor_before_partial_message_id"],
            int(record["cursor_before_offset"]),
        )
        if key in by_before:
            raise ValueError("portable fact cursor chain forks")
        by_before[key] = record
    coordinate: tuple[object, object, int] = (None, None, 0)
    result: list[str] = []
    seen: set[str] = set()
    while coordinate in by_before:
        record = by_before[coordinate]
        slice_key = str(record["slice_key"])
        if slice_key in seen:
            raise ValueError("portable fact cursor chain cycles")
        seen.add(slice_key)
        result.append(slice_key)
        coordinate = (
            record["cursor_after_message_id"],
            record["cursor_after_partial_message_id"],
            int(record["cursor_after_offset"]),
        )
    if len(seen) != len(outcomes):
        raise ValueError("portable fact cursor chain has an orphan outcome")
    if terminal is None:
        session = next(
            (
                record for record in grouped.get("session", [])
                if record["id"] == session_id
            ),
            None,
        )
        if session is not None:
            terminal = (
                session.get("facts_cursor_message_id"),
                session.get("facts_cursor_partial_message_id"),
                int(session.get("facts_cursor_offset") or 0),
            )
    if terminal is not None and coordinate != terminal:
        raise ValueError("portable fact chain does not terminate at its cursor")
    return result


def _validated_target_fact_chain(conn, session_id: str) -> list[str]:
    """Load one complete target ledger, rejecting cursor or proof corruption."""

    target_session = conn.execute(
        "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
        "facts_cursor_offset,facts_cursor_prompt_version,facts_retry_count,"
        "facts_retry_config_version,facts_quarantined,facts_message_id "
        "FROM sessions WHERE id=?", (session_id,),
    ).fetchone()
    if target_session is None:
        return []
    if (
        not facts_retry_state_is_valid(
            target_session["facts_retry_count"],
            target_session["facts_retry_config_version"],
            target_session["facts_quarantined"],
        )
        or target_session["facts_message_id"]
        != target_session["facts_cursor_message_id"]
        or (target_session["facts_cursor_partial_message_id"] is None)
        != (int(target_session["facts_cursor_offset"] or 0) == 0)
        or (
            target_session["facts_cursor_prompt_version"] is not None
            and not facts_generation_is_recognized(
                target_session["facts_cursor_prompt_version"]
            )
        )
    ):
        raise ValueError("portable facts collide with invalid target cursor state")
    terminal = (
        target_session["facts_cursor_message_id"],
        target_session["facts_cursor_partial_message_id"],
        int(target_session["facts_cursor_offset"] or 0),
    )
    target_grouped: dict[str, list[dict]] = defaultdict(list)
    owned_outcomes: list[dict] = []
    chain_cache: dict[str, bool | None] = {}
    for row in conn.execute(
        "SELECT slice_key FROM fact_extraction_outcomes "
        "WHERE session_id=? ORDER BY slice_key", (session_id,),
    ).fetchall():
        slice_key = str(row["slice_key"])
        if load_fact_outcome_source_manifest(
            conn, slice_key, verify_result=True, chain_cache=chain_cache
        ) is None:
            raise ValueError("portable facts collide with corrupt target history")
        state = _target_fact_wire_state(conn, slice_key)
        for kind, records in state.items():
            target_grouped[kind].extend(records)
        owned_outcomes.extend(state["fact_extraction_outcome"])
    chain = _portable_fact_chain(
        target_grouped, session_id, terminal=terminal,
        owned_outcomes=owned_outcomes,
    )
    if chain and not facts_generation_is_recognized(
        target_session["facts_cursor_prompt_version"]
    ):
        raise ValueError("target fact history lacks a recognized generation")
    return chain


def _preflight_v10_fact_target_collisions(
    conn, grouped: dict[str, list[dict]]
) -> dict[str, int]:
    imported_by_slice = _fact_wire_by_slice(grouped)
    slice_relations: dict[str, int] = {}
    target_chain_cache: dict[str, bool | None] = {}
    for slice_key, imported in imported_by_slice.items():
        existing = conn.execute(
            "SELECT 1 FROM fact_extraction_outcomes WHERE slice_key=?",
            (slice_key,),
        ).fetchone()
        if existing is None:
            slice_relations[slice_key] = 1
            continue
        if load_fact_outcome_source_manifest(
            conn, slice_key, verify_result=True,
            chain_cache=target_chain_cache,
        ) is None:
            raise ValueError("portable fact outcome collides with corrupt target state")
        target = _target_fact_wire_state(conn, slice_key)
        slice_relations[slice_key] = _fact_slice_prefix_relation(imported, target)

    imported_outcomes_by_session: dict[str, list[dict]] = defaultdict(list)
    for outcome in grouped.get("fact_extraction_outcome", []):
        imported_outcomes_by_session[str(outcome["session_id"])].append(outcome)
    session_relations: dict[str, int] = {}
    for session in grouped.get("session", []):
        session_id = str(session["id"])
        imported_terminal = (
            session.get("facts_cursor_message_id"),
            session.get("facts_cursor_partial_message_id"),
            int(session.get("facts_cursor_offset") or 0),
        )
        imported_chain = _portable_fact_chain(
            grouped, session_id, terminal=imported_terminal,
            owned_outcomes=imported_outcomes_by_session.get(session_id, []),
        )
        target_session = conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_cursor_offset,facts_cursor_prompt_version,facts_retry_count,"
            "facts_retry_config_version,facts_quarantined,facts_message_id "
            "FROM sessions WHERE id=?", (session_id,),
        ).fetchone()
        if target_session is None:
            session_relations[session_id] = 1
            continue
        target_chain = _validated_target_fact_chain(conn, session_id)
        shared = min(len(imported_chain), len(target_chain))
        if imported_chain[:shared] != target_chain[:shared]:
            raise ValueError("portable fact cursor chain diverges from target")
        signs = {
            slice_relations[slice_key]
            for slice_key in imported_chain[:shared]
            if slice_relations.get(slice_key, 0) != 0
        }
        chain_relation = (len(imported_chain) > len(target_chain)) - (
            len(imported_chain) < len(target_chain)
        )
        if chain_relation:
            signs.add(chain_relation)
        if len(signs) > 1:
            raise ValueError("portable fact histories are incomparable")
        session_relations[session_id] = next(iter(signs), 0)
    return session_relations


def _import_v10_fact_state(
    conn,
    grouped: dict[str, list[dict]],
    inserted: dict[str, int],
    *,
    session_relations: dict[str, int],
) -> None:
    """Restore exact fact history under the history-only authority boundary."""

    target_generations = {
        str(record["slice_key"]): (
            int(row["generation"]) if row is not None else 0
        )
        for record in grouped.get("fact_extraction_outcome", [])
        for row in [conn.execute(
            "SELECT generation FROM fact_extraction_outcomes WHERE slice_key=?",
            (record["slice_key"],),
        ).fetchone()]
    }
    new_slices = {
        slice_key for slice_key, generation in target_generations.items()
        if generation == 0
    }
    upgrade_slices = {
        str(outcome["slice_key"])
        for outcome in grouped.get("fact_extraction_outcome", [])
        if int(outcome["generation"])
        > target_generations[str(outcome["slice_key"])]
    }
    for kind in _FACT_WIRE_KINDS:
        inserted[kind] = 0
    advancing_sessions = {
        session_id for session_id, relation in session_relations.items()
        if relation > 0
    }
    if not new_slices and not upgrade_slices and not advancing_sessions:
        return
    outcomes = {
        str(record["slice_key"]): record
        for record in grouped.get("fact_extraction_outcome", [])
    }
    fact_ids: dict[tuple[str, str], int] = {}
    with core_db.evidence_history_mutation(conn):
        for slice_key in sorted(new_slices):
            record = outcomes[slice_key]
            columns = list(_V10_COLS_BY_KIND["fact_extraction_outcome"])
            staged = dict(record)
            staged.update({
                "source_manifest_version": None,
                "source_manifest_count": 0,
                "source_manifest_hash": None,
                "source_manifest_complete": 0,
            })
            conn.execute(
                f"INSERT INTO fact_extraction_outcomes({', '.join(columns)}) "
                f"VALUES ({', '.join('?' * len(columns))})",
                [staged[column] for column in columns],
            )
            inserted["fact_extraction_outcome"] += 1
        for record in grouped.get("fact_extraction_source_occurrence", []):
            if str(record["slice_key"]) not in new_slices:
                continue
            columns = list(_V10_COLS_BY_KIND[
                "fact_extraction_source_occurrence"
            ])
            conn.execute(
                f"INSERT INTO fact_extraction_source_occurrences({', '.join(columns)}) "
                f"VALUES ({', '.join('?' * len(columns))})",
                [record[column] for column in columns],
            )
            inserted["fact_extraction_source_occurrence"] += 1
        for slice_key in sorted(new_slices):
            record = outcomes[slice_key]
            conn.execute(
                "UPDATE fact_extraction_outcomes SET "
                "source_manifest_version=?,source_manifest_count=?,"
                "source_manifest_hash=?,source_manifest_complete=1 "
                "WHERE slice_key=?",
                (
                    record["source_manifest_version"],
                    record["source_manifest_count"],
                    record["source_manifest_hash"], slice_key,
                ),
            )
        for record in grouped.get("fact_extraction_revision", []):
            slice_key = str(record["slice_key"])
            if (
                slice_key not in new_slices
                and int(record["generation"]) <= target_generations[slice_key]
            ):
                continue
            columns = list(_V10_COLS_BY_KIND["fact_extraction_revision"])
            conn.execute(
                f"INSERT INTO fact_extraction_revisions({', '.join(columns)}) "
                f"VALUES ({', '.join('?' * len(columns))})",
                [record[column] for column in columns],
            )
            inserted["fact_extraction_revision"] += 1
        for record in grouped.get("narrative_fact", []):
            slice_key = str(record["source_outcome_key"])
            if slice_key not in new_slices | upgrade_slices:
                continue
            key = (slice_key, str(record["fact_key"]))
            existing_fact = conn.execute(
                "SELECT id FROM narrative_facts WHERE source_outcome_key=? "
                "AND fact_key=?", key,
            ).fetchone()
            if existing_fact is None:
                columns = list(_V10_COLS_BY_KIND["narrative_fact"])
                cur = conn.execute(
                    f"INSERT INTO narrative_facts({', '.join(columns)}) "
                    f"VALUES ({', '.join('?' * len(columns))})",
                    [record[column] for column in columns],
                )
                fact_ids[key] = int(cur.lastrowid)
                inserted["narrative_fact"] += 1
            else:
                fact_ids[key] = int(existing_fact["id"])
        for record in grouped.get("narrative_fact_lifecycle", []):
            slice_key = str(record["source_outcome_key"])
            if (
                slice_key not in new_slices
                and int(record["generation"]) <= target_generations[slice_key]
            ):
                continue
            fact_id = fact_ids.get((slice_key, str(record["fact_key"])))
            if fact_id is None:
                raise ValueError("portable fact lifecycle has no imported fact")
            conn.execute(
                "INSERT INTO narrative_fact_lifecycle("
                "fact_id,generation,direction,event_at,prompt_version,"
                "result_hash,recorded_at) VALUES (?,?,?,?,?,?,?)",
                (
                    fact_id, record["generation"], record["direction"],
                    record["event_at"], record["prompt_version"],
                    record["result_hash"], record["recorded_at"],
                ),
            )
            inserted["narrative_fact_lifecycle"] += 1
        for record in grouped.get("narrative_fact", []):
            slice_key = str(record["source_outcome_key"])
            if slice_key not in upgrade_slices:
                continue
            conn.execute(
                "UPDATE narrative_facts SET valid_at=?,invalid_at=?,"
                "current_generation=?,lifecycle_status=? WHERE "
                "source_outcome_key=? AND fact_key=?",
                (
                    record["valid_at"], record["invalid_at"],
                    record["current_generation"], record["lifecycle_status"],
                    slice_key, record["fact_key"],
                ),
            )
        for slice_key in sorted(upgrade_slices - new_slices):
            record = outcomes[slice_key]
            conn.execute(
                "UPDATE fact_extraction_outcomes SET prompt_version=?,"
                "generation=?,outcome_status=?,result_hash=?,succeeded_at=? "
                "WHERE slice_key=?",
                (
                    record["prompt_version"], record["generation"],
                    record["outcome_status"], record["result_hash"],
                    record["succeeded_at"], slice_key,
                ),
            )
        sessions = {
            str(record["id"]): record for record in grouped.get("session", [])
        }
        for session_id, relation in session_relations.items():
            if relation <= 0:
                continue
            record = sessions[session_id]
            conn.execute(
                "UPDATE sessions SET facts_message_id=?,"
                "facts_cursor_message_id=?,facts_cursor_partial_message_id=?,"
                "facts_cursor_offset=?,facts_cursor_prompt_version=?,"
                "facts_retry_count=?,facts_retry_config_version=?,"
                "facts_quarantined=? WHERE id=?",
                (
                    record["facts_message_id"],
                    record["facts_cursor_message_id"],
                    record["facts_cursor_partial_message_id"],
                    record["facts_cursor_offset"],
                    record["facts_cursor_prompt_version"],
                    record["facts_retry_count"],
                    record["facts_retry_config_version"],
                    record["facts_quarantined"], session_id,
                ),
            )
    imported_chain_cache: dict[str, bool | None] = {}
    for slice_key in sorted(new_slices | upgrade_slices):
        if load_fact_outcome_source_manifest(
            conn, slice_key, verify_result=True,
            chain_cache=imported_chain_cache,
        ) is None:
            raise ValueError("portable fact history failed runtime validation")


def _redact_portable_records(grouped: dict[str, list[dict]]) -> None:
    """Scrub all portable text before any destination SQL is executed.

    Coverage chunks need a semantic transform rather than a blind regex over
    JSON bytes: decode each canonical source record, redact its content,
    re-encode it canonically, and update the corresponding proof hash. Generic
    v37 chunks may contain surrounding text, which is scrubbed while their
    exact record line remains valid.
    """
    source_time_by_coverage: dict[tuple[int, str, str], object] = {}
    source_time_by_message: dict[tuple[str, int], object] = {}

    # Canonicalization deliberately removes punctuation. That can turn an
    # obvious secret such as ``alice.private@example.com`` into
    # ``alice_private_example_com``, which a regex redactor can no longer
    # recognize. Build a one-way identity map from the still-raw claim
    # surfaces before scrubbing them, then apply it to graph and alias natural
    # keys as one referentially consistent transform.
    normalized_sensitive_identities: dict[str, str] = {}
    for evidence in grouped.get("edge_evidence", []):
        for field in ("surface_subject", "surface_object"):
            surface = evidence.get(field)
            if not isinstance(surface, str):
                continue
            safe_surface = redaction.redact(surface)
            if safe_surface == surface:
                continue
            normalized = canonicalize.normalize(surface)
            if not normalized:
                continue
            fingerprint = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()[:12]
            normalized_sensitive_identities[normalized] = (
                f"{canonicalize.normalize(safe_surface)}_{fingerprint}"
            )

    fact_outcomes_by_key = {
        record.get("slice_key"): record
        for record in grouped.get("fact_extraction_outcome", [])
    }
    partial_fact_artifacts: set[tuple[object, object]] = set()
    for source in grouped.get("fact_extraction_source_occurrence", []):
        outcome = fact_outcomes_by_key.get(source.get("slice_key"))
        if outcome is not None and source.get("source_message_id") in {
            outcome.get("cursor_before_partial_message_id"),
            outcome.get("cursor_after_partial_message_id"),
        }:
            partial_fact_artifacts.add((
                source.get("source_coverage_chunk_id"),
                source.get("source_message_id"),
            ))

    chunks_by_id = {
        row.get("id"): row
        for row in grouped.get("chunk", [])
        if isinstance(row.get("id"), str)
    }
    redacted_sources: dict[
        tuple[str, int], tuple[str, str, str, str]
    ] = {}
    for chunk_id, chunk in chunks_by_id.items():
        text = chunk.get("text")
        if not isinstance(text, str):
            continue
        protected_lines: list[str] = []
        replacements: dict[str, str] = {}
        for line_index, line in enumerate(text.split("\n")):
            canonical = None
            try:
                payload = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                payload = None
            if (
                isinstance(payload, dict)
                and set(payload) == {"content", "id", "record_version", "role"}
                and isinstance(payload.get("id"), int)
                and not isinstance(payload.get("id"), bool)
                and payload.get("record_version") == MESSAGE_RECORD_VERSION
                and payload.get("role") in {"user", "assistant", "system", "tool"}
                and isinstance(payload.get("content"), str)
            ):
                expected = encode_message_record(
                    message_id=payload["id"],
                    role=payload["role"],
                    content=payload["content"],
                )
                if line == expected:
                    safe_content = redaction.redact(payload["content"])
                    if (
                        safe_content != payload["content"]
                        and (chunk_id, payload["id"]) in partial_fact_artifacts
                    ):
                        # Fact offsets are character coordinates in an immutable
                        # source unit.  Mask only the recognised sensitive
                        # spans at their original lengths so those coordinates
                        # stay exact without discarding the surrounding benign
                        # evidence.
                        safe_content = redaction.redact_preserving_length(
                            payload["content"]
                        )
                    canonical = encode_message_record(
                        message_id=payload["id"],
                        role=payload["role"],
                        content=safe_content,
                    )
                    redacted_sources[(chunk_id, payload["id"])] = (
                        payload["role"], payload["content"], safe_content, line
                    )
            elif (
                isinstance(payload, dict)
                and set(payload) == {
                    "content", "id", "record_version", "role", "session_id",
                    "source_created_at", "source_peer_id",
                    "source_workspace_id",
                }
                and isinstance(payload.get("id"), int)
                and not isinstance(payload.get("id"), bool)
                and payload.get("record_version")
                == MESSAGE_PROVENANCE_RECORD_VERSION
                and payload.get("role")
                in {"user", "assistant", "system", "tool"}
                and all(
                    isinstance(payload.get(field), str)
                    for field in (
                        "content", "session_id", "source_peer_id",
                        "source_workspace_id",
                    )
                )
                and isinstance(payload.get("source_created_at"), str)
            ):
                expected = encode_provenance_message_record(
                    message_id=payload["id"],
                    session_id=payload["session_id"],
                    role=payload["role"],
                    content=payload["content"],
                    source_created_at=payload["source_created_at"],
                    source_peer_id=payload["source_peer_id"],
                    source_workspace_id=payload["source_workspace_id"],
                )
                if line == expected:
                    safe_content = redaction.redact(payload["content"])
                    if (
                        safe_content != payload["content"]
                        and (chunk_id, payload["id"]) in partial_fact_artifacts
                    ):
                        safe_content = redaction.redact_preserving_length(
                            payload["content"]
                        )
                    canonical = encode_provenance_message_record(
                        message_id=payload["id"],
                        session_id=payload["session_id"],
                        role=payload["role"],
                        content=safe_content,
                        source_created_at=payload["source_created_at"],
                        source_peer_id=payload["source_peer_id"],
                        source_workspace_id=payload["source_workspace_id"],
                    )
                    redacted_sources[(chunk_id, payload["id"])] = (
                        payload["role"], payload["content"], safe_content, line
                    )
            if canonical is None:
                protected_lines.append(line)
            else:
                token = (
                    f"__HYMEM_PROTECTED_RECORD_{line_index}_"
                    f"{hashlib.sha256(line.encode('utf-8')).hexdigest()}__"
                )
                protected_lines.append(token)
                replacements[token] = canonical
        # Scrub the contiguous chunk so DOTALL patterns (notably multiline PEM
        # private keys) cannot survive between physical lines. Canonical JSONL
        # records are protected and reinserted after their decoded content has
        # been independently scrubbed and re-encoded above.
        safe_text = redaction.redact("\n".join(protected_lines))
        if replacements:
            protected_pattern = re.compile(
                r"__HYMEM_PROTECTED_RECORD_[0-9]+_[0-9a-f]{64}__"
            )
            safe_text = protected_pattern.sub(
                lambda match: replacements.get(match.group(0), match.group(0)),
                safe_text,
            )
        if redaction.redact(safe_text) != safe_text:
            raise ValueError("portable chunk redaction did not reach a fixed point")
        chunk["text"] = safe_text

    rewritten_source_records: dict[
        tuple[str, int], tuple[object, str]
    ] = {}
    for proof in grouped.get("message_retention_coverage", []):
        source_mid = proof.get("message_id")
        chunk_id = proof.get("chunk_id")
        source = redacted_sources.get((chunk_id, source_mid))
        if source is None:
            raise ValueError(
                "portable coverage proof has no canonical source record to redact"
            )
        role, original_content, content, original_record = source
        try:
            (
                expected_record,
                expected_hash,
                expected_hash_version,
                expected_record_version,
            ) = canonical_message_record(
                message_id=source_mid,
                session_id=proof.get("source_session_id"),
                role=role,
                content=original_content,
                source_created_at=proof.get("source_created_at"),
                source_peer_id=proof.get("source_peer_id"),
                source_workspace_id=proof.get("source_workspace_id"),
            )
        except (TypeError, ValueError):
            raise ValueError("portable coverage proof metadata is invalid") from None
        if (
            proof.get("source_role") != role
            or original_record != expected_record
            or proof.get("hash_version") != expected_hash_version
            or proof.get("record_version") != expected_record_version
            or proof.get("message_content_hash") != expected_hash
        ):
            raise ValueError("portable coverage proof metadata is invalid")
        raw_source_time = proof.get("source_created_at")
        if raw_source_time is None:
            safe_source_time = None
        else:
            try:
                safe_source_time = normalize_iso_timestamp(
                    raw_source_time, context="redacted source event"
                )
            except ValueError:
                safe_source_time = "0001-01-01T00:00:00.000Z"
        safe_record, safe_hash, _, _ = canonical_message_record(
            message_id=source_mid,
            session_id=proof.get("source_session_id"),
            role=role,
            content=content,
            source_created_at=safe_source_time,
            source_peer_id=proof.get("source_peer_id"),
            source_workspace_id=proof.get("source_workspace_id"),
        )
        current_record, _, _, _ = canonical_message_record(
            message_id=source_mid,
            session_id=proof.get("source_session_id"),
            role=role,
            content=content,
            source_created_at=raw_source_time,
            source_peer_id=proof.get("source_peer_id"),
            source_workspace_id=proof.get("source_workspace_id"),
        )
        artifact_key = (str(chunk_id), int(source_mid))
        previous_rewrite = rewritten_source_records.get(artifact_key)
        if previous_rewrite is not None and previous_rewrite != (
            safe_source_time, safe_hash
        ):
            raise ValueError("portable source versions disagree during redaction")
        if previous_rewrite is None and safe_record != current_record:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None or chunk["text"].count(current_record) != 1:
                raise ValueError("portable source clock redaction lost its record")
            chunk["text"] = chunk["text"].replace(current_record, safe_record, 1)
        rewritten_source_records[artifact_key] = (safe_source_time, safe_hash)
        proof["source_created_at"] = safe_source_time
        proof["message_content_hash"] = safe_hash
    # Generic coverage versions are caller-provided natural-key text.  Keep
    # the reserved lossless vocabulary byte-identical, but pseudonymize any
    # secret-bearing legacy version and rewrite every dependent foreign key as
    # one identity transform.  The version is not part of the canonical
    # message hash, so no source bytes are inferred or rewritten here.
    coverage_version_map: dict[tuple[int, str, str], str] = {}
    new_coverage_keys: set[tuple[int, str, str]] = set()
    for proof in grouped.get("message_retention_coverage", []):
        message_id = int(proof["message_id"])
        chunk_id = str(proof["chunk_id"])
        old_version = str(proof["coverage_version"])
        safe_version = redaction.redact(old_version)
        if safe_version != old_version:
            suffix = "#" + hashlib.sha256(
                old_version.encode("utf-8")
            ).hexdigest()[:12]
            safe_version = safe_version[: max(0, 128 - len(suffix))] + suffix
        if not safe_version or redaction.redact(safe_version) != safe_version:
            raise ValueError("portable coverage version redaction is invalid")
        old_key = (message_id, chunk_id, old_version)
        new_key = (message_id, chunk_id, safe_version)
        if new_key in new_coverage_keys:
            raise ValueError("portable coverage redaction collapsed identities")
        new_coverage_keys.add(new_key)
        coverage_version_map[old_key] = safe_version
        proof["coverage_version"] = safe_version
        safe_source_time = proof.get("source_created_at")
        source_time_by_coverage[new_key] = safe_source_time
        source_time_by_message[(str(proof["source_session_id"]), message_id)] = (
            safe_source_time
        )

    for kind in (
        "chunk_message_source", "episode_source_occurrence",
        "fact_extraction_source_occurrence", "edge_evidence",
    ):
        for record in grouped.get(kind, []):
            message_id = record.get("source_message_id")
            chunk_id = record.get("source_coverage_chunk_id")
            version = record.get("source_coverage_version")
            if message_id is None or chunk_id is None or version is None:
                continue
            old_key = (int(message_id), str(chunk_id), str(version))
            safe_version = coverage_version_map.get(old_key)
            if safe_version is None:
                raise ValueError(
                    "portable source reference has no coverage version to redact"
                )
            record["source_coverage_version"] = safe_version

    for record in grouped.get("user_profile_fact", []):
        key = (str(record.get("source_session_id")), int(record["source_message_id"]))
        if key in source_time_by_message:
            record["source_created_at"] = source_time_by_message[key]
    for kind in (
        "episode_source_occurrence", "fact_extraction_source_occurrence",
        "edge_evidence",
    ):
        for record in grouped.get(kind, []):
            message_id = record.get("source_message_id")
            chunk_id = record.get("source_coverage_chunk_id")
            version = record.get("source_coverage_version")
            if message_id is None or chunk_id is None or version is None:
                continue
            key = (int(message_id), str(chunk_id), str(version))
            if key in source_time_by_coverage and "source_created_at" in record:
                record["source_created_at"] = source_time_by_coverage[key]

    # Episode manifests bind the coverage content hash. Redaction rewrites the
    # canonical lossless artifact above, so carry that exact rewrite into every
    # occurrence and then recompute the ordered episode header. Ownership and
    # membership are unchanged; no range-derived sources are introduced.
    redacted_proofs = {
        (
            proof["message_id"], proof["chunk_id"], proof["coverage_version"]
        ): proof
        for proof in grouped.get("message_retention_coverage", [])
    }
    episode_sources: dict[str, list[tuple[int, BoundSourceOccurrence]]] = defaultdict(list)
    for source in grouped.get("episode_source_occurrence", []):
        proof = redacted_proofs.get((
            source["source_message_id"],
            source["source_coverage_chunk_id"],
            source["source_coverage_version"],
        ))
        if proof is None:
            raise ValueError("portable episode source has no redacted coverage proof")
        source["source_content_hash"] = proof["message_content_hash"]
        episode_sources[str(source["episode_id"])].append((
            int(source["ordinal"]),
            BoundSourceOccurrence(
                message_id=int(source["source_message_id"]),
                session_id=str(source["source_session_id"]),
                role=str(source["source_role"]),
                source_peer_id=source["source_peer_id"],
                source_workspace_id=source["source_workspace_id"],
                source_created_at=source["source_created_at"],
                coverage_chunk_id=str(source["source_coverage_chunk_id"]),
                coverage_version=str(source["source_coverage_version"]),
                content_hash=str(source["source_content_hash"]),
            ),
        ))
    for episode in grouped.get("episode", []):
        if episode.get("source_manifest_complete") != 1:
            continue
        sources = tuple(
            occurrence for _ordinal, occurrence in sorted(
                episode_sources.get(str(episode["id"]), []),
                key=lambda item: item[0],
            )
        )
        episode["source_manifest_hash"] = source_manifest_hash(
            EPISODE_SOURCE_MANIFEST_VERSION, sources
        )

    # Fact outcome identity is source-coordinate based, so redaction preserves
    # slice keys while rebinding every content hash, rendered-input hash, fact
    # payload key, revision hash, and lifecycle reference as one transform.
    fact_sources_by_slice: dict[str, list[dict]] = defaultdict(list)
    sensitive_fact_identities_by_slice: dict[str, set[str]] = defaultdict(set)
    for source in grouped.get("fact_extraction_source_occurrence", []):
        proof = redacted_proofs.get((
            source["source_message_id"], source["source_coverage_chunk_id"],
            source["source_coverage_version"],
        ))
        if proof is None:
            raise ValueError("portable fact source has no redacted coverage proof")
        source["source_content_hash"] = proof["message_content_hash"]
        source_slice_key = str(source["slice_key"])
        fact_sources_by_slice[source_slice_key].append(source)
        redacted_source = redacted_sources.get((
            source["source_coverage_chunk_id"], source["source_message_id"]
        ))
        if redacted_source is not None and redacted_source[1] != redacted_source[2]:
            for fragment in redaction.sensitive_fragments(redacted_source[1]):
                normalized_fragment = canonicalize.normalize(fragment)
                if normalized_fragment:
                    sensitive_fact_identities_by_slice[source_slice_key].add(
                        normalized_fragment
                    )

    redacted_fact_stream = _portable_lossless_fact_sources(grouped)
    for outcome in grouped.get("fact_extraction_outcome", []):
        slice_key = str(outcome["slice_key"])
        ordered = sorted(
            fact_sources_by_slice.get(slice_key, []),
            key=lambda source: int(source["ordinal"]),
        )
        occurrences: list[BoundSourceOccurrence] = []
        rendered: list[str] = []
        for source in ordered:
            pair = redacted_fact_stream.get((
                str(source["source_session_id"]), int(source["source_message_id"])
            ))
            if pair is None:
                raise ValueError("portable fact source redaction lost its bytes")
            occurrence, content = pair
            occurrences.append(occurrence)
            start = (
                int(outcome["cursor_before_offset"])
                if source["source_message_id"]
                == outcome["cursor_before_partial_message_id"] else 0
            )
            end = (
                int(outcome["cursor_after_offset"])
                if source["source_message_id"]
                == outcome["cursor_after_partial_message_id"] else len(content)
            )
            empty_whole = not content and start == 0 and end == 0
            if start < 0 or end > len(content) or (end <= start and not empty_whole):
                raise ValueError("portable redaction invalidated a fact offset")
            rendered.append(f"{occurrence.role}: {content[start:end]}")
        canonical_sources = tuple(occurrences)
        outcome["source_manifest_hash"] = source_manifest_hash(
            FACT_SOURCE_MANIFEST_VERSION, canonical_sources
        )
        outcome["input_hash"] = fact_input_hash("\n".join(rendered))

    fact_key_map: dict[tuple[str, str], str] = {}
    fact_items: dict[tuple[str, str], dict] = {}
    used_fact_keys_by_slice: dict[str, set[str]] = defaultdict(set)
    for fact in grouped.get("narrative_fact", []):
        old_key = str(fact["fact_key"])
        old_text = str(fact["text"])
        safe_text = redaction.redact(old_text)
        text_was_redacted = safe_text != old_text
        if text_was_redacted:
            suffix = "#" + hashlib.sha256(
                old_text.encode("utf-8")
            ).hexdigest()[:12]
            safe_text = safe_text[: max(0, 600 - len(suffix))] + suffix
        if redaction.redact(safe_text) != safe_text:
            raise ValueError("portable fact text redaction did not reach a fixed point")
        entities = loads_strict_json(fact["entities"])
        safe_entities: list[str] = []
        sensitive_identities = set(sensitive_fact_identities_by_slice.get(
            str(fact["source_outcome_key"]), ()
        ))
        for fragment in redaction.sensitive_fragments(old_text):
            normalized_fragment = canonicalize.normalize(fragment)
            if normalized_fragment:
                sensitive_identities.add(normalized_fragment)
        for entity in entities:
            entity_identity = canonicalize.normalize(entity)
            if entity_identity in sensitive_identities:
                safe_entity = "redacted_entity_" + hashlib.sha256(
                    entity.encode("utf-8")
                ).hexdigest()[:12]
            else:
                safe_entity = redaction.redact(entity)
                if safe_entity != entity:
                    safe_entity += "#" + hashlib.sha256(
                        entity.encode("utf-8")
                    ).hexdigest()[:12]
            if redaction.redact(safe_entity) != safe_entity:
                raise ValueError(
                    "portable fact entity redaction did not reach a fixed point"
                )
            normalized = canonicalize.normalize(safe_entity)
            if not normalized:
                raise ValueError("portable fact entity redaction became empty")
            if normalized not in safe_entities:
                safe_entities.append(normalized)
        item = {
            "text": safe_text,
            "date": fact.get("fact_date"),
            "entities": safe_entities,
        }
        new_key = fact_item_key(item)
        map_key = (str(fact["source_outcome_key"]), old_key)
        if new_key in used_fact_keys_by_slice[map_key[0]]:
            raise ValueError("portable fact redaction collapsed distinct facts")
        used_fact_keys_by_slice[map_key[0]].add(new_key)
        fact_key_map[map_key] = new_key
        fact_items[(map_key[0], new_key)] = item
        fact["text"] = safe_text
        fact["entities"] = json.dumps(safe_entities)
        fact["fact_key"] = new_key
    for event in grouped.get("narrative_fact_lifecycle", []):
        map_key = (str(event["source_outcome_key"]), str(event["fact_key"]))
        if map_key not in fact_key_map:
            raise ValueError("portable fact lifecycle lost its redacted payload")
        event["fact_key"] = fact_key_map[map_key]

    revisions_by_slice: dict[str, list[dict]] = defaultdict(list)
    for revision in grouped.get("fact_extraction_revision", []):
        revisions_by_slice[str(revision["slice_key"])].append(revision)
    events_by_slice_generation: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for event in grouped.get("narrative_fact_lifecycle", []):
        events_by_slice_generation[(
            str(event["source_outcome_key"]), int(event["generation"])
        )].append(event)
    outcome_by_slice = {
        str(outcome["slice_key"]): outcome
        for outcome in grouped.get("fact_extraction_outcome", [])
    }
    for slice_key, revisions in revisions_by_slice.items():
        active: set[str] = set()
        for revision in sorted(revisions, key=lambda row: int(row["generation"])):
            generation = int(revision["generation"])
            generation_events = events_by_slice_generation.get(
                (slice_key, generation), []
            )
            for event in generation_events:
                key = str(event["fact_key"])
                if int(event["direction"]) == 1:
                    active.add(key)
                else:
                    active.discard(key)
            result_hash = fact_result_hash(
                fact_items[(slice_key, key)] for key in active
            )
            revision["result_hash"] = result_hash
            revision["outcome_status"] = "success" if active else "empty"
            for event in generation_events:
                event["result_hash"] = result_hash
        outcome = outcome_by_slice.get(slice_key)
        if outcome is None or not revisions:
            raise ValueError("portable fact redaction lost an outcome revision")
        latest = max(revisions, key=lambda row: int(row["generation"]))
        outcome["result_hash"] = latest["result_hash"]
        outcome["outcome_status"] = latest["outcome_status"]

    def redact_json_node(value: object) -> object:
        if isinstance(value, str):
            return redaction.redact(value)
        if isinstance(value, list):
            return [redact_json_node(item) for item in value]
        if isinstance(value, dict):
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("portable JSON redaction found a non-text key")
                safe_key = redaction.redact(key)
                if safe_key in result:
                    raise ValueError("portable JSON redaction collapsed distinct keys")
                result[safe_key] = redact_json_node(item)
            return result
        return value

    def redact_json_text(value: object, *, context: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"portable {context} must be JSON text")
        try:
            decoded = loads_strict_json(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError(f"portable {context} is malformed JSON") from None
        safe = redact_json_node(decoded)
        if redact_json_node(safe) != safe:
            raise ValueError(f"portable {context} redaction is not idempotent")
        return json.dumps(
            safe, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def redact_maybe_json_text(value: object) -> object:
        if not isinstance(value, str):
            return value
        try:
            decoded = loads_strict_json(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return redaction.redact(value)
        safe = redact_json_node(decoded)
        return json.dumps(
            safe, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    json_text_fields = {
        "peer": ("metadata",),
        "session_peer": ("configuration",),
        "episode": ("participants", "key_entities"),
        "procedure": ("steps", "triggers", "entities_involved"),
    }
    for kind, fields in json_text_fields.items():
        for record in grouped.get(kind, []):
            for field in fields:
                record[field] = redact_json_text(
                    record.get(field), context=f"{kind}.{field}"
                )

    # These are replay/control-plane markers, not evidence. Preserve a
    # producer-recognized, secret-free marker exactly; rewind only malformed
    # or sensitive legacy spellings so a privacy-preserving restore does not
    # destroy a coherent cursor/quarantine state merely because redaction was
    # requested.
    digest_state_safe_by_session: dict[str, bool] = {}
    for record in grouped.get("session", []):
        digest_generations = (
            record.get("digest_cursor_prompt_version"),
            record.get("digest_published_generation"),
        )
        digest_prompt = record.get("digested_prompt_version")
        episode_prompt = record.get("episodes_prompt_version")
        digest_safe = bool(
            all(
                value is None
                or (
                    digest_generation_is_recognized(value)
                    and redaction.redact(value) == value
                )
                for value in digest_generations
            )
            and (
                digest_prompt is None
                or (
                    isinstance(digest_prompt, str)
                    and re.fullmatch(r"v[1-9][0-9]{0,5}", digest_prompt)
                    and redaction.redact(digest_prompt) == digest_prompt
                )
            )
            and (
                episode_prompt is None
                or (
                    isinstance(episode_prompt, str)
                    and re.fullmatch(
                        r"episodes\.granular\.v[1-9][0-9]{0,5}",
                        episode_prompt,
                    )
                    and redaction.redact(episode_prompt) == episode_prompt
                )
            )
        )
        digest_state_safe_by_session[str(record["id"])] = digest_safe
        if not digest_safe:
            for field, value in {
                "digested_message_id": None,
                "digested_prompt_version": None,
                "digest_cursor_message_id": None,
                "digest_cursor_partial_message_id": None,
                "digest_cursor_offset": 0,
                "digest_cursor_prompt_version": None,
                "digest_published_generation": None,
                "episodes_prompt_version": None,
            }.items():
                if field in record:
                    record[field] = value

        profile_generations = (
            record.get("profile_cursor_prompt_version"),
            record.get("profile_published_generation"),
        )
        profile_prompt = record.get("profile_prompt_version")
        profile_safe = bool(
            all(
                value is None
                or (
                    profile_generation_is_recognized(value)
                    and redaction.redact(value) == value
                )
                for value in profile_generations
            )
            and (
                profile_prompt is None
                or (
                    isinstance(profile_prompt, str)
                    and re.fullmatch(r"profile\.v[1-9][0-9]{0,5}", profile_prompt)
                    and redaction.redact(profile_prompt) == profile_prompt
                )
            )
        )
        if not profile_safe:
            for field, value in {
                "profile_prompt_version": None,
                "profile_cursor_message_id": None,
                "profile_cursor_partial_message_id": None,
                "profile_cursor_offset": 0,
                "profile_cursor_prompt_version": None,
                "profile_published_generation": None,
            }.items():
                if field in record:
                    record[field] = value
        for prefix in ("digest", "profile"):
            config_field = f"{prefix}_retry_config_version"
            retry_value = record.get(config_field)
            if isinstance(retry_value, str) and redaction.redact(retry_value) != retry_value:
                record[f"{prefix}_retry_count"] = 0
                record[config_field] = None
                record[f"{prefix}_quarantined"] = 0
    digest_slice_re = re.compile(
        r"after=(?:start|[1-9][0-9]{0,18});"
        r"partial=(?:none|[1-9][0-9]{0,18});"
        r"offset=(?:0|[1-9][0-9]{0,18});cap=[1-9][0-9]{0,8}"
    )
    for record in grouped.get("episode", []):
        generation = record.get("digest_generation")
        slice_key = record.get("digest_slice_key")
        safe = bool(
            digest_state_safe_by_session.get(str(record.get("session_id")), False)
            and (
                (generation is None and slice_key is None)
                or (
                    digest_generation_is_recognized(generation)
                    and isinstance(slice_key, str)
                    and digest_slice_re.fullmatch(slice_key)
                    and redaction.redact(generation) == generation
                    and redaction.redact(slice_key) == slice_key
                )
            )
        )
        if not safe:
            record["digest_slice_key"] = None
            record["digest_generation"] = None

    text_fields = {
        "session": ("summary", "auto_summary"),
        "chunk": ("salience_reason",),
        "episode": ("title", "summary"),
        "procedure": ("name", "description"),
    }
    for kind, fields in text_fields.items():
        for record in grouped.get(kind, []):
            for field in fields:
                if isinstance(record.get(field), str):
                    record[field] = redaction.redact(record[field])
    def redact_canonical_identity(value: object) -> object:
        if not isinstance(value, str):
            return value
        mapped = normalized_sensitive_identities.get(
            canonicalize.normalize(value)
        )
        if mapped is not None:
            return mapped
        safe = redaction.redact(value)
        if safe != value:
            fingerprint = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
            safe = f"{canonicalize.normalize(safe)}_{fingerprint}"
        return safe

    # These fields participate in natural keys. Stable source-derived suffixes
    # prevent two distinct identities from collapsing to one generic marker
    # during a privacy-preserving import.
    for record in grouped.get("edge", []):
        for field in ("subject_canonical", "object_canonical"):
            record[field] = redact_canonical_identity(record.get(field))
    for record in grouped.get("entity_alias", []):
        record["alias"] = redact_canonical_identity(record.get("alias"))
        record["canonical"] = redact_canonical_identity(record.get("canonical"))
    for record in grouped.get("profile_entry", []):
        value = record.get("text")
        if not isinstance(value, str):
            continue
        safe = redaction.redact(value)
        if safe != value:
            safe += "#" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        record["text"] = safe
    for record in grouped.get("user_profile_fact", []):
        if isinstance(record.get("value"), str):
            record["value"] = redaction.redact(record["value"])
        if record.get("slot") == "relationship":
            record["slot_key"] = _redact_profile_key(record.get("slot_key"))

    def redact_identity_text(value: object) -> object:
        if not isinstance(value, str):
            return value
        safe = redaction.redact(value)
        if safe != value:
            safe += "#" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return safe

    # Claim semantic fields feed interpretation/natural identities.  Stable
    # fingerprints keep two redacted secrets distinct while revealing neither.
    interpretation_by_wire_id: dict[int, str] = {}
    evidence_kind_by_wire_id: dict[int, str] = {}
    for record in grouped.get("edge_evidence", []):
        for field in (
            "surface_subject", "surface_object", "value_text", "value_unit",
            "temporal_scope", "weight_source", "extraction_prompt_version",
        ):
            record[field] = redact_identity_text(record.get(field))
        record["evidence_kind"] = redact_identity_text(record["evidence_kind"])
        record["superseded_reason"] = redact_maybe_json_text(
            record.get("superseded_reason")
        )
        record["interpretation_key"] = evidence_ledger._interpretation_key(
            polarity=int(record["polarity"]),
            evidence_weight=int(record["evidence_weight"]),
            weight_source=record["weight_source"],
            source_role=record.get("source_role"),
            surface_subject=record.get("surface_subject"),
            surface_object=record.get("surface_object"),
            value_text=record.get("value_text"),
            value_numeric=record.get("value_numeric"),
            value_unit=record.get("value_unit"),
            temporal_scope=record.get("temporal_scope"),
        )
        interpretation_by_wire_id[int(record["id"])] = record["interpretation_key"]
        evidence_kind_by_wire_id[int(record["id"])] = record["evidence_kind"]
    for record in grouped.get("claim_observation", []):
        record["prompt_version"] = redact_identity_text(record["prompt_version"])
        record["prompt_generation"] = evidence_ledger.prompt_generation(
            record["prompt_version"]
        )
        evidence_id = int(record["evidence_id"])
        record["interpretation_key"] = interpretation_by_wire_id[evidence_id]
        record["evidence_kind"] = evidence_kind_by_wire_id[evidence_id]
    observations_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for record in grouped.get("claim_observation", []):
        observations_by_chunk[str(record["chunk_id"])].append(record)
    edge_by_wire = {
        int(record["id"]): record for record in grouped.get("edge", [])
    }
    for record in grouped.get("claim_extraction_outcome", []):
        record["prompt_version"] = redact_identity_text(record["prompt_version"])
        record["prompt_generation"] = evidence_ledger.prompt_generation(
            record["prompt_version"]
        )
        record["result_hash"] = _wire_claim_result_hash(
            observations_by_chunk.get(str(record["chunk_id"]), []), edge_by_wire
        )
    manual_signal_keys: dict[tuple[int, str], str] = {}
    for record in grouped.get("edge_evidence_signal", []):
        old_signal_key = record["signal_key"]
        record["signal_key"] = redact_identity_text(old_signal_key)
        record["signal_kind"] = redact_identity_text(record["signal_kind"])
        if record["signal_kind"] == "manual_retraction":
            manual_signal_keys[(int(record["edge_id"]), old_signal_key)] = record[
                "signal_key"
            ]
        if isinstance(record.get("details"), str):
            record["details"] = redact_maybe_json_text(record["details"])
    for record in grouped.get("edge_lifecycle", []):
        if isinstance(record.get("details"), str):
            record["details"] = redact_maybe_json_text(record["details"])
    _rewrite_v7_lifecycle_keys(grouped, manual_signal_keys)


def _preflight_v6_export(conn) -> None:
    """Prove that the v6 snapshot can be restored before publishing it.

    A manifest proves byte completeness, not semantic validity.  In
    particular, an old profile row whose raw source and durable provenance are
    both gone would otherwise produce a perfectly checksummed backup that our
    own importer must reject.  Validate every exported coverage artifact, then
    require every typed profile assertion to cite one producer-bounded USER
    artifact using the same rules as import.
    """
    coverage_rows = conn.execute(
        "SELECT message_id, chunk_id, coverage_version "
        "FROM message_retention_coverage ORDER BY message_id, chunk_id"
    ).fetchall()
    for row in coverage_rows:
        message_id = row["message_id"]
        if (
            isinstance(message_id, bool)
            or not isinstance(message_id, int)
            or message_id <= 0
            or message_id > _MAX_SQLITE_ROWID - _ROWID_RESERVE_HEADROOM
        ):
            raise ValueError(
                "cannot export coverage message_id without portable rowid headroom"
            )
        try:
            validate_message_coverage_artifact(
                conn,
                message_id=message_id,
                chunk_id=row["chunk_id"],
                coverage_version=row["coverage_version"],
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cannot export corrupt coverage artifact for message {message_id}"
            ) from exc

    session_retry_rows = conn.execute(
        "SELECT id, digest_retry_count, digest_retry_config_version, "
        "digest_quarantined, profile_retry_count, "
        "profile_retry_config_version, profile_quarantined FROM sessions"
    ).fetchall()
    for row in session_retry_rows:
        if not digest_retry_state_is_valid(
            row["digest_retry_count"], row["digest_retry_config_version"],
            row["digest_quarantined"],
        ):
            raise ValueError(
                f"cannot export invalid digest retry state for session {row['id']}"
            )
        if not profile_retry_state_is_valid(
            row["profile_retry_count"], row["profile_retry_config_version"],
            row["profile_quarantined"],
        ):
            raise ValueError(
                f"cannot export invalid profile retry state for session {row['id']}"
            )

    profile_rows = conn.execute(
        "SELECT slot, slot_key, value, confidence, source_message_id, "
        "source_session_id, source_created_at FROM user_profile ORDER BY id"
    ).fetchall()
    for row in profile_rows:
        source_mid = row["source_message_id"]
        source_session = row["source_session_id"]
        if (
            isinstance(source_mid, bool)
            or not isinstance(source_mid, int)
            or source_mid <= 0
            or not isinstance(source_session, str)
            or not source_session.strip()
        ):
            raise ValueError(
                "cannot export profile fact without durable source provenance"
            )
        covered = covered_messages_after(
            conn,
            source_session,
            source_mid - 1,
            limit=1,
            roles=frozenset({"user"}),
            through_message_id=source_mid,
        )
        if (
            not covered
            or covered[0].message_id != source_mid
            or row["source_created_at"] != covered[0].source_created_at
        ):
            raise ValueError(
                "cannot export profile fact without a validated source artifact"
            )
        candidate = {
            "slot": row["slot"],
            "value": row["value"],
            "evidence_message_id": source_mid,
            "confidence": row["confidence"],
        }
        if row["slot"] == "relationship":
            candidate["slot_key"] = row["slot_key"]
        if len(validate_profile_items([candidate], {source_mid}, max_items=1)) != 1:
            raise ValueError("cannot export an invalid typed profile fact")


def _preflight_v7_export(conn) -> dict[str, list[dict]]:
    """Return a self-validated current snapshot or fail before publishing."""
    _preflight_v6_export(conn)
    mismatches = evidence_ledger.count_mismatches(conn)
    if mismatches:
        raise ValueError("cannot export knowledge graph with stale evidence counters")
    grouped = _collect_v10_records(conn)
    _validate_v6_record_scalars(grouped)
    _validate_v7_records(grouped)
    _validate_v9_records(grouped)
    _validate_v10_fact_records(grouped)
    for episode in grouped.get("episode", []):
        if (
            episode["source_manifest_complete"] == 1
            and load_episode_source_manifest(conn, episode["id"]) is None
        ):
            raise ValueError(
                "cannot export episode without a validated source manifest"
            )
    fact_chain_cache: dict[str, bool | None] = {}
    for outcome in grouped.get("fact_extraction_outcome", []):
        if load_fact_outcome_source_manifest(
            conn, outcome["slice_key"], verify_result=True,
            chain_cache=fact_chain_cache,
        ) is None:
            raise ValueError(
                "cannot export fact outcome without complete authority"
            )
    return grouped


def _edge_natural(record: dict) -> tuple[str, str, str]:
    return (
        record["subject_canonical"], record["predicate"],
        record["object_canonical"],
    )


def _evidence_wire_identity(record: dict) -> tuple[object, ...]:
    if record["provenance_status"] == "canonical":
        return (
            record["source_session_id"], record["source_message_id"],
            record["evidence_kind"], record["revision"],
        )
    return (record["chunk_id"], record["evidence_kind"], record["revision"])


def _evidence_wire_event(record: dict) -> str:
    if record["provenance_status"] == "canonical":
        return record["source_event_at"]
    return _normalized_wire_event(record.get("extracted_at"))


def _wire_evidence_natural_identity(
    record: dict, edge_by_wire: dict[int, dict]
) -> tuple[object, ...]:
    edge = edge_by_wire[int(record["edge_id"])]
    natural_edge = _edge_natural(edge)
    if record["provenance_status"] == "canonical":
        return (
            *natural_edge, "canonical", record["source_session_id"],
            int(record["source_message_id"]), record["evidence_kind"],
            int(record["revision"]), record["interpretation_key"],
        )
    return (
        *natural_edge, "legacy", record["chunk_id"], record["evidence_kind"],
        int(record["revision"]), record["interpretation_key"],
    )


def _wire_evidence_natural_key(
    record: dict, edge_by_wire: dict[int, dict]
) -> str:
    encoded = json.dumps(
        _wire_evidence_natural_identity(record, edge_by_wire),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _wire_evidence_cause_key(record: dict) -> str:
    if record["provenance_status"] == "canonical":
        return (
            f"canonical:{record['source_session_id']}:"
            f"{int(record['source_message_id'])}:{record['evidence_kind']}:"
            f"r{int(record['revision'])}"
        )
    return (
        f"legacy:{record['chunk_id']}:{record['evidence_kind']}:"
        f"r{int(record['revision'])}"
    )


def _wire_phase3_event_key(causes: list[dict]) -> str:
    cause_keys = sorted({_wire_evidence_cause_key(cause) for cause in causes})
    encoded = json.dumps(cause_keys, separators=(",", ":")).encode("utf-8")
    return "phase3-retraction:" + hashlib.sha256(encoded).hexdigest()


def _wire_value_event_key(
    lifecycle: dict,
    cause: dict,
    edge_by_wire: dict[int, dict],
) -> str:
    owner = edge_by_wire[int(lifecycle["edge_id"])]
    winner = edge_by_wire[int(cause["edge_id"])]
    return (
        f"value-supersession:{owner['subject_canonical']}:"
        f"{owner['predicate']}:{winner['object_canonical']}:"
        f"{lifecycle['event_at']}:"
        f"{_wire_evidence_natural_key(cause, edge_by_wire)}"
    )


def _rewrite_v7_lifecycle_keys(
    grouped: dict[str, list[dict]],
    manual_signal_keys: dict[tuple[int, str], str] | None = None,
) -> None:
    """Rebind derived event identities after an intentional wire transform.

    This is used only after redaction has changed natural evidence identities.
    Ordinary export validation never heals a corrupt source ledger.
    """
    edge_by_wire = {
        int(record["id"]): record for record in grouped.get("edge", [])
    }
    evidence_by_wire = {
        int(record["id"]): record for record in grouped.get("edge_evidence", [])
    }
    deps: dict[int, list[dict]] = defaultdict(list)
    for dependency in grouped.get("lifecycle_dependency", []):
        evidence = evidence_by_wire.get(int(dependency["evidence_id"]))
        if evidence is not None:
            deps[int(dependency["lifecycle_id"])].append(evidence)
    manual_signal_keys = manual_signal_keys or {}
    for lifecycle in grouped.get("edge_lifecycle", []):
        event_kind = lifecycle["event_kind"]
        if event_kind == "claim_assertion":
            source = evidence_by_wire[int(lifecycle["source_evidence_id"])]
            lifecycle["event_key"] = evidence_ledger.claim_assertion_event_key(
                source["source_session_id"], source["source_message_id"],
                source["evidence_kind"], source["revision"],
            )
        elif event_kind == "phase3_retraction":
            lifecycle["event_key"] = _wire_phase3_event_key(
                deps.get(int(lifecycle["id"]), [])
            )
        elif event_kind == "value_supersession":
            causes = deps.get(int(lifecycle["id"]), [])
            if len(causes) == 1:
                lifecycle["event_key"] = _wire_value_event_key(
                    lifecycle, causes[0], edge_by_wire
                )
        elif event_kind == "manual_retraction":
            # The old key identifies its signal before redaction. Resolve it
            # first, then bind the event to the signal's transformed key.
            old_prefix = "manual-retraction:"
            old_signal_key = (
                lifecycle["event_key"][len(old_prefix):]
                if lifecycle["event_key"].startswith(old_prefix) else None
            )
            transformed = manual_signal_keys.get(
                (int(lifecycle["edge_id"]), old_signal_key)
            )
            if transformed is not None:
                lifecycle["event_key"] = evidence_ledger.manual_retraction_event_key(
                    transformed
                )


def _mapped_edge_ids(conn, grouped: dict[str, list[dict]]) -> dict[int, int]:
    result: dict[int, int] = {}
    for record in grouped.get("edge", []):
        row = conn.execute(
            "SELECT id FROM knowledge_graph WHERE subject_canonical=? "
            "AND predicate=? AND object_canonical=?",
            _edge_natural(record),
        ).fetchone()
        if row is None:
            raise ValueError("portable edge mapping disappeared during import")
        result[int(record["id"])] = int(row["id"])
    return result


def _rows_match(row, record: dict, columns: list[str]) -> bool:
    return all(row[column] == record[column] for column in columns)


def _merged_timestamp(left: object, right: object, *, latest: bool) -> object:
    """Choose a timestamp commutatively by instant, then exact spelling."""
    values = [value for value in (left, right) if isinstance(value, str)]
    if not values:
        return None
    chooser = max if latest else min
    return chooser(values, key=lambda value: (_normalized_wire_event(value), value))


def _canonical_evidence_base(
    edge_id: int, record: dict | sqlite3.Row
) -> tuple[object, ...]:
    return (
        int(edge_id), record["source_session_id"],
        int(record["source_message_id"]), record["evidence_kind"],
    )


def _observation_identity(
    edge_id: int, record: dict | sqlite3.Row
) -> tuple[object, ...]:
    return (
        record["chunk_id"], int(edge_id), record["source_session_id"],
        int(record["source_message_id"]), record["evidence_kind"],
    )


def _observation_semantic(record: dict | sqlite3.Row) -> tuple[int, str]:
    return (int(record["polarity"]), str(record["interpretation_key"]))


def _import_v7_claim_state(
    conn,
    grouped: dict[str, list[dict]],
    inserted: dict[str, int],
) -> None:
    """Restore v7 graph history through natural-key ID maps."""
    from hymem.core.time import validate_event_clock

    # Validate the wire's two clocks before mutating claim state. Relative
    # source/transaction comparisons make replay deterministic. In addition,
    # one destination acceptance coordinate bounds every transaction timestamp:
    # HyMem does not support importing scheduled future authority.
    accepted_at = conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
    # Keep the causal source diagnostic first: this was the original v7 clock
    # invariant and is more actionable than a downstream coverage-clock error.
    for record in grouped.get("edge_evidence", []):
        if record["provenance_status"] == "canonical":
            validate_event_clock(
                conn,
                record["source_event_at"],
                record["extracted_at"],
                context="portable canonical evidence",
            )
    for record in grouped.get("session", []):
        validate_event_clock(
            conn, record["started_at"], accepted_at,
            context="portable session transaction",
        )
        if record["ended_at"] is not None:
            validate_event_clock(
                conn, record["ended_at"], accepted_at,
                context="portable session close transaction",
            )
            validate_timestamp_order(
                record["started_at"], record["ended_at"],
                context="portable session lifecycle",
            )
    for record in grouped.get("chunk", []):
        validate_event_clock(
            conn, record["created_at"], accepted_at,
            context="portable chunk transaction",
        )
    for record in grouped.get("message_retention_coverage", []):
        validate_event_clock(
            conn, record["created_at"], accepted_at,
            context="portable coverage transaction",
        )
        if record.get("source_peer_id") is not None:
            normalize_iso_timestamp(
                record["source_created_at"],
                context="portable external message source",
            )
            validate_event_clock(
                conn, record["source_created_at"], accepted_at,
                context="portable external message source acceptance",
            )
        elif record["source_created_at"] is not None:
            try:
                normalize_iso_timestamp(
                    record["source_created_at"],
                    context="portable message source",
                )
            except ValueError:
                # Lossless coverage can preserve exact pre-v40/direct-SQL
                # message metadata whose occurrence time was absent or
                # malformed.  Such a value is never transaction authority:
                # canonical evidence may reference it only through the
                # explicit ancient source-event sentinel validated above.
                continue
            validate_event_clock(
                conn, record["source_created_at"], accepted_at,
                context="portable message source acceptance",
            )
    for record in grouped.get("edge", []):
        for field in ("first_seen", "last_seen", "last_reinforced"):
            if record[field] is not None:
                validate_event_clock(
                    conn, record[field], accepted_at,
                    context=f"portable edge {field} transaction",
                )
    for record in grouped.get("edge_evidence", []):
        if record["provenance_status"] == "canonical":
            validate_event_clock(
                conn,
                record["source_event_at"],
                accepted_at,
                context="portable canonical evidence acceptance",
            )
        validate_event_clock(
            conn,
            record["extracted_at"],
            accepted_at,
            context="portable evidence transaction",
        )
        if record["published_at"] is not None:
            validate_event_clock(
                conn,
                record["published_at"],
                accepted_at,
                context="portable evidence publication transaction",
            )
        if record["superseded_at"] is not None:
            validate_event_clock(
                conn,
                record["superseded_at"],
                accepted_at,
                context="portable evidence retirement transaction",
            )
            if _normalized_wire_event(record["superseded_at"]) < (
                _normalized_wire_event(record["extracted_at"])
            ):
                raise ValueError(
                    "portable evidence retirement precedes extraction"
                )
    for record in grouped.get("edge_lifecycle", []):
        validate_event_clock(
            conn,
            record["event_at"],
            record["created_at"],
            context="portable lifecycle event",
        )
        validate_event_clock(
            conn,
            record["created_at"],
            accepted_at,
            context="portable lifecycle transaction",
        )
        validate_event_clock(
            conn,
            record["event_at"],
            accepted_at,
            context="portable lifecycle acceptance",
        )
    for record in grouped.get("edge_evidence_signal", []):
        validate_event_clock(
            conn,
            record["created_at"],
            accepted_at,
            context="portable evidence signal transaction",
        )
    for record in grouped.get("claim_observation", []):
        validate_event_clock(
            conn,
            record["observed_at"],
            accepted_at,
            context="portable claim observation transaction",
        )
    for record in grouped.get("claim_extraction_outcome", []):
        validate_event_clock(
            conn,
            record["succeeded_at"],
            accepted_at,
            context="portable claim outcome transaction",
        )

    edge_ids = _mapped_edge_ids(conn, grouped)

    alias_inserted = 0
    for record in sorted(grouped.get("entity_alias", []), key=lambda row: row["alias"]):
        existing = conn.execute(
            "SELECT canonical FROM entity_aliases WHERE alias=?", (record["alias"],)
        ).fetchone()
        if existing is not None:
            if existing["canonical"] != record["canonical"]:
                raise ValueError("portable entity alias collides with target")
            continue
        conn.execute(
            "INSERT INTO entity_aliases(alias,canonical) VALUES (?,?)",
            (record["alias"], record["canonical"]),
        )
        alias_inserted += 1
    inserted["entity_alias"] = alias_inserted

    # Publish manifests only after every exact coverage artifact has landed and
    # been validated by the base importer.
    members_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for member in grouped.get("chunk_message_source", []):
        members_by_chunk[member["chunk_id"]].append(member)
    manifest_inserted = 0
    member_inserted = 0
    for manifest in sorted(
        grouped.get("chunk_source_manifest", []), key=lambda row: row["id"]
    ):
        chunk_id = manifest["id"]
        desired = sorted(members_by_chunk[chunk_id], key=lambda row: row["ordinal"])
        header = conn.execute(
            "SELECT source_manifest_version,source_manifest_count FROM chunks WHERE id=?",
            (chunk_id,),
        ).fetchone()
        existing = conn.execute(
            "SELECT chunk_id,ordinal,source_message_id,source_session_id,"
            "source_coverage_chunk_id,source_coverage_version "
            "FROM chunk_message_sources WHERE chunk_id=? ORDER BY ordinal",
            (chunk_id,),
        ).fetchall()
        member_columns = list(_V7_COLS_BY_KIND["chunk_message_source"])
        if header["source_manifest_version"] is not None:
            if tuple(header) != (
                manifest["source_manifest_version"], manifest["source_manifest_count"],
            ) or len(existing) != len(desired) or any(
                not _rows_match(old, new, member_columns)
                for old, new in zip(existing, desired)
            ):
                raise ValueError("portable chunk source manifest collides with target")
            continue
        if existing:
            if len(existing) != len(desired) or any(
                not _rows_match(old, new, member_columns)
                for old, new in zip(existing, desired)
            ):
                raise ValueError("portable chunk source members collide with target")
        else:
            conn.executemany(
                "INSERT INTO chunk_message_sources("
                + ",".join(member_columns) + ") VALUES (?,?,?,?,?,?)",
                [[record[column] for column in member_columns] for record in desired],
            )
            member_inserted += len(desired)
        conn.execute(
            "UPDATE chunks SET source_manifest_version=?,source_manifest_count=? "
            "WHERE id=?",
            (
                manifest["source_manifest_version"],
                manifest["source_manifest_count"], chunk_id,
            ),
        )
        manifest_inserted += 1
    inserted["chunk_source_manifest"] = manifest_inserted
    inserted["chunk_message_source"] = member_inserted

    # Merge whole-chunk publication authority before observations. In
    # particular, a higher-generation empty result must delete the destination
    # chunk's older observations even though the wire contains no replacement
    # observation row. Conversely, a stale wire outcome contributes history but
    # never resurrects its observation set.
    stale_outcome_chunks: set[str] = set()
    outcome_affected_edge_ids: set[int] = set()
    outcome_inserted = 0
    outcome_changed = False
    with core_db.evidence_mutation(conn):
        for record in sorted(
            grouped.get("claim_extraction_outcome", []),
            key=lambda row: row["chunk_id"],
        ):
            chunk_id = str(record["chunk_id"])
            existing = conn.execute(
                "SELECT prompt_version,prompt_generation,result_hash,succeeded_at "
                "FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            incoming_generation = int(record["prompt_generation"])
            if existing is not None:
                target_generation = int(existing["prompt_generation"])
                if incoming_generation < target_generation:
                    stale_outcome_chunks.add(chunk_id)
                elif incoming_generation == target_generation:
                    if existing["result_hash"] != record["result_hash"]:
                        raise ValueError(
                            "same prompt generation claim extraction outcomes disagree"
                        )
                    winner_version = max(
                        str(existing["prompt_version"]),
                        str(record["prompt_version"]),
                    )
                    winner_succeeded = _merged_timestamp(
                        existing["succeeded_at"], record["succeeded_at"], latest=True
                    )
                    if (
                        winner_version != existing["prompt_version"]
                        or winner_succeeded != existing["succeeded_at"]
                    ):
                        conn.execute(
                            "UPDATE kg_claim_extraction_outcomes SET "
                            "prompt_version=?,succeeded_at=? WHERE chunk_id=?",
                            (winner_version, winner_succeeded, chunk_id),
                        )
                        conn.execute(
                            "UPDATE kg_claim_observations SET prompt_version=? "
                            "WHERE chunk_id=? AND prompt_generation=?",
                            (winner_version, chunk_id, target_generation),
                        )
                else:
                    outcome_affected_edge_ids.update(
                        evidence_ledger.begin_chunk_extraction_reconciliation(
                            conn,
                            chunk_id=chunk_id,
                            prompt_version=str(record["prompt_version"]),
                        )
                    )
                    conn.execute(
                        "UPDATE kg_claim_extraction_outcomes SET prompt_version=?,"
                        "prompt_generation=?,result_hash=?,succeeded_at=? "
                        "WHERE chunk_id=?",
                        (
                            record["prompt_version"], incoming_generation,
                            record["result_hash"], record["succeeded_at"], chunk_id,
                        ),
                    )
                    outcome_changed = True
            else:
                # Initialized v40 stores were conservatively backfilled by v41;
                # absence therefore means no proven prior whole-chunk outcome.
                # Any unattributed/observation state for this chunk is older
                # authority and is replaced by the incoming publication.
                outcome_affected_edge_ids.update(
                    evidence_ledger.begin_chunk_extraction_reconciliation(
                        conn,
                        chunk_id=chunk_id,
                        prompt_version=str(record["prompt_version"]),
                    )
                )
                conn.execute(
                    "INSERT INTO kg_claim_extraction_outcomes("
                    "chunk_id,prompt_version,prompt_generation,result_hash,succeeded_at) "
                    "VALUES (?,?,?,?,?)",
                    (
                        chunk_id, record["prompt_version"], incoming_generation,
                        record["result_hash"], record["succeeded_at"],
                    ),
                )
                outcome_inserted += 1
                outcome_changed = True
            # The outcome itself proves this exact prompt completed. Preserve
            # the runtime selection gate even though processed_chunks is not a
            # durable semantic record in older wire versions.
            conn.execute(
                "INSERT INTO processed_chunks(chunk_id,prompt_version,processed_at) "
                "VALUES (?,?,?) ON CONFLICT(chunk_id,prompt_version) DO UPDATE SET "
                "processed_at=MIN(processed_at,excluded.processed_at)",
                (chunk_id, record["prompt_version"], record["succeeded_at"]),
            )
    inserted["claim_extraction_outcome"] = outcome_inserted
    if outcome_changed:
        inserted["_claim_extraction_outcome_changed"] = 1

    # Merge observation authority before materializing evidence state. A wire
    # revision number is local to its source database: two stores can process
    # the same message independently and both call different interpretations
    # revision 1. Prompt generation, not import order or that local number,
    # decides which interpretation is current.
    wire_evidence = {
        int(record["id"]): record
        for record in grouped.get("edge_evidence", [])
    }
    affected_edge_ids = sorted(
        set(edge_ids.values()) | outcome_affected_edge_ids
    )
    target_observations: list[dict] = []
    if affected_edge_ids:
        placeholders = ",".join("?" for _ in affected_edge_ids)
        target_observations = [
            {**dict(row), "_origin": "target"}
            for row in conn.execute(
                "SELECT observation.*,outcome.succeeded_at AS _publication_at "
                "FROM kg_claim_observations observation "
                "JOIN kg_claim_extraction_outcomes outcome "
                "ON outcome.chunk_id=observation.chunk_id "
                "AND outcome.prompt_version=observation.prompt_version "
                "AND outcome.prompt_generation=observation.prompt_generation "
                f"WHERE observation.edge_id IN ({placeholders})",
                affected_edge_ids,
            ).fetchall()
        ]
    wire_outcomes = {
        (
            str(record["chunk_id"]), str(record["prompt_version"]),
            int(record["prompt_generation"]),
        ): record["succeeded_at"]
        for record in grouped.get("claim_extraction_outcome", [])
    }
    wire_observations: list[dict] = []
    for record in grouped.get("claim_observation", []):
        if record["chunk_id"] in stale_outcome_chunks:
            continue
        mapped = dict(record)
        mapped["edge_id"] = edge_ids[int(record["edge_id"])]
        mapped["_origin"] = "wire"
        mapped["_wire_evidence_id"] = int(record["evidence_id"])
        outcome_key = (
            str(record["chunk_id"]), str(record["prompt_version"]),
            int(record["prompt_generation"]),
        )
        if outcome_key not in wire_outcomes:
            raise ValueError(
                "portable claim observation lacks whole-chunk publication"
            )
        mapped["_publication_at"] = wire_outcomes[outcome_key]
        wire_observations.append(mapped)
    supported_wire_evidence_ids = {
        int(record["_wire_evidence_id"]) for record in wire_observations
    }

    def first_occurrence_audit_rank(record) -> tuple:
        """Choose one coherent immutable first-occurrence audit snapshot."""
        prompt = str(record["extraction_prompt_version"] or "")
        normalized_extracted = _normalized_wire_event(record["extracted_at"])
        audit_payload = json.dumps(
            [
                record["chunk_id"], record["surface_subject"],
                record["surface_object"], prompt, normalized_extracted,
            ],
            ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        )
        return (normalized_extracted, audit_payload)

    merged_observations: dict[tuple[object, ...], dict] = {}
    for candidate in [*target_observations, *wire_observations]:
        identity = _observation_identity(int(candidate["edge_id"]), candidate)
        previous = merged_observations.get(identity)
        if previous is None:
            merged_observations[identity] = candidate
            continue
        previous_generation = int(previous["prompt_generation"])
        candidate_generation = int(candidate["prompt_generation"])
        if (
            previous_generation == candidate_generation
            and _observation_semantic(previous) != _observation_semantic(candidate)
        ):
            raise ValueError(
                "portable observations diverge at one prompt generation"
            )
        previous_rank = (
            previous_generation,
            str(previous["prompt_version"]),
            _normalized_wire_event(previous.get("observed_at")),
            str(previous.get("observed_at") or ""),
        )
        candidate_rank = (
            candidate_generation,
            str(candidate["prompt_version"]),
            _normalized_wire_event(candidate.get("observed_at")),
            str(candidate.get("observed_at") or ""),
        )
        if candidate_rank > previous_rank:
            merged_observations[identity] = candidate

    observations_by_base: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for observation in merged_observations.values():
        base = (
            int(observation["edge_id"]), observation["source_session_id"],
            int(observation["source_message_id"]), observation["evidence_kind"],
        )
        observations_by_base[base].append(observation)
    authority: dict[tuple[object, ...], tuple[tuple[int, str], str]] = {}
    for base, observations in observations_by_base.items():
        winning_generation = max(
            int(record["prompt_generation"]) for record in observations
        )
        winners = [
            record for record in observations
            if int(record["prompt_generation"]) == winning_generation
        ]
        semantics = {_observation_semantic(record) for record in winners}
        if len(semantics) != 1:
            raise ValueError(
                "portable observations diverge at one prompt generation"
            )
        winner_at = max(
            _normalized_wire_event(record.get("_publication_at"))
            for record in winners
        )
        authority[base] = (next(iter(semantics)), winner_at)

    def effective_retirement_at(record, winner_at: str) -> str:
        """Close a losing revision no earlier than it could be published.

        A higher prompt generation can have completed earlier on another
        branch than a stale lower-generation extraction.  Retiring that stale
        evidence at the global winner's earlier clock would invert its
        transaction interval and violate ``published_at <= superseded_at``.
        Such a late-arriving loser instead receives a conservative zero-width
        interval at its own publication boundary.
        """
        # Schema v42 deliberately leaves staged/pre-migration orphan evidence
        # unpublished. Such a row is not historical authority, but it must not
        # poison recovery when a valid branch supplies newer prompt authority.
        # Keep its publication NULL and retire it at the winner boundary.
        if record["published_at"] is None:
            return _normalized_wire_event(winner_at)
        published_at = _normalized_wire_event(record["published_at"])
        return max(_normalized_wire_event(winner_at), published_at)

    # Existing lower-generation current rows must vacate the unique current
    # identity before a winning imported branch is inserted. Retirement state
    # is reducer-derived during an additive merge; incoming mutable flags are
    # never allowed to override a higher-generation local observation.
    with core_db.evidence_history_mutation(conn):
        for base, (semantic, winner_at) in authority.items():
            rows = conn.execute(
                "SELECT id,polarity,interpretation_key,published_at "
                "FROM kg_evidence "
                "WHERE edge_id=? AND source_session_id=? AND source_message_id=? "
                "AND evidence_kind=? AND provenance_status='canonical' "
                "AND is_current=1",
                base,
            ).fetchall()
            for row in rows:
                if _observation_semantic(row) != semantic:
                    retirement_at = effective_retirement_at(row, winner_at)
                    conn.execute(
                        "UPDATE kg_evidence SET is_current=0,superseded_at=?,"
                        "superseded_reason='lower_prompt_authority' WHERE id=?",
                        (retirement_at, row["id"]),
                    )

    evidence_id_map: dict[int, int] = {}
    evidence_inserted = 0
    evidence_columns = [
        column for column in _COLS_BY_KIND["edge_evidence"]
        if column not in {"id", "edge_id"}
    ]
    with core_db.evidence_history_mutation(conn):
        ordered_evidence = sorted(
            grouped.get("edge_evidence", []),
            key=lambda row: (
                _edge_natural(next(
                    edge for edge in grouped["edge"] if edge["id"] == row["edge_id"]
                )),
                row["provenance_status"], row["source_session_id"] or "",
                row["source_message_id"] or 0, row["chunk_id"],
                row["evidence_kind"], row["revision"], row["interpretation_key"],
            ),
        )
        for record in ordered_evidence:
            target_edge_id = edge_ids[int(record["edge_id"])]
            if record["provenance_status"] == "canonical":
                base = _canonical_evidence_base(target_edge_id, record)
                rows = conn.execute(
                    "SELECT * FROM kg_evidence WHERE edge_id=? "
                    "AND source_session_id=? AND source_message_id=? "
                    "AND evidence_kind=? AND provenance_status='canonical' "
                    "ORDER BY revision,id",
                    base,
                ).fetchall()
                immutable = (
                    "polarity", "evidence_weight", "weight_source", "source_role",
                    "source_peer_id", "source_workspace_id",
                    "value_text", "value_numeric", "value_unit", "temporal_scope",
                    "source_message_id", "source_session_id", "source_created_at",
                    "source_event_at", "source_coverage_chunk_id",
                    "source_coverage_version", "provenance_status",
                    "interpretation_key", "evidence_kind",
                )

                def history_clock(value):
                    if value is None:
                        return None
                    try:
                        return _normalized_wire_event(value)
                    except ValueError:
                        return ("invalid", str(value))

                def same_immutable_occurrence(row) -> bool:
                    # Revision ordinals are local.  A wire replay must first
                    # recognize an already-imported immutable occurrence even
                    # when deterministic renumbering moved it away from the
                    # source ordinal. Reducer-derived current/retirement state
                    # is intentionally excluded: a stale snapshot can call the
                    # same occurrence current after a newer target already
                    # retired it. A genuine revival has a new extraction and
                    # publication occurrence and therefore does not match.
                    return (
                        all(row[field] == record[field] for field in immutable)
                        and row["chunk_id"] == record["chunk_id"]
                        and row["surface_subject"] == record["surface_subject"]
                        and row["surface_object"] == record["surface_object"]
                        and row["extraction_prompt_version"]
                        == record["extraction_prompt_version"]
                        and history_clock(row["extracted_at"])
                        == history_clock(record["extracted_at"])
                        and history_clock(row["published_at"])
                        == history_clock(record["published_at"])
                    )

                immutable_matches = [
                    row for row in rows if same_immutable_occurrence(row)
                ]
                full_history_matches = [
                    row for row in immutable_matches
                    if int(row["is_current"]) == int(record["is_current"])
                    and history_clock(row["superseded_at"])
                    == history_clock(record["superseded_at"])
                    and row["superseded_reason"] == record["superseded_reason"]
                ]

                def select_occurrence(candidates):
                    if len(candidates) == 1:
                        return candidates[0]
                    ordinal = [
                        row for row in candidates
                        if int(row["revision"]) == int(record["revision"])
                    ]
                    if len(ordinal) == 1:
                        return ordinal[0]
                    return None

                # State is the strongest discriminator for same-millisecond
                # semantic revivals.  Only when it no longer matches because
                # a newer reducer retired the occurrence do we fall back to
                # the immutable first-occurrence audit.
                existing = select_occurrence(full_history_matches)
                if existing is None:
                    existing = select_occurrence(immutable_matches)
                if existing is None and (
                    len(full_history_matches) > 1 or len(immutable_matches) > 1
                ):
                    raise ValueError(
                        "portable evidence history matches multiple target revisions"
                    )
                preserve_distinct_interval = False
                if existing is None:
                    existing = next((
                        row for row in rows
                        if int(row["revision"]) == int(record["revision"])
                        and row["interpretation_key"]
                        == record["interpretation_key"]
                    ), None)
                revision_collision = next((
                    row for row in rows
                    if int(row["revision"]) == int(record["revision"])
                    and row["interpretation_key"] != record["interpretation_key"]
                ), None)
                state_diverges = (
                    existing is not None
                    and int(existing["is_current"]) != int(record["is_current"])
                )
                wanted = authority.get(base)
                occurrence_is_winner = bool(
                    wanted is not None
                    and wanted[0] == _observation_semantic(record)
                )
                if state_diverges and occurrence_is_winner and (
                    int(existing["is_current"]) == 1
                    or int(record["id"]) in supported_wire_evidence_ids
                ):
                    # Retirement is append-only. A current branch copy and a
                    # retired branch copy therefore represent two distinct
                    # transaction intervals even when their wire-local
                    # revision/interpretation are identical. Preserve both;
                    # deterministic renumbering below will assign the retired
                    # interval first in either import order.
                    revision_collision = existing
                    existing = None
                    preserve_distinct_interval = True
                # A prior branch merge may have deterministically moved this
                # semantic revision away from its wire-local ordinal. Only use
                # this fallback when that original ordinal is now occupied by
                # another interpretation; a free later ordinal with the same
                # semantic key can be a real append-only revival and must not
                # be collapsed.
                if (
                    existing is None
                    and revision_collision is not None
                    and not preserve_distinct_interval
                ):
                    semantic_matches = [
                        row for row in rows
                        if row["interpretation_key"] == record["interpretation_key"]
                    ]
                    if len(semantic_matches) == 1:
                        existing = semantic_matches[0]
                if existing is not None:
                    if any(existing[field] != record[field] for field in immutable):
                        raise ValueError(
                            "portable evidence revision collides with target"
                        )
                    incoming_rank = first_occurrence_audit_rank(record)
                    existing_rank = first_occurrence_audit_rank(existing)
                    earliest_published_at = earliest_timestamp_spelling(
                        existing["published_at"], record["published_at"]
                    )
                    merged_is_current = int(existing["is_current"])
                    merged_superseded_at = existing["superseded_at"]
                    merged_superseded_reason = existing["superseded_reason"]
                    if not merged_is_current:
                        retirement_candidates = []
                        for candidate in (existing, record):
                            if candidate["superseded_at"] is None:
                                continue
                            retirement_candidates.append((
                                _normalized_wire_event(
                                    candidate["superseded_at"]
                                ),
                                (
                                    candidate["superseded_reason"]
                                    != "lower_prompt_authority"
                                ),
                                str(candidate["superseded_reason"]),
                                str(candidate["superseded_at"]),
                            ))
                        if retirement_candidates:
                            (
                                _retirement_instant,
                                _specific_reason,
                                merged_superseded_reason,
                                _retirement_spelling,
                            ) = max(retirement_candidates)
                            # Durable transaction coordinates use the shared
                            # UTC-millisecond spelling, independent of which
                            # branch supplied an equivalent raw timestamp.
                            merged_superseded_at = _retirement_instant
                    if incoming_rank < existing_rank:
                        conn.execute(
                            "UPDATE kg_evidence SET chunk_id=?,surface_subject=?,"
                            "surface_object=?,extraction_prompt_version=?,extracted_at=?,"
                            "published_at=?,is_current=?,superseded_at=?,"
                            "superseded_reason=? "
                            "WHERE id=?",
                            (
                                record["chunk_id"], record["surface_subject"],
                                record["surface_object"],
                                record["extraction_prompt_version"],
                                _normalized_wire_event(record["extracted_at"]),
                                earliest_published_at,
                                merged_is_current, merged_superseded_at,
                                merged_superseded_reason,
                                existing["id"],
                            ),
                        )
                    elif (
                        existing["published_at"] != earliest_published_at
                        or existing["superseded_at"] != merged_superseded_at
                        or existing["superseded_reason"]
                        != merged_superseded_reason
                    ):
                        conn.execute(
                            "UPDATE kg_evidence SET published_at=?,superseded_at=?,"
                            "superseded_reason=? WHERE id=?",
                            (
                                earliest_published_at, merged_superseded_at,
                                merged_superseded_reason, existing["id"],
                            ),
                        )
                    target_evidence_id = int(existing["id"])
                else:
                    used_revisions = {int(row["revision"]) for row in rows}
                    target_revision = int(record["revision"])
                    if target_revision in used_revisions:
                        target_revision = max(used_revisions, default=0) + 1
                    wanted = authority.get(base)
                    semantic = _observation_semantic(record)
                    current = next(
                        (row for row in rows if int(row["is_current"]) == 1), None
                    )
                    should_be_current = bool(
                        wanted is not None
                        and wanted[0] == semantic
                        and current is None
                        and int(record["is_current"]) == 1
                    )
                    inserted_record = dict(record)
                    inserted_record["extracted_at"] = _normalized_wire_event(
                        record["extracted_at"]
                    )
                    if record["published_at"] is not None:
                        inserted_record["published_at"] = _normalized_wire_event(
                            record["published_at"]
                        )
                    if record["superseded_at"] is not None:
                        inserted_record["superseded_at"] = _normalized_wire_event(
                            record["superseded_at"]
                        )
                    inserted_record["revision"] = target_revision
                    inserted_record["is_current"] = 1 if should_be_current else 0
                    if should_be_current:
                        inserted_record["superseded_at"] = None
                        inserted_record["superseded_reason"] = None
                    elif wanted is not None and wanted[0] != semantic:
                        retirement_at = effective_retirement_at(record, wanted[1])
                        inserted_record["superseded_at"] = retirement_at
                        # A branch may already have closed this exact occurrence
                        # at the winning publication boundary with a more
                        # specific immutable audit reason (for example
                        # ``source_reinterpreted``).  Do not erase that reason
                        # merely because the portable reducer independently
                        # reaches the same boundary.  Replacing it here made a
                        # later zero-count replay silently mutate the export.
                        incoming_boundary_matches = bool(
                            not int(record["is_current"])
                            and record["superseded_at"] is not None
                            and history_clock(record["superseded_at"])
                            == history_clock(retirement_at)
                            and record["superseded_reason"]
                        )
                        inserted_record["superseded_reason"] = (
                            record["superseded_reason"]
                            if incoming_boundary_matches
                            else "lower_prompt_authority"
                        )
                    elif record["superseded_at"] is None:
                        retirement = evidence_ledger.claim_retirement_authority(
                            conn,
                            source_session_id=record["source_session_id"],
                            source_message_id=int(record["source_message_id"]),
                        )
                        if retirement is not None:
                            (
                                inserted_record["superseded_at"],
                                inserted_record["superseded_reason"],
                            ) = retirement
                        else:
                            inserted_record["superseded_at"] = (
                                record["source_event_at"]
                            )
                            inserted_record["superseded_reason"] = (
                                "portable_noncurrent_revision"
                            )
                    values = [
                        target_edge_id,
                        *[inserted_record[column] for column in evidence_columns],
                    ]
                    cur = conn.execute(
                        "INSERT INTO kg_evidence(edge_id,"
                        + ",".join(evidence_columns) + ") VALUES ("
                        + ",".join("?" for _ in values) + ")",
                        values,
                    )
                    target_evidence_id = int(cur.lastrowid)
                    evidence_inserted += 1
                evidence_id_map[int(record["id"])] = target_evidence_id
                continue
            else:
                where = (
                    "edge_id=? AND provenance_status='legacy_unattributed' "
                    "AND chunk_id=? AND evidence_kind=? AND revision=?"
                )
                params = (
                    target_edge_id, record["chunk_id"], record["evidence_kind"],
                    record["revision"],
                )
            existing = conn.execute(
                "SELECT * FROM kg_evidence WHERE " + where, params
            ).fetchone()
            comparable = dict(record)
            comparable["edge_id"] = target_edge_id
            if existing is not None:
                if not _rows_match(existing, comparable, ["edge_id", *evidence_columns]):
                    raise ValueError("portable evidence revision collides with target")
                target_evidence_id = int(existing["id"])
            else:
                values = [target_edge_id, *[record[column] for column in evidence_columns]]
                cur = conn.execute(
                    "INSERT INTO kg_evidence(edge_id," + ",".join(evidence_columns)
                    + ") VALUES (" + ",".join("?" for _ in values) + ")",
                    values,
                )
                target_evidence_id = int(cur.lastrowid)
                evidence_inserted += 1
            evidence_id_map[int(record["id"])] = target_evidence_id

        # Canonicalize lower-authority retirement metadata after the union so
        # importing branch A then B and B then A converges byte-for-byte on the
        # mutable authority projection.
        for base, (semantic, winner_at) in authority.items():
            rows = conn.execute(
                "SELECT id,polarity,interpretation_key,is_current,"
                "published_at,superseded_at,superseded_reason "
                "FROM kg_evidence WHERE edge_id=? AND source_session_id=? "
                "AND source_message_id=? AND evidence_kind=? "
                "AND provenance_status='canonical'",
                base,
            ).fetchall()
            for row in rows:
                if _observation_semantic(row) == semantic:
                    continue
                retirement_at = effective_retirement_at(row, winner_at)
                existing_boundary = (
                    history_clock(row["superseded_at"])
                    if row["superseded_at"] is not None else None
                )
                boundary_matches = existing_boundary == history_clock(retirement_at)
                # Preserve a specific nonempty retirement reason when the
                # existing immutable interval already closes at exactly the
                # reducer's authority boundary.  Genericize only a newly
                # retired row, a changed boundary, or missing reason.
                if (
                    int(row["is_current"]) != 0
                    or not boundary_matches
                    or not row["superseded_reason"]
                ):
                    conn.execute(
                        "UPDATE kg_evidence SET is_current=0,superseded_at=?,"
                        "superseded_reason='lower_prompt_authority' WHERE id=?",
                        (retirement_at, row["id"]),
                    )
    inserted["edge_evidence"] = evidence_inserted

    signal_inserted = 0
    signal_columns = [
        column for column in _V7_COLS_BY_KIND["edge_evidence_signal"]
        if column not in {"id", "edge_id"}
    ]
    with core_db.evidence_mutation(conn):
        for record in sorted(
            grouped.get("edge_evidence_signal", []),
            key=lambda row: (
                edge_ids[int(row["edge_id"])], row["signal_kind"], row["signal_key"],
            ),
        ):
            target_edge_id = edge_ids[int(record["edge_id"])]
            # Heal the two released-before-v7 local-id key forms in place.
            candidates = [record["signal_key"]]
            if record["signal_kind"] == "legacy_unattributed":
                if record["signal_key"] in {"legacy:positive", "legacy:negative"}:
                    candidates.append(f"{record['signal_key']}:{target_edge_id}")
            if record["signal_kind"] == "runtime_unattributed" and record[
                "signal_key"
            ].startswith("runtime-unattributed:polarity:"):
                polarity = record["signal_key"].rsplit(":", 1)[-1]
                candidates.append(f"edge:{target_edge_id}:polarity:{polarity}")
            placeholders = ",".join("?" for _ in candidates)
            existing = conn.execute(
                "SELECT * FROM kg_evidence_signals WHERE edge_id=? "
                "AND signal_kind=? AND signal_key IN (" + placeholders + ")",
                (target_edge_id, record["signal_kind"], *candidates),
            ).fetchone()
            comparable = dict(record)
            comparable["edge_id"] = target_edge_id
            if existing is not None:
                existing_values = dict(existing)
                existing_values["signal_key"] = _portable_signal_key(existing_values)
                if not all(
                    existing_values[column] == comparable[column]
                    for column in ["edge_id", *signal_columns]
                ):
                    raise ValueError("portable evidence signal collides with target")
                if existing["signal_key"] != record["signal_key"]:
                    conn.execute(
                        "UPDATE kg_evidence_signals SET signal_key=? WHERE id=?",
                        (record["signal_key"], existing["id"]),
                    )
            else:
                values = [target_edge_id, *[record[column] for column in signal_columns]]
                conn.execute(
                    "INSERT INTO kg_evidence_signals(edge_id," + ",".join(signal_columns)
                    + ") VALUES (" + ",".join("?" for _ in values) + ")",
                    values,
                )
                signal_inserted += 1
    inserted["edge_evidence_signal"] = signal_inserted

    observation_inserted = 0
    observation_columns = list(_V7_COLS_BY_KIND["claim_observation"])
    with core_db.evidence_history_mutation(conn):
        for record in sorted(
            grouped.get("claim_observation", []),
            key=lambda row: (
                row["chunk_id"], edge_ids[int(row["edge_id"])],
                row["source_session_id"], row["source_message_id"],
                row["evidence_kind"],
            ),
        ):
            if record["chunk_id"] in stale_outcome_chunks:
                continue
            mapped = dict(record)
            mapped["edge_id"] = edge_ids[int(record["edge_id"])]
            mapped["evidence_id"] = evidence_id_map[int(record["evidence_id"])]
            existing = conn.execute(
                "SELECT * FROM kg_claim_observations WHERE chunk_id=? AND edge_id=? "
                "AND source_session_id=? AND source_message_id=? AND evidence_kind=?",
                (
                    mapped["chunk_id"], mapped["edge_id"],
                    mapped["source_session_id"], mapped["source_message_id"],
                    mapped["evidence_kind"],
                ),
            ).fetchone()
            if existing is not None:
                existing_generation = int(existing["prompt_generation"])
                incoming_generation = int(mapped["prompt_generation"])
                if (
                    existing_generation == incoming_generation
                    and _observation_semantic(existing)
                    != _observation_semantic(mapped)
                ):
                    raise ValueError("portable claim observation collides with target")
                incoming_rank = (
                    incoming_generation, str(mapped["prompt_version"]),
                    _normalized_wire_event(mapped.get("observed_at")),
                    str(mapped.get("observed_at") or ""),
                )
                existing_rank = (
                    existing_generation, str(existing["prompt_version"]),
                    _normalized_wire_event(existing["observed_at"]),
                    str(existing["observed_at"] or ""),
                )
                if incoming_rank > existing_rank:
                    conn.execute(
                        "UPDATE kg_claim_observations SET polarity=?,prompt_version=?,"
                        "prompt_generation=?,evidence_id=?,interpretation_key=?,"
                        "observed_at=? WHERE chunk_id=? AND edge_id=? "
                        "AND source_session_id=? AND source_message_id=? "
                        "AND evidence_kind=?",
                        (
                            mapped["polarity"], mapped["prompt_version"],
                            mapped["prompt_generation"], mapped["evidence_id"],
                            mapped["interpretation_key"], mapped["observed_at"],
                            mapped["chunk_id"], mapped["edge_id"],
                            mapped["source_session_id"],
                            mapped["source_message_id"], mapped["evidence_kind"],
                        ),
                    )
                continue
            conn.execute(
                "INSERT INTO kg_claim_observations(" + ",".join(observation_columns)
                + ") VALUES (" + ",".join("?" for _ in observation_columns) + ")",
                [mapped[column] for column in observation_columns],
            )
            observation_inserted += 1

        # Revision ordinals are local implementation details until histories
        # from independently evolved stores meet. Rebind them deterministically
        # from prompt chronology and interpretation identity, then rekey all
        # pre-existing lifecycle rows before importing the wire lifecycle.
        canonical_bases = sorted({
            _canonical_evidence_base(
                edge_ids[int(record["edge_id"])], record
            )
            for record in grouped.get("edge_evidence", [])
            if record["provenance_status"] == "canonical"
        })
        for base in canonical_bases:
            rows = conn.execute(
                "SELECT ev.*,MIN(observation.prompt_generation) AS observed_generation "
                "FROM kg_evidence ev LEFT JOIN kg_claim_observations observation "
                "ON observation.evidence_id=ev.id WHERE ev.edge_id=? "
                "AND ev.source_session_id=? AND ev.source_message_id=? "
                "AND ev.evidence_kind=? AND ev.provenance_status='canonical' "
                "GROUP BY ev.id",
                base,
            ).fetchall()
            ordered = sorted(
                rows,
                key=lambda row: (
                    int(row["observed_generation"])
                    if row["observed_generation"] is not None
                    else evidence_ledger.prompt_generation(
                        str(row["extraction_prompt_version"] or "")
                    ),
                    str(row["interpretation_key"]), int(row["polarity"]),
                    1 if int(row["is_current"]) else 0,
                    _normalized_wire_event(row["extracted_at"]),
                    str(row["chunk_id"]), int(row["revision"]),
                ),
            )
            changes = [
                (int(row["id"]), wanted)
                for wanted, row in enumerate(ordered, start=1)
                if int(row["revision"]) != wanted
            ]
            if changes:
                temporary = max(int(row["revision"]) for row in rows) + len(rows) + 1
                for offset, (evidence_id, _wanted) in enumerate(changes):
                    conn.execute(
                        "UPDATE kg_evidence SET revision=? WHERE id=?",
                        (temporary + offset, evidence_id),
                    )
                for evidence_id, wanted in changes:
                    conn.execute(
                        "UPDATE kg_evidence SET revision=? WHERE id=?",
                        (wanted, evidence_id),
                    )
        evidence_ledger.recanonicalize_lifecycle_keys(conn)
    inserted["claim_observation"] = observation_inserted

    lifecycle_id_map: dict[int, int] = {}
    lifecycle_inserted = 0
    lifecycle_columns = [
        column for column in _V7_COLS_BY_KIND["edge_lifecycle"]
        if column not in {"id", "edge_id", "source_evidence_id"}
    ]
    wire_dependencies: dict[int, list[int]] = defaultdict(list)
    for dependency in grouped.get("lifecycle_dependency", []):
        wire_dependencies[int(dependency["lifecycle_id"])].append(
            int(dependency["evidence_id"])
        )
    with core_db.evidence_history_mutation(conn):
        for record in sorted(
            grouped.get("edge_lifecycle", []),
            key=lambda row: (
                edge_ids[int(row["edge_id"])], row["event_at"], row["event_key"],
            ),
        ):
            target_edge_id = edge_ids[int(record["edge_id"])]
            target_source_id = (
                evidence_id_map[int(record["source_evidence_id"])]
                if record["source_evidence_id"] is not None else None
            )
            target_dependency_ids = sorted({
                evidence_id_map[wire_evidence_id]
                for wire_evidence_id in wire_dependencies.get(int(record["id"]), [])
            })
            if record["event_kind"] == "claim_assertion":
                source = conn.execute(
                    "SELECT source_session_id,source_message_id,evidence_kind,revision "
                    "FROM kg_evidence WHERE id=?", (target_source_id,),
                ).fetchone()
                target_event_key = evidence_ledger.claim_assertion_event_key(
                    source["source_session_id"], source["source_message_id"],
                    source["evidence_kind"], source["revision"],
                )
            elif record["event_kind"] == "phase3_retraction":
                target_event_key = evidence_ledger.phase3_retraction_event_key(
                    conn, target_dependency_ids
                )
            elif record["event_kind"] == "value_supersession":
                if len(target_dependency_ids) != 1:
                    raise ValueError("portable value lifecycle lost its unique cause")
                target_event_key = evidence_ledger.value_supersession_event_key(
                    conn,
                    loser_edge_id=target_edge_id,
                    winner_evidence_id=target_dependency_ids[0],
                    event_at=record["event_at"],
                )
            elif record["event_kind"] == "manual_retraction":
                target_event_key = str(record["event_key"])
            else:
                target_event_key = str(record["event_key"])
            existing = conn.execute(
                "SELECT * FROM kg_edge_lifecycle WHERE edge_id=? AND event_key=?",
                (target_edge_id, target_event_key),
            ).fetchone()
            mapped = dict(record)
            mapped["edge_id"] = target_edge_id
            mapped["source_evidence_id"] = target_source_id
            mapped["event_key"] = target_event_key
            mapped["dependency_count"] = len(target_dependency_ids)
            compare = ["edge_id", "event_key", "event_kind", "direction", "event_at",
                       "source_evidence_id", "dependency_count", "details"]
            if existing is not None:
                if not _rows_match(existing, mapped, compare):
                    raise ValueError("portable lifecycle event collides with target")
                persisted_dependencies = {
                    int(item["evidence_id"])
                    for item in conn.execute(
                        "SELECT evidence_id FROM kg_lifecycle_dependencies "
                        "WHERE lifecycle_id=?", (existing["id"],),
                    ).fetchall()
                }
                if persisted_dependencies != set(target_dependency_ids):
                    raise ValueError("portable lifecycle dependencies collide with target")
                earliest_created_at = earliest_timestamp_spelling(
                    existing["created_at"], record["created_at"]
                )
                conn.execute(
                    "UPDATE kg_edge_lifecycle SET created_at=? WHERE id=?",
                    (earliest_created_at, existing["id"]),
                )
                target_lifecycle_id = int(existing["id"])
            else:
                values = [
                    target_edge_id, target_event_key, record["event_kind"],
                    record["direction"], record["event_at"], target_source_id,
                    len(target_dependency_ids), record["details"], record["created_at"],
                ]
                cur = conn.execute(
                    "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,direction,"
                    "event_at,source_evidence_id,dependency_count,details,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    values,
                )
                target_lifecycle_id = int(cur.lastrowid)
                lifecycle_inserted += 1
            lifecycle_id_map[int(record["id"])] = target_lifecycle_id
    inserted["edge_lifecycle"] = lifecycle_inserted

    dependency_inserted = 0
    with core_db.evidence_history_mutation(conn):
        for record in sorted(
            grouped.get("lifecycle_dependency", []),
            key=lambda row: (
                lifecycle_id_map[int(row["lifecycle_id"])],
                evidence_id_map[int(row["evidence_id"])],
            ),
        ):
            pair = (
                lifecycle_id_map[int(record["lifecycle_id"])],
                evidence_id_map[int(record["evidence_id"])],
            )
            existing = conn.execute(
                "SELECT 1 FROM kg_lifecycle_dependencies "
                "WHERE lifecycle_id=? AND evidence_id=?", pair,
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO kg_lifecycle_dependencies(lifecycle_id,evidence_id) "
                    "VALUES (?,?)", pair,
                )
                dependency_inserted += 1
    inserted["lifecycle_dependency"] = dependency_inserted

    affected = sorted(set(edge_ids.values()) | outcome_affected_edge_ids)
    # This is the same authority reducer used after a successful whole-chunk
    # replay. Besides counters/intervals it handles the legitimate terminal
    # state where all exact causes have been retired and no active transition
    # remains.
    evidence_ledger.finalize_chunk_extraction_reconciliation(conn, affected)


def _materialize_v6_graph_state(
    conn, grouped: dict[str, list[dict]], inserted: dict[str, int]
) -> None:
    """Convert v6 cached counters into conservative portable ledger rows."""
    edge_ids = _mapped_edge_ids(conn, grouped)
    affected = [
        edge_ids[int(record["id"])] for record in grouped.get("edge", [])
        if not record["derived"]
    ]
    evidence_ledger.capture_unattributed_counts(
        conn, affected, reason="v6 portable cached counter provenance unavailable"
    )
    from hymem.dreaming.bitemporal import normalized_event_at, record_lifecycle_event
    for record in grouped.get("edge", []):
        if record["derived"]:
            continue
        edge_id = edge_ids[int(record["id"])]
        details = "v6 portable lifecycle snapshot - exact transition provenance unavailable"
        if record["status"] == "active" and record["invalid_at"] is None:
            record_lifecycle_event(
                conn,
                edge_id=edge_id,
                event_key="portable-v6-legacy-state",
                event_kind="legacy_state",
                direction=1,
                event_at=normalized_event_at(
                    conn,
                    record["valid_at"] or record["first_seen"],
                    allow_legacy_unknown=True,
                ),
                details=details,
                recorded_at=_normalized_wire_event(
                    record["last_seen"] or record["first_seen"],
                    allow_legacy_unknown=True,
                ),
            )
        else:
            # A closed v6 cache contains both an open coordinate and a close
            # coordinate. Preserve both as conservative legacy snapshots so a
            # later reducer/v7 roundtrip cannot collapse the interval to
            # first_seen. Numeric key ranks make open precede close when the
            # instants are equal.
            record_lifecycle_event(
                conn,
                edge_id=edge_id,
                event_key="portable-v6-legacy-0-open",
                event_kind="legacy_state",
                direction=1,
                event_at=normalized_event_at(
                    conn,
                    record["valid_at"] or record["first_seen"],
                    allow_legacy_unknown=True,
                ),
                details=details,
                recorded_at=_normalized_wire_event(
                    record["last_seen"] or record["first_seen"],
                    allow_legacy_unknown=True,
                ),
            )
            record_lifecycle_event(
                conn,
                edge_id=edge_id,
                event_key="portable-v6-legacy-1-close",
                event_kind="legacy_state",
                direction=-1,
                event_at=normalized_event_at(
                    conn,
                    record["invalid_at"]
                    or record["last_seen"]
                    or record["first_seen"],
                    allow_legacy_unknown=True,
                ),
                details=details,
                recorded_at=_normalized_wire_event(
                    record["last_seen"] or record["first_seen"],
                    allow_legacy_unknown=True,
                ),
            )


def _normalize_v6_graph_materialization(grouped: dict[str, list[dict]]) -> None:
    """Project v6 cached edge intervals to their conservative local ledger.

    This is deliberately import-local; historical migrations keep their exact
    stored timestamp spellings. Applying the same projection before every v6
    target comparison makes semantically equivalent offset timestamps exactly
    idempotent after the first import's lifecycle reducer canonicalizes them.
    """
    for edge in grouped.get("edge", []):
        if int(edge["derived"]):
            continue
        if edge["status"] == "active" and edge["invalid_at"] is None:
            edge["valid_at"] = _normalized_wire_event(
                edge["valid_at"] or edge["first_seen"],
                allow_legacy_unknown=True,
            )
            edge["invalid_at"] = None
        else:
            edge["status"] = "retracted"
            valid_at = _normalized_wire_event(
                edge["valid_at"] or edge["first_seen"],
                allow_legacy_unknown=True,
            )
            invalid_at = _normalized_wire_event(
                edge["invalid_at"] or edge["last_seen"] or edge["first_seen"],
                allow_legacy_unknown=True,
            )
            edge["valid_at"] = valid_at
            edge["invalid_at"] = max(valid_at, invalid_at)


def export_jsonl(conn, path: str | Path) -> dict[str, int]:
    """Write the canonical state to `path` as JSON Lines. Returns per-kind
    row counts."""
    path = Path(path)
    # Export owns the read snapshot it opens below. Fail before creating a
    # temporary file when a caller already owns a transaction; attempting a
    # nested BEGIN and then rolling back `conn.in_transaction` would otherwise
    # discard the caller's unrelated uncommitted work.
    if conn.in_transaction:
        raise RuntimeError("cannot export inside a caller-owned transaction")
    counts: dict[str, int] = {}
    temp_file = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False,
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    temp_path = Path(temp_file.name)
    digest = hashlib.sha256()
    snapshot_started = False

    def write_hashed(obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n"
        temp_file.write(line)
        digest.update(line.encode("utf-8"))

    # One read transaction makes the JSONL a coherent snapshot. Without it a
    # concurrent dream could flip a publication marker between the session and
    # claim SELECTs, producing a cursor/state combination that never existed.
    try:
        conn.execute("BEGIN")
        snapshot_started = True
        grouped = _preflight_v7_export(conn)
        meta = {
            "type": "_meta",
            "format": "hymem-jsonl",
            "version": EXPORT_VERSION,
            "schema_version": core_db.schema_version(conn),
        }
        write_hashed(meta)
        for kind, _table, _cols in _EXPORT_SPEC:
            rows = grouped.get(kind, [])
            for source_record in rows:
                record = dict(source_record)
                if kind == "session":
                    staged = conn.execute(
                        "SELECT 1 FROM profile_staging WHERE session_id = ? LIMIT 1",
                        (record["id"],),
                    ).fetchone()
                    cursor_generation = record.get(
                        "profile_cursor_prompt_version"
                    )
                    published_generation = record.get(
                        "profile_published_generation"
                    )
                    user_tail = conn.execute(
                        "SELECT MAX(message_id) FROM message_retention_coverage "
                        "WHERE source_session_id = ? AND source_role = 'user' "
                        "AND coverage_version = 'dream-lossless-message-v1' "
                        "AND message_id <= ?",
                        (
                            record["id"],
                            int(record["coverage_message_id"])
                            if record.get("coverage_message_id") is not None
                            else -1,
                        ),
                    ).fetchone()[0]
                    cursor_id = record.get("profile_cursor_message_id")
                    cursor_semantically_complete = bool(
                        cursor_generation is None
                        and cursor_id is None
                    ) or bool(
                        profile_generation_is_recognized(cursor_generation)
                        and cursor_generation == published_generation
                        and (
                            (user_tail is None and cursor_id is None)
                            or (
                                user_tail is not None
                                and cursor_id is not None
                                and int(cursor_id) == int(user_tail)
                            )
                        )
                    )
                    incomplete = bool(
                        staged
                        or record.get("profile_cursor_partial_message_id")
                        is not None
                        or int(record.get("profile_cursor_offset") or 0) != 0
                        or not cursor_semantically_complete
                    )
                    if incomplete:
                        # Staging is intentionally not portable. Reset the
                        # input cursor so the destination replays exact
                        # coverage artifacts from the start while the last
                        # published claims remain visible.
                        record["profile_cursor_message_id"] = None
                        record["profile_cursor_partial_message_id"] = None
                        record["profile_cursor_offset"] = 0
                        record["profile_cursor_prompt_version"] = None
                        record["profile_retry_count"] = 0
                        record["profile_retry_config_version"] = None
                        record["profile_quarantined"] = 0
                write_hashed({"type": kind, "record": record})
            counts[kind] = len(rows)
        temp_file.write(json.dumps({
            "type": "_end",
            "counts": counts,
            "sha256": digest.hexdigest(),
        }, ensure_ascii=False, allow_nan=False) + "\n")
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
    except Exception:
        if snapshot_started and conn.in_transaction:
            conn.execute("ROLLBACK")
        temp_file.close()
        temp_path.unlink(missing_ok=True)
        raise
    else:
        if snapshot_started:
            conn.execute("COMMIT")
        try:
            os.replace(temp_path, path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    log.info("export.done path=%s counts=%s", path, counts)
    return counts


def import_jsonl(
    conn, path: str | Path, *, redact_values: bool = False, config=None
) -> dict[str, int]:
    """Load JSON Lines additively for disjoint, collision-free identities.

    Exact reimports are no-ops. Conflicting deterministic identities abort the
    complete import. Returns per-kind counts of rows actually inserted; the
    caller invalidates query-side caches afterwards.
    """
    if conn.in_transaction:
        raise RuntimeError("cannot import inside a caller-owned transaction")
    path = Path(path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    meta_version: int | None = None
    end_record: dict | None = None
    parsed_counts: dict[str, int] = defaultdict(int)
    unknown_kinds: list[str] = []
    wire_digest = hashlib.sha256()
    saw_end = False
    logical_index = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw_line in f:
            if not raw_line.strip():
                continue
            try:
                obj = loads_strict_json(raw_line)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("portable export contains invalid strict JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError("portable export line must be an object")
            kind = obj.get("type")
            if logical_index == 0 and kind != "_meta":
                raise ValueError("portable export header must be first")
            logical_index += 1
            if saw_end:
                raise ValueError("portable export has data after its end manifest")
            if kind == "_end":
                if meta_version is not None and meta_version >= 6 and set(obj) != {
                    "type", "counts", "sha256"
                }:
                    raise ValueError("portable end manifest has invalid fields")
                saw_end = True
                end_record = obj
                continue
            wire_digest.update(raw_line.encode("utf-8"))
            if kind == "_meta":
                if meta_version is not None:
                    raise ValueError("portable export contains multiple headers")
                if obj.get("format") != "hymem-jsonl":
                    raise ValueError("unrecognized portable export format")
                raw_version = obj.get("version")
                if (
                    isinstance(raw_version, bool)
                    or not isinstance(raw_version, int)
                    or raw_version <= 0
                    or raw_version > EXPORT_VERSION
                ):
                    raise ValueError("unsupported portable export version")
                meta_version = raw_version
                if meta_version >= 6:
                    if set(obj) != {"type", "format", "version", "schema_version"}:
                        raise ValueError("portable header has invalid fields")
                    if not _wire_int(obj.get("schema_version"), minimum=1):
                        raise ValueError("portable header has invalid schema version")
            elif kind in (
                _V10_TABLE_BY_KIND if (meta_version or 0) >= 10
                else _V9_TABLE_BY_KIND if (meta_version or 0) >= 9
                else _V8_TABLE_BY_KIND if (meta_version or 0) >= 8
                else _V7_TABLE_BY_KIND if (meta_version or 0) >= 7
                else _V6_TABLE_BY_KIND
            ):
                if meta_version is not None and meta_version >= 6 and set(obj) != {
                    "type", "record"
                }:
                    raise ValueError(f"portable {kind} envelope has invalid fields")
                record = obj.get("record")
                if not isinstance(record, dict):
                    raise ValueError(f"portable {kind} record must be an object")
                grouped[kind].append(record)
                parsed_counts[kind] += 1
            else:
                unknown_kinds.append(str(kind))
    if meta_version is None:
        raise ValueError("portable export is missing its header")
    if meta_version >= 6:
        version_tables = (
            _V10_TABLE_BY_KIND if meta_version >= 10
            else _V9_TABLE_BY_KIND if meta_version >= 9
            else _V8_TABLE_BY_KIND if meta_version >= 8
            else _V7_TABLE_BY_KIND if meta_version >= 7
            else _V6_TABLE_BY_KIND
        )
        version_columns = (
            _V10_COLS_BY_KIND if meta_version >= 10
            else _V9_COLS_BY_KIND if meta_version >= 9
            else _V8_COLS_BY_KIND if meta_version >= 8
            else _V7_COLS_BY_KIND if meta_version >= 7
            else _V6_COLS_BY_KIND
        )
        expected_counts = {kind: parsed_counts.get(kind, 0) for kind in version_tables}
        manifest_counts = end_record.get("counts") if end_record else None
        counts_valid = bool(
            isinstance(manifest_counts, dict)
            and set(manifest_counts) == set(version_tables)
            and all(_wire_int(value, minimum=0) for value in manifest_counts.values())
        )
        if (
            end_record is None
            or not counts_valid
            or manifest_counts != expected_counts
            or not isinstance(end_record.get("sha256"), str)
            or end_record.get("sha256") != wire_digest.hexdigest()
        ):
            raise ValueError("portable export is truncated or fails its manifest")
        if unknown_kinds:
            raise ValueError("portable export contains an unknown record kind")
        for kind, records in grouped.items():
            expected_columns = set(version_columns[kind])
            if any(set(record) != expected_columns for record in records):
                raise ValueError(
                    f"portable {kind} record does not match the v{meta_version} schema"
                )
        if meta_version < 8:
            _upgrade_pre_v8_peer_fields(grouped)
        if meta_version < 9:
            _upgrade_pre_v9_episode_fields(grouped)
        if meta_version < 10:
            _upgrade_pre_v10_fact_fields(grouped)
        if meta_version >= 7:
            interpretation_by_wire_id: dict[int, str] = {}
            for record in grouped.get("edge_evidence", []):
                record["interpretation_key"] = _portable_interpretation_key(record)
                interpretation_by_wire_id[int(record["id"])] = record[
                    "interpretation_key"
                ]
            for record in grouped.get("claim_observation", []):
                mapped = interpretation_by_wire_id.get(int(record["evidence_id"]))
                if mapped is not None:
                    record["interpretation_key"] = mapped
            for record in grouped.get("edge_evidence_signal", []):
                record["signal_key"] = _portable_signal_key(record)
        _validate_v6_record_scalars(grouped)
        # Reject poisoned source ids before validators traverse or index exact
        # coverage.  Besides preserving AUTOINCREMENT headroom, this makes the
        # cheap bounded scalar check precede any attacker-amplifiable work.
        for record in grouped.get("message_retention_coverage", []):
            source_mid = record.get("message_id")
            if (
                isinstance(source_mid, bool)
                or not isinstance(source_mid, int)
                or source_mid <= 0
                or source_mid > _MAX_SQLITE_ROWID - _ROWID_RESERVE_HEADROOM
            ):
                raise ValueError(
                    "portable coverage message_id leaves insufficient rowid headroom"
                )
        if meta_version >= 7:
            _validate_v7_records(grouped)
        if meta_version >= 9:
            _validate_v9_records(grouped)
        if meta_version >= 10:
            _validate_v10_fact_records(grouped)
        for record in grouped.get("session", []):
            if not digest_retry_state_is_valid(
                record.get("digest_retry_count"),
                record.get("digest_retry_config_version"),
                record.get("digest_quarantined"),
            ):
                raise ValueError("portable session has invalid digest retry state")
            if not profile_retry_state_is_valid(
                record.get("profile_retry_count"),
                record.get("profile_retry_config_version"),
                record.get("profile_quarantined"),
            ):
                raise ValueError("portable session has invalid profile retry state")
        if meta_version == 6:
            _normalize_v6_graph_materialization(grouped)

    if redact_values:
        # This must precede every destination mutation. In addition to keeping
        # durable storage safe, it prevents sqlite trace/error logs from ever
        # observing raw portable secrets. Pure in-memory structural checks,
        # such as the rowid-domain validation above, are safe to run first.
        _redact_portable_records(grouped)
        if meta_version >= 7:
            _validate_v7_records(grouped)
        if meta_version >= 9:
            _validate_v9_records(grouped)
        if meta_version >= 10:
            _validate_v10_fact_records(grouped)
    if meta_version >= 7:
        _preflight_v7_target_aliases(conn, grouped)
    fact_session_relations: dict[str, int] = {}

    inserted: dict[str, int] = {}
    newly_inserted_session_ids: set[str] = set()
    trusted_source_ids: set[int] = set()
    validated_by_session: dict[str, dict[int, object]] = defaultdict(dict)
    derived_cache_changed = False
    # v3 exports predate chunk purpose attribution.  Normalize any chunk that
    # backs a durable coverage proof before chunks are inserted; after the
    # proof lands the immutability trigger intentionally forbids relabeling.
    covered_chunk_ids = {
        record.get("chunk_id")
        for record in grouped.get("message_retention_coverage", [])
        if (
            record.get("chunk_id")
            and record.get("coverage_version") in LOSSLESS_READ_VERSIONS
        )
    }
    for record in grouped.get("chunk", []):
        if record.get("id") in covered_chunk_ids:
            record["chunk_kind"] = "coverage"
    for record in grouped.get("session", []):
        if record.get("summary") and not record.get("summary_source"):
            record["summary_source"] = "legacy"
    with core_db.transaction(conn):
        if meta_version >= 10:
            # Derive the fact-history relation only after BEGIN IMMEDIATE has
            # excluded a concurrent target advance. For a fresh session this
            # also preserves valid no-outcome retry/caught-up state; for equal
            # existing history, target-local operational state wins.
            fact_session_relations = _preflight_v10_fact_target_collisions(
                conn, grouped
            )
        if meta_version >= 6:
            if meta_version >= 7:
                # Derived rows are reproducible cache and are intentionally not
                # on the v7 wire. Remove the target cache first so a direct
                # imported fact can occupy the same natural triple, then rebuild
                # the closure from the merged direct graph below.
                imported_direct = {
                    _edge_natural(record) for record in grouped.get("edge", [])
                }
                derived_ids = [
                    int(row["id"])
                    for row in conn.execute(
                        "SELECT id,subject_canonical,predicate,object_canonical "
                        "FROM knowledge_graph WHERE derived=1 ORDER BY id"
                    ).fetchall()
                    if (
                        row["subject_canonical"], row["predicate"],
                        row["object_canonical"],
                    ) in imported_direct
                ]
                if derived_ids:
                    derived_cache_changed = True
                    with core_db.evidence_mutation(conn):
                        with contextlib.suppress(sqlite3.OperationalError):
                            conn.executemany(
                                "DELETE FROM vec_edges WHERE rowid=?",
                                [(edge_id,) for edge_id in derived_ids],
                            )
                        conn.executemany(
                            "DELETE FROM knowledge_graph WHERE id=?",
                            [(edge_id,) for edge_id in derived_ids],
                        )
            _preflight_v6_target_collisions(
                conn, grouped, merge_v7_edges=meta_version >= 7
            )
        for kind in _IMPORT_ORDER:
            table = _TABLE_BY_KIND[kind]
            drop_id = kind in _DROP_ID_ON_IMPORT
            n = 0
            for record in grouped.get(kind, []):
                if kind == "user_profile_fact":
                    # Raw messages are intentionally absent from portable
                    # memory. Always clear the compatibility FK; durable
                    # source_* fields remain authoritative. Deduplicate on claim +
                    # source identity so repeated imports are idempotent without
                    # trusting an autoincrement id from another store.
                    source_mid = record.get("source_message_id")
                    source_session = record.get("source_session_id")
                    if (
                        isinstance(source_mid, bool)
                        or not isinstance(source_mid, int)
                        or not isinstance(source_session, str)
                        or not source_session.strip()
                    ):
                        if meta_version >= 6:
                            raise ValueError(
                                "portable profile fact has invalid provenance"
                            )
                        log.warning("import.profile_skipped reason=missing_provenance")
                        continue
                    covered = covered_messages_after(
                        conn,
                        source_session,
                        int(source_mid) - 1,
                        limit=1,
                        roles=frozenset({"user"}),
                        through_message_id=int(source_mid),
                    )
                    if (
                        not covered
                        or covered[0].message_id != int(source_mid)
                        or (
                            (
                                meta_version >= 6
                                and record.get("source_created_at")
                                != covered[0].source_created_at
                            )
                            or (
                                meta_version < 6
                                and record.get("source_created_at") is not None
                                and record.get("source_created_at")
                                != covered[0].source_created_at
                            )
                        )
                    ):
                        if meta_version >= 6:
                            raise ValueError(
                                "portable profile fact has no validated source artifact"
                            )
                        log.warning(
                            "import.profile_skipped reason=invalid_coverage "
                            "session_id=%s message_id=%s",
                            source_session,
                            source_mid,
                        )
                        continue
                    candidate = {
                        "slot": record.get("slot"),
                        "value": record.get("value"),
                        "evidence_message_id": int(source_mid),
                        "confidence": record.get("confidence", 1.0),
                    }
                    if record.get("slot") == "relationship":
                        candidate["slot_key"] = record.get("slot_key")
                    clean = validate_profile_items(
                        [candidate], {int(source_mid)}, max_items=1
                    )
                    if len(clean) != 1:
                        if meta_version >= 6:
                            raise ValueError("portable profile fact is invalid")
                        log.warning("import.profile_skipped reason=invalid_claim")
                        continue
                    clean_item = {
                        **clean[0],
                        "source_message_id": int(source_mid),
                        "source_session_id": str(source_session),
                        "source_created_at": covered[0].source_created_at,
                    }
                    if record.get("invalid_at") is None:
                        # Active values must enter through the same chronology
                        # resolver as live extraction. Otherwise importing an
                        # older store can leave two active singleton values, or
                        # resurrect a stale value on re-import after it was
                        # superseded locally.
                        stored_key = (
                            _redact_profile_key(clean_item.get("slot_key"))
                            if redact_values else clean_item.get("slot_key")
                        )
                        stored_value = (
                            redaction.redact(clean_item["value"])
                            if redact_values else clean_item["value"]
                        )
                        added = persist_user_profile(
                            conn,
                            ProfileExtraction(items=[clean_item]),
                            redact_values=redact_values,
                        )
                        n += added
                        if added:
                            imported_valid_at = _interval_timestamp(
                                record.get("valid_at")
                            )
                            imported_created_at = _interval_timestamp(
                                record.get("created_at")
                            )
                            conn.execute(
                                "UPDATE user_profile SET "
                                "valid_at = COALESCE(?, valid_at), "
                                "created_at = COALESCE(?, created_at) "
                                "WHERE slot = ? AND slot_key IS ? "
                                "AND value = ? "
                                "AND source_message_id = ? "
                                "AND source_session_id = ?",
                                (
                                    imported_valid_at,
                                    imported_created_at,
                                    clean_item["slot"],
                                    stored_key,
                                    stored_value,
                                    int(source_mid),
                                    str(source_session),
                                ),
                            )
                            reconcile_profile_intervals(
                                conn,
                                clean_item["slot"],
                                stored_key,
                            )
                    else:
                        # Historical facts preserve their exported interval,
                        # but dedupe on immutable claim+source identity. The
                        # mutable invalid_at value must never be part of identity
                        # or an old export could resurrect a closed assertion.
                        if redact_values:
                            clean_item["value"] = redaction.redact(
                                clean_item["value"]
                            )
                            clean_item["slot_key"] = _redact_profile_key(
                                clean_item.get("slot_key")
                            )
                        identity_rows = conn.execute(
                            "SELECT id, value FROM user_profile "
                            "WHERE slot = ? AND slot_key IS ? "
                            "AND source_message_id IS ? "
                            "AND source_session_id IS ?",
                            (
                                clean_item["slot"],
                                clean_item.get("slot_key"),
                                int(source_mid),
                                str(source_session),
                            ),
                        ).fetchall()
                        value_key = " ".join(
                            clean_item["value"].casefold().split()
                        )
                        semantic_match = next(
                            (
                                row for row in identity_rows
                                if " ".join(row["value"].casefold().split())
                                == value_key
                            ),
                            None,
                        )
                        exclusive = (
                            clean_item["slot"] in SINGLE_VALUED_SLOTS
                            or clean_item["slot"] == "relationship"
                        )
                        if semantic_match is not None:
                            continue
                        if identity_rows and exclusive:
                            existing = min(
                                identity_rows,
                                key=lambda row: " ".join(
                                    row["value"].casefold().split()
                                ),
                            )
                            existing_key = " ".join(
                                existing["value"].casefold().split()
                            )
                            if value_key >= existing_key:
                                continue
                            conn.execute(
                                "UPDATE user_profile SET value = ?, "
                                "confidence = MAX(confidence, ?), valid_at = ?, "
                                "invalid_at = ?, created_at = ? WHERE id = ?",
                                (
                                    clean_item["value"], clean_item["confidence"],
                                    record.get("valid_at"),
                                    record.get("invalid_at"),
                                    record.get("created_at"), existing["id"],
                                ),
                            )
                            reconcile_profile_intervals(
                                conn, clean_item["slot"],
                                clean_item.get("slot_key"),
                            )
                            continue
                        cur = conn.execute(
                            """
                            INSERT INTO user_profile(
                                slot, slot_key, value, evidence_message_id,
                                confidence, valid_at, invalid_at, created_at,
                                source_message_id, source_session_id,
                                source_created_at
                            )
                            SELECT ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?
                            WHERE NOT EXISTS (
                                SELECT 1 FROM user_profile
                                WHERE slot = ? AND slot_key IS ? AND value = ?
                                  AND source_message_id IS ?
                                  AND source_session_id IS ?
                            )
                            """,
                            (
                                clean_item["slot"],
                                clean_item.get("slot_key"),
                                clean_item["value"],
                                clean_item["confidence"],
                                record.get("valid_at"),
                                record.get("invalid_at"),
                                record.get("created_at"),
                                int(source_mid),
                                str(source_session),
                                covered[0].source_created_at,
                                clean_item["slot"],
                                clean_item.get("slot_key"),
                                clean_item["value"],
                                int(source_mid),
                                str(source_session),
                            ),
                        )
                        n += cur.rowcount
                        if cur.rowcount:
                            reconcile_profile_intervals(
                                conn, clean_item["slot"],
                                clean_item.get("slot_key"),
                            )
                    continue
                unknown = set(record) - set(_COLS_BY_KIND[kind])
                if unknown:
                    log.warning(
                        "import.unknown_columns kind=%s columns=%s action=ignored",
                        kind,
                        ",".join(sorted(unknown)),
                    )
                # JSON keys are untrusted input. Interpolate identifiers only
                # from the static wire-format whitelist, never from the record.
                cols = [
                    c for c in _COLS_BY_KIND[kind]
                    if c in record and not (drop_id and c == "id")
                ]
                if not cols:
                    continue
                if meta_version >= 7 and kind == "edge":
                    existing_edge = conn.execute(
                        "SELECT * FROM knowledge_graph WHERE subject_canonical=? "
                        "AND predicate=? AND object_canonical=?",
                        _edge_natural(record),
                    ).fetchone()
                    if existing_edge is not None:
                        # Preserve any pre-v7/cache-only local contribution as
                        # explicitly unattributed before the imported ledger
                        # becomes the source of truth.
                        evidence_ledger.capture_unattributed_counts(
                            conn, [int(existing_edge["id"])],
                            reason="before v7 same-edge history merge",
                        )
                        conn.execute(
                            "UPDATE knowledge_graph SET first_seen=?,last_seen=?,"
                            "last_reinforced=? WHERE id=?",
                            (
                                _merged_timestamp(
                                    existing_edge["first_seen"], record["first_seen"],
                                    latest=False,
                                ),
                                _merged_timestamp(
                                    existing_edge["last_seen"], record["last_seen"],
                                    latest=True,
                                ),
                                _merged_timestamp(
                                    existing_edge["last_reinforced"],
                                    record["last_reinforced"], latest=True,
                                ),
                                existing_edge["id"],
                            ),
                        )
                        continue
                if meta_version >= 6 and _v6_existing_row_is_identical(
                    conn, kind, record
                ):
                    continue
                values_record = record
                if meta_version >= 10 and kind == "session":
                    # Fact cursors are published only after their complete
                    # outcome chain has landed and passed runtime validation.
                    values_record = dict(record)
                    values_record.update({
                        "facts_message_id": None,
                        "facts_cursor_message_id": None,
                        "facts_cursor_partial_message_id": None,
                        "facts_cursor_offset": 0,
                        "facts_cursor_prompt_version": None,
                        "facts_retry_count": 0,
                        "facts_retry_config_version": None,
                        "facts_quarantined": 0,
                    })
                if meta_version >= 9 and kind == "episode":
                    # Publication triggers require children to exist before a
                    # complete header. Stage the exact episode bytes as
                    # explicitly incomplete, insert proof rows later in the
                    # import order, then publish and revalidate below.
                    values_record = dict(record)
                    values_record.update({
                        "source_manifest_version": None,
                        "source_manifest_count": 0,
                        "source_manifest_hash": None,
                        "source_manifest_complete": 0,
                    })
                placeholders = ", ".join("?" * len(cols))
                insert_verb = "INSERT" if meta_version >= 6 else "INSERT OR IGNORE"
                cur = conn.execute(
                    f"{insert_verb} INTO {table}({', '.join(cols)}) "
                    f"VALUES ({placeholders})",
                    [values_record[c] for c in cols],
                )
                n += cur.rowcount
                if kind == "session" and cur.rowcount and record.get("id") is not None:
                    newly_inserted_session_ids.add(str(record["id"]))
            inserted[kind] = n
            if kind == "message_retention_coverage":
                # Validate every imported proof against its exact durable
                # canonical artifact before using its integer identity for a
                # cursor or AUTOINCREMENT reservation. Generic v37 proofs are
                # retained, but do not establish ordered-stream completeness.
                for record in grouped.get(kind, []):
                    try:
                        proof = validate_message_coverage_artifact(
                            conn,
                            message_id=record["message_id"],
                            chunk_id=record["chunk_id"],
                            coverage_version=record["coverage_version"],
                        )
                    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                        raise ValueError(
                            "portable coverage proof does not match its artifact"
                        ) from exc
                    trusted_source_ids.add(proof.message_id)
                    if record["coverage_version"] in LOSSLESS_READ_VERSIONS:
                        validated_by_session[proof.session_id][proof.message_id] = proof
                # A portable session's claimed producer frontier is untrusted.
                # Re-derive it from canonical, hash-checked, recognized
                # artifacts actually present in this import; otherwise a
                # forged future value can make all subsequently appended raw
                # messages look old and permanently suppress materialization.
                for imported_session_id in sorted(newly_inserted_session_ids):
                    validated_messages = validated_by_session.get(
                        imported_session_id, {}
                    )
                    validated_ids = sorted(validated_messages)
                    normalized_frontier = (
                        max(validated_ids) if validated_ids else None
                    )
                    conn.execute(
                        "UPDATE sessions SET coverage_message_id = ? WHERE id = ?",
                        (normalized_frontier, imported_session_id),
                    )
                    digest_state = conn.execute(
                        "SELECT digest_cursor_message_id, "
                        "digest_cursor_partial_message_id, digest_cursor_offset, "
                        "digest_cursor_prompt_version, digest_published_generation "
                        "FROM sessions WHERE id = ?",
                        (imported_session_id,),
                    ).fetchone()
                    digest_invalid = False
                    if digest_state:
                        cursor_mid = digest_state["digest_cursor_message_id"]
                        partial_mid = digest_state[
                            "digest_cursor_partial_message_id"
                        ]
                        offset = int(digest_state["digest_cursor_offset"] or 0)
                        generation = digest_state[
                            "digest_cursor_prompt_version"
                        ]
                        published = digest_state["digest_published_generation"]
                        cursor_id = int(cursor_mid) if cursor_mid is not None else None
                        partial_id = int(partial_mid) if partial_mid is not None else None
                        if cursor_id is not None and cursor_id not in validated_messages:
                            digest_invalid = True
                        if generation is not None and not digest_generation_is_recognized(
                            generation
                        ):
                            digest_invalid = True
                        if (cursor_id is not None or partial_id is not None or offset) and generation is None:
                            digest_invalid = True
                        if partial_id is None:
                            digest_invalid = digest_invalid or offset != 0
                        else:
                            partial = validated_messages.get(partial_id)
                            digest_invalid = digest_invalid or not (
                                partial is not None
                                and 0 < offset < len(partial.content)
                            )
                            ordered_ids = sorted(validated_messages)
                            next_index = (
                                ordered_ids.index(cursor_id) + 1
                                if cursor_id in ordered_ids else 0
                            )
                            digest_invalid = digest_invalid or (
                                next_index >= len(ordered_ids)
                                or ordered_ids[next_index] != partial_id
                            )
                        if (
                            generation is not None
                            and cursor_id == normalized_frontier
                            and partial_id is None
                            and generation != published
                        ):
                            digest_invalid = True
                    if digest_invalid:
                        conn.execute(
                            "UPDATE sessions SET digest_cursor_message_id = NULL, "
                            "digest_cursor_partial_message_id = NULL, "
                            "digest_cursor_offset = 0, "
                            "digest_cursor_prompt_version = NULL, "
                            "digest_retry_count = 0, "
                            "digest_retry_config_version = NULL, "
                            "digest_quarantined = 0 WHERE id = ?",
                            (imported_session_id,),
                        )
        if meta_version >= 10:
            _import_v10_fact_state(
                conn, grouped, inserted,
                session_relations=fact_session_relations,
            )
        # Coverage imports can retroactively introduce an eligible message
        # inside an already-published fact interval. Recheck the entire target
        # chain after all coverage/fact mutations, including for pre-v10
        # donors, so a missing-middle proof cannot launder a skipped source.
        for imported_session in grouped.get("session", []):
            _validated_target_fact_chain(conn, str(imported_session["id"]))
        if meta_version >= 9:
            for episode in grouped.get("episode", []):
                if episode["source_manifest_complete"] != 1:
                    continue
                conn.execute(
                    "UPDATE episodes SET source_manifest_version=?,"
                    "source_manifest_count=?,source_manifest_hash=?,"
                    "source_manifest_complete=1 WHERE id=?",
                    (
                        episode["source_manifest_version"],
                        episode["source_manifest_count"],
                        episode["source_manifest_hash"],
                        episode["id"],
                    ),
                )
                if load_episode_source_manifest(conn, episode["id"]) is None:
                    raise ValueError(
                        "portable episode source manifest failed runtime validation"
                    )
        if inserted.get("episode", 0) or inserted.get(
            "episode_source_occurrence", 0
        ):
            # Aggregation summaries/embeddings are reproducible caches over the
            # exact episode bytes and manifests. Never retain a tree built from
            # the pre-import episode set, even inside a long-lived process.
            conn.execute("DELETE FROM aggregation_nodes")
            conn.execute("DELETE FROM aggregation_leaf_state")
        if meta_version >= 7:
            _import_v7_claim_state(conn, grouped, inserted)
            from hymem.dreaming.inference import infer_transitive_edges

            policy_key = (
                "v1:retract_threshold=" + format(config.retract_threshold, ".17g")
                if config is not None else None
            )
            previous_policy = conn.execute(
                "SELECT value FROM schema_meta "
                "WHERE key='derived_inference_import_policy'"
            ).fetchone()
            direct_changed = derived_cache_changed or any(
                inserted.get(kind, 0)
                for kind in (
                    "edge", "edge_evidence", "edge_evidence_signal",
                    "claim_observation", "edge_lifecycle",
                    "lifecycle_dependency", "_claim_extraction_outcome_changed",
                )
            )
            if config is not None and (
                direct_changed
                or previous_policy is None
                or previous_policy["value"] != policy_key
            ):
                infer_transitive_edges(conn, config)
            elif config is None and direct_changed:
                # A caller using the lower-level function supplied no policy
                # with which to rebuild closure. Keeping the old closure after
                # direct history changed would knowingly expose stale facts,
                # so invalidate it completely and let the next configured
                # inference pass repopulate it.
                stale_ids = [
                    int(row["id"])
                    for row in conn.execute(
                        "SELECT id FROM knowledge_graph WHERE derived=1 ORDER BY id"
                    ).fetchall()
                ]
                if stale_ids:
                    with contextlib.suppress(sqlite3.OperationalError):
                        conn.executemany(
                            "DELETE FROM vec_edges WHERE rowid=?",
                            [(edge_id,) for edge_id in stale_ids],
                        )
                    with core_db.evidence_mutation(conn):
                        conn.executemany(
                            "DELETE FROM knowledge_graph WHERE id=?",
                            [(edge_id,) for edge_id in stale_ids],
                        )
                conn.execute(
                    "DELETE FROM schema_meta "
                    "WHERE key='derived_inference_import_policy'"
                )
        elif meta_version == 6:
            _materialize_v6_graph_state(conn, grouped, inserted)
        # Raw messages are intentionally not part of the portable-memory
        # format, but coverage artifacts retain their historical integer ids.
        # Reserve those ids so appending to an imported session cannot reuse an
        # id, collide with a deterministic msgcov_* artifact, or sit below the
        # imported coverage cursor and be skipped.
        max_source_id = max(trusted_source_ids) if trusted_source_ids else None
        if max_source_id is not None:
            cur = conn.execute(
                "UPDATE sqlite_sequence SET seq = MAX(seq, ?) WHERE name = 'messages'",
                (int(max_source_id),),
            )
            if not cur.rowcount:
                conn.execute(
                    "INSERT INTO sqlite_sequence(name, seq) VALUES ('messages', ?)",
                    (int(max_source_id),),
                )
        # Staged profile output is intentionally not imported. Reject any
        # externally supplied/inconsistent advanced cursor by rewinding it;
        # published claims/markers stay visible while the destination performs
        # a fresh artifact-backed walk.
        # Only sessions actually created by THIS import are normalized. A
        # same-id collision is additive merge, not permission to rewind an
        # unrelated local in-progress walk or discard its staging.
        for session_id in sorted(newly_inserted_session_ids):
            state = conn.execute(
                """
                SELECT profile_cursor_message_id,
                       profile_cursor_partial_message_id,
                       profile_cursor_offset, profile_cursor_prompt_version,
                       profile_published_generation, coverage_message_id
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if state is None:
                continue
            tail = conn.execute(
                """
                SELECT MAX(message_id) AS m
                FROM message_retention_coverage
                WHERE source_session_id = ?
                  AND source_role = 'user'
                  AND coverage_version = 'dream-lossless-message-v1'
                  AND message_id <= ?
                """,
                (
                    session_id,
                    int(state["coverage_message_id"])
                    if state["coverage_message_id"] is not None
                    else -1,
                ),
            ).fetchone()["m"]
            cursor_generation = state["profile_cursor_prompt_version"]
            incomplete = bool(
                state["profile_cursor_partial_message_id"] is not None
                or int(state["profile_cursor_offset"] or 0) != 0
                or (
                    cursor_generation is not None
                    and (
                        not profile_generation_is_recognized(cursor_generation)
                        or cursor_generation
                        != state["profile_published_generation"]
                        or state["profile_cursor_message_id"] is None
                        or (
                            (tail is None)
                            or int(state["profile_cursor_message_id"])
                            != int(tail)
                        )
                    )
                )
            )
            if incomplete:
                conn.execute(
                    """
                    UPDATE sessions
                    SET profile_cursor_message_id = NULL,
                        profile_cursor_partial_message_id = NULL,
                        profile_cursor_offset = 0,
                        profile_cursor_prompt_version = NULL,
                        profile_retry_count = 0,
                        profile_retry_config_version = NULL,
                        profile_quarantined = 0
                    WHERE id = ?
                    """,
                    (session_id,),
                )
        if redact_values:
            enforce_profile_redaction_policy(conn)
    inserted.pop("_claim_extraction_outcome_changed", None)
    log.info("import.done path=%s inserted=%s", path, inserted)
    return inserted
