"""Secret-free semantic attestation for persistent benchmark memory stores.

The LoCoMo/MSC cache boundary must bind the receipt to what SQLite *means*,
not to database-file bytes.  File hashes are sensitive to WAL checkpoints,
page allocation and VACUUM, while missing logical index rows can leave the
same source tables and a very different retrieval system.

This module therefore hashes a read-transaction snapshot as canonical row
sets.  Application tables are classified explicitly.  FTS and sqlite-vec
implementation tables are ignored only after their owning virtual table has
been recognized; their logical contents are attested through the virtual
table instead.  A future unknown application table fails closed until this
policy is deliberately revised.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import struct
from pathlib import Path
from typing import Iterable, Sequence


MATERIAL_STORE_ATTESTATION_VERSION = "hymem-material-store-attestation-v8"


class MaterialStoreAttestationError(RuntimeError):
    """A bounded structural attestation failure (never carries row values)."""

    def __init__(self, reason: str, *, tables: Iterable[str] = ()) -> None:
        self.reason = str(reason)
        normalized = sorted({_bounded_identifier(name) for name in tables})
        self.tables = tuple(normalized[:50])
        self.tables_truncated = len(normalized) > len(self.tables)
        super().__init__(self.reason)

    def safe_details(self) -> dict:
        return {
            "attestation_failure_reason": self.reason,
            "attestation_tables": list(self.tables),
            "attestation_tables_truncated": self.tables_truncated,
        }


# Tables whose rows constitute source, authoritative derived state, durable
# build state, or a read-side retrieval cache.  FTS and vec virtual tables are
# listed separately because they need logical (rather than shadow-row) reads.
_MATERIAL_TABLES = frozenset({
    "aggregation_generations",
    "aggregation_leaf_state",
    "aggregation_material_clock",
    "aggregation_material_epochs",
    "aggregation_node_embeddings",
    "aggregation_node_input_sources",
    "aggregation_node_inputs",
    "aggregation_node_source_occurrences",
    "aggregation_nodes",
    "aggregation_publication_state",
    "behavioral_markers",
    "chunk_embeddings",
    "chunk_extraction_terminal_losses",
    "chunk_message_sources",
    "chunks",
    "edge_embeddings",
    "embedding_cache",
    "entity_aliases",
    "entity_mentions",
    "entity_mention_observations",
    "entity_properties",
    "entity_property_observations",
    "entity_types",
    "entity_type_observations",
    "episode_embeddings",
    "episode_source_occurrences",
    "episodes",
    "fact_extraction_outcomes",
    "fact_extraction_revisions",
    "fact_extraction_source_occurrences",
    "kg_claim_extraction_outcomes",
    "kg_claim_observations",
    "kg_edge_lifecycle",
    "kg_evidence",
    "kg_evidence_extraction_audit",
    "kg_evidence_signals",
    "kg_lifecycle_dependencies",
    "knowledge_graph",
    "message_embeddings",
    "message_retention_coverage",
    "messages",
    "narrative_fact_embeddings",
    "narrative_fact_lifecycle",
    "narrative_facts",
    "peers",
    "phase1_generations",
    "phase1_auxiliary_outcomes",
    "procedures",
    "procedure_digest_publications",
    "processed_chunks",
    "profile_entries",
    "profile_entry_marker_evidence",
    "profile_marker_decisions",
    "profile_staging",
    "digest_staging",
    "rules",
    "rule_marker_decisions",
    "rule_marker_evidence",
    "schema_meta",
    "session_peers",
    "sessions",
    "temporal_mentions",
    "token_overlap_index",
    "user_profile",
})

_MATERIAL_VIEWS = frozenset({
    "aggregation_enabled_root_material",
    "aggregation_enabled_profile_sources",
    "aggregation_enabled_kg_sources",
    "aggregation_live_embedding_material",
    "aggregation_live_material",
    "aggregation_visible_episode_sources",
    "current_entity_mentions",
    "current_entity_properties",
    "current_entity_types",
    "current_phase1_publications",
    "current_profile_entries",
    "current_rules",
})

# These tables are deliberately outside material identity. They are bounded
# retry/health telemetry, lock/run history, or source-linked retraction audit.
# A no-op convergence may update/prune them without invalidating an otherwise
# identical store; none is read by extraction or retrieval.
_OPERATIONAL_TABLES = frozenset({
    "aggregation_build_health",
    "chunk_extraction_attempts",
    "coverage_integrity_failures",
    "dream_runs",
    "extraction_feedback",
    "run_lock",
})

_FTS_TABLES = frozenset({
    "aggregation_nodes_fts",
    "chunks_fts",
    "episodes_fts",
    "message_coverage_fts",
    "messages_fts",
    "narrative_facts_fts",
    "procedures_fts",
})

# sqlite-vec is optional, but ensure_vec_table creates this set atomically.
_VEC_TABLES = frozenset({
    "vec_chunks", "vec_edges", "vec_episodes", "vec_facts", "vec_messages",
})

# Columns with deliberately operational clocks/counters inside otherwise
# material tables.  All other present columns are included automatically, so a
# future column cannot silently escape the digest.  A schema-version change is
# separately bound in the build identity.
_IGNORED_COLUMNS: dict[str, frozenset[str]] = {
    "aggregation_generations": frozenset({"created_at"}),
    "aggregation_material_epochs": frozenset({"created_at"}),
    "aggregation_leaf_state": frozenset({"updated_at"}),
    "aggregation_node_embeddings": frozenset({"created_at"}),
    "aggregation_nodes": frozenset({"created_at"}),
    "chunk_embeddings": frozenset({"created_at"}),
    "chunk_extraction_terminal_losses": frozenset({"detected_at"}),
    "edge_embeddings": frozenset({"created_at"}),
    "embedding_cache": frozenset({"created_at"}),
    "episode_embeddings": frozenset({"created_at"}),
    "message_embeddings": frozenset({"created_at"}),
    "narrative_fact_embeddings": frozenset({"created_at"}),
    "phase1_generations": frozenset({"created_at"}),
    "phase1_auxiliary_outcomes": frozenset({"published_at"}),
    "entity_type_observations": frozenset({"observed_at"}),
    "entity_property_observations": frozenset({"observed_at"}),
    "entity_mention_observations": frozenset({"observed_at"}),
    "profile_entry_marker_evidence": frozenset({"created_at"}),
    "profile_marker_decisions": frozenset({"decided_at"}),
    "rule_marker_evidence": frozenset({"created_at"}),
    "rule_marker_decisions": frozenset({"decided_at"}),
    "processed_chunks": frozenset({"processed_at"}),
    "sessions": frozenset({
        "profile_retry_count", "profile_retry_config_version",
        "profile_quarantined", "facts_retry_count",
        "facts_retry_config_version", "facts_quarantined",
        "digest_retry_count", "digest_retry_config_version",
        "digest_quarantined",
    }),
}

_FTS_SHADOW_SUFFIXES = frozenset({
    "config", "content", "data", "docsize", "idx",
})
_VEC_SHADOW_RE = re.compile(
    r"^(?P<base>vec_(?:chunks|edges|episodes|facts|messages))_"
    r"(?:chunks|info|rowids|vector_chunks[0-9]+)$"
)


def _bounded_identifier(value: object) -> str:
    """Return a printable bounded table identifier, never arbitrary DB text."""

    text = str(value)[:96]
    return re.sub(r"[^A-Za-z0-9_.-]", "?", text)


def _quote_identifier(value: str) -> str:
    # Every caller passes a policy constant or sqlite_master identifier already
    # classified against a policy constant.  Quoting remains defence in depth.
    return '"' + value.replace('"', '""') + '"'


def _frame(digest, tag: bytes, payload: bytes) -> None:
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _value_bytes(value: object) -> tuple[bytes, bytes]:
    if value is None:
        return b"N", b""
    if isinstance(value, int):
        return b"I", str(value).encode("ascii")
    if isinstance(value, float):
        # IEEE bytes distinguish integer/real, signed zero, infinities and any
        # NaN payload SQLite/Python preserved without relying on JSON NaN.
        return b"F", struct.pack(">d", value)
    if isinstance(value, str):
        return b"T", value.encode("utf-8", errors="strict")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return b"B", bytes(value)
    raise TypeError(f"unsupported SQLite value type: {type(value).__name__}")


def _row_digest(columns: Sequence[str], values: Sequence[object]) -> bytes:
    if len(columns) != len(values):
        raise ValueError("row shape does not match its columns")
    digest = hashlib.sha256()
    _frame(digest, b"V", MATERIAL_STORE_ATTESTATION_VERSION.encode("ascii"))
    for column, value in zip(columns, values):
        _frame(digest, b"C", column.encode("utf-8"))
        tag, payload = _value_bytes(value)
        _frame(digest, tag, payload)
    return digest.digest()


def _set_digest(
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> tuple[int, str]:
    row_hashes = sorted(_row_digest(columns, tuple(row)) for row in rows)
    digest = hashlib.sha256()
    _frame(digest, b"V", MATERIAL_STORE_ATTESTATION_VERSION.encode("ascii"))
    _frame(digest, b"T", table.encode("ascii"))
    for row_hash in row_hashes:
        _frame(digest, b"R", row_hash)
    return len(row_hashes), "sha256:" + digest.hexdigest()


def _application_inventory(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return {str(row[0]): str(row[1] or "") for row in rows}


def _is_fts_shadow(name: str, inventory: dict[str, str]) -> bool:
    for base in _FTS_TABLES:
        prefix = f"{base}_"
        if name.startswith(prefix) and name[len(prefix):] in _FTS_SHADOW_SUFFIXES:
            return base in inventory and "USING fts5" in inventory[base]
    return False


def _is_vec_shadow(name: str, inventory: dict[str, str]) -> bool:
    match = _VEC_SHADOW_RE.fullmatch(name)
    if match is None:
        return False
    base = match.group("base")
    return base in inventory and "USING vec0" in inventory[base]


def _validate_inventory(inventory: dict[str, str]) -> frozenset[str]:
    names = set(inventory)
    missing = (_MATERIAL_TABLES | _FTS_TABLES | _OPERATIONAL_TABLES) - names
    if missing:
        raise MaterialStoreAttestationError(
            "missing_required_application_tables", tables=missing,
        )

    present_vec = names & _VEC_TABLES
    if present_vec and present_vec != _VEC_TABLES:
        raise MaterialStoreAttestationError(
            "incomplete_logical_vec_table_set", tables=_VEC_TABLES - present_vec,
        )

    known = _MATERIAL_TABLES | _FTS_TABLES | _OPERATIONAL_TABLES | _VEC_TABLES
    unknown = []
    for name in names - known:
        if name.startswith("sqlite_"):
            # SQLite reserves this namespace.  sqlite_sequence/stat tables are
            # allocator/planner implementation state, never application rows.
            continue
        if _is_fts_shadow(name, inventory) or _is_vec_shadow(name, inventory):
            continue
        unknown.append(name)
    if unknown:
        raise MaterialStoreAttestationError(
            "unknown_application_tables", tables=unknown,
        )

    for table in _FTS_TABLES:
        if "USING fts5" not in inventory[table]:
            raise MaterialStoreAttestationError(
                "invalid_logical_fts_table", tables=(table,),
            )
    for table in present_vec:
        if "USING vec0" not in inventory[table]:
            raise MaterialStoreAttestationError(
                "invalid_logical_vec_table", tables=(table,),
            )
    return frozenset(present_vec)


def _generic_table_state(conn: sqlite3.Connection, table: str) -> dict:
    info = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    present = [str(row[1]) for row in info]
    ignored = _IGNORED_COLUMNS.get(table, frozenset())
    missing_ignored = ignored - set(present)
    if missing_ignored:
        raise MaterialStoreAttestationError(
            "attestation_column_policy_mismatch", tables=(table,),
        )
    columns = [column for column in present if column not in ignored]
    if not columns:
        raise MaterialStoreAttestationError(
            "material_table_has_no_attested_columns", tables=(table,),
        )

    # Derived graph rows are regenerated on every dream.  Their surrogate id
    # and wall-clock defaults are not semantic; the unique triple and material
    # counters/status are.  Direct-edge temporal state remains fully attested.
    if table == "knowledge_graph":
        required = {
            "id", "subject_canonical", "predicate", "object_canonical",
            "pos_evidence", "neg_evidence", "first_seen", "last_seen",
            "last_reinforced", "valid_at", "invalid_at", "status", "derived",
        }
        if required - set(present):
            raise MaterialStoreAttestationError(
                "attestation_column_policy_mismatch", tables=(table,),
            )
        # Retain physical schema order and automatically bind future columns.
        # A future volatile field therefore fails reuse until it is consciously
        # classified; it cannot disappear from attestation by default.
        columns = present
        expressions = [
            _quote_identifier(column)
            if column not in {"id", "first_seen", "last_seen", "last_reinforced",
                              "valid_at", "invalid_at"}
            else (
                f"CASE WHEN derived=1 THEN NULL ELSE "
                f"{_quote_identifier(column)} END"
            )
            for column in columns
        ]
    else:
        expressions = [_quote_identifier(column) for column in columns]

    try:
        rows = conn.execute(
            f"SELECT {', '.join(expressions)} FROM {_quote_identifier(table)}"
        )
        count, digest = _set_digest(table, columns, rows)
    except MaterialStoreAttestationError:
        raise
    except Exception as exc:
        raise MaterialStoreAttestationError(
            "material_table_read_failed", tables=(table,),
        ) from exc
    return {"rows": count, "sha256": digest}


def _schema_state(
    conn: sqlite3.Connection,
    inventory: dict[str, str],
    present_vec: frozenset[str],
) -> dict:
    """Attest application DDL without hashing SQLite shadow implementations."""

    bases = _MATERIAL_TABLES | _FTS_TABLES | _OPERATIONAL_TABLES | present_vec
    rows: list[tuple[object, ...]] = []
    unknown: list[str] = []
    try:
        objects = conn.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE type IN ('table','index','trigger','view') "
            "ORDER BY type,name"
        ).fetchall()
        for row in objects:
            kind, name, owner, sql = map(lambda value: value or "", tuple(row))
            name = str(name)
            owner = str(owner)
            if kind == "table":
                if name in bases:
                    rows.append((kind, name, owner, sql))
                # Table shadows and sqlite internals were already classified by
                # _validate_inventory and deliberately do not enter DDL state.
                continue
            if kind == "view":
                if name in _MATERIAL_VIEWS:
                    rows.append((kind, name, owner, sql))
                else:
                    # Views carry retrieval authority. A future view must be
                    # consciously classified before benchmark reuse.
                    unknown.append(name)
                continue
            if name.startswith("sqlite_"):
                continue
            if owner in bases:
                rows.append((kind, name, owner, sql))
            elif (
                owner.startswith("sqlite_")
                or _is_fts_shadow(owner, inventory)
                or _is_vec_shadow(owner, inventory)
            ):
                continue
            else:
                unknown.append(name)
    except MaterialStoreAttestationError:
        raise
    except Exception as exc:
        raise MaterialStoreAttestationError("application_schema_read_failed") from exc
    if unknown:
        raise MaterialStoreAttestationError(
            "unknown_application_schema_objects", tables=unknown,
        )
    count, digest = _set_digest(
        "_schema", ("type", "name", "owner", "sql"), rows,
    )
    return {"rows": count, "sha256": digest}


_FTS_DOCUMENT_KEYS: dict[str, tuple[str, str]] = {
    "aggregation_nodes_fts": (
        "aggregation_nodes", "SELECT rowid, id FROM aggregation_nodes"
    ),
    "chunks_fts": ("chunks", "SELECT rowid, id FROM chunks"),
    "episodes_fts": ("episodes", "SELECT rowid, id FROM episodes"),
    "message_coverage_fts": ("chunks", "SELECT rowid, id FROM chunks"),
    "messages_fts": ("messages", "SELECT id, id FROM messages"),
    "narrative_facts_fts": (
        "narrative_facts",
        "SELECT id, session_id, fact_key, current_generation "
        "FROM narrative_facts",
    ),
    "procedures_fts": ("procedures", "SELECT rowid, id FROM procedures"),
}


def _logical_doc_keys(conn: sqlite3.Connection, table: str) -> dict[int, bytes]:
    _source, sql = _FTS_DOCUMENT_KEYS[table]
    keys: dict[int, bytes] = {}
    try:
        for row in conn.execute(sql):
            doc_id = int(row[0])
            logical_key = tuple(row[1:])
            keys[doc_id] = _row_digest(
                tuple(f"logical_key_{index}" for index in range(len(logical_key))),
                logical_key,
            )
    except Exception as exc:
        raise MaterialStoreAttestationError(
            "fts_document_mapping_failed", tables=(table,),
        ) from exc
    return keys


def _mapped_doc_key(keys: dict[int, bytes], doc_id: object) -> bytes:
    value = int(doc_id)
    if value in keys:
        return b"K" + keys[value]
    # Orphan/corrupt logical rows must still be represented deterministically.
    return b"O" + str(value).encode("ascii")


def _fts_table_state(conn: sqlite3.Connection, table: str, ordinal: int) -> dict:
    keys = _logical_doc_keys(conn, table)
    info = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    content_columns = [str(row[1]) for row in info]
    if not content_columns:
        raise MaterialStoreAttestationError(
            "logical_fts_table_has_no_columns", tables=(table,),
        )
    try:
        documents = (
            (_mapped_doc_key(keys, row[0]), *tuple(row[1:]))
            for row in conn.execute(
                f"SELECT rowid, {', '.join(_quote_identifier(c) for c in content_columns)} "
                f"FROM {_quote_identifier(table)}"
            )
        )
        document_count, document_digest = _set_digest(
            f"{table}:documents", ("logical_document", *content_columns), documents,
        )

        vocab = f"__hymem_attest_fts_{ordinal}"
        conn.execute(
            f"CREATE VIRTUAL TABLE temp.{_quote_identifier(vocab)} "
            f"USING fts5vocab(main, {_quote_identifier(table)}, instance)"
        )
        instances = (
            (row[0], _mapped_doc_key(keys, row[1]), row[2], row[3])
            for row in conn.execute(
                f"SELECT term, doc, col, offset FROM temp.{_quote_identifier(vocab)}"
            )
        )
        term_count, term_digest = _set_digest(
            f"{table}:terms", ("term", "logical_document", "column", "offset"),
            instances,
        )
    except Exception as exc:
        raise MaterialStoreAttestationError(
            "logical_fts_read_failed", tables=(table,),
        ) from exc
    combined = hashlib.sha256()
    _frame(combined, b"D", document_digest.encode("ascii"))
    _frame(combined, b"T", term_digest.encode("ascii"))
    return {
        "rows": document_count,
        "terms": term_count,
        "sha256": "sha256:" + combined.hexdigest(),
    }


_VEC_DOCUMENT_KEYS: dict[str, str] = {
    "vec_chunks": "SELECT rowid, id FROM chunks",
    "vec_edges": (
        "SELECT id, subject_canonical, predicate, object_canonical "
        "FROM knowledge_graph"
    ),
    "vec_episodes": "SELECT rowid, id FROM episodes",
    "vec_facts": (
        "SELECT id, session_id, fact_key, current_generation "
        "FROM narrative_facts"
    ),
    "vec_messages": (
        "SELECT message_id, source_coverage_chunk_id, text_hash "
        "FROM message_embeddings"
    ),
}


def _vec_table_state(conn: sqlite3.Connection, table: str) -> dict:
    keys: dict[int, bytes] = {}
    try:
        for row in conn.execute(_VEC_DOCUMENT_KEYS[table]):
            logical_key = tuple(row[1:])
            keys[int(row[0])] = _row_digest(
                tuple(f"logical_key_{index}" for index in range(len(logical_key))),
                logical_key,
            )
        rows = (
            (_mapped_doc_key(keys, row[0]), row[1])
            for row in conn.execute(
                f"SELECT rowid, embedding FROM {_quote_identifier(table)}"
            )
        )
        count, digest = _set_digest(
            table, ("logical_document", "embedding"), rows,
        )
    except Exception as exc:
        raise MaterialStoreAttestationError(
            "logical_vec_read_failed", tables=(table,),
        ) from exc
    return {"rows": count, "sha256": digest}


def _open_snapshot(path: Path) -> sqlite3.Connection:
    try:
        resolved = path.resolve(strict=True)
        conn = sqlite3.connect(
            resolved.as_uri() + "?mode=ro",
            uri=True,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        # v57's exact material-boundary validators depend on the same pure
        # connection-local UDFs as ordinary core readers.  Register them on
        # this read-only snapshot before checking any trigger/view/index
        # authority; a database file cannot carry UDF definitions itself.
        from hymem.core import db as core_db
        from hymem.dreaming.aggregation_generation import (
            aggregation_generation_registry_row_is_valid,
        )
        from hymem.dreaming.aggregation_material import (
            aggregation_material_registry_row_is_valid,
        )
        from hymem.extraction.producer import (
            phase1_generation_registry_row_is_valid,
        )

        core_db.register_read_authority_functions(conn)
        conn.create_function(
            "hymem_phase1_generation_registry_row_is_valid", 6,
            phase1_generation_registry_row_is_valid, deterministic=True,
        )
        conn.create_function(
            "hymem_aggregation_generation_registry_row_is_valid", 6,
            aggregation_generation_registry_row_is_valid, deterministic=True,
        )
        conn.create_function(
            "hymem_aggregation_material_registry_row_is_valid", 8,
            aggregation_material_registry_row_is_valid, deterministic=True,
        )
        # Loading the module is required to query logical vec0 tables.  It does
        # not write the main database.  Absence is acceptable only when the
        # inventory contains no vec0 table.
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pass
        conn.execute("BEGIN")
        return conn
    except Exception as exc:
        raise MaterialStoreAttestationError(
            "material_store_snapshot_open_failed",
        ) from exc


def material_store_state(path: str | Path) -> dict:
    """Return only version, logical row counts and digests for one DB snapshot."""

    conn = _open_snapshot(Path(path))
    try:
        from hymem.core import db as core_db

        version = core_db.schema_version(conn)
        if (
            version != core_db.EXPECTED_SCHEMA_VERSION
            or not core_db._v57_domain_present(conn)
            or not core_db._v57_material_bindings_present(conn)
        ):
            raise MaterialStoreAttestationError(
                "current_material_schema_boundary_is_malformed"
            )
        inventory = _application_inventory(conn)
        present_vec = _validate_inventory(inventory)
        tables: dict[str, dict] = {
            "_schema": _schema_state(conn, inventory, present_vec),
        }
        for table in sorted(_MATERIAL_TABLES):
            tables[table] = _generic_table_state(conn, table)
        for ordinal, table in enumerate(sorted(_FTS_TABLES)):
            tables[table] = _fts_table_state(conn, table, ordinal)
        for table in sorted(present_vec):
            tables[table] = _vec_table_state(conn, table)

        overall = hashlib.sha256()
        _frame(
            overall, b"V", MATERIAL_STORE_ATTESTATION_VERSION.encode("ascii")
        )
        for table, state in sorted(tables.items()):
            _frame(overall, b"T", table.encode("ascii"))
            _frame(overall, b"S", state["sha256"].encode("ascii"))
            for count_name in ("rows", "terms"):
                if count_name in state:
                    _frame(
                        overall, count_name[:1].upper().encode("ascii"),
                        str(state[count_name]).encode("ascii"),
                    )
        return {
            "version": MATERIAL_STORE_ATTESTATION_VERSION,
            "sha256": "sha256:" + overall.hexdigest(),
            "tables": tables,
        }
    except MaterialStoreAttestationError:
        raise
    except Exception as exc:
        raise MaterialStoreAttestationError(
            "material_store_snapshot_read_failed"
        ) from exc
    finally:
        try:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
        finally:
            conn.close()


def material_state_mismatch_tables(
    expected: object, actual: object
) -> tuple[list[str], bool]:
    """Return bounded table identifiers only; never values or row material."""

    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return ["<state>"], False
    expected_tables = expected.get("tables")
    actual_tables = actual.get("tables")
    if not isinstance(expected_tables, dict) or not isinstance(actual_tables, dict):
        return ["<tables>"], False
    mismatches = [
        _bounded_identifier(table)
        for table in sorted(set(expected_tables) | set(actual_tables), key=str)
        if expected_tables.get(table) != actual_tables.get(table)
    ]
    return mismatches[:50], len(mismatches) > 50
