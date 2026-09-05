from __future__ import annotations

import contextlib
import contextvars
import asyncio
import json
import logging
import math
import re
import sqlite3
import struct
import threading
from importlib.resources import files
from pathlib import Path
from typing import Iterator

from hymem.core.message_records import (
    encode_message_record,
    message_content_hash,
    message_record_matches_raw_source,
    message_record_matches_source,
    message_record_proof_valid,
)
from hymem.core.graph import live_edge_predicate
from hymem.core.time import (
    normalize_iso_timestamp,
    register_sqlite_time_functions,
    validate_event_clock,
)
from hymem.core.vectors import decode_vector

log = logging.getLogger("hymem.core.db")

EXPECTED_SCHEMA_VERSION = 46
_EVIDENCE_MUTATION_KEYS: contextvars.ContextVar[
    frozenset[tuple[int, int, int]]
] = contextvars.ContextVar("hymem_evidence_mutation_keys", default=frozenset())
_EVIDENCE_HISTORY_KEYS: contextvars.ContextVar[
    frozenset[tuple[int, int, int]]
] = contextvars.ContextVar("hymem_evidence_history_keys", default=frozenset())
_EVIDENCE_DESTRUCTIVE_KEYS: contextvars.ContextVar[
    frozenset[tuple[int, int, int]]
] = contextvars.ContextVar("hymem_evidence_destructive_keys", default=frozenset())


def _connection_authority_key(conn: sqlite3.Connection) -> tuple[int, int, int]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return id(conn), threading.get_ident(), id(task) if task is not None else 0


def _load_schema() -> str:
    return (files("hymem.core") / "schema.sql").read_text(encoding="utf-8")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _v43_domain_present(conn: sqlite3.Connection) -> bool:
    """Return whether the complete external-provenance domain exists.

    A few supported historical migration fixtures intentionally contain only
    the table under test.  They must still advance their version marker, but
    cannot install cross-table v43 constraints.  Real HyMem stores and fresh
    schema bootstraps contain this complete set.
    """
    return all(
        _table_exists(conn, table)
        for table in (
            "sessions", "messages", "chunks", "message_retention_coverage",
            "peers", "kg_evidence", "episodes", "procedures",
            "profile_staging", "temporal_mentions", "narrative_facts",
            "chunk_message_sources", "user_profile", "kg_claim_observations",
        )
    )


def connect(path: Path) -> sqlite3.Connection:
    from hymem.dreaming.lossless import coverage_chunk_id

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Used by the v37 coverage-ledger guard triggers. Keeping the hash function
    # in SQLite makes direct SQL obey the same source-lifecycle invariant as
    # the Python API: durable proof cannot be dropped after raw deletion.
    conn.create_function(
        "hymem_message_content_hash",
        2,
        message_content_hash,
        deterministic=True,
    )
    conn.create_function(
        "hymem_message_record_proof_valid",
        4,
        message_record_proof_valid,
        deterministic=True,
    )
    conn.create_function(
        "hymem_message_record_matches_source",
        10,
        message_record_matches_source,
        deterministic=True,
    )
    conn.create_function(
        "hymem_message_record_matches_raw_source",
        11,
        message_record_matches_raw_source,
        deterministic=True,
    )
    conn.create_function(
        "hymem_evidence_mutation_authorized",
        0,
        lambda: 1 if _connection_authority_key(conn) in _EVIDENCE_MUTATION_KEYS.get() else 0,
        deterministic=False,
    )
    conn.create_function(
        "hymem_evidence_history_authorized",
        0,
        lambda: 1 if _connection_authority_key(conn) in _EVIDENCE_HISTORY_KEYS.get() else 0,
        deterministic=False,
    )
    conn.create_function(
        "hymem_evidence_destructive_authorized",
        0,
        lambda: 1
        if _connection_authority_key(conn) in _EVIDENCE_DESTRUCTIVE_KEYS.get()
        else 0,
        deterministic=False,
    )
    conn.create_function(
        "hymem_message_record",
        3,
        lambda message_id, role, content: encode_message_record(
            message_id=int(message_id), role=str(role), content=str(content)
        ),
        deterministic=True,
    )
    conn.create_function(
        "hymem_coverage_chunk_id",
        2,
        lambda session_id, message_id: coverage_chunk_id(
            str(session_id), int(message_id)
        ),
        deterministic=True,
    )
    register_sqlite_time_functions(conn)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    # WAL is set here (not just in schema.sql) so it is active before any
    # schema creation or migration runs. journal_mode persists on the file;
    # synchronous is per-connection and must be set every time.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextlib.contextmanager
def evidence_mutation(conn: sqlite3.Connection) -> Iterator[None]:
    """Authorize one tightly scoped internal evidence/lifecycle rewrite."""
    key = _connection_authority_key(conn)
    token = _EVIDENCE_MUTATION_KEYS.set(_EVIDENCE_MUTATION_KEYS.get() | {key})
    try:
        yield
    finally:
        _EVIDENCE_MUTATION_KEYS.reset(token)


@contextlib.contextmanager
def evidence_history_mutation(conn: sqlite3.Connection) -> Iterator[None]:
    """Authorize a validated restore of immutable historical ledger rows.

    Normal runtime writers may only cite current evidence. Portability and
    complete-history merge paths additionally need to restore references to
    retired revisions without pretending those revisions are current. This
    narrower flag is meaningful only while ``evidence_mutation`` is also held.
    """
    key = _connection_authority_key(conn)
    token = _EVIDENCE_HISTORY_KEYS.set(_EVIDENCE_HISTORY_KEYS.get() | {key})
    try:
        with evidence_mutation(conn):
            yield
    finally:
        _EVIDENCE_HISTORY_KEYS.reset(token)


@contextlib.contextmanager
def evidence_destructive_mutation(conn: sqlite3.Connection) -> Iterator[None]:
    """Authorize explicit opt-in destruction of published graph history."""
    key = _connection_authority_key(conn)
    token = _EVIDENCE_DESTRUCTIVE_KEYS.set(
        _EVIDENCE_DESTRUCTIVE_KEYS.get() | {key}
    )
    try:
        with evidence_mutation(conn):
            yield
    finally:
        _EVIDENCE_DESTRUCTIVE_KEYS.reset(token)


def _load_vec_extension(conn: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except ImportError:
        return False
    except Exception as exc:
        log.info("sqlite-vec failed to load (%s); using Python cosine search", exc)
        return False


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(_load_schema())
    _load_vec_extension(conn)
    cur = schema_version(conn)
    if cur > EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {cur} is newer than code expects ({EXPECTED_SCHEMA_VERSION}). "
            f"Downgrading is not supported. Use a newer version of HyMem."
        )
    _run_migrations(conn)


def _install_evidence_revision_guards(conn: sqlite3.Connection) -> None:
    """Install the canonical v40 evidence guards from their owned migration.

    Schema v42 deliberately refreshes these two long provenance guards because
    their original timestamp comparison used SQLite's broader grammar. Keeping
    one source definition avoids a second large trigger copy drifting again.
    """
    script = files("hymem.core.migrations").joinpath(
        "040_claim_source_provenance.sql"
    ).read_text(encoding="utf-8")
    for name in (
        "kg_evidence_v40_insert_guard",
        "kg_evidence_v40_update_guard",
    ):
        marker = f"CREATE TRIGGER IF NOT EXISTS {name}"
        start = script.index(marker)
        end = script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(script[start:end])


def _install_evidence_publication_guards(conn: sqlite3.Connection) -> None:
    """Install/heal schema-42 publication guards from one SQL definition."""
    script = files("hymem.core.migrations").joinpath(
        "042_evidence_publication_clock.sql"
    ).read_text(encoding="utf-8")
    for name in (
        "kg_evidence_published_at_insert_guard",
        "kg_evidence_published_at_update_guard",
        "kg_evidence_v40_delete_guard",
        "kg_edge_lifecycle_update_guard",
        "kg_edge_lifecycle_delete_guard",
        "kg_lifecycle_dependencies_update_guard",
        "kg_lifecycle_dependencies_delete_guard",
    ):
        marker = f"CREATE TRIGGER {name}"
        start = script.index(marker)
        end = script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(script[start:end])


def _install_aggregation_source_guards(conn: sqlite3.Connection) -> None:
    """Install/heal v45's source-publication boundary from its owned SQL."""

    script = files("hymem.core.migrations").joinpath(
        "045_aggregation_source_provenance.sql"
    ).read_text(encoding="utf-8")
    names = (
        "episode_source_header_insert_guard",
        "episode_source_header_update_guard",
        "episode_source_bound_update_guard",
        "episode_source_occurrence_insert_guard",
        "episode_source_occurrence_update_guard",
        "episode_source_occurrence_delete_unpublishes",
        "aggregation_source_header_insert_guard",
        "aggregation_source_header_update_guard",
        "aggregation_source_bound_update_guard",
        "aggregation_source_occurrence_insert_guard",
        "aggregation_source_occurrence_update_guard",
        "aggregation_source_occurrence_delete_unpublishes",
    )
    for name in names:
        marker = f"CREATE TRIGGER IF NOT EXISTS {name}"
        start = script.index(marker)
        end = script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(script[start:end])


def _install_fact_authority_guards(conn: sqlite3.Connection) -> None:
    """Install/heal v46's authoritative fact publication guards."""

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fact_outcome_before_cursor ON "
        "fact_extraction_outcomes(session_id,cursor_before_message_id,"
        "cursor_before_partial_message_id,cursor_before_offset)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fact_outcome_after_cursor ON "
        "fact_extraction_outcomes(session_id,cursor_after_message_id,"
        "cursor_after_partial_message_id,cursor_after_offset)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fact_outcome_chain_order ON "
        "fact_extraction_outcomes(session_id,"
        "COALESCE(cursor_before_partial_message_id,cursor_before_message_id,-1),"
        "CASE WHEN cursor_before_partial_message_id IS NULL THEN 1 ELSE 0 END,"
        "cursor_before_offset,slice_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fact_outcome_replay_v46 ON "
        "fact_extraction_outcomes(session_id,source_manifest_complete,"
        "prompt_version)"
    )

    script = files("hymem.core.migrations").joinpath(
        "046_narrative_fact_authority.sql"
    ).read_text(encoding="utf-8")
    names = (
        "session_workspace_binding_guard",
        "fact_outcome_insert_guard",
        "fact_outcome_header_guard",
        "fact_outcome_bound_guard",
        "fact_outcome_result_guard",
        "fact_outcome_delete_guard",
        "fact_revision_insert_guard",
        "fact_revision_update_guard",
        "fact_revision_delete_guard",
        "fact_source_occurrence_insert_guard",
        "fact_source_occurrence_update_guard",
        "fact_source_occurrence_delete_guard",
        "narrative_fact_authority_insert_guard",
        "narrative_fact_authority_update_guard",
        "narrative_fact_bound_update_guard",
        "narrative_fact_lifecycle_projection_guard",
        "narrative_fact_delete_guard",
        "narrative_fact_lifecycle_insert_guard",
        "narrative_fact_lifecycle_update_guard",
        "narrative_fact_lifecycle_delete_guard",
    )
    for name in names:
        marker = f"CREATE TRIGGER IF NOT EXISTS {name}"
        start = script.index(marker)
        end = script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        _apply_migration_sql(conn, script[start:end])


def _install_external_peer_guards(conn: sqlite3.Connection) -> None:
    """Heal schema-43 external author/provenance guards on every startup."""
    script = files("hymem.core.migrations").joinpath(
        "043_external_peer_provenance.sql"
    ).read_text(encoding="utf-8")
    for name in (
        "message_retention_coverage_delete_guard",
        "message_retention_coverage_update_guard",
        "session_workspace_binding_guard",
        "session_peer_binding_insert_guard",
        "session_peer_binding_update_guard",
        "session_peer_delete_guard",
        "peer_identity_update_guard",
        "peer_identity_delete_guard",
        "message_external_provenance_insert_guard",
        "message_external_provenance_update_guard",
        "message_lossless_source_update_guard",
        "message_coverage_peer_insert_guard",
        "message_coverage_peer_update_guard",
        "kg_evidence_v43_peer_insert_guard",
        "kg_evidence_v43_peer_update_guard",
    ):
        marker = f"CREATE TRIGGER {name}"
        start = script.index(marker)
        end = script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(script[start:end])


def _ensure_message_coverage_fts(conn: sqlite3.Connection) -> None:
    """Heal and exactly rebuild the durable-message search shadow."""
    definition = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' "
        "AND name='message_coverage_fts'"
    ).fetchone()
    definition_sql = "" if definition is None else "".join(
        str(definition["sql"] or "").lower().split()
    )
    columns = [
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(message_coverage_fts)"
        ).fetchall()
    ] if definition is not None else []
    valid_shape = bool(
        columns == ["content"]
        and "usingfts5(content,content='',tokenize='porterunicode61')"
        in definition_sql
    )
    if definition is not None and not valid_shape:
        for name in (
            "message_coverage_fts_insert",
            "message_coverage_fts_delete",
            "message_coverage_fts_update_delete",
            "message_coverage_fts_update_insert",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute("DROP TABLE message_coverage_fts")
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS message_coverage_fts USING fts5(
            content,
            content='',
            tokenize='porter unicode61'
        );
        DROP TRIGGER IF EXISTS message_coverage_fts_insert;
        DROP TRIGGER IF EXISTS message_coverage_fts_delete;
        DROP TRIGGER IF EXISTS message_coverage_fts_update_delete;
        DROP TRIGGER IF EXISTS message_coverage_fts_update_insert;
        INSERT INTO message_coverage_fts(message_coverage_fts)
        VALUES('delete-all');
        INSERT INTO message_coverage_fts(rowid, content)
        SELECT rowid, json_extract(text, '$.content') FROM chunks
        WHERE chunk_kind = 'coverage'
          AND json_valid(text)
          AND json_type(text, '$.content') = 'text';
        CREATE TRIGGER message_coverage_fts_insert
        AFTER INSERT ON chunks
        WHEN new.chunk_kind = 'coverage'
         AND json_valid(new.text)
         AND json_type(new.text, '$.content') = 'text' BEGIN
            INSERT INTO message_coverage_fts(rowid, content)
            VALUES (new.rowid, json_extract(new.text, '$.content'));
        END;
        CREATE TRIGGER message_coverage_fts_delete
        AFTER DELETE ON chunks
        WHEN old.chunk_kind = 'coverage'
         AND json_valid(old.text)
         AND json_type(old.text, '$.content') = 'text' BEGIN
            INSERT INTO message_coverage_fts(
                message_coverage_fts, rowid, content
            ) VALUES (
                'delete', old.rowid, json_extract(old.text, '$.content')
            );
        END;
        CREATE TRIGGER message_coverage_fts_update_delete
        AFTER UPDATE OF text, chunk_kind ON chunks
        WHEN old.chunk_kind = 'coverage'
         AND json_valid(old.text)
         AND json_type(old.text, '$.content') = 'text' BEGIN
            INSERT INTO message_coverage_fts(
                message_coverage_fts, rowid, content
            ) VALUES (
                'delete', old.rowid, json_extract(old.text, '$.content')
            );
        END;
        CREATE TRIGGER message_coverage_fts_update_insert
        AFTER UPDATE OF text, chunk_kind ON chunks
        WHEN new.chunk_kind = 'coverage'
         AND json_valid(new.text)
         AND json_type(new.text, '$.content') = 'text' BEGIN
            INSERT INTO message_coverage_fts(rowid, content)
            VALUES (new.rowid, json_extract(new.text, '$.content'));
        END;
        """
    )


def _ensure_narrative_facts_fts(conn: sqlite3.Connection) -> None:
    """Heal and exactly rebuild the authoritative-current fact shadow.

    FTS5's external-content ``rebuild`` command mirrors every row in the
    content table.  That is deliberately wrong for narrative facts: legacy
    projections have no source proof and retracted projections are historical
    state.  Even if downstream joins reject them, indexing those documents
    changes BM25 corpus statistics.  Rebuild explicitly from the authoritative
    active subset and reinstall transition-aware triggers on every open.
    """

    definition = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' "
        "AND name='narrative_facts_fts'"
    ).fetchone()
    definition_sql = "" if definition is None else "".join(
        str(definition["sql"] or "").lower().split()
    )
    columns = [
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(narrative_facts_fts)"
        ).fetchall()
    ] if definition is not None else []
    valid_shape = bool(
        columns == ["text"]
        and (
            "usingfts5(text,content='narrative_facts',"
            "content_rowid='id',tokenize='porterunicode61')"
        ) in definition_sql
    )

    owner = not conn.in_transaction
    if owner:
        conn.execute("BEGIN IMMEDIATE")
    try:
        for name in (
            "narrative_facts_fts_insert",
            "narrative_facts_fts_delete",
            "narrative_facts_fts_update",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        if definition is not None and not valid_shape:
            conn.execute("DROP TABLE narrative_facts_fts")
        _apply_migration_sql(
            conn,
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS narrative_facts_fts USING fts5(
                text,
                content='narrative_facts',
                content_rowid='id',
                tokenize='porter unicode61'
            );
            INSERT INTO narrative_facts_fts(narrative_facts_fts)
            VALUES('delete-all');
            INSERT INTO narrative_facts_fts(rowid, text)
            SELECT id, text
            FROM narrative_facts
            WHERE source_outcome_key IS NOT NULL
              AND lifecycle_status = 'active'
              AND invalid_at IS NULL;
            CREATE TRIGGER narrative_facts_fts_insert
            AFTER INSERT ON narrative_facts
            WHEN new.source_outcome_key IS NOT NULL
             AND new.lifecycle_status = 'active'
             AND new.invalid_at IS NULL BEGIN
                INSERT INTO narrative_facts_fts(rowid, text)
                VALUES (new.id, new.text);
            END;
            CREATE TRIGGER narrative_facts_fts_delete
            AFTER DELETE ON narrative_facts
            WHEN old.source_outcome_key IS NOT NULL
             AND old.lifecycle_status = 'active'
             AND old.invalid_at IS NULL BEGIN
                INSERT INTO narrative_facts_fts(
                    narrative_facts_fts, rowid, text
                ) VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER narrative_facts_fts_update
            AFTER UPDATE OF text, source_outcome_key, lifecycle_status,
                            invalid_at
            ON narrative_facts BEGIN
                INSERT INTO narrative_facts_fts(
                    narrative_facts_fts, rowid, text
                )
                SELECT 'delete', old.id, old.text
                WHERE old.source_outcome_key IS NOT NULL
                  AND old.lifecycle_status = 'active'
                  AND old.invalid_at IS NULL;
                INSERT INTO narrative_facts_fts(rowid, text)
                SELECT new.id, new.text
                WHERE new.source_outcome_key IS NOT NULL
                  AND new.lifecycle_status = 'active'
                  AND new.invalid_at IS NULL;
            END;
            """,
        )
    except Exception:
        if owner:
            conn.execute("ROLLBACK")
        raise
    else:
        if owner:
            conn.execute("COMMIT")


def _ensure_post_migration_runtime_guards(conn: sqlite3.Connection) -> None:
    """Heal latest triggers only after their owning columns/tables exist."""
    tables = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        ).fetchall()
    }
    chunk_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
    } if "chunks" in tables else set()
    if (
        schema_version(conn) >= 38
        and {"chunks", "chunks_fts"}.issubset(tables)
        and {"chunk_kind", "text"}.issubset(chunk_columns)
    ):
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks
            WHEN new.chunk_kind = 'extraction' BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks
            WHEN old.chunk_kind = 'extraction' BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.rowid, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update_delete
            AFTER UPDATE OF text, chunk_kind ON chunks
            WHEN old.chunk_kind = 'extraction' BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.rowid, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update_insert
            AFTER UPDATE OF text, chunk_kind ON chunks
            WHEN new.chunk_kind = 'extraction' BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            """
        )
    if (
        schema_version(conn) >= 40
        and {
            "chunks", "chunk_message_sources", "kg_evidence",
            "kg_evidence_signals", "kg_claim_observations",
            "kg_edge_lifecycle", "kg_lifecycle_dependencies",
            "knowledge_graph",
        }.issubset(tables)
    ):
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS kg_evidence_signals_v40_insert_guard;
            DROP TRIGGER IF EXISTS kg_evidence_signals_v40_update_guard;
            DROP TRIGGER IF EXISTS kg_evidence_signals_v40_delete_guard;
            DROP TRIGGER IF EXISTS kg_claim_observations_insert_guard;
            DROP TRIGGER IF EXISTS kg_edge_lifecycle_insert_guard;
            DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_insert_guard;
            CREATE TRIGGER IF NOT EXISTS kg_evidence_signals_v40_insert_guard
            BEFORE INSERT ON kg_evidence_signals
            WHEN hymem_evidence_mutation_authorized() <> 1
              OR (new.signal_kind = 'manual_retraction' AND new.polarity <> -1) BEGIN
                SELECT RAISE(ABORT, 'kg evidence signals are internally managed');
            END;
            CREATE TRIGGER IF NOT EXISTS kg_evidence_signals_v40_update_guard
            BEFORE UPDATE ON kg_evidence_signals
            WHEN hymem_evidence_mutation_authorized() <> 1
              OR (new.signal_kind = 'manual_retraction' AND new.polarity <> -1) BEGIN
                SELECT RAISE(ABORT, 'kg evidence signals are internally managed');
            END;
            CREATE TRIGGER IF NOT EXISTS kg_evidence_signals_v40_delete_guard
            BEFORE DELETE ON kg_evidence_signals
            WHEN hymem_evidence_mutation_authorized() <> 1 BEGIN
                SELECT RAISE(ABORT, 'kg evidence signals are internally managed');
            END;
            CREATE TRIGGER IF NOT EXISTS kg_claim_observations_insert_guard
            BEFORE INSERT ON kg_claim_observations
            BEGIN
                SELECT RAISE(ABORT, 'claim observations are internally managed')
                WHERE hymem_evidence_mutation_authorized() <> 1;
                SELECT RAISE(ABORT, 'claim observation lacks canonical evidence')
                WHERE NOT (
                  EXISTS (
                    SELECT 1 FROM chunk_message_sources cms
                    JOIN chunks c ON c.id = cms.chunk_id
                    WHERE cms.chunk_id = new.chunk_id
                      AND cms.source_session_id = new.source_session_id
                      AND cms.source_message_id = new.source_message_id
                      AND c.source_manifest_version = 'claim-source-manifest-v1'
                  )
                  AND EXISTS (
                    SELECT 1 FROM kg_evidence ev
                    WHERE ev.id = new.evidence_id
                      AND ev.edge_id = new.edge_id
                      AND ev.source_session_id = new.source_session_id
                      AND ev.source_message_id = new.source_message_id
                      AND ev.evidence_kind = new.evidence_kind
                      AND ev.polarity = new.polarity
                      AND ev.interpretation_key = new.interpretation_key
                      AND ev.provenance_status = 'canonical'
                      AND (ev.is_current = 1
                           OR hymem_evidence_history_authorized() = 1)
                      AND ev.revision > 0
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM kg_claim_observations existing
                    WHERE existing.edge_id = new.edge_id
                      AND existing.source_session_id = new.source_session_id
                      AND existing.source_message_id = new.source_message_id
                      AND existing.evidence_kind = new.evidence_kind
                      AND existing.prompt_generation = new.prompt_generation
                      AND (existing.polarity <> new.polarity
                           OR existing.interpretation_key <> new.interpretation_key)
                  )
                );
            END;
            CREATE TRIGGER IF NOT EXISTS kg_edge_lifecycle_insert_guard
            BEFORE INSERT ON kg_edge_lifecycle
            BEGIN
                SELECT RAISE(ABORT, 'knowledge graph lifecycle events are internally managed')
                WHERE hymem_evidence_mutation_authorized() <> 1;
                SELECT RAISE(ABORT, 'invalid knowledge graph lifecycle event')
                WHERE NOT (
                  new.event_at = COALESCE(
                    hymem_normalize_iso_timestamp(new.event_at),
                    '0001-01-01T00:00:00.000Z'
                  )
                  AND (
                    (new.event_kind = 'claim_assertion' AND new.direction = 1
                     AND new.source_evidence_id IS NOT NULL
                     AND new.dependency_count = 0)
                    OR (new.event_kind = 'manual_retraction'
                        AND new.direction = -1 AND new.dependency_count = 0)
                    OR (new.event_kind = 'value_supersession'
                        AND new.direction = -1 AND new.dependency_count > 0
                        AND new.source_evidence_id IS NULL)
                    OR (new.event_kind = 'phase3_retraction'
                        AND new.direction = -1 AND new.dependency_count > 0
                        AND new.source_evidence_id IS NULL)
                    OR (new.event_kind = 'legacy_state'
                        AND new.source_evidence_id IS NULL
                        AND new.dependency_count = 0)
                  )
                  AND (new.source_evidence_id IS NULL OR EXISTS (
                    SELECT 1 FROM kg_evidence ev
                    WHERE ev.id = new.source_evidence_id
                      AND ev.edge_id = new.edge_id
                      AND ev.provenance_status = 'canonical'
                      AND (ev.is_current = 1
                           OR hymem_evidence_history_authorized() = 1)
                      AND ev.polarity = new.direction
                      AND ev.source_event_at = new.event_at
                  ))
                );
            END;
            CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_insert_guard
            BEFORE INSERT ON kg_lifecycle_dependencies
            BEGIN
                SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed')
                WHERE hymem_evidence_mutation_authorized() <> 1;
                SELECT RAISE(ABORT, 'invalid lifecycle evidence dependency')
                WHERE NOT EXISTS (
                  SELECT 1
                  FROM kg_edge_lifecycle lifecycle
                  JOIN kg_evidence ev ON ev.id = new.evidence_id
                  WHERE lifecycle.id = new.lifecycle_id
                    AND lifecycle.direction = -1
                    AND (ev.is_current = 1
                         OR hymem_evidence_history_authorized() = 1)
                    AND (
                      (lifecycle.event_kind = 'phase3_retraction'
                       AND lifecycle.edge_id = ev.edge_id AND ev.polarity = -1)
                      OR (lifecycle.event_kind = 'value_supersession'
                          AND ev.polarity = 1
                          AND EXISTS (
                            SELECT 1
                            FROM knowledge_graph loser
                            JOIN knowledge_graph winner
                              ON winner.subject_canonical = loser.subject_canonical
                             AND winner.predicate = loser.predicate
                             AND winner.object_canonical <> loser.object_canonical
                            WHERE loser.id = lifecycle.edge_id
                              AND winner.id = ev.edge_id
                              AND winner.derived = 0
                          ))
                    )
                );
            END;
            CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_update_guard
            BEFORE UPDATE ON kg_lifecycle_dependencies
            WHEN hymem_evidence_mutation_authorized() <> 1 BEGIN
                SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed');
            END;
            CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_delete_guard
            BEFORE DELETE ON kg_lifecycle_dependencies
            WHEN hymem_evidence_mutation_authorized() <> 1 BEGIN
                SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed');
            END;
            """
        )
    if (
        schema_version(conn) >= 41
        and {
            "chunks", "kg_evidence", "kg_claim_observations",
            "kg_claim_extraction_outcomes",
        }.issubset(tables)
        and {"source_manifest_version", "source_manifest_count"}.issubset(
            chunk_columns
        )
    ):
        # ``schema.sql`` deliberately uses IF NOT EXISTS so it remains safe as
        # a pre-migration bootstrap against old stores.  Once v41 is stamped,
        # replace the durable publication guards unconditionally: this heals a
        # process that created an earlier same-named trigger definition before
        # the final migration contract was installed.
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS kg_claim_extraction_outcomes_insert_guard;
            DROP TRIGGER IF EXISTS kg_claim_extraction_outcomes_update_guard;
            DROP TRIGGER IF EXISTS kg_claim_extraction_outcomes_delete_guard;
            DROP TRIGGER IF EXISTS chunk_source_manifest_header_update_guard;
            CREATE TRIGGER kg_claim_extraction_outcomes_insert_guard
            BEFORE INSERT ON kg_claim_extraction_outcomes
            WHEN hymem_evidence_mutation_authorized() <> 1
              OR length(trim(new.prompt_version)) = 0
              OR new.prompt_generation < 0
              OR substr(new.result_hash, 1, 7) <> 'sha256:'
              OR length(new.result_hash) <> 71
              OR substr(new.result_hash, 8) GLOB '*[^0-9a-f]*'
            BEGIN
                SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
            END;
            CREATE TRIGGER kg_claim_extraction_outcomes_update_guard
            BEFORE UPDATE ON kg_claim_extraction_outcomes
            WHEN hymem_evidence_mutation_authorized() <> 1
              OR length(trim(new.prompt_version)) = 0
              OR new.prompt_generation < 0
              OR substr(new.result_hash, 1, 7) <> 'sha256:'
              OR length(new.result_hash) <> 71
              OR substr(new.result_hash, 8) GLOB '*[^0-9a-f]*'
            BEGIN
                SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
            END;
            CREATE TRIGGER kg_claim_extraction_outcomes_delete_guard
            BEFORE DELETE ON kg_claim_extraction_outcomes
            WHEN hymem_evidence_mutation_authorized() <> 1
            BEGIN
                SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
            END;
            CREATE TRIGGER chunk_source_manifest_header_update_guard
            BEFORE UPDATE OF source_manifest_version, source_manifest_count ON chunks
            WHEN old.source_manifest_version IS NOT NULL
             AND (new.source_manifest_version IS NOT old.source_manifest_version
                  OR new.source_manifest_count IS NOT old.source_manifest_count)
             AND NOT (
                  new.source_manifest_version IS NULL
                  AND new.source_manifest_count IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM kg_evidence ev WHERE ev.chunk_id = old.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM kg_claim_observations observation
                      WHERE observation.chunk_id = old.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM kg_claim_extraction_outcomes outcome
                      WHERE outcome.chunk_id = old.id
                  )
             )
            BEGIN
                SELECT RAISE(ABORT, 'published chunk source manifest header is immutable');
            END;
            """
        )
    evidence_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(kg_evidence)").fetchall()
    } if "kg_evidence" in tables else set()
    if schema_version(conn) >= 42 and "published_at" in evidence_columns:
        # Heal the write-once publication boundary on every startup. A missing
        # trigger must not turn a stamped v42 store into mutable history.
        _install_evidence_revision_guards(conn)
        _install_evidence_publication_guards(conn)
    if (
        schema_version(conn) >= 43
        and {
            "sessions", "messages", "chunks", "message_retention_coverage",
            "peers", "session_peers", "kg_evidence", "episodes",
            "procedures", "profile_staging", "temporal_mentions",
            "narrative_facts", "chunk_message_sources", "user_profile",
            "kg_claim_observations",
        }.issubset(tables)
        and {"source_peer_id", "source_workspace_id"}.issubset(evidence_columns)
    ):
        _ensure_message_coverage_fts(conn)
        _install_external_peer_guards(conn)
    if schema_version(conn) >= 45 and _v45_domain_present(conn):
        _install_aggregation_source_guards(conn)
    if schema_version(conn) >= 46 and _v46_domain_present(conn):
        _ensure_narrative_facts_fts(conn)
        _install_fact_authority_guards(conn)


_MIGRATION_NAME_RE = re.compile(r"^(\d+)")
# Errors raised when a forward-only migration re-applies against a schema.sql
# database that already has the object. Tolerated so migrations stay no-ops.
_IDEMPOTENT_ERROR_MARKERS = ("duplicate column name", "already exists")


def _discover_migrations() -> list[tuple[int, object]]:
    """Return (version, traversable) for every NNN_*.sql under migrations/,
    sorted ascending by the leading integer."""
    pkg = files("hymem.core.migrations")
    found: list[tuple[int, object]] = []
    for entry in pkg.iterdir():
        name = entry.name
        if not name.endswith(".sql"):
            continue
        match = _MIGRATION_NAME_RE.match(name)
        if match is None:
            continue
        found.append((int(match.group(1)), entry))
    found.sort(key=lambda item: item[0])
    return found


def _split_sql_statements(script: str) -> list[str]:
    """Split a migration script into individual statements, treating a
    ``CREATE TRIGGER ... BEGIN ... END;`` block as one statement (its internal
    semicolons must not split it). Full-line ``--`` comments are dropped.

    Only FULL-LINE comments are stripped, so a semicolon inside a TRAILING
    ``--`` comment still terminates the statement and cuts a CREATE TABLE in
    half ("incomplete input"). Keep migration end-of-line comments
    semicolon-free; schema.sql has no such constraint (executescript hands the
    whole file to SQLite, which parses comments properly).
    """
    body = "\n".join(
        line for line in script.splitlines() if not line.strip().startswith("--")
    )
    parts = re.split(r"(\bBEGIN\b|\bEND\b|;)", body, flags=re.IGNORECASE)
    statements: list[str] = []
    buf: list[str] = []
    depth = 0
    for part in parts:
        token = part.strip().lower()
        if token == "begin":
            depth += 1
            buf.append(part)
        elif token == "end":
            depth = max(0, depth - 1)
            buf.append(part)
        elif part == ";" and depth == 0:
            buf.append(part)
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(part)
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _apply_migration_sql(conn: sqlite3.Connection, script: str) -> None:
    """Execute a migration script statement-by-statement, tolerating the
    idempotency errors a forward-only migration raises on an up-to-date DB
    (duplicate column / object already exists)."""
    for stmt in _split_sql_statements(script):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if any(m in str(exc).lower() for m in _IDEMPOTENT_ERROR_MARKERS):
                continue
            raise


def _v40_sql_is_complete(conn: sqlite3.Connection) -> bool:
    """Recognize the fully installed v40 DDL after a pre-stamp crash.

    Migration 040 rebuilds ``kg_evidence``. Replaying that rebuild after the
    SQL and data hook committed but before ``schema_meta`` was stamped would
    erase canonical provenance and revision history. Partial installs do not
    satisfy this deliberately strict shape/object check and are safe to
    rebuild from their still-pre-hook ledger.
    """
    def columns(table: str) -> set[str]:
        return {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }

    evidence_columns = {
        "source_message_id", "source_session_id", "source_created_at",
        "source_event_at", "source_coverage_chunk_id",
        "source_coverage_version", "provenance_status", "is_current",
        "superseded_at", "superseded_reason", "revision",
        "interpretation_key",
    }
    chunk_columns = {"source_manifest_version", "source_manifest_count"}
    if not evidence_columns.issubset(columns("kg_evidence")):
        return False
    if not chunk_columns.issubset(columns("chunks")):
        return False
    required_objects = {
        ("table", "chunk_message_sources"),
        ("table", "kg_edge_lifecycle"),
        ("table", "kg_lifecycle_dependencies"),
        ("table", "kg_claim_observations"),
        ("trigger", "kg_evidence_v40_insert_guard"),
        ("trigger", "kg_evidence_v40_update_guard"),
        ("trigger", "chunk_source_manifest_header_update_guard"),
        ("trigger", "kg_edge_lifecycle_insert_guard"),
        ("trigger", "kg_evidence_signals_v40_insert_guard"),
        ("trigger", "kg_lifecycle_dependencies_update_guard"),
        ("trigger", "kg_lifecycle_dependencies_delete_guard"),
        ("index", "idx_evidence_canonical_identity"),
    }
    found = {
        (str(row["type"]), str(row["name"]))
        for row in conn.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN "
            "('table','trigger','index')"
        ).fetchall()
    }
    return required_objects.issubset(found)


def _prepare_v40_legacy_shape(conn: sqlite3.Connection) -> None:
    """Complete only the legacy columns required by migration 040.

    The migration test matrix intentionally includes sparse but supported
    historical stores. SQL cannot conditionally ``ALTER`` a missing table, so
    this small preflight handles both an absent graph and old graph rows that
    predate bi-temporal columns.
    """
    tables = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "knowledge_graph" in tables:
        graph_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(knowledge_graph)").fetchall()
        }
        additions = {
            "pos_evidence": "INTEGER NOT NULL DEFAULT 0",
            "neg_evidence": "INTEGER NOT NULL DEFAULT 0",
            "first_seen": "TIMESTAMP",
            "last_seen": "TIMESTAMP",
            "valid_at": "TIMESTAMP",
            "invalid_at": "TIMESTAMP",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "derived": "BOOLEAN NOT NULL DEFAULT 0",
        }
        for name, declaration in additions.items():
            if name not in graph_columns:
                conn.execute(
                    f"ALTER TABLE knowledge_graph ADD COLUMN {name} {declaration}"
                )


def _v40_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether this historical store actually contains the graph domain."""
    tables = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    return {
        "knowledge_graph", "kg_evidence", "chunks", "sessions",
        "message_retention_coverage",
    }.issubset(tables)


def _v45_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether a partial historical store contains the aggregation domain."""

    required_tables = {
        "episodes", "aggregation_nodes", "message_retention_coverage",
    }
    if not all(_table_exists(conn, table) for table in required_tables):
        return False
    episode_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(episodes)").fetchall()
    }
    node_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(aggregation_nodes)").fetchall()
    }
    coverage_info = conn.execute(
        "PRAGMA table_info(message_retention_coverage)"
    ).fetchall()
    coverage_columns = {str(row["name"]) for row in coverage_info}
    coverage_pk = {
        str(row["name"]): int(row["pk"])
        for row in coverage_info if int(row["pk"] or 0) > 0
    }
    # Pre-v45 base columns used by the migration triggers.  New v45 columns
    # may already exist on a fresh schema bootstrap and are deliberately not
    # required here.
    return {
        "id", "session_id", "title", "summary",
    }.issubset(episode_columns) and {
        "id", "title", "summary", "member_episode_ids", "session_ids",
        "n_members", "n_sessions", "level", "is_root",
    }.issubset(node_columns) and {
        "message_id", "source_session_id", "source_role", "source_peer_id",
        "source_workspace_id", "source_created_at", "chunk_id",
        "coverage_version", "message_content_hash",
    }.issubset(coverage_columns) and coverage_pk == {
        "message_id": 1, "chunk_id": 2, "coverage_version": 3,
    }


def _v46_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether a partial historical store contains the complete facts domain."""

    if not all(_table_exists(conn, table) for table in (
        "sessions", "narrative_facts", "narrative_fact_embeddings",
        "message_retention_coverage", "chunks",
    )):
        return False
    session_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    fact_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(narrative_facts)").fetchall()
    }
    embedding_columns = {
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(narrative_fact_embeddings)"
        ).fetchall()
    }
    coverage_info = conn.execute(
        "PRAGMA table_info(message_retention_coverage)"
    ).fetchall()
    coverage_columns = {str(row["name"]) for row in coverage_info}
    coverage_pk = {
        str(row["name"]): int(row["pk"])
        for row in coverage_info if int(row["pk"] or 0) > 0
    }
    return {
        "id", "facts_message_id",
    }.issubset(session_columns) and {
        "id", "session_id", "start_message_id", "end_message_id", "text",
        "fact_date", "entities", "prompt_version", "valid_at", "invalid_at",
        "created_at",
    }.issubset(fact_columns) and {
        "fact_id", "vector_json", "model", "dim", "text_hash", "created_at",
    }.issubset(embedding_columns) and {
        "message_id", "source_session_id", "source_role", "source_peer_id",
        "source_workspace_id", "source_created_at", "chunk_id",
        "coverage_version", "message_content_hash",
    }.issubset(coverage_columns) and coverage_pk == {
        "message_id": 1, "chunk_id": 2, "coverage_version": 3,
    }


def _v46_sql_is_complete(conn: sqlite3.Connection) -> bool:
    """Recognize a fully applied v46 domain whose version stamp is stale.

    Replaying the v46 table rebuild would preserve the current fact projection
    but drop its lifecycle ledger. This deliberately strict structural check
    permits only guard healing plus an atomic stamp when all corrected tables
    and identities are already present.
    """

    required_columns = {
        "sessions": {
            "facts_cursor_message_id", "facts_cursor_partial_message_id",
            "facts_cursor_offset", "facts_cursor_prompt_version",
            "facts_retry_count", "facts_retry_config_version",
            "facts_quarantined",
        },
        "fact_extraction_outcomes": {
            "slice_key", "session_id", "prompt_version", "input_hash",
            "generation", "outcome_status", "result_hash",
            "source_manifest_version", "source_manifest_count",
            "source_manifest_hash", "source_manifest_complete", "succeeded_at",
        },
        "fact_extraction_source_occurrences": {
            "slice_key", "ordinal", "source_message_id", "source_session_id",
            "source_coverage_chunk_id", "source_coverage_version",
            "source_content_hash",
        },
        "fact_extraction_revisions": {
            "slice_key", "generation", "prompt_version", "outcome_status",
            "result_hash", "succeeded_at",
        },
        "narrative_facts": {
            "id", "source_outcome_key", "fact_key", "current_generation",
            "lifecycle_status", "created_at",
        },
        "narrative_fact_lifecycle": {
            "fact_id", "generation", "direction", "event_at",
            "prompt_version", "result_hash", "recorded_at",
        },
        "narrative_fact_embeddings": {
            "fact_id", "vector_json", "model", "dim", "text_hash",
            "created_at",
        },
    }
    for table, expected in required_columns.items():
        if not _table_exists(conn, table):
            return False
        actual = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if not expected.issubset(actual):
            return False
    unique_indexes: set[tuple[str, ...]] = set()
    for index in conn.execute("PRAGMA index_list(narrative_facts)").fetchall():
        if int(index["unique"] or 0) != 1:
            continue
        unique_indexes.add(tuple(
            str(row["name"])
            for row in conn.execute(
                f"PRAGMA index_info({index['name']})"
            ).fetchall()
        ))
    return (
        ("source_outcome_key", "fact_key") in unique_indexes
        and ("session_id", "start_message_id", "text") not in unique_indexes
        and _table_exists(conn, "narrative_facts_fts")
    )


def _seed_v40_legacy_lifecycle(conn: sqlite3.Connection) -> None:
    """Preserve pre-v40 materialized lifecycle without inventing a source."""
    tables = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if not {"knowledge_graph", "kg_edge_lifecycle"}.issubset(tables):
        return
    accepted_at = conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
    rows = conn.execute(
        "SELECT id,status,valid_at,invalid_at,first_seen,last_seen "
        "FROM knowledge_graph WHERE derived=0 ORDER BY id"
    ).fetchall()
    with evidence_mutation(conn):
        for row in rows:
            is_open = row["status"] == "active" and row["invalid_at"] is None
            raw_event = (
                (row["valid_at"] or row["first_seen"])
                if is_open
                else (row["invalid_at"] or row["last_seen"] or row["first_seen"])
            )
            if raw_event is None:
                event_at = "0001-01-01T00:00:00.000Z"
            else:
                try:
                    event_at = normalize_iso_timestamp(
                        raw_event,
                        context="pre-v40 lifecycle",
                    )
                    validate_event_clock(
                        conn,
                        event_at,
                        accepted_at,
                        context="pre-v40 lifecycle",
                    )
                except ValueError:
                    # Never turn SQLite-only Julian/calendar interpretations or
                    # unsupported future snapshots into durable valid history.
                    continue
            conn.execute(
                "INSERT OR IGNORE INTO kg_edge_lifecycle("
                "edge_id,event_key,event_kind,direction,event_at,details) "
                "VALUES (?,'legacy-state','legacy_state',?,?,?)",
                (
                    row["id"],
                    1 if is_open else -1,
                    event_at,
                    "pre-v40 lifecycle snapshot - exact transition provenance unavailable",
                ),
            )
    # Do not rewrite the pre-v40 materialized interval here.  Older migrations
    # deliberately preserved their historical timestamp spelling (for example
    # a date-only ``2024-01-01``).  The lifecycle snapshot stores a normalized
    # event key, while portability canonicalizes an equivalent representation
    # in-memory.  Operational claim events call the reducer themselves.


def _normalize_v40_portable_keys(conn: sqlite3.Connection) -> None:
    """Heal unreleased v40 keys that accidentally embedded local row ids."""
    from hymem.dreaming.evidence import (
        _interpretation_key,
        recanonicalize_lifecycle_keys,
    )

    candidates = conn.execute(
        """
        SELECT * FROM kg_evidence
        WHERE interpretation_key = 'legacy-migrated-v1'
           OR interpretation_key = 'legacy-unspecified'
           OR interpretation_key GLOB 'legacy-row:[0-9]*'
        ORDER BY id
        """
    ).fetchall()
    with evidence_history_mutation(conn):
        for row in candidates:
            semantic_key = _interpretation_key(
                polarity=int(row["polarity"]),
                evidence_weight=int(row["evidence_weight"]),
                weight_source=row["weight_source"],
                source_role=row["source_role"],
                surface_subject=row["surface_subject"],
                surface_object=row["surface_object"],
                value_text=row["value_text"],
                value_numeric=row["value_numeric"],
                value_unit=row["value_unit"],
                temporal_scope=row["temporal_scope"],
            )
            if semantic_key == row["interpretation_key"]:
                continue
            conn.execute(
                "UPDATE kg_evidence SET interpretation_key=? WHERE id=?",
                (semantic_key, row["id"]),
            )
            conn.execute(
                "UPDATE kg_claim_observations SET interpretation_key=? "
                "WHERE evidence_id=?",
                (semantic_key, row["id"]),
            )

        signals = conn.execute(
            "SELECT * FROM kg_evidence_signals ORDER BY id"
        ).fetchall()
        for row in signals:
            key = str(row["signal_key"])
            portable = key
            if re.fullmatch(r"legacy:(positive|negative):[0-9]+", key):
                portable = key.rsplit(":", 1)[0]
            elif re.fullmatch(r"edge:[0-9]+:polarity:-?1", key):
                portable = "runtime-unattributed:polarity:" + key.rsplit(":", 1)[-1]
            if portable == key:
                continue
            collision = conn.execute(
                "SELECT * FROM kg_evidence_signals WHERE edge_id=? "
                "AND signal_kind=? AND signal_key=?",
                (row["edge_id"], row["signal_kind"], portable),
            ).fetchone()
            if collision is not None:
                semantic = (
                    "polarity", "evidence_weight", "counts_toward_confidence",
                    "details", "created_at",
                )
                if any(collision[field] != row[field] for field in semantic):
                    raise RuntimeError("legacy evidence signal key collision")
                conn.execute(
                    "DELETE FROM kg_evidence_signals WHERE id=?", (row["id"],)
                )
            else:
                conn.execute(
                    "UPDATE kg_evidence_signals SET signal_key=? WHERE id=?",
                    (portable, row["id"]),
                )
    # Key healing is intentionally metadata-only.  Recomputing every interval
    # on startup would mutate timestamp spellings established by historical
    # migrations even when their temporal meaning is unchanged.
    recanonicalize_lifecycle_keys(conn)


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply every migration file whose version exceeds the DB's
    schema_version, bumping schema_version after each so an interrupted run
    resumes cleanly. Migrations are idempotent, so a fresh schema.sql database
    (which starts at version 1) runs them all as no-ops up to the latest."""
    cur = schema_version(conn)
    for version, entry in _discover_migrations():
        if version <= cur:
            continue
        apply_v40 = version != 40 or _v40_domain_present(conn)
        apply_version = apply_v40 and (
            version != 42
            or all(_table_exists(conn, table) for table in (
                "kg_evidence", "kg_claim_observations",
                "kg_claim_extraction_outcomes", "kg_edge_lifecycle",
            ))
        )
        if version == 43:
            apply_version = apply_version and _v43_domain_present(conn)
        if version == 45:
            apply_version = apply_version and _v45_domain_present(conn)
        if version == 46:
            apply_version = apply_version and _v46_domain_present(conn)
        if version == 40 and apply_v40 and not _v40_sql_is_complete(conn):
            _prepare_v40_legacy_shape(conn)
        if version == 46 and apply_version and _v46_sql_is_complete(conn):
            # SQL may have committed before an old process published the
            # version marker. Never replay the destructive facts-table rebuild
            # over an already-authoritative lifecycle ledger.
            with transaction(conn):
                _install_fact_authority_guards(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key, value) "
                    "VALUES ('schema_version', ?)", (str(version),),
                )
            log.info("recognized complete schema v%d (%s)", version, entry.name)
            continue
        if version == 46 and apply_version:
            # v46 replaces the v26 facts table to remove its lossy legacy
            # UNIQUE key. DDL and version publication must be one crash-atomic
            # unit: an interrupted DROP/rename can never strand half a domain
            # while schema_meta still advertises v45.
            with transaction(conn):
                with evidence_mutation(conn):
                    _apply_migration_sql(
                        conn, entry.read_text(encoding="utf-8")
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key, value) "
                    "VALUES ('schema_version', ?)", (str(version),),
                )
            log.info("migrated schema to v%d (%s)", version, entry.name)
            continue
        if apply_version and (version != 40 or not _v40_sql_is_complete(conn)):
            # Forward migrations are an internal, transactionally owned
            # rewrite.  This also lets a fresh bootstrap replay pre-v40 signal
            # seed statements after the latest schema has installed v40's
            # direct-SQL mutation guards.
            with evidence_mutation(conn):
                _apply_migration_sql(conn, entry.read_text(encoding="utf-8"))
        if version == 39:
            # The DDL is idempotent, so complete its Python data hook before
            # publishing the version marker. A crash here leaves v38 stamped
            # and startup safely replays both pieces instead of permanently
            # skipping canonical provenance materialization.
            _backfill_v39_message_coverage(conn)
        if version == 40 and apply_v40:
            # SQL establishes the guarded ledger first. This data hook can
            # then recognize only source chunks whose exact historical builder
            # text is reproducible from validated v38 artifacts. A crash keeps
            # v39 stamped and safely replays the idempotent hook on restart.
            _backfill_v40_chunk_manifests(conn)
            _seed_v40_legacy_lifecycle(conn)
        if version == 41:
            # Non-empty v40 observation sets prove that a source-validated
            # extraction was published. Old processed markers alone do not
            # prove an empty success, so the hook deliberately does not invent
            # empty authority. Keeping this before the version stamp gives the
            # Python data phase the same crash-replay semantics as v39/v40.
            _backfill_v41_claim_extraction_outcomes(conn)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(version),),
        )
        log.info("migrated schema to v%d (%s)", version, entry.name)
    if schema_version(conn) >= 40 and _v40_domain_present(conn):
        _normalize_v40_portable_keys(conn)
    if schema_version(conn) >= 41 and _table_exists(
        conn, "kg_claim_extraction_outcomes"
    ):
        _ensure_v41_claim_extraction_outcome_shape(conn)
        _refresh_v41_claim_extraction_outcomes(conn)
    if schema_version(conn) >= 39 and _table_exists(conn, "user_profile"):
        _ensure_profile_active_invariants(conn)
    _ensure_post_migration_runtime_guards(conn)


def _backfill_v39_message_coverage(conn: sqlite3.Connection) -> None:
    """Give upgraded profile provenance an exportable canonical source.

    Migration 039 can recover source ids from surviving raw USER rows. Cover
    every surviving raw row in each affected session before moving its ordered
    frontier, so upgrade→export→import is lossless without claiming a sparse
    one-message stream.
    """
    from hymem.dreaming.lossless import backfill_all_message_coverage

    with transaction(conn):
        sessions = conn.execute(
            "SELECT DISTINCT source_session_id AS session_id "
            "FROM user_profile WHERE source_session_id IS NOT NULL "
            "AND source_message_id IS NOT NULL ORDER BY source_session_id"
        ).fetchall()
        for row in sessions:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (row["session_id"],)
            ).fetchone()
            if exists is not None:
                backfill_all_message_coverage(conn, row["session_id"])


def _backfill_v40_chunk_manifests(conn: sqlite3.Connection) -> None:
    """Recover exact membership for recognized legacy chunk-builder output.

    The historical salience/baseline builder emitted either ``user: ...`` or
    one assistant endpoint followed by one user endpoint. We reproduce those
    bytes from immutable coverage; anything else remains deliberately
    unmanifested. Only a one-message chunk can also attribute an old claim
    exactly, so paired chunks replay under the v13 prompt instead of guessing.
    """
    from hymem.dreaming.lossless import validate_message_coverage_artifact
    from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
    from hymem.dreaming.evidence import (
        claim_assertion_event_key,
        prompt_generation,
    )

    with transaction(conn):
        rows = conn.execute(
            """
            SELECT id, session_id, start_message_id, end_message_id, text
            FROM chunks
            WHERE chunk_kind = 'extraction'
              AND source_manifest_version IS NULL
              AND COALESCE(salience_reason, '') <> 'short_session_fallback'
              AND session_id IS NOT NULL
              AND start_message_id IS NOT NULL
              AND end_message_id IS NOT NULL
            ORDER BY id
            """
        ).fetchall()
        for chunk in rows:
            ids = [int(chunk["start_message_id"])]
            if int(chunk["end_message_id"]) != ids[0]:
                ids.append(int(chunk["end_message_id"]))
            proofs = []
            valid = True
            for message_id in ids:
                proof_row = conn.execute(
                    """
                    SELECT chunk_id FROM message_retention_coverage
                    WHERE message_id = ? AND source_session_id = ?
                      AND coverage_version = ?
                    """,
                    (message_id, chunk["session_id"], LOSSLESS_COVERAGE_VERSION),
                ).fetchone()
                if proof_row is None:
                    valid = False
                    break
                try:
                    proof = validate_message_coverage_artifact(
                        conn, message_id=message_id,
                        chunk_id=proof_row["chunk_id"],
                        coverage_version=LOSSLESS_COVERAGE_VERSION,
                    )
                except (RuntimeError, TypeError, ValueError):
                    valid = False
                    break
                proofs.append(proof)
            if not valid or not proofs:
                continue
            if len(proofs) == 1:
                expected_text = f"{proofs[0].role}: {proofs[0].content}"
                roles_valid = proofs[0].role == "user"
            else:
                expected_text = (
                    f"assistant: {proofs[0].content}\nuser: {proofs[1].content}"
                )
                roles_valid = proofs[0].role == "assistant" and proofs[1].role == "user"
            if not roles_valid or chunk["text"] != expected_text:
                continue
            conn.executemany(
                """
                INSERT INTO chunk_message_sources(
                    chunk_id, ordinal, source_message_id, source_session_id,
                    source_coverage_chunk_id, source_coverage_version
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["id"], ordinal, proof.message_id,
                        proof.session_id, proof.chunk_id,
                        LOSSLESS_COVERAGE_VERSION,
                    )
                    for ordinal, proof in enumerate(proofs)
                ],
            )
            conn.execute(
                "UPDATE chunks SET source_manifest_version = ?, "
                "source_manifest_count = ? WHERE id = ?",
                ("claim-source-manifest-v1", len(proofs), chunk["id"]),
            )
        # Promote old claim rows only when the manifested chunk has exactly one
        # source AND that prospective source identity occurs exactly once. Do
        # this in a second pass so two ambiguous chunks cannot make the first
        # arrival authoritative.
        candidates = conn.execute(
            """
            SELECT ev.id, ev.edge_id, ev.chunk_id, ev.evidence_kind,
                   ev.polarity, ev.extraction_prompt_version, ev.extracted_at,
                   ev.interpretation_key,
                   cms.source_message_id, cms.source_session_id,
                   cms.source_coverage_chunk_id, cms.source_coverage_version,
                   mc.source_role, mc.source_created_at
            FROM kg_evidence ev
            JOIN chunks c ON c.id = ev.chunk_id
            JOIN chunk_message_sources cms
              ON cms.chunk_id = c.id AND cms.ordinal = 0
            JOIN message_retention_coverage mc
              ON mc.message_id = cms.source_message_id
             AND mc.chunk_id = cms.source_coverage_chunk_id
             AND mc.coverage_version = cms.source_coverage_version
            WHERE ev.provenance_status = 'legacy_unattributed'
              AND c.source_manifest_version = 'claim-source-manifest-v1'
              AND c.source_manifest_count = 1
            ORDER BY ev.id
            """
        ).fetchall()
        groups: dict[tuple[int, str, str, int], list[sqlite3.Row]] = {}
        for row in candidates:
            key = (
                int(row["edge_id"]), row["evidence_kind"],
                row["source_session_id"], int(row["source_message_id"]),
            )
            groups.setdefault(key, []).append(row)
        for rows_for_source in groups.values():
            if len(rows_for_source) != 1:
                continue
            row = rows_for_source[0]
            try:
                event_at = normalize_iso_timestamp(
                    row["source_created_at"],
                    context="legacy claim source",
                )
            except ValueError:
                # SQLite accepts bare Julian numbers, impossible calendar
                # dates, and other shapes that the public clock does not. Such
                # old rows remain explicitly unattributed rather than gaining
                # invented canonical history during startup healing.
                continue
            # A legacy row whose source clock leads its extraction clock cannot
            # be safely promoted: doing so would let a future assertion become
            # today's materialized state. Keep it explicitly unattributed so
            # schema healing remains available rather than bricking startup.
            from hymem.core.time import validate_event_clock

            try:
                validate_event_clock(
                    conn,
                    event_at,
                    row["extracted_at"],
                    context="legacy claim promotion",
                )
            except ValueError:
                continue
            with evidence_mutation(conn):
                conn.execute(
                    """
                    UPDATE kg_evidence
                    SET source_message_id = ?, source_session_id = ?,
                        source_role = ?, source_created_at = ?, source_event_at = ?,
                        source_coverage_chunk_id = ?, source_coverage_version = ?,
                        provenance_status = 'canonical'
                    WHERE id = ?
                    """,
                    (
                        row["source_message_id"], row["source_session_id"],
                        row["source_role"], row["source_created_at"], event_at,
                        row["source_coverage_chunk_id"],
                        row["source_coverage_version"], row["id"],
                    ),
                )
            with evidence_mutation(conn):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO kg_claim_observations(
                        chunk_id, edge_id, source_session_id, source_message_id,
                        evidence_kind, polarity, prompt_version, prompt_generation,
                        evidence_id, interpretation_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["chunk_id"], row["edge_id"], row["source_session_id"],
                        row["source_message_id"], row["evidence_kind"],
                        row["polarity"],
                        row["extraction_prompt_version"] or "pre-v40",
                        prompt_generation(
                            row["extraction_prompt_version"] or "pre-v40"
                        ),
                        row["id"], row["interpretation_key"],
                    ),
                )
            if int(conn.execute(
                "SELECT polarity FROM kg_evidence WHERE id = ?", (row["id"],)
            ).fetchone()[0]) == 1:
                from hymem.dreaming.bitemporal import record_lifecycle_event

                record_lifecycle_event(
                    conn,
                    edge_id=int(row["edge_id"]),
                    event_key=claim_assertion_event_key(
                        row["source_session_id"], row["source_message_id"],
                        row["evidence_kind"], 1,
                    ),
                    event_kind="claim_assertion",
                    direction=1,
                    event_at=event_at,
                    source_evidence_id=int(row["id"]),
                )


def _backfill_v41_claim_extraction_outcomes(conn: sqlite3.Connection) -> None:
    """Backfill only non-empty, coherent v40 chunk publications."""
    if not _table_exists(conn, "kg_claim_extraction_outcomes") or not _table_exists(
        conn, "kg_claim_observations"
    ):
        return
    from hymem.dreaming.evidence import claim_observation_result_hash

    chunks = conn.execute(
        "SELECT DISTINCT chunk_id FROM kg_claim_observations ORDER BY chunk_id"
    ).fetchall()
    with transaction(conn), evidence_mutation(conn):
        for item in chunks:
            chunk_id = str(item["chunk_id"])
            authority = conn.execute(
                "SELECT DISTINCT prompt_version,prompt_generation "
                "FROM kg_claim_observations WHERE chunk_id=?",
                (chunk_id,),
            ).fetchall()
            if len(authority) != 1:
                # A normal whole-chunk publication has one prompt authority.
                # Mixed legacy rows are not enough proof to fabricate one.
                continue
            row = authority[0]
            conn.execute(
                "INSERT OR IGNORE INTO kg_claim_extraction_outcomes("
                "chunk_id,prompt_version,prompt_generation,result_hash,succeeded_at) "
                "VALUES (?,?,?,?,COALESCE((SELECT MAX(observed_at) "
                "FROM kg_claim_observations WHERE chunk_id=?),CURRENT_TIMESTAMP))",
                (
                    chunk_id, row["prompt_version"], row["prompt_generation"],
                    claim_observation_result_hash(conn, chunk_id), chunk_id,
                ),
            )


def _ensure_v41_claim_extraction_outcome_shape(conn: sqlite3.Connection) -> None:
    """Heal an early stamped-v41 outcome FK from CASCADE to RESTRICT.

    Empty successful publications are the only durable proof that a newer
    extraction intentionally returned no claims. Cascading their row with a
    direct chunk deletion would permit a stale portable snapshot to resurrect
    those claims. Rebuild only the single table when its FK action is wrong;
    the runtime guard installer below then restores every dependent trigger.
    """
    foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(kg_claim_extraction_outcomes)"
    ).fetchall()
    if any(
        row["from"] == "chunk_id"
        and row["table"] == "chunks"
        and row["to"] == "id"
        and str(row["on_delete"]).upper() == "RESTRICT"
        for row in foreign_keys
    ):
        return
    expected_columns = {
        "chunk_id", "prompt_version", "prompt_generation", "result_hash",
        "succeeded_at",
    }
    actual_columns = {
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(kg_claim_extraction_outcomes)"
        ).fetchall()
    }
    if actual_columns != expected_columns:
        raise RuntimeError("unsupported claim extraction outcome table shape")

    conn.execute("SAVEPOINT hymem_heal_v41_outcome_fk")
    try:
        for trigger in (
            "kg_claim_extraction_outcomes_insert_guard",
            "kg_claim_extraction_outcomes_update_guard",
            "kg_claim_extraction_outcomes_delete_guard",
            "chunk_source_manifest_header_update_guard",
            "kg_evidence_published_at_insert_guard",
            "kg_evidence_published_at_update_guard",
            "kg_evidence_v40_delete_guard",
            "kg_edge_lifecycle_update_guard",
            "kg_edge_lifecycle_delete_guard",
            "kg_lifecycle_dependencies_update_guard",
            "kg_lifecycle_dependencies_delete_guard",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.execute(
            "ALTER TABLE kg_claim_extraction_outcomes "
            "RENAME TO kg_claim_extraction_outcomes_v41_old"
        )
        conn.execute(
            "CREATE TABLE kg_claim_extraction_outcomes("
            "chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE RESTRICT,"
            "prompt_version TEXT NOT NULL,"
            "prompt_generation INTEGER NOT NULL CHECK(prompt_generation >= 0),"
            "result_hash TEXT NOT NULL,"
            "succeeded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO kg_claim_extraction_outcomes("
            "chunk_id,prompt_version,prompt_generation,result_hash,succeeded_at) "
            "SELECT chunk_id,prompt_version,prompt_generation,result_hash,succeeded_at "
            "FROM kg_claim_extraction_outcomes_v41_old"
        )
        conn.execute("DROP TABLE kg_claim_extraction_outcomes_v41_old")
    except BaseException:
        conn.execute("ROLLBACK TO hymem_heal_v41_outcome_fk")
        conn.execute("RELEASE hymem_heal_v41_outcome_fk")
        raise
    conn.execute("RELEASE hymem_heal_v41_outcome_fk")


def _refresh_v41_claim_extraction_outcomes(conn: sqlite3.Connection) -> None:
    """Keep v41 hashes aligned with intentional v40 key normalization."""
    from hymem.dreaming.evidence import refresh_claim_extraction_outcomes

    rows = conn.execute(
        "SELECT chunk_id FROM kg_claim_extraction_outcomes ORDER BY chunk_id"
    ).fetchall()
    refresh_claim_extraction_outcomes(
        conn, [str(row["chunk_id"]) for row in rows]
    )


def _ensure_profile_active_invariants(conn: sqlite3.Connection) -> None:
    """Heal legacy rows, then install v39 profile shape/source guards."""
    owner = not conn.in_transaction
    if owner:
        conn.execute("BEGIN IMMEDIATE")
    try:
        from hymem.dreaming.lossless import covered_messages_after
        from hymem.dreaming.user_profile import reconcile_profile_intervals

        # Reopen may be healing rows written by an older binary (or by direct
        # SQL while guards were absent).  Remove the previous guard set before
        # doing that repair, then reinstall it in this same transaction below.
        # Otherwise a now-invalid legacy tuple can make its own cleanup fail.
        for trigger_name in (
            "user_profile_shape_insert_guard",
            "user_profile_shape_update_guard",
            "user_profile_source_insert_guard",
            "user_profile_source_update_guard",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")

        # Canonicalizing relationship keys can collapse case/whitespace
        # variants, so remove the old case-sensitive guard inside this same
        # transaction before healing their interval chain.
        conn.execute(
            "DROP INDEX IF EXISTS idx_user_profile_one_active_relationship"
        )
        # Empty legacy assertions cannot carry memory. Remove them before the
        # write guards make their invalid shape immutable; retain every
        # non-empty assertion.
        conn.execute("DELETE FROM user_profile WHERE trim(value) = ''")
        conn.execute(
            "UPDATE user_profile SET slot_key = NULL "
            "WHERE slot <> 'relationship' AND slot_key IS NOT NULL"
        )
        for row in conn.execute(
            "SELECT id, slot_key FROM user_profile WHERE slot = 'relationship'"
        ).fetchall():
            key = (
                row["slot_key"].strip().lower()
                if isinstance(row["slot_key"], str) and row["slot_key"].strip()
                else "[legacy-unknown]"
            )
            if key != row["slot_key"]:
                conn.execute(
                    "UPDATE user_profile SET slot_key = ? WHERE id = ?",
                    (key, row["id"]),
                )

        # Never bless partial or unverifiable legacy provenance. A durable
        # tuple survives only when the producer-bounded USER artifact proves
        # it. The nullable live FK is retained only when that same raw row is
        # still an exact copy; retention may later SET NULL without erasing the
        # durable source tuple.
        for row in conn.execute(
            "SELECT id, evidence_message_id, source_message_id, "
            "source_session_id, source_created_at FROM user_profile"
        ).fetchall():
            source_mid = row["source_message_id"]
            source_session = row["source_session_id"]
            proof = None
            if isinstance(source_mid, int) and isinstance(source_session, str):
                covered = covered_messages_after(
                    conn,
                    source_session,
                    source_mid - 1,
                    limit=1,
                    roles=frozenset({"user"}),
                    through_message_id=source_mid,
                )
                if (
                    covered
                    and covered[0].message_id == source_mid
                    and covered[0].source_created_at == row["source_created_at"]
                ):
                    proof = covered[0]
            if proof is None:
                if any(
                    row[name] is not None
                    for name in (
                        "evidence_message_id", "source_message_id",
                        "source_session_id", "source_created_at",
                    )
                ):
                    conn.execute(
                        "UPDATE user_profile SET evidence_message_id = NULL, "
                        "source_message_id = NULL, source_session_id = NULL, "
                        "source_created_at = NULL WHERE id = ?",
                        (row["id"],),
                    )
                continue
            evidence_mid = row["evidence_message_id"]
            if evidence_mid is not None:
                live = conn.execute(
                    "SELECT session_id, role, content, created_at FROM messages "
                    "WHERE id = ?",
                    (evidence_mid,),
                ).fetchone()
                if (
                    evidence_mid != source_mid
                    or live is None
                    or live["session_id"] != source_session
                    or live["role"] != "user"
                    or live["content"] != proof.content
                    or live["created_at"] != proof.source_created_at
                ):
                    conn.execute(
                        "UPDATE user_profile SET evidence_message_id = NULL "
                        "WHERE id = ?",
                        (row["id"],),
                    )
        keys = conn.execute(
            "SELECT DISTINCT slot, slot_key FROM user_profile "
            "WHERE slot IN ('name','role','employer','location','age_birthday') "
            "OR slot = 'relationship'"
        ).fetchall()
        for row in keys:
            reconcile_profile_intervals(conn, row["slot"], row["slot_key"])
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_profile_one_active_singleton "
            "ON user_profile(slot) WHERE invalid_at IS NULL AND "
            "slot IN ('name','role','employer','location','age_birthday')"
        )
        # COALESCE also protects malformed/legacy NULL relationship keys;
        # SQLite's ordinary UNIQUE semantics otherwise allow unlimited NULLs.
        conn.execute(
            "CREATE UNIQUE INDEX idx_user_profile_one_active_relationship "
            "ON user_profile(slot, lower(trim(COALESCE(slot_key, '')))) "
            "WHERE invalid_at IS NULL AND slot = 'relationship'"
        )
        shape_check = (
            "new.slot NOT IN ('role','name','employer','location','language',"
            "'relationship','possession','age_birthday','health_condition',"
            "'recurring_activity') OR "
            "trim(new.value) = '' OR "
            "new.confidence IS NULL OR NOT (new.confidence >= 0.0 "
            "AND new.confidence <= 1.0) OR "
            "(new.slot = 'relationship' AND "
            " (new.slot_key IS NULL OR trim(new.slot_key) = '' "
            "OR new.slot_key <> lower(trim(new.slot_key)))) OR "
            "(new.slot <> 'relationship' AND new.slot_key IS NOT NULL)"
        )
        for operation, suffix in (("INSERT", "insert"), ("UPDATE", "update")):
            conn.execute(
                f"CREATE TRIGGER user_profile_shape_{suffix}_guard "
                f"BEFORE {operation} ON user_profile WHEN {shape_check} "
                "BEGIN SELECT RAISE(ABORT, 'invalid user_profile shape'); END"
            )
        source_valid = """
            (
                new.source_message_id IS NOT NULL
                AND new.source_session_id IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM message_retention_coverage mc
                    JOIN sessions s ON s.id = mc.source_session_id
                    JOIN chunks c ON c.id = mc.chunk_id
                    WHERE mc.message_id = new.source_message_id
                      AND mc.source_session_id = new.source_session_id
                      AND mc.source_role = 'user'
                      AND mc.source_created_at IS new.source_created_at
                      AND mc.coverage_version = 'dream-lossless-message-v1'
                      AND s.coverage_message_id IS NOT NULL
                      AND mc.message_id <= s.coverage_message_id
                      AND c.chunk_kind = 'coverage'
                      AND c.session_id = mc.source_session_id
                      AND mc.chunk_id = hymem_coverage_chunk_id(
                          mc.source_session_id, mc.message_id
                      )
                      AND c.start_message_id = mc.message_id
                      AND c.end_message_id = mc.message_id
                      AND json_valid(c.text)
                      AND json_extract(c.text, '$.id') = mc.message_id
                      AND json_extract(c.text, '$.role') = 'user'
                      AND json_extract(c.text, '$.record_version') =
                          mc.record_version
                      AND hymem_message_record_proof_valid(
                          c.text, mc.message_content_hash,
                          mc.hash_version, mc.record_version
                      ) = 1
                      AND (
                          new.evidence_message_id IS NULL
                          OR EXISTS (
                              SELECT 1 FROM messages m
                              WHERE m.id = new.evidence_message_id
                                AND m.id = new.source_message_id
                                AND m.session_id = new.source_session_id
                                AND m.role = 'user'
                                AND m.created_at IS new.source_created_at
                                AND m.content = json_extract(c.text, '$.content')
                          )
                      )
                )
            )
        """
        conn.execute(
            "CREATE TRIGGER user_profile_source_insert_guard "
            "BEFORE INSERT ON user_profile "
            f"WHEN NOT ({source_valid}) "
            "BEGIN SELECT RAISE(ABORT, 'invalid user_profile provenance'); END"
        )
        # Existing unattributed rows are a supported legacy state. They may be
        # updated (for example interval reconciliation or confidence healing)
        # only while remaining unattributed; new rows must always carry a
        # producer-bounded USER source through the INSERT guard above.
        legacy_unchanged = """
            old.evidence_message_id IS NULL
            AND old.source_message_id IS NULL
            AND old.source_session_id IS NULL
            AND old.source_created_at IS NULL
            AND new.evidence_message_id IS NULL
            AND new.source_message_id IS NULL
            AND new.source_session_id IS NULL
            AND new.source_created_at IS NULL
        """
        conn.execute(
            "CREATE TRIGGER user_profile_source_update_guard "
            "BEFORE UPDATE ON user_profile "
            f"WHEN NOT (({source_valid}) OR ({legacy_unchanged})) "
            "BEGIN SELECT RAISE(ABORT, 'invalid user_profile provenance'); END"
        )
    except Exception:
        if owner:
            conn.execute("ROLLBACK")
        raise
    else:
        if owner:
            conn.execute("COMMIT")


_VEC_TABLES = frozenset({
    "vec_chunks", "vec_messages", "vec_edges", "vec_episodes", "vec_facts",
})


def _ensure_vec_table_named(conn: sqlite3.Connection, name: str, dim: int) -> bool:
    """Ensure one vec0 shadow and report whether it was newly created."""
    if name not in _VEC_TABLES:
        raise ValueError(f"unknown vec table: {name}")
    existed = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0(embedding float[{dim}])"
    )
    return not existed


def ensure_vec_table(
    conn: sqlite3.Connection, dim: int, *, model: str | None = None
) -> None:
    """Ensure all vec0 shadows exist for one exact vector-space identity.

    The virtual tables share ``vec_dim``/``vec_model`` metadata, so on a
    dimension or known-model change they are dropped and rebuilt in lockstep, then
    backfilled from their JSON mirror tables (chunk_embeddings /
    edge_embeddings / episode_embeddings).
    """
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("vector dimension must be a positive integer")
    if model is not None and (not isinstance(model, str) or not model):
        raise ValueError("vector model id must be non-empty")
    if not _load_vec_extension(conn):
        return
    try:
        existing_dim = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
        ).fetchone()
        existing_model = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'vec_model'"
        ).fetchone()
        try:
            stored_dim = int(existing_dim["value"]) if existing_dim else None
        except (TypeError, ValueError, OverflowError):
            stored_dim = None
        identity_matches = bool(
            stored_dim == dim
            and (
                model is None
                or (existing_model is not None and existing_model["value"] == model)
            )
        )
        if identity_matches:
            # The durable mirrors are authoritative and every normal persist
            # writes its one shadow row.  Re-scanning every corpus on every
            # message append made hot ingestion quadratic.  Backfill only a
            # shadow that had to be created; explicit resync remains the repair
            # path for physical-row drift.
            for table, backfill in (
                ("vec_chunks", _backfill_vec),
                ("vec_edges", _backfill_vec_edges),
                ("vec_messages", _backfill_vec_messages),
                ("vec_episodes", _backfill_vec_episodes),
                ("vec_facts", _backfill_vec_facts),
            ):
                if _ensure_vec_table_named(conn, table, dim):
                    backfill(conn, dim, model=model)
            return
        conn.execute("DELETE FROM schema_meta WHERE key IN ('vec_dim','vec_model')")
        for stale in (
            "vec_chunks", "vec_messages", "vec_edges", "vec_episodes",
            "vec_facts",
        ):
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f"DROP TABLE IF EXISTS {stale}")
        _ensure_vec_table_named(conn, "vec_chunks", dim)
        _ensure_vec_table_named(conn, "vec_messages", dim)
        _ensure_vec_table_named(conn, "vec_edges", dim)
        _ensure_vec_table_named(conn, "vec_episodes", dim)
        _ensure_vec_table_named(conn, "vec_facts", dim)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('vec_dim', ?)",
            (str(dim),),
        )
        if model is not None:
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) "
                "VALUES ('vec_model', ?)",
                (model,),
            )
        _backfill_vec(conn, dim, model=model)
        _backfill_vec_messages(conn, dim, model=model)
        _backfill_vec_edges(conn, dim, model=model)
        _backfill_vec_episodes(conn, dim, model=model)
        _backfill_vec_facts(conn, dim, model=model)
    except sqlite3.OperationalError:
        log.info("vec tables unavailable; using Python cosine search")


def _backfill_vec(
    conn: sqlite3.Connection, dim: int, *, model: str | None = None
) -> None:
    rows = conn.execute(
        "SELECT c.rowid, e.vector_json FROM chunk_embeddings e "
        "JOIN chunks c ON c.id = e.chunk_id "
        "WHERE c.chunk_kind = 'extraction' AND e.dim = ? "
        + ("AND e.model = ? " if model is not None else "")
        + "ORDER BY c.rowid",
        (dim, model) if model is not None else (dim,),
    ).fetchall()
    if not rows:
        return
    count = conn.execute("SELECT COUNT(*) AS c FROM vec_chunks").fetchone()["c"]
    if count >= len(rows):
        return

    for r in rows:
        try:
            vec = decode_vector(r["vector_json"])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        vec = _finite_vec(vec, dim)
        if vec is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (r["rowid"], _pack_vector(vec)),
        )
    log.info("backfilled vec_chunks with %d existing embeddings", len(rows))


def _backfill_vec_edges(
    conn: sqlite3.Connection, dim: int, *, model: str | None = None
) -> None:
    """Populate vec_edges (rowid = knowledge_graph.id) from cached edge vectors.

    Best-effort: embed_pending_edges is the authoritative refresh. This handles
    cold-start, dim changes, and pre-v6 DBs.
    """
    rows = conn.execute(
        f"""
        SELECT kg.id AS edge_id,
               kg.subject_canonical || ' ' || kg.predicate || ' '
                   || kg.object_canonical AS edge_text
        FROM knowledge_graph kg
        WHERE {live_edge_predicate('kg')}
        """
    ).fetchall()
    if not rows:
        return
    have = conn.execute("SELECT COUNT(*) AS c FROM vec_edges").fetchone()["c"]
    if have >= len(rows):
        return
    for r in rows:
        emb = conn.execute(
            "SELECT vector_json,dim,model FROM edge_embeddings WHERE edge_text = ?",
            (r["edge_text"],),
        ).fetchone()
        if emb is None:
            continue
        try:
            stored_dim = int(emb["dim"])
        except (TypeError, ValueError, OverflowError):
            continue
        if stored_dim != dim:
            continue
        if model is not None and emb["model"] != model:
            continue
        try:
            vec = decode_vector(emb["vector_json"])
        except (AttributeError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(vec, (list, tuple)) or len(vec) != dim:
            continue
        try:
            vector = [float(value) for value in vec]
            finite = all(math.isfinite(value) for value in vector)
            norm = math.sqrt(sum(value ** 2 for value in vector))
        except (TypeError, ValueError, OverflowError):
            finite = False
            norm = 0.0
            vector = []
        if not finite or not math.isfinite(norm) or norm <= 0.0:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO vec_edges(rowid, embedding) VALUES (?, ?)",
            (r["edge_id"], _pack_vector(vector)),
        )
    log.info("backfilled vec_edges from %d edge rows", len(rows))


def _backfill_vec_episodes(
    conn: sqlite3.Connection, dim: int, *, model: str | None = None
) -> None:
    """Populate vec_episodes from episode_embeddings on cold start / dim change."""
    rows = conn.execute(
        "SELECT em.episode_id, em.vector_json "
        "FROM episode_embeddings em "
        "JOIN episodes e ON e.id = em.episode_id "
        "JOIN sessions s ON s.id = e.session_id "
        "WHERE (e.digest_generation IS NULL "
        "OR e.digest_generation = s.digest_published_generation) "
        "AND em.dim = ? "
        + ("AND em.model = ?" if model is not None else ""),
        (dim, model) if model is not None else (dim,),
    ).fetchall()
    if not rows:
        return
    have = conn.execute("SELECT COUNT(*) AS c FROM vec_episodes").fetchone()["c"]
    if have >= len(rows):
        return
    # rowid in vec_episodes mirrors the episodes.rowid so a vec_search hit
    # joins back via rowid → episodes row in one step.
    id_rowid = {
        r["id"]: r["rowid"]
        for r in conn.execute(
            "SELECT e.id, e.rowid FROM episodes e "
            "JOIN sessions s ON s.id = e.session_id "
            "WHERE e.digest_generation IS NULL "
            "OR e.digest_generation = s.digest_published_generation"
        ).fetchall()
    }
    for r in rows:
        rowid = id_rowid.get(r["episode_id"])
        if rowid is None:
            continue
        try:
            vec = decode_vector(r["vector_json"])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        vec = _finite_vec(vec, dim)
        if vec is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
            (rowid, _pack_vector(vec)),
        )
    log.info("backfilled vec_episodes from %d episode rows", len(rows))


def _backfill_vec_facts(
    conn: sqlite3.Connection, dim: int, *, model: str | None = None
) -> None:
    """Populate vec_facts (rowid = narrative_facts.id, an INTEGER PRIMARY KEY,
    so VACUUM-stable like vec_edges) from the JSON mirror on cold start / dim
    change. Suppresses its own missing-table error so ensure_vec_table keeps
    working against a pre-v26 store."""
    with contextlib.suppress(sqlite3.OperationalError):
        rows = conn.execute(
            "SELECT fact_id, vector_json FROM narrative_fact_embeddings "
            "WHERE dim = ? "
            + ("AND model = ?" if model is not None else ""),
            (dim, model) if model is not None else (dim,),
        ).fetchall()
        if not rows:
            return
        have = conn.execute("SELECT COUNT(*) AS c FROM vec_facts").fetchone()["c"]
        if have >= len(rows):
            return
        for r in rows:
            try:
                vec = decode_vector(r["vector_json"])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            vec = _finite_vec(vec, dim)
            if vec is None:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO vec_facts(rowid, embedding) VALUES (?, ?)",
                (r["fact_id"], _pack_vector(vec)),
            )
        log.info("backfilled vec_facts from %d fact rows", len(rows))


def _backfill_vec_messages(
    conn: sqlite3.Connection, dim: int, *, model: str | None = None
) -> None:
    """Populate the stable message-id vec0 shadow from its durable mirror."""
    query = "SELECT message_id, vector_json FROM message_embeddings WHERE dim = ?"
    params: tuple[object, ...] = (dim,)
    if model is not None:
        query += " AND model = ?"
        params = (dim, model)
    try:
        rows = conn.execute(query + " ORDER BY message_id", params).fetchall()
    except sqlite3.OperationalError:
        return
    for row in rows:
        try:
            decoded = decode_vector(row["vector_json"])
        except (AttributeError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
        vector = _finite_vec(decoded, dim)
        if vector is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO vec_messages(rowid, embedding) VALUES (?, ?)",
            (row["message_id"], _pack_vector(vector)),
        )


def _finite_vec(value: object, dim: int) -> list[float] | None:
    """Return one exact, finite, non-zero vector or ``None``."""
    if not isinstance(value, (list, tuple)) or len(value) != dim:
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    return vector if math.isfinite(norm) and norm > 0.0 else None


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# ─────────────────────────────────────────────────────────────────────────────
# Rowid-shadow integrity. episodes / chunks / aggregation_nodes have TEXT
# primary keys, so their rowids are implicit — and SQLite's VACUUM may RENUMBER
# implicit rowids (it compacts freelist gaps). Everything keyed on those rowids
# from the outside silently decouples when that happens: the external-content
# FTS tables (chunks_fts / episodes_fts / aggregation_nodes_fts) start joining
# match hits to the wrong content rows, and the vec_* mirrors translate KNN
# hits to the wrong ids. Worse, the drift then compounds: each new episode
# INSERTs its vector at its (new) rowid, overwriting whichever old row happened
# to sit there. This was a root cause of the 2026-07-12 RAPTOR reuse
# instability — candidate blocking clustered on garbage neighborhoods after
# post-prune VACUUMs — and it quietly degrades plain FTS/vec retrieval too.
# messages_fts (content_rowid='id', INTEGER PRIMARY KEY), vec_edges (rowid =
# knowledge_graph.id, INTEGER PRIMARY KEY), and narrative_facts_fts/vec_facts
# (rowid = narrative_facts.id, INTEGER PRIMARY KEY) are VACUUM-stable and need
# nothing.
# ─────────────────────────────────────────────────────────────────────────────

_ROWID_FTS_TABLES = ("chunks_fts", "episodes_fts", "aggregation_nodes_fts")


def resync_rowid_shadows(conn: sqlite3.Connection) -> None:
    """Rebuild every index keyed on an implicit (renumberable) rowid from its
    content table's CURRENT rowids. Must be called immediately after VACUUM;
    also the repair step for stores VACUUMed by earlier releases. Idempotent,
    and cheap next to the VACUUM that makes it necessary."""
    # FTS5's external-content `rebuild` command indiscriminately indexes every
    # chunks row.  Coverage artifacts are durable source storage, not search
    # documents, so rebuild that table explicitly from extraction rows only.
    with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
        conn.execute(
            "INSERT INTO chunks_fts(rowid, text) "
            "SELECT rowid, text FROM chunks WHERE chunk_kind = 'extraction'"
        )
    with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
        conn.execute(
            "INSERT INTO message_coverage_fts(message_coverage_fts) "
            "VALUES('delete-all')"
        )
        conn.execute(
            "INSERT INTO message_coverage_fts(rowid, content) "
            "SELECT rowid, json_extract(text, '$.content') FROM chunks "
            "WHERE chunk_kind = 'coverage' AND json_valid(text) "
            "AND json_type(text, '$.content') = 'text'"
        )
    # Like coverage chunks, unpublished episode generations must stay out of
    # the physical FTS corpus: even filtered result rows would otherwise alter
    # BM25 IDF/ranking. Rebuild episode postings from the publication marker.
    with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
        conn.execute("INSERT INTO episodes_fts(episodes_fts) VALUES('delete-all')")
        conn.execute(
            "INSERT INTO episodes_fts(rowid, title, summary) "
            "SELECT e.rowid, e.title, e.summary FROM episodes e "
            "JOIN sessions s ON s.id = e.session_id "
            "WHERE e.digest_generation IS NULL "
            "OR e.digest_generation = s.digest_published_generation"
        )
    for fts in (
        name for name in _ROWID_FTS_TABLES
        if name not in {"chunks_fts", "episodes_fts"}
    ):
        with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
            conn.execute(f"INSERT INTO {fts}({fts}) VALUES('rebuild')")
    if not _load_vec_extension(conn):
        return
    dim_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
    ).fetchone()
    if not dim_row:
        return
    try:
        dim = int(dim_row["value"])
    except (TypeError, ValueError, OverflowError):
        log.warning("rowid shadow resync skipped: invalid vec_dim metadata")
        return
    model_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'vec_model'"
    ).fetchone()
    model = (
        str(model_row["value"])
        if model_row is not None and isinstance(model_row["value"], str)
        and model_row["value"]
        else None
    )
    for table, backfill in (
        ("vec_chunks", _backfill_vec),
        ("vec_episodes", _backfill_vec_episodes),
    ):
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            _ensure_vec_table_named(conn, table, dim)
            backfill(conn, dim, model=model)
    log.info("rowid shadows resynced (fts rebuilt, vec_chunks/vec_episodes refilled)")


def vec_episodes_aligned(conn: sqlite3.Connection, sample: int = 8) -> bool:
    """Cheap probe: do vec_episodes rows still hold the vectors of the episodes
    whose rowids they sit at? Compares the stored blob against the packed
    mirror vector for up to `sample` episodes drawn from both ends of the rowid
    range (a renumber shifts everything above the first closed gap, so the
    newest rows drift the most). Returns True when unverifiable (extension or
    table absent, no embedded episodes) so callers only act on a proven
    mismatch."""
    if not _load_vec_extension(conn) or not has_vec_table(conn, table="vec_episodes"):
        return True
    dim_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
    ).fetchone()
    if not dim_row:
        return True
    try:
        dim = int(dim_row["value"])
    except (TypeError, ValueError, OverflowError):
        log.warning("vec_episodes alignment unverifiable: invalid vec_dim metadata")
        return True
    if dim <= 0:
        log.warning("vec_episodes alignment unverifiable: invalid vec_dim metadata")
        return True
    half = max(1, sample // 2)
    try:
        rows = conn.execute(
            """
            SELECT * FROM (
                SELECT e.rowid AS rid, em.vector_json, em.model, em.dim
                FROM episodes e
                JOIN sessions s ON s.id = e.session_id
                JOIN episode_embeddings em ON em.episode_id = e.id
                WHERE e.digest_generation IS NULL
                   OR e.digest_generation = s.digest_published_generation
                ORDER BY e.rowid ASC LIMIT ?
            )
            UNION
            SELECT * FROM (
                SELECT e.rowid AS rid, em.vector_json, em.model, em.dim
                FROM episodes e
                JOIN sessions s ON s.id = e.session_id
                JOIN episode_embeddings em ON em.episode_id = e.id
                WHERE e.digest_generation IS NULL
                   OR e.digest_generation = s.digest_published_generation
                ORDER BY e.rowid DESC LIMIT ?
            )
            """,
            (half, half),
        ).fetchall()
        model_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='vec_model'"
        ).fetchone()
        model = model_row["value"] if model_row is not None else None
        for r in rows:
            if r["dim"] != dim or (model is not None and r["model"] != model):
                return False
            vec = _finite_vec(decode_vector(r["vector_json"]), dim)
            if vec is None:
                return False
            stored = conn.execute(
                "SELECT embedding FROM vec_episodes WHERE rowid = ?", (r["rid"],)
            ).fetchone()
            if stored is None or bytes(stored["embedding"]) != _pack_vector(vec):
                return False
    except (sqlite3.OperationalError, json.JSONDecodeError, TypeError, ValueError):
        return True    # unverifiable ≠ misaligned; never resync on a probe error
    return True


def heal_rowid_shadows(conn: sqlite3.Connection) -> bool:
    """Probe vec_episodes alignment and, on a proven mismatch, resync every
    rowid shadow. Returns True when a repair ran. Called by the dream runner
    before aggregation so stores skewed by pre-fix VACUUMs heal on their next
    dream instead of their next VACUUM."""
    if vec_episodes_aligned(conn):
        return False
    log.warning(
        "vec_episodes misaligned with episodes rowids (post-VACUUM renumber); "
        "resyncing all rowid shadows"
    )
    resync_rowid_shadows(conn)
    return True


def vec_search(
    conn: sqlite3.Connection,
    query_vector: list[float],
    top_k: int,
    *,
    table: str = "vec_chunks",
) -> list[tuple[int, float]]:
    if table not in _VEC_TABLES:
        raise ValueError(f"unknown vec table: {table}")
    if not _load_vec_extension(conn):
        return []
    try:
        rows = conn.execute(
            f"""
            SELECT rowid, distance
            FROM {table}
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (_pack_vector(query_vector), top_k),
        ).fetchall()
        return [(int(r["rowid"]), float(r["distance"])) for r in rows]
    except (sqlite3.OperationalError, TypeError):
        return []


def has_vec_table(conn: sqlite3.Connection, table: str = "vec_chunks") -> bool:
    if table not in _VEC_TABLES:
        raise ValueError(f"unknown vec table: {table}")
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    return int(row["value"]) if row else 0


def backfill_entity_mentions(conn: sqlite3.Connection) -> None:
    """Idempotent backfill: populate entity_mentions from existing chunks if empty.

    No-op if the table already has rows or if there are no chunks.
    """
    has_mentions = conn.execute(
        "SELECT 1 FROM entity_mentions LIMIT 1"
    ).fetchone()
    if has_mentions:
        return
    chunk_count = conn.execute(
        "SELECT COUNT(*) AS c FROM chunks WHERE chunk_kind = 'extraction'"
    ).fetchone()["c"]
    if not chunk_count:
        return

    from hymem.dreaming.mentions import index_chunk_mentions

    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE chunk_kind = 'extraction'"
    ).fetchall()
    total = 0
    for row in rows:
        total += index_chunk_mentions(conn, row["id"], row["text"])
    log.info("backfilled entity_mentions: chunks=%d mentions=%d", len(rows), total)
