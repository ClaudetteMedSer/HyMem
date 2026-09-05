"""Tests for the file-based schema migration runner (db._run_migrations and
its helpers). Migrations live as ``NNN_*.sql`` under hymem/core/migrations/;
the runner applies any whose version exceeds the DB's schema_version and is
idempotent against a fresh schema.sql database.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from hymem import portability
from hymem.core import db as core_db


def _cols(conn, table) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _has_table(conn, name) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _downgrade_fact_domain_to_v45(conn: sqlite3.Connection) -> None:
    """Turn a current empty fact domain into the exact pre-v46 table shape.

    The rest of the database stays current so the fixture exercises v46's real
    cross-table guards instead of a hand-written approximation of the large
    v45 schema.  This is test setup only; production downgrades are unsupported.
    """

    fact_tables = {
        "fact_extraction_outcomes",
        "fact_extraction_revisions",
        "fact_extraction_source_occurrences",
        "narrative_facts",
        "narrative_fact_lifecycle",
    }
    triggers = conn.execute(
        "SELECT name,sql,tbl_name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()
    for row in triggers:
        sql = str(row["sql"] or "")
        if (
            row["tbl_name"] in fact_tables
            or "facts_cursor_" in sql
            or "facts_retry_" in sql
            or "facts_quarantined" in sql
        ):
            # Names originate in sqlite_master, not external input.
            conn.execute(f'DROP TRIGGER IF EXISTS "{row["name"]}"')

    conn.execute("DROP TABLE IF EXISTS narrative_facts_fts")
    conn.execute("DROP TABLE narrative_fact_lifecycle")
    conn.execute("DROP TABLE narrative_fact_embeddings")
    conn.execute("DROP TABLE fact_extraction_source_occurrences")
    conn.execute("DROP TABLE fact_extraction_revisions")
    conn.execute("DROP TABLE narrative_facts")
    conn.execute("DROP TABLE fact_extraction_outcomes")

    conn.executescript(
        """
        CREATE TABLE narrative_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            start_message_id INTEGER NOT NULL,
            end_message_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            fact_date TEXT,
            entities TEXT NOT NULL DEFAULT '[]',
            prompt_version TEXT NOT NULL,
            valid_at TEXT,
            invalid_at TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (session_id, start_message_id, text)
        );
        CREATE INDEX idx_narrative_facts_session
            ON narrative_facts(session_id);
        CREATE VIRTUAL TABLE narrative_facts_fts USING fts5(
            text,
            content='narrative_facts',
            content_rowid='id',
            tokenize='porter unicode61'
        );
        CREATE TRIGGER narrative_facts_fts_insert
        AFTER INSERT ON narrative_facts BEGIN
            INSERT INTO narrative_facts_fts(rowid,text)
            VALUES (new.id,new.text);
        END;
        CREATE TRIGGER narrative_facts_fts_delete
        AFTER DELETE ON narrative_facts BEGIN
            INSERT INTO narrative_facts_fts(narrative_facts_fts,rowid,text)
            VALUES ('delete',old.id,old.text);
        END;
        CREATE TRIGGER narrative_facts_fts_update
        AFTER UPDATE ON narrative_facts BEGIN
            INSERT INTO narrative_facts_fts(narrative_facts_fts,rowid,text)
            VALUES ('delete',old.id,old.text);
            INSERT INTO narrative_facts_fts(rowid,text)
            VALUES (new.id,new.text);
        END;
        CREATE TABLE narrative_fact_embeddings (
            fact_id INTEGER PRIMARY KEY REFERENCES narrative_facts(id)
                ON DELETE CASCADE,
            vector_json TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            text_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    for column in (
        "facts_cursor_message_id",
        "facts_cursor_partial_message_id",
        "facts_cursor_offset",
        "facts_cursor_prompt_version",
        "facts_retry_count",
        "facts_retry_config_version",
        "facts_quarantined",
    ):
        conn.execute(f"ALTER TABLE sessions DROP COLUMN {column}")
    conn.execute(
        "UPDATE schema_meta SET value='45' WHERE key='schema_version'"
    )


# --- discovery -------------------------------------------------------------


def test_discover_migrations_is_contiguous_and_sorted():
    versions = [v for v, _ in core_db._discover_migrations()]
    assert versions == sorted(versions)
    # Migrations start at v2 (v1 is the schema.sql baseline) and reach the
    # version the code expects.
    assert versions[0] == 2
    assert versions[-1] == core_db.EXPECTED_SCHEMA_VERSION
    assert versions == list(range(2, core_db.EXPECTED_SCHEMA_VERSION + 1))


# --- statement splitter ----------------------------------------------------


def test_split_keeps_trigger_body_intact():
    """A CREATE TRIGGER ... BEGIN ...; ...; END; block is one statement, even
    though its body contains semicolons."""
    script = """
    -- a comment line
    ALTER TABLE t ADD COLUMN c TEXT;
    CREATE TRIGGER IF NOT EXISTS trg AFTER UPDATE ON t BEGIN
        INSERT INTO log VALUES ('a');
        INSERT INTO log VALUES ('b');
    END;
    """
    stmts = core_db._split_sql_statements(script)
    assert len(stmts) == 2
    assert stmts[0].startswith("ALTER TABLE")
    assert stmts[1].startswith("CREATE TRIGGER")
    # Both inner inserts survive inside the single trigger statement.
    assert stmts[1].count("INSERT INTO log") == 2
    assert stmts[1].rstrip().endswith("END;")


# --- fresh database --------------------------------------------------------


def test_fresh_db_lands_at_expected_version(tmp_path: Path):
    conn = core_db.connect(tmp_path / "fresh.sqlite")
    core_db.initialize(conn)
    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    # schema.sql already creates these; the no-op migrations didn't clobber them.
    assert "status" in _cols(conn, "procedures")
    assert "temporal_scope" in _cols(conn, "kg_evidence")
    conn.close()


def test_rerunning_migrations_is_a_noop(tmp_path: Path):
    """Applying migrations against an already-current DB raises nothing and
    leaves the version unchanged."""
    conn = core_db.connect(tmp_path / "rerun.sqlite")
    core_db.initialize(conn)
    v1 = core_db.schema_version(conn)
    core_db._run_migrations(conn)  # second pass
    assert core_db.schema_version(conn) == v1
    conn.close()


def test_fresh_schema_has_message_vector_freshness_and_composite_authority(
    tmp_path: Path,
):
    """The bootstrap schema must carry the complete v44 shape itself."""
    conn = core_db.connect(tmp_path / "fresh-v44-shape.sqlite")
    core_db.initialize(conn)
    try:
        assert "text_hash" in _cols(conn, "chunk_embeddings")
        assert _cols(conn, "message_embeddings") == {
            "message_id",
            "source_coverage_chunk_id",
            "source_coverage_version",
            "text_hash",
            "vector_json",
            "model",
            "dim",
            "created_at",
        }
        coverage_fk = [
            row
            for row in conn.execute("PRAGMA foreign_key_list(message_embeddings)")
            if row["table"] == "message_retention_coverage"
        ]
        assert [
            (row["from"], row["to"], row["on_delete"])
            for row in sorted(coverage_fk, key=lambda row: row["seq"])
        ] == [
            ("message_id", "message_id", "CASCADE"),
            ("source_coverage_chunk_id", "chunk_id", "CASCADE"),
            ("source_coverage_version", "coverage_version", "CASCADE"),
        ]
    finally:
        conn.close()


def test_public_initialize_upgrades_populated_v43_embeddings_idempotently(
    tmp_path: Path,
):
    """A real old embedding row survives v43→v44 and gains nullable freshness."""
    from hymem.dreaming.lossless import materialize_message_coverage

    db_path = tmp_path / "populated-v43-embeddings.sqlite"
    conn = core_db.connect(db_path)
    core_db.initialize(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('legacy-session')")
    conn.execute(
        "INSERT INTO messages(id,session_id,role,content) "
        "VALUES (7,'legacy-session','user','legacy semantic source')"
    )
    with core_db.transaction(conn):
        assert materialize_message_coverage(conn, "legacy-session") == 1
    coverage = conn.execute(
        "SELECT chunk_id FROM message_retention_coverage WHERE message_id=7"
    ).fetchone()
    conn.execute(
        "INSERT INTO chunk_embeddings(chunk_id,vector_json,model,dim,text_hash) "
        "VALUES (?,?,?,?,?)",
        (coverage["chunk_id"], "[1.0,0.0,0.0]", "legacy-space", 3, "old-hash"),
    )

    # Recreate exactly the two pre-v44 embedding shapes while retaining the
    # rest of the fully migrated v43 domain and its populated coverage parent.
    conn.execute("DROP TABLE message_embeddings")
    conn.execute("ALTER TABLE chunk_embeddings DROP COLUMN text_hash")
    conn.execute(
        "UPDATE schema_meta SET value='43' WHERE key='schema_version'"
    )
    conn.close()

    # Exercise the public startup path, including schema.sql-before-migrations.
    conn = core_db.connect(db_path)
    core_db.initialize(conn)
    try:
        assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION == 46
        assert "text_hash" in _cols(conn, "chunk_embeddings")
        row = conn.execute(
            "SELECT chunk_id,vector_json,model,dim,text_hash "
            "FROM chunk_embeddings"
        ).fetchone()
        assert tuple(row) == (
            coverage["chunk_id"], "[1.0,0.0,0.0]", "legacy-space", 3, None,
        )
        assert _has_table(conn, "message_embeddings")
        coverage_fk = [
            row
            for row in conn.execute("PRAGMA foreign_key_list(message_embeddings)")
            if row["table"] == "message_retention_coverage"
        ]
        assert [
            (row["from"], row["to"])
            for row in sorted(coverage_fk, key=lambda row: row["seq"])
        ] == [
            ("message_id", "message_id"),
            ("source_coverage_chunk_id", "chunk_id"),
            ("source_coverage_version", "coverage_version"),
        ]
    finally:
        conn.close()

    # A second process start must neither replay ALTER nor lose the old row.
    conn = core_db.connect(db_path)
    core_db.initialize(conn)
    try:
        assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION == 46
        assert conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] == 1
        assert conn.execute(
            "SELECT text_hash FROM chunk_embeddings"
        ).fetchone()[0] is None
    finally:
        conn.close()


# --- legacy database upgrade -----------------------------------------------


def test_initialize_on_existing_pre_v10_db_does_not_crash(tmp_path: Path):
    """Regression: initialize() runs schema.sql via executescript() BEFORE
    migrations. An existing `procedures` table predating the v10 `status`
    column is left untouched by CREATE TABLE IF NOT EXISTS, so a
    `CREATE INDEX ... ON procedures(status)` in schema.sql would crash it with
    "no such column: status". That index must live in migration 010 only.

    Exercises the *real* startup path (initialize), not just _run_migrations —
    the gap that let the bug ship.
    """
    conn = core_db.connect(tmp_path / "pre_v10.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '9');
        CREATE TABLE procedures (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            steps TEXT NOT NULL DEFAULT '[]',
            triggers TEXT NOT NULL DEFAULT '[]',
            entities_involved TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # Must not raise on the schema.sql pass, then migrate forward.
    core_db.initialize(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert "status" in _cols(conn, "procedures")
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='index' AND name='idx_procedures_status'"
    ).fetchone()
    assert idx is not None, "migration 010 must create the status index"
    conn.close()


def test_legacy_v1_db_upgrades_and_gains_columns(tmp_path: Path):
    """A pre-migration (v1) database carrying only the original tables is
    walked forward to the latest version, picking up every additive column,
    table, and the trigger-bearing v8 migration."""
    conn = core_db.connect(tmp_path / "legacy.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '1');
        CREATE TABLE sessions(id TEXT PRIMARY KEY);
        CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT NOT NULL, content TEXT NOT NULL);
        CREATE TABLE chunks(id TEXT PRIMARY KEY);
        CREATE TABLE kg_evidence(id INTEGER PRIMARY KEY, edge_id INTEGER,
            chunk_id TEXT, polarity INTEGER);
        CREATE TABLE knowledge_graph(id INTEGER PRIMARY KEY,
            subject_canonical TEXT, predicate TEXT, object_canonical TEXT,
            first_seen TIMESTAMP, last_seen TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active');
        CREATE TABLE dream_runs(id INTEGER PRIMARY KEY);
        CREATE TABLE episodes(id TEXT PRIMARY KEY, title TEXT, summary TEXT);
        CREATE VIRTUAL TABLE episodes_fts USING fts5(title, summary,
            content=episodes, content_rowid=rowid);
        CREATE TABLE procedures(id TEXT PRIMARY KEY, confidence REAL DEFAULT 1.0);
        """
    )

    core_db._run_migrations(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert "temporal_scope" in _cols(conn, "kg_evidence")  # v2
    assert "summary" in _cols(conn, "sessions")             # v3
    assert "derived" in _cols(conn, "knowledge_graph")      # v4
    assert _has_table(conn, "extraction_feedback")          # v5
    assert _has_table(conn, "edge_embeddings")              # v6
    assert _has_table(conn, "token_overlap_index")          # v7
    assert _has_table(conn, "episode_embeddings")           # v8 (alongside trigger)
    assert _has_table(conn, "entity_properties")            # v9
    assert "status" in _cols(conn, "procedures")            # v10
    assert "source_role" in _cols(conn, "kg_evidence")      # v11
    assert _has_table(conn, "messages_fts")                 # v13
    assert _has_table(conn, "temporal_mentions")            # v14
    assert "valid_at" in _cols(conn, "knowledge_graph")     # v15
    assert "invalid_at" in _cols(conn, "knowledge_graph")   # v15
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='index' AND name='idx_kg_validity'"
    ).fetchone()
    assert idx is not None, "migration 015 must create the validity index"
    conn.close()


def test_v15_backfills_validity_interval(tmp_path: Path):
    """Migration 015 seeds valid_at from first_seen for existing edges and
    closes invalid_at from last_seen for already-superseded ones, so pre-v15
    rows land with a populated (approximate) interval."""
    conn = core_db.connect(tmp_path / "v14.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '14');
        CREATE TABLE sessions(id TEXT PRIMARY KEY);
        CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT NOT NULL, content TEXT NOT NULL);
        CREATE TABLE dream_runs(id INTEGER PRIMARY KEY);
        CREATE TABLE knowledge_graph(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_canonical TEXT, predicate TEXT, object_canonical TEXT,
            first_seen TIMESTAMP, last_seen TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active');
        INSERT INTO knowledge_graph
            (subject_canonical, predicate, object_canonical, first_seen, last_seen, status)
        VALUES
            ('a','uses','b','2024-01-01','2024-02-01','active'),
            ('a','uses','c','2023-01-01','2023-06-01','retracted');
        """
    )

    core_db._run_migrations(conn)  # v14 -> only migration 015 applies

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    active = conn.execute(
        "SELECT valid_at, invalid_at FROM knowledge_graph WHERE object_canonical='b'"
    ).fetchone()
    superseded = conn.execute(
        "SELECT valid_at, invalid_at FROM knowledge_graph WHERE object_canonical='c'"
    ).fetchone()
    assert active["valid_at"] == "2024-01-01"
    assert active["invalid_at"] is None          # still valid
    assert superseded["valid_at"] == "2023-01-01"
    assert superseded["invalid_at"] == "2023-06-01"  # closed from last_seen
    conn.close()


def test_v21_rebuilds_predicate_check_preserving_data_and_fk(tmp_path: Path):
    """Migration 021 rebuilds knowledge_graph to widen the predicate CHECK
    (adding the v9 personal-life predicates). The rebuild must preserve every
    edge with its id (so kg_evidence.edge_id stays valid), keep enforcing the
    vocabulary (just wider), and restore FK enforcement after the table swap."""
    import sqlite3

    import pytest

    conn = core_db.connect(tmp_path / "v20.sqlite")
    # A pre-021 store: the OLD (narrow) predicate CHECK, a populated edge with
    # evidence counts and a validity stamp, and a kg_evidence child FK row.
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '20');
        -- a real v20 store carries dream_runs (v22 ALTERs it on the way up)
        CREATE TABLE dream_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP NOT NULL);
        -- ...and sessions (v24 ALTERs it on the way up)
        CREATE TABLE sessions(
            id TEXT PRIMARY KEY,
            digested_prompt_version TEXT,
            profile_prompt_version TEXT);
        CREATE TABLE knowledge_graph(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_canonical TEXT NOT NULL,
            predicate TEXT NOT NULL CHECK (predicate IN ('uses','prefers','configured_with')),
            object_canonical TEXT NOT NULL,
            pos_evidence INTEGER NOT NULL DEFAULT 0,
            neg_evidence INTEGER NOT NULL DEFAULT 0,
            first_seen TIMESTAMP, last_seen TIMESTAMP, last_reinforced TIMESTAMP,
            valid_at TIMESTAMP, invalid_at TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active', derived BOOLEAN NOT NULL DEFAULT 0,
            UNIQUE(subject_canonical, predicate, object_canonical));
        CREATE TABLE kg_evidence(id INTEGER PRIMARY KEY,
            edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
            chunk_id TEXT, polarity INTEGER);
        INSERT INTO knowledge_graph
            (id, subject_canonical, predicate, object_canonical, pos_evidence, valid_at, status)
            VALUES (42, 'project', 'configured_with', '78_percent', 3, '2024-03-20', 'active');
        INSERT INTO kg_evidence(id, edge_id, chunk_id, polarity) VALUES (7, 42, 'c1', 1);
        """
    )

    core_db._run_migrations(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    # Edge preserved with its id and payload; the FK child still resolves to it.
    edge = conn.execute(
        "SELECT id, predicate, object_canonical, pos_evidence, valid_at "
        "FROM knowledge_graph WHERE id = 42"
    ).fetchone()
    assert (edge["predicate"], edge["object_canonical"], edge["pos_evidence"], edge["valid_at"]) == (
        "configured_with", "78_percent", 1, "2024-03-20"
    )
    # The two counter units with no source row are retained for audit, but v36
    # conservatively excludes them from confidence instead of guessing origin.
    legacy = conn.execute(
        "SELECT evidence_weight, counts_toward_confidence "
        "FROM kg_evidence_signals WHERE edge_id = 42 AND polarity = 1"
    ).fetchone()
    assert (legacy["evidence_weight"], legacy["counts_toward_confidence"]) == (2, 0)
    assert conn.execute("SELECT edge_id FROM kg_evidence WHERE id = 7").fetchone()["edge_id"] == 42
    # The widened vocabulary admits a personal-life predicate…
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
        "VALUES ('user', 'owns', 'ford_f_150')"
    )
    # …but is still a closed CHECK, not removed.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
            "VALUES ('x', 'made_up', 'y')"
        )
    # FK enforcement restored after the swap (PRAGMA foreign_keys flipped back on).
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO kg_evidence(id, edge_id, chunk_id, polarity) VALUES (9, 99999, 'c2', 1)"
        )
    conn.close()


def test_v13_backfills_messages_fts_with_role_filter(tmp_path: Path):
    """Migration 013 backfills already-logged user/assistant turns into
    messages_fts and excludes tool/system turns — matching the live trigger's
    role guard so the index is consistent however a row arrived."""
    conn = core_db.connect(tmp_path / "v12.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '12');
        CREATE TABLE sessions(id TEXT PRIMARY KEY);
        CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT NOT NULL, content TEXT NOT NULL);
        CREATE TABLE dream_runs(id INTEGER PRIMARY KEY);
        CREATE TABLE knowledge_graph(id INTEGER PRIMARY KEY,
            subject_canonical TEXT, predicate TEXT, object_canonical TEXT,
            first_seen TIMESTAMP, last_seen TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active');
        INSERT INTO sessions(id) VALUES ('s');
        INSERT INTO messages(session_id, role, content) VALUES
            ('s','user','postgres is the primary datastore'),
            ('s','assistant','noted: postgres with pgbouncer'),
            ('s','tool','postgres tool dump postgres postgres');
        """
    )
    assert not _has_table(conn, "messages_fts")

    core_db._run_migrations(conn)  # from v12: migration 013 backfills messages_fts

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert _has_table(conn, "messages_fts")
    roles = [
        r["role"]
        for r in conn.execute(
            "SELECT m.role FROM messages_fts JOIN messages m "
            "ON m.id = messages_fts.rowid WHERE messages_fts MATCH ? ORDER BY m.id",
            ('"postgres"',),
        ).fetchall()
    ]
    assert roles == ["user", "assistant"]  # tool turn not backfilled
    conn.close()


def test_v14_adds_temporal_mentions_table(tmp_path: Path):
    """Migration 014 adds the temporal_mentions table + date index to a v13 DB.

    No backfill is expected (mentions are populated by the dream cycle's
    per-message pass), so the assertion is purely that the table and its
    normalized_date index now exist and the version advanced."""
    conn = core_db.connect(tmp_path / "v13.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '13');
        CREATE TABLE sessions(id TEXT PRIMARY KEY);
        CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT NOT NULL, content TEXT NOT NULL);
        CREATE TABLE dream_runs(id INTEGER PRIMARY KEY);
        CREATE TABLE knowledge_graph(id INTEGER PRIMARY KEY,
            subject_canonical TEXT, predicate TEXT, object_canonical TEXT,
            first_seen TIMESTAMP, last_seen TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active');
        """
    )
    assert not _has_table(conn, "temporal_mentions")

    core_db._run_migrations(conn)  # version 13 -> only migration 014 applies

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert _has_table(conn, "temporal_mentions")
    assert "normalized_date" in _cols(conn, "temporal_mentions")
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='index' AND name='idx_temporal_mentions_date'"
    ).fetchone()
    assert idx is not None, "migration 014 must create the normalized_date index"
    conn.close()


def test_v36_repairs_counters_and_resolves_legacy_polarity_collision(tmp_path: Path):
    """v36 keeps audit data but trusts only unique source-backed evidence."""
    import sqlite3

    import pytest

    conn = core_db.connect(tmp_path / "v35-evidence.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '35');
        CREATE TABLE sessions(id TEXT PRIMARY KEY);
        CREATE TABLE chunks(id TEXT PRIMARY KEY);
        CREATE TABLE knowledge_graph(
            id INTEGER PRIMARY KEY,
            subject_canonical TEXT,
            predicate TEXT,
            object_canonical TEXT,
            pos_evidence INTEGER NOT NULL DEFAULT 0,
            neg_evidence INTEGER NOT NULL DEFAULT 0,
            derived BOOLEAN NOT NULL DEFAULT 0
        );
        CREATE TABLE kg_evidence(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
            surface_subject TEXT,
            surface_object TEXT,
            value_text TEXT,
            value_numeric REAL,
            value_unit TEXT,
            temporal_scope TEXT,
            source_role TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(edge_id, chunk_id, polarity)
        );
        INSERT INTO chunks(id) VALUES ('c1'), ('c2'), ('c3');
        INSERT INTO knowledge_graph VALUES (1, 'a', 'uses', 'b', 4, 0, 0);
        INSERT INTO kg_evidence(
            id, edge_id, chunk_id, polarity, surface_subject, surface_object,
            value_text, value_numeric, value_unit, temporal_scope, source_role,
            extracted_at
        ) VALUES (
            10, 1, 'c1', 1, 'A', 'B', 'seventy-eight', 78.0, 'percent',
            '2024-Q1', 'user', '2024-03-20 12:34:56'
        );
        INSERT INTO knowledge_graph VALUES (2, 'c', 'uses', 'd', 1, 1, 0);
        INSERT INTO kg_evidence(id, edge_id, chunk_id, polarity)
            VALUES (20, 2, 'c2', 1);
        INSERT INTO kg_evidence(id, edge_id, chunk_id, polarity)
            VALUES (21, 2, 'c2', -1);
        INSERT INTO knowledge_graph VALUES (3, 'e', 'depends_on', 'f', 1, 0, 1);
        INSERT INTO knowledge_graph VALUES (4, 'g', 'uses', 'h', 0, 0, 0);
        """
    )

    core_db._run_migrations(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    inflated = conn.execute(
        "SELECT pos_evidence, neg_evidence FROM knowledge_graph WHERE id = 1"
    ).fetchone()
    assert (inflated["pos_evidence"], inflated["neg_evidence"]) == (1, 0)
    quarantine = conn.execute(
        """SELECT evidence_weight, counts_toward_confidence
           FROM kg_evidence_signals
           WHERE edge_id = 1 AND signal_kind = 'legacy_unattributed'"""
    ).fetchone()
    assert (quarantine["evidence_weight"], quarantine["counts_toward_confidence"]) == (
        3,
        0,
    )
    preserved = conn.execute(
        """SELECT id, surface_subject, surface_object, value_text,
                  value_numeric, value_unit, temporal_scope, source_role,
                  extracted_at, evidence_kind, evidence_weight, weight_source
           FROM kg_evidence WHERE edge_id = 1"""
    ).fetchone()
    assert tuple(preserved) == (
        10,
        "A",
        "B",
        "seventy-eight",
        78.0,
        "percent",
        "2024-Q1",
        None,  # v40 discards guessed chunk-level speaker provenance
        "2024-03-20 12:34:56",
        "extraction",
        1,
        "legacy_default",
    )

    # Last inserted interpretation wins; the old + row is not an independent
    # proof after the same source changed polarity.
    flipped = conn.execute(
        "SELECT pos_evidence, neg_evidence FROM knowledge_graph WHERE id = 2"
    ).fetchone()
    assert (flipped["pos_evidence"], flipped["neg_evidence"]) == (0, 1)
    remaining = conn.execute(
        "SELECT id, polarity FROM kg_evidence WHERE edge_id = 2"
    ).fetchall()
    assert [(row["id"], row["polarity"]) for row in remaining] == [(21, -1)]

    # Computed edges are outside the observed-evidence invariant.
    assert conn.execute(
        "SELECT pos_evidence FROM knowledge_graph WHERE id = 3"
    ).fetchone()["pos_evidence"] == 1

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO kg_evidence(edge_id, chunk_id, polarity) VALUES (2, 'c2', 1)"
        )

    # The legacy table-level UNIQUE(edge, chunk, polarity) autoindex must be
    # gone after the rebuild: distinct evidence classes may legitimately share
    # a source and polarity, while each class remains independently idempotent.
    with core_db.evidence_mutation(conn):
        conn.execute(
            """INSERT INTO kg_evidence(
                   edge_id, chunk_id, polarity, evidence_kind,
                   evidence_weight, weight_source
               ) VALUES (4, 'c3', 1, 'extraction', 1, 'test')"""
        )
        conn.execute(
            """INSERT INTO kg_evidence(
                   edge_id, chunk_id, polarity, evidence_kind,
                   evidence_weight, weight_source
               ) VALUES (4, 'c3', 1, 'reinforcement', 1, 'test')"""
        )
    kinds = conn.execute(
        """SELECT evidence_kind
           FROM kg_evidence
           WHERE edge_id = 4 AND chunk_id = 'c3'
           ORDER BY evidence_kind"""
    ).fetchall()
    assert [row["evidence_kind"] for row in kinds] == [
        "extraction",
        "reinforcement",
    ]
    foreign_keys = {
        (row["from"], row["table"], row["to"], row["on_delete"])
        for row in conn.execute("PRAGMA foreign_key_list(kg_evidence)")
    }
    assert foreign_keys == {
        ("edge_id", "knowledge_graph", "id", "CASCADE"),
        ("chunk_id", "chunks", "id", "RESTRICT"),
        (
            "source_message_id",
            "message_retention_coverage",
            "message_id",
            "RESTRICT",
        ),
        (
            "source_coverage_chunk_id",
            "message_retention_coverage",
            "chunk_id",
            "RESTRICT",
        ),
        (
            "source_coverage_version",
            "message_retention_coverage",
            "coverage_version",
            "RESTRICT",
        ),
    }
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_public_initialize_upgrades_populated_v35_before_latest_guards(tmp_path: Path):
    """Latest bootstrap DDL must not validate v38/v40 triggers too early.

    This is the public ``initialize`` path with real pre-v36 evidence and a
    pre-v37 chunks shape, not the sparse ``_run_migrations`` fixture above.
    """
    conn = core_db.connect(tmp_path / "populated-v35-public.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '35');
        CREATE TABLE sessions(id TEXT PRIMARY KEY, started_at TIMESTAMP,
            ended_at TIMESTAMP, summary TEXT);
        CREATE TABLE messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL, content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE chunks(
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            start_message_id INTEGER NOT NULL,
            end_message_id INTEGER NOT NULL,
            salience_reason TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE knowledge_graph(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_canonical TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_canonical TEXT NOT NULL,
            pos_evidence INTEGER NOT NULL DEFAULT 0,
            neg_evidence INTEGER NOT NULL DEFAULT 0,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_reinforced TIMESTAMP,
            valid_at TIMESTAMP,
            invalid_at TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active',
            derived BOOLEAN NOT NULL DEFAULT 0,
            UNIQUE(subject_canonical,predicate,object_canonical)
        );
        CREATE TABLE kg_evidence(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            polarity INTEGER NOT NULL,
            surface_subject TEXT, surface_object TEXT,
            value_text TEXT, value_numeric REAL, value_unit TEXT,
            temporal_scope TEXT, source_role TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(edge_id,chunk_id,polarity)
        );
        INSERT INTO sessions(id,started_at) VALUES ('legacy','2025-01-01');
        INSERT INTO messages(id,session_id,role,content,created_at)
            VALUES (1,'legacy','user','App uses Redis','2025-01-01');
        INSERT INTO chunks(id,session_id,start_message_id,end_message_id,
            salience_reason,text,created_at)
            VALUES ('legacy-chunk','legacy',1,1,'salient',
                    'user: App uses Redis','2025-01-01');
        INSERT INTO knowledge_graph(id,subject_canonical,predicate,
            object_canonical,pos_evidence,neg_evidence,status,derived)
            VALUES (1,'app','uses','redis',3,0,'active',0);
        INSERT INTO kg_evidence(edge_id,chunk_id,polarity,source_role)
            VALUES (1,'legacy-chunk',1,'user');
        """
    )

    core_db.initialize(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert "chunk_kind" in _cols(conn, "chunks")
    assert "interpretation_key" in _cols(conn, "kg_evidence")
    assert conn.execute(
        "SELECT pos_evidence FROM knowledge_graph WHERE id=1"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM kg_evidence_signals WHERE edge_id=1"
    ).fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_public_v39_singleton_upgrade_and_stale_stamp_replay_preserve_source(
    tmp_path: Path,
):
    """A real v39 singleton promotes exactly and v40 hook replay is lossless."""
    from hymem.dreaming.lossless import materialize_message_coverage

    path = tmp_path / "public-v39-singleton.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('legacy-source')")
    cur = conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at) "
        "VALUES ('legacy-source','user','App uses Redis','2026-01-01T01:00:00+01:00')"
    )
    message_id = int(cur.lastrowid)
    with core_db.transaction(conn):
        materialize_message_coverage(conn, "legacy-source")
    conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES "
        "('legacy-extraction','legacy-source',?,?, 'salient',"
        "'user: App uses Redis','extraction')",
        (message_id, message_id),
    )
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
        "pos_evidence,neg_evidence,status,derived) "
        "VALUES ('app','uses','redis',1,0,'active',0)"
    )
    edge_id = int(conn.execute(
        "SELECT id FROM knowledge_graph WHERE subject_canonical='app'"
    ).fetchone()[0])

    # Recreate the released v39 evidence shape while retaining its genuine v38
    # coverage artifact and old extraction row.
    with core_db.evidence_mutation(conn):
        # This fixture starts from a current database only as a convenient way
        # to seed the released v39 rows. A genuine v39 store cannot contain
        # the v43 binding trigger (which intentionally names v40 tables).
        conn.execute("DROP TRIGGER IF EXISTS session_workspace_binding_guard")
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND "
            "(name LIKE 'kg_%' OR name LIKE 'chunk_source_manifest_%' "
            "OR name LIKE 'chunk_message_sources_%')"
        ).fetchall():
            conn.execute(f"DROP TRIGGER IF EXISTS {row['name']}")
        conn.execute("DROP TABLE IF EXISTS kg_lifecycle_dependencies")
        conn.execute("DROP TABLE IF EXISTS kg_edge_lifecycle")
        conn.execute("DROP TABLE IF EXISTS kg_claim_observations")
        conn.execute("DROP TABLE IF EXISTS chunk_message_sources")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executescript(
        """
        CREATE TABLE kg_evidence_v39(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            polarity INTEGER NOT NULL CHECK (polarity IN (-1,1)),
            surface_subject TEXT, surface_object TEXT,
            value_text TEXT, value_numeric REAL, value_unit TEXT,
            temporal_scope TEXT, source_role TEXT,
            evidence_kind TEXT NOT NULL DEFAULT 'extraction',
            evidence_weight INTEGER NOT NULL DEFAULT 1,
            weight_source TEXT NOT NULL DEFAULT 'legacy_default',
            extraction_prompt_version TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(
        "INSERT INTO kg_evidence_v39(edge_id,chunk_id,polarity,source_role,"
        "evidence_kind,evidence_weight,weight_source,extraction_prompt_version) "
        "VALUES (?, 'legacy-extraction', 1, 'user', 'extraction', 1, "
        "'legacy_default', 'v12')",
        (edge_id,),
    )
    conn.execute("DROP TABLE kg_evidence")
    conn.execute("ALTER TABLE kg_evidence_v39 RENAME TO kg_evidence")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        "UPDATE schema_meta SET value='39' WHERE key='schema_version'"
    )
    migration_40 = dict(core_db._discover_migrations())[40]
    with core_db.evidence_mutation(conn):
        core_db._apply_migration_sql(
            conn, migration_40.read_text(encoding="utf-8")
        )
    # Crash point 1: SQL RELEASE completed, but neither the Python promotion
    # hook nor schema-version stamp ran. The row is still conservative legacy.
    assert core_db.schema_version(conn) == 39
    assert conn.execute(
        "SELECT provenance_status FROM kg_evidence"
    ).fetchone()[0] == "legacy_unattributed"
    conn.close()

    conn = core_db.connect(path)
    core_db.initialize(conn)
    promoted = conn.execute(
        "SELECT provenance_status,source_message_id,source_session_id,source_role,"
        "source_created_at,source_event_at,is_current FROM kg_evidence"
    ).fetchone()
    assert tuple(promoted) == (
        "canonical", message_id, "legacy-source", "user",
        "2026-01-01T01:00:00+01:00", "2026-01-01T00:00:00.000Z", 1,
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM kg_claim_observations"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM kg_edge_lifecycle "
        "WHERE event_kind='claim_assertion'"
    ).fetchone()[0] == 1
    conn.execute("DELETE FROM messages WHERE id=?", (message_id,))
    before = {
        table: [tuple(row) for row in conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ).fetchall()]
        for table in (
            "kg_evidence", "kg_claim_observations", "kg_edge_lifecycle",
            "chunk_message_sources",
        )
    }
    conn.execute("UPDATE schema_meta SET value='39' WHERE key='schema_version'")
    core_db.initialize(conn)
    after = {
        table: [tuple(row) for row in conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ).fetchall()]
        for table in before
    }
    assert after == before
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_v37_adds_empty_content_bound_message_coverage_ledger(tmp_path: Path):
    """Legacy stores gain the safety ledger without guessing old coverage."""
    import sqlite3

    import pytest

    conn = core_db.connect(tmp_path / "v36-message-coverage.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '36');
        CREATE TABLE messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP
        );
        CREATE TABLE chunks(
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            start_message_id INTEGER NOT NULL,
            end_message_id INTEGER NOT NULL,
            text TEXT NOT NULL
        );
        INSERT INTO messages(id, session_id, role, content, created_at)
            VALUES (7, 'legacy_session', 'user', 'legacy source', '2026-01-01');
        INSERT INTO chunks VALUES (
            'legacy_chunk', 'legacy_session', 7, 7, 'lossless record'
        );
        """
    )

    core_db._run_migrations(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert _has_table(conn, "message_retention_coverage")
    assert _cols(conn, "message_retention_coverage") == {
        "message_id",
        "source_session_id",
        "source_role",
        "source_created_at",
        "chunk_id",
        "message_content_hash",
        "hash_version",
        "record_version",
        "coverage_version",
        "created_at",
    }
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM message_retention_coverage"
    ).fetchone()["c"] == 0, "legacy summaries/chunks must not be trusted implicitly"
    index = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' "
        "AND name='idx_message_retention_coverage_chunk'"
    ).fetchone()
    assert index is not None
    foreign_keys = {
        (row["from"], row["table"], row["to"], row["on_delete"])
        for row in conn.execute("PRAGMA foreign_key_list(message_retention_coverage)")
    }
    assert foreign_keys == {("chunk_id", "chunks", "id", "RESTRICT")}

    conn.execute(
        "INSERT INTO message_retention_coverage("
        "message_id, source_session_id, source_role, source_created_at, chunk_id, "
        "message_content_hash, hash_version, record_version, coverage_version) "
        "VALUES (7, 'legacy_session', 'user', '2026-01-01', 'legacy_chunk', "
        "'hash', 'hash-v1', 'record-v1', 'producer-v1')"
    )
    conn.execute("DELETE FROM messages WHERE id = 7")
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM message_retention_coverage"
    ).fetchone()["c"] == 1
    with pytest.raises(sqlite3.IntegrityError, match="raw source is absent"):
        conn.execute("DELETE FROM message_retention_coverage")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM chunks WHERE id = 'legacy_chunk'")
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_v38_migrates_sparse_coverage_without_inventing_a_frontier(tmp_path: Path):
    """Sparse v37 proofs are exact artifacts, not ordered-session coverage."""
    import sqlite3

    import pytest

    from hymem.core.message_records import (
        MESSAGE_CONTENT_HASH_VERSION,
        MESSAGE_RECORD_VERSION,
        encode_message_record,
        message_content_hash,
    )

    conn = core_db.connect(tmp_path / "v37-sparse-coverage.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '37');
        CREATE TABLE sessions(
            id TEXT PRIMARY KEY,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            summary TEXT,
            digested_prompt_version TEXT,
            profile_prompt_version TEXT,
            digested_message_id INTEGER,
            facts_message_id INTEGER,
            episodes_prompt_version TEXT
        );
        CREATE TABLE messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP
        );
        CREATE TABLE chunks(
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            start_message_id INTEGER NOT NULL,
            end_message_id INTEGER NOT NULL,
            salience_reason TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text, content='chunks', content_rowid='rowid'
        );
        CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
        END;
        CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text)
            VALUES ('delete', old.rowid, old.text);
        END;
        CREATE TABLE message_retention_coverage(
            message_id INTEGER NOT NULL,
            source_session_id TEXT NOT NULL,
            source_role TEXT NOT NULL,
            source_created_at TIMESTAMP,
            chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE RESTRICT,
            message_content_hash TEXT NOT NULL,
            hash_version TEXT NOT NULL,
            record_version TEXT NOT NULL,
            coverage_version TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (message_id, chunk_id, coverage_version)
        );
        CREATE TABLE episodes(
            id TEXT PRIMARY KEY,
            session_id TEXT,
            title TEXT,
            summary TEXT,
            participants TEXT NOT NULL DEFAULT '[]',
            start_message_id INTEGER,
            end_message_id INTEGER,
            outcome TEXT,
            key_entities TEXT NOT NULL DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO sessions(id, summary) VALUES ('legacy', 'Curated old history');
        INSERT INTO episodes(id, session_id, title, summary)
            VALUES ('legacy_episode', 'legacy', 'Legacy', 'legacyepisodetoken');
        """
    )
    for message_id, content in ((1, "first sparse proof"), (3, "third sparse proof")):
        record = encode_message_record(
            message_id=message_id, role="user", content=content
        )
        chunk_id = f"legacy_cov_{message_id}"
        conn.execute(
            "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
            "salience_reason, text) VALUES (?, 'legacy', ?, ?, 'legacy', ?)",
            (chunk_id, message_id, message_id, record),
        )
        conn.execute(
            "INSERT INTO message_retention_coverage("
            "message_id, source_session_id, source_role, chunk_id, "
            "message_content_hash, hash_version, record_version, coverage_version) "
            "VALUES (?, 'legacy', 'user', ?, ?, ?, ?, 'caller-v37')",
            (
                message_id,
                chunk_id,
                message_content_hash("user", content),
                MESSAGE_CONTENT_HASH_VERSION,
                MESSAGE_RECORD_VERSION,
            ),
        )

    # A producer may have begun writing the reviewed ordered format before the
    # v38 schema migration itself ran.  The migration must install its raw
    # source immutability guard (and do so replay-safely), without inferring a
    # session frontier from this isolated proof.
    ordered_content = "ordered source remains canonical"
    ordered_record = encode_message_record(
        message_id=2, role="user", content=ordered_content
    )
    conn.execute(
        "INSERT INTO messages(id, session_id, role, content, created_at) "
        "VALUES (2, 'legacy', 'user', ?, '2024-01-02 03:04:05')",
        (ordered_content,),
    )
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES "
        "('ordered_cov_2', 'legacy', 2, 2, 'ordered', ?)",
        (ordered_record,),
    )
    conn.execute(
        "INSERT INTO message_retention_coverage("
        "message_id, source_session_id, source_role, source_created_at, chunk_id, "
        "message_content_hash, hash_version, record_version, coverage_version) "
        "VALUES (2, 'legacy', 'user', '2024-01-02 03:04:05', "
        "'ordered_cov_2', ?, ?, ?, 'dream-lossless-message-v1')",
        (
            message_content_hash("user", ordered_content),
            MESSAGE_CONTENT_HASH_VERSION,
            MESSAGE_RECORD_VERSION,
        ),
    )

    core_db._run_migrations(conn)

    state = conn.execute(
        "SELECT coverage_message_id, digest_cursor_message_id, "
        "digest_cursor_offset, digest_published_generation, summary_source "
        "FROM sessions WHERE id = 'legacy'"
    ).fetchone()
    assert state["coverage_message_id"] is None
    assert state["digest_cursor_message_id"] is None
    assert state["digest_cursor_offset"] == 0
    assert state["digest_published_generation"] == "legacy"
    assert state["summary_source"] == "legacy"
    assert {row["chunk_kind"] for row in conn.execute(
        "SELECT chunk_kind FROM chunks"
    )} == {"coverage"}
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM chunks_fts WHERE chunks_fts MATCH 'sparse'"
    ).fetchone()["c"] == 0
    assert {"digest_slice_key", "digest_generation"} <= _cols(conn, "episodes")
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM episodes_fts "
        "WHERE episodes_fts MATCH 'legacyepisodetoken'"
    ).fetchone()["c"] == 1
    with pytest.raises(
        sqlite3.IntegrityError,
        match="ordered digest source is immutable",
    ):
        conn.execute("UPDATE messages SET id = 20 WHERE id = 2")

    # Simulate interruption after every DDL/data statement but before the
    # schema-version stamp. Duplicate columns/triggers and the selective FTS
    # rebuild must all replay without changing the conservative state.
    conn.execute(
        "UPDATE schema_meta SET value = '37' WHERE key = 'schema_version'"
    )
    core_db._run_migrations(conn)
    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert conn.execute(
        "SELECT coverage_message_id FROM sessions WHERE id = 'legacy'"
    ).fetchone()["coverage_message_id"] is None
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM chunks_fts WHERE chunks_fts MATCH 'sparse'"
    ).fetchone()["c"] == 0
    assert conn.execute(
        "SELECT digest_published_generation FROM sessions WHERE id = 'legacy'"
    ).fetchone()["digest_published_generation"] == "legacy"
    with pytest.raises(
        sqlite3.IntegrityError,
        match="ordered digest source is immutable",
    ):
        conn.execute("UPDATE messages SET content = 'changed' WHERE id = 2")
    # Reopening a current v38 store replays schema.sql (but no migration). Its
    # compatibility trigger definitions must not overwrite the selective v38
    # episode trigger under the same stable name.
    core_db.initialize(conn)
    episode_update_trigger = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
        "AND name = 'episodes_fts_update'"
    ).fetchone()["sql"]
    assert "digest_published_generation" in episode_update_trigger
    conn.close()


def test_v39_profile_cursor_migration_backfills_only_user_provenance_and_replays(
    tmp_path: Path,
):
    """v39 keeps durable USER provenance without blessing assistant evidence.

    The fixture intentionally contains the smallest supported v38 surface and
    replays the migration after clearing its version stamp, matching an
    interruption after all statements but before the final schema-meta write.
    """
    import sqlite3

    import pytest

    conn = core_db.connect(tmp_path / "v38-profile-cursor.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '38');
        CREATE TABLE sessions(
            id TEXT PRIMARY KEY,
            coverage_message_id INTEGER
        );
        CREATE TABLE messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP
        );
        CREATE TABLE chunks(
            id TEXT PRIMARY KEY,
            session_id TEXT,
            start_message_id INTEGER,
            end_message_id INTEGER,
            salience_reason TEXT,
            text TEXT,
            chunk_kind TEXT NOT NULL DEFAULT 'extraction',
            created_at TIMESTAMP
        );
        CREATE TABLE message_retention_coverage(
            message_id INTEGER NOT NULL,
            source_session_id TEXT NOT NULL,
            source_role TEXT NOT NULL,
            source_created_at TIMESTAMP,
            chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE RESTRICT,
            message_content_hash TEXT NOT NULL,
            hash_version TEXT NOT NULL,
            record_version TEXT NOT NULL,
            coverage_version TEXT NOT NULL,
            created_at TIMESTAMP,
            PRIMARY KEY(message_id, chunk_id, coverage_version)
        );
        CREATE TABLE user_profile(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot TEXT NOT NULL,
            slot_key TEXT,
            value TEXT NOT NULL,
            evidence_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            valid_at TIMESTAMP,
            invalid_at TIMESTAMP,
            created_at TIMESTAMP
        );
        CREATE TABLE dream_runs(id INTEGER PRIMARY KEY, started_at TIMESTAMP);
        INSERT INTO sessions(id, coverage_message_id) VALUES ('legacy', 2);
        INSERT INTO messages(id, session_id, role, content, created_at) VALUES
            (1, 'legacy', 'user', 'I live in Utrecht', '2026-01-02 03:04:05'),
            (2, 'legacy', 'assistant', 'You live in Utrecht', '2026-01-02 03:05:05');
        INSERT INTO user_profile(slot, value, evidence_message_id, valid_at) VALUES
            ('location', 'Utrecht', 1, '2026-01-02 03:04:05'),
            ('role', 'engineer', 2, '2026-01-02 03:05:05');
        """
    )

    core_db._run_migrations(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert {
        "profile_cursor_message_id",
        "profile_cursor_partial_message_id",
        "profile_cursor_offset",
        "profile_cursor_prompt_version",
        "profile_published_generation",
        "profile_retry_count",
        "profile_retry_config_version",
        "profile_quarantined",
    } <= _cols(conn, "sessions")
    assert {
        "cursor_before_message_id",
        "cursor_before_partial_message_id",
        "cursor_before_offset",
        "cursor_after_message_id",
        "cursor_after_partial_message_id",
        "cursor_after_offset",
    } <= _cols(conn, "profile_staging")
    assert {"profile_items_extracted", "profile_failures"} <= _cols(
        conn, "dream_runs"
    )

    user_source = conn.execute(
        "SELECT source_message_id, source_session_id, source_created_at "
        "FROM user_profile WHERE evidence_message_id = 1"
    ).fetchone()
    assert tuple(user_source) == (1, "legacy", "2026-01-02 03:04:05")
    assistant_source = conn.execute(
        "SELECT evidence_message_id, source_message_id, source_session_id, "
        "source_created_at FROM user_profile WHERE value = 'engineer'"
    ).fetchone()
    assert tuple(assistant_source) == (None, None, None, None)

    # Retention clears only the compatibility FK; durable USER provenance is
    # still available to API consumers and future prompt rewinds.
    conn.execute("DELETE FROM messages WHERE id = 1")
    retained = conn.execute(
        "SELECT evidence_message_id, source_message_id, source_session_id, "
        "source_created_at FROM user_profile WHERE value = 'Utrecht'"
    ).fetchone()
    assert tuple(retained) == (None, 1, "legacy", "2026-01-02 03:04:05")

    conn.execute(
        "UPDATE schema_meta SET value = '38' WHERE key = 'schema_version'"
    )
    core_db._run_migrations(conn)
    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM user_profile"
    ).fetchone()["c"] == 2
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    source_created_at = retained["source_created_at"]
    for statement, params in (
        (
            "INSERT INTO user_profile(slot, value, confidence) "
            "VALUES ('role', 'unattributed', 0.9)",
            (),
        ),
        (
            "INSERT INTO user_profile(slot, value, source_message_id, "
            "source_session_id, source_created_at, confidence) "
            "VALUES ('invented', 'x', 1, 'legacy', ?, 0.9)",
            (source_created_at,),
        ),
        (
            "INSERT INTO user_profile(slot, value, source_message_id, "
            "source_session_id, source_created_at, confidence) "
            "VALUES ('language', 'Dutch', 1, 'legacy', ?, 2.0)",
            (source_created_at,),
        ),
    ):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(statement, params)
    conn.close()


def test_v38_uncovered_user_profile_survives_v39_export_import(tmp_path: Path):
    """Upgrade materializes the source proof before raw-free v6 portability."""
    src = core_db.connect(tmp_path / "v38-uncovered-profile.sqlite")
    core_db.initialize(src)
    for name in (
        "user_profile_shape_insert_guard",
        "user_profile_shape_update_guard",
        "user_profile_source_insert_guard",
        "user_profile_source_update_guard",
    ):
        src.execute(f"DROP TRIGGER IF EXISTS {name}")
    src.execute("INSERT INTO sessions(id) VALUES ('legacy-uncovered')")
    source_mid = src.execute(
        "INSERT INTO messages(session_id, role, content, created_at) "
        "VALUES ('legacy-uncovered', 'user', 'I live in Utrecht.', ?)",
        ("2026-07-01T12:00:00Z",),
    ).lastrowid
    src.execute(
        "INSERT INTO user_profile("
        "slot, value, evidence_message_id, confidence, valid_at"
        ") VALUES ('location', 'Utrecht', ?, 0.9, ?)",
        (source_mid, "2026-07-01T12:00:00Z"),
    )
    assert src.execute(
        "SELECT COUNT(*) FROM message_retention_coverage"
    ).fetchone()[0] == 0

    # The data shape is the pre-v39 one even though the initialized fixture has
    # additive future columns, letting export exercise the complete table set.
    src.execute(
        "UPDATE schema_meta SET value = '38' WHERE key = 'schema_version'"
    )
    core_db._run_migrations(src)

    upgraded = src.execute(
        "SELECT evidence_message_id, source_message_id, source_session_id, "
        "source_created_at FROM user_profile"
    ).fetchone()
    assert tuple(upgraded) == (
        source_mid,
        source_mid,
        "legacy-uncovered",
        "2026-07-01T12:00:00Z",
    )
    assert tuple(src.execute(
        "SELECT message_id, source_session_id, source_role "
        "FROM message_retention_coverage"
    ).fetchall()[0]) == (
        source_mid,
        "legacy-uncovered",
        "user",
    )

    out = tmp_path / "v39-upgraded-profile.jsonl"
    portability.export_jsonl(src, out)
    src.close()

    dst = core_db.connect(tmp_path / "v39-imported-profile.sqlite")
    try:
        core_db.initialize(dst)
        portability.import_jsonl(dst, out)
        imported = dst.execute(
            "SELECT slot, value, evidence_message_id, source_message_id, "
            "source_session_id, source_created_at, valid_at "
            "FROM user_profile"
        ).fetchone()
        assert tuple(imported) == (
            "location",
            "Utrecht",
            None,
            source_mid,
            "legacy-uncovered",
            "2026-07-01T12:00:00Z",
            "2026-07-01T12:00:00.000000+00:00",
        )
        assert tuple(dst.execute(
            "SELECT message_id, source_session_id, source_role "
            "FROM message_retention_coverage"
        ).fetchall()[0]) == (
            source_mid,
            "legacy-uncovered",
            "user",
        )
        assert dst.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    finally:
        dst.close()


_PROFILE_UNIQUE_INDEX_SQL = {
    "idx_user_profile_one_active_singleton": (
        "CREATE UNIQUE INDEX idx_user_profile_one_active_singleton "
        "ON user_profile(slot) WHERE invalid_at IS NULL AND "
        "slot IN ('name','role','employer','location','age_birthday')"
    ),
    "idx_user_profile_one_active_relationship": (
        "CREATE UNIQUE INDEX idx_user_profile_one_active_relationship "
        "ON user_profile(slot, lower(trim(COALESCE(slot_key, '')))) "
        "WHERE invalid_at IS NULL AND slot = 'relationship'"
    ),
}


def _profile_unique_index_sql(conn) -> dict[str, str]:
    return {
        row["name"]: row["sql"]
        for row in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
            "AND name IN (?, ?) ORDER BY name",
            tuple(_PROFILE_UNIQUE_INDEX_SQL),
        ).fetchall()
    }


def test_fresh_schema_exposes_exact_profile_unique_index_definitions(
    tmp_path: Path,
):
    conn = core_db.connect(tmp_path / "fresh-v39-profile-indexes.sqlite")
    try:
        core_db.initialize(conn)
        assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
        assert _profile_unique_index_sql(conn) == _PROFILE_UNIQUE_INDEX_SQL
    finally:
        conn.close()


def test_v39_migration_heals_profile_keys_before_exact_unique_indexes(
    tmp_path: Path,
):
    conn = core_db.connect(tmp_path / "migrated-v39-profile-indexes.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '38');
        CREATE TABLE sessions(
            id TEXT PRIMARY KEY,
            coverage_message_id INTEGER
        );
        CREATE TABLE user_profile(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot TEXT NOT NULL,
            slot_key TEXT,
            value TEXT NOT NULL,
            evidence_message_id INTEGER,
            confidence REAL NOT NULL DEFAULT 1.0,
            valid_at TIMESTAMP,
            invalid_at TIMESTAMP,
            created_at TIMESTAMP
        );
        INSERT INTO user_profile(
            id, slot, slot_key, value, valid_at, created_at
        ) VALUES
            (11, 'name', 'legacy-name-key', 'Atta',
                '2026-02-01 00:00:00', '2026-02-01 00:00:00'),
            (12, 'role', 'legacy-role-key', 'doctor',
                '2026-02-01 00:00:00', '2026-02-01 00:00:00'),
            (13, 'employer', 'legacy-employer-key', 'MedFlow',
                '2026-02-01 00:00:00', '2026-02-01 00:00:00'),
            (14, 'location', 'legacy-location-key', 'Utrecht',
                '2026-02-01 00:00:00', '2026-02-01 00:00:00'),
            (15, 'age_birthday', 'legacy-age-key', 'April 3',
                '2026-02-01 00:00:00', '2026-02-01 00:00:00'),
            -- Deliberately insert the newer relationship before the older one
            -- and give chronology the opposite id order. Healing must follow
            -- validity time, not insertion/rowid order.
            (30, 'relationship', NULL, 'spouse',
                '2026-03-01 00:00:00', '2026-03-01 00:00:00'),
            (90, 'relationship', NULL, 'friend',
                '2026-01-01 00:00:00', '2026-01-01 00:00:00');
        """
    )

    core_db._run_migrations(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert _profile_unique_index_sql(conn) == _PROFILE_UNIQUE_INDEX_SQL
    singleton_keys = conn.execute(
        "SELECT slot, slot_key FROM user_profile WHERE slot IN "
        "('name','role','employer','location','age_birthday') ORDER BY slot"
    ).fetchall()
    assert [(row["slot"], row["slot_key"]) for row in singleton_keys] == [
        ("age_birthday", None),
        ("employer", None),
        ("location", None),
        ("name", None),
        ("role", None),
    ]
    relationships = conn.execute(
        "SELECT id, value, slot_key, valid_at, invalid_at FROM user_profile "
        "WHERE slot = 'relationship' ORDER BY valid_at"
    ).fetchall()
    assert [tuple(row) for row in relationships] == [
        (
            90,
            "friend",
            "[legacy-unknown]",
            "2026-01-01 00:00:00",
            "2026-03-01 00:00:00",
        ),
        (30, "spouse", "[legacy-unknown]", "2026-03-01 00:00:00", None),
    ]

    # Startup replay must retain both the chosen active row and exact DDL.
    before = [tuple(row) for row in relationships]
    core_db._run_migrations(conn)
    replayed = conn.execute(
        "SELECT id, value, slot_key, valid_at, invalid_at FROM user_profile "
        "WHERE slot = 'relationship' ORDER BY valid_at"
    ).fetchall()
    assert [tuple(row) for row in replayed] == before
    assert _profile_unique_index_sql(conn) == _PROFILE_UNIQUE_INDEX_SQL
    conn.close()


def test_v39_profile_sql_guards_require_canonical_user_provenance_and_shape(
    tmp_path: Path,
):
    import sqlite3

    import pytest

    from hymem.dreaming.lossless import materialize_message_coverage

    conn = core_db.connect(tmp_path / "profile-sql-guards.sqlite")
    core_db.initialize(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('guarded')")
    user_mid = conn.execute(
        "INSERT INTO messages(session_id, role, content) "
        "VALUES ('guarded', 'user', 'I live in Utrecht')"
    ).lastrowid
    assistant_mid = conn.execute(
        "INSERT INTO messages(session_id, role, content) "
        "VALUES ('guarded', 'assistant', 'You live in Utrecht')"
    ).lastrowid
    with core_db.transaction(conn):
        materialize_message_coverage(conn, "guarded")
    timestamps = {
        row["id"]: row["created_at"]
        for row in conn.execute(
            "SELECT id, created_at FROM messages WHERE session_id = 'guarded'"
        )
    }

    def insert_sql(
        *, slot="location", key=None, value="Utrecht", confidence=0.9,
        source_mid=user_mid,
    ):
        return conn.execute(
            "INSERT INTO user_profile(slot, slot_key, value, "
            "evidence_message_id, source_message_id, source_session_id, "
            "source_created_at, confidence) VALUES (?, ?, ?, ?, ?, 'guarded', ?, ?)",
            (
                slot, key, value, source_mid, source_mid,
                timestamps[source_mid], confidence,
            ),
        )

    insert_sql()
    insert_sql(slot="relationship", key="anna", value="friend")
    bad = [
        lambda: conn.execute(
            "INSERT INTO user_profile(slot, value, confidence) "
            "VALUES ('role', 'doctor', 0.9)"
        ),
        lambda: insert_sql(slot="invented_slot"),
        lambda: insert_sql(slot="language", value="Dutch", confidence=2.0),
        lambda: insert_sql(slot="employer", value="Acme", confidence=float("inf")),
        lambda: insert_sql(slot="name", key="illegal-key", value="Atta"),
        lambda: insert_sql(
            slot="relationship", key=" Anna ", value="colleague"
        ),
        lambda: insert_sql(
            slot="possession", value="car", source_mid=assistant_mid
        ),
    ]
    for operation in bad:
        with pytest.raises(sqlite3.IntegrityError):
            operation()
    assert conn.execute("SELECT COUNT(*) FROM user_profile").fetchone()[0] == 2
    conn.close()


def test_v39_reopen_normalizes_legacy_relationship_key_collisions(tmp_path: Path):
    conn = core_db.connect(tmp_path / "legacy-relationship-keys.sqlite")
    core_db.initialize(conn)
    for name in (
        "user_profile_shape_insert_guard",
        "user_profile_shape_update_guard",
        "user_profile_source_insert_guard",
        "user_profile_source_update_guard",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute("DROP INDEX idx_user_profile_one_active_relationship")
    conn.executemany(
        "INSERT INTO user_profile(slot, slot_key, value, valid_at, created_at) "
        "VALUES ('relationship', ?, ?, ?, ?)",
        [
            (" Anna ", "friend", "2026-01-01", "2026-01-01"),
            ("anna", "spouse", "2026-02-01", "2026-02-01"),
            (None, "colleague", "2026-03-01", "2026-03-01"),
        ],
    )

    core_db._ensure_profile_active_invariants(conn)
    rows = conn.execute(
        "SELECT slot_key, value, invalid_at FROM user_profile "
        "ORDER BY valid_at"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("anna", "friend", "2026-02-01"),
        ("anna", "spouse", None),
        ("[legacy-unknown]", "colleague", None),
    ]
    before = [tuple(row) for row in rows]
    core_db._ensure_profile_active_invariants(conn)
    assert [tuple(row) for row in conn.execute(
        "SELECT slot_key, value, invalid_at FROM user_profile ORDER BY valid_at"
    )] == before
    conn.close()


def _publish_migration_test_claim(conn) -> str:
    """Create one real source-validated claim publication on ``conn``."""
    from hymem import HyMemConfig
    from hymem.dreaming import phase1
    from hymem.dreaming.chunks import Chunk, persist_chunks
    from hymem.dreaming.lossless import materialize_message_coverage
    from hymem.dreaming.phase1 import ChunkExtraction
    from hymem.extraction.triples import Triple

    conn.execute("INSERT INTO sessions(id) VALUES ('v41-source')")
    message_id = int(conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at) "
        "VALUES ('v41-source','user','App uses Redis','2026-01-01T00:00:00Z')"
    ).lastrowid)
    with core_db.transaction(conn):
        materialize_message_coverage(conn, "v41-source")
    chunk = Chunk(
        id="v41-extraction", session_id="v41-source",
        start_message_id=message_id, end_message_id=message_id,
        salience_reason="migration-test", text="user: App uses Redis",
        source_message_ids=(message_id,),
    )
    with core_db.transaction(conn):
        persist_chunks(conn, [chunk])
    sources = phase1._claim_sources_for_chunk(conn, chunk)
    with core_db.transaction(conn):
        phase1.persist_chunk_results(
            conn,
            chunk,
            ChunkExtraction(
                triples=[Triple(
                    "app", "uses", "redis", 1,
                    source_message_id=message_id,
                )],
                markers=[],
                claim_sources={source.message_id: source for source in sources},
                source_validated=True,
            ),
            prompt_version="v13",
            cfg=HyMemConfig(root=Path(".")),
        )
    return chunk.id


def test_stamped_v40_upgrades_to_v41_and_backfills_only_proven_nonempty_outcome(
    tmp_path: Path,
):
    path = tmp_path / "stamped-v40-outcome.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    chunk_id = _publish_migration_test_claim(conn)
    expected = tuple(conn.execute(
        "SELECT prompt_version,prompt_generation,result_hash "
        "FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone())
    # Recreate the exact released-v40 boundary: observations existed, but the
    # v41 publication table and its revised header guard did not.
    conn.execute("DROP TRIGGER chunk_source_manifest_header_update_guard")
    conn.execute("DROP TABLE kg_claim_extraction_outcomes")
    conn.execute(
        "UPDATE schema_meta SET value='40' WHERE key='schema_version'"
    )
    conn.close()

    conn = core_db.connect(path)
    core_db.initialize(conn)
    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert tuple(conn.execute(
        "SELECT prompt_version,prompt_generation,result_hash "
        "FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()) == expected
    assert {
        (row["from"], row["table"], row["to"], row["on_delete"])
        for row in conn.execute(
            "PRAGMA foreign_key_list(kg_claim_extraction_outcomes)"
        )
    } == {("chunk_id", "chunks", "id", "RESTRICT")}
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_stamped_v41_reopen_replaces_stale_outcome_and_manifest_guards(
    tmp_path: Path,
):
    path = tmp_path / "stamped-v41-stale-outcome-guards.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    chunk_id = _publish_migration_test_claim(conn)
    for name in (
        "kg_claim_extraction_outcomes_insert_guard",
        "kg_claim_extraction_outcomes_update_guard",
        "kg_claim_extraction_outcomes_delete_guard",
        "chunk_source_manifest_header_update_guard",
    ):
        conn.execute(f"DROP TRIGGER {name}")
        conn.execute(
            f"CREATE TRIGGER {name} BEFORE UPDATE ON chunks "
            "BEGIN SELECT 1; END"
            if name == "chunk_source_manifest_header_update_guard"
            else f"CREATE TRIGGER {name} BEFORE UPDATE ON "
                 "kg_claim_extraction_outcomes BEGIN SELECT 1; END"
        )
    conn.close()

    conn = core_db.connect(path)
    core_db.initialize(conn)
    definitions = {
        row["name"]: row["sql"]
        for row in conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name IN "
            "('kg_claim_extraction_outcomes_insert_guard',"
            "'kg_claim_extraction_outcomes_update_guard',"
            "'kg_claim_extraction_outcomes_delete_guard',"
            "'chunk_source_manifest_header_update_guard')"
        ).fetchall()
    }
    assert len(definitions) == 4
    assert "substr(new.result_hash, 8) GLOB '*[^0-9a-f]*'" in definitions[
        "kg_claim_extraction_outcomes_insert_guard"
    ]
    assert "kg_claim_extraction_outcomes outcome" in definitions[
        "chunk_source_manifest_header_update_guard"
    ]
    with pytest.raises(sqlite3.IntegrityError, match="internally managed"):
        conn.execute(
            "UPDATE kg_claim_extraction_outcomes SET prompt_version='v99' "
            "WHERE chunk_id=?", (chunk_id,),
        )
    with core_db.evidence_mutation(conn), pytest.raises(
        sqlite3.IntegrityError, match="internally managed"
    ):
        conn.execute(
            "UPDATE kg_claim_extraction_outcomes SET result_hash=? WHERE chunk_id=?",
            ("sha256:" + "a" * 63 + "z", chunk_id),
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE chunks SET source_manifest_version=NULL,"
            "source_manifest_count=NULL WHERE id=?", (chunk_id,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM chunks WHERE id=?", (chunk_id,))
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_stamped_v41_reopen_rebuilds_early_cascade_outcome_fk(tmp_path: Path):
    path = tmp_path / "stamped-v41-cascade-outcome.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    chunk_id = _publish_migration_test_claim(conn)
    for name in (
        "kg_claim_extraction_outcomes_insert_guard",
        "kg_claim_extraction_outcomes_update_guard",
        "kg_claim_extraction_outcomes_delete_guard",
        "chunk_source_manifest_header_update_guard",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.executescript(
        """
        ALTER TABLE kg_claim_extraction_outcomes
            RENAME TO kg_claim_extraction_outcomes_v41_final;
        CREATE TABLE kg_claim_extraction_outcomes(
            chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            prompt_version TEXT NOT NULL,
            prompt_generation INTEGER NOT NULL CHECK(prompt_generation >= 0),
            result_hash TEXT NOT NULL,
            succeeded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO kg_claim_extraction_outcomes
            SELECT * FROM kg_claim_extraction_outcomes_v41_final;
        DROP TABLE kg_claim_extraction_outcomes_v41_final;
        """
    )
    assert conn.execute(
        "PRAGMA foreign_key_list(kg_claim_extraction_outcomes)"
    ).fetchone()["on_delete"] == "CASCADE"
    conn.close()

    conn = core_db.connect(path)
    core_db.initialize(conn)
    assert conn.execute(
        "PRAGMA foreign_key_list(kg_claim_extraction_outcomes)"
    ).fetchone()["on_delete"] == "RESTRICT"
    assert conn.execute(
        "SELECT prompt_version FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()[0] == "v13"
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM chunks WHERE id=?", (chunk_id,))
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_v42_backfills_only_coherent_whole_chunk_publication(tmp_path: Path):
    path = tmp_path / "v41-coherent-publication.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    chunk_id = _publish_migration_test_claim(conn)
    evidence_id = int(conn.execute("SELECT id FROM kg_evidence").fetchone()[0])
    with core_db.evidence_history_mutation(conn):
        conn.execute(
            "UPDATE kg_evidence SET extracted_at=?,published_at=? WHERE id=?",
            (
                "2026-01-02T00:00:00.000Z",
                "2026-01-02T00:03:00.000Z",
                evidence_id,
            ),
        )
        conn.execute(
            "UPDATE kg_edge_lifecycle SET created_at=? "
            "WHERE source_evidence_id=?",
            ("2026-01-02T00:01:00.000Z", evidence_id),
        )
        conn.execute(
            "UPDATE kg_claim_observations SET observed_at=? WHERE evidence_id=?",
            ("2026-01-02T00:02:00.000Z", evidence_id),
        )
        conn.execute(
            "UPDATE kg_claim_extraction_outcomes SET succeeded_at=? "
            "WHERE chunk_id=?",
            ("2026-01-02T00:03:00.000Z", chunk_id),
        )
    conn.execute("DROP TRIGGER kg_evidence_published_at_update_guard")
    with core_db.evidence_mutation(conn):
        conn.execute(
            "UPDATE kg_evidence SET published_at=NULL WHERE id=?", (evidence_id,)
        )
    conn.execute("UPDATE schema_meta SET value='41' WHERE key='schema_version'")
    conn.close()

    conn = core_db.connect(path)
    core_db.initialize(conn)
    assert conn.execute(
        "SELECT published_at FROM kg_evidence WHERE id=?", (evidence_id,)
    ).fetchone()[0] == "2026-01-02T00:03:00.000Z"
    conn.close()


def test_v42_does_not_publish_staged_v41_orphan(tmp_path: Path):
    path = tmp_path / "v41-staged-orphan.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    chunk_id = _publish_migration_test_claim(conn)
    evidence_id = int(conn.execute("SELECT id FROM kg_evidence").fetchone()[0])
    conn.execute("DROP TRIGGER kg_evidence_published_at_update_guard")
    with core_db.evidence_mutation(conn):
        conn.execute("DELETE FROM kg_claim_observations WHERE chunk_id=?", (chunk_id,))
        conn.execute(
            "DELETE FROM kg_claim_extraction_outcomes WHERE chunk_id=?", (chunk_id,)
        )
        conn.execute(
            "UPDATE kg_evidence SET published_at=NULL WHERE id=?", (evidence_id,)
        )
    conn.execute("UPDATE schema_meta SET value='41' WHERE key='schema_version'")
    conn.close()

    conn = core_db.connect(path)
    core_db.initialize(conn)
    assert conn.execute(
        "SELECT published_at FROM kg_evidence WHERE id=?", (evidence_id,)
    ).fetchone()[0] is None
    conn.close()


def test_v42_restart_heals_publication_guard_and_protects_published_audit(
    tmp_path: Path,
):
    path = tmp_path / "v42-publication-guard-heal.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    _publish_migration_test_claim(conn)
    evidence_id = int(conn.execute("SELECT id FROM kg_evidence").fetchone()[0])
    for name in (
        "kg_evidence_published_at_update_guard",
        "kg_evidence_v40_delete_guard",
        "kg_edge_lifecycle_update_guard",
        "kg_edge_lifecycle_delete_guard",
        "kg_lifecycle_dependencies_update_guard",
        "kg_lifecycle_dependencies_delete_guard",
    ):
        conn.execute(f"DROP TRIGGER {name}")
    conn.close()

    conn = core_db.connect(path)
    core_db.initialize(conn)
    trigger = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' "
        "AND name='kg_evidence_published_at_update_guard'"
    ).fetchone()[0]
    assert "extracted_at" in trigger
    assert "superseded_reason" in trigger
    healed = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name IN "
            "('kg_evidence_v40_delete_guard','kg_edge_lifecycle_update_guard',"
            "'kg_edge_lifecycle_delete_guard',"
            "'kg_lifecycle_dependencies_update_guard',"
            "'kg_lifecycle_dependencies_delete_guard')"
        ).fetchall()
    }
    assert set(healed) == {
        "kg_evidence_v40_delete_guard",
        "kg_edge_lifecycle_update_guard",
        "kg_edge_lifecycle_delete_guard",
        "kg_lifecycle_dependencies_update_guard",
        "kg_lifecycle_dependencies_delete_guard",
    }
    assert "hymem_evidence_destructive_authorized" in healed[
        "kg_evidence_v40_delete_guard"
    ]
    assert all(
        "hymem_evidence_destructive_authorized" in healed[name]
        for name in (
            "kg_edge_lifecycle_update_guard",
            "kg_edge_lifecycle_delete_guard",
            "kg_lifecycle_dependencies_update_guard",
            "kg_lifecycle_dependencies_delete_guard",
        )
    )
    lifecycle_id = int(conn.execute(
        "SELECT id FROM kg_edge_lifecycle WHERE source_evidence_id=?",
        (evidence_id,),
    ).fetchone()[0])
    with core_db.evidence_mutation(conn), pytest.raises(
        sqlite3.IntegrityError, match="published evidence audit is immutable"
    ):
        conn.execute(
            "UPDATE kg_evidence SET extracted_at='2026-01-02T00:00:00.000Z' "
            "WHERE id=?",
            (evidence_id,),
        )
    with core_db.evidence_mutation(conn), pytest.raises(
        sqlite3.IntegrityError, match="published kg evidence history is immutable"
    ):
        conn.execute("DELETE FROM kg_evidence WHERE id=?", (evidence_id,))
    with core_db.evidence_mutation(conn), pytest.raises(
        sqlite3.IntegrityError, match="lifecycle history is immutable"
    ):
        conn.execute(
            "UPDATE kg_edge_lifecycle SET details='forged' WHERE id=?",
            (lifecycle_id,),
        )
    with core_db.evidence_mutation(conn), pytest.raises(
        sqlite3.IntegrityError, match="lifecycle history is immutable"
    ):
        conn.execute("DELETE FROM kg_edge_lifecycle WHERE id=?", (lifecycle_id,))
    with core_db.evidence_mutation(conn), pytest.raises(
        sqlite3.IntegrityError, match="published evidence audit is immutable"
    ):
        conn.execute(
            "UPDATE kg_evidence SET polarity=-1,interpretation_key='forged' "
            "WHERE id=?",
            (evidence_id,),
        )
    with core_db.evidence_history_mutation(conn):
        conn.execute(
            "UPDATE kg_evidence SET extracted_at='2026-01-02T00:00:00.000Z' "
            "WHERE id=?",
            (evidence_id,),
        )
    assert conn.execute(
        "SELECT extracted_at FROM kg_evidence WHERE id=?", (evidence_id,)
    ).fetchone()[0] == "2026-01-02T00:00:00.000Z"
    conn.close()


# --- migration 046 ---------------------------------------------------------


def _publish_migration_test_fact(
    conn: sqlite3.Connection, tmp_path: Path
) -> tuple[str, int]:
    """Publish one fully proved v46 fact through the production writer."""

    from hymem import HyMemConfig
    from hymem.dreaming.facts import extract_facts, persist_facts
    from hymem.dreaming.lossless import materialize_message_coverage
    from hymem.extraction.llm import StubLLMClient

    session_id = "v46-stale-stamp-source"
    conn.execute("INSERT INTO sessions(id) VALUES (?)", (session_id,))
    conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at) "
        "VALUES (?, 'user', 'The migration keeps this proved fact.', "
        "'2026-08-01T12:00:00.000Z')",
        (session_id,),
    )
    with core_db.transaction(conn):
        assert materialize_message_coverage(conn, session_id) == 1
    extraction = extract_facts(
        conn,
        session_id,
        StubLLMClient(
            default=(
                '[{"text":"The migration keeps this proved fact.",'
                '"date":null,"entities":["migration"]}]'
            )
        ),
        HyMemConfig(root=tmp_path),
    )
    assert extraction is not None and not extraction.parse_failed
    with core_db.transaction(conn):
        assert persist_facts(conn, session_id, extraction) == 1
    fact_id = int(conn.execute(
        "SELECT id FROM narrative_facts WHERE session_id=?", (session_id,)
    ).fetchone()[0])
    return str(extraction.slice_key), fact_id


def test_v46_complete_domain_with_stale_v45_stamp_preserves_fact_history(
    tmp_path: Path,
):
    """A stale marker heals guards; it must never replay the table rebuild."""

    path = tmp_path / "complete-v46-stale-v45.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    slice_key, fact_id = _publish_migration_test_fact(conn, tmp_path)
    before = {
        table: [tuple(row) for row in conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ).fetchall()]
        for table in (
            "fact_extraction_outcomes",
            "fact_extraction_revisions",
            "fact_extraction_source_occurrences",
            "narrative_facts",
            "narrative_fact_lifecycle",
        )
    }
    conn.execute("DROP TRIGGER fact_revision_update_guard")
    conn.execute(
        "UPDATE schema_meta SET value='45' WHERE key='schema_version'"
    )

    # This takes _v46_sql_is_complete's non-destructive healing path.  In
    # particular, trigger installation must not use executescript(), whose
    # implicit COMMIT would escape the surrounding transaction.
    core_db._run_migrations(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION == 46
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='trigger' "
        "AND name='fact_revision_update_guard'"
    ).fetchone() is not None
    after = {
        table: [tuple(row) for row in conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ).fetchall()]
        for table in before
    }
    assert after == before
    assert conn.execute(
        "SELECT source_outcome_key FROM narrative_facts WHERE id=?", (fact_id,)
    ).fetchone()[0] == slice_key
    assert conn.execute(
        "SELECT COUNT(*) FROM narrative_fact_lifecycle WHERE fact_id=?",
        (fact_id,),
    ).fetchone()[0] == 1
    conn.close()


def test_v46_mid_migration_failure_rolls_back_and_restart_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The destructive rebuild, FTS population, and stamp are crash-atomic."""

    conn = core_db.connect(tmp_path / "v45-failure-restart.sqlite")
    core_db.initialize(conn)
    _downgrade_fact_domain_to_v45(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('legacy-fact-session')")
    conn.execute(
        "INSERT INTO narrative_facts(id,session_id,start_message_id,"
        "end_message_id,text,prompt_version) "
        "VALUES (7,'legacy-fact-session',1,1,'legacy survives','facts.v2')"
    )
    conn.execute(
        "INSERT INTO narrative_fact_embeddings("
        "fact_id,vector_json,model,dim,text_hash,created_at) "
        "VALUES (7,'[1.0,0.0]','legacy-vector',2,'legacy-hash',"
        "'2026-08-01T12:00:00.000Z')"
    )
    conn.execute(
        "INSERT INTO narrative_facts(id,session_id,start_message_id,"
        "end_message_id,text,prompt_version) "
        "VALUES (100,'legacy-fact-session',2,2,'retired id','facts.v2')"
    )
    conn.execute("DELETE FROM narrative_facts WHERE id=100")

    def v45_state() -> dict[str, object]:
        object_names = {
            "narrative_facts",
            "narrative_fact_embeddings",
            "narrative_facts_fts",
            "narrative_facts_fts_data",
            "narrative_facts_fts_idx",
            "narrative_facts_fts_docsize",
            "narrative_facts_fts_config",
            "narrative_facts_fts_insert",
            "narrative_facts_fts_delete",
            "narrative_facts_fts_update",
        }
        placeholders = ",".join("?" for _ in object_names)
        return {
            # Include rootpage and original CREATE SQL: rollback must restore
            # the original table identities, not merely equivalent-looking
            # rows in a half-rebuilt replacement.
            "objects": [
                tuple(row) for row in conn.execute(
                    "SELECT type,name,tbl_name,rootpage,sql FROM sqlite_master "
                    f"WHERE name IN ({placeholders}) ORDER BY type,name",
                    tuple(sorted(object_names)),
                ).fetchall()
            ],
            "session_columns": [
                tuple(row) for row in conn.execute(
                    "PRAGMA table_info(sessions)"
                ).fetchall()
            ],
            "fact_columns": [
                tuple(row) for row in conn.execute(
                    "PRAGMA table_info(narrative_facts)"
                ).fetchall()
            ],
            "embedding_fk": [
                tuple(row) for row in conn.execute(
                    "PRAGMA foreign_key_list(narrative_fact_embeddings)"
                ).fetchall()
            ],
            "facts": [
                tuple(row) for row in conn.execute(
                    "SELECT * FROM narrative_facts ORDER BY id"
                ).fetchall()
            ],
            "embeddings": [
                tuple(row) for row in conn.execute(
                    "SELECT * FROM narrative_fact_embeddings ORDER BY fact_id"
                ).fetchall()
            ],
            "sequence": conn.execute(
                "SELECT seq FROM sqlite_sequence WHERE name='narrative_facts'"
            ).fetchone()[0],
            "fts_match": [
                tuple(row) for row in conn.execute(
                    "SELECT rowid,text FROM narrative_facts_fts "
                    "WHERE narrative_facts_fts MATCH 'legacy' ORDER BY rowid"
                ).fetchall()
            ],
            "schema_version": core_db.schema_version(conn),
            "foreign_key_check": [
                tuple(row) for row in conn.execute("PRAGMA foreign_key_check")
            ],
        }

    before = v45_state()
    assert before["sequence"] == 100
    assert before["fts_match"] == [(7, "legacy survives")]
    assert before["schema_version"] == 45
    assert before["foreign_key_check"] == []
    original_apply = core_db._apply_migration_sql
    reached_late_failure = False

    def fail_after_destructive_v46_rebuild(
        target: sqlite3.Connection, script: str,
    ) -> None:
        nonlocal reached_late_failure
        if script.lstrip().startswith("-- v46:"):
            for statement in core_db._split_sql_statements(script):
                original_apply(target, statement)
                if statement.lstrip().startswith(
                    "INSERT INTO narrative_facts_fts(rowid, text)"
                ):
                    reached_late_failure = True
                    # Prove the injection is after every destructive operation
                    # named by this regression, not near the first ALTER.
                    assert "source_outcome_key" in _cols(
                        target, "narrative_facts"
                    )
                    assert _has_table(target, "narrative_fact_lifecycle")
                    assert _has_table(target, "fact_extraction_outcomes")
                    assert target.execute(
                        "SELECT fact_id FROM narrative_fact_embeddings"
                    ).fetchone()[0] == 7
                    assert target.execute(
                        "SELECT seq FROM sqlite_sequence "
                        "WHERE name='narrative_facts'"
                    ).fetchone()[0] == 100
                    # v46's filtered FTS has already replaced the v45 shadow;
                    # the legacy-unproved row is intentionally absent.
                    assert target.execute(
                        "SELECT rowid FROM narrative_facts_fts "
                        "WHERE narrative_facts_fts MATCH 'legacy'"
                    ).fetchall() == []
                    raise RuntimeError("injected late v46 migration failure")
            raise AssertionError("v46 FTS population statement was not reached")
        original_apply(target, script)

    monkeypatch.setattr(
        core_db, "_apply_migration_sql", fail_after_destructive_v46_rebuild
    )
    with pytest.raises(RuntimeError, match="injected late v46 migration failure"):
        core_db._run_migrations(conn)

    assert reached_late_failure is True
    assert v45_state() == before
    assert not _has_table(conn, "fact_extraction_outcomes")

    monkeypatch.setattr(core_db, "_apply_migration_sql", original_apply)
    core_db._run_migrations(conn)
    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION == 46
    assert "facts_cursor_message_id" in _cols(conn, "sessions")
    legacy = conn.execute(
        "SELECT id,text,source_outcome_key,lifecycle_status "
        "FROM narrative_facts"
    ).fetchone()
    assert tuple(legacy) == (7, "legacy survives", None, "legacy_unproven")
    embedding = conn.execute(
        "SELECT fact_id,vector_json,model,dim,text_hash,created_at "
        "FROM narrative_fact_embeddings"
    ).fetchone()
    assert tuple(embedding) == (
        7, "[1.0,0.0]", "legacy-vector", 2, "legacy-hash",
        "2026-08-01T12:00:00.000Z",
    )
    assert conn.execute(
        "SELECT seq FROM sqlite_sequence WHERE name='narrative_facts'"
    ).fetchone()[0] == 100
    assert conn.execute(
        "SELECT rowid FROM narrative_facts_fts "
        "WHERE narrative_facts_fts MATCH 'legacy'"
    ).fetchall() == []
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_v46_preserves_narrative_fact_autoincrement_high_water(tmp_path: Path):
    """Deleted legacy row ids remain retired across the v46 table rebuild."""

    conn = core_db.connect(tmp_path / "v45-fact-sequence.sqlite")
    core_db.initialize(conn)
    _downgrade_fact_domain_to_v45(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('legacy-sequence')")
    for fact_id in (7, 100):
        conn.execute(
            "INSERT INTO narrative_facts(id,session_id,start_message_id,"
            "end_message_id,text,prompt_version) VALUES (?,?,?,?,?,?)",
            (
                fact_id, "legacy-sequence", fact_id, fact_id,
                f"legacy fact {fact_id}", "facts.v2",
            ),
        )
    conn.execute("DELETE FROM narrative_facts WHERE id=100")
    assert conn.execute(
        "SELECT seq FROM sqlite_sequence WHERE name='narrative_facts'"
    ).fetchone()[0] == 100

    core_db._run_migrations(conn)
    next_id = int(conn.execute(
        "INSERT INTO narrative_facts(session_id,start_message_id,end_message_id,"
        "text,prompt_version) VALUES "
        "('legacy-sequence',101,101,'post migration','facts.v2') "
        "RETURNING id"
    ).fetchone()[0])
    assert next_id > 100
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_v46_sparse_v45_domain_skips_safely_and_stamps(tmp_path: Path):
    """A misleading partial fact schema is not eligible for v46's ALTERs."""

    conn = core_db.connect(tmp_path / "sparse-v45-facts.sqlite")
    conn.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT)")
    conn.execute(
        "INSERT INTO schema_meta(key,value) VALUES ('schema_version','45')"
    )
    conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY,facts_message_id INTEGER)")
    conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE narrative_facts(id INTEGER PRIMARY KEY,text TEXT)")
    conn.execute(
        "CREATE TABLE narrative_fact_embeddings("
        "fact_id INTEGER PRIMARY KEY,vector_json TEXT,model TEXT,dim INTEGER,"
        "text_hash TEXT,created_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE message_retention_coverage(message_id INTEGER)"
    )

    core_db._run_migrations(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION == 46
    assert "facts_cursor_message_id" not in _cols(conn, "sessions")
    assert not _has_table(conn, "fact_extraction_outcomes")
    conn.close()
