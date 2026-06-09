"""Tests for the file-based schema migration runner (db._run_migrations and
its helpers). Migrations live as ``NNN_*.sql`` under hymem/core/migrations/;
the runner applies any whose version exceeds the DB's schema_version and is
idempotent against a fresh schema.sql database.
"""

from __future__ import annotations

from pathlib import Path

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
