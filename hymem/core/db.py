from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import struct
from importlib.resources import files
from pathlib import Path
from typing import Iterator

log = logging.getLogger("hymem.core.db")

EXPECTED_SCHEMA_VERSION = 5


def _load_schema() -> str:
    return (files("hymem.core") / "schema.sql").read_text(encoding="utf-8")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    # WAL is set here (not just in schema.sql) so it is active before any
    # schema creation or migration runs. journal_mode persists on the file;
    # synchronous is per-connection and must be set every time.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


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


def _run_migrations(conn: sqlite3.Connection) -> None:
    cur = schema_version(conn)
    if cur < 2:
        _migrate_v2(conn)
    if cur < 3:
        _migrate_v3(conn)
    if cur < 4:
        _migrate_v4(conn)
    if cur < 5:
        _migrate_v5(conn)


def _migrate_v2(conn: sqlite3.Connection) -> None:
    for col, col_type in [
        ("value_text", "TEXT"),
        ("value_numeric", "REAL"),
        ("value_unit", "TEXT"),
        ("temporal_scope", "TEXT"),
    ]:
        try:
            conn.execute(
                f"ALTER TABLE kg_evidence ADD COLUMN {col} {col_type}"
            )
        except sqlite3.OperationalError:
            pass
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', '2')"
    )
    log.info("migrated schema to v2 (numeric/temporal columns)")


def _migrate_v3(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN summary TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', '3')"
    )
    log.info("migrated schema to v3 (session summary column)")


def _migrate_v4(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE knowledge_graph ADD COLUMN derived BOOLEAN NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', '4')"
    )
    log.info("migrated schema to v4 (derived knowledge graph edges)")


def _migrate_v5(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS extraction_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
            chunk_text_snippet TEXT NOT NULL,
            extracted_subject TEXT NOT NULL,
            extracted_predicate TEXT NOT NULL,
            extracted_object TEXT NOT NULL,
            feedback_type TEXT NOT NULL DEFAULT 'retracted',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_created ON extraction_feedback(created_at)"
        )
    except sqlite3.OperationalError:
        pass
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', '5')"
    )
    log.info("migrated schema to v5 (extraction feedback table)")


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    if not _load_vec_extension(conn):
        return
    try:
        existing_dim = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
        ).fetchone()
        if existing_dim and int(existing_dim["value"]) == dim:
            return
        if existing_dim:
            conn.execute("DELETE FROM schema_meta WHERE key = 'vec_dim'")
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("DROP TABLE IF EXISTS vec_chunks")
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}])"
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('vec_dim', ?)",
            (str(dim),),
        )
        _backfill_vec(conn, dim)
    except sqlite3.OperationalError:
        log.info("vec_chunks table unavailable; using Python cosine search")


def _backfill_vec(conn: sqlite3.Connection, dim: int) -> None:
    rows = conn.execute(
        "SELECT c.rowid, e.vector_json FROM chunk_embeddings e "
        "JOIN chunks c ON c.id = e.chunk_id ORDER BY c.rowid"
    ).fetchall()
    if not rows:
        return
    count = conn.execute("SELECT COUNT(*) AS c FROM vec_chunks").fetchone()["c"]
    if count >= len(rows):
        return

    for r in rows:
        try:
            vec = json.loads(r["vector_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if len(vec) != dim:
            vec = list(vec) + [0.0] * (dim - len(vec))
        conn.execute(
            "INSERT OR IGNORE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (r["rowid"], _pack_vector(vec)),
        )
    log.info("backfilled vec_chunks with %d existing embeddings", len(rows))


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def vec_search(
    conn: sqlite3.Connection, query_vector: list[float], top_k: int
) -> list[tuple[int, float]]:
    if not _load_vec_extension(conn):
        return []
    try:
        rows = conn.execute(
            """
            SELECT rowid, distance
            FROM vec_chunks
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (_pack_vector(query_vector), top_k),
        ).fetchall()
        return [(int(r["rowid"]), float(r["distance"])) for r in rows]
    except (sqlite3.OperationalError, TypeError):
        return []


def has_vec_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
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
    chunk_count = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    if not chunk_count:
        return

    from hymem.dreaming.mentions import index_chunk_mentions

    rows = conn.execute("SELECT id, text FROM chunks").fetchall()
    total = 0
    for row in rows:
        total += index_chunk_mentions(conn, row["id"], row["text"])
    log.info("backfilled entity_mentions: chunks=%d mentions=%d", len(rows), total)
