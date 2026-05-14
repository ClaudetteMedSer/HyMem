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

EXPECTED_SCHEMA_VERSION = 6


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
    if cur < 6:
        _migrate_v6(conn)


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


def _migrate_v6(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_embeddings (
            edge_text TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    try:
        conn.execute(
            "ALTER TABLE dream_runs ADD COLUMN edges_embedded INTEGER NOT NULL DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', '6')"
    )
    log.info("migrated schema to v6 (edge embeddings table)")


_VEC_TABLES = frozenset({"vec_chunks", "vec_edges"})


def _ensure_vec_table_named(conn: sqlite3.Connection, name: str, dim: int) -> None:
    if name not in _VEC_TABLES:
        raise ValueError(f"unknown vec table: {name}")
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0(embedding float[{dim}])"
    )


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Ensure both vec_chunks and vec_edges exist at the given dim.

    The two virtual tables share the single 'vec_dim' schema_meta key, so on a
    dimension change they are dropped and rebuilt in lockstep, then backfilled
    from their JSON mirror tables (chunk_embeddings / edge_embeddings).
    """
    if not _load_vec_extension(conn):
        return
    try:
        existing_dim = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
        ).fetchone()
        if existing_dim and int(existing_dim["value"]) == dim:
            # Dim unchanged — still ensure vec_edges exists and is populated for
            # DBs that embedded chunks before vec_edges was introduced.
            _ensure_vec_table_named(conn, "vec_edges", dim)
            _backfill_vec_edges(conn, dim)
            return
        if existing_dim:
            conn.execute("DELETE FROM schema_meta WHERE key = 'vec_dim'")
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("DROP TABLE IF EXISTS vec_chunks")
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("DROP TABLE IF EXISTS vec_edges")
        _ensure_vec_table_named(conn, "vec_chunks", dim)
        _ensure_vec_table_named(conn, "vec_edges", dim)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('vec_dim', ?)",
            (str(dim),),
        )
        _backfill_vec(conn, dim)
        _backfill_vec_edges(conn, dim)
    except sqlite3.OperationalError:
        log.info("vec tables unavailable; using Python cosine search")


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


def _backfill_vec_edges(conn: sqlite3.Connection, dim: int) -> None:
    """Populate vec_edges (rowid = knowledge_graph.id) from cached edge vectors.

    Best-effort: embed_pending_edges is the authoritative refresh. This handles
    cold-start, dim changes, and pre-v6 DBs.
    """
    rows = conn.execute(
        """
        SELECT kg.id AS edge_id,
               kg.subject_canonical || ' ' || kg.predicate || ' '
                   || kg.object_canonical AS edge_text
        FROM knowledge_graph kg
        WHERE kg.status = 'active'
        """
    ).fetchall()
    if not rows:
        return
    have = conn.execute("SELECT COUNT(*) AS c FROM vec_edges").fetchone()["c"]
    if have >= len(rows):
        return
    for r in rows:
        emb = conn.execute(
            "SELECT vector_json FROM edge_embeddings WHERE edge_text = ?",
            (r["edge_text"],),
        ).fetchone()
        if emb is None:
            continue
        try:
            vec = json.loads(emb["vector_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if len(vec) != dim:
            vec = list(vec) + [0.0] * (dim - len(vec))
        conn.execute(
            "INSERT OR IGNORE INTO vec_edges(rowid, embedding) VALUES (?, ?)",
            (r["edge_id"], _pack_vector(vec)),
        )
    log.info("backfilled vec_edges from %d edge rows", len(rows))


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


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
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
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
    chunk_count = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    if not chunk_count:
        return

    from hymem.dreaming.mentions import index_chunk_mentions

    rows = conn.execute("SELECT id, text FROM chunks").fetchall()
    total = 0
    for row in rows:
        total += index_chunk_mentions(conn, row["id"], row["text"])
    log.info("backfilled entity_mentions: chunks=%d mentions=%d", len(rows), total)
