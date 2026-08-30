from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
import struct
from importlib.resources import files
from pathlib import Path
from typing import Iterator

from hymem.core.vectors import decode_vector

log = logging.getLogger("hymem.core.db")

EXPECTED_SCHEMA_VERSION = 34


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


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply every migration file whose version exceeds the DB's
    schema_version, bumping schema_version after each so an interrupted run
    resumes cleanly. Migrations are idempotent, so a fresh schema.sql database
    (which starts at version 1) runs them all as no-ops up to the latest."""
    cur = schema_version(conn)
    for version, entry in _discover_migrations():
        if version <= cur:
            continue
        _apply_migration_sql(conn, entry.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(version),),
        )
        log.info("migrated schema to v%d (%s)", version, entry.name)


_VEC_TABLES = frozenset({"vec_chunks", "vec_edges", "vec_episodes", "vec_facts"})


def _ensure_vec_table_named(conn: sqlite3.Connection, name: str, dim: int) -> None:
    if name not in _VEC_TABLES:
        raise ValueError(f"unknown vec table: {name}")
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0(embedding float[{dim}])"
    )


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Ensure vec_chunks, vec_edges, and vec_episodes exist at the given dim.

    The three virtual tables share the single 'vec_dim' schema_meta key, so on
    a dimension change they are dropped and rebuilt in lockstep, then
    backfilled from their JSON mirror tables (chunk_embeddings /
    edge_embeddings / episode_embeddings).
    """
    if not _load_vec_extension(conn):
        return
    try:
        existing_dim = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
        ).fetchone()
        if existing_dim and int(existing_dim["value"]) == dim:
            # Dim unchanged — still ensure vec_edges / vec_episodes / vec_facts
            # exist and are populated for DBs that embedded chunks before those
            # tables were introduced.
            _ensure_vec_table_named(conn, "vec_edges", dim)
            _backfill_vec_edges(conn, dim)
            _ensure_vec_table_named(conn, "vec_episodes", dim)
            _backfill_vec_episodes(conn, dim)
            _ensure_vec_table_named(conn, "vec_facts", dim)
            _backfill_vec_facts(conn, dim)
            return
        if existing_dim:
            conn.execute("DELETE FROM schema_meta WHERE key = 'vec_dim'")
            for stale in ("vec_chunks", "vec_edges", "vec_episodes", "vec_facts"):
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(f"DROP TABLE IF EXISTS {stale}")
        _ensure_vec_table_named(conn, "vec_chunks", dim)
        _ensure_vec_table_named(conn, "vec_edges", dim)
        _ensure_vec_table_named(conn, "vec_episodes", dim)
        _ensure_vec_table_named(conn, "vec_facts", dim)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('vec_dim', ?)",
            (str(dim),),
        )
        _backfill_vec(conn, dim)
        _backfill_vec_edges(conn, dim)
        _backfill_vec_episodes(conn, dim)
        _backfill_vec_facts(conn, dim)
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
            vec = decode_vector(r["vector_json"])
        except (json.JSONDecodeError, TypeError, ValueError):
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
            vec = decode_vector(emb["vector_json"])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if len(vec) != dim:
            vec = list(vec) + [0.0] * (dim - len(vec))
        conn.execute(
            "INSERT OR IGNORE INTO vec_edges(rowid, embedding) VALUES (?, ?)",
            (r["edge_id"], _pack_vector(vec)),
        )
    log.info("backfilled vec_edges from %d edge rows", len(rows))


def _backfill_vec_episodes(conn: sqlite3.Connection, dim: int) -> None:
    """Populate vec_episodes from episode_embeddings on cold start / dim change."""
    rows = conn.execute(
        "SELECT episode_id, vector_json FROM episode_embeddings"
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
        for r in conn.execute("SELECT id, rowid FROM episodes").fetchall()
    }
    for r in rows:
        rowid = id_rowid.get(r["episode_id"])
        if rowid is None:
            continue
        try:
            vec = decode_vector(r["vector_json"])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if len(vec) != dim:
            vec = list(vec) + [0.0] * (dim - len(vec))
        conn.execute(
            "INSERT OR IGNORE INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
            (rowid, _pack_vector(vec)),
        )
    log.info("backfilled vec_episodes from %d episode rows", len(rows))


def _backfill_vec_facts(conn: sqlite3.Connection, dim: int) -> None:
    """Populate vec_facts (rowid = narrative_facts.id, an INTEGER PRIMARY KEY,
    so VACUUM-stable like vec_edges) from the JSON mirror on cold start / dim
    change. Suppresses its own missing-table error so ensure_vec_table keeps
    working against a pre-v26 store."""
    with contextlib.suppress(sqlite3.OperationalError):
        rows = conn.execute(
            "SELECT fact_id, vector_json FROM narrative_fact_embeddings"
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
            if len(vec) != dim:
                vec = list(vec) + [0.0] * (dim - len(vec))
            conn.execute(
                "INSERT OR IGNORE INTO vec_facts(rowid, embedding) VALUES (?, ?)",
                (r["fact_id"], _pack_vector(vec)),
            )
        log.info("backfilled vec_facts from %d fact rows", len(rows))


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
    for fts in _ROWID_FTS_TABLES:
        with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
            conn.execute(f"INSERT INTO {fts}({fts}) VALUES('rebuild')")
    if not _load_vec_extension(conn):
        return
    dim_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
    ).fetchone()
    if not dim_row:
        return
    dim = int(dim_row["value"])
    for table, backfill in (
        ("vec_chunks", _backfill_vec),
        ("vec_episodes", _backfill_vec_episodes),
    ):
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            _ensure_vec_table_named(conn, table, dim)
            backfill(conn, dim)
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
    dim = int(dim_row["value"])
    half = max(1, sample // 2)
    try:
        rows = conn.execute(
            """
            SELECT * FROM (
                SELECT e.rowid AS rid, em.vector_json
                FROM episodes e JOIN episode_embeddings em ON em.episode_id = e.id
                ORDER BY e.rowid ASC LIMIT ?
            )
            UNION
            SELECT * FROM (
                SELECT e.rowid AS rid, em.vector_json
                FROM episodes e JOIN episode_embeddings em ON em.episode_id = e.id
                ORDER BY e.rowid DESC LIMIT ?
            )
            """,
            (half, half),
        ).fetchall()
        for r in rows:
            vec = decode_vector(r["vector_json"])
            if len(vec) != dim:
                vec = list(vec) + [0.0] * (dim - len(vec))
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
    chunk_count = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    if not chunk_count:
        return

    from hymem.dreaming.mentions import index_chunk_mentions

    rows = conn.execute("SELECT id, text FROM chunks").fetchall()
    total = 0
    for row in rows:
        total += index_chunk_mentions(conn, row["id"], row["text"])
    log.info("backfilled entity_mentions: chunks=%d mentions=%d", len(rows), total)
