from __future__ import annotations

import contextlib
import logging
import sqlite3

from hymem.config import HyMemConfig

log = logging.getLogger("hymem.dreaming.retention")


def prune_chunks(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    total = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    if total <= cfg.max_chunks:
        return 0

    keep_ids: set[str] = set()

    rows = conn.execute(
        "SELECT DISTINCT chunk_id FROM kg_evidence "
        "WHERE edge_id IN (SELECT id FROM knowledge_graph WHERE status = 'active')"
    ).fetchall()
    keep_ids.update(r["chunk_id"] for r in rows)

    rows = conn.execute(
        "SELECT id FROM chunks WHERE created_at >= datetime('now', ?)",
        (f"-{cfg.retention_days} days",),
    ).fetchall()
    keep_ids.update(r["id"] for r in rows)

    excess = total - cfg.max_chunks
    if keep_ids:
        placeholders = ",".join("?" * len(keep_ids))
        rows = conn.execute(
            f"SELECT id, rowid FROM chunks WHERE id NOT IN ({placeholders}) "
            "ORDER BY created_at ASC LIMIT ?",
            tuple(keep_ids) + (excess,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, rowid FROM chunks ORDER BY created_at ASC LIMIT ?",
            (excess,),
        ).fetchall()

    pruned = 0
    for r in rows:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (r["rowid"],))
        conn.execute("DELETE FROM chunks WHERE id = ?", (r["id"],))
        pruned += 1

    remaining = total - pruned
    log.info("retention.pruned pruned=%d kept=%d", pruned, remaining)
    return pruned


def prune_messages(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    """Delete raw messages of sessions older than message_retention_days that
    already carry a summary. Summary-gated so it's a no-op without an LLM (no
    summary is ever written), never destroying the only copy of data we can't
    reconstruct. Chunks keep their start/end_message_id provenance pointers
    (plain INTEGERs, not FKs); phase3 speaker-weighting degrades to weight 1
    when the joined message is gone — already the documented default."""
    cur = conn.execute(
        """
        DELETE FROM messages
        WHERE session_id IN (
            SELECT id FROM sessions
            WHERE summary IS NOT NULL
              AND started_at < datetime('now', ?)
        )
        """,
        (f"-{int(cfg.message_retention_days)} days",),
    )
    pruned = cur.rowcount or 0
    if pruned:
        log.info("retention.messages pruned=%d", pruned)
    return pruned


def prune_retracted_edges(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    """Hard-delete retracted knowledge_graph tombstones older than
    tombstone_retention_days. kg_evidence cascades via ON DELETE CASCADE.
    Derived edges are left alone (they're rebuilt each cycle by inference)."""
    cur = conn.execute(
        """
        DELETE FROM knowledge_graph
        WHERE status = 'retracted'
          AND derived = 0
          AND last_seen < datetime('now', ?)
        """,
        (f"-{int(cfg.tombstone_retention_days)} days",),
    )
    pruned = cur.rowcount or 0
    if pruned:
        log.info("retention.tombstones pruned=%d", pruned)
    return pruned


def prune_episodes_and_procedures(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    """Age out stale procedures using the retention_days window. Episodes are
    only aged out when episode_retention_days > 0: the default (0) keeps them
    forever because they're the leaves of the aggregation/digest tree, which
    full-rebuilds from live episodes each dream — deleting them makes the
    digest forget. FTS shadow tables and episode_embeddings are cleaned by the
    existing delete triggers / ON DELETE CASCADE."""
    pruned = 0

    if cfg.episode_retention_days > 0:
        cur = conn.execute(
            "DELETE FROM episodes WHERE created_at < datetime('now', ?)",
            (f"-{int(cfg.episode_retention_days)} days",),
        )
        pruned += cur.rowcount or 0

    cur = conn.execute(
        "DELETE FROM procedures WHERE status = 'stale' "
        "AND created_at < datetime('now', ?)",
        (f"-{int(cfg.retention_days)} days",),
    )
    pruned += cur.rowcount or 0

    if pruned:
        log.info("retention.episodes_procedures pruned=%d", pruned)
    return pruned


def prune_bookkeeping(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    """Cap the append-only bookkeeping tables, keeping only the newest rows.
    dream_runs grows one row per cycle; extraction_feedback grows per retraction
    though only the 10 newest are ever read."""
    pruned = 0

    cur = conn.execute(
        """
        DELETE FROM dream_runs
        WHERE id NOT IN (
            SELECT id FROM dream_runs ORDER BY started_at DESC LIMIT ?
        )
        """,
        (int(cfg.dream_runs_keep),),
    )
    pruned += cur.rowcount or 0

    cur = conn.execute(
        """
        DELETE FROM extraction_feedback
        WHERE id NOT IN (
            SELECT id FROM extraction_feedback ORDER BY created_at DESC LIMIT ?
        )
        """,
        (int(cfg.extraction_feedback_keep),),
    )
    pruned += cur.rowcount or 0

    if pruned:
        log.info("retention.bookkeeping pruned=%d", pruned)
    return pruned
