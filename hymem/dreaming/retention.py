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
