from __future__ import annotations

import sqlite3

from hymem.config import HyMemConfig


def decay(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    """Co-occurrence-aware decay.

    An edge decays only if its subject or object was *re-mentioned* recently
    without reinforcing the edge — i.e., the topic came up again and the user
    didn't restate the relationship. Stable facts in dormant topics are left
    alone.

    Implementation: for each edge whose last_reinforced is older than the decay
    window, check whether any chunk in that window touched the edge's subject or
    object surface form. If yes -> decay; if no -> leave alone.
    """
    cutoff_arg = f"-{int(cfg.decay_window_days)} days"

    rows = conn.execute(
        """
        SELECT id, subject_canonical, object_canonical
        FROM knowledge_graph
        WHERE status = 'active'
          AND (last_reinforced IS NULL OR last_reinforced < datetime('now', ?))
        """,
        (cutoff_arg,),
    ).fetchall()

    for row in rows:
        edge_id = row["id"]
        subj = row["subject_canonical"]
        obj = row["object_canonical"]

        recent_mention = conn.execute(
            """
            SELECT 1 FROM chunks
            WHERE created_at >= datetime('now', ?)
              AND (LOWER(text) LIKE '%' || ? || '%' OR LOWER(text) LIKE '%' || ? || '%')
              AND id NOT IN (
                  SELECT chunk_id FROM kg_evidence WHERE edge_id = ?
              )
            LIMIT 1
            """,
            (cutoff_arg, subj.replace("_", " "), obj.replace("_", " "), edge_id),
        ).fetchone()

        if not recent_mention:
            continue

        # Topic discussed without reinforcement → treat as soft contradiction.
        conn.execute(
            "UPDATE knowledge_graph SET neg_evidence = neg_evidence + 1 WHERE id = ?",
            (edge_id,),
        )

    # Anything whose smoothed confidence dropped below the threshold gets
    # retracted (kept for audit; excluded from query results).
    conn.execute(
        """
        UPDATE knowledge_graph
        SET status = 'retracted'
        WHERE status = 'active'
          AND (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) < ?
        """,
        (cfg.retract_threshold,),
    )
