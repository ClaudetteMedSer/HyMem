from __future__ import annotations

import logging
import sqlite3

from hymem.config import HyMemConfig
from hymem.core.db import backfill_entity_mentions

log = logging.getLogger("hymem.dreaming.phase3")


def decay(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    """Co-occurrence-aware decay + zero-positive retraction.

    Two retraction paths:
      1. Topic re-mentioned without reinforcement -> bump neg_evidence, then
         retract if smoothed confidence falls below cfg.retract_threshold.
      2. Edge has pos_evidence=0 and neg_evidence>=cfg.zombie_neg_threshold ->
         retract immediately. Catches facts extracted once and contradicted
         every time after, which the smoothed formula leaves stranded for
         several more cycles.

    Stable facts in dormant topics are left alone.
    """
    cutoff_arg = f"-{int(cfg.decay_window_days)} days"

    # Catch chunks that exist but were never indexed (e.g. pre-upgrade DBs).
    backfill_entity_mentions(conn)

    rows = conn.execute(
        """
        SELECT id, subject_canonical, object_canonical
        FROM knowledge_graph
        WHERE status = 'active'
          AND derived = 0
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
            SELECT 1 FROM entity_mentions em
            JOIN chunks c ON c.id = em.chunk_id
            WHERE em.entity_canonical IN (?, ?)
              AND c.created_at >= datetime('now', ?)
              AND em.chunk_id NOT IN (
                  SELECT chunk_id FROM kg_evidence WHERE edge_id = ?
              )
            LIMIT 1
            """,
            (subj, obj, cutoff_arg, edge_id),
        ).fetchone()

        if not recent_mention:
            continue

        # Topic discussed without reinforcement → treat as soft contradiction.
        conn.execute(
            "UPDATE knowledge_graph SET neg_evidence = neg_evidence + 1 WHERE id = ?",
            (edge_id,),
        )

    # Find every edge that will be retracted this pass — either by smoothed
    # confidence falling below the threshold, or by the zero-positive rule. We
    # select first so we can log feedback before flipping status.
    to_retract = conn.execute(
        """
        SELECT id, subject_canonical, predicate, object_canonical
        FROM knowledge_graph
        WHERE status = 'active'
          AND derived = 0
          AND (
              (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) < ?
              OR (pos_evidence = 0 AND neg_evidence >= ?)
          )
        """,
        (cfg.retract_threshold, cfg.zombie_neg_threshold),
    ).fetchall()

    for edge in to_retract:
        _record_retraction_feedback(conn, edge)

    if to_retract:
        ids = [e["id"] for e in to_retract]
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"UPDATE knowledge_graph SET status = 'retracted' WHERE id IN ({placeholders})",
            ids,
        )


def reinforce(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    """Soft positive reinforcement from co-mention.

    Mirror of decay: if a chunk in the reinforcement window mentions BOTH
    subject and object of an active edge — and that chunk hasn't already
    produced a kg_evidence row for the edge — bump pos_evidence by 1. The
    bump is capped at one per edge per cycle (we don't iterate all matching
    chunks). Co-occurrence is weak evidence, but it's how singleton edges
    (60% of the graph) ever get a second positive.
    """
    cutoff_arg = f"-{int(cfg.reinforce_window_days)} days"

    rows = conn.execute(
        """
        SELECT id, subject_canonical, object_canonical
        FROM knowledge_graph
        WHERE status = 'active'
          AND derived = 0
        """
    ).fetchall()

    bumped = 0
    for row in rows:
        edge_id = row["id"]
        subj = row["subject_canonical"]
        obj = row["object_canonical"]

        comention = conn.execute(
            """
            SELECT 1
            FROM entity_mentions em_s
            JOIN entity_mentions em_o
              ON em_s.chunk_id = em_o.chunk_id
            JOIN chunks c ON c.id = em_s.chunk_id
            WHERE em_s.entity_canonical = ?
              AND em_o.entity_canonical = ?
              AND c.created_at >= datetime('now', ?)
              AND em_s.chunk_id NOT IN (
                  SELECT chunk_id FROM kg_evidence WHERE edge_id = ?
              )
            LIMIT 1
            """,
            (subj, obj, cutoff_arg, edge_id),
        ).fetchone()

        if not comention:
            continue

        conn.execute(
            "UPDATE knowledge_graph "
            "SET pos_evidence = pos_evidence + 1, "
            "    last_reinforced = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (edge_id,),
        )
        bumped += 1

    if bumped:
        log.info("phase3.reinforce edges_bumped=%d", bumped)


def _record_retraction_feedback(conn: sqlite3.Connection, edge: sqlite3.Row) -> None:
    """Insert a row into extraction_feedback for the most recent positive
    evidence chunk of an edge that is about to be auto-retracted. Mirrors the
    pattern used by HyMem.retract_edge so few-shot negatives include both
    manually and automatically retracted facts.
    """
    evidence = conn.execute(
        """
        SELECT chunk_id FROM kg_evidence
        WHERE edge_id = ? AND polarity = 1
        ORDER BY extracted_at DESC LIMIT 1
        """,
        (edge["id"],),
    ).fetchone()
    if evidence is None:
        return

    chunk = conn.execute(
        "SELECT text FROM chunks WHERE id = ?", (evidence["chunk_id"],)
    ).fetchone()
    if chunk is None:
        return

    conn.execute(
        """
        INSERT OR IGNORE INTO extraction_feedback
            (chunk_id, chunk_text_snippet, extracted_subject,
             extracted_predicate, extracted_object, feedback_type)
        VALUES (?, ?, ?, ?, ?, 'retracted')
        """,
        (
            evidence["chunk_id"],
            chunk["text"][:600],
            edge["subject_canonical"],
            edge["predicate"],
            edge["object_canonical"],
        ),
    )
