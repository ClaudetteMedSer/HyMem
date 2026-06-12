from __future__ import annotations

import logging
import sqlite3

from hymem.config import HyMemConfig
from hymem.core.db import backfill_entity_mentions
from hymem.dreaming import bitemporal

log = logging.getLogger("hymem.dreaming.phase3")


def _chunk_first_message_role(conn: sqlite3.Connection, chunk_id: str) -> str | None:
    """Role of the chunk's first message — the author signal for speaker-
    weighted reinforcement. None if the chunk or message is gone."""
    row = conn.execute(
        "SELECT m.role FROM chunks c JOIN messages m ON m.id = c.start_message_id "
        "WHERE c.id = ?",
        (chunk_id,),
    ).fetchone()
    return row["role"] if row else None


def decay(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    """Co-occurrence-aware decay + negative-dominance retraction.

    Two retraction paths:
      1. Topic re-mentioned without reinforcement -> bump neg_evidence, then
         retract if smoothed confidence falls below cfg.retract_threshold.
      2. neg_evidence >= 2*pos_evidence + cfg.zombie_neg_threshold -> retract
         immediately. Catches edges where negatives clearly dominate but the
         smoothed-confidence rule would leave them stranded for several more
         cycles (e.g. +0/-2, +1/-4, +2/-6).

    Stable facts in dormant topics are left alone.
    """
    default_window = int(cfg.decay_window_days)
    # The recency probe ("was this topic discussed lately without reinforcing
    # the edge?") stays on the global window; only an edge's *eligibility* for
    # a negative bump is stretched per-predicate, so sticky predicates decay
    # slower without becoming more sensitive to recent mentions.
    mention_cutoff = f"-{default_window} days"

    # Catch chunks that exist but were never indexed (e.g. pre-upgrade DBs).
    backfill_entity_mentions(conn)

    active_predicates = [
        r["predicate"]
        for r in conn.execute(
            "SELECT DISTINCT predicate FROM knowledge_graph "
            "WHERE status = 'active' AND derived = 0"
        ).fetchall()
    ]

    for predicate in active_predicates:
        elig_window = int(cfg.predicate_half_life_days.get(predicate, default_window))
        elig_cutoff = f"-{elig_window} days"

        rows = conn.execute(
            """
            SELECT id, subject_canonical, object_canonical
            FROM knowledge_graph
            WHERE status = 'active'
              AND derived = 0
              AND predicate = ?
              AND (last_reinforced IS NULL OR last_reinforced < datetime('now', ?))
            """,
            (predicate, elig_cutoff),
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
                (subj, obj, mention_cutoff, edge_id),
            ).fetchone()

            if not recent_mention:
                continue

            # Topic discussed without reinforcement → treat as soft contradiction.
            conn.execute(
                "UPDATE knowledge_graph SET neg_evidence = neg_evidence + 1 WHERE id = ?",
                (edge_id,),
            )

    # Find every edge that will be retracted this pass — either by smoothed
    # confidence falling below the threshold, or by the negative-dominance
    # rule (neg >= 2*pos + zombie_neg_threshold). The dominance rule
    # generalizes the original zero-positive zombie rule: at pos=0 it reduces
    # to neg>=threshold (catching the historical 55 zombies), and at pos=1 it
    # fires at neg>=threshold+2 (catching gray-zone edges like
    # `hook uses nohup` +1/-4 that the smoothed-confidence rule misses).
    # We select first so we can log feedback before flipping status.
    to_retract = conn.execute(
        """
        SELECT id, subject_canonical, predicate, object_canonical
        FROM knowledge_graph
        WHERE status = 'active'
          AND derived = 0
          AND (
              (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) < ?
              OR neg_evidence >= 2 * pos_evidence + ?
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
        # Close the validity interval: these facts were superseded by the
        # contradicting evidence that drove the decay (schema v15 valid time).
        bitemporal.stamp_invalidation(conn, ids)


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
            SELECT em_s.chunk_id AS chunk_id
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

        # Speaker-weighted reinforcement: a co-mention in a user-opened chunk
        # counts more than one prefixed by (possibly confabulated) assistant
        # context. Defaults to weight 1 for assistant-prefixed chunks.
        role = _chunk_first_message_role(conn, comention["chunk_id"])
        weight = cfg.evidence_role_weights.get(role, 1) if role else 1

        conn.execute(
            "UPDATE knowledge_graph "
            "SET pos_evidence = pos_evidence + ?, "
            "    last_reinforced = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (weight, edge_id),
        )
        bumped += 1

    if bumped:
        log.info("phase3.reinforce edges_bumped=%d", bumped)


def _record_retraction_feedback(conn: sqlite3.Connection, edge: sqlite3.Row) -> None:
    """Insert a row into extraction_feedback for an edge about to be
    auto-retracted. Prefer the most recent positive-evidence chunk (the chunk
    that produced the wrong extraction), but fall back to negative evidence —
    zombie edges only have polarity=-1 rows, and skipping them was leaving
    extraction_feedback permanently empty for the most useful negative cases.
    """
    evidence = conn.execute(
        """
        SELECT chunk_id FROM kg_evidence
        WHERE edge_id = ?
        ORDER BY polarity DESC, extracted_at DESC LIMIT 1
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
