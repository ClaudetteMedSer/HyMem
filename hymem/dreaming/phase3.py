from __future__ import annotations

import logging
import sqlite3

from hymem.config import HyMemConfig
from hymem.core.db import backfill_entity_mentions
from hymem.core.graph import graph_clock_order_sql, live_edge_predicate
from hymem.dreaming import evidence

log = logging.getLogger("hymem.dreaming.phase3")


def decay(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    """Co-occurrence-aware decay + negative-dominance retraction.

    Two retraction paths:
      1. Topic re-mentioned without reinforcement -> record soft-negative
         evidence, then
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
            f"SELECT DISTINCT predicate FROM knowledge_graph kg "
            f"WHERE {live_edge_predicate('kg')}"
        ).fetchall()
    ]

    for predicate in active_predicates:
        elig_window = int(cfg.predicate_half_life_days.get(predicate, default_window))
        elig_cutoff = f"-{elig_window} days"

        rows = conn.execute(
            f"""
            SELECT kg.id, kg.subject_canonical, kg.object_canonical
            FROM knowledge_graph kg
            WHERE {live_edge_predicate('kg')}
              AND kg.predicate = ?
              AND (kg.last_reinforced IS NULL
                   OR hymem_timestamp_at_or_before(
                        kg.last_reinforced,
                        strftime('%Y-%m-%dT%H:%M:%fZ','now', ?)
                      ) = 1)
            """,
            (predicate, elig_cutoff),
        ).fetchall()

        for row in rows:
            edge_id = row["id"]
            subj = row["subject_canonical"]
            obj = row["object_canonical"]

            recent_mention = conn.execute(
                """
                SELECT em.chunk_id AS chunk_id FROM entity_mentions em
                JOIN chunks c ON c.id = em.chunk_id
                WHERE em.entity_canonical IN (?, ?)
                  AND hymem_timestamp_at_or_before(
                        strftime('%Y-%m-%dT%H:%M:%fZ','now', ?),
                        c.created_at
                      ) = 1
                  AND hymem_timestamp_at_or_before(
                        c.created_at,
                        strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
                      ) = 1
                  AND NOT EXISTS (
                      SELECT 1 FROM kg_evidence ev
                      WHERE ev.edge_id = ? AND ev.chunk_id = em.chunk_id
                        AND ev.is_current = 1
                        AND ev.evidence_kind <> 'decay'
                  )
                ORDER BY c.end_message_id DESC,
                         hymem_normalize_iso_timestamp(c.created_at) DESC,
                         em.chunk_id DESC
                LIMIT 1
                """,
                (subj, obj, mention_cutoff, edge_id),
            ).fetchone()

            if not recent_mention:
                continue

            # Topic discussed without reinforcement → one soft-negative
            # observation for this exact source chunk.  The ledger row is the
            # idempotency key, so unchanged dream cycles cannot repeatedly tax
            # the edge for the same conversation.
            evidence.record_chunk_evidence(
                conn,
                edge_id=edge_id,
                chunk_id=recent_mention["chunk_id"],
                evidence_kind="decay",
                polarity=-1,
                evidence_weight=1,
                weight_source="fixed_soft_decay:1",
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
        f"""
        SELECT kg.id, kg.subject_canonical, kg.predicate, kg.object_canonical
        FROM knowledge_graph kg
        WHERE {live_edge_predicate('kg', require_positive_majority=False)}
          AND (
              (kg.pos_evidence + 1.0)
                / (kg.pos_evidence + kg.neg_evidence + 2.0) < ?
              OR kg.neg_evidence >= 2 * kg.pos_evidence + ?
          )
        """,
        (cfg.retract_threshold, cfg.zombie_neg_threshold),
    ).fetchall()

    retracted_edges: list[sqlite3.Row] = []
    for edge in to_retract:
        positive = conn.execute(
            """
            SELECT MAX(hymem_normalize_iso_timestamp(source_event_at)) AS positive_at
            FROM kg_evidence
            WHERE edge_id = ? AND is_current = 1 AND polarity = 1
              AND provenance_status = 'canonical'
              AND hymem_normalize_iso_timestamp(source_event_at) IS NOT NULL
            """,
            (edge["id"],),
        ).fetchone()
        causes = conn.execute(
            """
            SELECT id, provenance_status, source_session_id,
                   source_message_id, evidence_kind, revision, chunk_id
            FROM kg_evidence
            WHERE edge_id = ? AND is_current = 1 AND polarity = -1
              AND hymem_timestamp_at_or_before(
                    extracted_at,
                    strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
                  ) = 1
              AND (
                provenance_status <> 'canonical'
                OR (
                  hymem_event_clock_is_valid(source_event_at,extracted_at)=1
                  AND hymem_timestamp_at_or_before(
                        published_at,
                        strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
                      )=1
                )
              )
            ORDER BY provenance_status, source_session_id,
                     source_message_id, evidence_kind, revision, chunk_id
            """,
            (edge["id"],),
        ).fetchall()
        if not causes:
            continue
        from hymem.dreaming.bitemporal import evidence_event_at

        cause_coordinates = {
            int(row["id"]): evidence_event_at(conn, int(row["id"]))
            for row in causes
        }
        positive_at = positive["positive_at"]
        close_at = max(cause_coordinates.values())
        if positive_at is not None and close_at < positive_at:
            # Accumulated but older contradictions cannot close a newer
            # assertion merely because their rows arrived later.
            continue
        from hymem.dreaming.bitemporal import record_lifecycle_event

        record_lifecycle_event(
            conn,
            edge_id=int(edge["id"]),
            event_key=evidence.phase3_retraction_event_key(
                conn, [int(row["id"]) for row in causes]
            ),
            event_kind="phase3_retraction",
            direction=-1,
            event_at=close_at,
            dependency_evidence_ids=[int(row["id"]) for row in causes],
            details="confidence_or_negative_dominance",
        )
        state = conn.execute(
            "SELECT status, invalid_at FROM knowledge_graph WHERE id = ?",
            (edge["id"],),
        ).fetchone()
        if state is not None and state["status"] == "retracted" \
                and state["invalid_at"] is not None:
            _record_retraction_feedback(conn, edge)
            retracted_edges.append(edge)

    if retracted_edges:
        log.info("phase3.retracted edges=%d", len(retracted_edges))


def reinforce(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    """Soft positive reinforcement from co-mention.

    Mirror of decay: the newest eligible chunk in the reinforcement window that
    mentions BOTH endpoints contributes one source-keyed positive observation.
    Re-running against the same newest chunk is a no-op; a genuinely newer
    co-mention can contribute once. Co-occurrence is weak evidence, but it's how
    singleton edges (60% of the graph) ever get a second positive.
    """
    cutoff_arg = f"-{int(cfg.reinforce_window_days)} days"

    rows = conn.execute(
        f"""
        SELECT kg.id, kg.subject_canonical, kg.object_canonical
        FROM knowledge_graph kg
        WHERE {live_edge_predicate('kg')}
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
              AND hymem_timestamp_at_or_before(
                    strftime('%Y-%m-%dT%H:%M:%fZ','now', ?),
                    c.created_at
                  ) = 1
              AND hymem_timestamp_at_or_before(
                    c.created_at,
                    strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
                  ) = 1
              AND NOT EXISTS (
                  SELECT 1 FROM kg_evidence ev
                  WHERE ev.edge_id = ? AND ev.chunk_id = em_s.chunk_id
                    AND ev.is_current = 1
                    AND ev.evidence_kind <> 'reinforcement'
              )
            ORDER BY c.end_message_id DESC,
                     hymem_normalize_iso_timestamp(c.created_at) DESC,
                     em_s.chunk_id DESC
            LIMIT 1
            """,
            (subj, obj, cutoff_arg, edge_id),
        ).fetchone()

        if not comention:
            continue

        # Co-mention is a synthetic signal, not a quoted claim. It therefore
        # stays explicitly unattributed at fixed weak weight instead of
        # borrowing the first speaker of a mixed-role chunk.
        mutation = evidence.record_chunk_evidence(
            conn,
            edge_id=edge_id,
            chunk_id=comention["chunk_id"],
            evidence_kind="reinforcement",
            polarity=1,
            evidence_weight=1,
            weight_source="fixed_unattributed_comention:1",
        )
        if mutation.contribution_changed:
            conn.execute(
                "UPDATE knowledge_graph "
                "SET last_reinforced = CURRENT_TIMESTAMP, "
                "    invalid_at = NULL "
                "WHERE id = ?",
                (edge_id,),
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
        f"""
        SELECT chunk_id FROM kg_evidence
        WHERE edge_id = ? AND is_current = 1
        ORDER BY polarity DESC, {graph_clock_order_sql('extracted_at')}, id
        LIMIT 1
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
