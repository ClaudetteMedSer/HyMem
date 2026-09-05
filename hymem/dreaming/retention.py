from __future__ import annotations

import contextlib
import logging
import sqlite3

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.core.message_records import (
    canonical_message_record,
    chunk_contains_message_record,
)

log = logging.getLogger("hymem.dreaming.retention")


def prune_chunks(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    # Durable coverage artifacts have their own lifecycle and never compete
    # with selective retrieval/extraction chunks for this soft cap.
    total = conn.execute(
        "SELECT COUNT(*) AS c FROM chunks WHERE chunk_kind = 'extraction'"
    ).fetchone()["c"]
    if total <= cfg.max_chunks:
        return 0

    keep_ids: set[str] = set()

    rows = conn.execute("SELECT DISTINCT chunk_id FROM kg_evidence").fetchall()
    keep_ids.update(r["chunk_id"] for r in rows)
    rows = conn.execute(
        "SELECT DISTINCT chunk_id FROM kg_claim_observations"
    ).fetchall()
    keep_ids.update(r["chunk_id"] for r in rows)
    # Empty successful extraction outcomes have no observation/evidence row,
    # but are durable authority over any older non-empty portable snapshot.
    rows = conn.execute(
        "SELECT DISTINCT chunk_id FROM kg_claim_extraction_outcomes"
    ).fetchall()
    keep_ids.update(r["chunk_id"] for r in rows)

    rows = conn.execute(
        "SELECT id FROM chunks WHERE chunk_kind = 'extraction' "
        "AND created_at >= datetime('now', ?)",
        (f"-{cfg.retention_days} days",),
    ).fetchall()
    keep_ids.update(r["id"] for r in rows)

    excess = total - cfg.max_chunks
    if keep_ids:
        placeholders = ",".join("?" * len(keep_ids))
        rows = conn.execute(
            f"SELECT id, rowid FROM chunks WHERE chunk_kind = 'extraction' "
            f"AND id NOT IN ({placeholders}) "
            "ORDER BY created_at ASC LIMIT ?",
            tuple(keep_ids) + (excess,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, rowid FROM chunks WHERE chunk_kind = 'extraction' "
            "ORDER BY created_at ASC LIMIT ?",
            (excess,),
        ).fetchall()

    pruned = 0
    for r in rows:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (r["rowid"],))
        with core_db.evidence_mutation(conn):
            conn.execute(
                "UPDATE chunks SET source_manifest_version = NULL, "
                "source_manifest_count = NULL WHERE id = ? "
                "AND NOT EXISTS (SELECT 1 FROM kg_evidence ev "
                "WHERE ev.chunk_id = chunks.id) "
                "AND NOT EXISTS (SELECT 1 FROM kg_claim_observations observation "
                "WHERE observation.chunk_id = chunks.id) "
                "AND NOT EXISTS (SELECT 1 FROM kg_claim_extraction_outcomes outcome "
                "WHERE outcome.chunk_id = chunks.id)",
                (r["id"],),
            )
            conn.execute(
                "DELETE FROM chunk_message_sources WHERE chunk_id = ? "
                "AND EXISTS (SELECT 1 FROM chunks c WHERE c.id = ? "
                "AND c.source_manifest_version IS NULL)",
                (r["id"], r["id"]),
            )
            conn.execute("DELETE FROM chunks WHERE id = ?", (r["id"],))
        pruned += 1

    remaining = total - pruned
    log.info("retention.pruned pruned=%d kept=%d", pruned, remaining)
    return pruned


def prune_messages(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    """Prune individually old, losslessly covered messages from ended sessions.

    Raw retention is explicit opt-in. A summary or digest watermark is never
    deletion proof: each eligible message needs a current content fingerprint
    backed by a surviving same-session chunk that still contains the canonical
    JSONL source record. Deleting ``messages`` fires the external-content FTS
    delete trigger and cascades temporal mentions. The durable coverage ledger,
    chunk, KG evidence, and evidence counters deliberately survive.
    """
    days = int(cfg.message_retention_days)
    if days <= 0:
        return 0

    cutoff = f"-{days} days"
    candidates = conn.execute(
        """
        SELECT m.id, m.session_id, m.role, m.source_peer_id,
               m.source_workspace_id, m.content, m.created_at
        FROM messages m
        JOIN sessions s ON s.id = m.session_id
        WHERE s.ended_at IS NOT NULL
          AND m.created_at < datetime('now', ?)
        ORDER BY m.id
        """,
        (cutoff,),
    ).fetchall()

    pruned = 0
    for message in candidates:
        (
            expected_record,
            expected_hash,
            expected_hash_version,
            expected_record_version,
        ) = canonical_message_record(
            message_id=int(message["id"]),
            session_id=message["session_id"],
            role=message["role"],
            content=message["content"],
            source_created_at=message["created_at"],
            source_peer_id=message["source_peer_id"],
            source_workspace_id=message["source_workspace_id"],
        )
        coverage_rows = conn.execute(
            """
            SELECT c.session_id, c.start_message_id, c.end_message_id, c.text
            FROM message_retention_coverage mc
            JOIN chunks c ON c.id = mc.chunk_id
            WHERE mc.message_id = ?
              AND mc.source_session_id = ?
              AND mc.source_role = ?
              AND mc.source_peer_id IS ?
              AND mc.source_workspace_id IS ?
              AND mc.source_created_at IS ?
              AND mc.message_content_hash = ?
              AND mc.hash_version = ?
              AND mc.record_version = ?
            """,
            (
                message["id"],
                message["session_id"],
                message["role"],
                message["source_peer_id"],
                message["source_workspace_id"],
                message["created_at"],
                expected_hash,
                expected_hash_version,
                expected_record_version,
            ),
        ).fetchall()
        covered = any(
            row["session_id"] == message["session_id"]
            and int(row["start_message_id"])
            <= int(message["id"])
            <= int(row["end_message_id"])
            and chunk_contains_message_record(
                chunk_text=row["text"],
                record=expected_record,
            )
            for row in coverage_rows
        )
        if not covered:
            continue

        # Re-check mutable eligibility and exact source bytes in the DELETE so
        # a concurrent/out-of-band edit cannot race the Python hash validation.
        cur = conn.execute(
            """
            DELETE FROM messages
            WHERE id = ? AND session_id = ? AND role = ? AND content = ?
              AND source_peer_id IS ? AND source_workspace_id IS ?
              AND created_at < datetime('now', ?)
              AND EXISTS (
                  SELECT 1 FROM sessions s
                  WHERE s.id = messages.session_id AND s.ended_at IS NOT NULL
              )
            """,
            (
                message["id"],
                message["session_id"],
                message["role"],
                message["content"],
                message["source_peer_id"],
                message["source_workspace_id"],
                cutoff,
            ),
        )
        pruned += cur.rowcount or 0

    if pruned:
        log.info("retention.messages pruned=%d", pruned)
    return pruned


def prune_retracted_edges(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    """Opt-in hard-delete of old retracted graph history.

    Values <= 0 keep tombstones, evidence, and lifecycle forever (the default),
    which is required for complete ``facts_at`` valid-time reconstruction.
    Positive windows retain the explicit historical destructive behavior;
    derived edges are left alone because inference rebuilds them separately.
    """
    if cfg.tombstone_retention_days <= 0:
        return 0
    rows = conn.execute(
        """
        SELECT id FROM knowledge_graph
        WHERE status = 'retracted'
          AND derived = 0
          AND hymem_timestamp_at_or_before(
                last_seen,
                strftime('%Y-%m-%dT%H:%M:%fZ','now', ?)
              ) = 1
        ORDER BY id
        """,
        (f"-{int(cfg.tombstone_retention_days)} days",),
    ).fetchall()
    ids = [int(row["id"]) for row in rows]
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    with contextlib.suppress(sqlite3.OperationalError):
        conn.executemany("DELETE FROM vec_edges WHERE rowid = ?", [(i,) for i in ids])
    from hymem.core.db import evidence_destructive_mutation

    with evidence_destructive_mutation(conn):
        cur = conn.execute(
            f"DELETE FROM knowledge_graph WHERE id IN ({placeholders})",
            ids,
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
            "DELETE FROM episodes AS e "
            "WHERE created_at < datetime('now', ?) "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM sessions s WHERE s.id = e.session_id "
            "  AND e.digest_generation IS NOT NULL "
            "  AND e.digest_generation = s.digest_cursor_prompt_version "
            "  AND (s.digest_published_generation IS NULL "
            "       OR e.digest_generation <> s.digest_published_generation)"
            ")",
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
