from __future__ import annotations

import contextlib
import logging
import os
import socket
import sqlite3
from dataclasses import dataclass

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import phase1, phase2, phase3
from hymem.dreaming.inference import infer_transitive_edges
from hymem.dreaming.chunks import extract_high_salience_chunks, persist_chunks
from hymem.dreaming.embeddings import embed_pending_chunks
from hymem.dreaming.episodes import extract_episodes_for_session
from hymem.dreaming.procedures import extract_procedures_for_session
from hymem.dreaming.mentions import index_chunk_mentions
from hymem.dreaming.retention import prune_chunks
from hymem.dreaming.summary import summarize_session
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient

log = logging.getLogger("hymem.dreaming")


@dataclass
class DreamReport:
    sessions_processed: int = 0
    chunks_seen: int = 0
    chunks_processed: int = 0
    triples_extracted: int = 0
    markers_extracted: int = 0
    chunks_embedded: int = 0
    skipped_locked: bool = False
    budget_exhausted: bool = False


def run_dreaming(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    llm: LLMClient,
    *,
    session_ids: list[str] | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> DreamReport:
    """Run all three dreaming phases. Holds an advisory lock so concurrent runs
    bail out instead of double-processing.
    """
    report = DreamReport()
    holder = f"{socket.gethostname()}:{os.getpid()}"

    run_id = conn.execute(
        "INSERT INTO dream_runs(started_at) VALUES (CURRENT_TIMESTAMP)"
    ).lastrowid

    if not _acquire_lock(conn, holder):
        report.skipped_locked = True
        log.info("dream.skipped_locked")
        conn.execute(
            "UPDATE dream_runs SET ended_at = CURRENT_TIMESTAMP, skipped_locked = 1 WHERE id = ?",
            (run_id,),
        )
        return report

    try:
        target_sessions = session_ids or _all_sessions(conn)
        log.info(
            "dream.start run_id=%d sessions=%d", run_id, len(target_sessions)
        )

        # Load recent extraction feedback for few-shot negative examples
        feedback_rows = conn.execute(
            """SELECT extracted_subject, extracted_predicate, extracted_object
               FROM extraction_feedback
               ORDER BY created_at DESC LIMIT 10"""
        ).fetchall()
        negative_examples = ""
        if feedback_rows:
            lines = [
                f"- \"{r['extracted_subject']} {r['extracted_predicate']} {r['extracted_object']}\" was WRONG. Do not extract this relationship."
                for r in feedback_rows
            ]
            negative_examples = "\n".join(lines)

        chunks_remaining = cfg.dream_budget

        for session_id in target_sessions:
            report.sessions_processed += 1
            chunks = extract_high_salience_chunks(
                conn, session_id, min_chars=cfg.salience_min_chars
            )
            report.chunks_seen += len(chunks)
            if not chunks:
                continue

            with core_db.transaction(conn):
                persist_chunks(conn, chunks)
                for chunk in chunks:
                    index_chunk_mentions(conn, chunk.id, chunk.text)

            for chunk in chunks:
                if chunks_remaining <= 0:
                    break
                # Skip chunks already processed with the current prompt version
                # without consuming the budget, so unprocessed chunks at the tail
                # of the session list don't get starved.
                already = conn.execute(
                    "SELECT 1 FROM processed_chunks WHERE chunk_id = ? AND prompt_version = ?",
                    (chunk.id, cfg.prompt_version),
                ).fetchone()
                if already:
                    continue
                chunks_remaining -= 1
                try:
                    with core_db.transaction(conn):
                        triples, markers = phase1.process_chunk(
                            conn, chunk, llm, prompt_version=cfg.prompt_version,
                            negative_examples=negative_examples,
                        )
                        if triples or markers:
                            report.chunks_processed += 1
                            report.triples_extracted += len(triples)
                            report.markers_extracted += len(markers)
                except Exception:
                    log.exception("phase1.llm_failure chunk_id=%s", chunk.id)
                    continue

            try:
                with core_db.transaction(conn):
                    episodes = extract_episodes_for_session(conn, session_id, llm)
                    log.debug("episodes session_id=%s count=%d", session_id, episodes)
            except Exception:
                log.exception("episodes.extraction_failure session_id=%s", session_id)

            try:
                with core_db.transaction(conn):
                    summary = summarize_session(conn, session_id, llm)
                    if summary:
                        log.debug("summary session_id=%s", session_id)
            except Exception:
                log.exception("summary.failure session_id=%s", session_id)

            try:
                with core_db.transaction(conn):
                    procs = extract_procedures_for_session(conn, session_id, llm)
                    if procs:
                        log.debug("procedures session_id=%s count=%d", session_id, procs)
            except Exception:
                log.exception("procedures.extraction_failure session_id=%s", session_id)

            if chunks_remaining <= 0:
                report.budget_exhausted = True
                log.info("dream.budget_exhausted budget=%d", cfg.dream_budget)
                break

        if embedding_client is not None:
            with core_db.transaction(conn):
                report.chunks_embedded = embed_pending_chunks(conn, embedding_client)

        log.info("phase2.start")
        with core_db.transaction(conn):
            phase2.consolidate_profile(conn, cfg)
            phase2.consolidate_insights(conn, cfg)
        profile_count = conn.execute(
            "SELECT COUNT(*) AS c FROM profile_entries"
        ).fetchone()["c"]
        log.info(
            "phase2.end profile_entries=%d insights=%d",
            profile_count,
            report.markers_extracted,
        )

        log.info("phase3.start")
        before_retracted = conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph WHERE status = 'retracted'"
        ).fetchone()["c"]
        with core_db.transaction(conn):
            phase3.decay(conn, cfg)
            derived = infer_transitive_edges(conn, cfg)
            if derived:
                log.info("inference.derived count=%d", derived)
            prune_chunks(conn, cfg)
            phase2.consolidate_insights(conn, cfg)  # refresh after decay
        after_retracted = conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph WHERE status = 'retracted'"
        ).fetchone()["c"]
        log.info("phase3.end retracted=%d", after_retracted - before_retracted)

        conn.execute(
            """
            UPDATE dream_runs
            SET ended_at = CURRENT_TIMESTAMP,
                sessions_processed = ?,
                chunks_seen = ?,
                chunks_processed = ?,
                chunks_embedded = ?,
                triples_extracted = ?,
                markers_extracted = ?,
                skipped_locked = 0
            WHERE id = ?
            """,
            (
                report.sessions_processed,
                report.chunks_seen,
                report.chunks_processed,
                report.chunks_embedded,
                report.triples_extracted,
                report.markers_extracted,
                run_id,
            ),
        )
        log.info(
            "dream.end run_id=%d sessions=%d chunks_processed=%d/%d triples=%d markers=%d budget_exhausted=%s",
            run_id,
            report.sessions_processed,
            report.chunks_processed,
            report.chunks_seen,
            report.triples_extracted,
            report.markers_extracted,
            report.budget_exhausted,
        )
        return report
    except Exception as exc:
        msg = str(exc)[:500]
        with contextlib.suppress(sqlite3.Error):
            conn.execute(
                "UPDATE dream_runs SET ended_at = CURRENT_TIMESTAMP, error = ? WHERE id = ?",
                (msg, run_id),
            )
        raise
    finally:
        _release_lock(conn, holder)


_LOCK_TTL_SECONDS = 300


def _acquire_lock(conn: sqlite3.Connection, holder: str) -> bool:
    try:
        conn.execute(
            "INSERT INTO run_lock(name, acquired_at, holder) VALUES ('dreaming', CURRENT_TIMESTAMP, ?)",
            (holder,),
        )
        return True
    except sqlite3.IntegrityError:
        pass

    # Lock exists — check whether it's stale (holder crashed without releasing).
    stale = conn.execute(
        "SELECT holder FROM run_lock WHERE name = 'dreaming'"
        " AND acquired_at < datetime('now', ?)",
        (f"-{_LOCK_TTL_SECONDS} seconds",),
    ).fetchone()
    if not stale:
        return False

    previous_holder = stale["holder"]
    log.warning(
        "dream.stale_lock_taken_over previous_holder=%s", previous_holder
    )
    conn.execute("DELETE FROM run_lock WHERE name = 'dreaming'")
    try:
        conn.execute(
            "INSERT INTO run_lock(name, acquired_at, holder) VALUES ('dreaming', CURRENT_TIMESTAMP, ?)",
            (holder,),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _release_lock(conn: sqlite3.Connection, holder: str) -> None:
    with contextlib.suppress(sqlite3.Error):
        conn.execute(
            "DELETE FROM run_lock WHERE name = 'dreaming' AND holder = ?",
            (holder,),
        )


def _all_sessions(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT id FROM sessions ORDER BY started_at").fetchall()
    return [r["id"] for r in rows]
