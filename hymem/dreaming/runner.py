from __future__ import annotations

import contextlib
import logging
import os
import socket
import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import phase1, phase2, phase3
from hymem.dreaming.inference import infer_transitive_edges
from hymem.dreaming.chunks import (
    Chunk,
    extract_baseline_chunks,
    extract_high_salience_chunks,
    persist_chunks,
)
from hymem.dreaming.embeddings import (
    ChunkEmbedRequest,
    assemble_chunk_pending,
    fetch_chunk_embeddings,
    fetch_edge_embeddings,
    fetch_episode_embeddings,
    persist_chunk_embeddings,
    persist_edge_embeddings,
    persist_episode_embeddings,
    prepare_chunk_embed_batch,
)
from hymem.dreaming.episodes import extract_episodes_for_session, persist_episodes
from hymem.dreaming.procedures import extract_procedures_for_session, persist_procedures
from hymem.dreaming.mentions import index_chunk_mentions
from hymem.dreaming.retention import (
    prune_bookkeeping,
    prune_chunks,
    prune_episodes_and_procedures,
    prune_messages,
    prune_retracted_edges,
)
from hymem.dreaming.summary import extract_session_summary, persist_session_summary
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
    chunks_embedded_from_cache: int = 0
    edges_embedded: int = 0
    edges_embedded_from_cache: int = 0
    episodes_embedded: int = 0
    episodes_embedded_from_cache: int = 0
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

    embed_executor: ThreadPoolExecutor | None = None
    embed_inflight: list[tuple[ChunkEmbedRequest, Future]] = []
    if embedding_client is not None:
        embed_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hymem-embed"
        )

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

        def _kickoff_chunk_embed(chunks_list: list[Chunk]) -> None:
            """Cache lookup on the main thread, then submit the embedder call
            to a single-worker background thread so Phase 1 LLM calls keep
            running in parallel with the (I/O-bound) embedding API call.

            Skips chunks that already have a row in chunk_embeddings — a
            re-run dream cycle re-persists the same chunk objects, and
            vec_chunks (vec0 virtual table) rejects ``INSERT OR REPLACE`` on
            existing rowids.
            """
            if embed_executor is None or embedding_client is None or not chunks_list:
                return
            ids = [c.id for c in chunks_list]
            placeholders = ",".join("?" * len(ids))
            already_embedded = {
                r["chunk_id"]
                for r in conn.execute(
                    f"SELECT chunk_id FROM chunk_embeddings "
                    f"WHERE chunk_id IN ({placeholders})",
                    tuple(ids),
                ).fetchall()
            }
            fresh = [c for c in chunks_list if c.id not in already_embedded]
            if not fresh:
                return
            request = prepare_chunk_embed_batch(
                conn,
                [(c.id, c.text) for c in fresh],
                embedding_client,
            )
            client = embedding_client
            miss_texts = request.miss_texts
            future: Future = embed_executor.submit(
                lambda: client.embed(miss_texts) if miss_texts else []
            )
            embed_inflight.append((request, future))

        for session_id in target_sessions:
            report.sessions_processed += 1
            chunks = extract_high_salience_chunks(
                conn, session_id, min_chars=cfg.salience_min_chars
            )
            report.chunks_seen += len(chunks)

            if chunks:
                with core_db.transaction(conn):
                    persist_chunks(conn, chunks)
                    for chunk in chunks:
                        index_chunk_mentions(conn, chunk.id, chunk.text)
                _kickoff_chunk_embed(chunks)

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
                    extraction = phase1.extract_chunk_results(
                        conn, chunk, llm,
                        prompt_version=cfg.prompt_version,
                        negative_examples=negative_examples,
                    )
                except Exception:
                    log.exception("phase1.llm_failure chunk_id=%s", chunk.id)
                    continue
                if extraction is None:
                    continue
                with core_db.transaction(conn):
                    phase1.persist_chunk_results(
                        conn, chunk, extraction, prompt_version=cfg.prompt_version,
                        cfg=cfg, embedding_client=embedding_client,
                    )
                if extraction.triples or extraction.markers:
                    report.chunks_processed += 1
                    report.triples_extracted += len(extraction.triples)
                    report.markers_extracted += len(extraction.markers)

            # Baseline backstop: if budget remains after the salience tier,
            # pull plain chunks (newest first) so every chunk eventually flows
            # through extraction. Capped per cycle by dream_baseline_budget.
            baseline: list = []
            if chunks_remaining > 0:
                baseline_cap = min(chunks_remaining, cfg.dream_baseline_budget)
                baseline = extract_baseline_chunks(
                    conn,
                    session_id,
                    prompt_version=cfg.prompt_version,
                    limit=baseline_cap,
                    min_chars=cfg.salience_min_chars,
                )
                if baseline:
                    report.chunks_seen += len(baseline)
                    with core_db.transaction(conn):
                        persist_chunks(conn, baseline)
                        for chunk in baseline:
                            index_chunk_mentions(conn, chunk.id, chunk.text)
                    _kickoff_chunk_embed(baseline)
                    for chunk in baseline:
                        if chunks_remaining <= 0:
                            break
                        chunks_remaining -= 1
                        try:
                            extraction = phase1.extract_chunk_results(
                                conn, chunk, llm,
                                prompt_version=cfg.prompt_version,
                                negative_examples=negative_examples,
                            )
                        except Exception:
                            log.exception(
                                "phase1.llm_failure chunk_id=%s tier=baseline",
                                chunk.id,
                            )
                            continue
                        if extraction is None:
                            continue
                        with core_db.transaction(conn):
                            phase1.persist_chunk_results(
                                conn, chunk, extraction,
                                prompt_version=cfg.prompt_version,
                                cfg=cfg, embedding_client=embedding_client,
                            )
                        if extraction.triples or extraction.markers:
                            report.chunks_processed += 1
                            report.triples_extracted += len(extraction.triples)
                            report.markers_extracted += len(extraction.markers)

            # Sessions with no content in either tier skip the per-session tail
            # blocks — matches the pre-baseline behavior and avoids wasted LLM
            # calls on empty sessions.
            if not chunks and not baseline:
                continue

            try:
                episodes_ext = extract_episodes_for_session(conn, session_id, llm)
            except Exception:
                log.exception("episodes.extraction_failure session_id=%s", session_id)
            else:
                if episodes_ext is not None and episodes_ext.items:
                    with core_db.transaction(conn):
                        count = persist_episodes(conn, session_id, episodes_ext)
                    log.debug("episodes session_id=%s count=%d", session_id, count)

            try:
                summary = extract_session_summary(conn, session_id, llm)
            except Exception:
                log.exception("summary.failure session_id=%s", session_id)
            else:
                if summary:
                    with core_db.transaction(conn):
                        persist_session_summary(conn, session_id, summary)
                    log.debug("summary session_id=%s", session_id)

            try:
                procedures_ext = extract_procedures_for_session(conn, session_id, llm)
            except Exception:
                log.exception("procedures.extraction_failure session_id=%s", session_id)
            else:
                if procedures_ext is not None and procedures_ext.items:
                    with core_db.transaction(conn):
                        count = persist_procedures(conn, session_id, procedures_ext)
                    log.debug("procedures session_id=%s count=%d", session_id, count)

            if chunks_remaining <= 0:
                report.budget_exhausted = True
                log.info("dream.budget_exhausted budget=%d", cfg.dream_budget)
                break

        if embedding_client is not None:
            # Drain background-embedded batches first, in one transaction. A
            # per-batch future failure is logged and skipped — the post-loop
            # fetch_chunk_embeddings call below catches anything missed
            # (skipped sessions, future raises, miss_texts size mismatch).
            persisted_ids: set[str] = set()
            if embed_inflight:
                with core_db.transaction(conn):
                    for request, future in embed_inflight:
                        try:
                            miss_vectors = future.result()
                        except Exception:
                            log.exception(
                                "embedding.background_failure batch_size=%d",
                                len(request.ids),
                            )
                            continue
                        try:
                            pending = assemble_chunk_pending(
                                conn, request, miss_vectors,
                                exclude_ids=persisted_ids,
                            )
                        except RuntimeError:
                            log.exception(
                                "embedding.assemble_failure batch_size=%d",
                                len(request.ids),
                            )
                            continue
                        if pending is None:
                            continue
                        report.chunks_embedded += persist_chunk_embeddings(
                            conn, pending
                        )
                        report.chunks_embedded_from_cache += pending.cache_hits
                        persisted_ids.update(pending.ids)
                embed_inflight.clear()

            pending_chunks = fetch_chunk_embeddings(conn, embedding_client)
            if pending_chunks is not None:
                with core_db.transaction(conn):
                    report.chunks_embedded += persist_chunk_embeddings(conn, pending_chunks)
                report.chunks_embedded_from_cache += pending_chunks.cache_hits

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
            phase3.reinforce(conn, cfg)
            phase3.decay(conn, cfg)
            derived = infer_transitive_edges(conn, cfg)
            if derived:
                log.info("inference.derived count=%d", derived)
            pruned = prune_chunks(conn, cfg)
            pruned += prune_messages(conn, cfg)
            pruned += prune_retracted_edges(conn, cfg)
            pruned += prune_episodes_and_procedures(conn, cfg)
            pruned += prune_bookkeeping(conn, cfg)
            phase2.consolidate_insights(conn, cfg)  # refresh after decay
            conn.execute("DELETE FROM token_overlap_index")
            _canon_rows = conn.execute(
                "SELECT DISTINCT subject_canonical AS c FROM knowledge_graph WHERE status='active' "
                "UNION "
                "SELECT DISTINCT object_canonical FROM knowledge_graph WHERE status='active'"
            ).fetchall()
            _index_data = []
            for _r in _canon_rows:
                _c = _r["c"]
                for _tok in _c.split("_"):
                    if _tok:
                        _index_data.append((_tok, _c))
            if _index_data:
                conn.executemany(
                    "INSERT OR IGNORE INTO token_overlap_index(token, canonical) VALUES (?, ?)",
                    _index_data,
                )

        # VACUUM can't run inside a transaction, so it goes after the phase-3
        # block commits. Only pay the full-rewrite cost when a sweep actually
        # freed a meaningful number of pages.
        if cfg.vacuum_after_prune and pruned >= cfg.vacuum_min_pruned:
            conn.execute("VACUUM")
            log.info("retention.vacuum pruned=%d", pruned)

        if embedding_client is not None:
            pending_edges = fetch_edge_embeddings(conn, embedding_client)
            if pending_edges is not None:
                with core_db.transaction(conn):
                    report.edges_embedded = persist_edge_embeddings(conn, pending_edges)
                report.edges_embedded_from_cache = pending_edges.cache_hits

            pending_episodes = fetch_episode_embeddings(conn, embedding_client)
            if pending_episodes is not None:
                with core_db.transaction(conn):
                    report.episodes_embedded = persist_episode_embeddings(
                        conn, pending_episodes
                    )
                report.episodes_embedded_from_cache = pending_episodes.cache_hits
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
                edges_embedded = ?,
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
                report.edges_embedded,
                report.triples_extracted,
                report.markers_extracted,
                run_id,
            ),
        )
        log.info(
            "dream.end run_id=%d sessions=%d chunks_processed=%d/%d triples=%d markers=%d chunks_from_cache=%d edges_from_cache=%d budget_exhausted=%s",
            run_id,
            report.sessions_processed,
            report.chunks_processed,
            report.chunks_seen,
            report.triples_extracted,
            report.markers_extracted,
            report.chunks_embedded_from_cache,
            report.edges_embedded_from_cache,
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
        if embed_executor is not None:
            embed_executor.shutdown(wait=True)
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
