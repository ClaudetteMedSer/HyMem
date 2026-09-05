from __future__ import annotations

import contextlib
import logging
import os
import socket
import sqlite3
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import bitemporal, phase1, phase2, phase3
from hymem.dreaming.aggregate import build_aggregation_nodes
from hymem.dreaming.inference import infer_transitive_edges
from hymem.dreaming.chunks import (
    Chunk,
    chunk_extraction_is_quarantined,
    extract_baseline_chunks,
    extract_high_salience_chunks,
    load_pending_persisted_chunks,
    persist_chunks,
)
from hymem.dreaming.embeddings import (
    ChunkEmbedRequest,
    assemble_chunk_pending,
    fetch_chunk_embeddings,
    fetch_edge_embeddings,
    fetch_episode_embeddings,
    fetch_fact_embeddings,
    fetch_message_embeddings,
    message_embedding_id_batches,
    persist_chunk_embeddings,
    persist_edge_embeddings,
    persist_episode_embeddings,
    persist_fact_embeddings,
    persist_message_embeddings,
    prepare_chunk_embed_batch,
)
from hymem.dreaming.digest import (
    active_episode_prompt_version,
    digest_attempt_max_chars,
    digest_config_version,
    digest_generation_matches_config,
    digest_retry_policy_version,
    extract_session_digest,
    record_digest_failure,
)
from hymem.dreaming.lossless import covered_messages_after, materialize_message_coverage
from hymem.dreaming.episodes import persist_episodes
from hymem.dreaming.facts import (
    extract_facts,
    fact_cursor_retry_unit_key,
    facts_attempt_max_chars,
    facts_config_version,
    facts_retry_policy_version,
    facts_retry_state_is_valid,
    facts_tail_message_id,
    fact_session_authority_is_valid,
    next_fact_outcome_for_replay,
    persist_facts,
    reextract_fact_outcome,
    record_fact_failure,
    record_fact_failure_if_pending,
)
from hymem.dreaming.procedures import persist_procedures
from hymem.dreaming.mentions import index_chunk_mentions
from hymem.dreaming.temporal import index_chunk_temporal_mentions
from hymem.dreaming.retention import (
    prune_bookkeeping,
    prune_chunks,
    prune_episodes_and_procedures,
    prune_messages,
    prune_retracted_edges,
)
from hymem.dreaming.summary import persist_auto_session_summary
from hymem.dreaming.value_supersession import supersede_competing_values
from hymem.dreaming.user_profile import (
    PROFILE_PROMPT_VERSION,
    extract_user_profile,
    enforce_profile_redaction_policy,
    profile_attempt_max_chars,
    profile_config_version,
    profile_generation_matches_config,
    profile_retry_policy_version,
    profile_user_tail_message_id,
    publish_profile_generation,
    record_profile_failure,
    stage_profile_extraction,
)
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient
from hymem.core.graph import live_edge_predicate

log = logging.getLogger("hymem.dreaming")


def _lossless_cursor_is_valid(
    conn: sqlite3.Connection,
    session_id: str,
    cursor_message_id: int | None,
    partial_message_id: int | None,
    offset: int,
    *,
    roles: frozenset[str] | None = None,
) -> bool:
    """Prove that a persisted cursor names real producer-bounded artifacts."""
    try:
        if cursor_message_id is not None:
            at_cursor = covered_messages_after(
                conn, session_id, int(cursor_message_id) - 1, limit=1,
                roles=roles, through_message_id=int(cursor_message_id),
            )
            if not at_cursor or at_cursor[0].message_id != int(cursor_message_id):
                return False
        if partial_message_id is None:
            return int(offset) == 0
        if int(offset) <= 0:
            return False
        next_rows = covered_messages_after(
            conn, session_id, cursor_message_id, limit=1, roles=roles,
            through_message_id=int(partial_message_id),
        )
        return bool(
            next_rows
            and next_rows[0].message_id == int(partial_message_id)
            and int(offset) < len(next_rows[0].content)
        )
    except (RuntimeError, TypeError, ValueError):
        return False


@dataclass
class DreamReport:
    sessions_processed: int = 0
    chunks_seen: int = 0
    chunks_processed: int = 0
    # Extractions that did not complete this run (unparseable / wrong-shaped
    # reply). NOT persisted to dream_runs: those chunks are held unmarked and
    # retried, so a rising count across consecutive dreams is the ingest
    # analogue of a stuck fusion. In-memory + dream.end only.
    chunk_extraction_failures: int = 0
    triples_extracted: int = 0
    markers_extracted: int = 0
    rules_extracted: int = 0
    chunks_embedded: int = 0
    chunks_embedded_from_cache: int = 0
    messages_embedded: int = 0
    messages_embedded_from_cache: int = 0
    edges_embedded: int = 0
    edges_embedded_from_cache: int = 0
    episodes_embedded: int = 0
    episodes_embedded_from_cache: int = 0
    aggregation_nodes_built: int = 0
    aggregation_nodes_reused: int = 0
    aggregation_fusion_failures: int = 0
    aggregation_input_episodes: int = 0
    aggregation_level0_missed: int | None = None
    aggregation_leaf_changed: int | None = None
    aggregation_predicted_rebuild: int | None = None
    aggregation_keying_residual: int | None = None
    # v33: `built - reused` split by tree level (level0 + rollup + root).
    aggregation_rebuilt_level0: int | None = None
    aggregation_rebuilt_rollup: int | None = None
    aggregation_rebuilt_root: int | None = None
    # v34: size of the digest leaf-set shift the binary flag stands in for.
    aggregation_leaf_added: int | None = None
    aggregation_leaf_removed: int | None = None
    aggregation_facts_rekey: int | None = None
    aggregation_blocking: str = ""
    digest_failures: int = 0
    digest_quarantined: int = 0
    episodes_created: int = 0
    facts_extracted: int = 0
    fact_failures: int = 0
    facts_embedded: int = 0
    facts_embedded_from_cache: int = 0
    profile_items_extracted: int = 0
    profile_failures: int = 0
    skipped_locked: bool = False
    budget_exhausted: bool = False


_MESSAGE_EMBEDDING_ISOLATION_ATTEMPTS = 16


def _persist_message_batch_with_failure_isolation(
    conn: sqlite3.Connection,
    embedding_client: EmbeddingClient,
    message_ids: tuple[int, ...],
) -> tuple[int, int, bool]:
    """Persist every healthy member of a bounded maintenance batch.

    A content-specific provider rejection is isolated by testing both halves
    and recursively splitting only the failing half. If both halves fail with
    the same exception type, treat that as provider-wide unavailability and
    stop; a hard attempt cap covers providers with unstable error types. This
    gives one poison occurrence bounded blast radius without turning a global
    10-second timeout into one timeout per message. All provider work remains
    outside transactions; each successful sub-batch gets one short write.
    """
    attempts = 0
    persisted = 0
    cache_hits = 0
    abort_cycle = False

    def attempt(batch: tuple[int, ...]) -> tuple[bool, type[Exception] | None]:
        nonlocal attempts, persisted, cache_hits, abort_cycle
        if not batch or attempts >= _MESSAGE_EMBEDDING_ISOLATION_ATTEMPTS:
            abort_cycle = True
            return False, None
        attempts += 1
        try:
            pending = fetch_message_embeddings(
                conn, embedding_client, message_ids=batch
            )
            if pending is not None:
                with core_db.transaction(conn):
                    persisted += persist_message_embeddings(conn, pending)
                cache_hits += pending.cache_hits
        except Exception as exc:
            log.error(
                "embedding.message_fetch_failure batch_size=%d error=%s",
                len(batch), type(exc).__name__,
            )
            return False, type(exc)
        return True, None

    def isolate_failed(
        batch: tuple[int, ...], parent_error: type[Exception] | None
    ) -> None:
        nonlocal abort_cycle
        if len(batch) <= 1 or abort_cycle:
            return
        midpoint = len(batch) // 2
        left, right = batch[:midpoint], batch[midpoint:]
        left_ok, left_error = attempt(left)
        right_ok, right_error = attempt(right)
        if abort_cycle:
            return
        if (
            not left_ok and not right_ok
            and left_error is not None and left_error is right_error
            and (parent_error is None or left_error is parent_error)
        ):
            # Both independent halves failed alike: overwhelmingly likely a
            # provider outage, not a single bad input. Bound this dream cycle.
            abort_cycle = True
            return
        if not left_ok:
            isolate_failed(left, left_error)
        if not right_ok:
            isolate_failed(right, right_error)

    ok, error_type = attempt(message_ids)
    if not ok:
        isolate_failed(message_ids, error_type)
    return persisted, cache_hits, abort_cycle


def _prepare_dedup_vectors(
    conn: sqlite3.Connection,
    extraction: phase1.ChunkExtraction,
    cfg: HyMemConfig,
    embedding_client: EmbeddingClient | None,
) -> dict[str, list[float]]:
    """Best-effort wrapper around :func:`phase1.prepare_dedup_vectors`.

    Runs the dedup candidate embed *before* the persist transaction is opened,
    so the network ``embed()`` call never happens under the SQLite write lock.
    A failure (flaky embedding endpoint, etc.) is logged and degrades to ``{}``
    — dedup simply doesn't fire for this chunk — mirroring the try/except
    tolerance the persist path already has around dedup.
    """
    try:
        return phase1.prepare_dedup_vectors(conn, extraction, cfg, embedding_client)
    except Exception as exc:
        log.error(
            "phase1.dedup_prepare_failure chunk_triples=%d error=%s",
            len(extraction.triples), type(exc).__name__,
        )
        return {}


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
        "INSERT INTO dream_runs(started_at, aggregation_effective) "
        "VALUES (CURRENT_TIMESTAMP, ?)",
        ("enabled" if cfg.aggregation_nodes_enabled else "disabled",),
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

        # Same-wave dedup pool, shared across ALL chunks/sessions of this dream
        # so a sibling triple in a later chunk collapses onto an edge minted by
        # an earlier chunk in the same cycle. In-memory only; no DB/network I/O.
        in_cycle_edges = phase1.new_in_cycle_pool()

        # Chunks this cycle has already sent to the LLM. The baseline backstop
        # selects on "no processed_chunks row", which used to exclude a failed
        # chunk automatically because failures were marked done. Held-for-retry
        # leaves them unmarked, so without this set the SAME chunk is extracted
        # twice in one dream — once per tier — doubling its LLM cost and
        # burning two budget slots and two retry attempts per cycle.
        attempted_this_cycle: set[str] = set()

        # Lease heartbeat (throttled). Refreshes the lock at most once per
        # _LOCK_REFRESH_INTERVAL_SECONDS of wall time. Called per session AND
        # per chunk so a single very heavy session can't let acquired_at age
        # past the TTL while the dream is still alive — a crashed holder simply
        # stops calling this and is reclaimed after the TTL. _last_heartbeat
        # starts at 0.0 so the first call always fires. _refresh_lock runs
        # OUTSIDE any core_db.transaction (autocommit) so the new timestamp is
        # immediately visible to other connections.
        _last_heartbeat = [0.0]

        def _heartbeat() -> None:
            now = time.monotonic()
            if now - _last_heartbeat[0] >= _LOCK_REFRESH_INTERVAL_SECONDS:
                _refresh_lock(conn, holder)
                _last_heartbeat[0] = now

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

        if cfg.redact_secrets:
            # Privacy policy tightening is local, global profile maintenance;
            # do it once per dream even when new extraction is disabled.
            with core_db.transaction(conn):
                enforce_profile_redaction_policy(conn)

        for session_id in target_sessions:
            _heartbeat()
            report.sessions_processed += 1
            # First establish a durable, exact source stream for every role and
            # every message length.  This local write is independent of
            # salience and of the LLM budget.  Digest/provenance work below is
            # never allowed to outrun it.
            try:
                while True:
                    with core_db.transaction(conn):
                        covered_now = materialize_message_coverage(
                            conn, session_id, limit=256
                        )
                    _heartbeat()
                    if covered_now < 256:
                        break
            except Exception:
                log.exception(
                    "coverage.materialization_failure session_id=%s", session_id
                )
                # Retrying next dream is lossless; proceeding with a partial
                # stream could advance derived cursors past an uncovered turn.
                continue
            # True once any chunk in this session is freshly extracted this run.
            # Drives the digest skip-guard: an unchanged, already-digested session
            # makes zero tail LLM calls.
            had_new_chunk_work = False
            chunks = extract_high_salience_chunks(
                conn, session_id, min_chars=cfg.salience_min_chars
            )
            report.chunks_seen += len(chunks)

            if chunks:
                with core_db.transaction(conn):
                    persist_chunks(conn, chunks)
                    for chunk in chunks:
                        index_chunk_mentions(conn, chunk.id, chunk.text)
                        index_chunk_temporal_mentions(conn, chunk.id)
                _kickoff_chunk_embed(chunks)

            for chunk in chunks:
                if chunks_remaining <= 0:
                    break
                _heartbeat()
                # Skip chunks already processed with the current prompt version
                # without consuming the budget, so unprocessed chunks at the tail
                # of the session list don't get starved.
                already = conn.execute(
                    "SELECT 1 FROM processed_chunks WHERE chunk_id = ? AND prompt_version = ?",
                    (chunk.id, cfg.prompt_version),
                ).fetchone()
                if already:
                    continue
                if chunk_extraction_is_quarantined(
                    conn,
                    chunk.id,
                    prompt_version=cfg.prompt_version,
                    max_attempts=cfg.chunk_extraction_max_attempts,
                ):
                    continue
                chunks_remaining -= 1
                attempted_this_cycle.add(chunk.id)
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
                if extraction.failed:
                    # Held, not marked: retried on the next dream. Audible so a
                    # chunk that fails forever is greppable rather than silent
                    # (the stuck-fusion lesson, applied to ingest).
                    #
                    # This must NOT skip persist. The attempt bookkeeping that
                    # bounds the retry lives there, so short-circuiting here
                    # left the v28 bound unreachable in production while the
                    # tests — which call persist directly — still passed.
                    report.chunk_extraction_failures += 1
                    log.warning(
                        "phase1.extraction_failed chunk_id=%s tier=salience "
                        "action=held_for_retry", chunk.id,
                    )
                else:
                    had_new_chunk_work = True
                # Embed dedup candidates OUTSIDE the write lock; best-effort so
                # a flaky embedder degrades to "no dedup", never aborts the dream.
                dedup_vectors = (
                    {}
                    if extraction.failed
                    else _prepare_dedup_vectors(
                        conn, extraction, cfg, embedding_client
                    )
                )
                staged_in_cycle_edges = None
                with core_db.transaction(conn):
                    staged_in_cycle_edges = phase1.persist_chunk_results(
                        conn, chunk, extraction, prompt_version=cfg.prompt_version,
                        cfg=cfg, embedding_client=embedding_client,
                        dedup_vectors=dedup_vectors,
                        in_cycle_edges=in_cycle_edges,
                    )
                if staged_in_cycle_edges is not None:
                    in_cycle_edges[:] = staged_in_cycle_edges
                if not extraction.failed and (
                    extraction.triples or extraction.markers
                ):
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
                    max_attempts=cfg.chunk_extraction_max_attempts,
                    exclude_ids=attempted_this_cycle,
                )
                if baseline:
                    report.chunks_seen += len(baseline)
                    with core_db.transaction(conn):
                        persist_chunks(conn, baseline)
                        for chunk in baseline:
                            index_chunk_mentions(conn, chunk.id, chunk.text)
                            index_chunk_temporal_mentions(conn, chunk.id)
                    _kickoff_chunk_embed(baseline)
                    for chunk in baseline:
                        if chunks_remaining <= 0:
                            break
                        if chunk.id in attempted_this_cycle:
                            # Already sent to the LLM by the salience tier this
                            # cycle and held for retry — not a second attempt.
                            continue
                        _heartbeat()
                        chunks_remaining -= 1
                        attempted_this_cycle.add(chunk.id)
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
                        if extraction.failed:
                            # See the salience-tier call site above — persist
                            # still runs, or the retry bound never accrues.
                            report.chunk_extraction_failures += 1
                            log.warning(
                                "phase1.extraction_failed chunk_id=%s "
                                "tier=baseline action=held_for_retry", chunk.id,
                            )
                        else:
                            had_new_chunk_work = True
                        # Embed dedup candidates OUTSIDE the write lock (see
                        # the salience-tier call site above).
                        dedup_vectors = (
                            {}
                            if extraction.failed
                            else _prepare_dedup_vectors(
                                conn, extraction, cfg, embedding_client
                            )
                        )
                        staged_in_cycle_edges = None
                        with core_db.transaction(conn):
                            staged_in_cycle_edges = phase1.persist_chunk_results(
                                conn, chunk, extraction,
                                prompt_version=cfg.prompt_version,
                                cfg=cfg, embedding_client=embedding_client,
                                dedup_vectors=dedup_vectors,
                                in_cycle_edges=in_cycle_edges,
                            )
                        if staged_in_cycle_edges is not None:
                            in_cycle_edges[:] = staged_in_cycle_edges
                        if not extraction.failed and (
                            extraction.triples or extraction.markers
                        ):
                            report.chunks_processed += 1
                            report.triples_extracted += len(extraction.triples)
                            report.markers_extracted += len(extraction.markers)

            # Prompt-salt replay cannot depend on live raw messages: retention
            # may have pruned them after storing the extraction chunks. Drain a
            # bounded durable backlog from the chunks themselves, excluding
            # anything already attempted this cycle or explicitly quarantined.
            if chunks_remaining > 0:
                backlog = load_pending_persisted_chunks(
                    conn,
                    session_id,
                    prompt_version=cfg.prompt_version,
                    limit=chunks_remaining,
                    max_attempts=cfg.chunk_extraction_max_attempts,
                    exclude_ids=attempted_this_cycle,
                )
                report.chunks_seen += len(backlog)
                for chunk in backlog:
                    if chunks_remaining <= 0:
                        break
                    _heartbeat()
                    chunks_remaining -= 1
                    attempted_this_cycle.add(chunk.id)
                    try:
                        extraction = phase1.extract_chunk_results(
                            conn,
                            chunk,
                            llm,
                            prompt_version=cfg.prompt_version,
                            negative_examples=negative_examples,
                        )
                    except Exception:
                        log.exception(
                            "phase1.llm_failure chunk_id=%s tier=persisted_backlog",
                            chunk.id,
                        )
                        continue
                    if extraction is None:
                        continue
                    if extraction.failed:
                        report.chunk_extraction_failures += 1
                        log.warning(
                            "phase1.extraction_failed chunk_id=%s "
                            "tier=persisted_backlog action=held_for_retry",
                            chunk.id,
                        )
                    else:
                        had_new_chunk_work = True
                    dedup_vectors = (
                        {}
                        if extraction.failed
                        else _prepare_dedup_vectors(
                            conn, extraction, cfg, embedding_client
                        )
                    )
                    staged_in_cycle_edges = None
                    with core_db.transaction(conn):
                        staged_in_cycle_edges = phase1.persist_chunk_results(
                            conn,
                            chunk,
                            extraction,
                            prompt_version=cfg.prompt_version,
                            cfg=cfg,
                            embedding_client=embedding_client,
                            dedup_vectors=dedup_vectors,
                            in_cycle_edges=in_cycle_edges,
                        )
                    if staged_in_cycle_edges is not None:
                        in_cycle_edges[:] = staged_in_cycle_edges
                    if not extraction.failed and (
                        extraction.triples or extraction.markers
                    ):
                        report.chunks_processed += 1
                        report.triples_extracted += len(extraction.triples)
                        report.markers_extracted += len(extraction.markers)

            # Per-session digest reads the independent lossless message stream,
            # never the selective/overlapping salience chunks.  Its prompt-
            # generation cursor is resumable inside an oversized message and
            # remains usable after raw-message retention because the backing
            # artifacts are protected by the v37 ledger.
            digested = conn.execute(
                "SELECT summary, summary_source, auto_summary, "
                "digested_prompt_version, profile_prompt_version, "
                "profile_cursor_message_id, "
                "profile_cursor_partial_message_id, profile_cursor_offset, "
                "profile_cursor_prompt_version, profile_published_generation, "
                "profile_retry_count, profile_retry_config_version, "
                "profile_quarantined, "
                "digested_message_id, episodes_prompt_version, "
                "coverage_message_id, digest_cursor_message_id, "
                "digest_cursor_partial_message_id, digest_cursor_offset, "
                "digest_cursor_prompt_version, digest_published_generation "
                ", digest_retry_count, digest_retry_config_version, "
                "digest_quarantined "
                "FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            # Plan C (schema v35): the episode prompt has its OWN per-session
            # stamp, for the same reason the profile call needed one at v19 —
            # the guard below keys on cfg.prompt_version, which an episode
            # granularity flip does not move, so without this leg an
            # already-digested session would keep its old-granularity episodes
            # forever and only never-digested sessions would get the new ones.
            # `active_episode_prompt_version` returns None when the flag is off,
            # which equals the NULL every pre-v35 row carries: a store that
            # never enables granularity can never see a mismatch here, so the
            # zero-tail-call steady state is untouched.
            episode_prompt_version = active_episode_prompt_version(
                cfg.episode_granularity_enabled
            )
            # Includes the framing generation and cap: changing either
            # repartitions input slices and therefore starts a safe full walk.
            # The persisted cursor value may append a unique ``|walk=`` token.
            # That token distinguishes two complete rebuilds under the SAME
            # configuration, so an authoritative shorter result can retire
            # stale rows only after its replacement walk reaches the tail.
            digest_config = digest_config_version(
                prompt_version=cfg.prompt_version,
                episode_prompt_version=episode_prompt_version,
                max_chars=cfg.dream_digest_max_chars,
                max_tokens=cfg.dream_digest_max_tokens,
                max_episodes=(
                    cfg.dream_max_episodes_per_session
                    if cfg.episode_granularity_enabled else None
                ),
            )
            stored_digest_generation = (
                digested["digest_cursor_prompt_version"] if digested else None
            )
            published_digest_generation = (
                digested["digest_published_generation"] if digested else None
            )
            cursor_current = (
                digested is not None
                and digest_generation_matches_config(
                    stored_digest_generation, digest_config
                )
            )
            published_current = (
                digested is not None
                and digested["digested_prompt_version"] == cfg.prompt_version
                and digested["episodes_prompt_version"] == episode_prompt_version
                and digest_generation_matches_config(
                    published_digest_generation, digest_config
                )
                and published_digest_generation == stored_digest_generation
            )
            cursor_message_id = (
                digested["digest_cursor_message_id"] if cursor_current else None
            )
            partial_message_id = (
                digested["digest_cursor_partial_message_id"]
                if cursor_current else None
            )
            cursor_offset = int(digested["digest_cursor_offset"] or 0) if cursor_current else 0
            coverage_tail = digested["coverage_message_id"] if digested else None
            digest_cursor_invalid = False
            if (
                cursor_current
                and not _lossless_cursor_is_valid(
                    conn, session_id, cursor_message_id,
                    partial_message_id, cursor_offset,
                )
            ):
                log.warning(
                    "digest.cursor_invalid session_id=%s cursor=%s tail=%s "
                    "action=rewind",
                    session_id, cursor_message_id, coverage_tail,
                )
                cursor_current = False
                cursor_message_id = None
                partial_message_id = None
                cursor_offset = 0
                digest_cursor_invalid = True
            newest_message_id = conn.execute(
                "SELECT MAX(id) AS m FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()["m"]
            caught_up = (
                coverage_tail is None
                or (
                    cursor_current
                    and cursor_offset == 0
                    and partial_message_id is None
                    and cursor_message_id is not None
                    and int(cursor_message_id) == int(coverage_tail)
                )
            )
            # A completed cursor whose published stamps were explicitly
            # invalidated requests a full re-digest (the existing operator/test
            # contract).  An IN-PROGRESS cursor legitimately has old/null
            # published stamps and must keep walking rather than rewind.
            digest_requires_rebuild = bool(
                cursor_current
                and not published_current
                and (
                    caught_up
                    or stored_digest_generation == published_digest_generation
                )
            )
            if digest_requires_rebuild:
                cursor_current = False
                cursor_message_id = None
                partial_message_id = None
                cursor_offset = 0
                caught_up = False
            digest_retry_key = digest_retry_policy_version(
                digest_config,
                max_attempts=cfg.digest_extraction_max_attempts,
                rebuild_from=(
                    stored_digest_generation
                    if digest_requires_rebuild or digest_cursor_invalid else None
                ),
                invalidated_stamp=(
                    (
                        "invalid-cursor" if digest_cursor_invalid
                        else digested["digested_prompt_version"]
                    )
                    if (digest_requires_rebuild or digest_cursor_invalid) and digested
                    else None
                ),
            )
            digest_retry_count = (
                int(digested["digest_retry_count"] or 0)
                if digested
                and digested["digest_retry_config_version"] == digest_retry_key
                else 0
            )
            digest_quarantined = bool(
                digested
                and digested["digest_retry_config_version"] == digest_retry_key
                and cfg.digest_extraction_max_attempts > 0
                and digest_retry_count >= cfg.digest_extraction_max_attempts
            )
            if coverage_tail is not None and not caught_up and not digest_quarantined:
                # A new full walk gets a distinct generation even when its
                # prompt/config is unchanged.  Successful partial slices store
                # this token with their cursor and reuse it on later dreams;
                # failed calls store neither.  Tail appends reuse the current
                # completed walk so previously published rows stay live.
                digest_build_generation = (
                    stored_digest_generation
                    if cursor_current and stored_digest_generation is not None
                    else f"{digest_config}|walk={uuid.uuid4().hex}"
                )
                # A v37/legacy summary may be the only surviving representation
                # of messages pruned before exact artifacts existed.  Seed the
                # first v38 walk with it.  Prompt-version rewinds also carry a
                # prior automatic summary so a partial rebuild never hides the
                # already-published history.
                if digested["auto_summary"]:
                    # A rewind re-reads the exact source stream, but it must
                    # never replace a complete published history with the
                    # first bounded slice of the rebuild.  Carrying the prior
                    # automatic summary forward keeps history available while
                    # the new prompt generation converges.
                    prior_auto_summary = digested["auto_summary"]
                elif (
                    digested["summary_source"] == "legacy"
                ):
                    prior_auto_summary = digested["summary"] or ""
                else:
                    prior_auto_summary = ""
                slice_key = (
                    f"after={cursor_message_id if cursor_message_id is not None else 'start'};"
                    f"partial={partial_message_id if partial_message_id is not None else 'none'};"
                    f"offset={cursor_offset};cap="
                    f"{digest_attempt_max_chars(cfg.dream_digest_max_chars, digest_retry_count)}"
                )
                digest_attempt_chars = digest_attempt_max_chars(
                    cfg.dream_digest_max_chars, digest_retry_count
                )
                try:
                    digest = extract_session_digest(
                        conn, session_id, llm,
                        max_tokens=cfg.dream_digest_max_tokens,
                        max_chars=digest_attempt_chars,
                        since_message_id=cursor_message_id,
                        partial_message_id=partial_message_id,
                        since_message_offset=cursor_offset,
                        prior_summary=prior_auto_summary,
                        granular=cfg.episode_granularity_enabled,
                        max_episodes=cfg.dream_max_episodes_per_session,
                    )
                except Exception:
                    report.digest_failures += 1
                    log.exception("digest.extraction_failure session_id=%s", session_id)
                    with core_db.transaction(conn):
                        newly_quarantined = record_digest_failure(
                            conn,
                            session_id,
                            max_attempts=cfg.digest_extraction_max_attempts,
                            retry_config_version=digest_retry_key,
                        )
                    if newly_quarantined:
                        report.digest_quarantined += 1
                    else:
                        report.budget_exhausted = True
                else:
                    if digest is not None and digest.parse_failed:
                        report.digest_failures += 1
                        with core_db.transaction(conn):
                            newly_quarantined = record_digest_failure(
                                conn,
                                session_id,
                                max_attempts=cfg.digest_extraction_max_attempts,
                                retry_config_version=digest_retry_key,
                            )
                        if newly_quarantined:
                            report.digest_quarantined += 1
                        else:
                            report.budget_exhausted = True
                    if digest is not None and not digest.parse_failed:
                        with core_db.transaction(conn):
                            # At most one unpublished replacement generation is
                            # retained. A successfully-started new build may
                            # discard abandoned staging, but never the marker's
                            # last complete generation.
                            conn.execute(
                                "DELETE FROM episodes WHERE session_id = ? "
                                "AND digest_generation IS NOT NULL "
                                "AND digest_generation <> ? "
                                "AND (? IS NULL OR digest_generation <> ?)",
                                (
                                    session_id,
                                    digest_build_generation,
                                    published_digest_generation,
                                    published_digest_generation,
                                ),
                            )
                            if digest.episodes.items:
                                ep_count = persist_episodes(
                                    conn, session_id, digest.episodes,
                                    granular=cfg.episode_granularity_enabled,
                                    supersede_window=None,
                                    digest_slice_key=slice_key,
                                    digest_generation=digest_build_generation,
                                )
                                report.episodes_created += ep_count
                                log.debug(
                                    "episodes session_id=%s count=%d", session_id, ep_count
                                )
                            summary_to_persist = digest.summary or prior_auto_summary
                            # An explicitly empty summary is a successful
                            # no-op, not a parse failure.  Persist its position
                            # too (retaining any prior text), so the summary and
                            # digest cursors cannot disagree about whether this
                            # material was examined.
                            persist_auto_session_summary(
                                conn,
                                session_id,
                                summary_to_persist,
                                covered_message_id=digest.covered_message_id,
                                partial_message_id=digest.partial_message_id,
                                covered_message_offset=digest.next_message_offset,
                            )
                            if summary_to_persist:
                                log.debug("summary session_id=%s", session_id)
                            if digest.procedures.items:
                                pr_count = persist_procedures(
                                    conn, session_id, digest.procedures
                                )
                                log.debug(
                                    "procedures session_id=%s count=%d", session_id, pr_count
                                )
                            conn.execute(
                                """
                                UPDATE sessions
                                SET digest_cursor_message_id = ?,
                                    digest_cursor_partial_message_id = ?,
                                    digest_cursor_offset = ?,
                                    digest_cursor_prompt_version = ?,
                                    digest_retry_count = 0,
                                    digest_retry_config_version = NULL,
                                    digest_quarantined = 0
                                WHERE id = ?
                                """,
                                (
                                    digest.covered_message_id,
                                    digest.partial_message_id,
                                    digest.next_message_offset,
                                    digest_build_generation,
                                    session_id,
                                ),
                            )
                            if digest.covered_message_id is not None:
                                conn.execute(
                                    "UPDATE sessions SET digested_message_id = "
                                    "MAX(COALESCE(digested_message_id, -1), ?) "
                                    "WHERE id = ?",
                                    (digest.covered_message_id, session_id),
                                )
                            if digest.caught_up:
                                # Publish prompt stamps and retire the previous
                                # complete generation only after the replacement
                                # walk is wholly durable.  A mid-walk failure
                                # therefore cannot mutate or hide the last
                                # complete set. Generation-scoped ids make the
                                # new rows staging until this atomic marker swap.
                                replacing_generation = (
                                    published_digest_generation
                                    != digest_build_generation
                                )
                                conn.execute(
                                    "DELETE FROM episodes WHERE session_id = ? "
                                    "AND digest_generation IS NOT NULL "
                                    "AND digest_generation <> ?",
                                    (session_id, digest_build_generation),
                                )
                                conn.execute(
                                    "UPDATE sessions SET digested_prompt_version = ?, "
                                    "episodes_prompt_version = ?, "
                                    "digest_published_generation = ? WHERE id = ?",
                                    (
                                        cfg.prompt_version,
                                        episode_prompt_version,
                                        digest_build_generation,
                                        session_id,
                                    ),
                                )
                                if replacing_generation:
                                    # Conditional FTS triggers deliberately did
                                    # not index staged rows. The marker and these
                                    # postings become visible in this same
                                    # transaction, after old published postings
                                    # were removed by their DELETE triggers.
                                    conn.execute(
                                        "INSERT INTO episodes_fts(rowid, title, summary) "
                                        "SELECT rowid, title, summary FROM episodes "
                                        "WHERE session_id = ? "
                                        "AND digest_generation = ?",
                                        (session_id, digest_build_generation),
                                    )
                            else:
                                # Completion loops historically use this flag
                                # to decide whether another dream pass is owed.
                                # A bounded digest slice is unfinished work even
                                # when the Phase-1 chunk budget remains.
                                report.budget_exhausted = True
            elif digest_quarantined and coverage_tail is not None and not caught_up:
                report.digest_quarantined += 1
                log.warning(
                    "digest.skipped_quarantined session_id=%s attempts=%d",
                    session_id,
                    digest_retry_count,
                )

            # Typed profile extraction has its own USER-only durable cursor.
            # Chunk salience is intentionally irrelevant: a short user turn
            # reopens this path, while assistant/system/tool-only traffic makes
            # zero profile calls. Successful bounded slices are redacted and
            # staged with the cursor; consumer-visible rows change only in the
            # transaction that reaches the complete USER tail.
            if cfg.profile_extraction_enabled:
                profile_config = profile_config_version(
                    max_chars=cfg.dream_digest_max_chars,
                    max_items=cfg.profile_max_items_per_session,
                    redact_values=cfg.redact_secrets,
                )
                stored_profile_generation = (
                    digested["profile_cursor_prompt_version"] if digested else None
                )
                published_profile_generation = (
                    digested["profile_published_generation"] if digested else None
                )
                profile_cursor_current = (
                    isinstance(stored_profile_generation, str)
                    and profile_generation_matches_config(
                        stored_profile_generation, profile_config
                    )
                )
                profile_published_current = (
                    digested is not None
                    and digested["profile_prompt_version"] == PROFILE_PROMPT_VERSION
                    and profile_generation_matches_config(
                        published_profile_generation, profile_config
                    )
                    and published_profile_generation == stored_profile_generation
                )
                profile_cursor_message_id = (
                    digested["profile_cursor_message_id"]
                    if profile_cursor_current else None
                )
                profile_partial_message_id = (
                    digested["profile_cursor_partial_message_id"]
                    if profile_cursor_current else None
                )
                profile_cursor_offset = (
                    int(digested["profile_cursor_offset"] or 0)
                    if profile_cursor_current else 0
                )
                profile_tail = profile_user_tail_message_id(conn, session_id)
                profile_cursor_invalid = False
                if (
                    profile_cursor_current
                    and not _lossless_cursor_is_valid(
                        conn, session_id, profile_cursor_message_id,
                        profile_partial_message_id, profile_cursor_offset,
                        roles=frozenset({"user"}),
                    )
                ):
                    log.warning(
                        "profile.cursor_invalid session_id=%s cursor=%s tail=%s "
                        "action=rewind",
                        session_id, profile_cursor_message_id, profile_tail,
                    )
                    profile_cursor_current = False
                    profile_cursor_message_id = None
                    profile_partial_message_id = None
                    profile_cursor_offset = 0
                    profile_cursor_invalid = True
                profile_caught_up = (
                    profile_tail is None
                    or (
                        profile_cursor_current
                        and profile_cursor_offset == 0
                        and profile_partial_message_id is None
                        and profile_cursor_message_id is not None
                        and int(profile_cursor_message_id) == int(profile_tail)
                    )
                )
                # An explicitly invalidated publication stamp requests a full
                # walk even if a prior cursor happened to reach the tail.
                profile_requires_rebuild = bool(
                    profile_cursor_current
                    and not profile_published_current
                    and (
                        profile_caught_up
                        or stored_profile_generation
                        == published_profile_generation
                    )
                )
                if profile_requires_rebuild:
                    profile_cursor_current = False
                    profile_cursor_message_id = None
                    profile_partial_message_id = None
                    profile_cursor_offset = 0
                    profile_caught_up = False

                profile_retry_key = profile_retry_policy_version(
                    profile_config,
                    max_attempts=cfg.profile_extraction_max_attempts,
                    rebuild_from=(
                        stored_profile_generation
                        if profile_requires_rebuild or profile_cursor_invalid
                        else None
                    ),
                    invalidated_stamp=(
                        (
                            "invalid-cursor" if profile_cursor_invalid
                            else digested["profile_prompt_version"]
                        )
                        if (profile_requires_rebuild or profile_cursor_invalid) and digested
                        else None
                    ),
                )
                profile_retry_count = (
                    int(digested["profile_retry_count"] or 0)
                    if digested
                    and digested["profile_retry_config_version"]
                    == profile_retry_key
                    else 0
                )
                profile_quarantined = bool(
                    digested
                    and digested["profile_retry_config_version"]
                    == profile_retry_key
                    and cfg.profile_extraction_max_attempts > 0
                    and profile_retry_count
                    >= cfg.profile_extraction_max_attempts
                )

                if profile_tail is None:
                    # No USER input is a distinct zero-call success. Publishing
                    # an empty generation prevents prompt/config ambiguity; a
                    # later user artifact is still detected by its own tail.
                    if not profile_published_current:
                        empty_generation = (
                            f"{profile_config}|walk={uuid.uuid4().hex}"
                        )
                        try:
                            with core_db.transaction(conn):
                                conn.execute(
                                    "DELETE FROM profile_staging WHERE session_id = ?",
                                    (session_id,),
                                )
                                conn.execute(
                                    """
                                    UPDATE sessions
                                    SET profile_prompt_version = ?,
                                        profile_cursor_message_id = NULL,
                                        profile_cursor_partial_message_id = NULL,
                                        profile_cursor_offset = 0,
                                        profile_cursor_prompt_version = ?,
                                        profile_published_generation = ?,
                                        profile_retry_count = 0,
                                        profile_retry_config_version = NULL,
                                        profile_quarantined = 0
                                    WHERE id = ?
                                    """,
                                    (
                                        PROFILE_PROMPT_VERSION,
                                        empty_generation,
                                        empty_generation,
                                        session_id,
                                    ),
                                )
                        except Exception:
                            report.profile_failures += 1
                            log.exception(
                                "profile.empty_publication_failure session_id=%s",
                                session_id,
                            )
                elif not profile_caught_up and not profile_quarantined:
                    profile_build_generation = (
                        stored_profile_generation
                        if profile_cursor_current
                        and stored_profile_generation is not None
                        else f"{profile_config}|walk={uuid.uuid4().hex}"
                    )
                    attempt_max_chars = profile_attempt_max_chars(
                        cfg.dream_digest_max_chars,
                        profile_retry_count,
                    )
                    try:
                        profile = extract_user_profile(
                            conn,
                            session_id,
                            llm,
                            max_chars=attempt_max_chars,
                            max_items=cfg.profile_max_items_per_session,
                            since_message_id=profile_cursor_message_id,
                            partial_message_id=profile_partial_message_id,
                            since_message_offset=profile_cursor_offset,
                        )
                    except Exception:
                        report.profile_failures += 1
                        log.exception(
                            "profile.extraction_failure session_id=%s", session_id
                        )
                        with core_db.transaction(conn):
                            quarantined = record_profile_failure(
                                conn,
                                session_id,
                                max_attempts=cfg.profile_extraction_max_attempts,
                                retry_config_version=profile_retry_key,
                            )
                        if not quarantined:
                            report.budget_exhausted = True
                    else:
                        if profile is None:
                            # profile_tail proved work existed, so None here is
                            # an invariant failure rather than a valid empty.
                            report.profile_failures += 1
                            log.warning(
                                "profile.missing_slice session_id=%s", session_id
                            )
                            with core_db.transaction(conn):
                                quarantined = record_profile_failure(
                                    conn,
                                    session_id,
                                    max_attempts=cfg.profile_extraction_max_attempts,
                                    retry_config_version=profile_retry_key,
                                )
                            if not quarantined:
                                report.budget_exhausted = True
                        elif profile.failed:
                            report.profile_failures += 1
                            log.warning(
                                "profile.extraction_failed session_id=%s reason=%s "
                                "returned=%d rejected=%d action=held_for_retry",
                                session_id,
                                profile.failure_reason,
                                profile.input_items,
                                profile.rejected_items,
                            )
                            with core_db.transaction(conn):
                                quarantined = record_profile_failure(
                                    conn,
                                    session_id,
                                    max_attempts=cfg.profile_extraction_max_attempts,
                                    retry_config_version=profile_retry_key,
                                )
                            if not quarantined:
                                report.budget_exhausted = True
                        else:
                            try:
                                with core_db.transaction(conn):
                                    # A successfully-started replacement may
                                    # retire abandoned staging, never published
                                    # profile rows.
                                    conn.execute(
                                        "DELETE FROM profile_staging "
                                        "WHERE session_id = ? AND generation <> ?",
                                        (session_id, profile_build_generation),
                                    )
                                    stage_profile_extraction(
                                        conn,
                                        session_id,
                                        profile_build_generation,
                                        profile,
                                        redact_values=cfg.redact_secrets,
                                    )
                                    conn.execute(
                                        """
                                        UPDATE sessions
                                        SET profile_cursor_message_id = ?,
                                            profile_cursor_partial_message_id = ?,
                                            profile_cursor_offset = ?,
                                            profile_cursor_prompt_version = ?,
                                            profile_retry_count = 0,
                                            profile_retry_config_version = NULL,
                                            profile_quarantined = 0
                                        WHERE id = ?
                                        """,
                                        (
                                            profile.covered_message_id,
                                            profile.partial_message_id,
                                            profile.next_message_offset,
                                            profile_build_generation,
                                            session_id,
                                        ),
                                    )
                                    persisted = 0
                                    if profile.caught_up:
                                        persisted = publish_profile_generation(
                                            conn,
                                            session_id,
                                            profile_build_generation,
                                        )
                                        conn.execute(
                                            "UPDATE sessions "
                                            "SET profile_prompt_version = ?, "
                                            "profile_published_generation = ? "
                                            "WHERE id = ?",
                                            (
                                                PROFILE_PROMPT_VERSION,
                                                profile_build_generation,
                                                session_id,
                                            ),
                                        )
                            except Exception:
                                report.profile_failures += 1
                                log.exception(
                                    "profile.persistence_failure session_id=%s",
                                    session_id,
                                )
                                with core_db.transaction(conn):
                                    quarantined = record_profile_failure(
                                        conn,
                                        session_id,
                                        max_attempts=cfg.profile_extraction_max_attempts,
                                        retry_config_version=profile_retry_key,
                                    )
                                if not quarantined:
                                    report.budget_exhausted = True
                            else:
                                report.profile_items_extracted += persisted
                                if persisted:
                                    log.debug(
                                        "profile session_id=%s rows=%d",
                                        session_id,
                                        persisted,
                                    )
                                if not profile.caught_up:
                                    report.budget_exhausted = True
                elif profile_quarantined:
                    log.warning(
                        "profile.skipped_quarantined session_id=%s attempts=%d",
                        session_id,
                        profile_retry_count,
                    )

            # v46 authoritative facts: one bounded call over the validated
            # lossless stream. The cursor can stop inside an oversized turn;
            # malformed/over-cap output records a durable held retry and never
            # advances. Each successful slice (including empty) publishes its
            # complete replacement set atomically with the cursor.
            if cfg.facts_extraction_enabled:
                try:
                    facts_state = conn.execute(
                        "SELECT facts_cursor_message_id,"
                        "facts_cursor_partial_message_id,facts_cursor_offset,"
                        "facts_cursor_prompt_version,facts_retry_count,"
                        "facts_retry_config_version,facts_quarantined "
                        "FROM sessions WHERE id = ?",
                        (session_id,),
                    ).fetchone()
                except sqlite3.OperationalError:
                    log.debug("facts.skipped_pre_v46 session_id=%s", session_id)
                else:
                    facts_cursor = facts_state["facts_cursor_message_id"]
                    facts_partial = facts_state["facts_cursor_partial_message_id"]
                    facts_offset = int(facts_state["facts_cursor_offset"] or 0)
                    current_facts_config = facts_config_version(cfg)
                    retry_state_valid = facts_retry_state_is_valid(
                        facts_state["facts_retry_count"],
                        facts_state["facts_retry_config_version"],
                        facts_state["facts_quarantined"],
                    )
                    cursor_valid = _lossless_cursor_is_valid(
                        conn, session_id, facts_cursor, facts_partial,
                        facts_offset, roles=frozenset({"user", "assistant"}),
                    )
                    if not cursor_valid:
                        report.fact_failures += 1
                        log.error(
                            "facts.invalid_cursor session_id=%s action=held",
                            session_id,
                        )
                        facts_caught_up = True
                    else:
                        facts_tail = facts_tail_message_id(conn, session_id)
                        facts_caught_up = bool(
                            facts_partial is None
                            and (
                                facts_tail is None
                                or facts_cursor == facts_tail
                            )
                        )
                    stale_slice = None
                    if cursor_valid:
                        try:
                            stale_slice = next_fact_outcome_for_replay(
                                conn, session_id, current_facts_config
                            )
                        except Exception:
                            report.fact_failures += 1
                            cursor_valid = False
                            log.exception(
                                "facts.invalid_outcome_chain session_id=%s action=held",
                                session_id,
                            )
                    stale_generation = None
                    if stale_slice is not None:
                        stale_row = conn.execute(
                            "SELECT generation FROM fact_extraction_outcomes "
                            "WHERE slice_key=?", (stale_slice,),
                        ).fetchone()
                        if stale_row is not None:
                            stale_generation = int(stale_row["generation"])
                    retry_unit_key = stale_slice or fact_cursor_retry_unit_key(
                        session_id, facts_cursor, facts_partial, facts_offset
                    )
                    retry_key = facts_retry_policy_version(
                        cfg, replay_slice_key=retry_unit_key
                    )
                    active_quarantine = bool(
                        retry_state_valid
                        and facts_state["facts_retry_config_version"] == retry_key
                        and int(facts_state["facts_quarantined"] or 0) == 1
                    )

                    def _hold_fact_failure() -> bool | None:
                        with core_db.transaction(conn):
                            quarantined = record_fact_failure_if_pending(
                                conn, session_id,
                                max_attempts=cfg.facts_extraction_max_attempts,
                                retry_config_version=retry_key,
                                expected_cursor_message_id=facts_cursor,
                                expected_partial_message_id=facts_partial,
                                expected_offset=facts_offset,
                                replay_slice_key=stale_slice,
                                expected_replay_generation=stale_generation,
                                target_publication_version=current_facts_config,
                            )
                        if quarantined is False:
                            report.budget_exhausted = True
                        return quarantined

                    if not cursor_valid:
                        if not active_quarantine:
                            _hold_fact_failure()
                        else:
                            log.warning(
                                "facts.invalid_state_quarantined session_id=%s "
                                "attempts=%d",
                                session_id,
                                int(facts_state["facts_retry_count"] or 0),
                            )

                    if cursor_valid and stale_slice is not None and not active_quarantine:
                        try:
                            facts = reextract_fact_outcome(
                                conn, stale_slice, llm, cfg,
                                _require_committed_chain=False,
                            )
                        except Exception:
                            report.fact_failures += 1
                            log.exception(
                                "facts.replay_failure session_id=%s slice_key=%s",
                                session_id, stale_slice,
                            )
                            _hold_fact_failure()
                        else:
                            if facts.parse_failed:
                                report.fact_failures += 1
                                _hold_fact_failure()
                            else:
                                persisted = 0
                                try:
                                    with core_db.transaction(conn):
                                        persisted = persist_facts(
                                            conn, session_id, facts,
                                            max_items=cfg.dream_max_facts_per_session,
                                            _defer_chain_audit=True,
                                        )
                                        stale_before = conn.execute(
                                            "SELECT 1 FROM fact_extraction_outcomes "
                                            "WHERE session_id=? AND "
                                            "source_manifest_complete=1 AND "
                                            "prompt_version<? LIMIT 1",
                                            (session_id, current_facts_config),
                                        ).fetchone()
                                        stale_after = conn.execute(
                                            "SELECT 1 FROM fact_extraction_outcomes "
                                            "WHERE session_id=? AND "
                                            "source_manifest_complete=1 AND "
                                            "prompt_version>? LIMIT 1",
                                            (session_id, current_facts_config),
                                        ).fetchone()
                                        remaining = int(
                                            stale_before is not None
                                            or stale_after is not None
                                        )
                                        if (
                                            int(remaining) == 0
                                            and not fact_session_authority_is_valid(
                                                conn, session_id
                                            )
                                        ):
                                            raise RuntimeError(
                                                "fact replay final authority audit failed"
                                            )
                                        conn.execute(
                                            "UPDATE sessions SET "
                                            "facts_cursor_prompt_version="
                                            "CASE WHEN ?=0 THEN ? ELSE "
                                            "facts_cursor_prompt_version END,"
                                            "facts_retry_count=0,"
                                            "facts_retry_config_version=NULL,"
                                            "facts_quarantined=0 WHERE id=?",
                                            (
                                                int(remaining), current_facts_config,
                                                session_id,
                                            ),
                                        )
                                except Exception:
                                    report.fact_failures += 1
                                    log.exception(
                                        "facts.replay_persistence_failure "
                                        "session_id=%s slice_key=%s",
                                        session_id, stale_slice,
                                    )
                                    _hold_fact_failure()
                                else:
                                    report.facts_extracted += persisted
                                    if (
                                        remaining
                                        or not facts_caught_up
                                    ):
                                        report.budget_exhausted = True
                    elif cursor_valid and stale_slice is None and not facts_caught_up and not active_quarantine:
                        prior_attempts = (
                            int(facts_state["facts_retry_count"] or 0)
                            if facts_state["facts_retry_config_version"] == retry_key
                            else 0
                        )
                        attempt_chars = facts_attempt_max_chars(
                            cfg.dream_digest_max_chars, prior_attempts
                        )
                        try:
                            facts = extract_facts(
                                conn, session_id, llm, cfg,
                                since_message_id=facts_cursor,
                                partial_message_id=facts_partial,
                                start_offset=facts_offset,
                                max_chars=attempt_chars,
                            )
                        except Exception:
                            report.fact_failures += 1
                            log.exception(
                                "facts.extraction_failure session_id=%s", session_id
                            )
                            _hold_fact_failure()
                        else:
                            if facts is None or facts.parse_failed:
                                report.fact_failures += 1
                                _hold_fact_failure()
                            else:
                                persisted = 0
                                try:
                                    with core_db.transaction(conn):
                                        persisted = persist_facts(
                                            conn, session_id, facts,
                                            max_items=cfg.dream_max_facts_per_session,
                                        )
                                except Exception:
                                    report.fact_failures += 1
                                    log.exception(
                                        "facts.persistence_failure session_id=%s",
                                        session_id,
                                    )
                                    _hold_fact_failure()
                                else:
                                    report.facts_extracted += persisted
                                    if not facts.caught_up:
                                        report.budget_exhausted = True
                                if persisted:
                                    log.debug(
                                        "facts session_id=%s rows=%d",
                                        session_id, persisted,
                                    )
                    elif cursor_valid and stale_slice is None and facts_caught_up:
                        if (
                            facts_state["facts_cursor_prompt_version"]
                            != current_facts_config
                        ):
                            with core_db.transaction(conn):
                                if not fact_session_authority_is_valid(
                                    conn, session_id
                                ):
                                    raise RuntimeError(
                                        "fact cursor authority audit failed"
                                    )
                                conn.execute(
                                    "UPDATE sessions SET "
                                    "facts_cursor_prompt_version=?,"
                                    "facts_retry_count=0,"
                                    "facts_retry_config_version=NULL,"
                                    "facts_quarantined=0 WHERE id=?",
                                    (current_facts_config, session_id),
                                )
                    elif cursor_valid and active_quarantine:
                        log.warning(
                            "facts.skipped_quarantined session_id=%s attempts=%d",
                            session_id, int(facts_state["facts_retry_count"] or 0),
                        )

            if chunks_remaining <= 0:
                report.budget_exhausted = True
                log.info("dream.budget_exhausted budget=%d", cfg.dream_budget)
                break

        if embedding_client is not None:
            # Drain background-embedded batches before opening any write
            # transaction. A
            # per-batch future failure is logged and skipped — the post-loop
            # fetch_chunk_embeddings call below catches anything missed
            # (skipped sessions, future raises, miss_texts size mismatch).
            persisted_ids: set[str] = set()
            if embed_inflight:
                for request, future in embed_inflight:
                    try:
                        miss_vectors = future.result()
                    except Exception as exc:
                        log.error(
                            "embedding.background_failure batch_size=%d error=%s",
                            len(request.ids), type(exc).__name__,
                        )
                        continue
                    try:
                        pending = assemble_chunk_pending(
                            conn, request, miss_vectors,
                            exclude_ids=persisted_ids,
                            resolved_model=embedding_client.model,
                            resolved_dim=embedding_client.dim,
                        )
                    except Exception as exc:
                        log.error(
                            "embedding.assemble_failure batch_size=%d error=%s",
                            len(request.ids), type(exc).__name__,
                        )
                        continue
                    if pending is None:
                        continue
                    with core_db.transaction(conn):
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

            # Exact message occurrences have their own durable semantic tier.
            # Fetch/embedding is deliberately outside the write transaction;
            # only the idempotent mirror/index update takes the writer lock.
            for message_batch in message_embedding_id_batches(conn):
                embedded, from_cache, abort_cycle = (
                    _persist_message_batch_with_failure_isolation(
                        conn, embedding_client, message_batch
                    )
                )
                report.messages_embedded += embedded
                report.messages_embedded_from_cache += from_cache
                if abort_cycle:
                    break

        log.info("phase2.start")
        with core_db.transaction(conn):
            # Idea B write-side: route imperative markers into agent_inferred
            # rules BEFORE consolidate_profile stamps them consolidated (both
            # read consolidated_at IS NULL). Gated; no new LLM call. Additive —
            # markers still become profile entries too.
            if cfg.rules_extraction_enabled:
                from hymem import rules as rules_mod
                report.rules_extracted += rules_mod.route_markers_to_rules(conn, cfg, llm=llm)
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
            # Open the bi-temporal validity interval on every edge minted this
            # cycle (direct + derived) from its source-message world date
            # (schema v15). After decay so retracted edges already carry an
            # invalid_at; write-once so this is idempotent across cycles.
            stamped = bitemporal.stamp_validity(conn)
            if stamped:
                log.info("bitemporal.valid_at_stamped count=%d", stamped)
            # Single-assertion supersession: once valid_at is stamped, close the
            # interval on older typed-value edges that a newer value replaced
            # (opt-in; needs valid_at, runs before retracted-edge pruning).
            if cfg.value_supersession_enabled:
                superseded = supersede_competing_values(conn, cfg)
                if superseded:
                    log.info("bitemporal.value_superseded count=%d", superseded)
            pruned = prune_chunks(conn, cfg)
            pruned += prune_messages(conn, cfg)
            pruned += prune_retracted_edges(conn, cfg)
            pruned += prune_episodes_and_procedures(conn, cfg)
            pruned += prune_bookkeeping(conn, cfg)
            phase2.consolidate_insights(conn, cfg)  # refresh after decay
            conn.execute("DELETE FROM token_overlap_index")
            _current_edge = live_edge_predicate()
            _canon_rows = conn.execute(
                f"SELECT DISTINCT subject_canonical AS c FROM knowledge_graph "
                f"WHERE {_current_edge} UNION "
                f"SELECT DISTINCT object_canonical FROM knowledge_graph "
                f"WHERE {_current_edge}"
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
        # freed a meaningful number of pages. VACUUM may RENUMBER the implicit
        # rowids of the TEXT-PK tables (episodes/chunks/aggregation_nodes),
        # divorcing their FTS and vec_* shadows — the resync restores the
        # mapping before anything (this dream's aggregation included) reads
        # through it. Skipping it caused the 29%-reuse storms of dream runs
        # 725/730 (2026-07-09/10): each post-gap prune crossed the VACUUM
        # threshold, renumbered episodes, and left KNN candidate blocking
        # translating neighbors to the wrong episode ids.
        if cfg.vacuum_after_prune and pruned >= cfg.vacuum_min_pruned:
            conn.execute("VACUUM")
            core_db.resync_rowid_shadows(conn)
            log.info("retention.vacuum pruned=%d", pruned)

        # Freeze the episode set the clusterer reads BEFORE the episode-
        # embedding drain below, so every episode inside the snapshot has its
        # vector persisted by the time clustering reads it. The ceiling itself
        # exists because the MCP server writes episodes asynchronously: a stray
        # landing mid-build would shift cluster membership -> new node ids -> a
        # spurious near-full refusion (dream runs 678/680, 2026-06-28). Taking
        # it AFTER the drain (the original order) left a second hole: a stray
        # landing between drain and ceiling joined the snapshot vector-less,
        # clustered on entities alone, then re-clustered WITH its vector next
        # dream — a guaranteed two-dream membership flip. Strays now land above
        # the ceiling and defer wholesale to the next dream.
        episode_ceiling = conn.execute(
            "SELECT MAX(e.rowid) AS m FROM episodes e "
            "JOIN sessions s ON s.id = e.session_id "
            "WHERE e.digest_generation IS NULL "
            "OR e.digest_generation = s.digest_published_generation"
        ).fetchone()["m"]

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

            # E1 narrative facts: embed batch OUTSIDE the write lock (fetch is
            # the network call, persist is the transaction) — the phase1
            # lock-free pattern. fetch returns None on a pre-v26 store.
            pending_facts = fetch_fact_embeddings(conn, embedding_client)
            if pending_facts is not None:
                with core_db.transaction(conn):
                    report.facts_embedded = persist_fact_embeddings(
                        conn, pending_facts
                    )
                report.facts_embedded_from_cache = pending_facts.cache_hits

        # Phase-2 RAPTOR aggregation. Runs last (needs the fresh episode
        # embeddings the clusterer reads) and is a no-op unless the layer is
        # enabled. Manages its own transactions; off by default so it costs
        # nothing for clients that haven't opted in.
        if cfg.aggregation_nodes_enabled:
            try:
                # Repair step for stores skewed by a pre-fix VACUUM (the
                # resync above only covers VACUUMs from now on): a proven
                # vec_episodes/rowid mismatch rebuilds all rowid shadows so
                # candidate blocking stops clustering on garbage neighborhoods.
                if core_db.heal_rowid_shadows(conn):
                    log.info("aggregate.pre_build_shadow_heal")
                agg = build_aggregation_nodes(
                    conn, cfg, llm, embedding_client,
                    episode_ceiling_rowid=episode_ceiling,
                )
                report.aggregation_nodes_built = agg.nodes
                report.aggregation_nodes_reused = agg.reused
                report.aggregation_fusion_failures = agg.fusion_failures
                report.aggregation_input_episodes = agg.input_episodes
                report.aggregation_blocking = agg.blocking
                report.aggregation_level0_missed = agg.level0_missed
                report.aggregation_leaf_changed = agg.leaf_changed
                report.aggregation_predicted_rebuild = agg.predicted_rebuild
                report.aggregation_keying_residual = agg.keying_residual
                report.aggregation_rebuilt_level0 = agg.rebuilt_level0
                report.aggregation_rebuilt_rollup = agg.rebuilt_rollup
                report.aggregation_rebuilt_root = agg.rebuilt_root
                report.aggregation_leaf_added = agg.leaf_added
                report.aggregation_leaf_removed = agg.leaf_removed
                report.aggregation_facts_rekey = agg.facts_rekey
            except Exception:
                log.exception("aggregate.build_failure")

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
                aggregation_nodes_built = ?,
                aggregation_nodes_reused = ?,
                aggregation_fusion_failures = ?,
                aggregation_input_episodes = ?,
                aggregation_blocking = ?,
                aggregation_level0_missed = ?,
                aggregation_leaf_changed = ?,
                aggregation_predicted_rebuild = ?,
                aggregation_keying_residual = ?,
                aggregation_facts_rekey = ?,
                aggregation_rebuilt_level0 = ?,
                aggregation_rebuilt_rollup = ?,
                aggregation_rebuilt_root = ?,
                aggregation_leaf_added = ?,
                aggregation_leaf_removed = ?,
                digest_failures = ?,
                digest_quarantined = ?,
                episodes_created = ?,
                facts_extracted = ?,
                fact_failures = ?,
                profile_items_extracted = ?,
                profile_failures = ?,
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
                report.aggregation_nodes_built,
                report.aggregation_nodes_reused,
                report.aggregation_fusion_failures,
                report.aggregation_input_episodes,
                report.aggregation_blocking,
                report.aggregation_level0_missed,
                report.aggregation_leaf_changed,
                report.aggregation_predicted_rebuild,
                report.aggregation_keying_residual,
                report.aggregation_facts_rekey,
                report.aggregation_rebuilt_level0,
                report.aggregation_rebuilt_rollup,
                report.aggregation_rebuilt_root,
                report.aggregation_leaf_added,
                report.aggregation_leaf_removed,
                report.digest_failures,
                report.digest_quarantined,
                report.episodes_created,
                report.facts_extracted,
                report.fact_failures,
                report.profile_items_extracted,
                report.profile_failures,
                run_id,
            ),
        )
        log.info(
            "dream.end run_id=%d sessions=%d chunks_processed=%d/%d chunk_extraction_failures=%d triples=%d markers=%d chunks_from_cache=%d messages_embedded=%d messages_from_cache=%d edges_from_cache=%d agg_nodes=%d agg_reused=%d agg_failures=%d agg_input=%d agg_level0_missed=%s agg_leaf_changed=%s agg_predicted=%s agg_keying_residual=%s agg_facts_rekey=%s agg_rebuilt_l0=%s agg_rebuilt_rollup=%s agg_rebuilt_root=%s agg_blocking=%s digest_failures=%d digest_quarantined=%d episodes_created=%d facts=%d fact_failures=%d profile_items=%d profile_failures=%d budget_exhausted=%s",
            run_id,
            report.sessions_processed,
            report.chunks_processed,
            report.chunks_seen,
            report.chunk_extraction_failures,
            report.triples_extracted,
            report.markers_extracted,
            report.chunks_embedded_from_cache,
            report.messages_embedded,
            report.messages_embedded_from_cache,
            report.edges_embedded_from_cache,
            report.aggregation_nodes_built,
            report.aggregation_nodes_reused,
            report.aggregation_fusion_failures,
            report.aggregation_input_episodes,
            report.aggregation_level0_missed,
            report.aggregation_leaf_changed,
            report.aggregation_predicted_rebuild,
            report.aggregation_keying_residual,
            report.aggregation_facts_rekey,
            report.aggregation_rebuilt_level0,
            report.aggregation_rebuilt_rollup,
            report.aggregation_rebuilt_root,
            report.aggregation_blocking,
            report.digest_failures,
            report.digest_quarantined,
            report.episodes_created,
            report.facts_extracted,
            report.fact_failures,
            report.profile_items_extracted,
            report.profile_failures,
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


_LOCK_TTL_SECONDS = 120

# How often, at most, a live dream refreshes its lease. Kept well under
# _LOCK_TTL_SECONDS so even a single ultra-heavy session (hundreds of chunks,
# many minutes) heartbeats several times before the lock could look stale.
_LOCK_REFRESH_INTERVAL_SECONDS = 30


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


def _refresh_lock(conn: sqlite3.Connection, holder: str) -> None:
    """Heartbeat the dreaming lease so a live (slow) dream keeps its lock fresh.

    Pushes ``acquired_at`` forward to now so a genuinely slow dream never
    crosses the ``_LOCK_TTL_SECONDS`` staleness line and gets taken over by a
    concurrent trigger. A crashed holder simply stops calling this, so its
    lock still ages past the TTL and is reclaimed.

    Best-effort like :func:`_release_lock` (a refresh failure must never abort
    the dream) and an autocommit bare execute so the new timestamp is visible
    to other connections immediately. The ``holder = ?`` guard means this only
    touches the row if THIS process still owns the lock — if another process
    legitimately took it over, this refresh is a no-op and cannot resurrect or
    steal the lock.
    """
    with contextlib.suppress(sqlite3.Error):
        conn.execute(
            "UPDATE run_lock SET acquired_at = CURRENT_TIMESTAMP"
            " WHERE name = 'dreaming' AND holder = ?",
            (holder,),
        )


def _all_sessions(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT id FROM sessions ORDER BY started_at").fetchall()
    return [r["id"] for r in rows]
