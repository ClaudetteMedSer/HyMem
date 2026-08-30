from __future__ import annotations

import contextlib
import logging
import os
import socket
import sqlite3
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import bitemporal, phase1, phase2, phase3
from hymem.dreaming.aggregate import build_aggregation_nodes
from hymem.dreaming.inference import infer_transitive_edges
from hymem.dreaming.chunks import (
    Chunk,
    extract_baseline_chunks,
    extract_fallback_chunk,
    extract_high_salience_chunks,
    persist_chunks,
)
from hymem.dreaming.embeddings import (
    ChunkEmbedRequest,
    assemble_chunk_pending,
    fetch_chunk_embeddings,
    fetch_edge_embeddings,
    fetch_episode_embeddings,
    fetch_fact_embeddings,
    persist_chunk_embeddings,
    persist_edge_embeddings,
    persist_episode_embeddings,
    persist_fact_embeddings,
    prepare_chunk_embed_batch,
)
from hymem.dreaming.digest import (
    active_episode_prompt_version,
    extract_session_digest,
)
from hymem.dreaming.episodes import persist_episodes
from hymem.dreaming.facts import extract_facts, persist_facts
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
from hymem.dreaming.summary import persist_session_summary
from hymem.dreaming.value_supersession import supersede_competing_values
from hymem.dreaming.user_profile import (
    PROFILE_PROMPT_VERSION,
    extract_user_profile,
    persist_user_profile,
)
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient

log = logging.getLogger("hymem.dreaming")


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
    episodes_created: int = 0
    facts_extracted: int = 0
    fact_failures: int = 0
    facts_embedded: int = 0
    facts_embedded_from_cache: int = 0
    profile_items_extracted: int = 0
    skipped_locked: bool = False
    budget_exhausted: bool = False


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
    except Exception:
        log.exception("phase1.dedup_prepare_failure chunk triples=%d",
                      len(extraction.triples))
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

        for session_id in target_sessions:
            _heartbeat()
            report.sessions_processed += 1
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
                dedup_vectors = _prepare_dedup_vectors(
                    conn, extraction, cfg, embedding_client
                )
                with core_db.transaction(conn):
                    phase1.persist_chunk_results(
                        conn, chunk, extraction, prompt_version=cfg.prompt_version,
                        cfg=cfg, embedding_client=embedding_client,
                        dedup_vectors=dedup_vectors,
                        in_cycle_edges=in_cycle_edges,
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
                        dedup_vectors = _prepare_dedup_vectors(
                            conn, extraction, cfg, embedding_client
                        )
                        with core_db.transaction(conn):
                            phase1.persist_chunk_results(
                                conn, chunk, extraction,
                                prompt_version=cfg.prompt_version,
                                cfg=cfg, embedding_client=embedding_client,
                                dedup_vectors=dedup_vectors,
                                in_cycle_edges=in_cycle_edges,
                            )
                        if extraction.triples or extraction.markers:
                            report.chunks_processed += 1
                            report.triples_extracted += len(extraction.triples)
                            report.markers_extracted += len(extraction.markers)

            # Sessions with no content in either tier used to skip the
            # per-session tail outright — which left short sessions (all user
            # turns under min_chars, no triggers) never digested,
            # digested_prompt_version NULL forever. Mint ONE fallback chunk
            # spanning the whole session so the digest has something to read;
            # truly empty sessions (no user/assistant content) still skip.
            if not chunks and not baseline:
                fallback = extract_fallback_chunk(
                    conn, session_id, max_chars=cfg.dream_digest_max_chars
                )
                if fallback is None:
                    continue
                report.chunks_seen += 1
                with core_db.transaction(conn):
                    persist_chunks(conn, [fallback])
                    index_chunk_mentions(conn, fallback.id, fallback.text)
                    index_chunk_temporal_mentions(conn, fallback.id)
                _kickoff_chunk_embed([fallback])
                # Deliberately NO phase-1 triple extraction and NO
                # had_new_chunk_work: the goal is digest/episode coverage, not
                # graph growth from diagnostic noise — and leaving the flag
                # unset keeps the digest skip-guard below zero-call on later
                # re-dreams of this (unchanged) session.

            # Per-session digest: one LLM call producing episodes + summary +
            # procedures together (replaces three separate calls). Skip it
            # entirely when this session was already digested under the current
            # prompt_version and no chunk was re-extracted this run, so
            # steady-state re-dreams of unchanged sessions cost zero tail calls.
            digested = conn.execute(
                "SELECT summary, digested_prompt_version, profile_prompt_version, "
                "digested_message_id, episodes_prompt_version "
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
            episodes_current = (
                digested is not None
                and digested["episodes_prompt_version"] == episode_prompt_version
            )
            # A REVERT (granular -> blob) is the one case where the blob arm
            # must supersede: the rows it is replacing were written under the
            # granular id shape, so they cannot collide with the bare-range ids
            # this pass writes and UPSERT alone would leave both granularities
            # of the same conversation in the store. Keyed on the STAMP, not on
            # the flag, so it fires only on a store that actually had
            # granularity on: a store that never enabled it reads NULL here,
            # passes no window, and keeps the purely additive blob behaviour it
            # has always had. That is what keeps the feature inert when off.
            had_granular_stamp = (
                digested is not None
                and digested["episodes_prompt_version"] is not None
            )
            already_digested = (
                digested is not None
                and digested["digested_prompt_version"] == cfg.prompt_version
                and episodes_current
            )
            has_summary = digested is not None and bool(digested["summary"])
            # Schema v24: the guard also asks whether the session has traffic
            # ABOVE the digest watermark. `had_new_chunk_work` alone was not
            # enough — messages landing in an already-digested session produce
            # no fresh extraction when their chunks were already processed (or
            # when the baseline tier picked them up), so the session stayed
            # skipped forever and its episodes froze while chunks kept growing.
            watermark = digested["digested_message_id"] if digested is not None else None
            newest_message_id = conn.execute(
                "SELECT MAX(id) AS m FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()["m"]
            caught_up = (
                newest_message_id is None
                or (watermark is not None and watermark >= newest_message_id)
            )
            if not (already_digested and caught_up and not had_new_chunk_work):
                try:
                    digest = extract_session_digest(
                        conn, session_id, llm,
                        max_tokens=cfg.dream_digest_max_tokens,
                        max_chars=cfg.dream_digest_max_chars,
                        # Resume at the watermark only when this session is
                        # already digested under the CURRENT prompt version. A
                        # prompt bump (or a never-digested session) re-reads
                        # from the start so the new prompt refreshes the
                        # existing episodes via UPSERT — the watermark update
                        # below is monotonic, so that re-read cannot un-cover a
                        # tail this session already had.
                        since_message_id=watermark if already_digested else None,
                        granular=cfg.episode_granularity_enabled,
                        max_episodes=cfg.dream_max_episodes_per_session,
                    )
                except Exception:
                    report.digest_failures += 1
                    log.exception("digest.extraction_failure session_id=%s", session_id)
                else:
                    if digest is not None and digest.parse_failed:
                        report.digest_failures += 1
                    if digest is not None:
                        with core_db.transaction(conn):
                            if digest.episodes.items:
                                ep_count = persist_episodes(
                                    conn, session_id, digest.episodes,
                                    granular=cfg.episode_granularity_enabled,
                                    # Supersede the episodes inside the window
                                    # this call re-read, on either side of a
                                    # granularity CHANGE: the rows being
                                    # replaced resolve to a different id shape,
                                    # so UPSERT alone would leave both
                                    # granularities of the same conversation in
                                    # the store. Only reached when the
                                    # extraction produced items (the `if`
                                    # above), so an empty reply can never delete
                                    # a previous extraction's work -- that leaves
                                    # stale rows standing, which is the
                                    # recoverable direction.
                                    supersede_window=(
                                        (digest.start_message_id,
                                         digest.covered_message_id)
                                        if (cfg.episode_granularity_enabled
                                            or had_granular_stamp)
                                        else None
                                    ),
                                )
                                report.episodes_created += ep_count
                                log.debug(
                                    "episodes session_id=%s count=%d", session_id, ep_count
                                )
                            # Don't overwrite an existing (possibly operator-curated)
                            # summary — matches the old extract_session_summary guard.
                            if digest.summary and not has_summary:
                                persist_session_summary(conn, session_id, digest.summary)
                                log.debug("summary session_id=%s", session_id)
                            if digest.procedures.items:
                                pr_count = persist_procedures(
                                    conn, session_id, digest.procedures
                                )
                                log.debug(
                                    "procedures session_id=%s count=%d", session_id, pr_count
                                )
                            # Mark digested only after a successful persist, so a
                            # failed call leaves the marker unset and retries next run.
                            conn.execute(
                                "UPDATE sessions SET digested_prompt_version = ? WHERE id = ?",
                                (cfg.prompt_version, session_id),
                            )
                            # ...and the episode-prompt stamp with it (v35).
                            # Written unconditionally, including the None the
                            # flag-off path supplies: that write is a no-op on
                            # any store that never enabled granularity (NULL
                            # over NULL), and it is what makes a REVERT correct
                            # — turning the flag back off clears the stamp in
                            # the same transaction that rewrote the session's
                            # episodes under the blob prompt.
                            conn.execute(
                                "UPDATE sessions SET episodes_prompt_version = ? "
                                "WHERE id = ?",
                                (episode_prompt_version, session_id),
                            )
                            # Advance the watermark only over what the LLM
                            # actually saw, and never backwards (a re-digest
                            # forced by a prompt bump re-reads from NULL and
                            # must not un-cover the tail). A parse failure
                            # reports no coverage, so the slice retries.
                            if digest.covered_message_id is not None:
                                conn.execute(
                                    "UPDATE sessions SET digested_message_id = "
                                    "MAX(COALESCE(digested_message_id, -1), ?) "
                                    "WHERE id = ?",
                                    (digest.covered_message_id, session_id),
                                )

            # P4 typed user-profile extraction: one LLM call over the session's
            # USER turns only (closed slot vocabulary, schema v18), piggybacking
            # on the per-session digest batching above but with its OWN
            # skip-guard (sessions.profile_prompt_version, schema v19): the
            # digest guard keys on cfg.prompt_version, so sharing it would mean
            # a PROFILE_PROMPT_VERSION bump alone never re-extracts an
            # already-digested session. Steady-state re-dreams of unchanged
            # sessions still make zero tail calls. Persisting is
            # supersession-aware and re-assert-idempotent, so re-running over
            # the same turns is safe. The LLM call runs OUTSIDE the write
            # transaction, like the digest.
            profile_current = (
                digested is not None
                and digested["profile_prompt_version"] == PROFILE_PROMPT_VERSION
            )
            if cfg.profile_extraction_enabled and not (
                profile_current and not had_new_chunk_work
            ):
                try:
                    profile = extract_user_profile(
                        conn, session_id, llm,
                        max_chars=cfg.dream_digest_max_chars,
                        max_items=cfg.profile_max_items_per_session,
                    )
                except Exception:
                    log.exception(
                        "profile.extraction_failure session_id=%s", session_id
                    )
                else:
                    if profile is not None:
                        # A valid extraction with ZERO items is a legitimate
                        # "nothing here": persist whatever validated (maybe
                        # nothing) and stamp the prompt version in the SAME
                        # transaction, so only an LLM failure (the except
                        # above) leaves the stamp unset and retries next run.
                        with core_db.transaction(conn):
                            persisted = persist_user_profile(
                                conn, profile,
                                redact_values=cfg.redact_secrets,
                            )
                            conn.execute(
                                "UPDATE sessions SET profile_prompt_version = ? "
                                "WHERE id = ?",
                                (PROFILE_PROMPT_VERSION, session_id),
                            )
                        report.profile_items_extracted += persisted
                        if persisted:
                            log.debug(
                                "profile session_id=%s rows=%d",
                                session_id, persisted,
                            )

            # E1 narrative facts: one LLM call over the session's raw
            # user/assistant tail (schema v26). Its own watermark
            # (sessions.facts_message_id, the v24 pattern) is the ONLY
            # skip-guard: unlike digest/profile there is no prompt-version
            # stamp, because a FACTS_PROMPT_VERSION bump extracts FORWARD ONLY
            # — covered ranges are never re-extracted, so re-reading them on a
            # bump would be wrong, not just wasteful. Steady-state re-dreams of
            # unchanged sessions cost zero calls (watermark at tail). A parse
            # failure counts into dream_runs.fact_failures and holds the
            # watermark, so the slice retries next dream. The watermark SELECT
            # doubles as the pre-v26 guard: on a store without the column the
            # whole block degrades to a debug log, no crash. LLM call outside
            # the write transaction, like the digest.
            if cfg.facts_extraction_enabled:
                try:
                    facts_watermark = conn.execute(
                        "SELECT facts_message_id FROM sessions WHERE id = ?",
                        (session_id,),
                    ).fetchone()["facts_message_id"]
                except sqlite3.OperationalError:
                    log.debug("facts.skipped_pre_v26 session_id=%s", session_id)
                else:
                    facts_caught_up = (
                        newest_message_id is None
                        or (facts_watermark is not None
                            and facts_watermark >= newest_message_id)
                    )
                    if not facts_caught_up:
                        try:
                            facts = extract_facts(
                                conn, session_id, llm, cfg,
                                since_message_id=facts_watermark,
                            )
                        except Exception:
                            report.fact_failures += 1
                            log.exception(
                                "facts.extraction_failure session_id=%s", session_id
                            )
                        else:
                            if facts is not None and facts.parse_failed:
                                report.fact_failures += 1
                            if facts is not None and not facts.parse_failed:
                                with core_db.transaction(conn):
                                    persisted = persist_facts(conn, session_id, facts)
                                    report.facts_extracted += persisted
                                    # Advance only over what the LLM actually
                                    # saw, never backwards — the digest
                                    # watermark contract.
                                    if facts.covered_message_id is not None:
                                        conn.execute(
                                            "UPDATE sessions SET facts_message_id = "
                                            "MAX(COALESCE(facts_message_id, -1), ?) "
                                            "WHERE id = ?",
                                            (facts.covered_message_id, session_id),
                                        )
                                if persisted:
                                    log.debug(
                                        "facts session_id=%s rows=%d",
                                        session_id, persisted,
                                    )

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
            "SELECT MAX(rowid) AS m FROM episodes"
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
                episodes_created = ?,
                facts_extracted = ?,
                fact_failures = ?,
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
                report.episodes_created,
                report.facts_extracted,
                report.fact_failures,
                run_id,
            ),
        )
        log.info(
            "dream.end run_id=%d sessions=%d chunks_processed=%d/%d chunk_extraction_failures=%d triples=%d markers=%d chunks_from_cache=%d edges_from_cache=%d agg_nodes=%d agg_reused=%d agg_failures=%d agg_input=%d agg_level0_missed=%s agg_leaf_changed=%s agg_predicted=%s agg_keying_residual=%s agg_facts_rekey=%s agg_rebuilt_l0=%s agg_rebuilt_rollup=%s agg_rebuilt_root=%s agg_blocking=%s digest_failures=%d episodes_created=%d facts=%d fact_failures=%d budget_exhausted=%s",
            run_id,
            report.sessions_processed,
            report.chunks_processed,
            report.chunks_seen,
            report.chunk_extraction_failures,
            report.triples_extracted,
            report.markers_extracted,
            report.chunks_embedded_from_cache,
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
            report.episodes_created,
            report.facts_extracted,
            report.fact_failures,
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
