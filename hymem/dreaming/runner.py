from __future__ import annotations

import contextlib
import inspect
import logging
import os
import secrets
import socket
import sqlite3
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields
from typing import Callable

from hymem.config import HyMemConfig
from hymem.deadline import (
    DeadlineBoundEmbeddingClient,
    DeadlineBoundLLMClient,
    DeadlineExceeded,
    MonotonicDeadline,
    use_deadline,
)
from hymem.core import db as core_db
from hymem.dreaming import bitemporal, phase1, phase2, phase3
from hymem.dreaming.aggregate import (
    aggregation_config_version,
    build_aggregation_nodes,
)
from hymem.dreaming.aggregation_health import (
    begin_aggregation_build,
    complete_aggregation_build,
    record_aggregation_build_failure,
)
from hymem.dreaming.aggregation_generation import (
    aggregation_generation_binding,
    register_aggregation_generation,
    validate_current_aggregation_generation_binding,
)
from hymem.dreaming.inference import infer_transitive_edges
from hymem.dreaming.chunks import (
    BASELINE_SALIENCE_REASON,
    Chunk,
    chunk_extraction_is_quarantined,
    extract_baseline_chunks,
    extract_high_salience_chunks,
    has_pending_persisted_chunks,
    load_pending_persisted_chunks,
    persist_chunks,
    recover_legacy_chunk_source_manifests,
    record_unrecoverable_chunk_losses,
    source_materialization_config_version,
)
from hymem.dreaming.embeddings import (
    ChunkEmbedRequest,
    _embedding_identity,
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
    digest_retry_is_quarantined,
    extract_session_digest,
    record_digest_failure,
)
from hymem.dreaming.lossless import (
    clear_coverage_integrity_failure,
    covered_messages_after,
    lossless_cursor_is_valid,
    materialize_message_coverage,
    record_coverage_integrity_failure,
)
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
    profile_retry_is_quarantined,
    profile_user_tail_message_id,
    publish_profile_generation,
    record_profile_failure,
    stage_profile_extraction,
)
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.contract import (
    extraction_cache_key,
    validate_effective_config_extraction_contract,
)
from hymem.extraction.producer import (
    _register_phase1_producer_proxy,
    phase1_generation_binding,
    validate_current_phase1_generation_binding,
)
from hymem.extraction.chunk import (
    SHIPPED_MAX_EXTRACTION_PROVIDER_ATTEMPTS_PER_CHUNK,
)
from hymem.extraction.llm import LLMClient, measure_provider_attempts
from hymem.core.graph import live_edge_predicate

log = logging.getLogger("hymem.dreaming")


class DreamLeaseLost(RuntimeError):
    """Public failure raised when a dream loses its cross-process lease.

    The internal fencing signal derives directly from ``BaseException`` so it
    cannot be swallowed by extraction recovery code.  Public callers should
    still receive an ordinary operational exception with a stable, sanitized
    message, so :func:`run_dreaming` translates the signal at its boundary.
    """


def _lossless_cursor_is_valid(
    conn: sqlite3.Connection,
    session_id: str,
    cursor_message_id: int | None,
    partial_message_id: int | None,
    offset: int,
    *,
    roles: frozenset[str] | None = None,
) -> bool:
    """Compatibility wrapper around the shared status/write-path validator."""
    return lossless_cursor_is_valid(
        conn,
        session_id,
        cursor_message_id,
        partial_message_id,
        offset,
        roles=roles,
    )


@dataclass
class DreamReport:
    sessions_processed: int = 0
    chunks_seen: int = 0
    # Newly and authoritatively completed Phase-1 units in this invocation.
    # A verified empty is a completion; a cached/already-processed unit is not.
    # Extracted triples and markers are counted independently below.
    chunks_processed: int = 0
    # Extractions that did not complete this run (unparseable / wrong-shaped
    # reply). NOT persisted to dream_runs: those chunks are held unmarked and
    # retried, so a rising count across consecutive dreams is the ingest
    # analogue of a stuck fusion. In-memory + dream.end only.
    chunk_extraction_failures: int = 0
    # Logical Phase-1 completions and measured underlying provider requests.
    # Digest/profile/facts calls are intentionally excluded.
    chunk_extraction_completion_calls: int = 0
    chunk_extraction_provider_attempts: int = 0
    # True only when an otherwise-actionable chunk was left untouched because
    # the soft per-cycle provider-attempt ceiling had already been reached.
    extraction_provider_attempt_budget_exhausted: bool = False
    # Sessions whose ordered lossless source failed structural validation in
    # this run. Durable identity/reason state lives in
    # coverage_integrity_failures and is cleared only by a complete good walk.
    coverage_integrity_failures: int = 0
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
    aggregation_build_exceptions: int = 0
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


# Current-schema classification for consumers that turn a DreamReport into a
# completion claim.  This is deliberately an exact partition, rather than a
# loose list of known failures: adding a field to DreamReport must classify it
# here before an MCP/benchmark-style consumer can accidentally ignore it.
DREAM_REPORT_COUNT_FIELDS = (
    "sessions_processed",
    "chunks_seen",
    "chunks_processed",
    "chunk_extraction_completion_calls",
    "chunk_extraction_provider_attempts",
    "triples_extracted",
    "markers_extracted",
    "rules_extracted",
    "chunks_embedded",
    "chunks_embedded_from_cache",
    "messages_embedded",
    "messages_embedded_from_cache",
    "edges_embedded",
    "edges_embedded_from_cache",
    "episodes_embedded",
    "episodes_embedded_from_cache",
    "aggregation_nodes_built",
    "aggregation_nodes_reused",
    "aggregation_input_episodes",
    "episodes_created",
    "facts_extracted",
    "facts_embedded",
    "facts_embedded_from_cache",
    "profile_items_extracted",
)
DREAM_REPORT_ERROR_FIELDS = (
    "chunk_extraction_failures",
    "coverage_integrity_failures",
    "aggregation_fusion_failures",
    "aggregation_build_exceptions",
    "digest_failures",
    "digest_quarantined",
    "fact_failures",
    "profile_failures",
)
DREAM_REPORT_NULLABLE_COUNT_FIELDS = (
    "aggregation_level0_missed",
    "aggregation_leaf_changed",
    "aggregation_predicted_rebuild",
    "aggregation_keying_residual",
    "aggregation_rebuilt_level0",
    "aggregation_rebuilt_rollup",
    "aggregation_rebuilt_root",
    "aggregation_leaf_added",
    "aggregation_leaf_removed",
    "aggregation_facts_rekey",
)
DREAM_REPORT_TEXT_FIELDS = ("aggregation_blocking",)
DREAM_REPORT_BOOLEAN_GATE_FIELDS = (
    "extraction_provider_attempt_budget_exhausted",
    "skipped_locked",
    "budget_exhausted",
)
DREAM_REPORT_FIELD_NAMES = (
    *DREAM_REPORT_COUNT_FIELDS,
    *DREAM_REPORT_ERROR_FIELDS,
    *DREAM_REPORT_NULLABLE_COUNT_FIELDS,
    *DREAM_REPORT_TEXT_FIELDS,
    *DREAM_REPORT_BOOLEAN_GATE_FIELDS,
)

if {field.name for field in fields(DreamReport)} != set(DREAM_REPORT_FIELD_NAMES):
    raise RuntimeError("DreamReport fields must have an explicit schema classification")
if len(DREAM_REPORT_FIELD_NAMES) != len(set(DREAM_REPORT_FIELD_NAMES)):
    raise RuntimeError("DreamReport schema classifications must not overlap")


class _CountingPhase1LLM:
    """Count actual Phase-1 provider attempts, including raised calls.

    ChunkResult/ChunkExtraction carry the normal-path count as a contract. This
    narrow wrapper additionally preserves exact accounting if an unexpected
    exception occurs after the provider returned but before phase1 can hand its
    result back to the runner.
    """

    def __init__(self, delegate: LLMClient):
        self.delegate = delegate
        self.completion_calls = 0
        self.provider_attempts = 0
        _register_phase1_producer_proxy(self, delegate)

    def _verified_delegate(self):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        return _verified_phase1_proxy_delegate(self)

    @property
    def request_attempts(self) -> int:
        return self.provider_attempts

    def track_provider_attempts(self):
        """Forward the optional concurrency-safe scope to the real client.

        ``measure_provider_attempts`` catches an absent delegate extension and
        falls back to this wrapper's cumulative counter for serial custom
        clients. Keeping the scope on the delegate lets nested chunk/runner
        meters observe the same request without consulting shared globals.
        """
        try:
            factory = getattr(self._verified_delegate(), "track_provider_attempts")
        except (AttributeError, TypeError):
            raise AttributeError(
                "delegate has no scoped provider-attempt telemetry"
            ) from None
        if not callable(factory):
            raise AttributeError(
                "delegate has no scoped provider-attempt telemetry"
            )
        return factory()

    def complete(self, request):
        delegate = self._verified_delegate()
        self.completion_calls += 1
        attempt_measurement = None
        try:
            with measure_provider_attempts(delegate) as attempt_measurement:
                return delegate.complete(request)
        finally:
            self.provider_attempts += (
                attempt_measurement.attempts
                if attempt_measurement is not None
                else 1
            )


class _HeartbeatLLMClient:
    """Refresh/check the dream lease around every logical LLM completion.

    The dedicated periodic heartbeater covers a single provider call that is
    blocked longer than the TTL on file-backed stores. These boundaries add a
    deterministic progress hook between extraction rerolls/splits and cover
    in-memory test stores, which cannot be shared cross-process in any case.
    """

    def __init__(self, delegate: LLMClient, heartbeat: Callable[[], None]):
        self._delegate = delegate
        self._heartbeat = heartbeat
        _register_phase1_producer_proxy(self, delegate)

    def _verified_delegate(self):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        return _verified_phase1_proxy_delegate(self)

    def __getattr__(self, name: str):
        return getattr(self._verified_delegate(), name)

    def complete(self, request):
        delegate = self._verified_delegate()
        self._heartbeat()
        try:
            return delegate.complete(request)
        finally:
            # If both the provider and ownership failed, lease loss wins: an
            # obsolete owner must not enter provider-failure publication paths.
            self._heartbeat()


_RUNNER_PROXY_DISPATCH_NAMES = (
    "embed", "complete", "model", "dim", "backend", "quality",
    "network_free", "close", "__getattribute__", "__getattr__",
    "_embed_serialized", "_validated_vector", "_verified_delegate",
)
_RUNNER_PROXY_ORIGINAL_GUARDS = {
    _CountingPhase1LLM: tuple(
        inspect.getattr_static(_CountingPhase1LLM, name, None)
        for name in _RUNNER_PROXY_DISPATCH_NAMES
    ),
    _HeartbeatLLMClient: tuple(
        inspect.getattr_static(_HeartbeatLLMClient, name, None)
        for name in _RUNNER_PROXY_DISPATCH_NAMES
    ),
}


class _DreamLeaseHeartbeat:
    """Renew a file-backed dreaming lease from one dedicated connection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        holder: str,
        *,
        interval_seconds: float,
    ) -> None:
        self._holder = holder
        self._interval_seconds = max(0.01, float(interval_seconds))
        self._stop = threading.Event()
        self._failed = threading.Event()
        self._failure: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._conn: sqlite3.Connection | None = None

        # sqlite reports an empty file for :memory: and temporary databases.
        # Such stores cannot be shared by another process, so boundary
        # heartbeats plus transaction fencing are the safe fallback.
        db_path = ""
        for row in conn.execute("PRAGMA database_list").fetchall():
            name = row[1] if not isinstance(row, sqlite3.Row) else row["name"]
            if name == "main":
                db_path = (
                    row[2] if not isinstance(row, sqlite3.Row) else row["file"]
                ) or ""
                break
        if db_path:
            heartbeat_conn: sqlite3.Connection | None = None
            try:
                heartbeat_conn = sqlite3.connect(
                    str(db_path),
                    isolation_level=None,
                    check_same_thread=False,
                    timeout=10.0,
                )
                heartbeat_conn.execute("PRAGMA busy_timeout = 10000")
                self._conn = heartbeat_conn
            except BaseException:
                if heartbeat_conn is not None:
                    heartbeat_conn.close()
                raise

    def start(self) -> None:
        if self._conn is None:
            return
        if self._thread is not None:
            raise RuntimeError("dream lease heartbeat already started")
        thread = threading.Thread(
            target=self._run,
            name="hymem-dream-lease-heartbeat",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                self._conn.close()
            self._conn = None
            raise
        self._thread = thread

    def _run(self) -> None:
        assert self._conn is not None
        try:
            while not self._stop.wait(self._interval_seconds):
                try:
                    _refresh_lock(self._conn, self._holder)
                except core_db.LeaseOwnershipLost as exc:
                    self._failure = exc
                    self._failed.set()
                    return
                except sqlite3.Error as exc:
                    # A short foreground BEGIN IMMEDIATE can make renewal busy.
                    # It also prevents takeover, so retry rather than declaring
                    # false lease loss. Exact-token zero-row renewal above is
                    # the authoritative ownership signal.
                    log.warning(
                        "dream.lease_heartbeat_sqlite_failure error=%s",
                        type(exc).__name__,
                    )
                except BaseException as exc:
                    self._failure = exc
                    self._failed.set()
                    return
        finally:
            with contextlib.suppress(sqlite3.Error):
                self._conn.close()

    def check(self) -> None:
        if self._failed.is_set():
            raise core_db.LeaseOwnershipLost(
                "dreaming lease renewal failed"
            ) from self._failure

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            if self._conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.close()
                self._conn = None
            return
        self._thread.join(timeout=11.0)
        if self._thread.is_alive():
            log.error("dream.lease_heartbeat_stop_timeout")


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
    deadline: MonotonicDeadline | None = None,
) -> DreamReport:
    """Run one dream with an optional caller-owned absolute deadline.

    Ordinary host calls pass no deadline and retain their historical behavior.
    Benchmark convergence supplies one absolute monotonic deadline shared by
    every cycle.  Transparent provider proxies make custom clients cooperative
    and ensure a provider result returned after expiry is never published.
    """

    # Resolve producer identity before transparent deadline/heartbeat/counting
    # wrappers are installed.  Those wrappers change execution control, not the
    # effective provider request, and an unknown wrapper must never invent a
    # fresh durable cache identity on every run.
    validate_effective_config_extraction_contract(cfg)
    phase1_identity_client = llm
    embedding_identity_client = embedding_client
    phase1_generation = phase1_generation_binding(cfg.prompt_version, llm)
    aggregation_generation = (
        aggregation_generation_binding(cfg, llm)
        if cfg.aggregation_nodes_enabled else None
    )

    if deadline is not None:
        if not isinstance(deadline, MonotonicDeadline):
            raise TypeError("deadline must be a MonotonicDeadline or None")
        deadline.check()
        llm = DeadlineBoundLLMClient(llm, deadline)
        _register_phase1_producer_proxy(llm, phase1_identity_client)
        if embedding_client is not None:
            embedding_client = DeadlineBoundEmbeddingClient(
                embedding_client, deadline,
            )
            _register_phase1_producer_proxy(
                embedding_client, embedding_identity_client,
            )
    try:
        with use_deadline(deadline):
            return _run_dreaming(
                conn,
                cfg,
                llm,
                session_ids=session_ids,
                embedding_client=embedding_client,
                deadline=deadline,
                phase1_generation=phase1_generation,
                phase1_identity_client=phase1_identity_client,
                aggregation_generation=aggregation_generation,
            )
    except core_db.LeaseOwnershipLost as exc:
        # Keep the BaseException-style sentinel strictly internal: ordinary
        # API/MCP/scheduler callers receive a conventional operational error.
        # Neither message includes the opaque random holder token.
        raise DreamLeaseLost("dreaming lease ownership lost") from exc


def _run_dreaming(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    llm: LLMClient,
    *,
    session_ids: list[str] | None = None,
    embedding_client: EmbeddingClient | None = None,
    deadline: MonotonicDeadline | None = None,
    phase1_generation: dict[str, object],
    phase1_identity_client: LLMClient,
    aggregation_generation: dict[str, object] | None,
) -> DreamReport:
    """Run all three dreaming phases. Holds an advisory lock so concurrent runs
    bail out instead of double-processing.
    """
    def _check_deadline() -> None:
        if deadline is not None:
            deadline.check()

    def _aggregation_health_write(write):
        """Publish one aggregation-health transition at a fenced boundary.

        The health helpers are intentionally usable by ordinary unbounded
        hosts as single autocommit statements. A dream always needs the
        stronger boundary supplied by ``core_db.transaction``: its pre-commit
        checks roll back on either deadline expiry or lease loss. The pending
        marker is therefore never cleared (or embellished as a handled
        failure) by late or obsolete aggregation control flow.
        """

        _check_deadline()
        with core_db.transaction(conn):
            value = write()
            _check_deadline()
        return value

    _check_deadline()
    validate_effective_config_extraction_contract(cfg)
    phase1_generation = validate_current_phase1_generation_binding(
        phase1_generation,
        prompt_version=cfg.prompt_version,
    )
    phase1_cache_key = str(phase1_generation["extraction_cache_key"])
    phase1_generation_key = str(phase1_generation["generation_key"])
    if cfg.aggregation_nodes_enabled:
        if aggregation_generation is None:
            raise RuntimeError("enabled aggregation has no producer generation")
        aggregation_generation = validate_current_aggregation_generation_binding(
            aggregation_generation, cfg=cfg,
        )
        aggregation_generation_key = str(
            aggregation_generation["generation_key"]
        )
    else:
        aggregation_generation_key = None
    aggregation_attempt_token: int | None = None
    report = DreamReport()
    aggregation_material_epoch_key: str | None = None
    holder = _new_lease_token()

    def _finish_run_housekeeping(run_id: int, error: str) -> None:
        """Terminalize lifecycle telemetry after an abort, never memory data."""

        with contextlib.suppress(sqlite3.Error):
            conn.execute(
                "UPDATE dream_runs SET ended_at = CURRENT_TIMESTAMP, error = ? "
                "WHERE id = ?",
                (error, run_id),
            )

    def _setup_run() -> tuple[str | None, int, bool]:
        """Create telemetry/lease with cleanup if setup is interrupted."""

        run_id: int | None = None
        acquired = False
        try:
            _check_deadline()
            aggregation_version = (
                aggregation_config_version(cfg)
                if cfg.aggregation_nodes_enabled else None
            )
            if aggregation_generation is not None:
                register_aggregation_generation(conn, aggregation_generation)
            run_id = int(conn.execute(
                "INSERT INTO dream_runs(started_at, aggregation_effective, "
                "aggregation_config_version, aggregation_generation_key) "
                "VALUES (CURRENT_TIMESTAMP, ?, ?, ?)",
                (
                    "enabled" if cfg.aggregation_nodes_enabled else "disabled",
                    aggregation_version,
                    aggregation_generation_key,
                ),
            ).lastrowid)
            acquired = _acquire_lock(conn, holder)
            _check_deadline()
            return aggregation_version, run_id, acquired
        except BaseException:
            # These two bounded writes are lifecycle housekeeping. They are
            # intentionally allowed after expiry so a timed-out benchmark does
            # not leave a live lease or an in-progress run that poisons the
            # next convergence attempt. No semantic/index/cursor state is
            # published after the deadline.
            if acquired:
                _release_lock(conn, holder)
            if run_id is not None:
                _finish_run_housekeeping(run_id, "setup_interrupted")
            raise

    aggregation_version, run_id, lock_acquired = _setup_run()

    if not lock_acquired:
        report.skipped_locked = True
        log.info("dream.skipped_locked")
        conn.execute(
            "UPDATE dream_runs SET ended_at = CURRENT_TIMESTAMP, skipped_locked = 1 WHERE id = ?",
            (run_id,),
        )
        return report

    embed_executor: ThreadPoolExecutor | None = None
    embed_inflight: list[tuple[ChunkEmbedRequest, Future]] = []
    try:
        if embedding_client is not None and deadline is None:
            # Deadline-bound benchmark work stays on the calling thread.
            # Python cannot safely kill a transport call running in a worker,
            # so waiting for the historical overlap executor would make the
            # advertised bound depend on an unkillable background thread. The
            # normal host path retains overlap; bounded runs use the existing
            # synchronous catch-all pass with a remaining-time request cap.
            embed_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="hymem-embed"
            )
    except BaseException:
        _release_lock(conn, holder)
        _finish_run_housekeeping(run_id, "executor_setup_interrupted")
        raise

    lease_heartbeat: _DreamLeaseHeartbeat | None = None
    lease_fence_token = None
    try:
        lease_heartbeat = _DreamLeaseHeartbeat(
            conn,
            holder,
            interval_seconds=_LOCK_REFRESH_INTERVAL_SECONDS,
        )
        lease_fence_token = core_db.activate_transaction_lease_fence(
            conn,
            name="dreaming",
            holder=holder,
        )
        lease_heartbeat.start()
    except BaseException as primary:
        cleanup_failures: list[tuple[str, BaseException]] = []
        for label, cleanup in (
            (
                "lease heartbeat",
                lease_heartbeat.stop if lease_heartbeat is not None else None,
            ),
            (
                "lease fence",
                (
                    lambda: core_db.deactivate_transaction_lease_fence(
                        lease_fence_token
                    )
                    if lease_fence_token is not None else None
                ),
            ),
            (
                "embedding executor",
                (
                    lambda: embed_executor.shutdown(wait=True)
                    if embed_executor is not None else None
                ),
            ),
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except BaseException as cleanup_error:
                cleanup_failures.append((label, cleanup_error))
        _release_lock(conn, holder)
        _finish_run_housekeeping(run_id, "lease_setup_interrupted")
        for label, cleanup_error in cleanup_failures:
            with contextlib.suppress(AttributeError, TypeError):
                primary.add_note(
                    f"{label} cleanup failed: {type(cleanup_error).__name__}"
                )
        raise

    try:
        _check_deadline()
        if session_ids:
            # A non-empty explicit replay/debug request owns its ordering.
            target_sessions = list(session_ids)
        else:
            target_sessions = _all_sessions(conn)
            if len(target_sessions) > 1:
                # A global Phase-1 ceiling combined with a fixed oldest-first
                # order can permanently starve later sessions when an active
                # early session receives work faster than the budget drains it.
                # Rotate the durable order once per run. This keeps each run
                # deterministic while giving every session the first slot over
                # repeated cycles; high, baseline, and tail work all retain the
                # same within-session priority.
                offset = (int(run_id) - 1) % len(target_sessions)
                target_sessions = (
                    target_sessions[offset:] + target_sessions[:offset]
                )
        log.info(
            "dream.start run_id=%d sessions=%d", run_id, len(target_sessions)
        )

        chunks_remaining = cfg.dream_budget
        baseline_remaining = cfg.dream_baseline_budget
        baseline_candidates_by_session: dict[str, list[Chunk]] = {}
        current_high_priority_chunk_ids: set[str] = set()
        current_baseline_chunk_ids: set[str] = set()
        # Baseline discovery is intentionally complete and durable, but its
        # external embedding workload must stay coupled to actual bounded
        # scheduling. The post-loop catch-all embed pass excludes these IDs
        # until the centralized Phase-1 path really attempts them.
        deferred_baseline_embedding_ids: set[str] = set()

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
            _check_deadline()
            assert lease_heartbeat is not None
            lease_heartbeat.check()
            now = time.monotonic()
            if now - _last_heartbeat[0] >= _LOCK_REFRESH_INTERVAL_SECONDS:
                _refresh_lock(conn, holder)
                _last_heartbeat[0] = now
            lease_heartbeat.check()

        # Every logical LLM completion (Phase 1 retries/splits plus digest,
        # profile, facts, rules, and aggregation fusion) now has deterministic
        # before/after renewal boundaries. The periodic dedicated connection
        # remains responsible while one provider call itself is blocked.
        heartbeat_delegate = llm
        llm = _HeartbeatLLMClient(heartbeat_delegate, _heartbeat)
        _register_phase1_producer_proxy(llm, heartbeat_delegate)

        def _kickoff_chunk_embed(chunks_list: list[Chunk]) -> None:
            """Cache lookup on the main thread, then submit the embedder call
            to a single-worker background thread so Phase 1 LLM calls keep
            running in parallel with the (I/O-bound) embedding API call.

            Skips chunks that already have a row in chunk_embeddings — a
            re-run dream cycle re-persists the same chunk objects, and
            vec_chunks (vec0 virtual table) rejects ``INSERT OR REPLACE`` on
            existing rowids.
            """
            _check_deadline()
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

        def _extract_phase1_chunk(chunk: Chunk, *, tier: str) -> str:
            """Attempt and atomically publish one eligible Phase-1 chunk.

            Returns ``attempted``, ``skipped``, ``chunk_budget``, or
            ``call_budget``. All three scheduling tiers use this one path so
            provider-attempt accounting and the no-partial-publication boundary
            cannot drift apart.
            """
            nonlocal chunks_remaining

            _check_deadline()
            if chunks_remaining <= 0:
                return "chunk_budget"
            if chunk.id in attempted_this_cycle:
                return "skipped"
            already = conn.execute(
                "SELECT 1 FROM current_phase1_publications publication "
                "WHERE publication.chunk_id=? "
                "AND publication.prompt_version=? "
                "AND publication.phase1_generation_key=?",
                (chunk.id, phase1_cache_key, phase1_generation_key),
            ).fetchone()
            if already:
                return "skipped"
            if chunk_extraction_is_quarantined(
                conn,
                chunk.id,
                prompt_version=cfg.prompt_version,
                max_attempts=cfg.chunk_extraction_max_attempts,
                phase1_generation_key=phase1_generation_key,
            ):
                return "skipped"

            call_limit = cfg.dream_extraction_provider_attempt_budget
            if (
                call_limit > 0
                and report.chunk_extraction_provider_attempts >= call_limit
            ):
                if not report.extraction_provider_attempt_budget_exhausted:
                    report.extraction_provider_attempt_budget_exhausted = True
                    report.budget_exhausted = True
                    log.info(
                        "dream.extraction_provider_attempt_budget_exhausted "
                        "budget=%d attempts_used=%d "
                        "shipped_max_chunk_overshoot=%d",
                        call_limit,
                        report.chunk_extraction_provider_attempts,
                        SHIPPED_MAX_EXTRACTION_PROVIDER_ATTEMPTS_PER_CHUNK,
                    )
                return "call_budget"

            _heartbeat()
            chunks_remaining -= 1
            attempted_this_cycle.add(chunk.id)
            counting_llm = _CountingPhase1LLM(llm)
            _register_phase1_producer_proxy(counting_llm, llm)
            if phase1_generation_binding(
                cfg.prompt_version, phase1_identity_client
            ) != phase1_generation:
                raise RuntimeError(
                    "Phase-1 producer identity changed before extraction"
                )
            try:
                extraction = phase1.extract_chunk_results(
                    conn,
                    chunk,
                    counting_llm,
                    prompt_version=cfg.prompt_version,
                    phase1_generation=phase1_generation,
                )
            except phase1.Phase1ProducerDriftError:
                raise
            except Exception:
                report.chunk_extraction_completion_calls += (
                    counting_llm.completion_calls
                )
                report.chunk_extraction_provider_attempts += (
                    counting_llm.provider_attempts
                )
                log.exception(
                    "phase1.llm_failure chunk_id=%s tier=%s "
                    "completion_calls=%d provider_attempts=%d",
                    chunk.id,
                    tier,
                    counting_llm.completion_calls,
                    counting_llm.provider_attempts,
                )
                return "attempted"

            # Do not publish a response if mutable client configuration drifted
            # while the request was in flight.  The prior generation remains
            # authoritative and the run fails visibly instead of labeling an
            # unknown effective request as the binding captured above.
            if phase1_generation_binding(
                cfg.prompt_version, phase1_identity_client
            ) != phase1_generation:
                raise RuntimeError(
                    "Phase-1 producer identity changed during extraction"
                )

            reported_completion_calls = (
                extraction.completion_calls if extraction is not None else 0
            )
            reported_provider_attempts = (
                extraction.provider_attempts if extraction is not None else 0
            )
            if (
                reported_completion_calls != counting_llm.completion_calls
                or reported_provider_attempts != counting_llm.provider_attempts
            ):
                # The wrapper is the exact boundary count. Keep operating, but
                # make any propagation regression immediately visible.
                log.error(
                    "phase1.provider_call_accounting_mismatch "
                    "chunk_id=%s tier=%s result_completion_calls=%d "
                    "actual_completion_calls=%d result_provider_attempts=%d "
                    "actual_provider_attempts=%d",
                    chunk.id,
                    tier,
                    reported_completion_calls,
                    counting_llm.completion_calls,
                    reported_provider_attempts,
                    counting_llm.provider_attempts,
                )
            report.chunk_extraction_completion_calls += (
                counting_llm.completion_calls
            )
            report.chunk_extraction_provider_attempts += (
                counting_llm.provider_attempts
            )
            if extraction is None:
                return "skipped"
            if extraction.failed:
                # Persist still runs: retry/quarantine bookkeeping lives at the
                # publication boundary, while no processed marker or partial
                # graph/profile result is created.
                report.chunk_extraction_failures += 1
                log.warning(
                    "phase1.extraction_failed chunk_id=%s tier=%s "
                    "reason=%s completion_calls=%d provider_attempts=%d "
                    "action=held_for_retry",
                    chunk.id,
                    tier,
                    extraction.failure_reason or "unspecified_failure",
                    counting_llm.completion_calls,
                    counting_llm.provider_attempts,
                )

            dedup_vectors = (
                {}
                if extraction.failed
                else _prepare_dedup_vectors(
                    conn, extraction, cfg, embedding_client
                )
            )
            _check_deadline()
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
            if not extraction.failed:
                report.chunks_processed += 1
                report.triples_extracted += len(extraction.triples)
                report.markers_extracted += len(extraction.markers)
            return "attempted"

        if cfg.redact_secrets:
            # Privacy policy tightening is local, global profile maintenance;
            # do it once per dream even when new extraction is disabled.
            with core_db.transaction(conn):
                enforce_profile_redaction_policy(conn)

        for session_id in target_sessions:
            _check_deadline()
            _heartbeat()
            report.sessions_processed += 1
            # First establish a durable, exact source stream for every role and
            # every message length.  This local write is independent of
            # salience and of the LLM budget.  Digest/provenance work below is
            # never allowed to outrun it.
            try:
                while True:
                    _check_deadline()
                    with core_db.transaction(conn):
                        covered_now = materialize_message_coverage(
                            conn, session_id, limit=256
                        )
                    _heartbeat()
                    if covered_now < 256:
                        break
            except Exception:
                with core_db.transaction(conn):
                    record_coverage_integrity_failure(
                        conn,
                        session_id,
                        reason="materialization_failure",
                    )
                report.coverage_integrity_failures += 1
                log.exception(
                    "coverage.materialization_failure session_id=%s", session_id
                )
                # Retrying next dream is lossless; proceeding with a partial
                # stream could advance derived cursors past an uncovered turn.
                continue
            # Fence both builders and the producer acknowledgement to the exact
            # frontier that existed after this run's coverage walk. A public
            # ingest can atomically append and cover a later turn before either
            # builder starts; it belongs to the next cycle and must not leak
            # into candidates attributed to this older acknowledgement.
            materialization_target = conn.execute(
                "SELECT coverage_message_id FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()["coverage_message_id"]
            try:
                chunks = extract_high_salience_chunks(
                    conn,
                    session_id,
                    min_chars=cfg.salience_min_chars,
                    through_message_id=materialization_target,
                )

                # Materialize every actionable baseline source before *any*
                # Phase-1 provider call. Chunk/provider budgets can stop
                # scheduling below, and raw-message retention runs later in
                # this cycle; neither may erase discovery of a short fact. The
                # builder reads the validated lossless stream, so an untouched
                # chunk later evicted by the soft extraction-chunk cap can be
                # recreated with the same ID, bytes, and exact source manifest
                # after raw rows are gone.
                baseline_candidates = extract_baseline_chunks(
                    conn,
                    session_id,
                    prompt_version=cfg.prompt_version,
                    limit=None,
                    min_chars=cfg.salience_min_chars,
                    max_attempts=cfg.chunk_extraction_max_attempts,
                    phase1_generation_key=phase1_generation_key,
                    exclude_ids=None,
                    through_message_id=materialization_target,
                )
            except Exception:
                # A corrupt proof must hold every derived cursor for this
                # session without aborting healthy sessions in the same dream.
                # No provider call or extraction-attempt row was created.
                with core_db.transaction(conn):
                    record_coverage_integrity_failure(
                        conn,
                        session_id,
                        reason="source_stream_invalid",
                    )
                report.coverage_integrity_failures += 1
                log.exception(
                    "phase1.source_stream_failure session_id=%s "
                    "action=held_for_repair",
                    session_id,
                )
                continue

            # Both builders consume the same complete, validated lossless
            # stream before filtering it into scheduling tiers. Reaching this
            # point is the repair/rebuild acknowledgement: materialization
            # alone can be a zero-row no-op over an already-corrupt frontier.
            with core_db.transaction(conn):
                coverage_failure_cleared = clear_coverage_integrity_failure(
                    conn, session_id
                )
            if coverage_failure_cleared:
                log.info(
                    "coverage.integrity_recovered session_id=%s", session_id
                )

            report.chunks_seen += len(chunks)
            current_high_priority_chunk_ids.update(chunk.id for chunk in chunks)
            current_baseline_chunk_ids.update(
                chunk.id for chunk in baseline_candidates
            )
            deferred_baseline_embedding_ids.update(
                chunk.id for chunk in baseline_candidates
            )
            baseline_candidates_by_session[session_id] = baseline_candidates
            chunks_to_materialize = [*chunks, *baseline_candidates]

            with core_db.transaction(conn):
                if chunks_to_materialize:
                    persist_chunks(conn, chunks_to_materialize)
                    # Discovery/persistence is local and deliberately complete.
                    # Retrieval indexes and external embeddings remain tied to
                    # scheduled work; eagerly indexing the entire low-priority
                    # baseline would turn a bounded LLM tier into an unbounded
                    # side workload.
                    for chunk in chunks:
                        index_chunk_mentions(conn, chunk.id, chunk.text)
                        index_chunk_temporal_mentions(conn, chunk.id)
                # Legacy provenance classification is part of the same local
                # producer acknowledgement. A crash cannot leave a manifestless
                # row invisible between a current marker and terminal status.
                recovered_manifests = recover_legacy_chunk_source_manifests(
                    conn, session_id
                )
                newly_terminal = record_unrecoverable_chunk_losses(
                    conn, session_id
                )
                conn.execute(
                    "UPDATE sessions SET source_materialized_message_id=?, "
                    "source_materialization_config_version=? WHERE id=?",
                    (
                        materialization_target,
                        source_materialization_config_version(
                            min_chars=cfg.salience_min_chars
                        ),
                        session_id,
                    ),
                )
            if recovered_manifests:
                log.info(
                    "phase1.source_manifest_recovered session_id=%s chunks=%d",
                    session_id,
                    recovered_manifests,
                )
            if newly_terminal:
                log.warning(
                    "phase1.source_manifest_terminal_loss session_id=%s "
                    "chunks=%d action=excluded_from_future_budgets",
                    session_id,
                    newly_terminal,
                )
            if chunks:
                _kickoff_chunk_embed(chunks)

            for chunk in chunks:
                _check_deadline()
                outcome = _extract_phase1_chunk(chunk, tier="salience")
                if outcome in {"chunk_budget", "call_budget"}:
                    break

            # Prompt-salt replay cannot depend on live raw messages: retention
            # may have pruned them after storing the extraction chunks. Drain a
            # bounded durable backlog from the chunks themselves, excluding
            # anything already attempted this cycle or explicitly quarantined.
            # Current baseline classification comes from exact source content,
            # not the stored scheduling label: ``salience_min_chars`` can change
            # while the deterministic chunk ID (correctly) stays the same.
            if chunks_remaining > 0:
                current_baseline_ids = {
                    chunk.id for chunk in baseline_candidates
                }
                backlog = load_pending_persisted_chunks(
                    conn,
                    session_id,
                    prompt_version=cfg.prompt_version,
                    limit=chunks_remaining + len(current_baseline_ids),
                    max_attempts=cfg.chunk_extraction_max_attempts,
                    phase1_generation_key=phase1_generation_key,
                    exclude_ids=attempted_this_cycle,
                    excluded_salience_reasons=(BASELINE_SALIENCE_REASON,),
                )
                backlog = [
                    chunk for chunk in backlog
                    if chunk.id not in current_baseline_ids
                ][:chunks_remaining]
                report.chunks_seen += len(backlog)
                for chunk in backlog:
                    _check_deadline()
                    outcome = _extract_phase1_chunk(
                        chunk, tier="persisted_backlog"
                    )
                    if outcome in {"chunk_budget", "call_budget"}:
                        break

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
            digest_quarantined = digest_retry_is_quarantined(
                digest_retry_count,
                digested["digest_retry_config_version"] if digested else None,
                retry_key=digest_retry_key,
                max_attempts=cfg.digest_extraction_max_attempts,
            )
            if coverage_tail is not None and not caught_up and not digest_quarantined:
                _check_deadline()
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
                profile_quarantined = profile_retry_is_quarantined(
                    profile_retry_count,
                    (
                        digested["profile_retry_config_version"]
                        if digested else None
                    ),
                    retry_key=profile_retry_key,
                    max_attempts=cfg.profile_extraction_max_attempts,
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
                    _check_deadline()
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
                        _check_deadline()
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
                        _check_deadline()
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

            # dream_budget bounds only Phase-1 chunk extraction. Digest,
            # profile, and facts use independent lossless streams and their
            # own retry/output bounds; continue across later sessions so an
            # old extraction backlog cannot starve a newly closed session's
            # tail work. Do not infer exhaustion merely because the counter
            # landed on zero: after every session has materialized/classified
            # its candidates, the pending-aware probe below decides whether
            # any genuinely schedulable work remains.

        # Baseline is a genuinely lower global priority: every session's live
        # and persisted high-salience work was offered above before a short,
        # non-trigger turn may consume either shared Phase-1 ceiling. Candidate
        # lists were built newest-first from durable coverage and the rotated
        # session order prevents one continuously busy old session from owning
        # the global baseline allowance across dreams.
        baseline_blocked = False
        if chunks_remaining > 0 and baseline_remaining > 0:
            for baseline_session_id in target_sessions:
                _check_deadline()
                candidates = baseline_candidates_by_session.get(
                    baseline_session_id, []
                )
                if not candidates:
                    continue
                baseline_cap = min(chunks_remaining, baseline_remaining)
                selected = candidates[:baseline_cap]
                report.chunks_seen += len(selected)
                for chunk in selected:
                    _check_deadline()
                    outcome = _extract_phase1_chunk(chunk, tier="baseline")
                    if outcome == "attempted":
                        baseline_remaining -= 1
                        deferred_baseline_embedding_ids.discard(chunk.id)
                        with core_db.transaction(conn):
                            index_chunk_mentions(conn, chunk.id, chunk.text)
                            index_chunk_temporal_mentions(conn, chunk.id)
                        _kickoff_chunk_embed([chunk])
                    if outcome in {"chunk_budget", "call_budget"}:
                        baseline_blocked = True
                        break
                if (
                    baseline_blocked
                    or chunks_remaining <= 0
                    or baseline_remaining <= 0
                ):
                    break
        if chunks_remaining <= 0 and not report.budget_exhausted:
            # A configured zero baseline allowance deliberately disables that
            # tier, so baseline-only rows must not turn exact high-priority
            # completion into an exhausted report. Otherwise the shared chunk
            # ceiling blocks both tiers and any eligible durable row means a
            # subsequent cycle is genuinely owed. This indexed LIMIT 1 probe
            # performs no provider/model work.
            pending_session_scope = (
                tuple(target_sessions) if session_ids else None
            )
            current_candidate_ids = set(current_high_priority_chunk_ids)
            if cfg.dream_baseline_budget > 0:
                current_candidate_ids.update(current_baseline_chunk_ids)
            chunk_budget_pending = (
                has_pending_persisted_chunks(
                    conn,
                    prompt_version=cfg.prompt_version,
                    max_attempts=cfg.chunk_extraction_max_attempts,
                    phase1_generation_key=phase1_generation_key,
                    session_ids=pending_session_scope,
                    included_chunk_ids=current_candidate_ids,
                )
                if current_candidate_ids else False
            )
            if not chunk_budget_pending:
                # Persisted backlog keeps its historical scheduling label, but
                # the current builders own tier classification. A chunk that
                # is baseline *now* is excluded here even if INSERT OR IGNORE
                # retained an old high-priority label; conversely, a current
                # high candidate with an old baseline label was covered by the
                # direct-id probe above.
                chunk_budget_pending = has_pending_persisted_chunks(
                    conn,
                    prompt_version=cfg.prompt_version,
                    max_attempts=cfg.chunk_extraction_max_attempts,
                    phase1_generation_key=phase1_generation_key,
                    session_ids=pending_session_scope,
                    excluded_chunk_ids=current_baseline_chunk_ids,
                    excluded_salience_reasons=(BASELINE_SALIENCE_REASON,),
                )
            if chunk_budget_pending:
                report.budget_exhausted = True
                log.info(
                    "dream.budget_exhausted budget=%d", cfg.dream_budget
                )
        elif cfg.dream_baseline_budget > 0 and baseline_remaining <= 0:
            # The baseline allowance is independent of ``dream_budget``.  A
            # cycle can therefore leave perfectly actionable short-turn work
            # behind while still having ordinary chunk capacity available.
            # Report that bounded-work stop just like the other resumable
            # ceilings, but only when the tier is enabled and work really
            # remains.  A configured zero deliberately disables inference for
            # this tier and must not make every dream claim exhaustion.
            _check_deadline()
            baseline_pending = has_pending_persisted_chunks(
                conn,
                prompt_version=cfg.prompt_version,
                max_attempts=cfg.chunk_extraction_max_attempts,
                phase1_generation_key=phase1_generation_key,
                session_ids=(tuple(target_sessions) if session_ids else None),
                included_chunk_ids=current_baseline_chunk_ids,
            )
            if baseline_pending and not report.budget_exhausted:
                report.budget_exhausted = True
                log.info(
                    "dream.baseline_budget_exhausted budget=%d",
                    cfg.dream_baseline_budget,
                )

        _check_deadline()
        if embedding_client is not None:
            # Drain background-embedded batches before opening any write
            # transaction. A
            # per-batch future failure is logged and skipped — the post-loop
            # fetch_chunk_embeddings call below catches anything missed
            # (skipped sessions, future raises, miss_texts size mismatch).
            persisted_ids: set[str] = set()
            if embed_inflight:
                for request, future in embed_inflight:
                    _check_deadline()
                    try:
                        miss_vectors = future.result()
                    except Exception as exc:
                        log.error(
                            "embedding.background_failure batch_size=%d error=%s",
                            len(request.ids), type(exc).__name__,
                        )
                        continue
                    try:
                        resolved_model, resolved_dim = _embedding_identity(
                            embedding_client
                        )
                        pending = assemble_chunk_pending(
                            conn, request, miss_vectors,
                            exclude_ids=persisted_ids,
                            resolved_model=resolved_model,
                            resolved_dim=resolved_dim,
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

            pending_chunks = fetch_chunk_embeddings(
                conn,
                embedding_client,
                exclude_ids=deferred_baseline_embedding_ids,
            )
            if pending_chunks is not None:
                with core_db.transaction(conn):
                    report.chunks_embedded += persist_chunk_embeddings(conn, pending_chunks)
                report.chunks_embedded_from_cache += pending_chunks.cache_hits

            # Exact message occurrences have their own durable semantic tier.
            # Fetch/embedding is deliberately outside the write transaction;
            # only the idempotent mirror/index update takes the writer lock.
            for message_batch in message_embedding_id_batches(conn):
                _check_deadline()
                embedded, from_cache, abort_cycle = (
                    _persist_message_batch_with_failure_isolation(
                        conn, embedding_client, message_batch
                    )
                )
                report.messages_embedded += embedded
                report.messages_embedded_from_cache += from_cache
                if abort_cycle:
                    break

        _check_deadline()
        log.info("phase2.start")
        with core_db.transaction(conn):
            # Idea B write-side: route imperative markers into agent_inferred
            # rules BEFORE consolidate_profile stamps them consolidated (both
            # read consolidated_at IS NULL). Gated; no new LLM call. Additive —
            # markers still become profile entries too.
            if cfg.rules_extraction_enabled:
                from hymem import rules as rules_mod
                report.rules_extracted += rules_mod.route_markers_to_rules(
                    conn, cfg, llm=llm,
                    phase1_generation_key=phase1_generation_key,
                )
            phase2.consolidate_profile(
                conn, cfg, phase1_generation_key=phase1_generation_key
            )
            phase2.consolidate_insights(conn, cfg)
        profile_count = conn.execute(
            "SELECT COUNT(*) AS c FROM current_profile_entries"
        ).fetchone()["c"]
        log.info(
            "phase2.end profile_entries=%d insights=%d",
            profile_count,
            report.markers_extracted,
        )

        _check_deadline()
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

        # VACUUM cannot share the atomic BEGIN IMMEDIATE ownership proof used by
        # every semantic/retrieval-state publication. It can also renumber the
        # implicit rowids backing FTS/vector shadows. If a lease crosses its TTL
        # during VACUUM, a waiting successor could win immediately afterward,
        # correctly fence this owner's resync, and leave those shadows skewed.
        # Therefore automatic VACUUM is deferred out of dreaming entirely.
        # Operators may run VACUUM plus core_db.resync_rowid_shadows as one
        # quiesced maintenance operation; aggregation-enabled runs still repair
        # pre-existing proven vector skew in the fenced heal step below.
        _check_deadline()
        if cfg.vacuum_after_prune and pruned >= cfg.vacuum_min_pruned:
            log.warning(
                "retention.vacuum_deferred_lease_fence pruned=%d "
                "action=run_quiesced_vacuum_then_resync_rowid_shadows",
                pruned,
            )

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

        _check_deadline()
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
        # enabled. Manages its own transactions; the shipped default is on,
        # while callers can explicitly opt out for a zero-cost control arm.
        _check_deadline()
        if cfg.aggregation_nodes_enabled:
            assert aggregation_version is not None
            # Fail closed across process death: this commit precedes every
            # fallible aggregation operation. Only the clean-result branch
            # below clears the matching durable pending identity.
            aggregation_attempt_token = _aggregation_health_write(
                lambda: begin_aggregation_build(
                    conn, aggregation_version,
                    generation_binding=aggregation_generation,
                )
            )
            try:
                # Repair step for stores skewed by a pre-fix VACUUM (the
                # resync above only covers VACUUMs from now on): a proven
                # vec_episodes/rowid mismatch rebuilds all rowid shadows so
                # candidate blocking stops clustering on garbage neighborhoods.
                if deadline is None:
                    with core_db.transaction(conn):
                        shadows_healed = core_db.heal_rowid_shadows(conn)
                    if shadows_healed:
                        log.info("aggregate.pre_build_shadow_heal")
                else:
                    # Resync is optional maintenance composed of several FTS
                    # and vector-table writes.  Do not start it in a bounded
                    # dream: unlike normal semantic transactions it cannot be
                    # made safe if the deadline lands between those writes.
                    # A clean read-only probe licenses aggregation; a proven
                    # mismatch remains pending for an unbounded repair rather
                    # than building a tree from corrupt rowid shadows.
                    shadows_aligned = core_db.vec_episodes_aligned(conn)
                    _check_deadline()
                    if not shadows_aligned:
                        raise RuntimeError(
                            "rowid shadow repair requires an unbounded dream"
                        )
                _check_deadline()
                agg = build_aggregation_nodes(
                    conn, cfg, llm, embedding_client,
                    episode_ceiling_rowid=episode_ceiling,
                    generation_binding=aggregation_generation,
                    health_managed=True,
                    health_attempt_token=aggregation_attempt_token,
                )
                # A custom/injected builder may ignore provider timeouts.  Its
                # result is unusable once the shared absolute deadline lands;
                # check immediately on return, before report attribution or
                # either durable health acknowledgement.
                _check_deadline()
                if (
                    isinstance(agg.fusion_failures, bool)
                    or not isinstance(agg.fusion_failures, int)
                    or agg.fusion_failures < 0
                ):
                    raise ValueError("invalid aggregation fusion-failure count")
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
                aggregation_material_epoch_key = agg.material_epoch_key
            except Exception as exc:
                # If an injected builder returns control by raising only after
                # the deadline, leave the pre-build marker pending.  Recording
                # that as a handled build failure would itself be an illegal
                # post-deadline write and would misdescribe a timeout.
                _check_deadline()
                # A total exception previously left the historical fusion
                # counter at zero, indistinguishable from a clean empty build.
                # Keep the semantic exception counter and also emit one
                # nonzero failure unit through the established report field.
                report.aggregation_build_exceptions = 1
                report.aggregation_fusion_failures = max(
                    1, report.aggregation_fusion_failures
                )
                _aggregation_health_write(
                    lambda: record_aggregation_build_failure(
                        conn,
                        aggregation_version,
                        aggregation_generation_key,
                        aggregation_attempt_token,
                        caught_exceptions=1,
                    )
                )
                log.error(
                    "aggregate.build_failure error=%s",
                    type(exc).__name__,
                )
            else:
                _check_deadline()
                if report.aggregation_fusion_failures > 0:
                    _aggregation_health_write(
                        lambda: record_aggregation_build_failure(
                            conn,
                            aggregation_version,
                            aggregation_generation_key,
                            aggregation_attempt_token,
                            fusion_failures=report.aggregation_fusion_failures,
                            material_epoch_key=agg.material_epoch_key,
                        )
                    )
                else:
                    _aggregation_health_write(
                        lambda: complete_aggregation_build(
                            conn, aggregation_version,
                            aggregation_generation_key,
                            aggregation_attempt_token,
                            expected_node_count=agg.nodes,
                            material_epoch_key=agg.material_epoch_key,
                            embedding_client=embedding_client,
                        )
                    )

        _check_deadline()
        after_retracted = conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph WHERE status = 'retracted'"
        ).fetchone()["c"]
        log.info("phase3.end retracted=%d", after_retracted - before_retracted)

        _check_deadline()
        with core_db.transaction(conn):
            conn.execute(
                """
                UPDATE dream_runs
                SET ended_at = CURRENT_TIMESTAMP,
                sessions_processed = ?,
                chunks_seen = ?,
                chunks_processed = ?,
                chunk_extraction_completion_calls = ?,
                chunk_extraction_provider_attempts = ?,
                extraction_provider_attempt_budget_exhausted = ?,
                coverage_integrity_failures = ?,
                chunks_embedded = ?,
                edges_embedded = ?,
                triples_extracted = ?,
                markers_extracted = ?,
                aggregation_nodes_built = ?,
                aggregation_nodes_reused = ?,
                aggregation_fusion_failures = ?,
                aggregation_build_exceptions = ?,
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
                aggregation_material_epoch_key = ?,
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
                    report.chunk_extraction_completion_calls,
                    report.chunk_extraction_provider_attempts,
                    int(report.extraction_provider_attempt_budget_exhausted),
                    report.coverage_integrity_failures,
                    report.chunks_embedded,
                    report.edges_embedded,
                    report.triples_extracted,
                    report.markers_extracted,
                    report.aggregation_nodes_built,
                    report.aggregation_nodes_reused,
                    report.aggregation_fusion_failures,
                    report.aggregation_build_exceptions,
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
                    aggregation_material_epoch_key,
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
            "dream.end run_id=%d sessions=%d chunks_processed=%d/%d chunk_extraction_failures=%d extraction_completion_calls=%d extraction_provider_attempts=%d extraction_provider_attempt_budget_exhausted=%s coverage_integrity_failures=%d triples=%d markers=%d chunks_from_cache=%d messages_embedded=%d messages_from_cache=%d edges_from_cache=%d agg_nodes=%d agg_reused=%d agg_failures=%d agg_exceptions=%d agg_input=%d agg_level0_missed=%s agg_leaf_changed=%s agg_predicted=%s agg_keying_residual=%s agg_facts_rekey=%s agg_rebuilt_l0=%s agg_rebuilt_rollup=%s agg_rebuilt_root=%s agg_blocking=%s digest_failures=%d digest_quarantined=%d episodes_created=%d facts=%d fact_failures=%d profile_items=%d profile_failures=%d budget_exhausted=%s",
            run_id,
            report.sessions_processed,
            report.chunks_processed,
            report.chunks_seen,
            report.chunk_extraction_failures,
            report.chunk_extraction_completion_calls,
            report.chunk_extraction_provider_attempts,
            report.extraction_provider_attempt_budget_exhausted,
            report.coverage_integrity_failures,
            report.triples_extracted,
            report.markers_extracted,
            report.chunks_embedded_from_cache,
            report.messages_embedded,
            report.messages_embedded_from_cache,
            report.edges_embedded_from_cache,
            report.aggregation_nodes_built,
            report.aggregation_nodes_reused,
            report.aggregation_fusion_failures,
            report.aggregation_build_exceptions,
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
    except core_db.LeaseOwnershipLost:
        # Lease loss is an ownership transition, not a provider-quality or
        # extraction failure. Semantic transactions already rolled back; only
        # this run's lifecycle row may be terminalized without the fence.
        _finish_run_housekeeping(run_id, "lease_lost")
        log.error("dream.lease_lost run_id=%d", run_id)
        raise
    except DeadlineExceeded:
        # Permissible post-deadline housekeeping: terminalize this run and
        # release its lease. Prohibited work (semantic facts/chunks/cursors,
        # embeddings, retry/quarantine state) has already been stopped by the
        # provider/item/transaction checks above.
        _finish_run_housekeeping(run_id, "deadline_exceeded")
        raise
    except Exception as exc:
        # Provider/transport exception prose routinely contains request URLs,
        # opaque deployment paths, response bodies, and credentials.  Durable
        # run state records only a closed failure class, never ``str(exc)``.
        safe_types = {
            "ConnectionError", "DataError", "DatabaseError", "IntegrityError",
            "InterfaceError", "MemoryError", "NotSupportedError", "OSError",
            "OperationalError", "ProgrammingError", "RuntimeError",
            "TimeoutError", "TypeError", "ValueError",
        }
        exception_type = type(exc).__name__
        if exception_type not in safe_types:
            exception_type = "ExternalError"
        msg = f"execution_failure:{exception_type}"
        with contextlib.suppress(sqlite3.Error):
            conn.execute(
                "UPDATE dream_runs SET ended_at = CURRENT_TIMESTAMP, error = ? WHERE id = ?",
                (msg, run_id),
            )
        raise
    finally:
        active_error = sys.exception()
        cleanup_error: BaseException | None = None
        try:
            if embed_executor is not None:
                # The executor performs network fetches only; result assembly
                # and persistence return to this fenced calling context. Keep
                # the heartbeat alive while an in-flight fetch winds down.
                embed_executor.shutdown(wait=True)
        except BaseException as exc:
            cleanup_error = exc
        try:
            assert lease_heartbeat is not None
            lease_heartbeat.stop()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        try:
            assert lease_fence_token is not None
            core_db.deactivate_transaction_lease_fence(lease_fence_token)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        _release_lock(conn, holder)
        if cleanup_error is not None:
            if active_error is None:
                raise cleanup_error
            log.error(
                "dream.cleanup_failure_preserving_primary error=%s",
                type(cleanup_error).__name__,
            )


_LOCK_TTL_SECONDS = 120

# How often, at most, a live dream refreshes its lease. Kept well under
# _LOCK_TTL_SECONDS so even a single ultra-heavy session (hundreds of chunks,
# many minutes) heartbeats several times before the lock could look stale.
_LOCK_REFRESH_INTERVAL_SECONDS = 30


def _new_lease_token() -> str:
    """Return a process-labelled but invocation-unique opaque lease token."""

    return (
        f"{socket.gethostname()}:{os.getpid()}:"
        f"{secrets.token_hex(16)}"
    )


def _acquire_lock(conn: sqlite3.Connection, holder: str) -> bool:
    """Atomically acquire an absent or provably stale dreaming lease.

    One SQLite UPSERT is the compare-and-swap boundary. Independent processes
    cannot both observe/delete a stale row and then race separate INSERTs; the
    conflict update succeeds for exactly one writer only while the row visible
    to that statement is still stale.
    """

    cursor = conn.execute(
        """
        INSERT INTO run_lock(name, acquired_at, holder)
        VALUES ('dreaming', CURRENT_TIMESTAMP, ?)
        ON CONFLICT(name) DO UPDATE SET
            acquired_at = excluded.acquired_at,
            holder = excluded.holder
        WHERE run_lock.acquired_at < datetime('now', ?)
        """,
        (holder, f"-{_LOCK_TTL_SECONDS} seconds"),
    )
    acquired = cursor.rowcount == 1
    if acquired:
        log.debug("dream.lease_acquired")
    return acquired


def _release_lock(conn: sqlite3.Connection, holder: str) -> bool:
    """Conditionally release only this exact invocation's token."""

    try:
        cursor = conn.execute(
            "DELETE FROM run_lock WHERE name = 'dreaming' AND holder = ?",
            (holder,),
        )
    except sqlite3.Error as exc:
        log.warning("dream.lease_release_failure error=%s", type(exc).__name__)
        return False
    return cursor.rowcount == 1


def _refresh_lock(conn: sqlite3.Connection, holder: str) -> None:
    """Heartbeat the dreaming lease so a live (slow) dream keeps its lock fresh.

    Pushes ``acquired_at`` forward to now so a genuinely slow dream never
    crosses the ``_LOCK_TTL_SECONDS`` staleness line and gets taken over by a
    concurrent trigger. A crashed holder simply stops calling this, so its
    lock still ages past the TTL and is reclaimed.

    This is an autocommit statement so the new timestamp is immediately visible
    across processes. A zero-row update is authoritative ownership loss and
    raises a BaseException-style control signal; it must never be swallowed as
    a best-effort refresh. SQLite operational failures propagate to the caller.
    """
    cursor = conn.execute(
        "UPDATE run_lock SET acquired_at = CURRENT_TIMESTAMP"
        " WHERE name = 'dreaming' AND holder = ?",
        (holder,),
    )
    if cursor.rowcount != 1:
        raise core_db.LeaseOwnershipLost("dreaming lease ownership lost")


def _all_sessions(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT id FROM sessions ORDER BY started_at, id"
    ).fetchall()
    return [r["id"] for r in rows]
