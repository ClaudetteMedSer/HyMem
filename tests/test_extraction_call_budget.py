from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hymem import HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.contract import extraction_cache_key


def _quiet_cfg(cfg: HyMemConfig, *, provider_attempt_budget: int) -> HyMemConfig:
    return replace(
        cfg,
        dream_budget=50,
        dream_baseline_budget=0,
        dream_extraction_provider_attempt_budget=provider_attempt_budget,
        salience_min_chars=1,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
    )


def _empty_extraction_llm() -> StubLLMClient:
    return StubLLMClient(
        fixtures={
            "Return the JSON object now": json.dumps({
                "episodes": [],
                "summary": "tail complete",
                "procedures": [],
            }),
        },
        default=json.dumps({
            "triples": [],
            "markers": [],
            "complete": True,
        }),
    )


def _seed_closed_sessions(hy: HyMem, *session_ids: str) -> None:
    for session_id in session_ids:
        hy.open_session(session_id)
        hy.log_message(
            session_id,
            "user",
            f"Substantive memory input for {session_id}.",
        )
        hy.close_session(session_id)


def test_cycle_provider_attempt_ceiling_preserves_atomicity_and_tail_fairness(cfg):
    llm = _empty_extraction_llm()
    hy = HyMem(_quiet_cfg(cfg, provider_attempt_budget=2), llm=llm)
    try:
        _seed_closed_sessions(hy, "a-first", "z-later")

        first = hy.dream()

        # One clean empty costs exactly initial + verification. The next chunk
        # is not started, failed, or quarantined, while both session digests run.
        assert first.chunk_extraction_completion_calls == 2
        assert first.chunk_extraction_provider_attempts == 2
        assert first.extraction_provider_attempt_budget_exhausted is True
        assert first.budget_exhausted is True
        digest_calls = [
            call for call in llm.calls
            if "Return the JSON object now" in call.user
        ]
        assert len(digest_calls) == 2

        chunks = hy.conn.execute(
            "SELECT id,session_id FROM chunks WHERE chunk_kind='extraction' "
            "ORDER BY session_id"
        ).fetchall()
        assert len(chunks) == 2
        processed = {
            row[0] for row in hy.conn.execute(
                "SELECT chunk_id FROM processed_chunks WHERE prompt_version=?",
                (extraction_cache_key(hy.config.prompt_version),),
            )
        }
        assert len(processed) == 1
        untouched = next(row["id"] for row in chunks if row["id"] not in processed)
        assert hy.conn.execute(
            "SELECT 1 FROM chunk_extraction_attempts WHERE chunk_id=?",
            (untouched,),
        ).fetchone() is None
        assert hy.dream_status()["pending_chunks"] == 1

        run = hy.conn.execute(
            "SELECT chunk_extraction_completion_calls,"
            "chunk_extraction_provider_attempts,"
            "extraction_provider_attempt_budget_exhausted "
            "FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert tuple(run) == (2, 2, 1)

        # The untouched chunk resumes on the next cycle. Landing exactly on the
        # threshold is not reported as blocked when no actionable work remains.
        second = hy.dream()
        assert second.chunk_extraction_completion_calls == 2
        assert second.chunk_extraction_provider_attempts == 2
        assert second.extraction_provider_attempt_budget_exhausted is False
        assert hy.dream_status()["pending_chunks"] == 0
    finally:
        hy.close()


def test_soft_ceiling_may_finish_one_inflight_chunk(cfg):
    llm = _empty_extraction_llm()
    hy = HyMem(_quiet_cfg(cfg, provider_attempt_budget=1), llm=llm)
    try:
        _seed_closed_sessions(hy, "a-first", "z-later")
        report = hy.dream()
        # The initial call starts below the ceiling; its mandatory verification
        # finishes atomically and produces the documented one-chunk overshoot.
        assert report.chunk_extraction_completion_calls == 2
        assert report.chunk_extraction_provider_attempts == 2
        assert report.extraction_provider_attempt_budget_exhausted is True
        assert hy.dream_status()["pending_chunks"] == 1
    finally:
        hy.close()


def test_zero_provider_attempt_budget_means_unlimited(cfg):
    llm = _empty_extraction_llm()
    hy = HyMem(_quiet_cfg(cfg, provider_attempt_budget=0), llm=llm)
    try:
        _seed_closed_sessions(hy, "a-first", "z-later")
        report = hy.dream()
        assert report.chunk_extraction_completion_calls == 4
        assert report.chunk_extraction_provider_attempts == 4
        assert report.extraction_provider_attempt_budget_exhausted is False
        assert hy.dream_status()["pending_chunks"] == 0
        assert hy.dream_status()["extraction_provider_attempt_budget"] == 0
    finally:
        hy.close()


@pytest.mark.parametrize("value", [-1, 1.5, True, "200"])
def test_provider_attempt_budget_rejects_invalid_values(tmp_path, value):
    with pytest.raises(ValueError, match="non-negative integer"):
        HyMemConfig(
            root=tmp_path,
            dream_extraction_provider_attempt_budget=value,
        )


class _InternallyRetryingEmptyLLM:
    """Three provider requests occur inside every completion invocation."""

    def __init__(self):
        self.request_attempts = 0
        self.calls = []

    def complete(self, request):
        self.calls.append(request)
        self.request_attempts += 3
        if "Return the JSON object now" in request.user:
            return json.dumps({
                "episodes": [], "summary": "tail", "procedures": [],
            })
        return json.dumps({
            "triples": [], "markers": [], "complete": True,
        })


def test_cycle_ceiling_counts_delegate_request_attempt_deltas(cfg):
    llm = _InternallyRetryingEmptyLLM()
    hy = HyMem(_quiet_cfg(cfg, provider_attempt_budget=3), llm=llm)
    try:
        _seed_closed_sessions(hy, "a-first", "z-later")
        report = hy.dream()
        # One logical empty verification pair contains six measured provider
        # requests. Digest attempts increment the same delegate metric but are
        # outside the narrow Phase-1 wrapper and therefore excluded.
        assert report.chunk_extraction_completion_calls == 2
        assert report.chunk_extraction_provider_attempts == 6
        assert report.extraction_provider_attempt_budget_exhausted is True
        assert llm.request_attempts > report.chunk_extraction_provider_attempts
        assert hy.dream_status()["pending_chunks"] == 1
    finally:
        hy.close()


class _InternallyRetryingFailureLLM:
    def __init__(self):
        self.request_attempts = 0

    def complete(self, _request):
        self.request_attempts += 3
        raise RuntimeError("provider retries exhausted")


def test_runner_accounts_every_attempt_when_provider_raises(cfg):
    llm = _InternallyRetryingFailureLLM()
    hy = HyMem(_quiet_cfg(cfg, provider_attempt_budget=200), llm=llm)
    try:
        _seed_closed_sessions(hy, "failing")
        report = hy.dream()
        # Phase-1 catches the provider exception and persists one failed chunk
        # attempt; request-level cost attribution still includes all retries.
        assert report.chunk_extraction_failures == 1
        assert report.chunk_extraction_completion_calls == 1
        assert report.chunk_extraction_provider_attempts == 3
        assert hy.conn.execute(
            "SELECT attempts FROM chunk_extraction_attempts"
        ).fetchone()[0] == 1
    finally:
        hy.close()


class _StaleCounterFailureLLM:
    """Legacy client whose cumulative counter does not advance on failure."""

    request_attempts = 0

    def complete(self, _request):
        raise RuntimeError("provider failed before legacy telemetry updated")


def test_raised_legacy_call_with_unchanged_counter_is_charged_once(cfg):
    llm = _StaleCounterFailureLLM()
    hy = HyMem(_quiet_cfg(cfg, provider_attempt_budget=200), llm=llm)
    try:
        _seed_closed_sessions(hy, "stale-counter-failure")
        report = hy.dream()

        assert report.chunk_extraction_failures == 1
        assert report.chunk_extraction_completion_calls == 1
        assert report.chunk_extraction_provider_attempts == 1
    finally:
        hy.close()


def test_baseline_tier_uses_shared_provider_attempt_accounting(cfg):
    llm = _empty_extraction_llm()
    quiet = replace(
        _quiet_cfg(cfg, provider_attempt_budget=200),
        dream_baseline_budget=1,
        salience_min_chars=10_000,
    )
    hy = HyMem(quiet, llm=llm)
    try:
        _seed_closed_sessions(hy, "baseline-only")
        report = hy.dream()
        assert report.chunk_extraction_completion_calls == 2
        assert report.chunk_extraction_provider_attempts == 2
        assert hy.dream_status()["pending_chunks"] == 0
    finally:
        hy.close()


def test_persisted_backlog_tier_uses_shared_provider_attempt_accounting(cfg):
    llm = _empty_extraction_llm()
    quiet = replace(
        _quiet_cfg(cfg, provider_attempt_budget=200),
        salience_min_chars=10_000,
        dream_baseline_budget=0,
    )
    hy = HyMem(quiet, llm=llm)
    try:
        session_id = "persisted-only"
        hy.open_session(session_id)
        message_id = hy.log_message(session_id, "user", "short source")
        hy.close_session(session_id)
        chunk = Chunk(
            id="persisted-only-chunk",
            session_id=session_id,
            start_message_id=message_id,
            end_message_id=message_id,
            salience_reason="persisted_test",
            text="user: short source",
            source_message_ids=(message_id,),
        )
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, session_id)
            persist_chunks(hy.conn, [chunk])

        report = hy.dream()
        assert report.chunk_extraction_completion_calls == 2
        assert report.chunk_extraction_provider_attempts == 2
        assert hy.conn.execute(
            "SELECT 1 FROM processed_chunks WHERE chunk_id=? AND prompt_version=?",
            (chunk.id, extraction_cache_key(hy.config.prompt_version)),
        ).fetchone() is not None
    finally:
        hy.close()
