from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem.dreaming.chunks import (
    BASELINE_SALIENCE_REASON,
    extract_baseline_chunks,
    extract_high_salience_chunks,
)
from hymem.dreaming.retention import prune_chunks
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.contract import extraction_cache_key


def _llm() -> StubLLMClient:
    return StubLLMClient(
        fixtures={
            "Return the JSON object now": json.dumps({
                "episodes": [],
                "summary": "complete tail",
                "procedures": [],
            }),
        },
        default=json.dumps({
            "triples": [],
            "markers": [],
            "complete": True,
        }),
    )


def _cfg(
    cfg: HyMemConfig,
    *,
    dream_budget: int = 50,
    baseline_budget: int = 10,
    provider_budget: int = 0,
    message_retention_days: int = 0,
    max_chunks: int = 50_000,
    retention_days: int = 90,
) -> HyMemConfig:
    return replace(
        cfg,
        dream_budget=dream_budget,
        dream_baseline_budget=baseline_budget,
        dream_extraction_provider_attempt_budget=provider_budget,
        salience_min_chars=30,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
        message_retention_days=message_retention_days,
        max_chunks=max_chunks,
        retention_days=retention_days,
    )


def _seed_short_facts(hy: HyMem, session_id: str, *facts: str) -> None:
    hy.open_session(session_id)
    for fact in facts:
        hy.log_message(session_id, "user", fact)
    hy.close_session(session_id)


def _baseline_rows(hy: HyMem):
    return hy.conn.execute(
        "SELECT id,start_message_id,end_message_id,source_manifest_count "
        "FROM chunks WHERE chunk_kind='extraction' "
        "AND salience_reason=? ORDER BY end_message_id",
        (BASELINE_SALIENCE_REASON,),
    ).fetchall()


def _processed_baseline_ids(hy: HyMem) -> set[str]:
    return {
        row[0]
        for row in hy.conn.execute(
            "SELECT pc.chunk_id FROM processed_chunks pc JOIN chunks c "
            "ON c.id=pc.chunk_id WHERE pc.prompt_version=? "
            "AND c.salience_reason=?",
            (
                extraction_cache_key(hy.config.prompt_version),
                BASELINE_SALIENCE_REASON,
            ),
        )
    }


def _processed_baseline_sessions(hy: HyMem) -> set[str]:
    return {
        row[0]
        for row in hy.conn.execute(
            "SELECT DISTINCT c.session_id FROM processed_chunks pc "
            "JOIN chunks c ON c.id=pc.chunk_id "
            "WHERE pc.prompt_version=? AND c.salience_reason=?",
            (
                extraction_cache_key(hy.config.prompt_version),
                BASELINE_SALIENCE_REASON,
            ),
        )
    }


def test_baseline_budget_is_real_and_durable_backlog_does_not_bypass_it(cfg):
    hy = HyMem(_cfg(cfg, baseline_budget=1), llm=_llm())
    try:
        _seed_short_facts(
            hy,
            "bounded-baseline",
            "My dog is Max.",
            "My cat is Bea.",
            "My owl is Pip.",
        )

        first = hy.dream()
        rows = _baseline_rows(hy)

        # Discovery/manifests are complete before inference, but only the
        # newest candidate consumes the shared Phase-1 path this cycle.
        assert len(rows) == 3
        assert all(row["source_manifest_count"] == 1 for row in rows)
        assert first.chunk_extraction_completion_calls == 2
        assert len(_processed_baseline_ids(hy)) == 1
        newest_id = rows[-1]["id"]
        assert _processed_baseline_ids(hy) == {newest_id}
        assert hy.dream_status()["pending_chunks"] == 2
        assert first.budget_exhausted is True

        second = hy.dream()
        assert second.chunk_extraction_completion_calls == 2
        assert len(_processed_baseline_ids(hy)) == 2
        assert hy.dream_status()["pending_chunks"] == 1
        assert second.budget_exhausted is True

        third = hy.dream()
        assert third.chunk_extraction_completion_calls == 2
        assert len(_processed_baseline_ids(hy)) == 3
        assert hy.dream_status()["pending_chunks"] == 0
        assert third.budget_exhausted is False
    finally:
        hy.close()


def test_baseline_budget_is_global_across_sessions(cfg):
    hy = HyMem(_cfg(cfg, baseline_budget=1), llm=_llm())
    try:
        _seed_short_facts(hy, "first-baseline", "My dog is Max.")
        _seed_short_facts(hy, "second-baseline", "My cat is Bea.")

        first = hy.dream()

        assert first.chunk_extraction_completion_calls == 2
        assert len(_baseline_rows(hy)) == 2
        assert len(_processed_baseline_ids(hy)) == 1
        assert hy.dream_status()["pending_chunks"] == 1

        second = hy.dream()
        assert second.chunk_extraction_completion_calls == 2
        assert len(_processed_baseline_ids(hy)) == 2
        assert hy.dream_status()["pending_chunks"] == 0
    finally:
        hy.close()


def test_consumed_baseline_budget_reports_retryable_attempt_as_pending(cfg):
    llm = StubLLMClient(
        fixtures={
            "Return the JSON object now": json.dumps({
                "episodes": [],
                "summary": "complete tail",
                "procedures": [],
            }),
        },
        default="{malformed",
    )
    hy = HyMem(_cfg(cfg, baseline_budget=1), llm=llm)
    try:
        _seed_short_facts(hy, "retryable-baseline", "My dog is Max.")

        report = hy.dream()

        assert report.chunk_extraction_failures == 1
        assert report.budget_exhausted is True
        assert hy.dream_status()["pending_chunks"] == 1
        assert _processed_baseline_ids(hy) == set()
    finally:
        hy.close()


def test_session_rotation_prevents_busy_old_session_starvation(cfg):
    hy = HyMem(_cfg(cfg, baseline_budget=1), llm=_llm())
    try:
        _seed_short_facts(hy, "a-old-session", "My dog is Max.")
        _seed_short_facts(hy, "z-later-session", "My cat is Bea.")

        hy.dream()
        assert _processed_baseline_sessions(hy) == {"a-old-session"}

        # Keep the earliest session busy. A fixed oldest-first order would use
        # the sole global slot here again and starve z-later-session forever.
        hy.log_message("a-old-session", "user", "My owl is Pip.")
        hy.close_session("a-old-session")
        second = hy.dream()

        assert second.chunk_extraction_completion_calls == 2
        assert _processed_baseline_sessions(hy) == {
            "a-old-session",
            "z-later-session",
        }
        assert hy.dream_status()["pending_chunks"] == 1
    finally:
        hy.close()


@pytest.mark.parametrize("value", [-1, 1.5, True, "10"])
def test_baseline_budget_rejects_invalid_values(tmp_path, value):
    with pytest.raises(ValueError, match="dream_baseline_budget"):
        HyMemConfig(root=tmp_path, dream_baseline_budget=value)


def test_chunk_budget_blocks_baseline_without_attempt_then_next_dream_resumes(cfg):
    hy = HyMem(
        _cfg(cfg, dream_budget=1, baseline_budget=1),
        llm=_llm(),
    )
    try:
        hy.open_session("chunk-budget")
        hy.log_message(
            "chunk-budget",
            "user",
            "This deliberately long high-priority source consumes the chunk budget.",
        )
        hy.log_message("chunk-budget", "user", "My dog is Max.")
        hy.close_session("chunk-budget")

        first = hy.dream()
        baseline = _baseline_rows(hy)
        assert len(baseline) == 1
        baseline_id = baseline[0]["id"]
        assert first.chunk_extraction_completion_calls == 2
        assert _processed_baseline_ids(hy) == set()
        assert hy.conn.execute(
            "SELECT 1 FROM chunk_extraction_attempts WHERE chunk_id=?",
            (baseline_id,),
        ).fetchone() is None

        second = hy.dream()
        assert second.chunk_extraction_completion_calls == 2
        assert _processed_baseline_ids(hy) == {baseline_id}
    finally:
        hy.close()


def test_all_high_salience_work_precedes_any_baseline_work(cfg):
    hy = HyMem(
        _cfg(cfg, dream_budget=1, baseline_budget=1),
        llm=_llm(),
    )
    try:
        _seed_short_facts(hy, "a-baseline-first", "My dog is Max.")
        hy.open_session("z-high-later")
        hy.log_message(
            "z-high-later",
            "user",
            "This later session contains a deliberately long high-priority source.",
        )
        hy.close_session("z-high-later")

        first = hy.dream()

        assert first.chunk_extraction_completion_calls == 2
        processed = hy.conn.execute(
            "SELECT c.session_id,c.salience_reason FROM processed_chunks pc "
            "JOIN chunks c ON c.id=pc.chunk_id WHERE pc.prompt_version=?",
            (extraction_cache_key(hy.config.prompt_version),),
        ).fetchall()
        assert [tuple(row) for row in processed] == [
            ("z-high-later", "long_user_turn")
        ]
        baseline_id = _baseline_rows(hy)[0]["id"]
        assert hy.conn.execute(
            "SELECT 1 FROM chunk_extraction_attempts WHERE chunk_id=?",
            (baseline_id,),
        ).fetchone() is None
    finally:
        hy.close()


def test_unattempted_baseline_rebuilds_after_raw_and_chunk_pruning(cfg):
    embeddings = StubEmbeddingClient()
    hy = HyMem(
        _cfg(
            cfg,
            baseline_budget=1,
            provider_budget=2,
            message_retention_days=1,
            max_chunks=1,
            retention_days=1,
        ),
        llm=_llm(),
        embedding_client=embeddings,
    )
    try:
        sid = "retained-baseline"
        hy.open_session(sid)
        hy.log_message(
            sid,
            "user",
            "This long high-priority turn deliberately consumes the provider ceiling.",
            created_at="2020-01-01T00:00:00Z",
        )
        assistant_id = hy.log_message(
            sid,
            "assistant",
            "Your service deploys to Oslo.",
            created_at="2020-01-01T00:00:01Z",
        )
        confirmation_id = hy.log_message(
            sid,
            "user",
            "Yes.",
            created_at="2020-01-01T00:00:02Z",
        )
        hy.close_session(sid)

        first = hy.dream()
        baseline = _baseline_rows(hy)
        assert len(baseline) == 1
        baseline_id = baseline[0]["id"]
        high_id = hy.conn.execute(
            "SELECT id FROM chunks WHERE chunk_kind='extraction' "
            "AND salience_reason <> ?",
            (BASELINE_SALIENCE_REASON,),
        ).fetchone()[0]
        assert baseline[0]["source_manifest_count"] == 2
        assert first.extraction_provider_attempt_budget_exhausted is True
        assert _processed_baseline_ids(hy) == set()
        assert hy.conn.execute(
            "SELECT 1 FROM chunk_extraction_attempts WHERE chunk_id=?",
            (baseline_id,),
        ).fetchone() is None
        assert hy.conn.execute(
            "SELECT 1 FROM chunk_embeddings WHERE chunk_id=?",
            (baseline_id,),
        ).fetchone() is None

        # The fresh pending chunk survives same-cycle max_chunks pressure, and
        # raw rows are pruned only after its exact source manifest exists.
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=?", (sid,)
        ).fetchone()[0] == 0
        assert [
            chunk.id for chunk in extract_high_salience_chunks(
                hy.conn, sid, min_chars=hy.config.salience_min_chars
            )
        ] == [high_id]
        assert [
            chunk.id for chunk in extract_baseline_chunks(
                hy.conn,
                sid,
                prompt_version=hy.config.prompt_version,
                limit=None,
                min_chars=hy.config.salience_min_chars,
            )
        ] == [baseline_id]
        assert [
            row[0]
            for row in hy.conn.execute(
                "SELECT source_message_id FROM chunk_message_sources "
                "WHERE chunk_id=? ORDER BY ordinal",
                (baseline_id,),
            )
        ] == [assistant_id, confirmation_id]

        # Once the untouched cache row ages past retention it may be evicted,
        # but the independent coverage stream is protected and is sufficient
        # to reconstruct the exact ID/text/manifest without raw messages.
        hy.conn.execute(
            "UPDATE chunks SET created_at=datetime('now', '-200 days') "
            "WHERE id=?",
            (baseline_id,),
        )
        assert prune_chunks(hy.conn, hy.config) == 1
        assert hy.conn.execute(
            "SELECT 1 FROM chunks WHERE id=?", (baseline_id,)
        ).fetchone() is None
        assert hy.dream_status()["pending_source_materialization"] == 1

        second = hy.dream()
        rebuilt = _baseline_rows(hy)
        assert [row["id"] for row in rebuilt] == [baseline_id]
        assert rebuilt[0]["source_manifest_count"] == 2
        assert second.chunk_extraction_completion_calls >= 2
        assert _processed_baseline_ids(hy) == {baseline_id}
        assert hy.conn.execute(
            "SELECT 1 FROM chunk_embeddings WHERE chunk_id=?",
            (baseline_id,),
        ).fetchone() is not None
        assert [
            row[0]
            for row in hy.conn.execute(
                "SELECT source_message_id FROM chunk_message_sources "
                "WHERE chunk_id=? ORDER BY ordinal",
                (baseline_id,),
            )
        ] == [assistant_id, confirmation_id]
        assert hy.dream_status()["pending_source_materialization"] == 0
    finally:
        hy.close()


def test_zero_baseline_budget_persists_without_generic_backlog_execution(cfg):
    hy = HyMem(_cfg(cfg, baseline_budget=0), llm=_llm())
    try:
        _seed_short_facts(hy, "disabled-baseline", "My dog is Max.")

        report = hy.dream()

        rows = _baseline_rows(hy)
        assert len(rows) == 1
        assert rows[0]["source_manifest_count"] == 1
        assert report.chunk_extraction_completion_calls == 0
        assert report.budget_exhausted is False
        assert _processed_baseline_ids(hy) == set()
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM chunk_extraction_attempts"
        ).fetchone()[0] == 0
        assert hy.dream_status()["pending_chunks"] == 1
    finally:
        hy.close()


def test_chunk_pressure_preserves_failed_attempt_state(cfg):
    hy = HyMem(
        _cfg(
            cfg,
            baseline_budget=0,
            max_chunks=0,
            retention_days=1,
        ),
        llm=_llm(),
    )
    try:
        _seed_short_facts(hy, "retry-state", "My dog is Max.")
        hy.dream()
        chunk_id = _baseline_rows(hy)[0]["id"]
        hy.conn.execute(
            "INSERT INTO chunk_extraction_attempts("
            "chunk_id,prompt_version,attempts,last_failure_reason) "
            "VALUES (?,?,1,'parse_failure')",
            (chunk_id, extraction_cache_key(hy.config.prompt_version)),
        )
        hy.conn.execute(
            "UPDATE chunks SET created_at=datetime('now', '-200 days') "
            "WHERE id=?",
            (chunk_id,),
        )

        assert prune_chunks(hy.conn, hy.config) == 0
        assert hy.conn.execute(
            "SELECT attempts FROM chunk_extraction_attempts WHERE chunk_id=?",
            (chunk_id,),
        ).fetchone()[0] == 1
        assert hy.conn.execute(
            "SELECT 1 FROM chunks WHERE id=?", (chunk_id,)
        ).fetchone() is not None
    finally:
        hy.close()


def test_prompt_bump_replays_baseline_from_coverage_after_raw_pruning(cfg):
    hy = HyMem(
        _cfg(cfg, message_retention_days=1),
        llm=_llm(),
    )
    try:
        sid = "baseline-prompt-replay"
        hy.open_session(sid)
        message_id = hy.log_message(
            sid,
            "user",
            "My dog is Max.",
            created_at="2020-01-01T00:00:00Z",
        )
        hy.close_session(sid)

        first = hy.dream()
        baseline_id = _baseline_rows(hy)[0]["id"]
        assert first.chunk_extraction_completion_calls == 2
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=?", (sid,)
        ).fetchone()[0] == 0

        hy.config = replace(hy.config, prompt_version="v21")
        second = hy.dream()

        assert second.chunk_extraction_completion_calls == 2
        assert hy.conn.execute(
            "SELECT 1 FROM processed_chunks "
                "WHERE chunk_id=? AND prompt_version=?",
                (baseline_id, extraction_cache_key("v21")),
        ).fetchone() is not None
        assert [
            row[0]
            for row in hy.conn.execute(
                "SELECT source_message_id FROM chunk_message_sources "
                "WHERE chunk_id=? ORDER BY ordinal",
                (baseline_id,),
            )
        ] == [message_id]
    finally:
        hy.close()


def test_threshold_increase_cannot_route_baseline_through_generic_backlog(cfg):
    config = replace(
        _cfg(cfg, baseline_budget=1),
        salience_min_chars=10,
    )
    hy = HyMem(config, llm=_llm())
    try:
        _seed_short_facts(hy, "threshold-drift", "My dog is Max.")
        first = hy.dream()
        chunk = hy.conn.execute(
            "SELECT id,salience_reason FROM chunks "
            "WHERE chunk_kind='extraction'"
        ).fetchone()
        assert first.chunk_extraction_completion_calls == 2
        assert chunk["salience_reason"] == "long_user_turn"

        # The immutable source identity keeps its original scheduling label,
        # but current threshold classification is baseline. With baseline off,
        # generic persisted backlog must not run it under the new prompt.
        hy.config = replace(
            hy.config,
            prompt_version="v21",
            salience_min_chars=1_000,
            dream_baseline_budget=0,
        )
        held = hy.dream()
        assert held.chunk_extraction_completion_calls == 0
        assert hy.conn.execute(
            "SELECT 1 FROM processed_chunks "
                "WHERE chunk_id=? AND prompt_version=?",
                (chunk["id"], extraction_cache_key("v21")),
        ).fetchone() is None

        hy.config = replace(hy.config, dream_baseline_budget=1)
        resumed = hy.dream()
        assert resumed.chunk_extraction_completion_calls == 2
        assert hy.conn.execute(
            "SELECT 1 FROM processed_chunks "
                "WHERE chunk_id=? AND prompt_version=?",
                (chunk["id"], extraction_cache_key("v21")),
        ).fetchone() is not None
    finally:
        hy.close()
