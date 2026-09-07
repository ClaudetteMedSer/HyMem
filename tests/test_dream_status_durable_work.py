from __future__ import annotations

import json
from dataclasses import replace

import pytest

from benchmarks import strictness
from hymem import HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import facts, runner
from hymem.dreaming.chunks import source_materialization_config_version
from hymem.dreaming.facts import (
    fact_cursor_retry_unit_key,
    facts_retry_policy_version,
)
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import StubLLMClient


def _cfg(cfg: HyMemConfig, **changes) -> HyMemConfig:
    return replace(
        cfg,
        dream_budget=20,
        dream_baseline_budget=20,
        salience_min_chars=1,
        aggregation_nodes_enabled=False,
        rules_extraction_enabled=False,
        **changes,
    )


def _pipeline_llm(
    *, digest: str | None = None, profile: str | None = None,
    fact_items: str | None = None,
) -> StubLLMClient:
    return StubLLMClient(
        fixtures={
            "structured technical relationships": json.dumps({
                "triples": [], "markers": [], "complete": True,
            }),
            "typed user-profile facts": (
                profile if profile is not None else json.dumps({"items": []})
            ),
            "Return the JSON array of narrative facts now.": (
                fact_items if fact_items is not None else "[]"
            ),
            "Return the JSON object now.": (
                digest if digest is not None else json.dumps({
                    "episodes": [],
                    "summary": "No additional durable details.",
                    "procedures": [],
                })
            ),
        },
        default=None,
    )


def _assert_clean_durable_work(status: dict) -> None:
    for key in (
        "pending_source_materialization", "pending_chunks", "pending_digests",
        "pending_profiles", "pending_facts", "pending_aggregation",
        "malformed_source_materialization", "malformed_digests",
        "malformed_profiles", "malformed_facts",
    ):
        assert status[key] == 0, (key, status[key])


def test_fresh_source_config_change_and_post_run_append_reopen_status(
    cfg: HyMemConfig,
):
    hy = HyMem(_cfg(cfg), llm=_pipeline_llm())
    try:
        hy.log_message("status-life", "user", "Remember the cobalt rollout.")
        before = hy.dream_status()
        assert before["pending_source_materialization"] == 1
        assert before["pending_digests"] == 1
        assert before["pending_profiles"] == 1
        assert before["pending_facts"] == 1

        report = hy.dream()
        assert report.chunk_extraction_failures == 0
        assert report.digest_failures == 0
        assert report.profile_failures == 0
        assert report.fact_failures == 0
        _assert_clean_durable_work(hy.dream_status())

        hy.config = replace(
            hy.config, salience_min_chars=hy.config.salience_min_chars + 1
        )
        assert hy.dream_status()["pending_source_materialization"] == 1
        hy.dream()
        _assert_clean_durable_work(hy.dream_status())

        hy.log_message("status-life", "user", "A later source turn arrived.")
        reopened = hy.dream_status()
        assert reopened["pending_source_materialization"] == 1
        assert reopened["pending_digests"] == 1
        assert reopened["pending_profiles"] == 1
        assert reopened["pending_facts"] == 1
    finally:
        hy.close()


def test_raw_tail_beyond_coverage_is_never_hidden_by_old_ack(cfg: HyMemConfig):
    hy = HyMem(
        _cfg(
            cfg, profile_extraction_enabled=False,
            facts_extraction_enabled=False,
        ),
        llm=_pipeline_llm(),
    )
    try:
        hy.log_message("raw-tail", "user", "The covered source turn.")
        hy.dream()
        row = hy.conn.execute(
            "SELECT coverage_message_id,source_materialized_message_id "
            "FROM sessions WHERE id='raw-tail'"
        ).fetchone()
        assert row["coverage_message_id"] == row["source_materialized_message_id"]

        hy.conn.execute(
            "INSERT INTO messages(session_id,role,content) VALUES (?,?,?)",
            ("raw-tail", "user", "A direct legacy append."),
        )
        status = hy.dream_status()
        assert status["pending_source_materialization"] == 1
        assert status["malformed_source_materialization"] == 0
    finally:
        hy.close()


def test_source_ack_rolls_back_when_local_classification_fails(
    cfg: HyMemConfig, monkeypatch: pytest.MonkeyPatch,
):
    hy = HyMem(
        _cfg(
            cfg, profile_extraction_enabled=False,
            facts_extraction_enabled=False,
        ),
        llm=_pipeline_llm(),
    )
    try:
        hy.log_message("ack-crash", "user", "A source requiring classification.")

        def fail_terminalize(*_args, **_kwargs):
            raise RuntimeError("injected classification seam")

        monkeypatch.setattr(
            runner, "record_unrecoverable_chunk_losses", fail_terminalize
        )
        with pytest.raises(RuntimeError, match="classification seam"):
            hy.dream()

        row = hy.conn.execute(
            "SELECT source_materialized_message_id,"
            "source_materialization_config_version FROM sessions "
            "WHERE id='ack-crash'"
        ).fetchone()
        assert tuple(row) == (None, None)
        assert hy.dream_status()["pending_source_materialization"] == 1
    finally:
        hy.close()


def test_source_builders_are_fenced_to_captured_coverage_frontier(
    cfg: HyMemConfig, monkeypatch: pytest.MonkeyPatch,
):
    hy = HyMem(
        _cfg(
            cfg, profile_extraction_enabled=False,
            facts_extraction_enabled=False,
        ),
        llm=_pipeline_llm(),
    )
    try:
        session_id = "source-frontier-race"
        initial_id = hy.log_message(session_id, "user", "The initial turn.")
        original = runner.extract_high_salience_chunks
        appended: list[int] = []

        def append_after_target(conn, source_session_id, **kwargs):
            if not appended:
                appended.append(
                    hy.log_message(
                        session_id, "user", "The concurrently appended turn."
                    )
                )
            return original(conn, source_session_id, **kwargs)

        monkeypatch.setattr(
            runner, "extract_high_salience_chunks", append_after_target
        )
        first = hy.dream()
        assert first.coverage_integrity_failures == 0
        assert appended and appended[0] > initial_id

        state = hy.conn.execute(
            "SELECT coverage_message_id,source_materialized_message_id "
            "FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        assert state["coverage_message_id"] == appended[0]
        assert state["source_materialized_message_id"] == initial_id
        assert hy.conn.execute(
            "SELECT 1 FROM chunks WHERE session_id=? AND chunk_kind='extraction' "
            "AND end_message_id=?",
            (session_id, appended[0]),
        ).fetchone() is None
        assert hy.dream_status()["pending_source_materialization"] == 1

        hy.dream()
        assert hy.conn.execute(
            "SELECT 1 FROM chunks WHERE session_id=? AND chunk_kind='extraction' "
            "AND end_message_id=?",
            (session_id, appended[0]),
        ).fetchone() is not None
        assert hy.dream_status()["pending_source_materialization"] == 0
    finally:
        hy.close()


@pytest.mark.parametrize(
    ("subsystem", "report_field", "pending_field", "quarantine_field"),
    (
        ("digest", "digest_failures", "pending_digests", "quarantined_digests"),
        ("profile", "profile_failures", "pending_profiles", "quarantined_profiles"),
        ("facts", "fact_failures", "pending_facts", "quarantined_facts"),
    ),
)
def test_first_consumer_failure_remains_pending_and_later_heals(
    cfg: HyMemConfig,
    subsystem: str,
    report_field: str,
    pending_field: str,
    quarantine_field: str,
):
    options = {
        "digest": {"digest": "not-json"},
        "profile": {"profile": "not-json"},
        "facts": {"fact_items": "not-json"},
    }[subsystem]
    hy = HyMem(_cfg(cfg), llm=_pipeline_llm(**options))
    try:
        hy.log_message(
            f"retry-{subsystem}", "user", f"Remember the {subsystem} retry."
        )
        first = hy.dream()
        assert getattr(first, report_field) == 1
        held = hy.dream_status()
        assert held[pending_field] == 1
        assert held[quarantine_field] == 0

        hy.set_llm(_pipeline_llm())
        healed = hy.dream()
        assert getattr(healed, report_field) == 0
        assert hy.dream_status()[pending_field] == 0
    finally:
        hy.close()


def test_malformed_dynamic_cursor_values_fail_closed_without_status_exception(
    cfg: HyMemConfig,
):
    hy = HyMem(_cfg(cfg), llm=_pipeline_llm())
    try:
        hy.log_message("malformed-cursors", "user", "A covered source.")
        coverage = hy.conn.execute(
            "SELECT coverage_message_id FROM sessions WHERE id='malformed-cursors'"
        ).fetchone()[0]
        hy.conn.execute(
            "UPDATE sessions SET source_materialized_message_id=?,"
            "source_materialization_config_version=?,digest_cursor_offset=?,"
            "profile_cursor_offset=?,facts_cursor_offset=? WHERE id=?",
            (
                int(coverage) + 1,
                source_materialization_config_version(
                    min_chars=hy.config.salience_min_chars
                ),
                "bad", "bad", "bad", "malformed-cursors",
            ),
        )

        status = hy.dream_status()
        assert status["malformed_source_materialization"] == 1
        assert status["malformed_digests"] == 1
        assert status["malformed_profiles"] == 1
        assert status["malformed_facts"] == 1

        hy.conn.execute(
            "UPDATE sessions SET source_materialized_message_id='bad' "
            "WHERE id='malformed-cursors'"
        )
        assert hy.dream_status()["malformed_source_materialization"] == 1
    finally:
        hy.close()


def test_disabled_optional_consumers_hide_historical_retry_corruption(
    cfg: HyMemConfig,
):
    hy = HyMem(
        _cfg(
            cfg, profile_extraction_enabled=False,
            facts_extraction_enabled=False,
        ),
        llm=_pipeline_llm(),
    )
    try:
        hy.open_session("disabled-consumers")
        hy.conn.execute(
            "UPDATE sessions SET profile_cursor_offset='bad',"
            "profile_retry_count='bad',profile_quarantined=1,"
            "facts_cursor_offset='bad',facts_retry_count='bad',"
            "facts_quarantined=1 WHERE id='disabled-consumers'"
        )
        status = hy.dream_status()
        for key in (
            "pending_profiles", "quarantined_profiles", "malformed_profiles",
            "pending_facts", "quarantined_facts",
            "quarantined_facts_malformed", "malformed_facts",
        ):
            assert status[key] == 0
    finally:
        hy.close()


def test_fact_marker_only_work_ignores_quarantine_and_heals_without_llm(
    cfg: HyMemConfig,
):
    llm = StubLLMClient(default=None)
    hy = HyMem(
        _cfg(cfg, profile_extraction_enabled=False),
        llm=llm,
    )
    try:
        session_id = "fact-marker-only"
        hy.open_session(session_id)
        unit = fact_cursor_retry_unit_key(session_id, None, None, 0)
        retry_key = facts_retry_policy_version(
            hy.config, replay_slice_key=unit, client=llm,
        )
        hy.conn.execute(
            "UPDATE sessions SET facts_retry_count=?,"
            "facts_retry_config_version=?,facts_quarantined=1 WHERE id=?",
            (hy.config.facts_extraction_max_attempts, retry_key, session_id),
        )

        held = hy.dream_status()
        assert held["pending_facts"] == 1
        assert held["quarantined_facts"] == 0
        assert facts.fact_quarantine_status(hy.conn, hy.config, client=llm)[
            "quarantined_facts"
        ] == 0

        report = hy.dream()
        assert report.fact_failures == 0
        assert llm.calls == []
        assert hy.dream_status()["pending_facts"] == 0

        # The same retry residue is impossible once the current local marker
        # has been published; it is audit-only malformed state, not work.
        hy.conn.execute(
            "UPDATE sessions SET facts_retry_count=?,"
            "facts_retry_config_version=?,facts_quarantined=1 WHERE id=?",
            (hy.config.facts_extraction_max_attempts, retry_key, session_id),
        )
        malformed = hy.dream_status()
        assert malformed["pending_facts"] == 0
        assert malformed["quarantined_facts"] == 0
        assert malformed["malformed_facts"] == 1
    finally:
        hy.close()


@pytest.mark.parametrize("column", ("source_manifest_hash", "result_hash"))
def test_current_fact_authority_hash_corruption_is_malformed_not_pending(
    cfg: HyMemConfig, column: str,
):
    hy = HyMem(_cfg(cfg), llm=_pipeline_llm())
    try:
        hy.log_message("fact-authority", "user", "Remember the durable fact.")
        hy.dream()
        outcome = hy.conn.execute(
            "SELECT slice_key FROM fact_extraction_outcomes "
            "WHERE session_id='fact-authority'"
        ).fetchone()
        assert outcome is not None
        with core_db.transaction(hy.conn):
            with core_db.evidence_destructive_mutation(hy.conn):
                hy.conn.execute(
                    f"UPDATE fact_extraction_outcomes SET {column}=? "
                    "WHERE slice_key=?",
                    ("sha256:" + "0" * 64, outcome["slice_key"]),
                )

        assert facts.fact_session_authority_is_valid(
            hy.conn, "fact-authority"
        ) is False
        status = hy.dream_status()
        assert status["malformed_facts"] == 1
        assert status["pending_facts"] == 0
        assert status["quarantined_facts"] == 0
    finally:
        hy.close()


def test_status_snapshot_does_not_commit_shared_reader_or_mix_later_write(
    cfg: HyMemConfig,
):
    hy = HyMem(
        _cfg(
            cfg, profile_extraction_enabled=False,
            facts_extraction_enabled=False,
        ),
        llm=_pipeline_llm(),
    )
    try:
        hy.log_message("snapshot", "user", "The initial source.")
        hy.dream()

        hy.read_conn.execute("BEGIN")
        try:
            assert hy.dream_status()["pending_source_materialization"] == 0
            assert hy.read_conn.in_transaction is True
        finally:
            hy.read_conn.execute("ROLLBACK")

        def append_between_halves(conn):
            before = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            hy.log_message("snapshot", "user", "A concurrent later source.")
            after = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            return {"snapshot_message_count": after, "expected_count": before}

        coherent = hy.dream_status_with_snapshot(append_between_halves)
        assert coherent["snapshot_message_count"] == coherent["expected_count"]
        assert coherent["pending_source_materialization"] == 0
        assert hy.dream_status()["pending_source_materialization"] == 1
    finally:
        hy.close()


def test_benchmark_durable_and_embedding_status_share_one_snapshot(
    cfg: HyMemConfig, monkeypatch: pytest.MonkeyPatch,
):
    hy = HyMem(
        _cfg(
            cfg, profile_extraction_enabled=False,
            facts_extraction_enabled=False,
        ),
        llm=_pipeline_llm(),
    )
    try:
        session_id = "benchmark-snapshot"
        hy.log_message(session_id, "user", "The initial source.")
        hy.dream()

        def embedding_probe(conn, _client):
            before = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            hy.log_message(session_id, "user", "A later concurrent source.")
            after = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            assert after == before
            return {
                "pending_chunk_embeddings": 0,
                "pending_message_embeddings": 0,
                "pending_edge_embeddings": 0,
                "pending_episode_embeddings": 0,
                "pending_fact_embeddings": 0,
            }

        monkeypatch.setattr(
            strictness, "embedding_backlog_status", embedding_probe
        )
        coherent = strictness.durable_indexing_status(hy, object())
        assert coherent["pending_source_materialization"] == 0
        assert coherent["pending_message_embeddings"] == 0
        assert hy.dream_status()["pending_source_materialization"] == 1
    finally:
        hy.close()


def test_fact_replay_uses_source_order_across_lower_and_higher_versions(
    cfg: HyMemConfig,
):
    current = _cfg(cfg, profile_extraction_enabled=False)
    higher = replace(
        current, dream_digest_max_chars=current.dream_digest_max_chars + 1
    )
    lower = replace(
        current, dream_digest_max_chars=current.dream_digest_max_chars - 1
    )
    llm = _pipeline_llm()
    hy = HyMem(current, llm=llm)
    try:
        session_id = "mixed-fact-generations"
        hy.log_message(session_id, "user", "The older higher-version source.")
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, session_id)
        first = facts.extract_facts(
            hy.conn, session_id, llm, hy.config
        )
        assert first is not None
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, session_id, first)

        hy.log_message(session_id, "user", "The newer lower-version source.")
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, session_id)
        second = facts.extract_facts(
            hy.conn,
            session_id,
            llm,
            hy.config,
            since_message_id=first.covered_message_id,
            partial_message_id=first.partial_message_id,
            start_offset=first.next_message_offset,
        )
        assert second is not None
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, session_id, second)

        current_version = facts.facts_config_version(current)
        higher_version = facts.facts_config_version(higher)
        lower_version = facts.facts_config_version(lower)
        with core_db.transaction(hy.conn):
            with core_db.evidence_destructive_mutation(hy.conn):
                for slice_key, version in (
                    (first.slice_key, higher_version),
                    (second.slice_key, lower_version),
                ):
                    revision = hy.conn.execute(
                        "SELECT generation,outcome_status,result_hash,succeeded_at "
                        "FROM fact_extraction_revisions WHERE slice_key=?",
                        (slice_key,),
                    ).fetchone()
                    assert revision is not None
                    hy.conn.execute(
                        "DELETE FROM fact_extraction_revisions WHERE slice_key=?",
                        (slice_key,),
                    )
                    hy.conn.execute(
                        "UPDATE fact_extraction_outcomes SET prompt_version=? "
                        "WHERE slice_key=?",
                        (version, slice_key),
                    )
                    hy.conn.execute(
                        "INSERT INTO fact_extraction_revisions("
                        "slice_key,generation,prompt_version,outcome_status,"
                        "result_hash,succeeded_at) VALUES (?,?,?,?,?,?)",
                        (
                            slice_key, revision["generation"], version,
                            revision["outcome_status"], revision["result_hash"],
                            revision["succeeded_at"],
                        ),
                    )
        rows = hy.conn.execute(
            "SELECT slice_key,prompt_version FROM fact_extraction_outcomes "
            "WHERE session_id=? ORDER BY generation",
            (session_id,),
        ).fetchall()
        assert rows[0]["prompt_version"] == higher_version > current_version
        assert rows[1]["prompt_version"] == lower_version < current_version
        assert facts.next_fact_outcome_for_replay(
            hy.conn, session_id, current_version
        ) == first.slice_key
    finally:
        hy.close()
