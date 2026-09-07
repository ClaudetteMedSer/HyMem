from __future__ import annotations

from dataclasses import replace

import pytest

from hymem import HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import facts
from hymem.dreaming.facts import (
    fact_cursor_retry_unit_key,
    facts_retry_policy_version,
)
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import StubLLMClient


def _cursor_retry_unit(hy: HyMem, session_id: str) -> str:
    row = hy.conn.execute(
        "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
        "facts_cursor_offset FROM sessions WHERE id=?",
        (session_id,),
    ).fetchone()
    assert row is not None
    return fact_cursor_retry_unit_key(
        session_id,
        row["facts_cursor_message_id"],
        row["facts_cursor_partial_message_id"],
        row["facts_cursor_offset"],
    )


def _set_quarantine(
    hy: HyMem,
    session_id: str,
    identity: str,
    *,
    attempts: int | None = None,
) -> None:
    hy.conn.execute(
        "UPDATE sessions SET facts_retry_count=?,"
        "facts_retry_config_version=?,facts_quarantined=1 WHERE id=?",
        (
            (
                hy.config.facts_extraction_max_attempts
                if attempts is None
                else attempts
            ),
            identity,
            session_id,
        ),
    )


def test_dream_status_reports_active_current_fact_quarantine(hy: HyMem):
    session_id = "active-fact-quarantine"
    hy.log_message(session_id, "user", "Remember the active fact retry.")
    identity = facts_retry_policy_version(
        hy.config,
        replay_slice_key=_cursor_retry_unit(hy, session_id),
        client=hy._llm,
    )
    _set_quarantine(hy, session_id, identity)

    status = hy.dream_status()

    assert status["quarantined_facts"] == 1
    assert status["quarantined_facts_malformed"] == 0


def test_dream_status_reopens_stale_fact_quarantine_identity(hy: HyMem):
    session_id = "stale-fact-quarantine"
    hy.log_message(session_id, "user", "Remember the stale fact retry.")
    stale_config = replace(
        hy.config,
        dream_digest_max_chars=hy.config.dream_digest_max_chars + 1,
    )
    stale_identity = facts_retry_policy_version(
        stale_config,
        replay_slice_key=_cursor_retry_unit(hy, session_id),
    )
    _set_quarantine(hy, session_id, stale_identity)

    status = hy.dream_status()

    assert status["quarantined_facts"] == 0
    assert status["quarantined_facts_malformed"] == 0


def test_dream_status_selects_stale_fact_outcome_as_current_retry_unit(
    cfg: HyMemConfig,
):
    old_config = replace(
        cfg,
        dream_digest_max_chars=cfg.dream_digest_max_chars + 1,
    )
    old = HyMem(old_config, llm=StubLLMClient(default="[]"))
    session_id = "stale-fact-outcome"
    try:
        old.log_message(session_id, "user", "The first old-policy fact.")
        with core_db.transaction(old.conn):
            materialize_message_coverage(old.conn, session_id)
        first = facts.extract_facts(
            old.conn,
            session_id,
            StubLLMClient(default="[]"),
            old.config,
        )
        assert first is not None
        with core_db.transaction(old.conn):
            facts.persist_facts(old.conn, session_id, first)
        first_slice_key = first.slice_key
        assert isinstance(first_slice_key, str)

        old.log_message(session_id, "user", "The second old-policy fact.")
        with core_db.transaction(old.conn):
            materialize_message_coverage(old.conn, session_id)
        second = facts.extract_facts(
            old.conn,
            session_id,
            StubLLMClient(default="[]"),
            old.config,
            since_message_id=first.covered_message_id,
            partial_message_id=first.partial_message_id,
            start_offset=first.next_message_offset,
        )
        assert second is not None
        with core_db.transaction(old.conn):
            facts.persist_facts(old.conn, session_id, second)
        second_slice_key = second.slice_key
        assert isinstance(second_slice_key, str)
        assert second_slice_key != first_slice_key
    finally:
        old.close()

    current = HyMem(cfg)
    try:
        assert facts.next_fact_outcome_for_replay(
            current.conn,
            session_id,
            facts.facts_config_version(current.config),
        ) == first_slice_key
        identity = facts_retry_policy_version(
            current.config, replay_slice_key=first_slice_key
        )
        _set_quarantine(current, session_id, identity)

        status = current.dream_status()

        assert status["quarantined_facts"] == 1
        assert status["quarantined_facts_malformed"] == 0

        later_identity = facts_retry_policy_version(
            current.config, replay_slice_key=second_slice_key
        )
        _set_quarantine(current, session_id, later_identity)
        later_status = current.dream_status()
        assert later_status["quarantined_facts"] == 0
        assert later_status["quarantined_facts_malformed"] == 0
    finally:
        current.close()


def test_dream_status_surfaces_malformed_flagged_fact_retry_state(hy: HyMem):
    session_id = "malformed-fact-quarantine"
    hy.log_message(session_id, "user", "Remember the malformed fact retry.")
    _set_quarantine(hy, session_id, "not-a-fact-retry-identity")

    status = hy.dream_status()

    assert status["quarantined_facts"] == 0
    assert status["quarantined_facts_malformed"] == 1


def test_dream_status_surfaces_flagged_invalid_fact_cursor_as_malformed(
    hy: HyMem,
):
    session_id = "invalid-fact-cursor"
    hy.log_message(session_id, "user", "A fact with a corrupted cursor.")
    hy.conn.execute(
        "UPDATE sessions SET facts_cursor_message_id=? WHERE id=?",
        ("not-an-integer", session_id),
    )
    retry_unit = fact_cursor_retry_unit_key(
        session_id,
        "not-an-integer",  # type: ignore[arg-type]
        None,
        0,
    )
    identity = facts_retry_policy_version(
        hy.config, replay_slice_key=retry_unit
    )
    _set_quarantine(hy, session_id, identity)

    status = hy.dream_status()

    assert status["quarantined_facts"] == 0
    assert status["quarantined_facts_malformed"] == 1


def test_dream_status_surfaces_flagged_incomplete_fact_outcome_as_malformed(
    cfg: HyMemConfig,
):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    session_id = "incomplete-fact-outcome"
    try:
        hy.log_message(session_id, "user", "A staged fact outcome.")
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, session_id)
        extraction = facts.extract_facts(
            hy.conn,
            session_id,
            StubLLMClient(default="[]"),
            hy.config,
        )
        assert extraction is not None
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, session_id, extraction)
        with core_db.transaction(hy.conn):
            with core_db.evidence_destructive_mutation(hy.conn):
                hy.conn.execute(
                    "UPDATE fact_extraction_outcomes SET "
                    "source_manifest_version=NULL,source_manifest_count=0,"
                    "source_manifest_hash=NULL,source_manifest_complete=0 "
                    "WHERE slice_key=?",
                    (extraction.slice_key,),
                )
        identity = facts_retry_policy_version(
            hy.config,
            replay_slice_key=_cursor_retry_unit(hy, session_id),
        )
        _set_quarantine(hy, session_id, identity)

        with pytest.raises(RuntimeError, match="publication is incomplete"):
            facts.next_fact_outcome_for_replay(
                hy.conn,
                session_id,
                facts.facts_config_version(hy.config),
            )
        status = hy.dream_status()

        assert status["quarantined_facts"] == 0
        assert status["quarantined_facts_malformed"] == 1
    finally:
        hy.close()


def test_dream_status_suppresses_fact_quarantine_when_extraction_disabled(
    cfg: HyMemConfig,
):
    disabled = HyMem(replace(cfg, facts_extraction_enabled=False))
    try:
        session_id = "disabled-fact-quarantine"
        disabled.log_message(
            session_id, "user", "A historical disabled fact retry."
        )
        identity = facts_retry_policy_version(
            disabled.config,
            replay_slice_key=_cursor_retry_unit(disabled, session_id),
        )
        _set_quarantine(disabled, session_id, identity)

        status = disabled.dream_status()

        assert status["quarantined_facts"] == 0
        assert status["quarantined_facts_malformed"] == 0
    finally:
        disabled.close()


def test_dream_status_zero_fact_retry_bound_ignores_flagged_old_state(
    cfg: HyMemConfig,
):
    zero_bound = HyMem(replace(cfg, facts_extraction_max_attempts=0))
    try:
        session_id = "zero-bound-fact-quarantine"
        zero_bound.log_message(
            session_id, "user", "A historical bounded fact retry."
        )
        historical_identity = facts_retry_policy_version(
            cfg,
            replay_slice_key=_cursor_retry_unit(zero_bound, session_id),
        )
        _set_quarantine(
            zero_bound,
            session_id,
            historical_identity,
            attempts=cfg.facts_extraction_max_attempts,
        )

        status = zero_bound.dream_status()

        assert status["quarantined_facts"] == 0
        assert status["quarantined_facts_malformed"] == 0
    finally:
        zero_bound.close()


@pytest.mark.parametrize(
    "disabled_config",
    (
        {"facts_extraction_enabled": False},
        {"facts_extraction_max_attempts": 0},
    ),
    ids=("extraction-disabled", "zero-retry-bound"),
)
def test_disabled_fact_quarantine_status_does_not_query_store(
    cfg: HyMemConfig,
    disabled_config: dict[str, object],
):
    class BombConnection:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("disabled fact status must not query the store")

    assert facts.fact_quarantine_status(
        BombConnection(), replace(cfg, **disabled_config)
    ) == {
        "quarantined_facts": 0,
        "quarantined_facts_malformed": 0,
    }
