"""v62 keeps source-proved item and summary acknowledgements independent."""
from __future__ import annotations

import sqlite3

import pytest

from hymem.core import db
from hymem.dreaming.digest import digest_config_version
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.summary_state import (
    SUMMARY_FAILURE_REASONS, classify_summary_state, mark_summary_current, record_summary_failure,
)
from hymem.session import _session_is_pristine, append_message, open_session


@pytest.fixture
def conn(tmp_path):
    value = db.connect(tmp_path / "frontier.sqlite")
    db.initialize(value)
    yield value
    value.close()


def _generation(char="a"):
    return digest_config_version(prompt_version="v1", episode_prompt_version=None,
                                 max_chars=8000, max_tokens=3072, max_episodes=None) + "|walk=" + char * 32


def _source(conn, sid="x"):
    open_session(conn, sid)
    ids = [append_message(conn, sid, "user", text)
           for text in ("The first exact source message.", "A second independently retained message.")]
    with db.transaction(conn):
        materialize_message_coverage(conn, sid)
    return ids


def _published(conn, sid="x"):
    first, last = _source(conn, sid)
    generation = _generation()
    conn.execute("UPDATE sessions SET digest_published_message_id=?,digest_published_generation=? WHERE id=?",
                 (last, generation, sid))
    with db.transaction(conn):
        mark_summary_current(conn, sid, "Both source messages were retained.",
                             generation=generation, covered_message_id=last)
    return first, last, generation


def _v61(conn):
    conn.execute("DROP TRIGGER IF EXISTS summary_state_workspace_guard")
    for name in ("digest_published_message_id", "auto_summary_generation",
                 "summary_failure_reason", "summary_failure_count"):
        conn.execute(f"ALTER TABLE sessions DROP COLUMN {name}")
    conn.execute("ALTER TABLE digest_staging DROP COLUMN summary_failure_reason")
    conn.execute("UPDATE schema_meta SET value='61' WHERE key='schema_version'")


def test_fresh_defaults_are_pristine_and_reopen_is_idempotent(conn):
    open_session(conn, "empty")
    assert db.schema_version(conn) == db.EXPECTED_SCHEMA_VERSION
    assert _session_is_pristine(conn, "empty")
    assert classify_summary_state(conn, "empty") == dict(
        summary_healthy=True, degraded=False, missing=False, malformed=False)
    before = tuple(conn.execute("SELECT * FROM sessions WHERE id='empty'").fetchone())
    db.initialize(conn)
    assert tuple(conn.execute("SELECT * FROM sessions WHERE id='empty'").fetchone()) == before
    open_session(conn, "empty", source_workspace_id="owned")
    assert conn.execute("SELECT source_workspace_id FROM sessions WHERE id='empty'").fetchone()[0] == "owned"


@pytest.mark.parametrize("active", ["completed", "replacement_partial", "forward_partial"])
def test_v61_backfills_only_published_summary_not_active_cursor(conn, active):
    first, last, generation = _published(conn)
    _v61(conn)
    new_generation = generation if active != "replacement_partial" else _generation("b")
    if active == "completed":
        cursor, partial, offset = last, None, 0
    else:
        cursor, partial, offset = first, last, 3
    conn.execute("UPDATE sessions SET digest_cursor_prompt_version=?,digest_cursor_message_id=?,"
                 "digest_cursor_partial_message_id=?,digest_cursor_offset=? WHERE id='x'",
                 (new_generation, cursor, partial, offset))
    before = tuple(conn.execute("SELECT summary,summary_source,auto_summary FROM sessions WHERE id='x'").fetchone())
    db.initialize(conn)
    row = conn.execute("SELECT * FROM sessions WHERE id='x'").fetchone()
    assert row["digest_published_message_id"] == last
    assert row["auto_summary_generation"] == generation
    assert (row["digest_cursor_message_id"], row["digest_cursor_partial_message_id"], row["digest_cursor_offset"]) == (cursor, partial, offset)
    assert tuple(row[key] for key in ("summary", "summary_source", "auto_summary")) == before
    assert classify_summary_state(conn, "x")["summary_healthy"]


@pytest.mark.parametrize("mutation", ["legacy", "unproved", "oversized", "partial", "missing"])
def test_v61_does_not_invent_uncertain_publication(conn, mutation):
    first, last, generation = _published(conn)
    _v61(conn)
    if mutation == "legacy":
        conn.execute("UPDATE sessions SET digest_published_generation='legacy'")
    elif mutation == "unproved":
        conn.execute("UPDATE sessions SET auto_summary_message_id=99999")
    elif mutation == "oversized":
        conn.execute("UPDATE sessions SET auto_summary=?,summary=?,summary_source='legacy'", ("x" * 750, "x" * 750))
    elif mutation == "partial":
        conn.execute("UPDATE sessions SET auto_summary_message_id=?,auto_summary_partial_message_id=?,auto_summary_message_offset=3", (first, last))
    else:
        conn.execute("UPDATE sessions SET auto_summary=NULL")
    before = tuple(conn.execute("SELECT summary,summary_source,auto_summary FROM sessions").fetchone())
    db.initialize(conn)
    row = conn.execute("SELECT * FROM sessions").fetchone()
    assert row["digest_published_message_id"] is None and row["auto_summary_generation"] is None
    assert tuple(row[key] for key in ("summary", "summary_source", "auto_summary")) == before


def test_v62_migration_rolls_back_ddl_backfill_and_stamp(conn, monkeypatch):
    _published(conn)
    _v61(conn)
    original = db._backfill_v62_summary_frontier

    def fail(value):
        original(value)
        raise RuntimeError("injected migration interruption")

    monkeypatch.setattr(db, "_backfill_v62_summary_frontier", fail)
    with pytest.raises(RuntimeError, match="injected"):
        db.initialize(conn)
    assert db.schema_version(conn) == 61 and not conn.in_transaction
    assert "digest_published_message_id" not in {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    monkeypatch.setattr(db, "_backfill_v62_summary_frontier", original)
    db.initialize(conn)
    assert db.schema_version(conn) == db.EXPECTED_SCHEMA_VERSION
    assert classify_summary_state(conn, "x")["summary_healthy"]


@pytest.mark.parametrize("column,value", [
    ("digest_published_message_id", 0), ("digest_published_message_id", -1),
    ("digest_published_message_id", "not-an-id"), ("digest_published_message_id", 1.25),
    ("auto_summary_generation", ""), ("summary_failure_count", -1),
    ("summary_failure_count", 1.5), ("summary_failure_count", "bad"),
    ("summary_failure_reason", ""), ("summary_failure_reason", "x" * 81),
    ("summary_failure_reason", "raw failure text"), ("summary_failure_reason", "UpperCase"),
    ("summary_failure_reason", "unknown_future_reason"),
])
def test_new_scalar_constraints_reject_malformed_sql(conn, column, value):
    open_session(conn, "empty")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(f"UPDATE sessions SET {column}=? WHERE id='empty'", (value,))


@pytest.mark.parametrize("mutation", ["unknown_generation", "missing_source", "summary_ahead", "broken_offset", "reason_only", "count_only", "text_missing", "item_source_missing", "bad_cursor_type", "unproved_coverage"])
def test_classifier_exposes_malformed_state_without_accepting_it(conn, mutation):
    first, last, generation = _published(conn)
    sql = {
        "unknown_generation": "auto_summary_generation='unknown'",
        "missing_source": "auto_summary_message_id=99999",
        "summary_ahead": f"digest_published_message_id={first}",
        "broken_offset": "auto_summary_message_offset=1",
        "reason_only": "summary_failure_reason='summary_output_cap'",
        "count_only": "summary_failure_count=1",
        "text_missing": "auto_summary=NULL",
        "item_source_missing": "digest_published_message_id=99999",
        "bad_cursor_type": "auto_summary_message_id='bad'",
        "unproved_coverage": "coverage_message_id=99999",
    }[mutation]
    conn.execute(f"UPDATE sessions SET {sql} WHERE id='x'")
    state = classify_summary_state(conn, "x")
    assert state["malformed"] and not state["summary_healthy"]


def test_stale_summary_preserved_until_fenced_complete_recovery(conn):
    first, last, generation = _published(conn)
    old = tuple(conn.execute("SELECT summary,auto_summary,auto_summary_message_id,auto_summary_generation FROM sessions").fetchone())
    following = append_message(conn, "x", "user", "A third source arrived after summary failure.")
    with db.transaction(conn):
        materialize_message_coverage(conn, "x")
        conn.execute("UPDATE sessions SET digest_published_message_id=? WHERE id='x'", (following,))
        record_summary_failure(conn, "x", "summary_output_cap")
        record_summary_failure(conn, "x", "prior_summary_gap")
    row = conn.execute("SELECT * FROM sessions").fetchone()
    assert tuple(row[key] for key in ("summary", "auto_summary", "auto_summary_message_id", "auto_summary_generation")) == old
    assert row["summary_failure_reason"] == "summary_output_cap" and row["summary_failure_count"] == 2
    assert classify_summary_state(conn, "x") == dict(summary_healthy=False, degraded=True, missing=False, malformed=False)
    with db.transaction(conn):
        mark_summary_current(conn, "x", "All three messages were covered.", generation=generation, covered_message_id=following)
    assert classify_summary_state(conn, "x")["summary_healthy"]
    assert conn.execute("SELECT summary_failure_count FROM sessions").fetchone()[0] == 0


def test_new_source_is_stale_but_old_summary_remains_contiguous_item_context(conn):
    first, last, generation = _published(conn)
    following = append_message(conn, "x", "user", "New material has not been summarized yet.")
    with db.transaction(conn):
        materialize_message_coverage(conn, "x")
    assert classify_summary_state(conn, "x") == dict(summary_healthy=False, degraded=True, missing=False, malformed=False)
    assert classify_summary_state(conn, "x", require_source_tail=False)["summary_healthy"]
    conn.execute("UPDATE sessions SET digest_published_message_id=? WHERE id='x'", (following,))
    assert not classify_summary_state(conn, "x")["summary_healthy"]
    assert not classify_summary_state(conn, "x", require_source_tail=False)["summary_healthy"]
    with db.transaction(conn):
        mark_summary_current(conn, "x", "All three source messages were summarized.",
                             generation=generation, covered_message_id=following)
    assert classify_summary_state(conn, "x")["summary_healthy"]


@pytest.mark.parametrize("policy", [None, 0, 1, "false", "true"])
def test_source_tail_policy_cannot_be_coerced_from_non_boolean(conn, policy):
    open_session(conn, "empty")
    with pytest.raises(ValueError, match="boolean"):
        classify_summary_state(conn, "empty", require_source_tail=policy)


@pytest.mark.parametrize("role", ["user", "assistant", "system", "tool"])
def test_raw_append_is_stale_before_coverage_without_breaking_item_continuity(conn, role):
    _published(conn)
    append_message(conn, "x", role, "A new raw turn not yet materialized.")
    assert classify_summary_state(conn, "x") == dict(summary_healthy=False, degraded=True, missing=False, malformed=False)
    assert classify_summary_state(conn, "x", require_source_tail=False)["summary_healthy"]


def test_raw_only_session_needs_summary_but_empty_session_does_not(conn):
    open_session(conn, "empty")
    assert classify_summary_state(conn, "empty")["summary_healthy"]
    append_message(conn, "empty", "user", "Source preceding the first coverage pass.")
    assert classify_summary_state(conn, "empty") == dict(summary_healthy=False, degraded=True, missing=True, malformed=False)
    # Continuity mode deliberately describes only the published/covered base.
    assert classify_summary_state(conn, "empty", require_source_tail=False)["summary_healthy"]


def test_pruned_raw_messages_do_not_erase_canonical_summary_freshness(conn):
    _published(conn)
    conn.execute("DELETE FROM messages WHERE session_id='x'")
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert classify_summary_state(conn, "x")["summary_healthy"]
    append_message(conn, "x", "user", "A new turn after safe raw retention.")
    assert classify_summary_state(conn, "x")["degraded"]
    assert classify_summary_state(conn, "x", require_source_tail=False)["summary_healthy"]


def test_uncovered_raw_tail_from_another_session_does_not_stale_summary(conn):
    _published(conn)
    open_session(conn, "other")
    append_message(conn, "other", "user", "Unrelated unmaterialized source.")
    assert classify_summary_state(conn, "x")["summary_healthy"]


@pytest.mark.parametrize("source", ["legacy", "operator"])
def test_recovery_preserves_exact_curated_or_legacy_summary(conn, source):
    first, last, generation = _published(conn)
    curated = "Exact old summary. " * 50
    conn.execute("UPDATE sessions SET summary=?,summary_source=?", (curated, source))
    with db.transaction(conn):
        mark_summary_current(conn, "x", "Accepted updated automatic summary.", generation=generation, covered_message_id=last)
    row = conn.execute("SELECT summary,summary_source FROM sessions").fetchone()
    assert tuple(row) == (curated, source)


def test_summary_helpers_are_transactional_and_rollback_cleanly(conn):
    first, last, generation = _published(conn)
    before = tuple(conn.execute("SELECT * FROM sessions").fetchone())
    with pytest.raises(RuntimeError, match="transaction"):
        record_summary_failure(conn, "x", "summary_output_cap")
    with pytest.raises(RuntimeError, match="fenced"):
        mark_summary_current(conn, "x", "Valid text.", generation=generation, covered_message_id=last)
    with pytest.raises(ValueError):
        with db.transaction(conn):
            record_summary_failure(conn, "x", "summary_output_cap")
            raise ValueError("rollback")
    assert tuple(conn.execute("SELECT * FROM sessions").fetchone()) == before


@pytest.mark.parametrize("change", ["old_frontier", "future_frontier", "foreign_generation", "oversized", "boolean_frontier"])
def test_summary_publication_cannot_bless_mismatched_result(conn, change):
    first, last, generation = _published(conn)
    value, text, gen = last, "A valid replacement summary.", generation
    if change == "old_frontier": value = first
    if change == "future_frontier": value = 99999
    if change == "foreign_generation": gen = _generation("b")
    if change == "oversized": text = "x" * 501
    if change == "boolean_frontier": value = True
    before = tuple(conn.execute("SELECT * FROM sessions").fetchone())
    with pytest.raises(RuntimeError), db.transaction(conn):
        mark_summary_current(conn, "x", text, generation=gen, covered_message_id=value)
    assert tuple(conn.execute("SELECT * FROM sessions").fetchone()) == before


def test_new_summary_state_prevents_adoption_and_guard_heals(conn):
    open_session(conn, "empty")
    conn.execute("DROP TRIGGER summary_state_workspace_guard")
    db.initialize(conn)
    with db.transaction(conn):
        record_summary_failure(conn, "empty", "summary_shape_failure")
    assert not _session_is_pristine(conn, "empty")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE sessions SET source_workspace_id='owned' WHERE id='empty'")


def test_missing_summary_is_not_healthy_even_with_completed_items(conn):
    first, last = _source(conn)
    conn.execute("UPDATE sessions SET digest_published_generation=?,digest_published_message_id=?", (_generation(), last))
    assert classify_summary_state(conn, "x") == dict(summary_healthy=False, degraded=True, missing=True, malformed=False)


def test_dropped_v62_storage_is_not_silently_recreated_on_reopen(conn):
    conn.execute("DROP TRIGGER digest_staging_workspace_guard")
    conn.execute("DROP TABLE digest_staging")
    with pytest.raises(RuntimeError, match="v62"):
        db.initialize(conn)
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='digest_staging'").fetchone() is None


@pytest.mark.parametrize("reason", sorted(SUMMARY_FAILURE_REASONS))
def test_finite_reasons_agree_in_session_and_stage_storage(conn, reason):
    open_session(conn, "x")
    with db.transaction(conn):
        record_summary_failure(conn, "x", reason)
    conn.execute(
        "INSERT INTO digest_staging(session_id,generation,slice_key,summary,summary_failure_reason,"
        "procedures_json,episodes_json,source_sha256,cursor_before_offset,cursor_after_offset) "
        "VALUES ('x',?,'test','',?,'[]','[]',?,0,0)", (_generation(), reason, "0" * 64),
    )
    assert conn.execute("SELECT summary_failure_reason FROM digest_staging").fetchone()[0] == reason


@pytest.mark.parametrize("reason", ["completion_failure", "unrecognized_new_failure", "", "provider raw private message"])
def test_unapproved_reason_cannot_be_recorded_or_degraded(conn, reason):
    _published(conn)
    with pytest.raises(RuntimeError), db.transaction(conn):
        record_summary_failure(conn, "x", reason)
    # Simulate pre-existing corruption without changing application admission.
    conn.execute("PRAGMA ignore_check_constraints=ON")
    conn.execute("UPDATE sessions SET summary_failure_reason=?,summary_failure_count=1", (reason,))
    conn.execute("PRAGMA ignore_check_constraints=OFF")
    state = classify_summary_state(conn, "x")
    assert state["malformed"] and not state["summary_healthy"]


def test_frontier_storage_validator_rejects_unconstrained_lookalike(conn):
    conn.execute("DROP TRIGGER summary_state_workspace_guard")
    conn.execute("ALTER TABLE sessions DROP COLUMN summary_failure_count")
    conn.execute("ALTER TABLE sessions ADD COLUMN summary_failure_count INTEGER NOT NULL DEFAULT 0")
    with pytest.raises(RuntimeError, match="v62"):
        db.initialize(conn)
