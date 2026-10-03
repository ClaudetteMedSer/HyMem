"""Item publication stays atomic while rolling-summary gaps remain explicit."""
from __future__ import annotations

import json
import re
import sqlite3

import pytest

from hymem import HyMem
from hymem.dreaming import digest, runner, summary_state
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.dreaming.summary_state import classify_summary_state
from tests.test_digest_publication import PublicationLLM, _finish, _initial, _state
from tests.test_lossless_digest import _quiet_cfg


class DegradingLLM(PublicationLLM):
    def __init__(self, *, fail_at="first"):
        super().__init__(emit_slice_artifacts=True)
        self.fail_at = fail_at
        self.did_fail = False
        self.digest_records = []

    def complete(self, request):
        raw = super().complete(request)
        if not request.system.startswith(("You analyze one conversation session", "You re-read one conversation session")):
            return raw
        if raw == "not-json":
            return raw
        bounds = re.search(r"chars=(\d+):(\d+)/(\d+)", request.user)
        start, end, size = map(int, bounds.groups())
        should_fail = (
            (self.fail_at == "first" and start == 0)
            or (self.fail_at == "middle" and 0 < start < end < size)
            or (self.fail_at == "last" and end == size)
        )
        data = json.loads(raw)
        failed = bool(should_fail and not self.did_fail)
        if failed:
            data["summary"] = None
            self.did_fail = True
        self.digest_records.append((start, end, size, failed, request))
        return json.dumps(data)


def make_degrading(cfg, fail_at="first"):
    llm = DegradingLLM(fail_at=fail_at)
    hy = HyMem(_quiet_cfg(cfg, dream_digest_max_chars=300), llm=llm)
    hy.log_message("x", "user", "alpha " + "long content " * 50)
    hy.close_session("x")
    return hy, llm


def row(hy):
    return hy.conn.execute("SELECT * FROM sessions WHERE id='x'").fetchone()


@pytest.mark.parametrize("fail_at", ["first", "middle", "last"])
def test_failed_summary_at_any_slice_cannot_block_items_or_claim_summary_coverage(cfg, fail_at):
    hy, llm = make_degrading(cfg, fail_at)
    try:
        _finish(hy)
        state = row(hy)
        assert llm.did_fail
        assert state["digest_cursor_message_id"] == state["digest_published_message_id"] == state["coverage_message_id"]
        assert state["digest_cursor_prompt_version"] == state["digest_published_generation"]
        assert state["auto_summary"] is state["auto_summary_generation"] is state["auto_summary_message_id"] is None
        assert state["summary_failure_reason"] == "summary_shape_failure"
        assert state["summary_failure_count"] == 1
        assert state["digest_retry_count"] == state["digest_quarantined"] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert _state(hy)["episodes"] and _state(hy)["procedures"]
        assert digest.digest_staging_cursor_is_valid(hy.conn, "x")
        health = classify_summary_state(hy.conn, "x")
        assert health["degraded"] and health["missing"] and not health["malformed"]
        failed_index = next(i for i, record in enumerate(llm.digest_records) if record[3])
        for record in llm.digest_records[failed_index + 1:]:
            assert "prior automatic summary is stale" in record[4].user
        old_calls = len(llm.digest_records)
        hy.dream()
        assert len(llm.digest_records) == old_calls
    finally:
        hy.close()


def test_partial_reopen_carries_last_accepted_private_context_with_gap(cfg):
    hy, llm = make_degrading(cfg, "middle")
    try:
        while not llm.did_fail:
            report = hy.dream()
            assert report.digest_failures == 0
        state = row(hy)
        assert state["digest_cursor_partial_message_id"] is not None
        assert state["digest_published_message_id"] is None
        generation = state["digest_cursor_prompt_version"]
        private, failure = digest.load_digest_staged_summary_state(
            hy.conn, "x", generation,
            (state["digest_cursor_message_id"], state["digest_cursor_partial_message_id"], state["digest_cursor_offset"]),
        )
        assert "oldpublished" in private and failure == "summary_shape_failure"
        config = hy.config
        hy.close()
        hy = HyMem(config, llm=llm)
        hy.dream()
        assert row(hy)["digest_cursor_prompt_version"] == generation
        request = llm.digest_records[-1][4]
        assert "oldpublished" in request.user and "prior automatic summary is stale" in request.user
        _finish(hy)
        assert row(hy)["summary_failure_reason"] == "summary_shape_failure"
    finally:
        hy.close()


def test_forward_append_after_failed_summary_preserves_original_gap_and_items(cfg):
    hy, llm = make_degrading(cfg)
    try:
        _finish(hy)
        old_items = _state(hy)
        old_tail = row(hy)["digest_published_message_id"]
        new_mid = hy.log_message("x", "assistant", "beta new independent tail")
        hy.close_session("x")
        _finish(hy)
        state = row(hy)
        assert state["digest_published_message_id"] == new_mid > old_tail
        assert state["summary_failure_reason"] == "summary_shape_failure"
        assert state["summary_failure_count"] == 2
        assert state["auto_summary"] is None
        assert "prior automatic summary is stale" in llm.digest_records[-1][4].user
        assert set(old_items["episodes"]).issubset(set(_state(hy)["episodes"]))
        assert digest.digest_staging_cursor_is_valid(hy.conn, "x")
    finally:
        hy.close()


def test_failed_replacement_preserves_last_public_summary_and_its_exact_frontier(cfg):
    hy, _ = _initial(cfg)
    try:
        previous = row(hy)
        preserved = {key: previous[key] for key in (
            "summary", "summary_source", "auto_summary", "auto_summary_generation",
            "auto_summary_message_id", "auto_summary_partial_message_id", "auto_summary_message_offset",
        )}
        llm = DegradingLLM(fail_at="middle")
        llm.label = llm.artifact_label = "replacement"
        hy._llm = llm
        hy.conn.execute("UPDATE sessions SET digested_prompt_version=NULL WHERE id='x'")
        hy.dream()
        assert {key: row(hy)[key] for key in preserved} == preserved
        _finish(hy)
        state = row(hy)
        assert {key: state[key] for key in preserved} == preserved
        assert state["digest_published_generation"] != previous["digest_published_generation"]
        assert state["summary_failure_reason"] == "summary_shape_failure"
        assert all("replacement" in item[1] for item in _state(hy)["episodes"])
    finally:
        hy.close()


def test_full_rebuild_can_recover_summary_only_after_entire_new_walk_succeeds(cfg):
    hy, llm = make_degrading(cfg)
    try:
        _finish(hy)
        assert row(hy)["summary_failure_count"] == 1
        hy.conn.execute("UPDATE sessions SET digested_prompt_version=NULL WHERE id='x'")
        hy.dream()
        assert "prior automatic summary is stale" not in llm.digest_records[-1][4].user
        assert row(hy)["summary_failure_count"] == 1
        _finish(hy)
        state = row(hy)
        assert state["summary_failure_reason"] is None and state["summary_failure_count"] == 0
        assert state["auto_summary_generation"] == state["digest_published_generation"]
        assert state["auto_summary_message_id"] == state["digest_published_message_id"]
        assert classify_summary_state(hy.conn, "x")["summary_healthy"]
    finally:
        hy.close()


@pytest.mark.parametrize("source_kind", ["legacy", "operator"])
def test_unbounded_preserved_summary_is_never_truncated_into_new_publication(cfg, source_kind):
    hy, llm = make_degrading(cfg)
    legacy = "ONLY_SURVIVING_LEGACY_HISTORY " * 100
    try:
        hy.conn.execute("UPDATE sessions SET summary=?,summary_source=? WHERE id='x'", (legacy, source_kind))
        _finish(hy)
        assert row(hy)["summary"] == legacy
        assert row(hy)["summary_source"] == source_kind
        assert row(hy)["auto_summary"] is None
        if source_kind == "legacy":
            assert all(legacy in record[4].user for record in llm.digest_records)
        else:
            assert all(legacy not in record[4].user for record in llm.digest_records)
    finally:
        hy.close()


def test_empty_model_summary_does_not_promote_truncated_legacy_fallback(cfg):
    class EmptySummaryLLM(DegradingLLM):
        def complete(self, request):
            raw = super().complete(request)
            if request.system.startswith("You analyze one conversation session"):
                value = json.loads(raw)
                value["summary"] = ""
                return json.dumps(value)
            return raw

    llm = EmptySummaryLLM(fail_at=None)
    hy = HyMem(_quiet_cfg(cfg, dream_digest_max_chars=300), llm=llm)
    legacy = "LEGACY_ONLY_DO_NOT_TRUNCATE " * 100
    try:
        hy.log_message("x", "user", "alpha " + "long content " * 30)
        hy.close_session("x")
        hy.conn.execute("UPDATE sessions SET summary=?,summary_source='legacy' WHERE id='x'", (legacy,))
        _finish(hy)
        state = row(hy)
        assert state["summary"] == legacy and state["auto_summary"] is None
        assert state["summary_failure_reason"] == "summary_output_cap"
        assert state["digest_published_message_id"] == state["coverage_message_id"]
        assert all(legacy in record[4].user for record in llm.digest_records)
    finally:
        hy.close()


def test_degraded_item_publication_failure_rolls_back_everything(cfg):
    hy, llm = make_degrading(cfg)
    try:
        hy.conn.execute(
            "CREATE TEMP TRIGGER fail_summary_metadata_write "
            "BEFORE UPDATE OF summary_failure_count ON sessions WHEN new.id='x' "
            "BEGIN SELECT RAISE(ABORT,'injected summary metadata write failure'); END"
        )
        for _ in range(25):
            before = tuple(row(hy))
            staged = [tuple(r) for r in hy.conn.execute("SELECT * FROM digest_staging ORDER BY slice_key")]
            try:
                hy.dream()
            except sqlite3.IntegrityError as exc:
                assert "metadata write failure" in str(exc)
                assert tuple(row(hy)) == before
                assert [tuple(r) for r in hy.conn.execute("SELECT * FROM digest_staging ORDER BY slice_key")] == staged
                break
        else:
            pytest.fail("publication failure was not exercised")
        # Retry the same generation after removing the injected database fault.
        # Rebinding a helper (or changing its closure) would intentionally
        # change producer identity and require a fresh complete source walk.
        hy.conn.execute("DROP TRIGGER fail_summary_metadata_write")
        _finish(hy)
        assert row(hy)["summary_failure_count"] == 1
    finally:
        hy.close()


@pytest.mark.parametrize("fault", ["source", "gap_disappears", "items"])
def test_invalid_staging_still_holds_publication_and_is_never_summarized_as_success(cfg, fault):
    hy, llm = make_degrading(cfg)
    try:
        hy.dream()
        hy.dream()
        generation = row(hy)["digest_cursor_prompt_version"]
        if fault == "source":
            hy.conn.execute("UPDATE digest_staging SET source_sha256=?", ("0" * 64,))
        elif fault == "gap_disappears":
            hy.conn.execute("UPDATE digest_staging SET summary_failure_reason=NULL WHERE cursor_before_offset>0")
        else:
            hy.conn.execute("UPDATE digest_staging SET episodes_json='[42]'")
        with pytest.raises(RuntimeError):
            digest.load_completed_digest_slices(hy.conn, "x", generation, require_complete=False)
        assert not digest.digest_staging_cursor_is_valid(hy.conn, "x")
        assert row(hy)["digest_published_message_id"] is None
        assert row(hy)["auto_summary"] is None
    finally:
        hy.close()


def test_summary_metadata_implementation_is_bound_to_digest_semantic_identity(monkeypatch):
    llm = PublicationLLM()
    before = semantic_generation_suffix("digest", llm)
    original = summary_state._integer

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(summary_state, "_integer", replacement)
    assert semantic_generation_suffix("digest", llm) != before


def test_full_replacement_repairs_only_known_malformed_published_marker(cfg):
    hy, _ = _initial(cfg, long=False)
    try:
        previous = row(hy)
        hy.conn.execute(
            "UPDATE sessions SET digest_published_generation=? WHERE id='x'",
            (previous["digest_published_generation"] + "-untrusted-suffix",),
        )
        assert classify_summary_state(hy.conn, "x")["malformed"]
        _finish(hy)
        current = row(hy)
        assert current["digest_published_generation"] != previous["digest_published_generation"]
        assert current["auto_summary_generation"] == current["digest_published_generation"]
        assert classify_summary_state(hy.conn, "x")["summary_healthy"]
    finally:
        hy.close()


@pytest.mark.parametrize("damage", [
    "summary_generation", "foreign_cursor", "future_cursor", "summary_offset",
    "failure_count", "missing_text", "oversize_text", "published_future",
])
def test_bad_published_marker_does_not_authorize_repair_of_other_corruption(cfg, damage):
    hy, _ = _initial(cfg, long=False)
    try:
        previous = row(hy)
        hy.conn.execute(
            "UPDATE sessions SET digest_published_generation=? WHERE id='x'",
            (previous["digest_published_generation"] + "-untrusted-suffix",),
        )
        if damage == "summary_generation":
            hy.conn.execute("UPDATE sessions SET auto_summary_generation='untrusted' WHERE id='x'")
        elif damage == "foreign_cursor":
            foreign = hy.log_message("foreign", "user", "independent foreign text")
            hy.conn.execute("UPDATE sessions SET auto_summary_message_id=? WHERE id='x'", (foreign,))
        elif damage == "future_cursor":
            hy.conn.execute("UPDATE sessions SET auto_summary_message_id=coverage_message_id+100 WHERE id='x'")
        elif damage == "summary_offset":
            hy.conn.execute("UPDATE sessions SET auto_summary_message_offset=1 WHERE id='x'")
        elif damage == "failure_count":
            hy.conn.execute("UPDATE sessions SET summary_failure_count=1 WHERE id='x'")
        elif damage == "missing_text":
            hy.conn.execute("UPDATE sessions SET auto_summary=NULL WHERE id='x'")
        elif damage == "oversize_text":
            hy.conn.execute("UPDATE sessions SET auto_summary=? WHERE id='x'", ("x" * 501,))
        elif damage == "published_future":
            hy.conn.execute("UPDATE sessions SET digest_published_message_id=coverage_message_id+100 WHERE id='x'")
        damaged = tuple(row(hy))
        with pytest.raises(RuntimeError, match="summary metadata is malformed"):
            hy.dream()
        assert tuple(row(hy)) == damaged
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging WHERE session_id='x'").fetchone()[0] == 0
    finally:
        hy.close()
