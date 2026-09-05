"""Regression contract for v38's lossless, cursor-driven session digest."""

from __future__ import annotations

import dataclasses
import json
import re
import sqlite3

import pytest

from hymem import HyMem, StubEmbeddingClient
from hymem.core import db as core_db
from hymem.core.message_records import (
    MESSAGE_CONTENT_HASH_VERSION,
    MESSAGE_RECORD_VERSION,
    encode_message_record,
    message_content_hash,
)
from hymem.dreaming.chunks import _chunk_id
from hymem.dreaming.digest import (
    digest_config_version,
    digest_generation_matches_config,
)
from hymem.dreaming.lossless import (
    LOSSLESS_COVERAGE_VERSION,
    coverage_chunk_id,
    covered_messages_after,
    materialize_message_coverage,
)
from hymem.dreaming.aggregate import load_clusterable_episodes
from hymem.dreaming.embeddings import fetch_episode_embeddings
from hymem.dreaming.message_coverage import (
    record_message_coverage,
    release_message_coverage,
)
from hymem.dreaming.summary import effective_session_summary
from hymem.dreaming.user_profile import profile_user_tail_message_id
from hymem.dreaming.retention import prune_episodes_and_procedures
from hymem.extraction.llm import LLMRequest
from hymem.query.augment import _episode_search


class RollingLLM:
    def __init__(
        self,
        *,
        emit_slice_artifacts: bool = False,
        emit_two_slice_artifacts: bool = False,
        empty_summary: bool = False,
    ):
        self.calls: list[LLMRequest] = []
        self.successful_digest_calls: list[LLMRequest] = []
        self.fail_next_digest = False
        self.emit_slice_artifacts = emit_slice_artifacts
        self.emit_two_slice_artifacts = emit_two_slice_artifacts
        self.emit_only_first_slice = False
        self.artifact_label = ""
        self.empty_summary = empty_summary

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if request.system.startswith((
            "You analyze one conversation session",
            "You re-read one conversation session",
        )):
            if self.fail_next_digest:
                self.fail_next_digest = False
                return "not-json"
            self.successful_digest_calls.append(request)
            prior_match = re.search(
                r"Prior automatic session summary \(may be empty\):\n"
                r'"""\n(.*?)\n"""',
                request.user,
                re.S,
            )
            prior = prior_match.group(1).strip() if prior_match else ""
            lower = request.user.lower()
            facts = []
            if "alpha" in lower or "alpha" in prior.lower():
                facts.append("alpha")
            if "beta" in lower or "beta" in prior.lower():
                facts.append("beta")
            summary = "" if self.empty_summary else (
                "Remembered " + " and ".join(facts) + " across the session."
                if facts else "Processed the complete conversation material."
            )
            episodes: list[dict] = []
            procedures: list[dict] = []
            if self.emit_slice_artifacts:
                chunk = re.search(r"\[chunk (msgcov_[0-9a-f]+)\]", request.user)
                offset = re.search(r"chars=(\d+):(\d+)/(\d+)", request.user)
                if (
                    chunk
                    and offset
                    and (
                        not self.emit_only_first_slice
                        or int(offset.group(1)) == 0
                    )
                ):
                    start = int(offset.group(1))
                    label = f"{self.artifact_label} " if self.artifact_label else ""
                    episodes = [{
                        "title": f"{label}Slice {start}",
                        "summary": (
                            f"{label}Processed the message slice beginning at {start}."
                        ),
                        "outcome": "informational",
                        "key_entities": [],
                        "chunk_ids": [chunk.group(1)],
                    }]
                    procedures = [{
                        "name": f"Handle slice {start}",
                        "description": "Processes one bounded message slice.",
                        "steps": [{"order": 1, "action": f"Process offset {start}", "tool": None}],
                        "triggers": ["slice"],
                        "entities_involved": [],
                        "chunk_ids": [chunk.group(1)],
                    }]
                    if self.emit_two_slice_artifacts:
                        episodes.append({
                            "title": f"{label}Second event {start}",
                            "summary": (
                                f"{label}A separate event shares this exact input range."
                            ),
                            "outcome": "resolved",
                            "key_entities": [],
                            "chunk_ids": [chunk.group(1)],
                        })
                        procedures.append({
                            "name": f"Verify slice {start}",
                            "description": "Verifies the independently derived slice output.",
                            "steps": [{"order": 1, "action": f"Verify offset {start}", "tool": None}],
                            "triggers": ["verify slice"],
                            "entities_involved": [],
                            "chunk_ids": [chunk.group(1)],
                        })
            return json.dumps({
                "episodes": episodes,
                "summary": summary,
                "procedures": procedures,
            })
        if "single pass" in request.system:
            return '{"triples":[],"markers":[]}'
        return "[]"


class FailingDigestLLM:
    def __init__(self, *, healthy: bool = False):
        self.healthy = healthy
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if request.system.startswith("You analyze one conversation session"):
            if self.healthy:
                return '{"episodes":[],"summary":"","procedures":[]}'
            return '{"episodes":[],"summary":"","procedures":[],"error":"refused"}'
        if "single pass" in request.system:
            return '{"triples":[],"markers":[]}'
        return "[]"


def _quiet_cfg(cfg, **changes):
    return dataclasses.replace(
        cfg,
        aggregation_nodes_enabled=False,
        facts_extraction_enabled=False,
        profile_extraction_enabled=False,
        **changes,
    )


def test_digest_failure_adapts_quarantines_and_output_knob_reopens(cfg):
    bad = FailingDigestLLM()
    config = _quiet_cfg(
        cfg,
        dream_digest_max_chars=1200,
        digest_extraction_max_attempts=2,
    )
    hy = HyMem(config, llm=bad)
    sid = "digest-quarantine"
    try:
        hy.log_message(sid, "user", "boundary " + ("x" * 3000))
        hy.close_session(sid)
        first = hy.dream()
        second = hy.dream()
        digest_calls = [
            call for call in bad.calls
            if call.system.startswith("You analyze one conversation session")
        ]
        assert first.digest_failures == second.digest_failures == 1
        assert first.budget_exhausted is True
        assert second.digest_quarantined == 1
        assert len(digest_calls) == 2
        assert len(digest_calls[1].user) < len(digest_calls[0].user)
        state = hy.conn.execute(
            "SELECT digest_retry_count, digest_quarantined, "
            "digest_cursor_message_id FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert state["digest_retry_count"] == 2
        assert state["digest_quarantined"] == 1
        assert state["digest_cursor_message_id"] is None
        before = len(digest_calls)
        skipped = hy.dream()
        assert skipped.digest_quarantined == 1
        assert len([
            call for call in bad.calls
            if call.system.startswith("You analyze one conversation session")
        ]) == before
        assert hy.dream_status()["quarantined_digests"] == 1
    finally:
        hy.close()

    # Raising an output-shaping knob changes the producer/retry salt. The
    # exact held source reopens immediately and a successful result clears the
    # old quarantine rather than waiting for message ids to change.
    good = FailingDigestLLM(healthy=True)
    healed_config = dataclasses.replace(
        config, dream_digest_max_tokens=config.dream_digest_max_tokens + 1
    )
    healed = HyMem(healed_config, llm=good)
    try:
        report = healed.dream()
        state = healed.conn.execute(
            "SELECT digest_retry_count, digest_quarantined, "
            "digest_cursor_message_id, digest_cursor_partial_message_id, "
            "coverage_message_id "
            "FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert report.digest_failures == 0
        assert state["digest_retry_count"] == 0
        assert state["digest_quarantined"] == 0
        assert state["digest_cursor_partial_message_id"] is not None
        # A bounded first success is allowed to remain partial; it must keep
        # completion loops alive rather than claim the tail.
        assert report.budget_exhausted is True
    finally:
        healed.close()


def test_future_digest_cursor_rewinds_instead_of_suppressing_source(cfg):
    llm = RollingLLM()
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "future-digest-cursor"
        message_id = hy.log_message(sid, "user", "alpha durable source")
        hy.close_session(sid)
        hy.dream()
        initial_calls = len(_digest_calls(llm))
        hy.conn.execute(
            "UPDATE sessions SET digest_cursor_message_id = ? WHERE id = ?",
            (message_id + 1000, sid),
        )

        report = hy.dream()
        assert report.digest_failures == 0
        assert len(_digest_calls(llm)) == initial_calls + 1
        assert "alpha durable source" in _digest_calls(llm)[-1].user
        state = hy.conn.execute(
            "SELECT digest_cursor_message_id, coverage_message_id "
            "FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert state["digest_cursor_message_id"] == state["coverage_message_id"]
    finally:
        hy.close()


def _digest_calls(llm: RollingLLM) -> list[LLMRequest]:
    return [
        call for call in llm.calls
        if call.system.startswith((
            "You analyze one conversation session",
            "You re-read one conversation session",
        ))
    ]


def test_public_ingest_immediately_covers_every_role_exactly(cfg):
    llm = RollingLLM()
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "all_roles"
        turns = [
            ("system", "system alpha"),
            ("user", "u"),
            ("assistant", ""),
            ("tool", "tool beta\nsecond line\u2028paragraph\u0085next"),
        ]
        ids = hy.log_messages(sid, turns)

        rows = hy.conn.execute(
            "SELECT c.id, c.start_message_id, c.end_message_id, c.text, "
            "c.chunk_kind, mc.message_id FROM chunks c "
            "JOIN message_retention_coverage mc ON mc.chunk_id = c.id "
            "WHERE c.session_id = ? ORDER BY mc.message_id",
            (sid,),
        ).fetchall()
        assert len(rows) == len(turns)
        for row, message_id, (role, content) in zip(rows, ids, turns):
            assert row["id"] == coverage_chunk_id(sid, message_id)
            assert row["id"] != _chunk_id(sid, message_id, message_id)
            assert row["chunk_kind"] == "coverage"
            assert row["start_message_id"] == row["end_message_id"] == message_id
            assert row["text"] == encode_message_record(
                message_id=message_id, role=role, content=content
            )

        hy.close_session(sid)
        hy.dream()
        call = _digest_calls(llm)[0].user
        assert "system alpha" in call and "tool beta" in call
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM processed_chunks pc "
            "JOIN chunks c ON c.id = pc.chunk_id WHERE c.chunk_kind = 'coverage'"
        ).fetchone()["c"] == 0
    finally:
        hy.close()


def test_recognized_rows_above_or_without_producer_frontier_are_unreadable(cfg):
    llm = RollingLLM()
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "sparse-recognized-frontier"
        first = hy.log_message(sid, "user", "alpha authorized")
        rogue = hy.log_message(sid, "user", "beta above frontier")
        # Simulate a sparse/imported recognized record whose proof exists but
        # whose ordered producer frontier never reached it.
        hy.conn.execute(
            "UPDATE sessions SET coverage_message_id = ? WHERE id = ?",
            (first, sid),
        )
        # Remove the raw tail so dream() cannot legitimately re-materialize it
        # and advance the producer frontier.  The adversarial input is the
        # durable, recognized-but-sparse coverage artifact by itself.
        hy.conn.execute("DELETE FROM messages WHERE id = ?", (rogue,))
        assert [message.message_id for message in covered_messages_after(
            hy.conn, sid, None
        )] == [first]
        assert profile_user_tail_message_id(hy.conn, sid) == first

        hy.close_session(sid)
        hy.dream()
        sent = _digest_calls(llm)[0].user
        assert "alpha authorized" in sent
        assert "beta above frontier" not in sent

        # No producer frontier authorizes no ordered input at all.
        hy.conn.execute(
            "UPDATE sessions SET coverage_message_id = NULL WHERE id = ?", (sid,)
        )
        assert covered_messages_after(hy.conn, sid, None) == []
        assert profile_user_tail_message_id(hy.conn, sid) is None

        # Once the producer-established frontier advances, the exact same
        # reviewed artifact becomes readable; no ledger MAX inference occurs.
        hy.conn.execute(
            "UPDATE sessions SET coverage_message_id = ? WHERE id = ?",
            (rogue, sid),
        )
        assert [message.message_id for message in covered_messages_after(
            hy.conn, sid, first
        )] == [rogue]
        assert profile_user_tail_message_id(hy.conn, sid) == rogue
    finally:
        hy.close()


def test_coverage_artifacts_never_enter_chunk_fts_or_change_bm25(cfg):
    hy = HyMem(_quiet_cfg(cfg), llm=RollingLLM())
    try:
        sid = "fts_isolation"
        source_id = hy.log_message(sid, "user", "coverageonlytoken commonword")
        hy.conn.execute(
            "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
            "salience_reason, text, chunk_kind) "
            "VALUES ('extract_fts', ?, ?, ?, 'test', "
            "'commonword extractiontoken', 'extraction')",
            (sid, source_id, source_id),
        )

        def _score() -> float:
            return float(hy.conn.execute(
                "SELECT bm25(chunks_fts) AS score FROM chunks_fts "
                "WHERE chunks_fts MATCH 'commonword'"
            ).fetchone()["score"])

        before = _score()
        hy.log_messages(
            sid,
            [("system", f"coverageonlytoken commonword {i}") for i in range(40)],
        )
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks_fts "
            "WHERE chunks_fts MATCH 'coverageonlytoken'"
        ).fetchone()["c"] == 0
        assert _score() == before

        # VACUUM repair must stay selective too; FTS5's ordinary external-
        # content rebuild would accidentally index all coverage rows.
        core_db.resync_rowid_shadows(hy.conn)
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks_fts "
            "WHERE chunks_fts MATCH 'coverageonlytoken'"
        ).fetchone()["c"] == 0
        assert _score() == before
    finally:
        hy.close()


@pytest.mark.parametrize(
    "tail",
    [
        [("assistant", "beta")],
        [("user", "beta")],
    ],
)
def test_short_or_assistant_only_tail_reopens_digest_and_rolls_summary(cfg, tail):
    llm = RollingLLM()
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "short_tail"
        hy.log_messages(sid, [("user", "alpha is the initial decision")])
        hy.close_session(sid)
        hy.dream()
        first = hy.conn.execute(
            "SELECT digest_cursor_message_id, auto_summary FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert "alpha" in first["auto_summary"]

        hy.log_messages(sid, tail)
        hy.close_session(sid)
        hy.dream()
        second = hy.conn.execute(
            "SELECT digest_cursor_message_id, auto_summary FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert second["digest_cursor_message_id"] > first["digest_cursor_message_id"]
        assert "alpha" in second["auto_summary"] and "beta" in second["auto_summary"]
        last = _digest_calls(llm)[-1].user
        assert "beta" in last
        assert "Remembered alpha" in last
    finally:
        hy.close()


def test_all_short_and_mixed_skipped_turns_are_all_digest_input(cfg):
    llm = RollingLLM()
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "mixed_short"
        hy.log_messages(sid, [
            ("user", "hi"),
            ("assistant", "ok"),
            ("user", "alpha is a long substantive decision worth retaining"),
            ("assistant", "tiny middle"),
            ("user", "beta is another long substantive decision worth retaining"),
        ])
        hy.close_session(sid)
        hy.dream()
        sent = _digest_calls(llm)[0].user
        for text in ("hi", "ok", "alpha", "tiny middle", "beta"):
            assert text in sent
        state = hy.conn.execute(
            "SELECT coverage_message_id, digest_cursor_message_id "
            "FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert state["digest_cursor_message_id"] == state["coverage_message_id"]

        before = len(_digest_calls(llm))
        hy.dream()
        assert len(_digest_calls(llm)) == before
    finally:
        hy.close()


@pytest.mark.parametrize(
    "corrupt_column",
    ["digest_cursor_prompt_version", "digest_published_generation"],
)
def test_malformed_digest_generation_cannot_claim_steady_state(
    cfg, corrupt_column,
):
    """A prefix-compatible but malformed imported stamp forces replay."""
    llm = RollingLLM()
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = f"malformed-digest-generation-{corrupt_column}"
        hy.log_message(sid, "user", "alpha must remain covered")
        hy.close_session(sid)
        hy.dream()
        before = len(_digest_calls(llm))
        row = hy.conn.execute(
            "SELECT digest_cursor_prompt_version, "
            "digest_published_generation FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        valid = row[corrupt_column]
        assert valid is not None
        forged = valid + "-untrusted-suffix"
        hy.conn.execute(
            f"UPDATE sessions SET {corrupt_column} = ? WHERE id = ?",
            (forged, sid),
        )

        hy.dream()
        assert len(_digest_calls(llm)) == before + 1
        repaired = hy.conn.execute(
            "SELECT digest_cursor_prompt_version, "
            "digest_published_generation FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        config = digest_config_version(
            prompt_version=hy.config.prompt_version,
            episode_prompt_version=None,
            max_chars=hy.config.dream_digest_max_chars,
            max_tokens=hy.config.dream_digest_max_tokens,
            max_episodes=None,
        )
        assert digest_generation_matches_config(
            repaired["digest_cursor_prompt_version"], config
        )
        assert digest_generation_matches_config(
            repaired["digest_published_generation"], config
        )
        assert forged not in tuple(repaired)
    finally:
        hy.close()


def test_all_short_initial_and_append_accept_explicit_empty_summary(cfg):
    """Small talk may validly yield summary="" and still complete coverage."""
    llm = RollingLLM(empty_summary=True)
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "all_short_empty"
        first_ids = hy.log_messages(sid, [("user", "hi"), ("assistant", "ok")])
        hy.close_session(sid)
        report = hy.dream()
        row = hy.conn.execute(
            "SELECT coverage_message_id, digest_cursor_message_id, "
            "auto_summary, auto_summary_message_id, summary_source "
            "FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert report.digest_failures == 0
        assert row["coverage_message_id"] == row["digest_cursor_message_id"] == first_ids[-1]
        assert row["auto_summary"] == ""
        assert row["auto_summary_message_id"] == first_ids[-1]
        assert row["summary_source"] == "auto"

        appended = hy.log_message(sid, "tool", "k")
        hy.close_session(sid)
        second = hy.dream()
        row = hy.conn.execute(
            "SELECT digest_cursor_message_id, auto_summary_message_id "
            "FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert second.digest_failures == 0
        assert row["digest_cursor_message_id"] == appended
        assert row["auto_summary_message_id"] == appended
        assert len(_digest_calls(llm)) == 2
    finally:
        hy.close()


def test_oversized_message_resumes_exact_offsets_and_retry_is_idempotent(cfg):
    llm = RollingLLM(emit_slice_artifacts=True)
    hy = HyMem(_quiet_cfg(cfg, dream_digest_max_chars=300), llm=llm)
    try:
        sid = "oversized"
        content = "ALPHA🙂e\u0301\n" + ("0123456789界" * 140) + "BETA"
        message_id = hy.log_message(sid, "user", content)
        hy.close_session(sid)

        first_report = hy.dream()
        first = dict(hy.conn.execute(
            "SELECT digest_cursor_message_id, digest_cursor_partial_message_id, "
            "digest_cursor_offset FROM sessions WHERE id = ?", (sid,),
        ).fetchone())
        assert first["digest_cursor_message_id"] is None
        assert first["digest_cursor_partial_message_id"] == message_id
        assert first["digest_cursor_offset"] > 0
        assert first_report.budget_exhausted is True

        llm.fail_next_digest = True
        hy.dream()
        failed = dict(hy.conn.execute(
            "SELECT digest_cursor_message_id, digest_cursor_partial_message_id, "
            "digest_cursor_offset FROM sessions WHERE id = ?", (sid,),
        ).fetchone())
        assert failed == first

        for _ in range(30):
            hy.dream()
            state = hy.conn.execute(
                "SELECT digest_cursor_message_id, digest_cursor_offset, "
                "digested_prompt_version FROM sessions WHERE id = ?", (sid,),
            ).fetchone()
            if state["digested_prompt_version"] == hy.config.prompt_version:
                break
        assert state["digest_cursor_message_id"] == message_id
        assert state["digest_cursor_offset"] == 0

        intervals = []
        for request in llm.successful_digest_calls:
            match = re.search(rf"message {message_id} role=user chars=(\d+):(\d+)/(\d+)", request.user)
            if match:
                intervals.append(tuple(map(int, match.groups())))
        assert intervals[0][0] == 0
        assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
        assert intervals[-1][1] == intervals[-1][2] == len(content)
        recovered = []
        for request, (start, end, _total) in zip(
            [r for r in llm.successful_digest_calls if f"message {message_id} " in r.user],
            intervals,
        ):
            body = request.user.split(
                f"chars={start}:{end}/{len(content)}]\n", 1
            )[1].split('\n"""\n\nReturn', 1)[0]
            recovered.append(body)
        assert "".join(recovered) == content
        assert len(hy.conn.execute(
            "SELECT id FROM episodes WHERE session_id = ?", (sid,),
        ).fetchall()) == len(intervals)
        assert len(hy.conn.execute(
            "SELECT id FROM procedures WHERE session_id = ?", (sid,),
        ).fetchall()) == len(intervals)
    finally:
        hy.close()


def test_multiple_outputs_sharing_one_digest_range_do_not_overwrite(cfg):
    llm = RollingLLM(
        emit_slice_artifacts=True,
        emit_two_slice_artifacts=True,
    )
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "same_range_outputs"
        hy.log_message(sid, "user", "alpha contains two independent actions")
        hy.close_session(sid)
        hy.dream()
        episodes = hy.conn.execute(
            "SELECT id FROM episodes WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()
        procedures = hy.conn.execute(
            "SELECT id FROM procedures WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()
        assert len({row["id"] for row in episodes}) == 2
        assert len({row["id"] for row in procedures}) == 2
    finally:
        hy.close()


def test_same_config_full_redigest_replaces_shorter_authoritative_result(cfg):
    """A successful rebuild can shrink two episodes to one, then to zero.

    Invalidating only the published stamp is the established explicit
    re-digest contract.  Each full walk must therefore receive a distinct
    generation even though its prompt/config and stable slice IDs are equal.
    """
    llm = RollingLLM(
        emit_slice_artifacts=True,
        emit_two_slice_artifacts=True,
    )
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "same_config_shrink"
        hy.log_message(sid, "user", "alpha has two initially extracted events")
        hy.close_session(sid)
        hy.dream()
        initial = hy.conn.execute(
            "SELECT id, digest_generation FROM episodes "
            "WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()
        assert len(initial) == 2
        initial_ids = {row["id"] for row in initial}
        initial_generation = {row["digest_generation"] for row in initial}
        assert len(initial_generation) == 1

        hy.conn.execute(
            "UPDATE sessions SET digested_prompt_version = NULL WHERE id = ?",
            (sid,),
        )
        llm.emit_two_slice_artifacts = False
        hy.dream()
        shorter = hy.conn.execute(
            "SELECT id, title, digest_generation FROM episodes "
            "WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()
        assert [row["title"] for row in shorter] == ["Slice 0"]
        assert shorter[0]["id"] not in initial_ids
        shorter_generation = {row["digest_generation"] for row in shorter}
        assert len(shorter_generation) == 1
        assert shorter_generation != initial_generation

        # A valid digest object with episodes=[] is authoritative.  Cleanup is
        # delayed until this replacement walk reaches the coverage tail, at
        # which point retaining the old row would be a stale retrieval result.
        hy.conn.execute(
            "UPDATE sessions SET digested_prompt_version = NULL WHERE id = ?",
            (sid,),
        )
        llm.emit_slice_artifacts = False
        hy.dream()
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM episodes WHERE session_id = ?", (sid,),
        ).fetchone()["c"] == 0
    finally:
        hy.close()


def test_failed_same_config_rebuild_keeps_last_complete_episode_set(cfg):
    llm = RollingLLM(
        emit_slice_artifacts=True,
        emit_two_slice_artifacts=True,
    )
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "same_config_failed_rebuild"
        hy.log_message(sid, "user", "alpha initially yields two events")
        hy.close_session(sid)
        hy.dream()
        before = [dict(row) for row in hy.conn.execute(
            "SELECT id, title, summary, digest_generation FROM episodes "
            "WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()]
        assert len(before) == 2

        hy.conn.execute(
            "UPDATE sessions SET digested_prompt_version = NULL WHERE id = ?",
            (sid,),
        )
        llm.emit_two_slice_artifacts = False
        llm.fail_next_digest = True
        report = hy.dream()
        after = [dict(row) for row in hy.conn.execute(
            "SELECT id, title, summary, digest_generation FROM episodes "
            "WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()]
        assert report.digest_failures == 1
        assert after == before
        assert hy.conn.execute(
            "SELECT digested_prompt_version FROM sessions WHERE id = ?", (sid,),
        ).fetchone()["digested_prompt_version"] is None
    finally:
        hy.close()


def test_oversized_same_config_rebuild_defers_stale_cleanup_until_tail(cfg):
    llm = RollingLLM(emit_slice_artifacts=True)
    llm.artifact_label = "oldpublished"
    hy = HyMem(_quiet_cfg(cfg, dream_digest_max_chars=300), llm=llm)
    try:
        sid = "oversized_same_config_rebuild"
        hy.log_message(
            sid,
            "user",
            "alpha" + ("0123456789界" * 130) + "omega",
        )
        hy.close_session(sid)
        for _ in range(30):
            hy.dream()
            published = hy.conn.execute(
                "SELECT digested_prompt_version FROM sessions WHERE id = ?",
                (sid,),
            ).fetchone()["digested_prompt_version"]
            if published == hy.config.prompt_version:
                break
        else:
            pytest.fail("initial oversized digest did not reach its tail")

        old_rows = [dict(row) for row in hy.conn.execute(
            "SELECT * FROM episodes "
            "WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()]
        old_ids = {row["id"] for row in old_rows}
        old_generation = {row["digest_generation"] for row in old_rows}
        assert len(old_ids) > 1 and len(old_generation) == 1
        marker = hy.conn.execute(
            "SELECT digest_published_generation FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()["digest_published_generation"]
        assert old_generation == {marker}
        before_ranking = [
            (hit.episode_id, hit.score)
            for hit in _episode_search(hy.conn, "oldpublished", top_k=50)
        ]
        assert {episode["id"] for episode in load_clusterable_episodes(hy.conn)} == old_ids

        hy.conn.execute(
            "UPDATE sessions SET digested_prompt_version = NULL WHERE id = ?",
            (sid,),
        )
        llm.emit_only_first_slice = True
        llm.artifact_label = "newstaged"
        first_rebuild = hy.dream()
        assert first_rebuild.budget_exhausted is True
        during = [dict(row) for row in hy.conn.execute(
            "SELECT * FROM episodes "
            "WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()]
        # No eager cleanup or UPSERT into published ids: every row in the last
        # complete result remains byte-for-byte unchanged, while the successful
        # first replacement slice is staged under its new build generation.
        still_published = [row for row in during if row["id"] in old_ids]
        assert still_published == old_rows
        staged = [row for row in during if row["id"] not in old_ids]
        assert len(staged) == 1
        assert staged[0]["digest_generation"] not in old_generation
        active = hy.conn.execute(
            "SELECT digest_cursor_prompt_version, digest_published_generation "
            "FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert active["digest_cursor_prompt_version"] == staged[0]["digest_generation"]
        assert active["digest_published_generation"] == marker

        # Staging is physical but unpublished: it gets no FTS posting, changes
        # no BM25 ranking, is excluded from vector work, and cannot enter the
        # Phase-2 aggregation loader.
        core_db.resync_rowid_shadows(hy.conn)
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM episodes_fts "
            "WHERE episodes_fts MATCH 'newstaged'"
        ).fetchone()["c"] == 0
        assert _episode_search(hy.conn, "newstaged", top_k=50) == []
        assert [
            (hit.episode_id, hit.score)
            for hit in _episode_search(hy.conn, "oldpublished", top_k=50)
        ] == before_ranking
        assert {episode["id"] for episode in load_clusterable_episodes(hy.conn)} == old_ids
        pending = fetch_episode_embeddings(hy.conn, StubEmbeddingClient())
        assert pending is not None and set(pending.ids) == old_ids

        # A paused rebuild may outlive the configured age cutoff. Retention can
        # prune abandoned generations, but never its active cursor generation.
        hy.conn.execute(
            "UPDATE episodes SET created_at = '2020-01-01' "
            "WHERE digest_generation = ?",
            (active["digest_cursor_prompt_version"],),
        )
        prune_episodes_and_procedures(
            hy.conn,
            dataclasses.replace(hy.config, episode_retention_days=1),
        )
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM episodes WHERE digest_generation = ?",
            (active["digest_cursor_prompt_version"],),
        ).fetchone()["c"] == 1

        llm.fail_next_digest = True
        failed_snapshot = [dict(row) for row in hy.conn.execute(
            "SELECT * FROM episodes WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()]
        failed = hy.dream()
        assert failed.digest_failures == 1
        assert [dict(row) for row in hy.conn.execute(
            "SELECT * FROM episodes "
            "WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()] == failed_snapshot

        for _ in range(30):
            hy.dream()
            published = hy.conn.execute(
                "SELECT digested_prompt_version FROM sessions WHERE id = ?",
                (sid,),
            ).fetchone()["digested_prompt_version"]
            if published == hy.config.prompt_version:
                break
        else:
            pytest.fail("replacement oversized digest did not reach its tail")

        final_rows = hy.conn.execute(
            "SELECT id, digest_generation FROM episodes "
            "WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()
        assert len(final_rows) == 1
        assert final_rows[0]["id"] not in old_ids
        assert final_rows[0]["digest_generation"] not in old_generation
        assert hy.conn.execute(
            "SELECT digest_published_generation FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()["digest_published_generation"] == final_rows[0]["digest_generation"]
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM episodes_fts "
            "WHERE episodes_fts MATCH 'oldpublished'"
        ).fetchone()["c"] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM episodes_fts "
            "WHERE episodes_fts MATCH 'newstaged'"
        ).fetchone()["c"] == 1
        assert [
            episode["id"] for episode in load_clusterable_episodes(hy.conn)
        ] == [final_rows[0]["id"]]
    finally:
        hy.close()


def test_sparse_unapproved_v37_proofs_never_become_digest_frontier(cfg):
    """Exact retention rows are not automatically a complete ordered stream."""
    hy = HyMem(_quiet_cfg(cfg), llm=RollingLLM())
    try:
        sid = "sparse_legacy"
        hy.conn.execute("INSERT INTO sessions(id) VALUES (?)", (sid,))
        for message_id, content in ((1, "first"), (3, "third")):
            hy.conn.execute(
                "INSERT INTO messages(id, session_id, role, content) "
                "VALUES (?, ?, 'user', ?)",
                (message_id, sid, content),
            )
            chunk_id = f"legacy_{message_id}"
            hy.conn.execute(
                "INSERT INTO chunks(id, session_id, start_message_id, "
                "end_message_id, salience_reason, text, chunk_kind) "
                "VALUES (?, ?, ?, ?, 'legacy-proof', ?, 'coverage')",
                (
                    chunk_id,
                    sid,
                    message_id,
                    message_id,
                    encode_message_record(
                        message_id=message_id, role="user", content=content
                    ),
                ),
            )
            record_message_coverage(
                hy.conn,
                message_id=message_id,
                chunk_id=chunk_id,
                coverage_version="caller-defined-v37-proof",
            )
        hy.conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))

        assert materialize_message_coverage(hy.conn, sid) == 0
        assert hy.conn.execute(
            "SELECT coverage_message_id FROM sessions WHERE id = ?", (sid,),
        ).fetchone()["coverage_message_id"] is None
        assert covered_messages_after(hy.conn, sid, None) == []
    finally:
        hy.close()


def test_ordered_proofs_are_immutable_and_append_uses_cursor_range(cfg):
    hy = HyMem(_quiet_cfg(cfg), llm=RollingLLM())
    try:
        sid = "cursor_hot_path"
        ids = hy.log_messages(
            sid,
            [("user", f"historical turn {i}") for i in range(300)],
        )
        first_chunk = coverage_chunk_id(sid, ids[0])
        with pytest.raises(RuntimeError, match="ordered digest coverage is immutable"):
            release_message_coverage(
                hy.conn,
                message_id=ids[0],
                chunk_id=first_chunk,
                coverage_version=LOSSLESS_COVERAGE_VERSION,
            )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="ordered digest coverage is immutable",
        ):
            hy.conn.execute(
                "DELETE FROM message_retention_coverage WHERE message_id = ? "
                "AND chunk_id = ? AND coverage_version = ?",
                (ids[0], first_chunk, LOSSLESS_COVERAGE_VERSION),
            )

        statements: list[str] = []
        hy.conn.set_trace_callback(statements.append)
        appended = hy.log_message(sid, "assistant", "new tail")
        hy.conn.set_trace_callback(None)
        coverage_selects = [
            statement for statement in statements
            if "SELECT id, role, content, created_at" in statement
            and "FROM messages" in statement
        ]
        assert len(coverage_selects) == 1
        assert "id >" in coverage_selects[0]
        assert "NOT EXISTS" not in coverage_selects[0]
        assert hy.conn.execute(
            "SELECT coverage_message_id FROM sessions WHERE id = ?", (sid,),
        ).fetchone()["coverage_message_id"] == appended
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM message_retention_coverage "
            "WHERE source_session_id = ? AND coverage_version = ?",
            (sid, LOSSLESS_COVERAGE_VERSION),
        ).fetchone()["c"] == 301
    finally:
        hy.close()


def test_ordered_coverage_makes_raw_source_fields_append_only(cfg):
    hy = HyMem(_quiet_cfg(cfg), llm=RollingLLM())
    try:
        sid = "immutable_ordered_source"
        message_id = hy.log_message(sid, "user", "old canonical content")
        hy.open_session("other_session")

        mutations = [
            ("id = ?", (message_id + 1000,)),
            ("session_id = ?", ("other_session",)),
            ("role = ?", ("assistant",)),
            ("content = ?", ("new divergent content",)),
            ("created_at = ?", ("2020-01-01 00:00:00",)),
        ]
        for assignment, values in mutations:
            with pytest.raises(
                sqlite3.IntegrityError,
                match="ordered digest source is immutable",
            ):
                hy.conn.execute(
                    f"UPDATE messages SET {assignment} WHERE id = ?",
                    (*values, message_id),
                )

        raw = hy.conn.execute(
            "SELECT session_id, role, content FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        assert tuple(raw) == (sid, "user", "old canonical content")
        covered = covered_messages_after(hy.conn, sid, None)
        assert [(row.role, row.content) for row in covered] == [
            ("user", "old canonical content")
        ]

        # Retention still has its intended, verified DELETE path.
        hy.conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE id = ?", (message_id,),
        ).fetchone()["c"] == 0
        assert covered_messages_after(hy.conn, sid, None)[0].content == (
            "old canonical content"
        )
    finally:
        hy.close()


def test_operator_summary_survives_while_auto_summary_keeps_rolling(cfg):
    llm = RollingLLM()
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "operator"
        hy.log_message(sid, "user", "alpha initial topic")
        hy.close_session(sid)
        hy.dream()
        # Simulate an older integration editing the compatibility column.  Its
        # divergence from the last auto copy is treated as operator intent.
        hy.conn.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
            ("Operator-authored summary that must remain exact.", sid),
        )
        hy.log_message(sid, "assistant", "beta")
        hy.close_session(sid)
        hy.dream()

        row = hy.conn.execute(
            "SELECT summary, summary_source, auto_summary FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert row["summary"] == "Operator-authored summary that must remain exact."
        assert row["summary_source"] == "operator"
        assert "alpha" in row["auto_summary"] and "beta" in row["auto_summary"]
        rendered = effective_session_summary(row)
        assert "Operator-authored" in rendered and "Automatic rolling" in rendered
    finally:
        hy.close()


def test_derived_write_failure_rolls_back_all_digest_progress(cfg, monkeypatch):
    llm = RollingLLM(emit_slice_artifacts=True)
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "write_failure"
        message_id = hy.log_message(sid, "user", "alpha write transaction")
        hy.close_session(sid)

        def _fail_write(*_args, **_kwargs):
            raise RuntimeError("injected summary write failure")

        with monkeypatch.context() as patcher:
            patcher.setattr(
                "hymem.dreaming.runner.persist_auto_session_summary",
                _fail_write,
            )
            with pytest.raises(RuntimeError, match="injected summary write failure"):
                hy.dream()

        state = hy.conn.execute(
            "SELECT digest_cursor_message_id, digest_cursor_partial_message_id, "
            "digest_cursor_offset, auto_summary FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert state["digest_cursor_message_id"] is None
        assert state["digest_cursor_partial_message_id"] is None
        assert state["digest_cursor_offset"] == 0
        assert state["auto_summary"] is None
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM episodes WHERE session_id = ?", (sid,),
        ).fetchone()["c"] == 0

        hy.dream()
        assert hy.conn.execute(
            "SELECT digest_cursor_message_id FROM sessions WHERE id = ?", (sid,),
        ).fetchone()["digest_cursor_message_id"] == message_id
    finally:
        hy.close()


def test_corrupt_coverage_artifact_holds_digest_cursor(cfg):
    llm = RollingLLM()
    hy = HyMem(_quiet_cfg(cfg), llm=llm)
    try:
        sid = "corrupt_coverage"
        content = "alpha exact source"
        hy.conn.execute("INSERT INTO sessions(id) VALUES (?)", (sid,))
        message_id = hy.conn.execute(
            "INSERT INTO messages(session_id, role, content) VALUES (?, 'user', ?)",
            (sid, content),
        ).lastrowid
        chunk_id = coverage_chunk_id(sid, message_id)
        hy.conn.execute(
            "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
            "salience_reason, text, chunk_kind) "
            "VALUES (?, ?, ?, ?, 'corrupt-import', ?, 'coverage')",
            (
                chunk_id,
                sid,
                message_id,
                message_id,
                encode_message_record(
                    message_id=message_id, role="user", content=content
                ),
            ),
        )
        created_at = hy.conn.execute(
            "SELECT created_at FROM messages WHERE id = ?", (message_id,),
        ).fetchone()["created_at"]
        hy.close_session(sid)
        # The current SQL boundary rejects this corruption. Remove that one
        # guard deliberately so the hot reader's independent fail-closed
        # behavior remains covered as well.
        hy.conn.execute("DROP TRIGGER message_coverage_peer_insert_guard")
        hy.conn.execute(
            "INSERT INTO message_retention_coverage("
            "message_id, source_session_id, source_role, source_created_at, "
            "chunk_id, message_content_hash, hash_version, record_version, "
            "coverage_version) VALUES (?, ?, 'user', ?, ?, 'bad', ?, ?, ?)",
            (
                message_id,
                sid,
                created_at,
                chunk_id,
                MESSAGE_CONTENT_HASH_VERSION,
                MESSAGE_RECORD_VERSION,
                LOSSLESS_COVERAGE_VERSION,
            ),
        )
        hy.conn.execute(
            "UPDATE sessions SET coverage_message_id = ? WHERE id = ?",
            (message_id, sid),
        )
        with pytest.raises(RuntimeError, match="coverage proof mismatch"):
            covered_messages_after(hy.conn, sid, None)

        hy.dream()
        assert _digest_calls(llm) == []
        assert hy.conn.execute(
            "SELECT digest_cursor_message_id FROM sessions WHERE id = ?", (sid,),
        ).fetchone()["digest_cursor_message_id"] is None
    finally:
        hy.close()


def test_hot_reader_rejects_reserved_version_with_forged_chunk_identity(cfg):
    hy = HyMem(_quiet_cfg(cfg), llm=RollingLLM())
    try:
        sid = "forged_reserved"
        content = "forged ordered source"
        hy.conn.execute("INSERT INTO sessions(id) VALUES (?)", (sid,))
        mid = hy.conn.execute(
            "INSERT INTO messages(session_id, role, content) "
            "VALUES (?, 'user', ?)",
            (sid, content),
        ).lastrowid
        created_at = hy.conn.execute(
            "SELECT created_at FROM messages WHERE id = ?", (mid,)
        ).fetchone()[0]
        canonical = encode_message_record(
            message_id=mid, role="user", content=content
        )
        hy.conn.execute(
            "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
            "salience_reason, text, chunk_kind) VALUES "
            "('forged-msgcov', ?, ?, ?, 'forged', ?, 'coverage')",
            (sid, mid, mid, canonical),
        )
        # Bypass the strengthened v43 identity/shape guard only to retain this
        # independent reader-corruption regression.
        hy.conn.execute("DROP TRIGGER message_coverage_peer_insert_guard")
        hy.conn.execute(
            "INSERT INTO message_retention_coverage(message_id, "
            "source_session_id, source_role, source_created_at, chunk_id, "
            "message_content_hash, hash_version, record_version, coverage_version) "
            "VALUES (?, ?, 'user', ?, 'forged-msgcov', ?, ?, ?, ?)",
            (
                mid, sid, created_at, message_content_hash("user", content),
                MESSAGE_CONTENT_HASH_VERSION, MESSAGE_RECORD_VERSION,
                LOSSLESS_COVERAGE_VERSION,
            ),
        )
        hy.conn.execute(
            "UPDATE sessions SET coverage_message_id = ? WHERE id = ?",
            (mid, sid),
        )
        with pytest.raises(RuntimeError, match="coverage proof mismatch"):
            covered_messages_after(hy.conn, sid, None)
    finally:
        hy.close()


def test_prompt_rewind_reads_artifacts_after_safe_raw_retention(cfg):
    first_llm = RollingLLM()
    first_cfg = _quiet_cfg(cfg, message_retention_days=1)
    hy = HyMem(first_cfg, llm=first_llm)
    sid = "retained"
    try:
        hy.log_message(sid, "user", "alpha retained source", created_at="2020-01-01")
        hy.close_session(sid)
        hy.dream()
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?", (sid,),
        ).fetchone()["c"] == 0
    finally:
        hy.close()

    second_llm = RollingLLM(emit_slice_artifacts=True)
    bumped = dataclasses.replace(first_cfg, prompt_version=first_cfg.prompt_version + ".next")
    hy2 = HyMem(bumped, llm=second_llm)
    try:
        hy2.dream()
        assert "alpha retained source" in _digest_calls(second_llm)[0].user
        assert "Remembered alpha" in _digest_calls(second_llm)[0].user
        row = hy2.conn.execute(
            "SELECT digested_prompt_version, digest_cursor_message_id "
            "FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert row["digested_prompt_version"] == bumped.prompt_version
        assert row["digest_cursor_message_id"] is not None
        participants = hy2.conn.execute(
            "SELECT participants FROM episodes WHERE session_id = ?", (sid,),
        ).fetchone()["participants"]
        assert json.loads(participants) == ["user"]
    finally:
        hy2.close()


def test_prompt_change_mid_message_rewinds_exactly_from_retained_artifact(cfg):
    first_llm = RollingLLM()
    first_cfg = _quiet_cfg(
        cfg,
        dream_digest_max_chars=300,
        message_retention_days=1,
    )
    sid = "mid_slice_rewind"
    content = "alpha🙂" + ("0123456789界" * 120) + "omega"
    hy = HyMem(first_cfg, llm=first_llm)
    try:
        message_id = hy.log_message(
            sid, "user", content, created_at="2020-01-01"
        )
        hy.close_session(sid)
        hy.dream()
        first_state = hy.conn.execute(
            "SELECT digest_cursor_partial_message_id, digest_cursor_offset, "
            "auto_summary FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert first_state["digest_cursor_partial_message_id"] == message_id
        assert first_state["digest_cursor_offset"] > 0
        assert "alpha" in first_state["auto_summary"]
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?", (sid,),
        ).fetchone()["c"] == 0
    finally:
        hy.close()

    second_llm = RollingLLM()
    bumped = dataclasses.replace(
        first_cfg, prompt_version=first_cfg.prompt_version + ".mid-slice"
    )
    hy2 = HyMem(bumped, llm=second_llm)
    try:
        report = hy2.dream()
        first_new_call = _digest_calls(second_llm)[0].user
        assert re.search(rf"message {message_id} role=user chars=0:\d+/", first_new_call)
        assert "Remembered alpha" in first_new_call
        assert report.budget_exhausted is True

        for _ in range(30):
            state = hy2.conn.execute(
                "SELECT digested_prompt_version, digest_cursor_message_id, "
                "digest_cursor_offset FROM sessions WHERE id = ?", (sid,),
            ).fetchone()
            if state["digested_prompt_version"] == bumped.prompt_version:
                break
            next_report = hy2.dream()
            after = hy2.conn.execute(
                "SELECT digested_prompt_version FROM sessions WHERE id = ?", (sid,),
            ).fetchone()
            if after["digested_prompt_version"] != bumped.prompt_version:
                assert next_report.budget_exhausted is True
        assert state["digest_cursor_message_id"] == message_id
        assert state["digest_cursor_offset"] == 0
    finally:
        hy2.close()
