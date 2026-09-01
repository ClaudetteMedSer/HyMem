"""Offline (Mac, no LLM/box) unit tests for the Stage-2 front-run gate
(benchmarks/episode_coverage_probe.py).

Same discipline as test_raptor_cluster_probe: the probe's pure core
(`characterize_coverage`) is importable and pinned here against a tiny store
built with the project's own fixtures (HyMem + StubLLM). The "dreamed" state is
simulated by setting sessions.digested_prompt_version / inserting episode rows
directly — no real dream, no real LLM, ever.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

from hymem import HyMem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from episode_coverage_probe import (  # noqa: E402
    characterize_coverage,
    open_store_readonly,
)

LONG_TURN = "Substantial content about the deployment pipeline. " * 20  # ~1k chars


def _mark_dreamed(hy: HyMem, sid: str, version: str = "v1") -> None:
    """Simulate a completed batched session digest for `sid`."""
    with hy.conn:
        hy.conn.execute(
            "UPDATE sessions SET digested_prompt_version = ? WHERE id = ?",
            (version, sid),
        )


def _insert_episode(hy: HyMem, sid: str, eid: str) -> None:
    """Insert an episode row directly (what a successful extraction persists)."""
    with hy.conn:
        hy.conn.execute(
            "INSERT INTO episodes (id, session_id, title, summary) VALUES (?,?,?,?)",
            (eid, sid, "An episode", "Something happened in this session."),
        )


@pytest.fixture
def probe_store(hy: HyMem, cfg) -> Path:
    """A closed store with one session per coverage bucket plus one covered.

    covered        dreamed, 1 episode          → not in any bucket
    never          NOT dreamed, 0 episodes     → never_dreamed
    zero-short     dreamed, 0 episodes, thin   → dreamed_zero_short
    zero-long      dreamed, 0 episodes, fat    → dreamed_zero_long
    """
    # Covered: dreamed and produced an episode.
    hy.log_messages("covered", [("user", LONG_TURN), ("assistant", LONG_TURN)])
    _mark_dreamed(hy, "covered")
    _insert_episode(hy, "covered", "covered@1-2")

    # Never dreamed: digested_prompt_version stays NULL.
    hy.log_messages("never", [("user", LONG_TURN), ("assistant", LONG_TURN)])

    # Dreamed but zero episodes, thin: 2 short turns (≤4 msgs AND ≤1500 chars).
    hy.log_messages("zero-short", [("user", "hi"), ("assistant", "hello!")])
    _mark_dreamed(hy, "zero-short")

    # Dreamed but zero episodes, substantial: 6 long turns (>4 msgs, >1500 chars).
    hy.log_messages(
        "zero-long",
        [("user", LONG_TURN), ("assistant", LONG_TURN)] * 3,
    )
    _mark_dreamed(hy, "zero-long")

    hy.close()
    return cfg.db_path


def test_bucket_classification_and_coverage(probe_store: Path):
    conn = open_store_readonly(probe_store)
    try:
        result = characterize_coverage(conn)
    finally:
        conn.close()

    assert result["total_sessions"] == 4
    assert result["covered_sessions"] == 1
    assert result["uncovered_sessions"] == 3
    assert result["coverage_fraction"] == pytest.approx(0.25)
    assert result["buckets"] == {
        "never_dreamed": 1,
        "dreamed_zero_short": 1,
        "dreamed_zero_long": 1,
    }

    by_id = {r["session_id"]: r for r in result["uncovered"]}
    assert set(by_id) == {"never", "zero-short", "zero-long"}
    assert by_id["never"]["bucket"] == "never_dreamed"
    assert by_id["never"]["digested_prompt_version"] is None
    assert by_id["zero-short"]["bucket"] == "dreamed_zero_short"
    assert by_id["zero-long"]["bucket"] == "dreamed_zero_long"
    assert by_id["zero-long"]["digested_prompt_version"] == "v1"
    # Covered session never appears in the uncovered listing.
    assert "covered" not in by_id


def test_per_session_record_metrics(probe_store: Path):
    conn = open_store_readonly(probe_store)
    try:
        result = characterize_coverage(conn)
    finally:
        conn.close()

    rec = {r["session_id"]: r for r in result["uncovered"]}["zero-long"]
    assert rec["n_messages"] == 6
    assert rec["n_user"] == 3
    assert rec["n_assistant"] == 3
    assert rec["n_other"] == 0
    assert rec["content_chars"] == 6 * len(LONG_TURN)
    assert rec["first_message_at"] is not None
    assert rec["last_message_at"] is not None
    assert rec["first_message_at"] <= rec["last_message_at"]


def test_short_boundary_is_either_axis(probe_store: Path):
    """A dreamed-zero session is 'short' when EITHER axis is thin, so widening
    the message threshold past zero-long's 6 turns flips it to short even
    though its char count is large."""
    conn = open_store_readonly(probe_store)
    try:
        result = characterize_coverage(conn, short_max_messages=10)
    finally:
        conn.close()
    rec = {r["session_id"]: r for r in result["uncovered"]}["zero-long"]
    assert rec["bucket"] == "dreamed_zero_short"


def test_tool_and_system_turns_excluded_from_chars(hy: HyMem, cfg):
    """Role mix is reported, but tool/system content never counts toward the
    short/long char boundary (mirrors the messages_fts ingest filter)."""
    hy.log_messages(
        "tools",
        [("user", "run it"), ("tool", "x" * 9000), ("system", "y" * 9000)],
    )
    _mark_dreamed(hy, "tools")
    hy.close()

    conn = open_store_readonly(cfg.db_path)
    try:
        result = characterize_coverage(conn)
    finally:
        conn.close()
    rec = {r["session_id"]: r for r in result["uncovered"]}["tools"]
    assert rec["content_chars"] == len("run it")
    assert rec["n_other"] == 2
    assert rec["bucket"] == "dreamed_zero_short"


def test_probe_opens_read_only(probe_store: Path):
    conn = open_store_readonly(probe_store)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO sessions (id) VALUES ('intruder')")
    finally:
        conn.close()


def test_missing_store_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        open_store_readonly(tmp_path / "nope.sqlite")


# ─────────────────────────────────────────────────────────────────────────────
# The never_dreamed reading (2026-09-01)
#
# `never_dreamed` cannot distinguish a runner that FAILED to reach a session
# from one that has not reached it YET, and the verdict guide offers only the
# first ("scheduler/runner bug; the fix is MECHANICAL"). On an LME store that
# reading is wrong: HyMem's dream is budgeted per cycle, benchmark adapters
# bulk-ingest and then dream ONCE, and the undigested tail is the designed
# shape. Measured on the surviving 2026-08-31 LME stores: 105 of 476 sessions
# digested (22.1%), with 36 sessions holding 397 messages left undigested after
# a single cycle. On the prod store the same bucket is not a gap at all -- all
# nine never_dreamed sessions hold zero messages.
# ─────────────────────────────────────────────────────────────────────────────
from episode_coverage_probe import (  # noqa: E402
    dream_history,
    never_dreamed_reading,
)


def _nd(n_messages):
    return [{"n_messages": m} for m in n_messages]


def _hist(cycles, **kw):
    h = {"cycles": cycles, "episodes_created": 0, "digest_failures": 0,
         "last": {"id": cycles, "sessions_processed": 1, "chunks_seen": 56,
                  "chunks_processed": 29}}
    h.update(kw)
    return h


def test_one_cycle_over_a_bulk_ingest_is_not_called_a_runner_bug():
    """THE case. This is every benchmark adapter's shape, and the guide's
    mechanical-fix reading would send someone to fix a scheduler that is
    working as designed."""
    r = never_dreamed_reading(_hist(1), _nd([12, 20, 5]))
    assert "ONE CYCLE OVER A BULK INGEST" in r
    assert "3 session(s) holding 37 message(s)" in r
    assert "not a runner bug" in r
    assert "bounded to the digested fraction" in r


def test_many_cycles_with_content_still_undigested_IS_the_runner_reading():
    """The budget explains one cycle. It does not explain five hundred."""
    r = never_dreamed_reading(_hist(500), _nd([12, 20]))
    assert "RUNNER GAP" in r
    assert "500 dream cycles" in r
    assert "rule out the per-cycle budget" in r


def test_empty_sessions_are_not_a_coverage_gap_at_all():
    """The prod store's nine never_dreamed sessions hold zero messages. No
    prompt and no scheduler can extract an episode from nothing, so counting
    them as a gap overstates the population any fix could reach."""
    r = never_dreamed_reading(_hist(500), _nd([0, 0, 0]))
    assert "NOT A GAP" in r
    assert "zero messages" in r


def test_one_session_with_content_is_enough_to_leave_the_not_a_gap_reading():
    r = never_dreamed_reading(_hist(500), _nd([0, 0, 3]))
    assert "NOT A GAP" not in r
    assert "1 session(s) holding 3 message(s)" in r


def test_an_empty_dream_runs_table_reads_as_never_ran():
    r = never_dreamed_reading(_hist(0), _nd([5]))
    assert "NEVER RAN" in r


def test_a_store_without_the_table_says_so_rather_than_guessing():
    r = never_dreamed_reading(None, _nd([5]))
    assert "UNAVAILABLE" in r
    assert "neither reading is supported" in r


def test_nothing_to_decide_when_the_bucket_is_empty():
    r = never_dreamed_reading(_hist(1), [])
    assert "nothing to decide" in r


def _store_with_dream_runs(tmp_path, rows):
    """A store carrying a `dream_runs` table with the columns this probe reads.

    Built explicitly rather than taken from `probe_store`: that fixture has
    never dreamed, so `dream_history` returns None on it and any assertion
    about the returned dict is skipped. A first cut of the test below did
    exactly that and passed against a mutation that put the chunk ratio back --
    the vacuous-check shape this whole reading exists to name."""
    db = tmp_path / "hist.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE dream_runs (id INTEGER PRIMARY KEY, "
        "sessions_processed INTEGER, chunks_seen INTEGER, "
        "chunks_processed INTEGER, episodes_created INTEGER, "
        "digest_failures INTEGER)")
    con.executemany("INSERT INTO dream_runs VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return db


def test_dream_history_reports_no_chunk_ratio(tmp_path):
    """`chunks_seen` is the candidate pool re-counted every cycle, not a
    backlog: the prod store's last cycles each saw ~330 chunks and processed
    0-2 because nothing was new. Summed it reads 425/330255, and
    processed/seen looks like a coverage fraction while being nothing of the
    kind. A caveat does not survive being quoted, so the number is not
    produced."""
    db = _store_with_dream_runs(tmp_path, [(1, 14, 56, 29, 48, 0),
                                           (2, 14, 60, 2, 3, 1)])
    conn = open_store_readonly(db)
    try:
        h = dream_history(conn)
    finally:
        conn.close()
    assert h is not None, "the fixture must actually have a dream_runs table"
    assert set(h) == {"cycles", "episodes_created", "digest_failures", "last"}
    assert "chunk_fraction" not in h


def test_dream_history_totals_and_last_cycle(tmp_path):
    """Totals where summing is meaningful (episodes, failures); the LAST cycle
    where it is not (chunks)."""
    db = _store_with_dream_runs(tmp_path, [(1, 14, 56, 29, 48, 0),
                                           (2, 14, 60, 2, 3, 1)])
    conn = open_store_readonly(db)
    try:
        h = dream_history(conn)
    finally:
        conn.close()
    assert h["cycles"] == 2
    assert h["episodes_created"] == 51
    assert h["digest_failures"] == 1
    assert h["last"] == {"id": 2, "sessions_processed": 14,
                         "chunks_seen": 60, "chunks_processed": 2}


def test_dream_history_on_an_empty_table_is_zero_cycles_not_none(tmp_path):
    """None means "cannot be read". Zero rows means "it never ran". Collapsing
    them would make an un-dreamed store indistinguishable from an old one."""
    db = _store_with_dream_runs(tmp_path, [])
    conn = open_store_readonly(db)
    try:
        h = dream_history(conn)
    finally:
        conn.close()
    assert h is not None and h["cycles"] == 0 and h["last"] is None


def test_dream_history_is_none_when_the_table_is_absent(tmp_path):
    db = tmp_path / "old.sqlite"
    sqlite3.connect(db).close()
    conn = open_store_readonly(db)
    try:
        assert dream_history(conn) is None
    finally:
        conn.close()


def test_the_reading_is_carried_in_the_result(probe_store):
    conn = open_store_readonly(probe_store)
    try:
        res = characterize_coverage(conn)
    finally:
        conn.close()
    assert "never_dreamed_reading" in res
    assert "dream_history" in res
    assert res["never_dreamed_reading"]
