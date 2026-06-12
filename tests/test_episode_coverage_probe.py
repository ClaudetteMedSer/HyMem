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
