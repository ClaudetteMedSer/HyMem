"""Tests for short-session lossless coverage (never-dreamed bug fix).

Both chunk tiers only mint chunks from USER turns that clear min_chars or a
trigger regex, so a session whose user turns are all short (test/WebSocket/
diagnostic sessions) used to produce zero chunks, skip the per-session tail,
and leave ``sessions.digested_prompt_version`` NULL forever.  The v38 fix gives
every message an exact coverage artifact, independent of either extraction
tier; truly empty sessions still skip.  Legacy fallback-builder unit tests stay
below to pin compatibility, but the runner no longer needs to mint one.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hymem import HyMem
from hymem.core.db import connect, initialize
from hymem.dreaming.chunks import _chunk_id, extract_fallback_chunk
from hymem.dreaming.lossless import coverage_chunk_id
from hymem.extraction.llm import StubLLMClient


# --- helpers ---------------------------------------------------------------


def _digest_llm(
    *,
    episodes: list[dict] | None = None,
    summary: str = "",
    procedures: list[dict] | None = None,
) -> StubLLMClient:
    """Stub returning one combined digest object for the batched call, and an
    empty array for triple/marker chunk calls. Keyed on the digest user-prompt
    closer ``Return the JSON object now`` (see tests/test_digest.py)."""
    payload = {
        "episodes": episodes or [],
        "summary": summary,
        "procedures": procedures or [],
    }
    return StubLLMClient(
        fixtures={"Return the JSON object now": json.dumps(payload)},
        default="[]",
    )


def _digest_calls(llm: StubLLMClient) -> list:
    """The subset of recorded calls that hit the batched digest prompt."""
    return [c for c in llm.calls if "Return the JSON object now" in c.user]


def _seed_session(hy: HyMem, sid: str, turns: list[tuple[str, str]]) -> None:
    hy.open_session(sid)
    for role, content in turns:
        hy.log_message(sid, role, content)
    hy.close_session(sid)


# All user turns well under salience_min_chars (30) and free of trigger words:
# both tiers mint zero chunks, so only the fallback can carry the session.
_SHORT_TURNS = [
    ("user", "ping"),
    ("assistant", "pong"),
    ("user", "ok"),
]


def _fallback_rows(hy: HyMem, sid: str) -> list[sqlite3.Row]:
    return hy.conn.execute(
        "SELECT * FROM chunks WHERE session_id = ? "
        "AND salience_reason = 'short_session_fallback'",
        (sid,),
    ).fetchall()


def _coverage_rows(hy: HyMem, sid: str) -> list[sqlite3.Row]:
    return hy.conn.execute(
        "SELECT * FROM chunks WHERE session_id = ? AND chunk_kind = 'coverage' "
        "ORDER BY start_message_id",
        (sid,),
    ).fetchall()


def _digested_version(hy: HyMem, sid: str) -> str | None:
    return hy.conn.execute(
        "SELECT digested_prompt_version FROM sessions WHERE id = ?", (sid,)
    ).fetchone()["digested_prompt_version"]


# --- (a) short session gets a fallback chunk and a digest stamp -------------


def test_short_session_gets_exact_coverage_and_digest_stamp(cfg):
    """A session whose user turns are all short (no triggers) must still be
    digested: one exact artifact per message exists and the prompt is stamped."""
    llm = _digest_llm(summary="A short diagnostic ping-pong exchange.")
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_short"
        _seed_session(hy, sid, _SHORT_TURNS)
        hy.dream()

        rows = _coverage_rows(hy, sid)
        assert len(rows) == len(_SHORT_TURNS)
        assert all(row["id"].startswith("msgcov_") for row in rows)
        assert _fallback_rows(hy, sid) == []

        assert _digested_version(hy, sid) == hy.config.prompt_version
        assert len(_digest_calls(llm)) == 1
    finally:
        hy.close()


# --- (b) episodes citing the fallback chunk persist -------------------------


def test_episode_citing_coverage_chunks_persists(cfg):
    """Digest episodes can cite exact message artifacts across a range."""
    llm = _digest_llm()  # fixture filled in below, once the chunk id is known
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_episode"
        _seed_session(hy, sid, _SHORT_TURNS)

        ids = [
            r["id"]
            for r in hy.conn.execute(
                "SELECT id FROM messages WHERE session_id = ? ORDER BY id", (sid,)
            )
        ]
        coverage_ids = [coverage_chunk_id(sid, message_id) for message_id in ids]
        episode = {
            "title": "Diagnostic ping",
            "summary": "A connectivity check session.",
            "outcome": "resolved",
            "key_entities": [],
            "chunk_ids": coverage_ids,
        }
        llm.fixtures["Return the JSON object now"] = json.dumps(
            {"episodes": [episode], "summary": "", "procedures": []}
        )

        hy.dream()

        ep = hy.conn.execute(
            "SELECT title, start_message_id, end_message_id "
            "FROM episodes WHERE session_id = ?",
            (sid,),
        ).fetchall()
        assert [r["title"] for r in ep] == ["Diagnostic ping"]
        assert ep[0]["start_message_id"] == ids[0]
        assert ep[0]["end_message_id"] == ids[-1]
    finally:
        hy.close()


# --- (c) truly empty session still skips ------------------------------------


def test_empty_session_still_skipped(cfg):
    """A session with ZERO messages makes no digest call and stays unstamped."""
    llm = _digest_llm(summary="Should never be requested.")
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_empty"
        hy.open_session(sid)
        hy.close_session(sid)
        hy.dream()

        assert _fallback_rows(hy, sid) == []
        assert _digested_version(hy, sid) is None
        assert _digest_calls(llm) == []
    finally:
        hy.close()


# --- (d) re-dream of the short session is free ------------------------------


def test_redream_short_session_makes_no_further_digest_calls(cfg):
    """The fallback chunk must not break the skip-guard: a second dream over
    the unchanged short session costs zero digest LLM calls."""
    llm = _digest_llm(summary="A short diagnostic ping-pong exchange.")
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_redream"
        _seed_session(hy, sid, _SHORT_TURNS)
        hy.dream()
        after_first = len(_digest_calls(llm))
        assert after_first == 1

        hy.dream()  # nothing changed
        assert len(_digest_calls(llm)) == after_first, "re-dream must skip the digest"
        assert len(_coverage_rows(hy, sid)) == len(_SHORT_TURNS)
        assert _fallback_rows(hy, sid) == []
    finally:
        hy.close()


# --- (e) normal sessions never get a fallback chunk -------------------------


def test_qualifying_session_gets_no_fallback_chunk(cfg):
    """The fallback only fires when BOTH tiers are empty: a session with a
    qualifying (long) user turn mints regular chunks and no fallback row."""
    llm = _digest_llm(summary="A regular substantive session about deploys.")
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_normal"
        _seed_session(
            hy,
            sid,
            [
                ("assistant", "how do we deploy to staging?"),
                ("user", "Build the docker image then kubectl apply the staging manifests."),
            ],
        )
        hy.dream()

        assert _fallback_rows(hy, sid) == []
        regular = hy.conn.execute(
            "SELECT salience_reason FROM chunks WHERE session_id = ? "
            "AND chunk_kind = 'extraction'", (sid,)
        ).fetchall()
        assert regular, "the regular tiers must have minted chunks"
        assert _digested_version(hy, sid) == hy.config.prompt_version
    finally:
        hy.close()


# --- (f) extract_fallback_chunk unit tests ----------------------------------


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "hymem.sqlite")
    initialize(c)
    c.execute("INSERT INTO sessions(id) VALUES ('s')")
    return c


def _add(conn: sqlite3.Connection, role: str, content: str) -> int:
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s', ?, ?)",
        (role, content),
    )
    return cur.lastrowid


def test_fallback_spans_first_to_last_and_includes_assistant(conn):
    first = _add(conn, "user", "hi")
    _add(conn, "assistant", "hello there")
    last = _add(conn, "user", "bye")

    chunk = extract_fallback_chunk(conn, "s", max_chars=1000)
    assert chunk is not None
    assert chunk.start_message_id == first
    assert chunk.end_message_id == last
    assert chunk.salience_reason == "short_session_fallback"
    assert chunk.id == _chunk_id("s", first, last)
    assert chunk.text == "user: hi\nassistant: hello there\nuser: bye"


def test_fallback_truncates_at_max_chars(conn):
    _add(conn, "user", "x" * 500)
    chunk = extract_fallback_chunk(conn, "s", max_chars=40)
    assert chunk is not None
    assert len(chunk.text) == 40
    assert chunk.text.startswith("user: xxx")


def test_fallback_returns_none_on_empty_session(conn):
    assert extract_fallback_chunk(conn, "s", max_chars=1000) is None


def test_fallback_returns_none_when_only_empty_content(conn):
    _add(conn, "user", "")
    _add(conn, "assistant", "")
    assert extract_fallback_chunk(conn, "s", max_chars=1000) is None


def test_fallback_ignores_non_user_assistant_roles(conn):
    _add(conn, "system", "you are a helpful bot")
    first = _add(conn, "user", "ok")
    chunk = extract_fallback_chunk(conn, "s", max_chars=1000)
    assert chunk is not None
    assert chunk.start_message_id == first
    assert chunk.text == "user: ok"
