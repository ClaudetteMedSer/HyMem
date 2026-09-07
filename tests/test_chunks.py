from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from hymem.core.db import connect, initialize
from hymem.dreaming.chunks import (
    BASELINE_SALIENCE_REASON,
    extract_baseline_chunks,
    extract_high_salience_chunks,
    persist_chunks,
)
from hymem.dreaming.lossless import materialize_message_coverage


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "hymem.sqlite")
    initialize(c)
    c.execute("INSERT INTO sessions(id) VALUES ('s')")
    return c


def _add_user(conn: sqlite3.Connection, content: str) -> None:
    conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s', 'user', ?)",
        (content,),
    )


def _reasons(conn: sqlite3.Connection, *, min_chars: int = 30) -> list[tuple[str, str]]:
    chunks = extract_high_salience_chunks(conn, "s", min_chars=min_chars)
    return [(c.salience_reason, c.text) for c in chunks]


def test_correction_wrong_use_not(conn):
    _add_user(conn, "Wrong, use psycopg3 not psycopg2.")
    out = _reasons(conn)
    assert len(out) == 1
    assert out[0][0] == "correction_or_preference_trigger"


def test_correction_use_x_not_y(conn):
    _add_user(conn, "We use FastAPI not Flask.")
    out = _reasons(conn)
    assert len(out) == 1
    assert out[0][0] == "correction_or_preference_trigger"


def test_correction_instead_of(conn):
    _add_user(conn, "Use uv instead of pip.")
    out = _reasons(conn)
    assert len(out) == 1
    assert out[0][0] == "correction_or_preference_trigger"


def test_neutral_statement_above_threshold_included(conn):
    msg = "We deployed the new build today."  # 32 chars, no trigger
    assert len(msg) >= 30 and len(msg) < 80
    _add_user(conn, msg)
    out = _reasons(conn)
    assert len(out) == 1
    assert out[0][0] == "long_user_turn"


def test_short_ack_excluded(conn):
    _add_user(conn, "ok thanks")
    out = _reasons(conn)
    assert out == []


def _baseline(conn, *, min_chars: int = 30):
    materialize_message_coverage(conn, "s")
    return extract_baseline_chunks(
        conn,
        "s",
        prompt_version="test-v1",
        limit=None,
        min_chars=min_chars,
    )


def test_short_fact_reaches_distinct_baseline_tier(conn):
    _add_user(conn, "My dog is Max.")

    assert _reasons(conn) == []
    chunks = _baseline(conn)

    assert len(chunks) == 1
    assert chunks[0].salience_reason == BASELINE_SALIENCE_REASON
    assert chunks[0].text == "user: My dog is Max."
    assert chunks[0].source_message_ids == (chunks[0].end_message_id,)


def test_short_confirmation_keeps_preceding_assistant_context(conn):
    assistant_id = conn.execute(
        "INSERT INTO messages(session_id, role, content) "
        "VALUES ('s', 'assistant', 'Your service deploys to Oslo.')"
    ).lastrowid
    _add_user(conn, "Yes.")

    chunks = _baseline(conn)

    assert len(chunks) == 1
    assert chunks[0].text == (
        "assistant: Your service deploys to Oslo.\nuser: Yes."
    )
    assert chunks[0].source_message_ids == (
        assistant_id,
        chunks[0].end_message_id,
    )


def test_baseline_excludes_only_definite_blank_noise(conn):
    _add_user(conn, " \t\n " * 20)
    assert _reasons(conn) == []
    assert _baseline(conn) == []


def test_high_and_baseline_candidates_are_disjoint_and_persist_once(conn):
    _add_user(conn, "My dog is Max.")
    _add_user(conn, "A neutral but sufficiently long statement about our service.")
    _add_user(conn, "No, use uv.")
    materialize_message_coverage(conn, "s")

    high = extract_high_salience_chunks(conn, "s", min_chars=30)
    baseline = extract_baseline_chunks(
        conn,
        "s",
        prompt_version="test-v1",
        limit=None,
        min_chars=30,
    )

    assert len(high) == 2
    assert len(baseline) == 1
    assert {chunk.id for chunk in high}.isdisjoint(
        {chunk.id for chunk in baseline}
    )
    persist_chunks(conn, [*high, *baseline])
    assert conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_kind='extraction'"
    ).fetchone()[0] == 3


def test_dutch_correction_trigger(conn):
    # Short Dutch correction — caught by trigger, not the length fallback.
    _add_user(conn, "Nee, gebruik uv.")
    out = _reasons(conn)
    assert len(out) == 1
    assert out[0][0] == "correction_or_preference_trigger"


def test_dutch_preference_trigger(conn):
    _add_user(conn, "Ik heb een voorkeur voor uv.")
    out = _reasons(conn)
    assert len(out) == 1
    assert out[0][0] == "correction_or_preference_trigger"


def test_dutch_we_gebruiken_trigger(conn):
    _add_user(conn, "We gebruiken uv.")
    out = _reasons(conn)
    assert len(out) == 1
    assert out[0][0] == "correction_or_preference_trigger"


def test_persisted_chunk_identity_refreshes_scheduling_reason(conn):
    """A tier label is mutable policy over one immutable source artifact.

    A baseline snapshot can be persisted immediately before the salience tier
    sees the same stable source range.  Session/range/text/kind and the exact
    manifest must collide exactly, while the latest validated producer policy
    must refresh scheduling metadata before publishing its acknowledgement.
    """
    _add_user(conn, "A sufficiently long source message for deterministic chunking.")
    materialize_message_coverage(conn, "s")
    chunk = extract_high_salience_chunks(conn, "s", min_chars=30)[0]
    persist_chunks(conn, [replace(chunk, salience_reason="baseline_backstop")])
    persist_chunks(conn, [chunk])

    stored = conn.execute(
        "SELECT salience_reason, source_manifest_count FROM chunks WHERE id = ?",
        (chunk.id,),
    ).fetchone()
    assert stored["salience_reason"] == chunk.salience_reason
    assert stored["source_manifest_count"] == 1
    with pytest.raises(RuntimeError, match="chunk identity collision"):
        persist_chunks(conn, [replace(chunk, text=chunk.text + " tampered")])
