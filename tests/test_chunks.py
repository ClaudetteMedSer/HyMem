from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hymem.core.db import connect, initialize
from hymem.dreaming.chunks import extract_high_salience_chunks


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
