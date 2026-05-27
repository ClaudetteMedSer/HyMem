from __future__ import annotations

import dataclasses

from hymem.dreaming.retention import (
    prune_bookkeeping,
    prune_episodes_and_procedures,
    prune_messages,
    prune_retracted_edges,
)
from tests.conftest import seed_edge


def _session(conn, sid: str, *, days_ago: int, summary: str | None) -> None:
    conn.execute(
        "INSERT INTO sessions(id, started_at, summary) "
        "VALUES (?, datetime('now', ?), ?)",
        (sid, f"-{days_ago} days", summary),
    )


def _message(conn, sid: str, text: str) -> None:
    conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES (?, 'user', ?)",
        (sid, text),
    )


# --- Item A: message pruning (summary-gated) ---


def test_prune_messages_only_old_summarized(hy, cfg):
    conn = hy.conn
    _session(conn, "old_sum", days_ago=200, summary="did stuff")
    _session(conn, "old_nosum", days_ago=200, summary=None)
    _session(conn, "recent_sum", days_ago=1, summary="did stuff")
    for sid in ("old_sum", "old_nosum", "recent_sum"):
        _message(conn, sid, "hello")

    pruned = prune_messages(conn, cfg)

    assert pruned == 1
    remaining = {
        r["session_id"]
        for r in conn.execute("SELECT DISTINCT session_id FROM messages").fetchall()
    }
    assert remaining == {"old_nosum", "recent_sum"}


def test_prune_messages_noop_without_summary(hy, cfg):
    # Mirrors the stub/no-LLM deployment: no summaries are ever written, so the
    # summary gate makes this a no-op and nothing irreplaceable is destroyed.
    conn = hy.conn
    _session(conn, "old_nosum", days_ago=999, summary=None)
    _message(conn, "old_nosum", "hello")

    assert prune_messages(conn, cfg) == 0
    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 1


# --- Item B: retracted tombstone hard-delete (+ cascade) ---


def test_prune_retracted_edges_cascades_evidence(hy, cfg):
    conn = hy.conn
    _session(conn, "s", days_ago=0, summary=None)
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES ('c1', 's', 1, 1, 'r', 'txt')"
    )
    # old retracted, recent retracted, active.
    seed_edge(conn, "a", "uses", "b", status="retracted", days_ago=200)
    seed_edge(conn, "c", "uses", "d", status="retracted", days_ago=1)
    seed_edge(conn, "e", "uses", "f", status="active", days_ago=200)
    old_id = conn.execute(
        "SELECT id FROM knowledge_graph WHERE subject_canonical='a'"
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO kg_evidence(edge_id, chunk_id, polarity) VALUES (?, 'c1', 1)",
        (old_id,),
    )

    pruned = prune_retracted_edges(conn, cfg)

    assert pruned == 1
    subs = {
        r["subject_canonical"]
        for r in conn.execute("SELECT subject_canonical FROM knowledge_graph").fetchall()
    }
    assert subs == {"c", "e"}  # old retracted gone; recent + active kept
    # cascade removed the evidence row
    assert (
        conn.execute("SELECT COUNT(*) AS c FROM kg_evidence").fetchone()["c"] == 0
    )


# --- Item C: bookkeeping caps ---


def test_prune_bookkeeping_keeps_newest(hy, cfg):
    conn = hy.conn
    small = dataclasses.replace(cfg, dream_runs_keep=3, extraction_feedback_keep=2)
    for i in range(6):
        conn.execute(
            "INSERT INTO dream_runs(started_at) VALUES (datetime('now', ?))",
            (f"-{i} days",),
        )
    for i in range(5):
        conn.execute(
            "INSERT INTO extraction_feedback(chunk_text_snippet, extracted_subject, "
            "extracted_predicate, extracted_object, created_at) "
            "VALUES ('s', 'a', 'uses', 'b', datetime('now', ?))",
            (f"-{i} days",),
        )

    pruned = prune_bookkeeping(conn, small)

    assert pruned == (6 - 3) + (5 - 2)
    assert conn.execute("SELECT COUNT(*) AS c FROM dream_runs").fetchone()["c"] == 3
    assert (
        conn.execute("SELECT COUNT(*) AS c FROM extraction_feedback").fetchone()["c"]
        == 2
    )


# --- Item D: episode + stale-procedure aging ---


def test_prune_episodes_and_procedures(hy, cfg):
    conn = hy.conn
    _session(conn, "s", days_ago=0, summary=None)
    conn.execute(
        "INSERT INTO episodes(id, session_id, title, summary, created_at) "
        "VALUES ('e_old', 's', 'Old', 'old summary', datetime('now', '-200 days'))"
    )
    conn.execute(
        "INSERT INTO episodes(id, session_id, title, summary, created_at) "
        "VALUES ('e_new', 's', 'New', 'new summary', datetime('now', '-1 days'))"
    )
    conn.execute(
        "INSERT INTO procedures(id, session_id, name, status, created_at) "
        "VALUES ('p_stale_old', 's', 'po', 'stale', datetime('now', '-200 days'))"
    )
    conn.execute(
        "INSERT INTO procedures(id, session_id, name, status, created_at) "
        "VALUES ('p_active_old', 's', 'pa', 'active', datetime('now', '-200 days'))"
    )
    conn.execute(
        "INSERT INTO procedures(id, session_id, name, status, created_at) "
        "VALUES ('p_stale_new', 's', 'pn', 'stale', datetime('now', '-1 days'))"
    )

    pruned = prune_episodes_and_procedures(conn, cfg)

    assert pruned == 2  # e_old + p_stale_old
    eps = {r["id"] for r in conn.execute("SELECT id FROM episodes").fetchall()}
    assert eps == {"e_new"}
    procs = {r["id"] for r in conn.execute("SELECT id FROM procedures").fetchall()}
    assert procs == {"p_active_old", "p_stale_new"}
    # FTS shadow stays in sync via the existing delete trigger.
    fts = conn.execute(
        "SELECT COUNT(*) AS c FROM episodes_fts WHERE episodes_fts MATCH 'old'"
    ).fetchone()["c"]
    assert fts == 0
