from __future__ import annotations

import dataclasses
import sqlite3

import pytest

from hymem.core import db as core_db
from hymem.dreaming.message_coverage import (
    LOSSLESS_COVERAGE_VERSION,
    coverage_chunk_id,
    encode_message_record,
    record_message_coverage,
    release_message_coverage,
)
from hymem.dreaming.retention import (
    prune_bookkeeping,
    prune_chunks,
    prune_episodes_and_procedures,
    prune_messages,
    prune_retracted_edges,
)
from tests.conftest import seed_edge


def _session(
    conn,
    sid: str,
    *,
    days_ago: int,
    summary: str | None,
    ended: bool = False,
) -> None:
    conn.execute(
        "INSERT INTO sessions(id, started_at, ended_at, summary) "
        "VALUES (?, datetime('now', ?), CASE WHEN ? THEN CURRENT_TIMESTAMP END, ?)",
        (sid, f"-{days_ago} days", ended, summary),
    )


def _message(conn, sid: str, text: str, *, days_ago: int = 0) -> int:
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content, created_at) "
        "VALUES (?, 'user', ?, datetime('now', ?))",
        (sid, text, f"-{days_ago} days"),
    )
    return int(cur.lastrowid)


def _cover(conn, sid: str, message_id: int, text: str) -> str:
    chunk_id = f"coverage_{message_id}"
    record = encode_message_record(
        message_id=message_id,
        role="user",
        content=text,
    )
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES (?, ?, ?, ?, 'lossless', ?)",
        (chunk_id, sid, message_id, message_id, record),
    )
    record_message_coverage(
        conn,
        message_id=message_id,
        chunk_id=chunk_id,
        coverage_version="test-lossless-v1",
    )
    return chunk_id


# --- Item A: safe, explicit raw-message pruning ---


@pytest.mark.parametrize("disabled_days", [0, -1])
def test_prune_messages_disabled_by_default_and_for_nonpositive_values(
    hy, cfg, disabled_days
):
    assert cfg.message_retention_days == 0
    conn = hy.conn
    _session(conn, "ended", days_ago=200, summary="summary", ended=True)
    message_id = _message(conn, "ended", "keep by default", days_ago=200)
    _cover(conn, "ended", message_id, "keep by default")

    disabled = dataclasses.replace(cfg, message_retention_days=disabled_days)
    assert prune_messages(conn, disabled) == 0
    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 1


def test_prune_messages_old_active_session_keeps_old_and_fresh_messages(hy, cfg):
    conn = hy.conn
    enabled = dataclasses.replace(cfg, message_retention_days=90)
    _session(conn, "active", days_ago=200, summary="summary", ended=False)
    old_id = _message(conn, "active", "old but session active", days_ago=150)
    fresh_id = _message(conn, "active", "fresh active tail", days_ago=0)
    _cover(conn, "active", old_id, "old but session active")
    _cover(conn, "active", fresh_id, "fresh active tail")

    assert prune_messages(conn, enabled) == 0
    assert {
        row["id"] for row in conn.execute("SELECT id FROM messages").fetchall()
    } == {old_id, fresh_id}


def test_prune_messages_uses_each_message_date_in_an_ended_session(hy, cfg):
    conn = hy.conn
    enabled = dataclasses.replace(cfg, message_retention_days=90)
    _session(conn, "ended", days_ago=200, summary=None, ended=True)
    old_id = _message(conn, "ended", "old covered turn", days_ago=150)
    fresh_id = _message(conn, "ended", "fresh covered turn", days_ago=1)
    _cover(conn, "ended", old_id, "old covered turn")
    _cover(conn, "ended", fresh_id, "fresh covered turn")

    assert prune_messages(conn, enabled) == 1
    remaining = conn.execute("SELECT id FROM messages").fetchall()
    assert [row["id"] for row in remaining] == [fresh_id]


def test_prune_messages_summary_and_digest_are_not_lossless_coverage(hy, cfg):
    conn = hy.conn
    enabled = dataclasses.replace(cfg, message_retention_days=90)
    _session(conn, "summarized", days_ago=200, summary="lossy", ended=True)
    message_id = _message(conn, "summarized", "only raw copy", days_ago=150)
    conn.execute(
        "UPDATE sessions SET digested_message_id = ? WHERE id = 'summarized'",
        (message_id,),
    )

    assert prune_messages(conn, enabled) == 0
    assert (
        conn.execute("SELECT content FROM messages").fetchone()["content"]
        == "only raw copy"
    )


def test_prune_messages_rejects_a_stale_content_hash(hy, cfg):
    conn = hy.conn
    enabled = dataclasses.replace(cfg, message_retention_days=90)
    _session(conn, "ended", days_ago=200, summary=None, ended=True)
    message_id = _message(conn, "ended", "hash-bound source", days_ago=150)
    _cover(conn, "ended", message_id, "hash-bound source")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE message_retention_coverage SET message_content_hash = ? "
            "WHERE message_id = ?",
            ("0" * 64, message_id),
        )
    conn.execute("DROP TRIGGER message_coverage_peer_update_guard")
    conn.execute(
        "UPDATE message_retention_coverage SET message_content_hash = ? "
        "WHERE message_id = ?",
        ("0" * 64, message_id),
    )

    assert prune_messages(conn, enabled) == 0
    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 1


def test_prune_messages_rejects_an_unknown_hash_version(hy, cfg):
    conn = hy.conn
    enabled = dataclasses.replace(cfg, message_retention_days=90)
    _session(conn, "ended", days_ago=200, summary=None, ended=True)
    message_id = _message(conn, "ended", "version-bound source", days_ago=150)
    _cover(conn, "ended", message_id, "version-bound source")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE message_retention_coverage SET hash_version = 'obsolete-v0' "
            "WHERE message_id = ?",
            (message_id,),
        )
    conn.execute("DROP TRIGGER message_coverage_peer_update_guard")
    conn.execute(
        "UPDATE message_retention_coverage SET hash_version = 'obsolete-v0' "
        "WHERE message_id = ?",
        (message_id,),
    )

    assert prune_messages(conn, enabled) == 0
    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 1


def test_pruned_message_artifact_and_kg_provenance_survive_next_chunk_cycle(
    hy, cfg
):
    conn = hy.conn
    enabled = dataclasses.replace(cfg, message_retention_days=90)
    _session(conn, "ended", days_ago=200, summary=None, ended=True)
    message_id = _message(conn, "ended", "retentionneedle source", days_ago=150)
    source_created_at = conn.execute(
        "SELECT created_at FROM messages WHERE id = ?", (message_id,)
    ).fetchone()["created_at"]
    chunk_id = _cover(conn, "ended", message_id, "retentionneedle source")
    conn.execute(
        "INSERT INTO temporal_mentions(message_id, session_id, raw_text) "
        "VALUES (?, 'ended', 'retentionneedle')",
        (message_id,),
    )
    seed_edge(conn, "service", "uses", "sqlite", pos=1)
    edge_id = conn.execute(
        "SELECT id FROM knowledge_graph WHERE subject_canonical = 'service'"
    ).fetchone()["id"]
    with core_db.evidence_mutation(conn):
        conn.execute(
            "INSERT INTO kg_evidence(edge_id, chunk_id, polarity) VALUES (?, ?, 1)",
            (edge_id, chunk_id),
        )
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM messages_fts WHERE messages_fts MATCH 'retentionneedle'"
    ).fetchone()["c"] == 1

    assert prune_messages(conn, enabled) == 1

    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 0
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM messages_fts WHERE messages_fts MATCH 'retentionneedle'"
    ).fetchone()["c"] == 0
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM temporal_mentions"
    ).fetchone()["c"] == 0
    coverage = conn.execute(
        "SELECT message_id, source_session_id, source_role, source_created_at, "
        "chunk_id FROM message_retention_coverage"
    ).fetchone()
    assert tuple(coverage) == (
        message_id,
        "ended",
        "user",
        source_created_at,
        chunk_id,
    )

    # Second retention cycle: even under chunk pressure, the only remaining
    # lossless source and its KG provenance must survive permanently.
    conn.execute(
        "UPDATE chunks SET created_at = datetime('now', '-200 days') WHERE id = ?",
        (chunk_id,),
    )
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text, created_at) VALUES "
        "('disposable', 'ended', ?, ?, 'test', 'other', "
        "datetime('now', '-200 days'))",
        (message_id, message_id),
    )
    constrained = dataclasses.replace(cfg, max_chunks=1)
    assert prune_chunks(conn, constrained) == 1

    assert conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) AS c FROM kg_evidence").fetchone()["c"] == 1
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM message_retention_coverage"
    ).fetchone()["c"] == 1
    edge = conn.execute(
        "SELECT pos_evidence, neg_evidence FROM knowledge_graph WHERE id = ?",
        (edge_id,),
    ).fetchone()
    assert (edge["pos_evidence"], edge["neg_evidence"]) == (1, 0)
    with pytest.raises(sqlite3.IntegrityError, match="raw source is absent"):
        conn.execute(
            "DELETE FROM message_retention_coverage WHERE message_id = ?",
            (message_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="raw source is absent"):
        conn.execute(
            "UPDATE message_retention_coverage SET coverage_version = 'tampered' "
            "WHERE message_id = ?",
            (message_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="covered lossless chunk"):
        conn.execute(
            "UPDATE chunks SET text = 'corrupted' WHERE id = ?", (chunk_id,)
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    with pytest.raises(RuntimeError, match="raw source is absent"):
        release_message_coverage(
            conn,
            message_id=message_id,
            chunk_id=chunk_id,
            coverage_version="test-lossless-v1",
        )


@pytest.mark.parametrize(
    "artifact_template",
    [
        # An exact record used only as a prefix is not a framed record.
        "{record} plus an unrecorded suffix",
        # Nor is the same record embedded in unrelated prose.
        "assistant quotation begins {record} quotation ends",
    ],
)
def test_record_message_coverage_rejects_prefix_and_embedded_text(
    hy, artifact_template
):
    conn = hy.conn
    _session(conn, "s", days_ago=0, summary=None)
    message_id = _message(conn, "s", "complete source text")
    record = encode_message_record(
        message_id=message_id,
        role="user",
        content="complete source text",
    )
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES ('ambiguous', 's', ?, ?, 'fallback', ?)",
        (message_id, message_id, artifact_template.format(record=record)),
    )

    with pytest.raises(ValueError, match="canonical message record"):
        record_message_coverage(
            conn,
            message_id=message_id,
            chunk_id="ambiguous",
            coverage_version="test-lossless-v1",
        )


def test_multiline_message_requires_and_accepts_exact_jsonl_framing(hy):
    conn = hy.conn
    _session(conn, "s", days_ago=0, summary=None)
    content = "first line\nassistant: looks like another record\nthird line"
    message_id = _message(conn, "s", content)
    canonical = encode_message_record(
        message_id=message_id,
        role="user",
        content=content,
    )
    assert "\n" not in canonical, "embedded newlines must be JSON escaped"
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES ('ambiguous_multiline', 's', ?, ?, "
        "'fallback', ?)",
        (message_id, message_id, f"user: {content}"),
    )
    with pytest.raises(ValueError, match="canonical message record"):
        record_message_coverage(
            conn,
            message_id=message_id,
            chunk_id="ambiguous_multiline",
            coverage_version="test-lossless-v1",
        )
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES ('multiline', 's', ?, ?, 'lossless', ?)",
        (message_id, message_id, canonical),
    )

    record_message_coverage(
        conn,
        message_id=message_id,
        chunk_id="multiline",
        coverage_version="test-lossless-v1",
    )
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM message_retention_coverage"
    ).fetchone()["c"] == 1


def test_reserved_ordered_version_rejects_nonproducer_artifact(hy):
    conn = hy.conn
    _session(conn, "reserved", days_ago=0, summary=None)
    message_id = _message(conn, "reserved", "ordered source")
    canonical = encode_message_record(
        message_id=message_id, role="user", content="ordered source"
    )
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text, chunk_kind) VALUES "
        "('not-the-producer-id', 'reserved', ?, ?, 'caller', ?, 'extraction')",
        (message_id, message_id, canonical),
    )
    with pytest.raises(ValueError, match="reserved ordered coverage"):
        record_message_coverage(
            conn,
            message_id=message_id,
            chunk_id="not-the-producer-id",
            coverage_version=LOSSLESS_COVERAGE_VERSION,
        )
    assert conn.execute(
        "SELECT COUNT(*) FROM message_retention_coverage"
    ).fetchone()[0] == 0
    assert coverage_chunk_id("reserved", message_id) != "not-the-producer-id"


def test_explicit_coverage_release_requires_the_raw_source(hy):
    conn = hy.conn
    _session(conn, "s", days_ago=0, summary=None)
    message_id = _message(conn, "s", "durable source")
    covered_chunk = _cover(conn, "s", message_id, "durable source")

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM chunks WHERE id = ?", (covered_chunk,))
    release_message_coverage(
        conn,
        message_id=message_id,
        chunk_id=covered_chunk,
        coverage_version="test-lossless-v1",
    )
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM message_retention_coverage"
    ).fetchone()["c"] == 0
    conn.execute("DELETE FROM chunks WHERE id = ?", (covered_chunk,))
    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 1


# --- Item B: retracted tombstone hard-delete (+ cascade) ---


def test_prune_retracted_edges_cascades_evidence(hy, cfg):
    conn = hy.conn
    enabled = dataclasses.replace(cfg, tombstone_retention_days=30)
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
    with core_db.evidence_mutation(conn):
        conn.execute(
            "INSERT INTO kg_evidence(edge_id, chunk_id, polarity) VALUES (?, 'c1', 1)",
            (old_id,),
        )

    pruned = prune_retracted_edges(conn, enabled)

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


@pytest.mark.parametrize("disabled_days", [0, -1])
def test_prune_retracted_edges_disabled_for_nonpositive_values(
    hy, cfg, disabled_days
):
    assert cfg.tombstone_retention_days == 0
    seed_edge(hy.conn, "history", "uses", "sqlite", status="retracted", days_ago=200)

    disabled = dataclasses.replace(cfg, tombstone_retention_days=disabled_days)
    assert prune_retracted_edges(hy.conn, disabled) == 0
    assert hy.conn.execute(
        "SELECT COUNT(*) FROM knowledge_graph WHERE subject_canonical='history'"
    ).fetchone()[0] == 1


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
    # Default episode_retention_days=0: episodes are kept forever (they're the
    # leaves of the digest tree), while stale procedures still age out on
    # retention_days.
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

    assert pruned == 1  # p_stale_old only; episodes never age out by default
    eps = {r["id"] for r in conn.execute("SELECT id FROM episodes").fetchall()}
    assert eps == {"e_old", "e_new"}
    procs = {r["id"] for r in conn.execute("SELECT id FROM procedures").fetchall()}
    assert procs == {"p_active_old", "p_stale_new"}


def test_prune_episodes_opt_in_retention(hy, cfg):
    # episode_retention_days > 0 restores age-based episode pruning, on its own
    # cutoff (30 here) rather than retention_days (90): e_mid at -40 days goes
    # too, proving the decoupling.
    conn = hy.conn
    opt_in = dataclasses.replace(cfg, episode_retention_days=30)
    _session(conn, "s", days_ago=0, summary=None)
    conn.execute(
        "INSERT INTO episodes(id, session_id, title, summary, created_at) "
        "VALUES ('e_old', 's', 'Old', 'old summary', datetime('now', '-200 days'))"
    )
    conn.execute(
        "INSERT INTO episodes(id, session_id, title, summary, created_at) "
        "VALUES ('e_mid', 's', 'Mid', 'mid summary', datetime('now', '-40 days'))"
    )
    conn.execute(
        "INSERT INTO episodes(id, session_id, title, summary, created_at) "
        "VALUES ('e_new', 's', 'New', 'new summary', datetime('now', '-1 days'))"
    )

    pruned = prune_episodes_and_procedures(conn, opt_in)

    assert pruned == 2  # e_old + e_mid
    eps = {r["id"] for r in conn.execute("SELECT id FROM episodes").fetchall()}
    assert eps == {"e_new"}
    # FTS shadow stays in sync via the existing delete trigger.
    fts = conn.execute(
        "SELECT COUNT(*) AS c FROM episodes_fts WHERE episodes_fts MATCH 'old'"
    ).fetchone()["c"]
    assert fts == 0
