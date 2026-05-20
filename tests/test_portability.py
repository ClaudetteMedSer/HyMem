"""Tests for memory export / import (improv item G).

`hy.export(path)` writes the canonical state as JSON Lines; `hy.import_(path)`
loads it back, additive and idempotent, with sessions ahead of their
dependents and FTS shadow tables kept in sync.
"""

from __future__ import annotations

import json

from hymem import HyMem, HyMemConfig

_EXPECTED = {
    "session": 1, "chunk": 1, "episode": 1,
    "procedure": 1, "edge": 1, "profile_entry": 1,
}


def _seed(hy: HyMem) -> None:
    conn = hy.conn
    conn.execute("INSERT INTO sessions(id, summary) VALUES ('s1', 'did stuff')")
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) "
        "VALUES ('c1', 's1', 1, 1, 'long_user_turn', 'we deploy postgres to prod')"
    )
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence) VALUES ('app', 'uses', 'postgres', 3)"
    )
    conn.execute(
        "INSERT INTO episodes(id, session_id, title, summary) "
        "VALUES ('e1', 's1', 'Setup', 'Configured the postgres connection pool')"
    )
    conn.execute(
        "INSERT INTO procedures(id, session_id, name, description, steps) "
        "VALUES ('p1', 's1', 'Deploy to staging', 'build and push', '[]')"
    )
    conn.execute(
        "INSERT INTO profile_entries(kind, text) VALUES ('preference', 'prefers postgres')"
    )


def test_export_import_roundtrip(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "src"))
    _seed(src)
    out = tmp_path / "export.jsonl"
    assert src.export(out) == _EXPECTED
    src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "dst"))
    try:
        assert dst.import_(out) == _EXPECTED
        # Content survived.
        assert dst.conn.execute(
            "SELECT summary FROM sessions WHERE id = 's1'"
        ).fetchone()["summary"] == "did stuff"
        # Edge is queryable through the timeline API.
        assert any(e.object == "postgres" for e in dst.timeline("app"))
        # FTS triggers fired on import → chunk and procedure are searchable.
        ctx = dst.augment("postgres deploy staging")
        assert any("postgres" in h.text for h in ctx.fts_hits)
        assert any(p.name == "Deploy to staging" for p in ctx.procedures)
    finally:
        dst.close()


def test_export_writes_meta_header(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path / "src"))
    _seed(hy)
    out = tmp_path / "export.jsonl"
    hy.export(out)
    meta = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert meta["type"] == "_meta"
    assert meta["format"] == "hymem-jsonl"
    assert meta["schema_version"] == 11
    hy.close()


def test_import_is_idempotent(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "src"))
    _seed(src)
    out = tmp_path / "export.jsonl"
    src.export(out)
    src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "dst"))
    try:
        dst.import_(out)
        second = dst.import_(out)
        assert sum(second.values()) == 0  # nothing new on re-import
        assert dst.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph"
        ).fetchone()["c"] == 1
    finally:
        dst.close()
