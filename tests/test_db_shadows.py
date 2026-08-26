"""Rowid-shadow integrity across VACUUM (hymem/core/db.py).

episodes / chunks / aggregation_nodes have TEXT primary keys, so their rowids
are implicit — and SQLite's VACUUM may renumber implicit rowids when earlier
deletions left gaps. Everything keyed on those rowids from the outside
(external-content FTS tables, the vec_* mirrors) silently decouples when that
happens: FTS match hits join back to the WRONG rows and vec KNN hits translate
to the WRONG ids. That skew was a root cause of the 2026-07-12 RAPTOR
fusion-reuse instability (candidate blocking clustered on garbage
neighborhoods after post-prune VACUUMs) and quietly corrupts plain retrieval
too. `resync_rowid_shadows` is the repair; `heal_rowid_shadows` is the
dream-time probe-and-repair for stores skewed by pre-fix VACUUMs.

The vec_* half needs a Python whose sqlite3 supports loadable extensions;
those assertions skip cleanly where it doesn't (macOS framework builds) and
run in CI / on the Hermes box.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem.core import db as core_db
from hymem.extraction.llm import StubLLMClient


def _extension_loading_available() -> bool:
    return hasattr(sqlite3.Connection, "enable_load_extension")


@pytest.fixture
def conn(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    yield hy.conn
    hy.close()


def _seed_episode(conn, eid, title):
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s1')")
    conn.execute(
        """INSERT INTO episodes(id, session_id, title, summary, participants,
                                start_message_id, end_message_id, outcome, key_entities)
           VALUES (?, 's1', ?, ?, '[]', 1, 2, NULL, '[]')""",
        (eid, title, f"summary of {title}"),
    )


def _fts_hits(conn, term: str) -> set[str]:
    """Episode ids reached the way retrieval reaches them: FTS rowid → episodes
    row. After a renumbering VACUUM these joins land on the wrong rows."""
    return {
        r["id"]
        for r in conn.execute(
            """SELECT e.id FROM episodes_fts f
               JOIN episodes e ON e.rowid = f.rowid
               WHERE episodes_fts MATCH ?""",
            (term,),
        )
    }


def test_resync_restores_fts_joins_after_vacuum(conn):
    # Distinct, FTS-friendly marker words; one deletion creates the rowid gap
    # a later VACUUM compacts away.
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    with core_db.transaction(conn):
        for i, w in enumerate(words):
            _seed_episode(conn, f"e{i}", f"notes about {w}")
    with core_db.transaction(conn):
        conn.execute("DELETE FROM episodes WHERE id = 'e0'")   # gap at rowid 1

    conn.execute("VACUUM")
    core_db.resync_rowid_shadows(conn)

    # Whether or not this SQLite build renumbered, every surviving episode must
    # be reachable through the FTS join under its own marker word.
    for i, w in enumerate(words):
        if i == 0:
            continue
        assert _fts_hits(conn, w) == {f"e{i}"}, f"FTS join broken for {w}"


def test_resync_is_idempotent_and_safe_on_fresh_store(conn):
    core_db.resync_rowid_shadows(conn)      # empty store: must not raise
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "notes about foxtrot")
    core_db.resync_rowid_shadows(conn)
    core_db.resync_rowid_shadows(conn)
    assert _fts_hits(conn, "foxtrot") == {"e1"}


def test_heal_is_noop_when_unverifiable(conn):
    # Without the vec extension (or before any embeddings exist) the probe
    # reports "unverifiable", and heal must not touch anything.
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "notes about golf")
    if not _extension_loading_available():
        assert core_db.vec_episodes_aligned(conn) is True
        assert core_db.heal_rowid_shadows(conn) is False


@pytest.mark.skipif(
    not _extension_loading_available(),
    reason="sqlite3 built without loadable-extension support",
)
def test_vec_episodes_misalignment_is_detected_and_healed(cfg):
    pytest.importorskip("sqlite_vec")
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    conn = hy.conn
    embed = StubEmbeddingClient()
    try:
        with core_db.transaction(conn):
            for i in range(6):
                _seed_episode(conn, f"e{i}", f"episode number {i}")
        vecs = embed.embed([f"episode number {i}" for i in range(6)])
        with core_db.transaction(conn):
            for i, v in enumerate(vecs):
                conn.execute(
                    "INSERT INTO episode_embeddings(episode_id, vector_json, model, dim, text_hash) "
                    "VALUES (?, ?, 'stub', ?, ?)",
                    (f"e{i}", json.dumps(v), len(v), f"h{i}"),
                )
        core_db.ensure_vec_table(conn, len(vecs[0]))
        if not core_db.has_vec_table(conn, table="vec_episodes"):
            pytest.skip("vec extension present but vec table unavailable")
        assert core_db.vec_episodes_aligned(conn) is True

        # Delete a low-rowid episode (cascades its embedding row), then VACUUM:
        # surviving episodes' rowids compact downward while vec_episodes keeps
        # the old numbering — the prod skew in miniature.
        with core_db.transaction(conn):
            conn.execute("DELETE FROM episodes WHERE id = 'e0'")
        conn.execute("VACUUM")

        if core_db.vec_episodes_aligned(conn):
            pytest.skip("this SQLite build did not renumber rowids on VACUUM")
        assert core_db.heal_rowid_shadows(conn) is True
        assert core_db.vec_episodes_aligned(conn) is True
    finally:
        hy.close()
