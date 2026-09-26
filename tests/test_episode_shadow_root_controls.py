"""Independent root acceptance of surplus-only episode shadow repair."""
import json
import sqlite3

import pytest

from hymem import HyMem, StubEmbeddingClient
from hymem.core import db
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.embeddings import embedding_text_hash
from hymem.extraction.llm import StubLLMClient
from tests.test_db_shadows import _seed_episode


@pytest.fixture
def populated(cfg):
    pytest.importorskip("sqlite_vec")
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    embed = StubEmbeddingClient()
    model, dim = embedding_storage_identity(embed)
    conn = hy.conn
    with db.transaction(conn), db.embedding_mutation(conn):
        _seed_episode(conn, "root-valid", "Root valid")
        vector = embed.embed(["Root valid\nsummary of Root valid"])[0]
        conn.execute(
            "INSERT INTO episode_embeddings(episode_id,vector_json,model,dim,"
            "text_hash,embedding_producer_key) VALUES (?,?,?,?,?,?)",
            ("root-valid", json.dumps(vector), model, dim,
             embedding_text_hash("Root valid\nsummary of Root valid"), model),
        )
    db.ensure_vec_table(conn, dim, model=model)
    if not db.has_vec_table(conn, table="vec_episodes"):
        hy.close()
        pytest.skip("sqlite-vec unavailable")
    assert db.vec_episodes_aligned(conn)
    key = db.episode_vector_rowids(["root-valid"])["root-valid"]
    for extra in (41, 42, 43):
        assert extra != key
        conn.execute("INSERT INTO vec_episodes(rowid,embedding) VALUES (?,?)",
                     (extra, db._pack_vector(vector)))
    try:
        yield conn, key, dim
    finally:
        hy.close()


def snapshot(conn):
    result = {}
    for table in ("episodes", "episode_embeddings", "sessions", "messages",
                  "message_retention_coverage", "episode_message_sources",
                  "schema_meta", "aggregation_build_health", "run_lock"):
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            result[table] = sorted(repr(tuple(row)) for row in conn.execute(
                'SELECT * FROM "' + table + '"'))
    return result


def test_root_preserves_authoritative_rows_valid_vector_and_repeat(populated):
    conn, key, _ = populated
    before = snapshot(conn)
    vector = bytes(conn.execute("SELECT embedding FROM vec_episodes WHERE rowid=?",
                                (key,)).fetchone()[0])
    assert db.prune_extra_episode_vectors(conn)
    assert snapshot(conn) == before
    assert [(int(r[0]), bytes(r[1])) for r in conn.execute(
        "SELECT rowid,embedding FROM vec_episodes")] == [(key, vector)]
    assert db.prune_extra_episode_vectors(conn) is False
    assert snapshot(conn) == before
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()


@pytest.mark.parametrize("dimension", ["0", "bad", "-1"])
def test_root_unverifiable_dimension_never_deletes(populated, dimension):
    conn, _, _ = populated
    conn.execute("UPDATE schema_meta SET value=? WHERE key='vec_dim'", (dimension,))
    before = [tuple(r) for r in conn.execute("SELECT rowid,embedding FROM vec_episodes")]
    assert db.prune_extra_episode_vectors(conn) is False
    assert [tuple(r) for r in conn.execute("SELECT rowid,embedding FROM vec_episodes")] == before


def test_root_producer_mismatch_never_promotes_or_deletes(populated):
    conn, _, _ = populated
    conn.execute("UPDATE schema_meta SET value=? WHERE key='vec_model'",
                 ("hymem-embedding-producer-v1:" + "a" * 64,))
    before = [tuple(r) for r in conn.execute("SELECT rowid,embedding FROM vec_episodes")]
    assert db.prune_extra_episode_vectors(conn) is False
    assert [tuple(r) for r in conn.execute("SELECT rowid,embedding FROM vec_episodes")] == before


def test_root_commit_failure_rolls_back_all_surplus_deletions(populated):
    conn, _, _ = populated
    before = [tuple(r) for r in conn.execute("SELECT rowid,embedding FROM vec_episodes")]

    class CommitFault:
        def __getattr__(self, name):
            return getattr(conn, name)

        def execute(self, sql, *args):
            if sql == "COMMIT":
                raise sqlite3.OperationalError("synthetic commit failure")
            return conn.execute(sql, *args)

    with pytest.raises(sqlite3.OperationalError, match="synthetic commit failure"):
        db.prune_extra_episode_vectors(CommitFault())
    assert not conn.in_transaction
    assert [tuple(r) for r in conn.execute("SELECT rowid,embedding FROM vec_episodes")] == before
    assert db.prune_extra_episode_vectors(conn)


def test_root_partial_delete_failure_rolls_back_earlier_deletion(populated):
    conn, _, _ = populated
    before = [tuple(r) for r in conn.execute("SELECT rowid,embedding FROM vec_episodes")]

    class DeleteFault:
        deletes = 0

        def __getattr__(self, name):
            return getattr(conn, name)

        def execute(self, sql, *args):
            if sql.startswith("DELETE FROM vec_episodes WHERE"):
                self.deletes += 1
                if self.deletes == 2:
                    raise sqlite3.OperationalError("synthetic second delete failure")
            return conn.execute(sql, *args)

    proxy = DeleteFault()
    with pytest.raises(sqlite3.OperationalError, match="synthetic second delete failure"):
        db.prune_extra_episode_vectors(proxy)
    assert proxy.deletes == 2 and not conn.in_transaction
    assert [tuple(r) for r in conn.execute("SELECT rowid,embedding FROM vec_episodes")] == before
