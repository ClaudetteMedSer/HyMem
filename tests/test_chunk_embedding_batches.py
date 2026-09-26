"""Catch-all chunk embedding transport stays bounded and resumable."""

from __future__ import annotations

import pytest

from hymem.core import db
from hymem.core.vectors import encode_vector
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.embeddings import (
    CHUNK_EMBEDDING_BATCH_SIZE,
    CHUNK_EMBEDDING_MAX_CHARS,
    chunk_embedding_id_batches,
    fetch_chunk_embeddings,
    persist_chunk_embeddings,
)
from hymem.extraction.embeddings import MappedStubEmbeddingClient, embedding_text_hash


@pytest.fixture
def conn(tmp_path):
    store = db.connect(tmp_path / "chunks.sqlite")
    db.initialize(store)
    store.execute("INSERT INTO sessions(id) VALUES ('source')")
    try:
        yield store
    finally:
        store.close()


def _chunk(conn, index: int, text: str, *, kind: str = "extraction") -> str:
    chunk_id = f"chunk-{index:04d}"
    conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES (?,'source',1,1,'test',?,?)",
        (chunk_id, text, kind),
    )
    return chunk_id


def _drain(conn, embedder, *, exclude_ids=None):
    persisted = 0
    for ids in chunk_embedding_id_batches(conn, exclude_ids=exclude_ids):
        pending = fetch_chunk_embeddings(conn, embedder, chunk_ids=ids)
        if pending is not None:
            with db.transaction(conn):
                persisted += persist_chunk_embeddings(conn, pending)
    return persisted


def test_batches_cover_all_rows_without_excluded_baseline_or_full_rescan(conn):
    count = CHUNK_EMBEDDING_BATCH_SIZE * 2 + 7
    for index in range(count):
        _chunk(conn, index, f"exact source {index}")
    excluded = {"chunk-0001", f"chunk-{CHUNK_EMBEDDING_BATCH_SIZE + 1:04d}"}
    _chunk(conn, 999, "coverage only", kind="coverage")
    embedder = MappedStubEmbeddingClient(conn=conn)

    batches = list(chunk_embedding_id_batches(conn, exclude_ids=excluded))
    assert len(batches) == 3
    assert all(len(batch) <= CHUNK_EMBEDDING_BATCH_SIZE for batch in batches)
    assert len({item for batch in batches for item in batch}) == count - len(excluded)
    assert _drain(conn, embedder, exclude_ids=excluded) == count - len(excluded)
    assert all(not state for state in embedder.transaction_states)
    assert all(len(call) <= CHUNK_EMBEDDING_BATCH_SIZE for call in embedder.calls)
    assert conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] == count - 2
    assert _drain(conn, embedder, exclude_ids=excluded) == 0
    assert _drain(conn, embedder) == 2


def test_character_budget_preserves_exact_text_and_cache_hits(conn):
    text = "  repeated exact text  " + "x" * 39900
    for index in range(5):
        _chunk(conn, index, text)
    embedder = MappedStubEmbeddingClient(conn=conn)
    batches = list(chunk_embedding_id_batches(conn))
    assert [len(batch) for batch in batches] == [3, 2]
    assert _drain(conn, embedder) == 5
    assert embedder.calls == [[text] * 3]
    assert all(sum(len(value) for value in call) <= CHUNK_EMBEDDING_MAX_CHARS
               for call in embedder.calls)


def test_later_provider_failure_keeps_earlier_committed_batch(conn):
    for index in range(CHUNK_EMBEDDING_BATCH_SIZE + 2):
        _chunk(conn, index, "POISON" if index == CHUNK_EMBEDDING_BATCH_SIZE else f"source {index}")
    embedder = MappedStubEmbeddingClient(fail_on="POISON", conn=conn)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        _drain(conn, embedder)
    assert conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] == CHUNK_EMBEDDING_BATCH_SIZE
    assert all(not state for state in embedder.transaction_states)


def test_public_fetch_is_bounded_and_oversize_is_explicit(conn):
    for index in range(CHUNK_EMBEDDING_BATCH_SIZE + 1):
        _chunk(conn, index, f"source {index}")
    embedder = MappedStubEmbeddingClient(conn=conn)
    pending = fetch_chunk_embeddings(conn, embedder)
    assert pending is not None and len(pending.ids) == CHUNK_EMBEDDING_BATCH_SIZE
    assert len(embedder.calls[0]) == CHUNK_EMBEDDING_BATCH_SIZE

    _chunk(conn, 999, "z" * (CHUNK_EMBEDDING_MAX_CHARS + 1))
    assert list(chunk_embedding_id_batches(conn))[-1] == ("chunk-0999",)
    with pytest.raises(ValueError, match="character limit"):
        fetch_chunk_embeddings(conn, embedder, chunk_ids=("chunk-0999",))
    assert all("z" * (CHUNK_EMBEDDING_MAX_CHARS + 1) not in call for call in embedder.calls)


def test_oversized_current_or_cached_chunk_needs_no_provider_request(conn):
    current_text = "c" * (CHUNK_EMBEDDING_MAX_CHARS + 1)
    cached_text = "d" * (CHUNK_EMBEDDING_MAX_CHARS + 1)
    current_id = _chunk(conn, 0, current_text)
    cached_id = _chunk(conn, 1, cached_text)
    embedder = MappedStubEmbeddingClient(conn=conn)
    model, dim = embedding_storage_identity(embedder)
    vector = encode_vector([0.0, 1.0, 0.0])
    with db.embedding_mutation(conn):
        conn.execute(
            "INSERT INTO chunk_embeddings(chunk_id,vector_json,model,dim,text_hash) "
            "VALUES (?,?,?,?,?)",
            (current_id, vector, model, dim, embedding_text_hash(current_text)),
        )
        conn.execute(
            "INSERT INTO embedding_cache(text_hash,model,vector_json,dim) "
            "VALUES (?,?,?,?)",
            (embedding_text_hash(cached_text), model, vector, dim),
        )
    assert list(chunk_embedding_id_batches(conn)) == [(current_id,), (cached_id,)]
    assert _drain(conn, embedder) == 1
    assert embedder.calls == []
    assert conn.execute(
        "SELECT text_hash FROM chunk_embeddings WHERE chunk_id=?", (cached_id,)
    ).fetchone()[0] == embedding_text_hash(cached_text)


def test_stale_and_malformed_stored_vectors_are_reembedded(conn):
    chunk_id = _chunk(conn, 0, "exact source")
    embedder = MappedStubEmbeddingClient(conn=conn)
    assert _drain(conn, embedder) == 1
    with db.embedding_mutation(conn):
        conn.execute(
            "UPDATE chunk_embeddings SET text_hash='stale',vector_json='[0,0,0]' "
            "WHERE chunk_id=?", (chunk_id,),
        )
    # A valid cache entry repairs the stale mirror without a provider call.
    embedder.calls.clear()
    assert _drain(conn, embedder) == 1
    assert embedder.calls == []
    with db.embedding_mutation(conn):
        conn.execute("DELETE FROM embedding_cache")
        conn.execute(
            "UPDATE chunk_embeddings SET vector_json='[0,0,0]' WHERE chunk_id=?",
            (chunk_id,),
        )
    assert _drain(conn, embedder) == 1
    assert embedder.calls == [["exact source"]]
