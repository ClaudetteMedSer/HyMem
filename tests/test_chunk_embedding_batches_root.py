"""Independent review controls for bounded chunk embedding publication."""
import pytest

from hymem import HyMem, HyMemConfig
from hymem.core import db
from hymem.extraction.embeddings import MappedStubEmbeddingClient
from hymem.extraction.llm import StubLLMClient
from hymem.dreaming.embeddings import CHUNK_EMBEDDING_BATCH_SIZE, chunk_embedding_id_batches, fetch_chunk_embeddings
from tests.test_chunk_embedding_batches import _chunk, conn


def test_real_dream_drains_multiple_batches_without_provider_write_lock(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path), llm=StubLLMClient())
    embed = MappedStubEmbeddingClient(conn=hy.conn)
    hy.set_embedding_client(embed)
    try:
        hy.open_session('source')
        for index in range(135):
            _chunk(hy.conn, index, f'unique synthetic {index}')
        report = hy.dream(session_ids=['source'])
        assert report.chunks_embedded == 135
        assert hy.conn.execute('SELECT COUNT(*) FROM chunk_embeddings').fetchone()[0] == 135
        assert all(len(batch) <= CHUNK_EMBEDDING_BATCH_SIZE for batch in embed.calls)
        assert all(not state for state in embed.transaction_states)
    finally:
        hy.close()


def test_bad_vectors_do_not_publish_partial_batch(conn):
    for index in range(3):
        _chunk(conn, index, f'synthetic {index}')
    with pytest.raises(RuntimeError, match='malformed'):
        fetch_chunk_embeddings(conn, MappedStubEmbeddingClient(default=[0, 0, 0]),
                               chunk_ids=next(chunk_embedding_id_batches(conn)))
    assert conn.execute('SELECT COUNT(*) FROM chunk_embeddings').fetchone()[0] == 0


def test_text_change_between_selection_and_fetch_rechecks_size(conn):
    chunk = _chunk(conn, 0, 'synthetic')
    batch = next(chunk_embedding_id_batches(conn))
    conn.execute('UPDATE chunks SET text=? WHERE id=?', ('x' * 128001, chunk))
    embed = MappedStubEmbeddingClient()
    with pytest.raises(ValueError, match='character limit'):
        fetch_chunk_embeddings(conn, embed, chunk_ids=batch)
    assert embed.calls == []
