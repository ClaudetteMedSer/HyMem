from __future__ import annotations

import json
import sqlite3

from hymem.extraction.embeddings import EmbeddingClient


def embed_pending_chunks(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> int:
    """Embed every chunk that doesn't yet have a row in chunk_embeddings.

    Returns the number of chunks newly embedded. Caller wraps in a transaction.
    """
    rows = conn.execute(
        """
        SELECT c.id, c.text FROM chunks c
        LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
        WHERE e.chunk_id IS NULL
        ORDER BY c.id
        """
    ).fetchall()
    if not rows:
        return 0

    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    vectors = embedder.embed(texts)
    if len(vectors) != len(ids):
        raise RuntimeError(
            f"embedding client returned {len(vectors)} vectors for {len(ids)} chunks"
        )

    model = embedder.model
    for chunk_id, vec in zip(ids, vectors):
        conn.execute(
            """
            INSERT OR REPLACE INTO chunk_embeddings(chunk_id, vector_json, model, dim)
            VALUES (?, ?, ?, ?)
            """,
            (chunk_id, json.dumps(vec), model, len(vec)),
        )
    return len(ids)
