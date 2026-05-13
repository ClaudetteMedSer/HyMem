from __future__ import annotations

import json
import sqlite3

from hymem.core import db as core_db
from hymem.extraction.embeddings import EmbeddingClient


def embed_pending_chunks(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> int:
    """Embed every chunk that doesn't yet have a row in chunk_embeddings.

    Returns the number of chunks newly embedded. Caller wraps in a transaction.
    """
    rows = conn.execute(
        """
        SELECT c.id, c.rowid, c.text FROM chunks c
        LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
        WHERE e.chunk_id IS NULL
        ORDER BY c.id
        """
    ).fetchall()
    if not rows:
        return 0

    ids = [r["id"] for r in rows]
    chunk_rowids = [r["rowid"] for r in rows]
    texts = [r["text"] for r in rows]
    vectors = embedder.embed(texts)
    if len(vectors) != len(ids):
        raise RuntimeError(
            f"embedding client returned {len(vectors)} vectors for {len(ids)} chunks"
        )

    dim = embedder.dim
    model = embedder.model
    core_db.ensure_vec_table(conn, dim)

    for chunk_id, chunk_rowid, vec in zip(ids, chunk_rowids, vectors):
        conn.execute(
            """
            INSERT OR REPLACE INTO chunk_embeddings(chunk_id, vector_json, model, dim)
            VALUES (?, ?, ?, ?)
            """,
            (chunk_id, json.dumps(vec), model, len(vec)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (chunk_rowid, core_db._pack_vector(vec)),
        )
    return len(ids)
