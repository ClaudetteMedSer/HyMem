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


def embed_pending_edges(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> int:
    """Embed every active edge whose triple text isn't yet cached, then refresh
    vec_edges rowids for all active edges.

    The cache (edge_embeddings) is keyed on triple text, not edge id, so derived
    edges — whose ids churn every dream run — reuse their vector instead of
    re-hitting the embedding API. Returns the number of newly embedded texts.
    Caller wraps in a transaction.
    """
    rows = conn.execute(
        """
        SELECT id, subject_canonical, predicate, object_canonical
        FROM knowledge_graph
        WHERE status = 'active'
        ORDER BY id
        """
    ).fetchall()
    if not rows:
        return 0

    edge_text_by_id = {
        r["id"]: f"{r['subject_canonical']} {r['predicate']} {r['object_canonical']}"
        for r in rows
    }

    # Embed only triple texts not already cached.
    pending_texts = sorted(
        text
        for text in set(edge_text_by_id.values())
        if conn.execute(
            "SELECT 1 FROM edge_embeddings WHERE edge_text = ?", (text,)
        ).fetchone()
        is None
    )
    newly_embedded = 0
    if pending_texts:
        vectors = embedder.embed(pending_texts)
        if len(vectors) != len(pending_texts):
            raise RuntimeError(
                f"embedding client returned {len(vectors)} vectors "
                f"for {len(pending_texts)} edges"
            )
        model = embedder.model
        for text, vec in zip(pending_texts, vectors):
            conn.execute(
                """
                INSERT OR REPLACE INTO edge_embeddings(edge_text, vector_json, model, dim)
                VALUES (?, ?, ?, ?)
                """,
                (text, json.dumps(vec), model, len(vec)),
            )
        newly_embedded = len(pending_texts)

    core_db.ensure_vec_table(conn, embedder.dim)

    # Rebuild vec_edges from scratch: derived edge ids churn every dream run, so
    # a full clear + reinsert is simpler and cheaper than reconciling rowids
    # (vec0 virtual tables don't support INSERT OR REPLACE on the primary key).
    if core_db.has_vec_table(conn, table="vec_edges"):
        conn.execute("DELETE FROM vec_edges")
        for edge_id, text in edge_text_by_id.items():
            emb = conn.execute(
                "SELECT vector_json FROM edge_embeddings WHERE edge_text = ?",
                (text,),
            ).fetchone()
            if emb is None:
                continue
            vec = json.loads(emb["vector_json"])
            conn.execute(
                "INSERT INTO vec_edges(rowid, embedding) VALUES (?, ?)",
                (edge_id, core_db._pack_vector(vec)),
            )
    return newly_embedded
