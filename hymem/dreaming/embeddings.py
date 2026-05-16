from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from hymem.core import db as core_db
from hymem.extraction.embeddings import EmbeddingClient


@dataclass
class PendingChunkEmbeddings:
    ids: list[str]
    chunk_rowids: list[int]
    vectors: list[list[float]]
    dim: int
    model: str


@dataclass
class PendingEdgeEmbeddings:
    edge_text_by_id: dict[int, str]
    new_text_vectors: dict[str, list[float]] = field(default_factory=dict)
    dim: int = 0
    model: str = ""


def fetch_chunk_embeddings(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> PendingChunkEmbeddings | None:
    """Read pending chunks and call the embedder. No write transaction held.

    Returns None when there are no chunks to embed.
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
        return None

    ids = [r["id"] for r in rows]
    chunk_rowids = [r["rowid"] for r in rows]
    texts = [r["text"] for r in rows]
    vectors = embedder.embed(texts)
    if len(vectors) != len(ids):
        raise RuntimeError(
            f"embedding client returned {len(vectors)} vectors for {len(ids)} chunks"
        )
    return PendingChunkEmbeddings(
        ids=ids,
        chunk_rowids=chunk_rowids,
        vectors=vectors,
        dim=embedder.dim,
        model=embedder.model,
    )


def persist_chunk_embeddings(
    conn: sqlite3.Connection, pending: PendingChunkEmbeddings
) -> int:
    """Insert pending chunk vectors into chunk_embeddings + vec_chunks.
    Caller wraps in core_db.transaction()."""
    core_db.ensure_vec_table(conn, pending.dim)
    for chunk_id, chunk_rowid, vec in zip(pending.ids, pending.chunk_rowids, pending.vectors):
        conn.execute(
            """
            INSERT OR REPLACE INTO chunk_embeddings(chunk_id, vector_json, model, dim)
            VALUES (?, ?, ?, ?)
            """,
            (chunk_id, json.dumps(vec), pending.model, len(vec)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (chunk_rowid, core_db._pack_vector(vec)),
        )
    return len(pending.ids)


def fetch_edge_embeddings(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> PendingEdgeEmbeddings | None:
    """Read active edges, determine which triple texts are uncached, and embed
    only those. No write transaction held.

    Returns None when there are no active edges. The cache (edge_embeddings)
    is keyed on triple text, not edge id, so derived edges — whose ids churn
    every dream run — reuse their vector instead of re-hitting the API.
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
        return None

    edge_text_by_id = {
        r["id"]: f"{r['subject_canonical']} {r['predicate']} {r['object_canonical']}"
        for r in rows
    }

    pending_texts = sorted(
        text
        for text in set(edge_text_by_id.values())
        if conn.execute(
            "SELECT 1 FROM edge_embeddings WHERE edge_text = ?", (text,)
        ).fetchone()
        is None
    )

    new_text_vectors: dict[str, list[float]] = {}
    if pending_texts:
        vectors = embedder.embed(pending_texts)
        if len(vectors) != len(pending_texts):
            raise RuntimeError(
                f"embedding client returned {len(vectors)} vectors "
                f"for {len(pending_texts)} edges"
            )
        new_text_vectors = dict(zip(pending_texts, vectors))

    return PendingEdgeEmbeddings(
        edge_text_by_id=edge_text_by_id,
        new_text_vectors=new_text_vectors,
        dim=embedder.dim,
        model=embedder.model,
    )


def persist_edge_embeddings(
    conn: sqlite3.Connection, pending: PendingEdgeEmbeddings
) -> int:
    """Persist newly embedded triple texts and rebuild vec_edges from cache.
    Caller wraps in core_db.transaction()."""
    for text, vec in pending.new_text_vectors.items():
        conn.execute(
            """
            INSERT OR REPLACE INTO edge_embeddings(edge_text, vector_json, model, dim)
            VALUES (?, ?, ?, ?)
            """,
            (text, json.dumps(vec), pending.model, len(vec)),
        )

    core_db.ensure_vec_table(conn, pending.dim)

    # Rebuild vec_edges from scratch: derived edge ids churn every dream run, so
    # a full clear + reinsert is simpler and cheaper than reconciling rowids
    # (vec0 virtual tables don't support INSERT OR REPLACE on the primary key).
    if core_db.has_vec_table(conn, table="vec_edges"):
        conn.execute("DELETE FROM vec_edges")
        for edge_id, text in pending.edge_text_by_id.items():
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
    return len(pending.new_text_vectors)
