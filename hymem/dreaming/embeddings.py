from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import cast

from hymem.core import db as core_db
from hymem.extraction.embeddings import EmbeddingClient, normalize_text


@dataclass
class PendingChunkEmbeddings:
    ids: list[str]
    chunk_rowids: list[int]
    vectors: list[list[float]]
    dim: int
    model: str
    text_hashes: list[str]
    from_cache: list[bool]
    cache_hits: int = 0


@dataclass
class PendingEdgeEmbeddings:
    edge_text_by_id: dict[int, str]
    new_text_vectors: dict[str, list[float]] = field(default_factory=dict)
    truly_new_texts: set[str] = field(default_factory=set)
    dim: int = 0
    model: str = ""
    cache_hits: int = 0


def fetch_chunk_embeddings(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> PendingChunkEmbeddings | None:
    """Read pending chunks and embed them, consulting embedding_cache first.

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
    model = embedder.model

    text_hashes = [
        hashlib.sha256(normalize_text(t).encode()).hexdigest() for t in texts
    ]
    cached_by_hash = _fetch_cached_vectors(conn, text_hashes, model)

    vectors_out: list[list[float] | None] = [None] * len(texts)
    from_cache = [False] * len(texts)
    miss_indices: list[int] = []
    miss_texts: list[str] = []

    for i, text_hash in enumerate(text_hashes):
        cached = cached_by_hash.get(text_hash)
        if cached is not None:
            vectors_out[i] = cached
            from_cache[i] = True
        else:
            miss_indices.append(i)
            miss_texts.append(texts[i])

    if miss_texts:
        embedded = embedder.embed(miss_texts)
        if len(embedded) != len(miss_texts):
            raise RuntimeError(
                f"embedding client returned {len(embedded)} vectors for {len(miss_texts)} chunks"
            )
        for idx, vec in zip(miss_indices, embedded):
            vectors_out[idx] = vec

    assert all(v is not None for v in vectors_out), "chunk embedding slot unfilled"
    final_vectors = cast(list[list[float]], vectors_out)

    return PendingChunkEmbeddings(
        ids=ids,
        chunk_rowids=chunk_rowids,
        vectors=final_vectors,
        dim=embedder.dim,
        model=model,
        text_hashes=text_hashes,
        from_cache=from_cache,
        cache_hits=sum(from_cache),
    )


def _fetch_cached_vectors(
    conn: sqlite3.Connection, text_hashes: list[str], model: str
) -> dict[str, list[float]]:
    """Single batched lookup against embedding_cache for the given hashes."""
    unique_hashes = list({h for h in text_hashes})
    if not unique_hashes:
        return {}
    placeholders = ",".join("?" * len(unique_hashes))
    rows = conn.execute(
        f"SELECT text_hash, vector_json FROM embedding_cache "
        f"WHERE model = ? AND text_hash IN ({placeholders})",
        (model, *unique_hashes),
    ).fetchall()
    return {r["text_hash"]: json.loads(r["vector_json"]) for r in rows}


def persist_chunk_embeddings(
    conn: sqlite3.Connection, pending: PendingChunkEmbeddings
) -> int:
    """Insert pending chunk vectors into chunk_embeddings + vec_chunks.
    Cache misses are also written to embedding_cache.
    Caller wraps in core_db.transaction()."""
    core_db.ensure_vec_table(conn, pending.dim)
    for chunk_id, chunk_rowid, vec, text_hash, is_cached in zip(
        pending.ids,
        pending.chunk_rowids,
        pending.vectors,
        pending.text_hashes,
        pending.from_cache,
    ):
        if not is_cached:
            conn.execute(
                """
                INSERT OR IGNORE INTO embedding_cache(text_hash, model, vector_json, dim)
                VALUES (?, ?, ?, ?)
                """,
                (text_hash, pending.model, json.dumps(vec), len(vec)),
            )
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
    truly_new_texts: set[str] = set()
    cache_hits = 0
    model = embedder.model

    if pending_texts:
        text_hashes = {
            text: hashlib.sha256(normalize_text(text).encode()).hexdigest()
            for text in pending_texts
        }
        cached_by_hash = _fetch_cached_vectors(
            conn, list(text_hashes.values()), model
        )

        miss_texts: list[str] = []
        for text in pending_texts:
            cached = cached_by_hash.get(text_hashes[text])
            if cached is not None:
                new_text_vectors[text] = cached
                cache_hits += 1
            else:
                miss_texts.append(text)

        if miss_texts:
            vectors = embedder.embed(miss_texts)
            if len(vectors) != len(miss_texts):
                raise RuntimeError(
                    f"embedding client returned {len(vectors)} vectors "
                    f"for {len(miss_texts)} edges"
                )
            new_text_vectors.update(dict(zip(miss_texts, vectors)))
            truly_new_texts = set(miss_texts)

    return PendingEdgeEmbeddings(
        edge_text_by_id=edge_text_by_id,
        new_text_vectors=new_text_vectors,
        truly_new_texts=truly_new_texts,
        dim=embedder.dim,
        model=model,
        cache_hits=cache_hits,
    )


def persist_edge_embeddings(
    conn: sqlite3.Connection, pending: PendingEdgeEmbeddings
) -> int:
    """Persist newly embedded triple texts and rebuild vec_edges from cache.
    True embedding misses are also written to embedding_cache.
    Caller wraps in core_db.transaction()."""
    for text, vec in pending.new_text_vectors.items():
        if text in pending.truly_new_texts:
            text_hash = hashlib.sha256(normalize_text(text).encode()).hexdigest()
            conn.execute(
                """
                INSERT OR IGNORE INTO embedding_cache(text_hash, model, vector_json, dim)
                VALUES (?, ?, ?, ?)
                """,
                (text_hash, pending.model, json.dumps(vec), len(vec)),
            )
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
