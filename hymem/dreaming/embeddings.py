from __future__ import annotations

import math
import re
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterator, Sequence, cast

from hymem.core import db as core_db
from hymem.core.graph import live_edge_predicate
from hymem.core.vectors import decode_vector, encode_vector
from hymem.dreaming.lossless import (
    COVERAGE_VALIDATION_COLUMNS,
    COVERAGE_VALIDATION_JOINS,
    validate_message_coverage_row,
)
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
from hymem.dreaming.facts import load_fact_source_manifests
from hymem.extraction.embeddings import EmbeddingClient, embedding_text_hash


MESSAGE_EMBEDDING_BATCH_SIZE = 64
FACT_EMBEDDING_BATCH_SIZE = 64
FACT_EMBEDDING_SCAN_SIZE = 256


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


@dataclass
class PendingEpisodeEmbeddings:
    """One-shot batch for episode embeddings. ``ids`` and ``rowids`` align
    with ``vectors``/``text_hashes``/``from_cache`` by index."""

    ids: list[str]
    rowids: list[int]
    vectors: list[list[float]]
    text_hashes: list[str]
    from_cache: list[bool]
    dim: int
    model: str
    cache_hits: int = 0


@dataclass
class PendingFactEmbeddings:
    """One-shot batch for current proof-valid narrative-fact embeddings.

    ``fact_ids`` align with ``vectors``/``text_hashes``/``from_cache`` by
    index. Immutable fact payloads may be retracted and resurrected; only the
    current authoritative projection may cross the provider boundary.
    """

    fact_ids: list[int]
    vectors: list[list[float]]
    text_hashes: list[str]
    from_cache: list[bool]
    dim: int
    model: str
    cache_hits: int = 0
    scan_state_key: str | None = None
    scan_after_id: int = 0
    scan_start_value: str | None = None
    scan_next_value: str | None = None


@dataclass
class PendingMessageEmbeddings:
    """Validated vectors for exact durable message occurrences.

    The embedding call happens while this object is assembled; persistence is
    a separate, short transaction and rechecks the immutable proof hash before
    writing.  Lists align by index.
    """

    message_ids: list[int]
    session_ids: list[str]
    chunk_ids: list[str]
    coverage_versions: list[str]
    content_hashes: list[str]
    text_hashes: list[str]
    vectors: list[list[float]]
    from_cache: list[bool]
    dim: int
    model: str
    cache_hits: int = 0


@dataclass
class ChunkEmbedRequest:
    """Per-batch state captured on the main thread before a background embed.

    Decouples the SQLite cache lookup (must run on the writer thread) from the
    embedding API call (offloaded to a background thread). After the embedder
    returns, ``assemble_chunk_pending`` merges these halves and produces a
    ``PendingChunkEmbeddings`` ready to persist.
    """

    ids: list[str]
    texts: list[str]
    text_hashes: list[str]
    cached_by_hash: dict[str, list[float]]
    miss_indices: list[int]
    miss_texts: list[str]
    model: str
    dim: int


def fetch_chunk_embeddings(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> PendingChunkEmbeddings | None:
    """Read pending chunks and embed them, consulting embedding_cache first.

    Returns None when there are no chunks to embed.
    """
    model, dim = _embedding_identity(embedder)
    all_rows = conn.execute(
        """
        SELECT c.id, c.rowid, c.text,
               e.vector_json AS stored_vector, e.model AS stored_model,
               e.dim AS stored_dim, e.text_hash AS stored_text_hash
        FROM chunks c
        LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
        WHERE c.chunk_kind = 'extraction'
        ORDER BY c.id
        """
    ).fetchall()
    rows = []
    for row in all_rows:
        current_hash = embedding_text_hash(row["text"])
        stored = None
        if (
            row["stored_text_hash"] == current_hash
            and row["stored_model"] == model
            and row["stored_dim"] == dim
        ):
            try:
                decoded = decode_vector(row["stored_vector"])
            except (AttributeError, UnicodeError, TypeError, ValueError):
                decoded = None
            stored = _finite_embedding_vector(decoded, expected_dim=dim)
        if stored is None:
            rows.append(row)
    if not rows:
        return None

    ids = [r["id"] for r in rows]
    chunk_rowids = [r["rowid"] for r in rows]
    texts = [r["text"] for r in rows]
    text_hashes = [embedding_text_hash(t) for t in texts]
    cached_by_hash = _fetch_cached_vectors(
        conn, text_hashes, model, expected_dim=dim
    )

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
        final_dim = _post_embed_identity(embedder, expected_model=model)
    else:
        final_dim = dim
    if final_dim != dim and any(from_cache):
        raise RuntimeError("embedding dimension changed with cached chunk vectors")
    final_vectors_or_none = [
        _finite_embedding_vector(vector, expected_dim=final_dim)
        for vector in vectors_out
    ]
    if any(vector is None for vector in final_vectors_or_none):
        raise RuntimeError("embedding client returned malformed chunk vectors")
    final_vectors = cast(list[list[float]], final_vectors_or_none)

    return PendingChunkEmbeddings(
        ids=ids,
        chunk_rowids=chunk_rowids,
        vectors=final_vectors,
        dim=final_dim,
        model=model,
        text_hashes=text_hashes,
        from_cache=from_cache,
        cache_hits=sum(from_cache),
    )


def _fetch_cached_vectors(
    conn: sqlite3.Connection,
    text_hashes: list[str],
    model: str,
    *,
    expected_dim: int | None = None,
) -> dict[str, list[float]]:
    """Single batched lookup against embedding_cache for the given hashes."""
    unique_hashes = list({h for h in text_hashes})
    if not unique_hashes:
        return {}
    placeholders = ",".join("?" * len(unique_hashes))
    dim_clause = " AND dim = ?" if expected_dim is not None else ""
    params: tuple[object, ...] = (
        (model, *unique_hashes, expected_dim)
        if expected_dim is not None
        else (model, *unique_hashes)
    )
    rows = conn.execute(
        f"SELECT text_hash, vector_json FROM embedding_cache "
        f"WHERE model = ? AND text_hash IN ({placeholders}){dim_clause}",
        params,
    ).fetchall()
    decoded: dict[str, list[float]] = {}
    for row in rows:
        try:
            vector = decode_vector(row["vector_json"])
        except (AttributeError, UnicodeError, TypeError, ValueError):
            continue
        if (
            isinstance(vector, list)
            and (
                expected_dim is None
                or _finite_embedding_vector(vector, expected_dim=expected_dim)
                is not None
            )
        ):
            decoded[row["text_hash"]] = vector
    return decoded


def _finite_embedding_vector(
    value: object, *, expected_dim: int
) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != expected_dim:
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    return vector if math.isfinite(norm) and norm > 0.0 else None


def _embedding_identity(embedder: EmbeddingClient) -> tuple[str, int]:
    """Read and validate the exact durable vector-space identity."""
    try:
        model = embedder.model
        dim = embedder.dim
    except Exception as exc:
        raise RuntimeError("embedding client identity is unavailable") from exc
    if (
        not isinstance(model, str) or not model
        or isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
    ):
        raise RuntimeError("embedding client has an invalid model/dimension")
    return model, dim


def _post_embed_identity(
    embedder: EmbeddingClient, *, expected_model: str
) -> int:
    """Verify a provider call did not silently switch vector spaces."""
    model, dim = _embedding_identity(embedder)
    if model != expected_model:
        raise RuntimeError("embedding client changed model during batch")
    return dim


def message_embedding_id_batches(
    conn: sqlite3.Connection,
    *,
    message_ids: Sequence[int] | None = None,
    batch_size: int = MESSAGE_EMBEDDING_BATCH_SIZE,
) -> Iterator[tuple[int, ...]]:
    """Yield bounded, deterministic occurrence-id batches.

    Hot ingestion passes its exact newly committed ids.  Dream maintenance
    pages the durable producer-bounded corpus by primary key, so one poisoned
    provider batch can be skipped without preventing later batches from making
    progress and no unbounded ``IN`` list reaches SQLite.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("message embedding batch size must be positive")
    if message_ids is not None:
        ids = tuple(dict.fromkeys(
            value
            for value in message_ids
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ))
        for start in range(0, len(ids), batch_size):
            yield ids[start:start + batch_size]
        return

    after = -1
    while True:
        rows = conn.execute(
            """
            SELECT mc.message_id
            FROM message_retention_coverage mc
            JOIN sessions source_session ON source_session.id = mc.source_session_id
            WHERE mc.coverage_version = ?
              AND mc.source_role IN ('user', 'assistant')
              AND source_session.coverage_message_id IS NOT NULL
              AND typeof(mc.message_id) = 'integer'
              AND mc.message_id <= source_session.coverage_message_id
              AND mc.message_id > ?
            ORDER BY mc.message_id
            LIMIT ?
            """,
            (LOSSLESS_COVERAGE_VERSION, after, batch_size),
        ).fetchall()
        if not rows:
            return
        batch = tuple(int(row["message_id"]) for row in rows)
        yield batch
        after = batch[-1]


def fetch_message_embeddings(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    *,
    message_ids: Sequence[int] | None = None,
) -> PendingMessageEmbeddings | None:
    """Embed missing/stale exact message occurrences outside a write lock.

    Coverage artifacts, rather than mutable/prunable raw rows, are the source
    corpus.  Every candidate is fully proof-validated before its content is
    sent to the embedder.  Existing rows are current only when text hash,
    model, dimension, and vector numerics all match.
    """
    model, initial_dim = _embedding_identity(embedder)

    id_clause = ""
    id_params: tuple[int, ...] = ()
    if message_ids is not None:
        ids = tuple(dict.fromkeys(
            value
            for value in message_ids
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ))[:MESSAGE_EMBEDDING_BATCH_SIZE]
        if not ids:
            return None
        id_clause = f"AND mc.message_id IN ({','.join('?' * len(ids))})"
        id_params = ids
    try:
        rows = conn.execute(
            f"""
            SELECT {COVERAGE_VALIDATION_COLUMNS},
                   me.text_hash AS stored_text_hash,
                   me.vector_json AS stored_vector_json,
                   me.model AS stored_model,
                   me.dim AS stored_dim
            FROM message_retention_coverage mc
            {COVERAGE_VALIDATION_JOINS}
            LEFT JOIN message_embeddings me ON me.message_id = mc.message_id
            WHERE mc.coverage_version = ?
              AND mc.source_role IN ('user', 'assistant')
              AND source_session.coverage_message_id IS NOT NULL
              AND typeof(mc.message_id) = 'integer'
              AND mc.message_id <= source_session.coverage_message_id
              {id_clause}
            ORDER BY mc.message_id
            """,
            (LOSSLESS_COVERAGE_VERSION, *id_params),
        ).fetchall()
    except sqlite3.OperationalError:
        return None

    candidates: list[tuple[object, str]] = []
    seen: set[int] = set()
    for row in rows:
        try:
            proof = validate_message_coverage_row(row)
        except (RuntimeError, TypeError, ValueError):
            continue
        if proof.message_id in seen:
            continue
        seen.add(proof.message_id)
        text_hash = embedding_text_hash(proof.content)
        stored = None
        if (
            row["stored_text_hash"] == text_hash
            and row["stored_model"] == model
            and row["stored_dim"] == initial_dim
        ):
            try:
                stored = decode_vector(row["stored_vector_json"])
            except (AttributeError, UnicodeError, TypeError, ValueError):
                stored = None
        if _finite_embedding_vector(stored, expected_dim=initial_dim) is not None:
            continue
        candidates.append((row, text_hash))
        # Provider requests stay bounded even when this low-level function is
        # called directly. Production callers page exact ids so later rows are
        # not starved by this cap.
        if len(candidates) >= MESSAGE_EMBEDDING_BATCH_SIZE:
            break
    if not candidates:
        return None

    text_hashes = [text_hash for _, text_hash in candidates]
    unique_hashes = list(dict.fromkeys(text_hashes))
    placeholders = ",".join("?" * len(unique_hashes))
    cache_rows = conn.execute(
        f"SELECT text_hash, vector_json, dim FROM embedding_cache "
        f"WHERE model = ? AND text_hash IN ({placeholders})",
        (model, *unique_hashes),
    ).fetchall()
    cached_raw = {row["text_hash"]: row for row in cache_rows}

    vectors: list[list[float] | None] = [None] * len(candidates)
    from_cache = [False] * len(candidates)
    occurrence_indices_by_hash: dict[str, list[int]] = {}
    miss_hashes: list[str] = []
    miss_texts: list[str] = []
    for index, ((row, text_hash)) in enumerate(candidates):
        occurrence_indices_by_hash.setdefault(text_hash, []).append(index)
        cached_row = cached_raw.get(text_hash)
        cached = None
        if cached_row is not None and cached_row["dim"] == initial_dim:
            try:
                decoded = decode_vector(cached_row["vector_json"])
            except (AttributeError, UnicodeError, TypeError, ValueError):
                decoded = None
            cached = _finite_embedding_vector(decoded, expected_dim=initial_dim)
        if cached is not None:
            vectors[index] = cached
            from_cache[index] = True
        elif text_hash not in miss_hashes:
            miss_hashes.append(text_hash)
            miss_texts.append(validate_message_coverage_row(row).content)

    if miss_texts:
        fresh = embedder.embed(miss_texts)
        if len(fresh) != len(miss_texts):
            raise RuntimeError(
                "embedding client returned the wrong number of message vectors"
            )
        final_dim = _post_embed_identity(embedder, expected_model=model)
        # A first OpenAI-compatible response may correct its declared dim. Any
        # cache hits admitted under the old declaration belong to a different
        # vector space and must be recomputed under the now-authoritative dim.
        if final_dim != initial_dim and any(from_cache):
            redo_hashes = list(dict.fromkeys(
                text_hashes[i] for i, cached in enumerate(from_cache) if cached
            ))
            first_index_by_hash = {
                text_hash: indices[0]
                for text_hash, indices in occurrence_indices_by_hash.items()
            }
            redo_texts = [
                validate_message_coverage_row(
                    candidates[first_index_by_hash[text_hash]][0]
                ).content
                for text_hash in redo_hashes
            ]
            redo = embedder.embed(redo_texts)
            if len(redo) != len(redo_texts):
                raise RuntimeError(
                    "embedding client returned the wrong number of message vectors"
                )
            redo_dim = _post_embed_identity(embedder, expected_model=model)
            if redo_dim != final_dim:
                raise RuntimeError(
                    "embedding client changed dimension during message retry"
                )
            for text_hash, candidate in zip(redo_hashes, redo):
                for index in occurrence_indices_by_hash[text_hash]:
                    vectors[index] = candidate
                    from_cache[index] = False
        for text_hash, candidate in zip(miss_hashes, fresh):
            for index in occurrence_indices_by_hash[text_hash]:
                vectors[index] = candidate
    else:
        final_dim = initial_dim

    validated = [
        _finite_embedding_vector(vector, expected_dim=final_dim)
        for vector in vectors
    ]
    if any(vector is None for vector in validated):
        raise RuntimeError("embedding client returned malformed message vectors")

    proofs = [validate_message_coverage_row(row) for row, _ in candidates]
    return PendingMessageEmbeddings(
        message_ids=[proof.message_id for proof in proofs],
        session_ids=[proof.session_id for proof in proofs],
        chunk_ids=[proof.chunk_id for proof in proofs],
        coverage_versions=[row["coverage_version"] for row, _ in candidates],
        content_hashes=[row["message_content_hash"] for row, _ in candidates],
        text_hashes=text_hashes,
        vectors=cast(list[list[float]], validated),
        from_cache=from_cache,
        dim=final_dim,
        model=model,
        cache_hits=sum(from_cache),
    )


def persist_message_embeddings(
    conn: sqlite3.Connection, pending: PendingMessageEmbeddings
) -> int:
    """Persist a prepared message batch idempotently in a short transaction."""
    core_db.ensure_vec_table(conn, pending.dim, model=pending.model)
    has_vec = core_db.has_vec_table(conn, table="vec_messages")
    persisted = 0
    for (
        message_id, session_id, chunk_id, coverage_version, content_hash,
        text_hash, candidate, is_cached,
    ) in zip(
        pending.message_ids, pending.session_ids, pending.chunk_ids,
        pending.coverage_versions, pending.content_hashes, pending.text_hashes,
        pending.vectors, pending.from_cache,
    ):
        vector = _finite_embedding_vector(candidate, expected_dim=pending.dim)
        if vector is None:
            continue
        parent = conn.execute(
            "SELECT 1 FROM message_retention_coverage "
            "WHERE message_id=? AND source_session_id=? AND chunk_id=? "
            "AND coverage_version=? AND message_content_hash=?",
            (message_id, session_id, chunk_id, coverage_version, content_hash),
        ).fetchone()
        if parent is None:
            continue
        if not is_cached:
            conn.execute(
                """
                INSERT INTO embedding_cache(text_hash, model, vector_json, dim)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(text_hash, model) DO UPDATE SET
                    vector_json=excluded.vector_json,
                    dim=excluded.dim,
                    created_at=CURRENT_TIMESTAMP
                """,
                (text_hash, pending.model, encode_vector(vector), pending.dim),
            )
        conn.execute(
            """
            INSERT INTO message_embeddings(
                message_id, source_coverage_chunk_id,
                source_coverage_version, text_hash, vector_json, model, dim
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                source_coverage_chunk_id=excluded.source_coverage_chunk_id,
                source_coverage_version=excluded.source_coverage_version,
                text_hash=excluded.text_hash,
                vector_json=excluded.vector_json,
                model=excluded.model,
                dim=excluded.dim,
                created_at=CURRENT_TIMESTAMP
            """,
            (
                message_id, chunk_id, coverage_version, text_hash,
                encode_vector(vector), pending.model, pending.dim,
            ),
        )
        if has_vec:
            conn.execute("DELETE FROM vec_messages WHERE rowid=?", (message_id,))
            conn.execute(
                "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
                (message_id, core_db._pack_vector(vector)),
            )
        persisted += 1
    return persisted


def persist_chunk_embeddings(
    conn: sqlite3.Connection, pending: PendingChunkEmbeddings
) -> int:
    """Insert pending chunk vectors into chunk_embeddings + vec_chunks.
    Cache misses are also written to embedding_cache.
    Caller wraps in core_db.transaction()."""
    core_db.ensure_vec_table(conn, pending.dim, model=pending.model)
    has_vec = core_db.has_vec_table(conn, table="vec_chunks")
    persisted = 0
    for chunk_id, chunk_rowid, vec, text_hash, is_cached in zip(
        pending.ids,
        pending.chunk_rowids,
        pending.vectors,
        pending.text_hashes,
        pending.from_cache,
    ):
        vec = _finite_embedding_vector(vec, expected_dim=pending.dim)
        if vec is None:
            continue
        parent = conn.execute(
            "SELECT rowid, text, chunk_kind FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if (
            parent is None
            or parent["chunk_kind"] != "extraction"
            or int(parent["rowid"]) != int(chunk_rowid)
            or embedding_text_hash(parent["text"]) != text_hash
        ):
            continue
        if not is_cached:
            conn.execute(
                """
                INSERT OR IGNORE INTO embedding_cache(text_hash, model, vector_json, dim)
                VALUES (?, ?, ?, ?)
                """,
                (text_hash, pending.model, encode_vector(vec), len(vec)),
            )
        conn.execute(
            """
            INSERT OR REPLACE INTO chunk_embeddings(
                chunk_id, vector_json, model, dim, text_hash
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (chunk_id, encode_vector(vec), pending.model, len(vec), text_hash),
        )
        if has_vec:
            # vec0 doesn't support INSERT OR REPLACE on the rowid PK (it raises
            # UNIQUE constraint failed), so delete any stale row first — same
            # guard as persist_episode_embeddings.
            conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (chunk_rowid,))
            conn.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                (chunk_rowid, core_db._pack_vector(vec)),
            )
        persisted += 1
    return persisted


def prepare_chunk_embed_batch(
    conn: sqlite3.Connection,
    chunk_id_text_pairs: list[tuple[str, str]],
    embedder: EmbeddingClient,
) -> ChunkEmbedRequest:
    """Capture all main-thread state for a chunk batch: cache lookup against
    ``embedding_cache`` and the indices of cache misses that still need an
    embedder call. The returned request can be passed to ``assemble_chunk_pending``
    once the embedder has returned vectors for ``miss_texts``."""
    ids = [cid for cid, _ in chunk_id_text_pairs]
    texts = [t for _, t in chunk_id_text_pairs]
    model, dim = _embedding_identity(embedder)
    text_hashes = [embedding_text_hash(t) for t in texts]
    cached_by_hash = _fetch_cached_vectors(
        conn, text_hashes, model, expected_dim=dim
    )
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    for i, h in enumerate(text_hashes):
        if cached_by_hash.get(h) is None:
            miss_indices.append(i)
            miss_texts.append(texts[i])
    return ChunkEmbedRequest(
        ids=ids,
        texts=texts,
        text_hashes=text_hashes,
        cached_by_hash=cached_by_hash,
        miss_indices=miss_indices,
        miss_texts=miss_texts,
        model=model,
        dim=dim,
    )


def assemble_chunk_pending(
    conn: sqlite3.Connection,
    request: ChunkEmbedRequest,
    miss_vectors: list[list[float]],
    *,
    exclude_ids: set[str] | None = None,
    resolved_model: str | None = None,
    resolved_dim: int | None = None,
) -> PendingChunkEmbeddings | None:
    """Merge cache hits + freshly embedded vectors into a PendingChunkEmbeddings.

    Queries chunk rowids fresh so a chunk pruned between submit and persist
    is dropped silently. Returns ``None`` if nothing remains to persist
    (everything filtered out by ``exclude_ids`` or chunk_id no longer present).
    """
    if len(miss_vectors) != len(request.miss_texts):
        raise RuntimeError(
            f"embedding client returned {len(miss_vectors)} vectors for "
            f"{len(request.miss_texts)} chunks"
        )
    final_model = request.model if resolved_model is None else resolved_model
    final_dim = request.dim if resolved_dim is None else resolved_dim
    if final_model != request.model:
        raise RuntimeError("embedding client changed model during chunk batch")
    if (
        isinstance(final_dim, bool)
        or not isinstance(final_dim, int)
        or final_dim <= 0
    ):
        raise RuntimeError("embedding client returned an invalid chunk dimension")
    validated_misses = [
        _finite_embedding_vector(vector, expected_dim=final_dim)
        for vector in miss_vectors
    ]
    if any(vector is None for vector in validated_misses):
        raise RuntimeError("embedding client returned malformed chunk vectors")
    n = len(request.ids)
    vectors: list[list[float] | None] = [None] * n
    from_cache = [False] * n
    for i in range(n):
        cached = request.cached_by_hash.get(request.text_hashes[i])
        validated_cached = _finite_embedding_vector(
            cached, expected_dim=final_dim
        )
        if validated_cached is not None:
            vectors[i] = validated_cached
            from_cache[i] = True
    for idx, vec in zip(request.miss_indices, validated_misses):
        vectors[idx] = vec

    skip = exclude_ids or set()
    placeholders = ",".join("?" * len(request.ids))
    rowid_rows = conn.execute(
        f"SELECT id, rowid FROM chunks WHERE chunk_kind = 'extraction' "
        f"AND id IN ({placeholders})",
        tuple(request.ids),
    ).fetchall()
    rowid_map = {r["id"]: r["rowid"] for r in rowid_rows}
    kept = [
        i for i, cid in enumerate(request.ids)
        if cid in rowid_map and cid not in skip and vectors[i] is not None
    ]
    if not kept:
        return None

    return PendingChunkEmbeddings(
        ids=[request.ids[i] for i in kept],
        chunk_rowids=[rowid_map[request.ids[i]] for i in kept],
        vectors=[vectors[i] for i in kept],  # type: ignore[misc]
        dim=final_dim,
        model=final_model,
        text_hashes=[request.text_hashes[i] for i in kept],
        from_cache=[from_cache[i] for i in kept],
        cache_hits=sum(from_cache[i] for i in kept),
    )


def fetch_edge_embeddings(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> PendingEdgeEmbeddings | None:
    """Read live direct edges, determine which triple texts are uncached, and embed
    only those. No write transaction held.

    Returns None when there are no live direct edges. The cache
    (edge_embeddings) is keyed on triple text, not edge id.
    """
    rows = conn.execute(
        f"""
        SELECT id, subject_canonical, predicate, object_canonical
        FROM knowledge_graph
        WHERE {live_edge_predicate()}
        ORDER BY id
        """
    ).fetchall()
    if not rows:
        return None

    edge_text_by_id = {
        r["id"]: f"{r['subject_canonical']} {r['predicate']} {r['object_canonical']}"
        for r in rows
    }

    model, dim = _embedding_identity(embedder)
    pending_texts = sorted(
        text
        for text in set(edge_text_by_id.values())
        if not _edge_embedding_is_current(conn, text, model=model, dim=dim)
    )

    new_text_vectors: dict[str, list[float]] = {}
    truly_new_texts: set[str] = set()
    cache_hits = 0
    if pending_texts:
        text_hashes = {
            text: embedding_text_hash(text) for text in pending_texts
        }
        cached_by_hash = _fetch_cached_vectors(
            conn, list(text_hashes.values()), model, expected_dim=dim
        )

        miss_texts: list[str] = []
        for text in pending_texts:
            cached = cached_by_hash.get(text_hashes[text])
            valid_cached = _finite_embedding_vector(
                cached, expected_dim=dim
            )
            if valid_cached is not None:
                new_text_vectors[text] = valid_cached
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
            final_dim = _post_embed_identity(embedder, expected_model=model)
            if final_dim != dim and cache_hits:
                raise RuntimeError(
                    "embedding dimension changed with cached edge vectors"
                )
            validated_vectors = [
                _finite_embedding_vector(vector, expected_dim=final_dim)
                for vector in vectors
            ]
            if any(vector is None for vector in validated_vectors):
                raise RuntimeError(
                    "embedding client returned malformed edge vectors"
                )
            new_text_vectors.update(dict(zip(
                miss_texts,
                cast(list[list[float]], validated_vectors),
            )))
            truly_new_texts = set(miss_texts)
            dim = final_dim

    return PendingEdgeEmbeddings(
        edge_text_by_id=edge_text_by_id,
        new_text_vectors=new_text_vectors,
        truly_new_texts=truly_new_texts,
        dim=dim,
        model=model,
        cache_hits=cache_hits,
    )


def persist_edge_embeddings(
    conn: sqlite3.Connection, pending: PendingEdgeEmbeddings
) -> int:
    """Persist newly embedded triple texts and rebuild vec_edges from cache.
    True embedding misses are also written to embedding_cache.
    Caller wraps in core_db.transaction()."""
    persisted = 0
    for text, candidate in pending.new_text_vectors.items():
        vec = _finite_embedding_vector(candidate, expected_dim=pending.dim)
        if vec is None:
            continue
        if text in pending.truly_new_texts:
            text_hash = embedding_text_hash(text)
            conn.execute(
                """
                INSERT OR IGNORE INTO embedding_cache(text_hash, model, vector_json, dim)
                VALUES (?, ?, ?, ?)
                """,
                (text_hash, pending.model, encode_vector(vec), len(vec)),
            )
        conn.execute(
            """
            INSERT OR REPLACE INTO edge_embeddings(edge_text, vector_json, model, dim)
            VALUES (?, ?, ?, ?)
            """,
            (text, encode_vector(vec), pending.model, len(vec)),
        )
        persisted += 1

    core_db.ensure_vec_table(conn, pending.dim, model=pending.model)

    # Rebuild vec_edges from scratch: derived edge ids churn every dream run, so
    # a full clear + reinsert is simpler and cheaper than reconciling rowids
    # (vec0 virtual tables don't support INSERT OR REPLACE on the primary key).
    if core_db.has_vec_table(conn, table="vec_edges"):
        conn.execute("DELETE FROM vec_edges")
        for edge_id, text in pending.edge_text_by_id.items():
            emb = conn.execute(
                "SELECT vector_json FROM edge_embeddings "
                "WHERE edge_text = ? AND model = ? AND dim = ?",
                (text, pending.model, pending.dim),
            ).fetchone()
            if emb is None:
                continue
            try:
                decoded = decode_vector(emb["vector_json"])
            except (AttributeError, UnicodeError, TypeError, ValueError):
                continue
            vec = _finite_embedding_vector(decoded, expected_dim=pending.dim)
            if vec is None:
                continue
            conn.execute(
                "INSERT INTO vec_edges(rowid, embedding) VALUES (?, ?)",
                (edge_id, core_db._pack_vector(vec)),
            )
    return persisted


def _edge_embedding_is_current(
    conn: sqlite3.Connection,
    edge_text: str,
    *,
    model: str,
    dim: int,
) -> bool:
    row = conn.execute(
        "SELECT vector_json FROM edge_embeddings "
        "WHERE edge_text=? AND model=? AND dim=?",
        (edge_text, model, dim),
    ).fetchone()
    if row is None:
        return False
    try:
        decoded = decode_vector(row["vector_json"])
    except (AttributeError, UnicodeError, TypeError, ValueError):
        return False
    return _finite_embedding_vector(decoded, expected_dim=dim) is not None


def _episode_embed_text(title: str, summary: str) -> str:
    return f"{title}\n{summary}"


def fetch_episode_embeddings(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> PendingEpisodeEmbeddings | None:
    """Build a batch of episode embeddings for any episode whose stored vector
    is missing or stale (text_hash mismatch). Cache-hit-aware via embedding_cache.

    Returns None when no episodes need (re-)embedding.
    """
    model, initial_dim = _embedding_identity(embedder)
    rows = conn.execute(
        """
        SELECT e.id, e.rowid AS rowid, e.title, e.summary,
               ee.text_hash AS stored_hash,
               ee.vector_json AS stored_vector,
               ee.model AS stored_model, ee.dim AS stored_dim
        FROM episodes e
        JOIN sessions s ON s.id = e.session_id
        LEFT JOIN episode_embeddings ee ON ee.episode_id = e.id
        WHERE e.digest_generation IS NULL
           OR e.digest_generation = s.digest_published_generation
        """
    ).fetchall()
    if not rows:
        return None

    pending_ids: list[str] = []
    pending_rowids: list[int] = []
    pending_hashes: list[str] = []
    pending_texts: list[str] = []
    for r in rows:
        text = _episode_embed_text(r["title"], r["summary"])
        text_hash = embedding_text_hash(text)
        if (
            r["stored_hash"] == text_hash
            and r["stored_model"] == model
            and r["stored_dim"] == initial_dim
        ):
            try:
                stored = decode_vector(r["stored_vector"])
            except (AttributeError, UnicodeError, TypeError, ValueError):
                stored = None
            if _finite_embedding_vector(stored, expected_dim=initial_dim) is not None:
                continue
        pending_ids.append(r["id"])
        pending_rowids.append(int(r["rowid"]))
        pending_hashes.append(text_hash)
        pending_texts.append(text)

    if not pending_ids:
        return None

    cached_by_hash = _fetch_cached_vectors(
        conn, pending_hashes, model, expected_dim=initial_dim
    )

    vectors_out: list[list[float] | None] = [None] * len(pending_ids)
    from_cache = [False] * len(pending_ids)
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    for i, h in enumerate(pending_hashes):
        cached = cached_by_hash.get(h)
        if cached is not None:
            vectors_out[i] = cached
            from_cache[i] = True
        else:
            miss_indices.append(i)
            miss_texts.append(pending_texts[i])

    if miss_texts:
        embedded = embedder.embed(miss_texts)
        if len(embedded) != len(miss_texts):
            raise RuntimeError(
                f"embedding client returned {len(embedded)} vectors for "
                f"{len(miss_texts)} episodes"
            )
        for idx, vec in zip(miss_indices, embedded):
            vectors_out[idx] = vec
        final_dim = _post_embed_identity(embedder, expected_model=model)
    else:
        final_dim = initial_dim
    if final_dim != initial_dim and any(from_cache):
        raise RuntimeError("embedding dimension changed with cached episode vectors")
    validated = [
        _finite_embedding_vector(vector, expected_dim=final_dim)
        for vector in vectors_out
    ]
    if any(vector is None for vector in validated):
        raise RuntimeError("embedding client returned malformed episode vectors")
    return PendingEpisodeEmbeddings(
        ids=pending_ids,
        rowids=pending_rowids,
        vectors=cast(list[list[float]], validated),
        text_hashes=pending_hashes,
        from_cache=from_cache,
        dim=final_dim,
        model=model,
        cache_hits=sum(from_cache),
    )


def fetch_fact_embeddings(
    conn: sqlite3.Connection, embedder: EmbeddingClient, *,
    batch_size: int = FACT_EMBEDDING_BATCH_SIZE,
) -> PendingFactEmbeddings | None:
    """Embed one bounded batch of current proof-valid facts without a vector.

    Uses the shared cache when possible and runs outside the write lock.
    Authority is fully proven before text reaches either a quality hook or the
    provider.  Candidate rows are keyset-paged, and proof caches are page-local,
    so a corrupt/cached prefix cannot starve later work or make one dream retain
    the whole fact corpus in Python.  Rows from one extraction unit in a page
    share one proof validation.
    Runs OUTSIDE the write lock (the phase1 lock-free pattern). Returns None
    when nothing is pending — including on a pre-v46 store, where the table
    does not exist."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("fact embedding batch_size must be a positive integer")
    model, initial_dim = _embedding_identity(embedder)
    page_size = max(32, batch_size * 2)
    scan_limit = max(FACT_EMBEDDING_SCAN_SIZE, batch_size)
    scan_state_key = (
        "fact_embedding_scan:"
        + embedding_text_hash(f"{model}\0{initial_dim}").rsplit(":", 1)[-1]
    )
    state = conn.execute(
        "SELECT value FROM schema_meta WHERE key=?", (scan_state_key,),
    ).fetchone()
    scan_start_value = str(state["value"]) if state is not None else None
    state_match = (
        re.fullmatch(r"v1:([0-9]{1,18}):([0-9]{1,18})", scan_start_value)
        if scan_start_value is not None else None
    )
    if state_match is None:
        scan_generation = 0
        last_id = 0
    else:
        scan_generation = int(state_match.group(1))
        last_id = int(state_match.group(2))
    pending_rows: list[sqlite3.Row] = []
    outcome_cache: OrderedDict = OrderedDict()
    chain_cache: OrderedDict = OrderedDict()
    scanned = 0
    saw_rows = False
    wrapped = False
    scan_after_id = last_id
    while len(pending_rows) < batch_size and scanned < scan_limit:
        take = min(page_size, scan_limit - scanned)
        try:
            rows = conn.execute(
                """
                SELECT f.id, f.session_id, f.text, f.source_outcome_key,
                       e.text_hash AS stored_hash,
                       e.vector_json AS stored_vector,
                       e.model AS stored_model, e.dim AS stored_dim
                FROM narrative_facts f
                JOIN fact_extraction_outcomes o
                  ON o.slice_key=f.source_outcome_key
                LEFT JOIN narrative_fact_embeddings e ON e.fact_id = f.id
                WHERE f.id > ?
                  AND f.source_outcome_key IS NOT NULL
                  AND f.lifecycle_status='active'
                  AND f.invalid_at IS NULL
                  AND o.outcome_status='success'
                  AND o.source_manifest_complete=1
                  AND o.source_manifest_version='fact-source-manifest-v1'
                  AND o.source_manifest_count > 0
                ORDER BY f.id
                LIMIT ?
                """,
                (scan_after_id, take),
            ).fetchall()
        except sqlite3.OperationalError:
            return None
        if not rows:
            if scan_after_id > 0 and not wrapped:
                # A bounded repair sweep cycles over older identities after it
                # reaches the current high-water. New suffix facts are always
                # visited first; quiescent work remains capped per dream.
                scan_after_id = 0
                wrapped = True
                continue
            scan_after_id = 0
            break
        saw_rows = True
        proofs = load_fact_source_manifests(
            conn,
            [int(row["id"]) for row in rows],
            outcome_cache=outcome_cache,
            chain_cache=chain_cache,
        )
        for row in rows:
            scanned += 1
            scan_after_id = int(row["id"])
            if int(row["id"]) not in proofs:
                continue
            current_hash = embedding_text_hash(row["text"])
            stored = None
            if (
                row["stored_hash"] == current_hash
                and row["stored_model"] == model
                and row["stored_dim"] == initial_dim
            ):
                try:
                    decoded = decode_vector(row["stored_vector"])
                except (AttributeError, UnicodeError, TypeError, ValueError):
                    decoded = None
                stored = _finite_embedding_vector(decoded, expected_dim=initial_dim)
            if stored is None:
                pending_rows.append(row)
                if len(pending_rows) >= batch_size:
                    break
        for row in rows:
            key = str(row["source_outcome_key"])
            if key in outcome_cache:
                outcome_cache.move_to_end(key)
            session_key = str(row["session_id"])
            if session_key in chain_cache:
                chain_cache.move_to_end(session_key)
        while len(outcome_cache) > 64:
            outcome_cache.popitem(last=False)
        while len(chain_cache) > 256:
            chain_cache.popitem(last=False)
        if len(rows) < take and len(pending_rows) < batch_size:
            scan_after_id = 0
            break
    if not pending_rows and not saw_rows:
        return None
    fact_ids = [int(r["id"]) for r in pending_rows]
    texts = [r["text"] for r in pending_rows]
    text_hashes = [embedding_text_hash(t) for t in texts]
    cached_by_hash = _fetch_cached_vectors(
        conn, text_hashes, model, expected_dim=initial_dim
    )

    vectors_out: list[list[float] | None] = [None] * len(texts)
    from_cache = [False] * len(texts)
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    for i, h in enumerate(text_hashes):
        cached = cached_by_hash.get(h)
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
                f"embedding client returned {len(embedded)} vectors for "
                f"{len(miss_texts)} facts"
            )
        for idx, vec in zip(miss_indices, embedded):
            vectors_out[idx] = vec
        final_dim = _post_embed_identity(embedder, expected_model=model)
    else:
        final_dim = initial_dim
    if final_dim != initial_dim and any(from_cache):
        raise RuntimeError("embedding dimension changed with cached fact vectors")
    validated = [
        _finite_embedding_vector(vector, expected_dim=final_dim)
        for vector in vectors_out
    ]
    if any(vector is None for vector in validated):
        raise RuntimeError("embedding client returned malformed fact vectors")
    return PendingFactEmbeddings(
        fact_ids=fact_ids,
        vectors=cast(list[list[float]], validated),
        text_hashes=text_hashes,
        from_cache=from_cache,
        dim=final_dim,
        model=model,
        cache_hits=sum(from_cache),
        scan_state_key=scan_state_key,
        scan_after_id=scan_after_id,
        scan_start_value=scan_start_value,
        scan_next_value=f"v1:{scan_generation + 1}:{scan_after_id}",
    )


def persist_fact_embeddings(
    conn: sqlite3.Connection, pending: PendingFactEmbeddings
) -> int:
    """Insert fact vectors into narrative_fact_embeddings + vec_facts (rowid =
    narrative_facts.id, an INTEGER PRIMARY KEY, so VACUUM-stable like
    vec_edges). Cache misses also land in embedding_cache. Caller wraps in
    core_db.transaction()."""
    core_db.ensure_vec_table(conn, pending.dim, model=pending.model)
    has_vec = core_db.has_vec_table(conn, table="vec_facts")
    persisted = 0
    for fact_id, vec, text_hash, is_cached in zip(
        pending.fact_ids,
        pending.vectors,
        pending.text_hashes,
        pending.from_cache,
    ):
        vec = _finite_embedding_vector(vec, expected_dim=pending.dim)
        if vec is None:
            continue
        if not is_cached:
            conn.execute(
                """
                INSERT OR IGNORE INTO embedding_cache(text_hash, model, vector_json, dim)
                VALUES (?, ?, ?, ?)
                """,
                (text_hash, pending.model, encode_vector(vec), len(vec)),
            )
        conn.execute(
            """
            INSERT OR REPLACE INTO narrative_fact_embeddings(
                fact_id, vector_json, model, dim, text_hash
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (fact_id, encode_vector(vec), pending.model, len(vec), text_hash),
        )
        if has_vec:
            # vec0 doesn't support INSERT OR REPLACE on the rowid PK, so delete
            # any stale row first — same guard as the other vec_* persists.
            conn.execute("DELETE FROM vec_facts WHERE rowid = ?", (fact_id,))
            conn.execute(
                "INSERT INTO vec_facts(rowid, embedding) VALUES (?, ?)",
                (fact_id, core_db._pack_vector(vec)),
            )
        persisted += 1
    if pending.scan_state_key is not None and pending.scan_next_value is not None:
        current = conn.execute(
            "SELECT value FROM schema_meta WHERE key=?",
            (pending.scan_state_key,),
        ).fetchone()
        current_value = str(current["value"]) if current is not None else None
        if current_value == pending.scan_start_value:
            conn.execute(
                "INSERT INTO schema_meta(key,value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (pending.scan_state_key, pending.scan_next_value),
            )
    return persisted


def persist_episode_embeddings(
    conn: sqlite3.Connection, pending: PendingEpisodeEmbeddings
) -> int:
    """UPSERT episode vectors into episode_embeddings + vec_episodes. Cache
    misses also land in embedding_cache. Caller wraps in core_db.transaction()."""
    core_db.ensure_vec_table(conn, pending.dim, model=pending.model)
    has_vec = core_db.has_vec_table(conn, table="vec_episodes")
    persisted = 0
    for ep_id, rowid, vec, text_hash, is_cached in zip(
        pending.ids,
        pending.rowids,
        pending.vectors,
        pending.text_hashes,
        pending.from_cache,
    ):
        vec = _finite_embedding_vector(vec, expected_dim=pending.dim)
        if vec is None:
            continue
        if not is_cached:
            conn.execute(
                """
                INSERT OR IGNORE INTO embedding_cache(text_hash, model, vector_json, dim)
                VALUES (?, ?, ?, ?)
                """,
                (text_hash, pending.model, encode_vector(vec), len(vec)),
            )
        conn.execute(
            """
            INSERT INTO episode_embeddings(episode_id, vector_json, model, dim, text_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(episode_id) DO UPDATE SET
                vector_json = excluded.vector_json,
                model = excluded.model,
                dim = excluded.dim,
                text_hash = excluded.text_hash
            """,
            (ep_id, encode_vector(vec), pending.model, len(vec), text_hash),
        )
        if has_vec:
            # vec0 doesn't support INSERT OR REPLACE on the rowid PK, so delete
            # any stale row first.
            conn.execute("DELETE FROM vec_episodes WHERE rowid = ?", (rowid,))
            conn.execute(
                "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
                (rowid, core_db._pack_vector(vec)),
            )
        persisted += 1
    return persisted
