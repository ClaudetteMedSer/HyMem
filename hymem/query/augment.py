from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field, replace

from hymem.config import HyMemConfig
from hymem.extraction.embeddings import EmbeddingClient
from hymem.query.entities import match_known_entities


@dataclass
class GraphFact:
    subject: str
    predicate: str
    object: str
    confidence: float
    pos_evidence: int
    neg_evidence: int


@dataclass
class FtsHit:
    chunk_id: str
    session_id: str
    text: str
    score: float
    score_kind: str = "bm25"


@dataclass
class AugmentedContext:
    """Structured context for the host (Hermes) to assemble into its prompt.

    Hermes decides ordering, headers, and token budget — HyMem only returns
    the pieces. This keeps prompt assembly out of the memory module.

    `fts_hits[i].score` carries different units depending on `score_kind`:
        - "bm25": SQLite FTS5 BM25 score (lower = better, often negative)
        - "rrf":  reciprocal rank fusion score from FTS+vector merge (higher = better)
    """

    user_md: str = ""
    memory_md: str = ""
    fts_hits: list[FtsHit] = field(default_factory=list)
    graph_facts: list[GraphFact] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)


def augment(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    user_message: str,
    *,
    embedding_client: EmbeddingClient | None = None,
) -> AugmentedContext:
    ctx = AugmentedContext()
    if cfg.user_md_path.exists():
        ctx.user_md = cfg.user_md_path.read_text(encoding="utf-8")
    if cfg.memory_md_path.exists():
        ctx.memory_md = cfg.memory_md_path.read_text(encoding="utf-8")

    fts = _fts_search(conn, user_message, top_k=cfg.fts_top_k)
    if embedding_client is not None:
        vec = _vector_search(
            conn,
            embedding_client,
            user_message,
            top_k=cfg.fts_top_k,
            max_scan=cfg.embedding_max_scan,
        )
        ctx.fts_hits = _rrf_merge(fts, vec, top_k=cfg.fts_top_k)
    else:
        ctx.fts_hits = fts

    ctx.matched_entities = match_known_entities(conn, user_message)
    if ctx.matched_entities:
        ctx.graph_facts = _graph_lookup(
            conn, ctx.matched_entities, top_k_per_entity=cfg.graph_top_k_per_entity
        )
    return ctx


_FTS_SAFE = re.compile(r"[^A-Za-z0-9_\- ]+")


def _fts_search(conn: sqlite3.Connection, query: str, *, top_k: int) -> list[FtsHit]:
    cleaned = _FTS_SAFE.sub(" ", query).strip()
    if not cleaned:
        return []
    # Build an OR query across tokens so partial matches still surface results.
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return []
    fts_query = " OR ".join(tokens)

    try:
        # bm25() is an FTS5 built-in; schema.sql declares chunks_fts with fts5.
        # If the table is ever migrated away from FTS5, this query will raise
        # OperationalError and fall through to the empty-results path below.
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, c.session_id, c.text, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        FtsHit(
            chunk_id=r["chunk_id"],
            session_id=r["session_id"],
            text=r["text"],
            score=float(r["score"]),
        )
        for r in rows
    ]


def _vector_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
    max_scan: int,
) -> list[FtsHit]:
    # TODO: O(n) Python cosine over all rows is fine up to ~5K chunks. Beyond that,
    # consider: (1) sqlite-vec extension for native ANN, (2) numpy batched dot product,
    # (3) periodic pruning of old embeddings.
    rows = conn.execute(
        """
        SELECT c.id AS chunk_id, c.session_id, c.text, e.vector_json
        FROM chunk_embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        ORDER BY c.created_at DESC
        LIMIT ?
        """,
        (max_scan,),
    ).fetchall()
    if not rows:
        return []

    qvec = embedder.embed([query])[0]
    qnorm = math.sqrt(sum(x * x for x in qvec)) or 1.0

    scored: list[tuple[float, sqlite3.Row]] = []
    for r in rows:
        vec = json.loads(r["vector_json"])
        if len(vec) != len(qvec):
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
        sim = dot / (qnorm * vnorm)
        scored.append((sim, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        FtsHit(
            chunk_id=r["chunk_id"],
            session_id=r["session_id"],
            text=r["text"],
            score=float(sim),
        )
        for sim, r in scored[:top_k]
    ]


def _rrf_merge(
    fts: list[FtsHit], vec: list[FtsHit], *, top_k: int, k: int = 60
) -> list[FtsHit]:
    # Reciprocal rank fusion: score = sum(1 / (k + rank)) across each list.
    by_id: dict[str, FtsHit] = {}
    scores: dict[str, float] = {}
    for rank, hit in enumerate(fts, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.chunk_id, hit)
    for rank, hit in enumerate(vec, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.chunk_id, hit)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        replace(by_id[cid], score=score, score_kind="rrf")
        for cid, score in ordered[:top_k]
    ]


def _graph_lookup(
    conn: sqlite3.Connection, entities: list[str], *, top_k_per_entity: int
) -> list[GraphFact]:
    facts: dict[tuple[str, str, str], GraphFact] = {}
    for entity in entities:
        rows = conn.execute(
            """
            SELECT subject_canonical AS s, predicate AS p, object_canonical AS o,
                   pos_evidence AS pos, neg_evidence AS neg,
                   (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) AS conf
            FROM knowledge_graph
            WHERE status = 'active'
              AND (subject_canonical = ? OR object_canonical = ?)
            ORDER BY conf DESC, last_reinforced DESC
            LIMIT ?
            """,
            (entity, entity, top_k_per_entity),
        ).fetchall()
        for r in rows:
            key = (r["s"], r["p"], r["o"])
            if key in facts:
                continue
            facts[key] = GraphFact(
                subject=r["s"],
                predicate=r["p"],
                object=r["o"],
                confidence=float(r["conf"]),
                pos_evidence=int(r["pos"]),
                neg_evidence=int(r["neg"]),
            )
    return list(facts.values())
