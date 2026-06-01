from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass, field, replace

from hymem.config import HyMemConfig
from hymem.core.vectors import decode_vector
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient
from hymem.query.entities import match_known_entities
from hymem.query.predicate_routing import route_predicates
from hymem.query.rerank import rerank as run_rerank
from hymem.session import Message, recent_messages

log = logging.getLogger("hymem.query.augment")


@dataclass
class GraphFact:
    subject: str
    predicate: str
    object: str
    confidence: float
    pos_evidence: int
    neg_evidence: int
    derived: bool = False
    why_retrieved: list[str] = field(default_factory=list)
    score: float = 0.0
    hedge_recommended: bool = False
    """Set by `_graph_lookup` from `cfg.hedge_confidence_threshold` and
    `cfg.hedge_min_evidence`. Hermes (or any consumer) reads this to decide
    whether to soften phrasing — HyMem only signals, never rewrites the
    fact text."""


@dataclass
class EpisodeHit:
    episode_id: str
    session_id: str
    title: str
    summary: str
    score: float
    score_kind: str = "bm25"
    """Source of the score: "bm25" (FTS only), "vec" (semantic only), or
    "rrf" (reciprocal-rank-fused FTS + vec)."""
    why_retrieved: list[str] = field(default_factory=list)
    """Short reason chips (e.g. `episode_fts("postgres pool")`,
    `episode_rrf(fts+vec, 0.0240)`) mirroring `GraphFact.why_retrieved`, so a
    consumer can quote why an episode surfaced instead of guessing."""


@dataclass
class FtsHit:
    chunk_id: str
    session_id: str
    text: str
    score: float
    score_kind: str = "bm25"
    why_retrieved: list[str] = field(default_factory=list)
    """Short reason chips (e.g. `fts_match("postgres pool")`,
    `vec_topk(sim=0.82)`, `rrf(fts+vec, 0.0240)`, `reranked`)."""


@dataclass
class ProcedureHit:
    procedure_id: str
    session_id: str
    name: str
    description: str
    steps: list[dict]
    score: float
    why_retrieved: list[str] = field(default_factory=list)
    """Short reason chips (e.g. `procedure_fts("deploy staging")`)."""


@dataclass
class AugmentedContext:
    """Structured context for the host (Hermes) to assemble into its prompt.

    Hermes decides ordering, headers, and token budget — HyMem only returns
    the pieces. This keeps prompt assembly out of the memory module.

    `fts_hits[i].score` carries different units depending on `score_kind`:
        - "bm25": SQLite FTS5 BM25 score (lower = better, often negative)
        - "rrf":  reciprocal rank fusion score from FTS+vector merge (higher = better)

    `recent_turns` is the working-memory tier: the last N raw turns of the
    active session, included so the host can surface within-session facts that
    have not yet been consolidated by dreaming. It is populated only when a
    `session_id` is passed to `augment()`; otherwise it stays empty.
    """

    user_md: str = ""
    memory_md: str = ""
    fts_hits: list[FtsHit] = field(default_factory=list)
    graph_facts: list[GraphFact] = field(default_factory=list)
    episodes: list[EpisodeHit] = field(default_factory=list)
    procedures: list[ProcedureHit] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    recent_turns: list[Message] = field(default_factory=list)


def augment(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    user_message: str,
    *,
    embedding_client: EmbeddingClient | None = None,
    llm: LLMClient | None = None,
    token_overlap_index: dict[str, list[str]] | None = None,
    session_id: str | None = None,
) -> AugmentedContext:
    ctx = AugmentedContext()
    if cfg.user_md_path.exists():
        ctx.user_md = cfg.user_md_path.read_text(encoding="utf-8")
    if cfg.memory_md_path.exists():
        ctx.memory_md = cfg.memory_md_path.read_text(encoding="utf-8")

    # Working-memory tier: the last N raw turns of the active session, so facts
    # stated this session are recallable before any dream has consolidated them.
    # `conn` here is the READ connection; recent_messages is a plain SELECT.
    if session_id is not None and cfg.working_memory_turns > 0:
        ctx.recent_turns = recent_messages(conn, session_id, cfg.working_memory_turns)

    # Pull a wider candidate pool when reranking is likely so the reranker
    # has room to reorder beyond the top-fts_top_k window; the final result
    # is still trimmed to fts_top_k after rerank.
    candidate_k = max(cfg.fts_top_k, cfg.rerank_top_k)
    fts = _fts_search(conn, user_message, top_k=candidate_k)
    vec: list[FtsHit] = []
    if embedding_client is not None:
        vec = _vector_search(
            conn,
            embedding_client,
            user_message,
            top_k=candidate_k,
            max_scan=cfg.embedding_max_scan,
        )
        ctx.fts_hits = _rrf_merge(fts, vec, top_k=candidate_k)
    else:
        ctx.fts_hits = fts

    rerank_enabled = (
        cfg.rerank_model == "cross-encoder" or llm is not None
    )
    if rerank_enabled and should_rerank(fts, vec, ctx.fts_hits, cfg.rerank_ambiguity_threshold):
        log.debug("rerank.triggered model=%s", cfg.rerank_model)
        ctx.fts_hits = run_rerank(
            user_message,
            list(ctx.fts_hits[: cfg.rerank_top_k]),
            top_k=cfg.fts_top_k,
            model=cfg.rerank_model,
            llm=llm,
            cross_encoder_model=cfg.rerank_cross_encoder_model,
        )
        # rerank() reorders via dataclasses.replace, preserving why_retrieved;
        # tag the survivors so the chip trail shows the rerank step. New list
        # per hit to avoid mutating the (shared) pre-rerank chip list.
        for hit in ctx.fts_hits:
            if hit.score_kind == "reranked":
                hit.why_retrieved = [*hit.why_retrieved, "reranked"]
    else:
        log.debug("rerank.skipped")
        ctx.fts_hits = ctx.fts_hits[: cfg.fts_top_k]

    ctx.episodes = _episode_search(
        conn, user_message,
        top_k=cfg.fts_top_k,
        embedding_client=embedding_client,
    )

    ctx.procedures = _procedure_search(conn, user_message, top_k=cfg.fts_top_k)

    matched = match_known_entities(conn, user_message)
    type_expanded, expansion_info = _expand_entities_by_type(conn, matched)
    # Free-text type/property expansion: the user may ask "what build tools
    # do we use?" without naming any specific entity. Map type/property
    # keywords in the message to canonicals tagged with that type or
    # property; merge into the entity set so Source 1 of the graph lookup
    # picks them up.
    query_type_expanded, query_type_info = _expand_entities_from_query(
        conn, user_message
    )
    overlap_expanded, overlap_info = _expand_entities_by_token_overlap(
        conn, matched,
        max_per_entity=cfg.graph_token_overlap_max_per_entity,
        common_token_threshold=cfg.graph_token_overlap_threshold,
        token_index=token_overlap_index,
    )
    combined = list(type_expanded)
    for e in query_type_expanded:
        if e not in combined:
            combined.append(e)
        # Surface the type label that justified the addition through the same
        # `entity_type:` reason channel as direct-entity type expansion.
        expansion_info.setdefault(e, query_type_info[e])
    for e in overlap_expanded:
        if e not in combined:
            combined.append(e)
    ctx.matched_entities = combined
    routed = route_predicates(user_message)
    ctx.graph_facts = _graph_lookup(
        conn, cfg, user_message, ctx.matched_entities, expansion_info, routed,
        overlap_info=overlap_info,
        embedding_client=embedding_client,
    )
    return ctx


def should_rerank(
    fts_hits: list[FtsHit],
    vec_hits: list[FtsHit],
    fused: list[FtsHit],
    threshold: float,
) -> bool:
    if not fts_hits or not vec_hits:
        return False

    if fts_hits and vec_hits:
        if fts_hits[0].chunk_id == vec_hits[0].chunk_id:
            return False

    if len(fused) < 2:
        return False

    score_1 = fused[0].score
    score_2 = fused[1].score
    if score_1 <= 0:
        return False
    drop = 1.0 - (score_2 / score_1)
    if drop > threshold:
        return False

    return True


_FTS_SAFE = re.compile(r"[^A-Za-z0-9_\- ]+")


def _fts_search(conn: sqlite3.Connection, query: str, *, top_k: int) -> list[FtsHit]:
    cleaned = _FTS_SAFE.sub(" ", query).strip()
    if not cleaned:
        return []
    # Build an OR query across tokens so partial matches still surface results.
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in tokens)

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

    chip = f'fts_match("{" ".join(tokens)}")'
    return [
        FtsHit(
            chunk_id=r["chunk_id"],
            session_id=r["session_id"],
            text=r["text"],
            score=float(r["score"]),
            why_retrieved=[chip],
        )
        for r in rows
    ]


def _embeddings_compatible(conn: sqlite3.Connection, embedder: EmbeddingClient) -> bool:
    """Guard against querying chunk embeddings written by a *different* model or
    dimension than the active embedder. A dim mismatch is caught per-vector in
    the python path, but the vec0 fast path has no such check, and a model swap
    that keeps the same dim would otherwise return silent garbage. When the
    stored model/dim disagree with the live client, skip vector search (FTS
    still runs) and log so the operator can re-embed.

    Checks every distinct (model, dim) pair, not just one row: a mixed corpus
    (some rows from an old model, some from the new) must still be treated as
    incompatible unless *all* stored vectors match the active client — otherwise
    a single matching first row would wave through a partially-stale corpus."""
    rows = conn.execute(
        "SELECT DISTINCT model, dim FROM chunk_embeddings"
    ).fetchall()
    if not rows:
        return True  # nothing stored yet — nothing to mismatch
    mismatched = [
        (r["model"], r["dim"])
        for r in rows
        if r["model"] != embedder.model or r["dim"] != embedder.dim
    ]
    if mismatched:
        log.warning(
            "vector search skipped: stored embeddings include model/dim %s but "
            "active client is model=%s dim=%s; re-embed to enable semantic recall",
            mismatched, embedder.model, embedder.dim,
        )
        return False
    return True


def _vector_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
    max_scan: int,
) -> list[FtsHit]:
    from hymem.core import db as core_db

    if not _embeddings_compatible(conn, embedder):
        return []
    if core_db.has_vec_table(conn):
        return _vec_search(conn, embedder, query, top_k=top_k)
    return _python_cosine_search(conn, embedder, query, top_k=top_k, max_scan=max_scan)


def _vec_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
) -> list[FtsHit]:
    from hymem.core import db as core_db

    qvec = embedder.embed([query])[0]
    hits = core_db.vec_search(conn, qvec, top_k)

    result: list[FtsHit] = []
    for chunk_rowid, distance in hits:
        row = conn.execute(
            "SELECT id AS chunk_id, session_id, text FROM chunks WHERE rowid = ?",
            (chunk_rowid,),
        ).fetchone()
        if row:
            sim = 1.0 / (1.0 + distance)
            result.append(
                FtsHit(
                    chunk_id=row["chunk_id"],
                    session_id=row["session_id"],
                    text=row["text"],
                    score=float(sim),
                    score_kind="vec",
                    why_retrieved=[f"vec_topk(sim={sim:.3f})"],
                )
            )
    return result


def _python_cosine_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
    max_scan: int,
) -> list[FtsHit]:
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
        vec = decode_vector(r["vector_json"])
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
            score_kind="vec",
            why_retrieved=[f"vec_topk(sim={sim:.3f})"],
        )
        for sim, r in scored[:top_k]
    ]


def _rrf_merge(
    fts: list[FtsHit], vec: list[FtsHit], *, top_k: int, k: int = 60
) -> list[FtsHit]:
    # Reciprocal rank fusion: score = sum(1 / (k + rank)) across each list.
    by_id: dict[str, FtsHit] = {}
    scores: dict[str, float] = {}
    fts_ids = {h.chunk_id for h in fts}
    vec_ids = {h.chunk_id for h in vec}
    for rank, hit in enumerate(fts, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.chunk_id, hit)
    for rank, hit in enumerate(vec, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.chunk_id, hit)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        replace(
            by_id[cid], score=score, score_kind="rrf",
            why_retrieved=_rrf_chips(by_id[cid].why_retrieved, cid, fts_ids, vec_ids, score),
        )
        for cid, score in ordered[:top_k]
    ]


def _rrf_chips(
    base: list[str], item_id: str, fts_ids: set[str], vec_ids: set[str], score: float
) -> list[str]:
    """Carry the kept hit's source chip (fts_match / vec_topk) and append a
    fused `rrf(<sources>, <score>)` chip naming which lists contributed."""
    if item_id in fts_ids and item_id in vec_ids:
        sources = "fts+vec"
    elif item_id in fts_ids:
        sources = "fts"
    else:
        sources = "vec"
    return [*base, f"rrf({sources}, {score:.4f})"]


_EDGE_SELECT = """
    SELECT id, subject_canonical AS s, predicate AS p, object_canonical AS o,
           pos_evidence AS pos, neg_evidence AS neg, derived,
           (julianday('now') - julianday(last_seen)) AS days_since
    FROM knowledge_graph
"""


def _graph_lookup(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    query: str,
    entities: list[str],
    expansion_info: dict[str, str],
    routed: frozenset[str],
    *,
    overlap_info: dict[str, str] | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[GraphFact]:
    """Hybrid edge ranker: gathers candidates from entity matches, semantic KNN,
    and predicate routing, then scores by semantic × confidence × recency × boost.

    With no embedding client the semantic source is skipped and the score
    collapses to confidence × recency × predicate_boost — close to the prior
    entity-anchored leaderboard behaviour.

    When no predicate routes (`routed` is empty), the ranker switches into a
    fallback with per-candidate scoring:
      - Entity-anchored candidates score by confidence × recency. Edge-level
        embeddings are too noisy ("rejects working" matches "projects working"
        for surface reasons) to compete with a deterministic entity hit, so
        Source 2 is skipped entirely when entities matched.
      - Semantic-only candidates (only present when no entity matched) score
        by similarity × confidence × recency.
      - Otherwise (no signal at all) score collapses to confidence × recency
        over a recent-edges seed so something graph-shaped is still shown.
    """
    fallback = not routed
    overlap_info = overlap_info or {}
    candidates: dict[tuple[str, str, str], dict] = {}

    def _ensure(row: sqlite3.Row) -> dict:
        key = (row["s"], row["p"], row["o"])
        c = candidates.get(key)
        if c is None:
            c = {
                "edge_id": int(row["id"]),
                "s": row["s"],
                "p": row["p"],
                "o": row["o"],
                "pos": int(row["pos"]),
                "neg": int(row["neg"]),
                "derived": bool(row["derived"]),
                "days_since": (
                    float(row["days_since"]) if row["days_since"] is not None else 0.0
                ),
                "semantic_score": 0.0,
                "semantic_retrieved": False,
                "entity_match": False,
                "entity_types": set(),
                "overlap_tokens": set(),
                "direct_anchor": False,
            }
            candidates[key] = c
        return c

    # Source 1 — entity-anchored (always).
    for entity in entities:
        rows = conn.execute(
            _EDGE_SELECT
            + """
            WHERE status = 'active'
              AND (subject_canonical = ? OR object_canonical = ?)
            ORDER BY (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) DESC,
                     last_reinforced DESC
            LIMIT ?
            """,
            (entity, entity, cfg.graph_top_k_per_entity),
        ).fetchall()
        for r in rows:
            c = _ensure(r)
            c["entity_match"] = True
            if entity in expansion_info:
                c["entity_types"].add(expansion_info[entity])
            if entity in overlap_info:
                c["overlap_tokens"].add(overlap_info[entity])
            else:
                # Any non-overlap anchor (direct match or type expansion) means
                # the edge isn't *only* surfaced via token-overlap — full weight.
                c["direct_anchor"] = True

    # Source 2 — semantic KNN. Skipped in the fallback path when entities
    # matched: edge-level embeddings are short and noisy, so they crowd out
    # entity-anchored candidates when the user named a known entity.
    skip_semantic = fallback and bool(entities)
    if embedding_client is not None and not skip_semantic:
        for edge_id, semantic_score in _semantic_edge_hits(
            conn, cfg, embedding_client, query
        ):
            row = conn.execute(
                _EDGE_SELECT + " WHERE id = ? AND status = 'active'",
                (edge_id,),
            ).fetchone()
            if row is None:
                continue
            c = _ensure(row)
            c["semantic_score"] = max(c["semantic_score"], semantic_score)
            c["semantic_retrieved"] = True

    # Source 3 — predicate-routed.
    if routed:
        pred_placeholders = ",".join("?" * len(routed))
        rows = conn.execute(
            _EDGE_SELECT
            + f"""
            WHERE status = 'active' AND predicate IN ({pred_placeholders})
            ORDER BY (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) DESC,
                     last_seen DESC
            LIMIT ?
            """,
            list(routed) + [cfg.graph_predicate_top_k],
        ).fetchall()
        for r in rows:
            _ensure(r)

    # Recency-only seeding: if the fallback path has no candidates at all
    # (no entity match, no semantic hit), pull a small set of recent active
    # edges so the graph_facts list isn't empty when something could be shown.
    if fallback and not candidates:
        for row in _recency_edges(conn, cfg.graph_top_k):
            _ensure(row)

    results: list[GraphFact] = []
    for c in candidates.values():
        confidence = (c["pos"] + 1.0) / (c["pos"] + c["neg"] + 2.0)
        recency_weight = math.exp(-c["days_since"] / cfg.graph_recency_half_life_days)
        semantic_score = c["semantic_score"]
        in_routed = c["p"] in routed

        why: list[str] = []
        if fallback:
            if c["entity_match"]:
                overlap_only = not c["direct_anchor"]
                if overlap_only:
                    score = (
                        cfg.graph_token_overlap_weight
                        * confidence
                        * recency_weight
                    )
                    why.append("fallback:entity_anchored:overlap")
                    for tok in sorted(c["overlap_tokens"]):
                        why.append(f"overlap_via:{tok}")
                else:
                    score = confidence * recency_weight
                    why.append("fallback:entity_anchored")
            elif semantic_score > 0:
                score = semantic_score * confidence * recency_weight
                why.append("fallback:semantic")
            else:
                score = confidence * recency_weight
                why.append("fallback:recency")
        else:
            predicate_boost = cfg.graph_predicate_boost if in_routed else 1.0
            score = (
                confidence
                * recency_weight
                * (semantic_score if semantic_score > 0 else 1.0)
                * predicate_boost
            )

        if c["semantic_retrieved"]:
            why.append(f"semantic_{max(0.0, semantic_score):.2f}")
        if in_routed:
            why.append(f"predicate:{c['p']}")
        for entity_type in sorted(c["entity_types"]):
            why.append(f"entity_type:{entity_type}")
        if c["days_since"] <= cfg.graph_recency_recent_days:
            why.append(f"recency_{round(c['days_since'])}d")
        if c["entity_match"]:
            why.append("entity_match")

        total_evidence = c["pos"] + c["neg"]
        hedge = (
            confidence < cfg.hedge_confidence_threshold
            or total_evidence < cfg.hedge_min_evidence
        )
        results.append(
            GraphFact(
                subject=c["s"],
                predicate=c["p"],
                object=c["o"],
                confidence=confidence,
                pos_evidence=c["pos"],
                neg_evidence=c["neg"],
                derived=c["derived"],
                why_retrieved=why,
                score=score,
                hedge_recommended=hedge,
            )
        )

    results.sort(key=lambda f: f.score, reverse=True)
    return results[: cfg.graph_top_k]


def _recency_edges(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Pull the most recent active edges by confidence × recency.

    Used by the no-predicate fallback when neither entity match nor semantic
    KNN produced any candidates, so something graph-shaped is still returned.
    """
    return conn.execute(
        _EDGE_SELECT
        + """
        WHERE status = 'active'
        ORDER BY (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) DESC,
                 last_seen DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _semantic_edge_hits(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    embedder: EmbeddingClient,
    query: str,
) -> list[tuple[int, float]]:
    """Return (edge_id, semantic_score) pairs for edges similar to the query."""
    from hymem.core import db as core_db

    if core_db._load_vec_extension(conn) and core_db.has_vec_table(
        conn, table="vec_edges"
    ):
        qvec = embedder.embed([query])[0]
        hits = core_db.vec_search(
            conn, qvec, cfg.graph_semantic_top_k, table="vec_edges"
        )
        return [(edge_id, 1.0 / (1.0 + distance)) for edge_id, distance in hits]
    return _python_cosine_edge_search(
        conn, embedder, query,
        top_k=cfg.graph_semantic_top_k, max_scan=cfg.embedding_max_scan,
    )


def _python_cosine_edge_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
    max_scan: int,
) -> list[tuple[int, float]]:
    rows = conn.execute(
        """
        SELECT kg.id AS edge_id, e.vector_json
        FROM knowledge_graph kg
        JOIN edge_embeddings e
          ON e.edge_text = kg.subject_canonical || ' ' || kg.predicate || ' '
                           || kg.object_canonical
        WHERE kg.status = 'active'
        ORDER BY kg.last_seen DESC
        LIMIT ?
        """,
        (max_scan,),
    ).fetchall()
    if not rows:
        return []

    qvec = embedder.embed([query])[0]
    qnorm = math.sqrt(sum(x * x for x in qvec)) or 1.0

    scored: list[tuple[float, int]] = []
    for r in rows:
        vec = decode_vector(r["vector_json"])
        if len(vec) != len(qvec):
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
        sim = dot / (qnorm * vnorm)
        scored.append((sim, r["edge_id"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(edge_id, sim) for sim, edge_id in scored[:top_k]]


def _episode_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 3,
    embedding_client: EmbeddingClient | None = None,
) -> list[EpisodeHit]:
    """Episode retrieval. Always runs the FTS path; when an embedding client
    is configured *and* vec_episodes has rows, also runs semantic KNN over
    title+summary embeddings and RRF-fuses the two ranked lists.

    Falls back to FTS-only (with score_kind="bm25") when there's no embedder,
    no vec_episodes table, or vec returns nothing — preserving the original
    behavior for clients that haven't dreamed any episode embeddings yet.
    """
    from hymem.core import db as core_db

    cleaned = _FTS_SAFE.sub(" ", query).strip()
    fts_hits: list[EpisodeHit] = []
    if cleaned:
        tokens = [t for t in cleaned.split() if len(t) >= 2]
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            episode_chip = f'episode_fts("{" ".join(tokens)}")'
            try:
                rows = conn.execute(
                    """SELECT e.id, e.session_id, e.title, e.summary, bm25(episodes_fts) AS score
                       FROM episodes_fts
                       JOIN episodes e ON e.rowid = episodes_fts.rowid
                       WHERE episodes_fts MATCH ?
                       ORDER BY score
                       LIMIT ?""",
                    (fts_query, top_k * 2),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for r in rows:
                fts_hits.append(
                    EpisodeHit(
                        episode_id=r["id"],
                        session_id=r["session_id"],
                        title=r["title"],
                        summary=r["summary"][:300],
                        score=float(r["score"]),
                        score_kind="bm25",
                        why_retrieved=[episode_chip],
                    )
                )

    vec_hits: list[EpisodeHit] = []
    if (
        embedding_client is not None
        and core_db._load_vec_extension(conn)
        and core_db.has_vec_table(conn, table="vec_episodes")
    ):
        qvec = embedding_client.embed([query])[0]
        try:
            hit_rows = core_db.vec_search(
                conn, qvec, top_k * 2, table="vec_episodes"
            )
        except Exception:
            hit_rows = []
        for rowid, distance in hit_rows:
            r = conn.execute(
                "SELECT id, session_id, title, summary FROM episodes WHERE rowid = ?",
                (rowid,),
            ).fetchone()
            if r is None:
                continue
            sim = 1.0 / (1.0 + distance)
            vec_hits.append(
                EpisodeHit(
                    episode_id=r["id"],
                    session_id=r["session_id"],
                    title=r["title"],
                    summary=r["summary"][:300],
                    score=float(sim),
                    score_kind="vec",
                    why_retrieved=[f"episode_vec(sim={sim:.3f})"],
                )
            )
            log.info(json.dumps({"ep": r["id"], "kind": "vec", "q": query}, default=str))

    if not vec_hits:
        return fts_hits[:top_k]
    if not fts_hits:
        return vec_hits[:top_k]
    merged = _rrf_merge_episodes(fts_hits, vec_hits, top_k=top_k)
    for hit in merged:
        log.info(json.dumps({"ep": hit.episode_id, "kind": "rrf", "q": query}, default=str))
    return merged


def _rrf_merge_episodes(
    fts: list[EpisodeHit],
    vec: list[EpisodeHit],
    *,
    top_k: int,
    k: int = 60,
) -> list[EpisodeHit]:
    """RRF over two ranked episode lists. Mirrors `_rrf_merge` (for chunks)
    but keyed on episode_id."""
    by_id: dict[str, EpisodeHit] = {}
    scores: dict[str, float] = {}
    for rank, hit in enumerate(fts, start=1):
        scores[hit.episode_id] = scores.get(hit.episode_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.episode_id, hit)
    for rank, hit in enumerate(vec, start=1):
        scores[hit.episode_id] = scores.get(hit.episode_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.episode_id, hit)
    fts_ids = {h.episode_id for h in fts}
    vec_ids = {h.episode_id for h in vec}
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        replace(
            by_id[eid], score=score, score_kind="rrf",
            why_retrieved=_episode_rrf_chips(
                by_id[eid].why_retrieved, eid, fts_ids, vec_ids, score
            ),
        )
        for eid, score in ordered[:top_k]
    ]


def _episode_rrf_chips(
    base: list[str], item_id: str, fts_ids: set[str], vec_ids: set[str], score: float
) -> list[str]:
    if item_id in fts_ids and item_id in vec_ids:
        sources = "fts+vec"
    elif item_id in fts_ids:
        sources = "fts"
    else:
        sources = "vec"
    return [*base, f"episode_rrf({sources}, {score:.4f})"]


def _procedure_search(conn: sqlite3.Connection, query: str, top_k: int = 3) -> list[ProcedureHit]:
    cleaned = _FTS_SAFE.sub(" ", query).strip()
    if not cleaned:
        return []
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in tokens)

    try:
        rows = conn.execute(
            """SELECT p.id, p.session_id, p.name, p.description, p.steps,
                      bm25(procedures_fts) AS score
               FROM procedures_fts
               JOIN procedures p ON p.rowid = procedures_fts.rowid
               WHERE procedures_fts MATCH ?
                 AND p.status = 'active'
               ORDER BY score
               LIMIT ?""",
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    chip = f'procedure_fts("{" ".join(tokens)}")'
    result: list[ProcedureHit] = []
    for r in rows:
        try:
            steps = json.loads(r["steps"]) if r["steps"] else []
        except json.JSONDecodeError:
            steps = []
        result.append(ProcedureHit(
            procedure_id=r["id"],
            session_id=r["session_id"],
            name=r["name"],
            description=r["description"] or "",
            steps=steps,
            score=float(r["score"]),
            why_retrieved=[chip],
        ))
    return result


def build_token_overlap_index(
    conn: sqlite3.Connection,
    *,
    write_conn: sqlite3.Connection | None = None,
) -> dict[str, list[str]]:
    """Map every underscore-segment token to the active canonicals containing
    it. Caller-cacheable; rebuild after a dream cycle that may have added,
    retracted, or merged edges.

    On a warm database the persistent ``token_overlap_index`` table is read
    directly (O(index rows) instead of O(active edges)). When the table is
    empty — cold start, post-migration, or after runner invalidated it —
    the function falls back to the full canonical scan and, if *write_conn* is
    provided, persists the result so the next cold start is fast.

    Public (no leading underscore) so callers — HyMem instances, background
    workers — can build, stash, and pass it back through `augment()` to avoid
    re-scanning the canonical set on every query. At a few hundred canonicals
    the scan is sub-millisecond; at tens of thousands it begins to matter.
    """
    persisted = conn.execute(
        "SELECT token, canonical FROM token_overlap_index"
    ).fetchall()
    if persisted:
        by_token: dict[str, list[str]] = {}
        for r in persisted:
            by_token.setdefault(r["token"], []).append(r["canonical"])
        return by_token

    rows = conn.execute(
        "SELECT DISTINCT subject_canonical AS c FROM knowledge_graph WHERE status='active' "
        "UNION "
        "SELECT DISTINCT object_canonical FROM knowledge_graph WHERE status='active'"
    ).fetchall()
    by_token = {}
    for r in rows:
        c = r["c"]
        for tok in c.split("_"):
            if tok:
                by_token.setdefault(tok, []).append(c)

    if write_conn is not None and by_token:
        # isolation_level=None means autocommit — wrap explicitly so a partial
        # write doesn't leave a half-populated table that subsequent cold starts
        # would trust as complete.
        write_conn.execute("BEGIN IMMEDIATE")
        try:
            write_conn.executemany(
                "INSERT OR IGNORE INTO token_overlap_index(token, canonical) VALUES (?, ?)",
                [(tok, c) for tok, canons in by_token.items() for c in canons],
            )
            write_conn.execute("COMMIT")
        except Exception:
            write_conn.execute("ROLLBACK")
            raise

    return by_token


def _expand_entities_by_token_overlap(
    conn: sqlite3.Connection,
    entities: list[str],
    *,
    max_per_entity: int = 5,
    common_token_threshold: int = 20,
    token_index: dict[str, list[str]] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Find other canonicals sharing a rare token segment with matched entities.

    A matched canonical like `atta_van_westreenen` has segments `atta`, `van`,
    `westreenen`. For each segment, look up other canonicals containing that
    token as a complete underscore-delimited segment — so `atta_projects`
    surfaces via the `atta` token while keeping single-prefix collisions like
    `attach_handler` out (different segment). Common tokens appearing in more
    than `common_token_threshold` canonicals (`system`, `service`, `data`) are
    dropped as noise. Returns the new entities to consider for Source 1, plus
    an `{entity: token}` map for the `overlap_via:` reason code.

    Pass `token_index` to reuse a prebuilt token-segment index (see
    `build_token_overlap_index`) so repeated `augment()` calls do not re-scan
    the canonical set. Falls back to an on-the-fly build when None.

    Returns ([], {}) when no input is multi-segment or no co-occurrence is
    found in the active graph. Single-token canonicals never trigger expansion
    — there is nothing to overlap on.
    """
    if not entities:
        return [], {}

    multi_token = [e for e in entities if "_" in e]
    if not multi_token:
        return [], {}

    by_token = token_index if token_index is not None else build_token_overlap_index(conn)

    matched_set = set(entities)
    seen: set[str] = set(matched_set)
    expansions: list[str] = []
    overlap_info: dict[str, str] = {}

    for entity in multi_token:
        added = 0
        for tok in entity.split("_"):
            if not tok or added >= max_per_entity:
                continue
            holders = by_token.get(tok, [])
            if len(holders) > common_token_threshold:
                continue  # common-token noise (`system`, `data`, …)
            for c in holders:
                if c in seen:
                    continue
                seen.add(c)
                expansions.append(c)
                overlap_info[c] = tok
                added += 1
                if added >= max_per_entity:
                    break
    return expansions, overlap_info


# Maps category-style query phrases to the entity-type labels emitted by the
# extraction prompt. Lets "what build tools do we use?" pull every canonical
# tagged `package_manager` even when no specific package manager is named.
# Phrases match against a normalised, lowercased copy of the user message.
_TYPE_QUERY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "package_manager": (
        "package manager", "package management", "build tool", "build tools",
        "dependency manager", "dependency management",
    ),
    "database": ("database", "databases", "datastore", "data store"),
    "language": ("programming language", "languages", "language stack"),
    "framework": ("framework", "frameworks", "web framework"),
    "service": ("service", "services", "microservice", "microservices"),
    "container": ("container", "containers", "containerization", "containerisation"),
    "platform": ("platform", "platforms", "cloud platform"),
    "testing_framework": ("test framework", "testing framework", "test runner", "test tooling"),
    "ci_tool": ("ci tool", "ci tools", "ci/cd", "ci pipeline", "continuous integration"),
    "monitoring_tool": ("monitoring", "observability", "metrics tool"),
    "config_file": ("config file", "configuration file", "config files"),
    "identity_provider": ("identity provider", "auth provider", "sso", "single sign-on"),
    "message_broker": ("message broker", "message queue", "queue system", "pubsub", "pub/sub"),
    "protocol": ("protocol", "protocols"),
    "environment": ("environment", "environments", "deployment environment"),
    "api": ("api", "apis"),
    "library": ("library", "libraries", "dependency", "dependencies"),
    "tool": ("tooling", "dev tool", "dev tools"),
}

# Maps category-style query phrases to entity_properties (key, value) filters.
# Lets a question about "build tools" also surface entities the LLM tagged
# with `category=build_tool` even if they have no entity_types row.
_PROPERTY_QUERY_KEYWORDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("category", "build_tool"): ("build tool", "build tools"),
    ("category", "database"): ("database", "databases"),
    ("category", "testing"): ("test framework", "testing framework", "test runner"),
    ("category", "deployment"): ("deployment", "deploy tool"),
    ("category", "observability"): ("monitoring", "observability"),
}


def _expand_entities_from_query(
    conn: sqlite3.Connection,
    user_message: str,
    *,
    max_per_type: int = 10,
) -> tuple[list[str], dict[str, str]]:
    """Pull canonicals whose type or property matches a category-style query.

    Scans the lowercased message for the configured keyword phrases. For each
    hit, returns up to ``max_per_type`` canonicals tagged with that type (via
    ``entity_types``) or carrying the matching ``(key, value)`` pair in
    ``entity_properties``. Returns ``(canonicals, {canonical: type_label})``
    where the label is the type or ``"key=value"`` rendering of the property.
    """
    msg = user_message.lower()
    matched_types: set[str] = set()
    for type_label, phrases in _TYPE_QUERY_KEYWORDS.items():
        if any(p in msg for p in phrases):
            matched_types.add(type_label)

    matched_props: list[tuple[str, str]] = []
    for (key, value), phrases in _PROPERTY_QUERY_KEYWORDS.items():
        if any(p in msg for p in phrases):
            matched_props.append((key, value))

    if not matched_types and not matched_props:
        return [], {}

    seen: set[str] = set()
    out: list[str] = []
    info: dict[str, str] = {}

    for type_label in sorted(matched_types):
        rows = conn.execute(
            "SELECT entity_canonical FROM entity_types WHERE type = ? LIMIT ?",
            (type_label, max_per_type),
        ).fetchall()
        for r in rows:
            ent = r["entity_canonical"]
            if ent in seen:
                continue
            seen.add(ent)
            out.append(ent)
            info[ent] = type_label

    for key, value in matched_props:
        rows = conn.execute(
            "SELECT entity_canonical FROM entity_properties WHERE key = ? AND value = ? LIMIT ?",
            (key, value, max_per_type),
        ).fetchall()
        for r in rows:
            ent = r["entity_canonical"]
            if ent in seen:
                continue
            seen.add(ent)
            out.append(ent)
            info[ent] = f"{key}={value}"

    return out, info


def _expand_entities_by_type(
    conn: sqlite3.Connection,
    entities: list[str],
    max_expanded: int = 10,
) -> tuple[list[str], dict[str, str]]:
    """For matched entities, find other entities of the same type.

    Returns (all_entities, expansion_info) where expansion_info maps each
    expanded entity to the type label that surfaced it (used for the
    `entity_type:` reason code).
    """
    if not entities:
        return entities, {}

    placeholders = ",".join("?" * len(entities))
    type_rows = conn.execute(
        f"SELECT DISTINCT type FROM entity_types WHERE entity_canonical IN ({placeholders})",
        entities,
    ).fetchall()
    if not type_rows:
        return entities, {}

    types = [r[0] for r in type_rows]
    type_placeholders = ",".join("?" * len(types))

    expanded_rows = conn.execute(
        f"""SELECT DISTINCT entity_canonical, type FROM entity_types
            WHERE type IN ({type_placeholders})
            AND entity_canonical NOT IN ({placeholders})
            LIMIT ?""",
        types + entities + [max_expanded],
    ).fetchall()

    expansion_info: dict[str, str] = {}
    expanded: list[str] = []
    for r in expanded_rows:
        ent = r["entity_canonical"]
        if ent not in expansion_info:
            expansion_info[ent] = r["type"]
            expanded.append(ent)

    return entities + expanded, expansion_info
