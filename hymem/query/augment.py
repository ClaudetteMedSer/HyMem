from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass, field, replace

from hymem.config import HyMemConfig
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import RERANK_SYSTEM, RERANK_USER_TEMPLATE
from hymem.query.entities import match_known_entities
from hymem.query.predicate_routing import route_predicates

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


@dataclass
class EpisodeHit:
    episode_id: str
    session_id: str
    title: str
    summary: str
    score: float


@dataclass
class FtsHit:
    chunk_id: str
    session_id: str
    text: str
    score: float
    score_kind: str = "bm25"


@dataclass
class ProcedureHit:
    procedure_id: str
    session_id: str
    name: str
    description: str
    steps: list[dict]
    score: float


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
    episodes: list[EpisodeHit] = field(default_factory=list)
    procedures: list[ProcedureHit] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)


def augment(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    user_message: str,
    *,
    embedding_client: EmbeddingClient | None = None,
    llm: LLMClient | None = None,
) -> AugmentedContext:
    ctx = AugmentedContext()
    if cfg.user_md_path.exists():
        ctx.user_md = cfg.user_md_path.read_text(encoding="utf-8")
    if cfg.memory_md_path.exists():
        ctx.memory_md = cfg.memory_md_path.read_text(encoding="utf-8")

    fts = _fts_search(conn, user_message, top_k=cfg.fts_top_k)
    vec: list[FtsHit] = []
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

    if llm is not None and should_rerank(fts, vec, ctx.fts_hits, cfg.rerank_ambiguity_threshold):
        log.debug("rerank.triggered")
        ctx.fts_hits = _rerank(user_message, list(ctx.fts_hits), llm, top_k=cfg.fts_top_k)
    else:
        log.debug("rerank.skipped")

    ctx.episodes = _episode_search(conn, user_message, top_k=cfg.fts_top_k)

    ctx.procedures = _procedure_search(conn, user_message, top_k=cfg.fts_top_k)

    matched = match_known_entities(conn, user_message)
    ctx.matched_entities, expansion_info = _expand_entities_by_type(conn, matched)
    routed = route_predicates(user_message)
    if ctx.matched_entities or embedding_client is not None or routed:
        ctx.graph_facts = _graph_lookup(
            conn, cfg, user_message, ctx.matched_entities, expansion_info, routed,
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
    from hymem.core import db as core_db

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
            result.append(
                FtsHit(
                    chunk_id=row["chunk_id"],
                    session_id=row["session_id"],
                    text=row["text"],
                    score=float(1.0 / (1.0 + distance)),
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
    embedding_client: EmbeddingClient | None = None,
) -> list[GraphFact]:
    """Hybrid edge ranker: gathers candidates from entity matches, semantic KNN,
    and predicate routing, then scores by semantic × confidence × recency × boost.

    With no embedding client the semantic source is skipped and the score
    collapses to confidence × recency × predicate_boost — close to the prior
    entity-anchored leaderboard behaviour.
    """
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
                "entity_match": False,
                "entity_types": set(),
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

    # Source 2 — semantic KNN (only with an embedding client).
    if embedding_client is not None:
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

    results: list[GraphFact] = []
    for c in candidates.values():
        confidence = (c["pos"] + 1.0) / (c["pos"] + c["neg"] + 2.0)
        recency_weight = math.exp(-c["days_since"] / cfg.graph_recency_half_life_days)
        semantic_score = c["semantic_score"]
        in_routed = c["p"] in routed
        predicate_boost = cfg.graph_predicate_boost if in_routed else 1.0
        score = (
            confidence
            * recency_weight
            * (semantic_score if semantic_score > 0 else 1.0)
            * predicate_boost
        )

        why: list[str] = []
        if semantic_score > 0:
            why.append(f"semantic_{semantic_score:.2f}")
        if in_routed:
            why.append(f"predicate:{c['p']}")
        for entity_type in sorted(c["entity_types"]):
            why.append(f"entity_type:{entity_type}")
        if c["days_since"] <= cfg.graph_recency_recent_days:
            why.append(f"recency_{round(c['days_since'])}d")
        if c["entity_match"]:
            why.append("entity_match")

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
            )
        )

    results.sort(key=lambda f: f.score, reverse=True)
    return results[: cfg.graph_top_k]


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
        vec = json.loads(r["vector_json"])
        if len(vec) != len(qvec):
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
        sim = max(0.0, dot / (qnorm * vnorm))
        scored.append((sim, r["edge_id"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(edge_id, sim) for sim, edge_id in scored[:top_k]]


def _episode_search(conn: sqlite3.Connection, query: str, top_k: int = 3) -> list[EpisodeHit]:
    cleaned = _FTS_SAFE.sub(" ", query).strip()
    if not cleaned:
        return []
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in tokens)

    try:
        rows = conn.execute(
            """SELECT e.id, e.session_id, e.title, e.summary, bm25(episodes_fts) AS score
               FROM episodes_fts
               JOIN episodes e ON e.rowid = episodes_fts.rowid
               WHERE episodes_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        EpisodeHit(
            episode_id=r["id"],
            session_id=r["session_id"],
            title=r["title"],
            summary=r["summary"][:300],
            score=float(r["score"]),
        )
        for r in rows
    ]


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
               ORDER BY score
               LIMIT ?""",
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

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
        ))
    return result


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


def _rerank(
    query: str,
    candidates: list[FtsHit],
    llm: LLMClient,
    top_k: int,
) -> list[FtsHit]:
    if not candidates:
        return candidates

    excerpts_lines = []
    for i, hit in enumerate(candidates):
        excerpts_lines.append(f"[{i}] {hit.text[:400]}")
    excerpts = "\n\n".join(excerpts_lines)

    request = LLMRequest(
        system=RERANK_SYSTEM,
        user=RERANK_USER_TEMPLATE.format(query=query, excerpts=excerpts),
        response_format="json",
    )
    raw = llm.complete(request)

    try:
        ratings = json.loads(raw)
    except json.JSONDecodeError:
        return candidates[:top_k]

    if not isinstance(ratings, list):
        return candidates[:top_k]

    relevance: dict[int, int] = {}
    for item in ratings:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        score = item.get("relevance")
        if isinstance(idx, int) and isinstance(score, (int, float)) and 0 <= idx < len(candidates):
            relevance[idx] = int(score)

    scored = []
    for i, hit in enumerate(candidates):
        rrf_score = hit.score if hit.score_kind == "rrf" else 0.0
        llm_score = relevance.get(i, 3)
        combined = llm_score * 100 + rrf_score
        scored.append((combined, hit))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [replace(hit, score=float(score), score_kind="reranked")
            for score, hit in scored[:top_k]]
