from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from hymem.config import HyMemConfig
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


@dataclass
class AugmentedContext:
    """Structured context for the host (Hermes) to assemble into its prompt.

    Hermes decides ordering, headers, and token budget — HyMem only returns
    the pieces. This keeps prompt assembly out of the memory module.
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
) -> AugmentedContext:
    ctx = AugmentedContext()
    if cfg.user_md_path.exists():
        ctx.user_md = cfg.user_md_path.read_text(encoding="utf-8")
    if cfg.memory_md_path.exists():
        ctx.memory_md = cfg.memory_md_path.read_text(encoding="utf-8")

    ctx.fts_hits = _fts_search(conn, user_message, top_k=cfg.fts_top_k)
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
