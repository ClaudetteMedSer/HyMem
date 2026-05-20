from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from hymem.config import HyMemConfig
from hymem.dreaming import canonicalize
from hymem.dreaming.chunks import Chunk
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient
from hymem.extraction.markers import Marker, extract_markers
from hymem.extraction.triples import Triple, extract_triples

log = logging.getLogger("hymem.dreaming.phase1")


@dataclass
class ChunkExtraction:
    """Raw phase-1 output: ready to persist, no DB writes performed yet."""
    triples: list[Triple]
    markers: list[Marker]
    entity_type_hints: dict[str, str] = field(default_factory=dict)
    entity_property_hints: dict[str, dict[str, str]] = field(default_factory=dict)


def extract_chunk_results(
    conn: sqlite3.Connection,
    chunk: Chunk,
    llm: LLMClient,
    *,
    prompt_version: str,
    negative_examples: str = "",
) -> ChunkExtraction | None:
    """Run phase-1 LLM extraction for a chunk. Returns None if already processed
    under the same prompt_version. No write transaction held; the LLM call
    runs outside any BEGIN IMMEDIATE so concurrent writers aren't blocked.
    """
    already = conn.execute(
        "SELECT 1 FROM processed_chunks WHERE chunk_id = ? AND prompt_version = ?",
        (chunk.id, prompt_version),
    ).fetchone()
    if already:
        return None

    triples, entity_type_hints, entity_property_hints = extract_triples(
        llm, chunk.text, negative_examples
    )
    markers = extract_markers(llm, chunk.text)
    return ChunkExtraction(
        triples=triples,
        markers=markers,
        entity_type_hints=entity_type_hints,
        entity_property_hints=entity_property_hints,
    )


def persist_chunk_results(
    conn: sqlite3.Connection,
    chunk: Chunk,
    extraction: ChunkExtraction,
    *,
    prompt_version: str,
    cfg: HyMemConfig | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> None:
    """Persist a ChunkExtraction. Caller wraps in core_db.transaction().

    When ``cfg`` is provided, positive evidence is weighted by the role of the
    chunk's first message (speaker-weighted evidence). When ``cfg`` and
    ``embedding_client`` are both provided and ``cfg.triple_dedup_enabled``, a
    brand-new triple is first checked against existing same-predicate edges by
    vector similarity, attaching evidence to a near-duplicate rather than
    minting a sibling edge.
    """
    source_role = _chunk_first_message_role(conn, chunk.id)
    role_weights = cfg.evidence_role_weights if cfg is not None else {}
    pos_weight = role_weights.get(source_role, 1) if source_role else 1
    for entity_name, entity_type in extraction.entity_type_hints.items():
        entity_canon = canonicalize.resolve(conn, entity_name)
        conn.execute(
            """INSERT OR IGNORE INTO entity_types(entity_canonical, type, confidence, source_chunk_id)
               VALUES (?, ?, 1.0, ?)""",
            (entity_canon, entity_type, chunk.id),
        )

    for entity_name, kv in extraction.entity_property_hints.items():
        if not kv:
            continue
        entity_canon = canonicalize.resolve(conn, entity_name)
        for key, value in kv.items():
            conn.execute(
                """INSERT INTO entity_properties(
                       entity_canonical, key, value, source_chunk_id, updated_at
                   ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(entity_canonical, key) DO UPDATE SET
                       value = excluded.value,
                       source_chunk_id = excluded.source_chunk_id,
                       updated_at = CURRENT_TIMESTAMP""",
                (entity_canon, key, value, chunk.id),
            )

    mentioned: set[str] = set()
    for t in extraction.triples:
        subj_canon, obj_canon = _upsert_triple(
            conn, chunk.id, t,
            source_role=source_role,
            pos_weight=pos_weight,
            cfg=cfg,
            embedding_client=embedding_client,
        )
        mentioned.add(subj_canon)
        mentioned.add(obj_canon)
    if mentioned:
        conn.executemany(
            "INSERT OR IGNORE INTO entity_mentions(chunk_id, entity_canonical) VALUES (?, ?)",
            [(chunk.id, e) for e in mentioned],
        )
    for m in extraction.markers:
        conn.execute(
            "INSERT INTO behavioral_markers(kind, statement, chunk_id) VALUES (?, ?, ?)",
            (m.kind, m.statement, chunk.id),
        )

    conn.execute(
        "INSERT OR IGNORE INTO processed_chunks(chunk_id, prompt_version) VALUES (?, ?)",
        (chunk.id, prompt_version),
    )
    log.debug(
        "phase1.chunk chunk_id=%s triples=%d markers=%d",
        chunk.id,
        len(extraction.triples),
        len(extraction.markers),
    )


def _chunk_first_message_role(conn: sqlite3.Connection, chunk_id: str) -> str | None:
    """Role of the chunk's first message (its start_message_id). Used to weight
    evidence by author. Returns None if the chunk or message is gone."""
    row = conn.execute(
        """
        SELECT m.role
        FROM chunks c
        JOIN messages m ON m.id = c.start_message_id
        WHERE c.id = ?
        """,
        (chunk_id,),
    ).fetchone()
    return row["role"] if row else None


def _entities_are_siblings(a: str, b: str, ratio: float) -> bool:
    """Lexical-sibling test for two canonical entity ids: share an underscore
    token, one is a substring of the other, or their difflib ratio clears
    `ratio`. Guards dedup against merging short, embedding-close-but-distinct
    names like `redis` / `redash`."""
    if a == b:
        return True
    a_tokens = {t for t in a.split("_") if len(t) >= 2}
    b_tokens = {t for t in b.split("_") if len(t) >= 2}
    if a_tokens & b_tokens:
        return True
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= ratio


def _find_near_duplicate_edge(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    embedder: EmbeddingClient,
    subj_canon: str,
    predicate: str,
    obj_canon: str,
) -> int | None:
    """Return the id of an existing active edge that is a near-duplicate of the
    candidate triple, or None. Three independent gates must all pass, so a
    merge is conservative:

      1. Predicate — must match exactly (`uses` and `avoids` embed close but
         mean the opposite).
      2. Structure — the existing edge must share the candidate's subject *or*
         object exactly, so only the *other* endpoint varies (a sibling
         canonical like `uv` / `uv_pip`, not a different fact).
      3. Lexical + cosine — the varying endpoint must be a lexical sibling
         (`_entities_are_siblings`) *and* the triple-text cosine must clear
         `cfg.triple_dedup_cosine_threshold`. The lexical gate stops false
         merges of short, embedding-close names the cosine gate alone misses.

    Reads cached vectors from ``edge_embeddings`` (populated by the post-phase3
    embed pass), so dedup fires against edges from prior cycles, not siblings
    minted in the same cycle. The candidate is embedded only when at least one
    structurally + lexically eligible edge exists to compare against.
    """
    rows = conn.execute(
        """
        SELECT kg.id AS edge_id, kg.subject_canonical AS s, kg.object_canonical AS o,
               e.vector_json AS vector_json
        FROM knowledge_graph kg
        JOIN edge_embeddings e
          ON e.edge_text = kg.subject_canonical || ' ' || kg.predicate || ' '
                           || kg.object_canonical
        WHERE kg.status = 'active' AND kg.predicate = ?
          AND (kg.subject_canonical = ? OR kg.object_canonical = ?)
        ORDER BY kg.last_seen DESC
        LIMIT ?
        """,
        (predicate, subj_canon, obj_canon, cfg.embedding_max_scan),
    ).fetchall()

    eligible = []
    for r in rows:
        same_subj = r["s"] == subj_canon
        same_obj = r["o"] == obj_canon
        if same_subj and same_obj:
            continue  # exact edge (shouldn't reach here — caller dedups new only)
        # Exactly one endpoint varies; check that the varying one is a sibling.
        if same_subj:
            varying_ok = _entities_are_siblings(
                obj_canon, r["o"], cfg.triple_dedup_lexical_ratio
            )
        else:
            varying_ok = _entities_are_siblings(
                subj_canon, r["s"], cfg.triple_dedup_lexical_ratio
            )
        if varying_ok:
            eligible.append(r)

    if not eligible:
        return None

    candidate_text = f"{subj_canon} {predicate} {obj_canon}"
    qvec = embedder.embed([candidate_text])[0]
    qnorm = math.sqrt(sum(x * x for x in qvec)) or 1.0
    best_id: int | None = None
    best_sim = -1.0
    for r in eligible:
        try:
            vec = json.loads(r["vector_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if len(vec) != len(qvec):
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
        sim = dot / (qnorm * vnorm)
        if sim > best_sim:
            best_sim = sim
            best_id = r["edge_id"]

    if best_id is not None and best_sim >= cfg.triple_dedup_cosine_threshold:
        return best_id
    return None


def _upsert_triple(
    conn: sqlite3.Connection,
    chunk_id: str,
    triple: Triple,
    *,
    source_role: str | None = None,
    pos_weight: int = 1,
    cfg: HyMemConfig | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> tuple[str, str]:
    subj_canon = canonicalize.resolve(conn, triple.subject)
    obj_canon = canonicalize.resolve(conn, triple.object)

    # Track surface forms as aliases so future mentions normalize the same way.
    canonicalize.register_alias(conn, triple.subject, subj_canon)
    canonicalize.register_alias(conn, triple.object, obj_canon)

    existing = conn.execute(
        "SELECT id FROM knowledge_graph "
        "WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?",
        (subj_canon, triple.predicate, obj_canon),
    ).fetchone()

    if existing is not None:
        edge_id = existing["id"]
    else:
        edge_id = None
        # Semantic dedup: attach to a near-duplicate sibling edge if one exists,
        # rather than spawning yet another canonical variant. Best-effort — a
        # failure here must not lose the triple.
        if cfg is not None and embedding_client is not None and cfg.triple_dedup_enabled:
            try:
                edge_id = _find_near_duplicate_edge(
                    conn, cfg, embedding_client, subj_canon, triple.predicate, obj_canon
                )
            except Exception:
                log.exception("phase1.dedup_failure subj=%s pred=%s obj=%s",
                              subj_canon, triple.predicate, obj_canon)
                edge_id = None
            if edge_id is not None:
                log.debug(
                    "phase1.triple_deduped candidate=%s %s %s -> edge_id=%d",
                    subj_canon, triple.predicate, obj_canon, edge_id,
                )
        if edge_id is None:
            conn.execute(
                """
                INSERT INTO knowledge_graph(
                    subject_canonical, predicate, object_canonical,
                    pos_evidence, neg_evidence, last_reinforced
                )
                VALUES (?, ?, ?, 0, 0, CURRENT_TIMESTAMP)
                ON CONFLICT(subject_canonical, predicate, object_canonical) DO NOTHING
                """,
                (subj_canon, triple.predicate, obj_canon),
            )
            edge_id = conn.execute(
                "SELECT id FROM knowledge_graph "
                "WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?",
                (subj_canon, triple.predicate, obj_canon),
            ).fetchone()["id"]

    if triple.polarity == 1:
        # pos_weight is the speaker weight for this chunk (>= 1).
        conn.execute(
            """
            UPDATE knowledge_graph
            SET pos_evidence = pos_evidence + ?,
                last_seen = CURRENT_TIMESTAMP,
                last_reinforced = CURRENT_TIMESTAMP,
                status = CASE WHEN status = 'retracted' THEN 'active' ELSE status END
            WHERE id = ?
            """,
            (pos_weight, edge_id),
        )
    else:
        # last_reinforced is intentionally not updated for negative polarity: a
        # contradiction does not "refresh" the edge, so phase3 decay still fires
        # for edges that have only ever seen negative evidence.
        conn.execute(
            """
            UPDATE knowledge_graph
            SET neg_evidence = neg_evidence + 1,
                last_seen = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (edge_id,),
        )

    conn.execute(
        """
        INSERT OR IGNORE INTO kg_evidence(
            edge_id, chunk_id, polarity, surface_subject, surface_object,
            value_text, value_numeric, value_unit, temporal_scope, source_role
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (edge_id, chunk_id, triple.polarity, triple.subject, triple.object,
         triple.value_text, triple.value_numeric, triple.value_unit,
         triple.temporal_scope, source_role),
    )

    return subj_canon, obj_canon
