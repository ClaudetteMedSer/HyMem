from __future__ import annotations

import logging
import sqlite3

from hymem.dreaming import canonicalize
from hymem.dreaming.chunks import Chunk
from hymem.extraction.llm import LLMClient
from hymem.extraction.markers import Marker, extract_markers
from hymem.extraction.triples import Triple, extract_triples

log = logging.getLogger("hymem.dreaming.phase1")


def process_chunk(
    conn: sqlite3.Connection,
    chunk: Chunk,
    llm: LLMClient,
    *,
    prompt_version: str,
) -> tuple[list[Triple], list[Marker]]:
    """Phase 1 work for a single chunk. Idempotent under (chunk_id, prompt_version).

    Caller wraps this in a transaction. Returns the raw extractions for logging.
    """
    already = conn.execute(
        "SELECT 1 FROM processed_chunks WHERE chunk_id = ? AND prompt_version = ?",
        (chunk.id, prompt_version),
    ).fetchone()
    if already:
        return [], []

    triples, entity_type_hints = extract_triples(llm, chunk.text)
    markers = extract_markers(llm, chunk.text)

    for entity_name, entity_type in entity_type_hints.items():
        entity_canon = canonicalize.resolve(conn, entity_name)
        conn.execute(
            """INSERT OR IGNORE INTO entity_types(entity_canonical, type, confidence, source_chunk_id)
               VALUES (?, ?, 1.0, ?)""",
            (entity_canon, entity_type, chunk.id),
        )

    mentioned: set[str] = set()
    for t in triples:
        subj_canon, obj_canon = _upsert_triple(conn, chunk.id, t)
        mentioned.add(subj_canon)
        mentioned.add(obj_canon)
    if mentioned:
        conn.executemany(
            "INSERT OR IGNORE INTO entity_mentions(chunk_id, entity_canonical) VALUES (?, ?)",
            [(chunk.id, e) for e in mentioned],
        )
    for m in markers:
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
        len(triples),
        len(markers),
    )
    return triples, markers


def _upsert_triple(conn: sqlite3.Connection, chunk_id: str, triple: Triple) -> tuple[str, str]:
    subj_canon = canonicalize.resolve(conn, triple.subject)
    obj_canon = canonicalize.resolve(conn, triple.object)

    # Track surface forms as aliases so future mentions normalize the same way.
    canonicalize.register_alias(conn, triple.subject, subj_canon)
    canonicalize.register_alias(conn, triple.object, obj_canon)

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

    if triple.polarity == 1:
        conn.execute(
            """
            UPDATE knowledge_graph
            SET pos_evidence = pos_evidence + 1,
                last_seen = CURRENT_TIMESTAMP,
                last_reinforced = CURRENT_TIMESTAMP,
                status = CASE WHEN status = 'retracted' THEN 'active' ELSE status END
            WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?
            """,
            (subj_canon, triple.predicate, obj_canon),
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
            WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?
            """,
            (subj_canon, triple.predicate, obj_canon),
        )

    edge_id = conn.execute(
        "SELECT id FROM knowledge_graph WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?",
        (subj_canon, triple.predicate, obj_canon),
    ).fetchone()["id"]

    conn.execute(
        """
        INSERT OR IGNORE INTO kg_evidence(
            edge_id, chunk_id, polarity, surface_subject, surface_object,
            value_text, value_numeric, value_unit, temporal_scope
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (edge_id, chunk_id, triple.polarity, triple.subject, triple.object,
         triple.value_text, triple.value_numeric, triple.value_unit, triple.temporal_scope),
    )

    return subj_canon, obj_canon
