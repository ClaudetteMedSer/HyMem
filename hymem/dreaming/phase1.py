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
from hymem.extraction.chunk import extract_chunk
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient
from hymem.extraction.markers import Marker
from hymem.extraction.triples import Triple

log = logging.getLogger("hymem.dreaming.phase1")


@dataclass
class ChunkExtraction:
    """Raw phase-1 output: ready to persist, no DB writes performed yet.

    ``failed`` carries :class:`~hymem.extraction.chunk.ChunkResult.failed`
    through to the persist step, which refuses to mark a failed chunk done.
    """
    triples: list[Triple]
    markers: list[Marker]
    entity_type_hints: dict[str, str] = field(default_factory=dict)
    entity_property_hints: dict[str, dict[str, str]] = field(default_factory=dict)
    failed: bool = False


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

    result = extract_chunk(llm, chunk.text, negative_examples)
    return ChunkExtraction(
        triples=result.triples,
        markers=result.markers,
        entity_type_hints=result.entity_type_hints,
        entity_property_hints=result.entity_property_hints,
        failed=result.failed,
    )


def prepare_dedup_vectors(
    conn: sqlite3.Connection,
    extraction: ChunkExtraction,
    cfg: HyMemConfig | None,
    embedding_client: EmbeddingClient | None,
) -> dict[str, list[float]]:
    """Embed the dedup candidate texts for a chunk *outside* any write lock.

    Returns ``{candidate_text: vector}`` keyed on the exact triple text
    ``f"{subj_canon} {predicate} {obj_canon}"`` that ``_upsert_triple`` will
    look up. The network ``embed()`` call must not run inside a
    ``BEGIN IMMEDIATE`` transaction; this function performs only DB *reads*
    (canonical resolution + existence/eligibility lookups) and the embed, so
    callers run it before opening the persist transaction.

    Embedding rule (chosen for same-wave collapse correctness, point 4 of the
    same-wave initiative):

      * Only triples with NO exact existing edge are candidates (the same
        "new only" gate as the persist path).
      * EVERY remaining new-triple candidate text is embedded — not just those
        with a structurally+lexically eligible *existing* edge.

    Why broaden from the previous "eligible-existing only" rule: same-wave
    sibling collapse (`_find_near_duplicate_in_cycle`) compares a new candidate
    against edges minted EARLIER IN THIS SAME CYCLE, which by definition are not
    yet in ``edge_embeddings``. The first variant in a sibling group therefore
    has no eligible *existing* edge, so under the old rule it would not be
    embedded and could never become a same-wave target for the second variant.
    Embedding every new candidate guarantees both the cross-cycle path
    (``_find_near_duplicate_edge``) and the same-wave path have a vector to use.

    Cost tradeoff: this issues one embed per distinct new-triple text per chunk,
    rather than only for texts with an eligible existing edge — i.e. strictly
    more embed calls than before. They remain batched into a single ``embed()``
    per chunk and run OUTSIDE the write lock, and deduplicating the graph is the
    whole point, so the extra embeds are an accepted cost.

    Returns ``{}`` when dedup is disabled or no embedding client is wired.
    Pure best-effort callers may additionally wrap this in try/except.
    """
    if cfg is None or embedding_client is None or not cfg.triple_dedup_enabled:
        return {}

    # Deduplicate candidate texts so a chunk repeating a triple embeds once.
    texts: list[str] = []
    seen: set[str] = set()
    for t in extraction.triples:
        subj_canon = canonicalize.resolve(conn, t.subject)
        obj_canon = canonicalize.resolve(conn, t.object)
        existing = conn.execute(
            "SELECT 1 FROM knowledge_graph "
            "WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?",
            (subj_canon, t.predicate, obj_canon),
        ).fetchone()
        if existing is not None:
            continue  # exact edge already present — not a dedup candidate
        candidate_text = f"{subj_canon} {t.predicate} {obj_canon}"
        if candidate_text not in seen:
            seen.add(candidate_text)
            texts.append(candidate_text)

    if not texts:
        return {}
    vectors = embedding_client.embed(texts)
    return dict(zip(texts, vectors))


def _record_failed_attempt(
    conn: sqlite3.Connection,
    chunk: Chunk,
    *,
    prompt_version: str,
    cfg: HyMemConfig | None,
) -> None:
    """Count a held failure, and give up once the bound is reached.

    The chunk stays unmarked (and so retried) until `attempts` reaches
    `chunk_extraction_max_attempts`, at which point it is marked done so it
    stops consuming a dream_budget slot on every cycle. Giving up is logged at
    WARNING: it is a real content loss, and the whole point of the held-retry
    change is that losses must be audible rather than silent. `max_attempts=0`
    disables the bound and retries forever.
    """
    row = conn.execute(
        "SELECT attempts FROM chunk_extraction_attempts "
        "WHERE chunk_id = ? AND prompt_version = ?",
        (chunk.id, prompt_version),
    ).fetchone()
    attempts = (row[0] if row else 0) + 1
    conn.execute(
        """INSERT INTO chunk_extraction_attempts(chunk_id, prompt_version, attempts,
                                                 last_failure_at)
           VALUES (?, ?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(chunk_id, prompt_version)
           DO UPDATE SET attempts = excluded.attempts,
                         last_failure_at = excluded.last_failure_at""",
        (chunk.id, prompt_version, attempts),
    )

    max_attempts = cfg.chunk_extraction_max_attempts if cfg is not None else 0
    if max_attempts and attempts >= max_attempts:
        conn.execute(
            "INSERT OR IGNORE INTO processed_chunks(chunk_id, prompt_version) VALUES (?, ?)",
            (chunk.id, prompt_version),
        )
        log.warning(
            "phase1.extraction_abandoned chunk_id=%s attempts=%d "
            "action=marked_done content_lost=1",
            chunk.id, attempts,
        )


def persist_chunk_results(
    conn: sqlite3.Connection,
    chunk: Chunk,
    extraction: ChunkExtraction,
    *,
    prompt_version: str,
    cfg: HyMemConfig | None = None,
    embedding_client: EmbeddingClient | None = None,
    dedup_vectors: dict[str, list[float]] | None = None,
    in_cycle_edges: list[_InCycleEdge] | None = None,
) -> None:
    """Persist a ChunkExtraction. Caller wraps in core_db.transaction().

    When ``cfg`` is provided, positive evidence is weighted by the role of the
    chunk's first message (speaker-weighted evidence). When ``cfg`` is provided,
    ``cfg.triple_dedup_enabled`` is set, and ``dedup_vectors`` carries a
    precomputed vector for a brand-new triple's candidate text, that triple is
    first checked against existing same-predicate edges by vector similarity,
    attaching evidence to a near-duplicate rather than minting a sibling edge.

    The candidate embedding is precomputed by :func:`prepare_dedup_vectors`
    *outside* this write transaction; ``persist_chunk_results`` never issues a
    network ``embed()`` call (``embedding_client`` is accepted only for
    signature compatibility and is no longer used here).

    ``in_cycle_edges`` is the shared same-wave pool (see
    :func:`new_in_cycle_pool`): a list of edges minted earlier in the current
    dream cycle, threaded by the runner across every chunk. A brand-new triple
    that finds no prior-cycle near-duplicate is also checked against this pool,
    so phrasal-variant siblings appearing in the SAME cycle (even in different
    chunks) collapse onto the first-minted edge instead of each minting its own.
    Pass ``None`` to disable same-wave collapse (cross-cycle dedup is unaffected).
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
            dedup_vectors=dedup_vectors,
            in_cycle_edges=in_cycle_edges,
        )
        mentioned.add(subj_canon)
        mentioned.add(obj_canon)
    if mentioned:
        conn.executemany(
            "INSERT OR IGNORE INTO entity_mentions(chunk_id, entity_canonical) VALUES (?, ?)",
            [(chunk.id, e) for e in mentioned],
        )
    for m in extraction.markers:
        # Write idempotence: re-extracting a chunk re-attaches its markers
        # without duplicating them, the way UNIQUE(edge_id, chunk_id, polarity)
        # already protects kg_evidence. Enforced here rather than by a unique
        # index because deduping an existing store to build that index is not
        # legacy-safe (see migration 028). Dreams hold a lock, so the
        # check-then-insert is not racing another writer.
        conn.execute(
            """INSERT INTO behavioral_markers(kind, statement, chunk_id)
               SELECT ?, ?, ?
               WHERE NOT EXISTS (
                   SELECT 1 FROM behavioral_markers
                   WHERE chunk_id = ? AND kind = ? AND statement = ?
               )""",
            (m.kind, m.statement, chunk.id, chunk.id, m.kind, m.statement),
        )

    # A FAILED extraction is never marked done. `processed_chunks` is the
    # one-shot gate — a row here means no dream will ever look at this chunk
    # again under this prompt version — so marking a chunk whose extraction did
    # not complete converts a transient provider hiccup into a permanent hole,
    # indistinguishable in the DB from a chunk that genuinely held nothing.
    # Holding the mark instead retries on the next dream, which is what the
    # digest (v24 watermark), facts (v26 watermark) and fusion paths all do.
    # A clean parse that yielded nothing IS marked: that is the real floor.
    if extraction.failed:
        _record_failed_attempt(conn, chunk, prompt_version=prompt_version, cfg=cfg)
    else:
        conn.execute(
            "INSERT OR IGNORE INTO processed_chunks(chunk_id, prompt_version) VALUES (?, ?)",
            (chunk.id, prompt_version),
        )
        # Consecutive-failure count, so a chunk that heals starts fresh.
        conn.execute(
            "DELETE FROM chunk_extraction_attempts "
            "WHERE chunk_id = ? AND prompt_version = ?",
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


def _structural_lexical_match(
    cfg: HyMemConfig,
    cand_subj: str,
    predicate: str,
    cand_obj: str,
    other_subj: str,
    other_pred: str,
    other_obj: str,
) -> bool:
    """Shared structural + lexical dedup gate for a candidate triple against
    one other triple/edge. Used by BOTH cross-cycle dedup (via
    ``_eligible_dedup_edges``) and same-wave dedup (via
    ``_find_near_duplicate_in_cycle``) so the gating is defined once and the
    two paths can never drift apart:

      * Predicate must match exactly.
      * Exactly one endpoint must be shared exactly (the other varies).
      * The varying endpoint must be a lexical sibling
        (``_entities_are_siblings``).

    The cosine gate is applied separately by each caller using its own vector
    pool; this function never sees vectors.
    """
    if other_pred != predicate:
        return False
    same_subj = other_subj == cand_subj
    same_obj = other_obj == cand_obj
    if same_subj and same_obj:
        return False  # exact edge — not a sibling (caller dedups new-only)
    if not (same_subj or same_obj):
        return False  # both endpoints differ — different fact, not a sibling
    if same_subj:
        return _entities_are_siblings(
            cand_obj, other_obj, cfg.triple_dedup_lexical_ratio
        )
    return _entities_are_siblings(
        cand_subj, other_subj, cfg.triple_dedup_lexical_ratio
    )


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


def _eligible_dedup_edges(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    subj_canon: str,
    predicate: str,
    obj_canon: str,
) -> list[sqlite3.Row]:
    """Existing active edges that pass the structural + lexical dedup gates for
    the candidate triple, with their cached vector_json attached.

    Two of the three dedup gates live here (the third — cosine — needs the
    candidate vector and stays in ``_find_near_duplicate_edge``):

      1. Predicate — must match exactly (`uses` and `avoids` embed close but
         mean the opposite).
      2. Structure — the existing edge must share the candidate's subject *or*
         object exactly, so only the *other* endpoint varies (a sibling
         canonical like `uv` / `uv_pip`, not a different fact).
      3. Lexical — the varying endpoint must be a lexical sibling
         (`_entities_are_siblings`), stopping false merges of short,
         embedding-close names the cosine gate alone misses.

    Reads cached vectors from ``edge_embeddings`` (populated by the post-phase3
    embed pass), so dedup fires against edges from prior cycles, not siblings
    minted in the same cycle. Shared by ``prepare_dedup_vectors`` (to decide
    whether a candidate is worth embedding at all) and
    ``_find_near_duplicate_edge`` (to score), so the gating is defined once.
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

    eligible: list[sqlite3.Row] = []
    for r in rows:
        # SQL already filtered predicate + one-endpoint-shared; reuse the shared
        # gate so the lexical-sibling check is defined identically to same-wave.
        if _structural_lexical_match(
            cfg, subj_canon, predicate, obj_canon, r["s"], predicate, r["o"]
        ):
            eligible.append(r)
    return eligible


def _find_near_duplicate_edge(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    candidate_vector: list[float] | None,
    subj_canon: str,
    predicate: str,
    obj_canon: str,
) -> int | None:
    """Return the id of an existing active edge that is a near-duplicate of the
    candidate triple, or None. Three independent gates must all pass, so a
    merge is conservative (predicate + structure + lexical are checked by
    ``_eligible_dedup_edges``; the cosine gate is applied here).

    ``candidate_vector`` is the candidate triple-text embedding precomputed by
    :func:`prepare_dedup_vectors` *outside* the write transaction. When it is
    missing/None (dedup disabled, no embedding client, or prepare failed),
    this returns None — a safe no-dedup fallback. No network ``embed()`` call
    is issued here; only the in-memory cosine over cached ``edge_embeddings``
    vectors runs under the lock.
    """
    if candidate_vector is None:
        return None

    eligible = _eligible_dedup_edges(conn, cfg, subj_canon, predicate, obj_canon)
    if not eligible:
        return None

    qvec = candidate_vector
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


@dataclass
class _InCycleEdge:
    """An edge minted earlier in THIS dream cycle, with the vector used at mint
    time, so later same-cycle sibling candidates can collapse onto it without
    waiting for the post-phase3 ``edge_embeddings`` write."""
    subject: str
    predicate: str
    object: str
    vector: list[float]
    edge_id: int


def new_in_cycle_pool() -> list[_InCycleEdge]:
    """Create the shared same-wave comparison pool for one dream cycle.

    The runner builds this once per ``run_dreaming`` call and threads the SAME
    list through every ``persist_chunk_results`` invocation, so an edge minted
    in an earlier chunk is a valid same-wave target for a candidate in a later
    chunk of the same cycle.
    """
    return []


def _find_near_duplicate_in_cycle(
    cfg: HyMemConfig,
    candidate_vector: list[float] | None,
    subj_canon: str,
    predicate: str,
    obj_canon: str,
    in_cycle_edges: list[_InCycleEdge],
) -> int | None:
    """Same-wave analogue of :func:`_find_near_duplicate_edge`.

    The ONLY difference from the cross-cycle path is the candidate pool: this
    scans edges minted earlier in the SAME dream cycle (held in memory, vectors
    from their own prepare step) instead of ``edge_embeddings``. The three gates
    — predicate exact, one-endpoint-shared structure, lexical sibling
    (``_structural_lexical_match``), plus cosine ≥ threshold — are identical, so
    same-wave can never merge anything cross-cycle dedup would reject.

    No DB access and no network I/O: pure in-memory cosine, safe under the lock.
    """
    if candidate_vector is None or not in_cycle_edges:
        return None

    qvec = candidate_vector
    qnorm = math.sqrt(sum(x * x for x in qvec)) or 1.0
    best_id: int | None = None
    best_sim = -1.0
    for e in in_cycle_edges:
        if not _structural_lexical_match(
            cfg, subj_canon, predicate, obj_canon,
            e.subject, e.predicate, e.object,
        ):
            continue
        vec = e.vector
        if len(vec) != len(qvec):
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
        sim = dot / (qnorm * vnorm)
        if sim > best_sim:
            best_sim = sim
            best_id = e.edge_id

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
    dedup_vectors: dict[str, list[float]] | None = None,
    in_cycle_edges: list[_InCycleEdge] | None = None,
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
        candidate_vector: list[float] | None = None
        # Semantic dedup: attach to a near-duplicate sibling edge if one exists,
        # rather than spawning yet another canonical variant. The candidate
        # vector was embedded outside this write transaction (see
        # prepare_dedup_vectors); here only the in-memory cosine runs.
        # Best-effort — a failure here must not lose the triple.
        if cfg is not None and dedup_vectors and cfg.triple_dedup_enabled:
            candidate_text = f"{subj_canon} {triple.predicate} {obj_canon}"
            candidate_vector = dedup_vectors.get(candidate_text)
            try:
                edge_id = _find_near_duplicate_edge(
                    conn, cfg, candidate_vector, subj_canon, triple.predicate, obj_canon
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
            # Same-wave collapse: no prior-cycle edge matched, so check siblings
            # minted earlier IN THIS cycle (cross-chunk too — the pool is shared
            # by the runner across all chunks). Same gates as cross-cycle dedup.
            if edge_id is None and in_cycle_edges is not None:
                try:
                    edge_id = _find_near_duplicate_in_cycle(
                        cfg, candidate_vector, subj_canon, triple.predicate,
                        obj_canon, in_cycle_edges,
                    )
                except Exception:
                    log.exception(
                        "phase1.samewave_dedup_failure subj=%s pred=%s obj=%s",
                        subj_canon, triple.predicate, obj_canon,
                    )
                    edge_id = None
                if edge_id is not None:
                    log.debug(
                        "phase1.triple_deduped_samewave candidate=%s %s %s -> edge_id=%d",
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
            # Register this freshly minted edge in the same-wave pool so later
            # sibling candidates (this chunk or a later one) can collapse onto
            # it. Only register when we have a vector to compare against — an
            # edge with no recorded vector simply can't be a same-wave target.
            if (
                in_cycle_edges is not None
                and candidate_vector is not None
                and cfg is not None
                and cfg.triple_dedup_enabled
            ):
                in_cycle_edges.append(
                    _InCycleEdge(
                        subject=subj_canon,
                        predicate=triple.predicate,
                        object=obj_canon,
                        vector=candidate_vector,
                        edge_id=edge_id,
                    )
                )

    if triple.polarity == 1:
        # pos_weight is the speaker weight for this chunk (>= 1).
        # Re-asserting a retracted edge reopens its validity interval: the
        # status flip to 'active' is useless without also clearing invalid_at,
        # since every reader that consumes active edges filters on
        # `invalid_at IS NULL` (state_anchor.py, aggregate.py). Without the
        # clear, an asserted → contradicted → re-asserted edge stays invisible
        # in the exact state where it should be most trusted. The clear is
        # unconditional (not gated on status='retracted') so a row that
        # already died in the corrupted state (active + stale invalid_at)
        # self-heals on its next positive mention. Mirrors the rules table's
        # re-assert contract (rules.py: status='active', invalid_at=NULL).
        conn.execute(
            """
            UPDATE knowledge_graph
            SET pos_evidence = pos_evidence + ?,
                last_seen = CURRENT_TIMESTAMP,
                last_reinforced = CURRENT_TIMESTAMP,
                status = CASE WHEN status = 'retracted' THEN 'active' ELSE status END,
                invalid_at = NULL
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
