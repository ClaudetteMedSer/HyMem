from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher

from hymem.config import HyMemConfig
from hymem.dreaming import canonicalize
from hymem.dreaming import evidence
from hymem.dreaming.chunks import Chunk
from hymem.dreaming.lossless import (
    CoveredMessage,
    covered_messages_after,
    validate_message_coverage_artifact,
)
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
from hymem.extraction.chunk import extract_chunk
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient
from hymem.extraction.markers import Marker
from hymem.extraction.triples import Triple
from hymem.core.graph import graph_clock_order_sql, live_edge_predicate
from hymem.core.time import normalize_iso_timestamp

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
    claim_sources: dict[int, CoveredMessage] = field(default_factory=dict)
    source_validated: bool = False


def _is_exact_published_replay(
    conn: sqlite3.Connection,
    chunk: Chunk,
    extraction: ChunkExtraction,
    *,
    prompt_version: str,
    cfg: HyMemConfig | None,
    dedup_vectors: dict[str, list[float]] | None,
    in_cycle_edges: list[_InCycleEdge] | None,
) -> bool:
    """Detect a pure claim replay before replacing its ledger projection.

    Global prompt authority can leave this chunk's observation pointing at a
    retired interpretation. Deleting and recreating that same observation
    would briefly make it current and force append-only reconciliation to mint
    two more revisions on every replay. A published result with the same exact
    source/surface semantics is already durable and needs no write at all.

    The claim outcome intentionally excludes marker/entity projections. Those
    writes are independently idempotent; skipping an already-published exact
    prompt replay also avoids rewriting their transaction clocks.
    """
    outcome = conn.execute(
        "SELECT outcome.prompt_version,outcome.prompt_generation,"
        "outcome.result_hash,outcome.succeeded_at,chunk.created_at "
        "FROM kg_claim_extraction_outcomes outcome "
        "JOIN chunks chunk ON chunk.id=outcome.chunk_id "
        "WHERE outcome.chunk_id=?",
        (chunk.id,),
    ).fetchone()
    if (
        outcome is None
        or outcome["prompt_version"] != prompt_version
        or int(outcome["prompt_generation"])
        != evidence.prompt_generation(prompt_version)
        or outcome["result_hash"]
        != evidence.claim_observation_result_hash(conn, chunk.id)
        or not conn.execute(
            "SELECT hymem_timestamp_at_or_before(?,?)",
            (outcome["created_at"], outcome["succeeded_at"]),
        ).fetchone()[0]
        or not conn.execute(
            "SELECT hymem_timestamp_at_or_before(?,"
            "strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds'))",
            (outcome["succeeded_at"],),
        ).fetchone()[0]
    ):
        return False
    authority = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN
                 ev.provenance_status='canonical'
                 AND ev.edge_id=observation.edge_id
                 AND ev.source_session_id=observation.source_session_id
                 AND ev.source_message_id=observation.source_message_id
                 AND ev.evidence_kind=observation.evidence_kind
                 AND ev.polarity=observation.polarity
                 AND ev.interpretation_key=observation.interpretation_key
                 AND hymem_event_clock_is_valid(
                       ev.source_event_at,ev.extracted_at
                     )=1
                 AND hymem_timestamp_at_or_before(
                       ev.extracted_at,ev.published_at
                     )=1
                 AND hymem_timestamp_at_or_before(
                       ev.published_at,outcome.succeeded_at
                     )=1
                 AND hymem_timestamp_at_or_before(
                       outcome.succeeded_at,
                       strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
                     )=1
                 AND hymem_timestamp_at_or_before(
                       ev.extracted_at,observation.observed_at
                     )=1
                 AND hymem_timestamp_gap_within(
                       observation.observed_at,outcome.succeeded_at,300
                     )=1
                 AND (
                   ev.superseded_at IS NULL
                   OR hymem_timestamp_at_or_before(
                        ev.published_at,ev.superseded_at
                      )=1
                 )
                 AND (
                   ev.polarity=-1 OR EXISTS (
                     SELECT 1 FROM kg_edge_lifecycle lifecycle
                     WHERE lifecycle.source_evidence_id=ev.id
                       AND lifecycle.edge_id=ev.edge_id
                       AND lifecycle.event_kind='claim_assertion'
                       AND lifecycle.direction=1
                       AND lifecycle.event_at=ev.source_event_at
                       AND hymem_timestamp_at_or_before(
                             ev.extracted_at,lifecycle.created_at
                           )=1
                       AND hymem_timestamp_at_or_before(
                             lifecycle.created_at,ev.published_at
                           )=1
                   )
                 )
               THEN 1 ELSE 0 END) AS healthy
        FROM kg_claim_observations observation
        JOIN kg_evidence ev ON ev.id=observation.evidence_id
        JOIN kg_claim_extraction_outcomes outcome
          ON outcome.chunk_id=observation.chunk_id
         AND outcome.prompt_version=observation.prompt_version
         AND outcome.prompt_generation=observation.prompt_generation
        WHERE observation.chunk_id=?
        """,
        (chunk.id,),
    ).fetchone()
    if int(authority["total"] or 0) != int(authority["healthy"] or 0):
        return False
    role_weights = cfg.evidence_role_weights if cfg is not None else {}
    semantic_rows: list[tuple[object, ...]] = []
    for triple in extraction.triples:
        source = extraction.claim_sources.get(int(triple.source_message_id))
        if source is None:
            return False
        subject = canonicalize.resolve(conn, triple.subject)
        object_ = canonicalize.resolve(conn, triple.object)
        edge = conn.execute(
            "SELECT id,subject_canonical,predicate,object_canonical "
            "FROM knowledge_graph WHERE subject_canonical=? AND predicate=? "
            "AND object_canonical=?",
            (subject, triple.predicate, object_),
        ).fetchone()
        if (
            edge is None
            and cfg is not None
            and cfg.triple_dedup_enabled
            and dedup_vectors
        ):
            candidate_text = f"{subject} {triple.predicate} {object_}"
            vector = dedup_vectors.get(candidate_text)
            edge_id = _find_near_duplicate_edge(
                conn, cfg, vector, subject, triple.predicate, object_,
                model=getattr(dedup_vectors, "model", None),
                dim=getattr(dedup_vectors, "dim", None),
            )
            if edge_id is None and in_cycle_edges is not None:
                edge_id = _find_near_duplicate_in_cycle(
                    cfg, vector, subject, triple.predicate, object_, in_cycle_edges,
                    model=getattr(dedup_vectors, "model", None),
                    dim=getattr(dedup_vectors, "dim", None),
                )
            if edge_id is not None:
                edge = conn.execute(
                    "SELECT id,subject_canonical,predicate,object_canonical "
                    "FROM knowledge_graph WHERE id=?", (edge_id,),
                ).fetchone()
        edge_tuple = (
            (edge["subject_canonical"], edge["predicate"], edge["object_canonical"])
            if edge is not None else (subject, triple.predicate, object_)
        )
        weight = role_weights.get(source.role, 1) if cfg is not None else 1
        weight_source = (
            f"configured_role:{source.role}"
            if cfg is not None else "default_weight:1"
        )
        semantic_rows.append((
            *edge_tuple,
            source.session_id,
            source.message_id,
            "extraction",
            int(triple.polarity),
            evidence._interpretation_key(
                polarity=int(triple.polarity),
                evidence_weight=weight,
                weight_source=weight_source,
                source_role=source.role,
                surface_subject=triple.subject,
                surface_object=triple.object,
                value_text=triple.value_text,
                value_numeric=triple.value_numeric,
                value_unit=triple.value_unit,
                temporal_scope=triple.temporal_scope,
            ),
        ))
    return evidence.claim_result_hash(semantic_rows) == outcome["result_hash"]


def _persist_replay_auxiliary(
    conn: sqlite3.Connection, chunk: Chunk, extraction: ChunkExtraction
) -> None:
    """Heal idempotent non-claim projections without touching claim clocks."""
    for entity_name, entity_type in extraction.entity_type_hints.items():
        entity = canonicalize.resolve(conn, entity_name)
        conn.execute(
            "INSERT OR IGNORE INTO entity_types("
            "entity_canonical,type,confidence,source_chunk_id) "
            "VALUES (?,?,1.0,?)",
            (entity, entity_type, chunk.id),
        )
    for entity_name, values in extraction.entity_property_hints.items():
        entity = canonicalize.resolve(conn, entity_name)
        for key, value in values.items():
            conn.execute(
                """
                INSERT INTO entity_properties(
                    entity_canonical,key,value,source_chunk_id,updated_at
                ) VALUES (?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(entity_canonical,key) DO UPDATE SET
                    value=excluded.value,
                    source_chunk_id=excluded.source_chunk_id,
                    updated_at=CURRENT_TIMESTAMP
                WHERE entity_properties.value IS NOT excluded.value
                   OR entity_properties.source_chunk_id IS NOT
                      excluded.source_chunk_id
                """,
                (entity, key, value, chunk.id),
            )
    mentioned = {
        canonicalize.resolve(conn, name)
        for triple in extraction.triples
        for name in (triple.subject, triple.object)
    }
    conn.executemany(
        "INSERT OR IGNORE INTO entity_mentions(chunk_id,entity_canonical) "
        "VALUES (?,?)",
        [(chunk.id, entity) for entity in sorted(mentioned)],
    )
    for marker in extraction.markers:
        conn.execute(
            """
            INSERT INTO behavioral_markers(kind,statement,chunk_id)
            SELECT ?,?,? WHERE NOT EXISTS (
                SELECT 1 FROM behavioral_markers
                WHERE chunk_id=? AND kind=? AND statement=?
            )
            """,
            (
                marker.kind, marker.statement, chunk.id,
                chunk.id, marker.kind, marker.statement,
            ),
        )
    conn.execute(
        "INSERT OR IGNORE INTO processed_chunks(chunk_id,prompt_version) "
        "VALUES (?,?)",
        (chunk.id, conn.execute(
            "SELECT prompt_version FROM kg_claim_extraction_outcomes "
            "WHERE chunk_id=?", (chunk.id,),
        ).fetchone()[0]),
    )
    conn.execute(
        "DELETE FROM chunk_extraction_attempts WHERE chunk_id=? AND "
        "prompt_version=(SELECT prompt_version FROM "
        "kg_claim_extraction_outcomes WHERE chunk_id=?)",
        (chunk.id, chunk.id),
    )


def _claim_source_record(source: CoveredMessage) -> str:
    """Prompt record whose stable id is also an immutable coverage citation."""
    return json.dumps(
        {
            "content": source.content,
            "source_created_at": source.source_created_at,
            "source_message_id": source.message_id,
            "source_peer_id": source.source_peer_id,
            "source_record_version": "hymem-claim-source-v2",
            "source_role": source.role,
            "source_session_id": source.session_id,
            "source_workspace_id": source.source_workspace_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _claim_sources_for_chunk(
    conn: sqlite3.Connection, chunk: Chunk
) -> list[CoveredMessage]:
    """Rebuild exact tagged input from v38 artifacts, never chunk prose."""
    if chunk.start_message_id < 1 or chunk.end_message_id < chunk.start_message_id:
        return []
    header = conn.execute(
        "SELECT source_manifest_version, source_manifest_count FROM chunks "
        "WHERE id = ? AND session_id = ? AND chunk_kind = 'extraction'",
        (chunk.id, chunk.session_id),
    ).fetchone()
    if (
        header is None
        or header["source_manifest_version"] != "claim-source-manifest-v1"
        or not isinstance(header["source_manifest_count"], int)
        or int(header["source_manifest_count"]) < 1
    ):
        return []
    rows = conn.execute(
        f"""
        SELECT ordinal, source_message_id, source_session_id,
               source_coverage_chunk_id, source_coverage_version
        FROM chunk_message_sources
        WHERE chunk_id = ? ORDER BY ordinal
        """,
        (chunk.id,),
    ).fetchall()
    if len(rows) != int(header["source_manifest_count"]) or [
        int(row["ordinal"]) for row in rows
    ] != list(range(len(rows))):
        return []
    sources: list[CoveredMessage] = []
    for row in rows:
        if (
            row["source_session_id"] != chunk.session_id
            or row["source_coverage_version"] != LOSSLESS_COVERAGE_VERSION
        ):
            return []
        proof = validate_message_coverage_artifact(
            conn,
            message_id=int(row["source_message_id"]),
            chunk_id=row["source_coverage_chunk_id"],
            coverage_version=row["source_coverage_version"],
        )
        sources.append(proof)
    if (
        not sources
        or sources[0].message_id != chunk.start_message_id
        or sources[-1].message_id != chunk.end_message_id
        or any(source.session_id != chunk.session_id for source in sources)
        or any(
            earlier.message_id >= later.message_id
            for earlier, later in zip(sources, sources[1:])
        )
    ):
        return []
    return sources


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

    try:
        sources = _claim_sources_for_chunk(conn, chunk)
    except (RuntimeError, TypeError, ValueError, sqlite3.DatabaseError):
        log.exception(
            "phase1.source_coverage_corrupt chunk_id=%s action=held", chunk.id
        )
        sources = []
    if not sources:
        log.warning(
            "phase1.source_coverage_missing chunk_id=%s action=held", chunk.id
        )
        return ChunkExtraction(triples=[], markers=[], failed=True)
    source_records = tuple(
        (source.message_id, _claim_source_record(source)) for source in sources
    )
    result = extract_chunk(
        llm, chunk.text, negative_examples, source_records=source_records
    )
    if not result.failed:
        polarities: dict[tuple[str, str, str, int], int] = {}
        unique: dict[tuple[str, str, str, int, int], Triple] = {}
        canonical_conflict = False
        for triple in result.triples:
            claim = (
                canonicalize.resolve(conn, triple.subject),
                triple.predicate,
                canonicalize.resolve(conn, triple.object),
                int(triple.source_message_id),
            )
            prior = polarities.get(claim)
            if prior is not None and prior != triple.polarity:
                canonical_conflict = True
                break
            polarities[claim] = triple.polarity
            identity = (*claim[:3], triple.polarity, claim[3])
            current = unique.get(identity)
            if current is None or repr(triple) < repr(current):
                unique[identity] = triple
        if canonical_conflict:
            log.warning(
                "phase1.canonical_response_conflict chunk_id=%s action=held",
                chunk.id,
            )
            result.failed = True
        else:
            result.triples = list(unique.values())
    return ChunkExtraction(
        triples=result.triples,
        markers=result.markers,
        entity_type_hints=result.entity_type_hints,
        entity_property_hints=result.entity_property_hints,
        failed=result.failed,
        claim_sources={source.message_id: source for source in sources},
        source_validated=True,
    )


class _PreparedDedupVectors(dict[str, list[float]]):
    """Validated batch plus the exact vector-space identity it was built in."""

    def __init__(self, *, model: str, dim: int):
        super().__init__()
        self.model = model
        self.dim = dim


def _validated_dedup_vector(
    value: object, *, dim: int
) -> list[float] | None:
    """Return one strict finite/non-zero vector, else fail closed.

    Bool and numeric strings are intentionally rejected.  Coercing hostile or
    provider-drifted coordinates at this write-routing boundary could compare a
    value the embedding provider never actually emitted in this vector space.
    """
    if not isinstance(value, (list, tuple)) or len(value) != dim:
        return None
    if any(
        isinstance(coordinate, bool)
        or not isinstance(coordinate, (int, float))
        for coordinate in value
    ):
        return None
    try:
        numeric = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(coordinate) for coordinate in numeric):
        return None
    norm = math.sqrt(sum(coordinate * coordinate for coordinate in numeric))
    if not math.isfinite(norm) or norm <= 0:
        return None
    return numeric


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

    try:
        model = embedding_client.model
        dim = embedding_client.dim
    except Exception:
        log.warning("phase1.dedup_embedding_identity_unavailable")
        return {}
    if (
        not isinstance(model, str) or not model.strip()
        or isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
    ):
        log.warning("phase1.dedup_embedding_identity_invalid")
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
    try:
        vectors = embedding_client.embed(texts)
        stable_model = embedding_client.model
        stable_dim = embedding_client.dim
    except Exception:
        log.warning("phase1.dedup_embedding_failed", exc_info=True)
        return {}
    if stable_model != model or stable_dim != dim:
        log.warning("phase1.dedup_embedding_identity_changed")
        return {}
    if not isinstance(vectors, (list, tuple)) or len(vectors) != len(texts):
        log.warning("phase1.dedup_embedding_cardinality_invalid")
        return {}

    validated: list[list[float]] = []
    for vector in vectors:
        numeric = _validated_dedup_vector(vector, dim=dim)
        if numeric is None:
            log.warning("phase1.dedup_embedding_vector_invalid")
            return {}
        validated.append(numeric)

    prepared = _PreparedDedupVectors(model=model, dim=dim)
    prepared.update(zip(texts, validated))
    return prepared


def _record_failed_attempt(
    conn: sqlite3.Connection,
    chunk: Chunk,
    *,
    prompt_version: str,
    cfg: HyMemConfig | None,
) -> None:
    """Count a held failure and audibly quarantine at the retry bound.

    A failed shape is never written to ``processed_chunks``. At the configured
    bound, selection skips the chunk as quarantined so it cannot starve the
    dream budget; its attempt row remains explicit operator-visible loss state.
    Changing the prompt/retry policy reopens it. ``max_attempts=0`` retries
    forever.
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
    if max_attempts and attempts == max_attempts:
        log.warning(
            "phase1.extraction_quarantined chunk_id=%s attempts=%d "
            "action=held_unprocessed cursor_advanced=0",
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
) -> list[_InCycleEdge] | None:
    """Persist a ChunkExtraction. Caller wraps in core_db.transaction().

    When ``cfg`` is provided, every claim is weighted by its exact cited source
    message role. When ``cfg`` is provided,
    ``cfg.triple_dedup_enabled`` is set, and ``dedup_vectors`` carries a
    precomputed vector for a brand-new triple's candidate text, that triple is
    first checked against existing same-predicate edges by vector similarity,
    attaching evidence to a near-duplicate rather than minting a sibling edge.

    The candidate embedding is precomputed by :func:`prepare_dedup_vectors`
    *outside* this write transaction; ``persist_chunk_results`` never issues a
    network ``embed()`` call (``embedding_client`` is accepted only for
    signature compatibility and is no longer used here).

    ``in_cycle_edges`` is the committed same-wave pool (see
    :func:`new_in_cycle_pool`), threaded by the runner across every chunk. This
    function returns a detached staged successor; the caller must publish it to
    the shared pool only *after* the surrounding transaction commits. Entries
    retain vectors even while dormant so a same-cycle exact assertion can make
    a negative-first edge authoritative without embedding under the write lock.
    A brand-new triple is checked only against authoritative entries. Pass
    ``None`` to disable same-wave collapse (cross-cycle dedup is unaffected).
    """
    # Work on a detached registry. The caller publishes it only after the
    # surrounding SQLite transaction commits, so rollback cannot leak a stale
    # edge id/vector into the next chunk's same-wave decisions.
    staged_in_cycle_edges = (
        [replace(entry) for entry in in_cycle_edges]
        if in_cycle_edges is not None else None
    )

    # A failed parse is not an extraction transaction.  Some split/provider
    # failure modes still return a partial list of valid-looking triples; writing
    # those and then retrying made an incomplete attempt observable and left no
    # deterministic way to remove items omitted by the healed reply.  Record only
    # the attempt.  The next successful extraction is applied atomically below.
    if extraction.failed:
        _record_failed_attempt(conn, chunk, prompt_version=prompt_version, cfg=cfg)
        log.debug(
            "phase1.chunk_failed_not_persisted chunk_id=%s triples=%d markers=%d",
            chunk.id,
            len(extraction.triples),
            len(extraction.markers),
        )
        return staged_in_cycle_edges

    if extraction.source_validated:
        identities: set[tuple[str, str, str, int]] = set()
        for triple in extraction.triples:
            source_mid = triple.source_message_id
            if source_mid not in extraction.claim_sources:
                raise ValueError("triple cites a source outside its validated input")
            identity = (
                canonicalize.resolve(conn, triple.subject),
                triple.predicate,
                canonicalize.resolve(conn, triple.object),
                int(source_mid),
            )
            if identity in identities:
                raise ValueError("duplicate claim citation in validated extraction")
            identities.add(identity)
    elif any(triple.source_message_id is not None for triple in extraction.triples):
        raise ValueError("unvalidated extraction cannot persist source citations")

    # A portable/newer successful result is authoritative for the whole chunk.
    # Replaying an older prompt must not first delete that result's observations
    # and only discover the stale generation after the damage is done.
    if extraction.source_validated and evidence.claim_extraction_prompt_is_stale(
        conn, chunk_id=chunk.id, prompt_version=prompt_version
    ):
        conn.execute(
            "INSERT OR IGNORE INTO processed_chunks(chunk_id, prompt_version) "
            "VALUES (?, ?)",
            (chunk.id, prompt_version),
        )
        conn.execute(
            "DELETE FROM chunk_extraction_attempts "
            "WHERE chunk_id = ? AND prompt_version = ?",
            (chunk.id, prompt_version),
        )
        return staged_in_cycle_edges

    if extraction.source_validated and _is_exact_published_replay(
        conn, chunk, extraction, prompt_version=prompt_version, cfg=cfg,
        dedup_vectors=dedup_vectors, in_cycle_edges=staged_in_cycle_edges,
    ):
        _persist_replay_auxiliary(conn, chunk, extraction)
        return staged_in_cycle_edges

    role_weights = cfg.evidence_role_weights if cfg is not None else {}
    affected_edge_ids: set[int] = set()
    if extraction.source_validated:
        affected_edge_ids = evidence.begin_chunk_extraction_reconciliation(
            conn, chunk_id=chunk.id, prompt_version=prompt_version
        )
        conn.execute("DELETE FROM entity_mentions WHERE chunk_id = ?", (chunk.id,))
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
        claim_source = (
            extraction.claim_sources.get(t.source_message_id)
            if extraction.source_validated else None
        )
        source_role = claim_source.role if claim_source is not None else None
        claim_weight = role_weights.get(source_role, 1) if source_role else 1
        subj_canon, obj_canon, edge_id, evidence_id = _upsert_triple(
            conn, chunk.id, t,
            source_role=source_role,
            evidence_weight=claim_weight,
            claim_source=claim_source,
            prompt_version=prompt_version,
            cfg=cfg,
            dedup_vectors=dedup_vectors,
            in_cycle_edges=staged_in_cycle_edges,
        )
        mentioned.add(subj_canon)
        mentioned.add(obj_canon)
        affected_edge_ids.add(edge_id)
        if claim_source is not None:
            evidence.record_claim_observation(
                conn,
                chunk_id=chunk.id,
                edge_id=edge_id,
                source_session_id=claim_source.session_id,
                source_message_id=claim_source.message_id,
                polarity=t.polarity,
                prompt_version=prompt_version,
                evidence_id=evidence_id,
            )
    if mentioned:
        conn.executemany(
            "INSERT OR IGNORE INTO entity_mentions(chunk_id, entity_canonical) VALUES (?, ?)",
            [(chunk.id, e) for e in mentioned],
        )
    for m in extraction.markers:
        # Write idempotence: re-extracting a chunk re-attaches its markers
        # without duplicating them, the way the v36 source-key constraint
        # protects kg_evidence. Enforced here rather than by a unique
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

    if extraction.source_validated:
        evidence.record_claim_extraction_outcome(
            conn, chunk_id=chunk.id, prompt_version=prompt_version
        )
        evidence.finalize_chunk_extraction_reconciliation(
            conn, affected_edge_ids
        )
        if staged_in_cycle_edges is not None:
            # Whole-chunk reconciliation can supersede observations touched
            # earlier in this transaction. Publish registry eligibility only
            # from the now-final canonical projection, never from a per-triple
            # intermediate counter state.
            for entry in staged_in_cycle_edges:
                entry.authoritative = (
                    _dedup_edge_has_authoritative_positive_majority(
                        conn, entry.edge_id
                    )
                )

    # A failed extraction returned above and is never marked done.
    # `processed_chunks` is the
    # one-shot gate — a row here means no dream will ever look at this chunk
    # again under this prompt version — so marking a chunk whose extraction did
    # not complete converts a transient provider hiccup into a permanent hole,
    # indistinguishable in the DB from a chunk that genuinely held nothing.
    # Holding the mark instead retries on the next dream, which is what the
    # digest (v24 watermark), facts (v26 watermark) and fusion paths all do.
    # A clean parse that yielded nothing IS marked: that is the real floor.
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
    return staged_in_cycle_edges


def _chunk_first_message_role(conn: sqlite3.Connection, chunk_id: str) -> str | None:
    """Role of the chunk's first message (its start_message_id). Used to weight
    evidence by author. Returns None if the chunk or message is gone."""
    row = conn.execute(
        """
        SELECT c.session_id, c.start_message_id, m.role
        FROM chunks c
        LEFT JOIN messages m
          ON m.id = c.start_message_id AND m.session_id = c.session_id
        WHERE c.id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    if row["role"] is not None:
        return row["role"]
    if row["session_id"] is None or row["start_message_id"] is None:
        return None
    start = int(row["start_message_id"])
    covered = covered_messages_after(
        conn,
        row["session_id"],
        start - 1,
        limit=1,
        through_message_id=start,
    )
    if covered and covered[0].message_id == start:
        return covered[0].role
    # No guessed/default role: a missing or mismatched durable proof must not
    # silently change evidence weighting after raw-message retention.
    return None


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


def _dedup_edge_has_authoritative_positive_majority(
    conn: sqlite3.Connection, edge_id: int
) -> bool:
    """Validate current/open authority and weighted majority from ledgers.

    Materialized graph counters are caches and are deliberately ignored. This
    shared check is used both for cross-cycle candidates and for the staged
    same-wave registry after whole-chunk publication/finalization.
    """
    from hymem.query.graph_state import (
        current_positive_state,
        validated_confidence_signal_totals,
        validated_current_evidence,
    )

    try:
        evidence_rows = validated_current_evidence(conn, edge_id=edge_id)
        if not evidence_rows:
            return False
        state = current_positive_state(
            conn, edge_id, limit=max(5, len(evidence_rows))
        )
        validated_ids = {int(row["evidence_id"]) for row in evidence_rows}
        citations = [
            citation for citation in (state[1] if state else ())
            if citation.evidence_id in validated_ids
        ]
        if state is None or not citations:
            return False
        positive = sum(
            int(row["evidence_weight"])
            for row in evidence_rows if int(row["polarity"]) == 1
        )
        negative = sum(
            int(row["evidence_weight"])
            for row in evidence_rows if int(row["polarity"]) == -1
        )
        signal_positive, signal_negative = (
            validated_confidence_signal_totals(
                conn, edge_ids={edge_id}
            ).get(edge_id, (0, 0))
        )
        positive += signal_positive
        negative += signal_negative
        return positive > negative and positive >= 0 and negative >= 0
    except (RuntimeError, TypeError, ValueError, sqlite3.Error):
        return False


def _eligible_dedup_edges(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    subj_canon: str,
    predicate: str,
    obj_canon: str,
    *,
    model: str,
    dim: int,
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
    # Dedup is a *write-routing* decision, not a read-side assertion that an
    # edge is currently true.  A pre-ledger/manual active edge can legitimately
    # have zero counters and no evidence yet (the historical seed shape used by
    # imports and old stores); the first canonical assertion should attach to
    # that identity instead of minting a sibling.  Keep that compatibility case
    # deliberately narrow: an edge with any evidence must satisfy the normal
    # positive-majority rule, while status/valid-time/publication/derived gates
    # are still supplied by ``live_edge_predicate``.  In particular, a zeroed
    # edge that has retracted, negative, unpublished, or corrupt evidence is not
    # a blank legacy seed and cannot be resurrected through semantic dedup.
    structurally_live = live_edge_predicate(
        "kg", require_positive_majority=False
    )
    pristine_seed = (
        "kg.pos_evidence = 0 AND kg.neg_evidence = 0 "
        "AND NOT EXISTS (SELECT 1 FROM kg_evidence hymem_any_ev "
        "WHERE hymem_any_ev.edge_id = kg.id) "
        "AND NOT EXISTS (SELECT 1 FROM kg_evidence_signals hymem_any_signal "
        "WHERE hymem_any_signal.edge_id = kg.id) "
        "AND NOT EXISTS (SELECT 1 FROM kg_claim_observations hymem_any_claim "
        "WHERE hymem_any_claim.edge_id = kg.id) "
        "AND NOT EXISTS (SELECT 1 FROM kg_edge_lifecycle hymem_any_lifecycle "
        "WHERE hymem_any_lifecycle.edge_id = kg.id)"
    )
    rows = conn.execute(
        f"""
        SELECT kg.id AS edge_id, kg.subject_canonical AS s, kg.object_canonical AS o,
               kg.pos_evidence AS pos_evidence,
               kg.neg_evidence AS neg_evidence,
               ({pristine_seed}) AS pristine_seed,
               e.vector_json AS vector_json
        FROM knowledge_graph kg
        JOIN edge_embeddings e
          ON e.edge_text = kg.subject_canonical || ' ' || kg.predicate || ' '
                           || kg.object_canonical
         AND e.model = ? AND e.dim = ?
        WHERE {structurally_live} AND kg.predicate = ?
          AND (kg.subject_canonical = ? OR kg.object_canonical = ?)
        ORDER BY {graph_clock_order_sql('kg.last_seen')}, kg.id
        """,
        (model, dim, predicate, subj_canon, obj_canon),
    ).fetchall()

    eligible: list[sqlite3.Row] = []
    for r in rows:
        # Cheap lexical/structural rejection comes before lifecycle replay so a
        # high-degree predicate cannot turn one candidate into N proof walks.
        if not _structural_lexical_match(
            cfg, subj_canon, predicate, obj_canon, r["s"], predicate, r["o"]
        ):
            continue
        # SQL already filtered predicate + one-endpoint-shared; reuse the shared
        # gate so the lexical-sibling check is defined identically to same-wave.
        if not bool(r["pristine_seed"]):
            if not _dedup_edge_has_authoritative_positive_majority(
                conn, int(r["edge_id"])
            ):
                continue
        eligible.append(r)
    return eligible


def _find_near_duplicate_edge(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    candidate_vector: list[float] | None,
    subj_canon: str,
    predicate: str,
    obj_canon: str,
    *,
    model: str | None,
    dim: int | None,
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
    if (
        candidate_vector is None
        or not isinstance(model, str) or not model
        or isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
    ):
        return None

    qvec = _validated_dedup_vector(candidate_vector, dim=dim)
    if qvec is None:
        return None

    eligible = _eligible_dedup_edges(
        conn, cfg, subj_canon, predicate, obj_canon, model=model, dim=dim
    )
    if not eligible:
        return None

    qnorm = math.sqrt(sum(x * x for x in qvec))
    best_id: int | None = None
    best_sim = -1.0
    for r in eligible:
        try:
            raw_vec = json.loads(r["vector_json"])
        except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
            continue
        vec = _validated_dedup_vector(raw_vec, dim=dim)
        if vec is None:
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        vnorm = math.sqrt(sum(x * x for x in vec))
        if not math.isfinite(vnorm) or vnorm <= 0:
            continue
        sim = dot / (qnorm * vnorm)
        if not math.isfinite(sim):
            continue
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
    model: str
    dim: int
    authoritative: bool = True


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
    *,
    model: str | None,
    dim: int | None,
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
    if (
        candidate_vector is None or not in_cycle_edges
        or not isinstance(model, str) or not model
        or isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
    ):
        return None

    qvec = _validated_dedup_vector(candidate_vector, dim=dim)
    if qvec is None:
        return None
    qnorm = math.sqrt(sum(x * x for x in qvec))
    best_id: int | None = None
    best_sim = -1.0
    for e in in_cycle_edges:
        if not e.authoritative:
            continue
        if e.model != model or e.dim != dim:
            continue
        if not _structural_lexical_match(
            cfg, subj_canon, predicate, obj_canon,
            e.subject, e.predicate, e.object,
        ):
            continue
        vec = _validated_dedup_vector(e.vector, dim=dim)
        if vec is None:
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        vnorm = math.sqrt(sum(x * x for x in vec))
        if not math.isfinite(vnorm) or vnorm <= 0:
            continue
        sim = dot / (qnorm * vnorm)
        if not math.isfinite(sim):
            continue
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
    evidence_weight: int = 1,
    claim_source: CoveredMessage | None = None,
    prompt_version: str | None = None,
    cfg: HyMemConfig | None = None,
    dedup_vectors: dict[str, list[float]] | None = None,
    in_cycle_edges: list[_InCycleEdge] | None = None,
) -> tuple[str, str, int, int]:
    subj_canon = canonicalize.resolve(conn, triple.subject)
    obj_canon = canonicalize.resolve(conn, triple.object)
    candidate_vector: list[float] | None = None
    minted_edge = False

    # Track surface forms as aliases so future mentions normalize the same way.
    canonicalize.register_alias(conn, triple.subject, subj_canon)
    canonicalize.register_alias(conn, triple.object, obj_canon)

    existing = conn.execute(
        "SELECT id, derived FROM knowledge_graph "
        "WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?",
        (subj_canon, triple.predicate, obj_canon),
    ).fetchone()

    if existing is not None:
        edge_id = existing["id"]
        if int(existing["derived"]):
            # An observed claim outranks an inferred closure. Promote the exact
            # row before attaching durable evidence so the next inference
            # rebuild cannot cascade-delete its provenance.
            conn.execute(
                """
                UPDATE knowledge_graph
                SET derived = 0, pos_evidence = 0, neg_evidence = 0,
                    status = 'active', valid_at = NULL, invalid_at = NULL,
                    last_seen = CURRENT_TIMESTAMP
                WHERE id = ? AND derived = 1
                """,
                (edge_id,),
            )
    else:
        edge_id = None
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
                    conn, cfg, candidate_vector, subj_canon, triple.predicate,
                    obj_canon,
                    model=getattr(dedup_vectors, "model", None),
                    dim=getattr(dedup_vectors, "dim", None),
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
                        model=getattr(dedup_vectors, "model", None),
                        dim=getattr(dedup_vectors, "dim", None),
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
            minted_edge = True

    selected = conn.execute(
        "SELECT derived FROM knowledge_graph WHERE id = ?", (edge_id,)
    ).fetchone()
    if selected is not None and int(selected["derived"]):
        conn.execute(
            """
            UPDATE knowledge_graph
            SET derived = 0, pos_evidence = 0, neg_evidence = 0,
                status = 'active', valid_at = NULL, invalid_at = NULL,
                last_seen = CURRENT_TIMESTAMP
            WHERE id = ? AND derived = 1
            """,
            (edge_id,),
        )

    mutation = evidence.record_chunk_evidence(
        conn,
        edge_id=edge_id,
        chunk_id=chunk_id,
        evidence_kind="extraction",
        polarity=triple.polarity,
        evidence_weight=evidence_weight,
        weight_source=(
            f"configured_role:{source_role or 'unknown'}"
            if cfg is not None
            else "default_weight:1"
        ),
        prompt_version=prompt_version,
        source_role=source_role,
        source_peer_id=(claim_source.source_peer_id if claim_source else None),
        source_workspace_id=(
            claim_source.source_workspace_id if claim_source else None
        ),
        surface_subject=triple.subject,
        surface_object=triple.object,
        value_text=triple.value_text,
        value_numeric=triple.value_numeric,
        value_unit=triple.value_unit,
        temporal_scope=triple.temporal_scope,
        source_message_id=(claim_source.message_id if claim_source else None),
        source_session_id=(claim_source.session_id if claim_source else None),
        source_created_at=(claim_source.source_created_at if claim_source else None),
        source_event_at=(
            _normalized_source_event_at(conn, claim_source.source_created_at)
            if claim_source else None
        ),
        source_coverage_chunk_id=(claim_source.chunk_id if claim_source else None),
        source_coverage_version=(LOSSLESS_COVERAGE_VERSION if claim_source else None),
    )

    if triple.polarity == 1 and mutation.contribution_changed:
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
            SET last_seen = CURRENT_TIMESTAMP,
                last_reinforced = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (edge_id,),
        )
        if claim_source is None:
            conn.execute(
                "UPDATE knowledge_graph SET status = 'active', invalid_at = NULL "
                "WHERE id = ?",
                (edge_id,),
            )
    elif triple.polarity == -1 and mutation.contribution_changed:
        # last_reinforced is intentionally not updated for negative polarity: a
        # contradiction does not "refresh" the edge, so phase3 decay still fires
        # for edges that have only ever seen negative evidence.
        conn.execute(
            """
            UPDATE knowledge_graph
            SET last_seen = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (edge_id,),
        )

    if in_cycle_edges is not None and mutation.contribution_changed:
        # Keep a staged registry of every same-cycle minted edge with a vector,
        # but let the matcher see only entries whose post-mutation state is a
        # current positive majority. A negative-first edge is therefore dormant;
        # a later exact weighted positive can activate its retained vector, and a
        # tie/close can make an earlier positive dormant again.
        row = conn.execute(
            "SELECT status,derived,invalid_at,pos_evidence,neg_evidence "
            "FROM knowledge_graph WHERE id=?",
            (edge_id,),
        ).fetchone()
        remains_positive = bool(
            row is not None
            and row["status"] == "active"
            and not int(row["derived"] or 0)
            and row["invalid_at"] is None
            and int(row["pos_evidence"] or 0) > int(row["neg_evidence"] or 0)
        )
        entry = next(
            (
                candidate for candidate in in_cycle_edges
                if candidate.edge_id == int(edge_id)
            ),
            None,
        )
        if entry is not None:
            entry.authoritative = remains_positive
        elif (
            minted_edge
            and candidate_vector is not None
            and cfg is not None
            and cfg.triple_dedup_enabled
            and isinstance(dedup_vectors, _PreparedDedupVectors)
            and _validated_dedup_vector(
                candidate_vector, dim=dedup_vectors.dim
            ) is not None
        ):
            in_cycle_edges.append(
                _InCycleEdge(
                    subject=subj_canon,
                    predicate=triple.predicate,
                    object=obj_canon,
                    vector=list(candidate_vector),
                    edge_id=int(edge_id),
                    model=dedup_vectors.model,
                    dim=dedup_vectors.dim,
                    authoritative=remains_positive,
                )
            )

    return subj_canon, obj_canon, int(edge_id), mutation.evidence_id


def _normalized_source_event_at(
    conn: sqlite3.Connection, source_created_at: str | None
) -> str:
    """Return the canonical valid-time coordinate for a claim source.

    Public ingestion has already validated and canonicalized this value. The
    fallback exists solely for pre-upgrade/direct-SQL messages whose historical
    timestamp is absent or malformed; treating those as ancient preserves the
    established conservative migration contract without admitting malformed
    values through the public API.
    """
    del conn
    try:
        return normalize_iso_timestamp(
            source_created_at,
            context="claim source created_at",
        )
    except ValueError:
        return "0001-01-01T00:00:00.000Z"
