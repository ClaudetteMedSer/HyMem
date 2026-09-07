"""Retroactive behavioral-edge deduplication and explicit merge application.

Same-wave collapse (see `phase1.py`) is forward-looking: it stops *new* dreams
from fanning one preference out into many phrasal-variant edges, but it does not
touch behavioral edges (`prefers` / `avoids` / `rejects`) that were minted before
it existed. Those keep inflating the `conflicts()` count.

This module reports which of those pre-existing edges *would* collapse if merged
on semantic similarity alone — deliberately **dropping the lexical-sibling gate**
that normal dedup applies. That gate (`_entities_are_siblings`) exists to stop
false merges of short, embedding-close tool *names* (`redis` / `redash`).
Behavioral *objects* are abstract phrases (`concise`, `brevity`, `short answers`)
where lexical siblinghood is the wrong test — semantic closeness is the signal.
Dropping it is exactly why this remains report-first: a human reviews the
proposed merges before asking :func:`apply_behavioral_merges` to collapse them.

The report is a pure read path and reuses cached `edge_embeddings` vectors (no
embedding-API call). Applying an accepted proposal moves durable provenance to
the survivor, records the alias, and removes the now-authority-free member edge.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import math
import sqlite3
from dataclasses import dataclass, field

from hymem.core.vectors import decode_vector
from hymem.core import db as core_db
from hymem.core.graph import graph_clock_order_sql, live_edge_predicate
from hymem.dreaming import evidence as evidence_ledger
from hymem.dreaming.aggregation_material import embedding_execution_identity

log = logging.getLogger("hymem.dreaming.behavioral_dedup")

# The multi-valued behavioral predicates that proliferate into phrasal variants.
# (uses / depends_on etc. are excluded: their objects are concrete named things
# where the lexical gate is still the right guard.)
BEHAVIORAL_PREDICATES: tuple[str, ...] = ("prefers", "avoids", "rejects")


@dataclass
class DuplicateMember:
    """A non-survivor edge proposed for merging into the cluster survivor."""
    edge_id: int
    object: str
    pos_evidence: int
    neg_evidence: int
    cosine_to_survivor: float
    vector_proof_sha256: str = ""


@dataclass
class ProposedMerge:
    """One cluster of behavioral edges that would collapse into `survivor`."""
    subject: str
    predicate: str
    survivor_id: int
    survivor_object: str
    survivor_pos: int
    survivor_neg: int
    members: list[DuplicateMember] = field(default_factory=list)
    embedding_producer_key: str = ""
    embedding_dim: int = 0
    survivor_vector_proof_sha256: str = ""

    @property
    def collapses(self) -> int:
        """How many edges this cluster would remove (members folded away)."""
        return len(self.members)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _validated_vector(value: object, *, dimension: int) -> list[float] | None:
    try:
        decoded = decode_vector(value)
        vector = [float(item) for item in decoded]
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if len(vector) != dimension or not all(math.isfinite(item) for item in vector):
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    return vector if math.isfinite(norm) and norm > 0.0 else None


def _vector_proof(
    *, edge_text: str, vector_json: str, model: str, dimension: int,
) -> str:
    payload = "\0".join((edge_text, model, str(dimension), vector_json))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_behavioral_duplicates(
    conn: sqlite3.Connection,
    *,
    cosine_threshold: float,
    embedding_client: object | None = None,
    predicates: tuple[str, ...] = BEHAVIORAL_PREDICATES,
) -> list[ProposedMerge]:
    """Report behavioral-edge clusters that would merge at `cosine_threshold`.

    Read-only. Groups active, non-derived behavioral edges by `(subject,
    predicate)`; within each group, greedily clusters around the
    highest-evidence edge (the proposed survivor), pulling in any other edge
    whose cached-vector cosine to that survivor is at least `cosine_threshold`.
    Clusters of two or more become a `ProposedMerge`. Edges with no cached
    `edge_embeddings` vector are skipped (they cannot be compared without an
    embed call, which this dry run deliberately avoids).

    Returns proposed merges sorted by how many edges each would collapse,
    descending — the biggest noise sources first.
    """
    if embedding_client is None:
        return []
    binding, producer_key, dimension = embedding_execution_identity(
        embedding_client
    )
    if (
        binding.get("identity_exact") is not True
        or binding.get("reuse_scope") != "durable"
        or dimension is None
    ):
        return []
    placeholders = ",".join("?" * len(predicates))
    rows = conn.execute(
        f"""
        SELECT kg.id AS edge_id, kg.subject_canonical AS s, kg.predicate AS p,
               kg.object_canonical AS o, kg.pos_evidence AS pos,
               kg.neg_evidence AS neg, e.edge_text AS edge_text,
               e.vector_json AS vector_json, e.model AS model, e.dim AS dim
        FROM knowledge_graph kg
        JOIN edge_embeddings e
          ON e.edge_text = kg.subject_canonical || ' ' || kg.predicate || ' '
                           || kg.object_canonical
        WHERE {live_edge_predicate('kg')}
          AND kg.predicate IN ({placeholders})
          AND e.model = ? AND e.dim = ?
        ORDER BY kg.subject_canonical, kg.predicate,
                 (kg.pos_evidence + kg.neg_evidence) DESC, kg.id ASC
        """,
        (*predicates, producer_key, dimension),
    ).fetchall()

    # Bucket by (subject, predicate), preserving the evidence-desc order so the
    # first edge seen in each group is the strongest → the proposed survivor.
    groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
    vectors: dict[int, list[float]] = {}
    proofs: dict[int, str] = {}
    for r in rows:
        vector = _validated_vector(r["vector_json"], dimension=dimension)
        if vector is None:
            continue
        edge_id = int(r["edge_id"])
        vectors[edge_id] = vector
        proofs[edge_id] = _vector_proof(
            edge_text=str(r["edge_text"]),
            vector_json=str(r["vector_json"]),
            model=str(r["model"]),
            dimension=int(r["dim"]),
        )
        groups.setdefault((r["s"], r["p"]), []).append(r)

    proposals: list[ProposedMerge] = []
    for (subject, predicate), edges in groups.items():
        if len(edges) < 2:
            continue
        unclustered = list(edges)
        while unclustered:
            survivor = unclustered.pop(0)
            svec = vectors[survivor["edge_id"]]
            members: list[DuplicateMember] = []
            still: list[sqlite3.Row] = []
            for cand in unclustered:
                sim = _cosine(svec, vectors[cand["edge_id"]])
                if sim >= cosine_threshold:
                    members.append(
                        DuplicateMember(
                            edge_id=cand["edge_id"],
                            object=cand["o"],
                            pos_evidence=cand["pos"],
                            neg_evidence=cand["neg"],
                            cosine_to_survivor=round(sim, 4),
                            vector_proof_sha256=proofs[cand["edge_id"]],
                        )
                    )
                else:
                    still.append(cand)
            unclustered = still
            if members:
                proposals.append(
                    ProposedMerge(
                        subject=subject,
                        predicate=predicate,
                        survivor_id=survivor["edge_id"],
                        survivor_object=survivor["o"],
                        survivor_pos=survivor["pos"],
                        survivor_neg=survivor["neg"],
                        members=members,
                        embedding_producer_key=producer_key,
                        embedding_dim=dimension,
                        survivor_vector_proof_sha256=proofs[
                            survivor["edge_id"]
                        ],
                    )
                )

    proposals.sort(key=lambda m: m.collapses, reverse=True)
    if embedding_execution_identity(embedding_client) != (
        binding, producer_key, dimension,
    ):
        raise RuntimeError("embedding identity changed during behavioral report")
    return proposals


def apply_behavioral_merges(
    conn: sqlite3.Connection,
    proposals: list[ProposedMerge],
    *,
    embedding_client: object | None = None,
) -> dict:
    """Execute the merges proposed by :func:`find_behavioral_duplicates`.

    For each cluster, folds all member edges into the survivor: evidence
    provenance is unioned and duplicate sources collapsed, member edge rows are
    removed, member object canonicals are aliased to the survivor's, and a
    source-linked retraction audit record is retained for each removed edge
    that has current positive chunk evidence.

    Caller must wrap this in a ``core_db.transaction()`` — this function does
    NOT open its own transaction so it can be part of a larger atomic unit.
    Returns ``{clusters_merged, edges_retracted, survivors_updated}``.

    Idempotent: calling it twice with the same proposals is a no-op on the
    second call (already-collapsed edges are absent).
    """
    clusters_merged = 0
    edges_retracted = 0
    survivors_updated = 0

    if not proposals:
        return {
            "clusters_merged": 0,
            "edges_retracted": 0,
            "survivors_updated": 0,
        }
    if embedding_client is None:
        raise ValueError("behavioral merge requires the live embedding producer")
    binding, producer_key, dimension = embedding_execution_identity(
        embedding_client
    )
    if (
        binding.get("identity_exact") is not True
        or binding.get("reuse_scope") != "durable"
        or dimension is None
    ):
        raise ValueError("behavioral merge requires an exact embedding producer")

    for proposal in proposals:
        if (
            proposal.embedding_producer_key != producer_key
            or proposal.embedding_dim != dimension
        ):
            continue
        # Guard: skip if the survivor itself was retracted between report and apply.
        survivor_active = conn.execute(
            f"""SELECT kg.id, e.edge_text, e.vector_json, e.model, e.dim
                FROM knowledge_graph kg
                JOIN edge_embeddings e
                  ON e.edge_text = kg.subject_canonical || ' ' || kg.predicate
                                     || ' ' || kg.object_canonical
                WHERE kg.id = ?
                  AND subject_canonical = ?
                  AND predicate = ?
                  AND object_canonical = ?
                  AND e.model = ? AND e.dim = ?
                  AND {live_edge_predicate('kg')}""",
            (
                proposal.survivor_id,
                proposal.subject,
                proposal.predicate,
                proposal.survivor_object,
                producer_key,
                dimension,
            ),
        ).fetchone()
        if (
            not survivor_active
            or _validated_vector(
                survivor_active["vector_json"], dimension=dimension,
            ) is None
            or _vector_proof(
                edge_text=str(survivor_active["edge_text"]),
                vector_json=str(survivor_active["vector_json"]),
                model=str(survivor_active["model"]),
                dimension=int(survivor_active["dim"]),
            ) != proposal.survivor_vector_proof_sha256
        ):
            continue

        proposed_member_ids = [m.edge_id for m in proposal.members]
        if not proposed_member_ids:
            continue
        proposed_placeholders = ",".join("?" * len(proposed_member_ids))
        live_rows = {
            int(row["id"]): row
            for row in conn.execute(
                f"""SELECT kg.id, kg.subject_canonical, kg.predicate,
                           kg.object_canonical, e.edge_text, e.vector_json,
                           e.model, e.dim
                    FROM knowledge_graph kg
                    JOIN edge_embeddings e
                      ON e.edge_text = kg.subject_canonical || ' ' || kg.predicate
                                         || ' ' || kg.object_canonical
                    WHERE kg.id IN ({proposed_placeholders})
                      AND e.model = ? AND e.dim = ?
                      AND {live_edge_predicate('kg')}""",
                (*proposed_member_ids, producer_key, dimension),
            ).fetchall()
        }
        # Treat a report as an optimistic snapshot: apply only member identities
        # that are still exactly the live triples the report described.  In
        # particular, a member retracted between report and apply must not gain
        # an alias or an audit row merely because another member still merges.
        live_members: list[DuplicateMember] = []
        seen_member_ids: set[int] = set()
        for member in proposal.members:
            row = live_rows.get(member.edge_id)
            if (
                row is None
                or member.edge_id in seen_member_ids
                or row["subject_canonical"] != proposal.subject
                or row["predicate"] != proposal.predicate
                or row["object_canonical"] != member.object
                or _validated_vector(
                    row["vector_json"], dimension=dimension,
                ) is None
                or _vector_proof(
                    edge_text=str(row["edge_text"]),
                    vector_json=str(row["vector_json"]),
                    model=str(row["model"]),
                    dimension=int(row["dim"]),
                ) != member.vector_proof_sha256
            ):
                continue
            seen_member_ids.add(member.edge_id)
            live_members.append(member)
        if not live_members:
            continue  # all members already retracted — idempotent
        member_ids = [member.edge_id for member in live_members]

        # Capture each removed edge's own source before provenance is moved to
        # the survivor. Looking it up afterwards loses the member identity and
        # can attribute every audit record to an unrelated merged source.
        audit_sources: list[tuple[str, str, DuplicateMember]] = []
        for member in live_members:
            source = conn.execute(
                f"""SELECT evidence.chunk_id, chunk.text
                    FROM kg_evidence evidence
                    JOIN chunks chunk ON chunk.id = evidence.chunk_id
                    WHERE evidence.edge_id = ?
                      AND evidence.polarity = 1
                      AND evidence.is_current = 1
                    ORDER BY {graph_clock_order_sql('evidence.extracted_at')},
                             evidence.id
                    LIMIT 1""",
                (member.edge_id,),
            ).fetchone()
            if source is not None:
                audit_sources.append(
                    (str(source["chunk_id"]), str(source["text"]), member)
                )

        # Move source rows first and rebuild the survivor cache from their
        # unique union.  Summing edge-level counters over-counted the same
        # chunk when two aliases had independently extracted it.
        evidence_ledger.move_edge_provenance(conn, proposal.survivor_id, member_ids)
        conn.execute(
            """UPDATE knowledge_graph
               SET last_seen = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (proposal.survivor_id,),
        )
        survivors_updated += 1

        placeholders = ",".join("?" * len(member_ids))

        # Retain a source-linked audit record for each source-backed member.
        for chunk_id, chunk_text, member in audit_sources:
            conn.execute(
                """INSERT OR IGNORE INTO extraction_feedback
                   (chunk_id, chunk_text_snippet, extracted_subject,
                    extracted_predicate, extracted_object, feedback_type)
                   VALUES (?, ?, ?, ?, ?, 'retracted')""",
                (
                    chunk_id,
                    chunk_text[:600],
                    proposal.subject,
                    proposal.predicate,
                    member.object,
                ),
            )

        # Clean up edge_embeddings for retracted members — the vectors are now
        # dead weight and would pollute future KNN searches. Do this before the
        # full canonical merge can remove a member edge's text identity.
        conn.execute(
            f"""DELETE FROM edge_embeddings
                WHERE edge_text IN (
                    SELECT subject_canonical || ' ' || predicate || ' '
                           || object_canonical
                    FROM knowledge_graph
                    WHERE id IN ({placeholders})
                )""",
            member_ids,
        )

        # This is an identity merge, not merely a lexical alias.  The member
        # object may also own producer-scoped hints/mentions or contextual-rule
        # triggers; the full merger moves and re-hashes those ledgers instead
        # of stranding them behind a one-hop alias.
        from hymem.dreaming import canonicalize

        for member in live_members:
            if member.object != proposal.survivor_object:
                canonicalize.merge(
                    conn,
                    keep=proposal.survivor_object,
                    drop=member.object,
                )

        # Behavioral collapse is identity deduplication, not a world-time
        # retraction. All durable provenance/history has moved to the survivor,
        # so retaining a direct provenance-empty tombstone creates an
        # unexportable graph authority and an alias-resolved dead edge. Remove
        # every cache row and then the proven-empty member atomically.
        with contextlib.suppress(sqlite3.OperationalError):
            conn.executemany(
                "DELETE FROM vec_edges WHERE rowid=?",
                [(edge_id,) for edge_id in member_ids],
            )
        leftovers = conn.execute(
            f"""
            SELECT id FROM knowledge_graph kg
            WHERE id IN ({placeholders}) AND (
                EXISTS (SELECT 1 FROM kg_evidence ev WHERE ev.edge_id=kg.id)
                OR EXISTS (SELECT 1 FROM kg_evidence_signals signal
                           WHERE signal.edge_id=kg.id)
                OR EXISTS (SELECT 1 FROM kg_claim_observations observation
                           WHERE observation.edge_id=kg.id)
                OR EXISTS (SELECT 1 FROM kg_edge_lifecycle lifecycle
                           WHERE lifecycle.edge_id=kg.id)
            )
            """,
            member_ids,
        ).fetchall()
        if leftovers:
            raise RuntimeError("behavioral merge left member provenance behind")
        with core_db.evidence_mutation(conn):
            conn.executemany(
                "DELETE FROM knowledge_graph WHERE id=?",
                [(edge_id,) for edge_id in member_ids],
            )

        clusters_merged += 1
        edges_retracted += len(member_ids)
        totals = conn.execute(
            "SELECT pos_evidence, neg_evidence FROM knowledge_graph WHERE id = ?",
            (proposal.survivor_id,),
        ).fetchone()
        log.info(
            "behavioral_dedup.applied subject=%s predicate=%s "
            "survivor=%s members=%d pos=%d neg=%d",
            proposal.subject, proposal.predicate,
            proposal.survivor_object, len(member_ids),
            totals["pos_evidence"], totals["neg_evidence"],
        )

    if embedding_execution_identity(embedding_client) != (
        binding, producer_key, dimension,
    ):
        raise RuntimeError("embedding identity changed during behavioral merge")
    return {
        "clusters_merged": clusters_merged,
        "edges_retracted": edges_retracted,
        "survivors_updated": survivors_updated,
    }
