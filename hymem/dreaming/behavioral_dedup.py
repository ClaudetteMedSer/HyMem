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
import logging
import math
import sqlite3
from dataclasses import dataclass, field

from hymem.core.vectors import decode_vector
from hymem.core import db as core_db
from hymem.core.graph import graph_clock_order_sql, live_edge_predicate
from hymem.dreaming import evidence as evidence_ledger

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


def find_behavioral_duplicates(
    conn: sqlite3.Connection,
    *,
    cosine_threshold: float,
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
    placeholders = ",".join("?" * len(predicates))
    rows = conn.execute(
        f"""
        SELECT kg.id AS edge_id, kg.subject_canonical AS s, kg.predicate AS p,
               kg.object_canonical AS o, kg.pos_evidence AS pos,
               kg.neg_evidence AS neg, e.vector_json AS vector_json
        FROM knowledge_graph kg
        JOIN edge_embeddings e
          ON e.edge_text = kg.subject_canonical || ' ' || kg.predicate || ' '
                           || kg.object_canonical
        WHERE {live_edge_predicate('kg')}
          AND kg.predicate IN ({placeholders})
        ORDER BY kg.subject_canonical, kg.predicate,
                 (kg.pos_evidence + kg.neg_evidence) DESC, kg.id ASC
        """,
        predicates,
    ).fetchall()

    # Bucket by (subject, predicate), preserving the evidence-desc order so the
    # first edge seen in each group is the strongest → the proposed survivor.
    groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for r in rows:
        groups.setdefault((r["s"], r["p"]), []).append(r)

    proposals: list[ProposedMerge] = []
    for (subject, predicate), edges in groups.items():
        if len(edges) < 2:
            continue
        vectors = {r["edge_id"]: decode_vector(r["vector_json"]) for r in edges}
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
                    )
                )

    proposals.sort(key=lambda m: m.collapses, reverse=True)
    return proposals


def apply_behavioral_merges(
    conn: sqlite3.Connection,
    proposals: list[ProposedMerge],
) -> dict:
    """Execute the merges proposed by :func:`find_behavioral_duplicates`.

    For each cluster, folds all member edges into the survivor: evidence
    provenance is unioned and duplicate sources collapsed, member edge rows are
    removed, member object canonicals are aliased to the survivor's, and
    retraction feedback is recorded so the next dream learns from the correction.

    Caller must wrap this in a ``core_db.transaction()`` — this function does
    NOT open its own transaction so it can be part of a larger atomic unit.
    Returns ``{clusters_merged, edges_retracted, survivors_updated}``.

    Idempotent: calling it twice with the same proposals is a no-op on the
    second call (already-collapsed edges are absent).
    """
    clusters_merged = 0
    edges_retracted = 0
    survivors_updated = 0

    for proposal in proposals:
        # Guard: skip if the survivor itself was retracted between report and apply.
        survivor_active = conn.execute(
            f"SELECT 1 FROM knowledge_graph WHERE id = ? "
            f"AND {live_edge_predicate()}",
            (proposal.survivor_id,),
        ).fetchone()
        if not survivor_active:
            continue

        proposed_member_ids = [m.edge_id for m in proposal.members]
        if not proposed_member_ids:
            continue
        proposed_placeholders = ",".join("?" * len(proposed_member_ids))
        member_ids = [
            int(row["id"])
            for row in conn.execute(
                f"SELECT id FROM knowledge_graph "
                f"WHERE id IN ({proposed_placeholders}) "
                f"AND {live_edge_predicate()}",
                proposed_member_ids,
            ).fetchall()
        ]
        if not member_ids:
            continue  # all members already retracted — idempotent

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

        # Record retraction feedback for each member edge.
        for member in proposal.members:
            evidence = conn.execute(
                f"""SELECT chunk_id FROM kg_evidence
                   WHERE edge_id = ? AND polarity = 1 AND is_current = 1
                   ORDER BY {graph_clock_order_sql('extracted_at')}, id
                   LIMIT 1""",
                (proposal.survivor_id,),
            ).fetchone()
            if evidence is None:
                continue
            chunk = conn.execute(
                "SELECT text FROM chunks WHERE id = ?", (evidence["chunk_id"],)
            ).fetchone()
            if chunk is None:
                continue
            conn.execute(
                """INSERT OR IGNORE INTO extraction_feedback
                   (chunk_id, chunk_text_snippet, extracted_subject,
                    extracted_predicate, extracted_object, feedback_type)
                   VALUES (?, ?, ?, ?, ?, 'retracted')""",
                (
                    evidence["chunk_id"],
                    chunk["text"][:600],
                    proposal.subject,
                    proposal.predicate,
                    member.object,
                ),
            )

        # Register object aliases: member object → survivor object.
        from hymem.dreaming import canonicalize

        for member in proposal.members:
            if member.object != proposal.survivor_object:
                canonicalize.register_alias(conn, member.object, proposal.survivor_object)

        # Clean up edge_embeddings for retracted members — the vectors are now
        # dead weight and would pollute future KNN searches.
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

    return {
        "clusters_merged": clusters_merged,
        "edges_retracted": edges_retracted,
        "survivors_updated": survivors_updated,
    }
