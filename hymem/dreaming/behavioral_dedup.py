"""Retroactive behavioral-edge dedup — **dry-run report only**.

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
Dropping it is exactly why this is report-first: a human reviews the proposed
merges before any apply step is built.

Pure read path: it reuses the cached `edge_embeddings` vectors (no embedding-API
call). Within a single `(subject, predicate)` group the subject and predicate are
identical, so the cosine between two full-triple-text vectors reflects the
*object* difference — which is what we want to cluster on.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

from hymem.core.vectors import decode_vector

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
        WHERE kg.status = 'active' AND kg.derived = 0
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
