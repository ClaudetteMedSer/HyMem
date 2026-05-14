"""Contradiction detection over the knowledge graph.

Surfaces two kinds of contradiction among active edges:
  - competing_object:    same subject + predicate, different objects, for
                         predicates where that is mutually exclusive-ish
                         (e.g. `prefers atta english` vs `prefers atta dutch`).
  - opposing_predicate:  same subject + object joined by an opposing predicate
                         pair (e.g. `prefers X Y` vs `rejects X Y`).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# Predicates where a single subject pointing at multiple objects is contradictory.
_EXCLUSIVE_PREDICATES = ("prefers", "runs_on", "requires_version", "deploys_to")

# Predicate pairs that contradict each other when they share subject + object.
_OPPOSING_PAIRS = frozenset(
    {
        frozenset({"prefers", "rejects"}),
        frozenset({"prefers", "avoids"}),
        frozenset({"uses", "rejects"}),
        frozenset({"uses", "avoids"}),
        frozenset({"depends_on", "conflicts_with"}),
    }
)


@dataclass
class Conflict:
    kind: str  # "competing_object" | "opposing_predicate"
    subject: str
    edge_a: tuple[str, str, str]
    edge_b: tuple[str, str, str]
    confidence_a: float
    confidence_b: float
    detail: str


def _conf(pos: int, neg: int) -> float:
    return (pos + 1.0) / (pos + neg + 2.0)


def find_conflicts(conn: sqlite3.Connection) -> list[Conflict]:
    """Return detected contradictions among active knowledge-graph edges."""
    return _competing_objects(conn) + _opposing_predicates(conn)


def _competing_objects(conn: sqlite3.Connection) -> list[Conflict]:
    placeholders = ",".join("?" * len(_EXCLUSIVE_PREDICATES))
    rows = conn.execute(
        f"""
        SELECT a.subject_canonical AS subj, a.predicate AS pred,
               a.object_canonical AS obj_a, b.object_canonical AS obj_b,
               a.pos_evidence AS pos_a, a.neg_evidence AS neg_a,
               b.pos_evidence AS pos_b, b.neg_evidence AS neg_b
        FROM knowledge_graph a
        JOIN knowledge_graph b
          ON a.subject_canonical = b.subject_canonical
         AND a.predicate = b.predicate
         AND a.object_canonical < b.object_canonical
        WHERE a.status = 'active' AND b.status = 'active'
          AND a.derived = 0 AND b.derived = 0
          AND a.predicate IN ({placeholders})
        """,
        _EXCLUSIVE_PREDICATES,
    ).fetchall()

    conflicts: list[Conflict] = []
    for r in rows:
        conflicts.append(
            Conflict(
                kind="competing_object",
                subject=r["subj"],
                edge_a=(r["subj"], r["pred"], r["obj_a"]),
                edge_b=(r["subj"], r["pred"], r["obj_b"]),
                confidence_a=_conf(r["pos_a"], r["neg_a"]),
                confidence_b=_conf(r["pos_b"], r["neg_b"]),
                detail=(
                    f"{r['subj']} [{r['pred']}] both {r['obj_a']} and {r['obj_b']}"
                ),
            )
        )
    return conflicts


def _opposing_predicates(conn: sqlite3.Connection) -> list[Conflict]:
    rows = conn.execute(
        """
        SELECT a.subject_canonical AS subj, a.object_canonical AS obj,
               a.predicate AS pred_a, b.predicate AS pred_b,
               a.pos_evidence AS pos_a, a.neg_evidence AS neg_a,
               b.pos_evidence AS pos_b, b.neg_evidence AS neg_b
        FROM knowledge_graph a
        JOIN knowledge_graph b
          ON a.subject_canonical = b.subject_canonical
         AND a.object_canonical = b.object_canonical
         AND a.predicate < b.predicate
        WHERE a.status = 'active' AND b.status = 'active'
          AND a.derived = 0 AND b.derived = 0
        """
    ).fetchall()

    conflicts: list[Conflict] = []
    for r in rows:
        if frozenset({r["pred_a"], r["pred_b"]}) not in _OPPOSING_PAIRS:
            continue
        conflicts.append(
            Conflict(
                kind="opposing_predicate",
                subject=r["subj"],
                edge_a=(r["subj"], r["pred_a"], r["obj"]),
                edge_b=(r["subj"], r["pred_b"], r["obj"]),
                confidence_a=_conf(r["pos_a"], r["neg_a"]),
                confidence_b=_conf(r["pos_b"], r["neg_b"]),
                detail=(
                    f"{r['subj']} [{r['pred_a']}] vs [{r['pred_b']}] {r['obj']}"
                ),
            )
        )
    return conflicts
