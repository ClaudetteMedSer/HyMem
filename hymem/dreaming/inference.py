from __future__ import annotations

from collections import deque
import logging
import sqlite3

from hymem.config import HyMemConfig

log = logging.getLogger("hymem.dreaming.inference")


def infer_transitive_edges(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    """Compute transitive closure for derived edges.

    Two rules, both emitting derived ``depends_on`` edges (the conservative
    interpretation: a chain through `uses`/`depends_on` implies the subject
    transitively depends on the terminal object):

      1. ``A depends_on B, B depends_on C`` → ``A depends_on C`` (BFS).
      2. ``A uses B, B depends_on C`` → ``A depends_on C`` (one-hop cross-
         predicate, matches the improv.md `transitively_depends_on` rule but
         folded into ``depends_on`` so the predicate vocabulary stays stable
         and no schema migration is required).

    Confidence is the product of the source edges' smoothed confidences. New
    edges below ``cfg.retract_threshold`` or duplicating a direct edge are
    skipped. All previously-derived edges are wiped first so a re-run
    refreshes the closure from scratch.

    Returns the total number of new derived edges inserted.
    """
    conn.execute("DELETE FROM knowledge_graph WHERE derived = 1")

    depends_rows = conn.execute(
        """SELECT subject_canonical AS s, object_canonical AS o,
                  (pos_evidence + 1.0)/(pos_evidence + neg_evidence + 2.0) AS conf
           FROM knowledge_graph
           WHERE predicate = 'depends_on' AND status = 'active' AND derived = 0"""
    ).fetchall()
    uses_rows = conn.execute(
        """SELECT subject_canonical AS s, object_canonical AS o,
                  (pos_evidence + 1.0)/(pos_evidence + neg_evidence + 2.0) AS conf
           FROM knowledge_graph
           WHERE predicate = 'uses' AND status = 'active' AND derived = 0"""
    ).fetchall()

    if not depends_rows and not uses_rows:
        return 0

    depends_graph: dict[str, list[tuple[str, float]]] = {}
    existing: set[tuple[str, str]] = set()
    for r in depends_rows:
        depends_graph.setdefault(r["s"], []).append((r["o"], float(r["conf"])))
        existing.add((r["s"], r["o"]))
    # Existing direct uses edges shouldn't be shadowed by a derived
    # depends_on with the same (subject, object) — track them too.
    direct_uses: set[tuple[str, str]] = {(r["s"], r["o"]) for r in uses_rows}

    derived_count = 0
    # Rule 1: depends_on chains.
    for start_node in list(depends_graph.keys()):
        best_conf: dict[str, float] = {}
        for neighbor, conf in depends_graph.get(start_node, []):
            if conf > best_conf.get(neighbor, 0):
                best_conf[neighbor] = conf

        queue: deque[tuple[str, float]] = deque(
            (n, c) for n, c in best_conf.items()
        )
        while queue:
            node, path_conf = queue.popleft()
            for neighbor, edge_conf in depends_graph.get(node, []):
                new_conf = path_conf * edge_conf
                if new_conf > best_conf.get(neighbor, 0):
                    best_conf[neighbor] = new_conf
                    queue.append((neighbor, new_conf))

        for target, conf in best_conf.items():
            if start_node == target:
                continue
            if (start_node, target) in existing:
                continue
            if conf < cfg.retract_threshold:
                continue
            conn.execute(
                """INSERT OR IGNORE INTO knowledge_graph
                   (subject_canonical, predicate, object_canonical, pos_evidence, neg_evidence, derived)
                   VALUES (?, 'depends_on', ?, 1, 0, 1)""",
                (start_node, target),
            )
            existing.add((start_node, target))
            derived_count += 1

    # Rule 2: `A uses B + B depends_on C → A depends_on C`. We don't chain
    # further (e.g. uses → depends_on → depends_on); the depends_on BFS above
    # already covers transitive propagation from B onward, but we read from
    # the freshly-extended `existing` set so those derived B→C edges
    # participate as second hops here.
    refreshed_depends: dict[str, list[tuple[str, float]]] = {}
    for r in conn.execute(
        """SELECT subject_canonical AS s, object_canonical AS o,
                  (pos_evidence + 1.0)/(pos_evidence + neg_evidence + 2.0) AS conf
           FROM knowledge_graph
           WHERE predicate = 'depends_on' AND status = 'active'"""
    ).fetchall():
        refreshed_depends.setdefault(r["s"], []).append((r["o"], float(r["conf"])))

    for r in uses_rows:
        a = r["s"]
        b = r["o"]
        uses_conf = float(r["conf"])
        for c, dep_conf in refreshed_depends.get(b, []):
            if a == c:
                continue
            if (a, c) in existing:
                continue
            if (a, c) in direct_uses:
                continue
            new_conf = uses_conf * dep_conf
            if new_conf < cfg.retract_threshold:
                continue
            conn.execute(
                """INSERT OR IGNORE INTO knowledge_graph
                   (subject_canonical, predicate, object_canonical, pos_evidence, neg_evidence, derived)
                   VALUES (?, 'depends_on', ?, 1, 0, 1)""",
                (a, c),
            )
            existing.add((a, c))
            derived_count += 1

    if derived_count:
        log.info("inference.derived count=%d", derived_count)
    return derived_count
