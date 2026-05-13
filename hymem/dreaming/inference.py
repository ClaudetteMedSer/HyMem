from __future__ import annotations

from collections import deque
import logging
import sqlite3

from hymem.config import HyMemConfig

log = logging.getLogger("hymem.dreaming.inference")


def infer_transitive_edges(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    """Compute transitive closure over depends_on edges and insert derived edges.

    Returns number of new derived edges created.
    """
    conn.execute("DELETE FROM knowledge_graph WHERE derived = 1")

    rows = conn.execute(
        """SELECT subject_canonical AS s, object_canonical AS o,
                  (pos_evidence + 1.0)/(pos_evidence + neg_evidence + 2.0) AS conf
           FROM knowledge_graph
           WHERE predicate = 'depends_on' AND status = 'active' AND derived = 0"""
    ).fetchall()

    if not rows:
        return 0

    graph: dict[str, list[tuple[str, float]]] = {}
    existing: set[tuple[str, str]] = set()
    for r in rows:
        graph.setdefault(r["s"], []).append((r["o"], float(r["conf"])))
        existing.add((r["s"], r["o"]))

    derived_count = 0
    for start_node in list(graph.keys()):
        best_conf: dict[str, float] = {}
        for neighbor, conf in graph.get(start_node, []):
            if conf > best_conf.get(neighbor, 0):
                best_conf[neighbor] = conf

        queue: deque[tuple[str, float]] = deque(
            (n, c) for n, c in best_conf.items()
        )
        while queue:
            node, path_conf = queue.popleft()
            for neighbor, edge_conf in graph.get(node, []):
                new_conf = path_conf * edge_conf
                if new_conf > best_conf.get(neighbor, 0):
                    best_conf[neighbor] = new_conf
                    queue.append((neighbor, new_conf))

        for target, conf in best_conf.items():
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
            derived_count += 1

    if derived_count:
        log.info("inference.derived count=%d", derived_count)
    return derived_count
