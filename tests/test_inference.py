from __future__ import annotations

from hymem.dreaming.inference import infer_transitive_edges

from tests.conftest import seed_edge


def _derived_edges(conn) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        "SELECT subject_canonical, predicate, object_canonical "
        "FROM knowledge_graph WHERE derived = 1 "
        "ORDER BY subject_canonical, object_canonical"
    ).fetchall()
    return [(r["subject_canonical"], r["predicate"], r["object_canonical"]) for r in rows]


def test_depends_on_chain_derives_transitive_edge(hy):
    seed_edge(hy.conn, "api", "depends_on", "postgresql", pos=10)
    seed_edge(hy.conn, "postgresql", "depends_on", "libpq", pos=10)

    count = infer_transitive_edges(hy.conn, hy.config)

    assert count >= 1
    derived = _derived_edges(hy.conn)
    assert ("api", "depends_on", "libpq") in derived


def test_uses_plus_depends_on_derives_transitive_dependency(hy):
    seed_edge(hy.conn, "api", "uses", "postgresql", pos=10)
    seed_edge(hy.conn, "postgresql", "depends_on", "docker", pos=10)

    count = infer_transitive_edges(hy.conn, hy.config)

    assert count >= 1
    assert ("api", "depends_on", "docker") in _derived_edges(hy.conn)


def test_existing_direct_edge_blocks_derived(hy):
    # If api already directly depends on docker, the derived chain should not
    # duplicate that edge (we'd be polluting confidence interpretation).
    seed_edge(hy.conn, "api", "uses", "postgresql", pos=10)
    seed_edge(hy.conn, "postgresql", "depends_on", "docker", pos=10)
    seed_edge(hy.conn, "api", "depends_on", "docker", pos=5)

    infer_transitive_edges(hy.conn, hy.config)

    derived = _derived_edges(hy.conn)
    # No derived edge between api and docker — the direct one already exists.
    assert ("api", "depends_on", "docker") not in derived


def test_low_confidence_chain_below_retract_threshold_skipped(hy):
    # A long chain of weak edges should fall below cfg.retract_threshold and
    # not be emitted as a derived edge.
    seed_edge(hy.conn, "a", "depends_on", "b", pos=0, neg=2)  # conf ~0.17
    seed_edge(hy.conn, "b", "depends_on", "c", pos=0, neg=2)
    seed_edge(hy.conn, "c", "depends_on", "d", pos=0, neg=2)

    infer_transitive_edges(hy.conn, hy.config)

    # 0.17 * 0.17 * 0.17 ≈ 0.005 — way below the 0.15 threshold.
    derived = _derived_edges(hy.conn)
    assert ("a", "depends_on", "d") not in derived


def test_rerun_refreshes_derived_edges(hy):
    seed_edge(hy.conn, "api", "depends_on", "postgresql", pos=10)
    seed_edge(hy.conn, "postgresql", "depends_on", "libpq", pos=10)
    infer_transitive_edges(hy.conn, hy.config)

    # Retract one of the source edges; rerun should drop the derived edge.
    hy.conn.execute(
        "UPDATE knowledge_graph SET status='retracted' "
        "WHERE subject_canonical='postgresql' AND object_canonical='libpq'"
    )
    infer_transitive_edges(hy.conn, hy.config)

    assert ("api", "depends_on", "libpq") not in _derived_edges(hy.conn)


def test_self_loop_not_emitted(hy):
    # A → B → A shouldn't emit a (A, depends_on, A) derived edge.
    seed_edge(hy.conn, "a", "depends_on", "b", pos=10)
    seed_edge(hy.conn, "b", "depends_on", "a", pos=10)

    infer_transitive_edges(hy.conn, hy.config)

    derived = _derived_edges(hy.conn)
    assert ("a", "depends_on", "a") not in derived
    assert ("b", "depends_on", "b") not in derived


def test_existing_uses_edge_not_shadowed_by_derived_depends_on(hy):
    # If api already directly uses docker, don't also emit a derived
    # depends_on between the same pair — keeps the graph from carrying both
    # a direct uses and an inferred dependency with the same endpoints.
    seed_edge(hy.conn, "api", "uses", "docker", pos=10)
    seed_edge(hy.conn, "api", "uses", "postgresql", pos=10)
    seed_edge(hy.conn, "postgresql", "depends_on", "docker", pos=10)

    infer_transitive_edges(hy.conn, hy.config)

    derived = _derived_edges(hy.conn)
    assert ("api", "depends_on", "docker") not in derived
