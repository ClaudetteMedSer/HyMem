from __future__ import annotations


def _seed_edge(hy, subject: str, predicate: str, obj: str, pos: int = 3, neg: int = 0) -> int:
    cur = hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (subject, predicate, obj, pos, neg),
    )
    return cur.lastrowid


def test_retract_existing_edge(hy):
    _seed_edge(hy, "med_flow", "depends_on", "redis")

    assert hy.retract_edge("med_flow", "depends_on", "redis") is True

    row = hy.conn.execute(
        "SELECT status, neg_evidence FROM knowledge_graph "
        "WHERE subject_canonical='med_flow' AND predicate='depends_on' "
        "AND object_canonical='redis'"
    ).fetchone()
    assert row["status"] == "retracted"
    assert row["neg_evidence"] == 1

    ctx = hy.augment("tell me about med_flow and redis")
    for fact in ctx.graph_facts:
        assert not (
            fact.subject == "med_flow"
            and fact.predicate == "depends_on"
            and fact.object == "redis"
        )


def test_retract_nonexistent_edge_returns_false(hy):
    assert hy.retract_edge("ghost", "uses", "nothing") is False


def test_retract_resolves_aliases(hy):
    hy.register_alias("MedFlow", "med_flow")
    _seed_edge(hy, "med_flow", "uses", "postgres")

    assert hy.retract_edge("MedFlow", "uses", "postgres") is True

    row = hy.conn.execute(
        "SELECT status FROM knowledge_graph "
        "WHERE subject_canonical='med_flow' AND predicate='uses' "
        "AND object_canonical='postgres'"
    ).fetchone()
    assert row["status"] == "retracted"


def test_retract_is_idempotent(hy):
    _seed_edge(hy, "med_flow", "uses", "kafka")

    assert hy.retract_edge("med_flow", "uses", "kafka") is True
    assert hy.retract_edge("med_flow", "uses", "kafka") is False

    row = hy.conn.execute(
        "SELECT status, neg_evidence FROM knowledge_graph "
        "WHERE subject_canonical='med_flow' AND predicate='uses' "
        "AND object_canonical='kafka'"
    ).fetchone()
    assert row["status"] == "retracted"
    assert row["neg_evidence"] == 1
