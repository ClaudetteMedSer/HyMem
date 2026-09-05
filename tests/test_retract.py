from __future__ import annotations

from hymem.core import db as core_db


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


def test_retract_populates_extraction_feedback(hy):
    """retract_edge writes a row to extraction_feedback so phase1 can inject
    the bad triple as a few-shot negative in the next dream cycle."""
    conn = hy.conn
    conn.execute("INSERT INTO sessions(id) VALUES ('s1')")
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) "
        "VALUES ('c1', 's1', 1, 1, 'correction_or_preference_trigger', "
        "'the api does NOT use docker, that was a misread of the dockerfile')"
    )
    edge_id = _seed_edge(hy, "api", "uses", "docker")
    with core_db.evidence_mutation(conn):
        conn.execute(
            "INSERT INTO kg_evidence(edge_id, chunk_id, polarity) VALUES (?, 'c1', 1)",
            (edge_id,),
        )

    assert hy.retract_edge("api", "uses", "docker") is True

    rows = conn.execute(
        "SELECT extracted_subject, extracted_predicate, extracted_object, "
        "       chunk_text_snippet, feedback_type "
        "FROM extraction_feedback ORDER BY id"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["extracted_subject"] == "api"
    assert rows[0]["extracted_predicate"] == "uses"
    assert rows[0]["extracted_object"] == "docker"
    assert rows[0]["feedback_type"] == "retracted"
    assert "docker" in rows[0]["chunk_text_snippet"]


def test_retract_without_evidence_writes_no_feedback(hy):
    """An edge with no positive kg_evidence rows leaves extraction_feedback
    empty — there's no chunk text to attach a negative example to."""
    _seed_edge(hy, "ghost_subject", "uses", "ghost_object")

    assert hy.retract_edge("ghost_subject", "uses", "ghost_object") is True
    count = hy.conn.execute(
        "SELECT COUNT(*) AS c FROM extraction_feedback"
    ).fetchone()["c"]
    assert count == 0
