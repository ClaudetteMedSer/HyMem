from __future__ import annotations

from tests.conftest import make_routed_llm


def test_full_dreaming_cycle_with_embeddings(hy_with_embed):
    sid = "integration-1"
    hy_with_embed.open_session(sid)
    hy_with_embed.log_message(
        sid, "assistant", "I'll containerize the dev setup with Docker."
    )
    hy_with_embed.log_message(
        sid, "user",
        "No, we use uv and system Python for local dev — please don't suggest Docker.",
    )
    hy_with_embed.close_session(sid)

    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
        {"subject": "local_dev", "predicate": "uses", "object": "Docker", "polarity": -1},
    ]
    markers = [{"kind": "rejection", "statement": "user avoids Docker for local development"}]
    hy_with_embed.set_llm(make_routed_llm(triples, markers))

    report = hy_with_embed.dream()

    assert report.chunks_processed >= 1
    assert report.triples_extracted >= 1
    assert report.markers_extracted >= 1
    assert report.chunks_embedded >= 1

    # Graph populated.
    kg_count = hy_with_embed.conn.execute(
        "SELECT COUNT(*) AS c FROM knowledge_graph"
    ).fetchone()["c"]
    assert kg_count >= 1

    # Markdown files written.
    user_md = hy_with_embed.config.user_md_path.read_text(encoding="utf-8")
    memory_md = hy_with_embed.config.memory_md_path.read_text(encoding="utf-8")
    assert "Behavioral Profile" in user_md
    assert "Project Insights" in memory_md

    # dream_runs row exists.
    runs = hy_with_embed.recent_dream_runs()
    assert len(runs) >= 1
    assert runs[0]["ended_at"] is not None
    assert runs[0]["chunks_embedded"] >= 1

    # chunk_embeddings populated.
    emb_count = hy_with_embed.conn.execute(
        "SELECT COUNT(*) AS c FROM chunk_embeddings"
    ).fetchone()["c"]
    assert emb_count >= 1


def test_retract_after_extraction_excludes_from_augment(hy):
    sid = "integration-2"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "Let's plan the deployment for med_flow.")
    hy.log_message(
        sid, "user",
        "MedFlow depends on Redis for the queue. Make sure to provision that.",
    )
    hy.close_session(sid)

    triples = [
        {"subject": "med_flow", "predicate": "depends_on", "object": "redis", "polarity": 1},
    ]
    hy.set_llm(make_routed_llm(triples, []))
    hy.dream()

    # Confirm extracted.
    ctx_before = hy.augment("tell me about med_flow and redis")
    assert any(
        f.subject == "med_flow" and f.predicate == "depends_on" and f.object == "redis"
        for f in ctx_before.graph_facts
    )

    # Retract and verify gone.
    assert hy.retract_edge("med_flow", "depends_on", "redis") is True
    ctx_after = hy.augment("tell me about med_flow and redis")
    for f in ctx_after.graph_facts:
        assert not (
            f.subject == "med_flow" and f.predicate == "depends_on" and f.object == "redis"
        )


def test_decay_then_retract_workflow(hy):
    """An edge that decayed below threshold is retracted; explicit retract is idempotent."""
    from hymem.dreaming import phase3

    # Seed an old, low-confidence edge with a recent mention to trigger decay.
    hy.conn.execute(
        "INSERT INTO sessions(id) VALUES ('decay-sess')"
    )
    hy.conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, salience_reason, text) "
        "VALUES ('c-decay', 'decay-sess', 1, 1, 'long_user_turn', "
        "'we redesigned the api but didnt change the database choice')"
    )
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('api', 'uses', 'fastapi', 0, 5, datetime('now', '-90 days'))"
    )

    phase3.decay(hy.conn, hy.config)

    row = hy.conn.execute(
        "SELECT status FROM knowledge_graph "
        "WHERE subject_canonical='api' AND object_canonical='fastapi'"
    ).fetchone()
    assert row["status"] == "retracted"

    # Explicit retract on already-retracted edge → no match (idempotent).
    assert hy.retract_edge("api", "uses", "fastapi") is False
    row = hy.conn.execute(
        "SELECT status FROM knowledge_graph "
        "WHERE subject_canonical='api' AND object_canonical='fastapi'"
    ).fetchone()
    assert row["status"] == "retracted"


def test_alias_then_extract_collapses_to_single_node(hy):
    hy.register_alias("MedFlow", "med_flow")

    sid = "alias-int"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "MedFlow uses postgres heavily for joins.")
    hy.log_message(
        sid, "user",
        "Yes, med_flow uses postgres for the analytics workload too — same DB.",
    )
    hy.close_session(sid)

    # The LLM (stub) returns triples with surface form 'MedFlow'; canonicalization
    # should resolve via the alias to 'med_flow'.
    triples = [
        {"subject": "MedFlow", "predicate": "uses", "object": "postgres", "polarity": 1},
        {"subject": "med_flow", "predicate": "uses", "object": "postgres", "polarity": 1},
    ]
    hy.set_llm(make_routed_llm(triples, []))
    hy.dream()

    rows = hy.conn.execute(
        "SELECT subject_canonical, object_canonical, pos_evidence "
        "FROM knowledge_graph WHERE predicate='uses' AND object_canonical='postgres'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["subject_canonical"] == "med_flow"
    # Both extractions must merge onto the same edge.
    assert rows[0]["pos_evidence"] >= 1
