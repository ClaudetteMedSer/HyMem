from __future__ import annotations

import json

from hymem.extraction.llm import StubLLMClient
from tests.conftest import make_routed_llm


def _seed_session_with_correction(hy):
    sid = "s1"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "I'll set up Docker for the local dev environment.")
    hy.log_message(
        sid,
        "user",
        "No, actually we don't use Docker for local dev anymore. We switched to uv and system Python.",
    )
    hy.close_session(sid)
    return sid


def test_phase1_extracts_and_writes_evidence(hy):
    _seed_session_with_correction(hy)
    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "Docker", "polarity": -1},
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
    ]
    markers = [{"kind": "preference", "statement": "user prefers uv for Python tooling"}]
    hy.set_llm(make_routed_llm(triples, markers))

    report = hy.dream()
    assert report.chunks_processed >= 1

    rows = hy.conn.execute(
        "SELECT subject_canonical, predicate, object_canonical, pos_evidence, neg_evidence "
        "FROM knowledge_graph ORDER BY object_canonical"
    ).fetchall()
    by_obj = {r["object_canonical"]: r for r in rows}
    assert by_obj["docker"]["neg_evidence"] == 1
    assert by_obj["docker"]["pos_evidence"] == 0
    assert by_obj["uv"]["pos_evidence"] == 1


def test_phase1_is_idempotent(hy):
    _seed_session_with_correction(hy)
    triples = [{"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1}]
    hy.set_llm(make_routed_llm(triples, []))

    hy.dream()
    hy.dream()  # second run must not double-count
    hy.dream()

    row = hy.conn.execute(
        "SELECT pos_evidence, neg_evidence FROM knowledge_graph WHERE object_canonical='uv'"
    ).fetchone()
    assert row["pos_evidence"] == 1


def test_phase2_writes_behavioral_profile_and_insights(hy):
    _seed_session_with_correction(hy)
    triples = [
        {"subject": "local_dev", "predicate": "depends_on", "object": "uv", "polarity": 1},
        {"subject": "ci_pipeline", "predicate": "depends_on", "object": "uv", "polarity": 1},
    ]
    markers = [
        {"kind": "preference", "statement": "user prefers uv for Python tooling"},
        {"kind": "rejection", "statement": "user avoids Docker for local development"},
    ]
    hy.set_llm(make_routed_llm(triples, markers))

    hy.dream()

    user_md = hy.config.user_md_path.read_text(encoding="utf-8")
    assert "Behavioral Profile" in user_md
    assert "uv" in user_md or "Docker" in user_md

    memory_md = hy.config.memory_md_path.read_text(encoding="utf-8")
    assert "Project Insights" in memory_md
    # Hub query should fire because uv is depended on by 2 subjects.
    assert "uv" in memory_md


def test_phase3_decay_only_affects_re_mentioned_topics(hy):
    """Stable facts in dormant topics must NOT decay just because time passed."""
    conn = hy.conn
    # Old edge that was never re-mentioned in any chunk.
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('app', 'uses', 'postgres', 5, 0, datetime('now', '-90 days'))"
    )
    # Old edge whose subject is mentioned in a recent chunk without reinforcement.
    conn.execute(
        "INSERT INTO sessions(id) VALUES ('s_decay')"
    )
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, salience_reason, text) "
        "VALUES ('c1', 's_decay', 1, 1, 'long_user_turn', 'we redesigned the api but didn t change the database choice')"
    )
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('api', 'uses', 'fastapi', 5, 0, datetime('now', '-90 days'))"
    )

    from hymem.dreaming.phase3 import decay
    decay(conn, hy.config)

    rows = {
        (r["subject_canonical"], r["object_canonical"]): r
        for r in conn.execute(
            "SELECT subject_canonical, object_canonical, pos_evidence, neg_evidence, status "
            "FROM knowledge_graph"
        ).fetchall()
    }
    # postgres edge: subject 'app' not mentioned in recent chunk → unchanged.
    assert rows[("app", "postgres")]["neg_evidence"] == 0
    # fastapi edge: subject 'api' mentioned recently without reinforcement → decayed.
    assert rows[("api", "fastapi")]["neg_evidence"] >= 1


def test_phase3_negative_dominance_retracts_gray_zone(hy):
    """Edges where negatives clearly dominate (e.g. pos=1, neg=4) must retract
    even though their smoothed confidence (0.33) is above retract_threshold."""
    conn = hy.conn
    # Classic zombie: pos=0, neg=2 — smoothed 0.25, above 0.15 threshold.
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('hook', 'uses', 'nohup_zombie', 0, 2, datetime('now'))"
    )
    # Gray-zone: pos=1, neg=4 — smoothed 0.33, above threshold but neg dominates.
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('hook', 'uses', 'nohup_grayzone', 1, 4, datetime('now'))"
    )
    # Mixed but not dominated: pos=2, neg=4 — should NOT retract.
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('hook', 'uses', 'mixed_signal', 2, 4, datetime('now'))"
    )

    from hymem.dreaming.phase3 import decay
    decay(conn, hy.config)

    statuses = {
        r["object_canonical"]: r["status"]
        for r in conn.execute(
            "SELECT object_canonical, status FROM knowledge_graph"
        ).fetchall()
    }
    assert statuses["nohup_zombie"] == "retracted"
    assert statuses["nohup_grayzone"] == "retracted"
    assert statuses["mixed_signal"] == "active"


def test_phase3_retraction_feedback_falls_back_to_negative_evidence(hy):
    """Auto-retracted zombie edges only have polarity=-1 evidence rows.
    extraction_feedback must still get populated so the prompt has few-shot
    negatives — the prior bug skipped these entirely."""
    conn = hy.conn
    conn.execute("INSERT INTO sessions(id) VALUES ('s_zombie')")
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) "
        "VALUES ('c_zombie', 's_zombie', 1, 1, 'correction_or_preference_trigger', "
        "'no, the gateway does not run on pid_77')"
    )
    cur = conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('gateway', 'runs_on', 'pid_77', 0, 2, datetime('now'))"
    )
    edge_id = cur.lastrowid
    conn.execute(
        "INSERT INTO kg_evidence(edge_id, chunk_id, polarity) VALUES (?, 'c_zombie', -1)",
        (edge_id,),
    )

    from hymem.dreaming.phase3 import decay
    decay(conn, hy.config)

    row = conn.execute(
        "SELECT extracted_subject, extracted_predicate, extracted_object, "
        "       chunk_text_snippet, feedback_type "
        "FROM extraction_feedback"
    ).fetchone()
    assert row is not None
    assert row["extracted_subject"] == "gateway"
    assert row["extracted_predicate"] == "runs_on"
    assert row["extracted_object"] == "pid_77"
    assert row["feedback_type"] == "retracted"
    assert "pid_77" in row["chunk_text_snippet"]
