from __future__ import annotations

from hymem.core import db as core_db
from hymem.dreaming import evidence
from hymem.dreaming.phase1_auxiliary import publish_phase1_auxiliaries
from hymem.dreaming.phase3 import decay
from hymem.extraction.producer import register_phase1_generation
from tests.conftest import make_routed_llm


def _publish_current_mentions(hy, chunk_id: str, entities: list[str]) -> None:
    """Publish an exact current Phase-1 mention projection for a fixture."""

    binding = hy._phase1_generation
    assert binding is not None
    generation_key = str(binding["generation_key"])
    cache_key = str(binding["extraction_cache_key"])
    with core_db.transaction(hy.conn):
        register_phase1_generation(hy.conn, binding)
        with core_db.evidence_mutation(hy.conn):
            hy.conn.execute(
                "INSERT OR REPLACE INTO kg_claim_extraction_outcomes("
                "chunk_id,prompt_version,prompt_generation,result_hash,"
                "phase1_generation_key) VALUES (?,?,?,?,?)",
                (
                    chunk_id,
                    cache_key,
                    evidence.prompt_generation(cache_key),
                    evidence.claim_result_hash([]),
                    generation_key,
                ),
            )
        publish_phase1_auxiliaries(
            hy.conn,
            chunk_id=chunk_id,
            phase1_generation_key=generation_key,
            extraction_cache_key=cache_key,
            entity_type_hints={},
            entity_property_hints={},
            entity_mentions=entities,
            markers=[],
        )
        hy.conn.execute(
            "INSERT INTO processed_chunks("
            "chunk_id,prompt_version,phase1_generation_key) VALUES (?,?,?)",
            (chunk_id, cache_key, generation_key),
        )


def test_entity_mentions_populated_after_dream(hy):
    sid = "s_idx"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "We use uv for the local dev environment.")
    hy.log_message(
        sid,
        "user",
        "Yes, we depend on uv now and we don't use Docker for local dev anymore.",
    )
    hy.close_session(sid)

    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
    ]
    hy.set_llm(make_routed_llm(triples, []))
    hy.dream()

    rows = hy.conn.execute(
        "SELECT entity_canonical FROM entity_mentions"
    ).fetchall()
    canonicals = {r["entity_canonical"] for r in rows}
    assert "uv" in canonicals
    assert "local_dev" in canonicals


def test_entity_mentions_deduped_per_chunk(hy):
    sid = "s_dedup"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "We use uv for the local dev environment.")
    hy.log_message(
        sid,
        "user",
        "Yes, we depend on uv now and we don't use Docker for local dev anymore.",
    )
    hy.close_session(sid)

    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
        {"subject": "local_dev", "predicate": "depends_on", "object": "uv", "polarity": 1},
        {"subject": "uv", "predicate": "replaces", "object": "docker", "polarity": 1},
    ]
    hy.set_llm(make_routed_llm(triples, []))
    hy.dream()

    rows = hy.conn.execute(
        "SELECT chunk_id, entity_canonical, COUNT(*) AS n "
        "FROM entity_mentions GROUP BY chunk_id, entity_canonical"
    ).fetchall()
    assert rows
    for r in rows:
        assert r["n"] == 1


def test_phase3_decay_uses_index_for_re_mentioned_topics(hy):
    """Mirrors test_phase3_decay_only_affects_re_mentioned_topics but seeds the
    index manually, exercising the new query path directly."""
    conn = hy.conn

    # Edge whose subject 'app' is NOT mentioned in any recent chunk.
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('app', 'uses', 'postgres', 5, 0, datetime('now', '-90 days'))"
    )
    # Edge whose subject 'api' IS re-mentioned in a recent chunk without
    # reinforcement.
    conn.execute("INSERT INTO sessions(id) VALUES ('s_decay')")
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, salience_reason, text) "
        "VALUES ('c1', 's_decay', 1, 1, 'long_user_turn', 'we redesigned the api')"
    )
    _publish_current_mentions(hy, "c1", ["api"])
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('api', 'uses', 'fastapi', 5, 0, datetime('now', '-90 days'))"
    )

    decay(conn, hy.config)

    rows = {
        (r["subject_canonical"], r["object_canonical"]): r
        for r in conn.execute(
            "SELECT subject_canonical, object_canonical, neg_evidence FROM knowledge_graph"
        ).fetchall()
    }
    assert rows[("app", "postgres")]["neg_evidence"] == 0
    assert rows[("api", "fastapi")]["neg_evidence"] >= 1


def test_backfill_idempotent(hy):
    conn = hy.conn
    conn.execute("INSERT INTO sessions(id) VALUES ('s_bf')")
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, salience_reason, text) "
        "VALUES ('c_bf', 's_bf', 1, 1, 'long_user_turn', 'we use uv for local dev')"
    )
    # Seed alias dictionary so the scan finds something.
    conn.execute(
        "INSERT INTO entity_aliases(alias, canonical) VALUES ('uv', 'uv')"
    )
    conn.execute(
        "INSERT INTO entity_aliases(alias, canonical) VALUES ('local_dev', 'local_dev')"
    )

    # Wipe and call.
    conn.execute("DELETE FROM entity_mentions")
    core_db.backfill_entity_mentions(conn)

    first = conn.execute(
        "SELECT chunk_id, entity_canonical FROM entity_mentions ORDER BY entity_canonical"
    ).fetchall()
    assert len(first) >= 1
    canonicals = {r["entity_canonical"] for r in first}
    assert "uv" in canonicals

    # Second call must be a no-op (table is non-empty).
    core_db.backfill_entity_mentions(conn)
    second = conn.execute(
        "SELECT chunk_id, entity_canonical FROM entity_mentions ORDER BY entity_canonical"
    ).fetchall()
    assert second == first
