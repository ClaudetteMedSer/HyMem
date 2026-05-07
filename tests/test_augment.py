from __future__ import annotations

from tests.conftest import make_routed_llm


def test_augment_returns_user_md_memory_md_and_graph_facts(hy):
    sid = "s1"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "We could use Docker for the local dev environment.")
    hy.log_message(sid, "user",
        "No, we use uv and system Python for local dev. Don't suggest Docker.")
    hy.close_session(sid)

    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
        {"subject": "local_dev", "predicate": "uses", "object": "Docker", "polarity": -1},
    ]
    markers = [{"kind": "rejection", "statement": "user avoids Docker for local development"}]
    hy.set_llm(make_routed_llm(triples, markers))
    hy.dream()

    ctx = hy.augment("Should I containerize the dev setup with docker?")

    assert "Behavioral Profile" in ctx.user_md
    assert "Project Insights" in ctx.memory_md
    assert "docker" in ctx.matched_entities

    # Graph facts should surface the rejection / negative evidence.
    facts_by_obj = {(f.subject, f.object): f for f in ctx.graph_facts}
    assert ("local_dev", "docker") in facts_by_obj
    docker_fact = facts_by_obj[("local_dev", "docker")]
    assert docker_fact.neg_evidence >= 1


def test_augment_without_dreaming_still_returns_empty_context(hy):
    ctx = hy.augment("hello world")
    assert ctx.matched_entities == []
    assert ctx.graph_facts == []
    assert ctx.fts_hits == []
