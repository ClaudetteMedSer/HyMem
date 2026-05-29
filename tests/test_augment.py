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


def test_working_memory_recalls_turn_before_any_dream(hy):
    # The core gap A1 fixes: a fact stated this session must be recallable via
    # augment() even though no dream has consolidated it into chunks/graph.
    sid = "wm1"
    hy.open_session(sid)
    hy.log_message(sid, "user", "My favorite database is duckdb.")

    ctx = hy.augment("what database do I like?", session_id=sid)

    # No dream ran, so the consolidated tiers stay empty...
    assert ctx.graph_facts == []
    # ...but the raw turn is surfaced via the working-memory tier.
    assert [m.content for m in ctx.recent_turns] == ["My favorite database is duckdb."]
    assert ctx.recent_turns[0].role == "user"
    assert ctx.recent_turns[0].session_id == sid


def test_working_memory_capped_and_ordered_oldest_to_newest(hy):
    sid = "wm2"
    hy.open_session(sid)
    cap = hy.config.working_memory_turns
    total = cap + 5
    for i in range(total):
        hy.log_message(sid, "user", f"turn {i}")

    ctx = hy.augment("anything", session_id=sid)

    # Capped at working_memory_turns.
    assert len(ctx.recent_turns) == cap
    # The most-recent `cap` turns, oldest -> newest.
    expected = [f"turn {i}" for i in range(total - cap, total)]
    assert [m.content for m in ctx.recent_turns] == expected


def test_working_memory_empty_without_session_id(hy):
    # Backward compatibility: augment() with no session_id behaves exactly as
    # before — no working-memory tier.
    sid = "wm3"
    hy.open_session(sid)
    hy.log_message(sid, "user", "something the user said")

    ctx = hy.augment("something")
    assert ctx.recent_turns == []


def test_recent_messages_zero_limit_returns_empty(hy):
    from hymem.session import recent_messages

    sid = "wm4"
    hy.open_session(sid)
    hy.log_message(sid, "user", "a turn")

    assert recent_messages(hy.read_conn, sid, 0) == []
