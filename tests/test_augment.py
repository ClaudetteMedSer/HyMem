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


# --- raw-message FTS tier (message_hits) ----------------------------------


def test_message_hits_recall_raw_turn_across_sessions_before_dream(hy):
    # The gap this closes: a fact stated in some *other* session, never dreamed,
    # is still keyword-recallable — unlike the working-memory tier (active
    # session only) or chunk-FTS (dreamed high-salience spans only).
    hy.open_session("past")
    hy.log_message("past", "user", "We migrated the billing service to CockroachDB.")
    hy.close_session("past")

    # No dream ran, and no session_id is passed (working-memory tier off).
    ctx = hy.augment("what database does billing use?")

    assert ctx.graph_facts == []   # nothing consolidated
    assert ctx.recent_turns == []  # no active session
    hit = next(h for h in ctx.message_hits if "CockroachDB" in h.text)
    assert hit.session_id == "past"
    assert hit.role == "user"
    assert hit.message_id > 0
    assert hit.created_at  # populated so a consumer can prefer recent statements
    assert hit.why_retrieved and hit.why_retrieved[0].startswith("message_fts(")


def test_message_hits_exclude_tool_and_system_turns(hy):
    sid = "roles"
    hy.open_session(sid)
    hy.log_message(sid, "user", "deploy uses terraform")
    hy.log_message(sid, "assistant", "noted, terraform it is")
    hy.log_message(sid, "tool", "terraform plan output terraform terraform")
    hy.log_message(sid, "system", "system note about terraform")

    ctx = hy.augment("terraform")

    roles = {h.role for h in ctx.message_hits}
    assert roles == {"user", "assistant"}  # tool/system never indexed
    assert len(ctx.message_hits) == 2


def test_message_hits_empty_on_no_match(hy):
    sid = "nomatch"
    hy.open_session(sid)
    hy.log_message(sid, "user", "we use redis for caching")

    ctx = hy.augment("quantum chromodynamics unrelated terms")
    assert ctx.message_hits == []


def test_message_hits_disabled_when_top_k_zero(cfg, stub_llm):
    from dataclasses import replace

    from hymem import HyMem

    hy = HyMem(replace(cfg, message_fts_top_k=0), llm=stub_llm)
    try:
        sid = "off"
        hy.open_session(sid)
        hy.log_message(sid, "user", "kafka is the message broker")
        ctx = hy.augment("kafka")
        assert ctx.message_hits == []
    finally:
        hy.close()


def test_message_hits_drop_after_message_pruned(hy):
    # The delete trigger must keep messages_fts in sync so retention doesn't
    # leave orphaned, unjoinable FTS rows.
    sid = "prune"
    hy.open_session(sid)
    mid = hy.log_message(sid, "user", "we standardized on pnpm for the monorepo")
    assert any("pnpm" in h.text for h in hy.augment("pnpm").message_hits)

    hy.conn.execute("DELETE FROM messages WHERE id = ?", (mid,))  # autocommit
    assert hy.augment("pnpm").message_hits == []


# --- MR aggregation (ability="MR") ----------------------------------------


def test_mr_aggregation_counts_all_matches_beyond_top_k(hy_agg):
    # The MR lever: message_fts_top_k=5, but aggregation must return ALL matches
    # with the true total — counting 5 of 9 mentions is the failure we fix.
    sid = "mr"
    hy_agg.open_session(sid)
    assert hy_agg.config.message_fts_top_k == 5
    for i in range(9):
        hy_agg.log_message(sid, "user", f"I added project card number {i} to my gallery")

    ctx = hy_agg.augment("how many project cards did I add to my gallery?", ability="MR")

    assert ctx.total_message_matches == 9
    assert len(ctx.message_hits) == 9
    # Chronological order (created_at, id) -> ascending message ids.
    ids = [h.message_id for h in ctx.message_hits]
    assert ids == sorted(ids)
    assert ctx.message_hits[0].score_kind == "aggregate"
    assert ctx.message_hits[0].why_retrieved[0].startswith("message_fts_aggregate(")


def test_mr_aggregation_cap_limits_rows_not_count(cfg, stub_llm):
    from dataclasses import replace

    from hymem import HyMem

    hy = HyMem(replace(cfg, message_fts_aggregate_cap=3), llm=stub_llm)
    try:
        sid = "cap"
        hy.open_session(sid)
        for i in range(7):
            hy.log_message(sid, "user", f"deployed service {i} to staging")

        ctx = hy.augment("how many deploys to staging?", ability="MR")

        assert ctx.total_message_matches == 7   # exact total regardless of cap
        assert len(ctx.message_hits) == 3       # rows capped
    finally:
        hy.close()


def test_mr_aggregation_stopword_filter_avoids_noise_matches(hy_agg):
    # A message sharing only stopwords with the question must NOT be counted —
    # proves the aggregate query drops do/have/many/etc. before matching.
    sid = "noise"
    hy_agg.open_session(sid)
    hy_agg.log_message(sid, "user", "I do have many of these things in my list")

    ctx = hy_agg.augment("how many widgets do I have?", ability="MR")

    # "widgets" is the only content token; the message has no "widget".
    assert ctx.total_message_matches == 0


def test_mr_aggregation_falls_back_when_query_all_stopwords(hy_agg):
    # If filtering empties the token set, fall back to len>=2 tokens so the
    # query is never empty.
    sid = "fb"
    hy_agg.open_session(sid)
    hy_agg.log_message(sid, "user", "many things happened")

    ctx = hy_agg.augment("how many do I have", ability="MR")

    # Fallback keeps how/many/do/have; "many" matches the message.
    assert ctx.total_message_matches >= 1


def test_ability_none_keeps_default_message_path(hy):
    # Backward compat: no ability -> top-k BM25 path, total stays 0.
    sid = "none"
    hy.open_session(sid)
    for i in range(9):
        hy.log_message(sid, "user", f"card {i} added to gallery")

    ctx = hy.augment("gallery cards")

    assert ctx.total_message_matches == 0
    assert len(ctx.message_hits) <= hy.config.message_fts_top_k


def test_unknown_ability_falls_back_to_default(hy):
    sid = "unk"
    hy.open_session(sid)
    for i in range(9):
        hy.log_message(sid, "user", f"item {i} in the gallery")

    ctx = hy.augment("gallery items", ability="BOGUS")

    assert ctx.total_message_matches == 0
    assert len(ctx.message_hits) <= hy.config.message_fts_top_k


def test_mr_ability_is_case_insensitive(hy_agg):
    sid = "ci"
    hy_agg.open_session(sid)
    for i in range(6):
        hy_agg.log_message(sid, "user", f"deployed build {i}")

    ctx = hy_agg.augment("how many builds deployed?", ability="mr")

    assert ctx.total_message_matches == 6


def test_mr_aggregation_counts_user_turns_only(hy_agg):
    # Assistant echoes match the same terms but must not be counted — the
    # question is about the user's actions, not the assistant's confirmations.
    sid = "ro"
    hy_agg.open_session(sid)
    for i in range(5):
        hy_agg.log_message(sid, "user", f"add widget {i} to the board")
        hy_agg.log_message(sid, "assistant", f"added widget {i} to the board")

    ctx = hy_agg.augment("how many widgets did I add to the board?", ability="MR")

    assert ctx.total_message_matches == 5
    assert len(ctx.message_hits) == 5
    assert all(h.role == "user" for h in ctx.message_hits)


def test_mr_aggregation_dedups_identical_restatements(hy_agg):
    # Literal restatements collapse to one; the genuinely distinct turn stays.
    sid = "dd"
    hy_agg.open_session(sid)
    for _ in range(3):
        hy_agg.log_message(sid, "user", "I deployed the API to prod")
    hy_agg.log_message(sid, "user", "I deployed the API to staging")

    ctx = hy_agg.augment("how many times did I deploy?", ability="MR")

    assert ctx.total_message_matches == 2  # prod (x3 -> 1) + staging
    assert len(ctx.message_hits) == 2


def test_mr_aggregation_keeps_distinct_numbered_events(hy_agg):
    # The under-count trap: near-identical turns differing only by a number are
    # distinct events and must NOT be collapsed by dedup.
    sid = "trap"
    hy_agg.open_session(sid)
    for i in range(6):
        hy_agg.log_message(sid, "user", f"I added card number {i}")

    ctx = hy_agg.augment("how many cards did I add?", ability="MR")

    assert ctx.total_message_matches == 6


def test_mr_aggregation_disabled_by_default(hy):
    # Counting is opt-in: with the default message_fts_aggregate_cap=0,
    # ability="MR" uses the normal top-k path, not the counting path.
    sid = "off"
    hy.open_session(sid)
    assert hy.config.message_fts_aggregate_cap == 0
    for i in range(9):
        hy.log_message(sid, "user", f"card {i} added to gallery")

    ctx = hy.augment("how many cards in the gallery?", ability="MR")

    assert ctx.total_message_matches == 0  # counting path off
    assert len(ctx.message_hits) <= hy.config.message_fts_top_k  # normal path ran


# --- ability="IF" procedure shaping ---------------------------------------


def test_if_ability_widens_procedure_budget(hy):
    # IF pulls a wider procedure set (procedure_top_k_if) than the default
    # fts_top_k, since procedures are the natural fit for step recall.
    sid = "ifp"
    hy.open_session(sid)
    for i in range(8):
        hy.conn.execute(
            "INSERT INTO procedures(id, session_id, name, description, steps) "
            "VALUES (?, ?, ?, ?, '[]')",
            (f"p{i}", sid, f"deploy service {i}", "deploy the service to staging"),
        )

    # Default budget: capped at fts_top_k (5).
    default = hy.augment("how do I deploy the service?")
    assert len(default.procedures) == hy.config.fts_top_k

    # ability="IF": widened to procedure_top_k_if (10) -> all 8 surface.
    iff = hy.augment("what steps to deploy the service?", ability="IF")
    assert len(iff.procedures) == 8
