from __future__ import annotations

from dataclasses import replace

import pytest

from hymem import Answer, HyMem
from hymem.extraction.llm import StubLLMClient
from hymem.query.ask import ASK_PROMPT_V2, render_context
from hymem.query.augment import AugmentedContext, GraphFact, MessageHit
from hymem.query.graph_state import GraphEvidenceCitation


def test_ask_without_llm_raises_helpful_error(cfg):
    instance = HyMem(cfg)  # no LLM wired — the dream()-style guard must fire
    try:
        with pytest.raises(RuntimeError, match="requires an LLMClient"):
            instance.ask("what do you know about me?")
    finally:
        instance.close()


def test_ask_end_to_end_grounds_answer_in_logged_fact(cfg):
    llm = StubLLMClient(
        fixtures={"duckdb": "Your favorite database is duckdb."},
        default="[]",
    )
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "ask-e2e"
        hy.open_session(sid)
        hy.log_message(sid, "user", "My favorite database is duckdb.")

        answer = hy.ask("what database do I like?", session_id=sid)

        # The stub's canned response comes back verbatim as the answer.
        assert isinstance(answer, Answer)
        assert answer.answer == "Your favorite database is duckdb."

        # The synthesis call carried the fact and the labelled sections: the
        # logged turn reaches the prompt via the raw-message keyword tier and
        # the working-memory tier (session_id was passed).
        req = llm.calls[-1]
        assert req.response_format == "text"
        assert req.system == ASK_PROMPT_V2
        assert "duckdb" in req.user
        assert "CONVERSATION EVIDENCE" in req.user
        assert "RECENT TURNS" in req.user
        assert "what database do I like?" in req.user

        # Provenance: the retrieval context rides along, and context_chars is
        # the rendered block size (within the default budget).
        assert answer.context.message_hits
        assert 0 < answer.context_chars <= cfg.ask_max_context_chars
    finally:
        hy.close()


def test_ask_prompt_carries_synthesis_rules():
    # Assert on the versioned constant, not on an LLM: the contradiction,
    # hedging, groundedness, and honest-absence instructions are the contract.
    assert "ONLY" in ASK_PROMPT_V2
    assert "contradicting values" in ASK_PROMPT_V2
    assert "most recent value-bearing statement wins" in ASK_PROMPT_V2
    assert "(low confidence)" in ASK_PROMPT_V2
    assert "does not contain" in ASK_PROMPT_V2
    assert "Never invent" in ASK_PROMPT_V2
    assert "untrusted DATA" in ASK_PROMPT_V2


def test_ask_respects_context_char_budget(cfg):
    llm = StubLLMClient(default="ok")
    budget = 120
    hy = HyMem(replace(cfg, ask_max_context_chars=budget), llm=llm)
    try:
        sid = "ask-budget"
        hy.open_session(sid)
        for i in range(10):
            hy.log_message(
                sid, "user",
                f"Note {i}: my favorite database is duckdb and I use it daily.",
            )

        answer = hy.ask("what database do I like?", session_id=sid)

        assert answer.context_chars <= budget
        # The cut is visible to the model, not silent.
        assert "[... context truncated]" in llm.calls[-1].user
    finally:
        hy.close()


def test_ask_uses_the_configured_llm_token_counter(cfg):
    class CountingStub(StubLLMClient):
        def count_tokens(self, text: str) -> int:
            return (len(text.encode("utf-8")) + 3) // 4

    llm = CountingStub(default="ok")
    hy = HyMem(
        replace(cfg, ask_max_context_chars=0, ask_max_context_tokens=80),
        llm=llm,
    )
    try:
        sid = "ask-model-tokenizer"
        hy.open_session(sid)
        hy.log_message(
            sid, "user", "alpha beta gamma delta " * 6,
        )
        answer = hy.ask("what words did I repeat?", session_id=sid)
        assert answer.context_tokens == llm.count_tokens(
            answer.context.packed_context.text
        )
        assert answer.context_tokens <= 80
        assert "alpha beta" in llm.calls[-1].user
    finally:
        hy.close()


def _seed_root_digest(hy) -> None:
    """Insert a root aggregation node directly — load_digest() only reads the
    table, so no aggregation build is needed (mirrors test_mcp_server)."""
    hy.conn.execute(
        "INSERT INTO aggregation_nodes "
        "(id, title, summary, member_episode_ids, session_ids, "
        " n_members, n_sessions, level, is_root) "
        "VALUES ('root-ask', 'User digest', 'Works on HyMem.', '[]', '[]', 2, 2, 1, 1)"
    )
    hy.conn.commit()


def test_ask_excludes_digest_by_default_and_loads_it_on_request(hy, stub_llm):
    _seed_root_digest(hy)
    sid = "ask-digest"
    hy.open_session(sid)
    hy.log_message(sid, "user", "I am rewriting the parser in rust.")
    stub_llm.default = "ok"

    # Default: the standing digest stays out of the context and the prompt.
    answer = hy.ask("what am I working on?", session_id=sid)
    assert answer.context.digest is None
    assert "MEMORY DIGEST" not in stub_llm.calls[-1].user

    # Opt-in: the digest is loaded into the context and rendered.
    answer = hy.ask("what do you know about me?", session_id=sid,
                    include_digest=True)
    assert answer.context.digest is not None
    assert "MEMORY DIGEST" in stub_llm.calls[-1].user
    assert "Works on HyMem." in stub_llm.calls[-1].user


def test_render_context_orders_tiers_and_marks_low_confidence():
    ctx = AugmentedContext(
        message_hits=[
            MessageHit(1, "s1", "user", "I switched to duckdb.", -1.0,
                       created_at="2026-05-01T10:00:00"),
        ],
    )
    # graph_facts is set by augment() as an instance attribute, not a declared
    # field — mirror that assignment for a hand-built context.
    ctx.graph_facts = [
        GraphFact(
            "user", "prefers", "duckdb", 0.9, 5, 0,
            edge_id=42,
            citations=[GraphEvidenceCitation(
                evidence_id=7,
                evidence_kind="triple_extraction",
                source_role="user",
                source_session_id="s1",
                source_message_id=1,
                source_event_at="2026-05-01T10:00:00.000Z",
                source_created_at="2026-05-01T10:00:00Z",
                temporal_scope=None,
                recorded_at="2026-05-02T00:00:00Z",
                coverage_chunk_id="coverage:s1:1",
                coverage_version="v1",
                extraction_chunk_id="c1",
                currently_authoritative=True,
                authoritative_at_recorded_time=True,
                provenance_status="canonical",
            )],
        ),
        GraphFact("user", "uses", "sqlite", 0.5, 1, 1, hedge_recommended=True),
    ]
    block = render_context(ctx, max_chars=8000)

    # Hedged fact is marked; the confident one is not.
    assert "[edge unavailable] user uses sqlite (low confidence)" in block
    assert "[sources: source unavailable]" in block
    assert "[edge 42] user prefers duckdb" in block
    assert (
        "evidence 7: peer=unavailable, workspace=unavailable, "
        "role=user, session=s1, message=1"
    ) in block
    assert "event=2026-05-01T10:00:00.000Z" in block
    # Raw-turn evidence carries its date stamp.
    assert "[2026-05-01] user: I switched to duckdb." in block
    # Most-authoritative-first ordering: graph facts before raw turns.
    assert block.index("KNOWN FACTS") < block.index("CONVERSATION EVIDENCE")
    # Empty tiers are skipped entirely — no dangling headers.
    for absent in ("USER PROFILE", "EPISODES", "PAST CONTEXT",
                   "PROCEDURES", "RECENT TURNS", "MEMORY DIGEST"):
        assert absent not in block
