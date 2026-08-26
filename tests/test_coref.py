"""E5 anaphora/ellipsis resolver — Campaign E, Step 2.

Two things are under test and they are NOT the same thing:

  * the resolver itself (`rewrite_query`) — trigger set, referent resolution,
    the append-never-replace invariant, EN + NL;
  * the wiring (`augment()`) — that a rewrite reaches every tier, that the
    no-`session_id` / flag-off paths are inert, and that `ctx.coref` records the
    decision either way.

The no-harm control (a self-contained query comes back byte-identical) is
asserted at BOTH levels, because that is the property the E5 gate reads.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from hymem import HyMemConfig
from hymem.extraction.llm import StubLLMClient
from hymem.query.coref import QueryRewrite, rewrite_query
from hymem.session import Message

from tests.conftest import seed_edge


def _turns(*pairs: tuple[str, str]) -> list[Message]:
    """Build a recent-turn window from (role, content) pairs, oldest first."""
    return [
        Message(id=i, session_id="s1", role=role, content=content)
        for i, (role, content) in enumerate(pairs, 1)
    ]


# --- Unit: triggers + resolution (no store) ---------------------------------


def test_pronoun_query_gains_referent(cfg: HyMemConfig) -> None:
    turns = _turns(
        ("user", "I moved the medflow deploy to fly.io last week"),
        ("assistant", "Noted — the medflow rollout is on fly.io now."),
    )
    out = rewrite_query("what did she say about that?", turns, cfg=cfg)
    assert out.changed
    assert out.rule == "pronoun"
    assert "medflow" in out.rewritten.lower()


def test_append_not_replace_preserves_original_bytes(cfg: HyMemConfig) -> None:
    """The additive invariant at query level: every original token survives."""
    turns = _turns(("user", "the medflow migration is scheduled for friday"))
    original = "and that one?"
    out = rewrite_query(original, turns, cfg=cfg)
    assert out.changed
    assert out.rewritten.startswith(original)
    assert out.rewritten != original


def test_self_contained_query_is_byte_identical(cfg: HyMemConfig) -> None:
    turns = _turns(("user", "the medflow migration is scheduled for friday"))
    original = "what did we decide about the postgres connection pool tuning?"
    out = rewrite_query(original, turns, cfg=cfg)
    assert not out.changed
    assert out.rewritten == original
    assert out.rule == "self_contained"


def test_demonstrative_plus_generic_noun_fires(cfg: HyMemConfig) -> None:
    turns = _turns(("user", "we kicked off the medflow rollout yesterday"))
    out = rewrite_query("how is that project going?", turns, cfg=cfg)
    assert out.changed
    assert out.rule == "demonstrative"
    assert "that project" in out.resolved


def test_ellipsis_followup_fires(cfg: HyMemConfig) -> None:
    turns = _turns(("user", "I booked the amsterdam flight for tuesday"))
    out = rewrite_query("the price?", turns, cfg=cfg)
    assert out.changed
    assert out.rule == "ellipsis"


def test_dutch_pronoun_and_ellipsis(cfg: HyMemConfig) -> None:
    turns = _turns(("user", "ik heb de medflow migratie naar vrijdag verzet"))
    pron = rewrite_query("wat zei hij daarover?", turns, cfg=cfg)
    assert pron.changed and pron.rule == "pronoun"
    ell = rewrite_query("en de prijs?", turns, cfg=cfg)
    assert ell.changed and ell.rule == "ellipsis"
    assert ell.rewritten.startswith("en de prijs?")


def test_dutch_demonstrative(cfg: HyMemConfig) -> None:
    turns = _turns(("user", "ik gebruik ruff voor de linting"))
    out = rewrite_query("werkt die tool goed?", turns, cfg=cfg)
    assert out.changed and out.rule == "demonstrative"


def test_no_turns_is_inert(cfg: HyMemConfig) -> None:
    for window in ([], None):
        out = rewrite_query("what about that?", window, cfg=cfg)
        assert not out.changed
        assert out.rule == "no_turns"


def test_window_holding_only_the_query_itself_is_inert(cfg: HyMemConfig) -> None:
    """A host that logs the incoming turn before calling augment() must not have
    the query resolved against itself."""
    out = rewrite_query("what about that?", _turns(("user", "what about that?")),
                        cfg=cfg)
    assert not out.changed
    assert out.rule == "no_turns"


def test_disabled_flag_is_inert(cfg: HyMemConfig) -> None:
    turns = _turns(("user", "the medflow migration is friday"))
    out = rewrite_query("what about that?", turns,
                        cfg=replace(cfg, coref_enabled=False))
    assert not out.changed
    assert out.rule == "disabled"


@pytest.mark.parametrize("bad", [None, 123, b"bytes", ""])
def test_non_string_and_empty_input_never_raise(cfg: HyMemConfig, bad: object) -> None:
    out = rewrite_query(bad, _turns(("user", "medflow is live")), cfg=cfg)
    assert not out.changed
    assert out.rule in {"empty", "non_str"}
    assert isinstance(out.rewritten, str)


def test_referent_window_is_capped(cfg: HyMemConfig) -> None:
    """Only the last `coref_max_turns` turns are searched — a stale entity far
    back in the session must not be pulled in."""
    turns = _turns(
        ("user", "stale_entity_alpha was the old plan"),
        *[("user", f"filler turn {i} about nothing") for i in range(6)],
    )
    out = rewrite_query("what about that?", turns,
                        cfg=replace(cfg, coref_max_turns=2))
    assert "stale_entity_alpha" not in out.rewritten


def test_no_referent_leaves_query_untouched(cfg: HyMemConfig) -> None:
    """A window with nothing resolvable (pure stopwords) yields no rewrite —
    never a rewrite with junk."""
    out = rewrite_query("what about that?", _turns(("user", "ok, and then?")),
                        cfg=cfg)
    assert not out.changed
    assert out.rule == "no_referent"
    assert out.rewritten == "what about that?"


# --- Unit: graph-entity resolution beats salient tokens ---------------------


def test_known_entity_from_graph_is_preferred(hy) -> None:
    """With a store, referents are CANONICAL graph names — the token the graph
    and entity tiers actually index on."""
    seed_edge(hy.conn, "medflow", "deploys_to", "fly.io")
    hy.conn.commit()
    turns = _turns(("user", "the medflow rollout slipped to friday"))
    out = rewrite_query("what about that?", turns, cfg=hy.config, conn=hy.conn)
    assert out.changed
    assert "medflow" in out.rewritten


def test_query_naming_a_known_entity_does_not_fire(hy) -> None:
    # postgres in SUBJECT position: `match_known_entities` only trusts an
    # object-position canonical when it also looks entity-shaped elsewhere.
    seed_edge(hy.conn, "postgres", "runs_on", "fly.io")
    hy.conn.commit()
    turns = _turns(("user", "we also run redis for the cache"))
    original = "is it faster than postgres?"
    out = rewrite_query(original, turns, cfg=hy.config, conn=hy.conn)
    assert not out.changed
    assert out.rewritten == original
    assert out.rule == "self_contained"


# --- Unit: Stage 2 LLM fallback gating -------------------------------------


def test_llm_fallback_not_called_by_default(cfg: HyMemConfig) -> None:
    llm = StubLLMClient(default="what did Sarah say about medflow?")
    out = rewrite_query("what did she say about that?",
                        _turns(("user", "sarah pushed the release")),
                        cfg=cfg, llm=llm)
    assert llm.calls == []
    assert out.rule == "pronoun"  # salient-token fallback, no call


def test_llm_fallback_appends_only_new_terms(cfg: HyMemConfig) -> None:
    llm = StubLLMClient(default="what did Sarah say about the medflow rollout?")
    original = "what did she say about that?"
    out = rewrite_query(
        original, _turns(("user", "ok and then?")),
        cfg=replace(cfg, coref_llm_enabled=True), llm=llm,
    )
    assert len(llm.calls) == 1
    assert out.changed and out.rule == "llm"
    # Append-only holds for Stage 2 too: original prefix intact, model terms added.
    assert out.rewritten.startswith(original)
    assert "medflow" in out.rewritten


def test_llm_fallback_failure_degrades_to_heuristic(cfg: HyMemConfig) -> None:
    class Boom:
        def complete(self, request):  # noqa: ANN001
            raise RuntimeError("backend down")

    turns = _turns(("user", "the medflow migration is friday"))
    out = rewrite_query("what about that?", turns,
                        cfg=replace(cfg, coref_llm_enabled=True), llm=Boom())
    # The entity/salient path still resolves — a coref miss never fails a query.
    assert out.changed
    assert "medflow" in out.rewritten


# --- Wiring: augment() ----------------------------------------------------


def _log(hy, session: str, pairs: list[tuple[str, str]]) -> None:
    hy.log_messages(session, [(role, content, None) for role, content in pairs])


def test_augment_records_coref_and_retrieves_the_antecedent(hy) -> None:
    _log(hy, "s1", [
        ("user", "I moved the medflow deploy to fly.io on friday"),
        ("assistant", "Got it, medflow now deploys to fly.io."),
    ])
    ctx = hy.augment("what about that?", session_id="s1")
    assert ctx.coref is not None
    assert ctx.coref.changed
    assert "medflow" in ctx.coref.rewritten
    # The rewrite reached the raw-message tier: the antecedent turn comes back
    # for a query that names nothing on its own.
    assert any("medflow" in h.text.lower() for h in ctx.message_hits)


def test_augment_without_session_id_is_inert(hy) -> None:
    _log(hy, "s1", [("user", "I moved the medflow deploy to fly.io")])
    ctx = hy.augment("what about that?")
    assert ctx.coref is None


def test_augment_coref_disabled_is_inert(cfg: HyMemConfig, stub_llm: StubLLMClient) -> None:
    from hymem import HyMem

    hy = HyMem(replace(cfg, coref_enabled=False), llm=stub_llm)
    try:
        _log(hy, "s1", [("user", "I moved the medflow deploy to fly.io")])
        ctx = hy.augment("what about that?", session_id="s1")
        assert ctx.coref is None
        assert not any("medflow" in h.text.lower() for h in ctx.message_hits)
    finally:
        hy.close()


def test_augment_self_contained_query_matches_the_disabled_control(
    cfg: HyMemConfig, stub_llm: StubLLMClient
) -> None:
    """No-harm control at the wiring level: for a self-contained query the tiers
    are identical with coref ON and OFF."""
    from hymem import HyMem

    turns = [
        ("user", "I moved the medflow deploy to fly.io on friday"),
        ("user", "the postgres connection pool was raised to 40"),
    ]
    q = "what did we set the postgres connection pool to?"

    def _hits(coref: bool) -> list[tuple[int, str]]:
        # Separate roots: the two arms must not share (and re-append to) one store.
        root = cfg.root / ("on" if coref else "off")
        root.mkdir()
        hy = HyMem(replace(cfg, root=root, coref_enabled=coref), llm=stub_llm)
        try:
            _log(hy, "s1", turns)
            ctx = hy.augment(q, session_id="s1")
            if coref:
                assert ctx.coref is not None and not ctx.coref.changed
            return [(h.message_id, h.text) for h in ctx.message_hits]
        finally:
            hy.close()

    assert _hits(True) == _hits(False)


def test_augment_ability_detection_reads_the_original_query(hy) -> None:
    """The router runs BEFORE the rewrite, so an appended referent clause cannot
    change the detected ability."""
    _log(hy, "s1", [("user", "I added three project cards for medflow")])
    ctx = hy.augment("how many did I add?", session_id="s1")
    assert ctx.detected_ability == "MR"
    assert ctx.coref is not None and ctx.coref.changed


def test_queryrewrite_is_frozen() -> None:
    out = QueryRewrite("q", False, "self_contained")
    with pytest.raises(Exception):
        out.rewritten = "mutated"  # type: ignore[misc]
