"""Idea B — mechanical compliance gate for `always_on` Rules (schema v23).

This is the NON-LLM half of the Idea-B validation (additional_planning.md §Idea B):
a rule is a standing behavioral imperative ("never suggest Docker"), so the
contract is *presence*, not a benchmark number. These assertions encode the gate:

  1. `always_on` rules appear in EVERY augment() ctx regardless of query.
  2. `contextual` rules appear IFF their trigger overlaps ctx.matched_entities.
  3. a contradicting rule closes the prior's validity interval (supersession).
  4. retracted / superseded rules never surface.
  5. default OFF: with `rules_enabled=False` the tier stays empty (no behaviour
     change until the box compliance gate clears).
  6. re-asserting an identical rule reinforces (pos++), never duplicates.
  7. a pre-v23 store degrades to [] rather than raising.

The LLM adherence harness (does the model OBEY the injected rule) is the box
gate and lives elsewhere; this file guarantees the mechanism it depends on.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from hymem import HyMem
from hymem.rules import (
    add_rule,
    list_rules,
    load_rules,
    retract_rule,
    route_markers_to_rules,
    rule_scope_for_marker,
)
from tests.conftest import make_routed_llm, seed_edge


@pytest.fixture
def hyr(cfg, stub_llm):
    """A HyMem with the rules tier switched on (default is OFF)."""
    inst = HyMem(replace(cfg, rules_enabled=True), llm=stub_llm)
    yield inst
    inst.close()


def _rule_texts(ctx) -> set[str]:
    return {r.text for r in ctx.rules}


# ── 1. always_on injects into every call ────────────────────────────────────

def test_always_on_rule_injected_regardless_of_query(hyr):
    hyr.add_rule("always run the tests before pushing")
    for q in ("how do I deploy?", "what's the weather", "unrelated nonsense xyz"):
        assert "always run the tests before pushing" in _rule_texts(hyr.augment(q))


def test_default_config_loads_rules(hy):
    """Default flipped ON 2026-07-27 (box adherence gate cleared): the stock `hy`
    fixture (default config) loads the tier, so a rule surfaces on any query."""
    hy.add_rule("always run the tests before pushing")
    assert "always run the tests before pushing" in _rule_texts(hy.augment("anything at all"))


def test_rules_enabled_but_empty_is_inert(hy):
    """Default is ON, but with no rules added the tier is empty — the basis of
    the zero-overhead claim (benchmarks/rules_overhead.py) that makes ON-by-
    default safe on every existing rule-less store."""
    assert hy.augment("anything at all").rules == []


def test_rules_tier_can_be_disabled(cfg, stub_llm):
    """A host can still opt out: rules_enabled=False → tier stays empty even with
    a rule present."""
    off = HyMem(replace(cfg, rules_enabled=False), llm=stub_llm)
    try:
        off.add_rule("always run the tests before pushing")
        assert off.augment("anything at all").rules == []
    finally:
        off.close()


# ── 2. contextual fires only on trigger overlap (load_rules unit) ───────────

def test_contextual_rule_fires_only_on_trigger_overlap(hyr):
    add_rule(hyr.conn, "prefer pytest here", scope="contextual",
             trigger_entities=["pytest"])
    hyr.conn.commit()
    # trigger present → fires
    assert any(r.text == "prefer pytest here"
               for r in load_rules(hyr.conn, ["pytest", "redis"], cap=16))
    # trigger absent → suppressed
    assert not any(r.text == "prefer pytest here"
                   for r in load_rules(hyr.conn, ["redis"], cap=16))
    # no matched entities at all → contextual never fires
    assert load_rules(hyr.conn, [], cap=16) == []


def test_always_on_ranks_before_contextual(hyr):
    add_rule(hyr.conn, "contextual one", scope="contextual",
             trigger_entities=["redis"])
    add_rule(hyr.conn, "always one", scope="always_on")
    hyr.conn.commit()
    loaded = load_rules(hyr.conn, ["redis"], cap=16)
    assert [r.text for r in loaded] == ["always one", "contextual one"]


# ── 3/4. supersession + retraction close the interval ───────────────────────

def test_supersession_closes_prior_and_only_new_surfaces(hyr):
    old = hyr.add_rule("deploy to fly.io")
    hyr.add_rule("deploy to aws now", supersedes=old)
    texts = _rule_texts(hyr.augment("where do we deploy"))
    assert "deploy to aws now" in texts
    assert "deploy to fly.io" not in texts
    # the superseded row is interval-closed, not deleted (bi-temporal history).
    row = hyr.conn.execute(
        "SELECT status, invalid_at FROM rules WHERE id=?", (old,)
    ).fetchone()
    assert row["status"] == "retracted" and row["invalid_at"] is not None


def test_retracted_rule_never_surfaces(hyr):
    rid = hyr.add_rule("never suggest docker")
    assert "never suggest docker" in _rule_texts(hyr.augment("containerize this"))
    retract_rule(hyr.conn, rid)
    hyr.conn.commit()
    assert "never suggest docker" not in _rule_texts(hyr.augment("containerize this"))


# ── 6. reinforce, never duplicate ───────────────────────────────────────────

def test_reassert_reinforces_without_duplicating(hyr):
    first = hyr.add_rule("always run the tests before pushing")
    again = hyr.add_rule("always run the tests before pushing")
    assert first == again                                   # same row
    rows = hyr.conn.execute(
        "SELECT pos_evidence FROM rules WHERE text=?",
        ("always run the tests before pushing",),
    ).fetchall()
    assert len(rows) == 1 and rows[0]["pos_evidence"] == 2   # reinforced


# ── 7. degrade on a pre-v23 store ───────────────────────────────────────────

def test_load_rules_degrades_without_table():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    assert load_rules(conn, ["anything"], cap=16) == []


# ── bad input ───────────────────────────────────────────────────────────────

def test_empty_text_and_bad_scope_rejected(hyr):
    with pytest.raises(ValueError):
        hyr.add_rule("   ")
    with pytest.raises(ValueError):
        add_rule(hyr.conn, "x", scope="bogus")


# ── render + prompt wiring (what the answerer actually sees) ─────────────────
# The compliance harness (and any HyMem.ask() consumer) can only test adherence
# if the rules reach the model. These lock that: rules render FIRST, and the
# system prompt gains the obey-directive iff rules are present.

def test_render_puts_rules_first_and_only_when_present():
    from hymem.query.augment import AugmentedContext
    from hymem.query.ask import render_context
    from hymem.rules import Rule

    empty = render_context(AugmentedContext(), max_chars=0)
    assert "STANDING RULES" not in empty

    ctx = AugmentedContext(
        user_profile=[],
        rules=[Rule(id=1, text="never suggest docker", scope="always_on")],
    )
    block = render_context(ctx, max_chars=0)
    assert "=== STANDING RULES (always follow) ===" in block
    assert "never suggest docker" in block
    # Rules lead the block so tail-truncation can never shed them.
    assert block.startswith("=== STANDING RULES")


def test_system_prompt_gains_directive_only_with_rules():
    from hymem.query.ask import ASK_PROMPT_V1, ASK_RULES_DIRECTIVE, _system_prompt

    assert _system_prompt(False) == ASK_PROMPT_V1          # base untouched
    withr = _system_prompt(True)
    assert withr.startswith(ASK_PROMPT_V1) and ASK_RULES_DIRECTIVE in withr


def test_ask_end_to_end_surfaces_rules(hyr, stub_llm):
    """hy.ask() renders rules into the context it grounds on (StubLLM answer is
    inert; the point is the rules reach the rendered block + Answer.context)."""
    hyr.add_rule("never suggest docker")
    ans = hyr.ask("how should I containerize this app?")
    assert "never suggest docker" in {r.text for r in ans.context.rules}


# ── list_rules (the whole rulebook, trigger-agnostic) ───────────────────────

def test_list_rules_returns_all_active_regardless_of_trigger(hyr):
    hyr.add_rule("always one")
    add_rule(hyr.conn, "ctx one", scope="contextual", trigger_entities=["redis"])
    hyr.conn.commit()
    assert {r.text for r in hyr.rules()} == {"always one", "ctx one"}
    # ordering: always_on first
    assert hyr.rules()[0].text == "always one"


# ── marker → rule routing (write-side extraction) ───────────────────────────

def test_rule_scope_classifier():
    # style is a durable directive, routed on kind alone
    assert rule_scope_for_marker("style", "Write commit messages in imperative mood") == "always_on"
    # rejection routes ONLY on a genuine imperative modal, never on the word
    # "rejects": the extractor writes a one-off ("rejects the LOWER() patch") and
    # a standing avoidance ("rejects MongoDB") in the same present tense, so the
    # token carries no signal (2026-07-27 real-marker precision fix).
    assert rule_scope_for_marker("rejection", "Never use MongoDB") == "always_on"
    assert rule_scope_for_marker("rejection", "The user rejects the LOWER() patch") is None
    # a correction routes only when imperative-shaped
    assert rule_scope_for_marker("correction", "Always run the tests before pushing") == "always_on"
    assert rule_scope_for_marker("correction", "The meeting is Tuesday, not Monday") is None
    # a preference is a taste, never a rule (even if phrased with 'never')
    assert rule_scope_for_marker("preference", "Never uses tabs, prefers spaces") is None


def _dream_with_markers(cfg, stub_llm, markers, *, extraction_on):
    hy = HyMem(
        replace(cfg, rules_enabled=True, rules_extraction_enabled=extraction_on),
        llm=make_routed_llm([], markers),
    )
    hy.log_message("s1", "user",
                   "Correction: never suggest Docker. Always use systemd instead.")
    hy.log_message("s1", "assistant", "Understood.")
    hy.dream()
    return hy


def test_dream_routes_imperative_markers_to_rules(cfg, stub_llm):
    markers = [
        {"kind": "rejection", "statement": "Never suggest Docker"},
        {"kind": "style", "statement": "Write commit messages in the imperative mood"},
        {"kind": "correction", "statement": "The meeting is Tuesday, not Monday"},  # one-off
    ]
    hy = _dream_with_markers(cfg, stub_llm, markers, extraction_on=True)
    try:
        active = hy.rules()
        texts = {r.text for r in active}
        assert "Never suggest Docker" in texts
        assert "Write commit messages in the imperative mood" in texts
        assert "The meeting is Tuesday, not Monday" not in texts   # non-imperative correction
        agent = [r for r in active if r.source == "agent_inferred"]
        assert len(agent) == 2 and all(r.scope == "always_on" for r in agent)
    finally:
        hy.close()


def test_dream_extraction_disabled_mints_no_rules(cfg, stub_llm):
    markers = [{"kind": "rejection", "statement": "Never suggest Docker"}]
    hy = _dream_with_markers(cfg, stub_llm, markers, extraction_on=False)
    try:
        assert hy.rules() == []       # write-side gated off → no agent_inferred rules
    finally:
        hy.close()


# ── candidate-suggestion pathway (read-only, no auto-write) ──────────────────
# The tagger is a high-RECALL detector that over-fires on one-offs, so instead of
# auto-injecting inferred rules (which never cleared the precision gate), we
# SUGGEST candidates and let the confirming human/agent be the precision gate.

class _DurabilityStub:
    """A fake durability tagger: any statement mentioning 'mongo' → the SAME
    canonical rule (so paraphrases across sessions collapse into one candidate
    with recurrence), 'docker' → its own rule, everything else one-off."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, req):
        import json
        self.calls += 1
        body = req.user.split("Markers:", 1)[1].rsplit("Return", 1)[0].strip()
        payload = json.loads(body)
        out = []
        for m in payload:
            s = m["statement"].lower()
            if "mongo" in s:
                out.append({"index": m["index"], "standing": True,
                            "confidence": 0.95, "rule": "Never use MongoDB"})
            elif "docker" in s:
                out.append({"index": m["index"], "standing": True,
                            "confidence": 0.9, "rule": "Never suggest Docker"})
            else:
                out.append({"index": m["index"], "standing": False,
                            "confidence": 0.1, "rule": None})
        return json.dumps(out)


def _seed_marker(conn, kind, statement, session_id, *, consolidated=False):
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (session_id,))
    chunk_id = f"chunk_{session_id}_{abs(hash(statement)) % 10**6}"
    conn.execute(
        "INSERT OR IGNORE INTO chunks(id, session_id, start_message_id, "
        "end_message_id, salience_reason, text) VALUES (?,?,?,?,?,?)",
        (chunk_id, session_id, 0, 0, "test", statement))
    conn.execute(
        "INSERT INTO behavioral_markers(kind, statement, chunk_id, consolidated_at) "
        "VALUES (?,?,?,?)",
        (kind, statement, chunk_id, "2020-01-01T00:00:00" if consolidated else None))


def _suggest(cfg, judge, seed):
    """A HyMem whose markers are seeded directly; returns (inst, candidates)."""
    from hymem.core.db import transaction
    from hymem.rules import suggest_rules_from_markers
    inst = HyMem(replace(cfg, rules_enabled=True), llm=judge)
    with transaction(inst.conn):
        seed(inst.conn)
    # read via the write conn so freshly-seeded rows are visible in one process
    return inst, suggest_rules_from_markers(inst.conn, inst.config, judge)


def test_suggest_groups_paraphrases_and_counts_sessions(cfg):
    judge = _DurabilityStub()

    def seed(conn):
        _seed_marker(conn, "rejection", "The user rejects using MongoDB", "s1")
        _seed_marker(conn, "rejection", "Avoid MongoDB, use Postgres", "s2")
        _seed_marker(conn, "correction", "The meeting is Tuesday not Monday", "s1")

    inst, cands = _suggest(cfg, judge, seed)
    try:
        # the two Mongo paraphrases collapse to ONE candidate over TWO sessions;
        # the one-off correction is not standing → no candidate.
        assert [c.text for c in cands] == ["Never use MongoDB"]
        c = cands[0]
        assert c.marker_count == 2 and c.session_count == 2
        assert c.kinds == ["rejection"] and not c.already_active
        # nothing was persisted — suggestion is read-only.
        assert inst.rules() == []
    finally:
        inst.close()


def test_suggest_flags_already_active_and_ranks_novel_first(cfg):
    judge = _DurabilityStub()

    def seed(conn):
        _seed_marker(conn, "rejection", "avoid mongodb entirely", "s1")   # → already active
        _seed_marker(conn, "rejection", "never suggest docker here", "s1")  # → novel
        _seed_marker(conn, "rejection", "and mongo again", "s2")          # reinforces the active one

    inst = HyMem(replace(cfg, rules_enabled=True), llm=judge)
    try:
        inst.add_rule("Never use MongoDB")           # already in force
        from hymem.core.db import transaction
        with transaction(inst.conn):
            seed(inst.conn)
        cands = inst.suggest_rules()
        by_text = {c.text: c for c in cands}
        assert by_text["Never use MongoDB"].already_active is True
        assert by_text["Never suggest Docker"].already_active is False
        # novel candidate ranks before the already-active one.
        assert cands[0].text == "Never suggest Docker"
    finally:
        inst.close()


def test_suggest_requires_llm_and_respects_limit(cfg, stub_llm):
    # no LLM → explicit error (the tagger is required, like ask()/dream()).
    no_llm = HyMem(replace(cfg, rules_enabled=True), llm=None)
    try:
        with pytest.raises(RuntimeError):
            no_llm.suggest_rules()
    finally:
        no_llm.close()

    judge = _DurabilityStub()

    def seed(conn):
        _seed_marker(conn, "rejection", "avoid mongodb", "s1")
        _seed_marker(conn, "rejection", "never suggest docker", "s1")

    inst = HyMem(replace(cfg, rules_enabled=True), llm=judge)
    try:
        from hymem.core.db import transaction
        with transaction(inst.conn):
            seed(inst.conn)
        assert len(inst.suggest_rules(limit=1)) == 1
    finally:
        inst.close()


def test_suggest_ignores_consolidated_and_empty_is_safe(cfg):
    judge = _DurabilityStub()

    def seed(conn):
        _seed_marker(conn, "rejection", "avoid mongodb", "s1", consolidated=True)

    inst, cands = _suggest(cfg, judge, seed)
    try:
        # a consolidated marker is out of the fresh-signal window → no candidate,
        # and a tagger with nothing to judge is never even called.
        assert cands == []
        assert judge.calls == 0
    finally:
        inst.close()
