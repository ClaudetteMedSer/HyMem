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
from hymem.rules import add_rule, load_rules, retract_rule
from tests.conftest import seed_edge


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


def test_default_off_no_rules_tier(hy):
    """The stock `hy` fixture has rules_enabled=False → tier stays empty."""
    hy.add_rule("always run the tests before pushing")
    assert hy.augment("anything at all").rules == []


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
