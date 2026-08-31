"""Offline tests for the BEAM adapter's episode-tier answer-bearing pre-check.

This instrument exists because the LME granularity guard tied, and the keep-db
probe showed why: both arms saturate the episode cap, so the lever changes
episode CONTENT only -- and on LME the tier never carried the answer, giving the
lever no path to the score. BEAM's EO/SUM abilities are where that could differ.

The tests below are mostly about the instrument's ability to be WRONG. The same
readout that motivated it also contained a structural zero read as a result
("recall_tier: episode 0", where `episode` was never an emittable value), so
what is asserted here is:

  * it CAN fire (a measure that cannot return a hit measures nothing);
  * it distinguishes "tier absent" from "tier present and empty" -- None vs 0.0;
  * its positive control (the message tier) is computed the same way, so a dead
    measure is visible as both columns collapsing rather than as a finding;
  * the tier names it counts are the names the adapter actually emits, so a
    rename cannot silently zero a column.

Skipped where `requests` is absent (the adapter imports it at module scope).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
from beam_adapter import (  # noqa: E402
    _content_tokens,
    _gold_tokens,
    _tier_coverage,
    episode_probe,
)

QUESTION = "What order did the user complete the migration milestones in?"
IDEAL = ("The user first drafted the schema migration, then benchmarked the "
         "aggregation rebuild, and finally flipped the granularity default.")


def _mem(tier: str, content: str) -> dict:
    return {"type": tier, "content": content, "confidence": 0.8}


_ANSWER_BEARING = _mem(
    "episode",
    "Migration work: drafted schema, benchmarked aggregation rebuild, "
    "then flipped the granularity default.")
_NARRATIVE = _mem(
    "episode", "Session summary: the user and assistant had a long discussion.")


def test_the_measure_can_actually_fire():
    """Positive control for the instrument itself.

    Without this, every other assertion is satisfied by a measure that returns
    nothing on every input -- which is precisely the failure being guarded.
    """
    probe = episode_probe([_ANSWER_BEARING, _mem("message_hit", "unrelated")],
                          QUESTION, IDEAL)
    assert probe["cov_episodes"] is not None and probe["cov_episodes"] > 0.5
    assert probe["cov_messages"] == 0.0


def test_narrative_episodes_read_low_against_a_healthy_control():
    """The LME shape: episodes carry the session's gist, messages carry the answer."""
    probe = episode_probe(
        [_NARRATIVE,
         _mem("message_hit", "I drafted the schema, benchmarked the aggregation "
                             "rebuild, then flipped the granularity default.")],
        QUESTION, IDEAL)
    assert probe["cov_episodes"] < 0.3
    assert probe["cov_messages"] > 0.5


def test_absent_tier_is_none_not_zero():
    """"Not retrieved" and "retrieved and carried nothing" must not share an
    encoding -- conflating them is how a structural zero passes for a finding."""
    probe = episode_probe([_mem("message_hit", "x y z")], QUESTION, IDEAL)
    assert probe["n_episodes"] == 0
    assert probe["cov_episodes"] is None
    assert probe["gold_in_episodes"] is None
    # ... while a present-but-empty tier reports a real 0.0.
    present = episode_probe([_NARRATIVE], QUESTION, IDEAL)
    assert present["n_episodes"] == 1 and present["cov_episodes"] == 0.0


def test_no_distinctive_gold_tokens_is_unmeasurable_not_zero():
    probe = episode_probe([_mem("episode", "anything at all")],
                          "What did they do?", "They did it.")
    assert probe["n_gold_tokens"] == 0
    assert probe["cov_episodes"] is None


def test_question_terms_are_subtracted_from_the_gold_set():
    """Retrieval selected these memories BY the question, so question terms are
    covered by every tier for free; leaving them in measures the retriever."""
    raw, net = _content_tokens(IDEAL), _gold_tokens(IDEAL, QUESTION)
    assert net < raw
    assert "migration" in raw and "migration" not in net


def test_coverage_is_none_on_an_empty_tier_and_a_bare_denominator():
    assert _tier_coverage(set(), ["text"]) is None
    assert _tier_coverage({"alpha"}, []) is None
    assert _tier_coverage({"alpha"}, ["   ", ""]) is None
    assert _tier_coverage({"alpha", "beta"}, ["alpha only"]) == 0.5


def test_probe_does_not_mutate_the_reader_input():
    """It is a recording instrument; the run must be identical with it present."""
    memories = [_ANSWER_BEARING, _mem("message_hit", "unrelated")]
    before = [dict(m) for m in memories]
    episode_probe(memories, QUESTION, IDEAL)
    assert memories == before


def test_counted_tier_names_are_the_names_the_adapter_emits():
    """A tier rename would silently zero a column and read as 'tier absent'.

    Asserted against the adapter source rather than a hand-kept list, so the
    test fails on the rename instead of on the next analysis built over it.
    """
    src = (_BENCH / "beam_adapter.py").read_text()
    emitted = set(re.findall(r'"type":\s*"([a-z_]+)"', src))
    for tier in ("episode", "message_hit", "procedure", "fts_hit", "graph_fact",
                 "recent"):
        assert tier in emitted, f"probe counts a tier the adapter never emits: {tier}"


def test_every_probe_key_is_present_on_every_question():
    """A missing key would make the per-ability readout skip rows unevenly,
    turning a coverage question into a survivorship one."""
    expected = {
        "n_memories", "n_episodes", "n_messages", "n_procedures", "n_fts",
        "n_graph", "n_recent", "n_gold_tokens", "cov_episodes", "cov_messages",
        "gold_in_episodes", "gold_in_messages",
    }
    for memories in ([], [_NARRATIVE], [_ANSWER_BEARING, _mem("recent", "hi")]):
        assert set(episode_probe(memories, QUESTION, IDEAL)) == expected


def test_both_decision_levers_are_pinned_explicitly(tmp_path, monkeypatch):
    """The pins must be PASSED, not inherited.

    `episode_granularity_enabled` currently agrees with the library default, so
    asserting the resulting value would pass just as well with no pin at all --
    and would keep passing right up until the default flipped and moved BEAM off
    its baseline silently. What is asserted instead is that the adapter names
    both levers itself, which is the property that survives a default change.
    """
    import hymem

    import beam_adapter

    seen: dict = {}
    real = hymem.HyMemConfig

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(hymem, "HyMemConfig", spy)
    adapter = beam_adapter.HyMemAdapter(tmp_path / "hymem.sqlite", api_key="unused")
    adapter.open()
    try:
        assert seen["aggregation_nodes_enabled"] is False
        assert seen["episode_granularity_enabled"] is False
    finally:
        adapter.close()
