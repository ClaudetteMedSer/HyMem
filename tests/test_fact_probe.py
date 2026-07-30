"""Offline tests for the E1 front-run probe (`benchmarks/fact_probe.py`).

No LLM, no network, no dataset file: everything here runs on synthetic fixtures.
Two things are pinned, and they are the two ways a probe lies:

  * the SELECTION rule — if it drifts, the gate is measured on the wrong
    questions and the number is meaningless (the readside §2.1 rule is
    reproduced exactly: category + wrong + recall_ceiling + no "none" tier);
  * the PLUMBING — selection → extraction → FTS index → gold containment. The
    `--sim` arm exists precisely so this path is exercisable before any spend,
    and the explicit-rowid FTS insert is asserted because a drifted external-
    content rowid is the trap that makes an index probe return a confident
    constant.

The gate ARITHMETIC is pinned too, including the case that matters most: three
of four criteria plus a missing faithfulness hand-score must read INCOMPLETE,
never PASS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from fact_probe import (  # noqa: E402
    _MAX_FACT_CHARS,
    _MAX_FACTS_PER_SESSION,
    index_facts,
    open_fact_index,
    run_question,
    search_facts,
    select_probe_sets,
    sim_extract,
    summarize,
    validate_facts,
)


# ── Selection ───────────────────────────────────────────────────────────────

def _row(qid: str, *, qtype: str = "multi-session", correct: bool = False,
         ceiling: bool | None = True, tiers: list[str] | None = None) -> dict:
    return {"question_id": qid, "question_type": qtype, "correct": correct,
            "recall_ceiling": ceiling,
            "gold_turn_tiers": ["message"] if tiers is None else tiers}


@pytest.fixture
def run() -> dict:
    return {"per_question": [
        # Synthesis misses — the target set.
        _row("ms-miss-1"),
        _row("ms-miss-2", tiers=["message", "fts"]),
        _row("ms-miss-3", tiers=["both"]),
        # Excluded: a gold turn reached NO tier (a floor row, not synthesis).
        _row("ms-floor", tiers=["message", "none"]),
        # Excluded: gold never entered the pool (a retrieval miss).
        _row("ms-retrieval", ceiling=False),
        # Excluded: unknown ceiling (uninstrumented row).
        _row("ms-unknown", ceiling=None),
        # Excluded: right answer — these are the control pool.
        _row("ms-hit-1", correct=True),
        _row("ms-hit-2", correct=True),
        _row("ms-hit-3", correct=True),
        _row("ms-hit-4", correct=True),
        # Excluded: other categories, and abstention (the `_abs` suffix is part
        # of question_type, so it can never match the answerable category).
        _row("tr-miss", qtype="temporal-reasoning"),
        _row("ms-abs-miss", qtype="multi-session_abs"),
    ]}


def test_selection_recovers_only_synthesis_misses(run: dict) -> None:
    misses, control, diag = select_probe_sets(run, seed=0)
    assert misses == ["ms-miss-1", "ms-miss-2", "ms-miss-3"]
    assert set(control) <= {"ms-hit-1", "ms-hit-2", "ms-hit-3", "ms-hit-4"}
    assert len(control) == len(misses)  # equal-sized control
    assert diag["n_misses"] == 3
    assert diag["floor_excluded"] == 1
    assert diag["retrieval_excluded"] == 1
    assert diag["category_rows"] == 10  # every `multi-session` row, `_abs` excluded


def test_control_sample_is_seed_stable(run: dict) -> None:
    """Paired arms across runs: the same seed must draw the same control."""
    a = select_probe_sets(run, seed=0)[1]
    b = select_probe_sets(run, seed=0)[1]
    c = select_probe_sets(run, seed=7)[1]
    assert a == b
    assert sorted(a) != sorted(c) or len(a) == 4  # differs unless the pool is exhausted


def test_selection_honors_category(run: dict) -> None:
    misses, _, diag = select_probe_sets(run, category="temporal-reasoning")
    assert misses == ["tr-miss"]
    assert diag["category"] == "temporal-reasoning"


def test_uninstrumented_run_raises_instead_of_reporting_zero(run: dict) -> None:
    """An uninstrumented source would yield an empty set that reads as 'no
    misses' — the exact silent failure the guard exists to prevent."""
    stripped = {"per_question": [
        {k: v for k, v in r.items() if k != "gold_turn_tiers"}
        for r in run["per_question"]
    ]}
    with pytest.raises(ValueError, match="gold_turn_tiers"):
        select_probe_sets(stripped)
    with pytest.raises(ValueError, match="per_question"):
        select_probe_sets({})


# ── Extraction validation ───────────────────────────────────────────────────

def test_validate_facts_accepts_a_clean_array() -> None:
    facts = validate_facts(
        '[{"text": "Atta moved the MedFlow deploy to fly.io", '
        '"date": "2023-11-30", "entities": ["MedFlow", "fly.io"]}]',
        session_date="2023-12-01T10:00:00",
    )
    assert len(facts) == 1
    assert facts[0]["date"] == "2023-11-30"
    assert facts[0]["entities"] == ["MedFlow", "fly.io"]


def test_validate_facts_tolerates_prose_wrapping() -> None:
    facts = validate_facts(
        'Here are the facts:\n[{"text": "the pool was raised to 40"}]\nDone.',
        session_date=None,
    )
    assert [f["text"] for f in facts] == ["the pool was raised to 40"]


@pytest.mark.parametrize("raw", ["", "not json", "{}", "[]", '[{"text": "  "}]', None, 42])
def test_validate_facts_drops_unusable_output(raw: object) -> None:
    assert validate_facts(raw, session_date=None) == []


def test_validate_facts_enforces_caps_and_dates() -> None:
    items = [{"text": f"fact number {i} about the postgres pool", "date": "friday"}
             for i in range(20)]
    facts = validate_facts(items, session_date="2023-11-30T09:00:00")
    # Cap truncates a runaway response BEFORE it can inflate the density number.
    assert len(facts) == _MAX_FACTS_PER_SESSION
    # A malformed date is dropped and the session date stands in; the fact is kept.
    assert all(f["date"] == "2023-11-30" for f in facts)
    long_fact = validate_facts([{"text": "x" * 2000}], session_date=None)
    assert len(long_fact[0]["text"]) == _MAX_FACT_CHARS


def test_no_session_date_leaves_the_fact_undated() -> None:
    """The date field must never be guessed — undated is a valid fact."""
    facts = validate_facts([{"text": "the pool was raised to 40"}], session_date=None)
    assert facts[0]["date"] is None


# ── Fact index ──────────────────────────────────────────────────────────────

def test_index_and_search_round_trip() -> None:
    conn = open_fact_index()
    index_facts(conn, "s1", [{"text": "Atta moved the MedFlow deploy to fly.io",
                              "date": "2023-11-30", "entities": ["medflow"]}])
    index_facts(conn, "s2", [{"text": "the postgres connection pool was raised to 40",
                              "date": None, "entities": []}])
    hits = search_facts(conn, "what did we set the postgres pool to?", top_k=5)
    assert hits
    assert "postgres" in hits[0]["text"]
    assert hits[0]["session_id"] == "s2"
    assert search_facts(conn, "unrelated kangaroo taxonomy", top_k=5) == []
    conn.close()


def test_fts_rowid_stays_pinned_to_its_content_row() -> None:
    """The external-content-rowid trap: a shadow whose rowids drift returns a
    confident constant. Every hit's text must equal its own content row's text."""
    conn = open_fact_index()
    for i in range(12):
        index_facts(conn, f"s{i}", [{"text": f"session {i} discussed widget alpha{i}",
                                     "date": None, "entities": []}])
    for i in range(12):
        hits = search_facts(conn, f"alpha{i}", top_k=5)
        assert hits, f"alpha{i} not found"
        row = conn.execute("SELECT text, session_id FROM facts WHERE id = ?",
                           (hits[0]["id"],)).fetchone()
        assert row["text"] == hits[0]["text"]
        assert row["session_id"] == f"s{i}"
    conn.close()


def test_search_ignores_untokenizable_queries() -> None:
    conn = open_fact_index()
    index_facts(conn, "s1", [{"text": "a fact", "date": None, "entities": []}])
    assert search_facts(conn, "?! ...", top_k=5) == []
    conn.close()


# ── End-to-end plumbing (the --sim arm) ─────────────────────────────────────

GOLD_TURN = ("I finally raised the postgres connection pool to 40 after the "
             "checkout timeouts last friday")


def _q_data() -> dict:
    return {
        "question_id": "ms-miss-1",
        "question": "what did I raise the postgres connection pool to?",
        "haystack_session_ids": ["s0", "s1"],
        "haystack_dates": ["2023/11/29 (Wed) 10:00", "2023/11/30 (Thu) 11:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "We talked about the sourdough starter "
                                         "and the weekend plans in detail here."},
             {"role": "assistant", "content": "Sounds good."}],
            [{"role": "user", "content": GOLD_TURN, "has_answer": True},
             {"role": "assistant", "content": "Noted."}],
        ],
    }


def test_sim_run_finds_gold_and_spends_nothing() -> None:
    out = run_question(_q_data(), llm=None, sim=True, max_sessions=0)
    assert out["error"] is None
    assert out["calls"] == 0            # --sim never calls a model
    assert out["gold_turns"] == 1
    assert out["gold_in_facts"] is True
    assert out["sessions_processed"] == 2
    assert len(out["retrieved"]) <= 5
    # The dump carries the source turns alongside the facts so the faithfulness
    # hand-score is self-contained.
    assert all("source_turns" in d for d in out["dump"])


def test_sim_run_misses_when_no_fact_carries_the_gold() -> None:
    """The gold-check must be able to FAIL — a probe that always says yes is the
    other way an index probe lies. A query sharing no token with any fact
    retrieves nothing, so containment must report False."""
    q = _q_data()
    q["question"] = "kangaroo marsupial taxonomy classification"
    out = run_question(q, llm=None, sim=True, max_sessions=0)
    assert out["retrieved"] == []
    assert out["gold_in_facts"] is False


def test_max_sessions_keeps_the_most_recent(monkeypatch) -> None:
    q = _q_data()
    out = run_question(q, llm=None, sim=True, max_sessions=1)
    assert out["sessions_processed"] == 1
    # The gold sits in the LAST session, so a recency cap of 1 still reaches it —
    # and the cap is label-free (it never consults which session holds gold).
    assert out["gold_in_facts"] is True


def test_sim_extract_skips_smalltalk_and_assistant_turns() -> None:
    facts = sim_extract(
        [{"role": "user", "content": "ok"},
         {"role": "assistant", "content": "A long assistant answer " * 10},
         {"role": "user", "content": GOLD_TURN}],
        session_date="2023-11-30T00:00:00",
    )
    assert len(facts) == 1
    assert facts[0]["date"] == "2023-11-30"


# ── Gate arithmetic ─────────────────────────────────────────────────────────

def _rows(n: int, covered: int, per_session: int) -> list[dict]:
    return [{"error": None, "gold_in_facts": i < covered,
             "facts_per_session": [per_session] * 3, "facts_total": per_session * 3,
             "parse_failures": 0, "calls": 3} for i in range(n)]


def test_gate_passes_only_with_all_four_criteria() -> None:
    misses, ctrl = _rows(10, 7, 5), _rows(10, 10, 6)
    assert summarize(misses, ctrl, 0.95)["verdict"] == "PASS"
    # Density below the pre-registered 60% → FAIL however good the rest is.
    assert summarize(_rows(10, 5, 5), ctrl, 0.99)["verdict"] == "FAIL"
    # Over-extraction on the misses (median > 8/session) → FAIL.
    assert summarize(_rows(10, 9, 12), ctrl, 0.99)["verdict"] == "FAIL"
    # Systematic over-extraction on the CONTROL (median > 12/session) → FAIL.
    assert summarize(misses, _rows(10, 10, 14), 0.99)["verdict"] == "FAIL"
    # Faithfulness below 0.9 → FAIL.
    assert summarize(misses, ctrl, 0.80)["verdict"] == "FAIL"


def test_missing_hand_score_is_incomplete_not_pass() -> None:
    s = summarize(_rows(10, 8, 5), _rows(10, 10, 6), None)
    assert s["verdict"].startswith("INCOMPLETE")
    assert s["gate"]["density_ok"] and not s["gate"]["faithfulness_ok"]


def test_errored_rows_are_excluded_from_the_rate() -> None:
    rows = _rows(4, 4, 5) + [{"error": "boom", "gold_in_facts": False,
                              "facts_per_session": [], "facts_total": 0,
                              "parse_failures": 0, "calls": 0}]
    s = summarize(rows, _rows(4, 4, 5), 0.95)
    assert s["n_misses"] == 4
    assert s["gold_in_facts_rate"] == 100.0
    assert s["errors"] == 1
