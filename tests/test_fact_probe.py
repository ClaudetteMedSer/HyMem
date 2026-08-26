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
    FACTS_PROMPT_V1,
    FACTS_PROMPTS,
    _MAX_FACT_CHARS,
    audit_fact_dates,
    extract_facts,
    build_faithfulness_sample,
    _MAX_FACTS_PER_SESSION,
    gold_session_ids,
    index_facts,
    open_fact_index,
    rescore_rows,
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
    # A malformed date is dropped and NOTHING stands in for it — the fact is kept
    # but stays undated (see test_validator_never_stamps_the_session_date...).
    assert all(f["date"] is None for f in facts)
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
    assert out["gold_session_in_facts"] is True
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
    assert out["gold_session_in_facts"] is False


def test_max_sessions_keeps_the_most_recent(monkeypatch) -> None:
    q = _q_data()
    out = run_question(q, llm=None, sim=True, max_sessions=1)
    assert out["sessions_processed"] == 1
    # The gold sits in the LAST session, so a recency cap of 1 still reaches it —
    # and the cap is label-free (it never consults which session holds gold).
    assert out["gold_session_in_facts"] is True


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
    return [{"error": None, "gold_session_in_facts": i < covered,
             "answer_in_facts": i < covered, "answer_checkable": True,
             "gold_verbatim_in_facts": False,
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


def test_parse_failure_ceiling_is_incomplete_never_fail() -> None:
    """Pre-registered 2026-07-31: parse_failures/calls > 2% makes the run
    UNREADABLE — truncation biases the four criteria in opposite directions
    (criterion 1 harder, criteria 3/4 easier), so a truncation-heavy run is
    indistinguishable from an honest FAIL without this counter. The ceiling
    must override even a run where all four criteria pass."""
    rows = [dict(r, parse_failures=1, calls=3) for r in _rows(10, 7, 5)]
    ctrl = [dict(r, parse_failures=0, calls=3) for r in _rows(10, 10, 6)]
    s = summarize(rows, ctrl, 0.95)
    assert all(s["gate"][k] for k in ("density_ok", "facts_per_session_ok",
                                      "control_ok", "faithfulness_ok"))
    assert s["verdict"].startswith("INCOMPLETE")
    assert "ceiling" in s["verdict"]
    # And it must override a FAIL-shaped run too: never read as FAIL.
    s2 = summarize([dict(r, parse_failures=2, calls=3) for r in _rows(10, 4, 5)],
                   ctrl, 0.80)
    assert s2["verdict"].startswith("INCOMPLETE")


def test_errored_rows_are_excluded_from_the_rate() -> None:
    rows = _rows(4, 4, 5) + [{"error": "boom", "gold_session_in_facts": False,
                              "answer_in_facts": False, "answer_checkable": True,
                              "gold_verbatim_in_facts": False,
                              "facts_per_session": [], "facts_total": 0,
                              "parse_failures": 0, "calls": 0}]
    s = summarize(rows, _rows(4, 4, 5), 0.95)
    assert s["n_misses"] == 4
    assert s["gold_in_facts_rate"] == 100.0
    assert s["errors"] == 1


def test_session_cap_flags_a_cut_gold_session() -> None:
    """`--max-sessions` is label-free, so it can throw away the gold session and
    force a miss. That must be REPORTED, not silently folded into the density
    number — a budget artifact read as a mechanism result is how a gate lies."""
    q = _q_data()
    # Reverse so the gold session is the OLDEST — a recency cap of 1 drops it.
    q["haystack_sessions"] = list(reversed(q["haystack_sessions"]))
    q["haystack_session_ids"] = list(reversed(q["haystack_session_ids"]))
    q["haystack_dates"] = list(reversed(q["haystack_dates"]))

    cut = run_question(q, llm=None, sim=True, max_sessions=1)
    assert cut["gold_cut_by_session_cap"] is True
    assert cut["gold_session_in_facts"] is False  # forced miss, not a real one
    s = summarize([cut], [], 0.95)
    assert s["gold_cut_by_session_cap"] == 1

    # Uncapped, the same question is found — proving the miss was the cap.
    full = run_question(q, llm=None, sim=True, max_sessions=0)
    assert full["gold_cut_by_session_cap"] is False
    assert full["gold_session_in_facts"] is True


def test_max_questions_budgets_are_nested() -> None:
    """The budget knob must be a prefix, not a fresh sample: n=2 ⊂ n=3, so two
    runs at different budgets are comparable instead of being different sets."""
    run = {"per_question": [
        _row(f"ms-miss-{i}") for i in range(5)
    ] + [_row(f"ms-hit-{i}", correct=True) for i in range(5)]}
    misses, _, _ = select_probe_sets(run, seed=0)
    assert misses[:2] == misses[:3][:2]


# ── The instrument fix: paraphrased facts (the 2026-07-30 0% artifact) ──────

PARAPHRASE = ("The postgres connection pool for the checkout service was "
              "raised to 40 to stop the timeouts.")


def test_paraphrased_fact_scores_on_provenance_not_verbatim() -> None:
    """THE regression for the 0%-vs-82% artifact.

    A faithful narrative fact is a REWRITE of its turn, so verbatim containment
    cannot fire — that is what made the first real run report a hard 0% while
    --sim (canned verbatim "facts") read 82%. Provenance must score it, and the
    verbatim reading must be visibly False so the lesson stays legible."""
    conn = open_fact_index()
    index_facts(conn, "s1", [{"text": PARAPHRASE, "date": None, "entities": []}])
    hits = search_facts(conn, "what did I raise the postgres connection pool to?",
                        top_k=5)
    conn.close()
    assert hits, "the paraphrase must still be RETRIEVABLE — only the check changed"
    # The old check: structurally incapable of firing on a rewrite.
    from fact_probe import _gold_in_pool
    assert not _gold_in_pool([GOLD_TURN], [h["text"] for h in hits])
    # The new check keys on WHICH SESSION the fact came from, so it fires.
    assert hits[0]["session_id"] == "s1"


def test_gold_session_ids_prefers_turn_flags_then_falls_back() -> None:
    q = _q_data()
    assert gold_session_ids(q) == {"s1"}          # via has_answer
    for sess in q["haystack_sessions"]:           # strip the turn-level flags
        for m in sess:
            m.pop("has_answer", None)
    assert gold_session_ids(q) == set()           # neither signal → excluded
    q["answer_session_ids"] = ["s1"]
    assert gold_session_ids(q) == {"s1"}          # via answer_session_ids


def test_short_answers_are_excluded_from_the_answer_check() -> None:
    """"40" would match inside some fact by chance and report as signal."""
    q = _q_data()
    q["answer"] = "40"
    out = run_question(q, llm=None, sim=True, max_sessions=0)
    assert out["answer_checkable"] is False
    assert out["answer_in_facts"] is False


def test_control_arm_density_is_reported_alongside_the_misses() -> None:
    """The control column is the validity check on the new measure: a check that
    returns the same constant on both arms is broken, not informative."""
    s = summarize(_rows(10, 6, 5), _rows(10, 10, 6), 0.95)
    assert s["density_misses"]["n"] == 10
    assert s["density_control"]["n"] == 10
    assert s["density_control"]["gold_session_rate"] == 100.0
    assert s["density_misses"]["gold_session_rate"] == 60.0
    # The gate reads the MISS arm only; the control is there to be looked at.
    assert s["gate"]["density_ok"]


def test_rescore_recomputes_density_without_re_extracting() -> None:
    """An instrument fix must not cost a re-run: the dump already carries every
    returned fact and its session, so the new readings are computable offline."""
    q = _q_data()
    q["answer"] = "40 connections after the checkout timeouts"
    fresh = run_question(q, llm=None, sim=True, max_sessions=0)
    fresh["_kind"] = "miss"

    # Simulate a dump written by the OLD code: strip every field the fix added.
    stale = {k: v for k, v in fresh.items()
             if k not in {"gold_session_in_facts", "answer_in_facts",
                          "answer_checkable", "gold_verbatim_in_facts",
                          "gold_session_ids"}}
    assert "gold_session_in_facts" not in stale

    [rescored] = rescore_rows([stale], {q["question_id"]: q})
    assert rescored["gold_session_in_facts"] is True
    assert rescored["gold_session_ids"] == ["s1"]
    assert rescored["calls"] == fresh["calls"]      # nothing re-extracted


def test_rescore_flags_a_question_missing_from_the_dataset() -> None:
    [row] = rescore_rows([{"question_id": "ghost", "retrieved": []}], {})
    assert "not in --dataset" in row["error"]


# ── Stratified faithfulness sample (the 2026-07-30 distractor-flood defect) ──

def _dumped_row(qid: str, gold_sid: str, filler: int) -> dict:
    """A row whose extraction covered one gold session and `filler` distractors —
    the real LME shape (LongMemEval pads every haystack with UltraChat/ShareGPT)."""
    dump = [{"session_id": gold_sid, "date": None,
             "facts": [{"text": "gold fact", "date": None, "entities": []}],
             "source_turns": "gold turns"}]
    dump += [{"session_id": f"ultrachat_{i}", "date": None,
              "facts": [{"text": f"filler {i}", "date": None, "entities": []}],
              "source_turns": f"filler turns {i}"} for i in range(filler)]
    return {"question_id": qid, "dump": dump}


def test_faithfulness_sample_reserves_half_for_gold_sessions() -> None:
    """A uniform sample over ~50 distractors to ~1 gold session is ~all
    distractors, so the hand-score would measure faithfulness on LME's padding
    instead of on the dated/numeric content the gate is about."""
    q = _q_data()                                   # gold session is "s1"
    rows = [_dumped_row(q["question_id"], "s1", filler=50)]
    by_id = {q["question_id"]: q}

    sample = build_faithfulness_sample(rows, by_id, size=20, seed=0)
    assert len(sample) == 20
    strata = [e["stratum"] for e in sample]
    # Only one gold session exists here, so the gold half cannot be filled — but
    # it must be present, and the budget must not shrink.
    assert "gold_bearing" in strata
    assert strata.count("distractor") == 19


def test_faithfulness_sample_fills_the_gold_half_when_it_can() -> None:
    q = _q_data()
    rows = [_dumped_row(f"q{i}", "s1", filler=20) for i in range(10)]
    by_id = {f"q{i}": q for i in range(10)}
    sample = build_faithfulness_sample(rows, by_id, size=20, seed=0)
    n_gold = sum(1 for e in sample if e["stratum"] == "gold_bearing")
    assert n_gold == 10                             # half the budget, as reserved
    assert len(sample) == 20


def test_uniform_sampling_would_have_missed_gold_entirely() -> None:
    """Pins WHY stratification was added: the old uniform draw over the same
    pool returns a sample with (almost) no gold-bearing sessions."""
    import random as _random

    q = _q_data()
    rows = [_dumped_row(q["question_id"], "s1", filler=50)]
    pool = [{"question_id": r["question_id"], **d}
            for r in rows for d in r["dump"]]
    uniform = _random.Random(0).sample(pool, 20)
    gold_hits = sum(1 for e in uniform if e["session_id"] == "s1")
    assert gold_hits <= 1                           # the defect, reproduced
    stratified = build_faithfulness_sample(rows, {q["question_id"]: q},
                                           size=20, seed=0)
    assert any(e["stratum"] == "gold_bearing" for e in stratified)


def test_sample_entries_carry_their_source_turns() -> None:
    """The hand-read must be self-contained: a value is scored against the turns
    shipped with it, never by re-joining the dataset."""
    q = _q_data()
    rows = [_dumped_row(q["question_id"], "s1", filler=3)]
    for entry in build_faithfulness_sample(rows, {q["question_id"]: q},
                                           size=4, seed=0):
        assert entry["source_turns"]
        assert entry["facts"]
        assert entry["question_id"] == q["question_id"]


# ── Date handling: the validator no longer fabricates (2026-07-30) ──────────

def test_validator_never_stamps_the_session_date_on_an_undated_fact() -> None:
    """THE regression for the faithfulness hand-read.

    The model returning `date: null` is it doing the right thing on a session
    that states no date. The validator used to overwrite that null with the
    SESSION's date, turning "undated" into a confident specific date — which the
    hand-read then scored as a model hallucination. `fact_date` is explicit dates
    only; relative references are E4's job."""
    facts = validate_facts([{"text": "the pool was raised", "date": None}],
                           session_date="2023-11-30T09:00:00")
    assert facts[0]["date"] is None


def test_malformed_date_is_dropped_not_replaced() -> None:
    facts = validate_facts([{"text": "a fact", "date": "last friday"}],
                           session_date="2023-11-30T09:00:00")
    assert facts[0]["date"] is None
    assert facts[0]["text"] == "a fact"          # the fact itself survives


def test_explicit_iso_date_is_preserved() -> None:
    facts = validate_facts([{"text": "a fact", "date": "2023-05-22"}],
                           session_date="2023-11-30T09:00:00")
    assert facts[0]["date"] == "2023-05-22"


def test_date_audit_separates_injected_from_model_supplied() -> None:
    """Lets a hand-score already performed on a pre-fix dump be re-attributed
    without re-extracting: only `model_supplied` dates are the model's to answer
    for."""
    rows = [{"dump": [
        {"session_id": "s1", "date": "2023-11-30T00:00:00", "facts": [
            {"text": "a", "date": "2023-11-30"},   # == session date → injected
            {"text": "b", "date": "2023-05-22"},   # differs → the model's
            {"text": "c", "date": None},           # undated
        ]},
    ]}]
    a = audit_fact_dates(rows)
    assert a["facts"] == 3
    assert a["dated"] == 2
    assert a["undated"] == 1
    assert a["injected_or_coincident"] == 1
    assert a["model_supplied"] == 1
    assert a["injected_share"] == 50.0


# ── FACTS_PROMPT_V2: the one allowed iteration ─────────────────────────────

def test_v2_prompt_removes_both_visible_defects() -> None:
    """The two defects the hand-read exposed must be gone from the text itself,
    not merely intended: the session-date licence and the fact quota."""
    _tag, system, template = FACTS_PROMPTS["v2"]
    assert "the session date applies" not in system      # no dating licence
    assert "2 to 8 facts" not in system                  # no quota floor
    assert "{date}" not in template                      # model never sees it
    assert "[]" in system                                # empty is a good answer
    # v1 is retained verbatim so the v1 run stays reproducible.
    assert FACTS_PROMPTS["v1"][1] is FACTS_PROMPT_V1
    assert FACTS_PROMPTS["v1"][0] != FACTS_PROMPTS["v2"][0]


def test_extract_facts_routes_to_the_selected_prompt_arm() -> None:
    class Recorder:
        def __init__(self):
            self.systems, self.users = [], []

        def chat(self, messages, temperature=0.1, max_tokens=1024):
            self.systems.append(messages[0]["content"])
            self.users.append(messages[1]["content"])
            return "[]"

    msgs = [{"role": "user", "content": "I raised the pool to 40 " * 5}]
    for arm, sentinel in (("v1", "the session date applies"),
                          ("v2", "THE ONLY RULE THAT MATTERS")):
        rec = Recorder()
        extract_facts(rec, msgs, session_date="2023-11-30T00:00:00",
                      prompt_version=arm)
        assert sentinel in rec.systems[0]
    # v2's user turn must not carry the session date anywhere.
    rec = Recorder()
    extract_facts(rec, msgs, session_date="2023-11-30T00:00:00", prompt_version="v2")
    assert "2023-11-30" not in rec.users[0]
