"""A split that reads "the judge is fine" without ever being able to say so.

`churn_decompose` attributes each flipped question to the judge or the reader
by asking whether the two runs produced the same answer text. That inference
is only as good as the variation in its key: if no two runs ever repeat an
answer string, the module reports 0% judge-side from a comparison that could
never have returned anything else.

These tests exist mostly to keep that failure mode impossible -- the
UNAVAILABLE branch, the concordant control rate, and the parser-consistency
check are all load-bearing, and each has a test that fails if it is removed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import churn_decompose as cd  # noqa: E402


def row(qid, correct, hyp, raw="yes", **kw):
    return {"question_id": qid, "correct": correct, "hypothesis": hyp,
            "judge_raw": raw, **kw}


def art(*rows):
    return {"per_question": list(rows)}


def test_same_answer_different_verdict_is_judge_side():
    d = cd.decompose(art(row("q1", True, "Paris")),
                     art(row("q1", False, "Paris", raw="no")))
    assert d["judge_side"] == 1 and d["answer_side"] == 0


def test_different_answer_different_verdict_is_answer_side():
    d = cd.decompose(art(row("q1", True, "Paris")),
                     art(row("q1", False, "Lyon", raw="no")))
    assert d["answer_side"] == 1 and d["judge_side"] == 0


def test_a_whitespace_only_difference_is_not_a_different_answer():
    """Otherwise re-wrapping inflates the expensive side of the split."""
    d = cd.decompose(art(row("q1", True, "Paris, France")),
                     art(row("q1", False, "Paris,\n  France", raw="no")))
    assert d["judge_side"] == 1
    assert d["exact_same_hypothesis"] == 0, "exact count stays exact"


def test_unscored_rows_are_excluded_not_counted_concordant():
    """D3: a judge that never answered has no verdict to flip. Counting it as
    agreement would dilute the churn rate with rows nobody judged."""
    a = art(row("q1", True, "Paris"), row("q2", None, ""),
            row("q3", True, "x", judge_error=True))
    b = art(row("q1", True, "Paris"), row("q2", None, ""),
            row("q3", False, "x", judge_error=True))
    d = cd.decompose(a, b)
    assert d["shared"] == 3 and d["unscored"] == 2 and d["n"] == 1
    assert d["discordant"] == 0


def test_no_repeated_answer_anywhere_is_UNAVAILABLE_not_zero_judge_churn():
    """The whole point of the module. Every answer differs, so the key is
    constant -- and a constant cannot explain a flip. Reporting "0%
    judge-side" here would be a fact about string equality, not the judge."""
    a = art(row("q1", True, "aaa"), row("q2", True, "bbb"))
    b = art(row("q1", False, "ccc", raw="no"), row("q2", True, "ddd"))
    d = cd.decompose(a, b)
    assert d["available"] is False
    assert d["judge_side"] == 0, "the count is zero, which is why it must "\
        "not be read as a finding"


def test_one_repeated_answer_is_enough_to_make_the_split_available():
    a = art(row("q1", True, "aaa"), row("q2", True, "same"))
    b = art(row("q1", False, "ccc", raw="no"), row("q2", True, "same"))
    d = cd.decompose(a, b)
    assert d["available"] is True
    assert d["concordant_same_hyp"] == 1


def test_the_concordant_control_rate_is_reported():
    """Without it there is nothing to compare the discordant rate against,
    and the attribution has no power statement at all."""
    a = art(row("q1", True, "same"), row("q2", True, "x"),
            row("q3", True, "keep"))
    b = art(row("q1", False, "same", raw="no"), row("q2", True, "y"),
            row("q3", True, "keep"))
    d = cd.decompose(a, b)
    assert d["identical_rate_discordant"] == 1.0
    assert d["identical_rate_concordant"] == pytest.approx(0.5)


def test_same_answer_and_same_raw_but_a_different_verdict_is_flagged():
    """`parse_judge_verdict` is pure string logic. If this ever fires the
    artifact is inconsistent and the split must not be read."""
    d = cd.decompose(art(row("q1", True, "Paris", raw="yes")),
                     art(row("q1", False, "Paris", raw="yes")))
    assert d["parser_impossible"] == ["q1"]


def test_an_honest_judge_flip_is_not_flagged_as_impossible():
    d = cd.decompose(art(row("q1", True, "Paris", raw="yes")),
                     art(row("q1", False, "Paris", raw="no")))
    assert d["parser_impossible"] == [] and d["judge_side"] == 1


def test_no_shared_questions_raises_rather_than_reporting_nothing():
    with pytest.raises(ValueError):
        cd.decompose(art(row("q1", True, "a")), art(row("q2", True, "a")))


def test_removing_judge_churn_lowers_the_mde():
    a = art(*[row(f"q{i}", True, "same") for i in range(10)])
    rows_b = [row("q0", False, "same", raw="no"),      # judge-side
              row("q1", False, "different", raw="no")]  # answer-side
    b = art(*rows_b, *[row(f"q{i}", True, "same") for i in range(2, 10)])
    d = cd.decompose(a, b)
    assert d["judge_side"] == 1 and d["answer_side"] == 1
    assert d["judge_share"] == pytest.approx(0.5)
    assert d["mde_pp_judge_free"] < d["mde_pp"]


def test_all_churn_judge_side_means_a_perfect_judge_leaves_none():
    a = art(row("q1", True, "same"), row("q2", True, "same2"))
    b = art(row("q1", False, "same", raw="no"), row("q2", True, "same2"))
    d = cd.decompose(a, b)
    assert d["mde_pp_judge_free"] is None, "no answer-side churn remains"


def test_report_on_an_unavailable_split_prints_no_attribution():
    """A reader who sees "judge-side 0" cannot un-see it. The UNAVAILABLE
    branch must RETURN, not merely warn -- the same rule guard_score follows
    for an unevidenced arm."""
    a = art(row("q1", True, "aaa"))
    b = art(row("q1", False, "ccc", raw="no"))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    text = "\n".join(lines)
    assert "UNAVAILABLE" in text
    assert "judge-side" not in text.lower().replace("judge-side (", "")


def test_report_prints_the_power_check():
    a = art(row("q1", True, "same"), row("q2", True, "k"))
    b = art(row("q1", False, "same", raw="no"), row("q2", True, "k"))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    text = "\n".join(lines)
    assert "power check" in text
    assert "among CONCORDANT" in text


def test_report_warns_when_the_parser_looks_non_deterministic():
    a = art(row("q1", True, "same", raw="yes"), row("q2", True, "k"))
    b = art(row("q1", False, "same", raw="yes"), row("q2", True, "k"))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    assert "not readable" in "\n".join(lines)


# --------------------------------------------------------------------------
# Stage 2: answer-side churn split by whether the reader's INPUT also moved,
# and the interval that must be attached to a judge flip count of zero.
# --------------------------------------------------------------------------


def test_a_flip_whose_retrieval_also_moved_counts_as_retrieval_side():
    d = cd.decompose(art(row("q1", True, "a", n_episodes=3), row("q2", True, "k")),
                     art(row("q1", False, "b", raw="no", n_episodes=9),
                         row("q2", True, "k")))
    assert d["retrieval_side_min"] == 1 and d["decoder_side_max"] == 0


def test_a_flip_on_an_unchanged_fingerprint_counts_as_decoder_side():
    d = cd.decompose(art(row("q1", True, "a", n_episodes=3), row("q2", True, "k")),
                     art(row("q1", False, "b", raw="no", n_episodes=3),
                         row("q2", True, "k")))
    assert d["decoder_side_max"] == 1 and d["retrieval_side_min"] == 0


def test_a_judge_side_flip_is_not_also_counted_in_the_context_split():
    """The two stages must not double-count the same flipped question."""
    d = cd.decompose(art(row("q1", True, "same", n_episodes=3)),
                     art(row("q1", False, "same", raw="no", n_episodes=9)))
    assert d["judge_side"] == 1
    assert d["retrieval_side_min"] == 0 and d["decoder_side_max"] == 0


def test_every_fingerprint_differing_makes_the_context_split_UNAVAILABLE():
    """Same guard as stage 1: a key that never repeats explains nothing."""
    a = art(row("q1", True, "a", n_episodes=1), row("q2", True, "c", n_episodes=2))
    b = art(row("q1", False, "b", raw="no", n_episodes=3),
            row("q2", True, "d", n_episodes=4))
    d = cd.decompose(a, b)
    assert d["context_available"] is False
    assert d["retrieval_side_min"] == 1, "the count exists but must not be read"


def test_the_concordant_fingerprint_rate_is_the_control_for_stage_2():
    a = art(row("q1", True, "x", n_episodes=1), row("q2", True, "y", n_episodes=2))
    b = art(row("q1", True, "x", n_episodes=1), row("q2", True, "y", n_episodes=8))
    d = cd.decompose(a, b)
    assert d["context_identical_rate_concordant"] == pytest.approx(0.5)


def test_ability_and_recall_tier_are_part_of_the_fingerprint():
    """A different route to the answer is a different context even when the
    pool sizes coincide."""
    a = art(row("q1", True, "a", n_episodes=3, ability_used="single-session"))
    b = art(row("q1", False, "b", raw="no", n_episodes=3, ability_used="multi-session"))
    d = cd.decompose(a, b)
    assert d["retrieval_side_min"] == 1


def test_zero_judge_flips_is_an_interval_not_a_zero_rate():
    """The load-bearing one. 0/181 must not be reported as 0%: the honest
    claim is the rule-of-three bound, ~1.6%."""
    assert cd.binomial_upper_95(0, 181) == pytest.approx(3.0 / 181, abs=3e-4)
    assert cd.binomial_upper_95(0, 181) > 0.0


def test_the_bound_loosens_as_the_sample_shrinks():
    assert cd.binomial_upper_95(0, 10) > cd.binomial_upper_95(0, 100)


def test_the_bound_is_undefined_with_no_sample_rather_than_zero():
    assert cd.binomial_upper_95(0, 0) is None


def test_the_bound_exceeds_the_point_estimate():
    d = cd.decompose(art(row("q1", True, "same"), row("q2", True, "s2")),
                     art(row("q1", False, "same", raw="no"), row("q2", True, "s2")))
    assert d["judge_flip_rate"] == pytest.approx(0.5)
    assert d["judge_flip_upper_95"] > d["judge_flip_rate"]


def test_report_states_the_bound_and_refuses_the_word_deterministic():
    a = art(row("q1", True, "same"), row("q2", True, "s2"))
    b = art(row("q1", True, "same"), row("q2", False, "other", raw="no"))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    text = "\n".join(lines)
    assert "one-sided 95%" in text
    assert "deterministic" not in text.replace("NOT 'the judge is deterministic'", "")


def test_report_on_an_unavailable_context_split_prints_no_bounds():
    """The fixture has to clear stage 1 to reach stage 2 at all: q1 repeats
    its answer, so hypothesis identity varies, while no fingerprint ever
    repeats. A fixture that fails stage 1 returns before stage 2 and tests
    nothing -- which is how the first version of this test passed against a
    mutation that deleted the branch it names."""
    a = art(row("q1", True, "same", n_episodes=1), row("q2", True, "c", n_episodes=2))
    b = art(row("q1", True, "same", n_episodes=7),
            row("q2", False, "d", raw="no", n_episodes=4))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    text = "\n".join(lines)
    assert "UNAVAILABLE" in text
    assert "retrieval moved too" not in text


# --------------------------------------------------------------------------
# The matched control for stage 2. Counting flips whose retrieval also moved
# says nothing on its own: retrieval moves constantly on questions whose
# verdict holds. Only the comparison licenses "fix retrieval".
# --------------------------------------------------------------------------


def test_the_matched_control_is_non_flips_whose_answer_also_moved():
    """Not all concordant questions -- those include the stable ones, where
    retrieval is stable for the same reason the answer is, which biases the
    control down and makes retrieval look guilty."""
    a = art(row("q1", True, "a", n_episodes=1),      # flip, ctx moved
            row("q2", True, "b", n_episodes=1),      # holds, answer+ctx moved
            row("q3", True, "same", n_episodes=1))   # holds, nothing moved
    b = art(row("q1", False, "z", raw="no", n_episodes=9),
            row("q2", True, "c", n_episodes=9),
            row("q3", True, "same", n_episodes=1))
    d = cd.decompose(a, b)
    assert d["ctx_moved_rate_discordant"] == pytest.approx(1.0)
    assert d["ctx_moved_rate_concordant_moved_answer"] == pytest.approx(1.0), \
        "q2 is the matched control; q3 must not dilute it"


def test_equal_rates_are_reported_as_no_association():
    a = art(row("q1", True, "a", n_episodes=1), row("q2", True, "b", n_episodes=1),
            row("q3", True, "keep"))
    b = art(row("q1", False, "z", raw="no", n_episodes=9),
            row("q2", True, "c", n_episodes=9), row("q3", True, "keep"))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    text = "\n".join(lines)
    assert "NOT over-represented" in text
    assert "does\n  not follow" in text


def test_over_representation_is_reported_as_ours_to_fix():
    a = art(row("q1", True, "a", n_episodes=1), row("q2", True, "b", n_episodes=1),
            row("q3", True, "keep"))
    b = art(row("q1", False, "z", raw="no", n_episodes=9),
            row("q2", True, "c", n_episodes=1),   # answer moved, ctx did not
            row("q3", True, "keep"))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    text = "\n".join(lines)
    assert "OVER-represented" in text
    assert "a flag of ours could reach it" in text


def test_the_verdict_sentence_follows_the_control_not_the_raw_count():
    """A non-zero retrieval count with a control that matches it must NOT
    read as 'retrieval churn is ours to fix'. That inference is the whole
    thing this control exists to block."""
    a = art(row("q1", True, "a", n_episodes=1), row("q2", True, "b", n_episodes=1),
            row("q3", True, "keep"))
    b = art(row("q1", False, "z", raw="no", n_episodes=9),
            row("q2", True, "c", n_episodes=9), row("q3", True, "keep"))
    d = cd.decompose(a, b)
    assert d["retrieval_side_min"] == 1, "the raw count is non-zero"
    lines: list[str] = []
    cd.report(d, out=lines.append)
    assert "a flag of ours could reach it" not in "\n".join(lines)


def test_the_matched_control_rate_is_printed_not_merely_computed():
    """A verdict sentence derived from a number the reader never sees is a
    number they cannot check. Both rates go on the page."""
    a = art(row("q1", True, "a", n_episodes=1), row("q2", True, "b", n_episodes=1),
            row("q3", True, "keep"))
    b = art(row("q1", False, "z", raw="no", n_episodes=9),
            row("q2", True, "c", n_episodes=9), row("q3", True, "keep"))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    text = "\n".join(lines)
    assert "among NON-flips whose answer also moved" in text
    assert "among answer-side FLIPS" in text


# --------------------------------------------------------------------------
# `context_sha` mode. Runs recording the hashed reader prompt turn the
# retrieval/decoder bounds into a partition -- but only if the two runs speak
# the same schema.
# --------------------------------------------------------------------------


def test_a_sha_pair_is_read_in_exact_mode():
    d = cd.decompose(
        art(row("q1", True, "a", context_sha="X"), row("q2", True, "k", context_sha="Y")),
        art(row("q1", False, "b", raw="no", context_sha="X"),
            row("q2", True, "k", context_sha="Y")))
    assert d["fingerprint_mode"] == "exact"
    assert d["decoder_side_max"] == 1, "same prompt, different answer"


def test_exact_mode_ignores_the_count_fields():
    """Under a hash the counts are redundant, and letting them vote would
    reintroduce the resolution problem the hash exists to remove."""
    d = cd.decompose(
        art(row("q1", True, "a", context_sha="X", n_episodes=3),
            row("q2", True, "k", context_sha="Y")),
        art(row("q1", False, "b", raw="no", context_sha="X", n_episodes=99),
            row("q2", True, "k", context_sha="Y")))
    assert d["decoder_side_max"] == 1 and d["retrieval_side_min"] == 0


def test_a_moved_sha_is_retrieval_side():
    d = cd.decompose(
        art(row("q1", True, "a", context_sha="X"), row("q2", True, "k", context_sha="Y")),
        art(row("q1", False, "b", raw="no", context_sha="Z"),
            row("q2", True, "k", context_sha="Y")))
    assert d["retrieval_side_min"] == 1 and d["decoder_side_max"] == 0


def test_one_run_with_sha_and_one_without_is_REFUSED():
    """The load-bearing guard. An artifact predating `context_sha` compared
    against one that has it would match no fingerprint anywhere and read as
    'every flip moved retrieval' -- a fact about the two schemas, not the two
    runs, and it would look exactly like a real finding."""
    with pytest.raises(ValueError, match="schemas"):
        cd.decompose(art(row("q1", True, "a", context_sha="X"),
                         row("q2", True, "k", context_sha="Y")),
                     art(row("q1", False, "b", raw="no"), row("q2", True, "k")))


def test_a_partly_stamped_run_is_REFUSED_not_read_as_counts():
    """Half the rows carrying a sha is neither mode, and silently falling
    back to counts would score half the run on a fingerprint the other half
    does not use."""
    with pytest.raises(ValueError, match="schemas"):
        cd.decompose(art(row("q1", True, "a", context_sha="X"), row("q2", True, "k")),
                     art(row("q1", False, "b", raw="no"), row("q2", True, "k")))


def test_two_runs_BOTH_partly_stamped_are_refused_too():
    """The pair looks self-consistent -- both runs are stamped the same way --
    so a mode check that only compares the two runs to each other passes it
    through. `fingerprint_mode` has to reject a single run that is partly
    stamped, not merely a pair that disagrees.

    Found by a mutation that called any stamped row "exact" and survived: the
    fixture above makes the PAIR mixed, so it could never have caught it."""
    with pytest.raises(ValueError, match="schemas"):
        cd.decompose(
            art(row("q1", True, "a", context_sha="X"), row("q2", True, "k")),
            art(row("q1", False, "b", raw="no", context_sha="X"),
                row("q2", True, "k")))


def test_fingerprint_mode_rejects_a_partly_stamped_run_on_its_own():
    assert cd.fingerprint_mode([{"context_sha": "X"}, {}]) == "mixed"
    assert cd.fingerprint_mode([{"context_sha": "X"}]) == "exact"
    assert cd.fingerprint_mode([{}, {}]) == "counts"


def test_a_legacy_pair_still_reads_in_counts_mode():
    d = cd.decompose(art(row("q1", True, "a", n_episodes=1), row("q2", True, "k")),
                     art(row("q1", False, "b", raw="no", n_episodes=1),
                         row("q2", True, "k")))
    assert d["fingerprint_mode"] == "counts"


def test_the_report_says_which_fingerprint_it_used():
    """A reader who cannot tell counts-mode from exact-mode cannot tell a
    bound from a partition."""
    lines: list[str] = []
    cd.report(cd.decompose(
        art(row("q1", True, "a", context_sha="X"), row("q2", True, "k", context_sha="Y")),
        art(row("q1", False, "b", raw="no", context_sha="X"),
            row("q2", True, "k", context_sha="Y"))), out=lines.append)
    text = "\n".join(lines)
    assert "A PARTITION" in text and "BOUNDS, NOT A PARTITION" not in text
    # The MODE ITSELF, not just the explanatory suffix beside it: a report
    # that drops the word is one a reader cannot check the partition claim
    # against.
    assert "fingerprint: exact" in text


def test_the_counts_report_still_carries_the_bounds_caveat():
    lines: list[str] = []
    cd.report(cd.decompose(
        art(row("q1", True, "a", n_episodes=1), row("q2", True, "k")),
        art(row("q1", False, "b", raw="no", n_episodes=1), row("q2", True, "k"))),
        out=lines.append)
    text = "\n".join(lines)
    assert "BOUNDS, NOT A PARTITION" in text
    assert "A PARTITION. The fingerprint is the rendered" not in text
    assert "fingerprint: counts" in text
