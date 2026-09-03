"""Projecting a better gate, without letting the projection flatter itself.

The claim is that scoring gate 4 on the questions the lever touched beats
scoring all 500, by sqrt(D_S / D_all). Three ways that arithmetic can produce
an encouraging number it has not earned, each with a test here:

  * calibrating on a real A/B, where the discordance read as churn contains
    the effect, so the projected gain is inflated by the thing it claims to
    detect;
  * a subset with no observed discordance, where sqrt(0) makes the projected
    MDE zero and the gate look infinitely sensitive when in fact its churn is
    simply unmeasured;
  * quoting the subset's MDE against the subset's own n, which makes any
    subset look better merely for being small.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import concentration_model as cm  # noqa: E402


def artifact(rows, *, lever=False, extra=None):
    cfg = {cm.LEVER: lever}
    cfg.update(extra or {})
    return {"config": cfg, "per_question": rows}


def rows(n, *, correct, **kw):
    return [{"question_id": f"q{i}", "correct": correct(i), **kw(i)}
            for i in range(n)]


def pair(n, flip_in, flip_out, n_in):
    """`n_in` questions carry a moved episode count; `flip_in` of those and
    `flip_out` of the rest are discordant."""
    a, b = [], []
    for i in range(n):
        inside = i < n_in
        flipped = (inside and i < flip_in) or \
                  (not inside and i < n_in + flip_out)
        a.append({"question_id": f"q{i}", "correct": True, "n_episodes": 5})
        b.append({"question_id": f"q{i}", "correct": not flipped,
                  "n_episodes": 9 if inside else 5})
    return artifact(a), artifact(b)


def run(a, b):
    lines: list[str] = []
    res = cm.report(a, b, out=lines.append)
    return res, "\n".join(lines)


# --------------------------------------------------- the structural refusal

def test_a_real_AB_pair_is_REFUSED():
    """The load-bearing one. On an A/B the discordance is churn PLUS effect,
    so every gain would be inflated by the very effect being projected."""
    a, b = pair(100, 4, 16, 20)
    b["config"][cm.LEVER] = True
    res, text = run(a, b)
    assert res["refused"] is True
    assert "REFUSED" in text
    # No projected NUMBER may be printed. ("gain" alone appears in the
    # refusal's own explanation of what it is declining to compute.)
    assert "gain:" not in text
    assert "projected MDE" not in text
    assert "indicator:" not in text


def test_a_pair_that_cannot_say_which_arm_it_is_is_also_REFUSED():
    a, b = pair(100, 4, 16, 20)
    del b["config"][cm.LEVER]
    res, _ = run(a, b)
    assert res["refused"] is True


def test_a_same_arm_pair_is_projected():
    a, b = pair(100, 4, 16, 20)
    res, text = run(a, b)
    assert "refused" not in res
    assert "episode_count" in res


# ------------------------------------------------------------- the arithmetic

def test_the_gain_is_the_root_of_the_retained_discordance():
    s = {"n": 100, "n_in": 20, "d_all": 20, "d_in": 4}
    p = cm.projection(s, 100)
    assert p["gain"] == pytest.approx(math.sqrt(20 / 4))
    assert p["retained_discordance"] == pytest.approx(0.2)


def test_both_MDEs_are_quoted_against_the_FULL_n():
    """The net questions a lever moves is an absolute count. Quoting the
    subset's MDE against the subset's own n would make every subset look
    better purely for being small, which is the opposite of a measurement."""
    s = {"n": 100, "n_in": 20, "d_all": 20, "d_in": 4}
    p = cm.projection(s, 100)
    assert p["mde_all"] == pytest.approx(100 * cm.Z95 * math.sqrt(20) / 100)
    assert p["mde_sub"] == pytest.approx(100 * cm.Z95 * math.sqrt(4) / 100)


def test_a_subset_that_keeps_all_the_churn_gains_nothing():
    s = {"n": 100, "n_in": 100, "d_all": 20, "d_in": 20}
    p = cm.projection(s, 100)
    assert p["gain"] == pytest.approx(1.0)


def test_a_noisier_subset_gains_less_than_its_size_suggests():
    """"Retrieval moved" is exactly the kind of indicator that selects
    unstable questions, so the gain must be driven by retained DISCORDANCE
    and not by the subset's size."""
    small_clean = cm.projection({"n": 100, "n_in": 20, "d_all": 20, "d_in": 4}, 100)
    small_noisy = cm.projection({"n": 100, "n_in": 20, "d_all": 20, "d_in": 16}, 100)
    assert small_noisy["gain"] < small_clean["gain"]


def test_break_even_leakage_is_the_share_of_effect_the_subset_may_lose():
    s = {"n": 100, "n_in": 20, "d_all": 20, "d_in": 4}
    p = cm.projection(s, 100)
    assert p["breakeven_leakage"] == pytest.approx(1 - p["mde_sub"] / p["mde_all"])
    assert 0 < p["breakeven_leakage"] < 1


# --------------------------------------------------------- the sqrt(0) trap

def test_a_subset_with_no_discordance_has_NO_projection_not_a_perfect_one():
    """sqrt(0) makes the projected MDE 0.00pp, which reads as a gate of
    unlimited resolution. It means the subset's churn was never observed."""
    p = cm.projection({"n": 100, "n_in": 5, "d_all": 20, "d_in": 0}, 100)
    assert p["available"] is False
    assert "not zero" in p["reason"]
    assert "mde_sub" not in p


def test_the_report_prints_no_gain_for_an_unestimated_subset():
    a, b = pair(100, 0, 20, 20)
    _, text = run(a, b)
    assert "NO PROJECTION" in text
    assert "gain:" not in text


def test_an_indicator_that_fires_on_nothing_has_no_projection():
    p = cm.projection({"n": 100, "n_in": 0, "d_all": 20, "d_in": 0}, 100)
    assert p["available"] is False
    assert "fires on no shared question" in p["reason"]


# ------------------------------------------------------------ subset_stats

def test_churn_is_reported_inside_and_outside():
    a, b = pair(100, 4, 16, 20)
    s = cm.subset_stats({r["question_id"]: r for r in a["per_question"]},
                        {r["question_id"]: r for r in b["per_question"]},
                        cm.INDICATORS["episode_count"])
    assert s["n_in"] == 20 and s["d_in"] == 4 and s["d_all"] == 20
    assert s["churn_in"] == pytest.approx(4 / 20)
    assert s["churn_out"] == pytest.approx(16 / 80)


def test_unscored_rows_are_excluded_from_the_calibration():
    """D3 again: a row the judge never scored has no outcome to be discordant
    about, and leaving it in would dilute the churn rate the projection rests
    on."""
    a, b = pair(10, 2, 2, 4)
    a["per_question"][0]["correct"] = None
    s = cm.subset_stats({r["question_id"]: r for r in a["per_question"]},
                        {r["question_id"]: r for r in b["per_question"]},
                        cm.INDICATORS["episode_count"])
    assert s["n"] == 9


def test_the_null_firing_rate_is_reported_as_contamination():
    """An indicator fires on a same-arm pair too. Those questions join the
    subset with churn and no signal, and the rate is the floor on that."""
    a, b = pair(100, 4, 16, 20)
    _, text = run(a, b)
    assert "CONTAMINATION FLOOR" in text
    assert "THIS IS NOT f" in text
    # The COUNT, not a bare "20%" -- the retained-discordance line happens to
    # print 20% too, so a looser assertion passes with this line deleted.
    assert "fires on 20/100 questions" in text


# -------------------------------------------------------------- indicators

def test_the_episode_indicator_does_not_fire_on_rows_that_lack_the_field():
    """Absent n_episodes is a gap in the instrument. `None != None` is False
    so it would not fire anyway -- pinned so a later refactor to `.get(k, 0)`
    cannot silently make every legacy row fire."""
    assert not cm.INDICATORS["episode_count"]({}, {})
    assert not cm.INDICATORS["episode_count"]({}, {"n_episodes": 3})


def test_the_sha_indicator_needs_both_sides_to_carry_one():
    assert not cm.INDICATORS["context_sha"]({"context_sha": "X"}, {})
    assert cm.INDICATORS["context_sha"]({"context_sha": "X"},
                                        {"context_sha": "Y"})
    assert not cm.INDICATORS["context_sha"]({"context_sha": "X"},
                                            {"context_sha": "X"})



# ------------------------------------------- the curve, and its denominator

def test_the_gain_curve_is_one_over_root_f():
    assert cm.gain_curve(0.25)["gain"] == pytest.approx(2.0)
    assert cm.gain_curve(1.0)["gain"] == pytest.approx(1.0)


def test_break_even_leakage_falls_as_the_lever_touches_more():
    """A lever that moves the whole run cannot be concentrated at all, and a
    subset that keeps everything tolerates no leakage because it buys
    nothing."""
    assert cm.gain_curve(1.0)["breakeven_leakage"] == pytest.approx(0.0)
    assert cm.gain_curve(0.25)["breakeven_leakage"] == pytest.approx(0.5)


def test_an_impossible_fraction_has_no_curve_point():
    assert cm.gain_curve(0.0)["available"] is False
    assert cm.gain_curve(1.5)["available"] is False


def test_the_report_leads_with_the_curve_not_the_null_firing_rate():
    """The correction this module needed. The null pair's firing rate is what
    the indicator does with NO lever set; reading it as f quotes the gain
    from the wrong population, and for an episode-count indicator on a
    granularity lever it is wrong by a lot."""
    a, b = pair(100, 4, 16, 20)
    _, text = run(a, b)
    assert "as a function of what the lever touches" in text
    curve_at = text.index("as a function of what the lever touches")
    assert curve_at < text.index("=== indicator:")
    # The ROWS, not just the header: a curve nobody can read is not a curve.
    # 1/sqrt(.25) = 2.00 and 1/sqrt(.50) = 1.41.
    assert "2.00x" in text and "1.41x" in text


def test_a_noisier_subset_is_flagged_as_breaking_the_curves_assumption():
    """The curve assumes a subset holds its share of the discordance. An
    indicator that selects unusually noisy questions does not, and the gain
    is then an upper bound rather than an estimate."""
    a, b = pair(100, 16, 4, 20)   # churn 80% inside, 5% outside
    _, text = run(a, b)
    assert "churn is NOT uniform here" in text
    assert "upper bound on the gain" in text


def test_uniform_churn_is_not_flagged():
    a, b = pair(100, 4, 16, 20)   # 20% inside, 20% outside
    _, text = run(a, b)
    assert "churn is NOT uniform" not in text


# --------------------------------------------------------------- saturation

def saturated_pair(n, n_at_ceiling, ceiling=10):
    """Both arms pinned at the ceiling on the first `n_at_ceiling` questions,
    so the episode-count indicator cannot fire there whatever the lever does."""
    a, b = [], []
    for i in range(n):
        pinned = i < n_at_ceiling
        a.append({"question_id": f"q{i}", "correct": True,
                  "n_episodes": ceiling if pinned else 3})
        b.append({"question_id": f"q{i}", "correct": i % 7 != 0,
                  "n_episodes": ceiling if pinned else 5})
    return artifact(a), artifact(b)


def test_a_saturated_indicator_is_flagged():
    """The finding that motivated this check: n_episodes is clamped at 10 on
    84% of the real run, so the indicator is a CONSTANT there. A constant
    cannot indicate anything, and a subset built from it selects questions
    that fell below the retrieval cap rather than questions the lever moved."""
    a, b = saturated_pair(100, 84)
    _, text = run(a, b)
    assert "SATURATED" in text
    assert "84/100" in text
    assert "ceiling of 10" in text


def test_an_unsaturated_indicator_is_not_flagged():
    a, b = saturated_pair(100, 5)
    _, text = run(a, b)
    assert "SATURATED" not in text


def test_the_ceiling_is_read_from_the_data_not_hardcoded():
    """A later change to retrieval depth must not silently disable the check."""
    a, b = saturated_pair(100, 84, ceiling=25)
    _, text = run(a, b)
    assert "ceiling of 25" in text


def test_saturation_needs_BOTH_arms_at_the_ceiling():
    """One arm at the cap and the other below it is the indicator firing,
    which is the opposite of blind."""
    a, b = saturated_pair(100, 0)
    for r in a["per_question"]:
        r["n_episodes"] = 10
    _, text = run(a, b)
    assert "SATURATED" not in text


def test_an_indicator_with_no_ceiling_check_is_not_flagged():
    """Only `episode_count` registers a saturation check; the others must not
    silently reuse it and report a ceiling they never measured."""
    assert set(cm.BLIND) == {"episode_count"}
    a, b = saturated_pair(100, 84)
    _, text = run(a, b)
    assert text.count("SATURATED") == 1
