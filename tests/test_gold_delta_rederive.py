"""The re-derivation's rules must be harsher than the calculation that passed.

docs/plans/2026-09-01-gold-delta-rederivation-protocol.md is a re-analysis of
data its author had already seen, and §0 says so. The only protection left is
that its rules are STRICTER than the post-hoc ±2·SE calculation that already
gave the welcome answer. That protection is worth exactly as much as these
tests: if `t_3 = 3.182` silently drifted back to `2`, or if the companion
quietly regained its OR, the spec's §0 would be a promise the code does not
keep.

So the load-bearing test here is `test_an_effect_that_passes_at_2se_fails_at_t3`
-- it constructs an effect that the calculation I already ran would have called
real, and asserts this one does not.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
_SCORER = _BENCH / "gold_delta_rederive.py"

POOL = ("EO", "IE", "IF", "KU", "MR", "PF", "SUM", "TR")
CONTROL = ("ABS", "CR")
ABILITIES = POOL + CONTROL
GOLD = ("B", "C1", "C2", "B2c")


def _rows(score_of):
    return [{"ability": ab, "question": f"q{ab}{i}", "answer": f"a{ab}{i}",
             "ideal_answer": f"gold{ab}{i}", "rubric": f"rub{ab}{i}",
             "gold_kind": "ideal_answer", "score": score_of(ab, i),
             "scores": [score_of(ab, i)], "judge_parse": "ok",
             # What the judge READ -- computed fresh per arm from its own
             # reparse, unlike every other field here, which is inherited.
             "judged_ideal": f"read{ab}{i}"}
            for ab in ABILITIES for i in range(16)]


def _write(path, score_of, **meta):
    path.write_text(json.dumps({
        "metadata": {"judge_model": "m", "date": "2026-09-01T00:00:00+00:00", **meta},
        "summary": {},
        "conversations": [{"questions": _rows(score_of)}]}))
    return path


def _run(tmp, arm_scores, anchor=lambda ab, i: 1.0, mutate=None):
    tmp.mkdir(parents=True, exist_ok=True)
    """Five artifacts on disk, then the scorer. `arm_scores[tag]` is a
    (ability, i) -> score callable."""
    paths = [_write(tmp / "A.json", anchor)]
    for tag in GOLD:
        p = _write(tmp / f"{tag}.json", arm_scores[tag], judge_gold=True)
        if mutate:
            mutate(tag, p)
        paths.append(p)
    r = subprocess.run([sys.executable, str(_SCORER), *map(str, paths), "--no-prereg"],
                       capture_output=True, text=True)
    return r


def _flat(units):
    """`units[tag]` pool rows dropped by 0.25 -- one 'unit' of pool delta,
    worth 0.25/128 = 0.1953pp each."""
    def make(tag):
        drop = {(POOL[j % 8], j // 8) for j in range(units[tag])}
        return lambda ab, i: 0.75 if (ab, i) in drop else 1.0
    return {t: make(t) for t in GOLD}


def _disjoint(n=8):
    """Every arm drops n pool rows, but a DIFFERENT n rows each. The arm means
    coincide exactly (E2 sees no spread at all) while the rows disagree (E1
    sees plenty) -- the case where E1 is the wider interval."""
    def make(tag):
        k = GOLD.index(tag)
        drop = {(POOL[j % 8], j // 8) for j in range(k * n, k * n + n)}
        return lambda ab, i: 0.75 if (ab, i) in drop else 1.0
    return {t: make(t) for t in GOLD}


def _verdict(out):
    line = [l for l in out.splitlines() if l.startswith("VERDICT: ")]
    return line[-1].split(": ", 1)[1] if line else None


# --- §0's promise: the multiplier ------------------------------------------

def test_an_effect_that_passes_at_2se_fails_at_t3(tmp_path):
    """THE test. Arm deltas whose |mean| exceeds 2·SE but not t_3·SE.

    The post-hoc calculation already reported to Atta used ±2·SE at n=4. This
    spec uses t_{0.975,3} = 3.182, which is 1.59x wider. An effect landing in
    that gap is one the old rule called real and the new rule must not."""
    units = {"B": 2, "C1": 10, "C2": 3, "B2c": 9}
    r = _run(tmp_path, _flat(units))
    assert r.returncode == 0, r.stdout + r.stderr

    arm = [-0.25 * u / 128 * 100 for u in units.values()]
    mean = sum(arm) / 4
    sd = (sum((a - mean) ** 2 for a in arm) / 3) ** 0.5
    assert abs(mean) > 2 * sd / 2, "fixture must pass under the OLD ±2·SE rule"
    assert abs(mean) < 3.182446 * sd / 2, "fixture must fail under the NEW t_3 rule"

    assert _verdict(r.stdout) != "CONFIRMED", r.stdout
    assert "WITHDRAWN AS A DEMONSTRATED RESULT" in r.stdout


def test_a_clear_effect_is_still_confirmed(tmp_path):
    """Harsher must not mean inert."""
    r = _run(tmp_path, _flat({"B": 8, "C1": 9, "C2": 10, "B2c": 11}))
    assert _verdict(r.stdout) == "CONFIRMED", r.stdout


def test_no_effect_is_not_confirmed(tmp_path):
    r = _run(tmp_path, _flat({"B": 0, "C1": 1, "C2": 0, "B2c": 1}))
    assert _verdict(r.stdout) == "NOT CONFIRMED", r.stdout


# --- §6.1 the envelope ------------------------------------------------------

def _halves(out):
    def half(prefix):
        line = [l for l in out.splitlines() if prefix in l][0]
        return float(line.split("+/- ")[1].split(" ")[0])
    return half("E1 row-level"), half("E2 arm-level")


def test_both_estimators_must_exclude_zero(tmp_path):
    """§6.1 is an intersection-union test: the conjunction is equivalent to
    'the wider interval governs' because the two share a centre, and unlike
    max-of-two-SEs it has a meaning."""
    wide_e2 = _run(tmp_path / "a", _flat({"B": 2, "C1": 10, "C2": 3, "B2c": 9}))
    wide_e1 = _run(tmp_path / "b", _disjoint())
    for r in (wide_e2, wide_e1):
        e1, e2 = _halves(r.stdout)
        line = [l for l in r.stdout.splitlines() if "BOTH must exclude" in l][0]
        assert float(line.split("+/- ")[1].split(" ")[0]) == pytest.approx(max(e1, e2))
    assert _halves(wide_e2.stdout)[1] > _halves(wide_e2.stdout)[0]
    assert _halves(wide_e1.stdout)[0] > _halves(wide_e1.stdout)[1]


# --- §6.1 the degrees of freedom -------------------------------------------

def test_rows_all_four_arms_agree_on_do_not_buy_degrees_of_freedom(tmp_path):
    """A row every arm scores identically is a STRUCTURAL zero, not a small
    sample. Counting it as 3 df was the draft's error; Satterthwaite is the
    fix. With 4 of 128 rows carrying all the variance, ν_eff must land near
    4×3 = 12, nowhere near 384."""
    r = _run(tmp_path, _flat({"B": 0, "C1": 4, "C2": 0, "B2c": 4}))
    line = [l for l in r.stdout.splitlines() if "E1 row-level" in l][0]
    nu = float(line.split("nu_eff ")[1].split(" ")[0])
    assert nu < 20, f"Satterthwaite df not applied: {line}"


def test_t_crit_reproduces_published_quantiles():
    """The hardcoded 2 is how the already-reported calculation went wrong, and
    §6.1's ν_eff is not known until runtime, so the quantile is computed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gdr", _SCORER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for df, want in ((1, 12.7062), (2, 4.30265), (3, 3.18245), (10, 2.22814),
                     (24, 2.06390), (383, 1.96609), (100000, 1.95996)):
        assert m.t_crit(df) == pytest.approx(want, abs=1e-4), df


# --- §6.7 concentration -----------------------------------------------------

def test_a_verdict_carried_by_one_ability_may_not_be_called_pool_wide(tmp_path):
    """§6.7. The shift is real and the verdict fires -- but if dropping one
    ability collapses the interval, calling it pool-wide is a false description
    of a true rejection, and description is where a verdict does its damage."""
    drop = {("EO", i) for i in range(16)}

    def make(ab, i):
        return 0.0 if (ab, i) in drop else 1.0
    r = _run(tmp_path, {t: make for t in GOLD})
    assert "carried by EO" in _verdict(r.stdout), r.stdout
    assert "FORBIDDEN" in r.stdout


def test_a_broad_shift_may_be_called_pool_wide(tmp_path):
    """The constraint must not fire on a genuinely distributed effect."""
    def make(ab, i):
        return 0.0 if ab in POOL and i < 8 else 1.0
    r = _run(tmp_path, {t: make for t in GOLD})
    assert _verdict(r.stdout) == "CONFIRMED", r.stdout
    assert "MAY be described as pool-wide" in r.stdout


# --- §5.1 / §6.5 the two questions ------------------------------------------

def test_the_process_question_is_reported_in_the_verdict_block(tmp_path):
    """The draft demoted §6.5 to a footnote without saying which way it lands.
    Hiding a number that contradicts the top-line is the failure mode this
    document exists to avoid, so it is printed whichever way it goes."""
    r = _run(tmp_path, _flat({"B": 8, "C1": 9, "C2": 10, "B2c": 11}))
    assert "SS6.5 THE PROCESS QUESTION" in r.stdout
    assert r.stdout.index("SS6.5") < r.stdout.index("SS6.4"), \
        "the process question must precede the descriptive companion"


def test_a_record_confirm_without_a_process_confirm_is_bound(tmp_path):
    """§5.1: when the two questions diverge, the reportable sentence is fixed
    in advance, because that is where 'REBASE survives' silently migrated
    between them last time."""
    r = _run(tmp_path, _disjoint())
    assert _verdict(r.stdout) == "CONFIRMED", r.stdout
    assert "SS5.1 BINDS" in r.stdout
    assert "NOT reportable" in r.stdout


# --- §6.4 the companion must not vote ---------------------------------------

def test_a_firing_companion_does_not_rescue_the_verdict(tmp_path):
    """The original §5 OR'd the companion in. This spec does not, and the
    reason is in §6.4: OR'ing gives a claim I have already twice announced a
    second chance to fire on the same data.

    Fixture: 12 pool rows cross t=0.45 identically in all four arms, so the
    companion's four-arm net has zero between-arm SD and a nonzero centre --
    it fires. The pool mean is arranged to sit exactly at zero."""
    flips = {(POOL[j // 16], j % 16) for j in range(12)}   # 0.5 -> 0.4, a lost flip
    up = {(POOL[4 + j // 16], j % 16) for j in range(6)}   # 0.0 -> 0.4, no flip

    def anchor(ab, i):
        return 0.5 if (ab, i) in flips else (0.0 if (ab, i) in up else 1.0)

    def make(raise_them):
        def f(ab, i):
            if (ab, i) in flips:
                return 0.4
            if (ab, i) in up:
                return 0.4 if raise_them else 0.0
            return 1.0
        return f

    arms = {"B": make(False), "C1": make(True), "C2": make(False), "B2c": make(True)}
    r = _run(tmp_path, arms, anchor=anchor)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "does NOT OR" in r.stdout
    assert _verdict(r.stdout) == "NOT CONFIRMED", r.stdout


# --- §6.0 the gate ----------------------------------------------------------

def test_a_systematic_control_shift_voids(tmp_path):
    """ABS/CR prompts are byte-identical whether gold is on or off. A control
    shift means the pool delta is unattributable, whatever the pool says."""
    def make(n):
        drop = {(CONTROL[j // 16], j % 16) for j in range(n)}
        return lambda ab, i: 0.75 if (ab, i) in drop else 1.0
    r = _run(tmp_path, {"B": make(3), "C1": make(4), "C2": make(4), "B2c": make(5)})
    assert r.returncode == 5, r.stdout
    assert "VOID" in r.stdout
    assert "VERDICT:" not in r.stdout, "a voided run must not print a verdict"


def test_a_clean_control_arm_passes_the_gate(tmp_path):
    r = _run(tmp_path, _flat({"B": 8, "C1": 9, "C2": 10, "B2c": 11}))
    assert "GATE PASSES" in r.stdout


# --- §4 preconditions -------------------------------------------------------

def _corrupt(field, value, tag_to_hit="C1"):
    def mutate(tag, path):
        if tag != tag_to_hit:
            return
        d = json.loads(path.read_text())
        d["conversations"][0]["questions"][0][field] = value
        path.write_text(json.dumps(d))
    return mutate


def test_a_difference_in_what_the_judge_read_aborts(tmp_path):
    """§4.2 as it should always have been.

    B carries no `dataset_revisions` and a rejudge reparses the dataset for
    gold. The question is whether the four arms reparsed the SAME gold, and the
    only field that can answer it is the one each arm computes from its own
    reparse."""
    r = _run(tmp_path, _flat({t: 4 for t in GOLD}),
             mutate=_corrupt("judged_ideal", "DIFFERENT GOLD"))
    assert r.returncode == 3
    assert "4.2 gold identity" in r.stdout
    assert "reparsed different gold" in r.stdout


def test_the_older_field_name_is_accepted(tmp_path):
    """The four arms on disk predate `judged_ideal` and record the same thing
    as `judge_ideal_used` (90ced81). §10 said they could not be retrofitted;
    they can, and this is what lets the scorer read them."""
    def mutate(tag, path):
        d = json.loads(path.read_text())
        for q in d["conversations"][0]["questions"]:
            q["judge_ideal_used"] = q.pop("judged_ideal")
        path.write_text(json.dumps(d))
    r = _run(tmp_path, _flat({t: 4 for t in GOLD}), mutate=mutate)
    assert r.returncode == 0, r.stdout[-2000:]
    assert "4.2 gold identity" in r.stdout


def test_an_arm_that_never_recorded_what_it_read_aborts(tmp_path):
    """A pre-4d9906b main-path artifact. §4.2 must refuse rather than fall back
    to an inherited field and report a pass it did not earn."""
    def mutate(tag, path):
        if tag != "C1":
            return
        d = json.loads(path.read_text())
        for q in d["conversations"][0]["questions"]:
            q.pop("judged_ideal")
        path.write_text(json.dumps(d))
    r = _run(tmp_path, _flat({t: 4 for t in GOLD}), mutate=mutate)
    assert r.returncode == 3
    assert "nothing here that could fail" in r.stdout


def test_a_constant_judged_gold_is_refused_as_powerless(tmp_path):
    """THE test this whole correction exists for (§10.3, turned on §4.2 itself).

    If every arm-row carries the same string, identity across arms is automatic
    and the check reports agreement exactly as it would if it had measured
    something. That is indistinguishable from a pass, so it must not be one."""
    def mutate(tag, path):
        d = json.loads(path.read_text())
        for q in d["conversations"][0]["questions"]:
            q["judged_ideal"] = "SAME FOR EVERY ROW"
        path.write_text(json.dumps(d))
    r = _run(tmp_path, _flat({t: 4 for t in GOLD}), mutate=mutate)
    assert r.returncode == 3
    assert "4.2 power" in r.stdout
    assert "carries no information" in r.stdout


@pytest.mark.parametrize("field", ["rubric", "ideal_answer", "gold_kind"])
def test_an_inherited_field_difference_still_aborts_as_a_swapped_file(tmp_path, field):
    """§4.2b. These fields are inherited from A by every arm, so a difference
    cannot mean "reparsed differently" -- it means one of the five files is not
    a rejudge of this anchor. The check is kept; only its claim changed."""
    r = _run(tmp_path, _flat({t: 4 for t in GOLD}), mutate=_corrupt(field, "DIFFERENT"))
    assert r.returncode == 3
    assert "4.2b inherited-field identity" in r.stdout
    assert "not four rejudges of one anchor" in r.stdout


def test_the_scorer_never_claims_the_inherited_fields_measured_a_reparse(tmp_path):
    """The retracted sentence, and the shape of it, must not come back."""
    src = (_BENCH / "gold_delta_rederive.py").read_text()
    assert "rubric identity" not in src
    assert "unwitnessed revision is not load-bearing" not in src


def test_a_differing_answer_aborts(tmp_path):
    r = _run(tmp_path, _flat({t: 4 for t in GOLD}), mutate=_corrupt("answer", "REWRITTEN"))
    assert r.returncode == 3
    assert "4.3 answer identity" in r.stdout
    assert "not a re-generated answer" in r.stdout


def test_an_unreadable_row_aborts(tmp_path):
    """With four arms there is no defensible way to average over a hole."""
    r = _run(tmp_path, _flat({t: 4 for t in GOLD}), mutate=_corrupt("scores", []))
    assert r.returncode == 3
    assert "4.5 readability" in r.stdout


def test_a_void_arm_is_refused(tmp_path):
    def mutate(tag, path):
        if tag != "B2c":
            return
        d = json.loads(path.read_text())
        d["metadata"]["void"] = {"reason": "silent zeros", "rule": "§4.4"}
        path.write_text(json.dumps(d))
    r = _run(tmp_path, _flat({t: 4 for t in GOLD}), mutate=mutate)
    assert r.returncode == 4
    assert "REFUSING" in r.stdout


def test_a_missing_row_aborts(tmp_path):
    def mutate(tag, path):
        if tag != "C2":
            return
        d = json.loads(path.read_text())
        d["conversations"][0]["questions"].pop()
        path.write_text(json.dumps(d))
    r = _run(tmp_path, _flat({t: 4 for t in GOLD}), mutate=mutate)
    assert r.returncode == 3
    assert "4.1 row identity" in r.stdout


# --- §6.2 exchangeability ---------------------------------------------------

def test_judge_strata_that_disagree_block_a_pooled_verdict(tmp_path):
    """Pooling two alias arms with two pin arms assumes the judge config is
    not a fixed effect. If it is, §6.1 is invalid and no verdict is issued --
    including a favourable one."""
    r = _run(tmp_path, _flat({"B": 0, "C1": 40, "C2": 40, "B2c": 0}))
    assert _verdict(r.stdout) == "NO POOLED VERDICT", r.stdout
    assert "FAILS" in r.stdout


def test_the_exchangeability_pass_is_labelled_weak(tmp_path):
    """A 2-df test that passes is not evidence; the spec says so and the
    output must too, because the output is what gets quoted."""
    r = _run(tmp_path, _flat({"B": 8, "C1": 9, "C2": 10, "B2c": 11}))
    assert "WEAK evidence" in r.stdout


# --- refusing to run unpinned -----------------------------------------------

def test_the_scorer_will_not_run_without_an_explicit_prereg_choice(tmp_path):
    paths = [_write(tmp_path / "A.json", lambda ab, i: 1.0)]
    for tag in GOLD:
        paths.append(_write(tmp_path / f"{tag}.json", lambda ab, i: 1.0))
    r = subprocess.run([sys.executable, str(_SCORER), *map(str, paths)],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "--prereg" in r.stdout
