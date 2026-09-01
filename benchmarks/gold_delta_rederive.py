#!/usr/bin/env python3
"""Read protocol of docs/plans/2026-09-01-gold-delta-rederivation-protocol.md.

Re-derives the BEAM gold-delta verdict from FOUR gold-on arms instead of the
single control arm whose SD_ctl = 0.0000 produced a zero-width band -- a band
that makes every nonzero effect significant by construction.

Written and committed before any statistic in the spec's SS6 was computed. It
implements the rules; it does not choose them. The four per-arm pool means were
already known when the spec was written (SS0 of the spec discloses this and says
why the rules here are deliberately harsher than the post-hoc calculation that
already passed); everything this file computes was not.

    gold_delta_rederive.py <A> <B> <C1> <C2> <B2c> --prereg <spec>
"""
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_adapter import resolve_prereg          # noqa: E402

POOL = ("EO", "IE", "IF", "KU", "MR", "PF", "SUM", "TR")
CONTROL = ("ABS", "CR")
COMPANION_T = 0.45      # original SS5, pre-registered from A's marginal


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                    -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + num * d
            d = 1.0 / (tiny if abs(d) < tiny else d)
            c = 1.0 + num / (tiny if abs(c) < tiny else c)
            h *= d * c
        if abs(d * c - 1.0) < 3e-16:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_crit(df, p=0.975):
    """Two-sided Student-t quantile, by bisection on the exact CDF.

    Hardcoding these is what let the already-reported calculation use 2 where
    it needed 3.182, and SS6.1's Satterthwaite df is not known until runtime, so
    there is nothing to hardcode even if it were safe."""
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    def cdf(t):
        return 1.0 - 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


T3, T2 = t_crit(3), t_crit(2)


def rows(path):
    with open(path) as f:
        d = json.load(f)
    return d["metadata"], {(q["ability"], q["question"]): q
                           for c in d["conversations"] for q in c["questions"]}


def abort(msg):
    print(f"\nABORT: {msg}")
    sys.exit(3)


def refuse_void(tag, meta):
    """A void artifact holds rows worth characterising and a verdict worth
    nothing. Preserving the evidence only helps if reading it as a result stays
    impossible -- otherwise the fix that saved the data becomes the reason
    someone quotes it."""
    v = meta.get("void")
    if v:
        print(f"REFUSING: {tag} is VOID - {v.get('reason')}. "
              f"Rule: {v.get('rule')}")
        sys.exit(4)


def judged_gold(row):
    """The text the judge actually read, under either of its two names.

    `judge_ideal_used` has been written by the rejudge path since 90ced81 --
    before the gold-delta phase began. `judged_ideal` was added to BOTH paths
    by 4d9906b, which is where SS10 concluded the existing arms "cannot be
    retrofitted". They can: the field was there the whole time, under the
    older name."""
    for f in ("judged_ideal", "judge_ideal_used"):
        if f in row:
            return row[f]
    return None


def e1(keys, arms, gold_tags, centre):
    """SS6.1 E1 with Satterthwaite df.

    The draft claimed 128*3 = 384 df. That is only right if the per-row
    variances are homogeneous, and they are not: a row all four arms agree on
    contributes a STRUCTURAL zero, not a small sample. sigma^2 is then an
    average of mostly-zeros whose sampling distribution is nothing like
    chi2_384, and the interval built on it is too narrow."""
    sv = [statistics.variance([arms[t][k]["score"] for t in gold_tags]) for k in keys]
    sigma2 = statistics.fmean(sv)
    ss = sum(x * x for x in sv)
    nu = 3.0 * sum(sv) ** 2 / ss if ss > 0 else 3.0 * len(sv)
    se = (sigma2 / (len(keys) * 4)) ** 0.5 * 100
    return t_crit(nu) * se, se, sigma2, nu


def ci(centre, half):
    return f"{centre:+.3f}pp +/- {half:.3f} -> [{centre - half:+.3f}, {centre + half:+.3f}]"


def excludes_zero(centre, half):
    return abs(centre) > half


def main():
    if len(sys.argv) < 6:
        print(__doc__)
        return 2
    paths = sys.argv[1:6]
    prereg = None
    if "--prereg" in sys.argv:
        prereg = resolve_prereg(sys.argv[sys.argv.index("--prereg") + 1])
    elif "--no-prereg" not in sys.argv:
        print("ERROR: pass --prereg <spec> or --no-prereg explicitly.")
        return 2

    tags = ("A", "B", "C1", "C2", "B2c")
    metas, arms = {}, {}
    for tag, p in zip(tags, paths):
        metas[tag], arms[tag] = rows(p)
        print(f"  {tag:4s} {Path(p).name}")
    if prereg:
        print(f"\nprereg blob={prereg['blob'][:12]} commit={prereg['commit'][:8]} "
              f"code={prereg['code_commit'][:8]}")

    gold_tags = ("B", "C1", "C2", "B2c")
    print("\n=== provenance ===")
    for tag in tags:
        m = metas[tag]
        pr = m.get("prereg") or {}
        print(f"  {tag:4s} judge={m.get('judge_model'):18s} gold={str(m.get('judge_gold')):5s} "
              f"extra={m.get('judge_extra_body')} date={m.get('date','')[:19]}")
        print(f"       prereg={str(pr.get('blob'))[:12]:12s} "
              f"dataset={(m.get('dataset_revisions') or {}).get('Mohammadta/BEAM','UNWITNESSED')[:12]}")

    # ---- SS4 preconditions -------------------------------------------------
    print("\n=== SS4 preconditions ===")
    for tag in tags:
        refuse_void(tag, metas[tag])
    print("  4.4 void refusal: all five arms non-void")

    keys = sorted(arms["A"])
    for tag in gold_tags:
        if sorted(arms[tag]) != keys:
            abort(f"4.1 row identity: {tag} keys differ from A")
    if len(keys) != 160:
        abort(f"4.1 row identity: {len(keys)} rows, expected 160")
    print(f"  4.1 row identity: {len(keys)}/160 keys identical across five arms")

    # SS4.2 GOLD IDENTITY, rewritten -- see the protocol's SS10 and SS11.
    # The original compared `ideal_answer`/`gold_kind`, which every rejudge arm
    # INHERITS from anchor A (`out = dict(r)`). Four copies of one field agree
    # by construction, so its passing said nothing about what each arm
    # reparsed. The field with power is the one each arm computes fresh from
    # its OWN reparse: the text the judge actually read.
    judged = {t: {k: judged_gold(arms[t][k]) for k in keys} for t in gold_tags}
    missing = [(t, k) for t in gold_tags for k in keys if judged[t][k] is None]
    if missing:
        abort(f"4.2 gold identity: {len(missing)} arm-rows record neither "
              f"'judged_ideal' nor 'judge_ideal_used'; e.g. {missing[0][0]} "
              f"{missing[0][1]}. Without what the judge read there is nothing "
              f"here that could fail, and SS4.2 must not run.")
    vals = [str(judged[t][k]) for t in gold_tags for k in keys]
    distinct = len(set(vals))
    # SS10.3 turned on this check itself: absent, empty or constant all report
    # agreement identically to a comparison that measured something.
    if distinct < 2 or all(not v.strip() for v in vals):
        abort(f"4.2 power: the judged-gold field takes {distinct} distinct "
              f"value(s) over {len(vals)} arm-rows; identity across arms is "
              f"then automatic and carries no information")
    bad = [k for k in keys
           if len({json.dumps(judged[t][k], sort_keys=True)
                   for t in gold_tags}) != 1]
    if bad:
        abort(f"4.2 gold identity: {len(bad)} rows differ in the text the "
              f"judge READ across the four arms; e.g. {bad[0]}. The arms "
              f"reparsed different gold and B's unwitnessed dataset revision "
              f"is load-bearing after all.")
    print(f"  4.2 gold identity: the text the judge READ is byte-identical "
          f"across B,C1,C2,B2c on {len(keys)}/160 rows "
          f"({distinct} distinct values, so the check could have failed) "
          f"-> the four reparses agree, measured rather than inferred")

    # SS4.2b -- the inherited fields, kept and relabelled. A difference here
    # cannot mean an arm reparsed differently, because no arm computes these;
    # it means one of the five files is not a rejudge of this anchor.
    for field in ("rubric", "ideal_answer", "gold_kind"):
        bad = [k for k in keys
               if len({json.dumps(arms[t][k].get(field), sort_keys=True)
                       for t in gold_tags}) != 1]
        if bad:
            abort(f"4.2b inherited-field identity: {len(bad)} rows differ in "
                  f"{field!r}; e.g. {bad[0]}. Every arm inherits this field "
                  f"from A, so a difference means these are not four rejudges "
                  f"of one anchor -- check which file was passed.")
    print("  4.2b inherited fields (rubric/ideal_answer/gold_kind) agree "
          "-- guaranteed by inheritance; detects a swapped file, nothing more")

    bad = [k for k in keys
           if len({arms[t][k].get("answer") for t in tags}) != 1]
    if bad:
        abort(f"4.3 answer identity: {len(bad)} rows differ in 'answer' across "
              f"A and the four arms. Also inherited, so this too means a file "
              f"that is not a rejudge of A -- not a re-generated answer.")
    print("  4.3 answer identity: answers byte-identical across A and all four "
          "arms (inherited; a swapped-file check)")

    for tag in tags:
        unreadable = [k for k in keys
                      if arms[tag][k].get("judge_error")
                      or str(arms[tag][k].get("judge_raw", "")).startswith("[LLM_ERROR")
                      or (arms[tag][k].get("rubric") and arms[tag][k].get("scores") == [])
                      or arms[tag][k].get("score") is None]
        if unreadable:
            abort(f"4.5 readability: {tag} has {len(unreadable)} unreadable rows; "
                  f"with four arms there is no defensible way to average over a hole")
    print("  4.5 readability: 160/160 readable in every arm")

    parse = {t: sum(1 for k in keys if arms[t][k].get("judge_parse") == "recovered")
             for t in gold_tags}
    print(f"  (descriptive) judge_parse recovered: "
          + ", ".join(f"{t} {parse[t]}" for t in gold_tags))

    # ---- quantities --------------------------------------------------------
    pool = [k for k in keys if k[0] in POOL]
    ctl = [k for k in keys if k[0] in CONTROL]
    if len(pool) != 128 or len(ctl) != 32:
        abort(f"pool {len(pool)} (want 128), control {len(ctl)} (want 32)")

    def d(tag, k):
        return arms[tag][k]["score"] - arms["A"][k]["score"]

    pool_mean = {t: statistics.fmean(d(t, k) for k in pool) for t in gold_tags}
    ctl_mean = {t: statistics.fmean(d(t, k) for k in ctl) for t in gold_tags}
    delta = statistics.fmean(pool_mean[t] for t in gold_tags)

    print("\n=== per-arm pool deltas (known before this spec; SS0) ===")
    for t in gold_tags:
        print(f"  {t:4s} pool {pool_mean[t] * 100:+.3f}pp   control {ctl_mean[t] * 100:+.3f}pp "
              f"({sum(1 for k in ctl if d(t, k))}/32 control rows churned)")
    print(f"  four-arm point estimate: {delta * 100:+.3f}pp")

    # ---- SS6.0 GATE --------------------------------------------------------
    print("\n=== SS6.0 GATE: four-arm control ===")
    ctl_vals = [ctl_mean[t] for t in gold_tags]
    ctl_c = statistics.fmean(ctl_vals) * 100
    ctl_sd = statistics.stdev(ctl_vals) * 100
    if ctl_sd == 0:
        flips = sum(1 for t in gold_tags for k in ctl if d(t, k))
        print(f"  degenerate SD=0 -> flip gate: {flips} control flips")
        if flips:
            print("  VOID: deterministic control that nevertheless flipped")
            return 5
        ctl_half = 0.0
    else:
        ctl_half = T3 * ctl_sd / (4 ** 0.5)
    print(f"  mean control delta {ci(ctl_c, ctl_half)}   (SD_arm {ctl_sd:.3f}pp, t3)")
    if excludes_zero(ctl_c, ctl_half):
        print("  *** VOID *** the arms carry a systematic shift on byte-identical "
              "prompts; the pool delta is unattributable. No verdict.")
        return 5
    print("  GATE PASSES: control interval contains zero")

    # ---- SS6.1 PRIMARY -----------------------------------------------------
    print("\n=== SS6.1 PRIMARY: the wider of two intervals ===")
    dc = delta * 100
    half_e1, se_e1, sigma2, nu = e1(pool, arms, gold_tags, dc)
    arm_sd = statistics.stdev([pool_mean[t] for t in gold_tags]) * 100
    se_e2 = arm_sd / (4 ** 0.5)
    half_e2 = T3 * se_e2
    print(f"  E1 row-level churn : sigma_churn {sigma2 ** 0.5:.5f}, SE {se_e1:.4f}pp, "
          f"nu_eff {nu:.1f} (t {t_crit(nu):.3f}) -> {ci(dc, half_e1)}")
    print(f"  E2 arm-level       : SD_arm {arm_sd:.4f}pp, SE {se_e2:.4f}pp, "
          f"nu 3 (t {T3:.3f})       -> {ci(dc, half_e2)}")
    half = max(half_e1, half_e2)
    print(f"  BOTH must exclude zero (SS6.1 intersection-union; equivalently the "
          f"wider, here {'E2' if half_e2 >= half_e1 else 'E1'}): {ci(dc, half)}")
    print(f"  E1 excludes 0: {excludes_zero(dc, half_e1)}   "
          f"E2 excludes 0: {excludes_zero(dc, half_e2)}")

    # ---- SS6.2 exchangeability --------------------------------------------
    print("\n=== SS6.2 exchangeability of the two judge configurations ===")
    alias = [pool_mean["B"] * 100, pool_mean["B2c"] * 100]
    pin = [pool_mean["C1"] * 100, pool_mean["C2"] * 100]
    contrast = statistics.fmean(alias) - statistics.fmean(pin)
    s_pooled = ((statistics.variance(alias) + statistics.variance(pin)) / 2) ** 0.5
    band = T2 * s_pooled
    print(f"  alias mean {statistics.fmean(alias):+.3f}pp, pin mean {statistics.fmean(pin):+.3f}pp")
    print(f"  contrast {contrast:+.3f}pp, pooled within-stratum SD {s_pooled:.3f}pp (2 df), "
          f"band +/-{band:.3f}pp")
    exch = not excludes_zero(contrast, band)
    print(f"  {'PASSES' if exch else 'FAILS'} -- and at 2 df a PASS is WEAK evidence "
          "of exchangeability (spec SS6.2). The real support is external: Step 1's "
          "score-level PASS, and D_alias 4/160 vs D_pin 7/160.")

    # ---- SS6.3 VERDICT -----------------------------------------------------
    print("\n=== SS6.3 VERDICT ===")
    if not exch:
        print("  NO POOLED VERDICT: the arms are not exchangeable (SS6.2). "
              "Reporting strata separately, each on 1 df, which supports nothing.")
        verdict = "NO POOLED VERDICT"
    elif excludes_zero(dc, half):
        print(f"  CONFIRMED. The pool gold-delta is real at the four-arm variance: "
              f"{ci(dc, half)}.")
        print("  REBASE REQUIRED stands, re-derived on a variance estimate rather "
              "than on a zero-width band.")
        verdict = "CONFIRMED"
    elif excludes_zero(dc, half_e1):
        print("  AMBIGUOUS. Not confirmed under the conservative envelope, though "
              "E1 alone would exclude zero.")
        print("  The original verdict is WITHDRAWN AS A DEMONSTRATED RESULT and "
              "reduced to the decision-theoretic asymmetry the original SS5 argued "
              "for -- a reason to act, not evidence that the effect exists.")
        verdict = "AMBIGUOUS"
    else:
        print("  NOT CONFIRMED. The record stands and Step 2's gold-delta "
              "justification is gone.")
        verdict = "NOT CONFIRMED"

    # ---- SS6.5 the process question (SS5.1) -- in the verdict block ------
    print("\n=== SS6.5 THE PROCESS QUESTION (SS5.1) -- reported here, not in a footnote ===")
    se_fresh = ((half / T3) ** 2 + sigma2 / 128 * 1e4) ** 0.5
    fresh_half = T3 * se_fresh
    print(f"  A as one draw: {ci(dc, fresh_half)}")
    print(f"  excludes 0: {excludes_zero(dc, fresh_half)}")
    if verdict == "CONFIRMED" and not excludes_zero(dc, fresh_half):
        print("  SS5.1 BINDS: the RECORD question confirms and the PROCESS question")
        print("  does not. Reportable: 'this record differs from what gold-on judging")
        print("  yields'. NOT reportable: 'the gold effect is established'.")
    print("  APPROXIMATION, unverified: A's gold-OFF churn variance taken equal to "
          "the measured gold-ON churn variance.")

    # ---- SS6.7 concentration: constrains WORDING, not whether it fired ---
    print("\n=== SS6.7 CONCENTRATION (constrains how the verdict may be worded) ===")
    contrib = {}
    for ab in POOL:
        ks = [k for k in pool if k[0] == ab]
        contrib[ab] = statistics.fmean(
            statistics.fmean(d(t, k) for k in ks) for t in gold_tags) * len(ks) / 128 * 100
    for ab in sorted(POOL, key=lambda a: contrib[a]):
        print(f"  {ab:4s} contribution to the pool mean {contrib[ab]:+.3f}pp")
    print(f"  (sum {sum(contrib.values()):+.3f}pp)")

    carried = []
    for ab in POOL:
        loo = [k for k in pool if k[0] != ab]
        loo_c = statistics.fmean(
            statistics.fmean(d(t, k) for k in loo) for t in gold_tags) * 100
        h1 = e1(loo, arms, gold_tags, loo_c)[0]
        loo_arm = [statistics.fmean(d(t, k) for k in loo) * 100 for t in gold_tags]
        h2 = T3 * statistics.stdev(loo_arm) / 2
        keeps = excludes_zero(loo_c, max(h1, h2))
        if not keeps:
            carried.append(ab)
        print(f"  drop {ab:4s} -> {ci(loo_c, max(h1, h2))}  "
              f"{'still excludes 0' if keeps else '*** INCLUDES 0 ***'}")

    if verdict == "CONFIRMED" and carried:
        verdict = f"CONFIRMED, carried by {'/'.join(carried)}"
        print(f"\n  BINDING (SS6.7): removing {'/'.join(carried)} makes the interval "
              f"include zero.")
        print("  The verdict fired and stands -- a shift concentrated in one ability is "
              "still a shift --")
        print("  but it MUST be reported as carried by that ability. The words "
              "'pool-wide' and 'broad'")
        print("  are FORBIDDEN in reporting it.")
    elif verdict == "CONFIRMED":
        print("\n  No single ability's removal collapses the interval -> it MAY be "
              "described as pool-wide.")

    # ---- SS6.4 companion (descriptive; does NOT OR into the verdict) -------
    print("\n=== SS6.4 COMPANION (descriptive only -- does NOT OR into SS6.3) ===")
    nets = []
    for t in gold_tags:
        gained = sum(1 for k in pool
                     if arms[t][k]["score"] >= COMPANION_T > arms["A"][k]["score"])
        lost = sum(1 for k in pool
                   if arms["A"][k]["score"] >= COMPANION_T > arms[t][k]["score"])
        nets.append((gained - lost) / 128 * 100)
        print(f"  {t:4s} D={gained + lost:3d}  gained {gained}  lost {lost}  "
              f"net {gained - lost:+d} ({nets[-1]:+.3f}pp)")
    nc, nsd = statistics.fmean(nets), statistics.stdev(nets)
    print(f"  four-arm net {ci(nc, T3 * nsd / 2)}  (SD_arm {nsd:.3f}pp, t3)")

    # ---- SS6.6 per-ability -------------------------------------------------
    print("\n=== SS6.6 PER-ABILITY (descriptive only; n=16, one flip = 6.25pp) ===")
    over = 0
    for ab in POOL:
        ks = [k for k in pool if k[0] == ab]
        means = [statistics.fmean(d(t, k) for k in ks) * 100 for t in gold_tags]
        over += abs(statistics.fmean(means)) > 6.25
        c, sd = statistics.fmean(means), statistics.stdev(means)
        print(f"  {ab:4s} {ci(c, T3 * sd / 2)}   "
              f"A={statistics.fmean(arms['A'][k]['score'] for k in ks) * 100:.2f}")
    print(f"  {over}/8 exceed the 6.25pp one-flip floor. The original SS8's "
          f"heterogeneity argument is still NOT re-banked -- these are n=16 "
          f"estimates, and SS6.7, not SS6.6, is what constrains the verdict's "
          f"wording.")

    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
