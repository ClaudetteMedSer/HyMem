#!/usr/bin/env python3
"""Where does the 8.4% verdict churn live -- the reader, or the judge?

`lme_noise_model.py` established that gate 4's resolution is set by churn and
not by sample size: 42 of 500 paired questions flipped between two runs of an
IDENTICAL arm, so the minimum detectable effect is 2.54pp however the bar is
drawn. LME-S has only 500 questions, so n cannot be raised. The only lever is
the churn itself -- and which fix to reach for depends entirely on which side
of the pipeline produces it:

  * JUDGE-SIDE churn -- the two runs produced the SAME answer text and the
    judge scored it differently. Cheap to attack: the judge call is
    max_tokens=10, so majority-of-three costs two extra tiny calls per
    question against a reader call carrying the whole episode pool.
  * ANSWER-SIDE churn -- the reader produced different text. Attacking that
    means touching retrieval or decoding, and it is already temperature=0.0
    (`longmemeval_adapter.py:1099`); the residue is provider-side
    non-determinism, which no flag of ours removes.

So the split decides whether judge-voting is worth building or is theatre.

WHAT MAKES THIS INSTRUMENT NON-VACUOUS. Hypothesis identity is only evidence
about a flip if it VARIES. If no two runs ever produce the same answer string,
"0% judge-side" is a fact about exact string comparison, not about the judge,
and this module says UNAVAILABLE rather than reporting a zero. It therefore
always prints the identical-hypothesis rate among CONCORDANT questions as the
control: the decomposition carries information only to the extent that rate
differs from the discordant one.

Offline: reads two artifacts of the same arm, makes no call.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# 1.96 sigma, two-sided alpha = 0.05 -- the same convention as lme_noise_model.
Z95 = 1.959963985


def _reject_retrieval_only(*artifacts):
    from run_registry import is_retrieval_only
    bad = [i for i, art in enumerate(artifacts) if is_retrieval_only(art)]
    if bad:
        raise ValueError(
            "a --retrieval-only artifact has no verdicts to be discordant "
            "about; churn cannot be measured from it")


def norm_hypothesis(s) -> str:
    """Whitespace-insensitive answer identity.

    A reply that differs only in wrapping is not a different answer, and
    counting it as reader churn would inflate the side of the split that is
    expensive to fix. The exact-match count is reported alongside so the
    choice is visible rather than buried."""
    return " ".join(str(s or "").split())


def binomial_upper_95(k: int, n: int) -> float | None:
    """One-sided 95% upper bound on a rate, exact (Clopper-Pearson).

    Solved by bisection on the binomial CDF rather than pulled from scipy,
    which is not a dependency of this repo. At k=0 it reduces to the familiar
    rule of three: 0/181 flips bounds the judge's flip rate at ~1.6%, which is
    what may honestly be claimed -- not "the judge is deterministic"."""
    if n <= 0:
        return None
    if k >= n:
        return 1.0

    def cdf(p: float) -> float:
        return sum(math.comb(n, i) * p ** i * (1.0 - p) ** (n - i)
                   for i in range(k + 1))

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if cdf(mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# The fields describing what the reader was HANDED, as recorded per question.
# Counts, not content -- see `context_fingerprint`.
CONTEXT_FIELDS = (
    "n_episodes", "n_facts", "n_procedures", "n_agg_nodes", "num_memories",
    "ability_used", "recall_tier", "gold_in_episodes", "gold_in_facts",
    "distill_fired", "distill_kept",
)


def fingerprint_mode(rows) -> str:
    """"exact" if every row carries `context_sha`, "counts" if none do.

    "mixed" is the case that matters. An artifact predating `context_sha`
    compared against one that has it would find NO fingerprint ever matching
    and read as "every flip moved retrieval" -- a fact about the two schemas,
    not about the runs. The comparison is refused rather than reported."""
    have = sum(1 for r in rows if r.get("context_sha"))
    if have == 0:
        return "counts"
    return "exact" if have == len(rows) else "mixed"


def context_fingerprint(r: dict, mode: str = "counts") -> tuple:
    """What the reader saw, to the resolution the artifact records.

    READ THE LIMIT. These are COUNTS and flags, not the retrieved text: two
    runs can hand the reader the same NUMBER of different episodes. So a
    fingerprint that MATCHES is weak evidence the context was identical,
    while one that DIFFERS is strong evidence it was not. That asymmetry is
    why the split below is reported as a lower bound on retrieval churn and
    an upper bound on decoder churn, never as a partition.

    It is the same lower-bound caveat `guard_score.fired_subset` carries, for
    the same reason, and it must not be quietly dropped here.

    Under mode="exact" the row carries `context_sha` -- a hash of the rendered
    reader prompt, added to the adapter for exactly this -- and the caveat
    lifts: the split becomes a partition rather than a pair of bounds."""
    if mode == "exact":
        return ("sha", r.get("context_sha"))
    return ("counts",) + tuple(r.get(k) for k in CONTEXT_FIELDS)


def is_scored(r: dict) -> bool:
    """D3: a judge that never answered is UNSCORED, not wrong.

    Such a row has no verdict to flip, so counting it as concordant would
    dilute the churn rate with rows the judge never read."""
    return r.get("correct") is not None and not r.get("judge_error")


def decompose(a: dict, b: dict) -> dict:
    _reject_retrieval_only(a, b)
    ar = {r["question_id"]: r for r in a.get("per_question", [])}
    br = {r["question_id"]: r for r in b.get("per_question", [])}
    shared = sorted(set(ar) & set(br))
    if not shared:
        raise ValueError("the two runs share no question ids")

    unscored = [q for q in shared if not (is_scored(ar[q]) and is_scored(br[q]))]
    usable = [q for q in shared if q not in set(unscored)]

    modes = {fingerprint_mode([ar[q] for q in usable]),
             fingerprint_mode([br[q] for q in usable])}
    fp_mode = modes.pop() if len(modes) == 1 else "mixed"
    if fp_mode == "mixed":
        raise ValueError(
            "one run records context_sha and the other does not; every "
            "fingerprint would differ and the split would describe the two "
            "schemas rather than the two runs")

    def fp(r):
        return context_fingerprint(r, fp_mode)

    judge_side, answer_side, conc_same, conc_diff = [], [], [], []
    # Answer-side flips split by whether the reader's INPUT also moved.
    ctx_same, ctx_diff = [], []
    conc_ctx_same = 0
    conc_diff_ctx_moved = 0
    exact_same = 0
    impossible = []
    for q in usable:
        ra, rb = ar[q], br[q]
        same_hyp = norm_hypothesis(ra.get("hypothesis")) == \
            norm_hypothesis(rb.get("hypothesis"))
        if str(ra.get("hypothesis") or "") == str(rb.get("hypothesis") or ""):
            exact_same += 1
        flipped = bool(ra.get("correct")) != bool(rb.get("correct"))
        if flipped and same_hyp:
            judge_side.append(q)
            # Same answer AND the same raw judge reply, yet a different
            # verdict, would mean the PARSER is non-deterministic. It is pure
            # string logic, so this set must be empty; if it is not, the
            # artifact is lying and the split cannot be read.
            if str(ra.get("judge_raw") or "") == str(rb.get("judge_raw") or ""):
                impossible.append(q)
        elif flipped:
            answer_side.append(q)
            (ctx_same if fp(ra) == fp(rb)
             else ctx_diff).append(q)
        else:
            if fp(ra) == fp(rb):
                conc_ctx_same += 1
            if same_hyp:
                conc_same.append(q)
            else:
                conc_diff.append(q)
                # The MATCHED control: the reader's text moved here too, and
                # the verdict did not. Fingerprint movement in this group is
                # movement that demonstrably did NOT cause a flip.
                if fp(ra) != fp(rb):
                    conc_diff_ctx_moved += 1

    n = len(usable)
    disc = len(judge_side) + len(answer_side)
    conc = len(conc_same) + len(conc_diff)
    ident_disc = len(judge_side) / disc if disc else None
    ident_conc = len(conc_same) / conc if conc else None
    return {
        "shared": len(shared),
        "unscored": len(unscored),
        "n": n,
        "discordant": disc,
        "judge_side": len(judge_side),
        "answer_side": len(answer_side),
        "concordant_same_hyp": len(conc_same),
        "concordant_diff_hyp": len(conc_diff),
        "exact_same_hypothesis": exact_same,
        "identical_rate_discordant": ident_disc,
        "identical_rate_concordant": ident_conc,
        # No question anywhere repeated its answer -> hypothesis identity is
        # constant, and a constant cannot explain a difference.
        "available": (len(judge_side) + len(conc_same)) > 0,
        "parser_impossible": impossible,
        "judge_share": (len(judge_side) / disc) if disc else None,
        # What the MDE would become if judge-side churn were eliminated
        # outright. The floor a perfect judge buys -- not a promise that
        # majority voting reaches it.
        "mde_pp": 100.0 * Z95 * math.sqrt(disc) / n if disc and n else None,
        "mde_pp_judge_free": (100.0 * Z95 * math.sqrt(len(answer_side)) / n
                              if len(answer_side) and n else None),
        # Answer-side churn, split by whether the retrieved context moved
        # too. Lower/upper bounds, not a partition -- see context_fingerprint.
        "retrieval_side_min": len(ctx_diff),
        "decoder_side_max": len(ctx_same),
        "context_identical_rate_concordant": (conc_ctx_same / conc
                                              if conc else None),
        # Fingerprint-moved rate among DISCORDANT, against the matched group:
        # concordant questions whose answer text also moved. If these agree,
        # retrieval movement does not predict a flip and the count above is
        # an association that is not there.
        "ctx_moved_rate_discordant": (len(ctx_diff) / (len(ctx_diff) + len(ctx_same))
                                      if (len(ctx_diff) + len(ctx_same)) else None),
        "ctx_moved_rate_concordant_moved_answer": (
            conc_diff_ctx_moved / len(conc_diff) if conc_diff else None),
        "context_available": (len(ctx_same) + conc_ctx_same) > 0,
        "fingerprint_mode": fp_mode,
        # A judge that saw byte-identical text twice and did not change its
        # mind. Zero flips out of a finite sample is NOT a zero rate, and
        # reporting it as one is the same overclaim as a bar with no interval.
        "judge_identical_pairs": len(judge_side) + len(conc_same),
        "judge_flip_rate": (len(judge_side) / (len(judge_side) + len(conc_same))
                            if (len(judge_side) + len(conc_same)) else None),
        "judge_flip_upper_95": binomial_upper_95(
            len(judge_side), len(judge_side) + len(conc_same)),
        "judge_side_ids": judge_side,
        "answer_side_ids": answer_side,
    }


def report(d: dict, out=print) -> dict:
    out("=== churn decomposition (two runs of ONE arm) ===")
    out(f"  shared questions: {d['shared']}   unscored (judge error): "
        f"{d['unscored']}   usable: {d['n']}")
    out("")
    if not d["available"]:
        out("  UNAVAILABLE — no question produced the same answer text twice,")
        out("  so hypothesis identity is constant and cannot explain a flip.")
        out("  This is a limit of exact-text comparison, NOT a finding that")
        out("  the judge is stable.")
        return d
    if d["parser_impossible"]:
        out(f"  ⚠ {len(d['parser_impossible'])} rows have the same answer AND "
            "the same raw judge reply")
        out("    but different verdicts. The parser is pure string logic, so "
            "that cannot")
        out("    happen: the artifact is inconsistent and the split below is "
            "not readable.")
        out(f"    ids: {', '.join(d['parser_impossible'][:5])}")
        out("")
    out(f"  discordant: {d['discordant']}/{d['n']} "
        f"({100.0 * d['discordant'] / d['n']:.1f}%)")
    out(f"    judge-side  (same answer, different verdict): {d['judge_side']}")
    out(f"    answer-side (different answer):               {d['answer_side']}")
    out(f"  concordant: {d['concordant_same_hyp'] + d['concordant_diff_hyp']}"
        f"  [same answer {d['concordant_same_hyp']}, "
        f"different {d['concordant_diff_hyp']}]")
    out("")
    out("=== power check — does answer identity carry information? ===")
    out(f"  same answer, among DISCORDANT: "
        f"{d['identical_rate_discordant']:.0%}")
    out(f"  same answer, among CONCORDANT: "
        f"{d['identical_rate_concordant']:.0%}")
    out("  The split is informative only insofar as these differ. If they")
    out("  match, answer identity predicts nothing about a flip and the")
    out("  judge/reader attribution below is not supported.")
    out("")
    out("=== what a perfect judge would buy ===")
    out(f"  judge-side share of churn: {d['judge_share']:.0%}")
    out(f"  judge saw byte-identical text twice on "
        f"{d['judge_identical_pairs']} questions "
        f"and changed its mind on {d['judge_side']}")
    if d["judge_flip_upper_95"] is not None:
        out(f"  judge flip rate <= {d['judge_flip_upper_95']:.1%} "
            "(one-sided 95%, Clopper-Pearson)")
        out("  NOT 'the judge is deterministic' — a rate of zero out of a")
        out("  finite sample is an interval, and that is the interval.")
    out(f"  MDE now:                       {d['mde_pp']:.2f}pp")
    if d["mde_pp_judge_free"] is None:
        out("  MDE with judge churn removed:  0.00pp (no answer-side churn "
            "at all)")
    else:
        out(f"  MDE with judge churn removed:  "
            f"{d['mde_pp_judge_free']:.2f}pp")
    out("  That is a FLOOR, not a forecast: majority voting reduces judge")
    out("  churn, it does not abolish it.")
    out("")

    out("=== answer-side churn: our retrieval, or the provider's decoder? ===")
    if not d["context_available"]:
        out("  UNAVAILABLE — no two runs ever handed the reader a matching")
        out("  context fingerprint, so the fingerprint is constant and cannot")
        out("  separate the two. A limit of the comparison, not a finding.")
        return d
    exact = d["fingerprint_mode"] == "exact"
    out(f"  fingerprint: {d['fingerprint_mode']}"
        + ("  (context_sha — the rendered reader prompt, hashed)" if exact
           else "  (counts and flags, not content)"))
    out(f"  retrieval moved too: "
        f"{d['retrieval_side_min']}" + ("" if exact else " (>= this many)"))
    out(f"  fingerprint matched, so decoder-side: "
        f"{d['decoder_side_max']}" + ("" if exact else " (<= this many)"))
    out("")
    out("  power check — does a moved fingerprint predict a flip?")
    r_d = d["ctx_moved_rate_discordant"]
    r_c = d["ctx_moved_rate_concordant_moved_answer"]
    out(f"    fingerprint moved, among answer-side FLIPS:        "
        f"{'—' if r_d is None else format(r_d, '.0%')}")
    out(f"    fingerprint moved, among NON-flips whose answer also moved: "
        f"{'—' if r_c is None else format(r_c, '.0%')}")
    out("    That second group is the matched control: the reader's text")
    out("    moved there too and the verdict held. If the two rates agree,")
    out("    retrieval movement does not predict a flip, and the count above")
    out("    is an association that is not present.")
    out("")
    if exact:
        out("  A PARTITION. The fingerprint is the rendered reader prompt,")
        out("  hashed, so the decoder-side questions were handed")
        out("  byte-identical text and nothing of ours produced the")
        out("  difference there.")
    else:
        out("  BOUNDS, NOT A PARTITION. The fingerprint is counts and flags,")
        out("  not the retrieved text: two runs can hand the reader the same")
        out("  NUMBER of different episodes. A fingerprint that DIFFERS proves")
        out("  the input moved; one that MATCHES does not prove it did not. So")
        out("  retrieval is a lower bound and the decoder an upper bound.")
        out("  Runs carrying `context_sha` lift this caveat.")
    out("")
    if r_d is not None and r_c is not None and r_d > r_c:
        out("  Moved retrieval is OVER-represented among flips, so some of")
        out("  this churn is OURS and a flag of ours could reach it — the")
        out("  reader already runs at temperature=0.0.")
    elif r_d is not None and r_c is not None:
        out("  Moved retrieval is NOT over-represented among flips: it is at")
        out("  least as common where the verdict held. Retrieval churn is real")
        out("  but is not what is flipping verdicts, and 'fix retrieval' does")
        out("  not follow from these counts.")
    elif d["retrieval_side_min"]:
        out("  Moved retrieval seen, but with no matched control there is")
        out("  nothing to compare it against.")
    else:
        out("  No flip carried a moved fingerprint: nothing here points at our")
        out("  retrieval, and the residue is provider-side non-determinism at")
        out("  temperature=0.0, which no flag of ours removes.")
    return d


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    args = p.parse_args()
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    report(decompose(a, b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
