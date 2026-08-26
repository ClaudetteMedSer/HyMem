#!/usr/bin/env python3
"""Audit of `longmemeval_adapter.judge_answer` — the single function behind
LoCoMo 68.2%, LME 68.4% and MSC ~84.0%.

    NOT YET RUN. The thresholds below are PRE-REGISTERED: they were written and
    committed before this script made its first LLM call, precisely so the
    verdict cannot be chosen after seeing the numbers.

WHAT IS BROKEN, AND WHY IT IS UNMEASURED
========================================
`judge_answer` is two lines::

    raw = llm.chat(messages, temperature=0.0, max_tokens=10)
    return "yes" in raw.lower()

`tests/test_judge_instrument.py` pins its behaviour offline. Two defect classes
survive that pinning because they need a real judge to size:

D1. THE CRITERION. The five non-abstention judge prompts instruct CONTAINMENT —
    "answer yes if the response contains the correct answer". A hedged refusal
    that recites context ("I can't tell which is her PB, though the context
    mentions 3:42") literally contains it. The `_abs` branch asks the RIGHT
    question ("does the model correctly identify the question as unanswerable"),
    so the two criteria disagree on exactly this shape. The offline test proves
    the judge is handed BYTE-IDENTICAL instructions for a committed answer and
    for a reciting refusal; how often a real judge then says "yes" is what this
    script measures.

    Blast radius is not uniform. MSC hard-codes `single-session-user` at its only
    judge call site, so 100% of MSC rows are containment-judged. LoCoMo routes
    only category 5 to the `_abs` branch. LME routes every `*_abs` type.

D2. THE DECISION RULE. `"yes" in raw.lower()` is an unanchored substring test.
    It is correct on compliant replies and fails safe on the empty reply, and it
    is wrong on "yes and no", on "not yes", and — the amplifier — on "yes" INSIDE
    another word: "yesterday" and "eyes" both score CORRECT. "yesterday" is not
    hypothetical; temporal-reasoning is a scored category on both LME and LoCoMo
    and is exactly where a non-compliant judge reply would use that word. The
    "no" half of a reply is never consulted, so every conflict resolves to
    correct. `max_tokens=10` truncates any non-compliant reply mid-sentence and
    the fragment is then scored as if it were a verdict.

    Nothing records `raw`. `judge_answer` returns a bare bool, so the rate is not
    merely unmeasured, it is unmeasurABLE from any stored run — the existing
    `--rejudge` paths in both adapters call `judge_answer` and inherit the same
    blindness. Recording `raw` is this script's entire reason to exist.

D3 (incidental, recorded here because the audit will surface it). `LLMClient.chat`
    returns `"[LLM_ERROR: ...]"` after exhausting retries. `judge_answer` scores
    that sentinel exactly like a genuine "no": the question is recorded WRONG and
    the outage is invisible. Both `--rejudge` paths test for the sentinel on the
    ANSWER only, never on the judge reply.

PRE-REGISTERED CRITERIA
=======================
Four numbers decide this, and they are banked here BEFORE the first run.

The unit of materiality throughout is "questions that could move", compared
against the smallest effect this battery is actually asked to resolve. Two
anchors from the project ledger set that scale:

  * the only measured signal in Campaign E is the LoCoMo E1 fired-subset McNemar
    at z = -2.40: b=10 / c=24, a net of 14 questions out of 800 with 34
    discordant pairs. That is the resolution the instrument must not blur.
  * a prior re-judge measured ZERO judge flips on 200 rows. Judge
    nondeterminism is therefore ~0, which has a sharp consequence: anything this
    audit finds is SYSTEMATIC, not noise. It does not average out with n, and it
    cannot be absorbed by the ~4-5%/question churn band — that band is defined
    for RANDOM reader churn and widens a confidence interval; a systematic
    criterion defect SHIFTS the point estimate. Bands do not protect against bias.

C1. MISSCORE RATE (D2) — the primary decision number.
    Definition: share of judged rows where the shipping substring rule and the
    reference first-token rule DISAGREE. This is the exposure, not the raw
    non-compliance rate: a non-compliant reply that still scores the way the
    judge meant costs nothing.

        >= 1.0%  MATERIAL. Fix the parse and re-baseline LoCoMo, LME and MSC.
        0.2-1.0% WATCH. Record the rate; do not re-baseline on it alone.
        <  0.2%  IMMATERIAL. Bank the number and close D2.

    Why 1.0%: on LoCoMo n=800 that is 8 questions. The measured E1 effect the
    battery rests on is a 14-question net over 34 discordant pairs. A defect
    source able to move 8 questions is roughly half that discordant mass and
    cannot be assumed away. On LME n=500 it is 5 questions, which is the same
    size as a per-category delta the ledger already classes as noise on a 70-item
    category — so 1.0% sits exactly at the boundary where a defect stops being
    absorbable.
    Why 0.2%: below ~1-2 questions per benchmark nothing can change a rank
    ordering at any n this repo runs.

    C1 IS A LOWER BOUND, not an estimate. The reference rule shares one blind
    spot with the shipping rule: "not yes" carries a word-boundary "yes" and no
    word-boundary "no", so both rules score it correct and the disagreement
    column cannot see it. Negated affirmatives are therefore counted separately
    (C1b) and are ADDITIVE to C1 when judging materiality. If C1 lands in WATCH
    and C1b is non-trivial, treat the pair as MATERIAL.

C2. NON-COMPLIANCE RATE (D2) — the leading indicator, NOT the decision number.
    Share of judged rows whose reply is not a bare yes/no after normalisation.
    Reported always, because it bounds C1 from above and because a rate that
    climbs after a model migration is the early warning that C1 will follow.
    No threshold triggers a re-baseline on its own. If C2 is high while C1 is
    near zero, the judge is chatty but consistent and only the parse is fragile.

C3. REFUSAL-SCORED-CORRECT RATE (D1) — the absolute-baseline number.
    Share of judged rows where the model answer is a refusal AND the judge
    scored it correct. Restricted to non-`_abs` rows: on an `_abs` row a
    refusal SHOULD score correct, and counting those would be a ceiling
    instrument reporting a huge, meaningless number.

        >= 2.0%  MATERIAL. The criterion is wrong in practice. Fix
                 `get_judge_prompt` and re-baseline all three canonical numbers.
        0.5-2.0% WATCH.
        <  0.5%  IMMATERIAL. Close D1 as a theoretical defect that does not fire.

    Why 2.0%: on LoCoMo n=800 that is 16 questions = 2.0pp on the headline,
    larger than the entire measured E1 all-800 net (-1.4pp) and, being
    one-directional, not absorbable by the churn band. A prior exists and it is
    close to this line: the Campaign E hand-check found 6 rows in a 66-flip LME
    dump where BOTH arms refused yet scored discordant, the judge crediting a
    gold value recited inside a refusal. Against ~500 LME rows that is ~1.2%,
    i.e. already in the WATCH band before this script runs. The ledger records
    that check as UNRUN on LoCoMo, which is where the battery's only significant
    result lives.
    Why 0.5%: ~4 questions on LoCoMo, below the discordant mass of any gate.

C4. ARM-ASYMMETRY (D1, and the reason a bad criterion can survive a gate).
    A criterion defect that fires equally on both arms of an A/B LARGELY CANCELS
    in a paired comparison: it biases the absolute baseline without biasing the
    delta. It stops cancelling exactly when the two arms differ in how often the
    reader refuses — which is precisely what happens when the feature under test
    changes abstention behaviour (E1 lost both `_abs` abstentions, 2/2).

        Refusal-rate difference between two compared arms >= 1.0pp
        => the defect does not cancel, and any gate decision resting on that
           pair is VOID until re-judged under a fixed criterion.

    This one costs NOTHING: it is a lexical classification of stored answers in
    two run files. `--pair A.json B.json` computes it with no LLM call at all,
    and it should be run BEFORE spending anything on C1/C3 — the same shape as
    the E4 free Step-0 pre-check that killed Fork A for the price of arithmetic.

DEGENERACY GUARDS (the trap this project has hit repeatedly: a ceiling
instrument, a degenerate criterion and an unreachable code path all read as PASS)
==============================================================================
G1. NO REFUSALS IN SAMPLE. If the refusal classifier flags zero rows, C3 has
    MEASURED NOTHING. It must report UNMEASURED, never a clean 0.0%. A 0%
    refusal rate on a real run means the classifier is broken or the sample is
    wrong, not that the criterion is safe.
G2. NO `_abs` ROWS. MSC has none by construction and a LoCoMo sample can easily
    have none. Then the abstention-branch contrast is unmeasured and the report
    says so instead of implying the branches agree.
G3. CLASSIFIER CEILING. If the refusal classifier flags more than 50% of rows it
    is matching something generic; the run is declared BROKEN rather than
    reporting a spectacular rate.
G4. HAND-CHECK REQUIRED — AND IT CORRECTS THE NUMBER, not just the gate. The refusal classifier and the gold-recitation check
    are lexical, and this project has already been burned by a surface check
    that false-positived at 55% against an 11% correct-answer control. So the
    verdict is INCOMPLETE — never PASS, never MATERIAL — until hand-check counts
    are supplied via `--handcheck FP,FN`. The script writes a labelled sample
    containing BOTH refusal-classified and committed-classified rows, so the
    control is a correct-answer control and not just a confirmation pass.
    The standing discipline is docs/diagnostic_controls.md.
G5. NOTHING JUDGED. Rows whose answer is empty or an `[LLM_ERROR]` sentinel are
    not judgeable; if that leaves no rows, the report says so.

COST DISCIPLINE
===============
No LLM client is constructed at import, and none is constructed at all without
an explicit `--spend`. The default run is a free pre-check: it loads the rows,
runs every lexical classifier, applies the degeneracy guards and prints the
exact call count a real pass would cost. Spend is one judge call per judged row.

    # free, spends nothing — run this first
    python judge_audit.py --run locomo_canonical.json --bench locomo
    python judge_audit.py --pair e1_on.json e1_off.json --bench locomo

    # costs tokens; one judge call per sampled row
    python judge_audit.py --run locomo_canonical.json --bench locomo \\
        --limit 200 --spend --judge-model gpt-oss-120b --out judge_audit.json

    # then hand-check the labelled sample it wrote, and re-report with controls
    python judge_audit.py --report judge_audit.json --handcheck FP,FN

POST-RUN LEDGER — LME pass 2026-08-25
====================================
    python judge_audit.py --run <LME canonical run json> --bench lme --spend \
        --handcheck 3,0 --judge-model deepseek-v4-flash \
        --judge-extra-body '{"thinking":{"type":"disabled"}}'

Judge = the shipping judge of the audited run (its metadata records
judge_model deepseek-v4-flash @ api.deepseek.com). The extra body is REQUIRED
for v4-flash: without it the reasoning preamble consumes the budget and content
comes back empty (probed: reasoning_content present, content '', finish=length).
500 rows, 500 judgeable, 30 _abs, 112 non-_abs refusals, 0 judge LLM_ERROR.

C1 0.00% — IMMATERIAL, but as a LOWER BOUND on a compliant judge: C2 0.00%
(every reply a bare yes/no) means the substring rule never had a non-compliant
reply to misfire on. It is LATENT in judge_answer, not absent. This run is
exactly the anchor-it-while-inert window.

C3 2.13% raw (10/470) -> hand-check corrected 1.87% (FP 3/25=12%, FN 0/25)
    -> WATCH. THE BAND IS NOT RESOLVED. Raw 10/470: Wilson 95% CI 1.16-3.87%.
    FP 3/25: Wilson 95% CI 4.2-30.0%. Breakeven FP for MATERIAL at m=10 is
    6.00% = 1.5/25; one hand-scored FP moves the verdict (0 FP -> 2.128%
    MATERIAL, 1 -> 2.043% MATERIAL, 2 -> 1.957% WATCH, 3 -> 1.872% WATCH).
    The 2.0% bar sits INSIDE both CIs: C3 ~= 2%, ON the bar, and no sample in
    hand places which side. Recorded as WATCH-at-bar, NOT as "below material".
    Numerator decomposition (hand-read): all 10 refusal-scored-correct rows are
    the containment criterion per its own prompt — no genuine judge error. 5
    pass the strict recitation test; the other 5 are ALL recites_gold FALSE
    NEGATIVES: the len(t)>2 filter excludes the numerals (the payload of
    temporal-reasoning golds — one row states BOTH "22 days" and "21 days"
    verbatim and still fails) while requiring the trailing "also acceptable"
    gloss; one row fails on the single word "taking"; the preference row is a
    paraphrase the rubric's "recalls and utilizes personal information" clause
    credits. => C3's size is D1 by design, not judge error; the instrument's
    recitation token rule, not the judge, is the mis-calibrated piece.

    RECITATION TOKEN RULE — PRE-REGISTERED AS ITS OWN CHANGE, NOT FOLDED IN.
    `recites_gold`'s len(t)>2 filter excludes the numerals and therefore
    under-counts reciting refusals. The C3 numerator is not the only thing it
    sizes: `free_precheck`'s ceiling numerator is `recites_gold` too. With the
    five FNs counted, the 2026-08-25 ceiling numerator is >=21 (>=4.47%), not
    16 (3.40%). Direction is conservative for the licence — an under-count can
    only wrongly refuse a spend, never wrongly license one — and this spend was
    reachable either way. Any loosening of the token rule changes the licence
    arithmetic as well as the C3 decomposition; gate it on its own.

    THE GATE, PRE-REGISTERED 2026-08-26 BEFORE `recites_gold_v2` SCORED A ROW.
    v2 = numerals are content (zero tolerance), gloss is not, prose gets
    `RECITE_ALPHA_COVERAGE` slack. `recites_gold` still ALIASES v1 and does not
    move until this returns PASS on the banked replies.

      R1 recall     — the documented FN MECHANISMS are recovered on fixtures:
                      numeral-carrying gold, mandatory-gloss gold, one-word
                      prose miss. Bar: each fixture v1 False -> v2 True.
      R2 precision  — hand-check the rows `--verify-recitation` reports as
                      NEWLY flagged. Bar: FP <= 1 of the newly-flagged sample.
      R3 discrimin. — free negative control: re-score with golds SHUFFLED
                      against answers. Bar: v2 shuffled fire rate <= v1's
                      + 2.0pp AND v2 true-pair rate >= 3x its own shuffled
                      rate. This is the arm that answers the retracted
                      55%-vs-11% episode directly: a rule that fires as often
                      on mismatched pairs is measuring text volume.
      R4 licence    — restate `c3_ceiling_non_abs` under both rules. REPORTED,
                      not a pass bar. A ceiling above the bar licenses a spend
                      and never substitutes for it.

    R1 ALONE CANNOT BE THE GATE — it re-finds the rows the rule was written
    from, which is a confirmation pass. R2 and R3 are the gate, and they are
    the arms v1 never needed, because loosening reverses the safe direction.

    Fixtures are RECONSTRUCTIONS of the mechanisms recorded above, not the
    original rows: those live in `judge_audit.json` on the box, and whether v2
    recovers all five of them is measured by `--verify-recitation` in A5, not
    asserted here. A fixture pass is not that measurement.

    PASS (R1-R3) -> flip the alias to v2, restate the C3 footnote and the
    licence numerator in this docstring and in additional_planning.md.
    FAIL on R2 or R3 -> close, keep v1, record. Do NOT re-tune the token rule
    until it passes; that is what makes a bar a bar.

    STATUS: RUN 2026-08-26 on Afrodite, `lme_audit_spend.json` (500 records).
    VERDICT: **R2 FAIL -> alias stays v1. Closed, not re-tuned.**

      R1 mechanisms  CONFIRMED on real rows, not only fixtures: the 3 rows v2
                     NO LONGER flags are all the numeral defect ("2.5 years" vs
                     "3.5 years older"; "27m45s" vs "26m30s"; "10 times" vs an
                     enumerated ride list). v1 was wrong on all three.
      R2 precision   **FAIL. Bar was FP <= 1 of the newly-flagged sample; the
                     hand-check found 2** (f420262c mentions 3 of 4 airlines
                     inside a refusal; 4dfccbf8 mentions ukulele lessons and
                     Rachel separately while correctly refusing). It fails on
                     BOTH readings of "sample": 2 of 12 newly-flagged, and 2 of
                     the 3 refusal-arm rows. No reading of R2 passes.
      R3 discrimin.  PASS, cleanly. Shuffled control v1 4.22% / v2 4.02% over
                     498 mismatched pairs vs a 52.00% true-pair rate — 13x
                     separation, well inside both halves of the bar. This is
                     the arm that answers the retracted 55%-vs-11% episode, and
                     it says v2 measures recitation, not text volume.
      R4 licence     REPORTED: ceiling 16/470 = 3.40% (v1) -> 19/470 = 4.04%
                     (v2 nominal) -> 17/470 = 3.62% (v2 hand-checked honest).

    WHY R2 IS NOT WAIVABLE HERE, in the two shapes the waiver was argued:

      (a) "report not barred" is R4's clause, verbatim and only R4's. R2's
          clause is "Bar: FP <= 1". Reading the waiver one row up the table
          converts the gate into its own confirmation pass.
      (b) "the FPs only inflate the upper bound, never deflate" describes the
          hazard R2 was written to catch, not a mitigation. The banked
          direction argument holds only for the STRICT rule: an under-count
          "can only wrongly refuse a spend, never wrongly license one".
          Loosening reverses it, and the ceiling is what licenses the spend.

    THE DECOMPOSITION IS WORSE THAN THE BAR, and it is the finding worth
    keeping: 8 of v2's 12 new flags are non-refusal prose rows that do not move
    the ceiling at all. On the refusal arm — the ONLY population the licence
    reads — v2 is 1 TP / 3. v2's gains land where they do not count and its
    errors concentrate where they do.

    AND THE LICENCE PREDICTION IS FALSIFIED INDEPENDENTLY. This block
    pre-registered ">=21 (>=4.47%)" with the five FNs counted. v2 measured 19
    (4.04%) nominal and 17 (3.62%) honest — below the pre-registration on both.
    So even waiving R2 entirely, the flip does not reach the numerator that
    motivated the work: the spend stays unlicensed either way. The flip would
    cost the frozen baseline and buy nothing.

    REVIVAL, if any, is a NEW gate with its own pre-registration written before
    scoring — not a re-tune of this one. The open question A5 actually surfaced
    is whether the refusal arm needs a different rule from the prose arm, since
    v2's precision splits ~10/12 overall against 1/3 there. Do not answer it by
    adjusting RECITE_ALPHA_COVERAGE until that gate exists.

D2 CLOSED BY FIX, 2026-08-25. `judge_answer` now calls
    `longmemeval_adapter.parse_judge_verdict`: word-boundary tokens, first
    verdict wins, negated affirmatives and the `[LLM_ERROR: ...]` sentinel score
    False. Landed in the INERT window this run established — C2 0.00%, C1b 0
    negated-yes, 0 judge-side sentinels — so it moved no canonical number and
    LoCoMo/LME/MSC are NOT re-baselined. `--verify-parse <this file>` re-scores
    the stored replies under the frozen legacy rule and the live one and must
    report 0 flips; it also reports how many replies COULD have flipped, because
    on an all-compliant corpus 0 flips is vacuous rather than reassuring.
    `shipping_verdict` is now the FROZEN pre-fix rule and must never be
    re-synced. Two shapes deliberately unchanged and pinned by test: "yes and no"
    (a criterion question, D1) and a truncated fragment carrying a bare "yes"
    (needs reply structure, and `max_tokens=10` is the frozen contract).

D3 CLOSED BY FIX, 2026-08-26. The visibility half. `judge_scored` returns
    `correct=None` for a sentinel — UNSCORED, not wrong — and all six call sites
    across the three adapters route through it; the ~15 accuracy summations drop
    unscored rows once per reporting entry point, and `judge_error_note` reports
    the count WITH its denominator (a zero over a run that made no judge calls
    is not reassurance). `judge_answer` keeps its bare bool and its fail-closed
    False, unchanged: it is the certified function, so the channel is a sibling
    (`judge_answer_raw`) rather than a modification, and rule 3's identity with
    `reference_verdict` survives intact.

    Landed in the same inert window as D2 and on the same argument — 0
    judge-side sentinels over 500 replies means the filter is the identity on a
    clean run, and a test asserts that rather than claiming it. No canonical
    number moves.

    The adapters now persist `judge_raw` (~10 tokens/row). THIS MODULE'S REASON
    TO EXIST NARROWS ACCORDINGLY: `--run --spend` re-judges because `raw` was
    discarded, and on any run made after 2026-08-26 it no longer is. A future
    audit of such a run should re-score the STORED replies (the `--verify-*`
    shape) rather than pay for new ones. C4 stays blocked on the old pair, but
    is unblocked for the next one.

C4. NOT RUN — blocked, recorded not dropped. No scored LoCoMo run pair exists
    on this box; the only conv-26 artifact is a --diag-only dump (correct=null,
    empty ai_answer, no reader calls by construction), which would have audited
    nothing. Recovering the pair = 1,600 reader calls, not proportionate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Pre-registered thresholds, as data so the report cannot quietly disagree with
# the docstring.
C1_MATERIAL, C1_WATCH = 1.0, 0.2      # misscore rate, % of judged rows
C3_MATERIAL, C3_WATCH = 2.0, 0.5      # refusal-scored-correct rate, %
C4_MATERIAL = 1.0                     # arm refusal-rate difference, pp
G3_CEILING = 50.0                     # refusal classifier ceiling, %
HANDCHECK_K = 25                      # rows per arm in the hand-check sample
HANDCHECK_FP_BROKEN = 0.5             # classifier FP rate above which C3 is void


# ── Reply classification (D2) ───────────────────────────────────────────

_DECORATION = " \t\r\n.,;:!*_`\"'()[]-–—"

# Deliberately NOT proposed as the fix. It is a REFERENCE rule whose only job is
# to size the exposure: where it and the shipping rule disagree, the shipping
# rule scored something other than the judge's verdict. Choosing the actual
# replacement is a separate decision, and it re-baselines three numbers.
_YES_WORD = re.compile(r"\byes\b")
_NO_WORD = re.compile(r"\bno\b")
# A NEGATED affirmative. Called out separately because the reference rule shares
# the shipping rule's blind spot here: "not yes" contains a word-boundary "yes"
# and no word-boundary "no" ("not" is not "no"), so BOTH rules score it correct
# and the disagreement column cannot see it. That is why C1 is reported as a
# LOWER BOUND on misscoring rather than as the misscore rate itself.
_NEGATED_YES = re.compile(r"\b(?:not|never|isn't|is not|wasn't|n't)\b[^.]{0,20}?\byes\b")


def normalise_reply(raw: str) -> str:
    return (raw or "").strip().strip(_DECORATION).strip().lower()


def shipping_verdict(raw: str) -> bool:
    """The LEGACY rule, FROZEN. Until 2026-08-25 this was `judge_answer` verbatim;
    it is what produced LoCoMo 68.2%, LME 68.4% and MSC ~84.0%.

    The duplication was written so the audit would still report correctly once
    `judge_answer` was fixed — "at which point the disagreement column becomes
    the before/after diff". That has now happened, so DO NOT re-sync this to the
    live parse: it is the historical baseline every canonical number was scored
    under, and re-syncing would make the diff a constant zero by construction.
    `landed_verdict` reads the live rule; `--verify-parse` diffs the two."""
    return "yes" in (raw or "").lower()


def landed_verdict(raw: str) -> bool:
    """The rule `judge_answer` runs TODAY, imported rather than restated.

    Imported precisely because restating it is what let the legacy copy above
    drift into a second source of truth. Lazily, because the audit's free paths
    must not pull in the adapter (and its `requests` import) to count rows."""
    from longmemeval_adapter import parse_judge_verdict

    return parse_judge_verdict(raw)


def reference_verdict(raw: str) -> bool:
    """First word-boundary verdict token wins; no token at all means no."""
    low = (raw or "").lower()
    y, n = _YES_WORD.search(low), _NO_WORD.search(low)
    if y and n:
        return y.start() < n.start()
    return bool(y)


def classify_reply(raw: str) -> dict:
    """Bucket a raw judge reply. Buckets are mutually exclusive and ordered from
    most to least specific."""
    text = raw or ""
    low = text.lower()
    norm = normalise_reply(text)

    if text.startswith("[LLM_ERROR"):
        bucket = "llm_error"          # D3: silently scored as a wrong answer
    elif not text.strip():
        bucket = "empty"
    elif norm == "yes":
        bucket = "compliant_yes"
    elif norm == "no":
        bucket = "compliant_no"
    elif _NEGATED_YES.search(low):
        bucket = "negated_yes"        # invisible to BOTH rules — see C1 caveat
    elif _YES_WORD.search(low) and _NO_WORD.search(low):
        bucket = "both_tokens"        # "yes and no", "no ... yes"
    elif "yes" in low and not _YES_WORD.search(low):
        bucket = "yes_substring_only"  # "yesterday", "eyes" — the amplifier
    elif _YES_WORD.search(low):
        bucket = "verbose_yes"
    elif _NO_WORD.search(low):
        bucket = "verbose_no"
    else:
        bucket = "no_verdict_token"   # a truncated preamble with no verdict

    ship, ref = shipping_verdict(text), reference_verdict(text)
    return {
        "bucket": bucket,
        "compliant": bucket in ("compliant_yes", "compliant_no"),
        "shipping": ship,
        "reference": ref,
        "disagrees": ship != ref,
        "negated": bool(_NEGATED_YES.search(low)),
        "chars": len(text),
    }


# ── Refusal classification (D1) ─────────────────────────────────────────
# The reader is INSTRUCTED to emit a specific phrase family when it cannot
# answer (longmemeval_adapter.py:275/282 and the permissive prompt), so the
# canonical markers are high-precision by construction. The loose markers are
# reported separately: if they carry the count, the classifier is doing
# something the prompts never asked for and the hand-check matters more.

_CANONICAL_REFUSAL = (
    "i don't have enough information",
    "i do not have enough information",
    "don't have enough information",
    "not enough information",
)

_LOOSE_REFUSAL = (
    "i can't tell", "i cannot tell", "i can't determine", "i cannot determine",
    "i don't know", "i do not know", "unable to determine", "cannot be determined",
    "isn't mentioned", "is not mentioned", "no mention of", "not specified",
    "doesn't specify", "does not specify", "the context does not",
    "the context doesn't", "unclear from the context",
)


def classify_refusal(answer: str) -> dict:
    low = (answer or "").lower()
    canon = [m for m in _CANONICAL_REFUSAL if m in low]
    loose = [m for m in _LOOSE_REFUSAL if m in low]
    return {"refusal": bool(canon or loose),
            "canonical": bool(canon),
            "loose_only": bool(loose and not canon)}


_TOKEN = re.compile(r"[a-z0-9]+")
_DIGIT = re.compile(r"\d")
# A parenthetical gloss: "(21 days is also acceptable)". Gold authors use it to
# widen what counts as right, so its tokens are ALTERNATIVES, not requirements.
_PAREN_GLOSS = re.compile(r"\([^)]*\)")
# The same widening written without brackets, as a trailing clause.
_TRAILING_GLOSS = re.compile(
    r"[,;]?\s*(?:\b(?:or|and)\b\s+)?[^,;]*?\bis\s+also\s+"
    r"(?:acceptable|correct|fine|ok|okay|valid)\b.*$")

# v2's one tunable, PRE-REGISTERED before any row was scored (see R1-R4 in the
# module docstring). Numerals are zero-tolerance; alphabetic content tokens get
# this much slack, which is what recovers the row that failed on the single word
# "taking". It self-scales: a 5-token gold may miss one, a 2-token gold may not.
RECITE_ALPHA_COVERAGE = 0.8


def _gold_primary_clause(gold: str) -> str:
    """Gold minus the gloss that widens it.

    A gold of the form `22 days (21 days is also acceptable)` states ONE fact
    and then names a second acceptable rendering of it. v1 tokenised the whole
    string and required every token, so the gloss became mandatory content — an
    answer reciting the fact perfectly still failed for want of the words
    "also acceptable"."""
    g = _PAREN_GLOSS.sub(" ", gold)
    g = _TRAILING_GLOSS.sub(" ", g)
    stripped = g.strip(" ,;:-")
    # Never return an empty clause: a gold that is ENTIRELY gloss falls back to
    # the original string rather than matching everything vacuously.
    return stripped if _TOKEN.search(stripped) else gold


def _content_tokens(gold: str, keep_short_numerals: bool) -> tuple[list[str], list[str]]:
    """(numeric, alphabetic) content tokens of `gold`.

    `keep_short_numerals` is the v1/v2 difference in one flag: v1's `len(t) > 2`
    filter silently deleted "22", "21", "5" — the payload of every
    temporal-reasoning gold — while keeping the prose around them."""
    nums, alphas = [], []
    for t in _TOKEN.findall(gold):
        if _DIGIT.search(t):
            if len(t) > 2 or keep_short_numerals:
                nums.append(t)
        elif len(t) > 2:
            alphas.append(t)
    return nums, alphas


def recites_gold_v1(answer: str, gold: str) -> bool:
    """The rule FROZEN as of 2026-08-25. DO NOT re-sync it to v2.

    This is the rule that produced the numbers already in the ledger: C3's
    numerator 10/470, its `rsc_recite` decomposition 5 of 10, and the
    `free_precheck` ceiling 16/470 = 3.40% that licensed the 2026-08-25 spend.
    Re-syncing it would make `--verify-recitation`'s before/after diff a
    constant zero BY CONSTRUCTION — the same trap `shipping_verdict` is frozen
    against, and one test exists solely to fail on a well-meant tidy-up.

    STRICT gold recitation: the normalised gold string appears verbatim, or
    every gold content token (len > 2) appears in the answer.

    Deliberately strict. A loose similarity check on this exact shape is what
    false-positived at 55% against an 11% correct-answer control earlier in this
    project; the resulting number was retracted. Strictness here biases the
    measured rate DOWN, which is the safe direction for a defect hunt: a rate
    that clears C3 despite a strict check is real."""
    if not gold or not answer:
        return False
    a, g = answer.lower(), gold.lower().strip()
    if g and g in a:
        return True
    toks = [t for t in _TOKEN.findall(g) if len(t) > 2]
    return bool(toks) and all(t in a for t in toks)


def recites_gold_v2(answer: str, gold: str) -> bool:
    """Numerals are content; gloss is not. Hand-read from C3's own numerator.

    All 10 rows of the 2026-08-25 C3 numerator were read by hand: zero genuine
    judge errors, and 5 of 10 were v1 FALSE NEGATIVES. Two mechanisms, both
    fixed here:

      1. `len(t) > 2` discarded "22", "21" — the entire payload of a
         temporal-reasoning gold. One row states BOTH "22 days" and "21 days"
         verbatim and v1 still scores it False.
      2. `all(...)` over the whole gold made the widening gloss MANDATORY, so
         the answer had to recite "also acceptable" to count as reciting.

    THE SAFE DIRECTION IS NOW REVERSED, and that is the whole reason this is
    gated separately rather than folded into the audit. v1's under-count could
    only wrongly REFUSE a spend; v2 can wrongly LICENSE one, because
    `free_precheck`'s ceiling numerator is this function. So v2 keeps numerals
    at ZERO tolerance (a wrong number is not a recitation) and buys its slack
    only on prose, bounded by `RECITE_ALPHA_COVERAGE` and checked against a
    shuffled-gold negative control (R3) rather than against intuition."""
    if not gold or not answer:
        return False
    a, g = answer.lower(), gold.lower().strip()
    if g and g in a:
        return True
    nums, alphas = _content_tokens(_gold_primary_clause(g), keep_short_numerals=True)
    if not nums and not alphas:
        return False
    if any(t not in a for t in nums):
        return False
    if not alphas:
        return True
    hit = sum(1 for t in alphas if t in a)
    return hit >= RECITE_ALPHA_COVERAGE * len(alphas)


# The rule the audit REPORTS under. Every caller goes through this name, so the
# three call sites (`rejudge_row`, `free_precheck`,
# `write_handcheck_sample_pre`) do not each grow a v1/v2 branch.
# RAN 2026-08-26 and FAILED R2 (2 FP, bar was <=1) — see the A5 verdict block in
# the module docstring. This stays v1 permanently unless a NEW pre-registration
# is banked first; re-tuning the token rule until R2 clears is the move the
# banked verdict language forbids by name. Flipping it would restate the C3
# footnote and the spend licence on a rule that failed its own precision bar.
recites_gold = recites_gold_v1

# ── Row loading and judge-input reconstruction ──────────────────────────
# The judge input MUST be rebuilt exactly as the adapter built it. A re-judge
# that reconstructs it differently measures prompt drift, not the judge.

_TRAP_RE = re.compile(r"^\[unanswerable; trap: (.*)\]$", re.S)


def load_rows(path: str) -> list[dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    for key in ("per_question", "results", "rows"):
        if isinstance(obj.get(key), list):
            return obj[key]
    raise SystemExit(f"{path}: no per-question list found (keys: {sorted(obj)[:10]})")


def judge_inputs(row: dict, bench: str) -> tuple[str, str, str, str]:
    """(question_type, question, gold, answer) exactly as the adapter passes them."""
    if bench == "locomo":
        from locomo_adapter import CATEGORY_JUDGE, _gold_for_judge
        cat = row["category"]
        qtype = CATEGORY_JUDGE[cat]
        gold = row.get("answer")
        if cat == 5:
            m = _TRAP_RE.match(str(gold))
            gold = _gold_for_judge(5, None, m.group(1) if m else "")
        return qtype, row["question"], (gold if gold is not None else ""), \
            str(row.get("ai_answer") or "")
    if bench == "msc":
        # msc_adapter.py:513 hard-codes this type at its only judge call site.
        return "single-session-user", row.get("question", ""), \
            str(row.get("answer", "")), str(row.get("ai_answer") or "")
    if bench == "lme":
        return row.get("question_type", ""), row.get("question", ""), \
            str(row.get("answer", "")), str(row.get("hypothesis") or "")
    raise SystemExit(f"unknown --bench {bench!r}")


def judgeable(answer: str) -> bool:
    return bool(answer) and not answer.startswith("[LLM_ERROR")


def stable_sample(rows: list[dict], limit: int | None, key: str = "id") -> list[dict]:
    """Deterministic content-addressed sample, so re-running the audit re-judges
    the SAME rows instead of a fresh draw whose delta would be pure sampling."""
    if not limit or limit >= len(rows):
        return rows
    def h(r: dict) -> str:
        ident = str(r.get(key) or r.get("qa_id") or r.get("question_id")
                    or r.get("question", ""))
        return hashlib.sha256(ident.encode("utf-8")).hexdigest()
    return sorted(rows, key=h)[:limit]


# ── Recording judge wrapper ─────────────────────────────────────────────

class RecordingJudge:
    """Wraps ANY injected `.chat()`-shaped client and keeps the raw reply.

    This wrapper is the whole instrument. `judge_answer` throws `raw` away, so
    the only way to count non-compliant replies without touching production code
    is to record them one level below it."""

    def __init__(self, inner):
        self.inner = inner
        self.replies: list[str] = []

    def chat(self, messages: list, temperature: float = 0.1,
             max_tokens: int = 1024) -> str:
        raw = self.inner.chat(messages, temperature=temperature,
                              max_tokens=max_tokens)
        self.replies.append(raw)
        return raw


def rejudge_row(client, row: dict, bench: str) -> dict:
    """One judge call, with the raw reply retained. `client` is INJECTED — this
    module never builds one for you."""
    from longmemeval_adapter import judge_answer

    qtype, question, gold, answer = judge_inputs(row, bench)
    rec = RecordingJudge(client)
    verdict = judge_answer(rec, qtype, question, gold, answer)
    raw = rec.replies[-1] if rec.replies else ""
    ref = classify_reply(raw)
    ref.update({
        "id": row.get("id") or row.get("qa_id") or row.get("question_id"),
        "question_type": qtype,
        "is_abs": "_abs" in qtype,
        "verdict": verdict,
        "verdict_original": row.get("correct"),
        "raw": raw,
        **{f"answer_{k}": v for k, v in classify_refusal(answer).items()},
        "recites_gold": recites_gold(answer, gold),
    })
    return rec_with_answer(ref, answer, gold)


def rec_with_answer(rec: dict, answer: str, gold: str) -> dict:
    """Answer/gold text is carried ONLY into the hand-check file, never into the
    printed report. Benchmark text is public, but a report is for counts."""
    rec["_answer"] = answer
    rec["_gold"] = gold
    return rec


# ── Free pre-check (no LLM) ─────────────────────────────────────────────

def free_precheck(rows: list[dict], bench: str) -> dict:
    """Counts only, no LLM. The load-bearing output is `c3_ceiling_non_abs`.

    `refusals_reciting_gold` alone is NOT comparable to the C3 bar. C3 is
    defined over non-`_abs` rows only (`build_report` denominates on
    `len(non_abs)`), while an `_abs` row that refuses AND recites is the
    INTENDED behaviour, not the defect. Counting the two together inflates the
    ceiling by exactly the rows the criterion excludes, and on a benchmark
    whose refusals concentrate in the abstention slice that difference can
    straddle the 2.0% bar in either direction.

    So the split is reported, and `c3_ceiling_non_abs` — the share of judged
    non-`_abs` rows where a refusal recites gold — is the free upper bound on
    C3. Reading it:

      ceiling < C3_MATERIAL -> the material branch is UNREACHABLE on this run;
                               a paid re-judge cannot produce a MATERIAL verdict
                               and is not worth spending.
      ceiling >= C3_MATERIAL -> reachable but NOT established: the ceiling
                               assumes every such row is scored correct by the
                               judge, which is what the paid pass measures.

    It is a ceiling, never a reading. A ceiling above the bar licenses the
    spend; it does not substitute for it.
    """
    n = len(rows)
    judged, abs_rows, refusals, canon, recite = 0, 0, 0, 0, 0
    judged_non_abs, refusals_non_abs, recite_non_abs = 0, 0, 0
    for r in rows:
        qtype, _q, gold, answer = judge_inputs(r, bench)
        is_abs = "_abs" in qtype
        if is_abs:
            abs_rows += 1
        if not judgeable(answer):
            continue
        judged += 1
        if not is_abs:
            judged_non_abs += 1
        cls = classify_refusal(answer)
        if cls["refusal"]:
            refusals += 1
            canon += cls["canonical"]
            if not is_abs:
                refusals_non_abs += 1
            if recites_gold(answer, gold):
                recite += 1
                if not is_abs:
                    recite_non_abs += 1
    return {"n": n, "judgeable": judged, "abs_rows": abs_rows,
            "refusals": refusals, "refusals_canonical": canon,
            "refusals_reciting_gold": recite,
            "refusal_rate": pct(refusals, judged),
            "judgeable_non_abs": judged_non_abs,
            "refusals_non_abs": refusals_non_abs,
            "refusals_reciting_gold_non_abs": recite_non_abs,
            "refusals_reciting_gold_abs": recite - recite_non_abs,
            "c3_ceiling_non_abs": pct(recite_non_abs, judged_non_abs)}


def verify_parse(records: list[dict]) -> dict:
    """Re-score every STORED raw judge reply under the legacy rule and under the
    rule `judge_answer` runs today, and count where they differ.

    This is the whole post-hoc verification of the 2026-08-25 parse fix, and it
    is free: the replies were already paid for. A flip here is a row whose
    recorded verdict would change, i.e. a canonical number that moves. Zero
    flips means the fix is a no-op ON THIS CORPUS — the strongest claim the
    evidence supports, and NOT a claim that the rules agree in general (they
    demonstrably do not; that disagreement is the entire point of landing it).

    Reported per bucket as well as in total: an all-compliant corpus can only
    ever produce zero flips, so the bucket table is what distinguishes "the fix
    is inert" from "there was nothing here that could have flipped". C2 = 0.00%
    means this run is the SECOND case, and the verdict text must say so.
    """
    flips = [r for r in records
             if shipping_verdict(r.get("raw", "")) != landed_verdict(r.get("raw", ""))]
    buckets = Counter(r.get("bucket", "?") for r in records)
    could_flip = sum(n for b, n in buckets.items()
                     if b not in ("compliant_yes", "compliant_no", "empty"))
    return {
        "n": len(records),
        "flips": len(flips),
        "flip_ids": [r.get("id") for r in flips],
        "rows_that_could_flip": could_flip,
        "buckets": dict(buckets),
        "verdict": (
            "NO-OP CONFIRMED — but VACUOUSLY: every stored reply is a bare "
            "yes/no, so no reply existed that either rule could score "
            "differently. The fix is inert here because the corpus is "
            "compliant, not because the rules agree."
            if len(flips) == 0 and could_flip == 0 else
            f"NO-OP CONFIRMED on {could_flip} non-compliant replies that COULD "
            "have flipped and did not."
            if len(flips) == 0 else
            f"NOT A NO-OP — {len(flips)} stored verdict(s) change. The affected "
            "benchmark number moves and must be re-baselined, not silently "
            "re-scored."),
    }


def pct(a: int, b: int) -> float:
    return 100.0 * a / b if b else 0.0


# R3's bars, banked with the rest of the gate in the module docstring.
R3_SHUFFLE_MARGIN_PP = 2.0    # v2 may fire at most this much more on mismatched pairs
R3_DISCRIM_RATIO = 3.0        # v2's true-pair rate must beat its own shuffled rate by this


def _recite_pairs(records: list[dict]) -> list[tuple[dict, str, str]]:
    """(record, answer, gold) for every record carrying both texts.

    `--out` records keep `_answer`/`_gold` (`rec_with_answer` sets them and
    nothing strips them before the write), which is what makes this whole
    verification free: no row is re-judged, only re-scored."""
    out = []
    for r in records:
        a, g = str(r.get("_answer") or ""), str(r.get("_gold") or "")
        if a and g:
            out.append((r, a, g))
    return out


def _shuffled_control(pairs: list[tuple[dict, str, str]]) -> list[tuple[str, str]]:
    """Each answer re-paired with the NEXT row's gold. Deterministic.

    Pairs whose rotated gold happens to equal their own are dropped: a duplicate
    gold would smuggle a true pair into the control arm and inflate the floor
    the discriminability ratio is measured against."""
    n = len(pairs)
    if n < 2:
        return []
    return [(a, pairs[(i + 1) % n][2])
            for i, (_r, a, g) in enumerate(pairs)
            if pairs[(i + 1) % n][2].strip().lower() != g.strip().lower()]


def _ceiling_under(pairs: list[tuple[dict, str, str]], rule) -> tuple[int, int, float]:
    """(numerator, denominator, pct) for `c3_ceiling_non_abs` under `rule`.

    Denominator is judged non-`_abs` rows, numerator is the refusals among them
    that recite — the same split `free_precheck` computes and `build_report`
    divides by. Recomputed here rather than imported so the two rules can be
    reported side by side from one banked file."""
    den = [(r, a, g) for r, a, g in pairs if not r.get("is_abs")]
    num = sum(1 for r, a, g in den if r.get("answer_refusal") and rule(a, g))
    return num, len(den), pct(num, len(den))


def verify_recitation(records: list[dict]) -> dict:
    """Re-score every STORED (answer, gold) pair under frozen v1 and under v2.

    Free: the judge calls were already paid for, and this touches no judge at
    all — only the instrument's own recitation rule.

    THE VACUITY SPLIT IS THE POINT, exactly as in `verify_parse`. A row whose
    gold appears VERBATIM in the answer is decided by the fast path both rules
    share, so no token-rule change can move it. Reporting "N rows, M changed"
    without saying how many rows the token rule was even consulted on would be a
    certificate signed by an instrument that never met the surface it certifies.
    `token_rule_consulted` is that denominator.

    Three arms, mapping to the banked gate:

      newly_flagged / no_longer_flagged -> R2. The two rules are NOT nested:
        v2 is looser on prose and gloss but STRICTER on numerals (it requires
        the short ones v1 deleted), so it can un-flag a row v1 flagged. Both
        directions are reported and both are hand-checkable.
      shuffled control                  -> R3, free, no hand-check needed.
      ceiling_v1 / ceiling_v2           -> R4, reported not barred.
    """
    pairs = _recite_pairs(records)
    verbatim = [p for p in pairs
                if p[2].strip().lower() and p[2].strip().lower() in p[1].lower()]
    consulted = len(pairs) - len(verbatim)

    newly, lost = [], []
    for r, a, g in pairs:
        v1, v2 = recites_gold_v1(a, g), recites_gold_v2(a, g)
        if v2 and not v1:
            newly.append(r)
        elif v1 and not v2:
            lost.append(r)

    ctrl = _shuffled_control(pairs)
    ctrl_v1 = pct(sum(1 for a, g in ctrl if recites_gold_v1(a, g)), len(ctrl))
    ctrl_v2 = pct(sum(1 for a, g in ctrl if recites_gold_v2(a, g)), len(ctrl))
    true_v2 = pct(sum(1 for _r, a, g in pairs if recites_gold_v2(a, g)), len(pairs))

    n1, d1, c1 = _ceiling_under(pairs, recites_gold_v1)
    n2, d2, c2 = _ceiling_under(pairs, recites_gold_v2)

    r3_margin_ok = ctrl_v2 <= ctrl_v1 + R3_SHUFFLE_MARGIN_PP
    r3_ratio_ok = ctrl_v2 == 0.0 or true_v2 >= R3_DISCRIM_RATIO * ctrl_v2
    r3_pass = bool(ctrl) and r3_margin_ok and r3_ratio_ok

    if consulted == 0:
        verdict = ("VACUOUS — every stored pair is decided by the verbatim fast "
                   "path both rules share. The token rule was consulted on ZERO "
                   "rows, so 0 changes here cannot fail to be 0. This file "
                   "cannot answer R2 or R3.")
    elif not ctrl:
        verdict = ("R3 UNMEASURED — the shuffled control is empty (fewer than "
                   "two distinct golds), so discriminability is untested. Do "
                   "not read the change counts as a PASS.")
    elif not r3_pass:
        verdict = (f"R3 FAIL — v2 fires on {ctrl_v2:.2f}% of MISMATCHED pairs "
                   f"(v1 {ctrl_v1:.2f}%, margin {R3_SHUFFLE_MARGIN_PP}pp; true-pair "
                   f"{true_v2:.2f}%, ratio bar {R3_DISCRIM_RATIO}x). The rule is "
                   "measuring text volume, not recitation. Close and keep v1.")
    else:
        verdict = (f"R3 PASS — v2 fires on {ctrl_v2:.2f}% of mismatched pairs vs "
                   f"{true_v2:.2f}% of true pairs. R2 IS STILL OPEN: hand-check "
                   f"the {len(newly)} newly-flagged row(s) before flipping the "
                   "alias. R3 clearing does not license the flip on its own.")

    return {
        "n": len(pairs),
        "verbatim_decided": len(verbatim),
        "token_rule_consulted": consulted,
        "newly_flagged": len(newly),
        "newly_flagged_ids": [r.get("id") for r in newly],
        "no_longer_flagged": len(lost),
        "no_longer_flagged_ids": [r.get("id") for r in lost],
        "control_n": len(ctrl),
        "control_rate_v1": ctrl_v1,
        "control_rate_v2": ctrl_v2,
        "true_pair_rate_v2": true_v2,
        "r3_pass": r3_pass,
        "ceiling_v1": {"num": n1, "den": d1, "pct": c1},
        "ceiling_v2": {"num": n2, "den": d2, "pct": c2},
        "verdict": verdict,
    }


def write_recitation_sample(records: list[dict], out: str,
                            k: int = HANDCHECK_K) -> Path | None:
    """R2's hand-check arm: the rows v2 flags and v1 did not, plus the reverse.

    Written to a file because R2 is the arm that cannot be automated — a token
    rule cannot tell you whether a refusal genuinely recited the gold. Labelled
    by direction so the two are scored separately: they are different errors
    (`newly` = a wrongly-licensed spend, `lost` = a numeral v1 never checked)."""
    rows = []
    for r, a, g in _recite_pairs(records):
        v1, v2 = recites_gold_v1(a, g), recites_gold_v2(a, g)
        if v1 == v2:
            continue
        rows.append({"id": r.get("id"), "direction": "newly" if v2 else "lost",
                     "is_abs": r.get("is_abs"), "verdict": r.get("verdict"),
                     "answer_refusal": r.get("answer_refusal"),
                     "_gold": g, "_answer": a})
    if not rows:
        return None
    newly = [r for r in rows if r["direction"] == "newly"][:k]
    lost = [r for r in rows if r["direction"] == "lost"][:k]
    path = Path(out).with_suffix(".recitation.json")
    path.write_text(json.dumps({"newly": newly, "lost": lost}, indent=2),
                    encoding="utf-8")
    return path


# Licence codes, in the order the branches must be evaluated.
LICENCE_NO_REFUSALS = "NO_REFUSALS"
LICENCE_NO_DENOMINATOR = "NO_DENOMINATOR"
LICENCE_UNREACHABLE = "UNREACHABLE"
LICENCE_REACHABLE = "REACHABLE"


def c3_spend_licence(pre: dict) -> tuple[str, str]:
    """Decide, for free, whether a paid re-judge could move C3 at all.

    Extracted from `main` so it is reachable from a test. Inlined in the printer
    it was only assertable by reading source, and a guard whose test cannot fail
    is not a guard.

    ORDER IS LOAD-BEARING. The degenerate cases must be caught BEFORE the
    numeric comparison, because 0/0 renders as 0.00% and would otherwise fall
    into UNREACHABLE — reporting a confident 'nothing to find here' for a run
    that measured nothing. That is the E3 failure mode: a criterion with no
    denominator reading as PASS.
    """
    if pre["refusals"] == 0:
        return LICENCE_NO_REFUSALS, (
            "G1: zero refusals — a spend here would MEASURE NOTHING. Do not spend.")
    if pre["judgeable_non_abs"] == 0:
        return LICENCE_NO_DENOMINATOR, (
            "C3 UNMEASURABLE — every judgeable row is _abs, so the criterion has "
            "no denominator. The 0.00% ceiling above is 0/0, NOT a clean result.")
    ceil_ = pre["c3_ceiling_non_abs"]
    if ceil_ < C3_MATERIAL:
        return LICENCE_UNREACHABLE, (
            f"SPEND NOT LICENSED for C3: even if the judge scored EVERY reciting "
            f"refusal correct, C3 would be {ceil_:.2f}% < {C3_MATERIAL:.1f}% — the "
            f"MATERIAL branch is unreachable on this run. (C1 is a separate "
            f"criterion and may still justify the spend.)")
    return LICENCE_REACHABLE, (
        f"SPEND LICENSED for C3: the ceiling {ceil_:.2f}% clears the "
        f"{C3_MATERIAL:.1f}% bar, so MATERIAL is REACHABLE. This is a ceiling, "
        f"not a reading — it assumes the judge scores every one of those rows "
        f"correct. Only the paid pass says how many it actually does.")


def pair_precheck(rows_a: list[dict], rows_b: list[dict], bench: str) -> dict:
    """C4, for free. If two compared arms refuse at different rates, a criterion
    defect does not cancel in the paired comparison and the gate is void."""
    a, b = free_precheck(rows_a, bench), free_precheck(rows_b, bench)
    diff = abs(a["refusal_rate"] - b["refusal_rate"])
    return {"a": a, "b": b, "diff_pp": diff,
            "verdict": "VOID — refusal asymmetry, criterion defect does not cancel"
                       if diff >= C4_MATERIAL else
                       "OK — arms refuse at comparable rates; defect largely cancels"}


# ── Report ──────────────────────────────────────────────────────────────

def build_report(recs: list[dict], precheck: dict, handcheck: tuple[int, int] | None) -> dict:
    n = len(recs)
    buckets = Counter(r["bucket"] for r in recs)
    noncompliant = sum(1 for r in recs if not r["compliant"])
    missc = [r for r in recs if r["disagrees"]]
    non_abs = [r for r in recs if not r["is_abs"]]
    refusals = [r for r in non_abs if r["answer_refusal"]]
    rsc = [r for r in refusals if r["verdict"]]
    rsc_recite = [r for r in rsc if r["recites_gold"]]

    c1 = pct(len(missc), n)
    c2 = pct(noncompliant, n)
    c3 = pct(len(rsc), len(non_abs))

    blocked: list[str] = []
    if n == 0:
        blocked.append("G5 NOTHING JUDGED — no judgeable rows in the sample.")
    if not refusals:
        blocked.append(
            "G1 UNMEASURED — zero refusals among non-_abs rows. C3 measured "
            "NOTHING; this is NOT a clean 0%. Either the classifier is broken "
            "or the sample is wrong.")
    if precheck["abs_rows"] == 0:
        blocked.append(
            "G2 UNMEASURED — no _abs rows in the sample, so the "
            "containment-vs-abstention criterion contrast is untested here. "
            "(Expected for MSC, which has no _abs rows by construction.)")
    if non_abs and pct(len(refusals), len(non_abs)) > G3_CEILING:
        blocked.append(
            f"G3 BROKEN — refusal classifier flagged "
            f"{pct(len(refusals), len(non_abs)):.1f}% of rows (> {G3_CEILING}%). "
            "It is matching something generic; do not read C3.")
    if handcheck is None:
        blocked.append(
            "G4 INCOMPLETE — no hand-check counts supplied. The refusal and "
            "gold-recitation checks are lexical. Hand-check the labelled sample "
            "(both refusal- AND committed-classified rows) and re-report with "
            "--handcheck FP,FN.")

    def band(v, mat, watch):
        return "MATERIAL" if v >= mat else ("WATCH" if v >= watch else "IMMATERIAL")

    # The hand-check does not merely unlock the gate — it CORRECTS the number.
    # Lifting G4 while still banding the raw rate would make the control
    # ceremonial: an operator could report a 50%-wrong classifier and the
    # verdict would not move. C3 is scaled by the measured false-positive rate
    # of the refusal classifier, and the banded value is the corrected one.
    c3_adj, fp_rate, fn_rate = c3, None, None
    if handcheck is not None:
        fp, fn = handcheck
        n_flag = min(HANDCHECK_K, len(refusals))
        n_ctrl = min(HANDCHECK_K, len(non_abs) - len(refusals))
        fp_rate = (fp / n_flag) if n_flag else None
        fn_rate = (fn / n_ctrl) if n_ctrl else None
        if fp_rate is not None:
            c3_adj = c3 * (1.0 - fp_rate)
        if fp_rate is not None and fp_rate > HANDCHECK_FP_BROKEN:
            blocked.append(
                f"HAND-CHECK BROKEN — refusal classifier false-positive rate "
                f"{fp_rate*100:.0f}% (> {HANDCHECK_FP_BROKEN*100:.0f}%). C3 is "
                "not measuring refusals; do not read it.")
        if fn_rate is not None and fn_rate > 0.2:
            blocked.append(
                f"HAND-CHECK UNDER-COUNT — classifier false-negative rate "
                f"{fn_rate*100:.0f}%: real refusals are being missed, so C3 is a "
                "floor. Widen the markers and re-run before reading a band.")

    c3_band = band(c3_adj, C3_MATERIAL, C3_WATCH)
    c1_band = band(c1, C1_MATERIAL, C1_WATCH)
    verdict = "INCOMPLETE" if blocked else \
        ("MATERIAL" if "MATERIAL" in (c1_band, c3_band) else
         "WATCH" if "WATCH" in (c1_band, c3_band) else "IMMATERIAL")

    return {
        "n_judged": n,
        "C1_misscore_rate": c1, "C1_band": c1_band,
        "C1_n": len(missc),
        "C1_is_a_lower_bound": True,
        "C1b_negated_yes_invisible_to_both_rules": sum(1 for r in recs if r["negated"]),
        "C2_noncompliance_rate": c2, "C2_n": noncompliant,
        "C3_refusal_scored_correct_rate": c3,
        "C3_rate_handcheck_adjusted": c3_adj,
        "C3_band": c3_band,
        "C3_n": len(rsc), "C3_denominator_non_abs": len(non_abs),
        "C3_of_which_recite_gold": len(rsc_recite),
        "n_refusals_non_abs": len(refusals),
        "buckets": dict(buckets),
        "llm_error_replies": buckets.get("llm_error", 0),
        # n_flag/n_ctrl are exported so the sampler's arm sizes can be pinned
        # against the divisors they are supposed to match. Left as locals, the
        # only way to assert that parity was to restate the definition in the
        # test — which is exactly the duplication that let the two drift apart.
        "handcheck": ({"fp": handcheck[0], "fn": handcheck[1],
                       "fp_rate": fp_rate, "fn_rate": fn_rate,
                       "n_flag": n_flag, "n_ctrl": n_ctrl}
                      if handcheck is not None else None),
        "blocked": blocked,
        "verdict": verdict,
    }


def print_report(rep: dict, precheck: dict) -> None:
    print(f"\n{'='*72}\nJUDGE AUDIT — pre-registered criteria (see module docstring)\n{'='*72}")
    print(f"  rows: {precheck['n']}   judgeable: {precheck['judgeable']}   "
          f"_abs rows: {precheck['abs_rows']}")
    print(f"  judged in this pass: {rep['n_judged']}")
    print(f"\n  C1 misscore (shipping vs reference rule disagree): "
          f"{rep['C1_misscore_rate']:.2f}%  ({rep['C1_n']})   -> {rep['C1_band']}")
    print(f"     LOWER BOUND: + {rep['C1b_negated_yes_invisible_to_both_rules']} "
          f"negated-yes replies that BOTH rules score correct")
    print(f"  C2 non-compliant replies (indicator only):        "
          f"{rep['C2_noncompliance_rate']:.2f}%  ({rep['C2_n']})")
    print(f"  C3 refusal scored CORRECT (non-_abs rows):        "
          f"{rep['C3_refusal_scored_correct_rate']:.2f}%  "
          f"({rep['C3_n']}/{rep['C3_denominator_non_abs']})")
    if rep["handcheck"]:
        print(f"     hand-check adjusted (FP rate "
              f"{(rep['handcheck']['fp_rate'] or 0)*100:.0f}%):              "
              f"{rep['C3_rate_handcheck_adjusted']:.2f}%   -> {rep['C3_band']}")
    else:
        print(f"     -> {rep['C3_band']} (UNADJUSTED — no hand-check)")
    print(f"     of which the refusal recites the gold value:   "
          f"{rep['C3_of_which_recite_gold']}")
    print(f"  refusals found (non-_abs): {rep['n_refusals_non_abs']}   "
          f"judge-side LLM_ERROR replies: {rep['llm_error_replies']}")
    print(f"\n  reply buckets: {rep['buckets']}")
    if rep["blocked"]:
        print("\n  GUARDS TRIPPED:")
        for b in rep["blocked"]:
            print(f"    - {b}")
    print(f"\n  VERDICT: {rep['verdict']}")
    if rep["verdict"] == "MATERIAL":
        print("    => fix the judge and re-baseline LoCoMo, LME and MSC together.")
    print(f"{'='*72}\n")


# ── CLI ─────────────────────────────────────────────────────────────────

def build_judge_client(args):
    """Constructed ONLY here, ONLY under --spend. Never at import."""
    import os
    from longmemeval_adapter import DEEPSEEK_BASE_URL, LLMClient
    extra = json.loads(args.judge_extra_body) if args.judge_extra_body else None
    return LLMClient(args.judge_model,
                     args.api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
                     base_url=args.judge_base_url or DEEPSEEK_BASE_URL,
                     extra_body=extra)


def main(argv=None, client=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", help="stored per-question results JSON")
    ap.add_argument("--pair", nargs=2, metavar=("A.json", "B.json"),
                    help="C4 arm-asymmetry pre-check — FREE, no LLM calls")
    ap.add_argument("--report", help="re-print a saved audit JSON (free)")
    ap.add_argument("--verify-parse", metavar="AUDIT.json",
                    help="FREE. Re-score the stored raw judge replies under the "
                         "frozen legacy rule and the live one, and count the "
                         "verdicts that change. Zero = the parse fix moved no "
                         "canonical number on that corpus.")
    ap.add_argument("--verify-recitation", metavar="AUDIT.json",
                    help="FREE. Re-score the stored (answer, gold) pairs under "
                         "frozen recites_gold_v1 and v2, with the vacuity split "
                         "and the shuffled-gold control (R3). Writes the "
                         "newly/no-longer-flagged rows for R2's hand-check.")
    ap.add_argument("--bench", choices=("locomo", "lme", "msc"), default="locomo")
    ap.add_argument("--limit", type=int, default=None,
                    help="deterministic content-addressed sample size")
    ap.add_argument("--spend", action="store_true",
                    help="REQUIRED to make any LLM call. Without it this is a "
                         "free pre-check that prints the cost it would incur.")
    ap.add_argument("--handcheck", metavar="FP,FN",
                    help="hand-check counts for the refusal/recitation "
                         "classifiers; required to lift the G4 INCOMPLETE gate")
    ap.add_argument("--out", default="judge_audit.json")
    ap.add_argument("--judge-model", default="gpt-oss-120b")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-extra-body", default=None, metavar="JSON")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)

    handcheck = None
    if args.handcheck:
        fp, fn = args.handcheck.split(",")
        handcheck = (int(fp), int(fn))

    if args.verify_recitation:
        saved = json.loads(Path(args.verify_recitation).read_text(encoding="utf-8"))
        res = verify_recitation(saved["records"])
        print(f"\n  RECITATION v1 -> v2 — {args.verify_recitation} (FREE, no LLM)")
        print(f"    stored pairs: {res['n']}")
        print(f"    decided by the verbatim fast path (immune): "
              f"{res['verbatim_decided']}")
        print(f"    token rule actually consulted on: "
              f"{res['token_rule_consulted']}")
        print(f"    NEWLY flagged by v2: {res['newly_flagged']}  "
              f"(R2 hand-check arm)")
        print(f"    NO LONGER flagged by v2: {res['no_longer_flagged']}  "
              f"(v2 is stricter on numerals)")
        print(f"    R3 control — v1 {res['control_rate_v1']:.2f}% / v2 "
              f"{res['control_rate_v2']:.2f}% on {res['control_n']} MISMATCHED "
              f"pairs; v2 true-pair {res['true_pair_rate_v2']:.2f}%")
        print(f"    R4 licence — ceiling v1 {res['ceiling_v1']['num']}/"
              f"{res['ceiling_v1']['den']} = {res['ceiling_v1']['pct']:.2f}%  ->  "
              f"v2 {res['ceiling_v2']['num']}/{res['ceiling_v2']['den']} = "
              f"{res['ceiling_v2']['pct']:.2f}%")
        sample = write_recitation_sample(saved["records"], args.verify_recitation)
        if sample:
            print(f"    R2 hand-check sample -> {sample}")
        print(f"\n    {res['verdict']}\n")
        return 0

    if args.verify_parse:
        saved = json.loads(Path(args.verify_parse).read_text(encoding="utf-8"))
        res = verify_parse(saved["records"])
        print(f"\n  PARSE BEFORE/AFTER — {args.verify_parse} (FREE, no LLM)")
        print(f"    stored replies: {res['n']}")
        print(f"    verdicts that CHANGE: {res['flips']}")
        print(f"    replies that COULD have changed (non-compliant): "
              f"{res['rows_that_could_flip']}")
        print(f"    reply buckets: {res['buckets']}")
        print(f"\n    {res['verdict']}\n")
        return 0

    if args.report:
        saved = json.loads(Path(args.report).read_text(encoding="utf-8"))
        rep = build_report(saved["records"], saved["precheck"], handcheck)
        print_report(rep, saved["precheck"])
        return 0

    if args.pair:
        a, b = (load_rows(p) for p in args.pair)
        res = pair_precheck(a, b, args.bench)
        print(f"\n  C4 arm asymmetry (FREE, no LLM):")
        print(f"    A refusal rate: {res['a']['refusal_rate']:.2f}% "
              f"({res['a']['refusals']}/{res['a']['judgeable']})")
        print(f"    B refusal rate: {res['b']['refusal_rate']:.2f}% "
              f"({res['b']['refusals']}/{res['b']['judgeable']})")
        print(f"    difference: {res['diff_pp']:.2f}pp  "
              f"(threshold {C4_MATERIAL}pp)\n    {res['verdict']}\n")
        return 0

    if not args.run:
        ap.error("one of --run, --pair or --report is required")

    rows = stable_sample(load_rows(args.run), args.limit)
    pre = free_precheck(rows, args.bench)
    to_judge = [r for r in rows if judgeable(judge_inputs(r, args.bench)[3])]

    if not args.spend:
        print(f"\n  FREE PRE-CHECK — {args.run} ({args.bench})")
        print(f"    rows: {pre['n']}   judgeable: {pre['judgeable']}   "
              f"_abs rows: {pre['abs_rows']}")
        print(f"    refusals (lexical): {pre['refusals']} "
              f"({pre['refusal_rate']:.2f}%)   "
              f"canonical-phrase: {pre['refusals_canonical']}")
        print(f"    refusals also reciting gold: {pre['refusals_reciting_gold']}"
              f"   (non-_abs: {pre['refusals_reciting_gold_non_abs']}, "
              f"_abs: {pre['refusals_reciting_gold_abs']})")
        ceil_ = pre["c3_ceiling_non_abs"]
        print(f"    C3 CEILING (non-_abs only): {ceil_:.2f}%  "
              f"({pre['refusals_reciting_gold_non_abs']}/"
              f"{pre['judgeable_non_abs']})   bar = {C3_MATERIAL:.1f}%")
        _code, _msg = c3_spend_licence(pre)
        print(f"    {_msg}")
        hc_path = write_handcheck_sample_pre(rows, args.bench, args.out)
        if hc_path:
            print(f"    hand-check sample (pre-spend, delegated to the paid "
                  f"sampler's selection) → {hc_path}")
        print(f"\n    a --spend pass would cost {len(to_judge)} judge calls "
              f"(1 per judgeable row, max_tokens=10).")
        print("    add --spend to run it.\n")
        return 0

    if client is None:
        client = build_judge_client(args)

    recs: list[dict] = []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            recs = list(pool.map(lambda r: rejudge_row(client, r, args.bench),
                                 to_judge))
    else:
        for i, r in enumerate(to_judge, 1):
            recs.append(rejudge_row(client, r, args.bench))
            if i % 25 == 0:
                print(f"  ── judged {i}/{len(to_judge)}", flush=True)

    rep = build_report(recs, pre, handcheck)
    Path(args.out).write_text(
        json.dumps({"precheck": pre, "report": rep, "records": recs}, indent=2),
        encoding="utf-8")
    write_handcheck_sample(recs, args.out)
    print_report(rep, pre)
    print(f"  full records → {args.out}")
    print(f"  hand-check sample → {Path(args.out).with_suffix('.handcheck.json')}")
    return 0


def write_handcheck_sample_pre(rows: list[dict], bench: str, out: str,
                               k: int = HANDCHECK_K) -> Path | None:
    """PRE-SPEND twin of write_handcheck_sample: same arm selection, no judge.

    Builds rec-shaped records from the run file alone (same classifiers the
    paid path applies), then delegates arm selection to write_handcheck_sample
    OUTRIGHT — one selection implementation, so a future fix to the paid
    sampler cannot leave a stale copy behind in the free path. Rows that are
    not judgeable are skipped, mirroring `to_judge`. Returns None when there is
    nothing to sample (no refusals).
    """
    recs = []
    for r in rows:
        qtype, _q, gold, answer = judge_inputs(r, bench)
        if not judgeable(answer):
            continue
        cls = classify_refusal(answer)
        recs.append({
            "id": r.get("id") or r.get("qa_id") or r.get("question_id"),
            "question_type": qtype,
            "question": r.get("question", ""),
            "is_abs": "_abs" in qtype,
            "answer_refusal": cls["refusal"],
            "answer_canonical": cls["canonical"],
            "answer_loose_only": cls["loose_only"],
            "recites_gold": recites_gold(answer, gold),
            "_answer": answer,
            "_gold": gold,
        })
    if not any(r["answer_refusal"] for r in recs):
        return None
    return write_handcheck_sample(recs, out, k)


def write_handcheck_sample(recs: list[dict], out: str,
                           k: int = HANDCHECK_K) -> Path:
    """G4's correct-answer control. Writes BOTH refusal-classified and
    committed-classified rows, labelled, so the hand-check can measure false
    POSITIVES and false NEGATIVES. A sample of only flagged rows would be a
    confirmation pass and could not produce an FN count.

    BOTH arms draw from non-`_abs` rows only, because that is the population
    `build_report` divides by (`n_flag`/`n_ctrl` at :617-618 are sized off
    `non_abs`). Drawing from all rows while dividing by the non-`_abs` count
    spends divisor slots on rows the criterion never counts, and understates
    the FP rate by exactly that fraction — measured at 6 of 25 slots on the
    2026-08-25 LME run, where the reader refused 28 of 30 abstention questions.
    The correction `c3_adj = c3 * (1 - fp_rate)` is applied to a non-`_abs`
    quantity, so its FP rate has to be measured on non-`_abs` rows.

    Consequence to keep in view: the refusal classifier is then never
    hand-validated on `_abs` rows. Nothing bands on that today — C3 is the only
    consumer and it excludes them — but a future criterion that reads abstention
    behaviour would need its own sample rather than this one.

    The `[:k]` slice is take-first, not `stable_sample`. Deliberate: the arm
    sizes must equal `n_flag`/`n_ctrl` exactly, and re-sampling a file an
    operator has already hand-scored would silently invalidate their work.
    """
    non_abs = [r for r in recs if not r["is_abs"]]
    flagged = [r for r in non_abs if r["answer_refusal"]][:k]
    control = [r for r in non_abs if not r["answer_refusal"]][:k]
    dest = Path(out).with_suffix(".handcheck.json")
    dest.write_text(json.dumps(
        {"instructions":
            "For each row decide if `_answer` is genuinely a refusal. "
            "FP = classified refusal but is a real answer. "
            "FN = classified committed but is a refusal. "
            "Re-report with --handcheck FP,FN.",
         "classified_refusal": flagged,
         "classified_committed_CONTROL": control},
        indent=2), encoding="utf-8")
    return dest


if __name__ == "__main__":
    raise SystemExit(main())
