# Pre-registration: the judge parser drops answers containing braces — 2026-09-01

Status: **DRAFT.** Branch: Beam-optimisation. **Zero API calls.** Every input
this needs is already on disk.

## 1. The defect

`judge_answer` extracts the judge's verdict with

    re.search(r'\{[^}]+\}', raw.replace('\n', ' '))

`[^}]+` stops at the **first** `}`. When a judge explanation quotes text
containing a brace, the match ends mid-string, `json.loads` raises
*Unterminated string*, the except path returns `{"score": 0.0, "scores": []}`,
and that fabricated 0.0 is **indistinguishable in the score column from a real
0.0**.

Observed (B2b, `...20260901T104602Z-VOID.json`), `finish_reason: "stop"`,
complete valid JSON, 230 chars:

    {"scores": [1], "total_score": 1.0, "explanation": "The response includes
    numeric status codes in the example 'Error ${response.status}:
    ${response.statusText}' and mentions status codes in the summary,
    satisfying the criterion."}

**The judge scored the row 1.0. The adapter recorded 0.0, because the answer
being graded contained `${response.status}`.**

The gold-delta pre-registration §3 already recorded that this regex "cannot
match nested JSON". The hazard was documented; only its consequence was not
measured.

## 2. Why this is a correction, not a change of meaning

The distinction matters because this campaign refuses retroactive score
changes. This one is different in kind:

- The judge's verdict is **not ambiguous and not missing**. It is present, in
  full, in `judge_raw`, and it says `scores: [1]`.
- The 0.0 is not a judgement the judge made. It is a **sentinel emitted by a
  reader that failed**, in a column that cannot express "I could not read this".
- So recovering `[1]` does not re-score the row. It records what the judge
  said, for the first time.

The honest framing: the score column has been carrying **two different
quantities** — the judge's verdict, and the parser's failure — with no way to
tell them apart. This separates them.

## 3. What is NOT in scope

- **No artifact is rewritten.** Historical files keep their recorded scores.
  The audit reports what WOULD change; it changes nothing.
- **No re-judging.** No API calls at all.
- **No change to prompts, models, `max_tokens`, or the rubric.** The request
  bytes are untouched, so runs stay comparable on every axis but this one.
- **No change to the no-rubric path**, which legitimately returns 0.0/[].

## 4. The fix

Replace the regex with a brace-matching scan that respects string literals and
escapes, returning the first complete top-level JSON object. On failure the
sentinel is unchanged, so a genuinely unreadable reply still reads as
unreadable.

Add `judge_parse` to each row, recorded alongside `judge_raw`:

| value | meaning |
|---|---|
| `"ok"` | parsed |
| `"recovered"` | the naive regex would have failed; brace-matching succeeded |
| `"unreadable"` | neither parses — the sentinel, honestly earned |

`"recovered"` exists so the fix's own footprint is visible per row rather than
inferred from a commit date.

## 5. Pre-registered audit (fixed before it is run)

`benchmarks/judge_parse_audit.py` reports, per artifact storing `judge_raw`:
rows, naive failures, recoverable count, and for each the judge's real scores
against the recorded score. **Read-only.**

Known before writing this, from the ad-hoc scan that found the defect: B 0/160,
C1 0/160, C2 0/160, B2b 1/160 recoverable. The audit must reproduce those, and
is a regression test on the parser as much as a measurement.

**Falsifier:** if the new parser changes any row where the naive regex already
succeeded, the fix is wrong — it must be strictly more permissive, never
different. Pinned by test, not by inspection.

## 6. The finding this cannot fix, and it is the larger one

**`results_20260831T165039Z.json` (A, the June-comparable anchor) and the two
earlier runs store no `judge_raw` at all.** The rejudge path stores raws; the
main run path does not. So for A — the record the entire campaign is anchored
to — this defect's reach is **unmeasurable in principle**, not merely unknown.

The defect is also **content-dependent, not random**: it fires when an
explanation contains `}`, which correlates with code, JSON and template
literals, concentrating in IF and PF, the two `compliance_spec` abilities about
API behaviour. So the expected bias is **ability-correlated silent deflation**,
not uniform noise — and it lands on the abilities whose answers most often
contain braces.

**Consequence, stated plainly:** a fresh gold-on canonical (model-pin Step 2)
would be the first BEAM run whose judge verdicts are both correctly parsed and
fully auditable. That is an argument FOR the rebase which is independent of the
gold-delta verdict — and it survives whichever way the B2 churn question lands.

## 7. Executed results

(empty)
