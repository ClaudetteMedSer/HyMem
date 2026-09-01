# Pre-registration: does the ALIAS judge churn? — B2 v0.3 — 2026-09-01

Status: **APPROVED for the run** (Atta, 2026-09-01: "let's run the parse
first"). Branch: Beam-optimisation. One run, ~160 judge calls, ~4 minutes.

Supersedes `2026-09-01-alias-churn-b2-v02-pre-reg.md`. v0.2 is not merely
superseded on style: **its §3 states in terms that the parse defect is NOT
fixed and that "this spec's gate over-voids"**. `cb1fd34` fixed it, so running
under v0.2 would mean running under a spec the code contradicts.

## 1. Question — unchanged from v0.1 and v0.2

Does the alias judge (`deepseek-chat`, gold on) reproduce itself across two
identical runs?

Step 1 measured the PIN at D_self = 7/160 and passed. That PASS is blind
between **World 1** (the alias churns too, so the gold-delta phase's
`SD_ctl = 0` was a small-sample draw and the zero-width band that produced
REBASE REQUIRED was an artifact) and **World 2** (the alias is deterministic
and the pin is the noisier instrument, so the migration doc's
byte-path-equivalence claim fails on the sampling axis).

- **Falsifier (H0):** D_alias = 0 → World 2.
- **Reject H0:** D_alias ≥ 1 → World 1.

## 2. What changed since v0.2, and why a third attempt should now land

Two runs voided on the same row. Neither was judge instability:

- **v0.1** voided and, under the code then live, **destroyed its 160 rows**.
  Fixed by 0adcd2c — a voided run now persists its rows.
- **v0.2** voided again. Because the artifact survived, the cause was readable
  from disk: a **complete, valid, `finish_reason: "stop"`** reply of 230 chars
  in which the judge wrote `scores: [1]`, defeated by
  `re.search(r'\{[^}]+\}')` stopping at the first `}` — one inside a
  `${response.status}` template literal the judge had quoted out of the answer.
  **Not truncation**; my v0.1 diagnosis of truncation was wrong and is
  corrected at source in both prior specs.
- Fixed by **cb1fd34**: `extract_judge_json` brace-matches with string-literal
  and escape tracking, tries the old regex first so it is strictly more
  permissive by construction, and records `judge_parse` ∈
  {ok, recovered, unreadable} per row.

**This run is the first B2 attempt whose reader can read the judge.**

## 3. Verified facts (2026-09-01, before the run)

- The parse audit (`benchmarks/judge_parse_audit.py`, read-only, zero API
  calls): **B 0/160, C1 0/160, C2 0/160 naive-parse failures**; B2b 1/160,
  recoverable, judge said `[1]` and the adapter recorded `0.0`.
- **Therefore the comparison is not confounded by the parser change.** This
  matters and would otherwise be a real threat to validity: B2c is scored by
  the NEW parser and B by the OLD one, so any row the old parser mis-read in B
  would show up as a difference caused by the fix rather than by the judge.
  **B has zero such rows**, measured rather than assumed, so on B's side the
  two parsers are provably identical and D_alias measures the judge alone.
- The pin's churn D_self = 7/160 was likewise audited clean (C1 and C2 both
  0/160), so the number this run is compared against carries no parse artifact.
- `P(0 movements in 160 | pin's rate 7/160) = 0.00078`. n=160 discriminates
  where the gold-delta phase's n=32 could not (P(0 in 32) = 0.239).

## 4. Gate (unchanged from v0.2 except that the fourth class is gone)

| class | signature | void? | counts? |
|---|---|---|---|
| **plumbing** | raw empty, or `[LLM_ERROR`-prefixed | **YES** | no |
| **truncation** | `finish_reason == "length"` AND non-empty AND parse-fail | no | **no — excluded, rate reported** |
| **readable** | a parsed `scores` list, `judge_parse` ∈ {ok, recovered} | no | yes |

v0.2's fourth class (*parse defect*) is **retired by construction**: a
brace-bearing reply now parses and is simply readable. A row that still fails
to parse after brace-matching is genuinely unreadable, so classing it as
plumbing is now correct rather than merely conservative.

**Ceiling** unchanged: truncation above **8/160 (5%)** makes the run INVALID
for the falsifier — report the rate, do not interpret.

## 5. Design

**B2c** = one run of B's exact invocation:
`--rejudge results_20260831T165039Z.json --judge-gold
--judge-model deepseek-chat` (no `--judge-extra-body`),
`--prereg docs/plans/2026-09-01-alias-churn-b2-v03-pre-reg.md`,
`--dataset-revision 3205395e897e7318c7b094ef4e6047b9b82dbb03`.
One variable against B: that it is a second run — plus the reader fix, whose
footprint on B's side is measured at zero (§3).

**Comparison set = READABLE rows**, n = 160 − (plumbing + truncation). Fixed
here so the denominator cannot be chosen after the counts.

## 6. Read protocol (fixed before counts)

- **D_alias** = readable rows where B2c ≠ B. **SD_alias_ctl** = SD of the
  (B2c − B) deltas over the readable ABS/CR control rows, that being the
  quantity the gold-delta primary's formula consumes.
- **World 2 ⇔ D_alias = 0** with truncation under its ceiling. Consequences:
  the migration doc's equivalence claim is false on the sampling axis; the
  gold-delta zero-width band stands as a real property of the alias; and pin
  adoption becomes a **deliberate trade** — witnessability against a measured
  loss of reproducibility — not inferable from Step 1's PASS.
- **World 1 ⇔ D_alias ≥ 1.** Recompute the gold-delta primary in **its own**
  formula (`SE = SD_alias_ctl/√128`, `band = 2·SE`) and report inside/outside
  for the −0.582pp effect. **If inside, REBASE REQUIRED is not supported at the
  measured variance** and must be re-derived before Step 2 is justified on
  those grounds.
- **Descriptive only, never reclassifying:** D_alias against the pin's 7/160;
  the truncation rate; the `judge_parse` distribution, in particular any
  `recovered` rows — those are rows this run reads and every prior run would
  have scored 0.0.

## 7. What a `recovered` row means for the comparison, decided in advance

If B2c contains a `recovered` row, the new parser read a verdict the old one
would have dropped. B's corresponding row was parsed cleanly by the old parser
(§3: B is 0/160), so the comparison stays sound — but the row is **counted in
D_alias like any other readable row**, and flagged in the readout.

It must **not** be excluded. Excluding rows because their handling improved
would bias the churn estimate toward zero, by dropping exactly the rows where
the judge's real verdict was hardest to obtain.

## 8. Executed results

(empty — nothing has been run under this pre-registration)
