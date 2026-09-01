# Pre-registration: does the ALIAS judge churn? — B2 v0.2 — 2026-09-01

Status: **DRAFT — awaiting Atta's approval for the spend.** Branch:
Beam-optimisation. One run, ~160 judge calls, ~4 minutes.

Supersedes `2026-09-01-alias-churn-b2-pre-reg.md`, whose run judged all 160
rows and then voided on a gate that could not tell the phenomenon under study
from a broken run — and, under the code live at the time, discarded the rows on
the way out. Both defects are fixed; this spec exists so the re-run is
authorised by a rule written **before** it, not by the one it broke.

## 1. Question — unchanged

Does the alias judge (`deepseek-chat`, gold on) reproduce itself across two
identical runs?

Step 1 measured the PIN at D_self = 7/160 and passed. That PASS is blind
between **World 1** (the alias churns too, so the gold-delta phase's
`SD_ctl = 0` was a small-sample draw and its zero-width band — the band that
produced REBASE REQUIRED — was an artifact) and **World 2** (the alias really
is deterministic and the pin is the noisier instrument, so the migration doc's
byte-path-equivalence claim fails on the sampling axis). One of those says the
pin is the wrong instrument.

- **Falsifier (H0):** the alias reproduces itself, D_alias = 0 → World 2.
- **Reject H0:** D_alias ≥ 1 → World 1.

## 2. What v0.1 got wrong (both defects, and the fixes)

**(a) The gate voided on the evidence it was commissioned to collect.** §4.3
was ported from the gold-delta phase, where a silent-0 meant broken PLUMBING.
In v0.1's run the plumbing was fine.

**CORRECTED after the v0.2 run: the cause was NOT truncation.** v0.1's abort
printer emitted `raw[:100]`, so the log line was cut off by the printer and I
read a display truncation as a model truncation. The v0.2 run hit the same row
with `finish_reason` recorded and the artifact preserved: the reply was 230
characters, `finish_reason: "stop"`, complete and valid JSON. The regex
`re.search(r'\{[^}]+\}')` stops at the first `}`, which was inside a
`${response.status}` template literal the judge had quoted from the answer, so
`json.loads` raised *Unterminated string* and the except path returned the
sentinel. A row scored 1.0 by the judge was recorded 0.0 because the answer
contained a brace — a hazard the gold-delta pre-registration §3 had already
documented ("cannot match nested JSON") and which I mis-attributed.

**(b) The refusal destroyed the evidence.** The silent-0 branch called
`sys.exit(3)` before the artifact write, so 160 judged rows — already paid for
— were discarded, and no B2 artifact exists. Fixed in 0adcd2c: a voided run now
persists its rows, marked in `metadata.void` and in the filename, exit code
unchanged, readout suppressed.

This is a known class in this codebase, not a novelty.
`longmemeval_adapter.py:1274-1279` records the sibling: LME's bare-bool judge
discarded its raw, which "is why `benchmarks/judge_audit.py` had to re-judge
500 rows to measure a rate that was already produced once and discarded."
Beam repeated it, and v0.1's run was the cost.

## 3. The gate (the whole point of v0.2)

**A first draft of this rule counted a truncated row into D_alias as a changed
row. That draft was wrong** and would have re-imported the defect this campaign
exists to eliminate: the truncated row's 0.0 is *fabricated* — neither the
judge's actual score nor a real 0.0 — so counting it would put a parse artifact
into the statistic as a measurement.

**Three classes, decided before the run:**

| class | signature | void? | counts in D_alias? |
|---|---|---|---|
| **plumbing** | `judge_raw` empty, or `[LLM_ERROR`-prefixed | **YES** | no |
| **truncation** | `finish_reason == "length"` **AND** non-empty content **AND** parse-fail | no | **no — excluded, rate reported** |
| **readable** | a parsed `scores` list | no | yes |

Truncation means the plumbing is fine (so it must not void) *and* the row is
unreadable (so it must not count). That is exactly the existing `explicit_err`
treatment: keep prior, exclude from the statistic, report the rate.

**The separator is the conjunction, not `finish_reason` alone.** Both failure
shapes carry `length`:

- the v4-flash trap is **empty** content + `length` — already caught by the
  falsy-content raise and rerouted to `[LLM_ERROR`, so it lands in *plumbing*;
- v0.1's failure is **non-empty** content + `length` + parse-fail — nothing
  catches it, and it lands in *truncation*.

Gating on `length` alone would merge them. Since 1d80f33 `finish_reason` is
recorded per row, so this discriminator is structural and fixed in advance
rather than a judgement call made with the answer visible.

**A FOURTH class, found by the v0.2 run itself and not anticipated here:**

| class | signature | void? | counts? |
|---|---|---|---|
| **parse defect** | `finish_reason == "stop"`, complete reply, naive regex fails, a brace-matching parse RECOVERS a scores list | should not | not until fixed |

This is neither plumbing nor truncation: the judge answered, completely and
correctly, and the reader dropped it. The v0.2 gate as implemented classes it
under *plumbing* and voids — which is wrong in its reason though conservative
in its effect. **It is not patched here.** The right answer is to fix the
parser rather than to add a class excusing its failures, and that is its own
pre-registration because it changes what a score means. Until then this spec's
gate over-voids, and that is the safe direction.

**Ceiling:** if truncation exceeds **8/160 (5%)** the run is INVALID for the
falsifier — too much of the sample is unreadable to say anything about the
rest. Report the rate, do not interpret. (Mirrors the gold-delta explicit-error
ceiling.)

## 4. Design

**B2b** = one run of B's exact invocation: `--rejudge` of
`results_20260831T165039Z.json`, `--judge-gold`,
`--judge-model deepseek-chat`, no `--judge-extra-body`,
`--prereg docs/plans/2026-09-01-alias-churn-b2-v02-pre-reg.md`,
`--dataset-revision 3205395e897e7318c7b094ef4e6047b9b82dbb03`.
One variable against B: that it is a second run.

**Comparison set = the READABLE rows only**, n = 160 − (plumbing + truncation).
Fixed here so the denominator cannot be chosen later.

## 5. Read protocol (fixed before counts)

- **D_alias** = readable rows where B2b ≠ B. **SD_alias_ctl** = SD of the
  (B2b − B) deltas over the readable ABS/CR control rows — that is the quantity
  the gold-delta primary's formula consumes.
- **World 2 ⇔ D_alias = 0** with the truncation rate under its ceiling. The
  alias is deterministic where the pin moves 7/160; P(0 in 160) = 0.00078 at
  the pin's rate, so this is not a sampling accident. Consequences: the
  migration doc's equivalence claim is false on the sampling axis and must be
  corrected; the gold-delta zero-width band stands as a real property of the
  alias; and **pin adoption becomes a deliberate trade** — witnessability
  against a measured loss of reproducibility — which cannot be inferred from
  Step 1's PASS.
- **World 1 ⇔ D_alias ≥ 1.** The gold-delta `SD_ctl = 0` was a draw. Recompute
  its primary in **its own formula** (`SE = SD_alias_ctl/√128`, `band = 2·SE`)
  and report inside/outside for the −0.582pp effect. **If inside, REBASE
  REQUIRED is not supported at the measured variance** and must be re-derived
  under a fresh pre-registration before Step 2 is justified on those grounds.
- **Descriptive only:** D_alias against the pin's 7/160; the truncation rate;
  and whether truncation concentrates in any ability. None of these reclassify.
- No top-up. A third run is a new pre-registration.

## 6. Known open item, named rather than silently carried

Until a parse fix is pre-registered and landed, **every run carries the
one-row-truncation hazard**, and a truncated row's fabricated 0.0 is
indistinguishable from a real 0.0 in the score column alone. Step 1 was checked
and is clean — B/C1/C2 each 0/160 unparseable, longest reply 879 chars — so
D_self = 7/160 is uncontaminated. That was luck, not protection.

The fix is **not** in scope here: changing the regex or the token ceiling
changes what a score means retroactively, across every historical artifact.
It needs its own pre-registration.

## 7. Executed results — B2b (2026-09-01): VOID again, and the cause is now known

Artifact **preserved this time**:
`results_20260831T165039Z-rejudged-deepseek-chat-20260901T104602Z-VOID.json`.
161 judge calls, 224s, canary OK. Counts: **1 silent-0 / 0 truncated / 0
explicit / 159 rejudged of 160.**

The truncation gate did not fire, correctly: `finish_reason` was `"stop"`. The
row fell through to *plumbing* and voided. That classification is wrong in its
reason — see §2(a) as corrected — but conservative in its effect.

**The fix from 0adcd2c paid for itself on its first exercise.** v0.1's identical
failure destroyed its 160 rows and left only a printer-truncated log line, from
which I drew the wrong conclusion. This run kept the rows, so the true cause was
readable from disk in one query instead of inferred from a fragment.

### 7.1 Free audit of every artifact that stores `judge_raw`

Because the raws are stored, the historical reach of this defect costs **zero
API calls** to measure. Comparing the naive regex against a brace-matching
parser that respects string literals:

| artifact | rows w/ raw | naive fails | recoverable |
|---|---|---|---|
| B (alias, 20260831T200531Z) | 160 | 0 | 0 |
| C1 (pin) | 160 | 0 | 0 |
| C2 (pin) | 160 | 0 | 0 |
| **B2b (alias, VOID)** | 160 | **1** | **1** — judge said `[1]`, recorded `0.0` |

**1 in 640 judged rows, and fully recoverable from data already on disk.**

**The more serious half: `results_20260831T165039Z.json` (A, the anchor) and the
two earlier runs store NO `judge_raw` at all.** Their exposure to this defect is
not merely unknown, it is **unmeasurable** — and A is the June-comparable anchor
the whole campaign is built on. The rejudge path stores raws; the main run path
does not.

### 7.2 What this does to the B2 question

**Still unanswered.** D_alias is uncomputed after two runs. But the reason is
now understood rather than mysterious, and it is not judge instability: the same
row failed in both attempts because the judge quoted a brace-bearing snippet
both times, while B happened not to. The parse defect is deterministic given the
reply; the reply text is what varies.

A third run would probably succeed, and would be luck. **The parse fix is now a
precondition for B2 being answerable rather than an improvement to it**, and it
is pre-registered separately.

### 7.3 Why this matters more than B2 does

The defect is **content-dependent, not random**: it fires when a judge
explanation contains `}`. That correlates with answers containing code, JSON or
template literals — which concentrates in IF and PF, the two
`compliance_spec` abilities about API behaviour. So the bias is
**ability-correlated**, not uniform noise, and it silently deflates exactly the
abilities most likely to quote code.

Every score this path has ever produced carries it, and for the runs without
stored raws it cannot be audited even in principle.
