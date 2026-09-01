# Pre-registration: does the ALIAS judge churn? (B2) — 2026-09-01 (v0.1)

Status: **DRAFT — awaiting Atta's approval for the spend.** Branch:
Beam-optimisation. One run, ~160 judge calls, ~5 minutes, ~5% of the model-pin
Step 2. Written and committed BEFORE the run, per the §4.1 discipline the
`--prereg` gate now enforces.

Arises from Step 1 of `2026-08-31→2026-09-01` model-pin pre-registration §8.1,
which measured something it was not designed to ask.

## 1. Question (what this decides)

Step 1 passed: the pinned judge and the alias judge agree on scores. It also
found that **the pinned judge is not score-deterministic** — 7/160 rows move
between two byte-identical invocations at temperature 0.0.

The gold-delta phase recorded the ALIAS as deterministic (32/32 control rows
identical, `SD_ctl = 0.0000`) and built a **zero-width band** on that zero.
Its −0.582pp primary read OUTSIDE that band, which is what produced
**REBASE REQUIRED**.

**Step 1's PASS is compatible with two opposite worlds, and cannot tell them
apart. That is the design gap this run closes.**

- **World 1 — the equivalence claim holds.** The alias churns at ~4.4% too;
  §8's 0/32 was a small-sample draw; the gold-delta verdict is band-dependent
  and does not survive a realistic variance estimate.
- **World 2 — the claim fails on the sampling axis.** The alias really is
  deterministic and the pin is not. Scores agree (hence the PASS) but the
  `deepseek-v4-flash`+thinking-disabled path is a **noisier instrument** than
  the alias. Then the pin is not a free upgrade: it buys witnessability at the
  cost of reproducibility, and the migration doc's "byte-path-equivalent"
  headline is false in a way score-agreement conceals.

Both worlds produce the same PASS. Only a direct measurement of the alias's
churn separates them, and one of them says **the pin is the wrong instrument**
— which is worth knowing before authoring anything that presumes it is right.

- **Falsifier (H0):** the alias reproduces itself exactly, 0/160. → World 2.
- **Reject H0:** any row moves. → World 1.

## 2. Design

**B2** = a second run of exactly B's invocation: `--rejudge` of
`results_20260831T165039Z.json`, `--judge-gold`, `--judge-model deepseek-chat`,
no `--judge-extra-body`. One variable between B and B2: that B2 is a second
run. Answers are byte-fixed from A, gold comes from the same reparse under the
same 160/160 guard, prompts are `_judge_messages` on both sides.

Two differences that are recorded and do not touch the request body:
`--prereg` (B predates the gate, so B carries `prereg: null`) and
`--dataset-revision 3205395e...` — which pins the same revision B read, the
dataset being unchanged on the Hub since 2026-01-30. Neither alters a byte the
judge sees.

## 3. Verified facts (2026-09-01)

- Step 1, from the scorer committed at 32e81ee before any comparison was run:
  D_self(pin) = 7/160, control arm 1/32 (SD 0.044194), pool 6/128 (SD
  0.088287). D_pin = 3/160, δ̄_pin = −0.3646pp. Gate clean on both arms.
- **The gold-delta primary recomputed under §8's OWN formula**
  (`SE = SD_ctl/√128`, `band = 2·SE`, SD_ctl from the 32 control-arm deltas),
  substituting the pin's control-arm SD for the alias's recorded zero:
  band = **±0.7812pp**, and |−0.582pp| < 0.7812 → **INSIDE**. It is inside on
  every candidate substitution (pin all-row ±1.4382pp, pin pool ±1.5607pp).
  The primary does not survive any realistic variance estimate.
- **The verdict turns on one row.** The SD_ctl at which −0.582pp is exactly
  borderline is 0.032923 — which, for a sample of 32 where a single row moves
  and the rest do not, is **one control row moving by 0.1862**. The pin moved
  one control row (CR) by 0.25. A single quarter-point move in 32 control rows
  is the whole distance between REBASE REQUIRED and H0 holding.
- **Power.** At the pin's measured rate, P(0 movements in 32) = 0.239 and
  P(0 in 36) = 0.200 — §8's evidence could not have distinguished determinism
  from 4.4% churn; a zero draw was a one-in-four event. P(0 in 160) = 0.00078.
  **B2 at n=160 can make that distinction where n=32 could not**, which is the
  entire reason this run is worth five minutes.
- **Evidence pointing the other way, recorded so it is not quietly dropped.**
  A→B was a 3.17h gap on the alias with 0/32 control movement; C1→C2 was a
  5-minute gap on the pin with 1/32. If both names reached the same servable
  configuration, a longer gap should expose at least as much drift, not less.
  This is weak — 0/32 vs 1/32 separates nothing on its own — but it points at
  World 2, and it is why this pre-registration does not presume World 1.

## 4. Procedure

1. Commit this pre-registration; the run passes `--prereg` naming it, so its
   blob and the code commit land in B2's metadata.
2. Run B2 once.
3. **Gate before any comparison:** B2 must show 0/160 silent-0 (non-empty
   rubric with `scores == []`) and 0/160 explicit `[LLM_ERROR`, matching B. A
   nonzero rate means the run is broken and no churn reading is interpretable.
4. Compute §5 and record in §7.

## 5. Read protocol (fixed before counts)

- **D_alias = #rows where B2 ≠ B**, over 160. **SD_alias** = SD of the
  (B2 − B) deltas; **SD_alias_ctl** = the same over the 32 ABS/CR control rows,
  because that is the quantity §8's formula actually consumes.
- **World 2 ⇔ D_alias = 0.** The alias is deterministic at n=160 where the pin
  moves 7/160. Consequences, fixed now: the migration doc's byte-path-
  equivalence claim is **false on the sampling axis** and must be corrected;
  §8's zero-width band stands as a real property of the alias; and **pin
  adoption becomes a deliberate trade rather than a free upgrade** — it must be
  argued on witnessability against a measured loss of reproducibility, not
  assumed from Step 1's PASS.
- **World 1 ⇔ D_alias ≥ 1.** §8's `SD_ctl = 0` was a small-sample draw.
  Then recompute the gold-delta primary with `SD_alias_ctl` in §8's own formula
  and report inside/outside. **If it is inside, REBASE REQUIRED is not
  supported at the measured variance** and must be re-derived under a fresh
  pre-registration before Step 2 can be justified on those grounds.
- **Descriptive only, never used to reclassify:** D_alias against the pin's
  7/160. If D_alias > 7 the alias is the noisier instrument and the pin's
  reproducibility cost is negative — which would strengthen pin adoption. This
  is reported, not decided here.
- No top-up. If B2 is ambiguous it is ambiguous; a third run is a new
  pre-registration, not a refinement of this one.

## 6. What this run does NOT decide

- It does not re-run or revise the gold-delta verdict. §8 measured what it
  measured and recorded its zero-width band as load-bearing in its own text.
  B2 establishes the fact that determines whether that verdict is
  band-dependent; re-deriving it is a separate, later pre-registration.
- It says nothing about the ANSWERER. Step 1's limit is unchanged: answers are
  not replayed here.
- It does not authorise Step 2.
- **A second fragility in §8, noted for that later re-derivation and not acted
  on here:** the verdict's other support, per-ability heterogeneity ("EO
  −7.57pp, SUM −4.37pp vs IF/KU +6.25pp"), sits at the same resolution limit.
  At n=16 one flip is 6.25pp, so EO/IF/KU are all ~one-flip magnitudes and
  SUM's −4.37pp is **below** one flip. With ~4.4% churn, several of the 24
  moved pool rows are expected to be churn. The heterogeneity is consistent
  with churn, so both of §8's supports are churn-ambiguous, not just the
  zero-width primary.

## 7. Executed results — B2 (2026-09-01): GATE FAILED, verdict VOID

Run under `--prereg` at a330fec8. Canary OK (217 chars). All 160 rows judged,
161 judge calls, 219s. Then:

    VOID: 1 silent-0 parse failures (A had 0/160; B rate must be <= A)
      IF | What are some common responses when something goes wrong wit
         | raw='{"scores": [1], "total_score": 1.0, "explanation": "The
            response includes numeric error status codes'

**Per §4.3 of this pre-registration, the churn reading is void. D_alias is NOT
computed and World 1 vs World 2 is NOT decided.** The gate said a nonzero
silent-0 rate means no churn reading is interpretable, and that rule was fixed
before the run. It is honoured here.

### 7.1 What the failure was — and it is not what the gate assumed

The judge did not refuse, error, or return empty. **It scored the row 1.0**,
wrote a long explanation, and hit `max_tokens=512` mid-sentence.
`judge_answer`'s `re.search(r'{...}')` needs a complete JSON object, found
none, took the except path, and returned `{"score": 0.0, "scores": []}`.

**A row the judge scored 1.0 was recorded as 0.0 because its explanation ran
long.** This is a latent defect on every run ever scored by this path, not a
property of B2 — and it is exactly the failure class this campaign exists to
catch: a number indistinguishable in the score column from a real 0.0.

It is deliberately not fixed yet. Changing the regex or the token ceiling
changes what a score means, retroactively, which is a pre-registered decision
and not a bug fix to slip in beside another change.

### 7.2 An observation, explicitly NOT the pre-registered statistic

B and B2 send byte-identical judge prompts. That row parsed in B (B was
0/160) and did not parse in B2. So the alias returned materially different
text for the same input on two runs.

That is a direct observation of alias non-determinism, and by the §5 letter
(`World 1 ⇔ D_alias ≥ 1`) it would decide the question — the row's score
differs between the two runs. **It is not being read that way**, because §4.3
voided the reading first and the two rules were fixed together. Reading past a
gate because the result on the other side is interesting is precisely the
post-hoc move this document exists to prevent. Recorded as an observation,
carrying no verdict.

It also implies B's own 0/160 silent-0 was a draw, not a property: at the one
observed occurrence per 160 rows, P(0 in 160) ≈ 0.366.

## 8. What this pre-registration got wrong

The §4.3 gate was ported from the gold-delta phase, where a silent-0 meant
**the plumbing was broken** — a mis-pinned model returning unparseable output.
Here the plumbing is fine and the silent-0 is **the phenomenon under study
manifesting**: the judge's own run-to-run variation, surfacing as a parse
outcome instead of a score change.

The gate therefore voids on exactly the evidence B2 was commissioned to
collect. That is a defect in this spec, not in the run, and naming it is
cheaper than routing around it.

**A superseding B2 v0.2 needs a gate that separates the two**, e.g.: a silent-0
whose `judge_raw` is empty or `[LLM_ERROR`-prefixed is plumbing and still
voids; a silent-0 whose raw contains a truncated but well-formed scoring
attempt is judge variation, is counted into D_alias as a changed row, and does
not void. That distinction has to be written down before the re-run, not
after, and the re-run is now non-destructive: since 0adcd2c a voided run
persists its rows marked `void`, so the evidence survives even if the gate
fires again.

**Not authorised under this spec.** B2 v0.2 is a new pre-registration and a new
run.
