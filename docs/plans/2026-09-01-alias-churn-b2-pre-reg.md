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

### 7.1 What the failure was — CORRECTED 2026-09-01 after B2 v0.2

**This section originally said the judge "hit `max_tokens=512` mid-sentence."
That was wrong, and the error was mine.** The abort printer emits `raw[:100]`,
so the log line was cut off by the PRINTER, and I read a display truncation as
a model truncation. I then built a truncation gate on that reading.

B2 v0.2 hit the same row and, because `finish_reason` was by then recorded and
the artifact preserved, the actual reply is on disk: **230 characters,
`finish_reason: "stop"` — complete and valid JSON.**

    {"scores": [1], "total_score": 1.0, "explanation": "The response includes
    numeric status codes in the example 'Error ${response.status}:
    ${response.statusText}' and mentions status codes in the summary,
    satisfying the criterion."}

`judge_answer`'s `re.search(r'\{[^}]+\}')` stops at the FIRST `}` — the one
inside `${response.status}`. It matched a fragment ending mid-string,
`json.loads` raised *Unterminated string*, the except path returned
`{"score": 0.0, "scores": []}`.

**A row the judge scored 1.0 was recorded as 0.0 because the answer being
graded contained a brace.** Not truncation: a naive regex meeting a `}` inside
a string literal.

Worse for my credit: **this was already documented.** The gold-delta
pre-registration §3 records that this regex "cannot match nested JSON". The
hazard was known and written down in this campaign's own spec, and I attributed
its symptom to something else.

**A row the judge scored 1.0 was recorded as 0.0 because its explanation ran
long.** This is a latent defect on every run ever scored by this path, not a
property of B2 — and it is exactly the failure class this campaign exists to
catch: a number indistinguishable in the score column from a real 0.0.

It is deliberately not fixed yet. Changing the regex or the token ceiling
changes what a score means, retroactively, which is a pre-registered decision
and not a bug fix to slip in beside another change.

### 7.2 B2's rows were DESTROYED, and the fix does not cover the run it is named for

B2 ran under `bb16b60`, where the silent-0 branch called `sys.exit(3)`
**before** the write block. `0adcd2c` — which makes a voided run persist its
rows — landed 3h23m later. **No B2 artifact exists.** There is no `-VOID` file
anywhere on the box and no third `deepseek-chat` rejudge in `hymem_beam/`; the
newest artifacts are C1 and C2.

So the raw quoted in §7.1 is a **stdout fragment** (`raw[:100]` from the abort
printer), not a persisted row. The 160 judged rows are gone and cannot be
re-read at any price short of buying them again. The fix is correct and
forward-looking; it did not protect the run that exposed the need for it.

### 7.3 An observation, and why it does NOT decide the question

B and B2 send byte-identical judge prompts. That row parsed in B (0/160) and
did not in B2, so the alias returned materially different **text** for the same
input across two runs.

An earlier draft of this section claimed that by §5's letter
(`World 1 ⇔ D_alias ≥ 1`) this would decide the question, and that only the
§4.3 gate prevented reading it that way. **That was wrong, and the correct
reason is stronger.**

D_alias counts rows where B2 ≠ B, which requires **a valid score on both
sides**. B2's row has no valid score: `judge_answer`'s parse failure returns
the sentinel `{"score": 0.0, "scores": []}`, and that 0.0 is *fabricated* — it
is neither the judge's actual 1.0 nor a real 0.0. So the comparison for that
row is **undefined, not "differs."** Refusing to read it is not discipline
overriding a result; there is no result to read. Declining to fabricate one is
the whole of it.

And "different text" does not entail "different score." It is a fact about the
alias's output-length distribution, not about its score determinism, and the
World 1 falsifier is about scores. The observation is real, it carries no
verdict, and it does not carry World 1 either.

It does imply B's own 0/160 silent-0 was a draw rather than a property: at one
observed occurrence per 160 rows, P(0 in 160) ≈ 0.366.

### 7.4 Step 1 is uncontaminated (checked, because it was reachable)

If truncation can fabricate a 0.0, the obvious worry is that Step 1's
D_self = 7/160 is partly parse artifact. It is not. C1, C2 and B all predate
`finish_reason` capture, so the check is structural — a truncated reply has no
complete `{...}` for the regex to match:

- **B, C1, C2: 0/160 unparseable raws and 0/160 sentinels each.** Longest reply
  879 chars, far below the ceiling that truncated B2's row.
- Every one of the 7 D_self rows and 3 D_pin rows carries a genuine `scores`
  list on **both** sides — e.g. the churning IE row is `[0,1]` vs `[0,0]`, real
  judge disagreement, not a parse failure.

**D_self = 7/160 stands as a clean measurement of the pin's churn**, which
matters because it is the number Step 2's risk assessment rests on.

## 8. What this pre-registration got wrong

The §4.3 gate was ported from the gold-delta phase, where a silent-0 meant
**the plumbing was broken** — a mis-pinned model returning unparseable output.
Here the plumbing is fine and the silent-0 is **the phenomenon under study
manifesting**: the judge's own run-to-run variation, surfacing as a parse
outcome instead of a score change.

The gate therefore voids on exactly the evidence B2 was commissioned to
collect. That is a defect in this spec, not in the run, and naming it is
cheaper than routing around it.

**A superseding B2 v0.2 needs a gate that separates the two.** A first draft of
that rule said: a truncated-but-well-formed scoring attempt is judge variation,
so count it into D_alias as a changed row. **That draft was wrong and would
have re-imported the exact defect this campaign exists to eliminate.**

The truncated row has no valid score. `judge_answer` returns the sentinel
`{"score": 0.0, "scores": []}`, a *fabricated* 0.0. Counting it into D_alias
compares that fabrication against B's real score and calls the difference
churn. That is a parse artifact entering the statistic as a measurement — the
same class as the silent-0 the gate was built to keep out, arriving through the
back door with the gate's own blessing.

**The correct treatment: exclude, do not void, do not count.** Truncation means
the plumbing is fine (so it must not void) *and* the row is unreadable (so it
must not count). That is precisely the shape of the existing `explicit_err`
bucket — keep prior, exclude from the statistic, report the rate separately.

**And the separator is a conjunction, not `finish_reason` alone.** Both failure
shapes carry `finish_reason == "length"`:

| shape | content | caught? |
|---|---|---|
| the v4-flash trap | **empty** + length | yes — the falsy raise, rerouted to `[LLM_ERROR` |
| B2's truncation | **non-empty** + length + parse-fail | **no** — nothing catches it |

Gating on `length` alone would mis-split them again. The rule needs all three
of: `finish_reason == "length"`, non-empty content, and parse failure. Since
1d80f33 that field is recorded per row, so v0.2's discriminator is structural
and fixable in advance rather than a judgement call made with the answer
already visible.

**Not authorised under this spec.** B2 v0.2 is a new pre-registration and a new
run. It is now non-destructive — since 0adcd2c a voided run persists its rows
— but that is a property of the NEXT run, not a recovery of this one.
