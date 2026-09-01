# Analysis protocol: re-derive the BEAM gold-delta verdict from four arms — 2026-09-01

**This is deliberately NOT titled a pre-registration.** §0 says why. Zero API
calls; read-only arithmetic over five artifacts already on disk, so it falls
inside standing permission. Step 2 (the rebase run) remains unauthorised and is
**not** part of this document.

Discharges the pending task recorded in
`2026-09-01-alias-churn-b2-v03-pre-reg.md` §8.2: *"the between-arm spread is a
direct estimate of the quantity §8 approximated with a single control arm, and
it needs its own pre-registration before anything is concluded from it."* It
discharges it by establishing that a pre-registration is no longer available
here, and by writing down the strongest thing that is.

## 0. Why this is not a pre-registration

**Every load-bearing number in §6 was already known before this document was
committed, and it was not known by accident.**

Two disclosures, in the order they happened:

1. **I had already seen the four per-arm pool means** (−0.582, −1.037, −1.037,
   −1.115 pp, recorded in v0.3 §8.2), had already run a ±2·SE calculation on
   them that excluded zero, and had already reported to Atta that
   "REBASE REQUIRED survives".
2. **The reviewer computed the rest of §6 while reviewing the draft.** I sent
   the draft to Hermes for exactly the kind of formula-level attack that caught
   my last two errors. Hermes did what the campaign's discipline demands —
   verified the prose against the artifacts on disk — and in doing so computed
   the per-ability decomposition, both estimators' intervals, the
   exchangeability contrast, the §4 preconditions and §6.5. It reported all of
   them to me before this file was committed.

That second disclosure is not a complaint. Sending the draft out for review was
right, and reading the artifacts was the reviewer doing its job correctly. But
the consequence is unavoidable: **the rules in §6 were fixed before I computed
the results and after I knew them.** A document in that position is a protocol,
not a pre-registration, and calling it one would be exactly the label inflation
this campaign exists to catch.

What the document is still worth:

- The rules below are **strictly harsher** than the calculation that already
  produced the welcome answer, and `tests/test_gold_delta_rederive.py` holds
  them to that. The load-bearing test constructs an effect the old ±2·SE rule
  calls real and asserts this protocol does not.
- The arms are **hash-locked** (§2) and **may not be topped up** (§8), so the
  one remaining degree of freedom — running a fifth arm because four gave an
  unwelcome answer — is closed.
- Everything the review reported is treated as a **claim to be verified**, not
  a result to be adopted. The reviewer's first parse of the artifacts was
  wrong (it reported `score`/`scores` as strings; they are floats and ints in
  all five files), so its numbers are checked by the scorer here rather than
  quoted. §8 records which of them held.

I have been wrong twice on this question, both times by treating a noisy
single-run estimate as a fixed quantity (v0.3 §8.3). The correct response to
that is not a better-looking document; it is a colder one.

## 1. Question

The original verdict (`2026-08-31-beam-gold-delta-rejudge-pre-reg.md` §8) was
**REBASE REQUIRED**, fired by a PRIMARY whose band was `±0.000pp` because the
single control arm gave `SD_ctl = 0.0000`. B2c has since measured that the alias
judge does churn (`D_alias = 4/160`), so that zero was a small-sample draw and
the band was an artifact.

**Is the pool gold-delta real when its variance is estimated from four arms
instead of from one 32-row control arm?**

- **H0:** the true pool gold-delta is 0 — the anchor's stored scores are what
  gold-on judging yields, up to judge churn.
- **Reject** → the delta is real and REBASE REQUIRED stands, re-derived.
- **Fail to reject** → the original verdict is not confirmed at the measured
  variance, and Step 2's justification on gold-delta grounds is withdrawn.

**I commit to reporting a failure to reject as prominently as a rejection, in
the same words, to Atta.** The prior claim is mine and retracting it is cheaper
than defending it.

## 2. Data — five artifacts, locked by hash

All in `/home/node/hymem_beam/` (`sha256`, first 16 hex):

| tag | file | judge | gold | sha256[:16] |
|---|---|---|---|---|
| **A** | `results_20260831T165039Z.json` | deepseek-chat | **off** | `32fdcf5cb552a3c5` |
| **B** | `…-rejudged-deepseek-chat-20260831T200531Z.json` | alias | on | `44fe25c7e00cd77d` |
| **C1** | `…-rejudged-deepseek-v4-flash-20260901T055638Z.json` | pin | on | `270a38ba7386a308` |
| **C2** | `…-rejudged-deepseek-v4-flash-20260901T055957Z.json` | pin | on | `a76b8134591968f6` |
| **B2c** | `…-rejudged-deepseek-chat-20260901T110546Z.json` | alias | on | `a3d1ebe0e1c01510` |

The hashes mean the arms cannot be silently swapped, re-run or topped up
between this document and its execution. Given §0, this locking is most of what
integrity is left; it is not negotiable.

**Excluded, with reason:** `…-20260901T104602Z-VOID.json` (B2b) — void, and it
carries the fabricated `0.0` from the parse defect. **Not available:** the B2
v0.1 arm, whose rows the pre-`0adcd2c` abort path destroyed. Four arms is what
exists.

## 3. Verified facts (before any statistic)

- **Parser comparability.** `judge_parse_audit.py`: B, C1 and C2 are each
  **0/160 naive-parse failures**, so their stored scores are parser-invariant.
  B2c has **1 `recovered`** row (IF) — the only row where old and new parsers
  could disagree — and it scored `1.0`, matching B. **The four arms are
  comparable across `cb1fd34`, measured rather than assumed.**
- **Provenance asymmetry.** C1, C2 and B2c carry `prereg` and
  `dataset_revisions` (`3205395e…`). **B carries neither** — it predates
  `abd692c`/`b4f4350`. A rejudge reparses the dataset for rubrics and ideal
  answers, so B's unwitnessed revision is a live threat, not a bookkeeping nit.
  §4.2 converts it into a measured precondition.

## 4. Preconditions (checked first; the analysis does not proceed past a failure)

1. **Row identity.** All five artifacts carry the same 160 `(ability, question)`
   keys → else **ABORT**.
2. **Gold identity.** `rubric`, `ideal_answer` and `gold_kind` byte-identical
   across B, C1, C2, B2c. If they are, the gold material that reached the judge
   was the same in all four arms whatever revision each reparsed, and B's
   missing stamp stops being load-bearing. Any differing row → **ABORT**, naming
   the field. Not a silent exclusion: with four arms a differing row means the
   arms are not measuring the same thing, and dropping it would hide that.
   `gold_kind` is included **deliberately** — the review called it over-strict as
   "a categorical label", but it *selects which text is fed to the judge as
   gold*, so it is scoring-criteria-bearing.
3. **Answer identity.** `answer` byte-identical across A and all four arms — a
   rejudge must not have re-generated answers → else **ABORT**.
4. **Void refusal.** Any arm with non-null `void` → **exit 4**.
5. **Readability.** Every row of every arm readable. Any hole → **ABORT**: with
   four arms there is no defensible way to average over one.

## 5. Quantities

Pool **P** = the 8 affected abilities (EO, IE, IF, KU, MR, PF, SUM, TR),
**n = 128**. Control **C** = ABS, CR, **n = 32**. Inherited unchanged from the
original §5; the pool is fixed by the anchor and is not re-chosen here.

`A_i` anchor score; `X_ki` gold-on score in arm `k ∈ {B, C1, C2, B2c}`;
`d_ki = X_ki − A_i`; `δ̄_k = mean_{i∈P} d_ki`; `δ̄ = mean_k δ̄_k`. All in **pp**.

### 5.1 Two different questions, and which one the verdict answers

**The record question (A fixed):** does *this stored record* differ from what
gold-on judging yields? A is the record under test, not a draw from a
population, so holding it fixed is correct — and rebasing is a decision about
this record.

**The process question (A as one draw):** is there a gold-on effect that a
fresh anchor would also show? This needs A's own judge noise, which the four
arms cannot supply, because they all rejudge the *same* A.

**These can diverge, and the review's computation indicates they do here — §6.5
is expected to include zero while §6.1 excludes it.** That is the price of the
A-fixed framing and it is stated here rather than buried: **§6.3's verdict
answers the record question only.** If §6.1 excludes zero and §6.5 does not,
the reportable sentence is *"this record differs from what gold-on judging
yields"* and **never** *"the gold effect is established"*. My post-hoc report to
Atta ("REBASE survives on firmer ground") argued from the four arms as samples
of a process — the process quantity — and §6.5 is where that claim has to be
settled. It must not migrate silently between the two.

## 6. Read protocol

### 6.0 GATE — four-arm control [HARSHER]

`δ̄_ctl,k = mean_{i∈C} d_ki`. ABS/CR prompts are byte-identical whether gold is
on or off, so every control delta is pure churn and `E[δ̄_ctl] = 0`.

Gate: `mean_k δ̄_ctl,k` against `t_{.975,3} · SD_k(δ̄_ctl,k)/√4`. **Excludes 0 →
the arms carry a systematic shift on byte-identical prompts → the pool delta is
unattributable → VOID**, no verdict, whatever §6.1 says. Degenerate SD = 0 →
the original's flip gate (any control flip → void).

### 6.1 PRIMARY — a conjunction of two tests [HARSHER]

Two estimators of the same centre under different assumptions:

**E1 — row-level churn.** `s²_i` = sample variance of `{X_ki}_k` over `i ∈ P`;
`σ̂²_churn = mean_i s²_i`; `SE_E1 = √(σ̂²_churn/(128·4))`.

**Degrees of freedom, corrected.** The draft claimed `128 × 3 = 384` df. That is
wrong whenever the per-row variances are heterogeneous — and the review reports
that ~120 of 128 pool rows have `s²_i = 0` (all four arms identical), so nearly
all the variance is carried by a handful of rows. `σ̂²_churn` is an average of
128 quantities that are mostly structural zeros, and its sampling distribution
is nothing like χ²₃₈₄. **Use the Satterthwaite effective df**
`ν_eff = 3·(Σ s²_i)² / Σ (s²_i)²`, computed and reported, with `t_{.975,ν_eff}`
from an exact quantile rather than a hardcoded constant. This is a real
correction and it widens E1.

**E2 — arm-level.** `s²_arm` = sample variance of `{δ̄_k}` (3 df);
`SE_E2 = √(s²_arm/4)`; `t_{.975,3} = 3.182` — **not 2**, which is what the
calculation I already reported used, and which is anti-conservative by 1.59×.

E1 cannot see an arm-wide correlated shift; E2 can, but on 3 df. Neither is
known to hold.

**RULE: H0 is rejected only if BOTH intervals exclude zero.** Since both share a
centre this is identical to "the wider interval governs", but the conjunctive
statement is the one with a meaning: it is an intersection–union test,
conservative by construction, and it answers the review's correct objection that
`max` of two SEs is not a probability model. What it buys is exactly one thing —
**a rejection cannot be an artifact of which estimator was chosen** — and that
is all it may be reported as buying.

### 6.2 Exchangeability of the two judge configurations

Pooling two alias arms with two pin arms assumes judge configuration is not a
fixed effect. Contrast `Δ_judge = mean(δ̄_B, δ̄_B2c) − mean(δ̄_C1, δ̄_C2)`, pooled
within-stratum SD on **2 df**, band `t_{.975,2}·s_pooled`.

**Exceeds the band → NOT exchangeable → §6.1 invalid → NO POOLED VERDICT**,
including a favourable one.

**Stated so it cannot be quoted as reassurance:** at 2 df this has almost no
power, so a PASS is *weak*. The real support for pooling is external — Step 1's
score-level PASS, and `D_alias = 4/160` against `D_pin = 7/160`.

### 6.3 VERDICT (strictly in this order)

1. §6.0 VOID → stop.
2. §6.2 fails → **NO POOLED VERDICT**.
3. Both §6.1 intervals exclude 0 → **CONFIRMED**, subject to §6.7's naming
   constraint. On the **record question only** (§5.1).
4. The envelope includes 0 but E1 alone excludes → **AMBIGUOUS**: withdrawn as
   a demonstrated result, reduced to the decision-theoretic asymmetry the
   original §5 argued for — *a reason to act, not evidence the effect exists*.
5. Both include 0 → **NOT CONFIRMED**. The record stands; Step 2's gold-delta
   justification is gone.

### 6.4 COMPANION — descriptive, does NOT OR into the verdict [HARSHER]

Binarize at the original's `t = 0.45`; per arm `D_k`, gained, lost, `net_k`;
four-arm mean net with its `t₃` band.

The original §5 OR'd the companion in, on the stated ground that wrongly
trusting a contaminated record costs more than an unnecessary rebase. **That is
a decision asymmetry and it does not transfer.** This is a measurement of a
claim I have already twice announced, and granting myself a second chance to
fire on the same data is precisely the looseness both prior errors came from.

### 6.5 The process question (§5.1) — reported, never suppressed

`SE_fresh = √(SE_primary² + σ̂²_churn/128)` with the same `t₃`, approximating A's
gold-off churn variance by the measured gold-on one (**unverified**, labelled
wherever it appears).

**This interval is reported in the verdict block, not in a footnote, whichever
way it lands.** The draft demoted it to "descriptive" without saying which way
it would go; the review's computation says it includes zero. Hiding a number
that contradicts the top-line is the failure mode this document exists to avoid.

### 6.6 Per-ability — descriptive

n=16, one flip = 6.25pp. Four-arm mean per ability with its `t₃` band. **No
per-ability delta is individually powered**, and the original §8's heterogeneity
argument ("EO −7.57pp, SUM −4.37pp vs IF/KU +6.25pp") is **not** re-banked.

The review argued from this that the pool mean cannot carry a verdict either,
since it is an average of quantities that individually cannot. **That inference
does not hold** — aggregating underpowered components is what pooling is *for*,
and n=128 is powered where n=16 is not. What the review's underlying concern
does establish is a real constraint on *description*, which §6.7 makes binding.

### 6.7 CONCENTRATION — constrains how a verdict may be worded, not whether it fires

The review reports that the pool mean's sign is carried by EO alone, with the
other seven abilities netting near zero. **If true, that does not invalidate
§6.3 — the H0 being rejected is "no shift", and a shift concentrated in one
ability is still a shift — but it makes "a pool-wide gold-delta" a false
description of a true rejection.** Description is where a verdict does its
damage, so:

- Report each ability's **contribution** `(n_a/128)·δ̄_a` to the pool mean.
- **Leave-one-ability-out:** recompute the pool mean and its §6.1 envelope over
  the other 7 abilities, for each ability in turn.
- **Binding rule.** If the verdict is CONFIRMED and *any* single ability's
  removal makes the leave-one-out interval include zero, the verdict **must** be
  reported as *"CONFIRMED, carried by <ability>"*, and the words "pool-wide" and
  "broad" are **forbidden** in reporting it. Only if no single ability's removal
  does that may it be described as pool-wide.

This is deliberately a reporting rule and not a verdict rule. Changing the
verdict *quantity* after seeing which ability carried it would be the textbook
post-hoc move, and §0 already concedes enough ground without adding that.

## 7. Procedure

1. This document is committed **before** `benchmarks/gold_delta_rederive.py`
   reads an artifact. The scorer takes `--prereg` and refuses an uncommitted or
   modified spec (`abd692c`).
2. The scorer is committed **before** its output is read (`32e81ee`, `bb16b60`).
3. Execution appends §8, recording what the rules produced — including a failure
   to reject, and including which of the review's reported numbers held.

## 8. Cost and non-actions

- **Zero API calls.** Five JSON reads and arithmetic. No `datasets` import, so
  `/home/node/.venv` runs it.
- **NOT in this phase:** Step 2 / the rebase run (unauthorised, ~91 min, needs
  Atta); any fresh judging; the `beam_runs.db` ingestion; the EO/SUM
  systematic-difference probe, which is a *future* target and not this one.
- **No arm may be added after §6 is computed.** A fifth arm run because four
  gave an unwelcome answer would destroy the only discipline this campaign has.

## 9. Executed results — 2026-09-01

Protocol blob `f0d0005bd48b` @ `4b91b733`, code `bc0ead5c`. Zero API calls.
Re-run after the §6.6 wording fix (`d81bf068`) reproduced every number.

### 9.1 Preconditions — all pass

160/160 keys identical across five arms; **rubric, ideal_answer and gold_kind
byte-identical across B, C1, C2, B2c**; answers byte-identical across all five;
160/160 readable in every arm; no arm void. `judge_parse` recovered: B 0, C1 0,
C2 0, **B2c 1**.

**§4.2 did the work it was written for.** B carries no `dataset_revisions` and
no `prereg`, and a rejudge reparses the dataset for gold — but the gold material
is byte-identical across all four arms, so **B's unwitnessed revision is not
load-bearing.** That is now measured rather than assumed.

### 9.2 Gate, primary, exchangeability

| | value |
|---|---|
| per-arm control δ̄ | B `+0.000` (0/32), C1 `+0.000` (0/32), C2 `+0.781` (1/32), B2c `+0.781` (1/32) |
| **§6.0 GATE** | mean `+0.391pp ± 0.718` → contains 0 → **PASSES** |
| point estimate | **δ̄ = −0.943pp** |
| **E1** row-level | σ̂_churn 0.04935, SE 0.2181pp, **ν_eff 16.6** (t 2.114) → **[−1.404, −0.482]** |
| **E2** arm-level | SD_arm 0.2437pp, SE 0.1218pp, ν 3 (t 3.182) → **[−1.331, −0.555]** |
| **§6.1** | **both exclude 0**; E1 is the wider |
| **§6.2** | contrast `+0.267pp`, band `±0.995pp` → PASSES (weak, 2 df) |

**The Satterthwaite correction mattered.** ν_eff is **16.6**, not the 384 the
draft claimed — 128 rows contributing three degrees of freedom each, when nearly
all of them are rows every arm agrees on and contribute a structural zero. The
honest df widens E1 from `±0.429` to `±0.461` and makes it the wider interval,
so **the estimator I expected to be decorative is the one that governs.**

### 9.3 VERDICT: **CONFIRMED, carried by EO/PF/SUM**

The pool gold-delta is real at the four-arm variance. **REBASE REQUIRED stands,
re-derived on a measured variance rather than a zero-width band** — and it
survives an intersection–union rule, a t₃ where I previously used 2, and a
control gate that could have voided it.

**But three of the eight abilities are individually load-bearing**, so §6.7
binds and the words *pool-wide* and *broad* are not available:

| ability | contribution to δ̄ | drop it → |
|---|---|---|
| **EO** | **−1.102pp** | `+0.182 ± 0.476` — **includes 0** |
| **PF** | **−0.586pp** | `−0.408 ± 0.455` — **includes 0** |
| **SUM** | **−0.361pp** | `−0.665 ± 0.693` — **includes 0** |
| MR | −0.391pp | `−0.631 ± 0.527` — excludes 0 |
| TR | +0.000pp | `−1.078 ± 0.527` — excludes 0 |
| IE | +0.033pp | `−1.115 ± 0.479` — excludes 0 |
| IF | +0.684pp | `−1.859 ± 0.491` — excludes 0 |
| KU | +0.781pp | `−1.970 ± 0.527` — excludes 0 |

Strip EO and the pool mean flips sign to `+0.182pp`. The record is not
uniformly degraded: **EO/PF/SUM/MR move down while IF/KU move up**, and the net
is negative. That heterogeneity is what the original §8 flagged and what §6.6
still refuses to re-bank at n=16 — but §6.7 makes it bind on the wording, which
is where a verdict does its damage.

### 9.4 §5.1 BINDS — the record question confirms, the process question does not

`§6.5: −0.943pp ± 1.463 → [−2.406, +0.520]` — **includes zero.**

So the reportable sentence is ***"this record differs from what gold-on judging
yields"***. **Not** *"the gold effect is established"*.

**This retracts the sentence I sent Atta.** I wrote that four arms show
"REBASE REQUIRED survives — the gold delta is real at about −0.94pp". That
argued from the arms as samples of a *process*, which is §6.5's quantity, and
§6.5 does not confirm. The verdict that survives is the *record* one. It is
sufficient for the rebase decision — rebasing is about this record — but it is a
narrower claim than the one I made, and the migration between the two is exactly
what §5.1 was written to stop. That is the **third** correction in this series,
and unlike the first two it is a correction of scope rather than of arithmetic.

### 9.5 Companion (descriptive; did not vote)

`D` = 12, 12, 11, 13; net `−6, −6, −7, −7`; four-arm `−5.078pp ± 0.718`,
excluding zero — where the original's binomial band (`2√D = 6.93`) put it
**inside**. The data did not change; the band construction did. Recorded because
it points the same way as the primary, and **not banked**, because §6.4 removed
its vote before any of these numbers existed.

### 9.6 The review's claims, checked

Verified against the artifacts, since the reviewer's first parse of them was
wrong:

- **Held exactly:** the per-ability table (EO −8.82, IE +0.26, IF +5.47,
  KU +6.25, MR −3.12, PF −4.69, SUM −2.89, TR 0.00); EO's contribution −1.10pp;
  §4.2/§4.3 passing with zero differing rows; **§6.5 including zero**; and
  **E1 being the wider interval — against my explicit prediction that it would
  not be.**
- **Held in substance:** "strip EO and it lands at +0.15pp" (actual `+0.182`);
  "~24 effective df, not 384" (actual **16.6** — right that the df was grossly
  overstated, and the true figure is lower still).
- **Overreached:** *"the pool mean's sign is one ability."* Three abilities are
  individually load-bearing, not one. The finding is real and it is why §6.7
  exists, but "it is EO" is too strong.
- **Wrong:** `score`/`scores` are not strings; they are floats and ints in all
  five artifacts. Its arithmetic was right anyway — but this is why the numbers
  were re-derived here rather than quoted.

### 9.7 Corrections to the scorer made after the run

`d81bf068` replaced a §6.6 line that printed "every one of those deltas sits at
or below the one-flip resolution floor" — false for EO (−8.82pp against a 6.25pp
floor) and for KU (exactly at it), and it would have printed under the very run
that falsified it. Wording only; the re-run reproduced every number.

### 9.8 What this does and does not license

- **Does:** the rebase decision has a defended basis — this record differs from
  gold-on judging by −0.943pp [−1.404, −0.482], driven by EO/PF/SUM.
- **Does not:** any claim that the gold-on effect is established as a property
  of the pipeline (§9.4), or any description of the shift as pool-wide (§9.3).
- **Still unauthorised:** Step 2. It needs Atta, it costs ~91 minutes, and
  nothing here changes that.
- **No arm may be added.** Four arms answered; a fifth would only be run
  because these four gave an answer someone disliked.

## 10. Retraction — §4.2 could not have failed

Found on 2026-09-01 while validating Step 2's canonical, after §9 was committed.

**§4.2 checked `rubric`, `ideal_answer` and `gold_kind` for byte-identity across
the four rejudge arms, and reported that this converted B's unwitnessed dataset
revision into a measured fact. It did not, and could not.**

The rejudge path (`_rejudge_run`) reparses the dataset for gold, judges against
`gold["gold_text"]`, and then writes back the row it inherited from the source
artifact — including that row's `ideal_answer` and `gold_kind`. All four arms
inherit those fields from the same anchor A. **Four copies of one field agree by
construction.** The check had no power to fail, so its passing carried no
information about what gold each arm actually reparsed.

`rubric` is the exception and is unaffected: the rejudge genuinely judges with
the inherited `r["rubric"]`, so identity across arms is both true and meaningful
there.

### 10.1 The check that does bear on it

The question §4.2 meant to answer is whether B's unpinned reparse resolved to
the same dataset content the other three pinned. That is answerable from the
dataset's history rather than from the artifacts:

`Mohammadta/BEAM` `main` has been at **`3205395e`** since **2026-01-30**. Its
eight commits are `initial commit`, `Upload dataset`, and six `Update README.md`
— **no data commit since 2025-11-11.** B ran on **2026-08-31**, seven months
after the last change to `main`, so an unpinned `load_dataset` at that moment
necessarily resolved to `3205395e`: the sha C1, C2 and B2c pinned.

**So the conclusion of §4.2 survives — the four arms judged against the same
gold — and the evidence I gave for it was wrong.** The distinction matters
because the original claim would have held for any dataset with any history,
which is what a check that cannot fail always looks like from the inside.

### 10.2 What this does and does not change

- **§9's verdict stands.** `CONFIRMED, carried by EO/PF/SUM`, the intervals, the
  gate and the concentration rule are untouched: they read `score`, and every
  arm's scores are its own.
- **§9.1's sentence "§4.2 did the work it was written for" is withdrawn.** It
  did not.
- **Future arms are witnessed properly.** `4d9906b` records `judged_ideal` — the
  text the judge actually read — on both paths, and `judge_gold` on the main
  path. A future §4.2 over post-`4d9906b` artifacts compares what was judged
  rather than what was copied.
- The four existing rejudge arms cannot be retrofitted. For them, §10.1 is the
  evidence, and it is external to the artifacts by necessity.

### 10.3 Why this one is worth reading twice

The protocol's §0 said its remaining protections were the harsher rules and the
hash-locking. It did not occur to me that a **precondition** could be the weak
point — I had treated §4.2 as the strongest part of the document and said so in
its commit message. A gate that cannot fail reports success indistinguishably
from a gate that passed, and nothing in the output tells them apart. The tests
in `tests/test_beam_judged_ideal.py` exist because that is not a thing to
rediscover.
