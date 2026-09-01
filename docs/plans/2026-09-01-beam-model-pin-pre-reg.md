# Pre-registration: BEAM model pin (deepseek-chat → v4-flash) — 2026-09-01 (v0.1)

Status: **DRAFT — awaiting review, then Atta's approval for the spend.**
Branch: Beam-optimisation. Phase 2 of the sequence banked in
`2026-08-31-beam-gold-delta-rejudge-pre-reg.md` §8 ("Next per plan: §6 registry
general fix, then model-pin pre-registration"). §6 landed in 74039ea.

Two steps, deliberately unequal in cost. **Step 1 is cheap and answers the only
question that can be answered cheaply. Step 2 is the expensive rebase and is
gated on Step 1 passing.** Anyone approving spend should be approving them
separately.

## 1. Question (what this decides)

The gold-delta phase (§8, 2026-08-31) returned **REBASE REQUIRED**: the June
record's meaning changed, so a fresh gold-on canonical is needed. That
canonical must be produced on some model. The choice is between:

- **the alias** `deepseek-chat` — currently working, but the server maps it to
  v4-flash non-thinking with no model-identity field in the response. Nothing
  in an artifact can witness which model actually graded it.
- **the pin** `deepseek-v4-flash` + `thinking: {"type": "disabled"}` — which
  `references/deepseek-model-migration.md` asserts is the "byte-path-equivalent
  of old deepseek-chat (non-thinking mode)".

Installing a new canonical whose `judge_model` is a name the server silently
remaps repeats, on the model axis, the exact defect the gold-delta phase
retired the June record for on the gold axis: a number whose referent can move
without the record changing. So the pin is preferred **if** the migration doc's
equivalence claim is true.

**That claim has never been measured. It is an assertion in a reference doc.**
Step 1 measures it, for ~5% of the cost of assuming it.

**How much a PASS is worth (stated before the run, so it cannot be inflated
after):** if the server's mapping is what the doc says, B's judge and C1's judge
are BOTH v4-flash-nonthinking — one reached by a name the server resolves, one
named explicitly. Then Step 1 is close to a tautology and passes trivially.
**A PASS is therefore weak positive evidence: it confirms the doc's claim held,
not that anything was learned.** Step 1's power lies entirely in the other
branch — it is cheap insurance against the mapping being wrong or having
drifted, not a discovery instrument. That asymmetry is the reason it is worth
ten minutes and would not be worth ninety.

- **Falsifier (H0), Step 1:** re-judging artifact A with the PINNED judge
  reproduces artifact B's scores exactly. Then the pin is score-equivalent on
  this workload, the doc's claim survives a real test, and B stays a valid
  comparator across the pin.
- **Reject H0:** the pin grades differently → the doc's "drop-in replacement"
  claim is false for this workload, and that is a finding worth recording
  whichever way Step 2 then goes.

## 2. Design

**Step 1 — pin equivalence (judge-only, replayable, cheap).**

- **B** = `results_20260831T165039Z-rejudged-deepseek-chat-20260831T200531Z.json`
  (judge_gold=true, judge=deepseek-chat, 160 rows, judge_calls=161 incl. canary,
  elapsed 290.9s).
- **C1** = the same rejudge of the same A, gold on, judge pinned:
  `--rejudge results_20260831T165039Z.json --judge-gold
   --judge-model deepseek-v4-flash
   --judge-extra-body '{"thinking": {"type": "disabled"}}'
   --prereg docs/plans/2026-09-01-beam-model-pin-pre-reg.md
   --dataset-revision 3205395e897e7318c7b094ef4e6047b9b82dbb03`
- **C2** = C1 repeated, unchanged. This is not redundancy: it measures the
  PINNED judge's own churn, without which a C1≠B difference cannot be
  attributed between "different model" and "non-deterministic model".
- One knob between B and C1, not two. It looks like a compound treatment —
  model name AND thinking flag — but the flag is not an independent variable:
  it is the mechanism that makes the explicit name mean what the alias already
  resolves to. You cannot send v4-flash without it; that is the trap, not a
  condition. So the comparison is "what we deploy now" vs "what we want to
  deploy", each taken whole.
- Answers are byte-fixed from A, gold is the same reparse under the same 160/160
  guard, prompts are `_judge_messages` on both sides — pinned byte-identical by
  `tests/test_judge_messages_byte_identical.py`.
- B, C1 and C2 are structurally comparable: each is a rejudge, so each carries
  its own judge canary and reports `judge_calls` = 160 rows + 1 canary, with
  `answer_calls = 0`. C1/C2 showing 161 is expected, not an anomaly.
- **Either causal path lands on the same conclusion.** If C1 ≠ B, the cause is
  either alias≠v4-flash or the flag changing v4-flash's behaviour, and Step 1
  cannot separate them — but it does not need to: both mean "byte-path-
  equivalent" is false for this workload. The finding is clean without
  disambiguation, which is what makes a two-path treatment acceptable here.

**Step 1 tests the JUDGE pin only.** Stated plainly because it is a real limit,
and stated precisely because the obvious phrasing overstates it:

- The answerer is not *unreplayable* — it is **expensive** to replay. Judging
  replays because A stored every answer; re-answering has nothing stored to
  replay, so an answerer-equivalence arm means ingestion + retrieval + a full
  answer pass on both models. That is a second Step 2, not a ten-minute
  rewrite. **The limit is a budget line, not a law of physics**, and it should
  not be quoted as though it were one.
- **The decisive reason is that no such arm is needed.** Step 2 defines a NEW
  canonical, and a fresh baseline does not need its answerer to equal A's — it
  needs its answerer to be *witnessed*, which is exactly what pinning buys and
  the alias cannot give. Equivalence-to-a-retired-record is not a property the
  new record requires. This, not the cost, is why the arm is absent.
- The pinned answerer's positive evidence is the main-path canary: a real
  answer prompt at the real 1024-token ceiling returning non-empty content on
  the pinned client. That is a liveness check, not an equivalence result.

**Nobody should read Step 1 passing as evidence about the answerer.**

**Step 2 — the rebase (full run, expensive).** Fresh gold-on run at the anchor's
configuration (100K, sample=8, top_k=10), answerer and judge both pinned. This
becomes the canonical; A retires as a comparison point and is never again read
as regression or improvement.

## 3. Verified facts (evidence, 2026-09-01, on this branch)

- Plumbing landed in **e3313c0** ("beam: model-pin plumbing — an empty
  completion can no longer score 0"): `LLMClient`
  takes `extra_body` and merges it last; `_call` raises on FALSY content, not
  merely null; `check_model_pin` refuses v4-flash-without-thinking-disabled and
  `thinking`-aimed-at-OpenAI/Gemini; a real-prompt canary runs on BOTH clients
  before the run spends; artifacts record `answer_extra_body`/`judge_extra_body`.
  Suite 1505 passed / 1 skipped (`/home/node/.venv`).
- **`extra_body` defaults to empty.** An unflagged run sends the same four body
  keys it sent before the plumbing existed, so A and B remain comparators. This
  is a deliberate divergence from `hymem/contrib/openai_client.py:81-86`'s
  `auto` host-substring gate: a library may inject by default, a benchmark may
  not, because injection retires comparators without anyone deciding to.
- **`longmemeval_adapter.py:384-392` raises only on `content is None`** — it
  does NOT implement the three-way `content or reasoning or reasoning_content`
  fallback the migration doc's table claimed. The trap shape is `content == ""`
  with `finish_reason=length`, which a null-check passes through. The doc row
  described a recommendation as landed code and is corrected in this series.
  LME's actual protection is its flag plus its canary, not its client.
- The judge is DeepSeek-only on beam (no `base_url` override, no provider
  resolution). Only the ANSWERER is provider-swappable (`ANSWER_PROVIDERS`:
  deepseek/gemini/openai), which is why the `thinking`-key-to-wrong-provider
  refusal exists at all.
- B's judge is **score-deterministic at temp 0.0**: §8 recorded 32/32 control
  rows per-row identical (SD_ctl = 0.0000), 4/4 fresh re-judges reproducing
  stored scores and raw lengths, and 104/128 exact-zero pool deltas. This is
  what makes Step 1 a sharp test rather than a noisy one — but it is a fact
  about the ALIAS, and C2 exists because it may not hold for the pin.
- Cost anchors, measured not estimated: B = 290.9s for 161 judge calls. A (full
  run, answers + judge + ingestion) = 5471.6s / 160 answer + 160 judge calls.
- **The dataset was the one input nothing witnessed.**
  `load_dataset("Mohammadta/BEAM")` carried no revision, which is the
  `deepseek-chat` hazard on the data axis: a name whose referent the host can
  move with no artifact showing it. The rejudge path was already covered by its
  160/160 reparse guard (it aborts if the gold moved, and never regenerates
  answers), but a full run has no stored baseline to diff against. Since
  abd692c's follow-up the revision is resolved, recorded, and pinnable.
  Resolved live 2026-09-01 under the run interpreter
  (`/home/node/hymem-env/bin/python` — the one with `datasets`, NOT the
  `.venv` the suite runs in): `Mohammadta/BEAM` =
  `3205395e897e7318c7b094ef4e6047b9b82dbb03`, `Mohammadta/BEAM-10M` =
  `9b2096193fe74e2837e4713e483351e19817773c`, both last modified
  2026-01-30 — stable for seven months, so this is insurance, not a live fire.

## 4. Procedure

1. Commit the plumbing and this pre-registration BEFORE any run, and record the
   pre-registration's commit hash in the metadata of every artifact produced
   under it, alongside `judge_gold` and `gap_hours`. A verdict whose spec-hash
   post-dates its artifact is void by construction.
   **Enforced, not merely asked for, since abd692c:** every invocation below
   passes `--prereg docs/plans/2026-09-01-beam-model-pin-pre-reg.md`, which
   refuses to run unless this file is committed and unmodified and unless the
   tracked code is clean, and writes `prereg.commit` / `prereg.blob` /
   `prereg.code_commit` into the artifact. Plumbing: e3313c0. Spec: 3924ed8.
2. Run C1. The canary must pass on the pinned judge — it is the first real
   evidence that the pin returns content at all on this path.
3. Run C2, identical invocation. Do not look at C1's scores first.
4. Hard gate, before any comparison: C1 and C2 must each have **0/160 silent-0**
   (non-empty rubric AND `scores == []`) and **0/160 explicit `[LLM_ERROR`**.
   B's rate was 0/160 on both. Any nonzero → the pin is broken, Step 1 FAILS,
   and no score comparison is interpretable. Report and stop.
5. Compute the statistics in §5 and record them in §8.
6. Only if Step 1 PASSES: land the default flip (§6), then request approval for
   Step 2 separately.
7. **Step 2 pins `--dataset-revision` to the SAME sha C1/C2 recorded.** Approval
   for Step 2 may arrive days after Step 1, and an unpinned canonical scored on
   a dataset that moved in between would be incomparable to the very arms that
   authorised it — silently, since only the artifact's revision field would
   differ. Pinning turns that into a no-op instead of a discovery.

## 5. Read protocol (fixed before counts)

- **D_self = #rows where C1 ≠ C2**, over 160. This is the pinned judge's own
  churn and it is read FIRST, before D_pin, so the band is not chosen with
  knowledge of the result.
- **D_pin = #rows where C1 ≠ B**, over 160. **δ̄_pin = mean(C1 − B)**.
- **PASS ⇔ the gate in §4.4 holds AND either:**
  - **D_self = 0 and D_pin = 0** — the pin is a deterministic, score-identical
    grader. The doc's equivalence claim is confirmed on this workload and B
    survives as a comparator across the pin; or
  - **D_self > 0 and |δ̄_pin| ≤ 2·SD_self/√160 and D_pin ≤ D_self** — the pin
    churns, but C1's distance from B is within the pin's own churn, i.e. the
    difference is not attributable to model identity.
- **FAIL ⇔ D_self = 0 and D_pin > 0.** A deterministic grader that disagrees
  with B on even one row is a DIFFERENT grader; "drop-in replacement" is then
  false for this workload and the doc must say so.
- **What FAIL means for Step 2, decided now rather than after seeing numbers.**
  The commitment first: FAIL does NOT send the canonical back to the alias.
  An unstable grader is worse than a different-but-stable one — the alias
  cannot be witnessed from an artifact, the pin can. Written down in advance so
  a FAIL cannot be quietly re-read later as a reason to keep the alias.

  But a bare count cannot tell two structurally different FAILs apart, and they
  do not cost the same, so **both sub-branches are decided here.** Let S be the
  rows where C1 ≠ B, and s(r) = sign(C1 − B) on each:

  - **FAIL₁ — rescale.** Every row in S shares one sign. The pin is uniformly
    harsher or laxer; B's *shape* is intact and the offset is recoverable.
    → Pin adopted. **B carries across as a shape comparator only**: per-ability
    ordering and relative structure survive, absolute comparisons against B are
    void, and δ̄_pin is recorded as the offset that makes that explicit.
  - **FAIL₂ — different interpreter.** Signs diverge within S. The pin is not
    shifted, it reads the rubric differently.
    → Pin adopted. **B does not carry across at all.**

  Sign-homogeneity is the discriminator because it is crisp enough to
  pre-register; the spread of the deltas within S is reported descriptively
  alongside it, never used to re-classify after the fact.

- **The full cost of FAIL₂, named now so it is not discovered at the readout:**
  the pinned canonical becomes a **stand-alone baseline with no ancestor**. No
  claim of the form "improved by X pp" is available against anything —
  only "the pinned model at this configuration scores Y". That is strictly
  weaker than what B-comparability currently affords, and it is the real price
  of the commitment above.
- **Per-ability (n=16): descriptive only.** One flip is 6.25pp; this table is
  for locating a difference, never for declaring one.
- No top-up, no re-ranking, no second cutoff. Step 1 is a replay: if it is
  ambiguous, it is ambiguous.

## 6. The default flip (its own commit, only after Step 1 PASSES)

`ANSWER_MODEL`/`JUDGE_MODEL` (`beam_adapter.py:44-45`) move to
`deepseek-v4-flash`. The flip is NOT part of the plumbing commit: the plumbing
is inert by design, while the flip is the change that decides what an unflagged
run means.

The flip needs a paired rule, because a v4-flash default with an empty
`extra_body` default would make `check_model_pin` refuse every unflagged run:
**when the resolved model is a DeepSeek v4-flash AND the operator passed no
`--{role}-extra-body`, default it to `thinking: {"type": "disabled"}`, PRINT
that it was defaulted, and record it in metadata like any other value.**

This is narrowly scoped auto-defaulting and the scope is the whole argument: the
match term is **`"v4-flash" in model`, NOT `"deepseek" in model`**. The latter —
the library client's gate — would also fire on `deepseek-chat`, silently adding
the flag to the alias path, changing A/B byte-identity and retiring the
comparator. The v4-flash term cannot fire on the alias. It fires only where the
alternative is a known-broken run, so it cannot retire a comparator: no artifact
worth comparing to was ever produced by bare v4-flash. On every path that has
working artifacts the default stays empty and the bytes stay unchanged.

Two constraints on the implementation, both load-bearing:

1. **Ordering.** The default must be applied strictly BETWEEN the flag-parse
   loop and `check_model_pin`. If it runs after the guard, a bare v4-flash run
   is refused before the default can fire and §6 silently does nothing.
2. **"Passed no flag" must mean ABSENT, not empty.** The flags' argparse default
   changes from `""` to `None`: `None` = absent → default may fire; `''` or
   `'{}'` = the operator explicitly asked for no extra body → the default does
   NOT fire and `check_model_pin` refuses the run. A convenience must never
   override an explicit statement, and today those two cases parse identically.

**Accepted forbearance:** with §6 and the guard both in place, thinking-ENABLED
v4-flash becomes unreachable through this adapter, so the cost of the trap can
no longer be measured here. That is the correct posture for a benchmark — the
trap is not a condition anyone should be able to select by accident — but it is
a capability given up, recorded here rather than rediscovered as a surprise.

## 7. Cost & non-actions

- **Step 1: ~320 judge calls, ~10 minutes, zero answer calls, no ingestion, no
  dream, no store writes.** Two replays of a judge pass measured at 290.9s.
- **Step 2: ~160 answer + ~160 judge calls plus ingestion and dream, ~91
  minutes** by A's measured 5471.6s. This is the expensive one and the only one
  that needs a real spend decision.
- NOT in this phase: the three-way `content or reasoning or reasoning_content`
  fallback (that is hardening for non-DeepSeek reasoning ANSWER models such as
  `gpt-oss-120b` via OpenRouter, and belongs with a run that actually uses one);
  arm A of the original three-arm design; competitor-table reads; any change to
  retrieval, dreaming, or scoring.
- Step 1 changes no code. If it fails, nothing has been spent but ten minutes
  and the finding is itself the deliverable.

## 8. Executed results — Step 1 (2026-09-01)

Artifacts, all in `/home/node/hymem_beam/`:
- **B** `results_20260831T165039Z-rejudged-deepseek-chat-20260831T200531Z.json`
- **C1** `...-rejudged-deepseek-v4-flash-20260901T055638Z.json`
- **C2** `...-rejudged-deepseek-v4-flash-20260901T055957Z.json`

C1 and C2 both record `prereg.blob` 86f0a91c315d, `prereg.code_commit`
b4f4350e, `dataset_revisions` {Mohammadta/BEAM: 3205395e...}. The scorer
(`benchmarks/step1_pin_compare.py`) was committed at **32e81ee before any
C1-vs-B comparison was computed** — both arms existed and had passed their
gates, no pre-registered statistic had been read.

**Gate (§4.4): PASSED.** C1 and C2 each 0/160 silent-0 and 0/160 explicit
`[LLM_ERROR`, matching B's rates. Canary OK on both runs — 217 chars of content
on the full judge prompt at max_tokens=512, on the pinned model that returns
`""` without the flag. 198s and 194s; 161 judge calls each (160 rows + 1
canary, counted separately from `judge_calls`).

**D_self (read first, per §5) = 7/160.** SD_self = 0.081355 → band =
2·SD_self/√160 = **±1.2863pp**. By arm: control ABS/CR 1/32 (SD 0.044194),
pool 6/128 (SD 0.088287).

**D_pin = 3/160. δ̄_pin = −0.3646pp.**

**VERDICT: PASS**, via the second branch — |δ̄_pin| 0.3646pp ≤ band 1.2863pp,
and D_pin 3 ≤ D_self 7. C1's distance from B is inside the pin's own
run-to-run churn, so the difference is not attributable to model identity.
Per §1 this is weak positive evidence, as pre-registered: it confirms the
migration doc's claim held, it does not establish that anything was learned.

Per-ability (descriptive only, n=16 each): pool B 0.5247 → C1 0.5201
(−0.46pp); control ABS/CR 0.5859 → 0.5859 (+0.00pp); overall 0.5369 → 0.5333
(−0.36pp).

### 8.1 The unanticipated finding: the pinned judge is not deterministic

**This was not a question Step 1 was designed to ask, and it is the most
consequential thing the run produced.** 7/160 rows (4.375%) move between two
byte-identical invocations of the pinned judge at temperature 0.0.

The gold-delta phase recorded the opposite for the alias — "Control arm ABS/CR:
32/32 per-row identical (SD_ctl = 0.0000). The judge is SCORE-DETERMINISTIC at
temp 0.0 on byte-identical prompts", plus 4/4 fresh re-judges reproducing — and
built a **zero-width band** on that zero. Its primary then read δ̄ = −0.582pp as
OUTSIDE that band, which is what produced **REBASE REQUIRED**; the companion
was INSIDE, and that pre-registration states plainly that the OR fired on the
zero-width primary alone.

Recomputed under **§8's own formula** (`SE = SD_ctl/√128`, `band = 2·SE`, with
SD_ctl the SD of the 32 control-arm deltas — not Step 1's `2·SD/√160`, which is
a different quantity), substituting the pin's control-arm SD 0.044194 for the
alias's recorded zero: **band = ±0.7812pp, and |−0.582pp| is INSIDE it.** It is
inside on every candidate substitution — pin all-row ±1.4382pp, pin pool
±1.5607pp. The primary does not survive any realistic variance estimate.

Sharper still: the SD_ctl at which −0.582pp is exactly borderline is 0.032923,
which for 32 rows where one moves and the rest do not is **one control row
moving by 0.1862**. The pin moved one control row (CR) by 0.25. A single
quarter-point move in 32 rows is the entire distance between REBASE REQUIRED
and H0 holding.

**Interpretation is deliberately withheld, because Step 1 cannot support it.**
The measured churn is the PIN's; the gold-delta band was the ALIAS's, and this
run establishes that the two graders agree on SCORES, not that they share a
CHURN RATE. Step 1's PASS is compatible with two opposite worlds and is blind
between them:

- **World 1** — the alias churns too, §8's 0/32 was a small-sample draw
  (P(0 in 32) = 0.239 at the pin's rate; P(0 in 36) = 0.200, so its evidence
  could not have distinguished determinism from 4.4% churn), and the gold-delta
  verdict is band-dependent.
- **World 2** — the alias really is deterministic and the pin is the noisier
  instrument. Scores agree, which is why the PASS fired, but the pin would then
  buy witnessability at a measured cost in reproducibility, and the migration
  doc's byte-path-equivalence headline would be false on the sampling axis.

Evidence pointing at World 2, recorded so the convenient reading is not banked
by default: A→B was a 3.17h gap on the alias with 0/32 control movement; C1→C2
was 5 minutes on the pin with 1/32. If both names reached the same servable
configuration, the longer gap should expose at least as much drift, not less.

The discriminating measurement is one alias re-run at n=160, where
P(0 | pin's rate) = 0.00078. It is pre-registered separately in
`2026-09-01-alias-churn-b2-pre-reg.md` and is **not authorised under this
spec**. Nothing here should be read as having decided between the two worlds.
