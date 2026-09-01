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
   --judge-extra-body '{"thinking": {"type": "disabled"}}'`
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

## 4. Procedure

1. Commit the plumbing and this pre-registration BEFORE any run, and record the
   pre-registration's commit hash in the metadata of every artifact produced
   under it, alongside `judge_gold` and `gap_hours`. A verdict whose spec-hash
   post-dates its artifact is void by construction.
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

## 8. Executed results

(empty — nothing has been run under this pre-registration)
