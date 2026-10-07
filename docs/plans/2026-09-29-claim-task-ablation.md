# Claim-task ablation — diagnosis before another semantic repair

## Established state

Classification v3 is mechanically verified, but not accepted for LME. Its sealed
live diagnostic passed 20/24 controls with three false rejections, one missed
correction and a failed prose canary. Root replay reproduced all 27 judgments;
176,340 known tokens, complete accounting and independently verified cleanup.
Nothing is running. Preserve v1/v2/v3 sources, labels, receipts and failed results.

The failing controls (2, 6, 9, 12) all returned `not_established` for the original
and every alternative. No parser or storage contract mismatch has been proven.
Null optional time scope is not documented as universal/current. Control 6 even
supplies its exact past scope. Passing controls 3 and 11 prevent a blanket
preference/context explanation. The v3 responses provide no negative rationale.

Do not claim negative answers are the shortest v3 response: an invented valid
one-claim example measured 480 serialized characters for original support,
1,420 for all-negative, and 1,649 for one supported alternative. These are not
token costs or evidence of a model optimization strategy.

## Predeclared diagnostic

Test whether requiring exhaustive alternative assessment affects original-claim
judgment. Arm A is the exact accepted v3 request/schema/parser. Arm B is an
inactive **diagnostic-only** original assessment: identical complete original
claim(s), source/context bytes, user payload and batch hash; same positive ledger,
quote/ownership/prefix/parent rules and bounds. Change only the instructions and
output schema needed to remove alternative assessment. Keep predicate definitions
and input vocabulary. Use an explicit distinct response schema for B. Never
manufacture negative alternative assessments or treat B as a correction gate.

Fixed schedule, maximum 29 new turns, no rechecks or adaptive fills:

1. Fixed controls 2, 3, 5, 6, 7, 9, 11, 12, 13, 17, 19, 21, ascending index;
   each receives A and B. Even indices AB, odd BA (24 turns).
2. Immediately after control 12's pair, one separately labeled B assessment of
   the same source and claim with only `uses` changed to `prefers` (one turn).
   This is not part of the identical-candidate A/B comparison.
3. The retained initial table and prose canary batches each receive A and B,
   table AB, prose BA (four turns). Reconstruct privately from the pinned retained
   ordinary outputs and accepted candidate path; no fresh ordinary extraction.

Report original states and validated ledgers, A's alternatives/final verdicts,
and all malformed results separately. Correction controls' original rejection
can be correct; B cannot establish correction uniqueness or full canary recovery.
Control 5's frozen scope ambiguity remains explicitly labeled and is not used as
decisive mechanism evidence. Controls 17, 19 and 21 must remain negative: a
positivity shift is not an improvement. Never change gold labels to pass.

Interpretation: A negative/B positive suggests task/prompt/schema sensitivity,
not proof of causality from one draw. Both negative leaves source interpretation
unresolved; both positive means the previous miss did not reproduce; A positive/B
negative is contrary evidence. Control 12's nominated `prefers` counterpart only
tests recognition when explicitly nominated, not equivalence to correction search.
There are no repeated draws, confidence estimates or general accuracy claims.

## Sequential implementation and verification

1. Separate GPT-6 Sol: inactive diagnostic contract, exact arm bindings and tests.
   Root reviews and independently verifies both schemas, identical A requests,
   unmodified B data, positive evidence guards, no fabricated alternatives,
   malformed controls and limits before acceptance.
2. Different Sol: source-bound diagnostic transport and fixed-schedule core.
   Reuse accepted warm transport/admission/accounting/cleanup; do not modify it.
   Root independently verifies every scheduled call, no rechecks, privacy-safe
   projections, per-turn schemas, failed-call accounting and fail-closed controls.
3. Different Sol: immutable bundle/runner/reader integration using existing proven
   containment and one-shot receipts. Root reviews, runs the actual entry with
   fake IO, source tamper and cleanup tests, then private zero-inference rehearsal.
4. Only after acceptance seal a fresh source/receipt-bound diagnostic, update the
   monitor before launch and run once. Privately replay every returned judgment
   and verify cleanup independently. No live test is launched by this plan alone.
5. Use the result to justify the next narrow Sol repair, root verification and
   bounded test. A diagnostic result is never full-LME readiness. Do not reroll
   any unchanged campaign to obtain a green draw.

## Unchanged boundaries

GPT-6 Luna subscription, low reasoning, existing auth/quota policy; 29 new turns,
500,000 known-token threshold, 1,800 seconds, 120-second invocations, one worker,
1,930-second service plus 10-second cleanup, 4 GiB/CPU200%/TasksMax256 and existing
25%-remaining quota floor. In-flight or missing usage is unknown, never zero.
No production changes, full-500, increased caps, credits, quota bypass or model
switch. Raw text/logs/stores/credentials remain private on Afrodite. Stop for
safety, integrity, accounting, cleanup or quota/auth failure; preserve valid
semantic rejections without aborting independent scheduled cases. Old pilots
and DeepSeek remain stopped; monitors paused until a new exact run is accepted.

## Status

Stage 1 accepted offline after root read-through and independent verification.
The separate Sol implementation passes 17 tests; root authored and passed 49
additional checks covering all frozen input equivalences, cross-arm binding,
positive evidence/qualifier/context guards, malformed responses and no fabricated
alternatives. These are synthetic contract checks, not semantic accuracy.
Contract SHA256: `6e8df574632533437d6c10db7391cc419da792f180f7794e96950bc318307f89`.
Arm B omits the alternatives field entirely and exposes no final selector.
Next split stage 2 into transport then core, with separate Sol implementations
and root acceptance between them. No new live run is accepted yet.

Stage 2a accepted offline. A different Sol implemented a minimal pinned subclass
of the accepted v3 adapter. Root reviewed its full source and independently ran
the fake-wire lifecycle matrix plus alternating-arm and cross-arm rejection tests.
**46 checks passed** (16 implementation, 30 root). Exact schemas, usage including
failed turns, quota floor, concurrent denial, source/import pinning and owned
cleanup are preserved. The helper executes from captured verified bytes, not a
possibly stale cached module. Transport SHA256:
`55d5fb9de06206156011e1c8cd603823273d3d8923a76c914ae861c508485346`.
Next: different Sol fixed-schedule core, then independent root verification.

The combined prior regression plus accepted new contract/transport checks passed
**1,601 tests, zero failures, 101.43 seconds**. Counts above overlap this total.
This is the scoped extraction/grounding/subscription/candidate/bundle regression,
not the full project suite or a live-model success claim.

Stage 2b accepted offline after root found and Sol repaired draft defects before
any launch: the canary loader initially required the corrected prose predicate;
failure metadata rejected its own cleanup-failure report; metadata types, labels
and complete-budget reconciliation needed stricter consistency checks. No live
run used those drafts. Root's physical-candidate test now reconstructs both
initial batches from eight invented ordinary responses, retains the original
prose `uses` predicate and rejects changed request bytes. It is input derivation,
not a passing semantic canary. **30 checks passed** (5 implementation, 25 root),
including the full 29-call synthetic schedule and beginning/end failure controls.
Core SHA256: `34b16df04807a71dea1d535591b0a481d88fa3b9185080f31e2545a79ffdd4e7`.
Next: separate Sol immutable host/runner/reader integration; no live calls yet.

Stage 3 accepted offline after the separate Sol implementation and root review.
Root fault tests exposed draft-only bootstrap/containment integration, incomplete
usage replay and cumulative/partial accounting gaps; Sol repaired them before
any server launch. The postrun path retains source and receipt checks without
requiring used scratch directories to be empty. Root independently reran
**33 integration tests (all passed)** and the accepted contract/transport/core
set (**142 passed**). These are synthetic/offline checks, not live accuracy.
The accepted immutable bundle is
`/private/tmp/hymem-claim-task-accepted-KXzUiPyJ/bundle`, derivation receipt
`a3924f4b61679eba6f24cc5e2e86c9d8b64fd2f0897b4d2fd484b72656440888`.
Next: source-only server staging, zero-inference containment smoke, private
read-only retained-input rehearsal, and a fresh sealed receipt before the one
fixed 29-call diagnostic. No production changes or full LME are authorized here.
