# LME repair plan — sequential Sol implementation, parent verification

## Scope and evidence

User request: diagnose and fix the three observed issues using separate Sol
agents, with parent verification before the next fix. Existing authorization
also permits necessary paid DeepSeek tests. No production deployment, service
restart, original-store mutation, or full-500 run is part of this repair plan.

The unchanged R5 sample-eight finished with seven completed questions and one
indexing failure (`gpt4_483dd43c`). Its checkpoint reports five correct answers,
two wrong answers and one failed question; these are provisional until the
artifact can be independently validated. The sole quarantined chunk is
`chk_d9c7f58f71d38ae508b690115183b0000dd34416`, three attempts, `resource_limit`,
detail `split:no_admissible_semantic_boundary`. The final validator rejects the
producer-generated `strict_failure` field on that indexing-failure row. Separately,
one recovery invocation on a Q1 benchmark-store clone recovered zero of ten
missing summaries: ten cap rejections, ten paid completions, preserved integrity.

Baseline source manifest:
`c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc`.
Immutable baseline:
`/private/tmp/hymem-parent-frozen-r5-20260924.dkx3z00n/verification-tree-r5`.
New isolated candidate:
`/private/tmp/hymem-r6-sequential-20260925.6dSIIq/candidate`.
The dirty main application checkout and all previous receipts remain unchanged.
Patches, test receipts and this plan are retained in the main checkout.

## Sequential gates

### 1. Failure-artifact validation

- Fresh Sol agent diagnoses the genuine writer → physical checkpoint →
  reconciliation → scored archive → strict validation path.
- Reproduce an indexing failure with the real writer APIs; then admit the
  correctly typed and consistent generated failure flag without admitting
  unknown fields, contradictory verdicts, scoring evidence or forged success.
- Parent reviews the patch, independently reproduces red/green controls, and
  runs affected protocol/registry/checkpoint/mixed-failure tests.
- Audit the same saved R5 sample-eight artifact read-only using an explicitly
  identified corrected validator. Preserve its one failed question and old
  source identity; never rewrite the artifact or call it a successful run.
- Gate: valid failed-run evidence can be inspected, malformed evidence remains
  rejected, test receipts and original artifact hashes are preserved.

### 2. Unsplittable extraction input

- Start a new Sol agent only after Gate 1 is accepted.
- Reconstruct the retained failing chunk on Afrodite using its exact canonical
  source records; raw benchmark text stays there. Reproduce the prepartition
  failure without an LLM and capture bounded lengths/boundary diagnostics.
- Implement a generic bounded fallback for the actual structural cause,
  preserving exact source spans, ownership, contiguous coverage and boundary
  context. Do not drop source, turn failed extraction into empty success, disable
  quarantine, or simply remove resource limits.
- Test dense prose, structured/multiline inputs, Unicode/escaping, long spans,
  seam claims, context-only non-evidence, malformed ownership, and finite
  depth/call limits. Update producer/contract identity where behavior changes.
- Parent verifies patch and tests, then the exact retained chunk offline and
  (if offline gates pass) in one bounded paid extraction replay on a copy.
- Gate: the original deterministic failure is removed, source/coverage and
  failure safety controls pass, and measured live extraction can finish. A
  smaller unit test alone is not sufficient.

### 3. Effective bounded summary recovery

- Start a third fresh Sol agent only after Gate 2 is accepted.
- Replace the contradictory “retain every distinct claim in ≤500 characters”
  recovery instruction with a versioned bounded factual-overview contract.
  Summaries remain non-authoritative; original sources and indexed detailed
  records are neither deleted nor modified. Do not claim semantic completeness.
- Preserve polarity, uncertainty and outcome status for included assertions;
  make prioritization/omission explicit. Preserve the 500-character storage
  constraint without slicing accepted claims mid-text. Any repair calls must
  be bounded and accounted, not an unbounded retry loop.
- Retain contiguous source-replay proofs, private partial drafts, durable retry
  exhaustion, lease/deadline cleanup, legacy/operator text ownership and atomic
  publication. Version affected identities and test old/new state handling.
- Parent verifies realistic dense and cumulative multi-window controls, Unicode
  limits, preservation on rejection, false-success negatives and all existing
  recovery/publication tests. Then test fresh copies of retained benchmark
  targets and invented semantic controls using bounded paid calls.
- Gate: demonstrated recovery effectiveness plus unchanged source/item state;
  passing schema checks or exit zero is not enough.

## Combined verification and reporting

After all three parent gates: freeze a new source/test inventory, run the full
offline suite and relevant Afrodite-runtime controls, then run an explicitly
versioned fresh LME regression test. Keep the prior sample in the evidence set;
no rerolls, altered denominators, hidden failures or deletion of failed outputs.
Paid tests run only on benchmark/invented data, with exact model/source/settings,
completion/HTTP accounting, deadlines and process cleanup. No credential content
or raw benchmark data is exported. Missing dollar-cost data stays unknown.

Report separately: data integrity, source-backed indexing, summary recovery,
run completion, answer accuracy, and full-benchmark limits. If any gate fails,
return to the responsible Sol agent and reverify; do not start the next fix.

## Progress

- Planning and isolated R5 copy: complete.
- Fix 1: parent accepted. New real-writer regression fails on R5 for both
  indexing failure codes; candidate passes 666 affected tests. The unchanged
  saved sample-eight validates with the corrected validator, preserving seven
  completed questions, one failure, five correct, two ordinary wrong, and its
  original checkpoint/archive hashes. No paid rerun. Receipt:
  `docs/patches/2026-09-25-parent-r6-fix1-audit.json`.
- Fix 2: Sol implemented unit-labelled numeric-table splitting. Parent red
  controls fail on R5 as expected; exact retained-source offline replay passes
  with five leaves and unchanged original store. Parent integration tests caught
  a canary-version compatibility regression: runtime v18 now describes splitter
  v11, while immutable archive-v18 policy correctly binds splitter v10. Returned
  to the same Sol agent for explicit current-version advancement and historical
  compatibility tests. The followup advances current canary to v19 and keeps
  historical v17/v18 rules pinned. Parent reran 992 tests: all pass. Revised
  offline replay verifies exact five-leaf coverage, unchanged original store,
  historical admission of the real saved R5 archive, and its rejection as
  current output. One bounded paid extraction replay is being launched under
  package `d3098c82c3790bc609f96b0535e17c9f725f27ee33223e6981001a00144ed9f3`.
  Parent accepted Gate 2: one live invocation completed extraction in 10
  completions / 10 HTTP attempts, producing four triples, no extraction failure.
  Exact coverage, citation checks, original store/source preservation and
  supervisor/client/lease-independent process cleanup passed. Total tokens
  41,919; dollar cost unavailable. Receipt:
  `docs/patches/2026-09-25-parent-r6-fix2-live.json`.
- Fix 3: fresh Sol implemented and froze the bounded-overview v2 prompt and
  historical v1 source-proof handling. Exhausted unchanged targets retain their
  exhaustion; partial old-version walks restart privately from the beginning.
  Parent red checks fail on genuine R5 as expected. Parent affected integration
  tests and the separate diagnostic-safety controls are running. Paid gate has
  not started; it will use a new Q1 backup and four separate invented controls,
  bounded at 124 completions / 372 HTTP attempts. Prior failed diagnostic output
  remains immutable. Gate 3 is NOT yet accepted.
  Update: the first live v2-prompt test failed effectiveness: 0/10 retained
  targets recovered (one private window advanced), 3/4 invented controls
  recovered. All eleven held windows failed `summary_output_cap`. There were
  17 completions / 17 HTTP attempts and 36,019 tokens; cost unknown. Original
  data, all non-summary state and cleanup passed. Parent returned Fix 3 to the
  same Sol agent for an explicitly selective v3 target with more length
  headroom (180–240 characters, at most two central propositions). This is a
  revised contract, not a reroll of the old failed diagnostic. Receipts:
  `docs/patches/2026-09-25-parent-r6-fix3-live-v1.json` and
  `docs/patches/2026-09-25-parent-r6-fix3-failure-metadata.json`.
  The parent has now reviewed v3 and passed 776 affected integration tests plus
  62 diagnostic controls. Its new zero-call preflight passed with both clones
  and the original unchanged, no credentials/network, and complete cleanup.
  One fresh live invocation completed under package
  `5bdd5362aaea8e9feb0d5630e40860b93ab1f968bd6e994e5194cba6a89289de`.
  It restored all ten retained targets and all four controls (32 completions /
  32 HTTP attempts, 60,235 tokens, unknown cost), with no holds and clean
  integrity/preservation/cleanup. Parent semantic review found two predeclared
  attribution omissions: Noel's update attribution and the assistant's
  refresh/comparison proposal. Included facts, polarity, uncertainty and
  outcomes were correct; these were omissions, not fabricated success.
  Gate 3 remains unaccepted: returned to the same Sol agent for v4 generic
  selected-update/proposal provenance preservation, not a benchmark-specific
  instruction or a silent relaxation of the control criteria. Full receipts:
  `docs/patches/2026-09-25-parent-r6-fix3-live-v2.json` and
  `docs/patches/2026-09-25-parent-r6-fix3-semantic-review-v2.json`.
  Sol has frozen the v4 attribution revision. Only the prompt/version and
  historical-version proof admission changed in production source; the output
  cap, source-window construction, parser, retry limits and publication path
  did not. Parent is running the 14-file integration gate. The unchanged
  diagnostic/control worker passed 62 parent tests with a new exclusive host
  root (`r6-fix3-summary-v3`); the old diagnostics stay immutable. New package
  `b926b016c93316c6c62e24d500e2edbd1e9234b52cb54c57d523466855a41169`
  is being transferred but no paid test is admitted before the offline gates.
  Parent accepted Gate 3 after 782 affected tests and 62 diagnostic controls,
  a clean zero-call preflight, and one v4 live invocation: 10/10 retained
  targets and 4/4 controls recovered in 32 completions / 32 HTTP attempts
  (62,140 tokens, unknown cost), zero holds. Manual review of the unchanged
  criteria found the required attribution, uncertainty, proposal status and
  negative outcomes preserved. Original/non-summary state and all cleanup
  checks passed. One overview was 322 characters: above the soft prompt target,
  below the unchanged 500-character parser/storage limit; no truncation or cap
  relaxation. This is bounded effectiveness evidence, not universal fidelity.
  Receipts: `docs/patches/2026-09-25-parent-r6-fix3-live-v3.json` and
  `docs/patches/2026-09-25-parent-r6-fix3-semantic-review-v3.json`.
- Combined offline verification: started against a frozen 468-file candidate,
  with 7,307 collected cases and the prior four declared skips. It overlaps the
  remaining Fix 3 verification to save time, without starting any further fix.
  Manifest `c1f9dc072f408ee0a64d7ac6d688d96de0828a700309fdbde30a66dccb2a98bb`.
- Deployment and fresh end-to-end LME regression: not performed.

### Additional verification issue

The full-suite progress exposed one failure whose expected node is the digest
squeeze parity control. Parent independently reproduced a preexisting
wall-clock-sensitive assertion on both R5 and R6: same-second graph insertions
pass, but a one-second advance reverses the valid recency order and fails its
hardcoded list. Current and counterfactual renderers remain exactly equal, with
zero restored edges, in both cases. The full shard traceback confirmed it.
Keep the failed suite receipt. After accepting Fix 3, assign a fresh Sol agent
to make this fixture deterministic and add explicit same/cross-second parity
controls without altering production ordering. Parent must review/retest and
freeze an updated test inventory before claiming a clean full-suite gate.

The full first inventory has now finished: 7,281 passed, four declared skips,
five failures and sixteen setup errors from denied localhost binds, plus the
one clock assertion above (7,307 total). The shard traceback confirms the exact
clock-order failure reproduced by the parent. A credential-free, non-loopback-
blocked rerun of both localhost test files outside the OS sandbox passed all
23 cases; this was a test-execution permission correction, not a code patch.
All frozen source and test hashes remained unchanged. The next complete gate
must use the same runtime network guard with local loopback permission.

With Gate 3 accepted, a fresh `gpt-6-sol` agent now owns only the clock fixture
fix in the isolated candidate's `tests/test_digest_squeeze_probe.py`. Parent
will require deterministic same-second and cross-second exact-order parity,
review production-source hashes remain unchanged, and verify the patch before
freezing the final combined suite. No production ordering change is authorized.

Parent accepted Fix 4: reviewed the test-only patch, ran 142 probe/anchor/
aggregation tests, and independently exercised four synthetic cases with real
ordering versus deliberately removed recency. The new test passes both clock
cases with correct ordering and rejects the broken cross-second ordering even
when both consumers share that defect. The seven application hashes are
unchanged. Receipts: `docs/patches/2026-09-25-parent-r6-fix4-final.xml` and
`docs/patches/2026-09-25-parent-r6-fix4-negative-control.json`.
An independent read-only review of the seven cumulative application deltas
found no actionable defects; this supplements, not replaces, parent review.

Final inventory frozen at
`/private/tmp/hymem-r6-sequential-20260925.6dSIIq/frozen-r6-final`, manifest
`bd8d0f3a8fb40bd6b77e7ca6579c8e5e8ee78733bea2bb22df71bc4b2c12eaa2`:
468 files, 7,319 collected cases and the same four explicit skips. Full suite
is running in four credential-free shards with required localhost permission
and the runtime non-loopback audit guard. The old C1 receipts remain retained.
Regression helpers are being repinned locally to this final inventory; no
regression paid run is admitted until the final full suite has passed.

Final full suite passed: 7,315 passed, four declared skips, zero failures/errors.
Parent independently reconciled 7,319 unique JUnit cases, every shard receipt,
all exact skip identities, and the unchanged 468-file inventory. Receipt:
`docs/patches/2026-09-25-parent-r6-final-full-acceptance.json`.
The cumulative 16-file R5-specific patch was independently applied to a clean
R5 copy and reproduced the complete final tree byte-for-byte. Source remains
isolated; this patch must not be blindly applied to the dirty main checkout.
The final regression harness passed 54 parent controls and Afrodite's actual
stock startup probe with zero calls, no credentials/network, exact full-data
selection and clean shutdown. Parent admission is now being installed for one
fresh failed-question regression, never a resumed or rerolled old run.

## Final parent acceptance

All four targeted fixes were implemented by separate `gpt-6-sol` agents and
accepted by the parent before the next implementation began. Revisions to an
unaccepted fix stayed with its original Sol agent. The full final suite and
patch reconstruction passed as recorded above.

The one fresh stock LME regression on `gpt4_483dd43c` completed on its first
attempt with a correct answer. Its canary passed once, and all 265 extraction
chunks, including the originally failing chunk, were processed in seven dream
cycles. Final item indexing is healthy with no extraction failure rows or
quarantined sessions. The network-disabled, credential-free validator verified
the strict scored artifact, physical checkpoint and one-attempt history,
source/dataset pins, role accounting and supervisor cleanup. Both live and
validation containers exited zero with PID zero and no OOM.

Measured usage including the separately accounted canary: 865 completions /
865 HTTP attempts, 2,768,899 tokens, approximately 31.5 minutes. Dollar cost is
unavailable, not zero. The parent independently summed all role meters. A
separate read-only SQLite audit found integrity `ok`, zero foreign-key
violations, unchanged database/WAL/SHM hashes and the original failed chunk
processed. No production stores or services were changed.

Eleven sessions remain missing summaries due to `summary_output_cap`; there
are no malformed summaries. This is explicitly reported as
`success_with_summary_degradation`, not healthy summaries. The repaired
explicit summary-recovery operation passed its separate retained-case/control
test, but was not injected into this stock benchmark. A transient extraction
attempt cleared during execution; its reason was not captured and is not
inferred. No rerolls, resumed campaigns, altered denominators or hidden
degradation were used.

Receipts:

- `docs/patches/2026-09-25-parent-r6-final-regression-acceptance.json`
- `docs/patches/2026-09-25-parent-r6-final-store-audit.json`
- `docs/patches/2026-09-25-parent-r6-final-full-acceptance.json`
- `docs/patches/2026-09-25-lme-r6-sequential-fixes.patch`

Limits: this is one targeted regression, not a representative sample, a full
500-question readiness guarantee, or an officially comparable score. Changes
remain in the verified isolated candidate and baseline-specific patch. The
dirty main application checkout has divergent/missing preimages and was not
overwritten; integration must reconcile those changes instead of blindly
applying the patch. Production deployment has not been performed.
