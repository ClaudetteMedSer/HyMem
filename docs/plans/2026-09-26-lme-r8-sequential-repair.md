# LME R8: diagnose, repair separately, verify independently

## Current status — September 26, 21:26 UTC

The full suite completed at 19:57 UTC: **7,864 passed, 4 failed, 4 skipped**, no
errors. The final 503-file candidate remains frozen. The targeted parent tests
and fresh paid retained-case checks passed, but the combined acceptance gate
did not. The headless continuation correctly stopped with `operator_pending`,
zero paid runs started and no premature gate admission. No new sample-eight
paid run or production deployment occurred. The four failures involve the
bounded-summary recovery fixture and three terminal-empty extraction cases;
their cause is being independently diagnosed before any test or code change.
The user requested Git publication; the exact current candidate is published
on `codex/r8-current-version-20260926`, commit
`5f936f21aaf59294507954ecd7ae28adecedda02`, with incomplete readiness stated
explicitly. The remote branch tip was independently verified; `Beam-optimisation`
remains at `5fb5ce491b254015684be2f3115a9d6b70b0c5a3`.
Historical launch details remain in
`docs/patches/2026-09-26-r8-headless-continuation-receipt.json`; they describe the
state at launch, not a completed test pass.

## Evidence and scope

The September 26 v64 sample-eight run finished after 10,435 seconds with seven
completed questions and one `indexing_failure:quarantined_extraction` on
`e61a7584`. Its unchanged checkpoint SHA256 is
`356841543c3010c4a7a0724b5b3f4263562057d5d7c61175be23e2bbff286ea4`.
The eight stores have 82 degraded/missing session summaries. Recorded pipeline
usage is 6,289 completions, 20,154,131 tokens and 8,949.8 seconds of provider
latency. The original result is now independently artifact-validated (Gate 1
below); it is not a score for the new candidate.

The automatic finalizer failed before creating a validation container. Its
runtime inventory compares whole stat objects before/after reading a file;
its own access-time update makes that comparison fail despite unchanged bytes.

User authorizes necessary paid testing, including production-derived memories.
Use the retained benchmark evidence first; no production restart, data repair,
or deployment is necessary to diagnose these failures. Preserve old artifacts,
source identities, failed results and original databases. Do not hide failed
questions, weaken extraction/quarantine criteria, raise caps without evidence,
or present a changed protocol as a comparable canonical baseline.

## Sequential implementation and parent gates

Read-only diagnosis can run in parallel. Only one fix is implemented at a time;
a distinct Sol agent owns each fix and the parent reviews and tests it before
the next implementation starts.

1. **Finalization controller.** Ignore access-time-only changes during hashing,
   while retaining content/identity/metadata and replacement/symlink checks.
   Add deterministic positive and negative controls. Parent runs red/green
   regressions and relevant controller/finalizer tests. Then use the corrected,
   explicitly identified controller for one network-disabled validation of the
   unchanged old run. Preserve the old failed finalizer receipts; do not rerun
   paid work or overwrite its controller installation.
2. **Quarantined extraction.** Identify the exact durable failure and reproduce
   it on the retained source offline. Implement a generic correction in a new
   candidate copied from the actual frozen 481-file source, not the divergent
   dirty checkout. Verify exact source spans, provenance, bounded work and
   rejection behavior, with historical producer compatibility when versioned.
   Parent verifies the original failing case, neighboring structural controls
   and integration tests, then a bounded paid replay if needed.
3. **Summary degradation.** Census exact failure reasons and scheduling of
   existing bounded recovery. Distinguish rejected generation, exhausted
   recovery and absent scheduling before changing anything. Fix the demonstrated
   cause without making summaries authoritative, truncating assertions, or
   changing canonical benchmark settings silently. Parent verifies realistic
   cumulative histories, attribution/polarity, preservation, deadlines, call
   accounting and fresh retained-case paid recovery.
4. **Explicit recovery cap feedback (confirmed by live Fix 3 verification).**
   A fresh Sol agent adds one source-only bounded repair to explicit recovery,
   including distinct truthful cap feedback on a later one-call invocation.
   Each dispatch must consume the existing durable and invocation budgets;
   preserve source/job/lease fences, original cursors, hard 500-character cap,
   historical proofs and exhausted budgets. Do not transfer rejected drafts into
   prompts, slice output, or change the verified normal-ingestion policy.
   Parent verifies offline controls and the saved failures before a new paid run.
5. **Repair feasibility (confirmed by live Fix 4 verification).** A separate
   Sol implements a repair-only v7 response with three independently complete
   selective overviews at decreasing detail. Validate the entire envelope and
   all alternatives; only excess length can make an alternative ineligible.
   Select the first complete alternative within 500 code points without joining
   fragments or slicing assertions. Primary output grammar, normal ingestion,
   budgets, source-only inputs and all dispatch/publication fences stay unchanged.
   Historical v1–v6 proofs remain readable. Parent tests strict negative controls
   and compact-summary semantics, then verifies retained cases and invented
   controls using a fresh bounded paid package. No repeated rerolls of a failed
   package; conditional feasibility is not universal model compliance.
6. **Runtime/cost verification.** Attribute expensive calls to actual pipeline
   phases. Correct proven redundant work if found, as another separately
   reviewed fix. Otherwise report inherent cost and explicitly separate any
   optimization experiment from canonical ingestion benchmarking.

## Combined acceptance

Freeze the final source/test identities; run the affected offline suite and
full regression suite in the appropriate runtime. Use existing bounded
diagnostic infrastructure, with explicit invocation limits, timeouts, accounting
and cleanup, rather than unbounded rerolls. Rejected outputs remain evidence.
Then run a fresh versioned LME regression on the same predeclared eight
questions, preserving the earlier result and reporting all questions. Full-500
readiness is not established by an eight-question pass alone.

Report independently: artifact validity, extraction convergence, summary
health, source/DB integrity, answer accuracy, runtime/cost and scope limits.

## Progress

- Plan created; three Sol agents assigned (controller implementation,
  extraction read-only diagnosis, summary/cost read-only diagnosis).
- No paid calls, production changes or new benchmark runs in this repair yet.
- Fix 1 implemented by Sol and parent reviewed: nonblocking no-follow descriptor
  hashing checks stable identity before/after the read and final pathname.
  Parent affected suite: 210 passed, one explicit fixture-only skip. Two earlier
  test invocations used missing/wrong historical fixture paths; those setup
  errors were corrected with the test's pinned R5 source, without test changes.
  Real Afrodite status now passes against the unchanged sealed runtime. Parent
  created the previously absent offline validation container, preserving old
  finalizer evidence; its result is pending before Gate 1 acceptance.
- Gate 1 accepted: validation container
  `8928f84002af11127c721dac509f84b66e9193784f8adf4c59ad143d37d7df04`
  exited 1 with the correct `validated_sample8_has_failures` result, not a
  validation crash. Network none, no credentials, PID zero, no OOM. Original
  checkpoint hash unchanged; archive SHA256
  `635bcc5b21921d62e0d8e40b279c8b9043572a2abdabd9e86c8ef18e32b15ca3`.
  Physical checkpoint, strict scored artifact, process cleanup and all usage
  were validated: 7 completed / 1 failed, five correct out of eight (62.5% for
  this development sample only), 6,311 total completions / 6,312 HTTP attempts,
  20,236,876 tokens. No new provider calls. Controller SHA256
  `e98f9e444fafcdfe9ac8ca65ed80a52d9f7bfdf912ed03cc7d4c2411d0834dc5`.
  Fix 2 implementation is now assigned to its separate Sol agent in
  `/private/tmp/hymem-r8-sequential-20260926.MAx3h1/candidate`; the frozen source
  and dirty main application checkout remain unchanged.
- Extraction diagnosis: the quarantined 1,528-character chunk has two
  `predicate:not_allowed` item violations. Recovery subdivides/re-extracts but
  does not pass contract diagnostics back to the model. A neighboring chunk
  also failed missing top-level `complete` and `markers` fields. Exact offending
  responses were not retained by the original run; reproduce with captured
  requests/responses before claiming a semantic cause.
- Summary diagnosis: all 82 failures are `summary_output_cap` with failure count
  one; all eight stores have zero summary-recovery rows. The proven selective
  v4 recovery contract is explicit-API-only, while normal digest repair still
  demands every claim in 500 characters. More normal dream cycles cannot
  recover the already-published degraded summaries.
- Gate 2 accepted for the demonstrated failures. Parent independently proved
  old-source rejection/new-source recovery with diagnostic-sensitive controls;
  all 1,050 affected extraction/canary/protocol/registry tests pass. Parent
  review additionally blocked repaired-empty outputs in all roles so invalid
  omission items cannot disappear into partial success. Vocabulary, source
  limits, quarantine limits and total completion budgets are unchanged.
  The canary is now v20; historical v17–v19 policies remain literal.
- Paid baseline replay reproduced both retained failures (7 completions/HTTP
  attempts, 22,958 tokens). The isolated candidate completed both (5
  completions/HTTP attempts, 18,931 tokens, 11 and 7 triples). Exact source
  coverage, reference DB/WAL/SHM hashes, source hashes, client closure and
  supervisor cleanup all passed. Dollar cost was unavailable. This verifies
  these cases, not universal model reliability or full LME readiness.
  Baseline package `2ff00904d790c418b611287377d567e5601ee44283abd9ca8011f115a09dc734`;
  candidate package `a4803da180f0bab2baa2c66e4829f0bf31a6de33bd859490a35d48facd89347c`.
  Private captures remain on Afrodite for exact-response replay. An initial
  zero-call package was rejected for macOS transport sidecars and preserved;
  the source bytes were unchanged and the clean transfer passed admission.
- Fix 3 is now assigned to its separate Sol agent: share the selective overview
  policy across normal primary generation, one bounded cap repair and explicit
  recovery. Existing item/retention/frontier contracts remain required gates.
- Independent captured-response audits passed on both the sealed baseline and
  candidate, in network-disabled containers without credentials. Every request
  matched the recorded wire body; all responses were consumed, and exact
  private triples/markers/failure details and counters reproduced. No API calls
  were made. Capture inventories: baseline
  `234007030932ac0faa74e99ef20da9fdddf86dfb7320d66872153a4a3d07a92e`,
  candidate `daebcf920437a3e842f08cf97d7980296ec055f24f324ff51dd741bb0fe0c6ef`.
- Fix 3 implementation uses shared `summary-overview-v1` for both normal
  primary prompts, their single cap repair, and explicit recovery v5. Historical
  v1–v4 recovery proofs remain readable; no source/item/retention rules or hard
  500-character acceptance limit changed. The short overview is explicitly
  selective, not a claim-completeness proof. This changes memory behavior and
  must be reported as part of the candidate, not hidden as an identical prompt.
- Sol's 921-test affected gate passed. Parent's broader 1,235-test gate found
  1,233 passes and two obsolete assertions requiring the former 350-character
  soft target. Their source-byte, rejected-draft exclusion and hard-limit
  checks already pass. Sol is updating only those target assertions; parent
  will retest them. No new application change follows from these failures.
- Restored the prior verified 21-file migration/replay/embedding test overlay
  into the candidate, preserving the new summary tests. Full collection is
  7,799: the prior 7,764 plus 20 extraction repair cases, one extraction identity
  case, five historical summary cases, and nine overview cases. The first
  remote package safely stopped at collection because the expected count
  omitted that one identity case; zero full-suite/provider calls ran. A second
  offline run was stopped explicitly to incorporate the obsolete-assertion
  correction before a final fresh run. All receipts and snapshots are retained.
- Live summary diagnostic is being independently reviewed before launch. It
  replays all 14 cap-degraded sessions in the failed Q8 store through the normal
  digest using the actual benchmark's 12,000-character windows and separated
  summary mode, then explicit recovery on a backup clone and four invented
  semantic controls. Shared maximum: 192 completions / 576 HTTP attempts / 45
  minutes. Private request/response capture will permit offline diagnosis.
- Cost attribution: recorded Phase 1 extraction used 4,633 of 6,289 memory
  calls (73.7%). Remaining 1,656 calls lack reliable per-phase token/latency
  attribution. Provider latency was 85.8% of total elapsed time. Fresh ingestion
  per question is the canonical adapter workflow; no additional duplicate-work
  defect was established. Do not change memory tiers or benchmark settings to
  make this verification appear faster.
- Parent verified the sole stale-test correction independently: 73 focused
  tests pass. Final v5 candidate has 501 files, inventory SHA256
  `bcbd065655ecc58941126435f5c062eac3c50dcade93f2e9cb20c4988167331d`.
  Its full-suite-v3 run is underway against an immutable separate snapshot.
- Final v5 source's retained extraction replay again passed both chunks (5
  completions / 5 HTTP attempts, 18,930 tokens, 18 triples); exact-request
  offline replay also passed. Manifest
  `aaafd698076295dfb82f67245ab32bedb4f8e0052fef5b8815615af9ad185f27`;
  capture inventory
  `3ab82b67d7d9397b18ffa32e34db235cdd28b16513713afd197d73497f7915f7`.
- Live Fix 3 verification is **not a combined pass**: normal ingestion completed
  14/14 retained sessions; all four invented controls passed structural checks
  and parent semantic review. Explicit recovery published 10/14 and held four.
  Exact captured responses were complete single-key JSON strings of 509, 540,
  513 and 501 characters. All were honest `summary_output_cap` rejections,
  consuming one of three durable attempts; no source/item changes or cleanup
  failures occurred. The normal path has cap feedback, but explicit recovery
  repeats its original request. Fix 4 addresses that demonstrated gap.
  Capsule manifest
  `96796f127496558683413a434403da07af3ab08a7c7dbba62d19d0ab5f4e1775`;
  capture inventory
  `acd1a734f6f22fd39df6afb0f2a7a7acc04b68ebd987008fcc5b6b66f696a7a3`.
  Cost: 56 completions / 56 HTTP attempts, 167,829 tokens, 145.2 seconds elapsed.
  Original store and all non-summary clone state remain unchanged. Private
  bodies stay on Afrodite; only invented controls are semantically reviewed.
  The fresh eight-question run remains gated; no reroll or production deployment
  has occurred.
- Fix 4 implemented as recovery v6 by a fresh Sol agent and independently
  reviewed by parent and the harness Sol: one source-only cap repair, charged
  before dispatch, with truthful persisted feedback on later invocations.
  Parent additionally required known cap rejection to survive interruption of
  the repair call. Hard limits, source fences and exhausted budgets remain.
  Parent 183-test gate passes; old-v5 red controls reproduce both fixed gaps.
  The normal digest semantic identity is unchanged from the verified v5 policy.
- Final v6 source is frozen at 502 files, inventory SHA256
  `f6f7af369c076ca9c28e047070fd96c260b6f8d842820bec96094aae1144d1c0`.
  Actual collection is 7,817 tests. Offline full-suite-v4 is running on Afrodite
  with no network or credentials; package SHA256
  `fa8bd83daf57235deec7e21b176600d2f5dc2a13f6c10a4cfa2e551367aab1d2`.
  The obsolete v5 full-suite-v3 was stopped explicitly at 27% without observed
  failures; its evidence is retained, and is not a completed acceptance gate.
- Fresh v6 summary replay passed zero-call admission and is running under the
  same 192/576/45-minute bounds, package SHA256
  `cf341386e467e7879cbabaa8380b761c3e905ec674757d3990842dbf0f81e67c`.
  Fresh exact-final-source extraction package SHA256
  `04a4a62b587b0bbdea19dfee7fc51e32fb9a63eb91f8fc6cbf4c00d400611f30`
  is undergoing zero-call admission. All original failed runs remain preserved;
  no production deployment, restart or new sample-eight launch has occurred.
- v6 live summary replay remains **honestly degraded**: normal 14/14 and all
  four semantic controls pass, explicit recovery publishes 13/14. The remaining
  source window returned 505 characters, then 519 on the numeric-feedback repair;
  both are complete valid JSON. Eight other cap repairs succeeded. The remaining
  job preserves its previous valid partial cursor and has consumed two of three
  attempts. No invalid summary was published. This is a demonstrated remaining
  feasibility gap, not a schema, transport or source-integrity failure.
  Cost: 68 completions / 68 HTTP attempts, 194,958 tokens; cleanup passed.
  Captures SHA256
  `fd645967f1596857eeb1621d73bc82d4dd3349991c94a20fd6e35e6a3779b325`.
  A fresh Sol is diagnosing a structural repair-only output contract rather than
  repeating the same prompts or raising limits. The v6 sample-eight package is
  prepared but remains ineligible and unlaunched.
- Exact v6 extraction passed both retained chunks: 6 calls / 6 HTTP attempts,
  22,845 tokens, 18 triples. Parent's exact-request/private-result offline audit
  passed with no API calls and removed its container. Capture inventory
  `7e2784898cd15c15e371f3bb5d5d3e2b8243864406c8c9ceb87ffa1898c82ed3`.
- Parent stopped the now-unaccepted v6 full-suite-v4 container, preserving its
  partial evidence. The next full suite will wait for successful targeted paid
  verification, so it is not repeatedly restarted for newly discovered cases.
- Fix 5 implemented by its separate Sol as recovery v7. Repair-only strict
  three-option validation preserves all existing call/attempt/lease/source
  protections and normal-ingestion policy. Sol's 211-test gate passes. Parent's
  independent 68-test gate passes, including whole-option selection and invalid
  unused-option rejection. A concurrent broader parent run collected an obsolete
  negative fixture that treated now-current v7 as unsupported; only that fixture
  was updated to unknown v8, and all eight corrected negative cases pass.
  The final frozen parent gate is running again; no application change resulted
  from the stale test. Frozen source: 503 files, SHA256
  `e38a26fec4e5b756d3bca7a36418f78648b7542b3b53805012a12c3d0e8365c7`.
  A fresh paid diagnostic will additionally check the exact failed repair input
  and direct repair-contract semantic controls; old unaccepted capsules will
  not be resumed or relabeled.
- Parent's final frozen v7 application gate passes all 238 cases, no failures,
  errors or skips. Actual full collection is 7,872. Separate application/test
  patch artifacts and exact apply/reverse provenance are preserved under
  `docs/patches/2026-09-26-r8-v7-*`, explicitly pending paid and full-suite gates.
  The expanded diagnostic uses the actual three-option repair parser on four
  invented control walks plus one exact pinned prior failed request, with no
  fabricated primary calls and within the unchanged shared paid bounds.
- V7 targeted paid gate passed on its first fresh invocation: normal ingestion
  14/14, explicit recovery 14/14 (zero held/exhausted/degraded/malformed), four
  normal controls and four direct repair-control walks complete. The exact
  previously failed 035 input repaired in one call to 361 characters. Parent
  reviewed all invented alternatives and selected outputs: attribution,
  uncertainty, proposal status, update order and injection rejection preserved.
  Selective shorter options may omit entire propositions; this is not a proof
  that every retained source claim appears in a summary. Intermediate windows
  correctly retain the earlier target until its later update is consumed.
  Capsule `7a0cdf8b54c6dd8c9de4b1d76114bfa3732f342e740b92b357555e0f5f767f91`;
  capture `4b25765887f1e3b3ead990b7fdcc3d300bdbf43d6edb738680ded82865879e95`.
  Cost: 75 completions / 75 HTTP attempts, 210,927 tokens; clean exit and source,
  reference, previous-capture and non-summary clone preservation all verified.
- Exact v7 extraction passed both retained chunks in 5 calls / 5 HTTP attempts,
  18,937 tokens. Parent exact-request offline audit passed with no API calls.
  Capsule `03f35d1259cd016c8034338196369bd0eb00a25ac19a60c2a3a70194469ba9d2`;
  audited capture `cf9a9b84b95ea80bab26cea7351a2dd5f4761d42973f2491758ce7ecacc68fab`.
- Final full-suite-v5 is now running headlessly on Afrodite (7872 collected
  expected), after targeted paid acceptance. Package SHA256
  `3de8eb910250c2cf2beebc6141dcafbe5fac6ef294fed9485f7933d9044bb2c1`.
  The parent-reviewed sample-eight gate has 37 passing offline tests and now
  requires all new repair evidence plus full-suite count/skip checks. No new
  sample-eight, production deployment or restart has occurred yet.
- Fresh sample-eight preparation `lme-r8-sample8-headless-v2` is staged with
  source 503/e38a26..., the exact original eight IDs, seed/model/thinking/indexing
  limits, and sealed runtime. Its real CLI pre-provider startup probe passed:
  zero completions/HTTP attempts, no real credentials, exact producer identity,
  checkpoint handles/client closed, PID zero, no OOM. Preparation manifest
  `b0bd977120d1afd6c5c94da5fa701881e2f493e8937b9503961dd9a5fbba4d46`.
  The live launch gate has not been written because full-suite-v5 is pending;
  no paid sample-eight rerun has started.
- A separate one-shot continuation is being implemented and independently
  tested for the already-authorized headless workflow: wait for this exact suite,
  verify all sealed gates, launch at most this fixed eight-question run, then
  perform offline validation even when the benchmark fails. Dependency failures
  must stop paid dispatch. No rerolls, full-500 run, production memory mount,
  deployment or restart are authorized by this continuation. It has not launched.
- Continuation verification completed: parent and independent Sol each passed
  all 60 offline controls, including strict gate denial, exact-once ownership,
  ambiguous-start accounting and unsuccessful-run validation. Parent additionally
  verified the real Afrodite capsule/source/runtime pins, genuine zero-call
  preflight, exact owned suite container and capsule-root extraction audit.
  A mistaken local test-file path caused one collection invocation to exit
  before testing; the corrected unchanged 60-test invocation passed.
- At 18:40:54 UTC, parent launched the one-shot host continuation, PID 953508,
  start ticks `190718827`. A separate SSH observation verified matching process
  identity, its own detached session/process group, exclusive intent, empty
  stderr and no premature admission or paid launch. It is waiting on the exact
  full-suite-v5 container. Helper SHA256
  `b206aae4bd4f558413e848a05b0892656e03fb1cf7240fd8345051462c701cf6`;
  configuration SHA256
  `b82c4cd8997c9e435f5e9fe6623c409d5295856a2163b896757916f36eabd7d3`.
  This is a single staged workflow, not a recurring monitor, deployment, or
  permission to launch full-500. Closing the laptop does not interrupt it.
- Full-suite-v5 completed at 19:57:13 UTC after 5,805.6 seconds: 7,864 passed,
  four failed, four expected transport-scope skips, zero errors. Both the source
  snapshot and tested source remained unchanged; cleanup completed with PID zero
  and no OOM. JUnit SHA256
  `d444bdb23f042f1f4eef30634f8991ae4993c1cb742236d503531f6b71501d2f`.
  The continuation stopped on the actual failed suite, with no paid or validation
  run dispatched. Its original receipts remain intact.
- A separate Sol reproduced all four failures on the frozen candidate and
  diagnosed stale fixtures/assertions. `DreamLLM` lacks the repair-only v7
  alternatives response and its final assertion counts only primary recovery
  requests. Three terminal-empty cases predate bounded contract repair: one
  expects the old failure classification, while two scripted clients omit the
  newly required third reply and accidentally manufacture `call_failure`.
  In-memory diagnostic controls preserve atomic rejection of empty repair,
  exact-source roles, omission verification and summary publication/state reset.
  No test or application changes have been made for these findings yet.
- Next sequential test corrections: update the summary fixture and verify all
  existing publication/state/budget invariants; then update terminal-empty
  fixtures and explicitly assert empty-repair rejection, source preservation
  and actual call counts. Use separate Sol implementation steps and parent
  verification. A fresh `candidate-test-sync` copy exists for these corrections;
  the 503-file v7 original remains immutable. A full-suite rerun is still required.
- User-requested Git publication completed in a managed isolated worktree.
  Exact candidate versus base: 96 modified and 73 added files; all 54 other
  pre-existing tracked files retained. Two scoped source-manifest/status files
  added, for 171 committed paths. Separate read-only reviews found no private
  memory, provider captures, database files or real credentials in this payload.
  The candidate's two existing whitespace warnings were preserved to keep its
  tested bytes exact; no unrelated main-checkout edits or staged changes were
  included. Remote commit is `5f936f21aaf59294507954ecd7ae28adecedda02`.
- Publication review noted a separate follow-up concern:
  `benchmarks/lme_canary_watch.py` logs exception text without sanitization.
  No exposed secret was found in the published source; this is a runtime logging
  risk to investigate separately, not a diagnosis of the four suite failures.
