# Sequential verification fixes

Baseline: `c0c22e7a083b248752e609af3b8c3fef3245431a`.

Final status: **all 14 finding gates accepted** (13 code findings and one
concurrency-test attribution correction). The final uninterrupted suite passed
**4,887 tests, zero failures/errors/skips**, with three dependency deprecation
warnings, in **2,424.76s (40m24s)**. Compilation and diff checks are clean.
Changes remain uncommitted; no production deployment, real-store migration,
or paid benchmark was performed. Deployment/replay costs and remaining
architectural limitations are recorded below.

This plan addresses the eleven confirmed findings from the 7 September review,
plus the associated failing/flaky tests and newly reproduced integration bugs.
Each implementation is assigned to a
separate implementation agent. After the app refused fresh agents at finding 7,
the user approved reusing completed agents; the sequential review gates remain
unchanged. The primary reviewer inspects the diff, reruns the original
reproduction (with explicit fixed-behavior assertions where needed), and runs
targeted and neighboring regression tests before accepting it and starting the
next implementation. Failures return to the same agent for correction.

No paid benchmark, real memory-store migration, remote deployment, or restart is
part of these local implementation gates. Passing tests are evidence of the
covered behavior, not a guarantee that no undiscovered bug exists.

## Finding gates

Execution order: **1, 2, 11, 3, 4, 5, 6, 7, 12, 6 correction, 8, 9, 10, 13, 14**. After accepting 1 and
2, the hashing-overhead fix was moved forward to make the repeated, otherwise
offline verification cycles practical. This changes ordering only; each gate
must still be accepted before the next implementation starts.

1. **Extraction split context** — preserve conversational dependencies across
   initial and recovery splits without allowing context-only citations.
2. **BEAM embedding telemetry** — accept the valid remote backend/token schema
   while rejecting network-free token claims.
3. **Semantic generation identity** — bind digest, profile, and facts to the
   actual producer and implementation; rebuild safely from retained sources.
4. **Digest publication atomicity** — stage summary/procedure changes with the
   replacement digest and publish only a completed generation.
5. **Procedure replacement** — reconcile completed source generations,
   including empty results, without removing unrelated procedures.
6. **Unicode canonicalization** — make normalization stable and retain
   meaningful letters/marks; version affected publication semantics.
7. **Benchmark registry admission** — enforce completion receipts, checkpoint
   evidence, and mode/scoring consistency across the affected adapters.
8. **Atomic MCP capture** — validate full requests before mutation and commit
   accepted message batches atomically.
9. **Markdown publication** — prevent sidecars from exposing rolled-back
   database state and make failed file publication recoverable.
10. **Shared anchor ordering** — align capped selection between aggregation and
    state expansion; eliminate real-clock flakiness in the parity fixture.
11. **Identity overhead** — reduce redundant executable hashing within a safe
    frozen execution scope without weakening mutation/integrity detection;
    rerun the unmodified scheduler deadline and long-running tests.
12. **MSC database path** — the real live indexing receipt test exposed that
    `MSCAdapter.open` opens `root/hymem.sqlite` while MSC recall/continuation
    pass `root/m.sqlite` and material attestation reads that nonexistent file.
    Align actual storage and attestation without weakening either contract;
    verify the actual caller path through ingest, convergence, and publication.
13. **Exact replay receipt stability** — the complete suite exposed that an
    unchanged lower-generation replay updates `processed_chunks.processed_at`.
    Make identical publication idempotent across clock boundaries without
    suppressing genuine generation replacement or repairing missing authority.
14. **Concurrent ingestion gate** — the full suite measured 106 ms ingestion
    p95 against a 100 ms limit. Diagnose real writer-lock contention separately
    from scheduling/CPU overhead, then correct the cause or establish a
    deterministic lock-regression test. Do not simply relax the latency bound.

The stale health-report wording assertion is checked alongside capture/report
changes. Whitespace is corrected with the touched test file or final hygiene
gate; neither should mask functional failures.

## Per-fix evidence

| Finding | Status | Primary verification |
| --- | --- | --- |
| 1 | Accepted | Initial broad gate caught a dense-canary regression (73 failed / 320 passed), returned and corrected. Primary final combined gate: 439 passed in 154.26s; unchanged original reproducer now yields the claim in both cases; same-peer dense table/prose control yields both claims; diff hygiene clean. 25 permanent regressions added. XML: `/private/tmp/hymem-fix01-accepted-gate.xml`. |
| 2 | Accepted | Primary registry/embedding gate: 183 passed in 117.05s. Original valid-remote artifact now admitted. Agent red gate reproduced 6 failures, then 134 strict + 68 adjacent tests passed. Canonical backend spelling corrected; 19 permanent cases cover valid/malformed tokens and remote counters. XML: `/private/tmp/hymem-fix02-primary.xml`. |
| 3 | Accepted | Root retained-only producer-swap/reopen and independent memory-only in-flight mutation/recovery gates pass (including one-attempt quarantine). Root prior-fix gate: 280 passed; current-schema consumers: 330 passed, 1 explicitly deferred stale wording assertion (finding 8). Broad core: 265 passed / 1 fixture failure, corrected with one stable producer rather than relaxed validation; final portable gate: 16 passed. Final production correction gate: 159 passed in 216.15s. 27 permanent semantic regressions. XML: `/private/tmp/hymem-fix03-primary.xml`, `/private/tmp/hymem-fix03-consumers.xml`, `/private/tmp/hymem-fix03-final-corrections.xml`, `/private/tmp/hymem-fix03-portable-final.xml`. |
| 4 | Accepted | Independent replacement/reopen/failure/completion and forward-tail late-summary rollback gates pass on final source. Root consumers: 321 passed in 255.79s; broad digest/migration/retention gate: 237 passed / 1 mutating-spy fixture failure. Passive tracing preserves the no-destructive-window assertion; corrected episode gate: 26 passed. Final semantic/publication gate: 46 passed; final staging/health including orphan corruption: 35 passed in 123.22s. 20 permanent publication regressions. XML: `/private/tmp/hymem-fix04-primary.xml`, `/private/tmp/hymem-fix04-consumers.xml`, `/private/tmp/hymem-fix04-final-alias.xml`, `/private/tmp/hymem-fix04-episode-final.xml`, `/private/tmp/hymem-fix04-staging-health-final.xml`. |
| 5 | Accepted | Root pre-fix empty-replay and same-name manual overwrite reproduced. Frozen-source independent empty, real-run repartition/union/tail/manual-content, portable ownership/feedback, and prior late-rollback checks pass. Root core procedure/publication/migration/retention/semantic gate: 189 passed in 288.56s; portable/attestation/server/granularity gate: 196 passed in 205.66s. Agent dedicated 18 passed; earlier broad 261 passed / 1 zero-count fixture failure corrected and independently covered by root gate. XML: `/private/tmp/hymem-fix05-primary.xml`, `/private/tmp/hymem-fix05-consumers.xml`. |
| 6 | Accepted | Original mixed-script/double-normalization and Hindi/CJK-digit collisions reproduced. Independent curated + 25,000 deterministic mixed/general Unicode fixed-point probes pass; real fact extraction/persistence/Unicode-only search/portable import retains distinct keys. Root primary: 196 passed in 191.18s; query/portable/procedure/server consumers: 199 passed in 196.66s; prior extraction/identity/BEAM fixes: 173 passed in 28.53s. 40 permanent Unicode cases; agent 238 core / 258 query / 158 final identity passed. XML: `/private/tmp/hymem-fix06-primary.xml`, `/private/tmp/hymem-fix06-consumers.xml`, `/private/tmp/hymem-fix06-prior-fixes.xml`. |
| 7 | Accepted | First full benchmark gate caught real store-pointer decoration and a stale canary fixture (1,294 pass / 2 fail), returned and corrected; original cleanup integration test stays unchanged. Root final frozen benchmark-family gate: **1,326 passed in 320.86s** (`/private/tmp/hymem-fix07-primary-final.xml`). Independent rebound checkpoint/mode/indexing mutations, all-four executable inventories, actual serialized convergence failures, zero-seam MSC/LoCoMo simulation pipelines, and LoCoMo retry/partial-success failures with 1/2 workers all pass. Store pointers and closed failure envelopes now match actual prepare_indexing output; post-indexing LoCoMo errors preserve its one indexing outcome. Review also corrected skipped typing, raw failure codes, Phase-1 authority loss (LME v5), unavailable-code serialization, and receipt dependency inventories. Agent final correction gate: 204 passed in 94.39s. Actual live receipt testing separately exposed finding 12. |
| 8 | Accepted | Original malformed second item/retry left one then two committed messages. Capture now validates the whole envelope before bootstrap; one opt-in `log_messages(close_session=True)` transaction commits the accepted batch, coverage/FTS/peers and closure. Root independent malformed-item/retry/late-insert/close-failure snapshots stay unchanged; complete closed batch is visible before embedding and targeted dream, ordinary batch remains open. Root frozen MCP/health/message/peer/Honcho/retention/lossless/integration/hardening gate: **446 passed in 250.59s**, one existing Starlette/httpx warning (`/private/tmp/hymem-fix08-primary.xml`). Agent initial red: 33 failed / 5 passed; first green 146 passed; 41 permanent cases now include strict flags and cross-connection real embedding observations. Stale health wording corrected with stronger exact blocker/no-false-success assertions. |
| 9 | Accepted | Original second-file failure left USER.md containing a profile rolled back to DB0. DB consolidation now commits before Phase2 files and Phase3 MEMORY refresh; caller-owned transactions defer file publication, standalone calls commit then publish. Root fault/repair probe passes with committed profile preserved, every write outside TX, and zero additional LLM calls for repair. Root frozen sidecar/Markdown/profile/auxiliary/producer/lease/deadline/portable/MCP/retention gate: **443 passed in 293.65s** (`/private/tmp/hymem-fix09-primary.xml`). Agent 17 focused / 316 final neighbors passed. 10 permanent sidecar cases cover SQL rollback in both phases, direct callers, manual sections, late deadline/lease guards and recovery; v3/v4 policy replay tested. No schema change. |
| 10 | Accepted | Original deterministic tie repro differed at cap 1 and cap 50; unequal-clock control agreed. Shared total ordering now preserves normalized recency, canonical semantic tie-breaks, existing proof filters and caps, and binds loaded helpers/constants into aggregation identity. Root independent fixed assertions pass for reversed insertion, equal/unequal clocks and caps 0/1/2/4/50 (`/private/tmp/hymem_fix10_primary_order.py`). Agent focused 87 passed; broad 465 passed / 1 stale v14 fixture failure: a current v16 export was merely relabelled v14 while retaining its newer manifest keys. Root verified strict importer rejection is correct and approved only a genuine-v14 test projection; all original invalidation/audit assertions remain, entire affected file 75 passed. Root frozen anchor/aggregation/retrieval/valid-time/generation/performance/scheduler gate: **498 passed in 291.65s**, one existing Starlette/httpx warning (`/private/tmp/hymem-fix10-primary.xml`). 25 permanent ordering cases added. |
| 11 | Accepted | Baseline: 20 identities = 1.6735s; empty dream 1.155s, one-source 3.299/3.326s (profiled). Primary post-fix: 20 identities = 0.5349s; empty dream 0.397s, one-source 1.012/1.026s. Independent mutable-state/code/closure/default/cache-poisoning checks pass. Broad primary gate: 380 passed in 262.36s; prior-fix gate: 159 passed in 29.06s. Original scheduler deadline passes unchanged. XML: `/private/tmp/hymem-fix11-primary.xml`, `/private/tmp/hymem-fix11-prior-fix-regressions.xml`. |
| 12 | Accepted | Root actual recall and recurrence calls reproduced missing `m.sqlite`, then both pass through real local convergence, retrieval and store publication with offline providers. Both callers now use `hymem.sqlite`; wrong paths fail before resources and post-open swaps fail attestation. 14 permanent cases cover live/skipped modes, retention/cleanup, wrong existing files, and real LoCoMo reuse. Root broad gate: 377 passed / 15 old attestation-fixture failures in 158.93s (`/private/tmp/hymem-fix12-primary.xml`). All 15 share the malformed `phase1-generation-v1:test` placeholder now rejected by issue 7; only that fixture value was corrected, not validation or assertions. Root final entire attestation + path gate: **62 passed in 59.48s** (`/private/tmp/hymem-fix12-primary-corrections.xml`). Agent 14 dedicated / 48 final attestation passed. |

## Final gate

Additional full-suite correction gates:

- **13 — Accepted.** Root deterministic clock-advanced repro failed only
  on the processed acknowledgement timestamp, then passed on the frozen fix
  (`/private/tmp/hymem_fix13_primary_replay.py`). Exact matching acknowledgements
  now remain untouched; missing/NULL acknowledgements repair, real generation
  replacement and staged-source repair still refresh completion timestamps,
  and auxiliary publications keep their own clocks. Original full dump/export
  assertions are retained and strengthened to cross a full SQL-clock day.
  Agent focused 9 and neighbors 436 passed. Root frozen claim/auxiliary/producer/
  portable/retry gate **443 passed in 207.92s**
  (`/private/tmp/hymem-fix13-primary.xml`); 7 permanent cases added and two
  original regressions strengthened. Compile/diff check clean.
- **14 — Accepted; test-attribution correction.** The old test passed
  alone (1 passed in 3.52s). Root instrumented five original workloads:
  ingestion p95 49.2–90.4 ms, ingestion thread CPU p95 7.0–7.5 ms, and
  BEGIN-to-first-statement p95 0.9–8.4 ms; none of 30 LLM calls entered a
  transaction. Normal dream transactions still held the writer lock for up to
  approximately 85–90 ms. End-to-end p95 cannot isolate provider-held lock
  duration. Only the test changed: synchronized real phase1/digest calls,
  independent fail-fast SQLite writer, third-connection batch/closure/coverage/
  FTS/embedding visibility, deliberately held-lock negative controls with
  retry, and assertion-failure cleanup. Production and general three-thread
  integration test stay unchanged. Agent deterministic 5 and neighbors 64
  passed. Root independent positive/negative probes pass for both real
  provider scopes (`/private/tmp/hymem_fix14_primary_overlap.py`); root frozen
  concurrency/lease/deadline/performance/capture/replay gate **122 passed in
  68.38s** (`/private/tmp/hymem-fix14-primary.xml`). Five deterministic cases
  replace one timing-attribution case; general concurrency test unchanged.
  This is not a production throughput optimization or a sub-100ms SLA.

- Preflight full collection after issue 7 exposed a missed issue-6 consumer:
  `benchmarks/fact_probe.py` imports the removed `_FTS_SAFE`. This blocks
  collection of `test_fact_probe.py` and `test_model_policy.py` (4,686 other
  tests collected). The earlier targeted Unicode gates passed, but this is an
  introduced regression and reopened issue 6's consumer gate. The separate
  correction cycle after issue 12 is now **accepted**: the probe imports and
  calls the maintained Unicode sanitizer; no obsolete ASCII alias was restored.
  21 permanent Unicode/operator/parity cases added. Root independent real
  SQLite import/search gate passes; root consumer suite **218 passed in 35.75s**
  (`/private/tmp/hymem-fix06-consumer-primary.xml`). At that preflight, a fresh
  process collected **4,799 tests across 158 files** without errors. That was
  collection only; the final full execution below also includes later cases.
- Ancillary current-checkout check: 82 model-policy, OpenAI-compatible-client,
  and provider-attempt regressions passed in 2.58s. Default dream budget is 50;
  deprecated aliases are rejected for execution and historical provenance is
  retained. XML: `/private/tmp/hymem-fixes-model-policy-check.xml`.
- First frozen, uninterrupted full suite completed: **4,874 passed / 2 failed /
  3 dependency deprecation warnings in 2,318.07s**. XML:
  `/private/tmp/hymem-fixes-full.xml`. Localhost embedding/SDK tests executed
  with explicit socket access; no paid run or production store was involved.
  The two failures are `test_claim_provenance.py::test_lower_generation_exact_replay_does_not_churn_restored_revision`
  (real processed timestamp churn) and
  `test_concurrency.py::test_ingestion_not_blocked_by_in_flight_dream`
  (p95 106 ms, cause under investigation). These extend the sequential cycle
  as findings 13 and 14; that first full gate was not accepted.
- **Final gate accepted.** After both additional corrections passed independent
  review, all agent/root test processes finished before the final frozen run.
  The same complete suite ran without interruption: **4,887 passed, 0 failed,
  0 errors, 0 skipped, 3 warnings in 2,424.76s**. JUnit counters and absence of
  failure/error elements independently checked:
  `/private/tmp/hymem-fixes-full-final.xml`. All newly added cases are included;
  no test was skipped or xfailed to obtain this result. The three warnings are
  existing Starlette/httpx and uvicorn/websockets dependency deprecations.
  Command: `python -m pytest -o addopts='' -q --durations=20 --junitxml=/private/tmp/hymem-fixes-full-final.xml`.
- `python -m compileall -q hymem benchmarks tests` and `git diff --check` pass.
  Final worktree inspection found only the intended source, tests, migrations,
  and documentation. HEAD remains the baseline above; no commits were made.
- These are offline/local integration results, not live extraction quality or
  LME/BEAM/LoCoMo competitiveness measurements. A representative live-model
  extraction canary and explicit rollout-cost review remain necessary before
  a paid definitive benchmark or production deployment.

## Deployment consequences

- Fix 1 bumps extraction's split contract to v10 so prior false-success results
  do not suppress corrected extraction.
- Fix 3 adds effective memory-producer and loaded-implementation commitments to
  digest/profile/fact generations. Historical config-only generations remain
  readable but require a one-time replay from retained sources; unknown custom
  memory routers conservatively replay after client replacement/restart. Health
  schema v7 distinguishes this stronger guarantee from old clean receipts.
  No real store has been replayed here.
- Fix 4 introduces schema v58 digest staging. Unfinished walks without valid
  staging replay safely on upgrade/import. This prevents future partial
  publication; it cannot reconstruct overwritten pre-upgrade summary/procedure
  history or undo partial content already delivered by older code.
- Fix 5 introduces schema v59 exact-content procedure ownership and portable
  v16 ownership declarations. Complete replays retire omitted owned rows via
  the existing stale lifecycle, retaining feedback until normal retention.
  Pre-v59 procedures, legacy imports without ownership declarations, and
  manual content edits remain protected/unknown-origin and need explicit
  review if obsolete; generated-looking names or ids do not prove ownership.
- Fix 6 versions script-preserving canonicalization, binds the actual Phase-1
  consumers and Unicode runtime, and advances auxiliary publication to v3 while
  retaining historical v0-v2 validation. Source-backed generations replay;
  no historical aliases are guessed, rekeyed, or merged. Lost distinctions need
  original retained sources. Full-text Unicode queries are preserved, but
  SQLite's existing vowel-mark candidate imprecision and unsegmented-language
  word boundaries are not redesigned. Exact entity identities remain distinct.
- Fix 7 adds independently reconcilable checkpoint-attestation v1 and preserves
  exact Phase-1 authority in LME indexing-summary v5. Old opaque checkpoint
  receipts and normalized v4 health summaries no longer qualify for current
  strict admission; historical evidence must not be upgraded by guessing.
  Public hashes establish consistency, not authenticity. MSC/LoCoMo fresh store
  pointers bind the archived indexing; reused pointers refer to the original
  unavailable build receipt and remain explicitly limited to lifecycle/syntax.
- Fix 8 provides capture-batch atomicity, not a global idempotency key. Invalid
  payloads and SQL/closure failures persist no partial accepted batch. An
  unexpected dream failure happens after a deliberately committed capture;
  resubmitting an already successful capture can still append it again.
- Fix 9 moves file publication after DB commit and advances marker-profile
  materialization policy to v5. Retained old-policy markers replay into current
  per-marker decisions; this is replacement, not append-only history. Ordinary
  dreams repair stale/missing sidecars without re-ingestion. Atomic replacement
  is per file only; lease/deadline checks are cooperative, not a SQLite/file
  atomic fence. Crashes can leave stale files or temporary files, while native
  profile reads remain database-authoritative. Manual sections are preserved.
- Fix 10 shares a deterministic KG anchor ordering across retrieval and digest
  selection, retaining each consumer's authority filters and budgets. Loaded
  ordering/clock helpers and constants participate in aggregation identity, so
  affected stored aggregates must rebuild rather than reuse old-policy output.
  This adds no schema migration or query-time LLM call.
- Fix 11 changes expanded code-body commitments into versioned immutable code
  digests. This deliberately changes executable identities for shared
  consumers, potentially rebuilding affected derived material and vector
  indexes. No production rebuild, store migration, or paid run has been
  triggered here. Retained-source coverage and rollout cost must be considered
  before deploying into existing benchmark/production stores.

The review's separate architectural limitations (endpoint-route privacy,
on-disk benchmark code attestations, indivisible oversized source coverage, and
profile omission semantics, plus measured ordinary writer-transaction latency)
remain explicitly tracked rather than silently
claimed solved by these fix gates. Any overlap uncovered during a gate must
be recorded and verified in that gate.
