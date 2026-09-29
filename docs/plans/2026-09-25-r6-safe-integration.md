# R6 safe integration — preserve existing work

## Authorization and boundaries

The user requested continuing the verified repairs, including paid tests where
necessary. Continue the separate-Sol implementation / parent-verification cycle.
The previous R6 full suite and paid regression remain valid for their exact
frozen tree, not for a new merged tree. No production deployment, restart,
production-memory transfer, or full-500 paid run is being inferred from a
local integration step.

Current main checkout: `/Users/attavanwestreenen/AGprojects/HyMem`, branch
`Beam-optimisation`, HEAD `5fb5ce491b254015684be2f3115a9d6b70b0c5a3`.
It contains substantial uncommitted application and experimental verifier work.
Do not reset, overwrite, delete, or silently disable it.

Verified R6 tree:
`/private/tmp/hymem-r6-sequential-20260925.6dSIIq/frozen-r6-final`.
Manifest SHA-256:
`bd8d0f3a8fb40bd6b77e7ca6579c8e5e8ee78733bea2bb22df71bc4b2c12eaa2`.

## Read-only diagnosis

Against the full R6 inventory, main has:

- Application/configuration: 183 identical, 14 divergent but clean HEAD files,
  27 divergent dirty files, seven missing files.
- Tests: 145 identical, 13 divergent clean files, 41 divergent dirty/untracked
  files, 29 missing files.
- Auxiliary assets: seven identical, one divergent dirty README, one missing.

For the 16-file R6 incremental patch alone, seven paths are absent and nine
diverge. It is not a valid whole-file replacement over main. A Sol read-only
review identified the following separate integration requirements:

1. The `strict_failure` protocol repair can be ported independently; both
   production hunks apply, but tests must target the current protocol schema.
2. Numeric-table partition hunks apply, but the current canary is v17 while
   R6 is v19. Historical v17/v18 policy admission, provider-truncation accounting
   and current/archive separation are prerequisites, not just a version bump.
3. Summary recovery requires the independent summary/item-frontier architecture,
   migrations 062/063 and missing state/recovery modules. Main retains the older
   experimental verifier pipeline. Combining both is an architectural change,
   not a small conflict resolution.
4. Main's committed clock fixture already includes stronger insertion-order,
   recency, cap and deliberate-order-regression controls. Retain it rather than
   replacing it with the R6 fixture.

The parent also found a local loaded-identity traversal optimization in
`hymem/extraction/producer.py`, with dedicated tests not present in R6. It must
be reviewed as a preservation candidate, not silently dropped or assumed safe.
R6 already retains the committed API lifecycle lock/cleanup, shared SQLite
statement-cache protection and producer-registry reentrancy repairs.

## Integration choice

The user selected the recommended clean R6 release checkout with "Proceed".
Use R6's verified independent-summary architecture and carry forward unrelated
current fixes, keeping the experimental checkout untouched. Do not resurrect
the older verifier pipeline. The new worktree starts at current HEAD so tracked
documentation and independent regression files absent from the frozen R6
inventory are retained. Overlay only the exact verified R6 inventory and the
already accepted stronger clock fixture before sequential preservation work.

## Sequential verification

1. In a disposable exact R6 copy, retain main's clock test unchanged. A fresh
   Sol agent performs this test-only integration; parent reviews the single-file
   change, verifies all application hashes remain R6, and reruns the controls.
2. After parent acceptance and the architecture choice, use a fresh Sol agent
   for each necessary preservation/integration change, including loaded-identity
   traversal if appropriate. Verify behavioral identity, resource bounds,
   mutation visibility and regression controls before the next change.
3. Freeze the complete chosen tree and its tests. Run focused and full offline
   gates without credentials, followed by target-runtime compatibility checks.
4. Use a new, bounded paid test only where an actual request/runtime behavior
   changed or offline coverage cannot settle the remaining risk. Preserve prior
   results, source identity, accounting and explicit summary warnings. Do not
   relabel the prior one-question test as a full-500 benchmark.

Disposable preparation directory:
`/private/tmp/hymem-r7-integration-20260925.vDa9yP`.
No source changes have yet been made to main or production.

## First preservation gate accepted

Fresh Sol agent `sol_preserve_clock_controls` prepared the disposable
`clock-proof/` tree and ran its focused tests. The parent independently repeated
all 21 tests: 21 passed, zero failures/errors/skips, 16.63 seconds, no provider
credentials or paid calls. Exact post-test inventory comparison confirmed all
468 paths present, all 231 application files unchanged, and precisely one test
file differing from R6. Its SHA-256 matches main's existing file:
`71a61b78ed684a1262ad34849a4a43d7f2fec9c8951569723d7dbdf17739e1f7`.
The other 467 files remain byte-identical to frozen R6. Parent receipt:
`docs/patches/2026-09-25-parent-r7-preserved-clock.xml`.

This accepts retaining existing clock tests, not a complete integration or
full-suite result for a merged tree. The architectural choice above is still
resolved in favor of the clean release route. Main application files and
production remain untouched; new paid calls so far: zero.

## Approved release preparation

New branch: `codex/r7-release-20260925`.
New worktree: `/private/tmp/hymem-r7-integration-20260925.vDa9yP/release`.
The four independent committed test files absent from the R6 snapshot must be
retained: `test_connection_lifecycle.py`, `test_locomo_judge_endpoint.py`,
`test_producer_gc_reentrancy.py`, and `test_shared_statement_cache.py`.
An independent read-only audit checks for other lost safeguards. The next
sequential Sol implementation will preserve the loaded-identity optimization;
parent acceptance is required before any further implementation.

### Identity preservation accepted

Sol integrated only `hymem/extraction/producer.py` and the existing independent
`tests/test_loaded_identity_traversal.py`. Parent reviewed the complete delta,
reran 186 focused tests (all passed), and independently compared 100 commitment
digests across R6 and the release using 48 mutable/aliased object graphs and two
module controls. All digests matched; mutations changed identity and restoration
restored it. The serializer's per-read memo remains bounded and does not cache
mutable state between calls. Receipt:
`docs/patches/2026-09-25-parent-r7-identity-acceptance.json`.

The audit identified two further application preservation units: direct CLI
checkout imports in `strictness.py`/`msc_registry.py`, then historical retired-
model archive admission in `msc_registry.py`. The first is now assigned to a
different Sol worker; the historical-model change must wait for parent CLI
acceptance. Independent current scheduler, lease, embedding concurrency, deadline,
peer-isolation, rehearsal and collection-order test safeguards will subsequently
be carried forward without reintroducing obsolete verifier fixtures.

### CLI preservation accepted

The separate Sol CLI change added only the checkout-root bootstraps to
`strictness.py` and `msc_registry.py`. Parent reviewed both exact diffs, reran
34 clean-interpreter tests (all passed), and independently reproduced the
R6 `ModuleNotFoundError: hymem` / release-success pair from an unrelated working
directory with site initialization disabled. No historical model policy changed
in this unit. Receipt: `docs/patches/2026-09-25-parent-r7-cli-acceptance.json`.
The next historical-model admission unit is now assigned to the other Sol worker.

### Historical archive admission accepted

Sol reproduced the retired-model archive rejection before changing the reader.
Parent reviewed the narrow MSC registry change and independently passed 225
archive, checkpoint and active-model-policy tests. Historical evidence remains
byte-preserved; new executions still reject retired aliases. One old tampering
fixture now uses an actually malformed model name instead of a valid retired
one. No endpoint, hash, canary or completion-policy validation was removed.
Receipt: `docs/patches/2026-09-25-parent-r7-historical-acceptance.json`.

The next separate Sol unit preserves existing scheduler, peer-isolation, lease,
embedding-concurrency, aggregation-budget and rehearsal regression safeguards,
without changing runtime code or reintroducing obsolete verifier fixtures.

### Causal concurrency and cleanup safeguards accepted

Parent reviewed all seven test-only deltas and independently passed 227 tests
in 110.52 seconds (zero failures/errors/skips; one dependency deprecation warning).
The controlled scheduler now tests actual pending-work, cooldown and cleanup
relationships; provider/embedding controls start their deadlock guards at the
relevant event instead of timing unrelated preflight work. Peer-isolation tests
retain real SQLite parsing and scorer decay. No runtime code changed.
Receipt: `docs/patches/2026-09-25-parent-r7-causal-safeguards.xml`.
The alternate Sol worker now carries forward the six remaining deadline,
collection-order, terminal-empty and extraction-identity test safeguards.

### Deadline and extraction-contract safeguards accepted

Parent reviewed the six-file test-only delta and independently passed all 397
cases in 86.76 seconds, with zero failures/errors/skips. Collection increased by
exactly 62 cases. Deadline controls distinguish before/at/after expiry, retain
real dreaming/status/attestation work, and still refuse late results. Terminal
empty controls preserve the full contract and atomic publication boundary;
the oversize refusal now targets R6's actual intact-input bound. The collection
subprocess disables pytest's cache so frozen snapshots remain unmodified.
Receipt: `docs/patches/2026-09-25-parent-r7-deadline-contract-safeguards.xml`.

The alternate Sol worker now updates only documentation, embedding CLI help and
its dedicated tests. The frozen/full/target gates follow parent acceptance.

### Documentation/help accepted; final freeze authorized

Parent reviewed the complete README delta against runtime source, independently
passed 58 guidance/default/recovery-CLI/startup tests, and proved the adapter AST
identical to R6 except exactly one help argument. The Sol gate passed 254 tests.
An independent reviewer approved the final descriptions and closed inventory:
231 source/config files, 239 Python test files and nine auxiliary files. Only
four source paths differ from R6 (the three accepted preservation paths plus
help-only adapter changes); only README differs among auxiliary inputs. All
541 preservation hashes in experimental main still match. No paid calls or
production changes were made during this integration.

Final README SHA-256: `60d0f5fc66de1a5f374447df3d8eb062ef79759d04437839ead0e74cc24a5066`.
Adapter SHA-256: `19ef4b5fa030d14c0c4e7e2c843ac9058bb09d19e4a704dbb84114b084f70f31`.
Parent receipt: `docs/patches/2026-09-25-parent-r7-guidance.xml`.
The freezer/target-runner isolation and negative controls independently passed
40 tests: `docs/patches/2026-09-25-parent-r7-diagnostic-controls.xml`.

### Final local suite and delivery patch accepted

Frozen manifest SHA-256:
`1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8`.
All four full-suite shards passed: 7,573 passed, four explicitly declared raw-SDK
skips, zero failures or errors. Parent independently reconciled every JUnit
testcase against the exact 7,577-node manifest, verified all 479 frozen/release
hashes after testing, and reconfirmed main's 541 preservation hashes. Receipt:
`docs/patches/2026-09-25-parent-r7-final-full-acceptance.json`.

The complete HEAD-to-release patch includes 87 modifications and 49 additions,
with no deletions or mode changes. A fresh HEAD archive plus this patch
reproduced all 533 repository files exactly: the 479 manifest files plus 54
unchanged HEAD extras. Forward application and reverse checks both passed.
Patch: `docs/patches/2026-09-25-lme-r7-release-from-head.patch`.
SHA-256: `0cea997edea0bfc80f5c38eb3368075a9d981491aca0dfa3c5c8c9cb03e3ca61`.
This is not a patch to apply blindly over dirty experimental main.

The initial Afrodite diagnostic upload timed out after 120 seconds. Read-only
checks proved the exact target directory absent and no installer process left.
A separate Sol fix added compressed, bounded SSH transfer and sanitized,
no-retry timeout handling. Parent independently passed 44 diagnostic controls.
The exact same archive then installed successfully. The isolated, credential-
free, network-disabled 1,823-case target gate is now running, followed by
startup, accepted producer identity and extraction-contract parity checks.
No production deployment/restart or new paid call has occurred.

### Final Afrodite gate accepted — integration complete

The isolated target run completed in approximately 23.4 minutes: 1,819 passed,
four declared raw-SDK skips, zero failures/errors. Parent reconciled all 1,823
JUnit cases against the exact manifest selection and downloaded only the three
allowlisted diagnostic receipts. Both R6 and R7 reached the real stock CLI's
pre-provider boundary, closed their clients/checkpoints, and produced the same
extraction-contract identity and the exact producer identity anchored in the
accepted R6 paid regression. All five supervised stages completed, reaped their
children and confirmed process-group cleanup; the container exited 0 with PID 0.
Source, diagnostic helpers and benchmark dataset were checked before and after.
No provider call was possible in the network-disabled container.

Final independent review also reconciled all local and target JUnit identities,
exact skip sets, receipt hashes, cleanup records, anchored producer/contract
parity, delivery patch, 479 frozen/release hashes and 541 main preservation
hashes. It found no blocker or unsupported completion claim.

Parent receipt:
`docs/patches/2026-09-25-parent-r7-final-target-acceptance.json`.
Final diagnostic-tool negative controls: 52 passed, zero failures/errors/skips
(`docs/patches/2026-09-25-parent-r7-final-diagnostic-controls.xml`).
The independent delivery review additionally verified all executable modes and
reverse-applied the patch back to the exact 484-file HEAD archive. Final parent
checks reconfirmed all 479 release/frozen hashes and main's 541 preservation
hashes. The release remains uncommitted in its separate worktree; the complete
patch is durably saved in the main repository's documentation directory.

### Evidence limits and promotion handoff

No additional paid replay is needed solely for this preservation integration:
the request/extraction pipeline and recorded producer/contract identities match
the already paid-tested R6 tree. That is evidence carry-forward, not a new live
R7 provider run or a full-500 readiness claim. The prior one-question stock run
completed correctly and indexed its items, but 11 sessions still disclosed
`summary_output_cap`; explicit summary recovery was tested separately and was
not secretly injected into the benchmark. Semantic completeness and benchmark
score improvement remain measurements for a subsequent representative/full run.

No production deployment, database migration, service restart or full-500 run
was performed in this integration. A later authorized deployment must preserve
each instance's local changes/configuration, rehearse schema migration on a
consistent store copy, and refresh installed package entry points: the new
`hymem-recover-summaries` console command is declared in `pyproject.toml` and is
not installed merely by copying Python source. Do not apply the delivery patch
blindly over the dirty experimental checkout.

### Authorized Hermes1 deployment and headless sample-eight follow-up

The user subsequently authorized deployment on Afrodite and explicitly selected
eight questions, not full-500. The frozen R7 source was rehearsed against a
consistent production copy: schema 61→63, all existing columns of 117 tables
unchanged, clean integrity/FKs, and successful reopen. Source-preservation review
confirmed the operator client extension and other maintained local changes.

Hermes1 was stopped at a confirmed idle boundary (graceful exit 143). A fresh
177,438,720-byte schema-61 backup was verified with the real R7 database runtime,
alongside source and installed-distribution/entrypoint backups. Database backup
SHA256: `ef47b37763ad46b288ce0fdff1b36500aa330985b7cb368f13cf964e9802b406`.
The exact 479-file release was installed and its editable package entrypoints
refreshed using pinned offline build wheels, without application dependency
updates. Live schema migration and reopen passed integrity/FK checks at v63.
Hermes2/3 and the shared embedding server were not restarted or changed.

Restart exposed a deployment configuration omission: existing MCP/Honcho
launchers still explicitly selected `deepseek-v4-flash`, which the release
correctly rejects for active execution. The initial container/hook starts did
not establish healthy Honcho. Root is aligning those explicit overrides to the
release's tested `deepseek-flash` request service, preserving credentials,
embedding settings and unrelated launcher bytes. Do not describe the initial
`start.json` as a successful service-health verification.

The independent eight-question preflight passed with zero provider calls:
398 sessions, 4,036 messages, fixed seed 0 and source indices
`[213,262,329,339,370,372,392,400]`. The live container was created but not yet
started at this checkpoint. Its identity is
`e73124b8ee6816275b5779052c094abb3fdb1722281f17a7a84069f7aed7f9bb`.
Private run root on Afrodite:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/r7-sample8-headless-v1`.
A separately reviewed one-shot detached finalizer is installed; it can run only
offline validation after this live container exits, never launch/resume paid work.
Paid-run and finalizer launch, healthy-service verification and benchmark outcome
must be recorded separately below when observed.

#### Deployment verified; sample-eight started headlessly

The model-only correction changed exactly one assignment each in the wrapper
and post-restart hook. The credential file stayed byte-identical. Both MCP and
Honcho now resolve `deepseek-flash`, with matching effective credentials and
embedding/request settings. Hermes1 was restarted and its established hook
run; Honcho and skill-search health returned 200. The other containers remained
running with unchanged start times.

A second genuine compatibility issue was isolated in the doctor: its old
one-token `ping` completion returns `finish_reason=length`, which the stricter
client correctly rejects. The exact live failure was `LLMOutputTruncatedError`;
actual MCP handshake/profile/retrieval and database checks already passed.
A separate Sol implementation changed only this diagnostic probe to request
`Reply with exactly OK.` with a 32-token budget. Parent independently passed
three positive/negative probe controls and four existing cleanup tests with
the candidate overlaid. An initial wrong-interpreter invocation lacked
`requests` and failed collection; the dependency-equipped rerun passed and is
the acceptance evidence. Truncation and transport failures remain failures.

The doctor-only production overlay is recorded separately from frozen R7:
`hymem/doctor.py` SHA256
`2ac786c6590aa7fbece726e6380981906d514fefe9f664a3ca4f7b1d1de7c15b`.
The portable patch is `docs/patches/2026-09-25-r7-doctor-probe-fix.patch`.
The running benchmark source, original experimental checkout and frozen R7
acceptance source were not changed by this diagnostic fix.

Final installed-service verification passed: doctor exit 0 (nine OK, zero FAIL,
two WARN), Honcho health, actual MCP handshake/profile/retrieval, all five
entrypoints, unchanged 86 non-HyMem distributions, 479 declared source hashes
with the explicit diagnostic-only override, schema 63, integrity/FKs, canonical
drift and lossless coverage. The two warnings remain `stored embedding
compatibility` and `summary context health`; no history was deleted or
diagnostic downgraded to manufacture an all-green result.

The authorized live container started at `2026-09-25T13:12:41.079401883Z`.
Its detached finalizer PID 3676344 was verified in a separate SSH connection
with parent PID 1 and session ID 3676344. It waits for this one owned live
container, then runs one network-disabled offline validation and saves
`finalizer-result.json`. It cannot launch another paid run or resume this one.
The benchmark itself has a nine-hour supervisor limit, per-question limits,
private persistent logs/checkpoint/stores and no automatic restart.

At handoff the first question was still indexing: 50 processed chunks,
96 episodes, zero observed chunk/session quarantines or instrumentation errors,
and two explicitly recorded summary-failure markers. This is progress, not
eight completed questions, a verified score, or a full-500 readiness result.
The server and Docker must remain running; the laptop is no longer required
for the benchmark or finalizer. The compact handoff receipt is
`docs/patches/2026-09-25-r7-deployment-headless-handoff.json`.
