# Hermes1 verification fixes — September 11, 2026

User approved implementation, independent verification and deployment after the
update exposed three issues. This supplements the update report; the interrupted
suite there remains an incomplete run, not a passing baseline.

## Current status

**Hermes1 deployment and live verification completed.** Restarted at
**19:25:02 UTC on September 11, 2026**. The complete corrected suite passed
**5,965 tests, zero failures/errors/skips**, and the separate installed-Hermes
contract receipt contains **46 passing cases**. Ten post-restart checks passed,
covering current-boot services/platforms, real model and embedding probes,
actual MCP reads, installed-SDK session reload/context behavior, private-clone
capture, final code/configuration pins and store integrity.

All **585 production coverage proofs** remain valid; canonical drift and FK
violations are zero. The one historical embedding-compatibility warning remains
explicit and unchanged. No production benchmark, dream/drain, vector rebuild,
historical rebinding or proof rewrite was performed. Hermes2/Hermes3/shared
embeddings were left unchanged. Reviewed source fixes remain uncommitted/unpushed.
Recovery backups and the complete verification archive are retained. Only the
19 owned, exited test containers were removed, after their receipts were
preserved. Final read-only service health remained green at **19:44:28 UTC**.

The chronological sections below preserve intermediate failures and rejected
runs; they are not the final acceptance result.

## Sequence and acceptance gates

1. **Shared connection initialization race** — one implementation agent; root
   reviews connection publication, failure cleanup, locking and scope behavior,
   then independently runs concurrent and adjacent regressions. No assertion
   that this proves the cause of the earlier unexplained Honcho stall.
2. **LoCoMo rejudge compatibility** — a new agent after gate 1; preserve Hermes1's
   custom judge endpoint and canary override, support legacy direct callers,
   verify actual endpoint selection and artifact provenance together.
3. **Scheduler timing tests** — a new agent after gate 2; test cooldown and kick
   semantics deterministically while retaining real-pipeline integration
   coverage. Do not count retries or relaxed verdicts as proof of correctness.
4. **Python shared-statement cache compatibility** — independently discovered
   during root verification, handled by a fourth new agent after gate 3.
   A stdlib-only 16-thread test on local Python 3.13.5 / SQLite 3.45.3 produced
   869/1,600 wrong rows with cache 128 and zero with cache 0. Target Python
   3.11.2 / SQLite 3.53.4 produced zero in both arms. This matches the open
   [CPython issue 118172](https://github.com/python/cpython/issues/118172).
   Verify a scoped mitigation and its overhead; do not attribute it to the
   previous Hermes1 outage or claim the initialization patch introduced it.
5. **Peer-provenance scoring test clock** — the first complete gate exposed a
   fifth issue; assign a new agent, independently distinguish real recency decay
   from foreign-evidence leakage, and fix the test without weakening isolation
   or changing production scoring. Check neighboring timing-dependent tests.
6. **Benchmark durable-status test budget** — the complete rerun exposed another
   incidental wall-clock assertion. Assign a sixth agent; preserve production
   timeout precedence and test the semantic/budget boundary deterministically.
7. **Producer-registry GC reentrancy deadlock** — target verification exposed a
   separate real runtime deadlock. Assign a seventh agent, reproduce against
   the unchanged former lock, verify cleanup and identity authority invariants,
   then repeat the target gates and store rehearsal.
8. **Scheduler unit-test synchronization and missing failure injection** — the
   next complete run exposed two more cold-start five-second assertions in a
   separate unit module. Assign an eighth agent, retain real-pipeline integration
   coverage, and verify actual exception recovery and deterministic scheduling.
9. **Private stack-dump diagnostics** — after a full-run worker crash during an
   asynchronous traceback dump, assign a ninth agent to supply normal Python-
   thread diagnostics. Independently verify them without changing assertions,
   application source, runtime versions or fatal error handling. Keep the
   original native crash cause explicitly uncertain.
10. **Digest squeeze-probe timestamp parity** — a tenth agent fixes the fixture
    after the next complete run exposes a false ordering assertion. Preserve
    production recency precedence and exact ordered current/fixed parity;
    independently reproduce from the actual failed synthetic fixture.
11. Freeze the exact reconciled Hermes1 candidate; run the complete HyMem suite
   in a no-network container without production data. Run Hermes-only contract
   tests separately in the appropriate environment.
12. Rehearse normal initialization on a consistent production-store copy; require
   no schema/data/coverage/canonical regression. Back up current source and data,
   verify idle state and protected configuration, deploy only reviewed files,
   restart Hermes1 alone, verify each applicable service and actual SDK/MCP paths.

## Boundaries

Preserve credentials, embedding identity, session-retirement mappings, local
customizations, historical evidence, and recovery archives. Do not launch scored
benchmarks or a production dream/drain, mutate historical proofs, commit/push,
or restart Hermes2/Hermes3/shared embeddings as part of these fixes.

## Results

### Gate 1 — passed

The implementation agent added 13 lifecycle regressions and passed 88 focused
tests. Root independently reviewed the publication/lock/cleanup behavior,
requested preservation of primary exceptions on cleanup failure, and passed
45 local lifecycle/bootstrap/startup tests. A separate root-authored paused
initializer and 16-consumer probe passed on the target runtime. The isolated
server gate passed **123 tests, zero failures/errors/skips**, with unchanged
candidate hashes during the run. Ordinary borrowed-handle SQL concurrency is
not serialized; close/set_llm still require hosts to quiesce existing users.

### Gate 2 — passed

The endpoint compatibility fix also corrects custom judge provenance in strict
run identities and bare rows. Root reviewed the actual-client/configured-endpoint
agreement check and preserved the server's separate canary override verbatim.
Root's final independent local gate passed **175 tests**; the reconciled server
candidate passed **227 tests, zero failures/errors/skips**, with unchanged source
hashes. The first local run recorded one new test-harness mismatch: the test
expected its structural exception to escape unchanged, while the established
checkpoint layer wraps that class. The corrected test uses an external-abort
sentinel and verifies client cleanup; the initial failure remains in its XML.

### Gate 3 — passed

The scheduler implementation is unchanged. Tests now park the real worker before
its real dream call, preserve actual pipeline clocks, and explicitly coordinate
in-flight kicks, an expired window, and coalescing. Negative controls prove the
oracle rejects bypassed cooldown and lost pending work. Completed real dreams
must process chunks with zero extraction-adjacent failure counters and preserve
captured messages/proof rows exactly. Root reviewed the final file and passed
**76 HTTP/scheduler module tests** independently. The same final target-runtime
module gate passed **76 tests, zero failures/errors/skips** with unchanged source
hashes. One early implementation-test failure was a stub routing collision
with the digest prompt, corrected by putting exact routes before broad needles.

The unchanged installed Hermes SDK retirement contract separately passed
**46 tests**, with network-blocked fake sessions and no production messages.

### Gate 4 — passed

Shared connection construction disables Python's statement cache on 3.12 and
newer until a verified safe upstream version boundary exists; 3.11 keeps 128
cached statements. Private snapshots/schema references retain their caches.
Root reviewed that scope and independently passed **38 cache/lifecycle/startup
tests**. The separate root-authored 16-consumer probe, previously failing on
Python 3.13, now passes locally as well as on 3.11. The implementation agent's
broader local gate passed **48 tests**, including concurrent ingestion/dream/read
and DB shadow checks. Its synthetic 3.13 comparison found old-cache wrong rows
and errors versus zero failures in 3,200 candidate queries. A tiny prepare-heavy
serial lookup measured +0.84 microseconds/query (1.59x); that is not representative
application or benchmark performance. Production 3.11's cache policy is unchanged.
The independent target-runtime cache/lifecycle/schema/concurrency gate passed
**44 tests, zero failures/errors/skips**, with unchanged source hashes.

### Final candidate

Frozen source manifest covers **481 paths** (including five already-absent tracked
egg-info files). Seven changed/added files: API lifecycle, core connection policy,
LoCoMo endpoint compatibility, and four regression-test files. Runtime custom
client/other-adapter overrides and protected deployment configuration remain
untouched. The complete no-network, four-worker suite is now running against the
frozen candidate. A real-store-copy startup/reopen rehearsal is running separately.
Neither gate has yet been recorded as passed; no production patch is deployed.

The real-store rehearsal has since completed successfully: **131 tables exactly
unchanged**, schema **61 → 61**, reopen unchanged, integrity OK, foreign keys zero,
canonical status OK, **585/585 lossless proofs valid**, no missing proofs/invalid
frontiers/recorded failures. The complete suite is still running. No live data
was written by the rehearsal.

### First complete suite — failed; preserved, not accepted

The first complete isolated run finished in **3,544.83 seconds**: **5,943 passed,
one failed, zero errors/skips**, with all candidate source bytes unchanged.
`test_scoped_graph_uses_only_local_authoritative_evidence` compared scores of
0.7499996414874606 and 0.7499984270793115 with relative tolerance 1e-6. Root
inspection found the scoped ranker reads SQLite's current Julian day and applies
normal exponential recency decay. The ratio corresponds to **4.197 seconds**;
the old tolerance implicitly allowed only about **2.592 seconds**. Production
remains unmodified, and the failed report and original manifest are retained.

### Gate 5 — passed

A fifth implementation agent changed only `tests/test_peer_provenance.py`:
controlled read-side SQLite time, real delegated date parsing, a 30-day simulated
decay check, exact complete-fact isolation, and a broader-scope negative control
that changes metrics, scores, and citations. Two neighboring score-comparison
tests now use the same controlled clock. Writer clocks and production scoring
remain unchanged. Agent and root each independently passed **all 73 provenance
tests** locally. Root's separate transparent-query-clock probe proved that ten
seconds alone breaks the original tolerance, foreign additions produce exactly
the same score at fixed time, and broader scope genuinely changes the result.
The independent target-runtime module gate passed **73 tests, zero failures,
errors or skips**, in **140.90 seconds**, with unchanged candidate bytes. Root's
separate probe also passed on that runtime. A new complete suite is now running
with six workers on the eight-CPU host; only the test fixture storage and worker
count differ from the initial run, not the runtime or provider configuration.

The candidate now changes eight files; only the provenance test differs from the
first full-run candidate. Runtime bytes are still identical to those used for
the successful real-store rehearsal. Completed failed-run fixtures were preserved
on disk rather than left in `/tmp` tmpfs; the final full run will also keep its
fixtures on disk to avoid consuming several GiB of live-service RAM. The first
Python copy attempt could not preserve an intentional FIFO test fixture and left
the original intact; the successful move used GNU `mv`, which preserves FIFOs.

### Second full attempt — interrupted after a new failure; not accepted

The six-worker rerun exposed an early failure in
`test_lme_durable_status_fails_on_only_current_fact_quarantine`: expected
`quarantined_extraction`, got `timeout_after_cycle`. Root gracefully interrupted
only the isolated pytest master to obtain diagnostics promptly. **1,196 passed,
one failed, zero errors/skips** were reported before interruption (exit 2,
565.32 seconds, unchanged candidate bytes). This is an incomplete run, never a
passing full baseline. Its manifest, report, and disk-backed fixtures are retained.

Root confirmed the actual contract: the absolute indexing deadline takes
precedence over semantic quarantine, including a deadline exhausted during the
durable-status scan. A real ten-second test deadline made classification depend
on host load. Changing production error precedence would be incorrect.

### Gate 6 — target verification in progress

The sixth implementation agent changed only
`tests/test_benchmark_adapter_strictness.py`. Its wrapper uses convergence's
existing `_clock` seam, retaining the real dream, propagated deadline, and real
status scans. Just-before/exact/after-expiry cases explicitly distinguish
quarantine from timeout and reject expired status publication. The neighboring
poison-embedding test keeps its real deadline-aware path and deterministic
`cycle_exception` check. No global clock or runtime change was made. Agent and
root each independently passed **114 adapter-strictness/indexing-deadline tests**;
the target-runtime gate is running. The candidate now changes nine files, and
all runtime bytes remain identical to the successful store rehearsal.

### Test-infrastructure isolation correction

Root found that the original broad read-only `/staging` mount made the private
rehearsal database copy incidentally reachable from earlier test containers.
The live database/home were never mounted, and networking was disabled, but the
old receipt's `production_store_mounted=false` does **not** establish that no
production-data copy was reachable. Those original receipts are preserved as-is
with this correction. New gates mount only the runner file, candidate, test
dependencies, and test output/fixture paths, excluding rehearsal copies. A
private expected-failure control verifies both inaccessible store paths and
immediate failure diagnostics. Future full-run failures can be diagnosed without
waiting for the final pytest summary.

### Target stall — actual runtime defect, not clock-fixture leakage

The 114-case target gate stalled after 92 tests with zero CPU activity. Its
producer source was still the original version, SHA256
`28b4b25fe256e291d2497d07c0a19f509368afc665adc7e470b447bde74b9d45`.
SIGINT sent only to the isolated pytest master was swallowed inside a weakref
callback, unblocking the process. Pytest then reported 114 passing cases with a
`PytestUnraisableExceptionWarning` containing `KeyboardInterrupt` in
`_drop_producer_proxy`. **That intervention-assisted exit 0 is not accepted as
a clean pass.** Its receipt is retained as rejected diagnostic evidence.

A second isolated diagnostic run reproduced the stall naturally. Its automatic
thread dump showed `_drop_producer_proxy` trying to re-acquire `_unknown_lock`
from a GC callback triggered inside `inspect.getattr_static`, while
`_phase1_proxy_source` already held the non-reentrant lock. The actual caller was
`DeadlineBoundLLMClient.complete` in the mid-cycle deadline regression. Root
terminated only that synthetic test master after preserving the stack. No live
service was signaled. Independent bounded target probes reproduced the old-lock
deadlock and passed with an injected reentrant-lock control. A same-order local
audit confirmed the test clock fixture restored all bindings/clocks/deadline
contexts and left no background threads. This defect predates the test changes;
the older production Honcho stall remains causally unproven without its stack.

### Gates 6 and 7 — passed cleanly after runtime fix

The seventh agent replaced only the producer registry's `Lock` with `RLock` and
added eight bounded subprocess regressions: actual proxy/unknown-client GC,
former-lock deadlock controls, nonweak eviction, stale-reference protection,
concurrent dispatch/collection, and durable-identity preservation. Exact weakref
matching, authority revocation, and proxy attestation logic are unchanged.
Root independently reviewed all critical sections and passed **202 local tests**
including the original stalled ordering. The final target gate passed **202
tests, zero failures/errors/skips**, in **236.13 seconds**, with source unchanged
and unraisable/thread exceptions treated as errors. No intervention was needed.
The implementation agent separately passed 115 adjacent regressions.

Root's target probe now completes with the actual patched source; restoring the
former lock in that disposable process still reproduces the deadlock. Tests
confirm unchanged exact Phase1, aggregation, digest/profile/facts identities;
the lock change alone does not request a material rebuild. Final producer source
SHA256 is `da19a52138752ff1e5ab05e2cb933eda9997500b4ea1f63ca2d6aa003ab06563`.

The final candidate covers **482 paths**, with **11 changed/added files**. A
fresh real-store backup/rehearsal passed again: schema **61 → 61**, all **131
tables unchanged**, reopen unchanged, integrity OK, FK zero, canonical OK, and
**585/585 valid coverage proofs**. Its first setup attempt could not read a
WAL-header snapshot from a read-only directory before initialization. Root
verified the closed backup had no WAL/SHM sidecars and that the transferred copy
hash matched exactly, then used `immutable=1` exclusively for that completed
read-only copy. It was never used on the live database. The failed setup logs
were retained. No production data was changed.

The next complete run (`full-verified`) is now running against this frozen
candidate, with disk-backed fixtures, restricted mounts, immediate failure
diagnostics, and fatal handling of unraisable/thread exceptions. No production
patch or restart has been performed during this fix cycle yet.

### Third full attempt — two scheduler unit failures; still running

The `full-verified` run exposed failures in
`test_scheduler_kick_runs_one_cycle` and
`test_scheduler_recovers_from_failing_cycle`: both require a real cold-store
initialization/dream to complete within five seconds. The latter additionally
claims to inject a failing cycle but never actually does. A separate eighth
implementation agent is correcting the unit-test synchronization and coverage;
the production scheduler and real-pipeline HTTP integration tests are unchanged.
The frozen full-suite candidate is not being changed while that run continues.
No deployment is authorized by this failed gate.

### Gate 8 — local verification passed; target gate running

The eighth agent changed only `tests/test_dream_scheduler.py`. Policy tests now
use the real scheduler thread with a small event-controlled fork; database and
provider latency no longer determine their verdict. The unchanged HTTP tests
still exercise real HyMem initialization, pipeline execution and proof
preservation. The unit cases verify exact cooldown boundaries/remaining delay,
coalescing without losing pending work, real exception and lease-loss recovery,
success counters, invalidation, close ordering and retryable shutdown. Negative
controls reject bypassed cooldown, dropped kicks, and an ordinary success
masquerading as failure injection.

Root reviewed synchronization and cleanup and independently passed **128
tests**, zero failures/errors/skips, in **94.589 seconds**, covering scheduler,
HTTP integration, connection lifecycle, concurrency and indexing deadlines. The
agent passed 95 adjacent tests, 128 cases across eight parallel repetitions,
and a 16-case in-process leak audit (all clocks/thread bindings restored; no
remaining worker). The only local warning was an installed FastAPI/Starlette
test-client dependency deprecation, not an application failure. Target checks
use a separate candidate differing only in this test file, while the original
full-run source remains frozen. Test SHA256:
`3388396b7abae6f555ed6822b6b61ea888108ebace1d365e75213b8a92ee077c`.

The target gate has now passed **128 tests, zero failures/errors/skips**, with
source bytes unchanged and unraisable/thread exceptions fatal. It took **858.30
seconds** while the broad gate was also running. Its 25-dream concurrency case
crossed the diagnostic stack-dump threshold; the stack showed active contract
identity computation. It completed naturally, without signals, retries or
intervention. The dump is retained; no new scheduler/runtime defect was shown.

The post-restart doctor wrapper additionally classifies warning labels and the
specific historical-embedding status, while retaining the maintained process-
exact probe and suppressing its raw output. Five offline negative/positive
receipt tests passed, including unknown-label and private-output suppression.

### Third full attempt — completed; failed evidence retained

The frozen `full-verified` candidate completed **5,956 cases: 5,954 passed,
two failed, zero errors/skips**, in **3,727.81 seconds**. Its only failures were
the two diagnosed scheduler unit tests. Candidate source hashes were unchanged;
no test was interrupted. This failed run and its pre-scheduler-fix manifest
remain retained and are not counted as a clean full pass.

### Final complete rerun — running

After the clean 128-case target gate and the completed failed run, root froze
only the scheduler test update into the final candidate. It covers **482 paths**
(477 files; five pre-existing deleted egg-info paths), with **12 changed/added
files**. Runtime code is byte-identical to the successful final store rehearsal.
The `full-finalized` complete run is now running with six workers, disk-backed
fixtures, no network, no live/copy store mounts, and fatal unraisable/thread
warnings. Deployment now requires this exact final source manifest's clean
full result, plus a fresh idle-state and live-source-integrity preflight.

### Fourth complete attempt — worker crash; not accepted

`full-finalized` completed **5,962 cases: 5,961 passed, zero assertion failures,
one worker-crash error, zero skips**, in **3,661.15 seconds**, with source hashes
unchanged. The crashed case was
`test_dreaming_ingestion_and_reads_coexist`. The container was not OOM-killed.
The crash occurred during the test harness's asynchronous 120-second
faulthandler dump: it printed one valid thread, then an impossible line number
(`1073615520`) and a truncated next frame before the worker disappeared.

This closely matches documented CPython stack-dumper defects
([116008](https://github.com/python/cpython/issues/116008),
[140815](https://github.com/python/cpython/issues/140815)). It does not yet prove
the original crash's native cause: no core was captured from the read-only work
directory, and the host kernel journal is not available to the current user.
A ninth agent initially investigated a paired reproducer; the deliberate-crash
method was stopped and was not run on the target. Its local timeout is not
evidence of the original cause. Work was narrowed to an ordinary Python-thread
diagnostic with benign tests. Production/source/Python versions are unchanged.
Honcho, dashboard, skill search and shared embeddings still return HTTP 200.
The failed run is retained, never counted as a passing full gate.

### Private diagnostic replacement — independently verified

The ninth agent supplied a private pytest plugin, not a repository/runtime
change. It replaces only the asynchronous traceback timer with a normal Python
thread that samples bounded stack locations once after 120 seconds. It emits
no frame locals and neither stops nor passes a slow test. Fatal-signal handlers,
assertions and fatal unraisable/thread exceptions remain enabled. An independent
7,200-second parent timeout rejects an indefinitely stalled suite; the sampling
thread itself cannot diagnose native code indefinitely holding the GIL.

Agent and root each passed **12 benign tests**, including ordinary sampling,
cancellation/no leaked threads, retained fatal-signal handlers, assertion/thread/
unraisable negative controls, and rejection of overlapping native timers or
invalid timeouts. The same 12 tests passed on the target runtime, **zero
failures/errors/skips**, with source unchanged and no store/network access.
Plugin SHA256:
`ca4286ccb17acb6c6e0cac1412b4e218f987d5750915bece095efb059612027c`.
The actual concurrency cases are now running with this diagnostic. The original
worker crash's native cause remains uncertain; a clean complete suite is still
required before deployment.

The target concurrency gate then passed **all six cases**, zero failures/errors/
skips, in **106.75 seconds**, including the worker-crash case. Source and runtime
versions were unchanged. The `full-safe` complete run now uses the same 5,962-case
selection and six-worker distribution as the rejected fourth run, changing only
private diagnostics and adding the independent hard timeout. Deployment requires
exact source hashes and exactly 5,962 passing cases with no skips/errors/failures.

### Fifth complete attempt — fixture ordering failure; not accepted

The `full-safe` run exposed a failure in the digest squeeze probe's no-cap
parity control. Root inspected its actual synthetic fixture read-only: both
edges have evidence margin 2, but `atta part_of medflow` was written at
`2026-09-11 18:03:49` and `medflow uses postgres` at `18:03:50`. Production's
documented order is evidence margin, recency, then semantic coordinates. The
newer edge correctly ranked first; the test incorrectly assumed a clock tie.
No production ordering/scoring fix is warranted.

Root independently replayed the failed fixture on a private copy. Tying only
its ranking timestamps restores lexical order; the one-second difference
recreates the original order. Both actual current/fixed loaders agree in both
arms, caps remain order-sensitive, and original fixture/canonical evidence/
coverage hashes remain unchanged. Source fixture SHA256:
`342cc8e7afedbb8e3f422e594b446a9a672108e0a1a7fcda01341f1e5955a8a9`.

The tenth agent changed only `tests/test_digest_squeeze_probe.py`: a private
connection-scoped SQL write clock makes the parity premise explicit while
retaining real evidence, coverage, lifecycle and publication writers. Both
insertion orders are tested. A deterministic one-second advance exercises the
real per-triple writer and retains newer-first precedence; cap controls and a
reversed-fixed-arm negative control retain exact ordered parity, not set parity.

The agent passed **107 adjacent tests**; root independently passed **198 tests**,
zero failures/errors/skips, in **117.781 seconds**, with fatal thread/unraisable
warnings. File SHA256:
`71a61b78ed684a1262ad34849a4a43d7f2fec9c8951569723d7dbdf17739e1f7`.
The target complete run is still finishing against unchanged source; its
failed result will be retained, not combined with focused passes and presented
as a clean full run. Five ordinary-thread slow-test samples have completed
without a worker crash so far. No production change/restart has occurred in
this fix cycle.

The fifth full run completed **5,962 cases: 5,961 passed, one fixture failure,
zero errors/skips**, in **3,614.68 seconds**, source unchanged. All five slow-test
samples completed with ordinary Python-thread diagnostics; there were no worker
crashes. This supports the private diagnostic replacement operationally, but
does not retrospectively prove the original native crash's cause. The sole
failure was the independently reproduced digest ordering fixture. Its complete
failed receipt and source manifest remain preserved.

### Gate 10 — target passed; complete rerun started

After the failed run exited, root froze only the digest squeeze-probe test
change. The final manifest remains **482 paths / 477 files**, now **13 changed/
added files**. Runtime code is still byte-identical to the successful final
store rehearsal. The target-runtime gate passed **198 tests, zero failures/
errors/skips**, in **325.24 seconds**, with source unchanged and fatal thread/
unraisable warnings. No intervention was required.

The `full-reviewed` complete run now uses the same six workers and test selection
plus the three new ordering cases: **5,965 cases expected**. It keeps the verified
ordinary-thread diagnostics, fatal error policies, isolated disk-backed fixtures,
no network/store access and 7,200-second hard bound. Deployment requires its exact
source manifest and a complete zero-failure/error/skip XML result; the earlier
failed runs are not accepted or discarded.

### Complete acceptance gate — passed

`full-reviewed` completed **5,965 passed, zero failures/errors/skips**, in
**3,618.56 seconds**, with an ordinary exit 0 and no interruption. Root's separate
verifier checked the XML, container exit/OOM status, all **477 file hashes / 482
manifest paths**, isolated mounts, diagnostic implementation and unchanged
runtime bytes relative to the final store rehearsal. Both corrected parity
orders, the second-boundary reproduction and reversed-arm negative control
passed in the full run. The former worker-crash concurrency case passed in
**210.582 seconds**. Five ordinary-thread stack samples completed without a
worker crash; the native asynchronous timer remained off while fatal error
handling stayed enabled.

One third-party `IncompleteFieldDefinitionWarning` concerning FastMCP's
`lifespan` annotation was emitted by the schema-signature test. It was not an
assertion error, and the public-schema/signature test passed. The pending live
MCP verification remains a separate acceptance check. No scored benchmark or
production dream/drain was used to obtain this gate.

### Deployment and live verification

After a fresh idle-state check (no active gateway/web runs, cron executions,
benchmarks, dream drivers or dream lease), root rechecked production integrity:
canonical OK, FK zero, all 585 proofs valid, and no attributed messages or
memberships inside NULL-owned sessions. The operations skill's backup-first
workflow retained exact pre-install source/configuration and an offline,
consistent database backup. Only the 13 reviewed files were installed while
Hermes1 was stopped. Hermes1 restarted at **2026-09-11 19:25:02 UTC**
(21:25:02 Europe/Amsterdam), based on upstream `af6a615` plus the reviewed
working-tree fixes. No commit/push was made.

All protected config hashes/modes remained unchanged, including credentials,
embedding identity, wrappers/hook and session-retirement mappings. Hermes2,
Hermes3 and the shared embedding container were not restarted or modified.
The first early readiness poll preceded gateway readiness; acceptance waited
for its running state rather than treating four HTTP 200s alone as sufficient.

Current-boot verification passed: one Honcho, two MCP servers/two watchdogs,
dashboard and skill-search processes all started after deployment; startup
completed with no current-boot exceptions/tracebacks. Gateway process identity
matches and email/Telegram are connected without reported errors. The optional
scheduler INFO marker is suppressed by the maintained launcher; its absence is
not counted as a failed worker.

Post-restart doctor passed with **zero failures** and the **one unchanged
historical stored-embedding-compatibility warning**. Actual model completion
and remote embedding probes passed, the active model remains Flash, both MCP
embedding configurations agree, backend is `openai_compatible`, and no feature-
hash fallback is active. Integrity/FK/canonical checks and **585/585 coverage
proofs** passed again. Two actual SDK config reloads, four mapped-session opens
and four context reads passed, with legacy history unchanged and zero synthetic
production messages.

Actual MCP initialization/listing advertised all 12 expected tools; profile,
augment, digest and rule-list reads succeeded, followed by a successful ping.
The extra verification server exited normally. Augment took 8.591 seconds and
digest 7.352 seconds during these checks; these are functional receipts, not a
latency benchmark. The installed SDK captured two synthetic messages on a
private production-store clone across two config reloads: two new valid proofs,
**587/587 clone proofs valid**, legacy rows unchanged, zero model calls and zero
production writes.

Final checks after those probes reconfirmed the regular process counts and
connected gateway platforms, no current-boot exceptions/tracebacks, production
integrity/FK/canonical/coverage status, all **482 manifest paths**, all three
capture integration pins, protected configuration hashes/modes and four HTTP
200 health responses. `/dream-status` is healthy, no dream is in progress or
leased, coverage-integrity failures and pending aggregation are zero. It still
reports **321 pending chunks** and last completed dream **1403**; no drain was
triggered to change those historical counters.

### Receipt archival — completed and independently verified

Persistent receipts on Hermes1 are under
`/home/node/.hermes/backups/hymem-verification-fixes-20260911/verification/`.
They retain the accepted complete/focused gates, rejected and interrupted runs,
negative controls, store rehearsals, all ten live checks, manifests, scripts,
agent/root verification receipts and this report. The consistent pre-install
database, exact source and protected-configuration backups remain alongside it.

A private archival helper initially stopped before removing any containers:
its synthetic-fixture guard incorrectly rejected an empty WAL and an ephemeral
SHM file left by read-only inspection. An eleventh bounded agent replaced the
main-file copy with SQLite's read-only backup API. Root's actual-fixture rehearsal
then caught missing connection-local CHECK functions on the snapshot connection;
the helper now installs the candidate's genuine read-authority validators on
both connections before validation. No CHECK bypass or production change was
used. Missing functions, wrong source identity, source mutation, rollback
journals, timeout and existing destinations still fail closed.

Agent and root each passed **10 offline archival tests**, including committed
nonempty WAL contents and retained CHECK rejection of invalid values. Root's
separate target-host rehearsal compared every row of all **106 tables** against
the original synthetic fixture, verified unchanged source main/WAL content and
standalone snapshot integrity, and confirmed overwrite refusal. The first
partial archive was preserved as `verification-initial-archive-attempt`, not
overwritten; the failed preliminary snapshot rehearsal also remains in staging.

The corrected archive completed. Root independently verified its accepted
**5,965-test** and **46-test** results, all ten successful live receipts, matching
synthetic snapshot bytes, directories at 0700/files at 0600, recovery backups
and the retained initial archive. Exactly **19 label-verified, exited test
containers** were removed, with none left under this operation's label. Their
verification evidence remains recoverable from the archive; no production
container, database, source tree or recovery backup was removed. This private
archival-only fix did not change the deployed application or invalidate the
complete application acceptance gate.

At **2026-09-11 19:44:28 UTC**, a final read-only check again found the gateway
running and idle and HTTP 200 from Honcho, dashboard, skill-search and the shared
embedding service. There was no further restart, benchmark or dream/drain.
