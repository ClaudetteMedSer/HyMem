# Schema 61: live repair and verification

## Final follow-up outcome — 2026-09-09 19:29 UTC

All three instances now carry the reviewed MCP worker and atomic-startup fixes,
with HyMem MCP request timeouts set to 1800 seconds. Hermes3's existing scheduled
job uses the configured MCP path, and its post-fix dream 5969 completed without
failure. Two source-proven historical graph mirrors were re-embedded through the
maintained repair API; unprovable graph records were not deleted or relabeled.
Separate implementation agents supplied each confirmed fix, followed by root
review, regression tests, private production-clone gates and sequential live
verification. No commit or push was made; local runtime changes and the 26 new
regression cases remain in the working tree.

Verification covered **5,683 current tests**: the complete restricted run reached
100%, with only five failures and 16 setup errors from sandbox-denied localhost
socket binds. All 21 affected cases passed the permission-enabled rerun. The
full collected test set has no remaining failing case. The pytest cache also
contains 97 obsolete identifiers from earlier revisions; none belongs to the
current collection, and they were not manually deleted to manufacture a pass.
Each instance additionally passed 372 selected tests in its installed production
environment and a private live-store clone rehearsal. Timeout tooling passed
83 root-run tests; the cleanup helper passed 16.

The approved temporary-backup cleanup was then applied against exact inventory
SHA256 `1e4c602a0a83b97b2f48e32f02d8437b0b3f001c6ce847ddb38dde5cc635bfdc`:

| Instance | Temporary files permanently removed | Logical bytes removed | Audit/quarantine files retained |
| --- | ---: | ---: | ---: |
| Hermes1 | 2,115 | 7,479,551,162 | 13 |
| Hermes2 | 1,026 | 83,249,462 | 11 |
| Hermes3 | 1,114 | 418,523,573 | 18 |
| Total | 4,255 | 7,981,324,197 (~7.43 GiB) | 42 |

The 42 retained files total 588,291 bytes and were verified byte-identical,
including Hermes1's unique original orphan quarantine. Routine/daily backups
and the standing historical database backups outside the temporary-repair scope
were not touched. 310 empty temporary directories were removed. The deletion
was permanent unlink, not Trash; the deleted temporary rollback/rehearsal copies
cannot be restored through this operation. Earlier artifact paths below now
contain retained receipts/manifests only, where applicable, rather than their
deleted backup payloads.

**Post-cleanup verification passed on all three instances:** actual MCP handshake,
profile and retrieval; configured remote embedding identity; SQLite integrity;
zero foreign-key violations; zero canonical drift; zero source-eligible or
rebuild-required incompatible mirrors; and current aggregate publication (nine
nodes on Hermes1, seven on Hermes3; no eligible aggregate inputs on Hermes2).
Hermes1's Honcho health check also passed. Live database inodes, unrelated
runtime files/customizations, and all retained receipt/quarantine bytes stayed
unchanged through cleanup.

This is not a claim that every raw historical diagnostic is green: Hermes1
retains 1,090 unverified historical chunk mirrors, and Hermes3 retains 11
unprovable graph mirrors plus 85 retired fact mirrors. Those agreed historical
limitations remain visible. No further actionable defect was found in these
checks, and no additional watcher or automation was created.

## Rollout chronology — pre-cleanup snapshots

Hermes3 now has both reviewed runtime fixes. The final startup-clone helper
passed root's independent synthetic nonempty-vector test; it includes vector
backing tables regardless of SQLite's connection-local table/shadow label.
Only positively identified FTS5 physical shadows are excluded from its logical
row digest, while their virtual rowsets and schema remain checked. The earlier
plan-only clone gate refused because of that verifier classification mismatch;
no code was installed by the refused plan.

The corrected Hermes3 rehearsal passed all 372 selected production tests and
checked a clone containing 14,182 logical rows across 103 tables: 1,404 DDL
boundaries, two blocked concurrent invalid writes, two clean opens (3.486 and
2.998 seconds), and a late injected failure with complete rollback. Integrity,
foreign keys, canonical state, logical rows and schema all passed. Source access
was read-only backup only, with zero provider attempts; the clone was removed.

At an idle boundary, Hermes3 alone was stopped, the single remaining `db.py`
delta installed, then restarted. All other runtime files, file ownership/mode,
the live database inode and the other instances' start times were preserved.
Artifact: `.hermes/backups/hymem-runtime-fixes-4xt53eec`. Runtime hashes:
`server.py` = `7eb8e9e2af0a62560c83114864c58322d0d1870ceb55c72bd5efe63f09e15f47`;
`core/db.py` = `d2cd8ae6de3654657b83ac3cf22df4250d763f52e16fa59e30c32a67ecfa9122`.
Post-restart verification passed actual bootstrap, the configured remote producer,
MCP profile/retrieval, integrity/FKs/canonical drift, seven-node current aggregate
publication, and zero repairable incompatible mirrors. Retained historical
mismatches are still reported honestly.

The corrected scheduled dream 5968 completed at 18:51:36 with zero fact failures
and all seven aggregates reused. After the startup patch, the supported job API
launched 5969; it completed at 19:05:03, processing 40 sessions, again with zero
failures and seven-of-seven reuse. Both job tools returned successfully and
released their leases. A proposed additional concurrent live-startup probe did
not pass its active-dream preflight; no claim of a successful live overlap is
made. Deterministic concurrency coverage is provided by the clone gates.

The full local suite has 5,683 tests. Its first five failures and next 16 setup
errors occur in localhost socket fixtures denied by the sandbox, before HyMem
logic is exercised. The seven embedding-health tests and all 16 Honcho SDK
contract tests pass with socket permission.
Do not describe the restricted run as a zero-failure single invocation.

Hermes1 and Hermes2 also retained the 300-second default MCP timeout. A separate
agent prepared a digest-pinned, byte-preserving alignment helper; root reviewed
it and independently passed all 83 tests. Hermes2's timeout-only update and the
actual installed loader passed, preserving all other configuration bytes and
mode/ownership. Its backup/receipt is `.hermes/backups/hymem-mcp-timeout-hok8qj18`;
new config SHA256 is
`96737f5a6d29d56d5c276a5eb441985b27a1ff8ecaa35d1935cc86800cd4a283`.
Hermes2's repeated runtime rehearsal passed 372 tests and a 7,550-row, 78-table
private clone with all 1,404 boundary checks and rollback invariants. Both runtime
files were installed while that instance alone was stopped; backup/receipt:
`.hermes/backups/hymem-runtime-fixes-wdvvydkx`. Post-restart actual MCP startup,
profile/retrieval, remote identity, integrity/FKs/canonical drift and source
health passed. Its vector inventory is empty/compatible, as before. Repeating
the timeout updater after restart was an exact no-op with the loader still
resolving 1800 seconds.

Hermes1's timeout-only correction also passed its installed loader. Artifact:
`.hermes/backups/hymem-mcp-timeout-j4wjfjh7`; new configuration SHA256:
`d00f764c9c5fe6f8b8fa279e85d9eb96a6950bdb97ad745ef2401467246180ff`.
Its runtime installation passed 372 production tests and the private-clone gate:
246,106 logical rows, 103 tables, all 1,404 DDL boundaries, two blocked writes and
late-failure rollback; total rehearsal 26.129 seconds, normal traced opens 3.980
and 3.877 seconds. Both runtime files were installed while Hermes1 alone was
stopped. Artifact: `.hermes/backups/hymem-runtime-fixes-11pe8gx9`. All unrelated
runtime files, including its custom client extension, remained pinned to its
own pre-update manifest; the live database inode and other instances' start
times were preserved.

Hermes1's post-restart verifier passed actual MCP startup/profile/retrieval,
Honcho health, serving configuration agreement, the correct remote embedding
producer, integrity, foreign keys, zero canonical drift, and nine-node current
aggregate publication. Its timeout replay was an exact no-op. All three now
carry the reviewed runtime fixes and 1800-second MCP timeout. Fresh Hermes2 and
Hermes3 checks also passed before cleanup inventory.

Historical-only state is retained: Hermes1 has 1,090 unverified historical chunk
mirrors (1,062 with terminal source loss); Hermes3 has 11 unprovable graph mirrors
and 85 explicitly retired fact mirrors. Each instance has zero source-eligible
or rebuild-required incompatible mirrors and zero malformed/unverified vector
payloads. This does not mean every raw doctor label becomes `compatible`.

Backup cleanup remains unapplied while the newly exposed MCP/startup defects
are verified. Root's independent focused startup/MCP gate passed 101 tests and
the full suite is still running. The startup implementation agent passed 462
unique focused tests. Each remaining instance must pass its clone rehearsal and
post-restart verification before the next instance is changed.

The post-5967 source scan found two incompatible edge mirrors with valid source
proof under the neutral diagnostic scope, but **zero** eligible under the actual
configured Phase-1 generation. These are older-generation mirrors, not a missed
current-dream indexing pass. Root verified this without provider requests or
database initialization. A maintained `reembed.repair` rehearsal then proved
that exactly two mirrors could be updated without altering source/history,
schema, or unrelated tables. The first private verifier incorrectly assumed
JSON vector storage; using the maintained vector decoder corrected that
verification-only error. No live write occurred on the failed rehearsal.

The rehearsed live repair succeeded with one embedding request and no LLM
requests or initialization. Exactly two mirrors and their corresponding vector
index entries changed. A second complete sweep repaired zero rows and made no
provider calls. Source/history and schema digests remained equal, integrity and
foreign keys passed, and the live database inode was preserved. The fresh backup
and receipt are in `.hermes/backups/hymem-historical-edge-reembed-n4ifxvrr`.

An independent read-only scan immediately after that repair found zero eligible/rebuild-required
incompatible mirrors on Hermes3: chunks 340/340 current, messages 118/118,
edges 171/182, episodes 28/28, facts 90/176. The 11 remaining incompatible edge
mirrors are still unprovable and the 86 incompatible fact mirrors are explicitly
retired. They are retained, not relabeled or deleted. The raw/source doctor's
historical `action_required` classification is not falsely reported as fully
compatible. The later successful 5968/5969 outcomes are recorded above; they
also re-embedded another existing fact mirror, leaving 85 retired incompatible
fact mirrors at the subsequent post-restart snapshot.

## Follow-up: the scheduler bypass was found and corrected

The four eligible fact-vector mismatches were transient: dream 5959 finished
at 16:36:56 UTC, and the subsequent Hermes3 runtime/source-health verification
passed with zero eligible incompatible mirrors. That run recorded one held
fact extraction failure, not an overall execution failure. Later read-only
status showed zero fact-retry/quarantined sessions; no error was manually
cleared.

The repeated `execution_failure:RuntimeError` had a separate deployment cause.
Hermes3's existing 30-minute job `e9eb69d1cd91`, **HyMem Autonome Dreaming**,
contained an inline script that directly constructed `FastEmbedClient()` and
`HyMem(...)`, bypassing the configured MCP/bootstrap remote embedding client.
Its scheduler output repeatedly contains the exact maintained rejection
`inexact embedding producer has no durable storage identity`. The local custom
adapter has no exact maintained producer declaration. The identity guard is
working as designed and was not relaxed. Old generated script copies include
a LocalHash variant too; the replacement explicitly prohibits their use.

This also explains why a successful startup/endpoint check did not catch the
bad writer: that check verified the service configuration, whereas the job
constructed its own clients. Failed dream rows only persist the error and end
time; their zero counters do **not** establish zero committed earlier work.
The old scheduled runs used aggregation generation ending `...cdaecc3`, while
dream 5959 used the configured generation ending `...126b96`. Switching callers
therefore also creates unnecessary replay/rebuild work.

A separate implementation agent supplied the replacement prompt and a guarded,
immutable transform. Root reviewed both and independently passed all 14
tests. A private-copy rehearsal used Hermes' actual installed job API and
proved that only this prompt changes, all five other jobs and all other fields
are preserved, and replay is idempotent. The live update used the scheduler's
cross-process lock, refusing its fail-open lock-timeout branch. The original
prompt SHA-256 was pinned to
`a4db6320897cc75d8a0db1cf1aa9660a0fa59e749a5062a545106fec5547a732`;
the replacement is
`52f0078804bae2d7ba4622e4afe378b79b1e498fe84d4b95a22e8b73afa6306b`.

The job now requests exactly one configured `hymem_dream` MCP call, with no
custom client, local fallback, disabled embeddings, identity bypass or retry
through a different producer. Its 30-minute schedule, delivery and other
settings are unchanged. Neither Hermes1 nor Hermes2 has the same obsolete
inline job. Their fresh runtime/health checks passed again.

The update receipt and pre-update scheduler copy remain on Hermes3 in
`.hermes/backups/hymem-dream-scheduler-s4lag5z2`. Dream 5960 had already started
at 17:07:56 using the old job snapshot before the prompt update. That old worker
went on to launch 5961, 5962 and 5963. Its attempts recorded RuntimeError,
ExternalError and OperationalError before 5963 eventually completed at
17:28:59. Those additional legacy-attempt categories are not attributed to new
runtime defects without an underlying trace.

Root waited for a lease-free boundary and restarted only Hermes3 to stop the
obsolete worker. Preconditions verified no other active cron job, the corrected
prompt, and exact runtime pins; postconditions verified the database inode and
runtime files unchanged and Hermes1/Hermes2 not restarted. The scheduler kept
the interrupted old execution honestly marked `unknown`; it was not rewritten
as a success. No database restore occurred.

The supported existing-job run API then dispatched the corrected job once.
Dream 5964 started at 17:30:06 with the configured generation ending `...126b96`,
not the old caller's `...cdaecc3`. This exposed a second deployment defect:
Hermes' default MCP request timeout is 300 seconds, and the job reported
`TimeoutError` at 17:35:09 before the dream completed. The scheduler execution's
`completed` status means its agent finished, not that the requested dream
succeeded. Root checked the underlying dream and tool outcome instead of
accepting that status as a pass. Exactly one dream was launched, with no
fallback retry. Later inspection proved its worker had exited and its lease
heartbeat had expired; the unfinished telemetry was not active work.

A new implementation agent supplied a pinned byte-preserving YAML transform.
Root reviewed it and independently passed all **45 tests**. A private rehearsal
through the installed Hermes MCP configuration loader passed, as did the live
loader after application. Only `mcp_servers.hymem.timeout: 1800` was added to
Hermes3's configuration. All original bytes, embedding settings, permissions,
ownership and unrelated parsed settings were preserved. Managed configuration
guards were checked, the original hash was pinned, and replacement was atomic.
The before/after configuration digests are respectively
`f46018dc64b72ffbb6e0d107867b11b840a4fc2a631fa5bb1074e890fc1f99fc` and
`90cbcb30876ed15721819c924cd06c1f80f1d1ea5a025dd71a34ec1ceef1f343`.
The private rollback copy and receipt are in
`.hermes/backups/hymem-mcp-timeout-vvqa0wc_`.

Hermes3 was reloaded again after verifying no active cron job and no live dream
worker; the only remaining lease was expired with a missing holder process.
Hermes1/Hermes2, all runtime bytes, and the live database inode were unchanged.
The next corrected job used normal lease compare-and-swap recovery, not manual
lease deletion or a fabricated success for interrupted dream 5964. Dream 5965
started at 17:50:10 with the configured producer, but its worker also exited
before completion. The heartbeat stopped at 17:53:18; a read-only check proved
the holder process was absent. The caller eventually failed at 18:00:14. No
successful dream was claimed, and backup deletion remains paused. The supported
manual-run API advances the next due timestamp; the 30-minute interval is
unchanged.

This exposed a further MCP transport defect. Root inspected both local MCP
1.27.1 and production MCP 1.29.0: FastMCP directly invokes synchronous tools on
its event loop. HyMem registers all twelve synchronous functions directly,
including the long-running dream. Hermes sends a liveness ping every 180 seconds
and reconnects after its 30-second ping timeout, even with a tool outstanding.
The observed worker loss matches that mechanism; increasing only the outer
request timeout cannot keep the blocked transport responsive. Container OOM
counters are zero and the existing core dumps predate today's runs.

A separate implementation agent reproduced the failure with an actual stdio
MCP subprocess: a blocked synthetic dream prevented a ping reply before release.
The same test passes after the patch. The MCP entry point now registers async
adapters backed by a single dedicated worker; bootstrap, all twelve tools and
teardown stay on that same worker. Public synchronous functions and all twelve
names/descriptions/input/output schemas are unchanged. Queued cancellation can
prevent execution, but already-running synchronous work cannot be safely
interrupted; it retains sole ownership until completion. Shutdown cancels queued
calls and waits for the active work before closing the store and clients.

Root independently reviewed the implementation and passed **267 tests**, covering
the ten new protocol/ownership regressions, existing MCP/capture tests, bootstrap
lifecycle and fail-closed tests, and dream lease/scheduler tests. A private copy
on Hermes3 passed all **243 selected tests** against production MCP 1.29.0,
AnyIO 4.14.2 and Pydantic 2.13.4. It made no live code or store changes. The sole
runtime change is `hymem/server.py`, pinned from
`ef03fc7b2bedcccedb1b764511b66a3de4748b1ba2f85a157a2063cacf0ae906`
to `7eb8e9e2af0a62560c83114864c58322d0d1870ceb55c72bd5efe63f09e15f47`.
Hermes3's guarded deployment succeeded and only that instance was restarted;
the other instances and the live database inode were unchanged. No commit or
push was made. The other instances must not be updated until live verification
and the following newly exposed startup-race fix are complete.

The first guarded installation refused before `os.replace`: its full runtime
inventory included its own same-directory staging file. Root verified the live
server still had the original hash, corrected the exact staging-file admission,
and reused only the verified two-file recovery artifact
`.hermes/backups/hymem-mcp-worker-zklufrmu`. No live code was replaced or service
restarted by that refused attempt. Production-SDK tests are repeated on retry.

After successful retry, dream 5966 failed at 18:17:50 with an OperationalError.
Root did not accept the scheduler agent's `completed` status as dream success.
A secret-safe read of the recorded tool response identified the exact internal
failure: `no such table: current_phase1_publications`. A normal runtime health
probe was opening another HyMem connection at that time. Startup unconditionally
drops the six v54 authority views, and several startup `executescript` calls
commit individual drop/create statements. This exposes missing views (and
potentially temporarily absent guards) to concurrent connections. A separate
implementation agent is reproducing and fixing the full startup repair boundary,
including caller-transaction and rollback behavior. No raw tool response or
credential content was exported from Afrodite.

To isolate transport liveness from concurrent startup repair, root started one
normal configured MCP dream (5967) through the installed console server with
the actual serving environment and independent protocol pings every 30 seconds.
Dream 5967 completed successfully at 18:30:04: 40 sessions processed, 13 facts,
12 episodes, seven rebuilt aggregate nodes, zero fact failures. Fourteen real
pings passed over the approximately 7.5-minute MCP call, beyond both the old
180+30-second keepalive failure and the old 300-second request timeout. The
lease was released and fact-retry/quarantined sessions both became zero. The
post-dream store health gate is running sequentially; startup atomicity remains
the next unresolved defect. No backup deletion has occurred.

Independent cleanup tests now pass **16 cases**, including the actual inventory
path, hardlinks, file/directory symlinks, changed bytes, unknown backup roots,
missing keepers, and preservation of standing backups. **No backups have been
deleted.** The refreshed plan includes the scheduler and timeout rollback
copies, preserving 27 receipts/quarantine files. Re-inventory after final
health verification. A further **203** bootstrap and embedding-source-health
regression tests passed independently; no runtime code changed in this follow-up.

## Later backup-cleanup preflight: deletion paused

After the user conditionally approved removing backups if no work remained,
fresh service checks passed on Hermes1 and Hermes2. Hermes3 passed its code,
startup, remote embedding, MCP, canonical-drift, integrity and publication
checks, but its source-health gate found **four eligible incompatible narrative
fact mirrors**. The remote producer and shadow metadata still match; this is
not evidence of another switch to LocalHash. Of its 74 incompatible fact
mirrors, 70 now classify as retired and four as source-eligible, whereas the
earlier verified snapshot classified all 74 as retired.

At 2026-09-09 16:36:55 UTC, dream 5959 (started 16:29:01) was still running
with a held/refreshed lease. These four embeddings may be pending within that
run; a persistent defect has not yet been established. Separately, completed
dreams 5958 and 5956 recorded the maintained safe failure category
`execution_failure:RuntimeError`; the underlying cause remains undiagnosed.
The old service-log traceback locations did not establish the current cause.

**No backups were deleted.** All earlier recovery artifacts below still exist.
The plan-only cleanup identified 4,244 temporary files totalling 7,953,230,663
bytes for removal, preserving 25 small receipts/quarantine files and leaving
standing historical/daily backups outside its scope. The cleanup helper passed
eight local safety tests, and the actual plan verified no running instance
process referenced the target backup directories. Its plan SHA-256 is
`628cf5ad1f28f644dc0948a8354abde2e2ee643cfca5ce7520afbb8aea39b1c8`.
That plan is not deletion authorization after future state changes: re-inventory
and verify health first. Scripts are in
`/private/tmp/hymem-backup-cleanup.8ijaFS/`.

The outcome table below records the successful deployment checks, not a claim
that this subsequent pending-work/failure finding has been resolved.

## Outcome

The approved release was rehearsed against fresh private copies, then applied
in place to Hermes2, Hermes3 and Hermes1, in that order. Each instance passed
its live service checks before the next was changed. All three are running
schema 61 with the reviewed runtime changes. No commit or push was made.

This is a verified service/repair outcome, **not an assertion that all historical
data has recoverable provenance or that future failures are impossible**.

| Check | Hermes1 | Hermes2 | Hermes3 |
| --- | --- | --- | --- |
| Schema | 61 | 61 | 61 |
| Canonical-drift findings | 0, previously 5 | 0 | 0 |
| SQLite integrity / foreign keys | OK / 0 faults | OK / 0 faults | OK / 0 faults |
| MCP handshake, 12 tools, profile, retrieval | Passed | Passed | Passed |
| Actual bootstrap remote embedding identity | Verified | Verified | Verified |
| Synthetic embedding endpoint request | Passed | Passed | Passed |
| Real synthetic extraction, `deepseek-v4-flash` | Passed | Passed | Passed |
| Valid published aggregate nodes | 9 | No eligible input | 8 |
| Honcho health | HTTP 200, healthy | Not deployed | Not deployed |
| Eligible incompatible stored mirrors remaining | 0 | 0 | 0 |
| Raw malformed / unverified vector counts | 0 / 0 | 0 / 0 | 0 / 0 |
| Historical source classification | Historical warning | Compatible | Action required: 8 unknown owners |

The last row is deliberately not hidden. A current-compatible vector means its
producer metadata and numerics match, not that every stored row is a current
authorized fact or that missing vectors have been created.

## What changed and why

### Canonical drift and collision preservation

The five Hermes1 findings represented two malformed canonical identities, not
five independent entities. The same identities were already present in schema
31, 33, 35 and 46 backups. They were noncanonical under the older normalization
policy too; this was not newly caused by Unicode normalization.

Old alias writes admitted unchecked targets. Existing malformed targets could
then propagate through mention indexing and derived-edge inference. Schema 60
adds SQL-NULL-safe write guards; the maintained writers normalize admitted
values, while unrelated edits of historical rows remain possible. Startup
validates/heals owned guards instead of silently permitting malformed writes.

Canonical collisions also exposed a preservation gap: distinct original
extraction variants could collapse into one representative. Schema 61 adds a
non-authoritative original-extraction audit. Merge and portable format 17 retain
the complete original occurrence set without counting that audit as evidence,
inventing past authority, or reviving retired claims. Exact manual signal/event
copies now coalesce together. Producerless portable formats 7–12 replay without
adding spurious evidence intervals or lifecycle dependencies.

The live Hermes1 repair touched 688 edges and removed all five findings:

- Evidence carriers: 20,623 to 20,595 through verified collision coalescence.
- Original owner occurrences: **all 20,623 preserved**, including 2,105 explicit
  audit rows; unchanged carriers remain their own original records.
- All 66 independently qualified current semantic claims were preserved.
- Five exact duplicate confidence signals coalesced; 145 surviving signal keys
  were remapped. Distinct payloads and manual signal/event pairing were retained.
- One exact duplicate lifecycle row and one observation coalesced. The verifier
  checked their mapped semantics, not merely their row counts.
- Sixty-six pre-existing unsupported current carriers were retired by the
  maintained reducer: 58 already had mismatched outcome binding, eight lacked
  the required outcome. No previously qualified current authority disappeared.
- 369 affected graph mirrors were created/rebuilt, using 12 embedding requests.
  All 7,811 live graph owners were indexed; the repeat was an exact physical
  no-op with no provider requests.

### Historical source loss

The older raw-pruning implementation removed messages based on a summary
without retaining sufficient exact source proof. Old backup schemas also lack
peer/workspace coordinates needed to establish occurrence-specific provenance.
Four backup audits yielded 132 renderable excerpts but **zero fully proven
restorations**. Text similarity and absent scope columns are not proof that a
message belonged to a particular native/null scope.

A separate current retention race was independently reproduced and fixed:
pruning now holds a writer transaction, uses savepoint-isolated rollback for
caller transactions, and revalidates exact timestamps and lease/deadline state.
This prevents that demonstrated maintained-path race; it does not prove the
race caused each older loss or recreate already missing metadata.

Source-aware diagnostics now distinguish eligible mismatches, retained-unverified
history, proven retirement and proven successful-empty source withdrawal. The
withdrawal proof requires complete retained receipts at the same producer
binding; a retracted flag or a matching prompt alone is insufficient.

### Unexpected local embedding fallback

Hermes3's vector changes match a completed dream at
2026-09-09 08:47:31–08:48:25 UTC. The stored old producer was independently
matched to the maintained LocalHash embedder. Its chunk/edge rewrite counts
match that dream, and the live database inode matches the earlier repair
receipt. The earlier repair had succeeded; the database was not replaced with
its old backup. The exact initiating caller/configuration failure for that dream
could not be recovered from available logs.

Two startup gaps were independently reproduced and fixed:

1. An explicitly configured remote endpoint could be rejected, lack credentials,
   or fail client construction and silently select a local producer. Startup now
   refuses before opening the store. Error reporting suppresses sensitive
   provider exception details and cleanup remains owned and tested.
2. Complete loss of embedding settings could still select the local default
   over a store containing remote vectors. A bounded read-only admission check
   now refuses that mismatch or unverifiable producer state. It observes
   committed WAL state, never initializes/migrates the store, and preserves
   deliberate local operation on fresh/pre-vector stores and explicit local
   re-embedding workflows.

The live Hermes3 repair rebuilt 447 eligible existing mirrors in 19 embedding
requests, then rebuilt eight aggregates in eight LLM requests. Repeating both
repairs required zero provider requests. The actual remote producer is shared
across all three instances. Its storage-key SHA-256 is:
`cea972cb2a6e3072ed56a3308efca4857b868323f710c166c6fcd1015a6081b4`.

Read-only fault injection against the actual stores confirmed that missing
configuration is refused on Hermes1/Hermes3, while Hermes2's empty vector store
retains the intentional local default. Explicitly rejected endpoints are refused
on all three. These probes changed only their own diagnostic-process environment;
they constructed no provider clients and performed no database row writes.

These are environment-bootstrap admission guards, not a global writer lock.
Direct API injection, intentional explicit producer changes, manual SQL or
running obsolete code remain operator responsibilities. All writers must retain
the same complete, verified environment.

## Verification evidence

Separate implementation agents were used for the fixes, followed by independent
root review and tests before accepting the next issue. The detailed sequential
record remains in `2026-09-09-historical-remediation-and-prevention.md`.

The initial three-shard integration run covered 5,571 cases: 5,521 passed,
29 canonical fixture/evaluation failures and 21 localhost sandbox socket
failures, with zero skips. The fixture/harness fixes and authorized localhost
reruns resolved these failures. Nine new harness regressions brought that
inventory to 5,580. The two startup fixes added 28 and 49 cases.

Final collection and XML testcase reconciliation show **5,657 unique passing
repository cases, no missing cases, and no unresolved failures/errors/skips**.
This is full-suite coverage plus verified reruns, not a claim that one initially
clean full run passed. Root independently passed the final 504-test and 553-test
startup/adjacent gates. The latter includes one additional scratch regression.
The installed FastAPI/Starlette httpx deprecation warning is not a test failure;
dependencies were not changed as part of this repair.

Private complete release rehearsals passed for all three, including actual
bootstrap construction on the repaired copies. Migration preserved all old
rows and added only the specified schema objects. Repair preservation,
rollback tests, repeat no-ops, reopen, foreign keys, FTS, publication and strict
vector/source classification gates passed. Root also independently checked the
one-off maintenance/host safety helpers, including failure and cleanup paths.

Live rollout used a fresh verified SQLite backup, a stopped normal container,
an isolated sole writer, exact code inventories, and in-place migration/repair.
No stale clone was installed, no automatic database restore was attempted, and
the source inode and backup bytes were verified unchanged. Only after complete
maintenance verification were 16 runtime files and 48 total release files
installed and the normal container restarted. Existing local customizations,
including Hermes1's LLM extra-body extension, were preserved.

The console-script MCP probe exercised actual startup, tool listing, profile and
retrieval. It kept the embedding client open through publication attestation.
An initial probe mistakenly closed that client too early and was corrected;
no production code or publication rule was changed to make that check pass.
Another probe initially compared raw Honcho/MCP environment dictionaries:
inherited credential aliases, explicit true/default-true aggregation flags and
the role-specific cooldown differed. Effective credentials, provider and
aggregation settings were then verified equal in memory without exposing values.

The three extraction canaries each returned a valid cited expected triple in
two real completion requests. They used only a newly authored fictional sentence,
performed no database access and sent no private memory. The older remote canary
was not displayed after safety review rejected possible sensitive-source exposure.
No full LME, BEAM or LoCoMo benchmark was run.

## What must remain visible

| Retained incompatible mirrors | Hermes1 | Hermes2 | Hermes3 |
| --- | --- | --- | --- |
| Unverified chunk provenance | 1,090 | 0 | 264 |
| Terminal source loss, subset of preceding row | 1,062 | 0 | 264 |
| Proven-retired fact mirrors | 0 | 0 | 74 |
| Unsafe-unknown graph mirrors | 0 | 0 | 8 |

Hermes1's remaining compatibility warning is historical, not an eligible
re-embedding backlog. Hermes3's eight unknown graph owners still lack complete
same-binding withdrawal/retirement proof; they remain excluded, not relabeled
as benign or silently deleted. Accordingly its source-health result remains
`action_required`. Closing these findings requires trustworthy original receipts
or source provenance that the inspected backups do not provide. Preserve the
backups and existing history; do not fabricate a green diagnostic.

## Recovery artifacts

Private artifacts remain on Afrodite in the respective instance's
`/home/node/.hermes/backups/` directory:

| Instance | Final full private rehearsal | Live backup, code inventory and repair receipt |
| --- | --- | --- |
| Hermes1 | `hymem-release-v61-rehearsal-_02ukigv` | `hymem-live-v61-hc665g4k` |
| Hermes2 | `hymem-release-v61-rehearsal-a1hytg93` | `hymem-live-v61-iq6agm_t` |
| Hermes3 | `hymem-release-v61-rehearsal-iaqmy109` | `hymem-live-v61-52miqspc` |

Each live artifact contains `baseline.sqlite`, `report.json`,
`code-before-manifest.json`, `code-after-manifest.json` and recoverable prior
release files. These schema-59 baselines are rollback evidence, **not instructions
to overwrite a now-active schema-61 store**. A recovery operation would need
fresh writer exclusion and an explicit plan for all newer data.

The frozen 59-member release archive SHA-256 is
`682135d304746ef67c4dcb75e12b625142704d4771eb1d2879129e42172652c3`.
Local reviewed scripts, manifest and test receipts are retained under
`/private/tmp/hymem-audit-review-root.WHciPj2Y/` and the referenced root XML files.
This final report was written after deployment; it does not alter the frozen
runtime bundle or imply that operational credentials were copied locally.
