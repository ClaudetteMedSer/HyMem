# Production claim-observation conflict repair

User authorized the proposed clone-based diagnosis and fix after production
dream 1406 reproduced `same-generation claim observations disagree` on R7.
Startup/doctor/read-path checks do not establish successful dreaming. The
existing sample-eight benchmark remains an independent, frozen run and must
not be changed, resumed or rerolled as part of this repair.

## Sequential gates

1. Preserve a consistent private copy of Hermes1's current store on Afrodite;
   inventory exact runtime/configuration and last failed publication. Export
   only counts, hashes, error classes and semantic field names, not private
   source text, keys or raw model replies.
2. Reproduce on that copy with no provider calls where possible. If the failed
   incoming result was not retained, isolate only the earliest eligible source
   unit and capture a bounded replay on the server. Do not burn another entire
   production dream just to recover diagnostics.
3. Have a separate Sol agent implement the evidence-backed fix in an isolated
   candidate. Preserve atomicity, provenance, current/history distinctions and
   the low-level disagreement guard; do not fabricate successful publication,
   silently discard contested evidence or change generation merely to evade it.
4. Parent review and independent positive/negative regressions, adjacent suite,
   and private-store replay. Verify that unrelated chunks can progress and
   unresolved claims remain explicitly visible. Require exact source pins and
   cleanup/accounting for any paid replay.
5. Only after those gates pass, perform a backed-up, scoped Hermes1 deployment
   and bounded production verification. Record actual dream outcome separately
   from health checks and from benchmark results. Keep rollback material.

Initial state: deployed schema 63, healthy Honcho/MCP/read paths; failed dream
1406 on 2026-09-25 14:09:39–14:15:22 UTC. At 16:00 UTC the backlog was
294 chunks / 30 digests / 43 profiles / 44 facts, with no quarantined categories.
No repair or additional live dream has been performed at plan creation.

## Progress and evidence

- The independent R7 sample-eight benchmark finished: 8/8 first attempts,
  0 failed/missing, 6/8 correct (75%). Offline scored-artifact validation passed.
  All eight questions explicitly had degraded/missing summaries (85 sessions
  across their separate stores). Usage: 6,348 completions, 6,349 HTTP attempts,
  20,328,150 tokens; no verified dollar-cost figure. This is neither a full-500
  readiness proof nor an official comparable score. See
  `../patches/2026-09-25-r7-sample8-result.json`.
- Private immutable production snapshot remains on Afrodite only, SHA256
  `0db99a70b01b20852ae08ebd1726be01f57b64651e8fd85f5e9ab29b1b3c6b85`.
  First pending chunk is `chk_12238afe485d5b2f4e7975b0d20b096ec20414d9`,
  sourced from messages 902/903. It has an older-generation publication but
  no observation for those source IDs in the current generation.
- A Sol agent reproduced an actual algorithm defect offline: semantic dedup
  can route distinct, differently qualified claims from one source/response to
  one edge, after which the unchanged observation guard correctly rejects it.
  Disabling dedup permits both claims. This mechanism was not yet established
  as the cause of the particular production failure at this stage.
- Candidate `hymem/dreaming/phase1.py` SHA256
  `31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136`
  screens only optional semantic dedup targets against existing same-source,
  same-generation semantics. A conflicting optional merge falls back to the
  candidate's own canonical edge. The exact canonical conflict guard remains
  fatal and atomic. No evidence deletion, history authorization, generation
  bump, or fake successful marker is introduced.
- Independent reviewer: 89 tests passed across provenance/dedup suites plus
  two custom cross-chunk/source-weight controls. Parent: initial 84-test gate,
  another 84-test provenance/generation/concurrency gate, 46-test
  dedup/bitemporal/dream gate, and 11 focused tests in the main checkout passed.
  Suites overlap; these are not additive unique-test counts. Parent added
  cross-chunk staged-pool, different-source, and exact-replay controls. The
  reviewed fix and 11 tests are integrated locally, not deployed.
- Auto-review initially blocked transfer of production text. User then
  explicitly approved this one chunk and its required context to
  `https://api.deepseek.com`, model `deepseek-flash`, maximum 16 completions /
  48 HTTP attempts. No production deployment or dream was part of that
  diagnostic authorization.
- Diagnostic preflight defects were corrected before paid calls: publication
  queries require HyMem's connection-local functions, and an older-generation
  publication must not block a current-generation pending chunk. Receipts are
  retained; both failed preflights made zero provider calls.
- Private capture-v3 succeeded: 5 completions / 5 HTTP attempts, 25 triples,
  12 dedup vectors. Producer generation matches actual production. The source
  snapshot and 479 frozen R7 files remained unchanged; live container exited
  0 with PID 0. Captured JSON and vectors remain private on Afrodite.
- Initial offline replay stopped before comparison because SQLite's logical
  dump requires the existing vec0 extension. Confirmed and corrected in the
  diagnostic clone opener, without a paid retry. Comparison now reuses the
  captured output and vectors with networking disabled. No production change
  or new production dream has been made.

## Verified result

The same captured real-store extraction was replayed against frozen R7 and
the candidate in separate network-disabled containers, using identical source
snapshot, triples, generation binding, and embedding vectors:

| Code / dedup | Outcome |
| --- | --- |
| Frozen R7 / enabled | Same-generation guard rejected an edge merge for message 902; `value_numeric` differed. Logical database digest after rollback exactly equalled before. |
| Frozen R7 / disabled | Published successfully. |
| Candidate / enabled | Published successfully, zero conflicting observations. |
| Candidate / disabled | Published successfully. |

The unchanged original guard therefore detects a conflict introduced by optional
semantic dedup, rather than an inconsistency present at rest. This isolates a
sufficient cause using the actual pending production source, not merely a
synthetic fixture. The old failed dream's exact model response was not retained;
this is a fresh captured response for the isolated suspect chunk, not a claim
to have reconstructed every intermediate state of dream 1406.

Parent post-verification of the candidate/dedup-enabled clone confirmed:

- exactly 1 current publication for the target under the expected generation;
- 25 current claim observations;
- 0 same-generation disagreeing groups;
- 0 evidence-ledger count mismatches;
- 0 foreign-key violations; SQLite integrity check `ok`.

Baseline, fixed replay, and final verification containers all exited cleanly
with PID 0. These checks made no provider calls. Total new diagnostic spend
remains 5 completions / 5 HTTP attempts. All raw source/capture/vector data
stays on Afrodite. The immutable reference and 479-file frozen baseline pins
were preserved; candidate differs in exactly `hymem/dreaming/phase1.py`.

Private receipts reside beneath
`claim-conflict-20260925-zqkdtxky/offline-compare-v1/` on Afrodite.
At the end of the private-clone verification, the implementation and regressions
were local only. Production success was not inferred from that replay.

## Authorized Hermes1 deployment and live verification

After the user approved deployment/restart and one normal production verification
dream, Hermes1 was stopped gracefully, backed up, and updated in exactly
`hymem/dreaming/phase1.py`. No schema, dependency, hook, credential, or configuration
change was made. Hermes2/3 and the embedding server were untouched. This is a
scoped working-tree deployment; no upstream commit or push has been performed.

- The consistent pre-deployment SQLite backup is retained privately beneath
  `claim-conflict-20260925-zqkdtxky/deploy-v1/pre-deploy.sqlite`, SHA256
  `0db99a70b01b20852ae08ebd1726be01f57b64651e8fd85f5e9ab29b1b3c6b85`.
  Its schema 63, foreign keys, and integrity were verified. The original Phase-1
  file is retained alongside it. An initial backup-check attempt used an
  unsuitable plain SQLite connection; it failed before application mutation.
  That stage was preserved, and the corrected HyMem-aware backup check passed.
- Container restart completed at 2026-09-25 16:49:57 UTC. Verified one Honcho
  and two MCP processes, identical effective configuration, healthy HTTP endpoint,
  working MCP profile/retrieval, all 479 source pins with the two explicitly
  allowed overrides (pre-existing doctor fix and new Phase-1 fix), unchanged
  runtime distributions/hooks/wrappers, schema 63, clean foreign keys and SQLite
  integrity. Doctor reported zero failures and two existing warnings: historical
  embedding compatibility and summary context health.
- Exactly one production dream, **1407**, started at 16:52:14 UTC under a
  server-owned supervisor. The worker has a 30-minute cooperative deadline and
  the parent a 31-minute hard limit with process cleanup. No automatic rerun.
  The helper reads credentials only from the running Honcho environment on the
  server; production text and credentials are not exported into this checkout.
- The former blocker chunk has successfully published in the current generation
  with **16 observations** from its fresh production extraction. This is separate
  from the offline captured response's 25 observations. At 16:57 UTC the cycle
  was still active and other chunks were progressing; terminal cycle success and
  store-wide completion remained unverified.
- Additional local checks: 76 tests passed in the combined claim/supervision gate;
  three loopback transport tests were blocked solely by sandbox socket permissions,
  then all three passed outside that sandbox using dummy local traffic, not paid
  provider calls.

Terminal verification must separately establish: (1) owned run and supervisor
finished/cleaned up, (2) all nonfatal failure and budget fields, and (3) fresh full
store status, including malformed state, historical terminal losses, authoritative
pending counts and summary degradation. Target publication alone is not proof
of any of these broader claims.

### Live cycle outcome: target recovered, later failure

Dream 1407 ended at **16:59:14 UTC with a later ValueError**, after 35 newly
committed chunks (processed row count 807 to 842). The original target remains
successfully published. This is **not a completed successful production cycle**.
The runner persists only `execution_failure:ValueError`, and the private worker
receipt records only the exception type. Neither retains the failing call-site
or incoming extraction, so the exact later cause cannot be inferred from that
code alone. No production retry was launched.

Usage was 154 completions / 154 HTTP attempts, 538,296 prompt tokens and 60,651
completion tokens (598,947 total); dollar cost is unavailable. This is separate
from the earlier five-call single-source diagnostic and from the frozen LME run.
The worker shut down cleanly; the supervisor reaped it, verified its process group
absent, and wrote terminal receipts. The dream lease count is zero.

Post-failure checks passed again: healthy Honcho, two MCP processes, profile and
retrieval calls, source/runtime/hook pins, schema 63, database integrity, foreign
keys, canonical drift and lossless coverage; doctor 9 OK / 2 existing warnings /
0 failures. Current authoritative backlog is 259 chunks / 28 digests / 42 profiles
/ 42 facts, plus three source-materialization sessions and one aggregation build.
No quarantine or coverage-integrity failure was introduced. Existing limitations
remain: 1,062 historical terminal-loss chunks, 78 malformed digest states (down
from 79), 82 degraded summary sessions including nine missing summaries, and zero
malformed summaries.

An independent reviewer is testing a potential order-dependent semantic-dedup
collision offline. It is a hypothesis about a remaining algorithm gap, **not yet
an attribution of dream 1407's generic ValueError**. A future paid diagnostic must
capture bounded private replay evidence and content-free traceback locations;
another uninstrumented full production dream is not a suitable next test.

### Follow-up offline defect and private snapshot

The reviewer reproduced a remaining algorithm defect with six synthetic controls:
if a different source already established exact edge B, and the next response
orders alias A (meaning 2) before exact B (meaning 1), optional dedup can occupy B
before its exact item reaches the unchanged guard. Both cached and same-wave
routes fail atomically; exact-first and dedup-disabled controls succeed. Parent
independently added four regression arms: both alias-first arms failed, and both
exact-first controls passed. This is proven offline, not an attribution of 1407.

A new Sol implementation agent is making optional merge selection aware of the
whole batch's exact claims. Initial existing-edge reservations passed 15 focused
tests; independent review is also testing targets created earlier in the same
batch, to avoid accepting a fix that only covers preexisting targets. No follow-up
application change has been deployed yet.

The consistent post-1407 store is retained privately on Afrodite at
`claim-conflict-20260925-zqkdtxky/post-dream1407/source.sqlite`, SHA256
`7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0`.
It has zero same-generation disagreeing groups at rest (a rollback can hide an
in-flight collision). A separate private inspection copy used read-only HyMem
queries; neither raw source nor values were exported locally.

The earliest pending non-baseline candidate in the last successfully publishing
session is `chk_c14182771bd8e2a7e583f7d693cee29a5fb159fe`: 1,298 characters,
messages 1016/1017, zero current-generation observations from overlapping chunks.
This ordering is a diagnostic lead, not proof it failed. The user explicitly
approved capture of this source and required context to DeepSeek (`deepseek-flash`,
16 completions / 48 HTTP attempts), retaining the response privately for offline
comparisons. This approval excludes deployment, restart and another production
dream. A separate Sol agent is adapting the bounded capture harness locally.

The approved capture completed successfully in `capture-next-v1`: **2 completions
/ 2 HTTP attempts**, nine triples and seven dedup vectors. Both preflight and live
containers exited 0 with PID 0, the reference/source pins were unchanged, and the
raw capture remains private on Afrodite. No second production dream was launched.
The original production response was not retained; this is a fresh captured
extraction for the nominated candidate, not a reconstruction claim.

The follow-up local implementation now reserves canonical natural identities
(subject, predicate, object, source session and source message), including edges
not yet present when a batch begins. This closes the three-item mint/alias/exact
case without precreating edges or reordering provider output. SHA256:
`540d257c596850b44adaacab922587b99445def7a7b6b6d7696e0739bbc2e27d`.
Sol: 17 focused and 47 adjacent tests passed. Parent: 76 focused/dedup/replay/ledger
tests passed on the final candidate. Parent subsequently reran the broader
244-test provenance/dedup/bitemporal/producer/replay gate on final SHA `540d257c...`:
244 passed in 153 seconds. The earlier intermediate-candidate run is separate;
all these suites overlap and are not additive unique-test counts.
Independent review: all 18 initial-publication
controls passed, including all six three-item permutations; counts and foreign
keys were correct.

Independent review also identified two **pre-existing** exact-replay routing
failures (cached equal-semantic aliases, and alias-S2 / exact-B-S1 / exact-B-S2
ordering). Both reproduce on deployed first-fix SHA `31973309...`; neither is
introduced by the new reservation. Sixteen other replay controls passed. These
limitations remain explicitly open rather than being folded into a clean-health
claim. Neither is yet attributed to dream 1407. The final candidate is local only,
pending private same-capture comparison; no follow-up deployment is authorized.

### Next-candidate comparison: no reproduction

The approved next capture was replayed under the two pinned fixes in
`offline-next-compare-v1`, using the same post-1407 snapshot, extraction and vectors:

| Code / dedup | Outcome |
| --- | --- |
| Deployed first fix `31973309...` / enabled | Persisted; no observation collision. |
| Deployed first fix / disabled | Persisted. |
| Local natural-key reservation `540d257c...` / enabled | Persisted; no observation collision. |
| Local natural-key reservation / disabled | Persisted. |

Both comparison containers exited 0 with PID 0, networking disabled and no
credentials mounted. No new paid calls were made. Capture SHA256
`491625247ae1837743507268770186fe256fdba45393e46d4d264507ed056239`;
vectors SHA256 `10a0636e9e21eb105448047761313e2286eafd127b7b0cad552962adf557676c`.
The final fixed/dedup-enabled private clone independently passed SQLite integrity,
foreign keys, canonical drift, evidence-ledger counts and same-generation
consistency (zero findings each), with one current publication and nine current
observations for the target.

Replay starts with an empty same-wave pool, unlike the live dream. A zero-network,
metadata-only check compared pre/post-1407 graphs: 247 new edges existed, but none
passed the structural/lexical and authority gates for any of this fresh response's
nine triples. Thus those omitted earlier-cycle new-edge targets cannot affect this
particular captured response. This does **not** recover the lost live response or
prove the nominated chunk was the failing one.

**Conclusion:** the candidate diagnostic is complete within its authorization
(2/16 completions, 2/48 HTTP attempts), but did not reproduce dream 1407. The local
follow-up fixes a separately proven batch-order defect and has no detected new
regression; it is not evidence that the later production failure has been solved.
The original response and traceback were not retained, so neither model variability
nor a particular application stage can honestly be assigned as the later cause.
No further paid capture, deployment/restart or production dream was started.

Next gate, authorized by the user's subsequent permission for all necessary
tests including paid tests: one instrumented dream on a private
clone, restricted to the affected session, retaining failing inputs and safe
call-site metadata; a maximum 64 completions / 192 HTTP attempts and a 15-minute
deadline. Stop on the first unexpected failure or cap, keep raw evidence on
Afrodite, and perform all code comparisons offline afterward. This is not a
production deployment, production dream, or full LME benchmark.

The first campaign retains the proposed 64-completion / 192-HTTP / 15-minute
budget despite the broader test authorization. A separate Sol agent implements
the capture worker; the parent independently implements and checks the server
supervisor and metadata export boundary. Production code, service processes and
the live store are not mounted writable into the diagnostic container. Source
and runtime pins are verified before and after; only the clone's work directory
is writable. First reproduction must be retained for offline comparisons before
any further paid retry.

Instrumented campaign v1 passed network-disabled source/identity preflight, then
stopped before any completion or HTTP admission: the optional exception journal
filled with 222 `GeneratorExit`, 28 `ValueError`, and six other events before the
first provider call. Its saturation incorrectly set the mandatory capture-error
flag. Both containers exited with PID zero; no paid call or application failure
was observed. A separate Sol correction excludes routine generator control flow
and makes optional caught-error truncation explicit without treating it as loss
of the independently retained fatal traceback, LLM replies, or publication inputs.
Actual journal write failures remain fail-closed. Campaign v1 remains preserved;
the corrected diagnostic uses a new v2 directory, not a resumed run.

### Instrumented v2 result: a different fatal boundary, now retained

The corrected recorder passed 19 local controls and the combined 121-test
client/producer/claim gate. Network-disabled preflight and live private-clone
containers both exited zero with PID zero. The diagnostic itself completed its
evidence-capture contract; **the cloned dream failed**, not succeeded.

- 49 LLM completions / 49 LLM HTTP attempts, 12 local embedding HTTP attempts;
  171,153 prompt + 13,111 completion = 184,264 tokens. All counters reconciled.
- 17 extraction/prepublication snapshots retained with their actual in-cycle
  pools, configurations, vectors and committed pre-write databases.
- Failure was `httpcore.RemoteProtocolError` → `httpx.RemoteProtocolError` →
  `openai.APIConnectionError` during `fetch_chunk_embeddings`, not the original
  claim-observation ValueError. No attribution of dream 1407 is claimed.
- The failing catch-all embedding request contained **1,125 texts / 2,675,490
  characters**, with maximum single text 23,426 characters. Application code
  sends every cache miss in one unbounded provider request at this boundary.
- The shared embedding container reports restart count one and a new start at
  17:53:46 UTC; it is currently running. The available lifecycle query returned
  no events and current `OOMKilled` is false, so an OOM cause is **not proven**.
  Do not repeat the oversized batch to reproduce the service interruption.
- Original source and frozen application pins are unchanged. Live Hermes1 was
  not restarted and its production database was not used by this dream. Raw
  evidence stays in `instrumented-dream-v2/work/live` on Afrodite.

A new Sol agent is implementing bounded catch-all embedding requests and
per-batch durable publication. Parent will review/test before replaying only the
captured embedding work against the local embedding service, without more LLM
calls or another oversized request. This is a distinct confirmed unbounded-batch
defect; fixing it must not be represented as solving the still-unidentified
original ValueError.

### Bounded chunk embedding candidate and safety review

A separate Sol implementation introduced keyset chunk-ID batches and short
per-batch commits; provider calls remain outside the write transaction. Parent
review found an oversized-but-already-embedded/cache-hit regression and required
that limits apply to actual outbound misses, not data that needs no provider
work. Source text and hashes are never truncated. Individually oversized remote
misses remain explicit errors and are not marked complete.

Final envelope is **16 texts / 128,000 characters per request**. The earlier
64-text proposal was reduced after inspecting the shared server: 3 GiB memory,
FastEmbed default internal batch size 256 and max token length 512. The isolated
verification adds an independent 512,000-UTF-8-text-byte admission limit, at most
128 internal embedding requests and a 15-minute deadline. No LLM is constructed
or credential loaded for this verification; route pinned to the existing internal
service. A 105-test affected gate and independent real-dream/cache/size controls
passed; the full final parent rerun is recorded separately.

The candidate SHA for `hymem/dreaming/embeddings.py` is
`53713f956efdabdd880dfda7c926db34c37e7e4def6833b7b8866a1af4f74978`.
The runner change is only the import and bounded fetch/commit loop; unrelated
dirty runner edits are preserved and must not be copied wholesale to production.
No new application fix is deployed.

`embedding-batches-v1` on Afrodite uses a sealed consistent postfailure reference
SHA `803efa98c90d154bd3b27c7aa504c25369dd59d88364aa75939c6fddd36f15aa`.
Its frozen 479-file candidate is first-fix R7 with only this embeddings override.
Host SHA `5875852a073228f614a035e4bc866c9a9a0fcb8e274b640dad659100e6905fcf`;
worker SHA `c312dc56a4724d5a825291b0567e6889a2242ca7df36bb69e27759fc1ecedb4d`.
Installation and detached launch succeeded; terminal outcome is not yet recorded.

Terminal verification subsequently **passed**. All 1,125 pending chunk vectors
were recovered on the private clone in 89 bounded internal embedding requests:
993 remote texts and 132 cache hits, 2,438,859 characters / 2,486,809 UTF-8 text
bytes. No LLM was constructed and no DeepSeek request was made. The same 1,411
extraction chunks were scanned before and after; pending count is now zero.
Dimension and producer identity stayed unchanged. Integrity, foreign keys,
canonical consistency and evidence-ledger checks passed. Graph and non-embedding
logical hashes were identical before and after. Both diagnostic containers
exited zero with PID zero; source remained unchanged. The final parent gate was
114 passed (overlapping, not additive with earlier gates).

Parent independently checked the shared services after completion: Honcho and
embedding health both HTTP 200. Hermes1 start time remained 16:49:57 UTC with
restart count zero; embedding server remained at 17:53:46 UTC / restart count one
(no additional restart during bounded verification). The final private clone
SHA is `a9380b76d4e414b252db493bd246f21cf808648ac12d63f85577ffe68efa8f42`.
Content-free evidence is recorded in
`docs/patches/2026-09-25-bounded-embedding-verification.json`.

Independent read-only audit found the same unbounded-request class in background
chunk dispatch, messages (character size), edges, episodes, Phase-1 dedup,
aggregation and reembedding repair. Offline recording clients reproduced these
without network. **Do not run another full live dream after only the catch-all
chunk fix.** The next sequential fix must apply a common bounded request planner
at these call sites, preserve per-response identity/material validation and
existing transaction boundaries, and avoid changing the underlying vector
producer merely for orchestration. The original claim ValueError and two
pre-existing exact-replay binding defects remain open and are not explained by
the new transport failure.

### Shared embedding bounds: parent acceptance and private-dream verification

A new Sol implementation now applies a common lossless request planner at all
audited application dispatches (chunks including background work, exact messages,
edges, facts, episodes, Phase-1 dedup, aggregation including dimension redo,
reembedding repair and query). The fixed tiny doctor probe is the only direct
application-call exception. The existing guarded provider and cache implementations
are unchanged, so orchestration does not fork the vector producer identity.

The planner validates the entire logical miss list before dispatch and preserves
text bytes, order and duplicates. Each request is bounded to 16 texts / 128,000
characters / 512,000 UTF-8 text bytes. It checks deadlines, caller material
guards, cardinality, producer identity, dimension and finite nonzero vectors
before dispatching the next request. Oversized individual inputs remain pending
with an explicit local error; no truncation or invented successful coverage.
Message maintenance does not bisect/resend deterministic local admission errors.

Parent review required corrections before acceptance: count actual split repair
requests rather than logical groups; classify identity-read failures as response
admission errors; preserve query malformed-vector status and report oversized
local input as not attempted. The parent gate passed **587 tests**, plus six
subtests, with one external Starlette deprecation warning. A subsequent query,
message and repair rerun passed 137 tests. Frozen first-fix R7 + embedding-only
patches passed 182 affected tests and a final 102 query/message tests. These gates
overlap; counts are not additive. A full frozen suite is running separately.

Next campaign `shared-embedding-dream-v1` uses the original post-1407 reference
`7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0`,
not a production write or a resumed run. The frozen first-fix tree is retained
with only eight embedding-related file overrides; unrelated dirty working-tree
summary/API changes and the second claim-reservation fix are excluded to keep
the comparison attributable. The candidate has 480 sealed files. The diagnostic
independently enforces the same request-size bounds before transport, and retains
raw inputs/replies only on Afrodite. Limits: 64 completions, 192 LLM HTTP attempts,
512 internal embedding HTTP attempts, 30-minute cooperative deadline and
31-minute host wait limit. Stop at first unexpected failure or cap; preserve
captured evidence for offline replay. No production deployment/restart is included.
The parent rerun of the diagnostic controls passed 19 tests and 17 subtests.
As with prior controllers, a host-level kill of the detached supervisor is not
covered by its Docker wait/stop cleanup; the worker also has its own deadline.

Staging succeeded with host SHA
`ca95bc8d06cc1cc9f84e80dcb91f13cdc333c681efa4e9aca5ae457d48cce1f0`
and worker SHA
`9d6f03ef88efd01c40fd94affc9b31dd7b6b6e879066ce784f08e74600f0b863`.
**Launch was rejected by the tool safety review before execution**: the general
paid-testing permission was not accepted as specific authorization to transmit
production-derived text from this targeted dream to DeepSeek. No new paid call
or diagnostic dream started. The user has been asked explicitly about this
payload/destination/campaign; do not retry or indirectly launch without that
approval. Offline tests continue. A separate reviewer added 27 diagnostic
controls, bringing the combined controller/worker gate to 46 tests and 17
subtests. Campaign `completed` means evidence collection completed; it is not
proof that every dream phase succeeded, so eventual report counters and store
checks still need separate inspection.

Read-only server confirmation returned `installed_not_launched` after the
rejection. No launch intent or paid diagnostic was issued. Full-suite execution
continues locally against
`/private/tmp/hymem-shared-embedding-20260925.X8jEOF/candidate`, with JUnit output
at `/private/tmp/hymem-shared-embedding-20260925.X8jEOF/full-suite.xml` and current
terminal session 76493. It had reached 10% without failures at this checkpoint;
do not report a final full-suite pass until its terminal result is collected.

### Renewed authorization and retained alias-registration failure

The user renewed permission after the explicit production-text notice. The same
staged launch was approved by tool review and ran once, with no production writes.
Offline preflight passed. The live private dream stopped on its first unexpected
failure after **2 LLM completions / 2 LLM HTTP attempts / 1 internal embedding
request**, 7,149 prompt + 552 completion = 7,701 tokens. Both diagnostic containers
exited zero with PID zero, and accounting, cleanup and source pins passed.
The worker status is `captured_failure`, not successful dream completion.

This failure is now retained: `ValueError` in `canonicalize.register_alias`, at
its owned-state guard, called from `_upsert_triple` while registering the object.
One complete extraction (nine triples) and its pre-write snapshot were captured.
Private metadata inspection identifies triple index seven: its normalized surface
already maps to the same terminal normalized canonical target. Historical state
still references the alias key (239 mentions, three types, five graph objects,
one graph subject). Re-registering the unchanged alias raises even though no new
mapping is proposed. The same pattern was independently reproduced synthetically.
This is an identified failure of the current diagnostic; the original 1407
response was not retained, so exact historical attribution remains unavailable.

A new Sol agent is implementing only idempotent registration: after validating
input, normalized target, absence of target chaining, and absence of retargeting,
an exact existing mapping returns without DML. New aliases over owned identities
and conflicting remappings remain prohibited. Historical ownership is retained,
not silently rewritten or reported repaired. Parent will verify the retained
response against baseline and candidate offline before another paid dream.

Captured evidence digest:
`88791503f8404076eed1ebdf60ef8b61d5553f42b543bca9ccb94a9783eea717`.
Live container: `cdbedf81c439ce817de669f4185e02ffc58e8c23ad9c96ce2e128c57bbb40396`.
Offline container: `4bee6924c47f63f1d48c7e449b6d1e12a26268c1289e372e0bbc9e7fa4adaa3d`.

### Alias fix review and offline checker correction

Alias registration now treats an existing unchanged mapping as a no-DML no-op
only after validating its normalized terminal target and rejecting retargeting.
Candidate canonicalization SHA is
`6fc1f95f5945ef28a99429e0ad9336a3ea4b5c28d85c22babff0747f792018c0`.
Parent regression gates passed 317 tests and a separate 61-test Unicode,
canonical-drift and bitemporal gate. These overlap earlier gates; they are not
additive. No application deployment has occurred.

The networkless `alias-replay-v1` comparison retained a strict failed receipt.
Baseline rejects the captured response at the alias owned-state guard with an
unchanged logical database, with semantic dedup both enabled and disabled.
The fixed candidate persists successfully, and its exact repeat leaves the
logical database unchanged; integrity, FK, canonical, ledger and same-generation
audits are clean in both arms. However, the harness reported publication zero
and correctly refused to authorize a subsequent paid dream.

Independent review and read-only checks of both private fixed databases prove a
diagnostic namespace error: the capture records the public prompt label, while
`persist_chunk_results` derives `extraction_cache_key(label)` before writing.
The harness queried the public label. Both actual databases contain exactly one
matching claim outcome, auxiliary outcome, processed binding and current-view
publication under the derived key; that key matches the captured generation.
The same distinction was reproduced with synthetic source-valid publication.

A fresh `alias-replay-v2` must use the derived key, cross-check it against both
captured and registered generation bindings, and retain the existing 0→1→1,
logical-repeat and integrity gates. The failed v1 receipt is not overwritten.
This correction changes diagnostic code only, with no new provider calls.

Full frozen suite remains in progress. Five socket-bind failures in
`test_ensure_embedding_server.py` passed on a permission-enabled rerun (7 tests);
the remaining failure/error reports still require terminal-output review.

The parent reran 58 diagnostic controls successfully, then installed and launched
`alias-replay-v2` with no network access. It **passed all strict verdicts**:
baseline rejects with unchanged state; fixed publishes 0→1 and exact replay stays
at one with unchanged logical state, in both dedup-on and dedup-off arms. All
before/after audits are clean. Both containers exited zero, PID zero, no OOM.
No provider calls were made. This closes the retained-response alias regression.

V2 worker SHA: `7b94a69f2a152f1a818ab7b5833b8d619734b3f7023de2dca89b4190f9013085`.
V2 host SHA: `6e740e40e6a64129acbc8c2bd002d10fe28bcce377081efef039ed967c006d2d`.
Baseline container: `48edef1bb458b102b19b27219694848eaf8146f6e11fc9d935b7ff98cacf7af4`.
Fixed container: `c0a5744ca03857e7927c6d5cab0facd403bc513442b760670c1b3cf341af2903`.

The separate known mutable-routing exact-replay proof gaps remain open; this
narrow alias fix does not claim to solve them. Read-only Sol review recommends
an order-sensitive validated-input fingerprint with schema, portability and
historical-fallback guards, not accepting a replay solely from routed outcomes.
No implementation of that broader change has been mixed into this candidate.

### Frozen full-suite result and next private-dream gate

The frozen embedding candidate's complete suite finished in 2,116.79 seconds:
7,615 passed, four skipped, six failed, 16 errors. Twenty-one failure/error
cases were `PermissionError` when binding ephemeral loopback test servers.
Permission-enabled rerun of both affected files passed all 23 tests; no source
change was required. The remaining assertion expected a single 36-text fact
embedding request, which conflicts with the new intentional 16-text limit.
Sol updated only that assertion to require `[16,16,4]` and exact ordered equality
of all 36 texts against their fact IDs, preserving scan/cache/CAS checks.
The focused test passed independently in both working and frozen trees; parent
also passed the working-tree alias+fact focused gate (22 tests). The full suite
was not rerun end-to-end after this test-only correction.
The parent then passed all 40 frozen fact-authority and embedding tests. All
original non-passing cases are now accounted for by successful targeted reruns;
the four skips are declared raw-client deadline/SDK-seal non-applicability.

Parent verified 78 next-stage diagnostic controls. `alias-dream-v1` installed
successfully with host
`a0f2c7c962185830e788aca6ddfe83bbda128cc91e6456bd627ff45f35dda671`,
the unchanged instrumented worker `9d6f03ef…0f0b863`, and the successful v2 replay
receipt SHA `817ca30456d8f64247139413428b975b8946b1958c2a1f6f2cc3a69700cb73d4`.

**The next paid launch was rejected before execution by tool safety review.**
It requires explicit authorization for pending production-derived session text
and necessary context sent to `https://api.deepseek.com`, `deepseek-flash`,
despite the renewed general paid-testing permission. The user was asked about
this exact payload/destination with caps 64 completions / 192 LLM HTTP attempts /
512 internal embedding requests / 30 minutes, no production writes or restart.
A subsequent read-only status confirmed `installed_not_launched`. Do not retry
or launch indirectly until the specific approval is provided. Offline work
continues; no additional paid calls were made in this continuation.

### Next sequential offline fix: locally authenticated exact replay

With the alias fix independently accepted, Sol added two deterministic
source-valid regression cases. The parent independently confirmed both fail on
the current guard: a previously published alias loses its optional vector-cache
route, or an ordered alias/exact batch is replayed with a fresh same-wave pool.
Both first publications are structurally valid. Recomputing their old routing
from current mutable state incorrectly changes the expected observation hash.

The next fix is in progress, not accepted or deployed: schema v62 adds an
optional **local-only** ordered-input receipt on a successful claim publication.
It binds full chunk/source identities, source bytes by hash, ordered triple
fields, effective evidence weights, prompt/generation and published observation
hash. Existing authority and clock checks remain prerequisites. A matching
receipt avoids redoing mutable routing; mismatch or NULL uses the existing
conservative path. Legacy rows get no invented receipt.

Do not add this bare fingerprint to portable authority: the destination cannot
prove original order/alias multiplicity from collapsed observation rows.
Export omits it and import clears it. Semantic merges invalidate it. Ordinary
reopen preserves it only when the bound outcome hash remains unchanged; the
next receipt comparison still checks the complete incoming source binding.

Review specifically covers pre-v62 backfills invoking the refresh helper,
old-schema bootstrap guards, v41 table-shape repair, repeated reopening,
coalescing merge survivor invalidation, and import rollback. The parent has
already passed 25 independent input-binding controls; integration/migration
acceptance remains pending. This patch is separate from the sealed schema-v61
`alias-dream-v1` candidate, which remains installed but not launched.

Read-only production checks during this work: Honcho and embedding health both
200, Hermes1 restart count zero / same 16:49:57 UTC start, embedding-server
restart count one / same 17:53:46 UTC start; neither service was restarted by
these tests.

The first broad parent gate caught 17 historical-migration regressions in the
draft v62 implementation (absent sparse-fixture domain and supported early-v41
table reconstructions), plus a test/implementation disagreement over the
missing-field repair policy. These were not production failures. Sol corrected
domain preflight and the existing lossless table-repair path: missing nullable
proof may be restored on supported historical shapes, but old rows stay NULL;
malformed constraints are rejected. Pre-v62 refreshes and bootstrap guards remain
compatible with tables that do not yet have the field. Independent review also
caught and closed survivor-edge and alias-only auxiliary merge invalidation.

The application patch was then frozen. Parent rerun: **346 passed**, plus
**96 adjacent replay-clock/producerless/Unicode/canonical-drift/016-lineage
checks passed**. A further 22 independent controls passed, including forged
proof injection into checksum-valid portable input and four weakened CHECK
constraint spoofs. Diagnostic helper gates passed 10 tests. Counts overlap.

Frozen proof override SHAs:

- `hymem/core/db.py`: `4c1afb8fc144a4300ac38a1deeb14363dc08e49fcd3151e7f687bfda65790ce0`
- `hymem/core/schema.sql`: `1798120e80098b8f8b8defc7da868addd91322f98024d08c4e683d3b087b3d6d`
- `hymem/core/migrations/062_local_claim_replay_proof.sql`: `45c79115922c8b06be4f9861b6d026204befdb28d90609d598f8d7f70bd097c6`
- `hymem/dreaming/canonicalize.py`: `73c1d656aaa471402732690712c9de86f82bea6e4a2bf6fef5677a1d1a4dcc18`
- `hymem/dreaming/evidence.py`: `c921d035c4af5b6d0c5add8cb9eb89069b3dbef4447078470ae77dc0dbb36771`
- `hymem/dreaming/phase1.py`: `1037bf6d62c3981add3702f96d2ce5bd79d8ebc831f15b30f95bb3724cfd7217`
- `hymem/portability.py`: `d50a7bc03dabcdf49cc8d6c55bae1ec17d0f72b4e31ce9b8204eca97d38cab70`

`proof-replay-v1` installed as a 481-file frozen candidate built from the prior
shared embedding tree with only these seven overrides. It includes the already
unit-tested batch-reservation fix in Phase 1, unlike the earlier alias-only
comparison. Parent launched its one network-none private rehearsal. Worker
SHA `014e045de6a5591eb87aabf0760247a69cd4d5a75f5bb7462e2d404caff7a9de`,
host SHA `716c5da9f91e4cbd98cdf2ecc75338c42c4be963a25a7b31bd96ca8d04dcd5e2`.
No production checkout/store or credential file is writable or mounted. It
checks all ordinary application rows across migration, no invented historical
proof, second initialization, publication 0→1→1, initialized reopen, exact-repeat
logical no-op, and all prior integrity audits in both dedup arms. Collect its
terminal receipt before reporting real-store-copy acceptance.

### Release-base correction: v63 must upgrade to v64

**The v1 proof rehearsal failed closed before calling `initialize`:**
`snapshot_schema_not_v61`. Read-only inspection confirmed both the original
private reference and retained prepublication snapshot are schema **63**, not
61. The frozen R7 candidate already contains
`062_independent_summary_frontier.sql` and
`063_private_summary_recovery.sql`. Earlier references above to the sealed
R7/alias candidate as schema 61 were incorrect.

The local proof draft was built against a stale schema-61 workspace. Its new
062 migration conflicts with the existing release history, and overlaying its
whole `db.py`, schema and portability files would remove R7 summary behavior.
Those seven overrides are **not an accepted release candidate**. The local
346/96-test passes do not establish compatibility with the actual R7 release.
The failed v1 receipt is retained; the production store and services were not
modified. Its container exited 1, PID zero, no OOM, network disabled.

Sol is rebasing only the reviewed proof, batch-reservation and alias-idempotence
deltas onto a separate frozen R7/schema-63 tree, using migration **064** and
preserving existing 062/063 summary migrations, fields, recovery and wire
contracts. The dirty workspace will not be overwritten wholesale. Acceptance
requires independent delta review, R7-based regression tests, and a fresh
network-free `proof-replay-v2` rehearsal of 63→64 before a full-suite or live
diagnostic launch. The explicit production-text diagnostic approval remains
pending; no new paid calls have been made.

### Corrected R7 candidate and private-copy audit discrepancy

The corrected application is isolated at
`/private/tmp/hymem-r7-proof-v64.kEJ30D`; its only differences from the frozen
shared-embedding R7 application are the six reviewed code/schema files plus
`064_local_claim_replay_proof.sql`. Existing 062/063 files are byte-identical.
The parent independently reviewed those diffs and passed **105** replay,
producer-binding and R7-lineage controls, then **500** migration, portability,
claim-history, canonicalization, alias and R7-summary controls. Counts overlap
earlier gates. A new independent test preserves a real synthetic in-progress
summary-recovery walk, and another proves rejected migration64 rolls back both
the column addition and version stamp. The application delta is retained in
`docs/patches/2026-09-25-r7-v64-local-claim-replay-app.patch`.

Fresh `proof-replay-v2` installed and ran with networking disabled. Worker
`8de6f54cc98abaaf5ac7f328ce2b4e4c5547128e580f3f0ecd82f8d6d353ec82`,
host `e16d27c3cd989b2664a74de19bb524ae685757a9f8fc19e4452e3b3c1d5b6bce`.
The schema63 precondition passed and the private copy upgraded to64, but the
worker stopped **before publication** at `upgrade_semantic_or_proof_drift`.
Container `4afaacb49818da2ecdd760ff9d99f8775a4051901699c0fc113b02a3a614e503`
exited1, PID0, no OOM. This is not an accepted replay gate.

A separate read-only comparison using fresh SQLite connections found **zero
different persisted ordinary tables**: every row count and ordered/unordered
row-content hash matches the source after excluding only the new proof column.
Schema changed63→64; historical non-NULL proofs remain zero. Therefore a
same-connection audit/virtual-table classification discrepancy is under
investigation; data corruption or application migration drift has not been
established. Preserve the v2 failure receipt, keep application bytes frozen,
and prove the discrepancy before correcting the checker or attempting a fresh
replay. No new paid calls, production writes or service restarts occurred.

The parent and reviewer independently reproduced a diagnostic false positive:
opening SQLite's schema before loading sqlite-vec initially classifies its
physical storage tables as `table`; an unrelated ALTER reclassifies some as
`shadow`. The v1 digest selected only `table`, so unchanged persisted rows drop
out of its audit. A fresh same-connection audit on the retained private copy is
being prepared to confirm this exact mechanism before any gate correction.

The unaccepted workspace-only v62 proof draft has now been reversed using the
reviewed proof-only patch; the four files that were clean before that draft
are clean again. Existing bounded embeddings, natural-key batch reservation
and alias-idempotence edits remain. The wrong new062 migration and four
proof-only workspace tests were removed, **not user data or backups**. The
corrected tests remain recoverable byte-for-byte under
`docs/patches/2026-09-25-r7-v64-tests/`, with a scoped test patch alongside the
R7 application patch. The isolated R7/v64 candidate and remote sealed copy are
unchanged. Do not deploy the stale local schema61 tree as the R7 release.

`proof-audit-v1` then **confirmed the diagnostic false positive on the actual
retained snapshot**, not just synthetic data. Exact and instrumented arms
reproduce identical old digest changes. Fifteen `vec_*_{chunks,info,rowids}`
tables switch `table`→`shadow`; all **118 physical main tables** have identical
ordered and unordered row hashes, with zero changed tables. The only expected
column addition is the nullable local proof, with zero non-NULL historical
values. Source snapshot is unchanged; container
`bc8e8c2195ec902190888e4852cbce0c631d97ff75eed7656b9c2476e6a772f3`
exited0, PID0, no OOM, networking disabled. Private metadata SHA:
`50b3bc284de59422573bcb635231d9365dbfe8d8eef5366ac864aee783ca5402`.

Fresh replay v3 will correct only the audit: include physical main tables
classified as either `table` or `shadow`, still excluding virtual projections,
temporary state, schema version, and only the new outcome proof field. Real
ordinary and shadow-table row mutations must still fail controls. All strict
migration/publication/reopen/exact-repeat/integrity gates remain. Application
bytes stay unchanged, and failed v1/v2 receipts stay immutable.

Workspace cleanup verification passed **91** embedding, batch-reservation and
alias tests. A fresh production health-only check still reports Honcho200 and
embedding200, original start times and unchanged restart counts (Hermes1 zero,
embedding-server one). No production dream or restart was performed.

### Retained-response acceptance: corrected replay v3 passed

The parent reviewed and ran fresh `proof-replay-v3`, using the **unchanged**
481-file R7/v64 candidate. The only diagnostic change is the proven stable
physical-table audit; real ordinary and shadow-row mutations still fail its
controls. Both dedup-on and dedup-off arms passed:

- Source63→64 preserves all audited physical main data; all 247 historical
  outcomes retain NULL local proof.
- Second initialization is a logical no-op.
- Retained response publishes0→1, survives close/reopen plus initialization,
  and exact repeat stays1→1 with unchanged logical state and proof.
- Every before/after audit: integrity OK, zero foreign-key, canonical-drift,
  ledger-count and same-generation conflict findings.

Container `994b1d564811af3c457d76da50e351aa31095f20ce14cac5a16b51fa42c76d96`
exited0, PID0, no OOM; no networked or paid calls. Successful result SHA:
`6e602bae89aee5f42be1f3f7efff87ba905a59bc134d09015ffe7a02a672c52d`.
V3 host SHA `a3ddbf16ce89c35f6f6a7f44c86a1089efbbc21abfab64dcc7d073b80772d92c`;
worker SHA `173e9cc13c7af1b6346100892dfb24c9cd84d1a36f69291db90dc6296e360554`.
Full frozen-suite verification on these exact application bytes is the next
gate, followed by the separately authorized private paid diagnostic if its
specific production-text transfer approval is supplied. No deployment yet.

### Full frozen suite launched headlessly

The first suite controller (`proof-pytest-v1`) rejected its own inventory
precondition before creating a test container: the source-pinned baseline has
241 test files / 235 test modules, not 246 files. The incorrect count included
five supplemental regression files from the local staging tree. Application
hashes were unchanged; no tests ran in that rejected stage.

The corrected `proof-pytest-v2` preserves the exact 481-file application
manifest and the exact 14-file test overlay. Its baseline inventory now pins
the verified file count, module count, and complete test-manifest hash. Parent
review and 53 diagnostic controls passed; local collection found 7,715 cases.
Remote installation succeeded, and one detached supervisor was launched
(PID4031366). This is **running, not a test pass**. The container has networking
disabled, no production store or credentials mounted, read-only candidate and
runtime mounts, and a bounded private writable work directory. Collection
must succeed before its single full-suite invocation. Controller SHA:
`3b4a1180dc50e176ea1964534f2562b4cc1a60425e04485ccc056f747b0f5a5e`;
test-overlay SHA:
`978b687ae717fd4a5166b7710715105f4ab092dfab9ba6d19e886b9391b65247`.

Remote collection then passed with exactly **7,715 cases**, zero collection
errors. The full run is active in container
`e50be3864cf98dea531f00a9c15bcd288e76e50c7e4446557bd0ca20c0884cb4`;
last metadata-only observation reports9%, still running, no OOM. This is not
a final suite result. Parent additionally passed91 diagnostic checker controls
and43 paid-harness controls (plus6 subtests), with no provider calls.

An independent reviewer probed source-object mismatch handling. Forged
internal `source_validated=True` objects can reach the pre-existing semantic
fallback in both baseline and candidate, but no normal producer/cache/import
constructor supplies those forged objects, and all3 reproduction arms leave
the complete durable dump unchanged. Changed role or claim semantics is
rejected. Parent independently passed all4 controls; no reachable new
application defect was demonstrated. The frozen candidate is unchanged.

The fresh `proof-v64-dream-v1` adapter is prepared locally only. It uses the
successful proof-v3 application and will require the actual successful
proof-pytest-v2 result hash before installation. Its production-text transfer
approval flag also remains false. It has not been uploaded or launched;
there are no new paid calls, deployment, production-store writes or restarts.

### Separate compatibility follow-up: cold imported semantic routes

The reviewer extended the check to a real synthetic export/import workflow;
parent reran all **6 controls** successfully. Complete historical same-producer
outcomes with NULL local proof remain in `current_phase1_publications` and are
skipped without provider calls or durable changes. However, a **cold import**
intentionally clears local proof and processed gates and carries no
`edge_embeddings`. The normal extractor can then return the same originally
semantic-routed claim, but persistence without the old route chooses a new
natural edge and correctly fails the unchanged outcome guard with
`same prompt generation claim extraction outcomes disagree`. The whole
transaction rolls back, with identical before/after dumps and clean FKs.

This is not a regression introduced by v64, but it is a confirmed remaining
instance of the replay problem outside the local-receipt guarantee. It must
not be reported as solved by the retained local replay acceptance. A Sol
design review is now scoped to a minimal, unambiguous reconstruction from
validated durable observations and exact interpretation/source identity,
without inventing portable proofs or relaxing outcome/observation guards.
No implementation has begun; the application under the active full-suite
run remains immutable. The regression is kept only in
`/private/tmp/hymem-v64-review.bKSzqF/test_review_controls.py` pending promotion
to a corrected regression in the next isolated candidate.

Design review recommends a read-only **NULL-proof-only** replay matcher after
the existing prompt/generation/outcome/clock/authority checks. It must validate
the complete incoming chunk/source objects against durable coverage, then
require a unique one-to-one mapping from incoming exact interpretations to
this chunk's persisted observations and their cited evidence, including
surface/typed/source fields. Reconstruct and compare the full authoritative
result hash; never infer a nearest semantic edge, create/backfill a local
proof, or relax the same-generation guard. On success, only the existing
idempotent auxiliary publication path may restore the processed gate.

Next sequential implementation gate: collect the current full-suite result;
then use a fresh Sol implementer on a separate copy, with a failing-first
normal cold-import regression and changed-source/claim, ambiguous mapping,
damaged-authority, empty-result and no-proof-invention controls. Parent must
independently review and rerun those controls before sealing a new candidate.
The currently running suite and all existing receipts remain immutable.

### Continuation authorization and sequencing

After the parent explicitly described the production-text transfer, provider,
model, 64/192/512 request limits, 30-minute deadline and exclusion of production
writes/deployment/restarts, the user replied: "You are permitted to run any and
all tests required to fix the issues, including paid tests. Continue please."
The parent treats this contextual response as approval for that **specific
bounded** diagnostic, not authority for an unlimited transfer or deployment.
The full-suite success gate still applies; no paid launch has occurred.

A fresh Sol agent (`sol_cold_import_replay_fix`) is preparing failing-first
tests in a separate temporary stage. It is explicitly restricted to tests
and staging until the current full-suite gate is collected. Application
changes, including the frozen candidate under test, remain prohibited there
until the parent opens the next implementation step.

The parent independently confirmed the new normal cold-import test fails at
the expected unchanged outcome guard; four existing authority/empty controls
pass. The expanded Sol matrix now has31 cases, including SQLite-authorizer
claim-write detection, ambiguous/extra observations and evidence/clock damage.
No application bytes changed while preparing those tests.

Sequencing adjustment: the full suite is CPU-active (not stalled) and remains
the **live-test/release gate**, but the isolated next implementation may now
proceed. The prior fix has already passed the parent's605 overlapping targeted
controls and the strict real retained-response replay/migration/reopen gate.
That fix-specific verification satisfies the sequential review step while the
broader frozen-suite run proceeds independently. The user was explicitly
informed. Sol is authorized to change only its separate stage's `phase1.py`
and new regression tests; the application under the running suite, production,
main dirty checkout and all sealed receipts remain untouched. The next
candidate still requires independent parent review and regression execution.

### Cold-import follow-up: parent boundary test and correction

The first isolated implementation passed its 31 new cases, but the parent
found a legitimate overlap case it rejected: an unchanged immutable evidence
revision can originate under v14 while an overlapping chunk's current
observation/outcome belongs to v15. Requiring the evidence's *origin* prompt
to equal the current observation's prompt was too strict. This was reproduced
through normal publication, export/import and extraction, not fabricated
invalid rows. Sol removed only that incorrect equality; observation/outcome
prompt and generation checks, source/clock/result authority and unique exact
citation matching remain required.

The parent independently reran **117 passing cases** on corrected phase1 SHA
`bc47739973a7d5c4825505f83486951b11e6b1ca0d4eeec8ab450dd9fc3272ac`.
The new parent regression covers dedup both on and off, verifies zero claim
table DML with SQLite's authorizer, restores only the imported publication
gate, preserves NULL local proof and checks foreign keys. Broader migration,
portability and history tests plus a separate independent review are pending.

The earlier frozen full suite remains unchanged (51% at the latest
metadata-only check, no failure markers; not yet a pass). A separate offline
suite adapter is being prepared for the new phase1 bytes and two additional
test modules. It must pin the new inventory explicitly: the old candidate's
suite and retained-response receipt cannot be represented as having tested
the cold-import change. No new paid calls or production changes have occurred.

Cold-import fix-specific acceptance is now complete: parent reran **495**
migration/portability/evidence/history/canonicalization/summary cases and the
independent reviewer's **4** normal-path retired-evidence controls, all passing.
Together with the117 focused cases these are616 executions (some coverage
overlaps), not a replacement for the full suite. The reviewer found no new
unsafe acceptance path. Missing or collapsed historical surface evidence still
fails conservatively rather than inventing a route.

The accepted delta is archived as
`docs/patches/2026-09-25-r7-v64-cold-replay-app.patch`
(SHA `09387f027f607328ca9c9afa12f664135fbc2544a07175df565fa79c121ee7fb`),
relative to the frozen v64 candidate, alongside the31 Sol tests,2 parent tests
and4 reviewer tests in the matching `-tests/` directory. Only `phase1.py`
changes in the application. No schema/wire/guard changes. Main checkout and
the earlier Afrodite candidate remain unchanged. Fresh health-only reads:
Honcho200 and embedding200.

Remaining acceptance gates are deliberately distinct:

1. Collect the original frozen v64 full-suite terminal receipt.
2. Run the final cold-import candidate's own complete offline suite and the
   retained production-response replay on its exact code bytes. A baseline
   receipt authenticates ancestry only, not execution of the new code.
3. Run the approved bounded private DeepSeek diagnostic only after its exact
   candidate's offline gates pass. Inspect the inner dream result, caught
   exceptions, accounting, cleanup and audits; an outer supervisor completion
   is not an application pass.
4. Report the resulting evidence and any remaining blocker. These private-copy
   tests do not themselves authorize production deployment/restarts or certify
   a full500-question canonical LME baseline.

Parent review of the new offline-suite adapter caught two setup issues before
upload: local preparation selected a remote-only base-runner path; and the
inherited baseline replay check compared the old receipt against the new
candidate hash. Sol is correcting both with inherited-gate integration tests.
These are diagnostic setup errors, not application regressions; the already
running suite and every sealed receipt remain untouched.

Both launcher errors now have failing-first tests and are corrected. Parent
reran59 launcher controls before and after sealing the exact adapter, then
prepared and uploaded only code/tests. Fresh `cold-replay-pytest-v1` installed
and launched headlessly (supervisor4080859, container
`9f4d40a740f0e44588a8496c41f77dd41d9efe1c6eb2712de4bf16e8f01a3ae6`).
Collection is **7752, errors0, exit0**; the full invocation is now running.
Final candidate481-file manifest:
`ed889d8970c6d7827315de34342996ad55182773c238922fbbdf8510d73c43a4`.
Sealed host `f66e270d7ac632791861ecbc8cc4657226b3acdc86aea7c8f8d8e51c8b0d0bf2`;
17-file overlay `0321fd5419d8079a2cb6ea9802ef4c64df0294a3d00dc7f4204b237b67608344`.
Original14-test submanifest remains identical. Because the original suite's
observed runtime is longer than expected, the **new** offline run has a
10800-second test bound/11200-second host bound; no existing run was altered.
Networknone, no credentials/private-store mounts, only `/work` writable.
Original v64 suite was74% with no failure markers at this point.

### Final-candidate retained replay passed

Fresh `cold-replay-proof-v1` reused the unchanged, pinned v3 worker against the
final481-file candidate, not the parent candidate. Parent reran45 controller
controls before and after sealing. Both dedup-on/off arms passed migration,
publication, initialized reopen and exact-repeat gates. All247 historical
outcomes kept NULL proofs during upgrade; physical-data digests before/after
upgrade matched; second initialization and exact repeat were logical no-ops.
Every integrity/FK/canonical/ledger/same-generation audit was clean.

Parent then independently revalidated installed source inventories, the result
verdict and actual terminal container state: exited0, PID0, noOOM, networknone.
Container `881718aad0c55fa0f4f4a0a1af449020ec7f6b756f555e8b2cf0ca296d2aeb89`;
resultSHA `7dd4482487f4bb99700a7fc75a27b59dc1dae898b39cae6b18c8e71da184dcc1`;
installSHA `0ca4216f3bb89c98e1be292eb34ad2785689dd5ca1b3283c9b441f31fa2db740`;
hostSHA `7afdb9067ca1483a6f17d42431c737ab78f49a37bd2543bdb52c06014a7a3586`.
No provider calls were made; raw evidence remains private on Afrodite.

The never-uploaded paid adapter has been retargeted to `cold-replay-dream-v1`:
only the final code will receive the approved paid diagnostic. Its final-replay
result is now sealed, but the final full-suite result is still unset and blocks
upload/launch. Parent passed86 admission/instrumentation controls plus6 subtests,
including rejection of wrong-candidate receipts, failed results, claim churn,
running containers and incomplete test accounting. Request/model/payload limits
are unchanged. The original old-v64 paid prototype will not be launched.

### Full-suite test isolation finding (not an application regression)

The original full suite showed one failure marker at81%. Mapping the progress
stream to its pinned7715-node collection identified
`test_unmodified_current_runtime_refuses_historical_application_cli`. Parent
reproduced its direct-script child under the exact clean Linux test environment:
exit1, empty stdout,196-byte stderr, `ModuleNotFoundError` for `hymem`, before
the application guard ran. Both local candidates passed that test because the
local installation supplied an ambient package. The neighboring rehearsal test
already used a checkout-pinned `current_cli_runtime` fixture; the application
test did not.

A fresh Sol agent changed **only this test** to use that existing fixture,
`-s -S -B -P`, explicit source-only PYTHONPATH and isolated cwd. Parent passed
all48 application/rehearsal controls, including poisoned inherited Python
environment checks. A read-only/synthetic Linux check of the corrected command
produced the exact expected guard refusal, empty stderr and zero writes or
created artifacts. Application helper/schema guard bytes are unchanged.
Fixed testSHA `14c63215f382f7f2b27b229fce79b9492b3794e20aa0cbdc80ef7ced61e8501f`
is archived in the cold-replay tests directory.

Rather than finish another run with the known broken test setup, parent
intentionally stopped only `cold-replay-pytest-v1`, recorded `parent-stop.json`
and preserved all logs/receipts. Container exited143, PID0, noOOM; its failed
terminal receipt is **not a pass**. The original older suite continues to
collect its remaining results.

Fresh `cold-replay-pytest-v2` changes only the test overlay (17→18 files,
original17 submanifest identical), stage identity and a stronger overlay check.
The final application manifest remains `ed889d...43a4`. Parent reran124
launcher/admission controls before and after sealing, installed and launched
one network-none full suite (supervisor4099375). HostSHA
`25c9de04b32f2e6a095a5860d3ab0fd3a4ba328f880f6d91a894de0d2d6dd227`;
overlaySHA `5bb544abc0df397bf6b284a29d44de11b129d9c17d86b6740e04857764fc37e7`.
Paid adapter now waits for this exact full-suite result; parent again passed
86 controls plus6 subtests after retargeting. No paid calls, deployment or
production restart occurred during this correction.

### Original-suite terminal evidence and final-run continuation

The original `proof-pytest-v2` run is now terminal: exited1, PID0, noOOM,
networked runs0; resultSHA
`025e197979427f01c6c7f8839778519d4506134d3b04425bf84f5d9570c139ae`.
It ran20:22:03–21:45:43UTC (5019seconds including collection), consistent with
its immutable5000-second full-test bound. Progress ended89%, one failure marker,
no error markers, and no complete full-counts/worker-result receipt. It is an
incomplete failed run, not a pass. The one failure was independently reproduced
as the child-import test setup issue above; this does not rule out failures in
its unexecuted tail. Logs and all sources remain preserved.

Final `cold-replay-pytest-v2` collected7752 without errors and is running in
container `df583e1263d8389a5f0bd214a51d2e8cf183ebb5ad801c437da07a205cfd4128`.
Parent also started a clean-environment local run of52 trailing modules against
the final candidate to check the unfinished tail independently. Neither that
subset nor any previous candidate receipt replaces the complete final gate.

A separate Sol read-only review confirmed that paid acceptance must inspect
the inner live dream status and target publication, not merely the supervisor's
`completed` status. The final live database must be audited and hashed remotely;
the worker artifact digest does not include that database. Provider-reported
attempt counts can legitimately be lower than traced admissions because trace
hooks precede deadline/transport admission. Such a difference needs explanation
from private events, not an unconditional equality assertion. Only counts,
booleans, fixed error codes, code locations and digests leave Afrodite.

The independent final-candidate trailing52-module run completed: **1300 passed**
in545.08seconds, exit0, under a clean environment with pytest plugin autoload
disabled. The sole warning is the installed Starlette/httpx test-client
deprecation, not an application failure. This provides additional tail coverage;
the complete pinned Linux suite remains the release gate.

An attempted reactivation of the existing `finish-lme-validation` heartbeat
was rejected by automatic review: persistent future execution of the paid
production-text diagnostic needs separate scheduling approval. The automation
remains PAUSED (verified from its unchanged configuration). Parent asked the
user explicitly whether to approve this bounded follow-up; no replacement or
indirect scheduler was created. The running detached suite and direct current
session checks are unaffected. Do not claim automatic continuation is enabled
unless a later successful tool result confirms it.

### Continued direct verification and postflight preparation

The user reiterated permission for necessary tests, including paid calls. Parent
continues the already-approved one-off diagnostic workflow directly; the optional
recurring heartbeat remains paused. No additional test approval is the blocker:
the complete final-candidate suite is the outstanding prerequisite.

A fresh Sol agent implemented the small diagnostic-only
`claim_conflict_private_dream_postflight.py`, reusing the hash-pinned existing
store-audit helper. It checks schema63→64 through `schema_meta`, completed new
dream status, exact target current publication joined to its nonnull local
proof, source immutability and integrity/canonical/ledger/conflict audits.
Quarantine IDs remain private sets: new entries are detected even when the
total count is unchanged, while unchanged baseline quarantine is distinguished
from a new regression. No application bytes were changed.

Parent reviewed the code and independently passed4 real-candidate integration
controls (3.49seconds), covering exact current-publication UDF behavior, wrong
generation rejection, imported pending state and dedup-on/off historical replay
that remains NULL-proof without claim mutation. The target of the planned paid
test is a new publication: the prior retained-response replay already verified
that its first publication acquires a proof. This requirement must not be
generalized to all legitimate historical replays. Postflight worker SHA
`b63900565d5195c466b9d106c1cbd349a657cc962d4c7d10773f9f7fb3d1acec`;
agent test SHA `56ba744c3e8df05a89ae9f81331d3a384af040fe91585e0d121fb12b7c1d730f`.
Nothing was uploaded or launched for postflight yet.

Parent review then caught a false-green gap in that **unsealed diagnostic**:
`chunk_extraction_failures` and `budget_exhausted` are not persisted in
`dream_runs`. One target can publish while another extraction is held for retry
below quarantine. The original audit alone cannot certify a clean invocation.
Sol corrected only the new diagnostic/tests; application, old pinned audit,
paid instrumentation and running full-suite bytes are unchanged.

Final postflight now requires `--worker-result /campaign/worker-result.json`:
capture the actual finished live container's JSON **privately on Afrodite**,
verify its projected metadata equals the sealed campaign stage, and mount that
record read-only. It validates the exact candidate DreamReport numeric/nullable/
boolean partition, zero error fields and false boolean gates, a positive
completed-chunk count, exact source/phase1/target/generation identity and agreement
with every overlapping persisted counter. The omitted text field
`aggregation_blocking` is checked privately in the stored latest dream row;
only presence/absence is exported. Both integrity summaries are safe output.

Parent reran the final12 postflight controls,4 original audit controls and4
independent real-candidate controls together under the clean frozen-candidate
environment: **20 passed in4.68seconds**. Final worker SHA
`1ca6318fdac58d7b0715547baa8b80d4119c9a1a2012fb9f58bf96c3a443b117`;
test SHA `978ddac453528968c7ad72056d17087e36cdc6d48bdebf30d71f82b64497c687`.
Prior postflight hashes above describe the earlier unshipped draft, not the
accepted checker. The full application suite was52%, no failure/error markers,
at this checkpoint. No paid calls or production changes have occurred.

### Final full-suite receipt independently verified — September26

The pinned Linux suite finished: **7748 passed, 4 skipped, 0 failures and
0 errors**, 7752 collected, one full run, networked runs0. Parent independently
recomputed the exact install/source/test pins and inspected the real container
configuration (network none, no credentials/production mounts) and terminal
state (exit0/PID0/noOOM). Source and application bytes stayed unchanged.
Result SHA `4f5725719b6cbdda61906874cc373cae6b9901547b9553dd06f0ab6ed6fd40ec`.
The paid private-dream check remains unlaunched while the user-requested
Afrodite historical-restart/monitor incident is investigated separately in
`2026-09-26-embedding-restart-alert.md`. This full suite does not itself verify
production deployment or a successful production dream/full LME run.

After the separate embedding-server incident repair and successful postdeploy
functional checks, parent sealed the exact suite hash into the approved private
adapter. The86 controls plus6 subtests passed again. Adapter source SHA
`38df85589fbc561e287bc75329b3ab785af001dd0f6bc3024cbb09f629820f0c`.
One headless campaign was installed and launched, supervisor375293. Its offline
preflight exited0; live container
`8a726a4be865885910f20a5d396f659a1cc59f3dbbe8eaae25052e42c38bad5d`
was independently checked against its exact configured image, mounts and
security settings. Limits remain64 completions /192 LLM HTTP /512 embedding /
704 total HTTP /1800seconds. No production HyMem migration or dream was started.

A Sol agent also prepared a separate network-none postflight host launcher.
Parent caught missing failure cleanup and unsealed SQLite sidecars in its
draft. The revised launcher verifies exact-container cleanup after failures,
rejects nonempty WAL/journal rather than checkpointing the original, captures
the full worker report privately, and requires a sealed successful campaign.
Parent reran launcher, accepted checker and audit tests: **39 passed**. The
launcher has not been uploaded or run; actual live completion and final database
hash must be reviewed before creating its seal. Only allowlisted metadata leaves
Afrodite; an attempted unprojected result read was blocked before execution and
replaced with a strictly numeric/status reader, with no private payload export.

### Private paid result and strict postflight — September26

The one-shot test completed17 chunks/171 triples,48 completions,48 LLM HTTP and
159 embedding requests (207 total), with exact provider accounting and clean
cleanup. The target current publication/local replay proof advanced0→1. No new
quarantine IDs, integrity/FK/canonical/ledger/same-generation findings.

Postflight-v1 encountered an audit-only SQLite read-only WAL-header opening
error. Root reproduced it; a separate Sol's immutable sealed-source reader
correction passed49 combined controls without changing application or audit
acceptance rules. Postflight-v2 correctly failed on aggregation rather than
turning completion into a false pass: inherited8 surplus episode-vector keys
block the deadline-bounded runner before aggregation. Root verified this from
baseline and final copies; all expected vectors match, and all8 extras have no
current episode or durable embedding-mirror key.

A new Sol's atomic surplus-only repair passed382 root-run focused/broad tests
plus a network-none replay on the real private copy. Exact44→36 derived rows,
all valid vector bytes and all other tables unchanged, repeat/reopen clean.
See `2026-09-26-episode-shadow-surplus.md` for separate new candidate identity
and evidence. The old candidate's failed audit remains a failed receipt; the
new candidate has not yet passed a full end-to-end dream or production rollout.
Production HyMem/schema migration and full LME remain unperformed. The separate
live embedding-server/health-monitor incident repair is deployed and healthy.
