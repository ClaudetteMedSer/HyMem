# Independent summary and item-index completion

## Approved scope

The user approved separating summary generation from source-backed indexing,
including paid verification where useful. Work is isolated at
`/private/tmp/hymem-summary-index-decoupling-20260919.zeYFpg/candidate`, based on
the previous frozen source plus its corrected tests. The unrelated dirty main
application checkout and production remain untouched. Deployment is not part
of implementation verification.

This supersedes unsuccessful prompt-only repair candidates as the proposed
fix. Those failed diagnostics remain recorded; none is relabeled successful.

## Contract

- Exact source coverage, item schemas/citations, transactional publication,
  producer identity, malformed state, deadlines, cleanup and cost accounting
  remain mandatory gates.
- A successfully parsed primary response whose episode/procedure items pass
  existing validation may publish those items despite a separately rejected
  context summary. Unknown transport/identity failures are not downgraded.
- Item publication gets its own full-message frontier, independent of the
  last accepted automatic-summary frontier. Summary text is not evidence
  authority and cannot confer source coverage.
- A failed summary preserves the prior accepted/legacy/operator text and its
  honest old frontier. Later suffix-only output cannot bridge the gap or mark
  the summary current. Subsequent item requests disclose stale prior context.
- Summary backlog and recovery have independent state and bounded retries;
  only contiguous replay of protected exact sources may clear degradation.
  An incomplete summary rebuild must not replace the last published summary.
- Durable diagnostics distinguish valid summary degradation/missing summaries
  from malformed summary state. Malformed state remains fatal. Empty pristine
  sessions remain pristine; legacy history is never silently truncated/deleted.
- Benchmark completion is versioned and explicitly reports
  `success_with_summary_degradation`. Every question, denominator, cost and
  warning remains in artifacts. This is not a pass of the previous fully
  healthy-summary completion contract.

## Implementation and verification sequence

1. Separate agents audit schema/publication, generation context, and benchmark
   admission. Parent independently traces the load-bearing code paths.
2. Implement schema v62 and summary-state classification; test fresh/upgrade/
   reopen, malformed-state rejection and conservative legacy handling.
3. Implement extraction outcome separation, then staging/runner publication.
   Parent reviews and adds independent controls before the next stage.
4. Implement independent contiguous summary recovery and restart behavior.
   Test gaps, partial messages, source pruning, rebuilds and forward appends.
5. Integrate durable status, Honcho/MCP/doctor, portability, benchmark protocols,
   registry and material attestations. Reject missing/forged/contradictory
   diagnostics and old evidence relabeled as the new contract.
6. Freeze exact source/test inventories; run local and Hermes-runtime gates,
   including end-to-end dream/publication, migration/import and benchmark
   completion tests. Independently review the diff and reproduce the patch.
7. First replay retained failures without new provider calls. Then preregister
   bounded live checks using only approved benchmark/synthetic data. Use paid
   calls to test actual application/benchmark completion, not blind rerolls of
   the same prompt. No production memory or credential transfers.
8. Report verified outcomes and remaining limits. A small replay is not a full
   LME readiness claim; full-run verification requires its own explicit result.

## Shared status interface

Dream status v8 adds nonnegative integer `summary_degraded_sessions`,
`summary_missing_sessions`, and `malformed_summaries`, plus boolean
`summary_healthy`. Booleans are not accepted as counts. Missing is a subset of
degraded; summary_healthy is exactly `(degraded == 0 and malformed == 0)`.
Malformed summaries join the mandatory blocking counts. Required item work
retains its existing pending/quarantine/failure meanings.

Benchmark indexing status becomes v4 and LME indexing summary becomes v6.
The configuration/manifest explicitly pins
`source-backed-index-with-explicit-summary-degradation-v1`. Core source-index
health and summary health are distinct; neither may silently overwrite the
other. A clean follow-up cycle cannot erase durable degradation.

## Progress

The implementation now includes schema v62 (independent public frontiers),
v63 (private bounded summary recovery), portable format v18, source-index
completion policy/version changes, and public health reporting. Development
gates have passed for extraction/publication, benchmark admission, portability,
read-side health and recovery; these are not yet one frozen full-suite receipt.

Independent review found and fixed missing producer-fingerprint aliases and
the Honcho text/health snapshot race. Recovery review additionally found an
open SQLite census cursor across provider calls and a short-summary validation
mismatch; both were fixed and their focused regression tests passed before freeze.

Recovery is explicitly invoked, not silently added to every dream or benchmark:
`HyMem.recover_summaries(...)` and `hymem-recover-summaries --apply` have separate
call/attempt/deadline bounds. The command without `--apply` only inspects an
existing current-schema database. Its provider accounting and health result
are independent of item indexing. Exhausted slices remain visible and do not
automatically reroll when a limit/model changes. Operator/legacy text remains
preserved. Partial recovery drafts are local-only and never exported.

Summary health means that the current accepted summary meets the existing
structural contract and has a contiguous, source-proved input frontier. It is
not a claim that an LLM-generated summary is semantically infallible. Neither
the prior frozen digest implementation nor this recovery path has a semantic
truth oracle. Live content controls and benchmark scores remain necessary.

Next verification is an immutable full offline gate, a network-denied migration
and replay rehearsal on copies of the 17 retained benchmark cases (10 facts,
7 digests), then at most 34 fresh DeepSeek completions / 34 HTTP attempts if
the prerequisite gates pass. Each case permits at most two completions; every
HTTP request is separately supervised with a 120-second deadline. No old
campaign is resumed. A returned model rejection may be recorded while other
independent cases continue; safety, identity, accounting or cleanup failures
halt. Original benchmark stores remain read-only, and raw benchmark data stays
on Afrodite. This diagnostic alone cannot establish full LME completion.

The approved 17-case paid diagnostic completed under the fixed 34-call /
34-HTTP cap and original immutable extraction source: 17 accepted cases,
21 fresh completions/HTTP attempts, no non-stop responses or item rejections.
All seven digests and ten fact cases were admitted; three digest summaries
remain explicitly degraded. Two fact cases returned empty outputs and two
cases returned accepted partial coverage; this is contract-admission evidence,
not a semantic-completeness or full catch-up claim. Accounting totals were
42,081 prompt tokens and 9,233 completion tokens (51,314 total). Independent
audit verified every worker's cleanup, all 230 source files and all ten original
store hashes. Summary receipt SHA-256:
`c697d59d6e8b820389eccf0b029e56ea4a7f0f82644aefcaf5d889139327e770`.
The exact container exited 0, no OOM, PID 0.
The later CLI-only correction and integration-test waits do not change that
extraction path. No deployment or production mutation has occurred. The
unrelated dirty main application checkout is still untouched.

## Frozen verification, revision 2

The initial 6,965-test run completed with 6,914 passes, 23 failures and 28 setup
errors. Its receipts are preserved unchanged. Failures were traced to missing
auxiliary test assets, an over-restrictive harness that blocked synthetic
loopback servers, and integration fixtures asserting the superseded coupled
summary/indexing policy or synthesizing old archives with new-only fields.
The strict direct-extraction controls were not relaxed. Updated integrations
assert independent frontiers, truthful degradation, unchanged item rows during
recovery, and persistence through reopen/import. Their five-file gate passed
273 tests; the parent reviewed the changes and is rerunning the full suite.

Revision 2 freezes the same 230 application files, 220 test files, and eight
separately pinned auxiliary assets. Its manifest is
`df94fc65708e43703faba0f3dc60dd4ec4b56de956a2b501b03f7b9ff35df885`.
The 6,965 local controls completed: 6,964 passed, one README/default assertion
failed, and none were skipped. The README came from the newer dirty checkout
and documented its `deepseek-flash` default, while this isolated candidate still
has the older `deepseek-v4-flash` default. A separate test-only documentation
snapshot now truthfully describes that candidate and explicitly warns that it
is not a current-provider recommendation; the three unchanged public-default
assertions pass against it. Application files and the failed full-suite receipt
remain unchanged. This is a full run plus a targeted documentation correction,
not a newly executed all-green full suite. Deployment must separately integrate
and verify current model policy; live diagnostics explicitly select the approved
`deepseek-flash` route with thinking disabled.

The 1,714 target-runtime controls completed: 1,711 passed, one direct-CLI import
failure, and two five-second scheduler-test timeouts, with no skips or setup
errors. A clean runtime exposed that direct `lme_registry.py` execution did not
add its checkout root before importing `hymem` through `strictness`; this was
masked by the laptop's installed package. A separate source correction and
clean-interpreter regression are being implemented before live admission.

The scheduler failures were independently reproduced with the unchanged app
and a diagnostic-only 30-second wait. Exact one/two-cycle assertions passed;
actual waits were 5.149 and 10.452 seconds on the one-CPU test container. The
proposed correction is a bounded integration-test wait, not a production
scheduler/deadline change. Original failed receipts remain preserved.

The corrected runtime bundle preserves the combined source/
test layout. Target containers have no network, credentials, or production
store mounts. Local service tests permit only explicit loopback connections.

The first retained-case preparation stopped before creating a clone because
the diagnostic helper incorrectly rejected the pinned ledger's additional
`chunks` metadata. This is a helper-schema mismatch, not a store migration
failure. Independent read-only checks confirmed all ten original store hashes,
schema v61, clean SQLite integrity and zero foreign-key violations. A revised
helper and real-ledger regression passed 87 checks. The v2 preparation migrated
all ten disposable copies successfully and preserved all original column
values. Scripted positive, negative and summary-degraded controls each completed
all 17 cases, with zero HTTP calls. They preserve rejection of invalid items and
report summary degradation separately. The failed first preparation is not
being overwritten or labeled successful. Exact replay of two saved response
pairs for one historically failing digest completed without network or
credentials. Both primary and repair request wires matched exactly. Each pair
now returns the same one source-backed episode with no procedures, while the
rejected summary remains explicitly degraded (`summary_output_cap`). Four saved
responses were consumed and zero new completions/HTTP calls made. The original
store stayed read-only; these extraction-only replays do not claim publication
or end-to-end benchmark completion. The independent receipt hash is
`76a3d9c06a3fa99f7b44fd7eece5c967649fd5f983ca224d280837c28df97957`;
the exact replay container exited 0, no OOM, PID 0.

The next genuine end-to-end target is Q1 `09ba9854` from the unchanged pinned
500-row dataset (SHA-256
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`).
Read-only verification establishes source index 210, all 44 sessions and 479
messages. Stock label-blind sampling with sample 1 and precomputed seed 53
selects that index. This is explicitly targeted regression evidence, not a
random-sample accuracy claim. Production remains untouched; the separate capped
retained-case diagnostic is complete.

## Revision 3 closure and actual Q1 startup failure

The direct registry CLI bootstrap correction and bounded scheduler-test waits
were verified together: 265 focused controls passed on both the laptop and
the isolated Hermes Python 3.11 runtime, with exact source/test inventories
checked before and after. The final inventory contains 6,972 collected tests
(the previous 6,965 plus seven clean-interpreter CLI controls). This is not a
new full-suite run. Patches 12/13 reproduce the exact revision-3 tree; production
and the unrelated dirty main application checkout remain untouched.

The real stock Q1 launch then failed before checkpoint creation, canary or any
provider call. Its process exited 1 with complete process-group cleanup; the
failed output is preserved under the isolated `q1-stock-v3/live-results`.
A second, credential-free, network-denied execution reproduced the exact
startup traceback: `_run_main` constructs the pipeline producer declaration,
which rejects the explicit `deepseek-flash` request because revision 3 only
automatically identifies the older `deepseek-v4-flash` official service.
Both deployment attestations are deliberately absent in the clean environment.
This is an integration defect in the candidate, not a model-output failure.
The earlier preflight checked source/dataset/selection but did not exercise
runtime producer identity; the retained-case harness also does not construct
the stock benchmark's manifest. Those checks were insufficient for startup.

Revision 4 must integrate current-model policy consistently across standalone
and runtime identities, active defaults, and historical artifact validation.
Custom endpoints and partial attestations must remain fail-closed. The official
service identity must not claim immutable provider weights. Add a provider-free
real CLI startup test and stronger preflight before another fresh Q1 attempt.
DeepSeek's current official documentation confirms `deepseek-flash` and the
retirement/rerouting of the former v4-flash names:
<https://api-docs.deepseek.com/updates/>. No failed run is resumed or relabeled.

The first credential-free diagnostic was itself over-restrictive: it rejected
socket allocation during import, before reaching the defect. Its receipt is
preserved. The corrected diagnostic permits allocation but denies outbound
connections/DNS, has no network or credential mount, and reached the above
ValueError with zero network attempts. Q1 also exposed a helper-only Python
bytecode-cache inventory issue; the generated cache was preserved separately,
and the reviewed helper now executes already-verified source bytes without
writing cache files. Neither diagnostic correction modifies application code.

## Revision 4 model-policy integration accepted offline

Current requests now consistently select `deepseek-flash` across standalone
producer declarations, real clients, server defaults and benchmark entry points.
Retired names fail before credential resolution; custom origins and incomplete
operator attestations remain rejected. An explicit historical-commitment reader
preserves older model/runtime evidence without reconstructing it using today's
producer, without relabeling its assurance, and without granting execution,
resume or official-export eligibility. Current strict validation is unchanged
in authority. The preserved `HYMEM_LLM_EXTRA_BODY` deployment customization was
not removed or changed; the isolated canonical recipe clears that environment
variable explicitly because it must not alter declared request bytes.

The clean R4 snapshot has 230 source files, 224 test files and eight auxiliary
assets; its full collection is 7,087 tests. The implementation agent's two
disjoint gates passed 509 and 197 tests. The parent independently reviewed the
diff and executed 453 controls on the frozen tree: zero failures, errors or
skips, with exact hashes checked before and after. Manifest SHA-256:
`a21f1019bf9f9c97e78a3b57f3649c6381fc88e7e21242cec4e5f9527ca987f7`.
Parent XML SHA-256:
`b23f7ba29adafbf9e0165af8f5b31c02bb05a2ed833c7e09cc54dcc54ae1d4a3`.
This is not a full-suite or paid-Q1 pass.

The stronger provider-free preflight independently constructs and closes the
real SDK client, matches its producer declaration to the standalone builder,
then reaches the real stock CLI's reader-client boundary. It checks the actual
checkpoint's producer equality and lease cleanup. Old R3 fails; corrected R4
passes. It uses invented test data, a synthetic placeholder key, no networking,
zero provider calls, and does not complete or score any question.

## Revision 5 response-boundary integration in progress

The earlier retained-case harness enforced response admission/accounting itself;
its success did not exercise the stock SDK and reader/judge transports. The
isolated candidate had not inherited the corresponding newer working-tree
fixes. An independent synthetic R3 baseline confirms this gap: 99 cases, 88
failures, seven passes and four intentionally inapplicable raw-client skips,
with no network. The defects include accepting non-stop replies, losing paid
usage on rejected replies, retrying received malformed raw responses, and
claiming exact usage while an attempt remains unaccounted or in flight.

A separate agent implements the narrow transport integration only after the
parent's R4 acceptance. Rejected text must never publish, but rejection must not
silently destroy existing bounded recovery: typed provider length truncation
must still permit source-safe chunk subdivision, honest summary-only degradation
after a valid primary digest, and held private summary recovery without advancing
its cursor. Unknown finishes, transport/identity failures and deadlines remain
fatal to their existing scope. Paid replies are accounted before admission;
unknown cost stays unknown. No extra HTTP reroll follows a received rejection.

The canary previously equated admitted replies with logical completion calls.
Its documented recovery path therefore needs an explicitly observed typed
truncation count, not an inferred missing-call count or relaxed inequality.
Version the resulting evidence contract, preserve historical v17 separately,
and retain the exact claim/context controls and 24-completion/72-HTTP envelope.
No new paid run, production deployment or restart occurs before this combined
integration is reviewed and tested.

## 2026-09-24 resumption and evidence recovery

The account-usage interruption stopped both implementation agents. On resumption,
the previous temporary candidate/helper/receipt files were absent (their empty
directories remained), and the running test session was no longer available.
The cause of file removal is not established. Do not treat the interrupted R4
full-suite shard as completed, or the vanished R5 candidate as implemented.

All durable patches 00–13 and the R3 manifest survived. The reconstruction helper
`docs/patches/2026-09-24-reconstruct-independent-summary-r3.py` reapplies every
patch with exact old bytes and hunk coordinates against the pinned Git base.
The implementation agent and parent independently reconstructed all 459 files:
230 application files, 221 test files, and eight test-only auxiliary assets.
Every hash matches the existing R3 manifest. This recovers source evidence;
it is not a fresh passing test run. New R4/R5 implementation and test receipts
will be preserved durably instead of relying on temporary-only files.

A durable offline runner now checks the exact source/test inventory before and
after execution, the full collected node inventory, the final shard selection,
executed node IDs, JUnit counts and explicitly declared skip IDs. Its 21 synthetic
checks include four subprocess executions. Independent review found and corrected
missing DNS/socket audit paths and possible false-green collection/receipt cases.
The network guard applies to the pytest process; child tests inherit a clean,
credential-free environment but require OS isolation for process-tree enforcement.
This runner makes no benchmark-readiness claim.

The previous R3 response baseline (88 failures, seven passes, four intentional
skips) was mixed evidence: its current-model fixture also hit R3's model-startup
defect. A clean transport before/after comparison must use reconstructed and
accepted R4; the old count is not a precise transport-defect census.

The reconstructed R4 model/startup increment is now accepted for the next
sequential fix. The parent reviewed all 16 changed source files and the new
regressions, independently reapplied patch14 plus its separate test-only README
patch, and verified all 462 resulting hashes. The fresh manifest is
`16cf214451f90ee9c7d2a73151a0b26c5bd9fe04404cc40b7370f03777c916d0`.
The agent's focused gate passed 513 tests. The parent's frozen-tree gate passed
464 tests with zero failures/errors/skips and exact before/after inventories;
18 real-startup diagnostic controls also passed against R4 and the R3 negative
control. These overlapping gates are not to be summed into a unique test count.
Separately, 284 restored-R3 summary/index-publication controls passed. No full
R4 or R5 suite result is claimed.

The unchanged 99-case response suite was then run against this accepted R4:
88 failures, seven passes, four explicitly inapplicable raw-backend skips,
zero setup errors. This fresh baseline reaches the working model route and
isolates the response-boundary defects. All 462 candidate files were checked
before and after, and 81 loaded candidate module paths were confirmed. Its XML
is durable at `docs/patches/2026-09-24-parent-r4-response-baseline.xml`.
R5 implementation proceeds from the frozen R4 tree, not from the dirty checkout.

The prior seven-file isolated Q1 runner bundle was recovered read-only from
Afrodite and independently verified byte-for-byte against its original pinned
manifest. No logs, credentials, memory records or production files were read.
An unsealed successor now includes genuine SDK/CLI startup checks and no-cache
verified helper loading; 12 parent helper controls passed. It has no manifest
or source pins and cannot yet launch. No new paid calls, production deployment
or restart have occurred during this resumption.

The resumed R5 source received a separate read-only canary/archive review. Two
false-green telemetry cases were found and corrected before acceptance: emitted
claims cannot be supported by rejected-only replies, and a passing four-leaf
canary still needs at least eight admitted responses. Independent synthetic
execution passed the normal 8/8 logical/admitted path and a recovered truncation
path with 11 logical calls, 10 admitted replies and one typed truncation. Paid
token telemetry includes the rejected reply. Forged reports with 0, 1, 2 or 7
admitted responses were rejected. The genuine unchanged R3/v17 fixture remains
historical-only, and mixed versions, altered caps/counters and broken commitment
cross-links remain rejected. These are provider-free checks, not a live result.

A fresh credential-free, network-disabled temporary container verified the
Afrodite test runtime: Python 3.11.2, pytest 9.1.1, openai 2.53.0, requests 2.34.2
and httpx 0.28.1. Only the existing virtual environment was mounted read-only;
no production stores or credentials were mounted. Its image remains
`sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5`.
The SDK differs from the laptop's, so final R5 target-runtime verification is
still required before paid Q1 admission.

The fresh R4 full suite completed across three disjoint exact-inventory shards:
7,087 passed, zero failures/errors/skips. The parent reconciled every JUnit hash,
the selected counts and all 462 file hashes afterward. The durable aggregate is
`docs/patches/2026-09-24-parent-r4-full-aggregate.json`. This is the accepted
model-compatibility baseline, not the subsequent combined R5 result.

R5 is now frozen: manifest
`c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc`, patch15
`4ba27727009ab6aae65e249f0f5896d3ea6570ccac7afaf48a588a56c4228256`.
There are 231 source files, 227 test files, nine auxiliary files and 7,247 test
nodes. The implementation gate passed 835 checks plus four explicitly declared
raw-backend-inapplicable skips. The parent independently reconstructed all 467
files from accepted R4 and exact patch hunks, then passed 156 response/recovery/
history controls plus those same four skips. All hashes matched before and
after. Eighteen real SDK/CLI startup and cleanup diagnostic controls also passed.
Parent full R5 and target-runtime gates remain pending; these overlapping focused
counts must not be summed as independent coverage or called LME readiness.

The Q1 source-only package was sealed locally and reverified (package manifest
`5cdb8db3213229aed6f140da6747ff2a1048433e75437b3a7feda6521f0b73a3`, seal receipt
`65d77ab5453b429325e0c0344a89da72cb650ddfe1d0203bb0087c3d1338acbc`). Its first
authorized upload into the fresh isolated `q1-stock-v4` directory copied macOS
AppleDouble sidecars as well as the expected source. Remote exact-inventory
verification rejected that unpacked copy before use; it has not been deployed,
imported or launched. The original failed unpacking and archive are preserved.
A metadata-free local archive was independently checked against all 241 sealed
file hashes (SHA-256
`5c656d651aeec87eb667f11ae0aba562568601afc0fd50d1fb9542bc80d82a8f`).
No corrected upload or paid run is claimed.

Separately, the auto-review safety check blocked the 467-file source/test/auxiliary
plus two-file runner/manifest upload intended for the network-disabled Afrodite
gate. That upload never executed, `offline-r5` remains absent, and no test
container started. Explicit user approval for that exact repository-code and
synthetic-fixture transfer was requested asynchronously. No workaround or retry
through another route was attempted; unaffected local R5 full-suite testing
continues. Production, credentials and memory stores are untouched, and this
continuation has made zero paid provider calls.

### Approved R5 execution continuation

The user's subsequent "Run it and verify that now it all works" approved the
specified source/test transfer. The parent full R5 suite has now completed:
7,243 passed, four predeclared raw-backend-inapplicable skips, zero failures or
errors. All 7,247 nodes across three disjoint shards and all 467 final file
hashes were reconciled in `2026-09-24-parent-r5-full-aggregate.json`.

The approved metadata-free target-gate transfer succeeded and the credential-free,
network-disabled 1,044-node Afrodite gate is running in container `421c0acb3695`.
The corrected Q1 archive was uploaded as `upload-v2.tgz`, verified against all
241 sealed hashes and staged from `sealed-v2` into the 231-file source-only
candidate and exact runner bundle. The rejected original archive remains intact.
The separate credential-free, network-disabled genuine SDK/stock-CLI preflight
is running in container `cad42c02b3fc`. Neither result is yet claimed. Live start
requires both containers to exit cleanly with success receipts and matching
source/dataset identity. Production remains untouched; no new paid calls yet.

The target-runtime gate completed successfully: 1,040 passed plus the exact
four declared skips, zero failures/errors, all 1,044 selected nodes reconciled
and all 467 source/test/support files unchanged before and after. Its container
exited 0, no OOM, PID 0; the supervisor reaped its child and confirmed the process
group absent, with no timeout or cleanup errors. JUnit SHA-256:
`62edb5220df5e478ea4fb8148067155072fb7c5238a20725dc9598c9ea2db379`.
The genuine separate preflight likewise passed and exited cleanly, making zero
provider calls. Its bounded parent receipt is
`2026-09-24-parent-q1-v4-preflight.json`.

After independently enforcing both successful terminal gates, source/dataset
cross-binding and the live container's complete configuration, the parent
started the fresh paid stock Q1 test in container `d859dca64f82`. It has no
production-memory mount, uses the sealed R5 source, and retains the fixed
5,400-second supervisor and 3,600-second indexing deadlines. No resume or
reroll is permitted. Exact usage and completion remain unverified until its
terminal artifacts, physical checkpoint and cleanup pass offline validation.
The complete 500-question benchmark and production deployment are not claimed.

### Final R5 Q1 result

The fresh paid stock Q1 completed in 1,624.38 seconds and answered correctly.
Five dream cycles completed with healthy item indexing and no quarantined work.
Ten sessions have explicitly degraded summaries; the declared outcome remains
`success_with_summary_degradation`. Reader and judge each made one measured
call, and strict artifact counts are one expected/attempted/completed question,
zero failed/missing questions, with no resume or repeated attempt.

The first independent postvalidator exposed a verifier-only bug: it compared
the raw physical checkpoint row with the archive's reconciled row, which adds
`strict_failure=False`. Exact host-side comparison confirmed all other fields
identical. A new agent repaired the verifier in a separately mounted successor;
42 real-checkpoint and negative controls passed, independently rerun by the
parent. The same saved archive/checkpoint then passed offline, read-only strict
validation with source/package/dataset identity, physical row/attempt-history
binding, accounting and clean process-group shutdown. Original rejected
validation and paid artifacts remain unchanged; no paid rerun occurred.

Including the dedicated eight-call canary, this Q1 used 806 completions/HTTP
attempts and 2,614,343 measured tokens. Dollar cost remains unavailable rather
than assumed zero. No additional provider calls were used for the verifier
repair. See `2026-09-24-lme-q1-verification-result.md` for exact receipts, pins and
limits. The full-500 benchmark, production deployment and completely healthy
summaries remain outside this success claim.
