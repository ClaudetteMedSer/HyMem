# Source-content recovery live diagnostic

## Scope and current status

The user explicitly approved one fresh bounded diagnostic: four retained LME
digest cases and eight verifier controls, benchmark text only, sent to
`https://api.deepseek.com` with `deepseek-v4-flash`. Limits are 32 completions /
96 HTTP attempts, 120 seconds per invocation and first-failure stop. The approval
does not authorize deployment, service restarts, production database access,
automatic rerolls, campaign resume or a full benchmark. Unused capacity is not
permission for a second diagnostic.

The approved live diagnostic ran once after all gates below passed and halted on
its first case: the source-recomposed summary was rejected again. The v14 and
prior v13 diagnostics are both closed. **Full LME readiness remains unconfirmed**;
this live failure must be understood and an end-to-end smoke is still required.

## Frozen inputs

Local private workspace: `/private/tmp/hymem-readiness-v14.pJ6FLa`.
Afrodite private stage:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-content-readiness-20260915-ixgX9A`.

The candidate contains the exact 449 Python/SQL inputs accepted by the 1,790-test
local source gate plus two unchanged auxiliary files, for 451 files total.
Source manifest SHA-256:
`f73ecaf984a720ff2cff65b647236ccef918ec73c16b3f717e60c85e876e445b`.
Source archive SHA-256:
`7330b8a4acd8b61b6cab9a5a8c5a8bc760a2a9fc911dbf407bc1acde942a92cd`.
No old live authority or old target-test verdict was copied into this stage.

The two retained fixture files are copied in place on Afrodite. The benchmark
source database remains read-only/immutable with SHA-256
`feac1d866dd7669892e75ea9a4c5b5c637315586ee6a286015353a55a4c56b93`.
Raw remote evidence stays private; only fixed status categories, counters and
validated hashes are exported during this diagnostic.

## Verification sequence

1. Rerun the exact 1,790-test selected source inventory in the target Linux
   runtime, network disabled, credentials and production stores unmounted.
2. Adapt the diagnostic with a separate agent, then independently review and
   test it. Six roles are individually recorded and exactly replayed: primary,
   optional compaction, verification, source-content recovery, reverification
   and optional format adjudication. The supervisor keeps one absolute deadline
   across all roles; semantic rejection cannot be overridden by format review.
3. Freeze the accepted helpers and tests, rerun the same helper inventory in an
   isolated Linux container, and reconcile exact inventories and input hashes.
4. Rehearse all twelve tasks against retained inputs with synthetic responses,
   no credentials and no network. Expected schedule: 23 synthetic calls,
   including complete six- and five-call recovery paths. This checks execution
   mechanics, not live model accuracy.
5. Bind fresh version-4 authority to the exact candidate, helpers, independently
   audited gates, rehearsal and current explicit approval. Dispatch once.
6. Audit provider usage, exact replay, first-failure halt, worker cleanup,
   immutable input hashes and unchanged installed runtime/production source.

An initial agent helper run exposed a stale test expectation (five reserved
calls instead of the new six plus a control). The assertion was corrected;
the subsequent focused run passed 70 tests. A separate early sandbox run could
not bind the synthetic loopback server. Those failed receipts are retained and
are not acceptance gates. Root's combined stable-input gate is authoritative.

No selected regression test count is presented as a full repository test suite
or proof of production/benchmark correctness.

## Helper gates accepted

Root's complete local helper gate passed **274 tests**, zero failures/errors/skips,
unchanged input hashes and zero non-loopback network violations. Eight new root
controls independently reject changed recovery/reverification parameters and
evidence payloads. The same 274-test inventory passed in the target Linux
runtime, with Docker network disabled and no credential or production-store
mounts. These counts overlap and must not be added together.

Frozen helper-manifest SHA-256:
`3138951565b4512d9077b708bee00504c8f3848dd489e55e985704dcfc4f681d`.
Helper-test-manifest SHA-256:
`d9e4f8ad2bf5d3b2ff979229895b3c396284714a75eb99bf99325368f75bd780`.
Helper archive SHA-256:
`00ab75474c47bffd388785d9b5e5faa40b8746a705f00fd55dac62422bda09cb`.

The independently audited helper-target verdict SHA-256 is
`d2de344c47cf6568beb05640bbd4e096129bf88bd24db2de813fdae06f001dc8`.

## Target source gate accepted

The exact selected Linux source inventory passed **1,790 tests**, zero failures,
errors or skips. Root reconciled the test inventory, unchanged 451-file candidate,
zero network attempts, container isolation and unchanged installed production
source/runtime. Target verdict SHA-256:
`0eff53f5ea6c181ee2aa73ad2b437bd675028780f1fdf3baf3c869fa3cf4fbf4`.

## Rehearsal accepted and single live dispatch

The no-network retained-input rehearsal completed twelve tasks and 23 synthetic
calls. Root independently verified the exact six/five/two/one-call schedules,
unchanged database/input hashes, zero provider attempts, deadline ownership and
reaping of all twelve workers. Rehearsal audit SHA-256:
`5333d642123fc7af3d1a33cf6d29287654c8a201391698c380b004768e9ec7e6`.

Fresh authority records the user's exact response, “Approve this bounded live
diagnostic”, to the question naming the benchmark payload, endpoint/model,
32/96 limits, deadlines and exclusions. Authorization SHA-256:
`140492faddae83cb68f71ac287deafff88d5673a45aaddfbc46d71663e6e91bb`.
Consent SHA-256:
`0ed10f12fb6e3eee8f20c8b6689fc9844be0cc00476f194021dffc01e73f8492`.

Authenticated preflight passed; the one-shot dispatcher started the owned
supervisor on Afrodite. Automatic rerolls remain disabled.

## Final live outcome and independent audit

The diagnostic halted on **task 1 of 12**, after **five completions / five HTTP
attempts in 10.908 seconds**. All five HTTP responses completed normally. Exact
role sequence: primary, summary compaction, fidelity verification, source-content
recovery, fidelity reverification. The final failure remains
`summary_content_unsupported` at `fidelity_verification`. No format adjudication
was attempted because semantic acceptance is a prerequisite.

Both verifier responses accepted the two episode titles, contents and formats;
both marked the summary's content and format unsupported. There were no
procedure items. Source recovery produced a changed candidate that reached full
reverification, so the new control-flow path executed. This does not prove the
generated replacement was faithful or that the verifier was correct. The raw
new candidates have not been exported or reviewed; verdict categories alone
cannot distinguish generation defects from false rejections.

The other eleven tasks, including all eight controls, were not attempted. Usage:
13,454 prompt tokens, 958 completion tokens, 14,412 total tokens. The reserved
six-completion/eighteen-HTTP task allowance is not the actual usage count. No
remaining allowance was reused and no second run was started.

Root independently reconciled committed requests, roles, outcomes and usage,
first-failure stopping, all owned-worker cleanup, immutable benchmark inputs,
unchanged installed production source/runtime and container identity. No
production database was opened, no deployment or restart occurred, and no full
or end-to-end LME benchmark was launched.

Live summary SHA-256:
`d3d26f3522b1129f905e86a62ba4735b30b7c054114caa7342a593b63158f260`.
Independent live audit SHA-256:
`cd240f92ce6f88fe6d74edb0be20046d5eb2424552f6cfe6ca2ad0847e7d886a`.

Next step requires narrowly scoped review of the two rejected summary candidates
and their verifier verdicts against the already approved source/prior/context.
That review needs no provider calls. Do not weaken the factual veto, declare
readiness, reroll this campaign or launch the canonical baseline from this result.

Follow-up on 2026-09-16: the user approved that review. It found a repeated loss
of explicit event order, a false format rejection of the grammatical repaired
summary, and no source/request wiring mismatch. Detailed findings and limitations
are in `2026-09-16-content-diagnostic-review.md`; no new provider call or runtime
change was made.
