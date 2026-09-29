# Fresh verifier-envelope diagnostic (v16)

## Authorized scope

The user said “Run it.” after the verified verifier-envelope/diagnostic-consumer
fix. This is one fresh four-retained-case/eight-control diagnostic using
`deepseek-v4-flash` at `https://api.deepseek.com`: at most **32 completions /
96 HTTP attempts**, 120 seconds per invocation plus two seconds cleanup,
first failure stops. No automatic reroll/resume, production memory submission,
deployment, restart or full LME benchmark is included. The closed v15 authority
is not reused.

Status: the single approved live diagnostic **halted on task 1/12** at summary
compaction. Root independently verified the failure, usage and cleanup.
The run is closed; no reroll/resume. **LME is not cleared.**

## Source freeze

Root reauthenticated the prior turn's accepted 2,473 distinct tests, exact test
inventory and unchanged source. The freeze contains 456 Python/SQL files plus
two unchanged auxiliary files; 64 selectors across 59 test modules.

- Source manifest: `dc52f75d5036486ff3ca0d58773640781edcb61f10415ea63451f0afe7702225`.
- Source archive: `b77c25e6f587cd004f33319c7494ef2e27fccb4b89a27ce26ddc443f30ad4009`.
- Local stage: `/private/tmp/hymem-readiness-v16.XdmzMA`.
- Private Afrodite stage: `/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-verdict-readiness-20260916-ErwEss`.

Only the existing authorized benchmark evidence is used. Raw source, requests,
responses and credentials remain private. The retained benchmark SQLite file
is opened immutable/read-only with unchanged-hash and no-sidecar checks.

## Helper adaptation and acceptance

A separate agent updated fresh helper copies for the new source inventory,
stage paths and v6 owned-binding/authorization version. Root reviewed the full
diff. The accepted parser, exact-raw validation, finish-metadata capture,
single-call controls, six-call pipeline maximum, single-use dispatch, absolute
deadlines, independent replay and worker cleanup are unchanged.

The agent passed 129 focused synthetic tests. Its initial invocations used the
wrong working directory for an inherited current-directory-based fixture; the
strict module-origin guard rejected them. Correcting only the invocation to use
the frozen candidate yielded the accepted result.

Root's first full helper run passed all 393 tests but was **not accepted**:
its broad file inventory included independent launch/audit scripts being
prepared concurrently. The harness now freezes all actually consumed test,
helper, runner, candidate and application inputs explicitly. The first receipt
remains unaccepted; a fresh full gate is running. No paid call occurred.

The exact source Linux gate runs in a disposable, resource-bounded container
with networking disabled and read-only code/runtime mounts. No credentials or
production database are mounted. Matching Linux helper acceptance and a
separately audited, zero-provider twelve-task rehearsal are also required
before the freshly approved live dispatch.

## Fresh local helper gate accepted

The complete clean rerun passed **393 distinct tests**, zero failures/errors/
skips, unchanged actual inputs and zero external network/provider calls.
Root verified every captured input hash and the full exact test inventory before
freezing the helper archive.

- Local JUnit: `e4b0aaffeb765ba3de4a5bb645b173fdfa938e83143a9d5c3e0203d421b3d736`.
- Frozen-input manifest: `85dadd84f832ed65fb405bc43534cc2df07e1dfba8c2d35eaf16e987c0d5698f`.
- Helper manifest: `85bf6d1c6d862f84bddd05f2f384fec889cde5c4b142f90aba1cc0508e569dd5`.
- Helper test manifest: `c0a434220a37802e251ed614e70efe4a64300bd5c33864ca43f7c096e8b4786a`.
- Helper archive: `3d1724345f796ecf4e763f91c9b4a4b535e523339e1ddd2d5efb0f0122fce0df`.

The matching Linux helper gate and source gate remain prerequisites. No live
authority has been created or dispatched at this point.

## First Linux helper gate rejected: scratch capacity exhausted

The first Linux helper run produced **315 passes / 78 failures** with unchanged
helper/source bytes and an exact matching test inventory. Root inspected the
first failure: SQLite reported `database or disk is full` while initializing a
synthetic fixture. The inherited 512 MiB `/tmp` tmpfs was insufficient for the
expanded suite's retained test databases. The receipt includes 41 direct SQLite
disk-full failures, an explicit `ENOSPC`, and subsequent supervision/receipt
failures consistent with exhausted scratch storage. No paid dispatch occurred.

Failed JUnit: `6cce4d7b98395bb84fc415c9d875539f6fc6de9f342842e6ea5242b6a33537ca`.
The failed container and receipts are retained as failures. A separate agent is
preparing a fresh runner with dedicated private disk-backed scratch, like the
source gate, and an independently checked mount layout. Application, helper and
test bytes, assertions, budgets and network isolation remain unchanged. Full
Linux helper acceptance still requires a complete fresh 393-test run.

Root reviewed and accepted the new launcher/auditor diff before execution.
Only scratch backing and verification receipts change: fresh
`helper-gate-v2-{launch,results,scratch}`, exact five-bind layout, private
0700 scratch owned by UID 1000, non-memory filesystem, at least 2 GiB free-space
preflight. Host free space exceeded 280 GiB. Network-none, read-only code/runtime,
2 GiB RAM, two CPU, 256-process cap and all helper/test bytes are unchanged.
Hash witnesses preserve the original failed scripts and receipts.

- Storage-fixed launcher: `3e7e9bf63067599fe9f761a393edcc52b0e28d63f477040d1b11d4df566edf25`.
- Independent v2 auditor: `c71b44af2c9f8a6117ee85d4c17000e4fb00edc21b575bf26eaee7d9413066cb`.

The complete unchanged Linux helper suite is rerunning. No live authorization
or provider call exists yet.

The storage-fixed container's initial immediate Docker inspection stopped before
start. The exact captured error was subsequently reproduced: optional
`.HostConfig.Tmpfs` is omitted by this Docker inspection response, and direct
Go-template access fails. A separate agent supplied new copies using
`(index .HostConfig "Tmpfs")`; the requirement that the value be absent/empty
and all exact isolation checks remain unchanged. Root verified that lookup
against the actual container, authenticated its never-started `created` state
and original inputs, then started **the same offline container once**. This was
not a live campaign retry. Prior scripts and pre-start failure history remain
preserved.

- Safe continuation: `f04d1eca023f7115eeef8c52326bb00ae7f28ed260a4e1896ba556affe08f50f`.
- Final helper auditor: `a9688aeed86a2aa6e86ab5fa1bb0503f18fa80ea3e006e3eeeea2ac87b691de5`.

The independent Linux source gate completed with **2,473 passes**, zero failures/
errors/skips, in 2,083.73 seconds. Root's source-receipt audit is pending; the
unchanged 393-test storage-fixed helper gate is running. No paid dispatch yet.

Root independently accepted the source gate: exact 2,473-test inventory, zero
container exit, zero network attempts, unchanged source/runtime/installed code
and verified read-only mounts. Source verdict:
`27a50bb9915be2365d31cfc9f9fda7478a992f172083fb4a0210e35612a4d27b`.
The single warning was pytest's already-imported `anyio` assertion-rewrite
warning, not an application test failure. All 458 local frozen input hashes
still match. The helper rerun remains the outstanding Linux gate.

## Linux helper gate accepted

The complete storage-fixed rerun passed **393 tests**, zero failures/errors/skips,
in 467.77 seconds. Exact test inventory and all input hashes matched, with zero
network violations. Root independently verified container isolation, private
disk-backed scratch, unchanged installed source/runtime and preserved original
failed receipts before accepting it.

- JUnit: `f43cb85f1bf9eda482540ba25681f8dd0ab8e9922f8abf11cb5824a644b022d1`.
- Helper verdict: `8bd3276f7b48985dacb6c31a603bcfe0708c342c90706b084c4439d6bc4c38af`.

The final rehearsal now runs in a fresh network-none container with 18 verified
mounts, no credentials and no production store. Both application and helper
Linux gates are accepted; live authority has not yet been issued.

## Rehearsal and one-shot live dispatch

Root independently accepted all twelve rehearsal tasks and 25 synthetic
completions: zero provider attempts, all owned workers reaped, no database
writes, retained benchmark bytes unchanged, and container exit zero.

- Rehearsal audit: `eef3cb812ad05288c78513f2c8bbe340efe46dcba016af130182b54c83ad163b`.
- Fresh single-use authority: `a606929859814d9e1ce2fcd84cb5f5322af3423b6e3e25c8786c25c019475538`.

The credential-free preflight passed and the detached Afrodite supervisor was
dispatched exactly once. Limits remain 32 completions / 96 HTTP attempts,
120 seconds plus two seconds cleanup per task, stop on first failure, no
rerolls/resume. No deployment, restart, production memory or full benchmark.

## Live result: summary compaction exceeded the cap

The first retained pipeline case stopped after **two completions / two HTTP
attempts in 7.307 seconds**. Primary generation returned a 507-code-point
trimmed summary against the 500 limit. Its one bounded compaction returned
**523 code points**, despite the correct prompt explicitly targeting 350 and
stating a hard maximum of 500. Both replies were valid JSON and both provider
finish reasons were `stop`; they used 532 and 116 completion tokens respectively
against 3,072-token request limits. This is model length noncompliance, not an
output-token truncation or another verifier-envelope parse failure.

The terminal result is `summary_output_cap` at `summary_compaction`. The
verifier, targeted content repair and final format verification were never
reached. **Eleven tasks remain unattempted**, including all verifier controls.
The previously fixed verifier envelope therefore received no new live exercise.
This run cannot establish that the subsequent repair path works.

Root independently reconciled exact usage (4,354 prompt + 648 completion =
5,002 total tokens), immutable reply commitments, mechanical replay, first-failure
halt, all owned workers reaped and verified entry/transport terminal receipts.
The retained benchmark SQLite hash and installed source/runtime were unchanged.
No production changes, deployment, restart, full benchmark or further provider
call occurred. Unused allowance does not reopen this single-use run.

- Live summary: `ccd1ebcf5dd00835f280b96ab34425c2c6a1728a7521a2432161acb44bc5fdca`.
- Independent live audit: `621b4ebbf05cc2507cc62c886c3f71bed9c3dab7725e38fed09c3b08b93c8fa1`.
- Independent counts-only cap audit: `82bd19f655724902f7d8172de85872839a33a9fd966e8c8167d1653c737dac72`.

Next implementation target: make the bounded summary-compaction path reliably
produce a within-cap candidate while retaining source fidelity checks, exact
prior/context, hard output validation and bounded cost. Do not silently truncate
the summary, relax the cap or reroll this diagnostic until a pass is obtained.
Any implementation needs separate verification and a fresh live diagnostic;
an end-to-end smoke remains required before the canonical baseline.
