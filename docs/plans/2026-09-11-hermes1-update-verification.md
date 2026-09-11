# Hermes1 update and restart — September 11, 2026

The user subsequently approved fixing the findings below. The current status
and subsequent verification are recorded in
[the follow-up fix-cycle report](2026-09-11-verification-fix-cycle.md).
This report preserves the initial update and its rejected/incomplete gates.

## Deployment

Hermes1 was fast-forwarded from `074c981` to the latest upstream
`Beam-optimisation` commit, `af6a615fa7fd1cf14c4c0a27b9fb236ae264f122`.
The container restarted at **10:43:45 UTC** (12:43:45 Europe/Amsterdam).
Hermes2, Hermes3 and the shared embedding container were not restarted or edited.

An isolated reconciliation and installation rehearsal verified **478 files**,
successful fast-forward, customization preservation, repeat no-op and preservation
of unrelated untracked files. The existing runtime patches already matched this
release: **zero HyMem runtime-source files changed** during this update. The
checkout, documentation and tests now reflect the release, while the separately
maintained client and benchmark customizations remain explicit working-tree changes.

Preserved exactly: the `HYMEM_LLM_EXTRA_BODY` client extension; four customized
benchmark modules; environment/embedding settings; secret-file permissions;
the protected Honcho session-retirement mapping and pinned integration hook.
Previously retired generated egg-info metadata remains inactive. No stash was
dropped, no unrelated benchmark artifact was removed, and no user data was rebound.

The operations skill required backup-first, real-store rehearsal and verification
of process environments, rather than treating configuration files or an HTTP
health check alone as sufficient. Its newer capture-recovery correction was used
in preference to the older contradictory rebinding/restart guidance.

## Pre-existing Honcho outage

Before deployment, Honcho PID 90 existed and listened on port 8765, but repeated
health requests timed out (including a 15-second request). Its accept queue held
21 connections, its main thread was waiting on a futex, and the current-boot log
contained no exception traceback. The data checks independently passed.

The scoped restart restored HTTP responses. The new main thread returned to its
normal epoll wait; the accept queue cleared, and subsequent SDK context requests
and repeated health requests succeeded. This establishes service restoration,
**not a diagnosed or permanently fixed deadlock**. Stack inspection was denied by
the container's process-access restrictions; no bypass or permission change was
attempted. No speculative source patch was shipped for this unexplained stall.

### Separate confirmed concurrency defect

A deterministic fresh-store probe of `HyMem.read_conn` reproduced publication
before initialization completes. The initializer was paused immediately after
assigning the shared reader and before its producer-specific scope setup. A
second caller received that connection immediately: `PRAGMA query_only` was
**0** before setup finished and **1** afterward. No production database or
provider was accessed, no source file was changed, and no write through the
premature reader was attempted.

Honcho routes share this HyMem object across request threads. This is an actual
connection-lifecycle race requiring correction and concurrency regression tests;
it is **not proof that this race caused the earlier process stall**. Additional
production fixes require approval under Hermes1's HyMem operations skill, so no
patch was applied silently as part of a version update.

## Verification

- Consistent production-store backup and ordinary initialization/reopen on a
  private copy: schema **61 → 61**, all **131 tables** logically unchanged,
  including vector/FTS contents. Integrity OK; foreign-key violations zero.
- Production lossless coverage: **585/585 valid**, no missing proofs, invalid
  frontiers or recorded coverage failures. Canonical drift zero; no messages or
  memberships attributed inside NULL-owned legacy sessions.
- Post-restart doctor: **zero failures, one unchanged historical warning**.
  Actual model completion and remote embedding connectivity passed. Producer
  identity matches; no feature-hash fallback is active. Both MCP environments
  match the Honcho embedding configuration, including `/v1`, dimension pin,
  deployment revision and tenant.
- Honcho, dashboard and skill-search health: **HTTP 200**. The shared embedding
  server also passed health and the actual embedding probe without a restart.
- Actual MCP protocol: initialize, ping and `hymem_profile` passed. Two regular
  watchdog/server pairs were present after restart. The verification client's
  separately created server closed normally.
- Extended MCP reads: all 12 tools advertised; profile, augment, digest and
  rule-list calls returned successfully, followed by another successful ping.
  During concurrent regression load, augment took 7.45 seconds and digest
  19.58 seconds. This is a successful functional probe, not a latency benchmark.
- Actual installed Hermes SDK: four mapped-session opens and four context reads
  across two config reloads passed; legacy source history stayed byte-equivalent.
- Actual SDK → private production-store clone: two captures and two new valid
  coverage proofs across manager/config reloads, **587/587 clone proofs valid**,
  unchanged legacy rows, integrity/FK checks passed. No model calls or synthetic
  production messages were used for this capture test.
- Dependency check: no broken requirements. Protected configuration hashes and
  Hermes2/Hermes3/embedding-container start times remained unchanged.
- All 478 deployed file hashes and the three installed capture-integration pins
  matched again after restart. `/dream-status` returned HTTP 200, zero coverage
  integrity failures, no running dream/lease and no pending aggregation. It
  reported **321 pending chunks**; no production drain was triggered. The most
  recent completed production dream remained 1403.

The installed-runtime gate completed with **166 passed, two failed**. The two
failures are the Honcho full-pipeline cooldown tests' five-second completion
assertions. A single unchanged rerun with our other test container paused passed
the first test and still failed the two-cycle test. A timing-only diagnostic
preserved that failure verdict while observing both cycles complete normally:
first dream 6.20 seconds, second 4.85 seconds, approximately 10.92 seconds from
the start of the five-second wait to completion. No scheduler exception occurred.
The **ten standalone scheduler tests passed**. The tests and their thresholds
were not changed, and a late completion was not substituted for a passing test.

The full regression gate was **gracefully interrupted at approximately 52%** to
request authorization for the newly reproduced defect. It is incomplete, not
a full-suite pass: its XML records **3,113 passed, one failed, zero errors/skips**
from 3,114 executed cases; the controller exited 2 for the deliberate interrupt.
Its recorded failure is
`test_locomo_rejudge_artifact_records_effective_body`: Hermes1's preserved
LoCoMo customization requires `args.judge_base_url`, while the existing direct
helper caller supplies the formerly supported namespace without that field.
The resulting `AttributeError` needs a compatibility fix or an explicit reviewed
contract/test update. This customization was preserved from before deployment;
it was not introduced by this fast-forward.

The initial overly broad collection
included an external Hermes-only test module in the HyMem environment and was
corrected before execution. An early in-container run was interrupted to permit
service restoration; neither attempt is counted as a completed full-suite pass.
The complete candidate suite ran in a separate, read-only, no-network container
with no production database mounted, and its source manifest matches deployment.
Four test workers use a shared hash seed because existing set-parametrized tests
otherwise collect in different orders across workers. The aborted parallel
collection and superseded serial attempts remain recorded and are not counted
as full-suite passes. Pinned parallel-test dependencies were installed only in
temporary space, not in either production Python environment. All test work is
stopped pending approval; live Hermes1 services remain available.

No scored LME/BEAM/LoCoMo benchmark, production dream, vector rebuild, database
repair, Git commit or push was performed. Scheduler regressions and private
capture tests must not be described as a newly completed production dream.

## Recovery material and receipts

On Hermes1:

- `/home/node/.hermes/backups/hymem-update-20260911/`: initial and offline consistent
  database backups, protected configuration backup, exact pre-update files,
  reconciled manifest and activation receipt.
- Retained pre-update tracked-code stash:
  `27691025cb19129bf9f8c33d2f883b0be8239a3c`.
- `/tmp/hymem-update-20260911.J8s0PkIC/`: preparation/rehearsal receipts, cloned
  capture verification and installed-runtime test XML.

On Afrodite:

- `/tmp/hymem-deploy-host-20260911.Dfz6CgIl/`: reviewed installer, pinned candidate,
  host deployment receipt and isolated full-suite receipts.

Key sanitized receipts and diagnostic scripts are additionally preserved under
`/home/node/.hermes/backups/hymem-update-20260911/verification/` so the next repair
does not depend on temporary test-container state. Temporary test containers
were stopped and removed; their reports and database/code backups were retained.

The September 10 recovery/quarantine backups, including the unique upload, remain
untouched. None of these backups is included in earlier cleanup authorization.
