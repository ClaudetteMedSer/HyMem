# Enforced wall-clock cancellation for bounded validation

Scope: the user approved fixing the timeout gap exposed by the closed v10
validation. Implementation and verification are offline only. No paid rerun,
production deployment, service restart, store migration, or reuse of old consent.
Preserve all existing worktree changes and the complete failed v10 evidence.

## Diagnosis and integration boundary

The synchronous OpenAI/HTTPX read timeout is an inactivity limit, not a total
response deadline. A finite loopback reproduction showed keepalive newlines
extending a 100 ms timeout past 600 ms. The maintained client rejects a late
result once it returns, but cannot preempt that call, and close waits for active
calls. A future/thread timeout therefore does not provide safe cancellation.

Use a freshly executed child for each whole diagnostic invocation, including
primary extraction, optional summary repair and fidelity verification. Construct
SDK and immutable SQLite handles in the child, never fork or pickle live clients,
locks or database connections. An owning parent enforces the deadline and reaps
the worker. This fixes the supervised validation path, not every in-process
HyMem/LME/BEAM/MSC call; their callbacks and producer-bound transports remain
cooperative and need a separately designed broader integration.

## Sequential implementation and root verification

1. A separate implementation agent builds a reusable POSIX owned-process
   supervisor and offline tests. Root reviews its code and independently tests
   timeout, output-at-deadline races, cancellation and cleanup before integration.
2. A fresh implementation agent integrates the accepted supervisor into a new
   diagnostic revision, preserving the old v10 directory. Root verifies real
   extraction/repair/verifier behavior under the shared deadline, durable
   reservations, partial evidence, exact terminal accounting and no reroll.
3. Root runs finite and indefinite loopback keepalive tests, ordinary idle waits,
   startup/IPC/cleanup stalls, concurrent workers, ignored graceful termination,
   child crashes, receipt failures and successful controls. No provider access.

## Acceptance contract

- One monotonic expiry is established before child startup and is never reset by
  model retries, repair or verification. Explicit, separately bounded cleanup
  allowance; timeout never becomes a success because a child exits during cleanup.
- On expiry or parent cancellation, terminate only the owned process group,
  escalate if necessary, and reap. No next invocation while exit/ownership is
  uncertain. No wait-only timeout that leaves its network worker alive.
- Bounded startup/IPC/output/cleanup as well as network wait. Use a fresh exec,
  no inherited live objects. Secrets are never in argv, saved plans or receipts.
- Reserve the whole invocation and maximum HTTP attempts durably before it can
  run. Retain worker/parent evidence. Interrupted requests have unknown completion,
  token use and billing unless independently attested; unknown is never zero.
- Parent owns terminal verdicts. Late or partial results cannot be published,
  and failure aborts the batch with no paid retry or recycling of unused budget.
- Source database is immutable/query-only and never copied back or published.
  New live authorization must bind the new supervisor, worker and preparation;
  old v10 authority cannot run the new boundary.

Work evidence: `/private/tmp/hymem-wall-clock-v11.rEZRO9`.
Original failure: `/private/tmp/hymem-live-validation-v10.j272XK/VALIDATION_RESULT.md`.

## Progress and evidence

Phase 1 accepted by root on the frozen implementation
`9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc`.
The implementation agent supplied 53 offline controls. Root reviewed the code,
found and required correction of macOS exited-process ownership/signal handling,
and independently ran those 53 plus 8 root controls: **61 passed, 0 failures,
errors or skips** (`root-supervisor-final.xml`). Root's additional controls used
the real maintained OpenAI SDK against dummy loopback servers: finite keepalive
success, indefinite keepalive termination, and ordinary idle termination.

Maintained-client/deadline/digest regression gate: **346 passed, 0 failures,
errors or skips** (`root-maintained-regressions.xml`). Every one of the 513
previously accepted baseline source files remains byte-identical. New code is
additive; no existing user changes were reverted.

The supervising CLI must be single-threaded and translate/coalesce shutdown
signals into Python cancellation. SIGKILL, host crashes, escaping workers and
uninterruptible kernel operations require external containment; this boundary
does not promise to solve those.

Phase 2 is now accepted locally. A fresh implementation agent added
`/private/tmp/hymem-wall-clock-v11.rEZRO9/supervised_validation.py`, hash
`30ad4f6bd3e5f973f184aafd899d382d03d86380ceb97a32a2d396adabe14ab5`.
Root independently reviewed the integration, required canonical-stage authority
binding and stricter receipt/control validation, and verified actual maintained
SDK primary/compaction/verifier calls against dummy loopback HTTP.

The final combined gate independently run by root passed **89 tests, 0 failures,
errors or skips**: 61 supervisor controls and 28 integration controls. These
include indefinite keepalives at every extraction stage, actual CLI-parent
SIGTERM with worker reaping, immutable synthetic database checks, durable
pre-dispatch intents, malformed/missing result rejection, record hash checks,
control mismatch, obsolete authorization rejection and cross-directory no-replay.
The separate maintained regression gate passed 346 tests. All 513 baseline
source files still match their prior accepted hashes after the final gate.

Final combined JUnit SHA256:
`8e76983355d88ea2d8a927ae306a3657a5df8147fcf4630cbc4f7a6453a30a8c`.
Maintained regression JUnit SHA256:
`bcabe1526d5fc164b33a9a938e2a8bfc4f8c2bd2ef82b7db1fc731abcf22e5d8`.
Evidence and limitations are recorded in the private work directory's
`ROOT_VERIFICATION.md` and `README-supervised-validation.md`.

Status: implementation and local verification complete. No live authority,
paid execution, production deployment or migration was created. This fixes the
supervised validation path only; existing unsupervised production/full-benchmark
call sites remain cooperative. Fresh target preparation and approval are needed
before any live experiment; the closed v10 experiment remains closed.

## Authorized isolated Linux verification

The first upload attempt was blocked by the transfer safeguard. The user then
explicitly approved uploading source/tests for the isolated Linux check. That
approval does not authorize deployment, production-store access or paid calls.

Root verified the uploaded archive's exact inventory and hashes. An initial
archive included macOS resource forks; it was replaced before execution with a
manifest-only archive of 525 regular files. The container has networking disabled,
a read-only root, read-only source/runtime mounts, no capabilities, a PID reaper,
and bounded CPU/memory/PIDs. Only a new results directory and temporary synthetic
test data are writable. No production database, credential directory or Docker
socket is mounted, and the entrypoint runs the test gate rather than service hooks.

The first target gate used Python 3.11.2, OpenAI 2.53.0, HTTPX 0.28.1 and SQLite
3.53.4: **430 passed, 5 failed**, zero errors/skips/xfails. Root reconciled all
435 collected cases and 1,305 phase records with JUnit and the local inventory.
All source inputs stayed unchanged and the only blocked external socket attempt
was the deliberately denied guard self-test. Docker exited 1 without OOM.

The failures exposed undersized offline fixture budgets, not an escaped deadline:
the two-second budget could expire during cold SDK startup before the intended
HTTP stage was reached. Workers still timed out and were reaped safely. A related
agent-written check could pass with both request and intent counts zero, which
did not prove stalled-stage coverage. A separate implementation agent corrected
only three test files; root reviewed the complete diffs. Real-SDK fixtures now
use a finite ten-second startup-inclusive budget, dummy stalls outlast the budget
plus cleanup, and exact nonempty stage/intent sequences are required. Tests bind
reservation, launch and terminal records to the same expiry and still require
bounded cleanup, unknown interrupted usage, immutable source and no second task.
Production deadlines and supervisor/frontend/core code remain unchanged.

The agent's corrected local fixture gate passed 36 tests. Root's full corrected
Mac gate passed **435 tests**, zero failures/errors/skips, 211.980 seconds;
the Linux rerun also passed **435 tests** against the same fresh frozen bundle.
Root reconciled the local exact test-name inventory and rechecked all source and
diagnostic hashes. Local JUnit SHA256:
`44bda3a434ac6f32e1d5397cbc4edf38235fbed24b34fac7d74bb471adeeec2f`.
The initial root Mac rerun lacked loopback-bind permission (21 PermissionError failures,
414 passes); a focused check confirmed the sandbox cause, and the real rerun
uses approved loopback permission with the external-socket audit guard retained.
The failed runs and original inputs are preserved, not replaced by later passes.

Final Linux acceptance: 397.515 seconds in JUnit, zero failures/errors/skips/xfails
or deselections; all 1,305 setup/call/teardown reports passed. Root independently
reconciled test inventories, counts, hashes and phase records, then reverified
container exit 0, no OOM/error and the exact isolation/mount whitelist. The sole
warning was pytest's AnyIO pre-import/assertion-rewriting warning, not a HyMem
runtime failure. No network attempts occurred beyond the deliberately blocked
audit-guard self-test; the container also had OS networking disabled.

Linux JUnit SHA256:
`e89700bb3fcb936f508bb337959306d9f1f449388e8a6b37e4d18a105f871499`.
Full Linux phase/result receipt SHA256:
`5c055eff65cc1259aaee278863f5ea632a2bba346124ffbf8b6bd3cbd63bc777`.
Evidence: `/private/tmp/hymem-wall-clock-v11.rEZRO9/linux-revised-result.bRAG1F`.
The corrected bundle changes only the three offline fixtures. The supervisor,
frontend, core helper and all 513 baseline source files are unchanged.

Status: approved offline Linux verification complete. Both test containers have
exited and audit artifacts are retained. No production deployment, restart,
database access/migration or paid call occurred. This acceptance covers the
supervised diagnostic; unsupervised production/full-benchmark call sites and live
LME semantic/convergence outcomes remain outside this verification.
