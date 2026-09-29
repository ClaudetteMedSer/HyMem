# Source-review Stage A — runner verification

## Scope

This follows the frozen [offline evaluation design](2026-09-17-source-review-evaluation.md).
The user's “run it” is being executed as the stated next step: implement and
verify the offline runner. This step does not create or reuse paid authority,
contact a provider, transfer data to Afrodite, deploy code or touch production
memory, processes or credentials. The runner explicitly rejects live mode.

The frozen 24 controls, gold, prompts, 48-task schedule, caps and protocol are
unchanged. Runner artifacts are additive under
`/private/tmp/hymem-source-review-eval.85z9PG/runner-v1`; independent root controls
and receipts are under the sibling `root-controls-v1` directory. All original
source files and prior paid-run receipts are preserved.

## Verification method

A fresh implementation agent adapts the previously reviewed collection and
supervision machinery. Root reviews the implementation, runs separate adversarial
auditor tests, and then rehearses all 48 tasks twice: once with a dry client,
and once through the actual maintained SDK using only a dummy localhost server.
Each invocation still has a 120-second absolute monotonic deadline, two-second
cleanup allowance, one completion and at most three accounted transport attempts.
The parent reserves capacity before launch. Permanent per-mode fences prohibit
resuming a partial run or silently overwriting receipts.

The independent auditor imports no runner code. It verifies the exact frozen
request bindings, committed response/usage records, attempt intents, full task
inventory, unique reservations, cleanup and progress/summary totals. Wall-clock
call times and monotonic ownership/deadline times remain separate. The exact
auditor must pass actual rehearsal receipts, not just synthetic unit fixtures.

The loopback response sequence deliberately includes malformed, uncertain and
structurally permissive replies. Those are synthetic plumbing controls, not
semantic gold or observed model quality. Subsequent replay uses the frozen
parser/scorer and keeps malformed/uncertain outcomes separate from acceptance.

## Regression environment

The initial 20-module run encountered three sandbox permission failures while
binding a localhost test server, before any request. Its receipt is preserved as
`root-controls-v1/source-regression-gate.xml`. No code change was made to mask
those failures. Repeating the same selection with permission for dummy localhost
servers passed **1,456 tests**, zero failures/errors/skips, recorded in
`source-regression-permitted-gate.xml`. This includes the prior 1,395-test gate
and 61 supervision tests; these are not additional distinct copies of the prior
tests. No external provider request was made.

## Completion evidence

The local offline rehearsal is complete. **No paid/external provider call was
made.** There is no live authority and the runner intentionally has no live
execution implementation. Afrodite/Linux target-host rehearsal has not occurred
in this step. This document is not a live-launch authorization or an LME-readiness
claim.

### Auditor compatibility issue caught and corrected

All 48 dry invocations completed successfully. The first root auditor then
rejected the documented `cleanup_term_failed` and `cleanup_kill_failed` warnings
from the existing supervisor. That supervisor preserves these warnings only
after exit zero, authoritative child reaping and confirmed process-group absence;
its comments describe macOS exited-group signalling behaviour. The receipts do
not contain the underlying errno, so no exact errno cause is claimed here.

The old auditor had treated every warning as unresolved, despite those recorded
postconditions. A fresh implementation agent fixed only a new copy of the
auditor, under `root-controls-v2`. Root reviewed the minimal change, independently
checked all 48 successful-cleanup receipts and confirmed their process groups
were still absent. The corrected auditor permits only the two exact, nonduplicate
warning names after all existing successful-exit/reap/group-absence/empty-error/
deadline checks pass. Warnings remain visible in the audit output. Unknown or
malformed warnings and unsafe cleanup states still fail.

The original auditor, failed orchestration attempt and dry collection are
preserved. The completed dry collection was re-audited, **not rerun**. The
loopback mode had never started; the corrected driver launched its first and
only collection. No collection fence was removed or reset. Neither the runner
nor any frozen source, prompt, label or schedule changed for this correction.

### Results

- Source/scorer/supervision regression gate: **1,456 passed**.
- Final runner, corrected independent auditor and actual-receipt attack gate:
  **378 passed** (112 runner, 244 auditor, 22 actual-receipt checks).
- Combined final gates: **1,834 unique tests**, zero failures/errors/skips.
  Earlier 203- and 91-test receipts are intermediate subsets, not additional
  distinct passing tests.
- Dry collection: 48/48 invocations, 48 synthetic completions, zero HTTP attempts.
- Real maintained SDK against dummy localhost: 48/48 invocations, 48 synthetic
  completions, exactly 48 localhost HTTP attempts, zero external attempts.
- The loopback server independently checked each actual HTTP body against the
  frozen request, including model, sampling, JSON mode and thinking override.
  Its reported usage is deliberately invented and cannot establish real cost.
- The exact corrected auditor passed both actual collections. All 96 recorded
  workers had successful cleanup; a final read-only check found all 96 process
  groups still absent. The dummy server was closed and its thread joined.
- Parser/scorer replay covered 96 responses and 152 labelled-target instances.
  Dry replies produced 48 malformed scopes. Loopback replies produced exactly
  16 malformed, 16 uncertain and 16 structurally permissive scopes. These are
  deliberately scripted outcomes, not evidence of model quality or semantic
  verification.
- All **488 frozen source/test files**, requests, labels, schedule, protocol and
  original preparation artifacts remain byte-identical. Final frozen-artifact
  audit matched the pre-run audit exactly. `git diff --check` passed.

A fresh read-only review also found no blocking offline runner issue and used
four effective audit-hook probes to verify denial of dry socket creation,
external DNS, a wrong localhost port and UDP send. The subsequent actual-receipt
auditor correction above was separately implemented and root-verified.

### Receipts and pins

The completed receipt is
`/private/tmp/hymem-source-review-eval.85z9PG/root-controls-v2/full-rehearsal.json`.
`closeout.json` pins the final gates and receipts and explicitly records that
live execution, target-host rehearsal and LME readiness are not established.
The collected dry and loopback data remain under `runner-v1`.

SHA-256 pins:

- Runner: `b6a89de012a3235ec4eb0fd9f91f0a1bbba9119cc63546beb28ab0d672b82b66`.
- Execution plan: `91ac3c32c7236528222b5354abaa39be06d3e92a2a893feb2f92c6e8f4c14159`.
- Helper manifest: `d3f5d8ef00e39e26b36f9a9f64818f0778c9b65bf0f7463946b23c4d9a5f4d51`.
- Corrected independent auditor: `8bf08231b8441bfdfc17dc381d5d55cb5ee3c6ec667d783a9528ab3abc63de0b`.
- Completed rehearsal: `c02584838008b133e23527e97d05232ce7a0af54b8580f198874a611bc21eed6`.
- Final 378-test receipt: `3b5c96496d325cd89263c275183ebe79fccd056cd6642ad03837240fa1d5a5b3`.

## Next boundary

A live diagnostic requires a separately reviewed one-shot authorization/launch
path and an offline rehearsal in its target environment. Obtain fresh explicit
approval covering transfer and provider access before live execution; old paid
authority is spent and must never be reused. The proposed scope remains only
24 invented controls, twice: at most 48 paid completions / 144 HTTP attempts to
`https://api.deepseek.com` using `deepseek-v4-flash`. No production memory,
deployment, restarts, full LME, rerolls or resumed campaigns are included. Do not
infer model-quality or cost success from the synthetic local receipts.
