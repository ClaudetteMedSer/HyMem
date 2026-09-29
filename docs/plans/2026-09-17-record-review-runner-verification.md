# Record-review evaluation — offline runner verification

## Scope

Continue from the frozen [evaluation design](2026-09-17-record-review-evaluation.md).
Implement and verify an offline-only runner for the exact 30 cases / 60 tasks,
without new provider calls, external transfer, credentials, production stores,
deployment, restarts or full LME. Live execution is not implemented or authorized.

The 502-file source snapshot, candidate, scorer, requests, labels, paired schedule
and protocol stay unchanged. Add runner artifacts under
`/private/tmp/hymem-record-eval.sa5mWb/runner-v1` and root verification artifacts
under `root-controls-v1`. Preserve earlier packages and all failed receipts.

## Sequential work

1. A fresh implementation agent adapts the previously reviewed offline collector
   and tests. Reuse unchanged capture/supervision helpers. Root reviews its diff,
   tests request/accounting/cap/fence failures independently, and accepts the
   implementation before any collection.
2. Root independently adapts the corrected receipt auditor without importing
   runner code. Verify strict usage, owned-worker cleanup/deadlines, committed
   request/response evidence and complete task inventories. Keep replay/fresh
   cohort identities bound to schedule and output. Read all warnings rather
   than silently discarding successful-cleanup warnings or assuming errno.
3. Rehearse the full frozen schedule once in dry mode and once through the actual
   maintained SDK against a dummy localhost server. The server verifies exact
   HTTP request bodies. Synthetic malformed, uncertain and permissive outputs
   exercise plumbing, never claim model quality or use semantic gold as replies.
4. Independently audit actual receipts, replay the frozen parser/scorer while
   separating cohorts, run adversarial receipt tests, verify worker/server
   cleanup and recheck frozen hashes. Run relevant source/supervision regressions.

Each invocation has one reserved completion, at most three HTTP attempts, a
120-second absolute monotonic deadline and two-second cleanup allowance. Retain
permanent per-mode collection fences: no resume, overwritten receipts or reroll.
Continue after semantic rejection; halt on unsafe, unknown or invalid execution
accounting. Wall-clock transport times and monotonic ownership/deadline times
remain separate. Network guards deny external traffic, redirects, arbitrary
localhost ports and all dry-mode sockets. Only dummy credentials are accepted.

## Completed local verification

The offline runner and local rehearsal are complete. **No paid or external
provider calls occurred.** No source/candidate/scorer, frozen request, label,
schedule, production process, credential or memory-store changes were made.
Only the additive runner/control artifacts and this report were created.

The implementation agent preserved the original capture/supervision helpers
byte-for-byte and adapted request reconstruction to record-review-v3 using the
frozen `max_calls=3` preparation contract. The new runner binds all cohort fields,
60/180 caps and every frozen evaluation gate. Root reviewed the complete prior
runner and adaptation diff, checked its new tests, then ran independent boundary
probes before freezing helper/plan receipts and starting collection.

The initial inherited test caught a missing terminal blank line in the copied
capture helper. The agent restored its exact original bytes; the failed test
receipt remains preserved. No frozen implementation was changed to accommodate
the copy error. No other implementation or auditor correction was needed during
the actual collections, and neither collection was repeated or resumed.

### Gates and actual receipts

- Source/scorer/supervision regressions: **2,110 passed**, including the previous
  2,049-test gate plus 61 supervision controls. Dummy localhost binding was
  explicitly permitted; no external request was involved.
- Final runner/auditor/actual-receipt gate: **431 passed** (144 runner controls,
  244 independent auditor controls, 15 independent boundary/reconstruction
  probes and 28 actual-receipt attacks).
- Together: **2,541 unique test identities**, zero failures/errors/skips in the
  final gates. Earlier 403- and 259-test gates are subsets, not additional tests.
  This is a targeted gate, not a full-repository suite or live quality result.
- Dry collection: 60/60 invocations, 60 synthetic completions, zero HTTP attempts.
- Maintained SDK to dummy localhost: 60/60 invocations, 60 synthetic completions,
  exactly 60 localhost HTTP attempts, zero external attempts. The server checked
  each actual request's exact model, prompts, sampling, JSON mode and thinking
  override against the frozen request, in its scheduled position.
- Independent auditor passed both actual collections, not merely unit fixtures.
  Every invocation has a unique reservation and committed response/accounting
  evidence; no unfinished or unreconciled completion remains.
- All 120 owned workers have successful cleanup receipts. A final read-only
  process-group check found all 120 recorded groups absent. The dummy server
  closed and its thread joined.
- Real socket audit-hook probes blocked dry socket creation, external DNS,
  connection to the wrong localhost port and UDP send before I/O.

Both collections recorded `cleanup_term_failed` and `cleanup_kill_failed` for all
60 workers. The already-reviewed supervisor/auditor contract permits precisely
these nonduplicate warnings only with exit zero, authoritative reaping, confirmed
group absence, bounded cleanup, successful terminal receipt and empty errors.
Every postcondition was independently verified here. Warnings remain visible;
the receipts do not include errno, so no exact underlying signal-failure cause
is inferred. Unknown warnings or unsafe cleanup states still fail closed.

### Synthetic parser/scorer replay

The frozen parser/scorer processed all 120 synthetic responses and 136 labelled
target instances, retaining case/repetition/cohort identities and separate
view/domain accounting:

- Dry: 60 malformed scopes by design.
- Loopback: exactly 20 malformed, 20 uncertain and 20 structurally permissive
  scopes by design. Rejection did not stop later independent cases.
- Per mode: 48 fresh scopes / 52 labelled target instances, and 12 replay scopes
  / 16 labelled target instances. No pooling of cohort accuracy denominators.

Synthetic replies are generated from scheduled check shapes and position, never
from expected gold verdicts. The permissive replies are not proofs of meaning.
The localhost server's token counts (one prompt token and two completion tokens
per response) are deliberately invented; no real token usage, cost, accuracy or
relative benefit has been measured. The runner collects outputs without semantic
retries and the scorer records misses without repairing them.

All **502 frozen Python/SQL files and 38 original preparation artifacts remain
byte-identical**. Re-running the independent frozen-package audit produced the
same manifest content. No old paid output was accessed or rescored.

## Receipts and pins

Under `/private/tmp/hymem-record-eval.sa5mWb/root-controls-v1`,
`full-rehearsal.json` binds the collection audits, dummy-server receipt and
parser replay; `closeout.json` binds the final test receipts and records the
explicit local-versus-live limits. Dry/loopback collected files remain under
`runner-v1`. All per-mode and rehearsal fences remain intact.

SHA-256 pins:

- Runner: `ce619adb261e26548a116c43026798fda2f7a3130254834cbab6cf531311821f`.
- Execution plan: `9f31dd2af923c48f9d77222d5425b3b7d80e29f91a0d28209e39a9ec89a93edc`.
- Helper manifest: `666501495e8fef24050e6b1746b9db8ec270ba37765ce9230df029cc632c3631`.
- Protocol pin: `a13eb63cdc0a5a60303a356b680996c8b691e3bc62f629933cc6f1c04a208bdb`.
- Independent auditor: `60dbf9afc708978ef1468ef7dd80147ab119f6a893409146fe6b9164d6edf5e9`.
- Rehearsal driver: `569f326956c961c75d65b01ea0f9a534afa1924df44f67723f57aead83961ccb`.
- Completed rehearsal: `b4f3ef36b509300dea4ba88bc1441909c034bd4b00286134da493ce0556e9f3e`.
- Final 431-test receipt: `019bdc899f0a758fd33aa3d81125caa395e8df199650b9fd10b130514b989923`.
- Source/supervision receipt: `74baa5dca57a0d295bad8d02a5b1345878cf08741a579bf8e38a90a046b00f27`.

## Next boundary

Local offline runner readiness is established; **live semantic accuracy and LME
readiness are not**. Afrodite/Linux target-host rehearsal has not occurred. A
separately reviewed one-shot authority/launch path and a target-host offline
rehearsal are prerequisites for proposing the fresh live diagnostic. The runner
here intentionally has no live execution implementation.

Fresh explicit approval must cover transfer and provider access for the exact
30 benchmark-only controls, twice, at most 60 completions / 180 HTTP attempts to
`https://api.deepseek.com` using `deepseek-v4-flash`. Prior paid authority is spent
and is not reused. No production memory, deployment, restarts, full LME, rerolls
or resumed campaigns are included. Passing this rehearsal grants none of that
authority and does not change the frozen package's authorization flags.
