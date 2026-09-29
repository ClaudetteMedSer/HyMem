# Compact-assessment runner — offline verification completed

## Status

This section records the state at the end of the offline-verification turn,
before the user's subsequent explicit approval. See the follow-up below.

The corrected diagnostic runner passed the complete network-disabled Afrodite
rehearsal. **No paid provider call was made; no live-run authority was created.**
Production HyMem, Hermes processes, stores and credentials were not modified.
This is runner verification, not model-quality evidence or LME clearance.

The proposed live comparison remains the frozen September 16 design: 49
benchmark-only cases, two repetitions of baseline versus compact assessment,
196 case-arm invocations, at most 326 completions / 978 HTTP attempts to
`https://api.deepseek.com` using `deepseek-v4-flash`.

The automatic approval reviewer blocked the authority-creation command because
it could verify the earlier explicit 238/714 approval, but not the expanded
326/978 scope from the latest “Continue please.” reply. The command did not run.
A subsequent read-only probe confirmed both `paid_authority_created=false` and
`live_started=false`. Do not reinterpret the prepared consent metadata as an
executed or accepted authority. Explicit approval of the larger scope is needed
before any paid launch; do not bypass the rejection, silently reduce the frozen
matrix, reuse old paid authority, resume an old campaign or reroll.

## Runner defect caught before provider access

The first offline rehearsal halted at task 7. Six tasks completed, using 12
synthetic completions and zero provider attempts. Its owned worker was reaped;
the failed rehearsal and fences were preserved in the original staging directory.

The diagnostic reconstruction check compared the baseline's original serialized
JSON with a fresh serialization of the fixture payload. The fixture artifact
sorted object keys; the original baseline used its construction order. Strictly
decoded data matched, but bytes differed. This was a runner-check defect, not a
model rejection or a change to production extraction.

An implementation agent corrected the check, followed by independent agent and
root review. It now requires strict decoded equivalence, unchanged system and
request parameters, and acceptance by the frozen runtime encoder, while keeping
the original baseline request bytes. Duplicate keys, nonfinite values, type or
content drift remain rejected. Compact requests still require exact runtime
reconstruction. All 49 frozen requests and negative controls were exercised.
No case, gold label, prompt, scoring threshold or runtime source was changed.

The corrected attempt used the fresh `verified-run` child directory, never
resetting or resuming the failed attempt.

## Verification results

- Local runner/rehearsal/reconstruction selection: **343 passed**.
- Separate local independent scorer/auditor selection: **165 passed**.
- Afrodite Linux selection: **343 passed, zero failures/errors/skips**. This
  repeats the local selection and is not 343 additional distinct tests.
- Full dry pass: 196/196 tasks, 326 synthetic completions, zero HTTP attempts.
- Full dummy loopback SDK pass: 196/196 tasks, 326 synthetic completions and
  exactly 326 localhost HTTP attempts, zero external provider attempts.
- Independent parser replay: all 652 request records matched the frozen plan.
  Deliberately invalid synthetic replies produced 196 baseline rejections and
  456 malformed compact scopes; these are test outcomes, not model-quality scores.
- All owned invocation cleanup and accounting reconciled. The isolated gate
  container exited 0, with no OOM. Root retrieved and hash-checked the plan, JUnit
  receipt, gate receipt, full rehearsal receipt and parser replay receipt.

The gate container had network mode `none`, a read-only root filesystem, dropped
capabilities and no production-memory or credential-file mounts. Only the
diagnostic stage and the Python environment required for verification were
mounted. No production process was restarted.

## Preserved artifacts

Local corrected stage:
`/private/tmp/hymem-assessment-eval.dHR3MU/verified-run`

Remote corrected stage:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-assessment-comparison-20260917-dHR3MU/verified-run`

The original failed rehearsal remains in the parent remote directory. The
corrected diagnostic container is
`fbc26c9986913a47cad18dd80d710e85cb14cf9cd970e8fe6014779f44c09a8c`
and is exited. No old receipts or fences were removed.

Key SHA-256 pins:

- Remote execution plan: `3712f046866af1b5b3c851dd0b7b49886a43acf622e0e335e347f72cb41afef0`.
- Corrected helper manifest: `6be82d799b648814e76f361b3f5190f08b80ec7c337b90609bb6e6435e40f21d`.
- Corrected runner: `2918ad7fb1f642c4d0c630cf2401e11cc6b06c0d75271208e726079241bf03a8`.
- Linux gate: `24370e3f6ebc403dd431e8bc2e1e8f171f905343dabe1b58dca871fd6daa06dc`.
- Linux JUnit: `98bedeacd781110a5b6f735435d66bee188744be1f675bb16e288ed37a4f5b5c`.
- Full rehearsal: `858352c21214a6b44e9b08102692a53f689de57d3574913f2a920768780379e7`.
- Parser replay: `89c0d1e7c9330c70245c48a63702b42634e807e0b04b55a715ea85ff943490bf`.

The original fixtures, gold, 476-file source snapshot and September 16 frozen
protocol retain their original hashes. Keep that preregistration unchanged.

## Next step

Obtain explicit approval for the 49-case benchmark-only transfer to DeepSeek and
the 326/978 paid-call cap, with the existing exclusions. Reconcile that explicit
approval in the authority/audit evidence before launch. Then collect once, audit
all accounting and owned-process cleanup, and score the complete frozen matrix
without semantic retries. Runtime adoption, deployments, restarts and full LME
remain outside this diagnostic's scope even if its model-quality gates pass.

## Follow-up: explicit approval and live dispatch

The user subsequently answered **“yes”** to the explicit 49-case DeepSeek
transfer and 326-completion/978-attempt question, including the exclusions above.
A separate `provider-confirmation.json` records that reply and exact question;
the old preparation records remain unchanged and cannot substitute for paid
consent. An independent agent updated the receipt auditor, root reviewed it,
and **183 local scorer/auditor tests passed**, including 18 added consent controls.
These replace the earlier 165-test selection; do not sum the repeated tests.

The confirmed authorizer rechecked the full offline gate, frozen inputs and
absence of a prior live launch. It created authority
`49a9d473772511363e1749e6ee8145184b7ef5b66d53aebafe58f41f3d5f10bd`.
The unchanged one-shot dispatcher accepted it and launched the approved
comparison on Afrodite. This was a fresh run, not a resumed or rerolled campaign.
All five runtime helpers, all 476 frozen source files, requests, labels, sampling
parameters and limits are unchanged. Live results still require complete
collection, independent accounting/cleanup verification and frozen scoring.

The run subsequently completed and was audited/scored. See
[the completed comparison](2026-09-17-assessment-live-comparison.md), including
the independent auditor's clock-domain defect, its correction and the failed
model-quality gates. The live authority is consumed; it must not be reused.
