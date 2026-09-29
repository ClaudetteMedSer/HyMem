# Retention inventory — bounded development pilot

## Status

The user requested continuation under their permission for necessary DeepSeek API
tests. The fresh 64-call pilot completed, separately from production. Collection,
accounting and cleanup passed. **The semantic pilot failed and review found a
fixture-contract defect; do not promote this candidate or use it as LME
clearance.** Independent source and matching reviews are frozen and reconciled.

This tests the [offline retention-inventory candidate](2026-09-18-retention-inventory.md).
It does not change the earlier failed 29/32 confirmation or reuse its spent
authorization.

## Frozen scope

Sixteen synthetic controls: twelve new development cases and the faithful/defective
pairs for the previously missed reimbursement prohibition and written-room
prerequisite. Two repetitions, each with one source-only inventory request and at
most one dependent matching request. Only the declared target scope is sampled;
the other scopes in each payload are preserved but outside this pilot.

Maximum **64 completions / 64 HTTP attempts**, no retries, rerolls, repairs or
resume. `deepseek-flash` at `https://api.deepseek.com`, temperature zero, thinking
disabled, JSON-object replies, 8,192 output tokens, 120-second owned invocation
deadline and two-second cleanup allowance. Returned model/fingerprint and usage
are recorded; the alias is not a weights-version pin. No baseline or paid judge.

Gold is frozen before collection: 22 targeted labels per repetition (15 retained,
five omitted, two altered), plus nine faithful whole-retention no-veto checks.
Independent source-only semantic alignment precedes candidate/verdict disclosure.
All inventories receive full-source review, but the targeted list is not claimed
exhaustive. The detailed protocol preserves malformed, uncertain, skipped and
incorrect results without shrinking denominators.

## Preparation and safety verification

A separate agent implemented the collector; root reviewed it, and independent
review covered the transport and container launcher. The final local gate passed
**174 tests** (60 collector, 114 transport/launcher). Root fixed overly permissive
JSON type equality, unexpected bundle-entry handling and the final launch-time
expiry check; collector corrections preserve explicit matching-bounds skips and
clear inherited process configuration. No production source was changed.

All eight faithful/defective or detailed/concise pairs have byte-identical source
inventory requests. Canonical source text, unit offsets and original scopes are
preserved. The frozen bundle contains 164 source files and six helper/input files,
plus its manifest; no credentials, production memory or gold labels.

The first portable transfer check rejected macOS metadata sidecars before any
container launch. That rejected bundle was preserved. Repackaging without those
sidecars changed no pinned source or input bytes.

The actual Linux rehearsal completed **64 owned workers**, zero HTTP attempts,
and clean exit/PID zero/no OOM. Root reviewed and ran the independently authored
receipt reconciliation: all 675 result files, all original requests, frozen
inventory bindings, monotonic deadlines and cleanup receipts agree. The rehearsal
used scripted outputs and demonstrates no model accuracy.

Live ran in a fresh read-only UID-1000 container with all capabilities dropped,
an empty production-home mount, read-only Python dependencies, and only the
explicit credential-file mount. The credential remains on Afrodite and enters
workers through private stdin, never logs or artifacts. No production store,
deployment, service restart or full benchmark is included.

## Pinned artifacts

Local receipts: `/private/tmp/hymem-inventory-live.OZQVF0`.
Remote isolated directory: `/tmp/hymem-inventory-live-OZQVF0`.

| Artifact | SHA-256 |
| --- | --- |
| Bundle manifest | `9e68b72eb4f628ed8a0591f639120a92316223a4d3b47489eb4f05a58569a30f` |
| Cases | `d08d8200c1779b8fa688ea7379d33877e1e79f06089b719ec0d1d738a4e7a9c5` |
| Gold (local only) | `e64ce625541217dc0fc1104470e4c9a167a4573a3d727c20ffbd5a06e9653881` |
| Collector | `649031a184658d5a705bee599c6be7c9a617ca58d5ae03d66c67a95814efef55` |
| Fresh authority | `50d72b7743e415951843b46bb46b5e6612eb79ddc7d831ced8bd5a2e2ff9e82a` |
| Rehearsal result-tree digest | `6e3a2b018b2c0ad68d689300b532c26eb9c47aad06b713534673b43d377d0278` |
| Live result-tree digest | `9c513c3cfa64ee57e3857b11f256ec9caaec057570dbcdb6efc11dcb75de065c` |
| Live accounting audit | `ff57eb85ccd072ec1c2523389f01432b469da8dda01497e98465e151ae5dc92a` |
| Root source-only review | `244e62564384322da60712edf8e1e0670f65f9e8ee28cdbfc7d513a6dabbc797` |
| Independent source-only review | `4dc6475fbefad12d0d5dd42c134857fae8831d810b8a6c739e0a5dc8f1ec8843` |
| Independent matching review | `b95618976650a6b1f6a1df117724f20fe4988f4da3ded95b017ec33724599c2d` |

The permission record quotes the user's broad API-testing permission and latest
`Continue`; the specific 64-call scope is explicitly root-derived, not presented
as a verbatim user quote. The fresh one-shot authority lasts three hours and
cannot resume or reuse an earlier run.

## Completed collection and source review

The isolated container ran September 18, 08:43:51–08:47:56 UTC, then exited with
code zero, PID zero, no OOM and unchanged isolation configuration. It made exactly
**64 completions / 64 HTTP attempts**, without retry: 32 inventories and 32
dependent matches. All replies parsed and finished with `stop`. All 64 workers
were cleaned up; the two independent accounting checks agree.

Reported usage: **72,493 prompt + 5,164 completion = 77,657 tokens**. Reasoning
token usage was absent in all 64 responses and remains unknown, not zero. Every
response reported `deepseek-flash`, fingerprint
`aeb56401ca74e127821c4f9126dcb669`; this is not an immutable weights pin.

Both source reviewers independently found 40 of 44 fixed targets fully represented
and four partially represented. The incomplete inventories quote a canonical noun
fragment without resolving its action and actor from legitimate same-message
boundary context. Both also found irrelevant background facts promoted to
mandatory retention, and a procedure's opening-before-collection order weakened
to two actions sharing an earlier prerequisite. Source review covered every
inventory; it does not prove that the remaining inventories are exhaustive.

The two previously missed omission mechanisms were caught in both repetitions.
Only **9 of 18 draws originally labelled faithful** avoided a veto. These are
frozen-label results, not a trustworthy whole-cohort accuracy estimate: independent
review subsequently found that three fixture families confuse procedure field
semantics. Confirmed false rejections still include both attribution draws and
all four incidental-detail draws, independently of that defect.

A scanner's matching requests were byte-identical across repetitions but received
conflicting verdicts: one rejected both constraints, the other retained both.
The supplied source roles, boundary text, candidate triggers and description are
present; this establishes judgment instability, not which conflicting verdict is
correct for every target and not a new transport loss.

### Fixture-contract defect discovered during independent review

`hymem/extraction/prompts/__init__.py` defines procedure `triggers` as words or
phrases used to **ask about** a procedure. They are not execution prerequisites.
The new scanner, cabinet and parcel fixture families put prerequisites only in
that retrieval field, and their labels wrongly treat this as preservation of
mandatory conditions. A trigger phrase alone cannot establish that a condition
applies to the procedure. This also makes the original faithful whole-scope
labels for those three families unreliable. The scanner's supervision constraint
is separately expressed in the description; it is not the same field-role issue.

The frozen gold, requests, outputs and scores must remain unchanged. Report this
defect alongside them, never silently reclassify cases to improve the score. New
versioned controls must put actual prerequisites in descriptive/instructional
fields, retain valid retrieval phrases separately, and keep the intended single
semantic mutation. The matcher also needs an explicit field-role contract; do
not fix the fixtures by redefining production `triggers`.

### Reconciled observations against the frozen labels

These counts retain all targets but **are not validated runtime accuracy** because
of the label/field-role defect above. Twelve condition-target observations belong
to affected pairs; ten individual labels are schema-ambiguous. The two defective
cabinet omission labels remain directly absent regardless of trigger semantics.

| Repetition | Correct | Incorrect | Ambiguous | Unassessed | Originally faithful: no veto |
| --- | --- | --- | --- | --- | --- |
| 1 | 14/22 | 5/22 | 1/22 | 2/22 | 4/9 |
| 2 | 17/22 | 3/22 | 0/22 | 2/22 | 5/9 |

The four unassessed entries are partial tray-event inventories, never successes.
The ambiguous entry has a correct dedicated frame-alteration judgment but an
incorrect retained judgment on a merged obligation containing that same claim.
Root's initial dedicated-only count credited it; independent review caught the
conflict. Final reconciliation conservatively denies success instead of choosing
the favorable duplicate. Both original reviews remain unchanged. All dynamic
obligation judgments and field locators remain in the receipts.

No known defective whole scope was affirmative (0/14), but unrelated vetoes and
invalid field-role labels prevent interpreting that as reliable defect detection.
The final reconciliation is
`/private/tmp/hymem-inventory-live.OZQVF0/final-semantic-reconciliation.json`.

## Remaining gates

The [sequential contract repair](2026-09-18-retention-contract-repair.md) preserves
all failed cases and fixes the diagnostic field-role mismatch first. Further
correction needs a fresh, versioned diagnostic and regression evaluation; the
spent pilot cannot be resumed or rerolled. No runtime integration, grounded
publication, original-Q1 convergence or fault-free LME has been established.
