# LME readiness — 2026-09-15

## Verdict

**Not yet cleared for the full paid baseline.** The last live diagnostic halted
on its first case: a source-recomposed summary failed reverification with
`summary_content_unsupported`. The 2026-09-16 local repair replaces generic
recomposition with source-linked targeted repair and separates mandatory format
verification from semantic checking. Its final local regression gates passed
**2,248 distinct selected tests**, with zero failures/errors/skips and unchanged
Python/SQL inputs. No new live-model reliability measurement or complete LME
smoke has passed with that source, and it has not been deployed to Hermes1
production.

The historical **1,790 selected source tests on Mac and Linux** and **274
diagnostic-harness tests on both platforms** below apply to the older v14
candidate, not the latest edits. See
`2026-09-16-targeted-summary-repair.md` for the new local evidence.

## Read-only checks completed

Checked on 2026-09-15, approximately 19:28 UTC:

- All 445 Python/SQL source-and-test files still match the accepted local
  format-repair gate's input manifest. That existing gate passed 1,665 tests
  with zero failures/errors/skips; it was not rerun or relabeled as a Linux gate.
- The authenticated Afrodite stage guard passed. Its frozen candidate,
  previously verified runtime and production source remain unchanged. The
  historical 7,406-test target gate applies to that older candidate, not the
  new format repair.
- Hermes1 is running, not restarting, with restart count zero and no recorded
  OOM kill in the current container state. Container identity still matches the
  reviewed deployment. These fields do not prove application correctness.
- From inside Hermes1, Honcho `/health` and
  `http://embedding-server:8766/health` both returned HTTP 200. These are liveness
  checks, not embedding-generation, vector-identity or indexing tests.
- A scoped process scan found no exact LME adapter process match. This is a
  point-in-time observation, not proof that all possible launchers or stale
  checkpoint locks are absent.
- The retained official S dataset has 500 distinct question IDs and matches
  SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
  All six category counts match the pinned protocol.
- The filesystem containing Hermes1's home had 293.84 GiB free (30.1% used).

The deployed `hymem/dreaming/digest.py` has SHA-256
`25a1c4279d54f7e243097c13e84e53ff3619e55cb5b739a7e3438ce3b97de0c9`.
The locally accepted replacement has SHA-256
`fdbb1a7df3be9df05b7e4df958fcfb45f9f7754a0b3d66afce2a934391b465df`.

## Original gate sequence (status updated below)

1. Freeze the latest source into a separate benchmark candidate on Afrodite;
   verify its exact source and run the relevant offline tests in the target
   runtime. A benchmark-only checkout does not require deploying to production
   or restarting Honcho/MCP.
2. Rebuild and independently verify the diagnostic's request accounting for
   the new fourth-stage format adjudication. The closed, failed diagnostic's
   old 20-completion/60-HTTP authority cannot be reused. A new four-pipeline,
   eight-control diagnostic reserves at most 24 completions / 72 HTTP attempts,
   with an owned 120-second deadline per invocation, stopping at first failure.
   Approval for this fresh paid diagnostic was requested during this inspection
   and had not been received when this report was written.
3. After that diagnostic passes, obtain a budget for a fresh end-to-end LME
   smoke, including entry canary, ingestion, in-run dreams, healthy convergence,
   answering, judging and final checkpoint/artifact publication. A canary pass
   alone did not prevent the prior Q1 failure and is insufficient now as well.
4. Freeze the intended baseline recipe and exact source identity before the
   full run. Keep health strictness enabled and preserve the paired baseline's
   dataset, seed, reader, judge, embeddings and retrieval settings. Do not
   resume an older-identity checkpoint as though it used the new code. A small
   smoke remains diagnostic, not a canonical 500-question score.

Passing these gates would support starting the full run; it would not guarantee
that a nondeterministic provider cannot fail later. No full-run fault-free claim
is made here. No provider calls, remote writes, production database opens,
deployments or restarts were performed during this readiness inspection.

The preceding implementation and its private receipt locations are documented
in `2026-09-15-format-adjudication.md` in this directory.

## Historical follow-up before explicit payload-transfer approval

The latest code is now frozen in an isolated Afrodite benchmark candidate.
Its selected **1,665-test Linux gate passed**, as did **228 helper tests on both
Mac and Linux** and the independently audited **12-task, 19-synthetic-call**
retained-input rehearsal. These checks made zero provider calls and did not
deploy or restart production. Two offline container setup failures were fixed
without changing the memory code; all original failed receipts are preserved.

The paid diagnostic has **not started**: auto-review blocked fresh authorization
pending explicit approval naming the retained benchmark text, exact DeepSeek
endpoint and 24/72 call limits together. The fresh live diagnostic and an
end-to-end LME smoke remain outstanding, so the full-baseline verdict is still
not cleared. Detailed evidence is in `2026-09-15-format-live-diagnostic.md`.

## Prior v13 result after explicit payload-transfer approval

The user approved sending the four retained cases and eight controls to the
specified DeepSeek endpoint under the 24/72 cap. Fresh preflight passed and the
single diagnostic executed. It halted on the **first case after two completions /
two HTTP attempts, in 7.41 seconds**, with `summary_content_unsupported` during
first-pass fidelity verification. Both HTTP responses completed normally.

The verifier accepted both episodes and their formats, and accepted summary
formatting, but rejected summary content. No format-adjudication call occurred:
content rejection correctly remains a veto. This does **not** establish whether
the generator introduced unsupported content or the verifier falsely rejected
a supported summary. The other 11 tasks were not attempted. No reroll or reuse
of remaining allowance is authorized.

Root independently audited usage and committed receipts, first-failure stop,
worker cleanup and unchanged benchmark-source hashes. A final read-only
postflight confirmed the installed runtime, production source and container
identity unchanged. No production database was opened, no deployment/restart
occurred, and no end-to-end LME run was launched.

Next steps recorded at that checkpoint were:

1. With explicit permission to inspect the minimum raw benchmark evidence,
   compare this rejected summary and verdict against its source. This review
   needs no additional provider calls. Preserve the factual veto while deciding
   whether the problem lies in generation or verification.
2. Implement and independently verify any demonstrated defect, then seek a fresh
   bounded live check; the executed diagnostic is closed and non-resumable.
3. Only after that passes, obtain a separate budget for the complete end-to-end
   smoke described above. A canary or selected regression gate alone cannot
   establish full-baseline readiness.

Current verdict remains **not ready for tomorrow's canonical baseline**.
Full receipt hashes and the closed-stage outcome are recorded in
`2026-09-15-format-live-diagnostic.md`.

## Follow-up: source review and local repair

The user authorized the minimum benchmark evidence review, including separate
approval for the prior rolling summary and 48-character boundary context.
Earlier topics and the riding request are supported by that continuity material;
no evidence-wiring defect was found. The rejected summary omits the explicit
order of two completed trip stops. That is a plausible fidelity defect, but the
verifier supplied no rationale, so its exact reason remains unknown.

A separate agent implemented one source-only summary recomposition after an
otherwise valid summary-content rejection. Unchanged or invalid rewrites stay
held; a changed summary must pass complete verification again, with items and
citations preserved. Root accepted a frozen 561-test core gate before moving to
a new agent's recorder fix, then accepted that fix with a frozen 129-test probe
gate. The recorder no longer assigns a first-verifier response to a failed
second verification that never dispatched. Normal cost remains two completions;
the combined worst case is now six.

The final combined local gate passed **1,790 tests, zero failures/errors/skips**,
with all 449 Python/SQL input hashes unchanged and zero network attempts. This
is a selected regression gate, not the full suite. Precise evidence is recorded
in `2026-09-15-summary-content-recovery.md`. These checks do not provide a new live
model result or deploy the new source to Hermes1. The former diagnostic remains
failed and closed. Before another live check, freeze and verify the new target
candidate and adapt/review its supervisor and accounting for the new task.
The same four-pipeline/eight-control plan would need at most **32 completions /
96 HTTP attempts**, not the old 24/72 allowance; this is a future bound, not
permission to run it. An end-to-end LME smoke is still required afterward.

**LME readiness remains unconfirmed; do not treat local regression success as
clearance for the canonical baseline.** No new provider calls, production
deployment, service restart or production database opens occurred in this repair.

## Latest v14 result: bounded recovery executed, live acceptance still fails

The user explicitly approved one fresh four-case/eight-control DeepSeek
diagnostic under the new 32-completion/96-HTTP limit. Before dispatch, root
accepted exact-inventory Linux source and helper gates and an independently
audited twelve-task, 23-synthetic-call no-network rehearsal. Fresh version-4
authority was bound to the current source, helpers, gates and approval.

The single live run stopped on its **first case after five completions / five
HTTP attempts in 10.908 seconds**. All responses returned normally. The new
source-content recovery ran and its changed summary reached full reverification,
but both initial and final verification marked summary content and formatting
unsupported. Both episodes passed their title/content/format checks. The final
failure was `summary_content_unsupported`; no factual veto was bypassed and no
format adjudication occurred. Eleven tasks remain unattempted, not passed.

Root verified usage, exact replay, first-failure halt, worker cleanup, unchanged
benchmark inputs and unchanged installed runtime/production source. No production
database open, deployment, restart, reroll or full benchmark occurred. The run is
closed; unused allowance cannot authorize another diagnostic.

**The canonical baseline is still not cleared.** The next diagnostic step is
reviewing the new rejected candidates and verdicts against the previously
approved evidence, without further provider calls, to distinguish genuine
summary defects from verifier false rejection. Detailed receipts and exact
hashes are recorded in `2026-09-15-content-live-diagnostic.md`.

## 2026-09-16: authorized evidence review completed

Both rejected summaries omit an explicit temporal ordering from the source.
The repaired summary also received a format rejection despite being a single
grammatical sentence of the permitted form. Exact source, prior/context,
candidate and parameter checks found no request-wiring defect. This supports
separating generation fidelity from verifier reliability; the categorical
verdicts do not reveal the verifier's precise rationale.

The proposed next fix is evidence-linked violation diagnostics and targeted
repair with fresh verification, not more generic retries or a relaxed factual
veto. This read-only review made no runtime changes or new provider calls.
LME is still not cleared. See `2026-09-16-content-diagnostic-review.md`.

## 2026-09-16: local targeted-repair implementation accepted

Separate agents implemented source-linked rejection diagnostics and targeted
repair, mandatory candidate-only grammar verification after semantic acceptance,
and recorder/simulator/cost alignment. Root independently reviewed each step and
accepted final frozen gates covering **2,248 distinct selected tests**. The
audit proved the entire original selection was retained, all 453 Python/SQL
input hashes remained unchanged, and no network/provider calls occurred. A
missed explicit test-client response was corrected and strengthened; its failed
receipt remains failed, with the complete final selection rerun successfully.

Ordinary successful digest slices now cost three logical completions; the
maximum stays six. Factual vetoes, exact source/citation scope, unchanged prior
continuity, one bounded repair and absolute invocation deadlines remain intact.
Details, limitations and hashes are in
`2026-09-16-targeted-summary-repair.md`.

These local results do not transfer the historical Linux gates to this source,
reopen the spent v14 authority or establish LME readiness. Next: adapt and review
the private diagnostic bindings for the new wire contracts, obtain fresh bounded
live-run authority, then require a successful end-to-end smoke before a full
canonical baseline. No deployment or production mutation was performed.

## 2026-09-16: fresh v15 diagnostic halted before targeted repair

The user requested the fresh bounded run. Root accepted the exact 2,248-test
Linux source gate, matching 304-test local/Linux helper gates and an independent
twelve-task/25-synthetic-call no-network rehearsal. A fresh, single-use v5
authorization bound those receipts and the current source to the 32-completion /
96-HTTP cap and 120-second invocation deadlines.

The live run stopped on **task 1/12 after three completions / three HTTP
attempts in 9.491 seconds**. Primary generation and summary compaction returned
valid JSON. The semantic verifier returned a 794-character incomplete JSON
reply (254 completion tokens against a requested maximum of 3,072), producing
`fidelity_parse_failure`. Its provider finish reason was not captured. Targeted
repair and final format verification were never reached; eleven tasks remain
unattempted.

Root confirmed clean worker shutdown, exact usage/receipts and unchanged
benchmark inputs and installed source/runtime. No further provider calls,
deployment, service restart, production database open or full LME benchmark
occurred. The one-shot run is closed, not resumable under unused capacity.

**LME remains unconfirmed.** The current immediate blocker is malformed verifier
output, not a measured success or failure of targeted summary repair. Full
receipts and limitations are in `2026-09-16-targeted-live-diagnostic.md`.

## 2026-09-16: saved-verdict parsing blocker fixed and verified offline

The authenticated failed reply contains complete verdict and diagnostic items;
only its final outer `]}` is missing. A dedicated agent implemented narrowly
bounded verifier-only envelope recovery, and root independently verified the
original saved reply in isolated, read-only, network-disabled validators. The
result changes from `fidelity_parse_failure` to the original intended
`summary_content_unsupported`, preserving both exact-source `omitted_outcome`
diagnostics. No rejection was converted into acceptance.

Root accepted the entire previous selection plus new controls: **2,473 distinct
application regression tests passed**, with frozen inputs and no network calls.
A second agent fixed the affected private diagnostic consumer, exact raw-response
validation and future finish metadata. Root independently accepted its **393-test
helper gate** using synthetic stores and dummy loopback HTTP only. The totals
overlap and must not be summed as a repository-wide unique count. Full evidence
and source hashes are in `2026-09-16-verdict-envelope-repair.md`.

This fixes the local parse blocker, not the untested later model-generated
summary repair. No new paid calls, production deployment, service restart or
production database open occurred. The v15 run remains failed and closed;
new helper copies are offline drafts needing fresh source/target binding and
authorization. **LME remains unconfirmed** until the bounded live diagnostic and
end-to-end smoke pass.

## 2026-09-16: fresh v16 diagnostic halted at summary compaction

After fresh independently accepted Linux gates (**2,473 source tests and 393
helper tests**) and a twelve-task no-network rehearsal, the user's “Run it.”
authorized one fresh 32-completion / 96-HTTP diagnostic. It stopped on task 1/12
after **two completions / two HTTP attempts in 7.307 seconds**.

Primary generation returned a 507-character summary; the single compaction
returned 523 characters, despite the correct explicit 350-character target and
500-character maximum. Both replies were valid JSON with provider finish reason
`stop`, far below their token limits. The failure was `summary_output_cap` at
`summary_compaction`. This is length noncompliance, not token truncation or a
new verifier parsing failure. No verifier was reached; eleven tasks were not
attempted, and the new verifier-envelope fix received no live exercise.

Root confirmed exact usage, mechanical replay, first-failure halt, clean worker
shutdown, unchanged benchmark inputs and unchanged installed source/runtime.
No deployment, restart, production mutation, full benchmark or automatic reroll
occurred. The single-use run is closed. Full evidence and counts-only audits are
in `2026-09-16-verdict-live-diagnostic.md`.

**LME is still not cleared.** The next implementation target is the bounded
summary-compaction path; accepting or silently truncating overlong summaries is
not an appropriate workaround. A successful fresh diagnostic and end-to-end
smoke remain prerequisites for the canonical baseline.
