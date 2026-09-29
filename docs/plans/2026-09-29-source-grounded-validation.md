# Source-grounded relationship validation

## Authority and evidence

On September 29 the user approved extending the repair to source-grounded
semantic validation, implemented by separate Sol agents with root verification
between stages. This supersedes the previous pause for that scope decision,
not the restrictions on production, models, spending, or immutable old runs.
The observed canary's exact offline replay established a predicate substitution
with otherwise correct entities, polarity and citation. The parser and strict
canary behaved correctly. The original transport incident remains unresolved.

Production and all old runs remain stopped/unchanged. The existing monitor
stays paused until a newly accepted, receipt-bound live diagnostic exists.
No full-500, blind reroll, model/auth switch, quota bypass or cap increase.

## Contract and acceptance

The narrow first contract covers extracted triples, including their qualifiers;
markers remain unchanged under the existing marker contract. It does not claim
new semantic certification of markers or completeness beyond the existing
omission check. Validate the merged primary-plus-omission triples before any
nonempty result publishes. Use bounded batches and the existing logical-call /
provider-attempt budget. A new check is a model judgment, not a proof of truth.

Each verdict is bound to the exact canonical batch of candidate identities and
owned sources. Source context may interpret a claim only within its existing
applicability range, never independently own it. Supported claims and proposed
predicate corrections require exact source evidence, with mechanically checked
quotes and owned-fragment contribution. Span existence alone does not establish
semantic entailment; independent tests must measure false acceptance/rejection.

Finite outcomes: supported, unsupported, uncertain, or predicate replacement.
Only predicate can change. Subject, object, polarity, source, qualifiers, count,
markers and other claims are preserved. Recheck the entire corrected result
once; a second correction, uncertain/unsupported item, malformed/incomplete
verdict, provider error or exhausted budget fails the whole unit. No silent
filtering, partial publication, recursive correction or recovery split that
erases the rejected claim. Detect correction collisions/conflicts atomically.

Legacy unattributed text can be checked as the exact input without inventing
message provenance; it must never acquire a fabricated source-message ID.

## Sequential implementation plan

1. Separate Sol: pure versioned request/response/evidence contract in a new
   extraction module, with bounded typed input, finite diagnostics, immutable
   batch binding and adversarial offline tests. No activation or model calls.
   Root reviews and independently tests before accepting.
2. New Sol: integrate the accepted contract into the actual extraction
   publication boundary and cache identity. Preserve the frozen candidate's
   newer extraction fixes; do not replace it with the older checkout module.
   Charge every check/recheck against existing caps and add explicit source-free
   accounting. Root tests atomic failure, correction, conflict, provenance,
   context, budget exhaustion, legacy input and all extraction paths.
3. New Sol: independent invented positive/negative controls covering preference,
   use, implicit support, negation, qualifiers, cross-paragraph/table/context
   references, multiple supported relations and malicious/ambiguous content.
   Root reviews expected labels independently. Version canary path accounting
   honestly without weakening its semantic oracle or hiding extra calls.
4. Only after offline acceptance: freeze a new candidate, predeclare a bounded
   diagnostic with immutable source/data/receipt identities, test retained
   failures privately and invented controls, and root-replay captured outputs.
   Measure false rejection and extra latency/calls as well as semantic accuracy.
   No automatic repeat of the previously accepted paired grounding campaign.
5. Only after diagnostic acceptance consider a fresh same-four-question pilot,
   using a newly bound monitor and existing budgets/resource/cleanup policy.
   Report index/summary health separately from answer correctness; require
   complete accounting and independent cleanup for success.

This adds model calls and may exhaust an unchanged cap earlier on dense cases.
Do not call the system faster or ready until measured. Any needed cap expansion
or materially broader repair requires direction.

Official evaluation guidance informed the use of fixed classification criteria
and independent edge-case controls rather than accepting model self-grading:
https://developers.openai.com/api/docs/guides/evaluation-best-practices

## Status

Design reviewed read-only by a separate Sol. Root accepted Step 1 after reviewing
the implementation and independently running 78 offline tests (25 Sol controls
plus 53 root controls). Root found and reproduced an integer-overflow escape;
Sol repaired it and the independent controls now pass. The bound context schema
preserves role, peer, time and original context-message identity, with separate
regions for each of two conversation records and their table headers/preludes.
Context citations never become ownership of the candidate claim.

Initial pure module SHA256:
`6cdf734a10dc14f2352323710219fa3475ea8363c1b813b7ea4e4a3a9e7943bc`.
This is wire-contract acceptance, not evidence that the model's semantic
judgments are accurate. No activation, remote change or model calls occurred.

Step 2 will use a new deterministic derivation of the actual frozen runtime,
not replace newer frozen extraction fixes with the older checkout module.
Original sources, inventory and receipts stay immutable. A new helper/module
and exact source-bound integration must be independently reviewed before any
subsequent control or live-diagnostic implementation.

During integration review root identified a missing second-level scope: a
conversation context can retain a table header applying only to a prefix of
that context's body. The original Sol extended the unpublished contract with
bound parent-region/prefix fields; parent provenance must match, and quoting a
nested header requires an in-range parent-body witness as well as owned text.
Complete conversation text is preserved, including unrelated later prose.
Root independently tested valid, out-of-range, missing-parent, cross-record,
metadata-mismatch and dual-context cases. The evidence array bound is eight
instead of four so both existing conversation records and their table context
are representable; invocation output maximum remains 4096 tokens and all
campaign/call limits remain unchanged. Truncated verdicts still fail atomically.

Step 1 reaccepted with **93 passing offline tests** (27 Sol and 66 root), raw
module SHA256 `dd1a49b56abf569b4476a0b735e67e72a86723bf88e3a4739998aceeabe19c18`.
No model calls, frozen-source edits or deployment. New Sol is implementing
Step 2 against this final contract.

## Step 2 accepted offline

Root independently reviewed the gate, exact guarded derivation and identity
binding, then ran **129 tests**: the 93 contract tests, 19 Sol integration
tests, and 17 independently authored root integration/identity tests.
Nine-claim/two-batch controls prove full-result recheck, unchanged qualifiers,
markers and hints, atomic negative/second-correction failures, cross-batch
collision/conflict rejection, and exact accounting of three provider attempts
per synthetic grounding invocation. Exhausted budgets remain resource failures.
Root also mutated live prompt, bounds, mapping/validation function bodies and
import bindings: each changed cache identity or failed closed. The original
508-file runtime was independently reverified unchanged.

Root requested and reverified two final safety details: output stamps cannot
write inside the original tree, and initial/recheck calls use separate source
lines so stage timing needs no frame-local or request-text inspection.

Accepted inactive local candidate:
`/private/tmp/hymem-semantic-step2-v2-candidate-20260929`.
Exactly two original files change (`chunk.py`, `contract.py`) and two modules
are added (`grounding.py`, `grounding_gate.py`); all other 506 original files
are byte-identical. No old frozen file or receipt was overwritten.

- 510-file map: `e9ca47f85046d4ad980a8abef0302e1ec36a9a91cc9616bb043784bc152640fb`
- Inventory file: `11ca4cdbba18e4b7e4b56d444062e39b789820b2f0055a32898bbd0b3a1e4664`
- Gate: `bb79e1b0baa1a16032532fb73ec448a7dd3dcab94fbf87f69b6e7931489f03f8`
- Builder: `a10ee5c5a1ba4f6a2694a88570a399c5fe081db02f0cb0ecfa5b92e5db06d7b2`
- New extraction identity suffix: `f349e2fa14d1778bc556869d346ca44183c025e779a919b3f15e82f7c3d78d46`
- Prior identity suffix: `adf248f4c33516573859422942386faacdd97bd65c7b0bd83a0854e323d90795`

Identity prefix remains `hymem-extraction-contract-sha256-v1:`. These are offline
mechanical gates, not a full-suite/live-accuracy result. Independent semantic
controls and versioned canary/stage accounting are next, before any paid call.

Root added two real Phase-1 publication controls using new temporary SQLite
stores and a fixed synthetic client (network disabled). A rejected claim writes
only one failed-attempt record: zero graph rows, observations or processed
markers. A corrected-and-rechecked claim stores exactly the corrected predicate,
one observation and one processed marker. Both pass, bringing the cumulative
offline gate count to 131; no production store was opened or modified.

## Independent controls accepted

A new Sol supplied 24 invented cases (28 candidate triples): 12 supported,
two required predicate recoveries and ten rejecting units. Root reviewed every
label, rejected ambiguous selection-as-preference examples, and required an
actual table row whose relationship depends on its header. Negative controls
accept either unsupported or uncertain (both reject publication); either label
on a positive is false rejection, and on a correction objective is missed
recovery. The current prompt does not guarantee it will choose correction.

Fixed fixture-plus-label SHA256:
`511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925`.
Root reran five construction tests and a separate physical-candidate test
checking 48 exemplar wire verdicts, provenance/qualifier binding and changed
correction hashes. All six pass. These exemplars are not model judgments or a
semantic accuracy score. No live calls have occurred.

Next: a new Sol versions the canary and stage accounting, retaining the exact
fixture and final semantic oracle while exposing every added validation and
recheck call. Old runners/readers/canaries and their receipts stay untouched.

Root reran all 137 accepted contract/integration/control tests together: all
pass. A separate network-disabled run against the physical candidate passed
26 existing fragment-reading-order and prompt-contract tests. This is a scoped
regression check, not a full-suite claim: older extraction test doubles do not
provide the newly required grounding verdicts. Live semantic accuracy remains
unmeasured. During review of the next draft root identified that corrections
are rechecked per leaf, not after all leaves; the new canary must track that
actual ordering instead of imposing a misleading global two-pass order.

## Canary and stage accounting accepted offline

A separate Sol supplied new versioned canary/accounting modules; the old ones
remain unchanged. Root reviewed both and independently ran **67 tests** (40 Sol
controls and 27 root controls), including the actual frozen extraction stack
wrapped by the new ledger. Supported, first/second-leaf correction and dual
correction paths correctly cost 10/11/12 calls. An actual contract-repair path
also has the correct stage label. Failed and unknown-usage invocations cannot
be reported as cleanly reconciled canaries.

Root reproduced and required fixes for per-leaf ordering, impossible split
attempt counts, and duplicate-emission failure telemetry. Strict final gold,
types, provenance, qualifiers, markers, duplicate rejection and the underlying
call ceilings remain intact. A separate synthetic false-support judgment does
not trick the final canary oracle into passing.

- Canary module SHA256: `3d132415573cb46c2b15a69a56776ea69b3410b350a8fdb8ba6a1f8cc85658af`
- Stage module SHA256: `4c654599a979e51aeb9f0985b091cd8ef38a639424f197c412da45e1f3270d30`

Cumulative new offline controls: **204 passing**, plus the 26 scoped existing
regression tests. No model calls or remote mutation so far. Next is a separate
Sol implementation of the bounded diagnostic described in
`2026-09-29-semantic-judge-diagnostic.md`, followed by root review. The proposed
29-call live check is not launched or a claim of LME readiness.

Before freezing, Sol additionally bound every grounding source/context field
to its exact ordinary request and checked imported module paths and fixed
fixture bytes. Root reviewed these final guards and reran all 67 tests against
the final hash above. These new modules are now frozen for downstream review.

Root subsequently accepted the separate diagnostic core after 46 additional
offline controls and a physical-candidate preflight: cumulative new checks 250.
Exact evidence hashes, the 29-new-judgment schedule and core hash are recorded
in `2026-09-29-semantic-judge-diagnostic.md`. Only retained evidence hashes/counts
were read remotely; no model calls or remote writes. A fresh Sol is implementing
the isolated host/entrypoint/reader stage; it must pass root review before any
launch. The old monitors remain paused.

Later status: the original live semantic diagnostic failed despite passing its
mechanical gates. Root privately replayed all 27 new judgments and verified the
failure and cleanup. The distinct findings and immutable receipts are in the
semantic-judge-diagnostic plan. The now source-bound decision-policy-v2 repair,
sequential Sol implementations and root verification are recorded in
`2026-09-29-grounding-decision-policy-v2.md`. No original experiment was resumed
and no full LME or production change follows from those diagnostic receipts.
