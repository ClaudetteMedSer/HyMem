# Source-review evaluation — offline preparation

## Scope and staged decision

Prepare a new diagnostic evaluation without paid calls, transfers, deployment,
restarts, credentials or production stores. Preserve the source-review prototype,
all existing fixtures/gold and paid receipts. The old aggregate retention labels
are not facet labels and will not be converted. New controls are invented
mechanism checks, not a representative LME holdout.

Stage A will evaluate 24 fresh controls (12 paired mechanisms), twice, using only
each control's explicitly selected target scope. This is 48 proposed completions,
not a whole digest pipeline or a baseline comparison. All other scopes are still
validated during preparation; none may silently provide evidence to the selected
scope. A subsequent retained-case baseline comparison needs its own frozen
labels/protocol and explicit approval; it is not automatically launched if A
passes. Existing experiment capacity is spent and cannot authorize either stage.

## Sequential work

1. A new implementation agent adds a pure source-review scorer and tests. Root
   independently attacks its accounting and target selection before acceptance.
2. Root creates the new paired fixture candidates and explicit facet labels while
   the scorer is implemented. Then a fresh agent independently reviews every
   source/candidate/label and fixes fixture issues; root verifies that review
   before freezing the package.
3. Freeze exact requests, labels kept outside requests, deterministic paired
   schedule, source/fixture hashes and accuracy/cost gates. Run offline regression
   and preparation audits. No old model output is repaired, transformed or scored
   as a new response, and no model is used to invent gold labels.

## Scoring contract

- Grounding labels select assertion/relations/outcome checks by field paths;
  expected verdicts are supported or unsupported. A group is an explicitly
  reviewed conjunction of grounding checks only.
- Retention labels select exactly one canonical chunk and facet. Expected states
  are retained, omitted, altered or not_applicable, never uncertain. Missing or
  ambiguous gold is excluded explicitly before results, not guessed or treated
  as successful abstention. Do not combine facets into one generic label.
- Exact status match is the primary retention metric. Separately report unsafe
  accepts (omitted/altered expected, retained/not_applicable observed), false
  rejects, false not-applicable, false applicable and wrong defect kind. A wrong
  omission-versus-alteration status does not count as an exact match.
- A malformed response invalidates its whole scope; malformed and uncertain are
  never successful defect detection. A rejection of an unrelated check cannot
  rescue the selected target. Validate labels even when the response is bad.
- Reject same-view overlapping checks, conflicting gold across views, duplicate
  selectors and inconsistent grounding conjunctions. Allow explicitly separate
  views to overlap, but never pool their denominators. Report grounding and
  retention denominators separately within each view.
- Scoring is pure and diagnostic: no provider, file/network access, automatic
  semantic classification or publication authorization.

## Proposed interpretation gates

- Integrity: complete immutable request/response/accounting receipts and confirmed
  cleanup; the exact independent auditor must pass against actual offline runner
  rehearsal receipts before any launch.
- Mechanisms: every scored primary, retention and faithful-surviving-claim control
  must match in both repetitions. Report every mismatch by pair and direction,
  with uncertainty and malformed output as separate misses. No majority vote,
  off-target rejection credit, rerolls or post-hoc exclusions.
- Faithful acceptance: independently reviewed faithful target scopes must also
  return `model_no_defect=True` across the complete response. This separate gate
  catches false rejections on unlabelled checks. A global veto on a defective
  case never substitutes for detecting its specifically labelled primary defect.
- Cost: predeclare an engineering ceiling of 5,000 total usage tokens per
  selected-scope call on average and 8,000 for any individual call. These are
  diagnostic budget gates, not price quotes or evidence of production savings.
  Report prompt/completion tokens, calls, attempts and elapsed time separately.
  A later full comparison must measure relative cost against its simultaneous
  baseline; stage A cannot establish that ratio.
- Even an all-green result only permits proposing the next evaluation. It cannot
  establish LME readiness or authorize runtime adoption. Any negative result
  requires reviewing the complete failure map before proposing further work.

## Frozen Stage A design (2026-09-17)

Offline preparation and independent review are complete. This freezes the
evaluation design, not a launch-ready runner: `runner_and_rehearsal_ready=false`
and `authorized_for_paid_launch=false`. No paid calls occurred. Runner adaptation
and an independently audited supervised offline rehearsal remain prerequisites
to requesting fresh live approval.

Each mechanism has a faithful and a defective counterpart. The selected scopes
and primary comparisons are:

| Mechanism | Scope | Primary comparison |
| --- | --- | --- |
| Boundary speaker | Episode | Correct versus swapped speaker attribution |
| Independent boundary entity | Episode | Canonical entity versus independent context entity |
| Prohibition omission | Procedure | Retained versus omitted prohibition |
| Condition omission | Procedure | Retained versus omitted prerequisite |
| Answered outcome | Summary | Retained final answer versus request alone |
| Source-local status | Episode | Correct status versus status transferred from another source |
| Required order | Procedure | Retained versus reversed prerequisite order |
| Material chronology | Summary | Retained versus reversed event order |
| Exclusivity scope | Episode | Bounded versus unsupported global exclusivity |
| Causal inference | Episode | Co-occurrence with caveat versus invented causation |
| Opaque identity | Episode | Role attribution versus inferred display name |
| Incidental versus outcome | Summary | Retained outcome versus incidental detail alone |

Independent fixture review prompted four pre-freeze clarifications, before any
provider response: source-local status now uses static blue label bands and
current sealed/open states; opaque identity uses a stable glaze preference;
the answered-outcome faithful summary explicitly calls the answer final; and
the faithful causal example explicitly retains the no-established-causality
caveat. Pair mechanisms and primary labels are unchanged. Source identifiers
are opaque and pair-stable, with all citations and labels remapped consistently.
The independent reviewer and root endorse all 12 faithful selected scopes for
whole-scope acceptance. Case names, variants, labels, rationales and exclusions
are outside model request bodies.

### Denominators and exclusions

Per repetition, report these separately; never compute a pooled accuracy:

| View / domain | Exact targets | Expected states |
| --- | ---: | --- |
| Primary grounding | 10 | 5 supported, 5 unsupported |
| Primary retention | 14 | 7 retained, 4 omitted, 3 altered |
| Additional retention | 8 | 6 retained, 2 not_applicable |
| Auxiliary grounding | 6 | 6 supported |

The 38 labelled targets select 38 distinct checks from 196 selected-scope
checks. Four exact retention labels are explicitly excluded: material-facts and
constraints for both boundary-speaker variants, where actor/facet attribution
makes the exact status ambiguous. This does not exclude their grounding targets
or the faithful case's whole-scope acceptance gate. The other unlabelled checks
are descriptive, but a veto on any faithful-scope check fails that separate gate.
There are 12 faithful whole-scope controls per repetition; these do not add
primary accuracy credit. Pairs and repetitions are dependent synthetic controls,
not independent representative LME samples.

### Proposed execution and stopping rules

- Candidate only: 24 selected scopes, two repetitions, at most 48 completions
  and 144 HTTP attempts. Every exact request appears once in each repetition.
- Proposed provider/model: `https://api.deepseek.com`, `deepseek-v4-flash`;
  temperature 0.0, JSON response format, `max_tokens=3072`.
- Deterministic hashed pair order uses seed
  `source-review-stage-a-20260917`. Faithful/defective order alternates by pair
  position and reverses in repetition two. The frozen schedule is authoritative.
- One invocation per scope, 120-second invocation deadline and two-second
  cleanup grace; at most three accounted transport attempts per completion.
  No nested unaccounted retries, semantic retries, rerolls, regeneration,
  campaign resume or additional canary calls.
- Continue independent cases after model-output rejection. Halt on safety,
  accounting, cleanup or infrastructure failure; preserve partial and unknown
  usage rather than treating it as zero.
- All exact targets must match in both repetitions, all faithful scopes must
  be globally accepted, and the green format gate requires zero malformed
  responses. Uncertainty remains a separate miss. Apply the token ceilings
  above and report actual usage, not character-based token estimates.
- A passing Stage A only supports proposing a separately frozen and approved
  retained-case simultaneous baseline comparison. It cannot trigger Stage B,
  runtime adoption, deployment or a full LME benchmark automatically.

### Verification and artifact pins

The implementation agent added the pure scorer and 117 tests. Root added 51
independent scoring attacks and accepted that gate before the fresh fixture
review agent worked on the controls and 185 fixture checks. Root then reviewed
the amended fixtures and ran the combined 18-module gate: **1,395 passed,
zero failed, zero errors, zero skipped**, with 1,395 unique JUnit test identities.
These are offline contract/regression checks, not observed model quality.

All 483 pre-existing Python/SQL files remain byte-identical to the start of this
work. Five additive source/test files bring the frozen source inventory to 488.
The source-review prototype itself remains unchanged. No production code,
stores, credentials, deployment or process state was changed by this step.

Package: `/private/tmp/hymem-source-review-eval.85z9PG`. It contains the 24 exact
request artifacts, separate `gold.json`, deterministic `schedule.json`, source
manifest, preparation audit, test receipts and independent `audit_frozen.py`.
`artifact-manifest.json` records both file-byte and canonical-JSON hashes; schedule
request hashes explicitly refer to canonical JSON. This distinction must be
preserved by the eventual runner.

The independent audit verified the written requests against reconstructed
requests, labels and schedule, all 488 source files and the actual JUnit receipt.
Preparation validated all 42 original scopes, scheduling only 24. Root also
individually attacked each of the 38 scored targets with scripted replies;
those checks validate scoring plumbing, not semantic accuracy.

SHA-256 pins (file bytes unless explicitly stated):

- Unchanged `benchmarks/digest_source_review.py`:
  `bb28eebc2400300c8ef05ca1fdd3d7462774dc0b37555c4c6ac7f3c0d8100b6f`.
- `benchmarks/digest_source_review_evaluation.py`:
  `d8b9495228da0498fc0fbce52291ada46cfd0e9816163346cd99659bfc3fc8e8`.
- `tests/digest_source_review_evaluation_fixtures.py`:
  `db26765ace6e331a4da5d16ca8fdcc5993a7a9949a8e948ea125546b70b52a93`.
- `gold.json`:
  `0f6d6f315083bcff70bc0ff56ea9873d25662b7cc7fa9788a5e65a4c5d9a2ac2`.
- `schedule.json`:
  `23df8dd183d1b1e260c5b0301a8466928f53a3efea448f5837885c8e4670eedc`.
- `source-manifest.json`:
  `ff1f2c3bfea0117e7890450bc419290f0f16c4f017e3b9288852c4d3eeed5226`.
- `final-gate.xml`:
  `59c696c5e4e7761de888bbe88090ff9396ff6026aa380d5f1d7aea3b069a6187`.

Remaining prelaunch work: adapt a supervised runner to these exact source-review
requests and scoring rules; independently verify its caps, immutable receipts,
accounting and cleanup; and run the exact independent auditor against actual
offline rehearsal receipts. Preserve this frozen design while doing so. Only
then request explicit approval for the bounded provider transfer and paid run.
