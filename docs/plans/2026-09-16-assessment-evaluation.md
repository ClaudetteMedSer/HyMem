# Compact-assessment evaluation — frozen offline preregistration

## Status and authority

The user's “Continue” authorizes preparing and verifying the next evaluation,
not reusing the spent ledger experiment's paid/data-transfer consent. This
preparation made **zero provider calls, zero external transfers and no production
changes**. The compact assessment, existing runtime, old gold and paid receipts
remain unchanged. No LME clearance follows from these offline results.

Prepared locally under `/private/tmp/hymem-assessment-eval.dHR3MU`:
fixed requests, separate gold, a complete task schedule, source hashes, scoring
tests and root audit receipts. The paid runner/receipt adapter still needs
adaptation and offline rehearsal against these new bindings before execution;
this is a frozen evaluation design, not an already-authorized launch package.
Neither setup nor launch may change its cases, prompts, labels or scoring rules
after seeing a response. A material design change requires a new preregistration.

## Fixed comparison

Use all 33 prior development cases, unchanged, plus 16 newly invented controls
created after the compact prototype froze. These new bytes exercise specified
failure mechanisms; they are **not a representative or independent LME holdout**.
Keep the original 40 legacy target labels and the uncertain retained-35 exclusion.
Never repair or convert previous paid outputs into new-format answers.

For each of 49 cases, compare the unchanged combined baseline verifier with the
compact scoped assessment, twice. Candidate/source bytes and request sampling
parameters are fixed. Only the verifier protocol differs. Case order is a
deterministic hash ordering; paired arm order alternates and reverses on the
second repetition. New controls are interleaved with development examples.

| Quantity | Baseline | Compact | Total |
| --- | ---: | ---: | ---: |
| Completions per repetition | 49 | 114 | 163 |
| Completions, two repetitions | 98 | 228 | **326** |
| Case-arm invocations | 98 | 98 | **196** |

Proposed execution: **Afrodite → `https://api.deepseek.com`,
`deepseek-v4-flash`, at most 326 paid completions / 978 HTTP attempts**.
Temperature 0.0, JSON response format and 3,072 output-token request cap remain
unchanged. Each case-arm invocation has a 120-second deadline and two-second
cleanup grace. At most three transport attempts per reserved completion;
disable any unaccounted nested retry. No semantic retries, parser repair,
regeneration, rerolls or campaign resumption. These are count/token request
bounds, not a dollar-price quote or predictions of actual token usage.

Continue independent cases after unsupported, uncertain or malformed model
output, recording each rejection. Halt on safety, infrastructure, accounting or
cleanup failure. Preserve partial results and unknown in-flight usage; never
silently re-run missing cases. After a halt, no remaining capacity automatically
authorizes another campaign.

## Gold and denominators

Code in `benchmarks/digest_assessment_evaluation.py` scores immutable original
responses against independently supplied check selectors. Selectors bind to
field paths or canonical chunk IDs, not guessed cN/sN positions. Entire malformed
scopes stay malformed: an apparently correct fragment cannot be salvaged.
Expected labels are only supported/unsupported. Uncertain and malformed answers
are misses/abstentions reported separately, not successful defect detection.

Different views deliberately overlap and **must not be pooled**:

| View | Targets per repetition | Supported / unsupported |
| --- | ---: | ---: |
| Legacy family comparison | 56 | 30 / 26 |
| New mechanism-specific primary checks | 16 | 8 / 8 |
| Separately labelled canonical retention | 25 | 18 / 7 |
| Omission controls' present assertions/relations | 12 | 12 / 0 |
| Existing negative cases' specific defect witnesses | 19 | 0 / 19 |

Legacy comparisons retain the previous 40 expectations and add 16 reviewed new
family labels. Compact legacy aggregation uses assertion/relation/outcome checks
for the designated family, not unrelated retention checks. The historical F3
summary-omission pair additionally uses its four newly reviewed retention checks;
new summary-omission controls also include their specifically reviewed retention
checks. This preserves the old summary-outcome requirement rather than treating
the surviving true request sentence as complete.

The new prohibition-omission procedure is especially important: both variants
are **supported under the old stated-claims criterion**, but only the variant
retaining the prohibition passes the new completeness criterion. Do not silently
turn the old supported label into an error, or claim a baseline regression based
on a requirement it did not have.

A coarse family match does not prove detection of its intended defect. The
compact witness/primary views therefore name the relevant assertion, relation,
outcome or source-retention checks. A veto on another check cannot cure a target
false accept. Baseline has no such fine-grained judgments; report its family
results separately without inventing check-level decisions. Keep supported-case
acceptance, false rejects, false accepts, uncertainty and malformed responses
visible for every scored view/cohort/repetition.

The 16 new cases form eight pairs: causal suffix, prior-only identity,
deferred-versus-resolved classification, answered-outcome omission, procedure
prohibition omission, cross-step ordering, incidental-versus-material omission,
and repeated Unicode source units with different statuses. Pair mutations and
source coordinates are exact and independently checked. Omission controls also
label actually present true assertions: detecting an omission cannot excuse
incorrectly rejecting a true surviving statement.

All 20 canonical units in the new target scopes have a documented retention
decision. Three defective variants—causal suffix, added identity and contradictory
outcome category—have **unscored** retention labels (`expected=None`): intact
original sentences alongside an incompatible added claim make the preservation
criterion ambiguous. Root excluded these before any new model output rather
than forcing a convenient answer. Their primary defects remain fully scored.
The other 17 new retention labels plus eight F3 labels form the 25-label
retention denominator. Unlabelled prior retention checks remain descriptive only.
Gold and rationales never enter model requests.

## Preregistered interpretation gates

1. **Integrity:** require complete frozen request/response/accounting receipts,
   exact payload and source bindings, observed finish reasons, per-call token
   counts and independently confirmed owned-process cleanup. Audit successful
   and failed attempts, not just nominal completion counts. Missing/untrusted
   receipts invalidate an accuracy claim; no cleanup/accounting warnings may be
   hidden by a semantic score.
2. **Transport:** report malformed scopes out of all 228 compact responses and
   98 baseline responses, by case/cohort and repetition. Zero compact malformed
   responses is the green transport gate; anything else keeps it uncleared.
   This is an observed finite-sample gate, not a guarantee of future reliability.
3. **Controlled mechanisms:** a green mechanism gate requires every scored new
   primary check, new non-excluded retention check, and omission auxiliary check
   to match in both repetitions. Show failures by pair/mechanism. Do not count
   malformed/uncertain as correct negatives or use off-target rejection credit.
4. **Development comparison:** report all original target expectations and the
   paired family comparison. A promising result requires fewer legacy target
   false accepts than the simultaneous baseline without reducing supported-target
   acceptance; also report specific witness catches, not just coarse family
   vetoes. Break out old and new cohorts. No post-hoc exclusions or threshold
   adjustment may turn a negative comparison into success.
5. **Cost:** report actual completions, HTTP attempts, prompt/completion tokens
   and wall time by arm. Compact has 228/98 ≈ 2.33× the scheduled completion
   count, not a proven token/cost advantage. Do not infer savings from short
   synthetic replies. Report per-correct-target cost without pooling views.
6. **Decision:** even all-green results justify only a subsequent untouched-case
   and end-to-end validation proposal. They do not authorize runtime adoption,
   deployment, process restarts or full LME. A failed gate calls for a consolidated
   failure analysis of the entire completed matrix, not another immediate reroll.

Cases, paired variants, repeated draws and reused sources are dependent. Do not
report them as independent population samples or compare today's scores against
historical model draws as if provider conditions were controlled. Two repetitions
can reveal instability but cannot establish its long-run rate.

## Independent review and offline verification

A separate implementation agent supplied fixtures and integrity tests; root read
the full source/candidate/gold pairs, requested the three ambiguity exclusions,
and froze the final labels. A separate scoring review found a genuine bug:
overlapping targets in the same view could count one check twice. The agent fixed
it, and root rechecked the code and adversarial tests. Same-view overlap is now
rejected; overlap across expressly separate views remains valid.

**754 tests passed across nine selected modules**, zero failures/errors/skips.
This includes 64 fixture tests, 69 root scoring tests and 14 independent scoring
attacks, plus the existing compact/isolation/ledger tests. It is not a new full
repository-suite run and must not be added to the previous 2,359 count as if all
tests were distinct. Root also individually exercised all 128 labelled groups
with scripted target/off-target vetoes; these are transport/scoring checks, not
observed model accuracy.

All **471 pre-existing Python/SQL files remain byte-identical**; five additive
evaluation/fixture/test files bring the frozen inventory to 476. The compact
prototype remains `8960442b26a8c42cd7620138573c589468b6659f6471e5ce080765cc3cdfba1a`.
Old fixtures, old gold and old paid receipts are unchanged. Offline guards
recorded zero network/credential-access attempts. Every prepared new request
resolves against its exact frozen candidate, source projection and parameters.

Artifact pins:

- Fixtures: `2798c3c1518e836b9f7d0c83d41147388e3d34e333d9cce748b4dd746ae48b9d`.
- Gold: `a4cd090eb9bd701086d052079bfbae672af379f7685053327bd8439f5d7f8ec3`.
- Proposed plan: `bf4ad5a7d1f41c0c2fc506024e1b754aafaa9be90409b0693f720e4032bb4d65`.
- Source manifest: `8941b94b569419995d3aaa969f1b64204137a3cd8b03f72f65109ab2c634aef7`.
- Raw JUnit: `6781bc79feeacb84e0d077f3a0cea413aefae5b727175445a890d3ef6cd13c95`.
- Root gate: `f31c2c05bd51f1f6f618dc30662e6e72a13ae84cea1ec1f6ae51923e9d107779`.

Next: obtain fresh explicit consent covering benchmark-only data and diagnostic
source transfer to Afrodite, and the specified provider/model/budget. Complete
and audit the fresh one-shot runner and Linux offline rehearsal before spending
that authority. Do not transfer production memories or credential files, deploy,
restart services, or launch full LME as part of this diagnostic.
