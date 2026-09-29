# Record-review evaluation — offline preparation

## Scope

Continue from the accepted record-review-v3 offline repair. Implement a pure
facet-aware scorer and independently review explicit labels before any new
provider response. Keep the diagnostic prompts, production code, old scorers,
fixtures, labels and paid outputs unchanged. No paid calls, network transfers,
credentials, deployment, restarts or full LME are authorized by this step.

## Sequential plan

1. A new agent implements an additive scorer and its tests. Root reviews the
   diff and independently attacks target selection, ambiguity and accounting
   before acceptance. In parallel root drafts new fixture candidates.
2. After that gate, a fresh agent reviews every fixture source/candidate/label,
   its exact changed paths and faithful whole-scope acceptance suitability.
   Root verifies the resulting corrections before freezing the design.
3. Run the combined offline regressions; freeze exact requests, separate gold,
   deterministic schedule, source hashes and gates. Independently rebuild/audit
   the package. This is preparation only, not a live-run implementation.

## Proposed design

Thirty selected scopes: six replay controls for the three observed failures and
24 fresh synthetic controls (12 pairs). Replay payloads are reused unchanged,
but receive new, manually justified facet labels rather than converted aggregate
relations labels. Report replay and new-control cohorts separately. Existing
18 development fixtures are not a holdout and are not counted as fresh controls.

The new pairs cover actor ownership, quoted speech, same-message context,
canonical versus opaque identity, names belonging to other people, closed-set
and global quantification, and source-owned retention/ordering regressions.
Candidate prompts are already fixed. These controls are invented mechanism
tests, not a representative or statistically independent LME sample.

Every label resolves exact semantic coordinates. Retention labels select one
source-owned facet; grounding labels may use an explicitly reviewed conjunction.
Legacy `relations` labels are rejected. All uncertainty/malformed responses are
misses, not successful defect detection. Off-target vetoes cannot rescue a
primary false accept. Views, domains and cohorts have separate denominators.
Whole-scope faithful acceptance is a separate gate, never inferred from a target
label or pooled into target accuracy.

Two repetitions would require 60 completions and at most 180 HTTP attempts,
with deterministic paired order and reversed within-pair order on repetition
two. No automatic launch, semantic retries, rerolls or resumed campaigns.
Proposed model: deepseek-v4-flash at api.deepseek.com, temperature 0.0, JSON,
max_tokens=3072; 120-second invocation deadline plus two-second cleanup grace.
Continue independent cases after semantic rejection but halt on safety,
accounting, infrastructure or cleanup failure. These are future execution
constraints, not permission to call the provider.

All labelled targets must match in both repetitions; all independently reviewed
faithful scopes must have no veto. No post-hoc exclusions. Report actual usage,
including unknown/partial usage; proposed cost gates are average total usage
at most 8,000 tokens/call and maximum 12,000 tokens for any call. These higher
engineering ceilings accommodate the documented larger input, not a claim of
savings. No simultaneous baseline is included, so no relative-quality or
relative-cost claim follows. Passing permits proposing further validation,
not runtime adoption or LME readiness.

## Accepted offline result

The scorer and independently reviewed evaluation design are complete and frozen.
**2,049 unique tests passed**, zero failures/errors/skips across 28 targeted
modules. This includes all 1,719 previous regression tests plus 180 scorer tests,
51 root scoring attacks and 99 fixture checks. It is not a full-repository test
run or a semantic-accuracy result. No provider calls occurred.

Root reviewed the complete scorer after its separate implementation agent,
accepted the 463-test intermediate gate, and only then dispatched a fresh label
review agent. Root independently checked the label review and ran the final
combined gate. The scorer reuses only frozen pure aggregation/count helpers;
it never changes the old source-review selector/parser or converts old gold.

Independent semantic review identified three fresh-fixture corrections before
freezing and before any model response:

- Completed inspection uses `resolved`, not the helper's default
  `informational`, for both variants.
- Reported intentions preserve the reporting relationship explicitly in both
  variants, avoiding a reported-claim versus unqualified-fact ambiguity.
- The summary distractor is meeting-agenda ink colour, not display-border colour
  that might itself be a material design specification.

The independent reviewer and root endorse all 15 faithful selected scopes for
whole-scope acceptance. Replay payloads remain byte-equivalent to the earlier
source-review controls. A test-only `.value` versus `.text` typo in the first
review receipt was corrected; that failing receipt remains preserved alongside
the final passing receipt.

## Frozen targets and interpretation

Per repetition, report these denominators separately:

| Cohort | View / domain | Targets | Expected states |
| --- | --- | ---: | --- |
| Replay | Primary grounding | 6 | 3 supported, 3 unsupported |
| Replay | Auxiliary grounding | 2 | 1 supported, 1 unsupported |
| Fresh | Primary grounding | 18 | 9 supported, 9 unsupported |
| Fresh | Primary retention | 6 | 3 retained, 2 omitted, 1 altered |
| Fresh | Auxiliary grounding | 2 | 2 supported |

There are 34 unique labelled targets among 429 selected-scope checks. The
remaining checks do not silently become labelled semantic successes. Fifteen
faithful whole-scope controls (three replay, twelve fresh) must separately have
no veto on any check. Four replay boundary-speaker retention labels remain
explicitly excluded due to facet-status ambiguity. These exclusions concern
target-label coverage only: they do not exempt the faithful scope from a veto.

Fresh pairs: dialogue ownership; reported intentions; contiguous same-message
speaker; canonical alias versus opaque peer; workspace not a person; separate
people's identities; bounded negative-predicate exclusivity; anaphoric bounded
cardinality; explicitly supported full-domain universality; prohibition
retention; mandatory sequencing; and material final decision retention.

The complete original 58 scopes are validated, but only 30 selected scopes enter
the proposed schedule. Two repetitions are 60 completions / at most 180 HTTP
attempts. Repetitions and paired synthetic cases are dependent, not independent
representative LME samples. No old paid output is rescored.

One repetition contains 522,263 input characters (104,627 replay and 417,636
fresh). The largest selected request has 18,287 input characters and 23 checks.
These are character counts, not token/cost estimates. Actual model token usage
and elapsed time remain unknown. The raised engineering token gates above are
explicit proposed acceptance ceilings, not measured savings or a hard billing
limit. Unknown usage must not be treated as zero.

## Artifacts, verification and remaining boundary

Package: `/private/tmp/hymem-record-eval.sa5mWb`. It contains exact request
artifacts, separate `gold.json`, deterministic `schedule.json`, source hashes,
test receipts and separate read-only preparation/audit scripts. The independent
audit rebuilds every exact request, binds every label, checks the entire schedule,
verifies source hashes and checks the actual JUnit test identities against the
prior gate. All **497 pre-existing Python/SQL files remain byte-identical**;
five additive files bring the source inventory to 502. Production code and the
record-review-v3 candidate are unchanged.

SHA-256 pins:

- Frozen candidate: `b2f16ec20569ad91a346928ff3227ce55c14b020c6ef0eff09b6b9d552d551b2`.
- Scorer: `055d1710494b64d654965910cc30e0c436c13bd5561d0d491737146cab11cf15`.
- Fixture source: `99b280c26aacb75ab9f899bd6374fc1402453fceecf396b3c7cda9894c5fd16e`.
- `artifact-manifest.json`: `cee7e0f3c22ab397c3f002c6278ed6c4ae51daca83e9224644ca75872e07df77`.
- `final-gate.xml`: `d61172cf81eed339b33e1b6f6ead18c025ab19c601cea8941ffd2df7c1c6e090`.
- `gold.json`: `dd01a38ff521abc2646b1f5a80d79b6e9acff6b9c809c60d308e3fa453b0438a`.
- `schedule.json`: `d175b180fe45c4d620de150644bf716420190ca05651b3cfbdad365de5b4d2a1`.

The manifest records both file-byte and canonical-JSON digests; schedule request
digests explicitly use canonical JSON. Future execution must preserve this
distinction. Saved flags remain `authorized_for_paid_launch=false` and
`runner_and_rehearsal_ready=false`.

Next: adapt and independently audit an offline supervised runner rehearsal for
this exact package before requesting fresh bounded live approval. This package
does not itself provide a runner, spend authorization, live accuracy, production
integration or LME readiness. No deployment, service restart or store mutation
was performed.
