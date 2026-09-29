# Source-focused verifier revision — offline development

## Why this change

The completed 49-case comparison did not establish improvement on the older
hard cases. Two compact replies assigned support to boundary context, material
prohibitions and answered outcomes were missed, and a changed second record
contaminated a retention judgment about an unchanged first record. The current
runtime, failed prototype, frozen gold, preregistration and paid receipts remain
unchanged. This is an additive diagnostic revision, not a deployment or LME
readiness claim.

## Sequential implementation and review

1. **Authority separation.** A new implementation agent builds an additive
   `benchmarks/digest_source_review.py` diagnostic. Root independently tests and
   reviews it before step 2. Reuse the strict v9 projection and binding checks;
   do not duplicate or weaken source validation. On the wire, canonical text,
   contextual metadata/boundary text, and summary-only prior continuity occupy
   distinct namespaces. Grounding replies separately name primary and context
   evidence. Attribution metadata alone cannot establish a content assertion;
   contextual references for a positive judgment must accompany their own
   canonical record. A prior summary cannot sponsor boundary context. Context
   remains available to interpret a continuing canonical phrase. This is a
   deliberately more conservative authority contract than the v1 diagnostic,
   not a proof that an arbitrary canonical citation entails a claim.
2. **Source-focused retention.** After the first gate, a fresh agent replaces
   the single whole-source retention verdict in the new diagnostic with fixed,
   source-owned obligations for material facts/results, constraints/prohibitions,
   and required ordering (including material chronology). Root independently
   verifies the change. Retention is
   scheduled before grounding and returns candidate-field witnesses, not source
   IDs that merely repeat the input. Outcomes distinguish retained, omitted,
   altered, not-applicable and uncertain. Retained/altered require candidate
   fields; omitted/not-applicable cannot claim witnesses. All source/facet
   obligations remain present, including when the candidate is empty. Unrelated
   candidate errors do not logically change another source's retention result.
   The full scoped window remains visible for cross-message interpretation;
   this is not semantic atomization or automatic source-completeness proof.
3. **Boundary attribution preservation.** Root's full-payload audit found an
   inherited projection loss: the old ledger's boundary units preserve text and
   coordinates but drop that context message's role/peer/workspace. Eight of the
   frozen cases include context from a different message with a different role.
   After the retention gate, a third agent adds explicitly associated contextual
   attribution to the new diagnostic only. It must never be mistaken for the
   current canonical speaker or promoted to primary evidence. Root independently
   verifies same-message and cross-message metadata, nullable values and bindings.
4. **Independent gate.** Root tests malformed/tampered inputs, authority mixing,
   exact fields/source coordinates, Unicode/repetition, contextual continuations,
   cross-source negative contamination, omission versus false assertion,
   mandatory ordering and prohibit/allow pairs. Replay all 49 original payloads
   through preparation only. Do not transform/rescore previous paid replies or
   modify their gold. Run the affected existing offline regressions and verify
   prior source files are byte-identical.

## Shared safety and cost contract

- One caller-owned invocation per original item/summary scope: no additional
  calls, model construction, credentials, network operations, retries, repairs,
  store access or runtime imports of the new diagnostic.
- Preserve request sampling/token parameters. Bound full input, complete output,
  references, checks and calls before execution; reject rather than truncate.
- Invalid model output rejects the whole scope. Client errors halt with sanitized
  metadata; deadlines and process interrupts propagate. Preflight all requests
  before the first invocation. Bind standalone parsing to exact original inputs.
- Diagnostic results never authorize publication or claim semantic verification.
  Keep grounding and retention results separately inspectable; not-applicable is
  a model judgment, not a mechanically proved absence of material information.
- Measure serialization/call costs offline without claiming token savings. More
  check facets can increase output cost even without additional calls.

## Acceptance boundary

These changes can repair authority assignment and make retention failures
explicitly attributable. They cannot establish that DeepSeek will obey the new
contract or judge meaning correctly. Exclusivity, identity, causality and
materiality remain semantic evaluation targets. No new paid calls, transfers,
deployment, restarts or full benchmark are authorized by this offline work.
Any further live experiment needs fresh frozen labels (including facet-level
retention labels), untouched controls, cost gates and explicit approval. The
existing aggregate retention labels cannot be silently reused as facet labels.

## Authority gate accepted

The first implementation agent completed authority separation. Root reviewed the
implementation and independently exercised contextual continuations, attribution
and boundary promotion, cross-record sponsorship, prior-summary sponsorship,
whole-scope rejection and the explicit limit that an irrelevant canonical ID
still does not establish entailment. **152 tests passed** (117 implementation,
35 independent root controls); the existing **754-test** verifier/scorer gate
also passed. These are structural tests, not paid semantic results.

Root's preparation-only audit covered all **49 frozen cases / 114 scopes**,
preserving 368 candidate fields, 146 canonical units, 182 contextual units and
all sampling parameters. The 476 existing Python/SQL files match the frozen
manifest. No old model answer was converted or rescored. Private receipts are
under `/private/tmp/hymem-source-review.OTzCQw`. The next step is a fresh agent's
retention implementation, followed by a new root review.

## Retention and attribution gates accepted

A second agent implemented the source-owned facets and candidate-field
witnesses. Root reviewed the code and independently tested prohibition omissions,
answered-outcome omissions, cross-field ordering, unchanged-versus-altered
records, empty candidates and the expanded caps. **239 tests passed** before
starting the attribution fix. An initial implementation-test fixture incorrectly
represented procedure steps as strings; the existing validator correctly rejected
it. The fixture was corrected to the required order/action/tool records, with
no parser relaxation. The failed and corrected gate receipts are both retained.

A third agent then repaired the inherited boundary-attribution loss. The new
wire mapping preserves the context message's exact role, peer and workspace,
including null versus empty values, linked to its existing boundary-source ID.
The canonical chunk remains the owner, not a substitute speaker. No additional
evidence IDs, checks or primary authority are created. Root independently
verified cross-message attribution and rehashed tampering before acceptance.

The final implementation is `benchmarks/digest_source_review.py`, protocol
`digest-source-review-v2`. Public entry points are `prepare_source_review`,
`parse_source_review` and `execute_source_review`. No runtime module imports it.
All outputs remain diagnostic: `semantic_verified` and
`publication_authorized` are always false. The combined flag is deliberately
named `model_no_defect`, with separate grounding and retention views, because
model-selected `not_applicable` is not proven absence of an obligation.

## Final independent verification

**1,042 tests passed, zero failures/errors/skips, across 15 selected modules.**
This includes 754 existing regression tests and 288 new tests (217 implementation
controls and 71 independent root controls). Counts include, rather than add to,
the earlier gates. Root reconciled the JUnit count against 1,042 unique testcase
records. This is a targeted offline suite, not the complete repository suite.

The final preparation-only audit again covers all 49 frozen cases / 114 scopes:

- 368 exact candidate fields, 146 canonical units and 182 contextual units;
- 1,115 scheduled checks, including 438 source/facet retention obligations;
- 36 contextual attribution mappings, including 16 cross-message mappings
  across scoped requests (the same source can appear in multiple scopes);
- unchanged sampling parameters and one invocation per original scope;
- all 476 prior Python/SQL files byte-identical to the frozen manifest.

There were no paid calls, external transfers, deployment, restarts or production
store access. No earlier response was repaired, converted or rescored; no frozen
gold was changed. `git diff --check` also passed.

Cost caveat: complete serialized input grows from **935,450 to 1,279,129
characters** across these 114 scopes, approximately **36.7% larger** than the
previous compact diagnostic. The largest input is 17,816 characters, with at
most 21 checks in one request. These are character measurements, not token,
latency or dollar estimates. Keeping call count unchanged does not establish a
cost improvement. Any live comparison must gate semantic benefit against cost.

Private receipts: `/private/tmp/hymem-source-review.OTzCQw`.

- Final implementation SHA-256:
  `bb28eebc2400300c8ef05ca1fdd3d7462774dc0b37555c4c6ac7f3c0d8100b6f`.
- Final JUnit SHA-256:
  `7c7bd97d55b291e96b000b42d9d734c6c58dcbc2dd0cc86ec9065dfb9856c2e7`.
- Final preparation audit SHA-256:
  `82d0f17aecb32d8afa737230841559afe0bfbc08446814e8c22b28a33021f9fb`.

## Remaining decision

**Offline contract accepted; model quality, runtime adoption and LME readiness
remain unproven.** The concrete lost-attribution bug and mechanical authority
assignment are addressed in this new diagnostic. Retention now has localized,
inspectable obligations, but a model may still choose the wrong status, cite an
irrelevant canonical unit, miss exclusivity/identity/causal defects, or wrongly
declare an obligation not applicable. Scripted controls explicitly preserve this
distinction instead of disguising it as semantic verification.

Before any fresh paid run, prepare new facet-level labels and untouched paired
controls, preserve the previous comparison as development evidence, and freeze
a quality-versus-cost protocol. The old aggregate retention gold must not be
silently reinterpreted. A later production decision also requires an approved
end-to-end smoke gate; these offline results do not clear the full LME benchmark.
