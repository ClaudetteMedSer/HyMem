# Source-owned retention inventory — offline candidate verified

## Status and scope

Subsequent [live development pilot](2026-09-18-retention-inventory-live-pilot.md):
collection passed, **semantic evaluation failed**. The offline verification below
establishes contract behavior only; the candidate is not ready for promotion.
Latest continuation: [summary coverage-owner correction](2026-09-18-summary-retention-scope.md).
Per-item supporting citations do not define exhaustive retention ownership;
the historical per-item API below must not become a general completeness gate.

Implemented and verified offline after the user's `Continue`. The completed frozen
confirmation remains failed at 29/32; its cases, labels, requests and responses
are immutable regression evidence. No runtime gate is relaxed, replaced or
deployed. This step adds an offline diagnostic and adversarial tests, not a
claim that model omission detection is fixed. No provider call, credential
access, server transfer, restart or full LME run is included.

## Diagnosis and design decision

The old source/facet-wide verdict accepted a missing prerequisite twice and a
missing prohibition once despite explicit omission instructions. A candidate
field ID is a locator, not evidence of completeness. The earlier quote-ledger
experiment also suffered substantial malformed replies and higher token use;
do not reintroduce verbatim model copying or character-offset arithmetic as the
new primary mechanism.

Add `benchmarks/digest_retention_inventory.py`, keeping all existing runtime and
diagnostic modules unchanged. A separate agent implements it and focused tests;
root reviews the code, writes independent attacks and verifies regressions.

1. Code partitions every nonempty canonical source losslessly into deterministic
   offset-bound location units. These are not atomic facts or semantic labels.
   Preserve the full scoped sources, exact metadata, boundary ownership and
   interpretation-only authority. Prior summaries are not new canonical facts.
2. A candidate-blind inventory request sees canonical source records and units,
   not candidate fields/text/hash. The model returns obligations with a facet,
   referenced unit IDs and short description, plus explicit no-material and
   uncertain unit lists. Code assigns obligation IDs and requires every unit to
   be accounted for; overlapping obligation spans are permitted. No-material
   units cannot overlap obligations or uncertainty. A partly understood unit
   may contain both known obligations and an explicit uncertainty flag; that
   flag blocks an affirmative retention result even if its known obligations
   match. No-material classification
   and semantic completeness remain model judgments, not proofs.
3. Freeze and bind that inventory before exposing the candidate. A separate
   matching request returns exactly one retained/omitted/altered/uncertain
   verdict per obligation. Retained/altered need candidate field locators;
   omitted forbids them. There is no candidate-conditioned not-applicable escape.
   Whole reply rejection applies to unknown/missing/duplicate references.
4. Empty/unresolved inventories cannot yield affirmative retention by vacuous
   truth. A structurally valid inventory or witness never grants semantic or
   publication authority. Keep uncertainty and structural failure explicit.

Candidate-blindness is conditional on the predeclared source scope: episode and
procedure scopes still use the original cited source set. It does not establish
coverage of uncited messages or global retention. Source-only requests must be
byte-identical for candidate-only changes that preserve this scope and sampling.

## Bounds, execution and verification

Reserve at most two retention invocations per original scope before starting.
These are additional to any unchanged grounding review, not a claim of two
calls for an entire verification pipeline. No retries, repair, hidden scope
skipping or overflow truncation. Invalid inventories have explicit dependent
matching skips; independent later scopes may continue. Client failures halt;
deadline/process interrupts propagate. The caller owns HTTP accounting and
absolute invocation deadlines. Validate all immutable inputs and plan hashes
before calls, and bind second-stage input to the actual accepted inventory.

Root tests lossless units/offsets, candidate-blindness, per-obligation omission,
wrong source/field authority, Unicode/repetition, prior/boundary exclusion,
malformed and oversized replies, forged/rehashed plans and inventories,
empty/unresolved states, reservation and actual call counts, exceptions/deadlines.
Include a valid-but-irrelevant witness example explicitly demonstrating the
remaining semantic limitation. Scripted mock success is not model accuracy.

Preserve the 16 frozen confirmation artifacts. Audit their preparation with no
provider and keep the three observed misses as named regression mechanisms;
do not rescore old replies under a different contract. Any new live candidate
must declare a fresh bounded run and report both semantic and format/cost
results. These now-observed cases are development/regression data, not an
untouched confirmation set.

Private receipts: `/private/tmp/hymem-retention-inventory.rhEcxN`.

## Completed verification — September 18

A separate implementation agent added the diagnostic and 65 focused tests.
Root reviewed the implementation and added 64 independent contract and lifecycle
tests. A second agent supplied 12 development controls and 54 fixture tests; a
read-only reviewer independently checked the final design and reran the 65
implementation tests. Root verified their output before accepting the candidate.

The final selected regression gate passed **2,361 tests**, with **four intentional
pre-existing skips**, zero failures and zero errors across 33 modules. It includes
all **183 new tests** (65 + 64 + 54); these are not additional to 2,361. This was a
related-module regression gate, not a new full-suite run. All **516 pre-existing
Python/SQL files** match their start-of-step hashes. Runtime and earlier diagnostic
sources were not changed.

The read-only reviewer independently reconciled the final XML, collected IDs,
phase reports and both source manifests, and reconstructed all 16 preparation
cases. No discrepancy was found. Root separately rechecked the 183-test breakdown
and source pins after the documentation update.

Independent root tests initially caught tuple-to-JSON serialization errors in
source attribution and obligation unit references. Both were fixed with explicit
JSON-list conversion, without weakening validation. Review also caught an
unrepresentable partial-uncertainty case: one location unit can contain a known
obligation and an unresolved detail. The final contract preserves both and blocks
affirmative retention whenever any uncertainty remains. Tests cover this in all
three scope kinds and through complete execution. A mock with an irrelevant but
valid field locator deliberately remains structurally valid, illustrating why
these tests do not prove model judgment accuracy.

The new development controls cover two constraints in one unit, a prerequisite
crossing a codepoint-unit boundary, a changed nonreceipt condition, an omitted
prohibition, boundary speaker ownership and a faithful omission of incidental
detail. They carry **18 targeted labels: 13 retained, three omitted, two altered**.
Gold labels stay outside model requests. These are targeted development controls,
not an unseen confirmation set or whole-scope semantic proof.

Root also prepared all **16 frozen confirmation cases** without provider access:
canonical sources, candidate fields, original offsets and sampling match the
frozen artifacts; all eight faithful/defective pairs have byte-identical source
inventory requests. No old response was rescored. The complete plans contain 30
original scopes, reserving **60 retention calls per single execution**; unchanged
grounding would be additional. Shorter first-stage input alone does not establish
lower total cost because matching and grounding calls remain additional.

### Test-harness accounting

Tests ran with an empty environment and a Python audit hook denying every socket
operation. There were **zero provider calls or network connections**. One
`urllib3.util.connection._has_ipv6` import-time capability probe attempted socket
construction and was denied before binding or connecting.

The initial final-gate wrapper exited nonzero even though pytest passed: it had
incorrectly required zero skips and zero attempted socket constructions. Its
original receipts remain intact. The second gate still denied every socket,
explicitly checked that sole capability-probe origin, and required the exact four
existing `raw` transport skips in `test_completion_response_admission.py`. Their
SDK counterparts ran. No test or application behavior was changed to make this
accounting check pass; unexpected skips or other socket events would fail it.

### Final source and receipt pins

All receipt paths below are under the private receipt directory above.

| Artifact | SHA-256 |
| --- | --- |
| `benchmarks/digest_retention_inventory.py` | `ca232c95faad118512720e4a429f89fa04a28c286abde34695514919935318fa` |
| `tests/test_digest_retention_inventory.py` | `0dcf24e6a86de8dd734ff097555fb60d027f67e538020615ca529ec3175bf3ba` |
| `tests/test_digest_retention_inventory_root.py` | `4e50ad47d40cbfaac7c640c004f621fabaf575164e69194555c0f70a963e2164` |
| `tests/digest_retention_inventory_fixtures.py` | `ba48456a194f465ae1a6b65b7b19a19a19174cc9a5b4e20d7a0173f903a41022` |
| `tests/test_digest_retention_inventory_fixtures.py` | `90667c85b06655813cf635a6b1c03e59c664db0c4fa576272f4f2fc6d7297530` |
| `original-inputs.json` | `e23c25edcead7f6353d52059698cfaa746c51b882cc564254bfd10e029b30cdd` |
| `final-inputs.json` | `f1dce9a69cacabcdd12eb1b180dde2e3fbcba58010322793ce98e6f17f377b16` |
| `final-preparation-audit.json` | `3f827ed26a88bb223cf4a7559bce431af71ede45fbf22832a898503dbf04d71d` |
| `v2-tests.xml` | `49d884ad83f29922a971907f58e8e11eb2bf8b24c3f6ef08e7bda260b79ad87e` |
| `v2-gate.json` | `01e46fa05ba1adf8fc06153eb9de1b1f8d56054d3175c02c9cb57b84f7e0f7fe` |

## Remaining decision boundary

At the original offline checkpoint this candidate had **not been queried against
DeepSeek**; the later failed live evaluations are linked above. Candidate-blind source
inventory may itself omit an obligation or misclassify material text; matching
may still falsely accept a missing condition. Structural coverage does not settle
either question. Neither semantic verification nor publication is authorized by
this module, and no runtime gate has been replaced.

Next is a fresh bounded live evaluation with immutable source/candidate controls,
separately measured inventory completeness and per-obligation matching accuracy,
faithful-control false vetoes, whole-scope outcomes, format failures and full
call/token accounting. Freeze labels before collection; include previously
unseen confirmation cases after development succeeds. The earlier 32-call
authority is spent and cannot be reused. Runtime integration, original-Q1 replay
and end-to-end LME verification remain subsequent gates, not completed work.
