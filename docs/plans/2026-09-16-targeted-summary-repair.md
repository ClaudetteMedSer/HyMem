# Targeted summary repair and independent format verification

## Final status

**Complete locally:** all three fixes were implemented by separate agents and
independently reviewed/verified by root. The final selected regression suite
passed **2,248 distinct tests**, with zero failures/errors/skips. This is not a
full-repository suite or proof of model accuracy. No paid calls, remote changes,
deployment, service restarts or production-store access occurred. LME is not yet
cleared: a fresh bounded live diagnostic and end-to-end smoke remain necessary.

## Scope and acceptance

Implement the user's requested fixes locally. No production access/deployment,
restart, paid provider calls or new live-run authority. Preserve existing dirty
worktree changes. Benchmark incident text is not a training fixture or hardcoded
acceptance rule; regression cases use synthetic evidence.

Implementation is sequential: a separate agent implements each bounded fix;
root reviews and independently checks it before assigning the next fix.

1. Add bounded, source-linked summary rejection diagnostics to the fidelity
   contract. Validate strict structure, allowed issue codes, exact source and
   candidate quotations, source references and bounds. Findings are untrusted
   repair hints, never factual authority. Reject invalid findings; a rejection
   without actionable validated findings must stay held, not trigger a generic
   reroll. Preserve original sources/prior, immutable items/citations, one repair,
   complete reverification and the existing absolute deadline.
2. Separate semantic verification from grammar. Source-aware verification owns
   factual acceptance; a candidate-only final request owns format acceptance.
   Factual rejection cannot reach or be overridden by grammar acceptance.
   Normal complete success becomes three calls; maximum remains six (generation,
   optional compaction, semantic verification, optional targeted repair,
   semantic reverification, final format verification).
3. Bring benchmark instrumentation, synthetic simulation, documentation and
   cost reporting into agreement. Preserve historical recorder data and never
   attribute an earlier response to a later undispatched failure.
4. Freeze and run a combined credential-free regression gate with networking
   blocked. Record exact inputs and results; distinguish offline control-flow
   proof from unmeasured live generation/verifier reliability.

Each step must cover malformed and adversarial output, omissions and temporal
relations, citation/continuity boundaries, unchanged repairs, semantic/grammar
disagreement, caps, deadlines, no partial publication, and producer identity.
No claim of LME readiness follows from offline tests alone. The v14 diagnostic
remains closed; a fresh explicitly budgeted live test and end-to-end smoke are
still required after these local fixes.

## Step 1 progress

Root's two-test red gate reproduced the pre-change generic repair after an
unexplained rejection, with unchanged inputs and no networking. The diagnostic
contract now accepts only fixed issue codes, exact candidate quotations and
bounded quotations from identified visible new spans or explicitly typed prior
continuity. Unknown, uncited, context-only, fabricated or duplicate references
fail closed. An absent diagnosis holds the original rejection without a reroll.

Root's preliminary expanded gate passed 24 independent tests, including actual
repair-path source/citation immutability, changed candidate reverification,
pre-dispatch input caps and second-pass semantic vetoes. This is not final core
acceptance or proof of live-model accuracy; the agent's broader core regression
and root's frozen-input acceptance gate are still pending.

Private receipts for this work are under
`/private/tmp/hymem-targeted-repair.P7UQ8g/`. Historical gates and live campaign
receipts are not edited or relabeled.

## Step 1 accepted; step 2 begins

Root reviewed the implementation and changes to existing rejection fixtures,
then accepted a frozen **1,235-test** gate covering all digest modules, semantic
generation and lossless publication. Zero failures/errors/skips; input hashes
unchanged; network attempts zero. The prior 627-test agent sweep is overlapping
and was provisional because a small edit occurred during it; it is not added to
the root count.

Accepted step-1 digest SHA-256:
`8dd530f9623d2fd37714169c6dc47b8563e9cb82f81aca8a03f7c005e9398a16`.
Root JUnit SHA-256:
`6da94a599908ec7808107ee8a269847e7cf75baf30bd4ed27cba74d566ea56a8`.
Root input-manifest SHA-256:
`5c967348d8c66a03d4fee4533c9f1ecf3ecd668d463a14f1f1ffc7b7194c4609`.
Receipts use `root-core-step1.*` in the private directory above.

Three independent pre-change grammar-isolation checks failed as expected:
the current combined-verifier path does not always request a separate grammar
decision. A new agent now owns separation into four semantic verdict families
and mandatory candidate-only final formatting. This leaves maximum calls at six
but raises the normal full success path to three. Step 1's accepted findings and
factual veto must remain unchanged.

## Step 2 review

The semantic response is now exactly four families (`episode_titles`,
`episode_content`, `procedures`, `summary_content`). Only after complete semantic
support does one mandatory candidate-only request check summary and episode
format. The wire contracts are fidelity v8 and candidate-format v2; legacy
internal format-adjudication names remain for compatibility. The source-linked
repair contract remains v2.

Root's frozen `root-format-step2` gate passed **187 tests**, zero
failures/errors/skips, unchanged inputs and zero network attempts. JUnit SHA-256:
`7070c136cdc75873118c58f95da9997bfab02d094a294a0ca96dbc81a373a2ad`;
input-manifest SHA-256:
`7f7986aa9b71735855d007386c2f3cc0b590c9971f18618b8ac7317322a6a744`.
This covers independent controls plus the format/repair interaction, immutable
request fields, malformed verdicts, deadline expiry and factual vetoes.

Review also found and corrected a stale public-summary rejection fixture that
could pass on an unrelated mock `KeyError`. It now rejects at the actual format
stage and asserts both dispatch and the explicit `summary_format_unsupported`
reason. Root accepted a further frozen **43-test** gate containing that corrected
test and the independent format/targeted-repair controls. JUnit SHA-256:
`0a70e8674255b8bdcc5268f58eb2da4b3acee613058456227bf96eaa353ffb36`;
input-manifest SHA-256:
`2be2820056bd0ae60a49f1ff8b4b30d45ccefad4941bf56ea01f162cf4fd858e`.
The counts overlap; they must not be added as unique tests.

Step 2 is accepted. Frozen runtime SHA-256:
`a210e044de389a2bb9139338bbec7b4e4bf73230856e55b224465e03bdee222c`.
The broader agent sweep was deliberately interrupted after 374 passes to avoid
duplicating the final combined run; its exit-2 receipt is **not acceptance**.
The agent's separate focused summary gate passed 9/9. A new agent now owns only
the recorder/simulator/cost alignment. Root's ten pre-change recorder controls
showed the expected stale version and simulator failures (5 failed, 5 passed),
with unchanged inputs and no network attempts.

## Step 3 accepted; combined gates running

Recorder v4 and the simulator now match the four-family semantic task and
mandatory candidate-only format task. Cost/help report normal 3/max 6 calls.
Source-linked repair requests retain the exact original generation input inside
their structured envelope. All three pre-dispatch input caps explicitly avoid
assigning an earlier response to a request that was never sent. Historical
v1/v2/v3 records keep their request/reply strings, hashes, metadata, versions and
original failure attribution when rescored; completion counts never dilute the
session-level failure denominator.

The agent's final frozen probe suite passed **177 tests**, zero
failures/errors/skips, unchanged inputs and no network attempts. Root reviewed
the recorder diff and accepted a further frozen **18-test** gate covering its
independent stage/cap/hash controls, corrected capture fixture, historical CLI
rescoring and cost help. These counts overlap and are not unique-test totals.
The initial probe gate found one stale capture fixture that mistook the new
grammar request for generation; it was corrected, strengthened with a success
assertion and rerun. Failed/mutable receipts remain failed, not relabeled.

Accepted recorder SHA-256:
`1cae27d13ed223d53f7452ef914180965d4d160a2e51f34913b4efabb096df8e`.
Root `root-probe-step3-final` JUnit SHA-256:
`c28c94bb82392b839c1d9a3e16a5e291087da65c3f696a3bf6d0873f7126f008`;
input-manifest SHA-256:
`d61ed01ae52a439377a9e7acec83fb67b000d34c959a6a53646069063918c2e2`.

Root is now running two final frozen, credential-cleared, network-denied gates:
the combined digest/probe/deadline/indexing suite and disjoint affected
integration controls for profile, procedures, concurrency, terminal loss,
aggregation provenance, store attestation, benchmark adapters, sidecar
publication, scheduler, MCP and portability. These are selected regression
gates, not a full-repository suite or live provider evaluation.

## Combined-gate finding and correction

The first integration gate passed 376 tests. The first combined gate completed
with 1,870 passes and two failures, both modes of the retained-source generation
replay test. Its explicit stub supplied a semantic verdict but no response for
the new mandatory format task; the runtime correctly rejected the default `[]`
as `format_adjudication_shape_failure`. This was a test-client migration gap,
not permission to bypass formatting.

The format agent added the missing explicit synthetic response and strengthened
the test to require initial publication plus exactly one format request before
pruning. Root reviewed the narrow test-only diff and independently passed both
modes with frozen inputs, no networking, and no runtime changes. The failed
combined receipt remains rejected. Root is rerunning the **same 2,248 distinct
tests** in three disjoint frozen groups, balanced by measured module duration;
the final audit must prove the exact original test set was retained.

## Final combined acceptance

All three final groups passed against identical frozen inputs:

| Receipt prefix | Passed | Failures / errors / skips |
| --- | ---: | --- |
| `root-final-core-a` | 1,009 | 0 / 0 / 0 |
| `root-final-core-b` | 863 | 0 / 0 / 0 |
| `root-final-integrations-v2` | 376 | 0 / 0 / 0 |

The independent final audit verified disjoint test IDs and exact equality with
the entire original 2,248-test selection, all **453 Python/SQL file hashes**
unchanged, matching receipt hashes, zero network attempts, and zero provider
calls. One third-party Starlette test-client deprecation warning remains; it is
not a runtime failure. `git diff --check` is clean.

JUnit SHA-256 values, in table order:

- `aa46a4cc6fce866f2bfd87e3f92088b34bdada94d703d690586c69bca5819509`
- `238abd5be5cd8703fdbf80f92c2cb60d3a5e0a2ab4cbd9f2ebb25c112bf5b704`
- `7ed7989c2715e71334a92d483efe8df9c13f3bd022789bf4fcbac95efbdd26f4`

Input-manifest SHA-256 values, in the same order:

- `25b49c5192e4aa2eb1e4880fcc0e58cd9615be1a16f39942ab02c2eb697b984a`
- `9cf0b4d2db2c303fe967aa11d90170347d60c6b6f645dbca5aaba09956629bf7`
- `32ae0e6699369f5e6f28a5905220c4e3cee14b4b489370cce75cedb978ff6f41`

Relative to the frozen v14 candidate, only two runtime Python/SQL files changed:
`hymem/dreaming/digest.py` (SHA `a210e044de389a2bb9139338bbec7b4e4bf73230856e55b224465e03bdee222c`)
and `benchmarks/episode_probe.py` (SHA `1cae27d13ed223d53f7452ef914180965d4d160a2e51f34913b4efabb096df8e`).
Other current-turn changes are regression fixtures/controls and documentation;
earlier dirty-worktree changes were preserved. The digest identity intentionally
changes and may replay retained digest work upon later deployment, without
changing Phase-1, fact or profile contracts.

The closed v14 live authority remains closed and cannot authorize another run.
Its private supervisor bindings/control requests also require adaptation and
fresh review for the new wire contracts before any newly approved live test.
Scripted approvals establish application control flow only; actual omission
recovery, verifier false-positive/false-negative rates and full LME convergence
remain unmeasured for this source.
