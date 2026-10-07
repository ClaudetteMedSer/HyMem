# Claim-first grounding v3 — sequential repair and verification

## Evidence and scope

The user requests continued diagnosis, a plan, separate Sol implementations,
independent root verification and repeated testing until the failure checks pass.
The last immutable classification-v2 diagnostic remains failed: 20/24 controls,
two false-support and two fixed-label false-rejection outcomes; table canary
passed and prose canary failed. All 28 judgments were independently replayed;
164,765 known tokens, complete usage, cleanup verified. No experiment is active.

The observed defects are full-claim semantic judgments: unsupported numeric
qualifiers and wrong ordered actor/object attribution can receive support;
implicit preference and contextual prose can be missed. Exact JSON, citations
and source identity do not prove entailment. No static transport defect was
established. These are empirical failure modes, not proof that a new prompt
will fix them. Control 5 has a scope/gold ambiguity (laptop-only non-use versus
an unqualified negative-use claim). Preserve its frozen label and failed result;
add complementary offline scope controls, never relabel the old experiment.

## Chosen hypothesis

Two independent read-only Sol reviews recommend explicit full-claim components
and named rather than positional predicate assessments. Use a claim-first v3:
expose the original complete claim and assess it directly. Only when it is
explicitly not established require all other 21 named predicate assessments.
An ambiguous original immediately yields uncertain. This avoids irrelevant
substitutions on supported originals without changing the selector's outcomes:
a supported original already wins; a correction still requires exactly one
supported alternative and explicit non-establishment of every other predicate.

For every positive assessment require an evidence ledger with named checks for
ordered attribution/roles, relation plus polarity, and every non-null qualifier
(value_text, value_numeric, value_unit, temporal_scope). Each check must explicitly
be supported and reference exact quotes in a bounded pool. Validate the whole
claim's evidence with the unchanged grounding-v2 quote/ownership/prefix/parent
rules. Component attestations make omissions inspectable; they are not a
deterministic semantic oracle. Do not infer support from token overlap.

Keep full trusted context and exact source bytes. Current claims lack a trusted
span, so do not guess a reduced context window or edit padding. Preserve eight
claims, eight distinct quotes per claim, 192 characters per quote, 65,536 response
characters, 4,096 requested output tokens and all existing source bounds.
The larger ledger may exceed model output limits on dense batches; reject rather
than silently drop fields, raise limits or claim guaranteed fit.

## Sequential gates

1. A fresh GPT-6 Sol implements only the inactive pure v3 contract and tests.
   Root independently reviews and tests request/schema binding, exact conditional
   coverage, evidence/check consistency, qualifiers, all selector branches,
   malformed/old responses, scope and quote limits before accepting it.
2. A different Sol implements only the inactive atomic gate and its tests.
   Root tests real source reconstruction, multibatch atomicity, collisions and
   exactly one full corrected-list recheck, callback failures and unchanged calls.
3. A different Sol binds an inactive subscription adapter to the accepted v3
   contract. Root verifies exact request and per-turn schema, source/cache pins,
   warm lifecycle, accounting, quota and cleanup with fake protocol IO.
4. Only after acceptance derive a fresh source-bound candidate and diagnostic
   bundle with separate Sol implementers and root verification between stages.
   Preserve old files and receipts. Exercise the real entry/default factory,
   private trusted-batch replay, all unchanged controls, canary and policy faults.
5. If the implementation and integration pass, stage only pinned source bytes
   privately on Afrodite. Verify zero-inference containment/cleanup and private
   synthetic rehearsal; record a new immutable inference receipt before launch.
6. One changed-candidate diagnostic may test this predeclared hypothesis on the
   same 24 controls plus retained canary, GPT-6 Luna subscription, at unchanged
   29-new-turn / 500,000-known-token / 1,800-second campaign caps. Rebind the
   monitor before launch. While active, observation is read-only. Privately replay
   all returned judgments and independently verify cleanup at terminal state.
7. If checks still fail, preserve the result, localize the finite failure data
   and repeat only after an evidence-backed change passes the same sequence.
   Do not blind-reroll. A clean diagnostic is not full-LME readiness: any further
   same-four-question pilot needs its own accepted source binding and receipt.

No production changes, full-500 launch, model/auth switch, increased caps, quota
bypass, credit purchase or raw private text/log/store/credential export. All old
pilots and the cancelled DeepSeek run stay stopped. Both monitors remain paused
until an exact justified new run is bound. Stop and ask for external quota or
materially broader scope. The accepted paired grounding diagnostic is never rerun.

## Status

Stage 1 accepted offline. Separate Sol implemented the inactive contract. Root
reviewed the full source and caught unsupported schema keywords before integration;
Sol removed them, preserving runtime conditional/uniqueness enforcement. Root's
110 independent counterexamples and 27 implementation checks pass (**137 new
checks**). With unchanged v1/v2 classification suites, **362 checks passed**.
These overlapping counts are synthetic contract tests, not model accuracy.

Contract SHA256: `435c0edf52197a5ffa9e715db24156e26109bf7445ad7c9baba2f632ba7f7a76`.
OpenAI Docs constrained the schema to the supported subset, including `$defs`
and `$ref`; the parser still enforces exact conditional coverage and source scope.
No old source, control label, model, budget or production path was changed.
Next gate: a different Sol implements only the inactive atomic gate.

Stage 2 accepted offline. A different Sol added the inactive gate; root reviewed
the full delta, which preserves the previous runtime except contract import,
version and documentation. The `_v2_source` conversion name is deliberately
retained for the unchanged v2 source dataclasses and downstream integration.
Root ran **93 gate checks**, including the initial 19 implementation and 15 independent new
checks plus the prior regression scope. Multi-batch atomic rejection, global
collision/conflict checks, nested source scope, callback exception identity and
exactly one full-list recheck passed. Gate SHA256:
`2f9c56e3b2bbf2a012cb6fb81cf960eeccddd0c8e05fce9bdf07c2c5fdc1c5d9`.
Next gate: separate Sol source-bound subscription adapter; still no activation.

Sol added two further context tests without changing gate bytes; root reran all
36 new gate checks successfully. Stage 3 is also accepted offline: a third Sol
bound the unchanged transport lifecycle to v3's source/AST identities and its own
internal module namespace. Root independently reviewed the complete delta and
ran **49 transport checks** (24 implementation, 25 root), including real inherited
fake-protocol parsing, per-turn schema isolation, quota, accounting, concurrency
and owned cleanup. No model call was made. Adapter SHA256:
`d3c955922219d99b359aa6619b821e7da1662e36ecea54d09b1267dce38d2c32`.
Bound contract AST identity:
`sha256:50d0ba72b0ea5290fba0541e42bc03f1880a3aa70c8014310a7b1b22be6adede`.
Next: separate Sol candidate derivation, then root physical-candidate verification.

Stage 4a accepted offline. A fourth Sol implemented the versioned candidate
builder. Root reviewed the full delta and independently exercised the physical
candidate: **26 checks passed** (10 implementation, 16 root), including real
chunk extraction, full-list correction/recheck, late rejection, collision and
budget failure, all-or-nothing outputs and transitive cache tamper detection.
Exactly two original files change and five helpers are added; 513 files total.

- Builder SHA256: `9595892e727661aee37aa5d4e0aa014935a7974e92f725c2216cc8f40e67f5e6`.
- Root candidate: `/private/tmp/hymem-classification-v3-root-YFcQl4dR/candidate`.
- Map: `/private/tmp/hymem-classification-v3-root-YFcQl4dR/map.json`.
- Map-file SHA256: `f63f656ee1aaa204c13a44f56cf1864596117f13f36189cc3b7ba05cd7646bab`.
- Mapping SHA256: `c52ced91c2168a3cdbb49ab72f4bed840c44763867ebd27e0277f48bdcead934`.
- Extraction identity: `hymem-extraction-contract-sha256-v1:21372a7db114466d4ad248c344ea4493606c858c9fd010c2f77456a9ad31b6c3`.

Independent read-only Sol audit found no pure-module blocker and highlighted
the required bundle rewrite of the transport contract path from repository to
the staged candidate; root will verify that exact binding before activation.
Next: a different Sol diagnostic bundle, still offline and source-bound.

Stage 4b accepted offline. A fifth Sol implemented the fresh bundle. Root reviewed
the generated binding changes and independently exercised real core/canary/stage,
all 24 fixed controls, whole-entry private replay, trusted-batch tampering,
default-factory containment, budget/accounting and finite failure paths. **90
checks passed** (5 implementation and 85 root). New ledger fault cases reject
missing/negative role checks, invalid indices, orphan quotes and old schema.
These remain synthetic judgments, not semantic success.

- Bundle source: `9207e7d757dcc1f2be10c51e5bb8b5666da7788950f1af832c88407751a70029`.
- Accepted bundle: `/private/tmp/hymem-classification-v3-accepted-Yy41cZsw/bundle`.
- Derivation receipt: `cf426350d77615dbb589a74823d49432a7d7bd0583b26a9b4d3e3360ed64f814`.
- Startup/observer: `e61eabe7ae53bd915acd092051014f5ba50dfb96b4c1a28f3b88ff2d6ce15cb7`.
- Entry: `79c675a1aa147df6dced69819d0025b13066208df6c5947ab59b0ec467866c2d`.
- Private replay: `d9d8db0bf9c8d03ae8296eeb378a0267501d52c5abb9bc2676d9862ca840a63f`.
- Root no-inference rehearsal: `656f9ef18865b3efc1218d83ff8e6f471182175b7e6c5f3128443a87bf176dd7`.

The 23 output hashes are sealed in the derivation receipt. Runtime v1/v2
classifiers are absent; the old builder is a derivation dependency only. The
staged adapter points to the candidate and verifies both source and AST identity.
Next gate: final combined scoped regression, then fresh private smoke/rehearsal.

Final combined scoped regression: **1,489 passed, zero failures, 100.27 seconds**.
This includes the new modules/candidate/bundle and unchanged prior classification,
grounding, subscription and transport regressions; stage counts overlap this
total. `git diff --check` passed. No full-project-suite or semantic-model-success
claim is made. Proceed to the separately sealed zero-inference smoke and private
rehearsal before the same-budget live measurement described above.

The subsequent live measurement is recorded in
`2026-09-29-classification-v3-live-diagnostic.md`. It failed and was independently
replayed: 20/24 controls, zero false support/malformed, three false rejections,
one missed correction, prose canary failed. Usage 27 turns/176,340 known tokens;
cleanup verified; monitor paused. V3 is mechanically accepted but **not accepted
for LME**. Continue diagnosis before another narrow repair; no blind reroll.
