# Grounding classification redesign — offline, inactive

## Authorization and purpose

The user said “Run them” after the failed policy-v2 diagnostic and the proposed
offline redesign. Continue sequential Sol implementation and independent root
verification. Do not rerun the unchanged diagnostic. Old sources, labels,
receipts and private evidence remain immutable. No production or benchmark is
activated by this plan.

Policy-v2 measured 19/24 passing controls, four malformed responses and one
missed correction; the retained prose canary correction also failed. This
establishes two distinct problems: output-contract adherence and semantic
classification/decision behavior. No change below constitutes proof of semantic
accuracy until separately measured.

## Design decisions

1. Retain the model-returned batch hash and strict caller-side canonical checks.
   A plain envelope around arbitrary text is not proof of generation provenance.
   Instead, constrain schema/version/hash at generation with request-specific
   `outputSchema`, while still rejecting nonconforming returned responses.
   Official app-server documentation exposes this per-turn field:
   https://learn.chatgpt.com/docs/app-server (Turns / Start a turn).
   The retained 0.158.0 `TurnStartParams.json` also declares it. These prove
   protocol availability, not that a live Luna turn will comply. The existing
   adapter records `response_format_effective=None` and does not send the field.
2. Separate semantic classification from deterministic correction selection.
   For each fixed subject, object, polarity, qualifiers and source, classify
   every allowed predicate as entailed, not entailed or uncertain. The model
   does not choose `supported` versus `replace_predicate`; code does. Keep the
   original predicate in the trusted canonical batch, but omit it from the
   model's classification payload to avoid making it the proposed answer.
3. Retain an entailed original. Correct only when the original is definitively
   not entailed, exactly one different predicate is entailed, and all other
   alternatives are definitively not entailed. Otherwise reject/mark uncertain.
   Evidence existence is not semantic proof. Never turn an uncertain original
   or unexamined alternatives into a unique correction.
4. Use a fixed-order 22-position tri-state vector, with a bounded per-claim
   evidence pool and indexed citations for positive positions. Reuse quotes
   across positions without duplicating text. Every positive requires all the
   existing source, exact-quote, owned-prefix and nested-parent-scope guards;
   negatives/uncertain positions must not cite evidence. Reject unused or
   duplicate evidence, duplicate/invalid references and incomplete vectors.
5. Preserve the current output-character and requested-token limits, call
   budgets and one full corrected-list recheck. Do not claim that a bounded
   character count proves fit under 4,096 model tokens. Test representative and
   worst-case serialization size offline; oversize output remains a rejection,
   not authority to increase caps or drop evidence.

## Sequential work and gates

1. A fresh Sol implements an inactive pure classification module and tests.
   Reuse the immutable v2 source/evidence validator; do not modify it or the
   active extraction path. Include a pure request-specific output schema.
   Root reviews and independently tests selection, all evidence guards,
   canonical/version isolation, schema/request mutations, size limits and
   unchanged fixture labels before another implementation starts.
2. A separate Sol implements an inactive source-bound subscription adapter
   that sends the exact schema only on its matching turn. Preserve model,
   authentication, quota, accounting, deadlines, isolation and cleanup. Test
   real protocol parsing against offline fake streams, warm reuse, concurrency,
   reordered/mismatched turns, failed dispatch, schema drift and no fallback.
   Root independently verifies the actual wire parameters and all outputs.
3. A separate Sol implements the inactive atomic gate integration and offline
   fixed-control/whole-canary simulations. No lexical semantic oracle and no
   relabeling the fixed cases. Root checks full-result atomicity, one recheck,
   source/qualifier immutability and unchanged budgets.
4. Only after all these gates are accepted may a source-bound diagnostic bundle
   and fresh immutable receipt be designed for live measurement. Keep the same
   24 controls plus retained canary and existing 29-turn/500,000-known-token/
   1,800-second ceiling. No blind reroll, larger campaign, model switch, full
   LME, or production deployment is authorized by this offline plan.

Both monitors stay paused. Offline synthetic verdicts test mechanics only;
they do not establish Luna accuracy, canary recovery or LME readiness.

## Stage 1 — accepted offline

Separate Sol implemented `grounding_classification_v1.py`. Root's independent
checks found and reproduced a typed-request binding defect (`False`/`0` equal
`0.0` under Python dataclass equality). Sol replaced this with exact field-type
and value checks. Root also required restoration of the prior predicate
definitions/identity/implicit-claim policy and removed assumptions about
undocumented schema keywords (`prefixItems`, `uniqueItems`). The schema now
uses documented structural primitives; the parser still enforces order and
uniqueness. Live backend acceptance has not been tested.

Root independently ran **114 passing new checks** (91 root, 23 implementation).
These include the unchanged 24 fixed cases using invented exemplar verdicts,
27 selector-state combinations, strict typed request binding, response/source
mutation, quote and scope rejection, ambiguous/uncertain corrections, schema
copy isolation and worst-shaped response serialization. The pre-change scoped
suite also passed unchanged: **354 tests**. This is not a full-project suite.

Module SHA256:
`7c62c6e58305a5b7be256825a52a8cc4b9119888011bf18fe267b18dccab2c06`.
Implementation-test SHA256:
`09cf7bc79bde5aeb4eac66fd6dcc24cf752ff66062eeb895ab1915d50493cbfa`.
The old v2 module remains
`377a688caf183f3645be246b77a445bd94d053def00fff87589061b4b2fc31ec`;
fixed fixture/label SHA remains
`511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925`.
An eight-claim, all-positive, maximum-pool/citation synthetic serialization is
20,093 characters. This is under the character cap but **does not establish
4,096-token fit**. All existing requested limits remain unchanged. No calls.

## Stage 2 — accepted offline

Another Sol implemented the inactive schema-bound subscription adapter. Root
independently exercised the real inherited preflight, budget settlement and
turn parser using fake protocol IO. The exact system/user/schema tuple is
bound per invocation; warm reuse and rotation do not reuse another request's
schema. Same-client concurrent calls reject before changing the active binding.
Model, auth, quota, policy and deadlines remain those of the pinned transport.

Root found and required repairs for factory setup ownership, failed warm
binding cleanup, imported-module identity and the double-failure case where
setup and the first process-close both fail. The adapter now retains an owned
raw-session reference until cleanup succeeds, permitting an explicit cleanup
retry. Prior successful dispatch metadata is not overwritten by a later
pre-dispatch rejection. A successful schema RPC is not treated as semantic
validation; `response_format_effective` remains unknown.

Root's final transport run: **163 passed**, including **39 new adapter checks**
(21 root, 18 implementation) and the unchanged base/warm-v2/warm-v3 suites.
No subprocess model runtime or provider was started; two isolated Python import
checks tested bad cached-module identities. These are offline transport tests,
not evidence that the live service accepts the schema.

Adapter SHA256:
`bf2b0e52d9e5ae8b6c5e6c01822f17afd0f21384006fa69fa6ab03bb8aed8511`.
Implementation-test SHA256:
`e5e34de9ad43c98f5888c284b18da9b1c746471c1c6f51b0dbf9bfaa26c1bd58`.
The old warm-v3 source remains
`0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d`.

## Stage 3 — accepted offline

A third Sol implemented an inactive classification gate. Root's independent
tests reproduced a v1/v2 source-dataclass mismatch before any invocation; the
gate now explicitly converts every source/context field to the v2 types. Root
also rejected a generic callback-exception wrapper because it erased typed
budget, quota and cleanup failures. Callback exceptions now propagate unchanged
to the caller that owns transport classification and stop policy.

Root independently ran **29 passing gate checks** (15 root, 14 implementation).
These cover atomic rejection across batches, one full corrected-list recheck,
collision/conflict rejection, unchanged source/qualifier fields, nested context
reconstruction, legacy NULL provenance, and callback failures without retries.
An end-to-end offline control exercises the gate through the new adapter and
the real transport parser using fake protocol IO, with exact schema dispatch,
synthetic usage settlement and cleanup. Its judgments are invented fixtures,
not model evidence or a semantic-accuracy result.

Gate SHA256:
`0c676c9e10f9dfb9fa005be4ec7b75c799d4e69e37d61ae0953a52e86ed59081`.
Implementation-test SHA256:
`a1803c7a0a3f32d65284cf60ddebc110b1d603b377cfbec748f77c6011152eb5`.

## Combined root verification and remaining work

Root independently ran **837 tests, all passing**, in 17.70 seconds:

```sh
/opt/anaconda3/bin/python3.13 -B -m pytest -o addopts='' -q \
  tests/test_grounding_classification_v1.py \
  tests/test_grounding_classification_root.py \
  tests/test_grounding_classification_gate_v1.py \
  tests/test_grounding_classification_gate_root.py \
  tests/test_codex_subscription_classification_v1.py \
  tests/test_codex_subscription_classification_root.py \
  tests/test_codex_subscription.py \
  tests/test_codex_subscription_warm_v2.py \
  tests/test_codex_subscription_warm_v3.py \
  tests/test_codex_subscription_warm_v3_root.py \
  tests/test_extraction_grounding_contract.py \
  tests/test_extraction_grounding_root.py \
  tests/test_extraction_grounding_v2.py \
  tests/test_extraction_grounding_v2_root.py \
  tests/test_extraction_predicate_grounding.py \
  tools/diagnostics/tests/test_luna_semantic_candidate_v2.py \
  tools/diagnostics/tests/test_luna_semantic_candidate_v2_root.py \
  tools/diagnostics/tests/test_luna_semantic_policy_bundle_v2.py \
  tools/diagnostics/tests/test_luna_semantic_policy_bundle_v2_root.py
```

This is a selected regression suite, not the full repository suite. Old
grounding-v1/v2, old gate, warm-v3 and fixed-control source hashes remain
unchanged. `git diff --check` passed. No model runtime/provider calls, remote
actions, deployment, benchmark reruns, commits or pushes occurred in this work.
Pre-existing worktree edits were preserved. Both monitors remain paused.

The three new modules remain inactive. The old frozen candidate and diagnostic
runner cannot be pointed at them: the callback now requires an explicit batch,
and classification responses require a different replay/measurement path.
The next step is a separately reviewed, source-bound candidate and diagnostic
bundle with updated callback integration and private response replay. After
offline integration checks and a fresh immutable receipt, the bounded live
measurement described above can test schema acceptance and semantic behavior.
Neither canary recovery nor LME readiness is established by these offline tests.
