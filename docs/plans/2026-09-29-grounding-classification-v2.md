# Classification v2 — inactive, evidence-group repair

## Authorization and boundaries

The user said “Continue” after the metadata diagnosis and proposed offline repair.
Use separate GPT-6 Sol agents sequentially for implementation, with independent
root review and tests before the next stage. Preserve the failed classification-v1
receipt, old sources and fixed labels. No live inference, SSH, provider call,
benchmark launch, production change or deployment is part of this offline work.
Both monitors stay paused. A live efficacy test requires separately accepted
integration, a concrete hypothesis and a new immutable bounded-run receipt.

## Settled design

A separate read-only Sol design review supports these focused changes:

1. Introduce an inactive `grounding_classification_v2.py` with a distinct version,
   batch identity and prompt. Keep the exact original predicate in trusted batch
   identity but omit it from the model-facing classification payload. Preserve
   all other claim/source/context bytes and all request limits.
2. Keep 22 explicit assessments in the fixed predicate order, using readable
   statuses: `supported`, `not_established`, `ambiguous`. `not_established` means
   the eligible evidence does not establish the fixed claim, not that the claim
   is factually false. Merely missing information does not require factual
   counterevidence. `ambiguous` means eligible evidence gives specific competing
   readings, unresolved plausible support or conflict. Do not convert an old
   `u`, an omitted status or ambiguous returned assessment into non-entailment.
3. Retain a supported original. Correct only when the original is explicitly
   not established, exactly one alternative is supported, and every remaining
   alternative is explicitly not established. Any ambiguity still blocks
   correction. The change clarifies the evidence-relative task; it does not
   weaken the selector or guarantee the live model will follow the distinction.
4. Replace orphan-prone `evidence_pool` plus 22 citation-index arrays with
   `support_groups`: each has nonempty `predicates` (allowed predicate names)
   and nonempty inline `evidence`. Predicate membership must partition exactly
   the supported statuses: no overlap, extra, missing or negative membership.
   Each group must pass the immutable grounding-v2 evidence validator for every
   listed predicate. Reject malformed evidence; never silently discard it.
5. Keep at most eight distinct evidence items per claim across groups and eight
   items per group. Quotes may overlap across groups but not repeat within one
   group. Keep 22 groups maximum, 192-character quotes, 65,536 response characters,
   eight claims per batch and the 4,096 requested-token limit. Grouping shared
   evidence reduces typical duplication; adversarial repetition can still exceed
   output bounds and must reject. Do not claim guaranteed token fit.
6. Keep exact source, owned-quote, context-prefix and nested-parent guards.
   Preserve atomic whole-list rejection, full corrected-list recheck exactly once,
   collision/conflict checks, callback failure identity and existing call budgets.
   No special-case fixture labels or lexical semantic oracle.

The table canary's all-uncertain response is not fixed by evidence formatting.
Document it separately; no synthetic or schema pass establishes live canary or
LME readiness. Do not relabel that response as successful under v2.

## Sequential implementation and verification

1. Sol implements only the pure inactive contract and its tests. Root independently
   tests all fixed controls with invented verdicts, ambiguity/absence distinctions,
   exact positive partition, cross-group overlap, global quote bounds, all evidence
   protections, request/version identity, schema-copy isolation and output sizes.
2. Only after acceptance, a different Sol implements the inactive atomic v2 gate.
   Root tests actual source reconstruction, multi-batch atomicity, one full-list
   recheck, conflicting corrections, legacy NULL provenance, callback errors and
   budgets before accepting the gate.
3. Only after acceptance, a different Sol binds an inactive subscription adapter
   to this exact contract. Keep the pinned transport, model/auth/quota/deadlines/
   cleanup unchanged. Root verifies wire-level schema isolation using fake IO,
   source/cache tampering, warm reuse, concurrency and failed setup/close paths.
   Official app-server documentation says `outputSchema` applies to the current
   turn only: https://learn.chatgpt.com/docs/app-server. Acknowledgment is not
   semantic validity. No API-key tool is available; no provider starts in tests.

No candidate/diagnostic runner activation is included in these three stages.
Record accepted hashes and test results here before proposing the next gate.

## Stage 1 accepted offline

Separate Sol implemented the new pure contract. Root reviewed the full source,
required the public `GroundingContext` type to remain available and removed an
unestablished `uniqueItems` schema keyword (runtime uniqueness checks remain).
Root authored 87 independent checks; combined with implementation tests, **111
new checks passed**. With unchanged v1 and grounding-v2 regression tests,
**350 checks passed**. These counts overlap and are not a full-project suite.

The tested one-supported/21-ambiguous and all-ambiguous patterns still reject
as uncertain. No old failed model response was converted to success. Group
partition, global eight-distinct-quote bound, nested context scope, invalid
unselected positives, old version rejection and typed request binding all pass.
Synthetic shared evidence groups fit the character cap; pathological repetition
over the cap rejects. Token fit and model efficacy remain unmeasured.

- Contract SHA256: `2520e4825df9e4d6f403a301451bb9701e052d9a331b5e31a9e106cbb69ad3c9`.
- Sol test SHA256: `aae13c9e88bcfb6115fb9548437fd4bbbeb95979b6d82a154e3f49bfa3ae4ad6`.

The pure module is accepted for the next offline gate stage, not activation.

## Stage 2 accepted offline

A different Sol implemented the inactive v2 atomic gate. Root reviewed the full
source and independently ran **59 gate checks**, including 30 new checks (15
root, 15 implementation) and the unchanged old gate suites. These verify nine
claims across multiple batches, full-list recheck, late initial/recheck rejection,
cross-batch collision/conflict, preserved qualifiers and source provenance,
nested header/prelude/parent scope, NULL legacy provenance, no malformed-output
rerolls and callback exception identity. Nothing is partially returned on failure.

Gate SHA256: `2b4ba5d0590bf3a27d57f3fa52e1d84011917b368088628df62b6faa3664d307`.
Sol test SHA256: `f224f1399a5db0d2988912741240b1ae4d85f018e54cb709a21c6199fcc47d3b`.
The stage-1 source and prior gate remain unchanged. The gate is accepted only
for offline transport integration; it is not wired into active extraction.

## Stage 3 accepted offline

A third Sol implemented the inactive, source-pinned v2 subscription adapter.
Root independently reviewed the complete diff: only the contract path/import,
source and AST identities, isolated internal module name, and docstring differ
from the prior adapter. The transport lifecycle code is identical. Root reran
the pinned earlier fake-protocol matrix against v2 and added direct checks for
old-v1 batch rejection, exact support-group schema, module isolation and a full
two-turn gate/adapter correction using invented responses.

The final new adapter suites passed **49 tests** (24 implementation, 25 root).
These include the 25%-remaining quota floor, warm reuse/rotation, request/turn
binding, failed dispatch and unknown usage, concurrent-request rejection,
raw-session ownership on setup/cleanup failure, cached-source tampering and
one-shot full-list correction recheck. All protocol input/output was fake;
the only subprocesses were isolated local Python import guards, not Codex.

- Adapter SHA256: `9d760cde8c3c547035f984049cff976df8339248cf10e8c1505243c7efd73763`.
- Sol test SHA256: `b457843b15a8f3f664f8afc50fc62165cb74f36d740a18c0798b09ac004c5ef5`.
- Bound contract AST identity: `sha256:999ba2f673ba27891489fe17607f1e55311aa34e7d1eda648ef997fa8b5ed95c`.

OpenAI Docs constrained this to the existing per-turn `outputSchema` field;
acknowledgment still never sets effective semantic/response-format compliance.
Model, auth, quota, isolation, deadlines, accounting and cleanup remain pinned
to the unchanged warm transport. This is offline adapter acceptance only.

## Final root verification and handoff

After all agents finished, root reran the combined scoped suite: **991 passed,
0 failed, in 20.20 seconds**. It covers all six new v2 test files, old
classification/gate/adapter tests, base/warm-v2/warm-v3 transport regressions,
grounding-v1/v2 and predicate-grounding tests, and the prior candidate/bundle
integration plus independent root tests. There are 190 new-version checks within
this total; the earlier stage counts overlap and must not be added to 991.
This is not the full project suite and all semantic responses are synthetic.
`git diff --check` passed.

Root verified that references to the three new runtime modules occur only among
those inactive modules: no active extraction, old candidate builder, diagnostic
runner or production path imports v2. The old source hashes and fixed labels are
still pinned by passing tests. Existing worktree changes were preserved. No
remote operations, model/provider calls, restarts, deployments, commits, pushes,
benchmark runs or automation changes occurred during this offline repair.

The three offline stages are complete. The retained table canary remains
unproven and the failed v1 diagnostic stays failed. Remaining before live
acceptance: derive a fresh candidate and source-bound diagnostic bundle for v2,
review explicit batch/private replay and new support-group handling, run whole-
entry synthetic and zero-inference containment checks, then record an immutable
receipt for a justified same-24-controls-plus-retained-canary measurement with
the existing 29-turn/500,000-known-token/1,800-second limits. Do not point the old
receipt at new bytes or launch LME on offline-test success.
