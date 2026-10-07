# Staged grounding repair after the fixed A/B diagnostic

## Proven observations, not a blanket model diagnosis

The completed diagnostic and independent replay are recorded in
`2026-09-29-claim-task-live-diagnostic.md`. Original-only B supported control 9
and the table that A rejected. Other earlier false negatives did not reproduce.
This supports testing task separation, not claiming one-draw causation.

A finite private check localized the prose malformed response: its `prefers`
assessment cited an 84-character quote correctly in `boundary`, then a
28-character quote that occurs exactly in owned text but labeled it
`conversation_0`, which does not exist for that source. The parser correctly
rejected it. The v3 output schema nevertheless permits every global region name
for every claim. This is a concrete avoidable schema/validator mismatch: it
allows an impossible region that trusted input can rule out before generation.
Do not relabel or repair the captured response; it remains failed.

Control 19's two rejected responses used exact quotes from owned and context
but joined an owned quote outside context's permitted prefix. The guard is
correct and must remain. No accepted false support in this diagnostic implies
neither broad correctness nor a reason to remove the guard.

## Sequential plan

1. Separate Sol: new inactive, versioned classification contract derived from
   v3, narrowing each claim's evidence-region schema to `owned` plus that cited
   source's actual attached regions. Preserve all full source bytes, selectors,
   bounds, quote checks, qualifiers, roles, prefix and parent guards. Distinct
   wire/extraction identity; leave v3 and its receipts immutable. Root proves
   the narrowed schema rejects the observed impossible-region shape and accepts
   correctly attributed evidence, including multi-source/legacy/parent cases;
   all prior safety controls must still reject. This fixes the schema gap, not
   semantic accuracy.
2. Different Sol after root acceptance: inactive staged original-only and
   alternatives-only contracts using the same bound-region safety layer. The
   first stage assesses original claims, never alternatives. Only a validated
   `not_established` result permits exhaustive assessment of every other named
   predicate. Do not insert fabricated original/alternative states. Trusted
   selection requires exactly one supported replacement and every other
   alternative explicitly not established. Ambiguity or multiple positives
   rejects. Root verifies binding and all malformed/selector branches.
3. Different Sol after acceptance: atomic staged gate. All fields except a
   uniquely validated predicate stay identical; one original-only recheck of
   the whole corrected list, with no correction during recheck. No best-effort
   dropping, acceptance of invalid evidence, hidden retries or relaxed gates.
   Root verifies request counts, whole-list atomicity and negative controls.
4. Only then source-bound transport/candidate/runner integration, each as a
   separate narrow Sol implementation followed by root verification. Reuse
   existing accounting, ownership, quota, containment and privacy rules.
5. Before any new live measurement, predeclare a finite focused schedule fitting
   **the existing 29-turn ceiling**, not a larger cap. Suggested first set:
   controls 9/12/13/17/19/21 and initial table/prose canaries (eight units, at
   most three stages each). No repetitions, refills or re-extraction calls.
   Seal a fresh immutable receipt only after independent offline tests, private
   retained-input rehearsal and zero-inference isolation smoke. Privately replay
   all results and verify cleanup. A passing focused test is not full LME.

## Limits and status

Same GPT-6 Luna subscription/auth/low reasoning. No production, full-500, quota
bypass, credits, model switch, spending-cap increase or unchanged reroll. Existing
29 turns/500,000-known-token threshold/1,800 seconds and server cleanup bounds
remain ceilings; extra stages must fit within them. Raw evidence stays private
on Afrodite. Model statements remain attestations, not entailment proofs.

The prior A/B run is complete and immutable; its monitor is paused. No next live
campaign is prepared or launched.

Step 1 accepted offline: a separate Sol added the inactive v4 classifier. Root
reviewed the complete v3/v4 diff: only wire-version identity and per-source region
enums changed. Root independently passed **307 scoped checks** covering both
versions, all frozen controls, unchanged selector/safety cases and the new schema
counterexamples. The observed wrong-region shape is now rejected by the schema;
properly attributed owned/boundary evidence is accepted mechanically. In-scope
region names still do not bypass prefix guards. This is a verified schema fix,
not a live model success.

Step 2 accepted offline: a different Sol added the inactive staged contract.
Root read the implementation completely and independently passed **74 focused
checks** (14 implementation tests and 60 root controls). Actual original responses
are canonicalized and hash-bound to alternatives; no negative assessments are
fabricated. Only negative indices receive exhaustive alternatives, and the
existing trusted selector and all evidence guards remain active. Root caught and
Sol repaired two draft defects before acceptance: nullable alternative assessments
in the output schema and boolean indices in a forged alternatives batch.

Accepted `hymem/extraction/grounding_staged_v1.py` SHA256:
`4862e4aedba5be756ea877d65a91b515a142b2fb46e2efb6f91c800e5096b3c9`.
Next: a new Sol implements the inactive atomic gate, followed by independent root
review and fault controls. No new live run is authorized by offline acceptance.

Step 3 accepted offline: a new Sol implemented the inactive atomic gate. Root
read the complete source and passed **962 scoped regression checks**, including
24 new root gate controls (counts overlap earlier suites; this is not a full-suite
claim). These cover late multi-batch corrections, complete recheck, atomic
failures, ambiguity without alternative work, exact prior-response binding,
collision/conflict rejection, unchanged budget exceptions, and prefix/parent/
role/evidence safety. Each bounded batch runs original then optional alternatives;
any correction triggers one original-only pass across the entire corrected list.

Accepted `hymem/extraction/grounding_staged_gate_v1.py` SHA256:
`e843a3112a2ed0e7900f97b19a944d0a74fa458682c88a9504379c08c07deaa8`.
Callback: `invoke(request, batch, stage, recheck)`, with original/v4 batch or
alternatives/staged batch. No production caller is activated. Next: separate
Sol transport implementation and independent root verification, before candidate
or runner integration.

Step 4a accepted offline: a separate Sol implemented the inactive staged
subscription transport. Root reviewed the complete source/diff and passed
**251 scoped transport/warm tests**, including 37 independent root controls.
The three stage types share the same budget and a fourth turn is blocked at a
three-turn limit. Old generic dispatch methods cannot bypass stage binding.
Root reproduced two draft problems (raw-vs-AST import commitments and three
unverified transitive-helper bindings); Sol fixed them, and the counterexamples
now fail closed before I/O. Existing quota, turn identity, unknown usage,
rotation and cleanup behavior is unchanged.

Accepted `benchmarks/codex_subscription_staged_v1.py` SHA256:
`3b4df724cafa77cf0f4ae794672a29bdf2bf76e38079337a3bd82ecae1685824`.
Next: new Sol for the isolated physical candidate builder, followed by root
byte-inventory, cache-identity, accounting and extraction-path fault checks.

Step 4b accepted offline: a new Sol added the isolated candidate builder. Root
reviewed every transformation and independently passed 22 physical-candidate
controls, including the real extraction path's multi-batch recheck, stage call
limits, malformed/provider/prior-binding failures and atomic removal of triples,
markers and type hints. Runtime mutation controls confirmed that each consumed
staged prompt/schema/parser and inherited source validator changes the cache
identity or fails closed. The candidate changes only two original files plus
six exact helper additions; production is untouched.

Builder SHA256: `b1b12a9a4e420c48f16a94866c2bae6cd9576b89e41c87c05373594b28303d0e`.
Root independently built `/private/tmp/hymem-staged-v1-root-fZvNIRgs/candidate`
(514 files), with sibling `map.json` SHA256
`228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf` and mapping
SHA256 `9e3fe4e495585f63bb560b123fbc0edb6441396ec28ae3f8fa6bfb6ba5b8cfae`.
Extraction identity:
`hymem-extraction-contract-sha256-v1:94a1adfc028694e9b1868ec8712baff38d4d7a99a9d2e3547c66819711890f95`.
Next: separate Sol for the finite eight-unit diagnostic core, followed by root
verification before source-bound host/reader/replay integration.

The candidate's final combined suite is **34 passed** (12 Sol and 22 root).
The next eight-unit measurement is explicitly a **staged contract probe** on
unchanged frozen v4 batches. It is not a live physical-gate or full-LME result.
Reconstructing raw source records from already-bound leaf/context objects would
introduce another input transformation; the probe instead uses the accepted
contract sequence directly, while the actual gate remains covered by the
independent physical extraction tests above. Frozen controls and both initial
canary batches keep their original predicates and complete source/context bytes.

Per unit: at most three turns, 100,000 known tokens and 240 seconds within the
unchanged global ceilings. Valid rejection, ambiguity or contract-rejected model
output ends that unit and permits the next independent case; no same-unit retry.
Transport, accounting, identity, quota or cleanup failure stops the campaign.
Malformed output remains a quality failure even if the safety gate rejected it.

Step 4c accepted offline: a separate Sol implemented the fixed eight-unit staged
diagnostic core. Root reviewed the full source and independently exercised all
gold outcomes, malformed stages, false supports, rejected rechecks, token-budget
overshoot, unknown usage, journal/cleanup failures and real physical-candidate
canary derivation using invented ordinary responses. **55 focused checks** and
**252 combined staged contract/gate/transport/candidate/core checks passed**.
Root found and Sol repaired draft scoring mistakes (unchanged `uses` prose
falsely scored correct and false support hidden inside a rejected claim pair),
plus metadata inconsistencies. The exact original and alternate states now
remain visible, and diagnostic completion never implies semantic acceptance.

Accepted `tools/diagnostics/luna_staged_core_v1.py` SHA256:
`a389f6e8a522d878a140234ac9c040f2e3daab77bfc7cbaeadf0f4d5b12676bd`.
Next: separate source-bound host/runner integration, followed by independent
root verification; separate read-only replay/observer integration afterward.
No new live calls have been made and no new campaign is launched.

Step 4d accepted offline: a new Sol implemented only bundle/host provenance.
Root reviewed the generated host and passed **21 checks**, including deriving
the exact fresh 514-file candidate through the real host preparation function
with invented retained metadata and all external commands denied. Root caught
two integration defects before acceptance: pending execution modules were not
bound to their supplied hashes, and accepted-tree verification referenced the
wrong builder object. Both counterexamples now pass. Existing one-shot launch,
admission, write-once and root-boundary functions remain AST-identical.

Accepted `tools/diagnostics/luna_staged_bundle_v1.py` SHA256:
`deb85959853afa6d2847e07ea90afde6f1763c6fbd42f8f99561de3cef1b6f82`.
This is **not a complete launch bundle**: preparation fails before creating
output until the real staged runner, observer and replay modules exist. No
executable placeholders or remote artifacts were created. Next: separate Sol
implements the staged runner and root verifies it before observer/replay work.

Step 4e accepted offline: a different Sol implemented the staged execution
wrapper. Root reviewed the full source and passed **32 runner checks**, including
actual isolated preflight imports from the 514-file candidate, runtime contract
identity, budget-class substitutions, pre-/post-run workdir rules, exact early
source pins, malformed/unknown-usage/cleanup failures, write failures and process
tracking failures. Root caught stale helper/import bindings and a boolean marker
comparison regression before acceptance; all counterexamples now fail closed.
Owned process records include each unit key, allowing independent cleanup proof
without assuming one process or one turn per unit.

Accepted `tools/diagnostics/luna_staged_run_v1.py` SHA256:
`82f34e19702b03b48ffeade68a94794c7aa1df18bcd006e340ff8f6272078987`.
No model calls, remote artifacts or launches. Full Linux/private-store preflight
is still a later gate. Next: separate read-only private replay implementation;
root validates every stage and accounting-prefix fault before observer work.

Step 4f accepted offline: a separate Sol implemented private, zero-inference
replay. Root reviewed the full source and passed **65 checks** (three Sol and
62 independent root controls). Replay binds exact requests, responses, schemas,
prior-response hashes, evaluations and chronological unit journals to cumulative
accounting. Interrupted dispatch, unknown usage and settled token overshoot are
retained without inventing completion. Root caught budget-shape assumptions,
boolean/integer equality, reordered unit blocks and partial-tail attribution
before acceptance. The CLI prohibits network, subprocesses and writes.

Accepted `tools/diagnostics/luna_staged_replay_v1.py` SHA256:
`33c5fb61f87fe228559c609e63bb19803b72d22dfa188f539b14306b542ff0a7`.
This proves offline reproducibility, not semantic accuracy or process cleanup.
Next: a new Sol implements the independent read-only observer; root verifies
terminal projection, replay, ownership, containment and failure cases before
startup integration. No new remote work or model calls have occurred.

Step 4g accepted offline: a new Sol implemented the read-only observer. Root
reviewed the source and passed **48 observer checks** (five Sol and 43 root),
with **69 combined observer/bundle checks**. These use real local runner journals
and independent replay with invented OS state. Progress must reconcile against
every replayed unit-finished record; exact typed failure markers, PID identities,
group absence, per-unit process coverage, service policy and admission all gate
clean completion. Root caught loose boolean equality and insufficient progress
reconciliation in the draft. Unknown usage remains unknown. Gold matches and
malformed outputs are reported separately from execution cleanup.

Accepted `tools/diagnostics/luna_staged_progress_v1.py` SHA256:
`1be546cf33b9e5364e29b8c2d196ed34caca478c417aff233c98c0d8f2f44e45`.
Root prepared the complete 37-code-file base bundle locally at
`/private/tmp/hymem-staged-bundle-root-B0isuUYP/bundle`, derivation receipt SHA256
`d121f9d8c89add8f3ebe6b3c2a56d37bfa3744b2c4392e30e7a98bdcca9d4a90`.
This has not been uploaded or launched. Next: a separate Sol derives a new
startup sidecar from the accepted adapter with scoped control-plane access and
the staged runner's direct containment hook. Root then verifies startup faults,
private input rehearsal and zero-inference containment before bounded inference.

Root's final combined pre-startup regression: **595 passed**, covering the v4
schema, staged contracts/gate/transport, candidate/core/bundle/runner, replay,
observer and root rehearsal helper. This is a scoped suite, not a full project
suite. The root rehearsal helper passed seven checks, including the physical
candidate and eight exact invented ordinary responses in an isolated process.
Its SHA256 is `670d3fa04385b81f40ac98414e188c95d3fdcb494953a107ca16859bd95f25c0`.
The live experiment is predeclared in `2026-09-29-staged-grounding-live-diagnostic.md`.

Step 4h accepted offline: a separate Sol implemented startup derivation. Root
reviewed the generator and emitted adapter, and passed **38 checks** (eight Sol,
30 root). These cover exact sidecars, bad pins, direct staged-runner containment,
restored control-plane proxies, real host one-shot markers, child environment
isolation, zero-admission versus unknown-usage failures, exact copied inventories,
and independently interpreted service/OOM/resource cleanup. No accepted source
was edited. Startup generator SHA256:
`09961e17fc0565049685ef9f8fe1a631132f2c9b237f52119370cb6efdfbfa8a`.

Root independently derived `/private/tmp/hymem-staged-startup-root-vBg9BWn4/bundle`;
startup receipt SHA256 `bf4ad18e66943c0ae850c09cea03faa775f377cf25c59178ba156cf3e9f8d0d1`,
adapter SHA256 `7a0fe60090075fd37acca0cead602a795c2562a1ec8481de14b0c05a5c1b7dce`.
All 37 base code files remain byte-identical. Next: private server rehearsal and
separate zero-inference smoke; only then seal the fixed eight-unit inference run.

## Live verification and decision boundary

The planned live diagnostic finished cleanly: eight units, 16 returned turns,
100,799 known tokens, zero malformed responses, independent replay and cleanup.
Six expected outcomes matched; role control 21 and the prose canary did not.
The full evidence, source hashes and paused monitor are recorded in
`2026-09-29-staged-grounding-live-diagnostic.md`.

Two additional read-only Sol reviews and an independently tested private evidence
checker ruled out missing prose context and incorrect applicability. A correct
invented preference witness passes unchanged validation. The role false support
is an erroneous model attestation; requiring literal subject names is not a safe
generic fix because it breaks legitimate pronouns, aliases and headers. No further
implementation defect has been established. Do not patch a validator just to make
these frozen examples pass or rerun unchanged until a favorable sample appears.

The next step needs an explicit experimental-design choice: retain strict semantic
canary admission and investigate a broader role/cue witness design on held-out
controls, or separately design a diagnostic-only LME mode that records semantic
misses as quality outcomes while keeping source/accounting/isolation integrity
guards mandatory. The latter is not automatically a canonical comparable score.
Neither option, nor a full LME launch, has been implemented/authorized by the
completed probe. Production is unchanged and all monitors are paused.
