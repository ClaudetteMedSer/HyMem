# Digest failure-origin correction

## Scope and reproduced defect

The user continued the LME-readiness work. This continuation uses local synthetic
tests only: no provider calls, new data destination, production store, deployment,
restart, default-policy change or full benchmark.

`extract_session_digest` collapsed normal failures from summary repair, the
second fidelity check and the final format check into `fidelity_verification`.
Exceptions during preparation of these later tasks could lose their origin too.
This obscured diagnosis; it did not cause the independently observed false
semantic verdicts. The episode probe contains historical inference for some
of these broad labels, but direct callers should not have to reconstruct them.

A separate implementation agent reproduced 21 failing cases in its initial
33-case suite. Root independently reproduced 16 stage mismatches in 24 controls,
covering both summary policies and optional compaction. Eight controls already
passed (diagnosis attribution and unchanged-repair veto preservation).

## Fix and boundaries

- Track the stage that produced the current rejection, including pre-dispatch
  input caps and malformed or negative replies.
- If a valid repair returns the unchanged rejected summary, retain the original
  fidelity veto and its origin; the last attempted task is not a new verdict.
- Extend the existing stage contexts over the corresponding preparation and
  assembled-validation work. Preserve already-attributed errors and do not
  catch deadline/cancellation `BaseException` signals.
- Add `stage` to the existing warning, keeping its event name and failure reason.
- Keep prompt bytes, source authority, call counts, acceptance rules, retry
  budgets, quarantine, publication and cursor guards unchanged.

The loaded digest implementation identity intentionally changes. A future
deployment may invalidate derived generations even though the wire prompts are
unchanged; this is not a production no-op and no deployment occurred here.

## Verification

Root's 12-case differential against the frozen pre-fix implementation passed.
Across both policies, all request objects and call sequences were identical;
all returned fields other than the intended failure-stage changes were equal.
The cases included direct success, repaired success, malformed repair, unchanged
repair, second-check rejection and format rejection. Seven pinned legacy prompt
hashes remained unchanged. This is behavioral regression evidence, not proof
that a model gives correct semantic or grammatical judgments.

The initial 24 root controls pass after the fix. Root added 12 preparation-error
and cancellation controls. An independent read-only patch review found no
request, branch, validation or publication changes beyond attribution; the
implementer's 53 attribution controls also pass.

The first broad run covered 4,997 tests: 4,944 passed and 53 failed, with no
errors or skips. Its receipt is retained, not relabelled a pass. Of the failures,
48 used old format-stage assertions imported before their test-file update
finished; a fresh run of all 86 tests in that module passed. Four cases in three
probe modules still expected the old second-verifier input-cap label. One
additional probe fixture accepted only a positional payload although the
pre-existing runtime encoder call supplies `system=`; it raised a test-double
`TypeError` instead of simulating the intended cap. A separate agent corrects
only those test expectations/signature, preserving the no-borrowed-reply checks.
Root reviewed the exact four-file correction; the agent's complete four-module
gate passed 100 tests with no failures, errors or skips. Runtime bytes remained
unchanged during these test-only repairs.

The final frozen broad rerun passed **5,017 tests across 93 modules, with zero
failures, errors or skips**, in 649 seconds. It includes all 53 agent attribution
controls and 36 independently authored root controls. The 20 additional cases
relative to the initial run are preparation/control-flow checks finalized before
the frozen run. All **477 watched application, benchmark and test Python files**
were byte-identical from start to finish; the gate rejects file drift.

Scratch receipts: `/private/tmp/hymem-stage-attribution.ZwtAyJ`, including
`frozen-regression.xml` and `frozen-regression.json`. Receipt SHA-256:
`d0d6a5acc04b4233093b16f1026820125a5613f222fcbad445f0deb4738822b6`.
Accepted `digest.py` SHA-256:
`1031cd69374384a180c05a4f964f63e94c440bade5a1a5c4c9127ed3e0b61aa2`.
`git diff --check` is clean. This is the related regression gate, not the entire
repository suite, production verification or a live-model readiness result.

## Remaining readiness blockers

A separate read-only audit found no stale exhaustive-retention rules or lost
prior-summary payload in the bounded fidelity/diagnosis wiring; its 253 tests
passed. The live false vetoes and missed citation defect therefore remain
unresolved, not repaired by this reporting correction.

A second review confirmed an existing generation/format mismatch: summaries
permit an implicit subject, and the granular episode example is a subjectless
past-tense fragment, while the format verifier requires complete sentences.
Complete passive sentences satisfy both, so the contract is not impossible;
the ambiguity does not explain the separate semantic failures. A repair should
align producer guidance with the existing gate, not weaken the gate. Preserve
legacy and bounded-v1 experimental attribution with an explicitly versioned
producer-format treatment rather than silently changing their prompts. No such
treatment was implemented or live-tested in this continuation.

The independent-provider comparison still requires a named destination and
authorization. No response from another provider or full LME completion is
claimed. The previous 92-call evidence remains frozen failed-quality evidence.
