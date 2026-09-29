# Correct retention ownership before another live experiment

## Finding

The continued architecture audit found a deterministic evaluation-design error:
**a supporting citation is not an exhaustive coverage assignment**. Two legitimate
procedures can cite the same message describing two independent technical tasks.
Each procedure must preserve its own task's conditions, not duplicate the other
task. Likewise, granular episodes intentionally represent separate events.

The v4 source-only inventory receives the entire cited source plus an item kind
and array index, but no independent task/event assignment. Swapping two different
procedures at fixed indices leaves those source-only requests byte-identical.
Requiring all material source obligations in each procedure therefore imposes an
incorrect contract. More detailed atomization would not supply that missing
ownership information.

This was independently confirmed against the producer contract in
`hymem/extraction/prompts/__init__.py`, runtime fidelity in
`hymem/dreaming/digest.py`, and the real diagnostic constructors/parsers. It is an
offline-diagnostic defect, not evidence that runtime extraction regressed.
The previous source-review prompt could consider candidate relevance; removing
the candidate without supplying source-owned task groups lost that distinction.

## Bounded correction

A separate agent added `benchmarks/digest_summary_retention.py`. It exposes an
explicit **summary-new-source-only** experiment:

- Validate the complete payload and every original projection before selecting
  the exact original summary scope. Item citations never redefine that scope.
- Reserve two invocations, not two per item. The complete record plan is retained
  for validation, not as an executable item-call schedule.
- Preserve the existing source-only and matching request bytes, source ownership,
  effective summary, interpretation context, and separately labelled prior.
- Require versioned immutable bindings at each stage; reject scope substitution,
  candidate mutation, incomplete references, malformed replies and uncertainty.
- Report no item-retention, grounding, prior-continuity, publication or semantic
  verification. No result is a whole-digest or LME acceptance.
- `collection_complete` means both replies returned without a client halt. An
  invalid/empty first inventory or a preparation-bound skip is not complete
  collection; an empty inventory cannot yield a vacuous retention affirmation.

The v4 module, its prompts and all frozen live evidence remain unchanged. This
new entry point does not repair or reinterpret the old per-item API. That API
remains available for historical reproduction, not a valid general per-item
completeness gate or a candidate for runtime promotion.

Root's review also required two corrections before acceptance: skipped dependent
work must not count as complete collection, and cancellation must propagate
unchanged even if a client callback invalidated the plan. Root tests use an
independently constructed multi-topic payload and adversarial callbacks.

## Verification

The implementation agent's related gate passed 390 tests. Root independently
reviewed the code and added 43 boundary/callback tests plus 18 fixture tests.
Another agent independently reviewed the final adapter and ran the two boundary
modules with network denied. Root's expanded final gate passed **3,207 tests,
four unchanged pre-existing skips, zero failures/errors across 47 modules**,
including runtime digest, failure-adaptation, recovery and fidelity controls.
All 213 watched source/test files remained unchanged throughout the final gate.
This is a related-module gate, not a new full-suite run or live semantic test.

The first expanded gate reported 46 failures because root's audit hook blocked
`socket.gethostname`, a local lookup required by dream lease-token generation.
All 46 failure traces contained that blocked call. Both the failed receipts and
the original harness remain preserved. A fresh harness allowed only that exact
local event, still denying every other `socket.*` event. The identical code and
tests then passed: 187 local hostname lookups, one blocked socket construction,
zero network connections in that gate. No application or test assertion was
changed to obtain the pass. A separate 54-test digest-only run also passed.

New fixtures were corrected before any provider query: semantic source-ID hints
became opaque IDs, and completed/blocked episode outcomes were explicitly aligned
with their sources. Four new controls contain 20 targeted labels (18 retained,
two omitted), including summary omissions that complete sibling items cannot
repair. These are AI-authored development controls, not untouched LME holdouts.
Old fixtures and all paid-run labels remain unchanged.

No production service, store, runtime acceptance rule, model prompt or retry
limit changed in this step. Comparing 164 previously pinned sources found only
the already-accepted v3-to-v4 diagnostic change; no runtime differences. No paid
call was made. The two selected model requests remain unchanged, so these tests
cannot establish improved semantic accuracy.

Receipts: `/private/tmp/hymem-summary-scope.IDIoey/` (`root-focused.xml`,
`regression.xml`, `regression-gate.json`, `regression-final.xml`,
`regression-final-gate.json`). Adapter SHA-256:
`d7e73df3cb1622d06a2c5eecc8cd3a516c251d54543e4c0128ed45785f7e2b73`.
Fixture SHA-256:
`a2be35bb80530982cc4ed3e635bad6ea7f5a86584592959bc921f62d598c24c4`.

## What this does not resolve

The [halted v3 evaluation](2026-09-18-retention-contract-repair.md) remains failed.
Its single-task background-detail vetoes, unsupported strengthening and actor
contradictions are not all explained away by this additional multi-topic scope
finding. Original denominators, labels, reviews and receipts remain unchanged.

The next semantic hypothesis must target the summary's real compression contract,
not arbitrary item exhaustiveness. Explicit source-only materiality decisions and
shared event/relation components may help, but neither typing nor source offsets
prove materiality, entailment or completeness. Test false acceptance alongside
false vetoes; do not give the matcher an obligation-waiver escape.

Per-item grounding and preservation of the selected task's mandatory qualifiers
remain separate requirements. Future item-set completeness needs independently
defined source task/event groups, qualifier closure and candidate-to-group
mapping. Summary coverage cannot substitute for these checks.

No runtime promotion or LME clearance follows this change. Any replacement must
also cover grounding, prior continuity, formatting, strict cursor/publication
rules and recovery, then pass untouched confirmation and the original-Q1
end-to-end smoke. Adding another diagnostic alone does not change the production
failure path.

## Architecture decision before further semantic implementation

The actual summary contract also has a feasibility limit: 500 Unicode codepoints,
one sentence, newly important outcomes plus distinct earlier topics. The current
source-only inventory has neither prior-summary contents nor an explicit
allocation/compression contract. Unbounded distinct material cannot in general
be preserved in a fixed-size string. A typed inventory cannot resolve that
capacity mismatch. This is a design limitation, not proof that every observed
small-case model error or the previous timeout was caused by capacity.

A possible next treatment separates bounded-summary faithfulness from memory
coverage, with explicit allocation and separate item-grounding/continuity checks.
It must be designed and versioned as a changed architecture; raw-source retention
must never be relabelled successful semantic extraction, nor may the old strict
baseline gates be silently relaxed. Root asked the user whether to proceed with
this versioned redesign or preserve the existing contract. No such redesign,
paid campaign or production change is authorized by this document itself.
