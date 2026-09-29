# Compact evidence-assessment prototype — offline only

## Scope

The user's “run it” follows the negative ledger comparison and its recommended
offline redesign. Implement and test that diagnostic redesign only: no paid
calls, external data transfers, credentials, production memory, rollout, restart
or full benchmark. The preceding 238/714 authority remains spent. Preserve all
existing runtime code, ledger/isolation prototypes, fixtures, gold and receipts.

Use a separate implementation agent; root independently reviews its code,
writes adversarial controls, verifies the 33 frozen payload projections and
runs the relevant regression suite before accepting the offline implementation.

## Contract

1. Reuse the strict existing v9 payload validation and item-specific evidence
   projection. Episode/procedure evidence remains limited to its own citations;
   boundary context is interpretation-only; prior summary is available only to
   summary continuity. Never include rejected raw summary as evidence.
2. Code creates immutable candidate fields and evidence units with deterministic
   identifiers, exact original strings and source coordinates. A unit is a
   complete projected source/metadata/context record, not a purported atomic
   fact or a minimal proof. Repeated words need no model-selected occurrence.
   Code also creates the expected check IDs and all diagnostic output metadata.
3. The model returns a flat JSON object keyed only by the expected check IDs.
   Each value is `[verdict, [source_ids...]]`, where verdict is supported,
   unsupported or uncertain. It returns no schema/scope/type envelope, copied
   candidate text, quotes, offsets, field paths or explanation. Exact key and
   value types remain strict; no repair, ignored metadata or partial salvage.
4. Each nonempty ordinary candidate field has two independent obligations:
   assertion support, and preservation of attribution/negation/modality/time/
   causality/order/quantification/exclusivity relationships. Full field text is
   preserved in both; code does not claim sentence splitting equals semantic
   atomization. Null/empty fields are retained in source input but need no
   assertion; whitespace-only nonempty fields must not silently disappear.
5. Episode outcome labels instead receive one explicit classification check.
   Describe their event-level meanings; do not demand that “informational” or
   “deferred” literally occur in source text. Do not exempt outcomes from
   support checks or permit invented successful completion.
6. Add one source-outcome-retention obligation per canonical-text unit in a
   scope. Judge material outcomes relevant to that candidate, not verbatim
   reproduction or every incidental detail. For summaries, consider the full
   new source window and effective summary; prior text cannot replace new
   evidence. For procedures, preserve applicable ordering and prohibitions;
   never imply that candidate-text coverage alone detects source omissions.
   These are additional diagnostic obligations, not replacements for the old
   target-family gold labels. For example, the previous faithful procedure
   label did not require restating an unviolated source prohibition. Future
   scoring must label retention separately and must not turn an off-target
   retention failure into credit for detecting an unsupported assertion.
7. Resolve returned IDs to immutable evidence records in code, without asking
   the model to reproduce quotes or choose duplicate occurrences. Unknown or
   duplicate IDs fail. A supported field/classification/relation needs primary
   support, or summary-only continuity; interpretation alone never suffices.
   A supported retention check must cite its own canonical unit. Negative and
   uncertain judgments may cite nothing but cannot cite unauthorized records.
8. Every check must be present exactly once. Empty check sets are unassessed,
   not vacuous success. Invalid output invalidates the complete scope. Valid
   outputs contain judgments, not proven facts: semantic_verified and
   publication_authorized are immutable false for every outcome/result.
9. Bound complete input/output, checks, evidence references and call count;
   reject oversized or tampered plans before any invocation. Preserve request
   sampling/token parameters. Execute once per scope with a caller-supplied
   client, no provider construction/retries/repair/store access. Continue after
   model-output rejection; halt on client errors; propagate deadlines and
   process interrupts. Validate bindings on standalone parsing and execution.

## Verification gates

- Implementation tests plus independent root tests: scope isolation, Unicode,
  repeated sources/phrases, preserved whitespace, categorical outcomes,
  whole-source retention, unsupported causal/identity judgments, duplicate and
  unknown IDs, missing checks, extra metadata, malformed JSON, forged plan and
  coordinates, bounds, interruption and no accidental publication authority.
- Include adversarial scripted supported judgments on wrong-but-authorized
  evidence: structural validity must explicitly remain distinct from semantic
  correctness. Never present synthetic judgments as observed model accuracy.
- Root audits projections of all 33 existing frozen benchmark-only payloads,
  including the 68 first-failure cases and the known semantic counterexamples.
  Reuse old failures as input characteristics, not as new-format model answers.
  Do not transform, repair or rescore the old paid responses.
- Verify the previous 468 Python/SQL source files remain byte-identical; the
  prototype and its tests are additive. Run the selected related regressions
  with networking/credential access prohibited.

## Acceptance boundary

Passing these tests can establish deterministic contract behavior only. It
cannot establish fewer live JSON errors, better semantic discrimination, lower
paid cost, procedure completeness or LME readiness. Any promising later result
needs a separately frozen/authorized live comparison, untouched examples and
an end-to-end gate. No parser relaxation or post-hoc reroll counts as a result.

## Implemented contract and focused verification

The additive prototype is `benchmarks/digest_evidence_assessment.py`. Its public
API is prepare_evidence_assessment, parse_evidence_assessment and
execute_evidence_assessment. The unchanged strict v9 projection supplies scoped
evidence; standalone parsing reconstructs derived fields/checks from the scoped
original snapshot, and execution rebuilds the complete plan before the first
client call. Rehashing changed derived metadata cannot authorize it.

Code resolves selected IDs to complete original evidence units. This fixes the
mechanical requirement to choose a unique occurrence of repeated names, not
semantic entailment: a selected full record can still contain irrelevant text.
Likewise, separating assertion, relation, outcome and retention judgments does
not establish logical atomicity or guarantee that an LLM will judge correctly.
The flat JSON response still needs model compliance; live JSON errors have not
been measured for this protocol.

The implementation agent's 162 tests and root's 80 independent controls all
pass together: **242 passed**, no failures/errors/skips. Root found and the
agent fixed one implementation bug: preparation accepted a character budget
too small for any complete valid reply. Preparation now checks the shortest
complete legal response, including an empty scope's two-character `{}`. Exact
boundary and forged-plan tests verify the fix before invocation. This is a
character bound, not a token estimate or a reason to alter sampling parameters.
Two initial
root-test assumptions were corrected without relaxing v9 input policy: blank
entity names are invalid, whereas whitespace in an optional description is
preserved; outcome instructions use the verb “classifies” rather than the noun
“classification.” Initial failing receipts are retained in the private gate.

Root independently audited all **33 cases / 86 scope requests**, covering 315
candidate fields, 283 evidence units and 691 checks. Every field, scope index,
source identity/coordinate/authority and sampling parameter matches the
permitted original input. All 68 previously malformed ledger scope inputs are
represented. Scripted new-format replies exercise transport only: no old paid
answer was converted, repaired or rescored. The old paid archive remains
byte-identical, as do all 468 prior source/test Python/SQL files.

The largest request among these cases has 17 checks and 14,153 input characters;
the largest complete scripted positive reply is 450 compact JSON characters.
These are deterministic serialization sizes, not actual model token/cost
measurements or an assurance of better paid performance.

Private verification directory: `/private/tmp/hymem-evidence-assessment.7DUzE2`.

- Prototype: `8960442b26a8c42cd7620138573c589468b6659f6471e5ce080765cc3cdfba1a`.
- Implementation tests: `f4f9005130b74735a318e157a62f5ef02c05bdcd55b068fe13d8069a12e2878c`.
- Root tests: `630fe784b0ffac3844d4390f97f044139634fdcad63712108a406d1934139573`.
- Projection audit: `6b94c505c89864875b279b34e07f046b4e0eac6185cf268dfa828e36334445a5`.
- Focused JUnit: `d62fec0d7f3739151bbd3e9b2d2310678944b85c944902e5ea0b099e5670ac12`.
- Frozen gate inputs: `78233aa944b07f273db623d95b74e9550c5e9d2a660e159962948dcd07b829b9`.

## Completed offline gate

**Offline implementation accepted; no runtime adoption or LME clearance.**
The selected related regression suite passed **2,359 tests / 48 modules** in
605 seconds, with zero failures, errors or skips. This includes the 242 new
implementation/root tests and the previous 2,117-test selected gate; it is not
a claim that the entire repository suite was run.

Root reconciled raw JUnit testcase counts, unique collected node IDs, selected
module inventories, the projection report and current file hashes. All 471
Python/SQL inputs match the gate snapshot: the prior 468 are unchanged and the
three additions are the prototype and its two test modules. Network and
credential-access guards recorded zero attempts. No provider was invoked, no
production memory opened and no rollout or restart performed. The previous
paid comparison and its negative adoption result remain intact.

- Broad JUnit: `1fd68f0fed46eab446aa7ee1841b0b0fceb18a9402215360ba94044aac35b0d0`.
- Broad gate: `27f85ee48c6b2f2f67daadbbaa23ef58792e58794a2a2d7d544ff9b687b023f5`.
- Independent closure: `a6f160288d97b739c6a8aa5006614303ae65b17dbb870371292a6beaab9adcd7`.

Next gate: define separate retention labels and a frozen live-evaluation
protocol, then obtain fresh explicit data-transfer/provider/budget authority.
Live validation must measure supported-case acceptance as well as false accepts,
malformed/uncertain responses and cost. No approval or unused capacity from the
spent comparison carries forward. Offline success alone does not establish an
improvement in semantic accuracy, actual model-format compliance or LME scores.

The next offline preparation is now complete: see the
[frozen evaluation protocol](2026-09-16-assessment-evaluation.md). It retains all
33 previous cases and adds 16 new paired controls, with independently separated
retention labels and explicit ambiguous-label exclusions. An independently found
same-view overlap bug in the new scorer was fixed before freezing. **754 selected
tests passed**; all 471 prior source/test files and the prototype remain unchanged.
The proposed two-repetition comparison is capped at **326 completions / 978 HTTP
attempts** and remains unauthorised. No paid calls, source transfers or runtime
changes occurred; the live runner still needs its own offline adaptation gate.
