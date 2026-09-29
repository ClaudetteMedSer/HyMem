# Versioned bounded-summary treatment

## Approved direction and contract

The user continued after the explicit proposal to separate bounded-summary
faithfulness from underlying-memory coverage. This is a new opt-in architecture
treatment, not a quieter success flag for the old baseline. The existing
`legacy_complete_v1` policy remains the default; `bounded_highlights_v1` is the
prospective treatment. No production deployment or LME clearance follows merely
from implementing it.

Both policies retain the 500-codepoint, one-sentence format and exact canonical
source boundaries. The bounded policy is a navigation/highlights view, not an
exhaustive ledger. Entire unselected topics may be omitted, including prior
topics when relevance/space requires it. Selected material must keep the actor,
outcome, conditions, negation, modality, chronology and scope that make it true.
Compression may not turn an answered request into an unresolved one, a proposed
action into completed execution, or a conditional instruction into an absolute
rule. Prior summaries remain fallible continuity, not new-source evidence.

The policy must not authorize arbitrary empty summaries for substantive material,
promote prior/context-only claims into new facts, widen item citations, or make
summary acceptance stand for item-set completeness. Source preservation,
successful derived processing and semantic completeness are different claims.

## Sequential implementation and verification

1. A separate agent implements pure versioned policy descriptors and stage
   instructions, without changing runtime. Root reviews the contract and tests
   before integration.
2. A separate implementation step integrates explicit policy selection through
   generation, compaction, fidelity screening, diagnosis and repair. Keep legacy
   request bytes unchanged. Root adds independent propagation/failure controls.
   No extra judge, retry or provider call is introduced.
3. Bind the policy into digest generation/retry identity and read-only health,
   preserving recognition of historical generations. A changed policy must not
   reuse a digest cursor or staging generation from another policy. Version the
   benchmark configuration and expose an explicit LME option, without permitting
   cross-treatment checkpoint reuse or mislabelling historical artifacts.
4. Independently test source preservation, recovery, quarantine and publication
   invariants, plus retrieval of an unhighlighted fact in a scoped store. These
   checks do not prove universal retrieval or complete semantic extraction.
5. Run the related offline regression gates, then a freshly bounded, frozen live
   comparison using approved benchmark/synthetic data. Review false acceptance
   and false vetoes under each declared policy. Preserve failed evidence. Only
   validated candidates proceed to untouched confirmation and original-Q1
   end-to-end smoke; do not clear LME on parser tests alone.

## Invariants and limits

- The legacy policy's prompt and validation behavior must remain unchanged.
- Failed/quarantined digest work remains unhealthy. Raw artifacts do not make a
  failed semantic extraction successful, and the strict benchmark gate stays on.
- All item extraction, citation, structural validation, lease, deadline, staged
  publication and complete source-cursor guards remain in force.
- Whole-topic selection and faithful abstraction are model judgments, not facts
  proven by a parser. New controls must distinguish selection from laundering a
  material qualifier out of a retained topic.
- Loaded code identity may invalidate prior derived generations even with the
  legacy policy; unchanged default behavior does not mean deployment is a no-op.
- No scores, live reliability improvements or deployment readiness are claimed
  before the corresponding test has run.

## Status

The pure policy, runtime propagation, generation/retry/health identity, and LME
CLI/configuration/registry integration are implemented. Legacy remains the
default. The opt-in CLI value is `--digest-summary-policy bounded_highlights_v1`;
it is a new treatment, not a corrected label for historical results.

Independent root review confirmed seven pre-edit legacy prompt/template hashes
are unchanged. Root's 50 runtime controls passed, covering both policies and
both primary modes, compaction, repair, unchanged item/source payloads, the
seven-call ceiling, actual-system input accounting, and post-repair vetoes.
Sixteen root controls also pin and structurally validate the twelve prospective
semantic fixtures; this does not establish model accuracy. The implementation
agent's 398-test runtime gate and the LME agent's 339-test gate passed separately.

Seven independently authored real-store controls passed: producer-identical
policy switches, failed and partially staged rebuilds, reopen/no-op behavior,
strict pending/quarantine rejection, and scoped lexical retrieval of omitted
source text after raw pruning with canonical bytes/hash/ownership unchanged.
These do not prove unscoped post-pruning recall, semantic retrieval accuracy or
complete derived-item extraction. Root reran these in the passing combined gate.

Read-side qualification is implemented and independently reviewed: only an
actually published, recognized bounded automatic summary receives the
non-exhaustive highlights label. Configured policy and private staging cannot
relabel an older publication. Honcho counts the complete label in its existing
whole-item context budget; stored summary bytes and their cap remain unchanged.

The combined root gate passed **4,210 tests, zero failures/errors**, with four
unchanged raw-SDK dependency skips, across 66 test modules. All 474 watched
Python source/test files stayed unchanged. Receipts: `regression-v2.xml` and
`regression-v2-gate.json`. Network connections were denied; local hostname
lookups and Unix socket construction for the in-process ASGI test client's
event-loop wakeup were allowed. The earlier interrupted `regression.*` gate
preserves four setup errors caused solely by denying that local socketpair;
no application change was made to resolve those harness errors.

One additional reporting defect was reproduced and fixed after integration:
the LME arm comparison counted the policy's validated duplicate inside effective
config as a second lever. The narrow fix discounts only that verified duplicate;
unrelated nested differences and JSON scalar types remain distinct. Historical
absence remains unevidenced. Root's 402-test follow-up passed (`root-lme-final.xml`).

The fresh bounded comparison follows the
[live protocol](2026-09-18-bounded-summary-live.md). All 115 collector/launcher
checks passed. A credential-free, network-disabled Linux rehearsal completed
all 40 tasks with 56 scripted calls and zero HTTP attempts; the container exited
cleanly. Root independently replayed every runtime task from recorded replies
against the frozen extractor and verified canonical source identities/bytes,
exact requests/results, ownership receipts and cleanup. This establishes wiring,
not model semantics. All 17 additional adversarial receipt-auditor controls
passed before paid launch.

The [60-call live pilot](2026-09-18-bounded-summary-live.md) is complete and fully
accounted, but did not clear the quality gate. Bounded runtime returned four of
four cases versus legacy two of four, while the fixed direct-control judgments
were 10/12 versus 11/12: bounded had two false vetoes; legacy had one false
acceptance. Both policies accepted an unauthorized claim in a retained item's
single-citation scope. Independent review also found one format false acceptance
and a legacy prior-continuity false veto. These are semantic/model-compliance
failures, not transport or parser failures. No runtime promotion, production
change or LME clearance follows. Earlier diagnostics and frozen evidence remain
unchanged. The separately bounded
[32-call model-only comparison](2026-09-18-verifier-model-comparison.md) also
completed cleanly but failed quality: Pro caught the retained citation defect,
yet falsely vetoed four of six faithful controls; Flash vetoed three. Neither
is promoted. Additional paid testing outside DeepSeek needs a provider choice
and new destination authorization; no such transfer is assumed.

A subsequent [local attribution correction](2026-09-18-digest-failure-attribution.md)
fixes later digest failures being mislabeled as initial fidelity failures. Its
frozen related regression gate passed 5,017 tests, with all 477 watched files
unchanged. Prompts and acceptance rules are unchanged; this improves diagnosis,
not semantic model reliability, and does not clear LME or deploy the code.

Receipts are under `/private/tmp/hymem-bounded-policy.1Z6v6R`. The initial root
runtime receipts preserve two test-harness mistakes (a nonexistent table name
and using untrimmed length in a scripted compaction dispatcher). Only the test
harness was corrected; `root-runtime-accepted.xml` is the passing 50-test rerun.
