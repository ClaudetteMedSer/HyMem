# Evidence-ledger prototype: offline implementation gate

## Why this is next

**Status: offline implementation gate complete.** Root independently reconciled
2,117 passing selected regressions across 46 modules, zero failures/errors/skips,
zero network attempts, and unchanged existing inputs. This is not a live semantic
quality result, runtime adoption decision or LME clearance.

The completed fixed-candidate comparison found valid JSON and reliable execution,
but isolation still approved all 20 retained known-defect observations. It also
reversed the faithful/defective citation control and added four false rejections.
The next experiment must expose what allegedly supports each claim, not merely
ask for another yes/no judgment. The existing runtime and the failed isolation
experiment remain unchanged so their receipts remain reproducible.

This is a **diagnostic-only evidence ledger**, not a publication validator. Exact
quotations prove location and source authority, not semantic entailment. A model
can still cite a genuine but irrelevant passage, group multiple assertions into
one claim, or mislabel a new fact as prior-summary continuity. None of those
limitations may be hidden behind an `all_supported` or publication-success flag.

## Scope and protocol

Implement a new `benchmarks/digest_evidence_ledger.py` with a separate agent;
root writes independent adversarial tests, audits every retained input projection,
and runs selected regressions. No production integration, stores, credentials,
provider calls, deployment, restart or full LME is authorized by this step.

Reuse the existing strict v9 input preparation from the isolation experiment.
There is one prepared request per episode, raw procedure and effective summary,
with exact caller sampling parameters and a complete up-front call reservation.
All plans and candidate/evidence records are immutable and hash-bound, and the
whole plan is revalidated before execution. No retries, repair or silent scope
omission. Ordinary client exceptions stop execution; deadline and process
interrupts propagate. The caller still owns provider accounting and deadlines.

Each request has `schema`, `scope` (`kind`, original `index`), `fields`, and
`evidence_sources`:

- Fields have `path` and exact `text`. Episode title, body, non-null outcome and
  every key entity are separate fields. Procedure name, non-null description,
  every action/non-null tool, trigger and entity are separate fields, preserving
  multiplicity and step order. Only the effective summary is a summary field;
  rejected raw text is excluded. Null and empty string fields contain no claim.
- Paths are JSON-pointer-like stable paths within the original candidate, such
  as `/candidate_body`, `/candidate_key_entities/0`, `/candidate/steps/0/action`
  and `/candidate_summary`. Whitespace-only nonempty fields remain covered.
- Evidence sources have `source_id` (`s0`, `s1`, ... within the scope), `kind`,
  `chunk_id`, `message_id`, `field`, `start`, `end`, exact `text`, and `allowed_use`.
  Only the item's own cited canonical records enter an item request; summary
  requests get the complete canonical window and prior continuity evidence.
- `canonical_text` uses original message offsets and `support` authority.
  `attribution` is a nonempty role/peer/workspace value, with field-relative
  offsets and `support` authority. Metadata identifiers are not display names.
  `boundary_context` retains its exact original span and permits only
  `interpretation`. `prior_summary` exists only for summaries and permits only
  `continuity`; it is never promoted to canonical evidence.
- Source units are emitted in canonical source order, visible text, role, peer,
  workspace, context, then nonempty prior summary last. No empty source unit is
  emitted. Provenance is retained even if two units contain identical text.

Responses are strict JSON, with exactly `schema`, `scope`, and `claims`.
Each claim has exactly `field`, `text`, `verdict` and `evidence`. Verdicts are
`supported`, `unsupported`, or `uncertain`. Each evidence reference has exactly
`source_id`, `quote` and `use`.

The model is asked to divide every field into minimal assertion-bearing pieces,
in field order. The validator requires their **exact concatenation** to equal
every original field: no omitted subject, negation, number, outcome, entity,
qualifier, whitespace or punctuation; no duplicate/overlapping/reordered fields.
The engine derives character offsets from the ordered text, rather than asking
the model to count characters. This proves textual coverage, not atomic semantic
coverage; a whole-field claim remains possible and is not secretly treated as
proof that every assertion was checked.

The engine resolves each nonempty exact quote to a **unique occurrence** inside
the specified evidence unit; absent, ambiguous, whitespace-only, unknown,
out-of-scope or authority-mismatched references invalidate the ledger. Evidence
quotes are not normalized, repaired, or located in other sources as a fallback.
Resolved offsets retain original coordinates for canonical/context text and
field-relative coordinates for attribution/prior summaries. Repeated identical
references within one claim are rejected. The same real evidence may legitimately
support multiple different claims.

A model-`supported` claim needs at least one permitted `support` or summary-only
`continuity` reference. Interpretation-only context cannot support it alone.
`unsupported`/`uncertain` claims may have no evidence; provided references are
still validated. An empty effective summary may produce an empty ledger, but
must not acquire an affirmative model-support result through vacuous truth.

Bound all request, output, claim and evidence-reference sizes before loops/calls;
reject booleans as integers, non-finite values, duplicate JSON keys, invalid
Unicode, malformed/extra fields and forged/mutated plans. No partial valid subset
is returned as an accepted ledger. Do not reuse the verdict-only closing-trailer
repair on source certificates; this experiment has a new, explicit strict-JSON
contract and leaves the production parser unchanged.

Results separately expose valid ledger structure, the model's recorded verdicts,
malformed ledgers and execution failures. **`semantic_verified` and
`publication_authorized` are always false.** In particular, a valid but irrelevant
quote can pass structural checks without establishing semantic support. Include
an explicit regression test demonstrating that limitation.

## Independent gates

1. Freeze all existing Python/SQL input hashes before implementation.
2. Agent implements only the new module and its own focused tests.
3. Root tests candidate coverage; source/offset identity; quote ambiguity; wrong
   item, context and prior-summary authority; procedure multiplicity; empty and
   Unicode fields; malformed/oversized evidence; plan forgery; exception/deadline
   behavior; and the distinction between quote validity and semantic truth.
4. Root mechanically projects all 21 frozen comparison fixtures without an LLM,
   verifying byte preservation and scope exclusions. Do not relabel historical
   verdicts as ledger passes. Add fresh invented paired stress cases for future
   evaluation, with labels based on permitted sources rather than old replies.
5. Run the new and selected existing digest/probe/publication/deadline regressions
   with networking blocked and credentials absent; reconcile collected tests,
   JUnit and source hashes. Correct findings before marking this gate complete.
6. Record actual results and limitations. A fresh live semantic evaluation and
   cost/length assessment will still be required; prior paid authority is spent.

## Implementation and independent verification

A separate agent implemented only the new ledger module and its own tests.
Root wrote separate adversarial tests and fresh synthetic controls, reviewed the
implementation, and mechanically audited its source projection. No existing
runtime or earlier diagnostic Python/SQL file changed.

Root caught and required correction of three implementation issues before
acceptance:

- Empty strings were exposed as candidate fields despite containing no claim;
  empty/null fields are now omitted, while whitespace-only fields remain covered.
- A list-size lower bound overcounted one comma and rejected some JSON values
  at their exact serialized-size limit. The bound now accepts the exact limit
  and rejects one-character-smaller limits.
- An empty summary incorrectly erased affirmative recorded verdicts for
  nonempty episode/procedure claims. Aggregate model support now requires at
  least one recorded claim, valid complete structure, and all recorded claims
  model-marked supported. This is still **not semantic verification**.

The implementation agent also tightened standalone scope validation against
empty fields, impossible summary indices and excessive field/source sizes.
Initial failed checks are retained, not overwritten. Root corrected a test-only
fixture import during collection; the subsequent focused run passed **211
tests** (109 implementation tests and 102 independent root tests).

Root's independently constructed expected projections match all **60 requests
from the 21 frozen comparison cases**, plus **26 requests from 12 fresh synthetic
cases**: 51 episode, two procedure and 33 summary scopes in total. Candidate
text, source order, exact message spans, authority categories and sampling
parameters are preserved; unrelated item evidence, rejected raw summaries and
item-ineligible prior summaries are excluded. No model was invoked and no
historical verdict was relabeled as a ledger success.

The fresh examples are six faithful/defective pairs: identity authority,
negation versus completed action, unwarranted exclusivity, removed citations,
boundary-context promotion and a prohibited procedure step. Each pair changes
one candidate field or citation list; expected labels stay outside requests.
These are synthetic development controls, **not independently held-out cases**
or live accuracy results. The earlier 21-case live comparison contained no
procedures; adding a procedure fixture does not establish live procedure quality.

Prepared requests contain 4,717–11,256 characters, including system prompts.
Illustrative whole-field `uncertain` responses without citations contain
191–1,197 characters. These are not token measurements or bounds on real
claim-by-claim responses; live token usage, truncation and quoting failure rates
remain unmeasured. A deterministic extra audit checked **10,009 generated JSON
values / 20,018 exact-size and one-character-under-limit checks**, all passing.

Private receipts: `/private/tmp/hymem-evidence-ledger.3YiwoJ`.

- Original 464-input manifest:
  `80540947bbe16cc637a9fad0bcb941ea98f500a46b013f9201e24533cf7aeb57`.
- Implementation module:
  `b9ffc0806d71840e3973f6f62c710e93b154970e05642eeb3290f98cb87c3123`.
- Independent projection report:
  `10e45bb9662f200a92c9c3c2944b0fcc9990fabb90ebcceaf845adcb805ca788`.
- Generated size-boundary report:
  `9d031d321f1058fe42d97640f476de4cb758e0d23025e2dc90a7e73955c13a47`.
- Selected-regression input manifest (468 Python/SQL files):
  `b3add692a4008ef39122313ce14fbc7efa7c864b29b346da18820dd0f5e285be`.
- Selected-regression JUnit:
  `7ac40c79bde65aba9972ee44410ed13566fa846e19f290c84bb195a5346c5cb1`.
- Selected-regression gate:
  `6074fe0eaeccbc47f87b30a44125ab8886efae27b707590a67210e01879af176`.
- Independent gate closure:
  `7b2983b6e8a6fcc1aeda42e6fe47a2be50107333e123c603593ca33f0e6af76d`.

The broader gate passed **2,117 tests across 46 selected modules**, in 593.25
seconds: all digest/probe test modules, lossless digest, semantic generation and
indexing deadlines. Root independently reconciled collected node IDs, raw JUnit
cases, zero failures/errors/skips, both new test-module counts, manifest hashes
and actual current file contents. Networking was blocked, credential-like
environment variables removed, and network attempts remained zero. All 464
pre-existing Python/SQL inputs are unchanged; the four new inputs are the module,
two test modules and synthetic fixture builder. No code changed during the gate.

**Decision: accept the offline diagnostic implementation, not a runtime fix.**
No live evaluation, provider spending, SSH, production-store access, deployment,
restart or full LME occurred. Structural source certificates remain insufficient
to establish semantic entailment. Neither this selected regression suite nor
the successful projection audit clears live model quality or LME readiness.

## Next gate, not execution authority

Only after the offline gate passes, prepare a separately frozen live adapter and
seek fresh explicit approval for its benchmark-only payloads and exact spending
cap. The completed 162-completion comparison is spent and cannot be resumed.

An interleaved baseline/ledger comparison must retain every predetermined
repetition, keep model/sampling/candidates fixed, and separately report malformed
ledgers, unsupported/uncertain verdicts, false accepts, false rejections, latency,
tokens and quoting overhead. Historical baseline results alone cannot control
for serving-time variation. Verify owned-worker cleanup, absolute deadlines,
per-attempt usage and unchanged inputs through the existing supervisor before
paid execution. The adapter must explicitly serialize the always-false semantic
and publication indicators; dataclass properties are not included by `asdict`.

Review the actual claim partitions and quotes against allowed evidence. A real
but irrelevant quote, an unexamined compound assertion or prior-summary
continuity used to invent a new fact must remain a semantic failure even when
the ledger parser accepts it. No parser relaxation, case removal or reroll to
obtain a passing result. A promising development result still needs separately
held-out cases and an end-to-end benchmark smoke before runtime adoption or LME
clearance.

The subsequent live comparison is tracked in
[its preparation report](2026-09-16-ledger-live-comparison.md). Its 238/714 paid
budget was explicitly approved; local worker/replay/scoring gates passed. After
the permission review's initial upload rejection, the user explicitly approved
source transfer. The remote snapshot passed hash verification and its separate
network-disabled Linux gate passed. The single paid comparison completed all
132 invocations / 238 completions / 238 HTTP attempts without execution faults.
Its result is negative for adoption: baseline 54/80 matched target decisions,
ledger 19/80; the ledger's two false accepts versus baseline's 26 come with 40
malformed and 19 uncertain targets, blocking 33/42 supported targets and using
2.18× the tokens. See the report for exact denominators, first-failure diagnosis,
grounded review and the next offline design gate. Runtime remains unchanged;
LME is not cleared, and this paid authority is spent.
