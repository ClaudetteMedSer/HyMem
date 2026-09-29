# Runtime digest fidelity repair plan

Status: F1–F7 and broader fixture compatibility accepted after sequential
implementation and independent root verification. The first complete offline
gate finished without acceptance and exposed F6 and F7, now corrected by fresh
agents and checked independently. Its frozen source and failed receipts remain
untouched. The second full gate below determines final suite acceptance; focused
passes alone do not establish it.
Approval covers application changes and offline tests only, not new provider
requests, deployment or a diagnostic reroll. Completed v9 evidence remains immutable.

## Diagnosis

The six-call v9 diagnostic completed mechanically but failed source-fidelity
review. Its primary and compaction prompts already request correct scope,
new-span citations, outcome/sequence retention and one-sentence summaries.
Current validators do not enforce those semantic rules: they check response
shape, field types, known citation identifiers and character limits. Existing
scope/sentence tests deliberately describe themselves as prompt-wiring tests,
not real-model adherence or semantic-validation tests.

Fresh F1 agent `fix_digest_title_scope_v10` and root independently reproduced
the runtime gap without provider calls. All of these titles are accepted
unchanged by `_validate_digest_response`:

| Candidate title | Candidate body | Assessment |
| --- | --- | --- |
| Atlas is Cedar-only | Available through Cedar, not Birch | Unsupported strengthening |
| Cache always works | May work if enabled; offline use unverified | Unsupported certainty |
| Caching is enabled | Caching is not enabled | Negation reversal |
| Atlas is Cedar-only | Available exclusively through Cedar | Supported positive control |

This does not establish that every title/body disagreement is unsupported by
the original source. The body is itself generated, not evidence. A legitimate
strong title can be grounded in the source even if its shortened body omits
that strength. Actual verification must use the item's cited, newly visible
source spans, with separately marked boundary context, not only its body.

## Architectural decision

Approved: one shared, batched source-aware verification request after the
final candidate passes structural checks and before any cursor or publication
advances. All items are checked in that request; the call count must not grow
with the number of episodes. Subsequent fixes extend this same boundary rather
than add a new request for every field.

This is a meaningful cost and behavior change. A candidate ordinarily uses
one completion, or two when summary compaction is needed; verification makes
those two or three. That is 100% or 50% more completion calls for these digest
paths, respectively, **not** an estimate of total benchmark cost or token cost.
The existing provider/client is used; no additional vendor or data destination
is proposed. Implementation approval is not permission for another paid run.

A strict verifier can introduce false rejections and more held attempts. Its
judgment is probabilistic screening, not canonical proof or a guarantee that
LME will finish without faults. Source bytes remain the evidence. All requests,
verdicts, failures and token/attempt counts must be observable and bounded.

The zero-additional-call alternative for F1 is to replace generated titles
with neutral canonical labels (or duplicate their full bodies). This removes
the independently generated title assertion but materially changes useful
titles, retrieval inputs and granular identity, and does not fix F2/F3. Do not
silently make that tradeoff. A keyword blacklist, title/body token overlap,
extractive substring with dropped negation, primary-model self-attestation,
or another prompt-only instruction is not an entailment fix.

## Sequential implementation and independent verification

The user requested a fresh implementation agent per issue, followed by root
verification before starting the next issue. Preserve that order.

### 1. F1 — title scope and the shared verification boundary

The assigned agent `fix_digest_title_scope_v10` completed the approved first stage.
Root owns independent controls and related fixture compatibility.

F1 acceptance: root compared the application diff directly with the immutable
v9 source archive; changes were confined to the digest module. Agent controls
passed 60/60. Root's first related run passed 442 cases and exposed two legacy
test clients without verification responses; both were explicitly adapted in
tests only, not via a production stub bypass. Final root rerun passed 146/146
(digest, title verifier, independent root controls and summary contract), with
the other related cases already passing. Fourteen independent root controls
exercise real bounded-window construction, forged source labels, Unicode,
two-attempt quarantine without recutting and zero publication/cursor movement.
Both guarded root runs blocked only their intentional offline self-test; no
provider requests occurred. Receipts are in
`/private/tmp/hymem-runtime-fidelity-v10.cQJixY/` (`f1-related-first.xml`,
`f1-root-first.xml`, `f1-root-acceptance.xml`). These tests establish wiring and
failure containment, not a real-model semantic-accuracy measurement.

- Construct exact visible evidence spans from the already validated in-memory
  covered messages and before/after cursor. Do not reopen full messages or
  parse source authority out of the rendered prompt's text labels.
- Check titles against their cited new spans. Mark preceding context as
  interpretation-only and prior summaries as derived continuity, not evidence.
- Require an exact indexed verdict schema with complete, unique item coverage;
  unsupported, uncertain, malformed, missing or extra verdicts must not approve
  a candidate. A favorable model assertion is still not proof of truth.
- Do not rewrite accepted titles, weaken explicitly supported exclusivity,
  discard individual failed items or silently advance the cursor.
- Add distinct verification failure attribution. Verification failures must
  not halve the input window as if they were evidence of input overflow.
- Bind the new loaded implementation/policy into digest producer identity.
  Keep Phase-1, facts, profile and standalone summary identities unchanged
  unless a genuinely shared implementation requires an explained change.
- Preserve the existing absolute deadline and a finite request/token ceiling;
  update call reservations where a new validation campaign is later prepared.
  Do not rewrite any completed v9 harness or acceptance receipt.

Root checks: negative scope/negation/certainty cases; positively supported
strong claims preserved; exact indexed verdict parsing; source/context boundary
tests; cancellation/timeouts/attempt limits; no staged or published output and
no cursor movement on rejection; successful exact-byte item preservation;
generation identity isolation. Scripted verdict tests establish wiring and
failure behavior, not real-provider semantic accuracy.

### 2. F2 — evidence coverage for every item field

Use a fresh implementation agent only after F1 passes root review.

F2 acceptance: fresh agent `fix_digest_citation_coverage_v10` extended the same
request to episode content/outcomes/entities and all procedure fields. Every
raw procedure entry retains its own citation check before normalization-based
deduplication; a supported duplicate cannot hide an unsupported citation set.
Root reviewed the complete production diff against the accepted F1 snapshot.
Final guarded root run passed 212/212 (F1/F2 controls, independent root controls,
publication, granular episodes, episode persistence and procedures), exit zero,
only the intentional socket self-test blocked. Root's actual extraction controls
confirm exact partial-cursor source spans and citation isolation; real runner
deadline controls confirm late results cannot publish or consume a quality retry.
Receipt: `f2-root-first.xml` in the private v10 evidence directory above.

- Extend the same batch verifier to episode narratives, titles and entities,
  and procedure claims if present. Check against each item's own citations,
  not the union of all source text in the session.
- Support a phrase crossing the old/new boundary only when new text participates
  in that phrase. An independent fact from preceding context needs its own new
  supporting source; a fact elsewhere in the window is not covered by a wrong
  item citation. Do not infer supporting citations merely from proximity.
- Never treat a prior automatic summary as new-item evidence or the known
  existence of a full message as permission to use its unseen suffix.

Root checks: the observed partial horseback-wish citation, an independent trip
  clause, correct additional citations, legitimate boundary-spanning phrases,
  unseen suffixes, role/peer attribution, multi-item citation isolation and
  faithful full-span positives. Same single verifier request, not another pass.

### 3. F3 — summary outcome and sequence preservation

Use a fresh implementation agent only after F2 passes root review.

F3 acceptance: fresh agent `fix_digest_summary_fidelity_v10` added mandatory
summary verification (including empty no-ops), exact final compaction wording,
new-source spans and separately labelled prior continuity. Overlong-prior no-ops
are held before verification, preventing the runner's historical `prior[:500]`
fallback from cutting accepted history. Root reviewed the complete diff against
F2. Guarded core run: 334/334 passed. Guarded integration run: 126 passed and one
old empty-no-op test lacked its new verifier response; that explicit fixture was
corrected and its independent rerun passed. Receipts: `f3-core-first.xml`,
`f3-integration-first.xml`, `f3-fixture-correction.xml`. No provider requests.

- Extend the shared verifier to the final rolling summary and any summary-only
  compaction. Compare with exact new source and prior-summary continuity; do
  not promote the rejected primary output into source truth.
- Preserve asked/answered state, completed versus intended actions, negation,
  material sequence and conditions. Topic names alone do not retain outcomes.
- Permit faithful umbrella compression of incidental examples; do not require
  every proper noun or confuse a topic omission with loss of a minor detail.
- Keep primary episodes/procedures immutable during summary-only repair.
  A repair that loses an important supported outcome is not successful merely
  because it is short. Do not silently truncate or advance with the old summary.

Root checks: supplied versus unanswered recommendations/directions, an actual
  Big-Sur-then-Monterey-style sequence with a missed activity, conditional
  subscription suggestions versus completed signup, category relations,
  acceptable detail compression and unchanged primary items. Verify compaction
  failure attribution and total request budget. No separate summary auditor call.

### 4. F4 — format enforcement without semantic damage

Use a fresh implementation agent only after F3 passes root review.

F4 acceptance: fresh agent `fix_digest_format_fidelity_v10` added separate summary
and episode format verdict families to the same request, and stopped digest
normalization from stripping meaningful quotation marks. Standalone summary
behavior remains unchanged. Root reviewed the diff and exercised format failures,
quoted names, abbreviations, versions, decimals, compaction and actual publication.
The final guarded root run passed 423/423; only the intentional network self-test
was blocked. One existing Starlette/httpx deprecation warning was non-fatal.
Receipt: `f4-root-first.xml`; accepted digest SHA-256:
`62a70e56a434a823a75f56f9251c7353ac1f0cc25fe06ef61e9ab78f2c9a5ee8`.

- Check the final one-sentence/no-enclosing-quotes contract separately from
  factual grounding, using the same shared verification request if the chosen
  architecture supports it. Keep failure types distinguishable.
- Do not split on every period, strip meaningful quotation content, truncate
  text, or conflate abbreviations, initials, decimals and versions with sentence
  boundaries. Do not weaken the frozen criterion after seeing the failing output.

Root checks: the observed sentence-plus-fragment, legitimate Dr./e.g./initials/
  versions/decimals, one-sentence semicolon clauses, harmless punctuation and
  quoted content, and the separate one-to-two-sentence episode policy. No extra
  format-specific provider loop.

## Exit gates and authority

### Additional F5 — remove duplicated source bytes from verifier transport

F5 acceptance: fresh agent `fix_digest_verifier_transport_v10` added the ordered
`source_catalog`, explicit per-item `cited_source_ids` and separately scoped
summary `new_source_ids`. Duplicate/ambiguous records and invalid references fail
closed; existing verdict validation, publication normalization, call accounting
and resource ceilings are unchanged. Agent: 349/349 passed. Root reviewed the
production diff against F4 and independently passed 459/459, including real
12-episode extraction, last-item rejection, genuine cap holds, exact partial
spans, procedure citation isolation, publication and deadline checks. Receipts:
`f5-agent-first.xml`, `f5-root-first.xml`. Only intentional offline guard probes
were blocked. The source-grounded direct reproducer now uses 14,234 JSON
characters (21,791 including the system prompt), retaining all 11,400 source
characters. Accepted digest SHA-256:
`a0ee309f3872c474d2d5aaf67c41075a08269062fa65e6f50e1ad0950f8b72e4`.

Root's offline reproduction used a valid 11,400-character source window and
12 episodes (the shipped granular cap), with only 1,688 characters of primary
JSON. Repeating the same source per item plus summary inflated verifier JSON to
152,791 characters, exceeding the 131,072-character input ceiling. This newly
introduced false hold is not a legitimate input-overflow failure.
An additional control with all twelve episode claims literally present in that
source produced 1,866-character primary JSON and 153,142-character verifier JSON,
confirming the same failure with source-grounded candidates and no provider calls.

After F4 acceptance, use a fresh agent to encode each exact canonical source
record once, with explicit per-item citation references and a separately scoped
summary reference list. Keep the source authority, offsets, attributed boundary
context, strict verdict coverage and resource/call limits unchanged. Do not
truncate, approximate evidence, pool item authority or raise the ceiling to hide
the duplication. Root must verify the 12-episode reproducer, procedure duplicates,
wrong/unseen/uncited references, exact Unicode bytes and unchanged single-call
accounting before accepting the final candidate. This is within the approved
single-pass architecture and offline-only implementation scope.

An additional root integration run passed 365 cases and failed 18 because old
test-only clients had no fidelity response. Explicit synthetic verdict fixtures
were added for these mechanics tests, without changing their assertions or adding
a production bypass. The first receipt (`remaining-integration-first.xml`) is
retained; the corrected integration run passed 403/403, including the additional
aggregation-provenance tests. Receipt: `remaining-integration-corrected.xml`;
only the deliberate offline self-test was blocked, with one non-fatal existing
Starlette/httpx deprecation warning. These fixtures are not model
accuracy measurements. Root compared all production and benchmark bytes against
the accepted v9 manifest: only `hymem/dreaming/digest.py` changed in this turn.

After each implementation: root reviews the diff against the immediate baseline,
reproduces the original defect, runs focused negative and positive controls,
then related integration tests. Only then begin the next issue with a fresh agent.
After all four and the additional F5 repair: run the complete offline suite against a fresh candidate, preserving
existing worktree edits and all prior evidence. Any target gate must use a new
explicitly tracked candidate; never rerun a used launch or relabel old acceptance.

The first local gate was prepared after the preceding passes, at
`/private/tmp/hymem-runtime-fidelity-v10.cQJixY/local-full-candidate-v10-1`.
`verdict.json` exists only after all four exact-inventory shards pass and reconcile;
its absence is not acceptance. The reviewed full-gate helpers remain unchanged
from the earlier campaign, but this candidate, source manifest, collection and
launch receipts are all new. No source edits are allowed during this gate.
Reviewed new source paths are explicitly listed in `reviewed-new-files.json`
in the private v10 directory. Removed prior paths: none.
Helper SHA-256 pins: `full_gate.py`
`22964cd02da91705f12123f24ff1b91e11e2b8ac3186c8a6b87bd767a0ef30d5`;
`offline_worker.py`
`c195650524b83fdefb248bd21524affee82e168e915b22d02532f2d9a572f0f9`.

First complete gate result: 6,972 tests, 6,942 passed, 14 failed, 16 setup errors,
zero skips. All four workers exited; source snapshots were unchanged and each
blocked only its deliberate network self-test. Twenty-one nonpasses were
localhost socket restrictions: separately permitted offline reruns passed all
7 embedding-health tests and all 16 real Honcho SDK contract tests. Do not weaken
or skip them; request loopback permission for their final gate partitions.
Seven other nonpasses involved old successful-digest fixtures without verifier
responses (store attestation, portability, MSC convergence and LME classification).
The two episode-probe failures are F7 below. The LME fixture also exposed a real
failure-report contract defect, F6; correcting only that fixture would hide it.

### F6 — preserve honest mixed LME failure evidence

Fresh agent `audit_mixed_lme_failure_envelope_v10` diagnosed read-only while the
first full gate remained frozen; implementation starts only after its workers
have all exited. `converge_indexing` can legitimately report pending work plus
quarantined/terminal-loss/malformed durable work with `complete=False`. The LME
validator incorrectly requires those failures to have `complete=True`, replacing
the useful `IndexingConvergenceError` with a structural `BenchmarkIntegrityError`
and preventing assignment of the canonical failure summary.

Repair current and legacy validation without forcing completion or healthy
status. Require positive, correctly typed blocker evidence and truthful mechanical
completion, preserving timeout precedence and coverage-retry semantics. Failed
artifacts must remain unscorable, with no reader/judge use or invented receipts.
Root independently verifies the real adapter path plus forged evidence/completion
negatives before the next implementation agent starts.

F6 accepted: root reviewed the complete protocol diff and both wire paths. The
shared mechanical-drainage check accepts honest mixed failures, rejects invented
blockers/completion, and prevents legacy healthy claims from ignoring supplied
digest/fact/profile errors or the provider-budget flag. Agent related gate:
667 passed. Independent root controls: 6 passed (actual fidelity rejection plus
fact quarantine, with and without a bounded cleanup failure). Root final related
gate: 760 passed, only deliberate network self-tests blocked. Receipts:
`f6-agent-related-first.xml`, `f6-root-controls-first.xml`,
`f6-root-acceptance-corrected-command.xml`; an earlier root command named a
nonexistent test file, ran zero tests and is retained separately, not acceptance.
Protocol SHA-256:
`9694381ee434b0394824eb74f765822762dc2e46e5fd7ee6fddc64a31711b304`.
Separately, the remaining successful-digest fixtures were corrected explicitly;
268 related tests passed (`final-fixture-correction.xml`).

### F7 — update the episode probe for multiple digest completions

Use a fresh implementation agent only after F6 acceptance. The probe currently
labels `llm.sent[-1]` as extractor input; after verification this is the verifier
packet, not the primary source prompt. Its simulation has no verifier response,
cost output still assumes one call per session, and its failure-rate denominator
uses completion calls, so extra calls incorrectly dilute session failure rates.

Retain exact primary input/hash separately from per-stage requests and failure
diagnostics; preserve counts even on later exceptions and reused clients. Add
explicit synthetic simulation verdicts, never production auto-approval. Report
two ordinary / at most three logical digest completions (provider retries separate),
and use attempted session digests, not extra verifier calls, for the unchanged
failure threshold. Do not alter completed historical evidence or launch a paid
probe. Root verifies success, primary/compaction/verifier failures, misleading
failure-rate dilution, source hashes and offline cost/simulation paths.

F7 accepted: fresh agent `fix_episode_probe_multicall_v10` implemented versioned
per-stage completion records and explicit synthetic simulation verdicts. Exact
primary source and reply fields remain separate from actual failing-stage evidence;
backend exceptions remain bounded transport failures, not fabricated empty replies.
The unchanged 2% ceiling now uses failed session attempts, including execution
errors, independently of completion count. CLI token declarations match the built-in
transport, and cost output accounts for two/three logical completions. Recorded
request parameters do not prove an arbitrary two-argument backend honored them.
Agent focused gate: 40 passed; related gate: 216 passed. Root reviewed all changed
source and new tests; independently ran 55 passing controls, including 15 additional
root cases with real stores, exact source hashes, reused clients and denominator
inflation up to twelve completions. Only deliberate socket self-tests were blocked.
Receipts: `f7-agent-first.xml`, `f7-agent-related-first.xml`, `f7-root-first.xml`.

### Final fresh offline gate

The corrected source is frozen for
`/private/tmp/hymem-runtime-fidelity-v10.cQJixY/local-full-candidate-v10-2`.
Its own `verdict.json`, emitted only after four successful exact-inventory shards
reconcile, determines acceptance. Until then this document claims focused
acceptance only. The reviewed helpers and their SHA-256 pins above are unchanged.
`reviewed-new-files-v10-2.json` explicitly includes twelve reviewed additions
relative to v9; no prior source paths were removed. Partitions containing the real
localhost embedding/Honcho fixtures require loopback permission, retaining
credential stripping and the offline external-socket guard. Source and this plan
must remain unchanged throughout collection, execution and reconciliation.

Real-provider accuracy, false-rejection rates, cost and LME convergence remain
unmeasured until a separately approved, preregistered paid validation. That run
must retain first draws and failures and use updated bounded accounting; do not
silently enlarge the completed campaign's limits or select only good responses.
Canary, sample smoke and full LME remain separate downstream gates. No production
deployment/restart, live-store migration, Hermes2/3 change or backup deletion is
part of this plan. The existing recurring follow-up remains paused; this local
implementation turn does not authorize resuming its paid or deployment workflow.
