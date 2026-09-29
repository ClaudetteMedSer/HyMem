# Offline summary recovery redesign — September 13, 2026

## Current result — live component test complete; semantic findings remain

Both approved fixes are implemented and independently checked. Summary-only
compaction preserves validated episodes/procedures; durable failure-specific
retry state no longer shrinks source for summary failures, including unexpected
parser/validation exceptions. Strict caps, atomic publication, total-attempt
quarantine and deadline propagation remain enforced.

- **6,424 suite tests passed**, zero failures/errors/skips: four complete,
  disjoint 1,606-test partitions covering all 201 registered test files.
- **120 root-authored independent regression checks passed** on final source.
- Root reconciled every XML testcase, per-file counts and unique identities,
  and rehashed all **492 source files**. The final candidate remains unchanged:
  `b68add8353dd23835f3fa22d1cbcec0f01bcdc3524a404d2f9e4324350e436b0`.
- Every accepted test process exited 0 and reported zero external socket
  attempts. The two real-HTTP test partitions used explicitly permitted
  localhost with external outbound traffic denied. No tests were skipped or
  weakened to resolve the app sandbox's localhost restriction.
- Existing dependency deprecation warnings remain (Starlette/httpx and
  uvicorn/websockets); no unraisable/thread-exception warning was accepted.

Accepted receipts: `full-v2-shard-1-localhost.xml`,
`full-v2-shard-2-localhost.xml`, `full-v2-shard-3.xml`, `full-v2-shard-4.xml`,
`root-final-independent-v2.xml` and `full-v2-verified.json`, under the private
artifact directory below. Earlier interrupted or permission-failing receipts
are historical and **not** part of this accepted set.

**Not deployed; one approved live component campaign completed.** The complete Hermes1-runtime suite also
passed: **6,424 tests, zero failures/errors/skips**, in 6,051.3 seconds. Root
verified exact testcase inventory, all 492 source hashes, unchanged production
source, runtime/dependency hashes, container isolation, exit 0 and no OOM.
The four-case synthetic target rehearsal and its independent replay passed.

Following fresh explicit payload/destination consent, the live campaign finished
all **four source walks on their first primary response**: **4 completions /
4 HTTP attempts**, zero retries, quarantines or live summary compactions.
Reported usage was **7,650 prompt + 1,797 completion = 9,447 tokens**. It emitted
six episodes, four summaries and no procedures; the foreground process exited
0 after 12.668 seconds. Independent replay in a separate network-disabled,
read-only-source container verified exact request bytes, extraction results,
accounting, contiguous source cursors, database integrity/FK0 and no database
writes. Postflight verified unchanged candidate, production source, runtime,
dependencies and immutable benchmark inputs, with client/connections closed.

**Semantic acceptance is withheld.** Root independently checked the separate
agent's assertion-level audit against all four exact requests and responses:

| Case | Summary chars | Remaining finding |
| --- | ---: | --- |
| Travel, blob | 451 | Actual past trip omitted from summary, although retained in an episode |
| Travel, granular | 484 | Actual past trip omitted from both summary and episodes |
| Music, blob | 353 | Podcast/documentary categories blurred; source wording strengthened to unsupported exclusivity |
| Music, granular | 425 | Main category/correction checks passed; minor two-sentence format noncompliance |

Both granular summaries contain two sentences despite the one-sentence prompt,
but obey the enforced 500-character cap. The material findings concern lost
personal experience, category distinctions and source faithfulness, not a new
sentence-count gate. No invented completed activity, unsupported episode
citation, route reversal or technical-procedure scope violation was found.
Original source remains intact; absent digest content is not raw-message loss.

The relevant new material was fully visible, and no response was clipped,
truncated, repaired or retried. The established gap is that structural
validation checks shape, fields, citation membership and length, not semantic
completeness or entailment. The prompt already requests experience/category/
correction retention. Prior-summary anchoring and granular decision/change/
outcome framing are possible contributors, **not isolated causes proved by
four draws**. Unsupported exclusivity is a source-faithfulness finding, not a
verified claim about current real-world platform availability.

This proves bounded component mechanics, **not full LME completion, score
improvement, durable full-dream publication or live repair effectiveness**.
No live repair was needed; repair-path evidence remains offline/synthetic.
No application source changed during this campaign. No production store was
opened, service restarted or automation resumed. The campaign is terminal:
**do not reuse its unused allowance or reroll for cleaner outputs**.

Current accepted live receipts are under the private campaign directory below:
`results/transfer-verified-json-v1/` (20 allow-listed JSON files plus the transfer
manifest), `results/root-postflight.json`, `root-live-audit-v1.json` and
`independent-live-semantic-audit-v1.md`. No database, credential file or raw
process log was returned. The frozen harness summary's `semantic_audit: pending`
is superseded by the separate root audit's `semantic_acceptance: false`, not
rewritten after review. SHA256 bindings:

- Live summary: `f17f3dd8864d7c6c779c3de73b8298a9673c258ba22e653aeae709f4ef2c2780`.
- Mechanical replay: `25daefd541eccf8c84944013ffa5761a5c31ea2cc5d8f16f7fb95aa29efdbbeb`.
- Transfer manifest: `5c00f0ec461cb6c68219073f8c4676d1a4580927cf61e3c737507de80310f131`.
- Independent semantic audit: `6b78946cae4bd64aea017621e41fd61ad7767add6d6613dd396da33e83eee12a`.
- Root live audit: `f4a624150a0b3dc90be6455ec1aac3ab210971c0f2d4605a29c0701b1cb5f1c5`.
- Root postflight: `47259a8f22ef6cbfbba322b0d0e3ab5dfedbd8335c25f1573e54fa0a06ea3b72`.

Next proposed work is an offline, generic user-event retention and category/
correction-preservation design, with source-grounded regression evidence and
independent per-fix verification. Do not special-case these benchmark names,
weaken integrity gates, claim score gains or launch another paid campaign as
part of this completed consent.

Historical launch blocks are retained: two previous platform-review rejections
occurred **before execution**, and checks confirmed no campaign output then.
The user subsequently explicitly approved sending selected LME travel/music
excerpts, prior summaries and extraction prompts to `https://api.deepseek.com`
with `deepseek-v4-flash`, capped at 64 completions / 192 HTTP attempts, including
third-party disclosure and API charges. `explicit-payload-consent.json` records
that new consent; it does not erase the earlier `platform-launch-block.json`.

The old 19-call campaign must not resume. Deployment will invalidate
digest/facts/profile auxiliary generations, and a blind in-place downgrade
with new retry state is unsafe. No SQL schema migration or production-store
operation was performed.

The remaining sections preserve the implementation and verification history;
the final result and accepted receipt set above supersede interim statuses.

## Authority and scope

### Completed approval — one private target campaign

The user explicitly approved the preceding scoped question: privately stage
the verified candidate on Hermes1, run isolated target checks, then one
four-case test against `https://api.deepseek.com` with `deepseek-v4-flash`,
at most **64 completions / 192 HTTP attempts**, and return benchmark-only JSON
to this Mac for independent audit. Fresh explicit payload/destination consent
resolved the subsequent launch block. This superseded the preparation-only paid
pause below for this **new campaign only**, which is now complete. No production changes, service
restarts, automatic rerolls, old-campaign resume, or automation resumption.

Private local work directory:
`/private/tmp/hymem-digest-redesign-campaign-20260913.hpwzt5/`.
Private target snapshot:
`/home/node/.hermes/benchmarks/lme-digest-redesign-20260913-02829h0t`
(host prefix `/opt/stacks/hermes/instance1/home`). All 492 source hashes match
the accepted `b68add…` manifest; staging rechecked production source hashes
before/after and never opened production memory. The owner account performed
staging; an initial noninteractive sudo attempt did not execute and made no
changes. Existing campaigns and private source databases were not overwritten.

Hermes1 runtime observed: Python 3.11.2, SQLite 3.53.4, pytest 9.1.1,
pydantic 2.13.4, openai 2.53.0, httpx 0.28.1. The fresh full suite has launched
once in `hymem-digest-redesign-full-02829h0t`, container
`b943a61d99f260c259805c451dafa2681063578c093aaef7d1ee53104cc9c343`:
two CPUs, 2 GiB memory/no extra swap, read-only source/runtime, network `none`,
no production store or credentials mounted. Local HTTP test fixtures retain
loopback inside this isolated namespace. **Full gate passed**: 6,424 tests,
zero failures/errors/skips, source unchanged, exit 0, no OOM. Test XML SHA256:
`412eb844440f28d6211f9b1c486c1e32c2df4c8438a80d622777a11a5c661fd9`.
Target runtime manifest SHA256:
`746386bd09874e89f87b4873cd044355252a851d35ff22fff3516b84378960f2`.
Root verifier SHA256:
`f1a48838118b04913105b56925db28601e46ff40f2670dee657686b69acb60a0`.
Accepted target JSON copied to local `results/preflight/target-full-verdict.json`;
no production database or credential file was transferred.

Separate agents completed the new harness and independent receipt replay.
Accepted controls so far: harness 42, verifier 106 (including 5 root variant
checks), root independent accounting/replay 30 and launch safeguards 42.
These counts overlap and must not be added as distinct coverage. Root also
accepted 8 adversarial full-suite XML-inventory checks. No application source
changed while the target suite ran.

Target preparation, synthetic rehearsal and independent replay each exited 0
in separate network-disabled containers. Root inspected actual Docker mounts,
limits, credential-environment names and exits, not just launcher intentions.
The rehearsal made 16 synthetic completions over 8 invocations: four rejected
repairs followed by four accepted summary-only repairs, all four source walks
complete, zero provider requests, source unchanged, no database writes.
The maintained-client instrumentation probe also passed without HTTP traffic.

Frozen harness SHA256:
`b29e9549fff44f2d13fb3990e450bb496ddc39ed4f8e22b7d60966f0de0e88ed`.
Independent verifier SHA256:
`68fc13ccecb07e33701be8f37c87d43984238b8472d3f295ad5cfbbb9a938713`.
New plan SHA256:
`89070b86d5db33cbf83f1459ce203c37c08fac0c220374e3285d2653edf46365`.
Target rehearsal summary SHA256:
`f3585c055c791ada9bfdbd2a00e9737474a38737937a2e99590e81dffd126929`.
Mechanical replay is explicitly not semantic acceptance or LME readiness.

Root/tool reviews caught and corrected validation-tool problems: optimization
could bypass assertion-based checks; quoted-summary normalization was stricter
in replay than in the application; exceptional synthetic fixtures did not
initially exercise their advertised paths. Failed intermediate XML remains
preserved and excluded from accepted coverage. Launch review additionally
required binding the exact observed Hermes1 container and host home mount,
plus validating authorization fully before publishing its final receipt or
importing unverified helpers. The authorizer correction is independently
accepted: final combined authorization/launch gate has 75 passing tests (42
launch, 28 authorizer, 5 parent-authored publication-boundary checks), including
fresh-process positive validation, missing client-probe/runtime evidence,
pre-import sentinels, changed evidence during validation and control-signal
cleanup. No final authorization survives a failed validation.

Frozen authorizer SHA256:
`182ccb14aae29b532c661d64f731d381ffcd91cd4fed2819718a2496f38203a0`.
Launcher SHA256:
`47f812211cae203fd018af098ee8474e99aec5da450b9933346a158d4350af19`.
Parent helper-review JSON SHA256:
`656af1244a437d61b98a1205575e4268559d23063a86f1ab0f5c81f199d79930`.
All are staged privately; this helper review alone cannot authorize calls.

**All offline gates and live mechanical replay passed; semantic findings remain.**
Neither historical rejected launch request executed. After fresh explicit
consent, the unchanged single-launch safeguards allowed one campaign, completed
in four calls under the unchanged 64/192 limits and semantic checklist. Root
rechecked postflight hashes and the exact returned JSON inventory, and verified
the independent semantic findings described above. No production changes,
service restarts, automatic rerolls or automation resumption occurred. Do not
rerun the launcher, preparation or exclusive authorizer: this campaign is
terminal. Earlier preparation-only and pending statuses below are historical.

### Historical preparation-only approval

The user approved implementing the proposed redesign and verifying it offline,
with additional paid calls paused. Preserve all existing dirty fixes and the
terminal September 12 campaign/receipts. No production deployment, service
restart, production-store operation, new benchmark campaign or paid LLM call.
The `finish-lme-validation` follow-up stays paused.

### September 13 continuation — fresh live-test proposal, not authorization

The user's request to continue is being used for offline preparation. Root
reconciled the accepted full-suite receipts and all 492 source hashes again:
the same 6,424 tests and candidate manifest still match. No suite rerun was
needed and no application/test file changed. There has been no remote action,
provider call, deployment, service restart or automation resumption in this
continuation.

The terminal September 12 harness cannot be reused as-is. Root confirmed three
instrument incompatibilities with the verified redesign:

1. Its request validator expects the repair system to append to the primary
   system; the new repair deliberately uses a fresh summary-only system.
2. Its transition logic halves source after every failure; current production
   logic separates total failures from input-related failures.
3. Its preparation and independent verifier require full request equality with
   old prompts, and pin the old 482-file candidate. New prompt/system and
   source identities must be explicitly recorded, not treated as old bytes.

The old campaign, scripts and receipts stay untouched. Separate agent
`prepare_redesign_preflight` implemented a small offline protocol adapter and
controls in `/private/tmp/hymem-digest-redesign-preflight-20260913.UNiBhw/`.
Root reviewed it and independently accepted **117 agent controls plus 19 root
controls**, zero failures/errors/skips and zero external socket attempts. The
root tests compare the adapter against actual extractor requests on synthetic
stores in both modes, including fenced replies, Unicode limits, duplicate/extra
keys, preserved primary items, failure-stage accounting and serialized local
retry state. They also reproduce the historical instrument incompatibilities
without executing its launcher or importing its network/authorization code.

The adapter deliberately returns no source proof or successful cursor; it does
not validate real source stores, enforce paid budgets, or implement publication.
It is not a paid launcher, remote staging operation, complete campaign harness,
or authorization receipt. The old harness and checklist hashes were rechecked
unchanged. Root rehashed all 492 application/test/source files again after the
preflight: the original accepted full-suite manifest still matches.

Accepted new receipts: `agent-protocol-tests.xml` (117),
`root-preflight-controls.xml` (19), and `root-offline-verdict.json` in the new
private directory. Adapter SHA256:
`c9400e894480be2df268ebdcf8d48433ec4d614de0ca14da3a347dd2fce0e686`.
`proposal.json` records the bounds below with `authorized: false`. An initial
root-only test fixture omitted the required config root; that test setup was
corrected and its failed XML retained separately, not counted as an accepted
gate or an application defect.

Proposed next authorized unit of work:

- Freeze a new private candidate on Afrodite/Hermes1 with the exact accepted
  source manifest. Never overlay the production checkout. Preserve all older
  snapshots, receipts and source stores. Verify target Python/SQLite/dependency
  identities rather than assuming the historical versions are still current.
- Integrate the protocol adapter into new versioned preparation, capture,
  replay and launch helpers. Before any paid client exists, verify exact source
  hashes, fresh request hashes, the approved model/endpoint, budget accounting,
  rejection of stale authorization, exclusive single launch, and synthetic
  success/failure/reopen/exception controls on the target runtime. Run a fresh
  isolated target-runtime full suite, with no provider network or credentials;
  permit localhost fixtures only. Any failing preflight stops before spending.
- Run **one** digest-only campaign over the same two benchmark sessions in
  blob and granular modes: four source walks, round-robin. Read the previously
  inventoried benchmark source databases immutable/read-only on Hermes1;
  verify their hashes, no WAL, integrity and foreign keys before and afterward.
  No production memory and no database export to the Mac.
- Preserve the original source starts, prior summaries and incident seed:
  one historical failure counted conservatively as one input failure, initial
  6,000-character window from the 12,000-character configured maximum. This
  explicitly seeded component experiment is not an assertion that a production
  upgrade keeps old-policy retry counts. Only new failures use the current
  failure-stage classifier; success resets both counters. Six total failures
  per held cursor remains the terminal bound; do not raise it or reset cases.
- Proposed total cap: **64 actual completion calls / 192 HTTP attempts**,
  including every summary repair and transport retry, at most two completions
  per invocation. Reserve two completions/six HTTP attempts before each
  invocation; keep its 120-second deadline. Use Hermes1's existing credential
  locally for `https://api.deepseek.com`, model `deepseek-v4-flash`, unchanged
  temperature 0, JSON response format and 3,072 output-token limit. Do not
  print, copy, or embed credentials. These are request caps, not a dollar cap.
- Return only the new benchmark request/response JSON and allow-listed
  accounting/hash/verification metadata to the private Mac task directory for
  root audit. Exclude databases, credentials, production memory and raw process
  logs. Retain all first draws, failures and repairs; no selective rerolls,
  campaign resume, automatic restart or reuse of unused historical allowance.

Acceptance has separate mechanical and semantic decisions. All four walks
must complete within the unchanged limits, with exact contiguous source
coverage, atomic rejection, complete accounting and no source-store writes.
Every accepted repaired digest must preserve its primary's validated episodes
and procedures; the summary must pass the unchanged strict validator. Root
must review all accepted claims and summaries against each invocation's exact
visible source, including personal-trip retention, media categories, platform
corrections, technical-procedure scope and the original topic checklist. Keep
the old checklist immutable, record new prompt/source hashes separately, and
report material omissions or unsupported claims rather than inventing a
post-hoc numeric pass threshold. Synthetic tests prove mechanics only.

This component test cannot certify durable full-dream publication, overall
LME completion, or improved scores. The strict canonical canary, a fresh
sample-10 smoke and then the full canonical run remain downstream gates. None
is launched or newly authorized by this proposal. Production deployment and
service restarts are also excluded. Ask for one explicit approval covering the
new isolated target checks, single bounded provider campaign and benchmark-only
JSON return before doing that remote/paid work; keep the existing follow-up
paused.

Prior full gate: 6,186 tests passed for the September 12 candidate, not this
redesign. Prior live gate failed after 19 calls; it must never restart. Root's
audit is in the private `digest-campaign-results/root-semantic-audit.md` under
`/private/tmp/hymem-lme-repair-20260912.oXxG6N/`.

## Implementation and independent verification sequence

1. **Fix 6: summary-only compaction.** Separate agent
   `fix_digest_summary_compaction` replaces the one full-object reroll with
   one dedicated summary-only request only after all non-summary fields pass
   and the sole failure is summary length. Use the exact original source/prior
   text, a fresh summary-focused system, a 350-character soft target and the
   unchanged 500-code-point hard cap. Validate a strict one-key summary object;
   merge only its nonempty bounded string into the validated primary object,
   then revalidate the assembled digest before any cursor/source proof. Valid
   primary episodes/procedures cannot be regenerated by a length repair.
   Preserve two calls maximum, deadline/accounting, atomic publication and
   explicit failure. Root independently reviews and tests before the next fix.
2. **Fix 7: failure-specific input shrinking.** After fix 6 passes root's gate,
   use a new implementation agent. Retain the total durable failure count and
   quarantine limit; separate failures that warrant input shrinking from
   summary-only failures. Verify reopen, forward/rebuild, malformed/historical
   state, status/export validation and successful-reset behavior. Do not infer
   failure type from prior length alone, reset attempts on failure-type changes,
   or add a silent/in-memory retry state that disappears on reopen.
3. **Combined offline gate.** Run focused cross-component tests and the full
   local test suite on the final source. Verify actual XML counts, skipped and
   failed tests, warnings, source changes and no provider use. A previous
   snapshot's pass does not certify changed code.

## Semantic checks and limits

Prompt guidance must distinguish core personal experiences from generic advice,
preserve media categories during compression, keep technical procedure scope,
and restrict source authority to exact visible spans. Never use a prior summary
or a rejected generated digest as canonical evidence for new episodes.

Offline tests can prove strict shape/length validation, preservation of original
objects, source framing, call bounds and correct atomic rejection. Canned
meaning-preserving responses and prompt-text checks **cannot prove** DeepSeek
will obey semantic guidance or converge in production. Do not introduce a
keyword blacklist masquerading as a semantic validator. Any later live campaign
needs a fresh bounded plan and approval; full LME readiness remains unverified.

The OpenAI Docs skill was used for the general distinction between structured
format correctness and semantic correctness, including simplifying subtasks.
This is not evidence of DeepSeek capability or authorization to switch models.
Primary guidance: https://developers.openai.com/api/docs/guides/structured-outputs

## Verification receipts

Root-only temporary test artifacts:
`/private/tmp/hymem-summary-redesign-20260913.hhJ00r/`.
Results and any discovered regressions will be appended as each gate completes.

### Fix 6 focused acceptance

Root reviewed the implementation and ran 58 independently authored controls;
all passed in 28.02 seconds. They check exact source/prior and request-parameter
preservation, source immutability, Unicode boundaries, meaningful nonempty repair,
invalid primary rejection without a correction, and atomic failure of malformed
repairs. The implementation agent's initial six-file run passed 239 tests; a
fresh 34-test compaction run includes two subsequently added assembled-object
revalidation controls (241 unique current tests covered). Root's broader
nine-file regression remains in progress, with no failure observed yet.

The frozen reviewed implementation hashes are:

- `hymem/dreaming/digest.py`:
  `810c56ca9ebe0b51d10fc11624c72d28704254ed6eda08fe4eb824380808c5d4`
- `hymem/extraction/prompts/__init__.py`:
  `dbe9648a517e608bdeaa884be71a41dda825e715913d9222eba816e5f38d8afb`

Root independently reproduced fix 7 before any retry changes: one failed
summary-only compaction followed by reopen shrank the next request's exact
visible source from 1,675 to 776 characters despite an unchanged cursor. The
expected-failure receipt is `root-retry-before-fix7.xml`. New implementation
agent `fix_digest_failure_adaptation` is investigating the durable-state design;
edits wait for the broader fix-6 run to finish to avoid mixed-source receipts.

The broader nine-file root regression subsequently finished successfully:
303 tests, zero failures/errors/skips. Only the existing Starlette test-client
deprecation warning appeared. Fix 7's separate agent was then authorized to edit.
A further 57 transport/deadline tests passed under a root test launcher that
removes inherited provider configuration/credentials and rejects non-loopback
socket connections in-process. No external socket attempt occurred. The
launcher's explicit rejection self-test passed before use; the normal tool
sandbox remains in force. This is not a claim of native subprocess isolation.

### Fix 7 implementation and focused acceptance

The second agent implemented explicit `primary` / `summary_compaction` failure
provenance, including ordinary completion exceptions with their original cause
retained. Deadline/cancellation `BaseException` signals still escape unchanged.
Primary parsing, truncation, shape and item/cap failures retain input adaptation;
summary validation and every compaction failure retain the same source window.
Every ordinary failure still consumes the existing total-attempt budget.

The existing retry TEXT field now carries the strict leading envelope
`digest-retry-state-v2|input-retries=N|<existing policy key>`. One decoder is
shared by scheduling, quarantine, persistence, health and portability validation.
Historical bare policies retain their old conservative input count. Malformed
count/key state holds without calls or reset; the redundant quarantine flag
remains non-authoritative as required by the existing corruption controls.
Successful progress resets both counts, and a real policy change still opens a
new budget. No SQL schema, export-format version or migration was changed.

Root's final independent checks passed **118 tests**, zero failures/errors/skips,
in 63.34 seconds with zero external socket attempts. The second agent's focused
file passed **175 tests**; imported source-budget checks explicitly assert 900
characters for wrapped mixed history versus 450 for historical bare history.
Root also reviewed preservation of staging/publication and failure provenance.
The broader cross-component regression is still running.

Compatibility consequence: changing the runner changes **all three auxiliary
generation identities** (digest, facts, profile), because their producer identity
already hashes the runner. Phase-1 identity is unchanged by these two fixes.
Do not remove that identity dependency to hide the rebuild cost. No deployment
or rebuild was performed here. Old validators reject the new retry prefix;
an old runner opened directly may instead treat an unfamiliar key as a changed
policy. Therefore a rollback must explicitly account for newly written retry
state: **do not perform a blind in-place code downgrade**.

### Final full-suite gate — running, not yet accepted

Fresh collection registered **6,399 tests in 201 files**. Four local pytest
processes partition the files exactly once (1,600 / 1,600 / 1,600 / 1,599 tests)
under the offline launcher. This is a sharded full suite, not a claim that one
single process exercised every possible cross-file order. Root will reconcile
all XML testcase counts and the file partition, rather than infer success from
progress dots. Unraisable/thread-exception warnings are treated as errors.

The frozen candidate inventory contains 492 regular files. Its canonical JSON
manifest SHA256 is
`84017734d7d43aae753cd624817193f1efc420401d564bb72888180698bc241c`.
Only this evolving verification note is excluded from the inventory. The source
manifest and collection/partition receipt are saved in the private artifact
directory above. No implementation or test edits are allowed during this gate.

### Gate interrupted for a newly reproduced stage-attribution gap

The first full gate above was deliberately stopped and is **not accepted**.
The agent's late audit found that deeply nested repair JSON raises
`RecursionError` outside the completion-only exception wrapper. Root confirmed
the parser failure and independently reproduced it through the real runner and
reopen: source again shrank 1,675 to 776 characters. This violates fix 7's
all-compaction-failures rule even though the first 118 / 175 focused controls
passed. Receipt: `root-parser-stage-before.xml` (expected failing regression).
On local Python 3.11 the JSON recursion limit is reached with a much smaller
response; this is not merely an oversized Python 3.13 test construction.

The same fix-7 agent is completing the boundary so ordinary errors in repair
parsing and assembled-object validation also retain compaction provenance.
Shared JSON parsing and Phase-1 are out of scope; deadlines/cancellation must
still propagate untouched. Root will verify the new regression before freezing
a new manifest and rerunning the entire suite. Earlier partial full-suite XML
and source/partition receipts are preserved, not overwritten or called passes.
Some deliberately interrupted workers emitted pytest teardown errors after
SIGINT; these are interrupted receipts, not completed application-test gates.

### Complete stage-boundary correction and replacement full gate

The agent extended the attribution boundary across the entire compaction task,
including request construction, response parsing and assembled validation. The
original exception cause is retained; already attributed same-stage exceptions
are not wrapped twice. Only ordinary exceptions are handled. There were no
shared JSON-parser, Phase-1 or additional runner edits for this follow-up.

Root's expanded **120 independent tests passed** in 66.30 seconds, including
the actual nested-JSON failure through runner/reopen. The agent's expanded
**200 focused tests passed** in 59.396 seconds. Five extra root checks compiled
the actual stage guard/class under local Python 3.11.3 and confirmed real JSON
recursion attribution, identical propagation of three control-flow exceptions,
and no double wrapping. This is a focused standard-library runtime check, not
a full HyMem test run on Python 3.11 (application dependencies are absent there).
The earlier root cross-component run completed **398/398** before interruption
was requested, but it predates this final stage-boundary completion.

The replacement full collection registers **6,424 tests / 201 files**, split
into four disjoint 1,606-test groups. It is currently running, not yet accepted.
The frozen 492-file manifest is now
`b68add8353dd23835f3fa22d1cbcec0f01bcdc3524a404d2f9e4324350e436b0`.
Only `digest.py` and `test_digest_failure_adaptation.py` differ from the first
inventory. New receipts use the `-v2` / `full-v2-shard-` names so the interrupted
gate remains unmodified. The reconciliation script checks every file hash,
disjoint file coverage, per-file testcase counts, unique XML testcase identities,
and zero failures/errors/skips. No additional paid call or deployment occurred.

### Localhost test-environment correction (no source changes)

Partition 1 reported five failures in `test_ensure_embedding_server.py`.
Root reproduced the first failure before any HyMem call: macOS's app sandbox
rejects `socket.bind(('127.0.0.1', 0))` with `EPERM`. The same setup is shared by
all five failing tests. This is distinct from the in-process external-socket
guard, which recorded zero external attempts.

Using the approved execution escalation, root reran that entire seven-test file
under an explicit macOS sandbox policy that permits localhost while denying
external outbound traffic. All seven passed without code or expectation edits.
The entire affected 1,606-test partition is now being rerun in those conditions;
the first permission-failing partition was stopped and is not accepted. The
other three disjoint partitions continue on the identical frozen source. The
accepted partition-1 filename will be `full-v2-shard-1-localhost.xml`; the
reconciliation script still requires all four complete partitions, every
registered test exactly once, unchanged source and zero failures/errors/skips.
No failure is skipped or reclassified as a passing test.

Partition 2 later reached the other real-socket fixture, the 16 Honcho SDK
contract tests. Root again reproduced `EPERM` at localhost `bind`, before any
application behavior. All **16 passed** under the same loopback-permitted,
external-outbound-denied policy. The entire second partition is being rerun
as `full-v2-shard-2-localhost.xml`, with unchanged code and expectations.
Repository inspection found these two fixtures are the only explicit socket
binds in the suite. The accepted gate will use the complete localhost-permitted
partitions 1/2 and the complete original partitions 3/4, all against the same
manifest. Future full local runs should preflight localhost binding or use this
external-network-denied policy from the start.
