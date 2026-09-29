# Digest fidelity fixes and LME completion

## Objective and current evidence

Active goal: fix observed errors and run the LME benchmark afterward. The prior
goal turn made progress: the bounded live campaign completed and independent
source-level audit exposed remaining semantic defects. It did not complete LME.

The terminal campaign `lme-digest-redesign-20260913-02829h0t` produced four valid
primary digests in four calls, without retries or compaction. Its accepted
source manifest is `b68add8353dd23835f3fa22d1cbcec0f01bcdc3524a404d2f9e4324350e436b0`.
Keep all requests, responses, failures and audit receipts immutable. Unused
allowance is not permission to reopen that campaign. Prior full-suite receipts
prove the old snapshot, not a subsequently edited candidate.

Independent semantic findings and full evidence are recorded in
[the previous report](2026-09-13-summary-recovery-redesign.md). All three material
findings occurred with fully visible source, without truncation or repair.
Original messages remain intact. Structural validity is not semantic fidelity.

## Sequential fixes

Each issue gets a separate implementation agent, followed by root code review
and independent tests before the next agent edits overlapping files. Preserve
existing dirty work; do not commit, deploy, restart services or modify production
memory merely to test a candidate.

1. **S1 — lost user experience.** Replace the granular decision/change/outcome-only
   eligibility rule with one also covering explicitly stated personal events
   and preferences, without fabricating activity. Give rolling summaries an
   explicit allocation order: retain new durable events and corrections, then
   compact older continuity by distinct topics, removing incidental lists
   before newly stated experience. Keep primary modes and recovery aligned.
   Agent: `fix_digest_event_retention`. Status: implemented and root-reviewed;
   offline mechanics accepted, live semantic outcome still pending.
2. **S2 — category loss.** Make typed topic/relation preservation the compression
   unit instead of a mixed list of retained names. Preserve category ownership,
   speaker attribution and the distinction between continuity and new evidence.
   Use generic examples, not benchmark-name special cases. Status: implemented
   and root-reviewed; offline mechanics accepted, live semantic outcome pending.
3. **S3 — unsupported strengthening.** Make source qualifiers, negation and scope
   part of what compression must preserve; an explicit narrower correction
   cannot become a broader assertion. Do not implement keyword blacklists or
   claim that a model's self-check is deterministic entailment validation.
   Status: implemented and root-reviewed; offline mechanics accepted, live
   semantic outcome pending.
4. **S4 — minor sentence-format drift.** Resolve any contradictory wording in
   summary/recovery instructions. Preserve the strict character limit; avoid a
   naive sentence splitter that misclassifies abbreviations and adds retries.
   Status: implemented and root-reviewed; offline mechanics accepted, live
   model-format adherence pending.
5. **S5 — misleading embedding CLI setup guidance.** Root found that the LME
   adapter's `--embeddings` help and default-constant comment still promise
   operation without environment setup, while `resolve_embedding_identity`
   correctly rejects missing deployment revision/tenant. Correct that guidance
   and test it against the fail-closed configuration contract. Do not weaken
   endpoint/identity policy or silently enable embeddings in the historical
   lexical-only baseline. Status: implemented and independently verified;
   guidance now matches actual parsing and fail-closed identity resolution.
6. **S6 — nondeterministic test collection.** Final reconciliation caught an
   unordered set used for parametrization of the eight final-cycle error
   controls. Sort that iterable without changing the production constant or
   assertions, and test real collection under different hash seeds. A fresh
   agent could not be created because of the platform thread limit, so the
   completed S5 agent implemented this separate issue, followed by root review.
   Status: independently verified; the combined local full gate passed.

These are prompt-policy defects and model-fidelity observations, not evidence
that the implementation can deterministically guarantee semantic recall.
Scripted tests prove request contracts, source preservation, validation,
identity and bounded recovery only. Live comparison must verify actual output.

## Validation path to the requested end state

1. Root-authored per-fix controls plus existing focused regressions, including
   immutable episodes/procedures on summary repair, atomic cursor advancement,
   citation rejection, Unicode limits, retry budgets and unchanged Phase-1
   identity. Record source hashes and exact test receipts. No external provider
   access in offline tests.
2. Freeze the combined candidate and run the full suite. Independently reconcile
   collected test identities, XML results, source/runtime hashes and process
   exit; do not combine overlapping selections into an inflated test count.
3. Fresh bounded live fidelity comparison, using unchanged source/checklist and
   all first draws. Retain category loss, omissions and unsupported assertions
   as findings, not post-hoc exceptions. Observe current authorization boundaries
   for paid calls and data transfer; no old campaign resume or selective rerolls.
   An incident-only component test is not a generalization or LME score claim.
4. Run the strict canonical canary and sample-10 smoke against the verified
   candidate with pinned data/model/configuration. Verify full-dream convergence,
   publication, cleanup, accounting and output integrity, not just API return
   codes. Diagnose and fix any new failure before retrying on a separately
   identified candidate/run. Do not waive quarantine or coverage errors.
5. Run the full canonical LME-500 after its gates pass. Keep benchmark source,
   reader/judge/pipeline identities and official-vs-local scoring distinction
   explicit. Verify all expected question IDs, clean indexing, final checkpoint,
   prediction/judging completeness, process exit and cleanup. Compare against
   the appropriate baseline only when identities are comparable; report changed
   extraction behavior and any limits on interpretation.

Completion requires the verified full run and resolution of observed failures,
not merely more passing component tests. Keep the goal active while downstream
gates remain unperformed or failed. Production deployment is a distinct action,
not a prerequisite for benchmarking a private candidate.

## Working receipts

New root-only offline artifacts:
`/private/tmp/hymem-digest-fidelity-20260913.xq6FJw/`.

The previous campaign, its live accounting and its semantic findings remain
unchanged under `/private/tmp/hymem-digest-redesign-campaign-20260913.hpwzt5/`.
The existing `finish-lme-validation` automation remains paused until explicitly
reconfigured for an authorized, identified run; it must not revive old work.

### S1 offline acceptance

Root reviewed the complete diff against the prior frozen campaign, confirming
only `hymem/dreaming/digest.py` and `hymem/extraction/prompts/__init__.py` changed
among its 492 files. The new `tests/test_digest_event_retention.py` adds 17
controls, including rejection of a real boundary-only preceding source ID.
Both primary modes and summary-only compaction share one allocation string;
episode eligibility explicitly includes experiences and preferences without
relaxing provenance or requiring a fabricated outcome. Default granularity is
still off. No validator, source framing, database schema, call limit or retry
budget was changed.

Accepted agent selections: 69 focused and 359 broader regression tests. Root's
separate gate passed 272 tests (seven independently authored controls plus
existing summary-contract, publication, lossless-digest and retry tests), zero
failures/errors/skips, 237.764 seconds. Selections overlap and are not additive.
All processes exited 0, with fatal unraisable/thread warnings and zero external
socket attempts. Root reconciled XML counts/identities and unchanged source.
The initial root-only test run used attribute access on a dictionary; its four
test-harness failures are retained in `root-event-retention-v1.xml` and excluded
from accepted results. Corrected verification is `root-event-retention-v2.xml`,
SHA256 `1bfb302c2e0f29d2a30afefd809a887a586b6da18b903d6cf6029174e6d606e2`.

S1 frozen application hashes:

- Digest: `6795d5f20f9271cb33da944e0ea5b37dc4c57686deeed68b570ffb5002134761`.
- Prompts: `7001d130fae57b07d850d4eb6dc6aba808b3a2db0ba5eae21c6a1497dfa1e524`.

These receipts demonstrate the corrected instruction contract and preserved
mechanics, not measured LLM recall. Proceed to the separate S2 implementation
only after the S1 agent and root test processes are terminal; both are now exit 0.

### S2 and S3 offline acceptance

S2 final source passed 84 agent checks and 66 root checks, zero failures/errors/
skips and external socket attempts. Root caught and corrected overly strict
new-item wording that contradicted the allowed bounded split-phrase context;
the final tests exercise real partial-message cursors in both primary and repair
paths. Earlier pre-refinement receipts remain historical, not acceptance for
the final refinement. Final XML hashes: `s2-focused-v2.xml`
`2692e552fa9fc0efa8e29acd14510ee2d32a0edeeeeec99fd45d503158234439`;
`root-s2-regression-v2.xml`
`428babf5f9cb7de38852f9954bcb6742a0cd7912cfc05b20685f2eb05c295ffe`.

S3 shared primary/repair policy preserves uncertainty, conditions, negation,
time, scope and correction direction, while allowing explicitly supported
strong claims. No validator, source template, call budget or default changed.
Agent 96 and root 75 checks passed with fatal unraisable/thread warnings and
zero external attempts; all processes exited 0. Root reconciled XML identities
and source hashes. Initial agent/root receipts retain eight fixture-expectation
failures: the test compared normalized procedures to wire objects containing
chunk IDs. Only that new test expectation was corrected; application bytes did
not change. Accepted receipts: `s3-focused-v2.xml`
`02ed43097c5b1e326fd6435ad7ac084161ebae28f0d16bb4beafd7c9627cf94e`;
`root-s3-v2.xml`
`abf135eda0dea6dba27796a466ff2451bc55ca6ad2e06fa15a13b139fe8d97a1`.
S3 prompts: `e478c8b8e880685651af11883eb51f11e9fe7ed59d414a7663707ce042cfa44b`;
digest: `99be41b731c1aa4f5eb2e0edd4b8a2cc6f2dd4490346ad3ffb42b032ac55f899`.
Selections overlap and are not additive; semantic fidelity remains unproven.

Root fully reviewed the local full-gate helpers and ran 78 synthetic acceptance/
rejection controls, including eight independently authored guard, execution-phase
and XML identity checks: exit 0, no failures/errors/skips, zero external attempts.
`full-gate/root-helper.xml` SHA256:
`146f5d86f6071ee18537cea69a1577ad1c2afaf753bcae652608ae0bd465a08d`.
This tests the harness, not the final application suite; the latter must follow
the combined candidate freeze.

### S4 offline acceptance

Primary and summary-only repair now share an exactly-one-sentence policy for
nonempty rolling summaries, with concise clause joining that preserves category
boundaries and qualifiers. Episode narratives keep their separate 1–2-sentence
contract. There is no punctuation detector or new retry; empty primary summaries
and the strict Unicode length checks are unchanged.

Agent 110 and root 78 checks passed, zero failures/errors/skips, zero external
socket attempts, both exit 0. Root verified identities and frozen bytes:
`s4-focused.xml` SHA256
`54ea85cdd12251eb15dc5c3ee3d8e40443e292868ab019a798f55ab4d3b8c7ad`;
`root-s4.xml` SHA256
`0525aa444aace528fa0fcd0405cb126ddffda686a821866297ac7d7c645ab2a7`.
Prompts: `e6b16a35b621b2dd9751b92a436778bb68aa73422b81db684fa2c4cd551b77d4`;
digest: `0dd779b6318164f3b6c978956a82ba135d78643500150240904e98bddf7c4a81`.
AST comparison to the previous accepted campaign confirmed no function/class
changes in these two modules through S1–S4; the changes are prompt constants and
their imports. This does not claim deterministic model sentence adherence.

Fresh read-only Afrodite metadata check confirmed Hermes1 running with the same
container/image and exact home bind as the previous campaign; runtime and offline
test-dependency paths remain present. No restart, credential-file read, credential
value return or production write was performed by that check.

### S5 and combined candidate

The LME CLI help and default-setting comment now describe the running embedding
service, explicit revision/tenant, automatic dimension pinning, reachable `/v1`
service URL and trusted-internal-HTTP opt-in. No executable policy changed and
the lexical-only default remains off. Agent and root each passed 19 checks
(overlapping selections), zero failures/errors/skips, exit 0 and no external
socket attempts. The tests exercise actual CLI help/parsing and real identity
resolution, including missing attestations, internal HTTP rejection, valid
pinning without a pin environment variable, and stable/tenant-sensitive keys.
Receipts: `s5-focused.xml`
`87363a7af43f0b33837444086a20f3849470c06bdbd44ab00252c5892cce32eb`;
`root-s5.xml`
`839c27b85e787808c49b9dfa4ddfb79c0cc0174a8454d7a45aa22027dbd31bd3`.
Adapter SHA256:
`9d0cfcb53f9f39207f208d76804cd70839bca297dd13a7fcf5fa8aa483132a8d`.

The combined candidate is frozen in `local-full-candidate-v1/source.json`.
Collection completed successfully: 497 source files, 206 test files, 6,496 exact
test identities. The five added test files account for 72 tests beyond the prior
6,424-test snapshot. All 6,496 cases ultimately passed, but final reconciliation
rejected the v1 gate because eight collected cases changed order across
processes. No full-suite acceptance was issued for v1; see S6 below.
Plan SHA256:
`d83f4c5fc0c5412f1ba082d3432e283c8454ba2ebbff617fd74a08281c41419c`.
Do not edit source during the gate; only this excluded status document and
private helper preparation may change. No paid diagnostic calls have been made.

Frozen manifest SHA256:
`21029b944460fe68e63848197699cc50e5ed24ddd4101c5e169e3d504c209586`.
Root comparison to the previous 492-file snapshot found only three changed
existing files (digest, prompts and LME adapter) plus the five reviewed new tests.

### Full-gate environment restriction and target preparation

The initial local groups encounter real-server fixture failures in
`test_honcho_contract.py` (group 4) and `test_ensure_embedding_server.py` (group 3).
Separate minimal reproductions both fail exactly at `socket.bind(127.0.0.1, 0)`
with sandbox `PermissionError: Operation not permitted`; both pass with scoped
localhost permission and the external socket guard unchanged. No application or
test fix is indicated by those reproductions. The unchanged whole groups reran
as `shard-3-localhost` and `shard-4-localhost`; all six launches are now terminal.
The initial five failures and sixteen setup errors are retained as
environment-restricted attempts. The localhost reruns passed every case, but
group 3's wrapper rejected collection ordering; the v1 gate is not accepted.
No attempt has been silently removed.

Root fully reviewed the target offline driver and protocol, then passed 62
synthetic controls (53 agent controls plus nine root controls), no failures/
errors/skips, exit 0, zero external attempts. `target-gate/root-target-helper.xml`
SHA256: `214c48818c198f396e492410d4b27b1172d5077553e6e4e966000e764cb0a7ae`.
Frozen driver: `2cac9cfe74d31bc2e4ddc9e0ca8273b4a0b36a405d5432290a782f730c01766e`;
README: `4d4089830db2456efafb5bb9817c6d9d4dc34bd072b47a56cefe5ee978fbfc25`.
No real candidate has been packaged/staged for the target yet. Actual target
execution must follow accepted local evidence and independent uploaded-helper
hash verification. It uses four child workers after exact collection, under one
network-none, read-only H1-image container with total 2 CPU/2 GiB limits. No
production memory or credentials are mounted. Only the two observed public CA
bundle paths are exempted from the runtime filename filter; candidate and
test-dependency credential filters stay strict.

### S6 acceptance and fresh full gate

The existing protocol test now sorts `_INDEXING_CYCLE_FAILURE_FIELDS` before
parametrization. All eight rejection assertions and the production constant
remain unchanged. Four new controls use real collection subprocesses under
hash seeds 0, 1 and 42, checking both exact field coverage and ordered identity.
Before the fix: three coverage controls passed and the ordering control failed.
After: all 12 agent checks passed. Root then ran the entire protocol-hardening
file plus the new controls: 174 passed, zero failures/errors/skips, exit 0 and
zero external socket attempts. Root XML SHA256:
`277c55f4b12ab9f7e131b3f6591ec13d08d1f6f2e83b56f65b816e7eee54bf84`.

Root also collected the entire suite independently under hash seeds 0, 1 and
999. All three produced the same ordered 6,500 unique identities from 207 test
files. New candidate: 498 source files; manifest
`cbdb49487a9c02b8e5a07a202f6cfd43f40578bfdebd4ced5233be7a93411e15`.
`local-full-candidate-v2` plan SHA256:
`6e2a8c06fe977af2278e729c8a49907e3020587fbc0818a9529be26ee135e6b3`.
All four full shards passed, 1,625 / 1,624 / 1,626 / 1,625 cases, with explicit
localhost permission for the two proven HTTP-fixture shards. All wrappers and
pytest children exited 0; no source edits, failures, errors or skipped cases.
Reconciliation accepted v2, and root separately matched all 6,500 unique XML
identities against the frozen collection and checked their process receipts.
Local verdict SHA256:
`239b1ae47702e42f7f459645694324c37103c8bc6a77c1738d7f4b279175fd06`.
The sole permitted blocked socket event per worker was the intentional no-packet
guard selftest. Dependency deprecation warnings remain visible; fatal thread/
unraisable warnings were not disabled. This acceptance applies only to the
local runtime, not the target runtime or actual model behavior.

The immutable semantic supplement v2 changes only source binding/version and
records the reason; review criteria are unchanged. SHA256:
`0d3adbd114594a734a96cecafd1cf3d7b4495392b804a97adfc65460dbf8ea5e`.
It stays outside model requests and grants no paid-call authority. Both earlier
supplement and rejected full-gate receipts remain unchanged.

### Private diagnostic preparation

The diagnostic now recognizes the real new target/local receipt layout and
source freeze. Source assignment does not mean gate acceptance. Fresh explicit
consent remains required before authorization, credential access or client
creation; no such consent or paid call has occurred during this continuation.
The original four-case protocol, source/checklist bytes, call caps, request
parameters and no-reroll rule remain unchanged.

Root review found a timeout cleanup defect in the private rehearsal launcher:
terminating the Docker CLI could leave its container running. The implementation
agent added unique launch-token ownership checks, bounded stop/kill/wait of the
exact inspected ID, and nonzero failure receipts for timeout/interruption or
unresolved cleanup. Unrelated containers are never touched. Root independently
tested identity rejection, failed waiting and authorization-publication safety.
Across the frozen private helper gates: 95 unique controls passed, zero failures/
errors/skips, zero external attempts, all processes exit 0. XML SHA256 values:
`0f804fef663cfa99a3a793c4b86ec473e322131928e80d0b6993f25bb536a0c5`
(91 tests), `d25a8725f16d5292c44e4439a18ebb81a471b9a170aca9a64bc35431335af10f`
(4 tests). These are synthetic helper checks, not an actual target rehearsal.
No accepted full-gate helper or frozen application source was changed.

### Accepted local gate; target transfer needs explicit approval

Preflight found that Afrodite's host Python has no pytest, and the test-plugin
directory contains only xdist/execnet and gate plugins. The terminal verifier's
former DEPS-only lookup would therefore fail importing `_pytest.junitxml`.
The private target driver now loads the exact byte-verified runtime's pure
Python parser after runtime/target checks, disables bytecode writes and rejects
foreign or unverified cached pytest modules. The normalization and all accepted
application/full-worker bytes remain unchanged. Real read-only host probes
confirmed successful parser import and 64 correctly attributed pytest modules.

Root reviewed the change and independently passed 68 focused controls, including
a host-without-pytest subprocess, immutable runtime files and strict XML display
identity rejection. Zero failures/errors/skips/external attempts; process exit 0.
Root XML SHA256:
`dfe77257e289f93234faff88711525b783ab988c03105899de349c4dd0d10e2a`.
Accepted target driver SHA256:
`f1127d2931015df22566c3a5c868ca56c91fa77452091b73ac7a8f2fdf5ea0cf`.
Earlier target-helper review versions remain historical; current review is
`root-target-helper-review-v3.json` in the private artifact directory.

The verified source-only bundle was created locally at `target-bundle-v1`:
498 source files, 6,500 tests; package SHA256
`1f0ecee544e64c89b26742776aed18bd010b1e569f4379e72b3aa4337f5bc4a2`.
It contains inventoried source/tests, helper scripts and verification receipts,
not memory databases, credentials or raw provider logs.

On 2026-09-13, the approval reviewer rejected the SCP transfer before execution:
general verification permission was not considered explicit authorization to
export this private source/test/helper/receipt payload to Afrodite. Do not retry
through another transport or indirect route. Ask the user for specific approval
to upload this bundle and run isolated, network-disabled target tests. The only
new remote artifact is the empty private upload directory
`/var/tmp/hymem-digest-fidelity-upload-20260913.8jVMVB`; no payload was transferred,
no target stage/container was created and no production change/restart occurred.
All local full-suite/helper test processes are terminal. Target verification,
live semantic comparison, canary, smoke and full LME remain incomplete. Fresh
paid diagnostic consent remains a separate pending requirement; no paid calls
were made. Keep the goal active; do not claim completion or bypass the rejection.

### Approved transfer and target run

The user subsequently answered **Yes** to the explicit request to upload the
private source, tests and validation receipts to Afrodite for isolated,
network-disabled testing, excluding memory databases and credentials. This
supersedes the transfer blocker only; it does not approve paid model calls,
production changes or service restarts.

Root revalidated the current source and accepted local bundle, confirmed the
existing upload directory was empty/private, transferred the exact eight-file
bundle, and independently authenticated uploaded and staged helper/package
hashes before executing them. New private stage:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-digest-fidelity-20260913-gm8wm5nn`.
Package SHA256 remains
`1f0ecee544e64c89b26742776aed18bd010b1e569f4379e72b3aa4337f5bc4a2`.

The launch process exited 0. Actual isolated container:
`hymem-fidelity-full-gm8wm5nn`, ID
`076d3f9894a3a3264661c88c7aaac67f3462de5d596f8db405afdc4aacf329fe`.
It launched the 6,500-test frozen inventory with network none, read-only
source/runtime, UID 1000, and total 2 CPU/2 GiB limits. No live memory store or
credential mounts, production deployment or service restarts. Target acceptance
still requires observed terminal execution and independent reconciliation.

### Target failure retained; deterministic test-clock correction

The target run is now stopped and **not accepted**. Shard 1 recorded failures in
`test_transient_aggregation_failure_heals_during_convergence` and
`test_permanent_aggregation_failure_exhausts_convergence_cap`. Both state/cycle
tests used a real ten-second deadline; constrained target setup exhausted it,
correctly producing `timeout_during_cycle` before their intended assertions.
The permanent-failure test expected `max_cycles_exhausted` instead.

Root verified the exact container identity/ownership and sent SIGINT to this
isolated test container only. Terminal exit 130, not OOM; the supervisor retained
`KeyboardInterrupt` and four terminated-worker receipts. No target acceptance
was produced. Independent terminal audit confirms source, runtime and production
source unchanged, network none, no production store access or provider calls.
Private receipt: `failed-target-audit-v1.json`. Failed stage/receipts remain intact.

A separate implementation agent reproduced both failures with a controlled
default clock, without sleeps or changing production deadlines. It injected the
existing explicit clock seam into these two state tests and the analogous
MSC/LoCoMo exact-budget drain test. New real-HyMem controls still verify setup
expiry, interrupted-cycle reporting, released lock and terminal telemetry at
cycle caps 1 and 3. Agent verification: 96 passed, zero failures/errors/skips or
external socket attempts. Root review/testing is in progress. This changes the
test inventory; the historical accepted local 6,500-test candidate cannot be
relabeled as acceptance of the next source freeze. Fresh local/target gates are
required before live semantic validation or LME.

Root independently accepted the clock correction: all 96 focused tests pass
(XML SHA256 `02ed16ac53f4eda13f186a601dc2b4a47cf0b1852d2473944f2f392582725049`),
plus both delayed-default-clock reproductions pass after correction. The first
root probe invocation omitted its plugin search path and failed before collection;
the corrected invocation used an explicit private path, no source change or skip.
Root AST-scanned the remaining un-injected `converge_indexing` calls: all are
stub-only unit tests, not additional real-HyMem setup tests. Production deadline
code and existing deadline tests remain unchanged.

A new agent corrected the remaining six-line internal embedding comment claiming
zero environment setup. Root independently verified exact executable AST equality
and 14 passing embedding-guidance cases, zero external attempts. Final adapter
SHA256 `d7e18d9e24915c8632ff32c77c6e9f022c3de8b8740713ff46fbaff896664d4e`.
These accepted corrections are ready for a new source freeze.

### Candidate v3 and downstream completion criteria

Frozen candidate v3: manifest
`f872e60741bcc5ff3a7ff537c27010945273b25a1b9a9c3fc9467cf2c2b845d4`,
498 files / 207 test files / 6,502 collected tests. Local plan SHA256
`99a9e36bf375a9925c9748be932384e5b3a3af06e7af63e9ed5c18350e6de8d0`.
Four fresh local shards were launched; no previous verdict is reused. Root checked
that only the three accepted files differ from candidate v2.

Private plan-only LME helper: root reviewed and independently passed 51 controls,
including smoke/full argv through the actual CLI parser and missing/failed target
rejection. It never reads credentials, launches work or grants authority. Updated
private diagnostic v3 has unchanged semantic criteria/protocol and new candidate
bindings; root independently passed 95 controls, zero failures/errors/skips or
external attempts. These preparations are not live semantic or full-suite passes.

A separate read-only audit confirmed that the maintained strict-artifact validator
validates both successful and honestly reported failed evidence. Therefore final
LME acceptance must additionally require this fresh launch's exact ordered IDs,
all N question entries completed once, zero terminal benchmark/judge/row-diagnostic/
lifecycle errors, empty top-level diagnostic and instrumentation errors, healthy
completed indexing and successful publication/cleanup. Ordinary wrong answers and
fully accounted bounded retries are not infrastructure failures; earlier reports
must remain visible. A persistent checkpoint lock file is expected—verify its
lease is released, never delete it merely because it exists. Canary usage is
separately metered; do not report pipeline-only totals as all-process spend or
invent token/cost precision when failed-attempt usage is unavailable. Preserve
the provisional local-judge label; full coverage alone is not official scoring.

### Candidate v3 local full gate accepted

All four actual local processes exited 0. Independent exact-node XML/worker/exit
reconciliation accepted **6,502 passed, zero failures/errors/skips**, unchanged
source. Selected shard times: 584.550, 775.848, 1006.626 and 812.460 seconds.
Local verdict SHA256
`5bd83ed91b2718e2b46f9d1c1fdb66a24b4e3687226aa4cce184e241f69751b8`.
Root audit: `root-local-full-v3-audit.json`. An initial read-only root audit used
the wrong XML property name and stopped; correcting it to the maintained
`full_gate_nodeid` allowed reconciliation without rerunning tests or modifying
receipts. The accepted helper's own independent display-identity checks passed.

The user-approved source-only target upload is being prepared in new private
directory `/var/tmp/hymem-digest-fidelity-upload-20260913.yyE8b7AZ`; root confirmed
it is canonical, empty and mode 0700 before transfer. No target acceptance or
paid diagnostic/LME result is claimed. Historical failed target stage is intact.

### Candidate v3 isolated target run active

Root authenticated and uploaded the exact eight-file source-only package, then
independently rechecked all staged source and helper bytes before launch.
Package SHA256:
`8163a6b710b788ed740cc52018b217d90095cdf11237d01e4ac7c8b78b2f2ed1`.
New target root:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-digest-fidelity-20260913-7zufzi7o`.
Actual container: `hymem-fidelity-full-7zufzi7o`, ID
`306ab0679f99e694b581fbb8201e2d63bc8ee9e632afa731d1b55b85afbe1966`.
Launch process exited 0; root's authenticated follow-up verified it is running
with the prescribed network-none/read-only/2-CPU/2-GiB isolation. Target collection
completed with process exit 0 and exactly the same 6,502 ordered node IDs. Runtime
receipt is present. Full target execution and terminal reconciliation remain
pending; do not call this a target pass or LME result.

Local full-suite and private helper test processes are terminal. Only the isolated
target suite remains active. Production code/stores were not deployed, migrated
or restarted. No credentials were transferred or paid calls made. The existing
paid-run automation remains paused; no stale authorization was revived. The
separately asked four-case paid DeepSeek diagnostic still requires a fresh explicit
reply, with its 64-completion/192-HTTP cap, benchmark-only JSON return and no-reroll
rule; upload approval does not supply that consent.

Next: observe this exact container's terminal state, then use the authenticated
staged `target_gate.py verify` once for terminal reconciliation (never overwrite
receipts or reuse the stage). Retain/diagnose any failure before proceeding. Only
after target acceptance and separately approved diagnostic scope may the reviewed
v3 live preparation run its network-none rehearsal and bounded paid campaign.

### Fresh diagnostic approval received

The user now explicitly replied **Yes!** to the final question approving the four
travel/music cases and revised prompts to DeepSeek Flash after the server suite
passes, capped at 64 completions / 192 HTTP attempts, with charges and benchmark-
only JSON return. This resolves the pending diagnostic-consent question for the
current frozen candidate and one campaign; do not ask again for that same scope.
Root recorded the exact question/reply and bindings in
`user-approval-20260913-v3.json`. This is an approval record, not a fabricated
prepared-plan or passed-test receipt. Formal harness consent must still bind the
actual prepared plan after accepted target verification and offline rehearsal.
No paid calls, production changes, restarts, new campaign rerolls or automatic
smoke/full launches are authorized by a test pass alone.

Current initial check: exact target container still running, no recorded failures,
no OOM/restart and no target verdict yet. A scoped current-task follow-up is being
configured to wait for this run and then perform the approved bounded diagnostic;
it replaces the old paused campaign prompt, never resumes that historical run.

Follow-up `finish-lme-validation` is now **ACTIVE**, in this existing task at its
preserved ten-minute cadence. Root reread its saved configuration and confirmed
the exact new prompt, status and target task. It performs quiet state checks;
actual target acceptance and offline rehearsal are mandatory before the one
newly approved diagnostic. It pauses on terminal diagnostic/audit completion or
an actionable blocking failure. It explicitly forbids historical campaign resume,
silent rerolls, automatic smoke/full launches and production changes. This
current scope replaces—rather than reuses—the obsolete paused authorization.
Keep the local machine on and app running for scheduled follow-ups; this is a
local task, not an always-on cloud monitor.

### Benchmark identity and comparison limits (unchanged)

The historical private launcher uses `--no-prereg`, with DeepSeek Flash as both
reader and local judge. The current adapter marks such a run provisional; its
`resolve_prereg` requires a clean git tree and a committed in-repository spec
for a canonical claim. A complete 500-question run under that historical
launcher is therefore a strict local comparison, **not** an officially judged
or canonically preregistered score. Do not silently relabel it. Preserve the
reader/judge/configuration when comparing the historical local baseline; any
separately requested canonical/official claim must satisfy its own provenance
and judge requirements before launch. Current fixes are still uncommitted.

### Candidate v3 target rejected; clean-checkout CLI repair

The later target run recorded four real CLI import failures: the strictness
smoke and the three LME arm-evidence subprocess tests. Running a script from
`benchmarks/` placed that directory, but not its checkout parent, on `sys.path`.
The shared helper's new HyMem imports therefore failed with `ModuleNotFoundError`.
The local editable installation had masked this portability bug.

Root stopped only the identified isolated test container with SIGINT: terminal
exit 130, no OOM. The authenticated terminal verifier ran once and correctly
rejected the nonzero exit; no target verdict exists. A separate read-only audit
confirmed unchanged candidate/runtime/production-source bytes, network none,
four terminated worker receipts and supervisor `KeyboardInterrupt`. The failed
stage and logs remain intact. This is not a passed server suite.

The scheduled follow-up is now **PAUSED**, verified by rereading its saved
configuration. The earlier ACTIVE/running statements above are historical.
The user's approval of the one as-yet-unrun four-case diagnostic is retained;
no provider calls have been made. New source must pass fresh gates and bind the
actual approved request scope before any paid launch. No production deployment,
store access or service restart occurred.

A separate agent added a guarded checkout-root import bootstrap to
`benchmarks/strictness.py` and `benchmarks/msc_registry.py`, plus 23 real
subprocess regressions. Tests use copied source and `-E -s -S` to exclude
editable/site/PYTHONPATH discovery, cover absolute/relative/module invocations,
and verify canonical deadline class identities. Before: ten import failures.
After: agent and independent root focused runs each passed 165 tests, zero
failures/errors/skips and external socket attempts, fatal thread/unraisable
warnings enabled. Root XML SHA256:
`305f23d17c2637410321ea4ddd5ab7bb56b6324cda699151081179ed975275bf`.
Root reviewed the 13-line application diff: no deadline, extraction, validation
or recovery behavior change. These overlapping selections are not additive.

The investigation also exposed a distinct BEAM module-entrypoint sibling import
failure. Root reproduced `python -m benchmarks.beam_adapter --help` failing on
`longmemeval_adapter`, and assigned a new implementation agent after accepting
the first repair. That repair and root review are pending. Freeze the next
candidate only after both issues are independently accepted; never reuse v3's
local acceptance for changed source.

Root has now independently accepted the BEAM repair: one canonical package
import, four clean-subprocess controls, and 351 passing broader BEAM/import/
deadline tests. Zero failures/errors/skips, exit 0, fatal thread/unraisable
warnings and zero external attempts. Root XML SHA256:
`a62ae98f288a2f80611745143a4eede4945cdb96f51e5f4ba6b6232733b8ea85`.
Actual direct and module CLIs both exit 0 with byte-identical help. Tests also
verify canonical helper identities, honest missing-dependency errors and BEAM
fingerprint sensitivity to imported source, not unused appended helpers.
The code fingerprint changes as expected; official judge text does not.

Root compared all prior frozen files: only the three benchmark import files
changed, plus the two reviewed new CLI test files. Every HyMem application file
and the LME adapter remain byte-identical to v3. Both implementation agents and
all focused test processes are terminal. A fresh v4 full gate is being prepared.

### Candidate v4 full gate and unchanged diagnostic scope

Fresh collection completed: 500 source files, 209 test files, **6,529** exact
nodes. Manifest:
`4aad4da9614807da037d51ab29f7e2ccc40136914584f28782f51722e7dc5beb`.
Plan SHA256:
`fede200fab8b1f4704350accb6712cb0312e41315decee2aad48de7938d9e8b3`.
Four actual local groups are running as `shard-1-final` through `shard-4-final`,
with 1,633 / 1,632 / 1,632 / 1,632 tests and scoped localhost permission for
groups 2 and 4. No failures recorded at the handoff check; terminal acceptance
is pending. Do not edit frozen source or relabel older accepted results.

Private diagnostic preparation `live-diagnostic-v4/` changes only three binding
assignments, their comment, README and supplement metadata. Root verified AST
equivalence after masking those assignments, identical semantic criteria, and
unchanged bytes for every other helper/protocol/checklist. All 26 historical
helper files and the original approval record remain intact. The 95 independent
offline helper controls passed, exit 0, zero failures/errors/skips or external
attempts; root XML SHA256:
`511e737c5cf02d105bcd0591b3a251bc201f96d17fd8da2f169e879b97ee4dd6`.
Supplement SHA256:
`98ead286f60c595190ea001926dabf5d656a858ca4feff36c208757745d6dce0`.

The user's actual approval concerns the one as-yet-unrun four-case diagnostic.
Inputs, extraction/repair prompts, endpoint/model, request parameters, 64/192
budgets and benchmark-JSON return remain unchanged. Only benchmark CLI import
plumbing and tests changed the source fingerprint. Rebinding this same unrun
experiment is not permission for a second campaign; the original approval
record is not rewritten. Formal harness consent must still bind the actual new
stage and prepared plan after accepted full suites and network-none rehearsal.
Evidence: `live-diagnostic-v4-rebind-evidence.json`. No paid calls have occurred.

Exact process handles, helper hashes and continuation boundaries are in
`candidate-v4-handoff.json`. The existing paused follow-up is being retargeted to
observe these four local processes, reconcile them if successful, then perform
the approved source-only isolated Afrodite validation and the one approved
diagnostic. It must pause on a blocking failure or diagnostic/audit completion;
no automatic smoke/full LME, production deployment or restart is included.

### Automation boundary: local-only follow-up active

The approval reviewer rejected re-enabling the wider scheduled continuation:
recurring source uploads to Afrodite and paid DeepSeek execution were not
considered explicitly authorized for that automation. The rejected update did
not execute; the saved automation was still PAUSED afterward. Do not bypass
the rejection through manual/indirect external execution or an old prompt.
Specific user approval for those scheduled external/paid actions is required.
The original one-diagnostic approval remains recorded, not replaced or consumed.

A safer **local-only** update succeeded. Root reread the saved configuration and
confirmed exact prompt, ACTIVE status and this task ID. It only observes the
already-running four local groups, reconciles successful local results once,
updates local evidence/status, reports completion/failure and pauses. It
explicitly forbids network/SSH/uploads/remote work/provider calls/credentials/
deployment/restarts and any automatic test restart. No new target stage or paid
campaign was created. At the final local snapshot all four groups were still
running with no recorded failures, source hashes unchanged and no verdict yet.

The next external step is blocked on explicit scheduled-action approval, not
on a presumed code/test pass. A failed status-document patch used a misspelled
path and made no changes; this corrected update records the actual state.

### Fresh explicit scheduled-action approval

The user answered **Yes.** to the exact question approving this scheduled
follow-up to upload the fixes to Afrodite, run isolated server tests, then run
the four-case DeepSeek Flash diagnostic within 64 completions / 192 HTTP
attempts and return benchmark-only JSON. This resolves the preceding scheduled
scope rejection. The immutable actual question/reply and limits are recorded
in `user-scheduled-approval-20260913-v4.json`; the original diagnostic approval
remains untouched. No duplicate campaign, production deployment/restart or
automatic smoke/full LME is included.

The tool accepted the wider scoped update. Root reread the saved automation and
verified its exact approved prompt, ACTIVE status and task ID. It retains all
local/target/rehearsal verification gates and pauses after the one diagnostic
and audit, or a blocking failure. The earlier local-only restriction above is
historical, superseded by this fresh explicit approval and accepted update.

All four actual local v4 processes are now terminal exit 0. The maintained
reconciler accepted 6,529 tests, zero failures/errors/skips and unchanged source;
root's separate exact-node/receipt audit follows before packaging. No paid calls
have occurred, and no target-v4 stage has yet been created.

### Local v4 accepted; exact full-payload upload approval required

Root separately reconciled all 6,529 exact XML identities, display identities,
setup/call/teardown phases, successful parent/worker exits, source snapshots,
runtime bindings and disjoint coverage. Zero failures/errors/skips. Local
verdict SHA256:
`357790bec6afcb8fcca301844685065f41130974234bffb73262fbc65fcb543b`.
Root audit: `root-local-full-v4-audit.json`. An initial read-only audit assumed
the guard selftest was a boolean; correcting that assertion to the maintained
success dictionary required no test rerun or receipt/source change.

The source-only eight-file package is complete locally (`target-bundle-v4`),
5,413,789 bytes, package SHA256:
`7b1686c7931ac78a5167cc9be0d656800272ea4f426243cd8d0bc95e207b9bf7`.
Its archive contains the full 500-file source/test snapshot. The new Afrodite
upload directory `/var/tmp/hymem-digest-fidelity-upload-20260913.45LHKvgu` was
created and verified empty/private/canonical. The SCP was then **rejected before
process creation**: the reviewer did not consider approval to upload “the fixes”
clear authorization for the complete private source/test snapshot, validation
metadata/helpers and exact destination. No payload was transferred or target
stage/container created. Do not retry through another transport or assemble
an indirect transfer to bypass this rejection.

The wider scheduled-action approval remains recorded and its automation update
did succeed, but this later exact-payload rejection is a distinct blocking
condition. Root paused the follow-up, reread the saved configuration and verified
its exact PAUSED prompt. No provider calls or production changes occurred.

`upload-scope-v4.md` inventories the exact eight-file full-source bundle and
18 reviewed diagnostic helper files, with sizes, hashes, contents and Afrodite
destinations. Ask explicitly for approval of this complete private payload,
not merely changed files. The diagnostic cap remains 64 completions / 192 HTTP
attempts, one unrun campaign, with all existing prerequisite checks. All local
test/helper/package processes are terminal; external work awaits this approval.

### Exact-payload approval received; v4 upload and staging completed

The user explicitly answered **Yes!** to uploading the complete private
500-file source/test snapshot, verification metadata and diagnostic helpers
listed in `upload-scope-v4.md`, to its listed isolated Afrodite destinations.
Root recorded the genuine question/reply and inventory binding in
`user-payload-approval-20260913-v4.json`. Inventory SHA256:
`85c55c380dd40c5bd035174d64b628f01e3b23bc5f0654d74d492304d8c7b3af`.
This resolves the exact-payload rejection only; all execution gates and the
existing single-campaign/cap/production boundaries remain unchanged.

Root rechecked unchanged frozen local source/helpers and the still-empty private
upload directory. The approved eight-file SCP **succeeded**, process exit 0;
root independently authenticated all eight uploaded hashes. The authenticated
staging helper then succeeded, creating only:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-digest-fidelity-20260913-8r4vcxge`.
All 500 staged source files and helpers were independently reauthenticated
before launch. Package SHA remains
`7b1686c7931ac78a5167cc9be0d656800272ea4f426243cd8d0bc95e207b9bf7`.
No credential files, production store or raw provider logs were copied.
Diagnostic helpers remain local until the server suite is accepted.

One isolated v4 server launch is now being observed. Never stage or launch a
duplicate, overwrite receipts or edit frozen candidate/bundle bytes. No paid
diagnostic call, production deployment or service restart has occurred.

The launch completed with process exit 0. Actual new container:
`hymem-fidelity-full-8r4vcxge`, ID
`f33a9d2144d10cae8c0a2eac19c5d011237b006e937f23d3d70813e798737019`.
Started 2026-09-13 at 14:23:03 UTC. Root independently verified its actual
network-none/read-only/2-CPU/2-GiB/UID-1000 boundaries and absence of production
store/credential mounts. Target collection exited 0 with the exact same ordered
6,529 nodes; target runtime metadata passed identity checks. No failures recorded
at startup, but the suite is still running and has no target verdict or terminal
verification receipts. This is not yet a server-suite or LME pass.

Current exact handoff: `target-v4-launch-handoff.json`. All old blocked/local-only
handoffs remain historical. The existing follow-up is being updated to this
identified running container and the fresh payload-specific approval; it must
not upload/stage/launch the v4 source again or repeat local reconciliation.

The follow-up update succeeded and is now **ACTIVE** at its preserved ten-minute
cadence in this task. Root reread the saved configuration and verified the exact
new prompt/status/task. It observes only container `f33a9d21...`, authenticates
and reconciles terminal server evidence once, and only after acceptance proceeds
to the approved 18-file diagnostic helper upload, network-none rehearsal and
one bounded diagnostic plus independent audit. Both actual scheduled-action
and exact-payload approvals are explicit in its prompt. It pauses on completion
or a blocking failure; no duplicate upload/launch, production change or automatic
smoke/full LME is included. The historical PAUSED/payload-blocked states above
are superseded, not instructions to repeat already-completed work.

### Target v4 failed; embedding concurrency-test repair in progress

The v4 target gate is now **stopped and not accepted**. Its shard 2 reported
`tests/test_embeddings.py::test_chunk_embedding_runs_in_parallel_with_phase1`:
the Phase-1 provider did not observe the embedding worker in flight. Root
authenticated the exact container/operation/isolation and recorded failure
before stopping only that test container with SIGINT. It terminated at
2026-09-13 15:24:00 UTC, exit 130, not OOM-killed. All four worker exit receipts
are retained (exit -9 after controlled supervisor interruption); the supervisor
records `KeyboardInterrupt`, only collection completed, and no acceptance.

The terminal verifier ran **exactly once** and correctly rejected exit 130.
Its exclusive Docker-log/isolation receipts now exist: never rerun that writer,
resume this stage, or overwrite failed evidence. Root separately verified the
candidate, runtime/dependency and production-source bytes unchanged; container
isolation remained valid, with no production-store or credential mounts.
Private evidence: `failed-target-audit-v4.json`. Failure traceback SHA256:
`95c089edaf89c27aade4c9edfc4ea6d65c4540db4b6ab15116711fb46fc331e7`.

The separate implementation agent reproduced the assertion by injecting a
one-time 2.25-second delay before actual Phase-1 extraction. The existing test's
embedding worker expires its own two-second event wait during variable-duration
preflight, even though all five embeddings and ten extraction calls complete.
Root independently inspected the runner: embedding submission precedes Phase 1,
and future draining follows the extraction loop. This establishes a timing bug
in the test, not evidence that production scheduling became serial.

The agent is replacing the timer-spanning-preflight oracle with causal event
coordination and an explicit serial-execution negative control, plus cleanup
checks. Root review and independent delayed-preflight/regression verification
remain pending. Only `tests/test_embeddings.py` is assigned for this repair;
no production request/prompt/accounting behavior is being changed.

Following the official OpenAI scheduled-task documentation, root paused
`finish-lme-validation` at this failure and reread its saved configuration:
exact PAUSED status, updated failure prompt and task ID verified. The previous
ACTIVE/running handoff above is historical. No replacement server run, new
source freeze, diagnostic-helper upload, formal consent, paid diagnostic,
smoke/full LME, production deployment or service restart has occurred. Existing
one-diagnostic approval is unconsumed, not a bypass of the failed server gate.

### Embedding overlap fix independently accepted; local v5 full suite running

The implementation agent completed the test-only repair. Root independently
passed **269 checks**, exit 0 for both actual processes and zero external socket
attempts, with fatal thread/unraisable warnings. The three root controls prove:
the authenticated frozen-v4 test fails under a real 2.5-second preflight delay;
the revised test passes the identical delay; and an in-memory mutation of the
actual inner runner that joins its embedding future before Phase 1 is rejected
by the revised overlap assertion. The other 266 checks cover embedding,
concurrency, deadlines, source health/withdrawal, bootstrap and LME guidance.

The first root serial-mutation probe selected the public wrapper instead of
the inner implementation and failed before injecting a mutation. Its source
and failed XML are preserved; the corrected v2 probe passed without changing
the application or maintained test source. Root evidence and review are in
`/private/tmp/hymem-root-overlap-20260913.f8Sdu3/`. Accepted test-file SHA256:
`3413cb83c3c7aedd7b34c23edac3523a7cb463cd9b36b1f9fc183382cd0f2b54`.
Root XML SHAs: `c31064ac9f9871746f6f0a06769b1c1257ebb28d57d00a314725d2a9da0e891a`
(3 controls), `954fa62dce975191999f2233a3a496caa56f24aa84fa8e418acab1f8fd63d991`
(266 regressions). All other 499 inventoried source files are unchanged.

Fresh local preparation `local-full-candidate-v5` collected **6,532 exact nodes**
from the same 209 test files / 500 source files. The only added nodes are the
three maintained overlap controls; none were removed. Source manifest:
`f21de39038a2b0a70c6cac5b996c02f6b1b3ac278755c2a3c48f275e606b5c8b`.
Plan SHA256: `cc66a4656b34182767eb052c7fd94502b36cca2365140f9f5e377461961d48f9`.
Four new whole-file groups of 1,633 tests are running, exactly once each.
Groups 1 and 3 have the specifically needed localhost-fixture permission;
the reviewed credential stripping and external-socket guard remain active.

Exact process handles and continuation boundaries: `candidate-v5-handoff.json`.
Do not edit frozen source/helpers, poll closed old processes, reuse failed v4
server evidence as acceptance, or reconcile before all current processes exit.
No replacement remote run, new source upload, paid diagnostic or full LME has
started. The follow-up may observe/reconcile this local-only run; external
continuation still needs current source-bound gates and transfer review.

The existing follow-up is now ACTIVE with a strictly **local-only** prompt:
observe these four already-running groups, reconcile/audit once if successful,
report completion/failure and pause. Root reread and verified its exact saved
prompt/status/task; cadence is unchanged. Remote uploads/tests/provider calls
and production operations are explicitly excluded. A separate agent is
preparing only private v5 diagnostic source/count bindings; no acceptance,
consent, upload or live campaign is being created by that preparation.

### Private v5 diagnostic binding independently reviewed

The separate agent completed the narrow private-helper update. Root independently
compared all 17 maintained files: only the README, three compatibility assignments
and supplement metadata differ from v4. Masking those assignments produces an
identical AST; all semantic criteria, request-bearing helpers, protocol, model
settings, caps and no-reroll rules are unchanged. All 26 historical v4 files remain
byte-identical. Root rehashed the 500 source files: v5 still matches its freeze,
with only `tests/test_embeddings.py` different from v4.

Root independently reran **95 helper controls**, actual exit 0, no failures,
errors, skips or external socket attempts, with fatal thread/unraisable warnings.
Root XML SHA256: `9c8215121448a3d5e47c2480e5d0cf7f1d2f51c1583c66908619b4b3aa42cb07`.
Review: `live-diagnostic-v5/root-helper-review.json`. This accepts only helper
code and synthetic controls. The local full suite remains running; replacement
target validation, rehearsal and semantic evidence are still pending. No remote
actions, formal consent, paid calls, deployment or full LME have occurred.

### Local v5 full suite accepted; external replacement awaiting approval

All four actual parent processes exited **0**, with 1,633 tests each. The
maintained reconciler ran exactly once and accepted **6,532 tests / 209 test
files / 500 source files**, zero failures, errors or skips. Root's independent
read-only audit verified the exact collected/executed node partition, all
19,596 setup/call/teardown reports, XML node/display identities, child exits,
source snapshots and current bytes, runtime metadata and offline guard receipts.
There were zero external application socket attempts; each worker's one blocked
attempt was only the explicit no-packet guard selftest. All local session handles
19696, 2818, 65647 and 99865 are closed: do not poll, restart or reconcile again.

Root audit: `root-local-full-v5-audit.json`. Local verdict SHA256:
`306e0c94019b3adfddd11ac1da2e9688c17bd42a597753d1095507cca6fea616`.
Source remains `f21de39038a2b0a70c6cac5b996c02f6b1b3ac278755c2a3c48f275e606b5c8b`.
Group durations were 659.471, 754.963, 942.058 and 840.203 seconds respectively.
This validates the local test-only repair, not target execution or LME quality.

The unchanged authenticated packaging helper completed locally, exit 0.
`target-bundle-v5` contains eight files, **5,415,364 bytes**. Root independently
verified its exact 500 regular archive members and every source/helper/evidence
hash. Package SHA256:
`53142df1c23eba3d5650e1054f49687ca01054fc2a8d73d7ff11603287adbe2b`.
The reviewed diagnostic transfer list is 18 files / **230,711 bytes**; excluded
caches and test XML are not part of the transfer. Root helper-review SHA256:
`11c01b3f77513480913dc46965291b08f1c50ac48b359d27dbb8ee46115d8c96`.
The complete proposed payload, destinations and bounded continuation are listed
in `/private/tmp/hymem-digest-fidelity-20260913.xq6FJw/upload-scope-v5.md`.

Following OpenAI Docs, root paused the existing `finish-lme-validation`
follow-up after local completion, preserving its cadence and task. Root reread
the saved configuration and verified the exact PAUSED status and updated prompt.
The prior ACTIVE/local-running sections are historical. Official reference:
[Scheduled tasks](https://learn.chatgpt.com/docs/automations).


No v5 SSH, upload, server run, rehearsal, formal consent, paid call, production
change or full LME has happened. Exact revised-payload/replacement-validation
approval is now required for the next external sequence. The failed v4 stage is
preserved and must not be reused; the original one-unrun diagnostic allowance
remains unconsumed and cannot bypass the new gates. The broader goal is ongoing.

### User approved revised v5 scope; target full suite running

The user replied **"Proceed."** to the explicit revised Afrodite transfer and
bounded validation request. This is genuine new approval, not an automatic goal
continuation. Recorded in `user-payload-approval-20260913-v5.json`, SHA256
`fe33dd50e396dca38b0f6e33ebad6c5199291dd5a354e43ac2fa0c9679ab67dc`;
it binds the complete `upload-scope-v5.md` payload, destinations and limits.
The approval-pending statements above are now historical. It permits the one
replacement target run and, only after successful gates, the original one-unrun
four-case diagnostic and benchmark-only JSON return. It does not authorize
production, restarts, Hermes2/3, smoke/full LME or additional paid campaigns.

Root reauthenticated every local payload file and the unchanged 500-file source.
Afrodite UID/GID and pinned image matched. The exact eight-file bundle transferred
successfully to the fresh private mode-0700 directory
`/var/tmp/hymem-digest-fidelity-upload-20260913-v5-r5tzhd2n`.
Root independently authenticated all eight uploaded files **before** executing
the staging helper, then authenticated the staged helpers before launching once.

New private target stage:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-digest-fidelity-20260913-ybqmwv0w`.
Container: `hymem-fidelity-full-ybqmwv0w`, ID
`829c6c52731f1c36d7fe94d48f3c95dad9652b819f6a0fa38b302c364c133796`.
Started **2026-09-13 16:05:58 UTC**. The launch command exited 0; its local
session 6584 is closed. Do not relaunch it. Root's fresh read-only Docker check
confirmed the exact container is running, network none, expected read-only
mounts/resources, no production-store/credential mounts and no OOM. Target
runtime identity passed; collection exited 0 and all four shards started.
No recorded failure was present at that observation; this is not acceptance.

The terminal verifier has **not** run for v5; use it once only when this exact
container is terminal, then independently audit the evidence. All failed old
target stages and their already-written terminal receipts remain untouched.
No diagnostic helper upload, formal campaign consent or paid request has yet
occurred. Current continuation: `candidate-v5-target-handoff.json`.

Root resumed the existing `finish-lme-validation` follow-up using OpenAI Docs
and the user's approved bounded sequence. The saved configuration was reread:
exact ACTIVE status/prompt, original cadence and original task were verified.
It observes only this v5 container, verifies terminal evidence once, and proceeds
to the one diagnostic only after target acceptance and rehearsal. It pauses on
completion or a blocking failure and stays quiet on unchanged state. The prior
local-only/PAUSED prompts are superseded; no old failed stage is reactivated.

A second fresh read-only observation confirmed the exact container still
running, not OOM-killed, with no shard failure event or terminal exit yet.
Root independently compared the server's **6,532 collected node IDs** to the
accepted local plan: exact match, successful collection, no deselection/skip
or application socket attempt. The four workers remain in progress; no target
acceptance verdict exists. No paid diagnostic call has started. Production
checkout/deployment and live memory were not modified by this continuation.

### Target v5 failed at a lease-test readiness assertion; follow-up paused

The 17:17 UTC heartbeat found a recorded shard-4 failure in
`tests/test_dream_lease.py::test_forced_takeover_while_provider_blocked_publishes_nothing`.
At line 311, `llm.entered.wait(timeout=5.0)` returned false **before** the test
performed its forced lease takeover. Traceback SHA256:
`98880314736423c28fcc6bc6a97eca70745d38fee53f79038e94c5800e63ac30`.
This proves a test readiness failure, not a production lease-fencing regression.
The same fixture also has a five-second provider-release deadline and closes
SQLite after a bounded thread join; slow preflight and cleanup need reproduction.

Root authenticated the exact live container, isolation and recorded failure,
then stopped **only** that private test container with SIGINT. It terminated
**2026-09-13 17:19:09 UTC**, exit **130**, not OOM-killed. All four worker exit
receipts are retained (exit -9 during supervisor cleanup), only collection
completed, and the supervisor recorded `KeyboardInterrupt`, accepted false.
The exclusive terminal verifier ran **once** and correctly rejected exit 130;
its Docker-log/isolation receipts exist. Never rerun that writer, restart this
stage or relabel its evidence as passing. No target acceptance verdict exists.

Root independently verified unchanged candidate, runtime, dependency and
production-source hashes plus valid terminal isolation. Evidence:
`failed-target-audit-v5.json`. Terminal isolation receipt SHA256:
`f0b11351505f9d3fbdd011ae1b8f95024c1702354d1a93491da0ae85ece654eb`;
result receipt SHA256:
`68a6d641b13d60276f806d1d2787e49d77c0926871a881918d14c26d018e2c71`.

Following OpenAI Docs, root paused `finish-lme-validation` and reread the saved
configuration: exact PAUSED status and failure prompt verified; cadence/task
preserved. The ACTIVE/running sections above are historical. No diagnostic
helpers, formal consent or paid requests were used, so the original one-unrun
diagnostic remains unconsumed, but blocked by failed server validation.

Under the user's standing per-issue fix-and-review workflow, a new implementation
agent is reproducing and narrowly repairing the lease-test synchronization
locally. Only `tests/test_dream_lease.py` is assigned; any application defect must
be reported before application edits. Root will independently verify final bytes
and negative controls. No replacement source freeze/upload/server launch, paid
campaign, deployment, restart or smoke/full LME is authorized by that local work.

### Lease-test readiness repair independently accepted locally

A fresh implementation agent repaired only `tests/test_dream_lease.py`; root
reviewed the complete diff against the exact frozen-v5 bytes. A real5.25-second
pre-provider delay reproduces the original line 311 readiness assertion, before
takeover. This establishes a matching fixture failure mechanism, not precise
measurement of provider-entry timing on Afrodite. The retained synthetic test
database was also inspected read-only: its dream ran 17:11:02–17:11:08 with no
terminal error and no publication/lock rows. This does not prove lease fencing
was exercised in the failed target test.

The dream now runs synchronously on its owner thread. A controller waits for
actual provider entry or explicit cancellation, performs takeover/renewal while
the call is demonstrably blocked, owns its SQLite connection and is joined
before test cleanup. The 30-second bound is a post-entry deadlock guard, not an
assumed preflight duration. Dedicated renewal ticks are deferred non-blockingly
until the controller has observed an actually stale lease. All previous lease
loss, chained cause, no-publication, successor, renewal and cleanup assertions
remain. Four permanent controls cover cancellation, missing provider invocation,
controller failure and absence of a preflight timer.

Root's separate verification passed **383 disjoint checks**: nine independently
authored real-delay/mutation/resource-cleanup controls and 374 broader dream,
lease, scheduler, embedding, generation and extraction-retry regressions. Both
actual processes exited 0; XML has zero failures, errors or skips, with fatal
unraisable/thread warnings and zero external socket attempts. Mutating real
fencing away fails the missing-lease-loss assertion; making real renewal a no-op
permits actual stale-lock takeover and fails the renewal assertion. Injected
early/controller failures leave all captured threads stopped and SQLite
connections closed. Agent's 26 overlapping checks are not added to this total.

Root receipt: `/private/tmp/hymem-root-lease-readiness-20260913.IK1P3y/root-review.json`.
Control XML SHA256:
`d4b4c71754d52831380b2b08882821422b801964e3561e7f900eb8873ecca4f8`;
broader XML SHA256:
`1369d3c1415fde7696a4cf274bc79c5be148b696c15fe7bdb73cc7e53236dabf`.
Final test-file SHA256:
`7c284aef8e512a559d1988e816aa9126da62fd48a2ac5ba7a8d7ca8d506a99cd`.
All other 499 frozen files are unchanged. Syntax and focused diff checks passed.
The agent's first original-file copy had one extra final blank line; that
receipt is preserved and explicitly distinguished from both the agent's later
exact-byte reproduction and root's independently authenticated exact-byte test.

No application edits, source refreeze, replacement full suite/upload/container,
paid request or production action occurred during this local repair. The v5
failure remains terminal and its verifier must not run again. The follow-up
remains PAUSED. Modified bytes require fresh full local/target validation; the
old 6,532-pass receipt is historical, not acceptance of this change. The proposed
bounded continuation is in the root evidence directory's
`proposed-continuation.md`; it is an approval request, not authorization.
LME completion and live semantic improvements remain unverified.

### User approved v6 continuation; new full local gate running

The user replied **"Yes"** to the explicit bounded continuation request after
the lease-test repair. This is genuine new authorization, recorded in
`user-payload-approval-20260913-v6.json`, SHA256
`485b1ea89f28078e319ab0cece39e3751b600e5c85c1db59737617d1630f3830`.
It binds the proposed continuation SHA256
`a57c9fb905bc2efbbb97ef14fde28d92a870fc117fa62f8259ba0664717e0c7d`:
fresh full local validation, complete revised private source/test bundle and
helper transfer, one isolated Afrodite gate, and only after successful gates and
rehearsal the original one-unrun bounded diagnostic. It includes resuming the
existing follow-up for this sequence. It excludes production, restarts, live
stores, Hermes2/3, deletion, canary/smoke/full LME, automatic replacement after
failure and extra paid campaigns. Prior approval-pending sections are historical.

Root rechecked the full Git-listed inventory: exactly 500 files, no additions or
removals, and only the accepted lease test differs from v5. The two previously
excluded evolving status documents remain the only exclusions. All three
maintained full/target gate helper hashes still match their reviewed values.

Preparation of `local-full-candidate-v6` completed with actual exit 0. Collection
completed cleanly with **6,536 exact nodes / 209 test files / 500 source files**;
no collection failure or deselection, and only the explicit no-packet guard
selftest. Source manifest SHA256:
`459998d71d23690f47c58846975ed86c4857885c3c1e4221ef49c1b4a0531a8d`.
Plan SHA256:
`943af9fb3b8c4a753b24dba0100768153e90c5738095c320d64902b08ef88231`.

Four whole-file disjoint groups of 1,634 were each launched once at approximately
17:48 UTC. Their launch IDs are `shard-1-initial` through `shard-4-initial`; tool
sessions are respectively **32324, 11693, 12598 and 18741**. At 17:50:45 UTC all
four logs were progressing, with no exit receipt or verdict yet. Never relaunch
them. Wait for actual parent and worker exits before the once-only reconciler
and root's independent audit. Preparation session 64261 is closed, exit 0.

Current handoff: `candidate-v6-local-handoff.json` in the private evidence root.
Root prepared `root_audit_local_v6.py` as a binding-only copy of the reviewed v5
independent audit, with exact new manifest/plan/counts; no audit criterion changed.
SHA256 `a9ed714b6459b35a1e5e05ffde18d07f720cb87b7fd34693b1124f8b2b13f030`.
It has not run and must wait for terminal full-gate acceptance evidence.

Using OpenAI Docs, root resumed the existing `finish-lme-validation` follow-up
for this approved sequence and reread its saved configuration: exact ACTIVE
status/prompt and preserved task/cadence verified. It stays quiet on unchanged
state and pauses on completion or actionable failure, without reviving any old
stage. Official reference: [Scheduled tasks](https://learn.chatgpt.com/docs/automations).

No v6 upload or target run, diagnostic metadata rebinding/upload, rehearsal,
formal consent or paid call has started. The full local gate is **running, not
accepted**. No application edits or production operations occurred this turn.

### V6 local results: localhost permission restriction; follow-up paused

The 18:13 UTC follow-up found all four initial launches terminal. Root consumed
their actual parent exits exactly once: sessions 32324/11693 exited 1;
12598/18741 exited 0. All four handles are now closed and must not be polled or
reused. Independent worker exit receipts agree. No reconciler or acceptance
verdict has run/been created.

Root independently audited every exact node, execution phase, XML identity,
intent/worker/exit/hash binding, source snapshot and runtime: **6,515 passed,
5 failures, 16 setup errors, 0 skipped**, out of 6,536. All 21 nonpassing checks
are `PermissionError: [Errno 1] Operation not permitted` at a bind to
`127.0.0.1` on an ephemeral port. Five belong to
`tests/test_ensure_embedding_server.py` in shard 1; 16 are setup failures in
`tests/test_honcho_contract.py` in shard 2. All 18 lease tests passed. The two
other shards each passed all 1,634 tests. No other failing phase exists.

A fresh standalone bind-and-close probe, with no HyMem import or outgoing
connection, failed identically under the default sandbox. Root reviewed the
fixtures: they require real in-process HTTP servers on localhost. Root's launch
setup caused this avoidable failure: earlier gates documented scoped localhost
permission, but v6 was launched entirely with default permissions. Whole-file
repartitioning also changes which numbered shards contain those fixtures.
Future execution must derive the affected groups from the current frozen plan
and preflight their localhost capability before starting long test runs.
This does not justify application edits, mocks, skips or weaker assertions.

Private failure handoff: `candidate-v6-local-failure-handoff.json`.
Root's read-only failure audit `root_audit_failed_local_v6.py` completed exit 0,
explicitly reporting gate accepted false; SHA256
`ed72565b2b59b10c212e561b3cb946cb04e8deec2cdb113c2a15011856646d32`.
All frozen source, helpers and runtime remain unchanged; every worker recorded
only its explicit no-packet socket selftest and zero external application
socket attempts. The failed and successful receipts are preserved unchanged.

Following OpenAI Docs, root paused `finish-lme-validation`, then reread and
verified its exact PAUSED prompt/status and unchanged task/cadence. ACTIVE and
running sections above are historical. No package, SSH, upload, target run,
diagnostic metadata rebind, rehearsal, formal consent, paid call or production
action occurred. The diagnostic and Afrodite target attempt remain unconsumed,
conditional on full local acceptance; LME is still unverified.

Proposed narrow recovery, pending direction: preflight scoped localhost access,
rerun only unchanged groups 1/2 under new launch IDs while retaining the failures,
then reconcile replacement 1/2 plus successful original 3/4 once. Root's private
acceptance audit must explicitly represent all six attempts; its prepared
four-original-success assumption is no longer usable unchanged. The immutable
gate helpers already support explicit selected launches and retained attempt
history. Details: `proposed-v6-localhost-retry.md` in the private evidence root.
No replacement source, automatic retry or downstream launch was performed.

### User approved scoped localhost retries; both running

The user replied **"Yes"** to retry only the two affected groups with localhost
access, retaining their failed evidence, then resume the already-approved
downstream sequence if they pass. Genuine approval is recorded in
`user-localhost-retry-approval-v6.json`, SHA256
`cc47c2da3548a378c372975ad58810f583a1edf384473a556d6dedac2fca4fb6`;
scope SHA256 `f395ace6b32679997f9289ab14efa74fcfe17797a46862b80a55b0af3db78476`.
No replacement source or extra paid authority was added.

Root verified an actual ephemeral localhost bind/listen/connect/accept/data
round trip with the specifically approved per-command sandbox exception:
exit 0, all sockets closed, no external endpoint or application import. The
frozen plan confirms the relevant files are in groups 1 and 2. Immediately
before retries, root re-audited all original failed/successful evidence and
unchanged source/runtime/helpers; exact counts and failure causes still match.

Only `shard-1-localhost` and `shard-2-localhost` were launched once, each with
the reviewed localhost-fixture exception and unchanged provider stripping,
immutable external-socket guard and fatal warning policy. Started at
18:28:03/18:28:08 UTC; tool sessions **23267 / 21803**, respectively. Each
contains 1,634 tests. At 18:32:26 UTC both logs were progressing without a
reported failure; neither had an exit receipt. This is not acceptance.
The original four sessions remain closed and must not be polled. Successful
original groups 3 and 4 were not rerun; both initial failures are preserved.

Prepared root audit: `root_audit_local_v6_six_attempts.py`, SHA256
`aed1ab1d4e830334435ea5ea42bbf8aa2ab1c00b1a4bab82e2c3afa07d9d7e4b`.
Root reviewed the changed selection/history block and verified the written
bytes against that transformation. The original selected-success assertions
are unchanged. Added checks cover exactly six attempts, all attempt metadata,
and retained failure exit/XML hashes anchored to the prior independent audit,
plus their actual node/phase/source/runtime/guard evidence. Syntax passed;
execution awaits actual successful retry exits and the once-only reconciler.
The older four-attempt audits remain historical and cannot certify this history.

After success, the reconciler must select `shard-1-localhost`,
`shard-2-localhost`, `shard-3-initial`, `shard-4-initial` in that order, exactly
once. Both failed attempts must remain in the six-attempt history. No acceptance
verdict exists and reconciliation has not run. Current handoff:
`candidate-v6-localhost-handoff.json`.

Following OpenAI Docs, root resumed `finish-lme-validation` for these exact
retries and the existing conditional Afrodite/diagnostic sequence. The saved
configuration was reread and exact ACTIVE prompt/status, cadence and task
verified. Earlier PAUSED statements are historical. It stays quiet on unchanged
state and pauses on a new actionable failure or completion. No v6 package,
SSH/upload/target run, diagnostic rebind/rehearsal/consent/paid request or
production operation has started.

### V6 local accepted; approved isolated Afrodite full gate launched

Both scoped localhost retries completed with actual parent and worker exit 0:
sessions 23267 and 21803 are closed. All six local attempt handles are now
closed. The once-only reconciler selected replacement groups 1/2 and original
successful groups 3/4, then root independently audited every exact node, phase,
XML identity, source/runtime/helper binding and all six attempts, including
the unchanged initial failure receipts. Both reconciliation and root audit
exited 0. **6,536 passed; 0 failures, errors or skips; 0 external application
socket attempts.** The deliberate no-packet guard selftests remain accounted.

Local verdict SHA256:
`6a9a2619bc31281662df99305ec70a1128394163b7d57caccc3c17bb01f5a92c`.
Independent `root-local-full-v6-audit.json` SHA256:
`8fc813f2535c39365fbc2799a306b7046403557324f956c3430f2cb0b5d5e2f9`.
Neither local reconciliation nor old four-attempt audits should run again.
Earlier running/not-accepted statements are historical.

Root packaged the accepted unchanged candidate: 500 source/test files, 209
test files, 6,536 exact nodes. Package SHA256:
`5d3143af06fd95825ced2d81dcd0f99f298acb9fc0f52d0f2d7b45f20b5e8349`.
The eight-file bundle totals 5,417,606 bytes. Root independently checked every
bundle hash and all 500 regular archive members; the explicit transfer exited
0. On Afrodite, root authenticated all eight uploaded files before any helper
execution, verified 0700/0600 permissions, then staged and independently checked
all staged bundle/source hashes and absent launch intent before launch.

Exact private stage:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-digest-fidelity-20260913-6v77k7sc`.
Container: `hymem-fidelity-full-6v77k7sc`, ID
`75b9ffa57798da5e16cd8bf2acc5514b77aed529791982760d32fe2a63a7de0f`.
The **one approved launch** completed with actual exit 0; session 27673 is
closed. Docker started at 18:54:02 UTC. Do not launch it again.

At 18:59:00 UTC a fresh read-only check, after re-authenticating the staged
bundle, confirmed the exact container/image identity and required isolation:
network none, read-only root/source/runtime/deps, private receipts/temp only,
two CPUs, 2 GiB memory/total swap, UID/GID 1000. Collection exited 0 and all four
test groups were running, with no failure markers in their logs at that instant.
**The target gate is running, not accepted.** No terminal verifier or terminal
capture has run; no target verdict exists. Observe compact state until terminal,
then verify exactly once and independently audit the evidence. Preserve failed
v5 and older stages without restarting or rerunning their terminal writers.

Current handoff: `candidate-v6-target-handoff.json` in the private evidence root.
Using OpenAI Docs, root updated the existing follow-up to this exact target and
the already-approved conditional downstream sequence, then reread and verified
its exact ACTIVE prompt/status and unchanged cadence/task. It remains quiet on
unchanged state and pauses on completion or a new actionable failure. Official
reference: [Scheduled tasks](https://learn.chatgpt.com/docs/automations).
Only after target acceptance may the
diagnostic metadata rebind/review/upload/rehearsal/actual-stage consent proceed.
That original one-unrun four-case paid diagnostic has still made **zero calls**.
No production deployment, restart, live-store access, Hermes2/3 operation,
deletion, canary, smoke or full LME occurred. Local and target unit tests do not
establish semantic LLM performance or completed LME readiness.

### V6 target test failure: attestation fixture deadline; follow-up paused

The 20:26 UTC check found a new `GATE_FAILURE` in shard 1:
`tests/test_store_attestation.py::test_embedding_receipt_attests_vectors_and_reuses_without_provider_probe`.
At line 958, `first.dream(max_cycles=10, timeout_s=10)` raised
`IndexingConvergenceError`. The underlying deadline expired at
`benchmarks/strictness.py:1098`, **after the dream returned and after its next
durable-status call returned**. The test never reached its receipt publication
or reuse assertions. No application/fixture edit or replacement run was made.

Root inspected only the existing synthetic fixture database, with SQLite
`mode=ro` and `query_only=ON`, reading counters rather than source content or
vectors. Its single dream ran from 20:23:52 to 20:23:59 UTC, processed/embedded
one chunk, and recorded no run error and zero coverage, digest, quarantine,
fact, profile or aggregation failure counters. The two extraction calls were
stubs; the embedding SDK was also a test fake. This is not another live-model
decline. Exact local XML evidence shows the same test passed in **3.696 s** in
the accepted retry and **3.836 s** in the retained initial attempt.

The demonstrated failure is a hard-coded 10-second fixture deadline expiring
in a non-deadline attestation test. Resource-sensitive timing under the shared
two-CPU gate is plausible, but has **not** been causally established by a
reproduction or profile; a genuine performance issue is not ruled out. Do not
weaken production deadlines or attestation checks based on this observation.

Credential-screened traceback: `candidate-v6-failure-trace-screened.json`,
SHA256 `6b00acc55dd6328e6717f5b42b792bed5503bf014835d2f75d9d81ba8f6f9be0`.
Read-only fixture/local comparison: `candidate-v6-fixture-diagnosis.json`,
SHA256 `cce2d52487b540713d89920762d53f53bc80319e7524e88af152cf159bdc369f`.
Current handoff: `candidate-v6-target-failure-handoff.json`, all in the private
evidence root. Earlier ACTIVE/running-without-failure sections are historical.

At **20:31:12 UTC** the exact container was still running, with the same one
reported failure, no OOM and verified unchanged isolation/bundle hashes. All
four shard process exits were still absent. The parent stops siblings when a
shard exits and fails validation, not immediately on a diagnostic per-test
marker. The original 7,200-second shared deadline remains unchanged. Root did
not manually stop/restart it, run terminal verification early, or fabricate a
terminal exit/count. **No terminal verifier/capture/verdict exists yet.** The
existing failed run still needs terminal evidence closure; never relaunch it.

Following OpenAI Docs, root paused `finish-lme-validation`, then reread and
verified the exact PAUSED prompt/status and preserved cadence/task. The pause
does not stop the already-running isolated container. Official reference:
[Scheduled tasks](https://learn.chatgpt.com/docs/automations).

The paid diagnostic remains at **zero calls**, with no metadata rebind, helper
upload, rehearsal or formal stage consent. No production deployment/restart,
live-store access, Hermes2/3 operation, deletion or LME launch occurred. The
local 6,536-test acceptance remains valid, but this target cannot be accepted.

Proposed, awaiting fresh direction: a fresh agent reproduces and minimally
repairs the attestation-timeout cause offline, followed by root review and
independent regression/deadline/negative checks; also close the existing failed
target evidence when terminal. Any revised full-suite payload/target launch is
proposed separately after that repair. See
`proposed-v6-attestation-timeout-repair.md` in the private evidence root.

### Attestation fixture repair accepted locally; v6 failed target closed

The user replied **"Yes"** to the bounded fresh-agent repair and independent
verification. Approval receipt `user-attestation-timeout-repair-approval-v6.json`
SHA256 `4585baac426680a481604f2db25848584a71d6130b5e497fc10ff63f120b8fe4`
binds the proposal SHA256
`3eed55b60604696c71292b5f9581b01dd0284dab51c9f9509b78c0142e92e78b`.
It did not authorize a replacement full-suite payload/server run, paid request,
or heartbeat resumption. Fresh implementation agent:
`/root/fix_attestation_fixture_timeout`.

Only `tests/test_store_attestation.py` changed, SHA256
`025906ca0e77fc9821700835b11ac4b235903bb67dcfd115323afecde7015c9d`.
Root rehashed the entire v6 inventory and verified all other **499 files**
unchanged. The fix replaces this one correctness fixture's 10-second deadline
with a named **finite 120-second** integration budget for both initial/reuse
waves; the ten-cycle limit and production deadline implementation/defaults are
unchanged. Real ingestion, extraction, embeddings, status, attestation,
close/reopen, receipt validation and zero-reprobe assertions remain. Added
assertions require consistent initial/reused indexing settings.

Five permanent controls retain the real integration path and use only the
convergence loop's supported injected clock: 11/119-second post-status delays
succeed, the old 10→11-second case fails, new 120→121-second expiry fails before
publication, and the exact120-second boundary during reuse fails while keeping
the original receipt. A larger genuine stall still fails intentionally.

Root independently tested an **actual 10.5-second post-status delay** against
both exact frozen-v6 and repaired source: old raises `timeout_after_cycle`
with no receipt; fixed reaches actual attestation/reuse successfully. Separate
negative mutations prove real vector-dimension corruption is rejected and an
extra fake-provider request on reuse still trips the zero-attempt assertion.
Captured adapters, clients and SQLite connections closed on every arm.
These four root controls plus **329 full-file related regressions** all passed,
with actual exits0, unique XML reconciliation, zero failures/errors/skips and
zero external application socket attempts. Fatal unraisable/thread warnings
remained enabled. Agent's overlapping checks are not added to root's **333**.
Root test sessions9934/48374/39016 are closed.

Root review: `/private/tmp/hymem-root-attestation-timeout-20260913.pRs5eN/root-review.json`,
SHA256 `af81fc6983107d3b11d4e8bd07f4ccf38e2109ef6768814443e75135a76f58c7`.
Agent report: `/private/tmp/hymem-attestation-timeout-repair-20260913.KKmV1H/agent-review.md`.
Profiling highlighted real canonical identity/status and SQLite work, not
provider waits. These measurements and the controlled reproductions establish
fixture timing sensitivity, **not** exact server-scheduler attribution or the
absence of all architectural performance regressions.

The original v6 container exited **1 at20:41:18 UTC**, without OOM. Shard1
finished1634 cases with1633 passing and the one known failure; the parent then
killed/joined siblings2–4, whose missing execution receipts do not certify their
passes or attempt counts. At20:43 root authenticated the bundle and ran the
terminal verifier **exactly once**: actual exit1, correctly rejecting the failed
container after preserving Docker log and terminal isolation evidence. Never
run that terminal writer again. Root's separate read-only forensic audit exited0,
confirmed source/runtime/dependency/production-code hashes unchanged, exact
isolation/receipt bindings, and no target acceptance verdict.

Terminal closure: `failed-target-audit-v6.json` in the private evidence root,
SHA256 `39fb49d82aa4beeabba60faa19940273d713bd6707356c984e99adb4dd4a80d5`.
Audit session86310 is closed. Earlier statements that v6 is still running or
awaits terminal verification are historical. No source update was deployed to
that stage or production. No production database was opened.

The focused repair is accepted; **complete updated-candidate local/server
validation has not run**. The old6,536-test local verdict certifies the prior
bytes only. The paid diagnostic remains at zero calls, and the follow-up remains
paused. No new payload/upload, server run, paid request, restart, Hermes2/3,
deletion, canary/smoke or full LME occurred. The next bounded continuation is
proposed in `/private/tmp/hymem-root-attestation-timeout-20260913.pRs5eN/proposed-continuation.md`:
full local gate, complete private source/test/helper transfer, one isolated
target gate, and only after acceptance/rehearsal the original unrun diagnostic.

### V7 approved continuation: complete local gate launched once

The user's new **"Yes"** approves the bounded continuation above, including
resuming the existing follow-up. Approval `user-payload-approval-20260913-v7.json`
has SHA256 `524b2f342a956f02e903d6dedd3f20a9109c945e9eb69ac1f0fd258a6dac9db5`
and binds proposal SHA256
`f8ac1f658d8e3ceb4e6d99358315e6748d189ea273e267b3843c6fb12409e7ef`.
It is not yet the later actual-stage formal paid-campaign consent receipt.

Before launch, root completed a scoped real localhost bind/listen/connect/
accept/data roundtrip (actual exit0, sockets closed, no external application
access). The affected groups were derived from the **new** collection plan:
embedding-server fixtures in group2 and Honcho fixtures in group3. Only those
groups received the scoped localhost permission exception; groups1/4 retain
default execution. Provider stripping, external socket guard and fatal warnings
remain unchanged. This avoids repeating v6's missing-localhost-permission issue.

Preparation exited0 (session79513 closed). New private gate:
`/private/tmp/hymem-digest-fidelity-20260913.xq6FJw/local-full-candidate-v7`,
ID `f19176f5-22ca-4c59-8a0f-9cfbef08a0a9`.
Source manifest SHA256
`d6a6eb3a077d9f5fa33aa3dc0affe8f2e60ba1f19293453eac49d1a1e1b9a8b3`;
plan SHA256 `21c25687de3dbe49ca2464b9e9717d422e1ab9881dd7d144af7612d61ac46207`.
The500-file inventory has no additions/removals; only the reviewed attestation
test differs from v6. Exact collection: **6,541 nodes across209 test files**,
partitioned1636/1635/1635/1635. No acceptance is inferred from collection.

Each initial group was launched **once**: shard1/session5124,
shard2/session71072, shard3/session23721 and shard4/session6252. At21:02:48 UTC,
all four actual parent sessions remained running with no exit/worker/XML closure
or verdict. Do not relaunch them. Handoff: `candidate-v7-local-handoff.json`.
Root prepared `root_audit_local_v7.py`, SHA256
`1fd5eaf95ad491ba1686c55384c4d25b074a3e419761590416456ad160911ccd`:
only current path/hash/count bindings change, and group counts now derive from
the authenticated plan. All exact node/phase/XML/source/runtime/guard and
four-attempt checks remain. It has not run; reconcile once and independently
audit only after all four actual parents and workers terminate successfully.

There is no v7 bundle, upload, target launch or paid diagnostic yet. The original
one-unrun diagnostic remains at **zero calls**. A new failure pauses the sequence
without automatic replacement or relaxed limits. No production deployment,
restart, live-memory/backups access, Hermes2/3, deletion, canary/smoke or full LME
is included. All old failed stages remain closed and preserved.

Using OpenAI Docs, root resumed the existing `finish-lme-validation` follow-up
for this exact v7 sequence, then reread and verified the complete saved ACTIVE
prompt and unchanged name/kind/cadence/task. It remains quiet on unchanged state
and pauses on a new actionable failure or completion. No duplicate follow-up
was created. Reference: [Scheduled tasks](https://learn.chatgpt.com/docs/automations).

### V7 local gate accepted; one isolated Afrodite full suite running

At the21:23 UTC follow-up, all four actual local tool parents returned exit0
and closed (sessions5124/71072/23721/6252). Every durable worker/exit was present
and successful. After authenticating the frozen helpers, root ran reconciliation
**once**, actual exit0, then the separate read-only root audit, actual exit0.
Accepted: **6,541 tests,209 files,500 source files; zero failures/errors/skips**.
The audit independently checked all exact node/phase/XML identities, all four
initial attempts, source/runtime bindings and guard counts. No external
application socket attempts occurred. There were no local retries in v7.

Local verdict SHA256
`b962a2eb149be5d010d76201dca70e277c72758b7d215e6b01be004ffe9bfabe`.
Root receipt `root-local-full-v7-audit.json`, SHA256
`77fda3f297ee880a8e77044ac6a97ab12e178776963256a55aec2c7951f6e453`.
Source manifest/plan remain the v7 values above. Earlier statements that v7
local validation is running/not accepted are historical, not current.

Only after that acceptance, root packaged the complete frozen source/test
inventory and local evidence with the three unchanged reviewed helpers.
Package session16710 exited0 and closed. Independent checks verified the exact
500 tar members and their content hashes, eight bundle-file hashes and private
permissions. Payload: **5,419,178 bytes**, uploaded once to
`/var/tmp/hymem-digest-fidelity-upload-20260913-v7-P8sh6enc`.
Package SHA256 `bfa246776a0da01eec1e5f9bc684bfc53a54c5888a2be7f7cbbe2ffce4eaea88`;
archive SHA256 `5c226e1a56d3087a065e246bf6cb748a62634bb4ef1dbc5840e3f418b081572c`.
Root independently authenticated all uploaded bytes before staging, then every
staged helper and all500 candidate files before launch. No production overlay.

Exact new private stage:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-digest-fidelity-20260913-4pqc4r9s`.
Container `hymem-fidelity-full-4pqc4r9s`, ID
`c485a148a3c50d59b735eb10fa615ae4ff2a942dd0a50d2477a4b21707ac26a2`.
The **one approved launch** exited0; session98835 is closed. Docker started
at21:26:30 UTC. Never relaunch or reuse this stage.

At21:27:34 UTC, read-only inspection verified exact image/container identity
and unchanged isolation: network none; read-only root/source/runtime/deps;
private receipts/temp only; two CPUs;2GiB memory/total swap;UID/GID1000; the
original shared7200-second deadline. Collection exited0 and all four groups
were running, with no diagnostic failure markers at that instant. **Target
gate running, not accepted.** No terminal verifier, capture or verdict exists.
Observe compact state until terminal, then verify once and independently audit.

Current handoff: `candidate-v7-target-handoff.json`, SHA256
`d0c40ca287fcf178d542bcb9795d5c74bdb060b8563e9883e4258c7bf88df8ee`.
Read-only observer `observe_target_v7.py`, SHA256
`95cb52a1fe814a7dd510befc7c71c42f11a03b8af0b6429d730cc961514f345b`.
Both are in the private evidence root. The older local handoff points to this
closure while preserving its initial running snapshot. No old failed stage or
terminal writer was restarted or reused.

The original one-unrun paid diagnostic remains at **zero calls**, with no
metadata rebind, upload, rehearsal or formal actual-stage consent yet. No
production deployment/restart, live-memory/backups access, Hermes2/3 operation,
deletion, canary/smoke/full LME occurred. Passing local tests alone does not
establish target acceptance, semantic gains or a completed LME benchmark.

Following OpenAI Docs, root updated the existing ACTIVE follow-up to this exact
v7 target, then independently reread and verified its complete saved prompt,
status, and unchanged cadence/name/kind/task. It remains quiet on unchanged
state and pauses on a new actionable failure or completion. Official reference:
[Scheduled tasks](https://learn.chatgpt.com/docs/automations).

### V7 target failures: CLI child import context; follow-up paused

The22:56 UTC check found two new `GATE_FAILURE` events in the existing v7
isolated full suite. Read-only extraction of event metadata identified:

- shard1: `tests/test_orphan_quarantine_rehearsal.py::test_unmodified_current_runtime_refuses_historical_quarantine_cli`, line298.
- shard4: `tests/test_orphan_quarantine_application.py::test_unmodified_current_runtime_refuses_historical_application_cli`, line416.

Both direct script-mode subprocesses raise **ModuleNotFoundError: No module
named hymem**, before the expected structured historical-schema refusal. Both
fail at their empty-stderr assertion. Root returned only test names, exception
types, source locations and hashes from private logs; raw traces/source payloads
were not copied into this report. Trace SHA256 values are respectively
`3800ebaec92cef7ed158dcf0fa834c7e43ebe92ef19648f572f070630d4f916f`
and `dd4cdb973766cf3b888a1bca1386772e80cace2aa0ca6f7a61c4ce7c232ca060`.

Code inspection shows these tests call the helper by absolute script path,
without explicit child import setup. The guarded worker adds its current
candidate directory to its **own** sys.path, while its environment sanitizer
removes inherited PYTHONPATH. That in-process path edit does not propagate to
these script-mode children. The evidence supports a source-only subprocess
test/setup defect, not a new LLM decline or the repaired attestation deadline.
The exact minimal repair and any ambient-install contribution still require
focused reproduction. No historical-schema safety guard should be weakened.

At **22:59:14 UTC on2026-09-13**, the exact v7 container remained running with
noOOM and verified unchanged bundle/isolation. Shards1/4 each had one failure;
all four shard worker/exit receipts were still absent. Shards2/3 had no failure
markers but were not certified successful. The original shared7200-second
deadline remains unchanged. Root did not stop/restart the container, change
the candidate/runtime, or run the terminal writer early. **No terminal
capture/verifier/verdict exists yet.** This failed run still needs evidence
closure after terminal; never relaunch it or retry a failed terminal writer.

Latest handoff: `candidate-v7-target-failure-handoff.json`, SHA256
`28827daf55465253975b18544d4663d3e6023b00922432ad8360bcf7cc175af9`.
The earlier v7 target handoff now points to it and retains its original running
snapshot as history. Both live in the private evidence root.

Using OpenAI Docs, root **paused** the existing follow-up and then reread and
verified the exact saved PAUSED prompt/status and preserved cadence/name/kind/
task. That pause does not stop the already-running test container. Reference:
[Scheduled tasks](https://learn.chatgpt.com/docs/automations).

Local6,541-test acceptance remains valid for these bytes, but the target cannot
be accepted. The original one-unrun diagnostic still has **zero paid calls**;
no diagnostic metadata rebind/upload/rehearsal/formal consent began. No source
fix, replacement run, production deployment/restart, live-memory/backups
operation, Hermes2/3, deletion or LME launch occurred during diagnosis.

Proposed next, awaiting fresh direction: a fresh agent reproduces and repairs
the shared CLI subprocess import setup offline; root independently checks
correct candidate imports, historical refusal/no-write behavior and negative
controls; also close the existing failed target evidence when terminal.
No replacement full-suite payload/run is included in that focused repair.
Proposal: `proposed-v7-cli-import-repair.md`, SHA256
`a5c589f74e318f700a315dea40e4aa5a63e324e29a5411658bce56a04540e9d8`,
in the private evidence root. No agent has been dispatched for this new defect.

### 2026-09-14: CLI subprocess repair accepted; failed v7 target closed

The user's fresh **"Yes"** approved the bounded offline repair and original
target evidence closure, not another complete validation/deployment campaign.
Approval `user-cli-import-repair-approval-v7.json`, SHA256
`104d07cc55b3d995b39bb82c254cc022f8a185ea77367349707117ba29ac7553`,
binds the proposal above. Fresh agent: `/root/fix_cli_subprocess_import_v7`.

The cause was subprocess import setup: the test parent could import this
checkout, but its sys.path change was not inherited by direct-script children.
Disabling ambient site/editable-install hooks reproduced both original failures
before CLI parsing. These were not new extraction declines or guard failures.

Only two test files changed. A shared test-only fixture supplies the explicit
checkout and a private copy of the already-installed sqlite_vec dependency.
It replaces inherited PYTHONPATH, removes PYTHONHOME and uses `-s -S -B -P`
to disable user/site/editable initialization, bytecode writes and implicit
working/script-directory imports. Python>=3.11 is the project's existing
requirement. No installation, global permission or runtime/helper change was
made. Both tests still run their real scripts directly with the original
arguments, timeouts, exact refusal JSON and empty-stderr assertions.

Source physical/logical checks are strengthened. Application refusal still
creates no destination. Rehearsal deliberately retains baseline/working clones
before refusal: those are now compared with the source, and quarantine/restore
artifacts must remain absent. It is not described as globally write-free.
Permanent controls verify exact candidate module paths/hash, current schema61
and full migrations, no-site execution, hostile ambient path isolation and
wrong-parent-checkout rejection.

Root independently accepted **90 unique checks**: 79 full related quarantine,
CLI and deterministic-collection regressions plus 11 controls. The controls
execute the exact archived-v7 test functions to reproduce both import failures,
then prove both repaired direct calls work. They deliberately remove the
candidate import path, inject historical-schema bypass into real children,
perform unexpected synthetic source writes, and create an unexpected application
artifact; every applicable tripwire rejects the mutation. All results have
zero failures/errors/skips, actual parent exits0 and zero external application
socket attempts; fatal thread/unraisable warnings remain enabled. Root sessions
62203/18710 are closed. Agent's overlapping48-test suite is not added to90.
The agent's initial new-test environment-phase assertion failure is preserved
in its report; the correction and all final checks are recorded separately.

Root review: `/private/tmp/hymem-root-cli-import-20260914.i8sBNh/root-review.json`,
SHA256 `99173cbf438a0dfa64537d52cd5174e2403b5dcb751e31879452dc6160305431`.
Root XML hashes: controls
`9b5cbf99e414ec03fcf77d9ea202277cdacbc355b99b00b1fecf38896a3529f7`;
regressions `94433a8bffa7914ec18eec88a56db71f9bb17a4cb08225546d1b3751ed9157fe`.
Agent report: `/private/tmp/hymem-cli-import-repair-20260914.ZOF7DE/agent-review.md`,
SHA256 `30020cd7b790b32718eba7311d5d6efdf03c7003911aacc69f56c92ec0b624f8`.

Root verified the full500-file v7 inventory: no additions/removals and all
other **498 files unchanged**, including both production quarantine helpers.
Changed hashes: rehearsal test
`57776543bd37f3f76c1ac38bf8cfc445f8be31bf3d394067c65772d9bae1a6b3`;
application test `19aa2384f56c4627a745102f4f681b60fb6b055edf4ead22dbae6c4760100bcc`.
Syntax, exact unique XML counters and `git diff --check` passed.

The original v7 target actually exited **1 at23:16:58 UTC on2026-09-13**, noOOM.
Shard4 completed1635 cases with1634 passes and one known failure; the parent
killed/joined shards1–3 (exit-9), whose missing worker/XML receipts do not certify
complete results or attempt counts. Root authenticated the bundle and ran the
exclusive terminal verifier **once**, actual exit1, correctly preserving and
rejecting the failed container. Session94877 is closed; never rerun that writer.
The separate read-only forensic audit exited0 (session66287 closed), validating
unchanged candidate/runtime/dependency/production-code hashes, exact isolation,
receipt bindings and both failure identities. No production database was opened.

Closure receipt: `failed-target-audit-v7.json` in the private evidence root,
SHA256 `cf3bf8b980eb54a8d2f032bfdf81394018f54716d4f4c55a4d5c4df78547fb89`.
The failure handoff points to this closure and repair review; its older running
snapshot is historical. All v7-and-older failed targets are closed/preserved.

The focused fix is accepted. **Complete revised-candidate local/Afrodite
validation has not run**, and the old6,541-test verdict certifies prior bytes
only. No replacement payload/upload/run, heartbeat resumption, diagnostic
rebind/rehearsal/paid call, production operation or LME launch occurred.
The original one-unrun diagnostic remains at zero calls. Proposed next bounded
continuation: `/private/tmp/hymem-root-cli-import-20260914.i8sBNh/proposed-continuation.md`.
It requires fresh direction before complete local/target validation and the
conditionally gated diagnostic; it does not include deployment or full LME.

### 2026-09-14: revised v8 full validation approved and launched

The user's fresh **"Yes"** approved that bounded continuation. Approval
`user-payload-approval-20260914-v8.json` in the private evidence root has SHA256
`4040d9165c10a6f258406116260a61c3e57d270e78268605af72b903d4aec16d`,
binding the proposal SHA256
`86aefcd01a7d41e2759f438f698d1e426b9b377b434521658499b475f7f0ffbe`.
This supersedes the preceding awaiting-direction status only within that scope.

Before preparation, root reauthenticated all three frozen gate helpers and the
500-file inventory: only the two reviewed quarantine test files differ from
v7; the other498 are unchanged, with no added/removed paths. A scoped ephemeral
127.0.0.1 bind/listen/connect/accept/data-roundtrip preflight exited0 and closed
its sockets. No external connection was made.

Preparation ran once and actually exited0 (session27878, closed). New gate:
`/private/tmp/hymem-digest-fidelity-20260913.xq6FJw/local-full-candidate-v8`.
Gate ID `4170fb51-68bf-4dfb-896c-4ce5c7cfc99a`; source manifest SHA256
`68eca0b155cc3064250492a8edeed73cf7866a5e8f12a12e9b22682adecdbeaf`;
plan SHA256 `58327fce8ea8fac9136b38c40fb2a644a9d98bf70aedfdbaf100bec3f398144b`.
Collection derived **6,543 exact tests /209 test files /500 source files**.
The new whole-file groups have1636/1636/1636/1635 cases. New plan derivation
places Honcho localhost tests in group1 and embedding-server tests in group4;
only those groups received the preflighted scoped permission. The original
offline audit hook, credential stripping and fatal warnings remain in force.

Four initial foreground launches were issued once: group1 session73642,
group2 session12125, group3 session17250 and group4 session21658. At the launch
handoff all are running, with no terminal receipts: **not accepted**. No retry
or reconciler has run. Root's independent audit is a binding-only adaptation
of the prior checked audit; syntax and exact diff were checked, but execution
must await complete terminal evidence. `root_audit_local_v8.py` SHA256
`9c212064520700a666ee27747488ebc1e9f2a7ad7c27b80bf04ca9f58ccf0019`.
The durable continuation is `candidate-v8-local-handoff.json` in the private
evidence root. Later closure records supersede this launch snapshot.

Using OpenAI Docs for the requested follow-up operation, root resumed the
existing **Finish LME validation** heartbeat and verified its exact saved
ACTIVE prompt, unchanged cadence/name/kind/task, quiet-on-unchanged behavior
and pause-on-new-failure/completion instruction. No duplicate was created.
Reference: [Scheduled tasks](https://learn.chatgpt.com/docs/automations).

The approved continuation is local acceptance, a full eight-file private
verified upload, exactly one isolated Afrodite complete suite, then the
original one-unrun bounded diagnostic only after both gates, metadata-only
fresh-agent rebind/root review, isolated rehearsal and genuine stage-bound
formal consent. Any new actionable failure pauses downstream execution.
No new payload/upload/target/paid call exists yet. Production, live stores,
Hermes2/3, deletion, canary/smoke/full LME and extra budget remain excluded.
The failed v7 and earlier target stages stay closed and preserved.

### 2026-09-14: complete local v8 accepted; one isolated target running

All four original local parents (73642/12125/17250/21658) actually exited0
and are closed. Their durable worker exits were also0. Root authenticated the
frozen helpers, source/plan/approval and audit script, ran the exclusive local
reconciler once (actual0), and independently audited its evidence (actual0).
**6,543 exact tests /209 test files /500 source files**, zero failures/errors/
skips, every expected setup/call/teardown and XML identity, unchanged source
and local runtime metadata, and zero external application socket attempts.
No replacement launches supplied this acceptance.

Local verdict SHA256
`7535a564e02257ab4189bb9ca83ce97e578d3dbb72a45b289d62d3f4df6a483f`.
Root receipt `root-local-full-v8-audit.json` SHA256
`2f64fae9b83d21d54b244e5463d90dee8b67920f23add77fba1a046fc38c7b53`.
Group counts1636/1636/1636/1635; XML durations763.721/763.211/958.784/
814.749 seconds. `candidate-v8-local-closure.json` supersedes the original
local running snapshot; the newer target handoff supersedes its next-step text.

The complete revised candidate and local evidence were packaged once; package
parent98252 closed0. The eight-file bundle is5,420,614 bytes, package SHA256
`dad2696cff28cd615609b5620d770f4dc79e30a41646e78b5743d08004502856`,
archive SHA256
`cf6a9ad899f3497ae5a7f37d2eabe6fca4f3d6587860e951ca21564569ecdbe4`.
Upload: `/var/tmp/hymem-digest-fidelity-upload-20260914-v8-kHfCz7dl`.
Root independently authenticated all eight uploaded files before execution,
then all eight staged files and all500 unpacked source hashes and private
permissions before launch. No production overlay or live DB operation occurred.

Exactly one new target launched, actual launch parent74946 closed0:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-digest-fidelity-20260913-mqruaarr`.
Container `hymem-fidelity-full-mqruaarr`, exact ID
`bdde398f4efe2672dcafcb58eb45a9ee55dcb3cc239790b39f1f9e39b83d3a9b`;
started at2026-09-14T07:27:18.035050118Z. At07:28:19 UTC it is running,
collection exited0, all four shards active with no failure markers, noOOM.
The authenticated read-only observer verified the unchanged network-none,
read-only source/runtime/root, UID1000, two-CPU/two-GiB and mount boundaries.
The original shared7200-second deadline remains. This is **not acceptance**:
no terminal capture, target verdict or terminal verifier has run.

Durable handoff: `candidate-v8-target-handoff.json` in the private evidence root.
Observer `observe_target_v8.py` SHA256
`80ed43aa7cc062fa2054f7a8d09c2257c1152bd5d2ab9df9db30252a37efa6b3`.
Root used OpenAI Docs to update the existing follow-up to this exact target,
verified the saved ACTIVE prompt and preserved cadence/name/kind/task. It stays
quiet on unchanged state and pauses on new actionable failure or bounded-scope
completion. Reference: [Scheduled tasks](https://learn.chatgpt.com/docs/automations).

The original diagnostic remains at zero paid calls. Its metadata rebind,
18-file upload, rehearsal and formal consent have not begun; both complete
gates plus root audit are prerequisites. No production restart/deployment,
live-memory/backups operation, Hermes2/3, deletion or LME launch occurred.
All prior failed stages remain closed; their terminal writers must not rerun.

### 2026-09-14: v8 target timed out; failed evidence closed and follow-up paused

The original target exited1 at2026-09-14T09:27:18.814000089Z, noOOM. The
recorded shared duration is7200.122989536 seconds, exception `TimeoutExpired`.
Collection and shard2 completed; shard2 has1,636 exact passing tests,4,908
successful phases, XML0 failures/errors/skips, actual exit0. Shards1/3/4 were
killed and joined with exit-9/`TimeoutExpired`; they lack final worker/XML
records. Their last log percentages96%/74%/92% are only progress, not certified
pass counts. No `GATE_FAILURE` markers were recorded, but an incomplete gate
cannot pass. Local6,543-test acceptance remains valid for the frozen bytes.

Root authenticated all eight bundle files and verified unused terminal captures,
then ran the original exclusive terminal verifier **once**. It actually exited1
and correctly refused the failed container; session30678 is closed. The private
Docker log and terminal isolation capture now exist; no target verdict exists.
**Do not call this terminal writer again.**

A separate root read-only forensic audit exited0 (session14163 closed). It
verified unchanged candidate/runtime/dependency/production-code hashes, exact
container identity and boundaries, worker intent/exit/file-hash bindings,
complete collection and shard2 node/phase/XML/guard evidence, and timeout state
for the other groups. Missing killed-worker receipts cannot certify their
external-attempt counts; Docker network-none prevented external transmissions.
Production code checks are not a service-health or DB-integrity assertion.

Receipt: `failed-target-audit-v8.json` in the private evidence root, SHA256
`50d13506d42238c1c9e40fab296cb5d1403a751c7483f85b4611794f33196cee`.
Audit script `audit_failed_target_v8.py` SHA256
`076e1093b8e76f27fd974ea40cfaa0b22e2bddcf864bd1d6332851134b40cfdc`.
Docker log SHA256
`4c20ab0dca15cd5b3b31cb7f2a4db8794bee992f9720fed6948ba75eecdfcb44`;
terminal isolation SHA256
`35dfdc730b15e153d574bc18ab14fa623fb6c925f4b1e2936054fe69d0add24c`;
result SHA256 `64f9e06180d9981bee45a1cc55debbc109d7c3f8f172a7962d2749e5395addec`.
The newer failure handoff points to this closure; older snapshots are historical.

Six of seven120-second diagnostic samples put the main test thread in loaded
identity construction/serialization (`_loaded_identity_value` or
`canonical_module_slice_sha256`); the seventh is lossless coverage validation.
These are selected slow-test samples, not an unbiased CPU profile or proof of
one universal cause. Source inspection shows recursive class/closure/container
expansion with a depth cap, while existing caching only covers immutable code
and disassembly. This is a concrete profiling lead. No LLM call was involved
in this offline failure, and no performance repair has been implemented yet.

Using OpenAI Docs, root paused the existing follow-up and verified its exact
saved PAUSED prompt and preserved cadence/name/kind/task. Reference:
[Scheduled tasks](https://learn.chatgpt.com/docs/automations). The pause follows
the user-approved new-failure stop condition; it does not mark LME complete.

No paid diagnostic calls, metadata rebind/upload/rehearsal, replacement target,
production operation or LME launch occurred. All500 local candidate files were
rechecked unchanged. Proposal awaiting fresh direction:
`proposed-v8-identity-performance-repair.md` in the private evidence root:
fresh-agent focused profiling/repair, root independent semantic/performance
verification, then a new complete local and one isolated target validation;
the original diagnostic remains conditional on both passing and its rehearsal.

### 2026-09-14: v9 identity-performance repair approved and under review

The user's fresh “Yes” accepted a fresh-agent identity-bottleneck repair, root
independent verification, and repeating the complete local and one isolated
target gate. Approval is recorded privately in
`user-identity-performance-approval-v9.json`; the scope proposal SHA256 is
`bee8dcb9a132466f2691f4b4eca7d0f25f33ebc61ad50ec8b26131ea27741429`.
All v8-and-earlier targets remain closed, with their terminal writers used.
No production, live-store, provider or replacement-target operation has occurred.

Fresh implementation agent `/root/fix_loaded_identity_traversal_v9` reproduced
repeated object/depth expansion in the real loaded contract. Root's independent
oracle loads the original identity functions from the authenticated v8 source
archive and checks the same loaded objects with both implementations. The
prototype passes the first85 independent record/hash/mutation/dynamic-hook/
thread/retention cases. Root additionally checked discovery-hook alias mutation;
the revised memo lifetime is one top-level value, not multiple module bindings.

This is not acceptance. Timing found both a substantial contract/slice speedup
and an initially slower simple-module path; the agent is addressing and measuring
that tradeoff. Permanent structural tests and final root regressions remain due.
Root evidence is under
`/private/tmp/hymem-root-identity-v9-20260914.Ef7MUc`. No v9 complete gate is frozen
or launched yet. Later closure records supersede this in-progress snapshot.

### 2026-09-14: v9 focused repair accepted; complete local gate launched

Fresh agent completed the bounded repair: producer.py SHA256
`2b4fa362c8a101cfd8dd28a842ce52052839f2c7ecc6cf0859befb123a68070c`;
new44-case traversal regression file SHA256
`3b43fe5428e5f608fd5408bae9aacef143a6863f450925011d9171c672ff2eee`.
No other included file changed from v8:499 existing files unchanged, one changed,
one added, none removed. Agent's245 focused checks passed, actual exit0.

Root independently reviewed the complete implementation and permanent tests,
and accepted392 distinct checks (179 identity +120 semantic/message/GC +93 root
controls), zero failures/errors/skips, all actual parents exit0. The authenticated
v8 oracle checks the same loaded executable/state, preserving exact JSON/hash
semantics, mutation/rebinding detection, depth/cycle behavior and public tree
independence. Negative controls reject stale cross-root/cross-call reuse and
disabled memoization. No external application socket attempts occurred.

The per-root cache is bounded to4096 entries/8MiB ASCII fragments, holds input
references against ID reuse, and shares no mutable identities across invocations.
Dynamic hooks invalidate prior/enclosing fragments; module discovery cannot reuse
earlier roots. Exact-callable authority is unchanged. Final local median speedups:
contract3.58x, lossless slice4.34x, semantic digest3.40x, profile3.63x, facts1.04x.
The simple digest/facts module subcomponent remains0.367ms slower per call;
this explicit tradeoff is outweighed in measured full semantic paths, not erased.
Target-runtime/full-suite performance is not yet proven.

Root review: `/private/tmp/hymem-root-identity-v9-20260914.Ef7MUc/root-review.md`,
SHA256 `bddca5dd8d166f9c8e44cc74a201cc245b3f4e5272b2ef8ffcb77bc8740d3730`.
The final interleaved profile receipt in that directory has SHA256
`35ca768036a9427f611409c4a014766b5ca5556e4146b29578244366fe59ea67`.

Preparation ran once, parent73676 closed0. Complete local v9 gate:
`/private/tmp/hymem-digest-fidelity-20260913.xq6FJw/local-full-candidate-v9`;
ID `cf29ccd9-b465-423a-97af-1ec66180c9ca`; manifest SHA256
`1d08103a4533dafbc529a7df86234995d3810888dc4b6fc34f83caf8d94c13b4`;
plan SHA256 `b6e72f800885ebbd77ae501c4c9daccbbc2b5f5845bc5fab5b4eb8afe79f215e`.
Derived6587 exact nodes/210 test files/501 source files; groups1648/1647/1646/1646.
Fresh source inspection locates real localhost bind fixtures only in
test_ensure_embedding_server.py and test_honcho_contract.py, both new group2.
An ephemeral127.0.0.1 roundtrip preflight exited0 and closed its sockets. Only
group2 received that scoped permission; other groups use the default sandbox.

Original parents19236/62689/64392/97724 launched once and are running at the
handoff snapshot. No retry or reconciler issued. Root audit script is a reviewed
binding/count-only adaptation of the previous independent audit, SHA256
`8e18b1165d3e0cd85d986a3b7627cff3cbefa5a1cc321066f6cbb429a22e29dd`; not yet run.
The durable continuation is `candidate-v9-local-handoff.json` in the evidence
root. Current state is **not full-suite acceptance**. All old targets stay closed.
No production, live-store, paid call or LME operation occurred.

Using OpenAI Docs, root resumed the existing Finish LME validation follow-up and
verified its exact saved ACTIVE prompt, unchanged cadence/name/kind/task, quiet
unchanged-state behavior, and pause-on-new-failure/completion rule. No duplicate
was created. It is bound to the v9 handoff and the approved complete-local → one
isolated-target sequence, with the original diagnostic still conditional on all
its prerequisites. Reference:
[Scheduled tasks](https://learn.chatgpt.com/docs/automations).

### 2026-09-14: complete local v9 accepted; one isolated target running

All original local parents19236/62689/64392/97724 actually exited0 and are closed.
The exclusive reconciler ran once, actual0; root independent audit actual0.
**6587 exact tests /210 test files /501 source files**, zero failures/errors/skips,
exact node/phase/XML identities, unchanged source and runtime metadata, no
external application socket attempts. Group XML durations500.233/669.258/
481.799/462.358 seconds. No replacement launches supplied acceptance.

Local verdict SHA256
`513d333b24e2d878ee103fbbdb88ad68fd1574c3982d8f6e2b27326b2addf4d1`;
root receipt `root-local-full-v9-audit.json` SHA256
`fef4b5fe7fb26382ef48dab7acf3ca3bfc36437eb3c9f185a93d8231a8868141`.
`candidate-v9-local-closure.json` supersedes the initial running snapshot.

The complete eight-file bundle is5435178 bytes. Package parent29380 closed0.
Package SHA256 `00b2f062011d94216d5284e384f17497a17601d5d7098d3fff56c37fd2eab359`;
archive SHA256 `a0a83bca9dad0a1ba38565915888bd314bffe3496c1f059985e4a505f3f4c220`.
Fresh private upload `/var/tmp/hymem-digest-fidelity-upload-20260914-v9-OQxJcHoE`,
parent33153 closed0; all eight files independently authenticated before stage.
Stage parent87834 closed0; all eight staged files and501 candidate source hashes
and private permissions independently checked before launch. No production overlay.

Exactly one target launch, parent79327 closed0:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-digest-fidelity-20260913-cm68mtm3`;
container `hymem-fidelity-full-cm68mtm3`, ID
`0ba36e479316c73329c281f4d31e558f41d2fa7e793ec10a88f53b32150affc1`.
Started2026-09-14T10:34:00.592463147Z. At10:34:50 UTC, collection exited0,
all four shards active, zero failure markers, noOOM. Authenticated read-only
observer verified pinned image/runtime mounts, UID1000, network-none,
read-only root/source/bundle/runtime/dependencies and2CPU/2GiB boundaries.
Shared7200-second deadline remains unchanged. **Not target acceptance.**

Durable handoff `candidate-v9-target-handoff.json` now supersedes the local closure's
next-step text. Observer `observe_target_v9.py` SHA256
`1b67b71307d0f9d5c37bf668193c924f6afdcfb0477b2b02093b8a2de82b2773`.
No terminal capture, verifier or verdict exists yet. Never poll with the once-only
terminal writer. All v8-and-earlier stages and terminal writers stay closed.

The original one-unrun diagnostic remains at zero paid calls and conditional on
both complete gates, root audit, metadata rebind/rehearsal and genuine stage consent.
No production restart/deployment, live-memory/backups operation, Hermes2/3,
deletion, canary/smoke or full LME launch occurred.

Using OpenAI Docs, root updated the existing follow-up to this exact v9 target
and verified the saved ACTIVE prompt and preserved cadence/name/kind/task.
It remains quiet on unchanged state and pauses on new actionable failure or
bounded-scope completion. Reference:
[Scheduled tasks](https://learn.chatgpt.com/docs/automations).

### 2026-09-14: complete target v9 accepted and closed

The original target finished at11:59:24.855 UTC, exit0/noOOM, in5123.629 seconds
(85.4 minutes), under the unchanged7200-second limit. All four original shards
completed; no replacement launch or relaxed resource limit supplied acceptance.
The once-only terminal verifier (parent61433 closed0) and separate root audit
(parent63891 closed0) accepted **6587 exact tests /210 test files /501 source
files**, zero failures/errors/skips. Exact nodes, all setup/call/teardown phases,
XML display identities, process exits, socket-guard receipts and hash bindings
reconciled. Candidate, runtime/dependency bytes and production source hashes
remain unchanged; network-none/read-only/UID1000/2CPU/2GiB boundaries verified.

Target verdict SHA256
`da25e96dec4f6dbd54a25037ae53bffa9233d23fe6c17fbc6ea89daf7d94b2e6`.
Root audit script SHA256
`dacea6c2670a39688c543ebb23728b241e480b9deed994577b36b8ffcf9b2bc1`;
root receipt `root-target-full-v9-audit.json` SHA256
`097ea705facfda5bfaaa6bbd2cf57a2a6c3a4d09b7999afd53f8934ff4f19223`.
`candidate-v9-target-closure.json` supersedes the running handoff; terminal
writers are used and closed. Group durations4682.913/5098.990/4595.086/4271.572s.

Both complete v9 gates are now accepted. This proves offline regression checks,
not LLM semantic quality, LME completion, production service health or DB health.
No production restart/deployment, store/backups operation, Hermes2/3 action,
deletion, canary/smoke/full LME or paid diagnostic call occurred.

A fresh metadata-only agent prepares `live-diagnostic-v9` from v5. Root must
independently review/test it before exact helper upload and network-none rehearsal.
The original one-unrun64-completion/192-HTTP diagnostic approval is unchanged;
actual-stage/plan consent remains uncreated. No extra campaign or budget granted.

### 2026-09-14: diagnostic rehearsals accepted; paid launch blocked before execution

Fresh agent `rebind_diagnostic_v9_metadata` prepared17 files; root's fresh helper
review is the18th upload file. Only README, three compatibility assignments/comment
and five supplement metadata fields changed from v5;14 other files and all request
code/semantic criteria are identical. Agent95 checks passed (parent82852 closed0);
root separately95 passed (parent5146 closed0), zero failures/errors/skips and no
external application socket attempts. Root XML SHA256
`aeeb73a9b4900bef722c9105ec03514ecddb07e9d3090a68d41131ff6baec029`.
Root helper review SHA256
`de7775171e901793f283d0c2ce7f2cd15524eda1b52fcb4c1faa7cf467d3db3f`;
root metadata audit SHA256
`330aeed5490140891a488d37a42822c2a78030eee8d838ecf0d417cdfce4d070`.

Exact18-file231346-byte upload parent46095 closed0; all18 helpers and8 frozen
bundle files independently authenticated before execution. Candidate501 files,
runtime and benchmark-only input hashes matched. Network-none prepare12516,
selftest97808, independent replay57150 and isolation audit34729 all closed0,
each exactly once. Four cases,8 synthetic invocations/16 synthetic completions,
zero provider attempts; four deliberate rejected repairs and four accepted repairs.
All requests/cursors/accounting and3container boundaries/launch labels verified.
Prepared plan SHA256
`1925b250dab76ee0f2aca33cee240478fb24bc09f01b914d1e821e2f99e9c46d`;
mechanical replay `a6ab55d12fb002b8e18ca962d8aaa416cd10d5641894178e43b01269cb2d6795`;
isolation verdict `5648025f139d89bcac15d5b096b10b06e11e0decdd2a6eaf1cb4fbc3a78810f1`.

Root created stage-bound consent from the preserved original Yes! record and
the reviewed authorization writer completed (parent43239 closed0). These files
are retained as history, but **do not resolve the safety-review blocker** below.
Consent SHA256 `ac4be79cb5cac6254a1bf91a55e552736d98cd27398785597efc9d26b783f93c`;
authorization `2ab57b33f4376a167e023972201fe5842d80be339128eca9987ce34ff8d1224d`.

The single paid-launch tool request was rejected by auto-review before process
creation: trusted user content did not explicitly authorize the private travel/music
benchmark payload and external DeepSeek destination. No workaround or retry was
attempted. A separate read-only check exited0 and confirmed launch/run/summary/
process-exit/process-log absent, calls/invocations empty: **zero paid calls**.
Fresh explicit confirmation of payload, destination and charged64/192 bounds is
required. `candidate-v9-diagnostic-launch-blocked.json` supersedes preparation;
do not launch from the old consent/authorization while this blocker is unresolved.
No production deployment/restart, live memory/backups operation, Hermes2/3,
deletion, canary/smoke/full LME or API payload transmission occurred.

Using OpenAI Docs, root paused the existing follow-up at this approval boundary
and verified the saved prompt/status and preserved cadence/name/kind/task and
notification policy. No duplicate automation was created. Reference:
[Scheduled tasks](https://learn.chatgpt.com/docs/automations).

### 2026-09-14: direct confirmation received; laptop-independent transport preparation

The user directly answered the explicit payload/destination/charged-limit question:
"Yes. Can I close the laptop after you send this all without interrupting the workflow?"
This resolves the earlier approval blocker for exactly the still-unrun four-case
diagnostic. Fresh addendum `explicit-user-confirmation-20260914.json` SHA256
`0ce889f95b8db1142ab511e2c0c1b936cd7b720a9e570965d95f17a51a051d3f`
binds the current question/reply, original technical authorization, stage, plan,
payload, DeepSeek endpoint/model and unchanged64/192 bounds. Earlier consent,
authorization and blocked receipt are preserved byte-identically as history.
This is not another campaign, a retry of an executed campaign or broader LME approval.

Fresh root read-only preflight (parent7792 closed0) authenticated18 helpers,
8 bundle files,501 candidate files, runtime and target verdict, plus original
authorization and unused launch markers. No credential access or paid call occurred.
The existing launcher is foreground over SSH; closing the laptop is not reliably
safe for its parent/exit capture. Fresh agent `detach_v9_diagnostic_transport`
is preparing a separately tested outer supervisor without modifying any accepted
source, helper, request, prompt, test criterion or budget. Root review and an actual
remote detached-process check are required before claiming laptop independence.
The local Codex follow-up and root semantic review still require the laptop/app
to be available; server-side execution is not a transfer of the whole workflow.
The existing follow-up remains paused during preparation. No paid launch yet.

### 2026-09-14: diagnostic completed independently; semantic findings stop downstream work

Fresh transport agent and independent root each passed35 synthetic controls.
Transport SHA256 `6c978c3d96c45f1ebd2ede6288bd51abf784f5674ed2d91cd4c07407a8360422`;
root test XML `4690f77661a1b4744b0cd3ca821fb924b65beadb80c851f99687d67a58fd0eb9`.
The unchanged foreground launcher is supervised by a separate double-forked
Afrodite session with private output and an exclusive dispatch fence. A separate
root synthetic server probe survived its original SSH/parent exit and SIGHUP,
rejected duplicate launches and preserved its deliberate child exit7. This
protects against laptop/SSH disconnect, not server shutdown or process termination.
No accepted application or diagnostic helper bytes were changed for transport.

The actual diagnostic executed exactly once after direct approval and a fresh
authentication of all18 helpers/8 bundle files/501 candidate files and runtime.
Dispatch parent96725 closed0; supervisor3413460 and launcher3413461 finished
at13:11:49 UTC, with matching actual launcher and inner process exit0. All four
cases completed in four invocations, **six completions/six HTTP attempts**:
13306 input +2201 output =15507 reported tokens. The two travel primaries exceeded
the500-character summary limit and their normal summary-only repairs succeeded;
neither music case needed repair. No case quarantine or held invocation remained.

Independent network-none live replay (parent36464 closed0, once) passed. Postflight
and screened exporter (parent73152 closed0, each once) passed. Benchmark input
databases, source, production source, runtime and dependencies were unchanged;
the production memory store was never opened. Actual client and connections
closed. Replay container `7ba2b794e52c46690fb09e905df844851f66a9a7e39f1a2644bcd2492f833807`
finished0/noOOM. All24 allowlisted JSON files (23 receipts plus manifest) were
returned to the reserved private Mac directory (parent34659 closed0) and independently
hash-verified. No raw logs, credentials or database files were transferred.

Mechanical verdict SHA256 `9feb4623be2306de07495e5794ab494fd285f8705203e4cd627534258c023e29`;
transfer manifest `b7871deb8ba78314e9f6ffaa23aec45e60dbb7049db9b253751410bfd6cc2ab8`.
Root reviewed all six responses, all four accepted extractions, exact source
ranges and prior continuity against the unchanged checklist/S1-S4. **Structural
completion is accepted; semantic fidelity is not.** Remaining findings:

1. **Unsupported scope in a title:** granular music's second episode is titled
   "Homecoming is Netflix-only, not YouTube". Its cited source establishes Netflix
   availability and non-availability on YouTube, not exclusivity across all
   platforms. The correctly scoped episode body does not cure its false title.
2. **Per-item evidence overreach:** granular travel's first episode cites only
   message214's new horseback-wish suffix, while adding the earlier Big Sur trip
   clause from boundary-only context and Santa Ynez specificity not in that new
   span. The trip is visible independently in message216, but this episode does
   not cite216. This is inadequate item-level support, not proof the trip is false.
3. **Outcome loss during compaction:** granular travel's shorter summary retains
   the stable/route questions but drops that recommendations/directions were
   provided. Topic names remain and episodes retain the answers; the narrower
   finding is lost rolling-summary state, not total topic disappearance.
4. **Separate format drift:** that same repair emits a sentence followed by an
   "Earlier topics" fragment, despite the one-sentence instruction. Other lower-
   confidence/secondary observations include weakened explicit trip sequence,
   ambiguous grouping of mixed music examples and residual literal quote escapes.

Root semantic receipt: `candidate-v9-diagnostic-root-review.json`, SHA256
`dba9578d87222741a86c0042efe473bf3513988bab74716e24108bb616d01b20`, under the
private working evidence directory. It records the review of every primary/final
summary, topic-level judgments, source caveats and exact input-receipt hashes. Earlier source
claims carried only by prior summaries remain unverified continuity. The replay
checks syntax, eligible repairs, identifiers, cursors and accounting; it is not
an entailment check for every title, entity or summary relation. The existing
prompt already states the important rules, so adding more prose alone is not
a demonstrated enforcement fix.

The bounded diagnostic is finished, all writers are used/closed and no job needs
the laptop to stay awake now. Using the OpenAI Docs scheduling guidance, the
existing follow-up stays paused with the completed evidence/new findings instead
of its obsolete approval blocker. A future local review/follow-up requires the
laptop/app to be available; remote execution alone does not move that work.
No new repair, reroll, canary, smoke, full LME, deployment, restart, live-memory/
backups operation, Hermes2/3 action or deletion is authorized by this completion.
Obtain direction before the next repair/campaign. No LME score improvement or
general extraction reliability is established by these four incident cases.

### 2026-09-14: continuation — runtime verification decision, no new implementation yet

The user requested continuation. Fresh F1 agent `fix_digest_title_scope_v10`
and root independently reproduced acceptance of unsupported title exclusivity,
certainty and negation reversal, alongside supported exclusivity. The current
scope tests are intentionally prompt-wiring checks, not semantic rejection gates.
No application/test/helper files or completed evidence were changed, and no
provider or remote operation occurred during this investigation.

The sequential implementation/root-verification plan is
[Runtime digest fidelity](2026-09-14-runtime-digest-fidelity.md). A recommended
shared source-aware verification boundary would add one completion per
structurally valid digest candidate (two total normally; three with summary
compaction). It checks all item fields and summary relations in one bounded
batch as subsequent issues are addressed, rather than multiplying calls by
item count. This can catch unsupported publication, but it is probabilistic
screening and can increase false rejections/held attempts. A neutral-title
alternative avoids extra calls for F1 but materially changes retrieval inputs
and does not solve F2/F3. Keyword filters or more prompt prose are not accepted
as a general semantic fix. Root is requesting the user's cost/reliability
choice before implementation; paid validation remains separately unapproved.
