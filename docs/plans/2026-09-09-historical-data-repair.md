# Approved historical-data repair and aggregate rebuild

## Scope and starting state

The user explicitly approved separately reviewed historical-data repair and a
whole-aggregate rebuild, including possible relationship changes and additional
LLM cost. Preserve unprovable history; do not invent canonical provenance or
delete stale mirrors merely to clear diagnostics. No credential rotation or
full benchmark is included.

Local baseline is clean commit `dafe518`, which includes the preceding two
performance fixes and reports. Live runtime hashes on all three instances match
those accepted implementations; Hermes1 Git HEAD remains `074c981` with preserved
customizations/patches, Hermes2/3 are archive installations. Actual service
interpreter remains `/home/node/hymem-env/bin/python3`.

Fresh consistent SQLite backups were created on all three instances at
`/home/node/.hermes/backups/hymem-history-20260909-01/baseline.sqlite`.
All are schema 59, pass full integrity checks and have zero FK faults.
Production content remains on Afrodite; only aggregate diagnostics leave it.

Hermes1 has five drift findings involving two ASCII canonical identities: one
case-fold only, one CamelCase-to-snake-case plus case-fold. The initial shape
inventory's casefold flag did not establish case-only changes; the strict clone
preflight caught this distinction before any repair. One identity touches no
graph edges; the other touches 688, with 157 potential
natural-edge collisions. Both normalized targets already have self-mapping
aliases, not aliases to other identities. Affected graph evidence includes 125
current and 67 retired canonical rows, plus 1,007 current and 73 retired legacy
rows. This is a substantial consolidation, not a five-row cosmetic edit.
Hermes2 and Hermes3 have no normalization drift. Hermes3 now has 41 episodes,
one more than the prior rollout snapshot; its current state is preserved.

## Sequential plan and acceptance gates

1. A separate implementation agent audits existing canonical repair. Reproduce
   and fix concrete defects with synthetic regressions. Root reviews and
   independently tests before moving to the next issue.
2. Rehearse normalization in one outer transaction on a fresh working clone.
   Inspect exact graph collisions, evidence revisions/observations/lifecycle,
   auxiliary ownership, source records, aggregate invalidation and vectors.
   Require no unexplained semantic or provenance changes and rollback proof.
3. Audit historical source recoverability on clones using maintained validators.
   Restore only evidence that can actually be proved. Preserve remaining
   explicitly unproven/retired history.
4. Preflight the whole aggregation publication: current proof-valid sources,
   old/new membership, selected clusters and conservative request ceilings.
   Rehearse the maintained builder with the real producer, bounded deadlines,
   and no production writes. Verify complete publication and reuse behavior.
5. Only after independent acceptance, apply scoped repairs with fresh backups,
   appropriate writer/lease fencing or maintenance quiescence. Rebuild affected
   embeddings and aggregates under the actual service runtime.
6. Verify source/evidence invariants, full integrity/FKs, live MCP retrieval,
   extraction capability, producer identity and doctor findings. Report any
   irrecoverable history honestly rather than claiming a false green state.

## Progress

Initial read-only inventory complete. No production historical-data changes
have been applied in this phase.

The implementation agent reproduced a canonical merge bug: a normalized target
that is already an alias to another identity can receive rewritten graph state
while lookup still routes elsewhere. Root accepted the narrow fail-closed guard
after reviewing the implementation and seven new synthetic regressions and
independently running 173 canonicalization, provenance, bitemporal and auxiliary
ownership tests (all passed). This problematic alias shape is absent from the
actual Hermes1 targets. The guard has not yet been deployed.

Initial read-only aggregate preflight proves all existing leaf memberships are
still eligible: 27 on Hermes1 and 28 on Hermes3; no old leaf is lost. The
conservative upper bounds are 20/28 fusion operations respectively (40/56 logical
completions, 120/168 HTTP attempts at the existing retry ceiling). No provider
request or database mutation occurred. Hermes3 unexpectedly has zero effective
clustering vectors despite prior compatibility repair; investigate its exact
text/producer bindings before generation.

All three freshly resolved producers and public transport/runtime commitments
still agree with the preceding accepted producer. Hermes3's 28 episode mirror
text hashes/dimensions match, but their model/producer keys do not. Its stored
layout currently lacks all 451 previously reported conversions, while the 15
already-current vectors belonging to retired facts remain current. A stale-producer
writer, restore, or prior verification-target mistake is not yet distinguished;
do not claim a confirmed cause. A narrower process-age check found three MCP
processes, all using the dedicated runtime, and no recognized shared-venv
service. A broader subagent metadata probe was rejected by auto-review and did
not run; it was not rerouted.

Private code candidates now exist on all three beneath the new backup directory.
They are exact copies of the live HyMem package (including local customizations)
except the accepted canonicalization guard. The live checkouts are untouched.
Root additionally passed 151 maintained aggregation/provenance/generation/health
tests and three synthetic tests of the private whole-publication rehearsal
(source mutation refusal, pre-provider fusion ceiling, and full reuse).

Root reviewed both separate-agent private helpers and passed 41 tests (30 source
audit, 11 canonical rehearsal). The source-audit counter subsequently received a
two-test regression fix for double-counted raw-absent endpoints; root's final
32-test audit gate passed. These are helper fixes, not production migrations.
The canonical rehearsal first refused its overly narrow case-only precondition;
a follow-up shape-only audit proves both targets are self-mapped ASCII
identifiers and the only additional operation is the maintained CamelCase
boundary splitting. Trial01 made no history changes.

Private source-audit copies were created at `source-audit/baseline.sqlite`,
validated with the required SQLite functions, compared across all logical
tables and schema (130/105/130 tables respectively), and closed in DELETE journal
mode. Original backup/live databases remain untouched. Read-only audits found:

- Hermes1: 286 chunk mirrors have current source proof; 1,062 lack raw source
  endpoints and matching retained coverage; 28 use an unsupported historical
  short-session builder. All 7,951 edge mirrors have acceptable owners and all
  60 fact mirrors have current proof.
- Hermes2: no existing vector mirrors.
- Hermes3: 76 chunk mirrors have proof, 264 lack source endpoints; 169 edge
  mirrors have acceptable owners and eight have no open active direct owner.
  Sixty fact mirrors have current proof; the other 90 are non-current/retired
  facts, not evidence that needs reactivation. Preserve them as history.

No current-store endpoint candidate was reconstructible. A bounded, read-only
audit of four already-known older Hermes1 HyMem snapshots is complete; absence
in the current store does not establish absence in every backup.

Independent review of root's aggregation helper found two verification gaps:
it counted only LLM HTTP attempts and inferred leaf kind from raw IDs. Both are
fixed: separate LLM/embedding counters enforce zero calls on the second build;
ambiguous episode/node ID collisions refuse before leaf inference. Root and
the reviewer independently passed five helper tests; the reviewer additionally
confirmed an embedding call injected only on attempt two is rejected.

Canonical trial02 passed the corrected lexical precondition, attempted the full
normalization inside one outer transaction, and refused `evidence_set_changed`.
The complete logical snapshot was verified restored by rollback. This is a
diagnostic finding, not yet proof of a merge bug: independently inspect whether
the differences are valid branch consolidation/retirement or unintended loss.
No live repair and no production code update has been performed in this phase.
The immutable source-audit rerun confirmed 1,764 missing endpoint occurrences on
Hermes1 and 439 on Hermes3 (no longer double-counted); chunk classifications are
unchanged.

### Older-source and normalization diagnostics

The older-source audit helper passed 44 tests independently on root. Its actual
read-only run examined the known v31, v33, v35 and v46 snapshots, preserved every
input byte, and made no provider calls. The two oldest backups contain 228
distinct matching historical endpoints, supporting exact native rendering of
132 of the 1,062 target chunks. Another 930 still lack endpoints in these four
backups. There were no conflicting matched source identities/content/timestamps.
However, the older schemas lack three occurrence-metadata columns needed by the
current proof contract; missing fields were not guessed as NULL. Thus these are
132 partial recovery candidates, not 132 approved restorations. Version-aware
historical provenance review remains necessary. This is not an exhaustive audit
of every backup or an authorization to import sources across instances.

A separately reviewed counts-only normalization diagnostic completed on another
private clone, made zero provider requests, and always rolled back. It verified
source/protected tables unchanged, integrity OK, zero foreign-key faults and
the original backup unchanged. Both normalization operations completed, but
the proposed merge would consolidate 28 evidence rows (one canonical and 27
legacy) and retire 66 current canonical rows. Every deleted row has exactly one
same-state immutable-field match; 22 differ in extraction audit payload such as
extraction time or surface spelling. Five surviving rows also change extraction
time. No legacy evidence was promoted, no evidence was invented, and all edge
reassignments preserve the normalized natural key. These counts do not alone
prove that audit/history changes are legitimate. The strict preservation gate
remains closed pending branch-authority and audit-representative accounting.

Independent code review explains why the audit representative may change: the
maintained merge deliberately chooses one deterministic first occurrence for
an exact branch revision. The 22 discarded audit variants therefore cannot be
described as complete-row duplicates. The diagnostic matcher also omitted
revision because merges may renumber it; actual pre-merge revision equivalence
still needs proof. All lower-prompt retirement paths use the exact reason
`lower_prompt_authority`, whereas the 66 observed retirements classified as
other. Final extraction reconciliation can also retire rows lacking authorized
observation/outcome support (`successful_reextract:*`), but that explanation is
not established for these rows. Before approval, compare authority before/after,
prove the chosen audit representative and all observation/lifecycle mappings,
and either preserve alternate audit occurrences or explicitly settle that
retention policy. Do not weaken the verifier to make this clone appear green.

### Provider authorization gate

The proposed private aggregation run was rejected by the safety gate before
execution: explicit approval is required for sending backup-derived private
memory to the configured external LLM/embedding destinations. No request ran,
no LLM cost was incurred by that attempt, and its artifact directory was not
created. The action was not retried or rerouted. Source-validated episode
summaries/supporting facts would go to the configured DeepSeek API; texts needing
vector repair would go to the configured local embedding server. Explicit
payload/destination approval is required before either real-provider rehearsal.
Production historical repairs and the local canonical guard remain undeployed.

The separately implemented private Hermes3 vector rehearsal is reviewed and
frozen at SHA256
`17eb6603f47ffbf98c35d0e4119b1b86e607216ae915db9e370025ba7dd3d370`.
Root independently passed 78 tests (24 helper regressions plus maintained
re-embedding/cost neighbors) in 49.18 seconds using synthetic embeddings only.
The helper binds the exact producer and private target, checks all five repaired
mirror types against vector-index bytes, preserves source/history/FTS, and
reopens for a fresh read-only audit with the same producer. Its expected result
on the current Hermes3 snapshot is 451 repairs and 347 incompatible historical
mirrors retained, with zero eligible incompatible mirrors—not blanket health.
Actual execution is still unattempted pending the provider authorization above.

## 2026-09-09 continuation: explicit provider consent

The user replied “You may” to the explicit question covering backup-derived
summaries/supporting facts sent to the configured DeepSeek API and embedding
texts sent to the local embedding server, for bounded tests and subsequently
verified rebuilds, including private-memory disclosure and LLM charges. This
resolves the provider-authorization blocker above. It does not waive the
independent canonical evidence/history preservation gates.

The actual Hermes3 private vector rehearsal passed: 451 mirrors repaired in
two bounded segments, 19 local embedding HTTP requests, zero LLM requests.
Fresh read-only reopening confirms zero eligible incompatible mirrors, the
exact agreed producer, unchanged source/history across 97 protected tables,
and unchanged source backup. The 347 incompatible historical mirrors remain
untouched; 362 total proof-ineligible mirrors include 15 already-compatible
vectors attached to retired facts. This distinction is intentional. A whole
aggregation rehearsal now uses a further clone of this verified vector result.

Both real-provider whole-aggregation rehearsals passed. Hermes1 built nine nodes
from all 27 eligible episodes using nine LLM HTTP requests and one embedding
request. Hermes3 built eight nodes from all 28 eligible episodes using eight
LLM HTTP requests and no node-embedding request. Each immediate second build
reused every node with zero LLM or embedding calls. Both reopened publications
validate, all leaf memberships remain present, and 111 non-aggregation tables
remain unchanged in each private clone. These are clone results, not live fixes.

Historical code review confirms the older endpoint metadata cannot be inferred:
schema-31/33 Honcho ingestion discarded workspace-qualified peer information,
so native and external occurrences were indistinguishable in the old row shape.
Migration 43 expressly forbids guessing identities from roles, sessions or the
current peer registry. The 132 renderer matches remain incomplete-provenance
candidates; restoration as verified-native evidence is not justified. A trusted
occurrence-specific historical receipt would be needed to recover attribution.

### Completed authority explanation; production gate still closed

Root independently passed all 14 tests for the frozen counts-only authority
diagnostic (`d5ff5063248fb278d8cb3a46160a035d9c7564ba7e20b826cdf5d216e052a0e2`)
in 8.36 seconds, then ran it on a new private Hermes1 clone. All 66 retirements
used `successful_reextract:no_current_authority`. They already lacked matching
authorized observation/outcome support before normalization, on both exact and
normalized edge identities: 58 had outcome-binding mismatches and eight lacked
an outcome. All remained unsupported afterward; this diagnostic found no
loss of previously valid authority caused by merging. These are pre-existing
authority inconsistencies, not proof of a newly introduced merge regression.

All 28 deleted duplicate branches had a unique original-revision peer that
survived, with the correct deterministic audit representative and earliest
publication. Twenty-two alternate audit variants still would be consolidated,
so complete occurrence-level audit retention is not established. The diagnostic
fully rolled back, preserved source/protected tables, left the original backup
unchanged, and made zero provider calls. No historical normalization is applied.

A read-only full-table comparison showed Hermes3 live state has changed since
the original backup (including graph state, dream runs and aggregation state).
Schema is unchanged, but exact logical equality fails. Accordingly, adopting
the old repaired clone is prohibited; any production repair must use a fresh
backup and maintained in-place operations, preserving all newer changes.

The safety gate rejected creation of the proposed production-maintenance
orchestrator before execution. It requires explicit production approval for
stopping Hermes3 and writing the live store, separately from the unresolved
Hermes1 history operation. No container was stopped and no live repair ran;
the denied script was not created and was not rerouted. The implementation
agent paused immediately; its local application-helper draft is untested,
unreviewed and not deployable. All provider tests and clone diagnostics have
finished. Specific production-maintenance approval is the next gate.

## Explicit production approval

The user answered “Definitely” to the explicit request to briefly stop/restart
Hermes1–3, take fresh backups, deploy the tested guard, and apply verified
vector/aggregate repairs to the live stores while leaving historical
normalization untouched. This resolves the preceding production approval gate.
Proceed one instance at a time, with independent helper review/tests and fresh
source-preservation checks. Do not adopt an older clone or discard newer data.

Root independently passed 112 application/re-embedding/rehearsal tests on the
frozen in-place Hermes3 helper, plus 85 host-orchestration/guard-install tests
and eight subtests. Review caught and corrected unsafe restart admission,
cleanup of a daemon-side writer after its Docker client exits, private-output
forwarding, optimized-Python admission, broken output pipes interfering with
restart, and the six-element vector-report protocol. All were helper defects
caught locally before production. A combined guard/data maintenance integration
is being checked separately before execution.

All 428 known deployed code files still match on each instance. The guard's
loaded `normalize`/`resolve` hash matches the previous revision, confirming no
Phase-1 canonicalization identity change. The broader derived-fact producer
stamp may change because it includes the full canonical module; verify current
fact source/publication compatibility after restart rather than assuming reuse.

Fresh read-only baselines: Hermes1 has zero eligible incompatible vectors among
the five ordinary mirror types; only 1,090 unprovable chunk mirrors and five
old aggregate mirrors remain incompatible. Its 27 eligible episodes still need
a valid aggregate publication; five canonical-drift findings remain intentionally
unrepaired. Hermes2 has zero vector mirrors and zero aggregation-eligible
episodes, so it needs no vector or aggregate generation. Both stores are schema
59, pass integrity/FK checks and use the expected live embedding producer.

The final Hermes3 host/guard integration passed root's independent 113-test
gate and eight subtests. The frozen host orchestrator is
`19ab3e931f3cbb075b3e7c35eeffea6d154bec121d8ae6a287e58dc5f5235c2d`.
The explicitly approved Hermes3 maintenance has started: its normal container
was stopped and the sole maintenance writer admitted. This is a fresh in-place
repair, not replacement with the earlier clone. Final acceptance still requires
the helper's source-preservation receipt, guard verification, safe restart and
independent runtime/MCP checks.

Hermes3 production maintenance passed. Its fresh backup is
`/home/node/.hermes/backups/hymem-live-repair-20260909-01/baseline.sqlite`;
the old guard and receipt are in `hymem-guard-deploy-20260909-01` alongside it.
Exactly 451 mirrors changed (76 chunks, 118 messages, 169 edges, 28 episodes,
60 facts), using 19 local embedding requests. Eight aggregate nodes were built
from all 28 eligible episodes using eight LLM requests; the second build reused
8/8 with zero provider calls. All 83 protected source tables stayed unchanged.
The guard changed exactly one of 428 tracked files and preserved ownership/mode.
Maintenance-writer cleanup and restart admission passed; the container restarted.

Root's post-startup checks independently confirm schema 59, integrity OK, zero
FK faults, the expected live producer and working local embedding endpoint,
zero eligible incompatible mirrors, a valid eight-node aggregate publication,
zero canonical drift, and matching service configuration. MCP initialization,
tool discovery, profile and synthetic retrieval all passed. All 428 known files
still match on each instance with only Hermes3 using the new guard so far.
The 347 incompatible but unprovable/retired historical mirrors remain untouched;
this is not a claim that every historical doctor warning has disappeared.

The separate Hermes1/Hermes2 application helper passed root's independent
77-test combined gate (59.66 seconds). Its frozen SHA256 is
`f221e838e476ed94e2ec02aa4afe5538a32876f81001051eb99d883c8dfedebb`.
Hermes1 mode never invokes vector repair and preserves all five existing
ordinary vector domains, including a synthetic inventory over 4,096 mirrors.
Hermes2 mode creates a verified backup and audits without changing any database
row, constructing an LLM client or calling a provider. Both refuse changed
preconditions rather than expanding their scope. Host orchestration is a
separate pending acceptance gate.

Root's final secondary-host/primary-host/guard gate passed 220 tests and eight
subtests (0.57 seconds). The frozen secondary host SHA256 is
`dacb21fd3c29c5ba6d9c7b266087aedcd477a5aef361dfcb8a1617cb8c801b64`.
It admits exactly one instance per invocation, applies a typed instance/mode
receipt, and disables networking for Hermes2's audit-only maintenance.
Hermes3's subsequent read-only recheck after normal MCP activity again confirms
the repaired identities and publication persist. Hermes1 maintenance now starts
under the accepted secondary orchestration; Hermes2 remains untouched so far.

Hermes3's standard doctor was also run under the actual service environment
(this performs normal initialization, not a strictly read-only audit). Storage,
LLM configuration, sqlite-vec, embedding endpoint, schema, FK integrity, active
embedding identity and canonical drift all pass. Its one remaining FAIL is
stored embedding compatibility from the explicitly retained historical mirrors;
the separate source-aware audit still has zero eligible incompatible vectors.

Hermes1 production maintenance passed and safely restarted. Its new verified
backup is `hymem-live-secondary-20260909-01/baseline.sqlite` beneath the same
private backups directory. Nine aggregate nodes were built from all 27 eligible
episodes using nine LLM requests and one embedding request; the second build
reused 9/9 with zero provider calls. All five existing ordinary vector domains
and source/history stayed unchanged. No vector repair, history normalization
or automatic database restoration was invoked. The guard changed one of 428
known files and preserved ownership/mode. Post-startup MCP initialization,
12-tool discovery, profile, retrieval and Honcho health passed. Full-file hash
verification still passes across all instances (new guard on Hermes1/3).

Hermes1's independent read-only post-startup inventory also passed: valid
nine-node publication, all 27 eligible episodes, zero eligible incompatible
vectors, expected producer, integrity OK and zero FK faults. Source table counts
match the fresh pre-maintenance baseline. The 1,090 unprovable historical chunk
mirrors and five pre-existing canonical-drift findings remain untouched.
Doctor now reports active embedding identity OK, one stored-compatibility FAIL
for retained history and one canonical-drift WARN, with all other checks OK.
Hermes2's offline backup/audit/guard maintenance starts only after these gates.

## Final production result

Hermes2's maintenance passed with a verified fresh
`hymem-live-secondary-20260909-01/baseline.sqlite` backup, zero database-row
changes and zero provider requests. Its guard update changed only one of 428
known files. Safe writer cleanup and restart passed. Root independently verified
its actual endpoint/producer, schema 59, integrity/FKs, unchanged source counts,
zero incompatible vectors, no canonical drift, MCP initialization/profile/
retrieval and all nine doctor checks (zero failures/warnings). It correctly has
no aggregate publication because it has zero eligible aggregation episodes;
no data or publication was fabricated to fill this empty state.

The final cross-instance check verifies all 428 known files on each installation
with the tested canonical guard now present on all three. The accepted runtime
also contains both `dafe518` performance fixes; live Git/archive metadata is not
represented as a new upstream commit. All original tracked customizations and
untracked assets remain preserved. No historical normalization, source
attribution invention or stale-backup installation was performed.

All three maintenance receipts verify fresh backups, source/history preservation,
safe writer cleanup and successful restart. Active embedding identities and
endpoints pass; Hermes1/3 publications validate; all three MCP profile/retrieval
probes pass, and Hermes1 Honcho is healthy. The source-aware audits have zero
eligible incompatible vectors. Hermes1/3 still report stored-compatibility FAIL
for retained unprovable/retired history, and Hermes1 retains its five known
canonical-drift findings. These explicitly excluded historical issues are not
reported as solved or silently discarded.

The new canonical guard and its seven regression tests remain uncommitted in
the local worktree; include them in the next upstream release to preserve this
deployed fix. No full benchmark or new dense-extraction quality evaluation was
run in this continuation. The previous rollout's three real-provider synthetic
extraction canaries remain the extraction smoke-test evidence. The previously
exposed API credential was not rotated here; its operator must revoke/rotate it
if that has not already been done.
