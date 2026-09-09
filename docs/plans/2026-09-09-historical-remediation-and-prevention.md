# Historical diagnostics, preservation and recurrence prevention

## Final outcome, 2026-09-09

The sequential fixes and approved live rollout are complete. Hermes1, Hermes2
and Hermes3 run the reviewed schema-61 release and passed live MCP, embedding,
synthetic extraction and integrity checks; Hermes1 Honcho is healthy. Hermes1's
five drift findings are zero with all 20,623 original occurrences preserved.
Hermes3's 447 eligible mismatched vectors and eight aggregates were rebuilt.
The two additional embedding-startup gaps were accepted after independent
504- and 553-test gates. Final testcase reconciliation covers 5,657 passed
repository cases with no unresolved failures/errors/skips.

Historical missing-provenance records remain preserved, and Hermes3's eight
unsafe-unknown graph owners remain excluded and explicitly flagged. They cannot
be honestly restored from the inspected backups. See
`2026-09-09-live-v61-verification.md` for current results, root causes, limitations,
rollback artifacts and verification details. The progress entries below are a
chronological log; their earlier "no production changes" statements describe
those stages, not the final state.

## Authorization and baseline

The user requested all recommendations from the preceding assessment, including
root-cause diagnosis and prevention. Baseline is clean local commit `f88e5d5`.
The previous production rollout has all three services running with verified
active embedding identities, zero eligible incompatible existing mirrors and
valid Hermes1/Hermes3 aggregate publications. Historical findings remain:
1,090 incompatible unprovable Hermes1 mirrors, 347 incompatible historical
Hermes3 mirrors and five Hermes1 canonical drift findings spanning two identities.

Retain source/history and private data on Afrodite. Do not guess historical
occurrence metadata, relabel incompatible vectors, reactivate retired claims,
silently drop audit variants or replace live stores with stale clones. Remote
operations require fresh checks, private backups, rehearsal and root acceptance.
No full expensive benchmark is authorized by inference. Existing approval for
bounded necessary provider tests/rebuilds remains subject to exact source proof.

## Sequential implementation gates

1. **Source-aware diagnostics.** Separate active incompatibility, unverifiable
   legacy history and proven retirement without losing the raw stored inventory
   or misrepresenting missing-row coverage. Independent root review and tests.
2. **Recovery and loss prevention.** Trace historical pruning/ingestion and
   source metadata loss through repository history and existing private backup
   evidence. Revalidate trusted source candidates only; recover only complete
   proofs. Reproduce and fix any remaining maintained-path source-loss defect;
   add end-to-end retention/reopen/re-extraction tests.
3. **Audit-preserving canonical repair and prevention.** Preserve every affected
   source/audit occurrence and its mapping when identities collide; demonstrate
   that no previously authorized evidence is lost and retired evidence stays
   retired. Trace drift write paths, close concrete validation gaps, and test
   normal ingestion, import, merge and maintenance paths. Use two sequential
   acceptance gates here: new-write admission/propagation, then collision audit
   preservation. Each implementation has its own agent and root review. The
   source-withdrawal diagnostic regression found during private verification
   receives its own intervening implementation/acceptance gate.
4. **Integration and production.** Run relevant cross-domain and migration
   tests. Rehearse on fresh private clones, compare full preservation proofs and
   retrieval before/after, then deploy one instance at a time with fresh backups,
   appropriate writer exclusion and independently verified restart/runtime checks.

Use a separate implementation agent per issue; root independently reviews/tests
before advancing. Diagnosis may run alongside the current issue, but later
implementation changes must not overlap the accepted-gate sequence.

## Progress

Local instructions inventory found no applicable `AGENTS.md`. No production
changes in this continuation. Diagnostics implementation is accepted after
root's code review and independent **153 passed** gate (53.15 seconds), covering
source-aware health, raw recovery inventory and re-embedding. An independent
implementation agent fixed the reproduced retention race; root independently
reviewed all changes and passed **83 tests** (37.81 seconds), including root's
unchanged RED reproducer. The agent additionally passed **358 adjacent tests**
for lossless digest, claim/peer provenance and extraction retry. Standalone
pruning now holds a writer transaction; caller transactions retain ownership
with savepoint-isolated rollback, deadline/lease checks and exact timestamp
revalidation.

Canonical write admission is accepted. Root reviewed migration 060, startup
guard healing, maintained writers, manual API validation and historical fixtures.
Root's expanded gate passed **299** tests and exposed two malformed ordinary
attestation fixtures plus 27 schema-59-only quarantine fixture errors. After
targeted fixture corrections, root independently passed **148** integration
tests (70.09 seconds). The implementation agent separately passed 235 canonical,
319 adjacent and 166 historical/attestation tests. The production one-off
quarantine helpers retain their schema-59/runtime-59 safety restriction; explicit
CLI tests verify current code still refuses them. Tests simulate the historical
contract without claiming to run a frozen old executable. Benchmark material
attestation now compares against the runtime's expected schema instead of
hardcoding 59. No live deployment yet. A new agent is implementing the proven
source-withdrawal diagnostic arm next, before collision audit preservation.

Root independently reviewed the withdrawal implementation and passed **210
tests** (99.45 seconds), including both reembed suites after narrow historical
fixture corrections. The maintained positive-to-empty reproducer now passes.
The private Hermes3 pre-repair backup still has eight unknown graph mirrors:
they do not yet satisfy the new complete proof. Do not relax the contract or
claim those rows are proven withdrawn; trace the exact failed check.

A fresh private Hermes1 clone passed schema **59 to 60**, all existing data rows
unchanged, six new canonical guards, integrity and foreign keys clean, repeated
initialization/reopen a no-op. No live code/data writes or provider requests.
Artifact: `hymem-v60-rehearsal-8e7rgxe6`. The first rehearsal stopped before any
database access because macOS tar included AppleDouble metadata not in the
exact code allowlist. Rebuilding with metadata disabled resolved that harness
failure; the strict archive admission was retained. Its rejected candidate is
preserved separately as `hymem-v60-rehearsal-zxar1rvp`.

The eight private withdrawal proof failures are specifically an earlier retained
retirement timestamp than the current winning same-prompt receipt. Retirement
reasons, authority tuples, registry bindings, prompt generations and publication
clock validity all match. The implementation agent is reproducing maintained
same-contract repeat/overlapping-chunk publication before deciding whether this
is a diagnostic false negative. No history has been modified to satisfy a proof.

Maintained four-case reproduction confirmed a narrower false negative: a later
overlapping empty chunk at the same exact binding may win without changing the
original retirement or lifecycle. Same-chunk/same-binding replay normally leaves
its clock unchanged. A changed producer at the same prompt can also leave an
earlier retirement, so replacing timestamp equality with an inequality alone is
unsafe. The scanner now requires a retained, independently complete original
receipt with identical producer binding and prompt for the later-overlap case.
Agent's final gate: 167 passed. Root's unchanged maintained reproduction now
passes both legitimate cases and retains both changed-prompt/producer failures.

Private Hermes3 has **zero** matching original same-binding empty-clock witness
candidates for its eight owners. The final diagnostic correctly leaves all eight
unknown; five reach the missing-original-witness guard, while three fail the
current winner's complete-empty proof first. No history/vectors were edited, no
provider requests made, and the backup remained byte-identical. This establishes
the proof gap, not the exact historical writer or a reason to weaken validation.

Root accepted the complete withdrawal gate after **221 passed** (120.64 seconds)
plus independent four-case maintained replay and the unchanged private-backup
check. Final health module SHA-256:
`c1d9b2c0e1029f6e583e1c3953c583d49799d5191b21967c3e7f29832d0a8326`.
A new agent is now implementing original-extraction audit preservation. Root's
pre-audit source snapshot is
`/private/tmp/hymem-audit-review-root.WHciPj2Y/accepted-pre-audit.tgz`.

The extraction-audit implementation is frozen for root acceptance. Root reviewed
the storage, migration/reopen admission, transfer/union, portability/redaction,
source pinning and actual opt-in tombstone pruning paths. Its independent gate
passed **383 tests** (297.49 seconds), followed by **233 migration/provenance/
bitemporal/Unicode tests** (131.82 seconds), zero failures/errors/skips. Agent's
final gates passed 136 and 210 tests. Review corrected NULL-accepting SQL checks
and an overly broad `INSERT OR IGNORE`; exact-key conflict handling now preserves
validation failures. There is no general public forgetting API in this checkout;
the maintained destructive path tested is `prune_retracted_edges` followed by
source pruning.

Root's independent private verifier passed 19 synthetic tests, including current
canonical authority, exact original multiplicity, corruption rejection and
fault-injected rollback. A first fresh private schema-59-to-61 rehearsal passed
the migration but stopped before repair because the verifier incorrectly assumed
the citation projection included all extraction fields. It now fetches complete
carriers only for independently qualified IDs. A subsequent private rehearsal
passed original-occurrence and current-authority comparisons, then refused a
confidence-signal multiplicity change. Its transaction rolled back. Determine
whether this is exact branch-copy coalescence or actual loss before adjusting
the verifier or authorizing live repair. Production remains unchanged.

An additional pre-existing portability defect was reproduced independently:
exact second replay of producerless formats 7–12 adds an evidence interval and
lifecycle dependency. Root's six-case regression is RED (7.36 seconds); the agent
also verified the same six failures using the accepted pre-audit importer.
Reproducer: `/private/tmp/hymem-pre-v13-replay.a85f3S/test_producerless_replay.py`.
No xfails or authority-policy changes were hidden in the audit implementation.
After the current gate is accepted, use a new implementation agent and separate
root verification for this reducer defect, before full integration/deployment.

Root accepted the extraction-audit implementation gate (383 + 233 independent
tests and the private original-occurrence/current-authority comparisons); the
whole production repair remains unaccepted until its complete private rehearsal
passes. Count-only diagnosis of the signal mismatch proved five exact cross-edge
copies coalesced, 145 surviving keys rekeyed, zero inserted signals, zero changed
surviving payloads and zero deleted manual pairs. The verifier now checks original
keys, exact copy identity, specified rekey hashes and paired manual events instead
of requiring unchanged duplicate multiplicity. Expanded verifier gate: 26 passed.

That diagnosis independently uncovered a separate pre-existing manual-pair bug:
two byte-identical signal/event pairs made with `record_signal()` defaults merge
to two signals but one lifecycle event. Early lifecycle clock normalization makes
the later raw-pair comparison falsely classify the duplicate as a conflict and
orphan its signal. Root's maintained regression fails in both merge directions
(2 failed, 1.00 second). A new implementation agent, `fix_manual_pair_coalescence`,
owns that narrowly bounded fix before the producerless-replay gate. Do not admit
orphaned manual signals, erase distinct clock spellings, or weaken history checks.
No production deployment or provider requests in this continuation.

The complete private rehearsal subsequently passed on fresh Hermes1 artifact
`hymem-v61-rehearsal-pydpzrkh`: schema 59→61 preserved all pre-existing rows;
normalization removed all five findings across two identities and 688 touched
edges. Evidence carriers went 20,623→20,595 while all **20,623 original owner
occurrences** remained represented (2,105 explicit audit rows; unchanged carriers
remain their own original record). All 66 currently qualified semantic claims
were unchanged. One duplicate lifecycle and one duplicate observation coalesced
without losing mapped semantics; five exact confidence-signal copies coalesced,
145 keys received verified collision suffixes, and surviving signal payloads were
unchanged. All source rows/vectors remained unchanged; integrity/FK checks,
reopen/repeated repair, baseline-byte preservation and injected full rollback
passed. No provider requests or live code/database writes.

The 66 evidence rows newly marked noncurrent during merge were already unsupported
under both exact and normalized authority before repair (58 mismatched outcome
bindings and eight missing outcomes). None lost previously qualified authority;
none was revived. This is distinct from the 66 valid current semantic claims.
After the separate manual-pair and producerless-replay fixes, re-pin the final
candidate and rerun integration/private gates before actual deployment.

Manual-pair preservation is accepted after root reviewed both changed files and
independently passed **279 tests** (205.64 seconds), including unchanged root2,
all 73 new cases, original-extraction root12, the private verifier, portability,
canonical merge and bitemporal history. Agent separately passed 423 distinct
cases. Bound manual events now move/coalesce atomically with their signals;
original raw clocks and distinct identities remain intact. Same-call three-way
copies are proven from original snapshots. A cross-call occupied generated suffix
with unavailable original-key provenance fails explicitly and rolls back rather
than guessing identity; this pre-existing ambiguity is not presented as repaired
historical provenance. Final evidence SHA:
`614d95e3a4ac2f9ab62ad42d09fa809dfc96dbd99f1a7ed313641eb46ebcb309`.
New agent `fix_producerless_replay` now owns the final independently reproduced
older-format replay issue. Root is preparing bounded post-normalization graph
embedding/index rebuild verification; merely leaving the renamed graph's old
text-keyed mirrors untouched would not establish working vector retrieval.

Producerless replay is accepted after root's independent **263 passed** gate
(248.01 seconds), including unchanged six-case RED reproduction, all 25 new
cases, portability, extraction audits, bitemporal history, evidence/manual pairs
and original-occurrence preservation. No failures/errors/skips. The defect added
one evidence revision and one assertion lifecycle per replay; dependencies did
not grow (earlier shorthand calling this a dependency was imprecise). Prompt-only
observations now cannot authorize a new interval; separately declared same-ms
revivals remain distinct. Final importer SHA:
`4917d7b5d7bf7a8fac45a8d02f21eeb9b7c698e9f06f4c57d3cf7825776548c7`.
The agent additionally passed 254 contract and 100 authority tests, then 31 final
tests after a set-membership optimization. A separate mixed-version arrival-order
probe differs in local retired-carrier count but retains identical qualified
authority, confidence and original payloads at both baseline and candidate; this
alone is not established as another bug and was not changed.

Root's bounded graph-index/aggregate operational helpers passed **14 tests**.
Only current maintained graph selection can be embedded; actual producer and
request meters are checked, renamed text mirrors are generated (not relabeled),
all unrelated source/history rows remain exact, and repeated graph rebuild is a
physical no-op. SQLite's two vec_edges AUTOINCREMENT sequence records are checked
separately; unrelated sequence modifications trigger full rollback. All 181 test
files are now running in three disjoint integration groups against frozen source.
Fresh final private schema-59-to-61 rehearsals are running for all three instances;
no live code/database writes or provider calls yet in this continuation.

### Final integration and private acceptance

All three final private rehearsals passed against frozen runtime code:

- Hermes1: `hymem-v61-final-dtfhsvwb`; five findings removed, 20,623 original
  occurrences preserved, 2,105 explicit audit rows, 66 qualified current semantic
  claims unchanged, full rollback/reopen/idempotence verified.
- Hermes2: `hymem-v61-final-cqn0dmg6`; 664 evidence occurrences preserved, no
  drift or repair changes, migration/reopen/rollback verified.
- Hermes3: `hymem-v61-final-_023p0ab`; 429 evidence occurrences and 76 qualified
  current semantic claims preserved, no drift or repair changes, migration/
  reopen/rollback verified.

Every existing row was unchanged by migration itself; each original backup
remained byte-identical. The final runtime archive has 14 production files and
seven code-only verification helpers; SHA-256:
`150d92ce24fb02e96871526d7761f46d4a6014a4338d9d0dad23af46983c287f`.
Its local manifest is
`/private/tmp/hymem-audit-review-root.WHciPj2Y/final-private-manifest.json`.

The full integration run covered **5,571** cases in three disjoint file groups,
with source hashes unchanged throughout: 5,521 passed, 29 canonical-admission
fixture/evaluation failures and 21 sandbox localhost socket failures; zero skips.
The seven embedding-server tests and 16 real-SDK/localhost Honcho tests were
independently rerun with socket permission: all passed. No external providers
or real credentials were used by those local tests.

A new agent (`fix_coref_eval_admission`) fixed the 29 related integration cases.
The coreference eval now normalizes only input graph endpoints into canonical
columns, retaining strict insertion/collision errors and using a real transaction
with close-on-failure. Ordinary raw-SQL fixtures now supply canonical spellings.
Shared test helpers, resolver code, input conversations, eval answers and gate
thresholds were not loosened. Nine new regression cases cover normalization,
collision/invalid-input rollback, connection closure and exact graph/control
behavior. Agent gates: 104 + 220 passed. Root reviewed all seven changed files
and independently passed **386** tests (155.66 seconds), including all affected
files, admission/Unicode, augmentation and benchmark identity checks. Coref CLI:
31/31 resolutions and zero of 12 control rewrites. Final coref harness SHA:
`d13c1068d7e9598ced69ac4b4654d5cfbb50cc191d94ec9575e1c23b60c40d64`.

Reconciling testcase IDs from the full run and these final independent reruns
gives **5,580 unique passed cases, zero unresolved failures/errors/skips**. This
is full-suite coverage plus verified reruns, not a claim that the initial full
run was green. The 14 runtime files remain unchanged by this harness adaptation.

### Provider approval boundary and fresh Hermes3 mismatch

Auto-review rejected the private provider-backed rehearsal before execution:
it could transmit private database-derived material to configured remote
providers without explicit approval for that particular data/destination scope.
No provider request occurred, no live code/database changes or restarts have
occurred in this continuation, and this rejection must not be bypassed.

Network-disabled, read-only alternatives confirmed Hermes1's 1,090 retained
unverified chunk mirrors (1,062 terminal source losses). They do not prove missing
mirror/index coverage after normalization; that still needs the bounded rebuild.
Hermes3's serving configuration resolves to the intended exact producer, but
its current live store and final private copy still have fallback-producer
metadata and all 177 old graph mirrors. The private source-aware audit finds
**447 eligible incompatible mirrors**: 76 chunks, 118 messages, 169 graph texts,
28 episodes and 56 facts. Its other incompatible rows are 264 unverified chunks,
74 retired facts and eight still-unproven graph histories. Preserve those;
do not relabel or declare them repaired. Counts are snapshot-specific and must
be freshly revalidated under the live maintenance fence.

This contradicts relying on the earlier successful-repair state as still current.
The retained Hermes3 receipt records 451 repairs and zero eligible mismatches at
2026-09-09 08:15:46 UTC; its three current MCP processes started afterwards
(08:15:52, 08:15:56 and 08:42:53 UTC), so survival of a pre-restart MCP process is
not established. The bounded container-log tail contained no recognized fallback
warning signatures. The exact intervening writer/restore event is **not yet
established**; do not invent that cause. Current diagnosis is sufficient to refuse
the previously prepared Hermes3 zero-pending graph-only plan.

Next approved work must first rehearse a fresh, bounded existing-mirror repair
for Hermes3, then graph shadows and aggregate publication, and prove the state
stays correct through the actual serving/dream paths. The current
`run_private_derived.py` is NOT ready to rerun unchanged: its Hermes3 branch assumes
current shadow metadata and zero pending graph mirrors, assumptions disproved by
the offline check. Its full-environment equality also needs narrowing to the
actual provider/configuration keys (three MCP processes differ in incidental
environment variables). The corrected offline reader compares only relevant
configuration and keeps all credentials in memory.

Approval received in the user's subsequent “Please do.”: task-necessary graph/source text to the configured
local `http://embedding-server:8766/v1` embedding service, and episode summaries
plus supporting stored memory to `https://api.deepseek.com` using
`deepseek-v4-flash`, for bounded private rehearsals and the subsequently verified
in-place live repairs. Reconfirm fresh source/candidate admission and bound calls;
no full LME/Beam/LoCoMo benchmark is implied. Approval resolves the prior provider
boundary; it does not waive private rehearsal, fresh admission, or preservation
checks. Leave live services/stores intact until those gates pass.

### Root-cause evidence and reproduced prevention gap

Approved provider rehearsals now passed on new private copies, not production:
Hermes3 `derived-_i88rxza`: 447 eligible mirrors (76/118/169/28/56), 19 embedding
requests; 8 aggregate nodes rebuilt using 8 LLM requests; repeats used zero
providers. Historical exclusions remain 264 chunks, 74 retired facts, 8 unknown
edges. Hermes1 `derived-zmkt0cmm`: 369 new normalized graph mirrors, 12 embedding
requests, 7,811 live graph owners indexed; all 9 aggregates reused, zero LLM
requests, repeat exact graph no-op. Hermes2 `derived-8hakcqmn`: empty vector
inventory, zero provider requests. All three passed source/history preservation,
zero canonical drift, schema61 reopen no-op, integrity/FK checks and unchanged
private input bytes. Schema61 mirror-helper independent gate: 24 passed.

Fresh Hermes3 turnover diagnosis rules out nested Docker mounts and confirms
the live database inode/device matches its prior maintenance receipt. Its dream
at 08:47:31–08:48:25 UTC rewrote 76 chunk and 169 edge mirrors; precisely those
rows have changed rowids/creation timestamps while their producer and vectors
match the old fallback baseline. Across all 24 later dreams, this is the sole
nonzero embedding-counter run. This is ordinary writer turnover, not evidence
that the earlier repair never occurred or that a database file was replaced.
Exact startup/configuration cause is still under investigation. Root reproduced
two independent startup-fallback branches offline: rejected explicit remote
endpoint or constructor failure installs a persistent local client that can
overwrite valid existing vectors. Separate implementation agent is fixing this
confirmed prevention gap; root review/gate is required before deployment.

Historical commit `07a60ae` pruned raw messages from old sessions based only on
the existence of a summary, without lossless source proof. Its alias writer
also accepted arbitrary canonical targets. These code paths explain how exact
sources could be discarded and mixed-case canonical targets accepted before
the current contracts; exact per-row writer attribution still requires private
chronology evidence and must not be invented.

Current raw retention defaults off and validates exact per-message coverage.
The normal dream runner encloses it in a writer transaction. Root nevertheless
reproduced a current standalone race in a synthetic two-connection test:
`prune_messages` on an autocommit connection validates a releasable coverage
artifact, a concurrent maintained `release_message_coverage` operation removes
it while raw still exists, then pruning deletes raw too. Both source copies are
gone. Regression is RED (1 failed, 0.61 seconds), at
`/private/tmp/hymem-history-prevention-root.FMejlT/test_retention_admission.py`.
This is a concrete prevention defect, not a claim that it caused the observed
pre-v37 historical losses. Implementation follows acceptance of diagnostics.

The earlier complete four-backup recovery audit remains the source-availability
result: 132 extraction excerpts could be rendered from matching old message
endpoints, but **zero** have complete occurrence-specific provenance. Schema
31/33/35 lacks peer/workspace source coordinates; absence of those columns is
not evidence of native/null scope. No trustworthy occurrence-specific receipt
has been identified to fill that gap. Do not restore those candidates or mark
their source manifests canonical. Retain the historical artifacts and warnings;
the atomicity fix prevents the separately reproduced current source-loss race,
but cannot reconstruct metadata the older system never retained.

Afrodite instructions were reread. They require stopping all work while a user
question/approval is pending; none is pending. No production writes in this
continuation.

Private read-only chronology confirmed the same two drifted identities already
exist in the known schema-31, 33, 35 and 46 backups, before the recent upgrades.
They are noncanonical under the old ASCII/CamelCase normalization policy too,
so this is not a newly introduced Unicode-normalization mismatch. Per-location
occurrence counts (alias target, mentions, graph subject, graph object) are
`[1,299,554,75]`, `[1,300,549,73]`, `[1,306,552,79]`, `[1,288,593,102]` and
`[1,303,587,102]` in the retained schema-59 audit baseline. A graph row can count
in both endpoint columns. All five backup bytes were verified unchanged; zero
database writes/provider requests. Schema-31/33/35 message rows lack the external
scope columns, while schema-46/59 have them. This dates the defect's existence,
not the exact historical caller that first wrote it.

Root independently reran the canonical diagnosis against a synthetic current
store: mention indexing and derived-edge inference both propagate existing
invalid canonical values into new rows. A collision also reduced two distinct
evidence audit variants to one representative without retaining the alternate.
These are confirmed present-day recurrence/preservation gaps. The canonical
fix must retain non-authoritative occurrence snapshots; adding confidence
signals is unsuitable because those can accidentally become counted authority.

The accepted diagnostic also ran against two named private, closed pre-repair
baselines with networking and initialization forbidden. Both files remained
byte-identical. Hermes1: 1,090 retained-unverified chunk mirrors (1,062 terminal
source losses) and the five then-unrebuilt aggregate mirrors correctly requiring
rebuild. Hermes3: all 451 then-repairable ordinary mirrors classified eligible,
264 retained-unverified chunks, 75 proven-retired facts, and eight unsafe-unknown
edge mirrors. These are historical inputs, not a new live-health claim; the
451 vectors and five aggregate mirrors were repaired in the preceding rollout.
The eight unknown graph owners are explicitly retracted with valid/invalid
clocks and lifecycle rows, but no current evidence or claim observations. They
still fail the maintained lifecycle proof; status flags alone must not downgrade
them to proven retirement. Investigate their exact proof failure before deciding
whether they have a supportable historical classification or need quarantine.

Further diagnosis found a documented alternate closure path: successful empty
re-extraction withdraws source authority, rather than asserting a negative fact.
The finalizer intentionally closes the cached edge with no negative lifecycle
event. A maintained synthetic positive-to-empty replay reproduces the private
shape. The diagnostic therefore needs a separately proven withdrawal arm; this
does not authorize treating arbitrary retracted flags as proof. Verify the
private rows' retired revisions, clocks, source manifests and winning empty
outcome before accepting that classification. Implement only after the current
canonical-admission gate, then independently retest.

### Extraction-occurrence preservation contract

Use one non-authoritative extraction-audit table, owned by the surviving evidence
carrier. Its immutable original edge tuple and extraction/source/surface/prompt
fields retain original spelling. Exclude local handles/revision ordinal and
mutable publication/retirement state, which remain in the existing evidence and
lifecycle ledgers. Original snapshots never count as confidence or authority.
Capture an unaudited carrier before topology changes or destructive coalescence;
an already-audited carrier's set is complete and must not absorb its synthesized
representative as another original. Union sets and remap through evidence IDs.
Keep chunk and coverage restrictions, explicit forgetting/deletion behavior,
portable remapping/redaction, typed strict JSON/hash validation, and idempotence.
Migration creates empty storage, not invented historical occurrences.

Root authored a separate three-branch merge regression covering all six merge
orders. It proves loss of original edge spellings and extraction variants:
**6 failed** (2.71 seconds), before audit preservation implementation. Scratch:
`/private/tmp/hymem-history-prevention-root.FMejlT/test_original_extraction_preservation.py`.

The same root-owned file now contains an independent portable three-branch
regression: **six additional failures** (16.54 seconds), one for every import
order, preserving original source chunk/surface/prompt/time tuples and exact
replay. Existing export normalizes `extracted_at` before writing the wire; the
new format must carry original audit data before that transformation as well.
This may require read-only, virtual original-audit records during export for
unaudited carriers, not mutation of the exporting database. Do not create local
audit storage on an otherwise exact no-op self-import when the original is
already represented unchanged by the existing carrier.

Implementation details for the collision gate:

- Suggested storage is `kg_evidence_extraction_audit` with `evidence_id`,
  `occurrence_hash`, canonical `payload_json`, and typed source projections.
  Payload is a versioned domain plus `original_edge` and `extraction`. The root
  regression reads these names. Composite ownership/hash primary key permits
  distinct genuine revisions to carry identical original payloads.
- Extraction fields: `chunk_id`, `polarity`, `surface_subject`, `surface_object`,
  `value_text`, `value_numeric`, `value_unit`, `temporal_scope`, `source_role`,
  `source_peer_id`, `source_workspace_id`, `evidence_kind`, `evidence_weight`,
  `weight_source`, `extraction_prompt_version`, `extracted_at`,
  `source_message_id`, `source_session_id`, `source_created_at`, `source_event_at`,
  `source_coverage_chunk_id`, `source_coverage_version`, `provenance_status`,
  `interpretation_key`. Retain original spelling/time strings, not normalized
  replacements. Strict scalar/JSON/hash validation must not invent valid clocks
  for malformed historical text. Invalid scalar types may fail closed without
  changing the original rows.
- Evidence ownership cascades on explicit deletion; source chunk and nullable
  coverage tuple use restrictive references. No raw-message foreign key: normal
  lossless raw pruning must work. Audit-only source chunks are pinned before
  retention touches vectors or manifests. Internal history authorization guards
  inserts/transfers; explicit destructive authorization must also permit deletion.
- A carrier with an audit set represents the complete original set. Never
  append its later reduced representative. Capture both collision sides before
  reduction, and any edge before noncollision topology changes. Preserve genuine
  separate revisions and the existing authority reducer unchanged.
- Portable format is currently **16**, so the next audit format must be **17**.
  Parse/validate incoming original sets before evidence collision processing;
  use the wire evidence map to transfer originals, not post-reduction carriers.
  Explicitly traverse/redact payload text, remap source references and recompute
  canonical hashes; deduplicate only identical transformed payloads. Preserve
  old format schemas and exact no-op reimport behavior.
- Migration **061** owns empty storage; schema.sql has a comment, not additive
  table creation. A stamped current store missing/forging storage must fail
  before bootstrap can silently recreate it. Restore only owned guards/indexes
  transactionally after exact table validation. Update explicit benchmark
  material-table attestation policy for the added durable audit table.
- Verify all six three-way merge and portable import orders, noncollision
  rename, redaction/reimport, same-millisecond revivals, no authority inflation,
  rollback/idempotence, source restrictions/raw retention/explicit forgetting,
  and migration/reopen/forged-storage behavior. Migration must not fabricate old
  audit rows. Fresh private canonical rehearsal then proves original-set and
  qualified-authority preservation before any production repair.
