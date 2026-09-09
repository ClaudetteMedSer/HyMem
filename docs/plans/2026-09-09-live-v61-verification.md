# Schema 61: live repair and verification

## Outcome

The approved release was rehearsed against fresh private copies, then applied
in place to Hermes2, Hermes3 and Hermes1, in that order. Each instance passed
its live service checks before the next was changed. All three are running
schema 61 with the reviewed runtime changes. No commit or push was made.

This is a verified service/repair outcome, **not an assertion that all historical
data has recoverable provenance or that future failures are impossible**.

| Check | Hermes1 | Hermes2 | Hermes3 |
| --- | --- | --- | --- |
| Schema | 61 | 61 | 61 |
| Canonical-drift findings | 0, previously 5 | 0 | 0 |
| SQLite integrity / foreign keys | OK / 0 faults | OK / 0 faults | OK / 0 faults |
| MCP handshake, 12 tools, profile, retrieval | Passed | Passed | Passed |
| Actual bootstrap remote embedding identity | Verified | Verified | Verified |
| Synthetic embedding endpoint request | Passed | Passed | Passed |
| Real synthetic extraction, `deepseek-v4-flash` | Passed | Passed | Passed |
| Valid published aggregate nodes | 9 | No eligible input | 8 |
| Honcho health | HTTP 200, healthy | Not deployed | Not deployed |
| Eligible incompatible stored mirrors remaining | 0 | 0 | 0 |
| Raw malformed / unverified vector counts | 0 / 0 | 0 / 0 | 0 / 0 |
| Historical source classification | Historical warning | Compatible | Action required: 8 unknown owners |

The last row is deliberately not hidden. A current-compatible vector means its
producer metadata and numerics match, not that every stored row is a current
authorized fact or that missing vectors have been created.

## What changed and why

### Canonical drift and collision preservation

The five Hermes1 findings represented two malformed canonical identities, not
five independent entities. The same identities were already present in schema
31, 33, 35 and 46 backups. They were noncanonical under the older normalization
policy too; this was not newly caused by Unicode normalization.

Old alias writes admitted unchecked targets. Existing malformed targets could
then propagate through mention indexing and derived-edge inference. Schema 60
adds SQL-NULL-safe write guards; the maintained writers normalize admitted
values, while unrelated edits of historical rows remain possible. Startup
validates/heals owned guards instead of silently permitting malformed writes.

Canonical collisions also exposed a preservation gap: distinct original
extraction variants could collapse into one representative. Schema 61 adds a
non-authoritative original-extraction audit. Merge and portable format 17 retain
the complete original occurrence set without counting that audit as evidence,
inventing past authority, or reviving retired claims. Exact manual signal/event
copies now coalesce together. Producerless portable formats 7–12 replay without
adding spurious evidence intervals or lifecycle dependencies.

The live Hermes1 repair touched 688 edges and removed all five findings:

- Evidence carriers: 20,623 to 20,595 through verified collision coalescence.
- Original owner occurrences: **all 20,623 preserved**, including 2,105 explicit
  audit rows; unchanged carriers remain their own original records.
- All 66 independently qualified current semantic claims were preserved.
- Five exact duplicate confidence signals coalesced; 145 surviving signal keys
  were remapped. Distinct payloads and manual signal/event pairing were retained.
- One exact duplicate lifecycle row and one observation coalesced. The verifier
  checked their mapped semantics, not merely their row counts.
- Sixty-six pre-existing unsupported current carriers were retired by the
  maintained reducer: 58 already had mismatched outcome binding, eight lacked
  the required outcome. No previously qualified current authority disappeared.
- 369 affected graph mirrors were created/rebuilt, using 12 embedding requests.
  All 7,811 live graph owners were indexed; the repeat was an exact physical
  no-op with no provider requests.

### Historical source loss

The older raw-pruning implementation removed messages based on a summary
without retaining sufficient exact source proof. Old backup schemas also lack
peer/workspace coordinates needed to establish occurrence-specific provenance.
Four backup audits yielded 132 renderable excerpts but **zero fully proven
restorations**. Text similarity and absent scope columns are not proof that a
message belonged to a particular native/null scope.

A separate current retention race was independently reproduced and fixed:
pruning now holds a writer transaction, uses savepoint-isolated rollback for
caller transactions, and revalidates exact timestamps and lease/deadline state.
This prevents that demonstrated maintained-path race; it does not prove the
race caused each older loss or recreate already missing metadata.

Source-aware diagnostics now distinguish eligible mismatches, retained-unverified
history, proven retirement and proven successful-empty source withdrawal. The
withdrawal proof requires complete retained receipts at the same producer
binding; a retracted flag or a matching prompt alone is insufficient.

### Unexpected local embedding fallback

Hermes3's vector changes match a completed dream at
2026-09-09 08:47:31–08:48:25 UTC. The stored old producer was independently
matched to the maintained LocalHash embedder. Its chunk/edge rewrite counts
match that dream, and the live database inode matches the earlier repair
receipt. The earlier repair had succeeded; the database was not replaced with
its old backup. The exact initiating caller/configuration failure for that dream
could not be recovered from available logs.

Two startup gaps were independently reproduced and fixed:

1. An explicitly configured remote endpoint could be rejected, lack credentials,
   or fail client construction and silently select a local producer. Startup now
   refuses before opening the store. Error reporting suppresses sensitive
   provider exception details and cleanup remains owned and tested.
2. Complete loss of embedding settings could still select the local default
   over a store containing remote vectors. A bounded read-only admission check
   now refuses that mismatch or unverifiable producer state. It observes
   committed WAL state, never initializes/migrates the store, and preserves
   deliberate local operation on fresh/pre-vector stores and explicit local
   re-embedding workflows.

The live Hermes3 repair rebuilt 447 eligible existing mirrors in 19 embedding
requests, then rebuilt eight aggregates in eight LLM requests. Repeating both
repairs required zero provider requests. The actual remote producer is shared
across all three instances. Its storage-key SHA-256 is:
`cea972cb2a6e3072ed56a3308efca4857b868323f710c166c6fcd1015a6081b4`.

Read-only fault injection against the actual stores confirmed that missing
configuration is refused on Hermes1/Hermes3, while Hermes2's empty vector store
retains the intentional local default. Explicitly rejected endpoints are refused
on all three. These probes changed only their own diagnostic-process environment;
they constructed no provider clients and performed no database row writes.

These are environment-bootstrap admission guards, not a global writer lock.
Direct API injection, intentional explicit producer changes, manual SQL or
running obsolete code remain operator responsibilities. All writers must retain
the same complete, verified environment.

## Verification evidence

Separate implementation agents were used for the fixes, followed by independent
root review and tests before accepting the next issue. The detailed sequential
record remains in `2026-09-09-historical-remediation-and-prevention.md`.

The initial three-shard integration run covered 5,571 cases: 5,521 passed,
29 canonical fixture/evaluation failures and 21 localhost sandbox socket
failures, with zero skips. The fixture/harness fixes and authorized localhost
reruns resolved these failures. Nine new harness regressions brought that
inventory to 5,580. The two startup fixes added 28 and 49 cases.

Final collection and XML testcase reconciliation show **5,657 unique passing
repository cases, no missing cases, and no unresolved failures/errors/skips**.
This is full-suite coverage plus verified reruns, not a claim that one initially
clean full run passed. Root independently passed the final 504-test and 553-test
startup/adjacent gates. The latter includes one additional scratch regression.
The installed FastAPI/Starlette httpx deprecation warning is not a test failure;
dependencies were not changed as part of this repair.

Private complete release rehearsals passed for all three, including actual
bootstrap construction on the repaired copies. Migration preserved all old
rows and added only the specified schema objects. Repair preservation,
rollback tests, repeat no-ops, reopen, foreign keys, FTS, publication and strict
vector/source classification gates passed. Root also independently checked the
one-off maintenance/host safety helpers, including failure and cleanup paths.

Live rollout used a fresh verified SQLite backup, a stopped normal container,
an isolated sole writer, exact code inventories, and in-place migration/repair.
No stale clone was installed, no automatic database restore was attempted, and
the source inode and backup bytes were verified unchanged. Only after complete
maintenance verification were 16 runtime files and 48 total release files
installed and the normal container restarted. Existing local customizations,
including Hermes1's LLM extra-body extension, were preserved.

The console-script MCP probe exercised actual startup, tool listing, profile and
retrieval. It kept the embedding client open through publication attestation.
An initial probe mistakenly closed that client too early and was corrected;
no production code or publication rule was changed to make that check pass.
Another probe initially compared raw Honcho/MCP environment dictionaries:
inherited credential aliases, explicit true/default-true aggregation flags and
the role-specific cooldown differed. Effective credentials, provider and
aggregation settings were then verified equal in memory without exposing values.

The three extraction canaries each returned a valid cited expected triple in
two real completion requests. They used only a newly authored fictional sentence,
performed no database access and sent no private memory. The older remote canary
was not displayed after safety review rejected possible sensitive-source exposure.
No full LME, BEAM or LoCoMo benchmark was run.

## What must remain visible

| Retained incompatible mirrors | Hermes1 | Hermes2 | Hermes3 |
| --- | --- | --- | --- |
| Unverified chunk provenance | 1,090 | 0 | 264 |
| Terminal source loss, subset of preceding row | 1,062 | 0 | 264 |
| Proven-retired fact mirrors | 0 | 0 | 74 |
| Unsafe-unknown graph mirrors | 0 | 0 | 8 |

Hermes1's remaining compatibility warning is historical, not an eligible
re-embedding backlog. Hermes3's eight unknown graph owners still lack complete
same-binding withdrawal/retirement proof; they remain excluded, not relabeled
as benign or silently deleted. Accordingly its source-health result remains
`action_required`. Closing these findings requires trustworthy original receipts
or source provenance that the inspected backups do not provide. Preserve the
backups and existing history; do not fabricate a green diagnostic.

## Recovery artifacts

Private artifacts remain on Afrodite in the respective instance's
`/home/node/.hermes/backups/` directory:

| Instance | Final full private rehearsal | Live backup, code inventory and repair receipt |
| --- | --- | --- |
| Hermes1 | `hymem-release-v61-rehearsal-_02ukigv` | `hymem-live-v61-hc665g4k` |
| Hermes2 | `hymem-release-v61-rehearsal-a1hytg93` | `hymem-live-v61-iq6agm_t` |
| Hermes3 | `hymem-release-v61-rehearsal-iaqmy109` | `hymem-live-v61-52miqspc` |

Each live artifact contains `baseline.sqlite`, `report.json`,
`code-before-manifest.json`, `code-after-manifest.json` and recoverable prior
release files. These schema-59 baselines are rollback evidence, **not instructions
to overwrite a now-active schema-61 store**. A recovery operation would need
fresh writer exclusion and an explicit plan for all newer data.

The frozen 59-member release archive SHA-256 is
`682135d304746ef67c4dcb75e12b625142704d4771eb1d2879129e42172652c3`.
Local reviewed scripts, manifest and test receipts are retained under
`/private/tmp/hymem-audit-review-root.WHciPj2Y/` and the referenced root XML files.
This final report was written after deployment; it does not alter the frozen
runtime bundle or imply that operational credentials were copied locally.
