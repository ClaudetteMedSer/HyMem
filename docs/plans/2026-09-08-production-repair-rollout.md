# Authorized production repair and sequential rollout

## Authorization and baseline

The user approved production application of the independently rehearsed orphan
quarantine, verification afterward, then updating Hermes2 and Hermes3 to the
newest HyMem version and fixing verified bugs encountered during verification.
This is not authorization to fabricate source history, erase additional
unrecoverable legacy data, rotate third-party credentials or run costly full
benchmarks.

Published `Beam-optimisation` revision verified using `git ls-remote`:
`074c981f094d4fddc9d3a7a9ada2da818f08f890`. The local tracked runtime initially
matched that commit; the two accepted performance patches below remain
uncommitted, together with their regression tests and operational reports.

Initial read-only inventory: all three instances run the older `b360b82` code
(Hermes2/3 are archive installations without Git metadata). All stores are at
schema 59 with WAL journaling and pass SQLite quick-check. Hermes1 has the
known single chunk FK fault; Hermes2/3 have none. Hermes1 has five tracked
local edits and additional untracked benchmark artifacts, which must survive.
All instances have matching internal embedding endpoint, dimension, deployment
revision and tenant settings. Honcho runs only on Hermes1.

## Sequential acceptance plan

1. A separate agent implements a narrowly guarded, locked production quarantine
   helper with durable native archive/full-baseline backup and rollback tests.
   Root reviews and independently verifies it before remote execution.
2. Recheck Hermes1's exact orphan and dependency state, preserve a fresh durable
   recovery backup, apply the minimum transactional quarantine and independently
   check database, keyword/vector indexes and runtime health.
3. Rehearse the published update on copies; preserve local customizations and
   code/database rollback artifacts. Update Hermes1 as needed for the new
   recovery/diagnostic fixes, then verify it before starting Hermes2.
4. Update Hermes2 alone, restart through its established container lifecycle,
   verify code, environment, database/migrations, MCP and provider probes.
5. Repeat for Hermes3 only after Hermes2 acceptance. Diagnose stale embeddings
   with source-authorized bounded recovery; never relabel incompatible vectors
   or manufacture canonical provenance to obtain a green status.
6. Delegate each newly verified code defect to a separate implementation agent,
   independently verify its fix, then resume the affected rollout stage.

Production content and credentials remain on Afrodite. Operational tools emit
only safe counts, booleans, hashes and reviewed paths. Reports must distinguish
healthy runtime behavior from unrecoverable historical data or incomplete
coverage; a perfect-state claim is not an acceptance criterion that overrides
the evidence.

## Results

All three instances have been updated and the bounded recovery sweeps and
post-recovery checks are complete. Services work, but historical incompatibility
remains explicitly reported; this is not a claim of three perfectly clean stores.

### Hermes1 production repair and base update

- Applied the locked, independently rehearsed quarantine of the single orphan.
  The exact delta was one chunk, one chunk embedding, four entity mentions and
  one logical vector, together with the chunk's FTS membership. No unrelated
  logical projection or schema changed. Root's independent checker verified 89
  projections, native archive equality and full-baseline restore equality.
  Integrity check passed and foreign-key violations fell from one to zero.
- Recovery artifacts are preserved at
  `/home/node/.hermes/backups/hymem-orphan-quarantine-20260908-production/`:
  `baseline.sqlite`, `quarantine.sqlite`, and `baseline-restored.sqlite`.
  Restoring the full baseline also restores the original known FK defect;
  archive-only reinsertion was not claimed or tested.
- Fast-forwarded Hermes1 to `074c981`, preserving all five tracked local edits
  and its untracked artifacts. Restarted through its normal container lifecycle.
- The old runtime's setuptools 66.1.1 could not build the updated editable
  package. Built an editable wheel using isolated setuptools 83.0.0 and wheel
  0.45.1, then installed only that wheel without dependencies. Recoverably moved
  stale generated source-tree `hymem.egg-info` into the rollout backup because
  it shadowed the new entry-point metadata. Five generated egg-info files are
  included in the published archive: their original hashes are now verified
  in the recovery archive, not misreported as unchanged in the live tree. The
  shared runtime dependencies were not upgraded; import origin, metadata and
  `hymem-reembed --help` now pass.
- Embedding endpoint and one-token LLM connectivity probes passed. Stored
  embedding audit still reports 10,002 incompatible vectors; recovery has not
  yet been applied at that initial probe. Historical source authorization must
  not be bypassed.

### Verification and newly found retrieval defect

The base-tree full suite completed with 5,089 passes, five failures and 16 setup
errors, all 21 caused by sandbox-denied loopback binding. The affected files
passed independently with loopback permission (7 and 16 tests respectively).
Thus all 5,110 collected cases passed across the full run and permitted reruns,
not in one uninterrupted green invocation. The final repair-helper gate also
passed all 44 cases. These runs predate the two performance fixes below.

Initial live MCP initialization, tool listing and profile calls passed, but
augmentation exceeded 90 seconds. A private-copy profile reproduced over 2.7 million timestamp
checks in 60 seconds inside the persisted token-overlap query before retrieval
could begin. A separate agent implemented a one-statement materialized
authority-set query, preserving the complete live-evidence predicate.

Root reviewed the two-file patch and independently reran 263 cases: all passed
in 172.18 seconds. A private-copy profile measured token-index construction at
0.44 seconds and augmentation at 4.4 seconds. The original regression fixture
required 787,968 timestamp comparisons for 512 edges; the new test requires at
most 512, with output equality against the original reference query. Authority,
producer scope, invalid clocks, stale caches and concurrent snapshot behavior
remain tested.

Hermes1 now runs `074c981` plus the reviewed, uncommitted query patch
(`hymem/query/augment.py` SHA-256
`69ec873f6b3854824a2035c2c0e1948dda36b728cab620f260ffa70a2bf1c3e9`).
The first patch attempt safely refused because of the known metadata archive
relocation; the retry verified those archived files and succeeded. One private
recovery trial was interrupted by that restart; it was discarded as acceptance
evidence and a fresh clone used instead. No production recovery had begun.
After deployment, MCP initialization, tool listing, profile and augmentation
all passed, as did Honcho HTTP health, embedding connectivity and FK checks.

### Bounded embedding recovery and remaining rollout

A fresh private-clone segment repaired 286 vectors in nine embedding batches,
with all 12 audited source tables unchanged and zero FK faults. Production
recovery then began with a fresh `database-before-reembedding.sqlite` backup.
It is bounded, resumable, embedding-only, and stops after one sweep; no full
LLM dream was requested. Missing historical proofs remain explicit blocked
items, not invented sources or silently relabeled embeddings.

Hermes2's private update rehearsal passed at schema 59: 98 physical tables
unchanged, zero FK faults, integrity OK and reopen a no-op. Hermes3's rehearsal
likewise passed with 118 unchanged physical tables and zero FK faults. Both
archive installations updated with all 428 release-file hashes verified,
accounting explicitly for the archived obsolete generated metadata. No local
customizations were overwritten. Each received the accepted token query patch.

The initial operational probes incorrectly used `/home/node/.venv/bin/python3`.
Inspection of actual running server arguments established that **all three
instances run `/home/node/hymem-env/bin/python3`**, including Honcho on Hermes1.
The dedicated environments contain sqlite-vec and the MCP/OpenAI dependencies;
Hermes2's initial missing-dependency probe results were not service failures.
The isolated editable wheel was subsequently installed and verified in the
actual environment on every instance, with all non-HyMem dependency versions
unchanged. Entry-point metadata and import origins were checked there.

This runtime distinction also changes transport/producer identity. The first
Hermes1 recovery attempt used the shared environment, so its batches were still
incompatible with the real service. Root caught that through an actual-runtime
health audit, interrupted only its recovery process, and repeated the private
trial under the correct interpreter. Source tables were unchanged. The new
runner asserts interpreter equality before inheriting the live MCP environment;
the corrected sweep replaces those provisional vectors rather than relabeling
them. The original pre-recovery backup remains intact.

Actual-runtime acceptance: Hermes1 MCP and Honcho passed; Hermes2 MCP, providers
and all nine doctor checks passed with zero failures/warnings; Hermes3 MCP,
providers, schema, canonical normalization and FKs passed, with only historical
embedding identity/compatibility failures remaining before its recovery.

### Second accepted performance fix: graph-source recovery lookup

A different agent narrowed graph candidates by exact stored text in a
materialized CTE before evaluating the unchanged full live-evidence predicate.
The 65-row overflow limit remains **after** authority validation, so retired
same-text owners cannot crowd out a later live owner or hide excessive owners.
No schema, producer identity, proof eligibility or publication guard changed.

Root independently verified 169 tests in 77.38 seconds, including the first
performance regression file. A private production-copy comparison of 64 edge
lookups preserved an identical aggregate native proof hash while reducing
timestamp checks from 584,512 to 64 and elapsed time from 7.392s to 0.353s.
The CLI-only patch is installed on all three without restarting services:
`hymem/reembed.py` SHA-256
`4cf89a3d43d013ca4a5cd4e79e3418a642e5a1004df0b57eef07bc37cad7bcc8`.
Original files are retained in each rollout backup. Correct-runtime bounded
recovery completed on Hermes1 and Hermes3; Hermes2 has no stored vectors
requiring conversion.

## Final acceptance and remaining data constraints

| Instance | Rebuilt vectors | Remaining incompatible | Final doctor |
| --- | ---: | ---: | --- |
| Hermes1 | 8,907 | 1,095 | Stored compatibility FAIL; five pre-existing canonical drift values WARN |
| Hermes2 | 0 (none stored) | 0 | All nine checks OK |
| Hermes3 | 451 | 347 | Stored compatibility FAIL only |

Both recovery sweeps reached EOF and stopped with no provider errors on the
correct-runtime sweep. Every eligible existing mirror was repaired. Independent
read-only eligibility audits found **zero still-eligible incompatible rows**.
The remaining records are retained rather than deleted or relabeled:

- Hermes1: 1,090 chunks lack canonical manifests; five aggregation embeddings
  require regeneration of their material publication.
- Hermes3: 264 chunks lack canonical manifests; eight edge mirrors have no
  acceptable live graph owner; 75 narrative-fact mirrors fail source proof.
  Its sweep also counted 15 already-compatible fact mirrors with rejected
  source proofs as blocked, so sweep-blocked (362) is not identical to remaining
  incompatible (347).
- Hermes1's five canonical-normalization findings already existed in the
  pre-recovery backup, at the same locations and counts: one alias, two entity
  mentions and two graph endpoints. They were not introduced or rewritten.

Root independently compared 12 source/authority tables on each instance against
its pre-recovery/pre-update backup: **all unchanged**. Full SQLite integrity
checks pass and all three stores have zero FK faults. MCP profile and retrieval
pass on all three after recovery; Honcho HTTP health passes on Hermes1. Actual
embedding endpoints, dimension 384, vector shadows, schema and canonical checks
pass except the explicitly listed historical compatibility/drift findings.

A separate agent supplied a private synthetic extraction canary. Root read the
helper, independently passed its nine offline adversarial tests (3.22s) and its
CLI self-test, then executed it using each actual service environment. All three
real DeepSeek tests passed: **two completions / two HTTP attempts per instance**,
one fact, one canonical evidence row, one observation, one live edge and one
current publication. Each private database passed integrity/FKs and cleanup;
no synthetic messages were written to production. This is a tiny extraction
smoke test, not a dense-excerpt quality benchmark.

### Why the final historical warnings are not automatically erased

The maintained aggregation-only builder replaces the **whole current publication**,
not five individual vectors. It has no node-selection/global-call-budget option
and direct use needs dreaming-lease or maintenance orchestration. Missing source
proofs can produce a smaller or empty replacement tree. A read-only membership
and cost preflight is needed before approving that broader rebuild; it was not
run merely to make doctor green. Likewise, normalizing old canonical coordinates
can merge records and rewrite evidence references; that needs a separately
reviewed data repair, not an incidental vector conversion.

Retained unproven historical data cannot become canonical by inventing a session,
manifest, parent or producer label. Earlier backup audit found no authoritative
source for the quarantined orphan. No additional history was discarded, no full
LLM dream or benchmark was run, and no third-party credential was rotated.

### Recovery artifacts and handoff

Each instance retains
`/home/node/.hermes/backups/hymem-rollout-20260908-074c981/` with the original
changed code, exact manifest, pre-update database, private rehearsal copies,
archived original metadata and the two pre-patch source files. Hermes1/3 also
retain `database-before-reembedding.sqlite`. Hermes1's separate orphan baseline,
native quarantine and tested full-baseline restoration are documented above.

The published branch was rechecked at the end and still points to `074c981`.
The final read-only cross-instance audit verified all 428 release files on each
instance (including original metadata in its archive), both runtime patches,
the actual recovery entrypoint, identical embedding producer digests and zero
leftover recovery jobs. Each container is running with two MCP processes;
Hermes1 additionally has its expected Honcho process. All five original Hermes1
local customizations are preserved.

The two tested runtime patches and their tests remain **uncommitted locally**;
the deployed installations include both patches. Preserve/include them in the
next upstream commit so a later archive deployment does not overwrite them.
The previously exposed DeepSeek credential still requires operator rotation;
the earlier redactor fix does not revoke a leaked key.
