# v47 upgrade failure: retired legacy evidence

Baseline: `b59e69e` (`bugfixes Astra`). This is a deployment regression found
after the earlier 14-finding verification sweep, not an expected index rebuild
or model-refusion cost. No production database is available in this workspace.

## Diagnosis

Migration 047 reconstructs lossless message coverage before retrying the v40
chunk-manifest backfill. This makes previously unmanifested legacy evidence
eligible for promotion. The promotion query checks provenance and singleton
source identity, but omits `is_current = 1`.

For a retired row, promotion attempts a new claim observation, which the
canonical-evidence SQL guard rejects. History-mutation authorization is not a
valid workaround: the same loop creates positive lifecycle assertions, and
`record_lifecycle_event` requires current canonical evidence. Retired revisions
must not be resurrected or acquire invented canonical history.

The reported production counts (945 candidates, 222 retired, 723 current) are
the operator's measurements; they have not been independently measured here.

## Fix and verification plan

1. Reproduce the failure before modifying production code, both in a permanent
   public-initialize regression and with a database created by the actual
   historical v46 package (`c215f52`).
2. Have the implementation agent restrict the backfill candidates to current
   evidence, with a provenance comment and focused regression tests. Preserve
   the SQL guards, lifecycle validation, revision state, and schema version.
3. Independently review the patch and retry the very same failed historical
   database. Check active positive/negative promotion, unchanged retired rows,
   recovered manifests, true source-loss handling, replay, and database
   integrity. Run migration/provenance/retention/portability regression tests.
4. Report verified scope and hand the patch back for another dry-run on a
   fresh production-store clone before any live deployment.

## Verification record

- **RED, permanent regression:** the implementation agent's public-initialize
  test reproduced `IntegrityError: claim observation lacks canonical evidence`
  before changing `hymem/core/db.py`.
- **RED, independent authentic upgrade:** root extracted the historical
  `c215f52` package into a temporary directory and used its own public
  initialization to create schema 46, with its original guards intact.
  The fixture contains surviving raw messages without coverage, exact
  unmanifested singleton chunks, current and retired evidence of both
  polarities, and a genuinely lost-source retired claim. Current-code
  initialization reproduced the exact error. Schema stayed at 46; four coverage
  records had committed, while manifest and observation writes rolled back.
  This confirms that the retry must handle already-committed coverage.
- **Implementation reviewed:** the only production behavior change is
  `AND ev.is_current = 1` in `_backfill_v40_chunk_manifests`, plus a comment
  explaining why later coverage recovery cannot authorize retired history.
  No migration version, guard, or lifecycle-validation change is needed.
- **GREEN, independent authentic retry:** with the patch, root reopened the
  same failed schema-46 database and public initialization completed to schema
  59, without cleanup or manual version changes before retry. All original
  fields of the three retired rows were unchanged; none acquired observations
  or lifecycle events. Both current polarities gained canonical source-backed
  observations; only the positive row gained an assertion. Four recoverable
  chunks gained manifests; only the genuinely lost-source chunk was marked
  terminal. `PRAGMA foreign_key_check` was empty and `PRAGMA integrity_check`
  returned `ok`. Reopening again, and a separate deliberately stale-v46-stamp
  replay, preserved complete evidence/observation/lifecycle/manifest/loss
  snapshots.
- **GREEN, independent fresh upgrade:** a second copy of the pristine
  archived-package v46 database upgraded directly to 59 and passed the same
  state, integrity, reopen, and replay assertions.
- **Agent gate:** five permanent regression cases plus five focused neighbors
  passed (10 total, 7.07 seconds). The new cases include all-retired stores,
  current-versus-retired source identity, ambiguous current sources, interrupted
  promotion rollback/retry, and negative controls that still reject retired
  resurrection, forged observations, and forged lifecycle assertions. These
  portable CI fixtures reconstruct the old data condition in the surrounding
  current schema; they do not claim to be an archival v46 DDL snapshot.
- **Root gate accepted:** the frozen migration/provenance/retention/portability
  run passed **301 tests in 269.37 seconds**, with zero failures, errors, or
  skips. Root independently verified the JUnit counters and absence of
  failure/error elements in `/private/tmp/hymem-v47-retired-root.xml`.
  No production or test edits were made during that run. Command:

  ```sh
  python -m pytest -o addopts='' -q \
    tests/test_legacy_retired_promotion.py tests/test_migrations.py \
    tests/test_terminal_chunk_loss.py tests/test_claim_provenance.py \
    tests/test_bitemporal.py tests/test_evidence_ledger.py \
    tests/test_portability.py tests/test_lossless_digest.py \
    tests/test_valid_time_readpaths.py --durations=15 \
    --junitxml=/private/tmp/hymem-v47-retired-root.xml
  ```

- Changed Python files compile successfully; `git diff --check` passes. Final
  changes are limited to the candidate filter/comment, five new test cases,
  and this verification record. No commit was made.

Temporary independent fixture/script:
`/private/tmp/hymem-v46-migration.bqzmKr/check_upgrade.py`; pristine backup:
`/private/tmp/hymem-v46-migration.bqzmKr/pristine-v46.sqlite`; interrupted-upgrade
database: `/private/tmp/hymem-v46-migration.bqzmKr/retry.sqlite`.

## Deployment boundary

The prior 4,887-test full-suite result predates this patch and did not cover
this retired-legacy promotion case. It is not evidence of a real-store
migration. This follow-up does not authorize or perform production changes,
service restarts, manual schema stamping, guard disabling, or paid benchmarks.
The full suite is not being repeated for this narrowly scoped follow-up;
acceptance is based on the explicit independent upgrade checks and targeted
regression gate above.
