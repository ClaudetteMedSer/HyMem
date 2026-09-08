# Deployment follow-up: aggregation schema lineage and internal embeddings

Baseline: `e662d4d`. The previous retired-evidence promotion fix is committed
and retained. No production service or database is being changed here.

## Sequential plan

1. **Aggregation-table lineage:** reproduce a public upgrade with an actual
   migration-016-created table. A separate implementation agent fixes the
   exact historical schema recognition and any guard-equivalence gaps, with
   permanent regressions. Root independently verifies the historical upgrade,
   populated data/rowid/FTS preservation, invalid-write rejection, retry/reopen,
   and neighboring aggregation/migration tests before accepting this issue.
2. **Internal embedding service:** after the first gate passes, use a different
   implementation agent to fix the diagnosed configuration/diagnostic path.
   Keep public cleartext endpoints rejected and cloud credentials isolated.
   Root checks the reported service hostname with and without explicit internal
   HTTP authorization and verifies the doctor/bootstrap/client path. Document
   settings that must reach the actual HyMem container process.

## Issue 1: aggregation-table lineage

The storage validator builds its two expected CREATE TABLE spellings from
`schema.sql`. Both contain a table-level source-manifest CHECK that never
existed in migration 016; migrations 017 and 045 add columns but do not add
that table-level constraint. Literal validation rejects a legitimate shipped
schema lineage.

Accepting that lineage must not weaken write validation. Source review also
found that existing trigger coverage must be checked for incomplete rows whose
`input_fingerprint` alone changes, because the missing table CHECK cannot
provide the fallback enforcement available to fresh-schema tables.

- **Root RED:** created `aggregation_nodes` using the actual archived migration
  016 CREATE statement, then used the historical `c215f52` public initializer
  to run its real ALTER migrations through schema 46. Seeded a legacy node at
  rowid 77, verified FTS lookup, and stored a legacy vector. Current public
  initialization fails with `schema v56 aggregation generation domain is
  malformed`, leaving schema 55 and the original node intact. This is an
  independently reproduced migration failure, not an expected rebuild cost.
- Independent fixture: `/private/tmp/hymem-deploy-two-fixes.XLEElN/aggregation_upgrade.py`.
  Pristine v46 and failed-upgrade databases are retained alongside it for
  post-fix verification.
- **Root RED, enforcement gap:** under the original v55 trigger, a
  fingerprint-only update to `invalid-fingerprint` succeeded on that historical
  table. The diagnostic probe was rolled back. Merely adding an accepted DDL
  spelling would leave fresh and migrated tables with different constraints.
- **Candidate fix reviewed:** recognize a third exact table spelling built
  from the owned 016 CREATE and 017/045/055/056/057 ALTER statements. No table
  rebuild or blanket removal of CHECK validation. Harden the v55 header guards
  to validate fingerprint-only updates and reject NULL publication predicates.
  Refresh owned guards using transaction-safe `execute`, not `executescript`.
- **Retry case found during independent verification:** failed v56 execution
  can leave a legitimate empty bootstrap-created generation registry and v56
  publication table beside a v55 node table. The repair recognizes only that
  exact empty bootstrap tail, at stamp 55 without a v56 marker. It is not an
  accepted reader/publication binding; normal atomic v56 migration must finish
  and pass the ordinary strict validator.
- **Root GREEN:** the same failed database resumed to v59 without cleanup;
  a second copy of the pristine historical v46 database also upgraded directly
  to v59. Both passed exact v57 binding validation, reopen stability, foreign-key
  and SQLite integrity checks, and FTS integrity checking. Node rowid 77,
  content, and search lookup survived. Malformed fingerprint, negative source
  count, and invalid node kind writes were rejected. Existing migration 057
  still purges unbound legacy vectors; this fix does not resurrect them.
- **Agent gate:** 17 new migration-016 regressions passed in 9.52 seconds;
  23 selected boundary neighbors passed in 15.95 seconds. The new tests cover
  exact migration-derived table shape, interruption at v56/v57, old/missing/
  forged guard repair, rollback, source retention, invalid fingerprints and
  NULL publication fields, and seven malformed/populated bootstrap negatives.
  The portable CI fixtures use current surrounding bootstrap domains; root's
  separate archived-package checks supply the authentic historical-v46 proof.
- **Reader distinction:** exact schema binding checks reject damaged guard
  definitions. Readers independently validate proof contents and may still
  serve an unchanged valid publication; they reject malformed proof contents.
  No new per-read schema inspection was added.
- **Issue 1 accepted:** root's frozen aggregation/migration gate passed
  **306 tests in 168.67 seconds**, with zero failures, errors, or skips. JUnit
  counters independently checked in `/private/tmp/hymem-aggregation016-root.xml`.
  Command:

  ```sh
  python -m pytest -o addopts='' -q \
    tests/test_aggregation_migration_016.py \
    tests/test_aggregation_provenance.py tests/test_aggregation_provenance_v55.py \
    tests/test_aggregation_generation_v56.py tests/test_aggregate.py \
    tests/test_migrations.py tests/test_legacy_retired_promotion.py \
    tests/test_terminal_chunk_loss.py --durations=15 \
    --junitxml=/private/tmp/hymem-aggregation016-root.xml
  ```

  Compilation and `git diff --check` also pass. No source/test edits occurred
  during the gate. Issue 2 implementation begins only after this acceptance.

## Issue 2: container-to-internal-service embeddings

Initial source inspection: non-loopback cleartext HTTP is rejected by default.
The code already recognizes `embedding-server` as an internal service hostname
and supports `HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP=1`. Bootstrap currently
discards the actionable policy error and reports only
`remote_embedding_endpoint_rejected` through the local lexical fallback.
The actual container process environment has not been inspected here.

Root independently reproduced the configuration distinction without network
requests: a clean synthetic environment with the reported URL selects the
local feature-hash fallback and the doctor's explanation omits the recovery
setting. Adding the existing internal-HTTP flag selects the OpenAI-compatible
embedding backend with its non-secret local dummy credential; unrelated cloud
and LLM keys are not inherited.

After issue 1 passed, a different implementation agent was assigned the
bootstrap/doctor diagnostic and container documentation fix. No global
HTTP-policy relaxation or automatic authorization is planned.

- **Independent real HTTP canary:** a temporary server bound only to loopback;
  test-only DNS mapping sent the `embedding-server` hostname to it. Without
  the opt-in, the doctor failed and sent zero HTTP requests. With the opt-in
  and explicit canary model/dimension/revision/tenant settings, the doctor
  returned OK after a real POST to `/v1/embeddings`, validated dimension 3,
  and sent only `Bearer local` despite unrelated fake cloud/LLM keys. This
  establishes that the existing authorized transport path works; it does not
  verify the operator's actual container network or embedding model.
  The sandbox initially prohibited binding a socket; the identical localhost-
  only check then passed with approved socket access. No production or paid
  endpoint was called. Script:
  `/private/tmp/hymem-deploy-two-fixes.XLEElN/internal_embedding_canary.py`.
- **Agent RED:** the initial three diagnostic tests failed before production
  edits because the opt-in hint was missing (0.19 seconds).
- **Implementation reviewed:** an optional, defaulted `EnvConfig` field
  captures the bounded, secret-free policy explanation during resolution.
  Doctor shows that snapshot while retaining FAIL; startup warns about the
  rejected endpoint and lexical fallback. Internal-HTTP advice is qualified
  against the existing host/path safety rules without changing environment,
  authorizing traffic, or constructing clients. Rejected URL, key, and flag
  values are not copied into the explanation. README now documents persistent
  container/process configuration and the existing producer requirements.
  Endpoint policy and embedding transport implementation files are unchanged.
- **Agent gate:** 46 new regression cases plus selected neighbors passed
  (117 tests total, 4.16 seconds). Coverage includes hostile URLs/flags,
  secret-free startup logs, diagnostic snapshots after environment changes,
  optional-field compatibility, intentional local fallback, allowed SDK client
  construction without requests, credential isolation, and retained producer
  pin/revision/tenant requirements.
- **Root post-fix HTTP canary:** the same approved localhost-only canary passed
  with an assertion for the new opt-in hint. It additionally confirmed public
  and malformed URLs send no extra requests, missing producer pin fails before
  HTTP, and loopback still works without the flag. Both actual requests used
  only the local dummy credential, and the server/thread closed cleanly.
- **Issue 2 code accepted:** root's frozen regression gate passed **272 tests
  in 56.07 seconds**, with zero failures, errors, or skips. This includes all
  46 new embedding cases and the 17 new aggregation regressions together.
  JUnit counters and absence of failure/error elements independently checked
  in `/private/tmp/hymem-internal-embedding-root.xml`. Command:

  ```sh
  python -m pytest -o addopts='' -q \
    tests/test_embedding_endpoint_diagnostics.py tests/test_endpoint_policy.py \
    tests/test_openai_client.py tests/test_bootstrap_lifecycle.py \
    tests/test_message_semantic.py tests/test_embeddings.py \
    tests/test_public_default_contract.py tests/test_aggregation_migration_016.py \
    --durations=15 --junitxml=/private/tmp/hymem-internal-embedding-root.xml
  ```

## Remaining deployment work and limits

The two code changes were implemented by different agents and independently
accepted in sequence. Changed Python files compile; `git diff --check` passes.
No commit, production store migration, live service restart, or actual container
environment update was performed. The full repository suite was not rerun;
the explicit gates above are the verification scope.

For the reported isolated container-network service, the launcher must supply
`HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP=1` alongside the existing correct
embedding URL to both HyMem processes. Keep the actual model/dimension and
operator-verified public producer declarations; do not invent those values.
Restart or recreate through the appropriate deployment mechanism so the running
process receives the updated environment, preserving storage mounts. The code
does not enable insecure HTTP automatically. The actual process configuration,
DNS/network reachability, and deployed model are still unverified here.

Before production deployment, repeat the migration dry-run on a fresh clone of
the actual store and run doctor under the same environment as the target HyMem
process. Doctor may make real provider requests; none were made against the
operator's providers in this verification.
