# Embedding recovery and deployment feedback fixes

Baseline: `b360b82`; working tree initially clean. No production HyMem service,
credential or database changes, or real-provider requests, are part of the
local code verification. The user separately identified Afrodite for the
requested deployment-redactor fix; that exact script's verification and any
backed-up deployment are recorded below.

## Sequential acceptance plan

1. Correct the complete remote/internal embedding setup examples and route
   guidance. Implementation agent: `fix_03_semantic_generation`. Root verifies
   configuration construction, SDK request routes with a mock transport, and
   neighboring bootstrap/endpoint tests before accepting.
2. Add row-level embedding recovery diagnostics so matching vector-shadow
   metadata cannot hide incompatible durable rows. A different implementation
   agent supplies regressions; root reproduces the mixed-producer false green
   and verifies healthy/mixed/malformed/absent-vector states independently.
3. Add bounded, resumable embedding-only repair with no extraction LLM calls,
   exact producer validation, safe concurrency and source/identity fences,
   useful progress/exit status, and repeat-run idempotence. A different agent
   implements after issue 2 is accepted. Root verifies recovery and refusal
   paths plus neighboring tests before accepting.
4. The deployment `/opt/stacks/hermes/redact.py` is on Afrodite. The user
   identified the connected server and authorized reaching it. Read-only SSH
   located the actual script; source was inspected through an independent
   local redactor, without reading credential files. Fix a copy and test it
   separately, then independently verify before an exact-target, backed-up
   deployment. Key rotation remains an operator action.
5. The reported orphaned production chunk requires its actual store/provenance
   on a clone. No destructive data repair or fabricated session is planned.
   Surface integrity faults safely where relevant to diagnostics/repair.

## Verification baseline

Prior read-only diagnosis passed 77 targeted tests on `b360b82`, reproduced
incomplete setup startup failure and `/embeddings` versus `/v1/embeddings`
request routing without network, and reproduced doctor reporting an OK identity
while a durable chunk retained an incompatible producer. A synthetic chunk
drain repaired that row and was idempotent. Those checks predate these fixes.

## Results

Implementation and root acceptance results will be recorded below in order.

### Issue 1 accepted: complete deployment examples

The first agent changed README and added nine focused offline regressions.
The opening quickstart no longer accidentally selects an incomplete remote
backend. Both remote recipes export all required producer settings and refuse
missing/empty operator declarations. The internal `/v1` example is explicitly
route-specific; custom route prefixes remain supported. Scheduler prose now
describes the actual event-driven worker and does not promise automatic drain.

Root reviewed the frozen diff and ran:

```sh
python -m pytest -o addopts='' -q \
  tests/test_readme_embedding_deployment.py \
  tests/test_embedding_endpoint_diagnostics.py tests/test_bootstrap_lifecycle.py \
  tests/test_endpoint_policy.py tests/test_dream_scheduler.py \
  --junitxml=/private/tmp/hymem-deploy-docs-root.xml
```

**133 passed in 7.84 seconds**; `git diff --check` passed. Actual SDK request
building was exercised through a mocked bottom HTTP boundary, with no network
traffic or provider charges. Issue 2 was delegated only after this acceptance.

### Issue 2 accepted: stored compatibility and foreign-key diagnostics

A different agent added the reusable read-only `scan_embedding_health` helper
and doctor integration. Six mirror tables are scanned under one snapshot,
streaming individual vectors. Counts partition stored rows into compatible,
incompatible, malformed or unverified; missing schema returns unknown counts.
Historical `embedding_cache` rows are excluded. This is explicitly **not** an
audit of missing embeddings, content hashes, source proofs or repair eligibility.
Foreign-key faults are safe per-table counts, not row IDs or source values.
Doctor now closes its connection on initialization failure and avoids exposing
untrusted exception/metadata values in the affected diagnostics.

Agent gates: 57 focused tests (5.28s), then 216 neighboring tests (36.54s).
Root reviewed the frozen implementation and independently verified:

- Real v59 two-chunk partial producer switch: old code falsely passed;
  fixed code fails until the stale row is re-embedded, then passes.
- All six mirrors: JSON/packed vectors, invalid numerics, wrong producer or
  dimension, unavailable identities/schema, write-denying SQLite authorizer,
  caller-owned transaction preservation, safe foreign-key counts.
- A real v59 synthetic orphaned chunk: doctor reports `chunks=1`, preserves
  the row and emits neither its ID nor source text.

Root gate:

```sh
python -m pytest -o addopts='' -q \
  tests/test_embedding_recovery_health.py tests/test_message_semantic.py \
  tests/test_bootstrap_lifecycle.py tests/test_embedding_endpoint_diagnostics.py \
  tests/test_readme_embedding_deployment.py tests/test_public_default_contract.py \
  tests/test_embeddings.py tests/test_endpoint_policy.py \
  --junitxml=/private/tmp/hymem-embedding-health-root.xml
```

**289 passed in 53.43 seconds**. Compilation and `git diff --check` pass;
endpoint policy, embedding transport and producer implementations are unchanged.
Independent scripts are in `/private/tmp/hymem-recovery-root.bKUByT/`.
Issue 3 began only after this acceptance, using a different implementation agent.

### Issue 3 accepted: bounded embedding-only recovery

The next agent added `hymem.reembed` and its console entry point without changing
the existing dream, transport, or producer implementations. Dry-run opens only
an existing current-schema database read-only. Apply uses the dreaming lease,
exact durable producer identity, source proof/render validation, bounded provider
batches, and transactional revalidation before updating existing mirrors and
their own shadow rows. It preserves unproven history and refuses foreign-key
corruption. Aggregation publications are reported for ordinary rebuild, not
independently relabeled. The durable cursor stores opaque rowids, not source text.

Root review required additional regressions for final-sweep revalidation,
source-render agreement, cursor privacy, producer fencing, and truthful progress
counters. `max-items` bounds cursor/provider work, not the full read-only audits.
Agent verification: 172 neighboring tests passed (115.10s), then 44 final
focused/documentation tests passed (18.27s).

Root independently exercised the public module CLI in fresh processes against
a real synthetic v59 store: no LLM credentials or calls, byte-identical database
after dry-run, explicit local apply opt-in, bounded cross-process resume,
unchanged source rows, clean integrity/FKs, and zero-call repeat idempotence.
Root's final frozen-source gate:

```sh
python -m pytest -o addopts='' -q \
  tests/test_reembed.py tests/test_embedding_recovery_health.py \
  tests/test_embeddings.py tests/test_message_semantic.py \
  tests/test_dream_lease.py tests/test_aggregation_provenance.py \
  tests/test_fact_authority.py tests/test_readme_embedding_deployment.py \
  tests/test_embedding_endpoint_diagnostics.py tests/test_bootstrap_lifecycle.py \
  --junitxml=/private/tmp/hymem-reembed-root-final.xml
```

**319 passed in 146.78 seconds**. Public CLI canary, compilation, and
`git diff --check` passed. No real embedding requests or production store writes
were performed. Issue 4 implementation starts only after this acceptance,
with a different agent from issue 3.

Additional unchanged-path deployment regression gate:

```sh
python -m pytest -o addopts='' -q \
  tests/test_aggregation_migration_016.py tests/test_aggregation_generation_v56.py \
  tests/test_aggregation_provenance_v55.py tests/test_legacy_retired_promotion.py \
  tests/test_migrations.py tests/test_endpoint_policy.py \
  --junitxml=/private/tmp/hymem-migration-endpoint-root.xml
```

**248 passed in 117.43 seconds**. These cover the earlier migration-016 lineage,
retired evidence promotion, aggregation guards, and endpoint-policy regressions.
This is targeted verification, not a full repository suite or live-store replay.

### Afrodite redaction baseline (read-only)

Actual script SHA-256:
`88ca18e67b136ccab1820dc4bc4916dd38c9b2e8836a86c5c519105a5f168f00`.
Owner `atta:atta`, mode `0755`, 4,550 bytes. No ancestor AGENTS.md found on
the script path. No credential files, services or databases were accessed.

Synthetic `sk-` plus 32 hex characters reproduces:

- `export FOO_API_KEY=...`: secret survives; counter says before=0, after=0.
- `BAR_API_KEY=...`: secret is removed; counter still says before=0.
- A bare credential at the start of a line: survives, before=1, after=1.

The actual value-shape rule already recognizes the `sk-` prefix. Tokenization
swallows the assignment prefix into the token, and line redaction requires a
preceding separator for unquoted tokens. The self-test has no built-in
canaries, so empty/no-recognized-secret input can pass without testing those
failure modes. This is more specific than the supplied inferred diagnosis.

### Issue 4 accepted and deployed: Afrodite redactor

A different implementation agent derived `tools/deployment/redact.py` from the
hash-verified actual source, with 38 permanent offline tests. Sensitive-key
assignments work after export/env prefixes; bare/quoted tokens work at line
start; JSON sensitive subtrees and embedded credential tokens are masked.
Detection and redaction agree. Built-in self-tests always run and use literal
canary-survival checks independent of the residual counter. CLI failures never
echo input paths, values, or exception details. This remains deliberately
over-redacting and best-effort, not a guarantee for arbitrary/encoded secrets.

Root reviewed the frozen implementation, required single-pass output assembly,
and independently verified **38 tests passed in 0.26s**, nine synthetic
redaction/detection cases plus joint redactor/detector fault injection, and
18 black-box stdin/JSON/idempotence/error-path cases. Compilation and diff checks
passed; built-in no-file self-test passes 10 checks.

Candidate/installed SHA-256:
`071056ff0724da9c18461d80e2bf7de0532acd0ec0305593c751aa137177f6ef`.
Root uploaded only the verified candidate and synthetic verification scripts
into a private staging directory on Afrodite, checked its hash, and ran both
independent checkers there. The live original hash was unchanged. Installation
refused a changed original, retained a hash-verified backup, and atomically
replaced only `/opt/stacks/hermes/redact.py`; owner `atta:atta`, mode `0755`.

Backup (original hash above):
`/opt/stacks/hermes/redact.py.backup-20260908-88ca18e67b13` on Afrodite.
Synthetic verification scripts remain in private
`/opt/stacks/hermes/.redactor-verify.EqNlFS9A/` for audit.
Post-installation, both independent checkers and the executable's built-in
self-test passed again; installed and backup hashes were verified. No service
restart, production database change, credential-file read, or key rotation was
performed. The exposed key still requires rotation; fixing redaction does not
revoke an already disclosed credential.

### Combined HyMem acceptance gate

The issue-3 and earlier deployment regression file sets above, plus
`tests/test_public_default_contract.py`, passed together on final HyMem code:
**570 passed in 263.45 seconds**
(`/private/tmp/hymem-combined-recovery-root.xml`). The standalone redactor adds
the 38 separately accepted tests above. These are targeted gates, not the full
repository suite, a production migration replay, or a paid-provider benchmark.
HyMem runtime changes remain local and uncommitted; only the standalone
Afrodite redactor was deployed.

### Issue 5: orphan confirmed; authoritative restoration blocked on source

After issue 4 acceptance, a different agent resolved only the exact hermes-1
container and its Mounts metadata, without inspecting environment/credentials.
The store is `/home/node/.hermes/hymem.sqlite` in that container. SQLite backup
from a `mode=ro`, query-only source created a private same-container snapshot:
`/tmp/hymem-orphan-audit-cyztl782/snapshot.sqlite` (168,337,408 bytes;
directory `0700`, file `0600`). No store was transferred off Afrodite or opened
through HyMem initialization. No production rows, services or credentials changed.

The agent confirmed schema 59 and one chunk foreign-key fault: the reported
rowid 1058 has no session parent. Root independently opened the snapshot with
`mode=ro&immutable=1` and confirmed:

- One chunk FK violation, matching rowid 1058.
- Zero parent rows, manifest entries, raw messages in its span or session,
  or lossless coverage for the chunk/session.
- Zero claim evidence or claim observations; one chunk embedding survives.

Output contained only counts/booleans, not source text or private identifiers.
These records cannot establish the missing session or canonical source history.
The orphan and its vector are preserved. An authoritative backup is required
for restoration; deleting/quarantining stored user content is a separate
operator disposition, not an automatic integrity fix. The new repair command
deliberately blocks all writes in this store until its FK fault is resolved.
The private audit snapshot is retained for follow-up, not a deployed migration.

### Remaining operator actions

- Rotate the disclosed DeepSeek credential and remove the reported literal
  credential from the launcher into protected credential storage. Credential
  files and wrappers were neither read nor changed in this work; fixing the
  redactor does not revoke the old key or remove existing transcript exposure.
- Identify an authoritative backup for the orphan's missing session/source, or
  explicitly decide how to dispose of that preserved unproven chunk.
- Deploy the local HyMem changes, rehearse repair on a current clone using the
  verified existing producer settings, and run bounded embedding recovery.
  No production re-embedding/provider requests were performed here. Aggregation
  publication rebuilds and unproven legacy rows retain the documented limits.

Follow-up: the requested backup search is complete. No authoritative parent or
source was found in the located HyMem and original Hermes backups. See
[the backup-selection audit](2026-09-08-orphan-backup-audit.md) for comparisons,
scope, preserved originals and the remaining data-disposition decision. No
production restoration or quarantine was performed.

The user's subsequent approval covered a **clone-only quarantine rehearsal**,
which is now independently verified. The expanded dependency audit found four
entity links in addition to the orphan/vector; these were archived too. The
repaired clone has zero FK faults, exact expected logical/index changes and a
verified full-baseline recovery copy. Root's vector-enabled local gate passed
99 tests. See [the rehearsal report](2026-09-08-orphan-quarantine-rehearsal.md)
for evidence, limitations and retained private artifacts. Production application
still requires a separate decision and was not performed.

Subsequent explicit approval covered production application and sequential
updates. Those are now performed: the exact orphan was recoverably quarantined,
all three instances updated, and eligible vectors re-embedded under their actual
service runtime. See [the production rollout report](2026-09-08-production-repair-rollout.md)
for the final checks, two additional accepted performance fixes and remaining
historical-data limitations. Earlier no-production statements above describe
their earlier phases, not the current deployment state.
