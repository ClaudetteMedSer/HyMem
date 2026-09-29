# Hermes1 LME: narrow deployment repair

## Scope

User requested fixing the repeated `chk_0e30d9c2` failure and the
`IndexingConvergenceError` to `BenchmarkIntegrityError` escalation. Preserve
Hermes1's operator customizations; do not deploy the broad local working tree,
new digest verifiers, different prompts/models, or higher retry caps. Do not
resume or relabel the old benchmark checkpoint. This is not a claim that the
full LME benchmark or semantic recall has been verified.

## Deployment evidence

Hermes1 remains at `af6a615fa7fd1cf14c4c0a27b9fb236ae264f122`, with existing
operator/source changes. Both affected files still exactly match the old code:

- `hymem/extraction/chunk.py` SHA256:
  `eb96764a56a55540f87ca5ac89cefdf2952ade504d0cfae04898b75bb8f958ab`.
- `benchmarks/lme_protocol.py` SHA256:
  `c7c9c8cbdf07bcaa0381b07dadf05f3e787dd26f4e88c0f1e2c1b3f34ade5a8d`.

The repairs previously existed in private verification snapshots but were not
deployed to these files. The remote repository's other modified files are not
repair targets. No production database is a repair target.

Local staging: `/private/tmp/hymem-lme-backport-20260918.pky9k0`.
Afrodite staging:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-backport-20260918-cquhSt`.
It contains a source-only base snapshot with existing tracked modifications and
untracked Python helpers preserved, a candidate, original target files, and test
receipts. Tests use Hermes1's existing Python environment and image, with network
disabled, sources/runtime read-only, and no credentials or production stores
mounted. An initial test collection lacked the tracked `tools` package; that
staging omission was corrected and the failed receipt retained.

## Exact latest-incident reproduction

Read-only inspection found `/tmp/hymem-lme-orav8hvl/hymem.sqlite` closed with no
WAL/SHM files. A copy remains on Afrodite; no raw benchmark content was transferred
to the laptop or sent to any provider. Copied database SHA256:
`d0d3d12c7a2acde059edc03b128d4a7e2ceedb2228628cea0da5ab5bacc9bbf1`.

The row for `chk_0e30d9c28d20616decdb0e8ffee554d12fb0115c` records three
attempts, `branch_incomplete`, and the nested terminal reason
`left.left.left.left.split:no_admissible_semantic_boundary` with `resource_limit`.

Using the target runtime's actual source builder, validated coverage manifests,
and canned contract-valid complete-empty replies:

| Code | Scripted calls | Result |
| --- | ---: | --- |
| Unchanged Hermes1 | 6 | Exact stored failure reason/details reproduced |
| Single-file clean-empty repair | 14 | Successful empty result, no failure |

Both arms used the same two canonical source records, source SHA256
`87072dbc8859de3690962f42df38f0009af449bb3a29767b82d6359ca20ebf3a`, and initial
request SHA256 `e1b17dff33c9d21fcafce3d56f0c238f902830c942ea229a194c9bb4db27c70f`.
Network calls: zero. SQLite changes: zero. Additional calls in the repaired arm
visit siblings that the failed arm never reached. This proves the false failure
classification in this particular incident; it is not proof of semantic recall.

## Sequential acceptance

1. Verify the existing single-file extraction repair (`clean_empty` v2 to v3),
   including exact-source reproduction and target-runtime regression tests.
2. Only then backport the failure-envelope repair with a separate agent, retaining
   honest pending/quarantined state, score-zero rows, and structural-error guards.
3. Verify both together against the unchanged target runtime; review the exact
   deployment diff and preserve original files before any installation.
4. Check loaded/deployed source identity and startup implications before changing
   live services. Fresh LME runs require fresh processes/checkpoints; do not
   overwrite old results or call an old quarantine successful.

Extraction accepted: 637 target-runtime tests passed in 429.16 seconds, zero
failures/errors/skips. Root independently checked the XML inventory, exact
single-application-file/three-test-file diff, container exit zero/no OOM/network
none, candidate hash, and unchanged production target hashes. Candidate
`chunk.py` SHA256:
`abf7a159b92a209fab1bcb424085027328c06d48d84dfac336f8bb04c37581ce`.

A fresh agent (`backport_lme_failure_envelope`) implemented only the
failure-envelope backport against the original Hermes1 protocol source. Root
reviewed and tested it independently before installation. The user subsequently
approved installation and restart explicitly: "Install it."

## Failure-envelope verification

The narrow protocol backport is SHA256
`9694381ee434b0394824eb74f765822762dc2e46e5fd7ee6fddc64a31711b304`.
Root reviewed its two added helpers and two modified validators. All other
application declarations, including current-format success validation, remain
unchanged from Hermes1. Archive/model/summary-policy changes from the broader
working tree are excluded.

An independent two-question control reproduces the fatal mechanical-completion
exception on unchanged Hermes1. On the candidate it records the first question
as false/incomplete/unhealthy, processes the second question, closes both
adapters, and computes 0.5 over both rows. No reader, judge or provider is used;
the second answer is a scripted control, not a measured LME result. The control
is retained locally as `tests/test_lme_failure_continuation.py`.

First target related gate: 854 passed, one failed, no skips/errors. The failure
was an obsolete expectation in
`test_extraction_loss_failures_require_mechanical_completion`: its supposedly
drained fixture actually includes a nonzero final coverage-failure counter and
therefore is genuinely incomplete. An earlier review describing it as drained
was incorrect. The application is unchanged. Only this test is replaced with
the already-existing four-state local regression matrix, for both quarantine
and terminal loss: accept pending/incomplete and drained/complete only as
unhealthy failures; reject pending/complete and drained/incomplete. All other
test declarations match the original target file. Its SHA256 is
`4878c5f53c26a16e337a16d53ef0c0595e1c2b3f74212b0e2d874c854a388464`.

Final target gate: **856 distinct tests passed in 326.56 seconds**, zero failures,
errors or skips. Root independently checked XML testcase uniqueness, inclusion
of both completion-state cases and the two-question continuation control,
container exit zero/no OOM/network none, unchanged application candidate hashes,
and unchanged installed production target hashes. All owned repair test
containers have stopped. Local current-tree continuation, terminal-empty,
mixed-envelope and root controls also passed: 449 tests in 7.64 seconds.

## Installation boundary

No direct production database edits or migrations are proposed. Nevertheless,
restarting services with the new extraction identity can trigger ordinary
background reprocessing and database writes; the earlier asynchronous question's
wording about databases remaining untouched must not be interpreted as a
guarantee against normal service activity. A separate benchmark checkout avoids
changing the running memory services. The user selected installation and restart.

## Installed and verified — 2026-09-18

Hermes1 was stopped gracefully while idle, the guarded backport installed, and
the same container started at **2026-09-18 14:40:00 UTC**. Installation verified
224 pre-existing runtime source files against the tested base, changed exactly
two application files plus six regression-test files, and preserved all 417
other source files. Operator/client customizations were not overwritten. The
Git commit remains `af6a615`; this is a documented working-tree backport, not a
claim that the entire latest local architecture was deployed.

Rollback copies of replaced files are under the Afrodite staging directory's
`installation-backup/`. A consistent SQLite backup remains there as
`production-preinstall.sqlite`, mode 0600, 177,373,184 bytes, SHA256
`2816e2665395de473e7a998c881685bed62a31c1f9e58c98b689253e8706f3d3`.
Backup verification initially lacked a connection-local SQLite validation UDF;
the completed backup was then successfully verified read-only using HyMem's
maintained function registration. No database repair or migration was performed.
No production database content was transferred off Afrodite.

Post-restart evidence:

- Fresh Honcho, gateway and two gateway MCP processes are present; Honcho and
  MCP configuration hashes match their pre-restart values and credentials remain
  present without being printed. Both target application hashes match the
  verified candidate. Fresh imports resolve to the installed checkout, report
  clean-empty v3, and contain the mixed-failure validation helper.
- Honcho `/health` and `/dream-status` return HTTP 200. Coverage-integrity
  failures and quarantined chunks are zero. Historical terminal-loss count
  remains 1,062. Existing digest/profile/fact/aggregation backlogs are unchanged.
- Read-only SQLite quick-check passes; FK violations remain zero; schema stays
  at 61. All six tracked table counts are unchanged (sessions, messages, chunks,
  evidence, claim observations and dream runs).
- Pending extraction increases from 276 to 321 under the new identity, as
  expected: 45 additional chunks are eligible for ordinary reprocessing. No
  dream was explicitly triggered and the dream-run count remains 502.
- Internal embedding probe uses only synthetic text, confirms the configured
  OpenAI-compatible backend is reachable with exact identity and 384 dimensions,
  and performs zero LLM calls. No full doctor was run because it also performs
  paid LLM probing and schema initialization.
- An isolated, network-disabled test container mounted the **installed** source
  read-only and passed **445 distinct post-install controls**, zero failures,
  errors or skips. Root verified the XML inventory independently.
- Hermes2 and Hermes3 are still running with their original September 9 start
  times; neither was edited or restarted. Embedding-server remains healthy.

Current status: **narrow backport installed and verified on Hermes1**. Fresh LME
runs must use fresh processes and a new checkpoint; retained failed runs remain
unchanged. No paid benchmark/model diagnostic or full LME run was started. These
checks verify the two reported failures and service availability, not full LME
completion or semantic accuracy. Deployment and runtime receipts are retained in
the Afrodite staging directory's `receipts/`.

## Fresh full-S run launched — 2026-09-18 14:56:51 UTC

**Later status:** stopped with user approval at 20:40:35 UTC after nine scored
indexing failures, during Q10. Evidence is preserved; no automatic resume.
See [stopped-run diagnosis](2026-09-18-lme-stopped-run-diagnosis.md).
The following records the initial launch verification, not current process state.

The user subsequently requested "Run it." One fresh **500-question** run is now
running on Hermes1, not another smoke/diagnostic campaign. The canary passed on
its first and only invocation. At the first post-launch verification, zero
questions had completed, one isolated question store existed, both detached
processes were alive, and the checkpoint status was `running`. This is launch
verification, not a completed benchmark or an accuracy claim.

Run directory inside Hermes1:
`/home/node/.hermes/benchmarks/lme-full-20260918-5HQ61m`.
Host directory on Afrodite:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m`.
Checkpoint: `results/checkpoint.json`. Private logs: `benchmark.log` and
`supervisor.log`. Launch/config/source receipts: `launch.json`, `running.json`,
`source-manifest.json`; normal supervisor completion writes `exit.json`.
Run ID: `sha256:877c8b0f88799c9c7fc1c3a29bb4380b8be80d74fd905b51fa762596b0562d10`.

The run freezes 225 installed source/config files (including the reviewed narrow
backport and existing operator changes) and an independently hash-verified copy
of the pinned 500-question dataset. No broad laptop working-tree changes were
deployed. The existing operator launch settings were preserved: S, sample 0,
seed 0, one worker, top-k 15, auto-ability, permissive-default, full dream,
healthy-indexing requirement, `deepseek-v4-flash` for all three LLM roles at
`https://api.deepseek.com`, thinking disabled, and `legacy-custom` judge.
Embeddings, aggregation nodes and episode granularity remain off, matching the
prior benchmark settings. `--no-prereg` labels this development evidence, not an
official-comparable holdout score. A fresh explicit checkpoint prevents resume.

Unlike the previous external six-attempt canary launcher, this supervisor runs
the adapter **once**, with no reroll/resume/retry-failures loop. Indexing failures
remain scored failures and stay in the denominator; structural integrity errors
remain fatal. Existing per-question bounds stay at 100 indexing cycles and
3,600 seconds. There is no new global paid-call/spend cap. `--keep-db` was added
solely to retain question evidence under this run's private `stores/` directory;
284 GiB disk space was available before launch. Production memory is neither
input nor output, and no production service was restarted for this benchmark.

The supervisor (container PID 431) was launched with `docker exec -d`; its adapter
child is PID 437. Laptop sleep/SSH disconnection will not stop the run. A Hermes1
container/server shutdown would stop it; there is deliberately no auto-resume.
Credentials are loaded from the existing remote environment, never placed in
command-line arguments or copied to the laptop. Files/logs are private (umask
077). An independent agent reviewed the launcher before execution; root
independently verified the live checkpoint's denominator, configuration and
process state afterward. No recurring monitoring automation was created.
