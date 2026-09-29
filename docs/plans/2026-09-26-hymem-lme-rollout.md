# HyMem candidate rollout and LME gates

Implementation checklist and verification record. Deployment and LME remain
gated on the evidence below; historical receipts do not certify this candidate.

## Current verification progress

- Root accepted the surplus-only episode-vector repair after 382 focused and
  broader regression tests and a network-isolated replay of the real store.
  The replay removed exactly eight surplus derived vectors, preserved all
  valid vector bytes and durable records, and was idempotent on reopen.
- Full-suite v1 stopped before collection because its diagnostic adapter used
  an invalid container-relative parent path. A separate Sol agent fixed only
  the new v2 adapter; root verified the regression and 50 adapter controls.
  V1 receipts are preserved. V2 completed exactly7,764 tests:7,760 passed,
  four expected raw-backend-only skips, zero failures/errors. Root verified
  exit0/PID0/noOOM, networknone, read-only inputs and unchanged tested source.
  Final receipt SHA3ecfaa8810a01c4c2db2d7e6f4e023b95db6a46572fa12cf8d47a0a2f7903a6d.
- A fresh bounded paid dream completed on a private store copy, with explicit
  provider and embedding budgets (128 completions, 384 LLM HTTP attempts,
  512 embedding HTTP attempts, 896 total, 45 minutes). The safety review first
  required specific production-memory transfer approval; the user then
  explicitly approved necessary paid testing including production-derived
  memory. The first launch was blocked before execution. The approved launch
  started one detached supervisor, passed its offline startup stage, and its
  live worker's isolation was independently inspected. No production database
  is mounted writable. It used59 completions/59 LLM HTTP attempts,159 embedding
  HTTP attempts,218 total attempts and193661 tokens, processing17 chunks and
  extracting175 triples. Independent postflight v3 passed, proving durable
  aggregation publication, exact episode-vector alignment, clean claim/evidence
  invariants and no new quarantines. Its seal is
  `68da9fa22a404aa14d3507622d6f7f437ac84b79188379ee057a1e23b2089699`.
  V1/V2 checker failures are preserved: the former omitted embedding identity,
  the latter compared cursors across different processing generations. V3
  uses independently computed current config hashes and maintained source/stage
  validators to prove two rebuilt bounded slices. Three domains retain valid
  pending work; session convergence is explicitly NOT certified.
  `aggregation_blocking` names the clustering
  strategy; a nonempty valid label is not itself an error.
- Hermes1 is deployed at schema64 with all481 candidate source files verified.
  Rollout r3 preserved all130 existing migration tables, reopened cleanly,
  and restarted Honcho plus both MCP processes. LLM/embedding configuration,
  role-specific flags, runtime, wrappers and hook remain unchanged. Read-only
  pre-dream checks passed; inherited8 surplus vectors/17 unverifiable episodes
  are unchanged pending the official bounded production repair dream.
- Root verified44 rollout-helper controls. r1 omitted sqlite-vec in its verifier;
  r2 refused apply before source changes because a stopped WAL remained. The old
  service was restored before preparing r3. The v3 helper adds an explicitly
  verified checkpoint, comparing all132 logical tables including schema/vector
  internals. Both failed stages and rollback material remain intact.
- Root also passed39 paid-diagnostic adapter/upload controls,67 V1/V2/production
  controls,42 V3/cross-version controls, and33 revised production/predream
  controls. Independent v3 audit exited0/PID0/noOOM with networknone and source
  unchanged. LME offline preflight passed with exact481 source files, no
  credentials and zero API calls.

## Deployed result and current headless run

Production dream1408 completed in431 seconds with58 DeepSeek completions,
161 embedding requests and189436 tokens. It processed17 chunks and189 triples,
with all reported error counters zero. Independent verification proved36 exact
episode vectors, no surplus/missing/different/unverifiable entries, clean
canonical/ledger/same-generation invariants and working MCP profile/retrieval.
The bounded dream deliberately leaves pending summary/profile/fact work; it is
not a claim of complete production backlog convergence.

Final doctor testing exposed a packaging regression: the frozen candidate
reintroduced the one-token probe, dropping the already accepted32-token
diagnostic-only patch. A separate Sol implemented exact carry-forward from the
verified backup, plus a repository probe regression. Root reran158 compatibility
tests (four expected raw-backend skips) and21 adapter/probe controls. The restored
doctor remains strict on truncated/malformed responses; no extraction rule or
benchmark prompt was relaxed. Frozen benchmark source remains5550…; production
manifest is `bc48b19354f0ec42cb00cbcd72cf8b5b75f48b8936443d56e9b57181cc44547c`,
differing only by the explicit doctor overlay2ac786…. Final strict postdeploy
passed: doctor10OK/0FAIL/1WARN (existing summary context health), actual MCP,
schema64, all481 source hashes including that overlay, unchanged runtime/config,
canonical and exact-vector checks. Receipt069bfbdc… is recorded in readiness JSON.
Afrodite stack returned46 passed/0warnings/0failures after the planned restart
observation cleared.

The fresh eight-question run is now running, not scored or finally validated.
Container `72a691c89b55b44bde0c83a58520311eb67c9c4c45509f70339deeff65248c23`
has no restart policy. Finalizer PID557925 was independently verified with
parentPID1/sessionID557925 and the correct owned run; it survives laptop/SSH
closure and runs one network-disabled validation after completion. It cannot
start another paid run. Canaryv19 passed8/8 calls, client cleanup verified,
35796tokens; Q1 indexing started with no early quarantines/instrumentation errors.
No full500 readiness, complete sample result or score is claimed.

Remote run root:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-v64-sample8-headless-v1`.
At completion inspect both live and validation exits, archive/checkpoint,
accounting and summary-degradation counts. `offline_validation_finished` alone
does not mean passed, and the finalizer deliberately leaves score_verified false
until root independently reads the validator evidence.

## Candidate and historical evidence

Use the exact assembled candidate `/private/tmp/hymem-v64-lme-exact481.cafkhqxa`,
manifest `5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576`
(481 files), including schema64 claim fixes, retained R7 summary migrations,
and the accepted surplus-only episode-vector repair. Never deploy the dirty
main checkout or substitute its schema61 source.

The September25 production receipt records schema63, deepseek-flash, embedding
dimension384 and 479 verified R7 files. This is historical evidence, not a
fresh production-state assertion. Its source manifest is
`1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8`.
Pinned container image:
`sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5`.

Prior R7 sample-eight completed with strict archive/checkpoint validation:
8 completed, zero failed/missing, 6 correct (75%), 6,348 provider completions,
6,349 HTTP attempts and 20,328,150 tokens; cost unavailable. All8 questions
had summary degradation (85 sessions). It passed process/item-indexing checks,
not clean summary health; the receipt explicitly denies full500 readiness and
official comparability. Keep that baseline and its artifacts unchanged.

## Readiness, rollout and rollback

1. Bind a fresh complete-suite receipt and fresh bounded paid dream/postflight
   to the exact new candidate manifest. Require claims, ledger, canonical,
   same-generation, integrity/FK and episode-vector alignment checks, plus
   successful aggregation without inherited orphan-vector exceptions.
2. Root reads current Hermes1 container/image/source/schema/config identity,
   effective process environment, runtime distribution, health and dream idle
   status. Preserve effective `HYMEM_LLM_EXTRA_BODY`, thinking, model, endpoint,
   credentials and all embedding settings. Record only hashes/booleans; never
   print credentials. Existing doctor hotfix and deployed phase1 differences
   must be accounted for explicitly in the source delta.
3. Prepare a fresh private rollout stage and reviewed manifest-bound deployment
   helper. Existing `lme_r7_deploy_apply.py` is hardwired to479 files, schema61
   backups and migration target63; `claim_conflict_deploy.py` only replaces
   phase1 at old fixed pins and assumes schema63. Neither is a valid v64
   deployment command. Adapt their guarded pattern, not their stale constants.
4. Rehearse candidate migration on a fresh production SQLite backup with
   network disabled; verify preservation of existing durable records,
   schema64, reopen/idempotence, integrity/FK and all new store invariants.
   Back up source delta, editable distribution/entrypoints and configuration.
5. After idle/PID/config recheck, gracefully stop only hermes-1 (120-second
   grace; exit137/OOM/forced termination fails). Take a new stopped backup.
   Apply only manifest-reviewed source paths atomically; preserve permissions.
   Use offline no-deps/no-index editable installation only if required by
   packaging changes. Offline migrate to64, verify reopen, then start Hermes1.
6. Adapt `lme_r7_postdeploy_verify.py` for candidate manifest/count/schema64 and
   current hotfix identities. Verify every source hash, unchanged dependencies,
   wrappers/hooks/config, honcho/MCP effective-environment parity, deepseek-flash,
   dimension384, health, doctor, MCP handshake/profile/retrieval and store checks.
7. Stop on any failed stage and inspect receipts before retry. Rollback remains
   explicit: while stopped restore the matching source/distribution/config set;
   restore SQLite only from the fresh corresponding backup after accounting
   for post-start writes. Never run old schema63 source against migrated64 DB
   or silently discard writes by an automatic restore.

## Fresh sample-eight, then full500

Reuse the reviewed headless structure in `tools/diagnostics/lme_r7_headless/`
and `lme_r7_headless_finalizer.py`, but reseal source/controller/helper pins and
new run roots. Existing controllers are bound to the completed R7 run and
must not be restarted, resumed or pointed at changed source. Candidate source,
dataset and runtime are read-only; production memory and Docker socket are
absent. Only live worker gets scoped server-side credential mount.

Keep dataset SHA
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`,
S/full500 source, sample8/seed0/workers1 and selected source indices
`[213,262,329,339,370,372,392,400]`. Model/answer/judge remain deepseek-flash,
endpoint `https://api.deepseek.com`, thinking disabled, answer/judge extra body
`{"thinking":{"type":"disabled"}}`, top-k15, auto-ability,
permissive-default, legacy-custom judge, no-prereg, protocol-split full,
indexing max100 cycles/3600seconds, require healthy, keep-db. Preserve canonical
producer identity, registry decisions, canary v19 and semantic split v11.
Production extra-body preservation is separate from the benchmark's intentional
canonical CLI extra-body configuration.

Seal/test offline startup and real-SDK producer binding with zero paid calls.
Then separately create/start detached live Docker worker; no restart policy.
Retain bounded private logs and process-group cleanup. Old sample-eight outer
limit is9hours/512MiB logs; full500 requires an explicitly sized supervisor
limit and validator supporting500 physical checkpoint rows, not reuse of the
eight-question fixed validator or nine-hour ceiling.

After exit, independent network-none/credential-free validation must bind raw
archive and actual checkpoint, all one-attempt histories, selected IDs,
per-role usage and separately metered canary. Require zero failed/missing and
healthy indexing; report wrong answers as score outcomes and summary degradation
separately. A clean sample-eight operational gate permits the authorized fresh
full500 run with unchanged protocol/registries. No overall paid-call cap was
requested; track actual calls/tokens, and leave unknown dollar cost unknown.
Do not call legacy-custom/no-prereg performance officially comparable. Compare
the same eight IDs to the75% historical baseline, with source revision and
summary health differences explicit. Full500 is its own fresh artifact/score.

## Evidence still required

Complete-suite, independent paid repair audit, fresh census, reviewed rollout
helpers and offline sample8 startup are now verified for this exact candidate.
Still required: fresh production migration rehearsal, controlled rollout and
post-restart production checks/dream, followed by the live sample8 gate. Full500
requires its own validator/supervisor seals after sample8 passes. Historical
receipts cannot substitute for these remaining gates.
