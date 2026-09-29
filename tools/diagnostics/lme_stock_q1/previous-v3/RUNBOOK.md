# Targeted stock LongMemEval Q1 preparation — v3

Status: prepared locally only. No remote launch, paid request, deployment or
benchmark result is implied by this package. Wait for final corrected local and
target-runtime gates and independent root review before using live mode.
The v1/v2 packages and their manifests remain unchanged. This revision repins
the source and gate identity only: all five executable package helpers remain
byte-identical to v2, including the worker and postvalidator. The structural
manifest schema remains v2; preparation_revision is 3.

The corrected R3 verification manifest is pinned to
`df200c11e9a286e7160d222720da705718b700b5f40b9bbaf5c4ca7029cd21cc`.
Its pure-source manifest is pinned to
`392bcea026823d2b0b38fd2bce89e1c16a522a17eb041bd11c002d904b45e76e`.
Exactly one of its 230 source files differs from R2: `benchmarks/lme_registry.py`
adds the clean-runtime direct-CLI import bootstrap. Extraction, benchmark recipe,
worker and scoring behavior are unchanged. Target gates and parent review are
pending at preparation time. The reviewed remote staging
root is `/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ`;
use its `offline-r3/candidate` as the read-only source mount, not the production
checkout. The combined test/asset tree is separately at `offline-r3/verification`.
Use the fresh sibling `q1-stock-v3` run root; never reuse an earlier run root.

## Exact question and recipe

The original remote 500-question S dataset remains at:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json`.
SHA256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
Never copy its content to the laptop or replace it with a one-row dataset.

The unchanged stock label-blind selector chooses source position 210 with
`--sample 1 --seed 53`; 53 is the smallest nonnegative seed doing so. Selector
source SHA256: `2f94272ef1383b1f5a7530644d28d1bdc9c7001231cd46297af392f33e74dc43`.
Remote preflight must independently confirm question ID `09ba9854`, 44 source
sessions and 479 messages. Seed selection deliberately targets a known failure;
this is a regression case, not a representative random sample or headline score.

The worker invokes the unmodified stock `longmemeval_adapter.py` CLI: S, sample1,
seed53, workers1, top-k15, auto-ability, permissive-default, full dream, ordinary
extraction canary, required healthy indexing, legacy-custom judge, `--no-prereg`,
full development split, fresh explicit checkpoint and `--keep-db`. Embeddings,
aggregation and episode granularity remain off. All three LLM roles explicitly
request `deepseek-flash` at `https://api.deepseek.com`, thinking disabled. This
requested alias is recorded, not represented as an immutable serving version or
silently equated with prior `deepseek-v4-flash` benchmark producers.

The stock per-question limits remain 100 dream cycles and 3,600 indexing seconds.
An external reviewed supervisor adds a 5,400-second total wall deadline plus ten
seconds for process-group cleanup. There is **no invented global paid-call or
dollar cap**. No reroll, automatic retry campaign, resume, canary bypass, summary
recovery command, input filtering, answer rewriting or scorer override is added.

## Remote staging and launch review

1. Root chooses a NEW private directory beneath Afrodite's benchmark staging
   tree. Stage this bundle only and separately identify the verified 230-file
   candidate directory. Do not mount the production checkout or memory database.
   Verify every helper hash and the manifest hash against local review; source
   hashes must equal the final corrected gate's unchanged source mapping.
   The source mount must contain exactly the 230 manifest-listed regular files,
   no other files of any extension (including hidden/root-level files), no
   symlinks anywhere and no devices/FIFOs/sockets. Empty regular directories are
   harmless. Do not use a source+tests+assets tree or a cache-bearing checkout.
2. Create empty UID1000 directories at `home`, `preflight-results`, and
   `live-results`, mode0700. The helper creates private benchmark/stores children
   under the latter and refuses an existing invocation/checkpoint. Preserve old
   results and do not repurpose a prior run's directories.
3. Call the pure `q1_stock_host.command(root, source, manifest_sha, live=False)`
   builder. It returns a `docker create` command, but performs no Docker action.
   Execute it only on Afrodite, then inspect that exact container ID using a
   restricted template covering ID, image, user, network, read-only root, entry
   point, command, init, mounts, resource limits and state—not environment or
   secrets. Require the returned plan exactly, before `docker start <exact-id>`.
4. Preflight must run with `network=none`, no credential mount and a clean exit.
   Independently check its counts, hashes, seed and unchanged source. This phase
   makes no provider calls. Do not proceed on missing, altered or partial output.
5. Only after final gate/root approval, build `live=True` with the same package
   and source pins. Verify the newly created container against that exact plan
   before starting its exact ID. `docker start` is detached by default; laptop
   sleep/disconnection does not interrupt the remote supervisor. Docker/host
   shutdown still stops it, and there is intentionally no auto-resume.

The existing image is pinned to
`sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5`.
The existing Python runtime is mounted read-only at `/home/node/hymem-env` from
`/opt/stacks/hermes/instance1/home/hymem-env`. Live mode mounts only the existing
remote `.env` file read-only at `/run/deepseek.env`. The reviewed credential loader
requires UID1000, file owner1000 and mode0600, accepts exactly one valid DeepSeek
key, and never sources a shell file. Credentials never appear in argv, stdin,
printed commands, manifests or laptop files. The worker places the key only in
its own fresh process environment immediately before the stock CLI runs.

The container has a read-only root, dropped capabilities, no-new-privileges,
UID1000, init/reaping, resource limits and a private empty home. Only the fresh
results bind is writable. Dataset, source, helpers, runtime and credential file
are read-only. The worker does not daemonize; the external supervisor owns its
process group and performs bounded termination/reaping. SIGTERM/SIGINT become
cancellation. SIGKILL/host failure cannot promise userspace cleanup; Docker
containment and post-run inspection remain mandatory.

## Completion evidence

Keep raw stdout/stderr, stores, checkpoint and result artifacts on Afrodite under
the private run directory. The supervisor verifies source/helper/dataset hashes
again after termination. Its receipt reports process exit and cleanup only;
exit0 is **not** interpreted as a passed question. Stock indexing failures can be
terminal scored failures while the CLI exits normally.

Root must inspect the exact stopped container: exited, no OOM, PID0, unchanged
image/config/mounts, and an ownership receipt showing the child reaped and group
absent. A failed/uncertain cleanup halts further work. Then use the stock registry
validation on the fresh archive/checkpoint and require exactly the pinned Q1,
no hidden attempts/resume, mechanical indexing completion, healthy item indexing,
reader/judge completion and truthful summary degradation metadata. A stale
summary is recorded separately, never relabelled current merely because items
indexed. Retain any failed run unchanged. One successful question still does not
verify full-500 completion, semantic quality or improved aggregate scores.

### Offline postvalidator

After the live container is stopped and its exact identity/state inspected,
call the pure `q1_stock_host.validation_command(root, source, manifest_sha)`.
Inspect and start its exact newly created container just as above. The plan has
`network=none`, no credential mount, and **all** bind mounts read-only; it reads
the existing `live-results` rather than creating another benchmark. It executes
`q1_stock_validate.py --manifest-sha256 <pinned-v3-manifest-sha>` with isolated
Python and bytecode writes disabled. It never imports or constructs a provider
client, runs a benchmark, opens the registry DB, or changes a checkpoint.

The validator verifies source/package/dataset identity, the latest pointer's
exact archive digest, the stock `validate_strict_artifact(..., require_scored=True)`
contract (including recomputed score and canary/indexing evidence), the physical
checkpoint's finalized projection and actual Q1 row, exactly one question
attempt/segment, and the terminal process receipt. It then checks the exact Q1
recipe and requires explicit cleanup-complete/reaped/group-absent evidence,
empty cleanup errors/warnings, actual measured reader and judge calls/HTTP
attempts, and healthy completed item indexing. It preserves and reports any
summary degradation separately. A valid but wrong answer is a performance
outcome, not a mechanical execution failure.

Only a bounded metadata JSON report reaches stdout: identity hashes, row counts,
score/verdict, item/summary health, and truthful per-role completion/HTTP/token
usage. Unavailable dollar cost remains null. Raw questions, answers, source
messages, credentials and logs stay remote. A structurally valid failed Q1
returns exit1 with available counters; malformed/missing evidence returns a
sanitized failure and no invented usage totals. Exit0 means this one scored Q1
completed mechanically, **not** full-500 readiness, semantic correctness,
representative quality or overall architecture acceptance. A metadata receipt
does not replace inspecting the actual stopped container and retained stores.
