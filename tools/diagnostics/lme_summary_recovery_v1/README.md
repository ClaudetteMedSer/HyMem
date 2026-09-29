# One-shot independent summary recovery on the Q1 benchmark clone

This diagnostic runs the unchanged R5 `run_summary_recovery` once on a new
SQLite-backup clone of the saved, completed benchmark-only Q1 store. It does not
run dreaming, extraction, a migration, reader/judge scoring, production work,
or a second recovery invocation. The original scored store and artifacts stay
read-only and unchanged; the diagnostic cannot alter their score or receipts.

## Parent-owned setup and launch

Use a fresh private output/work directory and a separately pinned worker file.
In a network-disabled staging step, create the clone with SQLite's `backup()`
from a `mode=ro` source connection. Mount the entire original store directory
read-only: its `-wal` and `-shm` files exist and must be visible. Do **not** use
`immutable=1`, an ordinary database-file copy, or omit WAL content.

Example container paths (all except `/work` and `/results` read-only):

```text
/candidate                       original 231-file R5 source tree
/r5/manifest.json                 original R5 manifest, pinned in worker
/diag/worker.py                   separately pinned diagnostic worker
/reference/hymem.sqlite           original store, with sibling WAL/SHM
/work/hymem.sqlite                fresh parent-prepared SQLite backup clone
/run/deepseek.env                 existing UID-1000/mode-0600 credential mount
/results                         fresh private receipts directory
```

The original store lives under:
`q1-stock-v4/live-results/stores/hymem-lme-ktsn8xyj/`.

Run a UID-1000, read-only-root, capability-dropped, no-new-privileges isolated
container using the same pinned image/runtime as Q1. Parent owns creation,
complete mount/resource/network inspection, and external process-group
supervision (1,860 seconds plus bounded cleanup). No credential content or raw
benchmark text is transferred into reports or to the local machine.

```sh
python3 -I -B /diag/worker.py \
  --source /candidate --r5-manifest /r5/manifest.json \
  --clone /work/hymem.sqlite --reference /reference/hymem.sqlite \
  --credential-file /run/deepseek.env --receipt /results/recovery.json
```

Before any paid launch, run the same worker with `--preflight`, omit
`--credential-file`, and choose a separate `/results/preflight.json` receipt in a
network-disabled container with **no credential mount**. It verifies the same
source, clone, baseline, and actual SDK producer binding using only a synthetic
key, closes that client, and requires zero calls and no store changes.

For paid mode, the parent supervisor must establish process ownership before
writing `worker.encoded(worker.AUTHORIZATION) + b'\n'` to stdin and closing the
pipe. The worker reads at most 256 bytes and admits only that exact canonical
frame before it reads credentials or creates a provider client. The authorization
dict binds this protocol version and the fixed R5 manifest pin; the separate
parent launch manifest pins the worker's own bytes.

Bounds are code-fixed: **100 logical completions, 3 rejection attempts per exact
slice, 8,000 input characters, 3,072 maximum output tokens, 1,800 seconds**.
The stock worker stops a rejected session after one held result in this
invocation; it does not reroll it. The fixed source's transport retry count is
three, so the derived overall HTTP-attempt ceiling is 300. Actual HTTP attempts,
admitted responses, tokens and unavailable costs are reported separately.
R5 accounts received rejected responses before admission; uncertain attempts
remain explicitly unavailable, never an invented zero-dollar charge.

## Verification and meaning of results

Before paid work, the worker verifies the source inventory, runtime versions,
clone/original full logical database equivalence, integrity and foreign keys,
schema 63, no existing private recovery or lease, and exactly ten well-formed
degraded sessions with fully indexed source. The real owned OpenAI-compatible
client is pinned to `deepseek-flash` at `https://api.deepseek.com`, thinking
disabled, with no inherited extra body. Its exact producer identity must match
the saved Q1 store's generation registry.

After the operation, every schema object and table must remain identical except
the private `summary_recovery` table and the nine explicitly named summary fields
in `sessions`. This binds **all** item/frontier, episode, graph, source, coverage,
provenance, vector, retry, quarantine and generation state, not just row counts.
Published summaries must reach the unchanged complete item frontier; unfinished
drafts must remain private and source-proof-valid. Healthy pre-existing and
operator/legacy summaries are protected. Integrity/FK checks, exact call
accounting, released lease, stopped heartbeat and client/database closure are
required. Original-store and source fingerprints are checked again afterward.

- `recovered_all`: all ten formerly degraded summaries now have current,
  source-complete publication metadata, with every invariant preserved.
- `honestly_degraded`: bounded work ended with held responses and/or private
  partial walks, without falsely advancing published summary coverage. This is
  a valid diagnostic outcome, **not** a claim that all summaries recovered.
- `error`: provider, accounting, identity, invariant, or cleanup failure. Paid
  usage is retained when available; no rerun is authorized by the worker.

Exit 0 means the diagnostic completed without an integrity/cleanup fault, not
necessarily `recovered_all`. The parent must inspect `status`, health counters,
and the external supervisor's final cleanup evidence. Summary semantic fidelity
is not proven by this structural recovery test, and neither Q1 nor this repair
certifies a full-500 benchmark run.

The 32 offline controls use the genuine frozen worker and SDK response boundary with
synthetic provider replies, including current publication, empty/truncated
holds, private partial drafts, fatal uncertainty, immutable-table violations,
operator preservation, paid accounting after audit failure, and WAL-aware
backup equivalence, a genuine zero-call SDK preflight, strict stdin authority,
and source inventory rejection for special files and traversal failures. Run
with `HYMEM_Q1_VERIFIER_SOURCE` pointing at frozen R5:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest \
  tools/diagnostics/tests/test_lme_summary_recovery_v1.py -q -p no:cacheprovider
```
