# Read-only Q1 postvalidation repair

This is a **verifier-only successor**, not a source, package, benchmark, or result
repair. Keep the paid run's original `/diag` bundle, `/candidate` source,
`/results` artifacts, and dataset read-only and unchanged. Mount this helper
separately at `/audit/q1_stock_validate.py`, pin its SHA-256 in the host launch
receipt, and invoke it with the original `--manifest-sha256` in a separate
credential-free, network-disabled container. The helper loads the original
manifest-pinned `q1_stock_run.py` from `/diag`; the original complete package and
source inventories are checked before and after validation.

The original helper compared a raw physical checkpoint row directly to the
archive row. `AtomicCheckpoint.record()` persists the raw row, while
`AtomicCheckpoint.reconcile()` / `reconcile_results()` derive `strict_failure`
for the archive. Thus a normal successful checkpoint lacks a field that its
own stock archive legitimately includes, causing a false rejection.

The successor reproduces that stock projection explicitly. It first checks
the exact question ID, scored mode, verdict key, one closed entry, one integer
attempt, outcome consistency, and the complete closed attempt-history record.
A pre-existing raw `strict_failure` must be a boolean equal to the derived
failure state. Failed entries additionally derive the bounded recorded failure
and a false verdict as the stock checkpoint does; they can only produce a
failed-completion report. Every remaining field is compared with canonical JSON,
so extra/missing fields and boolean/integer/float substitutions cannot disappear
through Python's ordinary equality rules.

All existing strict-artifact, recipe, checkpoint-attestation, usage, indexing,
process-cleanup, and before/after source checks remain. No saved artifact is
rewritten. No benchmark is rerun. This helper makes no provider calls and never
claims full-500 readiness from one Q1 run.

## Regression command

Set `HYMEM_Q1_VERIFIER_SOURCE` to the independently frozen R5 source tree. The
tests verify all 231 source hashes before and after execution against the durable
R5 manifest, and construct temporary checkpoints using the **actual stock**
`AtomicCheckpoint`, `record`, `finalize`, and `reconcile` APIs.

```sh
HYMEM_Q1_VERIFIER_SOURCE=/absolute/path/to/verification-tree-r5 \
  PYTHONDONTWRITEBYTECODE=1 python -B -m pytest \
  tools/diagnostics/tests/test_lme_q1_postvalidation_v2.py \
  -q -p no:cacheprovider
```

The 42 controls prove the original false rejection, successful correct and wrong
answers, explicitly degraded summaries with healthy item indexing, honest failed-row handling, exact row/history binding, type-sensitive
tamper rejection, and preserved recipe/attestation/cleanup/usage/indexing gates.
Fixtures contain invented text only; they do not claim to be complete strict
LME artifacts. The real saved run must still pass the unchanged stock strict
artifact validator before this helper's additional Q1-specific checks.
