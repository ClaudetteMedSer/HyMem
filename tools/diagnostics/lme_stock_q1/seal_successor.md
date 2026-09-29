# Historical R5 Q1 sealer

Execution note (2026-09-24): this exact sealer was used for the frozen R5 package,
whose paid Q1 run completed and answered correctly. Its original bundled
postvalidator has a recorded raw-versus-reconciled checkpoint comparison bug.
Keep that package immutable for reproducibility; use the separately pinned
`postvalidation_v2/q1_stock_validate.py` for validating the saved run. The v2
README documents the repair and real-checkpoint regression gate. Do not treat
the old bundled validator as the current verification recipe. See
`docs/plans/2026-09-24-lme-q1-verification-result.md` for the completed result.

The preparation specification below is retained as historical documentation.

`seal_successor.py` is inert on import and defaults to inspection. It has no
network, subprocess, Docker, provider or credential operation. The only code it
executes is the hash-pinned pending helper definitions and the independently
pinned label-blind selector function. It does not import the candidate app.
The 2026-09-24 controls use invented two-file source fixtures, not a real R5
candidate. Pending helpers and the historical seven-file package are unchanged.

## Input and approval boundary

Wait for the parent's acceptance of the final R5 tree. Supply the independently
approved manifest SHA-256 explicitly. Its `revision` must be `r5`, and its
`source_sha256` map must be nonempty. `--layout source` accepts precisely that
map. `--layout verification` additionally requires disjoint nonempty
`test_sha256` and `auxiliary_sha256` maps and checks that entire combined tree,
but copies **only** the source map. The full target-runtime test tree is a
separate parent-managed upload; it is not bundled into Q1.

Every input tree entry is inventoried before file contents are read. Unknown
files, any symlink, special files, path traversal, duplicate JSON keys and hash
drift fail closed. Source files must be `.py`, `.sql` or `.toml`; no credential,
database or benchmark-data files are copied. Helper pins are literal constants
in the sealer, not recomputed and silently accepted from a changed directory.
The selector, source index 210/seed 53, dataset SHA, model, endpoint, image and
stock arguments are bound to the reviewed predecessor. The file count is never
hardcoded. Expected target-runtime versions are metadata, not a test-pass claim.

Read-only inspection command (replace placeholders only after acceptance):

```sh
python3 -I -B tools/diagnostics/lme_stock_q1/seal_successor.py \
  --source-manifest /absolute/final-r5-manifest.json \
  --approved-manifest-sha256 APPROVED_SHA256 \
  --tree /absolute/frozen-r5-tree --layout verification \
  --remote-root /opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/q1-stock-v4 \
  --remote-source /opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r5/candidate
```

Only after separate parent approval, add `--seal-approved-source --output
/absolute/fresh/local-output`. The parent directory must already exist. Output
must be outside the input tree and the maintained helper directory. Existing
targets (including symlinks) are rejected. Files are created exclusively at
mode 0400 under new mode-0700 directories; partial output is retained on failure,
never overwritten, automatically retried or deleted.

## Exact output/upload interface

The output root contains:

- `candidate/`: source-map files only, preserving every exact relative path.
- `bundle/`: seven pinned pending helper/readme files plus `manifest.json`.
- `input-manifest.json`: original approved manifest bytes unchanged.
- `seal-receipt.json`: exact SHA-256 map of every other output file, source
  manifest pin, package manifest pin and truthful zero-call/non-readiness flags.

JSON artifacts use UTF-8, ASCII escapes, sorted keys, two-space indentation and
one trailing newline; NaN and Infinity are rejected. Every file pin is the
lowercase SHA-256 of exact bytes. The package `source_mapping_sha256` instead
hashes the source map's sorted, ASCII-escaped compact JSON with no newline,
matching the runner's `canonical()` function. Identical inputs produce identical
output bytes even under different fresh output paths.

Before later upload, retain the independently computed hash of
`seal-receipt.json`. Verify an existing output without writes using:

```sh
python3 -I -B tools/diagnostics/lme_stock_q1/seal_successor.py \
  --verify-sealed /absolute/sealed-output --receipt-sha256 RECEIPT_SHA256
```

The intended later remote destinations are `offline-r5/candidate` for the pure
candidate tree and `q1-stock-v4/bundle` for the bundle, under the fixed task root.
The sealer never contacts those paths, installs anything, creates a container,
reads `.env`, or launches a benchmark. It does not build a transfer archive.
The parent must separately review transfer inventory, target-runtime gates,
container configuration and execution authority. A sealed package is not proof
that tests, target-runtime preflight, Q1 or all 500 LME questions will pass.
