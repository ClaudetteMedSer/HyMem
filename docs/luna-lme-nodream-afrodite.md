# Luna-only LongMemEval-S no-dream run on Afrodite

This path evaluates all 500 source-ordered LongMemEval-S questions with the isolated HyMem message search path (`--no-dream`, `embeddings=False`, `aggregation_nodes=False`). Luna handles the reader, local judge, and any message rerank. Each one-shot window also makes one small Luna route probe. The code injects the SIWC OAuth Responses client into HyMem and sets the pipeline endpoint to `https://api.openai.com/v1`; receipts require `api_fallback_allowed=false`. The ordinary `benchmarks/longmemeval_adapter.py` still defaults to DeepSeek, so use the pinned commands below.

This is a complete **Luna no-dream diagnostic** score after the offline verifier confirms all 500 unique source IDs. It is not a canonical R9 artifact or an official LongMemEval model score. Billing is `siwc_server_enforced_plan_or_existing_credits_v1`: the route may use existing credits, so it does not prove zero credit spend.

## Prepare and check the one-row smoke capsule

Use the pinned runtime; system Python lacks the SIWC JWT dependency. Capsule roots are private directories directly under `/home/atta` and are one-shot.

```bash
RUNTIME=/home/atta/.hymem-siwc-runtime-v1/bin/python
LAUNCHER=/home/atta/HyMem-Luna-LME/tools/diagnostics/siwc_lme_launch_nodream_v12.py
SMOKE=/home/atta/.hymem-siwc-lme-nodream-smoke-20261003a
"$RUNTIME" -I -B "$LAUNCHER" --preflight-root "$SMOKE"
"$RUNTIME" -I -B "$LAUNCHER" --status
```

The smoke capsule has already been launched. Its receipt SHA-256 is `8ad6be2c2716850aa70b74e5e24a1da368a570c12371470f7035b041893d3be1`. To prepare a new one-row capsule instead, choose a fresh `.hymem-siwc-lme-nodream-*` root and use `--assemble-root ROOT --offset 0 --questions 1 --workers 1 --allow-repeat-window`. Assembly and preflight make zero model calls. Review the receipt before launching with `--launch-root ROOT --receipt-sha256 RECEIPT_SHA256_FROM_PREPARE`.

## Prepare the 500-row campaign

The campaign adopts the completed smoke row at offset 0 and plans the remaining 499 rows in windows of at most four. Preparation makes zero model calls and writes a private, immutable `manifest.json`. Choose explicit cumulative ceilings after reviewing the smoke usage and available subscription limits; the figures below admit one 8-million-token window at a time and can be extended explicitly if needed.

```bash
DRIVER=/home/atta/HyMem-Luna-LME/tools/diagnostics/siwc_lme_campaign_nodream_v2.py
CAMPAIGN=/home/atta/.hymem-siwc-lme-nodream-campaign-luna500-20261003a
"$RUNTIME" -I -B "$DRIVER" --prepare-campaign "$CAMPAIGN" \
  --adopt-root "$SMOKE" --batch-size 4 \
  --max-known-tokens 25000000 --max-turns 10000 --max-wall-seconds 86400
"$RUNTIME" -I -B "$DRIVER" --status-campaign "$CAMPAIGN"
```

Keep the `manifest_sha256` from the prepare output and review the exact capsule roots, row offsets, repeat flags, source pins, Luna model, and ceilings in `manifest.json`. The driver verifies the smoke capsule and every later window with the independent offline verifier before advancing. The first window at offset 0 is adopted; it will not be rerun.

Run the no-call campaign preflight with that pinned manifest SHA before dispatch:

```bash
"$RUNTIME" -I -B "$DRIVER" --preflight-campaign "$CAMPAIGN" \
  --manifest-sha256 MANIFEST_SHA256_FROM_PREPARE
```

## Run, stop, and resume

The foreground command blocks while it runs sequential windows. One process holds the campaign lock; each launch also takes the shared selection lock and is admitted only after prior benchmark units stop.

```bash
"$RUNTIME" -I -B "$DRIVER" --run-campaign "$CAMPAIGN" \
  --manifest-sha256 MANIFEST_SHA256_FROM_PREPARE
```

For unattended execution, put that exact command in a user systemd unit named outside `hymem-siwc-lme-nodream-*`, for example `lme-luna-controller-20261003a.service`, with `Restart=no`. Stopping the controller unit prevents future windows. A question unit already launched by it is separate and may finish; stopping that question unit makes its window fail. `--status-campaign` is read-only and reports the last safe reason, offset, observed totals, and capsule states. Restart the same command with the same manifest SHA to continue after an interrupted controller; a failed one-shot window is never retried automatically.

If the driver pauses at a cumulative ceiling, increase it with an explicit chained amendment, then rerun the same manifest. Use `budget_revision_sha256` from `--status-campaign` as the previous pin:

```bash
"$RUNTIME" -I -B "$DRIVER" --extend-campaign "$CAMPAIGN" \
  --manifest-sha256 MANIFEST_SHA256_FROM_PREPARE \
  --previous-budget-sha256 CURRENT_BUDGET_REVISION_SHA256 \
  --max-known-tokens NEW_TOTAL_TOKEN_CEILING \
  --max-turns NEW_TOTAL_TURN_CEILING \
  --max-wall-seconds NEW_TOTAL_WALL_SECONDS_CEILING
```

After a failed window, make a new campaign root and explicitly carry forward only its verified completed prefix. The failed offset needs `--allow-prior-overlap`; its old capsule remains intact for inspection.

```bash
NEXT=/home/atta/.hymem-siwc-lme-nodream-campaign-luna500-recovery01
"$RUNTIME" -I -B "$DRIVER" --prepare-campaign "$NEXT" \
  --resume-from-campaign "$CAMPAIGN" --allow-prior-overlap \
  --batch-size 4 --max-known-tokens 25000000 \
  --max-turns 10000 --max-wall-seconds 86400
```

The new manifest binds the previous manifest hash and the exact completed capsule receipts. It cannot mix full-dream and no-dream windows because the offline verifier checks one source graph and mode for the entire campaign.

## Verify the final 500-row diagnostic score

The driver writes `campaign-result.json` only after all 500 rows pass offline verification. Recheck it independently:

```bash
DATASET=/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json
VERIFY=/home/atta/HyMem-Luna-LME/tools/diagnostics/siwc_lme_aggregate_v1.py
"$RUNTIME" -I -B "$VERIFY" --dataset "$DATASET" \
  --campaign-result "$CAMPAIGN/campaign-result.json"
```

The verifier checks 500 unique source IDs, exact source-row hashes, one-shot receipts, checkpoint verdicts, private reader and judge evidence, model-route accounting, and shared source identity. It prints the Luna no-dream diagnostic accuracy with `official_model_score=false`.
