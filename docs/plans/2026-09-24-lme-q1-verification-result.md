# R5 LME Q1 verification — completed

The previously failing stock LME question `09ba9854` completed and was judged
correct on its first fresh attempt. The independent, offline verifier accepted
the scored archive, physical checkpoint, accounting, source identity and process
cleanup. This is a targeted end-to-end regression pass, not a full-500 result or
a production deployment.

## Verified outcome

- Duration: 1,624.38 seconds, approximately 27 minutes.
- Expected/attempted/completed questions: 1/1/1; failed/missing: 0/0.
- Reader and judge: one measured call each; answer judged correct.
- Five dream cycles completed; no quarantined extraction work at completion.
- Item indexing: complete and healthy.
- Summary health: **degraded**, explicitly reported for 10 sessions. The outcome
  is `success_with_summary_degradation`, not fully healthy summaries. Valid
  source-backed records remain independent of rejected summaries.
- Canary v18: passed, eight completions/eight HTTP attempts, client closed.
- Run, offline validator and diagnostic containers exited with no OOM and PID 0;
  the benchmark supervisor confirmed child reaping and an absent process group.
- One attempt only: no paid reroll, resume, modified dataset or altered result.

The unchanged 500-question dataset was read with the stock sample-1/seed-53
selector to choose this known regression question (source index 210). All its
44 sessions and 479 messages were included. This was deliberately targeted, not
a representative accuracy sample. The recipe used `deepseek-flash`, thinking
disabled, full dreaming, healthy-indexing enforcement, lexical retrieval, no
embeddings/aggregation/episode-granularity, and the legacy-custom judge protocol.

## Offline gates

- Parent-reconstructed full R5 suite: **7,243 passed**, four predeclared
  raw-backend-inapplicable skips, zero failures/errors; 7,247 exact nodes and all
  467 source/test/support file hashes reconciled.
- Afrodite runtime gate: **1,040 passed**, the same four declared skips, zero
  failures/errors; exact inventories checked before and after. These overlapping
  test counts must not be added as unique coverage.
- Genuine Afrodite SDK/stock-CLI startup check: passed without provider calls.
- Verifier repair: **42 regression tests passed**, independently rerun by the
  parent against real frozen-R5 checkpoint APIs.

## Verifier-only issue found and fixed

The original extra postvalidator rejected the successful run because it compared
a physical raw checkpoint row directly with a reconciled archive row. Stock
reconciliation adds `strict_failure=False`; every other shared value was exactly
equal. The original rejection is preserved.

A separate agent implemented a verifier-only successor mounted under `/audit`.
It reproduces the stock projection, binds every remaining field using canonical
JSON, checks exact IDs and types plus the one-attempt history, and retains all
strict artifact, recipe, attestation, accounting and cleanup gates. Legitimate
failed-row normalization still reports failure, never readiness. Real writer
regressions reproduce the old false rejection and reject tampered evidence.

The corrected verifier accepted the **same unchanged saved run**, offline and
read-only. No application code, original sealed helper, benchmark artifact,
checkpoint, source dataset or paid execution was rewritten or rerun.

## Paid accounting

Including the separately accounted canary:

- 806 completions, 806 HTTP attempts, 806 admitted responses.
- 2,469,114 prompt tokens + 145,229 completion tokens = **2,614,343 tokens**.
- Dollar cost is unavailable in the instrumentation; no zero-cost or price
  estimate is substituted.
- Diagnosis, verifier repair and repeated offline validation used zero additional
  provider calls. Previous retained-case diagnostics are excluded from these totals.

## Exact receipts and scope

R5 source manifest:
`c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc`.
Original Q1 package manifest:
`5cdb8db3213229aed6f140da6747ff2a1048433e75437b3a7feda6521f0b73a3`.
Corrected verifier:
`e322d4b0c9fd3f0c455d409186b4788a9287f076f27bb8729a65bb4d0c0bc9cb`.
Saved archive:
`85c7e13f5eb259fbcfc668b348f212141af2e1ec09363cd5faeb5dfc61145c25`.
Physical checkpoint:
`e64b32876078b0a82a004af28fec86fda28f49bcf67b8965a57ff12f7c8dece5`.

Durable local evidence is under `docs/patches/`:

- `2026-09-24-parent-r5-full-aggregate.json`
- `2026-09-24-lme-r5-target-receipt.json` and companion JUnit/supervisor/terminal files
- `2026-09-24-parent-q1-v4-preflight.json`
- `2026-09-24-q1-v4-postvalidation-rejection.json`
- `2026-09-24-parent-q1-postvalidation-v2-final.xml`
- `2026-09-24-q1-v4-postvalidation-v2.json`
- `2026-09-24-q1-v4-validation-v2-terminal.json`
- `2026-09-24-q1-v4-paid-usage.json`

Raw benchmark content and kept stores remain on Afrodite. The full 500-question
benchmark has **not** been run by this verification. Production Hermes services
and stores were not changed, and the dirty main application checkout remains
untouched. This verifies the isolated R5 candidate, not an installed production
upgrade or universal semantic correctness.
