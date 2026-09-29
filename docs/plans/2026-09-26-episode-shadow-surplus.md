# Bounded-dream episode shadow surplus

## Confirmed evidence

The schema64 private dream completed17 chunks and171 triples using48 paid
completions. Its target publication/local proof and all database-integrity
checks passed; no new quarantine IDs. The final postflight correctly failed on
aggregation_build_exceptions=1 (the fusion-failure counter is synthesized).

Read-only queries on the sealed baseline/final copies prove8 inherited surplus
vec_episodes keys:19 expected/27 actual before,36 expected/44 actual afterward.
No expected key is missing; all expected vector bytes match. None of the8
extras maps to any current episode or durable episode_embedding ID. The bounded
runner's alignment guard therefore deterministically refuses aggregation.
This is not evidence of a new model refusal or source-memory corruption.

The trace journal reached its256-event diagnostic cap before aggregation; it
cannot supply the exception traceback. The exact alignment result plus the
unconditional bounded runner branch establish the blocking condition directly.
Do not claim a historical VACUUM caused these orphan rows: that provenance is
not established, even though the old warning suggests it.

## Sequential plan

1. Separate Sol diagnoses the code without editing accepted snapshots (done).
2. New Sol implements a surplus-only repair in a fresh copy of the frozen
   candidate. Recompute authoritative expected vectors in one fenced writer
   snapshot; require all expected keys/bytes exact before deleting only surplus
   derived keys. Verify alignment before commit. Deadline/lease loss rolls back.
   Missing/different/unverifiable state remains an explicit failure. No full
   FTS/shadow rebuild, provider call, evidence edits or schema change.
3. Root reviews the diff and independently tests source/producer authority,
   corruption refusal, atomic rollback, repeat no-op and bounded runner behavior.
   Rerun relevant DB, aggregation, deadline, lease and embedding regressions.
4. Separate Sol prepares an offline private-copy replay; root audits and runs
   it with network disabled and without credentials. Original sealed source is
   read-only, all non-episode-shadow table contents must stay unchanged. Preserve
   existing failed receipts. Return counts/booleans/hashes only.
5. Only after these gates decide the next candidate-level live/full-suite gate.
   Do not deploy this HyMem application candidate or certify full LME from a
   successful index repair alone.

## Scope and identities

Accepted prior candidate manifest:
`ed889d8970c6d7827315de34342996ad55182773c238922fbbdf8510d73c43a4`.
Its full suite passed7748/4 skipped; this receipt does not cover subsequent code.
Frozen source `/private/tmp/hymem-r7-cold-replay-tests.4BJkt5` stays unchanged.
New implementation copy `/private/tmp/hymem-episode-shadow-fix.RjDDmJ`.
Private final source SHA:
`f4f9cb8ec8247d27ab2981af58ec0c044fb76cc71356f2e6a757540d4eae59ae`.
Failed postflight SHA:
`f80844f443264c8d37eec60d2084881e1ce3da3166378704c0cee83dd290e1d3`.

The independent September26 embedding-server/monitor incident repair is already
deployed and healthy (46 checks passed, no warning/failure). This plan neither
restarts services nor changes production databases.

## Root acceptance completed

Sol's positive bounded-dream regression failed on the unchanged prior candidate
and passed on the repair. Root caught a draft's reliance on the permissive
alignment probe for final verification; the accepted helper now also compares
the exact remaining key/vector mapping directly, so SQL errors roll back.

Root independently ran43 focused DB/deadline/own acceptance tests and339 broader
aggregation/provenance/generation/lease/embedding/retention tests: all382 passed.
Five separate offline replay-worker controls also passed. These are targeted
gates, not a new full7748-test-suite receipt. Original frozen source and the
dirty main application's files were not edited.

The network-none Afrodite replay passed against a fresh copy of the exact final
private store:44→36 episode shadow rows,8 surplus rows removed, every valid
vector byte preserved. All other table contents and SQLite schema fingerprints
stayed identical, including other derived indexes. Integrity/FK/canonical/ledger/
same-generation checks passed; repeat was a no-op and reopen remained aligned.
The source's hash and file identity stayed unchanged. Root separately checked
the real exited container/PID0/exit0/noOOM, exact mounts, no credentials/network,
all481 candidate file hashes and the result receipt.

New application manifest:
`5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576`.
Replay result SHA:
`9a05dfe4b6383b263e3d510a77e3550aab7f140036d538d0cdebf650a75ac823`.
Remote stage is `cold-replay-dream-v1/episode-shadow-replay-v1` under the private
claim-conflict benchmark directory; original failed audit and paid receipts
remain intact. No additional paid calls were made for this repair.

Durable application/test patches are in `docs/patches/2026-09-26-episode-shadow-*`;
both dry-run apply cleanly to the accepted frozen base. Main checkout is not a
deployable substitute for the assembled candidate. No production HyMem rollout,
production dream, or full LME completion has been certified. A fresh end-to-end
candidate gate and controlled rollout remain separate steps.
