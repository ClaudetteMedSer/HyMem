# Afrodite embedding restart incident — 2026-09-26

## Diagnosis and scope

The reported `CONTAINER RESTARTING` is not evidence of a current crash loop.
At 06:37 UTC, embedding-server was running/healthy, with RestartCount=1 and
StartedAt=2026-09-25T17:53:46.656672883Z. Hermes1/2/3 were running with count0.
The existing monitor equates any lifetime RestartCount>0 with an ongoing
failure; its `Since` field records alert-signature onset, not container restart.

The historical restart is real and coincided with our isolated diagnostic's
unbounded embedding request: 1125 texts / 2675490 characters. Memory exhaustion
is plausible but not proven: old kernel logs are unavailable to the current
user, and current-incarnation OOM counters cannot establish historical cause.
Do not reproduce that request. The separately tested client batching fix is in
the frozen HyMem candidate, not deployed production.

## Sequential implementation and root verification

1. Read-only incident investigation by separate Sol agent: lifecycle, logs,
   resource metadata and tiny synthetic functional probes; no raw production
   text, credentials, restart or load test.
2. Separate Sol repairs the monitor locally. Preserve missing/down/unhealthy
   checks; distinguish active restarting and new restart events from historical
   counts. Persist observations per container identity; fail safely on invalid
   data and failed persistence. Root reviews actual code and independently
   exercises stable history, new failures, reset/recreation and corrupt state.
3. Rehearse on Afrodite with isolated state and notifications disabled. Deploy
   only reviewed exact bytes, compare original SHA, keep recoverable backup.
   No service restart is needed for the monitoring script. Verify a scheduled
   real check and stable service identities afterward.
4. Review the embedding server's own request boundary. If unbounded requests
   remain able to destabilize it, use a new Sol implementation with explicit
   size rejection, offline adversarial tests, isolated rehearsal and root
   verification before any server update. Do not mask errors or reset counters.
5. Resume the outstanding bounded private HyMem diagnostic only after current
   service health is established and the final offline suite receipt is sealed.
   Health-monitor success is not proof that the production dream blocker or
   full LME run is fixed.

## Evidence and boundaries

Original monitor `/opt/stacks/hermes/stack-health-check.sh` SHA256:
`c254b11eecd4322f5110edf2d6d458dd57a83922cef0f3e2416c4058cc9ca378`.
The local copy is `tools/ops/afrodite/stack-health-check.sh`.
`--no-notify` still updates alert bookkeeping; tests must never use production
alert state or published-status destination. Preserve all unrelated changes.
The final isolated HyMem suite reports 7748 passed / 4 skipped / 0 failures /
0 errors; root receipt/pin revalidation remains to be recorded independently.
No paid call, production database migration or service restart has occurred in
this incident investigation.

## Monitor repair verified and installed

Separate Sol implemented per-container ID/count/StartedAt observations with
atomic persistence. Root rejected a draft that passed full Docker inspect
through helper argv; the accepted source projects only required metadata,
never credential environment or healthcheck logs. Root's real rehearsal also
caught missing-Health template handling for three containers; Sol corrected it
before deployment. These were unshipped draft defects, not production incidents.

Final source SHA `b3de67297a3e24680c0bd02d227f4a10c75b9a1f6915a0478cde43e05b826e13`.
Root reran13 offline tests and six independent Linux scenarios using the real
production branch. Full isolated rehearsal reproduced original46/0/1, then
corrected initial46/14/0 (explicit new baselines), then stable46/0/0. It checked
all14 actual containers, including those without Docker healthchecks.

Installed at epoch1790405705 with the cron lock held and current Docker
observations independently compared to the fresh rehearsal baselines. Real
alert bookkeeping was left unchanged; next normal cron reports recovery.
No service restarted. Recoverable original and receipts reside at
`/opt/stacks/hermes/health-repair-20260926-v2/`. Earlier rejected rehearsal is
preserved separately. Honcho container-local /health returned200/statusok;
embedding /health200 and real two-text finite384D inference passed.

Next repair is server-internal bounded inference, not a new 16-text request
rejection: existing production clients can submit larger valid lists. Preserve
compatibility by processing smaller model batches while keeping complete,
ordered responses. Keep dependency/model/config/vector identity unchanged;
rehearse from the exact current image without touching live service first.

## Embedding server repair independently verified and deployed

Separate Sol implemented transparent16-text inference batches, a shared
request-wide embedding/rerank/load lock, and off-event-loop embedding handling.
Root reviewed the code and added4 independent controls to5 agent tests: all9
passed locally and in the exact pinned production dependency image, offline.
The1125-text failure shape was tested with a fake embedder only, proving71
bounded model invocations and complete ordered output without a giant real
inference request. Later-batch errors return no partial success, and release
the lock. Dependency/model/config/version/vector-identity settings unchanged.

Candidate image `sha256:4843a433a78681afe25a39e069f212b20c588d5f134621282c7c97872ec11669`
adds only reviewed server source to the exact old image's filesystem layers.
Source SHA `0bf31ebda3d13e2b99a68bbccf9dafa571714167a27458e435cf6b8ae17bb435`.
Rehearsal used a network-none container with its own copied model cache:
33 synthetic vectors matched the old server bit-for-bit; reranking returned
two finite scores. With reranker loaded, a16-long-text batch and concurrent
8-document rerank passed. Health remained responsive (max85ms first test,
8ms mixed test); no restart, OOM or cgroup max event. Peak3,079,892,992 bytes
under3GiB cap. This is bounded validation, not a universal no-OOM guarantee.

The canary stopped cleanly before production replacement. Installed the exact
tested image at2026-09-26T07:06:44Z; new container
`631437fd181f9a656c19e5efd235e05246867146ee2f168e4d9f62f0b69b08de`.
All three Hermes container IDs remained unchanged. Real embedding inference
from each Hermes returned finite384D output; production reranking and Honcho
health passed. Source inside the running image was independently hash-checked.
Docker health healthy, restarts0, OOM counters0. Production cold rerank load
reclaimed file cache at the3GiB cap (max1916 events, not OOM); afterward current
memory2.73GB/anon2.07GB and10-second pressure0.00%, with no further restart.

Rollback source/config, parent image tag, build/test/probe/deploy receipts are
retained in `/opt/stacks/hermes/embedding-repair-20260926`. Production compose,
dependencies, model and vector producer pins were not changed. The monitor's
07:07 run correctly recorded the planned replacement as `CONTAINER RESTART
OBSERVED`; a subsequent stable normal run must clear that one-time event.
Do not manually erase the lifecycle evidence or relabel it as a crash loop.

The separate pending HyMem application candidate passed its full offline gate;
the bounded private paid diagnostic was staged afterward. This incident repair
does not deploy HyMem schema64 or certify the production dream backlog/full LME.

The normal health check at epoch1790406658 had zero failures and one transient
paging-flow warning (~10MB/min over233seconds during model reload/test startup).
No thresholds or counters were changed. Subsequent kernel observations showed
only3 additional swap-out pages over173seconds and zero current memory stall.
The normal post-settle check at epoch1790406922 passed **46/0/0**, exit0, and
updated the real published/alert state through the usual recovery path.
The paid private test completed with48 completions,48 LLM HTTP attempts,
159 embedding requests and207 total HTTP attempts; provider accounting agrees.
It processed17 chunks and171 triples. Every embedding request stayed within
the16-text client bound; the repaired server remained healthy/restarts0/noOOM.
The scheduled07:22 health check independently reported46/0/0.

## Separate private application audit: aggregation remains blocked

The first offline audit failed in the audit reader, not the application:
SQLite's WAL-format header on a read-only closed snapshot caused ordinary
mode=ro access to attempt unavailable bookkeeping. Root reproduced this and
verified a separate Sol's versioned correction: immutable reads only for the
exact hash-sealed, closed files with no outstanding WAL/journal, retaining all
original checks and failed audit receipts. No paid campaign was repeated.

The corrected audit completed and correctly returned FAIL, not a clean-pass
claim. The target current publication advanced0→1 with its local replay proof;
integrity, foreign-key, canonical-drift, evidence-ledger and same-generation
conflict checks all passed. No new quarantine IDs were introduced. However,
aggregation_build_exceptions=1 and aggregation_fusion_failures=1. The latter
is a synthetic failure counter for a total aggregation exception, not proof of
a failed model fusion call. Audit result SHA:
`f80844f443264c8d37eec60d2084881e1ce3da3166378704c0cee83dd290e1d3`.

Root independently checked the exact stopped network-none audit container,
its exit1/PID0/noOOM/configuration, candidate pins and unchanged sealed source.
Read-only inspection then proved the deterministic blocker: baseline episode
vectors expected19/actual27; final expected36/actual44. Both have exactly8
surplus shadow keys, no missing expected keys, and no differing expected vector
bytes. Alignment is false before and after the dream. The bounded runner
unconditionally refuses this state before invoking aggregation. These are
inherited derived-index rows, not a new claim or extraction failure.

Sequential follow-up plan (separate Sol implementer, root acceptance):
1. Reproduce inherited surplus-only drift in a synthetic bounded dream.
2. Add a narrow fenced, deadline-aware atomic cleanup of surplus episode-vector
   shadow keys only when every expected key and vector is already exact.
   Missing/different/unverifiable vectors still fail closed; no full FTS/index
   rebuild, evidence mutation or new provider calls.
3. Verify rollback under deadline/lease loss, genuine-corruption refusal,
   source-proof/producer authority, existing aggregation and deadline tests.
4. Replay on a fresh private copy, preserving original paid/audit artifacts.
   Do not deploy the HyMem candidate or mark full LME ready until its gates pass.

Follow-up repair now passes382 root-run targeted/broader tests and a real
network-none private-copy replay:8 orphan vector-index entries removed,36 valid
vectors unchanged, all other tables unchanged, integrity clean, repeat no-op
and reopen aligned. Separate source and receipts are documented in
`2026-09-26-episode-shadow-surplus.md`. This is not a production HyMem rollout.
Latest live check still shows46/0/0 and the deployed embedding server healthy,
restarts0/noOOM. All diagnostic containers are stopped; no paid test is running.
