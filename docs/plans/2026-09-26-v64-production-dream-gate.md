# Proposed bounded production dream gate

Planning only: this document does not execute or authorize a new helper.
Root's current read-only schema63 census found 19 valid expected episode vectors,
27 actual vectors, eight surplus keys, zero missing/different valid vectors,
and 17 episodes with unverifiable vectors. This is an inherited production
baseline. Migration64 preserves existing data and does not prune these vectors.
An unchanged baseline after migration is not a new deployment regression,
but it does not satisfy final strict vector health.

After exact candidate481 source, full-suite and bounded private-paid receipts
pass, complete copy migration rehearsal and guarded stop/apply/migrate/start.
First verify source64, runtime/config/wrapper/hook preservation, separate role
environment profiles, health, integrity/FK, ledger/canonical/same-generation
and schema64 without declaring vector health passed. Record the post-start
strict vector baseline independently. The final postdeploy verifier still fails
closed on surplus, missing, different, or unverifiable episode vectors.

Prepare a new, separately pinned production worker and supervisor around the
official build_from_env()/dream()/shutdown_instance() APIs. Adapt ownership,
private receipts and cleanup from claim_conflict_production_dream.py, and paid
attempt metering from the reviewed private instrumented worker. Never rerun the
old fixed production helper or point the private clone worker at production.
Bind every candidate file, helper/supervisor bytes, current role profile,
schema64, phase1 generation, exact frozen target chunk and resolved session
hash to the newly passed private-paid receipt. Resolve the target session from
the database, require its hash to match that private receipt, and call exactly
dream(session_ids=[session], deadline=MonotonicDeadline.after(2700)).
Do not broaden to an unscoped dream if that session fails to resolve or has
changed. Keep the raw session identifier and any provider content private.

Recheck idle state and no open durable dream run immediately before launch.
Use the official database-backed dream lease and require exactly one owned
new run, skipped_locked=false, durable ended_at and no run error. A durable,
exclusive intent/ownership receipt precedes worker stdin release; a stale
intent forbids automatic retry. Preserve the effective Honcho environment,
including absent EXTRA_BODY and its aggregation policy. No credential
overrides, benchmark extra-body injection, or settings edits.

Choose and externally seal cooperative bounds only after the current private
run finishes. The current private run's completion cap may be 128; the older
64 cap is historical and is not presumed adequate for production aggregation.
Pin the actual reviewed private bounds for completion calls, LLM HTTP,
embedding HTTP and total HTTP attempts, and include embedding
batch size/text byte bounds from that worker. Meter actual attempts and tokens
by role and retain bounded private logs. The outer supervisor limit is
2760 seconds, with process-group cleanup and a separately bounded cleanup
window. Deadline/budget exhaustion, forced termination, ambiguous run
ownership or failed client shutdown is a failed gate requiring inspection.
It must never trigger automatic SQLite restore, restart, or a second dream.

Independent postflight after the worker closes must verify candidate source,
runtime/config/role profiles, schema64, integrity/FK, ledger/canonical,
same-generation claim consistency, target current publication, durable dream
ownership/cleanup, aggregation completion and exact episode-vector comparison.
Report expected/actual counts, surplus/missing/different, and unverifiable
episodes. A targeted dream may populate embeddings needed by the accepted
surplus-only repair, then the normal aggregation path may prune the eight
inherited extras. Neither result is assumed in advance.

If the official targeted dream leaves unrelated unverifiable episodes or
surplus vectors, stop and report the remaining baseline. Prepare a fresh
copy rehearsal for any narrowly scoped derived maintenance proposal and have
root review the proof and source pins before implementation. Do not silently
expand the dream scope, invent a partial-vector pruning rule, or weaken the
accepted application's conservative repair eligibility.

Only a fully verified production dream/postflight and successful final
strict postdeploy receipt permit declaring production vector health restored.
Historical summary degradation remains a separately reported condition.

The tested private bounds are 128 completions, 384 LLM HTTP attempts,
512 embedding HTTP attempts and 896 combined attempts, with 2700-second
cooperative deadline and 2760-second supervisor wait. These are the proposed
production bounds after final independent v2 audit and root review; the
private run's bounded completion does not by itself prove session convergence.
