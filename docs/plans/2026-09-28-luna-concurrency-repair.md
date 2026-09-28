# Luna LME timeout diagnosis and bounded concurrency

## Request and boundaries

The user asked to diagnose the failed Luna pilot, make a plan, implement with
separate Sol agents and independently verify each change, including concurrent
questions. Keep the failed v2 pilot, its frozen sources, production stores and
the cancelled DeepSeek experiment unchanged. No resumption or automatic reroll.
New experiments are explicitly versioned; the model and strict healthy-indexing
and healthy-summary requirements do not change.

## Evidence

The source-pinned v2 failure is `timeout_during_cycle` after 3,605.86 indexing
seconds, during cycle three. Both completed cycles processed 50 chunks and
recorded zero chunk-extraction failures (104 and 140 extraction calls). Cycle
one recorded three digest failures and one fact failure; cycle two recorded one
digest failure. These are real retryable failures, not proof of universal model
reliability. The terminal accounting is 472 turns and 3,341,389 known tokens,
including the passing canary. No question was scored. Cleanup completed.

A query-only SQLite snapshot under the exact original producer/config confirmed
170 pending chunks and three pending digests at shutdown, with three missing/
degraded summaries. Quarantine, malformed state, coverage failures and terminal
losses were all zero. There were 273 extraction chunks in total. Cycle one took
39m07s; cycle two took 19m00s; cycle three was interrupted after 1m55s. Three
additional chunks had committed during that interrupted cycle. Both complete
cycle reports accurately show their 50-chunk budget exhaustion. This is a real
unfinished backlog, not a monitor false alarm. The original 4-million-token cap
would also leave little headroom beyond the elapsed-time failure.

Three fresh no-inference checks of the pinned Afrodite transport measured total
startup/preflight/cleanup time of 1.670, 1.451 and 1.291 seconds. Startup itself
was about 0.02 seconds; preflight took 1.2–1.6 seconds. All cleaned up, with no
model calls and a minimum reported quota of 80%. These later samples suggest
per-call isolation overhead matters, but do not explain the whole hour or
measure the prior run's actual model latency. Preserve isolation rather than
silently reusing conversation history as a performance shortcut.

The reported 178 pending units were the last **completed-cycle** snapshot,
not a terminal store census. Convergence intentionally does not complete a
snapshot while a cycle is interrupted. Do not call this a stuck chunk, silently
declare pending work healthy, or infer per-phase latency from total runtime.

The existing client is deliberately single-flight; simply sharing it between
workers is incorrect. Its fixed 90-minute/4-million-token pilot envelope is
also unsuitable as an implicit per-worker campaign budget. Per-question
concurrency improves aggregate throughput, not a question's own indexing time.

## Ordered implementation and verification

1. **Shared budget and concurrent transport (Sol, then root verification).**
   Add a separately versioned module using the reviewed App Server isolation
   helpers, fresh process/thread per invocation, exact Luna model, subscription
   auth only, no tools/API fallback/credits, and the existing 25% quota floor.
   Explicit finite aggregate and per-question turn/token/time limits; atomic
   admission and settlement; at most two in-flight turns. Unknown usage,
   isolation/quota/transport/cleanup errors stop future admissions globally.
   Record preflight, inference and cleanup timing separately. Tokens are a
   stop-before-next-known-usage limit, not a provider-enforced output ceiling;
   already-admitted calls can overshoot. Preserve known usage separately from
   unknown usage and never silently multiply the aggregate limit by workers.
   Test true overlap, admission races, exact caps, unknown usage, exceptions,
   failed cleanup, per-question limits, wall deadlines and quota failure.
2. **Concurrent question runner (new Sol task, after step 1 acceptance).**
   New versioned controller with one campaign-entry canary, source-ordered
   selection/results, distinct database/directory/client per question, bounded
   question workers, no automatic retries/resume. Reuse frozen evaluation
   semantics; no parallel writes to one question's store. Configurable finite
   indexing deadline independent of question/campaign wall limits. Retain
   strict health/summary/scoring validation. Atomically publish private
   progress/results plus source-free counts and safe status. Label last-cycle
   health as stale on interruption and include an independent post-run read-only
   census where safely available. Record failed questions without losing
   completed peer results; transport/safety/accounting errors halt the campaign.
3. **Independent verification (root, skeptical Sol review).**
   Run mock transport races, real frozen-runtime synthetic stores, disjoint
   DB/producer checks, deterministic ordering, false-answer-vs-execution-failure,
   canary failure, interrupted worker, artifact/cleanup failure and no hidden
   API construction. Re-run existing transport/pilot/canary regression tests.
   Fix any review findings before moving on.
4. **Bounded live verification, only after offline gates.**
   A tiny two-worker synthetic transport/isolation check may use the subscription
   route with explicit low caps. Full question completion needs a separately
   recorded fresh-run envelope sized from the census; no claim of end-to-end LME
   success based on mocks, a passing canary, or two synthetic calls. Keep all
   benchmark/model text and retained stores private on Afrodite.

## Acceptance and limitations

Success for implementation means verified bounded overlap with isolated stores,
global accounting/stop, honest progress and unchanged evaluation strictness.
Success for LME itself requires a fresh question to finish, validate healthy
indexing/summaries, produce a parseable score, and clean up. Increasing a timeout
is an explicitly new resource envelope, not an algorithmic speed-up. Subscription
temperature/output-cap/JSON-mode remain unsupported controls, so these are
experimental Luna comparisons rather than canonical API-equivalent scores.

The old Luna monitor is paused following terminal failure; the old DeepSeek
monitor remains paused. No automatic relaunch is part of this repair.

## Fix 1 verified

The first Sol implementation is in `benchmarks/codex_subscription_concurrent.py`
(SHA256 `e5f1eacbb02f809ee6246449b672dfcc839069da226230572913612acece069e`).
Root reviewed it, added independent race/cleanup controls and reran 101 focused
and existing transport tests successfully. A separate clean-process check
loaded the actual frozen extractor and proved a budget stop escapes after one
call without becoming an extraction retry. Stops are control-flow exceptions;
fatal admission failures publish the shared stop before potentially blocking
cleanup. This prevents a peer from starting during that cleanup window.

A fresh invented-text-only Afrodite probe ran exactly two subscription turns,
with two distinct ephemeral threads, matching outputs and 2.916 seconds of
overlapping model-call intervals. Wall time was 11.163 seconds. Both owned
process groups were verified gone; accounting was complete and no invocation
remained. Known reported usage was **8,464 tokens** (4,233 + 4,231), including
request overhead despite tiny answers. This exceeded the 2,000-token
stop-before-next threshold because both first turns had already been admitted;
the two-turn hard cap held. There is no claimed hard output-token ceiling.
Failed-turn usage is unknown; reported known tokens explicitly identify the
completed-turn subtotal. No LME question or production memory was sent.

Only diagnostic code was staged in
`/home/atta/.hymem-luna-concurrency-check-WK4guE`; the failed pilot and frozen
candidate remain unchanged. The second Sol implementation starts only after
this acceptance. No fresh full-LME run has been launched.

## Proposed next live envelope (not launched)

Use two source-ordered questions and two workers, with one entry canary. Indexing
gets 10,800 seconds per question (3 hours); each complete question gets 12,600
seconds (3.5 hours), 2,000 turns and a 12-million-known-token stop-before-next
threshold. The entry canary gets 12 turns, 160,000 known tokens and 600 seconds.
For this two-question validation only, the aggregate ceiling is 4,012 turns,
24,160,000 known tokens and 14,400 seconds (4 hours), including the canary.
The 25% quota admission floor remains mandatory. These are validation limits,
not a forecast of actual usage or an assertion that three hours guarantees
convergence. Existing semantics and all health gates remain unchanged.

Before launching, make a new private source-pinned bundle and receipt; enforce
the campaign wall limit plus cleanup allowance in a separate server-owned
process group/cgroup, verify disk space and memory capacity for two workers,
and install a new metadata-only monitor for that exact unit. Do not reuse the
old pilot's 90-minute unit or its paused monitor. Do not resume any old run.

The implementation can stream up to 500 explicitly selected questions through
two workers without buffering the full corpus. This is capacity, not launch
authorization or evidence that account quota can sustain all 500. Do not infer
a full-run budget from the two-question envelope above.

## Fix 2 accepted and final verification

The separate Sol controller implementation is
`tools/diagnostics/luna_subscription_lme_multi.py`, SHA256
`83fb27a9ee86c7f1d25ab7c6775dec31e9540321347971609de47a065734eef5`.
It streams prevalidated, individually fingerprinted questions through at most
two queued workers, rather than retaining the full corpus. It gives each
question its own store/client and preserves source-order results. The CLI
requires explicit indexing, question, canary and aggregate budgets. Incorrect
answers remain valid scores; unhealthy indexing does not. Failed local
questions retain completed peers; global safety/accounting failures stop new
admissions. In-flight progress is explicitly incomplete. Neither stale
last-cycle status nor successful process exit is promoted to healthy completion.

Root's actual frozen-runtime control found that a non-swallowable resource
stop released the dream lease but left its lifecycle row open. The controller
now terminalizes only open `dream_runs` telemetry in its own freshly created
question DB after unwinding, with an honest interruption reason. It verifies
the exact path, rejects symlinks, takes a nonblocking SQLite write transaction,
refuses any active lease and verifies zero open run rows. It never changes
messages, chunks, cursors, processed markers, claims or health evidence. The
root test proves all rows in messages/chunks/sessions/processed_chunks/claim
observations are unchanged, one run row is closed, and a second cleanup is a
no-op. Failure of these checks prevents a clean-completion claim.

Final root verification: **174 focused tests passed**, including existing
transport/pilot/canary/isolation/containment regressions, independent admission
races, real frozen-store isolation and producer/fork identity, an actual
coordinator with three questions on two workers, incorrect-score handling,
peer preservation, progress-write faults escaping the frozen extractor without
rerolls, worker/controller interrupts, prior-output preservation, atomic writes,
streamed source integrity and interruption housekeeping. Compilation passes.
A separate Sol skeptical review found no remaining blocking issue in this
implementation scope. The original transport, pilot and reader hashes are
unchanged. This is a focused regression suite, not a rerun of the full project
suite.

The finished controller was also staged in the private diagnostic directory
on Afrodite and exercised **without inference or DB writes**. All 508 frozen
source files verified; imports came from that runtime. Two source-ordered
questions validated (53 sessions/550 messages and 45 sessions/485 messages).
No production deployment, service restart, full LME or new LME question run
occurred. The two synthetic Luna calls described above verify transport overlap
and cleanup, not end-to-end LME convergence or accuracy. A fresh bounded
parallel question pilot remains the next live verification step.
