# Luna throughput implementation and verification

## Ordered request and initial Git checkpoint

The user requested: first push the current version, then implement the speed
plan with separate Sol agents and root verification of every output before the
next change. Local commit `e2bc707` preserves the audited current runtime,
benchmark/test source and Luna tooling (264 files). Private archives, receipts,
old deployment artifacts and machine-local settings were excluded. The push to
`https://github.com/ClaudetteMedSer/HyMem.git`, branch `Beam-optimisation`, was
blocked by the approval system because this is a broad public publication.
The user explicitly approved that payload/destination. The exact commit was
pushed successfully and the remote branch verified as
`e2bc7073b4902e3e0bd1edeee2ae04fc9815815b` before implementation began.

## Measured starting point

The completed two-worker pilot passed strict terminal validation: two completed
and correctly scored questions, healthy indexing/summaries, no pending work,
complete accounting and no remaining owned processes. Total wall time was
approximately 102.5 minutes. Known usage: 1,704 model turns, 12,350,176 tokens.
Question indexing took 6,086.6 and 5,384.8 seconds, across seven and five cycles.
The questions consumed 895 and 801 turns, respectively; the canary used eight.

Accumulated invocation timings (overlapping workers; not wall-clock phases):
2,214.4 seconds setup/admission, 8,801.8 seconds model-turn waiting/execution,
106.2 seconds cleanup. Peak unit memory was 318,291,968 bytes; total CPU time
1,406.2 seconds. This supports investigating throughput and repeated setup, not
claiming CPU tuning or process reuse alone will yield an order-of-magnitude gain.

## Step 1 — four independent question workers (Sol A)

Preserve all accepted two-worker source hashes and historical artifacts. Add a
new versioned transport/runner supporting one through four independent question
workers. Every question retains one store and at most one in-flight invocation.
The shared campaign ledger must atomically enforce aggregate limits; per-question
budgets are not multiplied implicitly. Preserve the 25% reported quota floor,
exact model/subscription route, source inventory, single entry canary, strict
indexing/summary/score checks, source-ordered results and honest cleanup.

Required verification by root: four-way overlap, fifth admission rejection,
budget races, quota/admission stop before cleanup, unknown usage, simultaneous
failure, interrupted workers, preservation of completed peers, distinct stores,
deterministic results, unchanged producer/evaluation semantics, no API fallback.
Keep the larger-run envelope explicit. A bounded four-way synthetic live check
can validate transport overlap and cleanup, but is not an LME throughput result.

Root accepted the implementation offline: 33 Sol tests plus eight independent
root controls pass, including real frozen-store isolation/producer identity,
failure propagation through the frozen extractor, interrupted dream cleanup,
and simultaneous settlement with one unknown-usage turn. Root found a flaky
interrupt test; Sol repaired its ordering and repeated both interrupt tests
20 times. The original four source pins remain unchanged. Live four-way
transport measurement remains pending and will not be claimed from these tests.

Subsequently root's live invented-text probe passed all four exact-response
checks with overlapping turn intervals, four independent processes, complete
known usage (16,968 reported tokens), and no remaining owned process groups.
Wall time was 9.23 seconds; unit peak memory 221.4 MiB. This is a transport
probe, not a four-question LME result. Fresh diagnostic root:
`/home/atta/.hymem-luna-four-check-cqivz1hU`; unit
`hymem-luna-four-check-cqivz1hU.service`. No production changes.

## Step 2 — warm process, fresh isolated requests (new Sol B)

Start only after root accepts step 1. Use a separately versioned, opt-in
transport with bounded process lifetime per worker/question. Reuse the serving
process/connection only; never reuse a conversation or previous model text.
Each completion starts a new ephemeral thread with identical instructions and
isolation checks. Keep dynamic auth/config/quota admission per completion;
validate any caching of genuinely immutable process metadata separately.
Model, reasoning, prompts, extraction, verification and scoring remain fixed.

Before accepting reuse, verify notification ownership, stale events, unique
thread IDs, thread unload, per-call deadline reset, quota changes, process death,
cleanup and interrupted calls. Fail closed on uncertainty; no hidden reconnect,
retry, resumption, thread-history reuse or provider fallback. Keep the process
count and retained thread count bounded and close workers before terminal
validation. Record warm/cold transport identity and timing distinctly.

Official App Server documentation says unsubscribing the last subscriber keeps
the thread loaded for a 30-minute inactivity grace period. Therefore an ACK is
not proof of immediate unload. Use a strict, bounded retired-thread set and
finite process rotation/reaping; do not wait 30 minutes per request or claim
threads were unloaded from an unsubscribe ACK. Late content/tool/usage events
must not enter a subsequent request.

Root must verify independent offline failure cases, then a small live
cross-request isolation/cleanup probe on invented text. An apparent speedup
does not excuse state leakage or reduced admission/health checks.

Root accepted the separate warm transport and runner after 23 focused checks
(including nine independent root tests) and a live four-call invented-text
probe. Corrections required by review covered late thread-start events, pending
retirement events, deadline checks before sending, retained handles after a
failed cleanup, consistent budget exception identity, flat source-pinned
loading, and closing each question's client before replacing that worker.

The live probe used four unique ephemeral threads and two processes, with
process counts `[1, 1, 1, 2]`: three requests shared the first process and the
fourth rotated. Prior-turn marker recall and local file access were unavailable;
each auth/model/config/quota check ran four times. Usage was fully reported:
17,069 tokens. Calls took 5.46, 3.52, 3.63 and 5.63 seconds (different synthetic
requests; not a controlled LME speed estimate). All owned process groups were
gone, and systemd independently reported inactive/dead, success, MainPID=0,
empty ControlGroup, zero restarts. Private diagnostic directory:
`/home/atta/.hymem-luna-warm-check-JPzLzjzT`.

The 300-second setting is a **between-call rotation threshold**, not a hard
idle-process expiry. Each invocation has an absolute deadline (at most 120
seconds), each process retains at most 16 request threads, and each question
closes its process before terminal status. A server-owned campaign containment
unit is still required for a headless benchmark. No immediate-unload or full
LME completion claim follows from the probe.

## Step 3 — measure call volume before architectural changes

Add source-free stage/call/latency accounting with a separate Sol agent only
after prior acceptance, if existing counters cannot attribute extraction,
verification, retries and digest work. Do not remove checks, change prompts,
change chunk boundaries, parallelize SQLite mutations or reduce indexing work
merely to improve elapsed time. Such changes require their own correctness and
quality experiment. The current request does not authorize a full-500 run.

Step 3 is implemented by a third Sol agent in a separate collector/profiled
runner. It identifies only pinned code paths/function names/call-site lines;
it never reads frame locals or request/response text. Labels include explicit
`unclassified`. Attempted invocations, admitted turns, known tokens, unknown
usage and elapsed invocation time are tracked separately per question/stage.
Provider-internal retry counts remain unknown. Root checked actual frozen
empty-verifier behavior, output equivalence, fail-closed accounting (no model
retry), and a five-question/four-worker integrated campaign using fake answers.

Final live integration: the existing Luna experimental extraction canary v2
passed on its sole attempt, eight calls, 68,388 reported tokens, 34.57 seconds.
Both expected core claims matched and the execution path was exact. Source-map
verification covered 508 files. One process served eight fresh threads
(one cold/seven warm invocations); cleanup passed. The new stage totals
reconciled exactly: four primary extraction calls, two empty verifiers and two
omission verifiers, with no repairs, terminal retries or unclassified calls.
Private directory: `/home/atta/.hymem-luna-profiled-canary-nFMpsFWb`.

These are transport/canary validations, not an LME answer-score or full-question
throughput comparison. No production deployment, production-memory mutation,
restart, previous-run resumption or full LME launch occurred. Across the three
live probes: 16 subscription turns and 102,425 known tokens; internal HTTP
attempt counts are unavailable. The old monitors remain paused.

Final root regression gate: **248 passed in 7.48 seconds**, no failures or
skips. All seven accepted historical transport/runner/launcher/reader hashes
remain byte-identical. All three new probe units independently report
inactive/dead, successful exit, MainPID=0, empty ControlGroup and zero restarts.

Verified implementation pins:

- Four-worker transport: `cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0`
- Warm transport: `d9c3e50b36af8618ee5a736105c37e1312cd44abf28f24f13979dda2d4f4da1a`
- Warm runner: `cb6eecfbb1b23982d81dafbf908aa135445566d2419d02db772374b2efd37873`
- Stage collector: `800ef9baedc9d68093b3160cc324ef17528dd0b89f7a734f220217b07323fba2`
- Profiled runner: `67004a01869d249cb3744c387d536668aea75c791cc986e7cd7b6aabf03001d2`

## Reporting and live boundaries

Report implementation/tests separately from measured live speed improvements.
Use source-pinned, fresh diagnostic directories for bounded subscription tests;
keep benchmark/model text, stores and credentials private on Afrodite. Do not
alter production, restart services, revive old experiments or modify completed
pilot artifacts. Any full-question cold/warm comparison must use independently
recorded, finite campaign limits and preserve a clean baseline.
