# Four-question Luna capacity-corrected pilot

## Verified correction and scope

The user's request authorizes sequential diagnosis, Sol implementation, root
verification and bounded reruns. The four-worker launcher inherited a 128-task
ceiling that was too tight. A real-LME diagnostic captured a thread/start RPC
error with 128 current/peak/max tasks and three kernel task-limit denials BEFORE
cleanup. The no-inference probe peaked at 120 with nine controller threads;
actual LME had 13. This is a launcher resource defect, not evidence of model unreliability.

Sol implemented a separately versioned launcher/reader. Root reviewed exact
diffs and 264 tests passed. The sole execution-policy change is TasksMax 128→256;
all source code/prompts/model/isolation/quota/quality/spend controls remain the
same. A same-wrapper 176-turn counterfactual under 256 completed all its admitted
calls, 1,273,940 known tokens, zero RPC failures/zero task denials, reconciled
usage and full process/store cleanup. It intentionally stopped at its budget,
not question completion. Full operational validation remains pending here.

This new attempt runs the same first four frozen LongMemEval-S questions in
four independent workers with GPT-6 Luna subscription auth. It is not a
canonical API score or full-500 result. Production is untouched. Historical
attempts/source pins remain intact. Never resume or reroll a previous run.

## Bound identity

- Root: `/home/atta/.hymem-luna-lme-capacity-jzipeyhr`
- Unit: `hymem-luna-lme-capacity-jzipeyhr.service`
- Cgroup: `/user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-capacity-jzipeyhr.service`
- Receipt SHA256: `91fec9aa64e15a954a2d2eec0609be2cc67a07f32c7c17cd467eb6b73515d158`
- Launcher: `tools/diagnostics/luna_subscription_capacity_launch.py`
- Launcher SHA256: `b848ad37f66680c8e423876b0aed2f4f99bdd6ee896247f1faa9461a50e43983`
- Reader: `tools/diagnostics/luna_subscription_capacity_progress.py`
- Reader SHA256: `2eed17065cf144a4f59d0af47ac2b6d937253f0b1c100b1ee134430a89789831`
- Profiled runner unchanged: `53628ac7e9c6107bb68d1cd4ebdf42d5a129b96c4d96360d4e7a6525be7bc739`
- Warm runner unchanged: `3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567`
- Warm transport unchanged: `9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593`

Receipt also binds base/concurrent transport, stage collector, pilot helper,
508-file frozen candidate inventory, dataset and Codex binary. Preparation
passed without inference; no launch at the time this document was created.

## Budgets and containment

Campaign: 8,012 turns / 48,160,000 known tokens / 14,400 seconds. Each question:
2,000 turns / 12,000,000 known tokens / 12,600 seconds; indexing 10,800 seconds.
Canary: 12 turns / 160,000 known tokens / 600 seconds.
Four workers; one invocation per question; processes rotate after 16 requests
or 300 seconds before a new call. Fresh ephemeral thread for every request.
Exact subscription auth/model/config admission every call, 25% quota floor,
no API keys/provider fallback/purchased credits. No score-based rerolls.

Systemd runtime 14,530s + stop 10s, TasksMax 256, memory 4 GiB, CPU 200%,
OOMPolicy=kill, KillMode=control-group, Restart=no, Type=exec,
RemainAfterExit=yes. Host floors: 6 GiB available memory / 20 GiB disk;
no active prior worker. Exclusive marker consumed
BEFORE launch. Ambiguous dispatch is inspected, never repeated. Logs/stores/
benchmark or model text/credentials stay private on Afrodite.

## Exact metadata-only monitoring

Verify reader hash first, then from the local HyMem workspace:

```sh
shasum -a 256 tools/diagnostics/luna_subscription_capacity_progress.py
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  /usr/bin/python3 -I -B - \
  --root /home/atta/.hymem-luna-lme-capacity-jzipeyhr \
  --unit hymem-luna-lme-capacity-jzipeyhr.service \
  --expected-cgroup /user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-capacity-jzipeyhr.service \
  --receipt-sha256 91fec9aa64e15a954a2d2eec0609be2cc67a07f32c7c17cd467eb6b73515d158 \
  < tools/diagnostics/luna_subscription_capacity_progress.py
```

Require `completed_and_clean=true`: all four private question rows pass frozen
index/summary and scoring-validity checks, final usage/stages reconcile, all
clients/owned runs/leases clean up and exact policy-bound unit has no remaining
processes. Correctness is separate from operational health; an incorrect answer
is valid measurement, not a reason to reroll. Missing health while indexing is
unknown. In-flight usage is incomplete; overlapping stage timings are not wall
time. Live pids counts/peaks/max-denial counters are reported; once cgroup is
removed terminal peak is unavailable, not zero. Retain observed numeric peaks
in later receipts rather than claiming unobserved lifetime maxima.

While this run is active, monitor read-only every 10 minutes: do not alter code,
caps/model, production or experiment; no provider calls/new runs/resumes/restarts.
Stay quiet on routine progress. Notify on completed questions, terminal outcome
or actionable integrity/policy/disk<20GiB/process/OOM/task-limit-denial issues,
or BOTH progress and log inactivity>12minutes. Retry unreadable JSON once.

On clean completion report validated health, correctness, known usage, timing,
cleanup and limits of this four-question result, then pause the heartbeat. On
failure preserve exact cause and cleanup. The user has authorized continuing
the evidence-driven Sol/root repair cycle: establish a reproducible defect,
separate Sol implementation, independent root verification and targeted tests,
then only a justified fresh same-four-question run with new receipt/monitor.
Never blind-reroll or bypass quota, increase spending caps or switch models.
Pause and request direction for external quota exhaustion or materially wider
scope. All earlier Luna runs and the cancelled DeepSeek run stay stopped;
DeepSeek monitor stays paused. Local polling/repairs require this app/computer;
the bounded server run/cleanup do not depend on the laptop remaining open.

## Terminal result and offline diagnosis

Dispatched once at approximately 16:45 UTC on September 28. Dispatch returned
success with `never_retry=true`. The run then failed its canary, before any
question started: 8 admitted/returned calls, 68,440 known tokens, complete and
reconciled usage. Wall time was 37.945 seconds. There was no transport failure;
the unit exited with status 1, no restarts, MainPID 0 and no cgroup processes.
Resource policy and all live source/data/inventory pins were verified. The
cgroup's removed terminal task counters are unknown, not zero. Disk was
215.912 GiB free. The heartbeat remains paused; this attempt is not resumed.

Root replayed all eight retained responses against their exact original
requests with networking disabled, all 508 frozen files verified and zero new
model calls. Evidence SHA256:
`fc07439a9b5c98a6487e65f1fef64d45775a4e0f1192005462c666fedc5a19d7`.
Private evidence remains at `run/private-canary-evidence.json` on Afrodite.

The replay reproduces the rejection: two expected exact claims **plus a third
claim**. Both expected core facts, source IDs, polarity and request paths were
correct. All four optional type fields were absent (allowed); no wrong or
invalid supplied types, property hints, markers or duplicate collapses occurred.
The third item came from primary call 7 and used the expected prose subject,
object, positive polarity and source ID, but predicate `uses` rather than the
expected `prefers`. The fixture establishes preference, not actual use. This
is unsupported predicate expansion, not a canary-oracle mismatch or recurrence
of the proven task-limit defect. A separate Sol review confirmed that the
strict gate intentionally rejects extra claims; no gate change is justified.

The throughput-only correction remains independently verified by the bounded
176-turn resource test, but the complete four-question run has NOT passed.
Root requested approval to extend scope to a separately versioned extraction
grounding candidate with independent positive/negative controls before another
LME attempt. No prompt, quality gate, candidate, model or budget has changed,
and no additional run has been launched pending that scope decision.
