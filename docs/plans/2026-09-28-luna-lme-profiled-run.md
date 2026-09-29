# Four-question warm Luna LME pilot

## Scope and preparation

The user's “Run it” authorizes one fresh, bounded four-question/four-worker
GPT-6 Luna subscription pilot using the accepted throughput implementation.
This is not a full-500 run, a canonical DeepSeek score, or a production change.
Select the first four questions in frozen dataset order, with independent stores.
Run one entry canary, then up to four simultaneous questions. No rerolls,
resumption, model changes, API-key fallback or purchased-credit use.

Preparation succeeded without inference. The launcher checked the staged
dependencies, all 508 frozen candidate files, dataset and collector call sites,
at least 6 GiB available memory and 20 GiB free disk, and no active prior Luna
worker (terminal active/exited units must have MainPID=0 and an empty cgroup).

- Root: `/home/atta/.hymem-luna-lme-profiled-4_yhy8b9`
- Unit: `hymem-luna-lme-profiled-4_yhy8b9.service`
- Expected cgroup: `/user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-profiled-4_yhy8b9.service`
- Launch receipt SHA256: `62a51cb14076fdfe9c6c5ea1d9e7fe58d10b02174954d5681a603efa40d3ed16`
- Launcher: `tools/diagnostics/luna_subscription_profiled_launch.py`
- Launcher SHA256: `f607b8e0b502fc3eaee881b3f4a41adc714b0cfb7faae3ad309fea91411c439b`
- Profiled runner SHA256: `67004a01869d249cb3744c387d536668aea75c791cc986e7cd7b6aabf03001d2`

The source-pinned receipt also binds the warm runner/transport, four-worker
transport, stage collector, base transport, pilot helper, source inventory and
Codex binary. The candidate, dataset, prompts, reasoning and evaluation remain
the same as the completed two-worker pilot.

## Explicit limits

- Campaign: 8,012 admitted turns, 48,160,000 known tokens, 14,400 seconds.
- Each question: 2,000 turns, 12,000,000 known tokens, 12,600 seconds;
  indexing timeout 10,800 seconds.
- Canary: 12 turns, 160,000 known tokens, 600 seconds.
- Four independent workers; at most one invocation per question.
- Recheck subscription auth, model availability, config isolation and reported
  quota before each invocation; stop at the 25% remaining-quota floor.
- A process serves at most 16 fresh ephemeral threads, rotating before the
  next call once its age reaches 300 seconds. This is not an idle expiry.
- Server-owned systemd containment: runtime 14,530 seconds plus 10 seconds
  stop timeout, 4 GiB memory, 200% CPU, 128 tasks, OOMPolicy=kill,
  KillMode=control-group, Restart=no, Type=exec, RemainAfterExit=yes.

The launch marker is consumed before dispatch. An ambiguous dispatch must be
inspected, never repeated. All raw logs, model/benchmark text and stores remain
private on Afrodite. A sanitized environment and private disk TMPDIR are used.
The run and its server-owned cleanup do not require the laptop to remain open.

## Verification and monitoring

Sol implemented a separate launcher and metadata-only reader; root reviews
and tests both before dispatch. Historical pinned launchers/readers are unchanged.
Live progress and terminal output must be read only through the reviewed reader.
Root's prelaunch gate passed 255 existing/launcher tests and 16 observer tests
(including nine independent root controls), 271 total. Root caught and Sol
fixed a reader suffix regex that excluded the launcher's generated underscore.
The final reader successfully observed the prepared, not-yet-launched root:
receipt, source, dataset, binary and 508-file inventory valid; 216.028 GiB free;
no model calls and no owned process. Absent progress/usage remained unknown.

Reader: `tools/diagnostics/luna_subscription_profiled_progress.py`
SHA256: `76ec13055b9d04dbed3250510aecbbc94b04902acd2442bfe14db1e1cc3d56be`

Run from `/Users/attavanwestreenen/AGprojects/HyMem`, verifying that hash first:

```sh
shasum -a 256 tools/diagnostics/luna_subscription_profiled_progress.py
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  /usr/bin/python3 -I -B - \
  --root /home/atta/.hymem-luna-lme-profiled-4_yhy8b9 \
  --unit hymem-luna-lme-profiled-4_yhy8b9.service \
  --expected-cgroup /user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-profiled-4_yhy8b9.service \
  --receipt-sha256 62a51cb14076fdfe9c6c5ea1d9e7fe58d10b02174954d5681a603efa40d3ed16 \
  < tools/diagnostics/luna_subscription_profiled_progress.py
```

Completion requires `completed_and_clean=true`: all four private question rows
pass the frozen indexing/summary protocol and scoring checks, usage and stage
accounting reconcile, cleanup is verified, and the exact resource-policy-bound
unit has exited successfully with no remaining cgroup processes. An incorrect
answer can still be a healthy completed question; report health and accuracy
separately. In-flight or unavailable usage is unknown, not zero. Do not infer
speedup from the small synthetic transport probes or compare these four
questions' aggregate wall time directly with a two-question run as if identical.

Poll at ten-minute intervals. Stay quiet for routine progress. Notify when a
question completes, the run terminates, or action is needed: integrity failure,
wrong containment policy, less than 20 GiB free disk, lost process/OOM, or both
progress and private-log inactivity exceeding twelve minutes while running.
Retry one temporarily unreadable JSON snapshot read-only. Pause the heartbeat
after terminal success/failure. Never restart or extend the experiment while
polling. Old Luna runs and the cancelled DeepSeek run remain stopped, and the
DeepSeek monitor remains paused. Local polling requires the app/computer awake.

## Launch receipt and first live check

Dispatched exactly once on 2026-09-28 at approximately 15:22:25 UTC;
systemd-run returned 0 and the consumed one-shot marker is retained privately.
At 68.4 seconds elapsed the reviewed observer verified active/running,
MainPID=3317956, exact cgroup with five processes (coordinator plus four workers),
all resource policies correct, zero restarts and 215.997 GiB free disk.
Receipt, binary, source, dataset and inventory pins remained valid.

The sole canary passed: eight settled calls, 68,390 known tokens, one process,
one cold and seven warm requests, successful canary cleanup. All four question
stores/workers started; none completed or failed yet. The snapshot had four
active invocations, 15 admitted turns and 103,384 settled known tokens.
In-flight usage is incomplete; stage totals may differ from admitted turns
until the active calls settle. No answer correctness or full-question speed
claim is available at launch.

The existing `monitor-luna-lme-pilot` heartbeat was updated to ACTIVE, every ten
minutes, pinned to this root/unit/receipt/reader. It is read-only and must pause
at terminal completion/failure. No new benchmark is authorized by monitoring.
