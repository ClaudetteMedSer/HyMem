# Attributed four-question Luna rerun

## Scope

User authorizes diagnosis, sequential Sol fixes, independent root verification
and bounded reruns until operationally fixed. This attempt repeats the same
first four frozen LME questions with four independent workers and GPT-6 Luna
subscription auth. Only failure attribution changed; candidate, dataset,
prompts, isolation, model, quality gates and limits are unchanged. It is neither
a canonical API score nor a full-500 run. No production deployment.

The preceding four-question pilot failed with an uninformative transport code.
The accepted attribution repair and 160-call synthetic stress receipt are in
`2026-09-28-luna-warm-recovery.md`. Stress passed but did not reproduce the LME
fault. Do not claim that fault fixed until this experiment validates completely.

## Immutable identity

- Root: `/home/atta/.hymem-luna-lme-attributed-q8mvj98i`
- Unit: `hymem-luna-lme-attributed-q8mvj98i.service`
- Cgroup: `/user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-attributed-q8mvj98i.service`
- Receipt SHA256: `52e876c682e424ea66e3361eef90b4afff8b0b7284c38c56fb7aaaf5567eba46`
- Warm transport v2 SHA256: `9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593`
- Warm runner v2 SHA256: `3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567`
- Profiled wrapper v2 SHA256: `53628ac7e9c6107bb68d1cd4ebdf42d5a129b96c4d96360d4e7a6525be7bc739`
- Launcher SHA256: `3759f03b8611df4083d1edf9cef81024b76705485dff3091359cc5db7115e903`
- Reader SHA256: `cd788915146fa40c222c9beb49387787c1797185c355e95cc33e7e165006f6f6`

All historical source-pinned files remain unchanged. Separate Sol agents
implemented the attribution, stress harness, integration and reader. Root
reviewed all changes and verified 243 regression tests, including real frozen
candidate loading and eleven independent metadata-boundary controls.

## Limits and launch

Campaign: 8,012 turns / 48,160,000 known tokens / 14,400 seconds. Each question:
2,000 turns / 12,000,000 known tokens / 12,600 seconds; indexing 10,800 seconds.
Canary: 12 turns / 160,000 known tokens / 600 seconds. Process rotation at 16
fresh ephemeral requests or 300 seconds before the next call. Subscription
auth, account/model/config isolation and 25% reported quota floor checked on
each invocation. No API-key fallback or purchased credits.

Systemd: 14,530 seconds runtime + 10 seconds stop, 4 GiB memory, CPU 200%,
TasksMax 128, OOMPolicy kill, KillMode control-group, Restart no, Type exec,
RemainAfterExit yes. Host admission requires 6 GiB available memory and 20 GiB
free disk, no active previous worker. One-shot marker consumed before dispatch;
ambiguous dispatch must be inspected, never repeated. Old Luna and DeepSeek
runs stay stopped. Raw text, logs, credentials and stores stay private remotely.

Preparation succeeded without inference. Launch is pending at this document's
creation; append dispatch and live verification receipts below.

## Metadata-only monitoring

Verify the reader hash first, then run from the local HyMem workspace:

```sh
shasum -a 256 tools/diagnostics/luna_subscription_profiled_v2_progress.py
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  /usr/bin/python3 -I -B - \
  --root /home/atta/.hymem-luna-lme-attributed-q8mvj98i \
  --unit hymem-luna-lme-attributed-q8mvj98i.service \
  --expected-cgroup /user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-attributed-q8mvj98i.service \
  --receipt-sha256 52e876c682e424ea66e3361eef90b4afff8b0b7284c38c56fb7aaaf5567eba46 \
  < tools/diagnostics/luna_subscription_profiled_v2_progress.py
```

Require `completed_and_clean=true`, not exit status alone: all four private rows
must satisfy the frozen indexing/summary protocol and scoring validity, final
usage/stage accounting must reconcile, all clients/stores clean up, exact unit
policy valid and no owned processes. Health and correctness are separate;
incorrect answers are measurements, never a reroll reason. In-flight usage and
missing final health are unknown. Overlapping timings are not campaign elapsed.

Monitor every ten minutes; stay quiet on routine progress. Notify on completed
question, terminal outcome or action needed: metadata/source integrity failure,
wrong unit/resource policy, disk below 20 GiB, lost process/OOM, or both progress
and private-log inactivity over twelve minutes. Retry temporarily unreadable
JSON once read-only. Do not start/resume/reroll or mutate the experiment while
it is active. If it fails, preserve and report first-failure code/phase and
cleanup. The user's current request authorizes continuing the sequential repair
workflow: establish a reproducible defect, separate Sol implementation, root
review/reproduction/tests, then only a justified fresh same-four-question run
with new pins/receipt and monitor identity. Never blind-reroll or bypass quota.
Pause for external quota exhaustion, ambiguous evidence requiring user choice,
or materially wider scope; pause after clean four-question completion.

The detached server-owned run/cleanup do not require the laptop. Local heartbeat
polls require this app/computer available. No extra questions authorized here.

## Dispatch and first live verification

Launched once at approximately 16:07:19 UTC on 2026-09-28; systemd dispatch
returned zero. At 29 seconds the independent reader verified all source,
inventory, dataset and binary pins, exact resource policy, active/running unit,
MainPID 3359753, two owned processes (controller plus canary worker), no restarts
and 215.993 GiB free disk. Canary was in flight: five settled turns / 42,344
known tokens; usage incomplete, no failure. No completion claim yet.

Existing heartbeat updated to this exact receipt/reader at ten-minute cadence.
While active it is read-only; after terminal failure it may continue the user's
authorized evidence-driven Sol/root repair cycle, never a blind reroll.

## Terminal failure and next diagnosis

The attributed attempt failed after 266.827 seconds. Canary passed, no questions
completed. Exact first cause: `rpc_failure:thread/start`, phase `preflight`,
worker 1, third process/fifth request, four retired threads, process age 16.188s;
the failing invocation had not admitted a model turn. Error JSON itself was not
retained by the pinned RPC layer, so the underlying rejection is still unknown.
Do not assume quota, lifecycle, malformed request or provider instability.

Final usage: 136 admitted turns, 983,262 known tokens, complete and stage-reconciled;
140 attempted invocations, four rejections including peer stops. All four clients
cleaned up. Independent systemd: failed/exit-code, status 1, MainPID 0, empty exact
cgroup, zero restarts, policy valid; 215.966 GiB free. Source/data/inventory valid.
No accuracy/index-health result exists. Historical files and private evidence
remain untouched. This attempt must not be resumed.

Root dispatched separate Sol agents for static `thread/start` diagnosis and a
bounded no-inference request-variant probe. The probe must retain bounded error
JSON privately and expose only fixed classifications/numeric metadata. No further
LME rerun until the rejection is explained and any actual fix is verified.
