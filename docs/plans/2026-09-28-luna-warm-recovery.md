# Luna warm-pilot diagnosis and sequential repair

## Authorized outcome

Diagnose the failed four-question/four-worker GPT-6 Luna pilot, use separate Sol
agents for fixes, independently verify every output, then rerun and iterate.
Keep the same model, frozen candidate/dataset, prompts, quality gates, four-way
question concurrency, quota floor and finite budgets. No production changes or
full-500 launch. Preserve failed-run evidence and all historical pinned sources.

## Confirmed evidence

The sole profiled run in `.hymem-luna-lme-profiled-4_yhy8b9` stopped after
384.332 seconds. Canary passed; zero questions finished. There were 136 admitted
turns and 978,389 reported tokens, reconciled with stage accounting. All question
clients cleaned up, the exact cgroup is empty, and the monitor is paused.
No code/dataset integrity, host resource policy or disk-floor failure was found.

Read-only private metadata confirms `transport_or_admission_failure` as the
first shared stop. Question 0 had 32 transport invocations but 31 admitted turns;
question 2 had 33 vs 32. Questions 1/3 received `campaign_stopped`. Four owned
interrupted dream telemetry rows were terminalized, with no remaining leases or
open runs. No question was scored. Total overlapping timing was 227.55 seconds
preflight, 731.47 model waiting/execution and 2.22 error cleanup.

The transport discards its original fixed safe exception code, making distinct
quota, protocol, lifecycle and timeout failures indistinguishable. This is a
confirmed observability defect; the triggering fault is not yet established.
The strict warm notification router and asynchronous account/quota notices are
candidate causes, not findings. Do not weaken guards based on these hypotheses.

## Ordered plan and acceptance

1. Root collects only bounded failure codes, numeric timings and lifecycle
   metadata. A no-inference admission/rotation probe may exercise fresh threads
   without sending benchmark text or calling a model.
2. Sol A implements a separately versioned warm transport with safe, fixed-code
   first-failure attribution (phase, RPC, numeric lifecycle/accounting metadata).
   No behavioral relaxation or retry. Root verifies privacy, precise attribution,
   first-failure retention, usage, deadline and cleanup tests before proceeding.
3. Use the attributed transport in a finite synthetic probe (or a fresh bounded
   benchmark only if needed) to reproduce the actual fault. Keep raw evidence
   private on Afrodite. Never guess a provider cause from a generic code.
4. A different Sol agent fixes each demonstrated defect with regression tests.
   Root independently reproduces the old failure, verifies the correction and
   negative controls, then runs broad transport/runner/monitor regression tests.
   Repeat this step if a genuinely distinct defect appears; do not stack
   unverified fixes.
5. Verify source-pinned integration, fresh-conversation isolation and cleanup;
   create a new one-shot receipt/root for the same four-question experiment.
   Launch once, inspect exact systemd policy and startup, and monitor without
   touching production. Never resume or overwrite the failed pilot.
6. Require four completed, independently healthy/scored question rows, complete
   known usage and reconciled stage accounting, and zero owned processes before
   calling the pilot fixed. Incorrect answers are valid measurements, not a
   reason to reroll. Record all attempts and any limitations. No automatic
   reroll for answer quality or externally exhausted subscription quota.

Official lifecycle source used: https://learn.chatgpt.com/docs/app-server .
Unsubscribe acknowledges detachment, not immediate unload; each request keeps a
fresh ephemeral conversation and process rotation remains bounded.

## First verification gate

Root ran 48 no-inference admissions/unsubscriptions across three processes,
16 fresh threads each: all passed in 38.567 seconds, clean shutdown, reported
quota remaining 75%. No benchmark text or model turns were sent. This does not
rule out faults after actual completed turns.

Sol A implemented `benchmarks/codex_subscription_warm_v2.py`, SHA256
`9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593`.
Root found and required fixes for the actual runner budget factory, peer-stop
misattribution and final-close fault attribution. The final version preserves
safe original codes and first-fault phase metadata without changing admission,
isolation or event acceptance. Root independently compared the entire WarmSession
AST: identical to the pinned predecessor. Historical source hashes are unchanged.
The 138-test targeted regression gate passed, including real frozen-runtime,
first-failure privacy, cleanup, 16/17-request rotation and age-boundary checks.

Next diagnostic is a fresh four-worker invented-text stress test, limited to
40 requests per worker (160 total), 1,200,000 known tokens and 900 seconds;
each worker is limited to 300,000 known tokens and 880 seconds. Full subscription
admission, 25% quota floor and strict lifecycle gates remain active. A separate
server-owned systemd unit will contain the probe. It is not an LME rerun or a
claim that the original failure has been fixed.

Stress probe dispatched once in `/home/atta/.hymem-luna-warm-stress-TmJ18EXC`,
unit `hymem-luna-warm-stress-TmJ18EXC.service`, receipt SHA256
`361eb8ef10bde7b6568ab11e92ab7696b851bbd03efec3e35598e59642dd9a0c`.
Probe source SHA256 `8aa58cb5e979e7d1dbd284fa88b14d9bbe6ac9e3efc87212880a363b44fff679`.
Separate Sol implementation received root privacy, one-shot and cleanup review;
five probe-specific tests passed. Remote pre-dispatch reverified source files,
binary and the full frozen inventory. Unit containment: RuntimeMax=930 seconds,
TimeoutStop=10 seconds, 4 GiB, CPU=200%, TasksMax=128, OOMPolicy=kill,
KillMode=control-group, Restart=no. Initial live metadata verified all policies
and five processes (coordinator plus four workers).

Final stress result: **PASS**, 160/160 successful invented-text calls in
180.289 seconds, 680,720 known tokens, complete accounting. Four-way overlap,
160 unique threads, three processes/two rotations per worker, each dynamic
admission check executed 40 times per worker. Isolation-marker checks passed.
All workers reported cleanup; independent systemd proof showed active/exited,
success, MainPID=0, empty exact cgroup, zero restarts. Accumulated overlapping
preflight/model time: 124.73 / 555.86 seconds. No benchmark answer was evaluated.
This rules out a simple deterministic rotation or short-request concurrency
failure; it does not identify the historical first error or prove LME readiness.

Root accepted the observability fix and authorized integration into a new
source-pinned four-question diagnostic rerun. Separate Sol agents own the
versioned runner/launcher and metadata reader; root must verify both and their
interface before dispatch. This rerun retains all LME semantics and guards.
Any failure must expose the original fixed code/phase, not just “other.”

Root independently reviewed the versioned runner, wrapper, launcher and reader.
One additional exception-path `setdefault` bug was found and fixed by Sol, with
an actual main-path regression test. Final combined gate: **243 passed** under
`/opt/anaconda3/bin/python3.13`; a first run under `/usr/local/bin/python3` had
242 passes and a local isolated-subprocess missing-`requests` dependency failure.
No benchmark dependency or gate was removed to get a pass. Eleven independent
root tests cover all fixed attribution atoms, malformed private fields,
in-flight usage and independent cgroup cleanup.

The new experiment and exact read-only command are recorded in
`2026-09-28-luna-lme-attributed-run.md`. Preparation verified all source pins,
the frozen 508-file candidate, dataset and host admission without inference.

The attributed rerun failed at `rpc_failure:thread/start`, before admitting the
failing turn. Its exact terminal receipt is in the attributed-run plan. Failure
reporting now works, but the RPC layer discarded its error response, so a
thread-start rejection versus malformed result and the underlying cause still
need measurement. Next: separately implemented no-inference thread-start probe
with private bounded RPC error retention and source-free cgroup task counters.

The suspected instruction-shape cause is weakened: failing stage is the static
empty-extraction verifier, whose identical system instructions already passed
earlier calls; user content is not sent in `thread/start`. Resource ceiling is
only a hypothesis, not a finding. A root model-free remote import of the full
508-file-verified frozen runner showed one OS thread before and after import,
RSS 10,836 to 57,012 KiB, no NumPy/Torch imports, CPU count eight. Thus import-time
native numerical thread pools do not explain the failure on this runtime.

Probe preparation notes (zero inference throughout): first staged root
`.hymem-luna-thread-start-y5yNepGW` was rejected by the exact source hash before
dispatch because an agent edit overlapped copying. Source was frozen; that root
is untouched and never launched. Second root `.hymem-luna-thread-start-kjfaEHWF`,
receipt `bb129c4d160899a8428445ab9d65cf1232078d3ca91cc0f22d058f7852102b8f`,
launched the accepted `bc0648bc36a6b67ea5e15b256bc5032d1edd765001243716492dd4eb832d5dc9`
probe but exited in setup, no run marker/thread starts. Root identified a probe
portability bug: actual hostname is case-equivalent to Afrodite, not literal
lowercase. UID 1000, directory 0700 and frozen loader all pass. Sol is adding a
case-insensitive exact-host check and regression test. No benchmark rerun or
production change was caused by either preparation failure.

Corrected no-inference probe SHA256
`c55ac5dbc75938ad5060a9e90e93c361d7374b911d33a4e939c79ef53749b223`
passed seven root-run tests and the real frozen prompt-loader check. Executed
once in `/home/atta/.hymem-luna-thread-start-t7G3gXJt`, unit of the same suffix,
receipt `10367df2dfa526f9ebf7721f314516b1631f785a172ef35987d23c096b5d6873`.
**128/128 thread admissions passed, zero model turns**, 26.025 seconds,
four workers, 2–3 processes per worker, all cleanup true. `pids.peak=120`,
`pids.max=128`, max-limit events zero; controller threads peak nine. Final unit
active/exited/success, MainPID zero, cgroup empty, policy correct, no restarts.
This establishes thin task headroom, not the original failure's cause.

Next authorized diagnostic is a new Sol-implemented instrumentation wrapper
over the frozen attributed runner. It retains all semantics but captures bounded
RPC error JSON privately and fixed resource/error metadata before cleanup.
It runs the same four question prefixes, at most 176 admitted turns, 1.5 million
known tokens and 900 seconds, under the unchanged TasksMax128 resource ceiling.
Per question 80 turns/600k known tokens/800s; canary12/160k/600s; indexing750s.
Budget exhaustion is an expected diagnostic stop, never healthy completion.
No further full four-question attempt until the root cause is established and
any correction receives separate Sol implementation plus root verification.

Resource diagnostic dispatched once at approximately 16:32 UTC:
output `/home/atta/.hymem-luna-lme-resource-yCvtluzS`,
unit `hymem-luna-lme-resource-yCvtluzS.service`,
stage `/home/atta/.hymem-luna-resource-bundle-FlMOnu5r`,
receipt `c4e2066e875db4cf9f07e560a70e19207025c6cc312c1ec003cc6bb7ae9ab0c1`.
Sol wrapper `tools/diagnostics/luna_lme_resource_probe.py` SHA256
`a5256703130385375114e27e26fc6f6e06a00fb215e2623adcb04d72a4548382`.
Five root-run tests passed, including actual pinned constructor/default-class
instrumentation and exact RPC response object identity. Source pins and all 508
files reverified before dispatch. Root remains empty until its one-shot marker;
launcher logs/temporary files stay in the separate stage directory.
Systemd runtime930s+stop10s, Tasks128, memory4GiB, CPU200%, OOMkill,
KillModecontrol-group, Restartno. First live check: canary passed8turns/68,398
tokens, task peak34, no max-limit events, all policy values correct.

**Reproduced resource defect:** at 68.059 seconds the probe got a genuine
JSON-RPC error (`-32603`) from `thread/start`. BEFORE cleanup, the resource
observer recorded exactly `pids.current=128`, `pids.peak=128`, `pids.max=128`,
and three kernel max-limit denials. Controller had 13 threads, versus nine in
the no-inference probe. Final counter was four denials. Thus the four-worker
launcher task ceiling is insufficient; an idle/lightweight probe is not a
representative resource gate. The resource sample is direct evidence, not an
inference from the coincident 136-turn counts in earlier runs.

Diagnostic stopped with canary pass, 13 admitted turns/103,785 known tokens,
usage complete, no completed questions. All four clients cleaned up, zero
active/in-flight calls; unit failed with status1, MainPID0, cgroup empty,
zero restarts, correct resource policy. No further work remains in that unit.
Private RPC message remains private (45 bytes; code -32603); fixed-keyword
classification only established thread/task terminology, so no raw message or
invented provider explanation is quoted.

Sol capacity repair: new launcher and reader, keep accepted LME/transport source
bytes and all prompts/models/quota/budgets unchanged, raise bounded TasksMax from
128 to 256 for four workers only. Memory4GiB/CPU200%/16-request process rotation
and all other bounds stay fixed. Add live numeric cgroup counters; do not invent
terminal peak values when the kernel cgroup is gone. Root must verify exact
policy-negative controls, then repeat the same 176-turn resource diagnostic
under TasksMax256. Require no RPC failures/limit denials, complete accounting
and cleanup at its expected budget stop before a full four-question pilot.

Root accepted the Sol capacity-only implementation after exact-diff review:
`luna_subscription_capacity_launch.py` SHA256
`b848ad37f66680c8e423876b0aed2f4f99bdd6ee896247f1faa9461a50e43983`;
`luna_subscription_capacity_progress.py` SHA256
`2eed17065cf144a4f59d0af47ac2b6d937253f0b1c100b1ee134430a89789831`.
Combined root gate263 passed; an additional one-shot main-path test was then
added, and all nine capacity tests passed. Exact policy tests reject128,257,
unlimited tasks and changes to other policies; reader exposes live denial counts.

Counterfactual verification launched once under **TasksMax256** at approximately
16:38 UTC, SAME wrapper SHA a525... and SAME176-turn/1.5M/900s diagnostic bounds:
stage `/home/atta/.hymem-luna-resource-bundle-xpwFTzuG`, output
`/home/atta/.hymem-luna-lme-resource-Ikr6tp1L`, unit
`hymem-luna-lme-resource-Ikr6tp1L.service`, receipt
`2e4700a004d7250bf04a155d0451583d0e3995fa59866f279ca3200ac479867d`.
Initial check verified the only resource change (128→256), no task denials,
correct remaining containment. Validation is pending; no full pilot dispatched.

**Capacity counterfactual verified:** reached the planned176-turn budget stop
with 1,273,940 known tokens, complete/reconciled stage accounting, zero RPC
errors and zero task-limit denials. Kernel peak128 with ceiling256. All four
clients cleaned up; four owned interrupted telemetry rows terminalized,
open runs0/leases0 for each. Unit exit1 is expected diagnostic budget exhaustion,
not benchmark success: MainPID0/empty cgroup/no restarts/resource policy correct.
No questions were completed or scored in this intentionally short diagnostic.

Final root regression gate:264 passed. The demonstrated task-ceiling defect is
corrected and survived the targeted workload; full four-question operational
verification is still required. Next fresh capacity pilot uses unchanged pinned
profiledv2 runner and original full4Qbudgets, only TasksMax256 plus new metadata
reader. See `2026-09-28-luna-lme-capacity-run.md` for exact identity and monitoring.

The capacity-corrected full pilot was dispatched once but failed its eight-call
canary before starting any question (68,440 known tokens, complete/reconciled
usage, unit exited cleanly). Root's exact-request offline replay found a genuine
third claim: the expected preference relation plus an unsupported `uses`
relation with the same subject/object/source. The strict oracle correctly
rejected it. A separate Sol inspection corroborated the fixture and existing
extra-claim negative tests. This is a new extraction-quality failure, not a
recurrence of the proven task-limit defect; missing optional types were allowed
and not responsible. Full diagnosis/receipt is in the capacity-run plan.

No blind reroll or weakened canary is authorized by this diagnosis. Root asked
the user whether to extend the throughput repair to a separately versioned
extraction-grounding candidate, retaining the strict gate and adding positive
and negative controls. Pending that choice, all runs remain stopped and the
heartbeat remains paused. Production and frozen benchmark semantics are intact.
