# Two-question parallel Luna LME pilot — 2026-09-28

## Authorization and scope

The user's “Run it” authorizes this one fresh two-question experiment following
the concurrency repair. It is not a full-500 launch, resumption, retry, or reroll.
The failed single-question pilots and cancelled DeepSeek run stay stopped.
Production services, production databases, frozen candidate sources and scoring
semantics are unchanged. Raw logs, stores, benchmark/model text and credentials
remain private on Afrodite. This is experimental subscription-mode evaluation,
not an API-equivalent canonical score.

## Prepared identity

- Root: `/home/atta/.hymem-luna-lme-multi-wgj_pdpd`
- Unit: `hymem-luna-lme-multi-e54e8f36c520e8ad.service`
- Expected cgroup: `/user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-multi-e54e8f36c520e8ad.service`
- Immutable `launch-receipt.json` SHA256: `76b62b5c6d5f9ce059842c5114e966f2451e661e2008bfd177dca874aea9ee24`
- Launcher SHA256: `8a83f90f5f2afda1535c5145b108432e78ac6630b02b21c99084af7170704c34`
- Runner SHA256: `83fb27a9ee86c7f1d25ab7c6775dec31e9540321347971609de47a065734eef5`
- Concurrent transport SHA256: `e5f1eacbb02f809ee6246449b672dfcc839069da226230572913612acece069e`
- Helper SHA256: `0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0`
- Base transport SHA256: `387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491`
- Inventory file SHA256: `852758cfa63902048669f78f2377db4354b2e196a16ccbb98a3e6cadca2590eb` (508 source files)
- Dataset SHA256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`

Preparation completed without inference. The one-shot `launch-attempt.json`
must never be removed/reset; a lost or ambiguous launch response requires a
read-only check of the exact unit and artifacts, never another launch attempt.

## Envelope and host admission

Two source-ordered questions, two workers, one entry canary. Each question has
its own database, client and output directory. Exact model `gpt-6-luna` on the
ChatGPT subscription through Codex 0.158.0, low reasoning. Fresh isolated
ephemeral process/thread per completion; no API fallback or credits. The 25%
remaining-quota floor applies to every reported window before each admission.

| Scope | Turns | Known-token stop-before-next | Wall seconds |
| --- | ---: | ---: | ---: |
| Canary | 12 | 160,000 | 600 |
| Each question | 2,000 | 12,000,000 | 12,600 |
| Entire campaign | 4,012 | 24,160,000 | 14,400 |

Indexing has 10,800 seconds per question. Token limits are not provider-enforced
output ceilings; in-flight calls may overshoot. Unknown/failed-turn usage is
not zero. Internal HTTP attempts are unavailable. Subscription temperature,
output-token ceilings and JSON-mode controls remain unsupported.

The independent systemd bound is 14,530 seconds plus a 10-second stop interval:
Type=exec, Restart=no, KillMode=control-group, RemainAfterExit=yes, MemoryMax=4GiB,
CPUQuota=200%, TasksMax=128, OOMPolicy=kill. TMPDIR is private disk-backed storage.
No credential file is copied or printed. Admission observed 216.12 GiB free,
7.76 GiB available memory, 8 CPUs, minimum quota 79%, no running prior Luna unit.
Source inventory/data verified, no inference during admission or preparation.

## Verification and monitoring

The prior implementation passed 174 focused tests. Before this launch root
reran 152 transport/runner/pilot/isolation/containment/canary tests and separately
verified launcher/metadata tests. Sol reviewed the launcher; monitoring review
requires exact source/receipt/resource/cgroup identity, not merely exit code.

The metadata reader is `tools/diagnostics/luna_subscription_multi_progress.py`,
SHA256 `2f91778c74e78fef7dd14bf62708f108c7003c33246393ed2d5d904973a9162f`.
Root's 9 launcher/reader tests pass; Sol re-review confirms the cgroup/receipt
proof gaps are resolved. A read-only pre-launch poll verified the fresh receipt,
binary, five bundle files, dataset and 508-file source inventory on Afrodite.
No launch had been dispatched at that pre-launch check.

Exact metadata-only polling command (verify reader SHA256 before each use):

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  'python3 -I -B - --root /home/atta/.hymem-luna-lme-multi-wgj_pdpd --unit hymem-luna-lme-multi-e54e8f36c520e8ad.service --expected-cgroup /user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-multi-e54e8f36c520e8ad.service --receipt-sha256 76b62b5c6d5f9ce059842c5114e966f2451e661e2008bfd177dca874aea9ee24' \
  < /Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics/luna_subscription_multi_progress.py
```

For a completed result require `completed_and_clean=true`, including healthy
indexing and summaries, parseable per-question scores, complete accounting,
verified unchanged sources/data, zero active invocations, empty exact cgroup,
successful exit and clean owned-store lifecycle telemetry. Answer correctness
is separate: an incorrect answer with valid indexing/scoring is not an execution
failure. While work is in flight, missing final health is unknown, not failure.

Polling must never launch, restart, resume, reroll, make model calls, edit code,
change production or export raw private material. Notify on either question's
completion, terminal success/failure, integrity/policy failure, lost process/OOM,
disk below 20 GiB, or both progress and log inactivity over 12 minutes while the
unit claims to run. Retry a temporarily unreadable JSON snapshot once read-only.
Routine progress remains quiet. Terminal completion/failure pauses the monitor.
The server-run campaign and cleanup bound survive laptop closure; local app
polling requires the app/computer to be available.

## Launch receipt

Dispatched once at approximately 09:50:52 UTC on 2026-09-28. The launch command
returned 0; this proves dispatch only. First reviewed metadata poll confirmed
the exact expected cgroup, Type/Restart/KillMode/Runtime/Memory/CPU/Tasks/OOM
policies, running main process, zero restarts, all source/data pins and disk
floor. The canary was in progress (four admitted turns; 33,848 completed-turn
known tokens); no question had started yet. No completion or accuracy claim.
The existing `monitor-luna-lme-pilot` heartbeat is updated for this exact new
two-question unit, every ten minutes; the DeepSeek monitor remains paused.

At approximately 09:52 UTC the canary had passed, both source-ordered question
workers had started, and two client invocations were active concurrently.
There were 15 admitted turns and 112,304 known completed-turn tokens; total
usage remained incomplete/in flight. Both questions were unfinished with no
reported stop. Exact unit/cgroup/resource policies and all pins remained valid,
disk free 216.103 GiB. Final indexing/summary health and accuracy remain unknown
until completion. No rerun or additional questions were authorized/launched.
