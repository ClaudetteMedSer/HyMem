# Four-question Luna grounding candidate pilot

## Authorization and acceptance

The user approved a separately versioned generic extraction-grounding correction,
strict positive/negative controls, root verification and another bounded LME
pilot. Sol implemented the prompt and source-bound integration; root reviewed
the outputs, ran offline regressions/fault tests, and verified the live paired
diagnostic. Full receipts and limitations are in
`2026-09-28-luna-predicate-grounding.md`.

The paired test completed 44/44 units: each arm passed 20 controls and two strict
canaries, with 112 calls / 846,174 known tokens, reconciled usage and full cleanup.
Root replayed all 112 exact requests offline; the candidate arm used this actual
frozen candidate. The baseline also passed, so this is acceptance evidence, not
a measured reliability gain. No oracle changes, output filters or gold leakage.

Only the approved ten prompt lines differ from the previous 508-file frozen
runtime. Summary/numeric-table fixes, strict canary, model, data order, transport,
quality gates, quota and spend caps remain unchanged. Four questions run in four
independent workers. Production is untouched. All old pilots and the cancelled
DeepSeek run remain stopped; its monitor remains paused. This is neither a
canonical API score nor a full-500 result.

## Exact bound identity

- Root: `/home/atta/.hymem-luna-lme-grounding-ram43gms`
- Candidate: `/home/atta/.hymem-luna-lme-grounding-ram43gms/candidate`
- Unit: `hymem-luna-lme-grounding-ram43gms.service`
- Cgroup: `/user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-grounding-ram43gms.service`
- Launch receipt SHA256: `d1dcc366287638ce2ade1c4eb34ce9d5dcb37673b99f60d20734f99812bc72de`
- Runner SHA256: `5956ebbed67b9e0a7bcbf812d7d819c1878f7fe91f8afd369447b539b2e81745`
- Launcher SHA256: `f1683059fa92f47f7b9a6bcecc1a420489f3c0135212fc474ac83a1105c68475`
- Reader SHA256: `bce6e26197d9831a90ba1015a7b129254c5368fa2adedff29050381ac31a1710`
- Candidate inventory SHA256: `896b89d56f393bf386cef7b400bbb1450306555ccb6d05f4b6c787e616bf104f`
- Candidate source-map SHA256: `217036b8089911c352cdf5994ef2b37915b5c68d2622a23c0642d264487dbe62`

Receipt additionally binds the immutable dependencies, original inventory,
dataset and Codex binary. The runner verifies the exact one-file delta; the
reader verifies it independently. Prelaunch SSH-stdin reader tested on the real
prepared root: receipt/source/data/inventory pins verified, no calls yet.

## Budgets and containment

GPT-6 Luna subscription only; fresh ephemeral thread per request; 25% reported
quota floor, no API keys, provider fallback or purchased credits. Campaign:
8,012 turns / 48,160,000 known tokens / 14,400 seconds. Each question: 2,000 turns /
12,000,000 known tokens / 12,600 seconds; indexing 10,800 seconds. Canary:
12 turns / 160,000 known tokens / 600 seconds. Per invocation 120-second deadline.
Warm process rotation after 16 requests or 300 seconds, four question workers.

Systemd: runtime 14,530s + 10s stop, TasksMax 256, memory 4 GiB, CPU 200%,
OOMPolicy kill, KillMode control-group, Restart no, RemainAfterExit yes.
Private empty working directory; server-owned cleanup independent of laptop.
Host admission: 6 GiB available memory, 20 GiB disk, no prior running pilot.
Exclusive launch marker is consumed before dispatch; never retry ambiguous
launches. All logs, stores, benchmark/model text and credentials stay private.

## Exact metadata-only monitoring

Verify the local reader hash, then run from the HyMem checkout:

```sh
shasum -a 256 tools/diagnostics/luna_grounding_lme_progress.py
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  /usr/bin/python3 -I -B - \
  --root /home/atta/.hymem-luna-lme-grounding-ram43gms \
  --unit hymem-luna-lme-grounding-ram43gms.service \
  --expected-cgroup /user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-grounding-ram43gms.service \
  --receipt-sha256 d1dcc366287638ce2ade1c4eb34ce9d5dcb37673b99f60d20734f99812bc72de \
  < tools/diagnostics/luna_grounding_lme_progress.py
```

While active, only read-only polling: no new model calls, restarts, resumes,
rerolls, code/model/budget changes or production writes. Stay quiet on routine
progress; notify completed questions, terminal outcomes, or integrity/policy/
process/OOM/task-denial issues, disk below 20 GiB, or both log and progress
inactivity over 12 minutes. Retry an unreadable JSON snapshot once read-only.

Require `completed_and_clean=true`, not exit status or terminal.validated alone:
all four question rows must pass index/summary health and score-validity checks,
usage and stages reconcile, clients/stores/leases clean up, and the exact
policy-bound unit has no processes. Correctness is separate, not a reroll
condition. Missing health while indexing is unknown; in-flight usage is incomplete.
Stage totals overlap in wall time; only terminal reconciliation is definitive.
Terminal removed cgroup counters are unknown, not zero.

On clean completion report validated health, correctness, usage, timing, cleanup
and four-question limitations, then pause the heartbeat. On terminal failure,
preserve and report the safe first-failure code/phase and independent cleanup.
The user permits continuing narrow, evidence-driven repairs: first prove the
defect, then separate Sol implementation, root review/reproduction/offline tests
and necessary bounded testing before any justified fresh same-four-question run.
Never blind-reroll, weaken quality/isolation checks, raise spending caps, change
model/auth, bypass quota, launch full-500 or change production. Pause/request
direction for external quota exhaustion or materially broader scope. Update the
receipt and monitor for any new run. Local polling/repair needs the app/computer;
the bounded server-run execution and cleanup do not.

## Status

Dispatched once on September 28 around 20:05:29 UTC; command returned 0 with
`never_retry=true`. Root verified the exact SSH-stdin reader against both the
unlaunched receipt and running unit.

At 70.6 seconds elapsed: strict canary passed in eight calls, all four questions
started concurrently, no question failures or transport first-failure, all live
source/data/inventory pins and resource policy verified. 22 admitted turns,
19 settled calls / 149,124 known tokens and three in flight; final usage is
therefore incomplete, not zero. Observed 115 current / 121 peak tasks against
256, zero task-limit denials, no restarts and 215.794 GiB free disk. Index and
summary health remain unknown until validated question completion.

Existing heartbeat `monitor-luna-lme-pilot` updated and confirmed ACTIVE,
every ten minutes, bound only to this receipt/reader/root. No full question has
completed yet; `completed_and_clean=false` is expected during indexing. This
startup check is not end-to-end success.

## Terminal failure, September 28 around 20:56 UTC

The reviewed reader verified the source/data/policy identities, but reported
`completed_and_clean=false`. The strict canary passed in eight calls; zero of
four questions completed. The originating failure was question index 3,
`fixed_other`, phase `run`, recorded last RPC `turn/start`. The other three
questions stopped with `campaign_stopped`. Last-RPC metadata does not prove
that the start RPC failed: this field also persists while turn events are read.

Elapsed time was 3,078.252 seconds. Accounting: 1,919 admitted turns, 1,918
returned calls, 13,757,776 known tokens, incomplete usage for one admitted
failed turn. These are not complete consumption totals. Final index/summary
health and answer correctness are unknown, with no private result rows.
All four worker cleanup flags were true; independently the unit had MainPID 0,
an empty control-group header, zero expected-cgroup processes and no restarts.
Exit status was 1. Removed terminal task counters are unknown; the last live
observation had peak 134 tasks of 256 and zero task-limit denials. Free disk
was 215.677 GiB. No production changes or new runs were made.

Sol and root independently found a confirmed diagnostic loss: unapproved
transport exception codes are reduced to `fixed_other`, then discarded. The
underlying event was not retained, so a specific provider/protocol cause is
not established. An on-host metadata-only check found no private traceback
files or traceback marker in the run log, and none of the finite error markers
examined. Result and progress were byte-identical, SHA256
`8811f83ac93ebec677b411ee78a24db187027be46eb518a720e812b9a7e4af04`.
Safe-terminal SHA256:
`81299acced01fd76515bf81c308b412f25ed759582b6a59c74fd77cf6ba0ca6e`.
The original private artifacts remain untouched. Repair plan:
`2026-09-28-luna-transport-failure-observability.md`.

A second reviewed terminal read confirmed unchanged results and no remaining
processes. Source-free stage accounting locates the one admitted failed call in
question 3's `extraction_primary_or_other` stage (141 admitted, 140 returned).
This identifies the caller, not the error's cause. The other three rejected
stage attempts were unadmitted calls after campaign stop. All 1,919 admitted
turns and 13,757,776 known tokens reconcile across stages, but incomplete failed
turn usage still prevents healthy terminal accounting validation.
