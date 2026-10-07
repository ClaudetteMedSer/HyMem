# Fresh Luna diagnostic pilot after the capacity correction

## Purpose and acceptance gates

The failed pilot and ordered repairs are preserved in
`2026-09-29-luna-diagnostic-terminal-repair.md`. User authority for unattended
repair, justified bounded pilots and subsequent full LME is recorded in
`2026-09-29-luna-authorized-continuation.md`.

This is one fresh same-four pilot to verify the restored 256-task policy with
the real dual ordinary/staged workload, not a retry of an unchanged experiment.
The last occurrence's `thread/start` internal error is consistent with the
known 128-task regression but its lost kernel counters do not prove that cause.
The new runner retains finite task-limit observations before admission and
first-fault cleanup. No answer tuning, quality-gate relaxation or additional
capacity counterfactual is permitted.

Root reviewed the separate Sol implementation and independently ran 380 offline
controls, including actual four-row checkpoint serialization, fault-before-
cleanup, exact resource policy and resource-fault-not-success controls. Actual
source-only host preflight passed with zero model calls, 514 candidate files,
nine code files and the fixed dataset/runtime. Old artifacts remain immutable.

## Exact new identity

- Root: `/home/atta/.hymem-lme-diagnostic-preflight-h0nfxj0v`.
- Unit: `hymem-luna-lme-diagnostic-preflight-h0nfxj0v.service`.
- Immutable launch receipt SHA256:
  `8905ddc19dc381cb95ad74250b06dc3a924809f5de92094c3653365972aed771`.
- Runner `tools/diagnostics/luna_lme_diagnostic_v2.py` SHA256:
  `5a9b29456c8fcc96fd9df7d1532a457662eccba63e5ef39f737121fe349565d9`.
- Launcher `tools/diagnostics/luna_lme_diagnostic_launch_v2.py` SHA256:
  `1ece110f0ff84d5e08c1178e5da91913aff20b35bf571385925ef953e3a176b8`.
- Reader `tools/diagnostics/luna_lme_diagnostic_progress_v3.py` SHA256:
  `91d2388a2ec558ac49e040b7bd5e2378c0c9abd1aab6b993d785bd7838f34242`.
- Candidate map SHA256:
  `9e3fe4e495585f63bb560b123fbc0edb6441396ec28ae3f8fa6bfb6ba5b8cfae`.
- Dataset SHA256:
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- Codex binary SHA256:
  `167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9`.

Receipt preparation independently verified source/dataset/runtime identity,
prior benchmarks stopped and host resource floors. No inference occurs during
preparation. No current live run may overlap this one.

## Fixed limits and semantics

GPT-6 Luna low, ChatGPT subscription, first four dataset questions, four workers.
Campaign: 8,012 turns / 48,160,000 known tokens / 14,400 seconds.
Per question: 2,000 turns / 12,000,000 known tokens / 12,600 seconds.
Indexing: 10,800 seconds. Canary: 12 turns / 160,000 known tokens / 600 seconds.
Systemd: 14,530 seconds plus ten-second stop, **256 tasks**, 4 GiB RAM,
200% CPU, control-group kill, OOM kill, no restart. Invocation, rotation,
quota floor, isolation and authentication controls are unchanged.

This is explicitly diagnostic, not canonical/API-equivalent. Semantic
quarantine and degraded summaries may be measured and scored; unknown/mixed,
provider/runtime, accounting, source or isolation faults remain fatal.
Keep correct/scored/selected, strict indexing health, summary degradation and
canary gold match separate. A low score is not grounds for rerolling.

## One-shot launch (root only)

Verify the launcher hash above first. Execute once only, after updating this
plan and the existing monitor with this identity. An ambiguous dispatch consumes
the one-shot marker; inspect read-only, never repeat the command.

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --launch-root /home/atta/.hymem-lme-diagnostic-preflight-h0nfxj0v --receipt-sha256 8905ddc19dc381cb95ad74250b06dc3a924809f5de92094c3653365972aed771' \
  < tools/diagnostics/luna_lme_diagnostic_launch_v2.py
```

## Read-only monitoring

Verify reader SHA256 above, then use this exact metadata-only command:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --root /home/atta/.hymem-lme-diagnostic-preflight-h0nfxj0v --receipt-sha256 8905ddc19dc381cb95ad74250b06dc3a924809f5de92094c3653365972aed771' \
  < tools/diagnostics/luna_lme_diagnostic_progress_v3.py
```

While active: no extra model calls, code/model/auth/budget changes, production
actions, restart/resume/reroll or overlapping experiment. Never export raw logs,
private rows/text/stores or credentials. Reader exports only validated counts,
finite failure classes and numeric resource evidence. Retry temporarily
unreadable JSON once. Unknown/in-flight usage is not zero; no completed question
alone does not prove a stall. This reader does not export intermediate canary
or log activity, so do not invent those observations.

Notify question completions, terminal outcomes or actionable resource/policy/
integrity/process/disk issues; keep routine checks quiet. Require
`completed_diagnostic_and_clean=true`, complete accounting and independent
cleanup before advancing to the separately versioned full500 implementation
and host verification. If failed, preserve evidence and diagnose; use another
separate Sol for a proven narrow defect, root verify before any justified pilot.
External quota/authentication/data blockers require pausing for user direction.
No automatic full-run restart/resume is authorized.

The detached server run and cleanup bound survive laptop closure. Local repair,
polling and the next launch still require this computer/app to be available.

## Dispatch status

Launched once at approximately 2026-09-29 22:01 UTC. The launcher returned 0,
`never_retry=true`, matching the receipt above. Independent source-verified
reader returned `checkpoint_running`, 0/4 scored, with live systemd/cgroup policy
verified and task counters current 28, peak 34, limit 256, denials 0. Completion,
usage, final indexing health and canary outcome remain unknown; this proves
startup, not successful completion or resolution of the original live fault.

The existing ten-minute `monitor-luna-lme-pilot` heartbeat was updated before
dispatch with these exact hashes/root/receipt and the conditional repair/full500
workflow. All old runs remain stopped. No overlapping experiment, production
action or extra model test was performed.

## Terminal outcome and next repair

At the 22:02 UTC heartbeat, root's pinned metadata reader verified terminal
`stage_accounting_failure`: 0/4 scored, 4 failed, 24 admitted turns,
183,828 known tokens, complete terminal accounting and independent cleanup.
Canary structural validation passed; gold did not match. No resource fault:
task peak 130 under limit 256, zero denials. This confirms the actual workload
exceeded the old 128 limit, without claiming the original lost counters exist.

The frozen run is stopped and must not restart. A bounded, read-only root
helper (`tools/diagnostics/luna_diagnostic_stage_failure_root.py`) verified the
same receipt/source/cleanup and projected only fixed error counts and
source-defined traceback function labels. Raw logs remain private. It found
four stage-accounting tracebacks, two profile-extraction failures and two
fact-extraction failures, with actual candidate functions `extract_user_profile`,
`extract_facts` and `_extract_facts_fresh`. A separate Sol independently
reproduced those real dream paths offline with invented source and a fake LLM.

Concrete defect: the benchmark-only `AccountedClient._stage` allowlist omits
profile and facts, although the frozen candidate enables them. Calls are
rejected before provider dispatch. This is not a model output fault or a
reason to disable either pipeline feature. No new inference is needed to prove
the defect. Audit the complete enabled call graph, then record and implement
a separate versioned attribution repair through Sol; root verifies real
candidate paths, unknown-path rejection, stage usage reconciliation and all
unchanged budget/isolation controls before any fresh receipt or pilot.
