# Luna LME diagnostic: authorized four-question launch

## Authority and identity

The user explicitly said “launch it” after accepting the prepared four-question,
four-worker GPT-6 Luna subscription diagnostic. This authorizes one fresh
diagnostic execution, not canonical/full-500 scoring, a reroll, model/auth
changes, or any production changes. All previous pilots and the cancelled
DeepSeek benchmark stay stopped.

- Root: `/home/atta/.hymem-lme-diagnostic-preflight-oi233cee`.
- Unit: `hymem-luna-lme-diagnostic-preflight-oi233cee.service`.
- Immutable receipt SHA256:
  `67ce8fec27a8848612c9ae99f3cbd9cf42da41c31762ed7970eca25604548c2c`.
- Launcher SHA256:
  `145dc2a3ad08d440dd750e1bfd22a8b3a7495f6d1309b8bf45d4e82e4057f5bf`.
- Runner SHA256:
  `a8f12c305825806fe4ce2b2f63025098632250f33f224dbee5b1fb6f104d5515`.
- Metadata reader SHA256:
  `3ceb2466ce0a3848e42e2264f1061f7905fd13a92deca84af812f0d79860480a`.

Implementation/verification history is in
`docs/plans/2026-09-29-diagnostic-lme-and-honcho-hang.md`. Root rechecked
prepared state and local source pins immediately before this launch. A separate
Sol read-only audit found no source/argument blocker. The launcher independently
rechecks current resources, stopped prior units, all frozen source/dataset/binary
inputs and the receipt before consuming its one-shot marker.

## Fixed limits and interpretation

Model/auth remain GPT-6 Luna, low reasoning, ChatGPT subscription. Four selected
questions run with four workers. Campaign caps remain 8,012 turns, 48,160,000
known tokens and 14,400 seconds. Per question: 2,000 turns, 12,000,000 known tokens,
12,600 seconds; indexing 10,800 seconds. Canary: 12 turns, 160,000 known tokens,
600 seconds. Systemd bounds: 14,530 seconds plus ten-second stop, 4 GiB RAM,
200% CPU, 128 tasks, control-group kill, OOM kill, no restart.

Diagnostic semantic loss may be measured without aborting the entire run.
Canonical indexing health remains false where appropriate. Unknown/mixed,
provider/runtime, accounting, integrity and isolation failures remain fatal.
Completion, correctness, strict indexing health, summary degradation and canary
gold match must be reported separately. This is not a canonical LME score.

The frozen benchmark candidate does not silently incorporate the production
SQLite hotfix; this run measures the exact approved frozen candidate.

## Sole launch command (root only, never retry)

Verify the launcher hash above before this command. The O_EXCL marker is consumed
before dispatch; ambiguous output requires read-only diagnosis, never a retry.

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --launch-root /home/atta/.hymem-lme-diagnostic-preflight-oi233cee --receipt-sha256 67ce8fec27a8848612c9ae99f3cbd9cf42da41c31762ed7970eca25604548c2c' \
  < tools/diagnostics/luna_lme_diagnostic_launch_v1.py
```

## Read-only monitoring

Verify the reader hash above, then:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --root /home/atta/.hymem-lme-diagnostic-preflight-oi233cee --receipt-sha256 67ce8fec27a8848612c9ae99f3cbd9cf42da41c31762ed7970eca25604548c2c' \
  < tools/diagnostics/luna_lme_diagnostic_progress_v1.py
```

Never export raw logs, private question rows, prompts/responses, stores or
credentials. Reader validates all source pins, checkpoint projections, result
reconciliation, live systemd/cgroup policy and empty descendant cgroup at exit.
Retry a transient unreadable JSON snapshot once read-only. If still unverified,
report it rather than treating it as clean. No model calls, launch, restart,
resume, reroll, production edit or source/cap/model/auth change during polling.

Notify on question completions, terminal state or actionable integrity/policy/
process failure. Unchanged question counts alone do not prove a stall. The
reader does not expose in-flight usage or stage timing; those are unknown,
not zero. It also does not expose intermediate canary status or log activity.
Do not invent those measurements. Preserve all four questions in the denominator.

Require `completed_diagnostic_and_clean=true` for clean diagnostic completion.
Then separately report strict health, degradation, correct/scored/selected
counts and known usage. On terminal failure report safe failure class and known
usage, distinguish unavailable cleanup proof from verified cleanup, and pause
the monitor. Never automatically start the next run. Local heartbeat polling
requires this app/computer; the detached server execution and stop bound do not.

## Launch receipt outcome

2026-09-29, independently observed at 21:25:08 UTC:

- Sole launcher returned `launch_command_returncode=0`, `never_retry=true`,
  matching the immutable receipt and reserved unit above.
- Independent reader returned `checkpoint_running`, scored 0/4. This status
  requires its live systemd/cgroup policy checks to pass, not just a dispatch
  return code. Usage, final indexing health and canary status remain unknown.
- Local launcher/reader controls re-run before dispatch: **19 passed**.
- Existing `monitor-luna-lme-pilot` heartbeat updated in place and activated
  every ten minutes with this exact source/receipt identity, read-only polling,
  meaningful-change notifications, and no reruns. `finish-lme-validation`
  remains paused. No extra run or production action was taken.

The launch is verified; successful completion is not yet established.
