# Fresh Luna diagnostic pilot after stage-attribution repair

## Evidence and purpose

The capacity pilot `h0nfxj0v` failed with `stage_accounting_failure`, 0/4
scored, 24 turns and 183,828 known tokens. Usage was complete and independent
cleanup verified. Task peak was 130/256 with zero denials. Preserve that run;
never restart it. Its canary was structurally valid but gold did not match.

Root's private-host, finite-label inspection and a separate Sol's actual
frozen-candidate offline reproduction proved that the benchmark accounting
wrapper omitted enabled profile, fact-extraction and retrieval-rerank paths.
These calls were rejected before provider dispatch, not rejected by the model.
The versioned fix names only the real source functions, retains unknown-path
rejection and does not disable any pipeline feature. Ordered evidence is in
`2026-09-29-luna-diagnostic-terminal-repair.md`; authority and full500 gates are
in `2026-09-29-luna-authorized-continuation.md`.

Root reviewed the separate GPT-6 Sol implementation, independently verified
actual dream/augment, fact-replay, reader/judge and staged-grounding calls with
a fake provider, and ran 390 passing offline regression checks. Unknown paths,
exact usage reconciliation, admission/cleanup and policy controls remain.
Root caught and corrected a stale embedded preflight hash before uploading.
Actual source-only Afrodite preflight passed with zero inference: 514 candidate
files, nine code files and the unchanged dataset/runtime. Receipt preparation
also verified all previous runs stopped. This fresh same-four pilot tests the
accepted attribution repair in the real workload; it is not an unchanged reroll.

## Immutable identity

- Root: `/home/atta/.hymem-lme-diagnostic-preflight-a5olbwr6`.
- Unit: `hymem-luna-lme-diagnostic-preflight-a5olbwr6.service`.
- Receipt SHA256: `305a15b3196e4bcb851a6fd39930599909429930330dc977f714642ff8d6154e`.
- Runner `tools/diagnostics/luna_lme_diagnostic_v3.py` SHA256:
  `e52392dff24a1b1234bc995bbe9a4259cfef6a1013e6f7a6133b99bfc73202c0`.
- Launcher `tools/diagnostics/luna_lme_diagnostic_launch_v3.py` SHA256:
  `7324f7a1a12ef19df8eb8be09d449836efe66ac9a58ac450452eaace7b0f4391`.
- Reader `tools/diagnostics/luna_lme_diagnostic_progress_v4.py` SHA256:
  `59dfc88ccb7ce846e8d4e2ba09ce1f0ab7c8e4139a56732b72bc2d682ef67d82`.
- Bundle assembler v3 SHA256:
  `ad6d2580fe747f053b8582a1ac6eb2e38fcfc1047bdc962dbf733a8b1a413df6`.
- Host preflight v3 SHA256:
  `3eaefdefb9a886f873cd22eedc4b9169311245e544b8b2d42f143050fcd9853b`.
- Candidate map SHA256:
  `9e3fe4e495585f63bb560b123fbc0edb6441396ec28ae3f8fa6bfb6ba5b8cfae`.
- Dataset SHA256:
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- Codex binary SHA256:
  `167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9`.

## Unchanged limits and acceptance

GPT-6 Luna low / ChatGPT subscription; four workers; first four questions in
frozen dataset order. Campaign 8,012 turns / 48,160,000 known tokens / 14,400s.
Each question: 2,000 turns / 12,000,000 known tokens / 12,600s; indexing 10,800s.
Canary: 12 turns / 160,000 known tokens / 600s. Server runtime 14,530s plus
10s stop; 256 tasks, 4 GiB RAM, 200% CPU, no restart, control-group kill and
OOM kill. Quota floor, invocation, rotation, isolation and authentication are
unchanged. No production changes or overlapping experiments.

Require `completed_diagnostic_and_clean=true`, all four scored, complete
reconciled usage, zero resource denials and independent cleanup. Report
correct/scored/selected, canary gold, strict indexing health and summary
degradation separately. Semantic degradation is a measured diagnostic result;
technical/provider/accounting/integrity/isolation faults remain fatal. Do not
tune to answers or require perfect accuracy. Not canonical or API-equivalent.

## One-shot dispatch

After checking the launcher hash and updating the monitor, execute once only:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --launch-root /home/atta/.hymem-lme-diagnostic-preflight-a5olbwr6 --receipt-sha256 305a15b3196e4bcb851a6fd39930599909429930330dc977f714642ff8d6154e' \
  < tools/diagnostics/luna_lme_diagnostic_launch_v3.py
```

An ambiguous launch consumes the attempt: inspect read-only, never repeat.

## Exact read-only monitoring

Verify the reader hash above, then:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --root /home/atta/.hymem-lme-diagnostic-preflight-a5olbwr6 --receipt-sha256 305a15b3196e4bcb851a6fd39930599909429930330dc977f714642ff8d6154e' \
  < tools/diagnostics/luna_lme_diagnostic_progress_v4.py
```

While active, no new provider calls, code/model/auth/budget changes, production
actions, restart/resume/reroll or overlapping experiment. Never export raw logs,
text, stores, private rows or credentials. Retry transient unreadable JSON once.
Unknown/in-flight usage is not zero; unchanged question counts are not a stall.
This reader does not expose intermediate canary, per-turn usage, log activity
or stage timing; do not invent them. Notify question completions, terminal
outcomes or actionable source/policy/process/OOM/task-denial/disk-under-20GiB
issues; routine checks stay quiet.

On failure preserve evidence, verify cleanup and follow the sequential Sol/root
repair gates. On clean completion proceed to separately versioned full500
implementation, offline and no-inference host verification before any full
launch. External quota/auth/data blockers require direction, not bypass.

Detached execution and its cleanup bound survive laptop closure. The local
monitor, repairs and next launch need this app/computer available, consistent
with [official scheduled-task guidance](https://learn.chatgpt.com/docs/automations?surface=app).

## Dispatch status

Launched exactly once at approximately 2026-09-29 22:22 UTC, after updating the
existing ten-minute monitor with this exact source/root/receipt identity and
conditional continuation. Launcher returned zero and `never_retry=true`, with
matching receipt. Never repeat this dispatch or resume it.

Root's independent hash-verified reader returned `checkpoint_running`, 0/4
scored, current 25 tasks, peak 31, limit 256, zero denials and no resource fault.
Usage, canary and final indexing/completion health remain unknown. This verifies
startup and policy, not successful completion or proof that all issues are fixed.
While active, freeze code and perform only the metadata polling above. All old
runs and the DeepSeek monitor remain stopped/paused; production is untouched.

## Terminal result

At the 22:53 UTC check, root verified terminal `incomplete_turn_or_usage`:
0/4 scored, 1,069 admitted turns, 7,608,931 known tokens, incomplete usage;
independent process cleanup passed and task peak was 132/256 with zero denials.
Canary structural validation passed; gold mismatch. Final indexing health is
unknown. This attempt must stay stopped. See
`2026-09-30-luna-incomplete-turn-diagnostics.md` for evidence, limitations and
the sequential Sol/root diagnostic repair plan. No new run has been launched.
