# Claim-first classification v3 — bounded live diagnostic

## Purpose and authorization

Continue the user's explicit sequential Sol repair/root verification/test loop.
The design, root verification and limitations are recorded in
`2026-09-29-grounding-v3-repair.md`. This one changed-candidate measurement tests
whether direct original-claim assessment plus explicit actor/qualifier checks
reduces v2's observed semantic errors. It is not a reroll of the old candidate.
The fixed 24 controls, labels, retained canary and ordinary extraction outputs
remain unchanged. Original claim visibility is a declared contract change, not
hidden relabeling. Do not treat schema or quote compliance as entailment proof.

Five separate Sol implementations passed root review in sequence. The combined
scoped regression passed 1,489 checks. New actual-entry/private-replay faults
pass. No production changes or full-project-suite/LME-readiness claim is made.

## Immutable identities and prelaunch gates

Accepted local bundle: `/private/tmp/hymem-classification-v3-accepted-Yy41cZsw/bundle`.
Derivation receipt SHA256:
`cf426350d77615dbb589a74823d49432a7d7bd0583b26a9b4d3e3360ed64f814`.
All 23 outputs were independently verified after source-only upload to
`/home/atta/.hymem-classification-v3-stage-c6Potc8q`.

Smoke-only root `/home/atta/.hymem-luna-classification-v3-probe-p1i0ye7j` passed
effective containment/resource policy, zero inference admission and independent
unit cleanup. Its receipt is
`767d32429c52afef8e1976c66b93016650ec1507ba4f6d933b2b79038ee2503c`,
sidecar `6782107c1a650a7c437dbaf219f60b83541d1a9067702100bafcfa8620cc0918`.
Never reuse the smoke root for inference.

The sole new inference-only preparation is:

- Root: `/home/atta/.hymem-luna-classification-v3-probe-4ipv54t4`.
- Unit: `hymem-luna-classification-v3-probe-4ipv54t4.service`.
- Base receipt: `891abce871b6f2ae6047f99f140ab240290bd272ef917cd8f7dd5119b1aa1768`.
- Inference sidecar: `09d0b116f2ed8dbfde084124d51ce60b79a989cf731302aba1934fed65c024cb`.
- Startup/observer: `e61eabe7ae53bd915acd092051014f5ba50dfb96b4c1a28f3b88ff2d6ce15cb7`.
- Entry: `79c675a1aa147df6dced69819d0025b13066208df6c5947ab59b0ec467866c2d`.
- Private replay: `d9d8db0bf9c8d03ae8296eeb378a0267501d52c5abb9bc2676d9862ca840a63f`.

Root's private rehearsal helper, SHA256
`656f9ef18865b3efc1218d83ff8e6f471182175b7e6c5f3128443a87bf176dd7`,
verified all 513 candidate files and eight retained ordinary responses against
29 synthetic judgments/203 token sentinels, zero calls and zero writes. Socket,
process and write operations were denied. This is not model accuracy evidence.

## Fixed bounds and policy

One fresh diagnostic only: GPT-6 Luna subscription, same model/auth/low reasoning,
24 controls and retained canary, 29 new turns, 500,000-known-token threshold and
1,800 seconds. Invocation deadline 120 seconds; one worker; service 1,930 seconds
plus ten-second cleanup. Per-control and canary caps, 25%-remaining quota floor,
4-GiB memory/CPU200%/TasksMax256, admission and isolation remain unchanged.
Known usage is incomplete while a call is in flight. No external credit purchase,
quota bypass, larger caps, production, full-500, model switch or raw text export.
All old pilots and cancelled DeepSeek stay stopped and immutable.

Rebind the Luna monitor before one-shot launch. While active, observation is
strictly read-only. Do not change code, call providers, restart, resume or reroll.
Stay quiet on routine heartbeat checks; notify terminal outcomes or actionable
integrity/quota/resource/process/cleanup issues, disk under 20 GiB, or both
progress and stderr inactivity over twelve minutes. Retry an unreadable snapshot
once. Pause the heartbeat on terminal state; never automatically relaunch.

## Reviewed observation and terminal replay

Hash-check the local startup/observer against its SHA above, then run:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite 'python3 -I -B - observe --root /home/atta/.hymem-luna-classification-v3-probe-4ipv54t4 --receipt-sha256 891abce871b6f2ae6047f99f140ab240290bd272ef917cd8f7dd5119b1aa1768 --adapter-sha256 e61eabe7ae53bd915acd092051014f5ba50dfb96b4c1a28f3b88ff2d6ce15cb7 --adapter-receipt-sha256 09d0b116f2ed8dbfde084124d51ce60b79a989cf731302aba1934fed65c024cb' < /private/tmp/hymem-classification-v3-accepted-Yy41cZsw/bundle/adapter-v2.py
```

After terminal state, hash-check the private replay against its SHA above and
independently replay every captured judgment without exporting raw text:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite 'python3 -I -B - --root /home/atta/.hymem-luna-classification-v3-probe-4ipv54t4 --receipt-sha256 891abce871b6f2ae6047f99f140ab240290bd272ef917cd8f7dd5119b1aa1768 --entry-sha256 79c675a1aa147df6dced69819d0025b13066208df6c5947ab59b0ec467866c2d' < /private/tmp/hymem-classification-v3-accepted-Yy41cZsw/bundle/verdict-replay.py
```

Acceptance requires `completed_and_clean=true`, independent private replay and
cleanup. Report false support, false rejection, malformed output, missed
correction/recheck and canary separately. The reader's `semantic_fix_accepted`
stays false because independent root replay is separate. An all-green diagnostic
is not an LME score or a guarantee of long-run reliability. A failure remains
failed; localize it before any new implementation or measurement.

## Status

The monitor was rebound before one-shot launch, which succeeded. The diagnostic
then terminated **failed** after 319.158 seconds: 20/24 controls passed, zero
false support, zero malformed output, three false rejections and one missed
correction. The retained canary failed after eight ordinary replays and two new
grounding calls. This is one draw, not a statistical improvement claim.

Observed usage: **27 subscription turns, 176,340 known tokens, complete usage**.
All source/candidate/receipt/retained-data checks passed. Root independently
verified failed-unit cleanup and all 25 owned process groups absent. Root
privately replayed all 27 captured judgments and reproduced all control/canary
outcomes exactly, with no new calls or writes. Private-result SHA256:
`8411ccaa86974a0b271091bd709043f30cef2454bc3f414f281c4fc0f6489ea6`.
The Luna monitor is paused again; nothing is running. No LME was launched.

Finite, result-hash-bound metadata localized the failures:

| Frozen control index | Meaning | Returned state |
| --- | --- | --- |
| 2 | Explicit preference, qualified to scripts | unsupported |
| 6 | Explicit past use with matching temporal scope | unsupported |
| 9 | Preference resolved through conversation context | unsupported |
| 12 | Use-to-preference correction | unsupported |

Each returned original `not_established` with every alternative also
`not_established`; none claimed ambiguity. Table canary was supported; prose
original and all alternatives were not established. The earlier v2 false-support
controls and fixed control 5 passed on this draw. No failure was relabeled.
The finite metadata helper exports only allowlisted states/control indices and
blocks writes/processes/network; SHA256
`3b735b2d8d92568381d7773e4c5fdf96a56a441ad836f4f614e55af1e4cb7918`.

Next is evidence-driven diagnosis of the observed false negatives, including
the actual intended meaning of null scope/qualifiers. Do not infer a cause from
these aggregate states or weaken the gate to force the frozen labels to pass.
No additional candidate or live campaign is accepted by this failed result.
