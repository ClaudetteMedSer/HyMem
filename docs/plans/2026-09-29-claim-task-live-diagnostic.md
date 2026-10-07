# Claim-task A/B diagnostic — one fixed measurement

## Purpose and accepted gates

Follow `2026-09-29-claim-task-ablation.md`. This tests whether exhaustive
alternative assessment changes original-claim judgments. It is not an unchanged
v3 reroll, a repaired semantic gate, a canonical score or LME readiness.
Fixed 29-turn schedule: 12 control pairs, one separately nominated preference,
two initial canary pairs. Same full source/candidate/user payload in each pair;
Arm A is exact v3, Arm B is diagnostic original-only. No negative alternatives
are fabricated, no gold labels changed. Control 5 remains scope-ambiguous.

Separate Sol implementations passed sequential root verification. The previous
scoped regression passed 1,601 checks; root then ran 142 contract/transport/core
checks and 33 integration checks (overlapping counts, not a full project suite).
The server containment smoke and private read-only retained-input rehearsal
passed. Rehearsal verified 513 candidate files, eight ordinary replays, 14 exact
input pairs and 29 synthetic judgments, with zero model calls/writes. This does
not establish model accuracy. No production changes.

Local immutable bundle: `/private/tmp/hymem-claim-task-accepted-KXzUiPyJ/bundle`.
Derivation receipt:
`a3924f4b61679eba6f24cc5e2e86c9d8b64fd2f0897b4d2fd484b72656440888`.
Clean server stage `/home/atta/.hymem-claim-task-stage-rQT20cDH`: all 29 outputs
hash-verified. An earlier unlaunched stage (`XiCmPb8y`) contained macOS metadata
sidecars and was rejected before preparation; it is not used.

Smoke-only root `/home/atta/.hymem-luna-claim-task-probe-lwgjv91_` passed effective
policy, zero admission and independent cleanup. Never reuse it for inference.
Receipt `1a8f322a11e77f3d8ce0bf0ba6df3adc90f5475ac49cd944a010710e9d6656d9`;
sidecar `73a91ef31aaa5f649334699144cb0691e232f6a92912281bded5047f7e31ed66`.

## Sole new inference receipt

- Root: `/home/atta/.hymem-luna-claim-task-probe-oo85_tyi`.
- Unit: `hymem-luna-claim-task-probe-oo85_tyi.service`.
- Base receipt: `8ed35fb94e77bc731d2d7b3b98fa0a33361175ff8c2104d02145eb241c1e0ce1`.
- Inference sidecar: `cc0626f9459df8c944c9e32f7ece48e2abb4116d014d3f68b8bb44fcf3433777`.
- Startup/observer: `2c057cf2eb099f2d226439c36ff48e55cf4013790e6f46e19f060b7d6fb4d223`.
- Entry: `975d30fd4f7787b65d31b0eef08df6483dd6ba1cd1e291e84a32ebe722793ebe`.
- Replay: `d6dcc88600e42f9c12eca74f4e31a150b5581d031d25a8c1b13932f02a8adc31`.
- Root rehearsal helper: `3465651f509fc28d9882d5cdeebdb807758571441d9e96fda8bad7223a329fd4`.

One launch only after rebinding monitor. GPT-6 Luna subscription, low reasoning,
same auth. 29 new turns, 500,000-known-token threshold, 1,800 seconds, one worker,
120-second invocations; 1,930-second service plus ten-second cleanup. Per-unit
1 turn/100,000 known tokens/240 seconds. 25%-remaining quota floor, 4 GiB,
CPU200%, TasksMax256. Missing/in-flight usage is incomplete, not zero.
No cap/model/auth changes, credit purchase, quota bypass, production or full LME.

## Read-only observation and independent replay

Verify local startup/observer SHA above, then:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite 'python3 -I -B - observe --root /home/atta/.hymem-luna-claim-task-probe-oo85_tyi --receipt-sha256 8ed35fb94e77bc731d2d7b3b98fa0a33361175ff8c2104d02145eb241c1e0ce1 --adapter-sha256 2c057cf2eb099f2d226439c36ff48e55cf4013790e6f46e19f060b7d6fb4d223 --adapter-receipt-sha256 cc0626f9459df8c944c9e32f7ece48e2abb4116d014d3f68b8bb44fcf3433777' < /private/tmp/hymem-claim-task-accepted-KXzUiPyJ/bundle/adapter-v2.py
```

After terminal state, verify local replay SHA above, then:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite 'python3 -I -B - --root /home/atta/.hymem-luna-claim-task-probe-oo85_tyi --receipt-sha256 8ed35fb94e77bc731d2d7b3b98fa0a33361175ff8c2104d02145eb241c1e0ce1 --entry-sha256 975d30fd4f7787b65d31b0eef08df6483dd6ba1cd1e291e84a32ebe722793ebe' < /private/tmp/hymem-claim-task-accepted-KXzUiPyJ/bundle/verdict-replay.py
```

Keep raw logs/model/source text/stores/credentials private. During active polling
no writes, calls, launches/restarts/resumes/rerolls or code changes. All old runs
and DeepSeek monitor stay stopped. Notify terminal outcome or actionable policy,
source integrity, quota/auth, process/OOM, cleanup, disk under 20 GiB, or both
progress and stderr inactivity over twelve minutes. Retry unreadable JSON once.
Stay quiet otherwise. Headless limits/cleanup are server-owned; local monitoring
and repair require this app/computer.

Diagnostic completion needs `completed_and_clean=true`, independent private
replay and cleanup, not just exit status. The reader deliberately always says
`semantic_accuracy_accepted=false` and `full_lme_ready=false`. Report finite
original states, A alternatives, malformed judgments, usage and timing; do not
treat B as a correction selector. Pause monitor at terminal state. Use the data
to justify a next repair; never blindly reroll or resume this experiment.

## Status

The one-shot diagnostic completed in **272.397 seconds**, with all 29 turns
returned, **180,693 known tokens**, complete accounting, no restarts/OOM, and
all 29 owned process groups absent. The reader reported
`completed_and_clean=true`; root independently replayed every returned judgment
without model calls/writes. Result SHA256:
`46eb87cf31c4c34e713c542f86e4f5d88af825283eedf24f218697907523c461`.
The monitor is paused; nothing is running. This is clean diagnostic execution,
**not** a semantic pass: 26 valid and three contract-rejected responses.

Finite, hash-bound metadata (root helper SHA256
`6245de9c0fe0bfdaa59e070f10cb3d1a45763c7d38b1f4f5d03f2ee6c481d691`)
shows:

- Controls 2, 3, 5, 6, 7, 11: original supported in both arms. Control 5's
  predeclared ambiguity remains; the prior misses on 2/6 did not reproduce.
- Control 9 and initial table canary: B supported, A all-negative. This suggests
  task sensitivity on this draw; it does not prove causation or stable accuracy.
- Control 12: original `uses` rejected in both, A nominated `prefers`; the separate
  explicitly nominated B `prefers` claim was supported.
- Control 13: original `prefers` rejected in both, but A missed the `uses`
  correction. B cannot assess correction discovery or uniqueness.
- Negative numeric control 17 and role control 21 were rejected in both arms.
- Negative prefix control 19: both arms claimed support but failed the existing
  `evidence:context_scope` guard. Neither false support was accepted.
- Initial prose: B rejected `uses` (correct for this candidate). A's original was
  also negative, but a supported alternative failed `evidence:context_missing`.

No gate is weakened, no failed result relabeled, no unchanged run rerolled, and
no LME launched. Next is narrow diagnosis of citation binding and staged
original-assessment/correction work before any new candidate is accepted.
