# Staged grounding: fixed eight-unit diagnostic

## Purpose and admission gates

Follow `2026-09-29-staged-grounding-repair.md`. The preceding immutable A/B
measurement found task-sensitive judgments and a concrete evidence-region schema
gap. This diagnostic measures the new source-bound region schema and sequential
original/alternatives/recheck contract. It does not claim to isolate those two
changes statistically, prove broad semantic accuracy, or measure LME performance.

The input order is fixed before observing results:

1. Control 9: supported original preference.
2. Control 12: original `uses` rejected; unique `prefers` correction and recheck.
3. Control 13: original `prefers` rejected; unique `uses` correction and recheck.
4. Control 17: unsupported numeric relation rejected.
5. Control 19: out-of-scope context must not establish the claim.
6. Control 21: incorrect role attribution must not establish either claim.
7. Initial table canary: original `deploys_to` supported.
8. Initial prose canary: original `uses` rejected; unique `prefers` correction
   and recheck.

All frozen source/context bytes and initial predicates remain unchanged. Eight
retained ordinary completions are replayed privately to derive the initial canary
batches. Their original request bytes must match exactly. No re-extraction, input
repair, retries, repeated units, substituted gold labels or refill calls.

Only validated negative originals permit alternatives. Any ambiguity rejects;
only a unique supported replacement with all other alternatives negative can
proceed to an original-only corrected-list recheck. Existing attribution, quote,
prefix, parent and role guards remain mandatory. Malformed output is a quality
failure even if execution and cleanup finish successfully.

## Unchanged bounds

GPT-6 Luna subscription, low reasoning, same authentication. Eight sequential
units, at most three stages each, within the existing 29-turn ceiling,
500,000-known-token stop-before-next-call threshold and 1,800 seconds. Per-unit
limits: three turns, 100,000 known tokens, 240 seconds. Invocation deadline
120 seconds. One active inference worker. Server runtime 1,930 seconds plus ten
seconds for cleanup; memory 4 GiB, CPU 200%, TasksMax 256, no automatic restart.
Quota floor remains 25%; disk floor 20 GiB and memory admission floor 6 GiB.
Unknown or in-flight usage is not zero. Settled overshoot must be reported.

Valid rejections and malformed model outputs end that unit and allow independent
remaining units. Transport, accounting, identity, quota or cleanup failure stops
the campaign. External quota/auth limitations require user direction; do not
switch models, bypass limits or buy credits. No production changes or full LME.

## Pending launch gates

The new contract, gate, transport, physical candidate, core, host bundle, runner,
replay and observer have passed sequential independent root verification.
Startup integration is still under implementation/review. Before any inference:

- Verify a fresh immutable complete bundle and startup sidecar.
- Rehearse the actual retained private inputs without network, subprocesses or
  writes; never export benchmark/model text, stores, logs or credentials.
- Run a separately sealed zero-inference containment smoke, verify cleanup and
  never reuse the smoke root for inference.
- Bind a fresh inference receipt and monitor to exact source hashes, then launch
  once. Record exact paths/hashes and polling command here before launch.

Startup is accepted: 38 focused checks (eight Sol/30 root) after the 595 scoped
pre-startup regression checks. Root verified every derived code byte. Local
bundle `/private/tmp/hymem-staged-startup-root-vBg9BWn4/bundle`; startup receipt
SHA256 `bf4ad18e66943c0ae850c09cea03faa775f377cf25c59178ba156cf3e9f8d0d1`.
Server stage `/home/atta/.hymem-staged-stage-WtacykVG`: all 39 outputs verified.

Smoke-only root `/home/atta/.hymem-luna-staged-probe-1ou7bhwd` passed private
rehearsal, effective containment and independent cleanup with zero model calls.
Receipt `be4b8753cbee99ffca205d56b949e061c4e81e039484918164d904693f348213`;
sidecar `2ec08a935be59195f267ead183db997857240e898070dd6afed51fbc02614841`.
Never reuse this root for inference.

## Sole inference receipt and exact read-only commands

- Root `/home/atta/.hymem-luna-staged-probe-30mduayp`.
- Unit `hymem-luna-staged-probe-30mduayp.service`.
- Launch receipt `21a480e4ebf6d823d916733bd2a414c49dc6311811ad144c50adcdc252c6341d`.
- Sidecar `43b2d8f6abc130a241c878d72088969bceac56d412857503d592bc4717d54f04`.
- Startup/observer adapter `7a0fe60090075fd37acca0cead602a795c2562a1ec8481de14b0c05a5c1b7dce`.
- Entry `82f34e19702b03b48ffeade68a94794c7aa1df18bcd006e340ff8f6272078987`.
- Private replay `33c5fb61f87fe228559c609e63bb19803b72d22dfa188f539b14306b542ff0a7`.
- Root rehearsal helper `670d3fa04385b81f40ac98414e188c95d3fdcb494953a107ca16859bd95f25c0`.

Verify the local adapter SHA above before metadata-only observation:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite 'python3 -I -B - observe --root /home/atta/.hymem-luna-staged-probe-30mduayp --receipt-sha256 21a480e4ebf6d823d916733bd2a414c49dc6311811ad144c50adcdc252c6341d --adapter-sha256 7a0fe60090075fd37acca0cead602a795c2562a1ec8481de14b0c05a5c1b7dce --adapter-receipt-sha256 43b2d8f6abc130a241c878d72088969bceac56d412857503d592bc4717d54f04' < /private/tmp/hymem-staged-startup-root-vBg9BWn4/bundle/adapter-run-staged-v1.py
```

After terminal state, verify the local replay SHA above before independently
replaying retained results without calls or writes:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite 'python3 -I -B - --root /home/atta/.hymem-luna-staged-probe-30mduayp --receipt-sha256 21a480e4ebf6d823d916733bd2a414c49dc6311811ad144c50adcdc252c6341d --entry-sha256 82f34e19702b03b48ffeade68a94794c7aa1df18bcd006e340ff8f6272078987' < /private/tmp/hymem-staged-startup-root-vBg9BWn4/bundle/code/tools/diagnostics/luna_staged_replay_v1.py
```

Read-only while active: no launches, restarts, resumes, rerolls, code/model/auth/
budget changes, additional model calls or production changes. No raw evidence
exports. Notify terminal state or actionable integrity/quota/auth/policy/process/
OOM/cleanup issues, disk below 20 GiB, or both progress and stderr inactivity over
twelve minutes while actually running. Retry transient unreadable JSON once.
Keep routine progress quiet. All old runs remain stopped; DeepSeek monitor paused.
Pause this monitor at terminal state. Service bounds are headless; local
monitoring and repair require this app/computer.

## Terminal interpretation

Require the reviewed observer's `completed_and_clean=true`, exact private replay,
settled accounting and independent process/cgroup cleanup for clean diagnostic
execution. Report per-unit gold match, rejection and malformed output separately.
`semantic_accuracy_accepted` and `full_lme_ready` stay false even if all eight
controls match. A passing focused probe would justify the next bounded physical
integration gate, not a full-500 launch. A failure requires a demonstrated defect
or a specific new diagnostic question before another attempt; never blind-reroll.

## Terminal outcome

The sole run completed and independently cleaned up: eight units, **16 admitted
and returned turns, 100,799 known tokens, complete usage, zero malformed units**.
All eight owned process groups are absent; cgroup empty, MainPID zero, no restarts
or OOM. Reader `completed_and_clean=true`; private replay verified all 16 stages
and responses. Result SHA256
`9b0f520349cf70cf9d4f07583ca0ab02a3f1a3b6a21872c72c25ddfb89f8f11e`.
The monitor is paused and no diagnostic is running.

**Six of eight expected outcomes matched**, not a semantic pass: controls
9/12/13/17/19 and the initial table matched. Role control 21 was atomically
rejected but failed the stricter per-claim gold check; initial prose was rejected
as unsupported, missing the expected preference correction. Both used two stages.
No guard was weakened and neither failed unit was rerolled. Next is finite
private stage-level diagnosis of these two misses before any further repair or
measurement. This result is not LME readiness or a benchmark score.

Root's separate no-write replay confirmed the same result hash and all 16
returned responses. Hash-bound finite metadata helper
`799f263aaf9614e21e69fdcef88d80b681765ee03b79ce6fa06b9de2b62cf197`
(three offline checks) localized the misses:

- Control 21 originals were `[not_established, supported]`; alternatives for
  the first claim were all negative. Thus the pair was rejected, but the second
  original was a sampled false support. No false claim was committed.
- Prose original was negative; every predicate-only alternative was negative.
  There was no malformed response or invalid quote driving that rejection.
- Control 19 original was negative; an alternative remained ambiguous and the
  claim was rejected as intended. Controls 12 and 13 each found the unique
  intended replacement and passed its original-only recheck.

Read-only Sol review found no proved wording/role defect. Next evidence gate is
a private deterministic check that the actual prose batch contains the left cue
and the owned right assertion inside its applicability prefix. Until that check
and diagnosis complete, no new inference run or production activation is justified.

The private evidence gate is now complete. A separate Sol supplied
`luna_staged_failure_evidence_v1.py` (SHA256
`1164c28462256a6077b458cd3866732805a796804cd9f07da31fe4b915a77136`);
root reviewed it and passed nine offline checks, including the actual physical
canary and missing-left/missing-right/truncated-prefix counterexamples. On the
hash-bound private run, accepted replay plus fresh deterministic derivation proved:

- The left preference cue is present in the bound context and the right cue in
  owned text, within the permitted prefix. No parent applicability is involved.
- An explicitly invented, correctly attributed `prefers` alternative and its
  recheck pass the unchanged parsers and selector without model calls. This is a
  mechanical witness, not a repaired model output or a newly measured success.
- The false-supported role claim cites one 15-character owned quote. Its object
  appears literally in that quote; its subject does not. Source mapping is valid.
  Literal absence alone is not a generic proof of wrong attribution (pronouns and
  aliases can be valid); the frozen control's semantics establish this miss.

Therefore these two misses cannot be attributed to absent evidence, applicability
loss, parser rejection, transport or accounting. A generic stricter role witness
design is under read-only review; no such change is accepted or implemented and
no new run is authorized merely by these finite findings. Keep prior gold labels,
quality gates, model/auth and caps unchanged.
