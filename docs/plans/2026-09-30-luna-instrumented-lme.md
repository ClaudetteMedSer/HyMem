# Instrumented same-four Luna LME pilot

## Purpose and current state

The user's continued authorization permits an evidence-based same-four diagnostic
LME pilot on GPT-6 Luna low through the existing ChatGPT account, including finite
existing credits. Automatic top-up is disabled by user report, not enforced by
the runner. No purchases, API fallback, model/auth switches or production changes.

The accepted sequential synthetic test h3yk00hg completed 16/16 requests on one
process (65,930 known tokens, complete usage, independent cleanup). It does not
explain the historical real-LME timeout or prove four-worker LME readiness. No
benchmark is currently running. Historical runs, source files and receipts stay
immutable; neither the synthetic test nor prior failed pilots will be repeated.

The next experiment tests the accepted per-window existing-credit handling on
real frozen benchmark payloads and four-worker contention while retaining finite
timeout evidence. This is diagnostic measurement, not a canonical API score or
a full-500 result. Observability is not claimed to fix the original timeout.

## Sequential implementation and root gates

1. A separate GPT-6 Sol agent implements observer v3 and its offline controls.
   Preserve SHA-pinned warm-v9 and v2 timing/queue instrumentation. Replace the
   probe's lifetime 16-record admission guard with a bounded tail, immutable first
   failed observation, finite counters and aggregate timing. The existing shared
   budget remains the sole request-admission authority. Add a narrow session
   observation hook for the structured wrapper without replacing its binding.
   Root independently checks more than 16 requests, failure after request 16,
   rotation, deadlines, usage uncertainty, finite validation and privacy.
2. Only after acceptance, a separate Sol implementation versions structured
   transport integration. It must share observer v3's exact warm-v9 module graph,
   budget and raw observed session while preserving trusted schema and fresh
   thread binding. Root verifies actual ordinary/structured protocol paths and
   successful/failing cleanup before accepting it.
3. A separate Sol implementation versions runner, source-only bundle, launcher,
   host preflight, metadata-only reader and installer. Keep frozen candidate and
   dataset/order, prompts, acceptance and all resource/admission limits intact.
   Export only bounded finite transport summaries and per-client first faults;
   the budget's first failure remains authoritative, with no invented cross-worker
   temporal correlation. Preserve diagnostics on canary/question/cleanup errors.
4. Root reviews diffs and runs independent fault controls, regressions, bounded
   terminal serialization, source-only isolated import and actual no-inference
   Afrodite verification. Prove no previous experiment is active, one-shot launch,
   correct source/receipt binding, exact cgroup policy and recursive cleanup.
5. Only after these gates pass, prepare a fresh immutable private root/receipt,
   record exact hashes and dispatch state here, update the existing paused monitor,
   and launch once. Ambiguous dispatch is investigated read-only, never repeated.

## Frozen experiment and limits

- Same frozen first four LongMemEval-S questions, source order, four workers.
- Candidate inventory: `228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf`;
  candidate map: `9e3fe4e495585f63bb560b123fbc0edb6441396ec28ae3f8fa6bfb6ba5b8cfae`;
  514 accepted candidate files, not the mutable checkout.
- Dataset: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- GPT-6 Luna low, existing ChatGPT account; billing policy
  `included_allowance_or_existing_finite_positive_credits_per_window_v2`.
- Campaign: 8,012 turns / 48,160,000 known tokens / 14,400 seconds.
- Each question: 2,000 turns / 12,000,000 known tokens / 12,600 seconds;
  indexing: 10,800 seconds.
- Canary: 12 turns / 160,000 known tokens / 600 seconds; invocation: 120 seconds.
- Warm process: 16 requests / 300 seconds. Original 4,096-event/final-output limits
  and exact agentMessage delta opt-out remain enforced.
- Server: 14,530 seconds plus 10-second stop, 256 tasks, 4 GiB RAM, 200% CPU,
  no restart, recursive control-group cleanup and OOM policy unchanged.

## Result interpretation and continuation

While active, monitor read-only: no extra calls, code/model/auth/budget changes,
overlapping experiments, restart/resume/reroll, production changes or raw private
text/log/store/credential export. Unknown in-flight usage is not zero.

A clean pilot requires all four scored, `completed_diagnostic_and_clean=true`,
complete reconciled usage, zero resource denials and independent cleanup.
It also requires ten valid observed client summaries, no observer failures or
saturation, and successful-call totals matching admitted budget turns (including
the separate canary total). Attempted-call counts may exceed admitted turns on
failed runs; they are not usage and must not replace the authoritative budget.
Correctness, canary gold match, strict indexing health and summary degradation
are separate results, not requirements for perfect accuracy. Do not tune to
answers or hide technical failures. External access/quota blocks require user
direction; no bypass or fallback. A failed pilot permits only evidence-based
sequential repair and independently verified fresh bounded testing, not rerolls.

Only a clean pilot unlocks preparation of the separately versioned, source-bound
full-500 diagnostic under the user's existing authorization. No full run is
launched by this plan or by reusing this pilot's eventual receipt.

## Dispatch

Launched once as `afit9i7d` at approximately 17:07 UTC. Gate 3 and actual
no-inference host checks passed. The existing monitor is active. No further
source changes, inference calls or launch attempts are permitted while it runs.

## Root observations during gate 1

Root reproduced a second v2 observer defect without provider calls: force ordinary
age rotation after one successful invented request, then execute a second. The
second success incorrectly carried `precleanup` and a last-event timestamp older
than its own invocation start. Rotation closes the old process inside the new
invocation; the v2 cleanup hook captured the preceding request and prevented the
new request's final snapshot. Pre-admission rejection could likewise capture old
evidence. The v3 implementation now restricts session snapshots to the current
invocation's deadline cycle. Transport timing/admission is unchanged.

Root-owned tests in `tests/test_luna_lme_observer_root.py` exercise 33 actual
fake-wire calls, rotation, timeout at 17, immutable first-failure retention after
tail eviction, unchanged budget rejection, privacy, detached projection and
malformed/saturation controls. Nine passed on the provisional implementation;
final source acceptance follows the implementation agent's completed controls.
The existing baseline transport/integration gate independently passed 731 tests.

Gate 1 accepted and frozen: `benchmarks/codex_subscription_timeout_v3.py`, SHA256
`2d59fb8c59c5e304e557b05dbd346998ed46a9c2ba14a726924dc0517d352778`.
Root reviewed the full v2-to-v3 diff, independently reproduced the old rotation
defect, passed all 14 new agent/root controls and 727 transport/regression controls
(the latter included eight root controls before the ninth was added). Historical
v2 and warm-v9 hashes are unchanged. Gate 2 implementation is now authorized;
no source staging or benchmark dispatch has occurred.

Gate 2 accepted and frozen: `benchmarks/codex_subscription_staged_v6.py`, SHA256
`558a6cdee2c5562ff1bd4106b3abc2dadad77a6b3fd980e69420678248af2900`.
Root independently reviewed the complete diff and verified 18 real framed fake-wire
structured requests across rotation, timeout at request 17 with unknown failed
usage and current-invocation evidence, plus AST identity of the trusted binding
wrapper and stage-validation method. The 67-test combined observer, staged and
root gate passed. A different Sol agent now implements gate 3; no live calls yet.

Root independently exercised the provisional runner's actual ordinary/structured
clients for 36 framed calls under one question budget, then partial construction,
malformed-summary capture, canary exceptions and question adapter/cleanup faults.
All six root controls passed; snapshots remained bounded and private strings were
absent. The broader unchanged transport/timeout/host/installer suite plus new
observer/staged/capture checks passed **1,003 tests**. Runner/reader packaging is
still under implementation and is not yet accepted or deployed.

## Gate 3 acceptance, September 30 17:04 UTC

Root independently reviewed all six new helper diffs and exercised actual ordinary
and structured fake-wire clients, partial construction, canary/worker/cleanup
failure capture, 531-file source assembly, remote archive decoder and isolated
source-only imports. The reader initially reused a first-fault-only event validator,
which rejected valid successes and partial timeouts. A separate Sol audit proved
this; the implementation agent corrected it and the finite failure-code vocabulary.
Root independently checked hundreds of malformed field/type mutations against the
frozen observer, real checkpoint writer/reader integration, ten-slot reconciliation,
bounded terminal payload, incomplete usage and cleanup, installer privacy and
consumed launch markers. Broad controls passed 1,016 tests; a supplementary
diagnostic/installer/one-shot gate passed 203 (overlapping, not summed).

Accepted frozen helper SHA256:

- runner v8: `7f96f2ac53039805d8324055edcc0902d7210195e075300eca1b0fb961764f82`
- bundle v8: `7a22d40fb46ce003456e330d3f40795137a7c7c09236d846c1615afbd074e1ff`
- host preflight v8: `84a1065f5a5bce7e13cd5914b83f3e9097600e7eb3773664fb8d773c64e04814`
- launcher v8: `6c620e446e6dbee4c1a68d1105dadba64510d3beed2a92f3b58f4b79185ef504`
- progress reader v9: `ad83c8c5fc5239646a3502fd91d68256645d8bc7428157e6765b8afcee974eb5`
- instrumented installer v1: `e1bd7af0cfe315b619fb092bc625530bfe0cf6868ddc80654a54d7396c997e28`

Fresh local source-only assembly: `/private/tmp/hymem-lme-instrumented-IjmZdT/bundle`.
It contains 514 unchanged candidate files, 16 pinned code files and one inventory;
no dataset, binary, credentials or launch receipt. No inference has occurred in
this continuation yet. The earlier synthetic tests will not be repeated.

The final read-only Sol audit found no remaining blocker. The reader admits a
fixed finite superset of observer failure-code labels shared with budget/runner
failures; this is not a privacy leak or clean-result bypass, since any observer
failure blocks clean status. Source-pinned emitters use their narrower vocabulary.

## Fresh instrumented diagnostic pilot: afit9i7d

Actual Afrodite preflight passed with zero model calls, exact 514 candidate/16 code
files, frozen dataset and accepted binary. Pilot-only installation/prepare passed
with zero model calls, including old-benchmark inactivity and host resource gates.

- Root: `/home/atta/.hymem-lme-diagnostic-preflight-afit9i7d`
- Unit: `hymem-luna-lme-diagnostic-preflight-afit9i7d.service`
- Launch receipt SHA256: `769811ec0c398474b544295a0ff406229e920ff178293591a0312eeee9548921`
- Policy and caps: exactly the frozen experiment above; no additional access test.
- Dispatch: **prepared, not yet attempted**. An attempted or ambiguous dispatch
  must never be repeated. Check the later dispatch entry before acting.

Verify the local reader SHA256
`ad83c8c5fc5239646a3502fd91d68256645d8bc7428157e6765b8afcee974eb5`, then run:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite 'python3 -I -B - --root /home/atta/.hymem-lme-diagnostic-preflight-afit9i7d --receipt-sha256 769811ec0c398474b544295a0ff406229e920ff178293591a0312eeee9548921' < /Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics/luna_lme_diagnostic_progress_v9.py
```

This is metadata-only. No logs, stores, prompt/response text or credentials are
exported. Intermediate canary/per-turn activity/log timing are not exposed; do
not invent them or treat unchanged scored counts as a stall. Terminal summaries
contain finite observer timing, not a complete per-stage performance profile.
For a read-only disk check use the same SSH options with `df -Pk /home/atta`;
notify below 20 GiB. Never use this monitor to make additional provider calls.

### Dispatch entry, September 30 17:07 UTC

The pinned reader independently reported `prepared_not_launched`. The existing
monitor has been updated to this exact root/unit/receipt/reader and activated.
The sole launch is now being attempted through the SHA-verified launcher v8 over
SSH stdin. **Do not retry this launch even if the dispatch response is lost.**
Only the metadata reader may resolve an ambiguous outcome.

Dispatch returned zero once, `never_retry=true`. Independent reader verification
at 17:07 UTC reported `checkpoint_running`, 0/4 scored, exact running policy,
27 current/33 peak tasks out of 256 and zero task denials. Usage and final
canary/index/summary health remain unknown while work is in flight. Available
disk was 225,556,664 KiB (about 215 GiB), above the 20-GiB floor. This is startup
verification, not a successful benchmark result. No full-500 run is active.

## Terminal observation, September 30 19:07 UTC

The pinned reader independently verified terminal failure and recursive cleanup:
0/4 scored, four failed checkpoint entries, `question_failure` as campaign and
budget stop, 4,586 admitted turns and 33,246,386 known tokens, complete usage.
Task peak 144/256, zero denials; no resource fault. The canary was structurally
valid but did not match gold. No answer accuracy or final indexing-health result
is established by zero scored rows. This attempt is consumed and must not restart.

All admitted calls reconcile with observed successes (11 canary plus 4,575
question calls). No admitted transport failure was recorded. Question 1 has
1,158 successful ordinary calls and no observer failure; the other question
clients each show one `fixed_other` observation before deadline setup after the
campaign stopped. These observations do not identify the original question-level
exception. The runner's broad exception handler reduces non-indexing exceptions
to `question_failure` without preserving their class or phase.

Ordered read-only diagnosis before any repair or further inference:

1. A separate GPT-6 Sol audits the frozen candidate/runner failure path and what
   retained evidence can establish the original failure, without editing it.
2. Another Sol prepares a source/receipt-bound, finite metadata projection of
   the four retained indexing summaries and checkpoint failure codes. Root
   reviews and tests privacy/type/boundary controls before running it read-only.
   Raw logs, text, rows, stores and credentials remain private on Afrodite.
3. Root independently checks the evidence, distinguishes an originating fault
   from sibling cancellation, and proves a concrete defect before a narrow
   versioned Sol repair. No reroll or full-500 launch follows from this failure.
   Missing required evidence or external access/quota blockers require pausing
   for user direction, not guessing a fix or relaxing gates.

### Post-terminal read-only evidence

Root accepted and ran the separate Sol metadata projections only after source
review and offline privacy/type controls. Frozen helper v1 SHA256:
`a892dbd759ab0ebd5bfacf81eee0b7e5e00f5de530cf1cc340453f235d164826`;
v2 SHA256:
`6a81b7feb31849e08dc4c860b21a59c1867074b9220c2068e93eb188de251ffd`.
Both require the exact source/receipt, failed terminal and independent cleanup
before fixed private metadata reads. They export no raw records or free text.

Question index 1 (the second question) retained a complete mechanical indexing
summary after 11 cycles and 7,070.703 seconds. It has 79 quarantined extraction
chunks, zero pending/malformed/non-chunk-quarantine counts, zero terminal source
loss or coverage failures, and healthy summaries (zero degraded/missing/malformed
sessions). Strict indexing is unhealthy. No private answer row or diagnostic
admission decision was persisted; other questions lack final indexing summaries.
This is not an accuracy result and does not prove the quarantine is admissible.

Root reproduced a separate reporting defect offline: distinct exceptions in the
actual worker lose their class/phase, and the checkpoint's safe failure filter
reduces the runner's question-failure labels to `unspecified_failure`. This explains
missing attribution, not the original live failure. Source inspection and a Sol
zero-inference real-candidate empty-store check found no definite post-indexing
API/schema defect. A further separate Sol is preparing a bounded read-only held
failure-category census, with source-bound semantic classification and no store
initialization or content export. Root must review and verify it before use.
No additional inference or rerun is justified at this point.

The separate Sol held-census helper was independently reviewed and passed 28
offline controls, including real SQLite mutation/ATTACH/view denials and frozen
reason-vocabulary checks. Its frozen SHA256 is
`c54817392dfa0cb939495b4ac0ba0a9e329a57a034285ce9c4ae66e3469ee1fc`.
The read-only execution used `mode=ro&immutable=1`, rejected any nonempty WAL,
retained the source/receipt/terminal/cleanup gates, and emitted only finite counts.
The conservative current-generation census equals all 79 reported held chunks:
74 `call_failure`, five `branch_incomplete`, zero admissible semantic failures.
Thus the recorded deficits do not meet diagnostic-mode admission; they must not
be relabeled quality misses to force scoring. This does not recover the discarded
original exception or establish which precise post-indexing statement raised.

All admitted transport calls still have successful observer records. Their stage
allocation on failed questions was not retained, so do not assume each failed
extraction corresponds to a successful transport call. Source inspection places
`call_failure` around the completion/attempt-measurement boundary. A Sol offline
replay of the exact frozen counting/heartbeat/memory/accounting wrapper chain with
the real SharedBudget and invented responses succeeds; there is no demonstrated
unconditional wrapper or budget-shape defect. Root is independently verifying that
replay and reviewing a finite exception-class log census, which must never export
raw lines. No rerun or full-500 launch is authorized by this evidence alone.

Root independently reran `test_luna_frozen_question_call_failure_seam.py`: the
normal frozen chain passes; an injected after-return heartbeat exception yields
`call_failure` despite a successfully accounted turn. This is a fault control,
not attribution of the historical cause. A source-pinned, bounded exception-type
census v1 (`d8aff9369a6586079f5a8d4f12faf350828f7af2e01b3aeeda559057897cd6c8`)
passed 93 offline controls and found zero exact bare warning lines in
`private-diagnostic-run.log`. That result does not prove no warnings were logged.
The source also defines `private-launch-stderr.log`, and a preexisting logging
handler can retain the original stderr after Python redirection. A versioned
extension is being reviewed for only those two paths and the exact standard
`WARNING:hymem.extraction.chunk:` logger prefix, with the same closed counters,
size/deadline limits and privacy gates. No raw lines are to be exported.

### Postmortem disposition: required attribution unavailable

Root reviewed the separate Sol two-log census v2, independently exercised its
privacy, exact-prefix, size, deadline and terminal/source gates, and verified
54 selected offline controls. Frozen helper SHA256:
`eca3213ee7ae23750103b953fffdafe3ea55713befc51c0a0def3e447f790fbb`.
The single read-only execution found zero matching exception-class warnings in
both source-defined log files. Source, receipt and independent cleanup gates
passed. Zero matching warnings does not mean no application failures occurred;
the retained held-chunk evidence already establishes those failures.

The original application exception and phase remain unavailable. The reproduced
reporting defect explains loss of attribution, but neither it nor the synthetic
after-return heartbeat control establishes the underlying live cause. No runtime,
candidate, model, billing, budget or production change was made during this
postmortem. No further inference, pilot reroll or full-500 launch is justified.

Pause the existing Luna heartbeat for user direction under the missing-required-
evidence rule. All attempts remain consumed and stopped; preserve the immutable
sources, receipts and private evidence. Do not resume merely because this entry
describes a possible next diagnostic.

Proposed next work, pending direction:

1. Separately implement finite first-application-fault class/phase capture in a
   versioned diagnostic wrapper and compatible checkpoint projection, without
   exporting messages, tracebacks, inputs or outputs and without changing gates.
2. Root independently reproduce the attribution loss and verify the fix across
   the frozen extraction wrapper chain, cancellation, privacy and cleanup paths.
3. Agree a tightly bounded question-path diagnostic with a concrete purpose,
   explicit caps and fresh source-bound receipt before any new live calls. It
   must stop at the first attributable fault, not repeat the four-question pilot
   just to see whether it passes. Any underlying repair then needs its own
   evidence, separate Sol implementation and root verification.

The result remains 0/4 scored with no accuracy estimate. The second question's
summaries were healthy but strict extraction indexing was not; other questions
have no final indexing-health result. Complete recorded usage is 4,586 admitted
calls / 33,246,386 known tokens, not a monetary cost estimate. Runtime cleanup is
independently verified; private evidence is retained, not deleted.

## Approved first-application-fault diagnostic

The user approved the proposed first-error capture, independent offline checks
and one small explicitly capped diagnostic. This supersedes the postmortem pause
only for this workflow; it does not authorize another four-question pilot by
itself. The existing monitor remains paused during implementation. All consumed
roots and receipts, frozen candidate and transport sources remain unchanged.

Purpose: distinguish the first application exception inside the frozen extraction
completion/attempt-measurement wrapper from transport failures, budget stops and
later indexing rejection. Persist an immutable finite class/phase projection
before a catch discards it. Never export messages, traceback strings, frame
locals, prompt/response text, source session identifiers, private rows or stores.

Ordered gates:

1. Separate Sol read-only design audit of the actual frozen extraction path.
2. Separate Sol implements a versioned, source-bound in-memory observation hook
   and finite checkpoint/result validator. Root reproduces the old loss and tests
   real frozen wrapper paths, swallowed exceptions, cancellation, privacy,
   restoration and first-fault immutability before acceptance.
3. A different Sol integrates a fresh diagnostic runner and source-bound host
   launcher/reader. Root reviews and tests admission, source isolation, terminal
   accounting, bounded serialization, one-shot launch and recursive cleanup;
   then performs actual zero-inference Afrodite containment checks.
4. Record a new private root/unit/source hashes/immutable receipt and update the
   existing monitor before the sole live dispatch. Never repeat an ambiguous
   attempt. Keep all private evidence on Afrodite.

Experiment limits: one worker, a fresh store for frozen dataset question index 1
(the second question), original ordered haystack ingestion and frozen candidate
dream/extraction path. No answer, gold comparison, judge or scoring, and no repeat
of accepted synthetic/canary diagnostics. Stop at the first captured application
fault, explicit provider denial, transport fault, or diagnostic cap. Maximum 12
admitted turns, 160,000 known tokens and 600 seconds; invocation 120 seconds,
original warm-process 16-request/300-second and 4,096-event/output bounds. Retain
256 tasks, 4 GiB RAM, 200% CPU, no restart and control-group cleanup; detached
service runtime maximum 730 seconds plus ten-second stop. Known-token admission
retains the existing observed-usage semantics; in-flight usage is not zero.

Model/auth/billing remain GPT-6 Luna low, the same ChatGPT account, included
allowance or existing finite positive credits per-window-v2. No purchases,
reload, API fallback, quota bypass or production changes. Automatic top-up is
user-attested off, not enforced by the runner. External access/quota blocks stop
the diagnostic and require direction.

This is an attribution probe, not an LME score or a quality pass. Failure capture
is not a repair of its cause. No reproduced fault before the cap is inconclusive;
do not automatically increase caps or repeat it. A single worker also does not
reproduce the old four-worker contention. Any subsequent underlying repair needs
its own concrete evidence, separate Sol implementation and root verification.

### Capture gate accepted

Separate Sol implementation `tools/diagnostics/luna_application_fault_capture_v1.py`
is frozen at SHA256
`a4436c0187852b8f342d166bd6fcb245092c29ead127891753ce263f02ed1e18`.
Root reviewed the source, required corrections for the grounding cause path and
phase attribution, and independently reran 48 selected agent/root controls. The
actual frozen counting/heartbeat/memory/accounting chain was exercised before
and after model return, including primary-error preservation through cleanup,
exact-type classification, code-identity spoof rejection and closed metadata.
The first-error callback receives only a detached finite projection before
unwinding, enabling a durable checkpoint even if later cleanup hangs. Callback
failure stops the probe; source and nested-install conflicts fail closed.

This gate repairs attribution only, not the unknown live application cause.
No inference or server modification has occurred. A separate Sol now implements
the thin one-question probe using the frozen transport/adapter graph; root must
accept that output before host launch integration and actual preflight.

### Concrete staged-proxy defect reproduced before live dispatch

Root found and independently reproduced a defect in the frozen candidate:
`_CountingPhase1LLM` has `complete` but neither `complete_stage` nor forwarding
for that method. The actual dream constructs this as the outer extraction
wrapper; nonempty extraction calls `client.complete_stage` during grounding.
The call therefore raises AttributeError before reaching the staged transport.
Root's `test_luna_counting_staged_gap_root.py` exercises the actual frozen
Counting -> Heartbeat -> Memory -> Accounted chain with invented responses:
two successful/accounted ordinary calls, zero staged calls, `call_failure`.
The accepted capture records the original `attribute_error`. A separate Sol
independently confirmed the call sites. This matches the historical failure
shape but does not prove it accounts for every historical quarantined chunk.

Probe implementation was paused before any live calls. Under the existing
evidence-driven repair authority, first repair this concrete defect rather than
spending live calls to rediscover a deterministic local error. A separate Sol
will derive a new isolated candidate from the frozen source, preserving all
historical files and receipts. Add explicit structured completion forwarding
with the same counting, attempt measurement and lease heartbeat semantics, and
include that entry point in class and per-instance producer integrity guards.
Do not introduce an unrestricted forwarding path or relax producer validation.
Root must verify ordinary/structured success and exception paths, exact argument
forwarding, accounting, heartbeat before/after, lease loss and tampering controls.

Only after that narrow repair is accepted may another Sol bind a new versioned
capture/probe to the new candidate inventory/hash and unchanged transport,
dataset/order, model/account/billing and diagnostic caps. The attribution probe
will then test the repaired question path, not the old broken wrapper. No live
canary, old-diagnostic repeat, pilot reroll, scoring or full500 is introduced.
No production deployment or mutation of the mutable checkout's candidate files
is authorized by this isolated diagnostic repair. New source identities and
actual zero-inference host verification remain required before one launch.

### Isolated staged-proxy repair accepted offline

Root reviewed the two-file diff and independently ran 14 candidate/negative-control
tests, including its own full nonempty extraction: two ordinary calls followed by
one valid staged grounding call, one accepted triple, matching stage usage,
logical/provider-attempt counts and two lease checks per invocation. A two-call
chunk limit still rejects grounding before dispatch. The original frozen source
still reproduces the missing-method error. Raised calls, multi-attempt accounting,
pre/post-call lease loss, class/instance tampering, every source hash and malformed
input trees were checked. All controls passed. The extraction contract identity
is unchanged. This proves the local wrapper defect repaired, not all live failures.

Accepted derivation helper `luna_staged_proxy_candidate_v1.py` SHA256:
`ee5185ca5c690b2528055205024efb10bb25aeb86a6b14e9a1aba9403f731e36`.
Source-only candidate: `/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle`.
Its 514-entry candidate map is
`f22cd2be376f019efa1d39cb6c2f1e43ffef2d7ea3bb7a07ac64479241cd4b11`,
inventory file SHA256
`1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd`.
Only `hymem/dreaming/runner.py`
(`94c36844910d962d21a08df657ba03beff16731028b83f21f9c15386074965f5`)
and `hymem/extraction/producer.py`
(`d19ee1e4a61201ba12fc7fcbc13d26a017d7e2fefef26a1b67a00f98e466b695`)
changed. All other 512 files and all historical source/receipt bytes are preserved.

Next gate is a separate versioned source-bound loader/capture integration for
this candidate, followed by root review before the probe resumes. No server
write or inference has occurred, and the monitor remains paused.

### Repaired-source loader and capture accepted

Root reviewed the versioned integration diff and independently passed 18 new
agent/root controls. The source-only import assembled the accepted candidate
with the frozen transport graph without loading benchmark data or invoking a
provider. It is explicitly non-runnable; the separate runnable loader still
requires the exact dataset and runtime binary. The new CLI cannot launch.
The capture requires the exact repaired dream/producer and runner/transport
modules and compiled methods, covering ordinary and structured heartbeat calls.
Root additionally injected first structured faults after settled usage, hostile
exception class names, subsequent cleanup errors and replaced callables; first
fault, stage accounting, finite metadata and fail-closed source gates held.

Accepted `luna_lme_diagnostic_v9.py` SHA256:
`b3e1135893a715dec4e138c25f6bf3c70f3912dbd5df81014ee7c8f2767fd278`.
Accepted `luna_application_fault_capture_v2.py` SHA256:
`e865366dc7ae1fc5cb72367c7f3d59c3c3d3227486728e7035b6e61242a44977`.
The historical v8 runner/v1 capture bytes remain unchanged. V1 capture had an
optional dream-source binding that would omit new heartbeat attribution, so it
must not be used with the repaired candidate. Only v2 is accepted for the new
probe. The independent existing transport/capture/candidate regression sweep
also passed 95 tests (overlapping the narrow gates, not an additional live run).

The next separate Sol implementation is the thin one-question probe. It must
reject source-only imports, preserve the original selected-row digest, publish
first-fault metadata atomically before cleanup, and keep a bounded inconclusive
outcome distinct from application/transport/resource failure. Host integration
and root offline plus actual no-inference containment verification remain ahead.

### Host boundary design for the single probe (not dispatched)

The separate Sol read-only audit selected the timeout-v3 host/launcher/reader
algorithms as containment templates and the LME-v8 source bundle/preflight as
source-assembly templates. Old pin constants, four-question receipts and old
entry points must not be reused. New versioned files will bind the accepted
candidate, runner9, capture2, final probe and exact unchanged transport closure.

The new private root needs two distinct irreversible modes: no-inference
containment verification, then the single probe. A durable attempt marker must
precede each systemd dispatch and an execution marker must precede work inside
the service. Ambiguous attempts remain consumed. Preparing the probe requires
independent post-exit policy and recursive cleanup verification for containment.
Host admission checks UID/user bus, the pinned runtime, disk >=20 GiB, available
RAM >=6 GiB and all previous Luna/LME/DeepSeek units plus the cancelled container
stopped. The reader rehashes the source/receipt closure and projects only bounded
validated metadata; it must not import the candidate, read stores or export logs.

The receipt explicitly binds original source question index 1 and its canonical
digest, one executed worker, no canary/search/answer/judge/scoring, the 12-turn /
160,000-observed-token / 600-second caps, indexing <=540 seconds, unchanged
transport limits and 730-second server runtime plus ten-second recursive stop.
No source-only import is inference admission. The zero-inference hosted checks
must also verify actual dataset/source-order availability without exporting any
row text. A first-fault checkpoint remains useful evidence even if the final
result is absent; independent cleanup is always a separate requirement.

### Thin probe accepted offline

Root reviewed the separate Sol probe implementation and independently passed
45 agent/root controls. Root caught and required a fix for duplicate question
registration: the actual ordinary transport owns registration and the structured
transport joins it through the existing alias; the probe must not pre-register.
The exact transport constructor was exercised and closed with zero sessions and
zero model calls. An independent actual repaired adapter/store/ingestion/dream
test, substituting only invented data and fake provider returns, stopped at
exactly 12 admitted calls with reconciled usage and clean resource ownership.
A second actual-path control captured the first injected application fault after
three calls, preserved the durable checkpoint and completed cleanup. Root also
tested malformed terminal states, first-fault persistence through resource and
cleanup faults, capped inconclusive outcomes and privacy-safe classification.

Accepted `luna_application_fault_probe_v2.py` SHA256:
`017c8d925368517b5be17d8373112c5b362810d37cb62dbcb716fdbc24e90c7c`.
Its `run_probe` requires the loaded runnable source graph, runner/capture modules,
fresh output, containment proof, exact binary digest and a verified resource
callback. It rereads only the frozen second source row and returns its canonical
SHA256, not its text. First-fault publication is atomic and write-once. Complete
usage, stage sums, status/stop consistency and cleanup are validated separately.

No live probe, service or provider call has occurred. Next: a separate Sol builds
the versioned host bundle/containment launcher/metadata reader; root must verify
it offline and on Afrodite with zero inference before the one authorized probe.

### Host boundary accepted offline

Separate Sol implementation and separate read-only Sol audit are complete.
Root independently reviewed all five helpers, required strict canonical receipt
types, terminal denial/OOM checks and preservation of an independently validated
first-fault checkpoint through malformed terminal results or later cleanup faults.
The final combined repair/host/installer suite passed 163 tests, plus the five
implementer's boundary controls (168 total selected tests, no live inference).
Root emulated the exact remote installer locally, verified its complete private
source output, and rejected duplicate/traversal/symlink/drift/extra archive members
before output-root creation. The actual isolated source-only import passed.

Accepted helper SHA256 identities:

- bundle v1: `f6b1dc0b5b1c3998f04ff8ac0c26dc722eca1bd7e2c9611c18d5feb02e30ffd7`
- host v1: `5a136d0a6d64018836982a9411338c2ae8adca3a5e3fa7b1d7e37e457ddc2dbe`
- launcher v1: `6b0e1d3921e7559f86e1213a5f01db5aaa1827ae1cff966a25323da3f99cdccc`
- reader v1: `9242f456e1b1fb00fd932684bb6d44af20b523ef784a6f71ccf854252ad46578`
- installer v1: `45527309dbe08e7c7723aeca082b6b946d391929f95f9f4e730f741bb6b0f059`

Root independently rebuilt the 533-file source-only bundle at
`/private/tmp/hymem-application-fault-root-e0b6Uz/bundle` and generated
`/private/tmp/hymem-application-fault-root-e0b6Uz/install-stdin.py`, SHA256
`1ed84f026c93a2cc6428f0e7c9c9b5367ced9b398782b0db34e3a80d33da1df4`.
The generated script only installs an exact verified source closure into a fresh
private host root; it cannot launch a service, import the candidate or call a model.
Next: execute that source installer once over SSH stdin, then prepare and execute
one zero-inference containment service and independently verify cleanup. Only
after that gate may the sole 12-turn one-question live probe be prepared. Its
immutable receipt and exact monitor must be recorded before dispatch. No server
action or model call had occurred at this acceptance entry.

### Hosted preflight found retained-exited admission defect

The source-only installer created
`/home/atta/.hymem-luna-application-fault-v1-795c0qr7`, closure SHA256
`d16f14556c73674df9b67ebb6315605910f497aa2f6a76fd3b1ca0c5e48d61e5`.
No containment or probe receipt/attempt exists; no service or model was launched.
An initial CLI argument rejection performed no preparation. Exact subsequent
prepare commands failed closed in prior-unit admission. Root's read-only census
proved all 19 blocking units were retained `active/exited` records with MainPID=0,
empty ControlGroup, and independently recursively empty expected cgroups. They
are not live competing benchmarks. V1 incorrectly treats this normal
RemainAfterExit state as running. The earlier generic offline policy tests did
not exercise this host-admission branch.

Narrow repair: a separate Sol creates launcher v2 and installer v2, preserving
all installed v1 bytes and this unlaunched root. Admit a prior service only when
MainPID=0, its state is inactive/dead, failed/failed or active/exited, its cgroup
is either blank or the exact derived user-service group, and the expected group
is independently recursively empty. Reject wrong/nonempty groups, symlinks,
live nested processes/threads and malformed unit names. Do not stop/reset old
units or change production. Root will independently reproduce v1's rejection,
verify v2's strict controls and rebuild a fresh source root before one containment
dispatch. Existing model, candidate, probe, accounting, resource and live-call
caps stay unchanged. This is not a live diagnostic rerun.

Root accepted separate Sol launcher v2 after reviewing its narrow diff and
passing 27 root installer/admission controls plus four Sol regression tests.
Root's actual `host_admission` fixture reproduces v1 rejection and v2 acceptance
only for independently empty stopped units; nested processes, threads, populated
groups, wrong groups, symlinks and unsafe names fail closed. The exact generated
installer was emulated locally for both versions with negative archive controls.
Launcher v2 SHA256:
`544b4b0ddc33feadd73a3556f7bb520ffdd8910e596328202eaaedd1e3147b32`.
Installer v2 SHA256:
`986bf52c2bba25d39c3d3a43aaa475b95da36246a9b37260bb5b8d09d2711a2d`.
All host v1/reader v1/candidate/probe hashes remain unchanged. Root generated
`/private/tmp/hymem-application-fault-root-e0b6Uz/install-v2-stdin.py`, SHA256
`02de86ad8ef8dbb9a67c30ccc1a5a8aa46e5e435c5ff70f2359a25e8954bfc2c`.
The new installer writes a fresh root, retaining the unlaunched v1 root and all
historical services untouched. Source install and containment verification are
next; no live-probe dispatch is authorized until their independent checks pass.

### Fresh attribution root: 2ll7gl_7

The v2 source-only installer succeeded at
`/home/atta/.hymem-luna-application-fault-v1-2ll7gl_7`, source closure
`d16f14556c73674df9b67ebb6315605910f497aa2f6a76fd3b1ca0c5e48d61e5`.
Actual no-inference preparation passed old-unit recursive cleanup, stopped old
DeepSeek container, disk/RAM floors, exact candidate/transport/runtime/dataset
identities and original source-row selection. Containment-only receipt SHA256:
`d764de74408d598e14d904c0d775bc50b1ced16caa5fa1065a36d96a5da02127`.
Unit: `hymem-luna-application-fault-v1-containment-2ll7gl_7.service`.
The next command attempts its single zero-inference dispatch; any ambiguous
outcome is consumed. This service loads/verifies the source and dataset but makes
no provider call. The live probe is not prepared or launched at this entry.

Containment was dispatched once and independently read as `verified_clean`,
`result_verified=true`, `source_verified=true`, and
`recursive_cleanup_verified=true`, with exact unit policy and stopped service.
This is zero-inference containment success, not an LME success.

The sole live probe is now prepared, not yet dispatched at this entry:

- root: `/home/atta/.hymem-luna-application-fault-v1-2ll7gl_7`
- unit: `hymem-luna-application-fault-v1-probe-2ll7gl_7.service`
- receipt SHA256: `6a259b47f4489ebeafa4de98bb737d3af7b3d70592578d9335ed0dd83836f595`
- original source index: 1; canonical row SHA256:
  `e2b389b7ad58e303eed4e9a97b62057aa27c6c23c06488f7c163bb10bdd38f1f`
- source closure: `d16f14556c73674df9b67ebb6315605910f497aa2f6a76fd3b1ca0c5e48d61e5`
- reader: `tools/diagnostics/luna_application_fault_progress_v1.py`, SHA256
  `9242f456e1b1fb00fd932684bb6d44af20b523ef784a6f71ccf854252ad46578`

Root independently read and verified receipt model/auth/billing/notification and
every cap: GPT-6 Luna low, same ChatGPT account, included allowance or existing
finite positive credits per-window-v2, user-attested auto-top-up off; one worker,
12 turns, 160,000 observed known-token threshold, 600-second campaign/question,
indexing <=540 seconds, invocation120 seconds, warm16requests/300seconds,
4096events and unchanged output bounds. No canary, search, answer, judge or score.
Kernel policy: 730 seconds runtime plus10 seconds stop,256tasks,4GiB RAM,200%CPU,
no restart, recursive control-group cleanup. The token threshold can be crossed
by a settled in-flight call; it is not a hard prepaid-token spending bound.

Exact read-only monitoring command, after verifying the local reader digest:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  /usr/bin/python3 -I -B - \
  --root /home/atta/.hymem-luna-application-fault-v1-2ll7gl_7 \
  --mode probe \
  --receipt-sha256 6a259b47f4489ebeafa4de98bb737d3af7b3d70592578d9335ed0dd83836f595 \
  < /Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics/luna_application_fault_progress_v1.py
```

The same SSH options with `df -Pk /home/atta` provide a read-only disk-floor check.
Do not export raw text/logs/stores/rows/credentials. While active, no source,
model, budget or production changes, extra calls, restarts, resumes or rerolls.
Finite terminal metadata and independent cleanup must be assessed separately;
exit status alone is never success. No fault before the cap is inconclusive,
not readiness for four questions/full500. Preserve first-fault metadata through
later failures. On terminal outcome report known usage and limitations and pause
the existing monitor; no automatic repeat or cap increase. The monitor is being
updated to these exact identities before the sole live dispatch.

### Live dispatch: 2ll7gl_7, 21:03 UTC

The existing monitor was updated to the exact probe identities and ACTIVE before
dispatch. The sole probe launch was attempted once at September30 21:02:58 UTC;
the launcher returned attempted=true, command_returncode=0, never_retry=true.
The independent metadata reader then confirmed execution_started=true,
source_verified=true, policy_verified=true, matching group and unit_stopped=false.
No terminal result or first fault exists at this initial observation; usage is
unknown/in-flight, not zero, and cleanup is not yet complete. The receipt and
attempt are consumed. Never repeat this command or resume the run. All further
work while active is read-only metadata monitoring. The server owns runtime and
cleanup limits; the local monitor owns reporting and pauses after the outcome.

### Terminal attribution result: 2ll7gl_7

At the 21:05 UTC follow-up, the pinned metadata reader independently verified
terminal status `inconclusive`, stop `campaign_budget_exhausted`, phase `dream`.
The sole attempt is consumed and stopped. It admitted and returned 12 calls,
with 84,458 known tokens, complete usage, zero in-flight/reserved calls and
reconciled stage accounting. No first application fault was captured. Adapter
and clients closed; exact source/unit policy and independent recursive process
cleanup all verified. Task peak61/256, zero denials. No additional run was launched.

Settled stage evidence:

| Stage | Attempts | Returned/admitted | Known tokens |
| --- | ---: | ---: | ---: |
| Extraction | 8 | 7 / 7 | 52,555 |
| Original grounding, initial | 2 | 2 / 2 | 11,640 |
| Original grounding, recheck | 1 | 1 / 1 | 5,951 |
| Alternative grounding, initial | 2 | 2 / 2 | 14,312 |

All other stage counts were zero. Root checked the `AccountedClient._call`
implementation: attempts increment in `finally`, even when the budget rejects
before admission. The extra extraction attempt is therefore consistent with
the terminal 12-call budget stop, not a thirteenth provider call. The twelve
returned/admitted calls and token sums reconcile exactly.

Five structured grounding calls now succeeded through the repaired real dream
path. This is direct live evidence that the missing-forwarding defect is fixed;
it does not establish that every historical failure is fixed. The diagnostic
stopped at its approved call cap before completing indexing and performed no
search, answer, judge or score. Final indexing/summary quality, answer accuracy,
long-run reliability and four-worker concurrency remain unmeasured. Do not call
this a passing LME question or a readiness result. The existing monitor is being
paused as required. No automatic repeat, cap increase, four-question pilot or
full500 launch follows this capped result; obtain direction for the next step.
