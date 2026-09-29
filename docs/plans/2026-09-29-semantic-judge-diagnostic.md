# Source-grounding diagnostic — V2 semantic gate failed; stopped

## Concrete purpose

Test the new source-grounding judgment contract on fixed independent controls,
then test whether it repairs the exact retained observed canary's predicate
substitution. This is not the old accepted paired prompt diagnostic, a new LME
pilot, a benchmark score, or proof that the older transport incident is fixed.

Implementation and root review must finish before a receipt is frozen or any
model call is dispatched. All old sources, results and receipts stay unchanged.
The existing monitors remain paused. No production or full-500 work.

## Frozen schedule and scoring

1. The 24 invented controls in `luna_semantic_cases.py`, in fixture order, once.
   Fixture-plus-label SHA256 is
   `511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925`.
   Use the exact accepted request builder/parser with their unmodified source
   objects. Do not translate them into different extraction-record metadata.
   Twelve supported units must retain all claims; ten rejecting units must
   reject every labeled claim; two correction objectives must return the exact
   expected predicate change, followed by one supported full-result recheck.
   Preserve all other fields. At most 26 new judgments for these controls.
2. Privately replay the eight recorded ordinary completions from the stopped
   observed canary into the actual integrated candidate. Assert byte-identical
   ordinary requests at every replay boundary. Only new grounding requests may
   reach Luna: table initial, prose initial, and prose correction recheck — at
   most three paid judgments. No new ordinary extraction calls. The original
   replay had two triples, not an extra third triple. Exact final canary gold,
   source, type, qualifier, marker, duplicate and path checks remain intact.

Expected maximum: **29 new model invocations**, plus eight replayed ordinary
completions that consume no new inference. Report these separately. New-known
tokens belong only to new model invocations; do not manufacture token usage for
replay or charge historical usage again. A hybrid replay is not a fresh canary.

For controls report false-support claims, false rejection, missed predicate
recovery, malformed verdicts and actual full-result outcomes separately. Never
hide a false acceptance behind aggregate accuracy. Unexpected or wrong initial
judgments are scored failures without speculative corrective rerolls. Continue
independent controls after model-contract rejection, but halt on provider,
quota, accounting, isolation, mapping, integrity or cleanup failure. No second
correction attempt, no automatic rerun, no cap expansion.

The control stage measures model judgment using the accepted wire contract; it
does not write a store or alone claim publication-gate coverage. The integrated
hybrid replay and existing independently verified offline publication controls
cover that distinction. Root must replay all captured new responses offline
against exact requests and labels before accepting a live result.

## Proposed fixed bounds and isolation

One worker, GPT-6 Luna subscription transport v3, unchanged auth and 25% quota
floor. Global maximum 29 admitted turns and 1,800 seconds; stop before the next
call once observed known usage reaches 500,000 tokens (not an output-token cap);
120-second invocation deadline. Invented units: one admitted initial turn and,
only for the two predeclared corrections, one recheck; each unit 100,000 known
tokens and 240 seconds. Hybrid canary: three new admitted turns, 160,000 known
tokens, 600 seconds. Existing 12-logical-call canary acceptance remains intact.
Any exhaustion is a stopped or inconclusive diagnostic, not a pass.
Per-unit token bounds have the same observed-usage semantics: the final admitted
turn can cross a threshold. Here "paid/new" means newly executed subscription
inference, not an API-key charge or a dollar-cost estimate.

Use a fresh private directory and source-bound one-shot launch marker; no
retry after ambiguous launch. Freeze candidate inventory, extraction identity,
fixture/label hash, runner/helpers/transport/binary hashes, retained evidence
hash and original receipt/result hashes, exact schedule and bounds in a new
immutable receipt. Store private requests before dispatch, returned responses
and transport/accounting metadata privately on Afrodite. Export only finite
codes, counts, hashes, timing, usage and cleanup results, never source text.

Server-owned containment: TasksMax256, 4 GiB memory, CPU200%, KillMode control-group,
Restart=no, OOMPolicy=kill, runtime 1,930 seconds plus ten-second stop timeout.
Admission requires old runs stopped, available memory at least 6 GiB and disk
at least 20 GiB. Verify source/import identity, no credential/API-key fallback,
empty task working directory, per-invocation ephemeral threads and warm process
rotation. Verify both owned process groups and service cgroup cleanup before
declaring completed-and-clean. Unknown usage remains unknown.

## Status

Read-only design checked by a separate Sol and corrected by root: both emitting
canary leaves require validation, so retained replay costs at most three new
judgments, not two. Root accepted the canary/stage implementation after 67
offline controls (40 Sol and 27 root), in addition to 137 earlier controls.
New module hashes are recorded in the parent plan. Next is a separate Sol's
runner implementation; no runner/host has been accepted or launched yet.

Root independently reverified retained Afrodite metadata read-only: original
launch receipt and private-result hashes match; `run/private-canary-evidence.json`
has SHA256 `a1fd2d45c4cf50bf483927b07dff5cad145ae3ce8a582f19d13dc1b11a92943c`,
132,062 bytes, eight requests/eight responses and zero provider truncations.
No raw text left Afrodite and no model calls or writes occurred. The first SSH
permission review timed out before execution; its one permitted retry completed.

## Probe core accepted offline

Root reviewed a separate Sol's new callable core and independently verified
**46 offline tests** (seven Sol and 39 root), including all fixed-label request
bytes, positive/negative/correction grading, no label/rationale prompt leakage,
and a synthetic hybrid against the actual candidate. It performs exactly three
new judgments plus eight replayed ordinary completions, reports only new tokens,
and blocks a mismatched ordinary replay before any new inference.

Root also injected provider failure, unknown usage, pre/post-dispatch evidence
write failure, cleanup failure and stage-accounting failure. Paid admission and
known-usage lower bounds remain visible; failures do not allow further calls.
Root required a narrow repair so the hybrid wrapper delegates the owned budget
halt when its stage ledger fails. All 46 tests and a read-only physical-candidate
preflight pass against the final frozen runner:
`3ccb7ec12f93f8502fe8ed1448a071ac977dd334d17821f661913d634c27b334`.

Core results deliberately cannot attest process/cgroup cleanup and always leave
`completed_and_clean=false` until a separate host validator supplies independent
cleanup proof. A fresh Sol implements that host/entrypoint stage next. No model
calls, launch, production changes, or remote writes have occurred.

## Host stage accepted offline

A separate Sol implemented one-shot preparation/launch, the isolated entrypoint
and metadata reader. Root reviewed all three and independently reran **50
offline checks**, including 43 root-authored controls. Together with the 250
previously accepted checks, 300 scoped controls pass; this is not a full-suite
or model-accuracy claim.

Root reproduced and required repairs for strict JSON types, pre-import inventory
validation, embedded warm-module identity, output-file creation conflicts,
pre-inference service/cgroup admission, preservation of usage on result-write
failure, terminal unit-state/resource checks and completed-unit accounting.
The real candidate's synthetic full diagnostic completes 25 units/29 invented
judgments plus eight ordinary replays. A forced result-write failure preserves
known usage and cannot claim completion. No live calls occurred in these tests.

Frozen host SHA256:
`7e190561958b126593a3e652ec2de35fb4cef27d4e5863284a75636abe531b07`.
Frozen entrypoint SHA256:
`60fc7fca900ef8320a410af2f5f01415f973f01907cf7c0456c0b4936611030a`.
Frozen progress reader SHA256:
`68f02223a561dd31a1a0417d8972ddafab7cbd26fc15500a9a2efc1b0cb9d355`.

Next: stage only this sealed diagnostic into a new private Afrodite directory,
derive and verify the 510-file candidate from the untouched previous 508-file
runtime, and run a read-only, zero-inference root rehearsal. Then review the
fresh immutable receipt before the single 29-turn diagnostic is launched.
Neither preparation nor rehearsal authorizes or starts a new LME benchmark.

## Prepared Afrodite instance and root rehearsal

Prepared, not yet launched:
`/home/atta/.hymem-luna-semantic-probe-8wftffh9`;
unit `hymem-luna-semantic-probe-8wftffh9.service`.
Immutable launch receipt SHA256:
`b7814a637ab7468a29150be0dca9d0195013f6684c8f24742c161ccb1991b4c7`.
Canonical 14-source pin-map SHA256:
`ba9ef4a8302ec84abd216864280b70c4e2c6b5f929b659fab0d68af4bb549aa6`.
Codex binary SHA256:
`167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9`.
Root read and independently checked the full metadata-only receipt against the
accepted local source hashes, fixed bounds, model, isolation and retained-input
identities. Original sources and production are unchanged.

Root's SSH-stdin rehearsal script SHA256:
`b7189c2f3ed825bae1c804b2f74441595d64c2105178a10d48c1ea473659fa53`.
It passed against the real retained eight completions and derived 510-file
candidate: 29 **synthetic** judgments, eight ordinary replays, zero model calls,
zero files written. Python audit hooks denied network/process creation and
filesystem mutation. The 203 synthetic token sentinels only exercise accounting;
they are not provider usage. Semantic model accuracy remains unmeasured.

Exact metadata-only polling command (first verify the local reader hash above):

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  'python3 -I -B - --root /home/atta/.hymem-luna-semantic-probe-8wftffh9 --receipt-sha256 b7814a637ab7468a29150be0dca9d0195013f6684c8f24742c161ccb1991b4c7 --expected-source-pins-sha256 ba9ef4a8302ec84abd216864280b70c4e2c6b5f929b659fab0d68af4bb549aa6' \
  < tools/diagnostics/luna_semantic_probe_progress.py
```

Polling must remain read-only. Never relaunch or retry an ambiguous launch;
never fetch raw evidence. Require terminal reconciliation and independent unit
and process-group cleanup, then root private offline replay before accepting
the semantic result. This run cannot authorize another pilot or production.

## First launch and startup defect

The one-shot launch returned zero, but the service exited one before making its
run directory. The metadata reader correctly reported
`ended_without_valid_terminal`, not running or completed. Root separately
verified MainPID zero, zero restarts, empty ControlGroup and absent expected
cgroup. No production or old experiment changed. The monitor is paused.

Root proved a new launcher integration defect: `env -i` removes the user-bus
environment; the entrypoint's `systemctl --user show` therefore cannot connect
and fails before `run.mkdir` and `run_campaign`. The exact status query with the
sealed environment returns one and a fixed missing-bus error; adding only
`XDG_RUNTIME_DIR=/run/user/1000` and
`DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus` returns zero. Source and
retained-data preflight still pass. With immutable entrypoint ordering and no
run directory, this establishes **zero model calls**; it is not an assumption
that missing usage means zero. No model-accuracy result exists.

The root-only diagnosis is `tools/diagnostics/luna_semantic_startup_root.py`;
it exports fixed booleans/status codes, not private log text. Old v1 sources,
receipt and failed unit remain immutable and must never be retried.

Narrow sequential repair plan:

1. A fresh GPT-6 Sol implements versioned control-plane bus access and finite
   pre-inference failure reporting. Preserve subscription child isolation and
   all resource/model/call bounds; no live actions by the agent.
2. Root independently reviews/tests the new version, including missing/wrong
   bus handling, explicit zero-admission failures and separate failure cleanup.
3. Run an actual detached **zero-inference containment smoke** with identical
   sanitized environment and kernel-policy checks. The previous rehearsal
   verified extraction wiring but did not exercise the systemd user-bus query.
4. Only after that passes may a new immutable receipt bind a justified fresh
   29-judgment diagnostic. No retry of v1, cap increase, LME or production work.

## Versioned startup repair accepted offline

Fresh Sol adapter `luna_semantic_probe_adapter_v2.py` has SHA256
`e6ac48313364367756afbbaa7720e9d65047b12590237d5974f8324b188bf504`.
Root reviewed the complete adapter and independently ran 19 controls (12 root,
seven Sol). It passes a fixed, ownership-checked user-bus environment only to
the exact service-status query. The unchanged subscription sanitizer excludes
both bus variables and API keys from model children. Missing/wrong-owner/
symlink/non-socket buses, altered source, mode switching and accidental inference
on the smoke route are rejected. Exception paths restore the scoped query
adapter; startup zero-usage claims require the campaign directory never existed.

Original v1 modules remain byte-identical. New sidecars additionally bind mode,
adapter source, entry action and base receipt; each launch is still one-shot.
The v2 reader verifies the complete original source inventory, admission and
launch marker before special smoke/startup outcomes, and separately reports
failed-unit cleanup without calling it a successful diagnostic.

Root also tested its private offline verdict-replay helper against the complete
synthetic journal and mutations of grounding trace, path validity and aggregate
pass state. It compares all deterministic canary fields, excluding synthetic
replay token values; actual usage must separately match the validated terminal.

Actual Afrodite containment smoke passed under the identical sanitized service
environment, v1 kernel-policy checks and v2 fixed user-bus query. Independent
reader confirmed effective policy, zero model calls/admitted turns/tokens and
complete service/cgroup cleanup. It is not a semantic or benchmark pass.
Smoke root `/home/atta/.hymem-luna-semantic-probe-v6d_3fr8`;
base receipt `70aa322d5fb4a4deaef283ecbe4911d8acd4fe3ae8b80a3e64a1d333bf943f83`;
smoke-mode sidecar `c166fef6bd325e549a57bc030e1d3777a4f7f5bd65567e0c0eb7d26df8e08e12`.
This smoke is immutable and must not be changed to inference mode.

## V2 bounded diagnostic — fresh identity

Prepared root `/home/atta/.hymem-luna-semantic-probe-30jkk7eg`,
unit `hymem-luna-semantic-probe-30jkk7eg.service`.
Base receipt SHA256:
`7848e4368554d12ddd5862550c51c0f932bde1170e8b873b765eda6899b33bc8`.
Inference-only sidecar SHA256:
`c162306300670e11dcbe47be1ccd6a4abef2c9833c09292846844fdc344fec2a`.
Adapter SHA256:
`e6ac48313364367756afbbaa7720e9d65047b12590237d5974f8324b188bf504`.
The original 14-source map and 510-file candidate inventory remain identical.
No changes to the 29-call/500,000 observed-token/1,800-second limits or model.

V2 metadata-only polling, after verifying the local adapter hash:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  'python3 -I -B - observe --root /home/atta/.hymem-luna-semantic-probe-30jkk7eg --receipt-sha256 7848e4368554d12ddd5862550c51c0f932bde1170e8b873b765eda6899b33bc8 --adapter-sha256 e6ac48313364367756afbbaa7720e9d65047b12590237d5974f8324b188bf504 --adapter-receipt-sha256 c162306300670e11dcbe47be1ccd6a4abef2c9833c09292846844fdc344fec2a' \
  < tools/diagnostics/luna_semantic_probe_adapter_v2.py
```

Do not launch from a polling turn. Require completed-and-clean metadata and root
private response replay for acceptance. `semantic_fix_accepted` remains false
in the reader because independent replay is a separate gate.

V2 one-shot launch returned zero. First root poll verified a genuinely active
running service, three completed controls / three settled new turns and 15,191
known tokens, with recent progress and adequate disk/memory. This settled
snapshot excludes in-flight work. No semantic accuracy or cleanup claim yet.
The monitor is active and bound only to this V2 root/sidecar. The preceding V1
failure and smoke remain stopped and immutable.

## V2 terminal result

Root's terminal metadata reader verified 25 finished units in 158.853 seconds:
24 controls plus the retained-response hybrid. Known usage is complete and
reconciled: **27 new Luna turns / 141,055 tokens**. All 25 owned process groups
are independently absent; the failed service is stopped, cgroup empty and
resource policy intact. Failure cleanup is verified separately from success.
The monitor is paused; no further model call or reroll is launched.

Controls: **20 passed**, one false-support unit/claim, two malformed units and
one missed-recovery unit. Two false-rejection claims fall within malformed-unit
outcomes; categories and claim counts are not interchangeable. The hybrid replay
consumed all eight retained ordinary responses but only two new judgments; it
did not reach its expected correction recheck. Thus `core_completed=false`,
`all_semantic_checks_passed=false`, and `completed_and_clean=false` despite all
units reaching a recorded outcome. There was no transport first-failure or
budget stop. This does not establish the original transport incident is fixed.

Root private offline response replay is next. Its reviewed helper SHA256 is
`38806d0c42c60c1af89a581d96be5657ad5a4c2a3517fd53c28f354f9b39e6ea`.
No semantic acceptance, new LME pilot or production deployment is justified by
these results. Preserve both positive and negative control outcomes.

Root privately replayed **all 27 returned judgments**, all 24 controls and the
hybrid through the exact candidate: exact canonical request matches and all
deterministic semantic/canary fields agree. No new inference or writes.
Private-result SHA256:
`7c05fb606399c16a52918408089c2fbf4804213cc9b07d7a4cf40577fef85620`.
Independent cleanup remains supplied by the terminal reader, not replay.

Finite failure localization (no raw output exported):

- Control 10, two conversation records: `evidence:quote_missing`; both supported
  claims were atomically rejected because the quoted evidence was not exact.
- Control 13, preference-to-use correction: the judge chose `unsupported`
  instead of proposing the entailed predicate-only replacement.
- Control 18, acknowledgment of a proposal: the judge falsely supported actual
  use. This is a real false-acceptance risk, not a parser or scoring failure.
- Control 19, out-of-prefix context: `evidence:context_scope`; the hard guard
  correctly blocked invalid context evidence, though the malformed verdict
  fails the clean negative-control objective.
- Hybrid: table claim supported; prose claim `uncertain`, with no proposed
  predicate. No correction recheck was justified or silently added.

A fresh Sol is reviewing the decision-policy specification read-only before
any additional implementation: the prompt specifies replacement format but
does not explicitly prioritize a uniquely supported predicate-only correction
over rejection of the original predicate. It also says not to substitute a
related predicate, which requires disambiguation from allowed correction.
These are hypotheses to review, not proof that another prompt or run will pass.
No additional live diagnostic is authorized by the paused monitor.

Follow-on: the user-authorized, separately versioned decision-policy repair and
fresh fixed-schedule diagnostic are recorded in
`2026-09-29-grounding-decision-policy-v2.md`. That plan contains the new sources,
independent gates and exact monitor pins. This document's failed v1-grounding
run and all of its outcomes/receipts remain immutable; it was not resumed.
