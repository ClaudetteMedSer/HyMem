# Diagnose incomplete-turn failure without weakening the benchmark

## Preserved live evidence

At the 2026-09-29 22:53 UTC check, source/receipt-verified pilot `a5olbwr6`
was terminal: 0/4 scored, four failed, 1,069 admitted turns, 7,608,931 known
tokens, incomplete usage, zero reserved/in-flight turns and independent cleanup
verified. First fault: `incomplete_turn_or_usage`, phase `run`, RPC
`turn/events`, process 16/request 7, six retired threads, zero warm pending
events. Task peak 132/256, zero denials, no resource fault. Canary structural
validation passed, gold mismatch. Final question indexing health is unknown.
Preserve this attempt unchanged; never resume or relaunch it.

Root's reviewed helper `luna_incomplete_turn_metadata_root.py` verified the
same receipt, source and cleanup before projecting finite labels/numbers only.
Question `q-0001` has incomplete usage; the other three and canary have complete
settled usage. The 47,699-byte private log contains no detailed first-fault
trace. It has 76 `chunk_grounding.call_failure` labels, but those labels alone
do not establish their cause or the cause of this first transport fault.
No raw logs, private question text, stores or credentials left Afrodite.

The frozen base parser uses a single final assertion for four distinct states:
missing turn completion (including its 4,096-event ceiling), missing final
message, absent usage, or zero usage. The warm observer drops accepted-event
state. `known_usage=false` means the parser did not return its tuple, not proof
that no positive usage update arrived. Its queue count excludes the reader/OS
queue. No event trace is retained by this transport. Root's offline controls
exercise all four states against the exact frozen function and establish the
diagnostic collapse. A late-usage sequence also rejects, but neither that order
in this live run nor its protocol validity is established.

[Official App Server documentation](https://learn.chatgpt.com/docs/app-server)
describes completion, item and usage notifications but does not establish the
specific ordering in this failed turn. Do not invent it or assert a provider,
quota, event-limit, or model-output cause from this metadata.

## Ordered repair and verification

1. Separate GPT-6 Sol implements a new versioned warm observer (preserving all
   old sources). Retain bounded text-free turn state before failure cleanup:
   number of consumed events, completion seen, final seen/count, usage update
   count and absent/zero/positive state, last finite event family. Bind state
   to the current fresh thread/turn; reset per invocation. Preserve original
   parser behavior, exact exception, all limits, event routing, usage accounting,
   isolation and cleanup. Do not drain extra events, retry, accept missing data,
   promote observed partial usage to complete usage, or record raw text/IDs.
2. Root independently reviews and verifies with actual frozen parser sequences:
   all four failure arms, normal success, event ceiling, state reset, wrong
   identity, duplicate/regressing/invalid usage, forbidden tool/model events,
   ordinary and staged paths, failure before cleanup and private-text canaries.
   Require proof of unchanged provider dispatch and acceptance decisions.
3. Only after acceptance, a separate Sol integrates exact new sources into
   versioned staged transport, diagnostic runner/launcher/reader and source-only
   preparation helpers. Reader must validate finite state and consistency,
   retain unknown values honestly, reject fabricated successful completion,
   preserve old receipts, denominator/accounting and all policy gates. Root
   independently tests integration and no-inference actual host preflight.
4. Only then consider one fresh same-four bounded diagnostic pilot with a new
   receipt. Its explicit purpose is to distinguish the previously collapsed
   runtime states while exercising unchanged behavior, not to claim the live
   failure repaired or try randomly for a pass. Record that purpose, exact
   sources, limits and monitor before launch. If no concrete underlying defect
   is exposed, do not endlessly reroll. External auth/quota blockers require
   pausing for user direction; no bypass or fallback.

The current run remains stopped. No full500 readiness or launch is justified.

## Observer acceptance

Root independently reviewed `benchmarks/codex_subscription_warm_v4.py`, SHA256
`43611c5b7c9b2242f8216daf1cb274c7b1f82c7c633abd11830bf5759fdb4138`,
and passed 494 offline controls including base, concurrent, warm v2/v3/v4,
staged v1, root JSON-shape/privacy/source and actual RPC-queue tests. Root caught
malformed JSON membership exceptions and loss of known reasoning-event family
labels during review; Sol corrected both before acceptance. The final observer
pins and delegates to the unchanged v3/v2/base sources. It adds no event reads,
provider turns, retries or accounting/acceptance changes. Failure state is copied
before cleanup and retained only for the incomplete-turn code in the run phase.

All four collapsed states are distinguishable in invented offline sequences;
the actual historical live state is still unknown. This accepts step 1 only.
A different Sol may now implement the source-bound versioned integration in
step 3; root must review and verify it before host preparation or inference.

## Integration acceptance

A separate GPT-6 Sol implemented staged transport v2, runner/launcher/bundle/
host-preflight v4 and reader v5. Root independently reviewed each delta against
its frozen predecessor and passed 695 offline controls. These include both real
ordinary/staged dispatch paths, the actual nested resource-observing budget,
missing and contradictory metadata, privacy/type controls, source identities,
one-shot launch/cleanup regressions and the actual remote archive-validation
prefix. Root caught a stale nine-file count in that embedded prefix; Sol fixed
it and added a real-archive regression before acceptance. Root independently
imported the assembled sources and verified the 514-file candidate inventory,
staged candidate binding and both warm v4/v3 origins with zero inference.

Accepted integration SHA256 identities:

- Staged v2: `73452c357f62051806371ac69e05b8fb7152a5fe6f8cee070dd685d27d6e5c11`.
- Runner v4: `a3811cbf2644d86a8ae87cff23a01c955a4df8ea4ecc4ec130bb77a473f357df`.
- Launcher v4: `ce30af975f5fc782aadecc770e5b8b52a471c396cde47021c88d55038186e34a`.
- Reader v5: `32caa15f7cdf3748ee4531c85be131c16229649f47618a70bd55df3734fe5207`.
- Bundle v4: `cfe6e4504d6d808717bdd3f02e19eaa362ba9a24530d91d01980a80508c76da5`.
- Host preflight v4: `ac4c06f180a59cca57a48879e800b87509d84f74ec72de1367bf9aebe54c8e8c`.

This accepts the instrumentation integration, not the unresolved historical live
fault. A real no-inference Afrodite preflight and fresh immutable receipt remain
required before the one justified diagnostic run described in step 4. Candidate,
model/auth, prompts, parser acceptance, usage accounting and all run/resource caps
remain unchanged. No historical run is resumed and no production store is touched.

## Fresh diagnostic pilot: x70ca5d2

Real Afrodite preflight passed with zero inference: exact 514 candidate files,
ten code files, frozen dataset and binary verified. The first local upload
invocation was blocked by the sandbox at SSH connection (`Operation not
permitted`); the approved network invocation completed the preflight once.
Receipt preparation then reverified sources/runtime/data and that previous
benchmark units and the cancelled DeepSeek container were stopped. Neither
preflight nor preparation creates inference. This single fresh pilot is justified
to distinguish missing completion/final/usage, zero usage and the event ceiling
in the previously collapsed live failure. It does not assert that instrumentation
fixed the underlying failure. Do not repeat it merely for a better draw.

### Identity and limits

- Private root: `/home/atta/.hymem-lme-diagnostic-preflight-x70ca5d2`.
- Unit: `hymem-luna-lme-diagnostic-preflight-x70ca5d2.service`.
- Receipt SHA256: `bd689672eefb5a0de3e993be4af05194fc6b5837c0fc573b1c03ab7d45dec24d`.
- Runner/launcher/reader/bundle/preflight and staged/warm source hashes are the
  accepted identities above, not the previous pilot's versions.
- Candidate map SHA256: `9e3fe4e495585f63bb560b123fbc0edb6441396ec28ae3f8fa6bfb6ba5b8cfae`.
- Inventory SHA256: `228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf`.
- Dataset SHA256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- Binary SHA256: `167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9`.

GPT-6 Luna low / ChatGPT subscription; first four questions in the same frozen
source order; four workers. Campaign: 8,012 turns / 48,160,000 known tokens /
14,400 seconds. Per question: 2,000 turns / 12,000,000 known tokens / 12,600
seconds; indexing 10,800 seconds. Canary: 12 turns / 160,000 tokens / 600
seconds. Systemd runtime: 14,530 seconds plus 10 seconds stop; 256 tasks,
4 GiB RAM, 200% CPU; no restart, control-group kill, OOM kill. Invocation,
quota, isolation and accounting gates unchanged. No overlapping experiments.

### One-shot dispatch and status

Prepared at 2026-09-29 23:18 UTC. The existing monitor now names this exact
source/root/unit/receipt and limits. Dispatched exactly once at approximately
23:21 UTC. The launcher returned zero, `never_retry=true`, with the expected
receipt. Root's independent hash-verified reader then reported
`checkpoint_running`, 0/4 scored, 25 current tasks, peak 31/256, zero denials and
no resource fault. Usage, canary, final indexing health and answer correctness
are unknown. This verifies startup/policy, not clean completion or a repaired
live fault. The attempt is consumed; never repeat or resume it. The recorded
command is:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --launch-root /home/atta/.hymem-lme-diagnostic-preflight-x70ca5d2 --receipt-sha256 bd689672eefb5a0de3e993be4af05194fc6b5837c0fc573b1c03ab7d45dec24d' \
  < tools/diagnostics/luna_lme_diagnostic_launch_v4.py
```

### Exact metadata-only monitoring

Verify local `tools/diagnostics/luna_lme_diagnostic_progress_v5.py` SHA256
`32caa15f7cdf3748ee4531c85be131c16229649f47618a70bd55df3734fe5207`, then:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --root /home/atta/.hymem-lme-diagnostic-preflight-x70ca5d2 --receipt-sha256 bd689672eefb5a0de3e993be4af05194fc6b5837c0fc573b1c03ab7d45dec24d' \
  < tools/diagnostics/luna_lme_diagnostic_progress_v5.py
```

While active, read-only monitoring only: no further model/provider calls,
source/model/auth/budget changes, production changes, restarts/resumes/rerolls or
overlap. Never export raw logs, benchmark/model text, stores, private rows or
credentials. Retry transient unreadable JSON once. Intermediate canary/usage/
log activity/stage timings are not exposed by this reader; do not invent them.
Unchanged question counts alone are not a stall; unknown/in-flight usage is not
zero. Notify question completions, terminal outcomes and actionable source/
policy/process/OOM/task-denial/disk-under-20-GiB failures; routine checks quiet.

Require `completed_diagnostic_and_clean=true`, all four scored, reconciled
complete usage, zero denials and independent cleanup. Report correctness, canary
gold, strict indexing health and summary degradation separately; do not require
perfect accuracy or tune to answers. This is diagnostic, not canonical/API-
equivalent. On failure preserve the finite first-fault observation, prove the
next defect before assigning a separate Sol, and independently review/test
before any further justified run. No blind rerolls, relaxed limits, model/auth
changes, quota bypass or API fallback. External quota/auth/data blockers need
user direction. A clean pilot unlocks only the separately versioned, verified
full500 preparation gates in `2026-09-29-luna-authorized-continuation.md`.

Detached server execution and its cleanup bound survive laptop closure. Local
polling, repair and subsequent launch require this app/computer to be available.

## Terminal failure: x70ca5d2 (2026-09-30 00:53 UTC check)

Root's hash-verified reader reports terminal failure, 0/4 scored, four failed,
3,710 admitted turns and 27,229,035 known tokens; usage is incomplete. Independent
process/cgroup cleanup is verified. Task peak was 140/256 with zero denials and
no resource fault. Canary structural validation passed, gold mismatch. Final
question indexing health remains unknown; no answer accuracy can be inferred
from zero scored questions. The attempt is consumed and must not be resumed or
repeated. No benchmark is running.

The matched first-fault notification is `responseStreamDisconnected`, HTTP 403,
`will_retry=true`, during `run` / `turn/events`. The recorded generic stop is
`fixed_other`; terminal reconciliation reports `final_accounting_or_dataset_failure`.
The failed turn did not return known complete usage. This is distinct from the
previous pilot's incomplete-turn assertion, and does not retrospectively explain
that failure. The provider's retry flag is not proof that recovery occurred.

A separate GPT-6 Sol read-only review confirmed that the frozen parser rejects
all `error` notifications, including retryable ones. This fail-closed behavior
was explicitly preserved by the accepted observability work, not introduced by
the latest observer. No adapter defect is established by the retained evidence.
The HTTP status alone does not establish quota exhaustion, expired credentials,
geographic policy, or any other specific upstream reason.

Root independently passed 303 focused transport/observer controls and replayed
an invented matched `responseStreamDisconnected` / HTTP 403 / `willRetry=true`
event through the hash-verified frozen parser. It rejects with the same fixed
error and retains only finite metadata. This confirms the code path, not the
unavailable upstream reason or whether a provider retry would have succeeded.

Next diagnostic, before deciding whether user direction is required:

1. Use a separate Sol to prepare a finite metadata-only account reader, bound
   to this stopped run and its accepted sources/binary. Permit only initialize,
   initialized, account/read with refreshToken=false and account/rateLimits/read.
   No thread or turn creation, inference, credential-file access, explicit token
   refresh, configuration/auth changes or benchmark rerun. Export only fixed
   status codes, finite plan/auth enums, quota numbers and owned-process cleanup.
2. Root independently reviews/tests source binding, the RPC allowlist, privacy,
   deadlines and cleanup before one actual zero-inference Afrodite check.
3. Preserve the result and pause for user direction if external access remains
   unresolved. Successful account metadata alone will not prove stream access
   restored or justify a blind pilot reroll. No full500 launch is justified.

### Zero-inference account check acceptance

A separate Sol prepared `tools/diagnostics/luna_subscription_access_metadata_v1.py`,
SHA256 `3fd553772d7865a10d2505ef2a7b4ea2d9f9a8a28ab24b64c45feb9a31cfa320`.
Root reviewed its complete source and independently passed its 14 offline
controls. Root also exercised the actual AST-extracted pinned transport: strict
configuration, all disabled-capability flags, sanitized environment, owned
process group, stderr suppression, real quota-notification rejection and bounded
line reads. Review corrections were made before acceptance; no earlier draft
was executed on Afrodite. The helper embeds the hash-verified reader, validates
the stopped run and sources before starting the pinned binary, permits only the
four metadata messages above, and verifies its own process-group cleanup.

One actual account-metadata check is now being dispatched using:

```sh
/opt/anaconda3/bin/python3.13 -I -B tools/diagnostics/luna_subscription_access_metadata_v1.py --local-ssh
```

The wrapper supplies the exact stopped-root/receipt, SSH BatchMode/10-second
connect/one attempt, a 45-second local bound and 20-second remote session deadline
plus bounded cleanup. This is not a benchmark launch or a model request. Do not
repeat automatically if the dispatch or result is ambiguous.

### Account result and blocked continuation

The single actual check completed with `metadata_read`, source/stopped-run
verification true and owned-process cleanup true. Afrodite reports ChatGPT
authentication, Pro plan, and one reported 10,080-minute quota window with 33%
remaining, above the unchanged 25% floor. Reset timestamp: 1791046713 (Unix UTC).
No inference, thread creation, explicit token refresh or benchmark dispatch was
performed. Account identifiers, credential files and raw error text were not
exported.

This snapshot does not indicate quota exhaustion or absent subscription login.
It does not test response streaming, rule out another upstream restriction, prove
the historical 403 cause, or establish recovery. The provider reason is not
retained in the available finite metadata. No repairable adapter defect has been
proved, so a new pilot or full500 run is not justified under the accepted gates.
Pause `monitor-luna-lme-pilot` for user direction on the upstream access failure;
keep all benchmarks stopped and all evidence/receipts intact. Do not blindly
reroll, ignore the 403/retry notification, switch auth/model or bypass controls.

## User-directed resolution and repair (September 30)

The user now requests: “Resolve the questions, then fix them.” This authorizes
investigating whether a retry-progress notification was incorrectly treated as
terminal and implementing proved defects. The previous monitors remain paused
while this work is supervised. The failed x70ca5d2 run remains immutable/stopped;
there is no authority to resume it, bypass access restrictions or erase its costs.

Plan, in order:

1. Establish the exact protocol contract, not merely that the adapter follows
   its old policy. Inspect the accepted binary's public schemas and available
   official-client code; where necessary, use a controlled localhost mock
   provider with invented data and no real credentials or external inference
   to observe the exact runtime's retry/terminal events. Separate Sol agents
   independently inspect the transport/accounting path and official-client
   evidence. Do not assume the retry flag guarantees eventual success.
2. Determine what historical 403 evidence is actually retained. Read only the
   bound run's artifacts on Afrodite and export finite labels/counts; never raw
   errors, benchmark text, credentials or unrelated account history. If the
   reason was discarded, record that limit rather than inventing a cause.
3. For each concrete defect, have a separate Sol implement a narrow versioned
   repair. Root independently reproduces and reviews it before the next repair.
   A valid recovery change must preserve turn identity, original deadline/event
   bounds, output/usage validation, quota/access policy and one admitted turn.
   It must never resubmit a prompt or create a new turn silently. Test progress
   followed by success, true terminal failures, repeated errors/deadline/event
   bounds, partial output, malformed/foreign notifications, accounting and
   recursive cleanup through real ordinary and staged paths.
4. If needed, add privacy-safe, bounded failure evidence so a future server
   rejection retains a diagnosable reason privately on Afrodite. Root verifies
   that public status cannot leak raw messages or falsely label success.
5. Only after accepted offline/runtime-mock verification, prepare a source-bound
   bounded live diagnostic with invented text to test current subscription
   access/recovery. Record its exact purpose, source identity, request/token/time
   caps and cleanup before a one-shot dispatch. A success is evidence of access
   at that time, not a retrospective explanation of the historical 403. No
   auth/model switching, API fallback, purchases, quota bypass or production edits.
6. Consider a fresh same-four diagnostic benchmark only if accepted repairs and
   measured access justify it. Preserve all existing per-question/pilot/resource
   caps and require a new immutable receipt and monitor binding before launch.
   Full500 still requires the clean-pilot and separately verified full-run gates.

### Independent protocol and retention evidence

Root independently read the official desktop client's packaged source in
`/Applications/ChatGPT.app/Contents/Resources/app.asar`. Renderer member
`webview/assets/app-shared-36eae88777f2.js`, SHA256
`bc90bb198f29b62bd745aa99dbfae6a81719d9739c22811a12a810f9813d6c43`,
maps `willRetry=true` to a reconnecting `stream-error`, while false becomes
`system-error`. Its status index excludes retrying errors from `latestTurnError`.
The main-process member `.vite/build/main-C5425b_s.js`, SHA256
`91a68c5f690e60033152a34cf0bf5c4234caeb9ddb47586fd3ef5b017bb29d64`,
also treats completion status separately. The bundled CLI is 0.158.0-alpha.2.1,
not the exact remote 0.158.0; this is corroboration, not proof of exact runtime
ordering or successful recovery. A separate Sol supplied an independent review.

Two root metadata-only on-host inspections reverified x70ca5d2's source/receipt
and cleanup. Its 370,089-byte private run log has zero `httpStatusCode`,
`responseStreamDisconnected`, `willRetry`, HTTP-403, forbidden, unauthorized or
structured error-event markers. All sixteen textual `403` occurrences are inside
hex/numeric strings, not standalone statuses. Launch stderr is empty. The
accepted adapter discards app-server stderr and projects away the error message;
the threads are ephemeral. No historical upstream reason is established by the
retained artifacts, and no raw private text left Afrodite.

The separate contract review confirms that unknown usage properly stops the
shared campaign. Changing that global stop is not a safe fix for this event.
The candidate repair is instead to wait for a validated terminal outcome of the
already-admitted turn when the exact protocol says the server is still retrying,
without another prompt/turn or looser final-output/usage requirements. Exact
runtime mock verification remains required before accepting that repair.

### Exact-runtime offline mock: predeclared verification

Root reviewed the separate Sol mock, SHA256
`80ddc981d4ec5839ffaeccc498f9628b5875226b6119cc3836b586c9f271c4b9`,
and independently passed 12 offline tests (one desktop loopback test skipped by
the sandbox). It tests the exact pinned 0.158.0 binary against an invented local
Responses service: ordinary completion, disconnect followed by recovery, HTTP403
followed by an available successful response, and repeated disconnect. One
turn/start per case, 28-second case deadline, at most four serviced mock requests,
finite output only. A negative result is evidence, not permission to change auth.

The first user-systemd IPAddressDeny containment probe did not prove outbound
denial; it is not used as a security boundary. Root instead verified an existing
Docker image under `--network none`: localhost works and an outbound documentation
IP has no route. The test uses a fresh non-root, read-only, capability-free
container with no host account/config mounts, sanitized environment, a private
tmpfs and only the pinned Codex binary mounted read-only. It cannot contact the
subscription service. Root wrapper `luna_retry_mock_host_verify_root.py` checks
actual container policy before starting, bounds execution at 145 seconds and
removes/verifies absence of its uniquely owned container independently. No
benchmark, production process or credential is accessed. This is offline protocol
verification, not a change of the real benchmark's provider/authentication.

The first dispatch command was blocked locally by its source-hash check: Sol had
made final usage-validation hardening edits after giving the earlier hash. No SSH
or container launch occurred. Root re-reviewed the delta and bound this run to
the final frozen hash above; no diagnostic result is implied by that local block.

The isolated v1 mock then ran once. The baseline delivered one mock HTTP response,
one final message and positive usage with matching identities, but the diagnostic
observer returned `unverified` before recording turn completion. It correctly
stopped before the three fault cases. Container network policy and independent
cleanup both verified. This is an unresolved mock-observer boundary, not evidence
of a live subscription or model failure. Preserve v1. The same diagnostic Sol is
preparing v2 to identify the rejected event using only the exact public schema's
finite notification names; root must review before another offline execution.

Root reviewed mock v2, SHA256
`58232a53c9f2588a91e3a15e2a53827116d6f425a2633d017dd38cbd7ddf4788`.
It only adds finite rejected-event attribution against the exact public schema;
it does not accept previously rejected events. The next network-none execution
has the same invented fixtures and bounds, and is justified specifically to
identify the observer boundary that stopped v1. No live model calls are involved.

V2 identified the exact boundary: `account/rateLimits/updated` after the final
item and positive usage. No retry cases ran. Root verified both cleanup layers
and network policy. The mock-only observer lacked a documented notification that
the actual benchmark already handles. V3 will validate that sparse schema without
altering real benchmark quota logic. New independent mock responses will also
have distinct response/item IDs per HTTP attempt, avoiding artificial duplicate
IDs in the recovery fixtures. This does not establish a live quota problem.

Root independently passed 406 offline transport tests including 52 additional
root-owned v5 controls: an error queued before turn/start acknowledgement,
old-parser abort versus new same-turn completion, unchanged absolute timeout,
explicit policy/auth/quota rejection, malformed shapes, lifecycle/tool/final/
usage gates, and original/staged transport regressions. The inactive v5 SHA256 is
`2df1ead8f6f1cee1f138075aa77d78c61ed28959195a7634ce59a0df00290702`.
Exact-runtime retry verification remains pending; no new benchmark was launched.

Root reviewed mock v3, SHA256
`2c8a60733fceb86a7afbb06928bf1ad072b81503e8faa35c291b524081127101`:
the sparse quota notification is now schema-validated in the mock observer only,
and mock HTTP attempts use unique IDs. The fake-session regression verifies that
this notification can precede completion. All three mock versions' offline
controls pass (43 passed; three local bind tests skipped). The justified next
network-none execution will exercise the original four finite cases with the
same limits; no account or external inference is available in that container.

### Retry-progress repair accepted after exact-runtime proof

The v3 mock ran all four cases once against the pinned 0.158.0 binary:

- Baseline: one mock HTTP request, one completed turn, final output and positive usage.
- Disconnect then recovery: two HTTP requests, one matched
  `responseStreamDisconnected / willRetry=true`, then the same turn completed
  with final output and positive usage.
- HTTP403 then recovery: two HTTP requests, one matched
  `responseStreamDisconnected / httpStatusCode=403 / willRetry=true`, then the
  same turn completed with final output and positive usage. This reproduces the
  historical notification's exact class/status/retry combination as nonterminal.
- Persistent disconnect: four serviced mock HTTP requests, four retry-progress
  events, no final/usage/completion; the unchanged diagnostic deadline stopped it.

Every case dispatched exactly one turn/start, identities matched, and both the
owned process groups and outer network-none container were independently cleaned
up. No real account or model was contacted. The sparse quota notifications were
mock-only unavailable quota, not evidence about the user's account.

Together with the 406 root-run offline controls, this proves a concrete adapter
defect: rejecting every `error` prematurely ends a turn the server is explicitly
retrying. Root accepts the frozen v5 repair (hash above). It waits only for a
typed, matched, finite retry-progress event on the already-admitted turn; it adds
no turn, prompt retry, quota bypass, deadline extension or output/usage relaxation.
This supersedes the earlier provisional assessment that no adapter defect was
established. It does not prove the historical 403 reason or that that particular
turn would have recovered. The historical reason remains unrecoverable.

Next narrow repair, using a different Sol: preserve bounded first/last matched
error details privately on failed turns, discard on success, and keep public
metadata text-free. Root must independently verify confidentiality, schema,
byte/file bounds, path safety, original accounting and cleanup before integration
or a one-shot invented-text live access check. All benchmarks remain stopped.

The private-capture scope is source-pinned warm v6 only, implemented by a different
Sol after core v5 acceptance. It collects at most first/last matched errors in
memory, byte-limits the two prose fields, discards successful-turn state, and
optionally writes failed-turn records to an existing owned 0700 directory using
0600 no-overwrite files with a cross-process record limit. Public failure
serialization stays unchanged. Sink failure must not replace the original
transport/accounting result. No stderr capture or credential-file access.

After acceptance, separately versioned integration must bind both ordinary and
staged clients to the new transport and opt-in sink while preserving all source,
data, quota, invocation and launch limits. Before any further LME attempt, a
fresh source-bound access diagnostic will use only invented text, GPT-6 Luna low
and the same subscription. It will be limited to at most two admitted turns and
32,000 known tokens, unchanged 120-second invocation bounds, no reroll, and owned
process cleanup. Record its exact receipt, root, runtime bound and command before
dispatch. It must stop immediately on access/quota or accounting failure. This
checks present access, not historical cause or benchmark quality. No live access
diagnostic or new benchmark has yet been launched.

### Private failed-turn evidence repair accepted

Root reviewed the separate Sol's warm v6, SHA256
`98422aa251ca9482a79be5851d48ae54decc17f17b6aa1149bd6b93b618b784b`,
and independently passed 469 transport/staged tests. These include root-owned
controls for JSON-escaped Unicode byte limits, malformed error metadata,
documented optional misalignment shape without retaining its nested prose,
concurrent writers, restrictive umask, directory/slot symlinks and observer/sink
fault isolation. Actual client controls prove discard on success and private
capture before failure cleanup without changing original accounting. Each private
record is at most 20,480 bytes; each existing owned 0700 directory holds at most
16 exclusive 0600 records. Public failure serialization remains unchanged.

The transport is accepted but not yet wired into a live runner. The next separate
Sol implementation is versioned staged/runner/launcher/reader/bundle integration;
root must verify ordinary and staged paths plus actual source-only assembly and
zero-inference host preparation before any invented-text access check. Historical
sources, receipts, candidate, dataset, limits and stopped runs remain untouched.

The small access diagnostic design is now explicit: a separate versioned tool,
fresh source-only private root and immutable receipt; two sequential invented-
text turns at most (ordinary then staged original), shared 32,000-known-token /
300-second campaign gate, original 120-second invocation bounds, a 330-second
user-systemd runtime plus ten-second stop bound, and unchanged 256-task / 4-GiB /
200%-CPU / control-group / no-restart policy. A one-shot marker precedes dispatch;
ambiguous dispatch cannot be retried. Stop on the first access/quota/accounting
failure, preserve private evidence, and verify the unit/cgroup cleanup separately.
The known-token limit is the existing observed-usage stop-before-next-turn gate,
not a hard single-response token ceiling. Report transport access and optional
semantic validity separately. This does not send benchmark or production text,
run LME, or establish the historical 403 reason. Implement only after root accepts
the preceding integration; source hashes and actual root/receipt must be recorded
before launch. No live access call has been dispatched.

Root assembled `/private/tmp/hymem-lme-retry-offline-assembly-v1` from the
unchanged accepted candidate and the integration draft bound to runner SHA256
`963c3e6077055ad3ab769deb816b682f7eac685913c5e5ee905a0a9ab528f4d5`.
The real embedded remote archive checks accept all 527 files (514 candidate,
12 code, one source map); a fresh isolated local import binds staged v3 to warm
v6 and the frozen candidate. The actual Afrodite zero-inference preflight passed
at `/home/atta/.hymem-lme-diagnostic-preflight-xvn648yr`, with the expected binary
and dataset hashes and zero model calls. This root has no launch receipt or
service. Integration acceptance still requires the final frozen regression pass.

### Versioned integration accepted

Root independently reviewed all deltas, tested actual ordinary/staged recovery,
terminal auth and missing-usage paths, verified a valid new receipt end-to-end,
and passed 1,050 offline tests with zero failures after the final source freeze.
The actual source/import/archive and Afrodite preflight above bind these same
bytes. No candidate, semantic prompt, dataset, cap, model/auth or cleanup gate
changed. An interim draft runner-hash edit correctly failed one source-bound
archive test; the extra redundant assertions were withdrawn before this final
complete passing run and all accepted bytes match the verified host root.

Accepted source SHA256s:

- staged v3: `23b2db6d74df5934a61a64ec00e15e8847f7b0cf890a45620df1295ce594c12d`
- runner v5: `963c3e6077055ad3ab769deb816b682f7eac685913c5e5ee905a0a9ab528f4d5`
- launcher v5: `a1921b4c199a5b3124ff8403c5cf31061e74028a17c91b837528478f35ab0f64`
- bundle v5: `2044754f0dc4d7ddd09e057a8e1211a60426e524fc27e490b9c4ab2d846caeb8`
- host preflight v5: `26aaff9f2833c58297a9553d7577f8388f847689a194fee9fefb12084a341fc1`
- reader v6: `6a578b650fc57b5abed199d9dd44ea1df52ac27bcebd0f2f1871f860c4512b90`

Root accepts the integration and now assigns the separately implemented bounded
invented-text access check. All old benchmarks and both monitors remain stopped;
this acceptance is not proof of live provider access or LME completion.

### Accepted two-turn current-access probe: xvn648yr

Root reviewed the separate Sol access implementation and independently passed
98 focused access, transport and integration controls, including 32 root-owned
access controls. Both invented requests are built and schema-validated before
any invocation. A first-turn failure prevents the second; a launch timeout leaves
its one-shot marker consumed. Public failure metadata is filtered by the accepted
transport serializer, and service exit alone never proves recursive cleanup.

Frozen access source SHA256:
`a0bc7ec37e243947418e92725dbd42e66adaea7941e3d25317c6cc4676c4c139`.
The root source-only installer SHA256 is
`351894394ae122c1e420d807f1d67f0938f0c41fac5bdb202b859106b7e4fa35`.
It exclusively installed this tool and the accepted launcher beside the unchanged
source bundle, then passed actual zero-inference preparation on Afrodite.

- Root: `/home/atta/.hymem-lme-diagnostic-preflight-xvn648yr`
- Unit: `hymem-luna-access-check-preflight-xvn648yr.service`
- Receipt SHA256: `06a5d2461d679bb5a4c945995a3ce6e20d84e599e5e2d48cbdfebdd7a2d7dc0e`
- Preparation reported zero model calls; the actual fixture/source/import checks passed.
- Limits remain two sequential turns / 32,000 known tokens / 300 seconds,
  120-second invocation, server 330 seconds plus ten-second stop, no restart,
  256 tasks / 4 GiB / 200% CPU and recursive control-group cleanup.
- Both old benchmark monitors remain paused. No benchmark is launched by this probe.

Historical pre-dispatch status: **dispatch was authorized for one attempt**.
That attempt is now terminal; see the result below. Do not launch it again.
Never repeat this launch if the command is ambiguous. The tool writes its
exclusive attempt marker before asking systemd to start the unit.

Exact command:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  python3 -I -B /home/atta/.hymem-lme-diagnostic-preflight-xvn648yr/access-check-v1.py \
  --launch-root /home/atta/.hymem-lme-diagnostic-preflight-xvn648yr \
  --receipt-sha256 06a5d2461d679bb5a4c945995a3ce6e20d84e599e5e2d48cbdfebdd7a2d7dc0e
```

Use the same command with `--inspect-root` instead of `--launch-root` for
metadata-only reads. The reader requires the source/receipt pins and independent
service/cgroup cleanup before returning a verified access result. No private
provider text is exported.

#### Access probe terminal result: stopped before inference

The one-shot launch command returned zero. The independent source-pinned reader
then verified `access_failed`, `failed_exit`, and recursive cleanup. The first
failure was **`credit_balance_present` at `account/rateLimits/read`**, with
`turn_admitted=false`. The budget admitted zero turns, retained zero known model
tokens, complete usage and no reservations/in-flight work. Both task denials and
OOM-kill counters were zero. No invented prompt, benchmark or production memory
was submitted to a model by this probe. Do not relaunch it.

This is the unchanged conservative account gate, not another observed stream
failure: base `quota_metadata` rejects `credits.hasCredits is True`. It does not
inspect the balance or unlimited flag. The boolean alone does not prove actual
credit consumption, purchased-vs-promotional origin, balance magnitude, or that
included subscription quota is exhausted. Historical 403 remains unexplained.
The exact protocol exposes those credit fields independently; any further
diagnosis must be metadata-only, with no login/refresh, turn or gate relaxation.
Both benchmark monitors remain paused and no new LME pilot is authorized from
this terminal result without resolving the account-policy conflict.

Root then passed the combined final regression command: **1,087 tests, zero
failures** (transport, diagnostic runner/reader, integration and access controls).
A separate Sol is implementing only a finite, zero-inference account metadata
projection to distinguish `hasCredits`, `unlimited`, balance category and quota
windows; it must reuse the accepted read-only RPC fence and cleanup, bind the
failed access receipt, and pass root review before one metadata read. This is not
a credit-gate change or another attempt at either the access probe or LME.

Root accepted that separate Sol helper after reviewing actual pinned-source
origins, strict finite projection and the existing absolute twenty-second
metadata RPC deadline/owned cleanup. Combined metadata/access controls passed:
89 tests plus eight subtests, including 33 independent root credit-projection
controls. Accepted `tools/diagnostics/luna_credit_metadata_v1.py` SHA256:
`b851f087972a939c21003d3203b9f647fd38bd109864e7b06585ecdcbec41eaa`.
It binds the stopped xvn648yr access receipt, verifies its independent cleanup,
then allows only initialize/initialized/account-read without refresh/rate-limit
read. Credit balance is projected only as missing/null/zero/positive/negative/
invalid, never a raw string; account names and bucket IDs never leave Afrodite.

One metadata-only invocation is now authorized; no inference, gate edits or
benchmark dispatch:

```sh
/opt/anaconda3/bin/python3.13 -I -B tools/diagnostics/luna_credit_metadata_v1.py --local-ssh
```

The single metadata read succeeded and verified owned process cleanup. It
reported ChatGPT Pro, one selected weekly bucket with **29% remaining**, reset
timestamp `1791046713`, `spend_control_reached=false`, no reached-limit state,
`has_credits=true`, `unlimited=false`, and a **positive** balance category. No
balance amount, account identifier or raw response was exported. No model call
was made. The unchanged admission gate still reports `credit_balance_present`.

This resolves the immediate access-probe stop: it is the conservative blanket
credit-availability gate, not proven quota exhaustion, a new model failure, or
the old HTTP403's explanation. Available subscription quota is only four
percentage points above the unchanged 25% reserve at this observation. Presence
of credits does not prove credit consumption, and this metadata cannot guarantee
the billing route of a future request. Root has not changed the gate, spent
credits, repeated the access probe, launched LME or changed production. Both
monitors remain paused. Request user direction about the spending boundary
before any policy change or fresh inference experiment; keep existing receipts
and all accepted source bytes immutable.

### Strict subscription-only boundary audit (2026-09-30)

The user resolved the policy choice: “Subscription only for now”, then “Enforce
the boundary then run the LME.” This does not authorize spending existing credits
or replacing the boundary with best-effort quota polling. No benchmark has been
launched under this instruction.

A separate GPT-6 Sol performed a read-only billing-boundary audit; root reviewed
the pinned Codex 0.158.0 generated protocol and the official app-server,
configuration and pricing documentation independently. No verified per-request
or session-level included-only billing selector or atomic subscription-quota
reservation was found for this route:

- `ordinaryUsageAllowed` is observed backend permission, not a request billing
  selector or reservation. Null is unavailable, not permission.
- `supportsLunaReserve` is a capability on a rate-limit read, not a hard no-credit
  spending mode. It was not enabled.
- `allowProviderModelFallback` concerns model substitution, and standard service
  tier concerns speed. Neither establishes included-only billing.
- Local shared budgets reserve local invocation slots, not provider allowance.
  Concurrent account usage and in-flight work prevent a polled percentage or
  the existing 25% reserve from guaranteeing zero credit spend.

The official [personal-plan credit documentation](https://help.openai.com/en/articles/12642688-using-credits-for-flexible-usage-in-chatgpt-personal-plans)
states that included usage is used first and credit balance is used after plan
limits. Automatic reload controls purchases, not a demonstrated prohibition on
spending existing credits. It also distinguishes partner-tool sharing controls;
those must not be assumed to apply to the native Codex route. The
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
did not expose a verified native no-credit switch. This is a scoped negative
finding, not a claim that no account-specific control could exist.

A read-only browser attempt at the official usage-dashboard URL redirected to
the logged-out ChatGPT page. No login, billing setting, credential, account or
authentication change was made. The temporary tab was closed. Account-specific
credit-spending controls therefore remain unverified.

Root verification this turn:

- 164 offline transport, concurrent-budget, access and credit-projection tests
  passed. No provider calls were made by these tests.
- The source-pinned read-only xvn648yr inspector again verified `access_failed`,
  `credit_balance_present` before turn admission, zero turns/known tokens,
  complete overall zero-turn accounting, zero denials/OOMs and independent
  recursive cleanup. This did not repeat the access probe.
- Both `monitor-luna-lme-pilot` and `finish-lme-validation` remain paused.
- Candidate and immutable transport/runner/launcher/reader sources, receipts,
  auth/model, resource policy and production remain unchanged.

Decision: keep the existing fail-closed admission guard. The previous 29%
remaining / positive-credit result is a historical metadata observation, not a
fresh guarantee or authority to infer. A provider-enforced account or request
control demonstrably prohibiting credit use is required before admitting work
on this positive-credit account under the user's strict boundary. Do not ask
again whether credits are allowed; they are not. Do not bypass the guard, rerun
the consumed access receipt, launch LME, or reactivate a monitor merely because
quota resets. Once a suitable control is verified, follow fresh source-bound
access and pilot validation before the full-run readiness gates.

### Superseding authority and accepted existing-credit policy (2026-09-30)

The user explicitly permits existing credit spending and reports automatic
top-up disabled: “Spending existing credits is fine. Automated top up is disabled.
Run the LME.” The strict no-credit launch blocker above is therefore superseded
for fresh runs, not retroactively removed from historical records. No purchase,
reload, credential/model switch, paid API fallback or production change is
authorized. Auto-top-up state is user-attested, not independently enforced by
our runner. The finite experiment limits remain unchanged.

Separate Sol implementation `benchmarks/codex_subscription_warm_v7.py` is accepted
by root at SHA256
`94234eca1daeb542a8d8f92a5b7178ba8b91060f33415f76343bd13e6d5953d9`.
Its explicit policy is
`included_allowance_or_existing_finite_positive_credits_v1`.
It loads a hash-pinned private v6 lineage and transforms only that lineage's
billing admission: a schema-shaped finite positive existing balance may admit
the associated quota bucket below the former 25% subscription reserve. Without
verified finite credits the reserve remains. Credits do not clear another
bucket's floor, malformed/unknown quota, explicit provider spending/limit
denials or an unapproved account plan. Raw balances are never returned.
Original source files and the separately imported original gate are unchanged.

Root independently reviewed the parser transformation and passed 672 offline
controls: 184 policy/base/v6 controls and 488 concurrent, v2-v5 and staged
regressions. New receipt-bound integration is assigned to a different Sol only
after that acceptance. Actual hosted preflight, fresh access receipt/check and
independent cleanup, and fresh diagnostic LME receipt remain required before
launch. All old runs and both monitors remain stopped at this point.

### Fresh existing-credit access check: j41u547s (2026-09-30)

Root accepted the separate Sol integration after independent source/diff review,
762 combined offline tests, source-only bundle/import verification and actual
zero-inference Afrodite preflight. An additional 67 installer privacy/fresh-root
controls passed independently. The assembled inventory remains 514 candidate
files, 13 code files and one source map; no candidate, prompt, model/auth,
individual limit or production changes were made.

Frozen accepted SHA256 identities:

- staged v4: `25844ae48b375ff7ba9b0dac118cd393e8c0d2111f935fb3ea104f12a19f05b6`
- runner v6: `bc055ebe4621ec729723357b7db09f66049b34aaa5d9d67cb09dae16178f680d`
- launcher v6: `5449880912f28aeeee89a4907599b82cab5b56a87df2af5414d70374cb3d15a7`
- progress v7: `2b0e8be7d4e01e9b6e30386eb59b3baa3cffd7c1402cb4ceafba47388d45992e`
- bundle v6: `09fb3892cafa2e944ac80b6d17edbe68c974e4f6f5dadcef7c153acbd0fc7d8a`
- host preflight v6: `87047c21d9fb809316e9ca45b9c59fe8e374860db4345d22833a82712c248558`
- access v2: `14758d9ccb30d95ec95cd427df148d19d12714df611ff6f995ae77842ce59d09`
- source installer: `1448af81b52b4b5df559d94e0edd8909ed91fa2cff02d783c99a10730853edba`

Fresh access root: `/home/atta/.hymem-lme-diagnostic-preflight-j41u547s`.
Unit: `hymem-luna-access-check-preflight-j41u547s.service`.
Receipt SHA256:
`40efe1ef1a84e005887ddcf8be05a9a5b2435859a51d6f2dd94493c6dcba0ee9`.
Preparation succeeded with zero model calls. Explicit receipt policy:
`included_allowance_or_existing_finite_positive_credits_v1`.
Only two invented-text turns are admitted: one ordinary and one structured.
Limits remain two turns / 32,000 known tokens / 300 seconds, 120 seconds per
invocation, server runtime 330 seconds plus ten-second stop, 256 tasks, 4 GiB
RAM and 200% CPU, no restart and recursive control-group cleanup.

Dispatch status: **one launch being attempted now; never repeat an attempted or
ambiguous launch**. Root must independently inspect `access_verified` and cleanup
before preparing a separate fresh LME pilot root. This receipt is not an LME run.

Read-only access inspection command:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite python3 -I -B /home/atta/.hymem-lme-diagnostic-preflight-j41u547s/access-check-v2.py --inspect-root /home/atta/.hymem-lme-diagnostic-preflight-j41u547s --receipt-sha256 40efe1ef1a84e005887ddcf8be05a9a5b2435859a51d6f2dd94493c6dcba0ee9
```

Dispatch result: accepted exactly once, command return code zero. Root's separate
read-only inspection then verified **access_verified**, clean exit, recursive and
client cleanup, ordinary completion, staged completion, acknowledged output
schema and valid staged response. Exactly two turns / 9,855 known tokens, complete
usage, zero reserved/in-flight work, zero task denials/OOMs and no first failure.
This establishes current ordinary/structured access, not that a long benchmark
cannot encounter a future provider failure or that no existing credits were
spent. The access root and receipt are consumed and must never be reused.

### Fresh existing-credit diagnostic LME pilot: 6vp8_6ae

This is the sole new authorized four-question/four-worker diagnostic pilot. It
tests the accepted same-turn retry-progress repair with verified ordinary and
structured access under the explicitly authorized existing-credit policy. The
two-turn access check above passed; a separate fresh source-only Afrodite
preflight and receipt preparation also passed with zero inference. All previous
experiments remain stopped. No source changes occurred after verification.

- Root: `/home/atta/.hymem-lme-diagnostic-preflight-6vp8_6ae`
- Unit: `hymem-luna-lme-diagnostic-preflight-6vp8_6ae.service`
- Launch receipt SHA256: `748f39376c6aed2dba11b3f74ff2afc0a5f2079380302d572669c7e2c05b15d1`
- Reader: `tools/diagnostics/luna_lme_diagnostic_progress_v7.py`
- Reader SHA256: `2b0e8be7d4e01e9b6e30386eb59b3baa3cffd7c1402cb4ceafba47388d45992e`
- Runner v6 SHA256: `bc055ebe4621ec729723357b7db09f66049b34aaa5d9d67cb09dae16178f680d`
- Launcher v6 SHA256: `5449880912f28aeeee89a4907599b82cab5b56a87df2af5414d70374cb3d15a7`
- Billing policy: `included_allowance_or_existing_finite_positive_credits_v1`

Unchanged caps: GPT-6 Luna low / ChatGPT subscription, four workers; campaign
8,012 turns / 48,160,000 known tokens / 14,400 seconds; each question 2,000 turns /
12,000,000 known tokens / 12,600 seconds; indexing 10,800 seconds; canary 12 turns /
160,000 known tokens / 600 seconds; invocation 120 seconds; server 14,530 seconds
plus ten-second stop; 256 tasks / 4 GiB RAM / 200% CPU, no restart and recursive
control-group cleanup. Existing credits may be spent; no purchase, enabling
reload, model/auth switch, API fallback or production change is authorized. The
user attests top-up is off; the experiment does not independently enforce it.

Dispatch status: **monitor updated; one launch being attempted now**.
Never repeat an attempted or ambiguous launch. Root must verify startup through
the source-pinned reader, not from command success alone.

Verify reader SHA256 above, then use this exact metadata-only command:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite python3 -I -B - --root /home/atta/.hymem-lme-diagnostic-preflight-6vp8_6ae --receipt-sha256 748f39376c6aed2dba11b3f74ff2afc0a5f2079380302d572669c7e2c05b15d1 < /Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics/luna_lme_diagnostic_progress_v7.py
```

While active: read-only polling, no overlapping experiment or code/model/auth/
budget/production changes, restart/resume/reroll, raw logs/text/stores/private rows
or credential export. The reader does not expose intermediate per-turn usage,
canary, log activity or stage timing; these remain unknown. Notify each completed
question and actionable metadata/source/policy/process/OOM/task-denial or
disk-under-20-GiB issue; keep routine progress quiet. Retry a transient unreadable
snapshot once read-only. An unchanged scored count does not prove a stall.

Require `completed_diagnostic_and_clean=true`, all four scored, reconciled complete
usage, zero resource denials and independently verified cleanup. Accuracy, canary
gold match, strict indexing health and summary degradation remain separate measured
outcomes. After clean completion, proceed to the separately versioned full-500
readiness/launch gates in the authorized continuation plan. Never label this
pilot canonical or API-equivalent. On failure retain evidence, verify cleanup,
prove a concrete defect before any separate Sol repair and root verification;
no blind rerolls or automatic full restart/resume. External auth/quota/access
blocks require user direction without fallback or purchases.

Dispatch completed exactly once at approximately **2026-09-30 10:24 UTC**, command
return code zero. Root then verified reader SHA256 and independently inspected
the live receipt/runtime: `checkpoint_running`, 0/4 scored, task current 25,
peak 31/256, zero denials and no reported resource fault. Canary, final health,
correctness and usage are not yet exposed and remain unknown; this is startup
verification, not completed benchmark verification. `monitor-luna-lme-pilot` is
ACTIVE every ten minutes with the exact new reader/root/receipt and existing-credit
authority; `finish-lme-validation` remains PAUSED. No additional model calls or
experiment changes are allowed during active polling. Never repeat this dispatch.

### Terminal event-ceiling failure: 6vp8_6ae (10:34 UTC check)

Root's hash-verified reader reports `terminal_incomplete_or_unclean`: 0/4 scored,
four failed, 190 admitted turns and 1,384,992 known tokens with incomplete usage.
Independent runtime/control-group cleanup is verified; task peak127/256, zero
denials and no resource fault. Canary structural validity passed, gold mismatch.
No final indexing health or answer accuracy can be inferred. This run is stopped
and its dispatch/receipt are consumed; no benchmark is active.

First fault is `incomplete_turn_or_usage`, run/turn-events, process3/request9.
The retained observation is concrete: 4,096 consumed events, no completed turn,
no final item, no usage update, last event `item_agentMessage_delta`, queue0 and
eight retired threads. Unlike the historical HTTP403, this directly identifies
the adapter's event ceiling. It does not establish how many output characters
were received, whether the model would finish, or complete billed usage.

The frozen parser counts each text fragment toward the same 4,096-event limit;
it validates but does not concatenate deltas, relying on authoritative completed
items. The [official App Server documentation](https://learn.chatgpt.com/docs/app-server)
documents optional exact-method notification suppression and authoritative
completed items, not a 4,096-fragment protocol maximum. Root is investigating a
fragmentation-sensitive transport defect, not assuming that output was valid.

Ordered investigation and possible repair:

1. Separate Sol read-only audit plus root offline reproduction compares identical
   invented final output/usage under coarse and >4,096 tiny-fragment streams.
   Preserve original deadline, output, event and budget limits; do not simply
   raise limits or silently drop arbitrary messages.
2. A separate Sol implements a bounded invented-data localhost mock diagnostic
   to establish exact pinned0.158.0 notification opt-out behavior. Root reviews
   and tests it before execution in a network-none disposable container with
   only the binary mounted, no real HOME/auth/data. Verify final/lifecycle/usage
   and errors remain visible; no real provider inference or account access.
3. Only after a concrete defect and safe remedy are verified, assign a separate
   Sol a new versioned narrow transport repair. Root independently checks
   ordinary/staged dispatch, malformed/foreign events, final/usage/accounting,
   budget/deadline/cleanup and frozen-source invariants before integration.
4. A different Sol then wires fresh source-bound runner/launcher/reader/bundle
   versions, if justified. Root independently verifies those plus no-inference
   host preflight before one fresh bounded same-four pilot and new monitor.
   No blind reroll, raised spending/invocation/resource caps, weakened quality
   acceptance, auth/model switch, purchases, production or full500 launch now.

Root independently reproduced the fragmentation defect through the actual warm7
RPC queue/parser in `tests/test_luna_fragmentation_root.py`: the same invented
5,000-character final answer with identical positive usage/completion succeeds
as one delta, but fails as 5,000 one-character deltas, producing the exact live
4,096-event observation above. All four boundary/reproduction controls passed.
This proves a local fragmentation-sensitive defect; it does not prove that the
historical unfinished model output would be valid or eventually finish. Proposed
remedy is exact server-side opt-out of `item/agentMessage/delta` only, with no
local swallowing of events, event-cap increase or quality-gate relaxation. Exact
pinned runtime behavior and preserved final/usage/error events remain to verify.

Root accepted the separate Sol standalone mock at SHA256
`916fd3c5ebacaff2e03b574de5f5f61b0d4886256f8952ca5447f07b37464812`
after full source review and 37 independently rerun controls (one local loopback
bind skipped by the sandbox). Root review corrected the draft's final-item field
and absolute-deadline handling before acceptance; no draft was executed remotely.
Root host wrapper `luna_delta_mock_host_verify_root.py` SHA256
`09cb89093a5922067ac72b7e16fc2698c025b76e18e377c7864fb7ede6550a19`
reuses the exact hash-verified prior network-none container boundary, with a new
source/schema and a finite metadata-only output projection. Original files stay
unchanged. At most three local fake HTTP responses, one turn per case, 32 seconds
per case, 6,000 mock-observer events to observe the 5,000-fragment fixture; this
does not change the benchmark's 4,096-event limit. Outer execution is bounded at
145 seconds, with independent uniquely owned container cleanup. Only the pinned
binary is mounted; fresh HOME/config, invented fake key and no account data or
external route. The third case tests visible HTTP400 error with opt-out.

This accepted zero-inference runtime diagnostic is being dispatched once using:

```sh
/opt/anaconda3/bin/python3.13 -I -B tools/diagnostics/luna_delta_mock_host_verify_root.py --source-sha256 916fd3c5ebacaff2e03b574de5f5f61b0d4886256f8952ca5447f07b37464812
```

The exact-runtime diagnostic completed once with `verified=true`. Baseline:
5,000 delta notifications, one matching final answer, one positive usage update
and completed turn. Opt-out: zero deltas, identical final-answer digest and usage
digest, same item-start/item-complete counts (two each) and completed turn. The
third opt-out case still received its matched error. Exactly three local mock
HTTP requests, one turn per case; no external provider/account calls. All owned
process groups and the outer network-none container were independently cleaned
up. Do not repeat this accepted diagnostic.

A new separate Sol is now implementing warm transport v8 only: hash-pinned v7
lineage, exact one-method initialize opt-out, immediate fail-closed detection if
that method still arrives, no local discards or parser/limit/billing changes.
Root must independently review and verify it before another agent integrates
new immutable staged/runner/launcher/reader/bundle sources. No new paid benchmark
or access probe is running; full500 remains gated on a clean diagnostic pilot.

Root accepted warm transport v8 at SHA256
`0a55d44053349eb90511a597dae20f295c19bb53c734a78b5db3c41197343c21`
after full source review and 243 independently rerun offline controls, including
root-owned final/usage, 5,000-character nonfragmented answer, cleanup-router delta,
provider denial, private error evidence, deadline, billing and frozen parser/cap
tests. Root caught a missing billing-policy export before integration; Sol fixed
it and the complete selected suite passed again. Final/output/event limits and
the frozen turn parser remain unchanged. Unexpected deltas fail immediately with
the finite `notification_optout_unverified` code; old sources stay unchanged.

A different Sol now owns new staged v5, runner/launcher/bundle/host-preflight v7
and reader v8 integration only. Those must bind both existing-credit billing and
exact notification policy into immutable receipts and run manifests, validate
the full source lineage, and retain every original budget/isolation/cleanup gate.
Root review, independent tests, source-only assembly and actual no-inference host
verification remain required before any new pilot. No live benchmark is running.

Root accepted the separate Sol integration after reviewing every changed line,
independently verifying all source pins, actual ordinary/staged paths, receipt
and run-manifest policy mismatch rejection, the frozen AtomicCheckpoint writer,
isolated source-only imports, remote archive validation and installer metadata
projection. The combined selected suite passed **1,279 tests, zero failures**.
The immutable bundle `/private/tmp/hymem-lme-delta-offline-assembly-v1` contains
514 unchanged candidate files, 14 code files and the source map (529 total),
with no dataset, binary, credentials, receipt or model calls. Accepted hashes:

- Staged v5: `9baee48c58696598ec0e14ca721ff214398cc8a9eedb801d115c34837d7f13c5`
- Runner v7: `de6ad419c4aa5c5b0db7e0fdbed4385a672325942381af83d45923b86573b737`
- Launcher v7: `f8a57192e913e3c9c8ab4e3fa6afa93e3a8f53174a8ff94f5571c84c9bbc6f25`
- Reader v8: `5e56efbadbab74af2a2b4019c19f36523d417890cce70ba1006b900553f72c00`
- Bundle v7: `40664c8408ccee96e7797156a7be35cb019b093dc93cfbb5fdc64a3ec3e631ba`
- Host preflight v7: `03732a8d4ad4994e804537007947df3a0525ee67f162f56ce61750d77df03229`
- Source installer v1: `3028b3997427fdc85dc479a14da810b66f46fcfa7890635c2c07a49395cf2caa`

The next accepted operations are actual no-inference host preflight of this
source-only bundle, then installer `--kind pilot` only to prepare a fresh
immutable receipt. The inert historical access checker included by that installer
will not be invoked; the accepted access check is not repeated. No new benchmark
may launch until the resulting root/unit/receipt and exact reader command are
recorded here and the monitor is updated. The new pilot is justified by the
verified fragmentation repair, not by a blind retry of the failed run.

### Fresh delta-opt-out diagnostic LME pilot: p9dzuk8y

Actual Afrodite preflight passed with zero model calls, 514 candidate files,
14 code files, exact original dataset/binary/inventory pins and four selected
questions. Source-only installation and receipt preparation passed; the old
experiments are stopped. Root then independently hash-verified reader v8 and
confirmed `prepared_not_launched` with exact source/policy/receipt validation.

- Root: `/home/atta/.hymem-lme-diagnostic-preflight-p9dzuk8y`
- Unit: `hymem-luna-lme-diagnostic-preflight-p9dzuk8y.service`
- Receipt SHA256: `5df45dee407347006c136642971e333ee8282bf8d83aaee545651570acf18380`
- Reader: `tools/diagnostics/luna_lme_diagnostic_progress_v8.py`
- Reader SHA256: `5e56efbadbab74af2a2b4019c19f36523d417890cce70ba1006b900553f72c00`
- Runner v7 SHA256: `de6ad419c4aa5c5b0db7e0fdbed4385a672325942381af83d45923b86573b737`
- Launcher v7 SHA256: `f8a57192e913e3c9c8ab4e3fa6afa93e3a8f53174a8ff94f5571c84c9bbc6f25`
- Billing: `included_allowance_or_existing_finite_positive_credits_v1`
- Notification policy: `agent_message_delta_optout_v1`

This new four-question/four-worker pilot tests the accepted fragmentation repair
in the same workload, with identical candidate, prompts, frozen dataset/source
order and measurement. The runtime's exact notification opt-out was already
verified on invented local responses; the model's long-run behavior is not yet
proven. Do not rerun that accepted mock or historical access/capacity/grounding
diagnostics. All prior experiments and cancelled DeepSeek stay stopped.

Unchanged limits: GPT-6 Luna low through the same ChatGPT account; campaign
8,012 turns / 48,160,000 known tokens / 14,400 seconds; each question 2,000 turns /
12,000,000 known tokens / 12,600 seconds; indexing 10,800 seconds; canary 12 turns /
160,000 tokens / 600 seconds; invocation 120 seconds; server 14,530 seconds plus
ten-second stop; 256 tasks / 4 GiB RAM / 200% CPU, no restart and recursive
control-group cleanup. Existing credits may be spent; top-up is user-attested
disabled, not runner-enforced. No purchase, enabling reload, model/auth switch,
API fallback, quota bypass or production change is authorized.

Dispatch status: **monitor updated; one launch being attempted now**.
Never repeat an attempted or ambiguous launch. Root must independently verify
startup using reader v8, not command return code alone.

After verifying the reader hash above, the exact read-only polling command is:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite python3 -I -B - --root /home/atta/.hymem-lme-diagnostic-preflight-p9dzuk8y --receipt-sha256 5df45dee407347006c136642971e333ee8282bf8d83aaee545651570acf18380 < /Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics/luna_lme_diagnostic_progress_v8.py
```

While active, read-only monitoring only: no extra model calls, source/model/auth/
budget/production changes, overlapping experiments, restart/resume/reroll or raw
logs/text/stores/private rows/credential export. Retry transient unreadable JSON
once. Unknown/in-flight usage is not zero; unchanged scored counts do not prove a
stall. Intermediate per-turn usage, canary, log activity and stage timing are not
exposed by this reader; do not invent them. Notify question completions, terminal
outcomes or actionable integrity/policy/process/OOM/task-denial/disk-under-20-GiB
issues; remain quiet otherwise.

Require `completed_diagnostic_and_clean=true`, all four scored, complete reconciled
usage, zero denials and independent cleanup. Correctness, canary gold match,
strict indexing health and summary degradation are separate measured outcomes.
After clean completion follow the full-500 readiness and one-shot launch gates
in the authorized continuation plan. This is diagnostic, not canonical or
API-equivalent. On failure preserve finite evidence and cleanup, prove a concrete
defect before a separate Sol repair and root review/testing; no blind reroll or
weakened caps/gates. External auth/quota/access blocks require user direction.

Dispatch completed **exactly once at approximately 2026-09-30 11:18 UTC**,
command return code zero. Root independently inspected the new receipt and live
unit using the hash-verified reader: `checkpoint_running`, 0/4 scored, task
current 26, peak 32/256, zero denials and no resource fault. Canary, complete
usage, answer correctness and final indexing/summary health are not yet known.
This verifies startup and containment, not benchmark completion or effectiveness
of the repair across all questions. Never repeat this dispatch. The existing
Luna monitor is ACTIVE every ten minutes with this exact root/receipt/reader;
cancelled DeepSeek and its paused monitor remain unchanged. From this point,
read-only monitoring only while the pilot runs.

### Terminal timeout: p9dzuk8y (11:29 UTC check)

The hash-verified reader reports `terminal_incomplete_or_unclean`: 0/4 scored,
four failed, 186 admitted turns and 1,357,767 known tokens, with interrupted-turn
usage unknown. First fault: `timeout`, run / turn-events, process 2, request 15,
14 retired threads, queue 0, admitted true and known usage false. Independent
runtime cleanup passed; task peak 124/256, zero denials and no resource fault.
Canary structural validity passed, gold mismatch. No final answer accuracy or
indexing health is established. This receipt is consumed; no benchmark is active.

Do not infer that the opt-out caused the timeout or that every deadline expiry
is a code defect. The adapter's existing invocation deadline is 120 seconds.
Root and a separate read-only Sol are auditing the exact frozen receive/RPC/
parser/deadline paths before proposing any repair. Caps remain unchanged; no
blind rerun, extra model calls or historical mock repetition is authorized.

Root's bounded read-only helper `luna_timeout_metadata_root_v1.py`, SHA256
`d4c04b2c8dc83823bf946b8684b37a9890c8e9ca864ab768e3b2f17b47a98fd6`,
has 18 passing offline projection/privacy controls. It first uses the pinned
reader to verify this terminal receipt and independent cleanup, then reads only
terminal bookkeeping and at most 80 fixed private error slots. It exports finite
process-age/aggregate timing/question counters and allowlisted error classes/
HTTP status/retry flags, never private error messages, raw logs, model text,
stores, private rows or credentials. No provider or runtime calls are made.
This evidence check may establish a cause, or confirm the data was not retained;
it does not itself justify another pilot or full500 launch.

The bounded metadata check completed once. Process age at the first fault was
182.905 seconds, below the process-age guard; this is not the failed invocation's
duration. No private error slots were retained. The canary settled 11 turns /
91,822 tokens with complete usage. Question usage was: q-0000 49 / 356,036
(complete), q-0001 31 / 224,912 (incomplete), q-0002 49 / 347,702 (complete), and
q-0003 46 / 337,295 (complete). All had zero in-flight entries at settlement.
These are call-accounting results, not completed/scored questions. Aggregate
settled timings were preflight 135.001 seconds, model 948.741 seconds and cleanup
0.256 seconds; concurrent totals are neither wall time nor failed-turn timing.

The separate read-only GPT-6 Sol audit found no demonstrated defect explaining
this timeout. Root independently reviewed the original deadline, turn/start
notification preservation, final/usage parser, opt-out and failure-recording
paths, then reran 119 offline deadline/transport/privacy controls, all passing.
The 120-second limit covers the whole invocation, including preflight; the exact
time remaining for model events is not retained. `known_usage=false` means the
parser did not return its result, not that no usage event arrived. `queue_count`
does not measure the stdout-reader queue. The existing observer exports its turn
state only for `incomplete_turn_or_usage`, not `timeout`, so this incident lacks
the completion/usage/event state required to distinguish provider delay from
local scheduling delay. A deadline-boundary queued-event edge is possible in
principle but unproven here; changing its semantics is not an established fix.

No candidate, transport, model, limit or production change was made, and no new
inference or benchmark was launched during this diagnosis. No accuracy or final
indexing/summary-health result exists. Independent runtime cleanup is verified.
The monitor is confirmed PAUSED for user direction because the required failed-turn
evidence is unavailable, as required by the continuation rules. No further pilot
or full500 launch is justified by this postmortem. A possible next step, requiring
an explicit diagnostic plan before execution, is bounded timeout-specific timing
and lifecycle observation rather than another undifferentiated LME rerun.

### Approved preparation: small timeout-focused diagnostic

The user answered “Yes” to preparing the proposed small timeout-focused
diagnostic. This turn prepares and independently tests the instruments; it does
not launch another LME pilot or a paid diagnostic. Both monitors stay paused.
The historical timeout cause remains unresolved.

1. A separate GPT-6 Sol implements a new source-pinned observational transport
   around accepted warm-v8. Record per-invocation phase durations, the original
   deadline, event receipt/consumption timing, finite final/completion/usage
   states, queue depth and bounded reader status before cleanup. Preserve the
   exact accepted parser, billing, delta opt-out, isolation, event/output limits,
   no-retry behavior and 120-second total invocation deadline. No prompts, model
   text, IDs, error prose or credentials may enter diagnostic output. Missing
   observations remain unknown; observing a final or usage shape is not proof of
   acceptance or settled usage. Historical sources and receipts stay unchanged.
2. Root independently reviews the implementation and reproduces delayed first
   event, partial output then silence, queued-event/deadline boundary, preflight
   delay, missing final/completion/usage, foreign/malformed events, opt-out
   violations and cleanup failures using invented offline data. Only after
   acceptance does a different Sol implement the small harness.
3. The proposed harness has 16 predeclared invented-text requests, four workers,
   at most 160,000 known tokens and 600 seconds total, with 120 seconds per
   invocation. Stop on the first technical/accounting/provider/resource failure;
   settle peers and independently verify recursive cleanup. No retries, rerolls,
   resumes, adaptive prompts, historical access-check repetition or real LME
   questions. Preserve GPT-6 Luna low, the same ChatGPT account, existing-credit
   billing policy, 256 tasks, 4 GiB RAM and 200% CPU. A future detached launch must
   use a new private root, one-shot source-bound receipt and a 730-second server
   runtime plus ten-second stop, after actual zero-inference host verification.
4. Root verifies harness source binding, admission/caps, finite metadata output,
   lifecycle/timing evidence, usage reconciliation, first-fault preservation and
   recursive cleanup. Successful synthetic calls establish only the diagnostic
   path, not full LME readiness or the cause of p9dzuk8y's timeout. A timeout
   should identify its observed phase/state without claiming provider fault.

No new model calls or host deployment is part of this preparation. The finite
diagnostic receipt and host verification must be reviewed before any later
execution; a quiet synthetic test must not automatically trigger a larger rerun.
The token limit retains the accepted transport's observed-token admission/stop
semantics: already admitted calls can exceed it, and failed-call usage may be
unknown. It is not a provider-enforced monetary or generated-token ceiling.
Sixteen requests across four warm workers also do not exercise p9's fifteenth
request on one process. These limitations must appear in any eventual result.

Root accepted the separate Sol observer `codex_subscription_timeout_v1.py` at
SHA256 `9fedd5c2151bf016b03c614dff8a2d8b267cd76e73a3d005d2be80c62c1b0b93`
after source review, root-owned failure reproductions, real local producer/
consumer queue controls and 709 independently rerun offline tests. Review caught
and corrected long-uptime timestamp rejection, cumulative queue-counter handling,
startup unknown-state rejection, unsafe JSON-shape projection and first-fault
overwriting during cleanup. No historical transport source changed.

The observer records original deadlines, whole-call start/duration, phase times,
local reader enqueue/dequeue timestamps, queue depth, reader/process state and
matched final/completed/usage shapes before cleanup. Queue observations begin
at `set_deadline` rather than cold process startup, with explicit saturation
flags for counters. Reader `stopped` does not distinguish EOF from malformed
input. Shape observation is not parser acceptance or settled usage. The frozen
pending-notification deadline behavior is deliberately unchanged and covered
by a root parity control: this is instrumentation, not a timeout-semantics fix.
Both monitors stay paused and no provider calls have been made.

### Timeout diagnostic preparation accepted (September 30)

Preparation is complete locally, **not deployed or launched**. A different
GPT-6 Sol implemented `tools/diagnostics/luna_timeout_probe_v1.py`; root reviewed
and independently tested the implementation before acceptance. The final
selected offline suite passes **743 tests**: 711 transport/observer controls
and 32 harness controls, including root-owned end-to-end tests of the actual
observer, warm parser and shared budget with invented local wire responses.
No diagnostic or benchmark inference calls were made in this preparation.

Accepted source identities:

- Observer: `benchmarks/codex_subscription_timeout_v1.py`, SHA256
  `9fedd5c2151bf016b03c614dff8a2d8b267cd76e73a3d005d2be80c62c1b0b93`.
- Harness: `tools/diagnostics/luna_timeout_probe_v1.py`, SHA256
  `066f7f60173b9849b18bf06845c55ee8d1ed0bd3a1cd4608e9f8d6868fa8d7d7`.
- Frozen warm-v8 remains unchanged at
  `0a55d44053349eb90511a597dae20f295c19bb53c734a78b5db3c41197343c21`.

The harness CLI has only local prepare/verify actions. Its callable probe uses
the fixed sixteen invented prompts, four workers, unchanged 120-second
invocation ceiling and 160,000-known-token/600-second campaign limits above.
It requires an explicit containment/resource attestor before setup, before each
admission and at terminal settlement. No attestor means no client creation.
There is no launch, retry, resume or paid-run CLI. It retains returned/failed/
not-attempted counts, known usage, first-fault metadata and finite observations;
missing or malformed telemetry marks the result incomplete rather than erasing
the original fault. Client cleanup is distinct from independent recursive
process cleanup, which remains explicitly unknown inside the running probe.

Root review found and Sol corrected draft problems in budget-stop reporting,
missing-observation reporting, malformed first-fault projection and source
loading. The loader now hashes and executes the same bytes. Root tested both
rejection before execution and disk mutation after the single verified read.
Additional controls verify exact bundle inventory, substituted-preparer
rejection, ambient-import rejection, four concurrent warm clients, denied
admission, terminal resource denial, unknown usage, token overshoot and
preservation of an original timeout through secondary cleanup failure.

Root also prepared a real 13-source local bundle and independently imported it
from `/private/tmp` in a fresh `python -I -B` process, without the repository on
the import path. The post-import exact inventory/hash verification passed:

- Accepted local bundle:
  `/private/tmp/hymem-luna-timeout-prep-NRzYGB/accepted-bundle`.
- Local preparation receipt SHA256:
  `b2b8f6462b3592bc67a2094d59d93aabad3bcd13ce03ad3bb40d9d225372ad1b`.
- This is **not** an Afrodite launch receipt; `host_launch_authorized=false`.
  The sibling `bundle` is an earlier source-only draft, never executed against
  a model, and must not be used for deployment or launch.

Before any live execution, a separately reviewed host wrapper/launcher and
metadata reader must bind this accepted source closure and the pinned runtime,
perform actual zero-inference host verification, create a fresh private root
and one-shot launch receipt, enforce the stated systemd resource/runtime policy,
and independently establish recursive cleanup after exit. Those host execution
components are **not implemented or verified by this local preparation**.
The local callback interface is not proof of actual containment. Both monitors
remain PAUSED; no pilot or full-500 LME is running.

The historical timeout cause and LME readiness remain unresolved. Success on
these short synthetic requests would validate the diagnostic path, not prove
the cause or exercise request fifteen on a single warm process. The OpenAI Docs
skill informed lifecycle instrumentation; it did not justify changing parser
acceptance, deadlines, account policy or experiment limits.

### Authorized continuation: host containment and bounded diagnostic

The user's subsequent “Continue” authorizes completing the next stated step:
host containment/cleanup wiring, independent root verification and one fresh
bounded sixteen-request diagnostic if those checks pass. This does not justify
an immediate LME rerun, longer calls, extra requests or an account/model change.

Ordered implementation gates:

1. A separate GPT-6 Sol implements a new minimal host runner/policy around the
   frozen observer and probe. Source closure remains a separate exact-inventory
   `bundle` subdirectory. Verify hashes before executing the same bytes, bind
   host root/UID/runtime and limits, and gate actual inference on the exact
   live systemd unit, main PID, cgroup membership and resource counters. Root
   reviews and adds independent offline fault/cleanup/privacy controls.
2. After root accepts that layer, a different Sol implements a one-shot
   installer/launcher and metadata-only reader using the accepted host code.
   Preparation must make no model calls. Include a separately named
   containment-only service to verify real policy and recursive cleanup without
   App Server inference. Preserve all prior sources and consumed receipts.
   Root independently tests source binding, malformed state, one-shot launch,
   missing-result failure, terminal policy and recursive descendant cleanup.
3. Stage only the accepted source closure and helpers into a fresh private
   Afrodite root. No credentials, LME dataset or production memory are copied.
   Verify old benchmarks stopped, pinned runtime, disk/memory floors and the
   actual no-inference containment check. Record receipt/root/unit/source hashes
   and dispatch state before making one live launch attempt. Never repeat an
   attempted or ambiguous launch.
4. Run only the sixteen predeclared invented requests using GPT-6 Luna low and
   the same ChatGPT account/existing-credit policy, four workers, 160,000 known
   tokens/600 seconds, 120 seconds per invocation, 730-second server runtime
   plus ten-second stop, 256 tasks/4 GiB RAM/200% CPU. Preserve the parser,
   billing, isolation and event/output bounds. Stop on first fault; settle
   admitted peers. Root reads finite metadata and independently checks cleanup.
   Success establishes only this small diagnostic path. Failure must be
   interpreted from observed timing/state, not guessed provider blame.

Both benchmark monitors remain paused during implementation. Update the
existing Luna monitor only with the accepted exact new identities and bounded
read-only diagnostic instructions before dispatch. Do not automatically follow
a quiet short diagnostic with a larger LME run or repeat an uninformative test.
No production service restart, memory migration, credit purchase, automatic
top-up change, paid API fallback or model/auth switch is in scope.

Root's initial read-only Afrodite survey on September 30 (16:33 Amsterdam /
14:33 UTC) verified UID 1000, the pinned Codex binary hash, 37 old benchmark
units without processes or remaining cgroups, the cancelled DeepSeek container
stopped, 215.15 GiB free disk and 7.18 GiB available memory. These are an initial
snapshot, not launch admission; repeat the bounded checks immediately before
any new dispatch. The unchanged 743-test transport/observer/probe suite passed
again. No live diagnostic has been dispatched.

#### Host runner accepted; launcher verification still pending

Root accepted the first separate Sol implementation after independently reading
the host source, reproducing receipt-origin/type problems, verifying the fixes
and rerunning 784 combined offline controls (including 41 host-layer controls).
The frozen host is `tools/diagnostics/luna_timeout_host_v1.py`, SHA256
`f38c77ee73b05e1004dd9863b873665c55f568bd4e50782701aeaa48ce11d93c`.
The frozen observer and thirteen-source probe bundle are unchanged.

The host checks its exact private installed origin, canonical receipt and
one-shot marker, live unit/main PID/cgroup membership and kernel resource limits.
Its separate containment-only entry point does not load the probe or construct
a client. Terminal cleanup requires a stopped unit with the correct policy and
an absent or recursively empty matching control group; a successful main-process
exit alone is insufficient. Invalid probe output cannot be copied into the safe
result. Successful probe reports require all nineteen containment observations.

A different Sol is implementing only the source installer, one-shot launcher
and finite metadata reader. Root must review and verify them, then perform the
actual zero-inference hosted check before the approved sixteen-request probe.
No host deployment, service launch or model call has happened at this gate.

#### Integration accepted for source-only host staging

Root independently reviewed the separate Sol launcher/reader and source-only
installer, corrected sequencing, receipt/marker validation, ambiguous-attempt
handling, privacy and durable marker issues, and passed 821 combined offline
tests. Root controls include a genuine local source copy followed by fresh
isolated `python -I -B` import of the installed thirteen-source closure.

Accepted immutable helper SHA256 values:

- Host: `f38c77ee73b05e1004dd9863b873665c55f568bd4e50782701aeaa48ce11d93c`.
- Launcher `luna_timeout_launch_v1.py`:
  `74ed7eb2f16d8bab19bbbab0ed7474a56d327f2c13c5a5448d495a3e6ea0d45a`.
- Reader `luna_timeout_progress_v1.py`:
  `fa15e623b33b0cfb386b2e3110e505d8374b2a4041acacf74d49f0afe591c5cf`.
- Installer `luna_timeout_install_v1.py`:
  `98ca478b0140ddd1e72180c3d099a35af97ab200256a8ff42174225e1167fc46`.

Installation is source-only: thirteen code files, one frozen preparation receipt
and three helpers. It does not execute installed source, launch a service, copy
credentials/data or make inference calls. Next: stage once, separately prepare
and dispatch the containment-only service once, read its validated result and
independently verify actual policy/recursive cleanup. Only then prepare the
fresh probe receipt, record exact identities and update the monitor before one
probe dispatch. Frozen code is not modified during any active service.

#### Fresh timeout diagnostic root: dpunlpef

Source-only install succeeded in `/home/atta/.hymem-luna-timeout-dpunlpef`.
Payload SHA256:
`981f7549654532942e9f0f180e70bbc01a2824dd3f56717a48f0ebe1b3c2d7b8`.
The actual hosted preparation reverified source imports, pinned runtime,
old benchmark inactivity and resource floors without inference.

Containment-only unit: `hymem-luna-timeout-containment-dpunlpef.service`.
Containment receipt SHA256:
`079a5ee5a7aedd543819337b0d2df3cdb6adcc4148ce1af6d4a9c1a738953b28`.
Dispatch status: **one containment-only dispatch being attempted**. Never repeat
an attempted or ambiguous launch. This service performs only containment/resource
checks; no client, App Server or model request. The probe is not yet prepared or
launched. Both benchmark monitors remain paused.

#### Containment passed; fresh probe prepared (17:06 Amsterdam / 15:06 UTC)

Containment dispatch returned zero. Root then used the hash-verified reader in
a separate SSH process: source/receipt integrity passed, `verified_and_clean=true`,
correct actual service/cgroup policy, stopped main process and independently
verified recursive cleanup. No model calls, no task denials or OOM events;
230,986,571,776 free disk bytes. This is containment proof, not model readiness.

Probe preparation subsequently passed, including source verification, prior
containment cleanup and fresh host admission. The sole probe is:

- Root: `/home/atta/.hymem-luna-timeout-dpunlpef`.
- Unit: `hymem-luna-timeout-probe-dpunlpef.service`.
- Launch receipt SHA256:
  `01f1e0cfddae84c7ec88cfe6515c366bd0f8a810e76e178b00c5aea8f56384cc`.
- Reader SHA256:
  `fa15e623b33b0cfb386b2e3110e505d8374b2a4041acacf74d49f0afe591c5cf`.
- Dispatch status: **one probe dispatch being attempted**. Only one dispatch is allowed;
  an ambiguous result or any attempt/execution/output artifact consumes it.

Scope remains sixteen fixed invented-text requests, four workers, GPT-6 Luna
low via the same ChatGPT account, existing finite credits allowed, no purchase
or automatic reload. Limits: sixteen admitted turns, 160,000 known tokens,
600 seconds overall, 120 seconds per invocation, 730-second server runtime plus
ten-second stop; 256 tasks, 4 GiB RAM, 200% CPU. Known-token admission ceilings
are not provider-enforced output/monetary ceilings; report in-flight overshoot or
unknown usage honestly. No model/parser/deadline/source change or LME run.

Metadata-only polling, after verifying the reader SHA above:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  python3 -I -B - \
  --root /home/atta/.hymem-luna-timeout-dpunlpef --mode probe \
  --receipt-sha256 01f1e0cfddae84c7ec88cfe6515c366bd0f8a810e76e178b00c5aea8f56384cc \
  < /Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics/luna_timeout_progress_v1.py
```

While active, read-only polling only; no extra model calls, overlapping tests,
restarts or mutation. Require `completed_and_clean=true`, sixteen returned
requests, reconciled complete usage, zero resource denials and independent
cleanup. A failed result retains finite lifecycle/timing evidence. This short
probe does not exercise request fifteen on a single warm process, so even a
clean result does not prove the historical timeout cause or LME readiness.
After terminal reporting, pause this monitor; do not automatically launch LME
or repeat a successful or uninformative probe.

#### Terminal dpunlpef: local billing-admission mismatch proved

The one probe dispatch returned zero; the service then failed before any model
turn was admitted. The pinned reader independently verified both terminal result
and recursive cleanup: four invocation attempts failed during preflight, zero
returned, twelve not attempted, zero admitted turns, zero known tokens, settled
usage complete. Budget stop: `quota_unverified`; first-fault serializer retained
`fixed_other` at preflight/thread-start with `turn_admitted=false`. No turn-start
or event phase began. Attempt durations were 1.87–1.95 seconds, mostly preflight;
all recorded resource gates passed, no OOM/task denial, all processes cleaned.
This did not reproduce or explain the historical 120-second LME timeout.

Root inspected the exact frozen source and reproduced a concrete policy defect
offline with invented metadata: `quota_metadata` accepts a finite positive
existing-credit balance with 1% included allowance remaining, but
`SharedBudget.before_turn` still requires every window to have >=25% remaining.
The credit proof is discarded between these two checks. The synthetic probe
returns exactly `quota_unverified`, zero admitted turns, zero provider calls.
This contradiction is proven; actual account balances were not exported and
the result is not proof of external quota exhaustion.

Next narrow repair: a separate GPT-6 Sol adds a versioned, source-pinned billing
transport that preserves verified finite-credit eligibility through admission
and enforces the same user-authorized policy in both checks. No fabricated quota
percentages, removed checks, raised caps, reload/purchases, model/auth switch or
provider-denial bypass. Missing/malformed/unlimited/unverified credit metadata
must remain fail-closed; explicit provider spend/quota denials remain terminal.
Root independently reproduces and verifies low-allowance/finite-credit success,
zero/missing/unlimited/malformed-credit rejection, explicit denials, per-bucket
scope, reserve/accounting/stop behavior and unchanged notifications/parser.
Only after that acceptance may a separate Sol version and integrate the frozen
observer/probe/host/launcher/reader for a fresh source-bound bounded diagnostic.
No previous root, attempt or receipt may be resumed or reused. LME remains stopped.

#### Credit-admission repair accepted; separate integration in progress

Root accepted the separate Sol `codex_subscription_warm_v9.py` repair, SHA256
`f9081dda08a6e1975a3e951190ae6580161d89817303a1b75d0ba21982fcd470`,
after independent source review, old-fails/new-passes reproduction through the
actual fake-client preflight/admission path, and 850 passing combined offline
tests. A second Sol's read-only audit found no remaining critical admission
issue in this scope. Root controls verify per-bucket proof ownership, rejection
of copied/forged/mutated low-allowance windows, explicit denials and malformed
credit metadata, and AST-equivalent preservation of all non-quota admission
guards. Reported allowance percentages remain unchanged; no credit balance is
exported or serialized into public metadata. Frozen v8 and v1 diagnostic sources
remain unchanged.

A separate Sol is now implementing new v2 observer/probe/host/launcher/reader/
installer files around this accepted v9 transport. The exact fourteen-code-file
source closure, new receipt identities and actual no-inference containment must
be independently accepted before a fresh sixteen-request diagnostic. Both
monitors remain paused; no new model calls have been made. A short success would
validate this billing path, not the historical long-running timeout or LME
readiness. No previous attempt is eligible for reuse.

#### Versioned credit diagnostic integration accepted

Root independently accepted the separate Sol integration after reviewing every
v1-to-v2 source difference, exercising actual four-worker observer/parser/budget
wiring with invented low-allowance credit metadata (old: zero admitted; new:
sixteen returned), and verifying a genuine eighteen-file local installation
followed by fresh isolated host/source import. Final combined regression run:
**906 passed**, with a fresh Python cache prefix. Initial stale test-version
references were corrected before this acceptance. A read-only Sol audit found
no remaining critical integration issue; the tempfile underscore namespace edge
case is covered by independent root and implementer controls.

Frozen SHA256 identities:

- Observer v2: `476e00bcae40c0a061a1b75b10797d4563fb9822012917382e27d9b5a6c93287`.
- Probe v2: `99fb78d584e7fe8ccd1fa7f7eefb6969a29f4604b23228ecc0054b95cc6b1a6c`.
- Host v2: `2e088269bb0e675589f9e71343e1812a6df4671fdcba1bc9ff1c733e65e20158`.
- Launcher v2: `c11a60e2702f422e770762d118464b19a06c03812619bd319657537194e5422c`.
- Reader v2: `497f6422363fd16af4ffd44ba89aac2095ac35605c39ed227ffe1de984a245f8`.
- Installer v2: `67fbdac5ef414e4eb9f291db1531f820a41bb714424698077a6f6f005f6c10df`.

Root independently prepared the fourteen-source closure at
`/private/tmp/hymem-luna-credit-probe-v2.bBLP6k/accepted-bundle`, preparation
receipt `e1291c0e4b3e533e6165a1f12d9c5a2b4947be8341cab7f7a3a2ff5716173d21`.
No credentials, model text, benchmark dataset or production memory are included.
Next is fresh source-only installation and actual zero-inference containment on
Afrodite. A fresh sixteen-request live diagnostic is justified specifically to
test the proven/repaired credit-admission contradiction while preserving timeout
observations. Same invented fixtures, four workers, 160,000 known tokens/600
seconds, 120-second invocations, 730-second server runtime plus ten-second stop,
256 tasks/4 GiB/200% CPU and all isolation/parser/denial/cleanup gates. No previous
attempt is resumed, and this does not authorize an automatic larger LME launch.

#### Fresh credit diagnostic root: ts1mkzsd

Source-only installation succeeded at
`/home/atta/.hymem-luna-timeout-v2-ts1mkzsd`; payload SHA256
`05fc1d56e565f3925a2c6b7f6af63965690f5eec26e972ca11192525286f4155`.
The actual source/runtime/old-run/resource preflight passed without inference.
Containment-only receipt:
`33c18f25bd293a2730fed9232e6fce1493cded7c5cc1fb002f98420179868823`.
Unit: `hymem-luna-timeout-v2-containment-ts1mkzsd.service`.
Dispatch status: **one containment-only dispatch being attempted**; never repeat
an attempted or ambiguous dispatch. No probe receipt or model run yet. The
containment service does not construct an App Server client or make model calls.
Both benchmark monitors remain paused at this gate.

#### ts1mkzsd containment passed; one fresh probe prepared

Containment dispatch returned zero. Root's separate hash-verified reader then
verified source/receipt/result integrity, `verified_and_clean=true`, exact actual
unit/cgroup policy, stopped unit and independent recursive cleanup. Zero model
calls, task denials and OOMs; free disk 230,986,129,408 bytes.

Probe preparation subsequently passed, including repeated old-run inactivity,
resource/source checks and prior containment cleanup:

- Root: `/home/atta/.hymem-luna-timeout-v2-ts1mkzsd`.
- Unit: `hymem-luna-timeout-v2-probe-ts1mkzsd.service`.
- Launch receipt SHA256:
  `4e2e31ab9807a4ef7968d298084402132d49fecaf21be6791637b33f13277311`.
- Reader `tools/diagnostics/luna_timeout_progress_v2.py` SHA256:
  `497f6422363fd16af4ffd44ba89aac2095ac35605c39ed227ffe1de984a245f8`.
- Dispatch status: **one probe dispatch being attempted**. Any attempt or
  ambiguous dispatch consumes this receipt; never repeat it.

This new run tests the accepted credit-admission repair, not a blind reroll of
the historical timeout. Sixteen fixed invented requests, four workers, GPT-6
Luna low/same ChatGPT account, verified existing finite credits allowed under
`included_allowance_or_existing_finite_positive_credits_per_window_v2`.
Automatic top-up remains disabled by user report, not independently enforced.
No purchases, reload changes, model/auth switch, API fallback or provider-denial
bypass. Limits remain 16 turns/160,000 known tokens/600 seconds overall,
120-second invocations, 730-second server runtime plus ten-second stop,
256 tasks/4 GiB RAM/200% CPU. Token admission is not a hard monetary ceiling;
unknown usage or in-flight overshoot must be reported honestly.

After verifying the local reader hash, the exact read-only polling command is:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  python3 -I -B - \
  --root /home/atta/.hymem-luna-timeout-v2-ts1mkzsd --mode probe \
  --receipt-sha256 4e2e31ab9807a4ef7968d298084402132d49fecaf21be6791637b33f13277311 \
  < /Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics/luna_timeout_progress_v2.py
```

While active, read-only metadata monitoring only. No source, budget, provider,
production or model changes; no overlapping calls, restart, resume or reroll.
No raw logs/model text/stores/credentials exported. Require
`completed_and_clean=true`, all sixteen returned, complete reconciled usage,
zero denials/OOM and independent cleanup. Preserve/report finite first failure
on failure; missing/in-flight usage is unknown. The reader has no live per-turn
progress; do not infer a stall from missing terminal result alone. Report the
terminal outcome and pause the monitor; no automatic larger LME launch or probe
repeat. Success does not exercise request fifteen on a single warm process or
prove the earlier 120-second LME timeout fixed. All older Luna/DeepSeek runs stay
stopped, and `finish-lme-validation` remains paused.

#### Terminal ts1mkzsd: sixteen-request diagnostic passed

The single probe dispatch returned zero at approximately 15:37 UTC on September
30. Root then read the terminal result using the hash-verified v2 reader in an
independent SSH process: **`completed_and_clean=true`**. All sixteen requests
were attempted, admitted and returned; zero failed or unattempted. Known usage
was **65,924 tokens**, complete and reconciled. No budget stop, runner fault,
host failure or first transport failure was recorded. All nineteen containment
observations passed with zero task denials and zero OOM events. Independent
runtime verification confirmed exact unit/cgroup policy, stopped unit and
recursive cleanup; disk remained above 20 GiB (230,984,736,768 free bytes).

Observed invocation durations were 3.26–6.70 seconds, median approximately 4.63
seconds, across four warm workers/four calls each; the first-to-last invocation
span was approximately 20.6 seconds, not a separately measured whole-service
duration. Cold preflight took 1.76–2.04 seconds; subsequent preflight 0.53–0.86
seconds. Turn event consumption took 2.60–5.90 seconds. Each call retained one
final response, positive usage and completion; local final queue delay was at
most 0.078 ms. These local queue observations do not measure provider-send timing.
Returned model text was not exported or scored for benchmark correctness.

This live result plus the independent old-fails/new-passes controls verifies the
credit-admission repair on the accepted diagnostic path. It does **not** prove
the historical LME timeout cause, request-fifteen behavior on one warm process,
semantic indexing health or LME answer quality. No LME/full500 was launched.
The successful receipt is consumed and must not be repeated. Pause the Luna
monitor after this terminal report; all old runs and DeepSeek stay stopped.
Further long-session/LME validation requires a justified next scope, not an
automatic repeat or an unsupported readiness claim.

### Authorized continuation: sixteen requests on one warm process

After the clean short diagnostic, the user said **Continue**. The next narrow
test addresses a concrete coverage gap: ts1mkzsd distributed sixteen requests
across four workers, while the earlier p9 timeout was reported at request fifteen
on one warm process. A clean short run is not a fix or reproduction of that
historical timeout. The latter process age of 182.905 seconds was measured at
failure, not at invocation start; it does not justify artificial pacing or a
larger lifetime/deadline.

Plan and independent gates:

1. A separate GPT-6 Sol creates a versioned v3 probe using the unchanged,
   hash-pinned timeout observer v2/warm v9. Allocate the same sixteen invented
   requests to one sequential worker, unchanged source order/prompts/output
   bounds. Total limits stay 16 turns/160,000 known tokens/600 seconds, invocation
   120 seconds, process maximum 16 requests/300 seconds. No fabricated process
   age or disabled rotation. The one worker's allocation is sixteen rather than
   four; aggregate spending and duration limits do not increase.
2. Add finite process-lifecycle coverage evidence from the accepted client and
   private in-process identity comparison, never exporting process/thread IDs
   or text. Require one process, zero rotations, one cold/fifteen warm calls and
   request positions 1–16 for complete coverage. A normal age rotation means
   this test did not cover its intended sequence, not proof of a transport bug.
   Preserve primary failure, known/unknown usage and cleanup evidence if a
   deadline, denial or other fault occurs. Do not retry missing coverage.
3. Root independently reviews and verifies real fake-transport sequential
   execution, identity/counter consistency, simulated rotation, timeout at
   request fifteen, unknown usage, caps, source binding and metadata privacy.
   Only after acceptance does a separate Sol version the host/launcher/reader/
   installer for a fresh private root/receipt. Preserve all historical files.
4. Root reviews and runs offline plus actual zero-inference hosted containment,
   records exact identities, updates the existing monitor, then dispatches the
   one-shot diagnostic once. Preserve server 730-second runtime plus ten-second
   stop, 256 tasks/4 GiB/200% CPU, model/account/credit/isolation/denial policies.
   During execution, read-only metadata polling only; no concurrent experiment.
5. Require all sixteen returned, full same-process sequence coverage, complete
   accounting, zero denials/OOM and independent recursive cleanup. Report and
   pause on terminal outcome. Success establishes only this longer request
   sequence; real LME payloads, four-worker contention, semantic indexing and
   answer correctness still require an explicitly instrumented benchmark pilot.
   No automatic full500 or repeat of a successful/uninformative test.

This changes scheduling/measurement of invented diagnostic data, not candidate
memory behavior or the accepted transport. Existing credits only under the same
account/GPT-6 Luna low; no purchases/top-up changes, API fallback, quota bypass,
model/auth switch or production changes. Official App Server documentation
confirms that unsubscribe does not immediately unload the thread; the process
reuse and existing finite rotation policy must therefore be measured, not
assumed. Source: https://learn.chatgpt.com/docs/app-server#unsubscribe-from-a-loaded-thread.

#### Sequence probe accepted offline

Root independently reviewed the separate Sol probe v3 at SHA256
`6a335463b58bde52d5814943d6d33bb066576cb24c1bc54fabfe38578cd299fd`.
The new probe and root controls pass 31 tests, including the actual accepted
client/parser/budget with invented protocol events, request-15 timeout with
unknown failed-turn usage, normal age rotation, process replacement, token
overshoot retention and type-strict privacy/coverage validation. The adjacent
transport selection passes 71 tests; the unchanged historical regression suite
separately passes 906. These counts overlap and must not be added as unique tests.
All checks are offline; no provider requests occurred.

Review caught and corrected an arbitrary token-overshoot validation cap,
boolean/integer equivalence and overly broad rotation classification before
acceptance. Only verified normal rotation is `coverage_not_reached`; unexpected
identity or counter disagreement is an integrity failure. Strong private object
references avoid identity reuse; no process/thread identifiers are exported.
No accepted transport or historical source was changed. Source-only preparation
and a different Sol's versioned host/launch/reader/installer integration are next;
the live diagnostic remains unlaunched until root verifies that integration.

#### Sequence host integration accepted offline

A different Sol implemented the minimal v3 host/launcher/reader/installer
integration. Root independently reviewed all four diffs and passed a combined
**974 offline tests**. These include actual installation of the exact 18-file
payload, fresh isolated import of its 14-source closure, source/receipt drift,
one-shot and ambiguous dispatch controls, and real-client invented-protocol
replays through probe, host validator and reader. Success, normal rotation,
unexpected process replacement, request-15 timeout/unknown usage and missing
independent cleanup are distinct and fail closed appropriately. A separate
read-only Sol audit found no remaining critical issue. No old sources changed.

Frozen SHA256 identities:

- Probe v3: `6a335463b58bde52d5814943d6d33bb066576cb24c1bc54fabfe38578cd299fd`.
- Preparation: `46d1698838930180ff66e308778751bc4e4e91588f3ce6b18f2d8f24b854020e`.
- Host v3: `7bc1e631f0bec04603aaaccd11a59217136a2e4a38ae5012cabe8d270479266b`.
- Launcher v3: `88f6c722a0bab7c4482f23dd52ad819339414bcd21872758c274163faa171f67`.
- Reader v3: `b630fa645eded98a51860b2c81450260cfc0fa60d4bcaee1da6a3e0dc99d48a6`.
- Installer v3: `e103f01950f3c66bc3d4d0d323b0da08e155a85fe3b2f1c4729343ade4a19a7f`.

Local accepted bundle:
`/private/tmp/hymem-luna-sequence-v3.cZGJnu/accepted-bundle`.
Next: source-only installation in a fresh Afrodite root, actual no-inference
containment plus independent cleanup, then receipt preparation and exactly one
authorized sequence diagnostic after recording its identity and monitor.
This is not an LME launch or proof of the historical timeout's cause.

#### Fresh single-process sequence diagnostic: h3yk00hg

The exact source-only installer completed with zero model calls. Payload SHA256:
`aa6636062ffba72ab6f92aa8d882bba8091f73f60f533d27d9f78a8301c7cc45`.
Actual containment-only service ran once and independently verified exact policy,
zero denials/OOM, stopped unit and recursive cleanup. Its receipt is
`9b9be8e773585546c63ce3388670cf1e8a438ccb8dfde965ee75f0eff5925773`.
No production files, stores, model/auth or transport sources changed. Host
admission reverified the frozen binary, sufficient disk/RAM and stopped old runs.

- Sole fresh root: `/home/atta/.hymem-luna-timeout-v3-h3yk00hg`.
- Sole probe unit: `hymem-luna-timeout-v3-probe-h3yk00hg.service`.
- Probe launch receipt: `71c5ea9e4e150e512a400924b58ce6cf98730ff35c873aae2547cb15e8fa5f3d`.
- Reader: `tools/diagnostics/luna_timeout_progress_v3.py`, SHA256
  `b630fa645eded98a51860b2c81450260cfc0fa60d4bcaee1da6a3e0dc99d48a6`.
- Other source/preparation identities: accepted v3 section immediately above.

Limits: same sixteen invented requests in source order, one sequential worker;
16 total turns / 160,000 observed known tokens / 600 seconds, 120 seconds per
invocation, at most 16 requests/300 seconds per warm process. No pacing, retries,
age manipulation or deadline extensions. Server 730 seconds plus ten-second
stop, 256 tasks / 4 GiB RAM / 200% CPU, no restart and recursive group cleanup.
GPT-6 Luna low through the same ChatGPT account; existing finite credits allowed,
top-up disabled by user report. No purchases, API fallback, quota bypass,
model/auth changes or production changes. Observed token limits stop further
admission; an in-flight overshoot or unknown usage is reported honestly.

Dispatch status: **receipt prepared; monitor being updated; one dispatch will
be attempted next**. An attempted or ambiguous dispatch is consumed and must
never be repeated. Source hash was verified locally before this command:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --launch-root /home/atta/.hymem-luna-timeout-v3-h3yk00hg --mode probe --receipt-sha256 71c5ea9e4e150e512a400924b58ce6cf98730ff35c873aae2547cb15e8fa5f3d' \
  < tools/diagnostics/luna_timeout_launch_v3.py
```

Exact metadata-only read, after verifying the reader SHA256 above:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --root /home/atta/.hymem-luna-timeout-v3-h3yk00hg --mode probe --receipt-sha256 71c5ea9e4e150e512a400924b58ce6cf98730ff35c873aae2547cb15e8fa5f3d' \
  < tools/diagnostics/luna_timeout_progress_v3.py
```

While active, strictly read-only monitoring. No extra provider calls, source,
model/auth/budget/production changes, overlap, resume/restart/reroll or private
text/log/store/credential export. No live per-turn progress is exposed; absent
terminal usage is unknown and does not prove a stall. Require
`completed_and_clean=true`, full same-process positions 1–16, 16 identity checks,
complete reconciled usage, zero denials/OOM and independent cleanup. Normal
rotation is coverage not reached; unexpected identity disagreement is a fault.
Report terminal outcomes and pause this monitor; no automatic LME/full500 or
repeat of successful/uninformative tests. This invented-data check does not
prove historical timeout resolution, real-payload indexing or answer quality.
All prior runs remain stopped; the DeepSeek monitor stays paused.

Dispatch completed exactly once at approximately **2026-09-30 16:09 UTC**,
launcher return code zero and `never_retry=true`. This receipt is consumed.
Only metadata monitoring follows; dispatch success is not test success.

#### Terminal h3yk00hg: complete same-process sequence, independently clean

At approximately **2026-09-30 16:10 UTC**, the hash-verified v3 reader independently
reported `completed_and_clean=true`: all sixteen attempted, admitted and returned,
zero failed/unattempted, **65,930 known tokens with complete usage**. Lifecycle
evidence verifies one process, zero rotations, one cold/fifteen warm calls,
positions 1–16 and sixteen private identity checks. No runner/host/budget or
first-transport fault was recorded. All nineteen resource samples passed with
zero denials/OOM. Independent post-exit inspection verified exact service policy,
stopped unit and recursive cleanup. Free disk was 230,983,274,496 bytes.

Invocation durations: 3.09–6.36 seconds, median 3.63 seconds. The first-to-last
invocation span was 61.84 seconds, not separately measured whole-service runtime.
Request fifteen took 3.63 seconds; request sixteen took 3.72 seconds. Each retained
one final answer, positive usage and completed turn. Summed observed event-phase
time was 47.60 seconds and preflight 13.77 seconds; final local queue delay was at
most 0.065 ms. These are local observer timings, not provider-send latency.
Sampled task count reached 25 (not a continuously recorded task peak); kernel
memory peak was 303,480,832 bytes. No model text or correctness evaluation was
exported; this is invented diagnostic text, not an LME score.

The successful receipt is consumed; do not repeat it. This demonstrates that
request ordinal fifteen and retained thread count alone do not deterministically
fail on these invented requests. It does not prove a cause or repair of the
historical real-payload timeout, longer wall-age behavior, four-worker contention,
semantic indexing or benchmark answer quality. Those remain unverified. No
LME/full500 is active or launched by this test. Pause the sequence monitor and
preserve every older stopped attempt. The next evidence-based step is separately
reviewed integration of the accepted timeout observer and credit gate into a
fresh instrumented four-question diagnostic LME, with unchanged candidate,
model/account, limits and isolation. It requires its own plan, separate Sol
implementation, root tests, hosted zero-inference checks and immutable receipt;
this successful synthetic receipt must not be repurposed or automatically rerun.
