# Diagnostic LME and Hermes1 Honcho hang

## Scope and acceptance

The user approved a distinct diagnostic LME mode and requested diagnosis and
repair of the reported HyMem v64 production hang. Preserve the existing dirty
tree and all immutable experiment sources/receipts. Do not resume old pilots,
change model/auth, launch a full benchmark, or weaken canonical/production
validation. Production text, credentials, raw logs and stores stay on Afrodite.

## Ordered workflow

1. Read-only diagnosis in parallel: root captures bounded live process/source
   metadata while separate Sol agents inspect provider deadlines/locking,
   Honcho/SQLite/event-loop behavior, and diagnostic LME integration.
2. Prove concrete incident defects with local fault controls. Preserve the
   frozen process until useful forensic evidence is captured. A blocked provider
   stack is not by itself proof of why HTTP stopped accepting requests.
3. Assign each narrow implementation to Sol separately. Root reviews the diff,
   reproduces the original defect, and verifies the fix plus adjacent behavior
   before accepting the next implementation. No abandoned timeout workers that
   can continue writing state; no manual dream-row stamps masquerading as a
   repair. Any recovery must bind exact process/lease identities.
4. Implement diagnostic LME explicitly opt-in, with noncanonical provenance and
   segregated checkpoint identity. Report model quality misses and rejected
   extraction/indexing deficits without accepting invalid records. Keep runtime,
   provenance, isolation, accounting, auth/quota, corruption and cleanup failures
   hard failures. Keep strict mode as default and unchanged in meaning.
5. Root tests actual failure/continuation/result paths with synthetic/offline
   inputs, strict-mode controls and resume mismatch controls. Only then consider
   a bounded live integration test if offline evidence cannot establish behavior.
6. Rehearse any production patch against an isolated copy/runtime, preserve
   existing deployment extensions, then recover the exact affected service and
   verify health, capture, embedding and dream behavior. Report separately what
   is locally verified, deployed, and live verified; do not claim all fixed from
   unit tests or a static health check.

## Initial evidence

- Local HEAD: `4e93f76`; production source identity not yet established.
- Sep 29 16:46 UTC: Hermes1 container is running, Honcho host PID 397152 has
  13 threads and has been alive about 91 minutes. Embedding container reports
  healthy. This does not establish Honcho request or provider health.
- Initial code inspection: provider request timeouts are configured, but
  synchronous read/retry scopes do not inherently impose a wall-clock deadline.
  Embedding transport lock spans network I/O; health handler uses the sync
  FastAPI worker pool. These are investigation leads, not yet accepted causes.

## Verification log

Root independently captured native stack metadata from preserved container PID
12381 (host PID 397152), source-stack digest
`0048a76703e38a1686586dd8384d32d0fbcf2096dc8f79755cff49454951b654`:

- GIL owner: `pthread_mutex_lock -> sqlite3_limit -> build_token_overlap_index`.
- Other query: `pthread_cond_timedwait -> SQLite Python callback -> sqlite3_step`.
- Main loop and both provider threads also wait in `pthread_cond_timedwait`.
- No active upload route; only six Python threads. Thus the confirmed incident
  is shared-connection SQLite mutex/GIL inversion, not established provider
  timeout failure or HTTP-worker exhaustion. The native profiler briefly
  attached without locals; only function/file/line metadata was exported.

Sol reproduced the inversion with a synthetic Python UDF. Root independently
reproduced it with repeated overlapping UDF scans on Python 3.11, both with
statement caching disabled and enabled. Both bounded subprocess controls had
to be killed by their parent. A statement-cache toggle is insufficient.

First implementation in progress: Python-side connection/cursor operation lock,
plus whole `core_db.transaction` ownership. Root is auditing metadata methods,
fetch/iteration, cancellation, context-manager ownership and exception cleanup.
No production files or processes have been changed.

Production is an overlay on `af6a615`, not byte-identical to local HEAD. Preserve
it; deployment must apply exact narrow transformations, not overwrite the tree.
Pre-repair production `hymem/core/db.py` SHA256:
`ceab71516bb72424eaf1d49eed50b821fa95b37843f233959e0e051c5c469fc3`.
Embedding client, token-overlap query and producer registry match local bytes.

Root acceptance of the first repair: six independent liveness controls passed
(including killed raw counterfactuals), 138 lifecycle/lease/deadline/registry/
Honcho tests passed, and 65 SQLite/cache/retrieval/bitemporal tests passed. The
16 real-HTTP Honcho SDK tests initially failed at local socket binding under the
sandbox, then all 16 passed with loopback permission. This was an environment
restriction, not a repaired product failure. Separate Sol implementation now
proceeds on diagnostic LME policy; an operations helper prepares a surgical
production rehearsal. Neither implies deployment yet.

Diagnostic mode design: a new candidate-injected adapter/policy module, leaving
the canonical CLI and immutable pilot sources unchanged. It uses the same
candidate convergence, retrieval and judging functions; a separate validated
policy may admit *measuring* pure semantic deficits without rewriting indexing
health. Unknown/mixed/truncated failure diagnostics remain hard failures. A
new runner must bind the helper identity and explicitly label results
noncanonical; source/usage/cleanup gates remain mandatory.

### Accepted SQLite rehearsal

Root's expanded eight liveness controls passed, including native Python 3.11
and interrupt cancellation. Additional scoped suites passed: 160 portability/
embedding/scheduler checks and 67 extraction identity/traversal/dreaming checks.
These overlap earlier suites and are not a claim of a complete project run.

Private production rehearsal: `/home/node/.hermes/repairs/honcho-sqlite-jd70_3xs`.
Original process identity remains PID 12381, start ticks 215405984, uid 1000,
executable `/usr/bin/python3.11`, cwd `/home/node/HyMem`.

- Candidate deployed-overlay db.py SHA256:
  `aa1a252c5a9b32e5b0f91962e5e0499f71525e7d725931ba16597f350f339fb9`.
- Serialized connection helper SHA256:
  `b46335c9e2f8f0453204e4fc9d3f75bc63b13a26bc808e2b0923bc3ea31eca5e`.
- Untouched consistent backup SHA256:
  `a83ef34da8f7ecc7828dd532c5f0909a15e1364bcfa18fd037eff8b45b5c68a9`.

Synthetic concurrent UDF access and two real token-overlap scans passed on a
separate rehearsal copy. `quick_check=ok`; counts unchanged: messages 605,
chunks 2,032, knowledge_graph 12,444, token_overlap_index 23,251. No provider
calls, production DB writes, source installation or process signals occurred
during staging/rehearsal. Installation/recovery is separately fenced and reviewed.

### Diagnostic integration gates

Root tested the actual frozen candidate and protocol (not only mocks): an empty
store converges with healthy status; the strict validator rejects synthetic
quarantine; the diagnostic adapter admits proven semantic quarantine while
retaining `outcome=failure, healthy=false`; provider failure is rejected; the
original store remains usable after fork cleanup. Root found three nested-detail
admission gaps and returned them to Sol before accepting the policy.

Next implementation is a direct, noncanonical runner, not mutation of the old
runner chain. The staged client accepts `complete_stage` only, so ordinary
extraction/digest/reader/judge calls need a warm delegate alongside it, sharing
one per-question budget and one source-bound producer identity. A validating
budget registration facade must not double per-question caps. Both delegates
must close, and the stage ledger must reconcile their combined usage. Preserve
all selected questions in the denominator and label unscored questions unknown.
Canary quality misses are distinct from integrity: never waive an opaque failure
or rerun a completed standalone diagnostic as a substitute for integration.

Root accepted the corrected policy after 65 checks, then added seven independent
staged-verdict controls. Nested failures now require a closed recursive proof;
only typed `grounding:verdict_unsupported`/`grounding:verdict_uncertain` qualify
as semantic grounding loss. Provider/source/support/budget/unknown diagnostics
remain fatal. The joint SQLite/diagnostic/operations regression run passed 111
checks. The direct Luna runner is now under separate Sol implementation.

### Recovery-script incident (separate from the SQLite repair)

The two source files were installed and hash-verified. The original frozen
process was stopped during recovery, but the listener guard raised
`port_owner_unresolved` before replacement launch. The recovery guard treated
an unresolved listening inode as terminal immediately after shutdown. Because
the original invocation environment existed only in that helper's memory,
blindly retrying is unsafe and prohibited.

Independent bounded metadata inspection proves: no matching process, no port
8765 listener, no restart log, and only the recovery-intent receipt exists.
Thus no replacement was launched. The exact command was recovered by hash:
`/home/node/hymem-env/bin/python3 /home/node/hymem-env/bin/hymem-honcho`,
NUL-terminated cmdline SHA256
`664846cdaa6d7ee5e16e2403a0a18461a004c27b676796b44ef340237f1c7429`.

A new, separately reviewed startup step must derive configuration privately
from the maintained launcher, not execute the whole hook. Sources inspected
without exporting values:

- `/home/node/.agent37/hooks/post-restart.sh` SHA256
  `4bd4cc0011f2bfa298cf073d823bce7aaf3a3a1fd3b4f801c010c89a8dcaef68`.
- `/home/node/.hermes/bin/hymem-server-wrapper` SHA256
  `685b198a87e22c877d562e126b0f13780e500e30383b7a3f43ae2b6b73930402`.

The Honcho block contains environment assignments on lines 78–91 (including a
whole `${DEEPSEEK_API_KEY:-}` reference, established by later syntax inspection) and
the maintained console launcher on line 92. A read-only plan must verify their
agreement with the wrapper/live MCP configuration before one-shot startup.
Wait for stable listener absence, write a private PID receipt immediately after
launch, and tolerate short listener-discovery races during bounded verification.
Never re-run the old recovery action or signal an unrelated process.

### Honcho restored and independently load-checked

The source-bound recovery helper initially failed closed on several parser
assumptions (empty environment values, an empty-default key reference, and
`1` versus `true` boolean spellings). Root also caught a plan-return arity
bug before launch and test pollution from an unrestored subprocess mock.
Sol corrected them; root exercised the corrected startup path and gates.
All 14 maintained Honcho assignments now match the live MCP producer settings
(aggregation is semantically equal). Credential values remained on Afrodite.

The one-shot source launch succeeded: container PID 13574, start ticks
216477316, UID 1000, `/usr/bin/python3.11`, cwd `/home/node/HyMem`.
Independent root inspection confirms exact command hash, sole ownership of
port 8765 and HTTP 200. Twenty search requests (four concurrent per round)
and five interleaved health requests passed; final health took 2 ms.
Production `quick_check=ok`, foreign-key violations=0; pending chunks are 201.
A second read-only pass brought successful searches to 40. Inspecting dream
rows shows the `in_progress=true` flag is explained by historical unfinished
rows 1410/1411: no new run row exists. The initial interpretation of that flag
as a new active dream was incorrect and was corrected to the user. This proves
restored liveness under concurrent read load, not new dream progress or
completion of the backlog. Existing malformed-digest and
historical terminal-loss counts remain visible. No capture test message was
inserted and no historical dream rows were manually terminalized.

Diagnostic LME policy/runner/bundle and independent controls now pass 89
offline tests, including the actual frozen adapter/protocol boundary. Root
found and verified fixes for containment callbacks returning false, cached
module-origin binding, final failed-question status, and staged canary context
binding. Full hosted preflight is next; no model inference or benchmark launch
has been performed for diagnostic mode.

Full hosted no-inference preflight subsequently passed at
`/home/atta/.hymem-lme-diagnostic-preflight-oi233cee`, checking the actual
frozen dataset, 514 candidate files, nine code files, module origins and
Python runtime. Runner SHA256
`a8f12c305825806fe4ce2b2f63025098632250f33f224dbee5b1fb6f104d5515`;
Codex binary SHA256
`167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9`.
This copied only verified source into a fresh private directory; it did not
copy credentials/dataset, create a service/run marker, or launch inference.
Root accepted the runner and assigned the separate one-shot launcher next;
the metadata reader follows only after launcher verification.

Root's final targeted SQLite/Honcho/lifecycle/lease/portability/query-cost
regression run passed 246 tests; the 16 real HTTP contract tests were blocked
at socket setup under the sandbox and all 16 passed with loopback permission.
No product test failures remain in that 262-test selection. The full project
suite was not rerun. Diagnostic tests pass 89 plus three source-upload/preflight
controls; counts overlap prior gates and are not a cumulative full-suite total.

Root accepted the one-shot launcher after code review and six offline controls,
then ran only its hosted `--prepare-root` action. Preparation passed host UID,
resource floors, stopped old benchmark workers, real source/dataset/binary
validation and exact receipt construction. No launch marker or service exists.

- Launcher SHA256:
  `145dc2a3ad08d440dd750e1bfd22a8b3a7495f6d1309b8bf45d4e82e4057f5bf`.
- Prepared receipt SHA256:
  `67ce8fec27a8848612c9ae99f3cbd9cf42da41c31762ed7970eca25604548c2c`.
- Reserved unit name (not launched):
  `hymem-luna-lme-diagnostic-preflight-oi233cee.service`.

Root also reproduced the clean-environment systemd-bus failure with a read-only
manager query: HOME/PATH alone fail; adding the fixed user-runtime/bus context
succeeds. Sol corrected the launcher parent environment. The pinned Codex
transport continues to strip those variables from model child environments.
Benchmark prompts/model/caps and the 514-file frozen candidate remain unchanged;
the production SQLite repair was not silently inserted into that benchmark
snapshot. A future diagnostic score must be labeled with this exact candidate.

### Final reader acceptance and handoff

Root reviewed the metadata-only reader and returned its first revision to Sol
for score reconciliation and live/failed-runtime handling. The accepted reader
verifies the receipt, all 514 candidate files and nine pinned code files. It
checks the actual systemd and cgroup resource limits, process membership, and
recursive cgroup emptiness (`cgroup.events populated=0`) before claiming clean
completion. It reconciles correctness against checkpoint rows and keeps
diagnostic completion, strict indexing health, summary degradation and canary
quality separate. Missing usage stays unknown. Stage timing is explicitly
unavailable for this runner rather than inferred.

- Reader: `tools/diagnostics/luna_lme_diagnostic_progress_v1.py`.
- Reader SHA256:
  `3ceb2466ce0a3848e42e2264f1061f7905fd13a92deca84af812f0d79860480a`.
- Root's combined policy/runner/bundle/preflight/launcher/reader selection:
  **111 passed in 9.50 seconds**, including the frozen candidate's actual
  AtomicCheckpoint writer and cgroup fault controls. This is distinct from the
  262-test SQLite/Honcho selection, not a full-project test result.
- `git diff --check`: clean.
- Read-only hosted reader verification: `prepared_not_launched`, selected
  denominator 4, no launch marker, and counts/usage/health correctly unknown.
- Final independent Honcho check: same PID 13574/start ticks 216477316, sole
  listener on 8765, HTTP 200 in 5 ms.

The prepared diagnostic can be inspected without inference using:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --root /home/atta/.hymem-lme-diagnostic-preflight-oi233cee --receipt-sha256 67ce8fec27a8848612c9ae99f3cbd9cf42da41c31762ed7970eca25604548c2c' \
  < tools/diagnostics/luna_lme_diagnostic_progress_v1.py
```

No diagnostic benchmark/model calls or fresh production dream/capture message
were initiated in this repair turn. Historical unfinished dream rows 1410/1411
remain untouched; the observed 201 pending chunks are not a verified drained
backlog. Existing malformed summaries and historical losses are not relabeled
healthy by this repair. The production fix is installed locally on Hermes1;
the source changes are not yet committed or pushed. Hermes2/3, the embedding
server and existing MCP processes were not restarted or changed.
