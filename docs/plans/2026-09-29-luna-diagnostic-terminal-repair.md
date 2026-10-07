# Diagnostic pilot failure: reporting repair and capacity regression

## Preserved terminal evidence

Pilot root `/home/atta/.hymem-lme-diagnostic-preflight-oi233cee`, receipt
`67ce8fec27a8848612c9ae99f3cbd9cf42da41c31762ed7970eca25604548c2c`,
failed at 2026-09-29 21:26:22 UTC (87 seconds after systemd startup).
The 21:35 heartbeat's accepted reader failed twice with `metadata_unverified`.
Root's separate read-only finite projection identified `terminal_budget_invalid`;
all source/receipt pins still verify.

- First fault: `rpc_failure:thread/start`, preflight, JSON-RPC `-32603`
  (`internal_error`), process 1/request 1, no retired threads or queued events.
  This particular request did not admit a model turn.
- Terminal: 0 scored, 4 failed; 14 admitted turns, 113,065 known tokens,
  complete usage accounting, zero reserved/in-flight calls.
- Canary: 11 calls/91,830 tokens, structurally valid, gold mismatch with proven
  semantic `branch_incomplete`. The diagnostic policy correctly continued; the
  transport failure is separate and remains fatal.
- Unit: failed/exit-code/status 1, no restarts, MainPID 0, empty ControlGroup;
  exact cgroup is absent. Kernel task counters at the time of failure were not
  retained, so current peak/denials are unknown, not zero. MemoryPeak 402,161,664.
- At first fault the shared ledger's usage-complete flag was false while peer
  turns were in flight; terminal settlement is complete. Do not misreport that
  intermediate flag as permanently unknown billed usage.

Root helper `tools/diagnostics/luna_diagnostic_failure_metadata_root.py` sends
the SHA-verified existing reader plus a finite metadata projection through SSH
stdin. It never imports benchmark code, reads logs/stores/private question rows,
or exposes raw server error text. Old artifacts are immutable; no rerun occurred.

## Concrete defects and ordered repairs

1. **Reader schema mismatch:** v1 permits only alphanumeric/underscore stop
   codes, but the pinned transport deliberately emits finite `family:RPC/method`
   codes. The valid terminal result is rejected. Also v1 conflates failed exit
   with unverified process cleanup. A separate Sol implements standalone reader
   v2 retaining v1/source pins, exact finite error vocabulary, first-failure
   projection and independent cleanup for failed units. Root must verify genuine
   transport/checkpoint values, private-text rejection and failure-not-success
   controls, then inspect the preserved terminal result with v2.
2. **Reintroduced capacity policy:** the new diagnostic launcher, runner and
   reader use TasksMax 128. The accepted September 28 capacity repair already
   proved 128 insufficient for four real LME workers and established a bounded
   256-task correction. See `2026-09-28-luna-lme-capacity-run.md` and
   `2026-09-28-luna-warm-recovery.md`: the previous controlled failure observed
   128/128 tasks and kernel denials before cleanup; its 176-turn counterfactual
   under 256 had no RPC failures/denials and complete accounting/cleanup.
   Restoring 128 in the new launcher is a concrete regression. The same RPC
   signature is consistent with recurrence, but does not by itself prove the
   kernel state of this latest occurrence.
3. After reader acceptance, separately implement a versioned capacity-corrected
   runner/launcher and matching observer. Preserve candidate, prompts, model,
   diagnostic/quality/accounting gates and spending/time caps. Examine the dual
   ordinary/staged clients' maximum process residence; add numeric cgroup
   task/denial samples before cleanup so a new failure cannot lose that evidence.
   Do not assume the old single-transport capacity test alone proves the new
   dual-transport workload. Verify offline, then use one fresh bounded same-four
   pilot with pre-cleanup resource observations to exercise the actual dual
   workload. The already accepted 176-turn capacity counterfactual must not be
   repeated. Do not rerun the known inadequate 128-task experiment merely to
   rediscover the old defect.

This is not a provider/model reliability conclusion. No quota or credential
bypass, production change, quality relaxation or full500 launch is justified
by these findings. Continue the user-authorized sequential Sol/root workflow
from `2026-09-29-luna-authorized-continuation.md`, updating exact run identities
before any justified future launch.

## First repair accepted; second in progress

Root reviewed standalone `luna_lme_diagnostic_progress_v2.py` (v1 unchanged),
SHA256 `9e9868228979d093223f8e4fa3e9f983dc88a070e3781a05e98a9a7e214e4eec`.
The 32-test reader/launcher/root-runner selection passed. Root required and
verified finite handling of `usage_unknown`, numeric/invalid RPC errors,
privacy-safe app-server error classes, and actual failed-unit cleanup controls
for populated descendant cgroups, wrong policy and nonzero MainPID.

Hosted read-only v2 verification succeeded against the untouched failed pilot:
`terminal_incomplete_or_unclean`, 0 scored/4 failed, exact first
`rpc_failure:thread/start` with -32603, 14 turns/113,065 tokens, terminal usage
complete, `runtime_cleanup_verified=true`, `completed_diagnostic_and_clean=false`.
It also correctly retains first-fault usage incompleteness separately from
settled terminal usage. No additional model calls were needed for this repair.

Exact command for this preserved failed experiment now uses v2:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  '/usr/bin/python3 -I -B - --root /home/atta/.hymem-lme-diagnostic-preflight-oi233cee --receipt-sha256 67ce8fec27a8848612c9ae99f3cbd9cf42da41c31762ed7970eca25604548c2c' \
  < tools/diagnostics/luna_lme_diagnostic_progress_v2.py
```

Only after that acceptance, a separate Sol was assigned the capacity-corrected
versioned runner/launcher/reader and source-only preparation helpers. Root must
review their frozen diff, run negative policy and dual-client observation
controls, and perform no-inference hosted preflight before creating a new
immutable receipt. There is currently no active pilot and no new launch.

## Capacity repair accepted for a fresh bounded pilot

At approximately 22:00 UTC root independently accepted the second Sol's
versioned capacity repair after review and **380 passing offline controls**:
354 existing diagnostic/transport/helper regressions and 26 new capacity,
source-bundle and root-written ledger/terminal controls. The latter exercise
the actual nested budget before admission and first-fault cleanup, retained
counter evidence, missing/unreadable observations, exact 256-task policy,
wrong-policy rejection, real frozen four-row checkpoint serialization and
failure-not-success controls. A draft source/pin mismatch was caught locally
and reconciled before staging; the final pins below were rechecked by root.

- Runner v2: `5a9b29456c8fcc96fd9df7d1532a457662eccba63e5ef39f737121fe349565d9`.
- Launcher v2: `1ece110f0ff84d5e08c1178e5da91913aff20b35bf571385925ef953e3a176b8`.
- Reader v3: `91d2388a2ec558ac49e040b7bd5e2378c0c9abd1aab6b993d785bd7838f34242`.
- Assembler v2: `168f70b8af14326adfde8e0cd1b6a0691653089e674c272580d5f62d50b9ceb5`.
- Host preflight v2: `d060fd3976e2a263f509d53954626bf954c77e0bc4979d6806e28d0226114e01`.

Changes restore the previously accepted 256-task bound and add finite numeric
task observations before both client paths admit a turn, before first-fault
cleanup, and at terminal settlement. The reader rejects resource inconsistency
and independently requires zero denials for clean completion. All candidate,
helper, prompt/measurement, model/auth, spending/time/memory/CPU limits and
four-question/four-worker selection remain unchanged. Source-only assembler and
host preflight preserve the accepted 514-file candidate and nine code files.
Old source versions, receipts and failed-run evidence are untouched.

This acceptance proves the implementation's offline controls, not resolution
of the latest live RPC failure. One fresh same-four pilot is justified to
exercise the restored capacity bound with the actual dual-client workload and
retain kernel counters if it fails. First perform actual no-inference host
preflight, prepare a new immutable receipt, and update the run plan/monitor
before dispatch. No additional capacity diagnostic or old-run reroll is needed.

## Next proven defect: incomplete stage attribution

The fresh 256-task pilot `h0nfxj0v` has now failed and cleaned up, as recorded in
`2026-09-29-luna-lme-diagnostic-capacity-run.md`: 24 turns/183,828 known tokens,
complete accounting, 0/4 scored. Task peak 130, no denials. The stop is
`stage_accounting_failure`, not a task fault or provider error.

Root's bounded metadata-only log census found the frozen profile and fact
extraction functions in four stage-accounting tracebacks. The raw 8,282-byte log
stays on Afrodite. Independent Sol offline execution of a real frozen
`HyMem.dream()` with invented input and a fake LLM reproduced those paths.
Root inspected the source: both features default on, but the accounting
allowlist contains only chunk extraction/grounding, digest, reader and judge.
Normal retrieval's `llm_rerank` is also reachable through `adapter.search` and
omitted. This is a benchmark wrapper coverage defect. Do not disable features
or allow arbitrary paths to get past the accounting guard.

Repair plan, ordered:

1. Finish a source-backed inventory of every model call reachable by this exact
   frozen pipeline. Distinguish actual enabled paths from optional APIs/features
   not used here. Root independently runs the real-candidate offline reproduction.
2. A new, separate GPT-6 Sol implements a versioned runner v3 with exact finite
   attribution for the proved missing stages, matching launcher v3/reader v4 and
   versioned source-only assembler/host preflight. Preserve all v1/v2 sources,
   accepted 256-task observer and all candidate/model/auth/budget/quality gates.
   Unknown paths and invalid ordinary/staged routing still fail before dispatch.
3. Test real frozen dream and retrieval, reader/judge and staged routes with
   fake provider accounting, not merely invented stack-frame filenames. Cover
   primary/recovery variants, unknown paths, failure settlement, stage/total
   reconciliation, denominator preservation and unchanged policies. Root reviews
   and independently exercises the output before acceptance.
4. Only after acceptance, assemble new source bytes, perform actual no-inference
   host preflight, create a fresh receipt, update exact plan/monitor identities,
   and launch one justified same-four diagnostic pilot. Never restart h0nfxj0v.
   No extra paid diagnostic is needed to establish this attribution defect.

The pilot is currently stopped. No full500 readiness claim is justified yet.

Root independently executed the source-only reproduction in an isolated Python
process. Exact real-stack census: 20 chunk calls classified extraction, one
digest classified digest, and one each profile/facts/rerank classified
unclassified. Source audit excludes standalone summary recovery and query.ask;
coreference is not active in this adapter route. Facts replay is a reachable
variant of the enabled facts tier and must receive its own exact classification.
Separate Sol `sol_stage_attribution_repair` owns only the new versioned files
and its tests. Root has not accepted them or authorized a fresh launch yet.

### Attribution repair accepted after root verification

At approximately 22:16 UTC, root reviewed the frozen v3 diff and independently
ran **390 passing offline regressions**. Root also separately ran the new
actual-candidate verifier: real HyMem dream/augment, profile, fresh facts,
authoritative fact replay (one charged fake turn), reader/judge entrypoints and
staged grounding all executed with fake completions and exactly reconciled
turn/token totals. Unknown calls still reject before dispatch; deliberate token
mismatch rejects reconciliation. Root's AST/byte controls prove unchanged
measurement, canary, question worker, budget/resource observers and cleanup;
reader/launcher logic only changes source and version bindings.

Only three ordinary-call attribution rules are added: profile, facts
(fresh/replay) and rerank. Candidate/helper/transports/prompts/model/authentication
and all limits remain unchanged. Root caught and returned a stale embedded
host-preflight hash before any upload; corrected inner/outer pins now have an
explicit offline AST regression check.

Frozen accepted hashes:

- Runner v3: `e52392dff24a1b1234bc995bbe9a4259cfef6a1013e6f7a6133b99bfc73202c0`.
- Launcher v3: `7324f7a1a12ef19df8eb8be09d449836efe66ac9a58ac450452eaace7b0f4391`.
- Reader v4: `59dfc88ccb7ce846e8d4e2ba09ce1f0ab7c8e4139a56732b72bc2d682ef67d82`.
- Assembler v3: `ad6d2580fe747f053b8582a1ac6eb2e38fcfc1047bdc962dbf733a8b1a413df6`.
- Host preflight v3: `3eaefdefb9a886f873cd22eedc4b9169311245e544b8b2d42f143050fcd9853b`.
- Real-candidate verifier: `858bfcb63e0e1e0db5bb5bf3556bd5ef855b4d2a746bc83748f8d62f29c71022`.

This justifies one fresh same-four pilot after actual no-inference hosted
preflight and a new immutable receipt/monitor identity. It does not establish
live completion or authorize retrying either preserved failed attempt.
