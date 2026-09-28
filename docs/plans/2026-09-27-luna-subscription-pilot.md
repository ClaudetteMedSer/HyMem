# Luna subscription-only pilot

## Follow-up diagnostic correction

The type differences below were **absent fields**, not incorrect supplied
labels. Retained-response inspection established that the table fact omitted
both optional type keys, as the production prompt permits. The prose fact
included its correct types. See `2026-09-28-luna-canary-contract-repair.md`
for the versioned experimental-gate repair. The v1 failure and its immutable
receipt remain historical evidence; they are not converted into a pass.

The experimental v2 repair is now independently verified: **505 offline tests
passed**, followed by one fresh canary passing with 8 calls in 44.196 seconds.
Private evidence replay and owned-process cleanup also passed. No LME question
was run; this is not a benchmark accuracy result. Details and the new receipt
are in the repair plan above.

## Live pilot result — typed canary failed; no question started

The authorized single GPT-6 Luna canary completed eight subscription turns in
40.254 seconds with 68,388 observed tokens and complete returned accounting.
It emitted **two facts**, but only one satisfied the full typed contract: the
table claim's subject and object types differed from the frozen expected types.
The prose claim matched. All eight responses were valid JSON, each declared
complete, with no markers. This is not a zero-emission or JSON transport failure.

Root replayed the retained responses entirely offline against the frozen
extractor. All eight request payloads matched byte-for-byte; the result again
contained two valid triples, no extraction failure, and precisely the two type
field differences on the table claim. The only execution-path counter mismatch
was its exact typed table-claim emission count (0 instead of 1). No new model
call was used for diagnosis and no raw response text left Afrodite.

The canary gate correctly prevented the one-question run. Final unit state is
failed/exit-code 1 (the intentional gate rejection), main PID 0, restart count 0,
and no remaining cgroup processes. Private progress reports cleanup success,
no in-flight invocation and `question_started=false`. The run is terminal;
do not relaunch/reroll it or switch models automatically. Internal provider HTTP
attempts remain unknown. Completion transport/isolation verification passed,
but Luna has not passed this frozen canary or produced an LME score.

See `2026-09-28-luna-pilot-receipt.json` for metadata. No production service,
memory, model configuration or cancelled DeepSeek run was changed.

## Latest result — completion and bounded isolation verified

Final root verification passed **182 offline tests**, including a root-owned
actual frozen-canary/extractor test, both exact experimental producer hooks,
real adapter open/fork/close with network and paid API construction blocked,
late usage loss, primary-error preservation and cleanup-failure accounting.
The separate Sol review found no remaining high-impact pilot blocker. Final
runner SHA256 is
`75a9772017b0a322a1163f95698caab70492d76811854d48b5e5b94790433c45`.

Afrodite's no-model timeout containment test passed: a transient user service
terminated both its parent and separately session-grouped child at its runtime
limit, with restart disabled. Probe SHA256 is
`a2a49b38917e4fd1658a8af9c42a5d196b4f79a0360afc4be998e2346ae015ad`;
private receipt is `/tmp/hymem-luna-containment-x7s1dz6c`. The pilot will use
the same whole-cgroup containment, 5,390 seconds runtime plus 10 seconds stop
allowance, and a one-shot launch receipt. Staged files are in
`/tmp/hymem-luna-pilot-kBWVxntw`; output is a fresh `run-v1` child. Pending
launch is the already-authorized single canary, then one source-first question
only on success. These tests are not a live canary or LME result.

The one-shot pilot has now launched successfully under user unit
`hymem-luna-pilot-20260928-kbwvxntw.service`, initial main PID `2717730`.
Root verified live unit settings: active/running, `KillMode=control-group`,
`Restart=no`, `NRestarts=0`, runtime 1h29m50s and stop timeout 10s. Launch
receipt is `/tmp/hymem-luna-pilot-kBWVxntw/launch-once.json`. Do not relaunch,
resume or reroll. Read only allowlisted metadata from `terminal.json` and
`run-v1/private-progress.json`; raw evidence/logs/store remain private.

Follow-up root verification passed **164 offline tests**, including the
previously unavailable benchmark-client cleanup tests. The missing `requests`
dependency was installed only into a task-specific temporary test directory;
neither system Python nor production was modified. The pilot harness is still
under review and has not started a benchmark question. Review found missing
canary execution-path checks, experimental producer declarations, and failure
accounting that needed hardening before admission.

Root accepted the Sol implementation after independent review and **129
passing offline tests**. The final transport hash is
`387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491`.
The actual Afrodite runtime then passed all three synthetic probes: exact
JSON nonce echo, no prior-conversation marker in a fresh thread, and an
unavailable response to local-file access with the fixture unchanged and no
write marker. Only user/assistant message items were observed. All owned
processes and process groups were cleaned up. Three complete turns used
12,851 observed tokens in 17.259 seconds; internal HTTP attempts are unknown.

The initial warning investigation ultimately identified two exact runtime
advisories: host-skill discovery is intentionally skipped, and the disabled
Code Mode host fails closed. The second notice's public executable message
was reconstructed and matched byte-for-byte by SHA256
`098e801ebc95c9c7312a945849442846324dcf639365a297313248993822711b`.
Neither tool access nor provider routing was enabled to remove these notices.
Only these exact, thread-bound notices are accepted, with a bounded event
count. Unknown notices, direct-tool fallback variants and wrong-thread notices
still stop. Arrival timing and one-notice-only assumptions were removed:
the protocol does not guarantee them, and they added no security protection.

Four earlier diagnostic turns were interrupted during warning investigation;
their usage is unknown, not zero. They remain in the receipt rather than being
counted as successful or silently discarded. Private warning evidence is under
`/tmp/hymem-luna-warning-We1ydAgE` and `/tmp/hymem-luna-warning-w0wsQkjA` on
Afrodite. No benchmark/production memory was submitted by these tests.

See `2026-09-28-luna-isolation-receipt.json`. The next stage is the separately
reviewed canary and one-question runner. No LME readiness or score is claimed
from these transport probes. The cancelled DeepSeek full-500 run remains
stopped; no production code, service or credentials were changed.

## September 28 — completion implementation and configuration admission

A Sol agent implemented the official App Server completion transport. Root
reviewed it and required fixes for bounded event reads, effective instruction
and capability configuration, runtime account/quota changes, single-flight
budget enforcement, and preservation of known token usage after failure.
Each completion uses a new server process and ephemeral thread; model reroutes,
tools, approvals, unexpected events, missing usage and ambiguous final outputs
stop further invocations. Internal provider HTTP attempts remain unknown.

Root independently passed 74 transport tests (including 12 added negative
controls) and 31 existing provider-accounting/extraction-identity tests.
The broader benchmark-cleanup test could not collect in this local Python
environment because `requests` is absent; it is not recorded as passing.

The actual metadata-only preflight on Afrodite passed with transport SHA256
`096779b6bcfe4cf98a925b5fa3d83c801d8577643579fc969f422905c86cf86e`:
ChatGPT authentication, exact GPT-6 Luna, 86% reported weekly quota remaining,
explicit empty environments, no instruction-source files/workspace roots,
never-approve policy, low reasoning, read-only/no-network permissions and
disabled tools/integrations. Zero turns were started by that preflight.

The older requirement for a literal resolved-tool inventory was not a
supported App Server capability. Admission instead uses the version-pinned
effective configuration and protocol controls, followed by a separate
synthetic runtime check. This is bounded isolation assurance, not a claim
that the entire effective tool catalogue is directly observable. The live
synthetic check is still pending at this entry. Completion remains disabled
by default and requires explicit admission by its caller.

The first synthetic live probe against that hash stopped on
`unexpected_notification:warning` after 2.913 seconds, before accepting an
answer. It initiated one turn; its token usage is unknown, and its owned
processes were cleaned up. This failed attempt must remain part of the record.
The other two checks were not run.

A separate Sol implemented a no-inference warning diagnostic. Root reviewed
it, passed its three tests and ran it against the same pinned transport.
It identified the exact startup notice for the experimental
`skip_host_skill_discovery` flag, which intentionally prevents loading host
skills. The warning's thread ID matched. Only classification and hash were
exported; raw warning metadata stays in the private Afrodite directory
`/tmp/hymem-luna-warning-We1ydAgE`. No additional model turn was used.
The warning hash is
`5e1ab79fba80daaf37190e68176dca33a52dc3141d1ccdd756002009aa128614`.
A narrow handler for this notice, and a regression fix requiring positive
final token usage rather than a zero placeholder, are under verification.

## September 28 — Codex updated; GPT-6 Luna discovered

The user authorized updating Codex on Afrodite, with GPT-5.6 Luna acceptable
only if GPT-6 Luna remained unavailable. Root ran the official standalone
`codex update` command. It updated 0.152.1 to 0.158.0, retained the existing
ChatGPT sign-in, and added the command directory to `/home/atta/.bashrc`.
The `/home/atta/.local/bin/codex` symlink now resolves to
`/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex`.

The refreshed official model catalogue now lists both GPT-6 Luna and
GPT-5.6 Luna. GPT-6 Luna remains the selected target; no fallback is needed.
Existing Codex processes 4535 and 1783801 remained running on their old
executable; root did not interrupt them. New invocations use the updated CLI.

A Sol agent updated the diagnostic's exact runtime pin and fixed one
compatibility false positive: 0.158.0 explicitly returns the default official
ChatGPT endpoint in `config/read`, rather than leaving it absent. Only that
exact official endpoint (with optional trailing slash) is accepted; custom
endpoints and provider overrides remain blocked. Root independently passed
all 43 tests, including old-version rejection and endpoint lookalikes.

Actual no-inference App Server preflight succeeded through ChatGPT auth,
exact GPT-6 Luna catalogue admission, quota inspection (87% of the reported
weekly window remaining), disabled-feature/MCP configuration checks and a
fresh ephemeral thread with the expected model and no instruction sources
or workspace roots. The report still truthfully says
`runtime_tool_inventory_unavailable`, `isolation_verified=false` and
`inference_enabled=false`: finishing transport isolation verification and
the completion implementation remains necessary before the approved canary
and single-question pilot. No model turns or benchmark runs were started.

## Authorization and boundaries

The user approved a subscription-only GPT-6 Luna adapter, isolation/accounting
tests, one extraction canary, then one LME question if the canary passes. This is
a fresh experimental baseline, not a resumption of R9. The cancelled DeepSeek
full-500 run and its paused monitor stay stopped. No HyMem deployment, production
memory, API-key billing, purchased credits, quota reset or provider fallback.

## Sequence

1. Verify the installed official Codex interface, ChatGPT authentication, exact
   `gpt-6-luna` availability, empty environments, disabled integrations/tools,
   fresh ephemeral context and truthful usage before any inference.
2. A Sol agent implements the isolated transport with mocked failure tests.
   Root audits and tests it, including runtime preflight and a tiny synthetic
   isolation probe. Stop if isolation or subscription-only routing cannot be
   established.
3. After transport acceptance, a separate Sol agent implements a standalone
   experimental pilot using frozen R9 ingestion/convergence/retrieval helpers,
   with Luna memory, reader and judge roles. Root verifies it before launching.
4. Run the extraction canary once; only on success run one question from the
   source-ordered frozen LongMemEval-S dataset. Preserve prompts and strict
   indexing/summary gates. Never disguise the Codex harness as canonical API
   protocol compliance. No canary rerolls or automatic benchmark retries.
5. Report canary status, completion/index health separately from correctness,
   elapsed time, observed usage and limitations; retain private evidence on
   Afrodite. No further questions without a new decision.

## Resource controls

The pilot is sequential. At most 1,200 model turns including canary, memory,
reader and judge; at most 90 minutes overall and 120 seconds per invocation.
Stop before another turn if observed tokens reach 4 million or any account
window has under 25% remaining. Token totals cannot include unknown in-flight
usage; report missing accounting as unknown and fail closed. Internal HTTP
attempts are not exposed and must not be fabricated. No API fallback.

## Protocol differences

Use the official Codex App Server through saved ChatGPT authentication, without
reading/copying credentials. Initial runtime was `codex-cli 0.152.1` on Afrodite;
see the September 28 update above for the verified 0.158.0 runtime.
Version-specific generated schemas expose `environments: []` and ephemeral
threads. App Server lacks the original API temperature/max-output-token fields;
record requested versus effective controls instead of claiming equivalence.
Preserve original system/user text where supported; disclose harness context.
Preserve R9's `legacy-custom` grading prompt; Luna grading is an experimental
judge, not the original official-model score. JSON response formatting is
prompt-only, not API-enforced JSON mode. Subscription usage is shared with
ordinary work.

## Evidence

Official documentation reviewed: authentication, SDK, App Server,
non-interactive mode, configuration reference and GPT-6 migration guidance.
Afrodite runtime: `/home/atta/.codex/packages/standalone/releases/0.152.1-x86_64-unknown-linux-musl/bin/codex`.
`codex login status` returned ChatGPT sign-in. No model call has yet been made.
Generated protocol schemas: `/tmp/hymem-luna-schema-6oEcWt/experimental` on
Afrodite, copied locally to the task temporary directory for review.

## September 27 result — blocked before inference

A separate Sol implemented the metadata-only preflight and a second Sol
reviewed the isolation/accounting contract. Root independently ran all 34
mocked tests successfully. The actual remote preflight was then exercised;
the installed runtime rejected the deprecated `tools.view_image` setting, so
the implementation now uses its supported `features.view_image=false` flag.
It also correctly accepts the benign `remoteControl/status/changed` event
only when the status is `disabled`.

The real preflight now reaches authenticated model discovery but stops with
`exact_model_unavailable`: Afrodite's official catalogue lists `gpt-reserve`,
`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, and
`codex-auto-review`, not `gpt-6-luna`. A separate official `codex debug models`
refresh confirmed the same result. Only runtime 0.152.1 is installed there.
The laptop's bundled 0.158.0-alpha.2.1 catalogue includes GPT-6 Luna. This
does not establish that an Afrodite upgrade will grant account access; that
must be rechecked if an upgrade is selected.

No model turns, canary, LME question, paid API call or benchmark rerun has
been started by this pilot. `complete()` is deliberately disabled; this is
not yet a functioning completion adapter or a verified Luna benchmark.
The strict model gate was not relaxed and GPT-5.6 Luna was not substituted.
Full isolation checks still need to finish after exact-model discovery.

Next decision: approve updating Afrodite's Codex runtime and recheck GPT-6
Luna, or explicitly select its currently listed GPT-5.6 Luna for the pilot.
No existing Codex installation, production service, credential file or
cancelled DeepSeek benchmark was changed. The new diagnostic module is
staged only in `/tmp/hymem-luna-schema-6oEcWt` on Afrodite.
