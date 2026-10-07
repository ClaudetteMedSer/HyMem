# Hermes-native OAuth transport for diagnostic LME

## Scope and evidence

The user's October 1 request authorizes fixing the benchmark integration using
their existing ChatGPT OAuth account and GPT-6 Luna. It does not authorize
credential disclosure, another account/model, purchases, automatic reload,
API-key fallback, or changing Hermes/HyMem production.

The last consumed four-question attempt already authenticated and returned 211
successful Luna calls before an admitted invocation timed out. A key change
alone is therefore not a proven fix for that timeout. The existing general LME
entrypoint only implements API-key Chat Completions; the subscription experiment
uses an additional Codex app-server thread/turn transport.

Read-only inspection of the installed Hermes source confirms its native
`openai-codex` provider uses OAuth with direct Responses streaming at
`https://chatgpt.com/backend-api/codex`. Its auth code explicitly supports
importing the Codex CLI credential. The account credential is present, in
ChatGPT mode, not expired according to its unverified local expiry claim, and
has no API key. No token or account identifier was exported.

This is the installed legacy Hermes/Codex route, not the newer Sign in with
ChatGPT direct-plan flow. Official documentation requires a separately granted
`chatgpt.tokens.use.direct` scope for that newer public `/v1/responses` flow;
the existing token does not declare that scope. We will not send it to the
public API or pretend that a Codex credential is a SIWC grant.

## Implementation

1. A separate GPT-6 Sol implements an isolated direct Responses transport and
   offline tests. Exact Luna/low, fixed TLS endpoint, no tools, no redirects,
   no retries/fallback, bounded request child process, explicit terminal
   completion and validated positive usage. Root reviews and independently
   tests it before integration. This removes agent thread routing from
   inference; it does not establish that provider timeouts cannot recur.
2. After acceptance, a separate Sol integrates the transport into the existing
   diagnostic LME bridge, preserving candidate/source order, memory/reader/judge
   roles, staged grounding schemas, budgets, quota/account admission and
   denominator semantics. Keep credential refresh managed by Codex; no copying
   refresh tokens into Hermes production or competing refresh-token owners.
3. Root verifies source closure, isolation, deadline/cleanup, usage settlement,
   staged-request binding and stop-on-denial controls offline, then performs
   source-only host checks. A new source-bound receipt must precede any bounded
   live validation or new benchmark launch. Never reuse a consumed attempt.

The four-question pilot's individual/campaign/time/resource caps remain as
recorded in `2026-10-01-luna-repaired-four-pilot.md`. HTTP streaming has a
different event vocabulary from app-server notifications; redundant text
deltas must have finite byte bounds without accumulating their text. Any
revised transport-specific bound must be explicit, not mislabeled as the old
notification policy. Observed token thresholds are not monetary caps.

## Current status

Native transport accepted offline at SHA256
`fa88c3c2aa9cf042cb0110e5d9f3eea4ffd951270daf57d27dbe972bd2105e94`.
Root independently reviewed the Sol implementation and passed 36 combined
controls, including 13 root-owned tests. The tests cover a real spawned process
that leaves a partial IPC message, deadline termination and reaping, 900,000
characters without pipe deadlock, exact text/schema mapping, model/usage checks,
TLS/redirect/proxy handling and bounded stream bytes. Root caught and required
corrections for usage shape, deadline, model binding and nested tool events
before acceptance. HTTP streaming limits are 16,000,000 bytes, 2,000,000 bytes
per line, 4,096 non-delta events and 1,000,000 final-output characters; ignored
delta text still counts against wire bytes. These are native HTTP limits, not
the old app-server notification opt-out policy.

The account/quota bridge is now accepted offline at SHA256
`4c7162ec021838487a38be4ab068b1b5176bfcee531f70e0e529c4fbf501c04f`.
Root independently reviewed the separate Sol output and passed 55 combined
transport/bridge controls, including its own actual blocked metadata child
termination/reaping test. Corrections made before acceptance include nullable
API-key fields, binding the managed auth file and account identity, immutable
first-fault recording and a metadata-process deadline watchdog. Admission plus
HTTP uses a 120-second active deadline; process teardown uses the existing
bounded cleanup allowance. Four invented concurrent HTTP calls settle correctly
under the same source-pinned budget and credit policy.

The next separate Sol implementation is the two-call native compatibility
harness. The transport is not live-verified. No new inference or benchmark
launch. Historical attempts remain consumed and both monitors remain paused.

## Direct-route compatibility gate

Before another four-question workload, verify the newly implemented native
endpoint with exactly two invented-text calls: one ordinary completion and one
strict JSON-schema completion. This is not a repeat of the historical app-server
access or grounding diagnostics: neither validated the native HTTP route.
The purpose is to establish authentication, exact Luna model acceptance, final
output, structured output and positive reconciled usage without consuming LME
questions to discover a protocol incompatibility.

Use a fresh private source-bound root and one-shot receipt, with at most two
turns, 160,000 observed known tokens, 300 seconds campaign time, 120 seconds
per invocation and a 330-second server runtime plus 10-second stop allowance.
Keep 256 tasks, 4 GiB, 200% CPU and recursive process cleanup. Metadata and
source-only host verification precede any live call. The output contains only
booleans/counts/fixed failure codes/timing; invented response text stays private.
An explicit access/auth/quota denial ends the attempt without a retry, model
change or API fallback. Passing this gate is transport compatibility, not LME
indexing success or answer accuracy. It does not authorize raising any pilot
limits or replaying a consumed launch.

The installed Hermes source also contains an auxiliary header helper that
advertises a first-party originator and shaped User-Agent specifically to avoid
Cloudflare challenges. That workaround is not copied into this adapter. The
reviewed direct transport uses bearer/account binding and ordinary content
headers only; any actual HTTP 403 remains a terminal access failure. No access
challenge workaround is authorized or attempted.

### Fresh source-only host root: wszo8v4u

October 1, approximately 10:16 Europe/Amsterdam: the unchanged reviewed v9
host source preflight installed a new private baseline at
`/home/atta/.hymem-lme-diagnostic-preflight-wszo8v4u` and verified the real frozen
candidate, source inventory, dataset and binary without inference. It contains
514 candidate and 16 baseline code files; the new native files and probe are
not yet installed at this entry. No receipt or dispatch has been consumed.
An independent read-only boolean check confirmed the existing auth file is
private, in ChatGPT mode with no API key, has consistent account/email claims,
and its access expiry covers an invocation. No credential or identifier was
exported. The new probe must extend this fresh source set before preparation;
the old full-pilot launcher must not dispatch this root.

### Native probe acceptance

Separate Sol probe `tools/diagnostics/hermes_native_oauth_probe_v1.py` is accepted
at SHA256 `64925490875f59486fb3a3b2d98e97b2eabf3935d7ad8e08769d9367b9d5ed6a`.
Root reviewed the complete implementation and independently ran 70 combined
controls. These include the real frozen staged schema and real budget/bridge
with invented local responses, four-way concurrent transport settlement,
unknown usage on an admitted failure, access-denial stop after one attempt,
canonical one-shot receipts, actual isolated source/fixture import and negative
completion gates. Source-only host preflight is not live compatibility.
No model call or dispatch has occurred at this entry. The next operation installs
only the four pinned new probe-related sources into `wszo8v4u` and prepares its
own receipt without inference; it does not prepare or dispatch the old pilot.

### Prepared native probe and dispatch: wszo8v4u

Root reviewed the separate Sol installer, SHA256
`e2209ba8dd79add09ebec3f9a53174b68b3a2c5b72b1b688ad947df712ce3c22`,
and independently passed its five offline controls (75 selected checks total).
The source staging tool had created `code` with mode 0775 beneath the private
0700 root; root verified ownership and tightened only that fresh directory to
0700 before installation. No production permissions changed.

Actual zero-inference installation and preparation succeeded:

- Root: `/home/atta/.hymem-lme-diagnostic-preflight-wszo8v4u`
- Unit: `hymem-luna-native-oauth-probe-preflight-wszo8v4u.service`
- Receipt SHA256: `475ac91dbfcc8687b9743f5ad0a549fd1932d0554975e6297c98d84efaefd3cb`
- Probe/inspector SHA256: `64925490875f59486fb3a3b2d98e97b2eabf3935d7ad8e08769d9367b9d5ed6a`
- Native and bridge source identities are frozen above. At most two invented
  calls, 160,000 observed known tokens, 300 seconds; 120-second invocation,
  330-second server runtime plus 10-second stop, 256 tasks/4 GiB/200% CPU.

Dispatch intent is recorded before the one-shot launch command. Root is about
to attempt the sole dispatch after updating the monitor. An attempted or
ambiguous launch is consumed and must never be repeated. No LME question or
full-500 launch is authorized by this probe receipt.

After verifying the local probe digest, the read-only inspection command is:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  /usr/bin/python3 -I -B \
  /home/atta/.hymem-lme-diagnostic-preflight-wszo8v4u/hermes_native_oauth_probe_v1.py \
  --inspect-root /home/atta/.hymem-lme-diagnostic-preflight-wszo8v4u \
  --receipt-sha256 475ac91dbfcc8687b9743f5ad0a549fd1932d0554975e6297c98d84efaefd3cb
```

The inspector revalidates its own source identity through the pinned immutable
receipt, the complete source graph and independent recursive cgroup cleanup.
Only `compatibility_verified` establishes this limited transport gate; it is
not indexing health, LME correctness or full-run success.

### Terminal outcome: wszo8v4u

The sole dispatch returned zero and was consumed. At approximately 10:35
Europe/Amsterdam, the reviewed inspector independently verified terminal failure
and recursive cleanup. One ordinary call was admitted; zero completed. The
finite first failure is `invalid_content_type`, phase `http`, with unknown
usage. Known recorded tokens are zero, but total usage is unknown, not zero.
The HTTP status was 200 (the media-type gate runs only after that check), but no
valid Responses completion was accepted. The structured call never started.
Admission took about 1.87 seconds; HTTP processing about 1.07 seconds. There
were no task denials or OOMs. Broker and recursive cgroup cleanup both passed.
The monitor is paused. This root and receipt remain consumed; no repeat.

The current reader cannot distinguish HTML/challenge, a JSON error, or a valid
non-streaming JSON completion from this one fixed media-type failure. The old
attempt retained no header/body data, so do not assert which case occurred.
Before another inference attempt, a separately versioned transport must prove
offline how these cases are classified, with no request/header/auth change,
no retries or challenge workaround. A normal JSON terminal response may be
accepted only through the same strict model/output/usage checks as SSE. Any
access/quota/auth rejection or HTML/challenge response stays terminal. This is
a narrowly justified native protocol compatibility repair, not proof the
underlying response is fixed; a fresh bounded two-call receipt is required.

A fresh source-only baseline was prepared at
`/home/atta/.hymem-lme-diagnostic-preflight-evpll7nk` with the same verified
514-file candidate and zero model calls. It has no probe sources, receipt or
dispatch yet. Version 2 must preserve the request, endpoint, account, model,
headers and all limits. Do not resend the consumed version-1 attempt.

### Version-2 protocol repair acceptance

Root independently reviewed the separate Sol transport v2 at SHA256
`6845e6a395b40d22cad204bb1e6f4e62001aaa0f9a17609e15051fde620df7f4`.
It hash-binds immutable v1 and retains the identical request bytes, headers,
endpoint, account/model controls, SSE validation and process deadline. A bounded
JSON response passes the same exact model/output/positive-usage checks; HTML,
challenge headers, unsupported media, malformed JSON and finite provider errors
are terminal. No fallback, retry or challenge workaround was added.

Root passed 119 selected offline controls, including independent partial-IPC
deadline/child cleanup, large returned payload, malformed JSON/usage/refusal
rejection, body non-reading on blocked media, source tamper and unchanged v1
bridge/probe gates. This is offline acceptance only. A separate Sol agent is
integrating versioned bridge/probe sources and the full native v1/v2 source
closure; no fresh inference has occurred.

Root then reviewed the complete bridge/probe v1-to-v2 diffs and independently
passed 29 integration controls, including its own four-way concurrency, account
switch rejection, actual metadata-child timeout, staged-schema settlement,
source-only assembly/import, and all new finite failure codes. The accepted
bridge v2 SHA256 is
`74d426d59db464cac2694a1633ae010febb065972f8f593a253e6ad298e28dd3`;
probe/inspector v2 SHA256 is
`809bda870e0fa6706d919321a799617c21d9bcb7560c0ecf7c61768f16e8d4a7`.
Both native v1 and native v2 are explicitly pinned in the new source closure
and receipt; their actual import origins are also verified. All model, account,
quota, invocation, probe and resource limits are unchanged. A separate Sol is
preparing the source-only five-file installer for `evpll7nk`; no dispatch yet.

Root accepted the separate installer v2 after reviewing its full diff and
independently passing 17 controls (165 selected offline controls including the
148-test regression gate). Installer SHA256:
`6f6429095f4bfd5df5d55a4bd2cbe1561969e89ab44e38d1c2f53f0342acd3bd`.
The only host mutation it performs is exclusive installation of the five pinned
probe-related files in fresh `evpll7nk`, followed by source-only preparation.
It cannot launch the service. Root will execute that one preparation next.

### Prepared v2 native probe and dispatch: evpll7nk

Actual zero-inference source installation and preparation succeeded at about
10:52 Europe/Amsterdam. Source graph, frozen candidate/runtime, invented
structured fixture, host admission and private-root checks passed.

- Root: `/home/atta/.hymem-lme-diagnostic-preflight-evpll7nk`
- Unit: `hymem-luna-native-oauth-probe-preflight-evpll7nk.service`
- Receipt SHA256: `680083cbbca542510d32928d55936d28683938a2e8484f9748dc193c5a5858d9`
- Probe/inspector SHA256: `809bda870e0fa6706d919321a799617c21d9bcb7560c0ecf7c61768f16e8d4a7`
- Native v1/v2 and bridge v2 hashes are accepted above and bound in the receipt.
- Unchanged two invented calls, 160,000 observed known tokens, 300 seconds;
  120-second active invocation, 330-second server runtime plus 10-second stop,
  256 tasks, 4 GiB RAM, 200% CPU, no restart, recursive cgroup cleanup.

Root records intent to attempt the sole dispatch after updating the monitor.
Any attempted or ambiguous launch is consumed and must never be repeated.
This v2 probe tests only the proven response-media handling defect. It adds no
header, retry, account/model change or access-challenge workaround. An HTML,
challenge, auth/access/quota failure is terminal and requires user direction.
There is no LME/full500 dispatch from this receipt.

Verify the local probe hash above, then use this read-only command:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  /usr/bin/python3 -I -B \
  /home/atta/.hymem-lme-diagnostic-preflight-evpll7nk/hermes_native_oauth_probe_v2.py \
  --inspect-root /home/atta/.hymem-lme-diagnostic-preflight-evpll7nk \
  --receipt-sha256 680083cbbca542510d32928d55936d28683938a2e8484f9748dc193c5a5858d9
```

The same SSH options with `df -Pk /home/atta` provide the disk-floor check.
Only `compatibility_verified` with validated complete usage, both responses,
structured-contract validity and independent cleanup establishes this limited
gate. Exit status or cleanup alone is not success. Unknown usage is not zero.

The sole v2 dispatch was attempted and returned launch command status 0.
Its receipt is consumed. This is launch evidence only, not compatibility success;
from this point, only read-only inspection is permitted while it is active.

### Terminal outcome: evpll7nk — paused for direction

At about 10:54 Europe/Amsterdam, the reviewed v2 inspector reported
`compatibility_failed`, `failed_exit`, and independently verified recursive
cleanup. The first ordinary request was admitted and failed in phase `http`
with fixed code `unsupported_media_type`. This establishes HTTP 200 but neither
recognized SSE nor JSON, and not a positively classified HTML/challenge response.
The exact unrecognized header value was not retained. Do not infer its contents,
authentication success, model access, or a provider denial from this result.

One call was admitted, zero completed, and the structured call did not start.
There are zero recorded known tokens but usage is incomplete/unknown, not zero.
Admission took 1.365 seconds and HTTP handling 3.138 seconds (total 4.504 seconds).
Reservations/in-flight were zero at terminal settlement; task denials and OOMs
were zero. Broker, containment and independent recursive cleanup all passed.
The read-only disk check found about 214 GiB available. No LME question ran.
The monitor is PAUSED; the cancelled DeepSeek monitor remains PAUSED. Both
native probe roots/receipts are consumed, with no automatic retry or reroll.

No further repairable parser defect has been proven by the live response, so
do not send another observation-only request or relax the media/output gates.
The direct adapter and its versioned integration passed 165 offline controls,
but native endpoint compatibility is not established.

Official OpenAI documentation identifies a different supported direct-plan
route: public `/v1/responses`, requiring explicit OAuth consent granting
`chatgpt.tokens.use.direct`. The existing Codex token was checked earlier and
does not declare that scope. Setting it as an API bearer cannot add that grant.
Moving to that distinct sign-in flow requires user direction and browser
consent; do not change auth or start that flow silently. Availability of exact
Luna on that new route must be checked after consent, not inferred from the
Codex catalog. No API-key fallback, purchase, model substitution or access
challenge workaround is allowed.

References:
- https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference
- https://developers.openai.com/siwc/token-sharing-open-source/sign-in
