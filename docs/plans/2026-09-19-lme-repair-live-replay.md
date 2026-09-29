# Bounded live replay of the three LME repair families

## Authority and scope

The user approved the next bounded DeepSeek replay after the offline repair
report. This is a fresh diagnostic, not a resumed campaign. The numerical
limits below are conservative implementation limits chosen for that approval.
No deployment, service restart, production-memory access, benchmark resume,
full LME run or promotion of the dirty main checkout is included.

Use exactly the candidate in
`docs/patches/2026-09-18-lme-candidate-manifest.json`: all 225 source/config hashes
must match before candidate imports. Its 867 targeted tests passed locally and
in the Hermes1 runtime. The existing stopped-run stores remain immutable.

## Frozen design and limits

- Endpoint: `https://api.deepseek.com`; requested model: `deepseek-v4-flash`.
- Existing candidate system/user requests, temperature, JSON-object format and
  token bounds are unchanged. Thinking is disabled. No extra-body extensions.
- One pass through 10 held fact slices, seven held digest slices, and the three
  exact failed chunks selected before provider output is observed.
- At most one completion per fact case, two per digest case, and 16 per chunk:
  **72 logical completions / 72 HTTP attempts** in total. No HTTP retries,
  repetitions, rerolls, or resumed campaign. The ceiling is not a target.
- Each completion has an owning 120-second wall-clock deadline and bounded
  cleanup. The campaign has a 30-minute monotonic execution deadline; final
  cleanup and read-only auditing may finish afterward.
- A chunk needing more than 16 calls is explicitly `harness_cap`, not an
  application-contract defect or an accepted empty. Existing runtime recovery
  is exercised only inside the diagnostic bound, never expanded.
- Independent model-output rejections remain recorded and the other planned
  cases continue. Transport, accounting, source-integrity, deadline or cleanup
  failure stops the campaign and leaves remaining cases explicitly unrun.

Facts/digests use the retained held cursors and configured 12,000-character
windows. These are not the original sixth-attempt requests, fresh whole-session
indexing, or post-upgrade scheduling. Chunk records are reconstructed exactly
from canonical manifests. No cursor reset, quarantine clearing or database
write is performed.

## Isolation and evidence

Run on Afrodite using a fresh detached UID-1000 container with the previously
verified Hermes1 image and dependency runtime. Candidate, helpers, retained
benchmark stores and the existing credential file are read-only. Only the new
private diagnostic output and temporary scratch space are writable. The live
transport allows only the exact HTTPS chat-completion URL, with redirects and
environment proxies disabled. Credential-free rehearsal has no network.

Read the existing key only in the live diagnostic and send it to owned workers
through private stdin, never command arguments, environment dumps or artifacts.
Full request/response evidence stays private on Afrodite. Provider reasoning
text is not retained. Only hashes, usage, statuses, failure reasons, output
counts and other bounded non-content metadata leave the server.

Before live execution, independently review the helper code and pass a
no-provider rehearsal. Freeze helper hashes, case/store hashes and source hashes.
Preserve an exclusive output-directory fence, reservations, exact request and
response hashes, HTTP attempt receipts and worker cleanup for every call.

## Interpretation

Report collection completion separately from runtime acceptance, empty output,
truncation, semantic correctness and LME readiness. A successful parse does not
prove recall or factual support. Non-stop provider finish reasons remain explicit
findings even if the candidate parser accepts the returned content. No score or
full-run completion claim can follow from this replay alone.

## Preflight accepted and live launch

Root passed 18 adapter, 44 runner and 61 supervisor controls (the three
loopback-only supervisor tests required network-capable sandbox permissions;
they passed with dummy local credentials and no provider access). The 20-case
inventory matches all ten earlier store hashes and all earlier facts/digest
user-request and chunk-source hashes. Order is fixed round-robin by family.

Both real-input, network-disabled Afrodite rehearsals completed 20 cases and
27 scripted completions each, zero HTTP attempts. The positive run accepted all
20 controls; the negative run rejected all 20 with the intended family reasons.
Root independently reconciled all 54 reservations, request/response hashes,
case counts/caps and owned-worker terminal/cleanup receipts. Both containers
exited 0 with PID 0 and no OOM.

Frozen manifest:
`674e725084db3dea89371490335f13cabb9b16e764bdab4de03e9f6a3b53360a`.
Detailed prelaunch receipt:
`docs/patches/2026-09-19-lme-live-preflight.json`.
Afrodite staging:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-repair-live-20260919-AvLFBi`.
The detached live container is `hymem-lme-repair-live-AvLFBi`, ID
`9a88a1ab54ac8cb58da5809f445cef0f5ac050e48302b45dc51fe4ad90fdb4ac`.

## Live result: halted after one paid request

The original single pass is spent and must not be resumed. It made exactly one
completion / one HTTP attempt. No campaign case completed: the first case halted
before parser dispatch, and the other 19 cases were never started. The response
reported 1,020 prompt tokens and 300 completion tokens (1,320 total), with
`finish_reason=stop`. Provider reasoning-token metadata was absent.

The harness requested `deepseek-v4-flash` but received the model label
`deepseek-flash`. Its strict label-equality guard halted with `response_model`;
this was a diagnostic preflight error, not an observed candidate extraction
failure. The newer main checkout already documented the retired alias, but root
missed that change while preparing the isolated diagnostic.

DeepSeek's [official documentation](https://api-docs.deepseek.com/) and
[September 10 update](https://api-docs.deepseek.com/updates/) explain that the old
V4-Flash aliases remain accepted but route to V4.1-Flash; the old model weights
are retired. Retaining the old request label therefore cannot reproduce an old
V4-Flash baseline. Matching labels do not prove immutable weights either.

The live container exited 1, PID 0, with no OOM. Root reconciled the sole
reservation, HTTP attempt, request/response hashes and cleanup receipts. The
worker was reaped and its process group absent; no cleanup warning remained.
The original halt receipt's `usage_complete=false` is preserved. The token
figures above are independently recovered from the saved response, not a claim
that the campaign completed normally.

A separate credential-free, network-disabled replay consumed that one saved
response against the byte-identical original request. The candidate accepted
five facts without a parse failure, database persistence or cursor advancement.
This establishes parser acceptance only, not semantic correctness or recall.
The offline replay container exited 0 with PID 0 and no OOM. The original live
campaign remains halted and its evidence unchanged.

## Corrected diagnostic prepared; new paid run awaits approval

The isolated `live_probe_v2.py` requests the documented current route
`deepseek-flash` and uses a new diagnostic schema. Root verified that its AST is
identical to v1 apart from the module documentation and `MODEL`/`SCHEMA`
constants. Strict response-label checking, prompts, temperatures, limits,
accounting and cleanup are unchanged. All 46 v2 runner tests pass independently;
the original v1 runner hash remains unchanged.

The proposed fresh run keeps the same 20 frozen cases, at most 72 completions /
72 HTTP attempts, 120 seconds per invocation and 30 minutes per campaign. It
requires a new manifest and output directory, plus a no-network real-input
rehearsal before launch. Including the stopped first pass, the proposed total
would be at most 73 paid calls. No second paid run has started. Approval was
requested separately; this document does not grant it.

There is still no full-LME readiness verdict, score, deployment or restart.

## Final read-only evidence audit

Both original real-input rehearsals have now also passed the standalone receipt
auditor: 20 cases and 27 scripted calls each, zero HTTP attempts; all 225 source
hashes and all ten original store hashes still match. The cases cover nine
stores; the tenth retained store contains no selected failure case but is still
included in the independent immutable-store ledger.

The auditor itself needed two fixture corrections before it could verify real
receipts: it had assumed every retained store contributed a case, and had
omitted the supervisor's standard `intent.json` from its expected file list.
Those assumptions are fixed without modifying any original campaign artifact.
Its 28 regression tests pass independently, now including receipt shapes
generated by the actual supervisor. These were audit-tool defects, not new
candidate memory-pipeline failures. Arbitrary extra artifacts remain rejected.

Metadata-only final receipt:
`docs/patches/2026-09-19-lme-live-halt-reconciliation.json`.
Raw benchmark requests and responses remain private on Afrodite.

## Fresh current-model pass approved

The user subsequently approved the proposed fresh run. Its new root is
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-repair-current-20260919-AKZh09`;
the original spent run remains untouched. The model is `deepseek-flash`, with
unchanged 20-case inputs and 72/72/120-second/30-minute bounds. The two passes
together cannot exceed 73 paid calls. No production deployment, full LME run,
reroll or resume is authorized by this pass.

Root independently reran 46 runner tests and 43 current-model receipt-auditor
tests: all passed. Both fresh network-disabled real-input rehearsals completed
20 cases / 27 scripted calls, zero HTTP, and passed full receipt auditing.
The pinned source still has 225 matching files and all ten store hashes match.

Fresh manifest:
`8d6f7e89c9e70b910813fe63a3b4fe2500249857fe6e4a838854d7cb6010bbcb`.
Prelaunch receipt: `docs/patches/2026-09-19-lme-current-live-preflight.json`.
This section records prelaunch verification, not a live success verdict.

## Completed fresh pass: collection succeeded; three application rejections

The fresh pass completed all 20 cases in about 156 seconds on September 18 UTC
(September 19 locally), using 30 completions / 30 HTTP attempts. All responses
reported `deepseek-flash`, the same reported fingerprint, and `finish_reason=stop`.
No transport failure, truncation, diagnostic call-cap, deadline or cleanup failure
occurred. Usage: 72,488 prompt + 8,285 completion = 80,773 reported tokens.
The missing reasoning-token metadata is not represented as zero. Both passes
together used 31 paid calls; no further paid call has been made.

| Family | Accepted, nonempty | Accepted, empty | Rejected |
| --- | ---: | ---: | ---: |
| Facts | 7 | 2 | 1 |
| Digest | 5 | 0 | 2 |
| Chunk | 0 | 3 | 0 |

The rejections were `output_capacity_exceeded`, `summary_output_cap` at
`summary_compaction`, and `episode_validation_failure` at `primary` (one rejected
episode item). Of the three digest cases that needed summary correction, two
recovered and one did not; a correction can still violate its 500-character limit.
All three chunk cases passed the runtime's primary/empty-verification path with
zero triples and markers, using 2, 6 and 2 calls. This does not demonstrate recall
or that emptiness is semantically justified.

The standalone current-route audit passed all 30 reservations, request/response
hashes, token counts, case limits, worker ownership and cleanup receipts. All 225
source hashes and all ten original store hashes still match. The container
exited 0 with PID 0 and no OOM. No original cursor or persisted record changed.

Root also independently reran ten existing retry/persistence integration tests;
all passed. For a fresh retry policy, fact-capacity and primary episode-validation
failures lead to smaller source windows. A summary-compaction failure instead
consumes an attempt without shrinking the source; six persistent failures can
still quarantine. The held-window direct replay bypasses that scheduler, so it
proves neither recovery nor inevitable production nonconvergence.

Final metadata receipt: `docs/patches/2026-09-19-lme-current-live-result.json`.
This is a completed diagnostic with three rejected cases, not a clean readiness
gate, full benchmark result, deployment or model-performance comparison.

## Exact saved-response diagnosis, zero additional API calls

A separate read-only, network-disabled container replayed all 30 saved responses
against byte-identical rebuilt requests. All 20 outcomes matched. Root reviewed
the inspector, corrected its manifest filename and saved-attempt telemetry,
and passed its six metadata/privacy tests before running it. The inspector exited
0 with PID 0 and no OOM; no credential was mounted.

The rejected replies now have concrete, directly verified causes:

1. Case 4 returned nine otherwise valid facts against an eight-item cap. It did
   not use the explicit `complete:false` capacity sentinel. The response was
   correctly held instead of silently dropping its ninth fact.
2. Case 5 returned a 644-character summary. Its summary-only repair was valid
   JSON but still 635 characters, above the 500-character cap. The repair kept
   its exact source inputs and failed honestly; no primary items were published.
3. Case 13 contained three episode items; the second omitted the required
   `title` field and included one unrecognized field. Its citations were all in
   the actual source allow-list, unique and correctly ordered. This was an item
   schema failure, not an evidence/provenance mismatch. The other two items
   validated independently, but the atomic digest was correctly held.

These saved responses provide reproducible offline regression evidence; they
must not be silently normalized into success or counted as recovered. No
additional model call, production edit or full LME run followed the diagnosis.
Offline receipt: `docs/patches/2026-09-19-lme-current-offline-inspection.json`.
