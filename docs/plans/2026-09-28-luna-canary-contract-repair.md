# Luna canary contract repair

## Diagnosis

The September 28 single subscription pilot failed its experimental v1 canary
after eight valid completions. Retained private evidence is
`/tmp/hymem-luna-pilot-kBWVxntw/run-v1/private-canary-evidence.json` on Afrodite,
SHA256 `0acb3cad96a92c487ba5cd6182caae0eb2730efb87b932c858575edcf091fe52`.
The table claim omitted **both** `subject_type` and `object_type` keys. It did
not supply incorrect types. The prose claim supplied both correct types.
The earlier description of incorrect types was too strong and is corrected
here; the original failed run and its receipt remain intact.

Both core facts, polarity and source attribution were correct. The frozen
extraction prompt explicitly makes entity types optional, and the production
validator accepts absent types. The v1 pilot copied the canonical canary's
stronger oracle, requiring all expected hints. A second hidden dependency is
the canonical `_response_expected_claim_indexes`: it also requires types, so
the execution-path report misclassified a correctly contextualized, untyped
claim as no exact-context emission. Offline byte-faithful replay reproduces
this without a model call. There is no evidence of transport mutation.

Two separate Sol reviews confirmed the contract mismatch. The actual Codex
route continues to preserve the system and user prompt texts; no new model,
temperature, response schema, or transport change is needed. Official protocol
reference: https://learn.chatgpt.com/docs/app-server.

## Scope and decisions

- Repair the experimental subscription screening canary, explicitly versioned
  v2. This is not a change to frozen R9, its canonical typed canary, or an
  official-model benchmark score. Preserve all historical artifacts.
- Do not add expected types to model responses or stored triples; do not
  hard-code answers into prompts or whitelist a Luna-specific wrong label.
- Missing optional types are diagnostic coverage, not missing core claims.
  Every supplied type on an expected claim must still match its expected type;
  invalid and conflicting hints must fail even if normalization/merging would
  otherwise remove them. Check admitted responses as well as final hints.
- Preserve exact core claim identity, polarity, provenance, fixture hashes,
  four-leaf structure, context applicability, protected list/fence atoms,
  verification branches, no extras/markers/properties/duplicates, bounded
  completion counts and truthful usage. Internal HTTP attempts stay unknown.
- Recompute only the four context-emission counters from original responses'
  normalized core claims and original request context. Reuse the frozen
  request/structural counters and counter validator. Clearly label this as
  core-claim path evidence, not typed-claim evidence.

## Sequential implementation and verification

1. **Sol fix 1:** implement v2 gate and report in the pilot plus offline tests.
   Root independently verifies positive typed/partially untyped/fully untyped
   cases, wrong and invalid supplied types, conflicting hints, wrong source or
   context, absent verification branches, extras, and accounting failures.
   Reapply the new gate to retained evidence offline on Afrodite; export only
   safe metadata. This is a new-policy replay, not a revision of v1's result.
2. **Sol fix 2:** after gate acceptance, implement a fresh canary-only diagnostic
   with pinned source/transport/gate identities, private raw evidence, safe
   reports, complete failure accounting and verified process cleanup. Root
   reviews and tests it before execution.
3. **Root live verification:** one fresh GPT-6 Luna subscription canary, no more
   than 24 completion invocations, 120 seconds per invocation and 10 minutes
   overall, no rerolls. Keep the existing quota floor and token stop controls.
   No paid API fallback, LME question, full run, production memory, deployment,
   or changes to the stopped DeepSeek run. Stop and report any failure honestly.

## Status

Fix 1 accepted after independent root verification and a second Sol review.
The expanded offline suite passed **183 tests**. Root integration controls use
the actual frozen extraction modules, including missing, incorrect, invalid
and null types, incorrect source IDs/context, and shared producer identity.

The retained eight responses replay byte-for-byte against the original eight
requests on Afrodite under the new v2 gate: **pass**, 2/2 exact core claims,
2/4 optional type fields present, zero wrong/invalid types, exact valid core
execution path, four leaves. All 508 frozen source files verified; **zero new
model turns**. This does not change the historical v1 failure. Private replay
receipt: `/tmp/hymem-luna-canary-v2-bVxzG9rA/offline-replay.json`.

Accepted gate SHA256:
`0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0`.
Fix 2 accepted after independent root fault-injection tests and Sol review.
The standalone runner is
`tools/diagnostics/luna_subscription_canary.py`, SHA256
`98de9a447f7409e5efc69b94944f0e8ebbb1c2745baf00dcf8ea31362be17aed`.
It imports no LME adapter and opens no dataset. It checks frozen inventory
before candidate imports, persists private request evidence before dispatch,
retains known usage/uncertainty on failures, and verifies owned-process cleanup.
Root tests caught and required correction of a final-progress write failure
that could otherwise leave a success flag set. The expanded final diagnostic
suite passed **193 tests**, and the unchanged canonical canary/extraction-retry
suite passed **312 tests** (505 total). The latter needed `ijson` installed only
in a disposable local test dependency directory; all three initial dependency
failures passed after that correction.

## Fresh live verification — passed

One fresh GPT-6 Luna subscription canary completed **8 calls in 44.196 seconds**,
68,388 reported tokens, usage complete. Two exact core claims, four leaves,
valid and exact core-context execution path; two optional types present and
two absent; zero incorrect/invalid supplied types. No reroll or model change.

The reviewed runner ran in its own Afrodite systemd user service, with
`RuntimeMaxSec=590`, `TimeoutStopSec=10`, `KillMode=control-group`, and `Restart=no`.
The runner also enforced 24 calls, 120 seconds per invocation and a 590-second
wall deadline. Root independently verified all journal entries against the
retained requests/responses, replayed the fresh responses with byte-identical
requests and no new inference, and obtained exactly the live gate result.
All 508 frozen files still match. All nine App Server process groups (preflight
plus eight invocations) are absent, the run cgroup has zero processes, and its
completed service has been stopped/inactivated with zero restarts.

Private metadata: `/tmp/hymem-luna-canary-v2-bVxzG9rA/root-postflight.json`.
Private evidence SHA256:
`830a62010ca03cd162a024be739c2bddd869b03f23d4b8e21142f8340e88067c`.
Public receipt: `docs/plans/2026-09-28-luna-canary-v2-receipt.json`.

**Result:** this experimental canary contract mismatch is fixed and freshly
verified. The original failed v1 evidence remains unchanged. No LME question,
production change, paid API fallback, or canonical R9 rerun occurred. This
does not establish LME accuracy, full-run reliability, complete type coverage,
or equivalence to API temperature/output-cap/JSON-mode controls. Provider
internal HTTP attempts remain unknown.
