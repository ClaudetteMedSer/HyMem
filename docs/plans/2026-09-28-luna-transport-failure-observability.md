# Luna terminal transport failure: evidence and sequential repair

## Established facts and limits

The grounding four-question pilot stopped after 1,919 admitted calls, before
any question finished. All workers and the owning service cleaned up. Its
strict canary had passed; no extraction or summary failure is established by
the terminal cause. Usage for one failed turn is unknown.

The immutable v2 warm adapter maps unapproved exception codes to `fixed_other`
and discards the original exception. Its last-RPC field remains `turn/start`
while consuming turn events. Root and a separate Sol read-only diagnosis agree
that the original cause cannot be identified from the retained standard
artifacts. No traceback or fixed error marker was found on-host. Do not label
the incident a quota error, server outage, unsupported notification or model
quality regression without new evidence.

The official App Server documentation describes an `error` notification and
structured `codexErrorInfo`. Root generated public JSON schemas offline from
the exact pinned 0.158.0 binary: `ErrorNotification` requires threadId, turnId,
willRetry and error. The accepted v2 parser rejects this documented event and
its safe-code allowlist omits it, producing `fixed_other` in a synthetic replay.
That is a reproducible diagnostic defect, not proof that this event caused the
recorded live failure. Official reference:
https://learn.chatgpt.com/docs/app-server#errors

## Plan and gates

1. Preserve the terminal run, receipts and source pins. No automatic resume,
   restart or reroll. Record safe counts, incomplete accounting and independently
   verified cleanup in the run plan.
2. A new Sol agent implements a separately versioned v3 transport diagnostic
   repair, leaving all pinned v2 files untouched. Retain bounded, finite failure
   families/event categories, structured error enum/HTTP status/boolean retry
   metadata and an accurate event phase. Never retain free-form server messages,
   unknown method names, text, identifiers, credentials or paths in public
   diagnostics. Validate thread/turn identity before attributing structured
   server errors. Unknown/malformed events must still fail closed. No new
   client retry, implicit model change, relaxed tool isolation, quota bypass,
   accounting invention or changed model-output/quality gates.
3. Root independently reproduces the v2 diagnostic loss offline, reviews v3,
   and tests permanent/transient error notifications, malformed/foreign events,
   unknown strings containing synthetic secrets, failure-vs-cleanup precedence,
   phase accuracy, accounting and successful warm lifecycle. Sol must not run
   model calls, deploy, mutate production or launch experiments.
4. Only after root acceptance, separately implement source-bound runner and
   metadata-reader integration with new immutable identities; independently
   test stdin loading, strict validation and terminal accounting/cleanup. An
   observation-only fix is not evidence that the underlying live cause is fixed.
5. Any further paid diagnostic or same-four-question run needs an explicit
   evidence-based purpose and a new receipt. No full-500, cap increases, blind
   retry, source/prompt/model changes or production deployment. External quota
   or a materially broader requirement means pause and request direction.

## Status

Terminal evidence preserved. Root verified the public protocol schema using
offline code generation; no model call was made. The original live cause remains
unresolved.

## Step 2 accepted: diagnostic adapter

Separate Sol implemented `benchmarks/codex_subscription_warm_v3.py`; root reviewed
and requested corrections before acceptance: exact sibling-byte loading rather
than a possibly cached package import; finite nested-field projection; total
handling of malformed JSON; retention of the original operational stop code;
and optional HTTP status semantics matching the pinned binary's schema.

Root independently wrote 173 adversarial/source-binding/wire tests. They cover
error events before and after the actual start response, matching and foreign
turns, no new turn/retry, RPC errors versus event errors, malformed JSON shapes,
synthetic-secret injection, stale metadata, and successful output/usage. Existing
Sol tests also cover first-fault precedence over cleanup and failed-turn usage.
The final root-run combined suite passed 352 tests, including unchanged v2/base,
concurrent transport, grounding integration and capacity guards.

Accepted adapter SHA256:
`0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d`.
Pinned v2/base/concurrent files and the failed run's reader remain byte-identical.
The adapter does not accept new notifications or continue/retry failed turns;
even a `willRetry=true` event retains the prior fail-closed behavior. It records
finite structured metadata without raw messages, identifiers, paths or text.
Failed usage remains unknown. Official OpenAI documentation and the exact
binary's offline schemas informed diagnostic classification, not a claim of
improved model reliability. No model calls or deployment occurred.

Step 4 runner/reader integration is next, using a new Sol agent and fresh files.
The public serializer is a safe projection, not a complete integrity validator;
the reader must still validate required fields and reject malformed metadata.
No paid run is launched or resumed by this acceptance.

Step 4 is in progress with separate Sol agent
`/root/sol_observed_lme_integration_v1`, owning only new
`luna_observed_lme.py`, `luna_observed_lme_launch.py`,
`luna_observed_lme_progress.py` and its integration tests. Root's initial review
flagged two draft hazards to correct before acceptance: v3 must execute only
after the frozen candidate has been verified/loaded, and its budget/client/
ConcurrentStop must all come from the same private-module lineage. Mixing two
independently executed exception classes can break fail-closed propagation.
These are draft-review findings, not deployed failures. Integration is NOT yet
accepted. Continue independent root review, isolated flat-bundle and real-client
fault tests before any new preparation, receipt or model call.

## Step 4 accepted: source-bound diagnostic integration

Root independently reviewed the final Sol implementation and ran the combined
base, concurrent, warm v2/v3, grounding, resource-policy and observed integration
suites: **606 passed**, no skips or failures. This includes 246 root-authored
integration/privacy/terminal fault controls, actual frozen 508-file loading,
client-to-budget-to-runner-to-reader failure propagation, source-pin checks,
SSH-stdin bootstrap, ambiguous dispatch consuming its one-shot marker, and
private empty workspace checks. All prior pinned sources remain unchanged.

Accepted new SHA256 identities:

- Runner: `2062a8816e8e22214a1e7febc4898322210c381abdf9bb0f34e802d61d111550`
- Launcher: `1f503eae5121306a9ca70a6d1286e7b1cb898186ac49979403ff835367e3df3e`
- SSH-stdin reader: `4c8862733d3950d8437fe2ad2376349ff0ce2b9935ea5561a79c52bfd611f303`

The reader requires all prior terminal health, correctness validity, accounting
and cleanup gates. It reports finite failure metadata, never free-form server
messages. The runner, client and budget share the same private module lineage.
The launcher verifies every source before executing the new runner. No new
notification is accepted, and failed turns are not retried.

## Step 5: justified fresh diagnostic, not a reliability claim

The next same-four-question run has one evidence-based purpose: recover the
finite protocol/provider failure classification lost by the preceding run, or
observe validated completion under the unchanged strict policy. The original
failure is not retrospectively recoverable. Synthetic controls establish that
the new instrumentation can preserve the classification; they do not establish
that the live fault has been fixed. This is not a blind attempt to improve the
score. There will be no automatic reroll if it fails or happens to pass.

Use the unchanged grounded candidate, same questions/order, GPT-6 Luna
subscription identity and original caps/containment. Fail closed on the first
underlying error; no additional retry or new call is allowed after campaign
stop. If an external quota/auth limitation is established, pause and request
direction. The accepted paired grounding test will not be rerun. Create a new
immutable receipt and bind the monitor before launch; record identities and
metadata-only command in `2026-09-28-luna-lme-observed-run.md`. No preparation or
launch is implied by this plan entry; record actual outcomes separately.

## Outcome and required scope decision

The new receipt-bound observed run was dispatched once. It stopped at the
canary after eight returned calls / 69,130 known tokens, before any question
started; usage reconciled and root independently verified process cleanup.
The new transport instrumentation recorded no transport fault. Therefore the
original long-run `fixed_other` cause is still unknown. See the observed-run
plan for immutable identities and exact terminal evidence.

Root replayed all eight saved responses offline, with exact canonical UTF-8
request-field matches, reproducing the recorded rejection. One prose triple
has the right subject/object/polarity/source but wrong predicate. A separate
Sol read-only review agrees: this is an observed model semantic error, not a
proved parser, transport or canary-oracle bug. Do not claim a quantified
stochastic failure rate from this single result. Root's replay helper SHA256:
`bded053a9cfdb6d2d698f5478cad28e395fec1cb077227fcba79b775cb796218`.

The existing prompt already states the semantic rule. The omission pass only
adds missed items and cannot remove/reclassify a structurally valid primary
claim. No automatic reroll or fixture-specific prompt tuning is justified.
The next proposed repair is materially broader than the accepted reporting
change and prompt-only correction, so pause the monitor and ask for direction.

If the user approves a source-grounded semantic-validation contract:

1. Specify bounded acceptance/rejection semantics, exact source/context
   ownership and behavior on unsupported, ambiguous and implicitly supported
   predicates. Do not silently drop a claim or certify a partial unit complete.
2. Separate Sol implementation of the narrow contract, including a new cache
   identity and atomic failure handling; no benchmark-oracle filtering or
   fixture-specific rules. Root independently reviews and reproduces it.
3. A separate Sol supplies invented precision/recall, polarity, contextual
   reference and multi-source controls; root checks false rejections as well as
   unsupported acceptance. Preserve all historical failed evidence. Any canary
   path accounting change must be explicit, versioned and equally strict.
4. Only after offline acceptance, predeclare a bounded comparative diagnostic
   and new immutable receipt under unchanged model/auth/spend/resource limits.
   Retain all outcomes; finite passing controls are not a reliability guarantee.
   Only then consider another same-four-question pilot, never full-500 or
   production deployment under this scope.
