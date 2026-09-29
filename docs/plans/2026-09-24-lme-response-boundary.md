# LME response-boundary repair (R5 recovery design)

Status: R5 implemented on 2026-09-24 in a fresh isolated candidate after parent
acceptance of reconstructed R4. The focused safety gate passed 307 cases with
four explicitly declared raw-backend-not-applicable skips; no failures/errors.
The independently reconstructed full R5 suite subsequently passed 7,243 tests
with the same four declared skips and zero failures/errors. Exact inventories
were reconciled in `2026-09-24-parent-r5-full-aggregate.json`. The genuine
Afrodite SDK/stock-CLI preflight also passed without provider calls and exited
cleanly. The target-runtime gate passed 1,040 tests plus the same four declared
skips, with zero failures/errors, exact before/after inventories and clean
process-group shutdown. Fresh paid stock Q1 then completed correctly in about
27 minutes, with healthy item indexing and 10 explicitly degraded session
summaries. Strict archive/checkpoint/accounting/cleanup validation passed after
a separately tested verifier-only raw-versus-reconciled-row correction; no paid
rerun or artifact change was needed. See `2026-09-24-lme-q1-verification-result.md`.
Do not infer full-500 readiness or production deployment from this one-question
result.

The previous temporary R5 candidate and R4 inputs disappeared during the
usage-limit suspension. The new evidence is the additive patch15 and R5
manifest in `docs/patches/`, not the vanished paths or receipts. Main application
checkout and previous additive patches remain unchanged. No provider calls,
remote access or deployment were performed for this implementation.

## Evidence and starting point

Before suspension, the independent synthetic R3 transport baseline had 99
cases: 88 failures, 7 passes and 4 intentional backend-specific skips. This was
a **mixed model-startup/response baseline**, not a precise transport-defect
census: its official `deepseek-flash` fixture lacked deployment attestations,
which R3's earlier producer policy did not admit. It verified the R3 source
before and after and made no network calls. Its temporary
receipt/XML no longer exists locally; these are recorded prior results, not a
new verification claim. Former receipt SHA-256:
`5e831c52868f8b2e6faedb3718fe87dbdd8b433c91311d2cbf452e0db7353120`.
The retained test source remains available as
`tests/test_completion_response_admission.py`, SHA-256
`51df50ac15250ffdbaf294d18a8afe634f262295f9dd92942bf3964c1eb8ca49`.

Read-only inspection of the current dirty main checkout confirms that it already
contains most observed-response accounting changes, but not the typed recovery
boundary or canary compatibility. It is not a substitute for the reconstructed
R4 summary/index separation candidate. Main SDK no longer contains R4's
`HYMEM_LLM_EXTRA_BODY` extension, so copying its file wholesale would remove a
deployment customization. Likewise, do not replace R4 protocol validation with
main's older indexing-summary schema.

Inspected main source hashes on 2026-09-24:

| File | SHA-256 |
| --- | --- |
| `hymem/contrib/openai_client.py` | `38471a49322bfb5adc9aa89d50322b4fb4d34c71dc3e7268d87af90eaa50976c` |
| `hymem/extraction/llm.py` | `c10e2187bc17be70800cf7d0a698d6b5bec790848c375e25faf38f3c89762fe2` |
| `benchmarks/longmemeval_adapter.py` | `0cb969b160407e35acb3fd928477b9565f818c033beb809fe06b7d4e455f9eb5` |

## Portable transport changes

Apply these narrowly to accepted R4, preserving its active-model policy, exact
request bodies, immutable runtime configuration, lifecycle/cleanup guards and
summary-degradation schema.

1. SDK initialization gains `_accounted_response_attempts`. Capture the caller's
   absolute deadline once per logical completion, before any provider attempt;
   reuse it for every retry timeout and final admission. Ambient context changes
   in provider callbacks must not clear or extend that owning deadline.
2. Starting a provider attempt invalidates token-usage exactness while that
   attempt is outstanding. A transport failure makes aggregate usage unknown
   permanently, even if a later retry provides known tokens. Maintain measured
   known subtotals without claiming that the missing cost was zero.
3. Account every received reply once, outside the transport retry loop, before
   post-return deadline, identity or content checks. Validate all three usage
   counts as finite nonnegative integers (not booleans) and require
   total = prompt + completion. Handle oversized numeric values safely. If
   integral floats are accepted, normalize them to integers for strict canary
   telemetry. Exactness requires all started attempts to have an accounted
   reply and no prior unknown attempt; concurrent in-flight attempts cannot
   produce an exact snapshot.
4. SDK admission requires exactly one choice, exact `finish_reason == "stop"`,
   and string content. Do not introduce JSON or nonempty-content policy here;
   extraction/reader validators own those contracts. `call_count` and
   `successful_responses` count admitted completions, not paid replies.
5. The raw LME reader/judge client accounts received JSON before admission and
   uses a typed nonretryable response rejection for invalid JSON/envelope,
   finish or content. It retains its existing first-choice policy when `n > 1`
   was explicitly requested: another choice cannot rescue a rejected first
   choice. Provider/HTTP failures keep the existing bounded retry policy.

Main's existing implementation is a reference for these hunks only. Its 99-case
test file exercises non-stop string rejection, paid rejected usage, malformed
envelopes, permanently unknown usage, concurrent attempts, multiple choices,
late/changed deadlines and post-response identity failure. It does **not** prove
the downstream extraction/recovery contracts below. No fresh test execution
is claimed by this design note.

## Typed length evidence and downstream preservation

Introduce a shared `LLMResponseError(RuntimeError)` and narrow
`LLMOutputTruncatedError(LLMResponseError)` in `hymem/extraction/llm.py`.
The latter has a constant safe message and carries **no partial text**, provider
payload, source excerpt or item authority. Emit it only for a trusted SDK
single-choice reply with exact string finish `length`, after paid usage,
owning-deadline and integrity checks. Multiple-choice, content-filter, unknown
finish, malformed envelope and transport/identity errors cannot masquerade as
length evidence. Never classify by exception-message substrings.

Changing finish admission alone introduces regressions:

- **Chunk extraction:** generic client exceptions become `call_failure`, which
  is deliberately not split-recoverable. Previously cut JSON took the bounded
  source subdivision path. Catch only typed length before the generic handler;
  return existing `incomplete_response` with constant detail
  `provider:finish_length` and no raw output. Keep existing call/attempt/depth
  ceilings, source boundaries, atom protection, omission verification and
  all-or-nothing publication. A truncated omission check cannot publish the
  earlier primary unchecked. Content-filter/identity/unknown errors remain
  non-split failures; cancellation/deadline BaseExceptions escape unchanged.
- **Digest primary:** truncated primary remains a failed unit, with no rescued
  items, summary or cursor. Its existing primary completion-failure path still
  holds the original authority and shrinks source on a later bounded attempt.
- **Digest summary repair:** catch typed length only at the repair call and
  produce existing bounded `output_truncated` at stage `summary_compaction`.
  Strict callers retain atomic failure. Separated callers retain independently
  validated primary episodes/procedures and exact source authority, with
  `summary=None` and `summary_failure_reason="output_truncated"`. Do not change
  source bytes, accept partial repair text, fabricate coverage or clear a prior
  summary gap. All other completion, identity and accounting errors stay fatal.
- **Independent summary recovery worker:** known typed length yields a held
  `output_truncated` job, with its already-reserved durable attempt retained,
  no draft/frontier advancement, and all lease/deadline/source guards still
  executed. Other failures keep existing fatal behavior and cleanup.
- **Facts:** no minimum change is needed. In the accepted R3/R4 design, cut JSON
  was `invalid_json`, not immediate capacity recovery. Both that rejection and
  a primary completion exception hold the unit and shrink the next-cycle input
  using the existing retry counter. Only a returned explicit `complete:false`
  empty envelope or fully valid over-cap set authorizes the existing immediate
  one-call, genuinely smaller source-prefix retry. Keep this distinction and
  verify it against rebuilt R4 before leaving facts untouched.

Seal the new error classes in SDK helper/runtime integrity checks, and include
the shared implementation in its import-time fingerprint. Phase-1 currently
slices the LLM module at `LLMRequest`/`measure_provider_attempts`; explicitly add
the typed error roots or prove equivalent complete fingerprint closure. Ensure
digest/summary-recovery call-site provenance also commits the imported boundary.
New implementation identities may invalidate derived work; never relabel old
producer/history records as current.

## Canary v18 and historical v17

Current canary v17 intentionally allows recovery within 24 completions / 72
provider attempts, not just its normal eight-call path. Its validator currently
requires admitted calls == logical completion calls for a pass. After typed
length recovery, source/claim evidence can be valid while that equality fails.
Do not merely remove the equality or count a rejected response as admitted.

For new v18, add one explicit nonnegative typed-length rejection count to the
recorded execution evidence. `_RecordingClient` is the observation point:
requests are recorded before delegate completion, successful responses after
it, and only the specific typed error increments the new count. Do not retain
partial text. Cross-check actual requests/returns/rejections with completion
counts and delegate telemetry. For a pass require admitted calls + proved typed
length rejections == logical completion calls. Failed reports may additionally
contain unrecovered generic failures, but cannot exceed their counted attempts.
Keep every fixture claim, exact context, list/fence atom, emission, source-ID,
minimum-call and 24/72 resource bound. Reject impossible or unknown fields.

Historical archive validation must not rebuild a past extraction identity from
today's code. The existing model-only archive path still calls current canary
policy/report validation and would reject old contracts after this change.

- Use a separate archive-only validator with an immutable v17 non-runtime
  policy template and structural validation of its recorded contract binding:
  exact keys `schema`, `prompt_version`, `identity`; known schema
  `hymem-extraction-contract-sha256-v1`; bounded trimmed prompt label; canonical
  SHA-256 identity spelling; exact policy/config/report cross-links.
- Preserve the known v17 key sets, fixed claims/context/caps and its original
  passed usage equality. Only v18 permits typed-length accounting. Reject
  unknown versions or mixed-version fields; never rewrite a v17 artifact to v18.
- A private shared report-validation core may accept an already-validated
  known-version policy context. Public current validators remain strict; the
  context cannot become a user-controlled bypass. Claim validation currently
  reconstructs `extraction_canary_policy()` and must instead use the validated
  context's fixed expected claims on this private path.
- Thread archive-only selection through both LME config binding and each
  execution-segment report. Maintain `historical_commitment_only` assurance and
  `live_execution_eligible=False`. Live admission, run/resume, strict export and
  current validators must not accept the historical route as fresh evidence.

## Verification and handoff gates

1. Independently accept rebuilt R4, then create and pin an isolated R5 candidate.
   Preserve R4 evidence and all additive patches 00–13. Record exact source,
   tests and auxiliary files, not a stale hard-coded file count.
2. Run the true synthetic response-admission baseline against accepted rebuilt
   R4, whose model admission matches the fixtures, then run it after R5.
   Preserve failing and passing receipts separately; do not attribute the old
   mixed R3 failure counts exclusively to transport behavior.
3. Exercise real maintained SDK dispatch with synthetic provider replies through
   real chunk splitting/omission, digest strict/separated repair, summary worker
   and configured canary. Test length at primary/verification boundaries;
   no raw partial publication; exact accounting; content-filter/unknown/identity
   failures not downgraded; deadline precedence; and unchanged bounded call caps.
4. Test v17 historical acceptance without mutation, wrong hashes/cross-links,
   tampered claims/context/counters, mixed/unknown versions, and rejection from
   live/current/resume/export routes. Test v18 admitted-plus-typed equality and
   rejection of fabricated deficits.
5. Run related SDK, provider-attempt, indexing-deadline, extraction, digest
   separation/publication, summary-recovery, canary, LME protocol/registry and
   client-cleanup gates, then parent review and full candidate verification.
6. Preserve R4's `HYMEM_LLM_EXTRA_BODY` extension byte-for-byte. Its per-call
   merge is not fully represented by the public effective-body identity;
   canonical benchmark preflight must explicitly require it to be absent.
   Retained helper transport verification is not proof of stock-client behavior.
7. Only after review/freeze may the parent run further authorized live tests.
   No claim that Q1 or full LME is ready follows from this design or unit tests.

## Fresh implementation receipt

- Accepted reconstructed R4's true response-admission baseline was 88 failures,
  seven passes and four intentional raw-backend skips out of 99 cases. R5's
  unchanged 99-case test file now passes all 95 applicable cases. This is the
  meaningful before/after comparison; the older R3 model-mismatched baseline
  above is not used as a transport-defect census.
- `2026-09-24-lme-15-response-boundary.patch` changes ten source files (one new),
  adds three regression files, narrowly repairs three old SDK/canary fixtures,
  and adds the genuine all-synthetic R3 v17 artifact as an auxiliary test asset.
  The response-admission file remains byte-identical at the recorded hash.
- `2026-09-24-lme-independent-summary-indexing-r5-manifest.json` pins all 467
  files: 231 source, 227 test and nine auxiliary. Its fresh collection contains
  7,247 exact node IDs and declares exactly four permitted skip IDs. The source
  and collection inventories were checked before and after collection/testing.
- The initial broad gate found only obsolete synthetic success responses that
  lacked `finish_reason="stop"`, and a literal expected canary map missing its
  new zero-valued counter. Corrections retain the original assertion bodies.
  The late SDK deadline test now additionally proves four observed billed tokens
  while retaining two attempts, one admitted response and deadline failure.
  The unrelated embedding transport's unavailable-token assertion is unchanged.
- v18 preserves the eight admitted verification responses minimum and bounds
  deduplicated claim emissions by both admitted texts and corresponding exact
  context requests. These new rules do not retroactively alter v17 evidence.
- Facts and semantic-generation implementation files are unchanged from R4.
  SDK extra-body customization is preserved. Canonical live preflight must
  still require `HYMEM_LLM_EXTRA_BODY` to be absent.
