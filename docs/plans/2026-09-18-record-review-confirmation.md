# Source-first record-review confirmation

## Current state

User approved the specifically described 32-call confirmation with the literal
reply `Run it`. Local preparation, the isolated Linux rehearsal, independent
receipt audit and credential-free exact-entry preflight passed. The approved
confirmation completed **32 calls / 32 HTTP attempts**, without retries.
Execution/accounting passed, but the semantic gate **failed: 29/32 controls**
(15/16 in repetition one, 14/16 in repetition two). No production deployment,
restart, store change or full LME run occurred. The candidate is not cleared for
runtime adoption or LME.

The subsequent [source-owned retention inventory](2026-09-18-retention-inventory.md)
is implemented and verified offline (2,361 passing related tests, four intentional
skips). It does not change this failed confirmation result and has not been
queried against DeepSeek or integrated into the runtime.

## Completed result — September 18

Collection ran **07:48:06.155–07:50:32.156 UTC** (146.001 seconds). All 32
responses finished with `stop`, parsed as strict valid reviews, and had known
prompt/completion usage. No truncated, malformed or uncertain scored targets;
no transport, accounting or worker-cleanup failures. The live container exited
zero, PID zero, without OOM. All 309 exported receipt files were hash-verified.
An independent agent reproduced the full frozen score byte-for-byte; root
verified equality and separately reviewed the failed raw responses. All frozen
input and helper hashes were rechecked unchanged after collection.
A second independent, standard-library-only reconciliation imported neither
collector nor scorer and checked every exported file, scheduled request,
attempt, worker lifecycle, authority deadline, mount and token sum. Root read
and reran it; its report was byte-identical. The minimum send-time authority
headroom was 6,946 seconds. Reconciliation script:
`/private/tmp/hymem-confirmation-reconcile-H76H4y.py`; report:
`/private/tmp/hymem-confirmation-reconciliation-H76H4y.json`.

| Frozen measure | Repetition 1 | Repetition 2 |
| --- | ---: | ---: |
| Complete control passes | 15/16 | 14/16 |
| Primary grounding | 10/10 | 10/10 |
| Primary retention | 5/6 | 4/6 |
| Auxiliary grounding | 18/18 | 18/18 |
| Auxiliary retention | 4/4 | 4/4 |
| Faithful whole scopes with no veto | 8/8 | 8/8 |

Three unmasked false accepts remain, all on omitted constraints:

- `procedure-confirmation-written-room-confirmation-defective`, repetitions
  one and two: the candidate deletes the written room-allocation prerequisite.
  Check `r1` nevertheless reports `retained` and cites all five remaining
  candidate fields. None contains the prerequisite. Every other scheduled check
  accepts, so no separate veto saves the record.
- `confirmation-repair-cafe-exclusion-defective`, repetition two: the candidate
  deletes the prohibition on reimbursement of whole replacement devices.
  Check `r4` reports `retained` against the remaining summary. Repetition one
  correctly reports `omitted`. The second response accepts the whole scope.

Root reread the exact canonical source windows, candidate fields, scheduled
checks and raw responses for both failed pairs and their faithful controls.
The required information is visibly absent; these are not reference-plane,
parser, label or truncation errors. The frozen labels remain unchanged. No
result was excluded, repaired or rerolled.

Independent raw-response review also found that some already-rejected
wrong-actor/wrong-name candidates still receive `retained` on an unlabelled
`material_facts` check. Other checks catch those cases. This is diagnostic
evidence of coarse retention judgments, not a new post-hoc gold target: the
frozen score stays 29/32. Passing labelled targets must not be described as
proof that every unlabelled judgment is correct.

All responses reported `deepseek-flash` and fingerprint
`aeb56401ca74e127821c4f9126dcb669`; neither establishes a weights-version pin.
Usage: **134,024 prompt + 9,668 completion = 143,692 tokens**. Reasoning-token
usage was absent in all 32 responses: unknown, not zero. Reasoning text was not
retained. The 32/32 authority is spent; no additional run was started.

### Interpretation and next boundary

Source-first layout passed the earlier development examples but is **not
sufficient** to catch important omissions on this frozen confirmation set.
The remaining weak point is the single source/facet-wide retention verdict:
valid candidate-field IDs can reference preserved material without proving
that every source condition survived. Existing prompt text already explicitly
requires every material item and classifies a missing prohibition as omitted;
the model did not reliably follow that rule here.

This identifies an observable completeness failure, not the model's internal
cause. Do not infer that layout caused a regression: this confirmation used
different cases and has no alphabetical comparison arm. Nor are two draws per
case a population reliability estimate.

The next architectural candidate should make source-derived retention
obligations individually inspectable and require candidate evidence for each,
with explicit missing/uncertain outcomes and bounded call accounting. An
itemized ledger remains a hypothesis: checking span existence cannot by itself
prove semantic preservation or completeness of the extracted obligations.
Keep these failures as regression evidence; do not replace the current gate
with weaker acceptance rules or tune this set and call it fresh confirmation.
No such integration or additional paid test is included in this completed run.

## Frozen question and decision rules

The preceding development comparison passed 8/8 cases twice with source-first
presentation, versus 6/8 twice with alphabetical presentation. That result is
not evidence of LME readiness. This test uses the sixteen previously frozen,
unqueried AI-authored confirmation controls: twelve episode/summary cases and
four procedure cases. Sources, candidates, expected labels and scoring remain
unchanged. These are controls for known mechanisms, not a real LME holdout or
a representative population sample.

One source-first V4 candidate, two repetitions: at most **32 completions and
32 HTTP attempts** to `https://api.deepseek.com` using `deepseek-flash`, thinking
disabled, no reasoning-effort field, temperature zero, JSON-object output and
max_tokens 8192. Each invocation has 120 seconds plus two seconds for cleanup.
No retry, repair, reroll, resumed campaign, changed label or excluded result.
Semantic rejection remains in its original cell and does not stop independent
cases. Transport, accounting, identity, safety or cleanup failure halts the run.

Per repetition, keep separate denominators: primary grounding 10, primary
retention 6, auxiliary grounding 18, auxiliary retention 4. Conjunctive selectors
count once. All targets must match and all eight faithful complete scopes must
have no veto, including checks without exact gold. Both repetitions must pass.
Execution/accounting success is separate from semantic quality.

Root reread all sixteen sources/candidates/labels before collection. One narrow
design caveat was recorded prospectively: the cross-message defective case's
single grammatical-subject substitution affects both conjoined predicates, not
only chair-setting ownership. Its existing unsupported actor/assertion labels
remain justified; no case or label was changed. The test therefore cannot
isolate a single-assertion causal effect.

## Preparation and independent checks

Package: `/private/tmp/hymem-confirmation-run.H76H4y`.
Fresh target:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-record-confirmation-20260918-H76H4y`.

Root independently passed **280 local tests**: 89 core preparation/scope/expiry
tests, 79 mocked live-control tests, 63 receipt/scoring tests and 49 ancillary
inventory/receipt controls. Every frozen
case and gold file matches the original sixteen-case collection byte-for-byte.
An independent reviewer also passed all 63 auditor tests and independently
reran the 79 live-helper tests. No test in these counts invokes a provider.

The final transfer contains exactly 564 allowlisted files, including its policy.
An independent reviewer verified every file hash and the recursive inventory,
pre-test absence of paid authority, and private mount-propagation checks.

Live transport now checks expiry at the immediate request hook before reserving
an attempt or sending. The host also requires over 4,204 seconds of authority
lifetime before create and again before start. Mock tests cover time crossing
expiry during SDK construction. The existing one-shot limits, isolated owned
workers, known-usage requirements and no-retry policy remain.

Before any paid call, the fresh credential-free, network-none Linux rehearsal
passed **137 tests, one explicit local-origin skip, and 79 live-control mocks**.
Root audited all 64 dry/loopback worker receipts, exact request bindings,
deadlines, accounting and cleanup; the container exited zero with no remaining
process or out-of-memory event. Root then issued fresh one-shot authority and
the exact-entry credential-free preflight passed. The private key is mounted read-only only for the
approved live entry; it is never exported or printed. The image's Hermes home
is masked and production memory is not mounted.

A separate prospective review of all sixteen actual sources/candidates/labels
found no incorrect gold or additional objective ambiguity. The existing
cross-message caveat and all frozen denominators were confirmed independently.

## Frozen pins

- Original sixteen-case collection: `9da229582e77eb3ec60c4105feedb6c2ac5eb3ae3f2c66c262d4cfa3d39a9ab1`.
- Inputs: `9fd484b7a81c4bfb48a558633146fec827a6f736af3c38a6a19e39839ef500ae`.
- Sources: `5826a4eb2f2dac84fd90ead6165892a229b981d55dde20d385b1d8f688b693b8`.
- Schedule: `329a2d7307183b7c68d9aa18c0b31192ef62d33e38d00d5113385e2ddf505cf0`.
- Protocol: `8bdf9f7ee96d853847ca003d34920fd3620f25c294dd41491c13aef714970f96`.
- Offline helpers: `28f939b30757ebc6e12efb8f50d9122be868cc7a26edb9ae313892df1fd5f329`.
- Live helpers: `811a225208338fbb9b563d2a1fc95ebc36eda1a8578e630045c67f2990ce8d7b`.
- Host launcher: `bbb9df156d22347254c2922411835430f76e266998d1f2d65f61c362d535a5cc`.
- Independent auditor: `46aefad1c218db21b7f0bc0de58bbe643a9e87c3c243529e140f4e58a1e50911`.
- Transfer policy: `0097feed6e5623c2111947eaad83ec60a67b24b6bbb6309caa916848626ead78`.
- Audited target proof: `578ffeef4539ca16b68c040fb36e767bcb2ea85d31223deb8c1206f39f796851`.
- Spent one-shot authority: `33f11c4f8127b04d9120e5a3421b7cfd91da9690c7d7cbde486501332d65dce2`.
- Live receipt archive: `100d885e622dbb7187d5fede6ec22933d22a0997fcf4b1bc8b68d6494380397f`.
- Live export manifest: `337b4de5c38bcaabcbf64e16e090b779048949e4b0b3f25aa5cac1db73a6d386`.
- Root live score: `11df158429313fa73b421da9f8e42f372e93aeef4b44c1f2c9135f176ea29358`.
- Independent reconciliation script: `8284d33ec6b6b6f466d5c5dcf0b07873ca3e8aab890a5ccccd498407533090ef`.
- Independent reconciliation report: `4dd663de0f7a4ad6c273a13eeaa5ca0c8e2e7fbc1434a96386c19d4c771d21ad`.

## Readiness boundary

Even a clean confirmation result does not authorize runtime adoption or establish
LME readiness. Whole-digest behavior, bounded runtime integration, durable
continuation/reopen, original failed Q1 and untouched real-adapter workload still
need separate verification. No such result is inferred from a passing harness.
