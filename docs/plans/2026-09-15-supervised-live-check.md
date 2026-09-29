# Supervised live diagnostic — executed once, halted on format rejection

The user explicitly approved this bounded diagnostic with "Yes" immediately
after root asked to prepare and run four problem cases plus eight controls on
Afrodite, capped at 20 completion calls/60 HTTP attempts, with 120-second enforced
deadlines, no automatic reruns, no production changes and no full LME run.
This is new approval, not reuse of the preceding offline approval or closed v10
campaign. The signed execution metadata is still contingent on fresh preparation
and independent acceptance of the credential-free rehearsal.

## Read-only readiness check

On 2026-09-15, root rechecked the current application and diagnostic against the
accepted 435-test input manifest: no byte drift. On Afrodite, the retained
benchmark-only stage remains available with all 513 candidate source hashes
matching, its four incident cases, and eight semantic controls. Its one retained
benchmark database still matches its recorded hash and has no WAL/SHM/journal
sidecars. The retained full-gate verdict records 7,406 accepted tests; that suite
was not rerun during this readiness check. No production database was opened,
credentials accessed, services changed, or provider requests made.

## Proposed bounded experiment

- Same configured `deepseek-v4-flash` model at `https://api.deepseek.com`;
  preserve source windows, prompts, temperature, token and retry semantics.
- Four digest pipeline cases, each followed by its faithful and one-defect
  verifier controls: at most 12 invocations in the fixed existing order.
- At most 20 logical completion calls and 60 HTTP attempts, including the
  maintained client's bounded retries. These are request ceilings, not a price
  quote. Interrupted usage and billing remain unknown unless independently known.
- One enforced 120-second deadline per whole invocation, including startup,
  primary extraction, optional compaction, fidelity verification and close;
  a separate two-second cleanup allowance. No per-stage deadline reset.
- Stop on the first timeout, unsafe cleanup, invalid receipt, rejected pipeline
  or control mismatch. No automatic reroll, campaign resume or recycled budget.
- Use only immutable benchmark source, private diagnostic outputs and the owned
  Python worker boundary inside the target container. No publication, production
  store access, deployment, restart or full benchmark execution.

## Work after explicit approval

1. Create a fresh canonical stage and revalidate source, runtime and input
   identities. Preserve the closed v10 campaign and its authority/evidence.
2. Prepare and independently review the supervised stage and a credential-free,
   zero-provider rehearsal with complete worker reaping and immutable source.
   Correct and verify any preparation defect before live execution.
3. Bind fresh authority to the exact accepted stage, helpers, plan, fixtures,
   rehearsal and the user's new approval. Never copy old consent/authority.
4. Execute the single bounded campaign. Independently reconcile attempts,
   returned responses, deadlines, source hashes and terminal receipts; review
   source fidelity of actual model output separately from mechanical success.
5. Report completion or the first blocker honestly. A diagnostic pass is not a
   full LME score; a full LME run remains a distinct, more expensive next step.

## Current preparation

Fresh private work: `/private/tmp/hymem-supervised-live-v12.cSFuAr`.
Fresh Afrodite stage:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/owned-digest-validation-20260915-4x1tmU`.
Canonical execution stage:
`/home/node/.hermes/benchmarks/owned-digest-validation-20260915-4x1tmU`.

The previously reviewed target guard was authenticated before a new read-only
preflight. It verified current runtime and production-source hashes, the accepted
513-file/7,406-test evidence and Hermes1 container identity. The new stage copies
only 519 explicitly listed source/input/evidence files: no database, old consent,
old authorization, prior paid output or failed campaign fence is copied.

A separate agent is implementing the actual-source dry rehearsal; another owns
the detached host transport and exit accounting. Root owns entry authentication,
credential handling, staging and independent review. The dry rehearsal uses
synthetic replies and fresh owned workers, not the live SDK worker; the latter
has the separately accepted 435-test Mac/Linux evidence. These are complementary
checks, not a claim of real-model semantic accuracy before live execution.

The preparation above completed before any credential access. Root independently
ran 72 preparation tests (zero failures/errors/skips), then accepted the isolated
Afrodite rehearsal: 12 tasks, 17 synthetic completions, zero provider attempts,
all workers reaped and all 519 staged input hashes unchanged. The rehearsal used
`--network none` and did not mount credentials. Root corrected only an audit
comparison that had assumed Docker preserves mount-array order; complete mount
entries and isolation settings matched after sorting by unique destination.

## Paid result — 2026-09-15

Fresh authority was bound to the user's current approval and exact accepted
helpers, stage, plan and rehearsal; its SHA-256 is
`303d4a39a43e995088f18e905feb168ca8784736c05bc2e7e72ee98123d5ac86`.
The detached Afrodite supervisor dispatched the campaign once. It halted after
the first pipeline case, `p-digest-0-blob`, as required by the first-failure rule.
The other 11 invocations, including all eight controls, were **not attempted**.

- Three completion calls, three HTTP attempts, all responses returned.
- Recorded usage: 7,873 prompt tokens + 721 completion tokens = 8,594 total.
  These are provider usage counters, not a verified monetary cost receipt.
- Whole invocation: approximately 8.08 seconds, within the 120-second deadline.
- Primary summary: 570 characters; summary-only compaction: 389 characters.
  The verifier received the exact compacted summary, not the rejected primary.
- The verifier supported both episode titles, both episode bodies, both episode
  formats and the summary's content. There were no procedure candidates.
  Only `summary_format` was `unsupported`, producing
  `summary_format_unsupported` at `fidelity_verification`.
- No summary, episode, procedure or source cursor was published. The captured
  result mechanically replayed exactly; all receipt hashes reconciled.
- Owned worker exit 0 and complete reaping/process-group absence are verified.
  Campaign/entry/Docker transport exit 2 consistently report the semantic halt;
  this was not an execution timeout or transport failure.
- Root rechecked all 513 candidate source files, 519 staged inputs and the
  immutable benchmark database: unchanged, with no database sidecars. Hermes1
  remains running. No production database was opened or service restarted.

The authenticated retained stage guard also passed a final read-only postflight:
installed runtime, production source, candidate source, container/image and home
binding all remain unchanged. Local `git diff --check` passed, and the six frozen
helper hashes still match their authorized values.

The stage remains closed and non-resumable. Unused budget is not authorization
for another draw. A full LME run was neither started nor authorized here.

## Independent reconciliation and remaining diagnosis

Target `root-live-audit.json` records a passing **receipt/safety audit**, not a
passing diagnostic. The diagnostic summary hash is
`abcc724bed266363755e3bd165d3e6eda120e2e919bc873706f5f6fca298947b`;
the worker commit hash is
`30a7fd4a619ae2fd2259d67456f5504b2e8cdc69fbf9b068b1be51401595a91c`.

Auto-review blocked exporting full benchmark plans/receipts to the laptop.
Root respected that boundary, audited the original content in place and emitted
only selected counts, booleans, controlled verdict categories and hashes. No
source windows or raw responses have been exported in this turn. Explicit user
permission was requested to inspect just the generated summary and verifier
verdict before deciding whether the format rejection is correct or a false
positive.

A separate read-only code audit identified a candidate contract inconsistency:
the primary summary prompt permits an "implicit subject", while the verifier
requires one complete grammatical sentence and rejects fragments. Application
code does not remove punctuation or truncate the repaired summary; it preserves
the returned text apart from surrounding whitespace. Existing format positives
use scripted verifier replies, so their passing results do not prove live model
agreement. This is a concrete policy inconsistency to investigate, **not yet a
demonstrated cause of this particular response**. Frozen helpers, prompts and
candidate source were not changed to make the diagnostic pass.

## Follow-up inspection and local repair

The user subsequently approved retrieving only the generated summary and verifier
verdict. Committed hashes were verified and credential patterns screened before
those fields were shown; no raw source windows or full request plans were exported.
Root and a separate reviewer agree that the 389-character summary meets the
verifier's format contract: explicit subjects, complete clauses joined by an
expressly allowed semicolon, one sentence and no Markdown/enclosing quotes.
The earlier implicit-subject hypothesis is therefore **not the cause of this
particular rejection**. This establishes one false-positive format verdict,
not its internal model cause, frequency or independently verified factual quality.

The user's following "Please fix" authorizes the local bounded adjudication repair
tracked in [the implementation plan](2026-09-15-format-adjudication.md).
The closed campaign above is not rerun or retrospectively marked successful.
