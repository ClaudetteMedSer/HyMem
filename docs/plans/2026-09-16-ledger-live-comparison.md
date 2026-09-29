# Fixed-candidate evidence-ledger comparison

## Scope and frozen protocol

The user approved a fresh benchmark-only comparison at `api.deepseek.com` using
`deepseek-v4-flash`: 33 cases, baseline and ledger arms, two repetitions, at most
238 paid completions / 714 HTTP attempts. Each invocation has a 120-second
absolute deadline and two-second owned-worker cleanup allowance. No production
memory, deployment, restart, full LME, model rerolls or campaign resume.

The 21 previous fixed-candidate comparison cases retain byte-identical payloads
and baseline requests. Twelve invented faithful/defective controls add identity,
negation/outcome, exclusivity, own citations, boundary-context and procedure-step
checks. These are development cases, not a representative or held-out LME set.
Both arms keep temperature zero, JSON mode and a 3,072-token request cap.
The only intentional arm change is the ledger verification protocol.

There are 132 invocations: 66 baseline calls and 172 ledger calls. AB/BA ordering
alternates within paired case/repetition blocks, balancing which arm goes first.
All predetermined repetitions count. Malformed or rejecting model output is
retained and does not stop independent later cases. Infrastructure/accounting,
deadline, source-identity and worker-cleanup failures stop the campaign; unused
reserved capacity is not permission to rerun it.

Separate implementation and root review cover the adapter. A separate reviewer
accepted all 12 new family-level gold labels. Root retained the previous 28
target labels unchanged, producing 40 preregistered targets: 21 supported and
19 unsupported, or 80 target decisions per arm over two repetitions.

Score the designated verdict family, not whole-request rejection. For a valid
ledger, any unsupported claim makes its family unsupported; otherwise any
uncertain claim makes it uncertain; only nonempty, all-supported claims yield
model-supported. An empty family is uncertain, not vacuously supported. Invalid
ledger structure invalidates every family in that scope without salvaging a
subset. Keep malformed, uncertain, false-accept and false-rejection counts
separate. Untargeted families remain unscored, with collateral vetoes visible.

Exact quotes prove location and authority, not entailment. Parsed claims and
quotes require source-grounded review even if every structural check succeeds.
`semantic_verified` and `publication_authorized` remain false throughout.
The context control combines context-authority and discussion-to-joining errors;
the identity control combines missing attribution and prior-summary leakage.
Success on either cannot identify a unique causal mechanism. Procedure results
do not establish safety-complete extraction of every omitted source prohibition.

## Preparation status

- Frozen application source retains the prior 2,117-test offline gate: 468
  Python/SQL files, including four new ledger source/test files.
- Agent adapter tests plus independent root controls: 229 passed locally.
- Real local owned-worker rehearsal passed: 132 dry invocations / 238 synthetic
  completions, then 132 localhost invocations / 238 dummy HTTP responses. All
  worker groups and the local server were reaped; zero external-provider calls
  and no credential reads. The initial sandbox denied the localhost listener;
  the explicit loopback-only continuation passed after permission review.
- Independent frozen-parser replay matched all 476 request bytes across both
  rehearsal modes; none of the invented responses was mistaken for support.
  The source snapshot remains byte-identical to all 468 current repository
  inputs. The separate network-disabled Linux gate on Afrodite subsequently
  passed the same 229 tests, both 132-invocation rehearsals and all 476 frozen
  request replays; its container exited zero, without an OOM.
- Afrodite's Hermes1 container and image identities match the prior verified
  deployment. A new private diagnostic stage was created, without touching
  production services or memory.
- Source upload was initially blocked by the permission review: the paid-run
  approval did not explicitly cover internal source-code transfer to Afrodite.
  The user subsequently explicitly approved that transfer. The whitelisted
  archive and helpers were uploaded; the remote preparation verified all 468
  source files, fixtures, gold and helper hashes. No credential files or
  production memory were transferred. The network-disabled Linux gate started
  in a separate read-only-root container without production-home or credential
  mounts.

Local stage: `/private/tmp/hymem-ledger-live.3WQhGg`.
Remote stage:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-ledger-comparison-20260916-3WQhGg`.

Frozen identities:

- Fixtures: `09a78b15323afe59a5a395cf4480e93ca583b175463b9c90314ccbd1e24eb2b7`.
- Source manifest: `46db58197f907a52d00992e5adfa1f8d65e61f170f70a60da76ab3757dbb6d85`.
- Source/fixture archive: `da5f8c0f0c9d70b603015e0ae59229cb62ba10667416ec5cbd6ec69026d4fb51`.
- Gold labels: `d739a22d4166ecff1530544d07ed8c63fbdb960cb8251b4e35760de43fc2830a`.
- Independent fresh review: `f5b281a31bed02b7d40b5749226de4745a225f38c1d891bf565a1afd5a7731d0`.
- Helpers: `166d09ab3708cbd7b0873e55e2758ffb7f1b47be349ee4777f097f06ac0ca244`.
- Local full rehearsal: `94491a87b1e609b465c39b58207dd183c654cecd417f231bc0a40a4b77be2e18`.

Root's independent scorer tests exposed a repeated-loader bug in the new
read-only scoring tool: its namespace-package check rejected an already loaded
frozen `benchmarks` package. The agent corrected it with scoped namespace lookup
and pinned search locations; mixed/external namespaces and spoofed origins still
fail closed. Root inspected the fix and independently ran the combined tests
from the repository working directory (where a second namespace lineage can
appear): **193 passed**, including seven unchanged independent root controls.
Initial failing reports remain in the private stage. This diagnostic-only fix
does not alter benchmark source, requests, gold, adapter or production runtime.

- Scorer: `11a71cad7a425e457e54e87213398ad85796aa8b39d72c92c52bad3db79dee4b`.
- Scorer implementation tests: `a167793dd9d6fd37d2479b7a965535d99ae08870ea50f367e790521bcb637d07`.

**Execution history: isolated Linux gate passed; single paid run dispatched.**
Root reconciled source/helper identities, both complete worker/cleanup chains,
the replay and the exited gate container before creating fresh authority. No
local rehearsal fences or plan files were copied to Afrodite. The host-owned
supervisor is detached from the laptop/SSH connection; there is no automatic
campaign reroll or resume. Collection is not a semantic pass.

- Remote plan: `a1f1d4dbbfef5a06376f26a79f0237e77df2fddd86cb7d3c414678293ae06f42`.
- Remote gate: `4e91a28447445d8f3a9cdd166ddad151c4a8ca63b3d6798bd870bb7e5a895b7d`.
- Remote full rehearsal: `5179f9cac02490cb4f1d0c0bd27f0bd23c0d7ccf424eb499cea9f252e98e4ddd`.
- Remote parser replay: `6f620412c3201bcc4c6097132069d55c2a138e2bad2022ca3873f2f3ddc892f2`.
- Fresh live authority: `d7736b84871776fa029556d830d1b885ae54f66a9d98c1d3c56012b008d1798c`.
- Independent receipt auditor: `5f6abb48a13bee9bc57a9a810c3fd21c4c31948cb232628ecf04ed93966a7dfe`;
  110 tests passed, rerun by root, including complete offline receipt audits and
  tamper controls. The completed live receipt audit also passed, independently
  rerun by its author and root.

The earlier comparison's authority remains spent. This document does not create
execution authority or establish LME readiness. A promising development result
still needs untouched cases and end-to-end confirmation before adoption.

## Completed live result and adoption decision

**Completed without execution faults; do not adopt the ledger or clear LME.**
All 132 invocations completed, using exactly 238 completions / 238 HTTP attempts.
All finish reasons were `stop`: no transport retry, length finish, deadline
failure, accounting discrepancy or cleanup warning. Total usage was 358,404
prompt + 53,048 completion = 411,452 tokens. The independent receipt audit and
root replay agree on every frozen request, authority, intent, committed result
and usage total. Maximum supervised invocation duration was 12.31 seconds,
inside the 120-second deadline.

A separate read-only Afrodite process probe found all 132 owned worker process
groups absent, the entry absent and both host transport/supervisor PIDs absent.
Hermes1's existing container/image identity remained unchanged and running.
This is not a comprehensive production-health check. No production memory,
runtime deployment, service restart or full LME was part of this experiment.

Preregistered target-family results, counting every repetition:

| Measure | Baseline | Evidence ledger |
| --- | ---: | ---: |
| Correct target decisions / 80 | 54 | 19 |
| Unsupported targets accepted / 38 | 26 | 2 |
| Unsupported targets explicitly caught / 38 | 12 | 10 |
| Supported targets accepted / 42 | 42 | 9 |
| Malformed target decisions / 80 | 0 | 40 |
| Uncertain target decisions / 80 | 0 | 19 |
| Explicit false unsupported verdicts | 0 | 0 |
| Model calls | 66 | 172 |
| Total tokens | 129,281 | 282,171 |

The ledger blocks 33 of 42 supported targets: 25 malformed and eight uncertain.
Those are operational false blocks, despite zero explicit `unsupported` false
vetoes. Its lower false-accept count therefore does **not** establish a usable
improvement. It costs 2.61× the calls and 2.18× the tokens in this sample; token
ratio is not a dollar-cost calculation. Paired matches: 14 both correct, 40
baseline-only, five ledger-only, 21 neither. Results are correlated development
observations, not population accuracy or end-to-end benchmark scores.

The simple new controls were 24/24 for baseline and 6/24 for the ledger. On the
retained known defects, baseline missed all 20; ledger explicitly detected
three, falsely accepted two, and left nine uncertain and six malformed. Thus
baseline's better aggregate score does not make baseline semantically adequate.

### Failure diagnosis, without repairing or rescoring responses

Root traced the unchanged frozen parser on all 172 ledger responses. There were
68 malformed response scopes (not the same denominator as 40 malformed scored
target decisions):

- 59 first fail the exact root shape because the model adds
  `"type": "json_object"`. The rest of the JSON key inventories match, but
  downstream evidence/content checks are not thereby proven valid.
- Eight first fail quote uniqueness. The quotes actually occur in the permitted
  source, but appear two, three or eight times. Common entity/tool names in
  otherwise simple controls trigger this; these are not absent quotations.
- One first fails exact candidate coverage because a summary fragment loses
  one leading space during model-generated segmentation.

These are first rejection reasons, not an exhaustive counterfactual analysis
of what would pass after dropping fields or changing evidence rules. All calls
ended normally, so raising the token cap or timeout is not supported by this
run. The original strict parser, original gold and original scores remain
unchanged. No malformed response was repaired or partially accepted.

Source-grounded review confirms useful individual catches, but also important
semantic limits. Both target false accepts are repeated versions of the same
retained summary: a long whole-field claim cites genuine text while accepting
unsupported viewing history and causality. In another case an unsupported
actor-name attribution is marked supported, but an uncertain outcome label
makes the aggregate target uncertain; the two target false accepts do not
exhaust erroneous supported subclaims. Four of eight positive-target
uncertainties are caused solely by outcome labels such as `informational` or
`deferred`, despite supported body/entity claims. Exact quotations and complete
character coverage do not establish entailment, atomicity or complete evidence.

The independent reviewer exhaustively checked all ten explicit negative
matches, both false accepts, all nine positive matches and all 19 uncertain
targets; root checked the report against score totals and inspected the causal,
identity and outcome examples directly. Of baseline's 26 false accepts, only
five become correct ledger rejections; 19 become malformed/uncertain and two
remain false accepts. The five gains are four correlated exclusivity decisions
and one identity decision. Across structurally valid episode scopes, 34 of 41
outcome claims are uncertain. This is a contract/usability failure as well as a
semantic-judgment limitation, not proof that a more conservative score is better.

### Next offline design gate

Keep this prototype diagnostic-only. The next design should move mechanical
JSON scaffolding, candidate text coverage and evidence coordinates into code,
leaving the model compact judgments over stable claim/source identifiers. It
must distinguish categorical outcome judgments from source-verbatim assertions,
and preserve attribution, negation, causality and exclusivity in every compound
claim. It also needs a separate obligation to preserve salient source outcomes:
covering all candidate text cannot detect important evidence omitted from that
text. Quote/location validity must remain distinct from semantic support.

Use these failures as frozen regression cases, with positive acceptance as an
explicit gate alongside false-accept detection. Do not simply loosen the parser
or reward abstention. Offline controls cannot prove that a new prompt performs
better live; any later comparison needs fresh authority, preserved development
results and separately untouched cases before end-to-end LME confirmation.
This recommendation does not authorize another paid run or deployment.

### Receipts

Private downloaded results: `/private/tmp/hymem-ledger-results.VenqYr`.
The single paid authority is consumed and cannot be resumed or rerolled.

- Live summary: `02c696a8627c33aa5089d2762983c7c5e9ffbfce4590567e481c38c5ed87bc81`.
- Downloaded receipt archive: `d23e85fd9f640bc017c37a3433f3d6219e3d648b349776a2aeaf3dbbc6bc32c0`.
- Independent receipt audit: `4760596eca70afee3f47308bced7be20e1201f523621016ed0da496e199b5024`.
- Score metadata: `624e5a043148644807e07b6be108813fba079b50c1788ee0795df69b222e47e0`.
- Private scored outcomes: `f53286c0b1e4fcf958e42430d8fda4cf69720d351cb1d44898c3e33adc6336b1`.
- First-failure parser trace: `0c66d6867904a861a710e17717bbc39c5a19e1334444ce56da74dd7e1ce4a3f1`.
- Independent grounded review: `bddaf18d4ffc4aa7e334ff3f82131b30841d5f297fa0a0659d915ff829ba17cb`.

## Subsequent offline work

The user's subsequent “run it” was applied to the recommended offline redesign,
not a replay of this spent campaign. The additive
[compact assessment prototype](2026-09-16-digest-evidence-assessment.md) passed
2,359 selected regressions across 48 modules and an independent 33-case/86-request
projection audit. It moves text preservation and evidence addressing into code
and separates assertion, relation, categorical outcome and source-retention
checks. One impossible-output-budget preflight bug was fixed and regression
tested before the final gate. All 468 prior Python/SQL inputs and this campaign's
receipts are unchanged. No new paid calls, production changes or LME clearance.
