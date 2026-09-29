# Evidence-isolated verifier experiment

## Scope and status

The user's next "run it" follows the completed 64-invocation paired experiment.
It authorizes preparing the recommended evidence-isolation candidate. This step
implements and verifies an **offline, diagnostic-only prototype**, not a new
provider campaign, production deployment or full LME run. Both prior live
authorities are closed. Existing worktree changes and private receipts remain
untouched.

The preceding comparison found five grounding-contract violations among accepted
outputs in each arm. In particular, prior-derived names entered episodes whose
own citations did not establish those identities. A shared request exposed the
verifier to evidence it was told not to use. This establishes a violated evidence
boundary, not the provider's internal reasoning or proof that separation will
solve every false accept.

## Hypothesis and design

Prepare a separate semantic request for each episode and each raw procedure,
plus one rolling-summary request. Each item request contains only its unchanged
candidate and exact, ordered own-cited source records. Neither prior summaries,
other candidates nor uncited catalog records enter that request. The summary
request retains its broader authorized new evidence and prior continuity, but
has no generated episodes/procedures or rejected raw summary to borrow from.

Candidate fields are still claims, not evidence. Episode titles and bodies retain
separate verdicts; title exclusivity, modality, attribution, boundary-context
limitations and all existing evidence rules remain in force. Interpretation-only
boundary context is preserved exactly, not scrubbed or promoted to independent
evidence. A model can still make semantic errors even with a narrower request.

The prototype consumes the existing v9 fidelity payload after the maintained
canonical-span builder. It is not imported by production code and cannot produce
a publishable SessionDigest, advance a cursor or write a store. Primary
generation, summary repair, format screening, retry policy and durable identities
are not changed by this experiment.

## Execution constraints

- Preflight the entire plan before any call: strict input shapes, unique and
  resolved references, ordered item indices, finite request parameters, exact
  serialized input limits and explicit completion ceiling.
- Freeze serialized requests and bind them to a version and content hashes.
  Caller mutation must not change an already prepared plan.
- One completion per scope, sequentially; no hidden retry, repair or omitted
  scope. Record ordinary vetoes and malformed replies rather than counting them
  as support; halt on execution errors and propagate deadline/process interrupts.
- Preserve existing bounded verdict parsing behavior before applying each
  scope's exact response schema. Never synthesize a summary approval for an
  item-only request.
- All-scope support is a diagnostic result only, not factual proof or publication
  authority. Format and complete-pipeline behavior are outside this prototype.
- With E episodes and P raw procedures, semantic verification requires
  **E + P + 1** completions. The previous six/seven-call pipeline contracts and
  spending approvals cannot be reused. HTTP accounting, endpoint/model identity,
  process supervision and a shared deadline remain requirements for any future
  approved live harness, not capabilities inferred from this helper.

## Work and verification gates

1. A fresh implementation agent builds the isolated preparation/execution module
   and focused unit tests, without modifying runtime or existing live helpers.
2. Root independently tests source exclusion, byte preservation, boundary
   context, duplicate procedures, no-op summaries, malformed and adversarial
   inputs/replies, immutable plans, bounds, exceptions and shared deadlines.
3. Root mechanically projects all 15 retained accepted outputs from the prior
   approved review through the new builder. This is a no-provider evidence-scope
   audit, not rerunning or relabeling historical model verdicts.
4. Run selected existing verifier/planner/probe/deadline regressions with no
   external provider calls. Verify existing runtime source hashes did not change.
5. Record actual results and limitations. Do not call the semantic problem fixed
   or clear LME on the strength of scripted/offline tests.

## Results

The diagnostic-only implementation is in
`benchmarks/digest_evidence_isolation.py`. No production module imports it.
Preparation freezes one request per scope and validates the complete call
reservation before execution. Existing bounded closing-container tolerance is
preserved; the new scope schemas never invent verdicts for absent scopes.

Root's first 80 independent tests found **seven failures** (73 passed): a valid
null episode outcome was rejected, while an invalid outcome, reversed item
citations, incomplete/reordered summary evidence, future boundary context and a
text-mode request were not rejected. The implementation agent corrected these
before acceptance. The failed receipt remains preserved. Root then independently
reran the implementation agent's 70 tests plus those 80 controls: **150 passed**.
Four further independent checks cover mixed title/body verdicts, rejecting old
combined replies and bounding the entire output. The broader frozen-input gate
includes all 154 focused tests.

### Completed independent regression gate

**1,906 tests passed** across 44 selected modules, with zero failures, errors or
skips, in 567.392 seconds. This includes the 70 agent tests and 84 root tests,
plus existing digest, fidelity/repair/format, diagnostic planner, episode-probe,
publication, lossless coverage, deadline and semantic-generation regressions.
It is a selected-regression run, not the entire repository suite.

The root-owned runner removed credential environment variables, blocked socket
connections/name resolution/datagram sends, reconciled collection against JUnit,
and compared all 464 current Python/SQL inputs before and after execution. There
were **zero network attempts** and **zero changed input files**. Every initially
failing root check now has a passing result in this complete final run. Source
projection, syntax parsing and `git diff --check` also passed.

- Final input inventory:
  `7ea4fce6af6fa182080b6a99d14091d62329ca7d8ceada7bfa6a86a8cd3f7373`.
- Final JUnit:
  `fe25bbc59b1b77b36a3e47e4931d9ae3023f5b979f3c4b32f07a1b4768a4c08b`.
- Independent `root-offline-gate.json`:
  `b280af512375022d917abd69e8785ab441723913001f61eb0b5f64069d2c0047`.

### Retained-input projection audit

All **15** previously accepted pipeline outputs were mechanically checked, yielding
**34** prepared requests: 19 individual episode checks and 15 summary checks.
There are no procedures in these retained outputs; synthetic unit tests cover
procedure multiplicity and separate citations. The audit checked exact source
records, citation order, candidate fields, attribution/boundary metadata,
effective summaries, prior continuity and sampling parameters. Full requests
(system plus user) range from 4,224 to 6,741 characters, within the fixed cap.

Eight inputs already use v9. The other seven use baseline v8; a private offline
adapter changes **only their schema label** on a copy after verifying all other
fields are identical. This is an explicit diagnostic input adaptation, not an
upgrade to historical verdicts or receipts. The original review bundle is
unchanged. No model was invoked, and none of the old outcomes was relabeled as
an isolated-verifier pass.

All 461 pre-existing Python/SQL source/test files match the pre-work inventory.
Only the new experiment and its new tests have been added to executable sources.
No production memory, service, store, credential or paid-call authority changed.

Private audit root: `/private/tmp/hymem-evidence-isolation.lF3gD3`.

- Original reviewed input:
  `214085ec24876fb9afd34858b66eb1c5b9f6d9452bae14be6fba31759980838d`.
- Projection audit:
  `018b2ab27d88c5c899b80eb05f5d0067812d73f3c5a30d7669547dbb6953e62c`.
- Implementation:
  `2bcfa5aab243727880ebd2c9d7b9136a176c8660a475d9d2a045b654bd417f55`.
- Agent tests:
  `774309849096a76ebfaa842a7d7060cee7ab629a8a5b85fa6b0d6011b6513a8f`.
- Root tests:
  `41e14ce303a86cd50aad5006d34924b1ba22aa03b5885341101e3a342e9f38a2`.

### Remaining gates

The prototype has not been run against a live model and is not wired into the
digest pipeline. Offline isolation success cannot establish fewer semantic false
accepts/vetoes, better LME scores or end-to-end reliability. It cannot fix summary
length failures by itself. The next live comparison should use **identical fixed
candidates** in both arms, including faithful and deliberately defective controls,
so stochastic generation length does not confound this verifier comparison.
Keep uncertain labels separate from confirmed semantic defects and report
per-field verdicts, not merely whole-request acceptance. A future harness must
bind new per-scope budgets, replay, accounting, implementation identities and
shared deadlines before fresh spending authority can be exercised.

Production adoption and full LME remain uncleared.

## Fixed-candidate live comparison preparation

The subsequent "Run it" request prompted preparation of a fresh comparison, not
reuse of the previous paired campaign's spent authority. Both arms receive the
same fixed candidate and permitted evidence. The baseline is the current v9
combined semantic verifier; the candidate is the isolated-scope prototype.

The frozen set contains the 15 retained accepted outputs plus six existing
faithful/defective F1–F3 controls. The two F4 grammar-only controls are excluded
because formatting is outside this semantic experiment. Two repetitions reverse
the adjacent arm order for each case: **84 invocations**, reserving **162
completions / 486 HTTP attempts** (42 baseline completions, 120 candidate
completions). Sampling stays T0, 3,072 tokens, JSON, `deepseek-v4-flash` at
`https://api.deepseek.com`. Each invocation has one absolute 120-second deadline
and two-second cleanup allowance. No generation, repair, automatic reroll,
campaign resume, production change, restart or full benchmark is included.

An independent source-grounded review preregistered 28 field/index targets:
13 unsupported and 15 supported. Retained task 35 is explicitly unscored because
its conditional-advice compression remains uncertain. Untargeted fields are not
assumed supported, and an unrelated veto does not count as detecting the
intended defect. Repeated or related candidates are not independent population
samples. These fixtures contain no procedures, so this experiment cannot measure
procedure-verification quality.

Fresh private helpers collect exact requests/responses in owned subprocesses.
Model vetoes and malformed replies remain observations; transport, accounting,
identity or cleanup failures halt. Interrupted-task usage is explicitly marked
unknown rather than reported as zero. All five helper files, frozen sources,
fixtures and preregistered labels are hash-bound. Root reran all focused helper
and independent adversarial tests: **213 passed**.

The complete dry run finished all 84 invocations / 162 synthetic completions,
with zero HTTP attempts. The first localhost SDK rehearsal could not start its
server because the sandbox denied the socket; it made no model calls and did not
consume the loopback or live execution fence. Its failure log is preserved.

After permission for the local socket, the full SDK loopback rehearsal also
passed: **84 invocations / 162 dummy HTTP attempts**, with exact request-body
matches, all owned workers and the local server reaped, unchanged bound inputs,
and zero external provider attempts or credential reads. Root separately replayed
both rehearsals through the frozen actual parsers: 168 invocation records / 324
requests matched; all 84 baseline replies and all 240 isolated-scope replies were
correctly rejected as malformed synthetic replies. None became a semantic
approval. This verifies collection/replay and cleanup, not model accuracy.

- Root helper-suite JUnit: `001dc428b3509976d9364213b1375a488a1e957861c3d3f5aec400e8d7db2003`.
- Five-helper manifest: `851529915dffa8c61fd0b1f94b04c177bfd9e2de8700963ffc47acfa446a0c6f`.
- Dry summary: `5cecbbba341b84203705fc590df88999e3d5553517b94fa46fe5709078c0dc48`.
- SDK loopback summary: `f93037374a1aba915760e3d6c6744abdac3f06b1f786e700dc038782f75bdb91`.

The local plan binds its absolute stage path and cannot be reused as remote live
authority. Remote preparation/rehearsal, a fresh root review and explicit consent
must precede any live dispatch. No live execution fence or authority exists.

### Initial transfer block (subsequently resolved)

Security review denied copying the frozen repository/benchmark archive to
Afrodite because that transfer lacked explicit authorization. The denied upload
did not run. At that point, only an export of the already-approved old control fixtures existed
in the new remote private directory. The user was asked specifically to
approve transfer of diagnostic code and benchmark-only text to Afrodite and the
162/486 capped comparison. No workaround transfer, credential-file transfer,
fresh live authority or paid invocation had occurred. Local preparation does not
clear production adoption or LME readiness.

Private preparation: `/private/tmp/hymem-isolation-live.A0CLcg`.

- Frozen fixtures: `05c42ed2f4eb53371206991a224652215a020a6e9fd0868736afdbbad0d639f0`.
- Frozen source manifest: `a018ce87cbbf944fc9f8d572b212a4346cf5fd9e017a2ca469b11d8b6f49a07c`.
- Preregistered gold review: `59908eef1fe1fef52f97537c85b96ff86bcd6bed569b381146f7b4d8946340cf`.
- Denied-transfer archive: `e4c9bd85e2c9285488ba31ca8ca272f5a7aa59c08165b503de75183a9993dcd9`.

Raw benchmark text remains in private artifacts, not this report.

## Approved transfer, remote verification and paid-launch block

The user then answered **"yes"** to the transfer-and-162/486-comparison question.
The source archive and explicit diagnostic helper files were transferred to the
verified private Afrodite directory. All archive, fixture, gold, source and
runtime-helper hashes matched. No credential file or production memory was
transferred. The running Hermes1 container ID, image and home mount matched the
prepared launch contract; no deployment or restart occurred.

Root launched a separate network-disabled, read-only-root Linux container with
only the diagnostic directory, read-only frozen inputs/runtime and scratch space
mounted. No production home, memory database or credential file was mounted.
It exited **0** after **213 tests**, both complete 84-invocation rehearsals and
the independent 168-invocation / 324-request parser replay. Every owned worker
and the dummy server was reaped; external provider attempts and credential reads
remained **zero**. Root then independently reconciled the receipts, commits,
input hashes, task accounting and container exit before recording fresh consent.

- Remote plan: `1762aa5ab7b11228595eff9af1ad76ec20162fd56c16ac2ef9edefe0c29825aa`.
- Remote helper JUnit: `0a12f68cb67adc7e2517d6d7574f8625e948c56737f64e2d1e3452a27b8f1011`.
- Remote full rehearsal: `0f0047e2ae88d89bbced8ed07f9852bdff8bdf045f719d731dc3da9ff8404ad3`.
- Remote parser replay: `403bce8e02b7c2f2858bcec5242a44df20a3fce5fbf1bdb9fa33de9e31eac254`.
- Recorded, **unused** one-shot authority: `47143606e5d2568c23781b1242351244f8266a0bf32ebd344a1b545c138bd7c5`.

The subsequent `host_dispatch.py start` tool action was **denied before
execution**. Security review required clearer explicit authorization for sending
this exact 21-case / two-arm / two-repetition payload to DeepSeek under the
162-completion / 486-HTTP-attempt limit. Root did not retry or use an indirect
launch. A read-only check confirmed `host-launch`, `entry-dispatch.json`,
`live-fence.json` and `live-results` are all absent. Therefore **no new paid calls
have run and no live result exists**. A new explicit question names the provider,
model, payload, cap and exclusions. Do not treat the unused authority file as
permission to bypass that block.

A separate agent prepared a read-only offline scorer. Root inspected it and
reran **89 synthetic tests, all passing**. It verifies all frozen requests and
complete receipts, then uses the actual frozen parsers to score 112 target
observations. Missed defects, explicit false vetoes, uncertainty, malformed
responses and other-field/other-scope vetoes remain separate. These scorer tests
are not live semantic evidence. Its SHA is
`4d57760cead158483a8aeb0d44822d62a34e8b125950b51b59e80331e354470d`.

Status at that turn: remote preparation complete; **paid launch blocked on explicit
provider/payload authorization**. Production adoption and full LME remain
uncleared.

## Explicit provider approval and live execution

The user answered the exact provider/payload/cap question with **"I explicitly
approve this DeepSeek data transfer and 162/486 paid diagnostic run"**. That reply
is preserved separately as `explicit-provider-consent.json`, linked to the
question-item identifier and the unchanged remote plan/unused authority hashes.
Read-only verification confirmed no dispatch, entry, live fence or results had
been created by the earlier denied tool action.

The approved launch then succeeded: detached Afrodite supervisor PID **1774756**.
It uses the already-verified frozen helpers and the single previously unused
authority; this is the first execution of this comparison, not a campaign resume
or reroll. The runtime key stays local to the approved process; no credential
file was transferred. The first 24 invocations reconciled 40 completions and
40 HTTP attempts without an infrastructure/accounting failure. Final semantic
results and full receipt audit are still pending at this progress checkpoint.
No production code, memory, service or full LME benchmark was changed.

## Completed live comparison — do not adopt this candidate

The first and only authorized execution completed all **84 invocations** in
419.915 seconds. It used **162 completions / 162 HTTP attempts**, below the
486-attempt ceiling: 240,178 prompt tokens and 6,209 completion tokens, 246,387
total. All responses finished with `stop`; there were no retries, truncations,
malformed semantic responses, unknown-usage events or infrastructure failures.
The detached dispatcher and entry exited successfully. Independent root audit
verified every exact request, worker commit, source binding, usage total and
owned-worker cleanup receipt. The Hermes1 container/image/home binding still
matched after execution. This is not a new production health audit.

Root separately replayed all 162 actual responses through the frozen semantic
parsers. Each arm has **56 pre-registered target decisions**: 28 field/index
labels repeated twice, including 30 supported and 26 unsupported observations.
These are correlated targeted cases, not independent samples or LME scores.

| Target-level result | Combined v9 baseline | Isolated candidate |
| --- | ---: | ---: |
| Correct decisions | 31 / 56 | 30 / 56 |
| Unsupported claims wrongly accepted | 25 / 26 | 22 / 26 |
| Supported claims wrongly rejected | 0 / 30 | 4 / 30 |
| Uncertain or malformed target verdicts | 0 | 0 |
| Completions | 42 | 120 |
| Total tokens | 96,061 | 150,326 |

Isolation caught three additional defective-control observations but introduced
four false rejections. The paired target comparison is 27 both correct, three
candidate-only correct, four baseline-only correct and 22 neither correct. It
used **2.86× the completions and 1.56× the tokens**; these ratios are not a dollar
cost estimate. This experiment does not establish a net semantic improvement.

### Failure localization

- Both arms missed **all 20 retained defect observations** (10 known defective
  retained outputs repeated twice). Isolation did not fix unsupported actor
  attribution, over-broad exclusivity titles or invented viewing status/causality.
- On F1, isolation rejected the deliberately exclusive title twice and accepted
  the faithful counterpart twice; baseline accepted both defective titles.
- On F2, isolation made the wrong decision in both directions, twice: it rejected
  the episode with the necessary additional citation, but accepted the same
  candidate after that citation was removed. An unrelated summary veto cannot
  be counted as detection of the episode's missing support.
- On F3, isolation rejected both the faithful and defective summary twice. The
  baseline accepted the faithful summary twice and rejected the defective one
  on only its second draw. Rejecting both variants is not successful discrimination.
- Four candidate target false accepts were accompanied by another scope's veto;
  those remain false accepts. Untargeted fields and retained-35 remain unscored.

Root checked the representative retained-05, retained-07 and retained-13 inputs
against their exact returned verdicts and the F2/F3 controls. The observed
failure is **semantic judgment despite well-formed output**, not a new transport
or JSON-parser failure. For example, the isolated episode request still approves
an actor name that its own visible source and attribution metadata do not supply.
Removing unrelated context therefore is not sufficient to enforce evidence
authority. The decision-only replies do not reveal the model's internal reason;
do not claim a proved serving-side or tokenization root cause from this run.

A second agent independently recounted all target decisions directly from raw
JSON replies without importing the scorer, obtaining the same totals. It also
checked 84 replies across 36 selected invocations (all six controls plus
retained-05/-07/-13, both arms and repetitions) against their exact requests and
permitted evidence. This confirmed the F2 reversed distinction and the continued
approval of unsupported attribution despite mechanically correct isolation.
The pre-registered labels were not changed after observing the model results.

### Decision and next gate

Keep this prototype diagnostic-only. **Do not deploy it or claim LME readiness.**
No production code or configuration was changed, and no additional paid run,
reroll or full benchmark was started. The one-shot authority is consumed.

The next design should make claim-to-source evidence auditable rather than
relying on a bare supported/unsupported decision: explicit claim decomposition,
source-linked support and deterministic checks of citation/attribution boundaries.
Exact source-span matching alone will not prove semantic entailment, so retain
these faithful/defective paired controls and add held-out cases before assessing
any replacement. This is a design recommendation, not proof that another schema
or model will fix the issue, and not authorization for another paid campaign.

Private completed receipts: `/private/tmp/hymem-isolation-results.eSIrHT`.

- Explicit provider consent: `09f3cee4fe67c16485c136da70648dd3c8eaacb846d1adc1fd0eafaaa44f91ce`.
- Live summary: `82460e4f628ae9d6d5ce8baafffe9f6ce183c4f00a572793471dec4100d3730a`.
- Independent root receipt audit: `1cdc9f6d0b962b2893554d45148c6ac8cfe799bd33ea3905469264b60797b5fd`.
- Independent frozen-parser target scoring: `d3be45c55e60a3c9b53ac78c9bd10e3af34bb2c4a70914425341cf44e05ac34b`.
- Downloaded receipt archive: `e4fe865e339f7321813f5304033ad264f9de4f53f8cb8182fbe47a2d0755de50`.

Raw benchmark text and provider replies remain in private receipts, not this
report. The result does not test generation, repair, format enforcement,
procedure quality or end-to-end LME convergence.
