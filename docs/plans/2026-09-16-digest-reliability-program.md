# Digest reliability: diagnose the system, then compare a bounded redesign

## Status and scope

The user requested the improved, whole-pipeline approach after the v16 live
diagnostic. Work starts with failure mapping and reusable diagnostic machinery,
not another production prompt patch. Existing dirty worktree changes are
preserved. No new provider calls, deployment, restart or production-store access
are authorized by this document alone. The user separately approved the initial
live mapping campaign on September 16: at most 112 completions / 336 HTTP
attempts, four retained cases, four synthetic stress cases and eight controls,
twice each; no production memory, deployment, restarts or full LME run.
The approved initial live mapping completed all **32 tasks**, with **66
completions / 66 HTTP attempts**. Root independently reconciled its receipts,
usage, worker cleanup and unchanged inputs. Pipeline contract acceptance was
**7/16**; verifier controls matched **8/16** preregistered expectations. This is
not LME clearance. The subsequently approved source-grounded adjudication is
complete: the retained control labels are justified, and one of the three
contract-accepted retained pipeline outputs has a confirmed unsupported title.
The next candidate must be judged on semantic errors as well as contract pass
rate. Mapping and adjudication made no runtime change. After the user's
subsequent "Run it", the separated-decision candidate was implemented locally
and passed the independently reconciled offline regression gate (status below).
This establishes tested control flow, not live model quality. No deployment,
restart or production-store access has occurred. The separately approved fresh
paired comparison has now completed **64/64 invocations**, using **139
completions / 139 HTTP attempts** within its 240/720 ceiling. The candidate
eliminated invalid semantic-verifier envelopes in this sample, but still accepted
claims unsupported by their permitted evidence. **Adoption and LME remain
uncleared.** The complete paired result and source-grounded review are below;
neither spent campaign authority permits another run.

This program has distinct gates. Completing the planner or its synthetic tests
does not complete the program, establish model quality, or clear LME.

## What the current evidence actually establishes

| Observation | Established finding | Not established |
| --- | --- | --- |
| v14 source review | Both rejected summaries lost a source-supported temporal relation; a repaired summary also received a false format rejection under the written contract. | The verifier's precise rationale or a population-wide error rate. |
| v15 live run | A verifier reply with completed items but missing outer JSON closure blocked the run. | Whether later targeted repair would have succeeded. |
| v16 live run | Primary summary 507 characters; one compaction 523 against cap 500, valid JSON and normal `stop` responses. | A regression caused by the verifier-envelope fix, which was never reached. |
| All three campaigns | Each stopped on its first pipeline case. | Results for the unattempted cases and verifier controls. |
| Offline regression gates | Bounds, validators, publication protection and specified control flow work for tested scripted inputs. | Reliability or factual accuracy of live generated outputs. |

The frozen baseline has a three-call normal path and six-call maximum: generation,
optional length compaction, semantic screening, optional summary-only repair and
re-screening, then mandatory candidate-only format screening. Any failure holds
the entire slice: no summary, episodes, procedures or cursor authority. A later
dream retry regenerates the primary candidate; failed valid components are not
durably reused. The production verifier also reports its first failed family,
so the old campaign was censored both within a response and across cases.

This coupling is safe against partial publication but operationally brittle.
Even an illustrative independent 99% success rate per slice gives only about
36.6% probability of an entirely successful 100-slice pass. These are not
measured HyMem probabilities; correlated failures can make retries still less
useful. A single successful canary is not a workload-reliability measurement.

## Gates and ownership

1. **Freeze the baseline and map known failure paths.** Use the actual frozen
   v16 candidate, not Git HEAD (the checkout has substantial uncommitted work).
   Baseline manifest: `dc52f75d5036486ff3ca0d58773640781edcb61f10415ea63451f0afe7702225`.
   Preserve the original failed receipts; never relabel them as repaired passes.
2. **Build and independently verify a reusable campaign planner/executor.** A
   separate agent implements the diagnostic-only module. Root reviews it and
   adds independent adversarial checks. It must distinguish model rejection
   from unsafe infrastructure failure and must not add a hidden retry loop.
3. **Map baseline behavior before choosing a candidate.** Run the predeclared
   matrix only with fresh explicit live authority. Keep every outcome. Review
   complete private responses against their exact approved sources, not a
   truncated excerpt or the verifier's verdict alone.
4. **Choose one bounded architectural hypothesis.** Define the predicted
   failure-rate, semantic-quality and cost effects before editing runtime.
   Use a separate implementation agent, then root verification. Do not deploy.
5. **Run an interleaved frozen-baseline/frozen-candidate comparison.** Both arms
   see identical cases, prior/context, extraction configuration, model and
   sampling settings. Alternate AB/BA within matched case/repetition blocks.
   Prompt/code differences intentionally under test belong to the arm identity,
   not the immutable evidence identity. A baseline map from an earlier time is
   not a substitute for the interleaved baseline arm.
6. **Confirm on untouched cases and an end-to-end smoke.** Separate diagnostic
   development cases from holdout workload cases. Repeated component success
   does not cover chunk extraction, dream convergence, persistence, retrieval
   or the answer/judge path. No canonical baseline clearance before those gates.

## Initial mapping matrix proposed for approval

One baseline arm, two predeclared repetitions, fixed order before any outcomes:

- Four already-approved retained pipeline cases: two incident source windows
  in blob and granular modes. These are four configurations, **not four
  independent real sessions**.
- Four new synthetic pipeline stress cases: near-full prior with tiny correction;
  dense structured source; split-boundary context; thin/no-new-information input.
  They require deterministic fixture construction and reviewed expected facts.
- Eight existing faithful/one-defect verifier controls, exercised directly even
  if generation fails. Do not pool their success with pipeline success.

This is **32 invocations**, at most **112 completions / 336 HTTP attempts**:
16 pipeline invocations × 6 plus 16 single-call controls. Each invocation has
one absolute 120-second deadline plus two seconds cleanup. This is a deliberately
small failure survey, not statistical proof of rare-failure reliability or a
representative LME population sample. Model is `deepseek-v4-flash` at
`https://api.deepseek.com`; temperature remains 0.0 and per-call token limit
remains 3,072. No provider/model/temperature workaround is silently introduced.

The four synthetic fixture bytes, all case/config/input hashes, exact schedule,
current source identity and helper identity must be frozen and reviewed before
dispatch. Pending fixtures or placeholder hashes cannot authorize execution.
The completed v16 authority is closed and cannot be reused. A paired comparison
or confirmation campaign requires its own explicit spending limit.

## Continuation and stopping rules

An isolated, fully accounted model/contract rejection is a diagnostic result.
Record it and move to the next independent scheduled task, without publishing
anything or retrying it to obtain a pass. Predetermined repetitions all count;
none are selected as the best response. Continue only after verified worker
cleanup and unchanged input identity.

Halt the campaign on unsafe/unknown cleanup, source/config/plan drift, missing or
untrusted receipts, unknown usage, budget overrun, infrastructure failure or
unexpected execution exceptions. Reserve worst-case per-task capacity before
dispatch. No dynamic task creation, automatic resume, credential output,
production mutation or full benchmark. Distinguish unused reserved capacity
from authorization for another campaign.

The new reusable executor accepts a trusted worker adapter. It is not a
sandbox and cannot physically enforce deadlines or network restrictions on an
arbitrary callback. A live adapter must separately use the existing process
supervisor, provider-attempt accounting, private capture, immutable source
checks and audited terminal receipts. Synthetic executor tests do not establish
that live adapter integration has been completed.

## Measurements and acceptance

- Always show planned, attempted, completed, rejected, infrastructure-error and
  unattempted task counts. An incomplete matrix cannot pass.
- Separate retained, synthetic and control populations and each input stratum.
  Report per-invocation cost and latency, not just per-call success.
- Record stages reached; mark blocked downstream stages unobserved, and optional
  unnecessary stages skipped. Never report an unrun stage as a success/failure.
- Enumerate all structurally valid verdict groups, not just the runtime's first
  returned rejection. Keep malformed responses separate from semantic vetoes.
- For comparison, retain matched-pair wins/losses/ties/missing, full failure
  distributions, cost and latency. Two repetitions can reveal gross failures;
  they cannot establish statistical superiority or independence of observations.
- Independent source-grounded review of accepted output and known positive/negative
  controls must guard against apparent reliability gains from false acceptance.
  Label reviewer provenance accurately: AI review is not a human audit.
  Model approval is not ground truth. Preserve chronology, negation, corrections,
  category distinctions, intent versus completed action and exact citation scope.
- Require zero known safety violations, no lost/duplicated coverage or partial
  publication, unchanged source inputs, bounded cost and deadlines. Do not relax
  a guard, silently truncate, discard hard cases, or rerun until passing.

## Redesign hypotheses to evaluate, not yet implemented

The source audit will rank these against the full mapped failure distribution:

- A single explicit bounded repair state machine with machine-measured defects
  (including character count) and source-linked semantic findings; revalidate
  every changed candidate, with no unbounded loop or higher hidden call budget.
- Reduce dependence on cosmetic model judgments when an equally strict,
  demonstrably correct deterministic representation exists. Do not substitute
  punctuation splitting for natural-language sentence judgment.
- Reuse already validated components across repair/retry boundaries using exact
  source/producer bindings, without granting failed candidates publication or
  cursor authority. Durable reuse would require storage/identity/migration tests.
- If the summary's strict representational contract is itself unsuitable, treat
  changing it as an explicit versioned architecture experiment with retrieval
  and benchmark comparability review, not an incidental bug fix.

The first live survey should determine which hypothesis merits implementation.
Making another production change before that evidence would repeat the reactive
process this program is intended to replace.

## Baseline and architecture audit receipts

Root rehashed all 458 frozen v16 inputs against both the current checkout and
the original immutable candidate; no mismatches. Existing source gates apply to
that exact unchanged source. New diagnostic code will have separate verification
and is not covered merely by inheriting the application's 2,473-test total.

The separate read-only architecture audit confirms that defaults allow six
consecutive digest attempts. At the six-completion per-attempt maximum, a stuck
slice can consume up to 36 logical calls before quarantine, excluding HTTP
retries. The live mapping runs one bounded attempt per scheduled task and does
not silently spend those durable dream-retry allowances.

The initial cap failure is already explicitly covered as an expected safe
rejection in `tests/test_digest_length_recovery.py`; its passing test never
established that a live compactor would obey the length limit. The new survey
must measure the latter separately.

Private preparation roots:

- Local: `/private/tmp/hymem-reliability-map.hQ8e5w`.
- Afrodite: `/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-reliability-map-20260916-kxiBAL`.

The old v16 stage, failures and spent authority remain untouched.

## Diagnostic planner accepted; live adapter under development

A separate agent implemented the reusable, offline-only planner/executor in
`benchmarks/digest_reliability.py`. Root reviewed it, requested corrections to
format-control routing and unsafe/forged-result handling, and independently
accepted a frozen **110-test** gate (90 implementation tests plus 20 root controls).
Zero failures/errors/skips, unchanged inputs and zero network attempts.

- Planner: `a70e2e29ecc9bc59b8280a2d716319b9acb475a9b4427d13b9f3d59e4def98b5`.
- Planner tests: `f062dda575fd401fa6bdb0fb8a43e8322fe8f9bed79d79f490f758ca021caa14`.
- Root JUnit: `f2c271a7a8736af6f2e0ac0e6b5e51e5815a159d67a7607230ffa0e9c89b594a`.
- Root input manifest: `01bac6e44f82f19983e7bfbd6a42fbf396335c89e5b843b51ec9a6189e48001b`.

The planner supports baseline mapping and balanced paired comparisons, preserves
all repetitions, separates controls/pipeline results and reports censored stages.
Its callback API alone cannot enforce process/network safety. A different agent
is now adapting the previously verified owned-worker implementation to that API.
The adapter must additionally require complete returned responses and token/HTTP
accounting; the old worker's broad success flag is insufficient for continuation.

The exact unchanged baseline and authenticated source-test evidence were copied
to the new private stage, without any old authority, consent or campaign fence.
Staging receipt: `331eb07ff8fe441b0c21c9a6db51f7ab69217f20d86bf02f3e623c797ba7dcdc`.
The four synthetic stress definitions are frozen separately with prior-summary
lengths 468, 116, 104 and 112 characters. Specification hash:
`66f431675a721419ba7777f3c5405e3d6efd2beb47c27d96d75fd55acb545cd9`.
Their reviewed fact checklists must never be injected as model instructions.

At the planner gate, no new live authority or provider call had occurred.
Baseline behavior has not been changed. Runtime redesign and candidate
comparison remain later gates.

## Live adapter frozen; independent local gate accepted

A different agent adapted the owned-worker implementation, leaving the accepted
planner, watchdog and application baseline unchanged. Root reviewed the changes,
requested stage-attribution and complete-verdict-family corrections, and added
16 independent adversarial controls. The frozen combined gate passed **440 tests**
with zero failures/errors/skips, exact collected inventory, unchanged inputs and
zero non-loopback network attempts/provider calls. This includes actual dummy-SDK
worker continuation after ordinary rejection and safe termination on timeouts or
late receipt errors, not only mocked callbacks.

- Seven-helper manifest: `4c8c765eb17e9c8cd1c273dcdc20f052b200c5ec435fea7edfbbf8e2e4ea0a5d`.
- Helper-test manifest: `f8602d2e53e4998ea34414d066138bd6458e7b2b9fd2e12a809ed28e70ba10c2`.
- Local helper JUnit: `eafeeb67e03dc50ae1866bdbb69ed2529aed574218e19b287f19c32e94edfd92`.
- Supervisor: `c79ed01d28ecd7d1281f420ff32a04ebc52e77c24a1c87b282336661744860d2`.
- Core diagnostic: `02fb3e377bc6c3989678e4fa8a8c0e518b1c71e888b173d27a4191dc24e7a945`.

Mapping completion is explicitly distinct from all outputs passing. Every
verifier call records validated counts for every verdict family. A pre-dispatch
limit rejection is not blamed on the preceding model call. Unknown earlier HTTP
attempts cannot be hidden by a later successful retry. Unexpected zero-call
outcomes halt: this adapter's frozen, nonempty prepared inputs require a primary
call before ordinary model-output rejection; it does not fabricate usage to
fit the planner.

The exact package has been staged on Afrodite without authority. Network-disabled
fixture construction and the independent Linux helper gate precede the actual
32-task dry rehearsal and any paid dispatch. The unchanged application source
continues to use its exact 2,473-test evidence; this is not a new full-suite run.

## Target preflight accepted; approved initial map dispatched

Linux independently passed the same **440-test** inventory: zero failures,
errors, skips or non-loopback network attempts, with unchanged inputs. Synthetic
fixtures passed immutable database integrity and foreign-key checks. Root then
reconciled the full **32-task / 74-synthetic-reply** rehearsal, including each
reservation, response, worker completion, deadline and cleanup. Zero provider
calls in all of these preflight stages.

- Linux helper verdict: `45ad7f40fa7c599bfb3e9c701b6f3137d441e965f67f0db644f839bf3c707296`.
- Root synthetic-fixture audit: `9b9bad68175842928a822045eaca42025434432db57ab269e6af20e749049dc1`.
- Root rehearsal audit: `576ba0e49a412ff9430574de7898b5675d746891b6090ee4420515766a507b8c`.
- Fresh single-use authority: `849ef4d51ab063425136d218849a5669501a520d0ab0cae4d529be94396aa132`.

This authority binds the user's explicit async approval of the initial mapping
campaign, not any prior "Run it" consent. The host dispatcher started it once;
no automatic rerolls or resume. The result can be a completed map containing
rejections, or a safety/accounting halt. Neither status alone clears LME.

## Completed baseline map and independent audit

All 32 predeclared invocations completed. The one-shot authority is now spent;
unused capacity does not authorize a rerun or a candidate campaign. No HTTP
retries, timeouts, unknown usage or cleanup failures occurred. All 66 provider
responses reported normal `stop`, not token-ceiling termination. Recorded usage
was 140,754 prompt tokens and 12,281 completion tokens (153,035 total); this is
not a dollar billing receipt.

| Population | Passed / attempted | Meaning |
| --- | --- | --- |
| Retained pipeline configurations | 3 / 8 | Two real windows, two extraction modes, two repetitions; not eight independent sessions. |
| Synthetic near-full prior | 0 / 2 | Compacted summaries remained 548 and 550 characters against cap 500. |
| Synthetic table | 2 / 2 | Configuration versus deployment and pending versus applied distinctions preserved on source review. |
| Synthetic boundary context | 2 / 2 | Visit chronology, negation and planned return preserved on source review. |
| Synthetic no-new-information slice | 0 / 2 | Summary-content veto persisted after repair. |
| Verifier controls | 8 / 16 | Two apparent false accepts, two apparent false rejects, four invalid verdict contracts, relative to frozen expectations. |

The nine pipeline failures were four length failures (three compaction, one
summary repair), two verifier-contract failures (one parse, one diagnostic
validation), and three summary-content rejections after re-verification. Seven
primary summaries exceeded 500 characters. Four of seven compactions became
length-compliant, but all seven corresponding pipelines eventually failed;
this **does not prove compaction caused their failures**, because compaction is
selected for difficult/overlong candidates. No summary repair yielded a complete
accepted pipeline in this small sample (one length failure, three later vetoes).

The seven accepted pipeline outputs all reached and passed final format
screening; the four direct format-control invocations also matched expectations.
That does not erase the earlier documented false format rejection or establish
population reliability. Accepted pipeline outputs are not automatically proven
faithful merely because the same model approved them.

One unchanged retained configuration passed repetition one and failed repetition
two: its summaries were respectively 478 characters, and 558 followed by a
508-character compaction. The application source did not change between them.
This directly establishes model-output variability as one contributor to the
apparent sequence of new failures; it does not rule out implementation defects.

### Structural diagnosis without exporting retained raw text

Root inspected captured replies in place, without additional model calls:

- Three unrecovered parse failures (one pipeline, two controls) encountered a
  JSON delimiter error at the final character. All ended normally at the provider;
  these are not demonstrated output-token truncations.
- Three diagnostic-validation failures (one pipeline, two controls) were valid
  JSON and hit the same exact guard: a `prior_derived_summary` reference was used
  with an issue code other than `prior_continuity` (`digest.py:1451`).
- Existing verdict-envelope recovery preserved semantic vetoes for three other
  replies that strict JSON parsing alone rejected at end-of-input. Their rejection
  remained semantic; recovery was not silently counted as acceptance.
- The frozen adapter's broad control `malformed_json` label includes schema/
  diagnostic failures. The four invalid controls were actually **two parse
  failures plus two diagnostic-validation failures**. Preserve the original
  report; use this exact failure breakdown, not a claim of four JSON syntax errors.

### Synthetic semantic review (root AI review, not human adjudication)

All four accepted table/boundary outputs were checked against the full invented
source and prior. The reviewed distinctions above were preserved. This is a
small, deliberately selected synthetic set, not representative LME evidence.

The no-new-information case exposed a concrete source-priority mistake. All four
initial/re-verification requests contained the exact 112-character prior, and
each candidate retained that prior verbatim as its prefix. The new source was
only an acknowledgment. The verifier nevertheless marked the retained prior
claims unsupported by citing the new acknowledgment. The written verifier
contract permits still-relevant prior continuity; it does not require those
unchanged prior claims to recur in the latest message.

In repetition one, the repaired candidate merely retained the prior and reported
the acknowledgment, yet re-verification rejected its prior-derived portion.
That is a source-grounded false veto in this synthetic case. The initial/second
repetition closing-session wording warrants separate scrutiny; do not infer
that every other clause or every whole-candidate rejection was necessarily wrong.
The actual request-byte checks rule out missing prior-context wiring in these
four calls. They do not establish how often the model makes this mistake.

### Audit receipts

At this stage, all retained raw requests/responses and source databases remained
on Afrodite. Only invented synthetic text had been inspected directly; exported
retained-campaign reports contained counts, enums and hashes. The later,
separately approved retained-evidence review is recorded below.

- Root live audit: `89fbd3b918e940bd1f732fd7501ea749e00b34833a76bc06a3d3aa10b5fe734d`.
- Machine-length audit: `e4fbcbfc4a1c5f1a9203c38a36e930114c51c777f09d8dec41c846fb43f4fba8`.
- Structural verifier audit: `8c6dc460a05abef110eb802e65bffafa7d9a2f62fb59fe54fc27c5fbdef5a7ef`.
- Exact synthetic-prior request checks: `9b40de1e3f87cebe3616d5725b92b3a5465b69399160c3c02944ee9975951194`.

## Adjudication completed: retained evidence and accepted outputs

The user approved retrieval and inspection of the retained benchmark sources,
controls and captured outputs, excluding new paid calls and production access.
A narrow exporter verified original plan/capture bindings before exporting four
case configurations, eight controls and 42 calls from 24 retained-case tasks.
Raw text is held only in the private review bundle, not this report or repository
fixtures. No source database, credential, unrelated record or production memory
was retrieved. No provider call was made during adjudication.

Root reviewed the complete retained sources, prior summaries, boundary context,
all eight pipeline trajectories and all controls. A separate read-only agent
independently adjudicated F1/F2 and the corresponding pipeline failures. This is
source-grounded AI review, not human adjudication or a population error estimate.

The root mechanical gate verified all eight primary requests and all sixteen
control requests against their frozen definitions, all twenty semantic request
catalogs and prior bindings, and twenty-one pipeline item-to-request bindings.
All four control pairs differ only in their intended field(s). Sampling and token
parameters are unchanged, and semantic/format system prompts match the frozen
runtime. Local network access was prohibited during these checks. Thus the
observations below are not explained by a missing prior or mismatched control
request in these captures.

| Control | Source-grounded label check | Captured result |
| --- | --- | --- |
| F1 title scope | Message 323 establishes availability on one platform and absence from another, not universal exclusivity; only the title changes. | Both faithful controls pass; both defective titles are falsely accepted in valid envelopes. |
| F2 own citations | Message 214's visible tail and boundary context authorize a continuing wish, not an independent trip; message 216 supplies the trip only when explicitly cited. The defect removes that citation alone. | All four response envelopes are invalid: two delimiter errors, two diagnostic-reference violations. Their textual approval of the defective item is secondary evidence, not an accepted valid verdict. |
| F3 supplied outcome | The faithful summary preserves the supplied recommendations/routes and legitimate compressed prior topics; the defect reduces supplied answers to requests/discussion. Only summary text changes. | Both faithful summaries receive false continuity vetoes. Both defective summaries are correctly rejected for omitted outcomes, alongside the same unjustified continuity complaint. |
| F4 sentence format | The faithful summary is one multiclause sentence; changing one semicolon boundary creates two complete sentences. Only summary text changes. | Both faithful and both defective controls match their labels. |

F3's faithful false veto is not inferred from a generic pass/fail label. The
captured issue specifically points to compressed older topics and quotes the
prior summary that supports them. The written contract permits that continuity;
the summary also updates the formerly unanswered request with the new supplied
outcomes. It neither invents those earlier topics nor treats them as new evidence.
Existing bounded closing-trailer recovery preserves both vetoes; these are not
unrecovered parse failures. The defective F3 summaries remain genuinely defective
despite their extra spurious continuity complaint.

### The accepted-output review changes the readiness interpretation

- **Task 19, retained granular case 1, repetition 2:** the second episode title
  repeats F1's unsupported exclusivity claim. Its only citation is message 323.
  Both semantic and final format screening pass. Root and the independent agent
  agree this is a known semantic false accept in an actual accepted pipeline,
  not just an artificial control failure.
- **Tasks 3 and 4:** no additional concrete fidelity violation was found in the
  other two accepted retained outputs. The four accepted synthetic outputs were
  reviewed previously. This is not proof that every possible defect was excluded.
- **Tasks 2 and 17:** the first episode cites only message 214 but imports a trip
  assertion and location claims from outside its authorized visible span. Three
  valid semantic envelopes (task 2 call 3; task 17 calls 3 and 5) approve both its
  title and content. These tasks ultimately fail on the summary, so this is
  three item-level false accepts across two rejected pipelines, not three
  additional accepted pipelines. A summary-only repair cannot correct those
  unchanged episode fields.

Consequently, the historical **7/16 pipeline contract acceptance** and **8/16
control matches** remain accurate recorded measurements, but neither is a
semantic-quality score. **One of three accepted retained outputs** (one of seven
accepted outputs overall in this small campaign) has a confirmed unsupported
claim. Do not rewrite the old report or present a higher pass rate as success
without this separate source-grounded quality assessment.

Several rejected retained outputs have independent content or format problems,
including item citation leakage, weakened trip ordering, and a sentence followed
by a fragment. An unjustified continuity complaint does not make the whole
candidate faithful. Task 2's repair also grows to 543 characters; task 17's
360-character repaired summary still accompanies the unchanged invalid episode.
Simply suppressing continuity vetoes would therefore expose rather than solve
some defects.

### Confirmed structural failures and limits of the diagnosis

The three unrecovered retained parse failures (tasks 1, 11 and 12) end with a
closing object delimiter where an array delimiter is required. They are not just
missing an append-only trailer; broadening the existing closure repair would
change its safety contract. The three diagnostic failures (tasks 26, 27 and 32)
combine the unsupported-claim code with a prior-summary reference, which the
strict issue schema deliberately forbids. Both distinctions are reproduced
offline with the frozen validator, without changing its behavior.

The evidence now separates three interacting problems: generation/repair failing
the hard length or fidelity requirements; incorrect semantic judgments despite
correctly delivered evidence; and decision replies becoming invalid because of
their attached diagnostic structure. The coupling of verdicts and diagnostics
is a plausible contributor, not a proven cause of the semantic errors. These
captures do not identify a provider-side mechanism or establish that every
incident was caused by a code regression.

Private receipts:

- Review bundle: `/private/tmp/hymem-reliability-map.hQ8e5w/approved-retained-review.json`;
  SHA-256 `d2403ee9b103a954b6b80a8085093b3dd3261a779a282ad6f8927a6d3ff1af5f`.
- Root mechanical report: `root-retained-review-checks.json` in the same directory;
  SHA-256 `5d676662f25718f2c8e5d95cc570700b581e5218b33a6f03a8ae7dd11bcecfbf`.
- Mechanical checks contain only IDs, field paths, counts and enums; they do not
  automate semantic ground truth. Root and independent-agent review provide the
  semantic conclusions above. Raw evidence remains outside committed artifacts.

## Next gate: one versioned experiment, not a production fix

Do not run the full LME baseline, relax acceptance thresholds, or authorize
spending from this survey. The initial campaign is complete and its authority is
spent. Source-grounded adjudication now supports retaining all eight original
control labels without modification.

The independent architecture audit recommends one candidate hypothesis:
**separate decision-only semantic screening from optional source-linked repair
diagnosis**. Keep primary generation, compaction, citation authority, final format
screening, one repair, atomic publication and failure cursor behavior unchanged.
Only a summary-only veto may reach the separate diagnosis; malformed diagnosis
cannot approve anything or override a veto. Reverify the complete repaired
candidate. This targets the observed decision/diagnostic coupling, not all length
failures, and may fail to fix semantic calibration. Those are testable limitations,
not reasons to stack more changes in the same candidate.

Its normal successful path would remain three completions, but the longest path
would become **seven**, not six. Any implementation/comparison must explicitly
version its stage/producer identity, reserves, HTTP budget, replay and tests, and
retain the 120-second invocation deadline. No silent budget increase, reuse of
spent authority, SQL migration or deployment is proposed here. Implement only
after adjudication, using a new implementation agent and independent root gates;
then request a fresh interleaved baseline/candidate spending allowance. A passing
component comparison would still need held-out confirmation and end-to-end
convergence/retrieval/answer-path smoke before canonical LME readiness.

### Candidate hypothesis and rejection criteria

The candidate's decision reply would contain only the four complete verdict
groups with exact index/verdict objects. If and only if a valid decision supports
every episode/procedure but vetoes the summary, a separately identified bounded
call may propose source-linked summary issues. That call cannot change a verdict,
add item authority or publish anything. Invalid/missing/unactionable diagnostics
leave the original veto in force. At most one existing summary repair follows;
the entire assembled candidate must then pass decision-only re-screening and
the mandatory unchanged candidate-only format screen. A second veto is terminal.

The testable prediction is fewer malformed verdicts and fewer diagnostic-schema
failures on decision calls. A semantic improvement from reduced instruction load
is only a hypothesis. Length failures are explicitly outside this candidate's
target; they remain failures, not exclusions. The extra diagnosis call can also
increase latency, budget consumption or deadline failures and must be measured.
If eventually deployed with the unchanged six-attempt durable retry limit, the
theoretical worst case per stuck slice rises from 36 to 42 logical completions,
excluding HTTP retries. The comparison performs one bounded attempt per scheduled
task, not those durable retries. Do not interpret a shorter decision prompt as an
established reduction in total cost per faithful accepted slice.

Before live comparison, the separate implementation agent and root must verify:

- Complete verdict coverage, unknown keys/indices, mixed item/summary vetoes,
  uncertainty, missing or malformed diagnosis, exact quote and citation bounds,
  untrusted hints, and adversarial prior/boundary scope.
- No item/procedure mutation in summary repair; no same-candidate approval from
  diagnosis; no re-repair loop or partial publication; unchanged failure cursors
  and source inputs; format screening only after complete semantic support.
- Explicit stage attribution and versioned producer/derived identities, seven-call
  worst-case accounting including HTTP retries, one absolute invocation deadline,
  late-receipt handling and owned-worker cleanup. Existing six-call authority must
  fail closed for the new candidate, not silently authorize extra work.
- Regression gates on the precise frozen candidate, plus root-owned adversarial
  checks. Scripted tests establish control flow and protection, not model accuracy.

The interleaved comparison must preserve paired evidence/sampling and every
repetition. Record syntax/schema failures, false accepts, false vetoes, full
pipeline results, stage latency and cost separately. Inspect accepted outputs
against sources, not merely their verifier verdicts. Any observed acceptance of
the known F1/F2 defects, newly ungrounded accepted output, or lost protection is
a blocker to adoption, even if the candidate's contract-pass rate increases.
Persistent F3 false vetoes also mean semantic calibration remains unresolved.
The original retained cases are development cases now, not holdout validation.
No claim of comparative superiority or canonical readiness follows from two
repetitions or a single clean canary.

## Local candidate implemented; independent offline gate reconciled

The user's "Run it" advanced the isolated candidate to implementation. A
separate runtime implementation agent changed only `hymem/dreaming/digest.py`
and `benchmarks/episode_probe.py`, adapting the affected synthetic fixtures and
adding implementation tests. A separate accounting agent updated the diagnostic
planner and its tests. Root reviewed both implementations and added independent
adversarial tests. Existing unrelated dirty worktree changes are preserved.

- Decision wire identity: `digest-fidelity-decisions-v9`; exactly four complete
  index/verdict arrays, with inline issues and extra fields rejected.
- Diagnosis identity: `digest-summary-source-linked-diagnosis-v1`; exact source,
  candidate, item and prior snapshot, strict issues-only response, no approval
  ability or closing-trailer salvage. Missing, empty or misbound hints hold the
  digest without a repair.
- One summary-only repair, then full semantic re-screening and unchanged final
  format screening. Three normal completions, seven maximum; same absolute
  deadline. No publication/cursor or SQL migration change.
- Probe record v5 attributes diagnosis and second verification separately.
  Root found that the inherited nested exception wrapper could erase the inner
  stage; the implementation agent corrected it, and root tests now cover exact
  diagnosis, repair, re-screen and format exception attribution.
- Planner v2 requires explicit per-arm contracts in serialized plans. Baseline
  reservations remain 6 completions/18 HTTP attempts; candidate reservations are
  7/21; controls remain 1/3. Closed v1 plans cannot silently load as v2. Stages
  mean dispatched calls, not local pre-dispatch checks, and contradictory earlier
  failure attribution fails closed. Reports distinguish an inapplicable baseline
  diagnosis stage from a censored candidate stage.

Root's new `tests/test_digest_decision_diagnosis_root.py` passed **55 independent
synthetic tests**, including exact source/item preservation, malformed/forged
diagnostics, obsolete inline replies, terminal vetoes, all seven deadline
positions, no publication, and new diagnosis policy/dispatch identity. That
focused result is not the integrated regression gate and does not measure real
model correctness.

Root's mechanical frozen-baseline audit also passed:

- The complete semantic evidence-rule string is byte-identical to the frozen
  baseline. Primary prompt files and compaction, repair and format prompts are
  unchanged.
- Source-window construction, fidelity payload construction, candidate validators,
  source-linked issue validation and closing-trailer parser implementations are
  unchanged (parser documentation was updated, not its behavior).
- Runner, generic deadlines, Phase-1 producer, shared JSON parser and semantic
  identity machinery are unchanged. Existing identity machinery binds the new
  digest implementation; synthetic mutation tests confirm that diagnosis changes
  affect digest identity without changing Phase-1.
- All **13** captured historical retained replies containing inline issues remain
  rejected by the new decision contract. This is mechanical compatibility testing,
  not rerunning the model or retroactively improving the old campaign's results.

Private verification root: `/private/tmp/hymem-separate-diagnosis.pasOQ3`.
The independent integrated gate freezes all application, benchmark and test
Python/SQL inputs, clears credentials from its environment, prohibits parent
non-loopback networking, and records collection/execution counts, JUnit results
and post-run hashes. Local process-supervisor tests use invented workers. Its
results are preserved and reconciled below. This is a composite selected-regression
gate, not a single clean run of the entire repository suite.

### Independent gate results

- `integrated_v1`: **2,740 passed, 5 failed**, no errors or skips, out of 2,745
  selected regression cases. All 463 source/test inputs remained unchanged.
  Two probe fixtures still expected the old call count and lacked a response for
  the new diagnosis request. Three dummy transport tests could not bind their
  local HTTP server because the sandbox denied localhost sockets. The failed
  receipt remains failed; it has not been relabeled as a pass.
- The implementation agent corrected the probe fixtures and added two explicit
  malformed-diagnosis cases. **No runtime or planner code changed.** The only
  source/test input difference between the broad and final gates is
  `tests/test_episode_probe_root_controls.py`; no other checked module imports
  that test module.
- An intermediate localhost-enabled transport rerun passed all 61 tests, but its
  all-input stability gate rejected it because the unrelated fixture edit
  occurred during that run. It is not used as acceptance evidence.
- `final_supplement_v1`: **276 passed**, no failures, errors or skips, in 59.52 s,
  with localhost access limited to dummy test servers. It reran the entire
  changed fixture module, both supervisor modules, all 55 root-owned candidate
  checks and all 143 planner checks. All 463 source/test inputs remained stable;
  every original failing test now has a passing receipt.
- Root mechanically reconciled test identities and hashes: **2,747 distinct
  selected cases have passing receipts on the reconciled inputs**, including
  the two added cases. Unchanged runtime and unchanged unaffected tests permit
  retaining the other broad-run passes. This is not a claim that all 2,747 ran
  together in one clean invocation, nor that the entire repository suite ran.

No provider calls occurred; the audited parent made zero non-loopback network
attempts. These gates do not measure false accepts/vetoes from the live model,
prove an LME score improvement, or establish end-to-end convergence. No deployment,
restart or production-store access occurred.

Private receipts under the verification root:

- Broad-run input manifest: `913f32e915b3ea57d8dd6bc9e1b3d447b31e65d8da24ad1c410e1f42961c437f`.
- Final input manifest: `4c89998708f177aa6d799d2a076a83cdb772fa49618d26e0382b0842d547fa5e`.
- Final JUnit: `15b3558ee5571a97ed1249783d5190f370881c83d6014f243b2ca1ba4209c144`.
- Composite report (`root-composite-verification.json`):
  `9c463e5dd8c6d20cabc2b63e170f06b813648e11299e4f95d7b546a097a73f63`.

Current candidate hashes:

- Digest: `8d42f1ece8bd8b97c572ac93ef395f406d0b7ab3b69738fee396d0e2a2aa2f0d`.
- Probe: `9c78be59e20ac4b0eee22500c412770dd226bff98ccd2c4de90f81ecade1e443`.
- Planner: `51e30b6d002daf47fb87033f33056a2f708ea7723af298bfee358661e975d193`.
- Root tests: `e3b1d618acd469f796150e136a7427b2f73cf497fc85f2a3ff33fa4377ce34d0`.
- Mechanical compatibility report: `f5453f42aa61dd3c6a795b8d18a94e7cb75e494e9bf92e1ac6e8f0ab75d6be85`.

### Approved fresh comparison boundary

Root requested fresh conditional approval for the same 16 cases, two repetitions,
both frozen arms interleaved: **64 invocations**, at most **240 completions / 720
HTTP attempts** (16 baseline pipeline invocations × 6, 16 candidate × 7, plus 32
single-call controls). Each invocation retains the 120-second absolute deadline
and two-second cleanup allowance. Provider/model and all sampling parameters
stay unchanged. Independent model rejections continue through the predeclared
matrix; unsafe cleanup, drift, untrusted receipts or unknown/over-budget usage
halt it. No production memory, deployment, restart or full LME run.

The user subsequently replied **"RUn it."** to the final response explicitly
requesting the 240/720 comparison limit. This is fresh approval for the stated
64-invocation comparison, not a budget renewal for the old campaign or authority
to deploy. At the initial preparation checkpoint no new paid calls had occurred.
The old 112/336 authority remains spent. A fresh paired adapter must support separate
frozen roots and both contracts, pass its independent offline integration gate
and no-provider rehearsal, and bind new inputs/identities/consent before dispatch.
Neither the old adapter's tests nor the new planner's synthetic simulation alone
establish that integration. LME remains uncleared.

The independent read-only adapter assessment identified a necessary isolation
boundary: the old single-arm helper imports application code during preparation,
parent-side replay and verdict observation, not only in the paid worker. Merely
switching `sys.path` between frozen roots would reuse cached modules from the
first arm. Fresh arm-specific processes must perform every application-dependent
operation; the parent can retain only planner, reservation and metadata duties.

The fresh adapter must also:

- Authenticate each arm's source manifest and gate separately, and select
  prepared requests by arm plus case. Rebuild controls with that arm's verifier
  contract while preserving exact candidate/evidence/labels. Compare immutable
  evidence independently of intentionally different prompt/schema bytes.
- Exclude the arm-specific call maximum from the shared configuration digest;
  bind it instead through the explicit pipeline contract and source identity.
  The old helper incorrectly assumes one shared six-call maximum for this new
  comparison use case.
- Recognize candidate diagnosis separately from semantic verdicts; reject any
  baseline diagnosis call; use `task.stages` and task-specific reservations.
  A pre-dispatch cap is not a paid stage failure and cannot inflate usage.
- Bind a fresh 64-task entry point/authority to both roots, current helper hashes,
  a mixed-arm rehearsal, and actual consent. Derive rehearsal call counts from
  the frozen schedule, not twice the previous campaign's count.
- Fault-inject swapped roots, cached imports, stale v8/v9 requests, wrong roles,
  candidate calls under baseline authority, missing usage and unclean worker or
  replay termination. Keep all old receipts and spent authority unchanged.

These are fresh comparison integration requirements, not evidence that the old
single-arm survey used the wrong application or that a new live run has begun.

### Paired runner preparation

Two separate implementation agents adapted the application-free paired runner
and its outer authorization/dispatch wrappers. Root reviewed both, added 27
independent adversarial checks, and caught a source-manifest/file-hash confusion
in authority preparation before dispatch. The correction has a regression test.
The frozen helper gate passes **237 tests locally and 237 on isolated Linux**,
with exact input/test inventories, no failures/skips and no provider calls.

- Fresh host stage: `/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-paired-comparison-20260916-RUsvWu`.
- Local helper/evidence staging: `/private/tmp/hymem-reliability-paired-adapter.o9DYl2`.
- Baseline source: `dc52f75d5036486ff3ca0d58773640781edcb61f10415ea63451f0afe7702225` (458 files).
- Candidate source: `107f047dad08e06277cc62be1983149b7e2a015c2871adcc38e41f6a145929d9` (463 files).
- Seven-helper manifest: `b9d32e50ec6ebb6854ae3ab76c8c632c70fa513589aa29ebb9c1eba7108d9a1b`.
- Reviewed helper/test inputs: `eb96aa0b4d4beb612be5df956a2f63576391c4d4e74976f56ce2913f29fffdc5`.

The original baseline, benchmark evidence and spent authority remain unchanged.
Only the copied synthetic database location was relocated, with the original
fixture metadata and hashes preserved. The same primary requests and source
proofs must match byte-for-byte across arms; controls use each arm's own verifier
contract but the same evidence, candidate and preregistered labels.

The no-SDK rehearsal completed all 64 tasks using 152 explicitly synthetic
completions and zero HTTP attempts. The second gate also completed all 64 tasks:
the exact 152 recorded request/response pairs passed through the maintained SDK
against a dummy localhost server in the network-disabled container. It used 152
dummy HTTP attempts, no provider requests or credentials, and the server and all
worker/replay processes were reaped. Root independently reconciled 1,889 receipt
files, unchanged inputs, per-arm evidence, deadlines, usage and Linux isolation.

- Supervised plan: `b6e32f78ec54b7502ef7f535a1830b0324ec603f077899274ee136c6610febbd`.
- Root preauthorization audit: `eda8c870712a003790cb0234c77639b9646cf23c62c4be3437cd06e1eaa8389c`.
- Fresh authority: `2ab92ce395afebadeeefe5cd438b5b7a8693fac059477d266021af9f25fcb36d`.

The one-shot live dispatcher accepted that authority and ran on Afrodite
(host supervisor PID 1554235). Its completion and quality findings follow. No
production modification, service restart or full LME run was performed. Rejected
cases remain rejected in the receipts; there was no automatic reroll.

### Completed paired comparison — adoption gate not met

All **64/64 invocations** finished. Root independently reconciled receipts,
per-arm source identities, accounting, exact replay and worker cleanup. The
dispatcher exited 0, all owned workers/replay processes were reaped, and frozen
inputs remained unchanged. There were no infrastructure failures, HTTP retries
or unknown-usage events. This is successful execution of the experiment, not a
passing product-readiness result.

The run used **139 completions / 139 HTTP attempts**, with 277,803 prompt tokens
and 21,662 completion tokens (299,465 total). All 139 finish reasons were `stop`.
The one-shot 240/720 authority is now consumed and closed; unused headroom is
not permission for a reroll, resume or further experiment.

| Observation | Frozen baseline | Separate-decision candidate |
| --- | ---: | ---: |
| Pipeline contract passes | 7/16 | 8/16 |
| Controls matching preregistered expectations | 6/16 | 8/16 |
| Invalid semantic-verifier envelopes | 9/28 | 0/29 |
| Accepted outputs with confirmed grounding violations | 5/7 | 5/8 |
| Completions / HTTP attempts | 67 / 67 | 72 / 72 |
| Total tokens | 152,007 | 147,458 |

Semantic-verifier denominators include pipeline screening, re-screening and
semantic controls, not format-only calls. The candidate made three separate
diagnosis calls. Baseline pipeline failures were three length-cap failures, two
content rejections, one format rejection, two parse failures and one diagnostics
failure. Candidate failures were two length-cap failures, three content
rejections and three format rejections.

The paired pipeline outcomes were six both-pass, seven both-fail, one
baseline-only pass and two candidate-only passes. The net extra candidate pass
came from the second `stress-near-prior` repetition: the unchanged primary
generation happened to fit the cap, whereas baseline generation and compaction
did not. The `digest-1-blob` outcomes swapped across repetitions, cancelling out.
This does not establish a causal end-to-end benefit from splitting verification
and diagnosis. The two extra matched controls were valid rejections of the F3
defective summary where baseline replies were malformed.

Important unresolved control results:

- Both arms accepted the unsupported F1 exclusivity title in both repetitions.
- The candidate accepted the uncited F2 episode claim in both repetitions; an
  unrelated summary rejection blocked the whole control. That is not successful
  detection of the intended defect.
- The candidate falsely rejected the faithful F3 summary twice. Baseline's
  corresponding replies were invalid; neither arm passed that control.
- Both arms matched all four F4 grammar-control expectations.

### Grounded review of every accepted pipeline output

A separate agent reviewed all 15 accepted outputs against the exact final
verifier payloads and permitted evidence. Root independently inspected the
sources/rules and reconciled the review to all accepted task IDs. Accepted
summaries, episode fields and citations match those payloads; all 15 outputs
contain zero procedures.

Ten accepted outputs have a confirmed grounding-contract violation, five per
arm. These findings concern evidence authority, not a claim that every affected
statement is false in the outside world:

- Four titles add all-platform exclusivity where the cited source establishes
  availability on one platform and absence from another (tasks 7, 8, 37, 38).
- One baseline summary adds unsupported viewing history and a causal relation
  (task 5).
- Five episodes import an actor name from the prior derived summary even though
  their own cited messages and attribution metadata do not establish that name
  (tasks 13, 14, 39, 43, 44). Prior-summary continuity is permitted for the rolling
  summary, but explicitly forbidden as episode evidence. Several also add an
  unsupported gendered pronoun.

Four dense-inventory outputs have no identified semantic defect in this review.
One other candidate output has an uncertain conditional-advice compression,
kept separate from confirmed errors. Possible subject-elliptical sentence issues
are also recorded separately rather than counted as semantic defects. These
small, repeated cases are not a population accuracy estimate.

**Decision:** keep the candidate experimental and LME uncleared. The wire-format
improvement is real in this sample, but semantic verification, length handling
and false-veto behavior still fail the adoption gate. No production change was
made. The next design to evaluate should isolate each item's allowed evidence
from rolling-summary context, with source-grounded false-accept and false-veto
controls before another paid comparison. This recommendation is not a new run
or deployment authorization.

Private receipts under `/private/tmp/hymem-reliability-paired-adapter.o9DYl2`:

- Live summary: `04945e497433ad73267fac431f2ea5db19b9456900b8d1d2140b33b6a6edcf0f`.
- Independent `root-live-audit.json`: `0b057533efc74179e72fac762750681e21b7bbe072b7a6ce74c6f1681012fc78`.
- Grounded-review input: `214085ec24876fb9afd34858b66eb1c5b9f6d9452bae14be6fba31759980838d`.
- `root-audits/quality-review.json`: `ac20146686a9b74017d25229821a60ae35de61f45c5709dfeef44d0f83c86a63`.

Raw benchmark text remains in private diagnostic artifacts, not this report.

The subsequent user request starts the offline, diagnostic-only
[evidence-isolation experiment](2026-09-16-digest-evidence-isolation.md). It does
not reopen this comparison's spent authority or supersede its negative adoption
decision. Production verification and full LME remain unchanged.

The follow-up fixed-candidate isolation comparison completed after local/Linux
verification and explicit provider/payload authorization resolved the earlier
upload and paid-launch blocks. It ran all **84 invocations / 162 completions /
162 HTTP attempts** without infrastructure, parsing or accounting failure.
Independent root receipt/actual-parser audits and a separate raw-reply recount
agree: baseline **31/56** target decisions correct, isolation **30/56**. Isolation
reduced false accepts from 25 to 22, but introduced four false rejections and
used 1.56× the tokens. Both arms missed all 20 retained known-defect observations.

**Decision: do not adopt the isolated candidate or clear full LME.** This is a
semantic-judgment failure despite valid output and mechanically correct evidence
isolation, not another transport/parser failure. Production is unchanged. The
authority is consumed; no reroll or new campaign is authorized. See the
[experiment report](2026-09-16-digest-evidence-isolation.md) for the preregistered
controls, correlations/limitations, exact scope, receipts and next design gate.

The subsequent "Do whatever is next" request led to a diagnostic-only
[claim/evidence-ledger prototype](2026-09-16-digest-evidence-ledger.md), implemented
by a separate agent and independently verified by root. Its offline gate passed
**2,117 tests / 46 selected modules**, with no failures/errors/skips, no network
attempts and all 464 existing Python/SQL inputs unchanged. Root audited all 60
request projections from the 21 frozen comparison cases, plus 26 projections
from 12 new synthetic development controls. No model was invoked.

The ledger requires exact candidate-text coverage and source-linked quotations;
it deliberately cannot mark semantic truth verified or authorize publication.
A genuine irrelevant quote can still pass its structural checks. The next gate
is a newly authorized, frozen, interleaved live evaluation with source-grounded
review, followed by separately held-out and end-to-end confirmation if promising.
**Offline diagnostic implementation accepted; runtime adoption and LME remain
uncleared.** No paid authority was reused and no production change was made.

The subsequently approved [ledger live comparison](2026-09-16-ledger-live-comparison.md)
completed 132 invocations / 238 completions / 238 HTTP attempts. Independent
receipt audit and actual-parser replay passed, with no execution/accounting or
cleanup faults. The semantic/contract result is **not suitable for adoption**:
baseline 54/80 matched target decisions versus ledger 19/80. Ledger reduced
target false accepts 26→2 but blocked 33/42 supported targets through malformed
or uncertain outputs, using 2.18× the tokens. First parser failures were 59
unexpected root fields, eight repeated/ambiguous quotes and one lost space.
Genuine quotations still failed to expose unsupported causality and attribution
inside some claims; aggregate uncertainty can hide supported erroneous subclaims.

**Do not deploy the ledger or clear full LME.** Next work is an offline contract
redesign that places mechanical rendering/source coordinates in code and tests
positive acceptance alongside false-accept detection. The frozen diagnostic
results must remain intact; any further paid evaluation requires fresh authority.
Production is unchanged and the 238/714 authority is consumed.

The requested next offline step is complete: the
[compact evidence-assessment prototype](2026-09-16-digest-evidence-assessment.md)
was implemented by a separate agent and independently reviewed/tested by root.
The final selected gate passed **2,359 tests / 48 modules**, zero failures,
errors or skips, with network and credential access prohibited. Root audited
all 33 frozen payloads / 86 requests and verified that the prior 468 Python/SQL
inputs and paid receipts remained unchanged. Code now owns text and source
coordinates; models return compact, source-linked judgments with separate
relation, outcome and retention obligations. A too-small-output-budget
preflight bug was caught, fixed and verified before acceptance.

This accepts only the deterministic offline implementation. No live semantic,
format-reliability or cost improvement is established, and no runtime deployment
or LME readiness is authorized. Retention needs its own gold labels before
scoring; a new paid comparison requires fresh explicit authority.

The next [offline evaluation preparation](2026-09-16-assessment-evaluation.md)
is complete: 33 unchanged development cases plus 16 new paired controls; old
family labels, specific defect witnesses and source-retention labels stay
separate. Three ambiguous supplemental retention labels are explicitly unscored.
Independent review caught and fixed duplicate scoring from overlapping targets
within one view. **754 selected tests passed**, with all 471 previous source/test
inputs unchanged and no provider calls or external transfers. The frozen proposed
comparison needs fresh **326-completion / 978-HTTP-attempt** authority, plus a
verified fresh runner before execution. This is not a paid launch, deployment
or LME-readiness result.
