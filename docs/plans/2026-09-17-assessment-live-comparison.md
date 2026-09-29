# Compact-assessment comparison — completed, not approved for adoption

## Decision

**Do not replace the current verifier with this prototype or declare LME ready.**
The approved diagnostic completed and its execution/accounting/cleanup audit
passed. The compact verifier narrowly passes the preregistered legacy
false-accept comparison, but fails both the zero-invalid-response and
all-controlled-mechanisms gates. It consumes **2.507× the baseline's tokens**.
On the older development cases there is no net gain in correct target decisions.

These are results for fixed benchmark candidates, not a full LME run, a
representative held-out evaluation, or proof about end-to-end memory scores.
No production code/store was changed and no Hermes service was restarted.

## Scope and execution

The user explicitly answered “yes” to the 49-case benchmark-only transfer to
`https://api.deepseek.com`, model `deepseek-v4-flash`, capped at 326 completions /
978 HTTP attempts. The separate confirmation superseded the insufficient
earlier preparation approval; it did not reuse a spent campaign authority.

The frozen September 16 protocol, 49 cases, gold labels, 476 source files,
request bytes and parameters remained unchanged. Two paired repetitions ran
once, with no semantic retries, resumed cases or rerolls.

- **196/196 case-arm tasks; 326/326 completions; 326 HTTP attempts.**
- All 326 finish reasons were `stop`; no transport retries or truncation.
- 583,844 prompt + 25,311 completion = **609,155 total tokens**.
- Entry-process elapsed time: **1,053.81 seconds**, approximately 17m34s.
- No unknown/unreconciled usage or cleanup warnings.
- Independent OS probe: all 196 recorded worker groups empty, entry process
  absent, host supervisor and Docker-exec processes absent.
- Hermes container identity/image unchanged and running. This is not an
  additional production-store health audit.

## Comparison results

Every repetition has 56 legacy targets: 30 supported and 26 unsupported.
Malformed/invalid or uncertain responses are misses, not successful defect
detection. The repetitions and case variants are dependent; do not interpret
these counts as independent population-accuracy estimates.

| Legacy metric | Baseline R1 | Compact R1 | Baseline R2 | Compact R2 |
| --- | ---: | ---: | ---: | ---: |
| Correct target decisions / 56 | 42 | 43 | 41 | 42 |
| Supported targets accepted / 30 | 30 | 30 | 30 | 30 |
| Unsupported targets falsely accepted / 26 | 13 | 12 | 15 | 12 |
| Unsupported targets correctly rejected / 26 | 12 | 13 | 11 | 12 |
| Invalid target verdicts | 1 | 1 | 0 | 1 |
| Uncertain target verdicts | 0 | 0 | 0 | 1 |

The narrower preregistered development gate therefore passes: fewer false
accepts without lower acceptance of supported legacy targets. **This alone is
not a promotion criterion.** In particular, turning a false acceptance into a
contract-invalid answer is not correct semantic rejection.

Cohort breakout exposes the limited benefit:

| Correct legacy decisions | Baseline R1 | Compact R1 | Baseline R2 | Compact R2 |
| --- | ---: | ---: | ---: | ---: |
| Previous development targets / 40 | 27 | 27 | 27 | 26 |
| Newly invented targets / 16 | 15 | 16 | 14 | 16 |

The compact model catches the specific old defect witnesses only **6/19 in R1
and 5/19 in R2**. Twelve are falsely accepted in each repetition, one is
contract-invalid in each, and one additional witness is uncertain in R2.
Coarse family results cannot be substituted for these specific defect checks.

Separate, deliberately overlapping compact reporting views:

| View | R1 correct | R2 correct | Failure detail per repetition |
| --- | ---: | ---: | --- |
| New primary checks | 15/16 | 15/16 | One false acceptance |
| All labelled retention | 21/25 | 21/25 | Three false acceptances, one false rejection |
| New-only labelled retention | 15/17 | 15/17 | One false acceptance, one false rejection |
| Present-claim auxiliary controls | 12/12 | 12/12 | None |

**Do not pool these views.** The primary and retention failures for the omitted
prohibition are the same underlying check, not independent discoveries. The
three pre-excluded ambiguous retention labels stayed excluded, as preregistered.

## Consolidated failure diagnosis

1. **Evidence authority still leaks through model judgments.**
   `F2-own-citations-defect`, episode 0, is contract-invalid in both repetitions.
   Both responses are valid JSON with all eight expected check IDs, valid
   verdict strings and known source IDs. But `c5/c6` claim support for the Big
   Sur entity using only `s2`, which is interpretation-only boundary context.
   The guard correctly rejects the whole scope. These are not API errors,
   token-ceiling cuts or malformed JSON: they used 108/114 completion tokens,
   one HTTP attempt each, and finished with `stop`. Other judgments also endorse
   the independent trip claim using mixed canonical/context citations; relaxing
   the structural guard would not fix the underlying evidence-policy failure.

2. **Material omissions remain invisible to some retention judgments.**
   `assessment-prohibition-omission-defective` omits an explicit prohibition
   from a procedure. Its `c6` retention check incorrectly returns supported in
   both repetitions. The old stated-claims criterion intentionally still labels
   the surviving steps supported; that old label must not be rewritten to make
   a misleading baseline comparison. Separately, the old
   `F3-summary-outcome-defect` still reduces supplied recommendations/routes to
   requests, and both omitted answer-source units are incorrectly marked retained.

3. **Per-source retention is not reliably localized.**
   `assessment-repeated-unicode-units-defective` correctly rejects the changed
   Ω-18 status, but also incorrectly rejects retention of Ω-17's unchanged,
   correctly preserved closed status. This happens in both repetitions. It is
   evidence of a nonlocal retention judgment, **not proof that Unicode or
   tokenization caused the failure**.

4. **Older semantic errors persist despite valid output structure.**
   False acceptances include availability expanded to global exclusivity,
   prior-only identity/uncited gender treated as episode authority, and invented
   viewing status/causality. Fine-grained check IDs and valid evidence IDs do not
   establish entailment. `fresh-exclusivity-scope-defective` regresses against
   baseline in R1; `fresh-citation-removal-defective` becomes uncertain rather
   than correctly unsupported in R2. `retained-39` improves only in R1.

The first three findings were independently reviewed against raw responses,
frozen source projections and gold, then checked by root. No post-hoc label
change or parser relaxation is justified by these results.

## Preregistered gates and cost

- Execution/receipt integrity: **pass**, following the audit-tool correction
  described below and independent process-liveness verification.
- Zero contract-invalid compact responses: **fail**, 2/228, both the same F2
  scope in different repetitions. Baseline has 1/98 invalid responses.
- All new controlled mechanisms correct: **fail**. Two distinct new controls
  recur as failures; overlapping reporting views must not inflate that count.
- Fewer legacy false acceptances without lower supported acceptance: **pass**
  under the preregistered narrow rule, with the cohort limitations above.
- Runtime adoption / deployment / LME readiness: **not authorized or established**.

| Actual resource use, both repetitions | Baseline | Compact |
| --- | ---: | ---: |
| Completions / HTTP attempts | 98 / 98 | 228 / 228 |
| Prompt tokens | 167,926 | 415,918 |
| Completion tokens | 5,770 | 19,541 |
| Total tokens | 173,696 | 435,459 |
| Sum of owned invocation durations | 411.94s | 576.27s |

Compact uses 2.327× the calls, 2.507× total tokens and 1.399× summed invocation
time. Total-token cost per correct legacy target is 2,067.88 versus 5,064.21 in
R1, and 2,118.17 versus 5,183.29 in R2. These are per-view resource ratios, not
pooled accuracy or dollar costs. Billing prices/cache discounts were not inferred.

## Audit-tool defect found and fixed after collection

The first independent audit rejected every call because it compared
`time.time()` call/attempt timestamps directly against `time.monotonic()` worker
ownership/finish timestamps. The paid collector and watchdog use these clocks
consistently; the new audit check did not. Its synthetic tests used one numerical
clock scale, while the prelaunch replay exercised the semantic parsers rather
than the complete independent receipt auditor. This was a verification gap in
our diagnostic tooling, not a new production failure or failed model invocation.

A separate agent fixed the auditor; root reviewed and reran **199 tests**
(86 audit + 113 scorer). These replace, not add to, the earlier 183-test selection.
The fixed auditor separately checks wall chronology and wall entry lifetime,
and monotonic ownership, completion-specific request intents, deadlines and
cleanup. Every call/intent/result remains hash-bound to the synchronous frozen
collector and worker commit. It explicitly reports that no paired-clock sample
exists; it does not estimate an offset or claim an absolute per-call conversion.

Root additionally ran the exact fixed auditor against **all 392 actual offline
tasks / 652 simulated requests** on Afrodite; both rehearsal modes passed. This
check was performed after the live collection, not retroactively before launch.
**Future prelaunch gates must audit the actual rehearsal receipts with the same
independent auditor used for final acceptance**, in addition to semantic replay.
The new read-only regression helper is `root_audit_offline_receipts.py` in the
private stage. Clock-domain fixtures now use realistic distinct origins and
adversarial lifetime, ordering and intent-binding controls.

All raw receipts, runtime helpers, source, labels and the semantic scorer were
unchanged by this fix. The three empty report sinks from the rejected first audit
remain preserved; verified reports have `-verified.json` names. No paid call
was repeated and no semantic gate was weakened.

## Next work, without another immediate paid reroll

Retain the current runtime. Use this complete failure map to design one coherent
offline change targeting evidence authority and per-source outcome/prohibition
preservation, with both faithful and defective controls. Keep the strict
context-only evidence guard. Do not count a source ID, structurally complete
output, an unrelated negative judgment or uncertainty as semantic proof.

Any subsequent candidate should first pass actual-receipt audit integration and
all frozen offline contract/regression checks, then receive a new preregistration
and separately approved untouched-case evaluation. These results do not justify
another unchanged reroll, production adoption or jumping to a definitive LME run.

## Receipts

Local results: `/private/tmp/hymem-assessment-results.bGPR1t`.
Local diagnostic code: `/private/tmp/hymem-assessment-eval.dHR3MU/verified-run`.
Remote stage:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-assessment-comparison-20260917-dHR3MU/verified-run`.

- Raw archive, 2,059 receipt files:
  `ef45dd64dc28db599040248e64f007f6dcae774ed156561c38eab0c8f80b14a8`.
- Execution plan:
  `3712f046866af1b5b3c851dd0b7b49886a43acf622e0e335e347f72cb41afef0`.
- Consumed one-shot authority:
  `49a9d473772511363e1749e6ee8145184b7ef5b66d53aebafe58f41f3d5f10bd`.
- Corrected independent auditor:
  `77fde8a9d3a6ed396bb0f87c148b40fb828e1a64029bf162414819d975c33bee`.
- Verified audit:
  `6fb5dad2be0e5ab66bbd9492e9fa4b7dd63910f8b7a499e4bee09b4ab3542b6c`.
- Verified private target score:
  `f63c7a7328f04feb656133c82de0c0c9cbd70206f9036e4e0e78bca51fc0706a`.
- Verified aggregate score:
  `fb4e7986ad297bd7ee06441f0b33767229c638fd7fcaca4121875ffd797cabe4`.

Raw benchmark text remains in the private diagnostic receipts, not this report.
