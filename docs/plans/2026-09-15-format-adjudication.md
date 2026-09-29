# Format-only false rejection repair

## Confirmed incident

The approved single paid diagnostic halted after one digest invocation with
`summary_format_unsupported`. User-authorized inspection of only the compacted
summary and verifier verdict confirmed a grammatical 389-character sentence,
with explicit subjects, an allowed semicolon and no Markdown/output quotes.
All factual verdicts were supported. The earlier implicit-subject hypothesis
does not explain this response. No new provider call was needed to inspect it.

## Implementation and independent verification

1. A separate implementation agent adds one bounded, candidate-only format
   adjudication after an otherwise valid first-pass verdict rejects formatting.
   Factual/title/procedure/summary-content failures and malformed schemas cannot
   reach or be overridden by it. All format verdicts must pass before publication.
2. Root reviews the change and adds independent adversarial controls for factual
   vetoes, exact candidate preservation, schema completeness, cost/deadline bounds
   and no-op retained summaries. Confirmed bad formats remain fail-closed.
3. Root audits benchmark consumers and stage/accounting instrumentation. Any
   integration defect is fixed by a separate agent, then independently checked
   before the next issue. Digest identity must change; unrelated memory-tier
   and Phase-1 identities must not.
4. Run targeted regression tests and proportionate wider offline gates. Preserve
   the closed paid campaign, its frozen source/helpers and historical receipts.

## Boundaries

The first-pass supported path costs no additional calls. Format-only rejection
can add at most one completion, using the same sampling/transport parameters and
remaining invocation deadline. This is not an automatic campaign rerun, model
switch, source recut or permission to spend old unused budget.

The adjudicator sees only exact effective summary and episode narrative text;
it sees no previous verdict, source catalog, titles, procedures or prior-summary
field. It cannot rewrite text or approve facts. No punctuation-count heuristic
is introduced. Independent model judgment can still err: offline tests establish
the recovery path and safety invariants, not the live false-positive rate.

This request authorizes local fixes and verification. No production deployment,
service restart or fresh paid campaign will be performed in this implementation
turn. The frozen three-call-per-pipeline diagnostic is not compatible with a
four-call maximum and must be explicitly rebuilt/reviewed before future use.

## Accepted core repair

The implementing agent changed digest policy to
`digest-fidelity-source-catalog-v6`, with
`digest-candidate-format-adjudication-v1`. Only a fully parsed format-only failure
can reach the candidate-only task. The first-pass semantic ordering and complete
shape validation remain decisive; strict two-array adjudication validates every
summary/body index and requires all supported verdicts. Failure remains in the
public `fidelity_verification` retry category, so it never causes source shrinking.

Root independently reviewed the production diff and accepted 216 focused tests
(including 25 root-authored tests) plus 502 wider integration tests. Root's public
dream tests cover both successful unchanged publication and complete rejection;
deadline tests use the same custom-client deadline proxy as the public runner.
A separate differential check compared the refactored first-pass parser with its
pre-change implementation on 3,620 valid and malformed inputs: identical results.
These are offline behavior checks, not a measurement of real-model grammar accuracy.

## Probe integration

After accepting the core fix, root reproduced a second issue: the episode probe
recorded the new task as `unknown`, advertised a three-call maximum, and could
attribute a late format failure to the earlier verifier response. A new agent
fixed that instrument separately. Root's ten new controls cover successful,
rejected, malformed, transport-failed and input-capped adjudication, each with and
without compaction. In particular, an input cap before dispatch has no response
to cite. All failure-rate denominators remain session/digest based, not call based.

The normal path remains two completions. Compaction plus adjudication can make
four, or twelve HTTP attempts under the shipped three-attempt client. A future
four-pipeline/eight-control diagnostic therefore needs ceilings of 24 completions
and 72 HTTP attempts if it reserves every task's maximum; the old 20/60 authority
does not cover that replacement campaign.

The probe now emits `episode-probe-multicall-v2` records, recognizes the exact
adjudication prompt, preserves primary request/reply fields, and reports the
four-call maximum in both help and cost output. Format input caps and transport
failures have no fabricated response; unknown failures cannot borrow an unknown
request's reply. Historical v1 records are not reinterpreted during rescoring.
Root's pre-fix reproduction failed with `unknown` stage; its preliminary rerun
passed all ten new controls. The agent's final probe gate passed 89 tests.

## Final independent gate

Root completed a fresh combined gate after both agents froze their files.
The gate selects all digest/probe test modules plus lossless publication,
semantic generations, LME failure envelopes, baseline durability, durable work,
indexing deadlines and the BEAM episode probe. It hashes 445 Python/SQL
source-and-test inputs before and after, removes credential-shaped environment
variables and blocks socket connections, DNS resolution and datagram sends.
The final result was **1,665 passed, zero failures/errors/skips**, in 492.27
seconds, exit 0. All hashed inputs remained byte-identical. The network guard
self-test passed and unexpected network attempts were zero. This supersedes
exploratory gates that loaded earlier fixture definitions.

Two preliminary runs had only test-maintenance failures: four legacy fixtures
still returned primary JSON for the new adjudication request, and one cost test
still expected a three-call maximum. Those fixtures were corrected to explicitly
model the new task while preserving their rejection/zero-spend assertions. No
test failure was hidden or reclassified as a production success.

Receipts are retained locally under
`/private/tmp/hymem-format-fix-20260915.YTQQYy/`:

- `root-final-gate.json`: final counters, unchanged-input assertion and network evidence.
- `root-final-gate.xml`: JUnit SHA-256
  `83a1522b712915f93516b87a082399b654e3bdf2cee2d05b67233bb791bb3963`.
- `final-gate-inputs.json`: input manifest SHA-256
  `afb8a855b77ab7aa00c027ccf35eadca70f3c9db64656bad3c390f5d0c4a2cf9`.
- `first-pass-differential.json`: 3,620 identical pre/post parser decisions.

Final runtime file hashes:

- `hymem/dreaming/digest.py`:
  `fdbb1a7df3be9df05b7e4df958fcfb45f9f7754a0b3d66afce2a934391b465df`.
- `benchmarks/episode_probe.py`:
  `cc568ba0c35994549a63d2db6bcae7207fb22dd248d27061dbba8d2b47fb5421`.

## Handoff

Both scoped local fixes are implemented and independently verified. No remote
files, services or production stores were changed in this repair turn; no paid
calls were made. The old campaign remains closed and failed. Offline controls
establish bounded recovery and invariant preservation, not measured live-model
accuracy or completion of the full LME benchmark. Deployment and a fresh reviewed,
budgeted live campaign are separate next steps. Digest identity changes may
replay retained source after a future deployment; no real-store migration or
replay was performed here.
