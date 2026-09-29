# Fresh format-repair diagnostic

The user approved one isolated diagnostic after the LME readiness report:
at most 24 completion calls / 72 HTTP attempts, 120 seconds per whole owned
invocation plus two seconds for cleanup, stopping at the first failure. This
does not authorize a production deployment, restart, full LME run, campaign
resume or automatic reroll. The earlier 20/60 campaign remains closed.

## Frozen revision and independent verification

Private local work: `/private/tmp/hymem-readiness-v13.Y3tuD9`.
Fresh Afrodite stage:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-format-readiness-20260915-vZww5w`.
Container path: `/home/node/.hermes/benchmarks/lme-format-readiness-20260915-vZww5w`.

The new candidate has 447 files: all 445 Python/SQL inputs of the accepted
1,665-test local gate, plus two unchanged test configuration/fixture files.
Its source manifest is
`61ead93c000cc438d286f2d189cfaacddda38f344c5369e7474f0cbd1b28b3bf`.
This is a selected regression gate, not a claim that the full repository suite
was rerun. The older 7,406-test Linux evidence is not relabeled for this revision.

A separate agent adapted the diagnostic to four pipeline calls and one call per
control. It now recognizes the exact adjudication task and independently replays
request order in the parent after verifying committed worker receipts. The new
authorization version is v3; old v2 authority is rejected. Root reviewed the
changes, added gate-binding controls and independently ran the combined helper,
entry, transport, rehearsal and supervisor tests: **228 passed**, zero failures,
errors or skips, unchanged inputs and no non-loopback network violations.

The same 228 tests passed on Afrodite in a disposable network-disabled container.
Root reconciled exact test identities, receipt hashes, exit status, isolation and
unchanged runtime/production-source checks. Helper target verdict SHA-256:
`d6ea5f5ab0571a7f2d3129f03c132f0ba93abe3a9766e6c60cf01a98a0070b2d`.
The six-helper manifest SHA-256 is
`fd7b9cb9018a939cecbc8ebe37d11226310f2bca0da7aa3d55980a2d7cc81131`.
An additional read-only agent review found no blocking launch-safety defects.

## Offline infrastructure findings

The first source-test container could not start because its launcher attempted
to create a nested file mount inside an already read-only directory. It never
ran tests or provider calls. The mount layout was corrected; original receipts
remain under `source-gate-launch-failed-mount`.

The next source gate completed with 1,278 passes, 352 failures and 35 errors.
An in-place classification found temporary-storage exhaustion in **all 387**
failing/error cases; the disposable container's 512 MiB `/tmp` was too small for
the accumulated synthetic SQLite fixtures. The first affected case was index
740. Detailed trace export was blocked by auto-review; no workaround exported
those traces. Only fixed-category counts were returned to diagnose the issue.

Those failed results and their launch evidence remain under
`source-gate-results-failed-temp-limit` and
`source-gate-launch-failed-temp-limit`. A fresh run of the unchanged source and
same 1,665 selected tests uses a dedicated disk-backed synthetic scratch
directory, retaining read-only source/runtime and network isolation. No memory
code or test expectation was changed to accommodate this environment failure.

The corrected Linux source gate passed **1,665 tests, zero failures/errors/skips**.
Root independently verified exact test identities, final hashes, container exit
and isolation, and unchanged runtime and production source. Its fresh target
verdict SHA-256 is
`6c27169d4349ddb01e20e91b7de55e6c594db9ca667da9f0b283bad63784658b`.

## Remaining gate sequence

With both target gates accepted, the actual-input, zero-provider rehearsal can
prepare the unsigned live request plan.
That rehearsal deliberately exercises compaction plus format adjudication and
must reconcile 12 owned/reaped invocations, 19 synthetic completions and zero
provider attempts. Fresh authority is then bound to both target gates, the
rehearsal, source/fixture hashes, all six helpers and this turn's explicit consent.

The eight retained live controls are direct first-pass factual/format verifier
checks. They do not measure the new adjudicator's false-accept rate. Adjudication
may be observed on the four actual incident pipelines when a format-only first
pass declines. Passing this targeted diagnostic is not proof that every LME
question will finish; a fresh end-to-end smoke remains a separate paid gate.

## Historical pre-launch checkpoint: authorization blocked

The actual-input rehearsal completed and root reconciled every receipt:
**12 invocations, 19 synthetic completions, zero provider attempts**, with all
workers reaped and no database writes. The source database and all staged input
hashes remained unchanged. Rehearsal audit SHA-256:
`13e6340f22ac98909e5469fa83167156af6154740fa55160aa1b1491af46c3c3`.
The unsigned supervised plan SHA-256 is
`a0739a173df61d17615350bc4b26c19726da2729355493051fdc0afe4dd953f6`.

Auto-review then blocked the command that would create fresh live authorization:
it requires explicit user approval naming the retained benchmark-text payload,
the exact DeepSeek destination and the 24/72 limits together. The rejected
command was not executed. No authorization was created, credentials read or paid
call dispatched; no indirect or alternative launch was attempted.

The remaining permission question is specifically whether to send the four
retained LME digest cases and eight verifier controls (benchmark text only; no
production memory or credentials in the payload) to
`https://api.deepseek.com`, using `deepseek-v4-flash`, with at most 24 completions /
72 HTTP attempts, 120 seconds per invocation and first-failure stop. Production
deployment/restart and a full LME run remain excluded. Technical preflight must
be rechecked if that approval arrives later; stale evidence is not a launch.

## Explicit approval received; live diagnostic halted

The user subsequently approved the exact retained benchmark-text payload,
`https://api.deepseek.com` destination, `deepseek-v4-flash` model, 24-completion /
72-HTTP limits, 120-second owned invocation deadlines and first-failure stop.
Root rechecked frozen inputs, helpers, accepted gate/rehearsal hashes, unchanged
runtime and production source, and absence of prior dispatch markers before
creating fresh authority. Authorization SHA-256:
`9cefc30e1b15c12971244bc49c3b80ba9ee2c0e3c0aabc66e1ceef59e98ea915`.

The one-shot diagnostic **ran and halted on the first pipeline case**:

- Two completions / two HTTP attempts, both normal responses: primary digest
  generation followed by first-pass fidelity verification.
- The owned invocation completed in 7.41 seconds; all owned workers were reaped.
- Failure: `summary_content_unsupported` at `fidelity_verification`.
- The verifier marked both episode titles, both episode bodies and their formats
  supported. It marked summary format supported but summary content unsupported;
  there were no procedure verdicts and no uncertain verdicts.
- No compaction or format-adjudication call occurred. The semantic veto correctly
  prevented format-only recovery from overriding a content rejection.
- Three remaining pipeline cases and all eight controls were **not attempted**.
- Accounted usage: 6,082 prompt tokens + 641 completion tokens = 6,723 tokens.
  These are recorded token counters, not a dollar-cost receipt.

This is a different failure category from the prior v12 format-only rejection.
Whether the generated summary is actually unsupported or the verifier made a
false-positive judgment is **not established**. The rejected extraction's empty
published fields are not evidence that the primary model generated nothing.
Resolving the cause requires reviewing the rejected candidate against its source,
without relaxing the factual veto or rerolling until a pass.

Root independently reconciled request roles, committed response and usage
receipts, deadline/reaping evidence, first-failure stop, unchanged benchmark
database and staged input hashes, and host/entry exit status. Root audit SHA-256:
`6c389e6ff17996133cd599883461b621bc8def306bac66747efe4330785cb9c0`.
Live summary SHA-256:
`1fe887780a7b599d575195f68bd127f980631ffb2aac7b51ebdd6b44cb728649`.
Receipts remain in the private Afrodite stage under `supervised-live-results/`
and `root-live-audit.json`; raw benchmark text and model responses were not
exported. A final read-only postflight confirmed the installed runtime,
production source and Hermes1 container identity unchanged; it did not open a
production store. No deployment, restart or end-to-end LME run occurred.

**This paid stage is closed and non-resumable; unused allowance is not a new
authorization. LME remains not cleared.** The next useful step is a narrowly
scoped review of the rejected summary, corresponding verifier verdict and minimum
retained benchmark source window. Copying those raw materials into this review
requires explicit permission; the provider-transfer approval is not a blanket
export authorization. No new provider calls are needed for that inspection.
