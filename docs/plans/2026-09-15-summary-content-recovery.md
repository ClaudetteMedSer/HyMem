# Bounded summary-content recovery

## Diagnosis and scope

The user approved inspection of the rejected summary, verifier verdict and
corresponding retained benchmark source window, then separately approved the
prior derived summary and 48-character boundary context. No production memory
or credentials were retrieved and no additional provider calls were made.

The prior supports the candidate's earlier photography, winery and travel-guide
topics. The boundary context completes an explicit riding wish. No evidence
wiring omission was found. The candidate does lose the source's explicit order
of two completed trip stops, which the fidelity contract requires preserving.
That is a plausible reason for `summary_content_unsupported`, but the captured
verdict has no rationale and does not establish the model's exact reasoning.

The demonstrated operational gap is that a structurally valid digest with a
rejected rolling summary has no bounded source-based content repair. A separate
read-only reviewer agreed that one recomposition followed by complete screening
can address that gap without overriding the original factual veto. It does not
establish that the original verdict was correct or that live LME will converge.

## Sequential implementation and independent verification

1. Root independently reproduced the missing recovery path with a synthetic
   source containing an explicit event sequence. The pre-fix expected-recovery
   test failed with `summary_content_unsupported`. Receipt:
   `/private/tmp/hymem-summary-content-fix.d6WZAb/root-red.xml`.
2. A separate agent implements one summary-only recomposition after a fully
   shape-valid summary-content rejection, while all episode title/content and
   procedure verdicts are supported. Root reviews and independently tests it.
3. After accepting the core change, a new agent updates benchmark instrumentation
   for the added request role, repeated verification and six-call worst-case
   budget. Root reproduces instrument defects and verifies the second fix.
4. Run a fresh combined offline regression gate with credentials removed,
   networking disabled and exact input hashes recorded. Do not reuse older
   selected/full-suite counts as proof for the changed candidate.

## Safety contract

- Original source, prior summary, context and generation parameters are preserved.
  Rejected candidate output and verifier verdicts are not evidence for rewriting.
- Only the rolling summary changes; episode/procedure fields and citations stay
  immutable. Empty, malformed, oversized or unchanged effective replacements
  remain rejected. No truncation or silent source recutting is permitted.
- The assembled candidate must pass structural validation and complete fresh
  six-family fidelity screening. No favorable prior verdict is reused as final
  authority, and no second content-repair cycle is allowed.
- Existing format-only adjudication remains possible only after the final
  semantic verdicts all pass. It cannot override factual rejection.
- Normal successful path remains two completions. Maximum becomes six:
  primary, optional length compaction, verification, content recomposition,
  verification, optional format adjudication. All use the existing invocation
  deadline; this change does not create fresh time allowance.
- Public failures remain in the fidelity-verification category to prevent
  shrinking source windows in response to semantic repair failures.
- Production is not deployed or restarted. The previous paid diagnostic remains
  closed; its 24/72 allowance cannot authorize testing the new six-call path.

Offline tests establish control flow, source preservation and rejection behavior,
not live model accuracy. A new reviewed, budgeted live check and subsequent
end-to-end smoke remain necessary before declaring canonical LME readiness.

## Preliminary verification (not the final gate)

Root's first post-change run passed all 85 selected new controls with zero
network attempts, but its full input-manifest check failed because the agent
was still updating older test fixtures during that run. That run is explicitly
not accepted as a frozen-input gate. Original receipts remain as
`root-core.xml`, `root-core.json` and `root-core.inputs.json` in the local receipt
directory above. A separate stable-input gate is required.

## Core gate accepted; recorder regressions reproduced

Root reviewed the core implementation against the frozen v13 source and all
four adapted older test fixtures. Those fixtures explicitly repeat their rejected
summary; they do not invent a favorable replacement or remove rejection checks.
The subsequent stable-input gate passed **561 tests, zero failures/errors/skips**,
with all input hashes unchanged and zero network attempts. JUnit SHA-256:
`9a87f6307dd9d334758d7d5c720ada0d1a9bacf6ba157850ad53b0639032476f`.
Input-manifest SHA-256:
`51eab92ad98f1ae4c970e70f6282ec42ea54a18b70d510ae8a28bd1b55686283`.
Receipts use the `root-core-stable` prefix in the same local directory.

After accepting that core gate, root reproduced three recorder failures:
the new request role is `unknown` with and without initial length compaction,
and an input cap before the second verification incorrectly attributes the
first verifier's reply to the later failure. The unchanged-file three-test red
gate is retained as `root-probe-red.*`; a new agent is addressing this distinct
instrumentation issue. The actual source/cursor authority remains held on the
input-cap failure; the defect concerns reporting and attribution, not publication.

## Recorder fix independently accepted

The second agent added exact summary-content-recovery task recognition and
`episode-probe-multicall-v3` records, retaining historical v1/v2 data unchanged.
Failure-response attribution now uses only the actual final invocation and
explicitly excludes pre-dispatch caps, so the second verifier cannot borrow the
first verifier's reply. CLI help and cost estimates state normal two, maximum
six logical completions; failure rates still count attempted session digests,
not the larger number of model calls.

Root reviewed that diff, adapted one old root fixture to explicitly return its
unchanged rejected summary, and added 14 independent controls. The stable
six-file probe gate passed **129 tests, zero failures/errors/skips**, including
successful six-call accounting, failed rewrites, second-verifier rejection,
pre-dispatch caps and final format failures. Input hashes remained unchanged;
network attempts were zero. JUnit SHA-256:
`7bdbabdc98485660e7421791f50e9a98a049635c933db2cb4c2b442bf6eb1264`.
Input-manifest SHA-256:
`b91de9bb95321a3497a66242e6ad0a6d882f514e2745f6c28038fbef5b379d82`.
Receipts use `root-probe-stable.*` in the same local directory.

Frozen runtime files for the final combined gate:

- `hymem/dreaming/digest.py`:
  `543d0f6008b07d39e3e4679959d92b8ac992bd5f5cf14e853d4de9918ee36ed1`.
- `benchmarks/episode_probe.py`:
  `5fe632d7ae60b31f993b80b9df07a12edaaf78abcea5f400c7d789f6a70d9f56`.

The core agent also ran 1,149 tests before a comments/docstrings-only cleanup,
then 87 focused/identity tests afterward. These overlap the root gates and must
not be added together as a count of unique tests. The final combined root gate
is the definitive local acceptance check for the assembled candidate.

## Final combined gate and handoff

Root's final assembled-candidate gate passed **1,790 tests, zero failures, errors
or skips**, exit 0. It selected 36 test modules covering digest generation,
source/fidelity contracts, content and format recovery, probe accounting,
lossless publication, semantic identities, LME failure envelopes, baseline
durability, durable-work status, indexing deadlines and the BEAM episode probe.
This is a selected regression gate, **not the full repository suite** and not
a live-model accuracy measurement.

All 449 tracked Python/SQL source-and-test inputs were byte-identical before and
after. The network guard self-test passed; unexpected network attempts were
zero. Receipts in `/private/tmp/hymem-summary-content-fix.d6WZAb/`:

- `root-final.json`: accepted counters and unchanged-input/network assertions.
- `root-final.xml`: JUnit SHA-256
  `64be35fe8961177a82e149b9413ac8732c7f3fd82c458b1fe87956774fe28afd`.
- `root-final.inputs.json`: manifest SHA-256
  `a571af0a4e0641ae5cf0363e9f1988dc1552acf289b80d98df0f611c2b6fcef4`.

The two runtime file hashes recorded above remain current. Root compared all
Python/SQL inputs against the prior frozen candidate: only the digest, probe,
six existing test modules and four new test modules differ; no previous source
file is missing. README and readiness documentation describe the new behavior
without changing historical receipt claims. `git diff --check` passed.

Both local fixes are implemented and independently verified. There was no new
paid call, production database open, deployment, restart or real-store migration.
The prior live diagnostic stays failed and closed. A new isolated target-runtime
candidate, revised/reviewed supervisor and fresh paid authorization are required
for live testing; the four-pipeline/eight-control maximum is now 32 completions /
96 HTTP attempts. A subsequent end-to-end LME smoke remains necessary. No claim
that the canonical benchmark is ready or will complete fault-free is made.
