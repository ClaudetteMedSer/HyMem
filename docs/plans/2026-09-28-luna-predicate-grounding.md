# Luna predicate-grounding repair and verification

## Authority and diagnosis

The user approved extending the throughput-only repair to a separately
versioned extraction-grounding candidate, while retaining the strict canary.
The prior capacity pilot and every earlier attempt remain stopped and immutable.
Production is out of scope. GPT-6 Luna subscription authentication, quota floor,
isolation, scoring, retry limits and the corrected TasksMax 256 stay unchanged.

The capacity pilot's byte-faithful offline replay returned both intended facts
plus an unsupported `uses` relation derived from a `prefers` assertion. The
canary was correct to reject it. This is a precision defect, not a reason to
filter model output against benchmark answers or weaken its oracle.

## Sequential plan

1. A Sol implementation adds a concise, generic predicate-grounding rule to
   the shared chunk extraction prompt. Each additional predicate needs its own
   source support; preserve independently asserted relations, polarity and
   provenance. No fixture names, canned answers, post-processing removal of
   surplus claims or parser/validator changes. Version the rule and verify its
   rendered bytes change the extraction contract identity.
2. A separate Sol agent supplies independent invented positive/negative controls
   for preference, usage, both, explicit negative use, different preferred/used
   objects, intent and cross-paragraph/table support. Root reviews the prompt
   and controls independently, tests actual frozen modules, and keeps the
   historical surplus-claim replay failing under the unchanged oracle.
3. After accepting the prompt change, a separate Sol implementation supplies a
   bounded paired live diagnostic. Predeclare ten controls, baseline/candidate
   arms, two repetitions each, and two complete canaries per arm. Initial cap:
   192 admitted turns, 2,000,000 known tokens, 1,800 seconds; every invocation
   retains the 120-second deadline and normal subscription quota checks.
   Continue independent semantic cases after model-output rejection; stop for
   transport, accounting, resource, isolation, quota or cleanup failure. Keep
   full raw evidence private on Afrodite. No result selection or rerolls.
4. Root independently verifies diagnostic code, source pins, negative tests,
   exact request transformation and cleanup before live dispatch. Success means
   the candidate preserves all positive controls, excludes unsupported relations
   and passes both strict canaries, with complete accounting/cleanup. Baseline
   results remain visible, even when also passing. A finite passing sample is
   not proof of zero future model errors or statistically demonstrated gain.
5. Only after acceptance, freeze a fresh candidate consisting of the previous
   508-file benchmark runtime plus the reviewed prompt change, with a new
   inventory and contract identity. A new source-bound runner/receipt must
   faithfully identify that candidate without weakening inventory checks.
   Run the same four questions concurrently under the existing full-pilot caps.
   Root verifies startup and binds the existing heartbeat to the new run.
   Require `completed_and_clean=true` for end-to-end success; answer correctness
   is a separate measurement, never a reroll condition.

If a test fails, preserve it, identify the precise failure and repeat the
separate Sol implementation/root verification cycle only for a justified fix.
Do not expand to full-500 or production, change model, bypass quota, raise spend
limits or substitute a weaker quality policy.

## Status

Sol's prompt implementation passed root review and 107 root-run tests covering
the new rule, extraction prompt contracts and the unchanged canonical canary.
Only the combined chunk prompt changed; all three extraction/verification
builders share the rule. The production-facing working-tree edit is uncommitted
and undeployed. A new frozen candidate must transplant only its three insertions,
not replace the whole prompt module with the older workspace file.

Root rejected an ambiguous stopped-use control before any model calls: its
explicit past positive usage could legitimately authorize a historical fact.
Sol replaced it with the prompt's unambiguous `no longer uses` form. The fixed
ten-control corpus SHA256 is
`56147b137c71236aa16afc8b0f2f413b90b255c79cbcb370fed50ef71879103c`.
Root ran 41 additional control/resource/transport tests and independently drove
all ten controls through the real frozen extractor using synthetic responses:
all source contracts valid, two calls per case, zero model calls. The control
grader measures core claim precision/recall; markers and optional hints are
counted but not semantically graded. The stricter canary still checks them.

A separate Sol agent is implementing the paired diagnostic. Root reviewed and
ran the mechanical candidate builder in isolated Python, creating
`/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/candidate`. All 508 files are
verified; only the approved ten prompt lines were added. Frozen summary and
numeric-table handling are preserved byte-for-byte. Root's independent positive
and surplus-claim canary controls pass on both old and new runtimes.

- Builder SHA256: `51e03ff2bec29f7517db027b103163698c18b9d34603ab950a36f5d09255e96a`
- Derived prompt SHA256: `17ee5017c54a1ba255e0220aa8e246766127cb71ced65fb2e8c812034f6e184c`
- Derived 508-file map SHA256: `217036b8089911c352cdf5994ef2b37915b5c68d2622a23c0642d264487dbe62`
- Derived inventory SHA256: `896b89d56f393bf386cef7b400bbb1450306555ccb6d05f4b6c787e616bf104f`
- Baseline extraction contract suffix: `f7e4c79b48d2ca7c41579552d86733674bbed3ce446e9ffd592c5d85ea080e40`
- Candidate extraction contract suffix: `adf248f4c33516573859422942386faacdd97bd65c7b0bd83a0854e323d90795`

Both contract identities carry the prefix `hymem-extraction-contract-sha256-v1:`.
The new prompt therefore cannot silently reuse old extraction cache entries.
Root also ran 426 frozen-candidate extraction/canary regression tests: all
passed. The paired probe, controls and derived full-pilot integration passed
49 independent root-run offline tests, including actual isolated candidate
loading, stdin reader bootstrap, unchanged terminal health gates, immediate
mapping-fault stop, call-accounting mismatch, leaked-process detection and
retention of paid-work accounting if final artifact writing fails.

The reviewed paired-probe SHA256 is
`a4e50e05d2a5eac8199fb6cf6dee950f7119f4f53d7859a1dbac155846932214`.
Its initial private staging root is
`/home/atta/.hymem-luna-grounding-probe-5THc8eip`.
The orchestration helper passed independent Sol review and root-rerun offline
tests (35 host/probe/control tests). It rejects symlink destinations and
non-private working directories; a failed or ambiguous dispatch consumes the
one-shot marker before execution and preserves diagnostic output privately.
The source-bound receipt is sealed; no model calls at preparation time:

- Unit: `hymem-luna-grounding-probe-5THc8eip.service`
- Receipt SHA256: `87ba2faaedc6e843d48323bd5d865defb1110d863522dd7a6bc51957f7f9fe79`
- Host helper SHA256: `3f7e31ca2ed5dee30ee93d3178d939bc2bdf3a29888e989f772de2cd36ab0676`
- Systemd: 1,830s + 10s stop, TasksMax 256, memory 4 GiB, CPU 200%,
  OOMPolicy kill, control-group cleanup, no restart, private empty working dir.
- Both arms execute the same frozen extractor and unchanged strict canary.
  Only the candidate's three system strings change, proven identical to the
  actual derived prompt builders. Other request fields are byte-identical.

Raw App Server
stderr is discarded by the immutable accepted transport; full request/response
evidence and runner stderr are retained privately, not exported.
The existing heartbeat remains paused.

Dispatched once around 19:54 UTC on September 28; systemd dispatch returned 0.
Startup verified exact policy/source pins, no restarts, 215.848 GiB free disk.
First completed snapshot: 2/44 units, one pass per arm, 4 settled turns and
28,478 known tokens; this excludes current in-flight work and is not a final
accounting or semantic verdict. No LME questions or production changes.

Official OpenAI prompting guidance informed the paired-evaluation requirement;
the application-specific grounding change is based on the retained failure,
not a claim that prompting can guarantee perfect model behavior.

## Paired diagnostic: accepted result

Both arms passed all 22 units: 20 controls plus two unchanged strict canaries.
All 44 scheduled units completed, using 112 admitted/returned calls and 846,174
known tokens, with complete/reconciled accounting and no outstanding reservations.
The reviewed runner reported `completed_and_clean=true` and
`all_candidate_passed=true`. Root separately checked exact source/resource pins,
all unit cleanup flags, sums of per-unit calls/tokens, exit status 0, MainPID 0,
empty control-group header and zero remaining cgroup processes. No restarts.
An intermediate kernel observation showed peak 42 tasks, zero task-limit
denials and zero OOM kills; removed terminal counters are unknown, not zero.

Root then independently replayed all 112 captured requests/responses offline
with networking disabled: 56 requests and 22 passing units per arm, byte-exact
wire-request matches. The candidate arm used the **physical derived candidate**
prepared for LME, not the diagnostic's request-rewriting client. No new model
calls. Replay source SHA256:
`da41012e38b34f6f9110097f14b7324f9cd89abda100a9fdcd3132029926a883`.

This accepts the candidate for the bounded four-question pilot. Since the
baseline also passed, it does not establish a statistically measured reliability
gain or guarantee future error-free extraction. The strict gate is unchanged.
See `2026-09-28-luna-lme-grounding-run.md` for the separate pilot receipt.
