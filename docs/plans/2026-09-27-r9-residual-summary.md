# R9: diagnose and repair the single residual summary failure

## Evidence and scope

The exact published source `a2f816136ff297a002b131894b21ab87f75ad9db`
(505-file inventory `52c8a4938e4420a936f9c507710414ea846c9232ddf6a461c66366958e8ec7cb`)
passed 7,886 tests with four skips and zero failures/errors. Its fixed eight-question
headless LME run completed and validated all eight attempts: six correct, two
incorrect, healthy item indexing throughout. One session in one question still
had a missing/degraded summary. Runtime 10,004.56 seconds; 6,263 completions/HTTP
attempts and 20,151,328 tokens. This is not a canonical full-500 score.

The user approved continuing this residual case and necessary paid testing.
Do not launch full-500, alter production, resume/relabel old campaigns, or export
raw retained benchmark/model text. Keep all original stores/receipts unchanged;
work on fresh clones and an isolated candidate. Preserve the dirty main checkout.

## Sequential plan

1. Identify the affected retained session and exact failure using read-only,
   allowlisted counts/status/hash metadata. Independently inspect normal summary,
   publication and recovery semantics. Distinguish a code defect from an honestly
   reported provider failure; do not weaken summary validation to hide it.
2. Freeze a bounded reproduction against the exact source and private store
   clone. Start offline; use paid calls only if needed for the missing evidence.
   Retain complete private request/response accounting on Afrodite and export
   only reviewed aggregate diagnostics.
3. If a defect is confirmed, assign a separate Sol implementation agent. Require
   a red regression, a narrow fix, and preservation of grounding, attribution,
   frontiers, deadlines, budgets and failed evidence. Root reviews the actual
   diff and independently verifies it before the next fix or paid gate.
4. Verify the affected real case plus appropriate positive/negative controls,
   then broader regression proportionate to the executable change. Use a new
   immutable source inventory and fresh receipts. Report separately execution
   completion, summary health, accuracy and usage. Do not declare full readiness
   from a successful single-case retry.

## Progress

- Diagnosis started; no new provider calls, deployment or production changes.
- Identified one affected session (hash
  `d1fca62fab454a5a2c6a521bfc4c584219549044a3f7b24ab391ad13063482b2`):
  eight messages, 6,996 source characters, complete item frontier, no quarantine,
  staging or recovery job. Its first summary was 503 characters; the normal
  repair also failed the cap. Only safe event/length metadata was retained for
  that historical attempt, so its rejected response cannot be replayed exactly.
- Code diagnosis: normal ingestion's repair still requests a single summary;
  explicit recovery v7 requests and validates three progressively shorter whole
  alternatives. This is a bounded repair inconsistency, not a frontier failure.
  A separate Sol is implementing one shared current repair contract for both
  paths while preserving ordinary primary grammar, one repair call, 500-character
  cap, old parser compatibility and existing explicit-recovery behavior. Root
  explicitly excluded unrelated changes to quote normalization or recovery version.
- A second Sol is building the source-exact baseline/candidate replay worker and
  a third its isolated host controls. Shared cap: 12 completions / 36 HTTP attempts
  / 900 seconds per fresh capsule, 120-second invocation deadlines. One retained
  case plus four invented controls; source and raw responses stay on Afrodite.
  Offline preflight and root review are required before any paid dispatch.
- Root's first focused candidate gate passed 600 tests, but the independent
  wider audit found 34 stale current-wire tests and a real double-normalization
  edge case: normal digest would clean nested quotes twice, unlike the baseline
  and explicit recovery. A fresh Sol is correcting that with actual-dispatch
  red tests before the candidate is sealed or sent to the provider.
- Baseline diagnostic v1 offline preflight failed before provider construction:
  first SQLite clone remained empty; the duration suggests its 120-second
  backup deadline, with the exact cause still under investigation. Zero
  completions/HTTP attempts; original source/reference hashes
  unchanged and process/connection cleanup clean. Preserve this failed capsule;
  diagnose the backup path before creating a fresh v2 capsule. This is a test
  harness failure, not evidence of a new production/LME failure.
- Root reproduced the harness failure with a fresh no-network, read-only
  reference probe. Five unpinned backup callbacks each reported SQLite status 0
  and 2,094/2,350 pages remaining; an explicit BEGIN plus schema read reduced
  remaining pages 2,094 → 1,838 → 1,582 → 1,326 → 1,070. Both probes deliberately
  stopped after five callbacks and reference hashes stayed unchanged. Restore
  the already-reviewed R6 pinned-read backup pattern; do not infer a concurrent
  writer from this evidence.
- Root authored a regression against the actual backup function: frozen v1
  fails with repeated WAL restart; v2 succeeds while preserving one snapshot.
  Parent harness gate: 26 passed. Fresh baseline-v2 preflight passed with zero
  completions/HTTP attempts, unchanged source/reference and clean process exit.
- Baseline-v2 paid replay (manifest
  `2c1112be2659cfeefc22635bdb519c8d52e6ef3e556d84b1241ec1e92de4b66a`)
  completed with 9 completions / 9 HTTP attempts / 20,646 tokens. Fresh normal
  extraction returned a valid 430-character summary; direct source-exact normal
  repair independently failed `summary_output_cap`. Explicit recovery used
  three calls and restored the target summary on its private clone. Four
  invented controls passed root semantic review (constraints, updated target,
  uncertainty and injection resistance). No rerolls, original-store writes,
  non-summary clone changes or cleanup faults. This reproduces the repair-path
  weakness with fresh output, not the exact missing historical response.
- Parent follow-up product gate: 277 passed, including actual normal dispatch
  nested-quote controls. Sol is completing the unified affected digest/summary
  gate before final candidate sealing; no candidate provider call yet.
- The unified affected gate subsequently found seven further old-format test
  fixtures (1,161 passed). Root checked the failures were strict shape
  rejections; Sol updated only those fixtures. Root's final focused gate passed
  122 tests. Candidate frozen: 508 files, 18 changed paths relative to R8,
  inventory `35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51`.
  The complete affected gate is rerunning against this frozen tree.
- Candidate-v2 zero-call preflight passed, with clones, reference and source
  unchanged and all cleanup checks green. Manifest
  `f6f8aff611137eb66b8908f244f4572ca838cffa6a2c55a43595c7e27b7e649e`.
- Headless full regression started on Afrodite in fresh `lme-r9-full-suite-v1`,
  manifest `c8f2cae80ac082b65c619402cdead920912073b4f64fcc734752316612a05842`.
  Linux collection matched 7,940 tests, exit 0. Full tests are still running;
  no final pass is claimed. No network, provider credentials or production
  mounts; resource limits 2 CPUs / 2 GiB. No R9 production deployment or push.
- Final unified affected gate: **1,168 passed**, zero failures/errors, one
  existing Starlette/httpx deprecation warning. Root inspected the JUnit receipt
  and rehashed the candidate before paid dispatch; source inventory unchanged.
- Candidate-v2 paid replay completed: **9 completions / 9 HTTP attempts /
  21,409 tokens**. Normal extraction accepted a 417-character summary. The
  source-exact repair offered complete alternatives of 282, 233 and 191
  characters; the actual current parser accepted the first. Explicit recovery
  used three calls and restored the target summary on its private clone.
  Original source/reference unchanged, non-summary clone state unchanged,
  no outstanding private recovery state, all process/client/connection cleanup
  checks passed. No rerolls. Total comparison usage: 18 completions / 18 HTTP
  attempts / 42,055 tokens; provider dollar cost not reported here.
- Root reviewed all invented alternatives and selected summaries. No unsupported
  claims were observed. Selected Cedar, injection and Lantern controls retained
  all stated review facts. The uncertainty control retained two confirmed absent
  receipts, five unresolved, hypothetical/unproved cause and none restored, but
  omitted the assistant's unexecuted refresh/comparison proposal. Record this
  as a retention shortfall against the stronger control checklist, not an
  invented/executed repair claim and not an unqualified four-control pass.
  Selective summaries may omit whole secondary propositions by policy; this
  small replay does not establish accuracy improvement or full LME readiness.
- Real-case checks establish source-exact input, contract/cap acceptance and
  recovery/publication invariants; no independent semantic review of the private
  real-case output is claimed. The full 7,940-test regression remains in progress.
