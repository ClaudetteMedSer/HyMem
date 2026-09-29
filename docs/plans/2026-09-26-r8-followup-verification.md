# R8 follow-up: repair the remaining verification gaps

## Starting evidence

The published snapshot is `5f936f21aaf59294507954ecd7ae28adecedda02` on
`codex/r8-current-version-20260926`. Its immutable 503-file inventory is
`e38a26fec4e5b756d3bca7a36418f78648b7542b3b53805012a12c3d0e8365c7`.
The completed full suite reports 7,864 passed, four failed, four skipped and
zero errors. The headless controller correctly prevented paid LME dispatch.
Separate read-only diagnosis reproduced all four failures as stale scripted
replies or assertions; this must be confirmed by parent review of each fix.

Work only in the fresh `candidate-test-sync` copy. Preserve the published
snapshot, failed receipts, production and the unrelated dirty main checkout.
The user authorizes necessary paid testing, including production-derived memory;
retained benchmark inputs are sufficient for these known issues.

## Sequential implementation and independent gates

1. **Summary fixture: separate Sol implementation.** Teach the integration
   fixture the existing v7 repair-only three-alternative envelope. Keep ordinary
   generation in its strict single-summary grammar. Check the embedded original
   source input rather than incorrectly requiring the repair wrapper to equal
   the primary request. Preserve every publication, item-frontier, attribution,
   retry, budget, reopen, portability and health assertion. Parent reproduces
   the original failure, reviews the diff and runs the affected summary suites.
2. **Terminal-empty extraction fixtures: a new Sol implementation.** Supply
   the actual bounded repair call in scripted sequences; require an invalid
   extraction followed by empty repair to fail atomically with the documented
   contract diagnostic. Assert real call counts, verification role and exact
   source preservation. Add successful nonempty-repair and failed/absent
   omission-verification controls where needed. Do not loosen clean-empty
   admission, failure accounting or production validators. Parent reviews and
   tests this independently before any next implementation.
3. **Canary watcher logging: diagnose, then another Sol if confirmed.** Check
   whether exception interpolation can expose provider response/request text or
   credentials. If confirmed, replace it with bounded safe diagnostics and
   synthetic secret-bearing exception tests. Preserve useful failure status,
   process control and exit behavior. Parent independently tests both leak
   prevention and operational behavior. No real credential is needed.
4. **Combined verification.** Freeze a new inventory and actual collection
   count. Run all affected tests, then a fresh complete offline suite on
   Afrodite. Verify source identity, failures/skips, deadlines and cleanup from
   final receipts rather than progress output. Preserve the previous failed
   full suite as failed evidence.
5. **Paid and end-to-end acceptance.** Verify the final candidate against both
   retained extraction failures, all 14 affected normal/recovery sessions and
   the existing semantic controls under the established bounded diagnostics.
   Reuse evidence only if exact applicable code identity is established and the
   admission policy explicitly supports it; do not forge new-source receipts.
   Prepare one new fixed eight-question LME run and headless validation, admitted
   only after all actual gates pass. No rerolls of failed capsules, no full-500
   run, and no production deployment from an unaccepted candidate.
6. **Publication and report.** Publish verified corrections to the dedicated
   Git branch without rewriting its history or touching `Beam-optimisation`.
   Report separately the test suite, targeted live effectiveness, LME indexing,
   summary health, score, usage and remaining limitations. An eight-question
   regression is not a canonical full-500 result or proof of perfect reliability.

## Progress

- Summary correction completed by its separate Sol, with 303 affected tests
  passing. Parent independently reproduced the original failure, reviewed the
  exact test-only diff and passed 84 integration/grammar/budget/command controls.
  Every original publication, frontier, retry, portability and health assertion
  remains; two grammar controls were added. Only the intended test file differs
  from the immutable original. No application changes or paid calls yet.
- The second Sol is correcting the terminal-empty scripted responses and
  strengthening source/call/atomicity controls; implementation is sequential
  after the first parent gate.
- Logging diagnosis confirms a second emitter in `hymem/extraction/retry.py`:
  raw provider exceptions can reach logging before extraction produces its safe
  failure code. The logging fix will address the emitter as well as the watcher,
  retaining bounded retry diagnostics rather than disabling logging globally.
  Retry behavior, deadlines, exception propagation and call accounting must
  remain unchanged; exact implementation identity changes must be honored.
- Terminal-empty correction completed by its separate Sol (348 affected tests
  passed). Parent reproduced the three original failures and reviewed the full
  diff, including six new source-exact positive/negative controls for primary
  and empty-verification repair. Parent's independent 321-test gate passed.
  Incomplete/parse failures retain their assertions; empty repair still rejects
  atomically, and nonempty repair still requires omission certification.
- Parent independently reproduced both logging exposures with invented marker
  exceptions only, zero paid calls. A third Sol is fixing the watcher and retry
  emitters with synthetic privacy/operational controls; other retry behavior
  and logging remain active. No application fix beyond those emitters is planned.
- Logging fix completed by its separate Sol: retry log records contain only a
  static event and numeric attempts/delay; watcher ERROR uses a static category,
  FAIL output admits only closed reasons and bounded true integer counters.
  The configured log path is no longer echoed. Parent requested isolation of
  `sys.path` in runpy tests to prevent Linux import-order contamination.
  Sol's corrected-interpreter gate passed 152 tests (an initial system-Python
  collection lacked `requests`; no dependencies were installed). Parent's
  independent affected gate passed 77 tests, plus two separately authored
  coupled watcher/real-retry controls. Same final exception, deadlines,
  backoff, cadence, provider arguments and exits remain tested; unrelated
  logging remains active. All probes used invented text, zero paid calls.
- Frozen follow-up inventory: 505 files, SHA256
  `52c8a4938e4420a936f9c507710414ea846c9232ddf6a461c66366958e8ec7cb`;
  actual collection 7,890. Exactly four existing files changed and two privacy
  test files were added. Original candidate bytes remain intact. Prompt v20,
  split v11 and summary recovery v7 remain unchanged; retry's executable change
  legitimately changes the local Python 3.13.5 extraction contract identity to
  `hymem-extraction-contract-sha256-v1:f7e4c79b48d2ca7c41579552d86733674bbed3ce446e9ffd592c5d85ea080e40`.
  Fresh paid and full-suite packages are being prepared against this exact
  inventory; previous receipts are not relabeled or reused as new-source passes.
- Afrodite's pinned runtime is Python 3.11.2 / Unicode 14.0.0 (local Unicode
  15.1.0); the contract deliberately binds interpreter/bytecode/Unicode identity.
  Its corresponding contract is
  `hymem-extraction-contract-sha256-v1:e880afacc41f22f98e3dc474025413e35ea1704ee4b167e587f97bc07af8ffc7`.
  Same source bytes therefore need not have cross-runtime cache identity. Both
  local import orders agreed; no identity override or cache transplant is used.
- Fresh final-chunk-v4 paid test passed both retained chunks in five completions
  / five HTTP attempts / 19,221 tokens. Parent network-none replay matched every
  exact request, private extraction result and counter with zero provider calls;
  source, reference and captured files stayed unchanged. Audit container removed.
  Capture inventory: `3abd9e76a3f35bf98072171844a98904c46add45c25c181a3b771b238261969a`.
- Fresh summary-v4 passed all 14 normal and all 14 recovery sessions, four normal
  controls, four repair-control walks and the exact retained repair case. Usage:
  72 completions / 72 HTTP attempts / 200,778 tokens. Parent independently read
  every invented normal output and repair alternative/selection against the
  controls: no invented completed actions, uncertainty preserved, updated owner
  and target attribution correct, and injected instructions not followed. Shorter
  selective alternatives omit details but do not contradict their source; selected
  outputs retain the required decisions. Original review-pending receipt remains
  unchanged; parent approval is separate. Cleanup and read-only checks passed.
- Full-suite-v6 is running headlessly against the frozen 505-file inventory and
  actual 7,890 collection. Container:
  `ce93099d6603e24ce167407871922a7ec15ee00a35aeb2235a9ab032656079a0`.
  This is not yet a full-suite pass or LME-readiness declaration.
- Controller review found a separate operational gap: a known exited/reaped
  live OOM skipped offline validation. A separate Sol is implementing the narrow
  fix with negative controls; ambiguous/running states must still stop and OOM
  must never count as live success. Frozen application source is unaffected.
- Headless OOM correction completed by a separate Sol with two reproduced red
  cases and 67 green tests. Parent reviewed the exact delta and independently
  passed those 67 tests plus 39 ownership/admission controls. Only known live
  OOM may proceed to validation; all other phase and ambiguity guards remain.
  Updated helper SHA256:
  `84872219a31ea1980d62ee0d0e4827a2d9f021eeb83349149a18e9ce1d97e92f`.
- Genuine headless-v3 preflight passed: 505 source files checked, exact runtime
  producer and checkpoint identities, actual stock CLI boundary reached, one
  client opened/closed, zero provider calls/HTTP attempts, no credentials loaded,
  cleanup successful. Its exited container is PID0/noOOM/exit0. The v2 config
  binds the actual suite/preflight containers and both fresh paid receipts plus
  root extraction audit; no placeholder gate or assumed full-suite pass is used.
- One-shot continuation-v2 detached on Afrodite at 2026-09-26 22:02:25 UTC:
  PID/session/process-group 1119508, start ticks 191927933. Parent independently
  verified process identity, exclusive intent, empty stdout/stderr, and no early
  paid dispatch. It waits for suite-v6, admits one eight-question development
  regression only on real success, then validates offline. No automatic retries
  or production deployment. Closing the laptop does not terminate this process.
  Config SHA256:
  `e0ca63490e2397bf412fd89d139a62683f91b7f85622f6b37cef54e458490d35`.
- The corrected 505-file source and truthful pending-suite release notes were
  committed in the isolated publication worktree as `a2f8161` on the existing
  `codex/r8-current-version-20260926` branch. Main checkout/index, historical
  receipts and `Beam-optimisation` are preserved. Remote push independently
  verified at `a2f816136ff297a002b131894b21ab87f75ad9db`; `Beam-optimisation`
  remains `5fb5ce491b254015684be2f3115a9d6b70b0c5a3`. Publication worktree is
  clean; no accepted-release claim is made.
- Final follow-up outcome (verified 2026-09-27): suite-v6 completed with 7,886
  passed, four skipped, zero failures/errors, unchanged source and clean cleanup.
  The one-shot eight-question run and offline validator both exited0/PID0/noOOM;
  eight attempted/completed, zero failed/missing, six correct and two incorrect.
  Runtime 10,004.56 seconds; 6,263 paid completions/HTTP attempts, 20,151,328 tokens.
  All eight item indexes were healthy, but one session in one question had a
  missing/degraded summary (zero malformed summaries). This is a real residual
  issue, not hidden by the validator's separate item-indexing success criterion.
  Source/dataset/physical-checkpoint binding and cleanup passed. No full-500,
  representative accuracy or production-readiness claim follows from this run.
  Residual diagnosis/fix is tracked in `2026-09-27-r9-residual-summary.md`.
