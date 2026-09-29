# Expanded frozen-R5 verification

User authorization: run all necessary tests, including paid tests. Production
deployment, service restarts and production-memory changes are not part of this
verification. All paid data is benchmark-only and stays on Afrodite except for
bounded result metadata. Provider: https://api.deepseek.com, `deepseek-flash`,
thinking disabled, the same already-verified current service as the Q1 test.

## Gates and sequence

1. Reconcile completed offline receipts against the unchanged R5 candidate.
   Completed: all 467 files match, 7,243 passed plus four declared skips; the
   Afrodite gate also matches. Repeating unchanged full suites adds little value.
2. Prepare, independently review and test an isolated successor to the Q1
   harness. Preserve all previous sealed packages, scored artifacts and stores.
3. Run one fixed stock LME sample: eight questions, seed 0, workers 1, untouched
   original 500-question dataset. Source indices were committed before metadata
   inspection: 213, 262, 329, 339, 370, 372, 392, 400. No rerolls or resume.
   The subsequent metadata census shows one multi-session, three temporal and
   four knowledge-update cases, 398 sessions and 4,036 messages. It is a limited
   label-blind completion test, not category-balanced or a full accuracy result.
4. Verify genuine scored archive, all eight raw physical checkpoint entries and
   single-attempt histories, item indexing, separately disclosed summary health,
   measured provider accounting, source/dataset pins and process cleanup.
   Wrong answers are performance measurements, not infrastructure failures.
5. Assess independent summary recovery using a new copy of the prior Q1's
   benchmark-only store, preserving the scored original and item publications.
   Define and review its bounded protocol before any recovery provider call.

## Limits

The sample keeps the stock recipe: healthy item indexing required, full dreams,
lexical retrieval, no embeddings or aggregation, legacy-custom judge protocol.
Each question retains 100 dream cycles / 3,600 seconds of indexing maximum;
external process supervision is 32,400 seconds with a ten-second cleanup bound.
There is no stock global completion cap, and none is claimed. All measured
calls/tokens must be reported; unavailable dollar costs remain unavailable.
Ordinary question failures remain in the result and the stock runner may
continue to subsequent independent questions. Structural/cleanup exceptions
halt; postvalidation refuses a clean result with accounting or integrity errors.
No instant stock accounting-error abort is claimed.

Full 500-question LME performance and live BEAM/LoCoMo scores remain separate
evaluations. Existing offline BEAM/LoCoMo adapter coverage is verified, not
misrepresented as a paid end-to-end pass. Neither sample success nor unit-test
counts establish universal semantic correctness or production readiness.

Status: the eight-question paid execution is running on Afrodite. All 187 new
harness controls passed (50 runner/startup, 86 verifier, 51 host/packaging),
including genuine stock archive validation and checkpoint reconciliation.
Independent review found and fixed census-schema and nested startup-proof gaps
in the new harness before launch. The real Afrodite startup preflight passed
without credentials or provider calls and exited cleanly. No application-source
changes were needed. At the 21:44 UTC progress check, one question was completed
and the second was indexing; no quarantined extraction rows or instrumentation
errors were observed. These are progress observations, not final validation.

Package manifest: `fa471c38440dd70e8527b75171b1af8fc9b48ee98081df9738b45093f41efffc`.
Live container: `56729d5659b7f8807956ff8a7552d548afaf9471449bf6e5c33c03baa6b42a87`.
Continuation state: `docs/patches/2026-09-24-expanded-verification-state.json`.

## Independent summary recovery — ineffective, integrity preserved

The separate one-shot recovery run on a fresh SQLite-backup copy of the completed
Q1 benchmark store finished cleanly. It made 10 logical completions / 10 HTTP
attempts, all admitted, consuming 22,879 prompt and 3,285 completion tokens
(26,164 total). Dollar cost was unavailable. **Zero of ten missing summaries
recovered.** All ten were held with `summary_output_cap`, one attempt each;
none advanced or published. No reroll or resumed paid recovery was performed.

The original scored store and every non-summary table/field stayed unchanged;
database integrity, publication provenance, accounting and process/client/lease
cleanup passed the independent read-only audit. Exit zero means safe diagnostic
completion, not effective recovery. The saved metadata receipt is
`docs/patches/2026-09-24-summary-recovery-readonly-audit-v2.json`. The original
audit receipt is preserved alongside the tightened successor.

The recovery contract requests a maximum 500-code-point summary while preserving
every distinct claim from an accumulating source walk, with an 8,000-character
source window. Dense inputs need not satisfy both conditions. The recorded cap
rejection is distinct from provider token truncation or an explicit empty
refusal. This identifies a capacity limitation; it does not establish a new
item-indexing failure. The ongoing sample keeps its frozen application and
recipe unchanged. It must report summary degradation separately.

Independent review and parent inspection located the conflicting requirements
in frozen R5 `hymem/dreaming/summary_recovery.py:47–62`, the source framing at
225–246 and admission at 249–264. The rejection code can also mean a raw response
over 65,536 characters; response lengths were not retained, so these receipts
do not identify which cap branch fired or prove each particular case impossible.
The loop intentionally holds a rejected session once per invocation rather than
spending its entire call budget on retries. Existing positive recovery controls
use small or repetitive sources with canned short responses; they validate
mechanics, not dense-history compression effectiveness.

Raising just a prompt/parser limit is not a complete fix: `clean_summary()` still
returns at most 500 characters and migration 063 constrains private drafts to
500. A subsequent application change must explicitly choose and version its
semantic contract. A lossy factual overview is consistent with summaries being
non-authoritative context and detailed claims remaining in items/source records;
if complete claim retention is required, segmented/hierarchical working summaries
are needed. Neither change is being silently applied during this verification.

New recovery diagnostic controls passed: 32 worker and 53 host/backup controls.
The initial credential-free V1 preflight exposed a paged read-only WAL-backup
restart loop and stopped before any paid calls. A pinned read transaction fixed
that diagnostic setup issue; fresh V2 preflight and the one paid V2 invocation
then ran. All failed setup artifacts were preserved. Further independent audit
review found and fixed insufficient validation of saved accounting types/ranges
and open-ended metadata projection. The parent reran all 116 audit controls and
then the credential-free, network-disabled, read-only audit on the same saved
paid artifacts. It passed with the identical paid-result hash and now reports
the ten cap failures directly from the database. No paid package, application
source, database or already-measured provider output changed.

No production deployment or full-500 readiness is claimed. Successful item
indexing, benchmark completion, summary recovery and answer accuracy are
separate outcomes; none substitutes for the others.

## One-shot completion verification on Afrodite

At 22:02 UTC (00:02 local, September 25), a separate, detached one-shot helper
was launched and its admitted/waiting state independently verified. It waits
only for the already-running exact sample-eight container. After clean exit it
runs the original sealed offline validator once, with network disabled, no
credentials and all bind mounts read-only. It cannot restart/resume/stop the
paid run. Its own offline validator has a ten-minute bound and scoped timeout
cleanup; exclusive latches refuse duplicate invocations. This is a dependent
test step, not a recurring automation or notification service.

The parent reran 41 controls (37 orchestration plus four genuine checkpoint /
validator projections). Correct answers, ordinary wrong answers, explicit
summary degradation and failed rows preserve their distinct verdicts. Wrong
container identities, source drift, duplicate execution, unsafe mounts,
timeouts and malformed metadata do not produce a success receipt.

Reviewed helper SHA-256:
`9784c2cb824dd86739c1f4f103f73cbb5e8924b42acf08cab3f3e7970ef81571`.
Launch/admission proof:
`docs/patches/2026-09-25-sample8-dependent-validation-launch.json`.
Parent control receipt:
`docs/patches/2026-09-25-parent-sample8-finish.xml`.

Under the remote sample-eight root, `finish-validation/waiting-live.json`
records admission, `validator.json` will hold bounded result metadata, and
`final.json` will record completion/failure and terminal-container proof.
**Do not manually start another validator while this helper owns that step.**
Raw scored artifacts, checkpoint, logs and retained stores remain on Afrodite.
No final benchmark-verification result has been observed yet. At the latest
progress check one question was completed; question two had published all
46 session item frontiers and was still completing chunk indexing, with no
quarantines or instrumentation errors observed.

## September 25 terminal result and corrected read-only audit

The preceding progress observations are historical. Sample-eight finished once:
seven completed, one indexing failure, zero missing. Five answers were correct,
two ordinarily wrong, and the failure remains in the eight-question denominator
(62.5% for this small, nonrepresentative, non-official-comparable sample).
This is not a fault-free benchmark result. Seven completed questions had 69
missing/degraded summaries; the failed question had another 13. Measured aggregate
usage was 5,978 admitted completions and 5,982 HTTP attempts; aggregate token and
dollar totals were unavailable and must not be invented.

The original sealed postvalidator failed because its indexing-failure allowlist
omitted the writer-generated `strict_failure` field. A fresh Sol agent repaired
that contract in an isolated R6 candidate. The parent reproduced both real-writer
regressions against R5, reviewed the patch, ran 666 affected tests successfully,
and independently revalidated the same saved artifact with only the corrected
validator. Archive/checkpoint/source hashes stayed unchanged; physical checkpoint
binding and the one failed question were preserved. No paid rerun occurred.
Acceptance receipt: `docs/patches/2026-09-25-parent-r6-fix1-audit.json`.

The failed chunk independently reproduces without an LLM: a space-delimited,
unit-labelled numeric table lacks a supported row-boundary rule. A fragment
containing 388 numeric rows is 8,033 source characters / 9,286 encoded characters,
exceeding the 8,000-character intact-input envelope. Three attempts cannot fix
this deterministic preflight failure. The sequential repair plan is
`docs/plans/2026-09-25-lme-sequential-sol-fixes.md`; no production deployment or
full-500 readiness is claimed.
