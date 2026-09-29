# Generic bounded clause packing for summary recovery

The user requested implementation after the offline reference review established
that the retained case's main information fits in 500 characters. This work is
isolated from the unrelated dirty main application checkout and all immutable
prior candidates. Working root:
`/private/tmp/hymem-summary-clause-packing-20260919.QdTCry`.

## Plan and acceptance gates

1. A separate implementation agent modifies the frozen candidate's summary
   repair prompt and adds a bounded clause compiler. Root independently tests
   and reviews its implementation. No source-specific reference or benchmark
   identifier is added to runtime.
2. One repair response may use `{"clauses":[["preferred","compact"],...]}`:
   1–16 ordered clauses, each containing 1–3 nonempty meaningful string variants.
   A clause represents an independent proposition or tightly coupled causal/
   qualifier bundle. Variants must be semantically equivalent and keep explicit
   actors and scope; they must not depend on another clause's selected wording.
3. Validate the entire plan before selecting. Precompute shortest remaining
   variant lengths; choose the first preferred variant that leaves room for
   every remaining clause. This yields the lexicographically earliest feasible
   index tuple. For one to three clauses, if that selection falls below the
   existing ten-character meaningful-summary minimum, exhaust at most 27 tuples
   in preference order to find the first fully admissible selection. Four or
   more meaningful clauses and their separators already satisfy that minimum.
   Render one whole trimmed variant per clause in original order with `; `
   separators; count all separators and Unicode code points.
4. Never drop a supplied clause, clip strings, silently salvage a malformed
   plan or exceed 500. No-fit is `summary_output_cap`; malformed/empty plans are
   honest failures. Revalidate the complete digest before assigning provenance
   or advancing a cursor. Existing single-summary and whole-alternative response
   compatibility remains unchanged. Primary input, exact repair input envelope,
   output-token limit, schema and two-logical-completion ceiling remain unchanged.
5. Independent tests compare the compiler with brute-force small-matrix oracles,
   including exact 500/501 Unicode/separator cases, malformed unused variants,
   atomic failure, unchanged source/items, cancellation, semantic-generation
   identity, and actual dream publication/retry/reopen. Run broader regression
   gates before freezing an application payload.
6. A fresh, separately pinned live diagnostic will contain two saved benchmark
   primary drafts and four invented semantic controls. Each gets one new repair
   response, maximum six paid completions and six HTTP attempts total, with no
   retries/rerolls/resume. Use the existing approved DeepSeek endpoint/model,
   temperature zero, JSON mode, thinking disabled, 3072 output tokens, 120-second
   invocation deadlines and a 900-second campaign ceiling. Before any paid call,
   complete credential-free network-disabled rehearsals and independent audit.
7. Predeclare control outcomes. Three invented controls should preserve meaning
   and fit; one deliberately impossible exact-identifier case must fail honestly.
   Accepting a shorter result that omits its required identifiers is a semantic
   failure, not success. Review selected outputs against source obligations,
   independently verify accounting and replay results without network access.
   No production memory, deployment, restart or full LME is part of this test.

## Limits that must remain explicit

Packing guarantees retention of every **provider-supplied** clause, not that the
provider supplied every required topic. It cannot prove variant equivalence or
semantic completeness. Source-span IDs or self-reported coverage do not turn
that into a proof. Prior automatic summary is continuity context, rejected
drafts are untrusted, and boundary context is not newly consumed evidence.

The manually constructed historical reference is an offline oracle only, not a canned
runtime response or model-success claim. Compiler/semantic changes alter digest
generation identity; this is not a no-op deployment. No candidate will be called
LME-ready on the basis of synthetic tests or successful transport alone.

## Implementation verification so far

The implementation agent's focused gate passed 242 tests. Root's independent
80-test gate passed, including a brute-force oracle and an independently found
short-selection issue fixed before freezing. A separate reviewer compared 3,000
generated matrices against exhaustive selection and full digest admission.

The frozen candidate contains 225 source/config files and 208 Python test files.
`hymem/dreaming/digest.py` SHA-256:
`03d62f806cd23c065ad3d99f0f317d8c9e7d9d671dc5e546d84c1d112ef06bf6`.
Only that application module changes versus the previous frozen candidate.
The local 822-test regression gate passed with zero failures, errors or skips;
source/test hashes matched before and after. Three disjoint Hermes-runtime
shards covering 621 tests are in progress. These are selected regression gates,
not a claim that the entire repository test suite has run.

The four invented controls are frozen with SHA-256
`04ae308b5b640aeab81050400109d413be8beed2ae7c3a425a5d8a4bd31e9cd1`.
The final diagnostic-helper suite passed 142 tests independently in both the
implementation agent and root runs. Credential-free, network-disabled positive
and negative rehearsals exercised six cases each. Independent accounting and
saved-response application replay reproduced all six acceptances and all six
over-budget rejections, respectively. Each audit verified all 225 source hashes
and ten retained store hashes; all five inventory/rehearsal/audit containers
exited 0 with PID 0 and no OOM. Scripted success is not model or semantic success.

The live request inventory and manifest are frozen. Live-manifest SHA-256:
`667c67db4ecde65ad4b0ae81cfec6c8fa8d8d0ffd3ef9ae90bb46ef2b292f424`.
The campaign may start only after the target-runtime regression gate passes.

Semantic review must distinguish the exact-identifier negative oracle from
ordinary topic-level summarization. Its accepted omissions would fail this
preregistered exact-value stress test, not prove that every reasonable concise
topic summary is invalid. Positive outcomes are judged semantically, not by
literal reference matching. All offered variants, not only the selected
assembly, must be reviewed for equivalent meaning and independent referents.
An empty plan is honest abstention, not proof the model calculated infeasibility.

## Live result: historical blocker remains

The target-runtime gate passed all 621 exact, disjoint tests with zero failures,
errors or skips; all three containers exited 0, PID 0, no OOM. Root independently
checked the downloaded JUnit/worker receipts and container exit metadata.

The single six-request live campaign completed without retries. All six replies
finished with `stop`; accounting was 6 completions / 6 HTTP attempts, with 6,846
prompt tokens and 1,619 completion tokens (8,465 total). A network-disabled audit
replayed every saved reply through the exact candidate application and verified
unchanged 225 source hashes and ten retained store hashes. The paid container
exited 0, PID 0, no OOM. No production process or store was changed.

Both retained historical drafts still failed: their shortest full clause plans
are 662 and 779 characters. Every supplied clause was retained by the compiler;
there was simply no fitting combination. Three positive invented controls were
structurally accepted at 242, 426 and 265 characters. The exact-identifier
negative control was honestly held: its shortest complete plan is 754 characters.
The successful synthetic controls do not cancel the historical failures.

Initial source-relative review also finds semantic defects in both rejected
historical plans: the new material's significant land/naval battles and Korea/
East Asia geography are absent. The old-draft plan additionally conflates the
Port Arthur lease with a separate threat and omits its lessor. Thus increasing
the cap would not by itself make those plans adequate. Full independent semantic
review is complete: both retained plans fail, the three positive controls pass
their main-proposition checks, and the exact-identifier negative is correctly
held. All 63 variants were reviewed. Cross-clause referents remain a wording-
contract caveat even where positive assembled meaning is sound. No artifact is
marked semantically ready or LME-ready.

The model largely segmented prior/draft prose into clauses, retained redundant
generic topic labels alongside their detailed causes, and often repeated the
same alternative without meaningful compression. That is observed behavior;
the separate causal effects of rejected-draft anchoring and prompt requirements
have not been isolated by an ablation.

Status: Patch 08 is a verified bounded compiler implementation but an
**unsuccessful live recovery candidate**. Do not deploy it as a demonstrated fix
for the historical LME blocker. Campaign spent; no reroll or resume permitted.
