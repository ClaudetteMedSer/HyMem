# Sequential repair of the stopped full-S run

User requested continuing the documented repair plan. Work is isolated from the
dirty main checkout and from production. No paid test, restart, deployment or
benchmark resume is included in this offline pass.

## Baseline and candidate

Workspace: `/private/tmp/hymem-lme-repair-20260918.Nod9S1`.

- `baseline/`: committed af6a615 tests/resources overlaid with the exact 225
  source/config files frozen for the stopped run. Every manifest hash verified.
- `candidate/`: separate baseline copy for sequential fixes.
- `receipts/`: original source manifest and independent verification results.

Only source, hashes and sanitized verification metadata were retrieved from Afrodite. Benchmark text, question
databases, credentials and production memory remain on the server. Existing
working-tree experiments are preserved and are not the deployment candidate.

## Acceptance sequence

1. Facts object/capacity contract and safe, attributable rejection diagnostics.
   Implementation agent: `diagnose_fact_quarantine_run`. Root reviews the exact
   diff and independently tests fresh extraction, historical replay, cursor
   preservation, overflow, invalid envelopes/items, identity changes and privacy.
   Status: accepted offline. Root independently passed 122 focused controls
   plus 74 adjacent facts/authority/FTS/semantic tests, zero failures/errors/skips.
   The unchanged baseline also passed those 74 tests. Public item validator is
   AST-identical; only two application files changed. Root independently checked
   that facts identity changes while digest/profile/Phase-1 identities do not.
   Snapshot: `facts-accepted/`; receipts: `facts-root-focused.xml` and
   `facts-root-adjacent.xml`. A separate container using Hermes1's exact image
   and Python 3.11 environment also passed all 196 distinct tests, with zero
   failures/errors/skips, exit 0 and no OOM. Its network mode was `none`, source
   and runtime mounts were read-only, and no production database or credential
   files were mounted. Receipt: `facts-target.xml`. No live semantic success is
   claimed.
2. Digest bound and reason-specific recovery, without the later unaccepted
   verifier architecture. A separate implementation agent starts only after
   facts acceptance. Root independently tests acceptance boundaries, unchanged
   valid episode/procedure content, prior context and rejection/quarantine.
   Status: accepted after root's independent 261-test gate, zero
   failures/errors/skips (`digest-root-accepted.xml`). Frozen snapshot:
   `digest-accepted/`; incremental artifact: `2026-09-18-lme-02-digest.patch`.
   Review corrected stale rejection-path variable references, positional
   dataclass drift and quote-wrapped blank/short summaries on both primary and
   repair entrances. Explicit whitespace-only primary empties remain compatible.
   Accepted ordinary summaries retain the existing normalization. The target
   Python 3.11 gate also passed all 261 tests on the same frozen source, exit 0,
   zero failures/errors/skips and no OOM (`digest-final-target.xml`). The earlier
   test-package collection error was fixed by including `benchmarks/`; no
   application change was needed for that setup error.
   The existing semantic identity machinery hashes `_run_dreaming` for every
   memory tier: reason-aware digest scheduling changes also invalidate facts
   and profile identities. Expect those tiers to replay after a later deployment;
   this pass does not refactor the identity mechanism or claim digest-only
   invalidation. The new retry wrapper is readable by the candidate, including
   historical bare keys, but an unpatched old binary cannot read new wrappers:
   any later deployment/rollback needs its ordinary matching store snapshot.
3. Exact retained chunk prepartition failures. A different implementation agent
   starts after digest acceptance. Root verifies safe source/context ownership
   and existing unsafe-split controls, then replays the three exact cases on
   Afrodite without network/model calls and without mutating retained stores.
   Status: accepted after root's independent 437-test gate, zero
   failures/errors/skips (`chunks-root-final.xml`), and the retained-source
   check below. Snapshot: `chunks-accepted/`; incremental artifact:
   `2026-09-18-lme-03-chunks.patch`. Normal 4,000-character splitting remains;
   a separately versioned 8,000-character intact-unit bound applies only when
   no safe split exists. Output completeness, prompts, source split policy,
   output tokens, depth, leaf count and call limits are unchanged. Existing
   zero-call safety negatives now exceed 8,000; the inherited pre-v3 bounded
   empty-pair fixture was updated without changing the deployed v3 behavior.
4. Combined target-runtime regression and retained-case offline verification.
   Artifact review must exclude unrelated source/model/benchmark changes.
   Status: passed on immutable `combined-verification/` source. Root's combined
   gate passed 867 distinct tests. The three independent target-runtime groups
   passed 169 facts + 261 digest + 437 chunk tests: exactly the same 867 unique
   test IDs, zero failures/errors/skips, exit 0 and no OOMs. All use the same
   final code; all 225 source/config hashes match. Target tests ran without
   network, credentials or production-database mounts. These are targeted
   regressions, not a claim that the entire repository suite was run.
   Retained-store protocol verification passed: ten fact sessions, seven digest
   sessions and the exact three failed chunk source sets; ten databases all
   quick-check OK, no foreign-key violations, no writes and unchanged file hashes.
   Chunk encoded leaf sizes/calls: Q9 `[4433]`/2, Q1 `[4635]`/2,
   Q8 `[4180,602,960]`/6. Facts/digest controls use held cursors with configured
   12,000-character windows, not original sixth-attempt requests or upgrade
   scheduling. All responses are scripted; no live semantic success is implied.
   Receipt: `combined-retained.json`. The verifier's initial overly broad
   metadata-equality assertion was corrected to recognize the existing
   `hymem-claim-source-fragment-v2` wrapper while checking original metadata,
   exact source substrings/offsets and complete ordered content reconstruction;
   no application change was needed for this verifier correction.
   Real initialization also passed on SQLite-backup clones of all ten stores:
   two opens/closes per store using the final candidate, schema 61 throughout,
   quick-check OK and no foreign-key violations. Canonical source rows, full
   session/retry state and dream history were unchanged. All original files
   remained byte-identical with no sidecars; temporary clones were removed.
   Embeddings were disabled and the no-call LLM recorded zero calls; socket
   auditing and Docker network isolation recorded no network attempts.
   Receipt: `combined-clone-open.json`. This checks initialization only, not a
   dream, retrieval quality or migration from an older schema.

## Reproducible artifacts

The main application checkout remains untouched by this repair pass. It has
unrelated working-tree experiments and is not the verified candidate.

Reconstruct in a new isolated directory from commit
`af6a615fa7fd1cf14c4c0a27b9fb236ae264f122`, then apply these patches in order:

1. `docs/patches/2026-09-18-lme-00-stopped-run-baseline.patch`: reproduces the
   frozen deployed working-tree overlay (10 changed source files and one new
   source file), not a new behavioral fix.
2. `docs/patches/2026-09-18-lme-01-facts.patch`.
3. `docs/patches/2026-09-18-lme-02-digest.patch`.
4. `docs/patches/2026-09-18-lme-03-chunks.patch`.

Verify every one of the 225 source hashes in
`docs/patches/2026-09-18-lme-candidate-manifest.json` before use. Root independently
applied the full patch sequence to committed blobs in memory: all 225 frozen
baseline hashes, all 225 candidate hashes, and all 31 files touched across the
four patches reproduce byte-for-byte. Patch hashes and changed-file hashes are
in `docs/patches/2026-09-18-lme-patch-verification.json`.

The compact verification receipt is
`docs/patches/2026-09-18-lme-verification-summary.json`. It records test counts,
receipt hashes, exact target image, retained-source and clone-open results,
scope exclusions and local/remote evidence locations. Remote staging is
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-sequential-repair-20260918-9zlJ0V`.
No production installation, service restart or paid API call occurred.

## Deployment constraints (not executed in this pass)

The chunk input-policy identity change also invalidates Phase-1 extraction cache
and publication authority. Together with the shared dreaming identity change,
expect broad current-generation re-extraction/rebuild, not just three chunk
retries. Reopening the clones does not measure that later rebuild cost.

Any eventual rollout must stop all readers and writers sharing each store, take
a consistent matching rollback snapshot, deploy one identical candidate, and
restart all applicable processes together. Identity commitments are import-time;
hot file replacement is not sufficient. Mixed old/new processes are unsupported
because old consumers cannot read the new digest retry-state wrapper even though
the schema remains 61. Never resume the old benchmark checkpoint with changed
code/identities or overwrite its failure evidence.

Diagnostics are bounded structured logger events captured durably in benchmark
logs; this pass does not introduce a database migration solely for diagnostic
fields. Exact source cursors and hashes permit attribution without persisting
raw responses or copying benchmark content off Afrodite.

Offline tests prove software contracts and state transitions, not that a live
model will extract every supported fact or that full500 will complete. A later
bounded live gate needs explicit failure-family coverage and complete accounting;
the chunk-only canary is insufficient. The stopped run remains negative evidence
and must not be rewritten as successful.
