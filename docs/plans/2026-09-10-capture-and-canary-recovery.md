# Capture ownership and extraction-canary recovery

## Outcome — September 10, 2026

Hermes1's capture ownership incident is repaired and deployed. Its new persistent
physical session keys are admitted by the live SDK across configuration reloads,
the legacy source history remains unchanged, and all retained lossless proofs
validate. The new doctor coverage check is installed on all three instances.

**The canonical extraction canary still fails.** The provider returns successful
but semantically incomplete output on the unchanged fixture. No scored LME,
BEAM or LoCoMo run, prompt workaround, model switch, retry-cap increase, or
retry-until-green selection was performed. This report is not a new benchmark
score or a claim that every model-behavior issue is fixed.

## Findings and fixes

1. **Ownership guard was already correct.** Independent reproduction required
   explicitly dropping the binding guard. Supported Python/SQL paths reject
   non-pristine legacy binding and mixed attribution. NULL ownership is part
   of the old proofs; no single binding can validate the original native history
   and the two later attributed uploads. Root independently passed the 98
   provenance/lossless tests. No production guard was weakened.
2. **Durable retirement was missing.** The gateway's stable logical key kept
   returning to the incompatible legacy session; its existing timestamp reset
   lived only in memory. A separate agent implemented a strict host-scoped
   `sessionKeyReplacements` map, preserving logical conversation/peer resolution.
   Unknown, ambiguous and malformed mappings fail closed. Two exact keys were
   mapped to fresh IDs in Hermes1's protected Honcho config. The pinned three-
   file integration patch is reapplied/verified by its pre-start hook, before
   gateway imports; unknown image versions require a reviewed rebase, not a
   blind overwrite. The parent launcher can continue after a failed hook, so
   image-update verification remains necessary.
3. **Rollback left two invalid proofs and two stale memberships.** Records
   2746/2747 were an uncited, unconsumed tail; 2746 contained unique source bytes.
   The clone-only recovery tool preserved exact original DB/typed rows before
   quarantining the two raw/proof/chunk sets, their two message vectors and stale
   memberships. Only the producer coverage frontier was rewound, to 2740;
   consumer frontiers, legacy evidence, schema and guard SQL stayed unchanged.
   No source attribution or proof bytes were rewritten and no synthetic replay
   was represented as original history.
4. **Doctor could falsely report green.** Schema open and cited-material checks
   missed the two uncited invalid proofs. The new bounded, read-only scan checks
   every retained proof through the maintained canonical validator, covered-raw
   gaps, invalid frontiers and recorded failures. Incomplete scans fail closed;
   it never repairs, clears failure ledgers, initializes the store or calls a
   provider. Root independently verified 179 related cases, including its 39
   new cases, and reproduced exactly two invalid proofs on the live pre-repair DB.
5. **The mutation experiment had concrete instrumentation bugs.** Its C5
   `ord(hex_character) % 16` generator produced digits only, its table summary
   used the prose claim index, and its delay setting was unused. The new offline
   helper preserves protected claims, exact request fields/serialization and
   character/byte length, produces actual hex characters, distinguishes expected
   claims from other emissions and plans explicit randomized/delayed schedules.
   It makes no model calls. The old paid runner entry point is retired; its exact
   original source and all raw receipts remain private and unchanged.

## Verification and deployment

Each confirmed fix used a separate implementation agent, followed by root review
and independent tests before proceeding. Verification includes:

- Baseline suite: **5,683 passed, zero failures/errors/skips**, 2,881.761 seconds.
  It started before the new files were added; it is not misrepresented as a
  full final-tree run. Newly added behavior was independently tested below.
- Final combined focused gate: **208 passed**, 34.15 seconds: coverage 39,
  installer 55, recovery 28, offline controls 68, metadata-safe notices 18.
- Actual installed-Hermes regression runner: **293 tests, 15 files**, retries
  disabled with `--file-retries 0`; private environment, SDK 2.2.0.
- Real SDK → private live-store clone: two captures and two valid new proofs
  across manager/config reloads; all legacy rows unchanged; no model calls.
- Private real-store recovery: exact expected logical-table, vector and actual
  FTS-posting delta, full proof/FK/integrity audit, ordinary clean reopen and
  repeat no-op. Injected adoption failure restored the original DB/config.
- Final offline adoption refused any source change since rehearsal. Hermes1
  was drained with zero active gateway/UI/cron work and no dream lock, then
  stopped. Adoption ran in its existing image with networking disabled; the
  original DB/WAL/SHM and config/hook were retained. Restart: **20:32:08 UTC**.
- Live SDK: four successful session opens over two config reloads, two fresh
  owned empty sessions, legacy-history hash unchanged. No fake messages were
  injected into production conversations; write/proof tests used the clone.
- Actual configured MCP handshake, ping and profile passed on all three.

| Final check | Hermes1 | Hermes2 | Hermes3 |
| --- | --- | --- | --- |
| Retained coverage proofs | 583/583 valid | 125/125 valid | 118/118 valid |
| SQLite integrity / FK failures | OK / 0 | OK / 0 | OK / 0 |
| Canonical drift / mixed-attribution residue | none / none | none / none | none / none |
| Remote embedding backend / fallback | verified / none | verified / none | verified / none |
| Doctor failures | 0 | 0 | 1 existing historical finding |
| Service change | scoped restart | no restart | no restart |

Hermes1 retains one historical warning. Hermes3's 11 unprovable graph mirrors
and 85 explicitly retired fact mirrors match the prior September 9 report;
there are no source-eligible or rebuild-required incompatible mirrors. The
historical `action_required` finding remains visible, not relabeled or deleted.

## Remaining extraction limitation

The one unchanged configured canary made 8 successful API calls in 8.79 seconds,
four initial leaves, 35,382 tokens. It returned one valid triple, one of two
supported claims; prose claim index 1 was missing. Failure:
`supported_claim_evidence_missing`; no truncation/budget exhaustion and clean
client closure. Model `deepseek-v4-flash`, thinking `auto`.

Report SHA256:
`0ef3b3ed08841bd310cba679bae2f609ed2a42e7e4a982bb0a010fedb0bc0f24`.

The C1 contrast remains descriptive evidence of request sensitivity, not proof
of a tokenizer/serving mechanism. Character length is not token length. The
faulty C5 arm and fixed-order/unused-delay design do not justify the claimed
same-class or spaced-retry conclusions. A different model/provider or extraction
contract needs a separately versioned controlled comparison; the current gate
must remain enforced while it fails. No full production dream was triggered;
Honcho health alone is not evidence of a newly completed dream.

## Receipts and retained recovery material

Private Afrodite paths (do not print their contents):

- `/home/node/.hermes/backups/hymem-capture-recovery-20260910-root/` — authoritative
  original DB, repaired clone, typed quarantine rows, ready receipt and frozen
  pre-adoption DB/WAL/SHM. The unique upload is preserved, not runtime-retrievable.
- `/home/node/.hermes/scripts/hymem-session-retirement-20260910/` — pinned installer,
  original/new config and hook, exact disabled old probe, activation receipt and
  operating corrections. Preserve these through restarts/image upgrades.
- `/tmp/hymem-capture-rehearsal.Rm05l2ex/` — private Hermes test checkout/environment,
  293-test receipt, SDK capture clone and initial recovery rehearsal.

Core local artifacts: `tools/deployment/hermes-session-key-retirement.md`,
`tools/deployment/incident-residue-recovery.md`, and
`docs/extraction-texture-diagnostics.md`. Operational guidance was updated to
prefer the verified clone-first recovery and corrected experiment interpretation.
Publication verification caught a notice placed ahead of YAML metadata; the two
original skill headers were immediately restored, then a separately implemented
and tested byte-splice helper inserted notices after the frontmatter. Hermes's
actual parser verified unchanged metadata and original body bytes for both.

Local starting HEAD `775ce35`; Hermes1 retains its existing `074c981` checkout
plus reviewed working-tree overlays and unrelated customizations. This was not
a whole-tree overwrite, Git release, commit or push. September 10 recovery
backups/quarantine were retained, not included in earlier cleanup authorization.
