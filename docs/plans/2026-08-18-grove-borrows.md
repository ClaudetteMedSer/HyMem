# Grove Borrows Implementation Plan

> **Borrowed from:** Grove Memory, "Living Memory" (Phoenix Grove Systems, 2026, pgsgrove.com/papers/living-memory) — use-weighted retrieval, labeled exploration, hypothesis-ledger governance, null-model calibration.
> **Status:** PRE-REGISTERED 2026-08-18. Stage 0 only. No LLM spend authorized by this document.
> **Full house-context section:** `additional_planning.md` → Plan E (E1–E4).

**Goal:** Port four independently gated mechanisms from Grove Memory into HyMem, each as a bounded, pre-registered tier: (E1) a labeled wildcard slot in query-time augment, (E2) a read-only recovery-rate gauge over the bitemporal store, (E3) trajectory-based resurfacing for retracted facts, (E4) a null-model threshold for the consolidation gate.

**Evidence caveat (why we pre-register):** Grove's paper withholds all numeric parameters (§10) and reports zero quantitative results; verified behaviors are qualitative. Verdict on their evidence: UNMEASURED. Every item below is a hypothesis gated on HyMem's own measured criteria — nothing is adopted on the paper's word.

**Architecture:** E1 is query-side, read-only, additive (mirrors Idea A / Plan D). E2 is dream-time read-only SQL + a log line. E3 is a dream-time read + a field on new candidates, reusing the existing bitemporal predicate (`invalid_at`, retracted status) — no ported status labels. E4 is an offline statistical gate that only suppresses. No schema change, no prompt change (house rules: prompt changes bump version constants; judge posture frozen).

---

## RAPTOR interference check (house rule)

- **E1 CLEAR** (mirrors Idea A): query-time read of existing rows; writes nothing; `_aggregation_search` writes a different ctx field; digest anchor read is dream-time and unaffected.
- **E2 CLEAR:** read-only SQL at dream time; touches no aggregation table, no cache id, no augment path.
- **E3 CLEAR:** dream-time read of the old edge's retraction record + a field on the new candidate; aggregation tables untouched; digest anchor predicate unchanged.
- **E4 CLEAR with a sequence tie:** suppresses surfaced candidates only; never adds content, never changes digest content or cache ids, never touches augment. Probe sequenced BEHIND the RAPTOR Stage 3c flip decision (same dependency as Plan C).

## Sequencing

- E1: independent of Campaign E; probe LLM-free; runs in parallel with G-F1b without touching the dream budget.
- E2: independent; LLM-free; can land any time.
- E3: BEHIND Campaign E (both touch the dreaming phase-1 re-assert path); probe shadow-first.
- E4: BEHIND the Stage 3c flip decision; probe shadow-first.

---

## Pre-registered gates (verdict must cite numbers; band arithmetic applies)

| # | Criterion | Pass bar | How measured |
|---|-----------|----------|--------------|
| E1-C1 | Mechanism: wildcard relevant-but-dormant | ≥5% of warm-mode sampled queries have a gold row reachable only via the wildcard slot | shadow probe vs `off` baseline |
| E1-C2 | Harm | 0 displacement: normal rows byte-identical vs `off`; never reorders; absent in `off` | unit test + probe |
| E1-C3 | Cost | 0 LLM calls; ≤1 extra FTS/vec tail read; ≤1 row / ≤200 tokens | probe counters |
| E1-C4 | Non-interference | per-category answerable deltas within ±2σ on non-target categories | A/B on fixed sets |
| E2-C1 | Mechanism | gauge reports the hand-computed fraction on a synthetic supersede→re-assert fixture | unit test |
| E2-C2 | Harm | 0: read-only, store unchanged | unit test (no-write assert) |
| E2-C3 | Cost | one indexed query per dream | probe |
| E2-C4 | Non-interference | digest cache id byte-identical before/after | unit test |
| E3-C1 | Mechanism | ≥80% of re-asserted triples with prior retraction carry `retraction_history` | unit + fixture tests |
| E3-C2 | Harm | 0: no change when no prior retraction; old edges never re-activated | unit test |
| E3-C3 | Cost | one indexed lookup per re-assertion; 0 LLM calls | probe |
| E3-C4 | Non-interference | contradiction/supersession counters and digest content unchanged | unit test |
| E4-C1 | Mechanism | null model rejects ≥95% of spurious clusters on shuffled fixture; real clusters survive | synthetic fixture |
| E4-C2 | Harm | 0: suppression only, additive nowhere | test |
| E4-C3 | Cost | offline shuffle+recompute, bounded runtime budget per dream | probe |
| E4-C4 | Non-interference | digest root and cache id byte-identical; surfaced set ⊆ today's set | test |

Verdict language: **PASS** (all C's) → flip the tier's default; **FAIL-mechanism** (C1 ✗) → close, record, no score-chasing; **UNMEASURED** (underpowered) → extend sample once or keep shadow, never claim. Band arithmetic applies to all A/B deltas (sd≈20/√n, 2σ; per-category <±5pp noise); quote net vs answerable per category, report n, cost counts, parse-failure rate, sample composition in every verdict.

---

## Task breakdown (TDD, bite-sized)

### E1 — Labeled wildcard slot in augment

**Files:** `hymem/config.py` (flag), `hymem/query/augment.py` (selection + slot), `hymem/query/augment.py` tests in the suite.

**Task E1-1 — config flag.** Add `augment_wildcard_mode: str = "off"` to `HyMemConfig` (valid: off/warm/hot). Failing test: config round-trip accepts warm/hot, rejects junk. Implement, pass, commit.

**Task E1-2 — dormant-band selection.** `_wildcard_candidates(...)`: rows that pass relevance screening (FTS/vec shortlist tail), `derived=0`, low recency, and zero-prior-surface counter over recent augment calls (config counter `wildcard_prior_window`, default e.g. 50). Failing test: fixture with one never-surfaced relevant row → exactly one candidate. Implement, pass, commit.

**Task E1-3 — slot append + stamp.** After the RRF merge, when mode != off, append ≤1 wildcard row; row gets `wildcard: true`; ctx gets a `wildcards` note (count + chosen id). Failing tests: (a) normal rows byte-identical vs off; (b) wildcard absent in off; (c) never more than 1; (d) never reorders existing rows. Implement, pass, commit.

**Task E1-4 — shadow probe + verdict.** Read-only probe over fixed store snapshot + LongMemEval/BEAM query sets, `off` baseline vs `warm`. Report C1–C4 numbers. Verdict per gate table. Commit probe script + verdict note. Stop condition: FAIL-mechanism → close E1, record in REJECTED, no score-chasing.

### E2 — Recovery-rate gauge

**Files:** new `hymem/dreaming/recovery_gauge.py` (+ tests), `hymem/dreaming/runner.py` (log line).

**Task E2-1 — SQL + function.** `recovery_rate(conn) -> float`: fraction of active derived=0 triples whose triple previously carried `invalid_at` and was re-asserted with fresh evidence. Failing test: synthetic supersede→re-assert fixture → hand-computed fraction; no-write assert (store byte-compare). Implement, pass, commit.

**Task E2-2 — dream-log reporting.** Runner emits `recovery_rate` in the dream summary (NOT in digest payload). Failing test: digest cache id byte-identical with gauge present. Implement, pass, commit. No auto-tuning ever.

### E3 — Trajectory-based resurfacing

**Files:** `hymem/dreaming/phase1.py` (re-assert hook), `hymem/query/conflicts.py` (lookup helper), tests.

**Task E3-1 — retraction lookup.** `prior_retraction(conn, s, p, o) -> Optional[reason]`: indexed lookup of the most recent invalid_at edge for the triple. Failing test: fixture with retracted triple → reason returned; no-retraction → None. Implement, pass, commit.

**Task E3-2 — attach on re-assert.** In the phase-1 re-assert path: if prior_retraction found, attach `retraction_history` (the stored reason text — 0 LLM calls) to the new candidate. Failing tests: (a) ≥80% fixture coverage — all re-asserted triples with prior retraction carry the field; (b) no behavioral change without prior retraction; (c) old edge stays invalid (never re-activated). Implement, pass, commit.

**Task E3-3 — shadow probe + verdict.** Counters: re-assertions with/without history; contradiction/supersession counters unchanged. Report C1–C4. Verdict per gate table. Stop condition: FAIL-mechanism → close, record.

### E4 — Null-model threshold for the consolidation gate

**Files:** new `hymem/dreaming/null_model.py` (+ tests), `hymem/dreaming/phase2.py` (`consolidate_insights` gate), config flag `consolidation_null_model: bool = False`.

**Task E4-1 — shuffle + metric recompute.** `null_distribution(...)`: permute episode/domain membership, recompute the detection metric N times (N configurable, default 100; bounded runtime). Failing test: on a fixture with one real cluster + K spurious ones, the spurious metrics land in the null band, the real one clears it. Implement, pass, commit.

**Task E4-2 — threshold + gate.** `consolidate_insights` consults `null_model.py` when `consolidation_null_model` is on; candidates below calibrated α (default 0.05 or measured spurious rate) are not surfaced. Failing tests: (a) surfaced set ⊆ today's set; (b) digest root and cache id byte-identical; (c) gate off → behavior unchanged. Implement, pass, commit.

**Task E4-3 — probe + verdict (BEHIND Stage 3c flip decision).** Shadow run over a store snapshot; report C1–C4. Verdict per gate table. Stop condition: FAIL-mechanism → close, record.

---

## Stop conditions (global)

- Any gate verdict FAIL-mechanism → close that item, record in `additional_planning.md` REJECTED list, no score-chasing.
- UNMEASURED → extend sample once or keep shadow; never claim.
- E3's probe does not start until Campaign E's phase-1 changes land; E4's probe does not start until the Stage 3c flip decision is made.
- No LLM spend beyond existing frozen judge usage; no prompt changes; no schema migration (all four are read/additive only).

## Out of scope (do not build under this plan)

- Dual-space structural-signature distillation, schema taxonomy/annotator lifecycle, full UCB machinery, recovery-gauge auto-tuning, checkpointed worldview diffing, MDL compression gate — all REJECTED 2026-08-18 (see `additional_planning.md` Plan E).
- Any change to retrieval ranking outside the single labeled slot (E1); any change to digest content or cache ids (E2/E4); any resurrection of retracted edges in place (E3).
