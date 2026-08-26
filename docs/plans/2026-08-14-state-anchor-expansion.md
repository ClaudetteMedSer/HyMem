# State-Anchor Expansion Implementation Plan

> **Borrowed from:** MindCache `collapsed_tree` (faisalhussain-devs/MindCache) — decision-anchor retrieval.
> **Status:** PRE-REGISTERED 2026-08-14. Stage 0 only. No LLM spend authorized by this document.
> **Full house-context section:** `additional_planning.md` → Plan D.

**Goal:** Add a retrieval-time *state-anchor expansion* tier to `query/augment.py`: seed a secondary lexical/vector expansion from the currently-active graph facts (the same selection the digest anchor uses) so that supporting evidence rows which share no lexical or vector overlap with the query — but overlap with the *current state* — become reachable.

**Architecture:** Purely query-side and read-only. "Current state" = the bitemporal active edge set (`status='active' AND derived=0 AND invalid_at IS NULL`, the exact predicate of `_anchor_facts`, `dreaming/aggregate.py:829`). Seed terms are extracted from those edges (subject canonical + predicate + object canonical + typed values) and run through the existing `_fts_search` / `_vec_search` / `_rrf_merge` machinery. Additive only: anchored rows are appended under a strict budget, never reorder or suppress existing rows, never touch another tier's budget (house hard rule).

**Tech stack:** Python, sqlite3 (existing store), existing FTS5 + embedding-server vector path, config flag in `HyMemConfig` (house pattern `xxx_enabled: bool = False`).

**Why the mechanism could matter here (mechanism, not score):**

1. **Supporting-evidence reachability.** A query names the state ("what model do we train on?") but the needed evidence rows ("installed CUDA 12.1 with PyTorch 2.2") share no lexical/vector overlap with the query — only with the *answer*. Existing entity-type / token-overlap expansion (`query/augment.py`) expands from **query** entities; state anchors expand from **answer-state** entities. Different seed source.
2. **Multi-hop state reconstruction.** For "what changed" questions, the superseded edge's rows are unreachable from the query but reachable from the active successor via shared subject.

**MindCache evidence caveat (why we pre-register):** MindCache's own docs grade decision anchors as "Observed in Development" (manual inspection), not quantitative; their BEAM pass-rate deltas (+15–20pp on n=20/conversation) sit inside the 2σ band (sd≈0.11 → ±31pp). The *idea* is plausible; the *evidence* is thin. We require measured anchored-only reachability before any flip.

---

## RAPTOR interference check (house rule, mirrors Idea A)

Query-time read of active edges; writes nothing to the store. `_aggregation_search` writes a different `AugmentedContext` field and shares no state with the anchor merge. The digest anchor read (`aggregate.py:801`, dream time) is unaffected by query-time reads. **Zero interference — clear to build.**

## Sequencing

- Independent of Campaign E. LLM-free at every stage (probe + expansion are lexical/vector only; no extraction, no new dream spend). Can run in parallel with G-F1b without touching the dream budget.
- No schema change, no migration, no prompt change (house rule: prompt changes bump version constants — we touch no prompts).

---

## Pre-registered gate (Stage 0 → verdict criteria)

Sample: fixed store snapshot; query set = LongMemEval subsets (contradiction resolution, knowledge update, preference following, temporal) + BEAM convs already in `benchmarks/`. Judge posture frozen (existing judged sets, no re-scoring).

| # | Criterion | Pass bar | How measured |
|---|-----------|----------|--------------|
| C1 | Mechanism: anchored-only hit rate | **≥5%** of sampled queries have ≥1 gold-evidence row reachable ONLY via anchor expansion (missed by current pipeline) | shadow probe, entry-key comparison |
| C2 | Harm: wrong-state pull | **0** invalidated/superseded-edge rows enter expansion (by construction; test asserts it) | probe + unit test incl. value_supersession'd edges |
| C3 | Cost | 0 LLM calls; ≤1 vector call per query; added context ≤5 rows / ≤400 tokens; latency budget +100ms | probe counters |
| C4 | Non-interference | Existing rows never reordered/suppressed; per-category answerable deltas within ±2σ on non-target categories | A/B on fixed sets |

Verdict language: **PASS** (C1+C2+C3+C4) → flip `state_anchor_enabled` default; **FAIL-mechanism** (C1 ✗) → close, record, no score-chasing; **UNMEASURED** (underpowered: per-category n too small) → extend sample once or keep shadow, never claim.

Band arithmetic applies: per-category deltas < ±5pp are noise (sd≈20/√n, 2σ); quote net vs answerable per category; report n, cost counts, parse-failure rate, sample composition in every verdict.

---

## Task breakdown (TDD, bite-sized)

### Task 1: Anchor-edge selection

**Objective:** Select the "currently active state" edge set with the exact digest-anchor predicate.

**Files:**
- Create: `hymem/query/state_anchor.py`
- Test: `tests/test_state_anchor.py`

**Step 1 — failing tests:** selection returns only `status='active' AND derived=0 AND invalid_at IS NULL` edges; a value_supersession'd edge (`invalid_at` closed) is excluded; derived edges excluded; cap respected (`cap=` default 20, house style like `aggregation_digest_anchor_facts`).

**Step 2:** run → FAIL (module missing).

**Step 3 — implement:** `select_anchor_edges(conn, cap=20) -> list[sqlite3.Row]` — copy the `WHERE` clause from `aggregate.py:829` verbatim; do NOT refactor `_anchor_facts` in this task (YAGNI; cross-module churn is a separate change if the probe justifies it).

**Step 4:** run → PASS. Commit: `feat(query): state-anchor edge selection (Plan D Task 1)`

### Task 2: Seed-term generation

**Objective:** Convert anchor edges into lexical seed terms.

**Files:**
- Modify: `hymem/query/state_anchor.py`
- Test: `tests/test_state_anchor.py`

**Step 1 — failing tests:** subject canonical + predicate + object canonical appear; typed values (numbers/dates/versions, the value_supersession v3 classes) appear; empty edge → no terms; dedup of repeated terms.

**Step 2:** run → FAIL. **Step 3 — implement:** `seed_terms_from_edges(edges) -> list[str]` — read `subject_canonical`, `predicate`, `object_canonical` (+ `object_canonical` typed-value parse mirroring `value_supersession.py` discriminators). **Step 4:** PASS. Commit: `feat(query): state-anchor seed terms (Plan D Task 2)`

### Task 3: Expansion core (FTS + optional vector)

**Objective:** Run seed terms through existing search machinery, RRF-merge, cap additions.

**Files:**
- Modify: `hymem/query/state_anchor.py`
- Test: `tests/test_state_anchor.py`

**Step 1 — failing tests:** `state_anchor_expand(conn, seed_terms, top_k=5)` returns FTS hits for a fixture store where an evidence row matches seed terms but not the query; returns ≤ top_k; rows deduped by entry key; zero-cost when no seed terms.

**Step 2:** run → FAIL. **Step 3 — implement:** reuse `_fts_search` (augment.py:692) over the evidence/episode FTS; optional `_vec_search` (augment.py:1405) only when a seed-term embedding is available (≤1 vector call per query, C3); merge via `_rrf_merge` (augment.py:1486). **Step 4:** PASS. Commit: `feat(query): state-anchor expansion core (Plan D Task 3)`

### Task 4: Shadow probe (read-only, LLM-free)

**Objective:** Measure fired/not-fired, anchored-only hits, wrong-state pulls, cost, latency — production `augment()` untouched.

**Files:**
- Create: `benchmarks/state_anchor_probe.py` (mirror `benchmarks/fact_probe.py` pattern: fixed store + fixed query set, JSON-lines + summary output)
- Test: `tests/test_state_anchor.py` (probe smoke test on fixture store)

**Step 1 — failing test:** probe on fixture store reports expected anchored-only hit for the crafted gold row. **Step 2:** FAIL. **Step 3 — implement:** for each query: run current `augment()` → collect entry keys; run expansion → collect anchor entry keys; emit per-query record {fired, anchored_only_keys, wrong_state_keys, added_tokens, vec_calls, latency_ms}; aggregate summary {hit_rate, wrong_state_rate, cost, p50/p95 latency}. **Step 4:** PASS. Commit: `feat(benchmarks): state-anchor shadow probe (Plan D Task 4)`

### Task 5: Pre-registered measurement run

**Objective:** Produce the verdict report against C1–C4 on the fixed sample. No new LLM spend (judged sets are frozen; expansion is LLM-free).

**Run:** `python benchmarks/state_anchor_probe.py` on the fixed snapshot. Report must contain: per-category anchored-only hit rate, wrong-state rate (must be 0), token/latency cost, per-category net vs answerable deltas with n and 2σ band, sample composition, parse-failure rate.

**Record verdict in `additional_planning.md` (Plan D section):** PASS / FAIL-mechanism / UNMEASURED, with the numbers. Commit: `docs: state-anchor probe verdict (Plan D)`

### Task 6: Gate-passed integration (only if PASS)

**Files:**
- Modify: `hymem/config.py` (`state_anchor_enabled: bool = False`), `hymem/query/augment.py` (additive merge after RRF, dedup by entry key, budget cap ≤5 rows/≤400 tokens)

**Step 1 — failing tests:** flag off → byte-identical context (suite regression); flag on → anchor rows appended, never reordering existing keys; cap enforced. **Step 2:** FAIL. **Step 3 — implement** additive hook. **Step 4:** full suite + LME non-regression A/B (house rule: non-regression confirmation, not tuning signal). **Step 5:** flip default only after A/B clean. Commit: `feat(query): state-anchor expansion behind flag (Plan D Task 6)`

---

## Files touched (complete list)

- Create: `hymem/query/state_anchor.py`, `tests/test_state_anchor.py`, `benchmarks/state_anchor_probe.py`
- Modify: `hymem/config.py` (flag, Task 6 only), `hymem/query/augment.py` (Task 6 only), `additional_planning.md` (verdict record)
- Never: schema, migrations, extraction prompts, judge posture

## Risks / stop conditions

- **Wrong-state pull > 0** → construction bug (C2 is a hard zero). Abort Task 3→5, fix or kill.
- **Anchored-only hit rate = 0** → no mechanism; close with FAIL-mechanism. Do not tune toward a score.
- **Context bloat** → cap is enforced in code (C3); probe measures tokens per query.
- **Recency skew:** the active set skews recent; for "what used to be true" queries anchors are *additive* context only — authority resolution stays with existing conflicts/supersession machinery; probe reports the category split so a misleading pattern is visible before any flip.
- **Shadow drift:** probe never mutates the store; production path untouched until Task 6.

## Explicitly out of scope

- Topic-tree taxonomy, Leiden partitioning, input denoiser, query-time constitution prompt, cross-encoder reranker (MindCache items analyzed 2026-08-14 and rejected — see Plan D section in `additional_planning.md`).
- Structural path indexing for FTS5 (candidate follow-on; separate pre-registration if FTS recall gaps persist after this tier).
