Here's the comprehensive expansion roadmap. All items in Tiers 1–3 are now
implemented; the section below documents where each landed, and the
"Future Directions" block at the bottom lists candidates for the next round
of work.

---

## Tier 1 — Immediate Wins  ✅ done

### 1a. Vector Search → sqlite-vec  ✅
- `vec_chunks` / `vec_episodes` vec0 virtual tables in
  [hymem/core/schema.sql](core/schema.sql) loaded via
  [hymem/core/db.py](core/db.py).
- Writes go to both `chunk_embeddings` (JSON, for cold-start / fallback)
  and the vec table in [hymem/dreaming/embeddings.py](dreaming/embeddings.py).
- `_vector_search` in [hymem/query/augment.py](query/augment.py) prefers
  the vec0 KNN path when the extension is present; the Python cosine loop
  is the documented fallback (`embedding_max_scan` still exists for that
  path, not the primary hot path).

### 1b. Expanded predicate vocabulary  ✅
- All 18 predicates live in `ALLOWED_PREDICATES` in
  [hymem/extraction/prompts/__init__.py](extraction/prompts/__init__.py)
  and the matching CHECK constraint in
  [hymem/core/schema.sql](core/schema.sql).
- The 8 added (`implements`, `contains`, `configured_with`,
  `requires_version`, `runs_on`, `connects_to`, `generates`, `tested_by`)
  are documented inline in the system prompt.

### 1c. Numeric & temporal fact extraction  ✅
- `kg_evidence` carries `value_text`, `value_numeric`, `value_unit`,
  `temporal_scope` columns.
- [hymem/extraction/triples.py](extraction/triples.py) parses each field
  and persists into evidence; the system prompt instructs extraction.
- `prompt_version` is now `"v7"` (covers 1b, 1c, 2b, 3a, and prompt
  refinements made since).

---

## Tier 2 — Structural Expansions  ✅ done

### 2a. Episodic Memory  ✅
- `episodes` + `episodes_fts` + `episode_embeddings` + `vec_episodes` in
  [hymem/core/schema.sql](core/schema.sql).
- [hymem/dreaming/episodes.py](dreaming/episodes.py) extracts +
  persists; the runner calls it between Phase 1 and Phase 2.
- `_episode_search` in [hymem/query/augment.py](query/augment.py) blends
  FTS and semantic KNN through RRF.

### 2b. Entity Hierarchies & Properties  ✅
- `entity_types` and `entity_properties` tables.
- The triple prompt asks for `subject_type` / `object_type` and per-entity
  property maps; [hymem/extraction/triples.py](extraction/triples.py)
  parses them and persists.
- Query-time expansion in
  [hymem/query/augment.py](query/augment.py) — both type-based (`build
  tools` → every `package_manager`) and property-based (`category=build_tool`)
  — feeds expanded entities into the graph lookup with reason chips.

### 2c. Hybrid Reranking  ✅
- [hymem/query/rerank.py](query/rerank.py) implements both backends:
  LLM rerank via the existing `LLMClient` Protocol, and a local
  cross-encoder via lazy-imported sentence-transformers
  (`mxbai-rerank-base` by default), with graceful fallback when the
  extra is not installed.
- Config knobs (`rerank_top_k`, `rerank_model`,
  `rerank_cross_encoder_model`, `rerank_ambiguity_threshold`) live in
  [hymem/config.py](config.py).
- Gated by `should_rerank` so it only fires when FTS and vec disagree
  enough to be worth the cost.

---

## Tier 3 — Advanced Capabilities  ✅ done

### 3a. Feedback-driven extraction  ✅
- `extraction_feedback` table in [hymem/core/schema.sql](core/schema.sql).
- `retract_edge` in [hymem/api.py](api.py) records a feedback row.
- [hymem/dreaming/runner.py](dreaming/runner.py) loads the 10 most-recent
  feedback rows and injects them as a negative-examples block into the
  triple prompt via `build_triple_system(negative_examples=...)`.

### 3b. Multi-hop inference  ✅
- `derived` column on `knowledge_graph`.
- [hymem/dreaming/inference.py](dreaming/inference.py) implements two
  derivation rules — `depends_on` chains (BFS) and the
  `uses + depends_on → depends_on` one-hop — folded into the existing
  predicate vocabulary so no schema migration is required.
- [hymem/query/augment.py](query/augment.py) surfaces derived edges with
  a confidence-times-product score so they don't shadow direct edges.

### 3c. Procedural Memory  ✅
- `procedures` + `procedures_fts` in [hymem/core/schema.sql](core/schema.sql).
- [hymem/dreaming/procedures.py](dreaming/procedures.py) extracts +
  persists; the runner calls it per session.
- `_procedure_search` in [hymem/query/augment.py](query/augment.py) returns
  ranked procedure hits with their step list parsed from JSON.
- Tested by [tests/test_procedures.py](../tests/test_procedures.py).

### 3d. Session Summarization  ✅
- `sessions.summary` column in [hymem/core/schema.sql](core/schema.sql).
- [hymem/dreaming/summary.py](dreaming/summary.py) generates a one-sentence
  LLM summary per session; the runner persists it.
- [hymem/honcho/app.py](honcho/app.py) `get_context` prefers
  `sessions.summary` over the `MEMORY.md` dump when the session has one.
- Tested by [tests/test_summary.py](../tests/test_summary.py).

---

## Future Directions

These are the candidate next-round improvements, framed for HyMem-as-embedded-
module (no CLI / service layer) and aligned with the
2026-05 hardening initiative (operational ease, pip-install simplicity).

### A. Procedural feedback loop  ✅
- `procedures.status` column (`active`/`stale`) in
  [hymem/core/schema.sql](core/schema.sql); migration `010` for old DBs.
- `HyMem.mark_procedure_stale(procedure_id)` in
  [hymem/api.py](api.py) flips status to `stale` and discounts `confidence`
  by `cfg.procedure_stale_confidence_factor`. Idempotent, symmetric to
  `retract_edge`.
- `_procedure_search` in [hymem/query/augment.py](query/augment.py) filters
  to `status = 'active'`, so stale procedures stop surfacing.
- Tested by [tests/test_procedures.py](../tests/test_procedures.py).

### B. Predicate-aware decay rates  ✅
- `HyMemConfig.predicate_half_life_days` is a tiered map: ~90d for
  preference predicates (`prefers`/`avoids`/`rejects`), ~60d for structural /
  dependency predicates (`depends_on`/`requires_version`/`part_of`/
  `implements`) so real dependencies don't decay before reinforcement, and a
  fallback to `decay_window_days` (30d) for volatile runtime predicates
  (`uses`/`runs_on`/`deploys_to`/...).
- `phase3.decay` in [hymem/dreaming/phase3.py](dreaming/phase3.py) groups
  active edges by predicate and applies the per-predicate window to the
  negative-bump eligibility check (the recency probe stays global).
- Tested by [tests/test_dreaming.py](../tests/test_dreaming.py).

### C. Schema migration runner  ✅
- Forward-only migrations now live as `NNN_*.sql` files under
  [hymem/core/migrations/](core/migrations/); the runner in
  [hymem/core/db.py](core/db.py) discovers them, applies any whose version
  exceeds the DB's `schema_version` (statement-by-statement, tolerating
  idempotency errors), and bumps the version. `schema.sql` stays the
  fresh-DB baseline; `pyproject.toml` ships the `.sql` files in the wheel.
- **Gotcha (learned the hard way):** `initialize()` runs `schema.sql` via
  `executescript()` *before* migrations. On an existing DB, `CREATE TABLE IF
  NOT EXISTS` is a no-op, so any **standalone** statement in `schema.sql` that
  references a migration-added column (a `CREATE INDEX`, a separate `ALTER`,
  …) crashes with "no such column". Such index/constraint statements must
  live in the migration file ONLY; the column may still sit in the
  `CREATE TABLE` (no-op on old DBs). This bit `idx_procedures_status` and is
  now documented in `schema.sql`.
- Tested by [tests/test_migrations.py](../tests/test_migrations.py) — including
  a regression test that drives the real `initialize()` path against a
  pre-v10 DB (not just `_run_migrations` in isolation, the gap that let the
  index bug ship).

### D. Speaker-weighted evidence  ✅
- `kg_evidence.source_role` column in
  [hymem/core/schema.sql](core/schema.sql); migration `011`.
- `phase1.persist_chunk_results` records the role of the chunk's first
  message and weights the positive `pos_evidence` bump by
  `cfg.evidence_role_weights` (default `{"user": 2}`); `phase3.reinforce`
  applies the same weight to co-mention reinforcement. Assistant-prefixed
  chunks keep weight 1, so the change is a no-op for the common case and
  boosts unprompted user-opened chunks.
- Tested by [tests/test_speaker_weighting.py](../tests/test_speaker_weighting.py).

### E. Triple semantic dedup at extraction time  ✅
- `_find_near_duplicate_edge` in [hymem/dreaming/phase1.py](dreaming/phase1.py):
  before minting a new edge it looks for an existing near-duplicate to attach
  evidence to. Three independent gates must all pass: (1) **predicate** matches
  exactly (`uses` / `avoids` never collapse); (2) **structure** — the existing
  edge shares the candidate's subject *or* object exactly, so only the other
  endpoint varies (a sibling canonical, not a different fact); (3) **lexical +
  cosine** — the varying endpoint is a lexical sibling
  (`cfg.triple_dedup_lexical_ratio`, shared token / substring / difflib ratio)
  *and* the triple-text cosine clears `cfg.triple_dedup_cosine_threshold`
  (0.97). The lexical gate stops false merges of short, embedding-close names
  (`redis` / `redash`). Gated on `cfg.triple_dedup_enabled` + an embedding
  client.
- Tested by [tests/test_dedup.py](../tests/test_dedup.py).

### F. Temporal / first-seen queries  ✅
- `query.entities.timeline` + `HyMem.timeline(entity)` in
  [hymem/api.py](api.py) return the first-seen active edge per predicate for
  an entity (resolved through aliases), oldest first, reading
  `knowledge_graph.first_seen`. No new schema.
- Tested by [tests/test_timeline.py](../tests/test_timeline.py).

### G. Memory export / import  ✅
- [hymem/portability.py](portability.py) emits the canonical state as JSON
  Lines (a `_meta` header then `{"type", "record"}` rows for sessions,
  chunks, edges, episodes, procedures, profile entries). `HyMem.export(path)`
  / `HyMem.import_(path)` in [hymem/api.py](api.py) wrap it. Import is
  additive + idempotent (INSERT-OR-IGNORE, sessions first; autoincrement ids
  dropped so they dedupe on natural keys) and keeps FTS shadow tables in sync.
- Tested by [tests/test_portability.py](../tests/test_portability.py).

### H. Retrieval explainability for FTS / episode / procedure hits  ✅
- `FtsHit`, `EpisodeHit`, and `ProcedureHit` in
  [hymem/query/augment.py](query/augment.py) now carry `why_retrieved` reason
  chips, mirroring `GraphFact`: `fts_match("postgres deploy")`,
  `vec_topk(sim=0.82)`, `rrf(fts+vec, 0.0240)`, `episode_fts(...)`,
  `procedure_fts(...)`, and a `reranked` tag when the reranker reorders.
- Tested by [tests/test_explainability.py](../tests/test_explainability.py).
- **Not yet surfaced downstream:** the chips live on the Python hit objects
  only — they don't reach the Honcho API response. See item L below.

### I. PII / secret redaction at chunk creation  *(security)*
Chunks are stored verbatim. A small redactor in
`hymem/dreaming/chunks.py` that detects common shapes (bearer tokens,
AWS keys, postgres URIs with passwords) and replaces them with markers
before persistence would keep the on-disk store safer. Drop-in, no
schema change.

### J. Conflict auto-resolution policy  *(medium)*
`hy.conflicts()` surfaces contradictions but never resolves them. A
configurable policy ("prefer newer", "prefer higher-confidence", "LLM
arbitrator") that walks each conflict and retracts the loser would
reduce manual operator load. Same machinery as `retract_edge`, just
wrapped in a chooser.

### K. Multilingual canonicalization audit  *(targeted)*
Per project memory, Dutch / Latin-script multilingual support is in
scope. Add a small test that seeds a Dutch chunk ("we gebruiken Postgres
voor de gebruikersdienst") and asserts canonicalization treats
`postgres` the same as it would for the English equivalent. Likely
already correct; just unverified.

### L. Surface why_retrieved chips through the Honcho API  *(small)*
Item H added `why_retrieved` reason chips to `FtsHit` / `EpisodeHit` /
`ProcedureHit`, but they live on the Python hit objects only — they don't
reach the Honcho API response, so an external (non-in-process) consumer still
can't see *why* a hit surfaced. Plumb the chips into the Honcho
`get_context` / search response shape in `hymem/honcho/` (adapters + models)
so downstream consumers can quote the reason instead of guessing. Focused
change: thread the field through the response serialization plus a contract
test asserting the chips round-trip. Mirrors how `GraphFact.why_retrieved`
should also be exposed there.
