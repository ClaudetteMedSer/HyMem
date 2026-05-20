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

### A. Procedural feedback loop  *(small)*
Symmetric to 3a, but for procedures. When Hermes surfaces a procedure via
`_procedure_search` and the user marks it as wrong / outdated (a new
`mark_procedure_stale` API), record it and either downgrade `confidence`
or skip it in future search. Procedures rot faster than declarative facts
— a "deploy" runbook from six months ago may be actively misleading — so
a confidence signal matters more here than for triples.

### B. Predicate-aware decay rates  *(small)*
Phase 3 currently applies one decay schedule to all edges. `prefers` /
`avoids` should decay slower than `uses` / `runs_on`, because preferences
are stickier than runtime state. Add a small `predicate -> half_life_days`
map to `HyMemConfig` and have `phase3.decay` consult it.

### C. Schema migration runner  *(operational)*
`schema_meta.schema_version` is set to `'1'` but no actual migration
runner exists — the schema is forward-only via `CREATE TABLE IF NOT
EXISTS`. As columns evolve (we've quietly added `summary`, `derived`,
`value_*`, `temporal_scope` etc.), an out-of-date database will silently
miss them. A small runner in `core/db.py` that compares
`schema_version` and applies migration scripts under
`hymem/core/migrations/` would harden upgrade paths — important for the
pip-install path where users may carry old DBs forward.

### D. Speaker-weighted evidence  *(medium)*
Right now every chunk contributes equally to edge evidence. Triples
extracted from user-authored chunks should carry more weight than ones
from assistant turns (which can be confabulated). Plumb the role of the
chunk's first message into `kg_evidence` and let phase 3 weight
`pos_evidence` accordingly. Small schema change, modest extraction
change, large quality win on noisy logs.

### E. Triple semantic dedup at extraction time  *(medium)*
We already have `edge_embeddings` keyed on triple text. Before inserting
a new `(s, p, o)`, look up nearest existing edges by vector and, if one
is within a tight cosine threshold, attach the new evidence to it
instead of creating a near-duplicate. Bounds the unbounded growth of
sibling canonicals like `"uv"` / `"uv_pip"` / `"uv_package_manager"`.

### F. Temporal / first-seen queries  *(small)*
`knowledge_graph.first_seen` exists but nothing reads it. Expose
`hy.timeline(entity)` returning the first-seen edge per predicate for an
entity, so Hermes can answer "when did we start using Postgres?" without
re-asking. Tiny API surface, no new schema.

### G. Memory export / import  *(operational)*
A `hy.export(path)` that emits the canonical state as JSONL (sessions,
chunks, edges, episodes, procedures, profile entries) and an `import`
counterpart. Useful for backups, project-to-project migration, and giving
external tools a stable inspection format. Stays in-process; no service
layer required.

### H. Retrieval explainability for FTS / episode / procedure hits  *(small)*
`AugmentedContext.graph_facts` carries `why_retrieved`; FTS hits and
episodes/procedures don't. Add a short reason chip
(`fts_match("postgres pool")`, `vec_topk(rrf=0.024)`, `procedure_fts`) to
each, so downstream prompts can quote the reason instead of guessing.

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
