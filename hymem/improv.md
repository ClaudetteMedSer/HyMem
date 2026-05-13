Here's a comprehensive expansion roadmap ordered by impact-to-effort ratio. Each tier builds on the last.

---

## Tier 1 — Immediate Wins (hours, not days)

### 1a. Vector Search → sqlite-vec (Scalability)

**Current bottleneck:** `hymem/query/augment.py:128-174` — cosine similarity is O(n) in pure Python, scanning all rows one at a time.

**Fix:** Replace with the `sqlite-vec` extension. It's a single C library loaded at runtime, provides native ANN with L2/cosine, and stores vectors as compact blobs instead of JSON strings.

Concrete changes:
- `hymem/core/schema.sql`: Add `chunk_embeddings_vec` virtual table
- `hymem/dreaming/embeddings.py`: Write to vec table instead of JSON
- `hymem/query/augment.py:128-174`: Replace the Python cosine loop with `SELECT ... FROM chunk_embeddings_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?`
- Drop the `embedding_max_scan=5000` cap entirely

**Impact:** Goes from 5K cap → millions of chunks. Query latency drops from ~50ms to ~2ms. This unblocks everything else — you can't build richer memory if retrieval doesn't scale.

### 1b. Expand the Predicate Vocabulary (Expressiveness)

**Current bottleneck:** 10 predicates hardcoded in `hymem/extraction/prompts/__init__.py:6-17` and enforced via CHECK constraint in `schema.sql:94-97`.

**Add these 8 predicates:**

| Predicate | Meaning | Example |
|---|---|---|
| `implements` | Entity A realizes interface/contract B | `auth_service implements OIDC` |
| `contains` | Entity A owns/holds B as subcomponent | `monorepo contains frontend` |
| `configured_with` | Entity A is parameterized by B | `nginx configured_with /etc/nginx.conf` |
| `requires_version` | Entity A needs specific version of B | `pipeline requires_version python>=3.11` |
| `runs_on` | Runtime / execution target | `api runs_on kubernetes` |
| `connects_to` | Network / data flow connection | `frontend connects_to backend:8080` |
| `generates` | Entity A produces/outputs B | `webpack generates bundle.js` |
| `tested_by` | Testing relationship | `auth_module tested_by pytest` |

Concrete changes:
- `hymem/extraction/prompts/__init__.py`: Add to `ALLOWED_PREDICATES` tuple
- `hymem/core/schema.sql`: Update the CHECK constraint
- Bump `prompt_version` → `"v2"` to force reprocessing of all chunks with the richer vocabulary

**Impact:** Doubles the knowledge graph's expressiveness. The most common missing relationship types (configuration, versioning, deployment, testing) become first-class.

### 1c. Numeric & Temporal Fact Extraction (Expressiveness)

**Current bottleneck:** Only bare `(S, P, O)` triples. No numbers, no dates, no "since when" scoping.

**Add to the triple extraction prompt and schema:**

```
# New triple fields in kg_evidence:
value_text TEXT,      # "5 seconds", ">=3.11"
value_numeric REAL,   # 5.0
value_unit TEXT,      # "seconds"
temporal_scope TEXT,  # "since 2024", "during migration"
```

The LLM already encounters these — it just can't store them. The prompt change is minimal: "If the predicate involves a numeric value, a version, a time window, or a quantity, include it in value_text."

Concrete changes:
- `kg_evidence` table: add `value_text`, `value_numeric`, `value_unit`, `temporal_scope` columns
- `hymem/extraction/triples.py`: Parse these fields from LLM output
- `hymem/extraction/prompts/__init__.py`: Update `TRIPLE_SYSTEM` to instruct extraction of numeric/temporal data
- Bump `prompt_version` → `"v2"`

**Impact:** The graph can now answer "What Python version does the project require?" and "When did we switch from Docker to uv?" without the user restating.

---

## Tier 2 — Structural Expansions (days)

### 2a. Episodic Memory Layer

**Current bottleneck:** Conversations leave chunks for FTS, sessions for lifecycle tracking, but no structured "what happened" records. The system can't answer "Tell me about last Tuesday's debugging session."

**Add `episodes` table:**

```sql
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    title TEXT NOT NULL,              -- LLM-generated: "Debugging Postgres connection pool"
    summary TEXT NOT NULL,            -- LLM-generated: 2-3 sentence narrative
    participants TEXT NOT NULL,       -- JSON array of peer roles
    start_message_id INTEGER,         -- first message in episode
    end_message_id INTEGER,           -- last message
    outcome TEXT,                     -- "resolved", "blocked", "deferred"
    key_entities TEXT,                -- JSON array: ["postgresql", "connection_pool"]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**How it works:** After Phase 1 chunking, before Phase 2 consolidation, add a new step: group adjacent chunks from the same session, ask the LLM to title + summarize + classify outcome. This is one LLM call per session (not per chunk), so cost is negligible.

Concrete changes:
- `hymem/core/schema.sql`: Add `episodes` table
- `hymem/dreaming/episodes.py`: New module — LLM-powered episode extraction
- `hymem/dreaming/runner.py`: Add episode extraction step between Phase 1 and Phase 2
- `hymem/query/augment.py`: Add episode search via FTS over title+summary
- `hymem/extraction/prompts/__init__.py`: Add episode summarization prompts

**Impact:** Hermes can now say "Last Tuesday we debugged the Postgres connection pool — the issue was `pool_size` set too low. We resolved it by bumping to 20." without the user providing any of that context.

### 2b. Entity Hierarchies & Properties

**Current bottleneck:** Every entity is a flat string. `uv` and `pip` and `poetry` are three unrelated nodes despite all being Python package managers.

**Add `entity_types` and `entity_properties` tables:**

```sql
CREATE TABLE entity_types (
    entity_canonical TEXT NOT NULL REFERENCES entity_aliases(canonical),
    type TEXT NOT NULL,              -- "package_manager", "database", "language", "service"
    confidence REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (entity_canonical, type)
);

CREATE TABLE entity_properties (
    entity_canonical TEXT NOT NULL REFERENCES entity_aliases(canonical),
    key TEXT NOT NULL,               -- "language", "runtime", "category"
    value TEXT NOT NULL,             -- "python", "kubernetes", "build_tool"
    source_chunk_id TEXT REFERENCES chunks(id),
    PRIMARY KEY (entity_canonical, key)
);
```

Populate from the LLM during Phase 1 extraction alongside triples. The extraction prompt already has entity names — adding type inference is a natural extension:
> "For each entity you mention, also output its type: language, framework, database, service, tool, library, file, environment, protocol, or container."

Concrete changes:
- `hymem/core/schema.sql`: Add entity type/property tables
- `hymem/extraction/triples.py`: Parse entity types from LLM output
- `hymem/extraction/prompts/__init__.py`: Add entity type inference to triple prompt
- `hymem/query/augment.py`: Use types for query expansion (if user asks about "package management", also retrieve all entities of type `package_manager`)
- Bump `prompt_version`

**Impact:** Query expansion by type means "what build tools do we use?" returns `uv`, `pip`, `poetry` even if the word "build" never appears near those entities.

### 2c. Hybrid Reranking (Retrieval Quality)

**Current bottleneck:** Reciprocal rank fusion (`augment.py:177-194`) is a simple heuristic. The top-5 results from FTS + vector are merged by rank position, not by actual relevance to the query.

**Add a cross-encoder reranking step:**

After RRF merge returns top-k candidates (say, k=20), run a lightweight cross-encoder model to score each candidate against the query:

```
candidate_text | query → relevance score [0,1]
```

Options: `mxbai-rerank-base` (384-dim, runs on CPU in <10ms per candidate), or use the existing LLM client with a simple prompt ("Rate relevance 1-5: query vs chunk").

Concrete changes:
- `hymem/query/rerank.py`: New module — cross-encoder or LLM-based reranking
- `hymem/query/augment.py`: Insert rerank step between RRF and final top-k selection
- `hymem/config.py`: Add `rerank_top_k: int = 20`, `rerank_model: str = "llm"` or `"cross-encoder"`

**Impact:** 20-40% improvement in retrieval precision without changing the underlying search infrastructure. This is the highest-impact single change for retrieval quality per line of code.

---

## Tier 3 — Advanced Capabilities (weeks)

### 3a. Feedback-Driven Extraction Improvement

**Current bottleneck:** Retracted edges are just marked `status='retracted'`. The system never learns why an edge was wrong to avoid similar mistakes.

**Add `extraction_feedback` table and a few-shot learner:**

```sql
CREATE TABLE extraction_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_text_snippet TEXT NOT NULL,   -- the text that caused the bad extraction
    extracted_triple TEXT NOT NULL,     -- the (S,P,O) that was wrong
    feedback_type TEXT NOT NULL,        -- 'retracted', 'corrected', 'confirmed'
    corrected_triple TEXT,              -- if corrected: what it should have been
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

When Hermes calls `hymem_retract`, store the chunk snippet and the bad triple. After N retractions accumulate (say, 20), inject them as few-shot negative examples into the extraction prompt:

> "Here are examples of triples that were PREVIOUSLY EXTRACTED INCORRECTLY. Do NOT extract these: ..."

Concrete changes:
- `hymem/core/schema.sql`: Add `extraction_feedback` table
- `hymem/dreaming/phase1.py`: Before extraction, load recent feedback and inject into prompt
- `hymem/api.py`: `retract_edge()` also stores feedback
- `hymem/extraction/prompts/__init__.py`: Add few-shot negative example placeholder

**Impact:** The system self-corrects. If it keeps hallucinating "project uses Docker" when it sees the word "Docker", after a few retractions it stops making that mistake.

### 3b. Multi-Hop Inference

**Current bottleneck:** `_graph_lookup()` in `augment.py:197-227` does 1-hop lookup. If `api depends_on postgresql` and `postgresql uses docker-compose`, asking "does api use docker?" gets nothing.

**Add transitive closure during dreaming, stored as derived edges:**

```sql
-- Derived edges from transitive closure (marked so they can be recomputed)
ALTER TABLE knowledge_graph ADD COLUMN derived BOOLEAN NOT NULL DEFAULT 0;
```

After Phase 3 decay, run a simple BFS from each entity to discover 2-hop paths. Add derived edges with `derived=1` and lower confidence (product of source confidences).

For the `depends_on` predicate specifically, also support:
- `A depends_on B, B depends_on C → A depends_on C` (transitive)
- `A uses B, B depends_on C → A transitively_depends_on C`

Concrete changes:
- `hymem/core/schema.sql`: Add `derived` column
- `hymem/dreaming/phase3.py` or new `phase4.py`: Transitive closure step
- `hymem/query/augment.py`: Include derived edges in graph lookup

**Impact:** The graph becomes a reasoning engine, not just a lookup table. Hermes can answer "Will switching Postgres versions affect the API?" without explicit documentation.

### 3c. Procedural Memory

**Current bottleneck:** Triples capture declarative knowledge (what is true). They don't capture procedural knowledge (how to do things).

**Add `procedures` table for step-by-step workflows:**

```sql
CREATE TABLE procedures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                -- "Deploy to staging"
    description TEXT,                  -- LLM summary
    steps TEXT NOT NULL,               -- JSON array: [{"order":1,"action":"...","tool":"..."}, ...]
    triggers TEXT,                     -- JSON array: ["deploy", "ship", "release"]
    entities_involved TEXT,            -- JSON array: ["docker", "kubernetes", "staging"]
    source_chunk_ids TEXT,             -- JSON array of chunk IDs
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

During Phase 1, the LLM is asked: "Did the user or assistant describe a step-by-step procedure? If yes, extract the steps as an ordered list."

**Impact:** Hermes can answer "How do I deploy to staging?" by retrieving the procedure from memory rather than asking the user to re-explain.

### 3d. Session Summarization

**Current bottleneck:** Honcho's deriver does session summarization. HyMem has no equivalent. Sessions end with `ended_at` set but no summary.

**Add LLM-generated session summaries during dreaming:**

When Phase 1 processes a session's chunks, also generate a one-sentence summary: "User debugged Postgres connection pool issues, resolved by adjusting pool_size and adding health checks."

Store in `sessions.summary` column and use in `get_context()` endpoint to provide the Honcho-expected `summary` field with real content instead of just dump MEMORY.md.

Concrete changes:
- `hymem/core/schema.sql`: Add `sessions.summary TEXT`
- `hymem/dreaming/runner.py`: After chunk extraction, generate session summary
- `hymem/honcho_server.py`: Use `sessions.summary` in context endpoint
- `hymem/extraction/prompts/__init__.py`: Add session summary prompt

**Impact:** Makes the Honcho context endpoint actually useful. Hermes gets a one-line "what happened last time" before diving into search results.

---

## Implementation Priority

```
Phase 1 (this week):
  ├── 1a: sqlite-vec vector search       ← unblocks everything
  ├── 1b: Expand predicates (8 new)       ← biggest expressiveness win per line
  └── 1c: Numeric/temporal facts          ← triples become useful, not just symbolic

Phase 2 (next week):
  ├── 2c: Hybrid reranking                ← largest retrieval quality gain
  ├── 2a: Episodic memory                 ← "what happened" recall
  └── 2b: Entity hierarchies + types      ← query expansion, better graph traversal

Phase 3 (following weeks):
  ├── 3a: Feedback-driven extraction      ← self-healing system
  ├── 3d: Session summarization           ← completeness for Honcho parity
  ├── 3b: Multi-hop inference             ← graph becomes reasoning engine
  └── 3c: Procedural memory               ← "how to" recall
```

