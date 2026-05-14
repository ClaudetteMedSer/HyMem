# HyMem — Definitive Walkthrough

---

## TL;DR / Executive Summary

**HyMem** is a local-first, embedded memory system for AI agents. It gives the **Hermes** agent persistent memory across conversations by extracting a structured SQLite knowledge graph from chat logs during idle "dreaming" cycles, then making that knowledge queryable at conversation time via keyword search, vector search, semantic graph search, and entity lookup. It also auto-maintains two Markdown files (`MEMORY.md` and `USER.md`) that the agent reads before each conversation.

**No cloud, no Postgres, no 500MB Docker images.** One SQLite file, two Markdown files, and a Python library. The LLM is only called during dreaming — query-time retrieval uses only FTS5 + cosine similarity + graph traversal, zero LLM calls.

**Two deployment modes:** an MCP tools server for direct agent integration, and a **Honcho v3-compatible HTTP server** so Hermes can use the standard `honcho-ai` SDK and treat HyMem as a drop-in replacement for Honcho's managed cloud service.

**~4,500 lines of Python**, zero npm, zero Docker required.

---

## Quickstart

`pip install` HyMem, point it at your model provider with environment
variables, and run a server — no config files.

```bash
pip install 'hymem[server]'

export HYMEM_LLM_API_KEY=sk-...        # extraction LLM (DeepSeek/OpenAI/...)
export HYMEM_EMBEDDING_API_KEY=sk-...  # optional — omit for FTS-only retrieval
export HYMEM_ROOT=~/.hermes            # optional — SQLite + Markdown live here

hymem-doctor      # preflight: verify config before launching
hymem-honcho      # Honcho v3-compatible HTTP server on :8765
# or
hymem-server      # MCP tools server
```

`hymem-doctor` prints the resolved provider/model/URLs and checks that the
keys work, the endpoints are reachable, `sqlite-vec` loads, the schema
migrates, and the embedding dimension matches any existing vector table. If the
LLM key is missing the servers refuse to start with a clear message rather than
failing deep inside the first request.

Defaults target DeepSeek; override `HYMEM_LLM_BASE_URL` / `HYMEM_LLM_MODEL`
(and the `HYMEM_EMBEDDING_*` equivalents) for OpenAI or any OpenAI-compatible
endpoint. Full inventory in [§9 Configuration](#9-configuration).

---

## 1. What It Does

HyMem solves a specific problem: **AI agents forget everything between sessions.** Every conversation starts from scratch. HyMem gives Hermes a persistent, structured memory of:

- **What tools/libraries/services the user uses** (knowledge graph triples)
- **What the user prefers, rejects, or avoids** (behavioral profile)
- **What the project's architecture looks like** (dependency hubs, tool preferences)
- **What was discussed in past conversations** (full-text searchable chunks, episodes, procedures)
- **How tasks are performed** (step-by-step procedural memory)

All of this is surfaced automatically to Hermes before each user message via the `augment()` call, so the agent can answer "We should use uv instead of Docker" without being reminded.

---

## 2. Architecture at 30,000 Feet

```
┌─────────────────────────────────────────────────────┐
│                    Hermes Agent                      │
│  (calls HyMem at start of conversation, after each   │
│   user message, and during idle time)                │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
     MCP Tools              Honcho v3 SDK
     (hymem-server)         (HONCHO_BASE_URL)
           │                      │
           ▼                      ▼
    ┌──────────────┐    ┌────────────────────┐
    │  server.py   │    │  honcho/  package  │
    │  7 MCP tools │    │  18 HTTP endpoints │
    └──────┬───────┘    └─────────┬──────────┘
           │                      │
           └──────────┬───────────┘
                      ▼
              ┌──────────────┐
              │ bootstrap.py  │  env → HyMem (shared singleton)
              └──────┬────────┘
                     ▼
              ┌──────────────┐
              │   api.py      │
              │  HyMem class  │
              └──────┬────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐ ┌──────────┐ ┌─────────┐
   │ Dreaming │ │  Query   │ │ Session │
   │ (3-phase)│ │ (augment)│ │  Log    │
   └────┬─────┘ └────┬─────┘ └────┬────┘
        │            │            │
        └────────────┼────────────┘
                     ▼
        ┌────────────────────────┐
        │   SQLite + Markdown     │
        │  (~/.hermes/hymem.sqlite│
        │   ~/.hermes/MEMORY.md   │
        │   ~/.hermes/USER.md)    │
        └────────────────────────┘
```

## 3. Directory Map

```
hymem/
├── api.py              Public HyMem class — single entry point for all ops
├── config.py           HyMemConfig dataclass — all tunable parameters
├── session.py          Session lifecycle (open/close) and message logging
├── bootstrap.py        Env-var resolution + build_from_env() + shared singleton
├── doctor.py           hymem-doctor — preflight diagnostics
├── server.py           MCP server — 7 tools (capture, log, dream, augment,
│                         profile, alias, retract)
├── honcho_server.py    Back-compat shim → hymem.honcho
│
├── honcho/             Honcho v3-compatible HTTP server
│   ├── app.py          FastAPI routes (18 endpoints) + entry point
│   ├── models.py       Typed Pydantic request models (one per endpoint body)
│   └── adapters.py     Response shaping + request-shape normalization
│
├── core/
│   ├── db.py           SQLite connection management (WAL), schema init, migrations
│   ├── schema.sql      19 tables + FTS5 virtual tables + triggers
│   └── markdown_io.py  Read/write HTML-comment-delimited sections in MD files
│
├── dreaming/
│   ├── runner.py       Orchestrates full pipeline with advisory lock
│   ├── chunks.py       Regex-based high-salience chunk extraction
│   ├── canonicalize.py Deterministic entity name normalization + aliases
│   ├── mentions.py     Entity mention indexing for decay calculations
│   ├── embeddings.py   Batch embedding of chunks + knowledge-graph edges (JSON + sqlite-vec)
│   ├── phase1.py       LLM extraction: triples + behavioral markers
│   ├── phase2.py       Consolidation: markers→profile, graph→MEMORY.md
│   ├── phase3.py       Co-occurrence-aware decay + retraction
│   ├── inference.py    Transitive closure over depends_on edges
│   ├── episodes.py     LLM-powered episodic memory extraction
│   ├── procedures.py   LLM-powered procedural memory extraction
│   ├── summary.py      LLM-powered session summarization
│   └── retention.py    Chunk pruning with graph-aware eviction
│
├── extraction/
│   ├── llm.py          LLMClient Protocol + StubLLMClient (for tests)
│   ├── embeddings.py   EmbeddingClient Protocol + StubEmbeddingClient
│   ├── triples.py      LLM prompt → (subject, predicate, object, polarity)
│   ├── markers.py      LLM prompt → behavioral markers
│   └── prompts/        System/user prompts for extraction, episodes, procedures,
│                         summaries, and reranking
│
├── query/
│   ├── augment.py      FTS5 + vector + RRF merge + LLM rerank + hybrid graph ranker
│   ├── predicate_routing.py  Keyword → predicate mapping for query expansion
│   ├── conflicts.py    Contradiction detection over the knowledge graph
│   └── entities.py     Token-based entity matching against knowledge graph
│
└── contrib/
    ├── openai_client.py          OpenAI-compatible LLM client (DeepSeek default)
    └── openai_embedding_client.py OpenAI-compatible embedding client
```

---

## 4. The Data Model (19 SQLite Tables)

**Conversation storage:**
- `sessions` — session ID + start/end timestamps + LLM-generated summary
- `messages` — raw turns (user, assistant, system, tool)

**Extraction artifacts:**
- `chunks` — high-salience conversation segments
- `chunks_fts` — FTS5 virtual table over chunk text (BM25 search)
- `chunk_embeddings` — JSON-encoded embedding vectors
- `processed_chunks` — idempotency tracking per (chunk_id, prompt_version)
- `entity_mentions` — inverted index: chunk → canonical entity
- `entity_types` — canonical entity → type classification (language, framework, database, etc.)

**Knowledge graph:**
- `knowledge_graph` — (subject, predicate, object) triples with evidence counters, confidence, status (active/stale/retracted), and derived flag for inferred edges
- `kg_evidence` — per-source provenance linking edges to chunks, with value/text/temporal metadata
- `edge_embeddings` — JSON-encoded embedding vectors for knowledge graph edges, keyed on triple text so churning derived edges reuse cached vectors
- `entity_aliases` — surface form → canonical entity mapping

**Behavioral profiling:**
- `behavioral_markers` — raw extracted signals (correction, preference, rejection, style)
- `profile_entries` — structured profile with evidence tracking

**Episodic & procedural memory:**
- `episodes` — named, summarized conversation segments with outcomes and entities
- `episodes_fts` — FTS5 search over episode titles and summaries
- `procedures` — step-by-step workflows with triggers and involved entities
- `procedures_fts` — FTS5 search over procedure names, descriptions, and steps

**Self-improvement:**
- `extraction_feedback` — wrongly-extracted triples stored as negative examples for future extraction

**Operational:**
- `peers` — Honcho peer registry (peer_id → role mapping)
- `run_lock` — advisory mutex for dreaming concurrency
- `dream_runs` — per-cycle audit log

---

## 5. The Dreaming Pipeline (How Memory Gets Built)

Dreaming is the offline process that converts raw chat logs into structured knowledge. It's called by Hermes after each conversation (or on a cron-like schedule). The pipeline holds an advisory lock so concurrent runs bail out safely.

### Phase 1 — Extraction (`dreaming/phase1.py`)

1. **Chunking**: Regex-based salience detection extracts high-signal conversation segments (min 30 chars). Chunks are persisted with a `salience_reason` field.
2. **Entity mention indexing**: Each chunk's text is scanned for known entity surface forms, populating the `entity_mentions` inverted index.
3. **LLM extraction**: Each unprocessed chunk is sent to the LLM with a locked-vocabulary prompt:
   - **Triples**: `{subject, predicate, object, polarity}` where predicate must be one of 18: `uses`, `depends_on`, `prefers`, `rejects`, `avoids`, `replaces`, `conflicts_with`, `deploys_to`, `part_of`, `equivalent_to`, `implements`, `contains`, `configured_with`, `requires_version`, `runs_on`, `connects_to`, `generates`, `tested_by`. Polarity is +1 (assertion) or -1 (negation/retraction). Optional fields: `value_text`, `value_numeric`, `value_unit`, `temporal_scope`.
   - **Markers**: `{kind, statement}` where kind is one of: `correction`, `preference`, `rejection`, `style`. Only explicit behavioral signals — no mood/emotion inference.
   - **Entity types**: LLM also infers entity type labels (language, framework, database, service, tool, etc.) for query expansion.
4. **Feedback-driven extraction**: Before processing, the runner loads up to 10 recently retracted triples from `extraction_feedback` and injects them into the prompt as negative examples: "DO NOT extract these relationships." This self-corrects past hallucination patterns.
5. **Entity canonicalization**: Surface forms (e.g., "Postgres", "PostgreSQL", "postgresql") are normalized via Unicode folding, CamelCase splitting, article/parenthetical stripping, and an alias table.
6. **Knowledge graph upsert**: New triples insert edges, repeated triples reinforce evidence counters. Negations add negative evidence.
7. **Idempotency**: Each chunk is processed at most once per `prompt_version`. Bump the version string in config and all chunks reprocess with new prompts.

### Inter-Phase Steps (`dreaming/runner.py`)

After chunk extraction per session, three additional LLM-powered steps run before Phase 2:

- **Episode extraction** (`episodes.py`): Groups adjacent chunks from the same session, asks the LLM to identify distinct episodes with titles, summaries, outcomes, and key entities. Stored in `episodes` (FTS5-searchable).
- **Session summarization** (`summary.py`): Generates a one-sentence summary of what was accomplished in the session. Stored in `sessions.summary` and surfaced via Honcho's context endpoint.
- **Procedure extraction** (`procedures.py`): Detects step-by-step workflows described in the conversation (e.g., "Deploy to staging"), extracts ordered steps with tools and triggers, and stores them in `procedures` (FTS5-searchable).

### Phase 2 — Consolidation (`dreaming/phase2.py`)

**Deterministic, no LLM.** Two sub-steps:

**Profile consolidation**: Unconsolidated behavioral markers are promoted into `profile_entries`. Repeats reinforce (+1 to `pos_evidence`), contradictions create separate entries rather than silently overwriting. Entries are capped at `profile_max_entries` (default: 16), dropping the weakest. The `USER.md` auto-section is rewritten via `markdown_io.write_section()`.

**Insight generation**: The knowledge graph is queried for:
- **Dependency hubs** — objects depended on by 2+ subjects with confidence > 0.6 (e.g., "`uv` is a shared dependency of: local_dev, ci_pipeline"). Only non-derived (direct) edges are considered.
- **Strong preferences/rejections** — edges with confidence > 0.7
- **Contradictions** — edges with both positive and negative evidence

Results are written to `MEMORY.md`'s auto-section, capped at `insights_max_entries` (default: 12).

### Phase 3 — Decay + Inference (`dreaming/phase3.py`, `inference.py`)

**Co-occurrence-aware decay.** The key insight: an edge should only lose confidence if the topic was re-discussed and the relationship wasn't restated. Dormant topics are left alone. Only non-derived edges decay — derived edges are recomputed from scratch.

1. For each active, non-derived edge whose `last_reinforced` is older than the decay window (default: 30 days):
   - Check if any chunk in the decay window mentions the edge's subject or object **without** providing evidence for the edge
   - If yes → add 1 to `neg_evidence` (soft contradiction)
   - If no → leave alone (topic hasn't resurfaced)
2. Any edge whose Laplace-smoothed confidence `(pos+1)/(pos+neg+2)` drops below the retract threshold (default: 0.15) gets `status = 'retracted'` — kept for audit but excluded from query results.

**Transitive inference** (`inference.py`): After decay, computes transitive closure over `depends_on` edges via BFS. For each entity, discovers 2+-hop dependency paths and inserts derived edges with `derived=1`. Confidence is the product of edge confidences along the path. Edges below the retract threshold are filtered out.

After inference, Phase 2 insights are refreshed to reflect the new graph state, and old unreferenced chunks are pruned via `retention.py`.

**Edge embedding** (`embeddings.py`): Once the graph has settled, every active edge — base and derived — is embedded as `"{subject} {predicate} {object}"` text into `edge_embeddings` and the `vec_edges` sqlite-vec table, so the query layer can do semantic search against the graph (see §6). The cache is keyed on triple text, not edge id: derived edges are deleted and recreated every cycle with fresh ids, so keying on text means a recreated edge reuses its cached vector instead of re-hitting the embedding API. Only genuinely new triple texts cost an embedding call. Skipped entirely when no embedding client is configured.

---

## 6. Query-Time Augmentation (How Memory Gets Used)

When Hermes receives a user message, it calls `hy.augment(user_message)` which returns an `AugmentedContext`:

```
AugmentedContext(
    user_md: str,              # USER.md content
    memory_md: str,            # MEMORY.md content
    fts_hits: list[FtsHit],    # Ranked relevant chunks
    graph_facts: list[GraphFact],     # Ranked knowledge graph edges (see below)
    episodes: list[EpisodeHit],       # Matching episodes
    procedures: list[ProcedureHit],   # Matching procedures
    matched_entities: list[str],      # Entities found in user message
)
```

Each `GraphFact` carries the edge (`subject`, `predicate`, `object`), its `confidence`, evidence counters, `derived` flag, the final `score`, and a `why_retrieved` list of reason codes explaining *why* the edge surfaced — e.g. `["semantic_0.84", "predicate:uses", "entity_type:framework", "recency_3d", "entity_match"]`. The reason codes are the ranking formula itself, so the agent gets an explanation with zero extra LLM calls.

**How it works:**

1. **Load profile + insights** from `USER.md` and `MEMORY.md` (file read, instant)
2. **Keyword search** (`_fts_search`): Sanitize the query, tokenize it, wrap each token in FTS5-safe quotes, build an OR query, run against SQLite FTS5 with BM25 scoring. Returns top-k chunks (default: 5).
3. **Vector search** (`_vector_search`, optional): If an embedding client is available, uses sqlite-vec ANN if available or falls back to Python cosine similarity (capped at `embedding_max_scan` default 5000). Returns top-k.
4. **Reciprocal rank fusion** (`_rrf_merge`): Merge FTS and vector results via RRF: `score = sum(1/(60 + rank))` across each list. This hybrid approach captures both keyword relevance and semantic similarity.
5. **LLM reranking** (`_rerank`, optional): When FTS and vector disagree on the #1 result enough to trigger ambiguity (configurable threshold), the LLM scores each candidate's relevance to the query on a 1-5 scale. RRF and LLM scores are combined for final ranking.
6. **Episode search** (`_episode_search`): FTS5 search over episode titles and summaries for the query.
7. **Procedure search** (`_procedure_search`): FTS5 search over procedure names, descriptions, and steps.
8. **Entity matching** (`match_known_entities`): Tokenize the user message (including 2-3 word n-grams), lookup each token against `entity_aliases`, return canonical IDs.
9. **Entity type expansion** (`_expand_entities_by_type`): For matched entities, find other entities of the same type (e.g., if user mentions `uv`, also surface `pip` and `poetry` since they're all `package_manager` type). Records which type surfaced each expanded entity for the `entity_type:` reason code.
10. **Predicate routing** (`predicate_routing.py`): Map natural-language cues in the query to typed predicates — "what technologies" → `uses`/`runs_on`, "depends on" → `depends_on`, "configured" → `configured_with`, etc. Routing only ever *adds* signal: matching predicates get a score boost and a `predicate:` reason code, but no edge is ever filtered out.
11. **Hybrid graph ranker** (`_graph_lookup`): Gathers candidate edges from three sources — entity-anchored lookup (the entity appears as subject/object), semantic KNN against `vec_edges` (the query is embedded and matched against edge embeddings; falls back to Python cosine over `edge_embeddings` when sqlite-vec is unavailable), and predicate-routed lookup. Each unique edge is then scored:

    ```
    score = confidence × recency_weight × semantic_score × predicate_boost
    recency_weight = exp(-days_since_last_seen / graph_recency_half_life_days)
    ```

    With no embedding client the semantic term drops out and the score collapses to `confidence × recency × predicate_boost` — close to the prior confidence-and-recency leaderboard. The top `graph_top_k` (default: 8) edges are returned, each with its `why_retrieved` reason codes.

Hermes then assembles the prompt with this context — HyMem never dictates prompt structure.

### Contradiction detection (`conflicts.py`)

Separately from `augment()`, `hy.conflicts()` scans the knowledge graph for contradictions and returns a list of `Conflict` objects. Two kinds are surfaced:

- **competing_object** — the same subject pointing at different objects under a mutually-exclusive predicate (e.g. `atta [prefers] english` vs `atta [prefers] dutch`).
- **opposing_predicate** — the same subject/object pair joined by an opposing predicate pair (e.g. `team [prefers] docker` vs `team [rejects] docker`).

It's pure SQL over the existing schema — no LLM call — and ignores retracted and derived edges.

---

## 7. The Two Server Modes

### MCP Server (`hymem-server` → `server.py`)

Exposes 7 tools via the Model Context Protocol:

| Tool | Purpose |
|---|---|
| `hymem_capture` | Log a full conversation as JSON array + optionally dream (preferred method) |
| `hymem_log` | Log one turn at a time (fallback) |
| `hymem_dream` | Run a dreaming cycle manually |
| `hymem_augment` | Retrieve graph facts + FTS context for a message |
| `hymem_profile` | Return USER.md + MEMORY.md |
| `hymem_alias` | Register surface-form→canonical mapping |
| `hymem_retract` | Retract a wrongly extracted edge |

### Honcho HTTP Server (`hymem-honcho` → `hymem/honcho/`)

A FastAPI server implementing a **Honcho v3-compatible REST protocol** — 18 endpoints. Hermes can use the standard `honcho-ai` Python SDK by setting `HONCHO_BASE_URL=http://127.0.0.1:8765`.

The server is a small package, not a monolith: `models.py` holds the typed Pydantic request models (so an SDK shape mismatch is a clean 422, not an `AttributeError` deep in a handler), `adapters.py` owns all response shaping and request-shape normalization (one place that knows "what shape the SDK expects"), and `app.py` holds the routes. The pinned `honcho-ai` SDK is exercised end-to-end against a live server in `test_honcho_contract.py`.

| Endpoint | Maps to | Notes |
|---|---|---|
| `POST /v3/workspaces` | Get-or-create workspace | SDK auto-calls via `_ensure_workspace()` |
| `GET /v3/workspaces/{wid}` | Get workspace | |
| `POST /v3/workspaces/{wid}/peers` | Get-or-create peer | Role auto-inferred from peer_id pattern |
| `GET /v3/workspaces/{wid}/peers/{pid}` | Get peer by ID | Returns role + metadata |
| `POST /v3/workspaces/{wid}/sessions` | Get-or-create session | Registers peers from `peer_names` |
| `GET /v3/workspaces/{wid}/sessions/{sid}` | Get session by ID | Returns is_active, metadata |
| `POST .../sessions/{sid}/messages` | Log turns + bg dream | Dream cooldown: 60s default |
| `POST .../sessions/{sid}/messages/upload` | File upload as message | For migrating MEMORY.md/USER.md |
| `POST .../sessions/{sid}/messages/list` | Paginated message listing | page + size, returns total/pages |
| `POST .../sessions/{sid}/search` | `hy.augment()` as Message objects | Graph facts + FTS hits |
| `GET .../sessions/{sid}/context` | MEMORY.md + USER.md + recent turns + summary | Session summary from dreaming |
| `POST .../sessions/{sid}/peers` | Register peers + role mappings | |
| `GET .../sessions/{sid}/peers/{pid}/config` | Per-session peer config | |
| `GET .../peers/{pid}/card` | USER.md behavioral profile | |
| `GET .../peers/{pid}/context` | Peer-scoped context with optional search | |
| `POST .../peers/{pid}/representation` | Update peer representation | |
| `POST .../peers/{pid}/chat` | Dialectic Q&A via `hy.augment()` | Natural language queries |
| `GET /health` | Health check | |

**Key design choices in the Honcho server:**
- **Dream cooldown**: Background dreaming kicks at most once per configurable cooldown (env: `HYMEM_DREAM_COOLDOWN_SECONDS`, default 60s). Uses FastAPI `BackgroundTasks` so the HTTP response isn't blocked.
- **Background dreaming on a forked connection**: `_background_dream` runs on `HyMem.fork()` — a fresh SQLite connection that reuses the live instance's LLM/embedding clients — so a dream cycle never collides with concurrent `add_messages` writes.
- **Batched ingestion**: `add_messages` logs a whole batch under one transaction via `HyMem.log_messages()` rather than one `BEGIN IMMEDIATE` per turn.
- **Role inference**: Peer IDs matching `user[-_]|human|client|telegram|discord|slack` → user role, `agent|hermes|assistant|ai[-_]|bot|llm` → assistant role.
- **No LLM in query path**: Search, context, and chat endpoints only call `hy.augment()` — zero LLM calls, zero latency beyond SQLite reads.

---

## 8. Key Design Decisions

**Locked vocabulary (18 predicates).** No open-ended relation extraction. This means the knowledge graph is clean, queryable, and predictable — no hallucinated "loves" or "feels" edges. The predicate set covers technical relationships comprehensively: usage, dependency, preference, rejection, replacement, deployment, composition, configuration, versioning, runtime, connectivity, generation, testing, and interface conformance. The tradeoff: some relationships won't fit the schema, but the system errs on the side of silence rather than noise.

**Host-agent responsibility split.** Hermes owns *when* to call HyMem; HyMem owns *how* memory works. HyMem never assembles prompts, never decides token budgets, never injects itself into the agent's reasoning loop. It just returns structured pieces.

**Laplace-smoothed confidence.** Every edge's confidence is `(pos+1)/(pos+neg+2)` — a Bayesian-style smoothing that starts at 0.5 for an untested fact and converges toward truth as evidence accumulates.

**Co-occurrence-aware decay.** Unlike simple TTL-based decay (which would kill all old facts regardless of relevance), HyMem only decays edges whose entities have been re-discussed without reinforcement. This keeps the graph accurate without requiring constant LLM re-extraction.

**Transitive inference.** `depends_on` edges are transitively closed via BFS after each dreaming cycle. If `api depends_on postgres` and `postgres depends_on docker`, a derived edge `api depends_on docker` is added (marked `derived=1`, confidence = product of edge confidences). Derived edges are recomputed from scratch each cycle and excluded from decay.

**Semantic, explainable graph retrieval.** The knowledge graph isn't just keyword-matched — edges are embedded during dreaming, so `augment()` ranks them against the query by `semantic × confidence × recency × predicate_boost`. Every returned fact carries `why_retrieved` reason codes derived directly from that formula, so the agent sees *why* a fact was surfaced without a second LLM call. When no embedding client is configured the ranker degrades gracefully to confidence-and-recency ordering.

**Feedback-driven extraction.** When an edge is retracted, its chunk text and the extracted triple are stored in `extraction_feedback`. Before the next dreaming cycle, up to 10 recent retractions are injected as negative examples into the extraction prompt, teaching the LLM to avoid repeating past mistakes.

**Prompt-versioned idempotency.** Changing `prompt_version` in config causes automatic reprocessing of all chunks with the new prompts. Backward-incompatible prompt changes are trivial.

**Schema version guard.** The database schema version is checked against an expected constant. If a newer-schema DB is opened with older code, initialization raises a clear error rather than silently corrupting data.

**WAL by default.** Every connection is opened in WAL mode with `synchronous=NORMAL` and a 10s busy timeout, set in `connect()` so it applies before any migration runs. Background dreaming and live message ingestion run on separate connections without blocking each other or query-time reads — exercised directly by `test_concurrency.py`.

**Zero-config startup.** `bootstrap.build_from_env()` is the single source of truth for environment-variable resolution; both server entry points and `hymem-doctor` build on it. A missing extraction-LLM key fails fast at startup with an actionable message instead of surfacing deep inside the first request. `hymem-doctor` runs the full preflight (keys, endpoint reachability, sqlite-vec, schema migration, embedding-dimension drift) and prints the resolved provider/model/URLs.

**No external dependencies at core.** The `hymem` package itself has zero dependencies beyond Python stdlib + SQLite. LLM clients, FastAPI, and sqlite-vec are optional extras (`hymem[server]`); the pinned `honcho-ai` SDK used by the contract tests is the `hymem[honcho]` extra. The `contrib/` layer provides OpenAI-compatible clients but can be swapped via the `LLMClient` and `EmbeddingClient` Protocols.

**Managed Markdown sections.** `USER.md` and `MEMORY.md` use HTML comment delimiters (`<!-- HyMem:auto:section:start -->` / `<!-- HyMem:auto:section:end -->`). Humans can edit everything outside these sections; HyMem only touches its auto-sections. Atomic writes via tempfile + `os.replace()` prevent corruption.

**Advisory lock with stale takeover.** The `run_lock` table prevents concurrent dreaming cycles. If a holder process crashes, the lock is released after 5 minutes of inactivity so the system doesn't deadlock.

---

## 9. Configuration

All runtime config via environment variables. No config files. Run
`hymem-doctor` to print the resolved configuration and preflight every check
(keys, endpoint reachability, `sqlite-vec`, schema migration, embedding
dimension).

| Variable | Default | Purpose |
|---|---|---|
| `HYMEM_ROOT` | `~/.hermes` | Directory for sqlite + markdown files |
| `HYMEM_LLM_API_KEY` | `DEEPSEEK_API_KEY` | LLM API key |
| `HYMEM_LLM_BASE_URL` | `https://api.deepseek.com` | LLM endpoint |
| `HYMEM_LLM_MODEL` | `deepseek-chat` | Extraction model |
| `HYMEM_EMBEDDING_API_KEY` | (falls back to LLM key) | Embedding API key |
| `HYMEM_EMBEDDING_BASE_URL` | `https://api.deepseek.com` | Embedding endpoint |
| `HYMEM_EMBEDDING_MODEL` | `deepseek-embedding` | Embedding model |
| `HYMEM_HONCHO_HOST` | `127.0.0.1` | Honcho server bind address |
| `HYMEM_HONCHO_PORT` | `8765` | Honcho server port |
| `HYMEM_DREAM_COOLDOWN_SECONDS` | `60` | Min seconds between bg dream kicks |

Tunable in `HyMemConfig` dataclass (programmatic):

| Parameter | Default | Purpose |
|---|---|---|
| `salience_min_chars` | 30 | Min chunk size before extraction |
| `fts_top_k` | 5 | FTS results to return |
| `graph_top_k_per_entity` | 3 | Entity-anchored graph facts per matched entity |
| `embedding_max_scan` | 5000 | Max embeddings to scan in Python fallback |
| `graph_semantic_top_k` | 10 | KNN candidates pulled from `vec_edges` |
| `graph_predicate_top_k` | 10 | Edges pulled per predicate-routed query |
| `graph_top_k` | 8 | Final graph facts returned by `augment()` |
| `graph_recency_half_life_days` | 30.0 | Half-life for edge recency decay |
| `graph_recency_recent_days` | 7.0 | `days_since` under this emits a `recency_Nd` reason code |
| `graph_predicate_boost` | 1.5 | Score multiplier for routed-predicate edges |
| `decay_window_days` | 30 | Decay look-back window |
| `decay_factor` | 0.9 | (reserved, not yet used) |
| `retract_threshold` | 0.15 | Confidence below which edges retract |
| `profile_max_entries` | 16 | Max profile entries in USER.md |
| `insights_max_entries` | 12 | Max insights in MEMORY.md |
| `prompt_version` | `"v4"` | Bump to force full reprocessing |
| `dream_budget` | 50 | Max chunks to process per dreaming cycle |
| `max_chunks` | 50000 | Soft cap on total stored chunks |
| `retention_days` | 90 | Chunks newer than this always kept |
| `rerank_ambiguity_threshold` | 0.6 | Min RRF score drop to skip LLM reranking |

---

## 10. Test Coverage

**113 tests total, 100% passing** across 17 test files:

- `test_dreaming.py` — Full pipeline: chunk→extract→consolidate→decay
- `test_extraction.py` — Triple extraction, marker extraction, polarity handling
- `test_canonicalize.py` — Entity normalization, alias resolution, merging
- `test_chunks.py` — Salience detection, chunk persistence
- `test_embeddings.py` — Embedding creation and query, stub determinism
- `test_augment.py` — FTS search, vector search, RRF merge, graph lookup
- `test_graph_semantic.py` — Edge embedding, hybrid graph ranker, `why_retrieved` codes, predicate routing, contradiction detection
- `test_markdown_io.py` — Section read/write atomicity
- `test_integration.py` — End-to-end capture→dream→augment, retract workflow
- `test_phase3_perf.py` — Decay correctness, mention indexing, backfill idempotency
- `test_mcp_server.py` — MCP tool correctness (all 7 tools)
- `test_retract.py` — Edge retraction, alias resolution, idempotency
- `test_dream_runs.py` — Audit log persistence, lock-skip recording, error recording
- `test_honcho_server.py` — Full Honcho v3 protocol (18 endpoints, all passing)
- `test_honcho_contract.py` — Real honcho-ai SDK driven against a live server (response-parse contract)
- `test_concurrency.py` — Dreaming + ingestion + reads coexisting under WAL

---

## 11. Comparison with Honcho

| Dimension | HyMem | Honcho (plastic-labs) |
|---|---|---|
| **Scope** | Single-agent memory module for Hermes | Multi-tenant platform for stateful agents |
| **Architecture** | Embedded library (SQLite + 2 servers) | Client-server (FastAPI + Postgres + Redis + workers) |
| **Storage** | 1 SQLite file + 2 Markdown files | Postgres + pgvector + Redis cache |
| **Entity model** | Simple: user + assistant roles | Peer paradigm: all participants are "peers" |
| **Memory extraction** | "Dreaming" — multi-phase LLM pipeline with locked vocabulary, transitive inference, episode/procedure extraction, feedback learning | "Deriver" — background workers doing representation, summarization, peer cards |
| **Ontology** | Locked 18-predicate vocabulary + entity types | Open-ended reasoning, no fixed ontology |
| **Query interface** | FTS5 + vector + RRF + LLM rerank + semantic graph ranking (with `why_retrieved` explainability) + predicate routing + episode/procedure search | Chat API (natural language), context (token-budgeted), hybrid search |
| **Decay** | Co-occurrence-aware with confidence thresholds | Continual representation updates (implicit) |
| **Contradiction detection** | `conflicts()` surfaces competing-object and opposing-predicate edges (pure SQL) | Not available |
| **Self-improvement** | Feedback-driven extraction (negative examples from retractions) | Not available |
| **Honcho SDK compat** | v3-compatible protocol via the `hymem.honcho` package (18 endpoints, real-SDK contract tests) | Native |
| **Deployment** | Local-only, pip install, zero config | Managed cloud (app.honcho.dev) or self-hosted Docker/Fly.io |
| **SDKs** | Python + MCP + Honcho SDK | Python + TypeScript |
| **Maturity** | v0.1.0, ~4,500 lines | v3.0.6, 514 commits, 3.4k stars |
| **License** | Not specified | AGPL-3.0 |

**The key philosophical difference:** Honcho is a platform — multi-tenant, cloud-native, with a broad API surface for many use cases. HyMem is a tool — focused, embeddable, opinionated about what memory should look like. HyMem's locked vocabulary, co-occurrence-aware decay, transitive inference, semantic-and-explainable graph ranking, and feedback learning are design bets that prioritize precision over recall. Honcho prioritizes flexibility and scale.

**HyMem can self-host the Honcho experience.** With the Honcho server fully functional (18 endpoints, verified against the pinned `honcho-ai` SDK), Hermes can use the standard `honcho-ai` SDK and get the same API surface as Honcho's cloud — search, context, chat, peer management, sessions, pagination — without leaving the machine. This is HyMem's headline feature: **Honcho compatibility without any infrastructure.**

---

## 12. Limitations & Known Gaps

- **O(n) vector search fallback**: Cosine similarity in pure Python is slow beyond ~5K chunks (capped at `embedding_max_scan`). sqlite-vec is integrated for ANN but not available on all platforms.
- **No streaming**: The Honcho server doesn't implement SSE streaming for chat responses.
- **Single-writer database**: WAL mode lets reads run concurrently with the writer, and dreaming/ingestion run on separate connections, but SQLite still serializes the two writers. Fine for a single-agent setup, not for multi-tenant.
- **No authentication**: Both MCP and Honcho servers are unauthenticated — they assume localhost-only access.
- **Latin-script only**: Canonicalization, query-time entity matching, and the LLM prompts handle Latin-script languages (English, Dutch, French, German, Spanish, etc.) — accents are folded into canonical keys. Chunking salience triggers are tuned for English and Dutch; other languages fall back to length-based salience (the LLM is still the real filter). Non-Latin scripts (CJK, Cyrillic, Arabic) are not supported.
- **LLM-dependent extraction quality**: While feedback learning helps, extraction quality ultimately depends on the LLM's capabilities. A weak LLM will produce noisy graphs.
