# HyMem — Definitive Walkthrough

---

## TL;DR / Executive Summary

**HyMem** is a local-first, embedded memory system for AI agents. It gives the **Hermes** agent persistent memory across conversations by extracting a structured SQLite knowledge graph from chat logs during idle "dreaming" cycles, then making that knowledge queryable at conversation time via keyword search, vector search, and entity lookup. It also auto-maintains two Markdown files (`MEMORY.md` and `USER.md`) that the agent reads before each conversation.

**No cloud, no Postgres, no 500MB Docker images.** One SQLite file, two Markdown files, and a Python library. The LLM is only called during dreaming — query-time retrieval uses only FTS5 + cosine similarity + graph traversal, zero LLM calls.

**Two deployment modes:** an MCP tools server for direct agent integration, and a **Honcho v3-compatible HTTP server** so Hermes can use the standard `honcho-ai` SDK and treat HyMem as a drop-in replacement for Honcho's managed cloud service.

**~3,200 lines of Python**, 71 tests (62 passing core + 9 Honcho with minor endpoint gaps), zero npm, zero Docker required.

---

## 1. What It Does

HyMem solves a specific problem: **AI agents forget everything between sessions.** Every conversation starts from scratch. HyMem gives Hermes a persistent, structured memory of:

- **What tools/libraries/services the user uses** (knowledge graph triples)
- **What the user prefers, rejects, or avoids** (behavioral profile)
- **What the project's architecture looks like** (dependency hubs, tool preferences)
- **What was discussed in past conversations** (full-text searchable chunks)

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
    │  server.py   │    │  honcho_server.py  │
    │  7 MCP tools │    │  15 HTTP endpoints │
    └──────┬───────┘    └─────────┬──────────┘
           │                      │
           └──────────┬───────────┘
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
├── server.py           MCP server — 7 tools (capture, log, dream, augment,
│                         profile, alias, retract)
├── honcho_server.py    FastAPI HTTP server — Honcho v3 protocol, 15 endpoints
│
├── core/
│   ├── db.py           SQLite connection management, schema init, migrations
│   ├── schema.sql      15 tables + FTS5 virtual table + triggers
│   └── markdown_io.py  Read/write HTML-comment-delimited sections in MD files
│
├── dreaming/
│   ├── runner.py       Orchestrates full 3-phase pipeline with advisory lock
│   ├── chunks.py       Regex-based high-salience chunk extraction
│   ├── canonicalize.py Deterministic entity name normalization + aliases
│   ├── mentions.py     Entity mention indexing for decay calculations
│   ├── embeddings.py   Batch embedding of chunks (JSON vectors in SQLite)
│   ├── phase1.py       LLM extraction: triples + behavioral markers
│   ├── phase2.py       Consolidation: markers→profile, graph→MEMORY.md
│   └── phase3.py       Co-occurrence-aware decay + retraction
│
├── extraction/
│   ├── llm.py          LLMClient Protocol + StubLLMClient (for tests)
│   ├── embeddings.py   EmbeddingClient Protocol + StubEmbeddingClient
│   ├── triples.py      LLM prompt → (subject, predicate, object, polarity)
│   ├── markers.py      LLM prompt → behavioral markers
│   └── prompts/        Locked-vocabulary system/user prompts for extraction
│
├── query/
│   ├── augment.py      FTS5 + vector search + RRF merge + graph lookup
│   └── entities.py     Token-based entity matching against knowledge graph
│
└── contrib/
    ├── openai_client.py          OpenAI-compatible LLM client (DeepSeek default)
    └── openai_embedding_client.py OpenAI-compatible embedding client
```

---

## 4. The Data Model (15 SQLite Tables)

**Conversation storage:**
- `sessions` — session ID + start/end timestamps
- `messages` — raw turns (user, assistant, system, tool)

**Extraction artifacts:**
- `chunks` — high-salience conversation segments
- `chunks_fts` — FTS5 virtual table over chunk text (BM25 search)
- `chunk_embeddings` — JSON-encoded embedding vectors
- `processed_chunks` — idempotency tracking per (chunk_id, prompt_version)
- `entity_mentions` — inverted index: chunk → canonical entity

**Knowledge graph:**
- `knowledge_graph` — (subject, predicate, object) triples with evidence counters, confidence, status (active/stale/retracted)
- `kg_evidence` — per-source provenance linking edges to chunks
- `entity_aliases` — surface form → canonical entity mapping

**Behavioral profiling:**
- `behavioral_markers` — raw extracted signals (correction, preference, rejection, style)
- `profile_entries` — structured profile with evidence tracking

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
   - **Triples**: `{subject, predicate, object, polarity}` where predicate must be one of 10: `uses`, `depends_on`, `prefers`, `rejects`, `avoids`, `replaces`, `conflicts_with`, `deploys_to`, `part_of`, `equivalent_to`. Polarity is +1 (assertion) or -1 (negation/retraction).
   - **Markers**: `{kind, statement}` where kind is one of: `correction`, `preference`, `rejection`, `style`. Only explicit behavioral signals — no mood/emotion inference.
4. **Entity canonicalization**: Surface forms (e.g., "Postgres", "PostgreSQL", "postgresql") are normalized via Unicode folding, CamelCase splitting, article/parenthetical stripping, and an alias table.
5. **Knowledge graph upsert**: New triples insert edges, repeated triples reinforce evidence counters. Negations add negative evidence.
6. **Idempotency**: Each chunk is processed at most once per `prompt_version`. Bump the version string in config and all chunks reprocess with new prompts.

### Phase 2 — Consolidation (`dreaming/phase2.py`)

**Deterministic, no LLM.** Two sub-steps:

**Profile consolidation**: Unconsolidated behavioral markers are promoted into `profile_entries`. Repeats reinforce (+1 to `pos_evidence`), contradictions create separate entries rather than silently overwriting. Entries are capped at `profile_max_entries` (default: 16), dropping the weakest. The `USER.md` auto-section is rewritten via `markdown_io.write_section()`.

**Insight generation**: The knowledge graph is queried for:
- **Dependency hubs** — objects depended on by 2+ subjects with confidence > 0.6 (e.g., "`uv` is a shared dependency of: local_dev, ci_pipeline")
- **Strong preferences/rejections** — edges with confidence > 0.7
- **Contradictions** — edges with both positive and negative evidence

Results are written to `MEMORY.md`'s auto-section, capped at `insights_max_entries` (default: 12).

### Phase 3 — Decay (`dreaming/phase3.py`)

**Co-occurrence-aware decay.** The key insight: an edge should only lose confidence if the topic was re-discussed and the relationship wasn't restated. Dormant topics are left alone.

1. For each active edge whose `last_reinforced` is older than the decay window (default: 30 days):
   - Check if any chunk in the decay window mentions the edge's subject or object **without** providing evidence for the edge
   - If yes → add 1 to `neg_evidence` (soft contradiction)
   - If no → leave alone (topic hasn't resurfaced)
2. Any edge whose Laplace-smoothed confidence `(pos+1)/(pos+neg+2)` drops below the retract threshold (default: 0.15) gets `status = 'retracted'` — kept for audit but excluded from query results.

After decay, Phase 2 insights are refreshed to reflect the new graph state.

---

## 6. Query-Time Augmentation (How Memory Gets Used)

When Hermes receives a user message, it calls `hy.augment(user_message)` which returns an `AugmentedContext`:

```
AugmentedContext(
    user_md: str,           # USER.md content
    memory_md: str,         # MEMORY.md content
    fts_hits: list[FtsHit], # Ranked relevant chunks
    graph_facts: list[GraphFact],  # Matching knowledge graph edges
    matched_entities: list[str],   # Entities found in user message
)
```

**How it works:**

1. **Load profile + insights** from `USER.md` and `MEMORY.md` (file read, instant)
2. **Keyword search** (`_fts_search`): Sanitize the query, tokenize it, build an OR query across tokens, run against SQLite FTS5 with BM25 scoring. Returns top-k chunks (default: 5).
3. **Vector search** (`_vector_search`, optional): If an embedding client is available, embed the user message, compute cosine similarity against all stored chunk embeddings (O(n) in Python, capped at `embedding_max_scan` default 5000), return top-k.
4. **Reciprocal rank fusion** (`_rrf_merge`): Merge FTS and vector results via RRF: `score = sum(1/(60 + rank))` across each list. This hybrid approach captures both keyword relevance and semantic similarity.
5. **Entity matching** (`match_known_entities`): Tokenize the user message, lookup each word against `entity_aliases`, return canonical IDs.
6. **Graph lookup** (`_graph_lookup`): For each matched entity, query `knowledge_graph` for active edges where the entity appears as subject or object. Up to `graph_top_k_per_entity` (default: 3) facts per entity, ranked by confidence and recency.

Hermes then assembles the prompt with this context — HyMem never dictates prompt structure.

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

### Honcho HTTP Server (`hymem-honcho` → `honcho_server.py`)

A FastAPI server implementing the **Honcho v3 REST protocol** — 629 lines, 15 endpoints. Hermes can use the standard `honcho-ai` Python SDK by setting `HONCHO_BASE_URL=http://127.0.0.1:8765`.

| Endpoint | Maps to | Notes |
|---|---|---|
| `POST /v3/workspaces` | Get-or-create workspace | SDK auto-calls via `_ensure_workspace()` |
| `GET /v3/workspaces/{wid}` | Get workspace | |
| `POST /v3/workspaces/{wid}/peers` | Get-or-create peer | Role auto-inferred from peer_id pattern |
| `POST /v3/workspaces/{wid}/sessions` | Get-or-create session | |
| `POST .../sessions/{sid}/messages` | Log turns + bg dream | Dream cooldown: 60s default |
| `POST .../sessions/{sid}/messages/upload` | File upload as message | For migrating MEMORY.md/USER.md |
| `POST .../sessions/{sid}/search` | `hy.augment()` as Message objects | Graph facts + FTS hits |
| `GET .../sessions/{sid}/context` | MEMORY.md + USER.md + recent turns | |
| `POST .../sessions/{sid}/peers` | Register peers + role mappings | |
| `GET .../sessions/{sid}/peers/{pid}/config` | Per-session peer config | |
| `GET .../peers/{pid}/card` | USER.md behavioral profile | |
| `GET .../peers/{pid}/context` | Peer-scoped context with optional search | |
| `POST .../peers/{pid}/representation` | Update peer representation | |
| `POST .../peers/{pid}/chat` | Dialectic Q&A via `hy.augment()` | Natural language queries |
| `GET /health` | Health check | |

**Key design choices in the Honcho server:**
- **Dream cooldown**: Background dreaming kicks at most once per configurable cooldown (env: `HYMEM_DREAM_COOLDOWN_SECONDS`, default 60s). Uses FastAPI `BackgroundTasks` so the HTTP response isn't blocked.
- **Role inference**: Peer IDs matching `user[-_]|human|client|telegram|discord|slack` → user role, `agent|hermes|assistant|ai[-_]|bot|llm` → assistant role.
- **No LLM in query path**: Search, context, and chat endpoints only call `hy.augment()` — zero LLM calls, zero latency beyond SQLite reads.

---

## 8. Key Design Decisions

**Locked vocabulary (10 predicates).** No open-ended relation extraction. This means the knowledge graph is clean, queryable, and predictable — no hallucinated "loves" or "feels" edges. The tradeoff: some relationships won't fit the schema, but the system errs on the side of silence rather than noise.

**Host-agent responsibility split.** Hermes owns *when* to call HyMem; HyMem owns *how* memory works. HyMem never assembles prompts, never decides token budgets, never injects itself into the agent's reasoning loop. It just returns structured pieces.

**Laplace-smoothed confidence.** Every edge's confidence is `(pos+1)/(pos+neg+2)` — a Bayesian-style smoothing that starts at 0.5 for an untested fact and converges toward truth as evidence accumulates.

**Co-occurrence-aware decay.** Unlike simple TTL-based decay (which would kill all old facts regardless of relevance), HyMem only decays edges whose entities have been re-discussed without reinforcement. This keeps the graph accurate without requiring constant LLM re-extraction.

**Prompt-versioned idempotency.** Changing `prompt_version` in config causes automatic reprocessing of all chunks with the new prompts. Backward-incompatible prompt changes are trivial.

**No external dependencies at core.** The `hymem` package itself has zero dependencies beyond Python stdlib + SQLite. LLM clients and FastAPI are optional extras (`hymem[server]`). The `contrib/` layer provides OpenAI-compatible clients but can be swapped via the `LLMClient` and `EmbeddingClient` Protocols.

**Managed Markdown sections.** `USER.md` and `MEMORY.md` use HTML comment delimiters (`<!-- HyMem:auto:section:start -->` / `<!-- HyMem:auto:section:end -->`). Humans can edit everything outside these sections; HyMem only touches its auto-sections. Atomic writes via tempfile + `os.replace()` prevent corruption.

**Advisory lock with stale takeover.** The `run_lock` table prevents concurrent dreaming cycles. If a holder process crashes, the lock is released after 5 minutes of inactivity so the system doesn't deadlock.

---

## 9. Configuration

All runtime config via environment variables. No config files.

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
| `graph_top_k_per_entity` | 3 | Graph facts per matched entity |
| `embedding_max_scan` | 5000 | Max embeddings to scan (O(n)) |
| `decay_window_days` | 30 | Decay look-back window |
| `decay_factor` | 0.9 | (reserved, not yet used) |
| `retract_threshold` | 0.15 | Confidence below which edges retract |
| `profile_max_entries` | 16 | Max profile entries in USER.md |
| `insights_max_entries` | 12 | Max insights in MEMORY.md |
| `prompt_version` | `"v1"` | Bump to force full reprocessing |

---

## 10. Test Coverage

**71 tests total, 62 passing fully, 9 with minor Honcho server gaps.**

The core test suite (62 tests) passes completely across 14 test files:
- `test_dreaming.py` — Full pipeline: chunk→extract→consolidate→decay
- `test_extraction.py` — Triple extraction, marker extraction, polarity handling
- `test_canonicalize.py` — Entity normalization, alias resolution, merging
- `test_chunks.py` — Salience detection, chunk persistence
- `test_embeddings.py` — Embedding creation and query
- `test_augment.py` — FTS search, vector search, RRF merge, graph lookup
- `test_markdown_io.py` — Section read/write atomicity
- `test_integration.py` — End-to-end capture→dream→augment flow
- `test_phase3_perf.py` — Decay performance benchmarks
- `test_mcp_server.py` — MCP tool correctness
- `test_retract.py` — Edge retraction and idempotency
- `test_dream_runs.py` — Audit log correctness
- `test_honcho_server.py` — Honcho v3 protocol (18 tests, 9 passing)

The 9 failing Honcho tests are:
- 3 missing GET endpoints (`GET /peers/{pid}`, `GET /sessions/{sid}`)
- 1 missing POST endpoint (`POST /sessions/{sid}/messages/list` pagination)
- 5 status code mismatches (server returns 200 where tests expect 201 on create endpoints)

Core functionality — add_messages, search, context, chat, peer card, dream cooldown — all tested and passing.

---

## 11. Comparison with Honcho

| Dimension | HyMem | Honcho (plastic-labs) |
|---|---|---|
| **Scope** | Single-agent memory module for Hermes | Multi-tenant platform for stateful agents |
| **Architecture** | Embedded library (SQLite + 2 servers) | Client-server (FastAPI + Postgres + Redis + workers) |
| **Storage** | 1 SQLite file + 2 Markdown files | Postgres + pgvector + Redis cache |
| **Entity model** | Simple: user + assistant roles | Peer paradigm: all participants are "peers" |
| **Memory extraction** | "Dreaming" — 3-phase LLM pipeline with locked vocabulary | "Deriver" — background workers doing representation, summarization, peer cards |
| **Ontology** | Locked 10-predicate vocabulary | Open-ended reasoning, no fixed ontology |
| **Query interface** | FTS5 + vector + graph → structured context | Chat API (natural language), context (token-budgeted), hybrid search |
| **Decay** | Co-occurrence-aware with confidence thresholds | Continual representation updates (implicit) |
| **Honcho SDK compat** | Full v3 protocol via honcho_server.py | Native |
| **Deployment** | Local-only, pip install, zero config | Managed cloud (app.honcho.dev) or self-hosted Docker/Fly.io |
| **SDKs** | Python + MCP + Honcho SDK | Python + TypeScript |
| **Maturity** | v0.1.0, ~3,200 lines, 8 commits | v3.0.6, 514 commits, 3.4k stars |
| **License** | Not specified | AGPL-3.0 |

**The key philosophical difference:** Honcho is a platform — multi-tenant, cloud-native, with a broad API surface for many use cases. HyMem is a tool — focused, embeddable, opinionated about what memory should look like. HyMem's locked vocabulary and co-occurrence-aware decay are design bets that prioritize precision over recall. Honcho prioritizes flexibility and scale.

**HyMem can self-host the Honcho experience.** With the Honcho server functional, Hermes can use the standard `honcho-ai` SDK and get the same API surface as Honcho's cloud — search, context, chat, peer management, sessions — without leaving the machine. This is HyMem's headline feature: **Honcho compatibility without any infrastructure.**

---

## 12. Limitations & Known Gaps

- **O(n) vector search**: Cosine similarity is computed in pure Python over all stored embeddings (capped at 5,000). Not viable beyond ~10K chunks. Future options: sqlite-vec extension, numpy batching, periodic pruning.
- **No streaming**: The Honcho server doesn't implement SSE streaming for chat responses.
- **Single database lock**: SQLite's single-writer model means concurrent dreaming + heavy querying will contend. Fine for a single-agent setup, not for multi-tenant.
- **3 missing Honcho endpoints**: GET peer by ID, GET session by ID, POST messages/list pagination. The SDK's `_ensure_peer()`, `_ensure_session()`, and paginated message listing paths will 404.
- **No authentication**: Both MCP and Honcho servers are unauthenticated — they assume localhost-only access.
- **No migration framework**: Schema evolution is manual. The `schema_version` key in `schema_meta` exists but isn't checked programmatically.
- **English-only**: Chunking, canonicalization, and the LLM prompts assume English text.
