# HyMem — Definitive Walkthrough

---

## TL;DR / Executive Summary

**HyMem** is a local-first, embedded memory system for AI agents. It gives the **Hermes** agent persistent memory across conversations by extracting a structured SQLite knowledge graph from chat logs during idle "dreaming" cycles, then making that knowledge queryable at conversation time via keyword search, vector search, semantic graph search, and entity lookup — plus a working-memory tier of recent raw turns so facts from the current session are recallable before they're dreamed. It also auto-maintains two Markdown files (`MEMORY.md` and `USER.md`) that the agent reads before each conversation, and — with aggregation enabled — a standing whole-store digest (`HyMem.digest()`, the RAPTOR root) answering "what do you know about this user?" for system-prompt injection.

**No cloud, no Postgres, no 500MB Docker images.** One SQLite file, two Markdown files, and a Python library. Query-time retrieval defaults to LLM-free — FTS5 + deterministic local feature vectors + graph traversal. An optional hybrid reranker (cross-encoder *or* the configured LLM) can be enabled to break ties when keyword and semantic search disagree, gated by an ambiguity threshold so the LLM hot path stays the exception, not the rule.

**Two deployment modes:** an MCP tools server for direct agent integration, and a **Honcho-compatible HTTP subset** covering the pinned `honcho-ai` SDK calls Hermes uses. It is an interoperability layer, not a drop-in implementation of every Honcho cloud behavior.

**~15,000 lines of Python**, zero npm, zero Docker required.

**LongMemEval evidence is tracked, not advertised as a current leaderboard score.** A historical full-S local-protocol development run recorded 70.0%, but it predates the strict pinned evaluator/dataset/manifest path and is neither an official-comparable result nor a current headline. The present harness records label-free routing, failures, provider identities, indexing health, and the exact denominator; no replacement score is claimed until that protocol is rerun. See [§11](#11-benchmark-evidence-longmemeval).

---

## Quickstart

`pip install` HyMem, point it at your model provider with environment
variables, and run a server — no config files. With no embedding settings,
this quickstart uses the deterministic no-network lexical feature-hash backend.
For network-backed semantic embeddings, apply one complete environment block
below **before** running the doctor or launching either server.

```bash
pip install 'hymem[server]'

export HYMEM_LLM_API_KEY=sk-...        # extraction LLM (DeepSeek/OpenAI/...)
export HYMEM_ROOT=~/.hermes            # optional — SQLite + Markdown live here
# RAPTOR aggregation + its standing digest are enabled by default.
# Optional opt-out: export HYMEM_AGGREGATION_NODES_ENABLED=false

hymem-doctor      # preflight: verify config before launching
hymem-honcho      # Honcho-compatible HTTP subset on :8765
# or
hymem-server      # MCP tools server
```

`hymem-doctor` reports the resolved provider/model and safe endpoint identity,
probes configured providers, checks that `sqlite-vec` loads and the schema migrates,
and reports embedding model/dimension mismatches in vector shadows. It also
counts producer-incompatible or malformed rows across all six durable embedding
mirrors and reports foreign-key violations, without repairing or deleting them.
This stored-row compatibility check is not a per-row vector recovery audit:
it excludes the historical general embedding cache and does not establish that
all existing source rows have current vectors or valid source proofs. If the
LLM key is missing the servers refuse to start with a clear message rather than
failing deep inside the first request.

The extraction LLM defaults target DeepSeek. Provider keys are origin-bound:
`DEEPSEEK_API_KEY` and `OPENAI_API_KEY` are inherited only by their exact
official HTTPS origins; a custom LLM endpoint must use `HYMEM_LLM_API_KEY` or
an explicit constructor/benchmark-role key. Embeddings are configured
independently: omission selects a deterministic no-network lexical feature
backend; an explicit remote endpoint should use HTTPS and
`HYMEM_EMBEDDING_API_KEY` (the general `OPENAI_API_KEY` is inherited only by
the official `api.openai.com` HTTPS origin). Full inventory in
[§9 Configuration](#9-configuration).

For an authenticated HTTPS embedding service, first obtain its actual API base,
model ID, output dimension, and public deployment revision/tenant from the
operator. This block refuses missing or empty values instead of choosing a model
or dimension for your deployment:

```bash
: "${HYMEM_EMBEDDING_BASE_URL:?Set the actual HTTPS API base, including its route prefix}"
: "${HYMEM_EMBEDDING_API_KEY:?Set the purpose-bound embedding API key}"
: "${HYMEM_EMBEDDING_MODEL:?Set the operator-verified embedding model ID}"
: "${HYMEM_EMBEDDING_DIM:?Set the operator-verified output dimension as a positive integer}"
: "${HYMEM_EMBEDDING_DEPLOYMENT_REVISION:?Set the operator-verified public deployment revision}"
: "${HYMEM_EMBEDDING_DEPLOYMENT_TENANT:?Set the operator-verified public deployment tenant}"
export HYMEM_EMBEDDING_BASE_URL HYMEM_EMBEDDING_API_KEY
export HYMEM_EMBEDDING_MODEL HYMEM_EMBEDDING_DIM
export HYMEM_EMBEDDING_DEPLOYMENT_REVISION HYMEM_EMBEDDING_DEPLOYMENT_TENANT
export HYMEM_EMBEDDING_PIN_DIMENSION=1
```

An embedding service outside the Hermes container can still be local to its
isolated container network. A service name such as `embedding-server` is not
loopback inside Hermes, so keep the HTTP exception explicit (HTTPS is preferred
whenever the service can terminate TLS). For that service **when its API is
served under `/v1`**, use this complete alternative:

```bash
: "${HYMEM_EMBEDDING_MODEL:?Set the operator-verified embedding model ID}"
: "${HYMEM_EMBEDDING_DIM:?Set the operator-verified output dimension as a positive integer}"
: "${HYMEM_EMBEDDING_DEPLOYMENT_REVISION:?Set the operator-verified public deployment revision}"
: "${HYMEM_EMBEDDING_DEPLOYMENT_TENANT:?Set the operator-verified public deployment tenant}"
export HYMEM_EMBEDDING_BASE_URL=http://embedding-server:8766/v1
export HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP=1
export HYMEM_EMBEDDING_MODEL HYMEM_EMBEDDING_DIM
export HYMEM_EMBEDDING_DEPLOYMENT_REVISION HYMEM_EMBEDDING_DEPLOYMENT_TENANT
export HYMEM_EMBEDDING_PIN_DIMENSION=1
# Set HYMEM_EMBEDDING_API_KEY only if that internal service authenticates.
# HyMem otherwise sends a non-secret local dummy; cloud/LLM keys are never reused.
```

Preserve the service's **actual URL route**: the SDK appends `embeddings` to the
configured API base; HyMem does not automatically append `/v1`. For example,
`http://embedding-server:8766/v1` targets `/v1/embeddings`, while the bare origin
targets `/embeddings`. A service mounted at `/proxy/openai` needs that prefix,
not a universal `/v1` suffix. Do not include the final `/embeddings` operation
in the base URL. Clear any stale embedding-specific API key when switching to
an unauthenticated internal service; do not copy a cloud or LLM key into it.

Keep the service's actual `HYMEM_EMBEDDING_MODEL` and `HYMEM_EMBEDDING_DIM`.
Network-backed durable vectors also require `HYMEM_EMBEDDING_PIN_DIMENSION=1`
and the existing, operator-verified **public**
`HYMEM_EMBEDDING_DEPLOYMENT_REVISION` and `HYMEM_EMBEDDING_DEPLOYMENT_TENANT`.
These bind the exact vector producer, not credentials; obtain missing values
from the service operator instead of guessing them or changing the model/dimension
to make a check pass. The HTTP exception does not waive these requirements.

Persist these settings in the environment supplied by the actual launcher to
**both** the Honcho service and MCP process, wherever both are deployed. They
must use the same base route, model, dimension, pin, revision and tenant for the
same vector producer; include the HTTP opt-in and purpose-bound credentials
where applicable. An export in an interactive shell,
or an edited `.env` file that the launcher never loads, does not update an
already-running process. Restart processes when their launcher reloads the
updated environment; if environment values are part of the container's creation
configuration, recreate/redeploy that container with the updated configuration
instead of merely restarting it. Preserve its existing storage mounts and other
settings. Verify the process-visible non-secret settings and run `hymem-doctor`
under the same environment—not just in a different login shell. Doctor's remote
checks issue real provider requests. A rejected endpoint remains `[FAIL]` even
though the server can use the lower-quality lexical fallback; the diagnostic
states the opt-in only for a valid internal HTTP endpoint. Public HTTP is never
enabled by this flag.

The Honcho dream scheduler is **event-driven, not periodic**: ingestion kicks
one bounded cycle, subject to cooldown. A restart, elapsed cooldown, restored
embedding service, or remaining backlog does not itself schedule more cycles.
A single dream does not guarantee that all vectors are regenerated: tiers
have their own eligibility and bounds. Further work needs another qualifying
event or an explicit `hymem_dream`/`HyMem.dream()` call. Check the resulting
status and relevant vector rows; do not infer complete vector recovery from
one successful dream, zero extraction backlog, or doctor preflight alone.

### Embedding-only repair of existing vectors

Use the same verified embedding environment as the service. This command does
not construct an LLM client or require an LLM key. Dry-run is the default: it
opens an **existing, current-schema** database read-only, makes no provider
requests, and never creates, migrates, or reconstructs source data.

```bash
python -m hymem.reembed --json
python -m hymem.reembed --apply --batch-size 16 --max-items 256 --timeout-seconds 60 --json
```

The equivalent `hymem-reembed` console command is available after reinstalling
the updated package; `python -m hymem.reembed` also works directly after a pull.
Use `--db /absolute/path/hymem.sqlite` to select another existing store.
An incomplete remote configuration is refused, never converted to a local
fallback. Intentional local-only apply additionally requires `--allow-local`.
Malformed explicit dimensions and unpinned remote producer declarations fail
closed before repair.

Apply acquires the same renewable lease as dreaming. It repairs only existing
chunk, message, live-edge, proof-valid episode, and current-fact vector mirrors;
it does **not** create missing mirrors or include the historical general
`embedding_cache`. Provider calls are embedding calls and can incur embedding
costs, but no extraction/digest LLM calls occur. Each invocation processes at
most `--max-items` rows through its repair cursor, including healthy and blocked
ones, and embeds at most `--batch-size` items per request. This limit excludes
read-only compatibility audits (which stream the full stored mirrors) and source
proof/collision checks; those remain subject to the cooperative time budget.
Repeat the same bounded command to
resume its producer-specific durable cursor; it resets only at the end of a
sweep. Dry-run always starts at the beginning without moving that cursor.

Exit status is nonzero for incomplete, blocked, or failed work. JSON reports
`health_before`, end-of-sweep `health_after`, per-invocation counters,
`sweep_complete`, and accumulated
`sweep_blocked`; zero means a completed clean/repaired **stored-vector scope**,
not complete memory regeneration. Foreign-key corruption blocks all writes.
Unproven historical rows (including extraction chunks without exact claim-source
manifests, or whose stored render disagrees with those sources) are preserved
and reported blocked. `pending` counts unrepaired candidates observed in this
invocation; `repaired` counts its committed updates. Aggregation vectors
are bound to published material: incompatible, corrupt, or unpublished rows are
reported `rebuild_required`, never relabeled or independently re-embedded.
Those require a normal aggregation/dream rebuild after repairing its inputs.
The time limit is cooperative: shipped transports cap requests to the remaining
budget, and late results are not committed. A failed provider batch retains its
cursor for retry. There is no automatic scheduler or background retry.

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
    │  9 MCP tools │    │  19 HTTP endpoints │
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
├── redaction.py        Best-effort secret/PII scrubbing at the ingest chokepoint
├── session.py          Session lifecycle (open/close), message logging, recent_messages
├── bootstrap.py        Env-var resolution + build_from_env() + shared singleton
├── doctor.py           hymem-doctor — preflight diagnostics (keys, endpoints,
│                         sqlite-vec, schema, embedding-dim drift, canonical drift)
├── server.py           MCP server — 12 tools (capture, log, dream, augment,
│                         ask, profile, digest, alias, retract, add/list/suggest rules)
├── honcho_server.py    Back-compat shim → hymem.honcho
│
├── honcho/             Honcho-compatible HTTP subset
│   ├── app.py          FastAPI routes (pinned-SDK surface + operations) + entry point
│   ├── models.py       Typed Pydantic request models (one per endpoint body)
│   └── adapters.py     Response shaping + request-shape normalization
│
├── core/
│   ├── db.py           SQLite connection management (WAL), schema init, migrations
│   ├── schema.sql      Authoritative SQLite tables, FTS5 virtual tables, and triggers
│   └── markdown_io.py  Read/write HTML-comment-delimited sections in MD files
│
├── dreaming/
│   ├── runner.py       Orchestrates full pipeline with advisory lock
│   ├── scheduler.py    Long-lived daemon thread owning background dream cycles
│   │                   (one forked HyMem connection for its whole lifetime)
│   ├── chunks.py       Regex-based high-salience chunk extraction
│   ├── canonicalize.py Deterministic entity name normalization + aliases +
│   │                   drift detection/repair (find_/repair_canonical_drift)
│   ├── mentions.py     Entity mention indexing for decay calculations
│   ├── temporal.py     Per-message date-mention indexing (temporal_mentions)
│   ├── dates.py        Stdlib-only date extraction primitives for the TR path
│   ├── embeddings.py   Batch embedding of chunks + knowledge-graph edges (JSON + sqlite-vec)
│   ├── phase1.py       Extraction persist + dedup (lock-free embed, same-wave collapse)
│   ├── digest.py       Batched per-session episodes+summary+procedures (one LLM call)
│   ├── phase2.py       Consolidation: markers→profile, graph→MEMORY.md
│   ├── phase3.py       Co-occurrence-aware decay + retraction
│   ├── inference.py    Transitive closure over depends_on edges
│   ├── bitemporal.py   Valid-time stamping: valid_at/invalid_at on edges
│   ├── value_supersession.py  Single-assertion supersession for typed-value
│   │                   edges — a knowledge UPDATE closes the old value without
│   │                   waiting for evidence accumulation
│   ├── user_profile.py Typed personal-fact slots (role/employer/location/…)
│   │                   the tech-domain graph vocabulary never captures
│   ├── episodes.py     LLM-powered episodic memory extraction
│   ├── procedures.py   LLM-powered procedural memory extraction
│   ├── summary.py      LLM-powered session summarization
│   ├── aggregate.py    RAPTOR cross-session aggregation: cluster episodes →
│   │                   fuse → hierarchy levels up to the root standing digest
│   ├── behavioral_dedup.py  Dry-run report of pre-collapse behavioral duplicates
│   └── retention.py    Chunk pruning with graph-aware eviction
│
├── extraction/
│   ├── llm.py          LLMClient Protocol + StubLLMClient (for tests)
│   ├── embeddings.py   EmbeddingClient Protocol + StubEmbeddingClient +
│   │                   exact-producer/dimension/text LRU cache
│   ├── chunk.py        Merged single-call chunk extraction (triples + markers)
│   ├── triples.py      Triple parsing/validation (subject, predicate, object, polarity)
│   ├── markers.py      Behavioral-marker parsing/validation
│   ├── retry.py        Bounded exponential backoff for external API calls
│   └── prompts/        System/user prompts for extraction, episodes, procedures,
│                         summaries, and reranking
│
├── query/
│   ├── augment.py      FTS5 + vector + RRF merge + LLM rerank + hybrid graph
│   │                   ranker (routed + per-candidate fallback branches,
│   │                   token-overlap entity expansion)
│   ├── ask.py          Dialectic endpoint behind HyMem.ask() — renders the
│   │                   retrieval tiers and makes one synthesis LLM call
│   ├── rerank.py       Cross-source rerank after RRF (LLM or cross-encoder
│   │                   backend, ambiguity-gated)
│   ├── count_routing.py Graph-native exact counting for in-domain MR queries
│   ├── intent.py       detect_ability() router — infers MR/TR question-type from
│   │                   the query alone (EN+NL regex, label-free) + AbilitySignal
│   │                   observability, so ability shaping fires in production with
│   │                   no oracle label
│   ├── predicate_routing.py  Keyword → predicate mapping for query expansion
│   ├── conflicts.py    Contradiction detection over the knowledge graph
│   └── entities.py     Token-based entity matching against knowledge graph
│                       (shape filter suppresses one-off gerund objects)
│
└── contrib/
    ├── openai_client.py          OpenAI-compatible LLM client (DeepSeek default)
    └── openai_embedding_client.py OpenAI-compatible embedding client
```

---

## 4. The Data Model

**Conversation storage:**
- `sessions` — session ID + start/end timestamps, independent lossless-coverage
  and resumable digest cursors, an automatic rolling summary, and provenance
  for the compatibility/operator summary; a published-generation marker keeps
  partial episode rebuilds out of retrieval and aggregation
- `messages` — raw turns (user, assistant, system, tool)
- `messages_fts` — FTS5 over raw turns, indexed live at ingest; powers the `message_hits` tier so a turn is recallable before any dream chunks it
- `message_retention_coverage` — immutable, content-hashed proof that each
  source message has an exact canonical JSONL backing artifact; this is the
  only proof that can authorize raw-message pruning

Portable v12 exports are coherent, manifest-backed snapshots. Imports merge
disjoint identities and exact reimports are idempotent; if an existing
session, chunk, or durable proof has the same identity but different canonical
state, the entire import fails closed instead of silently rebinding evidence.
Prompt-scoped extraction attempts and their failure diagnostics are local
operational staging: success clears them, and export/import deliberately resets
them instead of transporting another host's quarantine decision.
Aggregation build health is likewise local and is not exported: material
imports invalidate the destination's clean-build acknowledgement while keeping
its bounded local audit counters, so the destination cannot report an imported
tree healthy until a zero-failure build completes under its active config.
Narrative-fact outcomes, exact source occurrences, revisions, and lifecycle
events round-trip as one authority ledger; pre-v10 fact cursors are rewound
because a numeric message range alone is not provenance.

**Extraction artifacts:**
- `chunks` — purpose-tagged artifacts: selective `extraction` segments plus
  namespaced, exact per-message `coverage` records. Coverage records never run
  through Phase 1, retrieval, or embedding.
- `chunks_fts` — FTS5 virtual table over chunk text (BM25 search)
- `chunk_embeddings` — JSON-encoded embedding vectors
- `processed_chunks` — idempotency tracking per chunk and derived extraction
  contract namespace (SHA-256 of prompt bytes plus acceptance/recovery behavior,
  with the human `prompt_version` retained in the key)
- `chunk_extraction_attempts` — contract-scoped retry/quarantine state with the
  latest safe reason and field-level diagnostics (never raw model output)
- `chunk_extraction_terminal_losses` — explicit, prompt-independent loss state
  for legacy extraction chunks whose exact source manifest cannot be recovered;
  these rows never masquerade as successful `processed_chunks` entries and do
  not consume future extraction budgets
- `coverage_integrity_failures` — one bounded structural health row per
  affected session, keyed to the exact lossless validator configuration. It
  carries only enum reasons/counters/timestamps (never source or exception
  text) and clears after a successful complete walk once a transient cause is
  resolved or persistent artifact damage is repaired/rebuilt. A v38 frontier
  cannot reconstruct the identity of an occurrence if both its raw row and
  proof are destroyed out of band; the immutable SQL guards are the prevention
  boundary for that unsupported state
- `entity_mentions` — inverted index: chunk → canonical entity
- `entity_types` — canonical entity → type classification (language, framework, database, etc.)
- `entity_properties` — free-form key/value attributes per canonical entity (e.g. `language=python`), extracted alongside types
- `temporal_mentions` — per-message date mentions extracted at dream time; feeds the `ability="TR"` chronology
- `embedding_cache` — content-addressed (text, model) → vector cache deduplicating embedding-API calls across chunks/edges/episodes
- `fact_extraction_outcomes` + `fact_extraction_source_occurrences` — one
  immutable, lossless source unit per facts cursor step, including successful
  empty results; malformed/over-cap results durably retry without advancing
- `fact_extraction_revisions` + `narrative_fact_lifecycle` — authoritative
  replay history. Prompt/config bumps re-extract each stored unit on its exact
  original boundaries, retract omitted payloads, and deterministically
  resurrect payloads that return
- `narrative_facts` — the current fact projection. Its FTS and vector readers
  expose only active rows whose complete source/result/lifecycle proof validates

**Knowledge graph:**
- `knowledge_graph` — the materialized current cache of direct and inferred
  `(subject, predicate, object)` rows. It carries evidence counters,
  confidence, status, and one cached `valid_at`/`invalid_at` interval, but is
  not itself the historical authority.
- `kg_evidence` — the immutable, revisioned per-claim ledger (schema v40/41).
  Canonical rows identify the exact source session, message, role, source event
  time, lossless coverage artifact, extraction chunk, and interpretation.
  Re-extraction retires a revision with `superseded_at`; it never overwrites
  the old source claim.
- `kg_edge_lifecycle` + `kg_lifecycle_dependencies` — immutable assertion and
  retraction transitions with causal evidence links. Replaying eligible events
  yields half-open source-valid intervals (`valid_at <= t < invalid_at`) and
  correctly handles assert → retract → reassert without erasing old intervals.
- `edge_embeddings` — JSON-encoded embedding vectors for knowledge graph edges, keyed on triple text so churning derived edges reuse cached vectors
- `entity_aliases` — surface form → canonical entity mapping

**Behavioral profiling:**
- `behavioral_markers` — raw extracted signals (correction, preference, rejection, style)
- `profile_entries` — structured profile with evidence tracking
- `user_profile` — typed personal-fact slots (role, name, employer, location, language, …) with bi-temporal `valid_at`/`invalid_at`; consumed by the digest's verified-facts anchor, `ctx.user_profile`, and `HyMem.profile()`

**Episodic & procedural memory:**
- `episodes` — named, summarized conversation segments with outcomes and
  entities. A bounded replacement walk stages generation-scoped rows beside
  the last complete generation; readers expose only the session's atomically
  published generation (plus standalone NULL-generation rows).
- `episodes_fts` — FTS5 search over published episode titles and summaries;
  staged generations have no postings and therefore cannot perturb BM25 ranks
- `episode_embeddings` — vectors for semantic episode search (`vec_episodes`)
- `procedures` — step-by-step workflows with triggers and involved entities
- `procedures_fts` — FTS5 search over procedure names, descriptions, and steps

**Aggregation & digest (RAPTOR):**
- `aggregation_nodes` — cross-session cluster summaries plus hierarchy levels rolled up to one root **standing digest** (`HyMem.digest()`); levels ≥ 1 never enter query-time retrieval

Every published aggregation is bound to the exact declared LLM producer,
effective request policy, prompts, parser/recovery code, and material algorithm.
Changing that live producer hides the prior publication until a clean build.
With `llm=None`, reads may expose only a publication backed by a durable exact
declaration; this reports historical provenance and is not certification for a
new target producer. Provider aliases and tenant routing can identify only the
declared endpoint/model/transport contract—an undeclared server-side deployment
revision cannot be inferred. Arbitrary endpoint routes are retained only as a
canonical origin plus SHA-256 digest, never raw path text.
- `aggregation_nodes_fts` — FTS5 over node summaries for the additive,
  ability-gated query-time tier (TR-only by default)
- `aggregation_node_embeddings` — cache-keyed summary vectors, so re-dreaming a stable store re-fuses nothing

**Retraction audit:**
- `extraction_feedback` — source-linked records of manual, automatic, and
  dedup corrections. The historical table name is retained for compatibility;
  its values are retention-bounded audit data and never prompt instructions

**Operational:**
- `schema_meta` — schema version guard (see §8)
- `peers` — Honcho peer registry (peer_id → role mapping)
- `run_lock` — advisory mutex for dreaming concurrency
- `dream_runs` — per-cycle audit log
- `token_overlap_index` — persisted token → canonical map backing the overlap-expansion cache (empty = cold start; rebuilt on demand)

---

## 5. The Dreaming Pipeline (How Memory Gets Built)

Dreaming is the offline process that converts raw chat logs into structured knowledge. It's called by Hermes after each conversation (or on a cron-like schedule). The pipeline holds an advisory lock so concurrent runs bail out safely. A file-backed run renews that lease from a dedicated connection about every 30 seconds, including while a provider call is blocked, and also checks/refreshes ownership at session/chunk and logical LLM-call boundaries and between bounded coverage batches; transaction commits are independently ownership-fenced (see §8 *Advisory lock with lease heartbeat*).

Before any selective extraction, every surviving raw turn—user, assistant,
system, or tool, including empty and short turns—is materialized as one exact,
namespaced coverage record. Public ingestion writes the raw turn, artifact,
coverage proof, and coverage cursor in the same transaction. Legacy backfill is
paged in short transactions. These records form the durable session-digest
stream; salience never decides whether a turn is covered or summarized.
This guarantee costs roughly one additional chunk row, one ledger row, and one
canonical copy of the accepted content per message. Coverage artifacts are
intentionally exempt from `max_chunks`; leave raw retention enabled for easiest
auditability, or opt into `message_retention_days` once that duplication matters.
Schema version 52 (`052_source_materialization_cursor.sql`) adds a separate,
configuration-bound source-materialization
acknowledgement. Both Phase-1 source builders must persist every candidate through
the exact coverage frontier captured by that dream before the acknowledgement can
match. A concurrent append therefore belongs to the next frontier instead of being
silently skipped. Deleting any extraction chunk invalidates the acknowledgement in
the same transaction and reopens the source walk; deleting a coverage-only chunk
does not masquerade as an extraction-cache event.

### Phase 1 — Extraction (`dreaming/phase1.py`)

1. **Chunking**: Regex matches and user turns of at least 30 characters enter the high-priority tier. A separately capped baseline tier covers the remaining non-blank user turns, including short facts and contextual confirmations. Both builders read the validated lossless message stream. Each candidate and its exact source manifest are persisted before that candidate can be scheduled, and retention runs only after the complete session walk, so extraction budgets or raw-message pruning cannot strand undiscovered work. Chunks are persisted with a `salience_reason` field.
2. **Entity mention indexing**: Each chunk's text is scanned for known entity surface forms, populating the `entity_mentions` inverted index.
3. **LLM extraction (one merged contract per chunk)**: Each unprocessed chunk is sent to the LLM with a locked-vocabulary prompt that returns one JSON object with both `triples` and `markers`. Every otherwise-successful non-empty terminal unit receives exactly one omission-focused verification over the same source records; the verifier sees a compact list of accepted claim identities and must return only missed supported items. Exact repeats are deterministically deduplicated, while verifier failure, incompleteness, bad citation, or conflict fails the unit atomically. Every terminal primary-empty unit receives exactly one bounded empty-verification pass because an English cue regex cannot certify multilingual recall. A cue-bearing unit that can be split is source-safely recovered through fresh child primary calls, and every terminal child retains that same one-pass empty-check obligation. An unsplittable suspicious unit gets its second look but remains held if that verifier is also empty. A non-empty EMPTY output may trigger only the standard one-shot OMISSION pass; EMPTY-of-EMPTY and OMISSION-of-OMISSION never occur. An ordinary sparse non-empty unit or non-cue empty therefore costs two logical completion calls (normally two provider requests, at most six with the shipped retry policy), up from one before completeness checking. Predictable large inputs are pre-partitioned, while incomplete, truncated, saturated, or suspicious-empty replies get bounded source-safe subdivision/verification calls. A source-backed right-hand prose fragment receives an exact, separately labelled preceding window of at most 320 characters when a sentence or paragraph cut is chosen, so a claim spanning the boundary remains visible. The window applies only to the first 320 characters on the right, adds no calls, and is interpretation context rather than independent evidence; recursive splits reconstruct it from contiguous inherited bytes, while exact duplicate claims coalesce and polarity conflicts fail atomically. No branch is published unless every branch completes. The default 100,000-character accepted message pre-partitions into at most 32 initial leaves, which cost 64 calls when no further recovery is needed. A single binary recovery tree with 32 terminal leaves costs 95 calls (31 internal attempts plus 32 primary/verification pairs); arbitrary combinations of prepartitioning and later recovery are governed by the authoritative global 96-call counter and fail atomically if they need more. The shipped client has one explicit three-attempt retry layer and disables SDK retries, so the per-chunk hard ceiling remains 288 actual HTTP requests. Input beyond the fixed envelope fails atomically with a durable `resource_limit` diagnostic instead of being truncated or partially published. The combined prompt carries the full triple ruleset and the marker ruleset:
   - **Triples**: `{subject, predicate, object, polarity}` where predicate must be one of 22: the 18 technical predicates `uses`, `depends_on`, `prefers`, `rejects`, `avoids`, `replaces`, `conflicts_with`, `deploys_to`, `part_of`, `equivalent_to`, `implements`, `contains`, `configured_with`, `requires_version`, `runs_on`, `connects_to`, `generates`, and `tested_by`, plus the four personal-memory predicates `owns`, `located_in`, `participates_in`, and `has_attribute`. Polarity is +1 (assertion) or -1 (negation/retraction). Optional fields: `value_text`, `value_numeric`, `value_unit`, `temporal_scope`. The prompt explicitly authorises people, teams, projects, and codebases as subjects/objects, with a worked linking example — `"Atta is working on MedFlow"` → `(atta, part_of, medflow)` — so identity↔artifact relationships land as 1-hop graph edges instead of sibling canonicals that only a fuzzy text match could connect.
   - **Markers**: `{kind, statement}` where kind is one of: `correction`, `preference`, `rejection`, `style`. Only explicit behavioral signals — no mood/emotion inference.
   - **Entity types**: LLM also infers entity type labels (language, framework, database, service, tool, etc.) for query expansion.
4. **Retraction isolation**: Manual, automatic, and dedup corrections remain in
   `extraction_feedback` as bounded, source-linked audit evidence. Phase 1 never
   reads those model/data-derived values into a system prompt: doing so would
   leak corrections across sessions, suppress a later valid reassertion, and
   make prompt/cache identity depend on mutable store state.
5. **Entity canonicalization**: Surface forms (e.g., "Postgres", "PostgreSQL", "postgresql") are normalized via Unicode folding, CamelCase splitting, article/parenthetical stripping, and an alias table. The v2 Unicode policy uses one idempotent pipeline: it folds Latin accents but retains non-Latin letters and attached marks, so `東京2`/`大阪2` and `नील`/`नाल` remain distinct. Keys longer than 512 characters fail closed, never truncate. This policy and the loaded normalization implementation participate in Phase-1 and fact generations; auxiliary contract v3 preserves v0–v2 historical hash dispatch. Deployment therefore requires retained-source replay for affected generations. Existing aliases are not guessed, merged, or automatically rekeyed, and previously collapsed identities cannot be reconstructed without their original sources; review old aliases and oversized legacy names explicitly. Unicode full-text queries now reach SQLite intact, but its existing `unicode61` tokenizer can ignore Indic/Arabic marks: full-text results remain candidates, not proof of exact entity identity.
6. **Knowledge graph upsert + dedup (lock-free embedding)**: New triples insert edges, repeated triples reinforce evidence counters, negations add negative evidence. Before each chunk's persist transaction opens, dedup candidate vectors are batch-embedded *outside* the write lock (`prepare_dedup_vectors`); the in-transaction path then does pure SQL + in-memory cosine only — no embedding-API call ever runs while the SQLite writer lock is held. A new triple that is a near-duplicate of an existing edge (same predicate, one shared endpoint, lexical-sibling varying endpoint, cosine ≥ threshold) attaches its evidence to that edge instead of minting a sibling. **Same-wave collapse**: because `edge_embeddings` only holds *prior-cycle* vectors, sibling variants minted within the *same* dream are also compared against an in-memory pool of edges created earlier in the cycle (same gates), so a `prompt_version` re-extraction wave can't fan a single preference out into many phrasal-variant edges.
7. **Idempotency**: Each chunk is processed at most once per mechanically derived extraction-contract identity. The identity binds the rendered prompt bytes, strict JSON/item/source-ID acceptance, and bounded split/retry/recovery implementation to the human `prompt_version`. Changing any bound behavior makes legacy cache/attempt/outcome rows stale even if the label was accidentally left unchanged; deliberately bumping the label also forces bounded reprocessing. Prompt-independent terminal source losses cannot be reconstructed by a contract or version change (see §13 for the re-extraction-surge note).

### Inter-Phase Steps (`dreaming/runner.py`)

After chunk extraction per session, one batched digest call produces episodes,
an automatic summary, and procedures from the independent coverage stream. A
persistent message/character cursor makes each bounded window retryable and
idempotent: an oversized message resumes at the exact next character, and the
message-level watermark advances only after its final slice commits. Short and
assistant-only tails therefore reopen the digest even when they create no
salience chunk. Each new window receives the prior automatic summary and must
return the rolled-forward value (or an explicit empty no-op). Automatic text is
stored separately from operator/legacy summaries; Honcho renders the effective
combination without allowing dreaming to overwrite curated text. A prompt or
window-size change rewinds against the durable artifacts, including after raw
messages have been safely pruned. Episode replacement is generation-scoped:
partial rows remain durable for retry but are excluded from FTS, vectors, and
RAPTOR until the complete walk atomically swaps the published marker, so a
failed rebuild cannot mutate or hide the last complete episode set.

Summary, episodes, and the union of procedures from every bounded slice now
publish in the same completion transaction. A complete replacement (including
an empty result) marks omitted digest-owned procedures stale; a forward tail is
additive. Stale procedures are excluded from retrieval, while their confidence
feedback remains available if later re-extracted, until normal stale-retention
expiry. Schema v59 records exact-content digest ownership, and portable v16
preserves it across export/import. Unknown pre-v59 procedures, legacy imports
without that declaration, and manually edited content are not automatically
retired or overwritten—even when a digest emits the same name. These ambiguous
historical rows require explicit review; their origin cannot safely be inferred
from a generated-looking id. The ownership hash binds the declared payload and
published session generation; it is not a proof of LLM entailment or authenticity.

### Phase 2 — Consolidation (`dreaming/phase2.py`)

**Deterministic, no LLM.** Two sub-steps:

**Profile consolidation**: Producer-authorized behavioral markers are promoted into `profile_entries` with exact marker evidence. Repeats reinforce the authoritative projection, while contradictions remain separate instead of silently overwriting. `profile_max_entries` (default: 16) caps reads and `USER.md` rendering; supported ledger rows are retained so a producer replay cannot create an unrecoverable hole. The auto-section is rewritten via `markdown_io.write_section()`.

`USER.md` and `MEMORY.md` are repairable snapshots of committed database state.
Phase 2 commits its database changes before publishing either file, and Phase 3
refreshes insights only after its own commit. A file error can leave one sidecar
stale, but cannot expose a subsequently rolled-back consolidation. The next
ordinary dream repairs both without new ingestion; native profile reads already
render the database-authorized section in memory. Replacements are atomic per
file, not across both files or SQLite. Deadline/lease checks run immediately
before each replacement, but are cooperative checks rather than cross-resource
atomic fencing. Crashes may leave an old sidecar or temporary file; managed
sections are reconstructed on retry and manual sections are preserved.

Direct `consolidate_profile` calls inside a caller-owned transaction only
consolidate the database: call `publish_profile` after committing, or allow the
next dream to repair the file. Standalone calls commit their own transaction
before publishing. Profile materialization policy v5 causes retained v4 markers
to replay; old per-marker decisions are replaced during replay, not kept in an
append-only decision history.

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

**Bi-temporal lifecycle** (`dreaming/bitemporal.py`, schema v40–42): every
canonical positive claim appends an assertion at its normalized source-message
event time; phase-3, value-supersession, and manual decisions append causal
close events. Replaying the immutable ledger produces half-open validity
intervals and deterministic equal-time ordering. Public temporal readers never
substitute ingestion-time `first_seen` when source-valid evidence is missing.
Evidence `extracted_at`/`published_at`/`superseded_at` and lifecycle
`created_at` form the separate authority clock. Canonical assertions obey
`extracted_at <= lifecycle.created_at <= published_at <= superseded_at` (when
retired); `published_at` is written once only after the entire extraction chunk
succeeds. An explicit historical cutoff therefore uses that immutable first
publication boundary even after unchanged re-extraction. The no-cutoff path
also requires the latest matching observation and whole-chunk outcome to remain
coherent, so staged, orphaned, future, or superseded extraction state cannot
appear as current memory.
Canonical entity merges are not transaction-versioned: historical authority is
projected onto the **current canonical topology**, not presented as a literal
old graph shape.

Caller-supplied message `created_at` is the occurrence/source-valid timestamp,
not a way to schedule a fact to become effective later. It must be strict
ISO-8601 and cannot lead HyMem's observation clock by more than the shared
300-second producer-skew allowance. Single and batched ingestion reject a bad
clock atomically before raw or coverage rows land; lifecycle persistence and
portable v7 import enforce the same event-versus-recorded-time invariant. This
keeps the single cached current interval sound instead of folding future
transitions into today's state.

**Transitive inference** (`inference.py`): After decay, computes transitive closure for derived edges via two rules. (1) `A depends_on B, B depends_on C → A depends_on C` (BFS over the depends_on subgraph). (2) `A uses B, B depends_on C → A depends_on C` (one-hop cross-predicate, folded into `depends_on` so the predicate vocabulary stays stable and no schema migration is required). All derived edges are marked `derived=1` with confidence equal to the product of the source edges' smoothed confidences; previously-derived edges are wiped each cycle so the closure refreshes from scratch. Edges below the retract threshold are filtered out.

After inference, Phase 2 insights are refreshed to reflect the new graph state, and old unreferenced chunks are pruned via `retention.py`.

**Edge embedding** (`embeddings.py`): Once the graph has settled, every current
direct edge is embedded as `"{subject} {predicate} {object}"` text into
`edge_embeddings` and the `vec_edges` sqlite-vec table, so semantic retrieval
cannot reintroduce inferred, invalid, or negative-dominant rows excluded by the
public graph contract. The cache is keyed on triple text, not edge id. Only
genuinely new triple texts cost an embedding call, and Phase 1's lock-free dedup
pass has usually pre-warmed the content-addressed cache. Skipped entirely when
no embedding client is configured.

---

## 6. Query-Time Augmentation (How Memory Gets Used)

When Hermes receives a user message, it calls `hy.augment(user_message, session_id=...)` which returns an `AugmentedContext`:

```
AugmentedContext(
    user_md: str,              # USER.md content
    memory_md: str,            # MEMORY.md content
    fts_hits: list[FtsHit],    # Ranked relevant chunks
    message_hits: list[MessageHit],   # Raw-message lexical + semantic hits
    total_message_matches: int,       # Candidate count for ability="MR" (see below)
    graph_facts: list[GraphFact],     # Ranked knowledge graph edges (see below)
    facts: list[FactHit],             # Current exact-source narrative facts
    temporal_events: list[TemporalEvent], # Date-ordered chronology for ability="TR"
    episodes: list[EpisodeHit],       # Matching episodes
    procedures: list[ProcedureHit],   # Matching procedures
    matched_entities: list[str],      # Entities found in user message
    recent_turns: list[Message],      # Working-memory tier (see below)
    detected_ability: str | None,     # MR/TR inferred by detect_ability when the host passes no label
    detected_rule: str | None,        # which router rule fired (observability — see §3 intent.py)
)
```

**Raw-message tier (`message_hits`).** Alongside chunk retrieval, `augment()`
combines FTS5 with semantic scoring over exact, durable user/assistant message
occurrences. Each occurrence is embedded after its lossless coverage artifact
commits, so it remains recallable after opt-in raw pruning and can recover
paraphrases with no shared keyword. Peer/workspace/session scope is applied to
the validated coverage parent, not cached vector metadata. Lexical winners are
preserved when the local fallback's lower-quality vector arm adds candidates.
Hits remain separate from `fts_hits` (different granularity) and carry
`created_at`/`message_id`. Knob: `message_fts_top_k` (default 5, 0 disables).

**Narrative-fact tier (`facts`).** Extraction and retrieval ship enabled
together. Each bounded call consumes an exact lossless source slice; oversized
turns resume by character offset, valid `[]` advances, and malformed or lossy
output holds the cursor for adaptive retry. Search validates the committed
outcome chain, exact occurrence ownership, result revisions, and lifecycle
projection before ranking or provider hooks. The native tier is additive, then
joins the same occurrence-aware fusion and finite token packer as other memory;
source-equivalent representations deduplicate rather than crowding the prompt.
Knobs: `facts_extraction_enabled`, `facts_enabled` (both default on), and
`facts_top_k`.

**Ability hints (`augment(..., ability=...)`).** An optional question-type hint (`IE`, `MR`, `TR`, `SUM`, `IF`, `KU`) that shapes *what HyMem returns*, never how the host answers. An explicit host hint always wins; when the host passes none, `augment()` runs the label-free `detect_ability` router (`query/intent.py`) and fills `MR`/`TR` from the query text alone — so ability shaping fires in production without an oracle label, and the result records `detected_ability` + `detected_rule` for observability. Unknown/None leaves the default path byte-for-byte unchanged.

- **`ability="IF"`** (instruction/step recall — "what steps did I take to implement X?") pulls a wider procedure set (`procedure_top_k_if`, default 10) since procedures are the natural fit.
- **`ability="MR"`** ("how many X across all my requests?") is an **opt-in counting path** (default off — set `message_fts_aggregate_cap > 0`). LLMs count poorly across a long context, so HyMem does the deterministic part: it restricts to **user** turns (assistant echoes would double-count actions), drops query stopwords (EN + NL), collapses literal restatements (conservative dedup that never merges turns differing by a number or entity — so distinct events aren't under-counted), and returns the **distinct count** in `total_message_matches` plus those turns in `message_hits` as evidence. The count is a *candidate* — one turn may state several items or none — so the host's LLM verifies it against the turns; it stays exact even when evidence is capped. Crucially the count is **additive** — it layers on top of normal relevance retrieval rather than replacing it — so a mis-routed MR question still gets full context and stays answerable. Off by default because keyword counting only fits *lexical* "how many" questions; semantic ones ("how many different ways…") need the answering LLM.
- **`ability="TR"`** (temporal reasoning — "what did I do *before* X?", "how long ago?") builds a date-ordered chronology in `temporal_events` from dates extracted at dream time, cited direct graph assertions, and session-date anchors on retrieved turns. Graph events use lifecycle-derived `valid_at` plus the terminal current citation's optional scope; retired scopes and ingestion-time `first_seen` never supply a graph event date. Uncited, inferred, invalid, and negative-dominant rows are omitted from this chronology.

The host assembles these pieces into its prompt and **decides ordering** — for single-conversation *task-recall* questions, `message_hits`/`procedures` should outrank `graph_facts`, whose cross-session tool/preference facts can otherwise crowd the context.

**Working-memory tier (`recent_turns`).** All the fields above are built from *dreaming artifacts* (chunks, embeddings, graph) — so a fact stated this session is invisible to `augment()` until a dream runs. When called with a `session_id`, `augment()` also returns the last `working_memory_turns` (default 10) raw turns of that session, oldest→newest, so within-session facts are recallable *before* any dream has consolidated them. Omitting `session_id` leaves `recent_turns` empty (unchanged legacy behavior). The turns are already secret-redacted at ingest (see §8), so they are safe to surface.

Each default `GraphFact` is a current, direct observation and carries a stable
`edge_id`, source-valid `valid_at`/`invalid_at`, confidence/evidence counters,
`derived=False`, final score, and `why_retrieved` reason codes. It also carries
up to five deterministic, current-positive citations from the terminal open
interval. A citation includes its stable evidence id, real source
role/session/message, source event and creation times, coverage and extraction
chunk ids, revision authority flags, and optional temporal scope. Retired and
negative evidence never appears. Legacy/manual facts remain available to native
`augment()` with `citations=[]`; renderers say `source unavailable` rather than
inventing an author. Honcho Message-shaped routes omit graph facts until the
ledger can supply an exact originating peer id; source role alone is never
masqueraded as peer identity.

**Public graph read contracts:**

- `hy.timeline(entity)` returns the earliest **currently live direct** edge per
  predicate, ordered by source-valid `valid_at`. For compatibility each
  `TimelineEntry` retains the actual ingestion clock in `first_seen`; the two
  clocks are separate fields. Inferred rows are never offered because HyMem
  does not persist a truthful derivation-time interval for them.
- `hy.count_relations(...)` is an exact distinct count over current direct
  edges (`active`, open, and positive evidence strictly greater than negative).
  `include_derived=True` is the explicit closure-inclusive count mode; it does
  not claim observation time.
- `hy.facts_at(valid_time, recorded_at=None, entity=None)` replays immutable
  direct-edge lifecycle events and returns the half-open interval containing
  `valid_time`. With no cutoff it uses today's authoritative evidence revisions
  and dependencies. `recorded_at` instead selects revisions/events that existed
  at that authority cutoff; malformed or missing transaction metadata fails
  closed. It does not copy today's confidence counters into a historical
  result. This lifecycle view can be broader than live retrieval: a confidence
  tie remains in history until an actual close event is persisted, while
  augment/timeline/count hide it immediately. Entity merges are not
  transaction-versioned, so every result is projected onto the **current
  canonical topology**.

**How it works:**

1. **Load profile + insights** from `USER.md` and `MEMORY.md` (file read, instant)
2. **Keyword search** (`_fts_search`): Sanitize the query, tokenize it, wrap each token in FTS5-safe quotes, build an OR query, run against SQLite FTS5 with BM25 scoring. Returns top-k chunks (default: 5).
3. **Vector search** (`_vector_search`, optional): One validated query vector is reused by every semantic tier. Retrieval scores the durable JSON vectors whose exact provider/model, dimension, and current content hash match; malformed or stale rows fail closed. sqlite-vec tables are rebuildable acceleration shadows, never retrieval authority. Chunk scans are capped at `embedding_max_scan` (default 5000).
4. **Reciprocal rank fusion** (`_rrf_merge`): Merge FTS and vector results via RRF: `score = sum(1/(60 + rank))` across each list. This hybrid approach captures both keyword relevance and semantic similarity.
5. **LLM reranking** (`_rerank`, optional): When FTS and vector disagree on the #1 result enough to trigger ambiguity (configurable threshold), the LLM scores each candidate's relevance to the query on a 1-5 scale. RRF and LLM scores are combined for final ranking.
6. **Episode search** (`_episode_search`): FTS5 search over episode titles and summaries for the query.
7. **Procedure search** (`_procedure_search`): FTS5 search over procedure names, descriptions, and steps.
8. **Entity matching** (`match_known_entities`): Tokenize the user message (including 2-3 word n-grams), normalize each, and look them up against aliases and canonical names that participate in a current direct edge. The object-canonical branch additionally applies a live-direct *shape filter* — an object must also be a subject, have an `entity_types` record, or occur as an object more than once. Retracted, active-but-invalid, negative-dominant, and derived-only names cannot suppress semantic fallback or trigger rules.
9. **Entity expansion** — two complementary fuzzy-link layers:
    - **Type expansion** (`_expand_entities_by_type`): for each matched entity, find other entities of the same type (e.g., if user mentions `uv`, also surface `pip` and `poetry` since they're all `package_manager`). Records the type for the `entity_type:` reason code.
    - **Token-overlap expansion** (`_expand_entities_by_token_overlap`): for each multi-segment matched canonical (e.g. `atta_van_westreenen`), look up other canonicals that share a *rare* underscore-segment as a complete token. Common tokens (`system`, `data`, … — anything appearing in more than `graph_token_overlap_threshold` canonicals) are dropped as noise. Closes the gap where the LLM extracted sibling canonicals (`atta_van_westreenen` and `atta_projects`) without a linking edge. Edges anchored *only* via this fuzzy link score at `graph_token_overlap_weight × confidence × recency` and carry `fallback:entity_anchored:overlap` + `overlap_via:{token}` reason codes, so they surface without out-ranking direct entity matches.
10. **Predicate routing** (`predicate_routing.py`): Map natural-language cues in the query to typed predicates — "what technologies" → `uses`/`runs_on`, "depends on" → `depends_on`, "configured" → `configured_with`, etc. Routing only ever *adds* signal: matching predicates get a score boost and a `predicate:` reason code, but no edge is ever filtered out.
11. **Hybrid graph ranker** (`_graph_lookup`): Gathers candidate edges from three sources — entity-anchored lookup (the entity appears as subject/object), semantic KNN against `vec_edges` (Python cosine over `edge_embeddings` when sqlite-vec is unavailable), and predicate-routed lookup. Scoring branches on whether the query routed any predicate.

    **Routed path** — at least one predicate keyword matched:
    ```
    score = confidence × recency_weight × (semantic_score if > 0 else 1.0) × predicate_boost
    ```
    All three sources merge; predicate-matched edges get the `graph_predicate_boost` multiplier (default 1.5×).

    **Fallback path** — no predicate routed. Source 2 KNN is *skipped when entities matched* (edge-level embeddings over `"subject predicate object"` strings are short and noisy enough that they crowd out a deterministic entity hit), and each candidate scores by its own signal:
    - `fallback:entity_anchored` — entity-matched edges score `confidence × recency_weight`.
    - `fallback:entity_anchored:overlap` — same, but the edge was reached *only* via token-overlap expansion (no direct or type-expanded anchor): `graph_token_overlap_weight × confidence × recency_weight`. The triggering token is surfaced as `overlap_via:{token}`.
    - `fallback:semantic` — no entity matched, embedder present, KNN returned candidates: `semantic_score × confidence × recency_weight`.
    - `fallback:recency` — nothing else fired (e.g. dreaming hasn't embedded yet): `confidence × recency_weight` over a recent-edges seed so something graph-shaped is still shown.

    `recency_weight = exp(-days_since_last_seen / graph_recency_half_life_days)`. The top `graph_top_k` (default: 8) edges are returned, each tagged with its branch in `why_retrieved` alongside accumulated codes (`semantic_X.XX`, `predicate:p`, `entity_type:t`, `recency_Nd`, `entity_match`).

Hermes then assembles the prompt with this context — HyMem never dictates prompt structure.

### Contradiction detection (`conflicts.py`)

Separately from `augment()`, `hy.conflicts()` scans the knowledge graph for contradictions and returns a list of `Conflict` objects. Two kinds are surfaced:

- **competing_object** — the same subject pointing at different objects under a mutually-exclusive predicate (e.g. `atta [prefers] english` vs `atta [prefers] dutch`).
- **opposing_predicate** — the same subject/object pair joined by an opposing predicate pair (e.g. `team [prefers] docker` vs `team [rejects] docker`).

It's pure SQL over the existing schema — no LLM call — and ignores retracted and derived edges.

---

## 7. The Two Server Modes

### MCP Server (`hymem-server` → `server.py`)

Exposes 9 tools via the Model Context Protocol:

| Tool | Purpose |
|---|---|
| `hymem_capture` | Log a full conversation as JSON array + optionally dream (preferred method) |
| `hymem_log` | Log one turn at a time (fallback) |
| `hymem_dream` | Run a dreaming cycle manually |
| `hymem_augment` | Retrieve graph facts + FTS context for a message |
| `hymem_ask` | Dialectic Q&A — same retrieval, plus one LLM call that synthesizes a grounded answer (quotes values/dates, states both sides of a contradiction, says plainly when memory has no answer) |
| `hymem_profile` | Return USER.md + MEMORY.md |
| `hymem_digest` | Standing whole-store digest (RAPTOR root) with coverage + generated-at footer |
| `hymem_alias` | Register surface-form→canonical mapping |
| `hymem_retract` | Retract a wrongly extracted edge |

### Honcho HTTP Server (`hymem-honcho` → `hymem/honcho/`)

A FastAPI server implementing the **Honcho v3 subset used by Hermes**, plus a legacy GET representation alias and two HyMem-native operational routes (`/health`, `/dream-status`). It is tested against the pinned `honcho-ai` Python SDK; set `HONCHO_BASE_URL=http://127.0.0.1:8765` to use it. Unlisted Honcho cloud features are not implied.

The server is a small package, not a monolith: `models.py` holds the typed Pydantic request models (so an SDK shape mismatch is a clean 422, not an `AttributeError` deep in a handler), `adapters.py` owns all response shaping and request-shape normalization (one place that knows "what shape the SDK expects"), and `app.py` holds the routes. The pinned `honcho-ai` SDK is exercised end-to-end against a live server in `test_honcho_contract.py`.

| Endpoint | Maps to | Notes |
|---|---|---|
| `POST /v3/workspaces` | Get-or-create workspace | SDK auto-calls via `_ensure_workspace()` |
| `GET /v3/workspaces/{wid}` | Get workspace | |
| `POST /v3/workspaces/{wid}/peers` | Get-or-create peer | Role auto-inferred from peer_id pattern |
| `GET /v3/workspaces/{wid}/peers/{pid}` | Get peer by ID | Returns role + metadata |
| `POST /v3/workspaces/{wid}/sessions` | Get-or-create session | Registers peers from `peers`; accepts legacy `peer_names` |
| `GET /v3/workspaces/{wid}/sessions/{sid}` | Get session by ID | Returns is_active, metadata |
| `POST .../sessions/{sid}/messages` | Log turns + bg dream | Dream cooldown: 60s default |
| `POST .../sessions/{sid}/messages/upload` | File upload as message | For migrating MEMORY.md/USER.md |
| `POST .../sessions/{sid}/messages/list` | Paginated message listing | page + size, returns total/pages |
| `POST .../sessions/{sid}/search` | `hy.augment()` as Message objects | FTS/raw hits; graph facts only once exact originating peer identity is available |
| `GET .../sessions/{sid}/context` | Exact recent turns + scoped summary | Optional directional representation and card with `peer_target`; representation search/`limit_to_session` do not rewrite the global directional card |
| `POST .../sessions/{sid}/peers` | Register peers + role mappings | |
| `GET .../sessions/{sid}/peers/{pid}/config` | Per-session peer config | |
| `POST .../peers/{pid}/search` | Peer-authored search results | Exact workspace/peer provenance required |
| `GET .../peers/{pid}/card` | Self or directional representation | Optional `target` selects the observed peer; authorized shared-session evidence only |
| `GET .../peers/{pid}/context` | Directional peer context | Path peer is observer; optional `target` is observed peer |
| `POST .../peers/{pid}/representation` | Read directional representation | SDK contract is a POST read; optional session narrows scope |
| `GET .../peers/{pid}/representation` | Legacy representation read | Compatibility alias for older direct callers |
| `POST .../peers/{pid}/chat` | Scoped dialectic Q&A | Bounded iterative reasoning, deterministic fallback, JSON or SDK-compatible SSE |
| `GET /v3/workspaces/{wid}/conflicts` | Unsupported legacy guard | Returns 501; native graph conflicts are not workspace-partitioned |
| `GET /health` | Health check | |
| `GET /dream-status` | `hy.dream_status()` — indexing health/backlog | Current `hymem-dream-status-v7` snapshot: all durable pending/malformed/quarantine/loss classes, bounded coverage/aggregation health, prompt/config/producer identity, `in_progress`, last run |

`hymem-dream-status-v3` is a completion contract, not just a progress counter.
The API composes its status from one coherent SQLite read snapshot. Consumers that
claim completion require every defined blocking count and boolean gate, reconcile
the bounded terminal-loss and coverage details, and fail closed on a missing schema
marker or an unclassified health-like extension. In particular, `pending_chunks=0`
alone does not establish that source materialization, digest, profile, fact,
aggregation, quarantine, malformed-state, and coverage obligations are healthy.
Strict benchmark status extends this same snapshot with embedding-mirror backlog.

**Key design choices in the Honcho server:**
- **Event-driven dream cooldown**: Ingestion calls `DreamScheduler.kick()`; a long-lived daemon waits for kicks and spaces cycle starts by `HYMEM_DREAM_COOLDOWN_SECONDS` (default 60s). The cooldown is not a periodic timer, and a completed cycle does not automatically drain remaining work.
- **Background dreaming on a forked connection**: The scheduler owns one `HyMem.fork()` for its lifetime, with a separate SQLite connection and shared live LLM/embedding clients. It invalidates the foreground query caches after a cycle; SQLite still serializes writers.
- **Batched ingestion**: `add_messages` logs a whole batch under one transaction via `HyMem.log_messages()` rather than one `BEGIN IMMEDIATE` per turn.
- **Role inference**: Peer IDs matching `user[-_]|human|client|telegram|discord|slack` → user role, `agent|hermes|assistant|ai[-_]|bot|llm` → assistant role.
- **Bounded query and reasoning paths**: Search/context retrieval remains deterministic unless an optional reranker fires. Chat may additionally call the configured LLM through a bounded evidence-expansion loop; provider failure or unusable output falls back to deterministic, provenance-grounded text.
- **Explainability and provenance stay attached**: native graph results and
  `/chat` output carry stable edge/time fields, rank reasons, and bounded exact
  source workspace/peer/session/message/event citations. Derived graph claims
  are exposed only when those citations resolve to a validated exact source;
  ambiguous or corrupt ownership fails closed.

### Standing digest and the Honcho tenant boundary

The RAPTOR root, `MEMORY.md`, `USER.md`, profiles, and unowned rules are process-global artifacts. They remain available through native HyMem/MCP integration, but are deliberately excluded from Honcho peer representations and dialectic answers because they have no workspace-and-peer ownership proof. Honcho reads use directional `(observer → target)` collections built only from authorized shared sessions; explicit session scope is validated against workspace ownership and participant observation policy.

---

## 8. Key Design Decisions

**Locked vocabulary (22 predicates).** No open-ended relation extraction. Eighteen predicates cover technical relationships such as usage, dependency, preference, rejection, replacement, deployment, composition, configuration, versioning, runtime, connectivity, generation, testing, and interface conformance; four more cover personal possessions, location, activities, and attributes. This keeps the knowledge graph queryable and predictable rather than accepting hallucinated relation names. The tradeoff: some relationships won't fit the schema, but the system errs on the side of silence rather than noise.

**Host-agent responsibility split.** Hermes owns *when* to call HyMem; HyMem owns *how* memory works. Native retrieval returns structured pieces, while `HyMem.ask()` and Honcho chat deliberately assemble bounded, evidence-only synthesis prompts. HyMem never injects itself into an unrelated host prompt.

**Laplace-smoothed confidence.** Every edge's confidence is `(pos+1)/(pos+neg+2)` — a Bayesian-style smoothing that starts at 0.5 for an untested fact and converges toward truth as evidence accumulates.

**Co-occurrence-aware decay.** Unlike simple TTL-based decay (which would kill all old facts regardless of relevance), HyMem only decays edges whose entities have been re-discussed without reinforcement. This keeps the graph accurate without requiring constant LLM re-extraction.

**Transitive inference.** After each dreaming cycle, derived edges are computed via two rules: (1) BFS over `depends_on` chains (`A depends_on B, B depends_on C → A depends_on C`), and (2) a one-hop cross-predicate hop (`A uses B, B depends_on C → A depends_on C`). Both emit `depends_on` edges so the predicate vocabulary stays stable — no schema change needed for the cross-predicate case. Derived edges are marked `derived=1`, confidence is the product of source-edge confidences, and the whole derived set is wiped and recomputed each cycle so the closure stays consistent.

**Semantic, explainable graph retrieval.** The knowledge graph isn't just keyword-matched — edges are embedded during dreaming, so `augment()` ranks them against the query. When a predicate routes, scoring is `semantic × confidence × recency × predicate_boost`; otherwise a per-candidate fallback fires (`fallback:entity_anchored`, `fallback:semantic`, or `fallback:recency`) that won't let noisy edge-level KNN drown out a deterministic entity hit. Every returned fact carries `why_retrieved` reason codes derived directly from whichever branch fired, so the agent sees *why* a fact was surfaced without a second LLM call. With no embedding client the ranker degrades gracefully to confidence-and-recency ordering.

**Retraction audit, not prompt feedback.** When an edge is retracted, its source
snippet and triple may be stored in `extraction_feedback` for bounded audit.
Neither the runner nor the standalone/combined prompt builders accept those
values as instructions. A later source may therefore reassert the same fact on
its own evidence, and every Phase-1 prompt remains deterministic and covered by
the derived extraction-contract identity.

**Contract-versioned idempotency.** Phase-1 cache, retry, observation, and outcome rows use a derived contract namespace covering the actual prompt bytes and executable acceptance/recovery path. Changing `prompt_version` still deliberately forces reprocessing, while prompt/validator/recovery drift also invalidates old rows without relying on a manual bump. Raw legacy `vNN` rows remain auditable but never satisfy the active cache key. Prompt-independent terminal source losses remain explicit rather than being retried or marked successful; see §13 for the operational limits.

**Schema version guard.** The database schema version is checked against an expected constant. If a newer-schema DB is opened with older code, initialization raises a clear error rather than silently corrupting data.

**Canonical normalization at write, drift check at read.** Every entity name flowing into `entity_aliases` and `knowledge_graph` goes through `canonicalize.normalize()`. If a third-party tool or older code path ever writes around it, `find_canonical_drift()` surfaces the rows where `normalize(v) != v` and `hymem-doctor` flags them. `repair_canonical_drift()` rewrites drifted canonicals with `merge()` semantics — evidence sums on collision so two drifted forms of the same entity collapse cleanly. Auto-repair is opt-in; the doctor only reports, because rewriting a canonical can collide with an existing row and merge decisions belong to the operator.

**WAL by default.** Every connection is opened in WAL mode with `synchronous=NORMAL` and a 10s busy timeout, set in `connect()` so it applies before any migration runs. Background dreaming and live message ingestion run on separate connections without blocking each other or query-time reads — exercised directly by `test_concurrency.py`.

**Zero-config startup.** `bootstrap.build_from_env()` is the single source of truth for environment-variable resolution; both server entry points and `hymem-doctor` build on it. A missing extraction-LLM key fails fast at startup with an actionable message instead of surfacing deep inside the first request. `hymem-doctor` runs the full preflight (keys, endpoint reachability, sqlite-vec, schema migration, embedding-dimension drift, canonical-form drift in `entity_aliases` / `knowledge_graph`) and prints the resolved provider/model/URLs.

**No external dependencies at core.** The `hymem` package itself has zero dependencies beyond Python stdlib + SQLite. LLM clients, FastAPI, and sqlite-vec are optional extras (`hymem[server]`); the pinned `honcho-ai` SDK used by the contract tests is the `hymem[honcho]` extra. The `contrib/` layer provides OpenAI-compatible clients but can be swapped via the `LLMClient` and `EmbeddingClient` Protocols.

Digest, typed-profile, and narrative-fact generations bind their configuration,
loaded implementation, and effective memory producer. Changing these schedules
a lossless-source replay, including after raw messages have been pruned; a first
upgrade from configuration-only stamps also replays once. A stable producer
then returns to zero-call reuse. Maintained clients expose
`memory_producer_declaration()` automatically. Custom routers can implement
that hook returning `Phase1ProducerDeclaration` for their actual memory route;
the Phase-1 declaration is not assumed to identify the same model. Without a
memory declaration, reuse is limited to the exact live client instance and a
restart conservatively replays. Status v7 is required for this stronger
completion guarantee; older clean status receipts are not current evidence.

**Exact-producer LRU-cached embeddings.** Cold queries are dominated by the first `embed([query])` API call. `CachedEmbeddingClient` keys its LRU (default 128 entries) by the validated producer-space key, dimension, and exact input text, so Source 2 KNN and chunk search can safely share one query vector. A display model label alone is never cache authority. Maintained local embeddings carry a deterministic declaration; remote durable use additionally requires a pinned dimension plus operator-declared **public** deployment-revision and tenant labels. Undeclared or adaptive clients bypass the cache on every direct call and cannot select or populate durable mirrors, while high-level HyMem semantic ingestion/query rejects them before provider work. The public attestation labels must not contain credentials: their hashes are commitments, not secrecy, and low-entropy secrets remain guessable.

**Token-overlap index cached on the HyMem instance.** The token→canonicals map used by `_expand_entities_by_token_overlap` is built once per HyMem instance and reused across augments, then invalidated whenever the canonical set could have shifted (`dream()`, `merge_canonical()`, `retract_edge()`). External writers — e.g. a forked HyMem completing a background dream — call `invalidate_query_caches()` on the live instance so the cooldown is observable across both connections. At a few hundred canonicals the scan is sub-millisecond; the cache exists for graphs of tens of thousands of entities where the scan begins to matter.

**Managed Markdown sections.** `USER.md` and `MEMORY.md` use HTML comment delimiters (`<!-- HyMem:auto:section:start -->` / `<!-- HyMem:auto:section:end -->`). Humans can edit everything outside these sections; HyMem only touches its auto-sections. Atomic writes via tempfile + `os.replace()` prevent corruption.

**Advisory lock with lease heartbeat.** The `run_lock` table prevents concurrent dreaming cycles. A holder that crashes is reclaimed after a 2-minute stale-takeover TTL so the system doesn't deadlock. A genuinely slow run (a `prompt_version` re-extraction wave can take minutes) renews the lease about every 30 seconds from a dedicated SQLite connection, including while one provider call is in flight. Foreground holder-guarded checks also run at session/chunk boundaries, before and after each logical LLM completion, and after each bounded coverage batch; every core transaction separately verifies ownership before commit. A zero-row renewal is authoritative ownership loss: the obsolete worker is fenced out of later writes and cannot refresh or release another process's lease. In-memory databases, which cannot be shared cross-process, use the same foreground boundaries without a background connection. Net: a live owner keeps its exact invocation lease; only a crashed/stalled owner ages past the TTL.

**Secret redaction + ingest guards.** Message content is scrubbed for high-confidence secrets (API keys, JWTs, private-key blocks, bearer tokens, credentials-in-URLs, emails) at the single ingest chokepoint (`HyMem._prepare_content`) before it reaches SQLite, so the on-disk store and the chunks derived from it never hold the raw value (toggle: `redact_secrets`, default on). The same chokepoint caps message length (`max_message_chars`) and `augment()` caps query length (`max_query_chars`) so a pathological turn can't bloat the DB or stall extraction. Coverage is byte-exact for the value accepted into SQLite; the separate digest cap never truncates that stored value and instead advances through bounded slices.

**Working memory before dreaming.** `augment(session_id=...)` returns the last N raw turns of the active session (`recent_turns`, `working_memory_turns` default 10) so within-session facts are recallable immediately, without waiting for a dream to consolidate them — see §6.

**One merged extraction contract plus a bounded completeness check.** Phase-1 returns triples and markers from one primary response. Every successful non-empty terminal unit receives exactly one omission-focused pass. Every terminal primary-empty unit receives one direct empty-result check. Cue-bearing inputs may first subdivide through fresh child primary calls, but their terminal children retain the check; an unsplittable suspicious unit remains held when its verifier is also empty. A non-empty EMPTY output may trigger only the standard one-shot OMISSION pass; EMPTY-of-EMPTY and OMISSION-of-OMISSION never occur. Thus an ordinary sparse non-empty chunk or non-cue empty costs two logical completions (normally two provider requests and at most six with all shipped retries); cue-bearing recovery can cost more within the global envelope. Dense or malformed primary responses use the same hard-bounded recursive recovery path; a saturated array, incomplete branch, failed verifier, or exhausted resource envelope fails the whole chunk rather than publishing a plausible-looking subset.

**Lock-free dedup embedding + same-wave collapse.** Triple-dedup similarity vectors are embedded *before* the per-chunk write transaction opens, so no embedding-API round-trip is ever held inside the SQLite writer lock; the in-lock path is pure SQL + in-memory cosine. The same in-memory vectors let sibling variants minted within one dream collapse against each other (not just against prior-cycle edges), which curbs the phrasal-variant edge proliferation a `prompt_version` bump used to cause — see §5.

**Bounded external calls.** The contrib OpenAI-compatible LLM client uses one explicit three-attempt exponential-backoff layer during extraction. Its SDK retries are disabled (`max_retries=0`) and every HTTP attempt has a finite 120-second timeout, so its request counter and 288-request per-chunk envelope reflect actual transport attempts. Embedding calls sit on query and post-commit ingestion paths, so their SDK client instead uses one attempt with an explicit 10-second default timeout; provider failures abstain from semantic retrieval while lexical tiers continue.

---

## 9. Configuration

All runtime config via environment variables. No config files. Run
`hymem-doctor` to print the resolved configuration and preflight every check
(keys, endpoint reachability, `sqlite-vec`, schema migration, embedding
dimension).

| Variable | Default | Purpose |
|---|---|---|
| `HYMEM_ROOT` | `~/.hermes` | Directory for sqlite + markdown files |
| `HYMEM_LLM_API_KEY` | provider key for an exact official origin | Purpose-bound LLM key for the configured endpoint; `DEEPSEEK_API_KEY`/`OPENAI_API_KEY` never cross to a custom host |
| `HYMEM_LLM_BASE_URL` | `https://api.deepseek.com` | LLM endpoint |
| `HYMEM_LLM_MODEL` | `deepseek-v4-flash` | Extraction model; retired mutable aliases `deepseek-chat` and `deepseek-reasoner` are rejected before client construction |
| `HYMEM_LLM_THINKING` | `auto` | DeepSeek thinking-body policy; `auto` safely disables thinking on identified DeepSeek endpoints/models, while unrelated providers receive no vendor-specific body |
| `HYMEM_EMBEDDING_API_KEY` | none | Purpose-bound key for an explicit embedding endpoint; `OPENAI_API_KEY` is inherited only for the exact official HTTPS `api.openai.com` origin, while loopback uses `local` |
| `HYMEM_EMBEDDING_BASE_URL` | `local://feature-hash` | Actual OpenAI-compatible API base, including its route prefix (for example `/v1` when served there); no automatic `/v1` suffix. Remote URLs require HTTPS by default; omission selects the deterministic no-network fallback |
| `HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP` | unset (`false`) | Narrow emergency opt-in for cleartext embedding endpoints on a private/link-local IP or internal service DNS name (for example `http://embedding-server:8766/v1`); public HTTP remains forbidden and HTTPS is preferred. Still requires the complete pinned producer configuration above |
| `HYMEM_EMBEDDING_MODEL` | `hymem-local-feature-hash-v1` | Exact embedding-space model id |
| `HYMEM_EMBEDDING_DIM` | `384` | Exact embedding dimension |
| `HYMEM_EMBEDDING_PIN_DIMENSION` | unset (`false`) | Required for network-backed durable vectors: assert the configured dimension with `1`; does not override the service's returned dimension |
| `HYMEM_EMBEDDING_DEPLOYMENT_REVISION` | unset | Required public operator-verified model/deployment revision for network-backed durable vector identity; never a credential |
| `HYMEM_EMBEDDING_DEPLOYMENT_TENANT` | unset | Required public operator-verified tenant/deployment scope for network-backed durable vector identity; never a credential |
| `HYMEM_EMBEDDING_TIMEOUT_SECONDS` | `10` | Per-request timeout for an explicitly configured remote embedder; SDK retries are disabled |
| `HYMEM_HONCHO_HOST` | `127.0.0.1` | Honcho server bind address |
| `HYMEM_HONCHO_PORT` | `8765` | Honcho server port |
| `HYMEM_DREAM_COOLDOWN_SECONDS` | `60` | Min seconds between event-triggered background cycle starts; not a periodic schedule |
| `HYMEM_AGGREGATION_NODES_ENABLED` | unset (config default: on) | Master switch: RAPTOR aggregation + standing digest built at dream time; set `false` to opt out |
| `HYMEM_AGGREGATION_DIGEST_ENABLED` | unset (config default: on) | Sub-switch: roll cluster nodes up into the root digest (active only with the master switch on) |

Tunable in `HyMemConfig` dataclass (programmatic):

| Parameter | Default | Purpose |
|---|---|---|
| `salience_min_chars` | 30 | User-turn length for high-priority extraction; shorter non-blank/non-trigger turns remain eligible through the baseline tier |
| `dream_baseline_budget` | 10 | Global per-cycle max for short/non-trigger baseline chunks (newest-first within each session; session priority rotates across cycles); a positive exhausted allowance reports `budget_exhausted` while actionable work remains, while 0 keeps candidates durable without running them or claiming exhaustion |
| `redact_secrets` | `True` | Scrub secrets/PII from messages before storage |
| `max_message_chars` | 100000 | Truncate a logged message longer than this (0 disables) |
| `max_query_chars` | 10000 | Truncate an `augment()` query longer than this (0 disables) |
| `working_memory_turns` | 10 | Recent raw turns `augment(session_id=…)` returns (0 disables) |
| `fts_top_k` | 5 | FTS results to return |
| `message_fts_top_k` | 5 | Raw-message lexical + semantic hits in `message_hits` (0 disables) |
| `message_fts_aggregate_cap` | 0 | Opt-in `ability="MR"` counting path; evidence cap, count stays exact (0 = off) |
| `procedure_top_k_if` | 10 | Procedure budget for `ability="IF"` (else `fts_top_k`) |
| `graph_top_k_per_entity` | 3 | Entity-anchored graph facts per matched entity |
| `embedding_max_scan` | 5000 | Max embeddings to scan in Python fallback |
| `graph_semantic_top_k` | 10 | KNN candidates pulled from `vec_edges` |
| `graph_predicate_top_k` | 10 | Edges pulled per predicate-routed query |
| `graph_top_k` | 8 | Final graph facts returned by `augment()` |
| `graph_recency_half_life_days` | 30.0 | Half-life for edge recency decay |
| `graph_recency_recent_days` | 7.0 | `days_since` under this emits a `recency_Nd` reason code |
| `graph_predicate_boost` | 1.5 | Score multiplier for routed-predicate edges |
| `graph_token_overlap_weight` | 0.5 | Score multiplier for overlap-only entity-anchored edges |
| `graph_token_overlap_threshold` | 20 | Token segment shared by more than this many canonicals is treated as common-token noise |
| `graph_token_overlap_max_per_entity` | 5 | Max token-overlap expansions per matched canonical |
| `decay_window_days` | 30 | Decay look-back window |
| `decay_factor` | 0.9 | (reserved, not yet used) |
| `retract_threshold` | 0.15 | Confidence below which edges retract |
| `profile_max_entries` | 16 | Max profile entries in USER.md |
| `insights_max_entries` | 12 | Max insights in MEMORY.md |
| `profile_max_items_per_session` | 16 | Max typed-profile items per bounded response; overflow fails atomically and retries |
| `profile_extraction_max_attempts` | 6 | Consecutive profile failures before cursor-preserving quarantine; 0 retries forever |
| `prompt_version` | `"v20"` | Human extraction generation; bump to deliberately force full reprocessing. Phase-1 also derives a cache identity from prompt/validator/recovery behavior, so accidental code drift cannot reuse this label's old rows |
| `chunk_extraction_max_attempts` | 3 | Consecutive provider/output failures before prompt-scoped quarantine (never marked processed); irrecoverable source loss is terminal and prompt-independent |
| `digest_extraction_max_attempts` | 6 | Consecutive digest failures before cursor-preserving quarantine; <=0 retries forever |
| `aggregation_nodes_enabled` | `True` | Master switch for the RAPTOR aggregation layer + digest; set `False` to opt out |
| `aggregation_digest_enabled` | `True` | Build the root standing digest at dream time (needs the master switch) |
| `aggregation_inject_abilities` | `("TR",)` | Abilities whose queries surface aggregation nodes at query time (additive tier) |
| `dream_budget` | 50 | Max chunks to process per dreaming cycle |
| `dream_extraction_provider_attempt_budget` | 200 | Soft Phase-1 HTTP-attempt ceiling per cycle; 0 is unlimited. The shipped client's only retry layer makes at most three counted HTTP attempts per logical call (SDK retries off); the current chunk finishes atomically, so the cycle can overshoot by at most one 288-attempt chunk envelope |
| `dream_digest_max_chars` | 12000 | Per-call digest input window; oversized stored messages resume at exact character offsets |
| `dream_digest_max_tokens` | 3072 | Per-call digest output ceiling; changing it safely reopens/rebuilds held digest work |
| `max_chunks` | 50000 | Soft cap on retrieval/extraction chunks; exact coverage artifacts are excluded |
| `retention_days` | 90 | Chunks newer than this always kept |
| `message_retention_days` | 0 | Opt-in raw-message pruning age; <=0 retains forever. Enabled pruning requires an ended session and exact per-message lossless chunk coverage |
| `tombstone_retention_days` | 0 | Opt-in retracted-edge pruning age; <=0 keeps graph evidence and lifecycle history forever so `facts_at()` remains complete. Positive values cascade-delete old tombstones and their history |
| `extraction_feedback_keep` | 200 | Newest source-linked retraction-audit rows retained; 0 discards all at pruning. Audit rows never enter extraction prompts or benchmark material/score identity |
| `rerank_top_k` | 20 | Candidate pool size handed to the reranker before final top-k trim |
| `rerank_model` | `"llm"` | `"llm"` (uses the configured LLM client) or `"cross-encoder"` (local sentence-transformers) |
| `rerank_cross_encoder_model` | `"mixedbread-ai/mxbai-rerank-base-v1"` | HuggingFace model id used when `rerank_model="cross-encoder"` |
| `rerank_ambiguity_threshold` | 0.6 | Min RRF score drop required before reranking fires; set high to disable |

`DreamReport.chunk_extraction_completion_calls` reports logical extraction
calls; `chunk_extraction_provider_attempts` reports underlying HTTP attempts
when the client exposes a monotonic `request_attempts` counter, with a
one-attempt-per-completion fallback for generic clients. `dream_runs` persists
both along with `extraction_provider_attempt_budget_exhausted`.

---

## 10. Test Coverage

The core and benchmark suites cover the surfaces below. This is a representative
inventory rather than a frozen test count; use the current test run as the source
of truth (the LongMemEval/BEAM execution harness is described separately in §11):

- `test_dreaming.py` — Full pipeline: chunk→extract→consolidate→decay
- `test_extraction.py` — Triple extraction, marker extraction, polarity handling, numeric / temporal value parsing
- `test_canonicalize.py` — Entity normalization, alias resolution, merging, canonical-drift detection and repair, entity-shape filter for object-canonical matches
- `test_chunks.py` — Salience detection, chunk persistence
- `test_embeddings.py` — Embedding creation and query, stub determinism, LRU cache (hit/miss accounting, batch split, eviction policy)
- `test_augment.py` — FTS search, vector search, RRF merge, graph lookup
- `test_graph_semantic.py` — Edge embedding, hybrid graph ranker (routed + per-candidate fallback branches), `why_retrieved` codes, predicate routing, contradiction detection, token-overlap entity expansion, token-overlap index cache + invalidation
- `test_rerank.py` — Hybrid reranking: LLM and cross-encoder backends, ambiguity gating, graceful fallback when sentence-transformers is unavailable
- `test_entity_properties.py` — Entity-type and entity-property extraction + query-time expansion
- `test_episodes.py` — Episodic memory: stable ids within an active digest walk,
  atomic replacement generations, and semantic episode search via `vec_episodes`
- `test_procedures.py` — Procedural memory: extraction validation, step-order renormalization, FTS-backed surfacing through `augment()`
- `test_summary.py` — Session summarization: persistence, idempotency, validation (short / quoted / long), Honcho context preference
- `test_inference.py` — Transitive inference: depends_on BFS + uses-cross-predicate rule, derivation refresh, retract-threshold filtering
- `test_markdown_io.py` — Section read/write atomicity
- `test_integration.py` — End-to-end capture→dream→augment, retract workflow
- `test_phase3_perf.py` — Decay correctness, mention indexing, backfill idempotency
- `test_mcp_server.py` — MCP tool correctness (all 12 tools, incl. `hymem_ask` synthesis, `hymem_digest`, and rule add/list/suggest)
- `test_retract.py` — Edge retraction, alias resolution, idempotency, feedback-row recording
- `test_bitemporal.py` — Valid-time interval: valid_at from positive-evidence world date (write-once), invalid_at on supersession from newest negative-evidence date, as-of resolution, retract_edge interval-close, migration backfill + export round-trip
- `test_raptor_cluster_probe.py` — Pure connected-components clustering core of the Phase-2 RAPTOR front-run gate (`benchmarks/raptor_cluster_probe.py`): cosine/Jaccard primitives, OR-link predicate (embedding *or* entity overlap), transitive closure, embedding-bridge across disjoint entity sets, threshold sensitivity. The DB/dream side runs only on the box; this pins the clusterer the build reuses (re-exported from `hymem.dreaming.aggregate`, so probe/tests/production share one clusterer)
- `test_aggregate.py` — Phase-2 RAPTOR aggregation layer (`hymem/dreaming/aggregate.py` + the additive retrieval tier): cluster-selection policy (only cross-session, multi-member clusters become nodes; singletons/single-session dropped), content-hash node id (order-independent, membership-sensitive), the persisted node + summary-embedding shape, full-rebuild semantics (no stale node lingers), the enabled-by-default build/query guards plus explicit opt-out, and that retrieval surfaces nodes *additively* without displacing the episode tier
- `test_dream_runs.py` / `test_dream_lease.py` — Audit persistence, lock-skip/error recording, dedicated periodic renewal, logical-call and within-session boundary checks, exact-holder fencing, cleanup, and stale takeover
- `test_dream_status_durable_work.py` / `test_dream_status_fact_quarantine.py` — Coherent schema-v2 completion snapshots across source materialization, extraction, digest, profile, facts, aggregation, quarantine, malformed state, and terminal loss
- `test_dedup_delock.py` — Dedup similarity vectors embedded outside the write lock (`conn.in_transaction` False at every embed), behavior preserved end-to-end
- `test_dedup_samewave.py` — Same-cycle sibling collapse (same- and cross-chunk), no over-merging of non-siblings, lexical guard still applies, no in-lock embed
- `test_hardening.py` — Secret/PII redaction, message/query size caps, embedding model/dim guard (incl. mixed-corpus), external-call retry/backoff
- `test_dream_scheduler.py` — Background dream cooldown + concurrency
- `test_honcho_server.py` — Raw HTTP behavior for the supported Honcho subset
- `test_honcho_contract.py` — Pinned real SDK against a live server, including directional representations and SSE chat
- `test_concurrency.py` — Dreaming + ingestion + reads coexisting under WAL

---

## 11. Benchmark Evidence (LongMemEval)

HyMem's harness targets **LongMemEval-S**: 500 questions across six base question types, each answered over a multi-session conversation haystack. The strict adapter (`benchmarks/longmemeval_adapter.py`) runs the real pipeline per question — isolated ingest → bounded healthy indexing → `hy.augment()` → host reader → pinned or explicitly legacy judge — and preserves every expected ID, failure, model identity, and usage segment.

**Current evidence policy.** The default answer path is label-free: source `question_type`, `_abs`, answers, and answer-session marks cannot steer sampling, retrieval, or the reader. Oracle routing is an explicitly exploratory diagnostic. Full-set runs remain development evidence because LongMemEval has no official held-out split; the manifest separately records denominator validity, exact evaluator identity, pre-registration, and whether a run is scored or retrieval-only. A required row digest binds the ordered results, while the latest pointer and registry bind the immutable full-artifact digest; copied evidence is deduplicated and a reused basename with different bytes is rejected.

**Current indexing evidence contracts.** Strict convergence adds
`hymem-benchmark-indexing-status-v3` to the coherent
`hymem-dream-status-v7` snapshot. LME persists and validates
`hymem-lme-indexing-summary-v5`, retaining exact Phase-1 authority. All four
strict registries require the portable finalized-checkpoint projection described
in [archive admission](benchmarks/archive_evidence.md). MSC and LoCoMo persist
`hymem-benchmark-indexing-v4`, whose per-run report counts, fixed
`DreamReport` totals/flags, terminal cycle, final status, and provider usage must
reconcile. Their reusable-store envelope is
`hymem-benchmark-store-build-v7`: it hashes an exact sanitized
`hymem-benchmark-store-indexing-attestation-v2` projection, re-attests the live
SQLite material with `hymem-material-store-attestation-v5`, binds the exact
secret-free Phase-1 producer/effective request, and binds the
transitive material-side implementation with `hymem-material-code-closure-v3`.
Caller-controlled effective-request and retry mappings are retained only as
canonical SHA-256 digests; credentials or opaque provider metadata therefore
cannot leak through the registry, status, portable export, or store receipt.
Older or relabelled nested evidence fails closed rather than becoming a scored
cache hit.

**Cost-aware screening is exploratory.** A lower-tier model may screen a fixed,
stratified subset only when baseline and candidate use the same exact pinned
model ID, seed/config, and fixed judge. Write-side/extraction changes additionally
need an early screened-model × target-model interaction check because model
quality changes the stored substrate. Promote on paired uncertainty and health,
retrieval, and answer metrics—not a single score—then rerun the full definitive
protocol under exact `deepseek-v4-flash`. See
[the model-migration note](references/deepseek-model-migration.md#cost-aware-lower-tier-screening).

Reader packing has two explicitly different capacity policies. A locally supplied `tokenizer.json` is hashed, bound to the answer model, and fails closed if counting fails. Without one, the default is a 60,000-byte UTF-8 budget—not a token claim—checked together with 1,024 output tokens and a 256-token chat-framing reserve against the declared provider context ceiling. Raw retrieval receives a 60% selection reserve before summaries or distilled aids, although those aids still lead the rendered prompt. Official judge prompts/model/parser can match upstream scoring semantics, but the local safety-bounded three-attempt transport differs from upstream's unbounded retry policy; artifacts therefore record scoring-semantics alignment separately and do not claim full protocol/transport equivalence.

- **Oracle** — the question-type label drives retrieval shaping; exploratory only.
- **Label-free (`--auto-ability`, the default)** — the in-library router infers ability from the question alone.

**Historical local-protocol snapshot (development-only; not official-comparable): 70.0% overall.** This 500-question run used the then-current DeepSeek reader/judge and local prompts. It is retained to explain engineering decisions, not as a current benchmark claim. Its category table was:

| Question type | Score |
|---|---|
| Single-session-user (SS-U) | 95.7% |
| Knowledge-update (KU) | 76.9% |
| Temporal-reasoning (TR) | 72.9% |
| Single-session-preference (SS-P) | 66.7% |
| Single-session-assistant (SS-A) | 66.1% |
| Multi-session (MS) | 51.9% |
| **Overall** | **70.0%** |

External numbers are not directly comparable without matching answer models, evaluator, item count, and metric. For orientation only: Honcho reports vendor results of 90.4 with Haiku 4.5 and 92.6 with Gemini 3 Pro; Hindsight's official March 2026 result is 94.6 in its single-query setup; Mnemosyne reports 98.9 **Recall@All@5** on 100 LongMemEval items, a retrieval metric rather than end-to-end answer accuracy. BEAM-100K's 65.2 is a separate workload and denominator. None of these values establishes a HyMem ranking.

**Historical local A/B (500q, seed 0, label-free router, permissive default).** Under that superseded protocol, full-dream was 3.6 points above no-dream:

| Question type | No-dream | Full-dream | Δ |
|---|---|---|---|
| Knowledge-update (KU) | 70.5% | 76.9% | +6.4pp |
| Multi-session (MS) | 42.9% | 51.9% | +9.0pp |
| Temporal-reasoning (TR) | 74.4% | 72.9% | −1.5pp |
| Single-session-preference (SS-P) | 56.7% | 66.7% | +10.0pp |
| Single-session-assistant (SS-A) | 67.9% | 66.1% | −1.8pp |
| Single-session-user (SS-U) | 94.3% | 95.7% | +1.4pp |
| **Overall** | **66.4%** | **70.0%** | **+3.6pp** |

The biggest lifts land where cross-session consolidation matters most: **MS +9pp** (dream bridges fractured sessions), **SS-P +10pp** (preferences extracted into graph facts), **KU +6.4pp** (knowledge updates consolidated). The TR and SS-A dips are within LLM-judge variance (~1.5–2pp). The recall ceiling confirms the *mechanism*, not just the score: retrieval misses dropped 42 → 39, and the "both" recovery tier (message + graph facts) jumped **0 → 98** — dream's graph facts are now actively contributing — while MS ranking misses dropped 10 (67 → 57), so the consolidated facts also improve the reranker's candidate quality. Abstention improves in step (70.0% → 76.7% correct refusals): graph consolidation helps the system tell "I truly don't know" from "I just can't find it in messages." The cost is real — **140 min vs 12 min (11.7×), ~1.7M tokens** — which is why dream is a background idle cycle in production, not an inline step.

**What drives the score — every lever carries to production, none reads the oracle label:**

- **Raw-message FTS tier** (`message_hits`, schema v13, §6) — a direct BM25 index over raw turns, recallable across sessions and *before* any dream consolidates them. Historically the single biggest jump.
- **Label-free ability routing** (`detect_ability`, EN+NL regex, §3) — fills MR/TR shaping in production with no oracle label. MR layers a deterministic, **additive** user-turn count on normal retrieval (a false-positive route stays harmless — retrieval is never suppressed); TR injects a date-ordered chronology.
- **Recency-dated context** — `message_hits` are stamped with their `created_at`, plus a value-aware recency clause so the answerer prefers the most recent turn that actually *states* a value rather than a later tangential mention. Lifted knowledge-update from 62.8% → 75.6%.
- **Permissive, abstention-guarded default prompt** — recovers single-session-preference questions the strict prompt refused, while holding the abstention guard tight on single-turn facts.
- **Dreaming (graph consolidation)** — the background dream cycle extracts a cross-session knowledge graph whose facts join the retrieval pool; worth +3.6pp overall and the dominant lever on MS / SS-P / KU (the A/B above).

**In that historical run, the largest weakness was multi-session (51.9%).** LLM-free probes suggested the remaining misses were dominated by cross-session synthesis plus a small sparse-signal floor. This is a development hypothesis, not a current leaderboard diagnosis. The investigation and dead ends remain in `benchmarks/longmemeval_roadmap.md`.

**Historical RAPTOR build note (2026-06-11, before the default flip): gate PASSED → built.** Since the MS residual is *synthesis* (all gold turns reach context, the reader fails to fuse ~45 raw slots), the fix was upstream: pre-compute hierarchical aggregation nodes so the answer model fuses a handful of cluster summaries instead of dozens of raw turns. This only helps if clustering *co-locates* a question's synthesis inputs into a single node — so, mirroring the front-run discipline that killed the L3 diversity-pack, it was gated by an offline probe (`benchmarks/raptor_cluster_probe.py`) **before** any table or migration was built. The probe dreamed each MS miss into episodes, clustered them across sessions by embedding-or-entity overlap, and measured how often the gold-bearing episodes landed in one cluster. **Result (grid emb≥0.55 OR ent≥0.50, 53 MS ranking misses): of the 31 questions whose gold turns became episodes, 27 (87%) had all their gold episodes land in a single cluster (mean 1.13 gold-clusters/question against 16.8 clusters/question)** — one aggregation-node summary captured everything the reader had to fuse. The remaining 22/53 (42%) had *no* gold episode at all: a dream **coverage** gap (episode-extraction recall), not a clustering gap. High co-location licensed the initial build: schema v16 (`aggregation_nodes` + `aggregation_node_embeddings`, migration `016`), `dreaming/aggregate.py`, a dream-runner step, and an additive retrieval tier. The master switch initially shipped off; after the Stage 3c G-FLIP passed, `aggregation_nodes_enabled` flipped to `True` on 2026-08-26. Current code therefore builds nodes and the standing digest by default, while query exposure remains additive and TR-only by default; set the master switch to `False` to opt out. The layer is unit-covered offline (`test_aggregate.py`, plus the clusterer pinned in `test_raptor_cluster_probe.py`); the decisive co-location run was on the Hermes box (it needs real dreams).

**G4 verdict (2026-06-11): closed as an LME lever → pivoted to a standing digest.** The on/off LME A/B lost (69.0 vs 70.0): the nodes recovered no messages the message/chunk tiers missed — they only reshuffled ranking, crowding knowledge-update gold out of the answer pool — and a TR-gated re-run was a wash (temporal-reasoning dead flat; the earlier +3pp was run variance). Retrieval-side injection structurally can't win where raw-message FTS already closes recall, so the layer's consumption model is now **host-facing standing context**: schema v17 adds RAPTOR hierarchy levels — dreaming recursively rolls the level-0 cluster nodes plus all unclustered episodes (capped, most-recent-first) up into one **root digest node**, with a consecutive-chunk fallback that guarantees convergence even when nothing clusters — and `HyMem.digest()` returns that root for system-prompt injection: the "what do you know about me?" answer no keyword retrieval can produce. Levels ≥ 1 never enter the query-time tier (level-0 filter in `_aggregation_search`), so the digest cannot crowd retrieval by construction, and every fusion is reuse-cached by member-set hash, so re-dreaming a stable store costs zero LLM calls. The query-time tier stays additive and ability-gated (`cfg.aggregation_inject_abilities`, default TR-only); set `aggregation_nodes_enabled=False` to opt out of the layer entirely.

**Reproduce:**

```bash
# exploratory full-set development run; produces no official-comparable claim
python benchmarks/longmemeval_adapter.py --sample 0 --seed 0 --auto-ability \
    --workers 4 --permissive-default --no-prereg \
    --answer-model deepseek-v4-flash --judge-model deepseek-v4-flash \
    --hymem-model deepseek-v4-flash --hymem-thinking disabled
# fast exploratory no-dream A/B
python benchmarks/longmemeval_adapter.py --sample 0 --seed 0 --auto-ability \
    --workers 4 --no-dream --permissive-default --no-prereg \
    --answer-model deepseek-v4-flash --judge-model deepseek-v4-flash \
    --hymem-model deepseek-v4-flash --hymem-thinking disabled
# oracle-label diagnostic (explicitly exploratory)
python benchmarks/longmemeval_adapter.py --sample 0 --seed 0 \
    --no-auto-ability --workers 8 --no-prereg \
    --answer-model deepseek-v4-flash --judge-model deepseek-v4-flash \
    --hymem-model deepseek-v4-flash --hymem-thinking disabled
```

**Methodology.** Every diagnostic behind these numbers is run against a control
arm before its result is read — see
[docs/diagnostic_controls.md](docs/diagnostic_controls.md) for the standing
discipline, the measured false-alarm rates per benchmark, and the read-only
instruments. It is not optional housekeeping: a missing control has already cost
this project one retracted ceiling and three reversed lever orderings.

---

## 12. Comparison with Honcho

| Dimension | HyMem | Honcho (plastic-labs) |
|---|---|---|
| **Scope** | Single-agent memory module for Hermes | Multi-tenant platform for stateful agents |
| **Architecture** | Embedded library (SQLite + 2 servers) | Client-server (FastAPI + Postgres + Redis + workers) |
| **Storage** | 1 SQLite file + 2 Markdown files | Postgres + pgvector + Redis cache |
| **Entity model** | Native roles plus arbitrary workspace-scoped Honcho peers | Peer paradigm: all participants are "peers" |
| **Memory extraction** | "Dreaming" — multi-phase LLM pipeline with locked vocabulary, transitive inference, episode/procedure extraction, and source-linked correction audit | "Deriver" — background workers doing representation, summarization, peer cards |
| **Ontology** | Locked 22-predicate vocabulary + entity types | Open-ended reasoning, no fixed ontology |
| **Query interface** | FTS5 + vector + RRF + LLM rerank + semantic graph ranking (with `why_retrieved` explainability) + predicate routing + episode/procedure search | Chat API (natural language), context (token-budgeted), hybrid search |
| **Decay** | Co-occurrence-aware with confidence thresholds | Continual representation updates (implicit) |
| **Contradiction detection** | `conflicts()` surfaces competing-object and opposing-predicate edges (pure SQL) | Not available |
| **Correction audit** | Source-linked, retention-bounded retraction records; never reused as prompt instructions | Not available |
| **Honcho SDK compat** | Hermes-used v3 subset via `hymem.honcho`, pinned real-SDK contract tests | Native/full surface |
| **Deployment** | Local-only, pip install, zero config | Managed cloud (app.honcho.dev) or self-hosted Docker/Fly.io |
| **SDKs** | Python + MCP + Honcho SDK | Python + TypeScript |
| **Maturity** | v0.1.0, ~15,000 lines | v3.0.6, 514 commits, 3.4k stars |
| **License** | Not specified | AGPL-3.0 |

**The key philosophical difference:** Honcho is a platform — multi-tenant, cloud-native, with a broad API surface for many use cases. HyMem is a tool — focused, embeddable, opinionated about what memory should look like. HyMem's locked vocabulary, co-occurrence-aware decay, transitive inference, semantic-and-explainable graph ranking, and deterministic prompt isolation are design bets that prioritize precision and attributable evidence. Honcho prioritizes flexibility and scale.

**HyMem can self-host the Honcho workflow Hermes uses.** The pinned SDK can perform peer/session management, message ingestion, pagination, search, directional representations/context, and JSON or streaming chat without leaving the machine. This compatibility layer does not claim parity with Honcho's full cloud API or its learned representation semantics.

---

## 13. Limitations & Known Gaps

- **Bounded exact vector scans**: durable, identity- and content-validated vectors are the retrieval authority; sqlite-vec tables are rebuildable acceleration shadows. Chunk/aggregation scans are bounded by `embedding_max_scan` (default 5000), so very large stores still need a production ANN candidate strategy that preserves these validation checks.
- **Partial Honcho surface**: The pinned Hermes-used SDK paths are supported and contract-tested; unlisted Honcho cloud endpoints and operational semantics are not implemented. Chat streaming uses SDK-compatible SSE.
- **Representation is a safe directional proxy, not Honcho's durable conclusion model**: `(observer → target)` reads currently render proof-valid target-authored occurrences from authorized shared sessions. They do not persist observer/observed conclusions, and therefore cannot yet represent something the observer learned about the target solely from observer-authored or third-party text.
- **Single-writer database**: WAL mode lets reads run concurrently with the writer, and dreaming/ingestion run on separate connections, but SQLite still serializes the two writers. Fine for a single-agent setup, not for multi-tenant.
- **No authentication**: Both MCP and Honcho servers are unauthenticated — they assume localhost-only access.
- **Latin-script only**: Canonicalization, query-time entity matching, and the LLM prompts handle Latin-script languages (English, Dutch, French, German, Spanish, etc.) — accents are folded into canonical keys. Chunking salience triggers are tuned for English and Dutch; other languages fall back to length-based salience (the LLM is still the real filter). Non-Latin scripts (CJK, Cyrillic, Arabic) are not supported.
- **LLM-dependent extraction quality**: Extraction quality ultimately depends on the LLM's capabilities. Retraction records are audit evidence, not a hidden training channel; a weak LLM can still produce sparse, noisy, or malformed outputs that the strict contract must hold or reject.
- **Re-extraction surge after an extraction-contract change**: a prompt-version bump or any change to bound prompt/validator/recovery behavior makes every successfully processed, source-manifested extraction chunk eligible for the new contract, so a large store can require many dream cycles. That work is amortized: each cycle obeys `dream_budget`, the independent short-turn baseline budget, and the soft provider-attempt budget, while each chunk has a hard 96-logical-call / 288-HTTP-attempt envelope and publishes atomically. Under v20 every successful non-empty terminal result receives exactly one omission-verification pass (including results recovered by empty verification or terminal retry), while every terminal primary-empty unit receives exactly one direct empty-verification pass. Cue-driven child recovery preserves that obligation on each terminal child, and an unsplittable suspicious unit is held if its second look is still empty. A non-empty EMPTY output may trigger only the standard one-shot OMISSION pass; EMPTY-of-EMPTY and OMISSION-of-OMISSION never occur. Oversized records use block-aware source-split contract `hymem-source-semantic-split-v9`: prose may split at paragraph or conservatively classified sentence-terminal boundaries. A source-backed right continuation receives an exact, context-only preceding window of at most 320 characters from the same message, applicable only to its first 320 authoritative characters; recursive prose cuts stitch that window from contiguous inherited bytes without adding provider calls. Fenced code (including an unmatched fence through EOF), each top-level list item with its nested/blank/lazy continuations, a heading with its first body block, and a defensible colon-led introduction with its immediate list/code block remain atomic. Exact block edges can separate complete later sections or sibling items of a standalone list. A source-backed canonical table with a non-empty header plus matching delimiter may additionally split only after a validated body row, with that trust carried into recursive continuations. When a heading or defensible colon-led paragraph introduces that table, a continuation receives both the exact prelude and canonical header as separately labelled, same-source interpretation context; neither repeated slice becomes authoritative fragment `content`. An inherited context is immutable through its exact applicability range, so header-shaped body rows cannot replace it; local table discovery resumes after that range for genuinely later tables. Source-less tables are held because they cannot carry context safely. Table-looking text in code and apparent headerless pipe grids provide no boundary authority. Ambiguous abbreviation, initial, decimal, version-like punctuation, and generic soft newlines are not boundaries. If any other atomic block is itself oversized, or no admissible boundary fits the resource envelope, the unit deliberately remains held rather than sending incomplete evidence to the provider. Provider/output failures remain pending until the contract-scoped retry bound quarantines them; they are never marked successful. A legacy chunk whose exact source manifest is provably unrecoverable instead enters prompt-independent terminal source-loss state: changing the prompt cannot reconstruct its evidence or retry it, and strict indexing health continues to report the loss. `GET /dream-status` (or `HyMem.dream_status()`) exposes the public prompt label, active contract/cache identity, pending, quarantine, terminal-loss, and budget state so a long re-extraction wave is observable rather than mistaken for a hang.
- **Aggregation builds fail closed**: when aggregation is enabled, the runner persists a config-bound pending marker before entering the build. A partial fusion failure, total exception, or process death remains pending across restarts and strict benchmark convergence retries it; only a zero-failure build for the active material config clears the marker. Disabling the layer or switching configs does not let an unrelated old failure poison the active policy, and bounded lifetime counters retain the audit evidence without storing prompt/source/exception text.
- **Best-effort redaction, not a guarantee**: secret/PII scrubbing targets high-confidence patterns (provider key prefixes, JWTs, PEM blocks, bearer/credential strings, emails). It deliberately avoids generic high-entropy heuristics that would shred ordinary prose, so a novel or unstructured secret format can slip through. Treat it as defense-in-depth, not a substitute for not pasting secrets.
