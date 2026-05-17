# HyMem improvement plan

Snapshot date: 2026-05-17. Items from the previous roadmap (sqlite-vec, expanded predicates, numeric/temporal facts, episodes, entity types, hybrid reranking, extraction feedback, multi-hop inference, procedural memory, session summarization) are all shipped — see `hymem/core/schema.sql`, `hymem/query/augment.py`, `hymem/dreaming/`. What follows is what remains genuinely worth doing.

Items are independent unless noted, ordered by impact-to-effort within each tier. Each item is scoped to be picked up cold by a single agent.

---

## Tier 1 — Quick wins (hours)

### 1a. Persist the token-overlap index as a side table

**Why this exists:** [build_token_overlap_index](hymem/query/augment.py#L675-L696) walks every active canonical and splits on `_` to build the segment→canonicals map used by overlap-based entity expansion. It is cached in-memory per `HyMem` instance ([api.py:124-132](hymem/api.py#L124-L132)) and invalidated on `dream()`, `merge_canonical()`, `retract_edge()`. But: any cold start, any forked instance, and any external writer pays the full scan on the next `augment()`. At a few hundred canonicals this is sub-millisecond; at tens of thousands (the stated long-term target) it becomes the augment hot path.

**Concrete changes:**
- `hymem/core/schema.sql`: add
  ```sql
  CREATE TABLE IF NOT EXISTS token_overlap_index (
      token TEXT NOT NULL,
      canonical TEXT NOT NULL,
      PRIMARY KEY (token, canonical)
  );
  CREATE INDEX IF NOT EXISTS idx_token_overlap_token ON token_overlap_index(token);
  ```
- `hymem/query/augment.py`: modify `build_token_overlap_index` to prefer reading from the table; if empty, do the existing scan, write the result, and return it. Keep the in-memory cache on `HyMem` for the hot path — the table is the cold-path / cross-instance source of truth.
- `hymem/dreaming/runner.py`: after Phase 3 (after [runner.py:255-262](hymem/dreaming/runner.py#L255-L262)), `DELETE FROM token_overlap_index` and rebuild from current active canonicals in the same transaction.
- `hymem/api.py`: in `merge_canonical()` ([api.py:186](hymem/api.py#L186)) and `retract_edge()` ([api.py:191](hymem/api.py#L191)), update or clear affected rows so the table stays consistent without a full dream cycle.

**Done when:** A cold `HyMem` instance (no in-memory cache) returns the same overlap expansions as the existing path, and `augment()` issues no full-table scan over canonicals during the first call.

---

### 1b. Content-addressed embedding cache

**Why this exists:** `chunk_embeddings` ([schema.sql:68-74](hymem/core/schema.sql#L68-L74)) is keyed on `chunk_id`. Two chunks containing the same normalized text (common with paste-back, quoted code, repeated boilerplate, or near-identical session prefixes) re-call the embedding API and re-store the vector. With paid embedding APIs this is direct cash burn; with local models it's wall time.

**Concrete changes:**
- `hymem/core/schema.sql`: add
  ```sql
  CREATE TABLE IF NOT EXISTS embedding_cache (
      text_hash TEXT NOT NULL,
      model TEXT NOT NULL,
      vector_json TEXT NOT NULL,
      dim INTEGER NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (text_hash, model)
  );
  ```
  Hash is `sha256(normalized_text)` where normalization is: strip leading/trailing whitespace, collapse internal whitespace runs to single space, lowercase. Define the normalizer once in `hymem/extraction/embeddings.py` so chunk-side and edge-side use the same rule.
- `hymem/dreaming/embeddings.py`: in `fetch_chunk_embeddings` ([embeddings.py:28-60](hymem/dreaming/embeddings.py#L28-L60)), before calling `embedder.embed(texts)`, look up each text's hash in `embedding_cache`. Build two lists: cache-hit (skip embedder) and cache-miss (send to embedder). Merge results in original order. Apply the same pattern to `fetch_edge_embeddings`.
- `persist_chunk_embeddings` / `persist_edge_embeddings`: write back to `embedding_cache` for every miss before writing to `chunk_embeddings` / `edge_embeddings`.

**Notes:**
- `chunk_embeddings` and `edge_embeddings` stay as today — they hold the per-chunk/per-edge mapping. `embedding_cache` is the content-addressed layer underneath.
- Cap or prune is not needed in v1: at HyMem's scale the cache is bounded by unique-text count, not chunk count.
- Hash collisions: SHA-256 makes them irrelevant in practice; do not add collision-handling code.

**Done when:** Embedding a chunk whose normalized text already appears in `embedding_cache` issues zero calls to `EmbeddingClient.embed`. Add a counter to `DreamReport` (`chunks_embedded_from_cache: int`) so the win is observable.

---

### 1c. Parallelize chunk embedding with Phase 1 extraction

**Why this exists:** [runner.py:96-189](hymem/dreaming/runner.py#L96-L189) processes chunks one-by-one through `phase1.extract_chunk_results` (LLM call), then *after* the loop runs `fetch_chunk_embeddings` ([runner.py:232-236](hymem/dreaming/runner.py#L232-L236)) which embeds *all* unembedded chunks. These two API call streams are independent and both I/O-bound. Today they're serial.

**Concrete changes:**
- `hymem/dreaming/runner.py`: after each `persist_chunks` write (at [runner.py:106-109](hymem/dreaming/runner.py#L106-L109) and the baseline variant at [runner.py:159-162](hymem/dreaming/runner.py#L159-L162)), kick off chunk embedding for the just-written batch on a background thread (`concurrent.futures.ThreadPoolExecutor(max_workers=1)` is enough — embedding clients are typically threadsafe over HTTP; the SQLite write must remain on the main thread).
- The background task: `embedder.embed(texts)` only — produces a `PendingChunkEmbeddings`. Return it via the future.
- After the per-session loop, join all in-flight futures and call `persist_chunk_embeddings` on the main thread, in one transaction.
- If a future raises, log and continue. Do not let an embedding failure kill the dream cycle.

**Notes:**
- Do not parallelize the LLM extraction calls. They share the same client and ordering matters for `processed_chunks` updates.
- The existing post-loop `fetch_chunk_embeddings` becomes the fallback for chunks the background path missed (e.g., from sessions skipped at [runner.py:194-195](hymem/dreaming/runner.py#L194-L195)). Keep it.

**Done when:** Wall time for a dream cycle on a session with 10 high-salience chunks drops materially (target: 30%+ reduction) when both LLM and embedding clients have non-trivial latency. Verify with a stub embedder that artificially sleeps.

---

### 1d. Surface confidence and a hedge hint in `AugmentedContext`

**Why this exists:** `GraphFact` ([augment.py:20-30](hymem/query/augment.py#L20-L30)) already carries `confidence`, `pos_evidence`, `neg_evidence`. The consumer (Hermes) assembles the final prompt and has no easy signal that a `confidence=0.55` fact should be hedged ("you may use X") while a `confidence=0.95` fact can be stated flatly ("you use X"). Right now a single weak early extraction reads as assertive context indefinitely.

**Concrete changes:**
- `hymem/query/augment.py`: add a derived field on `GraphFact`:
  ```python
  @property
  def hedge_recommended(self) -> bool:
      return self.confidence < 0.75 or (self.pos_evidence + self.neg_evidence) < 3
  ```
  Thresholds: `0.75` and `3` are starting values — expose both via new `HyMemConfig` fields `hedge_confidence_threshold: float = 0.75` and `hedge_min_evidence: int = 3`. Compute the property by passing thresholds through, or set the value on the dataclass during construction in `_graph_lookup`.
- `hymem/config.py`: add the two new fields with docstrings.
- Same treatment for `EpisodeHit` and `ProcedureHit` if their scores are weak — but only if their scoring path admits a meaningful confidence (today they ride BM25, which is rank-only). For v1, skip those and only treat `GraphFact`.

**Notes:**
- No change to how Hermes consumes the data — this is additive. Whatever assembler Hermes uses can read `fact.hedge_recommended` if it wants to.
- Do **not** mutate the fact text or wrap it in "maybe" — leave that decision to the consumer. HyMem only signals.

**Done when:** `augment(...).graph_facts[i].hedge_recommended` returns `True` for low-confidence / thin-evidence facts and `False` otherwise, with thresholds reachable via config.

---

## Tier 2 — Bigger leverage (days)

### 2a. Skip-known-chunks pre-filter (the dominant cost win)

**Why this exists:** The dream cycle's dominant LLM cost is one call per chunk at [phase1.extract_chunk_results](hymem/dreaming/phase1.py#L24) inside the loops at [runner.py:124-138](hymem/dreaming/runner.py#L124-L138) and [runner.py:167-185](hymem/dreaming/runner.py#L167-L185). With `dream_budget=50`, that's up to 50 LLM calls per cycle. Many of these chunks re-confirm well-established edges (`uses python`, `depends_on postgres`) — the graph already holds the relationship at confidence > 0.9, and the LLM is paying full price to re-add evidence.

The pre-filter: for chunks whose mentioned entities all have outgoing high-confidence edges reinforced in the recent window, skip the LLM call and instead bump `last_reinforced` directly via the existing reinforce path. The baseline-tier backstop ([runner.py:147-189](hymem/dreaming/runner.py#L147-L189)) still pulls a few skipped chunks back through full extraction at low cadence, so genuinely new relationships on familiar entities are eventually caught.

**Concrete changes:**
- `hymem/dreaming/` (new module `skip_filter.py`): function `should_skip_extraction(conn, chunk, cfg) -> tuple[bool, list[int]]` returning `(skip, edge_ids_to_reinforce)`. Logic:
  1. Look up `entity_mentions` ([schema.sql:43-49](hymem/core/schema.sql#L43-L49)) for `chunk.id`. (Mentions are populated by `index_chunk_mentions` at [runner.py:109](hymem/dreaming/runner.py#L109) *before* the extraction loop — verify this is the case. If not, move mention indexing earlier or re-derive from text.)
  2. If the chunk has zero known canonical mentions: do not skip — let the LLM see it (this is where new entities get discovered).
  3. For each mentioned canonical, query `knowledge_graph` for active edges where it is subject or object. If *every* such edge has `confidence >= cfg.skip_confidence_threshold` and `last_reinforced >= now - cfg.skip_reinforce_window_days`: return `(True, [edge_ids])`.
  4. Otherwise: `(False, [])`.
- `hymem/dreaming/runner.py`: before the `phase1.extract_chunk_results` call at [runner.py:125](hymem/dreaming/runner.py#L125) and the baseline equivalent at [runner.py:168](hymem/dreaming/runner.py#L168), call `should_skip_extraction`. If skip: bump `last_reinforced = CURRENT_TIMESTAMP` and `pos_evidence = pos_evidence + 1` for the returned edges (transactional), record in `processed_chunks` with `prompt_version=cfg.prompt_version` so the chunk isn't re-considered, decrement `chunks_remaining`, increment a new `report.chunks_skipped` counter, and `continue`.
- `hymem/config.py`: add
  ```python
  skip_confidence_threshold: float = 0.9
  skip_reinforce_window_days: float = 14.0
  skip_filter_enabled: bool = True
  ```
  Keep the kill switch — this is a behavioral change in extraction and operators will want to A/B.
- `DreamReport` in `hymem/dreaming/runner.py`: add `chunks_skipped: int = 0` and persist to `dream_runs` (extend the schema with `chunks_skipped INTEGER NOT NULL DEFAULT 0` and update the UPDATE at [runner.py:274-298](hymem/dreaming/runner.py#L274-L298)).

**Notes:**
- The baseline-tier backstop at [runner.py:147-189](hymem/dreaming/runner.py#L147-L189) ensures starvation is bounded — a chunk skipped here is still eligible to be sampled later when budget remains.
- Skipping must still record in `processed_chunks` so the dream loop doesn't keep re-evaluating the same chunk forever.
- Edge case: a chunk mentioning a known entity *with a new relationship type*. The filter will skip it on the high-confidence-edges check. The baseline tier eventually catches this. If empirical loss is too high, tighten: require that the chunk's mentioned entities also collectively appear in an edge together (i.e., the chunk is reinforcing an existing pair, not introducing a new one).

**Done when:** With `skip_filter_enabled=True` on a populated graph, `report.chunks_skipped / report.chunks_seen` is meaningfully > 0 across a normal dream cycle, and no `phase1` LLM call is issued for chunks the filter accepts. Smoke test: a session whose chunks only reference well-established entities completes with `chunks_processed=0` and `chunks_skipped > 0`.

---

### 2b. Online thin extraction at write time

**Why this exists:** Today HyMem has no signal between "session writes a message" and "dream cycle runs." A user can complete a 30-message conversation and `augment()` will return no FTS / entity / graph context from it until the next cron tick. For an *embedded* memory module, that staleness is the single most visible weakness.

The architectural fix: split extraction into two passes.
- **Online (write-path, cheap, no LLM)**: chunk persistence, FTS indexing, regex behavioral markers, NER-style entity mention indexing. Runs synchronously in `log_message` / `log_messages`. Adds milliseconds.
- **Offline (dream, LLM)**: triple extraction, episodes, summary, procedures, embeddings. Unchanged.

After this split, `augment()` immediately surfaces FTS hits and entity-anchored graph lookups from in-progress sessions. The knowledge graph and episodes catch up at dream time.

**Concrete changes:**
- `hymem/dreaming/chunks.py`: refactor [extract_high_salience_chunks](hymem/dreaming/chunks.py#L52) so the regex / salience detection is callable at message-append time on a small rolling window (last N messages of the session). Keep the dream-time entry points; add `extract_online_chunks(conn, session_id, message_window) -> list[Chunk]` that does salience detection on freshly-appended messages only.
- `hymem/session.py`: in `append_message` ([session.py:29](hymem/session.py#L29)), after the INSERT, call the online extractor. Persist any detected chunks via the existing `persist_chunks` (FTS triggers fire automatically — see [schema.sql:59-64](hymem/core/schema.sql#L59-L64)). Index entity mentions via `index_chunk_mentions` from `hymem/dreaming/mentions.py`.
- Behavioral markers (the explicit regex-detected signals — `'correction'`, `'preference'`, `'rejection'`, `'style'`): move the marker detection out of phase1 (which extracts via LLM) and into a regex pass that can run online. If `hymem/dreaming/phase1.py` currently leans on the LLM for these, factor the regex layer out of any helper module that wraps it, or add a new `hymem/extraction/markers.py` with deterministic patterns. Insert markers into `behavioral_markers` ([schema.sql:157-165](hymem/core/schema.sql#L157-L165)) at append time.
- `hymem/dreaming/runner.py`: the existing `extract_high_salience_chunks` call at [runner.py:100-102](hymem/dreaming/runner.py#L100-L102) becomes a no-op for chunks already written online — the existing `processed_chunks` / `chunk_id` deduplication handles this naturally as long as IDs are stable. Verify `_chunk_id` ([chunks.py:193](hymem/dreaming/chunks.py#L193)) produces the same ID for the same `(session_id, start, end)` regardless of who calls it.
- `hymem/config.py`: `online_extraction_enabled: bool = True` kill switch.

**Performance budget for the online path:** target < 5ms p99 added latency to `log_message`. If the regex pass exceeds this, push it to a background thread and let `log_message` return immediately — but make sure the FTS / mention writes commit on the SQLite writer thread (SQLite is single-writer).

**Risks:**
- Doubles the write-path complexity. Keep the online path strictly regex/lookup based: no LLM, no embedding, no network.
- `index_chunk_mentions` needs to look up canonicals — make sure `entity_aliases` reads are fast (they're already indexed). At very high write rates this may matter.
- Salience regex evolved alongside the LLM extraction pipeline — moving it earlier may surface cases where the salience pattern is too eager. Treat any false positives as bugs to fix in the regex, not as reasons to gate the change.

**Done when:** Within the same process, `hm.log_message(...)` followed immediately by `hm.augment(...)` returns FTS hits and `matched_entities` from the just-logged content. Triples / episodes / summary remain dream-cycle-only.

---

## Pro memori

### Deeper graph reasoning (currently not pursued)

The Tier-3 "graph becomes a reasoning engine" idea (n-hop transitive inference beyond the current 2-hop closure in [hymem/dreaming/inference.py](hymem/dreaming/inference.py)) is intentionally on hold.

**Why not now:**
- Transitive closure compounds extraction errors. A single wrong edge at hop 1 spreads to every derived edge it touches at hop 2+. The current 2-hop closure is already at the edge of acceptable noise.
- The marginal retrieval gain from hops 3+ is small relative to the noise it introduces — most user queries are satisfied by 1-hop entity-anchored lookup plus the existing 2-hop derived edges.
- The right precondition is **extraction quality**, not graph depth. If 2a (skip-known-chunks) and a future extraction-quality pass land cleanly and the graph stabilizes at high confidence, *then* deeper traversal becomes interesting.

**Revisit when:**
- Median edge confidence on a mature graph is > 0.85.
- A measurable share of augment queries return zero `graph_facts` despite the user's question being semantically reachable in 3 hops.
- There is appetite for a confidence-decay-on-derivation policy (e.g., derived-edge confidence = product of source confidences × hop penalty) and a way to mark inferences "explanatory only, not assertive."

Until those land, do not deepen the inference layer.
