-- HyMem schema for a fresh database. Forward-only upgrades for existing DBs
-- live as NNN_*.sql files under hymem/core/migrations/, applied by the runner
-- in db.py. Keep this file and the migrations in sync: a column added here must
-- also have a migration so old databases pick it up.
--
-- CRITICAL: this file runs via executescript() BEFORE migrations. An existing
-- table is left untouched by `CREATE TABLE IF NOT EXISTS`, so any *standalone*
-- statement here that references a migration-added column (a `CREATE INDEX`, a
-- separate `ALTER`, etc.) will crash on old DBs with "no such column". Such
-- index/constraint statements must live in the migration file ONLY. The column
-- may still appear in the `CREATE TABLE` above (harmless no-op on old DBs,
-- correct on fresh ones).
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', '1');
-- Durable discriminator for pre-bootstrap v55 loss detection. The additive
-- loader could otherwise recreate a wholly dropped aggregation domain empty.
INSERT OR IGNORE INTO schema_meta(key, value)
VALUES ('aggregation_typed_provenance_schema', '55');

-- Raw session log. Hermes pushes messages in; HyMem owns the table.
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    -- Exact external namespace for Honcho-backed sessions. NULL is reserved
    -- for native/legacy HyMem sessions whose ownership is genuinely unknown.
    source_workspace_id TEXT CHECK (
        source_workspace_id IS NULL OR length(trim(source_workspace_id)) > 0
    ),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    summary TEXT,
    -- prompt_version of the last successful session digest (episodes+summary
    -- +procedures). The dream runner skips the per-session digest LLM call when
    -- this matches the current prompt_version and no chunk was re-extracted.
    -- Migration 012 adds this for existing DBs (ALTER lives there only).
    digested_prompt_version TEXT,
    -- PROFILE_PROMPT_VERSION of the last successful user-profile extraction
    -- (same skip mechanics as digested_prompt_version, but decoupled so a
    -- profile-prompt bump alone re-extracts). Migration 019 adds this for
    -- existing DBs (ALTER lives there only).
    profile_prompt_version TEXT,
    -- Resumable USER-only profile walk. The cursor is independent of chunk
    -- salience and digest coverage; its generation includes prompt/config plus
    -- one walk id. Partial outputs live in profile_staging and remain invisible
    -- until the walk reaches the ordered-stream tail.
    profile_cursor_message_id INTEGER,
    profile_cursor_partial_message_id INTEGER,
    profile_cursor_offset INTEGER NOT NULL DEFAULT 0
        CHECK (profile_cursor_offset >= 0),
    profile_cursor_prompt_version TEXT,
    profile_published_generation TEXT,
    profile_retry_count INTEGER NOT NULL DEFAULT 0
        CHECK (profile_retry_count >= 0),
    profile_retry_config_version TEXT,
    profile_quarantined BOOLEAN NOT NULL DEFAULT 0
        CHECK (profile_quarantined IN (0, 1)),
    -- Highest message id covered by a successful digest. The digest reads only
    -- chunks ABOVE this watermark, so a long-lived session's tail keeps getting
    -- digested instead of the head being re-read forever (the truncation was
    -- `combined[:max_chars]` over the whole session), and new traffic re-opens
    -- the digest even when no chunk was freshly extracted. NULL = no coverage
    -- recorded, so the next dream digests from the start of the session.
    -- Migration 024 adds this for existing DBs (ALTER lives there only).
    digested_message_id INTEGER,
    -- Facts watermark (v26): same mechanics as digested_message_id but for the
    -- narrative-facts extractor, its own column so facts and digest coverage
    -- advance independently. Migration 026 adds this for existing DBs.
    facts_message_id INTEGER,
    -- v46 authoritative fact walk. Unlike the compatibility watermark above,
    -- this cursor can stop inside one oversized lossless source turn.
    facts_cursor_message_id INTEGER,
    facts_cursor_partial_message_id INTEGER,
    facts_cursor_offset INTEGER NOT NULL DEFAULT 0
        CHECK (facts_cursor_offset >= 0),
    facts_cursor_prompt_version TEXT,
    facts_retry_count INTEGER NOT NULL DEFAULT 0
        CHECK (facts_retry_count >= 0),
    facts_retry_config_version TEXT,
    facts_quarantined BOOLEAN NOT NULL DEFAULT 0
        CHECK (facts_quarantined IN (0, 1)),
    -- EPISODE_GRANULAR_PROMPT_VERSION of the last digest that wrote this
    -- session's episodes (v35, Plan C). Same skip mechanics as
    -- profile_prompt_version, decoupled so an episode-granularity change alone
    -- re-extracts: the digest guard keys on cfg.prompt_version, which a
    -- granularity flip does not move. NULL = the shipping blob digest prompt
    -- (and every pre-v35 row), so a store that never enables granularity never
    -- sees a mismatch and pays no re-extraction. Migration 035 adds this for
    -- existing DBs (ALTER lives there only).
    episodes_prompt_version TEXT,
    -- Highest source message durably materialized as a canonical JSONL
    -- coverage artifact.  Independent from the LLM digest cursor.
    coverage_message_id INTEGER,
    -- Highest lossless frontier completely examined by both Phase-1 chunk
    -- producers.  The config identity prevents a changed producer from
    -- inheriting an older acknowledgement.  These are operational cursors:
    -- portable imports may omit them and safely rebuild from exact coverage.
    source_materialized_message_id INTEGER,
    source_materialization_config_version TEXT,
    -- Resumable input position for the current digest build generation.  The
    -- value identifies both its prompt/config and one full replacement walk.
    -- The offset belongs to the first covered message above
    -- digest_cursor_message_id and is zero at message boundaries.
    digest_cursor_message_id INTEGER,
    digest_cursor_partial_message_id INTEGER,
    digest_cursor_offset INTEGER NOT NULL DEFAULT 0
        CHECK (digest_cursor_offset >= 0),
    digest_cursor_prompt_version TEXT,
    -- Generation currently published to episode retrieval/embedding/
    -- aggregation.  A replacement walk writes distinct episode ids while this
    -- marker remains on the last complete generation, then switches it in the
    -- same transaction that retires the old rows.
    digest_published_generation TEXT,
    digest_retry_count INTEGER NOT NULL DEFAULT 0
        CHECK (digest_retry_count >= 0),
    digest_retry_config_version TEXT,
    digest_quarantined BOOLEAN NOT NULL DEFAULT 0
        CHECK (digest_quarantined IN (0, 1)),
    -- Automatic summaries roll with the digest.  `summary` remains the
    -- compatibility/operator value; source attribution prevents an automatic
    -- pass from overwriting an operator or conservatively-classified legacy
    -- value.  Migration 038 adds these columns to existing stores.
    auto_summary TEXT,
    auto_summary_message_id INTEGER,
    auto_summary_partial_message_id INTEGER,
    auto_summary_message_offset INTEGER NOT NULL DEFAULT 0,
    summary_source TEXT CHECK (summary_source IN ('auto','operator','legacy'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
    -- Honcho peer identity is not interchangeable with role. These nullable
    -- columns are an all-or-nothing pair; v43 guards validate them against the
    -- workspace peer registry and session binding.
    source_peer_id TEXT,
    source_workspace_id TEXT,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

-- FTS over raw message text — a direct keyword path to the session log,
-- complementing chunks_fts (which only covers high-salience spans materialized
-- during dreaming). Populated live at ingest via triggers, so a turn is
-- searchable the moment it is logged: across sessions and before any dream has
-- consolidated it. Only user/assistant turns are indexed; tool/system turns are
-- excluded as retrieval noise / index bloat. messages.id is INTEGER PRIMARY KEY
-- (a rowid alias), so content_rowid='id' joins straight back to the row.
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Messages are append-only (no UPDATE path in session.py), so insert + delete
-- triggers keep the index in sync; the WHEN guard mirrors the role filter so
-- tool/system turns never enter — and a delete of one safely no-ops.
CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages
WHEN new.role IN ('user','assistant') BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages
WHEN old.role IN ('user','assistant') BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

-- High-salience chunks identified during dreaming phase 1.
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    start_message_id INTEGER NOT NULL,
    end_message_id INTEGER NOT NULL,
    salience_reason TEXT NOT NULL,
    text TEXT NOT NULL,
    -- Coverage chunks contain one exact canonical message record.  They are
    -- durable source artifacts, not Phase-1 extraction/retrieval candidates.
    chunk_kind TEXT NOT NULL DEFAULT 'extraction'
        CHECK (chunk_kind IN ('extraction','coverage')),
    source_manifest_version TEXT,
    source_manifest_count INTEGER CHECK (
        source_manifest_count IS NULL OR source_manifest_count > 0
    ),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);

-- Explicit proof that one raw message is represented verbatim in a durable
-- chunk. Merely having a session summary, digest watermark, episode, or graph
-- edge is not lossless coverage and must never authorize raw-message deletion.
-- The content fingerprint prevents a stale record from covering a message
-- whose role/content was changed out-of-band; hash/record versions make that
-- contract evolvable. Source metadata survives raw-message deletion for audit.
-- The chunk FK is RESTRICT: the durable artifact cannot be deleted while this
-- proof exists. Migration 037 adds this table to existing databases.
CREATE TABLE IF NOT EXISTS message_retention_coverage (
    message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_role TEXT NOT NULL CHECK (source_role IN ('user','assistant','system','tool')),
    source_peer_id TEXT,
    source_workspace_id TEXT,
    source_created_at TIMESTAMP,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE RESTRICT,
    message_content_hash TEXT NOT NULL,
    hash_version TEXT NOT NULL,
    record_version TEXT NOT NULL,
    coverage_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (message_id, chunk_id, coverage_version)
);
CREATE INDEX IF NOT EXISTS idx_message_retention_coverage_chunk
    ON message_retention_coverage(chunk_id);
CREATE INDEX IF NOT EXISTS idx_message_retention_coverage_stream
    ON message_retention_coverage(
        source_session_id, coverage_version, message_id
    );

-- v50: durable fail-closed health signal when ordered coverage integrity could
-- not be established. One row per session bounds storage across transient or
-- persistent failures regardless of retry cadence.
-- Values are deliberately structural only: no source bytes, exception text,
-- endpoint data, or credentials are retained.
CREATE TABLE IF NOT EXISTS coverage_integrity_failures (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    config_version TEXT NOT NULL CHECK (
        config_version = 'lossless-coverage-integrity-v1|coverage=dream-lossless-message-v1|hash=sha256-role-content-v1|record=hymem-message-jsonl-v1'
    ),
    failure_reason TEXT NOT NULL CHECK (
        failure_reason IN ('materialization_failure', 'source_stream_invalid')
    ),
    occurrences INTEGER NOT NULL DEFAULT 1 CHECK (
        occurrences BETWEEN 1 AND 2147483647
    ),
    first_detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Coverage becomes immutable once its raw source is gone. The hash UDF is
-- registered by core/db.py on every HyMem connection. Together with the chunk
-- FK and chunk-update guard, even direct SQL cannot silently discard or mutate
-- the only durable copy; restore the raw source before explicitly releasing it.
CREATE TRIGGER IF NOT EXISTS message_retention_coverage_delete_guard
BEFORE DELETE ON message_retention_coverage
WHEN NOT EXISTS (
    SELECT 1 FROM messages m
    WHERE m.id = old.message_id
      AND m.session_id = old.source_session_id
      AND m.role = old.source_role
      AND m.created_at IS old.source_created_at
      AND old.hash_version = 'sha256-role-content-v1'
      AND hymem_message_content_hash(m.role, m.content) = old.message_content_hash
) BEGIN
    SELECT RAISE(ABORT, 'cannot release coverage while raw source is absent');
END;

CREATE TRIGGER IF NOT EXISTS message_retention_coverage_update_guard
BEFORE UPDATE ON message_retention_coverage
WHEN NOT EXISTS (
    SELECT 1 FROM messages m
    WHERE m.id = old.message_id
      AND m.session_id = old.source_session_id
      AND m.role = old.source_role
      AND m.created_at IS old.source_created_at
      AND old.hash_version = 'sha256-role-content-v1'
      AND hymem_message_content_hash(m.role, m.content) = old.message_content_hash
) BEGIN
    SELECT RAISE(ABORT, 'cannot mutate coverage while raw source is absent');
END;

-- Unlike generic v37 retention proofs, this producer is also the ordered
-- digest stream. Removing or rewriting a row behind its session cursor would
-- create a hole that hot-path ingestion cannot safely discover without an
-- O(history) scan, so recognized stream rows are permanently immutable.
CREATE TRIGGER IF NOT EXISTS message_lossless_stream_delete_guard
BEFORE DELETE ON message_retention_coverage
WHEN old.coverage_version = 'dream-lossless-message-v1' BEGIN
    SELECT RAISE(ABORT, 'ordered digest coverage is immutable');
END;
CREATE TRIGGER IF NOT EXISTS message_lossless_stream_update_guard
BEFORE UPDATE ON message_retention_coverage
WHEN old.coverage_version = 'dream-lossless-message-v1' BEGIN
    SELECT RAISE(ABORT, 'ordered digest coverage is immutable');
END;

-- Once a raw message enters the ordered stream, the canonical artifact and
-- proof are immutable.  Prevent an out-of-band UPDATE from making the source
-- table disagree with that durable copy.  DELETE deliberately remains legal:
-- verified raw-message retention depends on it after coverage is established.
CREATE TRIGGER IF NOT EXISTS message_lossless_source_update_guard
BEFORE UPDATE OF id, session_id, role, source_peer_id, source_workspace_id,
                 content, created_at ON messages
WHEN EXISTS (
    SELECT 1 FROM message_retention_coverage mc
    WHERE mc.message_id = old.id
      AND mc.coverage_version = 'dream-lossless-message-v1'
) BEGIN
    SELECT RAISE(ABORT, 'ordered digest source is immutable');
END;

CREATE TRIGGER IF NOT EXISTS message_retention_covered_chunk_update_guard
BEFORE UPDATE OF session_id, start_message_id, end_message_id, text, chunk_kind ON chunks
WHEN EXISTS (
    SELECT 1 FROM message_retention_coverage mc WHERE mc.chunk_id = old.id
) BEGIN
    SELECT RAISE(ABORT, 'cannot mutate a covered lossless chunk');
END;

-- v40 manifest tables, indexes, and guards live in migration 040.  They may
-- reference columns absent from an existing v39 database, while initialize()
-- intentionally executes this bootstrap schema before forward migrations.

-- Inverted index: which canonical entities does each chunk mention?
-- Populated after canonicalization runs and entities are known.
CREATE TABLE IF NOT EXISTS entity_mentions (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_canonical TEXT NOT NULL,
    PRIMARY KEY (chunk_id, entity_canonical)
);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_canonical ON entity_mentions(entity_canonical);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_chunk ON entity_mentions(chunk_id);

-- Explicit dates written in raw message text, extracted during dreaming
-- (dreaming/temporal.py). The temporal-reasoning (TR) augment path reads this
-- to return events already sorted by date, so the host LLM never has to find
-- dates in noise. `normalized_date` is ISO YYYY-MM-DD when a full date (incl.
-- year) was resolvable, else NULL for a year-less mention whose raw_text and
-- the turn's `created_at` still carry ordering signal. One row per distinct
-- (message_id, raw_text) so a re-dreamed chunk does not duplicate mentions.
-- Migration 014 adds this to existing DBs.
CREATE TABLE IF NOT EXISTS temporal_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    normalized_date TEXT,
    raw_text TEXT NOT NULL,
    surrounding_text TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (message_id, raw_text)
);
CREATE INDEX IF NOT EXISTS idx_temporal_mentions_date ON temporal_mentions(normalized_date);
CREATE INDEX IF NOT EXISTS idx_temporal_mentions_message ON temporal_mentions(message_id);

-- FTS over chunk text. External-content table keeps storage tight.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Dedicated exact-message corpus for durable coverage artifacts. Keeping it
-- separate from chunks_fts prevents duplicate coverage documents from
-- changing extraction BM25 statistics while allowing scoped Honcho retrieval
-- to survive raw-message pruning.
CREATE VIRTUAL TABLE IF NOT EXISTS message_coverage_fts USING fts5(
    content,
    content='',
    tokenize='porter unicode61'
);
-- Selective chunk and coverage FTS triggers are installed by migrations 038
-- and 043 (and healed at runtime). Defining them here would make schema-first
-- startup on a pre-v37 ``chunks`` table fail before ``chunk_kind`` exists.

-- Embedding vectors for chunks. JSON-encoded floats; cosine similarity
-- is computed in Python at query time.
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Semantic mirror for exact message occurrences.  Unlike raw `messages`, the
-- parent coverage artifact survives opt-in pruning, so semantic recall is
-- invariant across the retention boundary.  vec_messages is created lazily by
-- ensure_vec_table when sqlite-vec is installed.
CREATE TABLE IF NOT EXISTS message_embeddings (
    message_id INTEGER PRIMARY KEY,
    source_coverage_chunk_id TEXT NOT NULL,
    source_coverage_version TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL CHECK (dim > 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (
        message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE CASCADE
);

-- Embedding vectors for knowledge-graph edges. Keyed on the triple text
-- "{subject} {predicate} {object}" (not edge id) so derived edges, whose ids
-- churn every dream run, reuse a cached vector instead of re-embedding.
CREATE TABLE IF NOT EXISTS edge_embeddings (
    edge_text TEXT PRIMARY KEY,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Content-addressed embedding cache. Deduplicates API/model calls when two
-- chunks or edges share the same normalized text.
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    dim INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (text_hash, model)
);

-- Irrecoverable Phase-1 input loss.  Unlike provider/output quarantine this is
-- a property of the durable source artifact, not of a prompt version: once an
-- old extraction chunk has no exact claim-source manifest and its pre-upgrade
-- source bytes are gone, no future prompt can safely recreate that input.
-- Keeping this separate from processed_chunks makes the loss explicit without
-- counterfeiting a successful (possibly empty) extraction.
CREATE TABLE IF NOT EXISTS chunk_extraction_terminal_losses (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    reason TEXT NOT NULL CHECK (
        reason IN ('source_manifest_unrecoverable')
    ),
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chunk_extraction_terminal_losses_reason
    ON chunk_extraction_terminal_losses(reason);

-- Idempotency: each chunk processed at most once per prompt_version.
-- v53 generation bindings are registered before a live marker/outcome is
-- published. Legacy NULL bindings are retained only as untrusted history.
CREATE TABLE IF NOT EXISTS phase1_generations (
    generation_key TEXT PRIMARY KEY,
    extraction_cache_key TEXT NOT NULL,
    producer_identity_sha256 TEXT NOT NULL,
    identity_exact BOOLEAN NOT NULL CHECK (identity_exact IN (0, 1)),
    reuse_scope TEXT NOT NULL CHECK (
        reuse_scope IN ('durable', 'process_instance')
    ),
    binding_json TEXT NOT NULL CHECK (json_valid(binding_json)),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS processed_chunks (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    prompt_version TEXT NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    phase1_generation_key TEXT
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    PRIMARY KEY (chunk_id, prompt_version)
);
CREATE TRIGGER IF NOT EXISTS processed_chunks_terminal_loss_insert_guard
BEFORE INSERT ON processed_chunks
WHEN EXISTS (
    SELECT 1 FROM chunk_extraction_terminal_losses loss
    WHERE loss.chunk_id = new.chunk_id
)
BEGIN
    SELECT RAISE(ABORT, 'terminal extraction loss cannot be marked processed');
END;
CREATE TRIGGER IF NOT EXISTS processed_chunks_terminal_loss_update_guard
BEFORE UPDATE OF chunk_id, prompt_version ON processed_chunks
WHEN EXISTS (
    SELECT 1 FROM chunk_extraction_terminal_losses loss
    WHERE loss.chunk_id = new.chunk_id
)
BEGIN
    SELECT RAISE(ABORT, 'terminal extraction loss cannot be marked processed');
END;
CREATE TRIGGER IF NOT EXISTS chunk_extraction_terminal_loss_clear_processed
AFTER INSERT ON chunk_extraction_terminal_losses
BEGIN
    DELETE FROM processed_chunks WHERE chunk_id = new.chunk_id;
END;
CREATE TRIGGER IF NOT EXISTS chunk_extraction_terminal_loss_update_guard
BEFORE UPDATE ON chunk_extraction_terminal_losses
BEGIN
    SELECT RAISE(ABORT, 'terminal extraction loss is immutable');
END;

-- Latest successful source-validated claim interpretation for a chunk.  The
-- compact result hash makes an empty newer extraction portable authority: it
-- can retire an older store's observations without inventing a claim row.
CREATE TABLE IF NOT EXISTS kg_claim_extraction_outcomes (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE RESTRICT,
    prompt_version TEXT NOT NULL,
    prompt_generation INTEGER NOT NULL CHECK (prompt_generation >= 0),
    result_hash TEXT NOT NULL,
    succeeded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    phase1_generation_key TEXT
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT
);
CREATE TRIGGER IF NOT EXISTS kg_claim_extraction_outcomes_insert_guard
BEFORE INSERT ON kg_claim_extraction_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
  OR length(trim(new.prompt_version)) = 0
  OR new.prompt_generation < 0
  OR substr(new.result_hash, 1, 7) <> 'sha256:'
  OR length(new.result_hash) <> 71
  OR substr(new.result_hash, 8) GLOB '*[^0-9a-f]*'
BEGIN
    SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
END;
CREATE TRIGGER IF NOT EXISTS kg_claim_extraction_outcomes_update_guard
BEFORE UPDATE ON kg_claim_extraction_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
  OR length(trim(new.prompt_version)) = 0
  OR new.prompt_generation < 0
  OR substr(new.result_hash, 1, 7) <> 'sha256:'
  OR length(new.result_hash) <> 71
  OR substr(new.result_hash, 8) GLOB '*[^0-9a-f]*'
BEGIN
    SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
END;
CREATE TRIGGER IF NOT EXISTS kg_claim_extraction_outcomes_delete_guard
BEFORE DELETE ON kg_claim_extraction_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
END;

-- Consecutive failed extraction attempts per chunk (v28). A failed extraction
-- is HELD (no processed_chunks row) and retried next dream. At the configured
-- bound it remains explicitly unprocessed but selection quarantines it so a
-- permanently-broken chunk cannot consume dream_budget forever. Cleared on
-- success, so the count is consecutive failures.
CREATE TABLE IF NOT EXISTS chunk_extraction_attempts (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    prompt_version TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_failure_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_failure_reason TEXT,
    last_failure_details TEXT NOT NULL DEFAULT '[]',
    phase1_generation_key TEXT
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    PRIMARY KEY (chunk_id, prompt_version)
);

-- Entity canonicalization. surface forms map to a canonical id.
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias TEXT PRIMARY KEY,
    canonical TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_aliases_canonical ON entity_aliases(canonical);

-- Entity types: maps canonical entities to type labels.
CREATE TABLE IF NOT EXISTS entity_types (
    entity_canonical TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source_chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
    origin TEXT NOT NULL DEFAULT 'legacy_unattributed'
        CHECK (origin IN ('user', 'legacy_unattributed')),
    PRIMARY KEY (entity_canonical, type)
);
CREATE INDEX IF NOT EXISTS idx_entity_types_type ON entity_types(type);
CREATE INDEX IF NOT EXISTS idx_entity_types_entity ON entity_types(entity_canonical);

-- Entity properties: free-form key/value attributes for canonical entities
-- (e.g. language=python, runtime=node, category=build_tool). Populated
-- alongside entity_types during phase-1 extraction; one (entity, key) wins,
-- last write replaces. Source chunk is retained so a property can be traced
-- back to the LLM extraction that introduced it.
CREATE TABLE IF NOT EXISTS entity_properties (
    entity_canonical TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source_chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    origin TEXT NOT NULL DEFAULT 'legacy_unattributed'
        CHECK (origin IN ('user', 'legacy_unattributed')),
    PRIMARY KEY (entity_canonical, key)
);
CREATE INDEX IF NOT EXISTS idx_entity_properties_key ON entity_properties(key);
CREATE INDEX IF NOT EXISTS idx_entity_properties_value ON entity_properties(value);

-- Knowledge graph. Confidence is derived: (pos+1)/(pos+neg+2). Predicates locked.
CREATE TABLE IF NOT EXISTS knowledge_graph (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_canonical TEXT NOT NULL,
    predicate TEXT NOT NULL CHECK (predicate IN (
        'uses','depends_on','prefers','rejects','avoids',
        'replaces','conflicts_with','deploys_to','part_of','equivalent_to',
        'implements','contains','configured_with','requires_version',
        'runs_on','connects_to','generates','tested_by',
        'owns','located_in','participates_in','has_attribute'
    )),
    object_canonical TEXT NOT NULL,
    pos_evidence INTEGER NOT NULL DEFAULT 0,
    neg_evidence INTEGER NOT NULL DEFAULT 0,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_reinforced TIMESTAMP,
    -- Bi-temporal VALID time (distinct from the transaction-time columns above):
    -- valid_at = world date the fact became true, invalid_at = world date it was
    -- superseded (NULL = still valid). Added in migration 015; the validity index
    -- lives in that migration only (see its header for why it can't sit here).
    valid_at TIMESTAMP,
    invalid_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','stale','retracted')),
    derived BOOLEAN NOT NULL DEFAULT 0,
    UNIQUE(subject_canonical, predicate, object_canonical)
);
CREATE INDEX IF NOT EXISTS idx_kg_subject ON knowledge_graph(subject_canonical);
CREATE INDEX IF NOT EXISTS idx_kg_object ON knowledge_graph(object_canonical);
CREATE INDEX IF NOT EXISTS idx_kg_predicate ON knowledge_graph(predicate);
CREATE INDEX IF NOT EXISTS idx_kg_status ON knowledge_graph(status);

-- Per-source evidence so we keep many session refs per edge plus surface forms.
CREATE TABLE IF NOT EXISTS kg_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE RESTRICT,
    polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
    surface_subject TEXT,
    surface_object TEXT,
    value_text TEXT,
    value_numeric REAL,
    value_unit TEXT,
    temporal_scope TEXT,
    source_role TEXT CHECK (
        source_role IS NULL OR source_role IN ('user','assistant','system','tool')
    ),
    source_peer_id TEXT,
    source_workspace_id TEXT,
    -- v40 canonical rows cite one exact message and its immutable ordered
    -- coverage artifact. Legacy rows stay explicitly unattributed.
    evidence_kind TEXT NOT NULL DEFAULT 'extraction',
    evidence_weight INTEGER NOT NULL DEFAULT 1 CHECK (evidence_weight >= 1),
    weight_source TEXT NOT NULL DEFAULT 'legacy_default',
    extraction_prompt_version TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- First successful whole-chunk publication transaction for this revision.
    -- NULL means staged/unpublished canonical evidence.
    published_at TIMESTAMP,
    source_message_id INTEGER,
    source_session_id TEXT,
    source_created_at TIMESTAMP,
    source_event_at TEXT,
    source_coverage_chunk_id TEXT,
    source_coverage_version TEXT,
    provenance_status TEXT NOT NULL DEFAULT 'legacy_unattributed'
        CHECK (provenance_status IN ('canonical','legacy_unattributed')),
    interpretation_key TEXT NOT NULL DEFAULT 'legacy-unspecified',
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    is_current BOOLEAN NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    superseded_at TIMESTAMP,
    superseded_reason TEXT,
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_evidence_edge ON kg_evidence(edge_id);
CREATE INDEX IF NOT EXISTS idx_evidence_chunk ON kg_evidence(chunk_id);
-- v40 source indexes and provenance guards live in migration 040. Keeping
-- them out of this bootstrap file lets initialize() open an existing v39
-- table before the migration adds the referenced columns.

-- Evidence without a source chunk (explicit host actions) and quarantined
-- legacy counter deltas.  `counts_toward_confidence=0` preserves an unknown
-- historical delta for audit without allowing it to inflate current trust.
CREATE TABLE IF NOT EXISTS kg_evidence_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
    signal_key TEXT NOT NULL,
    signal_kind TEXT NOT NULL,
    polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
    evidence_weight INTEGER NOT NULL CHECK (evidence_weight >= 1),
    counts_toward_confidence BOOLEAN NOT NULL DEFAULT 1
        CHECK (counts_toward_confidence IN (0, 1)),
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(edge_id, signal_kind, signal_key)
);
CREATE INDEX IF NOT EXISTS idx_evidence_signals_edge
    ON kg_evidence_signals(edge_id);
-- v40 installs the signal mutation guards only after migration 036 has
-- materialized conservative legacy deltas. Startup heals the guard set once
-- the schema version is current.

-- v40 lifecycle storage is present in the baseline as well as migration 040.
-- Historical rebuild migrations temporarily drop these dependent triggers.
CREATE TABLE IF NOT EXISTS kg_edge_lifecycle (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
    event_key TEXT NOT NULL,
    event_kind TEXT NOT NULL CHECK (event_kind IN (
        'claim_assertion','manual_retraction','phase3_retraction',
        'value_supersession','legacy_state'
    )),
    direction INTEGER NOT NULL CHECK (direction IN (-1, 1)),
    event_at TEXT NOT NULL,
    source_evidence_id INTEGER REFERENCES kg_evidence(id) ON DELETE CASCADE,
    dependency_count INTEGER NOT NULL DEFAULT 0 CHECK (dependency_count >= 0),
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(edge_id, event_key)
);
CREATE INDEX IF NOT EXISTS idx_kg_edge_lifecycle_edge
    ON kg_edge_lifecycle(edge_id, event_at, event_key);

CREATE TABLE IF NOT EXISTS kg_lifecycle_dependencies (
    lifecycle_id INTEGER NOT NULL
        REFERENCES kg_edge_lifecycle(id) ON DELETE CASCADE,
    evidence_id INTEGER NOT NULL REFERENCES kg_evidence(id) ON DELETE CASCADE,
    PRIMARY KEY (lifecycle_id, evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_kg_lifecycle_dependencies_evidence
    ON kg_lifecycle_dependencies(evidence_id);

CREATE TRIGGER IF NOT EXISTS kg_edge_lifecycle_insert_guard
BEFORE INSERT ON kg_edge_lifecycle
BEGIN
    SELECT RAISE(ABORT, 'knowledge graph lifecycle events are internally managed')
    WHERE hymem_evidence_mutation_authorized() <> 1;
    SELECT RAISE(ABORT, 'invalid knowledge graph lifecycle event')
    WHERE NOT (
      new.event_at = COALESCE(
        hymem_normalize_iso_timestamp(new.event_at),
        '0001-01-01T00:00:00.000Z'
    )
    AND (
        (new.event_kind = 'claim_assertion' AND new.direction = 1
         AND new.source_evidence_id IS NOT NULL
         AND new.dependency_count = 0)
        OR (new.event_kind = 'manual_retraction'
            AND new.direction = -1 AND new.dependency_count = 0)
        OR (new.event_kind = 'value_supersession'
            AND new.direction = -1 AND new.dependency_count > 0
            AND new.source_evidence_id IS NULL)
        OR (new.event_kind = 'phase3_retraction'
            AND new.direction = -1 AND new.dependency_count > 0
            AND new.source_evidence_id IS NULL)
        OR (new.event_kind = 'legacy_state'
            AND new.source_evidence_id IS NULL
            AND new.dependency_count = 0)
    )
    AND (
        new.source_evidence_id IS NULL
        OR EXISTS (
            SELECT 1 FROM kg_evidence ev
            WHERE ev.id = new.source_evidence_id
              AND ev.edge_id = new.edge_id
              AND ev.provenance_status = 'canonical'
              AND (ev.is_current = 1
                   OR hymem_evidence_history_authorized() = 1)
              AND ev.polarity = new.direction
              AND ev.source_event_at = new.event_at
        )
      )
    );
END;
CREATE TRIGGER IF NOT EXISTS kg_edge_lifecycle_update_guard
BEFORE UPDATE ON kg_edge_lifecycle
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'knowledge graph lifecycle events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS kg_edge_lifecycle_delete_guard
BEFORE DELETE ON kg_edge_lifecycle
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'knowledge graph lifecycle events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_insert_guard
BEFORE INSERT ON kg_lifecycle_dependencies
BEGIN
    SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed')
    WHERE hymem_evidence_mutation_authorized() <> 1;
    SELECT RAISE(ABORT, 'invalid lifecycle evidence dependency')
    WHERE NOT EXISTS (
      SELECT 1
      FROM kg_edge_lifecycle lifecycle
      JOIN kg_evidence ev ON ev.id = new.evidence_id
      WHERE lifecycle.id = new.lifecycle_id
        AND lifecycle.direction = -1
        AND (ev.is_current = 1
             OR hymem_evidence_history_authorized() = 1)
        AND (
          (lifecycle.event_kind = 'phase3_retraction'
           AND lifecycle.edge_id = ev.edge_id AND ev.polarity = -1)
          OR (lifecycle.event_kind = 'value_supersession'
              AND ev.polarity = 1
              AND EXISTS (
                  SELECT 1
                  FROM knowledge_graph loser
                  JOIN knowledge_graph winner
                    ON winner.subject_canonical = loser.subject_canonical
                   AND winner.predicate = loser.predicate
                   AND winner.object_canonical <> loser.object_canonical
                  WHERE loser.id = lifecycle.edge_id
                    AND winner.id = ev.edge_id
                    AND winner.derived = 0
              ))
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_update_guard
BEFORE UPDATE ON kg_lifecycle_dependencies
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed');
END;
CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_delete_guard
BEFORE DELETE ON kg_lifecycle_dependencies
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed');
END;

-- Behavioral markers (explicit signals only).
CREATE TABLE IF NOT EXISTS behavioral_markers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('correction','preference','rejection','style')),
    statement TEXT NOT NULL,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    consolidated_at TIMESTAMP,
    phase1_generation_key TEXT
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_markers_consolidated ON behavioral_markers(consolidated_at);

-- Behavioral profile entries. Mirror of the Markdown section, structured.
CREATE TABLE IF NOT EXISTS profile_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('preference','avoidance','style','context')),
    text TEXT NOT NULL UNIQUE,
    pos_evidence INTEGER NOT NULL DEFAULT 1,
    neg_evidence INTEGER NOT NULL DEFAULT 0,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT NOT NULL DEFAULT 'legacy_unattributed'
        CHECK (source IN ('user', 'agent_inferred', 'legacy_unattributed'))
);

-- Honcho peer registry: maps Honcho peer_id → HyMem role.
-- Populated by the Honcho-compatible server when Hermes registers peers.
CREATE TABLE IF NOT EXISTS peers (
    id TEXT NOT NULL,
    workspace_id TEXT NOT NULL DEFAULT 'hermes',
    role TEXT NOT NULL DEFAULT 'user',
    metadata TEXT NOT NULL DEFAULT '{}',
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, workspace_id)
);

-- Session membership/configuration is distinct from workspace-wide peer
-- registration. In particular, adding a peer to one session must never
-- overwrite the peer's global registry row or imply membership elsewhere.
CREATE TABLE IF NOT EXISTS session_peers (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL,
    peer_id TEXT NOT NULL,
    configuration TEXT NOT NULL DEFAULT '{}',
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, workspace_id, peer_id),
    FOREIGN KEY (peer_id, workspace_id)
        REFERENCES peers(id, workspace_id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_session_peers_peer
    ON session_peers(workspace_id, peer_id, session_id);

-- Run lock so dreaming cycles don't overlap.
CREATE TABLE IF NOT EXISTS run_lock (
    name TEXT PRIMARY KEY,
    acquired_at TIMESTAMP NOT NULL,
    holder TEXT NOT NULL
);

-- Episodic memory: session summaries broken into named episodes.
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    participants TEXT NOT NULL DEFAULT '[]',
    start_message_id INTEGER,
    end_message_id INTEGER,
    outcome TEXT,
    key_entities TEXT NOT NULL DEFAULT '[]',
    -- Stable input-slice attribution for resumable v38 digest walks.  It lets
    -- a later slice of one oversized message coexist with earlier slices and
    -- scopes granularity supersession to the slice actually being replaced.
    digest_slice_key TEXT,
    -- All slices of one complete prompt/config walk share a generation.  An
    -- older generation is removed only after its replacement reaches the end,
    -- so failures cannot erase the last complete set of episodes. Rows whose
    -- generation differs from sessions.digest_published_generation are staged
    -- and excluded from every normal consumer until atomic publication.
    digest_generation TEXT,
    -- Exact message occurrences cited by this episode.  Numeric ranges remain
    -- compatibility metadata only and never authorize scoped disclosure.
    source_manifest_version TEXT,
    source_manifest_count INTEGER NOT NULL DEFAULT 0
        CHECK (source_manifest_count >= 0),
    source_manifest_hash TEXT,
    source_manifest_complete BOOLEAN NOT NULL DEFAULT 0
        CHECK (source_manifest_complete IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        (source_manifest_complete = 0
         AND source_manifest_count = 0
         AND source_manifest_hash IS NULL
         AND source_manifest_version IS NULL)
        OR
        (source_manifest_complete = 1
         AND source_manifest_version = 'episode-source-manifest-v1'
         AND source_manifest_count > 0
         AND length(source_manifest_hash) = 71
         AND source_manifest_hash GLOB 'sha256:*')
    )
);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);

CREATE TABLE IF NOT EXISTS episode_source_occurrences (
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    source_message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_role TEXT NOT NULL
        CHECK (source_role IN ('user','assistant','system','tool')),
    source_peer_id TEXT,
    source_workspace_id TEXT,
    source_created_at TIMESTAMP,
    source_coverage_chunk_id TEXT NOT NULL,
    source_coverage_version TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    PRIMARY KEY (episode_id, ordinal),
    UNIQUE (episode_id, source_session_id, source_message_id),
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_episode_source_occurrence
    ON episode_source_occurrences(source_session_id, source_message_id);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    title, summary,
    content='episodes', content_rowid='rowid',
    tokenize='porter unicode61'
);

-- The v38 migration replaces these compatibility triggers with publication-
-- aware variants. They remain column-agnostic here because schema.sql runs
-- before migrations and an old store's existing tables do not yet have the
-- v38 generation columns.
CREATE TRIGGER IF NOT EXISTS episodes_fts_insert AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, title, summary) VALUES (new.rowid, new.title, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS episodes_fts_delete AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, title, summary) VALUES ('delete', old.rowid, old.title, old.summary);
END;
-- UPSERTs against episodes (re-dreams updating title/summary for the same
-- message-range id) must keep the FTS shadow table in sync.
CREATE TRIGGER IF NOT EXISTS episodes_fts_update AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, title, summary) VALUES ('delete', old.rowid, old.title, old.summary);
    INSERT INTO episodes_fts(rowid, title, summary) VALUES (new.rowid, new.title, new.summary);
END;

-- Episode-level embeddings. Keyed by stable episode id (see
-- persist_episodes). The text_hash column points into embedding_cache so a
-- re-dreamed episode whose title/summary text changed gets re-embedded.
CREATE TABLE IF NOT EXISTS episode_embeddings (
    episode_id TEXT PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- RAPTOR cross-session aggregation nodes (schema v16; hierarchy in v17). A
-- level-0 node fuses a cluster of episodes (connected components over
-- embedding-OR-entity overlap) that span multiple sessions, so a synthesis
-- v56: exact secret-free identity of the LLM producer and executable
-- aggregation request/parser contract. This registry is local rebuild state,
-- not portable source material.
CREATE TABLE IF NOT EXISTS aggregation_generations (
    generation_key TEXT PRIMARY KEY CHECK (
        length(generation_key) = 96
        AND substr(generation_key,1,32) = 'hymem-aggregation-generation-v1:'
        AND substr(generation_key,33) NOT GLOB '*[^0-9a-f]*'
    ),
    material_config_version TEXT NOT NULL CHECK (
        length(material_config_version) = 92
        AND substr(material_config_version,1,28) =
            'aggregation-build-config-v1:'
        AND substr(material_config_version,29) NOT GLOB '*[^0-9a-f]*'
    ),
    producer_identity_sha256 TEXT NOT NULL CHECK (
        length(producer_identity_sha256) = 71
        AND producer_identity_sha256 GLOB 'sha256:*'
    ),
    identity_exact BOOLEAN NOT NULL CHECK (identity_exact IN (0,1)),
    reuse_scope TEXT NOT NULL CHECK (
        reuse_scope IN ('durable','process_instance')
        AND ((identity_exact=1 AND reuse_scope='durable')
             OR (identity_exact=0 AND reuse_scope='process_instance'))
    ),
    binding_json TEXT NOT NULL CHECK (json_valid(binding_json)),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- question reads a handful of cluster summaries instead of dozens of raw turns.
-- Levels >= 1 (v17) are RAPTOR rollups whose member_episode_ids hold CHILD ids
-- (lower-level node ids and/or pass-through episode ids), recursing until one
-- is_root digest node — the standing "what do you know about me" summary
-- exposed via HyMem.digest(). Only level 0 enters query-time retrieval. The
-- whole layer is additive and enabled by default; callers can opt out with
-- cfg.aggregation_nodes_enabled = False.
-- Rebuilt from scratch each dream — membership is a pure function of the
-- current episodes — so there is no stable-id UPSERT churn; the id is a content
-- hash of members, which also keys reuse of an unchanged node's LLM fusion.
CREATE TABLE IF NOT EXISTS aggregation_nodes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    member_episode_ids TEXT NOT NULL DEFAULT '[]',
    session_ids TEXT NOT NULL DEFAULT '[]',
    n_members INTEGER NOT NULL DEFAULT 0,
    n_sessions INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level INTEGER NOT NULL DEFAULT 0,
    is_root INTEGER NOT NULL DEFAULT 0,
    -- Flattened exact source set behind the effective fusion input.  Existing
    -- pre-v45 rows default to incomplete and remain non-authoritative physical
    -- history; no public/provider read serves them before an exact rebuild.
    source_manifest_version TEXT,
    source_manifest_count INTEGER NOT NULL DEFAULT 0
        CHECK (source_manifest_count >= 0),
    source_manifest_hash TEXT,
    source_manifest_complete BOOLEAN NOT NULL DEFAULT 0
        CHECK (source_manifest_complete IN (0, 1)),
    -- Hash of ordered member ids + exact rendered title/summary + each member's
    -- source-manifest state (and root-only extra prompt inputs).
    input_fingerprint TEXT,
    -- v55: typed effective-input proof.  ``member_episode_ids`` remains a
    -- compatibility rendering only; this manifest says whether each member is
    -- an episode, child node, or an explicitly typed authoritative anchor.
    input_manifest_version TEXT,
    input_manifest_count INTEGER NOT NULL DEFAULT 0
        CHECK (input_manifest_count >= 0),
    input_manifest_hash TEXT,
    input_manifest_complete BOOLEAN NOT NULL DEFAULT 0
        CHECK (input_manifest_complete IN (0, 1)),
    node_kind TEXT CHECK (
        node_kind IS NULL OR node_kind IN ('cluster','rollup','root')
    ),
    output_hash TEXT,
    -- One clean structural publication owns every servable node.  Candidate
    -- rows may exist while a build is pending, but readers accept only the
    -- exact set named by aggregation_publication_state.
    publication_id TEXT,
    build_config_version TEXT,
    aggregation_generation_key TEXT
        REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT,
    aggregation_request_hash TEXT CHECK (
        aggregation_request_hash IS NULL OR (
            length(aggregation_request_hash)=71
            AND aggregation_request_hash GLOB 'sha256:*'
        )
    ),
    CHECK (
        (source_manifest_complete = 0
         AND source_manifest_count = 0
         AND source_manifest_hash IS NULL
         AND (source_manifest_version IS NULL OR
              source_manifest_version = 'aggregation-source-manifest-v1')
         AND (input_fingerprint IS NULL OR
              (length(input_fingerprint) = 71
               AND input_fingerprint GLOB 'sha256:*')))
        OR
        (source_manifest_complete = 1
         AND source_manifest_version = 'aggregation-source-manifest-v1'
         AND source_manifest_count > 0
         AND length(source_manifest_hash) = 71
         AND source_manifest_hash GLOB 'sha256:*'
         AND length(input_fingerprint) = 71
         AND input_fingerprint GLOB 'sha256:*')
    )
);

CREATE TABLE IF NOT EXISTS aggregation_node_source_occurrences (
    node_id TEXT NOT NULL REFERENCES aggregation_nodes(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    source_message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_role TEXT NOT NULL
        CHECK (source_role IN ('user','assistant','system','tool')),
    source_peer_id TEXT,
    source_workspace_id TEXT,
    source_created_at TIMESTAMP,
    source_coverage_chunk_id TEXT NOT NULL,
    source_coverage_version TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    PRIMARY KEY (node_id, ordinal),
    UNIQUE (node_id, source_session_id, source_message_id),
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_aggregation_source_occurrence
    ON aggregation_node_source_occurrences(source_session_id, source_message_id);

-- v55: exact typed effective inputs and their per-input lossless proofs.
-- These rows preserve type across the hierarchy, so a text id that happens to
-- exist in both episodes and aggregation_nodes is never resolved by guesswork.
CREATE TABLE IF NOT EXISTS aggregation_node_inputs (
    node_id TEXT NOT NULL REFERENCES aggregation_nodes(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    input_kind TEXT NOT NULL CHECK (input_kind IN (
        'episode','aggregation_node','user_profile',
        'knowledge_graph','narrative_fact'
    )),
    source_key TEXT NOT NULL CHECK (length(source_key) > 0),
    source_ref_json TEXT NOT NULL CHECK (json_valid(source_ref_json)),
    payload_hash TEXT NOT NULL CHECK (
        length(payload_hash) = 71 AND payload_hash GLOB 'sha256:*'
    ),
    authority_hash TEXT NOT NULL CHECK (
        length(authority_hash) = 71 AND authority_hash GLOB 'sha256:*'
    ),
    source_manifest_count INTEGER NOT NULL CHECK (source_manifest_count > 0),
    source_manifest_hash TEXT NOT NULL CHECK (
        length(source_manifest_hash) = 71
        AND source_manifest_hash GLOB 'sha256:*'
    ),
    PRIMARY KEY (node_id, ordinal),
    UNIQUE (node_id, input_kind, source_key)
);

CREATE TABLE IF NOT EXISTS aggregation_node_input_sources (
    node_id TEXT NOT NULL,
    input_ordinal INTEGER NOT NULL CHECK (input_ordinal >= 0),
    source_ordinal INTEGER NOT NULL CHECK (source_ordinal >= 0),
    source_message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_role TEXT NOT NULL
        CHECK (source_role IN ('user','assistant','system','tool')),
    source_peer_id TEXT,
    source_workspace_id TEXT,
    source_created_at TIMESTAMP,
    source_coverage_chunk_id TEXT NOT NULL,
    source_coverage_version TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    PRIMARY KEY (node_id, input_ordinal, source_ordinal),
    UNIQUE (
        node_id, input_ordinal, source_session_id, source_message_id
    ),
    FOREIGN KEY (node_id, input_ordinal)
        REFERENCES aggregation_node_inputs(node_id, ordinal) ON DELETE CASCADE,
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_aggregation_input_source_occurrence
    ON aggregation_node_input_sources(source_session_id, source_message_id);

-- v55: the only authority for serving a root digest.  Build-attempt counters
-- remain operational in aggregation_build_health; this small semantic
-- projection is locally attested because changing it changes visibility.
-- Aggregation material is a rebuildable local cache and is not portable.
CREATE TABLE IF NOT EXISTS aggregation_publication_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    publication_id TEXT NOT NULL CHECK (
        length(publication_id) = 71 AND publication_id GLOB 'sha256:*'
    ),
    config_version TEXT NOT NULL CHECK (
        length(config_version) = 92
        AND substr(config_version,1,28) = 'aggregation-build-config-v1:'
    ),
    cluster_min_members INTEGER NOT NULL CHECK (cluster_min_members >= 1),
    cluster_min_sessions INTEGER NOT NULL CHECK (cluster_min_sessions >= 1),
    anchor_fact_cap INTEGER NOT NULL CHECK (anchor_fact_cap >= 0),
    root_node_id TEXT,
    node_count INTEGER NOT NULL CHECK (node_count >= 0),
    node_set_hash TEXT NOT NULL CHECK (
        length(node_set_hash) = 71 AND node_set_hash GLOB 'sha256:*'
    ),
    published_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP CHECK (
        typeof(published_at) = 'text'
        AND length(published_at) = 19
        AND strftime('%Y-%m-%d %H:%M:%S', published_at) IS NOT NULL
        AND strftime('%Y-%m-%d %H:%M:%S', published_at) = published_at
    ),
    aggregation_generation_key TEXT
        REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT,
    request_contract_sha256 TEXT CHECK (
        request_contract_sha256 IS NULL OR (
            length(request_contract_sha256)=71
            AND request_contract_sha256 GLOB 'sha256:*'
        )
    )
);

CREATE VIRTUAL TABLE IF NOT EXISTS aggregation_nodes_fts USING fts5(
    title, summary,
    content='aggregation_nodes', content_rowid='rowid',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS aggregation_nodes_fts_insert AFTER INSERT ON aggregation_nodes BEGIN
    INSERT INTO aggregation_nodes_fts(rowid, title, summary) VALUES (new.rowid, new.title, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS aggregation_nodes_fts_delete AFTER DELETE ON aggregation_nodes BEGIN
    INSERT INTO aggregation_nodes_fts(aggregation_nodes_fts, rowid, title, summary) VALUES ('delete', old.rowid, old.title, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS aggregation_nodes_fts_update AFTER UPDATE ON aggregation_nodes BEGIN
    INSERT INTO aggregation_nodes_fts(aggregation_nodes_fts, rowid, title, summary) VALUES ('delete', old.rowid, old.title, old.summary);
    INSERT INTO aggregation_nodes_fts(rowid, title, summary) VALUES (new.rowid, new.title, new.summary);
END;

-- Node-summary embeddings, keyed by node id. Retrieval does a Python-cosine
-- scan over these (no vec0 table) since the node count is small and query
-- exposure is narrowly ability-gated, keeping vec0 plumbing limited to
-- chunks/edges/episodes.
CREATE TABLE IF NOT EXISTS aggregation_node_embeddings (
    node_id TEXT PRIMARY KEY REFERENCES aggregation_nodes(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Durable leaf-set watermark (schema v30) for the leftover-displacement term
-- of the deficit model. Replaces a module global that only ever compared
-- against the previous dream IN THE SAME PROCESS, which on a box that starts a
-- fresh process per dream made `aggregation_leaf_changed` unreadable on 175 of
-- 187 rows. Written inside the node-persist transaction, so the watermark
-- advances only when the dream that consumed the leaf set landed; a store that
-- has never aggregated has no row, and that reads as NULL (unattributed), not
-- as an unchanged leaf set.
CREATE TABLE IF NOT EXISTS aggregation_leaf_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    fingerprint TEXT NOT NULL,
    n_leaves INTEGER NOT NULL,
    -- v34: the id list itself, so the shift can be sized and not merely
    -- detected. NULL on a pre-v34 watermark row, which must read as
    -- "unattributable" and never as an empty set.
    leaf_ids TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Typed user-profile slots (schema v18, Stage 1 / P4). Durable personal facts
-- the tech-domain knowledge-graph vocabulary can never hold (role, name,
-- employer, location, ...), extracted from USER turns only during dreaming
-- under the CLOSED slot vocabulary the CHECK enforces — the LLM cannot invent
-- a slot. slot_key parameterizes a slot ('relationship' is keyed by the other
-- person); NULL for unkeyed slots. Bi-temporal like knowledge_graph (v15
-- semantics): valid_at = world date the fact became true (evidence message
-- created_at), invalid_at = world date a conflicting value on the same
-- (slot, slot_key) superseded it (NULL = still valid). Consumed additively by
-- the digest's VERIFIED FACTS anchor, augment()'s ctx.user_profile tier, and
-- HyMem.profile(). The index is safe here (unlike migration-added columns)
-- because the table is created whole in this same script.
CREATE TABLE IF NOT EXISTS user_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot TEXT NOT NULL CHECK (slot IN (
        'role','name','employer','location','language','relationship',
        'possession','age_birthday','health_condition','recurring_activity'
    )),
    slot_key TEXT,
    value TEXT NOT NULL,
    evidence_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    -- Durable provenance survives raw-message retention. evidence_message_id
    -- remains the compatibility live FK; these source fields are authoritative
    -- after that FK is nulled by ON DELETE SET NULL.
    source_message_id INTEGER,
    source_session_id TEXT,
    source_created_at TIMESTAMP,
    confidence REAL NOT NULL DEFAULT 1.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    valid_at TIMESTAMP,
    invalid_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_profile_active
    ON user_profile(slot, slot_key, invalid_at);

-- Validated profile output is staged per bounded source slice. Values are
-- redacted before they enter this table. A caught-up transaction publishes all
-- rows through persist_user_profile, flips the session marker, and clears the
-- stage; incomplete/failed walks never alter the consumer-visible profile.
CREATE TABLE IF NOT EXISTS profile_staging (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    generation TEXT NOT NULL,
    slice_key TEXT NOT NULL,
    items_json TEXT NOT NULL CHECK (json_valid(items_json)),
    start_message_id INTEGER,
    start_message_offset INTEGER NOT NULL DEFAULT 0
        CHECK (start_message_offset >= 0),
    end_message_id INTEGER,
    cursor_before_message_id INTEGER,
    cursor_before_partial_message_id INTEGER,
    cursor_before_offset INTEGER NOT NULL DEFAULT 0
        CHECK (cursor_before_offset >= 0),
    cursor_after_message_id INTEGER,
    cursor_after_partial_message_id INTEGER,
    cursor_after_offset INTEGER NOT NULL DEFAULT 0
        CHECK (cursor_after_offset >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, generation, slice_key)
);
CREATE INDEX IF NOT EXISTS idx_profile_staging_generation
    ON profile_staging(session_id, generation);

-- Private digest output is retained for only the active bounded-slice walk.
-- The completed publication and its ordinary consumer tables remain intact.
CREATE TABLE IF NOT EXISTS digest_staging (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    generation TEXT NOT NULL,
    slice_key TEXT NOT NULL,
    summary TEXT NOT NULL CHECK (length(summary) <= 500),
    procedures_json TEXT NOT NULL CHECK (json_valid(procedures_json)),
    episodes_json TEXT NOT NULL CHECK (json_valid(episodes_json)),
    source_sha256 TEXT NOT NULL CHECK (length(source_sha256) = 64),
    cursor_before_message_id INTEGER,
    cursor_before_partial_message_id INTEGER,
    cursor_before_offset INTEGER NOT NULL CHECK (cursor_before_offset >= 0),
    cursor_after_message_id INTEGER,
    cursor_after_partial_message_id INTEGER,
    cursor_after_offset INTEGER NOT NULL CHECK (cursor_after_offset >= 0),
    PRIMARY KEY (session_id, generation, slice_key)
);

-- Procedural memory: step-by-step workflows extracted from conversations.
CREATE TABLE IF NOT EXISTS procedures (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    steps TEXT NOT NULL DEFAULT '[]',
    triggers TEXT NOT NULL DEFAULT '[]',
    entities_involved TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','stale')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_procedures_session ON procedures(session_id);
CREATE INDEX IF NOT EXISTS idx_procedures_entities ON procedures(entities_involved);
-- Explicit, exact-content ownership for complete local digest publications.
-- Unknown legacy/manual rows remain outside this ledger.
CREATE TABLE IF NOT EXISTS procedure_digest_publications (
    procedure_id TEXT PRIMARY KEY REFERENCES procedures(id) ON DELETE CASCADE,
    generation TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL CHECK (length(payload_sha256) = 64),
    retired INTEGER NOT NULL DEFAULT 0 CHECK (retired IN (0,1))
);
-- NOTE: the index on procedures.status lives ONLY in migration
-- 010_procedure_status.sql, never here. schema.sql runs via executescript()
-- BEFORE migrations, so on an existing pre-v10 DB the `procedures` table is
-- already present (CREATE TABLE IF NOT EXISTS is a no-op) and lacks `status`;
-- a `CREATE INDEX ... ON procedures(status)` here would crash with
-- "no such column: status". General rule: any index/constraint referencing a
-- migration-added column belongs in the migration file only.

CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts USING fts5(
    name, description, steps,
    content='procedures', content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS procedures_fts_insert AFTER INSERT ON procedures BEGIN
    INSERT INTO procedures_fts(rowid, name, description, steps) VALUES (new.rowid, new.name, new.description, new.steps);
END;
CREATE TRIGGER IF NOT EXISTS procedures_fts_delete AFTER DELETE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, name, description, steps) VALUES ('delete', old.rowid, old.name, old.description, old.steps);
END;
CREATE TRIGGER IF NOT EXISTS procedures_fts_update AFTER UPDATE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, name, description, steps) VALUES ('delete', old.rowid, old.name, old.description, old.steps);
    INSERT INTO procedures_fts(rowid, name, description, steps) VALUES (new.rowid, new.name, new.description, new.steps);
END;

-- Retraction audit: records the source and triple associated with automatic,
-- manual, or dedup corrections. These values are never prompt instructions.
CREATE TABLE IF NOT EXISTS extraction_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
    chunk_text_snippet TEXT NOT NULL,
    extracted_subject TEXT NOT NULL,
    extracted_predicate TEXT NOT NULL,
    extracted_object TEXT NOT NULL,
    feedback_type TEXT NOT NULL DEFAULT 'retracted',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON extraction_feedback(created_at);

-- Persistent token-overlap index for entity expansion in augment().
-- Rebuilt atomically at the end of every dream cycle; updated incrementally
-- by merge_canonical and retract_edge. Empty table signals a cold start —
-- build_token_overlap_index will scan canonicals and repopulate it.
CREATE TABLE IF NOT EXISTS token_overlap_index (
    token TEXT NOT NULL,
    canonical TEXT NOT NULL,
    PRIMARY KEY (token, canonical)
);
CREATE INDEX IF NOT EXISTS idx_token_overlap_token ON token_overlap_index(token);

-- Per-cycle dreaming run record. Populated by runner.run_dreaming for every
-- invocation (success, lock-skip, or error) so operators can observe cadence
-- and extraction quality over time.
CREATE TABLE IF NOT EXISTS dream_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    sessions_processed INTEGER NOT NULL DEFAULT 0,
    chunks_seen INTEGER NOT NULL DEFAULT 0,
    chunks_processed INTEGER NOT NULL DEFAULT 0,
    -- v49: exact Phase-1 provider attempts (including raised calls), and
    -- whether its soft cycle ceiling left an actionable chunk untouched.
    chunk_extraction_completion_calls INTEGER NOT NULL DEFAULT 0
        CHECK(chunk_extraction_completion_calls >= 0),
    chunk_extraction_provider_attempts INTEGER NOT NULL DEFAULT 0
        CHECK(chunk_extraction_provider_attempts >= 0),
    extraction_provider_attempt_budget_exhausted INTEGER NOT NULL DEFAULT 0
        CHECK(extraction_provider_attempt_budget_exhausted IN (0, 1)),
    -- v50: sessions whose ordered lossless source failed local validation in
    -- this cycle. The durable per-session ledger carries exact identities.
    coverage_integrity_failures INTEGER NOT NULL DEFAULT 0
        CHECK(coverage_integrity_failures >= 0),
    chunks_embedded INTEGER NOT NULL DEFAULT 0,
    edges_embedded INTEGER NOT NULL DEFAULT 0,
    triples_extracted INTEGER NOT NULL DEFAULT 0,
    markers_extracted INTEGER NOT NULL DEFAULT 0,
    aggregation_nodes_built INTEGER NOT NULL DEFAULT 0,
    aggregation_nodes_reused INTEGER NOT NULL DEFAULT 0,
    aggregation_fusion_failures INTEGER NOT NULL DEFAULT 0,
    -- v51: a total aggregation-build exception is separately attributable
    -- and is also reflected as a nonzero fusion-failure unit by the runner.
    aggregation_build_exceptions INTEGER NOT NULL DEFAULT 0
        CHECK(aggregation_build_exceptions >= 0),
    aggregation_config_version TEXT CHECK (
        aggregation_config_version IS NULL OR (
            length(aggregation_config_version) = 92
            AND substr(aggregation_config_version, 1, 28) =
                'aggregation-build-config-v1:'
            AND substr(aggregation_config_version, 29) NOT GLOB '*[^0-9a-f]*'
        )
    ),
    aggregation_input_episodes INTEGER NOT NULL DEFAULT 0,
    aggregation_blocking TEXT NOT NULL DEFAULT '',
    -- v29 deficit attribution (renumbered from 027): NULL = unattributed, NOT
    -- a fixed point. The fixed-point signature is level0_missed=0 AND
    -- leaf_changed=0 alongside a nonzero built count, so pre-v29 rows stay
    -- NULL rather than being backfilled into counterfeit fixed points.
    aggregation_level0_missed INTEGER,
    aggregation_leaf_changed INTEGER,
    -- v31 structural rebuild forecast. predicted counts nodes whose (level,
    -- member set) is absent from the previous tree; residual = actual -
    -- predicted counts nodes that kept their membership and still missed the
    -- fusion cache, which is an id-keying defect and reads on ONE dream.
    aggregation_predicted_rebuild INTEGER,
    aggregation_keying_residual INTEGER,
    aggregation_facts_rekey INTEGER,
    -- v32 effective layer state at dream start ('enabled'/'disabled'), so
    -- "layer on, nothing to do" is distinguishable from "layer off" per row --
    -- the defect that silently zeroed the flip-watch twice. NULL = pre-v32.
    aggregation_effective TEXT,
    -- v33 rebuild decomposition by tree level. Sums to (built - reused) by
    -- construction, so it is self-checking. Splits the leaf term from the tree
    -- term, which aggregation_leaf_changed (binary) cannot: it decides whether
    -- a low-reuse leaf-changed row is a benign digest cascade or the windowing
    -- confinement leaking. NULL = unattributed, never backfilled.
    aggregation_rebuilt_level0 INTEGER,
    aggregation_rebuilt_rollup INTEGER,
    aggregation_rebuilt_root INTEGER,
    -- v34: symmetric difference of the digest leaf set against the previous
    -- dream's, the continuous quantity aggregation_leaf_changed abbreviates.
    aggregation_leaf_added INTEGER,
    aggregation_leaf_removed INTEGER,
    -- v25 digest attribution: a per-session digest that raises or returns an
    -- unparseable payload is logged and skipped (one bad session must not abort
    -- a dream), and episode creation can stall silently while chunks keep
    -- arriving — the 2026-07-30 starvation bug. These make both visible without
    -- a join against episodes.
    digest_failures INTEGER NOT NULL DEFAULT 0,
    digest_quarantined INTEGER NOT NULL DEFAULT 0,
    episodes_created INTEGER NOT NULL DEFAULT 0,
    -- v26 facts attribution, mirroring the digest counters above: a stalled or
    -- failing narrative-facts extractor must be a one-line read.
    facts_extracted INTEGER NOT NULL DEFAULT 0,
    fact_failures INTEGER NOT NULL DEFAULT 0,
    -- v39 profile attribution. A malformed/invalid output increments failures
    -- and holds its independent source cursor; valid empty is not a failure.
    profile_items_extracted INTEGER NOT NULL DEFAULT 0,
    profile_failures INTEGER NOT NULL DEFAULT 0,
    skipped_locked INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    aggregation_generation_key TEXT
        REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_dream_runs_started ON dream_runs(started_at);

-- v51: singleton, bounded aggregation attempt state. The active attempt is
-- marked before the fallible build boundary and cleared only by an applicable
-- zero-failure result. Prompt text, source text, model/key/endpoint data, and
-- exception strings are forbidden by construction; only config hashes, enum
-- diagnostics, counters, and timestamps survive a restart.
CREATE TABLE IF NOT EXISTS aggregation_build_health (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    last_success_config_version TEXT CHECK (
        last_success_config_version IS NULL OR (
            length(last_success_config_version) = 92
            AND substr(last_success_config_version, 1, 28) =
                'aggregation-build-config-v1:'
            AND substr(last_success_config_version, 29)
                NOT GLOB '*[^0-9a-f]*'
        )
    ),
    last_success_at TIMESTAMP,
    pending_config_version TEXT CHECK (
        pending_config_version IS NULL OR (
            length(pending_config_version) = 92
            AND substr(pending_config_version, 1, 28) =
                'aggregation-build-config-v1:'
            AND substr(pending_config_version, 29)
                NOT GLOB '*[^0-9a-f]*'
        )
    ),
    pending_attempts INTEGER NOT NULL DEFAULT 0
        CHECK(pending_attempts BETWEEN 0 AND 2147483647),
    pending_caught_exceptions INTEGER NOT NULL DEFAULT 0
        CHECK(pending_caught_exceptions BETWEEN 0 AND 2147483647),
    pending_fusion_failures INTEGER NOT NULL DEFAULT 0
        CHECK(pending_fusion_failures BETWEEN 0 AND 2147483647),
    first_pending_at TIMESTAMP,
    last_attempt_at TIMESTAMP,
    total_caught_exceptions INTEGER NOT NULL DEFAULT 0
        CHECK(total_caught_exceptions BETWEEN 0 AND 2147483647),
    total_fusion_failures INTEGER NOT NULL DEFAULT 0
        CHECK(total_fusion_failures BETWEEN 0 AND 2147483647),
    superseded_pending_configs INTEGER NOT NULL DEFAULT 0
        CHECK(superseded_pending_configs BETWEEN 0 AND 2147483647),
    last_failure_config_version TEXT CHECK (
        last_failure_config_version IS NULL OR (
            length(last_failure_config_version) = 92
            AND substr(last_failure_config_version, 1, 28) =
                'aggregation-build-config-v1:'
            AND substr(last_failure_config_version, 29)
                NOT GLOB '*[^0-9a-f]*'
        )
    ),
    last_failure_kind TEXT CHECK (
        last_failure_kind IS NULL OR last_failure_kind IN (
            'exception', 'fusion_failure', 'exception_and_fusion'
        )
    ),
    last_failure_at TIMESTAMP,
    last_success_generation_key TEXT
        REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT,
    pending_generation_key TEXT
        REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT,
    last_failure_generation_key TEXT
        REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT,
    CHECK (
        (last_success_config_version IS NULL AND last_success_at IS NULL)
        OR
        (last_success_config_version IS NOT NULL AND last_success_at IS NOT NULL)
    ),
    CHECK (
        (
            pending_config_version IS NULL
            AND pending_attempts = 0
            AND pending_caught_exceptions = 0
            AND pending_fusion_failures = 0
            AND first_pending_at IS NULL
            AND last_attempt_at IS NULL
        )
        OR
        (
            pending_config_version IS NOT NULL
            AND pending_attempts >= 1
            AND first_pending_at IS NOT NULL
            AND last_attempt_at IS NOT NULL
        )
    ),
    CHECK (
        (
            last_failure_config_version IS NULL
            AND last_failure_kind IS NULL
            AND last_failure_at IS NULL
        )
        OR
        (
            last_failure_config_version IS NOT NULL
            AND last_failure_kind IS NOT NULL
            AND last_failure_at IS NOT NULL
        )
    )
);

-- v23: `always_on` Rules as a first-class node type (Idea B). Standing
-- behavioral imperatives ("always run the tests before pushing") injected into
-- every augment() context via ctx.rules; scope='contextual' rules fire only on
-- trigger_entities overlap with matched_entities. Bi-temporal like
-- knowledge_graph / user_profile (a contradicting rule closes invalid_at rather
-- than overwriting); text UNIQUE so re-assert reinforces. See hymem/rules.py.
-- Constraint (additional_planning.md §0): NOT fed into the RAPTOR digest anchor.
-- v46 authoritative narrative facts. Each source slice is bound to an exact
-- lossless occurrence manifest and an append-only extraction revision ledger.
-- The narrative_facts row is the guarded current projection; successful
-- replay can retract, correct, or resurrect a payload while lifecycle rows
-- retain history. Explicit fact dates are valid-time coordinates and relative
-- references remain NULL. Transaction time lives in succeeded_at/recorded_at.
CREATE TABLE IF NOT EXISTS fact_extraction_outcomes (
    slice_key TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    prompt_version TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    cursor_before_message_id INTEGER,
    cursor_before_partial_message_id INTEGER,
    cursor_before_offset INTEGER NOT NULL DEFAULT 0
        CHECK (cursor_before_offset >= 0),
    cursor_after_message_id INTEGER,
    cursor_after_partial_message_id INTEGER,
    cursor_after_offset INTEGER NOT NULL DEFAULT 0
        CHECK (cursor_after_offset >= 0),
    generation INTEGER NOT NULL CHECK (generation >= 1),
    outcome_status TEXT NOT NULL CHECK (outcome_status IN ('success','empty')),
    result_hash TEXT NOT NULL,
    source_manifest_version TEXT,
    source_manifest_count INTEGER NOT NULL DEFAULT 0
        CHECK (source_manifest_count >= 0),
    source_manifest_hash TEXT,
    source_manifest_complete BOOLEAN NOT NULL DEFAULT 0
        CHECK (source_manifest_complete IN (0, 1)),
    succeeded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fact_outcome_session
    ON fact_extraction_outcomes(session_id, cursor_after_message_id);
CREATE INDEX IF NOT EXISTS idx_fact_outcome_before_cursor
    ON fact_extraction_outcomes(
        session_id, cursor_before_message_id,
        cursor_before_partial_message_id, cursor_before_offset
    );
CREATE INDEX IF NOT EXISTS idx_fact_outcome_after_cursor
    ON fact_extraction_outcomes(
        session_id, cursor_after_message_id,
        cursor_after_partial_message_id, cursor_after_offset
    );
CREATE INDEX IF NOT EXISTS idx_fact_outcome_chain_order
    ON fact_extraction_outcomes(
        session_id,
        COALESCE(cursor_before_partial_message_id, cursor_before_message_id, -1),
        CASE WHEN cursor_before_partial_message_id IS NULL THEN 1 ELSE 0 END,
        cursor_before_offset, slice_key
    );
CREATE INDEX IF NOT EXISTS idx_fact_outcome_replay_v46
    ON fact_extraction_outcomes(
        session_id, source_manifest_complete, prompt_version
    );

CREATE TABLE IF NOT EXISTS fact_extraction_revisions (
    slice_key TEXT NOT NULL REFERENCES fact_extraction_outcomes(slice_key)
        ON DELETE CASCADE,
    generation INTEGER NOT NULL CHECK (generation >= 1),
    prompt_version TEXT NOT NULL,
    outcome_status TEXT NOT NULL CHECK (outcome_status IN ('success','empty')),
    result_hash TEXT NOT NULL,
    succeeded_at TIMESTAMP NOT NULL,
    PRIMARY KEY (slice_key, generation)
);

CREATE TABLE IF NOT EXISTS fact_extraction_source_occurrences (
    slice_key TEXT NOT NULL REFERENCES fact_extraction_outcomes(slice_key)
        ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    source_message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_role TEXT NOT NULL
        CHECK (source_role IN ('user','assistant','system','tool')),
    source_peer_id TEXT,
    source_workspace_id TEXT,
    source_created_at TIMESTAMP,
    source_coverage_chunk_id TEXT NOT NULL,
    source_coverage_version TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    PRIMARY KEY (slice_key, ordinal),
    UNIQUE (slice_key, source_session_id, source_message_id),
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_fact_source_occurrence
    ON fact_extraction_source_occurrences(source_session_id, source_message_id);

CREATE TABLE IF NOT EXISTS narrative_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_message_id INTEGER NOT NULL,
    end_message_id INTEGER NOT NULL,
    text TEXT NOT NULL,                    -- exact immutable payload variant
    fact_date TEXT,                        -- ISO or NULL (explicit dates; relatives = E4)
    entities TEXT NOT NULL DEFAULT '[]',   -- JSON array of canonical names
    prompt_version TEXT NOT NULL,          -- 'facts.v2' provenance tag
    valid_at TEXT,                         -- bi-temporal, mirrors knowledge_graph
    invalid_at TEXT,                       -- lifecycle-managed projection
    source_outcome_key TEXT REFERENCES fact_extraction_outcomes(slice_key)
        ON DELETE RESTRICT,
    fact_key TEXT,
    current_generation INTEGER,
    lifecycle_status TEXT NOT NULL DEFAULT 'legacy_unproven'
        CHECK (lifecycle_status IN ('active','retracted','legacy_unproven')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_narrative_facts_session
    ON narrative_facts(session_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_narrative_fact_authority_key
    ON narrative_facts(source_outcome_key, fact_key)
    WHERE source_outcome_key IS NOT NULL AND fact_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS narrative_fact_lifecycle (
    fact_id INTEGER NOT NULL REFERENCES narrative_facts(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL CHECK (generation >= 1),
    direction INTEGER NOT NULL CHECK (direction IN (-1, 1)),
    event_at TIMESTAMP NOT NULL,
    prompt_version TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fact_id, generation)
);
CREATE INDEX IF NOT EXISTS idx_narrative_fact_lifecycle_time
    ON narrative_fact_lifecycle(fact_id, event_at, generation);

-- content_rowid='id' on an INTEGER PRIMARY KEY (rowid alias): VACUUM-stable
-- like messages_fts/vec_edges, so no resync_rowid_shadows coverage needed.
-- Only current, authoritative projections belong in the search corpus. This
-- keeps legacy/retracted text out of both results and BM25 document-frequency
-- statistics. Startup healing exactly rebuilds this filtered shadow.
CREATE VIRTUAL TABLE IF NOT EXISTS narrative_facts_fts USING fts5(
    text,
    content='narrative_facts',
    content_rowid='id',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS narrative_facts_fts_insert
AFTER INSERT ON narrative_facts
WHEN new.source_outcome_key IS NOT NULL
 AND new.lifecycle_status = 'active'
 AND new.invalid_at IS NULL BEGIN
    INSERT INTO narrative_facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS narrative_facts_fts_delete
AFTER DELETE ON narrative_facts
WHEN old.source_outcome_key IS NOT NULL
 AND old.lifecycle_status = 'active'
 AND old.invalid_at IS NULL BEGIN
    INSERT INTO narrative_facts_fts(narrative_facts_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS narrative_facts_fts_update
AFTER UPDATE OF text, source_outcome_key, lifecycle_status, invalid_at
ON narrative_facts BEGIN
    INSERT INTO narrative_facts_fts(narrative_facts_fts, rowid, text)
    SELECT 'delete', old.id, old.text
    WHERE old.source_outcome_key IS NOT NULL
      AND old.lifecycle_status = 'active'
      AND old.invalid_at IS NULL;
    INSERT INTO narrative_facts_fts(rowid, text)
    SELECT new.id, new.text
    WHERE new.source_outcome_key IS NOT NULL
      AND new.lifecycle_status = 'active'
      AND new.invalid_at IS NULL;
END;

-- JSON mirror for fact vectors (vec_facts is created at runtime by
-- ensure_vec_table when sqlite-vec is present). Fact text is immutable, while
-- query joins still require a current proof-valid lifecycle projection.
CREATE TABLE IF NOT EXISTS narrative_fact_embeddings (
    fact_id INTEGER PRIMARY KEY REFERENCES narrative_facts(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL UNIQUE,
    scope TEXT NOT NULL DEFAULT 'always_on'
        CHECK (scope IN ('always_on', 'contextual')),
    trigger_entities TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT 'user'
        CHECK (source IN ('user', 'agent_inferred')),
    pos_evidence INTEGER NOT NULL DEFAULT 1,
    neg_evidence INTEGER NOT NULL DEFAULT 0,
    valid_at TIMESTAMP,
    invalid_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'retracted')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(scope, status, invalid_at);

-- v54 producer-bound whole-response auxiliary publication.  Existing
-- entity_types/entity_properties rows are compatibility history unless their
-- explicit origin is user; model output lives in the observation ledgers.
CREATE TABLE IF NOT EXISTS phase1_auxiliary_outcomes (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    extraction_cache_key TEXT NOT NULL,
    auxiliary_contract_key TEXT NOT NULL,
    result_hash TEXT NOT NULL CHECK (
        substr(result_hash, 1, 7) = 'sha256:' AND length(result_hash) = 71
        AND substr(result_hash, 8) NOT GLOB '*[^0-9a-f]*'
    ),
    entity_type_count INTEGER NOT NULL CHECK (entity_type_count >= 0),
    entity_property_count INTEGER NOT NULL CHECK (entity_property_count >= 0),
    entity_mention_count INTEGER NOT NULL CHECK (entity_mention_count >= 0),
    marker_count INTEGER NOT NULL CHECK (marker_count >= 0),
    published_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_id, phase1_generation_key)
);
CREATE INDEX IF NOT EXISTS idx_phase1_auxiliary_generation
    ON phase1_auxiliary_outcomes(phase1_generation_key, chunk_id);
CREATE TABLE IF NOT EXISTS entity_type_observations (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_canonical TEXT NOT NULL CHECK (
        length(trim(entity_canonical)) > 0
        AND hymem_entity_canonical_is_normalized(entity_canonical) = 1
    ),
    type TEXT NOT NULL CHECK (length(trim(type)) > 0),
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (
        typeof(confidence) IN ('integer', 'real')
        AND confidence >= 0.0 AND confidence <= 1.0
    ),
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_id, entity_canonical, type, phase1_generation_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_type_observations_lookup
    ON entity_type_observations(type, entity_canonical, phase1_generation_key);
CREATE INDEX IF NOT EXISTS idx_entity_type_observations_entity
    ON entity_type_observations(entity_canonical, phase1_generation_key);
CREATE TABLE IF NOT EXISTS entity_property_observations (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_canonical TEXT NOT NULL CHECK (
        length(trim(entity_canonical)) > 0
        AND hymem_entity_canonical_is_normalized(entity_canonical) = 1
    ),
    key TEXT NOT NULL CHECK (length(trim(key)) > 0),
    value TEXT NOT NULL,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_id, entity_canonical, key, phase1_generation_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_property_observations_lookup
    ON entity_property_observations(
        key, value, entity_canonical, phase1_generation_key
    );
CREATE INDEX IF NOT EXISTS idx_entity_property_observations_entity
    ON entity_property_observations(
        entity_canonical, key, phase1_generation_key
    );
CREATE TABLE IF NOT EXISTS entity_mention_observations (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_canonical TEXT NOT NULL CHECK (
        length(trim(entity_canonical)) > 0
        AND hymem_entity_canonical_is_normalized(entity_canonical) = 1
    ),
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_id,entity_canonical,phase1_generation_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_mention_observations_entity
    ON entity_mention_observations(entity_canonical,phase1_generation_key);
CREATE TABLE IF NOT EXISTS profile_entry_marker_evidence (
    profile_entry_id INTEGER NOT NULL
        REFERENCES profile_entries(id) ON DELETE CASCADE,
    marker_id INTEGER NOT NULL
        REFERENCES behavioral_markers(id) ON DELETE CASCADE,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (profile_entry_id, marker_id)
);
CREATE INDEX IF NOT EXISTS idx_profile_marker_generation
    ON profile_entry_marker_evidence(phase1_generation_key, marker_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_profile_marker_one_decision
    ON profile_entry_marker_evidence(marker_id);
CREATE TABLE IF NOT EXISTS profile_marker_decisions (
    marker_id INTEGER PRIMARY KEY
        REFERENCES behavioral_markers(id) ON DELETE CASCADE,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    profile_policy_key TEXT NOT NULL CHECK (
        length(trim(profile_policy_key)) > 0
    ),
    decision TEXT NOT NULL CHECK (
        decision IN ('materialized','manual_authority','identity_conflict')
    ),
    profile_entry_id INTEGER NOT NULL
        REFERENCES profile_entries(id) ON DELETE CASCADE,
    decided_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_profile_marker_decision_generation
    ON profile_marker_decisions(phase1_generation_key,marker_id);
CREATE TABLE IF NOT EXISTS rule_marker_evidence (
    rule_id INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    marker_id INTEGER NOT NULL
        REFERENCES behavioral_markers(id) ON DELETE CASCADE,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (rule_id, marker_id)
);
CREATE INDEX IF NOT EXISTS idx_rule_marker_generation
    ON rule_marker_evidence(phase1_generation_key, marker_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rule_marker_one_decision
    ON rule_marker_evidence(marker_id);
CREATE TABLE IF NOT EXISTS rule_marker_decisions (
    marker_id INTEGER PRIMARY KEY
        REFERENCES behavioral_markers(id) ON DELETE CASCADE,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    routing_key TEXT NOT NULL CHECK (length(trim(routing_key)) > 0),
    decision TEXT NOT NULL CHECK (decision IN ('routed', 'no_rule')),
    rule_id INTEGER REFERENCES rules(id) ON DELETE SET NULL,
    decided_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((decision='routed' AND rule_id IS NOT NULL)
        OR (decision='no_rule' AND rule_id IS NULL))
);
CREATE INDEX IF NOT EXISTS idx_rule_marker_decision_generation
    ON rule_marker_decisions(phase1_generation_key,marker_id);

/* The canonical v54 views/marker index/guards are installed by migration 054
   and healed at startup.  They cannot execute in this bootstrap script because
   initialize() runs schema.sql before upgrading an existing pre-v53 table that
   does not yet have phase1_generation_key/origin/source columns.
CREATE UNIQUE INDEX IF NOT EXISTS idx_behavioral_marker_generation_identity
    ON behavioral_markers(chunk_id, phase1_generation_key, kind, statement)
    WHERE phase1_generation_key IS NOT NULL;

DROP VIEW IF EXISTS current_entity_types;
CREATE VIEW current_entity_types AS
SELECT entity_canonical, type, confidence, source_chunk_id, origin
FROM entity_types WHERE origin = 'user'
UNION ALL
SELECT observation.entity_canonical, observation.type,
       observation.confidence, observation.chunk_id, 'agent_inferred'
FROM entity_type_observations observation
JOIN kg_claim_extraction_outcomes claim
  ON claim.chunk_id=observation.chunk_id
 AND claim.phase1_generation_key=observation.phase1_generation_key
JOIN phase1_auxiliary_outcomes auxiliary
  ON auxiliary.chunk_id=observation.chunk_id
 AND auxiliary.phase1_generation_key=observation.phase1_generation_key
 AND auxiliary.extraction_cache_key=claim.prompt_version
JOIN processed_chunks processed
  ON processed.chunk_id=claim.chunk_id
 AND processed.prompt_version=claim.prompt_version
 AND processed.phase1_generation_key=claim.phase1_generation_key
JOIN phase1_generations generation
  ON generation.generation_key=claim.phase1_generation_key
 AND generation.extraction_cache_key=claim.prompt_version
WHERE hymem_phase1_generation_is_current(
          generation.generation_key,generation.identity_exact
      )=1;

DROP VIEW IF EXISTS current_entity_properties;
CREATE VIEW current_entity_properties AS
WITH authorized AS (
 SELECT observation.entity_canonical,observation.key,observation.value,
        observation.chunk_id,observation.phase1_generation_key,
        chunk.end_message_id,chunk.created_at,
        ROW_NUMBER() OVER (
          PARTITION BY observation.entity_canonical,observation.key
          ORDER BY chunk.end_message_id DESC,
                   hymem_normalize_iso_timestamp(chunk.created_at) DESC,
                   observation.chunk_id DESC,
                   observation.phase1_generation_key DESC,
                   observation.value DESC
        ) AS authority_rank
 FROM entity_property_observations observation
 JOIN chunks chunk ON chunk.id=observation.chunk_id
 JOIN kg_claim_extraction_outcomes claim
   ON claim.chunk_id=observation.chunk_id
  AND claim.phase1_generation_key=observation.phase1_generation_key
 JOIN phase1_auxiliary_outcomes auxiliary
   ON auxiliary.chunk_id=observation.chunk_id
  AND auxiliary.phase1_generation_key=observation.phase1_generation_key
  AND auxiliary.extraction_cache_key=claim.prompt_version
 JOIN processed_chunks processed
   ON processed.chunk_id=claim.chunk_id
  AND processed.prompt_version=claim.prompt_version
  AND processed.phase1_generation_key=claim.phase1_generation_key
 JOIN phase1_generations generation
   ON generation.generation_key=claim.phase1_generation_key
  AND generation.extraction_cache_key=claim.prompt_version
 WHERE hymem_phase1_generation_is_current(
           generation.generation_key,generation.identity_exact
       )=1
)
SELECT entity_canonical,key,value,source_chunk_id,updated_at,origin
FROM entity_properties WHERE origin='user'
UNION ALL
SELECT authorized.entity_canonical,authorized.key,authorized.value,
       authorized.chunk_id,authorized.created_at,'agent_inferred'
FROM authorized
WHERE authorized.authority_rank=1
  AND NOT EXISTS (
    SELECT 1 FROM entity_properties manual
    WHERE manual.entity_canonical=authorized.entity_canonical
      AND manual.key=authorized.key AND manual.origin='user'
  );

DROP VIEW IF EXISTS current_profile_entries;
CREATE VIEW current_profile_entries AS
SELECT id,kind,text,pos_evidence,neg_evidence,first_seen,last_updated,source
FROM profile_entries WHERE source='user'
UNION ALL
SELECT entry.id,entry.kind,entry.text,COUNT(DISTINCT link.marker_id),
       entry.neg_evidence,entry.first_seen,entry.last_updated,entry.source
FROM profile_entries entry
JOIN profile_entry_marker_evidence link ON link.profile_entry_id=entry.id
JOIN behavioral_markers marker
  ON marker.id=link.marker_id
 AND marker.phase1_generation_key=link.phase1_generation_key
JOIN kg_claim_extraction_outcomes claim
  ON claim.chunk_id=marker.chunk_id
 AND claim.phase1_generation_key=marker.phase1_generation_key
JOIN phase1_auxiliary_outcomes auxiliary
  ON auxiliary.chunk_id=marker.chunk_id
 AND auxiliary.phase1_generation_key=marker.phase1_generation_key
 AND auxiliary.extraction_cache_key=claim.prompt_version
JOIN processed_chunks processed
  ON processed.chunk_id=claim.chunk_id
 AND processed.prompt_version=claim.prompt_version
 AND processed.phase1_generation_key=claim.phase1_generation_key
JOIN phase1_generations generation
  ON generation.generation_key=claim.phase1_generation_key
 AND generation.extraction_cache_key=claim.prompt_version
WHERE entry.source='agent_inferred'
  AND hymem_phase1_generation_is_current(
          generation.generation_key,generation.identity_exact
      )=1
GROUP BY entry.id;

DROP VIEW IF EXISTS current_rules;
CREATE VIEW current_rules AS
SELECT id,text,scope,trigger_entities,source,pos_evidence,neg_evidence,
       valid_at,invalid_at,status,created_at
FROM rules
WHERE source='user' AND status='active' AND invalid_at IS NULL
UNION ALL
SELECT rule.id,rule.text,rule.scope,rule.trigger_entities,rule.source,
       COUNT(DISTINCT link.marker_id),rule.neg_evidence,rule.valid_at,
       rule.invalid_at,rule.status,rule.created_at
FROM rules rule
JOIN rule_marker_evidence link ON link.rule_id=rule.id
JOIN behavioral_markers marker
  ON marker.id=link.marker_id
 AND marker.phase1_generation_key=link.phase1_generation_key
JOIN kg_claim_extraction_outcomes claim
  ON claim.chunk_id=marker.chunk_id
 AND claim.phase1_generation_key=marker.phase1_generation_key
JOIN phase1_auxiliary_outcomes auxiliary
  ON auxiliary.chunk_id=marker.chunk_id
 AND auxiliary.phase1_generation_key=marker.phase1_generation_key
 AND auxiliary.extraction_cache_key=claim.prompt_version
JOIN processed_chunks processed
  ON processed.chunk_id=claim.chunk_id
 AND processed.prompt_version=claim.prompt_version
 AND processed.phase1_generation_key=claim.phase1_generation_key
JOIN phase1_generations generation
  ON generation.generation_key=claim.phase1_generation_key
 AND generation.extraction_cache_key=claim.prompt_version
WHERE rule.source='agent_inferred'
  AND rule.status='active' AND rule.invalid_at IS NULL
  AND hymem_phase1_generation_is_current(
          generation.generation_key,generation.identity_exact
      )=1
GROUP BY rule.id;

CREATE TRIGGER IF NOT EXISTS profile_marker_evidence_lineage_guard
BEFORE INSERT ON profile_entry_marker_evidence
WHEN hymem_evidence_mutation_authorized() <> 1
  OR NOT EXISTS (
 SELECT 1 FROM behavioral_markers marker
 JOIN profile_entries entry ON entry.id=new.profile_entry_id
 WHERE marker.id=new.marker_id
   AND marker.phase1_generation_key=new.phase1_generation_key
   AND entry.source='agent_inferred'
   AND entry.text=marker.statement
   AND entry.kind=CASE marker.kind
       WHEN 'preference' THEN 'preference'
       WHEN 'rejection' THEN 'avoidance'
       WHEN 'style' THEN 'style'
       ELSE 'context'
   END
)
BEGIN
 SELECT RAISE(ABORT,'invalid profile marker lineage');
END;
CREATE TRIGGER IF NOT EXISTS rule_marker_evidence_lineage_guard
BEFORE INSERT ON rule_marker_evidence
WHEN hymem_evidence_mutation_authorized() <> 1
  OR NOT EXISTS (
 SELECT 1 FROM behavioral_markers marker
 JOIN rules rule ON rule.id=new.rule_id
 WHERE marker.id=new.marker_id
   AND marker.phase1_generation_key=new.phase1_generation_key
   AND rule.source='agent_inferred'
)
BEGIN
 SELECT RAISE(ABORT,'invalid rule marker lineage');
END;
CREATE TRIGGER IF NOT EXISTS phase1_auxiliary_outcome_insert_guard
BEFORE INSERT ON phase1_auxiliary_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS phase1_auxiliary_outcome_update_guard
BEFORE UPDATE ON phase1_auxiliary_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS phase1_auxiliary_outcome_delete_guard
BEFORE DELETE ON phase1_auxiliary_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_type_observation_insert_guard
BEFORE INSERT ON entity_type_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_type_observation_update_guard
BEFORE UPDATE ON entity_type_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_type_observation_delete_guard
BEFORE DELETE ON entity_type_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_property_observation_insert_guard
BEFORE INSERT ON entity_property_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_property_observation_update_guard
BEFORE UPDATE ON entity_property_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_property_observation_delete_guard
BEFORE DELETE ON entity_property_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS profile_marker_evidence_update_guard
BEFORE UPDATE ON profile_entry_marker_evidence
BEGIN SELECT RAISE(ABORT,'profile marker evidence is immutable'); END;
CREATE TRIGGER IF NOT EXISTS profile_marker_evidence_delete_guard
BEFORE DELETE ON profile_entry_marker_evidence
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'profile marker evidence is internally managed'); END;
CREATE TRIGGER IF NOT EXISTS rule_marker_evidence_update_guard
BEFORE UPDATE ON rule_marker_evidence
BEGIN SELECT RAISE(ABORT,'rule marker evidence is immutable'); END;
CREATE TRIGGER IF NOT EXISTS rule_marker_evidence_delete_guard
BEFORE DELETE ON rule_marker_evidence
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'rule marker evidence is internally managed'); END;
CREATE TRIGGER IF NOT EXISTS rule_marker_decision_insert_guard
BEFORE INSERT ON rule_marker_decisions
WHEN hymem_evidence_mutation_authorized() <> 1
  OR NOT EXISTS (
    SELECT 1 FROM behavioral_markers marker
    WHERE marker.id=new.marker_id
      AND marker.phase1_generation_key=new.phase1_generation_key
  )
  OR (new.decision='routed' AND NOT EXISTS (
    SELECT 1 FROM rules rule
    WHERE rule.id=new.rule_id AND rule.source='agent_inferred'
  ))
BEGIN SELECT RAISE(ABORT,'invalid rule marker decision'); END;
CREATE TRIGGER IF NOT EXISTS rule_marker_decision_update_guard
BEFORE UPDATE ON rule_marker_decisions
BEGIN SELECT RAISE(ABORT,'rule marker decision is immutable'); END;
CREATE TRIGGER IF NOT EXISTS rule_marker_decision_delete_guard
BEFORE DELETE ON rule_marker_decisions
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT,'rule marker decision is internally managed'); END;
*/
