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

-- Raw session log. Hermes pushes messages in; HyMem owns the table.
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
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
    profile_prompt_version TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);

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

CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

-- Embedding vectors for chunks. JSON-encoded floats; cosine similarity
-- is computed in Python at query time.
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

-- Idempotency: each chunk processed at most once per prompt_version.
CREATE TABLE IF NOT EXISTS processed_chunks (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    prompt_version TEXT NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
        'runs_on','connects_to','generates','tested_by'
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
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
    surface_subject TEXT,
    surface_object TEXT,
    value_text TEXT,
    value_numeric REAL,
    value_unit TEXT,
    temporal_scope TEXT,
    source_role TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(edge_id, chunk_id, polarity)
);
CREATE INDEX IF NOT EXISTS idx_evidence_edge ON kg_evidence(edge_id);
CREATE INDEX IF NOT EXISTS idx_evidence_chunk ON kg_evidence(chunk_id);

-- Behavioral markers (explicit signals only).
CREATE TABLE IF NOT EXISTS behavioral_markers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('correction','preference','rejection','style')),
    statement TEXT NOT NULL,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    consolidated_at TIMESTAMP
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
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    title, summary,
    content='episodes', content_rowid='rowid',
    tokenize='porter unicode61'
);

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
-- question reads a handful of cluster summaries instead of dozens of raw turns.
-- Levels >= 1 (v17) are RAPTOR rollups whose member_episode_ids hold CHILD ids
-- (lower-level node ids and/or pass-through episode ids), recursing until one
-- is_root digest node — the standing "what do you know about me" summary
-- exposed via HyMem.digest(). Only level 0 enters query-time retrieval. The
-- whole layer is additive and off by default (cfg.aggregation_nodes_enabled).
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
    is_root INTEGER NOT NULL DEFAULT 0
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
-- scan over these (no vec0 table) since the node count is small and the tier is
-- off by default — keeping the vec0 plumbing limited to chunks/edges/episodes.
CREATE TABLE IF NOT EXISTS aggregation_node_embeddings (
    node_id TEXT PRIMARY KEY REFERENCES aggregation_nodes(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    confidence REAL NOT NULL DEFAULT 1.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    valid_at TIMESTAMP,
    invalid_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_profile_active
    ON user_profile(slot, slot_key, invalid_at);

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

-- Extraction feedback: stores wrongly-extracted triples so future extractions
-- can learn from past mistakes.
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
    chunks_embedded INTEGER NOT NULL DEFAULT 0,
    edges_embedded INTEGER NOT NULL DEFAULT 0,
    triples_extracted INTEGER NOT NULL DEFAULT 0,
    markers_extracted INTEGER NOT NULL DEFAULT 0,
    aggregation_nodes_built INTEGER NOT NULL DEFAULT 0,
    aggregation_nodes_reused INTEGER NOT NULL DEFAULT 0,
    skipped_locked INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_dream_runs_started ON dream_runs(started_at);
