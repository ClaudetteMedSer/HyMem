-- HyMem schema. All migrations are forward-only; bump schema_version below.
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
    summary TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_procedures_session ON procedures(session_id);
CREATE INDEX IF NOT EXISTS idx_procedures_entities ON procedures(entities_involved);

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
    skipped_locked INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_dream_runs_started ON dream_runs(started_at);
