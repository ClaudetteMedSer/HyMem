-- v16: RAPTOR cross-session aggregation nodes (Phase 2).
-- Each node fuses a cross-session cluster of episodes into one summary, so a
-- multi-session synthesis question reads a few cluster summaries instead of
-- dozens of raw turns. Additive + off by default at query time
-- (cfg.aggregation_nodes_enabled). Forward-only and idempotent: re-applying
-- against a schema.sql DB that already has these objects is a tolerated no-op.
CREATE TABLE IF NOT EXISTS aggregation_nodes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    member_episode_ids TEXT NOT NULL DEFAULT '[]',
    session_ids TEXT NOT NULL DEFAULT '[]',
    n_members INTEGER NOT NULL DEFAULT 0,
    n_sessions INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

CREATE TABLE IF NOT EXISTS aggregation_node_embeddings (
    node_id TEXT PRIMARY KEY REFERENCES aggregation_nodes(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
