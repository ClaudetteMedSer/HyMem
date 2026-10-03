-- v8: episode-level embeddings + UPDATE-side FTS trigger for episodes.
CREATE TRIGGER IF NOT EXISTS episodes_fts_update AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, title, summary) VALUES ('delete', old.rowid, old.title, old.summary);
    INSERT INTO episodes_fts(rowid, title, summary) VALUES (new.rowid, new.title, new.summary);
END;
CREATE TABLE IF NOT EXISTS episode_embeddings (
    episode_id TEXT PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
