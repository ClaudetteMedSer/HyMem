-- v6: knowledge-graph edge embeddings + dream_runs counter.
CREATE TABLE IF NOT EXISTS edge_embeddings (
    edge_text TEXT PRIMARY KEY,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE dream_runs ADD COLUMN edges_embedded INTEGER NOT NULL DEFAULT 0;
