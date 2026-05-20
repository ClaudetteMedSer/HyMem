-- v9: free-form key/value attributes per canonical entity.
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
