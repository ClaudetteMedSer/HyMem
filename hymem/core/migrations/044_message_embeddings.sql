-- v44: semantic retrieval over the exact durable message-occurrence corpus.
--
-- message_id is the stable local occurrence id and therefore also the vec0
-- rowid.  The composite FK binds every vector to the immutable lossless
-- coverage proof whose decoded content was embedded.  Raw messages may be
-- pruned; their coverage proof and vector deliberately survive.
-- Some supported legacy fixtures predate the optional embedding mirror
-- entirely and run migrations without first executing schema.sql.  Create the
-- old shape before adding the freshness fingerprint so the forward migration
-- remains valid for those stores too.
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE chunk_embeddings ADD COLUMN text_hash TEXT;

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
