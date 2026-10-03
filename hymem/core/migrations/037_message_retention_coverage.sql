-- v37: raw-message retention requires explicit, content-bound, lossless proof.
--
-- No legacy rows are backfilled. Historical summaries, digest watermarks,
-- chunks, episodes, and graph evidence are all potentially lossy, so guessing
-- coverage during migration could authorize irreversible message deletion.

CREATE TABLE IF NOT EXISTS message_retention_coverage (
    message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_role TEXT NOT NULL CHECK (source_role IN ('user','assistant','system','tool')),
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

CREATE TRIGGER IF NOT EXISTS message_retention_covered_chunk_update_guard
BEFORE UPDATE OF session_id, start_message_id, end_message_id, text ON chunks
WHEN EXISTS (
    SELECT 1 FROM message_retention_coverage mc WHERE mc.chunk_id = old.id
) BEGIN
    SELECT RAISE(ABORT, 'cannot mutate a covered lossless chunk');
END;
