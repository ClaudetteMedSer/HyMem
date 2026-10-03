-- Independent, local-only recovery. No recovered draft is consumer-visible
-- until its complete source-proved target is atomically published.
CREATE TABLE IF NOT EXISTS summary_recovery (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    config_version TEXT NOT NULL,
    walk_id TEXT NOT NULL CHECK (length(walk_id)=32 AND walk_id NOT GLOB '*[^0-9a-f]*'),
    target_generation TEXT NOT NULL,
    target_message_id INTEGER NOT NULL CHECK (typeof(target_message_id)='integer' AND target_message_id>0),
    base_sha256 TEXT NOT NULL CHECK (length(base_sha256)=64 AND base_sha256 NOT GLOB '*[^0-9a-f]*'),
    target_source_sha256 TEXT NOT NULL CHECK (length(target_source_sha256)=64 AND target_source_sha256 NOT GLOB '*[^0-9a-f]*'),
    cursor_message_id INTEGER CHECK (cursor_message_id IS NULL OR (typeof(cursor_message_id)='integer' AND cursor_message_id>0)),
    cursor_partial_message_id INTEGER CHECK (cursor_partial_message_id IS NULL OR (typeof(cursor_partial_message_id)='integer' AND cursor_partial_message_id>0)),
    cursor_offset INTEGER NOT NULL DEFAULT 0 CHECK (typeof(cursor_offset)='integer' AND cursor_offset>=0),
    draft TEXT NOT NULL CHECK (typeof(draft)='text' AND length(draft)<=500),
    source_sha256 TEXT NOT NULL CHECK (length(source_sha256)=64 AND source_sha256 NOT GLOB '*[^0-9a-f]*'),
    attempt_limit INTEGER NOT NULL CHECK (typeof(attempt_limit)='integer' AND attempt_limit BETWEEN 1 AND 100),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (typeof(attempts)='integer' AND attempts BETWEEN 0 AND 100),
    failure_reason TEXT CHECK (failure_reason IS NULL OR failure_reason IN
        ('summary_output_cap','summary_shape_failure','summary_validation_failure',
         'parse_failure','output_truncated','shape_failure')),
    state_sha256 TEXT NOT NULL CHECK (length(state_sha256)=64 AND state_sha256 NOT GLOB '*[^0-9a-f]*'),
    CHECK ((cursor_partial_message_id IS NULL AND cursor_offset=0)
           OR (cursor_partial_message_id IS NOT NULL AND cursor_offset>0)),
    CHECK (attempts<=attempt_limit)
);
CREATE TRIGGER IF NOT EXISTS summary_recovery_workspace_guard
BEFORE UPDATE OF source_workspace_id ON sessions
WHEN new.source_workspace_id IS NOT old.source_workspace_id
 AND EXISTS (SELECT 1 FROM summary_recovery WHERE session_id=old.id)
BEGIN
    SELECT RAISE(ABORT, 'cannot rebind a summary recovery session');
END;
INSERT OR IGNORE INTO schema_meta(key,value) VALUES ('summary_recovery_schema','63');
