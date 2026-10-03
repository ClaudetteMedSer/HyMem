-- Items publish independently from rolling-summary availability. Existing
-- auto_summary_* values remain the last accepted summary frontier. Python
-- backfills the new item frontier only from coherent source-proved historical
-- publications, never from an active/incomplete walk. This migration and its
-- backfill/stamp execute in one transaction.
INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('summary_frontier_schema', '62');
ALTER TABLE sessions ADD COLUMN digest_published_message_id INTEGER CHECK (
    digest_published_message_id IS NULL OR
    (typeof(digest_published_message_id) = 'integer' AND digest_published_message_id > 0)
);
ALTER TABLE sessions ADD COLUMN auto_summary_generation TEXT CHECK (
    auto_summary_generation IS NULL OR
    (typeof(auto_summary_generation) = 'text' AND length(auto_summary_generation) > 0)
);
ALTER TABLE sessions ADD COLUMN summary_failure_reason TEXT CHECK (
    summary_failure_reason IS NULL OR
    (typeof(summary_failure_reason) = 'text' AND length(summary_failure_reason) BETWEEN 1 AND 80
     AND summary_failure_reason NOT GLOB '*[^a-z0-9_]*'
     AND summary_failure_reason IN ('summary_output_cap','summary_shape_failure','summary_validation_failure',
                                    'parse_failure','output_truncated','shape_failure','prior_summary_gap'))
);
ALTER TABLE sessions ADD COLUMN summary_failure_count INTEGER NOT NULL DEFAULT 0 CHECK (
    typeof(summary_failure_count) = 'integer' AND summary_failure_count >= 0
);
-- Supported sparse historical migration fixtures can omit the v58 domain.
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
ALTER TABLE digest_staging ADD COLUMN summary_failure_reason TEXT CHECK (
    summary_failure_reason IS NULL OR
    (typeof(summary_failure_reason) = 'text' AND length(summary_failure_reason) BETWEEN 1 AND 80
     AND summary_failure_reason NOT GLOB '*[^a-z0-9_]*'
     AND summary_failure_reason IN ('summary_output_cap','summary_shape_failure','summary_validation_failure',
                                    'parse_failure','output_truncated','shape_failure','prior_summary_gap'))
);
CREATE TRIGGER IF NOT EXISTS summary_state_workspace_guard
BEFORE UPDATE OF source_workspace_id ON sessions
WHEN new.source_workspace_id IS NOT old.source_workspace_id
 AND (old.digest_published_message_id IS NOT NULL
      OR old.auto_summary_generation IS NOT NULL
      OR old.summary_failure_reason IS NOT NULL
      OR old.summary_failure_count <> 0)
BEGIN
    SELECT RAISE(ABORT, 'cannot rebind a summary-state session');
END;
