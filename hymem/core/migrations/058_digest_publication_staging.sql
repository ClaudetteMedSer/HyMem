-- Private bounded-slice output, never a retrieval or portable authority.
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
CREATE TRIGGER IF NOT EXISTS digest_staging_workspace_guard
BEFORE UPDATE OF source_workspace_id ON sessions
WHEN new.source_workspace_id IS NOT old.source_workspace_id
 AND EXISTS (SELECT 1 FROM digest_staging WHERE session_id = old.id)
BEGIN
    SELECT RAISE(ABORT, 'cannot rebind a staged digest session');
END;
