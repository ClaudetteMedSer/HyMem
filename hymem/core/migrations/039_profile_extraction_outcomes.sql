-- v39: explicit extraction outcomes and a lossless typed-profile cursor.
--
-- Profile extraction now walks the reviewed v38 coverage stream independently,
-- USER roles only, and can resume inside an oversized message. Validated output
-- is redacted and staged per generation; consumer-visible rows change only in
-- the transaction that reaches the tail. Existing claims omitted by a rebuild
-- are conservatively retained because pre-v39 rows do not record every session
-- that supported a coalesced value.

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
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    start_message_id INTEGER,
    end_message_id INTEGER,
    salience_reason TEXT,
    text TEXT,
    chunk_kind TEXT NOT NULL DEFAULT 'extraction',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS message_retention_coverage (
    message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_role TEXT NOT NULL,
    source_created_at TIMESTAMP,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE RESTRICT,
    message_content_hash TEXT NOT NULL,
    hash_version TEXT NOT NULL,
    record_version TEXT NOT NULL,
    coverage_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (message_id, chunk_id, coverage_version)
);
CREATE TABLE IF NOT EXISTS user_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot TEXT NOT NULL,
    slot_key TEXT,
    value TEXT NOT NULL,
    evidence_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    valid_at TIMESTAMP,
    invalid_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_message_id INTEGER,
    source_session_id TEXT,
    source_created_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS dream_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP
);

-- Some supported migration fixtures predate the timestamp column even though
-- later retention/profile logic needs it. Real v38 stores already have it.
ALTER TABLE messages ADD COLUMN created_at TIMESTAMP;

ALTER TABLE sessions ADD COLUMN profile_cursor_message_id INTEGER;
ALTER TABLE sessions ADD COLUMN profile_cursor_partial_message_id INTEGER;
ALTER TABLE sessions ADD COLUMN profile_cursor_offset INTEGER NOT NULL DEFAULT 0
    CHECK (profile_cursor_offset >= 0);
ALTER TABLE sessions ADD COLUMN profile_cursor_prompt_version TEXT;
ALTER TABLE sessions ADD COLUMN profile_published_generation TEXT;
ALTER TABLE sessions ADD COLUMN profile_retry_count INTEGER NOT NULL DEFAULT 0
    CHECK (profile_retry_count >= 0);
ALTER TABLE sessions ADD COLUMN profile_retry_config_version TEXT;
ALTER TABLE sessions ADD COLUMN profile_quarantined BOOLEAN NOT NULL DEFAULT 0
    CHECK (profile_quarantined IN (0, 1));
ALTER TABLE sessions ADD COLUMN digest_retry_count INTEGER NOT NULL DEFAULT 0
    CHECK (digest_retry_count >= 0);
ALTER TABLE sessions ADD COLUMN digest_retry_config_version TEXT;
ALTER TABLE sessions ADD COLUMN digest_quarantined BOOLEAN NOT NULL DEFAULT 0
    CHECK (digest_quarantined IN (0, 1));

ALTER TABLE user_profile ADD COLUMN source_message_id INTEGER;
ALTER TABLE user_profile ADD COLUMN source_session_id TEXT;
ALTER TABLE user_profile ADD COLUMN source_created_at TIMESTAMP;

ALTER TABLE dream_runs ADD COLUMN profile_items_extracted INTEGER NOT NULL DEFAULT 0;
ALTER TABLE dream_runs ADD COLUMN profile_failures INTEGER NOT NULL DEFAULT 0;
ALTER TABLE dream_runs ADD COLUMN digest_quarantined INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS profile_staging (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    generation TEXT NOT NULL,
    slice_key TEXT NOT NULL,
    items_json TEXT NOT NULL CHECK (json_valid(items_json)),
    start_message_id INTEGER,
    start_message_offset INTEGER NOT NULL DEFAULT 0
        CHECK (start_message_offset >= 0),
    end_message_id INTEGER,
    cursor_before_message_id INTEGER,
    cursor_before_partial_message_id INTEGER,
    cursor_before_offset INTEGER NOT NULL DEFAULT 0
        CHECK (cursor_before_offset >= 0),
    cursor_after_message_id INTEGER,
    cursor_after_partial_message_id INTEGER,
    cursor_after_offset INTEGER NOT NULL DEFAULT 0
        CHECK (cursor_after_offset >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, generation, slice_key)
);
CREATE INDEX IF NOT EXISTS idx_profile_staging_generation
    ON profile_staging(session_id, generation);
ALTER TABLE profile_staging ADD COLUMN start_message_offset INTEGER NOT NULL DEFAULT 0
    CHECK (start_message_offset >= 0);
ALTER TABLE profile_staging ADD COLUMN cursor_before_message_id INTEGER;
ALTER TABLE profile_staging ADD COLUMN cursor_before_partial_message_id INTEGER;
ALTER TABLE profile_staging ADD COLUMN cursor_before_offset INTEGER NOT NULL DEFAULT 0
    CHECK (cursor_before_offset >= 0);
ALTER TABLE profile_staging ADD COLUMN cursor_after_message_id INTEGER;
ALTER TABLE profile_staging ADD COLUMN cursor_after_partial_message_id INTEGER;
ALTER TABLE profile_staging ADD COLUMN cursor_after_offset INTEGER NOT NULL DEFAULT 0
    CHECK (cursor_after_offset >= 0);
CREATE INDEX IF NOT EXISTS idx_user_profile_source
    ON user_profile(source_session_id, source_message_id);

-- Snapshot the live FK before future raw retention nulls it. Pre-v39 rows
-- whose source was already pruned cannot be reconstructed and remain NULL
-- rather than receiving guessed provenance.
UPDATE user_profile
SET source_message_id = evidence_message_id
WHERE source_message_id IS NULL
  AND evidence_message_id IS NOT NULL
  AND EXISTS (
      SELECT 1 FROM messages m
      WHERE m.id = user_profile.evidence_message_id
        AND m.role = 'user'
  );

UPDATE user_profile
SET source_session_id = (
        SELECT m.session_id FROM messages m
        WHERE m.id = user_profile.source_message_id AND m.role = 'user'
    ),
    source_created_at = (
        SELECT m.created_at FROM messages m
        WHERE m.id = user_profile.source_message_id AND m.role = 'user'
    )
WHERE source_message_id IS NOT NULL
  AND (source_session_id IS NULL OR source_created_at IS NULL);

UPDATE user_profile
SET source_session_id = COALESCE(source_session_id, (
        SELECT mc.source_session_id
        FROM message_retention_coverage mc
        JOIN sessions s ON s.id = mc.source_session_id
        WHERE mc.message_id = user_profile.source_message_id
          AND mc.coverage_version = 'dream-lossless-message-v1'
          AND mc.source_role = 'user'
          AND s.coverage_message_id IS NOT NULL
          AND mc.message_id <= s.coverage_message_id
        ORDER BY mc.created_at DESC LIMIT 1
    )),
    source_created_at = COALESCE(source_created_at, (
        SELECT mc.source_created_at
        FROM message_retention_coverage mc
        JOIN sessions s ON s.id = mc.source_session_id
        WHERE mc.message_id = user_profile.source_message_id
          AND mc.coverage_version = 'dream-lossless-message-v1'
          AND mc.source_role = 'user'
          AND s.coverage_message_id IS NOT NULL
          AND mc.message_id <= s.coverage_message_id
        ORDER BY mc.created_at DESC LIMIT 1
    ))
WHERE source_message_id IS NOT NULL;
