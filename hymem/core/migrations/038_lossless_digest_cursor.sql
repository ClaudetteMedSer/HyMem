-- v38: lossless message coverage and resumable, rolling session digests.
--
-- `chunk_kind` separates exact source artifacts from the selective chunks used
-- for graph extraction and retrieval.  Existing chunks are extraction chunks.
--
-- Coverage and digest cursors are deliberately independent.  Coverage is a
-- local SQLite write and may run ahead of the LLM-backed digest.  The digest
-- cursor also stores a character offset so one message larger than the prompt
-- budget is consumed over several successful calls without dropping its tail.
-- `digest_cursor_prompt_version` identifies the digest build currently being
-- walked: prompt/config identity plus a unique full-walk token. A version
-- change or explicit rebuild starts a fresh walk without pretending the old
-- watermark covered input under the replacement build.
--
-- `summary` remains the compatibility/operator-facing value.  New automatic
-- summaries are attributed explicitly and retained separately so a rolling
-- dream never overwrites an operator or conservatively-classified legacy
-- summary.  Existing non-empty summaries are marked legacy because older
-- schemas carried no reliable provenance.
--
-- The CREATE guards make this migration tolerant of the project's supported
-- sparse legacy fixtures/embedders, some of which legitimately omitted an
-- unused optional table while still carrying a later schema stamp.  On a real
-- store these are no-ops; on a sparse store they establish the table before
-- the additive ALTER statements below.
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    summary TEXT,
    digested_prompt_version TEXT,
    profile_prompt_version TEXT,
    digested_message_id INTEGER,
    facts_message_id INTEGER,
    episodes_prompt_version TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    start_message_id INTEGER,
    end_message_id INTEGER,
    salience_reason TEXT,
    text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    title TEXT,
    summary TEXT,
    participants TEXT NOT NULL DEFAULT '[]',
    start_message_id INTEGER,
    end_message_id INTEGER,
    outcome TEXT,
    key_entities TEXT NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE sessions ADD COLUMN summary TEXT;
-- Supported migration fixtures may carry the historical minimal `chunks(id)`
-- shape. Real v37 stores already have text; the migration runner treats that
-- duplicate-column error as an idempotent no-op.
ALTER TABLE chunks ADD COLUMN text TEXT;
ALTER TABLE chunks ADD COLUMN chunk_kind TEXT NOT NULL DEFAULT 'extraction'
    CHECK (chunk_kind IN ('extraction', 'coverage'));
ALTER TABLE sessions ADD COLUMN coverage_message_id INTEGER;
ALTER TABLE sessions ADD COLUMN digest_cursor_message_id INTEGER;
ALTER TABLE sessions ADD COLUMN digest_cursor_partial_message_id INTEGER;
ALTER TABLE sessions ADD COLUMN digest_cursor_offset INTEGER NOT NULL DEFAULT 0
    CHECK (digest_cursor_offset >= 0);
ALTER TABLE sessions ADD COLUMN digest_cursor_prompt_version TEXT;
ALTER TABLE sessions ADD COLUMN digest_published_generation TEXT;
ALTER TABLE sessions ADD COLUMN auto_summary TEXT;
ALTER TABLE sessions ADD COLUMN auto_summary_message_id INTEGER;
ALTER TABLE sessions ADD COLUMN auto_summary_partial_message_id INTEGER;
ALTER TABLE sessions ADD COLUMN auto_summary_message_offset INTEGER NOT NULL DEFAULT 0;
ALTER TABLE sessions ADD COLUMN summary_source TEXT
    CHECK (summary_source IN ('auto', 'operator', 'legacy'));
-- Supported migration fixtures include the historical minimal
-- `episodes(id,title,summary)` shape. Establish every column referenced by the
-- publication-aware FTS triggers before rebuilding their postings.
ALTER TABLE episodes ADD COLUMN session_id TEXT;
ALTER TABLE episodes ADD COLUMN title TEXT;
ALTER TABLE episodes ADD COLUMN summary TEXT;
ALTER TABLE episodes ADD COLUMN participants TEXT NOT NULL DEFAULT '[]';
ALTER TABLE episodes ADD COLUMN start_message_id INTEGER;
ALTER TABLE episodes ADD COLUMN end_message_id INTEGER;
ALTER TABLE episodes ADD COLUMN outcome TEXT;
ALTER TABLE episodes ADD COLUMN key_entities TEXT NOT NULL DEFAULT '[]';
ALTER TABLE episodes ADD COLUMN created_at TIMESTAMP;
ALTER TABLE episodes ADD COLUMN digest_slice_key TEXT;
ALTER TABLE episodes ADD COLUMN digest_generation TEXT;
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);

UPDATE sessions
SET summary_source = 'legacy'
WHERE summary IS NOT NULL AND TRIM(summary) <> '' AND summary_source IS NULL;

UPDATE episodes
SET digest_generation = 'legacy'
WHERE digest_generation IS NULL;

-- Pre-v38 episode rows were all published; no partial-build distinction
-- existed. Attribute them to one conservative legacy generation so the new
-- visibility predicate preserves the exact pre-migration behavior.
UPDATE sessions
SET digest_published_generation = 'legacy'
WHERE digest_published_generation IS NULL
  AND EXISTS (
      SELECT 1 FROM episodes e
      WHERE e.session_id = sessions.id
        AND e.digest_generation = 'legacy'
  );

-- v37 allowed any exact canonical backing chunk.  Once kind attribution
-- exists those artifacts must not remain retrieval/extraction candidates.
-- Drop the old immutability trigger before the backfill so migration replay is
-- safe even if a previous run reached the trigger recreation but not the
-- schema-version stamp.
DROP TRIGGER IF EXISTS message_retention_covered_chunk_update_guard;
UPDATE chunks
SET chunk_kind = 'coverage'
WHERE id IN (SELECT DISTINCT chunk_id FROM message_retention_coverage);

CREATE INDEX IF NOT EXISTS idx_message_retention_coverage_stream
    ON message_retention_coverage(
        source_session_id, coverage_version, message_id
    );

-- Coverage JSONL must not enter the FTS corpus at all: filtering result rows
-- after MATCH would still let the duplicate documents change BM25 IDF/ranking.
-- Rebuild explicitly rather than using FTS5's external-content `rebuild`
-- command, which would index every chunks row regardless of chunk_kind.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
DROP TRIGGER IF EXISTS chunks_fts_insert;
DROP TRIGGER IF EXISTS chunks_fts_delete;
DROP TRIGGER IF EXISTS chunks_fts_update_delete;
DROP TRIGGER IF EXISTS chunks_fts_update_insert;
INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all');
INSERT INTO chunks_fts(rowid, text)
SELECT rowid, text FROM chunks WHERE chunk_kind = 'extraction';
CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks
WHEN new.chunk_kind = 'extraction' BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks
WHEN old.chunk_kind = 'extraction' BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER chunks_fts_update_delete
AFTER UPDATE OF text, chunk_kind ON chunks
WHEN old.chunk_kind = 'extraction' BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER chunks_fts_update_insert
AFTER UPDATE OF text, chunk_kind ON chunks
WHEN new.chunk_kind = 'extraction' BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- Replacement episode generations are durable/resumable but unpublished
-- until their session marker flips at the tail. Keep them out of the physical
-- FTS corpus as well as result rows, otherwise duplicate staging text changes
-- BM25 document statistics and can reorder published hits.
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    title, summary,
    content='episodes', content_rowid='rowid',
    tokenize='porter unicode61'
);
DROP TRIGGER IF EXISTS episodes_fts_insert;
DROP TRIGGER IF EXISTS episodes_fts_delete;
DROP TRIGGER IF EXISTS episodes_fts_update;
DROP TRIGGER IF EXISTS episodes_fts_update_delete;
DROP TRIGGER IF EXISTS episodes_fts_update_insert;
DROP TRIGGER IF EXISTS episodes_fts_session_delete;
INSERT INTO episodes_fts(episodes_fts) VALUES('delete-all');
INSERT INTO episodes_fts(rowid, title, summary)
SELECT e.rowid, e.title, e.summary
FROM episodes e
JOIN sessions s ON s.id = e.session_id
WHERE e.digest_generation IS NULL
   OR e.digest_generation = s.digest_published_generation;
CREATE TRIGGER episodes_fts_insert AFTER INSERT ON episodes
WHEN new.digest_generation IS NULL OR new.digest_generation = (
    SELECT digest_published_generation FROM sessions WHERE id = new.session_id
) BEGIN
    INSERT INTO episodes_fts(rowid, title, summary)
    VALUES (new.rowid, new.title, new.summary);
END;
CREATE TRIGGER episodes_fts_delete AFTER DELETE ON episodes
WHEN old.digest_generation IS NULL OR old.digest_generation = (
    SELECT digest_published_generation FROM sessions WHERE id = old.session_id
) BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, title, summary)
    VALUES ('delete', old.rowid, old.title, old.summary);
END;
CREATE TRIGGER episodes_fts_update AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, title, summary)
    SELECT 'delete', old.rowid, old.title, old.summary
    WHERE old.digest_generation IS NULL OR old.digest_generation = (
        SELECT digest_published_generation FROM sessions WHERE id = old.session_id
    );
    INSERT INTO episodes_fts(rowid, title, summary)
    SELECT new.rowid, new.title, new.summary
    WHERE new.digest_generation IS NULL OR new.digest_generation = (
        SELECT digest_published_generation FROM sessions WHERE id = new.session_id
    );
END;
-- During an ON DELETE CASCADE the parent session is already absent when the
-- episode AFTER DELETE trigger runs, so it cannot resolve the publication
-- marker. Remove non-NULL published postings while the parent is still visible;
-- standalone NULL-generation rows remain the episode trigger's responsibility.
CREATE TRIGGER episodes_fts_session_delete BEFORE DELETE ON sessions
WHEN old.digest_published_generation IS NOT NULL BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, title, summary)
    SELECT 'delete', e.rowid, e.title, e.summary
    FROM episodes e
    WHERE e.session_id = old.id
      AND e.digest_generation = old.digest_published_generation;
END;

CREATE TRIGGER IF NOT EXISTS message_lossless_stream_delete_guard
BEFORE DELETE ON message_retention_coverage
WHEN old.coverage_version = 'dream-lossless-message-v1' BEGIN
    SELECT RAISE(ABORT, 'ordered digest coverage is immutable');
END;
CREATE TRIGGER IF NOT EXISTS message_lossless_stream_update_guard
BEFORE UPDATE ON message_retention_coverage
WHEN old.coverage_version = 'dream-lossless-message-v1' BEGIN
    SELECT RAISE(ABORT, 'ordered digest coverage is immutable');
END;

CREATE TRIGGER IF NOT EXISTS message_lossless_source_update_guard
BEFORE UPDATE OF id, session_id, role, content, created_at ON messages
WHEN EXISTS (
    SELECT 1 FROM message_retention_coverage mc
    WHERE mc.message_id = old.id
      AND mc.coverage_version = 'dream-lossless-message-v1'
) BEGIN
    SELECT RAISE(ABORT, 'ordered digest source is immutable');
END;

CREATE TRIGGER message_retention_covered_chunk_update_guard
BEFORE UPDATE OF session_id, start_message_id, end_message_id, text, chunk_kind
ON chunks
WHEN EXISTS (
    SELECT 1 FROM message_retention_coverage mc WHERE mc.chunk_id = old.id
) BEGIN
    SELECT RAISE(ABORT, 'cannot mutate a covered lossless chunk');
END;
