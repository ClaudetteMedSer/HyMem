-- v14: temporal_mentions — explicit dates extracted from raw message text for
-- the temporal-reasoning (TR) retrieval path. Each row ties a date written in a
-- turn ("we shipped on Feb 15") to that message, so TR can return events already
-- ordered by date instead of leaving the host LLM to find dates in noise.
--
-- `normalized_date` is ISO YYYY-MM-DD when a full date (incl. year) resolved,
-- else NULL for a year-less mention (its raw_text + the turn's created_at still
-- carry ordering signal). UNIQUE(message_id, raw_text) makes re-dreaming a chunk
-- idempotent. schema.sql creates the same objects for fresh DBs; this migration
-- adds them to existing (<= v13) DBs.
--
-- No backfill: temporal_mentions are populated by the dream cycle's per-message
-- pass (dreaming/temporal.py), so the next dream re-indexes existing chunks and
-- fills the table. A backfill here would need to re-parse every message and is
-- better left to the idempotent dream pass.
CREATE TABLE IF NOT EXISTS temporal_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    normalized_date TEXT,
    raw_text TEXT NOT NULL,
    surrounding_text TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (message_id, raw_text)
);
CREATE INDEX IF NOT EXISTS idx_temporal_mentions_date ON temporal_mentions(normalized_date);
CREATE INDEX IF NOT EXISTS idx_temporal_mentions_message ON temporal_mentions(message_id);
