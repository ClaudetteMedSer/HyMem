-- No backfill: historical ids/names do not establish digest ownership.
CREATE TABLE IF NOT EXISTS procedure_digest_publications (
    procedure_id TEXT PRIMARY KEY REFERENCES procedures(id) ON DELETE CASCADE,
    generation TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL CHECK (length(payload_sha256) = 64),
    retired INTEGER NOT NULL DEFAULT 0 CHECK (retired IN (0,1))
);
