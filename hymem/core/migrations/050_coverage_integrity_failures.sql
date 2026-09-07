-- v50: durable, bounded health state when ordered coverage integrity could not
-- be established.
--
-- A stream failure used to be logged and skipped, which let an empty
-- extraction backlog appear healthy. Keep one structural row per session so
-- repeated dream cycles cannot grow storage and never persist source bytes or
-- exception text. A successful complete stream walk removes the row.

CREATE TABLE IF NOT EXISTS coverage_integrity_failures (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    config_version TEXT NOT NULL CHECK (
        config_version = 'lossless-coverage-integrity-v1|coverage=dream-lossless-message-v1|hash=sha256-role-content-v1|record=hymem-message-jsonl-v1'
    ),
    failure_reason TEXT NOT NULL CHECK (
        failure_reason IN ('materialization_failure', 'source_stream_invalid')
    ),
    occurrences INTEGER NOT NULL DEFAULT 1 CHECK (
        occurrences BETWEEN 1 AND 2147483647
    ),
    first_detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE dream_runs ADD COLUMN
    coverage_integrity_failures INTEGER NOT NULL DEFAULT 0
    CHECK(coverage_integrity_failures >= 0);
