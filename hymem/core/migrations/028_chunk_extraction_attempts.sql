-- v28: durable held-attempt bound + marker write idempotence (2026-08-08).
--
-- Numbered 028 deliberately: v27 (dream_runs deficit attribution) lands from a
-- separate branch. The runner applies every file whose number exceeds the DB's
-- schema_version, so the two are order-independent and a gap is harmless.
--
-- PART 1 — chunk_extraction_attempts.
--
-- Ingest failures are now HELD rather than marked done (a failed extraction
-- leaves no processed_chunks row, so the next dream retries it). That closes
-- the permanent-hole class, but replaces it with an unbounded one: a chunk
-- that fails forever is retried on every dream, and because the runner
-- decrements its budget BEFORE extraction, each held chunk permanently
-- consumes one of dream_budget's slots. A large stuck cohort therefore starves
-- new chunks — the same shape as the fusion node that failed sixteen dreams
-- running, moved onto the ingest path.
--
-- This table bounds it: failures accrue per (chunk_id, prompt_version), and
-- once the count reaches chunk_extraction_max_attempts the chunk is marked
-- done and logged loudly. Giving up is a deliberate, audible, countable act
-- rather than the silent default it used to be. A success clears the row, so
-- the count is consecutive failures and a chunk that heals starts fresh.
--
-- Marker write idempotence is NOT here, deliberately. The natural design was
-- UNIQUE(chunk_id, kind, statement) on behavioral_markers, mirroring the
-- UNIQUE(edge_id, chunk_id, polarity) that already makes kg_evidence safe to
-- re-attach — but creating it requires deduping first, and with
-- PRAGMA foreign_keys=ON SQLite resolves a child table's FK parent when it
-- PREPARES the DELETE, not per row. behavioral_markers references chunks, so
-- the dedupe raises "no such table: main.chunks" on any legacy store that
-- predates chunks (an empty table and an EXISTS guard do not help: the failure
-- is at prepare time). Stubbing chunks here would risk minting a wrong-shaped
-- table that nothing later corrects. So idempotence is enforced at the write
-- instead, by phase1.persist_chunk_results, and fresh and migrated databases
-- stay identical.
CREATE TABLE IF NOT EXISTS chunk_extraction_attempts (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    prompt_version TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_failure_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_id, prompt_version)
);
