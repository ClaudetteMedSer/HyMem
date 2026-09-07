-- v47: prompt-independent terminal state for unrecoverable extraction input.
--
-- v40 deliberately left legacy extraction chunks unmanifested when their
-- exact source membership could not be reconstructed from lossless coverage.
-- Retrying those chunks cannot become safe: Phase 1 correctly fails closed,
-- but the old prompt-scoped retry ledger made every prompt bump reopen the
-- same permanent loss and consume dream_budget ahead of healthy work.
--
-- A terminal-loss row is neither a success nor a prompt-scoped quarantine.
-- It permanently removes the chunk from automatic extraction scheduling,
-- remains visible to operators and portable backups, and can be removed only
-- as an explicit repair before publishing a newly validated source manifest.
CREATE TABLE IF NOT EXISTS chunk_extraction_terminal_losses (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    reason TEXT NOT NULL CHECK (
        reason IN ('source_manifest_unrecoverable')
    ),
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chunk_extraction_terminal_losses_reason
    ON chunk_extraction_terminal_losses(reason);

DROP TRIGGER IF EXISTS processed_chunks_terminal_loss_insert_guard;
CREATE TRIGGER processed_chunks_terminal_loss_insert_guard
BEFORE INSERT ON processed_chunks
WHEN EXISTS (
    SELECT 1 FROM chunk_extraction_terminal_losses loss
    WHERE loss.chunk_id = new.chunk_id
)
BEGIN
    SELECT RAISE(ABORT, 'terminal extraction loss cannot be marked processed');
END;

DROP TRIGGER IF EXISTS processed_chunks_terminal_loss_update_guard;
CREATE TRIGGER processed_chunks_terminal_loss_update_guard
BEFORE UPDATE OF chunk_id, prompt_version ON processed_chunks
WHEN EXISTS (
    SELECT 1 FROM chunk_extraction_terminal_losses loss
    WHERE loss.chunk_id = new.chunk_id
)
BEGIN
    SELECT RAISE(ABORT, 'terminal extraction loss cannot be marked processed');
END;

DROP TRIGGER IF EXISTS chunk_extraction_terminal_loss_clear_processed;
CREATE TRIGGER chunk_extraction_terminal_loss_clear_processed
AFTER INSERT ON chunk_extraction_terminal_losses
BEGIN
    DELETE FROM processed_chunks WHERE chunk_id = new.chunk_id;
END;

DROP TRIGGER IF EXISTS chunk_extraction_terminal_loss_insert_guard;
CREATE TRIGGER chunk_extraction_terminal_loss_insert_guard
BEFORE INSERT ON chunk_extraction_terminal_losses
WHEN NOT EXISTS (
    SELECT 1 FROM chunks c
    WHERE c.id = new.chunk_id
      AND c.chunk_kind = 'extraction'
      AND COALESCE(c.salience_reason, '') <> 'short_session_fallback'
      AND c.source_manifest_version IS NULL
      AND c.source_manifest_count IS NULL
)
BEGIN
    SELECT RAISE(ABORT, 'terminal extraction loss requires an unmanifested extraction chunk');
END;

DROP TRIGGER IF EXISTS chunk_extraction_terminal_loss_update_guard;
CREATE TRIGGER chunk_extraction_terminal_loss_update_guard
BEFORE UPDATE ON chunk_extraction_terminal_losses
BEGIN
    SELECT RAISE(ABORT, 'terminal extraction loss is immutable');
END;

DROP TRIGGER IF EXISTS chunk_extraction_terminal_loss_manifest_guard;
CREATE TRIGGER chunk_extraction_terminal_loss_manifest_guard
BEFORE UPDATE OF source_manifest_version, source_manifest_count ON chunks
WHEN EXISTS (
    SELECT 1 FROM chunk_extraction_terminal_losses loss
    WHERE loss.chunk_id = old.id
)
 AND (
    new.source_manifest_version IS NOT old.source_manifest_version
    OR new.source_manifest_count IS NOT old.source_manifest_count
 )
BEGIN
    SELECT RAISE(ABORT, 'terminal extraction loss must be explicitly resolved');
END;

-- The data backfill is intentionally a Python migration hook.  It first
-- materializes exact coverage for any surviving raw messages and re-runs the
-- conservative v40 manifest recovery.  Only the chunks still unmanifested
-- after that proof-preserving recovery are inserted here as terminal losses.
