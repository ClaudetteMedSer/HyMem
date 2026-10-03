-- v52: durable Phase-1 source-producer acknowledgement.
--
-- Lossless coverage is committed at public ingestion time, before dreaming.
-- It therefore cannot prove that the salience and baseline chunk producers
-- have examined the covered source.  These operational cursors are published
-- only after both builders and candidate persistence complete.  NULL on an
-- existing store deliberately reopens every non-empty covered session once.

ALTER TABLE sessions ADD COLUMN source_materialized_message_id INTEGER;
ALTER TABLE sessions ADD COLUMN source_materialization_config_version TEXT;

-- Extraction chunks are a reconstructible cache over the protected source
-- stream. Deleting one (retention, maintenance, or direct SQL) revokes the
-- producer acknowledgement in the same transaction, so a health poll cannot
-- mistake an evicted untouched candidate for completed Phase-1 work. Coverage
-- artifacts have a distinct lifecycle and deliberately do not revoke it.
DROP TRIGGER IF EXISTS extraction_chunk_delete_invalidates_source_materialization;
CREATE TRIGGER extraction_chunk_delete_invalidates_source_materialization
AFTER DELETE ON chunks
WHEN old.chunk_kind = 'extraction'
BEGIN
    UPDATE sessions
    SET source_materialized_message_id = NULL,
        source_materialization_config_version = NULL
    WHERE id = old.session_id;
END;
