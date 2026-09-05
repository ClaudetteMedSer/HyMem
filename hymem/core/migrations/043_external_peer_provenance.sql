-- v43: exact external/Honcho peer provenance.
--
-- Roles are coarse behavioral classes, not author identities. New Honcho
-- traffic carries a workspace-qualified peer through the raw message,
-- immutable lossless proof, dream-time evidence, and portable citations.
-- Existing rows remain NULL/NULL: migration must never guess an identity from
-- a role, session id, or current peer registry.

ALTER TABLE sessions ADD COLUMN source_workspace_id TEXT CHECK (
    source_workspace_id IS NULL OR length(trim(source_workspace_id)) > 0
);
ALTER TABLE messages ADD COLUMN source_peer_id TEXT;
ALTER TABLE messages ADD COLUMN source_workspace_id TEXT;
ALTER TABLE message_retention_coverage ADD COLUMN source_peer_id TEXT;
ALTER TABLE message_retention_coverage ADD COLUMN source_workspace_id TEXT;
ALTER TABLE kg_evidence ADD COLUMN source_peer_id TEXT;
ALTER TABLE kg_evidence ADD COLUMN source_workspace_id TEXT;

CREATE TABLE IF NOT EXISTS session_peers (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL,
    peer_id TEXT NOT NULL,
    configuration TEXT NOT NULL DEFAULT '{}',
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, workspace_id, peer_id),
    FOREIGN KEY (peer_id, workspace_id)
        REFERENCES peers(id, workspace_id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_session_peers_peer
    ON session_peers(workspace_id, peer_id, session_id);

DROP TRIGGER IF EXISTS message_coverage_fts_insert;
DROP TRIGGER IF EXISTS message_coverage_fts_delete;
DROP TRIGGER IF EXISTS message_coverage_fts_update_delete;
DROP TRIGGER IF EXISTS message_coverage_fts_update_insert;
DROP TABLE IF EXISTS message_coverage_fts;
CREATE VIRTUAL TABLE message_coverage_fts USING fts5(
    content,
    content='',
    tokenize='porter unicode61'
);
INSERT INTO message_coverage_fts(rowid, content)
SELECT rowid, json_extract(text, '$.content')
FROM chunks
WHERE chunk_kind = 'coverage'
  AND json_valid(text)
  AND json_type(text, '$.content') = 'text';
CREATE TRIGGER message_coverage_fts_insert
AFTER INSERT ON chunks
WHEN new.chunk_kind = 'coverage'
 AND json_valid(new.text)
 AND json_type(new.text, '$.content') = 'text' BEGIN
    INSERT INTO message_coverage_fts(rowid, content)
    VALUES (new.rowid, json_extract(new.text, '$.content'));
END;
CREATE TRIGGER message_coverage_fts_delete
AFTER DELETE ON chunks
WHEN old.chunk_kind = 'coverage'
 AND json_valid(old.text)
 AND json_type(old.text, '$.content') = 'text' BEGIN
    INSERT INTO message_coverage_fts(message_coverage_fts, rowid, content)
    VALUES ('delete', old.rowid, json_extract(old.text, '$.content'));
END;
CREATE TRIGGER message_coverage_fts_update_delete
AFTER UPDATE OF text, chunk_kind ON chunks
WHEN old.chunk_kind = 'coverage'
 AND json_valid(old.text)
 AND json_type(old.text, '$.content') = 'text' BEGIN
    INSERT INTO message_coverage_fts(message_coverage_fts, rowid, content)
    VALUES ('delete', old.rowid, json_extract(old.text, '$.content'));
END;
CREATE TRIGGER message_coverage_fts_update_insert
AFTER UPDATE OF text, chunk_kind ON chunks
WHEN new.chunk_kind = 'coverage'
 AND json_valid(new.text)
 AND json_type(new.text, '$.content') = 'text' BEGIN
    INSERT INTO message_coverage_fts(rowid, content)
    VALUES (new.rowid, json_extract(new.text, '$.content'));
END;

-- v37's lifecycle guards knew only the role/content v1 proof. Refresh them so
-- attributed v43 records use the identity-bound v2 proof while native legacy
-- rows continue to validate under the frozen v1 representation.
DROP TRIGGER IF EXISTS message_retention_coverage_delete_guard;
CREATE TRIGGER message_retention_coverage_delete_guard
BEFORE DELETE ON message_retention_coverage
WHEN NOT EXISTS (
    SELECT 1 FROM messages message
    JOIN chunks source_chunk ON source_chunk.id = old.chunk_id
    WHERE message.id = old.message_id
      AND message.session_id = old.source_session_id
      AND message.role = old.source_role
      AND message.source_peer_id IS old.source_peer_id
      AND message.source_workspace_id IS old.source_workspace_id
      AND message.created_at IS old.source_created_at
      AND json_extract(source_chunk.text, '$.content') = message.content
      AND hymem_message_record_proof_valid(
            source_chunk.text, old.message_content_hash,
            old.hash_version, old.record_version
          ) = 1
)
BEGIN
    SELECT RAISE(ABORT, 'cannot release coverage while raw source is absent');
END;

DROP TRIGGER IF EXISTS message_retention_coverage_update_guard;
CREATE TRIGGER message_retention_coverage_update_guard
BEFORE UPDATE ON message_retention_coverage
WHEN NOT EXISTS (
    SELECT 1 FROM messages message
    JOIN chunks source_chunk ON source_chunk.id = old.chunk_id
    WHERE message.id = old.message_id
      AND message.session_id = old.source_session_id
      AND message.role = old.source_role
      AND message.source_peer_id IS old.source_peer_id
      AND message.source_workspace_id IS old.source_workspace_id
      AND message.created_at IS old.source_created_at
      AND json_extract(source_chunk.text, '$.content') = message.content
      AND hymem_message_record_proof_valid(
            source_chunk.text, old.message_content_hash,
            old.hash_version, old.record_version
          ) = 1
)
BEGIN
    SELECT RAISE(ABORT, 'cannot mutate coverage while raw source is absent');
END;

DROP TRIGGER IF EXISTS session_workspace_binding_guard;
CREATE TRIGGER session_workspace_binding_guard
BEFORE UPDATE OF source_workspace_id ON sessions
WHEN (
    old.source_workspace_id IS NOT NULL
    AND new.source_workspace_id IS NOT old.source_workspace_id
) OR (
    old.source_workspace_id IS NULL
    AND new.source_workspace_id IS NOT NULL
    AND (
      length(trim(new.source_workspace_id)) = 0
      OR old.ended_at IS NOT NULL
      OR old.summary IS NOT NULL
      OR old.digested_prompt_version IS NOT NULL
      OR old.profile_prompt_version IS NOT NULL
      OR old.profile_cursor_message_id IS NOT NULL
      OR old.profile_cursor_partial_message_id IS NOT NULL
      OR old.profile_cursor_offset <> 0
      OR old.profile_cursor_prompt_version IS NOT NULL
      OR old.profile_published_generation IS NOT NULL
      OR old.profile_retry_count <> 0
      OR old.profile_retry_config_version IS NOT NULL
      OR old.profile_quarantined <> 0
      OR old.digested_message_id IS NOT NULL
      OR old.facts_message_id IS NOT NULL
      OR old.episodes_prompt_version IS NOT NULL
      OR old.coverage_message_id IS NOT NULL
      OR old.digest_cursor_message_id IS NOT NULL
      OR old.digest_cursor_partial_message_id IS NOT NULL
      OR old.digest_cursor_offset <> 0
      OR old.digest_cursor_prompt_version IS NOT NULL
      OR old.digest_published_generation IS NOT NULL
      OR old.digest_retry_count <> 0
      OR old.digest_retry_config_version IS NOT NULL
      OR old.digest_quarantined <> 0
      OR old.auto_summary IS NOT NULL
      OR old.auto_summary_message_id IS NOT NULL
      OR old.auto_summary_partial_message_id IS NOT NULL
      OR old.auto_summary_message_offset <> 0
      OR old.summary_source IS NOT NULL
      OR EXISTS (SELECT 1 FROM messages WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM chunks WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM episodes WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM procedures WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM profile_staging WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM temporal_mentions WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM narrative_facts WHERE session_id = old.id)
      OR EXISTS (
        SELECT 1 FROM message_retention_coverage
        WHERE source_session_id = old.id
      )
      OR EXISTS (
        SELECT 1 FROM chunk_message_sources
        WHERE source_session_id = old.id
      )
      OR EXISTS (
        SELECT 1 FROM user_profile WHERE source_session_id = old.id
      )
      OR EXISTS (
        SELECT 1 FROM kg_evidence WHERE source_session_id = old.id
      )
      OR EXISTS (
        SELECT 1 FROM kg_claim_observations
        WHERE source_session_id = old.id
      )
      OR EXISTS (SELECT 1 FROM session_peers WHERE session_id = old.id)
    )
)
BEGIN
    SELECT RAISE(ABORT, 'session workspace binding is immutable');
END;

DROP TRIGGER IF EXISTS session_peer_binding_insert_guard;
CREATE TRIGGER session_peer_binding_insert_guard
BEFORE INSERT ON session_peers
WHEN NOT EXISTS (
    SELECT 1 FROM sessions session
    WHERE session.id = new.session_id
      AND session.source_workspace_id = new.workspace_id
)
BEGIN
    SELECT RAISE(ABORT, 'session peer workspace does not match session');
END;

DROP TRIGGER IF EXISTS session_peer_binding_update_guard;
CREATE TRIGGER session_peer_binding_update_guard
BEFORE UPDATE OF session_id, workspace_id, peer_id ON session_peers
WHEN NOT EXISTS (
    SELECT 1 FROM sessions session
    WHERE session.id = new.session_id
      AND session.source_workspace_id = new.workspace_id
)
OR EXISTS (
    SELECT 1 FROM messages message
    WHERE message.session_id = old.session_id
      AND message.source_workspace_id = old.workspace_id
      AND message.source_peer_id = old.peer_id
)
OR EXISTS (
    SELECT 1 FROM message_retention_coverage coverage
    WHERE coverage.source_session_id = old.session_id
      AND coverage.source_workspace_id = old.workspace_id
      AND coverage.source_peer_id = old.peer_id
)
BEGIN
    SELECT RAISE(ABORT, 'session peer identity is invalid or referenced');
END;

DROP TRIGGER IF EXISTS session_peer_delete_guard;
CREATE TRIGGER session_peer_delete_guard
BEFORE DELETE ON session_peers
WHEN EXISTS (
    SELECT 1 FROM messages message
    WHERE message.session_id = old.session_id
      AND message.source_workspace_id = old.workspace_id
      AND message.source_peer_id = old.peer_id
)
OR EXISTS (
    SELECT 1 FROM message_retention_coverage coverage
    WHERE coverage.source_session_id = old.session_id
      AND coverage.source_workspace_id = old.workspace_id
      AND coverage.source_peer_id = old.peer_id
)
BEGIN
    SELECT RAISE(ABORT, 'referenced session peer is immutable');
END;

DROP TRIGGER IF EXISTS peer_identity_update_guard;
CREATE TRIGGER peer_identity_update_guard
BEFORE UPDATE OF id, workspace_id, role ON peers
WHEN EXISTS (
    SELECT 1 FROM messages message
    WHERE message.source_workspace_id = old.workspace_id
      AND message.source_peer_id = old.id
)
OR EXISTS (
    SELECT 1 FROM message_retention_coverage coverage
    WHERE coverage.source_workspace_id = old.workspace_id
      AND coverage.source_peer_id = old.id
)
BEGIN
    SELECT RAISE(ABORT, 'referenced peer identity is immutable');
END;

DROP TRIGGER IF EXISTS peer_identity_delete_guard;
CREATE TRIGGER peer_identity_delete_guard
BEFORE DELETE ON peers
WHEN EXISTS (
    SELECT 1 FROM messages message
    WHERE message.source_workspace_id = old.workspace_id
      AND message.source_peer_id = old.id
)
OR EXISTS (
    SELECT 1 FROM message_retention_coverage coverage
    WHERE coverage.source_workspace_id = old.workspace_id
      AND coverage.source_peer_id = old.id
)
BEGIN
    SELECT RAISE(ABORT, 'referenced peer identity is immutable');
END;

DROP TRIGGER IF EXISTS message_external_provenance_insert_guard;
CREATE TRIGGER message_external_provenance_insert_guard
BEFORE INSERT ON messages
WHEN NOT (
    (new.source_peer_id IS NULL AND new.source_workspace_id IS NULL
     AND EXISTS (
       SELECT 1 FROM sessions session
       WHERE session.id = new.session_id
         AND session.source_workspace_id IS NULL
     ))
    OR (
      new.source_peer_id IS NOT NULL
      AND length(trim(new.source_peer_id)) > 0
      AND new.source_workspace_id IS NOT NULL
      AND length(trim(new.source_workspace_id)) > 0
      AND EXISTS (
        SELECT 1 FROM sessions session
        WHERE session.id = new.session_id
          AND session.source_workspace_id = new.source_workspace_id
      )
      AND EXISTS (
        SELECT 1 FROM peers peer
        WHERE peer.id = new.source_peer_id
          AND peer.workspace_id = new.source_workspace_id
          AND peer.role = new.role
      )
      AND EXISTS (
        SELECT 1 FROM session_peers member
        WHERE member.session_id = new.session_id
          AND member.workspace_id = new.source_workspace_id
          AND member.peer_id = new.source_peer_id
      )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'invalid external message provenance');
END;

DROP TRIGGER IF EXISTS message_external_provenance_update_guard;
CREATE TRIGGER message_external_provenance_update_guard
BEFORE UPDATE OF session_id, role, source_peer_id, source_workspace_id ON messages
WHEN NOT (
    (new.source_peer_id IS NULL AND new.source_workspace_id IS NULL
     AND EXISTS (
       SELECT 1 FROM sessions session
       WHERE session.id = new.session_id
         AND session.source_workspace_id IS NULL
     ))
    OR (
      new.source_peer_id IS NOT NULL
      AND length(trim(new.source_peer_id)) > 0
      AND new.source_workspace_id IS NOT NULL
      AND length(trim(new.source_workspace_id)) > 0
      AND EXISTS (
        SELECT 1 FROM sessions session
        WHERE session.id = new.session_id
          AND session.source_workspace_id = new.source_workspace_id
      )
      AND EXISTS (
        SELECT 1 FROM peers peer
        WHERE peer.id = new.source_peer_id
          AND peer.workspace_id = new.source_workspace_id
          AND peer.role = new.role
      )
      AND EXISTS (
        SELECT 1 FROM session_peers member
        WHERE member.session_id = new.session_id
          AND member.workspace_id = new.source_workspace_id
          AND member.peer_id = new.source_peer_id
      )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'invalid external message provenance');
END;

-- Refresh the ordered-source guard so attribution is append-only together
-- with role/content/time once its lossless proof exists.
DROP TRIGGER IF EXISTS message_lossless_source_update_guard;
CREATE TRIGGER message_lossless_source_update_guard
BEFORE UPDATE OF id, session_id, role, source_peer_id, source_workspace_id,
                 content, created_at ON messages
WHEN EXISTS (
    SELECT 1 FROM message_retention_coverage mc
    WHERE mc.message_id = old.id
      AND mc.coverage_version = 'dream-lossless-message-v1'
) BEGIN
    SELECT RAISE(ABORT, 'ordered digest source is immutable');
END;

DROP TRIGGER IF EXISTS message_coverage_peer_insert_guard;
CREATE TRIGGER message_coverage_peer_insert_guard
BEFORE INSERT ON message_retention_coverage
WHEN NOT (
  (
    (
      new.source_peer_id IS NULL
      AND new.source_workspace_id IS NULL
      AND EXISTS (
        SELECT 1 FROM sessions session
        WHERE session.id = new.source_session_id
          AND session.source_workspace_id IS NULL
      )
      AND NOT EXISTS (
        SELECT 1 FROM messages message
        WHERE message.id = new.message_id
          AND message.session_id = new.source_session_id
          AND NOT (
            message.role = new.source_role
            AND message.source_peer_id IS NULL
            AND message.source_workspace_id IS NULL
            AND message.created_at IS new.source_created_at
            AND hymem_message_record_matches_raw_source(
                  (SELECT source_chunk.text FROM chunks source_chunk
                   WHERE source_chunk.id = new.chunk_id),
                  message.id, message.session_id, message.role,
                  message.content, message.created_at,
                  message.source_peer_id, message.source_workspace_id,
                  new.message_content_hash, new.hash_version,
                  new.record_version
                ) = 1
          )
      )
    )
    OR (
      new.source_peer_id IS NOT NULL
      AND length(trim(new.source_peer_id)) > 0
      AND new.source_workspace_id IS NOT NULL
      AND length(trim(new.source_workspace_id)) > 0
      AND EXISTS (
        SELECT 1 FROM sessions session
        WHERE session.id = new.source_session_id
          AND session.source_workspace_id = new.source_workspace_id
      )
      AND EXISTS (
        SELECT 1 FROM peers peer
        WHERE peer.id = new.source_peer_id
          AND peer.workspace_id = new.source_workspace_id
          AND peer.role = new.source_role
      )
      AND EXISTS (
        SELECT 1 FROM session_peers member
        WHERE member.session_id = new.source_session_id
          AND member.workspace_id = new.source_workspace_id
          AND member.peer_id = new.source_peer_id
      )
      AND NOT EXISTS (
        SELECT 1 FROM messages message
        WHERE message.id = new.message_id
          AND message.session_id = new.source_session_id
          AND NOT (
            message.role = new.source_role
            AND message.source_peer_id = new.source_peer_id
            AND message.source_workspace_id = new.source_workspace_id
            AND message.created_at IS new.source_created_at
            AND hymem_message_record_matches_raw_source(
                  (SELECT source_chunk.text FROM chunks source_chunk
                   WHERE source_chunk.id = new.chunk_id),
                  message.id, message.session_id, message.role,
                  message.content, message.created_at,
                  message.source_peer_id, message.source_workspace_id,
                  new.message_content_hash, new.hash_version,
                  new.record_version
                ) = 1
          )
      )
    )
  )
  AND EXISTS (
    SELECT 1 FROM chunks source_chunk
    WHERE source_chunk.id = new.chunk_id
      AND hymem_message_record_matches_source(
            source_chunk.text, new.message_id, new.source_session_id,
            new.source_role, new.source_created_at, new.source_peer_id,
            new.source_workspace_id, new.message_content_hash,
            new.hash_version, new.record_version
          ) = 1
      AND (
        new.coverage_version <> 'dream-lossless-message-v1'
        OR (
          source_chunk.id = hymem_coverage_chunk_id(
              new.source_session_id, new.message_id
          )
          AND source_chunk.session_id = new.source_session_id
          AND source_chunk.start_message_id = new.message_id
          AND source_chunk.end_message_id = new.message_id
          AND source_chunk.chunk_kind = 'coverage'
          AND json_valid(source_chunk.text)
        )
      )
  )
)
BEGIN
    SELECT RAISE(ABORT, 'invalid external coverage provenance');
END;

DROP TRIGGER IF EXISTS message_coverage_peer_update_guard;
CREATE TRIGGER message_coverage_peer_update_guard
BEFORE UPDATE ON message_retention_coverage
WHEN NOT (
  (
    (
      new.source_peer_id IS NULL
      AND new.source_workspace_id IS NULL
      AND EXISTS (
        SELECT 1 FROM sessions session
        WHERE session.id = new.source_session_id
          AND session.source_workspace_id IS NULL
      )
      AND NOT EXISTS (
        SELECT 1 FROM messages message
        WHERE message.id = new.message_id
          AND message.session_id = new.source_session_id
          AND NOT (
            message.role = new.source_role
            AND message.source_peer_id IS NULL
            AND message.source_workspace_id IS NULL
            AND message.created_at IS new.source_created_at
            AND hymem_message_record_matches_raw_source(
                  (SELECT source_chunk.text FROM chunks source_chunk
                   WHERE source_chunk.id = new.chunk_id),
                  message.id, message.session_id, message.role,
                  message.content, message.created_at,
                  message.source_peer_id, message.source_workspace_id,
                  new.message_content_hash, new.hash_version,
                  new.record_version
                ) = 1
          )
      )
    )
    OR (
      new.source_peer_id IS NOT NULL
      AND length(trim(new.source_peer_id)) > 0
      AND new.source_workspace_id IS NOT NULL
      AND length(trim(new.source_workspace_id)) > 0
      AND EXISTS (
        SELECT 1 FROM sessions session
        WHERE session.id = new.source_session_id
          AND session.source_workspace_id = new.source_workspace_id
      )
      AND EXISTS (
        SELECT 1 FROM peers peer
        WHERE peer.id = new.source_peer_id
          AND peer.workspace_id = new.source_workspace_id
          AND peer.role = new.source_role
      )
      AND EXISTS (
        SELECT 1 FROM session_peers member
        WHERE member.session_id = new.source_session_id
          AND member.workspace_id = new.source_workspace_id
          AND member.peer_id = new.source_peer_id
      )
      AND NOT EXISTS (
        SELECT 1 FROM messages message
        WHERE message.id = new.message_id
          AND message.session_id = new.source_session_id
          AND NOT (
            message.role = new.source_role
            AND message.source_peer_id = new.source_peer_id
            AND message.source_workspace_id = new.source_workspace_id
            AND message.created_at IS new.source_created_at
            AND hymem_message_record_matches_raw_source(
                  (SELECT source_chunk.text FROM chunks source_chunk
                   WHERE source_chunk.id = new.chunk_id),
                  message.id, message.session_id, message.role,
                  message.content, message.created_at,
                  message.source_peer_id, message.source_workspace_id,
                  new.message_content_hash, new.hash_version,
                  new.record_version
                ) = 1
          )
      )
    )
  )
  AND EXISTS (
    SELECT 1 FROM chunks source_chunk
    WHERE source_chunk.id = new.chunk_id
      AND hymem_message_record_matches_source(
            source_chunk.text, new.message_id, new.source_session_id,
            new.source_role, new.source_created_at, new.source_peer_id,
            new.source_workspace_id, new.message_content_hash,
            new.hash_version, new.record_version
          ) = 1
      AND (
        new.coverage_version <> 'dream-lossless-message-v1'
        OR (
          source_chunk.id = hymem_coverage_chunk_id(
              new.source_session_id, new.message_id
          )
          AND source_chunk.session_id = new.source_session_id
          AND source_chunk.start_message_id = new.message_id
          AND source_chunk.end_message_id = new.message_id
          AND source_chunk.chunk_kind = 'coverage'
          AND json_valid(source_chunk.text)
        )
      )
  )
)
BEGIN
    SELECT RAISE(ABORT, 'invalid external coverage provenance');
END;

DROP TRIGGER IF EXISTS kg_evidence_v43_peer_insert_guard;
CREATE TRIGGER kg_evidence_v43_peer_insert_guard
BEFORE INSERT ON kg_evidence
WHEN NOT (
    (
      new.provenance_status = 'legacy_unattributed'
      AND new.source_peer_id IS NULL
      AND new.source_workspace_id IS NULL
    )
    OR (
      new.provenance_status = 'canonical'
      AND EXISTS (
        SELECT 1 FROM message_retention_coverage coverage
        JOIN chunks source_chunk ON source_chunk.id = coverage.chunk_id
        WHERE coverage.message_id = new.source_message_id
          AND coverage.chunk_id = new.source_coverage_chunk_id
          AND coverage.coverage_version = new.source_coverage_version
          AND coverage.source_peer_id IS new.source_peer_id
          AND coverage.source_workspace_id IS new.source_workspace_id
          AND hymem_message_record_matches_source(
                source_chunk.text, coverage.message_id,
                coverage.source_session_id, coverage.source_role,
                coverage.source_created_at, coverage.source_peer_id,
                coverage.source_workspace_id, coverage.message_content_hash,
                coverage.hash_version, coverage.record_version
              ) = 1
      )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'kg evidence external peer provenance mismatch');
END;

DROP TRIGGER IF EXISTS kg_evidence_v43_peer_update_guard;
CREATE TRIGGER kg_evidence_v43_peer_update_guard
BEFORE UPDATE OF source_peer_id, source_workspace_id, source_message_id,
                 source_coverage_chunk_id, source_coverage_version,
                 provenance_status ON kg_evidence
BEGIN
    SELECT RAISE(ABORT, 'kg evidence revisions are internally managed')
    WHERE hymem_evidence_mutation_authorized() <> 1;
    SELECT RAISE(ABORT, 'published evidence peer provenance is immutable')
    WHERE old.published_at IS NOT NULL
      AND hymem_evidence_history_authorized() <> 1
      AND (
        new.source_peer_id IS NOT old.source_peer_id
        OR new.source_workspace_id IS NOT old.source_workspace_id
      );
    SELECT RAISE(ABORT, 'kg evidence external peer provenance mismatch')
    WHERE NOT (
      (
        new.provenance_status = 'legacy_unattributed'
        AND new.source_peer_id IS NULL
        AND new.source_workspace_id IS NULL
      )
      OR (
        new.provenance_status = 'canonical'
        AND EXISTS (
          SELECT 1 FROM message_retention_coverage coverage
          JOIN chunks source_chunk ON source_chunk.id = coverage.chunk_id
          WHERE coverage.message_id = new.source_message_id
            AND coverage.chunk_id = new.source_coverage_chunk_id
            AND coverage.coverage_version = new.source_coverage_version
            AND coverage.source_peer_id IS new.source_peer_id
            AND coverage.source_workspace_id IS new.source_workspace_id
            AND hymem_message_record_matches_source(
                  source_chunk.text, coverage.message_id,
                  coverage.source_session_id, coverage.source_role,
                  coverage.source_created_at, coverage.source_peer_id,
                  coverage.source_workspace_id,
                  coverage.message_content_hash, coverage.hash_version,
                  coverage.record_version
                ) = 1
        )
      )
    );
END;
