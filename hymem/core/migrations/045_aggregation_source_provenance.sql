-- v45: exact, retention-verifiable source manifests for episodes and RAPTOR
-- aggregation nodes.  Historical rows are intentionally left incomplete; a
-- numeric message range cannot be promoted into exact occurrence provenance.

ALTER TABLE episodes ADD COLUMN source_manifest_version TEXT;
ALTER TABLE episodes ADD COLUMN source_manifest_count INTEGER NOT NULL DEFAULT 0
    CHECK (source_manifest_count >= 0);
ALTER TABLE episodes ADD COLUMN source_manifest_hash TEXT;
ALTER TABLE episodes ADD COLUMN source_manifest_complete BOOLEAN NOT NULL DEFAULT 0
    CHECK (source_manifest_complete IN (0, 1));

ALTER TABLE aggregation_nodes ADD COLUMN source_manifest_version TEXT;
ALTER TABLE aggregation_nodes ADD COLUMN source_manifest_count INTEGER NOT NULL DEFAULT 0
    CHECK (source_manifest_count >= 0);
ALTER TABLE aggregation_nodes ADD COLUMN source_manifest_hash TEXT;
ALTER TABLE aggregation_nodes ADD COLUMN source_manifest_complete BOOLEAN NOT NULL DEFAULT 0
    CHECK (source_manifest_complete IN (0, 1));
ALTER TABLE aggregation_nodes ADD COLUMN input_fingerprint TEXT;

CREATE TABLE IF NOT EXISTS episode_source_occurrences (
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    source_message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_role TEXT NOT NULL
        CHECK (source_role IN ('user','assistant','system','tool')),
    source_peer_id TEXT,
    source_workspace_id TEXT,
    source_created_at TIMESTAMP,
    source_coverage_chunk_id TEXT NOT NULL,
    source_coverage_version TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    PRIMARY KEY (episode_id, ordinal),
    UNIQUE (episode_id, source_session_id, source_message_id),
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_episode_source_occurrence
    ON episode_source_occurrences(source_session_id, source_message_id);

CREATE TABLE IF NOT EXISTS aggregation_node_source_occurrences (
    node_id TEXT NOT NULL REFERENCES aggregation_nodes(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    source_message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_role TEXT NOT NULL
        CHECK (source_role IN ('user','assistant','system','tool')),
    source_peer_id TEXT,
    source_workspace_id TEXT,
    source_created_at TIMESTAMP,
    source_coverage_chunk_id TEXT NOT NULL,
    source_coverage_version TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    PRIMARY KEY (node_id, ordinal),
    UNIQUE (node_id, source_session_id, source_message_id),
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_aggregation_source_occurrence
    ON aggregation_node_source_occurrences(source_session_id, source_message_id);

CREATE TRIGGER IF NOT EXISTS episode_source_header_insert_guard
BEFORE INSERT ON episodes
WHEN NOT (
    new.source_manifest_complete = 0
    AND new.source_manifest_count = 0
    AND new.source_manifest_hash IS NULL
    AND new.source_manifest_version IS NULL
) BEGIN
    SELECT RAISE(ABORT, 'episode source manifest must publish after its children');
END;

CREATE TRIGGER IF NOT EXISTS episode_source_header_update_guard
BEFORE UPDATE OF source_manifest_version, source_manifest_count,
                 source_manifest_hash, source_manifest_complete ON episodes
WHEN NOT (
    (new.source_manifest_complete = 0
     AND new.source_manifest_count = 0
     AND new.source_manifest_hash IS NULL
     AND new.source_manifest_version IS NULL)
    OR
    (new.source_manifest_complete = 1
     AND new.source_manifest_version = 'episode-source-manifest-v1'
     AND new.source_manifest_count > 0
     AND length(new.source_manifest_hash) = 71
     AND new.source_manifest_hash GLOB 'sha256:*'
     AND (SELECT COUNT(*) FROM episode_source_occurrences source
          WHERE source.episode_id = new.id) = new.source_manifest_count
     AND (SELECT MIN(ordinal) FROM episode_source_occurrences source
          WHERE source.episode_id = new.id) = 0
     AND (SELECT MAX(ordinal) FROM episode_source_occurrences source
          WHERE source.episode_id = new.id) = new.source_manifest_count - 1)
) BEGIN
    SELECT RAISE(ABORT, 'invalid episode source manifest publication');
END;

CREATE TRIGGER IF NOT EXISTS episode_source_bound_update_guard
BEFORE UPDATE OF session_id, title, summary ON episodes
WHEN old.source_manifest_complete = 1
BEGIN
    SELECT RAISE(ABORT, 'unpublish episode source manifest before changing input');
END;

CREATE TRIGGER IF NOT EXISTS episode_source_occurrence_insert_guard
BEFORE INSERT ON episode_source_occurrences
WHEN NOT EXISTS (
    SELECT 1
    FROM episodes episode
    JOIN message_retention_coverage proof
      ON proof.message_id = new.source_message_id
     AND proof.chunk_id = new.source_coverage_chunk_id
     AND proof.coverage_version = new.source_coverage_version
    WHERE episode.id = new.episode_id
      AND episode.source_manifest_complete = 0
      AND episode.session_id = new.source_session_id
      AND proof.source_session_id = new.source_session_id
      AND proof.source_role = new.source_role
      AND proof.source_peer_id IS new.source_peer_id
      AND proof.source_workspace_id IS new.source_workspace_id
      AND proof.source_created_at IS new.source_created_at
      AND proof.message_content_hash = new.source_content_hash
      AND proof.coverage_version = 'dream-lossless-message-v1'
) BEGIN
    SELECT RAISE(ABORT, 'episode source occurrence mismatches coverage');
END;

CREATE TRIGGER IF NOT EXISTS episode_source_occurrence_update_guard
BEFORE UPDATE ON episode_source_occurrences
BEGIN
    SELECT RAISE(ABORT, 'episode source occurrences are immutable');
END;

CREATE TRIGGER IF NOT EXISTS episode_source_occurrence_delete_unpublishes
AFTER DELETE ON episode_source_occurrences
BEGIN
    UPDATE episodes
    SET source_manifest_version = NULL,
        source_manifest_count = 0,
        source_manifest_hash = NULL,
        source_manifest_complete = 0
    WHERE id = old.episode_id AND source_manifest_complete = 1;
END;

CREATE TRIGGER IF NOT EXISTS aggregation_source_header_insert_guard
BEFORE INSERT ON aggregation_nodes
WHEN NOT (
    new.source_manifest_complete = 0
    AND new.source_manifest_count = 0
    AND new.source_manifest_hash IS NULL
    AND (new.source_manifest_version IS NULL OR
         new.source_manifest_version = 'aggregation-source-manifest-v1')
    AND (new.input_fingerprint IS NULL OR
         (length(new.input_fingerprint) = 71
          AND new.input_fingerprint GLOB 'sha256:*'))
) BEGIN
    SELECT RAISE(ABORT, 'aggregation source manifest must publish after its children');
END;

CREATE TRIGGER IF NOT EXISTS aggregation_source_header_update_guard
BEFORE UPDATE OF source_manifest_version, source_manifest_count,
                 source_manifest_hash, source_manifest_complete ON aggregation_nodes
WHEN NOT (
    (new.source_manifest_complete = 0
     AND new.source_manifest_count = 0
     AND new.source_manifest_hash IS NULL
     AND (new.source_manifest_version IS NULL OR
          new.source_manifest_version = 'aggregation-source-manifest-v1')
     AND (new.input_fingerprint IS NULL OR
          (length(new.input_fingerprint) = 71
           AND new.input_fingerprint GLOB 'sha256:*')))
    OR
    (new.source_manifest_complete = 1
     AND new.source_manifest_version = 'aggregation-source-manifest-v1'
     AND new.source_manifest_count > 0
     AND length(new.source_manifest_hash) = 71
     AND new.source_manifest_hash GLOB 'sha256:*'
     AND length(new.input_fingerprint) = 71
     AND new.input_fingerprint GLOB 'sha256:*'
     AND (SELECT COUNT(*) FROM aggregation_node_source_occurrences source
          WHERE source.node_id = new.id) = new.source_manifest_count
     AND (SELECT MIN(ordinal) FROM aggregation_node_source_occurrences source
          WHERE source.node_id = new.id) = 0
     AND (SELECT MAX(ordinal) FROM aggregation_node_source_occurrences source
          WHERE source.node_id = new.id) = new.source_manifest_count - 1)
) BEGIN
    SELECT RAISE(ABORT, 'invalid aggregation source manifest publication');
END;

CREATE TRIGGER IF NOT EXISTS aggregation_source_bound_update_guard
BEFORE UPDATE OF title, summary, member_episode_ids, session_ids,
                 n_members, n_sessions, level, is_root,
                 input_fingerprint ON aggregation_nodes
WHEN old.source_manifest_complete = 1
BEGIN
    SELECT RAISE(ABORT, 'unpublish aggregation source manifest before changing input');
END;

CREATE TRIGGER IF NOT EXISTS aggregation_source_occurrence_insert_guard
BEFORE INSERT ON aggregation_node_source_occurrences
WHEN NOT EXISTS (
    SELECT 1
    FROM aggregation_nodes node
    JOIN message_retention_coverage proof
      ON proof.message_id = new.source_message_id
     AND proof.chunk_id = new.source_coverage_chunk_id
     AND proof.coverage_version = new.source_coverage_version
    WHERE node.id = new.node_id
      AND node.source_manifest_complete = 0
      AND proof.source_session_id = new.source_session_id
      AND proof.source_role = new.source_role
      AND proof.source_peer_id IS new.source_peer_id
      AND proof.source_workspace_id IS new.source_workspace_id
      AND proof.source_created_at IS new.source_created_at
      AND proof.message_content_hash = new.source_content_hash
      AND proof.coverage_version = 'dream-lossless-message-v1'
) BEGIN
    SELECT RAISE(ABORT, 'aggregation source occurrence mismatches coverage');
END;

CREATE TRIGGER IF NOT EXISTS aggregation_source_occurrence_update_guard
BEFORE UPDATE ON aggregation_node_source_occurrences
BEGIN
    SELECT RAISE(ABORT, 'aggregation source occurrences are immutable');
END;

CREATE TRIGGER IF NOT EXISTS aggregation_source_occurrence_delete_unpublishes
AFTER DELETE ON aggregation_node_source_occurrences
BEGIN
    UPDATE aggregation_nodes
    SET source_manifest_version = 'aggregation-source-manifest-v1',
        source_manifest_count = 0,
        source_manifest_hash = NULL,
        source_manifest_complete = 0
    WHERE id = old.node_id AND source_manifest_complete = 1;
END;
