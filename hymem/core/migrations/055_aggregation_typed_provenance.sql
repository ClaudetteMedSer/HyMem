-- v55: complete typed aggregation provenance and structural publication.
--
-- Existing v45 message-only nodes remain physical history.  They cannot be
-- promoted: without typed membership and anchor rows there is no exact way to
-- distinguish an episode from a child node or to prove root facts.  Clearing
-- the old success acknowledgement forces a clean v55 build while preserving
-- every historical node/source row for audit and possible cache reuse.

INSERT OR IGNORE INTO schema_meta(key,value)
VALUES ('aggregation_typed_provenance_schema','55');

ALTER TABLE aggregation_nodes ADD COLUMN input_manifest_version TEXT;
ALTER TABLE aggregation_nodes ADD COLUMN input_manifest_count INTEGER NOT NULL
    DEFAULT 0 CHECK (input_manifest_count >= 0);
ALTER TABLE aggregation_nodes ADD COLUMN input_manifest_hash TEXT;
ALTER TABLE aggregation_nodes ADD COLUMN input_manifest_complete BOOLEAN NOT NULL
    DEFAULT 0 CHECK (input_manifest_complete IN (0, 1));
ALTER TABLE aggregation_nodes ADD COLUMN node_kind TEXT CHECK (
    node_kind IS NULL OR node_kind IN ('cluster','rollup','root')
);
ALTER TABLE aggregation_nodes ADD COLUMN output_hash TEXT;
ALTER TABLE aggregation_nodes ADD COLUMN publication_id TEXT;
ALTER TABLE aggregation_nodes ADD COLUMN build_config_version TEXT;

CREATE TABLE IF NOT EXISTS aggregation_node_inputs (
    node_id TEXT NOT NULL REFERENCES aggregation_nodes(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    input_kind TEXT NOT NULL CHECK (input_kind IN (
        'episode','aggregation_node','user_profile',
        'knowledge_graph','narrative_fact'
    )),
    source_key TEXT NOT NULL CHECK (length(source_key) > 0),
    source_ref_json TEXT NOT NULL CHECK (json_valid(source_ref_json)),
    payload_hash TEXT NOT NULL CHECK (
        length(payload_hash) = 71 AND payload_hash GLOB 'sha256:*'
    ),
    authority_hash TEXT NOT NULL CHECK (
        length(authority_hash) = 71 AND authority_hash GLOB 'sha256:*'
    ),
    source_manifest_count INTEGER NOT NULL CHECK (source_manifest_count > 0),
    source_manifest_hash TEXT NOT NULL CHECK (
        length(source_manifest_hash) = 71
        AND source_manifest_hash GLOB 'sha256:*'
    ),
    PRIMARY KEY (node_id, ordinal),
    UNIQUE (node_id, input_kind, source_key)
);

CREATE TABLE IF NOT EXISTS aggregation_node_input_sources (
    node_id TEXT NOT NULL,
    input_ordinal INTEGER NOT NULL CHECK (input_ordinal >= 0),
    source_ordinal INTEGER NOT NULL CHECK (source_ordinal >= 0),
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
    PRIMARY KEY (node_id, input_ordinal, source_ordinal),
    UNIQUE (
        node_id, input_ordinal, source_session_id, source_message_id
    ),
    FOREIGN KEY (node_id, input_ordinal)
        REFERENCES aggregation_node_inputs(node_id, ordinal) ON DELETE CASCADE,
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_aggregation_input_source_occurrence
    ON aggregation_node_input_sources(source_session_id, source_message_id);

CREATE TABLE IF NOT EXISTS aggregation_publication_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    publication_id TEXT NOT NULL CHECK (
        length(publication_id) = 71 AND publication_id GLOB 'sha256:*'
    ),
    config_version TEXT NOT NULL CHECK (
        length(config_version) = 92
        AND substr(config_version,1,28) = 'aggregation-build-config-v1:'
    ),
    cluster_min_members INTEGER NOT NULL CHECK (cluster_min_members >= 1),
    cluster_min_sessions INTEGER NOT NULL CHECK (cluster_min_sessions >= 1),
    anchor_fact_cap INTEGER NOT NULL CHECK (anchor_fact_cap >= 0),
    root_node_id TEXT,
    node_count INTEGER NOT NULL CHECK (node_count >= 0),
    node_set_hash TEXT NOT NULL CHECK (
        length(node_set_hash) = 71 AND node_set_hash GLOB 'sha256:*'
    ),
    published_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP CHECK (
        typeof(published_at) = 'text'
        AND length(published_at) = 19
        AND strftime('%Y-%m-%d %H:%M:%S', published_at) IS NOT NULL
        AND strftime('%Y-%m-%d %H:%M:%S', published_at) = published_at
    )
);

DROP TRIGGER IF EXISTS aggregation_source_header_insert_guard;
DROP TRIGGER IF EXISTS aggregation_source_header_update_guard;
DROP TRIGGER IF EXISTS aggregation_source_bound_update_guard;
DROP TRIGGER IF EXISTS aggregation_source_occurrence_insert_guard;
DROP TRIGGER IF EXISTS aggregation_source_occurrence_update_guard;
DROP TRIGGER IF EXISTS aggregation_source_occurrence_delete_unpublishes;
DROP TRIGGER IF EXISTS aggregation_input_insert_guard;
DROP TRIGGER IF EXISTS aggregation_input_update_guard;
DROP TRIGGER IF EXISTS aggregation_input_delete_unpublishes;
DROP TRIGGER IF EXISTS aggregation_input_source_insert_guard;
DROP TRIGGER IF EXISTS aggregation_input_source_update_guard;
DROP TRIGGER IF EXISTS aggregation_input_source_delete_unpublishes;
DROP TRIGGER IF EXISTS aggregation_publication_update_guard;

CREATE TRIGGER aggregation_source_header_insert_guard
BEFORE INSERT ON aggregation_nodes
WHEN NOT (
    new.source_manifest_complete = 0
    AND new.source_manifest_count = 0
    AND new.source_manifest_hash IS NULL
    AND (new.source_manifest_version IS NULL OR
         new.source_manifest_version = 'aggregation-source-manifest-v1')
    AND new.input_manifest_complete = 0
    AND new.input_manifest_count = 0
    AND new.input_manifest_hash IS NULL
    AND new.input_manifest_version IS NULL
    AND (new.node_kind IS NULL OR
         new.node_kind IN ('cluster','rollup','root'))
    AND (new.output_hash IS NULL OR (
         length(new.output_hash) = 71
         AND new.output_hash GLOB 'sha256:*'))
    AND (new.input_fingerprint IS NULL OR (
         length(new.input_fingerprint) = 71
         AND new.input_fingerprint GLOB 'sha256:*'))
    AND (new.publication_id IS NULL OR (
         length(new.publication_id) = 71
         AND new.publication_id GLOB 'sha256:*'))
    AND (new.build_config_version IS NULL OR (
         length(new.build_config_version) = 92
         AND substr(new.build_config_version,1,28) =
             'aggregation-build-config-v1:'))
) BEGIN
    SELECT RAISE(ABORT, 'aggregation proof must publish after its children');
END;

CREATE TRIGGER aggregation_source_header_update_guard
BEFORE UPDATE OF source_manifest_version, source_manifest_count,
                 source_manifest_hash, source_manifest_complete,
                 input_manifest_version, input_manifest_count,
                 input_manifest_hash, input_manifest_complete
ON aggregation_nodes
WHEN NOT (
    (new.source_manifest_complete = 0
     AND new.source_manifest_count = 0
     AND new.source_manifest_hash IS NULL
     AND (new.source_manifest_version IS NULL OR
          new.source_manifest_version = 'aggregation-source-manifest-v1')
     AND new.input_manifest_complete = 0
     AND new.input_manifest_count = 0
     AND new.input_manifest_hash IS NULL
     AND new.input_manifest_version IS NULL)
    OR
    (new.source_manifest_complete = 1
     AND new.source_manifest_version = 'aggregation-source-manifest-v1'
     AND new.source_manifest_count > 0
     AND length(new.source_manifest_hash) = 71
     AND new.source_manifest_hash GLOB 'sha256:*'
     AND new.input_manifest_complete = 1
     AND new.input_manifest_version = 'aggregation-input-manifest-v1'
     AND new.input_manifest_count > 0
     AND length(new.input_manifest_hash) = 71
     AND new.input_manifest_hash GLOB 'sha256:*'
     AND length(new.input_fingerprint) = 71
     AND new.input_fingerprint GLOB 'sha256:*'
     AND new.node_kind IN ('cluster','rollup','root')
     AND length(new.output_hash) = 71
     AND new.output_hash GLOB 'sha256:*'
     AND length(new.publication_id) = 71
     AND new.publication_id GLOB 'sha256:*'
     AND length(new.build_config_version) = 92
     AND substr(new.build_config_version,1,28) =
         'aggregation-build-config-v1:'
     AND (SELECT COUNT(*) FROM aggregation_node_source_occurrences source
          WHERE source.node_id = new.id) = new.source_manifest_count
     AND (SELECT MIN(ordinal) FROM aggregation_node_source_occurrences source
          WHERE source.node_id = new.id) = 0
     AND (SELECT MAX(ordinal) FROM aggregation_node_source_occurrences source
          WHERE source.node_id = new.id) = new.source_manifest_count - 1
     AND (SELECT COUNT(*) FROM aggregation_node_inputs input
          WHERE input.node_id = new.id) = new.input_manifest_count
     AND (SELECT MIN(ordinal) FROM aggregation_node_inputs input
          WHERE input.node_id = new.id) = 0
     AND (SELECT MAX(ordinal) FROM aggregation_node_inputs input
          WHERE input.node_id = new.id) = new.input_manifest_count - 1
     AND NOT EXISTS (
          SELECT 1 FROM aggregation_node_inputs input
          WHERE input.node_id = new.id
            AND (SELECT COUNT(*) FROM aggregation_node_input_sources source
                 WHERE source.node_id=input.node_id
                   AND source.input_ordinal=input.ordinal)
                <> input.source_manifest_count
     ))
) BEGIN
    SELECT RAISE(ABORT, 'invalid aggregation proof publication');
END;

CREATE TRIGGER aggregation_source_bound_update_guard
BEFORE UPDATE OF title, summary, member_episode_ids, session_ids,
                 n_members, n_sessions, level, is_root, input_fingerprint,
                 node_kind, output_hash, publication_id, build_config_version
ON aggregation_nodes
WHEN old.source_manifest_complete = 1 OR old.input_manifest_complete = 1
BEGIN
    SELECT RAISE(ABORT, 'unpublish aggregation proof before changing input');
END;

CREATE TRIGGER aggregation_source_occurrence_insert_guard
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
      AND node.input_manifest_complete = 0
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

CREATE TRIGGER aggregation_source_occurrence_update_guard
BEFORE UPDATE ON aggregation_node_source_occurrences
BEGIN
    SELECT RAISE(ABORT, 'aggregation source occurrences are immutable');
END;

CREATE TRIGGER aggregation_source_occurrence_delete_unpublishes
AFTER DELETE ON aggregation_node_source_occurrences
BEGIN
    UPDATE aggregation_nodes
    SET source_manifest_version = 'aggregation-source-manifest-v1',
        source_manifest_count = 0,
        source_manifest_hash = NULL,
        source_manifest_complete = 0,
        input_manifest_version = NULL,
        input_manifest_count = 0,
        input_manifest_hash = NULL,
        input_manifest_complete = 0
    WHERE id = old.node_id
      AND (source_manifest_complete = 1 OR input_manifest_complete = 1);
END;

CREATE TRIGGER aggregation_input_insert_guard
BEFORE INSERT ON aggregation_node_inputs
WHEN NOT EXISTS (
    SELECT 1 FROM aggregation_nodes node
    WHERE node.id=new.node_id
      AND node.source_manifest_complete=0
      AND node.input_manifest_complete=0
) BEGIN
    SELECT RAISE(ABORT, 'aggregation input requires unpublished node');
END;

CREATE TRIGGER aggregation_input_update_guard
BEFORE UPDATE ON aggregation_node_inputs
BEGIN
    SELECT RAISE(ABORT, 'aggregation inputs are immutable');
END;

CREATE TRIGGER aggregation_input_delete_unpublishes
AFTER DELETE ON aggregation_node_inputs
BEGIN
    UPDATE aggregation_nodes
    SET source_manifest_version = 'aggregation-source-manifest-v1',
        source_manifest_count = 0,
        source_manifest_hash = NULL,
        source_manifest_complete = 0,
        input_manifest_version = NULL,
        input_manifest_count = 0,
        input_manifest_hash = NULL,
        input_manifest_complete = 0
    WHERE id = old.node_id
      AND (source_manifest_complete = 1 OR input_manifest_complete = 1);
END;

CREATE TRIGGER aggregation_input_source_insert_guard
BEFORE INSERT ON aggregation_node_input_sources
WHEN NOT EXISTS (
    SELECT 1
    FROM aggregation_node_inputs input
    JOIN aggregation_nodes node ON node.id=input.node_id
    JOIN message_retention_coverage proof
      ON proof.message_id=new.source_message_id
     AND proof.chunk_id=new.source_coverage_chunk_id
     AND proof.coverage_version=new.source_coverage_version
    WHERE input.node_id=new.node_id
      AND input.ordinal=new.input_ordinal
      AND node.source_manifest_complete=0
      AND node.input_manifest_complete=0
      AND proof.source_session_id=new.source_session_id
      AND proof.source_role=new.source_role
      AND proof.source_peer_id IS new.source_peer_id
      AND proof.source_workspace_id IS new.source_workspace_id
      AND proof.source_created_at IS new.source_created_at
      AND proof.message_content_hash=new.source_content_hash
      AND proof.coverage_version='dream-lossless-message-v1'
) BEGIN
    SELECT RAISE(ABORT, 'aggregation input source mismatches coverage');
END;

CREATE TRIGGER aggregation_input_source_update_guard
BEFORE UPDATE ON aggregation_node_input_sources
BEGIN
    SELECT RAISE(ABORT, 'aggregation input sources are immutable');
END;

CREATE TRIGGER aggregation_input_source_delete_unpublishes
AFTER DELETE ON aggregation_node_input_sources
BEGIN
    UPDATE aggregation_nodes
    SET source_manifest_version = 'aggregation-source-manifest-v1',
        source_manifest_count = 0,
        source_manifest_hash = NULL,
        source_manifest_complete = 0,
        input_manifest_version = NULL,
        input_manifest_count = 0,
        input_manifest_hash = NULL,
        input_manifest_complete = 0
    WHERE id = old.node_id
      AND (source_manifest_complete = 1 OR input_manifest_complete = 1);
END;

CREATE TRIGGER aggregation_publication_update_guard
BEFORE UPDATE ON aggregation_publication_state
BEGIN
    SELECT RAISE(ABORT, 'aggregation publication state is immutable');
END;

DELETE FROM aggregation_publication_state;
