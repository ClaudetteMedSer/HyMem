-- v57: exact aggregation material-freshness epoch.
--
-- The LLM generation key (v56) and source-material epoch are independent and
-- compose on every node/publication. Legacy v56 publications are withdrawn:
-- they cannot prove the complete eligible episode/anchor/vector/blocking set.

INSERT OR IGNORE INTO schema_meta(key,value)
VALUES ('aggregation_material_epoch_schema','57');

CREATE TABLE aggregation_material_clock (
    id INTEGER PRIMARY KEY CHECK (id=1),
    revision INTEGER NOT NULL CHECK (
        typeof(revision)='integer'
        AND revision BETWEEN 0 AND 9223372036854775806
    ),
    clock_schema TEXT NOT NULL CHECK (
        clock_schema='hymem-aggregation-material-clock-v1'
    )
);
INSERT INTO aggregation_material_clock(id,revision,clock_schema)
SELECT 1,0,'hymem-aggregation-material-clock-v1'
WHERE NOT EXISTS (SELECT 1 FROM aggregation_material_clock);

CREATE TRIGGER aggregation_material_clock_insert_guard
BEFORE INSERT ON aggregation_material_clock
BEGIN
    SELECT RAISE(ABORT,'aggregation material clock is a guarded singleton');
END;

CREATE TRIGGER aggregation_material_clock_update_guard
BEFORE UPDATE ON aggregation_material_clock
WHEN new.id IS NOT old.id
  OR new.clock_schema IS NOT old.clock_schema
  OR typeof(new.revision)<>'integer'
  OR new.revision IS NOT old.revision+1
BEGIN
    SELECT RAISE(ABORT,'aggregation material clock may only advance once');
END;
CREATE TRIGGER aggregation_material_clock_delete_guard
BEFORE DELETE ON aggregation_material_clock
BEGIN
    SELECT RAISE(ABORT,'aggregation material clock is a guarded singleton');
END;
CREATE TRIGGER aggregation_material_clock_update_unpublishes
AFTER UPDATE ON aggregation_material_clock
BEGIN
    DELETE FROM aggregation_publication_state;
END;

CREATE TABLE aggregation_material_epochs (
    material_epoch_key TEXT PRIMARY KEY CHECK (
        length(material_epoch_key)=100
        AND substr(material_epoch_key,1,36)=
            'hymem-aggregation-material-epoch-v1:'
        AND substr(material_epoch_key,37) NOT GLOB '*[^0-9a-f]*'
    ),
    material_revision INTEGER NOT NULL CHECK (material_revision >= 0),
    config_version TEXT NOT NULL CHECK (
        length(config_version)=92
        AND substr(config_version,1,28)='aggregation-build-config-v1:'
        AND substr(config_version,29) NOT GLOB '*[^0-9a-f]*'
    ),
    snapshot_sha256 TEXT NOT NULL CHECK (
        length(snapshot_sha256)=71 AND snapshot_sha256 GLOB 'sha256:*'
    ),
    embedding_producer_key TEXT NOT NULL CHECK (
        length(embedding_producer_key)=92
        AND substr(embedding_producer_key,1,28)=
            'hymem-embedding-producer-v1:'
        AND substr(embedding_producer_key,29) NOT GLOB '*[^0-9a-f]*'
    ),
    identity_exact BOOLEAN NOT NULL CHECK (identity_exact IN (0,1)),
    reuse_scope TEXT NOT NULL CHECK (
        reuse_scope IN ('durable','process_instance')
        AND ((identity_exact=1 AND reuse_scope='durable')
             OR (identity_exact=0 AND reuse_scope='process_instance'))
    ),
    binding_json TEXT NOT NULL CHECK (json_valid(binding_json)),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE episode_embeddings ADD COLUMN embedding_producer_key TEXT CHECK (
    embedding_producer_key IS NULL OR (
        length(embedding_producer_key)=92
        AND substr(embedding_producer_key,1,28)=
            'hymem-embedding-producer-v1:'
        AND substr(embedding_producer_key,29) NOT GLOB '*[^0-9a-f]*'
    )
);
ALTER TABLE aggregation_node_embeddings
    ADD COLUMN embedding_producer_key TEXT CHECK (
        embedding_producer_key IS NULL OR (
            length(embedding_producer_key)=92
            AND substr(embedding_producer_key,1,28)=
                'hymem-embedding-producer-v1:'
            AND substr(embedding_producer_key,29) NOT GLOB '*[^0-9a-f]*'
        )
    );

-- Hot-path freshness probes must lead with the cited coordinate.  The older
-- provenance indexes lead with session/edge identity and turn each ordinary
-- message/coverage insert into a full historical scan.
CREATE INDEX idx_v57_episode_source_message
ON episode_source_occurrences(
    source_message_id,source_session_id,source_coverage_chunk_id,
    source_coverage_version
);
CREATE INDEX idx_v57_profile_source_message
ON user_profile(source_message_id,source_session_id,invalid_at);
CREATE INDEX idx_v57_evidence_source_message
ON kg_evidence(
    source_message_id,source_session_id,source_coverage_chunk_id,
    source_coverage_version,provenance_status,is_current
);
CREATE INDEX idx_v57_episode_source_peer
ON episode_source_occurrences(
    source_peer_id,source_workspace_id,source_session_id
);
CREATE INDEX idx_v57_coverage_source_peer
ON message_retention_coverage(
    source_peer_id,source_workspace_id,source_session_id,message_id
);
CREATE INDEX idx_v57_evidence_source_peer
ON kg_evidence(
    source_peer_id,source_workspace_id,source_session_id,
    provenance_status,is_current
);
CREATE INDEX idx_v57_episode_source_chunk
ON episode_source_occurrences(source_coverage_chunk_id,source_message_id);
CREATE INDEX idx_v57_coverage_chunk
ON message_retention_coverage(chunk_id,message_id);
CREATE INDEX idx_v57_evidence_source_chunk
ON kg_evidence(source_coverage_chunk_id,is_current,provenance_status);

-- Every durable vector space uses the same secret-free producer key in the
-- historical `model` column.  These guards also fence rolling pre-v57 writers
-- and direct SQL from putting endpoint paths or caller-controlled labels back
-- after the crash-atomic legacy purge.
--
-- Valid-looking vector bytes are not self-authenticating.  Only maintained
-- persistence APIs may create or replace cache/mirror rows; otherwise a direct
-- same-dimension UPDATE could be withdrawn and then laundered into the next
-- publication as a cache hit.  DELETE remains available for conservative
-- repair/eviction and cannot introduce a forged provider output.
CREATE TRIGGER embedding_cache_write_authority_insert
BEFORE INSERT ON embedding_cache
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;
CREATE TRIGGER embedding_cache_write_authority_update
BEFORE UPDATE ON embedding_cache
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;

CREATE TRIGGER message_embedding_write_authority_insert
BEFORE INSERT ON message_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;
CREATE TRIGGER message_embedding_write_authority_update
BEFORE UPDATE ON message_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;

CREATE TRIGGER chunk_embedding_write_authority_insert
BEFORE INSERT ON chunk_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;
CREATE TRIGGER chunk_embedding_write_authority_update
BEFORE UPDATE ON chunk_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;

CREATE TRIGGER edge_embedding_write_authority_insert
BEFORE INSERT ON edge_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;
CREATE TRIGGER edge_embedding_write_authority_update
BEFORE UPDATE ON edge_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;

CREATE TRIGGER episode_embedding_write_authority_insert
BEFORE INSERT ON episode_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;
CREATE TRIGGER episode_embedding_write_authority_update
BEFORE UPDATE ON episode_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;

CREATE TRIGGER narrative_fact_embedding_write_authority_insert
BEFORE INSERT ON narrative_fact_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;
CREATE TRIGGER narrative_fact_embedding_write_authority_update
BEFORE UPDATE ON narrative_fact_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;

CREATE TRIGGER aggregation_node_embedding_write_authority_insert
BEFORE INSERT ON aggregation_node_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;
CREATE TRIGGER aggregation_node_embedding_write_authority_update
BEFORE UPDATE ON aggregation_node_embeddings
WHEN hymem_embedding_mutation_authorized()<>1
BEGIN SELECT RAISE(ABORT,'embedding mirror write is not authorized'); END;

CREATE TRIGGER embedding_cache_model_insert_guard
BEFORE INSERT ON embedding_cache
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
BEGIN
    SELECT RAISE(ABORT,'unsafe embedding producer key');
END;
CREATE TRIGGER embedding_cache_model_update_guard
BEFORE UPDATE OF model ON embedding_cache
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
BEGIN
    SELECT RAISE(ABORT,'unsafe embedding producer key');
END;

CREATE TRIGGER chunk_embedding_model_insert_guard
BEFORE INSERT ON chunk_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'unsafe embedding producer key'); END;
CREATE TRIGGER chunk_embedding_model_update_guard
BEFORE UPDATE OF model ON chunk_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'unsafe embedding producer key'); END;

CREATE TRIGGER message_embedding_model_insert_guard
BEFORE INSERT ON message_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'unsafe embedding producer key'); END;
CREATE TRIGGER message_embedding_model_update_guard
BEFORE UPDATE OF model ON message_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'unsafe embedding producer key'); END;

CREATE TRIGGER edge_embedding_model_insert_guard
BEFORE INSERT ON edge_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'unsafe embedding producer key'); END;
CREATE TRIGGER edge_embedding_model_update_guard
BEFORE UPDATE OF model ON edge_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'unsafe embedding producer key'); END;

CREATE TRIGGER narrative_fact_embedding_model_insert_guard
BEFORE INSERT ON narrative_fact_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'unsafe embedding producer key'); END;
CREATE TRIGGER narrative_fact_embedding_model_update_guard
BEFORE UPDATE OF model ON narrative_fact_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'unsafe embedding producer key'); END;

CREATE TRIGGER episode_embedding_model_insert_guard
BEFORE INSERT ON episode_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
  OR new.embedding_producer_key IS NOT new.model
BEGIN SELECT RAISE(ABORT,'unsafe episode embedding producer key'); END;
CREATE TRIGGER episode_embedding_model_update_guard
BEFORE UPDATE OF model,embedding_producer_key ON episode_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
  OR new.embedding_producer_key IS NOT new.model
BEGIN SELECT RAISE(ABORT,'unsafe episode embedding producer key'); END;

CREATE TRIGGER aggregation_node_embedding_model_insert_guard
BEFORE INSERT ON aggregation_node_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
  OR new.embedding_producer_key IS NOT new.model
BEGIN SELECT RAISE(ABORT,'unsafe aggregation embedding producer key'); END;
CREATE TRIGGER aggregation_node_embedding_model_update_guard
BEFORE UPDATE OF model,embedding_producer_key ON aggregation_node_embeddings
WHEN length(new.model)<>92
  OR substr(new.model,1,28)<>'hymem-embedding-producer-v1:'
  OR substr(new.model,29) GLOB '*[^0-9a-f]*'
  OR new.embedding_producer_key IS NOT new.model
BEGIN SELECT RAISE(ABORT,'unsafe aggregation embedding producer key'); END;

CREATE TRIGGER schema_meta_vec_model_insert_guard
BEFORE INSERT ON schema_meta
WHEN new.key='vec_model' AND (
    length(new.value)<>92
    OR substr(new.value,1,28)<>'hymem-embedding-producer-v1:'
    OR substr(new.value,29) GLOB '*[^0-9a-f]*'
)
BEGIN SELECT RAISE(ABORT,'unsafe vec embedding producer key'); END;
CREATE TRIGGER schema_meta_vec_model_update_guard
BEFORE UPDATE OF key,value ON schema_meta
WHEN (old.key='vec_model' AND new.key IS NOT old.key)
  OR (new.key='vec_model' AND (
      length(new.value)<>92
      OR substr(new.value,1,28)<>'hymem-embedding-producer-v1:'
      OR substr(new.value,29) GLOB '*[^0-9a-f]*'
  ))
BEGIN SELECT RAISE(ABORT,'unsafe vec embedding producer key'); END;

ALTER TABLE aggregation_nodes ADD COLUMN aggregation_material_epoch_key TEXT
    REFERENCES aggregation_material_epochs(material_epoch_key) ON DELETE RESTRICT;
ALTER TABLE aggregation_publication_state
    ADD COLUMN aggregation_material_epoch_key TEXT
    REFERENCES aggregation_material_epochs(material_epoch_key) ON DELETE RESTRICT;
ALTER TABLE aggregation_publication_state
    ADD COLUMN material_revision INTEGER CHECK (
        material_revision IS NULL OR material_revision >= 0
    );
ALTER TABLE aggregation_publication_state
    ADD COLUMN node_embedding_count INTEGER CHECK (
        node_embedding_count IS NULL OR node_embedding_count >= 0
    );
ALTER TABLE aggregation_publication_state
    ADD COLUMN node_embedding_set_hash TEXT CHECK (
        node_embedding_set_hash IS NULL OR (
            length(node_embedding_set_hash)=71
            AND node_embedding_set_hash GLOB 'sha256:*'
        )
    );
ALTER TABLE aggregation_build_health
    ADD COLUMN last_success_material_epoch_key TEXT
    REFERENCES aggregation_material_epochs(material_epoch_key) ON DELETE RESTRICT;
ALTER TABLE aggregation_build_health
    ADD COLUMN pending_material_epoch_key TEXT
    REFERENCES aggregation_material_epochs(material_epoch_key) ON DELETE RESTRICT;
ALTER TABLE aggregation_build_health
    ADD COLUMN last_failure_material_epoch_key TEXT
    REFERENCES aggregation_material_epochs(material_epoch_key) ON DELETE RESTRICT;
ALTER TABLE aggregation_build_health
    ADD COLUMN attempt_serial INTEGER NOT NULL DEFAULT 0
    CHECK (attempt_serial BETWEEN 0 AND 9223372036854775806);
ALTER TABLE aggregation_build_health
    ADD COLUMN pending_attempt_token INTEGER
    CHECK (
        pending_attempt_token IS NULL OR
        pending_attempt_token BETWEEN 1 AND 9223372036854775806
    );

DROP TRIGGER IF EXISTS aggregation_health_attempt_insert_guard;
CREATE TRIGGER aggregation_health_attempt_insert_guard
BEFORE INSERT ON aggregation_build_health
WHEN NOT (
    typeof(new.attempt_serial)='integer'
    AND new.attempt_serial BETWEEN 0 AND 9223372036854775806
    AND (
      (new.pending_config_version IS NULL
       AND new.pending_generation_key IS NULL
       AND new.pending_material_epoch_key IS NULL
       AND new.pending_attempt_token IS NULL)
      OR
      (new.pending_config_version IS NOT NULL
       AND new.pending_generation_key IS NOT NULL
       AND new.pending_attempt_token IS new.attempt_serial
       AND new.pending_attempt_token BETWEEN 1 AND 9223372036854775806)
    )
)
BEGIN
    SELECT RAISE(ABORT,'aggregation health attempt state is malformed');
END;

DROP TRIGGER IF EXISTS aggregation_health_attempt_update_guard;
CREATE TRIGGER aggregation_health_attempt_update_guard
BEFORE UPDATE ON aggregation_build_health
WHEN NOT (
    typeof(new.attempt_serial)='integer'
    AND new.attempt_serial BETWEEN 0 AND 9223372036854775806
    AND (
      (new.pending_config_version IS NULL
       AND new.pending_generation_key IS NULL
       AND new.pending_material_epoch_key IS NULL
       AND new.pending_attempt_token IS NULL)
      OR
      (new.pending_config_version IS NOT NULL
       AND new.pending_generation_key IS NOT NULL
       AND new.pending_attempt_token IS new.attempt_serial
       AND new.pending_attempt_token BETWEEN 1 AND 9223372036854775806)
    )
)
BEGIN
    SELECT RAISE(ABORT,'aggregation health attempt state is malformed');
END;

ALTER TABLE dream_runs ADD COLUMN aggregation_material_epoch_key TEXT
    REFERENCES aggregation_material_epochs(material_epoch_key) ON DELETE RESTRICT;

CREATE VIEW aggregation_live_material AS
SELECT epoch.material_epoch_key,epoch.material_revision
FROM aggregation_publication_state publication
JOIN aggregation_material_epochs epoch
  ON epoch.material_epoch_key=publication.aggregation_material_epoch_key
JOIN aggregation_material_clock clock
  ON clock.id=1 AND clock.revision=epoch.material_revision
UNION
SELECT epoch.material_epoch_key,epoch.material_revision
FROM aggregation_build_health health
JOIN aggregation_material_epochs epoch
  ON epoch.material_epoch_key=health.pending_material_epoch_key
JOIN aggregation_material_clock clock
  ON clock.id=1 AND clock.revision=epoch.material_revision
WHERE health.id=1;

CREATE VIEW aggregation_enabled_root_material AS
SELECT live.material_epoch_key
FROM aggregation_live_material live
JOIN aggregation_material_epochs epoch
  ON epoch.material_epoch_key=live.material_epoch_key
WHERE json_extract(epoch.binding_json,'$.root_anchor_policy')='enabled'
  -- Root/profile/KG inputs are deterministically reselected and exact-proof
  -- hashed at capture, every provider fence, publication commit, and public
  -- read.  They deliberately do not share the episode/vector revision clock:
  -- duplicating cap/order/proof semantics in triggers caused unselected rows
  -- to withdraw otherwise identical publications.  Keep this owned view as
  -- an inert compatibility boundary for the retired trigger predicates.
  AND 0;

CREATE VIEW aggregation_live_embedding_material AS
SELECT epoch.embedding_producer_key AS producer_key,
       json_extract(epoch.binding_json,'$.embedding_dimension') AS dimension
FROM aggregation_live_material live
JOIN aggregation_material_epochs epoch
  ON epoch.material_epoch_key=live.material_epoch_key;

CREATE VIEW aggregation_visible_episode_sources AS
SELECT source.source_message_id,source.source_session_id,source.source_peer_id,
       source.source_workspace_id,source.source_coverage_chunk_id,
       source.source_coverage_version
FROM episode_source_occurrences source
JOIN episodes episode ON episode.id=source.episode_id
JOIN sessions session ON session.id=episode.session_id
WHERE episode.source_manifest_complete=1
  AND (episode.digest_generation IS NULL
       OR episode.digest_generation=session.digest_published_generation)
  AND EXISTS (SELECT 1 FROM aggregation_live_material);

CREATE VIEW aggregation_enabled_profile_sources AS
SELECT profile.source_message_id,profile.source_session_id,
       coverage.source_peer_id,coverage.source_workspace_id,
       coverage.chunk_id AS source_coverage_chunk_id,
       coverage.coverage_version AS source_coverage_version
FROM user_profile profile
LEFT JOIN message_retention_coverage coverage
  ON coverage.message_id=profile.source_message_id
 AND coverage.source_session_id=profile.source_session_id
WHERE profile.invalid_at IS NULL
  AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material);

CREATE VIEW aggregation_enabled_kg_sources AS
SELECT evidence.source_message_id,evidence.source_session_id,
       evidence.source_peer_id,evidence.source_workspace_id,
       evidence.source_coverage_chunk_id,evidence.source_coverage_version
FROM kg_evidence evidence
JOIN knowledge_graph edge ON edge.id=evidence.edge_id
WHERE evidence.provenance_status='canonical' AND evidence.is_current=1
  AND edge.derived=0 AND edge.status='active' AND edge.invalid_at IS NULL
  AND edge.pos_evidence>edge.neg_evidence
  AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material);

DROP TRIGGER IF EXISTS aggregation_material_epochs_insert_guard;
CREATE TRIGGER aggregation_material_epochs_insert_guard
BEFORE INSERT ON aggregation_material_epochs
WHEN hymem_aggregation_material_registry_row_is_valid(
    new.material_epoch_key,new.material_revision,new.config_version,
    new.snapshot_sha256,new.embedding_producer_key,new.identity_exact,
    new.reuse_scope,new.binding_json
) <> 1
BEGIN
    SELECT RAISE(ABORT,'invalid aggregation material epoch registry row');
END;

DROP TRIGGER IF EXISTS aggregation_material_epochs_update_guard;
CREATE TRIGGER aggregation_material_epochs_update_guard
BEFORE UPDATE ON aggregation_material_epochs
BEGIN
    SELECT RAISE(ABORT,'aggregation material epoch registry is immutable');
END;

DROP TRIGGER IF EXISTS aggregation_material_epochs_delete_guard;
CREATE TRIGGER aggregation_material_epochs_delete_guard
BEFORE DELETE ON aggregation_material_epochs
WHEN EXISTS (SELECT 1 FROM aggregation_nodes
             WHERE aggregation_material_epoch_key=old.material_epoch_key)
  OR EXISTS (SELECT 1 FROM aggregation_publication_state
             WHERE aggregation_material_epoch_key=old.material_epoch_key)
  OR EXISTS (SELECT 1 FROM aggregation_build_health
             WHERE last_success_material_epoch_key=old.material_epoch_key
                OR pending_material_epoch_key=old.material_epoch_key
                OR last_failure_material_epoch_key=old.material_epoch_key)
  OR EXISTS (SELECT 1 FROM dream_runs
             WHERE aggregation_material_epoch_key=old.material_epoch_key)
BEGIN
    SELECT RAISE(ABORT,'aggregation material epoch registry is immutable');
END;

DROP TRIGGER IF EXISTS aggregation_material_node_update_guard;
CREATE TRIGGER aggregation_material_node_update_guard
BEFORE UPDATE OF aggregation_material_epoch_key ON aggregation_nodes
WHEN old.source_manifest_complete=1 OR old.input_manifest_complete=1
  OR new.aggregation_material_epoch_key IS NULL
  OR NOT EXISTS (
      SELECT 1 FROM aggregation_material_epochs material
      WHERE material.material_epoch_key=new.aggregation_material_epoch_key
  )
BEGIN
    SELECT RAISE(ABORT,'invalid aggregation node material binding');
END;

DROP TRIGGER IF EXISTS aggregation_material_publication_insert_guard;
CREATE TRIGGER aggregation_material_publication_insert_guard
BEFORE INSERT ON aggregation_publication_state
WHEN new.aggregation_material_epoch_key IS NULL
  OR new.material_revision IS NULL
  OR new.node_embedding_count IS NULL
  OR new.node_embedding_set_hash IS NULL
  OR NOT EXISTS (
      SELECT 1
      FROM aggregation_material_epochs material
      WHERE material.material_epoch_key=new.aggregation_material_epoch_key
        AND material.material_revision=new.material_revision
        AND material.config_version=new.config_version
        AND (
            json_extract(material.binding_json,'$.fresh_until') IS NULL
            OR CURRENT_TIMESTAMP <
               json_extract(material.binding_json,'$.fresh_until')
        )
        AND new.material_revision=(
            SELECT revision FROM aggregation_material_clock WHERE id=1
        )
  )
  OR EXISTS (
      SELECT 1 FROM aggregation_nodes node
      WHERE node.publication_id=new.publication_id
        AND node.aggregation_material_epoch_key
            IS NOT new.aggregation_material_epoch_key
  )
BEGIN
    SELECT RAISE(ABORT,'invalid aggregation publication material binding');
END;

-- Source mutations advance the cheap freshness fence and atomically withdraw
-- visibility. Inserts are limited to rows that can immediately enter the
-- aggregation domain; raw message/coverage traffic alone must not make the
-- standing digest disappear between every user turn and the next dream.
DROP TRIGGER IF EXISTS aggregation_material_sessions_update;
CREATE TRIGGER aggregation_material_sessions_update
AFTER UPDATE OF id,digest_published_generation,source_workspace_id ON sessions
WHEN (old.id IS NOT new.id
  OR old.digest_published_generation IS NOT new.digest_published_generation
  OR old.source_workspace_id IS NOT new.source_workspace_id)
AND EXISTS (SELECT 1 FROM aggregation_live_material)
AND (
    EXISTS (SELECT 1 FROM episodes episode
            WHERE episode.source_manifest_complete=1 AND (
                (episode.session_id=old.id AND
                 (episode.digest_generation IS NULL OR
                  episode.digest_generation=old.digest_published_generation))
             OR (episode.session_id=new.id AND
                 (episode.digest_generation IS NULL OR
                  episode.digest_generation=new.digest_published_generation))
            ))
 OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources source
            WHERE source.source_session_id=old.id
               OR source.source_session_id=new.id)
 OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources source
            WHERE source.source_session_id=old.id
               OR source.source_session_id=new.id)
)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;

DROP TRIGGER IF EXISTS aggregation_material_episodes_insert;
CREATE TRIGGER aggregation_material_episodes_insert AFTER INSERT ON episodes
WHEN new.source_manifest_complete=1 AND EXISTS (
    SELECT 1 FROM sessions s WHERE s.id=new.session_id AND
    (new.digest_generation IS NULL OR
     new.digest_generation=s.digest_published_generation)
)
AND EXISTS (SELECT 1 FROM aggregation_live_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_episodes_update;
CREATE TRIGGER aggregation_material_episodes_update
AFTER UPDATE OF session_id,title,summary,key_entities,
                digest_slice_key,digest_generation,
                source_manifest_version,source_manifest_count,
                source_manifest_hash,source_manifest_complete
ON episodes
WHEN (old.session_id IS NOT new.session_id
   OR old.title IS NOT new.title OR old.summary IS NOT new.summary
   OR old.key_entities IS NOT new.key_entities
   OR old.digest_slice_key IS NOT new.digest_slice_key
   OR old.digest_generation IS NOT new.digest_generation
   OR old.source_manifest_version IS NOT new.source_manifest_version
   OR old.source_manifest_count IS NOT new.source_manifest_count
   OR old.source_manifest_hash IS NOT new.source_manifest_hash
   OR old.source_manifest_complete IS NOT new.source_manifest_complete)
  AND (EXISTS (
      SELECT 1 FROM sessions s WHERE s.id=old.session_id
        AND old.source_manifest_complete=1
        AND (old.digest_generation IS NULL OR
             old.digest_generation=s.digest_published_generation)
  ) OR EXISTS (
      SELECT 1 FROM sessions s WHERE s.id=new.session_id
        AND new.source_manifest_complete=1
        AND (new.digest_generation IS NULL OR
             new.digest_generation=s.digest_published_generation)
  ))
  AND EXISTS (SELECT 1 FROM aggregation_live_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_episodes_delete;
CREATE TRIGGER aggregation_material_episodes_delete AFTER DELETE ON episodes
WHEN old.source_manifest_complete=1 AND EXISTS (
    SELECT 1 FROM sessions s WHERE s.id=old.session_id AND
    (old.digest_generation IS NULL OR
     old.digest_generation=s.digest_published_generation)
)
AND EXISTS (SELECT 1 FROM aggregation_live_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;

DROP TRIGGER IF EXISTS aggregation_material_episode_sources_insert;
CREATE TRIGGER aggregation_material_episode_sources_insert
AFTER INSERT ON episode_source_occurrences
WHEN EXISTS (SELECT 1 FROM episodes e WHERE e.id=new.episode_id
             AND e.source_manifest_complete=1 AND EXISTS (
                 SELECT 1 FROM sessions s WHERE s.id=e.session_id AND
                 (e.digest_generation IS NULL OR
                 e.digest_generation=s.digest_published_generation)))
AND EXISTS (SELECT 1 FROM aggregation_live_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_episode_sources_update;
CREATE TRIGGER aggregation_material_episode_sources_update
AFTER UPDATE ON episode_source_occurrences
WHEN (old.episode_id IS NOT new.episode_id OR old.ordinal IS NOT new.ordinal
   OR old.source_message_id IS NOT new.source_message_id
   OR old.source_session_id IS NOT new.source_session_id
   OR old.source_role IS NOT new.source_role
   OR old.source_peer_id IS NOT new.source_peer_id
   OR old.source_workspace_id IS NOT new.source_workspace_id
   OR old.source_created_at IS NOT new.source_created_at
   OR old.source_coverage_chunk_id IS NOT new.source_coverage_chunk_id
   OR old.source_coverage_version IS NOT new.source_coverage_version
   OR old.source_content_hash IS NOT new.source_content_hash)
 AND (EXISTS (SELECT 1 FROM episodes e WHERE e.id=old.episode_id
             AND e.source_manifest_complete=1 AND EXISTS (
                 SELECT 1 FROM sessions s WHERE s.id=e.session_id AND
                 (e.digest_generation IS NULL OR
                  e.digest_generation=s.digest_published_generation)))
   OR EXISTS (SELECT 1 FROM episodes e WHERE e.id=new.episode_id
             AND e.source_manifest_complete=1 AND EXISTS (
                 SELECT 1 FROM sessions s WHERE s.id=e.session_id AND
                 (e.digest_generation IS NULL OR
                  e.digest_generation=s.digest_published_generation))))
 AND EXISTS (SELECT 1 FROM aggregation_live_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_episode_sources_delete;
CREATE TRIGGER aggregation_material_episode_sources_delete
AFTER DELETE ON episode_source_occurrences
WHEN EXISTS (SELECT 1 FROM episodes e WHERE e.id=old.episode_id
             AND e.source_manifest_complete=1 AND EXISTS (
                 SELECT 1 FROM sessions s WHERE s.id=e.session_id AND
                 (e.digest_generation IS NULL OR
                 e.digest_generation=s.digest_published_generation)))
AND EXISTS (SELECT 1 FROM aggregation_live_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;

DROP TRIGGER IF EXISTS aggregation_material_episode_embeddings_insert;
CREATE TRIGGER aggregation_material_episode_embeddings_insert
AFTER INSERT ON episode_embeddings
WHEN EXISTS (SELECT 1 FROM episodes e JOIN sessions s ON s.id=e.session_id
             WHERE e.id=new.episode_id AND e.source_manifest_complete=1
               AND (e.digest_generation IS NULL OR
                    e.digest_generation=s.digest_published_generation))
AND EXISTS (SELECT 1 FROM aggregation_live_embedding_material material
           WHERE material.producer_key=new.model
             AND material.dimension=new.dim)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_episode_embeddings_update;
CREATE TRIGGER aggregation_material_episode_embeddings_update
AFTER UPDATE OF episode_id,vector_json,model,dim,text_hash,embedding_producer_key
ON episode_embeddings
WHEN (old.episode_id IS NOT new.episode_id
   OR old.vector_json IS NOT new.vector_json OR old.model IS NOT new.model
   OR old.dim IS NOT new.dim OR old.text_hash IS NOT new.text_hash
   OR old.embedding_producer_key IS NOT new.embedding_producer_key)
 AND (EXISTS (SELECT 1 FROM episodes e JOIN sessions s ON s.id=e.session_id
             WHERE e.id=old.episode_id AND e.source_manifest_complete=1
               AND (e.digest_generation IS NULL OR
                    e.digest_generation=s.digest_published_generation))
   OR EXISTS (SELECT 1 FROM episodes e JOIN sessions s ON s.id=e.session_id
             WHERE e.id=new.episode_id AND e.source_manifest_complete=1
               AND (e.digest_generation IS NULL OR
                    e.digest_generation=s.digest_published_generation)))
AND EXISTS (SELECT 1 FROM aggregation_live_embedding_material material
           WHERE (material.producer_key=old.model AND material.dimension=old.dim)
              OR (material.producer_key=new.model AND material.dimension=new.dim))
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_episode_embeddings_delete;
CREATE TRIGGER aggregation_material_episode_embeddings_delete
AFTER DELETE ON episode_embeddings
WHEN EXISTS (SELECT 1 FROM episodes e JOIN sessions s ON s.id=e.session_id
             WHERE e.id=old.episode_id AND e.source_manifest_complete=1
               AND (e.digest_generation IS NULL OR
                    e.digest_generation=s.digest_published_generation))
AND EXISTS (SELECT 1 FROM aggregation_live_embedding_material material
           WHERE material.producer_key=old.model
             AND material.dimension=old.dim)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;

DROP TRIGGER IF EXISTS aggregation_material_profile_insert;
CREATE TRIGGER aggregation_material_profile_insert AFTER INSERT ON user_profile
WHEN new.invalid_at IS NULL AND EXISTS (
    SELECT 1 FROM message_retention_coverage c
    WHERE c.message_id=new.source_message_id
      AND c.source_session_id=new.source_session_id
      AND c.coverage_version='dream-lossless-message-v1'
)
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_profile_update;
CREATE TRIGGER aggregation_material_profile_update
AFTER UPDATE OF slot,slot_key,value,confidence,valid_at,invalid_at,
                source_message_id,source_session_id,source_created_at
ON user_profile
WHEN (old.slot IS NOT new.slot OR old.slot_key IS NOT new.slot_key
   OR old.value IS NOT new.value OR old.confidence IS NOT new.confidence
   OR old.valid_at IS NOT new.valid_at OR old.invalid_at IS NOT new.invalid_at
   OR old.source_message_id IS NOT new.source_message_id
   OR old.source_session_id IS NOT new.source_session_id
   OR old.source_created_at IS NOT new.source_created_at)
 AND (EXISTS (
      SELECT 1 FROM message_retention_coverage c
      WHERE old.invalid_at IS NULL AND c.message_id=old.source_message_id
        AND c.source_session_id=old.source_session_id
        AND c.coverage_version='dream-lossless-message-v1'
 ) OR EXISTS (
      SELECT 1 FROM message_retention_coverage c
      WHERE new.invalid_at IS NULL AND c.message_id=new.source_message_id
        AND c.source_session_id=new.source_session_id
        AND c.coverage_version='dream-lossless-message-v1'
 ))
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_profile_delete;
CREATE TRIGGER aggregation_material_profile_delete AFTER DELETE ON user_profile
WHEN old.invalid_at IS NULL AND EXISTS (
    SELECT 1 FROM message_retention_coverage c
    WHERE c.message_id=old.source_message_id
      AND c.source_session_id=old.source_session_id
      AND c.coverage_version='dream-lossless-message-v1'
)
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;

DROP TRIGGER IF EXISTS aggregation_material_graph_insert;
CREATE TRIGGER aggregation_material_graph_insert AFTER INSERT ON knowledge_graph
WHEN new.derived=0 AND new.status='active' AND new.invalid_at IS NULL
 AND new.pos_evidence>new.neg_evidence
 AND EXISTS (SELECT 1 FROM kg_evidence evidence
             WHERE evidence.edge_id=new.id
               AND evidence.provenance_status='canonical'
               AND evidence.is_current=1 AND evidence.polarity=1)
 AND NOT EXISTS (SELECT 1 FROM kg_evidence_signals signal
                 WHERE signal.edge_id=new.id
                   AND signal.counts_toward_confidence=1)
 AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_graph_update;
CREATE TRIGGER aggregation_material_graph_update
AFTER UPDATE OF subject_canonical,predicate,object_canonical,pos_evidence,
                neg_evidence,first_seen,last_seen,last_reinforced,valid_at,invalid_at,
                status,derived
ON knowledge_graph
WHEN (old.subject_canonical IS NOT new.subject_canonical
   OR old.predicate IS NOT new.predicate
   OR old.object_canonical IS NOT new.object_canonical
   OR old.pos_evidence IS NOT new.pos_evidence
   OR old.neg_evidence IS NOT new.neg_evidence
   OR old.first_seen IS NOT new.first_seen
   OR old.last_seen IS NOT new.last_seen
   OR old.last_reinforced IS NOT new.last_reinforced
   OR old.valid_at IS NOT new.valid_at OR old.invalid_at IS NOT new.invalid_at
   OR old.status IS NOT new.status OR old.derived IS NOT new.derived)
 AND ((old.derived=0 AND old.status='active' AND old.invalid_at IS NULL
       AND old.pos_evidence>old.neg_evidence
       AND EXISTS (SELECT 1 FROM kg_evidence evidence
                   WHERE evidence.edge_id=old.id
                     AND evidence.provenance_status='canonical'
                     AND evidence.is_current=1 AND evidence.polarity=1)
       AND NOT EXISTS (SELECT 1 FROM kg_evidence_signals signal
                       WHERE signal.edge_id=old.id
                         AND signal.counts_toward_confidence=1))
   OR (new.derived=0 AND new.status='active' AND new.invalid_at IS NULL
       AND new.pos_evidence>new.neg_evidence
       AND EXISTS (SELECT 1 FROM kg_evidence evidence
                   WHERE evidence.edge_id=new.id
                     AND evidence.provenance_status='canonical'
                     AND evidence.is_current=1 AND evidence.polarity=1)
       AND NOT EXISTS (SELECT 1 FROM kg_evidence_signals signal
                       WHERE signal.edge_id=new.id
                         AND signal.counts_toward_confidence=1)))
 AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_graph_delete;
CREATE TRIGGER aggregation_material_graph_delete BEFORE DELETE ON knowledge_graph
WHEN old.derived=0 AND old.status='active' AND old.invalid_at IS NULL
 AND old.pos_evidence>old.neg_evidence
 AND EXISTS (SELECT 1 FROM kg_evidence evidence
             WHERE evidence.edge_id=old.id
               AND evidence.provenance_status='canonical'
               AND evidence.is_current=1 AND evidence.polarity=1)
 AND NOT EXISTS (SELECT 1 FROM kg_evidence_signals signal
                 WHERE signal.edge_id=old.id
                   AND signal.counts_toward_confidence=1)
 AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;

DROP TRIGGER IF EXISTS aggregation_material_evidence_insert;
CREATE TRIGGER aggregation_material_evidence_insert AFTER INSERT ON kg_evidence
WHEN new.provenance_status='canonical' AND new.is_current=1 AND EXISTS (
    SELECT 1 FROM knowledge_graph kg WHERE kg.id=new.edge_id
      AND kg.derived=0 AND kg.status='active' AND kg.invalid_at IS NULL
      AND kg.pos_evidence>kg.neg_evidence
)
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_evidence_update;
CREATE TRIGGER aggregation_material_evidence_update AFTER UPDATE ON kg_evidence
WHEN (old.edge_id IS NOT new.edge_id OR old.chunk_id IS NOT new.chunk_id
 OR old.polarity IS NOT new.polarity
 OR old.surface_subject IS NOT new.surface_subject
 OR old.surface_object IS NOT new.surface_object
 OR old.value_text IS NOT new.value_text
 OR old.value_numeric IS NOT new.value_numeric
 OR old.value_unit IS NOT new.value_unit
 OR old.temporal_scope IS NOT new.temporal_scope
 OR old.source_role IS NOT new.source_role
 OR old.evidence_kind IS NOT new.evidence_kind
 OR old.evidence_weight IS NOT new.evidence_weight
 OR old.weight_source IS NOT new.weight_source
 OR old.extraction_prompt_version IS NOT new.extraction_prompt_version
 OR old.extracted_at IS NOT new.extracted_at
 OR old.source_message_id IS NOT new.source_message_id
 OR old.source_session_id IS NOT new.source_session_id
 OR old.source_created_at IS NOT new.source_created_at
 OR old.source_event_at IS NOT new.source_event_at
 OR old.source_coverage_chunk_id IS NOT new.source_coverage_chunk_id
 OR old.source_coverage_version IS NOT new.source_coverage_version
 OR old.provenance_status IS NOT new.provenance_status
 OR old.interpretation_key IS NOT new.interpretation_key
 OR old.revision IS NOT new.revision OR old.is_current IS NOT new.is_current
 OR old.superseded_at IS NOT new.superseded_at
 OR old.superseded_reason IS NOT new.superseded_reason
 OR old.published_at IS NOT new.published_at
 OR old.source_peer_id IS NOT new.source_peer_id
 OR old.source_workspace_id IS NOT new.source_workspace_id)
AND ((old.provenance_status='canonical' AND old.is_current=1)
  OR (new.provenance_status='canonical' AND new.is_current=1))
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_evidence_delete;
CREATE TRIGGER aggregation_material_evidence_delete AFTER DELETE ON kg_evidence
WHEN old.provenance_status='canonical' AND old.is_current=1
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;

DROP TRIGGER IF EXISTS aggregation_material_evidence_signal_insert;
CREATE TRIGGER aggregation_material_evidence_signal_insert
AFTER INSERT ON kg_evidence_signals
WHEN new.counts_toward_confidence=1
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_evidence_signal_update;
CREATE TRIGGER aggregation_material_evidence_signal_update
AFTER UPDATE ON kg_evidence_signals
WHEN (old.edge_id IS NOT new.edge_id OR old.signal_key IS NOT new.signal_key
   OR old.signal_kind IS NOT new.signal_kind OR old.polarity IS NOT new.polarity
   OR old.evidence_weight IS NOT new.evidence_weight
   OR old.counts_toward_confidence IS NOT new.counts_toward_confidence)
 AND (old.counts_toward_confidence=1 OR new.counts_toward_confidence=1)
 AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_evidence_signal_delete;
CREATE TRIGGER aggregation_material_evidence_signal_delete
AFTER DELETE ON kg_evidence_signals
WHEN old.counts_toward_confidence=1
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;

-- These ledgers can make an already-present graph edge acquire or lose exact
-- current authority without touching the edge/evidence row itself.
DROP TRIGGER IF EXISTS aggregation_material_lifecycle_insert;
CREATE TRIGGER aggregation_material_lifecycle_insert
AFTER INSERT ON kg_edge_lifecycle
WHEN new.event_kind='claim_assertion' AND new.direction=1
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_lifecycle_update;
CREATE TRIGGER aggregation_material_lifecycle_update
AFTER UPDATE ON kg_edge_lifecycle
WHEN (old.edge_id IS NOT new.edge_id OR old.event_kind IS NOT new.event_kind
 OR old.direction IS NOT new.direction OR old.event_at IS NOT new.event_at
 OR old.source_evidence_id IS NOT new.source_evidence_id
 OR old.created_at IS NOT new.created_at)
AND ((old.event_kind='claim_assertion' AND old.direction=1)
  OR (new.event_kind='claim_assertion' AND new.direction=1))
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_lifecycle_delete;
CREATE TRIGGER aggregation_material_lifecycle_delete
AFTER DELETE ON kg_edge_lifecycle
WHEN old.event_kind='claim_assertion' AND old.direction=1
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;

DROP TRIGGER IF EXISTS aggregation_material_claim_observation_insert;
CREATE TRIGGER aggregation_material_claim_observation_insert
AFTER INSERT ON kg_claim_observations
WHEN EXISTS (SELECT 1 FROM kg_evidence e WHERE e.id=new.evidence_id
             AND e.provenance_status='canonical' AND e.is_current=1)
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_claim_observation_update;
CREATE TRIGGER aggregation_material_claim_observation_update
AFTER UPDATE ON kg_claim_observations
WHEN (old.chunk_id IS NOT new.chunk_id OR old.edge_id IS NOT new.edge_id
 OR old.source_session_id IS NOT new.source_session_id
 OR old.source_message_id IS NOT new.source_message_id
 OR old.evidence_kind IS NOT new.evidence_kind
 OR old.polarity IS NOT new.polarity
 OR old.prompt_version IS NOT new.prompt_version
 OR old.prompt_generation IS NOT new.prompt_generation
 OR old.evidence_id IS NOT new.evidence_id
 OR old.interpretation_key IS NOT new.interpretation_key
 OR old.observed_at IS NOT new.observed_at
 OR old.phase1_generation_key IS NOT new.phase1_generation_key)
AND (EXISTS (SELECT 1 FROM kg_evidence e WHERE e.id=old.evidence_id
             AND e.provenance_status='canonical' AND e.is_current=1)
  OR EXISTS (SELECT 1 FROM kg_evidence e WHERE e.id=new.evidence_id
             AND e.provenance_status='canonical' AND e.is_current=1))
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_claim_observation_delete;
CREATE TRIGGER aggregation_material_claim_observation_delete
AFTER DELETE ON kg_claim_observations
WHEN EXISTS (SELECT 1 FROM kg_evidence e WHERE e.id=old.evidence_id
             AND e.provenance_status='canonical' AND e.is_current=1)
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;

DROP TRIGGER IF EXISTS aggregation_material_claim_outcome_insert;
CREATE TRIGGER aggregation_material_claim_outcome_insert
AFTER INSERT ON kg_claim_extraction_outcomes
WHEN EXISTS (SELECT 1 FROM kg_claim_observations o
             JOIN kg_evidence e ON e.id=o.evidence_id
             WHERE o.chunk_id=new.chunk_id
               AND o.prompt_version=new.prompt_version
               AND o.prompt_generation=new.prompt_generation
               AND e.provenance_status='canonical' AND e.is_current=1)
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_claim_outcome_update;
CREATE TRIGGER aggregation_material_claim_outcome_update
AFTER UPDATE ON kg_claim_extraction_outcomes
WHEN (old.chunk_id IS NOT new.chunk_id
 OR old.prompt_version IS NOT new.prompt_version
 OR old.prompt_generation IS NOT new.prompt_generation
 OR old.result_hash IS NOT new.result_hash
 OR old.succeeded_at IS NOT new.succeeded_at
 OR old.phase1_generation_key IS NOT new.phase1_generation_key)
AND (EXISTS (SELECT 1 FROM kg_claim_observations o
             JOIN kg_evidence e ON e.id=o.evidence_id
             WHERE o.chunk_id=old.chunk_id
               AND o.prompt_version=old.prompt_version
               AND o.prompt_generation=old.prompt_generation
               AND e.provenance_status='canonical' AND e.is_current=1)
  OR EXISTS (SELECT 1 FROM kg_claim_observations o
             JOIN kg_evidence e ON e.id=o.evidence_id
             WHERE o.chunk_id=new.chunk_id
               AND o.prompt_version=new.prompt_version
               AND o.prompt_generation=new.prompt_generation
               AND e.provenance_status='canonical' AND e.is_current=1))
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_claim_outcome_delete;
CREATE TRIGGER aggregation_material_claim_outcome_delete
AFTER DELETE ON kg_claim_extraction_outcomes
WHEN EXISTS (SELECT 1 FROM kg_claim_observations o
             JOIN kg_evidence e ON e.id=o.evidence_id
             WHERE o.chunk_id=old.chunk_id
               AND o.prompt_version=old.prompt_version
               AND o.prompt_generation=old.prompt_generation
               AND e.provenance_status='canonical' AND e.is_current=1)
AND EXISTS (SELECT 1 FROM aggregation_enabled_root_material)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;

-- Message/peer registry rows are inert unless an already materialized typed
-- episode/profile/KG source cites them.  This keeps ordinary new chat traffic
-- from withdrawing the digest, while authority edits/retractions do.
DROP TRIGGER IF EXISTS aggregation_material_message_insert;
CREATE TRIGGER aggregation_material_message_insert AFTER INSERT ON messages
WHEN EXISTS (
    SELECT 1 FROM message_retention_coverage coverage
    JOIN chunks source_chunk ON source_chunk.id=coverage.chunk_id
    WHERE coverage.message_id=new.id
      AND (
        EXISTS (SELECT 1 FROM aggregation_visible_episode_sources source
                WHERE source.source_message_id=coverage.message_id
                  AND source.source_session_id=coverage.source_session_id
                  AND source.source_coverage_chunk_id=coverage.chunk_id
                  AND source.source_coverage_version=coverage.coverage_version)
        OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources profile
                   WHERE profile.source_message_id=coverage.message_id
                     AND profile.source_session_id=coverage.source_session_id)
        OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources evidence
                   WHERE evidence.source_message_id=coverage.message_id
                     AND evidence.source_session_id=coverage.source_session_id
                     AND evidence.source_coverage_chunk_id=coverage.chunk_id
                     AND evidence.source_coverage_version=coverage.coverage_version)
      )
      AND NOT (
        new.session_id=coverage.source_session_id
        AND new.role=coverage.source_role
        AND new.source_peer_id IS coverage.source_peer_id
        AND new.source_workspace_id IS coverage.source_workspace_id
        AND new.created_at IS coverage.source_created_at
        AND hymem_message_record_matches_raw_source(
              source_chunk.text,new.id,new.session_id,new.role,new.content,
              new.created_at,new.source_peer_id,new.source_workspace_id,
              coverage.message_content_hash,coverage.hash_version,
              coverage.record_version
            )=1
      )
)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_message_update;
CREATE TRIGGER aggregation_material_message_update
AFTER UPDATE OF id,session_id,role,source_peer_id,source_workspace_id,content,created_at
ON messages
WHEN (old.id IS NOT new.id
 OR old.session_id IS NOT new.session_id OR old.role IS NOT new.role
 OR old.source_peer_id IS NOT new.source_peer_id
 OR old.source_workspace_id IS NOT new.source_workspace_id
 OR old.content IS NOT new.content OR old.created_at IS NOT new.created_at)
AND (EXISTS (SELECT 1 FROM aggregation_visible_episode_sources s
             WHERE s.source_message_id=old.id OR s.source_message_id=new.id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources p
             WHERE p.source_message_id=old.id OR p.source_message_id=new.id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources e
             WHERE e.source_message_id=old.id OR e.source_message_id=new.id))
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_message_delete;
CREATE TRIGGER aggregation_material_message_delete AFTER DELETE ON messages
WHEN EXISTS (
    SELECT 1 FROM message_retention_coverage coverage
    JOIN chunks source_chunk ON source_chunk.id=coverage.chunk_id
    WHERE coverage.message_id=old.id
      AND (
        EXISTS (SELECT 1 FROM aggregation_visible_episode_sources source
                WHERE source.source_message_id=coverage.message_id
                  AND source.source_session_id=coverage.source_session_id
                  AND source.source_coverage_chunk_id=coverage.chunk_id
                  AND source.source_coverage_version=coverage.coverage_version)
        OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources profile
                   WHERE profile.source_message_id=coverage.message_id
                     AND profile.source_session_id=coverage.source_session_id)
        OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources evidence
                   WHERE evidence.source_message_id=coverage.message_id
                     AND evidence.source_session_id=coverage.source_session_id
                     AND evidence.source_coverage_chunk_id=coverage.chunk_id
                     AND evidence.source_coverage_version=coverage.coverage_version)
      )
      AND NOT (
        old.session_id=coverage.source_session_id
        AND old.role=coverage.source_role
        AND old.source_peer_id IS coverage.source_peer_id
        AND old.source_workspace_id IS coverage.source_workspace_id
        AND old.created_at IS coverage.source_created_at
        AND hymem_message_record_matches_raw_source(
              source_chunk.text,old.id,old.session_id,old.role,old.content,
              old.created_at,old.source_peer_id,old.source_workspace_id,
              coverage.message_content_hash,coverage.hash_version,
              coverage.record_version
            )=1
      )
)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;

DROP TRIGGER IF EXISTS aggregation_material_peer_insert;
CREATE TRIGGER aggregation_material_peer_insert AFTER INSERT ON peers
WHEN EXISTS (
    SELECT 1 FROM aggregation_visible_episode_sources s
    WHERE s.source_peer_id=new.id AND s.source_workspace_id=new.workspace_id
) OR EXISTS (
    SELECT 1 FROM aggregation_enabled_profile_sources c
    WHERE c.source_peer_id=new.id AND c.source_workspace_id=new.workspace_id
) OR EXISTS (
    SELECT 1 FROM aggregation_enabled_kg_sources e
    WHERE e.source_peer_id=new.id AND e.source_workspace_id=new.workspace_id
)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_peer_update;
CREATE TRIGGER aggregation_material_peer_update
AFTER UPDATE OF id,workspace_id,role ON peers
WHEN (old.id IS NOT new.id OR old.workspace_id IS NOT new.workspace_id
 OR old.role IS NOT new.role)
AND (EXISTS (
    SELECT 1 FROM aggregation_visible_episode_sources s
    WHERE s.source_peer_id=old.id AND s.source_workspace_id=old.workspace_id
) OR EXISTS (
    SELECT 1 FROM aggregation_enabled_profile_sources c
    WHERE c.source_peer_id=old.id AND c.source_workspace_id=old.workspace_id
) OR EXISTS (
    SELECT 1 FROM aggregation_enabled_kg_sources e
    WHERE e.source_peer_id=old.id AND e.source_workspace_id=old.workspace_id
 ) OR EXISTS (
    SELECT 1 FROM aggregation_visible_episode_sources s
    WHERE s.source_peer_id=new.id AND s.source_workspace_id=new.workspace_id
 ) OR EXISTS (
    SELECT 1 FROM aggregation_enabled_profile_sources c
    WHERE c.source_peer_id=new.id AND c.source_workspace_id=new.workspace_id
 ) OR EXISTS (
    SELECT 1 FROM aggregation_enabled_kg_sources e
    WHERE e.source_peer_id=new.id AND e.source_workspace_id=new.workspace_id
))
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_peer_delete;
CREATE TRIGGER aggregation_material_peer_delete AFTER DELETE ON peers
WHEN EXISTS (
    SELECT 1 FROM aggregation_visible_episode_sources s
    WHERE s.source_peer_id=old.id AND s.source_workspace_id=old.workspace_id
) OR EXISTS (
    SELECT 1 FROM aggregation_enabled_profile_sources c
    WHERE c.source_peer_id=old.id AND c.source_workspace_id=old.workspace_id
) OR EXISTS (
    SELECT 1 FROM aggregation_enabled_kg_sources e
    WHERE e.source_peer_id=old.id AND e.source_workspace_id=old.workspace_id
)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;

DROP TRIGGER IF EXISTS aggregation_material_session_peer_insert;
CREATE TRIGGER aggregation_material_session_peer_insert AFTER INSERT ON session_peers
WHEN EXISTS (SELECT 1 FROM aggregation_visible_episode_sources s
             WHERE s.source_session_id=new.session_id
               AND s.source_peer_id=new.peer_id
               AND s.source_workspace_id=new.workspace_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources s
             WHERE s.source_session_id=new.session_id
               AND s.source_peer_id=new.peer_id
               AND s.source_workspace_id=new.workspace_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources e
             WHERE e.source_session_id=new.session_id
               AND e.source_peer_id=new.peer_id
               AND e.source_workspace_id=new.workspace_id)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_session_peer_update;
CREATE TRIGGER aggregation_material_session_peer_update AFTER UPDATE ON session_peers
WHEN (old.session_id IS NOT new.session_id
 OR old.workspace_id IS NOT new.workspace_id
 OR old.peer_id IS NOT new.peer_id)
AND (EXISTS (SELECT 1 FROM aggregation_visible_episode_sources s
             WHERE s.source_session_id=old.session_id
               AND s.source_peer_id=old.peer_id
               AND s.source_workspace_id=old.workspace_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources s
             WHERE s.source_session_id=old.session_id
               AND s.source_peer_id=old.peer_id
               AND s.source_workspace_id=old.workspace_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources e
             WHERE e.source_session_id=old.session_id
               AND e.source_peer_id=old.peer_id
               AND e.source_workspace_id=old.workspace_id)
  OR EXISTS (SELECT 1 FROM aggregation_visible_episode_sources s
             WHERE s.source_session_id=new.session_id
               AND s.source_peer_id=new.peer_id
               AND s.source_workspace_id=new.workspace_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources s
             WHERE s.source_session_id=new.session_id
               AND s.source_peer_id=new.peer_id
               AND s.source_workspace_id=new.workspace_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources e
             WHERE e.source_session_id=new.session_id
               AND e.source_peer_id=new.peer_id
               AND e.source_workspace_id=new.workspace_id))
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;
DROP TRIGGER IF EXISTS aggregation_material_session_peer_delete;
CREATE TRIGGER aggregation_material_session_peer_delete AFTER DELETE ON session_peers
WHEN EXISTS (SELECT 1 FROM aggregation_visible_episode_sources s
             WHERE s.source_session_id=old.session_id
               AND s.source_peer_id=old.peer_id
               AND s.source_workspace_id=old.workspace_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources s
             WHERE s.source_session_id=old.session_id
               AND s.source_peer_id=old.peer_id
               AND s.source_workspace_id=old.workspace_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources e
             WHERE e.source_session_id=old.session_id
               AND e.source_peer_id=old.peer_id
               AND e.source_workspace_id=old.workspace_id)
BEGIN UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1; END;

-- Coverage/chunk traffic invalidates only when an already materialized anchor
-- or episode cites it. Ordinary newly arrived lossless rows remain inert here.
DROP TRIGGER IF EXISTS aggregation_material_coverage_insert;
CREATE TRIGGER aggregation_material_coverage_insert
AFTER INSERT ON message_retention_coverage
WHEN EXISTS (SELECT 1 FROM aggregation_visible_episode_sources source
             WHERE source.source_message_id=new.message_id
               AND source.source_session_id=new.source_session_id
               AND source.source_coverage_chunk_id=new.chunk_id
               AND source.source_coverage_version=new.coverage_version)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources profile
             WHERE profile.source_message_id=new.message_id
               AND profile.source_session_id=new.source_session_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources evidence
             WHERE evidence.source_message_id=new.message_id
               AND evidence.source_session_id=new.source_session_id
               AND evidence.source_coverage_chunk_id=new.chunk_id
               AND evidence.source_coverage_version=new.coverage_version)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
END;
DROP TRIGGER IF EXISTS aggregation_material_coverage_update;
CREATE TRIGGER aggregation_material_coverage_update
AFTER UPDATE ON message_retention_coverage
WHEN (old.message_id IS NOT new.message_id
 OR old.source_session_id IS NOT new.source_session_id
 OR old.source_role IS NOT new.source_role
 OR old.source_peer_id IS NOT new.source_peer_id
 OR old.source_workspace_id IS NOT new.source_workspace_id
 OR old.source_created_at IS NOT new.source_created_at
 OR old.chunk_id IS NOT new.chunk_id
 OR old.message_content_hash IS NOT new.message_content_hash
 OR old.hash_version IS NOT new.hash_version
 OR old.record_version IS NOT new.record_version
 OR old.coverage_version IS NOT new.coverage_version)
AND (EXISTS (SELECT 1 FROM aggregation_visible_episode_sources source
             WHERE source.source_message_id=old.message_id
               AND source.source_session_id=old.source_session_id
               AND source.source_coverage_chunk_id=old.chunk_id
               AND source.source_coverage_version=old.coverage_version)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources profile
             WHERE profile.source_message_id=old.message_id
               AND profile.source_session_id=old.source_session_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources evidence
             WHERE evidence.source_message_id=old.message_id
               AND evidence.source_session_id=old.source_session_id
               AND evidence.source_coverage_chunk_id=old.chunk_id
               AND evidence.source_coverage_version=old.coverage_version)
  OR EXISTS (SELECT 1 FROM aggregation_visible_episode_sources source
             WHERE source.source_message_id=new.message_id
               AND source.source_session_id=new.source_session_id
               AND source.source_coverage_chunk_id=new.chunk_id
               AND source.source_coverage_version=new.coverage_version)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources profile
             WHERE profile.source_message_id=new.message_id
               AND profile.source_session_id=new.source_session_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources evidence
             WHERE evidence.source_message_id=new.message_id
               AND evidence.source_session_id=new.source_session_id
               AND evidence.source_coverage_chunk_id=new.chunk_id
               AND evidence.source_coverage_version=new.coverage_version))
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_coverage_delete;
CREATE TRIGGER aggregation_material_coverage_delete
AFTER DELETE ON message_retention_coverage
WHEN EXISTS (SELECT 1 FROM aggregation_visible_episode_sources source
             WHERE source.source_message_id=old.message_id
               AND source.source_session_id=old.source_session_id
               AND source.source_coverage_chunk_id=old.chunk_id
               AND source.source_coverage_version=old.coverage_version)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources profile
             WHERE profile.source_message_id=old.message_id
               AND profile.source_session_id=old.source_session_id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources evidence
             WHERE evidence.source_message_id=old.message_id
               AND evidence.source_session_id=old.source_session_id
               AND evidence.source_coverage_chunk_id=old.chunk_id
               AND evidence.source_coverage_version=old.coverage_version)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;

DROP TRIGGER IF EXISTS aggregation_material_chunk_update;
CREATE TRIGGER aggregation_material_chunk_update
AFTER UPDATE OF id,session_id,start_message_id,end_message_id,text,chunk_kind,
                source_manifest_version,source_manifest_count
ON chunks
WHEN (old.id IS NOT new.id OR old.session_id IS NOT new.session_id
 OR old.start_message_id IS NOT new.start_message_id
 OR old.end_message_id IS NOT new.end_message_id
 OR old.text IS NOT new.text OR old.chunk_kind IS NOT new.chunk_kind
 OR old.source_manifest_version IS NOT new.source_manifest_version
 OR old.source_manifest_count IS NOT new.source_manifest_count)
AND (EXISTS (SELECT 1 FROM aggregation_visible_episode_sources source
             WHERE source.source_coverage_chunk_id=old.id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources profile
             WHERE profile.source_coverage_chunk_id=old.id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources evidence
             WHERE evidence.source_coverage_chunk_id=old.id)
  OR EXISTS (SELECT 1 FROM aggregation_visible_episode_sources source
             WHERE source.source_coverage_chunk_id=new.id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources profile
             WHERE profile.source_coverage_chunk_id=new.id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources evidence
             WHERE evidence.source_coverage_chunk_id=new.id))
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_material_chunk_delete;
CREATE TRIGGER aggregation_material_chunk_delete AFTER DELETE ON chunks
WHEN EXISTS (SELECT 1 FROM aggregation_visible_episode_sources source
             WHERE source.source_coverage_chunk_id=old.id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_profile_sources profile
             WHERE profile.source_coverage_chunk_id=old.id)
  OR EXISTS (SELECT 1 FROM aggregation_enabled_kg_sources evidence
             WHERE evidence.source_coverage_chunk_id=old.id)
BEGIN
    UPDATE aggregation_material_clock SET revision=revision+1 WHERE id=1;
    DELETE FROM aggregation_publication_state;
END;

-- Node vector mutations are output/retrieval corruption, not source-material
-- revision changes. They still revoke publication immediately.
DROP TRIGGER IF EXISTS aggregation_node_embedding_unpublishes_insert;
CREATE TRIGGER aggregation_node_embedding_unpublishes_insert
AFTER INSERT ON aggregation_node_embeddings
WHEN EXISTS (
    SELECT 1 FROM aggregation_nodes node
    JOIN aggregation_publication_state publication
      ON publication.publication_id=node.publication_id
    WHERE node.id=new.node_id
)
BEGIN
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_node_embedding_unpublishes_update;
CREATE TRIGGER aggregation_node_embedding_unpublishes_update
AFTER UPDATE ON aggregation_node_embeddings
WHEN (old.node_id IS NOT new.node_id OR old.vector_json IS NOT new.vector_json
   OR old.model IS NOT new.model OR old.dim IS NOT new.dim
   OR old.text_hash IS NOT new.text_hash
   OR old.embedding_producer_key IS NOT new.embedding_producer_key)
 AND (EXISTS (
    SELECT 1 FROM aggregation_nodes node
    JOIN aggregation_publication_state publication
      ON publication.publication_id=node.publication_id
    WHERE node.id=old.node_id
 ) OR EXISTS (
    SELECT 1 FROM aggregation_nodes node
    JOIN aggregation_publication_state publication
      ON publication.publication_id=node.publication_id
    WHERE node.id=new.node_id
 ))
BEGIN
    DELETE FROM aggregation_publication_state;
END;
DROP TRIGGER IF EXISTS aggregation_node_embedding_unpublishes_delete;
CREATE TRIGGER aggregation_node_embedding_unpublishes_delete
AFTER DELETE ON aggregation_node_embeddings
WHEN EXISTS (
    SELECT 1 FROM aggregation_nodes node
    JOIN aggregation_publication_state publication
      ON publication.publication_id=node.publication_id
    WHERE node.id=old.node_id
)
BEGIN
    DELETE FROM aggregation_publication_state;
END;

DELETE FROM aggregation_publication_state;
UPDATE aggregation_build_health SET
    last_success_config_version=NULL,
    last_success_generation_key=NULL,
    last_success_at=NULL,
    pending_config_version=NULL,
    pending_generation_key=NULL,
    pending_attempt_token=NULL,
    pending_attempts=0,
    pending_caught_exceptions=0,
    pending_fusion_failures=0,
    first_pending_at=NULL,
    last_attempt_at=NULL;
