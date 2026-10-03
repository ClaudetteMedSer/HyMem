-- v56: exact aggregation LLM producer/effective-request identity.
-- Legacy v55 publications are withdrawn because they cannot prove which
-- producer answered their fusion prompts. Their physical rows remain only as
-- non-reusable history until the next successful full replacement.

INSERT OR IGNORE INTO schema_meta(key,value)
VALUES ('aggregation_generation_schema','56');

CREATE TABLE aggregation_generations (
    generation_key TEXT PRIMARY KEY CHECK (
        length(generation_key) = 96
        AND substr(generation_key,1,32) = 'hymem-aggregation-generation-v1:'
        AND substr(generation_key,33) NOT GLOB '*[^0-9a-f]*'
    ),
    material_config_version TEXT NOT NULL CHECK (
        length(material_config_version) = 92
        AND substr(material_config_version,1,28) =
            'aggregation-build-config-v1:'
        AND substr(material_config_version,29) NOT GLOB '*[^0-9a-f]*'
    ),
    producer_identity_sha256 TEXT NOT NULL CHECK (
        length(producer_identity_sha256) = 71
        AND producer_identity_sha256 GLOB 'sha256:*'
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

ALTER TABLE aggregation_nodes ADD COLUMN aggregation_generation_key TEXT
    REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT;
ALTER TABLE aggregation_nodes ADD COLUMN aggregation_request_hash TEXT CHECK (
    aggregation_request_hash IS NULL OR (
        length(aggregation_request_hash)=71
        AND aggregation_request_hash GLOB 'sha256:*'
    )
);
ALTER TABLE aggregation_publication_state
    ADD COLUMN aggregation_generation_key TEXT
    REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT;
ALTER TABLE aggregation_publication_state
    ADD COLUMN request_contract_sha256 TEXT CHECK (
        request_contract_sha256 IS NULL OR (
            length(request_contract_sha256)=71
            AND request_contract_sha256 GLOB 'sha256:*'
        )
    );
ALTER TABLE aggregation_build_health
    ADD COLUMN last_success_generation_key TEXT
    REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT;
ALTER TABLE aggregation_build_health
    ADD COLUMN pending_generation_key TEXT
    REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT;
ALTER TABLE aggregation_build_health
    ADD COLUMN last_failure_generation_key TEXT
    REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT;
ALTER TABLE dream_runs ADD COLUMN aggregation_generation_key TEXT
    REFERENCES aggregation_generations(generation_key) ON DELETE RESTRICT;

DROP TRIGGER IF EXISTS aggregation_generations_insert_guard;
CREATE TRIGGER aggregation_generations_insert_guard
BEFORE INSERT ON aggregation_generations
WHEN hymem_aggregation_generation_registry_row_is_valid(
    new.generation_key,new.material_config_version,
    new.producer_identity_sha256,new.identity_exact,
    new.reuse_scope,new.binding_json
) <> 1
BEGIN
    SELECT RAISE(ABORT,'invalid aggregation generation registry row');
END;

DROP TRIGGER IF EXISTS aggregation_generations_update_guard;
CREATE TRIGGER aggregation_generations_update_guard
BEFORE UPDATE ON aggregation_generations
BEGIN
    SELECT RAISE(ABORT,'aggregation generation registry is immutable');
END;

DROP TRIGGER IF EXISTS aggregation_generations_delete_guard;
CREATE TRIGGER aggregation_generations_delete_guard
BEFORE DELETE ON aggregation_generations
WHEN old.identity_exact=1
  OR EXISTS (SELECT 1 FROM aggregation_nodes
             WHERE aggregation_generation_key=old.generation_key)
  OR EXISTS (SELECT 1 FROM aggregation_publication_state
             WHERE aggregation_generation_key=old.generation_key)
  OR EXISTS (SELECT 1 FROM aggregation_build_health
             WHERE last_success_generation_key=old.generation_key
                OR pending_generation_key=old.generation_key
                OR last_failure_generation_key=old.generation_key)
  OR EXISTS (SELECT 1 FROM dream_runs
             WHERE aggregation_generation_key=old.generation_key)
BEGIN
    SELECT RAISE(ABORT,'aggregation generation registry is immutable');
END;

DROP TRIGGER IF EXISTS aggregation_generation_node_update_guard;
CREATE TRIGGER aggregation_generation_node_update_guard
BEFORE UPDATE OF aggregation_generation_key,aggregation_request_hash
ON aggregation_nodes
WHEN (old.source_manifest_complete=1 OR old.input_manifest_complete=1)
  OR new.aggregation_generation_key IS NULL
  OR new.aggregation_request_hash IS NULL
  OR NOT EXISTS (
      SELECT 1 FROM aggregation_generations generation
      WHERE generation.generation_key=new.aggregation_generation_key
        AND hymem_producer_generation_is_authorized(
            generation.generation_key,generation.identity_exact
        )=1
  )
BEGIN
    SELECT RAISE(ABORT,'invalid aggregation node generation binding');
END;

DROP TRIGGER IF EXISTS aggregation_generation_publication_insert_guard;
CREATE TRIGGER aggregation_generation_publication_insert_guard
BEFORE INSERT ON aggregation_publication_state
WHEN new.aggregation_generation_key IS NULL
  OR new.request_contract_sha256 IS NULL
  OR NOT EXISTS (
      SELECT 1 FROM aggregation_generations generation
      WHERE generation.generation_key=new.aggregation_generation_key
        AND generation.material_config_version=new.config_version
        AND json_extract(generation.binding_json,
            '$.contract.request_policy_sha256')=new.request_contract_sha256
        AND hymem_producer_generation_is_authorized(
            generation.generation_key,generation.identity_exact
        )=1
  )
  OR EXISTS (
      SELECT 1 FROM aggregation_nodes node
      WHERE node.publication_id=new.publication_id
        AND node.aggregation_generation_key IS NOT new.aggregation_generation_key
  )
  OR EXISTS (
      SELECT 1 FROM aggregation_nodes node
      WHERE node.publication_id=new.publication_id
        AND (node.aggregation_request_hash IS NULL
             OR length(node.aggregation_request_hash)<>71
             OR node.aggregation_request_hash NOT GLOB 'sha256:*')
  )
BEGIN
    SELECT RAISE(ABORT,'invalid aggregation publication generation binding');
END;

DELETE FROM aggregation_publication_state;
UPDATE aggregation_build_health SET
    last_success_config_version=NULL,
    last_success_at=NULL,
    pending_config_version=NULL,
    pending_attempts=0,
    pending_caught_exceptions=0,
    pending_fusion_failures=0,
    first_pending_at=NULL,
    last_attempt_at=NULL;
