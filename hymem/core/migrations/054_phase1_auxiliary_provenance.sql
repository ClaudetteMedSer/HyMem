-- v54: make every Phase-1 auxiliary projection producer-bound and replayable.
--
-- The pre-v54 entity hint tables are retained as compatibility/history tables.
-- Their new origin column deliberately defaults to legacy_unattributed: a
-- deleted source chunk used ON DELETE SET NULL, so NULL alone can never prove
-- that an old row was manually authored.  New manual rows must opt in to the
-- explicit user origin.
ALTER TABLE entity_types ADD COLUMN origin TEXT NOT NULL
    DEFAULT 'legacy_unattributed'
    CHECK (origin IN ('user', 'legacy_unattributed'));
ALTER TABLE entity_properties ADD COLUMN origin TEXT NOT NULL
    DEFAULT 'legacy_unattributed'
    CHECK (origin IN ('user', 'legacy_unattributed'));
ALTER TABLE profile_entries ADD COLUMN source TEXT NOT NULL
    DEFAULT 'legacy_unattributed'
    CHECK (source IN ('user', 'agent_inferred', 'legacy_unattributed'));

CREATE TABLE IF NOT EXISTS phase1_auxiliary_outcomes (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    extraction_cache_key TEXT NOT NULL,
    auxiliary_contract_key TEXT NOT NULL,
    result_hash TEXT NOT NULL CHECK (
        substr(result_hash, 1, 7) = 'sha256:'
        AND length(result_hash) = 71
        AND substr(result_hash, 8) NOT GLOB '*[^0-9a-f]*'
    ),
    entity_type_count INTEGER NOT NULL CHECK (entity_type_count >= 0),
    entity_property_count INTEGER NOT NULL CHECK (entity_property_count >= 0),
    entity_mention_count INTEGER NOT NULL CHECK (entity_mention_count >= 0),
    marker_count INTEGER NOT NULL CHECK (marker_count >= 0),
    published_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_id, phase1_generation_key)
);
CREATE INDEX IF NOT EXISTS idx_phase1_auxiliary_generation
    ON phase1_auxiliary_outcomes(phase1_generation_key, chunk_id);

CREATE TABLE IF NOT EXISTS entity_type_observations (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_canonical TEXT NOT NULL CHECK (
        length(trim(entity_canonical)) > 0
        AND hymem_entity_canonical_is_normalized(entity_canonical) = 1
    ),
    type TEXT NOT NULL CHECK (length(trim(type)) > 0),
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (
        typeof(confidence) IN ('integer', 'real')
        AND confidence >= 0.0 AND confidence <= 1.0
    ),
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (
        chunk_id, entity_canonical, type, phase1_generation_key
    )
);
CREATE INDEX IF NOT EXISTS idx_entity_type_observations_lookup
    ON entity_type_observations(type, entity_canonical, phase1_generation_key);
CREATE INDEX IF NOT EXISTS idx_entity_type_observations_entity
    ON entity_type_observations(entity_canonical, phase1_generation_key);

CREATE TABLE IF NOT EXISTS entity_property_observations (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_canonical TEXT NOT NULL CHECK (
        length(trim(entity_canonical)) > 0
        AND hymem_entity_canonical_is_normalized(entity_canonical) = 1
    ),
    key TEXT NOT NULL CHECK (length(trim(key)) > 0),
    value TEXT NOT NULL,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (
        chunk_id, entity_canonical, key, phase1_generation_key
    )
);
CREATE INDEX IF NOT EXISTS idx_entity_property_observations_lookup
    ON entity_property_observations(key, value, entity_canonical,
                                    phase1_generation_key);
CREATE INDEX IF NOT EXISTS idx_entity_property_observations_entity
    ON entity_property_observations(entity_canonical, key,
                                    phase1_generation_key);

CREATE TABLE IF NOT EXISTS entity_mention_observations (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_canonical TEXT NOT NULL CHECK (
        length(trim(entity_canonical)) > 0
        AND hymem_entity_canonical_is_normalized(entity_canonical) = 1
    ),
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_id, entity_canonical, phase1_generation_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_mention_observations_entity
    ON entity_mention_observations(entity_canonical,phase1_generation_key);

CREATE TABLE IF NOT EXISTS profile_entry_marker_evidence (
    profile_entry_id INTEGER NOT NULL
        REFERENCES profile_entries(id) ON DELETE CASCADE,
    marker_id INTEGER NOT NULL
        REFERENCES behavioral_markers(id) ON DELETE CASCADE,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (profile_entry_id, marker_id)
);
CREATE INDEX IF NOT EXISTS idx_profile_marker_generation
    ON profile_entry_marker_evidence(phase1_generation_key, marker_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_profile_marker_one_decision
    ON profile_entry_marker_evidence(marker_id);

CREATE TABLE IF NOT EXISTS profile_marker_decisions (
    marker_id INTEGER PRIMARY KEY
        REFERENCES behavioral_markers(id) ON DELETE CASCADE,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    profile_policy_key TEXT NOT NULL CHECK (
        length(trim(profile_policy_key)) > 0
    ),
    decision TEXT NOT NULL CHECK (
        decision IN ('materialized', 'manual_authority', 'identity_conflict')
    ),
    profile_entry_id INTEGER NOT NULL
        REFERENCES profile_entries(id) ON DELETE CASCADE,
    decided_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_profile_marker_decision_generation
    ON profile_marker_decisions(phase1_generation_key, marker_id);

CREATE TABLE IF NOT EXISTS rule_marker_evidence (
    rule_id INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    marker_id INTEGER NOT NULL
        REFERENCES behavioral_markers(id) ON DELETE CASCADE,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (rule_id, marker_id)
);
CREATE INDEX IF NOT EXISTS idx_rule_marker_generation
    ON rule_marker_evidence(phase1_generation_key, marker_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rule_marker_one_decision
    ON rule_marker_evidence(marker_id);

CREATE TABLE IF NOT EXISTS rule_marker_decisions (
    marker_id INTEGER PRIMARY KEY
        REFERENCES behavioral_markers(id) ON DELETE CASCADE,
    phase1_generation_key TEXT NOT NULL
        REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT,
    routing_key TEXT NOT NULL CHECK (length(trim(routing_key)) > 0),
    decision TEXT NOT NULL CHECK (decision IN ('routed', 'no_rule')),
    rule_id INTEGER REFERENCES rules(id) ON DELETE SET NULL,
    decided_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((decision = 'routed' AND rule_id IS NOT NULL)
        OR (decision = 'no_rule' AND rule_id IS NULL))
);
CREATE INDEX IF NOT EXISTS idx_rule_marker_decision_generation
    ON rule_marker_decisions(phase1_generation_key, marker_id);

-- Existing v53 cache acknowledgements do not prove that the auxiliary result
-- was captured under v54.  Preserve them as history but force one exact replay.
UPDATE processed_chunks SET phase1_generation_key = NULL
WHERE phase1_generation_key IS NOT NULL;

-- A table named like the future v54 ledger can exist in a manually repaired or
-- test-reconstructed older store.  Without any claim outcome its contents have
-- no parent publication at all, so remove that branch before the v54 stamp.
-- (A claim with a NULL generation is retained as explicitly producer-untrusted
-- history and remains ineligible for current reads.)
DELETE FROM behavioral_markers
WHERE phase1_generation_key IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM kg_claim_extraction_outcomes claim
      WHERE claim.chunk_id=behavioral_markers.chunk_id
  );
DELETE FROM entity_type_observations
WHERE NOT EXISTS (
    SELECT 1 FROM kg_claim_extraction_outcomes claim
    WHERE claim.chunk_id=entity_type_observations.chunk_id
);
DELETE FROM entity_property_observations
WHERE NOT EXISTS (
    SELECT 1 FROM kg_claim_extraction_outcomes claim
    WHERE claim.chunk_id=entity_property_observations.chunk_id
);
DELETE FROM entity_mention_observations
WHERE NOT EXISTS (
    SELECT 1 FROM kg_claim_extraction_outcomes claim
    WHERE claim.chunk_id=entity_mention_observations.chunk_id
);
DELETE FROM phase1_auxiliary_outcomes
WHERE NOT EXISTS (
    SELECT 1 FROM kg_claim_extraction_outcomes claim
    WHERE claim.chunk_id=phase1_auxiliary_outcomes.chunk_id
);

-- V53 producer keys covered only the claim projection.  A pre-v54 marker has
-- no whole-response auxiliary outcome/hash, so its generation column cannot
-- be treated as v54 authority even when the claim producer identity itself is
-- exact.  Keep the marker as explicitly unscoped history; a successful replay
-- will publish a new producer-bound row and materialization decision.
UPDATE behavioral_markers SET phase1_generation_key = NULL
WHERE phase1_generation_key IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM phase1_auxiliary_outcomes auxiliary
      WHERE auxiliary.chunk_id=behavioral_markers.chunk_id
        AND auxiliary.phase1_generation_key=
            behavioral_markers.phase1_generation_key
  );

-- A generation switch may retain historical marker rows.  Identity therefore
-- includes the generation; duplicate rows from old code are collapsed first.
DELETE FROM behavioral_markers
WHERE phase1_generation_key IS NOT NULL
  AND id NOT IN (
      SELECT MIN(id) FROM behavioral_markers
      WHERE phase1_generation_key IS NOT NULL
      GROUP BY chunk_id, phase1_generation_key, kind, statement
  );
CREATE UNIQUE INDEX IF NOT EXISTS idx_behavioral_marker_generation_identity
    ON behavioral_markers(
        chunk_id, phase1_generation_key, kind, statement
    ) WHERE phase1_generation_key IS NOT NULL;

DROP VIEW IF EXISTS current_phase1_publications;
CREATE VIEW current_phase1_publications AS
SELECT processed.chunk_id, processed.prompt_version,
       processed.phase1_generation_key, auxiliary.auxiliary_contract_key,
       auxiliary.result_hash
FROM processed_chunks processed
JOIN kg_claim_extraction_outcomes claim
  ON claim.chunk_id=processed.chunk_id
 AND claim.prompt_version=processed.prompt_version
 AND claim.phase1_generation_key=processed.phase1_generation_key
JOIN phase1_auxiliary_outcomes auxiliary
  ON auxiliary.chunk_id=processed.chunk_id
 AND auxiliary.phase1_generation_key=processed.phase1_generation_key
 AND auxiliary.extraction_cache_key=processed.prompt_version
JOIN phase1_generations generation
  ON generation.generation_key=processed.phase1_generation_key
 AND generation.extraction_cache_key=processed.prompt_version
WHERE hymem_phase1_generation_is_current(
          generation.generation_key,generation.identity_exact
      )=1
  AND hymem_auxiliary_contract_is_current(
          auxiliary.auxiliary_contract_key
      )=1;

DROP VIEW IF EXISTS current_entity_types;
CREATE VIEW current_entity_types AS
WITH authorized AS (
    SELECT observation.entity_canonical, observation.type,
           observation.confidence, observation.chunk_id,
           observation.phase1_generation_key, chunk.created_at,
           ROW_NUMBER() OVER (
               PARTITION BY observation.entity_canonical, observation.type
               ORDER BY hymem_normalize_iso_timestamp(chunk.created_at) DESC,
                        chunk.session_id DESC,
                        observation.chunk_id DESC,
                        observation.phase1_generation_key DESC,
                        observation.confidence DESC
           ) AS authority_rank
    FROM entity_type_observations observation
    JOIN chunks chunk ON chunk.id = observation.chunk_id
    JOIN kg_claim_extraction_outcomes claim
      ON claim.chunk_id = observation.chunk_id
     AND claim.phase1_generation_key = observation.phase1_generation_key
    JOIN phase1_auxiliary_outcomes auxiliary
      ON auxiliary.chunk_id = observation.chunk_id
     AND auxiliary.phase1_generation_key = observation.phase1_generation_key
     AND auxiliary.extraction_cache_key = claim.prompt_version
     AND hymem_auxiliary_contract_is_current(
             auxiliary.auxiliary_contract_key
         ) = 1
    JOIN processed_chunks processed
      ON processed.chunk_id = claim.chunk_id
     AND processed.prompt_version = claim.prompt_version
     AND processed.phase1_generation_key = claim.phase1_generation_key
    JOIN phase1_generations generation
      ON generation.generation_key = claim.phase1_generation_key
     AND generation.extraction_cache_key = claim.prompt_version
    WHERE hymem_phase1_generation_is_current(
              generation.generation_key, generation.identity_exact
          ) = 1
)
SELECT entity_canonical, type, confidence, source_chunk_id, origin
FROM entity_types
WHERE origin = 'user'
UNION ALL
SELECT authorized.entity_canonical, authorized.type,
       authorized.confidence, authorized.chunk_id, 'agent_inferred'
FROM authorized
WHERE authorized.authority_rank = 1
  AND NOT EXISTS (
      SELECT 1 FROM entity_types manual
      WHERE manual.entity_canonical = authorized.entity_canonical
        AND manual.type = authorized.type
        AND manual.origin = 'user'
  );

DROP VIEW IF EXISTS current_entity_properties;
CREATE VIEW current_entity_properties AS
WITH authorized AS (
    SELECT observation.entity_canonical, observation.key, observation.value,
           observation.chunk_id, observation.phase1_generation_key,
           chunk.created_at,
           ROW_NUMBER() OVER (
               PARTITION BY observation.entity_canonical, observation.key
               ORDER BY hymem_normalize_iso_timestamp(chunk.created_at) DESC,
                        chunk.session_id DESC,
                        observation.chunk_id DESC,
                        observation.phase1_generation_key DESC,
                        observation.value DESC
           ) AS authority_rank
    FROM entity_property_observations observation
    JOIN chunks chunk ON chunk.id = observation.chunk_id
    JOIN kg_claim_extraction_outcomes claim
      ON claim.chunk_id = observation.chunk_id
     AND claim.phase1_generation_key = observation.phase1_generation_key
    JOIN phase1_auxiliary_outcomes auxiliary
      ON auxiliary.chunk_id = observation.chunk_id
     AND auxiliary.phase1_generation_key = observation.phase1_generation_key
     AND auxiliary.extraction_cache_key = claim.prompt_version
     AND hymem_auxiliary_contract_is_current(
             auxiliary.auxiliary_contract_key
         ) = 1
    JOIN processed_chunks processed
      ON processed.chunk_id = claim.chunk_id
     AND processed.prompt_version = claim.prompt_version
     AND processed.phase1_generation_key = claim.phase1_generation_key
    JOIN phase1_generations generation
      ON generation.generation_key = claim.phase1_generation_key
     AND generation.extraction_cache_key = claim.prompt_version
    WHERE hymem_phase1_generation_is_current(
              generation.generation_key, generation.identity_exact
          ) = 1
)
SELECT entity_canonical, key, value, source_chunk_id, updated_at, origin
FROM entity_properties
WHERE origin = 'user'
UNION ALL
SELECT authorized.entity_canonical, authorized.key, authorized.value,
       authorized.chunk_id, authorized.created_at, 'agent_inferred'
FROM authorized
WHERE authorized.authority_rank = 1
  AND NOT EXISTS (
      SELECT 1 FROM entity_properties manual
      WHERE manual.entity_canonical = authorized.entity_canonical
        AND manual.key = authorized.key
        AND manual.origin = 'user'
  );

DROP VIEW IF EXISTS current_entity_mentions;
CREATE VIEW current_entity_mentions AS
SELECT observation.chunk_id,observation.entity_canonical,
       observation.phase1_generation_key,observation.observed_at
FROM entity_mention_observations observation
JOIN current_phase1_publications publication
  ON publication.chunk_id=observation.chunk_id
 AND publication.phase1_generation_key=observation.phase1_generation_key;

DROP VIEW IF EXISTS current_profile_entries;
CREATE VIEW current_profile_entries AS
SELECT id, kind, text, pos_evidence, neg_evidence, first_seen,
       last_updated, source
FROM profile_entries
WHERE source = 'user'
UNION ALL
SELECT entry.id, entry.kind, entry.text,
       COUNT(DISTINCT link.marker_id) AS pos_evidence,
       entry.neg_evidence, entry.first_seen, entry.last_updated, entry.source
FROM profile_entries entry
JOIN profile_entry_marker_evidence link
  ON link.profile_entry_id = entry.id
JOIN profile_marker_decisions decision
  ON decision.marker_id = link.marker_id
 AND decision.profile_entry_id = link.profile_entry_id
 AND decision.phase1_generation_key = link.phase1_generation_key
 AND hymem_profile_policy_is_current(decision.profile_policy_key) = 1
 AND decision.decision = 'materialized'
JOIN behavioral_markers marker
  ON marker.id = link.marker_id
 AND marker.phase1_generation_key = link.phase1_generation_key
JOIN kg_claim_extraction_outcomes claim
  ON claim.chunk_id = marker.chunk_id
 AND claim.phase1_generation_key = marker.phase1_generation_key
JOIN phase1_auxiliary_outcomes auxiliary
  ON auxiliary.chunk_id = marker.chunk_id
 AND auxiliary.phase1_generation_key = marker.phase1_generation_key
 AND auxiliary.extraction_cache_key = claim.prompt_version
 AND hymem_auxiliary_contract_is_current(
         auxiliary.auxiliary_contract_key
     ) = 1
JOIN processed_chunks processed
  ON processed.chunk_id = claim.chunk_id
 AND processed.prompt_version = claim.prompt_version
 AND processed.phase1_generation_key = claim.phase1_generation_key
JOIN phase1_generations generation
  ON generation.generation_key = claim.phase1_generation_key
 AND generation.extraction_cache_key = claim.prompt_version
WHERE entry.source = 'agent_inferred'
  AND hymem_phase1_generation_is_current(
          generation.generation_key, generation.identity_exact
      ) = 1
GROUP BY entry.id;

DROP VIEW IF EXISTS current_rules;
CREATE VIEW current_rules AS
SELECT id, text, scope, trigger_entities, source, pos_evidence, neg_evidence,
       valid_at, invalid_at, status, created_at
FROM rules
WHERE source = 'user'
  AND status = 'active' AND invalid_at IS NULL
UNION ALL
SELECT rule.id, rule.text, rule.scope, rule.trigger_entities, rule.source,
       COUNT(DISTINCT link.marker_id) AS pos_evidence,
       rule.neg_evidence, rule.valid_at, rule.invalid_at, rule.status,
       rule.created_at
FROM rules rule
JOIN rule_marker_evidence link ON link.rule_id = rule.id
JOIN rule_marker_decisions decision
  ON decision.marker_id = link.marker_id
 AND decision.rule_id = link.rule_id
 AND decision.phase1_generation_key = link.phase1_generation_key
 AND decision.decision = 'routed'
JOIN behavioral_markers marker
  ON marker.id = link.marker_id
 AND marker.phase1_generation_key = link.phase1_generation_key
JOIN kg_claim_extraction_outcomes claim
  ON claim.chunk_id = marker.chunk_id
 AND claim.phase1_generation_key = marker.phase1_generation_key
JOIN phase1_auxiliary_outcomes auxiliary
  ON auxiliary.chunk_id = marker.chunk_id
 AND auxiliary.phase1_generation_key = marker.phase1_generation_key
 AND auxiliary.extraction_cache_key = claim.prompt_version
 AND hymem_auxiliary_contract_is_current(
         auxiliary.auxiliary_contract_key
     ) = 1
JOIN processed_chunks processed
  ON processed.chunk_id = claim.chunk_id
 AND processed.prompt_version = claim.prompt_version
 AND processed.phase1_generation_key = claim.phase1_generation_key
JOIN phase1_generations generation
  ON generation.generation_key = claim.phase1_generation_key
 AND generation.extraction_cache_key = claim.prompt_version
WHERE rule.source = 'agent_inferred'
  AND rule.status = 'active' AND rule.invalid_at IS NULL
  AND hymem_phase1_generation_is_current(
          generation.generation_key, generation.identity_exact
      ) = 1
  AND hymem_rule_routing_is_current(decision.routing_key) = 1
GROUP BY rule.id;

CREATE TRIGGER IF NOT EXISTS profile_marker_evidence_lineage_guard
BEFORE INSERT ON profile_entry_marker_evidence
WHEN hymem_evidence_mutation_authorized() <> 1
  OR NOT EXISTS (
    SELECT 1 FROM behavioral_markers marker
    JOIN profile_entries entry ON entry.id = new.profile_entry_id
    WHERE marker.id = new.marker_id
      AND marker.phase1_generation_key = new.phase1_generation_key
      AND entry.source = 'agent_inferred'
      AND entry.text = marker.statement
      AND entry.kind = CASE marker.kind
          WHEN 'preference' THEN 'preference'
          WHEN 'rejection' THEN 'avoidance'
          WHEN 'style' THEN 'style'
          ELSE 'context'
      END
)
BEGIN
    SELECT RAISE(ABORT, 'invalid profile marker lineage');
END;

CREATE TRIGGER IF NOT EXISTS profile_marker_decision_insert_guard
BEFORE INSERT ON profile_marker_decisions
WHEN hymem_evidence_mutation_authorized() <> 1
  OR NOT EXISTS (
      SELECT 1 FROM behavioral_markers marker
      JOIN profile_entries entry ON entry.id=new.profile_entry_id
      WHERE marker.id=new.marker_id
        AND marker.phase1_generation_key=new.phase1_generation_key
        AND length(new.profile_policy_key)>0
        AND (
          (new.decision='materialized'
           AND entry.source='agent_inferred'
           AND entry.text=marker.statement
           AND entry.kind=CASE marker.kind
             WHEN 'preference' THEN 'preference'
             WHEN 'rejection' THEN 'avoidance'
             WHEN 'style' THEN 'style'
             ELSE 'context' END)
          OR (new.decision='manual_authority' AND entry.source='user'
              AND entry.text=marker.statement)
          OR (new.decision='identity_conflict'
              AND entry.source<>'user'
              AND entry.text=marker.statement
              AND entry.kind<>CASE marker.kind
                WHEN 'preference' THEN 'preference'
                WHEN 'rejection' THEN 'avoidance'
                WHEN 'style' THEN 'style'
                ELSE 'context' END)
        )
        AND (new.decision<>'materialized' OR EXISTS (
          SELECT 1 FROM profile_entry_marker_evidence link
          WHERE link.marker_id=new.marker_id
            AND link.profile_entry_id=new.profile_entry_id
            AND link.phase1_generation_key=new.phase1_generation_key
        ))
        AND (new.decision='materialized' OR NOT EXISTS (
          SELECT 1 FROM profile_entry_marker_evidence link
          WHERE link.marker_id=new.marker_id
        ))
  )
BEGIN SELECT RAISE(ABORT, 'invalid profile marker decision'); END;

CREATE TRIGGER IF NOT EXISTS rule_marker_evidence_lineage_guard
BEFORE INSERT ON rule_marker_evidence
WHEN hymem_evidence_mutation_authorized() <> 1
  OR NOT EXISTS (
    SELECT 1 FROM behavioral_markers marker
    JOIN rules rule ON rule.id = new.rule_id
    WHERE marker.id = new.marker_id
      AND marker.phase1_generation_key = new.phase1_generation_key
      AND rule.source = 'agent_inferred'
)
BEGIN
    SELECT RAISE(ABORT, 'invalid rule marker lineage');
END;

CREATE TRIGGER IF NOT EXISTS phase1_auxiliary_outcome_insert_guard
BEFORE INSERT ON phase1_auxiliary_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS phase1_auxiliary_outcome_update_guard
BEFORE UPDATE ON phase1_auxiliary_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS phase1_auxiliary_outcome_delete_guard
BEFORE DELETE ON phase1_auxiliary_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;

CREATE TRIGGER IF NOT EXISTS entity_type_observation_insert_guard
BEFORE INSERT ON entity_type_observations
WHEN hymem_evidence_mutation_authorized() <> 1
  OR hymem_entity_canonical_is_normalized(new.entity_canonical) <> 1
  OR length(trim(new.type)) = 0
  OR typeof(new.confidence) NOT IN ('integer', 'real')
  OR new.confidence < 0.0 OR new.confidence > 1.0
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_type_observation_update_guard
BEFORE UPDATE ON entity_type_observations
WHEN hymem_evidence_mutation_authorized() <> 1
  OR hymem_entity_canonical_is_normalized(new.entity_canonical) <> 1
  OR length(trim(new.type)) = 0
  OR typeof(new.confidence) NOT IN ('integer', 'real')
  OR new.confidence < 0.0 OR new.confidence > 1.0
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_type_observation_delete_guard
BEFORE DELETE ON entity_type_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;

CREATE TRIGGER IF NOT EXISTS entity_property_observation_insert_guard
BEFORE INSERT ON entity_property_observations
WHEN hymem_evidence_mutation_authorized() <> 1
  OR hymem_entity_canonical_is_normalized(new.entity_canonical) <> 1
  OR length(trim(new.key)) = 0
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_property_observation_update_guard
BEFORE UPDATE ON entity_property_observations
WHEN hymem_evidence_mutation_authorized() <> 1
  OR hymem_entity_canonical_is_normalized(new.entity_canonical) <> 1
  OR length(trim(new.key)) = 0
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_property_observation_delete_guard
BEFORE DELETE ON entity_property_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;

CREATE TRIGGER IF NOT EXISTS entity_mention_observation_insert_guard
BEFORE INSERT ON entity_mention_observations
WHEN hymem_evidence_mutation_authorized() <> 1
  OR hymem_entity_canonical_is_normalized(new.entity_canonical) <> 1
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_mention_observation_update_guard
BEFORE UPDATE ON entity_mention_observations
WHEN hymem_evidence_mutation_authorized() <> 1
  OR hymem_entity_canonical_is_normalized(new.entity_canonical) <> 1
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS entity_mention_observation_delete_guard
BEFORE DELETE ON entity_mention_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'Phase-1 auxiliaries are internally managed'); END;

CREATE TRIGGER IF NOT EXISTS profile_marker_evidence_update_guard
BEFORE UPDATE ON profile_entry_marker_evidence
BEGIN SELECT RAISE(ABORT, 'profile marker evidence is immutable'); END;
CREATE TRIGGER IF NOT EXISTS profile_marker_evidence_delete_guard
BEFORE DELETE ON profile_entry_marker_evidence
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'profile marker evidence is internally managed'); END;
CREATE TRIGGER IF NOT EXISTS profile_marker_decision_update_guard
BEFORE UPDATE ON profile_marker_decisions
BEGIN SELECT RAISE(ABORT, 'profile marker decision is immutable'); END;
CREATE TRIGGER IF NOT EXISTS profile_marker_decision_delete_guard
BEFORE DELETE ON profile_marker_decisions
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'profile marker decision is internally managed'); END;
CREATE TRIGGER IF NOT EXISTS rule_marker_evidence_update_guard
BEFORE UPDATE ON rule_marker_evidence
BEGIN SELECT RAISE(ABORT, 'rule marker evidence is immutable'); END;
CREATE TRIGGER IF NOT EXISTS rule_marker_evidence_delete_guard
BEFORE DELETE ON rule_marker_evidence
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'rule marker evidence is internally managed'); END;
CREATE TRIGGER IF NOT EXISTS rule_marker_decision_insert_guard
BEFORE INSERT ON rule_marker_decisions
WHEN hymem_evidence_mutation_authorized() <> 1
  OR length(trim(new.routing_key)) = 0
  OR NOT EXISTS (
      SELECT 1 FROM behavioral_markers marker
      WHERE marker.id=new.marker_id
        AND marker.phase1_generation_key=new.phase1_generation_key
  )
  OR (new.decision='routed' AND NOT EXISTS (
      SELECT 1 FROM rules rule
      WHERE rule.id=new.rule_id AND rule.source='agent_inferred'
  ))
  OR (new.decision='routed' AND NOT EXISTS (
      SELECT 1 FROM rule_marker_evidence link
      WHERE link.marker_id=new.marker_id AND link.rule_id=new.rule_id
        AND link.phase1_generation_key=new.phase1_generation_key
  ))
  OR (new.decision='no_rule' AND EXISTS (
      SELECT 1 FROM rule_marker_evidence link
      WHERE link.marker_id=new.marker_id
  ))
BEGIN SELECT RAISE(ABORT, 'invalid rule marker decision'); END;
CREATE TRIGGER IF NOT EXISTS rule_marker_decision_update_guard
BEFORE UPDATE ON rule_marker_decisions
BEGIN SELECT RAISE(ABORT, 'rule marker decision is immutable'); END;
CREATE TRIGGER IF NOT EXISTS rule_marker_decision_delete_guard
BEFORE DELETE ON rule_marker_decisions
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'rule marker decision is internally managed'); END;

CREATE TRIGGER IF NOT EXISTS behavioral_marker_producer_insert_guard
BEFORE INSERT ON behavioral_markers
WHEN new.phase1_generation_key IS NOT NULL AND (
  hymem_evidence_mutation_authorized() <> 1
  OR length(trim(new.statement)) = 0
)
BEGIN SELECT RAISE(ABORT, 'producer-bound markers are internally managed'); END;
CREATE TRIGGER IF NOT EXISTS behavioral_marker_semantic_update_guard
BEFORE UPDATE OF kind,statement,chunk_id,phase1_generation_key
ON behavioral_markers
WHEN old.phase1_generation_key IS NOT NULL
 AND (hymem_evidence_mutation_authorized() <> 1
      OR length(trim(new.statement)) = 0)
BEGIN SELECT RAISE(ABORT, 'producer-bound marker identity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS behavioral_marker_delete_guard
BEFORE DELETE ON behavioral_markers
WHEN old.phase1_generation_key IS NOT NULL
 AND hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'producer-bound markers are internally managed'); END;

CREATE TRIGGER IF NOT EXISTS linked_profile_semantic_update_guard
BEFORE UPDATE OF kind,text,source ON profile_entries
WHEN EXISTS (
    SELECT 1 FROM profile_entry_marker_evidence link
    WHERE link.profile_entry_id=old.id
) AND hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'linked inferred profile materialization is protected'); END;
CREATE TRIGGER IF NOT EXISTS linked_profile_delete_guard
BEFORE DELETE ON profile_entries
WHEN EXISTS (
    SELECT 1 FROM profile_entry_marker_evidence link
    WHERE link.profile_entry_id=old.id
) AND hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'linked inferred profile materialization is protected'); END;

CREATE TRIGGER IF NOT EXISTS linked_rule_semantic_update_guard
BEFORE UPDATE OF text,scope,trigger_entities,source,status,valid_at,invalid_at
ON rules
WHEN EXISTS (
    SELECT 1 FROM rule_marker_evidence link WHERE link.rule_id=old.id
) AND hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'linked inferred rule materialization is protected'); END;
CREATE TRIGGER IF NOT EXISTS linked_rule_delete_guard
BEFORE DELETE ON rules
WHEN EXISTS (
    SELECT 1 FROM rule_marker_evidence link WHERE link.rule_id=old.id
) AND hymem_evidence_mutation_authorized() <> 1
BEGIN SELECT RAISE(ABORT, 'linked inferred rule materialization is protected'); END;

-- Explicit/manual compatibility projections are public write surfaces.  Keep
-- their accepted domain aligned with the v14 wire contract so a successful
-- writer can never create a store that fails its own export/reopen checks.
CREATE TRIGGER IF NOT EXISTS entity_type_authority_insert_guard
BEFORE INSERT ON entity_types
WHEN hymem_entity_canonical_is_normalized(new.entity_canonical)<>1
  OR length(trim(new.type))=0
  OR typeof(new.confidence) NOT IN ('integer','real')
  OR new.confidence<0.0 OR new.confidence>1.0
  OR (new.origin='user' AND new.source_chunk_id IS NOT NULL)
BEGIN SELECT RAISE(ABORT, 'invalid entity type authority'); END;
CREATE TRIGGER IF NOT EXISTS entity_type_authority_update_guard
BEFORE UPDATE ON entity_types
WHEN hymem_entity_canonical_is_normalized(new.entity_canonical)<>1
  OR length(trim(new.type))=0
  OR typeof(new.confidence) NOT IN ('integer','real')
  OR new.confidence<0.0 OR new.confidence>1.0
  OR (new.origin='user' AND new.source_chunk_id IS NOT NULL)
BEGIN SELECT RAISE(ABORT, 'invalid entity type authority'); END;
CREATE TRIGGER IF NOT EXISTS entity_property_authority_insert_guard
BEFORE INSERT ON entity_properties
WHEN hymem_entity_canonical_is_normalized(new.entity_canonical)<>1
  OR length(trim(new.key))=0
  OR (new.origin='user' AND new.source_chunk_id IS NOT NULL)
BEGIN SELECT RAISE(ABORT, 'invalid entity property authority'); END;
CREATE TRIGGER IF NOT EXISTS entity_property_authority_update_guard
BEFORE UPDATE ON entity_properties
WHEN hymem_entity_canonical_is_normalized(new.entity_canonical)<>1
  OR length(trim(new.key))=0
  OR (new.origin='user' AND new.source_chunk_id IS NOT NULL)
BEGIN SELECT RAISE(ABORT, 'invalid entity property authority'); END;

CREATE TRIGGER IF NOT EXISTS profile_entry_domain_insert_guard
BEFORE INSERT ON profile_entries
WHEN length(trim(new.text))=0
  OR typeof(new.pos_evidence)<>'integer' OR new.pos_evidence<0
  OR typeof(new.neg_evidence)<>'integer' OR new.neg_evidence<0
  OR (new.first_seen IS NOT NULL AND (
      typeof(new.first_seen)<>'text' OR length(trim(new.first_seen))=0))
  OR (new.last_updated IS NOT NULL AND (
      typeof(new.last_updated)<>'text' OR length(trim(new.last_updated))=0))
BEGIN SELECT RAISE(ABORT, 'invalid profile entry domain'); END;
CREATE TRIGGER IF NOT EXISTS profile_entry_domain_update_guard
BEFORE UPDATE ON profile_entries
WHEN length(trim(new.text))=0
  OR typeof(new.pos_evidence)<>'integer' OR new.pos_evidence<0
  OR typeof(new.neg_evidence)<>'integer' OR new.neg_evidence<0
  OR (new.first_seen IS NOT NULL AND (
      typeof(new.first_seen)<>'text' OR length(trim(new.first_seen))=0))
  OR (new.last_updated IS NOT NULL AND (
      typeof(new.last_updated)<>'text' OR length(trim(new.last_updated))=0))
BEGIN SELECT RAISE(ABORT, 'invalid profile entry domain'); END;

CREATE TRIGGER IF NOT EXISTS rule_domain_insert_guard
BEFORE INSERT ON rules
WHEN length(trim(new.text))=0
  OR typeof(new.pos_evidence)<>'integer' OR new.pos_evidence<0
  OR typeof(new.neg_evidence)<>'integer' OR new.neg_evidence<0
  OR (new.valid_at IS NOT NULL AND (
      typeof(new.valid_at)<>'text' OR length(trim(new.valid_at))=0))
  OR (new.invalid_at IS NOT NULL AND (
      typeof(new.invalid_at)<>'text' OR length(trim(new.invalid_at))=0))
  OR (new.created_at IS NOT NULL AND (
      typeof(new.created_at)<>'text' OR length(trim(new.created_at))=0))
  OR json_valid(new.trigger_entities)<>1
  OR json_type(new.trigger_entities)<>'array'
  OR EXISTS (
      SELECT 1 FROM json_each(
          CASE WHEN json_valid(new.trigger_entities)
               THEN new.trigger_entities ELSE '[]' END
      ) item
      WHERE item.type<>'text' OR length(trim(item.value))=0
         OR hymem_entity_canonical_is_normalized(item.value)<>1
  )
  OR (new.scope='always_on' AND json_array_length(
      CASE WHEN json_valid(new.trigger_entities)
           THEN new.trigger_entities ELSE '[]' END)<>0)
  OR (new.scope='contextual' AND json_array_length(
      CASE WHEN json_valid(new.trigger_entities)
           THEN new.trigger_entities ELSE '[]' END)=0)
BEGIN SELECT RAISE(ABORT, 'invalid rule domain'); END;
CREATE TRIGGER IF NOT EXISTS rule_domain_update_guard
BEFORE UPDATE ON rules
WHEN length(trim(new.text))=0
  OR typeof(new.pos_evidence)<>'integer' OR new.pos_evidence<0
  OR typeof(new.neg_evidence)<>'integer' OR new.neg_evidence<0
  OR (new.valid_at IS NOT NULL AND (
      typeof(new.valid_at)<>'text' OR length(trim(new.valid_at))=0))
  OR (new.invalid_at IS NOT NULL AND (
      typeof(new.invalid_at)<>'text' OR length(trim(new.invalid_at))=0))
  OR (new.created_at IS NOT NULL AND (
      typeof(new.created_at)<>'text' OR length(trim(new.created_at))=0))
  OR json_valid(new.trigger_entities)<>1
  OR json_type(new.trigger_entities)<>'array'
  OR EXISTS (
      SELECT 1 FROM json_each(
          CASE WHEN json_valid(new.trigger_entities)
               THEN new.trigger_entities ELSE '[]' END
      ) item
      WHERE item.type<>'text' OR length(trim(item.value))=0
         OR hymem_entity_canonical_is_normalized(item.value)<>1
  )
  OR (new.scope='always_on' AND json_array_length(
      CASE WHEN json_valid(new.trigger_entities)
           THEN new.trigger_entities ELSE '[]' END)<>0)
  OR (new.scope='contextual' AND json_array_length(
      CASE WHEN json_valid(new.trigger_entities)
           THEN new.trigger_entities ELSE '[]' END)=0)
BEGIN SELECT RAISE(ABORT, 'invalid rule domain'); END;
