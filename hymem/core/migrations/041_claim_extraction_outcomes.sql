-- v41: portable whole-chunk claim extraction authority.
--
-- Observation rows describe returned claims, so an empty successful replay has
-- no row capable of superseding an older non-empty snapshot. This compact,
-- guarded publication marker carries the latest prompt generation and a hash
-- of the complete portable observation set, including the empty set.
CREATE TABLE IF NOT EXISTS kg_claim_extraction_outcomes (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE RESTRICT,
    prompt_version TEXT NOT NULL,
    prompt_generation INTEGER NOT NULL CHECK (prompt_generation >= 0),
    result_hash TEXT NOT NULL,
    succeeded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER IF NOT EXISTS kg_claim_extraction_outcomes_insert_guard
BEFORE INSERT ON kg_claim_extraction_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
  OR length(trim(new.prompt_version)) = 0
  OR new.prompt_generation < 0
  OR substr(new.result_hash, 1, 7) <> 'sha256:'
  OR length(new.result_hash) <> 71
  OR substr(new.result_hash, 8) GLOB '*[^0-9a-f]*'
BEGIN
    SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
END;

CREATE TRIGGER IF NOT EXISTS kg_claim_extraction_outcomes_update_guard
BEFORE UPDATE ON kg_claim_extraction_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
  OR length(trim(new.prompt_version)) = 0
  OR new.prompt_generation < 0
  OR substr(new.result_hash, 1, 7) <> 'sha256:'
  OR length(new.result_hash) <> 71
  OR substr(new.result_hash, 8) GLOB '*[^0-9a-f]*'
BEGIN
    SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
END;

CREATE TRIGGER IF NOT EXISTS kg_claim_extraction_outcomes_delete_guard
BEFORE DELETE ON kg_claim_extraction_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
END;

-- Publication proof is durable while an outcome exists, including an empty
-- one with no observation/evidence FK to protect the manifest on its behalf.
DROP TRIGGER IF EXISTS chunk_source_manifest_header_update_guard;
CREATE TRIGGER chunk_source_manifest_header_update_guard
BEFORE UPDATE OF source_manifest_version, source_manifest_count ON chunks
WHEN old.source_manifest_version IS NOT NULL
 AND (new.source_manifest_version IS NOT old.source_manifest_version
      OR new.source_manifest_count IS NOT old.source_manifest_count)
 AND NOT (
      new.source_manifest_version IS NULL
      AND new.source_manifest_count IS NULL
      AND NOT EXISTS (SELECT 1 FROM kg_evidence ev WHERE ev.chunk_id = old.id)
      AND NOT EXISTS (
          SELECT 1 FROM kg_claim_observations observation
          WHERE observation.chunk_id = old.id
      )
      AND NOT EXISTS (
          SELECT 1 FROM kg_claim_extraction_outcomes outcome
          WHERE outcome.chunk_id = old.id
      )
 )
BEGIN
    SELECT RAISE(ABORT, 'published chunk source manifest header is immutable');
END;
