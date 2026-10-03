-- v53: bind every live Phase-1 publication/cache marker to the exact
-- credential-free LLM producer and effective request policy.
--
-- Existing rows remain NULL deliberately. Prompt/validator identity alone
-- cannot prove which model, endpoint, thinking body, transport, or retry
-- policy produced them, so they are history rather than current cache
-- authority until a successful live re-extraction replaces them.
CREATE TABLE IF NOT EXISTS phase1_generations (
    generation_key TEXT PRIMARY KEY,
    extraction_cache_key TEXT NOT NULL,
    producer_identity_sha256 TEXT NOT NULL,
    identity_exact BOOLEAN NOT NULL CHECK (identity_exact IN (0, 1)),
    reuse_scope TEXT NOT NULL CHECK (
        reuse_scope IN ('durable', 'process_instance')
    ),
    binding_json TEXT NOT NULL CHECK (json_valid(binding_json)),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE processed_chunks ADD COLUMN phase1_generation_key TEXT
    REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT;
ALTER TABLE chunk_extraction_attempts ADD COLUMN phase1_generation_key TEXT
    REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT;
ALTER TABLE kg_claim_extraction_outcomes ADD COLUMN phase1_generation_key TEXT
    REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT;
ALTER TABLE kg_claim_observations ADD COLUMN phase1_generation_key TEXT
    REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT;
ALTER TABLE behavioral_markers ADD COLUMN phase1_generation_key TEXT
    REFERENCES phase1_generations(generation_key) ON DELETE RESTRICT;

CREATE INDEX IF NOT EXISTS idx_processed_chunks_phase1_generation
    ON processed_chunks(phase1_generation_key, chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_attempts_phase1_generation
    ON chunk_extraction_attempts(phase1_generation_key, chunk_id);
CREATE INDEX IF NOT EXISTS idx_claim_outcomes_phase1_generation
    ON kg_claim_extraction_outcomes(phase1_generation_key, chunk_id);
CREATE INDEX IF NOT EXISTS idx_claim_observations_phase1_generation
    ON kg_claim_observations(phase1_generation_key, chunk_id);
CREATE INDEX IF NOT EXISTS idx_behavioral_markers_phase1_generation
    ON behavioral_markers(phase1_generation_key, chunk_id);
