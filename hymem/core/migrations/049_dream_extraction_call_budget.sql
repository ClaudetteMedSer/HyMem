-- v49: make recursive Phase-1 extraction cost observable and bounded per cycle.
--
-- ``dream_budget`` counts chunks, but schema-v48 extraction may make multiple
-- logical completions, and each completion may make multiple HTTP attempts, to
-- certify one chunk atomically. These columns preserve both counts and
-- distinguish a soft provider-attempt ceiling that actually left actionable
-- extraction pending from a cycle that merely landed exactly on its configured
-- threshold. Digest/profile/fact calls are excluded.

ALTER TABLE dream_runs ADD COLUMN
    chunk_extraction_completion_calls INTEGER NOT NULL DEFAULT 0
    CHECK(chunk_extraction_completion_calls >= 0);

ALTER TABLE dream_runs ADD COLUMN
    chunk_extraction_provider_attempts INTEGER NOT NULL DEFAULT 0
    CHECK(chunk_extraction_provider_attempts >= 0);

ALTER TABLE dream_runs ADD COLUMN
    extraction_provider_attempt_budget_exhausted INTEGER NOT NULL DEFAULT 0
    CHECK(extraction_provider_attempt_budget_exhausted IN (0, 1));
