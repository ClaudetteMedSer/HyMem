-- v12: per-session digest marker. Records the prompt_version of the last
-- successful batched session digest (episodes + summary + procedures) so the
-- dream runner can skip the digest LLM call for sessions already processed
-- under the current prompt_version with no newly-extracted chunks.
ALTER TABLE sessions ADD COLUMN digested_prompt_version TEXT;
