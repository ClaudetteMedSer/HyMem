-- v19: profile.v2 reset (Stage 1 / P4 hardening).
--
-- The profile.v1 prompt FAILED the on-box hand-scored precision gate at ~8%
-- (8/98 rows correct — see benchmarks/raptor_digest_plan.md Stage 1): it
-- extracted facts about repositories, patients, and GitHub orgs as if they
-- were facts about the user. Two consequences land here:
--
-- 1. sessions.profile_prompt_version — a per-session stamp of the profile
--    prompt version last extracted, mirroring digested_prompt_version (v12).
--    The profile call previously shared the digest skip-guard, so bumping
--    PROFILE_PROMPT_VERSION alone could never re-extract an already-digested
--    session; with its own stamp the runner re-runs extraction exactly when
--    the prompt version changes (or new chunk work arrives).
--
-- 2. DELETE FROM user_profile — purge every profile.v1 row. They failed the
--    precision gate, they are fully regenerable by re-dreaming under the
--    fixed prompt, and schema v18 was never released so no third-party data
--    exists. On a fresh schema.sql database the table is empty and this is a
--    no-op.
--
-- Forward-only and idempotent: re-applying against a schema.sql DB that
-- already has the column raises "duplicate column name", which the migration
-- runner tolerates (the digested_prompt_version v12 precedent).
ALTER TABLE sessions ADD COLUMN profile_prompt_version TEXT;

DELETE FROM user_profile;
