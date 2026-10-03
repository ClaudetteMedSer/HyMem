-- v22: RAPTOR reuse-instability attribution columns (cron report 2026-07-12).
--
-- The fusion-reuse watch (dream_runs.aggregation_nodes_built/_reused, v20)
-- could show THAT a dream re-fused half the tree but not WHY — the July hunt
-- had to reverse-engineer causes from run timestamps. Three columns close
-- that:
--
--   aggregation_fusion_failures — fusions attempted and lost this run (LLM
--     transport error, non-JSON reply, empty result). Failures retry on every
--     subsequent dream until healed, and each fail→heal transition costs one
--     low-reuse run; a low-reuse row with failures > 0 is an LLM-flakiness
--     event, not membership churn.
--   aggregation_input_episodes — size of the snapshot the clusterer read. A
--     built-count drift (the report's 41-vs-46 flicker) is instantly
--     attributable to input-set changes vs. tree reshaping.
--   aggregation_blocking — candidate generator used: 'knn' or an
--     'exact:<reason>' fallback. Two trigger paths with different
--     environments (one missing sqlite-vec) cluster DIFFERENTLY and re-key
--     cached fusions on every alternation; a mode flip between consecutive
--     rows is that smoking gun.
--
-- Forward-only and idempotent: duplicate-column errors are tolerated by the
-- migration runner against schema.sql databases that already carry these.
ALTER TABLE dream_runs ADD COLUMN aggregation_fusion_failures INTEGER NOT NULL DEFAULT 0;
ALTER TABLE dream_runs ADD COLUMN aggregation_input_episodes INTEGER NOT NULL DEFAULT 0;
ALTER TABLE dream_runs ADD COLUMN aggregation_blocking TEXT NOT NULL DEFAULT '';
