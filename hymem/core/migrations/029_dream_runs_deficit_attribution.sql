-- v29: per-dream deficit attribution on dream_runs (renumbered from 027).
--
-- The 2026-08-07 gate read decomposed the reuse deficit into three channels:
-- level-0 re-keys (arrivals landing in clusters), the root's conditional
-- facts_hash re-key, and leftover displacement (aggregation_digest_max_leaves).
-- `aggregate.built` logs the first and third per dream, but honcho-run dreams'
-- stderr goes to the gateway pipe — log-only attribution was unreadable on the
-- main path, so the amplification model (rebuilt = A*level0_missed + root_term
-- + leaf_term) could not be checked per dream.
--
-- NULL is the honest default, deliberately unlike v22/v25's counters:
-- these columns are only meaningful for a dream that actually ran aggregation,
-- and level0_missed=0 + leaf_changed=0 is precisely the fixed-point signature
-- (rebuilt=0, 100% reuse) — backfilling zero across pre-v27 history would
-- manufacture ~1150 counterfeit fixed points into the exact table the
-- analysis reads. Queries must treat NULL as "unattributed".
ALTER TABLE dream_runs ADD COLUMN aggregation_level0_missed INTEGER;
ALTER TABLE dream_runs ADD COLUMN aggregation_leaf_changed INTEGER;
