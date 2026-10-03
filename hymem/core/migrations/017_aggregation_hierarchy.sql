-- v17: RAPTOR hierarchy levels + root digest on aggregation_nodes.
-- level 0 = episode-cluster nodes (the existing layer; the only level the
-- query-time retrieval tier surfaces). level >= 1 = rollup nodes whose
-- member_episode_ids hold CHILD ids (lower-level node ids and/or pass-through
-- episode ids). is_root marks the single top-of-tree digest node — the standing
-- "what do you know about me" summary exposed via HyMem.digest(). Forward-only
-- and idempotent: duplicate-column errors on re-application are tolerated.
ALTER TABLE aggregation_nodes ADD COLUMN level INTEGER NOT NULL DEFAULT 0;
ALTER TABLE aggregation_nodes ADD COLUMN is_root INTEGER NOT NULL DEFAULT 0;
