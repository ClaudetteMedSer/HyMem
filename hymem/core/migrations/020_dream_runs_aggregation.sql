-- v20: per-cycle RAPTOR aggregation counters in dream_runs.
--
-- build_aggregation_nodes computes how many fusions it rebuilt vs. served from
-- the content-hash cache, but until now that split only reached a log.info()
-- line — and the MCP server runs without logging.basicConfig(INFO), so it was
-- silently dropped. The reused count is exactly the dream-cost signal the
-- RAPTOR flip criteria watches (benchmarks/raptor_digest_plan.md Stage 3c:
-- "steady-state should be near-full reuse"), so it belongs in the durable
-- per-cycle row, not a scraped log.
--
-- aggregation_nodes_built was likewise computed into the in-memory DreamReport
-- but never persisted; both columns land here so a week of cycles accrues the
-- built/reused dataset automatically. Forward-only and idempotent: re-applying
-- against a schema.sql DB that already has these columns raises "duplicate
-- column name", which the migration runner tolerates.
ALTER TABLE dream_runs ADD COLUMN aggregation_nodes_built INTEGER NOT NULL DEFAULT 0;
ALTER TABLE dream_runs ADD COLUMN aggregation_nodes_reused INTEGER NOT NULL DEFAULT 0;
