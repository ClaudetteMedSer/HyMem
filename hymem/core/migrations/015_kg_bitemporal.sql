-- v15: bi-temporal validity interval on knowledge_graph edges.
--
-- The table already carries TRANSACTION time (first_seen / last_seen /
-- last_reinforced — when HyMem *learned* a fact). What was missing is VALID
-- time — when the fact was true *in the world*. Supersession today is a status
-- flip (active -> retracted/stale) that erases the "what was true as of date X"
-- axis; these two columns restore it without changing the existing decay logic.
--
--   valid_at   = world date the fact became true. Populated during dreaming
--                from the earliest POSITIVE-evidence source message created_at
--                (the same host-supplied timestamp the recency-dating lever
--                stamps onto message_hits); NULL until the stamping pass runs.
--   invalid_at = world date the fact stopped being true. Stamped at the moment
--                an edge is superseded (status -> retracted/stale): the newest
--                contradicting-evidence world date, falling back to the flip
--                time. NULL = still valid.
--
-- schema.sql adds the same two columns to the knowledge_graph CREATE TABLE so a
-- fresh DB has them; the ALTERs below add them to existing (<= v14) DBs (the
-- duplicate-column error on a fresh DB is tolerated by the runner). The index
-- lives in THIS migration ONLY: a standalone CREATE INDEX in schema.sql would
-- reference valid_at before the ALTER runs on an old DB and crash with
-- "no such column" (same constraint as 010_procedure_status / 012).
ALTER TABLE knowledge_graph ADD COLUMN valid_at TIMESTAMP;
ALTER TABLE knowledge_graph ADD COLUMN invalid_at TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_kg_validity
    ON knowledge_graph(subject_canonical, predicate, valid_at);

-- Backfill for rows that predate this migration. valid_at falls back to
-- first_seen (the best transaction-time proxy for when the fact entered);
-- invalid_at is seeded from last_seen for already-superseded edges so the
-- interval is closed for historical tombstones. Both are approximations for
-- pre-v15 rows — the dreaming stamping pass refines valid_at from real source
-- dates on the next cycle. Gated to NULLs so re-running is a no-op.
UPDATE knowledge_graph SET valid_at = first_seen WHERE valid_at IS NULL;
UPDATE knowledge_graph
    SET invalid_at = last_seen
    WHERE invalid_at IS NULL AND status IN ('stale', 'retracted');
