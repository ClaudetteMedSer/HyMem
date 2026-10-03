-- v10: procedure staleness signal. status='stale' procedures are skipped by
-- _procedure_search; mark_procedure_stale also downgrades confidence. ALTER
-- cannot carry the CHECK that schema.sql's CREATE TABLE has — the values are
-- enforced by the application, not the column constraint, on migrated DBs.
ALTER TABLE procedures ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
CREATE INDEX IF NOT EXISTS idx_procedures_status ON procedures(status);
