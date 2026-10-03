-- v48: retain safe, exact diagnostics for the latest failed Phase-1 attempt.
--
-- The old ledger recorded only an aggregate attempt count. A provider reply
-- with a bad predicate, missing source id, false completion certificate, or
-- structural truncation therefore looked identical after restart. These
-- columns contain only allowlisted reason and field-level codes assembled by
-- HyMem; raw model output and values are never stored.
--
-- The retry ledger is intentionally absent from portable exports: it is local
-- operational staging, is cleared after success, and resetting it on import is
-- safer than transporting a source host's quarantine decision.
ALTER TABLE chunk_extraction_attempts ADD COLUMN last_failure_reason TEXT;
ALTER TABLE chunk_extraction_attempts ADD COLUMN last_failure_details TEXT NOT NULL DEFAULT '[]';
