CREATE TABLE IF NOT EXISTS kg_evidence_extraction_audit (
    evidence_id INTEGER NOT NULL REFERENCES kg_evidence(id) ON DELETE CASCADE,
    occurrence_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE RESTRICT,
    source_message_id INTEGER,
    source_session_id TEXT,
    source_coverage_chunk_id TEXT,
    source_coverage_version TEXT,
    PRIMARY KEY (evidence_id, occurrence_hash),
    FOREIGN KEY (source_message_id, source_coverage_chunk_id, source_coverage_version)
        REFERENCES message_retention_coverage(message_id, chunk_id, coverage_version) ON DELETE RESTRICT,
    CHECK (hymem_extraction_audit_valid(evidence_id, occurrence_hash, payload_json,
        chunk_id, source_message_id, source_session_id,
        source_coverage_chunk_id, source_coverage_version) IS 1)
);
CREATE INDEX IF NOT EXISTS idx_extraction_audit_chunk ON kg_evidence_extraction_audit(chunk_id);
CREATE INDEX IF NOT EXISTS idx_extraction_audit_coverage ON kg_evidence_extraction_audit(source_message_id, source_coverage_chunk_id, source_coverage_version);
CREATE TRIGGER IF NOT EXISTS extraction_audit_insert_guard
BEFORE INSERT ON kg_evidence_extraction_audit
WHEN hymem_evidence_mutation_authorized() IS NOT 1 OR hymem_evidence_history_authorized() IS NOT 1
BEGIN SELECT RAISE(ABORT, 'extraction audit requires history authority'); END;
CREATE TRIGGER IF NOT EXISTS extraction_audit_update_guard
BEFORE UPDATE ON kg_evidence_extraction_audit
BEGIN SELECT RAISE(ABORT, 'extraction audit is immutable'); END;
CREATE TRIGGER IF NOT EXISTS extraction_audit_delete_guard
BEFORE DELETE ON kg_evidence_extraction_audit
WHEN hymem_evidence_mutation_authorized() IS NOT 1 OR
    (hymem_evidence_history_authorized() IS NOT 1 AND hymem_evidence_destructive_authorized() IS NOT 1)
BEGIN SELECT RAISE(ABORT, 'extraction audit deletion requires history or destructive authority'); END;
