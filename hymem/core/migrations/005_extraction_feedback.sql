-- v5: retraction audit table (historical name retained for compatibility).
CREATE TABLE IF NOT EXISTS extraction_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
    chunk_text_snippet TEXT NOT NULL,
    extracted_subject TEXT NOT NULL,
    extracted_predicate TEXT NOT NULL,
    extracted_object TEXT NOT NULL,
    feedback_type TEXT NOT NULL DEFAULT 'retracted',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON extraction_feedback(created_at);
