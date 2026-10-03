-- v2: numeric/temporal value columns on kg_evidence.
ALTER TABLE kg_evidence ADD COLUMN value_text TEXT;
ALTER TABLE kg_evidence ADD COLUMN value_numeric REAL;
ALTER TABLE kg_evidence ADD COLUMN value_unit TEXT;
ALTER TABLE kg_evidence ADD COLUMN temporal_scope TEXT;
