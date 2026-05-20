-- v4: mark inference-derived knowledge_graph edges.
ALTER TABLE knowledge_graph ADD COLUMN derived BOOLEAN NOT NULL DEFAULT 0;
