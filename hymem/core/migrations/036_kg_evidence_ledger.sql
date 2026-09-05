-- v36: make knowledge-graph evidence counters auditable and idempotent.
--
-- Existing extraction rows receive a conservative weight of one. Historical
-- speaker configuration is not recoverable, so guessing a larger role weight
-- would manufacture trust. Counter amounts above the old row-backed totals are
-- retained as `legacy_unattributed` signals with counts_toward_confidence=0:
-- operators can inspect them, but retries and old soft-decay loops no longer
-- inflate live confidence. Derived edges are intentionally untouched because
-- their 1/0 counters are computed placeholders, not observed evidence.

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    start_message_id INTEGER NOT NULL,
    end_message_id INTEGER NOT NULL,
    salience_reason TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kg_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
    surface_subject TEXT,
    surface_object TEXT,
    value_text TEXT,
    value_numeric REAL,
    value_unit TEXT,
    temporal_scope TEXT,
    source_role TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Latest-schema startup may have created the v40 lifecycle tables around an
-- old evidence shape. Drop their column-sensitive triggers before the first
-- ALTER below; migration 040 reinstalls the complete guard set.
DROP TRIGGER IF EXISTS kg_edge_lifecycle_insert_guard;
DROP TRIGGER IF EXISTS kg_edge_lifecycle_update_guard;
DROP TRIGGER IF EXISTS kg_edge_lifecycle_delete_guard;
DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_insert_guard;
DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_update_guard;
DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_delete_guard;

-- A few early development databases omitted columns which v1 schema had but
-- no numbered migration owned. These ALTERs make v36 self-healing for those
-- stores too; duplicate-column errors are ignored by the migration runner.
ALTER TABLE kg_evidence ADD COLUMN surface_subject TEXT;
ALTER TABLE kg_evidence ADD COLUMN surface_object TEXT;
ALTER TABLE kg_evidence ADD COLUMN value_text TEXT;
ALTER TABLE kg_evidence ADD COLUMN value_numeric REAL;
ALTER TABLE kg_evidence ADD COLUMN value_unit TEXT;
ALTER TABLE kg_evidence ADD COLUMN temporal_scope TEXT;
ALTER TABLE kg_evidence ADD COLUMN source_role TEXT;
ALTER TABLE kg_evidence ADD COLUMN extracted_at TIMESTAMP;
ALTER TABLE knowledge_graph ADD COLUMN derived BOOLEAN NOT NULL DEFAULT 0;

ALTER TABLE kg_evidence ADD COLUMN evidence_kind TEXT NOT NULL DEFAULT 'extraction';
ALTER TABLE kg_evidence ADD COLUMN evidence_weight INTEGER NOT NULL DEFAULT 1 CHECK (evidence_weight >= 1);
ALTER TABLE kg_evidence ADD COLUMN weight_source TEXT NOT NULL DEFAULT 'legacy_default';
ALTER TABLE kg_evidence ADD COLUMN extraction_prompt_version TEXT;

CREATE TABLE IF NOT EXISTS kg_evidence_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
    signal_key TEXT NOT NULL,
    signal_kind TEXT NOT NULL,
    polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
    evidence_weight INTEGER NOT NULL CHECK (evidence_weight >= 1),
    counts_toward_confidence BOOLEAN NOT NULL DEFAULT 1
        CHECK (counts_toward_confidence IN (0, 1)),
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(edge_id, signal_kind, signal_key)
);

CREATE INDEX IF NOT EXISTS idx_evidence_signals_edge
    ON kg_evidence_signals(edge_id);

INSERT OR IGNORE INTO kg_evidence_signals(
    edge_id, signal_key, signal_kind, polarity, evidence_weight,
    counts_toward_confidence, details
)
SELECT kg.id, 'legacy:positive', 'legacy_unattributed', 1,
       kg.pos_evidence - COALESCE((
           SELECT SUM(ev.evidence_weight)
           FROM kg_evidence ev
           WHERE ev.edge_id = kg.id AND ev.polarity = 1
       ), 0),
       0, 'pre-v36 counter delta retained for audit and excluded from confidence'
FROM knowledge_graph kg
WHERE kg.derived = 0
  AND kg.pos_evidence > COALESCE((
      SELECT SUM(ev.evidence_weight)
      FROM kg_evidence ev
      WHERE ev.edge_id = kg.id AND ev.polarity = 1
  ), 0);

INSERT OR IGNORE INTO kg_evidence_signals(
    edge_id, signal_key, signal_kind, polarity, evidence_weight,
    counts_toward_confidence, details
)
SELECT kg.id, 'legacy:negative', 'legacy_unattributed', -1,
       kg.neg_evidence - COALESCE((
           SELECT SUM(ev.evidence_weight)
           FROM kg_evidence ev
           WHERE ev.edge_id = kg.id AND ev.polarity = -1
       ), 0),
       0, 'pre-v36 counter delta retained for audit and excluded from confidence'
FROM knowledge_graph kg
WHERE kg.derived = 0
  AND kg.neg_evidence > COALESCE((
      SELECT SUM(ev.evidence_weight)
      FROM kg_evidence ev
      WHERE ev.edge_id = kg.id AND ev.polarity = -1
  ), 0);

-- Old uniqueness was a table constraint on (edge_id, chunk_id, polarity).
-- SQLite implements it as an autoindex which cannot be dropped or superseded,
-- so ALTER plus a new index would still reject two same-polarity evidence
-- kinds. Rebuild the table to remove that legacy constraint. If prompt changes
-- left more than one interpretation for a source and evidence kind, the last
-- inserted row wins deterministically during the copy.
PRAGMA foreign_keys = OFF;

SAVEPOINT kg_evidence_v36_rebuild;

-- Latest schema bootstraps may already expose the v40 lifecycle tables. Their
-- triggers mention v40 evidence columns, so temporarily remove the triggers
-- while this historical rebuild replays; migration 040 reinstalls them.
DROP TRIGGER IF EXISTS kg_edge_lifecycle_insert_guard;
DROP TRIGGER IF EXISTS kg_edge_lifecycle_update_guard;
DROP TRIGGER IF EXISTS kg_edge_lifecycle_delete_guard;
DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_insert_guard;
DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_update_guard;
DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_delete_guard;

DROP TABLE IF EXISTS kg_evidence_new;

-- A fresh latest-schema database replays numbered migrations from v1. v40's
-- manifest header guard references kg_evidence, so temporarily remove it while
-- this historical rebuild swaps that table; migration 040 reinstalls it.
DROP TRIGGER IF EXISTS chunk_source_manifest_header_update_guard;

CREATE TABLE kg_evidence_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
    surface_subject TEXT,
    surface_object TEXT,
    value_text TEXT,
    value_numeric REAL,
    value_unit TEXT,
    temporal_scope TEXT,
    source_role TEXT,
    evidence_kind TEXT NOT NULL DEFAULT 'extraction',
    evidence_weight INTEGER NOT NULL DEFAULT 1 CHECK (evidence_weight >= 1),
    weight_source TEXT NOT NULL DEFAULT 'legacy_default',
    extraction_prompt_version TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(edge_id, chunk_id, evidence_kind)
);

INSERT INTO kg_evidence_new(
    id, edge_id, chunk_id, polarity, surface_subject, surface_object,
    value_text, value_numeric, value_unit, temporal_scope, source_role,
    evidence_kind, evidence_weight, weight_source,
    extraction_prompt_version, extracted_at
)
SELECT
    id, edge_id, chunk_id, polarity, surface_subject, surface_object,
    value_text, value_numeric, value_unit, temporal_scope, source_role,
    evidence_kind, evidence_weight, weight_source,
    extraction_prompt_version, extracted_at
FROM kg_evidence
WHERE id IN (
    SELECT MAX(id)
    FROM kg_evidence
    GROUP BY edge_id, chunk_id, evidence_kind
)
ORDER BY id;

DROP TABLE kg_evidence;

ALTER TABLE kg_evidence_new RENAME TO kg_evidence;

CREATE INDEX IF NOT EXISTS idx_evidence_edge ON kg_evidence(edge_id);
CREATE INDEX IF NOT EXISTS idx_evidence_chunk ON kg_evidence(chunk_id);

RELEASE SAVEPOINT kg_evidence_v36_rebuild;

PRAGMA foreign_keys = ON;

UPDATE knowledge_graph
SET pos_evidence = COALESCE((
        SELECT SUM(ev.evidence_weight)
        FROM kg_evidence ev
        WHERE ev.edge_id = knowledge_graph.id AND ev.polarity = 1
    ), 0) + COALESCE((
        SELECT SUM(sig.evidence_weight)
        FROM kg_evidence_signals sig
        WHERE sig.edge_id = knowledge_graph.id AND sig.polarity = 1
          AND sig.counts_toward_confidence = 1
    ), 0),
    neg_evidence = COALESCE((
        SELECT SUM(ev.evidence_weight)
        FROM kg_evidence ev
        WHERE ev.edge_id = knowledge_graph.id AND ev.polarity = -1
    ), 0) + COALESCE((
        SELECT SUM(sig.evidence_weight)
        FROM kg_evidence_signals sig
        WHERE sig.edge_id = knowledge_graph.id AND sig.polarity = -1
          AND sig.counts_toward_confidence = 1
    ), 0)
WHERE derived = 0;

-- Chunk retention can delete evidence through an ON DELETE CASCADE without
-- calling Python. Keep the cache correct for that deletion path too. Inserts
-- and updates are performed through evidence.py, which handles source
-- idempotence and then reconciles the complete edge total.
CREATE TRIGGER IF NOT EXISTS kg_evidence_count_after_delete
AFTER DELETE ON kg_evidence
BEGIN
    UPDATE knowledge_graph
    SET pos_evidence = MAX(0, pos_evidence - old.evidence_weight)
    WHERE id = old.edge_id AND derived = 0 AND old.polarity = 1;
    UPDATE knowledge_graph
    SET neg_evidence = MAX(0, neg_evidence - old.evidence_weight)
    WHERE id = old.edge_id AND derived = 0 AND old.polarity = -1;
END;
