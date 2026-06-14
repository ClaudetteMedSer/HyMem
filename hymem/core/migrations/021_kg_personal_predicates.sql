-- v21: admit the v9 personal-life predicates (owns, located_in, participates_in,
-- has_attribute) into the knowledge_graph.predicate CHECK. The KU-coverage probe
-- showed only ~14% of knowledge-update gold values minted as edges because these
-- classes — possessions, residence, activities, personal metrics — had no
-- expressible predicate; extraction (prompt_version v9) now emits them, and they
-- must pass the table's locked-vocabulary CHECK.
--
-- SQLite cannot ALTER a CHECK constraint in place, so the table is rebuilt. The
-- rebuild mirrors the schema.sql definition exactly (so a fresh DB and a migrated
-- DB converge), with foreign keys disabled during the swap and kg_evidence.edge_id
-- preserved by copying `id` explicitly. DROP IF EXISTS on the scratch table keeps
-- the common interrupted-rerun case clean.
PRAGMA foreign_keys=OFF;

-- Guarantee the base evidence columns exist before the copy. On a real DB they
-- already do, so each ALTER raises "duplicate column name" — an error the
-- migration runner tolerates as a no-op (existing data untouched). On a minimal
-- pre-existing DB that predates them, the ALTER materialises them with defaults
-- so the rebuild's column list resolves uniformly either way.
-- (pos_evidence/neg_evidence/last_reinforced are base-schema columns no migration
-- adds; derived comes from migration 004, which is skipped when a DB starts at a
-- version past it. valid_at/invalid_at are guaranteed here by migration 015.)
ALTER TABLE knowledge_graph ADD COLUMN pos_evidence INTEGER NOT NULL DEFAULT 0;
ALTER TABLE knowledge_graph ADD COLUMN neg_evidence INTEGER NOT NULL DEFAULT 0;
ALTER TABLE knowledge_graph ADD COLUMN last_reinforced TIMESTAMP;
ALTER TABLE knowledge_graph ADD COLUMN derived BOOLEAN NOT NULL DEFAULT 0;

DROP TABLE IF EXISTS knowledge_graph_new;

CREATE TABLE knowledge_graph_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_canonical TEXT NOT NULL,
    predicate TEXT NOT NULL CHECK (predicate IN (
        'uses','depends_on','prefers','rejects','avoids',
        'replaces','conflicts_with','deploys_to','part_of','equivalent_to',
        'implements','contains','configured_with','requires_version',
        'runs_on','connects_to','generates','tested_by',
        'owns','located_in','participates_in','has_attribute'
    )),
    object_canonical TEXT NOT NULL,
    pos_evidence INTEGER NOT NULL DEFAULT 0,
    neg_evidence INTEGER NOT NULL DEFAULT 0,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_reinforced TIMESTAMP,
    valid_at TIMESTAMP,
    invalid_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','stale','retracted')),
    derived BOOLEAN NOT NULL DEFAULT 0,
    UNIQUE(subject_canonical, predicate, object_canonical)
);

INSERT INTO knowledge_graph_new (
    id, subject_canonical, predicate, object_canonical,
    pos_evidence, neg_evidence, first_seen, last_seen, last_reinforced,
    valid_at, invalid_at, status, derived
)
SELECT id, subject_canonical, predicate, object_canonical,
       pos_evidence, neg_evidence, first_seen, last_seen, last_reinforced,
       valid_at, invalid_at, status, derived
FROM knowledge_graph;

DROP TABLE knowledge_graph;
ALTER TABLE knowledge_graph_new RENAME TO knowledge_graph;

-- Recreate every index the dropped table carried: schema.sql's four plus the
-- bi-temporal validity index introduced in migration 015.
CREATE INDEX IF NOT EXISTS idx_kg_subject ON knowledge_graph(subject_canonical);
CREATE INDEX IF NOT EXISTS idx_kg_object ON knowledge_graph(object_canonical);
CREATE INDEX IF NOT EXISTS idx_kg_predicate ON knowledge_graph(predicate);
CREATE INDEX IF NOT EXISTS idx_kg_status ON knowledge_graph(status);
CREATE INDEX IF NOT EXISTS idx_kg_validity ON knowledge_graph(subject_canonical, predicate, valid_at);

PRAGMA foreign_keys=ON;
