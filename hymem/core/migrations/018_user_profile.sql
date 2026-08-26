-- v18: typed user-profile slots (Stage 1 / P4).
--
-- The knowledge graph's 18-predicate vocabulary is tech-domain, so durable
-- personal identity facts ("user is a bedrijfsarts in Amsterdam named Atta")
-- never become edges and the digest stays identity-thin. This table holds them
-- under a CLOSED slot vocabulary enforced by the CHECK below — the
-- schema-constrained, safe version of open-ended incidental extraction: the
-- LLM can never invent a slot, neither past validation nor past the database.
--
-- slot_key parameterizes a slot ('relationship' is keyed by the other person:
-- slot='relationship', slot_key='anna', value='sister'); NULL for unkeyed
-- slots. Rows are bi-temporal like knowledge_graph (v15 P2 semantics):
--   valid_at   = world date the fact became true (the evidence message's
--                created_at, falling back to insert time).
--   invalid_at = world date it was superseded by a conflicting value on the
--                same (slot, slot_key) — the new evidence's world date,
--                falling back to the flip time. NULL = still valid.
-- Single-valued slots (name, role, employer, location, age_birthday) and
-- 'relationship' per slot_key supersede on conflict; the other multi-valued
-- slots accumulate. Forward-only and idempotent: re-applying against a
-- schema.sql DB that already has these objects is a tolerated no-op.
CREATE TABLE IF NOT EXISTS user_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot TEXT NOT NULL CHECK (slot IN (
        'role','name','employer','location','language','relationship',
        'possession','age_birthday','health_condition','recurring_activity'
    )),
    slot_key TEXT,
    value TEXT NOT NULL,
    evidence_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 1.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    valid_at TIMESTAMP,
    invalid_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_profile_active
    ON user_profile(slot, slot_key, invalid_at);
