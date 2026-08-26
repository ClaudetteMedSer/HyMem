-- v23: `always_on` Rules as a first-class node type (Idea B).
--
-- HyMem's "always loaded" layer was scattered across the MEMORY.md / USER.md
-- auto-sections, profile_entries, and the closed-vocabulary user_profile slots.
-- None is a clean standing imperative ("always run the tests before pushing",
-- "never suggest Docker") — those are RULES, not facts/preferences, and they
-- competed for the capped insight/profile budgets. This table gives the
-- imperative subset a dedicated home injected into every augment() context call
-- (BrainDB's always_on rule node), consumed by hymem/rules.py::load_rules and
-- surfaced in ctx.rules.
--
--   scope='always_on'  → injected unconditionally into every call.
--   scope='contextual' → injected only when a trigger_entities member overlaps
--                        the call's matched_entities (canonicalized on both sides).
--
-- Rows are bi-temporal like knowledge_graph (v15) and user_profile (v18): a
-- contradicting rule closes the prior's validity interval (invalid_at +
-- status='retracted') rather than overwriting, so rule history is auditable.
-- `text` is UNIQUE so re-asserting an identical rule UPSERT-reinforces
-- (pos_evidence++) instead of duplicating.
--
-- Constraint (additional_planning.md §0): rules are NOT fed into the RAPTOR
-- digest's _anchor_facts block — that content hashes into the root digest cache
-- id, and coupling rule edits to digest regeneration is undesired. Rules stay a
-- parallel augment tier only.
--
-- Forward-only and idempotent: re-applying against a schema.sql DB that already
-- has these objects is a tolerated no-op (CREATE ... IF NOT EXISTS).
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL UNIQUE,
    scope TEXT NOT NULL DEFAULT 'always_on'
        CHECK (scope IN ('always_on', 'contextual')),
    trigger_entities TEXT NOT NULL DEFAULT '[]',   -- JSON list, for scope='contextual'
    source TEXT NOT NULL DEFAULT 'user'
        CHECK (source IN ('user', 'agent_inferred')),
    pos_evidence INTEGER NOT NULL DEFAULT 1,
    neg_evidence INTEGER NOT NULL DEFAULT 0,
    valid_at TIMESTAMP,                            -- bi-temporal, like knowledge_graph / user_profile
    invalid_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'retracted')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(scope, status, invalid_at);
