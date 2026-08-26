-- v30: durable leaf-set watermark for the leftover-displacement channel.
--
-- `aggregation_leaf_changed` (v29) is the third term of the deficit model
-- rebuilt ~ A*level0_missed + root_term + leaf_term. It was derived from
-- `_LAST_LEAF_SET`, a module global in hymem/dreaming/aggregate.py, so it was
-- readable only on the SECOND and later aggregation dreams within one process
-- lifetime: the first dream of a process has no predecessor in memory and
-- correctly writes NULL rather than a counterfeit 0.
--
-- On the box the honcho starts a fresh process per dream, so that condition
-- held almost always: the 2026-08-08 read found 175 of 187 rows unreadable.
-- The channel was therefore not under-sampled, it was structurally
-- unobservable on the main path, and a longer watch would have added rows
-- without adding leaf evidence. That also means any amplification bound fitted
-- today would silently absorb the leaf term into A.
--
-- Moving the watermark into the store makes the comparison "changed since the
-- last dream that PERSISTED aggregation", which is what the model means and
-- what survives a process restart. A fingerprint is stored rather than the id
-- list: the comparison only ever tests equality, and the leaf set is capped by
-- aggregation_digest_max_leaves but still large.
--
-- Single row by construction (the CHECK), written inside the same transaction
-- that persists the nodes, so the watermark advances if and only if the dream
-- that consumed that leaf set actually landed. NULL stays the honest first
-- reading -- now once per STORE rather than once per process.
CREATE TABLE IF NOT EXISTS aggregation_leaf_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    fingerprint TEXT NOT NULL,
    n_leaves INTEGER NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
