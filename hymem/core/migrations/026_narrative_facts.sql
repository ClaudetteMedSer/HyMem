-- v26: narrative facts — the E1 middle-granularity tier (2026-08-02).
--
-- Dream-time extraction of self-contained narrative facts ("Atta moved the
-- MedFlow deploy to fly.io"), stored append-only and served as an ADDITIVE
-- retrieval tier plus the lead evidence block in ask(). Sits between the
-- atomic knowledge-graph triple and the abstract episode summary — the
-- granularity gap the Campaign E / Hindsight analysis identified.
--
-- Gated build: G-F1 FAILED twice on deepseek-v4-flash (faithfulness
-- 0.55-0.76 vs a required 0.90) and this migration existed only as a spec.
-- The pre-registered revival gate G-F1b PASSED 2026-08-02 on gpt-oss-120b
-- (123/123 strict on the healed full-source sample), which is what re-opened
-- Step 4. The extraction prompt that cleared the gate ships VERBATIM as
-- FACTS_SYSTEM (prompt_version 'facts.v2'); see benchmarks/fact_probe.py for
-- the instrument and hymem/dreaming/facts.py for the build.
--
-- Design constraints carried from the gate:
--   * text is IMMUTABLE — append-only inserts, no UPDATE path. A prompt bump
--     extracts FORWARD ONLY (new ranges under the new tag); covered ranges are
--     never re-extracted, so a superseded prompt's facts remain attributable.
--   * invalid_at is the ONLY mutable field (E6 supersession closes it later);
--     retrieval filters on invalid_at IS NULL, audit keeps the row.
--   * fact_date holds EXPLICIT dates only (the conversation wrote YYYY-MM-DD);
--     relative references stay NULL — resolving them is E4's job, and stamping
--     the session date was a proven invention amplifier (G-F1's date lesson).
CREATE TABLE IF NOT EXISTS narrative_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_message_id INTEGER NOT NULL,
    end_message_id INTEGER NOT NULL,
    text TEXT NOT NULL,                    -- self-contained narrative, IMMUTABLE
    fact_date TEXT,                        -- ISO or NULL (explicit dates only, relatives = E4)
    entities TEXT NOT NULL DEFAULT '[]',   -- JSON array of canonical names
    prompt_version TEXT NOT NULL,          -- 'facts.v2' provenance tag
    valid_at TEXT,                         -- bi-temporal, mirrors knowledge_graph
    invalid_at TEXT,                       -- the ONLY mutable field (E6 closes it)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (session_id, start_message_id, text)
);
CREATE INDEX IF NOT EXISTS idx_narrative_facts_session
    ON narrative_facts(session_id);

-- FTS over fact text. content_rowid='id' on an INTEGER PRIMARY KEY (a rowid
-- alias), so — like messages_fts and vec_edges, and UNLIKE the TEXT-PK
-- episodes/chunks shadows — this index is VACUUM-stable and needs no
-- resync_rowid_shadows coverage. The update trigger exists for drift-proofing
-- only; the sanctioned UPDATE path (invalid_at) never touches text.
CREATE VIRTUAL TABLE IF NOT EXISTS narrative_facts_fts USING fts5(
    text,
    content='narrative_facts',
    content_rowid='id',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS narrative_facts_fts_insert AFTER INSERT ON narrative_facts BEGIN
    INSERT INTO narrative_facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS narrative_facts_fts_delete AFTER DELETE ON narrative_facts BEGIN
    INSERT INTO narrative_facts_fts(narrative_facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS narrative_facts_fts_update AFTER UPDATE ON narrative_facts BEGIN
    INSERT INTO narrative_facts_fts(narrative_facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO narrative_facts_fts(rowid, text) VALUES (new.id, new.text);
END;

-- JSON mirror for fact vectors (the vec_facts vec0 table is created at runtime
-- by ensure_vec_table when sqlite-vec is present, like the other vec_* tables).
-- text_hash points into embedding_cache; fact text is immutable, so a row here
-- means "embedded" — there is no staleness path.
CREATE TABLE IF NOT EXISTS narrative_fact_embeddings (
    fact_id INTEGER PRIMARY KEY REFERENCES narrative_facts(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Facts watermark, mirroring v24's digested_message_id but with its OWN column
-- so facts and digest coverage advance independently. NULL = no coverage
-- recorded; the next dream extracts from the start of the session.
ALTER TABLE sessions ADD COLUMN facts_message_id INTEGER;

-- Facts attribution on dream_runs, mirroring v25's digest counters: a stalled
-- or failing extractor must be a one-line read, not a join nobody runs.
ALTER TABLE dream_runs ADD COLUMN facts_extracted INTEGER NOT NULL DEFAULT 0;
ALTER TABLE dream_runs ADD COLUMN fact_failures INTEGER NOT NULL DEFAULT 0;
