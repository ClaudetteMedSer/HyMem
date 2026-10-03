-- v46: authoritative narrative-fact extraction, exact source manifests, and
-- auditable lifecycle.  Existing v26 facts intentionally remain unproven;
-- numeric message ranges are compatibility metadata and are never backfilled
-- into occurrence provenance.

ALTER TABLE sessions ADD COLUMN facts_cursor_message_id INTEGER;
ALTER TABLE sessions ADD COLUMN facts_cursor_partial_message_id INTEGER;
ALTER TABLE sessions ADD COLUMN facts_cursor_offset INTEGER NOT NULL DEFAULT 0
    CHECK (facts_cursor_offset >= 0);
ALTER TABLE sessions ADD COLUMN facts_cursor_prompt_version TEXT;
ALTER TABLE sessions ADD COLUMN facts_retry_count INTEGER NOT NULL DEFAULT 0
    CHECK (facts_retry_count >= 0);
ALTER TABLE sessions ADD COLUMN facts_retry_config_version TEXT;
ALTER TABLE sessions ADD COLUMN facts_quarantined BOOLEAN NOT NULL DEFAULT 0
    CHECK (facts_quarantined IN (0, 1));

ALTER TABLE narrative_facts ADD COLUMN source_outcome_key TEXT
    REFERENCES fact_extraction_outcomes(slice_key) ON DELETE RESTRICT;
ALTER TABLE narrative_facts ADD COLUMN fact_key TEXT;
ALTER TABLE narrative_facts ADD COLUMN current_generation INTEGER;
ALTER TABLE narrative_facts ADD COLUMN lifecycle_status TEXT NOT NULL
    DEFAULT 'legacy_unproven'
    CHECK (lifecycle_status IN ('active','retracted','legacy_unproven'));

CREATE TABLE IF NOT EXISTS fact_extraction_outcomes (
    slice_key TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    prompt_version TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    cursor_before_message_id INTEGER,
    cursor_before_partial_message_id INTEGER,
    cursor_before_offset INTEGER NOT NULL DEFAULT 0
        CHECK (cursor_before_offset >= 0),
    cursor_after_message_id INTEGER,
    cursor_after_partial_message_id INTEGER,
    cursor_after_offset INTEGER NOT NULL DEFAULT 0
        CHECK (cursor_after_offset >= 0),
    generation INTEGER NOT NULL CHECK (generation >= 1),
    outcome_status TEXT NOT NULL CHECK (outcome_status IN ('success','empty')),
    result_hash TEXT NOT NULL,
    source_manifest_version TEXT,
    source_manifest_count INTEGER NOT NULL DEFAULT 0
        CHECK (source_manifest_count >= 0),
    source_manifest_hash TEXT,
    source_manifest_complete BOOLEAN NOT NULL DEFAULT 0
        CHECK (source_manifest_complete IN (0, 1)),
    succeeded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fact_outcome_session
    ON fact_extraction_outcomes(session_id, cursor_after_message_id);
CREATE INDEX IF NOT EXISTS idx_fact_outcome_before_cursor
    ON fact_extraction_outcomes(
        session_id, cursor_before_message_id,
        cursor_before_partial_message_id, cursor_before_offset
    );
CREATE INDEX IF NOT EXISTS idx_fact_outcome_after_cursor
    ON fact_extraction_outcomes(
        session_id, cursor_after_message_id,
        cursor_after_partial_message_id, cursor_after_offset
    );
CREATE INDEX IF NOT EXISTS idx_fact_outcome_chain_order
    ON fact_extraction_outcomes(
        session_id,
        COALESCE(cursor_before_partial_message_id, cursor_before_message_id, -1),
        CASE WHEN cursor_before_partial_message_id IS NULL THEN 1 ELSE 0 END,
        cursor_before_offset, slice_key
    );
CREATE INDEX IF NOT EXISTS idx_fact_outcome_replay_v46
    ON fact_extraction_outcomes(
        session_id, source_manifest_complete, prompt_version
    );

CREATE TABLE IF NOT EXISTS fact_extraction_revisions (
    slice_key TEXT NOT NULL REFERENCES fact_extraction_outcomes(slice_key)
        ON DELETE CASCADE,
    generation INTEGER NOT NULL CHECK (generation >= 1),
    prompt_version TEXT NOT NULL,
    outcome_status TEXT NOT NULL CHECK (outcome_status IN ('success','empty')),
    result_hash TEXT NOT NULL,
    succeeded_at TIMESTAMP NOT NULL,
    PRIMARY KEY (slice_key, generation)
);

CREATE TABLE IF NOT EXISTS fact_extraction_source_occurrences (
    slice_key TEXT NOT NULL REFERENCES fact_extraction_outcomes(slice_key)
        ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    source_message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_role TEXT NOT NULL
        CHECK (source_role IN ('user','assistant','system','tool')),
    source_peer_id TEXT,
    source_workspace_id TEXT,
    source_created_at TIMESTAMP,
    source_coverage_chunk_id TEXT NOT NULL,
    source_coverage_version TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    PRIMARY KEY (slice_key, ordinal),
    UNIQUE (slice_key, source_session_id, source_message_id),
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_fact_source_occurrence
    ON fact_extraction_source_occurrences(source_session_id, source_message_id);

-- v26's UNIQUE(session_id,start_message_id,text) cannot represent a corrected
-- payload whose text is unchanged but date/entities differ. Rebuild the table
-- with the authoritative fact key as its only publication identity. IDs are
-- retained so existing embedding rowids remain stable.
DROP TRIGGER IF EXISTS session_workspace_binding_guard;
DROP TRIGGER IF EXISTS narrative_facts_fts_insert;
DROP TRIGGER IF EXISTS narrative_facts_fts_delete;
DROP TRIGGER IF EXISTS narrative_facts_fts_update;
DROP TABLE IF EXISTS narrative_facts_fts;
DROP TABLE IF EXISTS narrative_fact_lifecycle;
CREATE TABLE narrative_facts_v46 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_message_id INTEGER NOT NULL,
    end_message_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    fact_date TEXT,
    entities TEXT NOT NULL DEFAULT '[]',
    prompt_version TEXT NOT NULL,
    valid_at TEXT,
    invalid_at TEXT,
    source_outcome_key TEXT REFERENCES fact_extraction_outcomes(slice_key)
        ON DELETE RESTRICT,
    fact_key TEXT,
    current_generation INTEGER,
    lifecycle_status TEXT NOT NULL DEFAULT 'legacy_unproven'
        CHECK (lifecycle_status IN ('active','retracted','legacy_unproven')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO narrative_facts_v46(
    id,session_id,start_message_id,end_message_id,text,fact_date,entities,
    prompt_version,valid_at,invalid_at,source_outcome_key,fact_key,
    current_generation,lifecycle_status,created_at
)
SELECT id,session_id,start_message_id,end_message_id,text,fact_date,entities,
       prompt_version,valid_at,invalid_at,source_outcome_key,fact_key,
       current_generation,lifecycle_status,created_at
FROM narrative_facts;
-- Copying only surviving ids would otherwise reset AUTOINCREMENT to MAX(id).
-- Carry the old high-water onto the replacement table before DROP/RENAME so
-- deleted fact ids remain retired (external/vector identities must not alias).
INSERT INTO sqlite_sequence(name, seq)
SELECT 'narrative_facts_v46', legacy.seq
FROM sqlite_sequence AS legacy
WHERE legacy.name = 'narrative_facts'
  AND NOT EXISTS (
      SELECT 1 FROM sqlite_sequence AS replacement
      WHERE replacement.name = 'narrative_facts_v46'
  );
UPDATE sqlite_sequence
SET seq = MAX(
    CAST(COALESCE(seq, 0) AS INTEGER),
    CAST(COALESCE((
        SELECT legacy.seq FROM sqlite_sequence AS legacy
        WHERE legacy.name = 'narrative_facts'
    ), 0) AS INTEGER)
)
WHERE name = 'narrative_facts_v46';
CREATE TABLE narrative_fact_embeddings_v46 (
    fact_id INTEGER PRIMARY KEY REFERENCES narrative_facts_v46(id)
        ON DELETE CASCADE,
    vector_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO narrative_fact_embeddings_v46(
    fact_id,vector_json,model,dim,text_hash,created_at
)
SELECT fact_id,vector_json,model,dim,text_hash,created_at
FROM narrative_fact_embeddings;
DROP TABLE narrative_fact_embeddings;
DROP TABLE narrative_facts;
ALTER TABLE narrative_facts_v46 RENAME TO narrative_facts;
ALTER TABLE narrative_fact_embeddings_v46 RENAME TO narrative_fact_embeddings;

CREATE INDEX IF NOT EXISTS idx_narrative_facts_session
    ON narrative_facts(session_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_narrative_fact_authority_key
    ON narrative_facts(source_outcome_key, fact_key)
    WHERE source_outcome_key IS NOT NULL AND fact_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS narrative_fact_lifecycle (
    fact_id INTEGER NOT NULL REFERENCES narrative_facts(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL CHECK (generation >= 1),
    direction INTEGER NOT NULL CHECK (direction IN (-1, 1)),
    event_at TIMESTAMP NOT NULL,
    prompt_version TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fact_id, generation)
);
CREATE INDEX IF NOT EXISTS idx_narrative_fact_lifecycle_time
    ON narrative_fact_lifecycle(fact_id, event_at, generation);

CREATE VIRTUAL TABLE IF NOT EXISTS narrative_facts_fts USING fts5(
    text,
    content='narrative_facts',
    content_rowid='id',
    tokenize='porter unicode61'
);
-- An external-content ``rebuild`` would index every projection row, including
-- legacy facts without provenance and facts retracted by a later extraction
-- generation.  Besides making those rows searchable inside the shadow, their
-- document frequencies would silently perturb BM25 ranking for valid facts.
INSERT INTO narrative_facts_fts(rowid, text)
SELECT id, text
FROM narrative_facts
WHERE source_outcome_key IS NOT NULL
  AND lifecycle_status = 'active'
  AND invalid_at IS NULL;
CREATE TRIGGER IF NOT EXISTS narrative_facts_fts_insert
AFTER INSERT ON narrative_facts
WHEN new.source_outcome_key IS NOT NULL
 AND new.lifecycle_status = 'active'
 AND new.invalid_at IS NULL BEGIN
    INSERT INTO narrative_facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS narrative_facts_fts_delete
AFTER DELETE ON narrative_facts
WHEN old.source_outcome_key IS NOT NULL
 AND old.lifecycle_status = 'active'
 AND old.invalid_at IS NULL BEGIN
    INSERT INTO narrative_facts_fts(narrative_facts_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS narrative_facts_fts_update
AFTER UPDATE OF text, source_outcome_key, lifecycle_status, invalid_at
ON narrative_facts BEGIN
    INSERT INTO narrative_facts_fts(narrative_facts_fts, rowid, text)
    SELECT 'delete', old.id, old.text
    WHERE old.source_outcome_key IS NOT NULL
      AND old.lifecycle_status = 'active'
      AND old.invalid_at IS NULL;
    INSERT INTO narrative_facts_fts(rowid, text)
    SELECT new.id, new.text
    WHERE new.source_outcome_key IS NOT NULL
      AND new.lifecycle_status = 'active'
      AND new.invalid_at IS NULL;
END;

CREATE TRIGGER IF NOT EXISTS fact_outcome_insert_guard
BEFORE INSERT ON fact_extraction_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
  OR NOT (
      new.source_manifest_complete = 0
      AND new.source_manifest_count = 0
      AND new.source_manifest_hash IS NULL
      AND new.source_manifest_version IS NULL
      AND length(new.slice_key) = 71 AND new.slice_key GLOB 'sha256:*'
      AND length(new.input_hash) = 71 AND new.input_hash GLOB 'sha256:*'
      AND length(new.result_hash) = 71 AND new.result_hash GLOB 'sha256:*'
  )
BEGIN
    SELECT RAISE(ABORT, 'fact outcome must stage through its internal writer');
END;

CREATE TRIGGER IF NOT EXISTS fact_outcome_header_guard
BEFORE UPDATE OF source_manifest_version, source_manifest_count,
                 source_manifest_hash, source_manifest_complete
ON fact_extraction_outcomes
WHEN (old.source_manifest_complete = 1
      AND hymem_evidence_destructive_authorized() <> 1)
  OR hymem_evidence_mutation_authorized() <> 1
  OR NOT (
      (new.source_manifest_complete = 0
       AND new.source_manifest_count = 0
       AND new.source_manifest_hash IS NULL
       AND new.source_manifest_version IS NULL)
      OR
      (new.source_manifest_complete = 1
       AND new.source_manifest_version = 'fact-source-manifest-v1'
       AND new.source_manifest_count > 0
       AND length(new.source_manifest_hash) = 71
       AND new.source_manifest_hash GLOB 'sha256:*'
       AND (SELECT COUNT(*) FROM fact_extraction_source_occurrences source
            WHERE source.slice_key = new.slice_key) = new.source_manifest_count
       AND (SELECT MIN(ordinal) FROM fact_extraction_source_occurrences source
            WHERE source.slice_key = new.slice_key) = 0
       AND (SELECT MAX(ordinal) FROM fact_extraction_source_occurrences source
            WHERE source.slice_key = new.slice_key) = new.source_manifest_count - 1)
  )
BEGIN
    SELECT RAISE(ABORT, 'invalid fact source manifest publication');
END;

CREATE TRIGGER IF NOT EXISTS fact_outcome_bound_guard
BEFORE UPDATE OF slice_key, session_id, input_hash,
                 cursor_before_message_id, cursor_before_partial_message_id,
                 cursor_before_offset, cursor_after_message_id,
                 cursor_after_partial_message_id, cursor_after_offset
ON fact_extraction_outcomes
WHEN old.source_manifest_complete = 1
BEGIN
    SELECT RAISE(ABORT, 'fact extraction source unit is immutable');
END;

CREATE TRIGGER IF NOT EXISTS fact_outcome_result_guard
BEFORE UPDATE OF prompt_version, generation, outcome_status, result_hash,
                 succeeded_at
ON fact_extraction_outcomes
WHEN hymem_evidence_mutation_authorized() <> 1
  OR new.generation < old.generation
  OR length(new.result_hash) <> 71
  OR new.result_hash NOT GLOB 'sha256:*'
BEGIN
    SELECT RAISE(ABORT, 'fact outcome result is internally managed');
END;

CREATE TRIGGER IF NOT EXISTS fact_outcome_delete_guard
BEFORE DELETE ON fact_extraction_outcomes
WHEN hymem_evidence_destructive_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'published fact outcome history is immutable');
END;

CREATE TRIGGER IF NOT EXISTS fact_revision_insert_guard
BEFORE INSERT ON fact_extraction_revisions
WHEN hymem_evidence_mutation_authorized() <> 1
  OR length(new.result_hash) <> 71
  OR new.result_hash NOT GLOB 'sha256:*'
BEGIN
    SELECT RAISE(ABORT, 'fact extraction revisions are internally managed');
END;

CREATE TRIGGER IF NOT EXISTS fact_revision_update_guard
BEFORE UPDATE ON fact_extraction_revisions
BEGIN
    SELECT RAISE(ABORT, 'fact extraction revisions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS fact_revision_delete_guard
BEFORE DELETE ON fact_extraction_revisions
WHEN hymem_evidence_destructive_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'published fact revision history is immutable');
END;

CREATE TRIGGER IF NOT EXISTS fact_source_occurrence_insert_guard
BEFORE INSERT ON fact_extraction_source_occurrences
WHEN hymem_evidence_mutation_authorized() <> 1
  OR NOT EXISTS (
      SELECT 1
      FROM fact_extraction_outcomes outcome
      JOIN message_retention_coverage proof
        ON proof.message_id = new.source_message_id
       AND proof.chunk_id = new.source_coverage_chunk_id
       AND proof.coverage_version = new.source_coverage_version
      WHERE outcome.slice_key = new.slice_key
        AND outcome.source_manifest_complete = 0
        AND outcome.session_id = new.source_session_id
        AND proof.source_session_id = new.source_session_id
        AND proof.source_role = new.source_role
        AND proof.source_peer_id IS new.source_peer_id
        AND proof.source_workspace_id IS new.source_workspace_id
        AND proof.source_created_at IS new.source_created_at
        AND proof.message_content_hash = new.source_content_hash
        AND proof.coverage_version = 'dream-lossless-message-v1'
  )
BEGIN
    SELECT RAISE(ABORT, 'fact source occurrence mismatches coverage');
END;

CREATE TRIGGER IF NOT EXISTS fact_source_occurrence_update_guard
BEFORE UPDATE ON fact_extraction_source_occurrences
BEGIN
    SELECT RAISE(ABORT, 'fact source occurrences are immutable');
END;

CREATE TRIGGER IF NOT EXISTS fact_source_occurrence_delete_guard
BEFORE DELETE ON fact_extraction_source_occurrences
WHEN hymem_evidence_destructive_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'published fact source history is immutable');
END;

CREATE TRIGGER IF NOT EXISTS narrative_fact_authority_insert_guard
BEFORE INSERT ON narrative_facts
WHEN new.source_outcome_key IS NOT NULL AND (
    hymem_evidence_mutation_authorized() <> 1
    OR new.fact_key IS NULL OR length(new.fact_key) <> 71
    OR new.fact_key NOT GLOB 'sha256:*'
    OR new.current_generation IS NULL OR new.current_generation < 1
    OR new.lifecycle_status NOT IN ('active','retracted')
    OR NOT EXISTS (
        SELECT 1 FROM fact_extraction_outcomes outcome
        WHERE outcome.slice_key = new.source_outcome_key
          AND outcome.session_id = new.session_id
          AND outcome.source_manifest_complete = 1
    )
)
BEGIN
    SELECT RAISE(ABORT, 'authoritative facts require a published source outcome');
END;

CREATE TRIGGER IF NOT EXISTS narrative_fact_authority_update_guard
BEFORE UPDATE OF source_outcome_key, fact_key, current_generation,
                 lifecycle_status
ON narrative_facts
WHEN old.source_outcome_key IS NULL AND new.source_outcome_key IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'legacy facts cannot be promoted to authoritative');
END;

CREATE TRIGGER IF NOT EXISTS narrative_fact_bound_update_guard
BEFORE UPDATE OF session_id, start_message_id, end_message_id, text, fact_date,
                 entities, prompt_version, source_outcome_key, fact_key,
                 created_at
ON narrative_facts
WHEN old.source_outcome_key IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'authoritative fact identity is immutable');
END;

CREATE TRIGGER IF NOT EXISTS narrative_fact_lifecycle_projection_guard
BEFORE UPDATE OF valid_at, invalid_at, current_generation, lifecycle_status
ON narrative_facts
WHEN old.source_outcome_key IS NOT NULL
 AND hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'fact lifecycle is internally managed');
END;

CREATE TRIGGER IF NOT EXISTS narrative_fact_delete_guard
BEFORE DELETE ON narrative_facts
WHEN old.source_outcome_key IS NOT NULL
 AND hymem_evidence_destructive_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'published narrative fact history is immutable');
END;

CREATE TRIGGER IF NOT EXISTS narrative_fact_lifecycle_insert_guard
BEFORE INSERT ON narrative_fact_lifecycle
WHEN hymem_evidence_mutation_authorized() <> 1
  OR length(new.result_hash) <> 71
  OR new.result_hash NOT GLOB 'sha256:*'
  OR NOT EXISTS (
      SELECT 1 FROM narrative_facts fact
      WHERE fact.id = new.fact_id AND fact.source_outcome_key IS NOT NULL
  )
BEGIN
    SELECT RAISE(ABORT, 'fact lifecycle is internally managed');
END;

CREATE TRIGGER IF NOT EXISTS narrative_fact_lifecycle_update_guard
BEFORE UPDATE ON narrative_fact_lifecycle
BEGIN
    SELECT RAISE(ABORT, 'fact lifecycle history is immutable');
END;

CREATE TRIGGER IF NOT EXISTS narrative_fact_lifecycle_delete_guard
BEFORE DELETE ON narrative_fact_lifecycle
WHEN hymem_evidence_destructive_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'published fact lifecycle history is immutable');
END;

-- v43's ownership guard predated the fact outcome ledger.  Empty successful
-- units carry no narrative_facts row, but are still session-owned durable
-- history.  Reinstall the guard with the complete v46 cursor/outcome domain so
-- neither a cursor nor an EMPTY publication can be laundered into a workspace.
CREATE TRIGGER IF NOT EXISTS session_workspace_binding_guard
BEFORE UPDATE OF source_workspace_id ON sessions
WHEN (
    old.source_workspace_id IS NOT NULL
    AND new.source_workspace_id IS NOT old.source_workspace_id
) OR (
    old.source_workspace_id IS NULL
    AND new.source_workspace_id IS NOT NULL
    AND (
      length(trim(new.source_workspace_id)) = 0
      OR old.ended_at IS NOT NULL
      OR old.summary IS NOT NULL
      OR old.digested_prompt_version IS NOT NULL
      OR old.profile_prompt_version IS NOT NULL
      OR old.profile_cursor_message_id IS NOT NULL
      OR old.profile_cursor_partial_message_id IS NOT NULL
      OR old.profile_cursor_offset <> 0
      OR old.profile_cursor_prompt_version IS NOT NULL
      OR old.profile_published_generation IS NOT NULL
      OR old.profile_retry_count <> 0
      OR old.profile_retry_config_version IS NOT NULL
      OR old.profile_quarantined <> 0
      OR old.digested_message_id IS NOT NULL
      OR old.facts_message_id IS NOT NULL
      OR old.facts_cursor_message_id IS NOT NULL
      OR old.facts_cursor_partial_message_id IS NOT NULL
      OR old.facts_cursor_offset <> 0
      OR old.facts_cursor_prompt_version IS NOT NULL
      OR old.facts_retry_count <> 0
      OR old.facts_retry_config_version IS NOT NULL
      OR old.facts_quarantined <> 0
      OR old.episodes_prompt_version IS NOT NULL
      OR old.coverage_message_id IS NOT NULL
      OR old.digest_cursor_message_id IS NOT NULL
      OR old.digest_cursor_partial_message_id IS NOT NULL
      OR old.digest_cursor_offset <> 0
      OR old.digest_cursor_prompt_version IS NOT NULL
      OR old.digest_published_generation IS NOT NULL
      OR old.digest_retry_count <> 0
      OR old.digest_retry_config_version IS NOT NULL
      OR old.digest_quarantined <> 0
      OR old.auto_summary IS NOT NULL
      OR old.auto_summary_message_id IS NOT NULL
      OR old.auto_summary_partial_message_id IS NOT NULL
      OR old.auto_summary_message_offset <> 0
      OR old.summary_source IS NOT NULL
      OR EXISTS (SELECT 1 FROM messages WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM chunks WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM episodes WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM procedures WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM profile_staging WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM temporal_mentions WHERE session_id = old.id)
      OR EXISTS (SELECT 1 FROM narrative_facts WHERE session_id = old.id)
      OR EXISTS (
        SELECT 1 FROM fact_extraction_outcomes WHERE session_id = old.id
      )
      OR EXISTS (
        SELECT 1 FROM message_retention_coverage
        WHERE source_session_id = old.id
      )
      OR EXISTS (
        SELECT 1 FROM chunk_message_sources
        WHERE source_session_id = old.id
      )
      OR EXISTS (
        SELECT 1 FROM user_profile WHERE source_session_id = old.id
      )
      OR EXISTS (
        SELECT 1 FROM kg_evidence WHERE source_session_id = old.id
      )
      OR EXISTS (
        SELECT 1 FROM kg_claim_observations
        WHERE source_session_id = old.id
      )
      OR EXISTS (SELECT 1 FROM session_peers WHERE session_id = old.id)
    )
)
BEGIN
    SELECT RAISE(ABORT, 'session workspace binding is immutable');
END;

-- v26's numeric watermark cannot prove that the facts it skipped were ever
-- exported.  Rewind the new authoritative cursor; the old column remains only
-- as a compatibility mirror for future successful full-message boundaries.
UPDATE sessions
SET facts_cursor_message_id = NULL,
    facts_cursor_partial_message_id = NULL,
    facts_cursor_offset = 0,
    facts_cursor_prompt_version = NULL,
    facts_retry_count = 0,
    facts_retry_config_version = NULL,
    facts_quarantined = 0,
    facts_message_id = NULL;
