-- v40: exact message-level provenance for knowledge-graph claims.
--
-- Pre-v40 evidence identified only an extraction chunk.  A chunk can contain
-- several speakers and messages, so neither the author nor the valid-time of a
-- claim was recoverable.  Rebuild the ledger with an immutable reference to the
-- reviewed lossless-message artifact.  Historical rows are deliberately marked
-- legacy_unattributed; their old chunk-first role is discarded rather than
-- promoted into fabricated claim provenance. The only safe upgrade is a
-- singleton extraction chunk backed by a fully validated ordered artifact;
-- those rows are upgraded exactly, while every other old row survives as
-- legacy_unattributed.

PRAGMA foreign_keys = OFF;
SAVEPOINT kg_evidence_v40_rebuild;

-- Supported sparse legacy fixtures can carry a historical `chunks(id)` table.
-- Complete the shape before provenance recovery; duplicate-column errors are
-- ignored by the migration runner on normal stores.
ALTER TABLE chunks ADD COLUMN session_id TEXT;
ALTER TABLE chunks ADD COLUMN start_message_id INTEGER;
ALTER TABLE chunks ADD COLUMN end_message_id INTEGER;
ALTER TABLE chunks ADD COLUMN salience_reason TEXT;
ALTER TABLE chunks ADD COLUMN created_at TIMESTAMP;
ALTER TABLE chunks ADD COLUMN chunk_kind TEXT NOT NULL DEFAULT 'extraction';
ALTER TABLE chunks ADD COLUMN source_manifest_version TEXT;
ALTER TABLE chunks ADD COLUMN source_manifest_count INTEGER;

CREATE TABLE IF NOT EXISTS chunk_message_sources (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    source_message_id INTEGER NOT NULL,
    source_session_id TEXT NOT NULL,
    source_coverage_chunk_id TEXT NOT NULL,
    source_coverage_version TEXT NOT NULL,
    PRIMARY KEY (chunk_id, ordinal),
    UNIQUE (chunk_id, source_message_id),
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_chunk_message_sources_source
    ON chunk_message_sources(source_session_id, source_message_id);

CREATE TRIGGER IF NOT EXISTS chunk_source_manifest_shape_insert_guard
BEFORE INSERT ON chunks
WHEN new.source_manifest_version IS NOT NULL
  OR new.source_manifest_count IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'invalid chunk source manifest header');
END;
CREATE TRIGGER IF NOT EXISTS chunk_source_manifest_shape_update_guard
BEFORE UPDATE OF source_manifest_version, source_manifest_count ON chunks
WHEN NOT (
    (new.source_manifest_version IS NULL AND new.source_manifest_count IS NULL)
    OR (new.source_manifest_version = 'claim-source-manifest-v1'
        AND new.source_manifest_count > 0)
)
BEGIN
    SELECT RAISE(ABORT, 'invalid chunk source manifest header');
END;

CREATE TRIGGER IF NOT EXISTS chunk_message_sources_insert_guard
BEFORE INSERT ON chunk_message_sources
WHEN NOT EXISTS (
    SELECT 1
    FROM chunks extraction_chunk
    JOIN message_retention_coverage mc
      ON mc.message_id = new.source_message_id
     AND mc.source_session_id = new.source_session_id
     AND mc.chunk_id = new.source_coverage_chunk_id
     AND mc.coverage_version = new.source_coverage_version
    JOIN sessions s ON s.id = mc.source_session_id
    JOIN chunks source_chunk ON source_chunk.id = mc.chunk_id
    WHERE extraction_chunk.id = new.chunk_id
      AND extraction_chunk.chunk_kind = 'extraction'
      AND extraction_chunk.session_id = new.source_session_id
      AND new.source_message_id BETWEEN extraction_chunk.start_message_id
                                    AND extraction_chunk.end_message_id
      AND mc.coverage_version = 'dream-lossless-message-v1'
      AND s.coverage_message_id IS NOT NULL
      AND mc.message_id <= s.coverage_message_id
      AND source_chunk.id = hymem_coverage_chunk_id(
            mc.source_session_id, mc.message_id
          )
      AND source_chunk.session_id = mc.source_session_id
      AND source_chunk.start_message_id = mc.message_id
      AND source_chunk.end_message_id = mc.message_id
      AND source_chunk.chunk_kind = 'coverage'
      AND json_valid(source_chunk.text)
      AND hymem_message_record_proof_valid(
            source_chunk.text, mc.message_content_hash,
            mc.hash_version, mc.record_version
          ) = 1
) BEGIN
    SELECT RAISE(ABORT, 'chunk source manifest provenance mismatch');
END;
CREATE TRIGGER IF NOT EXISTS chunk_message_sources_published_insert_guard
BEFORE INSERT ON chunk_message_sources
WHEN EXISTS (
    SELECT 1 FROM chunks c WHERE c.id = new.chunk_id
      AND c.source_manifest_version = 'claim-source-manifest-v1'
)
BEGIN
    SELECT RAISE(ABORT, 'published chunk source manifest is immutable');
END;

CREATE TRIGGER IF NOT EXISTS chunk_message_sources_update_guard
BEFORE UPDATE ON chunk_message_sources
BEGIN
    SELECT RAISE(ABORT, 'published chunk source manifest is immutable');
END;
CREATE TRIGGER IF NOT EXISTS chunk_message_sources_delete_guard
BEFORE DELETE ON chunk_message_sources
WHEN EXISTS (
    SELECT 1 FROM chunks c WHERE c.id = old.chunk_id
      AND c.source_manifest_version = 'claim-source-manifest-v1'
) BEGIN
    SELECT RAISE(ABORT, 'published chunk source manifest is immutable');
END;
CREATE TRIGGER IF NOT EXISTS chunk_source_manifest_publish_guard
BEFORE UPDATE OF source_manifest_version, source_manifest_count ON chunks
WHEN new.source_manifest_version = 'claim-source-manifest-v1'
 AND old.source_manifest_version IS NULL
 AND NOT (
    new.source_manifest_count IS NOT NULL
    AND new.source_manifest_count > 0
    AND (SELECT COUNT(*) FROM chunk_message_sources cms
         WHERE cms.chunk_id = new.id) = new.source_manifest_count
    AND (SELECT MIN(ordinal) FROM chunk_message_sources cms
         WHERE cms.chunk_id = new.id) = 0
    AND (SELECT MAX(ordinal) FROM chunk_message_sources cms
         WHERE cms.chunk_id = new.id) = new.source_manifest_count - 1
    AND (SELECT source_message_id FROM chunk_message_sources cms
         WHERE cms.chunk_id = new.id ORDER BY ordinal LIMIT 1)
        = new.start_message_id
    AND (SELECT source_message_id FROM chunk_message_sources cms
         WHERE cms.chunk_id = new.id ORDER BY ordinal DESC LIMIT 1)
        = new.end_message_id
    AND NOT EXISTS (
        SELECT 1
        FROM chunk_message_sources earlier
        JOIN chunk_message_sources later
          ON later.chunk_id = earlier.chunk_id
         AND later.ordinal = earlier.ordinal + 1
        WHERE earlier.chunk_id = new.id
          AND earlier.source_message_id >= later.source_message_id
    )
 )
BEGIN
    SELECT RAISE(ABORT, 'chunk source manifest is incomplete');
END;
CREATE TRIGGER IF NOT EXISTS chunk_source_manifest_header_update_guard
BEFORE UPDATE OF source_manifest_version, source_manifest_count ON chunks
WHEN old.source_manifest_version IS NOT NULL
 AND (new.source_manifest_version IS NOT old.source_manifest_version
      OR new.source_manifest_count IS NOT old.source_manifest_count)
 AND NOT (
      new.source_manifest_version IS NULL
      AND new.source_manifest_count IS NULL
      AND NOT EXISTS (SELECT 1 FROM kg_evidence ev WHERE ev.chunk_id = old.id)
      AND NOT EXISTS (
          SELECT 1 FROM kg_claim_observations observation
          WHERE observation.chunk_id = old.id
      )
 )
BEGIN
    SELECT RAISE(ABORT, 'published chunk source manifest header is immutable');
END;

CREATE TRIGGER IF NOT EXISTS chunk_source_manifest_chunk_update_guard
BEFORE UPDATE OF session_id, start_message_id, end_message_id, text, chunk_kind
ON chunks
WHEN old.source_manifest_version = 'claim-source-manifest-v1'
BEGIN
    SELECT RAISE(ABORT, 'manifested extraction chunk is immutable');
END;

DROP TRIGGER IF EXISTS kg_evidence_count_after_delete;
DROP TRIGGER IF EXISTS chunk_source_manifest_header_update_guard;
DROP TRIGGER IF EXISTS kg_edge_lifecycle_insert_guard;
DROP TRIGGER IF EXISTS kg_edge_lifecycle_update_guard;
DROP TRIGGER IF EXISTS kg_edge_lifecycle_delete_guard;
DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_insert_guard;
DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_update_guard;
DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_delete_guard;
DROP TABLE IF EXISTS kg_evidence_v40;

CREATE TABLE kg_evidence_v40 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE RESTRICT,
    polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
    surface_subject TEXT,
    surface_object TEXT,
    value_text TEXT,
    value_numeric REAL,
    value_unit TEXT,
    temporal_scope TEXT,
    source_role TEXT CHECK (
        source_role IS NULL OR source_role IN ('user','assistant','system','tool')
    ),
    evidence_kind TEXT NOT NULL DEFAULT 'extraction',
    evidence_weight INTEGER NOT NULL DEFAULT 1 CHECK (evidence_weight >= 1),
    weight_source TEXT NOT NULL DEFAULT 'legacy_default',
    extraction_prompt_version TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_message_id INTEGER,
    source_session_id TEXT,
    source_created_at TIMESTAMP,
    source_event_at TEXT,
    source_coverage_chunk_id TEXT,
    source_coverage_version TEXT,
    provenance_status TEXT NOT NULL DEFAULT 'legacy_unattributed'
        CHECK (provenance_status IN ('canonical', 'legacy_unattributed')),
    interpretation_key TEXT NOT NULL DEFAULT 'legacy-unspecified',
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    is_current BOOLEAN NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    superseded_at TIMESTAMP,
    superseded_reason TEXT,
    FOREIGN KEY (
        source_message_id, source_coverage_chunk_id, source_coverage_version
    ) REFERENCES message_retention_coverage(
        message_id, chunk_id, coverage_version
    ) ON DELETE RESTRICT
);

INSERT INTO kg_evidence_v40(
    id, edge_id, chunk_id, polarity, surface_subject, surface_object,
    value_text, value_numeric, value_unit, temporal_scope, source_role,
    evidence_kind, evidence_weight, weight_source,
    extraction_prompt_version, extracted_at, provenance_status,
    interpretation_key
)
SELECT
    old.id, old.edge_id, old.chunk_id, old.polarity,
    old.surface_subject, old.surface_object,
    old.value_text, old.value_numeric, old.value_unit, old.temporal_scope, NULL,
    old.evidence_kind, old.evidence_weight, old.weight_source,
    old.extraction_prompt_version, old.extracted_at, 'legacy_unattributed',
    'legacy-migrated-v1'
FROM kg_evidence old
ORDER BY old.id;

DROP TABLE kg_evidence;
ALTER TABLE kg_evidence_v40 RENAME TO kg_evidence;

CREATE INDEX IF NOT EXISTS idx_evidence_edge ON kg_evidence(edge_id);
CREATE INDEX IF NOT EXISTS idx_evidence_chunk ON kg_evidence(chunk_id);
CREATE INDEX IF NOT EXISTS idx_evidence_source
    ON kg_evidence(source_session_id, source_message_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_canonical_identity
    ON kg_evidence(edge_id, source_session_id, source_message_id, evidence_kind)
    WHERE provenance_status = 'canonical' AND is_current = 1;
CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_legacy_identity
    ON kg_evidence(edge_id, chunk_id, evidence_kind)
    WHERE provenance_status = 'legacy_unattributed' AND is_current = 1;
CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_canonical_revision
    ON kg_evidence(
        edge_id, source_session_id, source_message_id, evidence_kind, revision
    ) WHERE provenance_status = 'canonical';
CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_legacy_revision
    ON kg_evidence(edge_id, chunk_id, evidence_kind, revision)
    WHERE provenance_status = 'legacy_unattributed';

-- A source proof contributes confidence once globally, while this authority
-- relation records which successfully extracted chunks currently assert it.
-- Overlapping chunks therefore form a deterministic union: replaying one
-- chunk empty cannot erase the same cited claim still asserted by another.
CREATE TABLE IF NOT EXISTS kg_claim_observations (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE RESTRICT,
    edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
    source_session_id TEXT NOT NULL,
    source_message_id INTEGER NOT NULL,
    evidence_kind TEXT NOT NULL DEFAULT 'extraction',
    polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
    prompt_version TEXT NOT NULL,
    prompt_generation INTEGER NOT NULL CHECK (prompt_generation >= 0),
    evidence_id INTEGER NOT NULL REFERENCES kg_evidence(id) ON DELETE RESTRICT,
    interpretation_key TEXT NOT NULL,
    observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (
        chunk_id, edge_id, source_session_id, source_message_id, evidence_kind
    )
);
CREATE INDEX IF NOT EXISTS idx_kg_claim_observations_source
    ON kg_claim_observations(
        edge_id, source_session_id, source_message_id, evidence_kind
    );
CREATE TRIGGER IF NOT EXISTS kg_claim_observations_insert_guard
BEFORE INSERT ON kg_claim_observations
BEGIN
    SELECT RAISE(ABORT, 'claim observations are internally managed')
    WHERE hymem_evidence_mutation_authorized() <> 1;
    SELECT RAISE(ABORT, 'claim observation lacks current canonical evidence')
    WHERE NOT (
      EXISTS (
        SELECT 1 FROM chunk_message_sources cms
        JOIN chunks c ON c.id = cms.chunk_id
        WHERE cms.chunk_id = new.chunk_id
          AND cms.source_session_id = new.source_session_id
          AND cms.source_message_id = new.source_message_id
          AND c.source_manifest_version = 'claim-source-manifest-v1'
    )
    AND EXISTS (
        SELECT 1 FROM kg_evidence ev
        WHERE ev.id = new.evidence_id
          AND ev.edge_id = new.edge_id
          AND ev.source_session_id = new.source_session_id
          AND ev.source_message_id = new.source_message_id
          AND ev.evidence_kind = new.evidence_kind
          AND ev.polarity = new.polarity
          AND ev.interpretation_key = new.interpretation_key
          AND ev.provenance_status = 'canonical'
          AND (ev.is_current = 1
               OR hymem_evidence_history_authorized() = 1)
          AND ev.revision > 0
    )
    AND NOT EXISTS (
        SELECT 1 FROM kg_claim_observations existing
        WHERE existing.edge_id = new.edge_id
          AND existing.source_session_id = new.source_session_id
          AND existing.source_message_id = new.source_message_id
          AND existing.evidence_kind = new.evidence_kind
          AND existing.prompt_generation = new.prompt_generation
          AND (
              existing.polarity <> new.polarity
              OR existing.interpretation_key <> new.interpretation_key
          )
      )
    );
END;
CREATE TRIGGER IF NOT EXISTS kg_claim_observations_update_guard
BEFORE UPDATE ON kg_claim_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'claim observations are internally managed');
END;
CREATE TRIGGER IF NOT EXISTS kg_claim_observations_delete_guard
BEFORE DELETE ON kg_claim_observations
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'claim observations are internally managed');
END;

CREATE TABLE IF NOT EXISTS kg_edge_lifecycle (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id INTEGER NOT NULL REFERENCES knowledge_graph(id) ON DELETE CASCADE,
    event_key TEXT NOT NULL,
    event_kind TEXT NOT NULL CHECK (event_kind IN (
        'claim_assertion','manual_retraction','phase3_retraction',
        'value_supersession','legacy_state'
    )),
    direction INTEGER NOT NULL CHECK (direction IN (-1, 1)),
    event_at TEXT NOT NULL,
    source_evidence_id INTEGER REFERENCES kg_evidence(id) ON DELETE CASCADE,
    dependency_count INTEGER NOT NULL DEFAULT 0 CHECK (dependency_count >= 0),
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(edge_id, event_key)
);
CREATE INDEX IF NOT EXISTS idx_kg_edge_lifecycle_edge
    ON kg_edge_lifecycle(edge_id, event_at, event_key);

CREATE TABLE IF NOT EXISTS kg_lifecycle_dependencies (
    lifecycle_id INTEGER NOT NULL
        REFERENCES kg_edge_lifecycle(id) ON DELETE CASCADE,
    evidence_id INTEGER NOT NULL REFERENCES kg_evidence(id) ON DELETE CASCADE,
    PRIMARY KEY (lifecycle_id, evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_kg_lifecycle_dependencies_evidence
    ON kg_lifecycle_dependencies(evidence_id);

CREATE TRIGGER IF NOT EXISTS kg_edge_lifecycle_insert_guard
BEFORE INSERT ON kg_edge_lifecycle
BEGIN
    SELECT RAISE(ABORT, 'knowledge graph lifecycle events are internally managed')
    WHERE hymem_evidence_mutation_authorized() <> 1;
    SELECT RAISE(ABORT, 'invalid knowledge graph lifecycle event')
    WHERE NOT (
      new.event_at = COALESCE(
        hymem_normalize_iso_timestamp(new.event_at),
        '0001-01-01T00:00:00.000Z'
    )
    AND (
        (new.event_kind = 'claim_assertion' AND new.direction = 1
         AND new.source_evidence_id IS NOT NULL
         AND new.dependency_count = 0)
        OR (new.event_kind = 'manual_retraction'
            AND new.direction = -1 AND new.dependency_count = 0)
        OR (new.event_kind = 'value_supersession'
            AND new.direction = -1 AND new.dependency_count > 0
            AND new.source_evidence_id IS NULL)
        OR (new.event_kind = 'phase3_retraction'
            AND new.direction = -1 AND new.dependency_count > 0
            AND new.source_evidence_id IS NULL)
        OR (new.event_kind = 'legacy_state'
            AND new.source_evidence_id IS NULL
            AND new.dependency_count = 0)
    )
    AND (
        new.source_evidence_id IS NULL
        OR EXISTS (
            SELECT 1 FROM kg_evidence ev
            WHERE ev.id = new.source_evidence_id
              AND ev.edge_id = new.edge_id
              AND ev.provenance_status = 'canonical'
              AND (ev.is_current = 1
                   OR hymem_evidence_history_authorized() = 1)
              AND ev.polarity = new.direction
              AND ev.source_event_at = new.event_at
        )
      )
    );
END;
CREATE TRIGGER IF NOT EXISTS kg_edge_lifecycle_update_guard
BEFORE UPDATE ON kg_edge_lifecycle
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'knowledge graph lifecycle events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS kg_edge_lifecycle_delete_guard
BEFORE DELETE ON kg_edge_lifecycle
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'knowledge graph lifecycle events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_insert_guard
BEFORE INSERT ON kg_lifecycle_dependencies
BEGIN
    SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed')
    WHERE hymem_evidence_mutation_authorized() <> 1;
    SELECT RAISE(ABORT, 'invalid lifecycle evidence dependency')
    WHERE NOT EXISTS (
      SELECT 1
      FROM kg_edge_lifecycle lifecycle
      JOIN kg_evidence ev ON ev.id = new.evidence_id
      WHERE lifecycle.id = new.lifecycle_id
        AND lifecycle.direction = -1
        AND (ev.is_current = 1
             OR hymem_evidence_history_authorized() = 1)
        AND (
          (lifecycle.event_kind = 'phase3_retraction'
           AND lifecycle.edge_id = ev.edge_id AND ev.polarity = -1)
          OR (lifecycle.event_kind = 'value_supersession'
              AND ev.polarity = 1
              AND EXISTS (
                  SELECT 1
                  FROM knowledge_graph loser
                  JOIN knowledge_graph winner
                    ON winner.subject_canonical = loser.subject_canonical
                   AND winner.predicate = loser.predicate
                   AND winner.object_canonical <> loser.object_canonical
                  WHERE loser.id = lifecycle.edge_id
                    AND winner.id = ev.edge_id
                    AND winner.derived = 0
              ))
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_update_guard
BEFORE UPDATE ON kg_lifecycle_dependencies
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed');
END;
CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_delete_guard
BEFORE DELETE ON kg_lifecycle_dependencies
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed');
END;

RELEASE SAVEPOINT kg_evidence_v40_rebuild;
PRAGMA foreign_keys = ON;

-- A canonical row is an exact citation into the reviewed ordered source
-- stream and into the extraction chunk that was shown to the model.  Missing
-- host timestamps use the fixed sentinel below; message/session/id then form
-- the deterministic secondary order in Python/SQL readers.
CREATE TRIGGER IF NOT EXISTS kg_evidence_v40_insert_guard
BEFORE INSERT ON kg_evidence
BEGIN
    SELECT RAISE(ABORT, 'kg evidence revisions are internally managed')
    WHERE hymem_evidence_mutation_authorized() <> 1;
    SELECT RAISE(ABORT, 'invalid kg evidence revision state')
    WHERE NOT (
        new.revision > 0
        AND length(trim(new.interpretation_key)) > 0
        AND (
            (new.is_current = 1
             AND new.superseded_at IS NULL
             AND new.superseded_reason IS NULL)
            OR (new.is_current = 0
                AND new.superseded_at IS NOT NULL
                AND length(trim(new.superseded_reason)) > 0)
        )
    );
    SELECT RAISE(ABORT, 'canonical kg evidence provenance mismatch')
    WHERE new.provenance_status = 'canonical' AND NOT (
        new.source_message_id IS NOT NULL
        AND new.source_session_id IS NOT NULL
        AND length(trim(new.source_session_id)) > 0
        AND new.source_role IN ('user','assistant','system','tool')
        AND new.source_event_at IS COALESCE(
            hymem_normalize_iso_timestamp(new.source_created_at),
            '0001-01-01T00:00:00.000Z')
        AND new.source_coverage_chunk_id IS NOT NULL
        AND new.source_coverage_version = 'dream-lossless-message-v1'
        AND EXISTS (
          SELECT 1
          FROM message_retention_coverage mc
          JOIN sessions source_session ON source_session.id = mc.source_session_id
          JOIN chunks source_chunk ON source_chunk.id = mc.chunk_id
          JOIN chunks extraction_chunk ON extraction_chunk.id = new.chunk_id
          WHERE mc.message_id = new.source_message_id
            AND mc.source_session_id = new.source_session_id
            AND mc.source_role = new.source_role
            AND mc.source_created_at IS new.source_created_at
            AND mc.chunk_id = new.source_coverage_chunk_id
            AND mc.coverage_version = new.source_coverage_version
            AND source_session.coverage_message_id IS NOT NULL
            AND mc.message_id <= source_session.coverage_message_id
            AND source_chunk.chunk_kind = 'coverage'
            AND source_chunk.id = hymem_coverage_chunk_id(
                  mc.source_session_id, mc.message_id
                )
            AND source_chunk.session_id = mc.source_session_id
            AND source_chunk.start_message_id = mc.message_id
            AND source_chunk.end_message_id = mc.message_id
            AND json_valid(source_chunk.text)
            AND json_extract(source_chunk.text, '$.id') = mc.message_id
            AND json_extract(source_chunk.text, '$.role') = mc.source_role
            AND json_extract(source_chunk.text, '$.record_version') = mc.record_version
            AND hymem_message_record_proof_valid(
                  source_chunk.text, mc.message_content_hash,
                  mc.hash_version, mc.record_version
                ) = 1
            AND extraction_chunk.chunk_kind = 'extraction'
            AND extraction_chunk.session_id = new.source_session_id
            AND extraction_chunk.source_manifest_version = 'claim-source-manifest-v1'
            AND extraction_chunk.source_manifest_count > 0
            AND new.source_message_id BETWEEN extraction_chunk.start_message_id
                                          AND extraction_chunk.end_message_id
            AND EXISTS (
              SELECT 1 FROM chunk_message_sources cms
              WHERE cms.chunk_id = extraction_chunk.id
                AND cms.source_message_id = new.source_message_id
                AND cms.source_session_id = new.source_session_id
                AND cms.source_coverage_chunk_id = new.source_coverage_chunk_id
                AND cms.source_coverage_version = new.source_coverage_version
            )
        )
      );
    SELECT RAISE(ABORT, 'legacy kg evidence must remain unattributed')
    WHERE new.provenance_status = 'legacy_unattributed' AND NOT (
        new.source_message_id IS NULL
        AND new.source_session_id IS NULL
        AND new.source_role IS NULL
        AND new.source_created_at IS NULL
        AND new.source_event_at IS NULL
        AND new.source_coverage_chunk_id IS NULL
        AND new.source_coverage_version IS NULL
      );
END;

CREATE TRIGGER IF NOT EXISTS kg_evidence_v40_update_guard
BEFORE UPDATE OF source_message_id, source_session_id, source_role,
                 source_created_at, source_event_at,
                 source_coverage_chunk_id, source_coverage_version,
                 provenance_status, chunk_id, revision,
                 is_current, superseded_at, superseded_reason,
                 edge_id, polarity, evidence_kind, evidence_weight,
                 weight_source, surface_subject, surface_object, value_text,
                 value_numeric, value_unit, temporal_scope,
                 extraction_prompt_version, extracted_at,
                 interpretation_key
ON kg_evidence
BEGIN
    SELECT RAISE(ABORT, 'kg evidence revisions are internally managed')
    WHERE hymem_evidence_mutation_authorized() <> 1;
    SELECT RAISE(ABORT, 'invalid kg evidence revision state')
    WHERE NOT (
        new.revision > 0
        AND length(trim(new.interpretation_key)) > 0
        AND (
            (new.is_current = 1
             AND new.superseded_at IS NULL
             AND new.superseded_reason IS NULL)
            OR (new.is_current = 0
                AND new.superseded_at IS NOT NULL
                AND length(trim(new.superseded_reason)) > 0)
        )
    );
    SELECT RAISE(ABORT, 'retired kg evidence revisions are immutable')
    WHERE old.is_current = 0 AND new.is_current = 1;
    SELECT RAISE(ABORT, 'canonical kg evidence provenance mismatch')
    WHERE new.provenance_status = 'canonical' AND NOT (
        new.source_message_id IS NOT NULL
        AND new.source_session_id IS NOT NULL
        AND length(trim(new.source_session_id)) > 0
        AND new.source_role IN ('user','assistant','system','tool')
        AND new.source_event_at IS COALESCE(
            hymem_normalize_iso_timestamp(new.source_created_at),
            '0001-01-01T00:00:00.000Z')
        AND new.source_coverage_chunk_id IS NOT NULL
        AND new.source_coverage_version = 'dream-lossless-message-v1'
        AND EXISTS (
          SELECT 1
          FROM message_retention_coverage mc
          JOIN sessions source_session ON source_session.id = mc.source_session_id
          JOIN chunks source_chunk ON source_chunk.id = mc.chunk_id
          JOIN chunks extraction_chunk ON extraction_chunk.id = new.chunk_id
          WHERE mc.message_id = new.source_message_id
            AND mc.source_session_id = new.source_session_id
            AND mc.source_role = new.source_role
            AND mc.source_created_at IS new.source_created_at
            AND mc.chunk_id = new.source_coverage_chunk_id
            AND mc.coverage_version = new.source_coverage_version
            AND source_session.coverage_message_id IS NOT NULL
            AND mc.message_id <= source_session.coverage_message_id
            AND source_chunk.chunk_kind = 'coverage'
            AND source_chunk.id = hymem_coverage_chunk_id(
                  mc.source_session_id, mc.message_id
                )
            AND source_chunk.session_id = mc.source_session_id
            AND source_chunk.start_message_id = mc.message_id
            AND source_chunk.end_message_id = mc.message_id
            AND json_valid(source_chunk.text)
            AND json_extract(source_chunk.text, '$.id') = mc.message_id
            AND json_extract(source_chunk.text, '$.role') = mc.source_role
            AND json_extract(source_chunk.text, '$.record_version') = mc.record_version
            AND hymem_message_record_proof_valid(
                  source_chunk.text, mc.message_content_hash,
                  mc.hash_version, mc.record_version
                ) = 1
            AND extraction_chunk.chunk_kind = 'extraction'
            AND extraction_chunk.session_id = new.source_session_id
            AND extraction_chunk.source_manifest_version = 'claim-source-manifest-v1'
            AND extraction_chunk.source_manifest_count > 0
            AND new.source_message_id BETWEEN extraction_chunk.start_message_id
                                          AND extraction_chunk.end_message_id
            AND EXISTS (
              SELECT 1 FROM chunk_message_sources cms
              WHERE cms.chunk_id = extraction_chunk.id
                AND cms.source_message_id = new.source_message_id
                AND cms.source_session_id = new.source_session_id
                AND cms.source_coverage_chunk_id = new.source_coverage_chunk_id
                AND cms.source_coverage_version = new.source_coverage_version
            )
        )
      );
    SELECT RAISE(ABORT, 'legacy kg evidence must remain unattributed')
    WHERE new.provenance_status = 'legacy_unattributed' AND NOT (
        new.source_message_id IS NULL
        AND new.source_session_id IS NULL
        AND new.source_role IS NULL
        AND new.source_created_at IS NULL
        AND new.source_event_at IS NULL
        AND new.source_coverage_chunk_id IS NULL
        AND new.source_coverage_version IS NULL
      );
    SELECT RAISE(ABORT, 'canonical kg evidence provenance is immutable')
    WHERE hymem_evidence_mutation_authorized() <> 1
      AND old.provenance_status = 'canonical' AND (
        new.source_message_id IS NOT old.source_message_id
        OR new.source_session_id IS NOT old.source_session_id
        OR new.source_role IS NOT old.source_role
        OR new.source_created_at IS NOT old.source_created_at
        OR new.source_event_at IS NOT old.source_event_at
        OR new.source_coverage_chunk_id IS NOT old.source_coverage_chunk_id
        OR new.source_coverage_version IS NOT old.source_coverage_version
        OR new.provenance_status IS NOT old.provenance_status
        OR new.chunk_id IS NOT old.chunk_id
        OR new.revision IS NOT old.revision
      );
END;
CREATE TRIGGER IF NOT EXISTS kg_evidence_v40_delete_guard
BEFORE DELETE ON kg_evidence
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'kg evidence revisions are internally managed');
END;

-- Non-chunk signals participate in the same cached confidence and lifecycle
-- state as claim evidence.  They therefore cannot be safely mutated by raw
-- SQL without the owning helper reconciling both views atomically.
CREATE TRIGGER IF NOT EXISTS kg_evidence_signals_v40_insert_guard
BEFORE INSERT ON kg_evidence_signals
WHEN hymem_evidence_mutation_authorized() <> 1
  OR (new.signal_kind = 'manual_retraction' AND new.polarity <> -1)
BEGIN
    SELECT RAISE(ABORT, 'kg evidence signals are internally managed');
END;
CREATE TRIGGER IF NOT EXISTS kg_evidence_signals_v40_update_guard
BEFORE UPDATE ON kg_evidence_signals
WHEN hymem_evidence_mutation_authorized() <> 1
  OR (new.signal_kind = 'manual_retraction' AND new.polarity <> -1)
BEGIN
    SELECT RAISE(ABORT, 'kg evidence signals are internally managed');
END;
CREATE TRIGGER IF NOT EXISTS kg_evidence_signals_v40_delete_guard
BEFORE DELETE ON kg_evidence_signals
WHEN hymem_evidence_mutation_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'kg evidence signals are internally managed');
END;

CREATE TRIGGER IF NOT EXISTS kg_evidence_count_after_delete
AFTER DELETE ON kg_evidence
BEGIN
    UPDATE knowledge_graph
    SET pos_evidence = MAX(0, pos_evidence - old.evidence_weight)
    WHERE id = old.edge_id AND derived = 0 AND old.polarity = 1
      AND old.is_current = 1;
    UPDATE knowledge_graph
    SET neg_evidence = MAX(0, neg_evidence - old.evidence_weight)
    WHERE id = old.edge_id AND derived = 0 AND old.polarity = -1
      AND old.is_current = 1;
END;

CREATE TRIGGER IF NOT EXISTS chunk_source_manifest_header_update_guard
BEFORE UPDATE OF source_manifest_version, source_manifest_count ON chunks
WHEN old.source_manifest_version IS NOT NULL
 AND (new.source_manifest_version IS NOT old.source_manifest_version
      OR new.source_manifest_count IS NOT old.source_manifest_count)
 AND NOT (
      new.source_manifest_version IS NULL
      AND new.source_manifest_count IS NULL
      AND NOT EXISTS (SELECT 1 FROM kg_evidence ev WHERE ev.chunk_id = old.id)
      AND NOT EXISTS (
          SELECT 1 FROM kg_claim_observations observation
          WHERE observation.chunk_id = old.id
      )
 )
BEGIN
    SELECT RAISE(ABORT, 'published chunk source manifest header is immutable');
END;
