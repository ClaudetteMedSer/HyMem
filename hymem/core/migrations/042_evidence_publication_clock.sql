-- v42: immutable first-publication transaction time for canonical evidence.
--
-- Observations and the per-chunk outcome describe the latest successful
-- extraction. Reusing those mutable rows as historical authority erased prior
-- knowledge after an unchanged re-extraction. `published_at` is set once only
-- after the complete chunk outcome succeeds and survives later replays.

ALTER TABLE kg_evidence ADD COLUMN published_at TIMESTAMP;

-- Current v41 rows are published only when their exact observation still has
-- a coherent whole-chunk outcome.  Using extracted_at alone would launder a
-- staged/orphan row into historical authority. Retired rows without a
-- surviving mutable observation use superseded_at as an explicit zero-width
-- fallback: it makes the history portable without asserting that the claim
-- was visible at any earlier transaction cutoff.
UPDATE kg_evidence
SET published_at = (
    SELECT MIN(hymem_normalize_iso_timestamp(outcome.succeeded_at))
    FROM kg_claim_observations observation
    JOIN kg_claim_extraction_outcomes outcome
      ON outcome.chunk_id = observation.chunk_id
     AND outcome.prompt_version = observation.prompt_version
     AND outcome.prompt_generation = observation.prompt_generation
    WHERE observation.evidence_id = kg_evidence.id
      AND observation.edge_id = kg_evidence.edge_id
      AND observation.source_session_id = kg_evidence.source_session_id
      AND observation.source_message_id = kg_evidence.source_message_id
      AND observation.evidence_kind = kg_evidence.evidence_kind
      AND observation.polarity = kg_evidence.polarity
      AND observation.interpretation_key = kg_evidence.interpretation_key
      AND hymem_event_clock_is_valid(
            kg_evidence.source_event_at, kg_evidence.extracted_at
          ) = 1
      AND hymem_timestamp_at_or_before(
            kg_evidence.extracted_at, observation.observed_at
          ) = 1
      AND hymem_timestamp_gap_within(
            observation.observed_at, outcome.succeeded_at, 300
          ) = 1
      AND hymem_timestamp_at_or_before(
            outcome.succeeded_at,
            strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
          ) = 1
      AND (
        kg_evidence.superseded_at IS NULL
        OR hymem_timestamp_at_or_before(
             outcome.succeeded_at, kg_evidence.superseded_at
           ) = 1
      )
      AND (
        kg_evidence.polarity = -1
        OR EXISTS (
          SELECT 1 FROM kg_edge_lifecycle lifecycle
          WHERE lifecycle.source_evidence_id = kg_evidence.id
            AND lifecycle.edge_id = kg_evidence.edge_id
            AND lifecycle.event_kind = 'claim_assertion'
            AND lifecycle.direction = 1
            AND lifecycle.event_at = kg_evidence.source_event_at
            AND hymem_timestamp_at_or_before(
                  kg_evidence.extracted_at, lifecycle.created_at
                ) = 1
            AND hymem_timestamp_at_or_before(
                  lifecycle.created_at, outcome.succeeded_at
                ) = 1
        )
      )
)
WHERE provenance_status = 'canonical'
  AND EXISTS (
    SELECT 1
    FROM kg_claim_observations observation
    JOIN kg_claim_extraction_outcomes outcome
      ON outcome.chunk_id = observation.chunk_id
     AND outcome.prompt_version = observation.prompt_version
     AND outcome.prompt_generation = observation.prompt_generation
    WHERE observation.evidence_id = kg_evidence.id
      AND observation.edge_id = kg_evidence.edge_id
      AND observation.source_session_id = kg_evidence.source_session_id
      AND observation.source_message_id = kg_evidence.source_message_id
      AND observation.evidence_kind = kg_evidence.evidence_kind
      AND observation.polarity = kg_evidence.polarity
      AND observation.interpretation_key = kg_evidence.interpretation_key
      AND hymem_event_clock_is_valid(
            kg_evidence.source_event_at, kg_evidence.extracted_at
          ) = 1
      AND hymem_timestamp_at_or_before(
            kg_evidence.extracted_at, observation.observed_at
          ) = 1
      AND hymem_timestamp_gap_within(
            observation.observed_at, outcome.succeeded_at, 300
          ) = 1
      AND hymem_timestamp_at_or_before(
            outcome.succeeded_at,
            strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
          ) = 1
      AND (
        kg_evidence.superseded_at IS NULL
        OR hymem_timestamp_at_or_before(
             outcome.succeeded_at, kg_evidence.superseded_at
           ) = 1
      )
      AND (
        kg_evidence.polarity = -1
        OR EXISTS (
          SELECT 1 FROM kg_edge_lifecycle lifecycle
          WHERE lifecycle.source_evidence_id = kg_evidence.id
            AND lifecycle.edge_id = kg_evidence.edge_id
            AND lifecycle.event_kind = 'claim_assertion'
            AND lifecycle.direction = 1
            AND lifecycle.event_at = kg_evidence.source_event_at
            AND hymem_timestamp_at_or_before(
                  kg_evidence.extracted_at, lifecycle.created_at
                ) = 1
            AND hymem_timestamp_at_or_before(
                  lifecycle.created_at, outcome.succeeded_at
                ) = 1
        )
      )
  );

UPDATE kg_evidence
SET published_at = hymem_normalize_iso_timestamp(superseded_at)
WHERE provenance_status = 'canonical'
  AND is_current = 0
  AND published_at IS NULL
  AND hymem_event_clock_is_valid(source_event_at, extracted_at) = 1
  AND hymem_timestamp_at_or_before(extracted_at, superseded_at) = 1
  AND hymem_timestamp_at_or_before(
        superseded_at,
        strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
      ) = 1
  AND (
    polarity = -1
    OR EXISTS (
      SELECT 1 FROM kg_edge_lifecycle lifecycle
      WHERE lifecycle.source_evidence_id = kg_evidence.id
        AND lifecycle.edge_id = kg_evidence.edge_id
        AND lifecycle.event_kind = 'claim_assertion'
        AND lifecycle.direction = 1
        AND lifecycle.event_at = kg_evidence.source_event_at
        AND hymem_timestamp_at_or_before(
              kg_evidence.extracted_at, lifecycle.created_at
            ) = 1
        AND hymem_timestamp_at_or_before(
              lifecycle.created_at, kg_evidence.superseded_at
            ) = 1
    )
  );

-- Startup reinstalls the canonical v40 revision guards from their source
-- definition with the shared timestamp UDF. Dropping here upgrades stores
-- whose existing trigger still used SQLite's divergent strftime grammar.
DROP TRIGGER IF EXISTS kg_evidence_v40_insert_guard;
DROP TRIGGER IF EXISTS kg_evidence_v40_update_guard;

DROP TRIGGER IF EXISTS kg_evidence_published_at_insert_guard;
CREATE TRIGGER kg_evidence_published_at_insert_guard
BEFORE INSERT ON kg_evidence
WHEN new.published_at IS NOT NULL AND NOT (
    hymem_evidence_history_authorized() = 1
    AND new.provenance_status = 'canonical'
    AND new.published_at = hymem_normalize_iso_timestamp(new.published_at)
    AND hymem_event_clock_is_valid(
          new.source_event_at, new.extracted_at
        ) = 1
    AND hymem_timestamp_at_or_before(new.extracted_at, new.published_at) = 1
    AND hymem_timestamp_at_or_before(
          new.published_at,
          strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
        ) = 1
    AND (
      new.superseded_at IS NULL
      OR hymem_timestamp_at_or_before(
           new.published_at, new.superseded_at
         ) = 1
    )
)
BEGIN
    SELECT RAISE(ABORT, 'evidence publication clock is internally managed');
END;

DROP TRIGGER IF EXISTS kg_evidence_published_at_update_guard;
CREATE TRIGGER kg_evidence_published_at_update_guard
BEFORE UPDATE OF published_at, extracted_at, provenance_status,
                 superseded_at, superseded_reason, is_current,
                 chunk_id, edge_id, polarity,
                 surface_subject, surface_object, value_text, value_numeric,
                 value_unit, temporal_scope, source_role, evidence_kind,
                 evidence_weight, weight_source, extraction_prompt_version,
                 source_message_id, source_session_id, source_created_at,
                 source_event_at, source_coverage_chunk_id,
                 source_coverage_version, interpretation_key, revision
ON kg_evidence
BEGIN
    SELECT RAISE(ABORT, 'retired evidence publication is immutable')
    WHERE old.is_current = 0
      AND hymem_evidence_history_authorized() <> 1
      AND (
        new.is_current IS NOT old.is_current
        OR new.superseded_at IS NOT old.superseded_at
        OR new.superseded_reason IS NOT old.superseded_reason
      );
    SELECT RAISE(ABORT, 'published evidence audit is immutable')
    WHERE old.published_at IS NOT NULL
      AND hymem_evidence_history_authorized() <> 1
      AND (
        new.chunk_id IS NOT old.chunk_id
        OR new.edge_id IS NOT old.edge_id
        OR new.polarity IS NOT old.polarity
        OR new.surface_subject IS NOT old.surface_subject
        OR new.surface_object IS NOT old.surface_object
        OR new.value_text IS NOT old.value_text
        OR new.value_numeric IS NOT old.value_numeric
        OR new.value_unit IS NOT old.value_unit
        OR new.temporal_scope IS NOT old.temporal_scope
        OR new.source_role IS NOT old.source_role
        OR new.evidence_kind IS NOT old.evidence_kind
        OR new.evidence_weight IS NOT old.evidence_weight
        OR new.weight_source IS NOT old.weight_source
        OR new.extraction_prompt_version IS NOT
           old.extraction_prompt_version
        OR new.extracted_at IS NOT old.extracted_at
        OR new.source_message_id IS NOT old.source_message_id
        OR new.source_session_id IS NOT old.source_session_id
        OR new.source_created_at IS NOT old.source_created_at
        OR new.source_event_at IS NOT old.source_event_at
        OR new.source_coverage_chunk_id IS NOT
           old.source_coverage_chunk_id
        OR new.source_coverage_version IS NOT old.source_coverage_version
        OR new.provenance_status IS NOT old.provenance_status
        OR new.interpretation_key IS NOT old.interpretation_key
        OR new.revision IS NOT old.revision
      );
    SELECT RAISE(ABORT, 'evidence publication clock is immutable')
    WHERE new.published_at IS NOT old.published_at AND NOT (
      new.published_at IS NOT NULL
      AND (
        (old.published_at IS NULL
         AND hymem_evidence_mutation_authorized() = 1)
        OR hymem_evidence_history_authorized() = 1
      )
    );
    SELECT RAISE(ABORT, 'evidence publication clock is incoherent')
    WHERE new.published_at IS NOT NULL AND NOT (
      new.provenance_status = 'canonical'
      AND new.published_at = hymem_normalize_iso_timestamp(new.published_at)
      AND hymem_event_clock_is_valid(
            new.source_event_at, new.extracted_at
          ) = 1
      AND hymem_timestamp_at_or_before(
            new.extracted_at, new.published_at
          ) = 1
      AND hymem_timestamp_at_or_before(
            new.published_at,
            strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
          ) = 1
      AND (
        new.superseded_at IS NULL
        OR hymem_timestamp_at_or_before(
             new.published_at, new.superseded_at
           ) = 1
      )
      AND (
        hymem_evidence_history_authorized() = 1
        OR new.polarity = -1
        OR EXISTS (
          SELECT 1 FROM kg_edge_lifecycle lifecycle
          WHERE lifecycle.source_evidence_id = new.id
            AND lifecycle.edge_id = new.edge_id
            AND lifecycle.event_kind = 'claim_assertion'
            AND lifecycle.direction = 1
            AND lifecycle.event_at = new.source_event_at
            AND hymem_timestamp_at_or_before(
                  new.extracted_at, lifecycle.created_at
                ) = 1
            AND hymem_timestamp_at_or_before(
                  lifecycle.created_at, new.published_at
          ) = 1
        )
      )
      AND (
        hymem_evidence_history_authorized() = 1
        OR new.published_at IS old.published_at
        OR EXISTS (
          SELECT 1
          FROM kg_claim_observations observation
          JOIN kg_claim_extraction_outcomes outcome
            ON outcome.chunk_id = observation.chunk_id
           AND outcome.prompt_version = observation.prompt_version
           AND outcome.prompt_generation = observation.prompt_generation
          WHERE observation.evidence_id = new.id
            AND observation.edge_id = new.edge_id
            AND observation.source_session_id = new.source_session_id
            AND observation.source_message_id = new.source_message_id
            AND observation.evidence_kind = new.evidence_kind
            AND observation.polarity = new.polarity
            AND observation.interpretation_key = new.interpretation_key
            AND new.published_at =
                hymem_normalize_iso_timestamp(outcome.succeeded_at)
            AND hymem_timestamp_at_or_before(
                  new.extracted_at, observation.observed_at
                ) = 1
            AND hymem_timestamp_gap_within(
                  observation.observed_at, outcome.succeeded_at, 300
                ) = 1
        )
      )
    );
END;

-- Published claim history may only be removed by a complete-history merge or
-- by the explicitly configured tombstone-retention path.  The ordinary
-- evidence mutation capability is deliberately insufficient: it is used by
-- normal extraction writers and must not double as permission to erase an
-- already-published transaction interval.
DROP TRIGGER IF EXISTS kg_evidence_v40_delete_guard;
CREATE TRIGGER kg_evidence_v40_delete_guard
BEFORE DELETE ON kg_evidence
WHEN hymem_evidence_mutation_authorized() <> 1
  OR (
    old.published_at IS NOT NULL
    AND hymem_evidence_history_authorized() <> 1
    AND hymem_evidence_destructive_authorized() <> 1
  )
BEGIN
    SELECT RAISE(ABORT, 'published kg evidence history is immutable');
END;

-- Lifecycle events and their dependency sets are the other half of the
-- append-only ledger.  Runtime code appends them under evidence_mutation, but
-- only a validated history rewrite (canonical merge/import) or an explicit
-- destructive-retention operation may edit or remove an existing event.
DROP TRIGGER IF EXISTS kg_edge_lifecycle_update_guard;
CREATE TRIGGER kg_edge_lifecycle_update_guard
BEFORE UPDATE ON kg_edge_lifecycle
WHEN hymem_evidence_history_authorized() <> 1
 AND hymem_evidence_destructive_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'knowledge graph lifecycle history is immutable');
END;

DROP TRIGGER IF EXISTS kg_edge_lifecycle_delete_guard;
CREATE TRIGGER kg_edge_lifecycle_delete_guard
BEFORE DELETE ON kg_edge_lifecycle
WHEN hymem_evidence_history_authorized() <> 1
 AND hymem_evidence_destructive_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'knowledge graph lifecycle history is immutable');
END;

DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_update_guard;
CREATE TRIGGER kg_lifecycle_dependencies_update_guard
BEFORE UPDATE ON kg_lifecycle_dependencies
WHEN hymem_evidence_history_authorized() <> 1
 AND hymem_evidence_destructive_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'lifecycle dependency history is immutable');
END;

DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_delete_guard;
CREATE TRIGGER kg_lifecycle_dependencies_delete_guard
BEFORE DELETE ON kg_lifecycle_dependencies
WHEN hymem_evidence_history_authorized() <> 1
 AND hymem_evidence_destructive_authorized() <> 1
BEGIN
    SELECT RAISE(ABORT, 'lifecycle dependency history is immutable');
END;
