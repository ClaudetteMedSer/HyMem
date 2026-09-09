-- Admit only normalization fixed points for new scalar canonical owners.
-- No historical rows are rewritten. UPDATE validates each changed field
-- independently, allowing an explicit repair to fix two legacy endpoints
-- in successive statements without authorizing new malformed values.
-- IS NOT 1 is intentional: NULL must fail closed, including custom UDFs.

CREATE TRIGGER IF NOT EXISTS entity_aliases_canonical_insert_guard
BEFORE INSERT ON entity_aliases
WHEN hymem_entity_canonical_is_normalized(new.alias) IS NOT 1
  OR hymem_entity_canonical_is_normalized(new.canonical) IS NOT 1
BEGIN
    SELECT RAISE(ABORT, 'canonical identity must be normalized');
END;

CREATE TRIGGER IF NOT EXISTS entity_aliases_canonical_update_guard
BEFORE UPDATE OF alias, canonical ON entity_aliases
WHEN (new.alias IS NOT old.alias AND hymem_entity_canonical_is_normalized(new.alias) IS NOT 1)
  OR (new.canonical IS NOT old.canonical AND hymem_entity_canonical_is_normalized(new.canonical) IS NOT 1)
BEGIN
    SELECT RAISE(ABORT, 'canonical identity must be normalized');
END;

CREATE TRIGGER IF NOT EXISTS knowledge_graph_canonical_insert_guard
BEFORE INSERT ON knowledge_graph
WHEN hymem_entity_canonical_is_normalized(new.subject_canonical) IS NOT 1
  OR hymem_entity_canonical_is_normalized(new.object_canonical) IS NOT 1
BEGIN
    SELECT RAISE(ABORT, 'canonical identity must be normalized');
END;

CREATE TRIGGER IF NOT EXISTS knowledge_graph_canonical_update_guard
BEFORE UPDATE OF subject_canonical, object_canonical ON knowledge_graph
WHEN (new.subject_canonical IS NOT old.subject_canonical AND hymem_entity_canonical_is_normalized(new.subject_canonical) IS NOT 1)
  OR (new.object_canonical IS NOT old.object_canonical AND hymem_entity_canonical_is_normalized(new.object_canonical) IS NOT 1)
BEGIN
    SELECT RAISE(ABORT, 'canonical identity must be normalized');
END;

CREATE TRIGGER IF NOT EXISTS entity_mentions_canonical_insert_guard
BEFORE INSERT ON entity_mentions
WHEN hymem_entity_canonical_is_normalized(new.entity_canonical) IS NOT 1
BEGIN
    SELECT RAISE(ABORT, 'canonical identity must be normalized');
END;

CREATE TRIGGER IF NOT EXISTS entity_mentions_canonical_update_guard
BEFORE UPDATE OF entity_canonical ON entity_mentions
WHEN (new.entity_canonical IS NOT old.entity_canonical AND hymem_entity_canonical_is_normalized(new.entity_canonical) IS NOT 1)
BEGIN
    SELECT RAISE(ABORT, 'canonical identity must be normalized');
END;

