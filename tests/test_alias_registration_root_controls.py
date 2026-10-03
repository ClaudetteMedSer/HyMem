"""Independent source-valid Phase 1 controls for an existing alias no-op.

Historical alias ownership is deliberately retained, not repaired or merged.
All providers are absent; no extraction or embedding network calls occur.
"""

from dataclasses import replace

import pytest

from hymem.config import HyMemConfig
from hymem.core import db
from hymem.dreaming import canonicalize, phase1
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.triples import Triple


@pytest.fixture
def alias_state(tmp_path):
    conn = db.connect(tmp_path / "alias-root-control.sqlite")
    db.initialize(conn)
    content = "Service uses Legacy Store. Legacy Store uses Redis."
    with db.transaction(conn):
        conn.execute("INSERT INTO sessions(id) VALUES ('alias-fixture')")
        mid = int(conn.execute(
            "INSERT INTO messages(session_id,role,content,created_at) "
            "VALUES ('alias-fixture','user',?,'2026-06-01T12:00:00Z')",
            (content,),
        ).lastrowid)
        materialize_message_coverage(conn, "alias-fixture")
        chunks = [Chunk(
            name, "alias-fixture", mid, mid, "test", "user: " + content,
            source_message_ids=(mid,),
        ) for name in ("alias-history", "alias-current")]
        persist_chunks(conn, chunks)
        canonicalize.register_alias(conn, "Legacy Store", "canonical_store")
        # These are normalized, schema-valid historical owners. Registering
        # this mapping again cannot be the action that strands them.
        conn.executemany(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,neg_evidence,status) "
            "VALUES (?,'uses',?,1,0,'active')",
            [("service", "legacy_store"), ("legacy_store", "redis")],
        )
        conn.execute(
            "INSERT INTO entity_types(entity_canonical,type,confidence,source_chunk_id) "
            "VALUES ('legacy_store','tool',1.0,'alias-history')"
        )
        conn.execute(
            "INSERT INTO entity_mentions(chunk_id,entity_canonical) "
            "VALUES ('alias-history','legacy_store')"
        )
    try:
        yield conn, mid, chunks[1], replace(
            HyMemConfig(root=tmp_path), triple_dedup_enabled=False,
        )
    finally:
        conn.close()


def _owners(conn):
    return {
        "aliases": [tuple(row) for row in conn.execute(
            "SELECT * FROM entity_aliases WHERE alias='legacy_store' OR canonical='legacy_store' ORDER BY alias"
        )],
        "edges": [tuple(row) for row in conn.execute(
            "SELECT * FROM knowledge_graph WHERE subject_canonical='legacy_store' "
            "OR object_canonical='legacy_store' ORDER BY id"
        )],
        "types": [tuple(row) for row in conn.execute(
            "SELECT * FROM entity_types WHERE entity_canonical='legacy_store' ORDER BY type"
        )],
        "mentions": [tuple(row) for row in conn.execute(
            "SELECT * FROM entity_mentions WHERE entity_canonical='legacy_store' ORDER BY chunk_id"
        )],
    }


def _persist(conn, chunk, triple, cfg):
    sources = phase1._claim_sources_for_chunk(conn, chunk)
    assert sources and all(source.message_id == triple.source_message_id for source in sources)
    extraction = phase1.ChunkExtraction(
        triples=[triple], markers=[], source_validated=True,
        claim_sources={source.message_id: source for source in sources},
    )
    with db.transaction(conn):
        phase1.persist_chunk_results(
            conn, chunk, extraction, prompt_version="v14", cfg=cfg,
        )


def test_exact_existing_mapping_is_no_dml_even_with_historical_owners(alias_state):
    conn, _mid, _chunk, _cfg = alias_state
    before, changes = tuple(conn.iterdump()), conn.total_changes
    assert canonicalize.resolve(conn, "Legacy Store") == "canonical_store"
    canonicalize.register_alias(conn, "Legacy Store", "canonical_store")
    canonicalize.register_alias(conn, "legacy_store", "canonical_store")
    assert tuple(conn.iterdump()) == before
    assert conn.total_changes == changes


@pytest.mark.parametrize("position", ["subject", "object"])
def test_real_phase1_existing_alias_does_not_rewrite_historical_owners(alias_state, position):
    conn, mid, chunk, cfg = alias_state
    before = _owners(conn)
    triple = (
        Triple("Legacy Store", "uses", "redis", 1, source_message_id=mid)
        if position == "subject" else
        Triple("service", "uses", "Legacy Store", 1, source_message_id=mid)
    )
    _persist(conn, chunk, triple, cfg)
    assert _owners(conn) == before
    selected = conn.execute(
        "SELECT kg.subject_canonical,kg.predicate,kg.object_canonical,"
        "observation.source_message_id,observation.polarity "
        "FROM kg_claim_observations observation "
        "JOIN knowledge_graph kg ON kg.id=observation.edge_id "
        "WHERE observation.chunk_id=?", (chunk.id,),
    ).fetchall()
    expected_edge = (("canonical_store", "uses", "redis") if position == "subject"
                     else ("service", "uses", "canonical_store"))
    assert [tuple(row) for row in selected] == [(*expected_edge, mid, 1)]
    assert conn.execute(
        "SELECT count(*) FROM current_phase1_publications WHERE chunk_id=?", (chunk.id,),
    ).fetchone()[0] == 1
    published = tuple(conn.iterdump())
    _persist(conn, chunk, triple, cfg)
    assert tuple(conn.iterdump()) == published
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("surface,target,reason", [
    ("service", "new_service", "already owns state"),
    ("Legacy Store", "other_store", "already maps"),
    ("new surface", "legacy_store", "must not itself be an alias"),
    ("Legacy Store", "Not Normalized", "normalized canonical"),
    ("!!!", "canonical_store", "must not be empty"),
])
def test_non_noop_guards_still_reject_without_any_mutation(alias_state, surface, target, reason):
    conn, _mid, _chunk, _cfg = alias_state
    before, changes = tuple(conn.iterdump()), conn.total_changes
    with pytest.raises(ValueError, match=reason):
        canonicalize.register_alias(conn, surface, target)
    assert tuple(conn.iterdump()) == before
    assert conn.total_changes == changes


def test_existing_same_mapping_cannot_bypass_target_chain_validation(alias_state):
    conn, _mid, _chunk, _cfg = alias_state
    conn.execute(
        "INSERT INTO entity_aliases(alias,canonical) VALUES ('older_surface','legacy_store')"
    )
    before, changes = tuple(conn.iterdump()), conn.total_changes
    with pytest.raises(ValueError, match="must not itself be an alias"):
        canonicalize.register_alias(conn, "Older Surface", "legacy_store")
    assert tuple(conn.iterdump()) == before
    assert conn.total_changes == changes

