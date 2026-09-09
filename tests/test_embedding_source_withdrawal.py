"""Successful-empty source withdrawal is not a negative world assertion."""
from dataclasses import asdict
import json
import sqlite3
import time

import pytest

from hymem import doctor, reembed
from hymem.config import HyMemConfig
from hymem.core import db
from hymem.dreaming import evidence
from hymem.extraction.triples import Triple
from tests.test_claim_provenance import _open, _messages, _chunk, _persist
from tests.test_embedding_source_health import CURRENT, OLD, scan, tier


@pytest.fixture
def withdrawn(tmp_path):
    conn = _open(tmp_path)
    message_id, = _messages(conn, [("user", "The app uses Redis", "2026-02-01T12:00:00Z")])
    chunk = _chunk(conn, "private-synthetic-replay", [message_id])
    cfg = HyMemConfig(root=tmp_path)
    claim = Triple("app", "uses", "redis", 1, source_message_id=message_id)
    _persist(conn, chunk, [claim], prompt_version="v13", cfg=cfg)
    _persist(conn, chunk, [claim], prompt_version="v14", cfg=cfg)
    _persist(conn, chunk, [], prompt_version="v15", cfg=cfg)
    with db.embedding_mutation(conn):
        conn.execute("INSERT INTO edge_embeddings(edge_text,model,dim,vector_json) "
                     "VALUES ('app uses redis',?,3,'[1,0,0]')", (OLD,))
    yield conn, tmp_path / "claim-provenance.sqlite", chunk, cfg, message_id
    conn.close()


def corrupt(conn, sql, params=()):
    """Model existing on-disk corruption, not a supported mutation path."""
    for name, in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
    conn.execute(sql, params)


def test_maintained_empty_publication_is_source_withdrawn_read_only(withdrawn):
    conn, path, *_ = withdrawn
    before = list(conn.iterdump())
    result = scan(path)
    row = tier(result, "edge_embeddings")
    assert result.status == "compatible" and result.inventory.status == "incompatible"
    assert row.source_withdrawn == 1
    assert row.retired == row.unsafe_unknown == row.source_eligible == 0
    detail = doctor._check_stored_embedding_health(conn, 3, CURRENT)[0]
    assert detail.status == doctor.OK
    assert "source-withdrawn=1" in detail.detail and "not a negative fact" in detail.detail
    assert "private" not in json.dumps(asdict(result))
    assert list(conn.iterdump()) == before


@pytest.mark.parametrize("sql", [
    "DELETE FROM kg_claim_extraction_outcomes",
    "UPDATE kg_claim_extraction_outcomes SET result_hash='sha256:forged'",
    "UPDATE kg_claim_extraction_outcomes SET prompt_generation=999",
    "UPDATE kg_claim_extraction_outcomes SET succeeded_at='9999-01-01T00:00:00Z'",
    "UPDATE kg_claim_extraction_outcomes SET succeeded_at='not-a-time'",
    "UPDATE kg_claim_extraction_outcomes SET phase1_generation_key=NULL",
    "UPDATE phase1_generations SET binding_json='{}'",
    "UPDATE phase1_generations SET producer_identity_sha256='forged'",
    "UPDATE phase1_generations SET identity_exact=0",
    "UPDATE phase1_generations SET extraction_cache_key='not-the-outcome'",
    "UPDATE kg_evidence SET superseded_reason='successful_reextract:no_current_authority'",
    "UPDATE kg_evidence SET superseded_reason='successful_reextract:other-receipt'",
    "UPDATE kg_evidence SET superseded_at='2020-01-01T00:00:00Z'",
    "UPDATE kg_evidence SET published_at='9999-01-01T00:00:00Z'",
    "UPDATE kg_evidence SET extracted_at='9999-01-01T00:00:00Z'",
    "UPDATE kg_evidence SET source_event_at='9999-01-01T00:00:00Z'",
    "UPDATE kg_evidence SET interpretation_key='forged-interpretation'",
    "UPDATE kg_evidence SET value_text='forged-value'",
    "UPDATE kg_evidence SET evidence_weight=100",
    "UPDATE kg_evidence SET source_role='assistant'",
    "UPDATE kg_evidence SET source_peer_id='invented-peer'",
    "UPDATE kg_evidence SET source_workspace_id='invented-workspace'",
    "UPDATE kg_evidence SET source_created_at='2020-01-01T00:00:00Z'",
    "UPDATE kg_evidence SET is_current=1",
    "UPDATE kg_evidence SET provenance_status='legacy_unattributed'",
    "UPDATE chunks SET text='broken retained source' WHERE chunk_kind='coverage'",
    "UPDATE chunks SET text='broken winning input' WHERE chunk_kind='extraction'",
    "UPDATE chunks SET source_manifest_count=2 WHERE chunk_kind='extraction'",
    "UPDATE chunks SET created_at='9999-01-01T00:00:00Z' WHERE chunk_kind='extraction'",
    "DELETE FROM chunk_message_sources",
    "UPDATE knowledge_graph SET pos_evidence=1",
    "UPDATE knowledge_graph SET derived=1",
    "UPDATE knowledge_graph SET valid_at='2020-01-01T00:00:00.000Z'",
    "UPDATE knowledge_graph SET invalid_at='2026-02-02T12:00:00.000Z'",
    "DELETE FROM kg_edge_lifecycle",
    "UPDATE kg_edge_lifecycle SET created_at='9999-01-01T00:00:00Z'",
    "UPDATE kg_edge_lifecycle SET created_at='2020-01-01T00:00:00Z'",
    "UPDATE kg_edge_lifecycle SET event_at='2020-01-01T00:00:00.000Z'",
    "UPDATE kg_edge_lifecycle SET event_key='forged-assertion'",
    "UPDATE kg_edge_lifecycle SET dependency_count=1",
    "UPDATE kg_edge_lifecycle SET source_evidence_id=NULL",
    "INSERT INTO kg_lifecycle_dependencies(lifecycle_id,evidence_id) "
    "SELECT lifecycle.id,evidence.id FROM kg_edge_lifecycle lifecycle,kg_evidence evidence LIMIT 1",
])
def test_incomplete_or_corrupt_withdrawal_never_becomes_historical_success(withdrawn, sql):
    conn, path, *_ = withdrawn
    corrupt(conn, sql)
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    row = tier(scan(path), "edge_embeddings")
    assert row.source_withdrawn == row.retired == 0
    assert row.unsafe_unknown == 1
    assert doctor._check_stored_embedding_health(conn, 3, CURRENT)[0].status == doctor.FAIL


def test_retained_source_integrity_failure_is_not_excused_by_empty_outcome(withdrawn):
    from hymem.dreaming.lossless import record_coverage_integrity_failure
    conn, path, *_ = withdrawn
    record_coverage_integrity_failure(conn, "s", reason="source_stream_invalid")
    row = tier(scan(path), "edge_embeddings")
    assert row.unsafe_unknown == 1 and row.source_withdrawn == 0


def test_overwritten_successful_receipt_cannot_prove_prior_withdrawal(withdrawn):
    conn, path, chunk, cfg, _ = withdrawn
    _persist(conn, chunk, [], prompt_version="v16", cfg=cfg)
    row = tier(scan(path), "edge_embeddings")
    assert row.unsafe_unknown == 1 and row.source_withdrawn == 0


def test_source_unqualified_empty_is_not_withdrawal(withdrawn):
    conn, path, *_ = withdrawn
    corrupt(conn, "UPDATE kg_evidence SET provenance_status='legacy_unattributed',source_message_id=NULL,"
                  "source_session_id=NULL,source_coverage_chunk_id=NULL,source_coverage_version=NULL")
    assert tier(scan(path), "edge_embeddings").unsafe_unknown == 1


def test_positive_signal_invalidates_empty_source_closure(withdrawn):
    conn, path, *_ = withdrawn
    edge_id = conn.execute("SELECT id FROM knowledge_graph").fetchone()[0]
    evidence.record_signal(conn, edge_id=edge_id, signal_key="positive", signal_kind="runtime_unattributed",
                           polarity=1, evidence_weight=1)
    row = tier(scan(path), "edge_embeddings")
    assert row.unsafe_unknown == 1 and row.source_withdrawn == 0


def test_withdrawn_source_proof_cannot_write_and_budget_failure_is_unknown(withdrawn, monkeypatch):
    from hymem import embedding_source_health as health
    conn, path, *_ = withdrawn
    original = reembed._source

    def readonly(reader, index, row):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.execute("UPDATE knowledge_graph SET pos_evidence=20")
        return original(reader, index, row)

    monkeypatch.setattr(reembed, "_source", readonly)
    assert tier(scan(path), "edge_embeddings").source_withdrawn == 1
    monkeypatch.setattr(reembed, "_source", lambda *_: (_ for _ in ()).throw(health._AuditLimit("source_proof_budget_exhausted")))
    result = scan(path)
    row = tier(result, "edge_embeddings")
    assert result.status == "unavailable"
    assert row.source_withdrawn is None and row.unsafe_unknown is None
    assert row.error_code == "source_proof_budget_exhausted"


def test_all_retained_revisions_are_validated_not_only_final_receipt(withdrawn):
    conn, path, chunk, cfg, message_id = withdrawn
    claim = Triple("app", "uses", "redis", 1, source_message_id=message_id)
    _persist(conn, chunk, [claim], prompt_version="v16", cfg=cfg)
    _persist(conn, chunk, [], prompt_version="v17", cfg=cfg)
    assert conn.execute("SELECT count(*) FROM kg_evidence").fetchone()[0] > 1
    assert tier(scan(path), "edge_embeddings").source_withdrawn == 1
    corrupt(conn, "UPDATE kg_evidence SET extracted_at='broken' WHERE revision=1")
    assert tier(scan(path), "edge_embeddings").unsafe_unknown == 1


def test_each_source_identity_needs_its_own_winning_empty_receipt(withdrawn):
    conn, path, _, cfg, _ = withdrawn
    mid, = _messages(conn, [("assistant", "The app also uses Redis", "2026-02-02T12:00:00Z")])
    second = _chunk(conn, "second-source", [mid])
    _persist(conn, second, [Triple("app", "uses", "redis", 1, source_message_id=mid)], prompt_version="v16", cfg=cfg)
    _persist(conn, second, [], prompt_version="v17", cfg=cfg)
    assert tier(scan(path), "edge_embeddings").source_withdrawn == 1
    corrupt(conn, "DELETE FROM kg_claim_extraction_outcomes WHERE chunk_id=?", (second.id,))
    row = tier(scan(path), "edge_embeddings")
    assert row.unsafe_unknown == 1 and row.source_withdrawn == 0


def test_nonempty_winning_result_is_outside_narrow_empty_withdrawal_proof(withdrawn):
    conn, path, chunk, cfg, mid = withdrawn
    _persist(conn, chunk, [Triple("other", "uses", "redis", 1, source_message_id=mid)],
             prompt_version="v16", cfg=cfg)
    # Model matching retirement bookkeeping while retaining real nonempty
    # winning observations. Missing evidence for this owner is insufficient.
    retired_at, reason = evidence.claim_retirement_authority(conn, source_session_id="s", source_message_id=mid)
    corrupt(conn, "UPDATE kg_evidence SET superseded_at=?,superseded_reason=? WHERE is_current=0",
            (retired_at, reason))
    assert tier(scan(path), "edge_embeddings").unsafe_unknown == 1


def test_shared_legacy_text_requires_every_owner_closed(withdrawn):
    from hymem.dreaming.bitemporal import record_lifecycle_event
    conn, path, *_ = withdrawn
    # Two historical natural tuples can share the old space-joined mirror key.
    # Current canonical admission correctly forbids creating these spellings.
    corrupt(conn, "UPDATE knowledge_graph SET object_canonical='redis uses cache'")
    conn.execute("UPDATE edge_embeddings SET edge_text='app uses redis uses cache'")
    other = conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,status,first_seen) "
                         "VALUES ('app uses redis','uses','cache','retracted','2020-01-01T00:00:00.000Z')").lastrowid
    row = tier(scan(path), "edge_embeddings")
    assert row.unsafe_unknown == 1 and row.source_withdrawn == 0
    record_lifecycle_event(conn, edge_id=other, event_key="legacy-state", event_kind="legacy_state",
                           direction=-1, event_at="2020-01-01T00:00:00.000Z")
    row = tier(scan(path), "edge_embeddings")
    assert row.source_withdrawn == 1 and row.retired == row.unsafe_unknown == 0


def test_proof_fetch_byte_bound_is_unknown_without_partial_success(withdrawn):
    conn, path, *_ = withdrawn
    # Each SQLite value is below the individual 8 MiB limit, but a single
    # proof's combined retained audit payload exceeds its bounded allowance.
    corrupt(conn, "UPDATE kg_evidence SET surface_subject=?,surface_object=?", ("x" * 4_200_000, "y" * 4_200_000))
    row = tier(scan(path), "edge_embeddings")
    assert row.source_withdrawn is None and row.unsafe_unknown is None
    assert row.error_code == "source_proof_budget_exhausted"


def test_compatible_withdrawn_vectors_skip_expensive_producer_proofs(withdrawn, monkeypatch):
    from hymem.extraction import producer
    conn, path, *_ = withdrawn
    with db.embedding_mutation(conn):
        conn.execute("UPDATE edge_embeddings SET model=?", (CURRENT,))
    monkeypatch.setattr(producer, "phase1_generation_registry_row_is_valid", lambda *_: pytest.fail("unneeded producer proof"))
    row = tier(scan(path), "edge_embeddings")
    assert row.total == row.source_withdrawn == 0


def test_consistently_forged_source_event_cannot_replace_exact_occurrence_clock(withdrawn):
    conn, path, *_ = withdrawn
    corrupt(conn, "UPDATE kg_evidence SET source_event_at='2020-01-01T00:00:00.000Z'")
    conn.execute("UPDATE kg_edge_lifecycle SET event_at='2020-01-01T00:00:00.000Z'")
    conn.execute("UPDATE knowledge_graph SET valid_at='2020-01-01T00:00:00.000Z',invalid_at='2020-01-01T00:00:00.000Z'")
    assert tier(scan(path), "edge_embeddings").unsafe_unknown == 1


def test_winning_receipt_cannot_precede_its_source_artifact(withdrawn):
    conn, path, *_ = withdrawn
    # Keep the finalizer's max(publication, outcome) retirement relationship
    # coherent. The independent artifact/publication clock still rejects this.
    corrupt(conn, "UPDATE kg_claim_extraction_outcomes SET succeeded_at='2020-01-01T00:00:00Z'")
    conn.execute("UPDATE kg_evidence SET superseded_at=published_at")
    assert tier(scan(path), "edge_embeddings").unsafe_unknown == 1


def test_valid_artifact_with_unrecognized_canonical_coverage_version_is_unknown(withdrawn):
    conn, path, *_ = withdrawn
    corrupt(conn, "SELECT 1")
    columns = [row[1] for row in conn.execute("PRAGMA table_info(message_retention_coverage)")]
    projection = ["'unrecognized-v1'" if name == "coverage_version" else name for name in columns]
    conn.execute("INSERT INTO message_retention_coverage(" + ",".join(columns) + ") SELECT "
                 + ",".join(projection) + " FROM message_retention_coverage")
    conn.execute("UPDATE kg_evidence SET source_coverage_version='unrecognized-v1'")
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    assert tier(scan(path), "edge_embeddings").unsafe_unknown == 1


def later_empty_overlap(withdrawn, *, prompt_version="v15"):
    conn, _, original, cfg, mid = withdrawn
    original_receipt = dict(conn.execute("SELECT * FROM kg_claim_extraction_outcomes WHERE chunk_id=?", (original.id,)).fetchone())
    # Keep the entire proof on maintained calls and actual SQLite clocks.
    time.sleep(1.05)
    later = _chunk(conn, "later-empty-source", [mid])
    _persist(conn, later, [], prompt_version=prompt_version, cfg=cfg)
    return original_receipt, later


def test_later_same_binding_empty_preserves_original_retirement_proof(withdrawn):
    conn, path, *_ = withdrawn
    before_evidence = list(map(tuple, conn.execute("SELECT * FROM kg_evidence")))
    before_lifecycle = list(map(tuple, conn.execute("SELECT * FROM kg_edge_lifecycle")))
    original, later = later_empty_overlap(withdrawn)
    current = conn.execute("SELECT * FROM kg_claim_extraction_outcomes WHERE chunk_id=?", (later.id,)).fetchone()
    assert current["phase1_generation_key"] == original["phase1_generation_key"]
    assert current["succeeded_at"] > original["succeeded_at"]
    assert list(map(tuple, conn.execute("SELECT * FROM kg_evidence"))) == before_evidence
    assert list(map(tuple, conn.execute("SELECT * FROM kg_edge_lifecycle"))) == before_lifecycle
    row = tier(scan(path), "edge_embeddings")
    assert row.source_withdrawn == 1 and row.unsafe_unknown == row.retired == 0


@pytest.mark.parametrize("sql", [
    "DELETE FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
    "UPDATE kg_claim_extraction_outcomes SET result_hash='sha256:forged' WHERE chunk_id=?",
    "UPDATE kg_claim_extraction_outcomes SET phase1_generation_key=NULL WHERE chunk_id=?",
    "UPDATE chunks SET text='broken original retirement input' WHERE id=?",
    "UPDATE chunks SET source_manifest_count=2 WHERE id=?",
    "UPDATE chunks SET created_at='9999-01-01T00:00:00Z' WHERE id=?",
    "DELETE FROM chunk_message_sources WHERE chunk_id=?",
])
def test_later_empty_cannot_substitute_for_missing_or_broken_original_receipt(withdrawn, sql):
    conn, path, original, *_ = withdrawn
    later_empty_overlap(withdrawn)
    corrupt(conn, sql, (original.id,))
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    row = tier(scan(path), "edge_embeddings")
    assert row.source_withdrawn == 0 and row.unsafe_unknown == 1


def test_different_prompt_overlap_does_not_renew_prior_retirement_proof(withdrawn):
    _, path, *_ = withdrawn
    later_empty_overlap(withdrawn, prompt_version="v16")
    row = tier(scan(path), "edge_embeddings")
    assert row.source_withdrawn == 0 and row.unsafe_unknown == 1


def test_same_prompt_changed_producer_does_not_replace_original_retirement_receipt(withdrawn):
    from hymem.dreaming import phase1
    from hymem.extraction.producer import phase1_generation_binding
    from tests.test_phase1_producer_identity import _DeclaredLLM
    conn, path, chunk, cfg, _ = withdrawn
    original = conn.execute("SELECT phase1_generation_key FROM kg_claim_extraction_outcomes").fetchone()[0]
    sources = phase1._claim_sources_for_chunk(conn, chunk)
    client = _DeclaredLLM("another-synthetic-model", "unused")
    extraction = phase1.ChunkExtraction(
        triples=[], markers=[], claim_sources={source.message_id: source for source in sources},
        source_validated=True, phase1_generation=phase1_generation_binding("v15", client),
    )
    time.sleep(1.05)
    with db.transaction(conn):
        phase1.persist_chunk_results(conn, chunk, extraction, prompt_version="v15", cfg=cfg)
    assert not client.calls
    assert conn.execute("SELECT phase1_generation_key FROM kg_claim_extraction_outcomes").fetchone()[0] != original
    row = tier(scan(path), "edge_embeddings")
    assert row.source_withdrawn == 0 and row.unsafe_unknown == 1


def test_original_retirement_witness_cannot_swallow_proof_budget_exhaustion(withdrawn, monkeypatch):
    from hymem import embedding_source_health as health
    _, path, original, *_ = withdrawn
    later_empty_overlap(withdrawn)
    source = reembed._source

    def exhaust_only_original(reader, index, row):
        if index == 0 and row["chunk_id"] == original.id:
            raise health._AuditLimit("source_proof_budget_exhausted")
        return source(reader, index, row)

    monkeypatch.setattr(reembed, "_source", exhaust_only_original)
    row = tier(scan(path), "edge_embeddings")
    assert row.source_withdrawn is None and row.unsafe_unknown is None
    assert row.error_code == "source_proof_budget_exhausted"
