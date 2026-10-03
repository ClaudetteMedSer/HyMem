"""Historical mismatches are not current operational embedding failures."""
from __future__ import annotations

from dataclasses import asdict
import json
import sqlite3

import pytest

from hymem import doctor, reembed
from hymem.core import db
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.extraction.embeddings import LocalHashEmbeddingClient
from tests.test_reembed import _seed_chunks, _seed_episode_fact, _corrupt_chunk_text

CURRENT = "hymem-embedding-producer-v1:" + "a" * 64
OLD = "hymem-embedding-producer-v1:" + "b" * 64


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "health.sqlite"
    conn = db.connect(path)
    db.initialize(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('source')")
    yield conn, path
    conn.close()


def scan(path, **kwargs):
    from hymem.embedding_source_health import scan_embedding_recovery_health
    return scan_embedding_recovery_health(path, live_model=CURRENT, live_dim=3, **kwargs)


def tier(report, table="chunk_embeddings"):
    return next(item for item in report.tables if item.table == table)


def legacy_chunk(conn, *, key="private-sentinel", reason="legacy", vector="[1,0,0]"):
    conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,salience_reason,text) "
        "VALUES (?,'source',9000,9000,?,'private source text')", (key, reason),
    )
    with db.embedding_mutation(conn):
        conn.execute("INSERT INTO chunk_embeddings(chunk_id,model,dim,vector_json) VALUES (?,?,3,?)",
                     (key, OLD, vector))


def test_retained_unmanifested_history_warns_without_rewriting_raw_inventory(store):
    conn, path = store
    legacy_chunk(conn)
    before = list(conn.iterdump())
    result = scan(path)
    assert result.status == "historical"
    assert tier(result).retained_unverified == 1
    assert tier(result).source_eligible == tier(result).unsafe_unknown == 0
    assert result.inventory.status == "incompatible"
    assert result.inventory.tables[0].incompatible == 1
    assert doctor._check_stored_embedding_health(conn, 3, CURRENT)[0].status == doctor.WARN
    assert list(conn.iterdump()) == before
    assert "private" not in json.dumps(asdict(result))


def test_known_short_session_builder_is_historical_not_proven_or_retired(store):
    conn, path = store
    legacy_chunk(conn, reason="short_session_fallback")
    result = scan(path)
    assert tier(result).retained_unverified == 1 and tier(result).retired == 0


def test_current_source_eligible_old_vectors_still_fail(store):
    conn, path = store
    _seed_chunks(conn, 1, messages=True)
    result = scan(path)
    assert result.status == "action_required"
    assert tier(result).source_eligible == 1
    assert tier(result, "message_embeddings").source_eligible == 1
    assert doctor._check_stored_embedding_health(conn, 3, CURRENT)[0].status == doctor.FAIL


def test_failed_current_proof_is_not_reclassified_as_history(store):
    conn, path = store
    _seed_chunks(conn, 1)
    _corrupt_chunk_text(conn, "different text")
    row = tier(scan(path))
    assert row.unsafe_unknown == 1 and row.retained_unverified == row.retired == 0
    assert doctor._check_stored_embedding_health(conn, 3, CURRENT)[0].status == doctor.FAIL


@pytest.mark.parametrize("vector", ["not-json-private", "[0,0,0]"])
def test_malformed_historical_vectors_always_fail(store, vector):
    conn, path = store
    legacy_chunk(conn, vector=vector)
    report = scan(path)
    assert report.status == "action_required"
    assert report.inventory.tables[0].malformed == 1
    assert tier(report).total == 0
    assert doctor._check_stored_embedding_health(conn, 3, CURRENT)[0].status == doctor.FAIL


def test_compatible_vectors_do_not_trigger_source_proof_or_mean_coverage(store, monkeypatch):
    conn, path = store
    legacy_chunk(conn)
    with db.embedding_mutation(conn):
        conn.execute("UPDATE chunk_embeddings SET model=?", (CURRENT,))
    monkeypatch.setattr(reembed, "_source", lambda *_: pytest.fail("unnecessary source proof"))
    result = scan(path)
    assert result.status == "compatible" and tier(result).total == 0
    detail = doctor._check_stored_embedding_health(conn, 3, CURRENT)[0].detail
    assert "not missing-row/source-proof coverage" in detail


def test_episode_fact_current_proofs_and_verified_fact_retirement(store, tmp_path):
    from hymem.config import HyMemConfig
    from hymem.dreaming import facts
    from hymem.extraction.llm import StubLLMClient
    conn, path = store
    _seed_chunks(conn, 1)
    _seed_episode_fact(conn, tmp_path)
    current = scan(path)
    assert tier(current, "episode_embeddings").source_eligible == 1
    assert tier(current, "narrative_fact_embeddings").source_eligible == 1
    key = conn.execute("SELECT slice_key FROM fact_extraction_outcomes").fetchone()[0]
    cfg = HyMemConfig(root=tmp_path)
    empty = facts.reextract_fact_outcome(conn, key, StubLLMClient(default="[]"), cfg)
    with db.transaction(conn):
        facts.persist_facts(conn, "source", empty)
    row = tier(scan(path), "narrative_fact_embeddings")
    assert row.retired == 1 and row.source_eligible == row.retained_unverified == row.unsafe_unknown == 0


def test_retraction_flag_without_lifecycle_proof_is_unknown(store, tmp_path):
    conn, path = store
    _seed_chunks(conn, 1)
    _seed_episode_fact(conn, tmp_path)
    with db.evidence_mutation(conn):
        conn.execute("UPDATE narrative_facts SET lifecycle_status='retracted'")
    row = tier(scan(path), "narrative_fact_embeddings")
    assert row.unsafe_unknown == 1 and row.retired == 0


@pytest.mark.parametrize("failure", [RuntimeError("private secret"), ValueError("private secret"),
                                     sqlite3.OperationalError("private secret")])
def test_proof_exception_is_unknown_not_healthy_or_historical(store, monkeypatch, failure):
    conn, path = store
    _seed_chunks(conn, 1)
    def fail(*_):
        raise failure
    monkeypatch.setattr(reembed, "_source", fail)
    report = scan(path)
    assert report.status in {"unavailable", "action_required"}
    assert tier(report).unsafe_unknown == 1 or tier(report).unsafe_unknown is None
    assert "private" not in json.dumps(asdict(report))


def test_bounded_candidate_scan_never_reports_partial_healthy_counts(store):
    conn, path = store
    for index in range(3):
        legacy_chunk(conn, key=str(index))
    report = scan(path, max_candidates=2)
    assert report.status == "unavailable"
    assert tier(report).total == 3 and tier(report).retained_unverified is None
    assert tier(report).error_code == "candidate_budget_exhausted"


def test_sql_work_budget_and_missing_file_fail_closed_without_creation(store, tmp_path):
    conn, path = store
    legacy_chunk(conn)
    report = scan(path, max_sql_steps=1)
    assert report.status == "unavailable"
    missing = tmp_path / "not-created.sqlite"
    assert scan(missing).status == "unavailable"
    assert not missing.exists()


def test_scanner_has_own_readonly_connection_without_touching_caller_transaction(store):
    conn, path = store
    legacy_chunk(conn)
    conn.execute("BEGIN")
    conn.execute("INSERT INTO schema_meta(key,value) VALUES ('private-test','uncommitted')")
    before = conn.total_changes
    assert scan(path).status == "historical"
    assert conn.in_transaction and conn.total_changes == before
    conn.rollback()


def test_h1_shaped_history_is_bounded_without_proving_healthy_7951_edges(store, monkeypatch):
    conn, path = store
    for index in range(1090):
        legacy_chunk(conn, key=str(index), reason="short_session_fallback" if index < 28 else "legacy")
    with db.embedding_mutation(conn):
        conn.executemany("INSERT INTO edge_embeddings(edge_text,model,dim,vector_json) VALUES (?,?,3,'[1,0,0]')",
                         [(f"private edge {index}", CURRENT) for index in range(7951)])
    monkeypatch.setattr(reembed, "_source", lambda *_: pytest.fail("unnecessary full-corpus proof"))
    report = scan(path)
    assert report.status == "historical"
    assert tier(report).retained_unverified == 1090
    assert report.inventory.tables[2].current_compatible == 7951


def test_raw_inventory_api_and_repair_schema_remain_unchanged(store):
    from hymem.embedding_health import scan_embedding_health
    conn, path = store
    legacy_chunk(conn)
    raw = scan_embedding_health(conn, live_model=CURRENT, live_dim=3)
    assert set(asdict(raw)) == {"live_identity_verified", "tables", "foreign_keys"}
    assert raw.status == "incompatible"
    report = reembed.repair(conn, LocalHashEmbeddingClient(dim_value=3))
    assert report.blocked == 1 and report.status == "blocked"


def test_inventory_and_classification_share_snapshot_despite_concurrent_commit(store, monkeypatch):
    from hymem import embedding_source_health as health
    conn, path = store
    legacy_chunk(conn)
    original = health.scan_embedding_health
    def concurrent_commit(reader, **kwargs):
        result = original(reader, **kwargs)
        conn.execute("DELETE FROM chunk_embeddings")
        return result
    monkeypatch.setattr(health, "scan_embedding_health", concurrent_commit)
    result = scan(path)
    assert result.inventory.tables[0].incompatible == tier(result).retained_unverified == 1
    monkeypatch.setattr(health, "scan_embedding_health", original)
    assert scan(path).inventory.tables[0].total == 0


def test_scanner_rejects_source_writes_and_never_initializes(store, monkeypatch):
    conn, path = store
    _seed_chunks(conn, 1)
    before = list(conn.iterdump())
    monkeypatch.setattr(db, "initialize", lambda *_: pytest.fail("scanner initialized database"))
    def write_attempt(reader, *_):
        reader.execute("UPDATE chunks SET salience_reason='not allowed'")
    monkeypatch.setattr(reembed, "_source", write_attempt)
    result = scan(path)
    assert result.status == "unavailable"
    assert tier(result).error_code == "source_schema_or_read_failure"
    assert list(conn.iterdump()) == before


def test_unbounded_proof_fetchall_is_stopped_and_counts_become_unknown(store, monkeypatch):
    conn, path = store
    _seed_chunks(conn, 1)
    def many_rows(reader, *_):
        reader.execute("WITH RECURSIVE t(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM t WHERE x<100000) SELECT x FROM t").fetchall()
        pytest.fail("proof fetchall escaped its bound")
    monkeypatch.setattr(reembed, "_source", many_rows)
    row = tier(scan(path))
    assert row.total == 1 and row.source_eligible is None
    assert row.error_code == "source_proof_budget_exhausted"


@pytest.mark.parametrize("bounds", [{"max_candidates": True}, {"max_candidates": 0},
                                  {"max_seconds": float("nan")}, {"max_seconds": 0},
                                  {"max_sql_steps": False}, {"max_sql_steps": 0}])
def test_invalid_bounds_are_unavailable_not_healthy(store, bounds):
    _, path = store
    report = scan(path, **bounds)
    assert report.status == "unavailable"
    assert tier(report).error_code == "invalid_bounds"


def test_elapsed_deadline_is_unavailable_even_on_empty_store(store):
    _, path = store
    report = scan(path, max_seconds=0.000000001)
    assert report.status == "unavailable"
    assert tier(report).error_code == "diagnostic_budget_exhausted"


@pytest.mark.parametrize("status,invalid_at,expected", [
    ("active", None, "unsafe_unknown"), ("stale", None, "unsafe_unknown"),
    ("retracted", None, "unsafe_unknown"),
    ("active", "2020-01-01T00:00:00.000Z", "unsafe_unknown"),
    ("active", "not-a-date", "unsafe_unknown"),
    ("retracted", "not-a-date", "unsafe_unknown"),
    ("retracted", "2999-01-01T00:00:00.000Z", "unsafe_unknown"),
])
def test_only_explicit_edge_retirement_is_nonoperational(store, status, invalid_at, expected):
    conn, path = store
    conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,status,invalid_at) "
                 "VALUES ('private','uses','test',?,?)", (status, invalid_at))
    with db.embedding_mutation(conn):
        conn.execute("INSERT INTO edge_embeddings(edge_text,model,dim,vector_json) "
                     "VALUES ('private uses test',?,3,'[1,0,0]')", (OLD,))
    row = tier(scan(path), "edge_embeddings")
    assert getattr(row, expected) == 1
    assert row.source_eligible == row.retained_unverified == 0


def test_retired_only_inventory_is_visible_without_operational_failure(store):
    from hymem.dreaming.bitemporal import record_lifecycle_event
    conn, path = store
    edge_id = conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,status,first_seen) "
                           "VALUES ('private','uses','test','retracted','2020-01-01T00:00:00.000Z')").lastrowid
    with db.evidence_mutation(conn):
        record_lifecycle_event(conn, edge_id=edge_id, event_key="legacy-state", event_kind="legacy_state",
                               direction=-1, event_at="2020-01-01T00:00:00.000Z")
    with db.embedding_mutation(conn):
        conn.execute("INSERT INTO edge_embeddings(edge_text,model,dim,vector_json) "
                     "VALUES ('private uses test',?,3,'[1,0,0]')", (OLD,))
    result = scan(path)
    assert result.status == "compatible" and result.inventory.status == "incompatible"
    detail = doctor._check_stored_embedding_health(conn, 3, CURRENT)[0]
    assert detail.status == doctor.OK and "explicitly-retired=1" in detail.detail
    assert "incompatible=1" in detail.detail


def test_absent_edge_owner_is_unknown_not_retired(store):
    conn, path = store
    with db.embedding_mutation(conn):
        conn.execute("INSERT INTO edge_embeddings(edge_text,model,dim,vector_json) "
                     "VALUES ('no owner',?,3,'[1,0,0]')", (OLD,))
    assert tier(scan(path), "edge_embeddings").unsafe_unknown == 1


def test_terminal_loss_is_visible_subset_not_invented_retirement(store):
    conn, path = store
    legacy_chunk(conn)
    conn.execute("INSERT INTO chunk_extraction_terminal_losses(chunk_id,reason) VALUES ('private-sentinel','source_manifest_unrecoverable')")
    report = scan(path)
    assert report.status == "historical" and tier(report).retained_unverified == 1
    assert tier(report).terminal_source_loss == 1 and tier(report).retired == 0
    detail = doctor._check_stored_embedding_health(conn, 3, CURRENT)[0].detail
    assert "terminal-source-loss=1" in detail


def test_unmanifested_chunk_in_session_with_integrity_failure_is_not_downgraded(store):
    conn, path = store
    legacy_chunk(conn)
    conn.execute("INSERT INTO coverage_integrity_failures(session_id,config_version,failure_reason) VALUES "
                 "('source','lossless-coverage-integrity-v1|coverage=dream-lossless-message-v1|hash=sha256-role-content-v1|record=hymem-message-jsonl-v1','source_stream_invalid')")
    row = tier(scan(path))
    assert row.unsafe_unknown == 1 and row.retained_unverified == 0


@pytest.mark.parametrize("family", ["episode", "fact", "aggregate", "aggregate-v1"])
def test_explicit_legacy_projection_shapes_remain_visible_history(store, family):
    conn, path = store
    if family == "episode":
        conn.execute("INSERT INTO episodes(id,session_id,title,summary) VALUES ('legacy','source','title','summary')")
        table, key, value = "episode_embeddings", "episode_id", "legacy"
    elif family == "fact":
        value = conn.execute("INSERT INTO narrative_facts(session_id,start_message_id,end_message_id,text,prompt_version) "
                             "VALUES ('source',1,1,'legacy fact','legacy')").lastrowid
        table, key = "narrative_fact_embeddings", "fact_id"
    else:
        conn.execute("INSERT INTO aggregation_nodes(id,title,summary,source_manifest_version) VALUES ('legacy','title','summary',?)",
                     ("aggregation-source-manifest-v1" if family.endswith("v1") else None,))
        table, key, value = "aggregation_node_embeddings", "node_id", "legacy"
    typed = family != "fact"
    with db.embedding_mutation(conn):
        conn.execute(f"INSERT INTO {table}({key},vector_json,model,dim,text_hash"
                     + (",embedding_producer_key) VALUES (?,'[1,0,0]',?,3,'old',?)" if typed
                        else ") VALUES (?,'[1,0,0]',?,3,'old')"),
                     (value, OLD, OLD) if typed else (value, OLD))
    result = scan(path)
    assert result.status == "historical"
    assert tier(result, table).retained_unverified == 1
