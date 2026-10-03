"""Token hints retain live authority without a per-token full graph scan."""

from __future__ import annotations

import sqlite3

import pytest

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.core.graph import live_edge_predicate
from hymem.core.time import timestamp_at_or_before
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.triples import Triple
from hymem.query.augment import build_token_overlap_index


@pytest.fixture
def conn(tmp_path):
    connection = core_db.connect(tmp_path / "token-overlap.sqlite")
    core_db.initialize(connection)
    try:
        yield connection
    finally:
        connection.close()


def _native(conn, subject="project_alpha", obj="component_beta", **overrides):
    fields = {
        "subject_canonical": subject, "predicate": "uses",
        "object_canonical": obj, "pos_evidence": 2, "neg_evidence": 0,
        "status": "active", "derived": 0,
        "valid_at": "2024-01-01T00:00:00.000Z", "invalid_at": None,
    }
    fields.update(overrides)
    return int(conn.execute(
        "INSERT INTO knowledge_graph (" + ",".join(fields) + ") VALUES ("
        + ",".join("?" for _ in fields) + ")", tuple(fields.values()),
    ).lastrowid)


def _seed_index(conn):
    """Include historical/inactive canonicals, as a stale persistent cache can."""
    entries = []
    for row in conn.execute(
        "SELECT subject_canonical,object_canonical FROM knowledge_graph ORDER BY id"
    ):
        for canonical in row:
            entries.extend((token, canonical) for token in canonical.split("_"))
    conn.executemany(
        "INSERT OR IGNORE INTO token_overlap_index(token,canonical) VALUES (?,?)",
        entries,
    )


def _reference(conn):
    """The original predicate/query semantics, independently retained as oracle."""
    rows = conn.execute(
        "SELECT token_index.token,token_index.canonical "
        "FROM token_overlap_index token_index WHERE EXISTS ("
        "SELECT 1 FROM knowledge_graph kg WHERE " + live_edge_predicate("kg")
        + " AND (kg.subject_canonical=token_index.canonical "
        "OR kg.object_canonical=token_index.canonical))"
    ).fetchall()
    result = {}
    if rows:
        for row in rows:
            result.setdefault(row["token"], []).append(row["canonical"])
        return result
    current = live_edge_predicate()
    for row in conn.execute(
        "SELECT DISTINCT subject_canonical AS c FROM knowledge_graph WHERE "
        + current + " UNION SELECT DISTINCT object_canonical FROM knowledge_graph WHERE "
        + current
    ):
        for token in row["c"].split("_"):
            if token:
                result.setdefault(token, []).append(row["c"])
    return result


def _count_clock_calls(conn):
    calls = [0]

    def counted(value, cutoff):
        calls[0] += 1
        return timestamp_at_or_before(value, cutoff)

    conn.create_function("hymem_timestamp_at_or_before", 2, counted, deterministic=True)
    return calls


def test_persisted_index_validates_graph_once_not_once_per_token(conn):
    # Enough distinct compound names to expose the old correlated status scan.
    # Count actual authority-parser calls, not scheduler-sensitive elapsed time.
    edge_count = 512
    with core_db.transaction(conn):
        for index in range(edge_count):
            _native(conn, f"project_{index:04d}_service", f"component_{index:04d}_runtime")
        _seed_index(conn)
    calls = _count_clock_calls(conn)
    result = build_token_overlap_index(conn)
    observed = calls[0]
    assert observed <= edge_count, (
        f"live clocks evaluated {observed} times for {edge_count} edges"
    )
    assert len(result["project"]) == edge_count
    assert len(result["runtime"]) == edge_count
    assert result == _reference(conn)


@pytest.mark.parametrize("cache", ["cold", "warm", "obsolete", "partial"])
def test_cached_and_cold_outputs_match_reference_without_trusting_dead_names(conn, cache):
    _native(conn)
    _native(conn, "project_gamma", "component_beta")
    _native(conn, "dead_project", "dead_component", status="retracted")
    _native(conn, "future_project", "future_component", valid_at="2999-01-01")
    if cache == "warm":
        _seed_index(conn)
    elif cache == "partial":
        conn.execute(
            "INSERT INTO token_overlap_index VALUES ('alpha','project_alpha')"
        )
    if cache != "cold":
        conn.executemany("INSERT OR IGNORE INTO token_overlap_index VALUES (?,?)", [
            ("dead", "dead_project"), ("future", "future_project"),
            ("untrusted", "not_a_graph_canonical"),
        ])
    before = conn.total_changes
    expected = _reference(conn)
    actual = build_token_overlap_index(conn)
    assert actual == expected
    assert conn.total_changes == before
    assert not conn.in_transaction
    assert "untrusted" not in actual
    assert not any(
        canonical.startswith(("dead_", "future_"))
        for canonicals in actual.values() for canonical in canonicals
    )


def test_cold_writer_persists_then_new_read_connection_rechecks_authority(conn, tmp_path):
    _native(conn)
    expected = _reference(conn)
    assert build_token_overlap_index(conn, write_conn=conn) == expected
    assert conn.execute("SELECT COUNT(*) FROM token_overlap_index").fetchone()[0] > 0
    reader = core_db.connect(tmp_path / "token-overlap.sqlite")
    try:
        reader.execute("PRAGMA query_only=ON")
        assert build_token_overlap_index(reader) == expected
        conn.execute("UPDATE knowledge_graph SET status='retracted'")
        assert build_token_overlap_index(reader) == _reference(reader) == {}
    finally:
        reader.close()


@pytest.mark.parametrize("change", [
    {"status": "retracted"}, {"invalid_at": "2024-02-01"},
    {"neg_evidence": 2}, {"derived": 1}, {"valid_at": "2999-01-01"},
    {"valid_at": "not-a-timestamp"}, {"valid_at": "2024-02-30"},
])
def test_live_predicate_rejections_are_identical_with_stale_index(conn, change):
    _native(conn, **change)
    _seed_index(conn)
    assert build_token_overlap_index(conn) == _reference(conn) == {}


def test_null_clock_and_mixed_live_dead_endpoint_compatibility_is_retained(conn):
    _native(conn, "native_project", "shared_component", valid_at=None)
    _native(conn, "dead_project", "shared_component", status="retracted")
    _seed_index(conn)
    result = build_token_overlap_index(conn)
    assert result == _reference(conn)
    assert result["shared"] == ["shared_component"]
    assert "dead" not in result


def _canonical(conn, tmp_path):
    conn.execute("INSERT INTO sessions(id) VALUES ('canonical-source')")
    message_id = int(conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at) VALUES "
        "('canonical-source','user','Project Alpha uses Component Beta.',"
        "'2024-01-01T00:00:00.000Z')"
    ).lastrowid)
    chunk = Chunk(
        id="canonical-chunk", session_id="canonical-source",
        start_message_id=message_id, end_message_id=message_id,
        source_message_ids=(message_id,), salience_reason="test",
        text="user: Project Alpha uses Component Beta.",
    )
    with core_db.transaction(conn):
        materialize_message_coverage(conn, "canonical-source")
        persist_chunks(conn, [chunk])
    sources = phase1._claim_sources_for_chunk(conn, chunk)
    with core_db.transaction(conn):
        phase1.persist_chunk_results(
            conn, chunk, phase1.ChunkExtraction(
                triples=[Triple(
                    "project_alpha", "uses", "component_beta", 1,
                    source_message_id=message_id,
                )], markers=[], source_validated=True,
                claim_sources={source.message_id: source for source in sources},
            ), prompt_version="v13", cfg=HyMemConfig(root=tmp_path),
        )
    generation = conn.execute(
        "SELECT phase1_generation_key FROM kg_claim_extraction_outcomes"
    ).fetchone()[0]
    _seed_index(conn)
    assert build_token_overlap_index(conn) == _reference(conn)
    assert build_token_overlap_index(conn)["project"] == ["project_alpha"]
    return generation


@pytest.mark.parametrize("table,mutation", [
    ("kg_evidence", "UPDATE kg_evidence SET published_at=NULL"),
    ("kg_evidence", "UPDATE kg_evidence SET source_event_at='not-a-clock'"),
    ("kg_evidence", "UPDATE kg_evidence SET extracted_at='2999-01-01'"),
    ("kg_claim_observations", "DELETE FROM kg_claim_observations"),
    ("kg_claim_observations", "UPDATE kg_claim_observations SET prompt_generation=prompt_generation+1"),
    ("kg_claim_extraction_outcomes", "DELETE FROM kg_claim_extraction_outcomes"),
    ("kg_edge_lifecycle", "DELETE FROM kg_edge_lifecycle"),
])
def test_corrupt_canonical_publication_never_authorizes_cached_name(conn, tmp_path, table, mutation):
    _canonical(conn, tmp_path)
    # Simulate imported/direct-SQL corruption in this disposable test database.
    for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (table,)
    ).fetchall():
        conn.execute('DROP TRIGGER "' + row[0].replace('"', '""') + '"')
    conn.execute(mutation)
    assert build_token_overlap_index(conn) == _reference(conn) == {}


def test_current_producer_scope_revalidated_across_calls(conn, tmp_path):
    generation = _canonical(conn, tmp_path)
    core_db.register_current_phase1_generation(conn, generation)
    first = build_token_overlap_index(conn)
    assert first
    core_db.register_current_phase1_generation(
        conn, "hymem-phase1-generation-v1:" + "f" * 64,
    )
    assert build_token_overlap_index(conn) == _reference(conn) == {}
    core_db.register_current_phase1_generation(conn, generation)
    assert build_token_overlap_index(conn) == _reference(conn) == first


@pytest.mark.parametrize("caller_snapshot", [False, True])
def test_independent_commit_during_validation_does_not_mix_snapshots(conn, tmp_path, caller_snapshot):
    _native(conn)
    _native(conn, "project_gamma", "component_delta")
    _seed_index(conn)
    expected = _reference(conn)
    writer = core_db.connect(tmp_path / "token-overlap.sqlite")
    writer.execute("PRAGMA busy_timeout=0")
    committed = []

    def commit_during_clock_check(value, cutoff):
        if not committed:
            writer.execute("UPDATE knowledge_graph SET status='retracted'")
            committed.append(True)
        return timestamp_at_or_before(value, cutoff)

    conn.create_function(
        "hymem_timestamp_at_or_before", 2, commit_during_clock_check, deterministic=True,
    )
    try:
        if caller_snapshot:
            conn.execute("BEGIN")
        assert build_token_overlap_index(conn) == expected
        assert committed == [True]
        assert conn.in_transaction is caller_snapshot
        if caller_snapshot:
            assert build_token_overlap_index(conn) == expected
            conn.execute("ROLLBACK")
        assert build_token_overlap_index(conn) == _reference(conn) == {}
        assert not conn.in_transaction
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        writer.close()


def test_sql_validation_failure_does_not_publish_or_mutate_cache(conn):
    _native(conn)
    _seed_index(conn)
    before = list(conn.execute("SELECT * FROM token_overlap_index"))

    def failed_clock(_value, _cutoff):
        raise RuntimeError("test authority failure")

    conn.create_function("hymem_timestamp_at_or_before", 2, failed_clock)
    with pytest.raises(sqlite3.OperationalError, match="user-defined function"):
        build_token_overlap_index(conn, write_conn=conn)
    assert list(conn.execute("SELECT * FROM token_overlap_index")) == before
    assert not conn.in_transaction
