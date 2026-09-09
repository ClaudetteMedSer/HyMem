"""Exact edge mirror candidates are narrowed before full live authority work."""
from __future__ import annotations

import sys

import pytest

from hymem import reembed
from hymem.config import HyMemConfig
from hymem.core import db
from hymem.core.graph import live_edge_predicate
from hymem.core.time import timestamp_at_or_before
from hymem.dreaming import phase1
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.triples import Triple
from hymem.extraction.embeddings import LocalHashEmbeddingClient
from tests.test_reembed import ExactEmbedder, observe_provider_calls


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "edges.sqlite")
    db.initialize(connection)
    try:
        yield connection
    finally:
        connection.close()


def _edge(conn, subject="target", obj="component", **changes):
    fields = dict(subject_canonical=subject, predicate="uses", object_canonical=obj,
                  pos_evidence=2, neg_evidence=0, status="active", derived=0,
                  valid_at="2024-01-01T00:00:00.000Z", invalid_at=None)
    fields.update(changes)
    return conn.execute(
        "INSERT INTO knowledge_graph (" + ",".join(fields) + ") VALUES ("
        + ",".join("?" for _ in fields) + ")", tuple(fields.values()),
    ).lastrowid


def _reference(conn, text):
    rows = conn.execute(
        "SELECT * FROM knowledge_graph WHERE " + live_edge_predicate()
        + " AND subject_canonical || ' ' || predicate || ' ' || object_canonical = ?"
        + " ORDER BY id LIMIT 65", (text,),
    ).fetchall()
    if not rows or len(rows) > 64:
        return None
    return text, tuple(tuple(row) for row in rows), tuple(row["id"] for row in rows)


def test_exact_text_filters_unrelated_edges_before_any_authority_clock(conn):
    with db.transaction(conn):
        for index in range(512):
            _edge(conn, f"unrelated_{index}")
        target = _edge(conn)
    calls = []
    def counted(value, cutoff):
        calls.append(value)
        return timestamp_at_or_before(value, cutoff)
    conn.create_function("hymem_timestamp_at_or_before", 2, counted, deterministic=True)
    actual = reembed._source(conn, 2, {"edge_text": "target uses component"})
    assert actual is not None and actual[2] == (target,)
    assert len(calls) == 1, f"authority clocks checked {len(calls)} rows for one exact candidate"
    assert actual == _reference(conn, "target uses component")
    calls.clear()
    assert reembed._source(conn, 2, {"edge_text": "absent uses component"}) is None
    assert calls == []


def _same_text(conn, count, *, dead=0):
    # Distinct historical coordinates can concatenate to the same mirror key.
    # Split one exact string at each ' uses ' occurrence without weakening the
    # real UNIQUE(subject,predicate,object) constraint.
    words = [f"part_{index}" for index in range(count + 1)]
    text = " uses ".join(words)
    ids = []
    for index in range(1, len(words)):
        changes = ({"status": "retracted"} if index <= dead and index % 2
                   else {"valid_at": "not-a-clock"} if index <= dead else {})
        ids.append(_edge(conn, " uses ".join(words[:index]), " uses ".join(words[index:]), **changes))
    return text, ids


def test_same_text_dead_prefix_cannot_hide_later_live_owner(conn):
    text, ids = _same_text(conn, 81, dead=80)
    actual = reembed._source(conn, 2, {"edge_text": text})
    assert actual == _reference(conn, text)
    assert actual is not None and actual[2] == (ids[-1],)


@pytest.mark.parametrize("live_count", [63, 64, 65])
def test_live_owner_limit_is_applied_after_authority_and_native_order_is_unchanged(conn, live_count):
    text, ids = _same_text(conn, live_count + 17, dead=17)
    actual = reembed._source(conn, 2, {"edge_text": text})
    assert actual == _reference(conn, text)
    if live_count > 64:
        assert actual is None
    else:
        assert actual[2] == tuple(ids[17:])
        assert actual[1] == tuple(tuple(row) for row in conn.execute(
            "SELECT * FROM knowledge_graph WHERE id>=? ORDER BY id", (ids[17],),
        ))


@pytest.mark.parametrize("changes,accepted", [
    ({"valid_at": None}, True), ({"valid_at": "2024-01-01"}, True),
    ({"valid_at": "not-a-clock"}, False), ({"valid_at": "2024-02-30"}, False),
    ({"valid_at": "2999-01-01"}, False), ({"invalid_at": "2024-01-01"}, False),
    ({"neg_evidence": 2}, False), ({"derived": 1}, False),
    ({"status": "stale"}, False),
])
def test_exact_candidate_keeps_all_live_clock_and_status_semantics(conn, changes, accepted):
    _edge(conn, **changes)
    before = conn.total_changes
    actual = reembed._source(conn, 2, {"edge_text": "target uses component"})
    assert actual == _reference(conn, "target uses component")
    assert (actual is not None) is accepted
    assert conn.total_changes == before and not conn.in_transaction


def _canonical(conn, tmp_path):
    conn.execute("INSERT INTO sessions(id) VALUES ('canonical-source')")
    message = int(conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at) VALUES "
        "('canonical-source','user','Target uses component.','2024-01-01T00:00:00.000Z')"
    ).lastrowid)
    chunk = Chunk("canonical-chunk", "canonical-source", message, message,
                  "test", "user: Target uses component.", (message,))
    with db.transaction(conn):
        materialize_message_coverage(conn, "canonical-source")
        persist_chunks(conn, [chunk])
    proofs = phase1._claim_sources_for_chunk(conn, chunk)
    with db.transaction(conn):
        phase1.persist_chunk_results(
            conn, chunk, phase1.ChunkExtraction(
                triples=[Triple("target", "uses", "component", 1, source_message_id=message)],
                markers=[], source_validated=True,
                claim_sources={proof.message_id: proof for proof in proofs},
            ), prompt_version="v13", cfg=HyMemConfig(root=tmp_path),
        )
    return conn.execute("SELECT phase1_generation_key FROM kg_claim_extraction_outcomes").fetchone()[0]


def test_canonical_source_revalidates_current_producer_and_publication_proof(conn, tmp_path):
    generation = _canonical(conn, tmp_path)
    db.register_current_phase1_generation(conn, generation)
    row = {"edge_text": "target uses component"}
    first = reembed._source(conn, 2, row)
    assert first is not None and first == _reference(conn, row["edge_text"])
    db.register_current_phase1_generation(conn, "hymem-phase1-generation-v1:" + "f" * 64)
    assert reembed._source(conn, 2, row) == _reference(conn, row["edge_text"]) is None
    db.register_current_phase1_generation(conn, generation)
    assert reembed._source(conn, 2, row) == first
    # Simulate damaged historical storage on this disposable database only.
    for trigger in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='kg_evidence'").fetchall():
        conn.execute('DROP TRIGGER "' + trigger[0].replace('"', '""') + '"')
    conn.execute("UPDATE kg_evidence SET published_at=NULL")
    assert reembed._source(conn, 2, row) == _reference(conn, row["edge_text"]) is None


@pytest.mark.parametrize("caller_snapshot", [False, True])
def test_candidate_and_authority_share_one_snapshot_without_consuming_callers_transaction(conn, tmp_path, caller_snapshot):
    _edge(conn)
    row = {"edge_text": "target uses component"}
    expected = _reference(conn, row["edge_text"])
    writer = db.connect(tmp_path / "edges.sqlite")
    committed = []
    def commit_during_clock(value, cutoff):
        if not committed:
            writer.execute("UPDATE knowledge_graph SET status='retracted'")
            committed.append(True)
        return timestamp_at_or_before(value, cutoff)
    conn.create_function("hymem_timestamp_at_or_before", 2, commit_during_clock, deterministic=True)
    try:
        if caller_snapshot:
            conn.execute("BEGIN")
        assert reembed._source(conn, 2, row) == expected
        assert committed == [True] and conn.in_transaction is caller_snapshot
        if caller_snapshot:
            assert reembed._source(conn, 2, row) == expected
            conn.rollback()
        assert reembed._source(conn, 2, row) is None
    finally:
        if conn.in_transaction:
            conn.rollback()
        writer.close()


@pytest.mark.parametrize("when", ["before_provider", "after_provider"])
def test_edge_source_revalidated_before_provider_and_under_publication_lock(conn, when):
    _edge(conn)
    old_model = embedding_storage_identity(LocalHashEmbeddingClient(dim_value=3, model_name="old"))[0]
    with db.embedding_mutation(conn):
        conn.execute("INSERT INTO edge_embeddings(edge_text,model,dim,vector_json) VALUES ('target uses component',?,3,'[1,0,0]')", (old_model,))
    before = tuple(conn.execute("SELECT * FROM edge_embeddings").fetchone())
    client = ExactEmbedder()
    previous = sys.getprofile()
    changed = []
    def observe(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        function = reembed._source if when == "before_provider" else ExactEmbedder.embed
        if frame.f_code is function.__code__ and event == "return" and not changed:
            conn.execute("UPDATE knowledge_graph SET status='retracted'")
            changed.append(True)
    sys.setprofile(observe)
    try:
        report = reembed.repair(conn, client, apply=True)
    finally:
        sys.setprofile(previous)
    assert changed == [True] and report.repaired == 0 and report.exit_code == 1
    assert len(client.calls) == (0 if when == "before_provider" else 1)
    assert tuple(conn.execute("SELECT * FROM edge_embeddings").fetchone()) == before
    assert not conn.execute("SELECT 1 FROM schema_meta WHERE key LIKE 'embedding_repair_scan_v1:%'").fetchall()
    assert not conn.in_transaction
