"""Independent network-free checks for a local-only exact replay proof."""

from dataclasses import replace
import hashlib
import json
import re
import sqlite3

import pytest

from hymem import HyMemConfig, portability
from hymem.core import db
from hymem.dreaming import canonicalize, phase1
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.producer import phase1_generation_binding
from hymem.extraction.triples import Triple
from tests.test_claim_semantic_dedup_guard import _chunk, _vectors
from tests.test_phase1_producer_identity import _DeclaredLLM


def _proof(conn, chunk_id):
    return conn.execute(
        "SELECT local_replay_proof FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()[0]


def _publish(conn, chunk, triples, cfg, generation, *, vectors=None):
    sources = phase1._claim_sources_for_chunk(conn, chunk)
    extraction = phase1.ChunkExtraction(
        triples=list(triples), markers=[], source_validated=True,
        claim_sources={source.message_id: source for source in sources},
        phase1_generation=generation,
    )
    with db.transaction(conn):
        phase1.persist_chunk_results(
            conn, chunk, extraction, prompt_version="v14", cfg=cfg,
            dedup_vectors=vectors,
        )
    return extraction


@pytest.fixture
def published(tmp_path):
    path = tmp_path / "local-proof.sqlite"
    conn = db.connect(path)
    db.initialize(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('s')")
    mid = int(conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at) "
        "VALUES ('s','user','Service uses Redis','2026-06-01T12:00:00.000Z')"
    ).lastrowid)
    with db.transaction(conn):
        materialize_message_coverage(conn, "s")
    chunk = _chunk(conn, mid, "root-proof")
    cfg = replace(HyMemConfig(root=tmp_path), triple_dedup_enabled=False)
    client = _DeclaredLLM("root-proof-model", "redis")
    generation = phase1_generation_binding("v14", client)
    triples = [Triple("service", "uses", "redis", 1, source_message_id=mid)]
    extraction = _publish(conn, chunk, triples, cfg, generation)
    state = dict(conn=conn, path=path, mid=mid, chunk=chunk, cfg=cfg,
                 generation=generation, triples=triples, extraction=extraction)
    try:
        yield state
        assert client.calls == []
    finally:
        state["conn"].close()


def test_live_proof_is_local_and_never_appears_on_wire(published, tmp_path):
    state = published
    proof = _proof(state["conn"], state["chunk"].id)
    assert isinstance(proof, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", proof)
    path = tmp_path / "wire.jsonl"
    portability.export_jsonl(state["conn"], path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    outcomes = [row["record"] for row in records if row["type"] == "claim_extraction_outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["phase1_generation_key"] == state["generation"]["generation_key"]
    assert "local_replay_proof" not in outcomes[0]
    assert proof not in path.read_text()


def test_reopen_preserves_local_proof_and_replay_avoids_claim_writes(published, monkeypatch):
    state = published
    proof = _proof(state["conn"], state["chunk"].id)
    assert proof is not None
    db.initialize(state["conn"])
    assert _proof(state["conn"], state["chunk"].id) == proof
    state["conn"].close()
    state["conn"] = db.connect(state["path"])
    db.initialize(state["conn"])
    assert _proof(state["conn"], state["chunk"].id) == proof
    before = list(state["conn"].iterdump())

    def must_not_write(*_args, **_kwargs):
        raise AssertionError("exact local proof unexpectedly rerouted the claim")

    monkeypatch.setattr(phase1, "_upsert_triple", must_not_write)
    _publish(state["conn"], state["chunk"], state["triples"],
             state["cfg"], state["generation"])
    assert list(state["conn"].iterdump()) == before


def test_stamped_v64_missing_proof_cannot_invent_authority(published):
    conn, chunk = published["conn"], published["chunk"]
    original = tuple(conn.execute(
        "SELECT chunk_id,prompt_version,prompt_generation,result_hash,succeeded_at,"
        "phase1_generation_key FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone())
    # Model a historical/manual table reconstruction without changing its
    # authoritative rows or safe FKs. No input proof can be inferred afterward.
    conn.execute("DROP TRIGGER kg_claim_extraction_outcomes_insert_guard")
    conn.execute("DROP TRIGGER kg_claim_extraction_outcomes_update_guard")
    conn.execute("ALTER TABLE kg_claim_extraction_outcomes DROP COLUMN local_replay_proof")
    assert db.schema_version(conn) == 64
    try:
        db.initialize(conn)
    except RuntimeError as exc:
        assert str(exc) == "schema v64 local claim replay proof shape is malformed"
        assert "local_replay_proof" not in {
            row["name"] for row in conn.execute("PRAGMA table_info(kg_claim_extraction_outcomes)")
        }
    else:
        # A nullable-only repair is safe too; neither strategy may derive a
        # proof from the pre-existing published observation set.
        assert _proof(conn, chunk.id) is None
    assert tuple(conn.execute(
        "SELECT chunk_id,prompt_version,prompt_generation,result_hash,succeeded_at,"
        "phase1_generation_key FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone()) == original
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("old_version", [46, 63])
def test_pre_v64_outcome_shape_and_absent_guards_upgrade_without_invented_proof(published, old_version):
    state, conn, chunk = published, published["conn"], published["chunk"]
    original = tuple(conn.execute(
        "SELECT chunk_id,prompt_version,prompt_generation,result_hash,succeeded_at,"
        "phase1_generation_key FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone())
    conn.execute("DROP TRIGGER kg_claim_extraction_outcomes_insert_guard")
    conn.execute("DROP TRIGGER kg_claim_extraction_outcomes_update_guard")
    conn.execute("ALTER TABLE kg_claim_extraction_outcomes DROP COLUMN local_replay_proof")
    conn.execute("UPDATE schema_meta SET value=? WHERE key='schema_version'", (str(old_version),))
    conn.close()
    state["conn"] = conn = db.connect(state["path"])
    db.initialize(conn)
    assert db.schema_version(conn) == 64
    assert _proof(conn, chunk.id) is None
    assert tuple(conn.execute(
        "SELECT chunk_id,prompt_version,prompt_generation,result_hash,succeeded_at,"
        "phase1_generation_key FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone()) == original
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE kg_claim_extraction_outcomes SET local_replay_proof=? WHERE chunk_id=?",
                     ("sha256:" + "0" * 64, chunk.id))


def test_cold_import_does_not_invent_local_proof(published, tmp_path):
    path = tmp_path / "wire.jsonl"
    portability.export_jsonl(published["conn"], path)
    conn = db.connect(tmp_path / "cold.sqlite")
    db.initialize(conn)
    try:
        portability.import_jsonl(conn, path)
        assert _proof(conn, published["chunk"].id) is None
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_import_rejects_injected_well_formed_origin_proof_even_with_valid_wire_checksum(published, tmp_path):
    origin_proof = _proof(published["conn"], published["chunk"].id)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", origin_proof)
    path = tmp_path / "forged-local-proof.jsonl"
    portability.export_jsonl(published["conn"], path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    outcomes = [row["record"] for row in records if row["type"] == "claim_extraction_outcome"]
    assert len(outcomes) == 1
    outcomes[0]["local_replay_proof"] = origin_proof
    assert records[-1]["type"] == "_end"
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records[:-1])
    records[-1]["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    path.write_text(body + json.dumps(records[-1], ensure_ascii=False) + "\n", encoding="utf-8")
    conn = db.connect(tmp_path / "forged-proof-target.sqlite")
    db.initialize(conn)
    try:
        before = list(conn.iterdump())
        with pytest.raises(ValueError, match="portable claim_extraction_outcome record does not match"):
            portability.import_jsonl(conn, path)
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes WHERE local_replay_proof IS NOT NULL"
        ).fetchone()[0] == 0
        assert list(conn.iterdump()) == before
        assert _proof(published["conn"], published["chunk"].id) == origin_proof
    finally:
        conn.close()


def test_same_identity_import_clears_touched_local_proof(published, tmp_path):
    conn, chunk = published["conn"], published["chunk"]
    assert _proof(conn, chunk.id) is not None
    original = conn.execute(
        "SELECT result_hash,phase1_generation_key FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone()
    path = tmp_path / "same.jsonl"
    portability.export_jsonl(conn, path)
    portability.import_jsonl(conn, path)
    assert _proof(conn, chunk.id) is None
    assert tuple(conn.execute(
        "SELECT result_hash,phase1_generation_key FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone()) == tuple(original)


def test_import_failure_rolls_back_proof_invalidation(published, tmp_path, monkeypatch):
    conn, chunk = published["conn"], published["chunk"]
    proof = _proof(conn, chunk.id)
    assert proof is not None
    path = tmp_path / "rollback.jsonl"
    portability.export_jsonl(conn, path)
    before = list(conn.iterdump())

    def fail_after_claim_import(*_args, **_kwargs):
        assert _proof(conn, chunk.id) is None
        raise ValueError("synthetic auxiliary import failure")

    monkeypatch.setattr(portability, "_import_v14_auxiliary_state", fail_after_claim_import)
    with pytest.raises(ValueError, match="synthetic auxiliary import failure"):
        portability.import_jsonl(conn, path)
    assert _proof(conn, chunk.id) == proof
    assert list(conn.iterdump()) == before


@pytest.mark.parametrize("coalescing", [False, True])
def test_authorized_merge_clears_only_affected_local_proof(published, coalescing):
    state, conn = published, published["conn"]
    unaffected = _chunk(conn, state["mid"], "root-unaffected")
    _publish(conn, unaffected,
             [Triple("unrelated", "uses", "postgres", 1, source_message_id=state["mid"])],
             state["cfg"], state["generation"])
    unaffected_proof = _proof(conn, unaffected.id)
    if coalescing:
        extra = _chunk(conn, state["mid"], "root-merge-target")
        _publish(conn, extra,
                 [Triple("service", "uses", "memcache", 1, source_message_id=state["mid"])],
                 state["cfg"], state["generation"])
    assert _proof(conn, state["chunk"].id) is not None
    with db.transaction(conn):
        canonicalize.merge(conn, "memcache", "redis")
    assert _proof(conn, state["chunk"].id) is None
    assert _proof(conn, unaffected.id) == unaffected_proof
    if coalescing:
        assert _proof(conn, extra.id) is None
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_alias_only_merge_without_graph_owner_clears_affected_proof(published):
    state, conn = published, published["conn"]
    chunk = _chunk(conn, state["mid"], "root-alias-only-merge")
    vectors = _vectors()
    with db.embedding_mutation(conn):
        conn.execute(
            "INSERT INTO edge_embeddings(edge_text,vector_json,model,dim) VALUES (?,?,?,?)",
            ("service uses redis", json.dumps([1.0, 0.0]), vectors.model, 2),
        )
    cfg = replace(state["cfg"], triple_dedup_enabled=True, triple_dedup_cosine_threshold=0.9)
    _publish(conn, chunk, [
        Triple("service", "uses", "redis", 1, source_message_id=state["mid"]),
        Triple("service", "uses", "redis_cache", 1, source_message_id=state["mid"]),
    ], cfg, state["generation"], vectors=vectors)
    assert _proof(conn, chunk.id) is not None
    assert conn.execute(
        "SELECT COUNT(*) FROM knowledge_graph WHERE object_canonical='redis_cache'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM entity_mention_observations "
        "WHERE chunk_id=? AND entity_canonical='redis_cache'", (chunk.id,),
    ).fetchone()[0] == 1
    with db.transaction(conn):
        canonicalize.merge(conn, "redis", "redis_cache")
    assert _proof(conn, chunk.id) is None
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_ordinary_sql_cannot_replace_local_proof(published):
    conn, chunk = published["conn"], published["chunk"]
    proof = _proof(conn, chunk.id)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE kg_claim_extraction_outcomes SET local_replay_proof=? WHERE chunk_id=?",
            ("sha256:" + "0" * 64, chunk.id),
        )
    assert _proof(conn, chunk.id) == proof


@pytest.mark.parametrize("bad", ["", "sha256:bad", "sha256:" + "G" * 64, 123])
def test_even_authorized_mutation_rejects_malformed_proof(published, bad):
    conn, chunk = published["conn"], published["chunk"]
    proof = _proof(conn, chunk.id)
    with pytest.raises(sqlite3.IntegrityError), db.evidence_mutation(conn):
        conn.execute(
            "UPDATE kg_claim_extraction_outcomes SET local_replay_proof=? WHERE chunk_id=?",
            (bad, chunk.id),
        )
    assert _proof(conn, chunk.id) == proof


_PROOF_CHECK_EXPRESSION = (
    "local_replay_proof IS NULL OR (length(local_replay_proof)=71 AND "
    "substr(local_replay_proof,1,7)='sha256:' AND "
    "substr(local_replay_proof,8) NOT GLOB '*[^0-9a-f]*')"
)
_PROOF_COLUMN = "local_replay_proof TEXT CHECK (" + _PROOF_CHECK_EXPRESSION + ")"


def test_local_proof_shape_validator_accepts_real_constraint():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("CREATE TABLE kg_claim_extraction_outcomes (" + _PROOF_COLUMN + ")")
        db._validate_v64_local_claim_replay_shape(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO kg_claim_extraction_outcomes VALUES ('not-a-proof')")
    finally:
        conn.close()


@pytest.mark.parametrize("spoof", ["or_one", "block_comment", "line_comment", "or_one_and_comment"])
def test_local_proof_shape_validator_rejects_weak_check_and_comment_spoofs(spoof):
    if spoof == "or_one":
        column = "local_replay_proof TEXT CHECK (" + _PROOF_CHECK_EXPRESSION + " OR 1)"
    elif spoof == "block_comment":
        column = "local_replay_proof TEXT CHECK (1) /* " + _PROOF_COLUMN + " */"
    elif spoof == "line_comment":
        column = "local_replay_proof TEXT CHECK (1) -- " + _PROOF_COLUMN + "\n"
    else:
        column = ("local_replay_proof TEXT CHECK (" + _PROOF_CHECK_EXPRESSION
                  + " OR 1) /* " + _PROOF_COLUMN + " */")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("CREATE TABLE kg_claim_extraction_outcomes (" + column + ")")
        # Establish that this is an actually weakened SQLite constraint, not
        # merely a differently spelled equivalent definition.
        conn.execute("INSERT INTO kg_claim_extraction_outcomes VALUES ('not-a-proof')")
        with pytest.raises(RuntimeError, match="schema v64 local claim replay proof shape is malformed"):
            db._validate_v64_local_claim_replay_shape(conn)
    finally:
        conn.close()
