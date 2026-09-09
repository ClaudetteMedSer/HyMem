from __future__ import annotations

import pytest

from benchmarks.multihop_miner import (
    _direct_edges,
    _dump_edges,
    _open_ro,
    _store_health,
)
from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.producer import phase1_generation_binding
from hymem.extraction.triples import Triple


def _persist_claim(hy: HyMem, producer: StubLLMClient, *, suffix: str) -> None:
    session_id = f"miner-authority-{suffix}"
    object_ = f"redis-{suffix}"
    hy.conn.execute("INSERT INTO sessions(id) VALUES (?)", (session_id,))
    message_id = int(hy.conn.execute(
        "INSERT INTO messages(session_id,role,content) VALUES (?,?,?)",
        (session_id, "user", f"The app uses {object_}."),
    ).lastrowid)
    chunk = Chunk(
        id=f"miner-authority-chunk-{suffix}",
        session_id=session_id,
        start_message_id=message_id,
        end_message_id=message_id,
        salience_reason="test",
        text=f"user: The app uses {object_}.",
        source_message_ids=(message_id,),
    )
    with core_db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, chunk.session_id)
        persist_chunks(hy.conn, [chunk])

    sources = phase1._claim_sources_for_chunk(hy.conn, chunk)
    extraction = phase1.ChunkExtraction(
        triples=[Triple(
            "app", "uses", object_, 1, source_message_id=message_id,
        )],
        markers=[],
        claim_sources={source.message_id: source for source in sources},
        source_validated=True,
        phase1_generation=phase1_generation_binding(
            hy.config.prompt_version, producer,
        ),
    )
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn,
            chunk,
            extraction,
            prompt_version=hy.config.prompt_version,
            cfg=hy.config,
        )


def test_raw_store_mode_and_graph_selectors_use_live_phase1_authority(cfg):
    producer_a = StubLLMClient(default="[]")
    hy = HyMem(cfg, llm=producer_a)
    try:
        _persist_claim(hy, producer_a, suffix="a")
        with core_db.transaction(hy.conn):
            # Active derived rows are not direct observation authority.
            hy.conn.execute(
                "INSERT INTO knowledge_graph("
                "subject_canonical,predicate,object_canonical,pos_evidence,"
                "neg_evidence,last_seen,last_reinforced,status,derived) "
                "VALUES ('app','uses','derived_shadow',1,0,"
                "datetime('now'),datetime('now'),'active',1)"
            )

        assert _direct_edges(hy.conn, ["app"]) == [("app", "uses", "redis_a")]
        assert _store_health(hy.conn) == (1, 1)
        dumped: dict[tuple[str, str, str], dict] = {}
        _dump_edges(hy.conn, dumped)
        assert set(dumped) == {("app", "uses", "redis_a")}

        # The same connection must stop exposing A's publication once B is the
        # selected producer, even though A remains durably preserved.
        hy.set_llm(StubLLMClient(default="{}"))
        assert _direct_edges(hy.conn, ["app"]) == []
        assert _store_health(hy.conn) == (0, 0)
        dumped = {}
        _dump_edges(hy.conn, dumped)
        assert dumped == {}
    finally:
        hy.close()

    # SQLite UDFs are connection-local. Raw benchmark read-only connections
    # install authority functions, infer the sole generation, and pin it.
    conn = _open_ro(cfg.db_path)
    try:
        assert _direct_edges(conn, ["app"]) == [("app", "uses", "redis_a")]
        assert _store_health(conn) == (1, 1)
        dumped = {}
        _dump_edges(conn, dumped)
        assert set(dumped) == {("app", "uses", "redis_a")}
    finally:
        conn.close()


def test_raw_store_mode_rejects_mixed_authorized_generations(cfg):
    producer_a = StubLLMClient(default="[]")
    producer_b = StubLLMClient(default="{}")
    hy = HyMem(cfg, llm=producer_a)
    try:
        _persist_claim(hy, producer_a, suffix="a")
        hy.set_llm(producer_b)
        _persist_claim(hy, producer_b, suffix="b")
    finally:
        hy.close()

    with pytest.raises(RuntimeError, match="multiple authorized Phase-1"):
        _open_ro(cfg.db_path)


def test_raw_store_mode_rejects_unbound_canonical_evidence(cfg):
    producer = StubLLMClient(default="[]")
    hy = HyMem(cfg, llm=producer)
    try:
        _persist_claim(hy, producer, suffix="legacy")
        with core_db.transaction(hy.conn):
            with core_db.evidence_history_mutation(hy.conn):
                hy.conn.execute(
                    "UPDATE kg_claim_extraction_outcomes "
                    "SET phase1_generation_key=NULL"
                )
    finally:
        hy.close()

    with pytest.raises(RuntimeError, match="without an authorized Phase-1"):
        _open_ro(cfg.db_path)
