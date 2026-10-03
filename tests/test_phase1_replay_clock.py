"""Exact publication acknowledgements are stable across SQL clock ticks."""
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from hymem import HyMem, HyMemConfig, portability
from hymem.core import db
from hymem.dreaming import phase1
from hymem.extraction.markers import Marker
from hymem.extraction.triples import Triple
from tests.test_claim_provenance import _open, _messages, _chunk, _persist
from tests.test_phase1_producer_identity import _DeclaredLLM, _seed_chunk, _extract_and_persist


def _clock(conn):
    value = [conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]]
    conn.create_function("current_timestamp", 0, lambda: value[0])
    return value


def _tick(value):
    value[0] = (datetime.fromisoformat(value[0]) + timedelta(seconds=1)).isoformat(" ")
    return value[0]


def _seed(tmp_path):
    conn = _open(tmp_path)
    clock = _clock(conn)
    mid, = _messages(conn, [("user", "The app uses Redis", "2024-01-01")])
    chunk = _chunk(conn, "clock-replay", [mid])
    claim = Triple("app", "uses", "redis", 1, source_message_id=mid)
    cfg = HyMemConfig(root=tmp_path)
    _persist(conn, chunk, [claim], prompt_version="v13", cfg=cfg)
    return conn, clock, chunk, claim, cfg


def _ack(conn, chunk):
    return dict(conn.execute("SELECT * FROM processed_chunks WHERE chunk_id=?", (chunk.id,)).fetchone())


def _claim_rows(conn):
    return {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            for table in ("kg_evidence", "kg_claim_observations", "kg_claim_extraction_outcomes", "kg_edge_lifecycle")}


def test_current_exact_replay_preserves_whole_dump_and_wire_after_clock_tick(tmp_path):
    conn, clock, chunk, claim, cfg = _seed(tmp_path)
    try:
        before = list(conn.iterdump())
        portability.export_jsonl(conn, tmp_path / "before.jsonl")
        _tick(clock)
        _persist(conn, chunk, [claim], prompt_version="v13", cfg=cfg)
        portability.export_jsonl(conn, tmp_path / "after.jsonl")
        assert list(conn.iterdump()) == before
        assert (tmp_path / "before.jsonl").read_bytes() == (tmp_path / "after.jsonl").read_bytes()
    finally:
        conn.close()


@pytest.mark.parametrize("damage", ["missing", "null_generation", "null_timestamp"])
def test_exact_replay_repairs_acknowledgement_and_clears_only_its_attempts(tmp_path, damage):
    conn, clock, chunk, claim, cfg = _seed(tmp_path)
    try:
        original = _ack(conn, chunk)
        claim_rows = _claim_rows(conn)
        if damage == "missing":
            conn.execute("DELETE FROM processed_chunks WHERE chunk_id=?", (chunk.id,))
        elif damage == "null_generation":
            conn.execute("UPDATE processed_chunks SET phase1_generation_key=NULL WHERE chunk_id=?", (chunk.id,))
        else:
            # The deployed schema explicitly permits a NULL processed_at.
            assert next(row for row in conn.execute("PRAGMA table_info(processed_chunks)")
                        if row["name"] == "processed_at")["notnull"] == 0
            conn.execute("UPDATE processed_chunks SET processed_at=NULL WHERE chunk_id=?", (chunk.id,))
        conn.execute("INSERT INTO chunk_extraction_attempts(chunk_id,prompt_version,attempts,phase1_generation_key) "
                     "VALUES (?,?,2,?)", (chunk.id, original["prompt_version"], original["phase1_generation_key"]))
        conn.execute("INSERT INTO chunk_extraction_attempts(chunk_id,prompt_version,attempts) "
                     "VALUES (?,'historical-other-cache',3)", (chunk.id,))
        _tick(clock)
        _persist(conn, chunk, [claim], prompt_version="v13", cfg=cfg)
        assert _ack(conn, chunk) == {**original, "processed_at": clock[0]}
        assert _claim_rows(conn) == claim_rows
        assert [tuple(row) for row in conn.execute("SELECT prompt_version,attempts FROM chunk_extraction_attempts")] == [
            ("historical-other-cache", 3),
        ]
        before = list(conn.iterdump())
        _tick(clock)
        _persist(conn, chunk, [claim], prompt_version="v13", cfg=cfg)
        assert list(conn.iterdump()) == before
    finally:
        conn.close()


def test_portable_restore_gets_one_fresh_ack_then_exact_replay_is_stable(tmp_path):
    conn, clock, chunk, claim, cfg = _seed(tmp_path / "source")
    restored = None
    try:
        path = tmp_path / "source.jsonl"
        portability.export_jsonl(conn, path)
        restored = db.connect(tmp_path / "restored.sqlite")
        db.initialize(restored)
        portability.import_jsonl(restored, path)
        assert restored.execute("SELECT COUNT(*) FROM processed_chunks").fetchone()[0] == 0
        restored_clock = _clock(restored)
        _tick(restored_clock)
        _persist(restored, chunk, [claim], prompt_version="v13", cfg=cfg)
        assert _ack(restored, chunk)["processed_at"] == restored_clock[0]
        before = list(restored.iterdump())
        portability.export_jsonl(restored, tmp_path / "restored-before.jsonl")
        _tick(restored_clock)
        _persist(restored, chunk, [claim], prompt_version="v13", cfg=cfg)
        portability.export_jsonl(restored, tmp_path / "restored-after.jsonl")
        assert list(restored.iterdump()) == before
        assert (tmp_path / "restored-before.jsonl").read_bytes() == (tmp_path / "restored-after.jsonl").read_bytes()
    finally:
        conn.close()
        if restored is not None:
            restored.close()


def test_changed_producer_refreshes_ack_but_does_not_reuse_old_claim_authority(cfg):
    first = _DeclaredLLM("clock-a", "redis")
    hy = HyMem(replace(cfg, aggregation_nodes_enabled=False), llm=first)
    try:
        clock = _clock(hy.conn)
        chunk = _seed_chunk(hy)
        assert _extract_and_persist(hy, chunk, first) is not None
        old = _ack(hy.conn, chunk)
        second = _DeclaredLLM("clock-b", "postgres")
        hy.set_llm(second)
        _tick(clock)
        assert _extract_and_persist(hy, chunk, second) is not None
        new = _ack(hy.conn, chunk)
        assert new["prompt_version"] == old["prompt_version"]
        assert new["phase1_generation_key"] != old["phase1_generation_key"]
        assert new["processed_at"] == clock[0] != old["processed_at"]
        assert hy.conn.execute("SELECT object_canonical FROM knowledge_graph WHERE pos_evidence>neg_evidence").fetchone()[0] == "postgres"
    finally:
        hy.close()


def test_auxiliary_change_still_publishes_with_its_own_clock_on_exact_claim_replay(tmp_path):
    conn, clock, chunk, claim, cfg = _seed(tmp_path)
    try:
        original = _ack(conn, chunk)
        claim_rows = _claim_rows(conn)
        _tick(clock)
        sources = phase1._claim_sources_for_chunk(conn, chunk)
        extraction = phase1.ChunkExtraction(
            triples=[claim], markers=[Marker("preference", "Prefer Redis")],
            claim_sources={source.message_id: source for source in sources}, source_validated=True,
        )
        with db.transaction(conn):
            phase1.persist_chunk_results(conn, chunk, extraction, prompt_version="v13", cfg=cfg)
        assert _ack(conn, chunk) == original
        assert _claim_rows(conn) == claim_rows
        assert conn.execute("SELECT published_at FROM phase1_auxiliary_outcomes WHERE chunk_id=?", (chunk.id,)).fetchone()[0] == clock[0]
        assert conn.execute("SELECT statement FROM behavioral_markers WHERE chunk_id=?", (chunk.id,)).fetchone()[0] == "Prefer Redis"
    finally:
        conn.close()
