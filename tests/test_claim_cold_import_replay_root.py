"""Independent historical-publication replay regressions (no providers)."""

from dataclasses import replace
import sqlite3

import pytest

from hymem import portability
from hymem.core import db
from hymem.dreaming import phase1
from hymem.extraction.contract import extraction_cache_key
from hymem.extraction.producer import phase1_generation_binding
from tests.test_claim_cold_import_replay import cold_alias, _claim_snapshot, _CLAIM_TABLES
from tests.test_claim_semantic_dedup_guard import _chunk, _vectors
from tests.test_phase1_producer_identity import _DeclaredLLM


@pytest.mark.parametrize("dedup_enabled", [True, False])
def test_cold_replay_can_cite_unchanged_evidence_from_an_older_prompt(
    cold_alias, tmp_path, dedup_enabled,
):
    state = cold_alias
    origin = state["origin"]
    client = _DeclaredLLM("cold-replay-model", "redis")
    vectors = _vectors()
    vectors["app uses redis"] = [1.0, 0.0]
    vectors["app uses redis_cache"] = [1.0, 0.0]
    # An overlapping chunk legitimately keeps the same immutable evidence
    # cited while the original chunk moves to a newer prompt generation.
    client.object_name = "redis_cache"
    keeper = _chunk(
        origin, state["alias_chunk"].source_message_ids[0], "overlapping-citation",
    )
    kept = phase1.extract_chunk_results(
        origin, keeper, client, prompt_version="v14",
        phase1_generation=state["generation"],
    )
    assert kept is not None and not kept.failed
    with db.transaction(origin):
        phase1.persist_chunk_results(
            origin, keeper, kept, prompt_version="v14",
            cfg=state["cfg"], dedup_vectors=vectors,
        )
    generation = phase1_generation_binding("v15", client)
    chunk = state["alias_chunk"]
    extraction = phase1.extract_chunk_results(
        origin, chunk, client, prompt_version="v15",
        phase1_generation=generation,
    )
    assert extraction is not None and not extraction.failed
    with db.transaction(origin):
        phase1.persist_chunk_results(
            origin, chunk, extraction, prompt_version="v15",
            cfg=state["cfg"], dedup_vectors=vectors,
        )
    row = origin.execute(
        "SELECT observation.prompt_version,ev.extraction_prompt_version "
        "FROM kg_claim_observations observation "
        "JOIN kg_evidence ev ON ev.id=observation.evidence_id "
        "WHERE observation.chunk_id=?", (chunk.id,),
    ).fetchone()
    assert tuple(row) == (extraction_cache_key("v15"), extraction_cache_key("v14"))
    wire = tmp_path / "new-prompt-old-evidence.jsonl"
    portability.export_jsonl(origin, wire)
    target = db.connect(tmp_path / "new-prompt-import.sqlite")
    db.initialize(target)
    try:
        portability.import_jsonl(target, wire)
        assert target.execute("SELECT COUNT(*) FROM edge_embeddings").fetchone()[0] == 0
        replay = phase1.extract_chunk_results(
            target, chunk, client, prompt_version="v15",
            phase1_generation=generation,
        )
        assert replay is not None and not replay.failed
        cfg = replace(state["cfg"], triple_dedup_enabled=dedup_enabled)
        assert not phase1._is_exact_published_replay(
            target, chunk, replay, prompt_version=extraction_cache_key("v14"),
            phase1_generation_key=state["generation"]["generation_key"],
            cfg=cfg, dedup_vectors=None, in_cycle_edges=None,
        )
        before = _claim_snapshot(target)
        writes = []

        def audit_write(action, table, *_args):
            if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE,
                          sqlite3.SQLITE_DELETE) and table in _CLAIM_TABLES:
                writes.append((action, table))
            return sqlite3.SQLITE_OK

        target.set_authorizer(audit_write)
        try:
            with db.transaction(target):
                phase1.persist_chunk_results(
                    target, chunk, replay, prompt_version="v15",
                    cfg=cfg, dedup_vectors=None,
                    in_cycle_edges=phase1.new_in_cycle_pool(),
                )
        finally:
            target.set_authorizer(None)
        assert writes == []
        assert _claim_snapshot(target) == before
        assert target.execute(
            "SELECT COUNT(*) FROM current_phase1_publications WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] == 1
        assert target.execute(
            "SELECT local_replay_proof FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] is None
        assert target.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        target.close()
