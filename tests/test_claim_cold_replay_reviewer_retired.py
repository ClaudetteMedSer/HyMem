"""Independent offline controls: historical citations must not resurrect evidence."""

from dataclasses import replace
import json
import sqlite3

import pytest

from hymem import HyMemConfig, portability
from hymem.core import db
from hymem.dreaming import phase1
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.producer import phase1_generation_binding
from tests.test_claim_cold_import_replay import _CLAIM_TABLES, _claim_snapshot
from tests.test_claim_semantic_dedup_guard import _chunk, _vectors
from tests.test_phase1_producer_identity import _DeclaredLLM


class _ChangingDeclaredLLM(_DeclaredLLM):
    polarity = 1
    qualifiers = None

    def complete(self, request):
        payload = json.loads(super().complete(request))
        for triple in payload["triples"]:
            triple["polarity"] = self.polarity
            triple.update(self.qualifiers or {})
        return json.dumps(payload)


@pytest.mark.parametrize("new_interpretation", ["negative", "typed"])
@pytest.mark.parametrize("dedup_enabled", [True, False])
def test_cold_replay_retired_revision_is_read_only(
    tmp_path, new_interpretation, dedup_enabled,
):
    origin = db.connect(tmp_path / "retired-origin.sqlite")
    target = db.connect(tmp_path / "retired-target.sqlite")
    for conn in (origin, target):
        db.initialize(conn)
    try:
        origin.execute("INSERT INTO sessions(id) VALUES ('s')")
        mids = [int(origin.execute(
            "INSERT INTO messages(session_id,role,content,created_at) "
            "VALUES ('s','user',?,?)", (content, timestamp),
        ).lastrowid) for content, timestamp in (
            ("App uses Redis", "2026-06-01T12:00:00Z"),
            ("App uses Redis cache", "2026-06-02T12:00:00Z"),
        )]
        with db.transaction(origin):
            materialize_message_coverage(origin, "s")
        seed, old_chunk = [
            _chunk(origin, mid, name)
            for mid, name in zip(mids, ("retired-seed", "retired-old"))
        ]
        cfg = replace(HyMemConfig(root=tmp_path), triple_dedup_enabled=True,
                      triple_dedup_cosine_threshold=0.9)
        vectors = _vectors()
        vectors["app uses redis"] = [1.0, 0.0]
        vectors["app uses redis_cache"] = [1.0, 0.0]
        client = _ChangingDeclaredLLM("retired-replay-model", "redis")
        v14 = phase1_generation_binding("v14", client)
        for chunk, object_name in ((seed, "redis"), (old_chunk, "redis_cache")):
            client.object_name = object_name
            extracted = phase1.extract_chunk_results(
                origin, chunk, client, prompt_version="v14",
                phase1_generation=v14,
            )
            assert extracted is not None and extracted.source_validated
            with db.transaction(origin):
                phase1.persist_chunk_results(
                    origin, chunk, extracted, prompt_version="v14", cfg=cfg,
                    dedup_vectors=vectors,
                )
            if chunk is seed:
                with db.embedding_mutation(origin):
                    origin.execute(
                        "INSERT INTO edge_embeddings(edge_text,vector_json,model,dim) "
                        "VALUES (?,?,?,2)",
                        ("app uses redis", json.dumps([1.0, 0.0]), vectors.model),
                    )
        assert origin.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0] == 1
        old_evidence = origin.execute(
            "SELECT evidence_id FROM kg_claim_observations WHERE chunk_id=?",
            (old_chunk.id,),
        ).fetchone()[0]

        newer = _chunk(origin, mids[1], "retired-newer")
        client.object_name = "redis"
        client.polarity = -1 if new_interpretation == "negative" else 1
        client.qualifiers = (
            {"value_numeric": 42, "value_unit": "ms", "temporal_scope": "2026"}
            if new_interpretation == "typed" else None
        )
        v15 = phase1_generation_binding("v15", client)
        changed = phase1.extract_chunk_results(
            origin, newer, client, prompt_version="v15", phase1_generation=v15,
        )
        assert changed is not None and changed.source_validated
        with db.transaction(origin):
            phase1.persist_chunk_results(
                origin, newer, changed, prompt_version="v15", cfg=cfg,
                dedup_vectors=vectors,
            )
        assert origin.execute(
            "SELECT is_current FROM kg_evidence WHERE id=?", (old_evidence,),
        ).fetchone()[0] == 0
        assert origin.execute(
            "SELECT evidence_id FROM kg_claim_observations WHERE chunk_id=?",
            (old_chunk.id,),
        ).fetchone()[0] == old_evidence

        wire = tmp_path / "retired.jsonl"
        portability.export_jsonl(origin, wire)
        portability.import_jsonl(target, wire)
        assert target.execute("SELECT COUNT(*) FROM edge_embeddings").fetchone()[0] == 0
        imported_old = target.execute(
            "SELECT ev.id,ev.is_current FROM kg_claim_observations obs "
            "JOIN kg_evidence ev ON ev.id=obs.evidence_id WHERE obs.chunk_id=?",
            (old_chunk.id,),
        ).fetchone()
        assert imported_old["is_current"] == 0
        client.object_name, client.polarity, client.qualifiers = "redis_cache", 1, None
        replay = phase1.extract_chunk_results(
            target, old_chunk, client, prompt_version="v14", phase1_generation=v14,
        )
        assert replay is not None and replay.source_validated
        replay_cfg = replace(cfg, triple_dedup_enabled=dedup_enabled)
        result_hash = target.execute(
            "SELECT result_hash FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (old_chunk.id,),
        ).fetchone()[0]
        assert phase1._matches_historical_citations(
            target, old_chunk, replay, cfg=replay_cfg, result_hash=result_hash,
        )
        before = _claim_snapshot(target)
        writes = []

        def audit(action, table, *_args):
            if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE,
                          sqlite3.SQLITE_DELETE) and table in _CLAIM_TABLES:
                writes.append((action, table))
            return sqlite3.SQLITE_OK

        target.set_authorizer(audit)
        try:
            with db.transaction(target):
                phase1.persist_chunk_results(
                    target, old_chunk, replay, prompt_version="v14", cfg=replay_cfg,
                    dedup_vectors=None, in_cycle_edges=phase1.new_in_cycle_pool(),
                )
        finally:
            target.set_authorizer(None)
        assert writes == []
        assert _claim_snapshot(target) == before
        assert target.execute(
            "SELECT is_current FROM kg_evidence WHERE id=?", (imported_old["id"],),
        ).fetchone()[0] == 0
        assert target.execute(
            "SELECT COUNT(*) FROM current_phase1_publications WHERE chunk_id=?",
            (old_chunk.id,),
        ).fetchone()[0] == 1
        assert target.execute(
            "SELECT local_replay_proof FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (old_chunk.id,),
        ).fetchone()[0] is None
        assert target.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        origin.close()
        target.close()
