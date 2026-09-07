"""Semantic attestation coverage for reusable MSC/LoCoMo SQLite stores."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import msc_adapter as msc
from benchmarks import strictness
from benchmarks.msc_adapter import MSCAdapter, prepare_indexing
from benchmarks.strictness import BenchmarkIntegrityError
from benchmarks.store_attestation import (
    MATERIAL_STORE_ATTESTATION_VERSION,
    MaterialStoreAttestationError,
    material_store_state,
)
from hymem import HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.contrib.openai_embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from hymem.dreaming.status import DREAM_STATUS_SCHEMA_VERSION
from hymem.dreaming.retention import prune_bookkeeping
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.extraction.embeddings import StubEmbeddingClient
from hymem.extraction.llm import StubLLMClient


def _hy(root: Path) -> HyMem:
    return HyMem(
        HyMemConfig(root=root, aggregation_nodes_enabled=False),
        llm=StubLLMClient(default='{"triples":[],"markers":[],"complete":true}'),
    )


def _item(text: str = "remember this") -> dict:
    return {
        "id": "conv",
        "sessions": [[{"role": "user", "content": text}]],
        "session_dates": ["2025-01-01"],
    }


def _healthy_indexing() -> dict:
    totals = {field: 0 for field in msc._DREAM_REPORT_TOTAL_FIELDS}
    final_cycle = {
        **{field: 0 for field in msc._CURRENT_DREAM_REPORT_FAILURE_FIELDS},
        **{field: False for field in msc._CURRENT_DREAM_REPORT_BOOLEAN_FIELDS},
    }
    final_status = {
        "dream_status_schema": DREAM_STATUS_SCHEMA_VERSION,
        "benchmark_indexing_status_schema": (
            strictness.BENCHMARK_INDEXING_STATUS_VERSION
        ),
        **{
            field: 0 for field in msc._FINAL_STATUS_HEALTH_FIELDS
            if field not in {
                "dream_status_schema", "benchmark_indexing_status_schema",
                "in_progress", "terminal_loss_reasons",
                "coverage_integrity_failure_reasons",
                "coverage_integrity_failure_details",
                "coverage_integrity_failure_details_truncated",
                "coverage_integrity_config_version",
                "phase1_backlog_status", "pending_chunks_authoritative",
                "phase1_generation_key",
            }
        },
        "phase1_backlog_status": "current_producer",
        "pending_chunks_authoritative": True,
        "phase1_generation_key": "hymem-phase1-generation-v1:" + "a" * 64,
        "terminal_loss_reasons": {},
        "coverage_integrity_failure_reasons": {},
        "coverage_integrity_failure_details": [],
        "coverage_integrity_failure_details_truncated": False,
        "coverage_integrity_config_version": (
            msc.COVERAGE_INTEGRITY_CONFIG_VERSION
        ),
        **{
            field: None
            for field in (
                *msc.DREAM_STATUS_AGGREGATION_AUTHORITY_FIELDS,
                *msc.DREAM_STATUS_AGGREGATION_MATERIAL_AUTHORITY_FIELDS,
                "aggregation_publication_generation",
                "aggregation_material_binding",
                "aggregation_material_revision",
            )
        },
        "aggregation_enabled": False,
        "in_progress": False,
    }
    return {
        "protocol": msc.INDEXING_PROVENANCE_VERSION,
        "scope_id": "locomo:conv",
        "mode": "converged",
        "comparable": True,
        "complete": True,
        "healthy": True,
        "convergence_count": 1,
        "cycles": 1,
        "settings": {
            "max_cycles_per_convergence": 2,
            "timeout_s_per_convergence": 10.0,
            "require_healthy": True,
        },
        "runs": [{
            "trigger": "end_of_history",
            "cycles": 1,
            "report_count": 1,
            "complete": True,
            "healthy": True,
            "failure_reason": None,
            "elapsed_s": 0.1,
            "dream_report_totals": dict(totals),
            "budget_exhausted_cycles": 0,
            "extraction_provider_attempt_budget_exhausted_cycles": 0,
            "skipped_locked_cycles": 0,
            "final_cycle": final_cycle,
            "final_status": dict(final_status),
        }],
        "dream_report_totals": totals,
        "budget_exhausted_cycles": 0,
        "extraction_provider_attempt_budget_exhausted_cycles": 0,
        "skipped_locked_cycles": 0,
        "final_status": final_status,
        "pipeline_usage": msc._known_zero_pipeline_usage(),
    }


def _adapter(root: Path) -> MSCAdapter:
    return MSCAdapter(root / "hymem.sqlite", sim=True).open()


def _write_material_code_tree(root: Path) -> dict[str, Path]:
    """Small relocatable tree exercising every supported import shape."""

    sources = {
        "benchmarks/msc_adapter.py": """
class MSCAdapter:
    def open(self):
        from hymem import HyMem, HyMemConfig
        try:
            import hymem.contrib.openai_client as pipeline_module
        except ImportError:
            from hymem.extraction import llm as pipeline_module
        from hymem.contrib.openai_embedding_client import Client as Embedder
        return HyMem, HyMemConfig, pipeline_module, Embedder

    def ingest(self):
        return self.hy.log_messages([])

    def dream(self):
        return self.hy.dream()

    def _durable_status(self):
        return self.hy.indexing_status()
""",
        "hymem/__init__.py": """
from hymem.api import HyMem
from hymem.config import HyMemConfig
""",
        "hymem/api.py": """
from hymem.core import db as core_db
from hymem.deadline import DEADLINE_POLICY
from hymem.query.graph_state import GRAPH_STATE_POLICY
class HyMem:
    pass
""",
        "hymem/config.py": "class HyMemConfig:\n    pass\n",
        "hymem/deadline.py": "DEADLINE_POLICY = 'deadline-v1'\n",
        "hymem/contrib/__init__.py": "CONTRIB_PACKAGE = 'v1'\n",
        "hymem/contrib/openai_client.py": """
from hymem.contrib.endpoint_policy import ENDPOINT_POLICY
from hymem.contrib.model_policy import MODEL_POLICY
from hymem.extraction.llm import LLM_POLICY
CLIENT_POLICY = ENDPOINT_POLICY + MODEL_POLICY + LLM_POLICY
""",
        "hymem/contrib/openai_embedding_client.py": """
from hymem.contrib.endpoint_policy import ENDPOINT_POLICY
from hymem.deadline import DEADLINE_POLICY
class Client:
    pass
""",
        "hymem/contrib/endpoint_policy.py": "ENDPOINT_POLICY = 'endpoint-v1'\n",
        "hymem/contrib/model_policy.py": "MODEL_POLICY = 'model-v1'\n",
        "hymem/core/__init__.py": "CORE_PACKAGE = 'v1'\n",
        "hymem/core/db.py": "DB_POLICY = 'db-v1'\n",
        "hymem/core/schema.sql": "CREATE TABLE material_v1 (id INTEGER);\n",
        "hymem/core/migrations/__init__.py": "MIGRATION_PACKAGE = 'v1'\n",
        "hymem/core/migrations/001_material.sql": (
            "ALTER TABLE material_v1 ADD COLUMN note TEXT;\n"
        ),
        "hymem/extraction/__init__.py": "EXTRACTION_PACKAGE = 'v1'\n",
        "hymem/extraction/llm.py": "LLM_POLICY = 'llm-v1'\n",
        "hymem/query/__init__.py": "QUERY_PACKAGE = 'v1'\n",
        "hymem/query/graph_state.py": "GRAPH_STATE_POLICY = 'graph-v1'\n",
        # A retrieval-only module which neither package initialization nor the
        # material entrypoints import. It must remain outside store identity.
        "hymem/query/reader_only.py": "READER_POLICY = 'reader-v1'\n",
    }
    paths: dict[str, Path] = {"root": root}
    for relative, source in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.lstrip(), encoding="utf-8")
        paths[relative] = path
    paths["hymem"] = root / "hymem"
    paths["adapter"] = root / "benchmarks/msc_adapter.py"
    return paths


def _material_code_hash(paths: dict[str, Path]) -> str:
    # Mutation/relocation tests must never observe a prior lru-cached digest.
    msc._material_hymem_code_hash.cache_clear()
    return msc._material_hymem_code_hash(
        adapter_path=paths["adapter"],
        hymem_path=paths["hymem"],
        root=paths["root"],
    )


def _pinned_embedding_client(monkeypatch, dimensions: list[int] | None = None):
    observed = list(dimensions or [])

    class FakeDefaultHttpxClient:
        def __init__(self, **kwargs):
            self.trust_env = kwargs.get("trust_env")

        def close(self):
            pass

    class FakeOpenAI:
        def __init__(self, **kwargs):
            def create(*, model, input, timeout=None):
                del model
                assert timeout is None or timeout > 0
                dimension = observed.pop(0) if observed else 3
                return SimpleNamespace(data=[
                    SimpleNamespace(
                        index=index,
                        embedding=[1.0] + [0.0] * (dimension - 1),
                    )
                    for index, _text in enumerate(input)
                ])

            self.embeddings = SimpleNamespace(create=create)
            self._client = kwargs["http_client"]
            self.api_key = kwargs.get("api_key")
            self.base_url = kwargs.get("base_url")
            self.organization = None
            self.project = None

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "openai",
        SimpleNamespace(
            OpenAI=FakeOpenAI, DefaultHttpxClient=FakeDefaultHttpxClient,
        ),
    )
    return OpenAICompatibleEmbeddingClient(
        api_key="secret-never-persist",
        base_url="https://embedding.example/v1",
        model="embedding-model",
        dim=3,
        pin_dimension=True,
        deployment_revision="public-revision-v1",
        deployment_tenant="public-tenant-v1",
    )


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    (
        ("hymem/query/graph_state.py", "graph-v1", "graph-v2"),
        ("hymem/deadline.py", "deadline-v1", "deadline-v2"),
        ("hymem/contrib/endpoint_policy.py", "endpoint-v1", "endpoint-v2"),
        ("hymem/contrib/model_policy.py", "model-v1", "model-v2"),
    ),
)
def test_material_code_identity_follows_transitive_hymem_dependencies(
    tmp_path, relative, old, new,
):
    paths = _write_material_code_tree(tmp_path / "source")
    before = _material_code_hash(paths)
    source = paths[relative].read_text(encoding="utf-8")
    paths[relative].write_text(source.replace(old, new), encoding="utf-8")

    assert _material_code_hash(paths) != before
    msc._material_hymem_code_hash.cache_clear()


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    (
        ("hymem/core/schema.sql", "material_v1", "material_v2"),
        ("hymem/core/migrations/001_material.sql", "note TEXT", "note BLOB"),
        ("hymem/core/migrations/__init__.py", "'v1'", "'v2'"),
    ),
)
def test_material_code_identity_includes_runtime_database_resources(
    tmp_path, relative, old, new,
):
    paths = _write_material_code_tree(tmp_path / "source")
    before = _material_code_hash(paths)
    source = paths[relative].read_text(encoding="utf-8")
    paths[relative].write_text(source.replace(old, new), encoding="utf-8")

    assert _material_code_hash(paths) != before
    msc._material_hymem_code_hash.cache_clear()


def test_material_code_identity_is_relocatable_and_excludes_reader_only_file(
    tmp_path, monkeypatch,
):
    left = _write_material_code_tree(tmp_path / "checkout-a")
    right = _write_material_code_tree(tmp_path / "checkout-b")
    expected = _material_code_hash(left)

    assert _material_code_hash(left) == expected
    assert _material_code_hash(right) == expected

    reader = left["hymem/query/reader_only.py"]
    reader.write_text("READER_POLICY = 'reader-v2'\n", encoding="utf-8")
    assert _material_code_hash(left) == expected

    # A full-run code protocol bump must not evict a proven material store;
    # material identity owns its independent framing version.
    monkeypatch.setattr(strictness, "CODE_IDENTITY_VERSION", "benchmark-code-v-next")
    assert _material_code_hash(left) == expected
    msc._material_hymem_code_hash.cache_clear()


def _configure_empty_indexing_llm(adapter: MSCAdapter) -> None:
    llm = StubLLMClient(
        fixtures={
            "Return the JSON object now": json.dumps({
                "episodes": [], "summary": "", "procedures": [],
            }),
        },
        default=json.dumps({
            "triples": [], "markers": [], "complete": True,
        }),
    )
    adapter.pipeline_llm = llm
    adapter.hy.set_llm(llm)
    adapter.hy.config = replace(
        adapter.hy.config,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        episode_granularity_enabled=False,
    )


def _changed_tables(left: dict, right: dict) -> set[str]:
    return {
        name for name in set(left["tables"]) | set(right["tables"])
        if left["tables"].get(name) != right["tables"].get(name)
    }


def test_material_state_detects_append_modify_and_delete_source_rows(tmp_path):
    hy = _hy(tmp_path)
    try:
        hy.log_messages("source", [("user", "alpha memory", "2025-01-01")])
        baseline = material_store_state(hy.config.db_path)

        hy.log_messages("unrelated", [("assistant", "beta memory", "2025-01-02")])
        appended = material_store_state(hy.config.db_path)
        assert appended["sha256"] != baseline["sha256"]
        assert {"sessions", "messages", "messages_fts"} <= _changed_tables(
            baseline, appended
        )

        hy.conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            ("2024-12-31", "unrelated"),
        )
        modified = material_store_state(hy.config.db_path)
        assert modified["sha256"] != appended["sha256"]
        assert "sessions" in _changed_tables(appended, modified)

        hy.conn.execute(
            "INSERT INTO sessions(id,started_at) VALUES (?,?)",
            ("deletable-empty-source", "2025-01-03"),
        )
        before_delete = material_store_state(hy.config.db_path)
        hy.conn.execute(
            "DELETE FROM sessions WHERE id=?", ("deletable-empty-source",)
        )
        deleted = material_store_state(hy.config.db_path)
        assert deleted["sha256"] != before_delete["sha256"]
        assert _changed_tables(before_delete, deleted) == {"sessions"}
    finally:
        hy.close()


def test_material_state_detects_derived_embedding_and_logical_fts_mutation(tmp_path):
    hy = _hy(tmp_path)
    try:
        hy.log_messages("source", [("user", "uniquequokka memory", "2025-01-01")])
        baseline = material_store_state(hy.config.db_path)

        hy.conn.execute(
            "INSERT INTO entity_aliases(alias,canonical) VALUES (?,?)",
            ("Quokka", "quokka"),
        )
        derived = material_store_state(hy.config.db_path)
        assert _changed_tables(baseline, derived) == {"entity_aliases"}

        model, dimension = embedding_storage_identity(
            StubEmbeddingClient(model_name="embed-test", dim_value=2)
        )
        with core_db.embedding_mutation(hy.conn):
            hy.conn.execute(
                "INSERT INTO embedding_cache(text_hash,model,vector_json,dim) "
                "VALUES (?,?,?,?)",
                ("sha256:cache", model, "[1.0,0.0]", dimension),
            )
        embedded = material_store_state(hy.config.db_path)
        assert _changed_tables(derived, embedded) == {"embedding_cache"}

        source = hy.conn.execute(
            "SELECT id,content FROM messages WHERE session_id='source'"
        ).fetchone()
        hy.conn.execute(
            "INSERT INTO messages_fts(messages_fts,rowid,content) "
            "VALUES ('delete',?,?)",
            (source["id"], source["content"]),
        )
        corrupted_fts = material_store_state(hy.config.db_path)
        assert _changed_tables(embedded, corrupted_fts) == {"messages_fts"}
    finally:
        hy.close()


def test_logical_vec_contents_are_attested_not_their_shadows(tmp_path):
    hy = _hy(tmp_path)
    try:
        model, dimension = embedding_storage_identity(
            StubEmbeddingClient(model_name="embed-test", dim_value=2)
        )
        hy.conn.execute("INSERT INTO sessions(id) VALUES ('s')")
        hy.conn.execute(
            "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
            "salience_reason,text) VALUES ('c','s',1,1,'test','vector text')"
        )
        core_db.ensure_vec_table(hy.conn, dimension, model=model)
        with core_db.embedding_mutation(hy.conn):
            hy.conn.execute(
                "INSERT INTO chunk_embeddings("
                "chunk_id,vector_json,model,dim,text_hash) VALUES (?,?,?,?,?)",
                ("c", "[1.0,0.0]", model, dimension, "sha256:text"),
            )
        rowid = hy.conn.execute(
            "SELECT rowid FROM chunks WHERE id='c'"
        ).fetchone()[0]
        hy.conn.execute(
            "INSERT INTO vec_chunks(rowid,embedding) VALUES (?,?)",
            (rowid, core_db._pack_vector([1.0, 0.0])),
        )
        baseline = material_store_state(hy.config.db_path)

        hy.conn.execute("DELETE FROM vec_chunks WHERE rowid=?", (rowid,))
        hy.conn.execute(
            "INSERT INTO vec_chunks(rowid,embedding) VALUES (?,?)",
            (rowid, core_db._pack_vector([0.0, 1.0])),
        )
        mutated = material_store_state(hy.config.db_path)

        assert _changed_tables(baseline, mutated) == {"vec_chunks"}
        assert all("vector_chunks" not in name for name in mutated["tables"])
    finally:
        hy.close()


def test_null_float_and_blob_values_are_framed_without_serialization(tmp_path):
    hy = _hy(tmp_path)
    try:
        hy.conn.execute(
            "INSERT INTO entity_types(entity_canonical,type,confidence,source_chunk_id) "
            "VALUES ('entity','project',0.25,NULL)"
        )
        hy.conn.execute(
            "INSERT INTO peers(id,workspace_id,role,metadata,registered_at) "
            "VALUES ('peer','workspace','user',?,NULL)",
            (sqlite3.Binary(b"\x00raw-secret-bytes\xff"),),
        )
        baseline = material_store_state(hy.config.db_path)
        assert "raw-secret-bytes" not in json.dumps(baseline)

        hy.conn.execute(
            "UPDATE entity_types SET confidence=0.5 WHERE entity_canonical='entity'"
        )
        float_changed = material_store_state(hy.config.db_path)
        assert _changed_tables(baseline, float_changed) == {"entity_types"}

        hy.conn.execute(
            "UPDATE peers SET metadata=? WHERE id='peer' AND workspace_id='workspace'",
            (sqlite3.Binary(b"\x00different\xff"),),
        )
        blob_changed = material_store_state(hy.config.db_path)
        assert _changed_tables(float_changed, blob_changed) == {"peers"}
    finally:
        hy.close()


def test_semantic_digest_ignores_insertion_page_order_and_vacuum(tmp_path):
    left = _hy(tmp_path / "left")
    right = _hy(tmp_path / "right")
    try:
        pairs = (("A", "a"), ("B", "b"), ("C", "c"))
        left.conn.executemany(
            "INSERT INTO entity_aliases(alias,canonical) VALUES (?,?)", pairs
        )
        right.conn.executemany(
            "INSERT INTO entity_aliases(alias,canonical) VALUES (?,?)",
            reversed(pairs),
        )
        right.conn.execute("VACUUM")

        assert material_store_state(left.config.db_path) == material_store_state(
            right.config.db_path
        )

        left.log_messages(
            "vacuum-source",
            [
                ("user", "alpha beta gamma", "2025-01-01"),
                ("assistant", "delta echo", "2025-01-02"),
            ],
        )
        before_vacuum = material_store_state(left.config.db_path)
        left.conn.execute("VACUUM")
        assert material_store_state(left.config.db_path) == before_vacuum
    finally:
        left.close()
        right.close()


def test_regenerated_derived_edge_surrogate_and_clocks_are_not_material(tmp_path):
    hy = _hy(tmp_path)
    try:
        hy.conn.execute(
            "INSERT INTO knowledge_graph("
            "subject_canonical,predicate,object_canonical,pos_evidence,neg_evidence,"
            "first_seen,last_seen,valid_at,derived) VALUES (?,?,?,?,?,?,?,?,1)",
            ("a", "depends_on", "c", 1, 0, "2024-01-01", "2024-01-02", "2024-01-01"),
        )
        baseline = material_store_state(hy.config.db_path)
        old_id = hy.conn.execute(
            "SELECT id FROM knowledge_graph WHERE derived=1"
        ).fetchone()[0]
        hy.conn.execute("DELETE FROM knowledge_graph WHERE id=?", (old_id,))
        hy.conn.execute(
            "INSERT INTO knowledge_graph("
            "subject_canonical,predicate,object_canonical,pos_evidence,neg_evidence,"
            "first_seen,last_seen,valid_at,derived) VALUES (?,?,?,?,?,?,?,?,1)",
            ("a", "depends_on", "c", 1, 0, "2030-01-01", "2030-01-02", "2030-01-01"),
        )
        regenerated = material_store_state(hy.config.db_path)

        assert hy.conn.execute(
            "SELECT id FROM knowledge_graph WHERE derived=1"
        ).fetchone()[0] != old_id
        assert regenerated == baseline
    finally:
        hy.close()


def test_actual_noop_dream_keeps_regenerated_derived_graph_digest_stable(tmp_path):
    adapter = _adapter(tmp_path)
    try:
        adapter.hy.conn.executemany(
            "INSERT INTO knowledge_graph("
            "subject_canonical,predicate,object_canonical,pos_evidence,derived) "
            "VALUES (?, 'depends_on', ?, 5, 0)",
            (("a", "b"), ("b", "c")),
        )
        adapter.dream(max_cycles=2, timeout_s=10)
        baseline = adapter.material_store_state()
        first = adapter.hy.conn.execute(
            "SELECT id FROM knowledge_graph WHERE derived=1"
        ).fetchone()
        assert first is not None

        adapter.dream(max_cycles=2, timeout_s=10)
        regenerated = adapter.hy.conn.execute(
            "SELECT id FROM knowledge_graph WHERE derived=1"
        ).fetchone()

        assert regenerated is not None
        assert regenerated[0] != first[0]
        assert adapter.material_store_state() == baseline
    finally:
        adapter.close()


def test_unknown_application_table_fails_closed_with_bounded_identifier(tmp_path):
    hy = _hy(tmp_path)
    try:
        name = "future_material_" + "x" * 120
        hy.conn.execute(f'CREATE TABLE "{name}" (value TEXT)')
        with pytest.raises(MaterialStoreAttestationError) as failed:
            material_store_state(hy.config.db_path)
        assert failed.value.reason == "unknown_application_tables"
        assert len(failed.value.tables[0]) == 96
        assert "value" not in failed.value.safe_details()
    finally:
        hy.close()


@pytest.mark.parametrize(
    "drop_sql",
    [
        "DROP VIEW aggregation_live_material",
        "DROP TRIGGER aggregation_material_clock_update_unpublishes",
        "DROP INDEX idx_v57_episode_source_message",
    ],
)
def test_material_attestation_requires_exact_v57_support_objects(
    tmp_path, drop_sql,
):
    hy = _hy(tmp_path)
    try:
        hy.conn.execute(drop_sql)
        with pytest.raises(MaterialStoreAttestationError) as failed:
            material_store_state(hy.config.db_path)
        assert (
            failed.value.reason
            == "current_material_schema_boundary_is_malformed"
        )
        assert failed.value.safe_details()["attestation_tables"] == []
    finally:
        hy.close()


def test_noop_dream_does_not_change_material_digest(tmp_path):
    adapter = _adapter(tmp_path)
    try:
        adapter.dream(max_cycles=2, timeout_s=10)
        before = adapter.material_store_state()
        adapter.dream(max_cycles=2, timeout_s=10)
        after = adapter.material_store_state()

        assert before == after
        assert before["version"] == MATERIAL_STORE_ATTESTATION_VERSION
        assert "dream_runs" not in before["tables"]
        assert "aggregation_build_health" not in before["tables"]
    finally:
        adapter.close()


def test_operational_retry_fields_do_not_change_material_digest(tmp_path):
    hy = _hy(tmp_path)
    try:
        hy.conn.execute("INSERT INTO sessions(id) VALUES ('retry-source')")
        baseline = material_store_state(hy.config.db_path)

        hy.conn.execute(
            "UPDATE sessions SET "
            "profile_retry_count=2, profile_retry_config_version='profile.v9', "
            "profile_quarantined=1, facts_retry_count=3, "
            "facts_retry_config_version='facts.v9', facts_quarantined=1, "
            "digest_retry_count=4, digest_retry_config_version='digest.v9', "
            "digest_quarantined=1 WHERE id='retry-source'"
        )

        assert material_store_state(hy.config.db_path) == baseline
    finally:
        hy.close()


def test_retraction_audit_rows_and_pruning_are_not_material_state(tmp_path):
    hy = _hy(tmp_path)
    try:
        hy.conn  # initialize the lazy file-backed store
        baseline = material_store_state(hy.config.db_path)
        hy.conn.execute(
            "INSERT INTO extraction_feedback("
            "chunk_text_snippet,extracted_subject,extracted_predicate,"
            "extracted_object) VALUES ('audit','service','uses','db')"
        )
        appended = material_store_state(hy.config.db_path)

        assert appended == baseline
        assert "extraction_feedback" not in appended["tables"]

        discard = replace(hy.config, extraction_feedback_keep=0)
        assert prune_bookkeeping(hy.conn, discard) == 1
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM extraction_feedback"
        ).fetchone()[0] == 0
        assert material_store_state(hy.config.db_path) == baseline
    finally:
        hy.close()


def test_source_materialization_ack_is_part_of_current_material_digest(tmp_path):
    hy = _hy(tmp_path)
    try:
        hy.conn.execute(
            "INSERT INTO sessions(id,coverage_message_id) VALUES ('source-ack',1)"
        )
        baseline = material_store_state(hy.config.db_path)
        hy.conn.execute(
            "UPDATE sessions SET source_materialized_message_id=1,"
            "source_materialization_config_version='producer-v1' "
            "WHERE id='source-ack'"
        )
        changed = material_store_state(hy.config.db_path)
        assert changed["version"] == MATERIAL_STORE_ATTESTATION_VERSION
        assert changed["sha256"] != baseline["sha256"]
        assert changed["tables"]["sessions"] != baseline["tables"]["sessions"]
    finally:
        hy.close()


def test_successful_reuse_attests_before_and_after_noop_convergence(tmp_path):
    adapter = _adapter(tmp_path)
    item = {"id": "conv", "sessions": [], "session_dates": []}
    args = SimpleNamespace(
        sim=False,
        no_dream=False,
        dream_per_session=False,
        indexing_max_cycles=2,
        indexing_timeout_s=10,
    )
    try:
        adapter.dream(max_cycles=2, timeout_s=10)
        receipt = adapter.publish_store_build_receipt(
            item, adapter.indexing_provenance(scope_id="locomo:conv")
        )
        adapter.indexing_runs.clear()

        indexing = prepare_indexing(
            adapter, item, args, scope_id="locomo:conv", reuse=True
        )

        assert indexing["store_build_receipt"] == {
            "version": msc.STORE_BUILD_RECEIPT_VERSION,
            "status": "validated",
            "identity_sha256": receipt["identity_sha256"],
            "indexing_sha256": receipt["indexing_sha256"],
            "material_state_sha256": receipt["material_state"]["sha256"],
            "file": ".hymem-benchmark-store-build.json",
        }
        assert adapter.material_store_state() == receipt["material_state"]
    finally:
        adapter.close()


@pytest.mark.parametrize("scope_id", ("msc:conv", "locomo:conv"))
def test_current_msc_and_locomo_provenance_publish_and_validate(
    tmp_path, scope_id,
):
    adapter = _adapter(tmp_path / scope_id.split(":", 1)[0])
    item = {"id": "conv", "sessions": [], "session_dates": []}
    try:
        adapter.dream(max_cycles=2, timeout_s=10)
        indexing = adapter.indexing_provenance(scope_id=scope_id)
        receipt = adapter.publish_store_build_receipt(item, indexing)

        assert receipt["indexing"]["protocol"] == (
            msc.INDEXING_PROVENANCE_VERSION
        )
        assert receipt["indexing"]["scope_id"] == scope_id
        assert receipt["indexing_sha256"] == strictness.content_hash(
            receipt["indexing"]
        )
        assert adapter.validate_store_build_receipt(item) == receipt
    finally:
        adapter.close()


def test_store_receipt_reuse_ignores_only_retraction_audit_retention(tmp_path):
    adapter = _adapter(tmp_path)
    item = {"id": "conv", "sessions": [], "session_dates": []}
    try:
        receipt = adapter.publish_store_build_receipt(
            item, _healthy_indexing()
        )
        identity = adapter.store_build_identity(item)
        adapter.hy.config = replace(
            adapter.hy.config, extraction_feedback_keep=0
        )

        assert adapter.store_build_identity(item) == identity
        assert "extraction_feedback_keep" not in identity["write_config"]
        adapter.hy.conn.execute(
            "INSERT INTO extraction_feedback("
            "chunk_text_snippet,extracted_subject,extracted_predicate,"
            "extracted_object) VALUES ('audit','service','uses','db')"
        )
        assert adapter.validate_store_build_receipt(item) == receipt

        assert prune_bookkeeping(adapter.hy.conn, adapter.hy.config) == 1
        assert adapter.validate_store_build_receipt(item) == receipt
    finally:
        adapter.close()


def test_receipt_contains_only_digest_of_secret_source_and_rejects_contamination(
    tmp_path,
):
    secret = "source-secret-never-serialize"
    item = _item(secret)
    adapter = _adapter(tmp_path)
    try:
        adapter.ingest(item)
        receipt = adapter.publish_store_build_receipt(item, _healthy_indexing())
        serialized = adapter.store_build_receipt_path.read_text(encoding="utf-8")

        assert secret not in serialized
        assert set(receipt["material_state"]) == {"version", "sha256", "tables"}
        assert receipt["identity"]["embedding"] == (
            msc._benchmark_embedding_identity(None)
        )
        assert receipt["embedding_state"]["enabled"] is False
        assert receipt["embedding_state"]["total_vectors"] == 0
        assert adapter.validate_store_build_receipt(item) == receipt

        adapter.hy.log_messages(
            "foreign-session", [("user", "unrelated contamination", "2025-02-01")]
        )
        with pytest.raises(strictness.IndexingConvergenceError) as failed:
            adapter.validate_store_build_receipt(item)
        assert failed.value.summary["failure_reason"] == "store_material_state_mismatch"
        assert "foreign-session" not in json.dumps(failed.value.summary)
        assert "messages" in failed.value.summary["mismatch_tables"]
    finally:
        adapter.close()


def test_embedding_build_identity_is_stable_before_and_after_provider_call(
    tmp_path, monkeypatch,
):
    adapter = _adapter(tmp_path)
    client = _pinned_embedding_client(monkeypatch)
    adapter.embedding_client = client
    adapter.hy.set_embedding_client(client)
    try:
        before = adapter.store_build_identity(_item())
        assert client.embed(["identity probe"]) == [[1.0, 0.0, 0.0]]
        after = adapter.store_build_identity(_item())

        assert before == after
        assert after["embedding"]["dimension"] == 3
        declaration = after["embedding"]["producer_binding"]["declaration"]
        assert declaration["configured_dimension"] == 3
        assert declaration["dimension_policy"] == "pinned"
        assert client.observed_dim == 3
    finally:
        client.close()
        adapter.close()


def test_later_provider_dimension_drift_prevents_receipt_publication(
    tmp_path, monkeypatch,
):
    adapter = _adapter(tmp_path)
    client = _pinned_embedding_client(monkeypatch, [3, 4])
    adapter.embedding_client = client
    adapter.hy.set_embedding_client(client)
    item = _item("a durable source message")
    try:
        # Hot ingestion persists a valid first batch in the configured space.
        adapter.ingest(item)
        assert adapter.hy.read_conn.execute(
            "SELECT COUNT(*) FROM message_embeddings WHERE dim=3"
        ).fetchone()[0] == 1

        with pytest.raises(RuntimeError, match="pinned configured dimension"):
            client.embed(["later contradictory batch"])
        assert client.dimension_integrity_ok is False

        with pytest.raises(
            BenchmarkIntegrityError,
            match="benchmark embedding client identity is invalid",
        ):
            adapter.publish_store_build_receipt(item, _healthy_indexing())
        assert not adapter.store_build_receipt_path.exists()
    finally:
        client.close()
        adapter.close()


def test_caught_first_dimension_mismatch_remains_fatal_after_recovery(
    tmp_path, monkeypatch,
):
    adapter = _adapter(tmp_path)
    client = _pinned_embedding_client(monkeypatch, [2, 3])
    adapter.embedding_client = client
    adapter.hy.set_embedding_client(client)
    item = _item("first batch is rejected but its source remains durable")
    try:
        # HyMem's hot-ingest path catches provider errors by design.
        adapter.ingest(item)
        assert client.dimension_integrity_ok is False
        assert adapter.hy.read_conn.execute(
            "SELECT COUNT(*) FROM message_embeddings"
        ).fetchone()[0] == 0

        # Even a later matching response cannot erase the observed conflict.
        assert client.embed(["recovered batch"]) == [[1.0, 0.0, 0.0]]
        with pytest.raises(
            BenchmarkIntegrityError,
            match="benchmark embedding client identity is invalid",
        ):
            adapter.publish_store_build_receipt(item, _healthy_indexing())
        assert not adapter.store_build_receipt_path.exists()
    finally:
        client.close()
        adapter.close()


def test_query_dimension_mismatch_cannot_degrade_to_scored_lexical_retrieval(
    tmp_path, monkeypatch,
):
    adapter = _adapter(tmp_path)
    client = _pinned_embedding_client(monkeypatch, [4])
    adapter.embedding_client = client
    adapter.hy.set_embedding_client(client)
    try:
        with pytest.raises(
            BenchmarkIntegrityError,
            match="configured benchmark embedding retrieval was unavailable",
        ):
            adapter.search("which migration was chosen?", top_k=3)
        assert client.dimension_integrity_ok is False
        attempts = client.request_attempts
        with pytest.raises(BenchmarkIntegrityError):
            adapter.search("do not retry a contradicted provider", top_k=3)
        assert client.request_attempts == attempts
    finally:
        client.close()
        adapter.close()


def test_embedding_receipt_attests_vectors_and_reuses_without_provider_probe(
    tmp_path, monkeypatch,
):
    item = _item("remember the cobalt migration decision")
    first = _adapter(tmp_path)
    _configure_empty_indexing_llm(first)
    first_client = _pinned_embedding_client(monkeypatch)
    first.embedding_client = first_client
    first.hy.set_embedding_client(first_client)
    try:
        first.ingest(item)
        first.dream(max_cycles=10, timeout_s=10)
        receipt = first.publish_store_build_receipt(
            item, first.indexing_provenance(scope_id="locomo:conv")
        )
        assert receipt["embedding_state"]["enabled"] is True
        assert receipt["embedding_state"]["dimension"] == 3
        assert receipt["embedding_state"]["total_vectors"] > 0
        assert receipt["embedding_state"]["vector_space_sha256"].startswith(
            "sha256:"
        )
        assert "secret-never-persist" not in json.dumps(receipt)
    finally:
        first_client.close()
        first.close()

    reopened = _adapter(tmp_path)
    _configure_empty_indexing_llm(reopened)
    reopened_client = _pinned_embedding_client(monkeypatch)
    reopened.embedding_client = reopened_client
    reopened.hy.set_embedding_client(reopened_client)
    args = SimpleNamespace(
        sim=False,
        no_dream=False,
        dream_per_session=False,
        indexing_max_cycles=10,
        indexing_timeout_s=10,
    )
    try:
        indexing = prepare_indexing(
            reopened, item, args, scope_id="locomo:conv", reuse=True
        )
        assert indexing["store_build_receipt"]["status"] == "validated"
        assert reopened_client.request_attempts == 0
        assert reopened.validate_store_build_receipt(item) == receipt
    finally:
        reopened_client.close()
        reopened.close()


def test_embedding_receipt_refuses_wrong_dimension_in_durable_mirror(
    tmp_path, monkeypatch,
):
    adapter = _adapter(tmp_path)
    client = _pinned_embedding_client(monkeypatch)
    adapter.embedding_client = client
    adapter.hy.set_embedding_client(client)
    item = _item("durable vector attestation source")
    try:
        adapter.ingest(item)
        with core_db.embedding_mutation(adapter.hy.conn):
            adapter.hy.conn.execute("UPDATE message_embeddings SET dim=2")
        with pytest.raises(
            BenchmarkIntegrityError,
            match="stored embedding vector does not match",
        ):
            adapter.publish_store_build_receipt(item, _healthy_indexing())
        assert not adapter.store_build_receipt_path.exists()
    finally:
        client.close()
        adapter.close()


def test_reusable_receipt_rejects_legacy_unpinned_embedding_client(tmp_path):
    adapter = _adapter(tmp_path)
    adapter.embedding_client = SimpleNamespace(model="legacy", dim=3)
    try:
        with pytest.raises(
            BenchmarkIntegrityError, match="durable exact embedding producer"
        ):
            adapter.store_build_identity(_item())
    finally:
        adapter.close()


def test_msc_open_enables_pinned_dimension_for_embedding_builds(
    tmp_path, monkeypatch,
):
    pytest.importorskip("openai")
    monkeypatch.setenv("HYMEM_EMBEDDING_API_KEY", "test-embedding-key")
    monkeypatch.setenv(
        "HYMEM_EMBEDDING_BASE_URL", "https://embedding.example/v1"
    )
    monkeypatch.setenv("HYMEM_EMBEDDING_MODEL", "pinned-model")
    monkeypatch.setenv("HYMEM_EMBEDDING_DIM", "7")
    adapter = MSCAdapter(
        tmp_path / "hymem.sqlite", api_key="test-pipeline-key", embeddings=True
    ).open()
    try:
        assert adapter.embedding_client.configured_dim == 7
        assert adapter.embedding_client.dim == 7
        assert adapter.embedding_client.observed_dim is None
        assert adapter.embedding_client.dimension_policy == "pinned"
        assert adapter.embedding_client.dimension_integrity_ok is True
    finally:
        adapter.close()


def test_receipt_rejects_database_swap(tmp_path):
    store = tmp_path / "store"
    replacement = tmp_path / "replacement"
    adapter = _adapter(store)
    item = _item()
    try:
        adapter.publish_store_build_receipt(item, _healthy_indexing())
    finally:
        adapter.close()

    other = _hy(replacement)
    try:
        other.log_messages("foreign", [("user", "other database", "2025-01-01")])
    finally:
        other.close()
    shutil.copyfile(replacement / "hymem.sqlite", store / "hymem.sqlite")

    reopened = _adapter(store)
    try:
        with pytest.raises(strictness.IndexingConvergenceError) as failed:
            reopened.validate_store_build_receipt(item)
        assert failed.value.summary["failure_reason"] == "store_material_state_mismatch"
    finally:
        reopened.close()


def test_malformed_schema_identity_fails_without_echoing_row_value(tmp_path):
    adapter = _adapter(tmp_path)
    secret = "schema-row-secret-never-report"
    try:
        adapter.publish_store_build_receipt(_item(), _healthy_indexing())
        adapter.hy.conn.execute(
            "UPDATE schema_meta SET value=? WHERE key='schema_version'",
            (secret,),
        )

        with pytest.raises(strictness.IndexingConvergenceError) as failed:
            adapter.validate_store_build_receipt(_item())

        assert failed.value.summary["failure_reason"] == (
            "store_build_identity_unavailable"
        )
        assert secret not in str(failed.value)
        assert secret not in json.dumps(failed.value.summary)
        assert "--fresh" in failed.value.summary["remediation"]
    finally:
        adapter.close()


def test_mapping_hash_change_and_missing_legacy_field_fail_closed(
    monkeypatch, tmp_path,
):
    adapter = _adapter(tmp_path)
    item = _item()
    try:
        adapter.publish_store_build_receipt(item, _healthy_indexing())
        original = MSCAdapter.ingest

        def changed_mapping(self, ex, **kwargs):
            return original(self, ex, **kwargs)

        monkeypatch.setattr(MSCAdapter, "ingest", changed_mapping)
        with pytest.raises(strictness.IndexingConvergenceError) as changed:
            adapter.validate_store_build_receipt(item)
        assert changed.value.summary["failure_reason"] == "store_build_identity_mismatch"
        assert changed.value.summary["mismatch_fields"] == [
            "identity.ingestion_mapping.sha256"
        ]

        monkeypatch.setattr(MSCAdapter, "ingest", original)
        receipt = json.loads(adapter.store_build_receipt_path.read_text())
        del receipt["identity"]["ingestion_mapping"]
        receipt["identity_sha256"] = strictness.content_hash(receipt["identity"])
        adapter.store_build_receipt_path.write_text(json.dumps(receipt))
        with pytest.raises(strictness.IndexingConvergenceError) as legacy:
            adapter.validate_store_build_receipt(item)
        assert legacy.value.summary["mismatch_fields"] == [
            "identity.ingestion_mapping"
        ]
        assert "--fresh" in legacy.value.summary["remediation"]
    finally:
        adapter.close()


def test_store_receipt_rejects_legacy_incomplete_material_code_identity(tmp_path):
    adapter = _adapter(tmp_path)
    item = _item()
    try:
        adapter.publish_store_build_receipt(item, _healthy_indexing())
        receipt = json.loads(adapter.store_build_receipt_path.read_text())
        # Model an otherwise self-consistent legacy receipt whose old manual
        # inventory omitted reachable HyMem dependencies.
        receipt["identity"]["hymem_code_sha256"] = "sha256:" + "0" * 64
        receipt["identity_sha256"] = strictness.content_hash(receipt["identity"])
        adapter.store_build_receipt_path.write_text(json.dumps(receipt))

        with pytest.raises(strictness.IndexingConvergenceError) as legacy:
            adapter.validate_store_build_receipt(item)

        assert legacy.value.summary["failure_reason"] == (
            "store_build_identity_mismatch"
        )
        assert legacy.value.summary["mismatch_fields"] == [
            "identity.hymem_code_sha256"
        ]
        assert "--fresh" in legacy.value.summary["remediation"]
    finally:
        adapter.close()


def test_store_build_receipt_rejects_corrupt_embedding_attestation(tmp_path):
    adapter = _adapter(tmp_path)
    item = _item()
    try:
        adapter.publish_store_build_receipt(item, _healthy_indexing())
        receipt = json.loads(adapter.store_build_receipt_path.read_text())
        receipt["embedding_state"]["sha256"] = "sha256:" + "0" * 64
        adapter.store_build_receipt_path.write_text(json.dumps(receipt))

        with pytest.raises(strictness.IndexingConvergenceError) as failed:
            adapter.validate_store_build_receipt(item)
        assert failed.value.summary["failure_reason"] == (
            "corrupt_store_build_receipt"
        )
        assert failed.value.summary[
            "recorded_embedding_state_sha256"
        ] == "sha256:" + "0" * 64
    finally:
        adapter.close()


@pytest.mark.parametrize("legacy_version", ("v1", "v2", "v3", "v4", "v5"))
def test_legacy_receipt_protocol_is_rejected_with_fresh_remediation(
    tmp_path, legacy_version,
):
    adapter = _adapter(tmp_path)
    try:
        adapter.publish_store_build_receipt(_item(), _healthy_indexing())
        receipt = json.loads(adapter.store_build_receipt_path.read_text())
        receipt["version"] = f"hymem-benchmark-store-build-{legacy_version}"
        adapter.store_build_receipt_path.write_text(json.dumps(receipt))

        with pytest.raises(strictness.IndexingConvergenceError) as legacy:
            adapter.validate_store_build_receipt(_item())
        assert legacy.value.summary["failure_reason"] == (
            "incompatible_store_build_receipt_version"
        )
        assert legacy.value.summary["recorded_receipt_version"].endswith(
            legacy_version
        )
        assert "--fresh" in legacy.value.summary["remediation"]
    finally:
        adapter.close()


def test_post_convergence_attestation_closes_provenance_to_scoring_seam(
    monkeypatch, tmp_path,
):
    adapter = _adapter(tmp_path)
    item = _item()
    args = SimpleNamespace(
        sim=False,
        no_dream=False,
        dream_per_session=False,
        indexing_max_cycles=2,
        indexing_timeout_s=10,
    )
    try:
        adapter.dream(max_cycles=2, timeout_s=10)
        adapter.publish_store_build_receipt(
            item, adapter.indexing_provenance(scope_id="locomo:conv")
        )
        adapter.indexing_runs.clear()
        original = adapter.indexing_provenance

        def contaminate_after_provenance(*, scope_id):
            indexing = original(scope_id=scope_id)
            adapter.hy.log_messages(
                "toctou", [("user", "late source", "2025-01-01")]
            )
            return indexing

        monkeypatch.setattr(adapter, "indexing_provenance", contaminate_after_provenance)
        with pytest.raises(strictness.IndexingConvergenceError) as failed:
            prepare_indexing(
                adapter, item, args, scope_id="locomo:conv", reuse=True
            )
        assert failed.value.summary["failure_reason"] == "store_material_state_mismatch"
        assert failed.value.summary["status"] == "failed_before_scoring"
    finally:
        adapter.close()


def test_reuse_rejects_newly_generated_malformed_provenance_before_scoring(
    monkeypatch, tmp_path,
):
    adapter = _adapter(tmp_path)
    item = {"id": "conv", "sessions": [], "session_dates": []}
    args = SimpleNamespace(
        sim=False,
        no_dream=False,
        dream_per_session=False,
        indexing_max_cycles=2,
        indexing_timeout_s=10,
    )
    try:
        adapter.dream(max_cycles=2, timeout_s=10)
        adapter.publish_store_build_receipt(
            item, adapter.indexing_provenance(scope_id="locomo:conv")
        )
        adapter.indexing_runs.clear()
        original_provenance = adapter.indexing_provenance
        original_validate = adapter.validate_store_build_receipt
        validations = 0

        def count_validation(source_item):
            nonlocal validations
            validations += 1
            return original_validate(source_item)

        def malformed_provenance(*, scope_id):
            provenance = original_provenance(scope_id=scope_id)
            provenance["cycles"] += 1
            return provenance

        monkeypatch.setattr(
            adapter, "validate_store_build_receipt", count_validation
        )
        monkeypatch.setattr(
            adapter, "indexing_provenance", malformed_provenance
        )

        with pytest.raises(BenchmarkIntegrityError, match="cycle arithmetic"):
            prepare_indexing(
                adapter, item, args, scope_id="locomo:conv", reuse=True
            )

        assert validations == 1
    finally:
        adapter.close()
