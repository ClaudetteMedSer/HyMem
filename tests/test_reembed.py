"""Offline operational recovery: bounded writes, exact producers, no LLM."""
from __future__ import annotations

import json
import sqlite3
import sys

import pytest

from hymem import reembed
from hymem.core import db
from hymem.core.vectors import encode_vector
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.embeddings import (
    fetch_chunk_embeddings, persist_chunk_embeddings,
    fetch_message_embeddings, persist_message_embeddings,
    fetch_episode_embeddings, persist_episode_embeddings,
    fetch_fact_embeddings, persist_fact_embeddings,
)
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.embeddings import LocalHashEmbeddingClient, embedding_text_hash


class ExactEmbedder:
    model = "repair-test"
    dim = 3
    network_free = True

    def __init__(self):
        self.calls = []

    def embedding_producer_declaration(self):
        return {
            "schema": "hymem-custom-embedding-producer-declaration-v2",
            "implementation": "tests.reembed", "implementation_revision": "v1",
            "deployment_revision": self.model, "deployment_tenant": "tests",
            "model_revision": self.model, "request_policy": "test-v1",
            "dimension": self.dim, "network_free": True,
        }

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture(autouse=True)
def observe_provider_calls():
    # An exact custom embed callable cannot dispatch through mutable helpers.
    # Trace passively instead of changing its producer-bound implementation.
    previous = sys.getprofile()
    def observe(frame, event, arg):
        if frame.f_code is ExactEmbedder.embed.__code__ and event == "call":
            frame.f_locals["self"].calls.append(list(frame.f_locals["texts"]))
        if previous is not None:
            previous(frame, event, arg)
    sys.setprofile(observe)
    try:
        yield
    finally:
        sys.setprofile(previous)


@pytest.fixture
def store(tmp_path):
    conn = db.connect(tmp_path / "hymem.sqlite")
    db.initialize(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('source')")
    try:
        yield conn
    finally:
        conn.close()


def _seed_chunks(conn, count=3, *, messages=False):
    old = LocalHashEmbeddingClient(dim_value=3, model_name="old")
    for index in range(count):
        mid = int(conn.execute("INSERT INTO messages(session_id,role,content) VALUES ('source','user',?)", (f"source number {index}",)).lastrowid)
        with db.transaction(conn):
            materialize_message_coverage(conn, "source")
            persist_chunks(conn, [Chunk(f"chunk-{index}", "source", mid, mid, "test", f"user: source number {index}", (mid,))])
    with db.transaction(conn):
        persist_chunk_embeddings(conn, fetch_chunk_embeddings(conn, old))
        if messages:
            persist_message_embeddings(conn, fetch_message_embeddings(conn, old))
    return old


def _mirrors(conn):
    return {name: [tuple(row) for row in conn.execute(f"SELECT * FROM {name} ORDER BY rowid")]
            for name in reembed.MIRROR_TABLES}


def _sources(conn):
    return {name: [tuple(row) for row in conn.execute(f"SELECT * FROM {name} ORDER BY rowid")]
            for name in ("sessions", "messages", "chunks", "message_retention_coverage", "chunk_message_sources",
                         "knowledge_graph", "kg_evidence", "episodes", "narrative_facts")}


def _corrupt_chunk_text(conn, text):
    # Model damaged historical bytes while restoring the genuine guard before
    # repair observes the database. Production writers cannot make this edit.
    trigger = conn.execute("SELECT sql FROM sqlite_master WHERE name='chunk_source_manifest_chunk_update_guard'").fetchone()[0]
    conn.execute("DROP TRIGGER chunk_source_manifest_chunk_update_guard")
    try:
        conn.execute("UPDATE chunks SET text=? WHERE id='chunk-0'", (text,))
    finally:
        conn.execute(trigger)


def test_dry_run_no_calls_writes_and_apply_resumes_bounded_batches(store):
    _seed_chunks(store, 5, messages=True)
    client = ExactEmbedder()
    sources, before = _sources(store), _mirrors(store)
    report = reembed.repair(store, client)
    assert report.status == "incomplete" and report.pending == 10 and report.sweep_complete
    assert client.calls == [] and _mirrors(store) == before and _sources(store) == sources
    assert not store.execute("SELECT 1 FROM schema_meta WHERE key LIKE 'embedding_repair_scan_v1:%'").fetchall()
    reports = [reembed.repair(store, client, apply=True, batch_size=2, max_items=3) for _ in range(4)]
    assert sum(report.repaired for report in reports) == 10
    assert reports[-1].status == "complete" and reports[-1].sweep_complete
    assert all(len(batch) <= 2 for batch in client.calls)
    assert _sources(store) == sources
    assert store.execute("SELECT COUNT(*) FROM run_lock").fetchone()[0] == 0
    before, calls = _mirrors(store), len(client.calls)
    assert reembed.repair(store, client, apply=True).status == "complete"
    assert _mirrors(store) == before and len(client.calls) == calls


def test_unproven_prefix_progress_and_no_missing_mirror_regeneration(store):
    store.execute("INSERT INTO chunks(id,session_id,start_message_id,end_message_id,salience_reason,text) VALUES ('aaa','source',1,1,'legacy','unproven')")
    with db.embedding_mutation(store):
        store.execute("INSERT INTO chunk_embeddings(chunk_id,vector_json,model,dim,text_hash) VALUES ('aaa','[1,0,0]',?,3,'old')", (embedding_storage_identity(ExactEmbedder())[0],))
    _seed_chunks(store, 3)
    store.execute("DELETE FROM chunk_embeddings WHERE chunk_id='chunk-1'")
    client = ExactEmbedder()
    reports = [reembed.repair(store, client, apply=True, max_items=1) for _ in range(4)]
    assert reports[0].blocked == 1 and reports[0].repaired == 0
    assert sum(report.repaired for report in reports) == 2
    assert reports[-1].sweep_complete and reports[-1].status == "blocked" and reports[-1].sweep_blocked == 1
    assert not store.execute("SELECT 1 FROM chunk_embeddings WHERE chunk_id='chunk-1'").fetchall()
    assert store.execute("SELECT model FROM chunk_embeddings WHERE chunk_id='aaa'").fetchone()[0] != embedding_storage_identity(client)[0]


@pytest.mark.parametrize("fault", ["source", "producer", "dimension", "lease", "provider", "deadline", "commit"])
def test_late_faults_leave_mirrors_and_cursor_unpublished(store, monkeypatch, fault):
    _seed_chunks(store, 1)
    before = _mirrors(store)
    client = ExactEmbedder()
    if fault == "source":
        action = lambda: _corrupt_chunk_text(store, "changed while provider runs")
    elif fault == "producer":
        action = lambda: setattr(client, "model", "changed")
    elif fault == "dimension":
        action = lambda: setattr(client, "dim", 4)
    elif fault == "lease":
        action = lambda: store.execute("UPDATE run_lock SET holder='new-owner' WHERE name='dreaming'")
    elif fault == "provider":
        def fail():
            raise RuntimeError("secret endpoint credential")
        action = fail
    elif fault == "deadline":
        from hymem.deadline import current_deadline
        action = lambda: object.__setattr__(current_deadline(), "expires_at", 0.0)
    else:
        store.execute("CREATE TEMP TRIGGER fail_cursor BEFORE INSERT ON schema_meta WHEN NEW.key LIKE 'embedding_repair_scan_v1:%' BEGIN SELECT RAISE(ABORT,'commit fault'); END")
        action = lambda: None
    previous = sys.getprofile()
    def observe(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        if frame.f_code is ExactEmbedder.embed.__code__ and event == "return":
            action()
    sys.setprofile(observe)
    try:
        report = reembed.repair(store, client, apply=True)
    finally:
        sys.setprofile(previous)
    assert len(client.calls) == 1
    assert report.exit_code == 1 and report.repaired == 0
    assert _mirrors(store) == before
    assert not store.execute("SELECT 1 FROM schema_meta WHERE key LIKE 'embedding_repair_scan_v1:%'").fetchall()
    assert not store.in_transaction
    assert "secret" not in json.dumps(reembed.asdict(report))
    if fault == "lease":
        assert store.execute("SELECT holder FROM run_lock").fetchone()[0] == "new-owner"
    if fault == "source":
        assert store.execute("SELECT text FROM chunks WHERE id='chunk-0'").fetchone()[0] == "changed while provider runs"


def test_busy_lease_fk_and_caller_transaction_prevent_provider_calls(store):
    _seed_chunks(store, 1)
    client = ExactEmbedder()
    store.execute("INSERT INTO run_lock(name,holder,acquired_at) VALUES ('dreaming','other',CURRENT_TIMESTAMP)")
    assert reembed.repair(store, client, apply=True).error_code == "lease_busy"
    store.execute("DELETE FROM run_lock")
    store.execute("BEGIN IMMEDIATE")
    assert reembed.repair(store, client, apply=True).error_code == "caller_transaction"
    assert store.in_transaction
    store.rollback()
    store.execute("PRAGMA foreign_keys=OFF")
    with db.embedding_mutation(store):
        store.execute("INSERT INTO chunk_embeddings(chunk_id,model,dim,vector_json) VALUES ('orphan',?,3,'[1,0,0]')", (embedding_storage_identity(client)[0],))
    before = _mirrors(store)
    assert reembed.repair(store, client, apply=True).error_code == "store_integrity"
    assert client.calls == [] and _mirrors(store) == before


def test_limited_edge_updates_preserve_unrelated_shadows(store):
    client = ExactEmbedder()
    model, dim = embedding_storage_identity(client)
    for index in range(3):
        store.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,pos_evidence) VALUES (?,'uses','redis',1)", (f"user_{index}",))
    with db.transaction(store), db.embedding_mutation(store):
        db.ensure_vec_table(store, dim, model=model)
        for index in range(3):
            store.execute("INSERT INTO edge_embeddings(edge_text,model,dim,vector_json) VALUES (?,?,3,?)", (f"user_{index} uses redis", model, "bad" if index == 0 else "[1,0,0]"))
            if db.has_vec_table(store, table="vec_edges"):
                store.execute("INSERT INTO vec_edges(rowid,embedding) VALUES (?,?)", (index + 1, db._pack_vector([1, 0, 0])))
    report = reembed.repair(store, client, apply=True, max_items=1)
    assert report.repaired == 1 and not report.sweep_complete
    if db.has_vec_table(store, table="vec_edges"):
        assert {row[0] for row in store.execute("SELECT rowid FROM vec_edges")} == {1, 2, 3}
    assert client.calls == [["user_0 uses redis"]]


def test_cli_readonly_no_create_no_llm_and_explicit_configuration(store, tmp_path, monkeypatch, capsys):
    for name in list(reembed.os.environ):
        if name.startswith("HYMEM_") or name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
            monkeypatch.delenv(name)
    from hymem import bootstrap
    monkeypatch.setattr(bootstrap, "build_from_env", lambda: pytest.fail("LLM bootstrap forbidden"))
    path = tmp_path / "hymem.sqlite"
    before = list(store.iterdump())
    assert reembed.main(["--db", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert list(store.iterdump()) == before
    assert reembed.main(["--db", str(path), "--apply", "--json"]) == 1
    capsys.readouterr()
    assert reembed.main(["--db", str(path), "--apply", "--allow-local", "--json"]) == 0
    capsys.readouterr()
    missing = tmp_path / "missing" / "new.sqlite"
    assert reembed.main(["--db", str(missing), "--json"]) == 1
    capsys.readouterr()
    assert not missing.parent.exists()
    monkeypatch.setenv("HYMEM_EMBEDDING_DIM", "abc")
    assert reembed.main(["--db", str(path), "--apply", "--allow-local", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "invalid_embedding_dimension"
    assert list(store.iterdump()) != []


def test_current_schema_required_no_implicit_upgrade(store):
    store.execute("UPDATE schema_meta SET value='58' WHERE key='schema_version'")
    before = list(store.iterdump())
    report = reembed.repair(store, ExactEmbedder(), apply=True)
    assert report.error_code == "current_schema_required" and list(store.iterdump()) == before


def test_eof_rechecks_earlier_rows_and_producer_cursor_is_separate(store):
    _seed_chunks(store, 2)
    client = ExactEmbedder()
    first = reembed.repair(store, client, apply=True, max_items=1)
    assert first.repaired == 1 and not first.sweep_complete
    with db.embedding_mutation(store):
        store.execute("UPDATE chunk_embeddings SET vector_json='corrupt again' WHERE chunk_id='chunk-0'")
    end = reembed.repair(store, client, apply=True)
    assert end.sweep_complete and end.status == "incomplete" and end.health_after
    assert reembed.repair(store, client, apply=True).status == "complete"
    client.model = "another-exact-producer"
    again = reembed.repair(store, client, apply=True, max_items=1)
    assert again.repaired == 1 and len(store.execute("SELECT key FROM schema_meta WHERE key LIKE 'embedding_repair_scan_v1:%'").fetchall()) == 2
    assert all("source number" not in row[0] and "chunk-" not in row[0]
               for row in store.execute("SELECT value FROM schema_meta WHERE key LIKE 'embedding_repair_scan_v1:%'"))


def _seed_episode_fact(conn, tmp_path):
    from hymem.config import HyMemConfig
    from hymem.dreaming.episodes import EpisodesExtraction, persist_episodes
    from hymem.dreaming.message_coverage import coverage_chunk_id
    from hymem.dreaming import facts
    from hymem.extraction.llm import StubLLMClient
    mid = conn.execute("SELECT id FROM messages ORDER BY id LIMIT 1").fetchone()[0]
    with db.transaction(conn):
        persist_episodes(conn, "source", EpisodesExtraction(items=[{
            "title": "Source episode", "summary": "source number zero",
            "outcome": "informational", "key_entities": ["source"],
            "chunk_ids": [coverage_chunk_id("source", mid)],
        }]))
    cfg = HyMemConfig(root=tmp_path)
    extraction = facts.extract_facts(conn, "source", StubLLMClient(default='[{"text":"A fact about source zero"}]'), cfg)
    assert extraction is not None
    # Direct low-level historical fact callers retain config-only authority.
    extraction.publication_version = facts.facts_config_version(cfg)
    with db.transaction(conn):
        assert facts.persist_facts(conn, "source", extraction) == 1
    old = LocalHashEmbeddingClient(dim_value=3, model_name="old")
    with db.transaction(conn):
        assert persist_episode_embeddings(conn, fetch_episode_embeddings(conn, old)) == 1
        assert persist_fact_embeddings(conn, fetch_fact_embeddings(conn, old)) == 1


def test_episode_fact_and_message_repair_keep_source_and_publication_proofs(store, tmp_path):
    _seed_chunks(store, 1, messages=True)
    _seed_episode_fact(store, tmp_path)
    before = _sources(store)
    client = ExactEmbedder()
    report = reembed.repair(store, client, apply=True)
    assert report.status == "complete" and report.repaired == 4
    assert _sources(store) == before
    assert reembed.repair(store, client).status == "complete"


@pytest.mark.parametrize("tier", ["message", "episode", "fact"])
def test_source_authority_is_rechecked_after_provider_for_every_proof_tier(store, tmp_path, tier):
    _seed_chunks(store, 1, messages=True)
    _seed_episode_fact(store, tmp_path)
    # Remove other existing mirrors from this fixture, not source data. The
    # repair command must never regenerate those missing tiers.
    keep = {"message": "message_embeddings", "episode": "episode_embeddings", "fact": "narrative_fact_embeddings"}[tier]
    for table in reembed.MIRROR_TABLES:
        if table != keep:
            store.execute(f"DELETE FROM {table}")
    before = _mirrors(store)
    client = ExactEmbedder()
    mutations = []
    previous = sys.getprofile()
    def observe(frame, event, arg):
        if previous:
            previous(frame, event, arg)
        if frame.f_code is ExactEmbedder.embed.__code__ and event == "return":
            if tier == "message":
                store.execute("UPDATE sessions SET coverage_message_id=NULL WHERE id='source'")
            elif tier == "episode":
                from hymem.dreaming.aggregation_provenance import unpublish_episode_source_manifest
                with db.transaction(store):
                    unpublish_episode_source_manifest(store, store.execute("SELECT id FROM episodes").fetchone()[0])
                    store.execute("UPDATE episodes SET summary='changed summary'")
            else:
                with db.evidence_mutation(store):
                    store.execute("UPDATE narrative_facts SET lifecycle_status='retracted'")
            mutations.append(tier)
    sys.setprofile(observe)
    try:
        report = reembed.repair(store, client, apply=True)
    finally:
        sys.setprofile(previous)
    assert len(client.calls) == 1 and report.repaired == 0 and report.exit_code == 1
    assert mutations == [tier]
    assert _mirrors(store) == before


def test_aggregation_rows_require_rebuild_never_relabel_or_embed(store):
    client = ExactEmbedder()
    old = LocalHashEmbeddingClient(dim_value=3, model_name="old")
    store.execute("INSERT INTO aggregation_nodes(id,title,summary,member_episode_ids) VALUES ('historical','Old','Old summary','[]')")
    with db.embedding_mutation(store):
        old_key = embedding_storage_identity(old)[0]
        store.execute("INSERT INTO aggregation_node_embeddings(node_id,vector_json,model,dim,text_hash,embedding_producer_key) VALUES ('historical','[1,0,0]',?,3,'old',?)", (old_key, old_key))
    before = _mirrors(store)
    report = reembed.repair(store, client, apply=True)
    assert report.status == "blocked" and report.rebuild_required == 1 and report.repaired == 0
    assert client.calls == [] and _mirrors(store) == before


def test_cli_apply_existing_rw_connection_does_not_need_llm_bootstrap(store, tmp_path, monkeypatch, capsys):
    _seed_chunks(store, 1, messages=True)
    for name in list(reembed.os.environ):
        if name.startswith("HYMEM_") or name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
            monkeypatch.delenv(name)
    from hymem import bootstrap
    monkeypatch.setattr(bootstrap, "build_from_env", lambda: pytest.fail("LLM bootstrap forbidden"))
    before = _sources(store)
    assert reembed.main(["--db", str(tmp_path / "hymem.sqlite"), "--apply", "--allow-local", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["repaired"] == 2 and report["provider_batches"] == 2
    assert _sources(store) == before


@pytest.mark.parametrize("raw", ["abc", "0", "-1", "nan", "", "1.5", " 3", "３"])
def test_malformed_explicit_dimension_rejected_before_client_and_store(monkeypatch, tmp_path, raw):
    monkeypatch.setenv("HYMEM_EMBEDDING_DIM", raw)
    from hymem import bootstrap
    monkeypatch.setattr(bootstrap, "resolve_env", lambda: pytest.fail("must validate explicit dimension first"))
    with pytest.raises(ValueError):
        reembed._configured_embedder(apply=True, allow_local=True)


def test_explicit_remote_fallback_never_allowed_as_local(monkeypatch):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "https://embeddings.invalid/v1")
    monkeypatch.delenv("HYMEM_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="embedding_credentials_missing"):
        reembed._configured_embedder(apply=True, allow_local=True)


def test_preexisting_chunk_text_cannot_borrow_valid_manifest_authority(store):
    _seed_chunks(store, 1)
    _corrupt_chunk_text(store, "private forged unrelated text")
    before = _mirrors(store)
    client = ExactEmbedder()
    report = reembed.repair(store, client, apply=True)
    assert report.status == "blocked" and report.blocked == 1
    assert client.calls == [] and _mirrors(store) == before


def test_producer_fence_inside_final_transaction_rolls_back_vectors_and_cursor(store):
    _seed_chunks(store, 1)
    client = ExactEmbedder()
    before = _mirrors(store)
    previous = sys.getprofile()
    def observe(frame, event, arg):
        if previous:
            previous(frame, event, arg)
        if frame.f_code is reembed._publish.__code__ and event == "return":
            assert store.in_transaction
            client.model = "late-producer-switch"
    sys.setprofile(observe)
    try:
        report = reembed.repair(store, client, apply=True)
    finally:
        sys.setprofile(previous)
    assert len(client.calls) == 1 and report.repaired == 0 and report.status == "error"
    assert _mirrors(store) == before
    assert not store.execute("SELECT 1 FROM schema_meta WHERE key LIKE 'embedding_repair_scan_v1:%'").fetchall()


def test_zero_and_wrong_count_provider_vectors_cannot_be_published(store):
    class Zero(ExactEmbedder):
        def embed(self, texts):
            return [[0.0, 0.0, 0.0] for _ in texts]
    class WrongCount(ExactEmbedder):
        def embed(self, texts):
            return []
    _seed_chunks(store, 1)
    before = _mirrors(store)
    for client in (Zero(), WrongCount()):
        report = reembed.repair(store, client, apply=True)
        assert report.status == "error" and report.provider_batches == 1 and report.repaired == 0
        assert _mirrors(store) == before


def test_episode_shadow_collision_blocks_without_overwrite(store, tmp_path, monkeypatch):
    _seed_chunks(store, 1)
    _seed_episode_fact(store, tmp_path)
    for table in reembed.MIRROR_TABLES:
        if table != "episode_embeddings":
            store.execute(f"DELETE FROM {table}")
    episode_id = store.execute("SELECT id FROM episodes").fetchone()[0]
    store.execute("INSERT INTO episodes(id,session_id,title,summary,outcome,key_entities) VALUES ('collider','source','Legacy','Unproven','informational','[]')")
    target = db.episode_vector_rowid(episode_id)
    monkeypatch.setattr(db, "episode_vector_rowid", lambda key: target)
    before = _mirrors(store)
    client = ExactEmbedder()
    report = reembed.repair(store, client, apply=True)
    assert report.status == "blocked" and report.blocked == 1 and client.calls == []
    assert _mirrors(store) == before


def test_same_dimension_producer_switch_preserves_already_compatible_shadow(store):
    _seed_chunks(store, 3)
    client = ExactEmbedder()
    model, dim = embedding_storage_identity(client)
    with db.embedding_mutation(store):
        store.execute("UPDATE chunk_embeddings SET model=?,vector_json='[1,0,0]' WHERE chunk_id='chunk-2'", (model,))
    compatible = tuple(store.execute("SELECT * FROM chunk_embeddings WHERE chunk_id='chunk-2'").fetchone())
    report = reembed.repair(store, client, apply=True, max_items=1)
    assert report.repaired == 1 and report.pending == 0
    assert tuple(store.execute("SELECT * FROM chunk_embeddings WHERE chunk_id='chunk-2'").fetchone()) == compatible
    if db.has_vec_table(store, table="vec_chunks"):
        rowids = {row[0] for row in store.execute("SELECT rowid FROM vec_chunks")}
        assert {row[0] for row in store.execute("SELECT rowid FROM chunks WHERE id IN ('chunk-0','chunk-2')")} <= rowids


def test_healthy_current_aggregation_is_a_noop_but_new_producer_requires_rebuild(store, tmp_path):
    from hymem.config import HyMemConfig
    from hymem.dreaming.aggregate import build_aggregation_nodes
    from tests.test_aggregation_provenance import _seed_native_episode, _aggregation_cfg, _fusion_llm
    _seed_native_episode(store, "agg-a", title="Shared", summary="First shared source", entity="thread")
    _seed_native_episode(store, "agg-b", title="Shared", summary="Second shared source", entity="thread")
    cfg = _aggregation_cfg(HyMemConfig(root=tmp_path))
    client = LocalHashEmbeddingClient(dim_value=3, model_name="aggregation-native")
    llm = _fusion_llm()
    built = build_aggregation_nodes(store, cfg, llm, client)
    assert built.nodes == 1 and store.execute("SELECT COUNT(*) FROM aggregation_node_embeddings").fetchone()[0] == 1
    before, calls = _mirrors(store), len(llm.calls)
    report = reembed.repair(store, client, apply=True)
    assert report.status == "complete" and report.repaired == 0 and report.rebuild_required == 0
    assert _mirrors(store) == before and len(llm.calls) == calls
    changed = reembed.repair(store, ExactEmbedder(), apply=True)
    assert changed.status == "blocked" and changed.rebuild_required == 1
    assert _mirrors(store)["aggregation_node_embeddings"] == before["aggregation_node_embeddings"]
    assert len(llm.calls) == calls
