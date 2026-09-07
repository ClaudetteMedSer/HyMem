"""Schema-v55 adversarial contracts for typed aggregation provenance."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import pytest

from benchmarks.store_attestation import material_store_state
from hymem import (
    HyMem,
    HyMemConfig,
    NodeSourceOccurrence,
    StubEmbeddingClient,
)
from hymem.core import db as core_db
from hymem.core.vectors import decode_vector, encode_vector
from hymem.dreaming import aggregate as aggregate_mod
from hymem.dreaming import phase1
from hymem.dreaming.aggregate import (
    _candidate_node_row,
    _node_frontier_item,
    _summarize_cluster,
    aggregation_config_version,
    build_aggregation_nodes,
    expand_node,
    fetch_node_embeddings,
    load_clusterable_episodes,
    load_digest,
)
from hymem.dreaming.aggregation_provenance import (
    aggregation_node_embedding_set_hash,
    aggregation_llm_request,
    aggregation_llm_request_hash,
    aggregation_node_id,
    aggregation_output_hash,
    aggregation_publication_id,
    aggregation_publication_node_set_hash,
    episode_input_proof,
    load_aggregation_node_proof,
    load_current_aggregation_publication,
    load_profile_anchor_inputs,
    persist_aggregation_source_manifest,
)
from hymem.dreaming.aggregation_material import (
    aggregation_material_binding,
    disabled_aggregation_phase1_scope_identity,
    embedding_producer_binding,
    embedding_storage_identity,
    register_aggregation_material_epoch,
)
from hymem.dreaming.aggregation_generation import (
    aggregation_generation_binding_for_contract,
    aggregation_generation_contract,
    register_aggregation_generation,
)
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.extraction.embeddings import (
    MappedStubEmbeddingClient,
    embedding_text_hash,
)
from hymem.extraction.llm import LLMRequest, StubLLMClient
from hymem.extraction.triples import Triple
from hymem.query.augment import _BoundQueryVector, _aggregation_search, augment
from tests.test_aggregation_provenance import (
    _aggregation_cfg,
    _fusion_llm,
    _rewrite_portable_wire,
    _seed_native_episode,
)


def _seed_tree(hy: HyMem, cfg: HyMemConfig, *, embedding_client=None):
    for cluster in range(4):
        for member in range(2):
            _seed_native_episode(
                hy.conn,
                f"v55-tree-{cluster}-{member}",
                title=f"Needle {cluster}/{member}",
                summary=f"needle cluster {cluster} source {member}",
                entity=f"v55-cluster-{cluster}",
            )
    enabled = replace(
        _aggregation_cfg(cfg, digest=True),
        aggregation_max_members=2,
        aggregation_max_cluster_size=2,
        aggregation_digest_anchor_facts=0,
        augment_include_digest=True,
        aggregation_inject_abilities=(),
    )
    result = build_aggregation_nodes(
        hy.conn, enabled, _fusion_llm(), embedding_client
    )
    assert result.fusion_failures == 0
    publication = load_current_aggregation_publication(hy.conn)
    assert publication is not None and publication.root_node_id is not None
    assert max(int(proof.row["level"]) for proof in publication.nodes.values()) >= 2
    return enabled, publication


def _seed_exact_profile_anchor(hy: HyMem) -> int:
    _episode_id, _chunk_id, message_id = _seed_native_episode(
        hy.conn,
        "v55-profile-source",
        title="Profile source",
        summary="The user lives in Amsterdam.",
        entity="profile-only",
    )
    source = hy.conn.execute(
        "SELECT created_at FROM messages WHERE id=?", (message_id,)
    ).fetchone()[0]
    with core_db.transaction(hy.conn):
        cursor = hy.conn.execute(
            "INSERT INTO user_profile(slot,value,evidence_message_id,"
            "source_message_id,source_session_id,source_created_at,confidence) "
            "VALUES ('location','Amsterdam',?,?,?, ?,0.9)",
            (message_id, message_id, "v55-profile-source", source),
        )
    return int(cursor.lastrowid)


def _seed_exact_kg_anchor(hy: HyMem, cfg: HyMemConfig) -> int:
    session_id = "v55-kg-source"
    with core_db.transaction(hy.conn):
        hy.conn.execute("INSERT INTO sessions(id) VALUES (?)", (session_id,))
        message_id = int(hy.conn.execute(
            "INSERT INTO messages(session_id,role,content) "
            "VALUES (?,'user','Atta is part of Medflow.')",
            (session_id,),
        ).lastrowid)
        materialize_message_coverage(hy.conn, session_id)
    chunk = Chunk(
        id="v55-kg-extraction",
        session_id=session_id,
        start_message_id=message_id,
        end_message_id=message_id,
        salience_reason="test exact anchor",
        text="user: Atta is part of Medflow.",
        source_message_ids=(message_id,),
    )
    with core_db.transaction(hy.conn):
        persist_chunks(hy.conn, [chunk])
    sources = phase1._claim_sources_for_chunk(hy.conn, chunk)
    extraction = ChunkExtraction(
        triples=[Triple(
            "atta", "part_of", "medflow", 1,
            source_message_id=message_id,
        )],
        markers=[],
        claim_sources={source.message_id: source for source in sources},
        source_validated=True,
    )
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn, chunk, extraction, prompt_version=cfg.prompt_version, cfg=cfg,
        )
    return int(hy.conn.execute(
        "SELECT id FROM knowledge_graph WHERE subject_canonical='atta' "
        "AND predicate='part_of' AND object_canonical='medflow'"
    ).fetchone()[0])


def _clone_episode_with_id(
    conn: sqlite3.Connection, old_id: str, new_id: str,
) -> None:
    """Clone one exact episode/proof under an adversarial cross-kind id."""

    row = conn.execute("SELECT * FROM episodes WHERE id=?", (old_id,)).fetchone()
    assert row is not None
    sources = conn.execute(
        "SELECT * FROM episode_source_occurrences WHERE episode_id=? "
        "ORDER BY ordinal",
        (old_id,),
    ).fetchall()
    with core_db.transaction(conn):
        conn.execute(
            "INSERT INTO episodes(id,session_id,title,summary,participants,"
            "start_message_id,end_message_id,outcome,key_entities,digest_slice_key,"
            "digest_generation,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                new_id, row["session_id"], row["title"], row["summary"],
                row["participants"], row["start_message_id"],
                row["end_message_id"], row["outcome"], row["key_entities"],
                row["digest_slice_key"], row["digest_generation"],
                row["created_at"],
            ),
        )
        for source in sources:
            conn.execute(
                "INSERT INTO episode_source_occurrences(episode_id,ordinal,"
                "source_message_id,source_session_id,source_role,source_peer_id,"
                "source_workspace_id,source_created_at,source_coverage_chunk_id,"
                "source_coverage_version,source_content_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    new_id, source["ordinal"], source["source_message_id"],
                    source["source_session_id"], source["source_role"],
                    source["source_peer_id"], source["source_workspace_id"],
                    source["source_created_at"],
                    source["source_coverage_chunk_id"],
                    source["source_coverage_version"],
                    source["source_content_hash"],
                ),
            )
        conn.execute(
            "UPDATE episodes SET source_manifest_version=?,"
            "source_manifest_count=?,source_manifest_hash=?,"
            "source_manifest_complete=? WHERE id=?",
            (
                row["source_manifest_version"], row["source_manifest_count"],
                row["source_manifest_hash"], row["source_manifest_complete"],
                new_id,
            ),
        )
        conn.execute("DELETE FROM episodes WHERE id=?", (old_id,))


def _persist_candidate_publication(
    conn: sqlite3.Connection,
    rows: list[dict],
    *,
    config_version: str,
    root_id: str | None,
    cluster_min_members: int = 2,
    cluster_min_sessions: int = 2,
    anchor_fact_cap: int = 20,
) -> None:
    """Persist a hash-consistent candidate, including adversarial shapes."""

    assert rows
    generation_binding = rows[0]["_generation_binding"]
    material_binding = rows[0]["_material_binding"]
    assert all(row["_generation_binding"] == generation_binding for row in rows)
    assert all(row["_material_binding"] == material_binding for row in rows)
    generation_key = generation_binding["generation_key"]
    material_epoch_key = material_binding["material_epoch_key"]
    material_revision = material_binding["material_revision"]
    request_contract_sha256 = generation_binding["contract"][
        "request_policy_sha256"
    ]
    node_set_hash = aggregation_publication_node_set_hash(rows)
    node_embedding_set_hash = aggregation_node_embedding_set_hash(())
    published_at = conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
    publication_id = aggregation_publication_id(
        config_version=config_version,
        root_id=root_id,
        node_count=len(rows),
        node_set_hash=node_set_hash,
        published_at=published_at,
        cluster_min_members=cluster_min_members,
        cluster_min_sessions=cluster_min_sessions,
        anchor_fact_cap=anchor_fact_cap,
        generation_key=generation_key,
        material_epoch_key=material_epoch_key,
        material_revision=material_revision,
        node_embedding_count=0,
        node_embedding_set_hash=node_embedding_set_hash,
        request_contract_sha256=request_contract_sha256,
    )
    for row in rows:
        row["publication_id"] = publication_id
    with core_db.transaction(conn):
        generation = register_aggregation_generation(conn, generation_binding)
        assert generation["generation_key"] == generation_key
        material = register_aggregation_material_epoch(conn, material_binding)
        assert material["material_epoch_key"] == material_epoch_key
        conn.execute("DELETE FROM aggregation_publication_state")
        conn.execute("DELETE FROM aggregation_nodes")
        for row in rows:
            conn.execute(
                "INSERT INTO aggregation_nodes("
                "id,title,summary,member_episode_ids,session_ids,n_members,"
                "n_sessions,level,is_root,input_fingerprint,node_kind,"
                "output_hash,publication_id,build_config_version,"
                "aggregation_generation_key,aggregation_request_hash,"
                "aggregation_material_epoch_key) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["id"], row["title"], row["summary"],
                    row["member_episode_ids"], row["session_ids"],
                    row["n_members"], row["n_sessions"], row["level"],
                    row["is_root"], row["input_fingerprint"],
                    row["node_kind"], row["output_hash"], publication_id,
                    config_version, generation_key,
                    row["aggregation_request_hash"],
                    material_epoch_key,
                ),
            )
            persist_aggregation_source_manifest(
                conn,
                row["id"],
                inputs=row["input_proofs"],
                node_kind=row["node_kind"],
                publication_id=publication_id,
                build_config_version=config_version,
                aggregation_generation_key=generation_key,
                aggregation_material_epoch_key=material_epoch_key,
            )
        conn.execute(
            "INSERT INTO aggregation_publication_state("
            "id,publication_id,config_version,cluster_min_members,"
            "cluster_min_sessions,anchor_fact_cap,root_node_id,node_count,"
            "node_set_hash,published_at,aggregation_generation_key,"
            "request_contract_sha256,aggregation_material_epoch_key,"
            "material_revision,node_embedding_count,node_embedding_set_hash) "
            "VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                publication_id, config_version, cluster_min_members,
                cluster_min_sessions, anchor_fact_cap, root_id, len(rows),
                node_set_hash, published_at, generation_key,
                request_contract_sha256,
                material_epoch_key, material_revision, 0,
                node_embedding_set_hash,
            ),
        )


def _candidate_from_episode_proofs(
    proofs, *, kind: str, level: int, config_version: str, label: str,
) -> dict:
    contract = aggregation_generation_contract(
        _aggregation_cfg(HyMemConfig(root=Path("/fixture-generation")))
    )
    contract["material_config_version"] = config_version
    generation = aggregation_generation_binding_for_contract(
        contract, _fusion_llm()
    )
    disabled_scope, disabled_exact, disabled_reuse = (
        disabled_aggregation_phase1_scope_identity()
    )
    material = aggregation_material_binding(
        material_revision=0,
        config_version=config_version,
        episode_ceiling_rowid=None,
        episode_records=(),
        anchor_records=(),
        blocking={"mode": "adversarial-fixture"},
        embedding_binding=embedding_producer_binding(None),
        embedding_dimension=None,
        node_embedding_required=False,
        root_anchors_enabled=False,
        phase1_scope_sha256=disabled_scope,
        phase1_scope_identity_exact=disabled_exact,
        phase1_scope_reuse_scope=disabled_reuse,
    )
    request_hash = aggregation_llm_request_hash(
        aggregation_llm_request(tuple(proofs), node_kind=kind)
    )
    row = _candidate_node_row(
        inputs=tuple(proofs),
        fused={
            "title": label, "summary": f"{label} exact summary",
            "_aggregation_request_hash": request_hash,
        },
        node_kind=kind,
        level=level,
        config_version=config_version,
        generation_key=generation["generation_key"],
        request_hash=request_hash,
        material_epoch_key=material["material_epoch_key"],
        reused=False,
    )
    row["_generation_binding"] = generation
    row["_material_binding"] = material
    return row


def test_complete_multilevel_publication_survives_restart(cfg):
    hy = HyMem(cfg)
    root = None
    enabled = None
    try:
        enabled, publication = _seed_tree(hy, cfg)
        root = publication.root_node_id
        assert load_digest(hy.conn).node_id == root
        assert expand_node(hy.conn, root) is not None
    finally:
        hy.close()

    assert enabled is not None
    reopened = HyMem(enabled)
    try:
        publication = load_current_aggregation_publication(reopened.conn)
        assert publication is not None and publication.root_node_id == root
        assert load_digest(reopened.conn).node_id == root
        assert expand_node(reopened.conn, root) is not None
        leaf_id = next(
            node_id for node_id, proof in publication.nodes.items()
            if proof.row["node_kind"] == "cluster"
        )
        leaf = reopened.expand_node(leaf_id).episodes[0]
        assert leaf.source_message_ids
        assert isinstance(leaf.source_occurrences[0], NodeSourceOccurrence)
    finally:
        reopened.close()


def test_non_ascii_member_and_session_ids_build_and_survive_restart(cfg):
    hy = HyMem(cfg)
    enabled = _aggregation_cfg(cfg)
    try:
        for suffix in ("a", "b"):
            old_id, _chunk, _message = _seed_native_episode(
                hy.conn, f"session-café-{suffix}", title="Café",
                summary=f"exact café source {suffix}", entity="café-thread",
            )
            _clone_episode_with_id(hy.conn, old_id, f"café-{suffix}")
        result = build_aggregation_nodes(hy.conn, enabled, _fusion_llm())
        assert result.nodes == 1 and result.fusion_failures == 0
        publication = load_current_aggregation_publication(hy.conn)
        assert publication is not None
        row = next(iter(publication.nodes.values())).row
        assert json.loads(row["member_episode_ids"]) == ["café-a", "café-b"]
        assert "café" in row["member_episode_ids"]
        assert "session-café" in row["session_ids"]
    finally:
        hy.close()

    reopened = HyMem(enabled)
    try:
        publication = load_current_aggregation_publication(reopened.conn)
        assert publication is not None and len(publication.nodes) == 1
    finally:
        reopened.close()


def test_episode_and_node_id_collision_resolves_by_typed_kind(cfg):
    hy = HyMem(cfg)
    try:
        digest_cfg = replace(
            _aggregation_cfg(cfg, digest=True),
            aggregation_max_members=2,
            aggregation_max_cluster_size=2,
            aggregation_digest_anchor_facts=0,
        )
        llm = _fusion_llm()
        for sid in ("collision-a", "collision-b"):
            _seed_native_episode(
                hy.conn, sid, title="Shared", summary="collision shared",
                entity="collision-shared",
            )
        build_aggregation_nodes(hy.conn, digest_cfg, llm)
        child_id = hy.conn.execute(
            "SELECT id FROM aggregation_nodes WHERE node_kind='cluster'"
        ).fetchone()[0]

        old_episode_id, _chunk, _message = _seed_native_episode(
            hy.conn, "collision-leaf", title="Leaf",
            summary="collision distinct leaf", entity="collision-distinct",
        )
        _clone_episode_with_id(hy.conn, old_episode_id, str(child_id))
        for index in range(3):
            _seed_native_episode(
                hy.conn, f"collision-extra-{index}", title=f"Extra {index}",
                summary=f"distinct collision branch {index}",
                entity=f"collision-extra-{index}",
            )
        build_aggregation_nodes(hy.conn, digest_cfg, llm)
        publication = load_current_aggregation_publication(hy.conn)
        assert publication is not None and publication.root_node_id is not None
        assert max(
            int(proof.row["level"]) for proof in publication.nodes.values()
        ) >= 2
        colliding = [
            item
            for proof in publication.nodes.values()
            for item in proof.inputs
            if item.source_ref.get("id") == child_id
        ]
        assert {item.kind for item in colliding} == {
            "aggregation_node", "episode",
        }
        child_expansion = expand_node(hy.conn, child_id)
        assert child_expansion is not None and child_expansion.child_nodes == []
        episode_parent = next(
            proof.row["id"] for proof in publication.nodes.values()
            if any(
                item.kind == "episode" and item.source_ref.get("id") == child_id
                for item in proof.inputs
            )
        )
        expanded = expand_node(hy.conn, str(episode_parent))
        assert expanded is not None
        assert child_id in [item.id for item in expanded.episodes]
    finally:
        hy.close()


def test_digest_disabled_publication_serves_only_valid_level_zero(cfg):
    hy = HyMem(cfg)
    try:
        for sid in ("flat-a", "flat-b"):
            _seed_native_episode(
                hy.conn, sid, title="Flat needle", summary="flat needle source",
                entity="flat-publication",
            )
        build_aggregation_nodes(hy.conn, _aggregation_cfg(cfg), _fusion_llm())
        publication = load_current_aggregation_publication(hy.conn)
        assert publication is not None
        assert publication.root_node_id is None
        assert publication.nodes
        assert all(
            proof.row["node_kind"] == "cluster" and proof.row["level"] == 0
            for proof in publication.nodes.values()
        )
        assert load_digest(hy.conn) is None
        hits = _aggregation_search(hy.conn, "flat needle", top_k=3)
        assert {hit.node_id for hit in hits} == set(publication.nodes)
        assert expand_node(hy.conn, hits[0].node_id) is not None
    finally:
        hy.close()


def test_public_api_requires_active_material_config(cfg, tmp_path):
    active = replace(
        _aggregation_cfg(cfg, digest=True),
        root=tmp_path / "active-config",
        aggregation_inject_abilities=("TR",),
        augment_include_digest=True,
    )
    hy = HyMem(active)
    root_id = None
    try:
        for sid in ("active-config-a", "active-config-b"):
            _seed_native_episode(
                hy.conn, sid, title="Active needle", summary="active source",
                entity="active-config",
            )
        build_aggregation_nodes(hy.conn, hy.config, _fusion_llm())
        digest = hy.digest()
        assert digest is not None
        root_id = digest.node_id
        assert hy.expand_node(root_id) is not None
        assert hy.augment("needle", ability="TR").aggregation_nodes
    finally:
        hy.close()

    variants = (
        replace(active, aggregation_nodes_enabled=False),
        replace(active, aggregation_digest_enabled=False),
        replace(active, redact_secrets=not active.redact_secrets),
    )
    for changed in variants:
        reopened = HyMem(changed)
        try:
            assert reopened.digest() is None
            assert reopened.expand_node(str(root_id)) is None
            context = reopened.augment("needle", ability="TR")
            assert context.digest is None
            assert context.aggregation_nodes == []
        finally:
            reopened.close()


def test_fusion_cache_never_relabels_old_output_for_new_config(cfg):
    hy = HyMem(cfg)
    try:
        for sid in ("cache-config-a", "cache-config-b"):
            _seed_native_episode(
                hy.conn, sid, title="Cache", summary="unchanged exact input",
                entity="cache-config",
            )
        first_cfg = _aggregation_cfg(cfg)
        first = build_aggregation_nodes(hy.conn, first_cfg, _fusion_llm())
        assert first.nodes == 1 and first.reused == 0
        old_id = hy.conn.execute(
            "SELECT id FROM aggregation_nodes"
        ).fetchone()[0]

        changed_cfg = replace(first_cfg, redact_secrets=not first_cfg.redact_secrets)
        assert aggregation_config_version(changed_cfg) != aggregation_config_version(
            first_cfg
        )
        llm = _fusion_llm()
        second = build_aggregation_nodes(hy.conn, changed_cfg, llm)
        assert second.nodes == 1 and second.reused == 0
        assert llm.calls
        proof = load_aggregation_node_proof(hy.conn, old_id)
        assert proof is None
        replacement = hy.conn.execute(
            "SELECT id,build_config_version FROM aggregation_nodes"
        ).fetchone()
        assert replacement["id"] != old_id
        assert replacement["build_config_version"] == aggregation_config_version(
            changed_cfg
        )
    finally:
        hy.close()


def test_exact_profile_anchor_is_prompted_typed_and_authority_bound(cfg):
    hy = HyMem(cfg)
    try:
        profile_id = _seed_exact_profile_anchor(hy)
        llm = _fusion_llm()
        enabled = replace(
            _aggregation_cfg(cfg, digest=True),
            aggregation_digest_anchor_facts=8,
        )
        result = build_aggregation_nodes(hy.conn, enabled, llm)
        assert result.fusion_failures == 0
        digest_call = next(
            call for call in llm.calls
            if "standing digest of everything known" in call.system
        )
        assert "user location Amsterdam" in digest_call.user
        root = load_current_aggregation_publication(hy.conn)
        assert root is not None and root.root_node_id is not None
        typed = hy.conn.execute(
            "SELECT i.*,s.source_message_id,s.source_session_id "
            "FROM aggregation_node_inputs i "
            "JOIN aggregation_node_input_sources s "
            "ON s.node_id=i.node_id AND s.input_ordinal=i.ordinal "
            "WHERE i.node_id=? AND i.input_kind='user_profile'",
            (root.root_node_id,),
        ).fetchone()
        assert typed is not None
        assert json.loads(typed["source_ref_json"])["value"] == "Amsterdam"
        assert typed["source_session_id"] == "v55-profile-source"

        # Confidence is part of the exact authoritative profile projection.
        hy.conn.execute(
            "UPDATE user_profile SET confidence=0.8 WHERE id=?", (profile_id,)
        )
        assert load_current_aggregation_publication(hy.conn) is None
        assert load_digest(hy.conn) is None
    finally:
        hy.close()


def test_exact_kg_anchor_is_prompted_and_evidence_mutation_invalidates(cfg):
    hy = HyMem(cfg)
    try:
        _seed_native_episode(
            hy.conn, "v55-kg-leaf", title="Leaf", summary="A leaf.",
            entity="leaf-only",
        )
        edge_id = _seed_exact_kg_anchor(hy, cfg)
        llm = _fusion_llm()
        enabled = replace(
            _aggregation_cfg(cfg, digest=True),
            aggregation_digest_anchor_facts=8,
        )
        assert build_aggregation_nodes(hy.conn, enabled, llm).fusion_failures == 0
        call = next(
            item for item in llm.calls
            if "standing digest of everything known" in item.system
        )
        assert "atta part_of medflow" in call.user
        publication = load_current_aggregation_publication(hy.conn)
        assert publication is not None and publication.root_node_id is not None
        typed = hy.conn.execute(
            "SELECT * FROM aggregation_node_inputs WHERE node_id=? "
            "AND input_kind='knowledge_graph'",
            (publication.root_node_id,),
        ).fetchone()
        assert typed is not None
        assert json.loads(typed["source_ref_json"]) == {
            "object": "medflow", "predicate": "part_of", "subject": "atta",
        }

        # The full current evidence projection participates in authority_hash.
        with core_db.evidence_mutation(hy.conn):
            hy.conn.execute(
                "UPDATE knowledge_graph SET last_seen="
                "datetime(last_seen,'-1 second') WHERE id=?",
                (edge_id,),
            )
        assert load_current_aggregation_publication(hy.conn) is None
        assert load_digest(hy.conn) is None
    finally:
        hy.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "leaf_output", "root_output", "leaf_source_hash", "root_input_count",
        "missing_child", "missing_source", "orphan_same_publication",
    ],
)
def test_any_node_or_tree_tamper_invalidates_whole_publication(cfg, mutation):
    hy = HyMem(cfg)
    try:
        enabled, publication = _seed_tree(hy, cfg)
        root_id = str(publication.root_node_id)
        leaf_id = next(
            node_id for node_id, proof in publication.nodes.items()
            if proof.row["node_kind"] == "cluster"
        )
        if mutation in {"leaf_output", "root_output"}:
            hy.conn.execute("DROP TRIGGER aggregation_source_bound_update_guard")
            target = leaf_id if mutation == "leaf_output" else root_id
            hy.conn.execute(
                "UPDATE aggregation_nodes SET summary=summary || ' forged' WHERE id=?",
                (target,),
            )
        elif mutation == "leaf_source_hash":
            hy.conn.execute(
                "UPDATE aggregation_nodes SET source_manifest_hash=? WHERE id=?",
                ("sha256:" + "f" * 64, leaf_id),
            )
        elif mutation == "root_input_count":
            hy.conn.execute("DROP TRIGGER aggregation_source_header_update_guard")
            hy.conn.execute(
                "UPDATE aggregation_nodes SET input_manifest_count="
                "input_manifest_count+1 WHERE id=?",
                (root_id,),
            )
        elif mutation == "missing_child":
            hy.conn.execute("DELETE FROM aggregation_nodes WHERE id=?", (leaf_id,))
        elif mutation == "missing_source":
            hy.conn.execute(
                "DELETE FROM aggregation_node_input_sources WHERE node_id=? "
                "AND input_ordinal=0 AND source_ordinal=0",
                (leaf_id,),
            )
        else:
            hy.conn.execute(
                "INSERT INTO aggregation_nodes(id,title,summary,publication_id) "
                "VALUES ('v55-orphan','needle orphan','forged',?)",
                (publication.publication_id,),
            )

        assert load_current_aggregation_publication(hy.conn) is None
        assert load_digest(hy.conn) is None
        assert expand_node(hy.conn, root_id) is None
        assert _aggregation_search(hy.conn, "needle", top_k=3) == []
        assert augment(hy.conn, enabled, "needle", ability="TR").aggregation_nodes == []
    finally:
        hy.close()


def test_hash_consistent_singleton_rollup_is_not_a_producer_shape(cfg):
    hy = HyMem(cfg)
    try:
        episode_proofs = []
        for sid in ("singleton-a", "singleton-b"):
            episode_id, _chunk, _message = _seed_native_episode(
                hy.conn, sid, title="Singleton", summary="exact singleton source",
                entity="singleton-rollup",
            )
            episode_proofs.append(episode_input_proof(
                hy.conn, episode_id, with_session_prefix=True,
            ))
        assert all(episode_proofs)
        build_cfg = _aggregation_cfg(cfg, digest=True)
        version = aggregation_config_version(build_cfg)
        cluster = _candidate_from_episode_proofs(
            episode_proofs, kind="cluster", level=0,
            config_version=version, label="Cluster",
        )
        child_proof = _node_frontier_item(
            cluster, vector=None, entities=set()
        )["input_proof"]
        singleton = _candidate_from_episode_proofs(
            [child_proof], kind="root", level=1,
            config_version=version, label="Singleton rollup",
        )
        singleton["node_kind"] = "rollup"
        singleton["is_root"] = 0
        singleton["id"] = aggregation_node_id(
            singleton["member_ids_list"], node_kind="rollup",
            input_fingerprint=singleton["input_fingerprint"],
        )
        singleton["output_hash"] = aggregation_output_hash(
            "rollup", singleton["title"], singleton["summary"]
        )
        root_input = _node_frontier_item(
            singleton, vector=None, entities=set()
        )["input_proof"]
        root = _candidate_from_episode_proofs(
            [root_input], kind="root", level=2,
            config_version=version, label="Root",
        )
        _persist_candidate_publication(
            hy.conn, [cluster, singleton, root], config_version=version,
            root_id=root["id"],
        )
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


def test_cluster_publication_enforces_active_member_and_session_minima(cfg):
    hy = HyMem(cfg)
    try:
        version = aggregation_config_version(_aggregation_cfg(cfg))
        one_id, _chunk, _message = _seed_native_episode(
            hy.conn, "cluster-singleton", title="Singleton cluster",
            summary="one exact input", entity="singleton-cluster",
        )
        singleton_proof = episode_input_proof(
            hy.conn, one_id, with_session_prefix=True,
        )
        assert singleton_proof is not None
        singleton = _candidate_from_episode_proofs(
            [singleton_proof], kind="cluster", level=0,
            config_version=version, label="Forged singleton cluster",
        )
        _persist_candidate_publication(
            hy.conn, [singleton], config_version=version, root_id=None,
            cluster_min_members=2, cluster_min_sessions=1,
        )
        assert load_aggregation_node_proof(hy.conn, singleton["id"]) is not None
        assert load_current_aggregation_publication(hy.conn) is None

        with core_db.transaction(hy.conn):
            hy.conn.execute("DELETE FROM aggregation_publication_state")
            hy.conn.execute("DELETE FROM aggregation_nodes")
            hy.conn.execute("INSERT INTO sessions(id) VALUES ('same-session')")
            message_ids = [
                int(hy.conn.execute(
                    "INSERT INTO messages(session_id,role,content) "
                    "VALUES ('same-session','user',?)",
                    (f"same-session source {index}",),
                ).lastrowid)
                for index in range(2)
            ]
            materialize_message_coverage(hy.conn, "same-session")
            chunks = [
                Chunk(
                    id=f"same-session-chunk-{index}",
                    session_id="same-session",
                    start_message_id=message_id,
                    end_message_id=message_id,
                    salience_reason="same-session cluster proof",
                    text=f"user: same-session source {index}",
                    source_message_ids=(message_id,),
                )
                for index, message_id in enumerate(message_ids)
            ]
            persist_chunks(hy.conn, chunks)
            from hymem.dreaming.episodes import EpisodesExtraction, persist_episodes

            persist_episodes(
                hy.conn, "same-session", EpisodesExtraction(items=[
                    {
                        "title": f"Same {index}",
                        "summary": f"same-session summary {index}",
                        "outcome": "informational",
                        "key_entities": ["same-session"],
                        "chunk_ids": [chunk.id],
                    }
                    for index, chunk in enumerate(chunks)
                ]),
            )
        same_proofs = [
            episode_input_proof(
                hy.conn, str(row[0]), with_session_prefix=True,
            )
            for row in hy.conn.execute(
                "SELECT id FROM episodes WHERE session_id='same-session' ORDER BY id"
            )
        ]
        assert len(same_proofs) == 2 and all(same_proofs)
        same_session = _candidate_from_episode_proofs(
            same_proofs, kind="cluster", level=0,
            config_version=version, label="Forged same-session cluster",
        )
        _persist_candidate_publication(
            hy.conn, [same_session], config_version=version, root_id=None,
            cluster_min_members=2, cluster_min_sessions=2,
        )
        assert load_aggregation_node_proof(hy.conn, same_session["id"]) is not None
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


@pytest.mark.parametrize("anchor_cap,anchor_count", [(0, 1), (1, 2)])
def test_root_anchor_count_cannot_exceed_published_cap(
    cfg, anchor_cap, anchor_count,
):
    hy = HyMem(cfg)
    try:
        member_id, _chunk, _message = _seed_native_episode(
            hy.conn, f"anchor-cap-member-{anchor_cap}", title="Member",
            summary="exact root member", entity="anchor-cap-member",
        )
        _source_episode, _source_chunk, source_message = _seed_native_episode(
            hy.conn, f"anchor-cap-source-{anchor_cap}", title="Anchor source",
            summary="exact profile source", entity="anchor-cap-source",
        )
        source = hy.conn.execute(
            "SELECT session_id,created_at FROM messages WHERE id=?",
            (source_message,),
        ).fetchone()
        with core_db.transaction(hy.conn):
            for index in range(anchor_count):
                hy.conn.execute(
                    "INSERT INTO user_profile(slot,value,evidence_message_id,"
                    "source_message_id,source_session_id,source_created_at,confidence) "
                    "VALUES ('possession',?,?,?,?,?,0.9)",
                    (
                        f"anchor-{anchor_cap}-{index}", source_message,
                        source_message, source["session_id"], source["created_at"],
                    ),
                )
        member_proof = episode_input_proof(
            hy.conn, member_id, with_session_prefix=False,
        )
        anchors = load_profile_anchor_inputs(hy.conn, anchor_count)
        assert member_proof is not None and len(anchors) == anchor_count
        enabled = replace(
            _aggregation_cfg(cfg, digest=True),
            aggregation_digest_anchor_facts=anchor_cap,
        )
        version = aggregation_config_version(enabled)
        root = _candidate_from_episode_proofs(
            [member_proof, *anchors], kind="root", level=1,
            config_version=version, label="Forged over-cap root",
        )
        _persist_candidate_publication(
            hy.conn, [root], config_version=version, root_id=root["id"],
            anchor_fact_cap=anchor_cap,
        )
        assert load_aggregation_node_proof(hy.conn, root["id"]) is not None
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


@pytest.mark.parametrize("field", ["member_episode_ids", "session_ids"])
def test_semantic_but_noncanonical_member_json_is_rejected(cfg, field):
    hy = HyMem(cfg)
    try:
        proofs = []
        for sid in ("json-a", "json-b"):
            episode_id, _chunk, _message = _seed_native_episode(
                hy.conn, sid, title="JSON", summary="canonical JSON source",
                entity="canonical-json",
            )
            proofs.append(episode_input_proof(
                hy.conn, episode_id, with_session_prefix=True,
            ))
        assert all(proofs)
        version = aggregation_config_version(_aggregation_cfg(cfg))
        row = _candidate_from_episode_proofs(
            proofs, kind="cluster", level=0,
            config_version=version, label="Canonical JSON",
        )
        parsed = json.loads(row[field])
        row[field] = json.dumps(parsed, ensure_ascii=False, indent=1)
        _persist_candidate_publication(
            hy.conn, [row], config_version=version, root_id=None,
        )
        assert load_aggregation_node_proof(hy.conn, row["id"]) is None
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


def test_hash_consistent_shared_child_dag_is_not_a_publication(cfg):
    hy = HyMem(cfg)
    try:
        proofs = []
        for index in range(6):
            episode_id, _chunk, _message = _seed_native_episode(
                hy.conn, f"dag-{index}", title=f"DAG {index}",
                summary=f"exact DAG source {index}", entity=f"dag-{index // 2}",
            )
            proofs.append(episode_input_proof(
                hy.conn, episode_id, with_session_prefix=True,
            ))
        assert all(proofs)
        version = aggregation_config_version(_aggregation_cfg(cfg, digest=True))
        clusters = [
            _candidate_from_episode_proofs(
                proofs[index:index + 2], kind="cluster", level=0,
                config_version=version, label=f"Cluster {index // 2}",
            )
            for index in range(0, 6, 2)
        ]
        child_inputs = [
            _node_frontier_item(row, vector=None, entities=set())["input_proof"]
            for row in clusters
        ]
        rollup_a = _candidate_from_episode_proofs(
            [child_inputs[0], child_inputs[1]], kind="rollup", level=1,
            config_version=version, label="Rollup A",
        )
        rollup_b = _candidate_from_episode_proofs(
            [child_inputs[0], child_inputs[2]], kind="rollup", level=1,
            config_version=version, label="Rollup B",
        )
        root = _candidate_from_episode_proofs(
            [
                _node_frontier_item(
                    rollup_a, vector=None, entities=set()
                )["input_proof"],
                _node_frontier_item(
                    rollup_b, vector=None, entities=set()
                )["input_proof"],
            ],
            kind="root", level=2, config_version=version, label="DAG root",
        )
        _persist_candidate_publication(
            hy.conn, [*clusters, rollup_a, rollup_b, root],
            config_version=version, root_id=root["id"],
        )
        assert load_aggregation_node_proof(hy.conn, root["id"]) is not None
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


@pytest.mark.parametrize("with_root", [False, True])
def test_duplicate_episode_leaf_across_branches_is_not_a_partition(cfg, with_root):
    hy = HyMem(cfg)
    try:
        proofs = []
        for index in range(3):
            episode_id, _chunk, _message = _seed_native_episode(
                hy.conn, f"duplicate-leaf-{index}", title=f"Leaf {index}",
                summary=f"exact duplicate-leaf source {index}",
                entity=f"duplicate-leaf-{index}",
            )
            proofs.append(episode_input_proof(
                hy.conn, episode_id, with_session_prefix=True,
            ))
        assert all(proofs)
        version = aggregation_config_version(
            _aggregation_cfg(cfg, digest=with_root)
        )
        cluster_a = _candidate_from_episode_proofs(
            [proofs[0], proofs[1]], kind="cluster", level=0,
            config_version=version, label="Duplicate A",
        )
        cluster_b = _candidate_from_episode_proofs(
            [proofs[0], proofs[2]], kind="cluster", level=0,
            config_version=version, label="Duplicate B",
        )
        rows = [cluster_a, cluster_b]
        root_id = None
        if with_root:
            root = _candidate_from_episode_proofs(
                [
                    _node_frontier_item(
                        cluster_a, vector=None, entities=set()
                    )["input_proof"],
                    _node_frontier_item(
                        cluster_b, vector=None, entities=set()
                    )["input_proof"],
                ],
                kind="root", level=1, config_version=version,
                label="Duplicate root",
            )
            rows.append(root)
            root_id = root["id"]
        _persist_candidate_publication(
            hy.conn, rows, config_version=version, root_id=root_id,
        )
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


@pytest.mark.parametrize(
    ("title", "summary"),
    [
        ("", "summary"),
        (" title ", "summary"),
        ("title", "\t"),
        ("x" * 301, "summary"),
        ("title", "x" * 2001),
    ],
)
def test_hash_consistent_noncanonical_output_is_hidden(cfg, title, summary):
    hy = HyMem(cfg)
    try:
        proofs = []
        for sid in ("output-a", "output-b"):
            episode_id, _chunk, _message = _seed_native_episode(
                hy.conn, sid, title="Output", summary="exact output source",
                entity="output-language",
            )
            proofs.append(episode_input_proof(
                hy.conn, episode_id, with_session_prefix=True,
            ))
        assert all(proofs)
        version = aggregation_config_version(_aggregation_cfg(cfg))
        row = _candidate_from_episode_proofs(
            proofs, kind="cluster", level=0,
            config_version=version, label="Initially valid",
        )
        row["title"] = title
        row["summary"] = summary
        row["output_hash"] = aggregation_output_hash("cluster", title, summary)
        _persist_candidate_publication(
            hy.conn, [row], config_version=version, root_id=None,
        )
        assert load_aggregation_node_proof(hy.conn, row["id"]) is None
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


@pytest.mark.parametrize("mutation", ["malformed_ref", "type_confusion", "cross_session"])
def test_typed_input_malformed_type_confused_or_cross_session_fails_closed(
    cfg, mutation,
):
    hy = HyMem(cfg)
    try:
        first, _chunk, _message = _seed_native_episode(
            hy.conn, "typed-a", title="A", summary="needle a", entity="typed"
        )
        _seed_native_episode(
            hy.conn, "typed-b", title="B", summary="needle b", entity="typed"
        )
        _third, _chunk, _message = _seed_native_episode(
            hy.conn, "typed-c", title="C", summary="other", entity="other"
        )
        build_aggregation_nodes(hy.conn, _aggregation_cfg(cfg), _fusion_llm())
        publication = load_current_aggregation_publication(hy.conn)
        assert publication is not None
        node_id = next(iter(publication.nodes))
        input_row = hy.conn.execute(
            "SELECT ordinal FROM aggregation_node_inputs WHERE node_id=? "
            "AND input_kind='episode' AND json_extract(source_ref_json,'$.id')=?",
            (node_id, first),
        ).fetchone()
        ordinal = int(input_row["ordinal"])
        if mutation in {"malformed_ref", "type_confusion"}:
            hy.conn.execute("DROP TRIGGER aggregation_input_update_guard")
            if mutation == "malformed_ref":
                hy.conn.execute(
                    "UPDATE aggregation_node_inputs SET source_ref_json=? "
                    "WHERE node_id=? AND ordinal=?",
                    (json.dumps({"id": first, "unexpected": True}), node_id, ordinal),
                )
            else:
                hy.conn.execute(
                    "UPDATE aggregation_node_inputs SET input_kind='aggregation_node' "
                    "WHERE node_id=? AND ordinal=?", (node_id, ordinal),
                )
        else:
            hy.conn.execute("DROP TRIGGER aggregation_input_source_update_guard")
            replacement = hy.conn.execute(
                "SELECT * FROM episode_source_occurrences WHERE episode_id=?",
                (_third,),
            ).fetchone()
            hy.conn.execute(
                "UPDATE aggregation_node_input_sources SET "
                "source_message_id=?,source_session_id=?,source_role=?,"
                "source_peer_id=?,source_workspace_id=?,source_created_at=?,"
                "source_coverage_chunk_id=?,source_coverage_version=?,"
                "source_content_hash=? WHERE node_id=? AND input_ordinal=?",
                (
                    replacement["source_message_id"],
                    replacement["source_session_id"], replacement["source_role"],
                    replacement["source_peer_id"],
                    replacement["source_workspace_id"],
                    replacement["source_created_at"],
                    replacement["source_coverage_chunk_id"],
                    replacement["source_coverage_version"],
                    replacement["source_content_hash"], node_id, ordinal,
                ),
            )
        assert load_aggregation_node_proof(hy.conn, node_id) is None
        assert load_current_aggregation_publication(hy.conn) is None
        assert _aggregation_search(hy.conn, "needle") == []
    finally:
        hy.close()


def test_duplicate_typed_input_and_source_are_rejected_by_storage(cfg):
    hy = HyMem(cfg)
    try:
        _seed_native_episode(
            hy.conn, "duplicate-a", title="A", summary="a", entity="duplicate"
        )
        _seed_native_episode(
            hy.conn, "duplicate-b", title="B", summary="b", entity="duplicate"
        )
        build_aggregation_nodes(hy.conn, _aggregation_cfg(cfg), _fusion_llm())
        node_id = hy.conn.execute(
            "SELECT id FROM aggregation_nodes WHERE level=0"
        ).fetchone()[0]
        hy.conn.execute(
            "UPDATE aggregation_nodes SET source_manifest_version="
            "'aggregation-source-manifest-v1',source_manifest_count=0,"
            "source_manifest_hash=NULL,source_manifest_complete=0,"
            "input_manifest_version=NULL,input_manifest_count=0,"
            "input_manifest_hash=NULL,input_manifest_complete=0 WHERE id=?",
            (node_id,),
        )
        original = hy.conn.execute(
            "SELECT * FROM aggregation_node_inputs WHERE node_id=? AND ordinal=0",
            (node_id,),
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute(
                "INSERT INTO aggregation_node_inputs(node_id,ordinal,input_kind,"
                "source_key,source_ref_json,payload_hash,authority_hash,"
                "source_manifest_count,source_manifest_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    node_id, 99, original["input_kind"], original["source_key"],
                    original["source_ref_json"], original["payload_hash"],
                    original["authority_hash"], original["source_manifest_count"],
                    original["source_manifest_hash"],
                ),
            )
        source = hy.conn.execute(
            "SELECT * FROM aggregation_node_input_sources WHERE node_id=? "
            "AND input_ordinal=0 AND source_ordinal=0", (node_id,),
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute(
                "INSERT INTO aggregation_node_input_sources(" 
                "node_id,input_ordinal,source_ordinal,source_message_id,"
                "source_session_id,source_role,source_peer_id,source_workspace_id,"
                "source_created_at,source_coverage_chunk_id,source_coverage_version,"
                "source_content_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    node_id, 0, 99, source["source_message_id"],
                    source["source_session_id"], source["source_role"],
                    source["source_peer_id"], source["source_workspace_id"],
                    source["source_created_at"], source["source_coverage_chunk_id"],
                    source["source_coverage_version"], source["source_content_hash"],
                ),
            )
    finally:
        hy.close()


def test_failed_fusion_and_embedding_withdraw_but_retain_old_rows(cfg):
    hy = HyMem(cfg)
    try:
        for sid in ("failure-a", "failure-b"):
            _seed_native_episode(
                hy.conn, sid, title=sid, summary="shared", entity="failure"
            )
        enabled = _aggregation_cfg(cfg, digest=True)
        build_aggregation_nodes(hy.conn, enabled, _fusion_llm())
        old_ids = {
            row[0] for row in hy.conn.execute("SELECT id FROM aggregation_nodes")
        }
        _seed_native_episode(
            hy.conn, "failure-new", title="new", summary="new leaf", entity="new"
        )
        root_failure = StubLLMClient(
            fixtures={
                "fuse several related episodes": json.dumps({
                    "title": "cluster", "summary": "cluster",
                }),
                "standing digest of everything known": "{",
            },
            default="[]",
        )
        result = build_aggregation_nodes(hy.conn, enabled, root_failure)
        assert result.fusion_failures > 0
        assert load_current_aggregation_publication(hy.conn) is None
        assert old_ids <= {
            row[0] for row in hy.conn.execute("SELECT id FROM aggregation_nodes")
        }

        # Heal once, then force a provider failure on a re-keyed level-0 node.
        build_aggregation_nodes(hy.conn, enabled, _fusion_llm())
        old_ids = {
            row[0] for row in hy.conn.execute("SELECT id FROM aggregation_nodes")
        }
        _seed_native_episode(
            hy.conn, "failure-member", title="member", summary="shared",
            entity="failure",
        )
        failing = MappedStubEmbeddingClient(
            model="aggregation-failure-v1", dim=16, fail_on="", conn=hy.conn,
        )
        with pytest.raises(RuntimeError, match="provider unavailable"):
            build_aggregation_nodes(
                hy.conn, enabled, _fusion_llm(), failing,
            )
        assert failing.transaction_states
        assert not any(failing.transaction_states)
        assert load_current_aggregation_publication(hy.conn) is None
        assert old_ids <= {
            row[0] for row in hy.conn.execute("SELECT id FROM aggregation_nodes")
        }
    finally:
        hy.close()


def test_unpublished_text_never_reaches_embedding_provider(cfg):
    hy = HyMem(cfg)
    try:
        _seed_native_episode(
            hy.conn, "provider-a", title="A", summary="safe needle", entity="p"
        )
        _seed_native_episode(
            hy.conn, "provider-b", title="B", summary="safe needle", entity="p"
        )
        build_aggregation_nodes(hy.conn, _aggregation_cfg(cfg), _fusion_llm())
        hy.conn.execute(
            "INSERT INTO aggregation_nodes(id,title,summary,member_episode_ids,"
            "session_ids,n_members,n_sessions,level,is_root) VALUES "
            "('unpublished-poison','POISON','must never embed','[\"x\"]',"
            "'[\"x\"]',1,1,0,0)"
        )
        embedder = StubEmbeddingClient()
        pending = fetch_node_embeddings(hy.conn, embedder)
        assert pending is not None
        assert pending.node_ids != ["unpublished-poison"]
        assert all("POISON" not in text for batch in embedder.calls for text in batch)
    finally:
        hy.close()


def test_unpublished_embedding_cannot_trigger_query_provider(cfg):
    hy = HyMem(cfg)
    try:
        for sid in ("query-provider-a", "query-provider-b"):
            _seed_native_episode(
                hy.conn, sid, title="Safe", summary="safe publication",
                entity="query-provider",
            )
        build_aggregation_nodes(hy.conn, _aggregation_cfg(cfg), _fusion_llm())
        embedder = StubEmbeddingClient()
        producer_key, dimension = embedding_storage_identity(embedder)
        poison_text = "POISON\nunpublished provider bait"
        poison_vector = [0.0] * dimension
        poison_vector[0] = 1.0
        with core_db.transaction(hy.conn), core_db.embedding_mutation(hy.conn):
            hy.conn.execute(
                "INSERT INTO aggregation_nodes(id,title,summary) "
                "VALUES ('unpublished-query-poison','POISON',"
                "'unpublished provider bait')"
            )
            hy.conn.execute(
                "INSERT INTO aggregation_node_embeddings("
                "node_id,vector_json,model,dim,text_hash,"
                "embedding_producer_key) VALUES (?,?,?,?,?,?)",
                (
                    "unpublished-query-poison", encode_vector(poison_vector),
                    producer_key, dimension,
                    embedding_text_hash(poison_text),
                    producer_key,
                ),
            )
        assert _aggregation_search(
            hy.conn, "lexically absent query", embedding_client=embedder,
            top_k=1, max_scan=1,
        ) == []
        assert embedder.calls == []
    finally:
        hy.close()


def test_embedding_scan_order_ignores_unattested_timestamps(cfg, tmp_path):
    embedder = StubEmbeddingClient()
    hy = HyMem(replace(cfg, root=tmp_path / "embedding-order"))
    try:
        for group in range(3):
            for member in range(2):
                _seed_native_episode(
                    hy.conn, f"order-{group}-{member}", title=f"Order {group}",
                    summary=f"attested candidate {group}", entity=f"order-{group}",
                )
        build_aggregation_nodes(
            hy.conn, _aggregation_cfg(cfg), _fusion_llm(), embedder
        )
        node_ids = sorted(
            row[0] for row in hy.conn.execute(
                "SELECT node_id FROM aggregation_node_embeddings"
            )
        )
        assert len(node_ids) == 3
        target = node_ids[0]
        query_vector = decode_vector(hy.conn.execute(
            "SELECT vector_json FROM aggregation_node_embeddings WHERE node_id=?",
            (target,),
        ).fetchone()[0])
        model, dimension = embedding_storage_identity(embedder)
        bound_query_vector = _BoundQueryVector(
            tuple(query_vector), model, dimension,
        )

        with core_db.embedding_mutation(hy.conn):
            for index, node_id in enumerate(node_ids):
                hy.conn.execute(
                    "UPDATE aggregation_node_embeddings SET created_at=? "
                    "WHERE node_id=?",
                    (f"200{index}-01-01 00:00:00", node_id),
                )
        first_receipt = material_store_state(hy.config.db_path)
        first = _aggregation_search(
            hy.conn, "lexically absent ordering query", top_k=1,
            embedding_client=embedder, query_vector=bound_query_vector,
            max_scan=1,
        )

        with core_db.embedding_mutation(hy.conn):
            for index, node_id in enumerate(reversed(node_ids)):
                hy.conn.execute(
                    "UPDATE aggregation_node_embeddings SET created_at=? "
                    "WHERE node_id=?",
                    (f"200{index}-01-01 00:00:00", node_id),
                )
        second_receipt = material_store_state(hy.config.db_path)
        second = _aggregation_search(
            hy.conn, "lexically absent ordering query", top_k=1,
            embedding_client=embedder, query_vector=bound_query_vector,
            max_scan=1,
        )
        assert [hit.node_id for hit in first] == [target]
        assert [hit.node_id for hit in second] == [target]
        assert first_receipt["sha256"] == second_receipt["sha256"]
    finally:
        hy.close()


def test_fetch_embeddings_renders_only_snapshot_proof_rows(
    cfg, monkeypatch,
):
    hy = HyMem(cfg)
    try:
        for sid in ("fetch-snapshot-a", "fetch-snapshot-b"):
            _seed_native_episode(
                hy.conn, sid, title="Safe title", summary="SAFE PROVED OUTPUT",
                entity="fetch-snapshot",
            )
        build_aggregation_nodes(hy.conn, _aggregation_cfg(cfg), _fusion_llm())
        saved = load_current_aggregation_publication(hy.conn)
        assert saved is not None
        node_id = next(iter(saved.nodes))
        hy.conn.execute("DROP TRIGGER aggregation_source_bound_update_guard")
        hy.conn.execute(
            "UPDATE aggregation_nodes SET summary='POISON NEVER PROVED' "
            "WHERE id=?", (node_id,),
        )
        monkeypatch.setattr(
            aggregate_mod, "load_current_aggregation_publication",
            lambda _conn, **_kwargs: saved,
        )

        embedder = MappedStubEmbeddingClient(
            model="snapshot-render-v1", dim=16, conn=hy.conn,
        )
        pending = fetch_node_embeddings(hy.conn, embedder)
        assert pending is not None
        assert embedder.transaction_states
        assert not any(embedder.transaction_states)
        provider_text = "\n".join(text for batch in embedder.calls for text in batch)
        assert "needle links the exact source episodes" in provider_text
        assert "POISON NEVER PROVED" not in provider_text
    finally:
        hy.close()


def test_cluster_fusion_renders_exact_snapshot_input_proofs(
    cfg, monkeypatch,
):
    hy = HyMem(cfg)
    try:
        episode_ids = []
        for sid in ("cluster-snapshot-a", "cluster-snapshot-b"):
            episode_id, _chunk, _message = _seed_native_episode(
                hy.conn, sid, title="Safe episode", summary="SAFE EPISODE BYTES",
                entity="cluster-snapshot",
            )
            episode_ids.append(episode_id)
        # Capture the coherent episode/proof snapshot before the adversarial
        # row mutation.  The optimized loader now constructs both renderings
        # from that one resolved row+manifest instead of reloading through a
        # monkeypatchable helper.
        members = load_clusterable_episodes(hy.conn)
        assert {item["id"] for item in members} == set(episode_ids)
        hy.conn.execute("DROP TRIGGER episode_source_bound_update_guard")
        hy.conn.execute(
            "UPDATE episodes SET summary='POISON PROMPT BYTES' WHERE id=?",
            (episode_ids[0],),
        )
        llm = _fusion_llm()
        assert _summarize_cluster(members, _aggregation_cfg(cfg), llm) is not None
        prompt = llm.calls[-1].user
        assert "SAFE EPISODE BYTES" in prompt
        assert "POISON PROMPT BYTES" not in prompt
    finally:
        hy.close()


class _QualitySpyEmbedder:
    def __init__(self):
        self.inner = StubEmbeddingClient()
        self.quality_reads = 0

    @property
    def model(self):
        return self.inner.model

    @property
    def dim(self):
        return self.inner.dim

    @property
    def quality(self):
        self.quality_reads += 1
        return "semantic"

    def embed(self, texts):
        return self.inner.embed(texts)


def test_unscoped_poison_cannot_starve_fts_or_vector_validation(
    cfg, monkeypatch,
):
    embedder = MappedStubEmbeddingClient(
        model="unscoped-poison-v1", dim=16, quality="semantic",
    )
    hy = HyMem(cfg)
    try:
        for group in range(3):
            for member in range(2):
                _seed_native_episode(
                    hy.conn, f"poison-valid-{group}-{member}",
                    title=f"Needle {group}",
                    summary=f"needle valid cluster {group}",
                    entity=f"poison-group-{group}",
                )
        build_aggregation_nodes(
            hy.conn, _aggregation_cfg(cfg), _fusion_llm(), embedder
        )
        publication = load_current_aggregation_publication(
            hy.conn, embedding_client=embedder,
        )
        assert publication is not None and len(publication.nodes) == 3

        poison_text = "needle unpublished poison"
        poison_vector = embedder.embed([poison_text])[0]
        producer_key, dimension = embedding_storage_identity(embedder)
        with core_db.transaction(hy.conn), core_db.embedding_mutation(hy.conn):
            # Larger than the historical 1024-row raw cap. Neither arm may
            # LIMIT this prefix before checking publication + typed proofs.
            for index in range(1100):
                node_id = f"aaa-unpublished-poison-{index:04d}"
                hy.conn.execute(
                    "INSERT INTO aggregation_nodes(id,title,summary,"
                    "member_episode_ids,session_ids,n_members,n_sessions,"
                    "created_at,level,is_root,input_fingerprint) "
                    "VALUES (?,'needle','unpublished poison','[\"legacy\"]',"
                    "'[\"legacy\"]',1,1,'9999-01-01 00:00:00',0,0,?)",
                    (node_id, "sha256:" + "0" * 64),
                )
                hy.conn.execute(
                    "INSERT INTO aggregation_node_embeddings(node_id,vector_json,"
                    "model,dim,text_hash,created_at,embedding_producer_key) "
                    "VALUES (?,?,?,?,?,'9999-01-01 00:00:00',?)",
                    (
                        node_id, encode_vector(poison_vector), producer_key,
                        dimension, embedding_text_hash(poison_text), producer_key,
                    ),
                )

        # The published material requires this exact vector producer even
        # when the current query has lexical hits; omission is not authority
        # to serve a historical vector-bound tree.
        fts = _aggregation_search(
            hy.conn, "needle", top_k=3, embedding_client=embedder,
        )
        assert {hit.node_id for hit in fts} == set(publication.nodes)
        assert all("unpublished-poison" not in hit.node_id for hit in fts)

        good_id = sorted(publication.nodes)[0]
        good_vector = decode_vector(hy.conn.execute(
            "SELECT vector_json FROM aggregation_node_embeddings WHERE node_id=?",
            (good_id,),
        ).fetchone()[0])
        bound_good_vector = _BoundQueryVector(
            tuple(good_vector), producer_key, dimension,
        )
        quality_reads = 0
        import importlib
        query_augment_module = importlib.import_module("hymem.query.augment")
        original_quality_check = query_augment_module._quality_allows_candidate

        def counted_quality_check(client, query, text):
            nonlocal quality_reads
            quality_reads += 1
            return original_quality_check(client, query, text)

        monkeypatch.setattr(
            query_augment_module, "_quality_allows_candidate",
            counted_quality_check,
        )
        vector = _aggregation_search(
            hy.conn, "no lexical match here", top_k=1,
            embedding_client=embedder, max_scan=1,
            query_vector=bound_good_vector,
        )
        assert [hit.node_id for hit in vector] == [good_id]
        # Bad text hashes and malformed vectors are rejected before the
        # provider-defined quality property and unpublished poison does not
        # consume max_scan.
        assert quality_reads == 1

        # v57 makes the published vector set part of the material authority.
        # Corrupting even one published mirror therefore withdraws the whole
        # publication rather than serving around two unauthenticated rows.
        with core_db.embedding_mutation(hy.conn):
            hy.conn.execute(
                "UPDATE aggregation_node_embeddings SET text_hash=? "
                "WHERE node_id=?",
                ("0" * 64, good_id),
            )
        assert load_current_aggregation_publication(
            hy.conn, embedding_client=embedder,
        ) is None
        assert _aggregation_search(
            hy.conn, "needle", top_k=3, embedding_client=embedder,
        ) == []
    finally:
        hy.close()


def test_unsourced_and_forged_complete_legacy_episodes_never_reach_providers(cfg):
    embedder = StubEmbeddingClient()
    hy = HyMem(cfg)
    try:
        for sid in ("proved-a", "proved-b"):
            _seed_native_episode(
                hy.conn, sid, title="Proved", summary="proved provider input",
                entity="proved-provider",
            )
        with core_db.transaction(hy.conn):
            hy.conn.execute("INSERT INTO sessions(id) VALUES ('legacy-unsourced')")
            hy.conn.execute(
                "INSERT INTO episodes(id,session_id,title,summary,key_entities) "
                "VALUES ('legacy-incomplete','legacy-unsourced','DO NOT FUSE',"
                "'DO NOT EMBED INCOMPLETE','[\"legacy\"]')"
            )
            hy.conn.execute(
                "INSERT INTO episodes(id,session_id,title,summary,key_entities) "
                "VALUES ('legacy-forged','legacy-unsourced','DO NOT FUSE',"
                "'DO NOT EMBED FORGED','[\"legacy\"]')"
            )
            hy.conn.execute("DROP TRIGGER episode_source_header_update_guard")
            hy.conn.execute(
                "UPDATE episodes SET source_manifest_version="
                "'episode-source-manifest-v1',source_manifest_count=1,"
                "source_manifest_hash=?,source_manifest_complete=1 "
                "WHERE id='legacy-forged'",
                ("sha256:" + "f" * 64,),
            )
        core_db._install_aggregation_source_guards(hy.conn)

        llm = _fusion_llm()
        build_aggregation_nodes(
            hy.conn, _aggregation_cfg(cfg), llm, embedder
        )
        provider_text = "\n".join(
            [call.user for call in llm.calls]
            + [text for batch in embedder.calls for text in batch]
        )
        assert "DO NOT FUSE" not in provider_text
        assert "DO NOT EMBED" not in provider_text
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM aggregation_node_inputs "
            "WHERE input_kind='episode' AND json_extract(source_ref_json,'$.id') "
            "IN ('legacy-incomplete','legacy-forged')"
        ).fetchone()[0] == 0
    finally:
        hy.close()


class _PromptSelectiveLLM:
    def __init__(self):
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if request.user == "FULL INPUT":
            return "{"
        if request.user == "SHRUNK INPUT":
            return json.dumps({"title": "wrong", "summary": "wrong"})
        return "[]"


def test_shrink_success_is_never_published_under_full_input_identity():
    llm = _PromptSelectiveLLM()
    shrink_calls = 0

    def shrink():
        nonlocal shrink_calls
        shrink_calls += 1
        return "SHRUNK INPUT"

    assert aggregate_mod._llm_fuse(
        "FULL INPUT", llm, system="test", shrink=shrink
    ) is None
    assert shrink_calls == 0
    assert [call.user for call in llm.calls] == ["FULL INPUT", "FULL INPUT"]


def test_export_import_reexport_is_exact_and_aggregation_stays_local(cfg, tmp_path):
    src = HyMem(replace(cfg, root=tmp_path / "portable-src"))
    try:
        _seed_native_episode(
            src.conn, "portable-a", title="A", summary="a", entity="portable"
        )
        _seed_native_episode(
            src.conn, "portable-b", title="B", summary="b", entity="portable"
        )
        build_aggregation_nodes(src.conn, _aggregation_cfg(src.config), _fusion_llm())
        first = tmp_path / "first.jsonl"
        src.export(first)
    finally:
        src.close()

    dst = HyMem(replace(cfg, root=tmp_path / "portable-dst"))
    try:
        dst.import_(first)
        assert dst.conn.execute(
            "SELECT COUNT(*) FROM aggregation_nodes"
        ).fetchone()[0] == 0
        assert dst.conn.execute(
            "SELECT COUNT(*) FROM aggregation_publication_state"
        ).fetchone()[0] == 0
        second = tmp_path / "second.jsonl"
        dst.export(second)
        assert second.read_bytes() == first.read_bytes()
        assert sum(dst.import_(first).values()) == 0
    finally:
        dst.close()


def test_forged_typed_wire_rolls_back_without_withdrawing_publication(cfg, tmp_path):
    source = HyMem(replace(cfg, root=tmp_path / "wire-source"))
    try:
        source.open_session("wire-new")
        source.log_message("wire-new", "user", "new portable source")
        wire = tmp_path / "forged.jsonl"
        source.export(wire)
    finally:
        source.close()
    _rewrite_portable_wire(
        wire,
        lambda body: body.append({
            "type": "aggregation_node_input",
            "record": {"input_kind": "forged-anchor"},
        }),
    )

    target = HyMem(replace(cfg, root=tmp_path / "wire-target"))
    try:
        _seed_native_episode(
            target.conn, "wire-a", title="A", summary="a", entity="wire"
        )
        _seed_native_episode(
            target.conn, "wire-b", title="B", summary="b", entity="wire"
        )
        build_aggregation_nodes(
            target.conn, _aggregation_cfg(target.config), _fusion_llm()
        )
        before = dict(target.conn.execute(
            "SELECT * FROM aggregation_publication_state"
        ).fetchone())
        before_sessions = target.conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        with pytest.raises(ValueError):
            target.import_(wire)
        assert dict(target.conn.execute(
            "SELECT * FROM aggregation_publication_state"
        ).fetchone()) == before
        assert target.conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0] == before_sessions
        assert load_current_aggregation_publication(target.conn) is not None
    finally:
        target.close()


def test_v14_import_conservatively_invalidates_local_publication(cfg, tmp_path):
    source = HyMem(replace(cfg, root=tmp_path / "v14-source"))
    try:
        source.open_session("v14-new")
        source.log_message("v14-new", "user", "old format source")
        wire = tmp_path / "v14.jsonl"
        source.export(wire)
    finally:
        source.close()
    _rewrite_portable_wire(
        wire,
        lambda body: body[0].update({"version": 14, "schema_version": 54}),
    )

    target = HyMem(replace(cfg, root=tmp_path / "v14-target"))
    try:
        _seed_native_episode(
            target.conn, "v14-a", title="A", summary="a", entity="v14"
        )
        _seed_native_episode(
            target.conn, "v14-b", title="B", summary="b", entity="v14"
        )
        build_aggregation_nodes(
            target.conn, _aggregation_cfg(target.config), _fusion_llm()
        )
        assert load_current_aggregation_publication(target.conn) is not None
        target.import_(wire)
        assert load_current_aggregation_publication(target.conn) is None
        # Portable import withdraws read authority immediately; historical
        # rows remain audit evidence until a successful replacement build.
        assert target.conn.execute(
            "SELECT COUNT(*) FROM aggregation_nodes"
        ).fetchone()[0] > 0
    finally:
        target.close()


def test_v15_in_place_profile_strengthening_invalidates_aggregation(
    cfg, tmp_path,
):
    source = HyMem(replace(cfg, root=tmp_path / "profile-update-source"))
    wire = tmp_path / "profile-update.jsonl"
    try:
        for value, confidence in (("Car", 0.5), ("Bike", 0.8)):
            _episode, _chunk, message_id = _seed_native_episode(
                source.conn, f"profile-{value.lower()}", title=value,
                summary=f"The user owns a {value}.", entity="portable-profile",
            )
            created_at = source.conn.execute(
                "SELECT created_at FROM messages WHERE id=?", (message_id,),
            ).fetchone()[0]
            with core_db.transaction(source.conn):
                source.conn.execute(
                    "INSERT INTO user_profile("
                    "slot,value,evidence_message_id,confidence,source_message_id,"
                    "source_session_id,source_created_at) "
                    "VALUES ('possession',?,?,?,?,?,?)",
                    (
                        value, message_id, confidence, message_id,
                        f"profile-{value.lower()}", created_at,
                    ),
                )
        source.export(wire)
    finally:
        source.close()

    target = HyMem(replace(cfg, root=tmp_path / "profile-update-target"))
    try:
        target.import_(wire)
        build_cfg = replace(
            _aggregation_cfg(target.config, digest=True),
            aggregation_digest_anchor_facts=1,
        )
        build_aggregation_nodes(target.conn, build_cfg, _fusion_llm())
        publication = load_current_aggregation_publication(target.conn)
        assert publication is not None and publication.root_node_id is not None
        selected = target.conn.execute(
            "SELECT source_ref_json FROM aggregation_node_inputs "
            "WHERE node_id=? AND input_kind='user_profile'",
            (publication.root_node_id,),
        ).fetchall()
        assert [json.loads(row[0])["value"] for row in selected] == ["Bike"]

        def strengthen_car(body):
            car = next(
                item for item in body
                if item.get("type") == "user_profile_fact"
                and item["record"].get("value") == "Car"
            )
            car["record"]["confidence"] = 0.99

        _rewrite_portable_wire(wire, strengthen_car)
        imported = target.import_(wire)
        assert imported.get("user_profile_fact", 0) == 0
        assert target.conn.execute(
            "SELECT confidence FROM user_profile WHERE value='Car'"
        ).fetchone()[0] == pytest.approx(0.99)
        assert load_current_aggregation_publication(target.conn) is None
        # Root-domain triggers are intentionally inert in v57.  Exact anchor
        # re-selection rejects the old publication without deleting its
        # immutable audit rows.
        assert target.conn.execute(
            "SELECT COUNT(*) FROM aggregation_nodes"
        ).fetchone()[0] > 0
    finally:
        target.close()


def test_source_proof_mutation_changes_material_attestation(cfg, tmp_path):
    hy = HyMem(replace(cfg, root=tmp_path / "attestation"))
    try:
        _seed_native_episode(
            hy.conn, "attest-a", title="A", summary="a", entity="attest"
        )
        _seed_native_episode(
            hy.conn, "attest-b", title="B", summary="b", entity="attest"
        )
        build_aggregation_nodes(hy.conn, _aggregation_cfg(hy.config), _fusion_llm())
        before = material_store_state(hy.config.db_path)
        hy.conn.execute("DROP TRIGGER aggregation_input_source_update_guard")
        hy.conn.execute(
            "UPDATE aggregation_node_input_sources SET source_content_hash=? "
            "WHERE rowid=(SELECT rowid FROM aggregation_node_input_sources LIMIT 1)",
            ("f" * 64,),
        )
        core_db._install_aggregation_source_guards(hy.conn)
        after = material_store_state(hy.config.db_path)
        assert before["sha256"] != after["sha256"]
        assert (
            before["tables"]["aggregation_node_input_sources"]["sha256"]
            != after["tables"]["aggregation_node_input_sources"]["sha256"]
        )
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


def test_publication_authority_alone_changes_material_attestation(cfg, tmp_path):
    hy = HyMem(replace(cfg, root=tmp_path / "publication-attestation"))
    try:
        _seed_native_episode(
            hy.conn, "publication-a", title="A", summary="a",
            entity="publication-attest",
        )
        _seed_native_episode(
            hy.conn, "publication-b", title="B", summary="b",
            entity="publication-attest",
        )
        build_aggregation_nodes(hy.conn, _aggregation_cfg(hy.config), _fusion_llm())
        before = material_store_state(hy.config.db_path)
        state = dict(hy.conn.execute(
            "SELECT * FROM aggregation_publication_state"
        ).fetchone())
        hy.conn.execute("DELETE FROM aggregation_publication_state")
        hy.conn.execute(
            "INSERT INTO aggregation_publication_state("
            "id,publication_id,config_version,cluster_min_members,"
            "cluster_min_sessions,anchor_fact_cap,root_node_id,node_count,"
            "node_set_hash,published_at,aggregation_generation_key,"
            "request_contract_sha256,aggregation_material_epoch_key,"
            "material_revision,node_embedding_count,node_embedding_set_hash) "
            "VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                state["publication_id"], state["config_version"],
                state["cluster_min_members"], state["cluster_min_sessions"],
                state["anchor_fact_cap"], "forged-root", state["node_count"],
                state["node_set_hash"],
                state["published_at"],
                state["aggregation_generation_key"],
                state["request_contract_sha256"],
                state["aggregation_material_epoch_key"],
                state["material_revision"], state["node_embedding_count"],
                state["node_embedding_set_hash"],
            ),
        )
        after = material_store_state(hy.config.db_path)
        assert before["sha256"] != after["sha256"]
        assert (
            before["tables"]["aggregation_publication_state"]["sha256"]
            != after["tables"]["aggregation_publication_state"]["sha256"]
        )
        unchanged = {
            table for table in before["tables"]
            if before["tables"][table]["sha256"]
            == after["tables"][table]["sha256"]
        }
        assert set(before["tables"]) - unchanged == {
            "aggregation_publication_state"
        }
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


@pytest.mark.parametrize(
    "malformed",
    ["", "not-a-utc-timestamp!", "2026-99-99 99:99:99", "SECRET\n## injection"],
)
def test_publication_timestamp_storage_rejects_malformed_text(cfg, malformed):
    hy = HyMem(cfg)
    try:
        version = aggregation_config_version(_aggregation_cfg(cfg))
        empty_hash = aggregation_publication_node_set_hash(())
        hy.conn.execute("DELETE FROM aggregation_publication_state")
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute(
                "INSERT INTO aggregation_publication_state("
                "id,publication_id,config_version,cluster_min_members,"
                "cluster_min_sessions,anchor_fact_cap,root_node_id,node_count,"
                "node_set_hash,published_at) VALUES (1,?,?,2,2,20,NULL,0,?,?)",
                (
                    "sha256:" + "0" * 64, version, empty_hash, malformed,
                ),
            )
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


def test_canonical_publication_timestamp_replacement_breaks_identity(cfg):
    hy = HyMem(cfg)
    try:
        for sid in ("clock-a", "clock-b"):
            _seed_native_episode(
                hy.conn, sid, title="Clock", summary="clock-bound source",
                entity="clock-bound",
            )
        build_aggregation_nodes(
            hy.conn, _aggregation_cfg(cfg, digest=True), _fusion_llm()
        )
        state = dict(hy.conn.execute(
            "SELECT * FROM aggregation_publication_state"
        ).fetchone())
        replacement = "2099-01-01 00:00:00"
        assert replacement != state["published_at"]
        hy.conn.execute("DELETE FROM aggregation_publication_state")
        hy.conn.execute(
            "INSERT INTO aggregation_publication_state("
            "id,publication_id,config_version,cluster_min_members,"
            "cluster_min_sessions,anchor_fact_cap,root_node_id,node_count,"
            "node_set_hash,published_at,aggregation_generation_key,"
            "request_contract_sha256,aggregation_material_epoch_key,"
            "material_revision,node_embedding_count,node_embedding_set_hash) "
            "VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                state["publication_id"], state["config_version"],
                state["cluster_min_members"], state["cluster_min_sessions"],
                state["anchor_fact_cap"], state["root_node_id"], state["node_count"],
                state["node_set_hash"], replacement,
                state["aggregation_generation_key"],
                state["request_contract_sha256"],
                state["aggregation_material_epoch_key"],
                state["material_revision"], state["node_embedding_count"],
                state["node_embedding_set_hash"],
            ),
        )
        assert load_current_aggregation_publication(hy.conn) is None
        assert load_digest(hy.conn) is None
    finally:
        hy.close()


def test_unattested_node_clock_cannot_change_digest_context(cfg, tmp_path):
    hy = HyMem(replace(cfg, root=tmp_path / "digest-clock"))
    try:
        for sid in ("digest-clock-a", "digest-clock-b"):
            _seed_native_episode(
                hy.conn, sid, title="Clock", summary="clock context source",
                entity="digest-clock",
            )
        build_aggregation_nodes(
            hy.conn, _aggregation_cfg(cfg, digest=True), _fusion_llm()
        )
        before_digest = load_digest(hy.conn)
        assert before_digest is not None
        before_block = before_digest.as_context_block()
        before_receipt = material_store_state(hy.config.db_path)
        hy.conn.execute(
            "UPDATE aggregation_nodes SET created_at='2099-01-01 00:00:00'"
        )
        after_digest = load_digest(hy.conn)
        after_receipt = material_store_state(hy.config.db_path)
        assert after_digest is not None
        assert after_digest.as_context_block() == before_block
        assert after_receipt["sha256"] == before_receipt["sha256"]
    finally:
        hy.close()


def _downgrade_empty_v55_to_v54(conn: sqlite3.Connection) -> None:
    # Remove the complete v57 tail before reconstructing its authentic v56
    # predecessor.  Later support triggers name v56 columns and would make an
    # otherwise-correct v55 rollback fixture fail during ALTER TABLE.
    statements = core_db._split_sql_statements(
        files("hymem.core.migrations").joinpath(
            "057_aggregation_material_epoch.sql"
        ).read_text(encoding="utf-8")
    )
    for statement in statements:
        match = re.match(r"\s*CREATE\s+TRIGGER\s+(\w+)", statement, re.I)
        if match:
            conn.execute(f'DROP TRIGGER IF EXISTS "{match.group(1)}"')
    for statement in statements:
        match = re.match(
            r"\s*CREATE\s+(?:UNIQUE\s+)?(INDEX|VIEW)\s+(\w+)",
            statement, re.I,
        )
        if match:
            conn.execute(
                f'DROP {match.group(1).upper()} IF EXISTS "{match.group(2)}"'
            )
    conn.execute("DELETE FROM aggregation_publication_state")
    for table, columns in (
        ("episode_embeddings", ("embedding_producer_key",)),
        ("aggregation_node_embeddings", ("embedding_producer_key",)),
        ("aggregation_nodes", ("aggregation_material_epoch_key",)),
        ("aggregation_publication_state", (
            "aggregation_material_epoch_key", "material_revision",
            "node_embedding_count", "node_embedding_set_hash",
        )),
        ("aggregation_build_health", (
            "last_success_material_epoch_key", "pending_material_epoch_key",
            "last_failure_material_epoch_key", "attempt_serial",
            "pending_attempt_token",
        )),
        ("dream_runs", ("aggregation_material_epoch_key",)),
    ):
        for column in columns:
            conn.execute(f'ALTER TABLE "{table}" DROP COLUMN "{column}"')
    conn.execute("DROP TABLE aggregation_material_epochs")
    conn.execute("DROP TABLE aggregation_material_clock")
    conn.execute(
        "DELETE FROM schema_meta WHERE key='aggregation_material_epoch_schema'"
    )
    conn.execute("UPDATE schema_meta SET value='56' WHERE key='schema_version'")

    for trigger in (
        "aggregation_health_attempt_insert_guard",
        "aggregation_health_attempt_update_guard",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    for trigger in (
        "aggregation_generations_insert_guard",
        "aggregation_generations_update_guard",
        "aggregation_generations_delete_guard",
        "aggregation_generation_node_update_guard",
        "aggregation_generation_publication_insert_guard",
    ):
        conn.execute(f"DROP TRIGGER {trigger}")
    for table, columns in (
        ("aggregation_nodes", (
            "aggregation_request_hash", "aggregation_generation_key",
        )),
        ("aggregation_publication_state", (
            "request_contract_sha256", "aggregation_generation_key",
        )),
        ("aggregation_build_health", (
            "last_failure_generation_key", "pending_generation_key",
            "last_success_generation_key",
        )),
        ("dream_runs", ("aggregation_generation_key",)),
    ):
        for column in columns:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    conn.execute("DROP TABLE aggregation_generations")
    conn.execute(
        "DELETE FROM schema_meta WHERE key='aggregation_generation_schema'"
    )
    for trigger in (
        "aggregation_source_header_insert_guard",
        "aggregation_source_header_update_guard",
        "aggregation_source_bound_update_guard",
        "aggregation_source_occurrence_insert_guard",
        "aggregation_source_occurrence_update_guard",
        "aggregation_source_occurrence_delete_unpublishes",
        "aggregation_input_insert_guard", "aggregation_input_update_guard",
        "aggregation_input_delete_unpublishes",
        "aggregation_input_source_insert_guard",
        "aggregation_input_source_update_guard",
        "aggregation_input_source_delete_unpublishes",
        "aggregation_publication_update_guard",
    ):
        conn.execute(f"DROP TRIGGER {trigger}")
    conn.execute("DROP TABLE aggregation_node_input_sources")
    conn.execute("DROP TABLE aggregation_node_inputs")
    conn.execute("DROP TABLE aggregation_publication_state")
    for column in (
        "build_config_version", "publication_id", "output_hash", "node_kind",
        "input_manifest_complete", "input_manifest_hash",
        "input_manifest_count", "input_manifest_version",
    ):
        conn.execute(f"ALTER TABLE aggregation_nodes DROP COLUMN {column}")
    conn.execute(
        "UPDATE schema_meta SET value='54' WHERE key='schema_version'"
    )
    conn.execute(
        "DELETE FROM schema_meta WHERE key='aggregation_typed_provenance_schema'"
    )


def test_v55_migration_failure_rolls_back_columns_tables_and_stamp(
    cfg, monkeypatch,
):
    hy = HyMem(cfg)
    try:
        _downgrade_empty_v55_to_v54(hy.conn)
        original = core_db._apply_migration_sql

        def fail_after_first_statement(conn, script):
            first = core_db._split_sql_statements(script)[0]
            conn.execute(first)
            raise RuntimeError("injected v55 migration failure")

        monkeypatch.setattr(core_db, "_apply_migration_sql", fail_after_first_statement)
        with pytest.raises(RuntimeError, match="injected v55 migration failure"):
            core_db._run_migrations(hy.conn)
        assert core_db.schema_version(hy.conn) == 54
        assert "input_manifest_version" not in {
            row["name"] for row in hy.conn.execute(
                "PRAGMA table_info(aggregation_nodes)"
            )
        }
        assert hy.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='aggregation_node_inputs'"
        ).fetchone() is None

        monkeypatch.setattr(core_db, "_apply_migration_sql", original)
        core_db._run_migrations(hy.conn)
        assert core_db.schema_version(hy.conn) == core_db.EXPECTED_SCHEMA_VERSION
        assert core_db._v56_generation_bindings_present(
            hy.conn, allow_v57=True,
        )
        assert core_db._v57_material_bindings_present(hy.conn)
    finally:
        hy.close()


@pytest.mark.parametrize(
    "trigger",
    [
        "aggregation_source_header_insert_guard",
        "aggregation_source_header_update_guard",
        "aggregation_source_bound_update_guard",
        "aggregation_source_occurrence_insert_guard",
        "aggregation_source_occurrence_update_guard",
        "aggregation_source_occurrence_delete_unpublishes",
        "aggregation_input_insert_guard", "aggregation_input_update_guard",
        "aggregation_input_delete_unpublishes",
        "aggregation_input_source_insert_guard",
        "aggregation_input_source_update_guard",
        "aggregation_input_source_delete_unpublishes",
        "aggregation_publication_update_guard",
    ],
)
def test_reopen_replaces_every_missing_or_weakened_v55_trigger(tmp_path, trigger):
    path = tmp_path / f"{trigger}.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    conn.execute(f"DROP TRIGGER {trigger}")
    conn.execute(
        f"CREATE TRIGGER {trigger} AFTER INSERT ON aggregation_nodes "
        "BEGIN SELECT 1; END"
    )
    conn.close()

    reopened = core_db.connect(path)
    try:
        core_db.initialize(reopened)
        assert core_db._v56_generation_bindings_present(
            reopened, allow_v57=True,
        )
        assert core_db._v57_material_bindings_present(reopened)
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "damage",
    ["lookalike_publication", "missing_input_table", "duplicate_index",
     "descending_index", "unknown_trigger", "missing_parent",
     "missing_whole_domain"],
)
def test_reopen_rejects_v55_lookalike_tables_indexes_and_triggers(tmp_path, damage):
    path = tmp_path / f"{damage}.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    if damage == "lookalike_publication":
        conn.execute("DROP TABLE aggregation_publication_state")
        conn.execute("CREATE TABLE aggregation_publication_state(foo TEXT)")
    elif damage == "missing_input_table":
        conn.execute("DROP TABLE aggregation_node_input_sources")
        conn.execute("DROP TABLE aggregation_node_inputs")
    elif damage == "duplicate_index":
        conn.execute("CREATE INDEX v55_duplicate ON aggregation_nodes(id)")
    elif damage == "descending_index":
        conn.execute("DROP INDEX idx_aggregation_input_source_occurrence")
        conn.execute(
            "CREATE INDEX idx_aggregation_input_source_occurrence ON "
            "aggregation_node_input_sources(source_session_id COLLATE NOCASE DESC,"
            "source_message_id)"
        )
    elif damage == "unknown_trigger":
        conn.execute(
            "CREATE TRIGGER arbitrary_v55_trigger AFTER INSERT "
            "ON aggregation_publication_state BEGIN SELECT 1; END"
        )
    elif damage == "missing_parent":
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DROP TABLE aggregation_nodes")
    else:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DROP TABLE aggregation_node_input_sources")
        conn.execute("DROP TABLE aggregation_node_inputs")
        conn.execute("DROP TABLE aggregation_publication_state")
        conn.execute("DROP TABLE aggregation_nodes")
    conn.close()

    reopened = core_db.connect(path)
    try:
        with pytest.raises(RuntimeError, match="schema v56 aggregation generation"):
            core_db.initialize(reopened)
    finally:
        reopened.close()
