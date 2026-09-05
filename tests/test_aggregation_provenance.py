"""Adversarial contracts for exact episode/RAPTOR source provenance."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest

from hymem import HyMem, HyMemConfig, StubEmbeddingClient, portability
from hymem.core import db as core_db
from hymem.core.vectors import encode_vector
from hymem.dreaming import aggregate as aggregate_mod
from hymem.dreaming import episodes as episodes_mod
from hymem.dreaming.aggregate import build_aggregation_nodes
from hymem.dreaming.aggregation_provenance import (
    AGGREGATION_SOURCE_MANIFEST_VERSION,
    combine_source_occurrences,
    load_aggregation_source_manifest,
    load_episode_source_manifest,
    persist_aggregation_source_manifest,
    persist_episode_source_manifest,
    unpublish_episode_source_manifest,
)
from hymem.dreaming.episodes import (
    EpisodesExtraction,
    persist_episodes,
    validate_episode_items,
)
from hymem.dreaming.lossless import (
    LOSSLESS_COVERAGE_VERSION,
    coverage_chunk_id,
    materialize_message_coverage,
)
from hymem.extraction.embeddings import embedding_text_hash
from hymem.extraction.llm import LLMRequest, StubLLMClient
from hymem.query.augment import (
    AugmentedContext,
    _aggregation_search,
)
from hymem.query.fusion import enrich_context_provenance, scope_context_in_place


_FUSION = json.dumps({
    "title": "Needle synthesis",
    "summary": "needle links the exact source episodes",
})
_ROOT = json.dumps({"title": "Root", "summary": "Exact standing digest."})


def _rewrite_portable_wire(path, mutate) -> None:
    """Rewrite checksummed JSONL after an intentional adversarial mutation."""

    objects = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert objects[-1]["type"] == "_end"
    body, end = objects[:-1], objects[-1]
    mutate(body)
    end["counts"] = {
        kind: sum(item.get("type") == kind for item in body)
        for kind in end["counts"]
    }
    encoded = [json.dumps(item, ensure_ascii=False) + "\n" for item in body]
    end["sha256"] = hashlib.sha256(
        "".join(encoded).encode("utf-8")
    ).hexdigest()
    path.write_text(
        "".join(encoded) + json.dumps(end, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _downgrade_portable_v9_to_v8(path) -> None:
    """Produce a structurally exact v8 wire with unattributed episodes."""

    objects = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    body: list[dict] = []
    for item in objects[:-1]:
        if item["type"] == "_meta":
            item["version"] = 8
            item["schema_version"] = 44
            body.append(item)
            continue
        kind = item["type"]
        if kind not in portability._V8_TABLE_BY_KIND:
            continue
        item["record"] = {
            column: item["record"][column]
            for column in portability._V8_COLS_BY_KIND[kind]
        }
        body.append(item)
    encoded = [json.dumps(item, ensure_ascii=False) + "\n" for item in body]
    end = {
        "type": "_end",
        "counts": {
            kind: sum(item.get("type") == kind for item in body)
            for kind in portability._V8_TABLE_BY_KIND
        },
        "sha256": hashlib.sha256(
            "".join(encoded).encode("utf-8")
        ).hexdigest(),
    }
    path.write_text(
        "".join(encoded) + json.dumps(end, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _fusion_llm() -> StubLLMClient:
    return StubLLMClient(
        fixtures={
            "fuse several related episodes": _FUSION,
            "combined summary that loses no thread": _FUSION,
            "standing digest of everything known": _ROOT,
        },
        default="[]",
    )


def _aggregation_cfg(cfg: HyMemConfig, *, digest: bool = False) -> HyMemConfig:
    return replace(
        cfg,
        aggregation_nodes_enabled=True,
        aggregation_digest_enabled=digest,
        aggregation_blocking_top_k=0,
    )


def _seed_native_episode(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    title: str,
    summary: str,
    entity: str,
) -> tuple[str, str, int]:
    with core_db.transaction(conn):
        conn.execute("INSERT INTO sessions(id) VALUES (?)", (session_id,))
        message_id = int(conn.execute(
            "INSERT INTO messages(session_id,role,content) VALUES (?,'user',?)",
            (session_id, f"{summary} source"),
        ).lastrowid)
        materialize_message_coverage(conn, session_id)
        chunk_id = coverage_chunk_id(session_id, message_id)
        persist_episodes(
            conn,
            session_id,
            EpisodesExtraction(items=[{
                "title": title,
                "summary": summary,
                "outcome": "informational",
                "key_entities": [entity],
                "chunk_ids": [chunk_id],
            }]),
        )
    episode_id = conn.execute(
        "SELECT id FROM episodes WHERE session_id=?", (session_id,)
    ).fetchone()[0]
    return str(episode_id), chunk_id, message_id


def _seed_external_episode(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    peer_id: str,
    workspace_id: str,
    entity: str,
) -> str:
    with core_db.transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO peers(id,workspace_id,role,metadata) "
            "VALUES (?,?,'user','{}')",
            (peer_id, workspace_id),
        )
        conn.execute(
            "INSERT INTO sessions(id,source_workspace_id) VALUES (?,?)",
            (session_id, workspace_id),
        )
        conn.execute(
            "INSERT INTO session_peers(session_id,workspace_id,peer_id) "
            "VALUES (?,?,?)",
            (session_id, workspace_id, peer_id),
        )
        message_id = int(conn.execute(
            "INSERT INTO messages(session_id,role,source_peer_id,"
            "source_workspace_id,content) VALUES (?,'user',?,?,?)",
            (session_id, peer_id, workspace_id, f"needle {entity} source"),
        ).lastrowid)
        materialize_message_coverage(conn, session_id)
        persist_episodes(
            conn,
            session_id,
            EpisodesExtraction(items=[{
                "title": f"{entity} thread",
                "summary": f"needle notes for {entity}",
                "outcome": "informational",
                "key_entities": [entity],
                "chunk_ids": [coverage_chunk_id(session_id, message_id)],
            }]),
        )
    return str(conn.execute(
        "SELECT id FROM episodes WHERE session_id=?", (session_id,)
    ).fetchone()[0])


def test_episode_manifest_is_canonical_exact_replay_and_retention_safe(cfg):
    hy = HyMem(cfg)
    try:
        with core_db.transaction(hy.conn):
            hy.conn.execute("INSERT INTO sessions(id) VALUES ('ordered')")
            message_ids = [
                int(hy.conn.execute(
                    "INSERT INTO messages(session_id,role,content) "
                    "VALUES ('ordered','user',?)",
                    (f"turn {index}",),
                ).lastrowid)
                for index in range(3)
            ]
            materialize_message_coverage(hy.conn, "ordered")
            cited = [
                coverage_chunk_id("ordered", message_ids[2]),
                coverage_chunk_id("ordered", message_ids[0]),
                coverage_chunk_id("ordered", message_ids[2]),
            ]
            persist_episodes(
                hy.conn,
                "ordered",
                EpisodesExtraction(items=[{
                    "title": "Sparse episode",
                    "summary": "Only the endpoint turns are cited.",
                    "key_entities": ["sparse"],
                    "chunk_ids": cited,
                }]),
            )
        episode = hy.conn.execute(
            "SELECT rowid,* FROM episodes WHERE session_id='ordered'"
        ).fetchone()
        sources = load_episode_source_manifest(hy.conn, episode["id"])
        assert sources is not None
        assert [source.message_id for source in sources] == [
            message_ids[0], message_ids[2]
        ]
        with core_db.transaction(hy.conn):
            persist_episode_source_manifest(
                hy.conn,
                episode["id"],
                (sources[1], sources[0], sources[1]),
            )
        assert load_episode_source_manifest(hy.conn, episode["id"]) == sources
        original = (episode["rowid"], episode["source_manifest_hash"])

        # Equivalent citations in a different model order are an exact replay,
        # not a new source identity or a new episode row.
        with core_db.transaction(hy.conn):
            persist_episodes(
                hy.conn,
                "ordered",
                EpisodesExtraction(items=[{
                    "title": "Sparse episode",
                    "summary": "Only the endpoint turns are cited.",
                    "key_entities": ["sparse"],
                    "chunk_ids": list(reversed(cited)),
                }]),
            )
        replay = hy.conn.execute(
            "SELECT rowid,* FROM episodes WHERE id=?", (episode["id"],)
        ).fetchone()
        assert (replay["rowid"], replay["source_manifest_hash"]) == original
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM episode_source_occurrences WHERE episode_id=?",
            (episode["id"],),
        ).fetchone()[0] == 2

        # Proof is bound to the lossless artifact, not continued raw retention.
        hy.conn.execute("DELETE FROM messages WHERE session_id='ordered'")
        retained = load_episode_source_manifest(hy.conn, episode["id"])
        assert retained is not None
        assert [source.message_id for source in retained] == [
            message_ids[0], message_ids[2]
        ]
    finally:
        hy.close()


def test_extraction_chunk_manifest_resolves_skipped_ids_and_corruption_rolls_back(cfg):
    hy = HyMem(cfg)
    try:
        with core_db.transaction(hy.conn):
            hy.conn.execute("INSERT INTO sessions(id) VALUES ('extraction-proof')")
            message_ids = [
                int(hy.conn.execute(
                    "INSERT INTO messages(session_id,role,content) "
                    "VALUES ('extraction-proof','user',?)",
                    (f"source {index}",),
                ).lastrowid)
                for index in range(3)
            ]
            materialize_message_coverage(hy.conn, "extraction-proof")
            chunk_id = "exact-extraction-chunk"
            hy.conn.execute(
                "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
                "salience_reason,text,chunk_kind) VALUES (?,?,?,?,'test',?,'extraction')",
                (
                    chunk_id,
                    "extraction-proof",
                    message_ids[0],
                    message_ids[2],
                    "user: source 0\nuser: source 2",
                ),
            )
            for ordinal, message_id in enumerate((message_ids[0], message_ids[2])):
                hy.conn.execute(
                    "INSERT INTO chunk_message_sources(chunk_id,ordinal,"
                    "source_message_id,source_session_id,source_coverage_chunk_id,"
                    "source_coverage_version) VALUES (?,?,?,?,?,?)",
                    (
                        chunk_id,
                        ordinal,
                        message_id,
                        "extraction-proof",
                        coverage_chunk_id("extraction-proof", message_id),
                        LOSSLESS_COVERAGE_VERSION,
                    ),
                )
            hy.conn.execute(
                "UPDATE chunks SET source_manifest_version='claim-source-manifest-v1',"
                "source_manifest_count=2 WHERE id=?",
                (chunk_id,),
            )
            persist_episodes(
                hy.conn,
                "extraction-proof",
                EpisodesExtraction(items=[{
                    "title": "Exact extraction",
                    "summary": "Skipped middle message intentionally.",
                    "chunk_ids": [chunk_id],
                }]),
            )
        episode = hy.conn.execute(
            "SELECT id FROM episodes WHERE session_id='extraction-proof'"
        ).fetchone()[0]
        assert [source.message_id for source in (
            load_episode_source_manifest(hy.conn, episode) or ()
        )] == [message_ids[0], message_ids[2]]

        # A claimed non-empty manifest whose proof rows were damaged on disk is
        # not a legacy empty citation and may not degrade to numeric-range
        # inference.  Build a valid published manifest first, then emulate
        # corruption by bypassing only the normal immutability trigger.
        with core_db.transaction(hy.conn):
            hy.conn.execute(
                "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
                "salience_reason,text,chunk_kind) VALUES ('corrupt-extraction',"
                "?,?,?,'test',?,'extraction')",
                (
                    "extraction-proof",
                    message_ids[0],
                    message_ids[2],
                    "user: source 0\nuser: source 1\nuser: source 2",
                ),
            )
            for ordinal, message_id in enumerate(message_ids):
                hy.conn.execute(
                    "INSERT INTO chunk_message_sources(chunk_id,ordinal,"
                    "source_message_id,source_session_id,source_coverage_chunk_id,"
                    "source_coverage_version) VALUES ('corrupt-extraction',?,?,?,?,?)",
                    (
                        ordinal,
                        message_id,
                        "extraction-proof",
                        coverage_chunk_id("extraction-proof", message_id),
                        LOSSLESS_COVERAGE_VERSION,
                    ),
                )
            hy.conn.execute(
                "UPDATE chunks SET source_manifest_version='claim-source-manifest-v1',"
                "source_manifest_count=3 WHERE id='corrupt-extraction'"
            )
            hy.conn.execute("DROP TRIGGER chunk_message_sources_delete_guard")
            hy.conn.execute(
                "DELETE FROM chunk_message_sources "
                "WHERE chunk_id='corrupt-extraction' AND ordinal=1"
            )
        before = hy.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        with pytest.raises(ValueError, match="corrupt manifest"):
            with core_db.transaction(hy.conn):
                persist_episodes(
                    hy.conn,
                    "extraction-proof",
                    EpisodesExtraction(items=[{
                        "title": "Must roll back",
                        "summary": "No exact sources exist.",
                        "chunk_ids": ["corrupt-extraction"],
                    }]),
                )
        assert hy.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == before
    finally:
        hy.close()


def test_episode_rewrite_rekeys_fusion_and_forecast_without_stale_rows(cfg):
    hy = HyMem(cfg)
    try:
        ep1, chunk1, _ = _seed_native_episode(
            hy.conn, "rewrite-a", title="Shared", summary="version one", entity="thread"
        )
        _seed_native_episode(
            hy.conn, "rewrite-b", title="Shared", summary="stable", entity="thread"
        )
        llm = _fusion_llm()
        enabled = _aggregation_cfg(cfg)
        first = build_aggregation_nodes(hy.conn, enabled, llm)
        first_node = hy.conn.execute(
            "SELECT id,input_fingerprint FROM aggregation_nodes WHERE level=0"
        ).fetchone()
        assert first.nodes == 1
        assert load_aggregation_source_manifest(hy.conn, first_node["id"])

        steady = build_aggregation_nodes(hy.conn, enabled, llm)
        assert steady.reused == 1
        sources = load_aggregation_source_manifest(hy.conn, first_node["id"])
        assert sources is not None
        with core_db.transaction(hy.conn):
            persist_aggregation_source_manifest(
                hy.conn,
                first_node["id"],
                occurrences=(sources[1], sources[0], sources[1]),
                input_fingerprint=first_node["input_fingerprint"],
            )
        assert load_aggregation_source_manifest(hy.conn, first_node["id"]) == sources

        episode_rowid = hy.conn.execute(
            "SELECT rowid FROM episodes WHERE id=?", (ep1,)
        ).fetchone()[0]
        with core_db.transaction(hy.conn):
            persist_episodes(
                hy.conn,
                "rewrite-a",
                EpisodesExtraction(items=[{
                    "title": "Shared",
                    "summary": "version two",
                    "key_entities": ["thread"],
                    "chunk_ids": [chunk1],
                }]),
            )
        assert hy.conn.execute(
            "SELECT rowid FROM episodes WHERE id=?", (ep1,)
        ).fetchone()[0] == episode_rowid

        rebuilt = build_aggregation_nodes(hy.conn, enabled, llm)
        second_node = hy.conn.execute(
            "SELECT id FROM aggregation_nodes WHERE level=0"
        ).fetchone()[0]
        assert second_node != first_node["id"]
        assert rebuilt.nodes - rebuilt.reused == rebuilt.predicted_rebuild
        assert rebuilt.keying_residual == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM aggregation_node_source_occurrences "
            "WHERE node_id=?",
            (first_node["id"],),
        ).fetchone()[0] == 0
        assert len([
            call for call in llm.calls
            if "fuse several related episodes" in call.system
        ]) == 2
    finally:
        hy.close()


def test_prompt_membership_never_exceeds_persisted_membership(cfg):
    hy = HyMem(cfg)
    try:
        for index in range(5):
            _seed_native_episode(
                hy.conn,
                f"cap-{index}",
                title=f"Episode {index}",
                summary=f"summary {index}",
                entity="one-component",
            )
        llm = _fusion_llm()
        enabled = replace(
            _aggregation_cfg(cfg),
            aggregation_max_cluster_size=5,
            aggregation_max_members=2,
        )
        build_aggregation_nodes(hy.conn, enabled, llm)
        persisted_sizes = sorted(
            row["n_members"] for row in hy.conn.execute(
                "SELECT n_members FROM aggregation_nodes WHERE level=0"
            ).fetchall()
        )
        prompt_sizes = sorted(
            call.user.count("\n\n---\n\n") + 1
            for call in llm.calls
            if "fuse several related episodes" in call.system
        )
        assert persisted_sizes
        assert persisted_sizes == prompt_sizes
        assert max(persisted_sizes) <= 2
    finally:
        hy.close()


def test_root_with_unsourced_anchor_is_explicitly_incomplete(cfg, monkeypatch):
    hy = HyMem(cfg)
    try:
        _seed_native_episode(
            hy.conn, "root-a", title="A", summary="a", entity="root-thread"
        )
        _seed_native_episode(
            hy.conn, "root-b", title="B", summary="b", entity="root-thread"
        )
        monkeypatch.setattr(
            aggregate_mod, "_anchor_facts", lambda _conn, _cap: ["verified anchor"]
        )
        build_aggregation_nodes(
            hy.conn, _aggregation_cfg(cfg, digest=True), _fusion_llm()
        )
        root = hy.conn.execute(
            "SELECT id,source_manifest_complete,source_manifest_count "
            "FROM aggregation_nodes WHERE is_root=1"
        ).fetchone()
        assert root is not None
        assert (root["source_manifest_complete"], root["source_manifest_count"]) == (0, 0)
        assert load_aggregation_source_manifest(hy.conn, root["id"]) is None
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM aggregation_node_source_occurrences WHERE node_id=?",
            (root["id"],),
        ).fetchone()[0] == 0
    finally:
        hy.close()


def test_exact_sources_propagate_through_rollups_and_incomplete_child_quarantines_ancestors(
    cfg, monkeypatch
):
    hy = HyMem(cfg)
    try:
        episode_ids: list[str] = []
        # Four independent level-0 clusters force an interior rollup at fan-in
        # two, while every leaf retains one exact occurrence.
        for cluster in range(4):
            for member in range(2):
                episode_id, _, _ = _seed_native_episode(
                    hy.conn,
                    f"tree-{cluster}-{member}",
                    title=f"Tree {cluster}/{member}",
                    summary=f"cluster {cluster} member {member}",
                    entity=f"cluster-{cluster}",
                )
                episode_ids.append(episode_id)
        monkeypatch.setattr(aggregate_mod, "_anchor_facts", lambda _conn, _cap: [])
        enabled = replace(
            _aggregation_cfg(cfg, digest=True),
            aggregation_max_members=2,
            aggregation_max_cluster_size=2,
            aggregation_digest_anchor_facts=0,
        )
        build_aggregation_nodes(hy.conn, enabled, _fusion_llm())

        def descendant_episode_ids(node_id: str, seen: frozenset[str] = frozenset()):
            assert node_id not in seen
            row = hy.conn.execute(
                "SELECT member_episode_ids FROM aggregation_nodes WHERE id=?",
                (node_id,),
            ).fetchone()
            assert row is not None
            result: list[str] = []
            for member_id in json.loads(row["member_episode_ids"]):
                child = hy.conn.execute(
                    "SELECT 1 FROM aggregation_nodes WHERE id=?", (member_id,)
                ).fetchone()
                if child is None:
                    result.append(member_id)
                else:
                    result.extend(descendant_episode_ids(member_id, seen | {node_id}))
            return result

        nodes = hy.conn.execute(
            "SELECT id,level,is_root FROM aggregation_nodes ORDER BY level,id"
        ).fetchall()
        assert max(row["level"] for row in nodes) >= 2
        for node in nodes:
            manifest = load_aggregation_source_manifest(hy.conn, node["id"])
            assert manifest is not None
            expected = combine_source_occurrences(
                load_episode_source_manifest(hy.conn, episode_id) or ()
                for episode_id in descendant_episode_ids(node["id"])
            )
            assert manifest == expected

        # Quarantine one leaf and rebuild. Every node containing that leaf, and
        # therefore the whole-store root, must become explicitly incomplete;
        # unaffected sibling subtrees stay complete.
        incomplete_episode = episode_ids[0]
        with core_db.transaction(hy.conn):
            unpublish_episode_source_manifest(hy.conn, incomplete_episode)
        build_aggregation_nodes(hy.conn, enabled, _fusion_llm())
        rebuilt_nodes = hy.conn.execute(
            "SELECT id,is_root,source_manifest_complete FROM aggregation_nodes"
        ).fetchall()
        saw_unaffected = False
        for node in rebuilt_nodes:
            descendants = descendant_episode_ids(node["id"])
            affected = incomplete_episode in descendants
            if affected:
                assert node["source_manifest_complete"] == 0
                assert load_aggregation_source_manifest(hy.conn, node["id"]) is None
            else:
                saw_unaffected = True
                assert node["source_manifest_complete"] == 1
        assert saw_unaffected
        root = next(row for row in rebuilt_nodes if row["is_root"] == 1)
        assert root["source_manifest_complete"] == 0
    finally:
        hy.close()


def test_scoped_search_validates_before_fts_and_vector_limits(cfg):
    embedder = StubEmbeddingClient()
    hy = HyMem(cfg, embedding_client=embedder)
    try:
        # One admissible node: two sessions, but every exact source belongs to
        # the same external peer/workspace.
        for sid in ("valid-1", "valid-2"):
            _seed_external_episode(
                hy.conn, sid, peer_id="peer-a", workspace_id="workspace-a",
                entity="valid-topic",
            )
        # Same workspace but mixed peers; peer-scoped retrieval must reject it.
        _seed_external_episode(
            hy.conn, "peer-mix-1", peer_id="peer-a", workspace_id="workspace-a",
            entity="peer-mix",
        )
        _seed_external_episode(
            hy.conn, "peer-mix-2", peer_id="peer-b", workspace_id="workspace-a",
            entity="peer-mix",
        )
        # Mixed workspaces must fail a workspace boundary as well.
        _seed_external_episode(
            hy.conn, "workspace-mix-1", peer_id="peer-a",
            workspace_id="workspace-a", entity="workspace-mix",
        )
        _seed_external_episode(
            hy.conn, "workspace-mix-2", peer_id="peer-z",
            workspace_id="workspace-z", entity="workspace-mix",
        )
        build_aggregation_nodes(
            hy.conn, _aggregation_cfg(cfg), _fusion_llm(), embedder
        )
        nodes = hy.conn.execute(
            "SELECT id FROM aggregation_nodes WHERE level=0"
        ).fetchall()
        valid_node = next(
            row["id"] for row in nodes
            if {
                source.source_peer_id
                for source in load_aggregation_source_manifest(hy.conn, row["id"]) or ()
            } == {"peer-a"}
            and {
                source.source_workspace_id
                for source in load_aggregation_source_manifest(hy.conn, row["id"]) or ()
            } == {"workspace-a"}
        )

        # A corrupt/incomplete prefix larger than one validation page used to
        # consume both the FTS LIMIT and vector max_scan before proof checks.
        bad_text = "needle needle needle invalid composite"
        bad_vector = embedder.embed([bad_text])[0]
        with core_db.transaction(hy.conn):
            for index in range(40):
                node_id = f"aaa-invalid-{index:02d}"
                hy.conn.execute(
                    "INSERT INTO aggregation_nodes(id,title,summary,"
                    "member_episode_ids,session_ids,n_members,n_sessions,"
                    "created_at,level,is_root,input_fingerprint) "
                    "VALUES (?,?,?, '[\"legacy\"]','[\"legacy\"]',1,1,"
                    "'9999-01-01 00:00:00',0,0,?)",
                    (
                        node_id,
                        "needle needle needle",
                        "invalid composite",
                        "sha256:" + ("0" * 64),
                    ),
                )
                hy.conn.execute(
                    "INSERT INTO aggregation_node_embeddings(node_id,vector_json,"
                    "model,dim,text_hash) VALUES (?,?,?,?,?)",
                    (
                        node_id,
                        encode_vector(bad_vector),
                        embedder.model,
                        embedder.dim,
                        embedding_text_hash(bad_text),
                    ),
                )

        query_vector = embedder.embed(["needle"])[0]
        hits = _aggregation_search(
            hy.conn,
            "needle",
            top_k=1,
            embedding_client=embedder,
            max_scan=1,
            query_vector=query_vector,
            source_peer_id="peer-a",
            source_workspace_id="workspace-a",
        )
        assert [hit.node_id for hit in hits] == [valid_node]
        assert hits[0].source_provenance_complete is True
        assert all(
            source.source_peer_id == "peer-a"
            and source.source_workspace_id == "workspace-a"
            for source in hits[0].source_occurrences
        )
        assert _aggregation_search(
            hy.conn,
            "needle",
            top_k=1,
            source_session_id="valid-1",
        ) == []

        # Pin the public caller path too: scoped requests used to suppress the
        # entire aggregation tier before `_aggregation_search` ran.  An
        # explicit TR ability guarantees the tier fires, and the final fused
        # view may contain provenance for the one admissible node only.
        public = hy.augment(
            "needle",
            ability="TR",
            source_peer_id="peer-a",
            source_workspace_id="workspace-a",
        )
        assert [node.node_id for node in public.aggregation_nodes] == [valid_node]
        aggregation_artifacts = {
            provenance.artifact_id
            for evidence in public.fused_evidence
            for provenance in evidence.provenance
            if provenance.tier == "aggregation"
        }
        assert aggregation_artifacts == {valid_node}

        # Final enrichment is a second fail-closed boundary: if proof changes
        # after native retrieval, the mutable DTO is removed before fusion.
        hy.conn.execute(
            "UPDATE aggregation_nodes SET source_manifest_hash=? WHERE id=?",
            ("sha256:" + ("f" * 64), valid_node),
        )
        context = AugmentedContext(aggregation_nodes=hits)
        enrich_context_provenance(hy.conn, context)
        assert context.aggregation_nodes[0].source_provenance_complete is False
        scope_context_in_place(
            context,
            source_session_id=None,
            source_peer_id="peer-a",
            source_workspace_id="workspace-a",
        )
        assert context.aggregation_nodes == []
    finally:
        hy.close()


def test_sql_publication_guards_reject_malformed_shapes(cfg):
    hy = HyMem(cfg)
    try:
        episode_id, _, _ = _seed_native_episode(
            hy.conn, "guard-a", title="A", summary="a", entity="guard"
        )
        _seed_native_episode(
            hy.conn, "guard-b", title="B", summary="b", entity="guard"
        )
        build_aggregation_nodes(hy.conn, _aggregation_cfg(cfg), _fusion_llm())
        node = hy.conn.execute(
            "SELECT id FROM aggregation_nodes WHERE level=0"
        ).fetchone()[0]

        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute(
                "UPDATE episodes SET source_manifest_count=99 WHERE id=?",
                (episode_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute(
                "UPDATE episode_source_occurrences SET ordinal=7 "
                "WHERE episode_id=? AND ordinal=0",
                (episode_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute(
                "UPDATE aggregation_nodes SET input_fingerprint=? WHERE id=?",
                ("sha256:" + ("1" * 64), node),
            )
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute(
                "INSERT INTO aggregation_nodes(id,title,summary,"
                "source_manifest_version,source_manifest_count,"
                "source_manifest_hash,source_manifest_complete,input_fingerprint) "
                "VALUES ('bad-shape','bad','bad',?,1,?,1,?)",
                (
                    AGGREGATION_SOURCE_MANIFEST_VERSION,
                    "sha256:" + ("2" * 64),
                    "sha256:" + ("3" * 64),
                ),
            )
    finally:
        hy.close()


class _DigestEpisodeLLM:
    def __init__(self, chunk_ids: list[str] | str):
        self.chunk_ids = [chunk_ids] if isinstance(chunk_ids, str) else chunk_ids

    def complete(self, request: LLMRequest) -> str:
        if request.system.startswith((
            "You analyze one conversation session",
            "You re-read one conversation session",
        )):
            return json.dumps({
                "episodes": [{
                    "title": "Transactional episode",
                    "summary": "This cites a non-empty exact source.",
                    "outcome": "informational",
                    "key_entities": [],
                    "chunk_ids": self.chunk_ids,
                }],
                "summary": "A bounded summary.",
                "procedures": [],
            })
        return "[]"


def test_digest_proof_failure_rolls_back_episode_and_watermark(cfg, monkeypatch):
    hy = HyMem(replace(cfg, profile_extraction_enabled=False), llm=StubLLMClient(default="[]"))
    try:
        message_id = hy.log_message(
            "rollback", "user", "enough material for a transactional digest episode"
        )
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "rollback")
        hy.set_llm(_DigestEpisodeLLM(coverage_chunk_id("rollback", message_id)))

        original = episodes_mod.resolve_cited_episode_sources

        def corrupt_proof(conn, session_id, chunk_ids):
            assert chunk_ids
            # Resolve first so this is specifically a proof failure after a
            # valid, non-empty citation—not a parser/empty-output path.
            assert original(conn, session_id, chunk_ids)
            raise ValueError("transient corrupt episode coverage proof")

        monkeypatch.setattr(
            episodes_mod, "resolve_cited_episode_sources", corrupt_proof
        )
        with pytest.raises(ValueError, match="corrupt episode coverage"):
            hy.dream(session_ids=["rollback"])
        state = hy.conn.execute(
            "SELECT digest_cursor_message_id,digested_message_id,auto_summary "
            "FROM sessions WHERE id='rollback'"
        ).fetchone()
        assert tuple(state) == (None, None, None)
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE session_id='rollback'"
        ).fetchone()[0] == 0
    finally:
        hy.close()


def test_nonempty_invalid_citation_cannot_normalize_to_unattributed_or_advance(cfg):
    hy = HyMem(
        replace(cfg, profile_extraction_enabled=False),
        llm=StubLLMClient(default="[]"),
    )
    try:
        message_id = hy.log_message(
            "invalid-citation", "user", "source for a mixed citation response"
        )
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "invalid-citation")
        valid = coverage_chunk_id("invalid-citation", message_id)

        cleaned = validate_episode_items(
            [{
                "title": "Mixed citation",
                "summary": "One source is hallucinated.",
                "chunk_ids": [valid, "msgcov_hallucinated"],
            }],
            {valid},
        )
        assert cleaned[0]["chunk_ids"] == [valid]
        assert cleaned[0]["_source_citations_invalid"] is True
        with pytest.raises(ValueError, match="invalid non-empty source citation"):
            with core_db.transaction(hy.conn):
                persist_episodes(
                    hy.conn, "invalid-citation", EpisodesExtraction(cleaned)
                )

        # The integrated digest validator also holds the cursor before the
        # persistence boundary when that exact mixed list arrives on the wire.
        hy.set_llm(_DigestEpisodeLLM([valid, "msgcov_hallucinated"]))
        report = hy.dream(session_ids=["invalid-citation"])
        assert report.digest_failures == 1
        state = hy.conn.execute(
            "SELECT digest_cursor_message_id,digested_message_id "
            "FROM sessions WHERE id='invalid-citation'"
        ).fetchone()
        assert tuple(state) == (None, None)
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE session_id='invalid-citation'"
        ).fetchone()[0] == 0
    finally:
        hy.close()


def test_v45_sparse_domain_skips_safely_and_stamps(tmp_path):
    conn = core_db.connect(tmp_path / "sparse-v44.sqlite")
    try:
        conn.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT)")
        conn.execute(
            "INSERT INTO schema_meta(key,value) VALUES ('schema_version','44')"
        )
        # A misleadingly named but structurally insufficient coverage table
        # must not make v45 execute cross-table ALTER/TRIGGER statements.
        conn.execute(
            "CREATE TABLE message_retention_coverage(message_id INTEGER)"
        )
        core_db._run_migrations(conn)
        assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION == 46
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='episode_source_occurrences'"
        ).fetchone() is None
    finally:
        conn.close()


def test_v9_episode_proof_roundtrip_rebuilds_cache_and_exact_reimport_keeps_it(
    tmp_path,
):
    source = HyMem(HyMemConfig(root=tmp_path / "portable-source"))
    wire = tmp_path / "episode-proof-v9.jsonl"
    try:
        _seed_native_episode(
            source.conn,
            "portable-a",
            title="Portable A",
            summary="needle portable alpha",
            entity="portable-thread",
        )
        _seed_native_episode(
            source.conn,
            "portable-b",
            title="Portable B",
            summary="needle portable beta",
            entity="portable-thread",
        )
        build_aggregation_nodes(
            source.conn, _aggregation_cfg(source.config), _fusion_llm()
        )
        assert source.conn.execute(
            "SELECT COUNT(*) FROM aggregation_nodes"
        ).fetchone()[0] > 0
        source.conn.execute("DELETE FROM messages")
        assert source.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        counts = source.export(wire)
        assert counts["episode"] == 2
        assert counts["episode_source_occurrence"] == 2
        assert "aggregation_node" not in counts
    finally:
        source.close()

    target = HyMem(HyMemConfig(root=tmp_path / "portable-target"))
    try:
        # The cache is not portable. Any episode/proof insertion invalidates a
        # pre-existing local tree so it cannot survive with stale membership.
        target.conn.execute(
            "INSERT INTO aggregation_nodes(id,title,summary,input_fingerprint) "
            "VALUES ('stale-cache','stale','stale',?)",
            ("sha256:" + ("0" * 64),),
        )
        imported = target.import_(wire)
        assert imported["episode"] == 2
        assert imported["episode_source_occurrence"] == 2
        assert target.conn.execute(
            "SELECT COUNT(*) FROM aggregation_nodes"
        ).fetchone()[0] == 0
        episode_ids = [
            row["id"] for row in target.conn.execute(
                "SELECT id FROM episodes ORDER BY id"
            ).fetchall()
        ]
        assert episode_ids
        assert all(
            load_episode_source_manifest(target.conn, episode_id) is not None
            for episode_id in episode_ids
        )

        build_aggregation_nodes(
            target.conn, _aggregation_cfg(target.config), _fusion_llm()
        )
        rebuilt_ids = {
            row["id"] for row in target.conn.execute(
                "SELECT id FROM aggregation_nodes"
            ).fetchall()
        }
        assert rebuilt_ids
        assert sum(target.import_(wire).values()) == 0
        assert {
            row["id"] for row in target.conn.execute(
                "SELECT id FROM aggregation_nodes"
            ).fetchall()
        } == rebuilt_ids
    finally:
        target.close()


def test_v9_redacted_episode_proof_rehashes_coverage_occurrence_and_header(tmp_path):
    secret = "sk-ABCD1234efgh5678ijkl"
    source = HyMem(HyMemConfig(
        root=tmp_path / "redacted-source", redact_secrets=False
    ))
    wire = tmp_path / "redacted-source.jsonl"
    reexport = tmp_path / "redacted-target.jsonl"
    try:
        with core_db.transaction(source.conn):
            source.conn.execute("INSERT INTO sessions(id) VALUES ('redacted-proof')")
            message_id = int(source.conn.execute(
                "INSERT INTO messages(session_id,role,content) "
                "VALUES ('redacted-proof','user',?)",
                (f"The deployment token is {secret}",),
            ).lastrowid)
            materialize_message_coverage(source.conn, "redacted-proof")
            persist_episodes(
                source.conn,
                "redacted-proof",
                EpisodesExtraction(items=[{
                    "title": "Secret-bearing source",
                    "summary": f"The deployment token was {secret}.",
                    "chunk_ids": [coverage_chunk_id("redacted-proof", message_id)],
                }]),
            )
        original = source.conn.execute(
            "SELECT id,source_manifest_hash FROM episodes"
        ).fetchone()
        original_coverage_hash = source.conn.execute(
            "SELECT message_content_hash FROM message_retention_coverage"
        ).fetchone()[0]
        source.conn.execute("DELETE FROM messages")
        source.export(wire)
        assert secret in wire.read_text(encoding="utf-8")
    finally:
        source.close()

    target = HyMem(HyMemConfig(
        root=tmp_path / "redacted-target", redact_secrets=True
    ))
    try:
        target.import_(wire)
        coverage = target.conn.execute(
            "SELECT message_content_hash FROM message_retention_coverage"
        ).fetchone()[0]
        occurrence = target.conn.execute(
            "SELECT source_content_hash FROM episode_source_occurrences"
        ).fetchone()[0]
        episode = target.conn.execute(
            "SELECT source_manifest_hash,summary FROM episodes WHERE id=?",
            (original["id"],),
        ).fetchone()
        assert coverage == occurrence
        assert coverage != original_coverage_hash
        assert episode["source_manifest_hash"] != original["source_manifest_hash"]
        assert secret not in episode["summary"]
        assert load_episode_source_manifest(target.conn, original["id"]) is not None

        target.export(reexport)
        exported = reexport.read_text(encoding="utf-8")
        assert secret not in exported
        assert "[REDACTED-API-KEY]" in exported
    finally:
        target.close()


def test_v8_v9_episode_provenance_merge_is_monotonic(tmp_path):
    source = HyMem(HyMemConfig(root=tmp_path / "monotonic-source"))
    v9 = tmp_path / "monotonic-v9.jsonl"
    v8 = tmp_path / "monotonic-v8.jsonl"
    try:
        _seed_native_episode(
            source.conn,
            "monotonic",
            title="Same episode",
            summary="same portable bytes",
            entity="portable",
        )
        source.export(v9)
        v8.write_bytes(v9.read_bytes())
        _downgrade_portable_v9_to_v8(v8)
    finally:
        source.close()

    target = HyMem(HyMemConfig(root=tmp_path / "monotonic-target"))
    try:
        first = target.import_(v8)
        assert first["episode"] == 1
        row = target.conn.execute(
            "SELECT id,source_manifest_complete FROM episodes"
        ).fetchone()
        assert row["source_manifest_complete"] == 0
        assert load_episode_source_manifest(target.conn, row["id"]) is None

        upgrade = target.import_(v9)
        assert upgrade["episode"] == 0
        assert upgrade["episode_source_occurrence"] == 1
        upgraded = load_episode_source_manifest(target.conn, row["id"])
        assert upgraded is not None and len(upgraded) == 1

        # A later legacy export is weaker evidence and therefore a no-op; it
        # cannot erase, reject, or downgrade already validated exact proof.
        assert sum(target.import_(v8).values()) == 0
        assert load_episode_source_manifest(target.conn, row["id"]) == upgraded
    finally:
        target.close()


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        (
            "boolean-complete",
            lambda body: next(
                item for item in body if item.get("type") == "episode"
            )["record"].__setitem__("source_manifest_complete", True),
        ),
        (
            "wrong-count",
            lambda body: next(
                item for item in body if item.get("type") == "episode"
            )["record"].__setitem__("source_manifest_count", 2),
        ),
        (
            "wrong-hash",
            lambda body: next(
                item for item in body if item.get("type") == "episode"
            )["record"].__setitem__(
                "source_manifest_hash", "sha256:" + ("f" * 64)
            ),
        ),
        (
            "noncontiguous-ordinal",
            lambda body: next(
                item
                for item in body
                if item.get("type") == "episode_source_occurrence"
            )["record"].__setitem__("ordinal", 1),
        ),
        (
            "coverage-hash-mismatch",
            lambda body: next(
                item
                for item in body
                if item.get("type") == "message_retention_coverage"
            )["record"].__setitem__("message_content_hash", "0" * 64),
        ),
    ],
)
def test_v9_forged_episode_proof_fails_before_target_writes(tmp_path, case, mutate):
    source = HyMem(HyMemConfig(root=tmp_path / f"forged-source-{case}"))
    wire = tmp_path / f"forged-{case}.jsonl"
    try:
        _seed_native_episode(
            source.conn,
            "forged-session",
            title="Forged",
            summary="proof must remain exact",
            entity="portable",
        )
        source.export(wire)
    finally:
        source.close()
    _rewrite_portable_wire(wire, mutate)

    target = HyMem(HyMemConfig(root=tmp_path / f"forged-target-{case}"))
    try:
        with pytest.raises(ValueError):
            target.import_(wire)
        assert target.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert target.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
    finally:
        target.close()


def test_v9_conflicting_complete_episode_proof_rejects_atomically(tmp_path):
    def make_source(root, cited_indexes, wire):
        store = HyMem(HyMemConfig(root=root))
        try:
            with core_db.transaction(store.conn):
                store.conn.execute("INSERT INTO sessions(id) VALUES ('proof-conflict')")
                message_ids = [
                    int(store.conn.execute(
                        "INSERT INTO messages(session_id,role,content) "
                        "VALUES ('proof-conflict','user',?)",
                        (f"portable source {index}",),
                    ).lastrowid)
                    for index in range(3)
                ]
                materialize_message_coverage(store.conn, "proof-conflict")
                persist_episodes(
                    store.conn,
                    "proof-conflict",
                    EpisodesExtraction(items=[{
                        "title": "Same bounded episode",
                        "summary": "Same base bytes with competing exact proof.",
                        "chunk_ids": [
                            coverage_chunk_id("proof-conflict", message_ids[index])
                            for index in cited_indexes
                        ],
                    }]),
                )
            store.export(wire)
            return store.conn.execute("SELECT id FROM episodes").fetchone()[0]
        finally:
            store.close()

    left_wire = tmp_path / "proof-left.jsonl"
    right_wire = tmp_path / "proof-right.jsonl"
    left_id = make_source(
        tmp_path / "proof-left", (0, 2), left_wire
    )
    right_id = make_source(
        tmp_path / "proof-right", (0, 1, 2), right_wire
    )
    assert left_id == right_id

    target = HyMem(HyMemConfig(root=tmp_path / "proof-target"))
    try:
        target.import_(left_wire)
        original = load_episode_source_manifest(target.conn, left_id)
        assert original is not None and len(original) == 2
        before_counts = tuple(target.conn.execute(
            "SELECT (SELECT COUNT(*) FROM sessions),"
            "(SELECT COUNT(*) FROM chunks),"
            "(SELECT COUNT(*) FROM episode_source_occurrences)"
        ).fetchone())
        with pytest.raises(ValueError, match="different complete provenance"):
            target.import_(right_wire)
        assert load_episode_source_manifest(target.conn, left_id) == original
        assert tuple(target.conn.execute(
            "SELECT (SELECT COUNT(*) FROM sessions),"
            "(SELECT COUNT(*) FROM chunks),"
            "(SELECT COUNT(*) FROM episode_source_occurrences)"
        ).fetchone()) == before_counts
    finally:
        target.close()
