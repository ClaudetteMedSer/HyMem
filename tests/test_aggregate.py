"""Offline (StubLLM / StubEmbedding, no box) tests for the Phase-2 RAPTOR
cross-session aggregation layer (hymem/dreaming/aggregate.py + the additive
retrieval tier in hymem/query/augment.py).

The pure connected-components clusterer is pinned separately in
test_raptor_cluster_probe.py (the probe re-exports it from this module); here we
cover the build's *policy* (which clusters become nodes), the persisted-node
shape, the full-rebuild semantics, the off-by-default guard, and that retrieval
surfaces nodes ADDITIVELY without displacing episodes.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem.core import db as core_db
from hymem.dreaming.aggregate import (
    build_aggregation_nodes,
    _node_id,
    select_clusters,
)
from hymem.extraction.llm import StubLLMClient
from hymem.query.augment import augment

# The aggregation summary call is keyed on a phrase unique to AGGREGATE_SYSTEM, so
# it never collides with the episode/digest fixtures keyed on the user template.
_NODE_JSON = json.dumps({
    "title": "Postgres across projects",
    "summary": "Both the billing and the analytics project run on Postgres; "
               "the billing DB was later sharded.",
})


def _agg_llm() -> StubLLMClient:
    return StubLLMClient(
        fixtures={"fuse several related episodes": _NODE_JSON}, default="[]"
    )


def _enabled(cfg: HyMemConfig) -> HyMemConfig:
    return replace(cfg, aggregation_nodes_enabled=True)


def _ep(eid: str, sid: str, entities: list[str], vector=None) -> dict:
    return {
        "id": eid, "session_id": sid, "title": eid, "summary": eid,
        "entities": set(entities), "vector": vector,
    }


def _seed_episode(conn, eid, sid, title, summary, entities, start=1, end=2):
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (sid,))
    conn.execute(
        """INSERT INTO episodes(id, session_id, title, summary, participants,
                                start_message_id, end_message_id, outcome, key_entities)
           VALUES (?, ?, ?, ?, '[]', ?, ?, NULL, ?)""",
        (eid, sid, title, summary, start, end, json.dumps(entities)),
    )


@pytest.fixture
def conn(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    yield hy.conn
    hy.close()


# ── build policy: which clusters become nodes (pure, offline) ────────────────

def test_select_clusters_keeps_cross_session_multi_member(cfg):
    eps = [
        _ep("e1", "s1", ["postgres", "billing"]),
        _ep("e2", "s2", ["postgres", "billing"]),   # links e1 by entity jaccard
    ]
    clusters = select_clusters(eps, cfg)
    assert len(clusters) == 1
    assert {m["id"] for m in clusters[0]} == {"e1", "e2"}


def test_select_clusters_drops_single_session_cluster(cfg):
    # Two episodes that DO link, but both live in one session → no cross-session
    # synthesis to aggregate, so the policy drops it.
    eps = [
        _ep("e1", "s1", ["postgres", "billing"]),
        _ep("e2", "s1", ["postgres", "billing"]),
    ]
    assert select_clusters(eps, cfg) == []


def test_select_clusters_drops_singleton(cfg):
    eps = [
        _ep("e1", "s1", ["postgres"]),
        _ep("e2", "s2", ["kafka"]),     # disjoint → two singletons, none kept
    ]
    assert select_clusters(eps, cfg) == []


def test_node_id_stable_for_membership_and_order_independent():
    a = _node_id(["e2", "e1"])
    b = _node_id(["e1", "e2"])
    assert a == b                                  # order-independent
    assert a != _node_id(["e1", "e2", "e3"])       # membership change → new id


# ── DB build + persistence ───────────────────────────────────────────────────

def test_build_creates_node_for_cross_session_cluster(conn, cfg):
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "Set up the billing service on Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics warehouse also runs Postgres.", ["postgres", "billing"])

    built = build_aggregation_nodes(conn, _enabled(cfg), _agg_llm(), embed)
    assert built == 1

    row = conn.execute(
        "SELECT id, title, member_episode_ids, session_ids, n_members, n_sessions "
        "FROM aggregation_nodes"
    ).fetchone()
    assert row["n_members"] == 2
    assert row["n_sessions"] == 2
    assert set(json.loads(row["member_episode_ids"])) == {"e1", "e2"}
    assert set(json.loads(row["session_ids"])) == {"s1", "s2"}
    assert row["title"] == "Postgres across projects"

    # The node summary is embedded (additive retrieval needs the vector).
    emb = conn.execute(
        "SELECT node_id, dim FROM aggregation_node_embeddings"
    ).fetchone()
    assert emb["node_id"] == row["id"]
    assert emb["dim"] == 16


def test_build_is_noop_when_disabled(conn, cfg):
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "t1", "s", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "t2", "s", ["postgres", "billing"])
    built = build_aggregation_nodes(conn, cfg, _agg_llm(), None)  # cfg disabled by default
    assert built == 0
    assert conn.execute("SELECT COUNT(*) AS c FROM aggregation_nodes").fetchone()["c"] == 0


def test_build_skips_single_session(conn, cfg):
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "t1", "s", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s1", "t2", "s", ["postgres", "billing"])
    built = build_aggregation_nodes(conn, _enabled(cfg), _agg_llm(), None)
    assert built == 0
    assert conn.execute("SELECT COUNT(*) AS c FROM aggregation_nodes").fetchone()["c"] == 0


def test_rebuild_replaces_stale_nodes(conn, cfg):
    acfg = _enabled(cfg)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "t1", "s", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "t2", "s", ["postgres", "billing"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), None)
    first_id = conn.execute("SELECT id FROM aggregation_nodes").fetchone()["id"]

    # Add a third member to the cluster → different membership → different node id.
    with core_db.transaction(conn):
        _seed_episode(conn, "e3", "s3", "t3", "s", ["postgres", "billing"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), None)

    rows = conn.execute("SELECT id, n_members FROM aggregation_nodes").fetchall()
    assert len(rows) == 1                       # full rebuild — no stale node lingers
    assert rows[0]["id"] != first_id
    assert rows[0]["n_members"] == 3


# ── additive retrieval tier ──────────────────────────────────────────────────

def test_retrieval_surfaces_nodes_additively(conn, cfg):
    acfg = _enabled(cfg)
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), embed)

    ctx = augment(conn, acfg, "postgres billing", embedding_client=embed)
    assert ctx.aggregation_nodes, "enabled layer should surface a node"
    node = ctx.aggregation_nodes[0]
    assert node.title == "Postgres across projects"
    assert set(node.session_ids) == {"s1", "s2"}
    # Additive: the per-session episodes are still returned, not displaced.
    assert ctx.episodes, "episode tier must still fire alongside the node tier"


def test_retrieval_empty_when_disabled(conn, cfg):
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
    # Build with the layer enabled so node rows exist...
    build_aggregation_nodes(conn, _enabled(cfg), _agg_llm(), embed)
    # ...but query with it disabled (the default cfg): the tier must not run.
    ctx = augment(conn, cfg, "postgres billing", embedding_client=embed)
    assert ctx.aggregation_nodes == []
