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
    load_digest,
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


_DIGEST_JSON = json.dumps({
    "title": "User digest",
    "summary": "Runs billing and analytics on Postgres; also cycles on weekends.",
})


_ROLLUP_JSON = json.dumps({
    "title": "Mixed threads",
    "summary": "Several distinct topics, each preserved.",
})


def _agg_llm() -> StubLLMClient:
    # Routes on prompt phrases unique to each system prompt: level-0 cluster
    # fusion (AGGREGATE_SYSTEM), intermediate rollups (ROLLUP_SYSTEM), and the
    # root digest (DIGEST_SYSTEM).
    return StubLLMClient(
        fixtures={
            "fuse several related episodes": _NODE_JSON,
            "combined summary that loses no thread": _ROLLUP_JSON,
            "standing digest of everything known": _DIGEST_JSON,
        },
        default="[]",
    )


def _enabled(cfg: HyMemConfig, *, digest: bool = False) -> HyMemConfig:
    """Aggregation layer on; the v17 digest rollup only when a test opts in, so
    the level-0 policy tests keep asserting against level-0 rows alone."""
    return replace(cfg, aggregation_nodes_enabled=True,
                   aggregation_digest_enabled=digest)


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


def test_rebuild_reuses_fusion_for_unchanged_cluster(conn, cfg):
    # An unchanged member set must NOT pay a second LLM fusion call: the
    # content-hash node id keys the stored title/summary for reuse. Second
    # build runs with an LLM whose fixture would CHANGE the summary — the
    # stored fusion surviving proves no call was made for the stable cluster.
    acfg = _enabled(cfg)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "t1", "s", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "t2", "s", ["postgres", "billing"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), None)
    first = conn.execute("SELECT id, title, summary FROM aggregation_nodes").fetchone()

    poisoned = StubLLMClient(
        fixtures={"fuse several related episodes": json.dumps(
            {"title": "WRONG", "summary": "this fusion must never be used"})},
        default="[]",
    )
    built = build_aggregation_nodes(conn, acfg, poisoned, None)

    row = conn.execute("SELECT id, title, summary FROM aggregation_nodes").fetchone()
    assert built == 1
    assert row["id"] == first["id"]
    assert row["title"] == first["title"] == "Postgres across projects"
    assert row["summary"] == first["summary"]


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

    ctx = augment(conn, acfg, "postgres billing", embedding_client=embed,
                  ability="TR")
    assert ctx.aggregation_nodes, "enabled layer should surface a node for TR"
    node = ctx.aggregation_nodes[0]
    assert node.title == "Postgres across projects"
    assert set(node.session_ids) == {"s1", "s2"}
    # Additive: the per-session episodes are still returned, not displaced.
    assert ctx.episodes, "episode tier must still fire alongside the node tier"


def test_retrieval_gated_to_tr_by_default(conn, cfg):
    # The G4 A/B verdict: broad injection reshuffles ranking against gold
    # message hits (KU −9.0pp), so by default the tier only fires for TR.
    acfg = _enabled(cfg)
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), embed)

    # Unrouted (ability None) and a non-allowlisted ability both get no nodes...
    assert augment(conn, acfg, "postgres billing",
                   embedding_client=embed).aggregation_nodes == []
    assert augment(conn, acfg, "postgres billing", embedding_client=embed,
                   ability="KU").aggregation_nodes == []
    # ...and the host's ability hint is case-insensitive on the gate.
    assert augment(conn, acfg, "postgres billing", embedding_client=embed,
                   ability="tr").aggregation_nodes


def test_retrieval_broad_mode_with_empty_allowlist(conn, cfg):
    # Empty allowlist = the broad pre-G4 behavior (every query gets the tier),
    # kept so the A/B remains reproducible.
    acfg = replace(_enabled(cfg), aggregation_inject_abilities=())
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), embed)

    ctx = augment(conn, acfg, "postgres billing", embedding_client=embed)
    assert ctx.aggregation_nodes, "broad mode should fire without an ability"


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


# ── v17 hierarchy / root digest ──────────────────────────────────────────────

def test_digest_root_covers_nodes_and_unclustered_episodes(cfg, conn):
    # e1+e2 cluster (level-0 node); e3 is a disjoint singleton no cluster
    # absorbs — the digest must still cover it, so the root's members are the
    # node AND the pass-through episode (whole-store coverage, not just threads).
    assert cfg.aggregation_digest_enabled, "digest must be on by default"
    acfg = _enabled(cfg, digest=True)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e3", "s3", "Weekend cycling",
                      "Started cycling on weekends.", ["cycling"])
    built = build_aggregation_nodes(conn, acfg, _agg_llm(), None)
    assert built == 2                            # level-0 node + root

    root = conn.execute(
        "SELECT * FROM aggregation_nodes WHERE is_root = 1").fetchone()
    assert root is not None and root["level"] >= 1
    node_id = conn.execute(
        "SELECT id FROM aggregation_nodes WHERE level = 0").fetchone()["id"]
    assert set(json.loads(root["member_episode_ids"])) == {node_id, "e3"}
    assert json.loads(root["session_ids"]) == ["s1", "s2", "s3"]

    digest = load_digest(conn)
    assert digest is not None
    assert digest.title == "User digest"
    assert "cycles" in digest.summary
    assert digest.n_sessions == 3
    assert digest.n_sessions_total == 3


def test_digest_absent_when_rollup_disabled(cfg, conn):
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
    build_aggregation_nodes(conn, _enabled(cfg, digest=False), _agg_llm(), None)
    assert load_digest(conn) is None
    # Level-0 layer is untouched by the digest switch.
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM aggregation_nodes WHERE level = 0"
    ).fetchone()["c"] == 1


def test_digest_root_excluded_from_retrieval_tier(cfg, conn):
    # The root says "cycles on weekends" — a query for exactly that must NOT
    # surface it in ctx.aggregation_nodes: levels >= 1 are standing context for
    # HyMem.digest(), never retrieval competitors (the G4 crowding lesson).
    acfg = replace(_enabled(cfg, digest=True),
                   aggregation_inject_abilities=())   # broad mode: tier always on
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e3", "s3", "Weekend cycling",
                      "Started cycling on weekends.", ["cycling"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), embed)
    assert load_digest(conn) is not None

    ctx = augment(conn, acfg, "cycles on weekends digest", embedding_client=embed)
    root_id = conn.execute(
        "SELECT id FROM aggregation_nodes WHERE is_root = 1").fetchone()["id"]
    assert root_id not in {h.node_id for h in ctx.aggregation_nodes}


def test_digest_converges_on_fully_disjoint_episodes(cfg, conn):
    # No two episodes link (disjoint entities, no vectors) and the fan-in is
    # forced tiny, so natural clustering makes NO progress at every level: the
    # consecutive-chunk fallback must still reduce the frontier and reach a
    # root. 5 leaves @ fan_in 2 → 2 rollup levels before the root fusion.
    acfg = replace(_enabled(cfg, digest=True), aggregation_max_members=2)
    with core_db.transaction(conn):
        for i in range(1, 6):
            _seed_episode(conn, f"e{i}", f"s{i}", f"Topic {i}",
                          f"Notes about topic {i}.", [f"topic{i}"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), None)

    digest = load_digest(conn)
    assert digest is not None
    assert digest.n_sessions == 5
    levels = [r["level"] for r in conn.execute(
        "SELECT level FROM aggregation_nodes WHERE is_root = 0").fetchall()]
    assert levels and all(lv >= 1 for lv in levels)   # no level-0 clusters formed


def test_digest_leaf_cap_samples_across_history_not_recency(cfg, conn):
    # 4 disjoint episodes, leaf cap 2: a recency slice would digest only
    # {e3, e4} (the narrow "last fortnight" digest seen on the first prod
    # build); even sampling must keep the span's endpoints {e1, e4}.
    acfg = replace(_enabled(cfg, digest=True), aggregation_digest_max_leaves=2)
    with core_db.transaction(conn):
        for i in range(1, 5):
            _seed_episode(conn, f"e{i}", f"s{i}", f"Topic {i}",
                          f"Notes about topic {i}.", [f"topic{i}"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), None)

    root = conn.execute(
        "SELECT member_episode_ids FROM aggregation_nodes WHERE is_root = 1"
    ).fetchone()
    assert set(json.loads(root["member_episode_ids"])) == {"e1", "e4"}


def test_digest_rebuild_reuses_fusions(cfg, conn):
    # Unchanged store → unchanged member sets at EVERY level → the rebuild must
    # reuse all stored fusions (level-0, rollups, root) without an LLM call.
    # The second build's LLM would poison any node it actually fused.
    acfg = _enabled(cfg, digest=True)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e3", "s3", "Weekend cycling",
                      "Started cycling on weekends.", ["cycling"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), None)
    before = load_digest(conn)

    wrong = json.dumps({"title": "WRONG", "summary": "must not be used"})
    poisoned = StubLLMClient(
        fixtures={
            "fuse several related episodes": wrong,
            "combined summary that loses no thread": wrong,
            "standing digest of everything known": wrong,
        },
        default="[]",
    )
    build_aggregation_nodes(conn, acfg, poisoned, None)
    assert poisoned.calls == []                  # every fusion reused
    after = load_digest(conn)
    assert after is not None and after.summary == before.summary


def test_hymem_digest_api_none_by_default(cfg):
    # Default config: aggregation layer off → digest() is None, no error.
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    try:
        assert hy.digest() is None
    finally:
        hy.close()
