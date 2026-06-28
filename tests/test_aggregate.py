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
from hymem.core.vectors import encode_vector
from hymem.dreaming.aggregate import (
    build_aggregation_nodes,
    cluster_episodes,
    Digest,
    expand_node,
    generate_candidate_pairs,
    load_clusterable_episodes,
    load_digest,
    _CLUSTER_SALT,
    _node_id,
    _ROLLUP_SALT,
    _ROOT_SALT,
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


# ── Stage-3a chaining guard: max_cluster_size recency-window split ───────────
# The prod store grew ONE 348-episode component spanning 61 sessions through
# transitive OR-link chaining (cluster_size_probe, 2026-06-12); the guard
# splits over-cap components into recency windows. Membership semantics changed
# → cluster salt bumped to v3 (rollup/root salts unchanged: their cache ids key
# on member sets, which change naturally).

def _entity_chain(n: int, start_message_ids: list[int] | None = None) -> list[dict]:
    """n episodes where ONLY consecutive pairs entity-link at jaccard exactly
    0.50 (= the production ent threshold): {t0}, {t0,t1}, {t1}, {t1,t2}, ...
    Non-adjacent pairs overlap at most 1/3, so one big component can exist only
    through union-find transitivity — the failure mode the guard splits."""
    eps = []
    for k in range(n):
        j = k // 2
        ents = [f"t{j}"] if k % 2 == 0 else [f"t{j}", f"t{j + 1}"]
        ep = _ep(f"e{k}", f"s{k}", ents)
        if start_message_ids is not None:
            ep["start_message_id"] = start_message_ids[k]
        eps.append(ep)
    return eps


def test_salts_pinned():
    # Membership semantics changed (recency-window split) → cluster salt MUST
    # be v3, or cached pre-guard mega-cluster fusions survive the fix (the
    # "Acme Corp" lesson). Rollup/root deliberately NOT bumped.
    assert _CLUSTER_SALT == "cluster.v3"
    assert _ROLLUP_SALT == "rollup.v2"
    assert _ROOT_SALT == "root.v4"


def test_over_cap_component_splits_into_windows_deterministically():
    eps = _entity_chain(7)                      # ONE component of 7, uncapped
    uncapped = cluster_episodes(eps, 0.55, 0.50)
    assert len(set(uncapped.values())) == 1

    capped = cluster_episodes(eps, 0.55, 0.50, max_cluster_size=3)
    windows: dict[int, set[str]] = {}
    for eid, label in capped.items():
        windows.setdefault(label, set()).add(eid)
    assert all(len(w) <= 3 for w in windows.values())
    # 7 = 1 + 3 + 3: full windows align to the newest end, the undersized
    # window holds the OLDEST episodes (input order = recency fallback).
    assert sorted(len(w) for w in windows.values()) == [1, 3, 3]
    # Deterministic: a second run is identical, label for label.
    assert cluster_episodes(eps, 0.55, 0.50, max_cluster_size=3) == capped


def test_max_cluster_size_none_is_the_uncapped_clusterer():
    eps = _entity_chain(7)
    assert (cluster_episodes(eps, 0.55, 0.50, max_cluster_size=None)
            == cluster_episodes(eps, 0.55, 0.50))


def test_windows_are_recency_ordered_by_start_message_id():
    # Input order is SHUFFLED relative to recency: start_message_id (a global
    # AUTOINCREMENT messages.id in prod, carried by load_clusterable_episodes)
    # is the ordering signal, not list position. Recency order by message id:
    # e0(10) e1(20) e2(30) e3(40) e4(50) e5(60) e6(70).
    eps = _entity_chain(7, start_message_ids=[10, 20, 30, 40, 50, 60, 70])
    eps = [eps[i] for i in (4, 0, 6, 2, 5, 1, 3)]   # shuffle input order

    capped = cluster_episodes(eps, 0.55, 0.50, max_cluster_size=3)
    windows: dict[int, set[str]] = {}
    for eid, label in capped.items():
        windows.setdefault(label, set()).add(eid)
    # Consecutive recency slices, full windows at the newest end:
    assert sorted(windows.values(), key=len) == sorted(
        [{"e0"}, {"e1", "e2", "e3"}, {"e4", "e5", "e6"}], key=len)


def test_component_exactly_at_cap_is_not_split():
    eps = _entity_chain(7)
    capped = cluster_episodes(eps, 0.55, 0.50, max_cluster_size=7)
    assert capped == cluster_episodes(eps, 0.55, 0.50)   # one component, intact


def test_build_respects_aggregation_max_cluster_size(conn, cfg):
    # Integration: a 7-episode transitive chain across 7 sessions, cap 3 → no
    # persisted level-0 node may exceed 3 members. Windows of 3 span 3 sessions
    # (pass min_members/min_sessions); the undersized window of 1 is dropped by
    # the SAME downstream policy as any singleton.
    for k in range(7):
        j = k // 2
        ents = [f"t{j}"] if k % 2 == 0 else [f"t{j}", f"t{j + 1}"]
        _seed_episode(conn, f"e{k}", f"s{k}", f"ep {k}", f"about {ents}",
                      ents, start=10 * (k + 1), end=10 * (k + 1) + 1)
    cfg = replace(_enabled(cfg), aggregation_max_cluster_size=3)
    n = build_aggregation_nodes(conn, cfg, _agg_llm()).nodes
    rows = conn.execute(
        "SELECT n_members, member_episode_ids FROM aggregation_nodes "
        "WHERE level = 0").fetchall()
    assert n == len(rows) == 2                       # two full windows kept
    assert all(r["n_members"] <= 3 for r in rows)
    members = {frozenset(json.loads(r["member_episode_ids"])) for r in rows}
    assert members == {frozenset({"e1", "e2", "e3"}),
                       frozenset({"e4", "e5", "e6"})}


def test_build_uncapped_when_config_cap_is_zero(conn, cfg):
    # House style: 0 = uncapped (translated to None at the call site).
    for k in range(7):
        j = k // 2
        ents = [f"t{j}"] if k % 2 == 0 else [f"t{j}", f"t{j + 1}"]
        _seed_episode(conn, f"e{k}", f"s{k}", f"ep {k}", f"about {ents}",
                      ents, start=10 * (k + 1), end=10 * (k + 1) + 1)
    cfg = replace(_enabled(cfg), aggregation_max_cluster_size=0)
    assert build_aggregation_nodes(conn, cfg, _agg_llm()).nodes == 1
    row = conn.execute(
        "SELECT n_members FROM aggregation_nodes WHERE level = 0").fetchone()
    assert row["n_members"] == 7                     # the raw mega-component


# ── phase-3 snapshot ceiling: async episodes must not poison the rebuild ──────
# The MCP server writes episodes asynchronously; one landing between the dream's
# phase-3 boundary and the clustering read shifts a cluster's member set → new
# `_node_id` → a spurious near-full refusion (prod dream runs 678/680,
# 2026-06-28). The runner snapshots MAX(episodes.rowid) at that boundary and
# threads it in as `episode_ceiling_rowid`.

def _level0_members(conn) -> set[str]:
    members: set[str] = set()
    for r in conn.execute(
            "SELECT member_episode_ids FROM aggregation_nodes WHERE level = 0"):
        members.update(json.loads(r["member_episode_ids"]))
    return members


def test_aggregation_ceiling_excludes_post_snapshot_episodes(conn, cfg):
    cfg = _enabled(cfg)
    llm = _agg_llm()
    for eid, sid in (("e1", "s1"), ("e2", "s2")):     # one cross-session cluster
        _seed_episode(conn, eid, sid, eid, eid, ["postgres", "billing"])

    ceiling = conn.execute("SELECT MAX(rowid) AS m FROM episodes").fetchone()["m"]

    first = build_aggregation_nodes(conn, cfg, llm, None,
                                    episode_ceiling_rowid=ceiling)
    assert (first.nodes, first.reused) == (1, 0)      # fused fresh
    assert _level0_members(conn) == {"e1", "e2"}

    # An async stray lands AFTER the snapshot, sharing the cluster's entities so
    # that — uncapped — it WOULD join and change the member set.
    _seed_episode(conn, "e3", "s3", "e3", "e3", ["postgres", "billing"])

    second = build_aggregation_nodes(conn, cfg, llm, None,
                                     episode_ceiling_rowid=ceiling)
    assert _level0_members(conn) == {"e1", "e2"}      # stray above the ceiling
    assert (second.nodes, second.reused) == (1, 1)    # cached fusion reused

    # Control: lift the ceiling and the same stray DOES join → membership change
    # → fresh fusion. Proves the ceiling, not luck, is what excluded it.
    third = build_aggregation_nodes(conn, cfg, llm, None)
    assert _level0_members(conn) == {"e1", "e2", "e3"}
    assert third.reused == 0


# ── Stage-3b candidate blocking: O(n²) all-pairs → entity index + vec KNN ────
# Prod timing (2026-06-12): 395 episodes → 77,815 pair tests → 4.04s per dream,
# past the 2s gate. Blocking only generates the candidate pair list; the pure
# clusterer's contract (and the probe's exact all-pairs default) is preserved.

def _seed_embedded_episode(conn, embed, eid, sid, summary, entities,
                           start=1, end=2):
    """Episode + episode_embeddings row (vector = stub embed of `summary`, so
    identical summaries cosine-link at 1.0). Callers run
    `core_db.ensure_vec_table(conn, embed.dim)` AFTER seeding — its cold-start
    backfill mirrors episode_embeddings into vec_episodes by rowid, the same
    path a real store takes."""
    _seed_episode(conn, eid, sid, eid, summary, entities, start, end)
    vec = embed.embed([summary])[0]
    conn.execute(
        "INSERT INTO episode_embeddings(episode_id, vector_json, model, dim, text_hash) "
        "VALUES (?, ?, ?, ?, ?)",
        (eid, encode_vector(vec), embed.model, embed.dim, f"hash:{eid}"),
    )


def test_candidate_pairs_none_default_is_exact_all_pairs():
    # (a) The probe contract: default None — and an explicit full pair set —
    # both reproduce today's all-pairs labels exactly.
    eps = _entity_chain(7)
    base = cluster_episodes(eps, 0.55, 0.50)
    assert cluster_episodes(eps, 0.55, 0.50, candidate_pairs=None) == base
    all_pairs = {
        tuple(sorted((a["id"], b["id"])))
        for i, a in enumerate(eps) for b in eps[i + 1:]
    }
    assert cluster_episodes(eps, 0.55, 0.50, candidate_pairs=all_pairs) == base
    # And the cap path composes with candidate_pairs unchanged.
    assert (cluster_episodes(eps, 0.55, 0.50, max_cluster_size=3,
                             candidate_pairs=all_pairs)
            == cluster_episodes(eps, 0.55, 0.50, max_cluster_size=3))


def test_generate_candidate_pairs_entity_index_exact_plus_knn(conn, cfg):
    # (b) Entity arm: exactly the pairs sharing >= 1 entity. Cosine arm: only
    # the KNN pairs of the (two) episodes that carry vectors. e5/e6 share NO
    # entity but have identical summaries → only the vec arm can pair them.
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "t1", "billing", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "t2", "analytics", ["postgres"])
        _seed_episode(conn, "e3", "s3", "t3", "stream", ["kafka"])
        _seed_episode(conn, "e4", "s4", "t4", "stream2", ["kafka", "flink"])
        _seed_embedded_episode(conn, embed, "e5", "s5", "same words", ["solo5"])
        _seed_embedded_episode(conn, embed, "e6", "s6", "same words", ["solo6"])
    core_db.ensure_vec_table(conn, embed.dim)

    eps = load_clusterable_episodes(conn)
    pairs = generate_candidate_pairs(conn, eps, emb_top_k=24)
    assert pairs == {("e1", "e2"), ("e3", "e4"), ("e5", "e6")}


def test_blocking_exact_when_k_covers_store(conn, cfg):
    # (c) k >= n-1 → blocked labels == exact labels on a real store with both
    # link kinds present (cosine via identical stub vectors, entity via jaccard).
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_embedded_episode(conn, embed, "e1", "s1", "postgres rollout notes",
                               ["postgres"])
        _seed_embedded_episode(conn, embed, "e2", "s2", "postgres rollout notes",
                               ["billing"])                       # cosine link to e1
        _seed_embedded_episode(conn, embed, "e3", "s3", "kafka pipeline",
                               ["kafka", "stream"])
        _seed_embedded_episode(conn, embed, "e4", "s4", "totally different text",
                               ["kafka", "stream"])               # entity link to e3
        _seed_embedded_episode(conn, embed, "e5", "s5", "weekend cycling",
                               ["cycling"])                       # singleton
    core_db.ensure_vec_table(conn, embed.dim)

    eps = load_clusterable_episodes(conn)
    exact = cluster_episodes(eps, 0.55, 0.50)
    pairs = generate_candidate_pairs(conn, eps, emb_top_k=24)
    assert pairs is not None
    assert cluster_episodes(eps, 0.55, 0.50, candidate_pairs=pairs) == exact


def test_generate_candidate_pairs_none_without_vec_table(conn, cfg):
    # (d) sqlite_vec is optional: no vec_episodes table → None → the caller
    # falls back to exact all-pairs, embedded small stores unchanged.
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "t1", "s", ["postgres"])
        _seed_episode(conn, "e2", "s2", "t2", "s", ["postgres"])
    conn.execute("DROP TABLE IF EXISTS vec_episodes")   # simulate vec-less store
    assert not core_db.has_vec_table(conn, table="vec_episodes")
    eps = load_clusterable_episodes(conn)
    assert generate_candidate_pairs(conn, eps, emb_top_k=24) is None


def test_generate_candidate_pairs_none_when_top_k_zero(conn, cfg):
    # (e) aggregation_blocking_top_k=0 disables blocking even with the vec
    # table present.
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_embedded_episode(conn, embed, "e1", "s1", "postgres", ["postgres"])
        _seed_embedded_episode(conn, embed, "e2", "s2", "postgres", ["postgres"])
    core_db.ensure_vec_table(conn, embed.dim)
    assert core_db.has_vec_table(conn, table="vec_episodes")
    eps = load_clusterable_episodes(conn)
    assert generate_candidate_pairs(conn, eps, emb_top_k=0) is None
    assert generate_candidate_pairs(conn, eps, emb_top_k=24) is not None


def test_blocking_reduces_candidate_pair_count(conn, cfg):
    # (f) ~60 episodes, sparse entities (20 entities × 3 episodes) and clustered
    # vectors (6 identical-summary groups of 10): the candidate set must be
    # strictly smaller than all n(n-1)/2 pairs, while still covering every
    # entity-sharing pair exactly (the lossless arm).
    embed = StubEmbeddingClient()
    n = 60
    with core_db.transaction(conn):
        for i in range(n):
            _seed_embedded_episode(
                conn, embed, f"e{i:02d}", f"s{i:02d}",
                f"thread {i % 6} progress notes", [f"t{i % 20}"],
                start=i + 1, end=i + 2,
            )
    core_db.ensure_vec_table(conn, embed.dim)

    eps = load_clusterable_episodes(conn)
    pairs = generate_candidate_pairs(conn, eps, emb_top_k=5)
    assert pairs is not None
    assert len(pairs) < n * (n - 1) // 2
    entity_pairs = {
        tuple(sorted((a["id"], b["id"])))
        for i, a in enumerate(eps) for b in eps[i + 1:]
        if a["entities"] & b["entities"]
    }
    assert entity_pairs <= pairs


def test_build_same_nodes_with_blocking_on_and_off(conn, cfg):
    # (g) Integration through select_clusters/build: with k covering the store,
    # blocking on and off persist identical level-0 nodes.
    from dataclasses import replace as _replace
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_embedded_episode(conn, embed, "e1", "s1", "billing on postgres",
                               ["postgres", "billing"])
        _seed_embedded_episode(conn, embed, "e2", "s2", "analytics on postgres",
                               ["postgres", "billing"])
        _seed_embedded_episode(conn, embed, "e3", "s3", "weekend cycling",
                               ["cycling"])
    core_db.ensure_vec_table(conn, embed.dim)

    def _nodes():
        return {
            (r["id"], r["member_episode_ids"]) for r in conn.execute(
                "SELECT id, member_episode_ids FROM aggregation_nodes"
            ).fetchall()
        }

    blocking_on = _replace(_enabled(cfg), aggregation_blocking_top_k=24)
    blocking_off = _replace(_enabled(cfg), aggregation_blocking_top_k=0)
    build_aggregation_nodes(conn, blocking_on, _agg_llm())
    nodes_on = _nodes()
    build_aggregation_nodes(conn, blocking_off, _agg_llm())   # full rebuild
    assert nodes_on == _nodes()
    assert nodes_on, "the postgres cluster must produce a node either way"


# ── DB build + persistence ───────────────────────────────────────────────────

def test_build_creates_node_for_cross_session_cluster(conn, cfg):
    embed = StubEmbeddingClient()
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "Set up the billing service on Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics warehouse also runs Postgres.", ["postgres", "billing"])

    built = build_aggregation_nodes(conn, _enabled(cfg), _agg_llm(), embed).nodes
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
    built = build_aggregation_nodes(conn, cfg, _agg_llm(), None).nodes  # cfg disabled by default
    assert built == 0
    assert conn.execute("SELECT COUNT(*) AS c FROM aggregation_nodes").fetchone()["c"] == 0


def test_build_skips_single_session(conn, cfg):
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "t1", "s", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s1", "t2", "s", ["postgres", "billing"])
    built = build_aggregation_nodes(conn, _enabled(cfg), _agg_llm(), None).nodes
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
    built = build_aggregation_nodes(conn, acfg, poisoned, None).nodes

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
    built = build_aggregation_nodes(conn, acfg, _agg_llm(), None).nodes
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


def test_digest_root_fusion_is_grounded_in_graph_facts(cfg, conn):
    # The root prompt must carry the VERIFIED FACTS block built from active
    # non-derived graph edges — the store-grounded anchor that gives the model
    # true identity signals instead of a vacuum to fill (the Acme incident).
    from tests.conftest import seed_edge
    acfg = _enabled(cfg, digest=True)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Weekend cycling",
                      "Started cycling on weekends.", ["cycling"])
        seed_edge(conn, "atta", "part_of", "medflow", pos=5)
        seed_edge(conn, "noise", "uses", "leftpad", pos=3, derived=1)  # excluded
    llm = _agg_llm()
    build_aggregation_nodes(conn, acfg, llm, None)

    digest_calls = [c for c in llm.calls
                    if "standing digest of everything known" in c.system]
    assert len(digest_calls) == 1
    assert "atta part_of medflow" in digest_calls[0].user
    assert "leftpad" not in digest_calls[0].user        # derived edges excluded
    assert load_digest(conn) is not None


def test_digest_regenerates_when_graph_facts_change(cfg, conn):
    # Same tree membership, changed graph → the anchor hash in the root's
    # cache id must force a fresh fusion (a digest pinned to stale ground
    # truth is the failure the anchor exists to prevent).
    from tests.conftest import seed_edge
    acfg = _enabled(cfg, digest=True)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Weekend cycling",
                      "Started cycling on weekends.", ["cycling"])
        seed_edge(conn, "atta", "part_of", "medflow", pos=5)
    build_aggregation_nodes(conn, acfg, _agg_llm(), None)

    with core_db.transaction(conn):
        seed_edge(conn, "atta", "prefers", "duckdb", pos=4)
    fresh = StubLLMClient(
        fixtures={"standing digest of everything known": json.dumps(
            {"title": "Fresh digest", "summary": "Re-grounded."})},
        default="[]",
    )
    build_aggregation_nodes(conn, acfg, fresh, None)
    digest = load_digest(conn)
    assert digest is not None and digest.title == "Fresh digest"
    # And with the graph unchanged, a further rebuild reuses the new root.
    build_aggregation_nodes(conn, acfg, StubLLMClient(default="[]"), None)
    assert load_digest(conn).title == "Fresh digest"


def test_digest_anchor_disabled_with_zero_cap(cfg, conn):
    from tests.conftest import seed_edge
    acfg = replace(_enabled(cfg, digest=True), aggregation_digest_anchor_facts=0)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Weekend cycling",
                      "Started cycling on weekends.", ["cycling"])
        seed_edge(conn, "atta", "part_of", "medflow", pos=5)
    llm = _agg_llm()
    build_aggregation_nodes(conn, acfg, llm, None)

    digest_calls = [c for c in llm.calls
                    if "standing digest of everything known" in c.system]
    assert len(digest_calls) == 1
    assert "atta part_of medflow" not in digest_calls[0].user
    assert "(none)" in digest_calls[0].user


def test_hymem_digest_api_none_by_default(cfg):
    # Default config: aggregation layer off → digest() is None, no error.
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    try:
        assert hy.digest() is None
    finally:
        hy.close()


# ── Stage 5: delivery surfaces ───────────────────────────────────────────────

def test_digest_as_context_block_carries_summary_and_staleness():
    # The one canonical render every delivery surface uses: title header,
    # summary body, and a footer that makes coverage + build time visible.
    d = Digest(title="User digest", summary="Runs billing on Postgres.",
               n_sessions=3, n_sessions_total=4,
               generated_at="2026-06-12 09:00:00")
    block = d.as_context_block()
    assert block.startswith("## User digest")
    assert "Runs billing on Postgres." in block
    assert "covering 3 of 4 sessions" in block
    assert "generated 2026-06-12 09:00:00" in block


def test_digest_as_context_block_without_generated_at():
    # A root row with a NULL created_at loads as generated_at="" — the footer
    # then states coverage only rather than printing an empty timestamp.
    d = Digest(title="t", summary="s", n_sessions=1, n_sessions_total=1,
               generated_at="")
    block = d.as_context_block()
    assert "covering 1 of 1 sessions" in block
    assert "generated" not in block


def test_augment_digest_none_by_default(cfg, conn):
    # augment() stays lean: even with a built root digest, ctx.digest is None
    # unless the host opts in via cfg.augment_include_digest.
    acfg = _enabled(cfg, digest=True)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), None)
    assert load_digest(conn) is not None

    ctx = augment(conn, acfg, "postgres billing")
    assert ctx.digest is None


def test_augment_digest_included_when_opted_in(cfg, conn):
    acfg = replace(_enabled(cfg, digest=True), augment_include_digest=True)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
    build_aggregation_nodes(conn, acfg, _agg_llm(), None)

    ctx = augment(conn, acfg, "postgres billing")
    assert ctx.digest is not None
    assert ctx.digest.title == "User digest"
    assert ctx.digest.n_sessions_total == 2


def test_augment_digest_opt_in_none_when_no_root(cfg, conn):
    acfg = replace(cfg, augment_include_digest=True)
    ctx = augment(conn, acfg, "anything at all")
    assert ctx.digest is None


# ── Stage 4b: drill-down API ─────────────────────────────────────────────────

def _build_digest_tree(cfg, conn):
    """e1+e2 cluster into a level-0 node; e3 is a pass-through leaf — so the
    root's members are one child NODE and one EPISODE, the mixed shape the
    drill-down must resolve."""
    acfg = _enabled(cfg, digest=True)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing on Postgres",
                      "The billing service uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Analytics on Postgres",
                      "Analytics also uses Postgres.", ["postgres", "billing"])
        _seed_episode(conn, "e3", "s3", "Weekend cycling",
                      "Started cycling on weekends.", ["cycling"], start=5, end=9)
    build_aggregation_nodes(conn, acfg, _agg_llm(), None)


def test_expand_root_resolves_child_node_and_passthrough_episode(cfg, conn):
    _build_digest_tree(cfg, conn)
    digest = load_digest(conn)
    assert digest is not None and digest.node_id  # the traversal entry point

    exp = expand_node(conn, digest.node_id)
    assert exp is not None
    assert exp.is_root and exp.level >= 1
    assert exp.title == "User digest"
    assert exp.missing_member_ids == []

    assert len(exp.child_nodes) == 1
    child = exp.child_nodes[0]
    assert child.level == 0
    assert child.title == "Postgres across projects"
    assert child.n_members == 2 and child.n_sessions == 2

    assert [e.id for e in exp.episodes] == ["e3"]
    leaf = exp.episodes[0]
    assert leaf.session_id == "s3"
    assert (leaf.start_message_id, leaf.end_message_id) == (5, 9)


def test_expand_level0_node_returns_member_episodes_only(cfg, conn):
    _build_digest_tree(cfg, conn)
    root = expand_node(conn, load_digest(conn).node_id)

    exp = expand_node(conn, root.child_nodes[0].id)
    assert exp is not None
    assert exp.level == 0 and not exp.is_root
    assert exp.child_nodes == []
    assert {e.id for e in exp.episodes} == {"e1", "e2"}
    assert {e.session_id for e in exp.episodes} == {"s1", "s2"}


def test_expand_unknown_node_returns_none(cfg, conn):
    _build_digest_tree(cfg, conn)
    assert expand_node(conn, "no-such-node") is None


def test_expand_reports_dangling_members(cfg, conn):
    # Honest-read contract: a member id that resolves to neither table is
    # reported, not silently dropped (only reachable via store surgery).
    _build_digest_tree(cfg, conn)
    digest = load_digest(conn)
    with core_db.transaction(conn):
        conn.execute("DELETE FROM episodes WHERE id = 'e3'")

    exp = expand_node(conn, digest.node_id)
    assert exp.missing_member_ids == ["e3"]
    assert len(exp.child_nodes) == 1 and exp.episodes == []


def test_hymem_expand_node_api(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    try:
        assert hy.expand_node("anything") is None  # empty store, no error
    finally:
        hy.close()
