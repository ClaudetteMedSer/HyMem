"""The structural rebuild forecast (schema v31).

The amplification bound `rebuilt ~ A*level0_missed + root + leaf` was never
fittable on the production store: the 2026-08-09 dispersion read found
`level0_missed` pinned at 3 for 11 of 13 dreams, so the slope had one x-value
and the intercept absorbed the leaf term. The exit is to compute the quantity
instead of estimating the constant — a node must rebuild when its membership is
new, and the previous tree plus this dream's grouping says exactly which nodes
those are.

    predicted = nodes whose (level, member set) is absent from the old tree
    actual    = nodes whose id missed the fusion cache
    residual  = actual - predicted

The test that carries the design is `test_a_salt_bump_shows_up_as_pure_keying
_residual`: it injects the exact failure the reuse watch keeps hitting
(membership identical, id changed) and shows the residual names it on ONE
dream, with no bar and no dispersion. `test_arrivals_are_explained_not
_flagged` is its control — a real membership change must leave the residual at
zero however large the rebuild is, or the instrument just re-reports the
rebuild it was supposed to explain.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hymem import HyMem, StubEmbeddingClient
from hymem.core import db as core_db
from hymem.dreaming import aggregate as agg_mod
from hymem.dreaming.aggregate import _forecast_rebuild, build_aggregation_nodes
from hymem.extraction.llm import StubLLMClient

_NODE_JSON = json.dumps({"title": "Postgres", "summary": "Postgres everywhere."})
_ROLLUP_JSON = json.dumps({"title": "Mixed", "summary": "Several threads."})
_DIGEST_JSON = json.dumps({"title": "Digest", "summary": "Everything known."})


def _agg_llm() -> StubLLMClient:
    return StubLLMClient(
        fixtures={
            "fuse several related episodes": _NODE_JSON,
            "combined summary that loses no thread": _ROLLUP_JSON,
            "standing digest of everything known": _DIGEST_JSON,
        },
        default="[]",
    )


def _cfg(cfg):
    return replace(cfg, aggregation_nodes_enabled=True,
                   aggregation_digest_enabled=True)


def _seed_episode(conn, eid, sid, entities):
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (sid,))
    conn.execute(
        """INSERT INTO episodes(id, session_id, title, summary, participants,
                                start_message_id, end_message_id, outcome, key_entities)
           VALUES (?, ?, ?, ?, '[]', 1, 2, NULL, ?)""",
        (eid, sid, f"Topic {eid}", f"Notes about {eid}.", json.dumps(entities)),
    )


@pytest.fixture
def conn(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    yield hy.conn
    hy.close()


def _seed_pairs(conn, n_pairs: int, start: int = 1) -> None:
    """Cross-session PAIRS, so each pair survives `select_clusters` and becomes
    a real level-0 node with ancestors — the structure amplification runs up."""
    with core_db.transaction(conn):
        for i in range(start, start + n_pairs):
            _seed_episode(conn, f"e{i}a", f"s{i}a", [f"topic{i}"])
            _seed_episode(conn, f"e{i}b", f"s{i}b", [f"topic{i}"])


def _dream(conn, cfg):
    return build_aggregation_nodes(conn, _cfg(cfg), _agg_llm(), None)


# ── the pure predictor ─────────────────────────────────────────────────────

def test_forecast_counts_new_membership_as_predicted():
    prev = {(0, frozenset({"e1", "e2"}))}
    rows = [
        {"level": 0, "member_ids": ["e1", "e2"], "is_root": 0, "reused": True},
        {"level": 0, "member_ids": ["e3", "e4"], "is_root": 0, "reused": False},
    ]
    f = _forecast_rebuild(rows, prev)
    assert (f.predicted, f.actual, f.residual, f.facts_rekey) == (1, 1, 0, 0)


def test_forecast_flags_membership_identical_rebuilds():
    """Same members, still rebuilt, and not the root — nothing in the build
    explains it. This is the whole point of the instrument."""
    prev = {(0, frozenset({"e1", "e2"}))}
    rows = [{"level": 0, "member_ids": ["e1", "e2"], "is_root": 0, "reused": False}]
    f = _forecast_rebuild(rows, prev)
    assert (f.predicted, f.actual, f.residual) == (0, 1, 1)


def test_forecast_charges_an_unchanged_root_to_the_facts_hash():
    """The root keys on the VERIFIED FACTS hash as well as membership, so a
    membership-identical root rebuild is a known cause and must NOT inflate the
    residual — otherwise the detector fires once per graph edit."""
    prev = {(1, frozenset({"n1", "n2"}))}
    rows = [{"level": 1, "member_ids": ["n1", "n2"], "is_root": 1, "reused": False}]
    f = _forecast_rebuild(rows, prev)
    assert (f.predicted, f.actual, f.residual, f.facts_rekey) == (1, 1, 0, 1)


def test_forecast_is_level_aware():
    """A level-1 node whose members happen to match a level-0 member set is
    still a new node; keying membership without the level would silently
    excuse it."""
    prev = {(0, frozenset({"a", "b"}))}
    rows = [{"level": 1, "member_ids": ["a", "b"], "is_root": 0, "reused": False}]
    assert _forecast_rebuild(rows, prev).residual == 0


# ── end to end through a real build ────────────────────────────────────────

def test_steady_state_predicts_and_observes_zero(conn, cfg):
    _seed_pairs(conn, 3)
    _dream(conn, cfg)
    second = _dream(conn, cfg)
    assert second.predicted_rebuild == 0
    assert second.keying_residual == 0
    assert second.nodes == second.reused


def test_arrivals_are_explained_not_flagged(conn, cfg):
    """The control for the salt test below. New episodes rebuild real nodes all
    the way to the root, and every one of those rebuilds must be PREDICTED —
    residual 0 no matter how large the amplification. An instrument that
    flagged this would just be re-reporting the rebuild."""
    _seed_pairs(conn, 3)
    _dream(conn, cfg)
    _seed_pairs(conn, 2, start=10)
    result = _dream(conn, cfg)

    rebuilt = result.nodes - result.reused
    assert rebuilt > 0, "arrivals must actually rebuild something"
    assert result.predicted_rebuild == rebuilt
    assert result.keying_residual == 0


def test_a_salt_bump_shows_up_as_pure_keying_residual(conn, cfg, monkeypatch):
    """THE detector. Bumping the cluster salt re-keys every level-0 node while
    membership is untouched — the salt/CDC/rowid-shadow class the reuse watch
    kept hitting. Predicted stays 0 for those nodes (their membership is in the
    old tree), actual counts them, so the residual names the defect on a SINGLE
    dream with no bar to calibrate and no dispersion to wait for.
    """
    _seed_pairs(conn, 3)
    _dream(conn, cfg)
    assert _dream(conn, cfg).keying_residual == 0        # quiet before

    monkeypatch.setattr(agg_mod, "_CLUSTER_SALT", "cluster.vTEST")
    result = _dream(conn, cfg)

    assert result.keying_residual > 0
    # Every level-0 node re-keyed, and nothing about membership changed.
    assert result.predicted_rebuild < result.nodes - result.reused


def test_a_changed_facts_anchor_is_attributed_not_residual(conn, cfg, monkeypatch):
    """A changed knowledge graph legitimately re-keys the root over an
    unchanged tree. It must land in facts_rekey, leaving the residual clean."""
    _seed_pairs(conn, 3)
    _dream(conn, cfg)
    assert _dream(conn, cfg).keying_residual == 0

    monkeypatch.setattr(agg_mod, "_anchor_facts",
                        lambda conn_, cap: ["a brand new verified fact"])
    result = _dream(conn, cfg)

    assert result.facts_rekey == 1
    assert result.keying_residual == 0


def test_forecast_is_persisted_on_the_dream_run(cfg):
    """The columns are the readable surface — honcho dreams' stderr goes to the
    gateway pipe, which is what made the v29 channel unreadable in the first
    place."""
    hy = HyMem(_cfg(cfg), llm=_agg_llm(), embedding_client=StubEmbeddingClient())
    try:
        _seed_pairs(hy.conn, 3)
        hy.dream()
        row = hy.conn.execute(
            "SELECT aggregation_predicted_rebuild AS predicted, "
            "       aggregation_keying_residual AS residual, "
            "       aggregation_facts_rekey AS facts "
            "FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["predicted"] is not None
        assert row["residual"] == 0
        assert row["facts"] == 0
    finally:
        hy.close()


# ── v33 rebuild decomposition by tree level ──────────────────────────────────
# `aggregation_leaf_changed` is BINARY, so a low-reuse row carrying
# leaf_changed=1 admits two readings that it cannot separate: the digest leaf
# set shifted and cascaded (benign, structural) or the level-0 windowing
# confinement leaked (the row class raptor_digest_plan.md routes to REOPEN THE
# WINDOWING ANALYSIS). Three production rows sat in that ambiguity across three
# separate readings — #1183 (7.3 rebuilds per level-0 miss), #1307 (6.3) and
# #1317 (6.7), all with keying_residual=0. Splitting `actual` by level decides
# it on ONE dream, the same move v31 made for keying.


def _forecast(rows, prev):
    return _forecast_rebuild(rows, prev)


def _row(node_id, level, members, *, is_root=0, reused=False):
    return {"id": node_id, "level": level, "member_ids": list(members),
            "is_root": is_root, "reused": reused}


def test_decomposition_sums_to_actual():
    """The triple is self-checking by construction — if it ever stops summing
    to `actual`, the split is lying and so is anything read off it."""
    rows = [
        _row("a", 0, ["e1", "e2"]),
        _row("b", 0, ["e3", "e4"], reused=True),
        _row("c", 1, ["a", "b"]),
        _row("d", 1, ["x"]),
        _row("r", 2, ["c", "d"], is_root=1),
    ]
    f = _forecast(rows, set())
    assert f.rebuilt_level0 + f.rebuilt_rollup + f.rebuilt_root == f.actual
    assert (f.rebuilt_level0, f.rebuilt_rollup, f.rebuilt_root) == (1, 2, 1)


def test_leaf_cascade_and_cluster_churn_are_distinguishable():
    """The whole point: two dreams with the SAME rebuild total and the same
    binary leaf_changed read differently once the level split is available.

    Reading (a) — benign digest cascade: one level-0 node rebuilt, the rest of
    the churn is the rollup chain above it.
    Reading (b) — confinement leak: the same total, but it is the CLUSTER layer
    churning, which is what the windowing reversal exists to prevent.
    """
    cascade = [_row("a", 0, ["e1"])] + [
        _row(f"r{i}", 1, [f"m{i}"]) for i in range(4)
    ]
    leak = [_row(f"c{i}", 0, [f"e{i}"]) for i in range(5)]

    fa, fb = _forecast(cascade, set()), _forecast(leak, set())
    assert fa.actual == fb.actual == 5          # indistinguishable before v33
    assert (fa.rebuilt_level0, fa.rebuilt_rollup) == (1, 4)
    assert (fb.rebuilt_level0, fb.rebuilt_rollup) == (5, 0)


def test_reused_nodes_are_not_counted_in_any_term():
    rows = [_row(f"c{i}", 0, [f"e{i}"], reused=True) for i in range(3)]
    f = _forecast(rows, set())
    assert (f.actual, f.rebuilt_level0, f.rebuilt_rollup, f.rebuilt_root) == (0, 0, 0, 0)


def test_root_is_counted_as_root_not_rollup():
    """The root sits at level >= 1 but keys on the anchor-facts hash as well as
    membership, so folding it into the rollup term would smear a facts-rekey
    into the cascade signal the split exists to isolate."""
    rows = [_row("r", 3, ["c1"], is_root=1)]
    f = _forecast(rows, set())
    assert (f.rebuilt_root, f.rebuilt_rollup, f.rebuilt_level0) == (1, 0, 0)


def test_decomposition_is_persisted_on_the_dream_run(cfg):
    hy = HyMem(_cfg(cfg), llm=_agg_llm(), embedding_client=StubEmbeddingClient())
    try:
        _seed_pairs(hy.conn, 3)
        hy.dream()
        row = hy.conn.execute(
            "SELECT aggregation_nodes_built AS built, "
            "       aggregation_nodes_reused AS reused, "
            "       aggregation_rebuilt_level0 AS l0, "
            "       aggregation_rebuilt_rollup AS rollup, "
            "       aggregation_rebuilt_root AS root "
            "FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["l0"] is not None
        # The self-check, end to end through the store.
        assert row["l0"] + row["rollup"] + row["root"] == row["built"] - row["reused"]
    finally:
        hy.close()
