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
