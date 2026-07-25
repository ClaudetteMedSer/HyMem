"""Track A — synthetic bridging-chain probe for query-time multi-hop traversal.

This is the *controlled* half of the Idea-A recall probe (additional_planning.md
§Idea A / benchmarks/readside_synthesis_plan.md Track A): a seeded graph with a
known chain, ground-truth bridge edges, and deterministic decay/min_score maths.
It encodes the G-A1 gate as assertions — bridging-edge recall rises when
`graph_multihop_enabled` flips on, and recall on the 1-hop control does NOT drop
(the additive invariant as a metric). The mined LME/BEAM slice is the second
probe source and runs on the box.

Chain under test (seed = ``atta``):

    atta —part_of→ medflow —deploys_to→ fly.io —located_in→ aws

Source 1 (1-hop entity anchor) retrieves only ``atta —part_of→ medflow``. The
bridge ``medflow —deploys_to→ fly.io`` is two edges out; the deep edge
``fly.io —hosted_by→ aws`` is three.
"""

from __future__ import annotations

import dataclasses

import pytest

from hymem.query.augment import _graph_lookup, _multihop_edges
from tests.conftest import seed_edge

# Laplace-smoothed confidence of a fresh edge (pos=1, neg=0): (1+1)/(1+0+2).
FRESH_CONF = 2.0 / 3.0
DECAY = 0.5
BRIDGE = ("medflow", "deploys_to", "fly.io")   # hop-2, the edge Source 1 misses
DEEP = ("fly.io", "located_in", "aws")         # hop-3
CONTROL = ("atta", "owns", "laptop")           # 1-hop direct hit (control set)
ANCHOR = ("atta", "part_of", "medflow")        # 1-hop direct hit (Source 1)


def _seed_chain(conn) -> None:
    seed_edge(conn, "atta", "part_of", "medflow")
    seed_edge(conn, "medflow", "deploys_to", "fly.io")
    seed_edge(conn, "fly.io", "located_in", "aws")
    seed_edge(conn, "atta", "owns", "laptop")        # direct 1-hop control
    seed_edge(conn, "unrelated", "runs_on", "k8s")   # unrelated island


def _on(cfg, **kw):
    return dataclasses.replace(cfg, graph_multihop_enabled=True, **kw)


def _facts(conn, cfg):
    return _graph_lookup(
        conn, cfg,
        "where is the project atta works on deployed?",
        ["atta"], {}, frozenset(),
        overlap_info={}, embedding_client=None,
    )


def _triples(facts):
    return {(f.subject, f.predicate, f.object): f for f in facts}


# ── G-A1: bridging-edge recall off → on ────────────────────────────────────

def test_bridge_absent_without_multihop(hy, cfg):
    """OFF (default): the 2-hop bridge is not retrieved; direct hits are."""
    _seed_chain(hy.conn)
    got = _triples(_facts(hy.conn, cfg))
    assert BRIDGE not in got
    assert DEEP not in got
    assert ANCHOR in got and CONTROL in got          # Source 1 unaffected


def test_bridge_recovered_with_multihop(hy, cfg):
    """ON: the bridge appears, carrying an honest `fallback:multihop:2hop` chip."""
    _seed_chain(hy.conn)
    got = _triples(_facts(hy.conn, _on(cfg)))
    assert BRIDGE in got, "multi-hop must recover the hop-2 bridge edge"
    assert "fallback:multihop:2hop" in got[BRIDGE].why_retrieved
    # default max_hops=2 → the hop-3 deep edge stays out.
    assert DEEP not in got


def test_control_recall_not_dropped_and_additive(hy, cfg):
    """The additive invariant as a metric: every edge present OFF is still present
    ON at the SAME score, and the bridge is strictly discounted below every
    direct hit (multi-hop adds, never displaces)."""
    _seed_chain(hy.conn)
    off = _triples(_facts(hy.conn, cfg))
    on = _triples(_facts(hy.conn, _on(cfg)))

    assert set(off).issubset(on)                     # nothing lost
    for key, off_fact in off.items():
        assert on[key].score == pytest.approx(off_fact.score, rel=1e-6), key
        assert "multihop" not in " ".join(on[key].why_retrieved), key

    direct_min = min(on[k].score for k in off)       # weakest direct hit
    assert on[BRIDGE].score < direct_min, "bridge must not outrank any direct hit"


def test_multihop_reason_absent_when_disabled(hy, cfg):
    """OFF path never touches the traversal — no multihop reason chips leak."""
    _seed_chain(hy.conn)
    for fact in _facts(hy.conn, cfg):
        assert "multihop" not in " ".join(fact.why_retrieved)


# ── _multihop_edges determinism (decay / min_score / depth / dedup) ─────────

def test_path_score_compounds_with_decay(hy, cfg):
    """hop-2 path_score = conf² · decay² (compounding edge weights)."""
    _seed_chain(hy.conn)
    out = _multihop_edges(hy.conn, _on(cfg), ["atta"])
    assert BRIDGE in out
    assert out[BRIDGE]["hop"] == 2
    expected = (FRESH_CONF ** 2) * (DECAY ** 2)       # = 1/9
    assert out[BRIDGE]["path_score"] == pytest.approx(expected, rel=1e-9)


def test_min_score_prunes_the_bridge(hy, cfg):
    """A min_score above the hop-2 compounding score prunes the bridge."""
    _seed_chain(hy.conn)
    out = _multihop_edges(hy.conn, _on(cfg, graph_multihop_min_score=0.2), ["atta"])
    assert BRIDGE not in out                          # 1/9 ≈ 0.111 < 0.2


def test_max_hops_below_two_disables(hy, cfg):
    """max_hops < 2 short-circuits: no traversal, no edges."""
    _seed_chain(hy.conn)
    assert _multihop_edges(hy.conn, _on(cfg, graph_multihop_max_hops=1), ["atta"]) == {}


def test_depth_control_two_vs_three(hy, cfg):
    """max_hops gates chain length: the hop-3 deep edge appears only at depth 3."""
    _seed_chain(hy.conn)
    at2 = _multihop_edges(hy.conn, _on(cfg, graph_multihop_min_score=0.01), ["atta"])
    assert DEEP not in at2                            # default max_hops=2 stops early

    at3 = _multihop_edges(
        hy.conn, _on(cfg, graph_multihop_max_hops=3, graph_multihop_min_score=0.01),
        ["atta"],
    )
    assert DEEP in at3 and at3[DEEP]["hop"] == 3
    expected = (FRESH_CONF ** 3) * (DECAY ** 3)       # = 1/27 ≈ 0.037
    assert at3[DEEP]["path_score"] == pytest.approx(expected, rel=1e-9)


def test_never_emits_seed_incident_edges(hy, cfg):
    """1-hop edges incident to a seed are Source 1's — the traversal never
    re-emits them (no double-count / mislabel)."""
    _seed_chain(hy.conn)
    out = _multihop_edges(
        hy.conn, _on(cfg, graph_multihop_max_hops=3, graph_multihop_min_score=0.01),
        ["atta"],
    )
    assert ANCHOR not in out and CONTROL not in out
    for (s, _p, o) in out:
        assert s != "atta" and o != "atta"


def test_empty_seeds_returns_nothing(hy, cfg):
    _seed_chain(hy.conn)
    assert _multihop_edges(hy.conn, _on(cfg), []) == {}
