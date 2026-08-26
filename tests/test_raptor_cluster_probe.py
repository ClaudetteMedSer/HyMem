"""Offline (Mac, no LLM/box) unit tests for the Phase-2 RAPTOR front-run gate's
pure clustering core (benchmarks/raptor_cluster_probe.py::cluster_episodes).

This is the G5 "cheap offline probe gates the build" discipline: the DB/dream
side of the probe only runs on the Hermes box, but the connected-components
clustering — the logic Phase-2 dreaming/aggregate.py reuses — is a pure function
and is pinned here so the build inherits a verified clusterer.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from raptor_cluster_probe import (  # noqa: E402
    _cosine,
    _jaccard,
    _linked,
    cluster_episodes,
)


def _ep(eid, vector=None, entities=None):
    return {"id": eid, "vector": vector, "entities": set(entities or [])}


# ── similarity primitives ────────────────────────────────────────────────────

def test_cosine_identical_is_one():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_dim_mismatch_and_empty_are_zero():
    assert _cosine([1.0, 0.0], [1.0]) == 0.0
    assert _cosine([], [1.0]) == 0.0


def test_jaccard_basic_and_empty():
    assert _jaccard({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)
    assert _jaccard({"a", "b", "c"}, {"a"}) == pytest.approx(1 / 3)
    assert _jaccard(set(), {"a"}) == 0.0       # empty entity set never links


# ── link predicate: OR of the two signals ───────────────────────────────────

def test_linked_via_embedding_only():
    e1 = _ep("1", vector=[1.0, 0.0], entities=["x"])
    e2 = _ep("2", vector=[0.99, 0.14], entities=["y"])   # no entity overlap
    assert _linked(e1, e2, emb_threshold=0.9, ent_threshold=0.5)


def test_linked_via_entities_only():
    e1 = _ep("1", vector=[1.0, 0.0], entities=["alice", "paris"])
    e2 = _ep("2", vector=[0.0, 1.0], entities=["alice", "paris"])  # orthogonal vecs
    assert _linked(e1, e2, emb_threshold=0.9, ent_threshold=0.5)


def test_not_linked_when_both_signals_weak():
    e1 = _ep("1", vector=[1.0, 0.0], entities=["alice"])
    e2 = _ep("2", vector=[0.0, 1.0], entities=["bob"])
    assert not _linked(e1, e2, emb_threshold=0.9, ent_threshold=0.5)


def test_missing_vector_falls_back_to_entities():
    e1 = _ep("1", vector=None, entities=["alice", "paris"])
    e2 = _ep("2", vector=None, entities=["alice", "paris"])
    assert _linked(e1, e2, emb_threshold=0.9, ent_threshold=0.5)
    assert not _linked(_ep("1", entities=["alice"]), _ep("2", entities=["bob"]),
                       emb_threshold=0.9, ent_threshold=0.5)


# ── connected-components clustering (the gate's core) ────────────────────────

def test_singletons_when_nothing_links():
    eps = [_ep("1", entities=["a"]), _ep("2", entities=["b"]), _ep("3", entities=["c"])]
    labels = cluster_episodes(eps, emb_threshold=0.9, ent_threshold=0.5)
    assert len(set(labels.values())) == 3      # three distinct clusters


def test_transitive_closure_merges_chain():
    # 1—2 share 'a', 2—3 share 'b', 1 and 3 share nothing → still one cluster.
    eps = [
        _ep("1", entities=["a"]),
        _ep("2", entities=["a", "b"]),
        _ep("3", entities=["b"]),
    ]
    labels = cluster_episodes(eps, emb_threshold=0.9, ent_threshold=0.5)
    assert labels["1"] == labels["2"] == labels["3"]
    assert len(set(labels.values())) == 1


def test_two_distinct_clusters():
    eps = [
        _ep("a1", entities=["alice"]),
        _ep("a2", entities=["alice"]),
        _ep("b1", entities=["bob"]),
        _ep("b2", entities=["bob"]),
    ]
    labels = cluster_episodes(eps, emb_threshold=0.9, ent_threshold=0.5)
    assert labels["a1"] == labels["a2"]
    assert labels["b1"] == labels["b2"]
    assert labels["a1"] != labels["b1"]
    assert len(set(labels.values())) == 2


def test_embedding_bridges_clusters_entities_would_not():
    # Entity sets disjoint, but embeddings tie a1 and b1 → one cluster of all four.
    eps = [
        _ep("a1", vector=[1.0, 0.0], entities=["alice"]),
        _ep("a2", vector=[0.2, 0.9], entities=["alice"]),
        _ep("b1", vector=[0.99, 0.14], entities=["bob"]),  # ~a1 by cosine
        _ep("b2", vector=[0.1, 1.0], entities=["bob"]),
    ]
    labels = cluster_episodes(eps, emb_threshold=0.95, ent_threshold=0.5)
    assert labels["a1"] == labels["b1"]        # embedding bridge

    # Raise the bar so the bridge breaks → back to two entity clusters.
    strict = cluster_episodes(eps, emb_threshold=0.999, ent_threshold=0.5)
    assert strict["a1"] != strict["b1"]


def test_empty_input():
    assert cluster_episodes([], emb_threshold=0.9, ent_threshold=0.5) == {}


def test_labels_cover_every_episode():
    eps = [_ep(str(i), entities=[chr(97 + i)]) for i in range(5)]
    labels = cluster_episodes(eps, emb_threshold=0.9, ent_threshold=0.5)
    assert set(labels) == {e["id"] for e in eps}
