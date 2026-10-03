"""Grove E4's null model, checked before anything is built on it.

E4 proposed gating `consolidate_insights`' hub candidates on "domain-label
shuffling (episode/domain membership permuted)". The hub statistic is
`GROUP BY object_canonical HAVING COUNT(*) >= 2` -- a function of the OBJECT
degree distribution and nothing else. Permuting subject or domain labels leaves
every object's degree exactly where it was, so the recomputed statistic is
identical on every permutation. A gate calibrated against that null accepts and
rejects precisely what it would have without one, while printing a
calibrated-looking alpha.

The load-bearing test is `test_the_proposed_null_cannot_move_the_statistic`:
it asserts the invariance directly rather than letting the probe report a
distribution nobody checks is non-degenerate.

Measured on the box store (2026-09-01): 70 eligible edges, 49 subjects, 58
objects, 9 observed hubs; subject-shuffle distribution = {9} over 2000
permutations; object-reassignment null mean 19.7 with p05 17. The observed
count is BELOW its own null, so the false-discovery defence E4 imports has
nothing to defend against on this corpus.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from consolidation_null_probe import (  # noqa: E402
    hub_count,
    probe,
    shuffle_objects,
    shuffle_subjects,
)


def _star(n_subjects, obj="hub"):
    """One object depended on by n distinct subjects."""
    return [(f"s{i}", obj) for i in range(n_subjects)]


def _matching(n):
    """n subjects, n objects, one edge each — no hub is possible."""
    return [(f"s{i}", f"o{i}") for i in range(n)]


# ------------------------------------------------------------ statistic ----

def test_hub_count_is_the_features_own_rule():
    assert hub_count(_star(3)) == 1
    assert hub_count(_matching(10)) == 0
    assert hub_count(_star(2) + _matching(5)) == 1


def test_a_single_subject_twice_still_counts_as_a_hub():
    """`COUNT(*) >= 2` counts EDGES, not distinct subjects, so one subject with
    two edges renders as 'a shared dependency of: X, X'. Latent on the box
    today (all nine hubs have distinct subjects) and pinned so a future store
    that hits it is not read as a real hub."""
    assert hub_count([("s0", "o"), ("s0", "o")]) == 1


# ----------------------------------------------------------- invariance ----

def test_the_proposed_null_cannot_move_the_statistic():
    """THE test this probe exists for. E4's null is a permutation of subject
    labels; the statistic does not read them."""
    rng = random.Random(0)
    edges = _star(4) + _star(3, "hub2") + _matching(20)
    before = hub_count(edges)
    seen = {hub_count(shuffle_subjects(edges, rng)) for _ in range(500)}
    assert seen == {before}, "a null that moves the statistic would falsify this"


def test_shuffle_subjects_does_permute_something():
    """Guards the test above from passing because the shuffle is a no-op."""
    rng = random.Random(1)
    edges = [(f"s{i}", f"o{i}") for i in range(50)]
    assert shuffle_subjects(edges, rng) != edges


def test_the_object_null_does_move_the_statistic():
    rng = random.Random(0)
    edges = _matching(60)
    seen = {hub_count(shuffle_objects(edges, sorted({o for _s, o in edges}), rng))
            for _ in range(200)}
    assert len(seen) > 1
    assert 0 not in seen or len(seen) > 1


# -------------------------------------------------------------- reading ----

def test_a_dispersed_graph_reads_DEFICIT():
    """The box's shape: many objects, few repeats. Chance collisions in 58 bins
    beat what the extractor actually found."""
    r = probe(_matching(70), trials=2000, seed=0)
    assert "DEFICIT" in r["reading"]
    assert "delete every candidate" in r["reading"]


def _two_regular(n_objects):
    """Each object depended on by exactly two distinct subjects."""
    return [(f"s{i}", f"o{i // 2}") for i in range(2 * n_objects)]


def test_an_evenly_paired_graph_reads_EXCESS():
    """The reading must be reachable, or the probe cannot tell a corpus that
    needs E4's defence from one that does not."""
    r = probe(_two_regular(30), trials=2000, seed=0)
    assert "EXCESS" in r["reading"]


def test_the_two_readings_are_reachable_from_one_probe():
    """Both branches on the same code path, so neither is dead."""
    a = probe(_matching(70), trials=1000, seed=0)["reading"]
    b = probe(_two_regular(30), trials=1000, seed=0)["reading"]
    assert a != b


def test_a_real_hub_makes_the_statistic_SMALLER_not_larger():
    """The finding that decides how any verdict here may be read.

    `COUNT(*) >= 2` counts OBJECTS carrying two or more edges, so it is maximal
    at an even 2-pairing and falls off on both sides. A star -- one object
    everything depends on, the thing "hub" names -- concentrates 60 edges onto
    ONE object and scores 1, below its own null. EXCESS therefore means "more
    evenly paired than chance", not "more clustered than chance", which is a
    mismatch with the coincidental-CLUSTER framing E4 imports."""
    star = [(f"s{i}", "h") for i in range(60)]
    assert hub_count(star) == 1
    assert hub_count(_two_regular(30)) == 30
    assert "DEFICIT" in probe(star + _matching(3), trials=1000, seed=0)["reading"]


def test_the_probe_reports_the_invariance_it_found():
    r = probe(_star(4) + _matching(20), trials=500, seed=0)
    assert r["subject_shuffle_distinct_values"] == [r["observed_hubs"]]


def test_probe_records_its_own_parameters():
    """A stored result that cannot say which seed or trial count produced it is
    not reproducible."""
    r = probe(_matching(30), trials=750, seed=7)
    assert r["trials"] == 750 and r["seed"] == 7 and r["min_edges"] == 2
