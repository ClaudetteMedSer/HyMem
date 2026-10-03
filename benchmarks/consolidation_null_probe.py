#!/usr/bin/env python3
"""Grove E4 front-run probe: does the consolidation gate need a null model?

READ-ONLY, zero LLM, offline. Point it at a hymem.sqlite.

WHAT E4 PROPOSED
----------------
`consolidate_insights` (`hymem/dreaming/phase2.py:87`) surfaces "Project
Insights" into MEMORY.md. Its first family is HUBS: an object with >= 2 active,
non-derived `depends_on` edges above a confidence floor, rendered as "`X` is a
shared dependency of: A, B". Grove E4 proposed gating those candidates on a
null model -- "domain-label shuffling (episode/domain membership permuted)
recomputes the detection metric distribution" -- to operationalise the paper's
false-discovery-scaling defence, since coincidence grows superlinearly with
corpus size.

THE FIRST FINDING IS THAT THE PROPOSED NULL CANNOT MOVE THE STATISTIC
--------------------------------------------------------------------
The hub count is `GROUP BY object_canonical HAVING COUNT(*) >= 2`. It is a
function of the OBJECT degree distribution and nothing else. Permuting subject
(or domain) labels rearranges which subjects sit under which object; it leaves
every object's degree exactly where it was, so the recomputed statistic is
identical on every permutation, by construction. A gate calibrated against that
null accepts and rejects precisely what it would have without one -- and prints
a calibrated-looking alpha while doing it. `--demo-invariance` runs the shuffle
and shows the distribution collapsing to a point.

A NULL THAT CAN MOVE
--------------------
To ask "is this object a shared dependency, or a coincidence of who happened to
get talked about?", the null has to break the object degree distribution too:
reassign each edge's object uniformly over the observed object vocabulary and
recount. That is the balls-in-bins null this probe reports.

WHAT THE STATISTIC ACTUALLY MEASURES (check this before reading a verdict)
-------------------------------------------------------------------------
"Number of objects with >= 2 edges" is NOT a concentration measure, and the
name "hub" invites reading it as one. It peaks at an even 2-pairing and falls
away on BOTH sides. Measured, 60-63 edges, 3000 trials each:

    shape                       observed   null mean   reading
    star (3 objects take all)          3        6.00   DEFICIT
    2-regular (30 objects x 2)        30       17.90   EXCESS
    matching (60 objects x 1)          0       15.80   DEFICIT

So a genuine hub -- one object everything depends on -- makes this statistic
SMALLER, and reads as a deficit. EXCESS means "more evenly paired than chance",
not "more clustered than chance". That is a mismatch with the false-discovery
framing E4 imports, which is about coincidental CLUSTERS, and it is worth
settling before any alpha is calibrated on it.

READING THE RESULT
------------------
  observed >> null   the detector surfaces more shared dependencies than
                     chance; E4's defence is warranted and a calibrated alpha
                     is worth building.
  observed ~= null   the candidates are indistinguishable from coincidence;
                     surfacing them at all is the thing to reconsider.
  observed << null   the graph is MORE DISPERSED than chance -- objects are
                     mostly mentioned once, there is no excess of shared
                     dependencies, and a false-discovery gate has nothing to
                     defend against. Suppressing at alpha would delete every
                     candidate, real ones included.

Usage:
  python consolidation_null_probe.py ~/.hermes/hymem.sqlite
  python consolidation_null_probe.py store.sqlite --trials 20000 --demo-invariance
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

# Mirrors phase2.consolidate_insights' hub query exactly. If that query moves,
# this probe is measuring something the feature no longer does.
HUB_CONFIDENCE = 0.6
HUB_MIN_EDGES = 2

EDGE_SQL = """
    SELECT subject_canonical AS s, object_canonical AS o
    FROM knowledge_graph
    WHERE predicate = 'depends_on'
      AND status = 'active'
      AND derived = 0
      AND (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) > ?
"""


def open_store_readonly(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_edges(conn: sqlite3.Connection,
               confidence: float = HUB_CONFIDENCE) -> list[tuple[str, str]]:
    return [(r["s"], r["o"]) for r in conn.execute(EDGE_SQL, (confidence,))]


def hub_count(edges, min_edges: int = HUB_MIN_EDGES) -> int:
    """The feature's own statistic: objects carrying >= min_edges edges."""
    deg: dict[str, int] = {}
    for _s, o in edges:
        deg[o] = deg.get(o, 0) + 1
    return sum(1 for v in deg.values() if v >= min_edges)


def shuffle_subjects(edges, rng) -> list[tuple[str, str]]:
    """E4's proposed null: permute the subject labels, objects held fixed."""
    subs = [s for s, _o in edges]
    rng.shuffle(subs)
    return [(s, o) for s, (_old, o) in zip(subs, edges)]


def shuffle_objects(edges, objects, rng) -> list[tuple[str, str]]:
    """A null that reaches the statistic: reassign each edge's object
    uniformly over the observed object vocabulary."""
    return [(s, objects[rng.randrange(len(objects))]) for s, _o in edges]


def probe(edges, trials: int = 20000, seed: int = 0,
          min_edges: int = HUB_MIN_EDGES) -> dict:
    rng = random.Random(seed)
    observed = hub_count(edges, min_edges)
    objects = sorted({o for _s, o in edges})
    subjects = sorted({s for s, _o in edges})

    dist = sorted(hub_count(shuffle_objects(edges, objects, rng), min_edges)
                  for _ in range(trials))
    n = len(dist) or 1
    ge = sum(1 for x in dist if x >= observed) / n
    le = sum(1 for x in dist if x <= observed) / n

    # E4's own null, run so its invariance is measured rather than asserted.
    inv = sorted({hub_count(shuffle_subjects(edges, rng), min_edges)
                  for _ in range(min(trials, 2000))})

    if not dist:
        reading = "no trials"
    elif ge <= 0.05:
        reading = ("EXCESS — the detector surfaces more shared dependencies "
                   "than chance. E4's false-discovery defence is warranted.")
    elif le <= 0.05:
        reading = ("DEFICIT — the graph is MORE DISPERSED than chance: objects "
                   "are mostly mentioned once and there is no excess of shared "
                   "dependencies for a null-model gate to filter. Suppressing "
                   "at alpha would delete every candidate, the real ones "
                   "included.")
    else:
        reading = ("INDISTINGUISHABLE — the candidates sit inside the null "
                   "band. The question is not how to filter them but whether "
                   "to surface them at all.")

    return {
        "edges": len(edges), "subjects": len(subjects), "objects": len(objects),
        "observed_hubs": observed,
        "trials": trials, "seed": seed, "min_edges": min_edges,
        "null_mean": (sum(dist) / n) if dist else None,
        "null_p05": dist[int(0.05 * n)] if dist else None,
        "null_median": dist[n // 2] if dist else None,
        "null_p95": dist[min(int(0.95 * n), n - 1)] if dist else None,
        "p_null_ge_observed": ge,
        "p_null_le_observed": le,
        "subject_shuffle_distinct_values": inv,
        "reading": reading,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="path to an existing hymem.sqlite (read-only)")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confidence", type=float, default=HUB_CONFIDENCE)
    ap.add_argument("--min-edges", type=int, default=HUB_MIN_EDGES)
    ap.add_argument("--demo-invariance", action="store_true",
                    help="print the subject-shuffle distribution in full")
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    conn = open_store_readonly(a.db)
    try:
        edges = load_edges(conn, a.confidence)
    finally:
        conn.close()
    if not edges:
        print("\nNo eligible `depends_on` edges — the hub family is empty on "
              "this store, so there is no population for E4 to gate.")
        return 0

    r = probe(edges, a.trials, a.seed, a.min_edges)
    print("\nConsolidation null-model probe (Grove E4 front-run)")
    print(f"  eligible depends_on edges: {r['edges']}  "
          f"({r['subjects']} subjects, {r['objects']} objects)")
    print(f"  observed hubs (>= {r['min_edges']} edges): {r['observed_hubs']}")
    print(f"\n  E4's proposed null (permute SUBJECT labels), "
          f"{min(a.trials, 2000)} permutations:")
    vals = r["subject_shuffle_distinct_values"]
    print(f"    distinct values of the statistic: {vals}")
    if len(vals) == 1:
        print("    → the statistic is INVARIANT under this null. It is a "
              "function of the\n      OBJECT degree distribution, which "
              "permuting subjects does not touch, so a\n      gate calibrated "
              "here accepts and rejects exactly what it would without one.")
    if a.demo_invariance:
        print(f"    (full set printed above; {min(a.trials, 2000)} draws)")
    print(f"\n  A null that reaches the statistic (reassign OBJECTS "
          f"uniformly), {r['trials']} trials:")
    print(f"    null mean {r['null_mean']:.2f}   "
          f"p05/median/p95 {r['null_p05']}/{r['null_median']}/{r['null_p95']}")
    print(f"    P(null >= observed) = {r['p_null_ge_observed']:.4f}   "
          f"P(null <= observed) = {r['p_null_le_observed']:.4f}")
    print(f"\n  READING: {r['reading']}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
