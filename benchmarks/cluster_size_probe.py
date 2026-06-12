#!/usr/bin/env python3
"""Stage-3a chaining-guard probe: cluster-size distribution on a dreamed store.

Connected-components over OR-links (cosine over episode embeddings OR jaccard
over key entities — `hymem.dreaming.aggregate._linked`) chains TRANSITIVELY:
A~B and B~C put A and C in one cluster even when A and C share nothing. On a
real store that can snowball into one mega-cluster whose fusion is mush. Before
`aggregation_nodes_enabled` flips on in prod (raptor_digest_plan.md, Stage 3a),
this probe measures the cluster-size distribution on the PROD store:

  max cluster size < --cap (default 15, the plan's mega-cluster line)
      → chaining is bounded in practice → the guard is moot at this cap.
  max cluster size >= --cap
      → mega-cluster(s) exist → the chaining guard handles them at build time.

The guard is BUILT (2026-06-12, after this probe found a 348-episode component
spanning 61 sessions on the prod store): `cluster_episodes` splits over-cap
components into recency-ordered windows, governed by the config knob
`aggregation_max_cluster_size` (default 15; 0 = uncapped), salt `cluster.v3`.
This probe deliberately keeps calling the clusterer UNCAPPED so it measures RAW
transitive chaining — i.e. what the guard would split — not the post-guard
distribution.

Offline, LLM-less, and READ-ONLY: the store is opened via sqlite URI mode=ro,
so the probe can point at the live prod file without any risk of writing to it.
The input must be a store a dream pass already ran over (episodes +
episode_embeddings populated) — the probe clusters what dreaming produced, it
does not dream.

Like benchmarks/raptor_cluster_probe.py, the loader and the clusterer are
RE-EXPORTED VERBATIM from hymem.dreaming.aggregate (the canonical home), so the
components measured here are exactly the RAW components production forms before
its `max_cluster_size` window split (the probe calls the clusterer without the
cap on purpose) — probe and prod can never silently drift. The default
thresholds are the production `HyMemConfig` defaults for the same reason.

Usage (on the box, against the prod store):
  python cluster_size_probe.py ~/.hermes/hymem.sqlite
  # threshold sweep around the production point:
  python cluster_size_probe.py ~/.hermes/hymem.sqlite --grid 0.55:0.50,0.65:0.50,0.75:0.40
  # per-cluster dump for inspecting a mega-cluster's membership:
  python cluster_size_probe.py ~/.hermes/hymem.sqlite --json /tmp/cluster_sizes.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Re-exported production core (canonical home: hymem.dreaming.aggregate). The
# probe adds ONLY read-only measurement around these — no local clustering.
# ─────────────────────────────────────────────────────────────────────────────
from hymem.config import HyMemConfig
from hymem.dreaming.aggregate import cluster_episodes, load_clusterable_episodes

# Default thresholds = the production defaults `build_aggregation_nodes` runs
# with (cfg.aggregation_emb_threshold / cfg.aggregation_ent_threshold), pulled
# from the dataclass so a config change re-points the probe automatically.
DEFAULT_EMB_THRESHOLD: float = HyMemConfig.__dataclass_fields__[
    "aggregation_emb_threshold"].default
DEFAULT_ENT_THRESHOLD: float = HyMemConfig.__dataclass_fields__[
    "aggregation_ent_threshold"].default

# The plan's mega-cluster line, and now the default of the BUILT guard's knob
# (HyMemConfig.aggregation_max_cluster_size).
DEFAULT_CAP: int = 15

# Histogram buckets: (lo, hi inclusive, label); hi=None means unbounded.
_BUCKETS: tuple[tuple[int, int | None, str], ...] = (
    (1, 1, "1"),
    (2, 4, "2-4"),
    (5, 9, "5-9"),
    (10, 14, "10-14"),
    (15, None, "15+"),
)

VERDICT_SKIP = "no mega-cluster at this cap; chaining guard has nothing to split"
VERDICT_GUARD = ("mega-cluster present → the BUILT chaining guard splits it at "
                 "build time into recency windows of aggregation_max_cluster_size "
                 "(salt cluster.v3); this probe shows the RAW uncapped chaining")


def open_store_readonly(path: Path) -> sqlite3.Connection:
    """Open an existing hymem sqlite store strictly read-only (URI mode=ro).

    The probe may point at the LIVE prod store, so it must be impossible for it
    to write: mode=ro makes every write a sqlite-level error, and we never take
    locks a writer would block on beyond sqlite's own read locking.
    """
    if not path.exists():
        raise FileNotFoundError(f"store not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row     # load_clusterable_episodes reads by name
    has_episodes = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='episodes'"
    ).fetchone()
    if not has_episodes:
        conn.close()
        raise RuntimeError(
            f"{path} has no `episodes` table — not a (current-schema) hymem "
            f"store. The probe needs a store a dream pass has run over."
        )
    return conn


def _histogram(sizes: list[int]) -> dict[str, int]:
    hist = {label: 0 for _, _, label in _BUCKETS}
    for s in sizes:
        for lo, hi, label in _BUCKETS:
            if s >= lo and (hi is None or s <= hi):
                hist[label] += 1
                break
    return hist


def probe_cluster_sizes(
    conn: sqlite3.Connection,
    emb_threshold: float = DEFAULT_EMB_THRESHOLD,
    ent_threshold: float = DEFAULT_ENT_THRESHOLD,
    cap: int = DEFAULT_CAP,
) -> dict:
    """Cluster the store's episodes with the PRODUCTION clusterer and report the
    cluster-size distribution plus the Stage-3a verdict.

    Read-only: only SELECTs through `load_clusterable_episodes`. Returns a dict
    with n_episodes, n_clusters, histogram (fixed buckets 1 / 2-4 / 5-9 / 10-14
    / 15+), max/mean cluster size, the largest cluster's members (episode ids +
    distinct session ids, so a mega-cluster is inspectable without re-running),
    a per-cluster list (for --json dumps), `guard_needed`, and `verdict`.
    """
    episodes = load_clusterable_episodes(conn)
    labels = cluster_episodes(episodes, emb_threshold, ent_threshold)

    grouped: dict[int, list[dict]] = {}
    for ep in episodes:
        grouped.setdefault(labels[ep["id"]], []).append(ep)

    clusters = [
        {
            "label": label,
            "size": len(members),
            "episode_ids": [m["id"] for m in members],
            "session_ids": sorted({m["session_id"] for m in members}),
        }
        for label, members in sorted(grouped.items())
    ]
    clusters.sort(key=lambda c: (-c["size"], c["episode_ids"][0]))

    sizes = [c["size"] for c in clusters]
    max_size = max(sizes, default=0)
    mean_size = (sum(sizes) / len(sizes)) if sizes else 0.0
    largest = clusters[0] if clusters else None
    guard_needed = max_size >= cap

    return {
        "emb_threshold": emb_threshold,
        "ent_threshold": ent_threshold,
        "cap": cap,
        "n_episodes": len(episodes),
        "n_clusters": len(clusters),
        "histogram": _histogram(sizes),
        "max_cluster_size": max_size,
        "mean_cluster_size": mean_size,
        "largest_cluster": largest,
        "clusters": clusters,
        "guard_needed": guard_needed,
        "verdict": VERDICT_GUARD if guard_needed else VERDICT_SKIP,
    }


def _print_report(rep: dict) -> None:
    print(f"  emb≥{rep['emb_threshold']:.2f} OR ent≥{rep['ent_threshold']:.2f} "
          f"(cap={rep['cap']}):")
    print(f"    episodes={rep['n_episodes']}   clusters={rep['n_clusters']}   "
          f"max size={rep['max_cluster_size']}   "
          f"mean size={rep['mean_cluster_size']:.2f}")
    hist = "   ".join(f"{label}: {rep['histogram'][label]}"
                      for _, _, label in _BUCKETS)
    print(f"    size histogram:  {hist}")
    largest = rep["largest_cluster"]
    if largest:
        sids = ", ".join(largest["session_ids"])
        print(f"    largest cluster ({largest['size']} episodes) spans "
              f"{len(largest['session_ids'])} session(s): {sids}")
    print(f"    VERDICT: {rep['verdict']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("store", type=Path,
                    help="path to an existing DREAMED hymem sqlite store "
                         "(opened read-only; never written)")
    ap.add_argument("--emb-threshold", type=float, default=DEFAULT_EMB_THRESHOLD,
                    help=f"cosine link threshold (default: production "
                         f"{DEFAULT_EMB_THRESHOLD})")
    ap.add_argument("--ent-threshold", type=float, default=DEFAULT_ENT_THRESHOLD,
                    help=f"entity-jaccard link threshold (default: production "
                         f"{DEFAULT_ENT_THRESHOLD})")
    ap.add_argument("--grid", default=None,
                    help="sweep 'emb:ent,emb:ent,...' (overrides single thresholds)")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP,
                    help=f"mega-cluster line: max size >= cap → guard needed "
                         f"(default {DEFAULT_CAP}, per raptor_digest_plan.md 3a)")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="write per-cluster records (one block per grid point) "
                         "to this json path")
    args = ap.parse_args(argv)

    grid = (
        [tuple(float(x) for x in pair.split(":")) for pair in args.grid.split(",")]
        if args.grid else [(args.emb_threshold, args.ent_threshold)]
    )

    print("\nStage-3a chaining-guard probe (cluster sizes on the prod store)")
    print(f"  Store: {args.store}   (read-only)\n", flush=True)

    try:
        conn = open_store_readonly(args.store)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        blocks: list[dict] = []
        for emb_t, ent_t in grid:
            rep = probe_cluster_sizes(conn, emb_t, ent_t, args.cap)
            print("=" * 64)
            _print_report(rep)
            print()
            blocks.append(rep)
    finally:
        conn.close()

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"store": str(args.store), "cap": args.cap, "grid": blocks},
            indent=2,
        ))
        print(f"Per-cluster records written to {args.json}\n")

    print("VERDICT GUIDE:")
    print(f"  max cluster size < cap on the production thresholds "
          f"(emb {DEFAULT_EMB_THRESHOLD} / ent {DEFAULT_ENT_THRESHOLD})")
    print(f"      → {VERDICT_SKIP}")
    print("  any mega-cluster at/above cap")
    print(f"      → {VERDICT_GUARD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
