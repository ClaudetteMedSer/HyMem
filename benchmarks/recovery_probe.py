#!/usr/bin/env python3
"""Stage-0 gate for Grove E2 (recovery-rate gauge) — READ-ONLY, LLM-free.

WHAT THIS MEASURES, AND WHY IT IS A DIFF AND NOT A COUNT
--------------------------------------------------------
`phase1.py:654-666` resurrects a retracted edge to `status='active'` but never
clears `invalid_at`, and nothing in `hymem/` ever clears it on
`knowledge_graph` (only `rules.py:407` does, on a different table). So

    !! PREMISE RETRACTED 2026-08-27. The paragraph above was true when this
    !! probe was written and is FALSE as of commit 8c6925c (2026-08-25 20:09
    !! UTC), which clears `invalid_at` on any positive mention -- in BOTH
    !! `phase1._upsert_triple` and `phase3.reinforce`. Consequence for this
    !! instrument: the `recovered` population now SELF-DRAINS. A recovered
    !! edge heals on its next positive mention, so on a live store this probe
    !! reads `recovered = 0` in the healthy steady state, and the strong
    !! `INERT` verdict -- which REQUIRES a non-empty recovered population the
    !! clause still fails to bar -- becomes unreachable rather than unmet.
    !! That is the degenerate-criterion trap this file's own verdict function
    !! was amended to guard against, arriving one level up: a criterion that
    !! reports the same thing on every store. Read `INERT-EMPTY` from this
    !! probe as "no stock right now", NEVER as "the mechanism is empty".
    !! The measured Aug-25 strong-INERT reading STANDS as taken; what does
    !! not stand is any expectation of reproducing it. See the E2 STATUS
    !! block in additional_planning.md for the re-derived verdict.


    status='active' AND derived=0 AND invalid_at IS NOT NULL

is the population of facts that were retracted and have since been re-asserted.
Counting that population OVERSTATES what the defect costs, because
`_anchor_facts` (`aggregate.py:801-835`) -- the ONLY query in the codebase that
reads `invalid_at` on this table -- also requires `pos_evidence > neg_evidence`,
takes only the top `cap`, and lets profile rows consume part of that cap. An
edge only reaches retraction by accumulating negative evidence, so most
recovered edges fail the margin and would be excluded anyway.

The headline is therefore a COUNTERFACTUAL DIFF: run the real anchor query
twice, once as production runs it and once with the `invalid_at IS NULL` clause
removed, and report how many facts the clause actually costs the digest.

    anchor_delta = len(anchor_without_clause) - len(anchor_with_clause)

PRE-REGISTERED READING (banked 2026-08-25, before this was run anywhere)
------------------------------------------------------------------------
  anchor_delta == 0
        The clause is INERT on this store. Grove E2 Stage 1 (the in-dream
        gauge) is closed FAIL-mechanism -- building an instrument over an empty
        population is the unreachable-code-path trap. Plan D may copy the
        `_anchor_facts` predicate verbatim.
  anchor_delta > 0 AND evidence_backed == 0
        Something closes these edges, but not contradicting evidence. NOT a
        recovery finding. Report the buckets and stop; do not quote a rate.
  anchor_delta > 0 AND evidence_backed > 0
        Sized defect. Quote `anchor_delta / cap` as the rate -- never the bare
        count, and never `anchor_delta > 0` as a gate, because ordinary
        decay-then-re-mention churn satisfies any `> 0` bar on any store.
        REQUIRED before the number is quoted: hand-verify >= 3 rows from the
        `evidence_backed` bucket against their kg_evidence polarity/extracted_at
        ordering. Without the hand-check this cannot distinguish genuine
        recovery from value oscillation.

WHAT THIS PROBE STRUCTURALLY CANNOT SEE (so a low number is not safety)
-----------------------------------------------------------------------
  - Recovery by RE-INSERTION. `retention.prune_retracted_edges` hard-deletes
    tombstones 30 days after `last_seen` (`config.py:538`). A fact retracted,
    pruned, then re-learned mints a NEW row with `invalid_at IS NULL`. Invisible
    here, and probably the most common real recovery.
  - `behavioral_dedup.py:260-263` retracts WITHOUT calling stamp_invalidation,
    so those tombstones carry `invalid_at IS NULL` and can never surface as a
    recovery. Counted separately as `unstamped_tombstones`.
  - Canonicalization drift: phase1's re-assert needs an exact
    (subject, predicate, object) match, so an alias re-point mints a new row.
  - The count is a STOCK (currently active, previously closed), not a flow. An
    edge recovered and then re-retracted has left the numerator.

WHAT IS NOT USED, AND WHY
--------------------------
`invalid_at = last_seen` looks like a migration-015 backfill detector and is
NOT one: `phase1.py:658` sets `last_seen = CURRENT_TIMESTAMP` in the same UPDATE
that creates a recovery, so on the numerator that counter reads 0 by
construction -- a degenerate criterion that would report "no artifacts" on every
store. Backfill is detected against the evidence trail instead: a genuine
evidence-driven closure has a matching kg_evidence row with polarity = -1.

Note `prune_chunks` (`retention.py:19-23`) keeps chunks whose evidence belongs
to an ACTIVE edge, so a recovered edge's negative evidence is protected once it
is active again -- but it was unprotected during the retracted window, so
`no_negative_evidence` conflates backfill with cascade-deleted evidence.

Usage (run on the box, from benchmarks/):
  python recovery_probe.py ~/.hermes/hymem.sqlite
  python recovery_probe.py <locomo --db-dir>/conv-26/hymem.sqlite --json out.json

Run it on BOTH a personal store and a benchmark-ingested store: retraction
dynamics in a tech-domain personal store say nothing about LoCoMo state
questions, and Plan D is gated on LoCoMo.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hymem.dreaming.user_profile import load_profile  # noqa: E402

BUCKETS = ("evidence_backed", "value_oscillation", "no_negative_evidence", "unordered")

# The edge half of `_anchor_facts` (aggregate.py:826-834), verbatim apart from
# the clause under test. Keeping the ORDER BY identical matters: the block is a
# top-`cap` list, so the diff depends on the ordering as much as the filter.
_ANCHOR_EDGES = """
    SELECT subject_canonical AS s, predicate AS p, object_canonical AS o
    FROM knowledge_graph
    WHERE status = 'active' AND derived = 0 {clause}
      AND pos_evidence > neg_evidence
    ORDER BY pos_evidence - neg_evidence DESC, last_seen DESC, id
    LIMIT ?
"""

_WITH_CLAUSE = "AND invalid_at IS NULL"
_WITHOUT_CLAUSE = ""


def open_store_readonly(path: str | Path) -> sqlite3.Connection:
    """Open an existing hymem sqlite store strictly read-only (URI mode=ro).

    The probe must never write to the prod store; mode=ro makes any stray write
    raise sqlite3.OperationalError instead of mutating the DB.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"store not found: {p}")
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _anchor_edge_facts(conn: sqlite3.Connection, clause: str, limit: int) -> list[str]:
    if limit <= 0:
        return []
    rows = conn.execute(_ANCHOR_EDGES.format(clause=clause), (limit,)).fetchall()
    return [f"{r['s']} {r['p']} {r['o']}" for r in rows]


def _scalar(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def _classify(conn: sqlite3.Connection, row: sqlite3.Row) -> str:
    """Which cause bucket a recovered edge falls in.

    Only `evidence_backed` may be quoted as genuine recovery: a contradicting
    (polarity=-1) evidence row, and a positive row extracted AFTER the closure
    that could actually have caused the resurrection.
    """
    edge_id, invalid_at = row["id"], row["invalid_at"]

    sibling = conn.execute(
        """
        SELECT 1 FROM knowledge_graph
        WHERE subject_canonical = ? AND predicate = ? AND object_canonical != ?
          AND valid_at IS NOT NULL AND valid_at = ?
        LIMIT 1
        """,
        (row["subject_canonical"], row["predicate"], row["object_canonical"], invalid_at),
    ).fetchone()
    if sibling is not None:
        # value_supersession closes the loser at the WINNER's valid_at, so a
        # sibling opening exactly at this closure is that mechanism, not a
        # too-aggressive retraction gate.
        return "value_oscillation"

    n_neg = int(conn.execute(
        "SELECT COUNT(*) FROM kg_evidence WHERE edge_id = ? AND polarity = -1",
        (edge_id,),
    ).fetchone()[0])
    if n_neg == 0:
        return "no_negative_evidence"

    n_pos_after = int(conn.execute(
        "SELECT COUNT(*) FROM kg_evidence "
        "WHERE edge_id = ? AND polarity = 1 AND extracted_at > ?",
        (edge_id, invalid_at),
    ).fetchone()[0])
    return "evidence_backed" if n_pos_after > 0 else "unordered"


def measure_recovery(conn: sqlite3.Connection, *, cap: int = 20) -> dict:
    """Size what the `invalid_at IS NULL` clause costs the digest anchor.

    Returns the counterfactual diff (`anchor_delta`, the gate), the raw recovered
    stock for context, denominators, and a cause bucket per recovered edge.
    Read-only: issues SELECTs only.
    """
    # Profile rows lead the anchor block and consume the cap first; they are
    # identical in both arms, so only the remaining edge budget can differ.
    n_profile = len(load_profile(conn, cap=cap))
    remaining = cap - n_profile

    with_clause = _anchor_edge_facts(conn, _WITH_CLAUSE, remaining)
    without_clause = _anchor_edge_facts(conn, _WITHOUT_CLAUSE, remaining)
    added = [f for f in without_clause if f not in set(with_clause)]

    recovered_rows = conn.execute(
        "SELECT id, subject_canonical, predicate, object_canonical, invalid_at, "
        "       pos_evidence, neg_evidence, last_seen "
        "FROM knowledge_graph "
        "WHERE status = 'active' AND derived = 0 AND invalid_at IS NOT NULL "
        "ORDER BY pos_evidence - neg_evidence DESC, id"
    ).fetchall()

    buckets = dict.fromkeys(BUCKETS, 0)
    rows: list[dict] = []
    for r in recovered_rows:
        bucket = _classify(conn, r)
        buckets[bucket] += 1
        rows.append({
            "edge_id": r["id"],
            "fact": f"{r['subject_canonical']} {r['predicate']} {r['object_canonical']}",
            "invalid_at": r["invalid_at"],
            "pos_evidence": r["pos_evidence"],
            "neg_evidence": r["neg_evidence"],
            "margin_ok": r["pos_evidence"] > r["neg_evidence"],
            "bucket": bucket,
        })

    return {
        "cap": cap,
        "profile_rows": n_profile,
        "edge_budget": remaining,
        # ── the gate ──
        "anchor_delta": len(without_clause) - len(with_clause),
        "anchor_added": added,
        "anchor_with_clause": len(with_clause),
        "anchor_without_clause": len(without_clause),
        # ── context: the raw stock overstates the impact, see the docstring ──
        "recovered": len(recovered_rows),
        "active_total": _scalar(
            conn, "SELECT COUNT(*) FROM knowledge_graph "
                  "WHERE status = 'active' AND derived = 0"),
        "retracted_total": _scalar(
            conn, "SELECT COUNT(*) FROM knowledge_graph "
                  "WHERE status = 'retracted' AND derived = 0"),
        "unstamped_tombstones": _scalar(
            conn, "SELECT COUNT(*) FROM knowledge_graph "
                  "WHERE status = 'retracted' AND derived = 0 "
                  "AND invalid_at IS NULL"),
        "buckets": buckets,
        "rows": rows,
    }


def _verdict(report: dict) -> tuple[str, str]:
    """Apply the pre-registered reading. Returns (verdict, one-line reason).

    The first branch is a DEGENERACY guard, not part of the pre-registration:
    `_anchor_facts` gives profile rows the whole cap first, so a store with
    >= cap active profile rows leaves NO edge budget, and the diff is then 0
    for arithmetic reasons rather than because the clause is inert. Measured on
    the box 2026-08-25: 20 profile rows against cap=20 -> edge budget 0. Without
    this branch the probe prints the pre-registered "INERT" reading on a store
    that cannot answer the question -- the degenerate-criterion trap, one level
    up from the one this probe was built to avoid.
    """
    if report["edge_budget"] <= 0:
        return ("VACUOUS",
                f"{report['profile_rows']} profile rows consume the whole "
                f"cap={report['cap']}, so the edge budget is 0 and the diff is 0 "
                "by construction -- this store cannot answer the question. NOT "
                "evidence of anything; re-run with a larger --cap or on a store "
                "with a smaller profile")
    if report["anchor_delta"] == 0 and report["recovered"] == 0:
        return ("INERT-EMPTY",
                "no recovered edges exist at all, so the clause has nothing to "
                "bar. A correct close for THIS store, but it says nothing about "
                "a store where retractions actually fire")
    if report["anchor_delta"] == 0:
        return ("INERT",
                f"{report['recovered']} recovered edges exist and the invalid_at "
                "clause still costs the digest nothing; Grove E2 Stage 1 closes "
                "FAIL-mechanism, Plan D may copy the predicate verbatim")
    if report["buckets"]["evidence_backed"] == 0:
        return ("UNATTRIBUTED",
                "the clause excludes facts, but no closure is backed by "
                "contradicting evidence -- report the buckets, do not quote a rate")
    return ("SIZED",
            "sized defect; hand-verify >= 3 evidence_backed rows before quoting "
            f"the rate {report['anchor_delta']}/{report['cap']}")


def _render(report: dict, *, path: str) -> str:
    verdict, reason = _verdict(report)
    b = report["buckets"]
    out = [
        "",
        f"  store             {path}",
        f"  cap               {report['cap']} "
        f"({report['profile_rows']} profile + {report['edge_budget']} edge budget)"
        + ("   !! profile rows consume the entire cap: the digest anchor "
           "contains ZERO graph edges" if report["edge_budget"] <= 0 else ""),
        "",
        "  ── mechanism (read this first) ──",
        f"  anchor with       {report['anchor_with_clause']} facts",
        f"  anchor without    {report['anchor_without_clause']} facts",
        f"  ANCHOR_DELTA      {report['anchor_delta']}"
        f"   ({100.0 * report['anchor_delta'] / max(report['cap'], 1):.1f}% of cap)",
        "",
        "  ── the recovered stock (context; it OVERSTATES the impact) ──",
        f"  recovered         {report['recovered']} "
        f"of {report['active_total']} active non-derived edges",
        f"  retracted         {report['retracted_total']} "
        f"({report['unstamped_tombstones']} never stamped -> invisible to this probe)",
        "",
        "  ── cause buckets over the recovered stock ──",
        f"  evidence_backed      {b['evidence_backed']}"
        "   <- the ONLY bucket quotable as genuine recovery",
        f"  value_oscillation    {b['value_oscillation']}",
        f"  no_negative_evidence {b['no_negative_evidence']}",
        f"  unordered            {b['unordered']}",
        "",
        f"  VERDICT  {verdict} -- {reason}",
    ]
    if report["anchor_added"]:
        out += ["", "  facts the clause costs the digest:"]
        out += [f"    - {f}" for f in report["anchor_added"]]
    if b["evidence_backed"]:
        out += ["", "  hand-verify these against kg_evidence "
                    "(polarity=-1 then polarity=1, extracted_at ascending):"]
        out += [f"    edge {r['edge_id']}: {r['fact']}  (invalid_at {r['invalid_at']})"
                for r in report["rows"] if r["bucket"] == "evidence_backed"][:5]
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="path to an existing hymem.sqlite (opened read-only)")
    ap.add_argument("--cap", type=int, default=20,
                    help="aggregation_digest_anchor_facts (default: 20, the "
                         "production default -- change it only to sweep)")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="also write the full report, including per-row buckets")
    args = ap.parse_args()

    conn = open_store_readonly(args.db)
    try:
        # The box store is live WAL: a dream landing mid-read yields a torn
        # snapshot. data_version changes when another connection commits.
        before = conn.execute("PRAGMA data_version").fetchone()[0]
        report = measure_recovery(conn, cap=args.cap)
        after = conn.execute("PRAGMA data_version").fetchone()[0]
        if before != after:
            print("!! the store was written during the read (data_version "
                  f"{before} -> {after}) -- a dream landed mid-probe. Re-run; "
                  "do not quote this reading.", file=sys.stderr)
            return 2
    finally:
        conn.close()

    print(_render(report, path=args.db))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"  wrote {args.json}\n", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
