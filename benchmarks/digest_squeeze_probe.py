#!/usr/bin/env python3
"""Stage-0 sizing probe for the DIGEST PROFILE SQUEEZE -- READ-ONLY, zero LLM.

WHAT IS BROKEN
--------------
`_anchor_facts` (`hymem/dreaming/aggregate.py:801-836`) builds the root
digest's VERIFIED FACTS block from two sources under ONE shared cap
(`aggregation_digest_anchor_facts`, default 20): profile rows are rendered
first, and

    remaining = cap - len(profile)
    if remaining <= 0:
        return profile                          # :823-824, the early return

returns before the graph-edge query ever runs. `config.py:197` documents that
knob as "Max ACTIVE, non-derived KNOWLEDGE-GRAPH EDGES injected into the root
digest fusion" -- an EDGE budget. The P4 profile tier later began consuming it
without the contract changing, and `_anchor_facts`' own docstring still uses
prioritization language ("profile rows outrank edges", "graph edges fill the
remainder") that the early return silently converts into EXCLUSION.

The squeeze is monotonic: `possession`, `health_condition`,
`recurring_activity` and `relationship` (per `slot_key`) sit outside
`SINGLE_VALUED_SLOTS` (`user_profile.py:80`) and accumulate without bound, so
once a store's active profile crosses the cap it never comes back, and the
root digest fuses on profile alone forever. `facts_hash` (`aggregate.py:1241`)
is then keyed on profile alone too, so a changed graph no longer regenerates
the digest.

WHY THE HEADLINE IS A DIFF AND NOT A COUNT
------------------------------------------
"How many edges does the store have" is not the question -- the block is a
top-`cap` list, so what the defect costs is bounded by the cap, not by the
graph. The probe therefore renders the block TWICE on one snapshot:

  CURRENT  production `_anchor_facts(conn, cap)` itself. Not a copy: the
           probe imports the function under test, so it cannot drift from it.
  FIXED    what the separate-budget fix would render -- `load_profile` capped
           at `profile_cap`, then `select_anchor_edges` capped independently
           at `edge_cap` (`hymem/query/state_anchor.py`, whose predicate is
           already parity-controlled against `_anchor_facts` by
           `tests/test_state_anchor.py:320`).

    edges_restored = edge lines in FIXED that are absent from CURRENT

THIS IS REPORTING PLUS DEGENERACY GUARDS, NOT A BAR
---------------------------------------------------
The box outcome is already known (22 active profile rows against cap 20), so a
"does the diff move" criterion would be a ceiling instrument -- it can only
read PASS. Nothing here is a gate. What the verdicts do is stop a store that
CANNOT answer the question from being read as if it had:

  DISABLED      cap <= 0. The block is off ("0 disables"); no squeeze exists to
                size. Reporting SQUEEZED off an edge budget of 0 here would be
                the degenerate-criterion trap.
  VACUOUS       no anchor-ELIGIBLE edge exists. A store with no edges cannot
                show restoration; its 0 is arithmetic, not evidence. Checked
                FIRST, so it outranks SQUEEZED.
  ZERO-PROFILE  no active profile rows -> the two arms must be identical and
                the diff must be EXACTLY 0. This is the only genuine zero-diff
                control, and it exists only as a fixture: neither real store is
                one.
  NOT-SQUEEZED  0 < profile < cap. NOT a null: a partly-filled profile still
                costs the store every edge past the remaining budget (LoCoMo
                conv-26 at 16/20 leaves a 4-slot budget against 55 edges). The
                verdict means "not TOTAL exclusion", nothing more.
  SQUEEZED      profile >= cap -> edge budget 0, the box's shape.

WHAT THIS PROBE DOES NOT FIX, AND MUST NOT BE READ AS FIXING
------------------------------------------------------------
`load_profile(conn, cap=20)` truncates a 22-row profile to 20 at
`user_profile.py:343`, and those tail rows sort last precisely because the
accumulating slots sort last in `_SLOT_ORDER`. The separate-budget fix does
NOT repair that -- the FIXED arm still renders only `profile_cap` profile
lines. `profile_tail_dropped` is a REPORTING figure for a separate, still-open
defect. It is never folded into `edges_restored`.

CONTENT SAFETY -- A DELIBERATE DIVERGENCE FROM `recovery_probe.py`
------------------------------------------------------------------
`recovery_probe --json` writes raw fact text to disk. This probe does NOT.
Profile rows and graph edges are the user's real conversation content, so
`--json` emits COUNTS AND VERDICT ONLY (an allow-list, pinned by
`test_the_json_payload_carries_no_fact_text`), and the restored lines reach
stdout only behind `--show-restored`, default OFF. An agent-run invocation can
therefore never surface user content.

Usage (run on the BOX -- no store exists on the dev machine):
  python3 benchmarks/digest_squeeze_probe.py ~/.hermes/hymem.sqlite
  python3 benchmarks/digest_squeeze_probe.py <locomo db-dir>/conv-26/hymem.sqlite

Run it on BOTH: the personal store is the squeezed case, conv-26 the partial
one. Neither is a zero-diff control.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hymem.dreaming.aggregate import _anchor_facts  # noqa: E402
from hymem.dreaming.user_profile import (  # noqa: E402
    load_profile,
    render_profile_fact,
)
from hymem.query.state_anchor import select_anchor_edges  # noqa: E402

# Reuse the sibling benchmark's read-only opener rather than a third copy of
# the URI dance (the locomo_adapter/msc_adapter sibling-import idiom).
from recovery_probe import open_store_readonly  # noqa: E402

# The JSON payload's allow-list. Counts, caps and the verdict -- no fact text,
# ever. Adding a key here is a content-safety decision, not a formatting one.
_JSON_KEYS = (
    "cap",
    "edge_cap",
    "profile_cap",
    "n_profile_active",
    "n_profile_rendered",
    "profile_tail_dropped",
    "edge_budget_today",
    "n_edges_active",
    "n_edges_active_total",
    "n_current_facts",
    "n_fixed_facts",
    "edges_restored",
    "profile_lines_restored",
    "verdict",
    "reason",
)

_N_ELIGIBLE_EDGES = """
    SELECT COUNT(*) FROM knowledge_graph
    WHERE status = 'active' AND derived = 0 AND invalid_at IS NULL
      AND pos_evidence > neg_evidence
"""

_N_ACTIVE_EDGES = """
    SELECT COUNT(*) FROM knowledge_graph
    WHERE status = 'active' AND derived = 0
"""


class SnapshotMoved(RuntimeError):
    """Another connection committed while the probe was reading. The box store
    is live WAL, so a dream landing mid-read yields a torn snapshot; the
    reading is discarded rather than quoted."""


def _data_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA data_version").fetchone()[0])


def _edge_lines(conn: sqlite3.Connection, cap: int) -> list[str]:
    """The anchor block's EDGE half, rendered exactly as `_anchor_facts` does.

    Selection is delegated to `state_anchor.select_anchor_edges`, which is the
    verbatim `_anchor_facts` predicate and carries its own parity control
    (`tests/test_state_anchor.py:320`) -- so the probe adds no third copy of
    that SQL.
    """
    return [
        f"{r['subject_canonical']} {r['predicate']} {r['object_canonical']}"
        for r in select_anchor_edges(conn, cap=cap)
    ]


def _profile_lines(conn: sqlite3.Connection, cap: int) -> list[str]:
    """The anchor block's PROFILE half, rendered exactly as `_anchor_facts`
    does. `cap <= 0` -> [] (the `0 disables` convention, held per budget)."""
    if cap <= 0:
        return []
    return [render_profile_fact(e) for e in load_profile(conn, cap=cap)]


def fixed_facts(
    conn: sqlite3.Connection, *, edge_cap: int, profile_cap: int
) -> list[str]:
    """The counterfactual block: SEPARATE budgets, no early return, profile
    rows still first. This is what the gated fix would render; it does not
    exist in `hymem/` yet, which is why the probe builds it here."""
    return _profile_lines(conn, profile_cap) + _edge_lines(conn, edge_cap)


def measure_squeeze(
    conn: sqlite3.Connection,
    *,
    cap: int = 20,
    edge_cap: int | None = None,
    profile_cap: int | None = None,
) -> dict:
    """Size what the shared cap costs the root digest on ONE store.

    `cap` is production's `aggregation_digest_anchor_facts` (the CURRENT arm).
    `edge_cap` / `profile_cap` are the FIXED arm's independent budgets and
    default to `cap`, so the diff isolates the shared-budget/early-return
    behaviour and nothing else. Read-only: SELECTs only.
    """
    edge_cap = cap if edge_cap is None else edge_cap
    profile_cap = cap if profile_cap is None else profile_cap

    n_profile_active = len(load_profile(conn))
    n_profile_rendered = len(load_profile(conn, cap=cap)) if cap > 0 else 0
    n_edges_active = int(conn.execute(_N_ELIGIBLE_EDGES).fetchone()[0])
    n_edges_active_total = int(conn.execute(_N_ACTIVE_EDGES).fetchone()[0])

    # The CURRENT arm is the production function itself -- no copy to drift.
    current = _anchor_facts(conn, cap)
    fixed = fixed_facts(conn, edge_cap=edge_cap, profile_cap=profile_cap)

    # Profile rows lead the block in both arms (pinned by
    # tests/test_user_profile.py:398), so a count split is exact and needs no
    # set membership -- which would be wrong anyway if an edge line ever
    # collided with a profile line.
    current_profile = current[:n_profile_rendered]
    current_edges = current[n_profile_rendered:]
    n_fixed_profile = len(_profile_lines(conn, profile_cap))
    fixed_profile, fixed_edges = fixed[:n_fixed_profile], fixed[n_fixed_profile:]

    seen_edges = set(current_edges)
    restored_lines = [line for line in fixed_edges if line not in seen_edges]
    seen_profile = set(current_profile)

    report = {
        "cap": cap,
        "edge_cap": edge_cap,
        "profile_cap": profile_cap,
        "n_profile_active": n_profile_active,
        "n_profile_rendered": n_profile_rendered,
        # REPORTING ONLY -- a separate, still-open defect (user_profile.py:343).
        # Never a restoration figure; the fix does not repair it.
        "profile_tail_dropped": max(0, n_profile_active - n_profile_rendered),
        "edge_budget_today": len(current_edges),
        "n_edges_active": n_edges_active,
        "n_edges_active_total": n_edges_active_total,
        "n_current_facts": len(current),
        "n_fixed_facts": len(fixed),
        # ── the headline ──
        "edges_restored": len(restored_lines),
        "profile_lines_restored": len(
            [line for line in fixed_profile if line not in seen_profile]
        ),
        # Content: stdout only, behind --show-restored. Never in the JSON.
        "restored_lines": restored_lines,
    }
    report["verdict"], report["reason"] = _verdict(report)
    return report


def measure_snapshot(conn: sqlite3.Connection, **kwargs) -> dict:
    """`measure_squeeze` bracketed by a `PRAGMA data_version` check.

    The box store is live WAL: a dream committing mid-read would give a
    half-old, half-new block and a diff that means nothing. Raises
    `SnapshotMoved` instead of returning it.
    """
    before = _data_version(conn)
    report = measure_squeeze(conn, **kwargs)
    after = _data_version(conn)
    if before != after:
        raise SnapshotMoved(f"data_version {before} -> {after}")
    return report


def _verdict(report: dict) -> tuple[str, str]:
    """Which reading this store supports. Ordered -- the first matching branch
    wins, and the order is load-bearing: VACUOUS outranks SQUEEZED because a
    store with no eligible edges shows a 0 diff for arithmetic reasons, and
    ZERO-PROFILE outranks NOT-SQUEEZED because 0 is also < cap.

    None of these is a bar. They exist so a store that cannot answer the
    question is never read as an answer.
    """
    if report["cap"] <= 0:
        return ("DISABLED",
                "cap=0 disables the VERIFIED FACTS block entirely, so there is "
                "no shared budget and no squeeze to size on this store")
    if report["n_edges_active"] == 0:
        return ("VACUOUS",
                "no anchor-eligible graph edge exists (active, non-derived, "
                "invalid_at IS NULL, pos>neg), so restoration has nothing to "
                "restore -- this store cannot answer the question, and its 0 "
                "is NOT evidence that the squeeze is harmless")
    if report["n_profile_active"] == 0:
        return ("ZERO-PROFILE",
                f"no active profile rows, so both arms must be identical; diff "
                f"reads {report['edges_restored']} (anything but 0 means the "
                "probe is measuring something other than the squeeze). This is "
                "the genuine zero-diff control -- and only a fixture is one")
    if report["n_profile_active"] < report["cap"]:
        return ("NOT-SQUEEZED",
                f"{report['n_profile_active']} profile rows against cap="
                f"{report['cap']} leave an edge budget of "
                f"{report['edge_budget_today']}: the block is not TOTALLY "
                f"profile-only, but the shared cap still costs it "
                f"{report['edges_restored']} edges. NOT a null control")
    return ("SQUEEZED",
            f"{report['n_profile_active']} active profile rows fill cap="
            f"{report['cap']}, so the early return fires and the digest fuses "
            f"on profile alone: {report['edges_restored']} of "
            f"{report['n_edges_active']} eligible edges never reach it, and "
            "facts_hash is keyed on profile alone, so a changed graph no "
            "longer regenerates the digest")


def _json_payload(report: dict) -> dict:
    """Counts and verdict only -- see the module docstring's content-safety
    note. Built from an allow-list so a field added to the report later cannot
    leak fact text into a file by default."""
    return {k: report[k] for k in _JSON_KEYS if k in report}


def _render(report: dict, *, path: str, show_restored: bool = False) -> str:
    out = [
        "",
        f"  store             {path}",
        f"  cap               {report['cap']}"
        f"   (fix would use edge_cap={report['edge_cap']}, "
        f"profile_cap={report['profile_cap']})",
        "",
        "  ── the block as rendered TODAY ──",
        f"  profile active    {report['n_profile_active']}"
        f"   ({report['n_profile_rendered']} rendered)",
        f"  edge budget       {report['edge_budget_today']}"
        + ("   !! profile rows consume the entire cap: the digest anchor "
           "holds ZERO graph edges" if report["edge_budget_today"] == 0
           and report["cap"] > 0 else ""),
        f"  facts in block    {report['n_current_facts']}",
        "",
        "  ── the counterfactual (separate budgets) ──",
        f"  eligible edges    {report['n_edges_active']}"
        f"   (of {report['n_edges_active_total']} active non-derived)",
        f"  facts in block    {report['n_fixed_facts']}",
        f"  EDGES_RESTORED    {report['edges_restored']}",
        "",
        "  ── separate, still-open defect (NOT repaired by the fix) ──",
        f"  profile tail dropped  {report['profile_tail_dropped']}"
        "   (user_profile.py:343; reporting only, never a restored count)",
        "",
        f"  VERDICT  {report['verdict']} -- {report['reason']}",
    ]
    if report["restored_lines"]:
        if show_restored:
            out += ["", "  edges the shared cap costs the digest:"]
            out += [f"    - {line}" for line in report["restored_lines"]]
        else:
            out += ["",
                    f"  ({report['edges_restored']} restored edge lines "
                    "withheld: they are the user's conversation content. "
                    "Pass --show-restored to print them locally.)"]
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("db", help="path to an existing hymem.sqlite (opened read-only)")
    ap.add_argument("--cap", type=int, default=20,
                    help="aggregation_digest_anchor_facts as production runs it "
                         "(default: 20, the shipped default -- change it "
                         "only to sweep)")
    ap.add_argument("--edge-cap", type=int, default=None,
                    help="the fix's independent EDGE budget (default: --cap)")
    ap.add_argument("--profile-cap", type=int, default=None,
                    help="the fix's independent PROFILE budget (default: --cap)")
    ap.add_argument("--show-restored", action="store_true",
                    help="print the restored fact lines. They are real "
                         "conversation content -- local terminal only, never "
                         "into a file, a log or an agent context")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="write COUNTS AND VERDICT ONLY (no fact text, ever)")
    args = ap.parse_args(argv)

    conn = open_store_readonly(args.db)
    try:
        report = measure_snapshot(conn, cap=args.cap, edge_cap=args.edge_cap,
                                  profile_cap=args.profile_cap)
    except SnapshotMoved as exc:
        print(f"!! the store was written during the read ({exc}) -- a dream "
              "landed mid-probe. Re-run; do not quote this reading.",
              file=sys.stderr)
        return 2
    finally:
        conn.close()

    print(_render(report, path=args.db, show_restored=args.show_restored))
    if args.json:
        Path(args.json).write_text(json.dumps(_json_payload(report), indent=2))
        print(f"  wrote {args.json} (counts only)\n", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
