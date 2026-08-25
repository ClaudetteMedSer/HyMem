#!/usr/bin/env python3
"""Stage-1 instrument for the DIGEST PROFILE SQUEEZE -- READ-ONLY, LLM-LESS.

WHAT THIS IS
------------
Stage 0 (`benchmarks/digest_squeeze_probe.py`) sizes how many graph edges the
shared anchor cap costs the root digest. It cannot say whether those edges are
worth anything: no scored benchmark consumes the digest -- `locomo_adapter`,
`longmemeval_adapter`, `msc_adapter` and `beam_adapter` contain zero references
to it -- so there is no A/B for this change at any budget. The gate is
therefore built on FAITHFULNESS and NON-REGRESSION of the digest text itself,
hand-scored, and its verdict must never be quoted as a score.

This script renders both root-fusion prompts over the SAME tree on the SAME
snapshot and prints them. You paste them into the box LLM yourself.

  ARM A  CURRENT -- `_anchor_facts(conn, cap)` exactly as production calls it:
                    one shared cap, profile rows first, early return at
                    `aggregate.py:823-824`.
  ARM B  FIXED   -- the counterfactual: independent profile and edge budgets,
                    no early return (`digest_squeeze_probe.fixed_facts`).

Everything outside the VERIFIED FACTS block is byte-identical between the arms,
so a scored difference is attributable to the squeeze and to nothing else.

IT MUST NOT CALL A MODEL, AND IT MUST NOT WRITE A FILE
-------------------------------------------------------
Mirrors `benchmarks/profile_prompt_dump.py`: read-only (`mode=ro`), no LLM
client imported anywhere (pinned by an AST test), no output path argument at
all. The rendered prompts contain the user's real digest, profile and
conversation content -- they go to the human's terminal and nowhere else:
never a repo artifact, never a log, never back into an agent context.

WHY THE GUARDS, WHEN THE SCORER IS A HUMAN
-------------------------------------------
A ceiling instrument, a degenerate criterion and an unreachable code path all
read as PASS. So the dump refuses rather than prints when it cannot pose the
question:

  no root digest / no member resolves -> an EMPTY prompt. A human scoring one
      produces a confident constant. Refused (rc=1).
  the two facts blocks are IDENTICAL -> nothing to score; S1-C1 would read 0
      for a reason unrelated to the fix. This is what a zero-profile store
      does. Refused (rc=3).
  `PRAGMA data_version` moved -> the arms came off two snapshots and part of
      the difference is a dream landing mid-dump. Refused (rc=2).
  members that resolve to neither a node nor an episode are COUNTED in the
      header, so a scorer knows the summaries half is incomplete.

BANKED CRITERIA (fixed before this ever ran; printed with the prompts)
-----------------------------------------------------------------------
  S1-C1  grounding gain -- claims in the FIXED digest grounded in restored
         edges and absent or wrong in the CURRENT one     >=3 of 10 slots
  S1-C2  faithfulness  -- claims in the FIXED digest contradicted by its own
         facts block                                      0 (hard zero)
  S1-C3  no regression -- claims correct in the CURRENT digest absent or
         degraded in the FIXED one                        0
  S1-C4  cost -- block <=40 lines; exactly one extra root fusion per store;
         levels >=1 byte-identical                        as stated

S1-C3 is the load-bearing one. The fix is ADDITIVE, and additive is precisely
the shape narrative facts had when they cost 2.9pp on LoCoMo by DISTRACTION,
not crowding -- the ON arm saw a strict superset and still lost. Additive is
not a safety argument.

n=10 claim-slots on one store is a qualitative faithfulness gate in the G-F1b
mould, not a powered test. It can REFUSE the fix; it can never certify a
benefit.

Usage (run on the BOX -- no store exists on the dev machine):
  python3 benchmarks/digest_squeeze_dump.py ~/.hermes/hymem.sqlite
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hymem.config import HyMemConfig  # noqa: E402
from hymem.dreaming.aggregate import (  # noqa: E402
    _anchor_facts,
    _items_text,
    load_digest,
)
from hymem.extraction.prompts import (  # noqa: E402
    DIGEST_SYSTEM,
    DIGEST_USER_TEMPLATE,
)

# Reuse, not re-implementation: the read-only opener from the sibling probe,
# and the counterfactual arm + snapshot guard from Stage 0, so Stage 0 and
# Stage 1 can never disagree about what "the fix would render" means.
from recovery_probe import open_store_readonly  # noqa: E402
from digest_squeeze_probe import (  # noqa: E402
    SnapshotMoved,
    _data_version,
    fixed_facts,
    measure_squeeze,
)

DUMP_MODULE_PATH = str(Path(__file__).resolve())

_CRITERIA = """\
#   S1-C1  grounding gain -- claims in the FIXED digest that are grounded in a
#          restored edge and absent or WRONG in the CURRENT one   BAR >=3 of 10
#   S1-C2  faithfulness -- claims in the FIXED digest contradicted by its own
#          VERIFIED FACTS block                                   BAR 0 (hard)
#   S1-C3  no regression -- claims correct in the CURRENT digest that are
#          absent or degraded in the FIXED one                    BAR 0
#   S1-C4  cost -- facts block <=40 lines; exactly one extra root fusion per
#          store; levels >=1 byte-identical                       as stated
#
#   PASS (all four) -> ship. S1-C1 fails -> the restored edges add nothing
#   measurable: do NOT ship the additive version. S1-C2 or S1-C3 fails -> the
#   fix harms the digest: close or redesign, no score-chasing.
#
#   S1-C3 is load-bearing. The fix is ADDITIVE, and additive is the shape
#   narrative facts had when they cost 2.9pp on LoCoMo by DISTRACTION, not
#   crowding. n=10 slots on one store can REFUSE the fix; it can never
#   certify a benefit -- say so in the verdict."""

_HOW_TO_SCORE = """\
#   1. Paste the SYSTEM PROMPT plus ARM A's user prompt into the box LLM;
#      collect the JSON digest. Repeat for ARM B in a FRESH context.
#   2. Sample 10 claim-slots from the two summaries and score S1-C1..C3.
#   3. Record the numbers in additional_planning.md. Do not paste either
#      digest back into an agent context -- it is conversation content."""


class NoRootDigest(RuntimeError):
    """The store has no root digest, or its root resolves to no members. Either
    way the summaries half of the prompt would be empty, and an empty prompt
    hand-scores as a confident constant. The dump refuses rather than prints."""


def root_items(conn: sqlite3.Connection) -> tuple[list[dict], list[str]]:
    """The root's fusion inputs, in the PERSISTED MEMBER ORDER.

    Member order is fusion-input order and `_items_text` joins in that order,
    so the order is part of the prompt. `expand_node` is deliberately NOT
    reused: it splits members into `child_nodes` and `episodes`, which loses
    the interleaving. The resolution order (aggregation node first, then
    episode) is the same as `expand_node:1396-1434`.

    Returns (items, unresolved_member_ids). Read-only.
    """
    digest = load_digest(conn)
    if digest is None:
        raise NoRootDigest("this store has no root digest node")
    row = conn.execute(
        "SELECT member_episode_ids FROM aggregation_nodes WHERE id = ?",
        (digest.node_id,),
    ).fetchone()
    member_ids = json.loads(row["member_episode_ids"]) if row is not None else []

    items: list[dict] = []
    missing: list[str] = []
    for member_id in member_ids:
        node = conn.execute(
            "SELECT id, title, summary FROM aggregation_nodes WHERE id = ?",
            (member_id,),
        ).fetchone()
        if node is not None:
            items.append({"id": node["id"], "title": node["title"] or "",
                          "summary": node["summary"] or ""})
            continue
        ep = conn.execute(
            "SELECT id, title, summary FROM episodes WHERE id = ?",
            (member_id,),
        ).fetchone()
        if ep is not None:
            items.append({"id": ep["id"], "title": ep["title"] or "",
                          "summary": ep["summary"] or ""})
            continue
        # A root kept through a failed fusion can point at replaced nodes
        # (aggregate.py's root_failed branch). Counted, never silently dropped.
        missing.append(member_id)
    return items, missing


def _facts_block(facts: list[str]) -> str:
    """The VERIFIED FACTS block exactly as `aggregate.py:1240` builds it,
    including the literal "(none)" placeholder a disabled cap produces."""
    return "\n".join(f"- {f}" for f in facts) if facts else "(none)"


def build_arms(
    conn: sqlite3.Connection,
    *,
    cap: int = 20,
    edge_cap: int | None = None,
    profile_cap: int | None = None,
    max_members: int | None = None,
) -> dict:
    """Both root-fusion prompts over one tree and one snapshot.

    The summaries half (`text`) is rendered ONCE and shared, so the arms differ
    only in the facts block. `HyMemConfig` is constructed purely to supply
    `aggregation_max_members` to `_items_text`; construction is inert (there is
    no `__post_init__` and no config validation anywhere in `hymem/`), and
    `max_members` overrides it for a box whose config is not the default.
    """
    items, missing = root_items(conn)
    if not items:
        raise NoRootDigest(
            f"the root digest resolves to no members ({len(missing)} member id(s) "
            "match neither an aggregation node nor an episode)"
        )

    cfg = HyMemConfig(root=Path("."))
    if max_members is not None:
        cfg = replace(cfg, aggregation_max_members=max_members)
    text = _items_text(items, cfg)

    current = _facts_block(_anchor_facts(conn, cap))
    fixed = _facts_block(fixed_facts(
        conn,
        edge_cap=cap if edge_cap is None else edge_cap,
        profile_cap=cap if profile_cap is None else profile_cap,
    ))
    sizing = measure_squeeze(conn, cap=cap, edge_cap=edge_cap,
                             profile_cap=profile_cap)
    return {
        "root_id": load_digest(conn).node_id,
        "items": items,
        "missing": missing,
        "text": text,
        "current_facts": current,
        "fixed_facts": fixed,
        "current_prompt": DIGEST_USER_TEMPLATE.format(facts=current, text=text),
        "fixed_prompt": DIGEST_USER_TEMPLATE.format(facts=fixed, text=text),
        "edges_restored": sizing["edges_restored"],
        "verdict": sizing["verdict"],
        "cap": sizing["cap"],
        "edge_cap": sizing["edge_cap"],
        "profile_cap": sizing["profile_cap"],
    }


def _print_dump(arms: dict, *, path: str) -> None:
    bar = "=" * 72
    print("# digest profile squeeze — Stage 1 hand-scoring dump")
    print(f"# store: {path}")
    print(f"# root:  {arms['root_id']}")
    print(f"# members: {len(arms['items'])} resolved, {len(arms['missing'])} "
          "unresolved member id(s)"
          + ("   !! the summaries half is INCOMPLETE" if arms["missing"] else ""))
    print(f"# cap={arms['cap']}  edge_cap={arms['edge_cap']}  "
          f"profile_cap={arms['profile_cap']}   Stage-0 verdict "
          f"{arms['verdict']}, edges_restored={arms['edges_restored']}")
    print("#")
    print("# HOW TO SCORE")
    print(_HOW_TO_SCORE)
    print("#")
    print("# BANKED CRITERIA (fixed before this ever ran)")
    print(_CRITERIA)
    print("#")
    print("# The prompts below are the user's real conversation content. Local "
          "terminal only:")
    print("# never a file, never a log, never pasted back into an agent context.")
    print()
    print(bar)
    print("DIGEST SYSTEM PROMPT (shared by both arms)")
    print(bar)
    print(DIGEST_SYSTEM)
    print()
    print(bar)
    print("ARM A — CURRENT (shared cap, early return: production today)")
    print(bar)
    print(arms["current_prompt"])
    print()
    print(bar)
    print("ARM B — FIXED (separate profile and edge budgets)")
    print(bar)
    print(arms["fixed_prompt"])
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("db", help="path to an existing hymem.sqlite (opened read-only)")
    ap.add_argument("--cap", type=int, default=20,
                    help="aggregation_digest_anchor_facts as production runs it "
                         "(default: 20, the shipped default)")
    ap.add_argument("--edge-cap", type=int, default=None,
                    help="the fix's independent EDGE budget (default: --cap)")
    ap.add_argument("--profile-cap", type=int, default=None,
                    help="the fix's independent PROFILE budget (default: --cap)")
    ap.add_argument("--max-members", type=int, default=None,
                    help="aggregation_max_members, only if the box overrides it")
    args = ap.parse_args(argv)

    conn = open_store_readonly(args.db)
    try:
        before = _data_version(conn)
        try:
            arms = build_arms(conn, cap=args.cap, edge_cap=args.edge_cap,
                              profile_cap=args.profile_cap,
                              max_members=args.max_members)
        except NoRootDigest as exc:
            print(f"!! no root digest to score: {exc}. Dream this store first; "
                  "do NOT hand-score an empty prompt.", file=sys.stderr)
            return 1
        if _data_version(conn) != before:
            print("!! the store was written during the read (data_version "
                  "moved) -- a dream landed mid-dump, so the two arms are not "
                  "one snapshot. Re-run; do not score this pair.",
                  file=sys.stderr)
            return 2
    finally:
        conn.close()

    if arms["current_facts"] == arms["fixed_facts"]:
        print("!! the two VERIFIED FACTS blocks are identical on this store, so "
              "there is nothing to score: the arms would differ in no way and "
              "S1-C1 would read 0 for a reason unrelated to the fix. Run the "
              "Stage-0 probe for the verdict "
              f"({arms['verdict']}) instead.", file=sys.stderr)
        return 3

    _print_dump(arms, path=args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
