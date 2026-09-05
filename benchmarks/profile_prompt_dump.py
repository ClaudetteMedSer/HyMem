#!/usr/bin/env python3
"""Stage-1 / P4 front-run gate: dump the EXACT user-profile extraction prompts.

The plan's gate (raptor_digest_plan.md, Stage 1): before trusting the typed
user-profile extraction, run the extraction PROMPT manually over ~20 sessions
from the prod store and hand-score precision/recall of the slot assertions.
The closed vocabulary should make precision >= 0.9; if it doesn't, STOP — do
not enable the tier.

This script is offline, LLM-less and READ-ONLY (sqlite URI mode=ro — it can
never write to the store). It samples N sessions evenly across the store's
history (not just the newest stretch) and prints, per session, the exact
rendered extraction prompt over that session's USER turns ONLY. The rendering
is delegated to `hymem.dreaming.user_profile.build_profile_user_prompt` — the
same window renderer the dream phase calls — so what you paste into the LLM on
the box is the exact first production slice (same [msg N] tags and character
cap). Later slices are cursor-driven and include labeled boundary context. The
shared system prompt is printed once at the top.

Scoring procedure:
  1. For each printed session, paste the SYSTEM PROMPT + the session's user
     prompt into the box LLM and collect the named `{"items": [...]}` object.
  2. Per emitted item, judge: is the (slot, value) actually asserted by the
     cited [msg N] turn? precision = correct / emitted.
  3. Per session, note profile facts the user clearly stated that the LLM
     missed. recall = found / stated (rough is fine — precision is the gate).

Usage:
    python3 benchmarks/profile_prompt_dump.py /path/to/hymem.sqlite
    python3 benchmarks/profile_prompt_dump.py /path/to/hymem.sqlite --sessions 30
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hymem.dreaming.user_profile import (  # noqa: E402
    PROFILE_PROMPT_VERSION,
    build_profile_user_prompt,
    fetch_user_turns,
)
from hymem.extraction.prompts import USER_PROFILE_SYSTEM  # noqa: E402


def _evenly_spaced(seq: list, cap: int) -> list:
    """At most `cap` elements spread evenly across `seq` (first and last kept).
    Local copy: the aggregation layer replaced its version with a churn-stable
    hash-rank sample (2026-07-12 reuse fix); for a one-shot offline dump, even
    spacing over the store's history is still exactly right."""
    n = len(seq)
    if cap <= 0 or n <= cap:
        return list(seq)
    if cap == 1:
        return [seq[-1]]
    idx = sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})
    return [seq[i] for i in idx]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the exact per-session user-profile extraction "
                    "prompts for manual precision/recall scoring (read-only)."
    )
    parser.add_argument("store", help="Path to an existing hymem.sqlite")
    parser.add_argument(
        "--sessions", type=int, default=20,
        help="How many sessions to sample, spread evenly across the store "
             "(default 20)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=12000,
        help="Char cap on the first rendered prompt body; keep at production "
             "default (cfg.dream_digest_max_chars = 12000) so the gate tests "
             "resumable-window settings",
    )
    args = parser.parse_args()

    store = Path(args.store)
    if not store.exists():
        print(f"error: store not found: {store}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    session_ids = [
        r["id"]
        for r in conn.execute("SELECT id FROM sessions ORDER BY started_at, id")
    ]
    sampled = _evenly_spaced(session_ids, args.sessions)

    print(f"# user-profile prompt dump — {PROFILE_PROMPT_VERSION}")
    print(f"# store: {store}  sessions: {len(sampled)}/{len(session_ids)} "
          f"(evenly sampled)  max_chars: {args.max_chars}")
    print()
    print("=" * 72)
    print("SYSTEM PROMPT (shared by every session below)")
    print("=" * 72)
    print(USER_PROFILE_SYSTEM)

    skipped = 0
    for sid in sampled:
        turns = fetch_user_turns(conn, sid)
        if not turns:
            skipped += 1
            continue
        print("=" * 72)
        print(f"SESSION {sid}  ({len(turns)} user turns)")
        print("=" * 72)
        print(build_profile_user_prompt(turns, max_chars=args.max_chars))
        print()

    if skipped:
        print(f"# skipped {skipped} sampled session(s) with no user turns "
              f"(the dream phase skips them too)")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
