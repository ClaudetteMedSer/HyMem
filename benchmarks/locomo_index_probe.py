#!/usr/bin/env python3
"""Why didn't a gold turn surface? Read-only probe of the persisted stores.

The audit can localise a miss to "never entered the retrieval pool", but it
cannot say WHY, and the three candidate causes want three different fixes:

  NOT INGESTED  the turn is not in `messages` at all        -> adapter/ingest bug
  NOT INDEXED   in `messages` but not in `messages_fts`     -> trigger/role-filter bug
  NOT MATCHED   indexed, but an FTS query for the question's -> query-side: tokenization,
                terms does not return it                       stopwords, name rewriting

Only the third is a tuning problem; the first two are defects, and neither is
fixable by any aperture or prompt lever. Distinguishing them costs nothing —
`--db-dir` stores are on disk and this opens them read-only, runs no model, and
writes nothing.

Note `messages_fts` uses tokenize='porter unicode61' and indexes ONLY
user/assistant turns (schema.sql) — a turn absent from the index because of the
role filter is a mapping question for the adapter, not a HyMem bug.

Usage:
  python locomo_index_probe.py RESULTS.json --data data/locomo10.json \
      --db-dir <same dir the run used> [--ids conv-26_q158,...] [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from locomo_adapter import load_locomo_data
from msc_adapter import _lex_match, _content_tokens


def _fts_query(text: str, n: int = 8) -> str:
    """An OR-query of the longest content tokens, quoted so FTS5 can't read a
    stray token as an operator. Longest-first because rare words carry the
    signal; this approximates a keyword path, not HyMem's real query builder."""
    toks = sorted(_content_tokens(text), key=len, reverse=True)[:n]
    return " OR ".join(f'"{t}"' for t in toks if t.isalnum())


def probe(db: Path, turn: str, question: str) -> tuple[str, str]:
    """Return (verdict, detail) for one evidence turn in one store."""
    if not db.exists():
        return "NO STORE", str(db)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        key = " ".join(turn.split()[:6])
        row = con.execute(
            "SELECT id, role FROM messages WHERE content LIKE ? LIMIT 1",
            (f"%{key}%",)).fetchone()
        if row is None:
            return "NOT INGESTED", f"no messages row LIKE '{key[:40]}…'"
        mid, role = row
        # messages_fts is an EXTERNAL-CONTENT table (content='messages'), so a
        # bare `WHERE rowid = ?` is answered from `messages` and returns a row
        # even for turns the trigger never indexed. Index membership has to be
        # tested through the index: query the turn's OWN rare tokens and see
        # whether its rowid comes back.
        def _match(query: str) -> list[int] | str:
            if not query:
                return "no usable content tokens"
            try:
                return [r[0] for r in con.execute(
                    "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? "
                    "ORDER BY rank LIMIT 200", (query,)).fetchall()]
            except sqlite3.OperationalError as e:
                return f"FTS error: {e}"

        self_hits = _match(_fts_query(turn))
        if isinstance(self_hits, str):
            return "QUERY ERROR", self_hits
        if mid not in self_hits:
            return "NOT INDEXED", (f"messages.id={mid} role={role!r} is not "
                                   f"retrievable by its own terms — absent from "
                                   f"messages_fts (role filter or trigger)")
        q = _fts_query(question)
        ids = _match(q)
        if isinstance(ids, str):
            return "NOT MATCHED", ids
        if mid in ids:
            return "MATCHED", f"rank {ids.index(mid) + 1} of {len(ids)} — reachable by FTS"
        return "NOT MATCHED", f"indexed (id={mid}) but absent from {len(ids)} FTS hits"
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="the --out file whose misses you audited")
    ap.add_argument("--data", required=True)
    ap.add_argument("--db-dir", required=True, help="the SAME --db-dir the run used")
    ap.add_argument("--user-speaker", choices=["a", "b"], default="a")
    ap.add_argument("--ids", default=None,
                    help="comma-separated question ids (e.g. the recall bucket); "
                         "default = every non-cat-5 miss in the file")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if args.ids:
        keep = {s.strip() for s in args.ids.split(",")}
        rows = [r for r in rows if r["id"] in keep]
        if missing := keep - {r["id"] for r in rows}:
            print(f"[warn] {len(missing)} id(s) not in the results file: "
                  f"{sorted(missing)[:5]}\n")
    else:
        rows = [r for r in rows if not r["correct"] and r["category"] != 5]
    if not rows:
        sys.exit("Nothing to probe.")

    ev_map = {c["id"]: c["evidence_map"]
              for c in load_locomo_data(args.data, user_speaker=args.user_speaker)}

    verdicts: Counter[str] = Counter()
    shown = 0
    for r in rows:
        db = Path(args.db_dir) / r["conv_id"] / "hymem.sqlite"
        for eid in (r.get("evidence") or []):
            hit = ev_map.get(r["conv_id"], {}).get(eid)
            if not hit:
                verdicts["NO ANNOTATION"] += 1
                continue
            _, turn = hit
            v, detail = probe(db, turn, r["question"])
            verdicts[v] += 1
            if not args.limit or shown < args.limit:
                shown += 1
                print(f"[{r['id']} {eid}] {v}\n    {detail}\n    turn: {turn[:150]}")

    print(f"\n=== {sum(verdicts.values())} evidence turns probed ===")
    for v, n in verdicts.most_common():
        print(f"  {v:<14} {n:>4}")
    print("\n  NOT INGESTED / NOT INDEXED are DEFECTS — no aperture or prompt "
          "lever reaches them.\n  NOT MATCHED is query-side and is where "
          "--name-prefix could plausibly help.\n  MATCHED means FTS can reach the "
          "turn, so the loss happened downstream (ranking/fusion).")


if __name__ == "__main__":
    main()
