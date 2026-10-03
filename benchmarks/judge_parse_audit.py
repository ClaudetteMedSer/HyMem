#!/usr/bin/env python3
"""§5 of the judge-parse-fix pre-registration. READ-ONLY, zero API calls.

How far does the brace defect reach into artifacts already on disk? Because
`judge_raw` is stored on the rejudge path, that question costs nothing to
answer -- the judge's real verdict is sitting in every file next to the score
that contradicts it.

Reports, per artifact: rows carrying a raw, how many the old regex could not
read, and how many of those the brace-matching parser recovers. Rewrites
nothing.

    judge_parse_audit.py <artifact.json> [...]
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_adapter import extract_judge_json  # noqa: E402


def naive(raw):
    """The old parser, verbatim, so the delta is measured against what shipped
    rather than against a description of it."""
    m = re.search(r'\{[^}]+\}', raw.replace('\n', ' '))
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def main() -> int:
    print(f"{'artifact':<58}{'raw':>5}{'fail':>6}{'recov':>7}{'unread':>8}")
    total_rec = 0
    no_raw = []
    for path in sys.argv[1:]:
        d = json.load(open(path))
        rows = [q for c in d.get("conversations", []) for q in c.get("questions", [])]
        withraw = [q for q in rows if q.get("judge_raw")]
        name = Path(path).name
        if not withraw:
            no_raw.append((name, len(rows)))
            print(f"{name[:56]:<58}{'-':>5}{'-':>6}{'-':>7}{'-':>8}  NO RAW STORED")
            continue
        fails = [q for q in withraw if naive(q["judge_raw"]) is None]
        rec, unread = [], []
        for q in fails:
            obj, _ = extract_judge_json(q["judge_raw"])
            (rec if obj is not None else unread).append((q, obj))
        total_rec += len(rec)
        print(f"{name[:56]:<58}{len(withraw):>5}{len(fails):>6}{len(rec):>7}{len(unread):>8}")
        for q, obj in rec:
            scores = obj.get("scores", [])
            real = sum(scores) / len(scores) if scores else 0.0
            print(f"      RECOVERABLE {q['ability']:<4} judge said scores={scores} "
                  f"(= {real:.4f}); recorded {q['score']:.4f}")
    print(f"\n  {total_rec} recoverable row(s) across the audited artifacts.")
    if no_raw:
        print("\n  UNAUDITABLE — these store no judge_raw, so their exposure to this")
        print("  defect cannot be measured even in principle:")
        for name, n in no_raw:
            print(f"    {name} ({n} rows)")
        print("  The rejudge path stores raws; the main run path does not.")
    print("\n  Nothing was rewritten.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
