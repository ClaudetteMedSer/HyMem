#!/usr/bin/env python3
"""Classify the RAPTOR reuse watch and take gate G-FLIP. Read-only, no LLM.

Implements Phase 0 of `benchmarks/readside_synthesis_plan.md` (§0.1 pull, §0.2
classify, §0.3 gate) against the box store's `dream_runs` rows. Every row in
the window gets exactly one label, top to bottom:

  deploy-refusion    reused ~0 on a full tree -> a salt bump legitimately
                     rebuilt everything. Exactly ONE expected per deploy that
                     bumped `_CLUSTER_SALT` / `_ROLLUP_SALT`; the script cannot
                     verify a bump happened, so it prints a CHECK line — a
                     refusion with no salt change is a red flag, not an excuse.
  failure-attributed failures > 0. LLM-flakiness event, not membership churn.
  failure-heal       the run right after a failure row, still below the bar
                     with failures = 0 (migration 022: "each fail->heal
                     transition costs one low-reuse run"). Excluded but counted
                     against the failure budget, and flagged for sign-off.
                     `--no-heal-grace` makes these unclassifiable instead.
  blocking-flip      blocking mode changed vs the previous aggregation row
                     ('knn' <-> 'exact:<reason>'). Env-parity bug; blocks.
  append             input_episodes grew. IN the verdict — this is the failure
                     mode both prior watches died on.
  quiescent          none of the above. IN the verdict.

Rows that did no aggregation work (lock-skips, errors, built = 0) are excluded
as `no-agg` and never reach the gate.

Usage (on the box, store is opened read-only):
  python flipwatch_classify.py \
      --db ~/.hermes/hymem.sqlite --since 2026-07-12 \
      --csv ~/.hermes/benchmarks/flipwatch_2026-07.csv \
      --out flipwatch_result.md

Paste the emitted block into `benchmarks/raptor_digest_plan.md` Stage 3c.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

# §0.3, from the banked flip criterion.
REUSE_BAR = 90.0
MAX_FAILURE_ROWS = 2
MIN_VERDICT_ROWS = 5

# A deploy-refusion is a near-total rebuild of a full-size tree, not a partial
# dip: reuse at or under this, with a built count in line with the window.
REFUSION_REUSE_MAX = 10.0
REFUSION_BUILT_FRACTION = 0.8

COLUMNS = (
    "id, started_at, "
    "aggregation_nodes_built AS built, "
    "aggregation_nodes_reused AS reused, "
    "aggregation_fusion_failures AS failures, "
    "aggregation_input_episodes AS input_eps, "
    "aggregation_blocking AS blocking, "
    "skipped_locked, error"
)


def pull(db: Path, since: str) -> list[dict]:
    """Read the watch window read-only (§0.1). Never opens the store rw."""
    uri = f"file:{db}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT {COLUMNS} FROM dream_runs "
            "WHERE started_at >= ? ORDER BY id",
            (since,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        sys.exit(
            f"error: {exc}\n"
            "The attribution columns land in migration 022 (schema v22). A store "
            "without them predates the 2026-07-12 fixes and cannot be classified."
        )
    finally:
        conn.close()
    return [dict(r) for r in rows]


def reuse_pct(row: dict) -> float | None:
    return round(100.0 * row["reused"] / row["built"], 1) if row["built"] else None


def classify(rows: list[dict], heal_grace: bool = True) -> list[dict]:
    """Label every row per §0.2. Order matters: earlier labels win."""
    agg = [r for r in rows if r["built"] and not r["skipped_locked"] and not r["error"]]
    builts = sorted(r["built"] for r in agg)
    median_built = builts[len(builts) // 2] if builts else 0
    refusion_seen = False
    prev_agg: dict | None = None
    prev_failed = False

    out = []
    for row in rows:
        pct = reuse_pct(row)
        rec = dict(row, reuse_pct=pct, note="")

        if row["skipped_locked"]:
            rec["label"] = "no-agg"
            rec["note"] = "lock-skip"
            out.append(rec)
            continue
        if row["error"]:
            rec["label"] = "no-agg"
            rec["note"] = f"error: {str(row['error'])[:60]}"
            out.append(rec)
            continue
        if not row["built"]:
            rec["label"] = "no-agg"
            rec["note"] = "no aggregation work"
            out.append(rec)
            continue

        is_refusion = (
            pct is not None
            and pct <= REFUSION_REUSE_MAX
            and row["built"] >= REFUSION_BUILT_FRACTION * median_built
        )
        blocking_flipped = (
            prev_agg is not None
            and row["blocking"]
            and prev_agg["blocking"]
            and row["blocking"] != prev_agg["blocking"]
        )
        appended = prev_agg is not None and row["input_eps"] > prev_agg["input_eps"]

        if is_refusion and not refusion_seen:
            refusion_seen = True
            rec["label"] = "deploy-refusion"
            rec["note"] = "CHECK: only valid if this deploy bumped a fusion salt"
        elif is_refusion:
            rec["label"] = "unclassifiable"
            rec["note"] = "second near-total refusion in window (only one expected)"
        elif row["failures"] > 0:
            rec["label"] = "failure-attributed"
            rec["note"] = f"{row['failures']} fusion failure(s)"
        elif prev_failed and pct is not None and pct < REUSE_BAR and heal_grace:
            rec["label"] = "failure-heal"
            rec["note"] = (
                f"REVIEW: heal run after #{prev_agg['id']}; "
                "confirm the fusions healed rather than re-keyed"
            )
        elif blocking_flipped:
            rec["label"] = "blocking-flip"
            rec["note"] = f"{prev_agg['blocking']} -> {row['blocking']}"
        elif appended:
            rec["label"] = "append"
            rec["note"] = f"input_eps {prev_agg['input_eps']} -> {row['input_eps']}"
        else:
            rec["label"] = "quiescent"

        # A verdict row below the bar fits none of labels 1-4: a fifth cause
        # exists, and §0.3 makes that an automatic FAIL.
        if rec["label"] in ("append", "quiescent") and pct is not None and pct < REUSE_BAR:
            rec["note"] = (rec["note"] + "; " if rec["note"] else "") + (
                f"{pct}% < {REUSE_BAR:g}% bar"
            )

        prev_failed = row["failures"] > 0
        prev_agg = row
        out.append(rec)
    return out


def gate(rows: list[dict]) -> tuple[str, list[str], list[str]]:
    """Apply G-FLIP (§0.3). Returns (verdict, check lines, advisory lines)."""
    verdict_rows = [r for r in rows if r["label"] in ("append", "quiescent")]
    below = [r for r in verdict_rows if r["reuse_pct"] is not None and r["reuse_pct"] < REUSE_BAR]
    blocking = [r for r in rows if r["label"] == "blocking-flip"]
    failures = [r for r in rows if r["label"] in ("failure-attributed", "failure-heal")]
    unclassifiable = [r for r in rows if r["label"] == "unclassifiable"]
    appends = [r for r in verdict_rows if r["label"] == "append"]

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    checks = [
        f"- [{mark(not below)}] every verdict row (append + quiescent) at reuse >= "
        f"{REUSE_BAR:g}% — {len(verdict_rows) - len(below)}/{len(verdict_rows)}"
        + (f"; offenders: {', '.join('#' + str(r['id']) for r in below)}" if below else ""),
        f"- [{mark(not blocking)}] zero blocking-flip rows — found {len(blocking)}"
        + (f": {', '.join('#' + str(r['id']) for r in blocking)}" if blocking else ""),
        f"- [{mark(len(failures) <= MAX_FAILURE_ROWS)}] failure-attributed rows <= "
        f"{MAX_FAILURE_ROWS} — found {len(failures)}"
        + (f": {', '.join('#' + str(r['id']) for r in failures)}" if failures else ""),
        f"- [{mark(not unclassifiable)}] no unclassifiable low-reuse row — found "
        f"{len(unclassifiable)}"
        + (f": {', '.join('#' + str(r['id']) for r in unclassifiable)}" if unclassifiable else ""),
        f"- [{mark(len(verdict_rows) >= MIN_VERDICT_ROWS)}] sanity floor: >= "
        f"{MIN_VERDICT_ROWS} verdict rows — {len(verdict_rows)}",
    ]

    advisories = []
    if not appends:
        advisories.append(
            "**append-untested**: zero append rows in the verdict set. The banked "
            "gate does not require them, but both prior watches died precisely on "
            "append dreams — a pass carried entirely by quiescent rows is weak "
            "evidence. Extend the watch until real traffic adds episodes."
        )
    elif len(appends) < 3:
        advisories.append(
            f"**thin append coverage**: only {len(appends)} append row(s) "
            f"({', '.join('#' + str(r['id']) for r in appends)}). Prefer a few more "
            "before flipping."
        )
    if any(r["label"] == "failure-heal" for r in rows):
        advisories.append(
            "**heal-grace applied**: one or more low-reuse rows were excused as "
            "fail->heal runs. That extension is read off migration 022's prose, not "
            "the letter of §0.2 — sign it off by hand, or rerun with "
            "`--no-heal-grace` to see the strict verdict."
        )
    if any(r["label"] == "deploy-refusion" for r in rows):
        advisories.append(
            "**refusion CHECK**: confirm the deploy at that row bumped `_CLUSTER_SALT` "
            "or `_ROLLUP_SALT` (hymem/dreaming/aggregate.py). A refusion without a "
            "salt bump is an unexplained rebuild and should be treated as "
            "unclassifiable."
        )

    if len(verdict_rows) < MIN_VERDICT_ROWS and not (below or blocking or unclassifiable):
        return "INSUFFICIENT", checks, advisories
    passed = not below and not blocking and not unclassifiable and len(failures) <= MAX_FAILURE_ROWS
    return ("PASS" if passed else "FAIL"), checks, advisories


def render(rows: list[dict], verdict: str, checks: list[str], advisories: list[str],
           db: Path, since: str) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    lines = [
        "### RESULT — flip-watch classification (G-FLIP)",
        "",
        f"Window: `started_at >= {since}` · store `{db}` (read-only) · "
        f"{len(rows)} dream_runs rows · generated by `benchmarks/flipwatch_classify.py`.",
        "",
        "| id | started_at | built | reused | reuse% | fail | input_eps | blocking | label | note |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pct = "" if r["reuse_pct"] is None else f"{r['reuse_pct']}"
        lines.append(
            f"| {r['id']} | {r['started_at']} | {r['built']} | {r['reused']} | {pct} | "
            f"{r['failures']} | {r['input_eps']} | {r['blocking'] or '—'} | "
            f"`{r['label']}` | {r['note']} |"
        )

    lines += [
        "",
        "Labels: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        "",
        "**Gate G-FLIP**",
        "",
        *checks,
        "",
        f"**VERDICT: {verdict}**",
        "",
    ]

    if advisories:
        lines += ["Advisories (beyond the banked gate):", ""]
        lines += [f"- {a}" for a in advisories] + [""]

    if verdict == "PASS":
        lines += [
            "Next per §0.4: flip `aggregation_nodes_enabled` in `hymem/config.py:112` "
            "to `True` (docstring records the flip date + this block), full suite "
            "green, doc ripple (this RESULT + \"FLIPPED <date>\", "
            "`hymem/Hermes_instruction.md` item 2 -> legacy note, "
            "`additional_planning.md` Plan C -> \"UNBLOCKED <date>\"). No salt is "
            "bumped by the flip, so the post-deploy verification dream must show "
            "high reuse — a refusion there is itself a red flag.",
        ]
    elif verdict == "FAIL":
        lines += [
            "Do NOT flip. Bank this as a third failed-watch RESULT and route by "
            "label (§0.3): `blocking-flip` -> env parity (sqlite-vec on both trigger "
            "paths); a `failure` streak -> LLM transport/retry work; an "
            "`unclassifiable` row -> reopen the windowing analysis with that row's "
            "`input_eps`/`built` delta as the starting evidence. Plan C stays blocked.",
        ]
    else:
        lines += [
            f"Fewer than {MIN_VERDICT_ROWS} verdict rows: extend the watch rather "
            "than deciding (§0.3 sanity floor).",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path.home() / ".hermes" / "hymem.sqlite",
                    help="store to read (opened read-only)")
    ap.add_argument("--since", default="2026-07-12",
                    help="window start, matched against started_at (default: the "
                         "bb96057 deploy date)")
    ap.add_argument("--csv", type=Path, help="also dump the classified rows as CSV")
    ap.add_argument("--out", type=Path, help="write the RESULT block here (also printed)")
    ap.add_argument("--no-heal-grace", action="store_true",
                    help="do not excuse the run after a fusion failure; strict §0.2")
    args = ap.parse_args()

    if not args.db.exists():
        sys.exit(f"error: no store at {args.db} (run this on the box)")

    rows = pull(args.db, args.since)
    if not rows:
        sys.exit(f"error: no dream_runs rows with started_at >= {args.since}")

    classified = classify(rows, heal_grace=not args.no_heal_grace)
    verdict, checks, advisories = gate(classified)
    block = render(classified, verdict, checks, advisories, args.db, args.since)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["id", "started_at", "built", "reused", "reuse_pct", "failures",
                  "input_eps", "blocking", "skipped_locked", "error", "label", "note"]
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(classified)
        print(f"[csv] {args.csv}", file=sys.stderr)

    if args.out:
        args.out.write_text(block)
        print(f"[out] {args.out}", file=sys.stderr)
    print(block)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
