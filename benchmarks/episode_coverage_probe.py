#!/usr/bin/env python3
"""Stage-2 front-run gate: episode-extraction COVERAGE on a dreamed prod store.

The Stage-2 finding (raptor_digest_plan.md): 26/91 prod sessions produced ZERO
episodes, so neither the root digest nor the aggregation clusters can ever cover
them — the same root cause as the banked LME finding (42% of MS misses had no
gold episode). Three different causes need three different fixes, and the plan's
gate is explicit: build NOTHING until this probe says which bucket dominates.

This probe is offline, LLM-less and READ-ONLY (sqlite URI mode=ro — it can never
write to the prod store). Point it at an existing, already-dreamed hymem.sqlite
and it characterizes every session that has no episode row, assigning one of
three buckets:

  never_dreamed       sessions.digested_prompt_version IS NULL — the dream
                      scheduler/runner never ran the batched session digest
                      over this session at all.
  dreamed_zero_short  digested, zero episodes, and the session is THIN
                      (≤ --short-max-messages messages OR ≤ --short-max-chars
                      total user+assistant content chars) — plausibly below the
                      extractor's minimum-content bar.
  dreamed_zero_long   digested, zero episodes, but the session is SUBSTANTIAL
                      (above BOTH thresholds) — an extraction-recall problem in
                      the episode prompt itself.

Read the verdict like this (over the uncovered sessions):
  never_dreamed dominates
        → scheduler/runner bug; the fix is mechanical (make the runner reach
          every session), no prompt work. Cheapest possible outcome.
  dreamed_zero_short dominates
        → the episode extractor's minimum-content threshold is too strict, or
          the prompt refuses thin sessions → relax + add "a single substantive
          exchange is an episode" instruction.
  dreamed_zero_long dominates
        → extraction-prompt RECALL problem on real content → worth a dedicated
          fix + a re-run of the banked MS coverage numbers afterwards.

CAVEATS (same spirit as gold_rank_probe's):
  - The short/long boundary is heuristic. A session is "short" when EITHER axis
    is thin (few turns OR little text) — a 2-turn session can still carry 10k
    chars and a 12-turn session can be all "ok"/"thanks". If the verdict is
    borderline, sweep the thresholds and eyeball the --json dump instead of
    trusting one cut.
  - Chars are counted over user+assistant turns only (tool/system turns are
    retrieval noise and are likewise excluded from messages_fts); the role mix
    in the per-session listing shows how much "other" volume was ignored.
  - digested_prompt_version records the LAST successful digest's prompt version,
    not a per-version history: a session digested under an old prompt and
    skipped since still counts as dreamed. That is the correct reading for this
    gate (the extractor DID see it and returned nothing).

Usage (run on the Hermes box, from benchmarks/):
  python episode_coverage_probe.py ~/.hermes/memory/hymem.sqlite
  python episode_coverage_probe.py store.sqlite --short-max-messages 6 \
      --short-max-chars 2500 --json coverage_audit.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

BUCKETS = ("never_dreamed", "dreamed_zero_short", "dreamed_zero_long")

# Roles whose content chars count toward the short/long boundary. Mirrors the
# messages_fts ingest filter (tool/system turns are noise, not memory content).
_CONTENT_ROLES = ("user", "assistant")


def open_store_readonly(path: str | Path) -> sqlite3.Connection:
    """Open an existing hymem sqlite store strictly read-only (URI mode=ro).

    The probe must never write to the prod store; mode=ro makes any stray
    write raise sqlite3.OperationalError instead of mutating the DB.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"store not found: {p}")
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Pure core — imported by tests/test_episode_coverage_probe.py; main() below is
# only the CLI/printing shell over this (the raptor_cluster_probe contract:
# the logic a build would reuse is a plain importable function).
# ─────────────────────────────────────────────────────────────────────────────


def dream_history(conn: sqlite3.Connection) -> dict | None:
    """What the dream runner has actually done on this store, from `dream_runs`.

    Reports CYCLES, and deliberately not a chunk ratio. `chunks_seen` is the
    size of the candidate pool the cycle considered, re-counted from scratch
    every cycle -- on the prod store each of the last cycles saw ~330 chunks and
    processed 0-2, because there was nothing new. Summed across cycles it reads
    425/330255, and `chunks_processed / chunks_seen` looks like a coverage
    fraction while being nothing of the kind. The number is not reported rather
    than reported with a caveat, because a caveat does not survive being quoted.

    Returns None on a store whose schema predates `dream_runs`; the caller then
    says the reading is unavailable rather than assuming either answer."""
    try:
        rows = conn.execute(
            "SELECT id, sessions_processed, chunks_seen, chunks_processed, "
            "episodes_created, digest_failures FROM dream_runs ORDER BY id"
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    if not rows:
        return {"cycles": 0, "episodes_created": 0, "digest_failures": 0,
                "last": None}
    last = rows[-1]
    return {
        "cycles": len(rows),
        "episodes_created": sum(r["episodes_created"] or 0 for r in rows),
        "digest_failures": sum(r["digest_failures"] or 0 for r in rows),
        "last": {
            "id": last["id"],
            "sessions_processed": last["sessions_processed"],
            "chunks_seen": last["chunks_seen"],
            "chunks_processed": last["chunks_processed"],
        },
    }


def never_dreamed_reading(history: dict | None, never_dreamed: list[dict]) -> str:
    """Which reading of `never_dreamed` the evidence supports.

    The bucket alone cannot tell a runner that FAILED to reach a session from
    one that has not reached it YET, and the two have different fixes -- the
    verdict guide below offers only the first. HyMem's dream is incremental by
    design: each cycle spends a chunk budget on the most salient candidates and
    leaves the rest. Over a long-lived store that converges; over a BULK INGEST
    followed by exactly one cycle, which is what every benchmark adapter does,
    it does not."""
    if not never_dreamed:
        return "no never_dreamed sessions, so this reading has nothing to decide"
    with_content = [r for r in never_dreamed if r["n_messages"] > 0]
    if not with_content:
        return (f"NOT A GAP — all {len(never_dreamed)} never_dreamed session(s) "
                f"hold zero messages. There is nothing in them to extract, and "
                f"no prompt or scheduler change can create an episode from an "
                f"empty session.")
    msgs = sum(r["n_messages"] for r in with_content)
    if history is None:
        return ("UNAVAILABLE — this store has no `dream_runs` table, so the "
                "runner's own record cannot be read and neither reading is "
                "supported")
    if history["cycles"] == 0:
        return (f"NEVER RAN — `dream_runs` is empty. {len(with_content)} "
                f"session(s) holding {msgs} message(s) are un-digested for the "
                f"most mundane reason there is.")
    if history["cycles"] == 1:
        return (f"ONE CYCLE OVER A BULK INGEST — {len(with_content)} session(s) "
                f"holding {msgs} message(s) carry no digest after a single "
                f"dream. The per-cycle chunk budget makes that the EXPECTED "
                f"shape, not a runner bug: prompt work cannot reach these "
                f"sessions, and further cycles can. Read any episode-quality "
                f"result on this store as bounded to the digested fraction.")
    return (f"RUNNER GAP — {history['cycles']} dream cycles have run and "
            f"{len(with_content)} session(s) holding {msgs} message(s) still "
            f"carry no digest. Repeated cycles rule out the per-cycle budget, "
            f"so this is the mechanical scheduler reading.")


def characterize_coverage(
    conn: sqlite3.Connection,
    short_max_messages: int = 4,
    short_max_chars: int = 1500,
) -> dict:
    """Characterize episode coverage over all sessions in an opened store.

    Returns a dict with totals, the coverage fraction, one record per UNCOVERED
    session (zero episodes) carrying its bucket, and a per-bucket histogram.
    Read-only: issues SELECTs only.
    """
    rows = conn.execute(
        """
        SELECT
            s.id                                              AS session_id,
            s.digested_prompt_version                         AS digested_prompt_version,
            COUNT(m.id)                                       AS n_messages,
            COALESCE(SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END), 0)      AS n_user,
            COALESCE(SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END), 0) AS n_assistant,
            COALESCE(SUM(CASE WHEN m.role NOT IN ('user','assistant')
                              THEN 1 ELSE 0 END), 0)          AS n_other,
            COALESCE(SUM(CASE WHEN m.role IN ('user','assistant')
                              THEN LENGTH(m.content) ELSE 0 END), 0) AS content_chars,
            MIN(m.created_at)                                 AS first_message_at,
            MAX(m.created_at)                                 AS last_message_at,
            (SELECT COUNT(*) FROM episodes e
              WHERE e.session_id = s.id)                      AS n_episodes
        FROM sessions s
        LEFT JOIN messages m ON m.session_id = s.id
        GROUP BY s.id
        ORDER BY s.id
        """
    ).fetchall()

    total = len(rows)
    covered = sum(1 for r in rows if r["n_episodes"] > 0)
    uncovered: list[dict] = []
    histogram = {b: 0 for b in BUCKETS}

    for r in rows:
        if r["n_episodes"] > 0:
            continue
        if r["digested_prompt_version"] is None:
            bucket = "never_dreamed"
        elif (
            r["n_messages"] <= short_max_messages
            or r["content_chars"] <= short_max_chars
        ):
            bucket = "dreamed_zero_short"
        else:
            bucket = "dreamed_zero_long"
        histogram[bucket] += 1
        uncovered.append({
            "session_id": r["session_id"],
            "n_messages": r["n_messages"],
            "content_chars": r["content_chars"],
            "n_user": r["n_user"],
            "n_assistant": r["n_assistant"],
            "n_other": r["n_other"],
            "digested_prompt_version": r["digested_prompt_version"],
            "first_message_at": r["first_message_at"],
            "last_message_at": r["last_message_at"],
            "bucket": bucket,
        })

    history = dream_history(conn)
    return {
        "dream_history": history,
        "never_dreamed_reading": never_dreamed_reading(
            history, [u for u in uncovered if u["bucket"] == "never_dreamed"]),
        "total_sessions": total,
        "covered_sessions": covered,
        "uncovered_sessions": total - covered,
        "coverage_fraction": (covered / total) if total else 0.0,
        "short_max_messages": short_max_messages,
        "short_max_chars": short_max_chars,
        "buckets": histogram,
        "uncovered": uncovered,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / report shell
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(result: dict) -> None:
    total = result["total_sessions"]
    covered = result["covered_sessions"]
    uncovered = result["uncovered"]

    print("\nEpisode-coverage probe (Stage-2 front-run gate)")
    print(f"  Sessions: {total}   with ≥1 episode: {covered}   "
          f"coverage: {100 * result['coverage_fraction']:.1f}%")
    print(f"  Short/long boundary: ≤{result['short_max_messages']} messages "
          f"OR ≤{result['short_max_chars']} user+assistant chars")
    h = result.get("dream_history")
    if h is None:
        print("  Dream history: UNAVAILABLE (no `dream_runs` table in this store)")
    elif h["cycles"] == 0:
        print("  Dream history: 0 cycles — the dream never ran here")
    else:
        last = h["last"]
        print(f"  Dream history: {h['cycles']} cycle(s), "
              f"{h['episodes_created']} episodes created, "
              f"{h['digest_failures']} digest failures; last cycle #{last['id']} "
              f"processed {last['chunks_processed']} of "
              f"{last['chunks_seen']} candidate chunks over "
              f"{last['sessions_processed']} session(s)")
    print()

    if not uncovered:
        print("  No uncovered sessions — nothing to bucket. Stage 2 is a no-op.")
        return

    print(f"  UNCOVERED sessions ({len(uncovered)}):")
    header = (f"  {'session_id':<28} {'msgs':>5} {'chars':>7} "
              f"{'u/a/o':>8} {'digest_ver':<12} {'first..last message':<34} bucket")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rec in uncovered:
        mix = f"{rec['n_user']}/{rec['n_assistant']}/{rec['n_other']}"
        ver = rec["digested_prompt_version"] or "-"
        span = (f"{rec['first_message_at'] or '?'} .. {rec['last_message_at'] or '?'}"
                if rec["n_messages"] else "(no messages)")
        print(f"  {rec['session_id']:<28} {rec['n_messages']:>5} "
              f"{rec['content_chars']:>7} {mix:>8} {ver:<12} {span:<34} {rec['bucket']}")

    print("\n  Bucket histogram:")
    n_unc = len(uncovered)
    for b in BUCKETS:
        n = result["buckets"][b]
        bar = "█" * n
        print(f"    {b:<20} {n:>4}  ({100 * n / n_unc:5.1f}%)  {bar}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("store", help="path to an existing (dreamed) hymem.sqlite")
    ap.add_argument("--short-max-messages", type=int, default=4,
                    help="a zero-episode dreamed session with ≤ this many messages "
                         "is bucketed short (default 4)")
    ap.add_argument("--short-max-chars", type=int, default=1500,
                    help="… or with ≤ this many total user+assistant content chars "
                         "(default 1500)")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="dump the full result (incl. per-session records) to this "
                         "json path for audit")
    args = ap.parse_args(argv)

    conn = open_store_readonly(args.store)
    try:
        result = characterize_coverage(
            conn,
            short_max_messages=args.short_max_messages,
            short_max_chars=args.short_max_chars,
        )
    finally:
        conn.close()

    _print_report(result)

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nPer-session records written to {args.json}")

    print(f"\n  never_dreamed reading (from the runner's own record):"
          f"\n    {result['never_dreamed_reading']}")
    print("\nVERDICT GUIDE (dominant bucket → Stage-2 fix):")
    print("  never_dreamed dominates")
    print("      → scheduler/runner bug — sessions the dream loop never digested;")
    print("        fix is MECHANICAL (make the runner reach them), no prompt work.")
    print("  dreamed_zero_short dominates")
    print("      → extractor's minimum-content threshold too strict / prompt refuses")
    print("        thin sessions → RELAX + add 'a single substantive exchange is an")
    print("        episode' instruction.")
    print("  dreamed_zero_long dominates")
    print("      → extraction-prompt RECALL problem on substantial sessions →")
    print("        dedicated prompt fix + re-run the banked MS coverage numbers.")
    print("  (Gate: build nothing until one bucket clearly dominates; if it's a")
    print("   borderline mix, sweep --short-max-* and audit the --json dump.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
