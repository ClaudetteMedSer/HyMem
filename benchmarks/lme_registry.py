#!/usr/bin/env python3
"""Run registry for LongMemEval benchmark executions.

Builds/updates a small SQLite DB (lme_runs.db) that records one row per
run JSON: date, scores, and the effective flags that were on/off at the
time the run was created.  The goal is a queryable answer to "which
config produced this score?" — something the per-run JSON files cannot
answer, and which would have surfaced the 2026-08-26 aggregation-default
flip had it existed.

Usage:
  lme_registry.py ingest [FILE ...]      Add runs from JSON files (default: all archives)
  lme_registry.py list [--limit N] [--flag COL]   Print table
  lme_registry.py query "SQL"            Raw query against the DB

Flags are recorded exactly as they appear in each run's config block.
If a key is absent (e.g. older format, or a lever that was not recorded),
the value is stored as NULL — this registry does not guess; the
date-check / provenance analysis is done on top of NULLs, not inside
them.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

try:  # package import (tests): benchmarks.run_registry
    from . import run_registry as rr
except (ImportError, ValueError):  # direct CLI: python benchmarks/lme_registry.py
    import run_registry as rr

DB = Path(os.environ.get("LME_REGISTRY_DB", "/home/node/.hermes/benchmarks/lme_runs.db"))
BENCH_DIR = Path(os.environ.get("LME_BENCH_DIR", "/home/node/.hermes/benchmarks"))

# Columns that exist across the known config formats (superset).
# Each maps to config-key or is a fixed transform.
FLAG_COLUMNS = [
    "auto_ability",
    "no_dream",
    "permissive_default",
    "embeddings",
    "graph_facts_first",
    "graph_multihop",
    "distill",
    "rerank_top_k",
    "sample",
    "scale",
    "seed",
    "workers",
    "top_k",
    "answer_model",
    "judge_model",
    "aggregation_nodes_enabled",   # NULL where never recorded (pre-pin formats)
    "episode_granularity_enabled", # NULL where never recorded (pre-2247074)
    "value_supersession_enabled",  # NULL where never recorded
]
# Config keys known to be *absent* in run JSONs even though they were
# active: pre-6543ee6 runs never carried these in the config block; the guard
# arms and all earlier rows fill them via `--set` only (NULL = not recorded).
# Post-6543ee6 the adapter writes them into the config block at :2887, so
# newer runs ingest without --set. Rows predating 6543ee6 stay NULL —
# the gap is visible, not guessed.
KNOWN_ABSENT = {"aggregation_nodes_enabled", "episode_granularity_enabled",
                "value_supersession_enabled"}

SCORE_CATEGORIES = [
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    archive     TEXT UNIQUE NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'archive',  -- archive | variant | rejudge
    run_date    TEXT NOT NULL,          -- from JSON date field (UTC-ish ISO);
                                        -- rejudge rows: last stem stamp (exec)
    source_date TEXT,              -- filename timestamp stamp (first
                                        -- stamp / rejudged_from); NULL for
                                        -- analyst-named variants (no stamp)
    -- flags (all NULL if not present in the run config block)
    auto_ability               INTEGER,
    no_dream                   INTEGER,
    permissive_default         INTEGER,
    embeddings                 INTEGER,
    graph_facts_first          INTEGER,
    graph_multihop             INTEGER,
    distill                    INTEGER,
    rerank_top_k               INTEGER,
    sample                     INTEGER,
    scale                      TEXT,
    seed                       INTEGER,
    workers                    INTEGER,
    top_k                      INTEGER,
    answer_model               TEXT,
    judge_model                TEXT,
    mr_aggregate_additive      INTEGER,  -- old-format name of the aggregation lever
    aggregation_nodes_enabled  INTEGER,
    episode_granularity_enabled INTEGER,
    value_supersession_enabled INTEGER,
    -- scores
    overall                    REAL,
    multi_session              REAL,
    single_session_assistant   REAL,
    single_session_preference  REAL,
    single_session_user        REAL,
    knowledge_update           REAL,
    temporal_reasoning         REAL,
    count                      INTEGER,
    answer_calls               INTEGER,
    judge_calls                INTEGER,
    total_tokens               INTEGER,
    elapsed_s                  REAL,
    flags_provenance           TEXT,     -- how recorded + analyst-set flags were established
    extras                     TEXT     -- JSON: full config block + scores, for anything unmodeled
);
CREATE INDEX IF NOT EXISTS idx_runs_date ON runs(run_date);
CREATE INDEX IF NOT EXISTS idx_runs_flags ON runs(auto_ability, no_dream, permissive_default);
CREATE INDEX IF NOT EXISTS idx_runs_aggr ON runs(aggregation_nodes_enabled);
"""


def connect():
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    # §6 migration (2026-08-31): source_date became nullable — NULL is the
    # honest value where a stem carries no stamp (beam/locomo-style rows,
    # analyst-renamed LME variants).  SQLite cannot drop a NOT NULL in
    # place, so rebuild the table when the old constraint is present.
    cols = {r[1]: r for r in con.execute("PRAGMA table_info(runs)")}
    if cols.get("source_date") and cols["source_date"][3]:  # notnull flag
        con.execute("ALTER TABLE runs RENAME TO runs_prestamp")
        con.executescript(SCHEMA)
        con.execute("INSERT INTO runs SELECT * FROM runs_prestamp")
        con.execute("DROP TABLE runs_prestamp")
        con.commit()
    return con


def _to_int(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


# ---- stamp policy (§6, 2026-08-31) --------------------------------------
# LME adapter-written names always carry a \\d{8}T\\d{6}Z stamp (archive +
# rejudge): policy "required" — a missing stamp RAISES via the shared
# helper.  A NULL there is a defect, not a domain fact (it would look
# identical to a legitimately stamp-less beam/locomo row).  Analyst-renamed
# variants (longmemeval-v2-hymem-additive.json, -auto-ability-fulldream.json)
# carry no stamp by construction -> policy "optional", NULL recorded.
STAMP_POLICY = {"archive": "required", "rejudge": "required", "variant": "optional"}


def _stamp_fields(kind: str, archive: str, data: dict, cfg: dict):
    """(run_date, source_date, total_tokens, elapsed_s) under §6 rules.

    Rejudge: run_date = last stem stamp (rejudge exec; the artifact's own
    date field is the SOURCE's date, inherited) or the artifact date fallback;
    source_date = first stamp of rejudged_from (the source pointer); stats
    = NULL (inherited-but-wrong is worse than missing).
    Archive: run_date = artifact date; source_date = first stem stamp
    (required policy -> raises if a stamp-bearing name lacks one).
    Variant: same, optional policy (analyst labels may carry no stamp).
    """
    policy = STAMP_POLICY.get(kind, "optional")
    if kind == "rejudge":
        source_date, exec_date = rr.rejudge_dates(
            archive, cfg.get("rejudged_from") or "", policy)
        return (exec_date or str(data.get("date", ""))[:19],
                source_date, None, None)
    return (str(data.get("date", ""))[:19],
            rr.stem_source_date(archive, policy),
            cfg.get("total_tokens"), cfg.get("elapsed_s"))


def ingest_file(con, path: Path, overrides: dict | None = None):
    """Insert one run JSON. Returns 'inserted' | 'skipped' | 'error'."""
    overrides = overrides or {}
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return f"error: {e}"
    cfg = dict(data.get("config") or {})
    # Recorded-absent levers: the adapter never wrote these into config,
    # even when active.  Fill only from explicit analyst overrides; the
    # extras blob records that they were supplied out-of-band.
    for k, v in overrides.items():
        if k in OVERRIDE_COLUMNS:
            cfg[k] = v
    scores = data.get("scores") or {}

    def cat(name):
        v = scores.get(name)
        return round(v["accuracy"], 3) if isinstance(v, dict) else None

    overall = scores.get("OVERALL")
    overall_acc = round(overall["accuracy"], 3) if isinstance(overall, dict) else None

    # sanity: overwrite only if identical file already in DB
    archive = path.name
    exists = con.execute("SELECT id FROM runs WHERE archive=?", (archive,)).fetchone()
    if exists:
        return "skipped"

    if "rejudged" in archive:
        kind = "rejudge"
    elif "-seed" in archive or archive.startswith("longmemeval-v2-hymem-2"):
        kind = "archive"
    else:
        kind = "variant"

    run_date, source_date, total_tokens, elapsed_s = _stamp_fields(
        kind, archive, data, cfg)

    extras = {
        "config": cfg,
        "scores": scores,
        "raw_json_keys": sorted(data.keys()),
        "analyst_set": overrides,  # non-empty only when --set was used
    }
    proven = ["recorded" if k not in overrides else f"analyst:{k}={overrides[k]}" for k in overrides]
    prov = "recorded" if not overrides else "recorded + " + "; ".join(proven)
    row = (archive,
           kind,
           run_date,
           source_date,
           _to_int(cfg.get("auto_ability")),
           _to_int(cfg.get("no_dream")),
           _to_int(cfg.get("permissive_default")),
           _to_int(cfg.get("embeddings")),
           _to_int(cfg.get("graph_facts_first")),
           _to_int(cfg.get("graph_multihop")),
           _to_int(cfg.get("distill")),
           cfg.get("rerank_top_k"),
           cfg.get("sample"),
           cfg.get("scale"),
           cfg.get("seed"),
           cfg.get("workers"),
           cfg.get("top_k"),
           cfg.get("answer_model"),
           cfg.get("judge_model"),
           _to_int(cfg.get("mr_aggregate_additive")),
           cfg.get("aggregation_nodes_enabled"),
           cfg.get("episode_granularity_enabled"),
           cfg.get("value_supersession_enabled"),
           overall_acc,
           cat("multi-session"),
           cat("single-session-assistant"),
           cat("single-session-preference"),
           cat("single-session-user"),
           cat("knowledge-update"),
           cat("temporal-reasoning"),
           _to_int(cfg.get("count")) or (overall.get("count") if isinstance(overall, dict) else None),
           cfg.get("answer_calls"),
           cfg.get("judge_calls"),
           total_tokens,
           elapsed_s,
           prov,
           json.dumps(extras, default=str),
           )
    try:  # row order must match INSERT: ..., elapsed_s, flags_provenance, extras
        con.execute("""INSERT INTO runs (
            archive, kind, run_date, source_date,
            auto_ability, no_dream, permissive_default, embeddings,
            graph_facts_first, graph_multihop, distill, rerank_top_k,
            sample, scale, seed, workers, top_k, answer_model, judge_model,
            mr_aggregate_additive,
            aggregation_nodes_enabled, episode_granularity_enabled,
            value_supersession_enabled,
            overall, multi_session, single_session_assistant,
            single_session_preference, single_session_user, knowledge_update,
            temporal_reasoning, count, answer_calls, judge_calls,
            total_tokens, elapsed_s, flags_provenance, extras)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
        return "inserted"
    except sqlite3.IntegrityError:
        return "skipped"


OVERRIDE_COLUMNS = {
    "aggregation_nodes_enabled", "episode_granularity_enabled",
    "value_supersession_enabled", "auto_ability", "no_dream",
    "permissive_default", "embeddings", "graph_facts_first",
    "graph_multihop", "distill", "rerank_top_k", "sample", "scale",
    "seed", "workers", "top_k", "answer_model", "judge_model",
}


def cmd_ingest(files, overrides=None):
    con = connect()
    if not files:
        files = sorted(BENCH_DIR.glob("longmemeval-v2-hymem*.json"))
        # Exclude the "latest" pointer (it duplicates an archive) and any
        # file that isn't a run record.
        files = [f for f in files if f.name != "longmemeval-v2-hymem.json"]
    stats = {}
    for f in files:
        r = ingest_file(con, Path(f), overrides)
        stats[r] = stats.get(r, 0) + 1
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    print(f"DB: {DB}  rows={total}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    missing = con.execute(
        "SELECT COUNT(*) FROM runs WHERE aggregation_nodes_enabled IS NULL AND mr_aggregate_additive IS NULL").fetchone()[0]
    if missing:
        print(f"  NOTE: {missing} rows have neither aggregation_nodes_enabled nor mr_aggregate_additive recorded")


def cmd_list(limit=30, flag=None):
    con = connect()
    cols = ["id", "run_date", "archive", "overall", "multi_session",
            "auto_ability", "no_dream", "permissive_default",
            "aggregation_nodes_enabled", "episode_granularity_enabled", "answer_model"]
    if flag:
        cols.append(flag)
    q = f"SELECT {', '.join(cols)} FROM runs ORDER BY run_date DESC LIMIT ?"
    rows = con.execute(q, (limit,)).fetchall()
    print(" | ".join(cols))
    print("-" * 120)
    for r in rows:
        print(" | ".join("-" if v is None else str(v) for v in r))


def cmd_query(sql):
    con = connect()
    cur = con.execute(sql)
    print(" | ".join(d[0] for d in cur.description))
    for r in cur.fetchall():
        print(" | ".join("-" if v is None else str(v) for v in r))


def cmd_backfill(bench_dir=None):
    """§6.4: recompute stamp-derived fields for EVERY row from its artifact
    file; UPDATE where the value differs.  The read-back is a DIFF against
    the pre-backfill values, not a spot check that new values parse — the
    interesting failure is a row that changes when it shouldn't."""
    con = connect()
    bench = Path(bench_dir or BENCH_DIR)
    rows = con.execute(
        "SELECT id, archive, kind, run_date, source_date, total_tokens, "
        "elapsed_s FROM runs ORDER BY id").fetchall()
    changed, missing = [], []
    for _id, archive, kind, cur_rd, cur_sd, cur_tt, cur_es in rows:
        if kind == "doc":
            continue
        p = bench / archive
        if not p.exists():
            missing.append((_id, archive, "file not found"))
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            missing.append((_id, archive, f"unreadable: {e}"))
            continue
        try:
            cfg = dict(data.get("config") or {})
            new_rd, new_sd, new_tt, new_es = _stamp_fields(kind, archive, data, cfg)
        except ValueError as e:
            missing.append((_id, archive, f"recompute failed: {e}"))
            continue
        diffs = []
        if cur_rd != new_rd:
            diffs.append(("run_date", cur_rd, new_rd))
        if cur_sd != new_sd:
            diffs.append(("source_date", cur_sd, new_sd))
        if cur_tt != new_tt:
            diffs.append(("total_tokens", cur_tt, new_tt))
        if cur_es != new_es:
            diffs.append(("elapsed_s", cur_es, new_es))
        if not diffs:
            continue
        changed.append((_id, archive, kind, diffs))
        con.execute(
            "UPDATE runs SET run_date=?, source_date=?, total_tokens=?, "
            "elapsed_s=? WHERE id=?",
            (new_rd, new_sd, new_tt, new_es, _id))
    con.commit()
    print(f"backfill: {len(changed)} row(s) changed, {len(missing)} row(s) "
          f"skipped (missing/unreadable/recompute-failed)")
    for _id, archive, kind, diffs in changed:
        for f, old, newv in diffs:
            print(f"  id={_id} [{kind}] {archive} {f}: {old!r} -> {newv!r}")
    for _id, archive, why in missing:
        print(f"  id={_id} {archive}: SKIPPED ({why})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "ingest":
        ov = {}
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--set" and i + 1 < len(args):
                k, v = args[i + 1].split("=", 1)
                ov[k] = v
                args.pop(i)
                args.pop(i)
            else:
                i += 1
        cmd_ingest(args or None, ov)  # noqa: E1123
    elif cmd == "backfill":
        cmd_backfill()
    elif cmd == "list":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--limit", type=int, default=30)
        p.add_argument("--flag")
        a = p.parse_args(sys.argv[2:])
        cmd_list(a.limit, a.flag)
    elif cmd == "query":
        cmd_query(" ".join(sys.argv[2:]))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
