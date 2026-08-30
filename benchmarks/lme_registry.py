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
# active: these are filled from `overrides` in the adapter, which never
# made it into the config block.  Recorded as NULL so the gap is visible.
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
    run_date    TEXT NOT NULL,          -- from JSON date field (UTC-ish ISO)
    source_date TEXT NOT NULL,          -- archive file stem timestamp as provenance
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
           str(data.get("date", ""))[:19],
           path.stem[:16],
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
           cfg.get("total_tokens"),
           cfg.get("elapsed_s"),
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
