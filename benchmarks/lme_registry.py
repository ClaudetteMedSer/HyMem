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
  lme_registry.py audit [--strict]       Date-check the aggregation label
  lme_registry.py arms A.json B.json --lever K   Can this pair evidence its A/B?

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


def _require_run_date(run_date: str | None, archive: str) -> str:
    """§6.5: this table declares `run_date TEXT NOT NULL` (see SCHEMA).

    iso_ts records an absent date as None, which beam/locomo store as NULL
    but LME's schema cannot.  Raise the §6-style defect here rather than
    let SQLite raise IntegrityError from inside the INSERT: an LME
    artifact carrying neither a date field nor an exec stamp is a defect
    in the artifact, and the error should name the file that caused it.
    """
    if run_date is None:
        raise ValueError(
            f"no usable run_date for {archive!r}: this benchmark's table "
            "declares run_date NOT NULL -- an artifact with no date field "
            "and no exec stamp is a defect, not a domain fact"
        )
    return run_date


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
        return (_require_run_date(
                    rr.iso_ts(exec_date or data.get("date")), archive),
                source_date, None, None)
    return (_require_run_date(rr.iso_ts(data.get("date")), archive),
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
    interesting failure is a row that changes when it shouldn't.

    §6.5: a row whose artifact BENCH_DIR cannot supply is UNREACHABLE,
    reported separately from unreadable/recompute-failed and returned as
    a count so main() exits nonzero.  LME artifacts all live in one dir,
    so no multi-dir search is needed here (beam splits across two) — but
    the guarantee all three registries share is that a row the backfill
    could not migrate can never read as success."""
    con = connect()
    bench = Path(bench_dir or BENCH_DIR)
    rows = con.execute(
        "SELECT id, archive, kind, run_date, source_date, total_tokens, "
        "elapsed_s FROM runs ORDER BY id").fetchall()
    changed, missing, unreachable = [], [], []
    for _id, archive, kind, cur_rd, cur_sd, cur_tt, cur_es in rows:
        if kind == "doc":
            continue
        p = bench / archive
        if not p.exists():
            unreachable.append((_id, archive, f"no artifact in {bench}"))
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
          f"skipped (unreadable/recompute-failed), "
          f"{len(unreachable)} row(s) UNREACHABLE")
    for _id, archive, kind, diffs in changed:
        for f, old, newv in diffs:
            print(f"  id={_id} [{kind}] {archive} {f}: {old!r} -> {newv!r}")
    for _id, archive, why in missing:
        print(f"  id={_id} {archive}: SKIPPED ({why})")
    for _id, archive, why in unreachable:
        print(f"  id={_id} {archive}: UNREACHABLE ({why})")
    if unreachable:
        print("  NOTE: unreachable rows were NOT migrated -- fix the "
              "artifact path or LME_BENCH_DIR and re-run")
    return len(unreachable)



# ---------------------------------------------------------------- audit ----
#
# The date-check this module's docstring promises ("done on top of NULLs, not
# inside them"). It exists because `aggregation_nodes_enabled = 0` in this DB
# is not one claim but three, and only one of them is a measurement:
#
#   RECORDED  the run's own config block carried the key (post-6543ee6).
#   ASSERTED  an analyst supplied it with `--set` after the fact.
#   ABSENT    NULL -- nothing recorded it at all.
#
# An ASSERTED or ABSENT 0 is a statement about what the operator MEANT. Whether
# the run actually had the layer off depends on the code that ran, and for four
# days it did not follow the intent:
#
#   2026-08-26T16:26:57Z  52adfe5  library default aggregation_nodes_enabled
#                                  False -> True (G-FLIP PASS)
#   2026-08-30T20:50:00Z  2247074  longmemeval_adapter pins the lever BOTH ways
#
# Between those two commits the adapter set only the True leg, so an
# un-flagged run inherited the library's new True. Any row in that window
# labelled 0 is contradicted by the code that produced it, however the label
# got there. Outside it the label is honoured: before, the default agreed with
# it; after, the pin enforces it.
#
# `run_date` is the END of the run -- the archive stamp is written when the
# file is. A run that ENDED after the pin may have STARTED before it, so the
# window test uses `run_date - elapsed_s` where elapsed_s is known. On the two
# 2026-08-30/31 guard arms that is the difference between a 26-minute margin
# and no margin at all, and the conservative reading is the only safe one.
AGGREGATION_DEFAULT_FLIP = "2026-08-26T16:26:57Z"   # 52adfe5
AGGREGATION_ADAPTER_PIN = "2026-08-30T20:50:00Z"    # 2247074
# Before this commit `--episode-granularity` did not exist, so no run through
# this adapter could have had it on, whatever a later --set says.
EPISODE_GRANULARITY_LEVER = "2026-08-30T20:50:00Z"  # 2247074, same commit


def _iso_z(ts):
    """Normalise the registry's several stamp spellings to a sortable Z form."""
    if not ts:
        return None
    t = str(ts).strip().replace(" ", "T")
    if t.endswith("Z"):
        t = t[:-1]
    if "+" in t[10:]:
        t = t[:10] + t[10:].split("+")[0]
    return t[:19] + "Z"


def _minus_seconds(iso, seconds):
    """`iso` less `seconds`, as a Z string. Used to turn a run's END stamp into
    its START, because that is the instant the code version is decided."""
    import datetime as _dt
    if iso is None or not seconds:
        return iso
    t = _dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
    return (t - _dt.timedelta(seconds=float(seconds))).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def label_source(provenance, column, value):
    """Where this row's value for `column` came from -- not what it says."""
    if value is None:
        return "ABSENT"
    if provenance and f"analyst:{column}=" in provenance:
        return "ASSERTED"
    return "RECORDED"


def audit_row(run_date, elapsed_s, aggr, epg, provenance):
    """Verdict for one row. Returns (start, verdict, note).

    CONTRADICTED is the only failing verdict, and it is reserved for a claim
    the code of the day could not have honoured."""
    end = _iso_z(run_date)
    start = _minus_seconds(end, elapsed_s)
    src_a = label_source(provenance, "aggregation_nodes_enabled", aggr)
    src_e = label_source(provenance, "episode_granularity_enabled", epg)

    if epg == 1 and start is not None and start < EPISODE_GRANULARITY_LEVER:
        return start, "CONTRADICTED", (
            f"episode_granularity_enabled=1 ({src_e}) on a run that started "
            f"{start}, before the lever existed ({EPISODE_GRANULARITY_LEVER}, "
            f"2247074) -- no run through this adapter could have had it on")

    if start is None:
        return start, "UNKNOWN", "no run_date, so no window can be decided"
    in_window = AGGREGATION_DEFAULT_FLIP <= start < AGGREGATION_ADAPTER_PIN
    if not in_window:
        side = "pre-flip" if start < AGGREGATION_DEFAULT_FLIP else "post-pin"
        return start, "OK", f"{side}; aggregation label {src_a} and honoured"
    if aggr == 0:
        return start, "CONTRADICTED", (
            f"aggregation_nodes_enabled=0 ({src_a}) on a run that started "
            f"{start}, inside the unpinned window -- the adapter set only the "
            f"True leg, so the layer was ON regardless of the label")
    if aggr is None:
        return start, "CONTRADICTED", (
            "aggregation_nodes_enabled is ABSENT on a run inside the unpinned "
            f"window (started {start}) -- it inherited the library's True, so "
            "this row is an aggregation-ON run and must not read as unknown")
    return start, "OK", f"in-window but labelled ON ({src_a}); no conflict"


def cmd_audit(strict=False):
    con = connect()
    rows = con.execute(
        "SELECT id, archive, kind, run_date, elapsed_s, "
        "aggregation_nodes_enabled, episode_granularity_enabled, "
        "flags_provenance FROM runs ORDER BY run_date, id").fetchall()
    results = []
    for _id, archive, kind, rd, es, aggr, epg, prov in rows:
        if kind == "doc":
            continue
        start, verdict, note = audit_row(rd, es, aggr, epg, prov)
        results.append((_id, archive, start, verdict, note))

    window = [r for r in results
              if r[2] is not None
              and AGGREGATION_DEFAULT_FLIP <= r[2] < AGGREGATION_ADAPTER_PIN]
    bad = [r for r in results if r[3] == "CONTRADICTED"]
    unknown = [r for r in results if r[3] == "UNKNOWN"]

    print(f"\n=== aggregation-label audit — {len(results)} run(s) ===")
    print(f"  unpinned window: [{AGGREGATION_DEFAULT_FLIP}, "
          f"{AGGREGATION_ADAPTER_PIN})  (52adfe5 → 2247074)")
    # A window with nothing in it is the outcome that most resembles a pass, so
    # it is stated as a count rather than left to be inferred from silence. The
    # check having no work to do is a finding about the ledger, not a clean
    # bill of health for a check that ran.
    print(f"  runs whose execution overlapped that window: {len(window)}")
    if not window:
        print("  → the contamination has no victims in this ledger: every run "
              "either predates the\n    default flip or postdates the adapter "
              "pin. Nothing here was silently aggregation-ON.")
    for _id, archive, start, verdict, note in results:
        if verdict == "OK" and not strict:
            continue
        print(f"  [{verdict}] id={_id} {archive}\n      started {start}: {note}")
    print(f"\n  CONTRADICTED {len(bad)}   UNKNOWN {len(unknown)}   "
          f"OK {len(results) - len(bad) - len(unknown)}")
    if bad:
        print("  → these rows must NOT be used as an aggregation-OFF baseline "
              "for the episode-granularity flip.")
    return len(bad)



def cmd_arms(path_a, path_b, lever):
    """Ask whether a claimed A/B pair can evidence its own contrast.

    Run BEFORE reading the two scores, for the same reason a pre-registration is
    written before a run: once the numbers are in, an unevidenced pair reads as
    a result rather than as a question."""
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    verdict, note, confounds = rr.arm_evidence(
        a.get("config"), b.get("config"), lever)
    print(f"\n=== arm evidence — {lever} ===")
    print(f"  A  {Path(path_a).name}")
    print(f"  B  {Path(path_b).name}")
    print(f"  [{verdict}] {note}")
    if confounds:
        print(f"  confounded on {len(confounds)} other key(s): "
              f"{', '.join(confounds)}")
    if verdict != rr.ARM_EVIDENCED:
        print("  → this pair cannot discharge a gate on that lever. The scores "
              "may still be\n    correct; what is missing is any way to tell a "
              "real null from two runs of one arm.")
        return 1
    return 0


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
        if cmd_backfill():
            sys.exit(1)   # §6.5: unreachable rows were not migrated
    elif cmd == "list":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--limit", type=int, default=30)
        p.add_argument("--flag")
        a = p.parse_args(sys.argv[2:])
        cmd_list(a.limit, a.flag)
    elif cmd == "query":
        cmd_query(" ".join(sys.argv[2:]))
    elif cmd == "arms":
        import argparse
        p = argparse.ArgumentParser(prog="lme_registry.py arms")
        p.add_argument("a")
        p.add_argument("b")
        p.add_argument("--lever", required=True)
        a = p.parse_args(sys.argv[2:])
        if cmd_arms(a.a, a.b, a.lever):
            sys.exit(1)
    elif cmd == "audit":
        if cmd_audit(strict="--strict" in sys.argv[2:]):
            sys.exit(1)   # a contradicted label is a hard failure, not a note
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
