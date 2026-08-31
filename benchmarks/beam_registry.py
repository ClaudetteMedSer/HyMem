#!/usr/bin/env python3
"""Run registry for BEAM benchmark executions.

Same model as lme_registry.py: one row per run JSON, recording date,
scores, and the effective flags (recorded vs analyst-set, never guessed).
DB is restart-safe: /home/node/.hermes/benchmarks/beam_runs.db.

BEAM run JSONs come in three dialects:
  A. hardcoded saves: {date, config{scales,sample,top_k,answer_model,
     judge_model}, scores{ABILITY: pct}}               (v13/v14)
  B. same shape but NO date                             (v15; date: None)
  C. adapter-native: {metadata{date,answer_model,...}, summary{scale:
     {ABILITY: frac}}}                                  (v16+)
The adapter at HEAD records metadata (models, sample, top_k,
context_memories, calls, elapsed) but NEVER the levers that shape the
store (facts, facts-extraction, embeddings, graph_multihop, no_dream,
distill, aggregation/episode levers) — the same recording gap as LME.
Those columns are NULL unless analyst-supplied via --set.

Usage:
  beam_registry.py ingest [FILE ...] [--set k=v ...]
  beam_registry.py list [--limit N] [--flag COL]
  beam_registry.py query "SQL"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:  # package import (tests): benchmarks.run_registry
    from . import run_registry as rr
    from .run_registry import (
        DEFAULT_REGISTRY_DIR,
        _coerce,
        cmd_ingest,
        cmd_list,
        cmd_query,
        connect,
    )
except (ImportError, ValueError):  # direct CLI: python benchmarks/beam_registry.py
    import run_registry as rr
    from run_registry import (
        DEFAULT_REGISTRY_DIR,
        _coerce,
        cmd_ingest,
        cmd_list,
        cmd_query,
        connect,
    )

DB_ENV = "BEAM_REGISTRY_DB"

BEAM_COLUMNS = [
    ("archive", "TEXT"), ("kind", "TEXT"), ("run_date", "TEXT"),
    ("source_date", "TEXT"), ("scale", "TEXT"),
    ("sample", "INTEGER"), ("top_k", "INTEGER"),
    ("context_memories", "INTEGER"),
    ("answer_model", "TEXT"), ("judge_model", "TEXT"),
    # levers the adapter never records (recording gap) -> NULL unless --set
    ("embeddings", "INTEGER"), ("facts", "INTEGER"),
    ("facts_extraction", "INTEGER"), ("graph_multihop", "INTEGER"),
    ("no_dream", "INTEGER"), ("distill", "INTEGER"),
    ("episode_granularity_enabled", "INTEGER"),
    ("aggregation_nodes_enabled", "INTEGER"),
    ("value_supersession_enabled", "INTEGER"),
    # scores (percent 0-100; dialect C converted from fraction)
    ("overall", "REAL"),
    ("ability_abs", "REAL"), ("ability_cr", "REAL"), ("ability_eo", "REAL"),
    ("ability_ie", "REAL"), ("ability_if", "REAL"), ("ability_ku", "REAL"),
    ("ability_mr", "REAL"), ("ability_pf", "REAL"), ("ability_sum", "REAL"),
    ("ability_tr", "REAL"),
    ("count", "INTEGER"), ("answer_calls", "INTEGER"),
    ("judge_calls", "INTEGER"), ("total_tokens", "INTEGER"),
    ("elapsed_s", "REAL"),
]

BEAM_ABILITIES = ("ABS", "CR", "EO", "IE", "IF", "KU", "MR", "PF", "SUM", "TR")

BEAM_OVERRIDES = {
    "sample", "top_k", "context_memories", "answer_model", "judge_model",
    "embeddings", "facts", "facts_extraction", "graph_multihop", "no_dream",
    "distill", "episode_granularity_enabled", "aggregation_nodes_enabled",
    "value_supersession_enabled",
}

# record-doc may also carry scores/counts (they come from a documented
# run, not from a file we parsed).
BEAM_DOC_OVERRIDES = BEAM_OVERRIDES | {
    "overall", "ability_abs", "ability_cr", "ability_eo", "ability_ie",
    "ability_if", "ability_ku", "ability_mr", "ability_pf", "ability_sum",
    "ability_tr", "count", "answer_calls", "judge_calls", "run_date",
}

# §6.5: beam artifacts are split across two dirs -- the v13-v16 archives
# sit beside the registry, the current results_*.json land in the beam
# run output dir.  Override with BEAM_ARTIFACT_DIRS (os.pathsep-separated).
ARTIFACT_DIRS = tuple(
    Path(p) for p in os.environ["BEAM_ARTIFACT_DIRS"].split(os.pathsep)
) if os.environ.get("BEAM_ARTIFACT_DIRS") else (
    DEFAULT_REGISTRY_DIR,
    Path.home() / "hymem_beam",
)

SPEC = {
    "db_file": "beam_runs.db",
    "artifact_dirs": ARTIFACT_DIRS,
    "columns": BEAM_COLUMNS,
    "overrides": BEAM_OVERRIDES,
    "patterns": ("beam-v*.json", "beam-*.json"),
    "excludes": ("latest", "comparison"),
    "kind_class": "beam",
    # §6 stamp policy: beam stems carry no \\d{8}T\\d{6}Z stamp
    # (beam-v14-preference-fix.json) -> recording NULL is the domain
    # truth, not a defect.  Rejudge artifacts carry stamps (source +
    # exec) and read the source date from their own rejudged_from.
    "stamp_policy": "optional",
    "gap_label": "levers (facts/facts_extraction/embeddings/no_dream/aggregation)",
    "gap_note": (
        "SELECT COUNT(*) FROM runs WHERE facts IS NULL AND facts_extraction IS NULL "
        "AND embeddings IS NULL AND no_dream IS NULL "
        "AND aggregation_nodes_enabled IS NULL"
    ),
}


def _beam_kind(name: str) -> str:
    if "rejudge" in name:
        return "rejudge"
    if name.startswith("beam-v"):
        return "archive"
    return "variant"


def _beam_row(data: dict, path: Path) -> dict:
    cfg = dict(data.get("config") or {})
    meta = dict(data.get("metadata") or {})
    scores = data.get("scores") or {}
    summary = data.get("summary") or {}

    if summary:  # Dialect C: {scale: {ABILITY: frac}}
        scale = next(iter(summary))
        raw = summary[scale]
        scores = {k: (v * 100 if v is not None else None)
                  for k, v in raw.items()}
        eff_cfg = {**meta, **cfg}
    else:  # Dialect A/B
        scale = (cfg.get("scales") or [None])[0]
        eff_cfg = cfg

    kind = _beam_kind(path.name)
    # §6: stamp-derived dates.  Rejudge: source_date from rejudged_from,
    # run_date = last stamp (rejudge exec); stats NULL (inherited-but-wrong
    # is worse than missing).  Archive/variant: first stem stamp or NULL.
    if kind == "rejudge":
        src_date, exec_date = rr.rejudge_dates(
            path.name, meta.get("rejudged_from") or "",
            SPEC.get("stamp_policy", "optional"))
        run_date = rr.iso_ts(exec_date or data.get("date") or meta.get("date"))
        source_date = src_date
        total_tokens = None
        elapsed_s = None
    else:
        run_date = rr.iso_ts(data.get("date") or meta.get("date"))
        source_date = rr.stem_source_date(
            path.name, SPEC.get("stamp_policy", "optional"))
        total_tokens = cfg.get("total_tokens")
        elapsed_s = meta.get("elapsed_s") or cfg.get("elapsed_s")

    row = {c: None for c, _ in BEAM_COLUMNS}
    row["archive"] = path.name
    row["kind"] = kind
    row["run_date"] = run_date
    row["source_date"] = source_date
    row["scale"] = scale
    row["sample"] = eff_cfg.get("sample") or (meta.get("sample") if meta else None)
    row["top_k"] = eff_cfg.get("top_k") or (meta.get("top_k") if meta else None)
    row["context_memories"] = eff_cfg.get("context_memories")
    row["answer_model"] = eff_cfg.get("answer_model")
    row["judge_model"] = eff_cfg.get("judge_model")
    for key in ("embeddings", "facts", "facts_extraction", "graph_multihop",
                "no_dream", "distill", "episode_granularity_enabled",
                "aggregation_nodes_enabled", "value_supersession_enabled"):
        row[key] = _coerce(eff_cfg.get(key), "INTEGER")
    for ab in BEAM_ABILITIES:
        v = scores.get(ab)
        row[f"ability_{ab.lower()}"] = (
            round(v, 3) if isinstance(v, (int, float)) else None)
    row["overall"] = (
        round(scores.get("OVERALL"), 3)
        if isinstance(scores.get("OVERALL"), (int, float)) else None)
    row["count"] = cfg.get("count") or (meta.get("count") if meta else None)
    row["answer_calls"] = cfg.get("answer_calls") or (meta.get("answer_calls") if meta else None)
    row["judge_calls"] = cfg.get("judge_calls") or (meta.get("judge_calls") if meta else None)
    row["total_tokens"] = total_tokens
    row["elapsed_s"] = elapsed_s
    row["extras"] = json_dumps({
        "config": data.get("config"),
        "metadata": data.get("metadata"),
        "scores": scores,
        "raw_json_keys": sorted(data.keys()),
    })
    return row


def json_dumps(obj):
    import json
    return json.dumps(obj, default=str)


_ROW_BUILDER = _beam_row
_KIND = _beam_kind


def _backfill(db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _beam_row
    return rr.cmd_backfill(spec, db_path=db_path)


def _ingest(files, overrides=None, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _beam_row
    return cmd_ingest(spec, files or None, overrides, db_path=db_path)


def _record_doc(archive, overrides=None, db_path=None):
    """Enter a run whose run-file is lost but is documented elsewhere
    (e.g. beam-results-history.md).  Provenance starts with
    'analyst:doc=' — never 'recorded'."""
    import json
    import sqlite3
    overrides = dict(overrides or {})
    con = connect(spec=dict(SPEC), db_path=db_path)
    ex = con.execute("SELECT id FROM runs WHERE archive=?", (archive,)).fetchone()
    if ex:
        return "skipped"
    row = {c: None for c, _ in BEAM_COLUMNS}
    applied = {}
    for k, v in overrides.items():
        if k not in BEAM_DOC_OVERRIDES:
            continue
        typ = dict(BEAM_COLUMNS).get(k, "TEXT")
        # §6.5: doc rows are analyst-set and carry bare dates.  cmd_backfill
        # canonicalises them in place too, but a row entered without a
        # later backfill would otherwise sit at width 10 in the sort
        # column -- so both entry points canonicalise on write.
        row[k] = rr.iso_ts(v) if k == "run_date" else _coerce(v, typ)
        applied[k] = row[k]
    row["archive"] = archive
    row["kind"] = "doc"
    row["source_date"] = "DOC"
    names = [c for c, _ in BEAM_COLUMNS] + ["flags_provenance", "extras"]
    vals = [row.get(c) for c in names]
    prov = "analyst:doc=" + archive
    if applied:
        prov += "; " + "; ".join(f"analyst:{k}={v}" for k, v in applied.items())
    vals[-2] = prov
    vals[-1] = json.dumps({"doc_row": True, "analyst_set": applied}, default=str)
    con.execute(
        f"INSERT INTO runs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
        vals)
    con.commit()
    print(f"row added (kind=doc, prov='{prov}')")


def _list(limit=30, flag=None, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _beam_row
    return cmd_list(spec, limit, flag, db_path)


def _query(sql, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _beam_row
    return cmd_query(spec, sql, db_path)


def _parse_set(args):
    ov = {}
    out = []
    i = 0
    while i < len(args):
        if args[i] == "--set" and i + 1 < len(args):
            k, v = args[i + 1].split("=", 1)
            ov[k] = v
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out, ov


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "ingest":
        files, ov = _parse_set(sys.argv[2:])
        _ingest(files or None, ov)
    elif cmd == "backfill":
        if _backfill():
            sys.exit(1)   # §6.5: unreachable rows were not migrated
    elif cmd == "record-doc":
        files, ov = _parse_set(sys.argv[2:])
        if not files:
            print("record-doc --archive NAME [--set k=v ...]")
            sys.exit(1)
        _record_doc(files[0], ov)
    elif cmd == "list":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--limit", type=int, default=30)
        p.add_argument("--flag")
        a = p.parse_args(sys.argv[2:])
        _list(a.limit, a.flag)
    elif cmd == "query":
        _query(" ".join(sys.argv[2:]))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
