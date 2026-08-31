#!/usr/bin/env python3
"""Run registry for LoCoMo benchmark executions.

Same model as lme_registry.py / beam_registry.py: one row per run file,
date, scores, flags (recorded vs analyst-set, never guessed).  DB is
restart-safe: /home/node/.hermes/benchmarks/locomo_runs.db.

LoCoMo run files are HARSHER than the others:
- The adapter writes a BARE LIST of per-question rows to --out (no date,
  no config block, no flags, no scores).  Scores must be computed from
  the rows: overall (correct/n), answerable (cats 1-4), abstention
  (cat 5), and per-category rates.
- Diag-only files (all rows correct=null, e.g. locomo_conv26_diag.json)
  get overall=NULL and kind='diag' — recorded, not fabricated.
- Recovery-probe artifacts are probes, not runs, and are excluded by
  default (kind='probe' if explicitly passed).
- Documented runs whose JSON was lost (the canonical n=800 run of
  2026-07-29 lives only in locomo_adapter_spec.md) can be entered with
  `record-doc` — provenance is 'analyst:doc=...' so it can never be
  confused with a recorded run file.

Usage:
  locomo_registry.py ingest [FILE ...] [--set k=v ...]
  locomo_registry.py record-doc --archive NAME [--set k=v ...]
  locomo_registry.py list [--limit N] [--flag COL]
  locomo_registry.py query "SQL"
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

try:  # package import (tests): benchmarks.locomo_registry
    from . import run_registry as rr
except (ImportError, ValueError):  # direct CLI: python benchmarks/locomo_registry.py
    import run_registry as rr

DB_ENV = "LOCOMO_REGISTRY_DB"

LOCOMO_COLUMNS = [
    ("archive", "TEXT"), ("kind", "TEXT"), ("run_date", "TEXT"),
    ("source_date", "TEXT"),
    # flags — never recorded by the adapter; NULL unless --set
    ("sample", "INTEGER"), ("seed", "INTEGER"), ("workers", "INTEGER"),
    ("top_k", "INTEGER"), ("message_fts_top_k", "INTEGER"),
    ("rerank_top_k", "INTEGER"), ("fts_top_k", "INTEGER"),
    ("graph_top_k", "INTEGER"), ("max_context_chars", "INTEGER"),
    ("embeddings", "INTEGER"), ("no_dream", "INTEGER"),
    ("dream_per_session", "INTEGER"), ("facts", "INTEGER"),
    ("facts_extraction", "INTEGER"), ("rules_extraction", "INTEGER"),
    ("graph_multihop", "INTEGER"),
    ("user_speaker", "TEXT"), ("name_prefix", "INTEGER"),
    ("answerable_clause", "INTEGER"),
    # models
    ("answer_model", "TEXT"), ("judge_model", "TEXT"),
    # scores (percent 0-100; NULL for diag/probe)
    ("overall", "REAL"), ("answerable", "REAL"), ("abstention", "REAL"),
    ("cat_1", "REAL"), ("cat_2", "REAL"), ("cat_3", "REAL"),
    ("cat_4", "REAL"), ("cat_5", "REAL"),
    ("count", "INTEGER"), ("answer_calls", "INTEGER"),
    ("judge_calls", "INTEGER"),
]

LOCOMO_OVERRIDES = {
    "sample", "seed", "workers", "top_k", "message_fts_top_k",
    "rerank_top_k", "fts_top_k", "graph_top_k", "max_context_chars",
    "embeddings", "no_dream", "dream_per_session", "facts",
    "facts_extraction", "rules_extraction", "graph_multihop",
    "user_speaker", "name_prefix", "answerable_clause",
    "answer_model", "judge_model",
}

# record-doc may also carry the scores themselves (they come from the
# documented run, not from a file that computes them).
DOC_OVERRIDES = LOCOMO_OVERRIDES | {
    "overall", "answerable", "abstention",
    "cat_1", "cat_2", "cat_3", "cat_4", "cat_5",
    "count", "answer_calls", "judge_calls", "run_date",
}

SPEC = {
    "db_file": "locomo_runs.db",
    "columns": LOCOMO_COLUMNS,
    "overrides": LOCOMO_OVERRIDES,
    "patterns": ("locomo*.json", "locomo-*.json"),
    "excludes": ("recovery_probe_", "planD_", "locomo_stores"),
    # §6 stamp policy: locomo stems carry no \\d{8}T\\d{6}Z stamp
    # (locomo_conv26_diag.json) -> NULL is the domain truth, not a defect.
    "stamp_policy": "optional",
    "gap_label": "flags (the adapter records none of them in --out files)",
    "gap_note": (
        "SELECT COUNT(*) FROM runs WHERE sample IS NULL AND top_k IS NULL "
        "AND embeddings IS NULL AND facts IS NULL"
    ),
}

PROBE_PREFIX = "recovery_probe_"


def _locomo_kind(name: str) -> str:
    if name.startswith(PROBE_PREFIX):
        return "probe"
    if "diag" in name:
        return "diag"
    if "rejudged" in name:
        return "rejudge"
    return "archive"


def _locomo_row(data, path: Path) -> dict:
    row = {c: None for c, _ in LOCOMO_COLUMNS}
    row["archive"] = path.name
    row["kind"] = _locomo_kind(path.name)
    # §6: stamp-derived source date (NULL when the stem carries none).
    row["source_date"] = rr.stem_source_date(
        path.name, SPEC.get("stamp_policy", "optional"))
    if isinstance(data, dict):  # probe artifacts / metadata wrappers
        row["run_date"] = str(data.get("date", ""))[:19]
        rows = data.get("results") or data.get("rows") or []
        if not rows and "correct" not in data:
            rows = []
    else:
        rows = data if isinstance(data, list) else []
        row["run_date"] = ""  # bare lists record no date

    scored = [r for r in rows if isinstance(r.get("correct"), bool)]
    n = len(scored)
    if n:
        n_correct = sum(1 for r in scored if r["correct"])
        row["overall"] = round(n_correct / n * 100, 3)
        ans = [r for r in scored if r.get("category") != 5]
        abst = [r for r in scored if r.get("category") == 5]
        if ans:
            row["answerable"] = round(
                sum(1 for r in ans if r["correct"]) / len(ans) * 100, 3)
        if abst:
            row["abstention"] = round(
                sum(1 for r in abst if r["correct"]) / len(abst) * 100, 3)
        for c in range(1, 6):
            cc = [r for r in scored if r.get("category") == c]
            if cc:
                row[f"cat_{c}"] = round(
                    sum(1 for r in cc if r["correct"]) / len(cc) * 100, 3)
    row["count"] = n or (len(rows) if rows else None)
    row["extras"] = json.dumps({
        "n_rows": len(rows) if isinstance(rows, list) else 0,
        "n_scored": n,
        "raw_is_list": isinstance(data, list),
        "keys": sorted(rows[0].keys()) if isinstance(rows, list) and rows else [],
    }, default=str)
    return row


def _backfill(db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _locomo_row
    return rr.cmd_backfill(spec, db_path=db_path)


def _ingest(files, overrides=None, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _locomo_row
    return rr.cmd_ingest(spec, files or None, overrides, db_path=db_path)


def _record_doc(archive, overrides=None, db_path=None):
    """Enter a run whose run-file is lost but is documented elsewhere.

    Provenance starts with 'analyst:doc=' — NOT 'recorded'.  Only
    whitelisted columns can be set.  The archive string is the doc
    reference, e.g. 'locomo_adapter_spec.md:2026-07-29 n=800'.
    """
    overrides = dict(overrides or {})
    con = rr.connect(SPEC, db_path)
    ex = con.execute("SELECT id FROM runs WHERE archive=?", (archive,)).fetchone()
    if ex:
        return "skipped"
    row = {c: None for c, _ in LOCOMO_COLUMNS}
    applied = {}
    for k, v in overrides.items():
        if k not in DOC_OVERRIDES:
            continue
        typ = dict(SPEC["columns"]).get(k, "TEXT")
        row[k] = rr._coerce(v, typ)
        applied[k] = row[k]
    row["archive"] = archive
    row["kind"] = "doc"
    row["source_date"] = "DOC"
    names = [c for c, _ in LOCOMO_COLUMNS] + ["flags_provenance", "extras"]
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
    print(f"DB: {db_path or rr.DEFAULT_REGISTRY_DIR / SPEC['db_file']}  row added (kind=doc, prov='{prov}')")


def _list(limit=30, flag=None, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _locomo_row
    return rr.cmd_list(spec, limit, flag, db_path)


def _query(sql, db_path=None):
    spec = dict(SPEC)
    spec["builder"] = _locomo_row
    return rr.cmd_query(spec, sql, db_path)


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
        _backfill()
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
