#!/usr/bin/env python3
"""Generic run-registry core for HyMem benchmarks (BEAM, LoCoMo).

One SQLite DB per benchmark, stored restart-safe under
/home/node/.hermes/benchmarks/<bench>_runs.db (persistent home volume —
NEVER /tmp; the container wipes /tmp on restart).

For each run it records: date, scores, effective flags (on/off), and the
provenance of those flags — recorded in the run JSON vs analyst-supplied
(--set) — so a silent config flip in library defaults stays queryable.

Rules (established 2026-08-30 with the LME registry):
- Record exactly what the run JSON says; NULL, never guess.
- `--set k=v` marks the value analyst-supplied in flags_provenance and
  extras.analyst_set — a derived value can't masquerade as recorded.
- One row per archive file; re-ingest is idempotent (UNIQUE archive).
- DBs live beside the LME registry: /home/node/.hermes/benchmarks/.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

DEFAULT_REGISTRY_DIR = Path("/home/node/.hermes/benchmarks")

# Per-benchmark specs.  Each entry:
#   db_file      - SQLite file under DEFAULT_REGISTRY_DIR
#   columns      - ordered list of (name, type) for flags+scores
#   overrides    - column names settable via --set (whitelist)
#   patterns     - glob patterns searched by default ingest
#   excludes     - filename substrings never auto-ingested
#   builder      - row_builder(data, path) -> dict (column -> value)
#   kind_class   - kind_from_name(name) -> str
#   gap_note     - SQL condition counted for the print-time NOTE


def _to_int(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    try:
        s = str(v).strip()
        if s == "":
            return None
        f = float(s)
        return int(f) if f.is_integer() else round(f)
    except (TypeError, ValueError):
        return None


def _coerce(v, typ):
    if v is None:
        return None
    if typ == "INTEGER":
        return _to_int(v)
    if typ == "REAL":
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return str(v)


def _make_table(con, spec):
    typed = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
    typed += [f"{name} {typ}" for name, typ in spec["columns"]]
    typed += ["flags_provenance TEXT", "extras TEXT"]
    con.executescript(
        f"CREATE TABLE IF NOT EXISTS runs (\n    {', '.join(typed)}\n);\n"
        "CREATE INDEX IF NOT EXISTS idx_run_date ON runs(run_date);\n"
    )


def connect(spec, db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or (DEFAULT_REGISTRY_DIR / spec["db_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    _make_table(con, spec)
    return con


def ingest_file(con, spec, path, overrides=None) -> str:
    """Insert one run JSON.  Returns 'inserted' | 'skipped' | 'error: ...'."""
    overrides = overrides or {}
    try:
        data = json.loads(Path(path).read_text())
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"
    row = spec["builder"](data, Path(path))
    # Whitelisted overrides only; anything else is ignored loudly.
    applied = {}
    for k, v in overrides.items():
        if k not in spec["overrides"]:
            continue
        col_types = dict(spec["columns"])
        row[k] = _coerce(v, col_types.get(k, "TEXT"))
        applied[k] = row[k]
    archive = row["archive"] = Path(path).name

    ex = con.execute("SELECT id FROM runs WHERE archive=?", (archive,)).fetchone()
    if ex:
        return "skipped"

    prov = "recorded" if not applied else "recorded + " + "; ".join(
        f"analyst:{k}={v}" for k, v in applied.items())
    extras = json.loads(row.pop("extras", "{}") or "{}")
    extras["analyst_set"] = applied
    row["flags_provenance"] = prov
    row["extras"] = json.dumps(extras, default=str)

    names = [c for c, _ in spec["columns"]] + ["flags_provenance", "extras"]
    vals = [row.get(c) for c in names]
    try:
        con.execute(
            f"INSERT INTO runs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
            vals)
        return "inserted"
    except sqlite3.IntegrityError:
        return "skipped"


def cmd_ingest(spec, files=None, overrides=None, bench_dir=None, db_path=None):
    con = connect(spec, db_path)
    bench_dir = Path(bench_dir or DEFAULT_REGISTRY_DIR)
    if not files:
        files = sorted({f for p in spec["patterns"]
                        for f in bench_dir.glob(p)
                        if not any(x in f.name for x in spec.get("excludes", ()))})
    stats = {}
    for f in files:
        r = ingest_file(con, spec, f, overrides)
        stats[r] = stats.get(r, 0) + 1
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    db_path = db_path or (DEFAULT_REGISTRY_DIR / spec["db_file"])
    print(f"DB: {db_path}  rows={total}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    note = spec.get("gap_note")
    if note:
        n = con.execute(note).fetchone()[0]
        if n:
            print(f"  NOTE: {n} rows record none of the {spec.get('gap_label', 'core levers')} "
                  "-- fill via --set with known launch flags")


def cmd_list(spec, limit=30, flag=None, db_path=None):
    con = connect(spec, db_path)
    cols = ["id", "run_date", "archive", "kind"]
    if any(c == "overall" for c, _ in spec["columns"]):
        cols.append("overall")
    if flag:
        cols.append(flag)
    q = f"SELECT {', '.join(cols)} FROM runs ORDER BY run_date DESC LIMIT ?"
    rows = con.execute(q, (limit,)).fetchall()
    print(" | ".join(cols))
    print("-" * 120)
    for r in rows:
        print(" | ".join("-" if v is None else str(v) for v in r))


def cmd_query(spec, sql, db_path=None):
    con = connect(spec, db_path)
    cur = con.execute(sql)
    print(" | ".join(d[0] for d in cur.description))
    for r in cur.fetchall():
        print(" | ".join("-" if v is None else str(v) for v in r))
