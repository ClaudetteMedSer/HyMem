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
import re
import sqlite3
from pathlib import Path

DEFAULT_REGISTRY_DIR = Path("/home/node/.hermes/benchmarks")

# ---------------------------------------------------------------- stamps ---
# The registry's date-provenance policy (pre-registration §6, 2026-08-31):
# `source_date` is the filename's timestamp stamp, never a truncated stem.
#   * LME adapter files always carry a \\d{8}T\\d{6}Z stamp (archive +
#     rejudge names) -> policy "required": a missing stamp RAISES.  A NULL
#     there is a defect, not a domain fact, and would look identical to a
#     legitimately stamp-less beam/locomo row.
#   * beam/locomo stems do not carry stamps (beam-v14-preference-fix.json,
#     locomo_conv26_diag.json) -> policy "optional": NULL records the
#     absence; the registry never guesses.
#   * Rejudge artifacts: first stamp = the SOURCE run's date, last stamp =
#     the rejudge execution date.  Source pointer is read from the
#     artifact's own rejudged_from (reliable for both LME and beam) rather
#     than from the artifact stem when available.

STAMP_RE = re.compile(r"\d{8}T\d{6}Z")

# The same stamp as it appears in a run_date value.  The trailing Z is
# optional here (it is NOT in STAMP_RE, which matches stamps embedded in
# filenames): a stamp that lost its Z would otherwise fail every branch of
# iso_ts and pass through at width 15 -- a mixed width in the sort column,
# which is the exact defect §6.5 exists to close.
COMPACT_TS_RE = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z?")


def stem_stamps(name: str) -> list[str]:
    """All \\d{8}T\\d{6}Z stamps in a filename, in order of appearance."""
    return STAMP_RE.findall(name or "")


def stem_source_date(name: str, policy: str = "optional") -> str | None:
    """First timestamp stamp in a filename — the run's own source date.

    policy='optional': no stamp -> None (beam/locomo stems carry none;
    the registry records the absence, never guesses).
    policy='required': no stamp -> ValueError — a benchmark declared
    stamp-bearing (LME adapter files) must always yield a stamp here.
    """
    stamps = stem_stamps(name)
    if not stamps:
        if policy == "required":
            raise ValueError(
                f"no timestamp stamp ({STAMP_RE.pattern}) in {name!r}: this "
                "benchmark's filenames are declared stamp-bearing "
                "(policy='required') — a missing stamp is a defect, not a "
                "domain fact"
            )
        return None
    return stamps[0]


def rejudge_dates(own_name: str, source_name: str | None = None,
                  policy: str = "optional") -> tuple[str | None, str | None]:
    """(source_date, run_date) for a REJUDGE artifact filename.

    source_date = first stamp of the artifact's own recorded source
    pointer (rejudged_from) — the only reliable source of the source
    run's date; falls back to the first stamp of the own stem when it
    carries >= 2 stamps (LME/beam rejudge stems embed source-stem +
    exec-stamp), else None (a 1-stamp stem is the exec stamp only —
    beam rejudge of a stamp-less v13-v16 source).
    run_date = last stamp of the own stem (the rejudge execution time),
    None when the stem carries no exec stamp.

    policy='required' -> ValueError when the stem carries no stamp at all.
    """
    stamps = stem_stamps(own_name)
    if not stamps:
        if policy == "required":
            raise ValueError(
                f"no timestamp stamp ({STAMP_RE.pattern}) in {own_name!r}: "
                "declared stamp-bearing (policy='required') — a missing "
                "stamp is a defect, not a domain fact"
            )
        return None, None
    src_stamps = stem_stamps(source_name) if source_name else []
    if src_stamps:
        source_date = src_stamps[0]
    elif len(stamps) >= 2:
        source_date = stamps[0]
    else:
        source_date = None
    return source_date, stamps[-1]

def iso_ts(ts) -> str | None:
    """Normalise a run_date to canonical 19-char ISO 'YYYY-MM-DDTHH:MM:SS'.

    §6.5: run_date is the registry's sort key -- the only date column any
    ORDER BY reads (cmd_list here, _list in lme_registry).  SQLite sorts
    TEXT lexicographically, so mixing formats in that column silently
    reorders the table: at index 4 a compact stamp has '0' (0x30) where
    ISO has '-' (0x2D), so EVERY compact row outranks EVERY ISO row
    regardless of real time, and a 10-char bare date sorts below any
    19-char value from the same day.  Fixed width is the property that
    makes lexicographic ordering chronological; nothing else here does.

    All four shapes present in the live registries are normalised:
      '20260831T200531Z'     stem stamp    -> '2026-08-31T20:05:31'
      '20260831T200531'      stamp, no Z   -> '2026-08-31T20:05:31'
      '2026-08-31T16:50:39'  ISO datetime  -> unchanged
      '2026-06-09'           bare date     -> '2026-06-09T00:00:00'
      '' / None              absent        -> None

    Absent becomes None rather than '': source_date already records
    absence as NULL (§6.1), and run_date recording it as '' left one
    clause with two conventions -- '' also sorts to the bottom on DESC
    and to the top on ASC, so it is never merely cosmetic.

    source_date is deliberately NOT routed through this: it is the
    provenance pointer -- the literal filename stamp, greppable against
    the artifact name -- and is never sorted on.  Normalising it would
    destroy the property that makes it auditable.
    """
    if ts is None:
        return None
    s = str(ts).strip()
    if not s:
        return None
    m = COMPACT_TS_RE.fullmatch(s)
    if m:
        return "{}-{}-{}T{}:{}:{}".format(*m.groups())
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s + "T00:00:00"
    return s[:19] or None


# Per-benchmark specs.  Each entry:
#   db_file      - SQLite file under DEFAULT_REGISTRY_DIR
#   artifact_dirs- dirs searched for a row's archive file, in order.
#                  A benchmark's artifacts need not all live in one
#                  place (beam: beam-v*.json in the registry dir,
#                  results_*.json in the run output dir), and a
#                  backfill that cannot reach a row cannot migrate it.
#   columns      - ordered list of (name, type) for flags+scores
#   overrides    - column names settable via --set (whitelist)
#   patterns     - glob patterns searched by default ingest
#   excludes     - filename substrings never auto-ingested
#   builder      - row_builder(data, path) -> dict (column -> value)
#   kind_class   - kind_from_name(name) -> str
#   gap_note     - SQL condition counted for the print-time NOTE


# ----------------------------------------------------------- arm evidence ---
#
# An A/B is two artifacts and a claim that they differ in one lever. The claim
# has to come from somewhere, and a filename is not somewhere: `guard-epg-on`
# and `guard-epg-off` are the operator's memory of what they typed, written
# into a stem. Neither is a `--set` in this registry -- that is the same memory,
# recorded later, by the same person.
#
# The 2026-08-30/31 episode-granularity guard is the worked example. Both arms
# ran BEFORE 6543ee6 taught the adapter to write the lever into its config
# block, so the two blocks are byte-identical except `elapsed_s`, and nothing
# inside either file says which arm it is. The scores came out 71.0 and 71.0.
# That is exactly what a real null looks like -- and exactly what two runs of
# the SAME configuration look like. The artifacts cannot separate the two
# readings, so the pair cannot discharge a non-regression gate on that lever
# however clean the numbers are.
#
# This is the unreachable-code-path shape from docs/diagnostic_controls.md,
# arriving one level up: not an instrument that never touched the lever, but a
# pair of results that cannot show whether it did.

def is_retrieval_only(artifact) -> bool:
    """Did this run make no answer or judge call?

    `--retrieval-only` produces an artifact with a full complement of rows and
    NO verdicts, so that `f` -- the fraction of questions a lever moves -- can
    be measured without paying for the reader. Every scorer must REFUSE such an
    artifact rather than caveat it: `accuracy` reads `correct: None` as
    unscored and returns 0.0 on an empty denominator, which prints as a real
    number, and a 0% arm beside a 69% arm looks like a catastrophic regression
    rather than a category error.

    Reads the config first and falls back to the rows, so an artifact written
    before the flag existed -- or hand-edited -- still cannot slip through."""
    artifact = artifact or {}
    if (artifact.get("config") or {}).get("retrieval_only"):
        return True
    rows = artifact.get("per_question") or []
    return bool(rows) and all(r.get("retrieval_only") for r in rows)


ARM_EVIDENCED = "EVIDENCED"      # both blocks record the lever, and they differ
ARM_SAME = "SAME_ARM"            # both record it, and they agree
ARM_UNEVIDENCED = "UNEVIDENCED"  # at least one block does not record it


def arm_evidence(cfg_a, cfg_b, lever, ignore=("elapsed_s", "total_tokens")):
    """Can these two config blocks evidence that they are opposite arms?

    Returns (verdict, note, confounds) where `confounds` lists the OTHER keys
    that also differ -- an evidenced contrast is still not a clean one if the
    arms moved more than one lever.

    Deliberately reads only the artifacts. Anything an analyst can supply after
    the fact is the claim under test, not evidence for it."""
    a, b = dict(cfg_a or {}), dict(cfg_b or {})
    missing = [n for n, c in (("A", a), ("B", b)) if lever not in c]
    confounds = sorted(
        k for k in set(a) | set(b)
        if k != lever and k not in ignore and a.get(k) != b.get(k))
    if missing:
        return ARM_UNEVIDENCED, (
            f"{lever!r} is absent from the config block of arm(s) "
            f"{', '.join(missing)}, so neither file states which arm it is; "
            f"a stem or a --set is the operator's recollection, not a record "
            f"of what ran"), confounds
    if a[lever] == b[lever]:
        return ARM_SAME, (
            f"both arms recorded {lever}={a[lever]!r} -- this pair is not an "
            f"A/B on that lever, whatever it is named"), confounds
    return ARM_EVIDENCED, (
        f"{lever}: A={a[lever]!r} B={b[lever]!r}, recorded in both blocks"
    ), confounds


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
    # Benchmark registries are long-lived.  CREATE TABLE IF NOT EXISTS does
    # not evolve an existing table when a newly auditable field is added, so
    # apply a narrow additive migration for every declared column.  Names and
    # types come from the in-repository static spec, never from run artifacts.
    existing = {
        row[1] for row in con.execute("PRAGMA table_info(runs)").fetchall()
    }
    for name, typ in spec["columns"] + [
        ("flags_provenance", "TEXT"), ("extras", "TEXT")
    ]:
        if name not in existing:
            con.execute(f"ALTER TABLE runs ADD COLUMN {name} {typ}")
            existing.add(name)


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


def cmd_backfill(spec, bench_dir=None, db_path=None):
    """§6.4: recompute stamp-derived fields for EVERY row from its artifact
    file; UPDATE where the value differs.  The read-back is a DIFF against
    the pre-backfill values, not a spot check that new values parse — the
    interesting failure is a row that changes when it shouldn't.

    Fields recomputed: every field in field_list (source_date, run_date,
    total_tokens, elapsed_s -- whichever the spec declares) for every
    non-doc/probe row.  The builder decides what each kind yields; the
    backfill does not special-case rejudge.
    doc/probe rows: source_date and stats untouched (no artifact to
    recompute from), run_date canonicalised in place -- it shares the
    sort column with every other row.
    Rows whose artifact no declared dir can supply are UNREACHABLE, not
    'skipped': they were not migrated, and the count is returned so the
    CLI can exit nonzero.
    """
    con = connect(spec, db_path)
    # §6.5: search every declared artifact dir.  An explicit bench_dir
    # overrides the spec (single-dir callers and tests).
    if bench_dir is not None:
        search_dirs = [Path(bench_dir)]
    else:
        search_dirs = [Path(d) for d in
                       (spec.get("artifact_dirs") or [DEFAULT_REGISTRY_DIR])]
    field_list = [c for c in ("source_date", "run_date", "total_tokens",
                              "elapsed_s") if any(c == n for n, _ in spec["columns"])]
    cols = ["id", "archive", "kind"] + field_list
    rows = con.execute(f"SELECT {', '.join(cols)} FROM runs ORDER BY id").fetchall()
    changed, missing, unreachable = [], [], []
    for r in rows:
        _id, archive, kind = r[0], r[1], r[2]
        cur = dict(zip(field_list, r[3:]))
        if kind == "doc" or kind == "probe":
            # §6.5: doc/probe rows have no artifact to recompute from, but
            # their analyst-typed run_date still shares the sort column.
            # Canonicalise it in place so the migration leaves no mixed
            # widths behind; idempotent, and a no-op once already ISO.
            if "run_date" in field_list:
                norm = iso_ts(cur["run_date"])
                if norm != cur["run_date"]:
                    changed.append((_id, archive, kind,
                                    [("run_date", cur["run_date"], norm)]))
                    con.execute("UPDATE runs SET run_date=? WHERE id=?",
                                (norm, _id))
            continue
        p = next((d / archive for d in search_dirs if (d / archive).exists()),
                 None)
        if p is None:
            unreachable.append((_id, archive, "no artifact in " + ", ".join(
                str(d) for d in search_dirs)))
            continue
        try:
            data = json.loads(Path(p).read_text())
        except Exception as e:  # noqa: BLE001
            missing.append((_id, archive, f"unreadable: {e}"))
            continue
        try:
            new = spec["builder"](data, Path(p))
        except ValueError as e:
            # loud defect (e.g. stamp-bearing LME name without a stamp)
            missing.append((_id, archive, f"recompute failed: {e}"))
            continue
        diffs = [(f, cur[f], new.get(f)) for f in field_list
                 if cur[f] != new.get(f)]
        if not diffs:
            continue
        changed.append((_id, archive, kind, diffs))
        set_clause = ", ".join(f"{f}=?" for f in field_list)
        con.execute(f"UPDATE runs SET {set_clause} WHERE id=?",
                    [new.get(f) for f in field_list] + [_id])
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
        # §6.5: a row the backfill could not read is a row it did not
        # migrate.  Reporting that as a benign "skipped" is how a
        # half-done migration passes for a finished one -- the caller
        # gets a nonzero count so the CLI can exit nonzero.
        print("  NOTE: unreachable rows were NOT migrated -- fix the "
              "artifact path or spec['artifact_dirs'] and re-run")
    return len(unreachable)


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
