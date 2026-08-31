"""Tests for benchmarks/lme_registry.py.

Registry semantics under test:
- ingest records exactly the flags present in the run JSON (no guessing);
- idempotency: same file twice -> single row;
- --set overrides land in the row AND are flagged in flags_provenance /
  extras.analyst_set, so a derived value never masquerades as recorded;
- kind classification: archive / variant / rejudge;
- old-format lever (mr_aggregate_additive) is captured separately from
  the new-format lever (aggregation_nodes_enabled).
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import benchmarks.lme_registry as reg  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "runs.db"
    bench = tmp_path / "bench"
    bench.mkdir()
    monkeypatch.setenv("LME_REGISTRY_DB", str(db))
    monkeypatch.setenv("LME_BENCH_DIR", str(bench))
    # Re-import so module-level DB/constants pick up the patched env.
    import importlib
    mod = importlib.reload(reg)
    return mod, db, bench


def make_run(path: Path, date="2026-08-05T05:49:14", **cfg):
    run = {
        "benchmark": "LongMemEval",
        "version": "v2-hymem-tr-mr-wired",
        "date": date,
        "config": {
            "scale": "s", "sample": 0, "seed": 0, "top_k": 15,
            "auto_ability": True, "workers": 8, "no_dream": False,
            "permissive_default": True, "embeddings": False,
            "answer_model": "deepseek-v4-flash",
            "judge_model": "deepseek-v4-flash",
            "answer_calls": 500, "judge_calls": 500,
            "total_tokens": 1000000, "elapsed_s": 10000.0,
            **cfg,
        },
        "scores": {"OVERALL": {"accuracy": 71.4, "count": 500},
                   "multi-session": {"accuracy": 59.4, "count": 133}},
        "per_question": [],
    }
    path.write_text(json.dumps(run))


def test_ingest_records_only_present_flags(tmp_db):
    mod, db, bench = tmp_db
    make_run(bench / "longmemeval-v2-hymem-20260805T054914Z-seed0.json",
             aggregation_nodes_enabled=True, episode_granularity_enabled=False)
    mod.cmd_ingest([str(bench / "longmemeval-v2-hymem-20260805T054914Z-seed0.json")])
    con = sqlite3.connect(db)
    row = con.execute("SELECT * FROM runs").fetchone()
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    r = dict(zip(cols, row))
    assert r["auto_ability"] == 1
    assert r["aggregation_nodes_enabled"] == 1
    assert r["episode_granularity_enabled"] == 0
    assert r["overall"] == 71.4
    assert r["kind"] == "archive"
    assert r["flags_provenance"] == "recorded"
    con.close()


def test_ingest_is_idempotent(tmp_db):
    mod, db, bench = tmp_db
    p = bench / "longmemeval-v2-hymem-20260805T054914Z-seed0.json"
    make_run(p)
    mod.cmd_ingest([str(p)])
    mod.cmd_ingest([str(p)])
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    con.close()


def test_set_override_is_provenance_flagged(tmp_db):
    mod, db, bench = tmp_db
    p = bench / "longmemeval-v2-hymem-20260805T054914Z-seed0.json"
    make_run(p)  # config omits the two levers entirely
    mod.cmd_ingest([str(p)], {"aggregation_nodes_enabled": 0,
                              "episode_granularity_enabled": 1})
    con = sqlite3.connect(db)
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    r = dict(zip(cols, con.execute("SELECT * FROM runs").fetchone()))
    assert r["aggregation_nodes_enabled"] == 0
    assert r["episode_granularity_enabled"] == 1
    assert "analyst" in r["flags_provenance"]
    extras = json.loads(r["extras"])
    assert extras["analyst_set"] == {"aggregation_nodes_enabled": 0,
                                     "episode_granularity_enabled": 1}
    con.close()


def test_kind_classification(tmp_db):
    mod, db, bench = tmp_db
    # §6: rejudge *adapter* names always carry two stamps (source + exec);
    # a stamp-less rejudge name is a defect and now RAISES (see
    # test_missing_stamp_archive_raises) — so the kind test uses the
    # realistic stamped name.
    rj = "longmemeval-v2-hymem-20260610T094858Z-seed0-rejudged-deepseek-v4-flash-20260725T191314Z.json"
    make_run(bench / rj, date="2026-06-10T09:48:58",
             rejudged_from="longmemeval-v2-hymem-20260610T094858Z-seed0.json")
    make_run(bench / "longmemeval-v2-hymem-baseline.json")
    mod.cmd_ingest([str(bench / rj), str(bench / "longmemeval-v2-hymem-baseline.json")])
    con = sqlite3.connect(db)
    kinds = {a: k for a, k in con.execute("SELECT archive, kind FROM runs")}
    assert kinds[rj] == "rejudge"
    assert kinds["longmemeval-v2-hymem-baseline.json"] == "variant"
    con.close()


def test_ingest_archive_source_date_is_stamp(tmp_db):
    mod, db, bench = tmp_db
    make_run(bench / "longmemeval-v2-hymem-20260805T054914Z-seed0.json",
             total_tokens=1718606, elapsed_s=8404.2232)
    mod.cmd_ingest([str(bench / "longmemeval-v2-hymem-20260805T054914Z-seed0.json")])
    con = sqlite3.connect(db)
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    r = dict(zip(cols, con.execute("SELECT * FROM runs").fetchone()))
    # §6: full stamp, not the old truncated stem[:16]
    assert r["source_date"] == "20260805T054914Z"
    assert r["run_date"] == "2026-08-05T05:49:14"
    assert r["total_tokens"] == 1718606
    con.close()


def test_missing_stamp_archive_raises_loud(tmp_db):
    # stamp-bearing (LME adapter) names must yield a stamp; NULL there is
    # a defect, not a domain fact — the registry raises instead of
    # recording a value that would look like a legitimate beam/locomo row.
    mod, db, bench = tmp_db
    make_run(bench / "longmemeval-v2-hymem-no-stamp-seed0.json")
    with pytest.raises(ValueError, match="stamp"):
        mod.cmd_ingest([str(bench / "longmemeval-v2-hymem-no-stamp-seed0.json")])


def test_rejudge_row_dates_and_null_stats(tmp_db):
    # the actual id=41 shape: artifact 'date' is the SOURCE's date; the
    # rejudge rows must read run_date from the exec stamp, source_date
    # from the source pointer, and record NULL stats.
    mod, db, bench = tmp_db
    rj = ("longmemeval-v2-hymem-20260610T094858Z-seed0-rejudged-"
          "deepseek-v4-flash-20260725T191314Z.json")
    make_run(bench / rj, date="2026-06-10T09:48:58",  # inherited source date
             rejudged_from="longmemeval-v2-hymem-20260610T094858Z-seed0.json",
             total_tokens=1718606, elapsed_s=8404.2232)  # inherited stats
    mod.cmd_ingest([str(bench / rj)])
    con = sqlite3.connect(db)
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    r = dict(zip(cols, con.execute("SELECT * FROM runs").fetchone()))
    assert r["kind"] == "rejudge"
    assert r["run_date"] == "20260725T191314Z"     # exec stamp, not source date
    assert r["source_date"] == "20260610T094858Z"  # source pointer stamp
    assert r["total_tokens"] is None               # §6.2: inherited => NULL
    assert r["elapsed_s"] is None
    con.close()


def test_backfill_diff_and_idempotence(tmp_db):
    # Seed the DB with old-builder values (truncated stems, inherited
    # stats) and verify the read-back is a diff against pre-backfill
    # values; exactly the wrong fields flip; a second run is a no-op.
    mod, db, bench = tmp_db
    src = "longmemeval-v2-hymem-20260610T094858Z-seed0.json"
    rj = ("longmemeval-v2-hymem-20260610T094858Z-seed0-rejudged-"
          "deepseek-v4-flash-20260725T191314Z.json")
    make_run(bench / src, date="2026-06-10T09:48:58",
             total_tokens=1718606, elapsed_s=8404.2232)
    make_run(bench / rj, date="2026-06-10T09:48:58",
             rejudged_from=src, total_tokens=1718606, elapsed_s=8404.2232)
    con = sqlite3.connect(db)
    mod.connect().close()  # ensure schema exists before seeding
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    seed_cols = [c for c in cols if c != "id"]
    def seed(archive, kind, run_date, source_date, **kw):
        row = dict.fromkeys(seed_cols)  # type: ignore[var-annotated]
        row.update({"archive": archive, "kind": kind, "run_date": run_date,
                    "source_date": source_date,
                    "answer_calls": 500, "judge_calls": 500,
                    "total_tokens": 1718606, "elapsed_s": 8404.2232,
                    "flags_provenance": "recorded", "extras": "{}", **kw})
        con.execute(f"INSERT INTO runs ({', '.join(seed_cols)}) "
                    f"VALUES ({', '.join('?' * len(seed_cols))})",
                    [row[c] for c in seed_cols])
    # old-builder shape: truncated stem source_date, rejudge inherits the
    # source's run_date/stats
    seed(src, "archive", "2026-06-10T09:48:58", "longmemeval-v2-h")
    seed(rj, "rejudge", "2026-06-10T09:48:58", "longmemeval-v2-h")
    con.commit()
    con.close()

    mod.cmd_backfill()
    con = sqlite3.connect(db)
    rows = {}
    for r in con.execute("SELECT * FROM runs"):
        d = dict(zip(cols, r))
        rows[d["archive"]] = d
    sr = rows[src]
    rr = rows[rj]
    # archive row: source_date fixed; run_date + stats identical (no false diff)
    assert sr["source_date"] == "20260610T094858Z"
    assert sr["run_date"] == "2026-06-10T09:48:58"
    assert sr["total_tokens"] == 1718606
    # rejudge row: run_date -> exec stamp, source_date -> stamp, stats NULLed
    assert rr["run_date"] == "20260725T191314Z"
    assert rr["source_date"] == "20260610T094858Z"
    assert rr["total_tokens"] is None
    assert rr["elapsed_s"] is None
    con.close()

    mod.cmd_backfill()  # idempotence: no row should change a second time
    con = sqlite3.connect(db)
    rr2 = dict(zip(cols, con.execute(
        "SELECT * FROM runs WHERE archive=?", (rj,)).fetchone()))
    assert rr2["run_date"] == "20260725T191314Z"
    assert rr2["elapsed_s"] is None
    con.close()


def test_old_format_lever_captured_separately(tmp_db):
    mod, db, bench = tmp_db
    make_run(bench / "longmemeval-v2-hymem-20260605T071918Z-seed0.json",
             mr_aggregate_additive=False, **{"aggregation_nodes_enabled": None})
    mod.cmd_ingest([str(bench / "longmemeval-v2-hymem-20260605T071918Z-seed0.json")])
    con = sqlite3.connect(db)
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    r = dict(zip(cols, con.execute("SELECT * FROM runs").fetchone()))
    assert r["mr_aggregate_additive"] == 0
    con.close()
