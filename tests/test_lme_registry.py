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
    make_run(bench / "longmemeval-v2-hymem-rejudged-deepseek-v4-flash.json")
    make_run(bench / "longmemeval-v2-hymem-baseline.json")
    mod.cmd_ingest([str(bench / "longmemeval-v2-hymem-rejudged-deepseek-v4-flash.json"),
                    str(bench / "longmemeval-v2-hymem-baseline.json")])
    con = sqlite3.connect(db)
    kinds = {a: k for a, k in con.execute("SELECT archive, kind FROM runs")}
    assert kinds["longmemeval-v2-hymem-rejudged-deepseek-v4-flash.json"] == "rejudge"
    assert kinds["longmemeval-v2-hymem-baseline.json"] == "variant"
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
