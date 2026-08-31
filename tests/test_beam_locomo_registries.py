"""Tests for benchmarks/run_registry.py + beam_registry.py + locomo_registry.py.

Same bar as test_lme_registry.py:
- BEAM: dialect A (date+config+scores %), dialect B (no date),
  dialect C (metadata+summary fractions -> %);
- flags from config are recorded; levers never recorded stay NULL;
- dialect C fractions are converted to percent (comparable across dialects);
- --set override lands in row AND flags_provenance / extras.analyst_set;
- idempotency: same file twice -> one row;
- LoCoMo: bare per-question rows -> scores computed (overall/answerable/
  abstention/cats); diag rows (correct=null) -> kind='diag', overall NULL;
- LoCoMo record-doc: provenance starts with 'analyst:doc=', never 'recorded'.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import benchmarks.beam_registry  # noqa: E402
import benchmarks.locomo_registry  # noqa: E402
import benchmarks.run_registry  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path):
    return tmp_path / "runs.db"


def _row(con, archive):
    return con.execute("SELECT * FROM runs WHERE archive=?", (archive,)).fetchone()


def _cols(con):
    return [d[0] for d in con.execute("SELECT * FROM runs LIMIT 0").description]


# ---------------------------------------------------------------------------
# BEAM
# ---------------------------------------------------------------------------

def make_beam(path: Path, dialect="A", date="2026-06-01T14:58:04Z", **cfg):
    cfg.setdefault("scales", ["100K"])
    cfg.setdefault("sample", 5)
    cfg.setdefault("top_k", 10)
    cfg.setdefault("answer_model", "deepseek-chat")
    cfg.setdefault("judge_model", "deepseek-chat")
    if dialect == "C":
        data = {
            "metadata": {
                "date": date,
                "answer_model": cfg["answer_model"],
                "judge_model": cfg["judge_model"],
                "scales": cfg["scales"],
                "sample": cfg["sample"],
                "top_k": cfg["top_k"],
                "context_memories": cfg["top_k"] * 3,
                "elapsed_s": 949.1,
                "answer_calls": 100,
                "judge_calls": 100,
            },
            "summary": {"100K": {"ABS": 1.0, "CR": 0.05, "OVERALL": 0.4743888888888888}},
            "conversations": [],
        }
    else:
        data = {
            "benchmark": "BEAM",
            "version": "v13-hymem-beam-optimisation",
            "date": None if dialect == "B" else date,
            "config": cfg,
            "scores": {"ABS": 100.0, "CR": 10.0, "OVERALL": 37.0},
        }
    path.write_text(json.dumps(data))
    return path


def test_beam_dialect_a_records_config_and_null_levers(tmp_db, tmp_path):
    p = make_beam(tmp_path / "beam-v13-hymem.json")
    benchmarks.beam_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    r = _row(con, p.name)
    assert r is not None
    idx = _cols(con)
    assert r[idx.index("run_date")] == "2026-06-01T14:58:04"
    assert r[idx.index("overall")] == 37.0          # percent, as stored
    assert r[idx.index("sample")] == 5
    assert r[idx.index("top_k")] == 10
    assert r[idx.index("ability_abs")] == 100.0
    assert r[idx.index("ability_cr")] == 10.0
    # Levers never in config -> NULL, not guessed
    assert r[idx.index("facts")] is None
    assert r[idx.index("embeddings")] is None
    assert r[idx.index("no_dream")] is None
    assert r[idx.index("aggregation_nodes_enabled")] is None
    assert r[idx.index("flags_provenance")] == "recorded"


def test_beam_dialect_b_no_date(tmp_db, tmp_path):
    p = make_beam(tmp_path / "beam-v15-mr-tr-fix.json", dialect="B")
    benchmarks.beam_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, p.name)
    assert r[idx.index("run_date")] == ""            # recorded as absent, no guess
    assert r[idx.index("overall")] == 37.0


def test_beam_dialect_c_fractions_to_percent(tmp_db, tmp_path):
    p = make_beam(tmp_path / "beam-v16-mr-tr-fix.json", dialect="C")
    benchmarks.beam_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, p.name)
    assert r[idx.index("run_date")] == "2026-06-01T14:58:04"
    assert r[idx.index("overall")] == pytest.approx(47.439)
    assert r[idx.index("ability_abs")] == 100.0
    assert r[idx.index("ability_cr")] == 5.0
    assert r[idx.index("context_memories")] == 30
    assert r[idx.index("answer_calls")] == 100
    assert r[idx.index("elapsed_s")] == pytest.approx(949.1)


def test_beam_idempotent_and_set_provenance(tmp_db, tmp_path):
    p = make_beam(tmp_path / "beam-v13-hymem.json")
    benchmarks.beam_registry._ingest([p], db_path=tmp_db)
    benchmarks.beam_registry._ingest([p], db_path=tmp_db)  # same file again
    con = sqlite3.connect(tmp_db)
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    # Analyst-set lever lands + provenance marks it
    benchmarks.beam_registry._ingest([make_beam(tmp_path / "beam-v14.json")],
                                     {"facts": 1, "no_dream": 0}, db_path=tmp_db)
    idx = _cols(con)
    r = _row(con, "beam-v14.json")
    assert r[idx.index("facts")] == 1
    assert r[idx.index("no_dream")] == 0
    assert "analyst:facts=1" in r[idx.index("flags_provenance")]
    ex = json.loads(r[idx.index("extras")])
    assert ex["analyst_set"] == {"facts": 1, "no_dream": 0}


def test_beam_override_whitelist_ignores_unknown(tmp_db, tmp_path):
    p = make_beam(tmp_path / "beam-v13-hymem.json")
    benchmarks.beam_registry._ingest([p], {"not_a_column": 42}, db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, p.name)
    assert r[idx.index("flags_provenance")] == "recorded"


def test_beam_record_doc_provenance(tmp_db, tmp_path):
    benchmarks.beam_registry._record_doc(
        "beam-results-history.md:2026-06-09 e3c8955 51.3%",
        {"overall": 51.3, "ability_ie": 63.9, "sample": 3,
         "run_date": "2026-06-09"},
        db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, "beam-results-history.md:2026-06-09 e3c8955 51.3%")
    assert r is not None
    assert r[idx.index("kind")] == "doc"
    assert r[idx.index("overall")] == pytest.approx(51.3)
    assert r[idx.index("ability_ie")] == pytest.approx(63.9)
    assert r[idx.index("run_date")] == "2026-06-09"
    prov = r[idx.index("flags_provenance")]
    assert prov.startswith("analyst:doc=")
    assert "recorded" not in prov


# ---------------------------------------------------------------------------
# §6 stamp semantics (2026-08-31)
# ---------------------------------------------------------------------------

def _make_beam_results(path: Path, date="2026-08-31T16:50:39.701802+00:00",
                       **meta_extra):
    """Dialect-C beam artifact with a stamped results_* filename (run A)."""
    data = {
        "metadata": {
            "date": date,
            "answer_model": "deepseek-chat",
            "judge_model": "deepseek-chat",
            "sample": 5,
            "top_k": 10,
            "elapsed_s": 5471.57,
            "answer_calls": 160,
            "judge_calls": 160,
            **meta_extra,
        },
        "summary": {"100K": {"ABS": 1.0, "OVERALL": 0.51}},
        "conversations": [],
    }
    path.write_text(json.dumps(data))
    return path


def test_beam_stampless_archive_source_date_null(tmp_db, tmp_path):
    p = make_beam(tmp_path / "beam-v13-hymem.json")
    benchmarks.beam_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    r = _row(con, p.name)
    assert r[_cols(con).index("source_date")] is None


def test_beam_stamped_archive_source_date_from_stamp(tmp_db, tmp_path):
    p = _make_beam_results(tmp_path / "results_20260831T165039Z.json")
    benchmarks.beam_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, p.name)
    assert r[idx.index("source_date")] == "20260831T165039Z"
    assert r[idx.index("run_date")] == "2026-08-31T16:50:39"
    assert r[idx.index("elapsed_s")] == pytest.approx(5471.57)  # own stats kept


def test_beam_rejudge_dates_and_null_stats(tmp_db, tmp_path):
    # Run B shape: stamped source pointer + exec stamp in the own stem.
    p = _make_beam_results(
        tmp_path / "results_20260831T165039Z-rejudged-deepseek-chat-20260831T200531Z.json",
        date="2026-08-31T20:05:31.089405+00:00",
        rejudged_from="results_20260831T165039Z.json",
        judge_gold=True, a_date="2026-08-31T16:50:39.701802+00:00",
        elapsed_s=290.9, answer_calls=0, judge_calls=161)
    benchmarks.beam_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, p.name)
    assert r[idx.index("kind")] == "rejudge"
    assert r[idx.index("source_date")] == "20260831T165039Z"  # source's stamp
    assert r[idx.index("run_date")] == "20260831T200531Z"     # exec stamp
    assert r[idx.index("total_tokens")] is None               # §6.2: NULL
    assert r[idx.index("elapsed_s")] is None                  # §6.2: NULL


def test_beam_rejudge_one_stamp_source_date_null(tmp_db, tmp_path):
    # beam rejudge of a stamp-less v13-v16 source: the only stamp is the
    # exec — source_date must NOT read the exec stamp as the source date.
    p = _make_beam_results(
        tmp_path / "beam-v16-mr-tr-fix-rejudged-deepseek-v4-flash-20260831T200531Z.json",
        date="2026-08-31T20:05:31.089405+00:00",
        rejudged_from="beam-v16-mr-tr-fix.json", judge_gold=True)
    benchmarks.beam_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, p.name)
    assert r[idx.index("source_date")] is None
    assert r[idx.index("run_date")] == "20260831T200531Z"


def test_beam_backfill_diff_and_idempotence(tmp_db, tmp_path):
    # Seed the DB with OLD-builder values (truncated stems, inherited stats)
    # and check the read-back is a diff against pre-backfill values, that
    # exactly the wrong fields flip, and that a second run changes nothing.
    a = _make_beam_results(tmp_path / "results_20260831T165039Z.json")
    b = _make_beam_results(
        tmp_path / "results_20260831T165039Z-rejudged-deepseek-chat-20260831T200531Z.json",
        date="2026-08-31T20:05:31.089405+00:00",
        rejudged_from="results_20260831T165039Z.json",
        judge_gold=True, a_date="2026-08-31T16:50:39.701802+00:00",
        elapsed_s=290.9, answer_calls=0, judge_calls=161)
    con = sqlite3.connect(tmp_db)
    spec = dict(benchmarks.beam_registry.SPEC)
    spec["builder"] = benchmarks.beam_registry._beam_row
    benchmarks.run_registry.connect(spec, db_path=tmp_db).close()  # create schema
    names = [c for c, _ in spec["columns"]] + ["flags_provenance", "extras"]
    # old-style A row: truncated stem source_date, own meta run_date
    con.execute(f"INSERT INTO runs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                [a.name, "variant", "2026-08-31T16:50:39", "results_20260831",
                 None] + [None] * (len(names) - 5))
    con.execute(f"INSERT INTO runs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                [b.name, "rejudge", "2026-08-31T20:05:31", "results_20260831",
                 None] + [None] * (len(names) - 5))
    con.commit()
    con.close()

    benchmarks.run_registry.cmd_backfill(spec, bench_dir=tmp_path, db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    ra = _row(con, a.name)
    rb = _row(con, b.name)
    # A: only source_date moved (run_date was already own-meta-derived)
    assert ra[idx.index("source_date")] == "20260831T165039Z"
    assert ra[idx.index("run_date")] == "2026-08-31T16:50:39"
    assert ra[idx.index("elapsed_s")] == pytest.approx(5471.57)
    # B: source_date fixed, run_date fixed to exec stamp, stats NULLed
    assert rb[idx.index("source_date")] == "20260831T165039Z"
    assert rb[idx.index("run_date")] == "20260831T200531Z"
    assert rb[idx.index("total_tokens")] is None
    assert rb[idx.index("elapsed_s")] is None
    # Idempotence: a second backfill must not touch anything (the diff is
    # against pre-backfill values; nothing should change a second time).
    con.close()

    benchmarks.run_registry.cmd_backfill(spec, bench_dir=tmp_path, db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    rb2 = _row(con, b.name)
    assert rb2[idx.index("run_date")] == "20260831T200531Z"
    assert rb2[idx.index("elapsed_s")] is None
    con.close()


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------

def make_locomo(path: Path, rows=None, date=None):
    rows = rows or [
        {"id": i, "category": (i % 5) + 1, "correct": True, "question": f"q{i}"}
        for i in range(20)
    ]
    data = rows if date is None else {"date": date, "results": rows}
    path.write_text(json.dumps(data))
    return path


def test_locomo_rows_compute_scores(tmp_db, tmp_path):
    rows = [
        {"category": 1, "correct": True},
        {"category": 1, "correct": False},
        {"category": 2, "correct": True},
        {"category": 5, "correct": True},   # cat5 on-topic abstention answered correctly
        {"category": 5, "correct": False},
    ]
    p = make_locomo(tmp_path / "locomo_results_20260729T000000Z.json", rows)
    benchmarks.locomo_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, p.name)
    assert r[idx.index("overall")] == pytest.approx(60.0)     # 3/5
    assert r[idx.index("answerable")] == pytest.approx(66.667)  # 2/3
    assert r[idx.index("abstention")] == pytest.approx(50.0)    # 1/2
    assert r[idx.index("cat_1")] == pytest.approx(50.0)
    assert r[idx.index("cat_2")] == pytest.approx(100.0)
    assert r[idx.index("cat_5")] == pytest.approx(50.0)
    assert r[idx.index("count")] == 5
    assert r[idx.index("flags_provenance")] == "recorded"
    # No date in bare lists -> recorded as absent
    assert r[idx.index("run_date")] == ""


def test_locomo_diag_rows_no_score(tmp_db, tmp_path):
    rows = [{"category": 1, "correct": None}, {"category": 2, "correct": None}]
    p = make_locomo(tmp_path / "locomo_conv26_diag.json", rows)
    benchmarks.locomo_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, p.name)
    assert r[idx.index("kind")] == "diag"
    assert r[idx.index("overall")] is None
    assert r[idx.index("count")] == 2


def test_locomo_probe_classified(tmp_db, tmp_path):
    p = make_locomo(tmp_path / "recovery_probe_locomo26_1304.json",
                    rows=[{"category": 1, "correct": True}])
    benchmarks.locomo_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, p.name)
    assert r[idx.index("kind")] == "probe"
    assert r[idx.index("overall")] == 100.0


def test_locomo_record_doc_provenance(tmp_db, tmp_path):
    benchmarks.locomo_registry._record_doc(
        "locomo_adapter_spec.md:2026-07-29 n=800",
        {"overall": 74.1, "answerable": 68.2, "sample": 800},
        db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = _row(con, "locomo_adapter_spec.md:2026-07-29 n=800")
    assert r is not None
    assert r[idx.index("kind")] == "doc"
    assert r[idx.index("overall")] == pytest.approx(74.1)
    assert r[idx.index("answerable")] == pytest.approx(68.2)
    prov = r[idx.index("flags_provenance")]
    assert prov.startswith("analyst:doc=")
    assert "analyst:sample=800" in prov      # doc row marks all values analyst-set
    assert "recorded" not in prov


def test_locomo_record_doc_idempotent(tmp_db, tmp_path):
    a = "locomo_adapter_spec.md:2026-07-29 n=800"
    benchmarks.locomo_registry._record_doc(a, {"overall": 74.1}, db_path=tmp_db)
    assert benchmarks.locomo_registry._record_doc(a, {"overall": 74.1}, db_path=tmp_db) == "skipped"
    con = sqlite3.connect(tmp_db)
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# §6 stamp semantics (2026-08-31)
# ---------------------------------------------------------------------------

def test_locomo_source_date_null_without_stamp(tmp_db, tmp_path):
    p = make_locomo(tmp_path / "locomo_conv26_diag.json",
                    rows=[{"category": 1, "correct": None}])
    benchmarks.locomo_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    r = _row(con, p.name)
    assert r[_cols(con).index("source_date")] is None


def test_locomo_source_date_stamp_when_present(tmp_db, tmp_path):
    p = make_locomo(tmp_path / "locomo_results_20260729T000000Z.json")
    benchmarks.locomo_registry._ingest([p], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    r = _row(con, p.name)
    assert r[_cols(con).index("source_date")] == "20260729T000000Z"


def test_locomo_backfill_diff_and_idempotence(tmp_db, tmp_path):
    p = make_locomo(tmp_path / "locomo_conv26_diag.json",
                    rows=[{"category": 1, "correct": None}])
    con = sqlite3.connect(tmp_db)
    spec = dict(benchmarks.locomo_registry.SPEC)
    spec["builder"] = benchmarks.locomo_registry._locomo_row
    benchmarks.run_registry.connect(spec, db_path=tmp_db).close()  # create schema
    names = [c for c, _ in spec["columns"]] + ["flags_provenance", "extras"]
    # old-builder row: truncated stem source_date
    con.execute(f"INSERT INTO runs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                [p.name, "diag", "", "locomo_conv26_di"] + [None] * (len(names) - 4))
    con.commit()
    con.close()

    benchmarks.run_registry.cmd_backfill(
        spec, bench_dir=tmp_path, db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    r = _row(con, p.name)
    assert r[_cols(con).index("source_date")] is None
    con.close()

    # idempotent: second run must not touch the row
    benchmarks.run_registry.cmd_backfill(spec, bench_dir=tmp_path, db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    r2 = _row(con, p.name)
    assert r2[_cols(con).index("source_date")] is None
    con.close()
