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
