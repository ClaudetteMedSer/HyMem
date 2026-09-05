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
from benchmarks.strictness import build_manifest  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path):
    return tmp_path / "runs.db"


def _row(con, archive):
    return con.execute("SELECT * FROM runs WHERE archive=?", (archive,)).fetchone()


def _cols(con):
    return [d[0] for d in con.execute("SELECT * FROM runs LIMIT 0").description]


def _strict_usage(calls, prompt, completion, *, latency=1.0):
    return {
        "calls": calls, "calls_available": True,
        "request_attempts": calls, "request_attempts_available": True,
        "successful_responses": calls,
        "successful_responses_available": True,
        "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "latency_s": latency, "latency_available": True,
        "cost_usd": None, "cost_available": False,
        "token_usage_available": True,
    }


def _strict_local_embedding_usage():
    return {
        "configured": True, "backend": "local_feature_hash",
        "quality": "lexical", "network_free": True,
        "model": "feature-hash-v1", "dimension": 384,
        "identity_consistent": True, "instances": 2,
        "calls": 6, "calls_available": True,
        "request_attempts": 0, "request_attempts_available": True,
        "successful_responses": 0,
        "successful_responses_available": True,
        "input_count": 30, "input_count_available": True,
        "input_characters": 600, "input_characters_available": True,
        "prompt_tokens": None, "total_tokens": None,
        "provider_token_usage_available": False,
        "latency_s": 0.5, "latency_available": True,
        "cost_usd": None, "cost_available": False,
    }


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
    assert r[idx.index("run_date")] is None          # absent -> NULL (§6.5)
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
    assert r[idx.index("run_date")] == "2026-06-09T00:00:00"   # §6.5 padded
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


def _make_strict_beam(path: Path, *, segment_status="complete"):
    embedding = {
        "configured": True, "backend": "local-hash",
        "model": "feature-hash-v1", "base_url": "local://feature-hash",
        "dimension": 384, "quality": "lexical-feature-hash",
        "network_free": True, "fallback_policy": "none",
        "fallback_reason": None,
    }
    config = {
        "scales": ["100K", "500K"], "sample": 1,
        "sample_strategy": "seeded-label-blind-hash-v1",
        "subset_run": True, "top_k": 7,
        "max_input_tokens": 16000, "indexing_max_cycles": 100,
        "indexing_timeout_s": 3600.0,
        "indexing_require_healthy": True,
        "facts": True, "facts_extraction": False,
        "embedding": embedding,
        "effective_hymem_config": {
            "facts_enabled": True,
            "facts_extraction_enabled": False,
            "graph_multihop_enabled": False,
            "aggregation_nodes_enabled": False,
            "episode_granularity_enabled": False,
            "value_supersession_enabled": True,
        },
        "judge_protocol": "official",
        "official_judge_protocol_match": True,
        "official_protocol_aligned": True,
        "official_denominator_validated": False,
        "official_judge_prompt_hash": (
            benchmarks.beam_registry.BEAM_OFFICIAL_JUDGE_PROMPT_HASH
        ),
        "official_judge_upstream_commit": (
            benchmarks.beam_registry.BEAM_UPSTREAM_COMMIT
        ),
        "official_judge_evaluator_url": (
            benchmarks.beam_registry.BEAM_OFFICIAL_EVALUATOR_URL
        ),
        "official_judge_prompt_url": (
            benchmarks.beam_registry.BEAM_OFFICIAL_PROMPT_URL
        ),
        "oracle_ability": False, "judge_gold": True,
        "prereg": {
            "path": "benchmarks/beam-prereg.md", "commit": "b" * 40,
            "blob": "c" * 40,
            "committed_at": "2026-09-04T10:00:00+00:00",
            "code_commit": "d" * 40,
        },
        "dataset_revisions": {
            benchmarks.beam_registry.BEAM_REPO: "a" * 40
        },
        "dataset_revision_provenance_complete": True,
        "answer_extra_body": {}, "judge_extra_body": {},
        "label_free_answer_path": True,
        "exploratory_label_steering": False,
        "exploratory_non_comparable": True,
        "scored_run": True,
    }
    models = {
        "reader": {
            "provider": "deepseek", "model": "reader-pinned",
            "base_url": "https://api.deepseek.com",
        },
        "judge": {
            "provider": "openai", "model": "gpt-4.1-mini",
            "base_url": benchmarks.beam_registry.OFFICIAL_JUDGE_BASE_URL,
            "temperature": 0.0, "max_tokens": None, "extra_body": {},
            "protocol": "official",
            "upstream_commit": benchmarks.beam_registry.BEAM_UPSTREAM_COMMIT,
            "prompt_hash": (
                benchmarks.beam_registry.BEAM_OFFICIAL_JUDGE_PROMPT_HASH
            ),
        },
        "memory_pipeline": {
            "provider": "openai-compatible", "model": "pipeline-pinned",
            "base_url": "https://api.deepseek.com",
            "thinking_mode": "off", "effective_extra_body": {},
        },
        "embedding": embedding,
    }
    rows = []
    score_by_scale = {scale: {} for scale in ("100K", "500K")}
    for scale in ("100K", "500K"):
        for ability in benchmarks.beam_registry.BEAM_ABILITIES:
            scores_for_ability = []
            for ordinal in range(2):
                failed = (
                    scale == "100K" and ability == "CR" and ordinal == 0
                )
                score = (
                    (1.0 if ability == "ABS" else 0.5)
                    if scale == "100K"
                    else (1.0 if ability == "CR" else 0.0)
                )
                if failed:
                    score = 0.0
                scores_for_ability.append(score)
                criterion_scores = (
                    [1.0, 1.0] if score == 1.0
                    else [0.0, 1.0] if score == 0.5
                    else [0.0, 0.0]
                )
                rubric = ["criterion one", "criterion two"]
                criteria = [{
                    "criterion_index": criterion_index,
                    "rubric_item": rubric_item,
                    "raw": json.dumps({
                        "score": criterion_score, "reason": "audited reason",
                    }),
                    "finish_reason": "stop", "parse": "ok",
                    "score": criterion_score, "reason": "audited reason",
                } for criterion_index, (rubric_item, criterion_score) in enumerate(
                    zip(rubric, criterion_scores)
                )]
                rows.append({
                    "question_id": (
                        f"beam:{scale}:conversation-{scale}:{ability}:{ordinal}"
                    ),
                    "scale": scale, "conv_id": f"conversation-{scale}",
                    "ability": ability, "oracle_ability": ability,
                    "detected_ability": None, "ability_used": None,
                    "question": f"Question for {ability}?",
                    "answer": "A benchmark answer",
                    "rubric": rubric, "score": score,
                    "llm_judge_score": score,
                    "correct": score == 1.0 and not failed,
                    "result_valid": not failed, "judge_protocol": "official",
                    "judge_parse": "not_called" if failed else "ok",
                    "scores": [] if failed else criterion_scores,
                    "judge_criterion_results": [] if failed else criteria,
                    "benchmark_failure": (
                        "conversation_failure: fixture" if failed else None
                    ),
                })
            score_by_scale[scale][ability] = (
                sum(scores_for_ability) / len(scores_for_ability)
            )
    expected_ids = [row["question_id"] for row in rows]
    manifest = build_manifest(
        benchmark="BEAM", code_sha256="sha256:" + "c" * 64,
        data_sha256="sha256:" + "d" * 64,
        config=config, models=models, seed=17, expected_ids=expected_ids,
        protocol_split="full",
    )
    summary = {}
    summary_counts = {}
    for scale in ("100K", "500K"):
        summary[scale] = dict(score_by_scale[scale])
        scale_rows = [row for row in rows if row["scale"] == scale]
        summary[scale]["OVERALL"] = (
            sum(row["score"] for row in scale_rows) / len(scale_rows)
        )
        summary_counts[scale] = {
            ability: 2 for ability in benchmarks.beam_registry.BEAM_ABILITIES
        }
        summary_counts[scale]["OVERALL"] = 20
    data = {
        "benchmark": "BEAM",
        "version": "strict-v1",
        "date": "2026-09-04T12:00:01+00:00",
        "manifest": manifest,
        "config": manifest["config"],
        "models": manifest["models"],
        "summary": summary,
        "summary_counts": summary_counts,
        "execution": {
            "counts": {
                "expected": 40, "attempted": 40, "unique_attempted": 40,
                "total_attempts": 40, "completed": 39,
                "failed": 1, "missing": 0,
            },
            "segments": [{
                "segment_id": "process-a", "status": segment_status,
                "attempted_attempts": 40,
                "elapsed_s": 3.5,
                "reader_usage": _strict_usage(39, 4, 6),
                "judge_usage": _strict_usage(78, 12, 8),
                "memory_pipeline_usage": _strict_usage(4, 10, 20),
                "embedding_usage": _strict_local_embedding_usage(),
            }],
        },
        "per_question": rows,
    }
    path.write_text(json.dumps(data))
    return path


def test_beam_strict_archive_discovery_multiscale_and_metadata(
    tmp_db, tmp_path,
):
    archive = _make_strict_beam(
        tmp_path / "results_20260904T120000Z-strict-deadbeef.json"
    )
    # Mutable pointers are never auto-ingested as a second result row.
    (tmp_path / "results_latest.json").write_text(json.dumps({
        "archive": archive.name,
        "run_id": json.loads(archive.read_text())["manifest"]["run_id"],
    }))
    spec = dict(benchmarks.beam_registry.SPEC)
    spec["builder"] = benchmarks.beam_registry._beam_row
    benchmarks.run_registry.cmd_ingest(
        spec, bench_dir=tmp_path, db_path=tmp_db
    )
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    row = _row(con, archive.name)
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert row[idx.index("kind")] == "archive"
    assert row[idx.index("source_date")] == "20260904T120000Z"
    assert row[idx.index("run_date")] == "2026-09-04T12:00:01"
    assert row[idx.index("scale")] == "100K,500K"
    assert row[idx.index("overall")] == pytest.approx(31.25)
    assert row[idx.index("ability_abs")] == pytest.approx(50.0)
    assert row[idx.index("answer_model")] == "reader-pinned"
    assert row[idx.index("judge_model")] == "gpt-4.1-mini"
    assert row[idx.index("count")] == 40
    assert row[idx.index("answer_calls")] == 39
    assert row[idx.index("judge_calls")] == 78
    assert row[idx.index("total_tokens")] == 60
    assert row[idx.index("elapsed_s")] == pytest.approx(3.5)
    assert row[idx.index("embeddings")] == 1
    assert row[idx.index("facts")] == 1
    assert row[idx.index("facts_extraction")] == 0
    assert row[idx.index("graph_multihop")] == 0
    assert row[idx.index("no_dream")] == 0
    assert row[idx.index("distill")] == 0
    assert row[idx.index("aggregation_nodes_enabled")] == 0
    assert row[idx.index("value_supersession_enabled")] == 1
    assert row[idx.index("run_id")].startswith("sha256:")
    assert row[idx.index("protocol_split")] == "full"
    assert row[idx.index("development_only")] == 1
    assert row[idx.index("exploratory_non_comparable")] == 1
    assert row[idx.index("label_free_answer_path")] == 1
    assert row[idx.index("judge_protocol")] == "official"
    assert row[idx.index("official_judge_protocol_match")] == 1
    assert row[idx.index("dataset_revisions_complete")] == 1
    extras = json.loads(row[idx.index("extras")])
    assert extras["execution_disclosure"]["segments_complete"] is True
    assert extras["execution"]["segments"][0]["segment_id"] == "process-a"
    assert extras["manifest"]["run_id"] == row[idx.index("run_id")]


def test_beam_strict_incomplete_segment_never_claims_exact_usage(
    tmp_db, tmp_path,
):
    archive = _make_strict_beam(
        tmp_path / "results_20260904T120000Z-strict-incomplete.json",
        segment_status="running",
    )
    benchmarks.beam_registry._ingest([archive], db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    row = _row(con, archive.name)
    assert row[idx.index("count")] == 40
    assert row[idx.index("answer_calls")] is None
    assert row[idx.index("judge_calls")] is None
    assert row[idx.index("total_tokens")] is None
    assert row[idx.index("elapsed_s")] is None
    extras = json.loads(row[idx.index("extras")])
    assert extras["execution_disclosure"]["segments_complete"] is False


def test_beam_strict_multiscale_without_counts_is_rejected(tmp_path):
    archive = _make_strict_beam(
        tmp_path / "results_20260904T120000Z-strict-malformed.json"
    )
    data = json.loads(archive.read_text())
    data.pop("summary_counts")
    with pytest.raises(ValueError, match="stored summary/counts are partial"):
        benchmarks.beam_registry._beam_row(data, archive)


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
    assert r[idx.index("run_date")] == "2026-08-31T20:05:31"     # exec stamp
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
    assert r[idx.index("run_date")] == "2026-08-31T20:05:31"


def test_beam_backfill_canonicalises_doc_rows(tmp_db, tmp_path):
    # §6.5: doc rows have no artifact to recompute from, so cmd_backfill
    # skips the rebuild -- but their analyst-typed run_date still shares the
    # sort column, and a bare date sits at width 10 among 19-char values.
    # Without this test, deleting the doc branch from cmd_backfill is silent.
    spec = dict(benchmarks.beam_registry.SPEC)
    spec["builder"] = benchmarks.beam_registry._beam_row
    benchmarks.run_registry.connect(spec, db_path=tmp_db).close()
    names = [c for c, _ in spec["columns"]] + ["flags_provenance", "extras"]
    con = sqlite3.connect(tmp_db)
    con.execute(f"INSERT INTO runs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                ["beam-results-history.md:2026-06-09 e3c8955 51.3%", "doc",
                 "2026-06-09", "DOC", None] + [None] * (len(names) - 5))
    con.commit()
    con.close()

    benchmarks.run_registry.cmd_backfill(spec, bench_dir=tmp_path, db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    r = con.execute("SELECT * FROM runs").fetchone()
    assert r[idx.index("run_date")] == "2026-06-09T00:00:00"
    assert r[idx.index("source_date")] == "DOC"   # provenance untouched
    con.close()

    # Idempotent: the canonical value must survive a second pass unchanged.
    benchmarks.run_registry.cmd_backfill(spec, bench_dir=tmp_path, db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    assert con.execute("SELECT run_date FROM runs").fetchone()[0] == "2026-06-09T00:00:00"
    con.close()


def test_beam_backfill_unreachable_row_is_not_a_silent_skip(tmp_db, tmp_path):
    # §6.5: a row whose artifact no dir can supply was NOT migrated.  It must
    # be counted as unreachable (the CLI exits nonzero on it), not folded in
    # with unreadable/recompute-failed rows where it reads as benign.
    spec = dict(benchmarks.beam_registry.SPEC)
    spec["builder"] = benchmarks.beam_registry._beam_row
    benchmarks.run_registry.connect(spec, db_path=tmp_db).close()
    names = [c for c, _ in spec["columns"]] + ["flags_provenance", "extras"]
    con = sqlite3.connect(tmp_db)
    con.execute(f"INSERT INTO runs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                ["results_20260831T165039Z.json", "variant",
                 "2026-08-31T16:50:39", None, None] + [None] * (len(names) - 5))
    con.commit()
    con.close()
    assert benchmarks.run_registry.cmd_backfill(
        spec, bench_dir=tmp_path, db_path=tmp_db) == 1


def test_beam_backfill_searches_every_artifact_dir(tmp_db, tmp_path):
    # §6.5: beam artifacts are split across two dirs.  Resolution must try
    # each declared dir, or the CLI (which passes no bench_dir) can never
    # reach the results_*.json rows -- the ones a rejudge actually writes.
    other = tmp_path / "hymem_beam"
    other.mkdir()
    a = _make_beam_results(other / "results_20260831T165039Z.json")
    spec = dict(benchmarks.beam_registry.SPEC)
    spec["builder"] = benchmarks.beam_registry._beam_row
    spec["artifact_dirs"] = (tmp_path, other)
    benchmarks.run_registry.connect(spec, db_path=tmp_db).close()
    names = [c for c, _ in spec["columns"]] + ["flags_provenance", "extras"]
    con = sqlite3.connect(tmp_db)
    con.execute(f"INSERT INTO runs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                [a.name, "variant", "2026-08-31T16:50:39", "results_20260831",
                 None] + [None] * (len(names) - 5))
    con.commit()
    con.close()

    # No bench_dir: resolution comes from spec["artifact_dirs"] alone.
    assert benchmarks.run_registry.cmd_backfill(spec, db_path=tmp_db) == 0
    con = sqlite3.connect(tmp_db)
    idx = _cols(con)
    assert _row(con, a.name)[idx.index("source_date")] == "20260831T165039Z"
    con.close()


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
    assert rb[idx.index("run_date")] == "2026-08-31T20:05:31"
    assert rb[idx.index("total_tokens")] is None
    assert rb[idx.index("elapsed_s")] is None
    # Idempotence: a second backfill must not touch anything (the diff is
    # against pre-backfill values; nothing should change a second time).
    con.close()

    benchmarks.run_registry.cmd_backfill(spec, bench_dir=tmp_path, db_path=tmp_db)
    con = sqlite3.connect(tmp_db)
    rb2 = _row(con, b.name)
    assert rb2[idx.index("run_date")] == "2026-08-31T20:05:31"
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
    assert r[idx.index("run_date")] is None          # absent -> NULL (§6.5)


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
