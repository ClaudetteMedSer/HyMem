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
from benchmarks.strictness import build_manifest, content_hash  # noqa: E402
from benchmarks.lme_protocol import (  # noqa: E402
    LME_EVALUATOR_COMMIT,
    LME_EVALUATOR_SHA256,
    LME_EVALUATOR_URL,
    LME_LOCAL_RETRY_POLICY,
    LME_UPSTREAM_RETRY_POLICY,
)


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


def make_strict_run(path: Path, *, segment_status="complete"):
    config = {
        "scales": "S", "sample": 2, "seed": 3, "top_k": 15,
        "workers": 2, "auto_ability": True, "no_dream": False,
        "permissive_default": False, "embeddings": False,
        "graph_facts_first": False, "distill": False,
        "distill_prompt_version": "v2",
        "label_free_answer_path": True, "scored_run": True,
        "retrieval_only": False,
        "exploratory_label_steering": False,
        "exploratory_non_comparable": True,
        "subset_run": True,
        "official_denominator_validated": False,
        "source_order_validated": False,
        "indexing_require_healthy": True,
        "historical_local_judge_prompts_exact_official": False,
        "official_judge_match": False,
        "source_ids_hash": content_hash(["q1", "q2"]),
        "source_qtype_counts": {"multi-session": 2},
        "dataset_revision": "fixture-revision",
        "dataset_sha256": content_hash("data"),
        "dataset_expected_count": 2,
        "sample_strategy": "sha256-seed-source-index-preserve-order-v1",
        "prereg": None,
        "judge_protocol": "legacy-custom",
        "max_input_tokens": None,
        "max_input_bytes": 60000,
        "provider_context_tokens": 65536,
        "indexing_max_cycles": 4,
        "indexing_timeout_s": 900.0,
        "rerank_top_k": None,
        "graph_multihop_max_hops": None,
        "graph_multihop_decay": None,
        "graph_multihop_min_score": None,
        "context_policy": {
            "name": "conservative-utf8-byte-query-head-tail-v2",
            "budget_unit": "utf8_bytes",
            "max_input_tokens": None,
            "max_input_bytes": 60000,
            "provider_context_window_tokens": 65536,
            "reserved_output_tokens": 1024,
            "reserved_transport_overhead_tokens": 256,
            "tokenizer": None,
            "tokenizer_failure_policy": "not-applicable",
            "source_boundaries": ["head", "query-window", "tail"],
            "raw_evidence_reserve_fraction": 0.60,
            "min_semantic_excerpt_alnum": 8,
            "min_semantic_excerpt_chars": 12,
            "gold_access": False,
        },
        "judge_transport_retry_policy": LME_LOCAL_RETRY_POLICY,
        "official_transport_retry_policy": LME_UPSTREAM_RETRY_POLICY,
        "official_transport_exact": False,
        "retrieval_usage_owner": "none",
        "answer_model": "reader-pinned",
        "answer_base_url": "https://reader.example/v1",
        "answer_extra_body_obj": {},
        "judge_model": "judge-pinned",
        "judge_base_url": "https://judge.example/v1",
        "judge_extra_body_obj": {},
        "hymem_model": "pipeline-pinned",
        "hymem_base_url": "https://pipeline.example/v1",
        "hymem_thinking": "off",
        "embedding_runtime": {
            "configured": False, "backend": "none", "quality": "none",
            "network_free": True, "model": None, "base_url": None,
            "dimension": None, "fallback_policy": "none",
        },
        "aggregation_nodes": False,
        "aggregation_broad": False,
        "episode_granularity": True,
        "value_supersession": True,
        "graph_multihop": False,
        "rerank_model": None,
        "rerank_message_hits": None,
        "rules": None,
        "rules_extraction": None,
        "facts": None,
        "facts_extraction": None,
        "evaluator_commit": LME_EVALUATOR_COMMIT,
        "evaluator_sha256": LME_EVALUATOR_SHA256,
        "evaluator_url": LME_EVALUATOR_URL,
        "effective_hymem_config": {
            "message_fts_top_k": 15, "fts_top_k": 10,
            "graph_top_k": 10,
            "aggregation_nodes_enabled": False,
            "aggregation_inject_abilities": ["TR"],
            "episode_granularity_enabled": True,
            "value_supersession_enabled": True,
            "graph_multihop_enabled": False,
            "rerank_top_k": 20,
            "rerank_model": "llm",
            "rerank_message_hits": True,
            "graph_multihop_max_hops": 2,
            "graph_multihop_decay": 0.5,
            "graph_multihop_min_score": 0.05,
            "rules_enabled": True,
            "rules_extraction_enabled": False,
            "facts_enabled": True,
            "facts_extraction_enabled": True,
        },
    }
    config["no_dream"] = True
    models = {
        "reader": {
            "provider": "openai-compatible", "model": "reader-pinned",
            "base_url": "https://reader.example/v1", "temperature": 0.0,
            "max_tokens": 1024, "extra_body": {},
        },
        "judge": {
            "provider": "openai-compatible", "model": "judge-pinned",
            "base_url": "https://judge.example/v1", "temperature": 0.0,
            "max_tokens": 10, "n": None, "extra_body": {},
            "protocol": "legacy-custom",
            "evaluator_commit": LME_EVALUATOR_COMMIT,
            "evaluator_sha256": LME_EVALUATOR_SHA256,
            "verdict_parser": "anchored-exclusive-yes-no-local-v1",
            "prompt_exact_official": False,
            "retry_policy": LME_LOCAL_RETRY_POLICY,
        },
        "memory_pipeline": {
            "provider": "openai-compatible", "model": "pipeline-pinned",
            "base_url": "https://pipeline.example/v1",
            "thinking_mode": "off", "effective_extra_body": {},
        },
        "embedding": config["embedding_runtime"],
    }

    def usage(calls, prompt, completion):
        return {
            "calls": calls, "calls_available": True,
            "request_attempts": calls, "request_attempts_available": True,
            "successful_responses": calls,
            "successful_responses_available": True,
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "token_usage_available": True,
            "latency_s": 1.0, "latency_available": True,
            "cost_usd": None, "cost_available": False,
        }

    embedding_usage = {
        "configured": False, "backend": "none", "quality": "none",
        "network_free": True, "model": None, "dimension": None,
        "identity_available": True,
        "calls": 0, "calls_available": True,
        "request_attempts": 0, "request_attempts_available": True,
        "successful_responses": 0, "successful_responses_available": True,
        "input_count": 0, "input_count_available": True,
        "input_characters": 0, "input_characters_available": True,
        "prompt_tokens": None, "total_tokens": None,
        "provider_token_usage_available": False,
        "latency_s": 0.0, "latency_available": True,
        "cost_usd": None, "cost_available": False,
    }
    manifest = build_manifest(
        benchmark="LongMemEval", code_sha256=content_hash("code"),
        data_sha256=content_hash("data"), config=config, models=models,
        seed=3, expected_ids=["q1", "q2"], protocol_split="full",
    )
    artifact = {
        "benchmark": "LongMemEval", "version": "strict-v1",
        "date": "2026-09-04T12:00:01+00:00", "manifest": manifest,
        "config": config, "models": models,
        "scores": {
            "OVERALL": {"accuracy": 50.0, "count": 2},
            "multi-session": {"accuracy": 50.0, "count": 2},
        },
        "execution": {
            "counts": {
                "expected": 2, "attempted": 2, "completed": 1,
                "unique_attempted": 2, "total_attempts": 2,
                "failed": 1, "missing": 0,
            },
            "segments": [{
                "segment_id": "process-a", "status": segment_status,
                "elapsed_s": 4.0, "attempted_attempts": 2,
                "model_identities": models,
                "reader_usage": usage(2, 8, 2),
                "judge_usage": usage(2, 18, 2),
                "retrieval_usage": usage(0, 0, 0),
                "memory_pipeline_usage": usage(3, 25, 5),
                "embedding_usage": embedding_usage,
                "indexing_runs": [],
            }],
        },
        "per_question": [
            {
                "question_id": "q1", "question_type": "multi-session",
                "question": "q", "answer": "a", "hypothesis": "a",
                "correct": True, "judge_raw": "yes",
                "judge_protocol": "legacy-custom", "judge_parse_valid": True,
                "context_sha": "a" * 64,
                "benchmark_failure": None, "retrieval_only": False,
                "oracle_ability": "MR", "detected_ability": "MR",
                "ability_used": "MR",
                "distill_fired": False, "distill_calls": 0,
            },
            {
                "question_id": "q2", "question_type": "multi-session",
                "correct": False, "retrieval_only": False,
                "benchmark_failure": "execution_failure: fixture",
                "oracle_ability": "MR", "detected_ability": None,
                "ability_used": None,
                "distill_fired": False, "distill_calls": 0,
            },
        ],
    }
    artifact["result_digest"] = content_hash(artifact["per_question"])
    path.write_text(json.dumps(artifact))
    return path


def test_strict_lme_archive_discovery_and_nested_metadata(tmp_db):
    mod, db, bench = tmp_db
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-deadbeef.json"
    )
    (bench / "longmemeval-v2-hymem.json").write_text(json.dumps({
        "archive": archive.name,
        "run_id": json.loads(archive.read_text())["manifest"]["run_id"],
    }))
    mod.cmd_ingest(None)
    con = sqlite3.connect(db)
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    rows = con.execute("SELECT * FROM runs").fetchall()
    assert len(rows) == 1
    row = dict(zip(cols, rows[0]))
    assert row["kind"] == "archive"
    assert row["source_date"] == "20260904T120000Z"
    assert row["run_date"] == "2026-09-04T12:00:01"
    assert row["scale"] == "S"
    assert row["answer_model"] == "reader-pinned"
    assert row["judge_model"] == "judge-pinned"
    assert row["count"] == 2
    assert row["answer_calls"] == 2 and row["judge_calls"] == 2
    assert row["retrieval_calls"] == 0
    assert row["total_tokens"] == 60
    assert row["elapsed_s"] == pytest.approx(4.0)
    assert row["aggregation_nodes_enabled"] == 0
    assert row["episode_granularity_enabled"] == 1
    assert row["strict_validated"] == 1
    assert row["official_comparable"] == 0
    assert row["development_only"] == 1
    assert row["usage_exact"] == 1
    artifact = json.loads(archive.read_text())
    assert row["result_digest"] == artifact["result_digest"]
    assert row["artifact_digest"] == content_hash(artifact)
    assert row["protocol_split"] == "full"
    assert row["calibration_receipt_hash"] is None
    assert row["evaluator_commit"] == LME_EVALUATOR_COMMIT
    assert row["evaluator_sha256"] == LME_EVALUATOR_SHA256
    assert row["reader_provider"] == "openai-compatible"
    assert row["reader_base_url"] == "https://reader.example/v1"
    assert row["judge_provider"] == "openai-compatible"
    assert row["judge_base_url"] == "https://judge.example/v1"
    assert row["pipeline_provider"] == "openai-compatible"
    assert row["pipeline_model"] == "pipeline-pinned"
    assert row["pipeline_base_url"] == "https://pipeline.example/v1"
    assert row["embedding_backend"] == "none"
    assert row["embedding_model"] is None
    assert row["embedding_base_url"] is None
    assert row["embedding_quality"] == "none"
    assert row["embedding_network_free"] == 1
    extras = json.loads(row["extras"])
    assert extras["strict_validation"]["usage_exact"] is True


def test_strict_lme_incomplete_execution_never_claims_exact_usage(tmp_db):
    mod, db, bench = tmp_db
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-running.json",
        segment_status="running",
    )
    mod.cmd_ingest([str(archive)])
    con = sqlite3.connect(db)
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    row = dict(zip(cols, con.execute("SELECT * FROM runs").fetchone()))
    assert row["count"] == 2
    assert row["answer_calls"] is None and row["judge_calls"] is None
    assert row["total_tokens"] is None and row["elapsed_s"] is None
    assert row["usage_exact"] == 0


def test_lme_registry_explicitly_dereferences_stable_pointer(tmp_db):
    mod, db, bench = tmp_db
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-pointer.json"
    )
    artifact = json.loads(archive.read_text())
    pointer = bench / "longmemeval-v2-hymem.json"
    pointer.write_text(json.dumps({
        "archive": archive.name, "run_id": artifact["manifest"]["run_id"],
        "artifact_digest": content_hash(artifact),
    }))
    mod.cmd_ingest([str(pointer)])
    con = sqlite3.connect(db)
    cols = [d[0] for d in con.execute("SELECT * FROM runs").description]
    row = dict(zip(cols, con.execute("SELECT * FROM runs").fetchone()))
    assert row["archive"] == archive.name
    assert row["artifact_digest"] == content_hash(artifact)
    assert row["overall"] == 50.0 and row["count"] == 2
    assert row["answer_model"] == "reader-pinned"
    renamed_pointer = bench / "mutable-alias.json"
    renamed_pointer.write_bytes(pointer.read_bytes())
    assert mod.ingest_file(mod.connect(), renamed_pointer) == "skipped"
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_legacy_two_field_pointer_is_safely_bound_by_target_run_id(tmp_db):
    mod, _db, bench = tmp_db
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-oldpointer.json"
    )
    artifact = json.loads(archive.read_text())
    pointer = bench / "legacy-latest.json"
    pointer.write_text(json.dumps({
        "archive": archive.name,
        "run_id": artifact["manifest"]["run_id"],
    }))
    con = mod.connect()
    assert mod.ingest_file(con, pointer) == "inserted"
    con.commit()
    row = con.execute("SELECT archive, artifact_digest, extras FROM runs").fetchone()
    assert row[0] == archive.name
    assert row[1] == content_hash(artifact)
    assert json.loads(row[2])["pointer_resolution"] == (
        "legacy-pointer-run-id-validated-digest-computed"
    )


def test_registry_dedupes_by_target_artifact_identity_and_rejects_name_collision(
    tmp_db,
):
    mod, _db, bench = tmp_db
    con = mod.connect()
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-digest.json"
    )
    assert mod.ingest_file(con, archive) == "inserted"
    con.commit()
    assert mod.ingest_file(con, archive) == "skipped"

    copied = bench / (
        "longmemeval-v2-hymem-20260904T120001Z-seed3-strict-copy.json"
    )
    copied.write_bytes(archive.read_bytes())
    assert mod.ingest_file(con, copied) == "skipped"
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1

    legacy = bench / "longmemeval-v2-hymem-20260805T054914Z-seed0.json"
    make_run(legacy)
    assert mod.ingest_file(con, legacy) == "inserted"
    con.commit()
    make_run(legacy, permissive_default=False)
    assert mod.ingest_file(con, legacy) == (
        "error: archive basename collision with different artifact digest"
    )


def test_same_basename_rejects_a_second_independently_valid_strict_result(tmp_db):
    mod, db, bench = tmp_db
    con = mod.connect()
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-collision.json"
    )
    assert mod.ingest_file(con, archive) == "inserted"
    con.commit()

    # Turn the one successful answer from correct to wrong and reseal every
    # affected derived field. This remains a valid strict artifact, not a
    # digest-invalid tamper; only its immutable basename collides.
    replacement = json.loads(archive.read_text())
    replacement["per_question"][0].update({"correct": False, "judge_raw": "no"})
    replacement["scores"] = {
        "OVERALL": {"accuracy": 0.0, "count": 2},
        "multi-session": {"accuracy": 0.0, "count": 2},
    }
    replacement["result_digest"] = content_hash(replacement["per_question"])
    archive.write_text(json.dumps(replacement))

    assert mod.validate_strict_artifact(replacement)["scores"]["OVERALL"][
        "accuracy"
    ] == 0.0
    assert mod.ingest_file(con, archive) == (
        "error: archive basename collision with different artifact digest"
    )
    stored = sqlite3.connect(db).execute(
        "SELECT overall, artifact_digest FROM runs"
    ).fetchone()
    assert stored[0] == 50.0
    assert stored[1] != content_hash(replacement)


def test_pointer_digest_mismatch_is_explicit_and_inserts_nothing(tmp_db):
    mod, db, bench = tmp_db
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-pointerbad.json"
    )
    artifact = json.loads(archive.read_text())
    pointer = bench / "longmemeval-v2-hymem.json"
    pointer.write_text(json.dumps({
        "archive": archive.name,
        "run_id": artifact["manifest"]["run_id"],
        "artifact_digest": content_hash("wrong artifact"),
    }))
    result = mod.ingest_file(mod.connect(), pointer)
    assert result == "error: artifact pointer digest mismatch"
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_pointer_with_uncommitted_extra_identity_fields_fails_closed(tmp_db):
    mod, db, bench = tmp_db
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-pointerextra.json"
    )
    artifact = json.loads(archive.read_text())
    pointer = bench / "longmemeval-v2-hymem.json"
    pointer.write_text(json.dumps({
        "archive": archive.name,
        "run_id": artifact["manifest"]["run_id"],
        "artifact_digest": content_hash(artifact),
        "overall": 100.0,
    }))
    assert mod.ingest_file(mod.connect(), pointer) == (
        "error: artifact pointer has unexpected fields"
    )
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_registry_digest_unique_partial_index_closes_concurrent_duplicate_race(
    tmp_db,
):
    mod, _db, _bench = tmp_db
    first = mod.connect()
    second = mod.connect()
    sql = first.execute(
        "SELECT sql FROM sqlite_master WHERE name='idx_runs_artifact_digest'"
    ).fetchone()[0]
    assert "UNIQUE" in sql.upper() and "WHERE artifact_digest IS NOT NULL" in sql
    digest = content_hash({"same": "artifact"})
    first.execute(
        "INSERT INTO runs (archive, kind, run_date, artifact_digest) "
        "VALUES (?, 'archive', ?, ?)",
        ("first.json", "2026-09-04T12:00:00", digest),
    )
    first.commit()
    with pytest.raises(sqlite3.IntegrityError):
        second.execute(
            "INSERT INTO runs (archive, kind, run_date, artifact_digest) "
            "VALUES (?, 'archive', ?, ?)",
            ("second.json", "2026-09-04T12:00:01", digest),
        )


def test_strict_score_tamper_fails_closed_without_registry_row(tmp_db):
    mod, db, bench = tmp_db
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-tamper.json"
    )
    artifact = json.loads(archive.read_text())
    artifact["scores"]["OVERALL"]["accuracy"] = 100.0
    archive.write_text(json.dumps(artifact))
    result = mod.ingest_file(mod.connect(), archive)
    assert result.startswith("error: strict LongMemEval validation failed")
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_strict_shape_cannot_downgrade_to_legacy_on_missing_manifest(tmp_db):
    mod, db, bench = tmp_db
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-shape.json"
    )
    artifact = json.loads(archive.read_text())
    artifact.pop("manifest")
    artifact["version"] = "legacy-looking"
    archive.write_text(json.dumps(artifact))
    result = mod.ingest_file(mod.connect(), archive)
    assert result.startswith("error: strict LongMemEval validation failed")
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_strict_retrieval_diagnostic_registers_with_null_scores(tmp_db):
    mod, db, bench = tmp_db
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-retrieval.json"
    )
    artifact = json.loads(archive.read_text())
    cfg = artifact["config"]
    cfg["scored_run"] = False
    cfg["retrieval_only"] = True
    artifact["scores"] = {}
    for index, row in enumerate(artifact["per_question"]):
        row["retrieval_only"] = True
        if not row.get("benchmark_failure"):
            row["correct"] = None
            row["context_sha"] = f"{index + 1:064x}"
    for key in ("reader_usage", "judge_usage"):
        usage = artifact["execution"]["segments"][0][key]
        usage.update({
            "calls": 0, "request_attempts": 0,
            "successful_responses": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0,
            "latency_s": 0.0,
        })
    old = artifact["manifest"]
    artifact["manifest"] = build_manifest(
        benchmark="LongMemEval", code_sha256=old["code_hash"],
        data_sha256=old["data_hash"], config=cfg, models=artifact["models"],
        seed=old["seed"], expected_ids=["q1", "q2"], protocol_split="full",
    )
    artifact["result_digest"] = content_hash(artifact["per_question"])
    archive.write_text(json.dumps(artifact))
    con = mod.connect()
    assert mod.ingest_file(con, archive) == "inserted"
    con.commit()
    columns = [item[0] for item in con.execute("SELECT * FROM runs").description]
    row = dict(zip(columns, con.execute("SELECT * FROM runs").fetchone()))
    assert row["strict_validated"] == 1
    assert row["scored_run"] == 0 and row["retrieval_only"] == 1
    assert row["overall"] is None and row["multi_session"] is None
    assert row["count"] == 2
    assert row["retrieval_calls"] == 0


def test_registry_stores_retrieval_distillation_usage_additively(tmp_db):
    mod, _db, bench = tmp_db
    archive = make_strict_run(
        bench / "longmemeval-v2-hymem-20260904T120000Z-seed3-strict-retrievalcost.json"
    )
    artifact = json.loads(archive.read_text())
    cfg = artifact["config"]
    cfg.update({
        "scored_run": False, "retrieval_only": True, "distill": True,
        "retrieval_usage_owner": "separate-retrieval-meter",
    })
    artifact["scores"] = {}
    for index, row in enumerate(artifact["per_question"]):
        row["retrieval_only"] = True
        row["distill_fired"] = True
        row["distill_calls"] = 4 if index == 0 else 3
        if not row.get("benchmark_failure"):
            row["correct"] = None
    segment = artifact["execution"]["segments"][0]
    for key in ("reader_usage", "judge_usage", "memory_pipeline_usage"):
        segment[key].update({
            "calls": 0, "request_attempts": 0, "successful_responses": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "latency_s": 0.0,
        })
    segment["retrieval_usage"].update({
        "calls": 7, "request_attempts": 7, "successful_responses": 7,
        "prompt_tokens": 77, "completion_tokens": 0, "total_tokens": 77,
        "latency_s": 1.0,
    })
    artifact["retrieval_cost"] = {
        "usage_owner": "separate-retrieval-meter",
        "llm_calls": 7, "answer_calls": 0, "judge_calls": 0,
        "distill_calls": 7,
    }
    artifact["result_digest"] = content_hash(artifact["per_question"])
    old = artifact["manifest"]
    artifact["manifest"] = build_manifest(
        benchmark="LongMemEval", code_sha256=old["code_hash"],
        data_sha256=old["data_hash"], config=cfg, models=artifact["models"],
        seed=old["seed"], expected_ids=["q1", "q2"], protocol_split="full",
    )
    segment["model_identities"] = artifact["models"]
    archive.write_text(json.dumps(artifact))
    con = mod.connect()
    assert mod.ingest_file(con, archive) == "inserted"
    con.commit()
    retrieval_calls, total_tokens = con.execute(
        "SELECT retrieval_calls, total_tokens FROM runs"
    ).fetchone()
    assert retrieval_calls == 7
    assert total_tokens == 77


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
    assert r["run_date"] == "2026-07-25T19:13:14"     # exec stamp, not source date
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
    assert rr["run_date"] == "2026-07-25T19:13:14"
    assert rr["source_date"] == "20260610T094858Z"
    assert rr["total_tokens"] is None
    assert rr["elapsed_s"] is None
    con.close()

    mod.cmd_backfill()  # idempotence: no row should change a second time
    con = sqlite3.connect(db)
    rr2 = dict(zip(cols, con.execute(
        "SELECT * FROM runs WHERE archive=?", (rj,)).fetchone()))
    assert rr2["run_date"] == "2026-07-25T19:13:14"
    assert rr2["elapsed_s"] is None
    con.close()


def test_backfill_unreachable_row_is_not_a_silent_skip(tmp_db):
    # §6.5: the guarantee all three registries share -- a row whose artifact
    # BENCH_DIR cannot supply was NOT migrated, so it is counted as
    # unreachable (main() exits nonzero on it) rather than folded into the
    # benign-looking unreadable/recompute-failed bucket.  LME is the largest
    # registry; a half-run migration reading as success costs most here.
    mod, db, bench = tmp_db
    con = mod.connect()
    con.execute("INSERT INTO runs (archive, kind, run_date) VALUES (?,?,?)",
                ("longmemeval-v2-hymem-20260610T094858Z-seed0.json",
                 "archive", "2026-06-10T09:48:58"))
    con.commit()
    con.close()
    # The artifact was never written to bench/ -> unreachable, count of 1.
    assert mod.cmd_backfill() == 1


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
