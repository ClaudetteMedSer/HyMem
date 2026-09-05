"""Adversarial registry checks for strict and legacy BEAM artifacts."""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks import beam_registry, run_registry
from benchmarks.strictness import build_manifest, content_hash


def _usage(calls: int, prompt: int, completion: int, *, latency: float = 1.0):
    return {
        "calls": calls,
        "calls_available": True,
        "request_attempts": calls,
        "request_attempts_available": True,
        "successful_responses": calls,
        "successful_responses_available": True,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "latency_s": latency,
        "cost_usd": None,
        "token_usage_available": True,
        "latency_available": True,
        "cost_available": False,
    }


def _local_embedding_usage(*, instances: int = 2):
    return {
        "configured": True,
        "backend": "local_feature_hash",
        "quality": "lexical",
        "network_free": True,
        "model": "feature-hash-v1",
        "dimension": 384,
        "identity_consistent": True,
        "instances": instances,
        "calls": 6,
        "calls_available": True,
        "request_attempts": 0,
        "request_attempts_available": True,
        "successful_responses": 0,
        "successful_responses_available": True,
        "input_count": 30,
        "input_count_available": True,
        "input_characters": 600,
        "input_characters_available": True,
        "prompt_tokens": None,
        "total_tokens": None,
        "provider_token_usage_available": False,
        "latency_s": 0.5,
        "latency_available": True,
        "cost_usd": None,
        "cost_available": False,
    }


def _unavailable_usage():
    return {
        "calls": None, "calls_available": False,
        "request_attempts": None, "request_attempts_available": False,
        "successful_responses": None,
        "successful_responses_available": False,
        "prompt_tokens": None, "completion_tokens": None,
        "total_tokens": None,
        "latency_s": None, "latency_available": False,
        "cost_usd": None, "cost_available": False,
        "token_usage_available": False,
    }


def _unavailable_embedding_usage():
    return {
        "configured": True, "backend": "unavailable", "quality": "none",
        "network_free": None, "model": None, "dimension": None,
        "identity_available": False,
        "calls": None, "calls_available": False,
        "request_attempts": None, "request_attempts_available": False,
        "successful_responses": None,
        "successful_responses_available": False,
        "input_count": None, "input_count_available": False,
        "input_characters": None, "input_characters_available": False,
        "prompt_tokens": None, "total_tokens": None,
        "provider_token_usage_available": False,
        "latency_s": None, "latency_available": False,
        "cost_usd": None, "cost_available": False,
    }


def _strict_artifact() -> dict:
    embedding = {
        "configured": True,
        "backend": "local-hash",
        "model": "feature-hash-v1",
        "base_url": "local://feature-hash",
        "dimension": 384,
        "quality": "lexical-feature-hash",
        "network_free": True,
        "fallback_policy": "none",
        "fallback_reason": None,
    }
    config = {
        "scales": ["100K", "500K"],
        "sample": 1,
        "sample_strategy": "seeded-label-blind-hash-v1",
        "subset_run": True,
        "top_k": 7,
        "max_input_tokens": 16000,
        "indexing_max_cycles": 100,
        "indexing_timeout_s": 3600.0,
        "indexing_require_healthy": True,
        "judge_protocol": "official",
        "official_judge_protocol_match": True,
        "official_protocol_aligned": True,
        "official_denominator_validated": False,
        "official_judge_prompt_hash": beam_registry.BEAM_OFFICIAL_JUDGE_PROMPT_HASH,
        "official_judge_upstream_commit": beam_registry.BEAM_UPSTREAM_COMMIT,
        "official_judge_evaluator_url": beam_registry.BEAM_OFFICIAL_EVALUATOR_URL,
        "official_judge_prompt_url": beam_registry.BEAM_OFFICIAL_PROMPT_URL,
        "oracle_ability": False,
        "judge_gold": True,
        "prereg": {
            "path": "benchmarks/beam-prereg.md",
            "commit": "b" * 40,
            "blob": "c" * 40,
            "committed_at": "2026-09-04T10:00:00+00:00",
            "code_commit": "d" * 40,
        },
        "dataset_revisions": {
            beam_registry.BEAM_REPO: "a" * 40,
        },
        "dataset_revision_provenance_complete": True,
        "embedding": embedding,
        "facts": True,
        "facts_extraction": False,
        "effective_hymem_config": {
            "facts_enabled": True,
            "facts_extraction_enabled": False,
            "graph_multihop_enabled": False,
            "episode_granularity_enabled": False,
            "aggregation_nodes_enabled": False,
            "value_supersession_enabled": True,
        },
        "answer_extra_body": {},
        "judge_extra_body": {},
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
            "model": "gpt-4.1-mini",
            "provider": "openai",
            "base_url": beam_registry.OFFICIAL_JUDGE_BASE_URL,
            "temperature": 0.0,
            "max_tokens": None,
            "extra_body": {},
            "protocol": "official",
            "upstream_commit": beam_registry.BEAM_UPSTREAM_COMMIT,
            "prompt_hash": beam_registry.BEAM_OFFICIAL_JUDGE_PROMPT_HASH,
        },
        "memory_pipeline": {
            "provider": "openai-compatible", "model": "pipeline-pinned",
            "base_url": "https://api.deepseek.com",
            "thinking_mode": "off",
            "effective_extra_body": {},
        },
        "embedding": embedding,
    }
    rows: list[dict] = []
    for scale in config["scales"]:
        for ability in beam_registry.BEAM_ABILITIES:
            for ordinal in range(2):
                failed = scale == "100K" and ability == "CR" and ordinal == 0
                if scale == "100K":
                    score = 1.0 if ability == "ABS" else 0.5
                else:
                    score = 1.0 if ability == "CR" else 0.0
                if failed:
                    score = 0.0
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
                    "finish_reason": "stop",
                    "parse": "ok",
                    "score": criterion_score,
                    "reason": "audited reason",
                } for criterion_index, (rubric_item, criterion_score) in enumerate(
                    zip(rubric, criterion_scores)
                )]
                rows.append({
                    "question_id": (
                        f"beam:{scale}:conversation-{scale}:{ability}:{ordinal}"
                    ),
                    "scale": scale,
                    "conv_id": f"conversation-{scale}",
                    "ability": ability,
                    "oracle_ability": ability,
                    "detected_ability": None,
                    "ability_used": None,
                    "question": f"Question for {ability}?",
                    "answer": "A benchmark answer",
                    "rubric": rubric,
                    "score": score,
                    "llm_judge_score": score,
                    # Deliberately not used for registry scoring: 0.5 is valid
                    # official rubric credit even though this bool is false.
                    "correct": score == 1.0 and not failed,
                    "result_valid": not failed,
                    "judge_protocol": "official",
                    "judge_parse": "not_called" if failed else "ok",
                    "scores": [] if failed else criterion_scores,
                    "judge_criterion_results": [] if failed else criteria,
                    "benchmark_failure": (
                        "conversation_failure: fixture" if failed else None
                    ),
                })
    ids = [row["question_id"] for row in rows]
    manifest = build_manifest(
        benchmark="BEAM",
        code_sha256="sha256:" + "c" * 64,
        data_sha256="sha256:" + "d" * 64,
        config=config,
        models=models,
        seed=17,
        expected_ids=ids,
        protocol_split="full",
    )
    summary: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for scale in config["scales"]:
        scale_rows = [row for row in rows if row["scale"] == scale]
        summary[scale] = {}
        counts[scale] = {}
        for ability in beam_registry.BEAM_ABILITIES:
            ability_rows = [row for row in scale_rows if row["ability"] == ability]
            summary[scale][ability] = sum(row["score"] for row in ability_rows) / len(
                ability_rows
            )
            counts[scale][ability] = len(ability_rows)
        summary[scale]["OVERALL"] = sum(row["score"] for row in scale_rows) / len(
            scale_rows
        )
        counts[scale]["OVERALL"] = len(scale_rows)
    return {
        "benchmark": "BEAM",
        "version": "strict-v1",
        "date": "2026-09-04T12:00:01+00:00",
        "manifest": manifest,
        "config": manifest["config"],
        "models": manifest["models"],
        "summary": summary,
        "summary_counts": counts,
        "execution": {
            "counts": {
                "expected": 40,
                "attempted": 40,
                "unique_attempted": 40,
                "total_attempts": 40,
                "completed": 39,
                "failed": 1,
                "missing": 0,
            },
            "segments": [{
                "segment_id": "process-a",
                "status": "complete",
                "attempted_attempts": 40,
                "elapsed_s": 4.0,
                "reader_usage": _usage(39, 40, 60),
                "judge_usage": _usage(78, 120, 80),
                "memory_pipeline_usage": _usage(4, 20, 30),
                "embedding_usage": _local_embedding_usage(),
            }],
        },
        "per_question": rows,
    }


def _rehash_manifest(data: dict) -> None:
    manifest = data["manifest"]
    manifest["run_id"] = content_hash({
        key: value for key, value in manifest.items() if key != "run_id"
    })


def _rebind_manifest(data: dict) -> None:
    """Rebind coherent top-level config/models before semantic tamper probes."""

    data["manifest"]["config"] = deepcopy(data["config"])
    data["manifest"]["models"] = deepcopy(data["models"])
    data["manifest"]["config_hash"] = content_hash(data["config"])
    data["manifest"]["model_hash"] = content_hash(data["models"])
    _rehash_manifest(data)


def _official_full_artifact() -> dict:
    """Expand the subset fixture to the exact official 100K denominator."""

    data = _strict_artifact()
    templates = [
        row for row in data["per_question"] if row["scale"] == "100K"
    ]
    rows = []
    for conversation_index in range(20):
        for template in templates:
            row = deepcopy(template)
            row["conv_id"] = f"official-conversation-{conversation_index:02d}"
            row["question_id"] = (
                f"beam:100K:{row['conv_id']}:"
                f"{row['ability']}:{template['question_id'].rsplit(':', 1)[-1]}"
            )
            rows.append(row)
    data["config"].update({
        "scales": ["100K"],
        "sample": None,
        "sample_strategy": "all",
        "subset_run": False,
        "official_denominator_validated": True,
        "exploratory_non_comparable": False,
    })
    data["per_question"] = rows
    data["manifest"] = build_manifest(
        benchmark="BEAM",
        code_sha256="sha256:" + "c" * 64,
        data_sha256="sha256:" + "d" * 64,
        config=data["config"],
        models=data["models"],
        seed=17,
        expected_ids=[row["question_id"] for row in rows],
        protocol_split="full",
    )
    data["config"] = data["manifest"]["config"]
    data["models"] = data["manifest"]["models"]
    failed = sum(not row["result_valid"] for row in rows)
    data["execution"]["counts"] = {
        "expected": 400,
        "attempted": 400,
        "unique_attempted": 400,
        "total_attempts": 400,
        "completed": 400 - failed,
        "failed": failed,
        "missing": 0,
    }
    data["execution"]["segments"][0]["attempted_attempts"] = 400
    data["execution"]["segments"][0]["reader_usage"] = _usage(380, 400, 600)
    data["execution"]["segments"][0]["judge_usage"] = _usage(760, 1200, 800)
    data["execution"]["segments"][0]["memory_pipeline_usage"] = _usage(
        80, 200, 300,
    )
    data["execution"]["segments"][0]["embedding_usage"] = (
        _local_embedding_usage(instances=20)
    )
    data.pop("summary", None)
    data.pop("summary_counts", None)
    return data


def test_strict_registry_recomputes_continuous_scores_and_protocol_posture(tmp_path):
    data = _strict_artifact()
    path = tmp_path / "results_20260904T120001Z-123456-strict-deadbeef.json"
    row = beam_registry._beam_row(data, path)

    # Continuous official rubric mean: 0.5 receives half-credit.  A binary
    # `correct` rate would produce a different and invalid registry score.
    assert row["overall"] == pytest.approx(31.25)
    assert row["ability_abs"] == pytest.approx(50.0)
    assert row["ability_cr"] == pytest.approx(62.5)
    assert row["ability_eo"] == pytest.approx(25.0)
    assert row["count"] == 40
    assert row["source_date"] == "20260904T120001Z"
    assert row["run_id"] == data["manifest"]["run_id"]
    assert row["development_only"] == 1
    assert row["exploratory_non_comparable"] == 1
    assert row["label_free_answer_path"] == 1
    assert row["judge_protocol"] == "official"
    assert row["official_judge_protocol_match"] == 1
    assert row["dataset_revisions_complete"] == 1
    assert row["facts"] == 1
    assert row["facts_extraction"] == 0
    assert row["graph_multihop"] == 0
    assert row["episode_granularity_enabled"] == 0
    assert row["aggregation_nodes_enabled"] == 0
    assert row["value_supersession_enabled"] == 1
    assert row["no_dream"] == 0
    assert row["distill"] == 0
    extras = json.loads(row["extras"])
    assert extras["summary_disclosure"] == {
        "source": "recomputed_from_durable_rows",
        "stored_summary_present": True,
        "stored_summary_validated": True,
    }
    assert extras["manifest"] == data["manifest"]


def test_strict_registry_cannot_downgrade_missing_envelope_to_legacy(tmp_path):
    for filename, removed in (
        (
            "results_20260904T120001Z-123456-strict-deadbeef.json",
            ("manifest", "version"),
        ),
        ("renamed-artifact.json", ("manifest", "version")),
        ("renamed-no-models.json", ("manifest", "version", "models")),
        ("renamed-no-execution.json", ("manifest", "version", "execution")),
    ):
        data = _strict_artifact()
        for field in removed:
            data.pop(field)
        with pytest.raises(ValueError, match="strict BEAM artifact lacks"):
            beam_registry._beam_row(data, tmp_path / filename)


def test_strict_registry_rejects_present_unknown_strict_version(tmp_path):
    data = _strict_artifact()
    data["version"] = "strict-v999"
    with pytest.raises(ValueError, match="version is unsupported"):
        beam_registry._beam_row(data, tmp_path / "renamed-artifact.json")

    data = _strict_artifact()
    data["version"] = "strict-v999"
    for field in ("manifest", "execution", "per_question", "models"):
        data.pop(field)
    with pytest.raises(ValueError, match="strict BEAM artifact lacks"):
        beam_registry._beam_row(data, tmp_path / "renamed-version-only.json")


@pytest.mark.parametrize(
    ("summary", "counts"),
    [({}, {}), ({}, None), (None, {})],
)
def test_strict_registry_rejects_present_empty_summary_evidence(
    tmp_path, summary, counts,
):
    data = _strict_artifact()
    if summary is None:
        data.pop("summary")
    else:
        data["summary"] = summary
    if counts is None:
        data.pop("summary_counts")
    else:
        data["summary_counts"] = counts
    with pytest.raises(ValueError, match="summary/counts are partial"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d["manifest"].__setitem__("seed", 99), "run_id"),
        (lambda d: d["per_question"].__setitem__(1, d["per_question"][0]), "duplicate"),
        (lambda d: d["per_question"].reverse(), "id order/hash"),
        (lambda d: d["execution"]["counts"].__setitem__("completed", 18), "counts"),
        (lambda d: d["per_question"][0].__setitem__("score", float("nan")), "score"),
        (lambda d: d["per_question"][0].__setitem__("result_valid", 1), "result_valid"),
        (lambda d: d["per_question"][0].__setitem__("correct", False), "correctness"),
        (lambda d: d["per_question"][0].__setitem__("scores", [0.0]), "rubric evidence"),
        (lambda d: d["execution"]["segments"][0].__setitem__("status", "banana"), "status"),
        (
            lambda d: d["execution"]["segments"][0]["embedding_usage"].__setitem__(
                "dimension", 768
            ),
            "embedding execution identity",
        ),
    ],
)
def test_strict_registry_rejects_identity_row_and_execution_tampering(
    tmp_path, mutation, message,
):
    data = _strict_artifact()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_coherently_rehashed_false_posture(tmp_path):
    data = _strict_artifact()
    data["manifest"]["development_only"] = False
    _rehash_manifest(data)
    with pytest.raises(ValueError, match="development-only posture"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    "prereg",
    [
        "claimed",
        {"path": "benchmarks/spec.md"},
        {
            "path": "../outside.md", "commit": "a" * 40, "blob": "b" * 40,
            "committed_at": "2026-09-04T10:00:00+00:00",
            "code_commit": "c" * 40,
        },
    ],
)
def test_strict_registry_rejects_fake_or_unsafe_prereg(tmp_path, prereg):
    data = _strict_artifact()
    data["config"]["prereg"] = prereg
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="pre-registration"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_boolean_summary_count(tmp_path):
    data = _strict_artifact()
    data["summary_counts"]["100K"]["ABS"] = True
    with pytest.raises(ValueError, match="coverage/counts"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_recomputes_official_judge_identity(tmp_path):
    data = _strict_artifact()
    data["models"]["judge"]["model"] = "not-official"
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="official judge disclosure"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "deepseek"),
        ("base_url", "https://example.test/v1"),
        ("temperature", 0.1),
        ("max_tokens", 10),
        ("extra_body", {"thinking": {"type": "disabled"}}),
        ("upstream_commit", "0" * 40),
        ("prompt_hash", "sha256:" + "0" * 64),
    ],
)
def test_strict_registry_recomputes_every_official_judge_pin(
    tmp_path, field, value,
):
    data = _strict_artifact()
    data["models"]["judge"][field] = value
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="official judge disclosure"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("reader", "model", "", "reader model"),
        ("reader", "base_url", "https://user:secret@example.test", "reader endpoint"),
        ("memory_pipeline", "provider", " ", "memory pipeline provider"),
        (
            "memory_pipeline", "base_url",
            "https://pipeline.example/v1?token=secret", "memory pipeline endpoint",
        ),
    ],
)
def test_strict_registry_rejects_malformed_or_unsafe_model_identity(
    tmp_path, target, field, value, message,
):
    data = _strict_artifact()
    data["models"][target][field] = value
    _rebind_manifest(data)
    with pytest.raises(ValueError, match=message):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_raw_criterion_json_contradiction(tmp_path):
    data = _strict_artifact()
    criterion = data["per_question"][0]["judge_criterion_results"][0]
    criterion["raw"] = json.dumps({"score": 0.0, "reason": "different reason"})
    with pytest.raises(ValueError, match="criterion evidence disagrees"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_transport_sentinel_in_valid_criterion(tmp_path):
    data = _strict_artifact()
    criterion = data["per_question"][0]["judge_criterion_results"][0]
    criterion["raw"] = (
        '[LLM_ERROR: transport] {"score": 1.0, "reason": "audited reason"}'
    )
    with pytest.raises(ValueError, match="criterion has transport error"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_boolean_seed_and_invalid_numeric_config(tmp_path):
    data = _strict_artifact()
    data["manifest"]["seed"] = True
    _rehash_manifest(data)
    with pytest.raises(ValueError, match="seed"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    for field, value in (
        ("top_k", -1),
        ("max_input_tokens", True),
        ("indexing_max_cycles", 0),
        ("indexing_timeout_s", 0.0),
    ):
        broken = _strict_artifact()
        broken["config"][field] = value
        _rebind_manifest(broken)
        with pytest.raises(ValueError, match=field):
            beam_registry._beam_row(broken, tmp_path / "results_strict.json")


@pytest.mark.parametrize("value", [False, None])
def test_strict_registry_requires_healthy_indexing_protocol(tmp_path, value):
    data = _strict_artifact()
    if value is None:
        data["config"].pop("indexing_require_healthy")
    else:
        data["config"]["indexing_require_healthy"] = value
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="indexing_require_healthy"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("official_judge_evaluator_url", "official evaluator URL"),
        ("official_judge_prompt_url", "official prompt URL"),
    ],
)
def test_strict_registry_binds_pinned_official_source_urls(tmp_path, field, message):
    data = _strict_artifact()
    data["config"][field] = "https://github.com/example/lookalike"
    _rebind_manifest(data)
    with pytest.raises(ValueError, match=message):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_requires_scored_beam_posture(tmp_path):
    data = _strict_artifact()
    data["config"]["scored_run"] = False
    data["manifest"]["scored_run"] = False
    data["manifest"]["development_only"] = True
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="must be scored runs"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize("role", ["answer", "judge"])
@pytest.mark.parametrize(
    "reserved", ["model", "messages", "temperature", "max_tokens"],
)
def test_strict_registry_rejects_reserved_request_extensions(
    tmp_path, role, reserved,
):
    data = _strict_artifact()
    data["config"][f"{role}_extra_body"] = {reserved: "override"}
    if role == "judge":
        data["models"]["judge"]["extra_body"] = {reserved: "override"}
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="reserved|official judge disclosure"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_binds_judge_extra_body_to_model_identity(tmp_path):
    data = _strict_artifact()
    data["config"]["judge_extra_body"] = {"provider_option": "audited"}
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="official judge disclosure|extra_body"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider", "other", "memory pipeline provider"),
        ("thinking_mode", None, "thinking mode"),
        (
            "effective_extra_body",
            {"thinking": {"type": "disabled"}},
            "effective extra_body",
        ),
    ],
)
def test_strict_registry_binds_memory_pipeline_runtime_identity(
    tmp_path, field, value, message,
):
    data = _strict_artifact()
    data["models"]["memory_pipeline"][field] = value
    _rebind_manifest(data)
    with pytest.raises(ValueError, match=message):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_requires_effective_hymem_levers(tmp_path):
    data = _strict_artifact()
    data["config"].pop("effective_hymem_config")
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="effective HyMem config is absent"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    for field in beam_registry.STRICT_HYMEM_BOOLEAN_FIELDS:
        broken = _strict_artifact()
        broken["config"]["effective_hymem_config"][field] = 1
        _rebind_manifest(broken)
        with pytest.raises(ValueError, match=field):
            beam_registry._beam_row(broken, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("requested", "effective"),
    [
        ("facts", "facts_enabled"),
        ("facts_extraction", "facts_extraction_enabled"),
        ("graph_multihop", "graph_multihop_enabled"),
    ],
)
def test_strict_registry_binds_requested_to_effective_hymem_levers(
    tmp_path, requested, effective,
):
    data = _strict_artifact()
    data["config"][requested] = not data["config"]["effective_hymem_config"][effective]
    _rebind_manifest(data)
    with pytest.raises(ValueError, match=f"requested {requested}"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize("fixed", ["no_dream", "distill"])
def test_strict_registry_rejects_conflicting_fixed_beam_protocol(tmp_path, fixed):
    data = _strict_artifact()
    data["config"][fixed] = True
    _rebind_manifest(data)
    with pytest.raises(ValueError, match=fixed):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_validates_split_limitation_and_calibration_shape(tmp_path):
    data = _strict_artifact()
    data["manifest"]["protocol_limitation"] = "clean official split"
    _rehash_manifest(data)
    with pytest.raises(ValueError, match="protocol limitation"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    data = _strict_artifact()
    data["manifest"]["calibration_receipt_hash"] = "sha256:" + "a" * 64
    _rehash_manifest(data)
    with pytest.raises(ValueError, match="cannot carry a calibration receipt"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    data = _strict_artifact()
    data["manifest"]["protocol_split"] = "holdout"
    data["manifest"]["protocol_limitation"] = (
        beam_registry.BEAM_PROTOCOL_LIMITATIONS["holdout"]
    )
    data["manifest"]["calibration_receipt_hash"] = "claimed"
    _rehash_manifest(data)
    with pytest.raises(ValueError, match="calibration receipt hash"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_shortened_subset_denominator(tmp_path):
    data = _strict_artifact()
    data["config"]["sample"] = 2
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="subset denominator"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dimension", True),
        ("model", ""),
        ("base_url", "local://other"),
        ("quality", "semantic"),
        ("network_free", False),
        ("fallback_policy", "silent"),
    ],
)
def test_strict_registry_rejects_malformed_embedding_schema(tmp_path, field, value):
    data = _strict_artifact()
    data["config"]["embedding"][field] = value
    data["models"]["embedding"][field] = value
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="embedding"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_boolean_observed_embedding_dimension(tmp_path):
    data = _strict_artifact()
    data["execution"]["segments"][0]["embedding_usage"]["dimension"] = True
    with pytest.raises(ValueError, match="embedding execution"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_binds_question_id_and_success_payload(tmp_path):
    data = _strict_artifact()
    data["per_question"][0]["question_id"] = "beam:100K:other:ABS:0"
    data["manifest"]["expected_ids_hash"] = content_hash([
        row["question_id"] for row in data["per_question"]
    ])
    _rehash_manifest(data)
    with pytest.raises(ValueError, match="bound to scale/conversation"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    for field, value in (
        ("question", " "),
        ("answer", None),
        ("answer", "   "),
        ("rubric", []),
    ):
        broken = _strict_artifact()
        broken["per_question"][0][field] = value
        with pytest.raises(ValueError, match="successful row payload|rubric evidence"):
            beam_registry._beam_row(broken, tmp_path / "results_strict.json")

    broken = _strict_artifact()
    broken["per_question"][0]["oracle_ability"] = "CR"
    with pytest.raises(ValueError, match="oracle ability"):
        beam_registry._beam_row(broken, tmp_path / "results_strict.json")


def test_strict_registry_binds_label_free_route_evidence(tmp_path):
    data = _strict_artifact()
    valid = next(row for row in data["per_question"] if row["result_valid"])
    valid["detected_ability"] = None
    valid["ability_used"] = valid["ability"]
    with pytest.raises(ValueError, match="routing evidence"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    data = _strict_artifact()
    valid = next(row for row in data["per_question"] if row["result_valid"])
    valid["detected_ability"] = "CR"
    valid["ability_used"] = "CR"
    with pytest.raises(ValueError, match="detected ability"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    for missing in ("detected_ability", "ability_used"):
        data = _strict_artifact()
        valid = next(row for row in data["per_question"] if row["result_valid"])
        valid.pop(missing)
        with pytest.raises(ValueError, match="lacks routing evidence"):
            beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_inconsistent_populated_failed_route(tmp_path):
    data = _strict_artifact()
    failed = next(row for row in data["per_question"] if not row["result_valid"])
    failed["detected_ability"] = "MR"
    failed["ability_used"] = "TR"
    with pytest.raises(ValueError, match="failed row routing evidence"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_unsupported_scale_without_summary(tmp_path):
    data = _strict_artifact()
    data.pop("summary")
    data.pop("summary_counts")
    data["config"]["scales"] = ["2M", "500K"]
    for row in data["per_question"]:
        if row["scale"] == "100K":
            row["scale"] = "2M"
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="unsupported scales"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("criterion_index", 1, "criterion evidence disagrees"),
        ("rubric_item", "other", "criterion evidence disagrees"),
        ("score", 0.5, "criterion evidence disagrees"),
        ("reason", "", "criterion evidence disagrees"),
        ("parse", "unreadable", "criterion evidence disagrees"),
    ],
)
def test_strict_registry_rejects_criterion_evidence_drift(
    tmp_path, field, bad, message,
):
    data = _strict_artifact()
    data["per_question"][0]["judge_criterion_results"][0][field] = bad
    with pytest.raises(ValueError, match=message):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_nonmean_and_nonternary_row_scores(tmp_path):
    data = _strict_artifact()
    row = next(item for item in data["per_question"] if item["score"] == 0.5)
    row["scores"] = [0.25, 0.75]
    with pytest.raises(ValueError, match="criterion score"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    data = _strict_artifact()
    row = next(item for item in data["per_question"] if item["score"] == 0.5)
    row["scores"] = [0.0, 0.0]
    row["judge_criterion_results"][1]["score"] = 0.0
    row["judge_criterion_results"][1]["raw"] = json.dumps({
        "score": 0.0, "reason": "audited reason",
    })
    with pytest.raises(ValueError, match="criterion mean"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_validates_segment_ids_states_attempts_and_running_identity(
    tmp_path,
):
    data = _strict_artifact()
    data["execution"]["segments"][0]["status"] = "running"
    # A prior interrupted segment is legitimate; exact usage simply becomes
    # unavailable in the registry.
    row = beam_registry._beam_row(data, tmp_path / "results_strict.json")
    assert row["total_tokens"] is None

    for mutate, message in (
        (
            lambda d: d["execution"]["segments"].append(
                deepcopy(d["execution"]["segments"][0])
            ),
            "duplicated",
        ),
        (
            lambda d: d["execution"]["segments"][0].__setitem__(
                "attempted_attempts", True
            ),
            "attempted_attempts",
        ),
        (
            lambda d: d["execution"]["segments"][0].__setitem__(
                "attempted_attempts", 39
            ),
            "segment attempts",
        ),
        (
            lambda d: d["execution"]["segments"][0]["embedding_usage"].__setitem__(
                "model", "drift-even-while-running"
            ),
            "embedding execution identity",
        ),
    ):
        broken = _strict_artifact()
        broken["execution"]["segments"][0]["status"] = "running"
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            beam_registry._beam_row(broken, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [("network_free", False), ("quality", "semantic")],
)
def test_strict_registry_binds_embedding_execution_posture(tmp_path, field, value):
    data = _strict_artifact()
    data["execution"]["segments"][0]["embedding_usage"][field] = value
    with pytest.raises(ValueError, match="embedding execution identity"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_requires_complete_embedding_identity_and_typed_usage(tmp_path):
    data = _strict_artifact()
    data["execution"]["segments"][0]["embedding_usage"][
        "identity_consistent"
    ] = False
    with pytest.raises(ValueError, match="embedding identity"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    data = _strict_artifact()
    usage = data["execution"]["segments"][0]["embedding_usage"]
    usage["input_count"] = True
    with pytest.raises(ValueError, match="embedding input_count"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    data = _strict_artifact()
    usage = data["execution"]["segments"][0]["embedding_usage"]
    usage["calls_available"] = False
    with pytest.raises(ValueError, match="claims an unavailable value"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_malformed_complete_usage_and_elapsed(tmp_path):
    data = _strict_artifact()
    data["execution"]["segments"][0]["elapsed_s"] = float("nan")
    with pytest.raises(ValueError, match="elapsed_s"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    data = _strict_artifact()
    data["execution"]["segments"][0]["reader_usage"]["calls"] = True
    with pytest.raises(ValueError, match="reader calls"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    data = _strict_artifact()
    usage = data["execution"]["segments"][0]["memory_pipeline_usage"]
    usage["latency_available"] = False
    with pytest.raises(ValueError, match="claims an unavailable value"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("usage_key", "field", "value", "message"),
    [
        ("reader_usage", "total_tokens", 999, "token totals"),
        ("judge_usage", "prompt_tokens", 120.5, "prompt_tokens"),
        ("memory_pipeline_usage", "total_tokens", 50.5, "total_tokens"),
    ],
)
def test_strict_registry_requires_integral_reconciled_provider_tokens(
    tmp_path, usage_key, field, value, message,
):
    data = _strict_artifact()
    data["execution"]["segments"][0][usage_key][field] = value
    with pytest.raises(ValueError, match=message):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prompt_tokens", 4.5, "prompt_tokens"),
        ("total_tokens", 5, "token totals"),
    ],
)
def test_strict_registry_requires_integral_reconciled_remote_embedding_tokens(
    tmp_path, field, value, message,
):
    data = _strict_artifact()
    remote = {
        "configured": True,
        "backend": "openai-compatible",
        "model": "text-embedding-3-small",
        "base_url": "https://api.openai.com/v1",
        "dimension": 1536,
        "quality": "semantic",
        "network_free": False,
        "fallback_policy": "fail-closed",
        "fallback_reason": None,
    }
    data["config"]["embedding"] = deepcopy(remote)
    data["models"]["embedding"] = deepcopy(remote)
    usage = data["execution"]["segments"][0]["embedding_usage"]
    usage.update({
        "configured": True,
        "backend": "openai_compatible",
        "quality": "semantic",
        "network_free": False,
        "model": beam_registry._manifested_embedding_execution_identity(remote)[1],
        "dimension": 1536,
        "identity_consistent": True,
        "instances": 2,
        "calls": 6,
        "calls_available": True,
        "request_attempts": 6,
        "request_attempts_available": True,
        "successful_responses": 6,
        "successful_responses_available": True,
        "input_count": 30,
        "input_count_available": True,
        "input_characters": 600,
        "input_characters_available": True,
        "prompt_tokens": 4,
        "total_tokens": 4,
        "provider_token_usage_available": True,
        "latency_s": 0.5,
        "latency_available": True,
        "cost_usd": None,
        "cost_available": False,
    })
    usage[field] = value
    _rebind_manifest(data)
    with pytest.raises(ValueError, match=message):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize(
    ("usage_key", "message"),
    [("reader_usage", "reader usage"), ("judge_usage", "judge usage")],
)
def test_strict_registry_usage_calls_cover_durable_scored_work(
    tmp_path, usage_key, message,
):
    data = _strict_artifact()
    usage = data["execution"]["segments"][0][usage_key]
    usage.update({
        "calls": 0,
        "request_attempts": 0,
        "successful_responses": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })
    with pytest.raises(ValueError, match=message):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_allows_unavailable_zero_attempt_recovery_segment(tmp_path):
    data = _strict_artifact()
    data["execution"]["segments"].append({
        "segment_id": "interrupted-before-provider",
        "status": "running",
        "attempted_attempts": 0,
        "elapsed_s": 0.1,
        "reader_usage": _unavailable_usage(),
        "judge_usage": _unavailable_usage(),
        "memory_pipeline_usage": _unavailable_usage(),
        "embedding_usage": _unavailable_embedding_usage(),
    })
    row = beam_registry._beam_row(data, tmp_path / "results_strict.json")
    assert row["answer_calls"] is None
    assert row["total_tokens"] is None


def test_strict_registry_validates_dataset_repo_coverage_and_hash_posture(tmp_path):
    data = _strict_artifact()
    data["config"]["dataset_revisions"] = {"lookalike/BEAM": "a" * 40}
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="repository coverage"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")

    for field, value, message in (
        ("code_hash", "not-a-hash", "code_hash"),
        ("data_hash", "sha256:short", "data_hash"),
        ("official_split", True, "official split/comparability"),
        ("official_comparable", True, "official split/comparability"),
    ):
        broken = _strict_artifact()
        broken["manifest"][field] = value
        _rehash_manifest(broken)
        with pytest.raises(ValueError, match=message):
            beam_registry._beam_row(broken, tmp_path / "results_strict.json")


def test_strict_registry_rejects_false_official_denominator_claim(tmp_path):
    data = _strict_artifact()
    data["config"]["official_denominator_validated"] = True
    _rebind_manifest(data)
    with pytest.raises(ValueError, match="official denominator claim"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_accepts_exact_official_denominator_and_distribution(
    tmp_path,
):
    data = _official_full_artifact()
    row = beam_registry._beam_row(data, tmp_path / "results_strict.json")
    assert row["count"] == 400
    assert row["overall"] == pytest.approx(52.5)
    assert row["exploratory_non_comparable"] == 0


def test_strict_registry_rejects_corrupt_full_split_ability_distribution(tmp_path):
    data = _strict_artifact()
    data.pop("summary")
    data.pop("summary_counts")
    data["per_question"][0]["ability"] = "EO"
    data["per_question"][0]["oracle_ability"] = "EO"
    with pytest.raises(ValueError, match="ability distribution"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


@pytest.mark.parametrize("remove_from", ["summary", "summary_counts"])
def test_strict_registry_rejects_partial_stored_summary(tmp_path, remove_from):
    data = _strict_artifact()
    data[remove_from]["500K"].pop("TR")
    with pytest.raises(ValueError, match="coverage|counts"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_rejects_stored_score_tamper(tmp_path):
    data = _strict_artifact()
    data["summary"]["100K"]["EO"] += 0.01
    with pytest.raises(ValueError, match="stored summary differs"):
        beam_registry._beam_row(data, tmp_path / "results_strict.json")


def test_strict_registry_recomputes_when_optional_summary_is_absent(tmp_path):
    data = _strict_artifact()
    data.pop("summary")
    data.pop("summary_counts")
    row = beam_registry._beam_row(data, tmp_path / "results_strict.json")
    assert row["overall"] == pytest.approx(31.25)
    disclosure = json.loads(row["extras"])["summary_disclosure"]
    assert disclosure["stored_summary_present"] is False
    assert disclosure["stored_summary_validated"] is False


def test_strict_registry_accepts_payload_free_checkpoint_recovery_envelope(tmp_path):
    data = _strict_artifact()
    for key in ("benchmark", "version", "date", "summary", "summary_counts"):
        data.pop(key, None)
    data["created_at"] = "2026-09-04T12:34:56.987654+00:00"
    row = beam_registry._beam_row(
        data, tmp_path / "results_20260904T123456Z-recovered.json"
    )
    assert row["overall"] == pytest.approx(31.25)
    assert row["run_date"] == "2026-09-04T12:34:56"
    assert json.loads(row["extras"])["summary_disclosure"][
        "stored_summary_present"
    ] is False


def test_legacy_scalar_scale_is_normalized_and_malformed_scale_rejected(tmp_path):
    data = {
        "benchmark": "BEAM",
        "config": {"scales": "100K"},
        "scores": {"ABS": 50.0, "OVERALL": 25.0},
    }
    row = beam_registry._beam_row(data, tmp_path / "beam-v1.json")
    assert row["scale"] == "100K"
    bad = deepcopy(data)
    bad["config"]["scales"] = ["100K", 500]
    with pytest.raises(ValueError, match="malformed"):
        beam_registry._beam_row(bad, tmp_path / "beam-v1.json")


def test_legacy_multiscale_without_counts_keeps_aggregate_null(tmp_path):
    data = {
        "metadata": {"scales": ["100K", "500K"]},
        "summary": {
            "100K": {"ABS": 1.0, "OVERALL": 0.5},
            "500K": {"ABS": 0.0, "OVERALL": 0.25},
        },
    }
    row = beam_registry._beam_row(data, tmp_path / "results_legacy.json")
    assert row["scale"] == "100K,500K"
    assert row["overall"] is None
    assert row["ability_abs"] is None
    disclosure = json.loads(row["extras"])["legacy_score_disclosure"]
    assert disclosure["aggregate_available"] is False
    assert disclosure["aggregate_reason"] == "counts_unavailable"
    assert disclosure["per_scale_summary"] == data["summary"]


def test_legacy_multiscale_uses_only_complete_integral_counts(tmp_path):
    data = {
        "metadata": {"scales": ["100K", "500K"]},
        "summary": {
            "100K": {"ABS": 1.0, "OVERALL": 0.5},
            "500K": {"ABS": 0.0, "OVERALL": 0.25},
        },
        "summary_counts": {
            "100K": {"ABS": 1, "OVERALL": 2},
            "500K": {"ABS": 3, "OVERALL": 4},
        },
    }
    row = beam_registry._beam_row(data, tmp_path / "results_legacy.json")
    assert row["ability_abs"] == pytest.approx(25.0)
    assert row["overall"] == pytest.approx(33.333)
    data["summary_counts"]["500K"]["ABS"] = True
    row = beam_registry._beam_row(data, tmp_path / "results_legacy.json")
    assert row["overall"] is None
    assert json.loads(row["extras"])["legacy_score_disclosure"][
        "aggregate_reason"
    ] == "malformed_counts"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, -1, 101])
def test_legacy_malformed_percent_is_null_with_disclosure(tmp_path, bad):
    data = {
        "config": {"scales": "100K"},
        "scores": {"OVERALL": bad},
    }
    row = beam_registry._beam_row(data, tmp_path / "beam-v1.json")
    assert row["overall"] is None
    assert json.loads(row["extras"])["legacy_score_disclosure"][
        "invalid_score_metrics"
    ] == ["OVERALL"]


def test_results_pointer_is_rejected_even_when_explicit(tmp_path):
    with pytest.raises(ValueError, match="mutable pointer"):
        beam_registry._beam_row(
            {"archive": "results_real.json", "run_id": "sha256:" + "a" * 64},
            tmp_path / "results_latest.json",
        )


def test_default_discovery_and_additive_existing_db_migration(tmp_path):
    artifact = _strict_artifact()
    archive = tmp_path / "results_20260904T120001Z-123456-strict-deadbeef.json"
    archive.write_text(json.dumps(artifact), encoding="utf-8")
    (tmp_path / "results_latest.json").write_text(
        json.dumps({"archive": archive.name, "run_id": artifact["manifest"]["run_id"]}),
        encoding="utf-8",
    )
    db = tmp_path / "beam.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "archive TEXT, run_date TEXT)"
    )
    con.commit()
    con.close()

    spec = dict(beam_registry.SPEC)
    spec["builder"] = beam_registry._beam_row
    run_registry.cmd_ingest(spec, bench_dir=tmp_path, db_path=db)
    con = sqlite3.connect(db)
    columns = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
    assert "run_id" in columns
    rows = con.execute(
        "SELECT archive, source_date, run_id FROM runs"
    ).fetchall()
    assert rows == [(
        archive.name,
        "20260904T120001Z",
        artifact["manifest"]["run_id"],
    )]
