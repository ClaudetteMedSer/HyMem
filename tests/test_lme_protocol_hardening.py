from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import lme_protocol as protocol
from benchmarks import lme_registry
from benchmarks import longmemeval_adapter as lme
from benchmarks.strictness import (
    BenchmarkIntegrityError,
    build_manifest,
    content_hash,
    embedding_usage_snapshot,
    usage_snapshot,
)


@pytest.mark.parametrize(
    "argv",
    [
        [sys.executable, "-m", "benchmarks.longmemeval_adapter", "--help"],
        [sys.executable, "benchmarks/longmemeval_adapter.py", "--help"],
        [sys.executable, "-m", "benchmarks.lme_registry", "--help"],
        [sys.executable, "benchmarks/lme_registry.py", "--help"],
    ],
)
def test_lme_package_and_direct_cli_help(argv):
    completed = subprocess.run(
        argv,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "LongMemEval" in completed.stdout


def _usage(*, calls: int = 0, attempts: int | None = None,
           total_tokens: int = 0):
    attempts = calls if attempts is None else attempts
    client = type("Meter", (), {
        "call_count": calls,
        "request_attempts": attempts,
        "successful_responses": calls,
        "prompt_tokens": total_tokens,
        "completion_tokens": 0,
        "total_tokens": total_tokens,
        "total_latency_s": 0.0,
        "cost_usd": 0.0,
        "token_usage_available": True,
    })()
    return usage_snapshot(client)


def _zero_usage():
    return _usage()


def _base_identity():
    embedding = {
        "configured": False,
        "backend": "none",
        "quality": "none",
        "network_free": True,
        "model": None,
        "base_url": None,
        "dimension": None,
        "fallback_policy": "none",
    }
    config = {
        "scales": "S",
        "sample": 1,
        "seed": 7,
        "top_k": 15,
        "workers": 1,
        "max_input_tokens": None,
        "max_input_bytes": 60000,
        "provider_context_tokens": 65536,
        "indexing_max_cycles": 4,
        "indexing_timeout_s": 900.0,
        "auto_ability": True,
        "no_dream": True,
        "embeddings": False,
        "graph_facts_first": False,
        "permissive_default": False,
        "distill": False,
        "distill_prompt_version": "v2",
        "retrieval_only": False,
        "label_free_answer_path": True,
        "scored_run": True,
        "exploratory_label_steering": False,
        "exploratory_non_comparable": True,
        "subset_run": True,
        "official_denominator_validated": False,
        "source_order_validated": False,
        "indexing_require_healthy": True,
        "historical_local_judge_prompts_exact_official": False,
        "official_judge_match": False,
        "judge_transport_retry_policy": protocol.LME_LOCAL_RETRY_POLICY,
        "official_transport_retry_policy": protocol.LME_UPSTREAM_RETRY_POLICY,
        "official_transport_exact": False,
        "retrieval_usage_owner": "none",
        "sample_strategy": "sha256-seed-source-index-preserve-order-v1",
        "source_ids_hash": content_hash(["qid"]),
        "source_qtype_counts": {"multi-session": 1},
        "dataset_revision": "fixture",
        "dataset_sha256": content_hash("fixture-data"),
        "dataset_expected_count": 1,
        "prereg": None,
        "judge_protocol": "legacy-custom",
        "answer_model": "reader",
        "answer_base_url": "https://reader.example/v1",
        "answer_extra_body_obj": {},
        "judge_model": "judge",
        "judge_base_url": "https://judge.example/v1",
        "judge_extra_body_obj": {},
        "hymem_model": "pipeline",
        "hymem_base_url": "https://pipeline.example/v1",
        "hymem_thinking": "off",
        "rerank_top_k": None,
        "graph_multihop_max_hops": None,
        "graph_multihop_decay": None,
        "graph_multihop_min_score": None,
        "aggregation_nodes": False,
        "aggregation_broad": False,
        "episode_granularity": False,
        "value_supersession": True,
        "graph_multihop": False,
        "rerank_model": None,
        "rerank_message_hits": None,
        "rules": None,
        "rules_extraction": None,
        "facts": None,
        "facts_extraction": None,
        "embedding_runtime": embedding,
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
        "effective_hymem_config": {
            "message_fts_top_k": 15,
            "fts_top_k": 10,
            "graph_top_k": 10,
            "aggregation_nodes_enabled": False,
            "aggregation_inject_abilities": ["TR"],
            "episode_granularity_enabled": False,
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
        "evaluator_commit": protocol.LME_EVALUATOR_COMMIT,
        "evaluator_sha256": protocol.LME_EVALUATOR_SHA256,
        "evaluator_url": protocol.LME_EVALUATOR_URL,
    }
    models = {
        "reader": {
            "provider": "openai-compatible",
            "model": "reader",
            "base_url": "https://reader.example/v1",
            "temperature": 0.0,
            "max_tokens": 1024,
            "extra_body": {},
        },
        "judge": {
            "provider": "openai-compatible",
            "model": "judge",
            "base_url": "https://judge.example/v1",
            "temperature": 0.0,
            "max_tokens": 10,
            "n": None,
            "extra_body": {},
            "protocol": "legacy-custom",
            "evaluator_commit": protocol.LME_EVALUATOR_COMMIT,
            "evaluator_sha256": protocol.LME_EVALUATOR_SHA256,
            "verdict_parser": "anchored-exclusive-yes-no-local-v1",
            "prompt_exact_official": False,
            "retry_policy": protocol.LME_LOCAL_RETRY_POLICY,
        },
        "memory_pipeline": {
            "provider": "openai-compatible",
            "model": "pipeline",
            "base_url": "https://pipeline.example/v1",
            "thinking_mode": "off",
            "effective_extra_body": {},
        },
        "embedding": embedding,
    }
    return config, models


def make_artifact(*, retrieval_only: bool = False):
    config, models = _base_identity()
    if retrieval_only:
        config.update({
            "retrieval_only": True,
            "scored_run": False,
            "exploratory_non_comparable": True,
        })
    manifest = build_manifest(
        benchmark="LongMemEval",
        code_sha256=content_hash("code"),
        data_sha256=config["dataset_sha256"],
        config=config,
        models=models,
        seed=7,
        expected_ids=["qid"],
        protocol_split="full",
    )
    if retrieval_only:
        row = {
            "question_id": "qid",
            "question_type": "multi-session",
            "correct": None,
            "benchmark_failure": None,
            "retrieval_only": True,
            "context_sha": "a" * 64,
            "oracle_ability": "MR",
            "detected_ability": None,
            "ability_used": None,
            "distill_fired": False,
            "distill_calls": 0,
        }
        scores = {}
    else:
        row = {
            "question_id": "qid",
            "question_type": "multi-session",
            "question": "question",
            "answer": "gold",
            "hypothesis": "gold",
            "correct": True,
            "judge_raw": "yes",
            "judge_protocol": "legacy-custom",
            "judge_parse_valid": True,
            "judge_error": False,
            "context_sha": "a" * 64,
            "benchmark_failure": None,
            "retrieval_only": False,
            "oracle_ability": "MR",
            "detected_ability": None,
            "ability_used": None,
            "distill_fired": False,
            "distill_calls": 0,
        }
        scores = {
            "multi-session": {"accuracy": 100.0, "count": 1},
            "OVERALL": {"accuracy": 100.0, "count": 1},
        }
    zero = _zero_usage()
    reader_usage = zero if retrieval_only else _usage(calls=1)
    judge_usage = zero if retrieval_only else _usage(calls=1)
    artifact = {
        "benchmark": "LongMemEval",
        "version": "strict-v1",
        "date": "2026-09-04T12:00:00+00:00",
        "manifest": manifest,
        "config": config,
        "models": models,
        "scores": scores,
        "execution": {
            "counts": {
                "expected": 1,
                "attempted": 1,
                "unique_attempted": 1,
                "total_attempts": 1,
                "completed": 1,
                "failed": 0,
                "missing": 0,
            },
            "segments": [{
                "segment_id": "segment-a",
                "status": "complete",
                "elapsed_s": 1.0,
                "attempted_attempts": 1,
                "model_identities": models,
                "reader_usage": reader_usage,
                "judge_usage": judge_usage,
                "retrieval_usage": zero,
                "memory_pipeline_usage": zero,
                "embedding_usage": embedding_usage_snapshot(
                    None, configured=False
                ),
                "indexing_runs": [],
            }],
        },
        "per_question": [row],
    }
    artifact["result_digest"] = content_hash(artifact["per_question"])
    artifact["abstention_diagnostics"] = (
        {} if retrieval_only else protocol._abstention_from_rows([row])
    )
    artifact["conditional_judged_only"] = {
        "accuracy": None if retrieval_only else 1.0,
        "count": 0 if retrieval_only else 1,
    }
    return artifact


def _refresh_result_digest(artifact):
    artifact["result_digest"] = content_hash(artifact["per_question"])


def _refresh_manifest(artifact):
    old = artifact["manifest"]
    artifact["manifest"] = build_manifest(
        benchmark="LongMemEval", code_sha256=old["code_hash"],
        data_sha256=old["data_hash"], config=artifact["config"],
        models=artifact["models"], seed=old["seed"],
        expected_ids=[row["question_id"] for row in artifact["per_question"]],
        protocol_split=old["protocol_split"],
    )
    for segment in artifact["execution"]["segments"]:
        segment["model_identities"] = artifact["models"]


def test_official_truth_pins_and_literal_parser():
    assert protocol.LME_EVALUATOR_COMMIT == "9e0b455f4ef0e2ab8f2e582289761153549043fc"
    assert protocol.LME_EVALUATOR_SHA256 == (
        "sha256:ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"
    )
    assert protocol.LME_S_DATASET_REVISION == "98d7416c24c778c2fee6e6f3006e7a073259d48f"
    assert protocol.LME_S_DATASET_SHA256 == (
        "sha256:d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
    )
    assert protocol.LME_S_EXPECTED_COUNT == 500
    assert protocol.LME_S_SOURCE_IDS_HASH == (
        "sha256:a4849b8afda6b6ed31ead4fc28d00784d2d5fef945be87642f5ce3ab710b21c4"
    )
    assert protocol.LME_S_QTYPE_COUNTS == {
        "single-session-user": 70,
        "single-session-assistant": 56,
        "multi-session": 133,
        "temporal-reasoning": 133,
        "knowledge-update": 78,
        "single-session-preference": 30,
    }
    assert protocol.LME_OFFICIAL_JUDGE_MODEL == "gpt-4o-2024-08-06"
    assert protocol.LME_OFFICIAL_JUDGE_BASE_URL == "https://api.openai.com/v1"
    assert protocol.LME_OFFICIAL_JUDGE_TEMPERATURE == 0.0
    assert protocol.LME_OFFICIAL_JUDGE_MAX_TOKENS == 10
    assert (
        protocol.LME_HISTORICAL_LOCAL_JUDGE_PROMPTS_EXACT_OFFICIAL is False
    )
    assert protocol.LME_LOCAL_PROMPTS_EXACT_OFFICIAL is False
    assert protocol.is_official_abstention_id("prefix_abs_middle")
    assert not protocol.is_official_abstention_id("ordinary")
    assert protocol.parse_official_verdict("yesterday") is True
    assert protocol.parse_official_verdict("NO") is False


def test_official_prompt_uses_qid_abstention_without_mutating_qtype():
    prompt = lme.get_official_judge_prompt(
        "multi-session", "qid_abs_marker", "q", "explanation", "unknown"
    )
    assert "unanswerable question" in prompt
    assert "Explanation: explanation" in prompt
    assert "Correct Answer" not in prompt
    assert "answer no. \n\nQuestion" in lme.get_official_judge_prompt(
        "multi-session", "qid", "q", "a", "r"
    )


@pytest.mark.parametrize(
    ("question_type", "expected"),
    [
        (question_type, template)
        for question_type in (
            "single-session-user", "single-session-assistant", "multi-session",
        )
        for template in [
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: Q\n\nCorrect Answer: A\n\nModel Response: H\n\nIs the model response correct? Answer yes or no only."
        ]
    ] + [
        (
            "temporal-reasoning",
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: Q\n\nCorrect Answer: A\n\nModel Response: H\n\nIs the model response correct? Answer yes or no only.",
        ),
        (
            "knowledge-update",
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: Q\n\nCorrect Answer: A\n\nModel Response: H\n\nIs the model response correct? Answer yes or no only.",
        ),
        (
            "single-session-preference",
            "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: Q\n\nRubric: A\n\nModel Response: H\n\nIs the model response correct? Answer yes or no only.",
        ),
    ],
)
def test_official_prompt_templates_are_byte_pinned(question_type, expected):
    assert lme.get_official_judge_prompt(
        question_type, "qid", "Q", "A", "H"
    ) == expected


def test_official_abstention_prompt_is_byte_pinned_by_substring_qid():
    expected = (
        "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: Q\n\nExplanation: A\n\nModel Response: H\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    )
    assert lme.get_official_judge_prompt(
        "multi-session", "before_abs_after", "Q", "A", "H"
    ) == expected


def test_official_protocol_rejects_nonexact_judge_identity_even_if_rehashed():
    artifact = make_artifact()
    artifact["config"]["judge_protocol"] = "official"
    artifact["config"]["official_judge_match"] = False
    artifact["models"]["judge"]["protocol"] = "official"
    artifact["models"]["judge"]["verdict_parser"] = (
        protocol.LME_OFFICIAL_VERDICT_PARSER
    )
    artifact["models"]["judge"]["prompt_exact_official"] = True
    artifact["per_question"][0]["judge_protocol"] = "official"
    _refresh_result_digest(artifact)
    artifact["execution"]["segments"][0]["model_identities"] = artifact["models"]
    old = artifact["manifest"]
    artifact["manifest"] = build_manifest(
        benchmark="LongMemEval", code_sha256=old["code_hash"],
        data_sha256=old["data_hash"], config=artifact["config"],
        models=artifact["models"], seed=old["seed"],
        expected_ids=["qid"], protocol_split="full",
    )
    with pytest.raises(BenchmarkIntegrityError, match="exact pinned judge identity"):
        protocol.validate_strict_artifact(artifact)


def test_real_source_schema_edge_cases_are_accepted_and_preserved():
    row = {
        "question_id": "qid",
        "question_type": "multi-session",
        "question": "question",
        "answer": 42,
        "question_date": "2025-01-03",
        "answer_session_ids": ["duplicate"],
        "haystack_session_ids": ["duplicate", "duplicate"],
        "haystack_dates": ["2025-01-01", "2025-01-02"],
        "haystack_sessions": [
            [{"role": "user", "content": ""}],
            [{"role": "assistant", "content": "answer", "has_answer": True}],
        ],
    }
    assert protocol.validate_lme_dataset([row], scale="S")[0] == row
    assert row["question_type"] == "multi-session"
    broken = deepcopy(row)
    broken["haystack_dates"] = []
    with pytest.raises(BenchmarkIntegrityError, match="lengths differ"):
        protocol.validate_lme_dataset([broken], scale="S")


@pytest.mark.parametrize("reserved", [
    "model", "messages", "temperature", "max_tokens", "n",
])
def test_extra_body_rejects_every_reserved_request_key(reserved):
    with pytest.raises(BenchmarkIntegrityError, match="core field"):
        protocol.normalize_extra_body({reserved: 1}, label="judge")
    assert protocol.normalize_extra_body(None, label="judge") == {}


def test_endpoint_credentials_never_cross_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    assert lme.resolve_endpoint_key(
        role="reader", base_url=lme.DEEPSEEK_BASE_URL,
        explicit_key=None, deepseek_key="deepseek-secret",
    ) == "deepseek-secret"
    with pytest.raises(BenchmarkIntegrityError, match="answer-api-key"):
        lme.resolve_endpoint_key(
            role="reader", base_url="https://reader.example/v1",
            explicit_key=None, deepseek_key="deepseek-secret",
        )
    assert lme.resolve_endpoint_key(
        role="judge", base_url=protocol.LME_OFFICIAL_JUDGE_BASE_URL,
        explicit_key=None, deepseek_key="deepseek-secret",
    ) == "openai-secret"


@pytest.mark.parametrize("endpoint", [
    "https://user:secret@reader.example/v1",
    "https://reader.example/v1?token=secret",
    "http://reader.example/v1",
    "https://reader.example:invalid/v1",
    "https://[broken/v1",
    "https://reader.example\\evil/v1",
    "https://reader.example/v1\nsecond",
])
def test_endpoint_validation_rejects_ambiguous_or_unsafe_urls(endpoint):
    with pytest.raises(BenchmarkIntegrityError, match="endpoint|plaintext"):
        protocol.validate_safe_endpoint(endpoint, label="reader")
    assert protocol.validate_safe_endpoint(
        "http://127.0.0.1:8766/v1/", label="local reader"
    ) == "http://127.0.0.1:8766/v1"


def test_context_policy_and_pipeline_provider_are_not_self_asserted():
    artifact = make_artifact()
    artifact["config"]["context_policy"]["source_boundaries"] = ["head"]
    old = artifact["manifest"]
    artifact["manifest"] = build_manifest(
        benchmark="LongMemEval", code_sha256=old["code_hash"],
        data_sha256=old["data_hash"], config=artifact["config"],
        models=artifact["models"], seed=old["seed"],
        expected_ids=["qid"], protocol_split="full",
    )
    with pytest.raises(BenchmarkIntegrityError, match="context packing policy"):
        protocol.validate_strict_artifact(artifact)

    artifact = make_artifact()
    artifact["models"]["memory_pipeline"]["provider"] = "deepseek"
    artifact["execution"]["segments"][0]["model_identities"] = artifact["models"]
    old = artifact["manifest"]
    artifact["manifest"] = build_manifest(
        benchmark="LongMemEval", code_sha256=old["code_hash"],
        data_sha256=old["data_hash"], config=artifact["config"],
        models=artifact["models"], seed=old["seed"],
        expected_ids=["qid"], protocol_split="full",
    )
    with pytest.raises(BenchmarkIntegrityError, match="pipeline identity"):
        protocol.validate_strict_artifact(artifact)


def test_model_bound_tokenizer_identity_is_bound_to_reader_model():
    artifact = make_artifact()
    policy = artifact["config"]["context_policy"]
    policy.update({
        "name": "model-bound-tokenizer-query-head-tail-v2",
        "budget_unit": "model_tokens",
        "max_input_tokens": 512,
        "max_input_bytes": None,
        "provider_context_window_tokens": 4096,
        "tokenizer": {
            "configured": True,
            "backend": "huggingface-tokenizers-json",
            "bound_model": "reader",
            "file_sha256": content_hash("local-tokenizer-json"),
            "local_only": True,
        },
        "tokenizer_failure_policy": "fail-closed",
    })
    artifact["config"].update({
        "max_input_tokens": 512,
        "max_input_bytes": None,
        "provider_context_tokens": 4096,
    })
    _refresh_manifest(artifact)
    protocol.validate_strict_artifact(artifact)

    artifact["config"]["context_policy"]["tokenizer"]["bound_model"] = "other"
    _refresh_manifest(artifact)
    with pytest.raises(BenchmarkIntegrityError, match="tokenizer identity"):
        protocol.validate_strict_artifact(artifact)


def test_token_aware_excerpt_keeps_answer_bearing_tail_and_is_deterministic():
    text = "HEAD " + ("middle " * 1000) + "ANSWER-BEARING-TAIL"
    memories = [{"type": "message_hit", "content": text}]
    kwargs = {
        "max_input_tokens": 260,
        "token_counter": lambda value: len(value.encode("utf-8")),
        "prompt_prefix": "system:s\nuser:CONTEXT:\n",
        "prompt_suffix": "\nQUESTION: where is the answer?\nANSWER:",
        "query": "answer tail",
    }
    first = lme._render_answer_context(
        memories, None, 0, None, None, None, **kwargs
    )
    second = lme._render_answer_context(
        memories, None, 0, None, None, None, **kwargs
    )
    assert first == second
    assert "ANSWER-BEARING-TAIL" in first
    assert kwargs["token_counter"](
        kwargs["prompt_prefix"] + first + kwargs["prompt_suffix"]
    ) <= kwargs["max_input_tokens"]


def test_mr_prompt_preserves_every_retrieved_tier_and_auxiliary_once():
    memories = [
        {"type": "message_hit", "content": "ASSISTANT_GOLD unique"},
        {"type": "message_hit", "content": "[user] USER_DISTRACTOR unique"},
        {"type": "fts", "content": "FTS_EVIDENCE unique"},
        {"type": "episode", "content": "EPISODE_EVIDENCE unique"},
        {"type": "procedure", "content": "PROCEDURE_EVIDENCE unique"},
        {"type": "graph_fact", "content": "GRAPH_FACT_EVIDENCE unique"},
    ]
    event = SimpleNamespace(
        date="2025-01-01", text="TEMPORAL_EVIDENCE unique", source="event"
    )
    messages = lme.build_answer_messages(
        memories, "combine everything", ability="MR", total_matches=6,
        temporal_events=[event], aggregation_nodes=["AGGREGATION_EVIDENCE unique"],
        narrative_facts=["NARRATIVE_EVIDENCE unique"],
        distilled=["DISTILLED_EVIDENCE unique"], max_input_bytes=20_000,
    )
    assert messages[0]["content"] == lme.ANSWERING_MR_PROMPT
    rendered = messages[1]["content"]
    for sentinel in (
        "ASSISTANT_GOLD", "USER_DISTRACTOR", "FTS_EVIDENCE",
        "EPISODE_EVIDENCE", "PROCEDURE_EVIDENCE", "GRAPH_FACT_EVIDENCE",
        "TEMPORAL_EVIDENCE", "AGGREGATION_EVIDENCE", "NARRATIVE_EVIDENCE",
        "DISTILLED_EVIDENCE",
    ):
        assert rendered.count(sentinel) == 1


def test_mr_exact_and_retrieval_count_sentinels_survive_when_they_fit():
    exact = lme._render_answer_context(
        [], "MR", 9, SimpleNamespace(count=4, counted="projects"),
        None, None, max_input_bytes=2_000,
    )
    assert "graph-native count: 4 distinct projects" in exact
    fallback = lme._render_answer_context(
        [], "MR", 9, None, None, None, max_input_bytes=2_000,
    )
    assert "HyMem counted 9 distinct user messages" in fallback


def test_packing_rejects_generated_stubs_and_keeps_later_answer_evidence():
    memories = [
        {"type": "episode", "content": "[MEM] " + ("verbose " * 500)},
        {"type": "episode", "content": "ANSWERBEARING zebra"},
    ]
    kwargs = dict(
        max_input_bytes=115, prompt_prefix="system:s\nuser:CONTEXT:\n",
        prompt_suffix="\nQUESTION: zebra?\nANSWER:", query="zebra",
    )
    first = lme._render_answer_context(
        memories, None, 0, None, None, None, **kwargs
    )
    second = lme._render_answer_context(
        memories, None, 0, None, None, None, **kwargs
    )
    assert first == second
    assert "ANSWERBEARING zebra" in first
    assert "\n[MEM] [\n" not in f"\n{first}\n"
    assert "\n[MEM] [ME\n" not in f"\n{first}\n"
    assert len((kwargs["prompt_prefix"] + first + kwargs["prompt_suffix"]).encode()) <= 115


def test_full_one_character_raw_value_is_preserved_exactly_when_it_fits():
    rendered = lme._render_answer_context(
        [{"type": "fts_hit", "content": "7"}],
        None, 0, None, None, None, max_input_bytes=100,
    )
    assert rendered == "[MEM] 7"


def test_tiny_budget_is_deterministic_and_never_emits_a_header_stub():
    prefix = "system:s\nuser:CONTEXT:\n"
    suffix = "\nQUESTION: q\nANSWER:"
    budget = len((prefix + suffix).encode("utf-8")) + 3
    kwargs = {
        "max_input_bytes": budget,
        "prompt_prefix": prefix,
        "prompt_suffix": suffix,
        "query": "q",
    }
    memories = [{"type": "episode", "content": "[MEM] " + "verbose " * 50}]
    first = lme._render_answer_context(
        memories, None, 0, None, None, None, **kwargs
    )
    second = lme._render_answer_context(
        memories, None, 0, None, None, None, **kwargs
    )
    assert first == second == ""
    assert first not in {"[", "[ME", "[MEM]"}


def test_auxiliary_crowding_cannot_consume_reserved_raw_evidence():
    first = lme._render_answer_context(
        [{"type": "message_hit", "content": "ANSWERBEARING compact raw"}],
        "MR", 3, None, None,
        ["summary " + "S" * 2_000],
        distilled=["distilled " + "D" * 2_000],
        narrative_facts=["narrative " + "N" * 2_000],
        max_input_bytes=180,
    )
    second = lme._render_answer_context(
        [{"type": "message_hit", "content": "ANSWERBEARING compact raw"}],
        "MR", 3, None, None,
        ["summary " + "S" * 2_000],
        distilled=["distilled " + "D" * 2_000],
        narrative_facts=["narrative " + "N" * 2_000],
        max_input_bytes=180,
    )
    assert first == second
    assert "ANSWERBEARING compact raw" in first
    assert not any(line in {"[", "[ME", "[MEM]"} for line in first.splitlines())


def test_exact_counter_and_truthful_byte_capacity_policies(tmp_path):
    counter = lambda value: len(value.encode("utf-8"))
    prefix = "system:reader\nuser:CONTEXT:\n"
    suffix = "\nQUESTION: q?\nANSWER:"
    rendered = lme._render_answer_context(
        [{"type": "episode", "content": "tail " * 500}],
        None, 0, None, None, None,
        max_input_tokens=180, max_input_bytes=600,
        token_counter=counter, prompt_prefix=prefix, prompt_suffix=suffix,
        query="tail",
    )
    assert counter(prefix + rendered + suffix) <= 180

    args = SimpleNamespace(
        answer_base_url=lme.DEEPSEEK_BASE_URL,
        answer_model=lme.ANSWER_MODEL,
        provider_context_tokens=None,
        max_input_bytes=lme.DEFAULT_MAX_INPUT_BYTES,
        max_input_tokens=None,
        tokenizer_json=None,
    )
    policy, configured_counter = lme.resolve_context_policy(args)
    assert configured_counter is None
    assert policy["budget_unit"] == "utf8_bytes"
    assert policy["max_input_tokens"] is None
    assert policy["max_input_bytes"] > 16_000
    large = lme.build_answer_messages(
        [{"type": "episode", "content": "A" * 30_000}], "Q?",
        max_input_bytes=policy["max_input_bytes"],
    )
    visible = "system:" + large[0]["content"] + "\nuser:" + large[1]["content"]
    assert len(visible.encode("utf-8")) > 16_000
    assert len(visible.encode("utf-8")) <= policy["max_input_bytes"]


def test_local_tokenizer_resolution_is_offline_hashed_and_model_bound(
    monkeypatch, tmp_path,
):
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text('{"fixture":"local-only"}')
    exact_counter = lambda value: len(value.split()) or (1 if value else 0)
    monkeypatch.setattr(
        lme, "_load_local_tokenizer_counter", lambda path: exact_counter
    )
    args = SimpleNamespace(
        answer_base_url="https://reader.example/v1", answer_model="reader-v1",
        provider_context_tokens=4096, max_input_bytes=60_000,
        max_input_tokens=512, tokenizer_json=str(tokenizer_file),
    )
    policy, counter = lme.resolve_context_policy(args)
    assert counter is exact_counter
    assert policy["budget_unit"] == "model_tokens"
    assert policy["max_input_bytes"] is None
    assert policy["tokenizer"] == {
        "configured": True,
        "backend": "huggingface-tokenizers-json",
        "bound_model": "reader-v1",
        "file_sha256": lme.file_hash(tokenizer_file),
        "local_only": True,
    }
    assert policy["tokenizer_failure_policy"] == "fail-closed"


def test_context_ceiling_reserves_output_and_chat_framing_margin(
    monkeypatch, tmp_path,
):
    args = SimpleNamespace(
        answer_base_url="https://reader.example/v1", answer_model="reader-v1",
        provider_context_tokens=(
            60_000 + lme.READER_OUTPUT_RESERVE_TOKENS
            + lme.READER_TRANSPORT_OVERHEAD_TOKENS - 1
        ),
        max_input_bytes=60_000, max_input_tokens=None, tokenizer_json=None,
    )
    with pytest.raises(BenchmarkIntegrityError, match="output/framing reserves"):
        lme.resolve_context_policy(args)

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"fixture":"offline"}')
    monkeypatch.setattr(
        lme, "_load_local_tokenizer_counter", lambda _path: len,
    )
    args = SimpleNamespace(
        answer_base_url="https://reader.example/v1", answer_model="reader-v1",
        provider_context_tokens=(
            512 + lme.READER_OUTPUT_RESERVE_TOKENS
            + lme.READER_TRANSPORT_OVERHEAD_TOKENS - 1
        ),
        max_input_bytes=60_000, max_input_tokens=512,
        tokenizer_json=str(tokenizer),
    )
    with pytest.raises(BenchmarkIntegrityError, match="output/framing reserves"):
        lme.resolve_context_policy(args)


def test_unknown_reader_endpoint_requires_explicit_context_ceiling():
    args = SimpleNamespace(
        answer_base_url="https://reader.example/v1", answer_model="reader-v1",
        provider_context_tokens=None, max_input_bytes=60_000,
        max_input_tokens=None, tokenizer_json=None,
    )
    with pytest.raises(BenchmarkIntegrityError, match="provider-context-tokens"):
        lme.resolve_context_policy(args)


def test_tokenizer_failure_fallback_is_deterministic_but_strict_mode_fails_closed():
    def broken(_value):
        raise RuntimeError("counter broke")

    kwargs = dict(
        max_input_tokens=100, max_input_bytes=220, token_counter=broken,
        prompt_prefix="system:s\nuser:CONTEXT:\n",
        prompt_suffix="\nQUESTION: q\nANSWER:", query="answer",
    )
    memories = [{"type": "episode", "content": "answer " * 100}]
    first = lme._render_answer_context(
        memories, None, 0, None, None, None, **kwargs
    )
    second = lme._render_answer_context(
        memories, None, 0, None, None, None, **kwargs
    )
    assert first == second
    assert len((kwargs["prompt_prefix"] + first + kwargs["prompt_suffix"]).encode()) <= 220
    with pytest.raises(BenchmarkIntegrityError, match="configured reader tokenizer failed"):
        lme._render_answer_context(
            memories, None, 0, None, None, None,
            **kwargs, fail_on_tokenizer_error=True,
        )


@pytest.mark.parametrize("policy", [
    {"max_input_bytes": 2_000},
    {
        "max_input_tokens": 2_000,
        "token_counter": lambda value: len(value.encode("utf-8")),
        "fail_on_tokenizer_error": True,
    },
])
def test_retrieval_and_full_context_hashes_match_under_same_policy(policy):
    class Reader:
        def __init__(self):
            self.messages = None

        def chat(self, messages, **_kwargs):
            self.messages = messages
            return "answer"

    memories = [{"type": "episode", "content": "evidence " * 500}]
    kwargs = {**policy, "ability": "MR", "total_matches": 1}
    reader = Reader()
    _answer, full_hash = lme.answer_question_raw(reader, memories, "Q?", **kwargs)
    retrieval_messages = lme.build_answer_messages(memories, "Q?", **kwargs)
    assert reader.messages == retrieval_messages
    assert full_hash == lme.context_sha(retrieval_messages)


def test_protocol_module_bytes_participate_in_manifest_identity(tmp_path):
    adapter = tmp_path / "benchmarks" / "longmemeval_adapter.py"
    strictness = tmp_path / "benchmarks" / "strictness.py"
    protocol_file = tmp_path / "benchmarks" / "lme_protocol.py"
    registry = tmp_path / "benchmarks" / "run_registry.py"
    hymem = tmp_path / "hymem"
    adapter.parent.mkdir()
    hymem.mkdir()
    for path, value in (
        (adapter, "adapter"), (strictness, "strictness"),
        (protocol_file, "protocol-v1"), (registry, "registry"),
        (hymem / "core.py", "core"),
    ):
        path.write_text(value)
    kwargs = {
        "adapter_path": adapter, "strictness_path": strictness,
        "protocol_path": protocol_file, "run_registry_path": registry,
        "hymem_path": hymem, "root": tmp_path,
    }
    before = lme.longmemeval_code_hash(**kwargs)
    manifest_config = {
        "label_free_answer_path": True, "scored_run": True,
        "exploratory_label_steering": False,
        "exploratory_non_comparable": True,
    }
    manifest_before = build_manifest(
        benchmark="LongMemEval", code_sha256=before,
        data_sha256=content_hash("data"), config=manifest_config, models={"m": 1},
        seed=0, expected_ids=["qid"], protocol_split="full",
    )
    protocol_file.write_text("protocol-v2")
    after = lme.longmemeval_code_hash(**kwargs)
    manifest_after = build_manifest(
        benchmark="LongMemEval", code_sha256=after,
        data_sha256=content_hash("data"), config=manifest_config, models={"m": 1},
        seed=0, expected_ids=["qid"], protocol_split="full",
    )
    assert before != after
    assert manifest_before["run_id"] != manifest_after["run_id"]

    # The adapter also imports the registry classifier and shared strictness
    # primitives; neither local dependency may drift outside run identity.
    protocol_file.write_text("protocol-v1")
    registry.write_text("registry-v2")
    assert lme.longmemeval_code_hash(**kwargs) != before
    registry.write_text("registry")
    strictness.write_text("strictness-v2")
    assert lme.longmemeval_code_hash(**kwargs) != before


def test_reader_error_prevents_judge_and_gold_diagnostics_are_nonblocking(monkeypatch):
    state = {"reader_called": False, "judge_calls": 0}

    class Reader:
        def chat(self, *_args, **_kwargs):
            state["reader_called"] = True
            return "[LLM_ERROR: outage]"

    class Judge:
        def chat(self, *_args, **_kwargs):
            state["judge_calls"] += 1
            raise AssertionError("judge must not be called")

    class Memory:
        last_indexing_summary = None

        def ingest_sessions(self, *_args, **_kwargs):
            return {
                "sessions": 1, "messages": 1, "chars": 3,
                "empty_messages_skipped": 0,
            }

        def search(self, *_args, **_kwargs):
            return ([{"type": "message_hit", "content": "memory"}],
                    1, None, [], [], [], {"message": [], "fts": []})

    def broken_diagnostic(_row):
        assert state["reader_called"] is True
        raise RuntimeError("optional gold diagnostic broke")

    monkeypatch.setattr(lme, "_extract_gold_turns", broken_diagnostic)
    row = lme.evaluate_question(
        Reader(), Judge(), Memory(), {
            "question_id": "qid_abs_marker",
            "question_type": "multi-session",
            "question": "question",
            "answer": "gold",
            "question_date": "2025-01-02",
            "haystack_sessions": [[{"role": "user", "content": "x"}]],
            "haystack_session_ids": ["s"],
            "haystack_dates": ["2025-01-01"],
            "answer_session_ids": ["s"],
        }, 5, no_dream=True,
    )
    assert row["benchmark_failure"] == "reader_transport_or_empty_response"
    assert row["correct"] is False
    assert row["recall_diagnostic_error"].startswith("RuntimeError")
    assert state["judge_calls"] == 0


def test_label_and_gold_fields_cannot_change_the_label_free_reader_context():
    namespaces = []

    class Memory:
        last_indexing_summary = None

        def ingest_sessions(self, *_args, **kwargs):
            namespaces.append(kwargs.get("namespace"))
            return {
                "sessions": 1, "messages": 1, "chars": 8,
                "empty_messages_skipped": 0,
            }

        def search(self, *_args, **_kwargs):
            return ([{"type": "message_hit", "content": "same memory"}],
                    1, None, [], [], [], {
                        "message": ["same memory"], "fts": [],
                    })

    base = {
        "question_id": "ordinary-qid",
        "question_type": "single-session-user",
        "question": "What project did I mention?",
        "answer": "secret gold A",
        "question_date": "2025-01-02",
        "haystack_sessions": [[{
            "role": "user", "content": "same memory", "has_answer": True,
        }]],
        "haystack_session_ids": ["s"],
        "haystack_dates": ["2025-01-01"],
        "answer_session_ids": ["s"],
    }
    relabeled = deepcopy(base)
    relabeled.update({
        "question_id": "ordinary_abs_marker",
        "question_type": "knowledge-update",
        "answer": "different secret gold B",
        "answer_session_ids": ["not-a-source-session"],
    })
    relabeled["haystack_sessions"][0][0]["has_answer"] = False

    rows = [
        lme.evaluate_question(
            lme.PoisonLLM("reader"), lme.PoisonLLM("judge"), Memory(), data,
            5, no_dream=True, retrieval_only=True, auto_ability=True,
        )
        for data in (base, relabeled)
    ]
    assert rows[0]["context_sha"] == rows[1]["context_sha"]
    assert namespaces == [base["question"], base["question"]]
    assert rows[0]["ability_used"] == rows[1]["ability_used"]


def test_strict_validator_rejects_judge_evidence_after_reader_failure():
    artifact = make_artifact()
    row = artifact["per_question"][0]
    row.update({
        "correct": False,
        "benchmark_failure": "reader_transport_or_empty_response",
        "hypothesis": "[LLM_ERROR: outage]",
        "judge_raw": "yes",
        "judge_error": False,
        "judge_parse_valid": None,
    })
    _refresh_result_digest(artifact)
    with pytest.raises(
        BenchmarkIntegrityError, match="reader failure reached or fabricated judge"
    ):
        protocol.validate_strict_artifact(artifact)


def test_official_blank_judge_reply_is_transport_failure():
    class BlankJudge:
        def chat(self, *_args, **_kwargs):
            return "  \n "

    verdict, raw = lme.judge_scored(
        BlankJudge(), "multi-session", "q", "a", "response",
        question_id="qid", protocol="official",
    )
    assert verdict is None
    assert raw == "  \n "


def test_dream_convergence_keeps_cleanup_baseexceptions_as_evidence():
    class Fork:
        def dream(self):
            return {"budget_exhausted": False, "skipped_locked": False}

        def dream_status(self):
            return {"pending_chunks": 0, "quarantined_chunks": 0}

        def close(self):
            raise KeyboardInterrupt("fork close")

    class Parent:
        def fork(self):
            return Fork()

        def invalidate_query_caches(self):
            raise SystemExit("cache close")

    adapter = object.__new__(lme.HyMemAdapter)
    adapter.hy = Parent()
    adapter.last_indexing_summary = None
    summary = adapter.dream_and_wait(timeout=5.0, max_cycles=2)
    assert summary["complete"] is True and summary["healthy"] is True
    assert len(summary["cleanup_errors"]) == 2
    assert summary["cleanup_errors"][0].startswith("dream_fork_close: KeyboardInterrupt")
    assert summary["cleanup_errors"][1].startswith("query_cache_invalidation: SystemExit")


def test_question_lifecycle_usage_and_cleanup_baseexceptions_never_mask_row(
    monkeypatch,
):
    class Adapter:
        pipeline_llm = object()
        embedding_client = None
        last_indexing_summary = None

        def open(self):
            return self

        def close(self):
            raise KeyboardInterrupt("close")

    expected = {
        "question_id": "qid",
        "question_type": "multi-session",
        "correct": True,
        "benchmark_failure": None,
    }
    monkeypatch.setattr(lme, "_adapter_for_args", lambda *_args: Adapter())
    monkeypatch.setattr(lme, "evaluate_question", lambda *_args, **_kwargs: dict(expected))
    monkeypatch.setattr(
        lme, "usage_snapshot",
        lambda _client: (_ for _ in ()).throw(SystemExit("usage")),
    )
    import shutil
    monkeypatch.setattr(
        shutil, "rmtree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit("rmtree")),
    )
    monkeypatch.setattr(
        lme.gc, "collect",
        lambda: (_ for _ in ()).throw(SystemExit("gc")),
    )
    args = SimpleNamespace(
        keep_db=False, embeddings=False, top_k=5, auto_ability=True,
        no_dream=True, graph_facts_first=False, permissive_default=False,
        distill=False,
        distill_prompt_version=lme.DEFAULT_DISTILL_PROMPT_VERSION,
        max_input_tokens=16000,
    )
    row = lme._evaluate_one_question(
        0, 1, {"question_id": "qid", "question_type": "multi-session"},
        args, object(), object(), "key",
    )
    assert {key: row[key] for key in expected} == expected
    assert [item.split(":", 1)[0] for item in row["lifecycle_errors"]] == [
        "pipeline_usage", "adapter_close", "temporary_store_cleanup", "gc",
    ]


def test_primary_baseexception_wins_over_meter_and_cleanup_baseexceptions(
    monkeypatch,
):
    class Adapter:
        pipeline_llm = object()
        embedding_client = None
        last_indexing_summary = None

        def open(self):
            return self

        def close(self):
            raise GeneratorExit("cleanup")

    monkeypatch.setattr(lme, "_adapter_for_args", lambda *_args: Adapter())
    monkeypatch.setattr(
        lme, "evaluate_question",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("primary benchmark interrupt")
        ),
    )
    monkeypatch.setattr(
        lme, "usage_snapshot",
        lambda _client: (_ for _ in ()).throw(SystemExit("meter cleanup")),
    )
    args = SimpleNamespace(
        keep_db=False, embeddings=False, top_k=5, auto_ability=True,
        no_dream=True, graph_facts_first=False, permissive_default=False,
        distill=False,
        distill_prompt_version=lme.DEFAULT_DISTILL_PROMPT_VERSION,
        max_input_tokens=16000,
    )
    with pytest.raises(KeyboardInterrupt, match="primary benchmark interrupt"):
        lme._evaluate_one_question(
            0, 1, {"question_id": "qid", "question_type": "multi-session"},
            args, object(), object(), "key",
        )


def test_duplicate_source_session_ids_receive_distinct_internal_keys():
    calls = []

    class Store:
        def log_messages(self, session_id, entries):
            calls.append((session_id, entries))

    adapter = object.__new__(lme.HyMemAdapter)
    adapter.hy = Store()
    result = adapter.ingest_sessions(
        [
            [{"role": "user", "content": "first"}],
            [{"role": "assistant", "content": "second"}],
        ],
        ["duplicate", "duplicate"],
        ["2025-01-01", "2025-01-02"],
        namespace="label-blind question namespace",
    )
    assert result["messages"] == 2
    assert calls[0][0] != calls[1][0]
    assert calls[0][0].endswith("_0_0")
    assert calls[1][0].endswith("_1_0")


def test_strict_validator_recomputes_scores_and_judge_evidence():
    artifact = make_artifact()
    assert protocol.validate_strict_artifact(artifact)["scores"]["OVERALL"]["accuracy"] == 100.0
    artifact["scores"]["OVERALL"]["accuracy"] = 0.0
    with pytest.raises(BenchmarkIntegrityError, match="stored score"):
        protocol.validate_strict_artifact(artifact)

    artifact = make_artifact()
    artifact["per_question"][0]["judge_raw"] = "no"
    _refresh_result_digest(artifact)
    with pytest.raises(BenchmarkIntegrityError, match="raw judge"):
        protocol.validate_strict_artifact(artifact)


def test_result_digest_rejects_post_hoc_hypothesis_edit():
    artifact = make_artifact()
    artifact["per_question"][0]["hypothesis"] = "post-hoc replacement"
    with pytest.raises(BenchmarkIntegrityError, match="result digest"):
        protocol.validate_strict_artifact(artifact)


def test_retrieval_distillation_usage_is_additive_and_exact():
    artifact = make_artifact(retrieval_only=True)
    config = artifact["config"]
    config["distill"] = True
    config["retrieval_usage_owner"] = "separate-retrieval-meter"
    row = artifact["per_question"][0]
    row["distill_fired"] = True
    row["distill_calls"] = 7
    segment = artifact["execution"]["segments"][0]
    segment["retrieval_usage"] = _usage(
        calls=7, attempts=7, total_tokens=77
    )
    artifact["retrieval_cost"] = {
        "usage_owner": "separate-retrieval-meter",
        "llm_calls": 7,
        "answer_calls": 0,
        "judge_calls": 0,
        "distill_calls": 7,
    }
    _refresh_result_digest(artifact)
    _refresh_manifest(artifact)
    validated = protocol.validate_strict_artifact(
        artifact, require_scored=False
    )
    assert validated["retrieval_calls"] == 7
    assert validated["answer_calls"] == 0
    assert validated["total_tokens"] == 77


def test_retrieval_usage_rejects_missing_availability_and_double_counting():
    artifact = make_artifact(retrieval_only=True)
    del artifact["execution"]["segments"][0]["retrieval_usage"][
        "calls_available"
    ]
    with pytest.raises(BenchmarkIntegrityError, match="calls_available"):
        protocol.validate_strict_artifact(artifact, require_scored=False)

    artifact = make_artifact()
    artifact["config"]["distill"] = True
    artifact["config"]["retrieval_usage_owner"] = "reader"
    artifact["execution"]["segments"][0]["retrieval_usage"] = _usage(
        calls=1, total_tokens=11
    )
    _refresh_manifest(artifact)
    with pytest.raises(BenchmarkIntegrityError, match="double-counted"):
        protocol.validate_strict_artifact(artifact)


def test_retrieval_usage_reconciles_across_resume_segments_once():
    artifact = make_artifact(retrieval_only=True)
    artifact["config"]["distill"] = True
    artifact["config"]["retrieval_usage_owner"] = "separate-retrieval-meter"
    artifact["per_question"][0].update({
        "distill_fired": True, "distill_calls": 7,
    })
    first = artifact["execution"]["segments"][0]
    first["retrieval_usage"] = _usage(
        calls=3, attempts=3, total_tokens=33
    )
    second = deepcopy(first)
    second.update({
        "segment_id": "segment-b", "attempted_attempts": 0,
        "reader_usage": _zero_usage(), "judge_usage": _zero_usage(),
        "retrieval_usage": _usage(calls=4, attempts=4, total_tokens=44),
        "memory_pipeline_usage": _zero_usage(),
    })
    artifact["execution"]["segments"].append(second)
    artifact["retrieval_cost"] = {
        "usage_owner": "separate-retrieval-meter",
        "llm_calls": 7,
        "answer_calls": 0,
        "judge_calls": 0,
        "distill_calls": 7,
    }
    _refresh_result_digest(artifact)
    _refresh_manifest(artifact)
    validated = protocol.validate_strict_artifact(
        artifact, require_scored=False
    )
    assert validated["retrieval_calls"] == 7
    assert validated["total_tokens"] == 77


def test_registry_compat_execution_counts_retrieval_usage_once():
    artifact = make_artifact(retrieval_only=True)
    segment = artifact["execution"]["segments"][0]
    artifact["config"]["distill"] = True
    artifact["config"]["retrieval_usage_owner"] = "separate-retrieval-meter"
    segment["retrieval_usage"] = _usage(calls=7, attempts=7, total_tokens=77)
    summary, disclosure = lme_registry._strict_execution(artifact)
    assert disclosure["total_tokens_available"] is True
    assert summary["answer_calls"] == 0
    assert summary["judge_calls"] == 0
    assert summary["retrieval_calls"] == 7
    assert summary["total_tokens"] == 77


def test_retrieval_attempts_cannot_understate_durable_distill_fanout():
    artifact = make_artifact(retrieval_only=True)
    artifact["config"]["distill"] = True
    artifact["config"]["retrieval_usage_owner"] = "separate-retrieval-meter"
    artifact["per_question"][0].update({
        "distill_fired": True, "distill_calls": 7,
    })
    artifact["execution"]["segments"][0]["retrieval_usage"] = _usage(
        calls=3, attempts=3, total_tokens=33
    )
    _refresh_result_digest(artifact)
    _refresh_manifest(artifact)
    with pytest.raises(BenchmarkIntegrityError, match="below durable distillation"):
        protocol.validate_strict_artifact(artifact, require_scored=False)


def test_strict_validator_recomputes_optional_router_and_recall_summaries():
    artifact = make_artifact()
    rows = artifact["per_question"]
    artifact["router_diagnostics"] = protocol._router_from_rows(rows)
    artifact["recall_diagnostics"] = protocol._recall_from_rows(rows)
    protocol.validate_strict_artifact(artifact)

    artifact["router_diagnostics"]["false_positives"] += 1
    with pytest.raises(BenchmarkIntegrityError, match="stored router summary"):
        protocol.validate_strict_artifact(artifact)

    artifact = make_artifact()
    artifact["diagnostic_errors"] = {"router": "RuntimeError: presentation"}
    artifact["router_diagnostics"] = {}
    # A disclosed optional-diagnostic failure does not discard valid rows; the
    # strict reader recomputes the diagnostic for downstream consumers.
    validated = protocol.validate_strict_artifact(artifact)
    assert validated["scores"]["OVERALL"]["accuracy"] == 100.0


def test_running_recovery_segment_is_valid_but_exact_usage_is_null():
    artifact = make_artifact()
    artifact["execution"]["segments"][0]["status"] = "running"
    validated = protocol.validate_strict_artifact(artifact)
    assert validated["answer_calls"] is None
    assert validated["judge_calls"] is None
    assert validated["total_tokens"] is None
    assert validated["elapsed_s"] is None


def test_failed_indexing_attempt_can_be_followed_by_healthy_resume():
    artifact = make_artifact()
    artifact["config"]["no_dream"] = False
    failed = {
        "question_id": "qid",
        "cycles": 1,
        "max_cycles": 1,
        "timeout_s": 1.0,
        "elapsed_s": 1.1,
        "complete": False,
        "healthy": False,
        "failure_reason": "max_cycles_exhausted",
        "reports": [{"budget_exhausted": True, "skipped_locked": False}],
        "final_status": {"pending_chunks": 1, "quarantined_chunks": 0},
        "quarantined": {},
    }
    healthy = {
        "question_id": "qid",
        "cycles": 1,
        "max_cycles": 2,
        "timeout_s": 2.0,
        "elapsed_s": 0.5,
        "complete": True,
        "healthy": True,
        "failure_reason": None,
        "reports": [{"budget_exhausted": False, "skipped_locked": False}],
        "final_status": {"pending_chunks": 0, "quarantined_chunks": 0},
        "quarantined": {},
    }
    artifact["per_question"][0]["indexing"] = dict(healthy)
    _refresh_result_digest(artifact)
    artifact["execution"]["segments"][0]["indexing_runs"] = [failed, healthy]
    original = artifact["manifest"]
    artifact["manifest"] = build_manifest(
        benchmark="LongMemEval", code_sha256=original["code_hash"],
        data_sha256=original["data_hash"], config=artifact["config"],
        models=artifact["models"], seed=original["seed"],
        expected_ids=["qid"], protocol_split="full",
    )
    assert protocol.validate_strict_artifact(artifact)["counts"]["completed"] == 1


def test_effective_lever_tamper_fails_even_with_rehashed_manifest():
    artifact = make_artifact()
    artifact["config"]["effective_hymem_config"]["rerank_top_k"] = 999
    original = artifact["manifest"]
    artifact["manifest"] = build_manifest(
        benchmark="LongMemEval", code_sha256=original["code_hash"],
        data_sha256=original["data_hash"], config=artifact["config"],
        models=artifact["models"], seed=original["seed"],
        expected_ids=["qid"], protocol_split="full",
    )
    with pytest.raises(BenchmarkIntegrityError, match="effective HyMem lever"):
        protocol.validate_strict_artifact(artifact)


def test_configured_embedding_identity_and_provider_usage_are_reconciled():
    artifact = make_artifact()
    identity = {
        "configured": True,
        "backend": "openai_compatible",
        "quality": "semantic",
        "network_free": False,
        "model": "openai-compatible:https://embed.example/v1::embed-v1",
        "request_model": "embed-v1",
        "base_url": "https://embed.example/v1",
        "dimension": 3,
        "fallback_policy": "fail-closed",
    }
    artifact["config"]["embeddings"] = True
    artifact["config"]["embedding_runtime"] = identity
    artifact["models"]["embedding"] = identity
    artifact["execution"]["segments"][0]["model_identities"] = artifact["models"]
    artifact["execution"]["segments"][0]["embedding_usage"] = {
        "configured": True,
        "backend": "openai_compatible",
        "quality": "semantic",
        "network_free": False,
        "model": identity["model"],
        "dimension": 3,
        "identity_consistent": True,
        "instances": 1,
        "calls": 2,
        "calls_available": True,
        "request_attempts": 2,
        "request_attempts_available": True,
        "successful_responses": 2,
        "successful_responses_available": True,
        "input_count": 4,
        "input_count_available": True,
        "input_characters": 100,
        "input_characters_available": True,
        "prompt_tokens": 12,
        "total_tokens": 12,
        "provider_token_usage_available": True,
        "latency_s": 0.2,
        "latency_available": True,
        "cost_usd": None,
        "cost_available": False,
    }
    old = artifact["manifest"]
    artifact["manifest"] = build_manifest(
        benchmark="LongMemEval", code_sha256=old["code_hash"],
        data_sha256=old["data_hash"], config=artifact["config"],
        models=artifact["models"], seed=old["seed"],
        expected_ids=["qid"], protocol_split="full",
    )
    validated = protocol.validate_strict_artifact(artifact)
    assert validated["total_tokens"] == 12

    unavailable = deepcopy(artifact)
    unavailable["execution"]["segments"][0]["embedding_usage"] = {
        "configured": True,
        "backend": "unavailable",
        "quality": "none",
        "network_free": None,
        "model": None,
        "dimension": None,
        "identity_available": False,
        "calls": None,
        "calls_available": False,
        "request_attempts": None,
        "request_attempts_available": False,
        "successful_responses": None,
        "successful_responses_available": False,
        "input_count": None,
        "input_count_available": False,
        "input_characters": None,
        "input_characters_available": False,
        "prompt_tokens": None,
        "total_tokens": None,
        "provider_token_usage_available": False,
        "latency_s": None,
        "latency_available": False,
        "cost_usd": None,
        "cost_available": False,
    }
    unavailable["execution"]["segments"][0]["instrumentation_errors"] = [
        "qid: embedding usage/identity unavailable"
    ]
    assert protocol.validate_strict_artifact(unavailable)["total_tokens"] is None
    unavailable["execution"]["segments"][0]["instrumentation_errors"] = []
    with pytest.raises(BenchmarkIntegrityError, match="embedding execution identity"):
        protocol.validate_strict_artifact(unavailable)

    artifact["execution"]["segments"][0]["embedding_usage"]["total_tokens"] = 13
    with pytest.raises(BenchmarkIntegrityError, match="token totals"):
        protocol.validate_strict_artifact(artifact)


def test_retrieval_only_is_valid_diagnostic_but_never_official_export(tmp_path):
    artifact = make_artifact(retrieval_only=True)
    validated = protocol.validate_strict_artifact(
        artifact, require_scored=False
    )
    assert validated["scores"] == {}
    assert validated["retrieval_calls"] == 0
    with pytest.raises(BenchmarkIntegrityError, match="retrieval-only"):
        protocol.export_official_predictions(artifact, tmp_path / "out.jsonl")


def _full_failure_artifact(monkeypatch):
    ids = [f"synthetic-{index:03d}" for index in range(500)]
    ids[17] = "synthetic_abs_017"
    monkeypatch.setattr(protocol, "LME_S_SOURCE_IDS_HASH", content_hash(ids))
    monkeypatch.setattr(protocol, "LME_S_QTYPE_COUNTS", {"multi-session": 500})
    config, models = _base_identity()
    config.update({
        "sample": 0,
        "subset_run": False,
        "sample_strategy": "all-source-order",
        "no_dream": False,
        "official_denominator_validated": True,
        "source_order_validated": True,
        "source_ids_hash": content_hash(ids),
        "source_qtype_counts": {"multi-session": 500},
        "dataset_revision": protocol.LME_S_DATASET_REVISION,
        "dataset_sha256": protocol.LME_S_DATASET_SHA256,
        "dataset_url": protocol.LME_S_DATASET_URL,
        "dataset_expected_count": 500,
    })
    manifest = build_manifest(
        benchmark="LongMemEval", code_sha256=content_hash("code"),
        data_sha256=protocol.LME_S_DATASET_SHA256, config=config,
        models=models, seed=7, expected_ids=ids, protocol_split="full",
    )
    rows = [{
        "question_id": qid,
        "question_type": "multi-session",
        "correct": False,
        "benchmark_failure": (
            "missing_prediction" if index == 9 else "execution_failure: fixture"
        ),
        "retrieval_only": False,
        "oracle_ability": "MR",
        "detected_ability": None,
        "ability_used": None,
        "distill_fired": False,
        "distill_calls": 0,
    } for index, qid in enumerate(ids)]
    zero = _zero_usage()
    scores = {
        "multi-session": {"accuracy": 0.0, "count": 500},
        "OVERALL": {"accuracy": 0.0, "count": 500},
    }
    artifact = {
        "benchmark": "LongMemEval", "version": "strict-v1",
        "date": "2026-09-04T12:00:00+00:00", "manifest": manifest,
        "config": config, "models": models, "scores": scores,
        "abstention_diagnostics": protocol._abstention_from_rows(rows),
        "conditional_judged_only": {"accuracy": None, "count": 0},
        "execution": {
            "counts": {
                "expected": 500, "attempted": 499,
                "unique_attempted": 499, "total_attempts": 499,
                "completed": 0, "failed": 500, "missing": 1,
            },
            "segments": [{
                "segment_id": "segment-full", "status": "complete",
                "elapsed_s": 2.0, "attempted_attempts": 499,
                "model_identities": models,
                "reader_usage": zero, "judge_usage": zero,
                "retrieval_usage": zero,
                "memory_pipeline_usage": zero,
                "embedding_usage": embedding_usage_snapshot(
                    None, configured=False
                ),
                "indexing_runs": [],
            }],
        },
        "per_question": rows,
    }
    artifact["result_digest"] = content_hash(rows)
    return artifact, ids


def test_official_alignment_is_distinct_from_development_comparability(
    monkeypatch,
):
    artifact, ids = _full_failure_artifact(monkeypatch)
    config = artifact["config"]
    judge = artifact["models"]["judge"]
    config.update({
        "judge_protocol": "official",
        "judge_model": protocol.LME_OFFICIAL_JUDGE_MODEL,
        "judge_base_url": protocol.LME_OFFICIAL_JUDGE_BASE_URL,
        "judge_extra_body_obj": {},
    })
    judge.update({
        "provider": "openai",
        "model": protocol.LME_OFFICIAL_JUDGE_MODEL,
        "base_url": protocol.LME_OFFICIAL_JUDGE_BASE_URL,
        "temperature": protocol.LME_OFFICIAL_JUDGE_TEMPERATURE,
        "max_tokens": protocol.LME_OFFICIAL_JUDGE_MAX_TOKENS,
        "n": 1,
        "extra_body": {},
        "protocol": "official",
        "evaluator_commit": protocol.LME_EVALUATOR_COMMIT,
        "evaluator_sha256": protocol.LME_EVALUATOR_SHA256,
        "verdict_parser": protocol.LME_OFFICIAL_VERDICT_PARSER,
        "prompt_exact_official": True,
        "retry_policy": protocol.LME_LOCAL_RETRY_POLICY,
    })
    config["official_judge_match"] = protocol.official_judge_match(
        config, artifact["models"]
    )
    artifact["execution"]["segments"][0]["model_identities"] = artifact["models"]
    old = artifact["manifest"]
    artifact["manifest"] = build_manifest(
        benchmark="LongMemEval", code_sha256=old["code_hash"],
        data_sha256=old["data_hash"], config=config,
        models=artifact["models"], seed=old["seed"],
        expected_ids=ids, protocol_split="full",
    )
    validated = protocol.validate_strict_artifact(artifact)
    assert validated["official_scoring_semantics_aligned"] is True
    assert validated["official_protocol_aligned"] is False
    assert validated["official_denominator_validated"] is True
    assert validated["development_only"] is True
    assert validated["official_comparable"] is False


def test_official_export_is_exact_ordered_exclusive_and_failure_complete(
    tmp_path, monkeypatch,
):
    artifact, ids = _full_failure_artifact(monkeypatch)
    destination = tmp_path / "official.jsonl"
    summary = protocol.export_official_predictions(artifact, destination)
    lines = [json.loads(line) for line in destination.read_text().splitlines()]
    assert summary["count"] == 500
    assert [line["question_id"] for line in lines] == ids
    assert all(list(line) == ["question_id", "hypothesis"] for line in lines)
    assert all(line["hypothesis"] == "" for line in lines)
    with pytest.raises(BenchmarkIntegrityError, match="already exists"):
        protocol.export_official_predictions(artifact, destination)


def test_official_export_rejects_an_unfinalized_execution(tmp_path, monkeypatch):
    artifact, _ids = _full_failure_artifact(monkeypatch)
    artifact["execution"]["segments"][0]["status"] = "running"
    with pytest.raises(BenchmarkIntegrityError, match="completed strict"):
        protocol.export_official_predictions(artifact, tmp_path / "official.jsonl")


def test_official_export_cli_constructs_zero_clients(monkeypatch, tmp_path):
    calls = {"clients": 0, "exports": 0}

    class ForbiddenClient:
        def __init__(self, *_args, **_kwargs):
            calls["clients"] += 1
            raise AssertionError("provider client constructed")

    def fake_export(source, destination):
        calls["exports"] += 1
        return {"count": 500, "path": str(destination)}

    monkeypatch.setattr(lme, "LLMClient", ForbiddenClient)
    monkeypatch.setattr(lme, "export_official_predictions", fake_export)
    monkeypatch.setattr(sys, "argv", [
        "longmemeval_adapter.py", "--export-official", "artifact.json",
        "--official-output", str(tmp_path / "official.jsonl"),
    ])
    lme._main()
    assert calls == {"clients": 0, "exports": 1}


def _write_cli_fixture_dataset(directory: Path) -> list[dict]:
    rows = []
    for index in range(2):
        rows.append({
            "question_id": f"qid-{index}",
            "question_type": "multi-session",
            "question": f"question {index}",
            "answer": f"answer {index}",
            "question_date": "2025-01-03",
            "answer_session_ids": [f"s-{index}"],
            "haystack_session_ids": [f"s-{index}"],
            "haystack_dates": ["2025-01-01"],
            "haystack_sessions": [[{
                "role": "user", "content": f"answer {index}",
                "has_answer": True,
            }]],
        })
    (directory / "longmemeval_s_cleaned.json").write_text(json.dumps(rows))
    return rows


def test_freeze_calibration_constructs_zero_provider_clients(monkeypatch, tmp_path):
    _write_cli_fixture_dataset(tmp_path)
    constructed = []

    class ForbiddenClient:
        def __init__(self, *_args, **_kwargs):
            constructed.append(True)
            raise AssertionError("provider client constructed")

    receipt = tmp_path / "frozen.json"
    monkeypatch.setattr(lme, "LLMClient", ForbiddenClient)
    monkeypatch.setattr(sys, "argv", [
        "longmemeval_adapter.py", "--data-dir", str(tmp_path),
        "--results-dir", str(tmp_path), "--sample", "0", "--no-prereg",
        "--freeze-calibration", str(receipt),
    ])
    lme._main()
    assert constructed == []
    frozen = json.loads(receipt.read_text())
    assert frozen["claim"] == "internal deterministic split; not an official benchmark split"


def test_completed_checkpoint_path_constructs_zero_provider_clients(monkeypatch, tmp_path):
    rows = _write_cli_fixture_dataset(tmp_path)
    constructed = []

    class ForbiddenClient:
        def __init__(self, *_args, **_kwargs):
            constructed.append(True)
            raise AssertionError("provider client constructed")

    durable_rows = tuple({
        "question_id": row["question_id"],
        "question_type": row["question_type"],
        "correct": False,
        "benchmark_failure": "execution_failure: fixture",
        "retrieval_only": False,
        "oracle_ability": "MR",
        "detected_ability": None,
        "ability_used": None,
    } for row in rows)

    resumed_segments = []

    class CompleteLedger:
        def __init__(self, _path, *, manifest, expected_ids, **_kwargs):
            self.manifest = manifest
            self.pending_ids = ()
            self.path = tmp_path / "complete.checkpoint.json"

        def reconcile(self):
            return SimpleNamespace(rows=durable_rows)

        def finalize(self):
            return {
                "counts": {"expected": 2},
                "execution_segments": list(resumed_segments),
            }

        def update_execution_segment(self, _segment_id, segment):
            resumed_segments.append(segment)

        def close(self):
            pass

    published = []
    monkeypatch.setattr(lme, "LLMClient", ForbiddenClient)
    monkeypatch.setattr(lme, "AtomicCheckpoint", CompleteLedger)
    monkeypatch.setattr(
        lme, "publish_checkpoint_artifact",
        lambda ledger, path, payload: published.append((ledger, path, payload)) or {},
    )
    monkeypatch.setattr(lme, "write_latest_pointer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "longmemeval_adapter.py", "--data-dir", str(tmp_path),
        "--results-dir", str(tmp_path), "--sample", "0", "--no-prereg",
        "--resume-from", str(tmp_path / "complete.checkpoint.json"),
    ])
    owned = []
    lme._main(owned)
    assert constructed == []
    assert len(published) == 1
    assert len(owned) == 1
    assert len(resumed_segments) == 1
    assert resumed_segments[0]["status"] == "complete"
    assert resumed_segments[0]["attempted_attempts"] == 0


@pytest.mark.parametrize(("embeddings", "meter_failure"), [
    (False, False), (True, False), (True, True),
])
def test_adapter_emits_a_strictly_validated_deterministic_smoke_artifact(
    monkeypatch, tmp_path, embeddings, meter_failure,
):
    rows = _write_cli_fixture_dataset(tmp_path)

    class MeterOnlyClient:
        call_count = 0
        request_attempts = 0
        successful_responses = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_latency_s = 0.0
        cost_usd = 0.0
        token_usage_available = True

        def __init__(self, *_args, **_kwargs):
            pass

        def chat(self, *_args, **_kwargs):
            raise AssertionError("failure-only smoke rows make no provider call")

    def fail_without_spend(_qi, _total, q_data, *_args, **_kwargs):
        return {
            "question_id": q_data["question_id"],
            "question_type": q_data["question_type"],
            "correct": False,
            "benchmark_failure": "execution_failure: deterministic smoke",
            "retrieval_only": False,
            "oracle_ability": "MR",
            "detected_ability": None,
            "ability_used": None,
            "distill_fired": False,
            "distill_calls": 0,
        }

    monkeypatch.setattr(lme, "LLMClient", MeterOnlyClient)
    monkeypatch.setattr(lme, "_evaluate_one_question", fail_without_spend)
    if meter_failure:
        monkeypatch.setattr(
            lme, "usage_snapshot",
            lambda _client: (_ for _ in ()).throw(
                KeyboardInterrupt("meter unavailable")
            ),
        )
    argv = [
        "longmemeval_adapter.py", "--data-dir", str(tmp_path),
        "--results-dir", str(tmp_path), "--sample", "0", "--no-prereg",
        "--no-dream", "--workers", "2", "--api-key", "fixture-key",
    ]
    if embeddings:
        argv.append("--embeddings")
    monkeypatch.setattr(sys, "argv", argv)
    lme.main()
    archives = list(tmp_path.glob("longmemeval-v2-hymem-*-strict-*.json"))
    assert len(archives) == 1
    artifact = json.loads(archives[0].read_text())
    validated = protocol.validate_strict_artifact(artifact)
    assert validated["counts"]["expected"] == len(rows)
    assert validated["scores"]["OVERALL"] == {
        "accuracy": 0.0, "count": len(rows),
    }
    assert [row["question_id"] for row in validated["rows"]] == [
        row["question_id"] for row in rows
    ]
    assert artifact["models"]["memory_pipeline"]["provider"] == "deepseek"
    assert artifact["config"]["distill_prompt_version"] == "v2"
    assert artifact["models"]["embedding"]["configured"] is embeddings
    if meter_failure:
        assert validated["answer_calls"] is None
        assert validated["total_tokens"] is None


def test_cli_wrapper_closes_owned_checkpoint_on_baseexception(monkeypatch):
    closed = []

    class Ledger:
        def close(self):
            closed.append(True)
            raise SystemExit("cleanup")

    def interrupted(owned):
        owned.append(Ledger())
        raise KeyboardInterrupt("stop")

    monkeypatch.setattr(lme, "_main", interrupted)
    with pytest.raises(KeyboardInterrupt, match="stop"):
        lme.main()
    assert closed == [True]
