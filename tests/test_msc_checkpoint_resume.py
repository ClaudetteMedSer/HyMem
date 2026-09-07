"""Restart-safety and strict-artifact coverage for the MSC recall runner."""

from __future__ import annotations

import copy
import json
import sys
import threading
from pathlib import Path

import pytest

from benchmarks import msc_adapter as msc
from benchmarks import msc_registry
from tests.archive_evidence_fixtures import skipped_indexing, scoped_indexing
from benchmarks.extraction_canary import (
    ExtractionCanaryError,
)
from benchmarks.strictness import (
    AtomicCheckpoint,
    BenchmarkCleanupError,
    content_hash,
    embedding_usage_snapshot,
)
from hymem.extraction.llm import LLMRequest


def _example(item_id: str) -> dict:
    return {
        "id": item_id,
        "sessions": [[{"role": "user", "content": "I have two dogs."}]],
        "session_dates": ["2026-01-01 00:00"],
        "question": "How many dogs?",
        "answer": "Two.",
        "persona_facts": ["I have two dogs"],
        "n_sessions": 1,
    }


def _row(example: dict, *, failed: bool = False) -> dict:
    row = {
        "id": example["id"],
        "question_id": example["id"],
        "question_type": "recall",
        "correct": not failed,
    }
    if failed:
        row["benchmark_failure"] = "probe_failure:RuntimeError"
    return row


def _zero_llm(calls: int = 0) -> dict:
    return {
        "calls": calls, "calls_available": True,
        "request_attempts": calls, "request_attempts_available": True,
        "successful_responses": calls,
        "successful_responses_available": True,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "token_usage_available": True,
        "latency_s": 0.0, "latency_available": True,
        "cost_usd": 0.0, "cost_available": True,
    }


def _zero_embedding() -> dict:
    return embedding_usage_snapshot(None, configured=False)


def _runtime(example: dict, *, pipeline_calls: int = 0, args=None) -> dict:
    scope = f"msc:{example['id']}"
    return {
        "scope_id": scope,
        "indexing": skipped_indexing(scope) if args is None else scoped_indexing(scope, args),
        "memory_pipeline_usage": _zero_llm(pipeline_calls),
        "embedding_usage": _zero_embedding(),
    }


class _CanaryClient:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.model = msc._HYMEM_MODEL
        self.base_url = msc._DEEPSEEK_BASE_URL
        self.thinking_mode = "auto"
        self.effective_extra_body = {"thinking": {"type": "disabled"}}
        self.call_count = 0
        self.request_attempts = 0
        self.successful_responses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.cost_usd = None
        self.token_usage_available = True

    def complete(self, request: LLMRequest) -> str:
        raise AssertionError("identity-only canary fixture must not make calls")


def _passed_canary() -> dict:
    """Build an exact live report from the current canary policy contract."""

    policy = msc.extraction_canary_policy()
    claims = copy.deepcopy(policy["expected_claims"])
    completion_calls = policy["minimum_pass_completion_calls"]
    client = _CanaryClient()
    report = {
        **policy,
        "status": "passed",
        "client": {
            "client_class": (
                f"{type(client).__module__}.{type(client).__qualname__}"
            ),
            "model": client.model,
            "base_url": client.base_url,
            "thinking_mode": client.thinking_mode,
            "effective_extra_body": copy.deepcopy(
                client.effective_extra_body
            ),
        },
        "client_closed": True,
        "completion_calls": completion_calls,
        "provider_attempts": completion_calls,
        "initial_prepartition_leaves": policy[
            "expected_prepartition_leaves"
        ],
        "duplicate_triples_collapsed": 0,
        "usage": _zero_llm(completion_calls),
        "execution_path": copy.deepcopy(policy["normal_execution_path"]),
        "matched_supported_claims": len(claims),
        "missing_expected_claim_indexes": [],
        "valid_triples_returned": len(claims),
        "valid_markers_returned": 0,
        "claim_evidence": [
            {"expected_claim_index": index, **claim}
            for index, claim in enumerate(claims)
        ],
    }
    msc.validate_extraction_canary_report(
        report,
        expected_mode="required",
        expected_client=msc.extraction_canary_client_policy(
            base_url=client.base_url,
            model=client.model,
            thinking=client.thinking_mode,
        ),
        require_client_closed=True,
    )
    return report


def _failed_canary() -> dict:
    """Apply the intended clean-empty failure to a current valid baseline."""

    report = _passed_canary()
    report.update({
        "status": "failed",
        "failure_reason": "clean_empty",
        "failure_details": [],
        "matched_supported_claims": 0,
        "missing_expected_claim_indexes": list(
            range(len(report["expected_claims"]))
        ),
        "valid_triples_returned": 0,
        "valid_markers_returned": 0,
        "claim_evidence": [],
    })
    msc.validate_extraction_canary_report(
        report,
        expected_mode="failed",
        expected_client=msc.extraction_canary_client_policy(
            base_url=report["client"]["base_url"],
            model=report["client"]["model"],
            thinking=report["client"]["thinking_mode"],
        ),
        require_client_closed=True,
    )
    return report


def _argv(
    tmp_path: Path, checkpoint_flag: str, checkpoint: Path, *extra: str,
) -> list[str]:
    return [
        "msc_adapter.py", "--sim", "--no-dream",
        checkpoint_flag, str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
        *extra,
    ]


def _success(example, _args, _answer, _judge, **kwargs):
    row = _row(example)
    kwargs["on_checkpoint"](row, _runtime(example, args=_args))
    return row


class _UsageClient:
    def __init__(self):
        self.call_count = 0
        self.request_attempts = 0
        self.successful_responses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.cost_usd = 0.0
        self.token_usage_available = True

    def close(self):
        return None


def _scored_success(example, args, answer, judge, **kwargs):
    for client in (answer, judge):
        client.call_count += 1
        client.request_attempts += 1
        client.successful_responses += 1
    return _success(example, args, answer, judge, **kwargs)


def _archive(tmp_path: Path) -> tuple[Path, dict]:
    archives = list((tmp_path / "results").glob("msc-*-strict-*.json"))
    assert len(archives) == 1
    return archives[0], json.loads(archives[0].read_text())


def _rehash_manifest(artifact: dict) -> None:
    manifest = artifact["manifest"]
    manifest["run_id"] = content_hash({
        key: value for key, value in manifest.items() if key != "run_id"
    })


def _rebind_config(artifact: dict) -> None:
    artifact["manifest"]["config"] = copy.deepcopy(artifact["config"])
    artifact["manifest"]["config_hash"] = content_hash(artifact["config"])
    _rehash_manifest(artifact)


def _rebind_models(artifact: dict) -> None:
    artifact["manifest"]["models"] = copy.deepcopy(artifact["models"])
    artifact["manifest"]["model_hash"] = content_hash(artifact["models"])
    for segment in artifact["execution"]["segments"]:
        segment["model_identities"] = copy.deepcopy(artifact["models"])
    _rehash_manifest(artifact)


@pytest.fixture(autouse=True)
def _stable_code_hash(monkeypatch):
    monkeypatch.setattr(msc, "msc_code_hash", lambda: "sha256:" + "a" * 64)


def test_crash_after_durable_callback_resumes_only_pending_in_exact_order(
    monkeypatch, tmp_path,
):
    examples = [_example("q1"), _example("q2"), _example("q3")]
    checkpoint = tmp_path / "crash.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: examples)

    class Crash(BaseException):
        pass

    def crashing(example, _args, _answer, _judge, **kwargs):
        kwargs["on_checkpoint"](_row(example), _runtime(example))
        raise Crash("after durable callback")

    monkeypatch.setattr(msc, "run_recall", crashing)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    with pytest.raises(Crash):
        msc.main()
    state = json.loads(checkpoint.read_text())
    assert list(state["entries"]) == ["q1"]
    assert not list((tmp_path / "results").glob("msc-*-strict-*.json"))

    seen = []

    def resumed(example, _args, _answer, _judge, **kwargs):
        seen.append(example["id"])
        return _success(example, _args, _answer, _judge, **kwargs)

    monkeypatch.setattr(msc, "run_recall", resumed)
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, "--resume-from", checkpoint)
    )
    msc.main()
    _, artifact = _archive(tmp_path)
    assert seen == ["q2", "q3"]
    assert [row["question_id"] for row in artifact["per_question"]] == [
        "q1", "q2", "q3",
    ]


def test_parallel_out_of_order_callbacks_reconcile_to_manifest_order(
    monkeypatch, tmp_path,
):
    examples = [_example("q1"), _example("q2"), _example("q3")]
    checkpoint = tmp_path / "parallel.checkpoint.json"
    barrier = threading.Barrier(3)
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: examples)

    def evaluate(example, _args, _answer, _judge, **kwargs):
        barrier.wait(timeout=5)
        # Deterministic reverse completion after all workers have started.
        if example["id"] == "q1":
            threading.Event().wait(0.03)
        elif example["id"] == "q2":
            threading.Event().wait(0.01)
        return _success(example, _args, _answer, _judge, **kwargs)

    monkeypatch.setattr(msc, "run_recall", evaluate)
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--checkpoint", checkpoint, "--workers", "3"),
    )
    msc.main()
    _, artifact = _archive(tmp_path)
    assert [row["question_id"] for row in artifact["per_question"]] == [
        "q1", "q2", "q3",
    ]
    assert artifact["execution"]["counts"]["unique_attempted"] == 3


def test_duplicate_callback_aborts_without_archive(monkeypatch, tmp_path):
    example = _example("q1")
    checkpoint = tmp_path / "duplicate.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])

    def duplicate(item, _args, _answer, _judge, **kwargs):
        row = _row(item)
        kwargs["on_checkpoint"](row, _runtime(item))
        kwargs["on_checkpoint"](row, _runtime(item))
        return row

    monkeypatch.setattr(msc, "run_recall", duplicate)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    with pytest.raises(BaseException) as caught:
        msc.main()
    assert isinstance(caught.value.__cause__, msc.BenchmarkIntegrityError)
    assert json.loads(checkpoint.read_text())["entries"]["q1"]["attempts"] == 1
    assert not list((tmp_path / "results").glob("msc-*-strict-*.json"))


def test_inconsistent_callback_identity_aborts_without_archive(
    monkeypatch, tmp_path,
):
    example = _example("q1")
    checkpoint = tmp_path / "identity-row.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])

    def inconsistent(item, _args, _answer, _judge, **kwargs):
        row = _row(item)
        row["id"] = "other"
        kwargs["on_checkpoint"](row, _runtime(item))
        return row

    monkeypatch.setattr(msc, "run_recall", inconsistent)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    with pytest.raises(BaseException) as caught:
        msc.main()
    assert isinstance(caught.value.__cause__, msc.BenchmarkIntegrityError)
    assert json.loads(checkpoint.read_text())["entries"] == {}
    assert not list((tmp_path / "results").glob("msc-*-strict-*.json"))


@pytest.mark.parametrize(
    "returned",
    [
        None,
        {"id": "other", "question_type": "recall", "correct": True},
        {"id": "q1", "question_type": "recall", "correct": True},
    ],
    ids=("malformed", "wrong-id", "missing-callback"),
)
def test_return_contract_violation_aborts_without_failed_score_or_archive(
    monkeypatch, tmp_path, returned,
):
    example = _example("q1")
    checkpoint = tmp_path / "returned-contract.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])
    monkeypatch.setattr(msc, "run_recall", lambda *_a, **_k: returned)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    with pytest.raises(BaseException) as caught:
        msc.main()
    assert isinstance(caught.value.__cause__, msc.BenchmarkIntegrityError)
    assert json.loads(checkpoint.read_text())["entries"] == {}
    assert not list((tmp_path / "results").glob("msc-*-strict-*.json"))


def test_failure_after_atomic_row_write_aborts_but_row_is_resumable(
    monkeypatch, tmp_path,
):
    example = _example("q1")
    checkpoint = tmp_path / "post-write.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])
    original = msc.AtomicCheckpoint.record

    def write_then_fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("private post-write failure")

    monkeypatch.setattr(msc.AtomicCheckpoint, "record", write_then_fail)
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    with pytest.raises(BaseException) as caught:
        msc.main()
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "q1" in json.loads(checkpoint.read_text())["entries"]
    assert not list((tmp_path / "results").glob("msc-*-strict-*.json"))


def test_item_failure_plus_item_cleanup_failure_blocks_publication(
    monkeypatch, tmp_path,
):
    example = _example("q1")
    checkpoint = tmp_path / "combined-failure.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])

    class Adapter:
        APERTURE = msc.MSCAdapter.APERTURE

        def __init__(self, *_a, **kwargs):
            self.pipeline_llm = None
            self.embedding_client = None
            self.embeddings = kwargs.get("embeddings", False)
            self.sim = kwargs.get("sim", False)

        def open(self):
            return self

        def close(self, **_kwargs):
            raise RuntimeError("private close path /secret/store")

    monkeypatch.setattr(msc, "MSCAdapter", Adapter)
    monkeypatch.setattr(
        msc, "prepare_indexing",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("private provider token secret")
        ),
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    with pytest.raises(BenchmarkCleanupError):
        msc.main()
    state_text = checkpoint.read_text()
    assert "q1" in json.loads(state_text)["entries"]
    assert "private provider" not in state_text
    assert "/secret/store" not in state_text
    assert not list((tmp_path / "results").glob("msc-*-strict-*.json"))


def test_terminal_resume_does_no_canary_client_store_or_question_work(
    monkeypatch, tmp_path,
):
    example = _example("q1")
    checkpoint = tmp_path / "terminal.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    msc.main()

    def forbidden(*_a, **_k):
        raise AssertionError("terminal resume performed benchmark work")

    monkeypatch.setattr(msc, "run_recall", forbidden)
    monkeypatch.setattr(msc, "run_configured_extraction_canary", forbidden)
    monkeypatch.setattr(msc, "_build_llm", forbidden)
    monkeypatch.setattr(msc, "MSCAdapter", forbidden)
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, "--resume-from", checkpoint)
    )
    msc.main()


def test_retry_failure_is_attempted_once_per_invocation(monkeypatch, tmp_path):
    example = _example("q1")
    checkpoint = tmp_path / "retry.checkpoint.json"
    calls = []
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])

    def fail(item, _args, _answer, _judge, **kwargs):
        calls.append(item["id"])
        row = _row(item, failed=True)
        kwargs["on_checkpoint"](row, _runtime(item))
        return row

    monkeypatch.setattr(msc, "run_recall", fail)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    msc.main()
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--resume-from", checkpoint, "--retry-failures"),
    )
    msc.main()
    state = json.loads(checkpoint.read_text())
    assert calls == ["q1", "q1"]
    assert state["entries"]["q1"]["attempts"] == 2
    assert state["counts"]["total_attempts"] == 2
    assert state["execution_segments"][-1]["attempted_attempts"] == 1


def test_failed_row_is_terminal_without_explicit_retry(monkeypatch, tmp_path):
    example = _example("q1")
    checkpoint = tmp_path / "terminal-failure.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])

    def fail(item, _args, _answer, _judge, **kwargs):
        row = _row(item, failed=True)
        kwargs["on_checkpoint"](row, _runtime(item))
        return row

    monkeypatch.setattr(msc, "run_recall", fail)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    msc.main()
    monkeypatch.setattr(
        msc, "run_recall",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("failure retried without --retry-failures")
        ),
    )
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, "--resume-from", checkpoint)
    )
    msc.main()


def test_identity_mismatch_fails_before_canary_client_or_store(
    monkeypatch, tmp_path,
):
    example = _example("q1")
    checkpoint = tmp_path / "identity.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    msc.main()

    touched = []

    def forbidden(*_a, **_k):
        touched.append(True)
        raise AssertionError("work happened before identity rejection")

    monkeypatch.setattr(msc, "run_recall", forbidden)
    monkeypatch.setattr(msc, "run_configured_extraction_canary", forbidden)
    monkeypatch.setattr(msc, "_build_llm", forbidden)
    monkeypatch.setattr(msc, "MSCAdapter", forbidden)
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--resume-from", checkpoint, "--top-k", "11"),
    )
    with pytest.raises(msc.BenchmarkIntegrityError, match="identity mismatch"):
        msc.main()
    assert touched == []


def test_stale_v16_checkpoint_manifest_cannot_resume(monkeypatch, tmp_path):
    example = _example("q1")
    checkpoint = tmp_path / "stale-v16.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    msc.main()

    state = json.loads(checkpoint.read_text())
    manifest = state["manifest"]
    manifest["config"]["extraction_canary"]["version"] = (
        "hymem-phase1-extraction-canary-v16"
    )
    manifest["config_hash"] = content_hash(manifest["config"])
    manifest["run_id"] = content_hash({
        key: value for key, value in manifest.items() if key != "run_id"
    })
    state["run_id"] = manifest["run_id"]
    checkpoint.write_text(json.dumps(state, sort_keys=True))

    touched = []

    def forbidden(*_args, **_kwargs):
        touched.append(True)
        raise AssertionError("stale checkpoint reached benchmark work")

    monkeypatch.setattr(msc, "run_recall", forbidden)
    monkeypatch.setattr(msc, "run_configured_extraction_canary", forbidden)
    monkeypatch.setattr(msc, "_build_llm", forbidden)
    monkeypatch.setattr(msc, "MSCAdapter", forbidden)
    monkeypatch.setattr(sys, "argv", _argv(
        tmp_path, "--resume-from", checkpoint,
    ))

    with pytest.raises(msc.BenchmarkIntegrityError, match="run identity mismatch"):
        msc.main()
    assert touched == []


def test_select_examples_obeys_receipt_order_not_input_order():
    examples = [_example("q1"), _example("q2"), _example("q3")]
    selected = msc._select_examples_by_id(examples, ("q3", "q1"))
    assert [item["id"] for item in selected] == ["q3", "q1"]


def test_freeze_and_holdout_are_no_spend_and_exactly_receipt_bound(
    monkeypatch, tmp_path,
):
    examples = [_example(f"q{i}") for i in range(1, 7)]
    receipt = tmp_path / "calibration.json"
    checkpoint = tmp_path / "holdout.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: examples)

    def forbidden(*_a, **_k):
        raise AssertionError("calibration freeze spent benchmark work")

    monkeypatch.setattr(msc, "run_recall", forbidden)
    monkeypatch.setattr(msc, "run_configured_extraction_canary", forbidden)
    monkeypatch.setattr(msc, "_build_llm", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--sim", "--no-dream",
        "--freeze-calibration", str(receipt), "--dev-fraction", "0.5",
    ])
    msc.main()
    frozen = json.loads(receipt.read_text())

    seen = []

    def evaluate(example, _args, _answer, _judge, **kwargs):
        seen.append(example["id"])
        return _success(example, _args, _answer, _judge, **kwargs)

    monkeypatch.setattr(msc, "run_recall", evaluate)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--sim", "--no-dream",
        "--protocol-split", "holdout",
        "--calibration-receipt", str(receipt),
        "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    msc.main()
    assert seen == frozen["holdout_ids"]
    state = json.loads(checkpoint.read_text())
    assert state["expected_ids"] == frozen["holdout_ids"]
    assert state["manifest"]["calibration_receipt_hash"] == frozen["receipt_hash"]


def test_calibration_identity_mismatch_fails_before_checkpoint_or_spend(
    monkeypatch, tmp_path,
):
    examples = [_example(f"q{i}") for i in range(1, 5)]
    receipt = tmp_path / "calibration.json"
    checkpoint = tmp_path / "mismatch.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: examples)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--sim", "--no-dream",
        "--freeze-calibration", str(receipt),
    ])
    msc.main()

    touched = []
    monkeypatch.setattr(
        msc, "run_configured_extraction_canary",
        lambda **_k: touched.append("canary"),
    )
    monkeypatch.setattr(
        msc, "_build_llm", lambda *_a, **_k: touched.append("client")
    )
    monkeypatch.setattr(
        msc, "run_recall", lambda *_a, **_k: touched.append("question")
    )
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--sim", "--no-dream", "--top-k", "11",
        "--protocol-split", "holdout",
        "--calibration-receipt", str(receipt),
        "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(msc.BenchmarkIntegrityError, match="receipt config_hash"):
        msc.main()
    assert touched == []
    assert not checkpoint.exists()
    assert not (tmp_path / "results").exists()


def test_full_split_rejects_calibration_receipt_before_dataset_or_spend(
    monkeypatch, tmp_path,
):
    touched = []
    monkeypatch.setattr(
        msc, "load_msc_data", lambda *_a, **_k: touched.append("dataset")
    )
    monkeypatch.setattr(
        msc, "run_configured_extraction_canary",
        lambda **_k: touched.append("canary"),
    )
    monkeypatch.setattr(
        msc, "_build_llm", lambda *_a, **_k: touched.append("client")
    )
    checkpoint = tmp_path / "full-receipt.checkpoint.json"
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--sim", "--no-dream",
        "--calibration-receipt", str(tmp_path / "unused.json"),
        "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(SystemExit) as caught:
        msc.main()
    assert caught.value.code == 2
    assert touched == []
    assert not checkpoint.exists()
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("flag", ["--checkpoint", "--resume-from"])
def test_checkpoint_paths_cannot_alias_latest_pointer(monkeypatch, tmp_path, flag):
    results = tmp_path / "results"
    results.mkdir()
    latest = results / "msc-latest.json"
    original = b'{"archive":"banked.json","run_id":"sha256:banked"}\n'
    latest.write_bytes(original)
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [_example("q1")])
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--sim", "--no-dream", flag, str(latest),
        "--results-dir", str(results),
    ])
    with pytest.raises(SystemExit) as caught:
        msc.main()
    assert caught.value.code == 2
    assert latest.read_bytes() == original


def test_out_cannot_alias_checkpoint(monkeypatch, tmp_path):
    target = tmp_path / "same.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [_example("q1")])
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--sim", "--no-dream",
        "--checkpoint", str(target), "--out", str(target),
        "--results-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(SystemExit) as caught:
        msc.main()
    assert caught.value.code == 2
    assert not target.exists()


def test_malformed_returned_canary_is_durable_before_validation(
    monkeypatch, tmp_path,
):
    example = _example("q1")
    checkpoint = tmp_path / "canary.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])
    malformed = _passed_canary()
    malformed["private_note"] = (
        "Authorization: Bearer private-token /secret/path"
    )
    monkeypatch.setattr(
        msc, "run_configured_extraction_canary",
        lambda **_k: copy.deepcopy(malformed),
    )
    touched = []
    monkeypatch.setattr(
        msc, "_build_llm",
        lambda *_a, **_k: touched.append("client"),
    )
    monkeypatch.setattr(
        msc, "run_recall",
        lambda *_a, **_k: touched.append("question"),
    )
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(msc.BenchmarkIntegrityError, match="extraction canary"):
        msc.main()
    state_text = checkpoint.read_text()
    state = json.loads(state_text)
    report = state["execution_segments"][0]["extraction_canary"]
    assert report == {
        "status": "invalid_report",
        "failure_reason": "internal_validation_failure",
        "reported_status": "passed",
        "completion_calls": malformed["completion_calls"],
        "provider_attempts": malformed["provider_attempts"],
        "client_closed": True,
        "usage": malformed["usage"],
    }
    assert "private-token" not in state_text
    assert "/secret/path" not in state_text
    assert touched == []
    assert not list((tmp_path / "results").glob("msc-*-strict-*.json"))
    resumed = AtomicCheckpoint(
        checkpoint,
        manifest=state["manifest"],
        expected_ids=state["expected_ids"],
        resume=True,
        scored=True,
    )
    resumed.close()

    # A later healthy process can continue from the bounded sentinel, perform
    # the pending question once, and publish an artifact whose consumer accepts
    # both historical and current preflight evidence.
    monkeypatch.setattr(
        msc, "run_configured_extraction_canary", lambda **_k: _passed_canary()
    )
    monkeypatch.setattr(msc, "_build_llm", lambda *_a, **_k: _UsageClient())
    monkeypatch.setattr(msc, "run_recall", _scored_success)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--resume-from", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    msc.main()
    archive, _artifact = _archive(tmp_path)
    recovered = msc_registry.load_msc_artifact(archive)
    statuses = [
        segment["extraction_canary"]["status"]
        for segment in recovered["execution"]["segments"]
    ]
    assert statuses == ["invalid_report", "passed"]


def test_failed_canary_then_resume_pass_is_registry_valid(monkeypatch, tmp_path):
    example = _example("q1")
    checkpoint = tmp_path / "failed-canary.checkpoint.json"
    failed = _failed_canary()
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])

    def fail_canary(**_kwargs):
        raise ExtractionCanaryError("synthetic failed canary", failed)

    monkeypatch.setattr(msc, "run_configured_extraction_canary", fail_canary)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(ExtractionCanaryError):
        msc.main()
    state = json.loads(checkpoint.read_text())
    assert state["execution_segments"][0]["extraction_canary"]["status"] == "failed"
    assert state["entries"] == {}

    monkeypatch.setattr(
        msc, "run_configured_extraction_canary", lambda **_k: _passed_canary()
    )
    monkeypatch.setattr(msc, "_build_llm", lambda *_a, **_k: _UsageClient())
    monkeypatch.setattr(msc, "run_recall", _scored_success)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--resume-from", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    msc.main()
    archive, _artifact = _archive(tmp_path)
    recovered = msc_registry.load_msc_artifact(archive)
    statuses = [
        segment["extraction_canary"]["status"]
        for segment in recovered["execution"]["segments"]
    ]
    assert statuses == ["failed", "passed"]


def test_shared_client_usage_is_snapshotted_once_not_per_question(
    monkeypatch, tmp_path,
):
    examples = [_example("q1"), _example("q2")]
    checkpoint = tmp_path / "usage.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: examples)

    class Counter:
        def __init__(self):
            self.call_count = 0
            self.request_attempts = 0
            self.successful_responses = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.total_latency_s = 0.0
            self.cost_usd = 0.0
            self.token_usage_available = True

        def close(self):
            return None

    clients = []

    def build(*_a, **_k):
        client = Counter()
        clients.append(client)
        return client

    def evaluate(example, _args, answer, judge, **kwargs):
        for client in (answer, judge):
            client.call_count += 1
            client.request_attempts += 1
            client.successful_responses += 1
        row = _row(example)
        kwargs["on_checkpoint"](row, _runtime(example, args=_args))
        return row

    monkeypatch.setattr(msc, "_build_llm", build)
    monkeypatch.setattr(msc, "run_recall", evaluate)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--no-dream",
        "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    msc.main()
    _, artifact = _archive(tmp_path)
    segment = artifact["execution"]["segments"][0]
    assert len(clients) == 2
    assert segment["reader_usage"]["calls"] == 2
    assert segment["judge_usage"]["calls"] == 2
    assert segment["memory_pipeline_usage"]["calls"] == 0
    assert segment["attempted_attempts"] == 2
    msc_registry.validate_msc_artifact(artifact)

    for role in ("reader_usage", "judge_usage"):
        forged = copy.deepcopy(artifact)
        usage = forged["execution"]["segments"][0][role]
        usage["calls"] = 1
        usage["request_attempts"] = 1
        usage["successful_responses"] = 1
        with pytest.raises(
            msc.BenchmarkIntegrityError, match="calls are below"
        ):
            msc_registry.validate_msc_artifact(forged)


def test_shared_client_cleanup_failure_leaves_checkpoint_without_archive(
    monkeypatch, tmp_path,
):
    example = _example("q1")
    checkpoint = tmp_path / "client-cleanup.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])

    class BadClient:
        call_count = request_attempts = successful_responses = 1
        prompt_tokens = completion_tokens = total_tokens = 0
        total_latency_s = 0.0
        cost_usd = 0.0
        token_usage_available = True

        def close(self):
            raise RuntimeError("private transport cleanup")

    monkeypatch.setattr(msc, "_build_llm", lambda *_a, **_k: BadClient())
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--no-dream",
        "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(BenchmarkCleanupError):
        msc.main()
    assert checkpoint.exists()
    assert json.loads(checkpoint.read_text())["status"] == "complete"
    assert not list((tmp_path / "results").glob("msc-*-strict-*.json"))


def test_producer_validation_failure_closes_clients_and_publishes_nothing(
    monkeypatch, tmp_path,
):
    example = _example("q1")
    checkpoint = tmp_path / "producer-validation.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])
    clients = []

    class Client(_UsageClient):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    def build(*_args, **_kwargs):
        client = Client()
        clients.append(client)
        return client

    def reject(_artifact):
        assert clients and all(not client.closed for client in clients)
        raise msc.BenchmarkIntegrityError("synthetic producer validation failure")

    monkeypatch.setattr(msc, "_build_llm", build)
    monkeypatch.setattr(msc, "run_recall", _scored_success)
    monkeypatch.setattr(msc_registry, "validate_msc_artifact", reject)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--no-dream",
        "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(
        msc.BenchmarkIntegrityError, match="producer validation failure"
    ):
        msc.main()
    assert len(clients) == 2 and all(client.closed for client in clients)
    assert json.loads(checkpoint.read_text())["status"] == "complete"
    assert not list((tmp_path / "results").glob("msc-*-strict-*.json"))
    assert not (tmp_path / "results" / "msc-latest.json").exists()


def test_legacy_sidecar_failure_occurs_after_authoritative_publication(
    monkeypatch, tmp_path,
):
    example = _example("q1")
    checkpoint = tmp_path / "sidecar.checkpoint.json"
    sidecar = tmp_path / "legacy.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])
    monkeypatch.setattr(msc, "run_recall", _success)
    original = Path.write_text

    def fail_sidecar(self, *args, **kwargs):
        if self == sidecar:
            raise OSError("presentation failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_sidecar)
    monkeypatch.setattr(sys, "argv", _argv(
        tmp_path, "--checkpoint", checkpoint, "--out", str(sidecar),
    ))
    with pytest.raises(OSError, match="presentation failure"):
        msc.main()
    assert list((tmp_path / "results").glob("msc-*-strict-*.json"))
    assert (tmp_path / "results" / "msc-latest.json").exists()


def test_failures_do_not_leak_raw_exception_or_operational_path(
    monkeypatch, tmp_path,
):
    example = _example("q1")
    checkpoint = tmp_path / "bounded.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [example])
    monkeypatch.setattr(
        msc, "run_recall",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("Bearer secret-value at /private/store.sqlite")
        ),
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    msc.main()
    _, artifact = _archive(tmp_path)
    encoded = json.dumps(artifact)
    assert "secret-value" not in encoded
    assert "/private/store.sqlite" not in encoded
    row = artifact["per_question"][0]
    assert row["benchmark_failure"] == "probe_failure:RuntimeError"


def test_manifest_binds_effective_defaults_flags_and_search_aperture(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "identity.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [_example("q1")])
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", _argv(
        tmp_path, "--checkpoint", checkpoint,
        "--graph-multihop", "--no-facts", "--rules-extraction",
    ))
    msc.main()
    config = json.loads(checkpoint.read_text())["manifest"]["config"]
    effective = config["effective_hymem_config"]
    assert config["pipeline_search_multiplier"] == 3
    assert effective["message_fts_top_k"] == 15
    assert effective["fts_top_k"] == 10
    assert effective["graph_top_k"] == 10
    assert effective["rerank_top_k"] == 20
    assert effective["graph_multihop_enabled"] is True
    assert effective["facts_enabled"] is False
    assert effective["rules_extraction_enabled"] is True
    assert effective["aggregation_nodes_enabled"] is False
    assert isinstance(effective["content_redaction_enabled"], bool)
    assert "redact_secrets" not in effective


def test_simulation_archive_has_stub_identity_and_no_accuracy(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sim.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [_example("q1")])
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    msc.main()
    archive, artifact = _archive(tmp_path)
    assert artifact["manifest"]["scored_run"] is False
    assert artifact["scores"] is None
    assert artifact["strict_accuracy"] is None
    assert artifact["models"]["reader"]["configured"] is False
    assert artifact["models"]["judge"]["configured"] is False
    assert artifact["models"]["memory_pipeline"]["provider"] == "local_stub"
    assert artifact["models"]["embedding"]["configured"] is False
    assert msc_registry.load_msc_artifact(archive)["strict_accuracy"] is None


def test_sim_embeddings_rejected_before_dataset_checkpoint_or_results(
    monkeypatch, tmp_path,
):
    touched = []
    monkeypatch.setattr(
        msc, "load_msc_data", lambda *_a, **_k: touched.append("dataset")
    )
    checkpoint = tmp_path / "invalid.checkpoint.json"
    monkeypatch.setattr(sys, "argv", _argv(
        tmp_path, "--checkpoint", checkpoint, "--embeddings",
    ))
    with pytest.raises(SystemExit) as caught:
        msc.main()
    assert caught.value.code == 2
    assert touched == []
    assert not checkpoint.exists()
    assert not (tmp_path / "results").exists()


def test_no_dream_and_dream_per_session_rejected_before_dataset_or_spend(
    monkeypatch, tmp_path,
):
    touched = []
    monkeypatch.setattr(
        msc, "load_msc_data", lambda *_a, **_k: touched.append("dataset")
    )
    monkeypatch.setattr(
        msc, "_build_llm", lambda *_a, **_k: touched.append("client")
    )
    checkpoint = tmp_path / "invalid-dream-policy.checkpoint.json"
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--no-dream", "--dream-per-session",
        "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(SystemExit) as caught:
        msc.main()
    assert caught.value.code == 2
    assert touched == []
    assert not checkpoint.exists()
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "model_flag",
    ["--answer-model", "--judge-model", "--hymem-model"],
)
def test_deprecated_model_alias_rejected_before_dataset_client_or_checkpoint(
    monkeypatch, tmp_path, model_flag,
):
    touched = []
    monkeypatch.setattr(
        msc, "load_msc_data", lambda *_a, **_k: touched.append("dataset")
    )
    monkeypatch.setattr(
        msc, "_build_llm", lambda *_a, **_k: touched.append("client")
    )
    checkpoint = tmp_path / "alias.checkpoint.json"
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", model_flag, "deepseek-chat", "--no-dream",
        "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(SystemExit) as caught:
        msc.main()
    assert caught.value.code == 2
    assert touched == []
    assert not checkpoint.exists()
    assert not (tmp_path / "results").exists()


def test_recurrence_rejects_strict_controls_before_dataset_work(
    monkeypatch, tmp_path,
):
    touched = []
    monkeypatch.setattr(
        msc, "load_msc_data", lambda *_a, **_k: touched.append("dataset")
    )
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--probe-mode", "recurrence", "--sim",
        "--checkpoint", str(tmp_path / "forbidden.json"),
    ])
    with pytest.raises(SystemExit) as caught:
        msc.main()
    assert caught.value.code == 2
    assert touched == []


def test_registry_default_scan_excludes_latest_checkpoint_and_legacy_sidecar(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "run.checkpoint.json"
    sidecar = tmp_path / "results" / "legacy.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [_example("q1")])
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", _argv(
        tmp_path, "--checkpoint", checkpoint, "--out", str(sidecar),
    ))
    msc.main()
    paths = msc_registry.discover_msc_archives(tmp_path / "results")
    assert len(paths) == 1
    assert paths[0].name.startswith("msc-")
    assert paths[0].name != "msc-latest.json"
    assert len(msc_registry.scan_msc_archives(tmp_path / "results")) == 1
    assert msc_registry.load_msc_artifact(
        tmp_path / "results" / "msc-latest.json"
    )["manifest"]["expected_count"] == 1


def test_msc_registry_rejects_stale_v16_canary_report(monkeypatch, tmp_path):
    checkpoint = tmp_path / "stale-canary.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [_example("q1")])
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", _argv(
        tmp_path, "--checkpoint", checkpoint,
    ))
    msc.main()
    _, artifact = _archive(tmp_path)
    artifact["execution"]["segments"][0]["extraction_canary"]["version"] = (
        "hymem-phase1-extraction-canary-v16"
    )

    with pytest.raises(msc.BenchmarkIntegrityError, match="extraction canary"):
        msc_registry.validate_msc_artifact(artifact)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda c: c.__setitem__("indexing_max_cycles", 0),
            "config bounds",
        ),
        (
            lambda c: c.__setitem__("indexing_timeout_s", 0.0),
            "indexing timeout",
        ),
        (
            lambda c: c.__setitem__("sample_strategy", "source-order"),
            "sample strategy",
        ),
        (
            lambda c: c.__setitem__("embeddings", True),
            "embedding configuration",
        ),
        (
            lambda c: c["extraction_canary"].__setitem__(
                "max_completion_calls",
                c["extraction_canary"]["max_completion_calls"] + 1,
            ),
            "canary policy",
        ),
        (
            lambda c: c["extraction_canary"].__setitem__(
                "version", "hymem-phase1-extraction-canary-v16",
            ),
            "canary policy",
        ),
        (
            lambda c: c["effective_hymem_config"][
                "extraction_contract"
            ].__setitem__(
                "identity",
                "hymem-extraction-contract-sha256-v1:" + "0" * 64,
            ),
            "extraction-canary policy",
        ),
        (
            lambda c: c["effective_hymem_config"][
                "extraction_contract"
            ].pop("identity"),
            "extraction-canary policy",
        ),
        (
            lambda c: c["effective_hymem_config"].__setitem__(
                "graph_multihop_enabled", True
            ),
            "requested/effective",
        ),
        (
            lambda c: c["effective_hymem_config"].__setitem__(
                "content_redaction_enabled", "yes"
            ),
            "effective HyMem flag",
        ),
        (
            lambda c: c["effective_hymem_config"].__setitem__(
                "facts_enabled",
                not c["effective_hymem_config"]["facts_enabled"],
            ),
            "effective HyMem identity",
        ),
        (
            lambda c: c["effective_hymem_config"].__setitem__(
                "facts_extraction_enabled",
                not c["effective_hymem_config"]["facts_extraction_enabled"],
            ),
            "effective HyMem identity",
        ),
        (
            lambda c: c["effective_hymem_config"].__setitem__(
                "rules_extraction_enabled",
                not c["effective_hymem_config"]["rules_extraction_enabled"],
            ),
            "effective HyMem identity",
        ),
        (
            lambda c: c["effective_hymem_config"].__setitem__(
                "rerank_top_k",
                c["effective_hymem_config"]["rerank_top_k"] + 1,
            ),
            "effective HyMem identity",
        ),
        (
            lambda c: c.__setitem__("dream_per_session", True),
            "dream policy",
        ),
        (
            lambda c: c.__setitem__("synthetic_base_date", "2024-01-01"),
            "synthetic date",
        ),
    ],
)
def test_registry_rejects_score_affecting_config_semantic_tamper(
    monkeypatch, tmp_path, mutation, message,
):
    checkpoint = tmp_path / "config-tamper.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [_example("q1")])
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    msc.main()
    _, original = _archive(tmp_path)
    artifact = copy.deepcopy(original)
    mutation(artifact["config"])
    _rebind_config(artifact)
    with pytest.raises(msc.BenchmarkIntegrityError, match=message):
        msc_registry.validate_msc_artifact(artifact)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda m: m["reader"].__setitem__("client_class", "forged.Client"),
            "reader request identity",
        ),
        (
            lambda m: m["reader"].__setitem__("provider", "forged"),
            "endpoint identity",
        ),
        (
            lambda m: m["reader"].__setitem__("temperature", 0.5),
            "reader request identity",
        ),
        (
            lambda m: m["reader"].__setitem__("max_tokens", 2048),
            "reader request identity",
        ),
        (
            lambda m: m["reader"].__setitem__(
                "extra_body", {"model": "override"}
            ),
            "request body",
        ),
        (
            lambda m: m["reader"].__setitem__(
                "extra_body", {"thinking": {"type": "enabled"}}
            ),
            "request body",
        ),
        (
            lambda m: m["judge"].__setitem__("protocol", "forged"),
            "judge request identity",
        ),
        (
            lambda m: m["judge"].__setitem__("max_tokens", 1024),
            "judge request identity",
        ),
        (
            lambda m: m["memory_pipeline"].__setitem__(
                "client_class", "forged.Client"
            ),
            "memory-pipeline request identity",
        ),
        (
            lambda m: m["memory_pipeline"].__setitem__("provider", "forged"),
            "endpoint identity",
        ),
        (
            lambda m: m["memory_pipeline"].__setitem__(
                "effective_extra_body", {}
            ),
            "memory-pipeline request identity",
        ),
        (
            lambda m: m["memory_pipeline"].__setitem__(
                "base_url", "http://public.example/v1"
            ),
            "provider identity is unsafe",
        ),
        (
            lambda m: m["memory_pipeline"].__setitem__(
                "model", "deepseek-chat"
            ),
            "provider identity is unsafe",
        ),
        (
            lambda m: m["embedding"].__setitem__("backend", "forged"),
            "embedding identity is malformed or obsolete",
        ),
    ],
)
def test_registry_rejects_fully_rebound_model_identity_tamper(
    monkeypatch, tmp_path, mutation, message,
):
    checkpoint = tmp_path / "model-tamper.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [_example("q1")])
    monkeypatch.setattr(msc, "_build_llm", lambda *_a, **_k: _UsageClient())
    monkeypatch.setattr(msc, "run_recall", _scored_success)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--no-dream",
        "--checkpoint", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    msc.main()
    _, original = _archive(tmp_path)
    artifact = copy.deepcopy(original)
    mutation(artifact["models"])
    _rebind_models(artifact)
    with pytest.raises(msc.BenchmarkIntegrityError, match=message):
        msc_registry.validate_msc_artifact(artifact)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda a: a["config"].__setitem__("top_k", 99), "top-level identity"),
        (lambda a: a["manifest"].__setitem__("schema", "forged"), "manifest"),
        (lambda a: a["execution"]["counts"].__setitem__("completed", 0), "counts"),
        (lambda a: a["per_question"][0].__setitem__("correct", "yes"), "verdict"),
        (
            lambda a: a["per_question"][0].__setitem__("id", "different"),
            "id/question_id",
        ),
        (
            lambda a: a["per_question"][0].__setitem__(
                "question_type", "other"
            ),
            "question type",
        ),
        (
            lambda a: a["per_question"][0].__setitem__(
                "indexing_scope_id", "msc:other"
            ),
            "indexing scope",
        ),
        (lambda a: a.__setitem__("result_digest", "sha256:" + "0" * 64), "digest"),
        (lambda a: a.__setitem__("strict_accuracy", 0.0), "accuracy"),
        (lambda a: a["models"]["reader"].__setitem__("configured", True), "identity"),
        (
            lambda a: a["execution"]["segments"][0].__setitem__(
                "indexing_runs", []
            ),
            "indexing evidence",
        ),
        (
            lambda a: a["execution"]["segments"][0]["indexing_runs"][0]
            .__setitem__("scope_id", "msc:unknown"),
            "indexing evidence",
        ),
        (lambda a: a["execution"]["segments"].append(
            copy.deepcopy(a["execution"]["segments"][0])
        ), "segment id"),
    ],
)
def test_registry_rejects_tamper_classes(
    monkeypatch, tmp_path, mutation, message,
):
    checkpoint = tmp_path / "tamper.checkpoint.json"
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [_example("q1")])
    monkeypatch.setattr(msc, "run_recall", _success)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    msc.main()
    archive, original = _archive(tmp_path)
    artifact = copy.deepcopy(original)
    mutation(artifact)
    if artifact["manifest"] != original["manifest"]:
        _rehash_manifest(artifact)
    with pytest.raises(msc.BenchmarkIntegrityError, match=message):
        msc_registry.validate_msc_artifact(artifact)
