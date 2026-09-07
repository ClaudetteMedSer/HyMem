"""Focused restart-safety tests for the strict LoCoMo runner."""

from __future__ import annotations

import copy
import json
import sys
import threading
from pathlib import Path

import pytest

from benchmarks import locomo_adapter as locomo
from benchmarks import locomo_registry
from benchmarks.strictness import (
    BenchmarkIntegrityError,
    content_hash,
    embedding_usage_snapshot,
)


def _conversation(conv_id: str, *ids: str) -> dict:
    return {
        "id": conv_id,
        "speaker_a": "Ada",
        "speaker_b": "Ben",
        "sessions": [[{"role": "user", "content": "source"}]],
        "session_dates": ["2026-01-01 00:00"],
        "n_sessions": 1,
        "evidence_map": {},
        "qa": [
            {
                "qa_id": item_id,
                "question_id": item_id,
                "question": f"question {item_id}",
                "answer": "answer",
                "adversarial_answer": "",
                "category": 1,
                "qtype": "multi-hop",
                "judge_type": "multi-session",
                "evidence": [],
            }
            for item_id in ids
        ],
    }


def _row(conv: dict, question: dict, *, failed: bool = False) -> dict:
    row = {
        "id": question["qa_id"],
        "question_id": question["question_id"],
        "conv_id": conv["id"],
        "question_type": question["qtype"],
        "category": question["category"],
        "question": question["question"],
        "correct": not failed,
    }
    if failed:
        row["benchmark_failure"] = "execution_failure:RuntimeError"
    return row


def _runtime(conv: dict) -> dict:
    zero = {
        "calls": 0, "calls_available": True,
        "request_attempts": 0, "request_attempts_available": True,
        "successful_responses": 0,
        "successful_responses_available": True,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "token_usage_available": True,
        "latency_s": 0.0, "latency_available": True,
        "cost_usd": None, "cost_available": False,
    }
    return {
        "scope_id": f"locomo:{conv['id']}",
        "indexing": {
            "scope_id": f"locomo:{conv['id']}",
            "complete": False, "healthy": False, "comparable": False,
            "skip_reason": "simulation",
        },
        "memory_pipeline_usage": zero,
        "embedding_usage": embedding_usage_snapshot(None, configured=False),
    }


def _argv(tmp_path: Path, checkpoint_flag: str, checkpoint: Path, *extra: str):
    return [
        "locomo_adapter.py", "--sim", "--no-dream",
        checkpoint_flag, str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
        *extra,
    ]


def _strict_archive(monkeypatch, tmp_path: Path) -> tuple[Path, dict]:
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / "registry.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    def evaluate(conversation, _args, _answer, _judge, **kwargs):
        row = _row(conversation, conversation["qa"][0])
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    locomo.main()
    archive = next((tmp_path / "results").glob("locomo-*strict-*.json"))
    return archive, json.loads(archive.read_text())


def _rehash_manifest(artifact: dict) -> None:
    manifest = artifact["manifest"]
    manifest["run_id"] = content_hash({
        key: value for key, value in manifest.items() if key != "run_id"
    })


@pytest.fixture(autouse=True)
def _stable_code_hash(monkeypatch):
    monkeypatch.setattr(locomo, "locomo_code_hash", lambda: "sha256:" + "a" * 64)


def test_crash_after_row_is_durable_and_resume_skips_it(monkeypatch, tmp_path):
    conv = _conversation("conv-one", "q1", "q2")
    checkpoint = tmp_path / "run.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    class Crash(BaseException):
        pass

    def crashing(conversation, _args, _answer, _judge, **kwargs):
        first = conversation["qa"][0]
        kwargs["on_checkpoint"](_row(conversation, first), _runtime(conversation))
        raise Crash("process died after durable row")

    monkeypatch.setattr(locomo, "evaluate_conversation", crashing)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    with pytest.raises(Crash):
        locomo.main()

    crashed = json.loads(checkpoint.read_text())
    assert list(crashed["entries"]) == ["q1"]
    assert crashed["status"] == "running"

    seen = []

    def resumed(conversation, _args, _answer, _judge, **kwargs):
        assert kwargs["pending_ids"] == {"q2"}
        seen.extend(kwargs["pending_ids"])
        rows = []
        for question in conversation["qa"]:
            row = _row(conversation, question)
            kwargs["on_checkpoint"](row, _runtime(conversation))
            rows.append(row)
        return rows

    monkeypatch.setattr(locomo, "evaluate_conversation", resumed)
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, "--resume-from", checkpoint)
    )
    locomo.main()

    state = json.loads(checkpoint.read_text())
    assert seen == ["q2"]
    assert state["status"] == "complete"
    assert state["counts"]["completed"] == 2
    assert list(state["entries"]) == ["q1", "q2"]


def test_parallel_callbacks_reconcile_in_manifest_order(monkeypatch, tmp_path):
    convs = [
        _conversation("conv-a", "q1", "q2"),
        _conversation("conv-b", "q3", "q4"),
    ]
    checkpoint = tmp_path / "parallel.checkpoint.json"
    barrier = threading.Barrier(2)
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: convs)

    def evaluate(conversation, _args, _answer, _judge, **kwargs):
        barrier.wait(timeout=5)
        rows = []
        for question in reversed(conversation["qa"]):
            row = _row(conversation, question)
            kwargs["on_checkpoint"](row, _runtime(conversation))
            rows.append(row)
        return rows

    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--checkpoint", checkpoint, "--workers", "2"),
    )
    locomo.main()

    state = json.loads(checkpoint.read_text())
    assert state["counts"]["unique_attempted"] == 4
    archives = list((tmp_path / "results").glob("locomo-*strict-*.json"))
    assert len(archives) == 1
    artifact = json.loads(archives[0].read_text())
    assert [row["question_id"] for row in artifact["per_question"]] == [
        "q1", "q2", "q3", "q4",
    ]


def test_resume_identity_mismatch_fails_before_work(monkeypatch, tmp_path):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / "identity.checkpoint.json"
    work = []
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    def evaluate(conversation, _args, _answer, _judge, **kwargs):
        work.append(True)
        question = conversation["qa"][0]
        row = _row(conversation, question)
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    locomo.main()
    work.clear()
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--resume-from", checkpoint, "--top-k", "11"),
    )
    with pytest.raises(locomo.BenchmarkIntegrityError, match="identity mismatch"):
        locomo.main()
    assert work == []


def test_manifest_binds_msc_adapter_effective_aperture(monkeypatch, tmp_path):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / "aperture.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    def evaluate(conversation, _args, _answer, _judge, **kwargs):
        question = conversation["qa"][0]
        row = _row(conversation, question)
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    locomo.main()
    manifest = json.loads(checkpoint.read_text())["manifest"]
    effective = manifest["config"]["effective_hymem_config"]
    assert effective["message_fts_top_k"] == 15
    assert effective["fts_top_k"] == 10
    assert effective["graph_top_k"] == 10
    assert effective["rerank_top_k"] == 20
    assert isinstance(effective["content_redaction_enabled"], bool)
    assert "redact_secrets" not in effective


def test_sim_manifest_records_only_actual_stub_identities(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sim-identity.checkpoint.json"
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint),
    )
    locomo.main()
    archive = next((tmp_path / "results").glob("locomo-*strict-*.json"))
    artifact = json.loads(archive.read_text())
    models = artifact["models"]
    assert models["reader"] == {
        "configured": False, "client_class": None,
        "provider": "none", "model": None, "base_url": None,
    }
    assert models["judge"] == {
        "configured": False, "client_class": None,
        "provider": "none", "model": None, "base_url": None,
    }
    assert models["memory_pipeline"]["client_class"] == (
        "hymem.extraction.llm.StubLLMClient"
    )
    assert models["memory_pipeline"]["provider"] == "local_stub"
    assert models["memory_pipeline"]["model"] is None
    assert models["memory_pipeline"]["base_url"] is None
    assert models["embedding"]["configured"] is False
    assert "deepseek" not in json.dumps(models).casefold()
    result = artifact["per_question"][0]
    assert result["answer_model"] is None
    assert result["answer_base_url"] is None
    assert result["judge_model"] is None
    assert result["judge_base_url"] is None
    assert result["hymem_model"] is None
    assert result["hymem_base_url"] is None
    assert result["hymem_client_class"] == (
        "hymem.extraction.llm.StubLLMClient"
    )
    assert "api.deepseek.com" not in json.dumps(artifact).casefold()


def test_sim_embeddings_rejected_before_checkpoint_or_archive(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "sim-embeddings.checkpoint.json"
    results = tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            tmp_path, "--checkpoint", checkpoint, "--embeddings",
        ),
    )
    with pytest.raises(SystemExit) as caught:
        locomo.main()
    assert caught.value.code == 2
    assert not checkpoint.exists()
    assert not results.exists()


def test_terminal_resume_constructs_no_clients_or_benchmark_work(
    monkeypatch, tmp_path,
):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / "terminal.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    def evaluate(conversation, _args, _answer, _judge, **kwargs):
        question = conversation["qa"][0]
        row = _row(conversation, question)
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    locomo.main()

    monkeypatch.setattr(
        locomo, "evaluate_conversation",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("terminal resume evaluated work")
        ),
    )
    monkeypatch.setattr(
        locomo, "_build_llm",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("terminal resume built a provider client")
        ),
    )
    monkeypatch.setattr(
        locomo, "run_configured_extraction_canary",
        lambda **_k: (_ for _ in ()).throw(
            AssertionError("terminal resume repeated the canary")
        ),
    )
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, "--resume-from", checkpoint)
    )
    locomo.main()


def test_failed_rows_are_terminal_unless_retry_is_explicit(monkeypatch, tmp_path):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / "retry.checkpoint.json"
    attempts = []
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    def fail(conversation, _args, _answer, _judge, **kwargs):
        attempts.append("failed")
        question = conversation["qa"][0]
        row = _row(conversation, question, failed=True)
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", fail)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    locomo.main()

    monkeypatch.setattr(
        locomo, "evaluate_conversation",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("terminal failure retried implicitly")
        ),
    )
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, "--resume-from", checkpoint)
    )
    locomo.main()

    def pass_retry(conversation, _args, _answer, _judge, **kwargs):
        attempts.append("completed")
        question = conversation["qa"][0]
        row = _row(conversation, question)
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", pass_retry)
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--resume-from", checkpoint, "--retry-failures"),
    )
    locomo.main()
    state = json.loads(checkpoint.read_text())
    assert attempts == ["failed", "completed"]
    assert state["entries"]["q1"]["attempts"] == 2
    assert state["entries"]["q1"]["status"] == "completed"


def test_retry_that_fails_again_is_recorded_exactly_once(monkeypatch, tmp_path):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / "retry-fails.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    def failed(conversation, _args, _answer, _judge, **kwargs):
        row = _row(conversation, conversation["qa"][0], failed=True)
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", failed)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    locomo.main()
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--resume-from", checkpoint, "--retry-failures"),
    )
    locomo.main()

    state = json.loads(checkpoint.read_text())
    entry = state["entries"]["q1"]
    assert entry["attempts"] == len(entry["attempt_history"]) == 2
    assert entry["status"] == "failed"
    assert state["counts"]["total_attempts"] == 2
    assert state["execution_segments"][-1]["attempted_attempts"] == 1


def test_exception_after_failed_callback_does_not_record_id_twice(
    monkeypatch, tmp_path,
):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / "retry-exception.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    def failed(conversation, _args, _answer, _judge, **kwargs):
        row = _row(conversation, conversation["qa"][0], failed=True)
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", failed)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    locomo.main()

    def failed_then_raises(conversation, _args, _answer, _judge, **kwargs):
        row = _row(conversation, conversation["qa"][0], failed=True)
        kwargs["on_checkpoint"](row, _runtime(conversation))
        raise RuntimeError("private provider diagnostic")

    monkeypatch.setattr(locomo, "evaluate_conversation", failed_then_raises)
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--resume-from", checkpoint, "--retry-failures"),
    )
    locomo.main()
    state = json.loads(checkpoint.read_text())
    assert state["entries"]["q1"]["attempts"] == 2
    assert state["counts"]["total_attempts"] == 2


def test_duplicate_callback_aborts_without_archive(monkeypatch, tmp_path):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / "duplicate.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    def duplicate(conversation, _args, _answer, _judge, **kwargs):
        row = _row(conversation, conversation["qa"][0])
        kwargs["on_checkpoint"](row, _runtime(conversation))
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", duplicate)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    with pytest.raises(BaseException) as caught:
        locomo.main()
    assert isinstance(caught.value.__cause__, locomo.BenchmarkIntegrityError)
    state = json.loads(checkpoint.read_text())
    assert state["entries"]["q1"]["attempts"] == 1
    assert not list((tmp_path / "results").glob("locomo-*strict-*.json"))


def test_checkpoint_write_failure_aborts_without_publication(
    monkeypatch, tmp_path,
):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / "write-crash.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])
    original_record = locomo.AtomicCheckpoint.record

    def persist_then_crash(self, *args, **kwargs):
        original_record(self, *args, **kwargs)
        raise RuntimeError("simulated process death after atomic replace")

    monkeypatch.setattr(locomo.AtomicCheckpoint, "record", persist_then_crash)

    def evaluate(conversation, _args, _answer, _judge, **kwargs):
        row = _row(conversation, conversation["qa"][0])
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", checkpoint))
    with pytest.raises(BaseException) as caught:
        locomo.main()
    assert isinstance(caught.value, RuntimeError)
    assert "process death" in str(caught.value)
    assert "q1" in json.loads(checkpoint.read_text())["entries"]
    assert not list((tmp_path / "results").glob("locomo-*strict-*.json"))


@pytest.mark.parametrize("workers", [1, 2])
def test_checkpoint_record_abort_snapshots_spend_then_resumes_once(
    monkeypatch, tmp_path, workers,
):
    """A rejected row still owns its attempt/spend, but never becomes a row."""

    conversations = [
        _conversation("conv-zero", "q0"),
        _conversation("conv-one", "q1"),
    ]
    checkpoint = tmp_path / f"record-fault-resume-{workers}.checkpoint.json"
    results_dir = tmp_path / "results"
    monkeypatch.setattr(
        locomo, "load_locomo_data", lambda *_a, **_k: conversations
    )

    original_record = locomo.AtomicCheckpoint.record
    original_update_segment = locomo.AtomicCheckpoint.update_execution_segment
    fault = RuntimeError("primary checkpoint secret=primary-token")
    record_faulted = False
    phase = "fault"
    peer_started = threading.Event()
    peer_drained = threading.Event()
    post_drain_snapshots = []

    def runtime_with_spend(conversation):
        runtime = _runtime(conversation)
        calls = 2 if conversation["id"] == "conv-zero" else 3
        runtime["memory_pipeline_usage"].update({
            "calls": calls,
            "request_attempts": calls + 1,
            "successful_responses": calls,
            "prompt_tokens": calls * 10,
            "completion_tokens": calls * 2,
            "total_tokens": calls * 12,
            "latency_s": calls / 10,
        })
        return runtime

    def fail_record_once(self, *args, **kwargs):
        nonlocal record_faulted
        if not record_faulted:
            record_faulted = True
            raise fault
        return original_record(self, *args, **kwargs)

    monkeypatch.setattr(locomo.AtomicCheckpoint, "record", fail_record_once)

    def observe_final_snapshot(self, segment_id, metrics):
        if phase == "fault" and metrics.get("status") == "complete":
            if workers == 2:
                assert peer_drained.is_set(), (
                    "abort usage was frozen before the running peer drained"
                )
            post_drain_snapshots.append(copy.deepcopy(metrics))
        return original_update_segment(self, segment_id, metrics)

    monkeypatch.setattr(
        locomo.AtomicCheckpoint,
        "update_execution_segment",
        observe_final_snapshot,
    )

    def evaluate(conversation, _args, _answer, _judge, **kwargs):
        question = conversation["qa"][0]
        row = _row(conversation, question)
        runtime = runtime_with_spend(conversation)
        kwargs["_on_attempt"](question["question_id"])
        if phase == "fault" and workers == 2:
            if conversation["id"] == "conv-zero":
                assert peer_started.wait(5), "peer evaluation did not start"
            else:
                peer_started.set()
                assert kwargs["_parallel_stop"].wait(5), (
                    "checkpoint abort did not stop the peer"
                )
                try:
                    kwargs["on_checkpoint"](row, runtime)
                finally:
                    peer_drained.set()
                raise AssertionError("an aborted future published its row")
        kwargs["on_checkpoint"](row, runtime)
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    worker_args = ("--workers", str(workers)) if workers > 1 else ()
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--checkpoint", checkpoint, *worker_args),
    )

    with pytest.raises(BaseException) as caught:
        locomo.main()

    assert caught.value is fault
    aborted = json.loads(checkpoint.read_text())
    assert aborted["status"] == "running"
    assert aborted["entries"] == {}
    assert len(post_drain_snapshots) == 1
    first_segment = aborted["execution_segments"][0]
    assert first_segment["status"] == "complete"
    assert first_segment["attempted_attempts"] == workers
    assert first_segment["memory_pipeline_usage"]["calls"] == (
        2 if workers == 1 else 5
    )
    assert not list(results_dir.glob("locomo-*strict-*.json"))
    assert not (results_dir / "locomo-latest.json").exists()

    phase = "resume"
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--resume-from", checkpoint, *worker_args),
    )
    locomo.main()

    resumed = json.loads(checkpoint.read_text())
    assert resumed["status"] == "complete"
    assert list(resumed["entries"]) == ["q0", "q1"]
    assert all(
        entry["attempts"] == len(entry["attempt_history"]) == 1
        for entry in resumed["entries"].values()
    )
    assert resumed["execution_segments"][0] == first_segment
    assert resumed["execution_segments"][1]["status"] == "complete"
    assert resumed["execution_segments"][1]["attempted_attempts"] == 2
    assert resumed["counts"]["total_attempts"] == workers + 2
    archives = list(results_dir.glob("locomo-*strict-*.json"))
    assert len(archives) == 1
    artifact = json.loads(archives[0].read_text())
    assert [row["question_id"] for row in artifact["per_question"]] == [
        "q0", "q1",
    ]
    # The strict registry must accept orphan attempt accounting rather than
    # silently dropping the first run's spend at ingestion time.
    assert locomo_registry._locomo_row(artifact, archives[0]) is not None


def test_provider_capable_locomo_abort_resume_retains_reader_and_judge_spend(
    monkeypatch, tmp_path,
):
    conv = _conversation("conv-provider", "q-provider")
    checkpoint = tmp_path / "provider-spend.checkpoint.json"
    results_dir = tmp_path / "results"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])
    monkeypatch.setattr(locomo, "_print_report", lambda *_a, **_k: None)

    class MeterClient:
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

        def spend(self):
            self.call_count += 1
            self.request_attempts += 1
            self.successful_responses += 1
            self.prompt_tokens += 2
            self.completion_tokens += 1
            self.total_tokens += 3

        def close(self):
            pass

    monkeypatch.setattr(
        locomo, "_build_llm", lambda *_args, **_kwargs: MeterClient(),
    )

    def paid_evaluate(conversation, _args, answer, judge, **kwargs):
        question = conversation["qa"][0]
        kwargs["_on_attempt"](question["question_id"])
        answer.spend()
        judge.spend()
        row = _row(conversation, question)
        runtime = _runtime(conversation)
        runtime["indexing"] = None
        kwargs["on_checkpoint"](row, runtime)
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", paid_evaluate)
    original_record = locomo.AtomicCheckpoint.record
    primary = RuntimeError("one-shot provider row persistence fault")
    failed = False

    def fail_once(self, *args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise primary
        return original_record(self, *args, **kwargs)

    monkeypatch.setattr(locomo.AtomicCheckpoint, "record", fail_once)
    base_argv = [
        "locomo_adapter.py", "--no-dream", "--api-key", "fixture-key",
        "--answer-api-key", "fixture-key", "--judge-api-key", "fixture-key",
        "--results-dir", str(results_dir),
    ]
    monkeypatch.setattr(sys, "argv", [
        *base_argv, "--checkpoint", str(checkpoint),
    ])

    with pytest.raises(RuntimeError) as caught:
        locomo.main()
    assert caught.value is primary
    aborted = json.loads(checkpoint.read_text())
    assert aborted["entries"] == {}
    assert aborted["execution_segments"][0]["reader_usage"]["calls"] == 1
    assert aborted["execution_segments"][0]["judge_usage"][
        "request_attempts"
    ] == 1
    assert not list(results_dir.glob("locomo-*strict-*.json"))

    monkeypatch.setattr(locomo.AtomicCheckpoint, "record", original_record)
    monkeypatch.setattr(sys, "argv", [
        *base_argv, "--resume-from", str(checkpoint),
    ])
    locomo.main()

    final = json.loads(checkpoint.read_text())
    assert final["counts"]["total_attempts"] == 2
    assert [segment["reader_usage"]["calls"] for segment in
            final["execution_segments"]] == [1, 1]
    assert [segment["judge_usage"]["calls"] for segment in
            final["execution_segments"]] == [1, 1]
    archive = next(results_dir.glob("locomo-*strict-*.json"))
    registry_row = locomo_registry._locomo_row(
        json.loads(archive.read_text()), archive,
    )
    assert registry_row["answer_calls"] == 2
    assert registry_row["judge_calls"] == 2


def test_parallel_abort_handoff_keeps_pre_qa_indexing_spend_without_row(
    monkeypatch, tmp_path,
):
    conversations = [
        _conversation("conv-zero", "q0"),
        _conversation("conv-one", "q1"),
    ]
    checkpoint = tmp_path / "pre-qa-indexing-spend.checkpoint.json"
    monkeypatch.setattr(
        locomo, "load_locomo_data", lambda *_a, **_k: conversations
    )

    class PipelineMeter:
        call_count = 0
        request_attempts = 0
        successful_responses = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_latency_s = 0.0
        cost_usd = None
        token_usage_available = True

    class EmbeddingMeter:
        backend = "none"
        quality = "none"
        network_free = True
        model = None
        dim = None
        call_count = 0
        request_attempts = 0
        successful_responses = 0
        input_count = 0
        input_characters = 0
        prompt_tokens = None
        total_tokens = None
        token_usage_available = False
        total_latency_s = 0.0
        cost_usd = None

    closed = []

    class Adapter:
        APERTURE = dict(locomo.MSCAdapter.APERTURE)

        def __init__(self, db_path, **_kwargs):
            self.db_path = Path(db_path)
            self.pipeline_llm = PipelineMeter()
            self.embedding_client = EmbeddingMeter()

        def open(self):
            return self

        def close(self):
            closed.append(self.db_path)

    monkeypatch.setattr(locomo, "MSCAdapter", Adapter)

    peer_indexed = threading.Event()
    stops = {}

    def prepare(adapter, conversation, _args, *, scope_id, reuse):
        del reuse
        if conversation["id"] == "conv-zero":
            pipeline_calls, embedding_calls = 2, 1
            assert peer_indexed.wait(5), "peer did not reach indexing"
        else:
            pipeline_calls, embedding_calls = 7, 4
            peer_indexed.set()
            assert stops[conversation["id"]].wait(5), (
                "peer indexing was not cancelled after the record fault"
            )
        adapter.pipeline_llm.call_count = pipeline_calls
        adapter.pipeline_llm.request_attempts = pipeline_calls + 1
        adapter.pipeline_llm.successful_responses = pipeline_calls
        adapter.pipeline_llm.prompt_tokens = pipeline_calls * 10
        adapter.pipeline_llm.completion_tokens = pipeline_calls * 2
        adapter.pipeline_llm.total_tokens = pipeline_calls * 12
        adapter.pipeline_llm.total_latency_s = pipeline_calls / 10
        adapter.embedding_client.call_count = embedding_calls
        adapter.embedding_client.request_attempts = embedding_calls + 1
        adapter.embedding_client.successful_responses = embedding_calls
        adapter.embedding_client.input_count = embedding_calls * 2
        adapter.embedding_client.input_characters = embedding_calls * 20
        adapter.embedding_client.total_latency_s = embedding_calls / 10
        return {
            "scope_id": scope_id,
            "complete": False,
            "healthy": False,
            "comparable": False,
            "skip_reason": "simulation",
        }

    monkeypatch.setattr(locomo, "prepare_indexing", prepare)
    evaluated = []

    def evaluate_qa(question, conversation, *_args, **_kwargs):
        evaluated.append(conversation["id"])
        return _row(conversation, question)

    monkeypatch.setattr(locomo, "evaluate_qa", evaluate_qa)
    actual_evaluate = locomo.evaluate_conversation

    def expose_stop(conversation, *args, **kwargs):
        stops[conversation["id"]] = kwargs["_parallel_stop"]
        return actual_evaluate(conversation, *args, **kwargs)

    monkeypatch.setattr(locomo, "evaluate_conversation", expose_stop)
    fault = RuntimeError("one-shot record fault")
    record_calls = 0

    def fail_record(_self, *_args, **_kwargs):
        nonlocal record_calls
        record_calls += 1
        raise fault

    monkeypatch.setattr(locomo.AtomicCheckpoint, "record", fail_record)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            tmp_path, "--checkpoint", checkpoint, "--workers", "2",
        ),
    )

    with pytest.raises(BaseException) as caught:
        locomo.main()

    assert caught.value is fault
    assert record_calls == 1
    assert evaluated == ["conv-zero"]
    assert len(closed) == 2
    state = json.loads(checkpoint.read_text())
    assert state["entries"] == {}
    segment = state["execution_segments"][0]
    assert segment["status"] == "complete"
    assert segment["attempted_attempts"] == 1
    assert segment["memory_pipeline_usage"]["calls"] == 9
    assert segment["memory_pipeline_usage"]["request_attempts"] == 11
    assert segment["memory_pipeline_usage"]["total_tokens"] == 108
    assert segment["embedding_usage"]["calls"] == 5
    assert segment["embedding_usage"]["request_attempts"] == 7
    assert segment["embedding_usage"]["input_count"] == 10
    assert [run["scope_id"] for run in segment["indexing_runs"]] == [
        "locomo:conv-one", "locomo:conv-zero",
    ]
    assert not list((tmp_path / "results").glob("locomo-*strict-*.json"))
    assert not (tmp_path / "results" / "locomo-latest.json").exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_checkpoint_abort_keeps_primary_when_segment_snapshot_fails(
    monkeypatch, tmp_path, capsys, workers,
):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / f"record-and-segment-fault-{workers}.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    primary_fault = RuntimeError("primary secret=primary-token")
    secondary_fault = RuntimeError("secondary secret=secondary-token")
    original_update_segment = locomo.AtomicCheckpoint.update_execution_segment
    primaries_seen = []

    def fail_record(_self, *_args, **_kwargs):
        raise primary_fault

    def fail_abort_snapshot(self, segment_id, metrics):
        if metrics.get("status") == "complete":
            primaries_seen.append(sys.exc_info()[1])
            raise secondary_fault
        return original_update_segment(self, segment_id, metrics)

    def evaluate(conversation, _args, _answer, _judge, **kwargs):
        question = conversation["qa"][0]
        kwargs["_on_attempt"](question["question_id"])
        kwargs["on_checkpoint"](
            _row(conversation, question), _runtime(conversation)
        )
        raise AssertionError("failed persistence callback returned")

    monkeypatch.setattr(locomo.AtomicCheckpoint, "record", fail_record)
    monkeypatch.setattr(
        locomo.AtomicCheckpoint,
        "update_execution_segment",
        fail_abort_snapshot,
    )
    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    worker_args = ("--workers", str(workers)) if workers > 1 else ()
    monkeypatch.setattr(
        sys, "argv",
        _argv(tmp_path, "--checkpoint", checkpoint, *worker_args),
    )

    with pytest.raises(BaseException) as caught:
        locomo.main()

    assert caught.value is primary_fault
    assert primaries_seen == [caught.value]
    notes = "\n".join(getattr(caught.value, "__notes__", ()))
    assert '"stage":"execution_segment_snapshot"' in notes
    assert '"exception_type":"RuntimeError"' in notes
    assert "primary-token" not in notes
    assert "secondary-token" not in notes
    assert "secondary-token" not in capsys.readouterr().err
    state = json.loads(checkpoint.read_text())
    assert state["status"] == "running"
    assert state["entries"] == {}
    assert state["execution_segments"][0]["status"] == "running"
    assert state["execution_segments"][0]["attempted_attempts"] == 0
    assert not list((tmp_path / "results").glob("locomo-*strict-*.json"))


def test_parallel_checkpoint_abort_cancels_queued_locomo_evaluation(
    monkeypatch, tmp_path,
):
    conversations = [
        _conversation(f"conv-{index}", f"q{index}") for index in range(10)
    ]
    checkpoint = tmp_path / "bounded-parallel.checkpoint.json"
    monkeypatch.setattr(
        locomo, "load_locomo_data", lambda *_a, **_k: conversations
    )

    started: list[str] = []
    started_lock = threading.Lock()
    peer_started = threading.Event()
    abort_observed = threading.Event()

    def gated_evaluate(conversation, _args, _answer, _judge, **kwargs):
        stop = kwargs["_parallel_stop"]
        with started_lock:
            started.append(conversation["id"])
            ordinal = len(started)
        question = conversation["qa"][0]
        row = _row(conversation, question)
        if ordinal == 1:
            assert peer_started.wait(5), "second initial worker never started"
            kwargs["on_checkpoint"](row, _runtime(conversation))
            raise AssertionError("failed persistence callback returned")
        if ordinal == 2:
            peer_started.set()
            assert stop.wait(5), "parallel abort was not signalled"
            abort_observed.set()
            # Exercise the lock-protected callback gate after the fault: this
            # must stop before a second AtomicCheckpoint.record invocation.
            kwargs["on_checkpoint"](row, _runtime(conversation))
            raise AssertionError("post-abort persistence callback returned")
        raise AssertionError("queued conversation executed after abort")

    fault = RuntimeError("synthetic checkpoint persistence failure")
    record_calls = 0

    def fail_record(_self, *_args, **_kwargs):
        nonlocal record_calls
        record_calls += 1
        raise fault

    monkeypatch.setattr(locomo, "evaluate_conversation", gated_evaluate)
    monkeypatch.setattr(locomo.AtomicCheckpoint, "record", fail_record)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            tmp_path, "--checkpoint", checkpoint, "--workers", "2",
        ),
    )

    with pytest.raises(BaseException) as caught:
        locomo.main()

    assert caught.value is fault
    assert record_calls == 1
    assert set(started) == {"conv-0", "conv-1"}
    assert len(started) == 2
    assert abort_observed.is_set()
    state = json.loads(checkpoint.read_text())
    assert state["status"] == "running"
    assert state["entries"] == {}
    assert not list((tmp_path / "results").glob("locomo-*strict-*.json"))
    assert not (tmp_path / "results" / "locomo-latest.json").exists()


def test_parallel_structural_worker_stops_queued_locomo_evaluation(
    monkeypatch, tmp_path,
):
    conversations = [
        _conversation(f"conv-{index}", f"q{index}") for index in range(10)
    ]
    checkpoint = tmp_path / "bounded-worker-fault.checkpoint.json"
    monkeypatch.setattr(
        locomo, "load_locomo_data", lambda *_a, **_k: conversations
    )

    fault = BenchmarkIntegrityError("synthetic structural worker failure")
    started: list[str] = []
    started_lock = threading.Lock()
    peer_started = threading.Event()
    abort_observed = threading.Event()

    def gated_failure(conversation, _args, _answer, _judge, **kwargs):
        stop = kwargs["_parallel_stop"]
        with started_lock:
            started.append(conversation["id"])
            ordinal = len(started)
        if ordinal == 1:
            assert peer_started.wait(5), "second initial worker never started"
            kwargs["_on_fatal_abort"](fault)
            raise fault
        if ordinal == 2:
            peer_started.set()
            assert stop.wait(5), "worker fault did not signal cancellation"
            abort_observed.set()
            raise locomo._ParallelConversationStopped()
        raise AssertionError("queued conversation executed after worker fault")

    record_calls = 0

    def count_record(_self, *_args, **_kwargs):
        nonlocal record_calls
        record_calls += 1
        raise AssertionError("a structural worker fault must not be recorded")

    monkeypatch.setattr(locomo, "evaluate_conversation", gated_failure)
    monkeypatch.setattr(locomo.AtomicCheckpoint, "record", count_record)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            tmp_path, "--checkpoint", checkpoint, "--workers", "2",
        ),
    )

    with pytest.raises(BaseException) as caught:
        locomo.main()

    assert type(caught.value).__name__ == "_CheckpointPersistenceAbort"
    assert caught.value.__cause__ is fault
    assert record_calls == 0
    assert set(started) == {"conv-0", "conv-1"}
    assert len(started) == 2
    assert abort_observed.is_set()
    state = json.loads(checkpoint.read_text())
    assert state["status"] == "running"
    assert state["entries"] == {}


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("failure_site", ["returned_rows", "evaluator"])
def test_structural_conversation_failures_abort_without_publication(
    monkeypatch, tmp_path, workers, failure_site,
):
    """Structural adapter faults are not ordinary per-conversation failures."""

    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / f"structural-{failure_site}-{workers}.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    def structurally_invalid(*_args, **_kwargs):
        if failure_site == "evaluator":
            raise locomo.BenchmarkIntegrityError("synthetic structural failure")
        return {}  # `_record_returned` requires a list of exact row objects.

    monkeypatch.setattr(locomo, "evaluate_conversation", structurally_invalid)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            tmp_path, "--checkpoint", checkpoint,
            "--workers", str(workers),
        ),
    )

    with pytest.raises(BaseException) as caught:
        locomo.main()

    assert type(caught.value).__name__ == "_CheckpointPersistenceAbort"
    assert isinstance(caught.value.__cause__, locomo.BenchmarkIntegrityError)
    state = json.loads(checkpoint.read_text())
    assert state["status"] == "running"
    assert state["entries"] == {}
    assert not list((tmp_path / "results").glob("locomo-*strict-*.json"))
    assert not (tmp_path / "results" / "locomo-latest.json").exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_inner_question_integrity_failure_aborts_without_publication(
    monkeypatch, tmp_path, workers,
):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / f"inner-structural-{workers}.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    class Adapter:
        APERTURE = dict(locomo.MSCAdapter.APERTURE)
        pipeline_llm = None

        def __init__(self, db_path, **_kwargs):
            self.db_path = Path(db_path)

        def build_config(self):
            from hymem import HyMemConfig
            return HyMemConfig(root=self.db_path.parent)

        def open(self):
            return self

        def close(self):
            pass

    monkeypatch.setattr(locomo, "MSCAdapter", Adapter)
    monkeypatch.setattr(
        locomo, "prepare_indexing", lambda *_args, **_kwargs: {
            "scope_id": "locomo:conv-one",
            "complete": False,
            "healthy": False,
            "comparable": False,
            "skip_reason": "simulation",
        },
    )
    monkeypatch.setattr(
        locomo, "evaluate_qa",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BenchmarkIntegrityError("synthetic inner integrity failure")
        ),
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            tmp_path, "--checkpoint", checkpoint,
            "--workers", str(workers),
        ),
    )

    with pytest.raises(BaseException) as caught:
        locomo.main()

    assert type(caught.value).__name__ == "_CheckpointPersistenceAbort", repr(
        caught.value
    )
    assert isinstance(caught.value.__cause__, BenchmarkIntegrityError)
    state = json.loads(checkpoint.read_text())
    assert state["status"] == "running"
    assert state["entries"] == {}
    assert not list((tmp_path / "results").glob("locomo-*strict-*.json"))
    assert not (tmp_path / "results" / "locomo-latest.json").exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_ordinary_conversation_failure_remains_a_durable_failed_row(
    monkeypatch, tmp_path, workers,
):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / f"provider-failure-{workers}.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])
    monkeypatch.setattr(
        locomo, "evaluate_conversation",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("provider down")),
    )
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            tmp_path, "--checkpoint", checkpoint,
            "--workers", str(workers),
        ),
    )

    locomo.main()

    state = json.loads(checkpoint.read_text())
    assert state["status"] == "complete"
    assert state["entries"]["q1"]["row"]["benchmark_failure"] == (
        "conversation_failure:RuntimeError"
    )
    assert len(list((tmp_path / "results").glob("locomo-*strict-*.json"))) == 1


@pytest.mark.parametrize("flag", ["--checkpoint", "--resume-from"])
def test_checkpoint_must_not_alias_latest_pointer_bytes(
    monkeypatch, tmp_path, flag,
):
    conv = _conversation("conv-one", "q1")
    results = tmp_path / "results"
    results.mkdir()
    latest = results / "locomo-latest.json"
    original = b'{"archive":"banked.json","run_id":"sha256:banked"}\n'
    latest.write_bytes(original)
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--sim", "--no-dream",
        flag, str(latest), "--results-dir", str(results),
    ])
    with pytest.raises(SystemExit) as caught:
        locomo.main()
    assert caught.value.code == 2
    assert latest.read_bytes() == original
    assert not list(results.glob("locomo-*strict-*.json"))


def test_freeze_and_holdout_select_exact_receipt_ids(monkeypatch, tmp_path):
    conv = _conversation("conv-one", "q1", "q2", "q3", "q4")
    receipt = tmp_path / "calibration.json"
    checkpoint = tmp_path / "holdout.checkpoint.json"
    seen = []
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--sim", "--no-dream",
        "--freeze-calibration", str(receipt), "--dev-fraction", "0.5",
    ])
    locomo.main()
    frozen = json.loads(receipt.read_text())

    def evaluate(conversation, _args, _answer, _judge, **kwargs):
        rows = []
        for question in conversation["qa"]:
            seen.append(question["question_id"])
            row = _row(conversation, question)
            kwargs["on_checkpoint"](row, _runtime(conversation))
            rows.append(row)
        return rows

    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--sim", "--no-dream",
        "--protocol-split", "holdout",
        "--calibration-receipt", str(receipt),
        "--resume-from", str(checkpoint),
        "--results-dir", str(tmp_path / "results"),
    ])
    # A resume path must pre-exist; use --checkpoint for the first holdout run.
    sys.argv[sys.argv.index("--resume-from")] = "--checkpoint"
    locomo.main()
    assert seen == frozen["holdout_ids"]


@pytest.mark.parametrize("mode", ["--diag-only", "--rejudge"])
def test_non_scored_legacy_modes_reject_checkpoint_flags(
    monkeypatch, tmp_path, mode,
):
    argv = [
        "locomo_adapter.py", mode,
        str(tmp_path / "source.json") if mode == "--rejudge" else "",
        "--checkpoint", str(tmp_path / "run.json"),
    ]
    monkeypatch.setattr(sys, "argv", [item for item in argv if item])
    with pytest.raises(SystemExit) as caught:
        locomo.main()
    assert caught.value.code == 2


def test_strict_registry_accepts_adapter_archive(monkeypatch, tmp_path):
    archive, artifact = _strict_archive(monkeypatch, tmp_path)
    row = locomo_registry._locomo_row(artifact, archive)
    assert row["count"] == 1
    assert row["overall"] is None
    assert row["kind"] == "simulation"
    assert row["message_fts_top_k"] == 15
    assert row["rerank_top_k"] == 20
    assert row["fts_top_k"] == 10
    assert row["graph_top_k"] == 10
    assert row["facts"] is True
    assert row["facts_extraction"] is True
    assert row["rules_extraction"] is False
    assert json.loads(row["extras"])["strict"] is True


def test_strict_registry_accepts_zero_tier_ablation(monkeypatch, tmp_path):
    conv = _conversation("conv-one", "q1")
    checkpoint = tmp_path / "zero-tier.checkpoint.json"
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])

    def evaluate(conversation, _args, _answer, _judge, **kwargs):
        row = _row(conversation, conversation["qa"][0])
        kwargs["on_checkpoint"](row, _runtime(conversation))
        return [row]

    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    monkeypatch.setattr(
        sys, "argv",
        _argv(
            tmp_path, "--checkpoint", checkpoint,
            "--message-fts-top-k", "0",
        ),
    )
    locomo.main()
    archive = next((tmp_path / "results").glob("locomo-*strict-*.json"))
    artifact = json.loads(archive.read_text())
    assert artifact["config"]["effective_hymem_config"][
        "message_fts_top_k"
    ] == 0
    assert locomo_registry._locomo_row(artifact, archive)[
        "message_fts_top_k"
    ] == 0


@pytest.mark.parametrize("mutation", ["drift", "missing"])
def test_strict_registry_rejects_effective_extraction_contract_tamper(
    monkeypatch, tmp_path, mutation,
):
    archive, artifact = _strict_archive(monkeypatch, tmp_path)
    binding = artifact["config"]["effective_hymem_config"][
        "extraction_contract"
    ]
    if mutation == "missing":
        binding.pop("identity")
    else:
        binding["identity"] = (
            "hymem-extraction-contract-sha256-v1:" + "0" * 64
        )
    artifact["manifest"]["config"] = copy.deepcopy(artifact["config"])
    artifact["manifest"]["config_hash"] = content_hash(artifact["config"])
    _rehash_manifest(artifact)

    with pytest.raises(ValueError, match="extraction canary"):
        locomo_registry._locomo_row(artifact, archive)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda a: a["config"].__setitem__("top_k", 999), "top-level identity"),
        (lambda a: a.__setitem__("benchmark", "Forged"), "benchmark/version"),
        (lambda a: a.__setitem__("version", "strict-v999"), "benchmark/version"),
        (lambda a: a["manifest"].__setitem__("schema", "forged"), "schema"),
        (lambda a: a["manifest"].__setitem__("config_hash", "sha256:" + "0" * 64), "config hash"),
        (lambda a: a["manifest"].__setitem__("model_hash", "sha256:" + "0" * 64), "model hash"),
        (lambda a: a["execution"]["counts"].__setitem__("completed", 0), "counts"),
        (lambda a: a["per_question"][0].__setitem__("correct", "yes"), "verdict"),
        (lambda a: a.__setitem__("result_digest", "sha256:" + "0" * 64), "result digest"),
        (lambda a: a["scores"]["OVERALL"].__setitem__("accuracy", 0.0), "scores"),
        (lambda a: a.__setitem__("strict_accuracy", 0.0), "strict accuracy"),
        (lambda a: a["execution"]["checkpoint"].__setitem__("state_sha256", "bad"), "checkpoint digest"),
        (lambda a: a["per_question"][0].__setitem__("strict_failure", True), "failed row"),
        (lambda a: a["execution"]["segments"][0]["reader_usage"].update({"calls_available": True, "calls": -1}), "usage"),
        (lambda a: a["execution"]["segments"][0]["embedding_usage"].__setitem__("backend", "forged"), "embedding runtime identity"),
        (lambda a: a["execution"]["segments"].append(copy.deepcopy(a["execution"]["segments"][0])), "segment id"),
    ],
)
def test_strict_registry_rejects_tamper_classes(
    monkeypatch, tmp_path, mutation, message,
):
    archive, original = _strict_archive(monkeypatch, tmp_path)
    artifact = copy.deepcopy(original)
    mutation(artifact)
    if artifact["manifest"] != original["manifest"]:
        _rehash_manifest(artifact)
    with pytest.raises(ValueError, match=message):
        locomo_registry._locomo_row(artifact, archive)


def test_registry_rejects_latest_pointer(tmp_path):
    pointer = {
        "archive": "locomo-banked.json",
        "run_id": "sha256:" + "a" * 64,
        "artifact_digest": "sha256:" + "b" * 64,
    }
    with pytest.raises(ValueError, match="mutable pointer"):
        locomo_registry._locomo_row(pointer, tmp_path / "locomo-latest.json")


def test_registry_default_scan_excludes_latest_pointer(tmp_path):
    pointer = tmp_path / "locomo-latest.json"
    pointer.write_text(json.dumps({
        "archive": "locomo-banked.json",
        "run_id": "sha256:" + "a" * 64,
        "artifact_digest": "sha256:" + "b" * 64,
    }))
    db_path = tmp_path / "locomo-runs.db"
    spec = dict(locomo_registry.SPEC)
    spec["builder"] = locomo_registry._locomo_row
    locomo_registry.rr.cmd_ingest(
        spec, files=None, bench_dir=tmp_path, db_path=db_path,
    )
    con = locomo_registry.sqlite3.connect(db_path)
    try:
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    finally:
        con.close()
