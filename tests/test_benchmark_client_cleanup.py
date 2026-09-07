"""Provider-client ownership and teardown regressions for benchmark runners."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(_BENCH))

import longmemeval_adapter as lme  # noqa: E402
import beam_adapter as beam  # noqa: E402
import msc_adapter as msc  # noqa: E402
import locomo_adapter as locomo  # noqa: E402
from benchmarks.strictness import (
    BenchmarkIntegrityError,
    IndexingConvergenceError,
    OwnedResourceScope,
    publish_prepared_artifact_after_cleanup,
    run_cleanup_actions,
)
import hymem  # noqa: E402
from hymem import doctor as doctor_module  # noqa: E402
from hymem.deadline import DeadlineExceeded  # noqa: E402
from hymem.contrib import openai_client as openai_client_module  # noqa: E402
from hymem.contrib import openai_embedding_client as embedding_client_module  # noqa: E402
from hymem.extraction import embeddings as extraction_embeddings  # noqa: E402
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient


class _Closable:
    def __init__(self, name: str, events: list[str], *, fail: bool = False):
        self.name = name
        self.events = events
        self.fail = fail
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        self.events.append(f"close:{self.name}")
        if self.fail:
            raise RuntimeError(f"{self.name} close failed")


def _doctor_llm_config():
    return SimpleNamespace(
        llm_base_url="https://api.deepseek.com/v1",
        llm_model="deepseek-v4-flash",
        has_llm_key=True,
        llm_api_key="provider-key-must-not-leak",
    )


def test_doctor_closes_failed_llm_probe_without_masking_diagnostic(monkeypatch):
    events: list[str] = []

    class Probe(_Closable):
        def complete(self, _request):
            raise LookupError("provider-key-must-not-leak")

    probe = Probe("doctor-cleanup-secret", events, fail=True)
    monkeypatch.setattr(
        openai_client_module,
        "OpenAICompatibleClient",
        lambda **_kwargs: probe,
    )

    result = doctor_module._check_llm(_doctor_llm_config())

    assert result.status == doctor_module.FAIL
    assert result.detail.endswith("unreachable: LookupError")
    assert "provider-key-must-not-leak" not in result.detail
    assert "doctor-cleanup-secret" not in result.detail
    assert probe.close_calls == 1
    assert events == ["close:doctor-cleanup-secret"]


def test_doctor_cleanup_does_not_replace_control_flow_primary(monkeypatch):
    events: list[str] = []
    primary = KeyboardInterrupt("operator interrupt")

    class Probe(_Closable):
        def complete(self, _request):
            raise primary

    probe = Probe("doctor-cleanup-secret", events, fail=True)
    monkeypatch.setattr(
        openai_client_module,
        "OpenAICompatibleClient",
        lambda **_kwargs: probe,
    )

    with pytest.raises(KeyboardInterrupt, match="operator interrupt") as caught:
        doctor_module._check_llm(_doctor_llm_config())

    assert caught.value is primary
    assert probe.close_calls == 1
    assert events == ["close:doctor-cleanup-secret"]
    notes = "\n".join(primary.__notes__)
    assert "doctor-cleanup-secret" not in notes
    assert "RuntimeError" in notes


@pytest.mark.parametrize(
    "cleanup_control",
    [KeyboardInterrupt("cleanup interrupt"), SystemExit("cleanup exit")],
)
def test_doctor_reported_probe_failure_does_not_swallow_cleanup_control_flow(
    monkeypatch, cleanup_control,
):
    events: list[str] = []

    class Probe:
        def complete(self, _request):
            raise LookupError("ordinary probe failure")

        def close(self):
            events.append("close")
            raise cleanup_control

    monkeypatch.setattr(
        openai_client_module,
        "OpenAICompatibleClient",
        lambda **_kwargs: Probe(),
    )

    with pytest.raises(type(cleanup_control)) as caught:
        doctor_module._check_llm(_doctor_llm_config())

    assert caught.value is cleanup_control
    assert events == ["close"]


def test_owned_resource_scope_deduplicates_roles_and_reverses_construction_order():
    events: list[str] = []
    shared = _Closable("shared", events)
    pipeline = _Closable("pipeline", events)

    with OwnedResourceScope("test resources") as owned:
        owned.own(shared, label="reader")
        owned.own(shared, label="judge")
        owned.own(pipeline, label="pipeline")

    assert events == ["close:pipeline", "close:shared"]
    assert shared.close_calls == pipeline.close_calls == 1
    # Both the scope and the clients it manages tolerate defensive re-closes.
    assert owned.close() == ()
    assert shared.close_calls == pipeline.close_calls == 1


def test_cleanup_failure_is_fatal_only_without_a_primary_exception(capsys):
    clean_run_events: list[str] = []
    with pytest.raises(BenchmarkIntegrityError, match="cleanup failed") as clean:
        with OwnedResourceScope("clean run") as owned:
            owned.own(_Closable("reader", clean_run_events, fail=True))
    assert "reader close failed" not in str(clean.value)
    assert clean.value.cleanup_errors == ({
        "stage": "resource_close", "exception_type": "RuntimeError",
    },)

    primary_events: list[str] = []
    primary = RuntimeError("benchmark failed first")
    with pytest.raises(RuntimeError, match="benchmark failed first") as caught:
        with OwnedResourceScope("failed run") as owned:
            owned.own(_Closable("judge", primary_events, fail=True))
            raise primary
    assert caught.value is primary
    notes = "\n".join(caught.value.__notes__)
    assert "judge close failed" not in notes
    assert '"stage":"resource_close"' in notes
    warning = capsys.readouterr().err
    assert "judge close failed" not in warning
    assert '"exception_type":"RuntimeError"' in warning


def test_cleanup_control_flow_is_rethrown_after_all_actions_are_attempted():
    events: list[str] = []
    interrupt = KeyboardInterrupt("do not swallow")

    def first():
        events.append("first")
        raise interrupt

    def second():
        events.append("second")
        raise RuntimeError("secret second message")

    with pytest.raises(KeyboardInterrupt, match="do not swallow") as caught:
        run_cleanup_actions([
            ("adapter_close", first),
            ("temporary_store_cleanup", second),
        ])

    assert caught.value is interrupt
    assert events == ["first", "second"]
    notes = "\n".join(interrupt.__notes__)
    assert "secret second message" not in notes
    assert '"stage":"temporary_store_cleanup"' in notes


@pytest.mark.parametrize("failure_stage", ["resource_close", "checkpoint_close"])
def test_cleanup_failure_prevents_archive_publication(
    tmp_path: Path, failure_stage: str,
):
    events: list[str] = []
    archive = tmp_path / "must-not-exist.json"

    def cleanup(stage: str):
        def action():
            events.append(stage)
            if stage == failure_stage:
                raise RuntimeError(f"secret-{stage}-failure")
        return action

    with pytest.raises(BenchmarkIntegrityError, match="cleanup failed") as caught:
        publish_prepared_artifact_after_cleanup(
            archive,
            {"status": "success"},
            cleanup_actions=[
                ("resource_close", cleanup("resource_close")),
                ("checkpoint_close", cleanup("checkpoint_close")),
            ],
        )

    assert events == ["resource_close", "checkpoint_close"]
    assert not archive.exists()
    assert "secret-" not in str(caught.value)


@pytest.mark.parametrize(
    "client_type",
    [OpenAICompatibleClient, OpenAICompatibleEmbeddingClient],
)
def test_openai_sdk_clients_close_the_transport_exactly_once(client_type):
    events: list[str] = []
    transport = _Closable("sdk", events)
    client = object.__new__(client_type)
    client._client = transport
    client._close_lock = threading.Lock()
    if client_type is OpenAICompatibleEmbeddingClient:
        client._transport_lock = threading.RLock()
    else:
        client._close_condition = threading.Condition(client._close_lock)
        client._active_calls = 0
    client._closed = False

    client.close()
    client.close()

    assert events == ["close:sdk"]
    assert transport.close_calls == 1


@pytest.mark.parametrize(
    "client_type",
    [OpenAICompatibleClient, OpenAICompatibleEmbeddingClient],
)
def test_openai_client_rebind_never_redirects_owned_close_target(
    monkeypatch, client_type,
):
    import openai

    events: list[str] = []
    transports: list[_Closable] = []

    def http_factory(**_kwargs):
        transport = _Closable("original", events)
        transports.append(transport)
        return transport

    class Resource:
        def create(self, **_kwargs):
            raise AssertionError("provider should not be called")

    class SDK:
        def __init__(self, **kwargs):
            self._client = kwargs["http_client"]
            self.embeddings = Resource()
            self.chat = SimpleNamespace(completions=Resource())

        def close(self):
            self._client.close()

    monkeypatch.setattr(openai, "DefaultHttpxClient", http_factory)
    monkeypatch.setattr(openai, "OpenAI", SDK)
    if client_type is OpenAICompatibleEmbeddingClient:
        client = client_type(
            api_key="wire-key", pin_dimension=True,
            deployment_revision="public-release", deployment_tenant="public-tenant",
        )
    else:
        client = client_type(api_key="wire-key")
    replacement = _Closable("replacement", events)
    client._client = replacement

    client.close()
    client.close()

    assert transports[0].close_calls == 1
    assert replacement.close_calls == 0
    assert events == ["close:original"]


@pytest.mark.parametrize(
    ("module", "entrypoint"),
    [
        (lme, "_main"),
        (beam, "_main"),
        (msc, "main"),
        (locomo, "main"),
    ],
)
def test_shared_clients_outlive_workers_and_usage_snapshot_then_close_once(
    monkeypatch, module, entrypoint,
):
    events: list[str] = []
    worker_count = 4

    class Shared(_Closable):
        call_count = 4
        request_attempts = 4
        successful_responses = 4
        prompt_tokens = 8
        completion_tokens = 4
        total_tokens = 12
        total_latency_s = 0.25
        cost_usd = None
        token_usage_available = True

        def close(self):
            assert events.count("worker_done") == worker_count
            assert events[-1] == "usage_snapshot"
            super().close()

    shared = Shared("reader+judge", events)

    def run_main(*args):
        owned = args[-1]
        owned.own(shared, label="reader")
        owned.own(shared, label="judge")
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            list(pool.map(lambda _index: events.append("worker_done"), range(worker_count)))
        assert shared.close_calls == 0
        assert lme.usage_snapshot(shared)["calls"] == worker_count
        events.append("usage_snapshot")
        return "complete"

    monkeypatch.setattr(module, "_run_main", run_main)
    assert getattr(module, entrypoint)() == "complete"
    assert events == [
        *("worker_done" for _ in range(worker_count)),
        "usage_snapshot",
        "close:reader+judge",
    ]
    assert shared.close_calls == 1


def test_runner_cleanup_failure_does_not_replace_primary_benchmark_error(
    monkeypatch, capsys,
):
    events: list[str] = []
    primary = LookupError("scoring failed")

    def run_main(_ledgers, owned):
        owned.own(_Closable("reader", events, fail=True), label="reader")
        raise primary

    monkeypatch.setattr(lme, "_run_main", run_main)
    with pytest.raises(LookupError, match="scoring failed") as caught:
        lme._main()

    assert caught.value is primary
    assert events == ["close:reader"]
    notes = "\n".join(primary.__notes__)
    assert "reader close failed" not in notes
    assert '"stage":"resource_close"' in notes
    warning = capsys.readouterr().err
    assert "reader close failed" not in warning
    assert '"exception_type":"RuntimeError"' in warning


@pytest.mark.parametrize("module", [msc, locomo])
def test_shared_client_cleanup_is_required_before_result_file_publication(
    monkeypatch, tmp_path: Path, module,
):
    events: list[str] = []
    transport = _Closable("secret-provider", events, fail=True)
    owned = OwnedResourceScope("runner providers")
    owned.own(transport, label="provider")
    destination = tmp_path / f"{module.__name__}.json"
    monkeypatch.setattr(sys, "argv", [
        f"{module.__name__}.py",
        "--sim",
        "--no-dream",
        "--sample", "1",
        "--out", str(destination),
    ])

    with pytest.raises(BenchmarkIntegrityError, match="cleanup failed") as caught:
        module._run_main(owned)

    assert transport.close_calls == 1
    assert events == ["close:secret-provider"]
    assert not destination.exists()
    assert "secret-provider" not in str(caught.value)


def test_lme_rejudge_closes_judge_before_immutable_publication(
    monkeypatch, tmp_path: Path,
):
    source = tmp_path / "lme-source.json"
    source.write_text(json.dumps({
        "config": {"judge_model": "old"},
        "per_question": [{
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "q",
            "answer": "a",
            "hypothesis": "",
            "correct": False,
        }],
    }))
    events: list[str] = []

    class Judge(_Closable):
        call_count = 0
        total_tokens = 0

    judge = Judge("lme-rejudge-secret", events, fail=True)
    monkeypatch.setattr(lme, "LLMClient", lambda *_args, **_kwargs: judge)
    args = SimpleNamespace(
        rejudge=str(source),
        judge_model="judge-v1",
        judge_base_url="https://example.test/v1",
        judge_extra_body_obj=None,
        judge_protocol="legacy-custom",
        workers=1,
        extra_body_defaulted=[],
    )
    owned = OwnedResourceScope("LME rejudge")

    with pytest.raises(BenchmarkIntegrityError, match="cleanup failed") as caught:
        lme._rejudge_run_impl(args, "key", owned)

    assert judge.close_calls == 1
    assert not list(tmp_path.glob("lme-source-rejudged-*.json"))
    assert "lme-rejudge-secret" not in str(caught.value)


def test_beam_rejudge_closes_judge_before_immutable_publication(
    monkeypatch, tmp_path: Path,
):
    source = tmp_path / "beam-source.json"
    source.write_text(json.dumps({
        "metadata": {
            "judge_model": "old", "date": "2026-09-05T10:00:00+00:00",
        },
        "conversations": [{
            "conv_id": "c1", "scale": "100K", "questions": [{
                "question_id": "q1", "ability": "IE", "question": "q",
                "answer": "reader answer", "ideal_answer": "gold",
                "rubric": ["criterion"], "score": 0.0, "scores": [0.0],
            }],
        }],
    }))
    events: list[str] = []

    class Judge(_Closable):
        call_count = 0

        def chat(self, *_args, **_kwargs):
            self.call_count += 1
            return "canary content"

    judge = Judge("beam-rejudge-secret", events, fail=True)
    monkeypatch.setattr(beam, "LLMClient", lambda *_args, **_kwargs: judge)
    monkeypatch.setattr(beam, "check_model_pin", lambda *_args: None)
    monkeypatch.setattr(
        beam, "resolve_dataset_revisions", lambda *_args: {beam.BEAM_REPO: "a" * 40}
    )
    monkeypatch.setattr(
        beam, "load_beam_conversations", lambda *_args, **_kwargs: {"100K": []}
    )
    gold = {"IE": {"q": {"gold_text": "gold"}}}
    monkeypatch.setattr(beam, "_rejudge_gold_map", lambda *_args, **_kwargs: (gold, gold))
    monkeypatch.setattr(beam, "judge_answer", lambda *_args, **_kwargs: {
        "score": 1.0,
        "scores": [1.0],
        "judge_raw": "valid",
        "judge_finish_reason": "stop",
        "judge_parse": "ok",
    })
    args = SimpleNamespace(
        rejudge=str(source),
        judge_model="judge-v1",
        judge_gold=True,
        judge_extra_body_obj={},
        dataset_revision=None,
        prereg_obj=None,
        extra_body_defaulted=[],
    )
    owned = OwnedResourceScope("BEAM rejudge")

    with pytest.raises(BenchmarkIntegrityError, match="cleanup failed") as caught:
        beam._rejudge_run_impl(args, "key", owned)

    assert judge.close_calls == 1
    assert not list(tmp_path.glob("beam-source-rejudged-*.json"))
    assert "beam-rejudge-secret" not in str(caught.value)


@pytest.mark.parametrize("module,adapter_type", [
    (lme, lme.HyMemAdapter),
    (beam, beam.HyMemAdapter),
])
def test_dream_cleanup_preserves_primary_convergence_and_attempts_invalidation(
    monkeypatch, module, adapter_type,
):
    events: list[str] = []

    class Fork:
        def dream(self):
            raise AssertionError("convergence stub should own the call")

        def close(self):
            events.append("fork_close")
            raise RuntimeError("api_key=cleanup-secret")

    class Parent:
        def fork(self):
            return Fork()

        def invalidate_query_caches(self):
            events.append("cache_invalidation")

    summary = {
        "cycles": 0,
        "cleanup_errors": [],
        "failure_reason": "timeout_during_cycle",
    }
    primary = IndexingConvergenceError("deadline convergence failed", summary)

    def fail_convergence(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(module, "converge_indexing", fail_convergence)
    if module is lme:
        monkeypatch.setattr(
            lme,
            "canonicalize_lme_indexing_summary",
            lambda value: dict(value),
        )
    adapter = object.__new__(adapter_type)
    adapter.hy = Parent()
    adapter.embedding_client = None
    adapter.last_indexing_summary = None

    with pytest.raises(
        IndexingConvergenceError, match="deadline convergence failed"
    ) as caught:
        adapter.dream_and_wait(timeout=1.0, max_cycles=1)

    assert caught.value is primary
    assert caught.value.summary["failure_reason"] == "timeout_during_cycle"
    assert caught.value.summary["cleanup_errors"] == [{
        "stage": "dream_fork_close", "exception_type": "RuntimeError",
    }]
    assert events == ["fork_close", "cache_invalidation"]
    evidence = json.dumps(caught.value.summary) + "\n" + "\n".join(
        getattr(caught.value, "__notes__", [])
    )
    assert "cleanup-secret" not in evidence


def test_deadline_converted_by_convergence_remains_primary_during_cleanup():
    events: list[str] = []

    class Fork:
        def dream(self, *, deadline=None):
            assert deadline is not None
            raise DeadlineExceeded("provider deadline detail")

        def close(self):
            events.append("fork_close")
            raise RuntimeError("cleanup-secret")

    class Parent:
        def fork(self):
            return Fork()

        def invalidate_query_caches(self):
            events.append("cache_invalidation")

    adapter = object.__new__(beam.HyMemAdapter)
    adapter.hy = Parent()
    adapter.embedding_client = None
    adapter.last_indexing_summary = None

    with pytest.raises(
        IndexingConvergenceError,
        match="memory indexing exceeded its deadline",
    ) as caught:
        adapter.dream_and_wait(timeout=1.0, max_cycles=1)

    assert caught.value.summary["failure_reason"] == "timeout_during_cycle"
    assert caught.value.summary["cleanup_errors"] == [{
        "stage": "dream_fork_close", "exception_type": "RuntimeError",
    }]
    assert events == ["fork_close", "cache_invalidation"]
    evidence = json.dumps(caught.value.summary) + "\n" + "\n".join(
        caught.value.__notes__
    )
    assert "provider deadline detail" not in evidence
    assert "cleanup-secret" not in evidence


@pytest.mark.parametrize("module,adapter_type", [
    (lme, lme.HyMemAdapter),
    (beam, beam.HyMemAdapter),
])
def test_successful_dream_is_fatal_when_cache_invalidation_fails(
    monkeypatch, module, adapter_type,
):
    events: list[str] = []

    class Fork:
        def dream(self):
            raise AssertionError("convergence stub should own the call")

        def close(self):
            events.append("fork_close")

    class Parent:
        def fork(self):
            return Fork()

        def invalidate_query_caches(self):
            events.append("cache_invalidation")
            raise RuntimeError("secret invalidation failure")

    monkeypatch.setattr(
        module,
        "converge_indexing",
        lambda *_args, **_kwargs: {"cycles": 1, "cleanup_errors": []},
    )
    if module is lme:
        monkeypatch.setattr(
            lme,
            "canonicalize_lme_indexing_summary",
            lambda value: {"outcome": "success", **dict(value)},
        )
    adapter = object.__new__(adapter_type)
    adapter.hy = Parent()
    adapter.embedding_client = None
    adapter.last_indexing_summary = None

    with pytest.raises(BenchmarkIntegrityError, match="cleanup failed") as caught:
        adapter.dream_and_wait(timeout=1.0, max_cycles=1)

    assert events == ["fork_close", "cache_invalidation"]
    assert adapter.last_indexing_summary["cleanup_errors"] == [{
        "stage": "query_cache_invalidation",
        "exception_type": "RuntimeError",
    }]
    assert "secret invalidation failure" not in str(caught.value)


def test_benchmark_pinned_embedding_wrapper_closes_only_explicit_owned_transport():
    events: list[str] = []
    inner = type("Inner", (), {"model": "embed", "dim": 3})()
    transport = _Closable("embedding", events)
    wrapper = beam.BenchmarkPinnedEmbeddingClient(
        inner,
        expected_dimension=3,
        owned_transport=transport,
    )

    wrapper.close()
    wrapper.close()

    assert transport.close_calls == 1
    assert events == ["close:embedding"]


def test_benchmark_pinned_embedding_transport_rebind_revokes_identity_but_closes_owner():
    from hymem.dreaming.aggregation_material import embedding_producer_binding
    from hymem.extraction.embeddings import LocalHashEmbeddingClient

    events: list[str] = []
    original = _Closable("original", events)
    replacement = _Closable("replacement", events)
    inner = LocalHashEmbeddingClient(dim_value=8, model_name="local-test")
    wrapper = beam.BenchmarkPinnedEmbeddingClient(
        inner, expected_dimension=8, owned_transport=original,
    )
    assert embedding_producer_binding(wrapper)["identity_exact"] is True

    wrapper._owned_transport = replacement
    assert embedding_producer_binding(wrapper)["identity_exact"] is False
    wrapper.close()

    assert original.close_calls == 1
    assert replacement.close_calls == 0


def test_benchmark_pinned_embedding_wrapper_never_replays_after_close():
    class Inner:
        model = "embed"
        dim = 3

        def __init__(self):
            self.calls = 0

        def embed(self, _texts):
            self.calls += 1
            return [[1.0, 0.0, 0.0]]

    inner = Inner()
    wrapper = beam.BenchmarkPinnedEmbeddingClient(
        inner, expected_dimension=3,
    )
    assert wrapper.embed(["before"]) == [[1.0, 0.0, 0.0]]
    wrapper.close()

    with pytest.raises(BenchmarkIntegrityError, match="client is closed"):
        wrapper.embed(["after"])
    assert inner.calls == 1


def test_beam_embedding_transport_closes_if_wrapper_construction_fails(
    monkeypatch,
):
    events: list[str] = []
    transport = _Closable("embedding-construction-secret", events)
    primary = KeyboardInterrupt("wrapper construction interrupted")

    monkeypatch.setattr(
        embedding_client_module,
        "OpenAICompatibleEmbeddingClient",
        lambda **_kwargs: transport,
    )

    def interrupt_wrapper(_transport):
        raise primary

    monkeypatch.setattr(
        extraction_embeddings, "CachedEmbeddingClient", interrupt_wrapper,
    )
    config = {
        "backend": "openai_compatible",
        "request_base_url": "https://embedding.example.test/v1",
        "request_model": "embedding-v1",
        "dimension": 3,
        "deployment_revision": "public-release-1",
        "deployment_tenant": "public-tenant-1",
    }

    with pytest.raises(
        KeyboardInterrupt, match="wrapper construction interrupted"
    ) as caught:
        beam.build_embedding_client(config, api_key="secret")

    assert caught.value is primary
    assert transport.close_calls == 1
    assert events == ["close:embedding-construction-secret"]


def test_lme_question_snapshots_pipeline_and_embedding_before_adapter_close(
    monkeypatch,
):
    events: list[str] = []

    class Adapter:
        pipeline_llm = object()
        embedding_client = object()
        last_indexing_summary = None

        def open(self):
            return self

        def close(self):
            assert events == ["pipeline_snapshot", "embedding_snapshot"]
            events.append("adapter_close")

    monkeypatch.setattr(lme, "_adapter_for_args", lambda *_args: Adapter())
    monkeypatch.setattr(
        lme,
        "evaluate_question",
        lambda *_args, **_kwargs: {
            "question_id": "qid",
            "question_type": "multi-session",
            "correct": True,
        },
    )
    monkeypatch.setattr(
        lme, "usage_snapshot",
        lambda _client: events.append("pipeline_snapshot") or {},
    )
    monkeypatch.setattr(
        lme, "embedding_usage_snapshot",
        lambda *_args, **_kwargs: events.append("embedding_snapshot") or {},
    )
    args = SimpleNamespace(
        keep_db=False,
        embeddings=True,
        top_k=5,
        auto_ability=True,
        no_dream=True,
        graph_facts_first=False,
        permissive_default=False,
        distill=False,
        distill_prompt_version=lme.DEFAULT_DISTILL_PROMPT_VERSION,
        retrieval_only=False,
        max_input_tokens=16000,
        max_input_bytes=64000,
        token_counter=None,
        judge_protocol="legacy-custom",
        indexing_max_cycles=1,
        indexing_timeout_s=1.0,
        indexing_require_healthy=True,
    )

    row = lme._evaluate_one_question(
        0,
        1,
        {"question_id": "qid", "question_type": "multi-session"},
        args,
        object(),
        object(),
        "key",
    )

    assert row["correct"] is True
    assert events == [
        "pipeline_snapshot", "embedding_snapshot", "adapter_close",
    ]


@pytest.mark.parametrize("adapter_kind", ["lme", "beam", "msc"])
def test_memory_adapters_close_owned_store_embedding_and_pipeline_once(
    monkeypatch, tmp_path, adapter_kind,
):
    events: list[str] = []
    pipeline = _Closable("pipeline", events)
    embedding = _Closable("embedding", events)

    class Store(_Closable):
        def __init__(self, *_args, **_kwargs):
            super().__init__("store", events)

    monkeypatch.setattr(
        openai_client_module,
        "OpenAICompatibleClient",
        lambda **_kwargs: pipeline,
    )
    monkeypatch.setattr(
        embedding_client_module,
        "OpenAICompatibleEmbeddingClient",
        lambda **_kwargs: embedding,
    )
    monkeypatch.setattr(hymem, "HyMem", Store)
    monkeypatch.setattr(
        beam, "build_embedding_client", lambda *_args, **_kwargs: embedding
    )

    if adapter_kind == "lme":
        adapter = lme.HyMemAdapter(
            tmp_path / "lme.sqlite", api_key="key", embeddings=True,
        )
    elif adapter_kind == "beam":
        adapter = beam.HyMemAdapter(
            tmp_path / "beam.sqlite", api_key="key",
            embedding_backend="none",
        )
    else:
        adapter = msc.MSCAdapter(
            tmp_path / "hymem.sqlite", api_key="key", embeddings=True,
        )

    adapter.open()
    adapter.close()
    adapter.close()

    assert events == ["close:store", "close:embedding", "close:pipeline"]
    assert (pipeline.close_calls, embedding.close_calls) == (1, 1)
