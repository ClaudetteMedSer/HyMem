"""Regression coverage for MSC/LoCoMo's bounded indexing contract."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import strictness
from benchmarks import locomo_adapter as locomo
from benchmarks import msc_adapter as msc
from benchmarks.strictness import (
    BenchmarkCleanupError,
    BenchmarkIntegrityError,
    IndexingConvergenceError,
)
from benchmarks.msc_adapter import (
    MSCAdapter,
    prepare_indexing,
    run_or_record_indexing_failure,
)
from hymem import HyMem, HyMemConfig
from hymem.dreaming.status import DREAM_STATUS_SCHEMA_VERSION
from hymem.dreaming.runner import DreamReport
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.producer import Phase1ProducerDeclaration


class _ScalarConn:
    def __init__(self, owner):
        self.owner = owner

    def create_function(self, *_args, **_kwargs):
        pass

    def execute(self, sql, _params=()):
        assert "facts_quarantined" in sql
        value = (
            0 if "<>1" in sql else self.owner.facts_quarantined
        )

        class _Cursor:
            @staticmethod
            def fetchone():
                return (value,)

        return _Cursor()


class _DreamHandle:
    def __init__(self, reports, statuses, *, facts_quarantined=0):
        self.reports = list(reports)
        self.statuses = list(statuses)
        self.cycles = 0
        self.facts_quarantined = facts_quarantined
        self.config = HyMemConfig(root=Path("/benchmark-test"))
        self.read_conn = _ScalarConn(self)
        self.closed = False

    def dream(self):
        report = asdict(DreamReport())
        report.update(self.reports[self.cycles])
        self.cycles += 1
        return report

    def dream_status(self):
        index = max(self.cycles - 1, 0)
        status = dict(self.statuses[min(index, len(self.statuses) - 1)])
        if self.facts_quarantined:
            status["quarantined_facts"] = self.facts_quarantined
        status.setdefault("quarantined_facts_malformed", 0)
        return status

    def close(self):
        self.closed = True


class _Parent:
    def __init__(self, handle):
        self.handle = handle
        self.invalidations = 0

    def fork(self):
        return self.handle

    def invalidate_query_caches(self):
        self.invalidations += 1

    @property
    def read_conn(self):
        return self.handle.read_conn

    @property
    def config(self):
        return self.handle.config

    def dream_status(self):
        return self.handle.dream_status()


class _Usage:
    call_count = 7
    request_attempts = 9
    successful_responses = 7
    prompt_tokens = 70
    completion_tokens = 20
    total_tokens = 90
    total_latency_s = 1.25
    cost_usd = 0.125
    token_usage_available = True


class _ReceiptPipeline:
    def __init__(self):
        self.model = "pipeline-model"
        self.base_url = "https://memory.example/v1"
        self.thinking = "disabled"
        self.effective_extra_body = {"temperature": 0}
        self.api_key = ""

    def phase1_producer_declaration(self):
        return Phase1ProducerDeclaration(
            client_id="tests.msc.ReceiptPipeline",
            implementation="tests.msc.receipt-pipeline-v1",
            model=self.model,
            endpoint=self.base_url,
            effective_request={
                "thinking": self.thinking,
                "effective_body": self.effective_extra_body,
                "temperature": 0,
            },
            retry_policy={"owner": "test", "attempts": 1},
        )


def _adapter(reports, statuses, *, facts_quarantined=0):
    handle = _DreamHandle(
        reports, statuses, facts_quarantined=facts_quarantined
    )
    adapter = object.__new__(MSCAdapter)
    adapter.hy = _Parent(handle)
    adapter.pipeline_llm = _Usage()
    adapter.last_indexing_summary = None
    adapter.indexing_runs = []
    adapter.indexing_skip = None
    adapter.validate_store_build_receipt = lambda _item: {
        "status": "complete_healthy",
        "identity_sha256": "sha256:fake",
        "material_state": {"sha256": "sha256:state"},
    }
    return adapter, handle


def _status(**overrides):
    value = {
        "dream_status_schema": DREAM_STATUS_SCHEMA_VERSION,
        "pending_source_materialization": 0,
        "pending_chunks": 0,
        "pending_digests": 0,
        "pending_profiles": 0,
        "pending_facts": 0,
        "pending_aggregation": 0,
        "quarantined_chunks": 0,
        "quarantined_digests": 0,
        "quarantined_profiles": 0,
        "quarantined_facts": 0,
        "quarantined_facts_malformed": 0,
        "terminal_loss_chunks": 0,
        "terminal_loss_reasons": {},
        "coverage_integrity_failures": 0,
        "coverage_integrity_failure_reasons": {},
        "coverage_integrity_failure_details": [],
        "coverage_integrity_failure_details_truncated": False,
        "coverage_integrity_config_version": (
            msc.COVERAGE_INTEGRITY_CONFIG_VERSION
        ),
        "malformed_source_materialization": 0,
        "malformed_digests": 0,
        "malformed_profiles": 0,
        "malformed_facts": 0,
        "phase1_backlog_status": "current_producer",
        "pending_chunks_authoritative": True,
        "phase1_generation_key": "hymem-phase1-generation-v1:test",
        "aggregation_generation_key": None,
        "aggregation_publication_generation_key": None,
        "aggregation_last_success_generation_key": None,
        "aggregation_material_epoch_key": None,
        "aggregation_last_success_material_epoch_key": None,
        "aggregation_enabled": False,
        "aggregation_publication_generation": None,
        "aggregation_material_binding": None,
        "aggregation_material_revision": None,
        "in_progress": False,
    }
    value.update(overrides)
    return value


def test_durable_status_preserves_native_fact_quarantine_authority():
    handle = _DreamHandle(
        [{"budget_exhausted": False}],
        [_status(quarantined_facts=1)],
        facts_quarantined=0,
    )

    status = strictness.durable_indexing_status(handle, None)

    assert status["quarantined_facts"] == 1
    assert status["quarantined_facts_malformed"] == 0


def test_durable_status_rejects_unavailable_phase1_producer_authority():
    handle = _DreamHandle(
        [{"budget_exhausted": False}],
        [_status(
            phase1_backlog_status="producer_unavailable",
            pending_chunks_authoritative=False,
            phase1_generation_key=None,
        )],
    )

    with pytest.raises(
        BenchmarkIntegrityError, match="exact Phase-1 producer"
    ):
        strictness.durable_indexing_status(handle, None)


def test_exact_one_chunk_budget_drain_converges_with_max_cycles_one(tmp_path):
    llm = StubLLMClient(
        fixtures={
            "Return the JSON object now": json.dumps({
                "episodes": [],
                "summary": "Indexed the exact memory completely.",
                "procedures": [],
            }),
        },
        default=json.dumps({
            "triples": [], "markers": [], "complete": True,
        }),
    )
    hy = HyMem(
        HyMemConfig(
            root=tmp_path,
            dream_budget=1,
            dream_baseline_budget=1,
            salience_min_chars=1,
            profile_extraction_enabled=False,
            facts_extraction_enabled=False,
            rules_extraction_enabled=False,
            aggregation_nodes_enabled=False,
        ),
        llm=llm,
    )
    try:
        hy.open_session("exact-drain")
        hy.log_message(
            "exact-drain", "user", "One actionable memory to index exactly."
        )
        hy.close_session("exact-drain")

        summary = strictness.converge_indexing(
            hy.dream,
            status=hy.dream_status,
            max_cycles=1,
            timeout_s=10,
        )

        assert summary["complete"] is True
        assert summary["cycles"] == 1
        assert summary["reports"][0]["budget_exhausted"] is False
        assert summary["final_status"]["pending_chunks"] == 0
    finally:
        hy.close()


def test_nonexhausted_report_does_not_override_durable_pending_work():
    adapter, handle = _adapter(
        [
            {"budget_exhausted": False, "chunk_extraction_provider_attempts": 3},
            {"budget_exhausted": False, "chunk_extraction_provider_attempts": 2},
        ],
        [_status(pending_chunks=1), _status()],
    )

    summary = adapter.dream(max_cycles=3, timeout_s=10)
    receipt = adapter.indexing_provenance(scope_id="msc:one")

    assert summary["cycles"] == handle.cycles == 2
    assert receipt["complete"] is True and receipt["healthy"] is True
    assert receipt["dream_report_totals"][
        "chunk_extraction_provider_attempts"
    ] == 5
    assert receipt["pipeline_usage"]["request_attempts"] == 9
    assert adapter.hy.invalidations == 1 and handle.closed is True


def test_msc_dream_cleanup_preserves_convergence_primary_and_safe_evidence(
    monkeypatch,
):
    adapter, handle = _adapter(
        [{"budget_exhausted": False}], [_status()]
    )
    events = []

    def fail_close():
        events.append("fork_close")
        raise RuntimeError("Bearer cleanup-secret")

    def invalidate():
        events.append("cache_invalidation")

    handle.close = fail_close
    adapter.hy.invalidate_query_caches = invalidate
    primary = IndexingConvergenceError(
        "indexing deadline exceeded",
        {"failure_reason": "timeout_during_cycle", "cycles": 0},
    )
    monkeypatch.setattr(
        msc,
        "converge_indexing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(
        IndexingConvergenceError, match="indexing deadline exceeded"
    ) as caught:
        adapter.dream(max_cycles=2, timeout_s=1.0)

    assert caught.value.summary["failure_reason"] == "timeout_during_cycle"
    assert caught.value.summary["cleanup_errors"] == [{
        "stage": "dream_fork_close", "exception_type": "RuntimeError",
    }]
    assert events == ["fork_close", "cache_invalidation"]
    assert "cleanup-secret" not in json.dumps(caught.value.summary)


def test_msc_success_with_both_cleanup_failures_is_fatal_and_attempts_both(
    monkeypatch,
):
    adapter, handle = _adapter(
        [{"budget_exhausted": False}], [_status()]
    )
    events = []

    def fail_close():
        events.append("fork_close")
        raise RuntimeError("secret close")

    def fail_invalidation():
        events.append("cache_invalidation")
        raise ValueError("secret invalidation")

    handle.close = fail_close
    adapter.hy.invalidate_query_caches = fail_invalidation

    with pytest.raises(BenchmarkIntegrityError, match="cleanup failed") as caught:
        adapter.dream(max_cycles=2, timeout_s=10.0)

    assert events == ["fork_close", "cache_invalidation"]
    assert caught.value.cleanup_errors == (
        {"stage": "dream_fork_close", "exception_type": "RuntimeError"},
        {"stage": "query_cache_invalidation", "exception_type": "ValueError"},
    )
    assert "secret" not in str(caught.value)


def test_budgeted_digest_and_aggregation_work_forces_another_cycle():
    adapter, _handle = _adapter(
        [
            {
                "budget_exhausted": True,
                "aggregation_nodes_built": 2,
                "episodes_created": 3,
            },
            {
                "budget_exhausted": False,
                "aggregation_nodes_built": 1,
                "episodes_created": 1,
            },
        ],
        [_status(), _status()],
    )

    adapter.dream(max_cycles=3, timeout_s=10)
    receipt = adapter.indexing_provenance(scope_id="msc:digest")

    assert receipt["cycles"] == 2
    assert receipt["budget_exhausted_cycles"] == 1
    assert receipt["dream_report_totals"]["aggregation_nodes_built"] == 3
    assert receipt["dream_report_totals"]["episodes_created"] == 4


@pytest.mark.parametrize(
    ("status", "facts_quarantined", "reason"),
    [
        (_status(quarantined_chunks=1), 0, "quarantined_extraction"),
        (_status(terminal_loss_chunks=1), 0, "terminal_extraction_source_loss"),
        (_status(coverage_integrity_failures=1), 0, "coverage_integrity_failure"),
        (_status(), 1, "quarantined_extraction"),
    ],
)
def test_unhealthy_durable_state_fails_before_scoring(
    status, facts_quarantined, reason
):
    reports = [{"budget_exhausted": False}]
    if reason == "coverage_integrity_failure":
        # Coverage is locally repairable, so strict convergence consumes its
        # finite retry bound before returning the durable health failure.
        reports.append({"budget_exhausted": False})
    adapter, _handle = _adapter(
        reports, [status],
        facts_quarantined=facts_quarantined,
    )

    with pytest.raises(IndexingConvergenceError) as failed:
        prepare_indexing(
            adapter,
            {},
            SimpleNamespace(
                sim=False,
                no_dream=False,
                dream_per_session=False,
                indexing_max_cycles=2,
                indexing_timeout_s=10,
            ),
            scope_id="locomo:bad",
            reuse=True,
        )

    assert failed.value.summary["status"] == "failed_before_scoring"
    assert failed.value.summary["scope_id"] == "locomo:bad"
    assert failed.value.summary["failure_reason"] == reason
    assert failed.value.summary["pipeline_usage"]["request_attempts"] == 9


def test_cycle_cap_and_timeout_retain_loud_failure_metadata(monkeypatch):
    capped, _handle = _adapter(
        [{"budget_exhausted": False}, {"budget_exhausted": False}],
        [_status(pending_chunks=1), _status(pending_chunks=1)],
    )
    args = SimpleNamespace(
        sim=False,
        no_dream=False,
        dream_per_session=False,
        indexing_max_cycles=2,
        indexing_timeout_s=10,
    )
    with pytest.raises(IndexingConvergenceError) as cap:
        prepare_indexing(capped, {}, args, scope_id="msc:cap", reuse=True)
    assert cap.value.summary["failure_reason"] == "max_cycles_exhausted"
    assert cap.value.summary["cycles"] == 2

    timed, _handle = _adapter(
        [{"budget_exhausted": False}], [_status()]
    )
    ticks = iter((0.0, 0.0, 2.0, 2.0))
    monkeypatch.setattr(strictness.time, "monotonic", lambda: next(ticks))
    args.indexing_timeout_s = 1
    with pytest.raises(IndexingConvergenceError) as timeout:
        prepare_indexing(timed, {}, args, scope_id="msc:timeout", reuse=True)
    assert timeout.value.summary["failure_reason"] == "timeout_after_cycle"
    assert timeout.value.summary["status"] == "failed_before_scoring"


@pytest.mark.parametrize("reason", ["no_dream", "simulation"])
def test_explicit_skip_is_noncomparable_zero_cost_and_does_not_dream(reason):
    adapter, handle = _adapter(
        [{"budget_exhausted": False}], [_status(pending_chunks=4)]
    )
    receipt = prepare_indexing(
        adapter,
        {"id": "skip", "sessions": [], "session_dates": []},
        SimpleNamespace(
            sim=reason == "simulation",
            no_dream=reason == "no_dream",
            dream_per_session=True,
            indexing_max_cycles=7,
            indexing_timeout_s=12,
        ),
        scope_id="locomo:skip",
        reuse=False,
    )

    assert handle.cycles == 0
    assert receipt["mode"] == "skipped_non_comparable"
    assert receipt["skip_reason"] == reason
    assert receipt["complete"] is False and receipt["healthy"] is False
    assert receipt["settings_applied"] is False
    assert receipt["pipeline_usage"]["request_attempts"] == 0


def test_skipped_indexing_refuses_a_reused_possibly_dreamed_store():
    adapter, handle = _adapter(
        [{"budget_exhausted": False}], [_status()]
    )
    with pytest.raises(IndexingConvergenceError) as failed:
        prepare_indexing(
            adapter,
            {},
            SimpleNamespace(
                sim=False,
                no_dream=True,
                dream_per_session=False,
                indexing_max_cycles=2,
                indexing_timeout_s=10,
            ),
            scope_id="locomo:skip-reuse",
            reuse=True,
        )
    assert handle.cycles == 0
    assert failed.value.summary["failure_reason"] == (
        "skipped_indexing_reused_store"
    )
    assert "--fresh" in failed.value.summary["remediation"]


def test_reused_store_is_always_validated_and_converged():
    class Adapter:
        pipeline_llm = _Usage()

        def __init__(self):
            self.calls = []
            self.validations = 0

        def ingest(self, *_args, **_kwargs):
            raise AssertionError("a reused store must not be ingested twice")

        def dream(self, **kwargs):
            self.calls.append(kwargs)

        def validate_store_build_receipt(self, _item):
            self.validations += 1
            return {
                "status": "complete_healthy",
                "identity_sha256": "sha256:fake",
                "indexing_sha256": "sha256:indexing",
                "material_state": {"sha256": "sha256:state"},
            }

        def indexing_provenance(self, *, scope_id):
            return _healthy_indexing(scope_id)

    adapter = Adapter()
    receipt = prepare_indexing(
        adapter,
        {"id": "reuse"},
        SimpleNamespace(
            sim=False,
            no_dream=False,
            dream_per_session=False,
            indexing_max_cycles=11,
            indexing_timeout_s=22,
        ),
        scope_id="locomo:reuse",
        reuse=True,
    )

    assert receipt["scope_id"] == "locomo:reuse"
    assert adapter.calls == [{
        "max_cycles": 11,
        "timeout_s": 22.0,
        "trigger": "reused_store_validation",
    }]
    assert adapter.validations == 2
    assert receipt["store_build_receipt"]["material_state_sha256"] == (
        "sha256:state"
    )


def test_dream_per_session_means_full_bounded_wave_per_session():
    class Hy:
        def __init__(self):
            self.logged = []

        def log_messages(self, session_id, turns):
            self.logged.append((session_id, turns))

    adapter = object.__new__(MSCAdapter)
    adapter.hy = Hy()
    calls = []
    adapter.dream = lambda **kwargs: calls.append(kwargs)
    item = {
        "id": "live",
        "sessions": [
            [{"role": "user", "content": "one"}],
            [{"role": "assistant", "content": "two"}],
        ],
        "session_dates": ["2025-01-01", "2025-01-02"],
    }

    adapter.ingest(
        item,
        dream_each=True,
        indexing_max_cycles=13,
        indexing_timeout_s=17,
    )

    assert [call["trigger"] for call in calls] == [
        "after_session:0", "after_session:1"
    ]
    assert all(call["max_cycles"] == 13 for call in calls)
    assert all(call["timeout_s"] == 17 for call in calls)


def test_locomo_persists_one_owning_usage_receipt_per_conversation(
    monkeypatch, tmp_path
):
    indexing = {
        "scope_id": "locomo:conv",
        "complete": True,
        "healthy": True,
        "comparable": True,
        "pipeline_usage": {"request_attempts": 12},
    }

    class Adapter:
        pipeline_llm = object()

        def __init__(self, *_args, **_kwargs):
            pass

        def open(self):
            return self

        def close(self):
            pass

    monkeypatch.setattr(locomo, "MSCAdapter", Adapter)
    monkeypatch.setattr(
        locomo, "prepare_indexing", lambda *_args, **_kwargs: indexing
    )
    monkeypatch.setattr(
        locomo,
        "evaluate_qa",
        lambda q, *_args, **_kwargs: {
            "id": q["qa_id"],
            "question_id": q["question_id"],
            "correct": True,
        },
    )
    monkeypatch.setattr(locomo, "model_identity_fields", lambda *_args: {})
    args = SimpleNamespace(
        db_dir=None,
        keep_db=False,
        api_key="",
        sim=False,
        hymem_model="model",
        hymem_base_url="https://example.test/v1",
        hymem_thinking="disabled",
        embeddings=False,
        rules_extraction=None,
        graph_multihop=False,
        facts=None,
        facts_extraction=None,
        message_fts_top_k=None,
        rerank_top_k=None,
        fts_top_k=None,
        graph_top_k=None,
        diag_only=False,
    )
    conv = {
        "id": "conv",
        "n_sessions": 1,
        "qa": [
            {"qa_id": "q1", "question_id": "q1"},
            {"qa_id": "q2", "question_id": "q2"},
        ],
    }

    rows = locomo.evaluate_conversation(conv, args, None, None)

    assert rows[0]["indexing"] is indexing
    assert "indexing" not in rows[1]
    assert rows[1]["indexing_ref"] == "locomo:conv"
    assert sum("indexing" in row for row in rows) == 1
    assert sum(
        "pipeline_usage" in row.get("indexing", {}) for row in rows
    ) == 1


def test_locomo_question_failure_persists_only_bounded_exception_type(
    tmp_path, monkeypatch,
):
    secret = "LOCOMO_PRIVATE_SENTINEL_381"
    detail = (
        f"Bearer {secret} at /home/node/private/{secret}/key.json via "
        f"https://user:{secret}@provider.example/v1?token={secret}"
    )

    class Adapter:
        pipeline_llm = None

        def __init__(self, *_args, **_kwargs):
            pass

        def open(self):
            return self

        def close(self):
            pass

    monkeypatch.setattr(locomo, "MSCAdapter", Adapter)
    monkeypatch.setattr(
        locomo, "prepare_indexing", lambda *_args, **_kwargs: {
            "scope_id": "locomo:conv",
            "complete": True,
            "healthy": True,
            "comparable": True,
            "skip_reason": None,
            "pipeline_usage": {},
        },
    )

    def fail_question(*_args, **_kwargs):
        raise RuntimeError(detail)

    monkeypatch.setattr(locomo, "evaluate_qa", fail_question)
    monkeypatch.setattr(locomo, "model_identity_fields", lambda *_args: {})
    args = SimpleNamespace(
        db_dir=None, fresh=False, keep_db=False, api_key="", sim=False,
        hymem_model="model", hymem_base_url="https://example.test/v1",
        hymem_thinking="disabled", embeddings=False,
        rules_extraction=None, graph_multihop=False, facts=None,
        facts_extraction=None, message_fts_top_k=None, rerank_top_k=None,
        fts_top_k=None, graph_top_k=None, diag_only=False,
        answerable_clause=False,
    )
    conv = {
        "id": "conv", "n_sessions": 1,
        "qa": [{
            "qa_id": "q1", "question_id": "q1", "qtype": "type",
            "category": 1, "question": "benchmark question",
        }],
    }
    row = locomo.evaluate_conversation(conv, args, None, None)[0]
    assert row["benchmark_failure"] == "execution_failure:RuntimeError"
    wire = json.dumps(row)
    assert secret not in wire
    assert "/home/node/private" not in wire
    assert "Bearer" not in wire


@pytest.mark.parametrize(
    "error_type", [BenchmarkIntegrityError, BenchmarkCleanupError]
)
def test_locomo_question_structural_failure_escapes_item_boundary(
    tmp_path, monkeypatch, error_type,
):
    events = []

    class Adapter:
        pipeline_llm = None

        def __init__(self, *_args, **_kwargs):
            pass

        def open(self):
            return self

        def close(self):
            events.append("adapter_close")

    primary = error_type("structural question failure")
    monkeypatch.setattr(locomo, "MSCAdapter", Adapter)
    monkeypatch.setattr(
        locomo, "prepare_indexing", lambda *_args, **_kwargs: {
            "scope_id": "locomo:conv",
            "complete": True,
            "healthy": True,
            "comparable": True,
            "skip_reason": None,
            "pipeline_usage": {},
        },
    )
    monkeypatch.setattr(
        locomo, "evaluate_qa",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    args = SimpleNamespace(
        db_dir=None, fresh=False, keep_db=False, api_key="", sim=False,
        hymem_model="model", hymem_base_url="https://example.test/v1",
        hymem_thinking="disabled", embeddings=False,
        rules_extraction=None, graph_multihop=False, facts=None,
        facts_extraction=None, message_fts_top_k=None, rerank_top_k=None,
        fts_top_k=None, graph_top_k=None, diag_only=False,
        answerable_clause=False,
    )
    conv = {
        "id": "conv", "n_sessions": 1,
        "qa": [{
            "qa_id": "q1", "question_id": "q1", "qtype": "type",
            "category": 1, "question": "benchmark question",
        }],
    }

    with pytest.raises(error_type) as caught:
        locomo.evaluate_conversation(conv, args, None, None)

    assert caught.value is primary
    assert events == ["adapter_close"]


def _receipt_adapter(tmp_path):
    class Cursor:
        @staticmethod
        def fetchone():
            return (50,)

    class Conn:
        @staticmethod
        def execute(sql):
            assert "schema_version" in sql
            return Cursor()

    adapter = object.__new__(MSCAdapter)
    adapter.db_path = tmp_path / "hymem.sqlite"
    adapter.hymem_model = "pipeline-model"
    adapter.hymem_base_url = "https://memory.example/v1"
    adapter.hymem_thinking = "disabled"
    adapter.pipeline_llm = _ReceiptPipeline()
    adapter.embedding_client = None
    adapter.hy = SimpleNamespace(
        config=HyMemConfig(root=tmp_path), read_conn=Conn()
    )
    adapter.material_store_state = lambda: {
        "version": msc.MATERIAL_STORE_ATTESTATION_VERSION,
        "sha256": "sha256:" + "1" * 64,
        "tables": {
            "messages": {"rows": 1, "sha256": "sha256:" + "2" * 64},
        },
    }
    return adapter


def _receipt_item(text="remember this"):
    return {
        "id": "conv",
        "sessions": [[{"role": "user", "content": text}]],
        "session_dates": ["2025-01-01"],
    }


def _healthy_indexing(scope_id="locomo:conv"):
    totals = {field: 0 for field in msc._DREAM_REPORT_TOTAL_FIELDS}
    final_cycle = {
        **{field: 0 for field in msc._CURRENT_DREAM_REPORT_FAILURE_FIELDS},
        **{field: False for field in msc._CURRENT_DREAM_REPORT_BOOLEAN_FIELDS},
    }
    final_status = {
        "dream_status_schema": DREAM_STATUS_SCHEMA_VERSION,
        "benchmark_indexing_status_schema": (
            strictness.BENCHMARK_INDEXING_STATUS_VERSION
        ),
        **{
            field: 0 for field in msc._FINAL_STATUS_HEALTH_FIELDS
            if field not in {
                "dream_status_schema", "benchmark_indexing_status_schema",
                "in_progress", "terminal_loss_reasons",
                "coverage_integrity_failure_reasons",
                "coverage_integrity_failure_details",
                "coverage_integrity_failure_details_truncated",
                "coverage_integrity_config_version",
                "phase1_backlog_status", "pending_chunks_authoritative",
                "phase1_generation_key",
            }
        },
        "phase1_backlog_status": "current_producer",
        "pending_chunks_authoritative": True,
        "phase1_generation_key": "hymem-phase1-generation-v1:test",
        "terminal_loss_reasons": {},
        "coverage_integrity_failure_reasons": {},
        "coverage_integrity_failure_details": [],
        "coverage_integrity_failure_details_truncated": False,
        "coverage_integrity_config_version": (
            msc.COVERAGE_INTEGRITY_CONFIG_VERSION
        ),
        **{
            field: None
            for field in (
                *msc.DREAM_STATUS_AGGREGATION_AUTHORITY_FIELDS,
                *msc.DREAM_STATUS_AGGREGATION_MATERIAL_AUTHORITY_FIELDS,
                "aggregation_publication_generation",
                "aggregation_material_binding",
                "aggregation_material_revision",
            )
        },
        "aggregation_enabled": False,
        "in_progress": False,
    }
    run = {
        "trigger": "end_of_history",
        "cycles": 1,
        "report_count": 1,
        "complete": True,
        "healthy": True,
        "failure_reason": None,
        "elapsed_s": 0.1,
        "dream_report_totals": dict(totals),
        "budget_exhausted_cycles": 0,
        "extraction_provider_attempt_budget_exhausted_cycles": 0,
        "skipped_locked_cycles": 0,
        "final_cycle": final_cycle,
        "final_status": dict(final_status),
    }
    return {
        "protocol": msc.INDEXING_PROVENANCE_VERSION,
        "scope_id": scope_id,
        "mode": "converged",
        "comparable": True,
        "complete": True,
        "healthy": True,
        "convergence_count": 1,
        "cycles": 1,
        "settings": {
            "max_cycles_per_convergence": 2,
            "timeout_s_per_convergence": 10.0,
            "require_healthy": True,
        },
        "runs": [run],
        "dream_report_totals": totals,
        "budget_exhausted_cycles": 0,
        "extraction_provider_attempt_budget_exhausted_cycles": 0,
        "skipped_locked_cycles": 0,
        "final_status": final_status,
        "pipeline_usage": msc._known_zero_pipeline_usage(),
    }


def test_store_build_receipt_missing_and_mismatch_fail_closed(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    with pytest.raises(IndexingConvergenceError) as missing:
        adapter.validate_store_build_receipt(_receipt_item())
    assert missing.value.summary["failure_reason"] == (
        "missing_store_build_receipt"
    )
    assert missing.value.summary["receipt_file"] == msc.STORE_BUILD_RECEIPT_NAME
    assert "receipt_path" not in missing.value.summary
    assert str(adapter.store_build_receipt_path) not in json.dumps(
        missing.value.summary
    )
    assert "--fresh" in missing.value.summary["remediation"]

    adapter.publish_store_build_receipt(
        _receipt_item(), _healthy_indexing()
    )
    with pytest.raises(IndexingConvergenceError) as mismatch:
        adapter.validate_store_build_receipt(_receipt_item("different bytes"))
    assert mismatch.value.summary["failure_reason"] == (
        "store_build_identity_mismatch"
    )
    assert mismatch.value.summary["mismatch_fields"] == [
        "identity.source_sha256"
    ]


def test_store_build_receipt_match_and_atomic_exclusive_publication(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    item = _receipt_item()

    published = adapter.publish_store_build_receipt(item, _healthy_indexing())
    loaded = json.loads(adapter.store_build_receipt_path.read_text())

    assert loaded == published
    assert adapter.validate_store_build_receipt(item) == published
    assert not list(tmp_path.glob(f".{adapter.store_build_receipt_path.name}.*"))
    with pytest.raises(BenchmarkIntegrityError, match="refusing to overwrite"):
        adapter.publish_store_build_receipt(item, _healthy_indexing())


def test_store_build_receipt_rejects_model_and_write_config_mismatch(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    item = _receipt_item()
    adapter.publish_store_build_receipt(item, _healthy_indexing())

    adapter.pipeline_llm.model = "other-pipeline-model"
    with pytest.raises(IndexingConvergenceError) as model:
        adapter.validate_store_build_receipt(item)
    assert model.value.summary["mismatch_fields"] == [
        "identity.pipeline.generation_key",
        "identity.pipeline.producer.declaration.model",
        "identity.pipeline.producer.identity_sha256",
    ]

    adapter.pipeline_llm.model = "pipeline-model"
    adapter.hy.config = replace(adapter.hy.config, dream_budget=999)
    with pytest.raises(IndexingConvergenceError) as config:
        adapter.validate_store_build_receipt(item)
    assert config.value.summary["mismatch_fields"] == [
        "identity.write_config.dream_budget"
    ]


@pytest.mark.parametrize(
    ("field", "changed_value", "mismatch_field"),
    (
        (
            "evidence_role_weights",
            {"user": 3},
            "identity.write_config.evidence_role_weights.user",
        ),
        (
            "triple_dedup_enabled",
            False,
            "identity.write_config.triple_dedup_enabled",
        ),
        (
            "triple_dedup_cosine_threshold",
            0.96,
            "identity.write_config.triple_dedup_cosine_threshold",
        ),
        (
            "triple_dedup_lexical_ratio",
            0.84,
            "identity.write_config.triple_dedup_lexical_ratio",
        ),
    ),
)
def test_store_build_receipt_rejects_graph_write_config_mismatch(
    tmp_path, field, changed_value, mismatch_field,
):
    adapter = _receipt_adapter(tmp_path)
    item = _receipt_item()
    adapter.publish_store_build_receipt(item, _healthy_indexing())

    adapter.hy.config = replace(
        adapter.hy.config, **{field: changed_value}
    )
    with pytest.raises(IndexingConvergenceError) as mismatch:
        adapter.validate_store_build_receipt(item)

    assert mismatch.value.summary["failure_reason"] == (
        "store_build_identity_mismatch"
    )
    assert mismatch.value.summary["mismatch_fields"] == [mismatch_field]


def test_store_build_receipt_graph_write_defaults_and_mapping_order_are_stable(
    tmp_path,
):
    defaults = _receipt_adapter(tmp_path / "defaults")
    write_config = defaults.store_build_identity(_receipt_item())["write_config"]
    assert write_config["evidence_role_weights"] == {"user": 2}
    assert write_config["triple_dedup_enabled"] is True
    assert write_config["triple_dedup_cosine_threshold"] == 0.97
    assert write_config["triple_dedup_lexical_ratio"] == 0.85

    left = _receipt_adapter(tmp_path / "left")
    right = _receipt_adapter(tmp_path / "right")
    left.hy.config = replace(
        left.hy.config,
        evidence_role_weights={"assistant": 1, "user": 2},
    )
    right.hy.config = replace(
        right.hy.config,
        evidence_role_weights={"user": 2, "assistant": 1},
    )
    left.publish_store_build_receipt(_receipt_item(), _healthy_indexing())
    right.publish_store_build_receipt(_receipt_item(), _healthy_indexing())

    assert left.store_build_receipt_path.read_bytes() == (
        right.store_build_receipt_path.read_bytes()
    )


def test_store_build_receipt_rejects_legacy_graph_write_identity(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    item = _receipt_item()
    adapter.publish_store_build_receipt(item, _healthy_indexing())
    receipt = json.loads(adapter.store_build_receipt_path.read_text())
    omitted = (
        "evidence_role_weights",
        "triple_dedup_enabled",
        "triple_dedup_cosine_threshold",
        "triple_dedup_lexical_ratio",
    )
    for field in omitted:
        del receipt["identity"]["write_config"][field]
    receipt["identity_sha256"] = strictness.content_hash(receipt["identity"])
    adapter.store_build_receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(IndexingConvergenceError) as legacy:
        adapter.validate_store_build_receipt(item)

    assert legacy.value.summary["failure_reason"] == (
        "store_build_identity_mismatch"
    )
    assert legacy.value.summary["mismatch_fields"] == sorted(
        f"identity.write_config.{field}" for field in omitted
    )
    assert "--fresh" in legacy.value.summary["remediation"]


def test_store_build_receipt_never_serializes_endpoint_or_body_secrets(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    secret = "do-not-persist-this-secret"
    adapter.pipeline_llm.api_key = secret
    adapter.pipeline_llm.effective_extra_body = {
        "temperature": 0,
        "metadata": {"foo": secret},
    }

    published = adapter.publish_store_build_receipt(
        _receipt_item(), _healthy_indexing()
    )
    serialized = adapter.store_build_receipt_path.read_text()

    assert secret not in serialized
    assert "memory.example/v1" not in serialized
    declaration = published["identity"]["pipeline"]["producer"][
        "declaration"
    ]
    assert declaration["endpoint_origin"] == "https://memory.example"
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}", declaration["endpoint_sha256"]
    )
    nested = declaration["effective_request"]
    assert set(nested) == {"schema", "sha256"}
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", nested["sha256"])


@pytest.mark.parametrize(
    ("endpoint", "body"),
    [
        (
            "https://user:private@memory.example/v1?api_key=private",
            {"temperature": 0},
        ),
        (
            "https://memory.example/v1",
            {"authorization": "Bearer private", "temperature": 0},
        ),
    ],
)
def test_store_receipt_rejects_credential_bearing_producer_declaration(
    tmp_path, endpoint, body,
):
    adapter = _receipt_adapter(tmp_path)
    adapter.pipeline_llm.base_url = endpoint
    adapter.pipeline_llm.effective_extra_body = body

    with pytest.raises(
        BenchmarkIntegrityError,
        match="pipeline producer declaration is invalid",
    ) as failed:
        adapter.publish_store_build_receipt(
            _receipt_item(), _healthy_indexing()
        )
    assert "private" not in str(failed.value)
    assert not adapter.store_build_receipt_path.exists()


@pytest.mark.parametrize("corruption", ["invalid_json", "bad_digest"])
def test_store_build_receipt_rejects_corruption_with_fresh_remediation(
    tmp_path, corruption
):
    adapter = _receipt_adapter(tmp_path)
    item = _receipt_item()
    adapter.publish_store_build_receipt(item, _healthy_indexing())
    if corruption == "invalid_json":
        adapter.store_build_receipt_path.write_text("{")
        expected_reason = "malformed_store_build_receipt"
    else:
        receipt = json.loads(adapter.store_build_receipt_path.read_text())
        receipt["identity_sha256"] = "sha256:" + "0" * 64
        adapter.store_build_receipt_path.write_text(json.dumps(receipt))
        expected_reason = "corrupt_store_build_receipt"

    with pytest.raises(IndexingConvergenceError) as failed:
        adapter.validate_store_build_receipt(item)
    assert failed.value.summary["failure_reason"] == expected_reason
    assert failed.value.summary["remediation"] == (
        "rebuild this conversation store with --fresh"
    )


def test_store_build_receipt_is_never_published_for_unhealthy_index(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    indexing = _healthy_indexing()
    indexing["healthy"] = False
    with pytest.raises(BenchmarkIntegrityError, match="complete and healthy"):
        adapter.publish_store_build_receipt(_receipt_item(), indexing)
    assert not adapter.store_build_receipt_path.exists()


@pytest.mark.parametrize("protocol", (None, "hymem-benchmark-indexing-v2"))
def test_store_build_receipt_rejects_old_or_missing_nested_indexing_protocol(
    tmp_path, protocol,
):
    adapter = _receipt_adapter(tmp_path)
    indexing = _healthy_indexing()
    if protocol is None:
        indexing.pop("protocol")
    else:
        indexing["protocol"] = protocol
    with pytest.raises(BenchmarkIntegrityError, match="indexing provenance"):
        adapter.publish_store_build_receipt(_receipt_item(), indexing)

    current = _healthy_indexing()
    adapter.publish_store_build_receipt(_receipt_item(), current)
    receipt = json.loads(adapter.store_build_receipt_path.read_text())
    if protocol is None:
        receipt["indexing"].pop("protocol")
    else:
        receipt["indexing"]["protocol"] = protocol
    receipt["indexing_sha256"] = strictness.content_hash(receipt["indexing"])
    adapter.store_build_receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(IndexingConvergenceError) as caught:
        adapter.validate_store_build_receipt(_receipt_item())
    assert caught.value.summary["failure_reason"] == (
        "malformed_store_build_receipt"
    )


def test_store_build_receipt_rejects_reduced_relabelled_indexing(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    reduced = {
        "protocol": msc.INDEXING_PROVENANCE_VERSION,
        "scope_id": "locomo:conv",
        "mode": "converged",
        "comparable": True,
        "complete": True,
        "healthy": True,
    }

    with pytest.raises(BenchmarkIntegrityError, match="exact schema"):
        adapter.publish_store_build_receipt(_receipt_item(), reduced)

    assert not adapter.store_build_receipt_path.exists()


def test_store_build_receipt_rejects_stale_nested_indexing_digest(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    adapter.publish_store_build_receipt(_receipt_item(), _healthy_indexing())
    receipt = json.loads(adapter.store_build_receipt_path.read_text())
    receipt["indexing"]["cycles"] = 2
    adapter.store_build_receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(IndexingConvergenceError) as caught:
        adapter.validate_store_build_receipt(_receipt_item())

    assert caught.value.summary["failure_reason"] == "corrupt_store_build_receipt"
    assert caught.value.summary["recorded_indexing_sha256"] is not None
    assert caught.value.summary["actual_indexing_sha256"] is not None


@pytest.mark.parametrize(
    "mutate",
    (
        lambda p: p.__setitem__("convergence_count", 2),
        lambda p: p.__setitem__("cycles", 2),
        lambda p: p["runs"][0].__setitem__("report_count", 2),
        lambda p: p["dream_report_totals"].__setitem__("chunks_seen", 1),
        lambda p: p.__setitem__("budget_exhausted_cycles", 1),
    ),
)
def test_store_build_receipt_rejects_recomputed_inconsistent_indexing(
    tmp_path, mutate,
):
    adapter = _receipt_adapter(tmp_path)
    adapter.publish_store_build_receipt(_receipt_item(), _healthy_indexing())
    receipt = json.loads(adapter.store_build_receipt_path.read_text())
    mutate(receipt["indexing"])
    receipt["indexing_sha256"] = strictness.content_hash(receipt["indexing"])
    adapter.store_build_receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(IndexingConvergenceError) as caught:
        adapter.validate_store_build_receipt(_receipt_item())

    assert caught.value.summary["failure_reason"] == "malformed_store_build_receipt"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dream_status_schema", None),
        ("dream_status_schema", "hymem-dream-status-v1"),
        ("benchmark_indexing_status_schema", None),
        (
            "benchmark_indexing_status_schema",
            "hymem-benchmark-indexing-status-v1",
        ),
    ),
)
def test_store_build_receipt_rejects_missing_or_old_final_status_schema(
    tmp_path, field, value,
):
    adapter = _receipt_adapter(tmp_path)
    indexing = _healthy_indexing()
    for status in (indexing["final_status"], indexing["runs"][0]["final_status"]):
        if value is None:
            status.pop(field)
        else:
            status[field] = value

    with pytest.raises(BenchmarkIntegrityError):
        adapter.publish_store_build_receipt(_receipt_item(), indexing)

    assert not adapter.store_build_receipt_path.exists()


def test_store_build_receipt_rejects_recomputed_coverage_policy_drift(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    adapter.publish_store_build_receipt(_receipt_item(), _healthy_indexing())
    receipt = json.loads(adapter.store_build_receipt_path.read_text())
    for status in (
        receipt["indexing"]["final_status"],
        receipt["indexing"]["runs"][0]["final_status"],
    ):
        status["coverage_integrity_config_version"] = "forged-coverage-v9"
    receipt["indexing_sha256"] = strictness.content_hash(receipt["indexing"])
    adapter.store_build_receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(IndexingConvergenceError) as caught:
        adapter.validate_store_build_receipt(_receipt_item())

    assert caught.value.summary["failure_reason"] == "malformed_store_build_receipt"


def test_store_build_receipt_rejects_v5_envelope(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    adapter.publish_store_build_receipt(_receipt_item(), _healthy_indexing())
    receipt = json.loads(adapter.store_build_receipt_path.read_text())
    receipt["version"] = "hymem-benchmark-store-build-v5"
    adapter.store_build_receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(IndexingConvergenceError) as caught:
        adapter.validate_store_build_receipt(_receipt_item())

    assert caught.value.summary["failure_reason"] == (
        "incompatible_store_build_receipt_version"
    )


def test_store_receipt_rejects_phase1_producer_switch(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    adapter.publish_store_build_receipt(_receipt_item(), _healthy_indexing())
    adapter.pipeline_llm = StubLLMClient(default="{}")

    with pytest.raises(IndexingConvergenceError) as caught:
        adapter.validate_store_build_receipt(_receipt_item())

    assert caught.value.summary["failure_reason"] == (
        "store_build_identity_mismatch"
    )


@pytest.mark.parametrize("scope_id", ("msc:conv", "locomo:conv"))
def test_store_build_receipt_accepts_current_msc_and_locomo_scope(
    tmp_path, scope_id,
):
    adapter = _receipt_adapter(tmp_path)
    receipt = adapter.publish_store_build_receipt(
        _receipt_item(), _healthy_indexing(scope_id)
    )

    assert receipt["version"] == msc.STORE_BUILD_RECEIPT_VERSION
    assert receipt["indexing"]["scope_id"] == scope_id
    assert receipt["indexing_sha256"] == strictness.content_hash(
        receipt["indexing"]
    )
    assert adapter.validate_store_build_receipt(_receipt_item()) == receipt


def test_indexing_provenance_rejects_reduced_dream_report():
    assert set(asdict(DreamReport())) == set(msc._DREAM_REPORT_FIELDS)
    adapter, _handle = _adapter([{}], [_status()])
    adapter.dream(max_cycles=1, timeout_s=10)
    del adapter.indexing_runs[0]["reports"][0]["chunks_seen"]

    with pytest.raises(BenchmarkIntegrityError, match="reports are malformed"):
        adapter.indexing_provenance(scope_id="msc:one")


def test_store_build_receipt_allows_failed_calls_with_eventual_success(tmp_path):
    adapter = _receipt_adapter(tmp_path)
    indexing = _healthy_indexing()
    indexing["pipeline_usage"].update({
        "calls": 2,
        "calls_available": True,
        "request_attempts": 3,
        "request_attempts_available": True,
        "successful_responses": 1,
        "successful_responses_available": True,
    })

    receipt = adapter.publish_store_build_receipt(_receipt_item(), indexing)

    assert receipt["indexing"]["pipeline_usage"]["calls"] == 2
    assert adapter.validate_store_build_receipt(_receipt_item()) == receipt


def test_cli_failure_sidecar_is_immutable_bounded_evidence(tmp_path):
    out = tmp_path / "results.json"
    secret = "SIDECAR_PRIVATE_SENTINEL_604"
    absolute = f"/home/node/private/{secret}/receipt.json"
    detail = (
        f"Bearer {secret} at {absolute} via "
        f"https://user:{secret}@provider.example/v1?token={secret}"
    )
    failure = IndexingConvergenceError(
        "failed",
        {
            "scope_id": "locomo:conv",
            "failure_reason": f"cycle_exception: RuntimeError: {detail}",
            "receipt_path": absolute,
            "provider_exception": detail,
            "cycles": 2,
        },
    )

    def work():
        raise failure

    with pytest.raises(IndexingConvergenceError):
        run_or_record_indexing_failure(
            work,
            benchmark="locomo",
            out_path=str(out),
            extraction_canary={"status": "passed"},
        )
    sidecar = tmp_path / "results.json.indexing-failure.json"
    artifact = json.loads(sidecar.read_text())
    assert artifact["status"] == "failed_before_scoring"
    assert artifact["indexing"]["failure_reason"] == (
        "cycle_exception:RuntimeError"
    )
    wire = sidecar.read_text(encoding="utf-8")
    assert secret not in wire
    assert absolute not in wire
    assert "Bearer" not in wire

    with pytest.raises(IndexingConvergenceError):
        run_or_record_indexing_failure(
            work,
            benchmark="locomo",
            out_path=str(out),
            extraction_canary={"status": "passed"},
        )
    assert json.loads(sidecar.read_text()) == artifact


def test_locomo_legacy_reuse_fails_with_fresh_remediation_then_fresh_builds(
    monkeypatch, tmp_path
):
    db_dir = tmp_path / "stores"
    root = db_dir / "conv"
    root.mkdir(parents=True)
    (root / "hymem.sqlite").touch()
    events = []

    class Adapter:
        pipeline_llm = object()

        def __init__(self, *_args, **_kwargs):
            pass

        def open(self):
            return self

        def close(self):
            pass

    def prepare(*_args, reuse=False, **_kwargs):
        events.append(("prepare", reuse))
        if reuse:
            raise IndexingConvergenceError(
                "legacy receipt missing; rerun with --fresh",
                {
                    "failure_reason": "missing_store_build_receipt",
                    "remediation": "rebuild this conversation store with --fresh",
                },
            )
        return {
            "scope_id": "locomo:conv",
            "complete": True,
            "healthy": True,
            "comparable": True,
        }

    monkeypatch.setattr(locomo, "MSCAdapter", Adapter)
    monkeypatch.setattr(locomo, "prepare_indexing", prepare)
    monkeypatch.setattr(
        locomo,
        "evaluate_qa",
        lambda q, *_args, **_kwargs: {
            "id": q["qa_id"], "question_id": q["question_id"], "correct": True
        },
    )
    monkeypatch.setattr(locomo, "model_identity_fields", lambda *_args: {})
    args = SimpleNamespace(
        db_dir=str(db_dir),
        fresh=False,
        keep_db=False,
        api_key="",
        sim=False,
        hymem_model="model",
        hymem_base_url="https://example.test/v1",
        hymem_thinking="disabled",
        embeddings=False,
        rules_extraction=None,
        graph_multihop=False,
        facts=None,
        facts_extraction=None,
        message_fts_top_k=None,
        rerank_top_k=None,
        fts_top_k=None,
        graph_top_k=None,
        diag_only=False,
    )
    conv = {
        "id": "conv",
        "n_sessions": 1,
        "qa": [{"qa_id": "q", "question_id": "q"}],
    }

    with pytest.raises(IndexingConvergenceError) as legacy:
        locomo.evaluate_conversation(conv, args, None, None)
    assert legacy.value.summary["remediation"].endswith("--fresh")

    args.fresh = True
    rows = locomo.evaluate_conversation(conv, args, None, None)
    assert rows[0]["correct"] is True
    assert events == [("prepare", True), ("prepare", False)]
