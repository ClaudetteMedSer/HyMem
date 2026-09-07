from __future__ import annotations

import dataclasses
import json

import pytest

from benchmarks.strictness import (
    IndexingConvergenceError,
    converge_indexing,
    durable_indexing_status,
)
from hymem import HyMem, HyMemConfig
from hymem.dreaming.aggregate import AggregationResult
from hymem.dreaming.aggregate import (
    aggregation_config_version,
    build_aggregation_nodes as real_build_aggregation_nodes,
)
from hymem.dreaming.aggregation_health import (
    begin_aggregation_build,
    complete_aggregation_build,
)
from hymem.dreaming.aggregation_generation import aggregation_generation_binding
from hymem.extraction.llm import StubLLMClient


def _cfg(tmp_path, **changes) -> HyMemConfig:
    base = HyMemConfig(
        root=tmp_path,
        aggregation_nodes_enabled=True,
        aggregation_digest_enabled=False,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
    )
    return dataclasses.replace(base, **changes)


def _memory(config: HyMemConfig) -> HyMem:
    return HyMem(config, llm=StubLLMClient(default="[]"))


def test_partial_fusion_failure_is_durable_and_server_serializable(
    tmp_path, monkeypatch,
):
    from fastapi.testclient import TestClient

    import hymem.honcho.app as server
    from hymem.dreaming import runner

    config = _cfg(tmp_path)
    hy = _memory(config)
    monkeypatch.setattr(
        runner,
        "build_aggregation_nodes",
        lambda *args, **kwargs: AggregationResult(
            nodes=3, reused=1, fusion_failures=2, input_episodes=7,
        ),
    )
    try:
        report = hy.dream()
        assert report.aggregation_fusion_failures == 2
        assert report.aggregation_build_exceptions == 0
        status = hy.dream_status()
        assert status["pending_aggregation"] == 1
        assert status["aggregation_active_build_attempts"] == 1
        assert status["aggregation_active_caught_exceptions"] == 0
        assert status["aggregation_active_fusion_failures"] == 2
        assert status["aggregation_total_fusion_failures"] == 2
        assert status["aggregation_last_failure_kind"] == "fusion_failure"
        assert status["last_run"]["aggregation_fusion_failures"] == 2
        assert status["last_run"]["aggregation_build_exceptions"] == 0
        assert status["last_run"]["aggregation_config_version"] == status[
            "aggregation_config_version"
        ]

        server.set_hy(hy)
        if server._scheduler is not None:
            server._scheduler.stop()
            server.set_scheduler(None)
        with TestClient(server.app) as client:
            body = client.get("/dream-status").json()
        assert body["pending_aggregation"] == 1
        assert body["aggregation_active_fusion_failures"] == 2
        assert body["aggregation_last_failure_kind"] == "fusion_failure"
    finally:
        hy.close()

    # The pending identity and bounded counters survive a new process/handle.
    reopened = HyMem(config)
    try:
        persisted = reopened.dream_status()
        assert persisted["pending_aggregation"] == 1
        assert persisted["aggregation_active_fusion_failures"] == 2
        assert persisted["aggregation_total_fusion_failures"] == 2
    finally:
        reopened.close()


def test_total_build_exception_is_never_a_zero_failure(tmp_path, monkeypatch):
    from hymem.dreaming import runner

    secret = "sk-should-never-enter-durable-health"

    def explode(*args, **kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(runner, "build_aggregation_nodes", explode)
    hy = _memory(_cfg(tmp_path))
    try:
        report = hy.dream()
        assert report.aggregation_build_exceptions == 1
        assert report.aggregation_fusion_failures == 1
        status = hy.dream_status()
        assert status["pending_aggregation"] == 1
        assert status["aggregation_active_caught_exceptions"] == 1
        assert status["aggregation_active_fusion_failures"] == 0
        assert status["aggregation_total_caught_exceptions"] == 1
        assert status["aggregation_last_failure_kind"] == "exception"
        assert status["last_run"]["aggregation_build_exceptions"] == 1
        assert status["last_run"]["aggregation_fusion_failures"] == 1
        assert secret not in json.dumps(status)
        row = hy.conn.execute(
            "SELECT * FROM aggregation_build_health WHERE id=1"
        ).fetchone()
        assert secret not in json.dumps(dict(row))
    finally:
        hy.close()


def test_clean_return_without_exact_publication_cannot_complete_health(
    tmp_path, monkeypatch,
):
    from hymem.dreaming import runner

    monkeypatch.setattr(
        runner,
        "build_aggregation_nodes",
        lambda *args, **kwargs: AggregationResult(
            nodes=7, reused=0, fusion_failures=0,
        ),
    )
    hy = _memory(_cfg(tmp_path))
    try:
        with pytest.raises(
            RuntimeError, match="exact current publication"
        ):
            hy.dream()
        status = hy.dream_status()
        assert status["pending_aggregation"] == 1
        assert status["aggregation_active_build_attempts"] == 1
        assert status["aggregation_last_success_config_version"] is None
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM aggregation_publication_state"
        ).fetchone()[0] == 0
    finally:
        hy.close()


def test_prebuild_pending_marker_survives_process_death(tmp_path):
    config = _cfg(tmp_path)
    hy = _memory(config)
    version = aggregation_config_version(config)
    generation = aggregation_generation_binding(config, hy._llm)
    try:
        attempt_token = begin_aggregation_build(
            hy.conn, version, generation_binding=generation
        )
        built = real_build_aggregation_nodes(
            hy.conn, config, StubLLMClient(default="[]"),
            health_managed=True, health_attempt_token=attempt_token,
        )
        complete_aggregation_build(
            hy.conn, version, generation["generation_key"],
            attempt_token,
            expected_node_count=built.nodes,
        )
        assert hy.dream_status()["pending_aggregation"] == 0
        # Model the process boundary precisely: begin committed, build never
        # returned, so neither failure-recording nor success runs.
        attempt_token = begin_aggregation_build(
            hy.conn, version, generation_binding=generation
        )
        wrong_version = "aggregation-build-config-v1:" + ("0" * 64)
        assert wrong_version != version
        with pytest.raises(RuntimeError, match="no matching pending build"):
            complete_aggregation_build(
                hy.conn, wrong_version, generation["generation_key"],
                attempt_token,
                expected_node_count=0,
            )
    finally:
        hy.close()

    reopened = HyMem(config)
    try:
        status = reopened.dream_status()
        assert status["pending_aggregation"] == 1
        assert status["aggregation_active_build_attempts"] == 1
        assert status["aggregation_active_caught_exceptions"] == 0
        assert status["aggregation_active_fusion_failures"] == 0
    finally:
        reopened.close()


def test_transient_aggregation_failure_heals_during_convergence(
    tmp_path, monkeypatch,
):
    from hymem.dreaming import runner

    outcomes = iter((1, 0))

    def staged_build(*args, **kwargs):
        if next(outcomes):
            return AggregationResult(
                nodes=1, reused=0, fusion_failures=1, input_episodes=2,
            )
        return real_build_aggregation_nodes(*args, **kwargs)

    monkeypatch.setattr(runner, "build_aggregation_nodes", staged_build)
    hy = _memory(_cfg(tmp_path))
    try:
        summary = converge_indexing(
            hy.dream,
            status=lambda: durable_indexing_status(hy, None),
            max_cycles=3,
            timeout_s=10,
        )
        assert summary["cycles"] == 2
        assert summary["complete"] is True
        assert summary["healthy"] is True
        assert summary["reports"][0]["aggregation_fusion_failures"] == 1
        assert summary["reports"][1]["aggregation_fusion_failures"] == 0
        status = hy.dream_status()
        assert status["pending_aggregation"] == 0
        assert status["aggregation_active_build_attempts"] == 0
        assert status["aggregation_total_fusion_failures"] == 1
        assert status["aggregation_last_failure_kind"] == "fusion_failure"
        assert status["aggregation_last_success_config_version"] == status[
            "aggregation_config_version"
        ]
    finally:
        hy.close()


def test_permanent_aggregation_failure_exhausts_convergence_cap(
    tmp_path, monkeypatch,
):
    from hymem.dreaming import runner

    monkeypatch.setattr(
        runner,
        "build_aggregation_nodes",
        lambda *args, **kwargs: AggregationResult(
            nodes=0, reused=0, fusion_failures=1,
        ),
    )
    hy = _memory(_cfg(tmp_path))
    try:
        with pytest.raises(IndexingConvergenceError) as failed:
            converge_indexing(
                hy.dream,
                status=lambda: durable_indexing_status(hy, None),
                max_cycles=2,
                timeout_s=10,
            )
        assert failed.value.summary["failure_reason"] == "max_cycles_exhausted"
        assert failed.value.summary["cycles"] == 2
        assert failed.value.summary["final_status"]["pending_aggregation"] == 1
        assert hy.dream_status()["aggregation_active_build_attempts"] == 2
        assert hy.dream_status()["aggregation_total_fusion_failures"] == 2
    finally:
        hy.close()


def test_positive_report_failure_retries_even_without_durable_status_field():
    reports = iter((
        {"budget_exhausted": False, "aggregation_fusion_failures": 1},
        {"budget_exhausted": False, "aggregation_fusion_failures": 0},
    ))
    result = converge_indexing(
        lambda: next(reports),
        status=lambda: {"pending_chunks": 0},
        max_cycles=2,
        timeout_s=10,
    )
    assert result["cycles"] == 2


def test_mcp_report_never_labels_failed_aggregation_complete(
    tmp_path, monkeypatch,
):
    import hymem.server as server
    from hymem.dreaming import runner

    monkeypatch.setattr(
        runner,
        "build_aggregation_nodes",
        lambda *args, **kwargs: AggregationResult(
            nodes=0, reused=0, fusion_failures=4,
        ),
    )
    hy = _memory(_cfg(tmp_path))
    try:
        server.set_hy(hy)
        result = server._do_dream()
        assert result.startswith("dreaming incomplete/unverified —")
        assert "finished cleanly" not in result
    finally:
        hy.close()


def test_disabled_and_changed_config_do_not_misapply_old_failure(
    tmp_path, monkeypatch,
):
    from hymem.dreaming import runner

    config_a = _cfg(tmp_path)
    monkeypatch.setattr(
        runner,
        "build_aggregation_nodes",
        lambda *args, **kwargs: AggregationResult(
            nodes=0, reused=0, fusion_failures=1,
        ),
    )
    first = _memory(config_a)
    try:
        first.dream()
        failed_version = first.dream_status()["aggregation_config_version"]
    finally:
        first.close()

    disabled = HyMem(dataclasses.replace(config_a, aggregation_nodes_enabled=False))
    try:
        status = disabled.dream_status()
        assert status["pending_aggregation"] == 0
        assert status["aggregation_enabled"] is False
        assert status["aggregation_total_fusion_failures"] == 1
        assert status["aggregation_stale_pending_config_version"] == failed_version
    finally:
        disabled.close()

    config_b = dataclasses.replace(
        config_a,
        aggregation_emb_threshold=config_a.aggregation_emb_threshold + 0.01,
    )
    changed = _memory(config_b)
    try:
        before = changed.dream_status()
        assert before["aggregation_config_version"] != failed_version
        assert before["pending_aggregation"] == 1
        assert before["aggregation_active_fusion_failures"] == 0
        assert before["aggregation_stale_pending_config_version"] == failed_version

        monkeypatch.setattr(
            runner,
            "build_aggregation_nodes",
            real_build_aggregation_nodes,
        )
        changed.dream()
        healed = changed.dream_status()
        assert healed["pending_aggregation"] == 0
        assert healed["aggregation_stale_pending_config_version"] is None
        assert healed["aggregation_total_fusion_failures"] == 1
        assert healed["aggregation_superseded_pending_configs"] == 1
    finally:
        changed.close()
