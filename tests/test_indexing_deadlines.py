from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pytest

from benchmarks.lme_protocol import (
    LME_INDEXING_SUMMARY_VERSION,
    canonicalize_lme_indexing_summary,
)
from benchmarks.strictness import IndexingConvergenceError, converge_indexing
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.contrib.openai_embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from hymem.core import db as core_db
from hymem.deadline import (
    DeadlineBoundLLMClient,
    DeadlineExceeded,
    MonotonicDeadline,
    use_deadline,
)
from hymem.dreaming.aggregate import AggregationResult, aggregation_config_version
from hymem.extraction.llm import LLMRequest
from hymem.extraction.retry import with_retry
from hymem import HyMem, HyMemConfig


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class _FakeDefaultHttpxClient:
    """Owned no-proxy transport stub for SDK-bound deadline tests."""

    def __init__(self, **kwargs) -> None:
        self.trust_env = kwargs.get("trust_env")

    def close(self) -> None:
        pass


def _maintenance_config(tmp_path, **changes) -> HyMemConfig:
    values = {
        "root": tmp_path,
        "aggregation_nodes_enabled": False,
        "profile_extraction_enabled": False,
        "facts_extraction_enabled": False,
        "rules_extraction_enabled": False,
        "vacuum_after_prune": True,
        "vacuum_min_pruned": 1,
    }
    values.update(changes)
    return HyMemConfig(**values)


def _force_vacuum_threshold(monkeypatch: pytest.MonkeyPatch, runner) -> None:
    monkeypatch.setattr(runner, "prune_chunks", lambda *_args: 1)
    for name in (
        "prune_messages",
        "prune_retracted_edges",
        "prune_episodes_and_procedures",
        "prune_bookkeeping",
    ):
        monkeypatch.setattr(runner, name, lambda *_args: 0)


def test_convergence_expiry_before_first_cycle_makes_no_call() -> None:
    ticks = iter((0.0, 1.0, 1.0))
    calls = 0

    def clock() -> float:
        return next(ticks)

    def dream() -> dict:
        nonlocal calls
        calls += 1
        return {"budget_exhausted": False, "skipped_locked": False}

    with pytest.raises(IndexingConvergenceError) as caught:
        converge_indexing(
            dream,
            status=lambda: {"pending_chunks": 0},
            max_cycles=1,
            timeout_s=1.0,
            _clock=clock,
        )

    assert calls == 0
    assert caught.value.summary["failure_reason"] == "timeout_before_cycle"
    assert caught.value.summary["cycles"] == 0


def test_convergence_propagates_one_absolute_deadline_across_cycles() -> None:
    clock = FakeClock()
    seen: list[MonotonicDeadline] = []
    statuses = iter(({"pending_chunks": 1}, {"pending_chunks": 0}))

    def dream(*, deadline: MonotonicDeadline) -> dict:
        seen.append(deadline)
        clock.advance(0.2)
        return {
            "budget_exhausted": len(seen) == 1,
            "skipped_locked": False,
        }

    summary = converge_indexing(
        dream,
        status=lambda: next(statuses),
        max_cycles=2,
        timeout_s=1.0,
        _clock=clock,
    )

    assert summary["healthy"] is True
    assert len(seen) == 2 and seen[0] is seen[1]
    assert seen[0].expires_at == 1.0
    assert seen[0].remaining() == pytest.approx(0.6)


def test_convergence_supports_positional_only_deadline_callback() -> None:
    clock = FakeClock()
    seen: list[MonotonicDeadline] = []

    def dream(deadline: MonotonicDeadline, /) -> dict:
        seen.append(deadline)
        return {"budget_exhausted": False, "skipped_locked": False}

    summary = converge_indexing(
        dream,
        status=lambda: {"pending_chunks": 0},
        max_cycles=1,
        timeout_s=1.0,
        _clock=clock,
    )
    assert summary["healthy"] is True
    assert len(seen) == 1 and seen[0].expires_at == 1.0


def test_status_crossing_deadline_does_not_duplicate_completed_report() -> None:
    clock = FakeClock()

    def dream(*, deadline: MonotonicDeadline) -> dict:
        assert deadline.expires_at == 1.0
        clock.advance(0.5)
        return {"budget_exhausted": False, "skipped_locked": False}

    def status() -> dict:
        clock.advance(0.5)
        return {"pending_chunks": 0}

    with pytest.raises(IndexingConvergenceError) as caught:
        converge_indexing(
            dream,
            status=status,
            max_cycles=2,
            timeout_s=1.0,
            _clock=clock,
        )

    summary = caught.value.summary
    assert summary["failure_reason"] == "timeout_after_cycle"
    assert summary["cycles"] == 1
    assert len(summary["reports"]) == 1
    assert summary["elapsed_s"] == 1.0


def test_ignoring_provider_is_reported_when_control_returns_without_publication() -> None:
    clock = FakeClock()
    provider_calls = 0
    writes: list[str] = []

    class IgnoringProvider:
        def complete(self, _request) -> str:
            nonlocal provider_calls
            provider_calls += 1
            clock.advance(2.0)
            return "late result"

    def dream(*, deadline: MonotonicDeadline) -> dict:
        client = DeadlineBoundLLMClient(IgnoringProvider(), deadline)
        client.complete(LLMRequest(system="s", user="u"))
        writes.append("published")
        return {"budget_exhausted": False, "skipped_locked": False}

    with pytest.raises(IndexingConvergenceError) as caught:
        converge_indexing(
            dream,
            status=lambda: pytest.fail("late dream must not call status"),
            max_cycles=3,
            timeout_s=1.0,
            _clock=clock,
        )

    assert provider_calls == 1
    assert writes == []
    assert caught.value.summary["failure_reason"] == "timeout_during_cycle"
    assert caught.value.summary["reports"] == []
    assert caught.value.summary["elapsed_s"] == 2.0

    canonical = canonicalize_lme_indexing_summary(caught.value.summary)
    assert canonical["schema"] == LME_INDEXING_SUMMARY_VERSION
    assert canonical["failure"] == {
        "code": "timeout_during_cycle",
        "exception_type": None,
    }
    assert canonical["final_status"] is None


def test_mid_cycle_item_boundary_stops_calls_and_late_writes() -> None:
    clock = FakeClock()
    calls: list[int] = []
    writes: list[int] = []

    class Provider:
        def complete(self, _request) -> str:
            item = len(calls)
            calls.append(item)
            clock.advance(0.4)
            return str(item)

    def dream(*, deadline: MonotonicDeadline) -> dict:
        provider = DeadlineBoundLLMClient(Provider(), deadline)
        for item in range(10):
            deadline.check()
            provider.complete(LLMRequest(system="s", user=str(item)))
            deadline.check()
            writes.append(item)
        return {"budget_exhausted": False, "skipped_locked": False}

    with pytest.raises(IndexingConvergenceError) as caught:
        converge_indexing(
            dream,
            max_cycles=1,
            timeout_s=1.0,
            _clock=clock,
        )

    assert calls == [0, 1, 2]
    assert writes == [0, 1]
    assert caught.value.summary["failure_reason"] == "timeout_during_cycle"


def test_retry_backoff_cannot_start_an_attempt_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    deadline = MonotonicDeadline(0.75, clock=clock)
    attempts = 0
    sleeps: list[float] = []

    def attempt() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("transient")

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        clock.advance(delay)

    monkeypatch.setattr("hymem.extraction.retry.time.sleep", sleep)
    with use_deadline(deadline), pytest.raises(DeadlineExceeded):
        with_retry(attempt, attempts=5, base_delay=0.5, max_delay=8.0)

    assert attempts == 2
    assert sleeps == pytest.approx([0.5, 0.25])


def test_deadline_rolls_back_transaction_that_crosses_boundary(tmp_path) -> None:
    conn = core_db.connect(tmp_path / "deadline.sqlite")
    conn.execute("CREATE TABLE sample(value TEXT)")
    clock = FakeClock()
    deadline = MonotonicDeadline(1.0, clock=clock)
    try:
        with use_deadline(deadline), pytest.raises(DeadlineExceeded):
            with core_db.transaction(conn):
                conn.execute("INSERT INTO sample(value) VALUES ('late')")
                clock.advance(1.0)
        assert conn.execute("SELECT value FROM sample").fetchall() == []
    finally:
        conn.close()


def test_deadline_interrupted_setup_terminalizes_run_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    from hymem.dreaming import runner
    from hymem.extraction.llm import StubLLMClient

    cfg = HyMemConfig(
        root=tmp_path,
        aggregation_nodes_enabled=False,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
    )
    conn = core_db.connect(cfg.db_path)
    core_db.initialize(conn)
    clock = FakeClock()
    deadline = MonotonicDeadline(1.0, clock=clock)
    real_acquire = runner._acquire_lock

    def acquire_then_expire(connection, holder):
        acquired = real_acquire(connection, holder)
        clock.advance(1.0)
        return acquired

    monkeypatch.setattr(runner, "_acquire_lock", acquire_then_expire)
    try:
        with pytest.raises(DeadlineExceeded):
            runner.run_dreaming(
                conn,
                cfg,
                StubLLMClient(default="[]"),
                deadline=deadline,
            )
        assert conn.execute("SELECT * FROM run_lock").fetchall() == []
        run = conn.execute(
            "SELECT ended_at,error FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run["ended_at"] is not None
        assert run["error"] == "setup_interrupted"
    finally:
        conn.close()


def test_bounded_dream_defers_vacuum_and_rowid_resync(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    from hymem.dreaming import runner
    from hymem.extraction.llm import StubLLMClient

    cfg = _maintenance_config(tmp_path)
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    _ = hy.conn  # complete one-time v57 physical scrub before the dream spy
    _force_vacuum_threshold(monkeypatch, runner)
    resync_calls = 0
    sql: list[str] = []

    def resync(_conn) -> None:
        nonlocal resync_calls
        resync_calls += 1

    monkeypatch.setattr(core_db, "resync_rowid_shadows", resync)
    hy.conn.set_trace_callback(sql.append)
    try:
        report = hy.dream(
            deadline=MonotonicDeadline(10.0, clock=FakeClock())
        )
        assert report.skipped_locked is False
        assert not any(statement.strip().upper() == "VACUUM" for statement in sql)
        assert resync_calls == 0
    finally:
        hy.conn.set_trace_callback(None)
        hy.close()


def test_unbounded_dream_defers_unfenceable_vacuum_and_rowid_resync(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog,
) -> None:
    from hymem.dreaming import runner
    from hymem.extraction.llm import StubLLMClient

    cfg = _maintenance_config(tmp_path)
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    _ = hy.conn  # complete one-time v57 physical scrub before the dream spy
    _force_vacuum_threshold(monkeypatch, runner)
    resync_calls = 0
    sql: list[str] = []

    def resync(connection) -> None:
        nonlocal resync_calls
        assert connection is hy.conn
        resync_calls += 1

    monkeypatch.setattr(core_db, "resync_rowid_shadows", resync)
    hy.conn.set_trace_callback(sql.append)
    try:
        report = hy.dream()
        assert report.skipped_locked is False
        assert not any(statement.strip().upper() == "VACUUM" for statement in sql)
        assert resync_calls == 0
        assert "vacuum_deferred_lease_fence" in caplog.text
    finally:
        hy.conn.set_trace_callback(None)
        hy.close()


def test_expiry_at_premaintenance_boundary_rolls_back_and_starts_no_vacuum(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    from hymem.dreaming import runner
    from hymem.extraction.llm import StubLLMClient

    clock = FakeClock()
    cfg = _maintenance_config(tmp_path)
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    sql: list[str] = []
    resync_calls = 0

    hy.conn.execute(
        "INSERT INTO token_overlap_index(token, canonical) VALUES ('keep', 'keep')"
    )

    def expire_during_prune(*_args) -> int:
        clock.advance(1.0)
        return 1

    def resync(_conn) -> None:
        nonlocal resync_calls
        resync_calls += 1

    monkeypatch.setattr(runner, "prune_chunks", expire_during_prune)
    for name in (
        "prune_messages",
        "prune_retracted_edges",
        "prune_episodes_and_procedures",
        "prune_bookkeeping",
    ):
        monkeypatch.setattr(runner, name, lambda *_args: 0)
    monkeypatch.setattr(core_db, "resync_rowid_shadows", resync)
    hy.conn.set_trace_callback(sql.append)
    try:
        with pytest.raises(DeadlineExceeded):
            hy.dream(deadline=MonotonicDeadline(1.0, clock=clock))

        assert not any(statement.strip().upper() == "VACUUM" for statement in sql)
        assert resync_calls == 0
        # Phase 3 crossed the boundary inside its transaction, so its index
        # rewrite was rolled back rather than becoming a post-deadline write.
        assert [
            tuple(row) for row in hy.conn.execute(
                "SELECT token,canonical FROM token_overlap_index"
            ).fetchall()
        ] == [("keep", "keep")]
        run = hy.conn.execute(
            "SELECT ended_at,error FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run["ended_at"] is not None
        assert run["error"] == "deadline_exceeded"
    finally:
        hy.conn.set_trace_callback(None)
        hy.close()


def test_bounded_misaligned_rowid_shadows_fail_closed_without_resync(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    from hymem.dreaming import runner
    from hymem.extraction.llm import StubLLMClient

    cfg = _maintenance_config(
        tmp_path,
        aggregation_nodes_enabled=True,
        vacuum_after_prune=False,
    )
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    _ = hy.conn  # complete one-time v57 physical scrub before the dream spy
    build_calls = 0
    resync_calls = 0

    monkeypatch.setattr(core_db, "vec_episodes_aligned", lambda _conn: False)

    def resync(_conn) -> None:
        nonlocal resync_calls
        resync_calls += 1

    def build(*_args, **_kwargs) -> AggregationResult:
        nonlocal build_calls
        build_calls += 1
        return AggregationResult(0, 0)

    monkeypatch.setattr(core_db, "resync_rowid_shadows", resync)
    monkeypatch.setattr(runner, "build_aggregation_nodes", build)
    try:
        report = hy.dream(
            deadline=MonotonicDeadline(10.0, clock=FakeClock())
        )
        assert report.aggregation_build_exceptions == 1
        assert report.aggregation_fusion_failures == 1
        assert build_calls == 0
        assert resync_calls == 0
        status = hy.dream_status()
        assert status["pending_aggregation"] == 1
        assert status["aggregation_active_build_attempts"] == 1
        assert status["aggregation_active_caught_exceptions"] == 1
        assert status["aggregation_last_success_config_version"] is None
    finally:
        hy.close()


def test_expiry_during_rowid_alignment_probe_leaves_pending_health_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    from hymem.dreaming import runner
    from hymem.extraction.llm import StubLLMClient

    clock = FakeClock()
    cfg = _maintenance_config(
        tmp_path,
        aggregation_nodes_enabled=True,
        vacuum_after_prune=False,
    )
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    _ = hy.conn  # complete one-time v57 physical scrub before the dream spy
    build_calls = 0
    resync_calls = 0

    def expire_probe(_conn) -> bool:
        clock.advance(1.0)
        return False

    def resync(_conn) -> None:
        nonlocal resync_calls
        resync_calls += 1

    def build(*_args, **_kwargs) -> AggregationResult:
        nonlocal build_calls
        build_calls += 1
        return AggregationResult(0, 0)

    monkeypatch.setattr(core_db, "vec_episodes_aligned", expire_probe)
    monkeypatch.setattr(core_db, "resync_rowid_shadows", resync)
    monkeypatch.setattr(runner, "build_aggregation_nodes", build)
    try:
        with pytest.raises(DeadlineExceeded):
            hy.dream(deadline=MonotonicDeadline(1.0, clock=clock))

        assert build_calls == 0
        assert resync_calls == 0
        status = hy.dream_status()
        assert status["pending_aggregation"] == 1
        assert status["aggregation_active_build_attempts"] == 1
        assert status["aggregation_active_caught_exceptions"] == 0
        assert status["aggregation_active_fusion_failures"] == 0
        assert status["aggregation_last_success_config_version"] is None
    finally:
        hy.close()


@pytest.mark.parametrize("raises", [False, True])
def test_late_aggregation_control_never_acknowledges_pending_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path, raises: bool,
) -> None:
    from hymem.dreaming import runner
    from hymem.extraction.llm import StubLLMClient

    clock = FakeClock()
    cfg = _maintenance_config(
        tmp_path,
        aggregation_nodes_enabled=True,
        vacuum_after_prune=False,
    )
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    complete_calls = 0
    failure_calls = 0

    monkeypatch.setattr(core_db, "vec_episodes_aligned", lambda _conn: True)

    def late_build(*_args, **_kwargs) -> AggregationResult:
        clock.advance(1.0)
        if raises:
            raise RuntimeError("late injected failure")
        return AggregationResult(nodes=3, reused=2)

    def complete(*_args, **_kwargs) -> None:
        nonlocal complete_calls
        complete_calls += 1

    def record_failure(*_args, **_kwargs) -> None:
        nonlocal failure_calls
        failure_calls += 1

    monkeypatch.setattr(runner, "build_aggregation_nodes", late_build)
    monkeypatch.setattr(runner, "complete_aggregation_build", complete)
    monkeypatch.setattr(runner, "record_aggregation_build_failure", record_failure)
    try:
        with pytest.raises(DeadlineExceeded):
            hy.dream(deadline=MonotonicDeadline(1.0, clock=clock))

        assert complete_calls == 0
        assert failure_calls == 0
        status = hy.dream_status()
        assert status["pending_aggregation"] == 1
        assert status["aggregation_active_build_attempts"] == 1
        assert status["aggregation_active_caught_exceptions"] == 0
        assert status["aggregation_active_fusion_failures"] == 0
        assert status["aggregation_last_success_config_version"] is None
        run = hy.conn.execute(
            "SELECT ended_at,error FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run["ended_at"] is not None
        assert run["error"] == "deadline_exceeded"
    finally:
        hy.close()


@pytest.mark.parametrize("fusion_failures", [0, 2])
def test_on_time_bounded_aggregation_preserves_normal_health_acknowledgment(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fusion_failures: int,
) -> None:
    from hymem.dreaming import runner
    from hymem.extraction.llm import StubLLMClient

    cfg = _maintenance_config(
        tmp_path,
        aggregation_nodes_enabled=True,
        vacuum_after_prune=False,
    )
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    monkeypatch.setattr(core_db, "vec_episodes_aligned", lambda _conn: True)
    real_build = runner.build_aggregation_nodes

    def build(*args, **kwargs):
        if fusion_failures:
            return AggregationResult(
                nodes=1, reused=0, fusion_failures=fusion_failures,
            )
        # Completion may acknowledge only a publication the producer actually
        # materialized. The real zero-input build publishes the exact empty set.
        return real_build(*args, **kwargs)

    monkeypatch.setattr(runner, "build_aggregation_nodes", build)
    try:
        report = hy.dream(
            deadline=MonotonicDeadline(10.0, clock=FakeClock())
        )
        assert report.aggregation_fusion_failures == fusion_failures
        assert report.aggregation_build_exceptions == 0
        status = hy.dream_status()
        assert status["pending_aggregation"] == int(fusion_failures > 0)
        assert status["aggregation_active_fusion_failures"] == fusion_failures
        assert status["aggregation_total_fusion_failures"] == fusion_failures
        assert (status["aggregation_last_success_config_version"] is not None) == (
            fusion_failures == 0
        )
    finally:
        hy.close()


@pytest.mark.parametrize("fusion_failures", [0, 1])
def test_deadline_crossing_inside_aggregation_ack_rolls_it_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fusion_failures: int,
) -> None:
    from hymem.dreaming import runner
    from hymem.dreaming import aggregation_health
    from hymem.extraction.llm import StubLLMClient

    clock = FakeClock()
    cfg = _maintenance_config(
        tmp_path,
        aggregation_nodes_enabled=True,
        vacuum_after_prune=False,
    )
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    version = aggregation_config_version(cfg)

    monkeypatch.setattr(core_db, "vec_episodes_aligned", lambda _conn: True)
    real_build = runner.build_aggregation_nodes

    def build(*args, **kwargs):
        if fusion_failures:
            return AggregationResult(
                nodes=1, reused=0, fusion_failures=fusion_failures,
            )
        return real_build(*args, **kwargs)

    monkeypatch.setattr(runner, "build_aggregation_nodes", build)

    if fusion_failures:
        real_ack = aggregation_health.record_aggregation_build_failure

        def late_ack(conn, config_version, *args, **kwargs) -> None:
            real_ack(conn, config_version, *args, **kwargs)
            clock.advance(1.0)

        monkeypatch.setattr(runner, "record_aggregation_build_failure", late_ack)
    else:
        real_ack = aggregation_health.complete_aggregation_build

        def late_ack(conn, config_version, *args, **kwargs) -> None:
            real_ack(conn, config_version, *args, **kwargs)
            clock.advance(1.0)

        monkeypatch.setattr(runner, "complete_aggregation_build", late_ack)

    try:
        with pytest.raises(DeadlineExceeded):
            hy.dream(deadline=MonotonicDeadline(1.0, clock=clock))

        status = hy.dream_status()
        assert status["aggregation_config_version"] == version
        assert status["pending_aggregation"] == 1
        assert status["aggregation_active_build_attempts"] == 1
        assert status["aggregation_active_caught_exceptions"] == 0
        assert status["aggregation_active_fusion_failures"] == 0
        assert status["aggregation_total_caught_exceptions"] == 0
        assert status["aggregation_total_fusion_failures"] == 0
        assert status["aggregation_last_success_config_version"] is None
    finally:
        hy.close()


def test_deadline_bypasses_best_effort_extraction_and_releases_dream_lock(
    tmp_path,
) -> None:
    clock = FakeClock()
    calls = 0

    class LateLLM:
        def complete(self, _request) -> str:
            nonlocal calls
            calls += 1
            clock.advance(2.0)
            return '{"triples": [], "markers": [], "complete": true}'

    cfg = HyMemConfig(
        root=tmp_path,
        aggregation_nodes_enabled=False,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        rules_extraction_enabled=False,
        dream_budget=10,
    )
    hy = HyMem(cfg, llm=LateLLM())
    try:
        hy.log_message(
            "deadline-session",
            "user",
            "This deliberately long source turn should enter extraction and "
            "must never be marked processed from a provider result that is late.",
        )
        with pytest.raises(IndexingConvergenceError) as caught:
            converge_indexing(
                hy.dream,
                status=lambda: pytest.fail(
                    "deadline failure must not continue to durable status"
                ),
                max_cycles=2,
                timeout_s=1.0,
                _clock=clock,
            )
        assert caught.value.summary["failure_reason"] == "timeout_during_cycle"
        assert calls == 1
        assert hy.conn.execute("SELECT * FROM run_lock").fetchall() == []
        assert hy.conn.execute(
            "SELECT COUNT(*) AS n FROM processed_chunks"
        ).fetchone()["n"] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) AS n FROM chunk_extraction_attempts"
        ).fetchone()["n"] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) AS n FROM knowledge_graph"
        ).fetchone()["n"] == 0
        run = hy.conn.execute(
            "SELECT ended_at,error FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run["ended_at"] is not None
        assert run["error"] == "deadline_exceeded"
    finally:
        hy.close()


def test_network_clients_cap_each_request_to_exact_remaining_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_calls: list[dict] = []
    embedding_calls: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            def complete(**request):
                llm_calls.append(request)
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="ok")
                    )],
                    usage=None,
                )

            def embed(**request):
                embedding_calls.append(request)
                return SimpleNamespace(
                    data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])],
                    usage=None,
                )

            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=complete)
            )
            self.embeddings = SimpleNamespace(create=embed)
            self._client = kwargs["http_client"]
            self.api_key = kwargs.get("api_key")
            self.base_url = kwargs.get("base_url")
            self.organization = kwargs.get("organization")
            self.project = kwargs.get("project")

        def close(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=FakeOpenAI,
            DefaultHttpxClient=_FakeDefaultHttpxClient,
        ),
    )
    llm = OpenAICompatibleClient(
        api_key="key", base_url="https://provider.example/v1", model="m",
    )
    embedder = OpenAICompatibleEmbeddingClient(
        api_key="key", base_url="https://embed.example/v1", model="e",
        dim=2, timeout=2.0,
    )
    clock = FakeClock(10.0)
    deadline = MonotonicDeadline(10.75, clock=clock)

    with use_deadline(deadline):
        assert llm.complete(LLMRequest(system="s", user="u")) == "ok"
        assert embedder.embed(["text"]) == [[1.0, 0.0]]

    assert llm_calls[0]["timeout"] == pytest.approx(0.75)
    assert embedding_calls[0]["timeout"] == pytest.approx(0.75)

    clock.now = 10.75
    with use_deadline(deadline), pytest.raises(DeadlineExceeded):
        llm.complete(LLMRequest(system="s", user="must not start"))
    with use_deadline(deadline), pytest.raises(DeadlineExceeded):
        embedder.embed(["must not start"])
    assert len(llm_calls) == len(embedding_calls) == 1
    assert llm.request_attempts == 1
    assert embedder.request_attempts == 1


def test_late_network_responses_are_attempted_but_never_counted_successful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    late = False

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            def cross_if_late() -> None:
                if late:
                    clock.advance(2.0)

            def complete(**_request):
                cross_if_late()
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="ok")
                    )],
                    usage=SimpleNamespace(
                        prompt_tokens=1, completion_tokens=1, total_tokens=2,
                    ),
                )

            def embed(**_request):
                cross_if_late()
                return SimpleNamespace(
                    data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])],
                    usage=SimpleNamespace(prompt_tokens=1, total_tokens=1),
                )

            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=complete)
            )
            self.embeddings = SimpleNamespace(create=embed)
            self._client = kwargs["http_client"]
            self.api_key = kwargs.get("api_key")
            self.base_url = kwargs.get("base_url")
            self.organization = kwargs.get("organization")
            self.project = kwargs.get("project")

        def close(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=FakeOpenAI,
            DefaultHttpxClient=_FakeDefaultHttpxClient,
        ),
    )
    llm = OpenAICompatibleClient(
        api_key="key", base_url="https://provider.example/v1", model="m",
    )
    embedder = OpenAICompatibleEmbeddingClient(
        api_key="key", base_url="https://embed.example/v1", model="e", dim=2,
    )

    with use_deadline(MonotonicDeadline(1.0, clock=clock)):
        llm.complete(LLMRequest(system="s", user="on-time"))
        embedder.embed(["on-time"])
    assert llm.token_usage_available is True
    assert embedder.token_usage_available is True

    late = True
    clock.now = 0.0
    with use_deadline(MonotonicDeadline(1.0, clock=clock)), pytest.raises(
        DeadlineExceeded
    ):
        llm.complete(LLMRequest(system="s", user="late"))
    assert llm.request_attempts == 2
    assert llm.call_count == llm.successful_responses == 1
    assert llm.token_usage_available is False

    clock.now = 0.0
    with use_deadline(MonotonicDeadline(1.0, clock=clock)), pytest.raises(
        DeadlineExceeded
    ):
        embedder.embed(["late"])
    assert embedder.request_attempts == 2
    assert embedder.call_count == embedder.successful_responses == 1
    assert embedder.token_usage_available is False
    assert llm.total_latency_s >= 0.0
    assert embedder.total_latency_s >= 0.0


def test_timeout_cap_uses_one_clock_read_and_is_strictly_positive() -> None:
    ticks = iter((0.9, 1.0))
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return next(ticks)

    deadline = MonotonicDeadline(1.0, clock=clock)
    assert deadline.cap_timeout(120.0) == pytest.approx(0.1)
    assert calls == 1


@pytest.mark.parametrize("reason", ["timeout_during_cycle", "timeout_after_cycle"])
def test_lme_v2_accepts_timeout_at_exact_boundary(reason: str) -> None:
    reports = (
        [] if reason == "timeout_during_cycle"
        else [{
            "chunk_extraction_failures": 0,
            "coverage_integrity_failures": 0,
            "digest_failures": 0,
            "digest_quarantined": 0,
            "profile_failures": 0,
            "fact_failures": 0,
            "aggregation_fusion_failures": 0,
            "aggregation_build_exceptions": 0,
            "budget_exhausted": False,
            "extraction_provider_attempt_budget_exhausted": False,
            "skipped_locked": False,
        }]
    )
    summary = canonicalize_lme_indexing_summary({
        "cycles": len(reports),
        "max_cycles": 2,
        "timeout_s": 1.0,
        "elapsed_s": 1.0,
        "complete": False,
        "healthy": False,
        "failure_reason": reason,
        "reports": reports,
        "final_status": {},
        "quarantined": {},
    })
    assert summary["schema"] == LME_INDEXING_SUMMARY_VERSION
    assert summary["failure"]["code"] == reason
    assert len(summary["reports"]) == len(reports)
    assert summary["final_status"] is None


def test_unscoped_retry_behavior_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def attempt() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("retry")
        return "ok"

    monkeypatch.setattr("hymem.extraction.retry.time.sleep", lambda _delay: None)
    # Explicitly demonstrate that a None scope is equivalent to no scope.
    with nullcontext(), use_deadline(None):
        assert with_retry(attempt) == "ok"
    assert attempts == 3
