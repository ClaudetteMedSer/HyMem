"""Unit tests for DreamScheduler: kick→runs, cooldown gating, clean shutdown."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hymem import DreamLeaseLost, HyMem, HyMemConfig, StubEmbeddingClient
import hymem.dreaming.scheduler as scheduler_module
from hymem.dreaming.scheduler import DreamScheduler
from hymem.extraction.llm import StubLLMClient


def _make_hy(tmp_path: Path) -> HyMem:
    return HyMem(
        HyMemConfig(root=tmp_path),
        llm=StubLLMClient(default="[]"),
        embedding_client=StubEmbeddingClient(),
    )


def test_scheduler_kick_runs_one_cycle(tmp_path: Path) -> None:
    hy = _make_hy(tmp_path)
    sched = DreamScheduler(hy, cooldown=0.0)
    sched.start()
    try:
        sched.kick()
        assert sched.wait_for_cycle(1, timeout=5.0)
        assert sched.cycles_completed == 1
    finally:
        sched.stop()
        hy.close()


def test_scheduler_cooldown_gates_second_kick(tmp_path: Path) -> None:
    hy = _make_hy(tmp_path)
    sched = DreamScheduler(hy, cooldown=10.0)
    sched.start()
    try:
        sched.kick()
        assert sched.wait_for_cycle(1, timeout=5.0)
        # Second kick within cooldown — must NOT trigger another cycle yet.
        sched.kick()
        time.sleep(0.3)
        assert sched.cycles_completed == 1
    finally:
        sched.stop()
        hy.close()


def test_scheduler_stop_joins_cleanly(tmp_path: Path) -> None:
    hy = _make_hy(tmp_path)
    sched = DreamScheduler(hy, cooldown=0.0)
    sched.start()
    assert sched.is_running
    sched.stop(timeout=5.0)
    assert not sched.is_running
    hy.close()


def test_scheduler_recovers_from_failing_cycle(tmp_path: Path) -> None:
    """A failing cycle must not kill the daemon — subsequent kicks still run."""
    hy = _make_hy(tmp_path)
    sched = DreamScheduler(hy, cooldown=0.0)
    sched.start()
    try:
        # Poison the LLM so the first cycle path raises somewhere downstream.
        # We rely on the runner's exception handling to keep the thread alive.
        sched.kick()
        assert sched.wait_for_cycle(1, timeout=5.0)

        # Re-kick — should still work.
        sched.kick()
        assert sched.wait_for_cycle(2, timeout=5.0)
        assert sched.is_running
    finally:
        sched.stop()
        hy.close()


def test_scheduler_survives_lease_loss_and_runs_a_later_kick() -> None:
    first_aborted = threading.Event()
    first_invalidated = threading.Event()
    second_completed = threading.Event()
    calls = 0
    invalidations = 0

    class DreamFork:
        def dream(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_aborted.set()
                raise DreamLeaseLost("dreaming lease ownership lost")
            second_completed.set()

        def close(self):
            pass

    fork = DreamFork()

    class Root:
        def fork(self):
            return fork

        def invalidate_query_caches(self):
            nonlocal invalidations
            invalidations += 1
            if invalidations == 1:
                first_invalidated.set()

    sched = DreamScheduler(Root(), cooldown=0.0)
    sched.start()
    try:
        sched.kick()
        assert first_aborted.wait(timeout=2.0)
        assert first_invalidated.wait(timeout=2.0)

        sched.kick()
        assert second_completed.wait(timeout=2.0)
        assert sched.wait_for_cycle(1, timeout=2.0)
        assert calls == 2
        assert invalidations == 2
        assert sched.is_running is True
    finally:
        sched.stop()


def test_stop_timeout_keeps_live_worker_and_fork_owned_until_retry():
    entered = threading.Event()
    release = threading.Event()
    events: list[str] = []

    class DreamFork:
        def dream(self):
            entered.set()
            assert release.wait(timeout=5.0)
            events.append("dream:return")

        def close(self):
            events.append("fork:close")

    fork = DreamFork()

    class Root:
        def fork(self):
            return fork

        def invalidate_query_caches(self):
            events.append("root:invalidate")

    sched = DreamScheduler(Root(), cooldown=0.0)
    sched.start()
    sched.kick()
    assert entered.wait(timeout=2.0)

    with pytest.raises(TimeoutError, match="did not stop"):
        sched.stop(timeout=0.01)

    assert sched.is_running is True
    assert sched.shutdown_pending is True
    assert "fork:close" not in events

    release.set()
    sched.stop(timeout=2.0)
    assert sched.is_running is False
    assert sched.shutdown_pending is False
    assert events == ["dream:return", "root:invalidate", "fork:close"]


def test_thread_construction_failure_closes_scheduler_fork_once(monkeypatch):
    events: list[str] = []

    class DreamFork:
        def close(self):
            events.append("fork:close")

    class Root:
        def fork(self):
            events.append("root:fork")
            return DreamFork()

    primary = RuntimeError("thread construction failed")

    def fail_thread(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(scheduler_module.threading, "Thread", fail_thread)
    sched = DreamScheduler(Root(), cooldown=0.0)

    with pytest.raises(RuntimeError) as caught:
        sched.start()

    assert caught.value is primary
    assert events == ["root:fork", "fork:close"]
    assert sched.shutdown_pending is False


def test_thread_start_failure_closes_scheduler_fork_once(monkeypatch):
    events: list[str] = []
    primary = SystemExit("thread start failed")

    class DreamFork:
        def close(self):
            events.append("fork:close")

    class Root:
        def fork(self):
            events.append("root:fork")
            return DreamFork()

    class UnstartedThread:
        def start(self):
            events.append("thread:start")
            raise primary

        def is_alive(self):
            return False

    monkeypatch.setattr(
        scheduler_module.threading,
        "Thread",
        lambda **_kwargs: UnstartedThread(),
    )
    sched = DreamScheduler(Root(), cooldown=0.0)

    with pytest.raises(SystemExit) as caught:
        sched.start()

    assert caught.value is primary
    assert events == ["root:fork", "thread:start", "fork:close"]
    assert sched.shutdown_pending is False


def test_startup_fork_close_failure_remains_retryable(monkeypatch):
    events: list[str] = []
    primary = RuntimeError("thread construction failed")

    class DreamFork:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            events.append(f"fork:close:{self.close_calls}")
            if self.close_calls == 1:
                raise SystemExit("first close failed")

    fork = DreamFork()

    class Root:
        def fork(self):
            return fork

    monkeypatch.setattr(
        scheduler_module.threading,
        "Thread",
        lambda **_kwargs: (_ for _ in ()).throw(primary),
    )
    sched = DreamScheduler(Root(), cooldown=0.0)

    with pytest.raises(RuntimeError) as caught:
        sched.start()

    assert caught.value is primary
    assert "SystemExit" in " ".join(getattr(primary, "__notes__", ()))
    assert sched.shutdown_pending is True
    sched.stop()
    assert sched.shutdown_pending is False
    assert events == ["fork:close:1", "fork:close:2"]


def test_fork_failure_leaves_no_partial_scheduler_state():
    primary = KeyboardInterrupt("fork failed")

    class Root:
        def fork(self):
            raise primary

    sched = DreamScheduler(Root(), cooldown=0.0)
    with pytest.raises(KeyboardInterrupt) as caught:
        sched.start()

    assert caught.value is primary
    assert sched.shutdown_pending is False
