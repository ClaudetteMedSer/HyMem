"""Unit scheduler policy/lifecycle tests, independent of database latency.

The real HyMem fork/dream pipeline is exercised by the controlled HTTP scheduler
tests in test_honcho_server.py. These tests run the real scheduler thread with a
small fork double so cold SQLite initialization is not an implicit five-second
performance requirement of a scheduler unit test.
"""
from __future__ import annotations

from contextlib import contextmanager
from queue import Empty, Queue
import threading
from types import SimpleNamespace

import pytest

from hymem import DreamLeaseLost
import hymem.dreaming.scheduler as scheduler_module
from hymem.dreaming.scheduler import DreamScheduler


_COORDINATION_TIMEOUT = 5.0  # Deadlock guard; no provider or database work.


class _SchedulerControl:
    def __init__(self, monkeypatch, *, cooldown=0.0):
        self.now = 1000.0
        self.events = Queue()
        self.actions = Queue()
        self.cooldown_release = threading.Event()
        self.errors = []
        self.started = 0
        self.invalidations = 0
        self.forks = 0
        self.closes = 0
        self.lifecycle_events = []
        control = self

        class DreamFork:
            def dream(self):
                control.started += 1
                control.events.put(("dream_started", control.started))
                try:
                    action = control.actions.get(timeout=_COORDINATION_TIMEOUT)
                except Empty as exc:
                    control.errors.append(exc)
                    raise AssertionError("test did not release the dream") from exc
                if action is not None:
                    control.events.put(("dream_raised", action))
                    raise action
                control.lifecycle_events.append("dream:return")
                control.events.put(("dream_returned", control.started))

            def close(self):
                assert not control.scheduler.is_running
                control.lifecycle_events.append("fork:close")
                control.closes += 1

        class Root:
            def fork(self):
                control.forks += 1
                return DreamFork()

            def invalidate_query_caches(self):
                control.lifecycle_events.append("root:invalidate")
                control.invalidations += 1

        self.scheduler = DreamScheduler(Root(), cooldown=cooldown)
        # Replace this module's reference, not the process-wide time module.
        # The real Events/Queue/join keep their real bounded coordination clocks.
        monkeypatch.setattr(
            scheduler_module, "time",
            SimpleNamespace(monotonic=lambda: self.now,
                            sleep=scheduler_module.time.sleep),
        )
        original_kick_wait = self.scheduler._kick.wait

        def kick_wait():
            self.events.put(("waiting_for_kick", (
                self.scheduler.cycles_completed, self.scheduler._kick.is_set(),
                self.invalidations,
            )))
            return original_kick_wait()

        def cooldown_wait(timeout):
            self.events.put(("cooldown", timeout))
            if not self.cooldown_release.wait(_COORDINATION_TIMEOUT):
                self.errors.append("test did not release the cooldown")
                raise AssertionError(self.errors[-1])
            self.cooldown_release.clear()
            return self.scheduler._stop.is_set()

        monkeypatch.setattr(self.scheduler._kick, "wait", kick_wait)
        monkeypatch.setattr(self.scheduler._stop, "wait", cooldown_wait)

    def expect(self, kind, value):
        try:
            observed = self.events.get(timeout=_COORDINATION_TIMEOUT)
        except Empty:
            pytest.fail(f"scheduler did not reach {kind}: {self.errors!r}")
        assert observed == (kind, value), f"expected {kind}: got {observed!r}"

    def finish_cycle(self, ordinal, *, completed=None, invalidations=None,
                     pending=False):
        self.actions.put(None)
        self.expect("dream_returned", ordinal)
        self.expect("waiting_for_kick", (
            ordinal if completed is None else completed, pending,
            ordinal if invalidations is None else invalidations,
        ))


@contextmanager
def _controlled_scheduler(monkeypatch, *, cooldown=0.0):
    control = _SchedulerControl(monkeypatch, cooldown=cooldown)
    try:
        control.scheduler.start()
        control.expect("waiting_for_kick", (0, False, 0))
        yield control
    finally:
        # Release either parked path even after a failed assertion. Always join
        # before monkeypatch restores hooks or any test-owned object is closed.
        control.scheduler._stop.set()
        control.actions.put(None)
        control.cooldown_release.set()
        control.scheduler.stop(timeout=_COORDINATION_TIMEOUT)
        assert not control.scheduler.is_running
        assert not control.scheduler.shutdown_pending
        assert control.forks == control.closes == 1
        assert not control.errors


def test_scheduler_kick_runs_one_cycle(monkeypatch) -> None:
    with _controlled_scheduler(monkeypatch) as control:
        control.scheduler.kick()
        control.expect("dream_started", 1)
        assert control.scheduler.cycles_completed == 0
        control.finish_cycle(1)
        assert control.scheduler.wait_for_cycle(1, timeout=0.0)
        assert not control.scheduler.wait_for_cycle(2, timeout=0.0)
        assert control.started == control.invalidations == 1


@pytest.mark.parametrize("elapsed", [4.0, 10.0, 11.0])
def test_scheduler_cooldown_gates_second_kick(monkeypatch, elapsed) -> None:
    with _controlled_scheduler(monkeypatch, cooldown=10.0) as control:
        control.scheduler.kick()
        control.expect("dream_started", 1)
        control.finish_cycle(1)
        control.now += elapsed
        control.scheduler.kick()
        if elapsed < 10.0:
            control.expect("cooldown", 10.0 - elapsed)
            assert control.started == control.scheduler.cycles_completed == 1
            control.now += 10.0 - elapsed
            control.cooldown_release.set()
        # At or beyond the boundary no timed wait is allowed.
        control.expect("dream_started", 2)
        control.finish_cycle(2)


def _assert_in_flight_kicks_are_cooled(control, *, drop_kick=False):
    control.scheduler.kick()
    control.expect("dream_started", 1)
    # These happen after the first kick was consumed, so they must preserve one
    # pending cycle. Event semantics coalesce the burst rather than count it.
    for _ in range(10):
        control.scheduler.kick()
    if drop_kick:
        control.scheduler._kick.clear()
    control.finish_cycle(1, pending=True)
    control.expect("cooldown", 10.0)
    assert control.started == control.scheduler.cycles_completed == 1


def test_scheduler_coalesces_in_flight_kicks_without_losing_work(monkeypatch):
    with _controlled_scheduler(monkeypatch, cooldown=10.0) as control:
        _assert_in_flight_kicks_are_cooled(control)
        control.now += 10.0
        control.cooldown_release.set()
        control.expect("dream_started", 2)
        control.finish_cycle(2)
        assert control.started == control.scheduler.cycles_completed == 2


@pytest.mark.parametrize("defect", ["bypassed_cooldown", "lost_pending_kick"])
def test_scheduler_cooldown_oracle_rejects_defects(monkeypatch, defect):
    with _controlled_scheduler(monkeypatch, cooldown=10.0) as control:
        if defect == "bypassed_cooldown":
            control.scheduler._cooldown = 0.0
        expected = "cooldown" if defect == "bypassed_cooldown" else "waiting_for_kick"
        with pytest.raises(AssertionError, match=f"expected {expected}"):
            _assert_in_flight_kicks_are_cooled(
                control, drop_kick=defect == "lost_pending_kick",
            )


def test_scheduler_stop_joins_cleanly(monkeypatch) -> None:
    with _controlled_scheduler(monkeypatch) as control:
        sched = control.scheduler
        assert sched.is_running
        sched.start()  # Starting twice must not create a second owner.
        assert control.forks == 1
        sched.stop(timeout=_COORDINATION_TIMEOUT)
        assert not sched.is_running
        assert not sched.shutdown_pending
        assert control.started == control.invalidations == 0
        assert control.closes == 1
    assert control.closes == 1  # Repeated stop is also harmless.


def _assert_failed_cycle(control, error, *, suppress_failure=False):
    control.scheduler.kick()
    control.expect("dream_started", 1)
    control.actions.put(None if suppress_failure else error)
    control.expect("dream_raised", error)
    control.expect("waiting_for_kick", (0, False, 0))


def test_scheduler_recovers_from_failing_cycle(monkeypatch, caplog) -> None:
    """A real exception reaches the scheduler; it is not counted as a success."""
    error = RuntimeError("injected dream failure")
    with _controlled_scheduler(monkeypatch) as control:
        _assert_failed_cycle(control, error)
        failures = [record for record in caplog.records
                    if record.getMessage() == "dream_scheduler.cycle_failed"]
        assert len(failures) == 1
        assert failures[0].exc_info[1] is error
        assert control.scheduler.is_running
        control.scheduler.kick()
        control.expect("dream_started", 2)
        control.finish_cycle(2, completed=1, invalidations=1)
        assert control.started == 2
        assert control.scheduler.cycles_completed == 1
        assert control.scheduler.is_running


def test_scheduler_failure_oracle_rejects_a_successful_cycle(monkeypatch):
    with _controlled_scheduler(monkeypatch) as control:
        # Regression control for the old test, which claimed to poison the LLM
        # but actually ran two ordinary successful cycles.
        with pytest.raises(AssertionError, match="expected dream_raised"):
            _assert_failed_cycle(
                control, RuntimeError("injected dream failure"), suppress_failure=True,
            )


def test_scheduler_survives_lease_loss_and_runs_a_later_kick(monkeypatch, caplog):
    error = DreamLeaseLost("dreaming lease ownership lost")
    with _controlled_scheduler(monkeypatch) as control:
        control.scheduler.kick()
        control.expect("dream_started", 1)
        control.actions.put(error)
        control.expect("dream_raised", error)
        control.expect("waiting_for_kick", (0, False, 1))
        assert "dream_scheduler.cycle_aborted_lease_lost" in caplog.messages
        assert "dream_scheduler.cycle_failed" not in caplog.messages
        control.scheduler.kick()
        control.expect("dream_started", 2)
        control.finish_cycle(2, completed=1, invalidations=2)
        assert control.started == control.invalidations == 2
        assert control.scheduler.cycles_completed == 1
        assert control.scheduler.is_running


def test_stop_timeout_keeps_live_worker_and_fork_owned_until_retry(monkeypatch):
    with _controlled_scheduler(monkeypatch) as control:
        sched = control.scheduler
        sched.kick()
        control.expect("dream_started", 1)
        # This timeout is the lifecycle behavior under test, not a dream SLO.
        with pytest.raises(TimeoutError, match="did not stop"):
            sched.stop(timeout=0.01)
        assert sched.is_running
        assert sched.shutdown_pending
        assert control.closes == control.invalidations == 0
        control.actions.put(None)
        control.expect("dream_returned", 1)
        sched.stop(timeout=_COORDINATION_TIMEOUT)
        assert not sched.is_running
        assert not sched.shutdown_pending
        assert sched.cycles_completed == control.invalidations == control.closes == 1
        assert control.lifecycle_events == [
            "dream:return", "root:invalidate", "fork:close",
        ]


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
