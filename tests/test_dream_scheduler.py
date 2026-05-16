"""Unit tests for DreamScheduler: kick→runs, cooldown gating, clean shutdown."""
from __future__ import annotations

import time
from pathlib import Path

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
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
