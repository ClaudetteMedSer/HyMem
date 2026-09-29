"""Network-free worker checks using a private in-memory run ledger."""
from __future__ import annotations

import importlib.util
import io
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from hymem.deadline import MonotonicDeadline


HELPER = Path(__file__).resolve().parents[1] / "claim_conflict_production_dream.py"


@dataclass
class _Report:
    skipped_locked: bool = False
    chunks_processed: int = 1


class _MemoryHyMem:
    def __init__(self, worker):
        self.worker = worker
        self._phase1_generation = {"generation_key": worker.GENERATION}
        self._llm = object()
        self.deadlines = []
        self.later_run_skipped = None
        self.active = False
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE dream_runs (
                id INTEGER PRIMARY KEY, started_at TEXT, ended_at TEXT,
                error TEXT, skipped_locked INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE current_phase1_publications (
                chunk_id TEXT, phase1_generation_key TEXT
            );
            INSERT INTO dream_runs(id, started_at, ended_at)
                VALUES (99, 'earlier-start', 'earlier-end');
        """)

    def dream_status(self):
        return {"in_progress": self.active, "pending_chunks": 1}

    def dream(self, *, deadline):
        self.deadlines.append(deadline)
        self.conn.execute(
            "INSERT INTO dream_runs(id,started_at,ended_at) "
            "VALUES (100,'owned-start','owned-end')"
        )
        if self.later_run_skipped is not None:
            self.conn.execute(
                "INSERT INTO dream_runs(id,started_at,ended_at,skipped_locked) "
                "VALUES (101,'other-start','other-end',?)",
                (int(self.later_run_skipped),),
            )
        self.conn.execute(
            "INSERT INTO current_phase1_publications VALUES (?,?)",
            (self.worker.TARGET, self.worker.GENERATION),
        )
        return _Report()


@pytest.fixture
def worker_harness(monkeypatch):
    spec = importlib.util.spec_from_file_location("claim_dream_test_worker", HELPER)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    # The deployed helper deliberately adds its remote release to sys.path.
    # Restore that import side effect in this local test process.
    original_path = sys.path[:]
    try:
        spec.loader.exec_module(worker)
    finally:
        sys.path[:] = original_path

    hy = _MemoryHyMem(worker)
    bootstrap = ModuleType("hymem.bootstrap")
    bootstrap.build_from_env = Mock(return_value=hy)
    bootstrap.shutdown_instance = Mock(return_value=True)
    strictness = ModuleType("benchmarks.strictness")
    strictness.usage_snapshot = Mock(return_value={"calls": 1})
    monkeypatch.setitem(sys.modules, "hymem.bootstrap", bootstrap)
    monkeypatch.setitem(sys.modules, "benchmarks.strictness", strictness)
    monkeypatch.setattr(worker, "verify", Mock())
    monkeypatch.setattr(worker, "save", Mock())
    monkeypatch.setattr(worker.logging, "disable", Mock())
    monkeypatch.setattr(
        sys, "stdin",
        SimpleNamespace(buffer=io.BytesIO(b"approved-production-dream-v1")),
    )
    harness = SimpleNamespace(
        worker=worker, hy=hy, bootstrap=bootstrap, strictness=strictness,
    )
    try:
        yield harness
    finally:
        hy.conn.close()


def _result(harness):
    harness.worker.save.assert_called_once()
    name, value = harness.worker.save.call_args.args
    assert name == "dream-result.json"
    return value


def test_normal_worker_calls_dream_once_with_bounded_deadline(worker_harness):
    h = worker_harness
    assert h.worker.worker() == 0
    assert len(h.hy.deadlines) == 1
    deadline = h.hy.deadlines[0]
    assert isinstance(deadline, MonotonicDeadline)
    assert 0 < deadline.remaining() <= 1800
    h.bootstrap.build_from_env.assert_called_once_with()
    h.bootstrap.shutdown_instance.assert_called_once_with(h.hy)
    result = _result(h)
    assert result["status"] == "completed"
    assert result["cleanup_ok"] is True
    assert result["run"]["id"] == 100
    assert result["target_current_publications"] == 1


def test_wrong_authorization_never_opens_store_or_calls_dream(
    worker_harness, monkeypatch,
):
    h = worker_harness
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"wrong")))
    with pytest.raises(RuntimeError, match="production_dream_authorization_missing"):
        h.worker.worker()
    h.worker.verify.assert_not_called()
    h.bootstrap.build_from_env.assert_not_called()
    h.bootstrap.shutdown_instance.assert_not_called()
    assert h.hy.deadlines == []
    h.worker.save.assert_not_called()


def test_wrong_generation_closes_instance_without_calling_dream(worker_harness):
    h = worker_harness
    h.hy._phase1_generation = {"generation_key": "different-generation"}
    assert h.worker.worker() == 1
    assert h.hy.deadlines == []
    h.bootstrap.shutdown_instance.assert_called_once_with(h.hy)
    result = _result(h)
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["cleanup_ok"] is True


def test_later_skipped_run_does_not_replace_owned_run(worker_harness):
    h = worker_harness
    h.hy.later_run_skipped = True
    assert h.worker.worker() == 0
    assert len(h.hy.deadlines) == 1
    assert h.hy.conn.execute("SELECT MAX(id) FROM dream_runs").fetchone()[0] == 101
    result = _result(h)
    assert result["status"] == "completed"
    assert result["run"]["id"] == 100
    assert result["run"]["started_at"] == "owned-start"


def test_two_new_nonskipped_runs_fail_ownership_verification(worker_harness):
    h = worker_harness
    h.hy.later_run_skipped = False
    assert h.worker.worker() == 1
    assert len(h.hy.deadlines) == 1
    h.bootstrap.shutdown_instance.assert_called_once_with(h.hy)
    result = _result(h)
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert "run" not in result


def test_usage_failure_cannot_bypass_shutdown(worker_harness):
    h = worker_harness
    h.strictness.usage_snapshot.side_effect = RuntimeError("private provider detail")
    assert h.worker.worker() == 1
    assert len(h.hy.deadlines) == 1
    h.bootstrap.shutdown_instance.assert_called_once_with(h.hy)
    result = _result(h)
    assert result["status"] == "failed"
    assert result["usage_error_type"] == "RuntimeError"
    assert result["cleanup_ok"] is True
    assert "private provider detail" not in repr(result)


def test_existing_active_dream_prevents_another_call(worker_harness):
    h = worker_harness
    h.hy.active = True
    assert h.worker.worker() == 1
    assert h.hy.deadlines == []
    h.bootstrap.shutdown_instance.assert_called_once_with(h.hy)
    assert _result(h)["status"] == "failed"
