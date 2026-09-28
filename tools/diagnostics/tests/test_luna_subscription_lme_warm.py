"""Offline runner lifecycle checks; no provider calls."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from tools.diagnostics.tests.test_luna_subscription_lme_multi_v2 import FakeBudget

SOURCE = Path(__file__).resolve().parents[1] / "luna_subscription_lme_warm.py"
SPEC = importlib.util.spec_from_file_location("warm_runner_test_subject", SOURCE)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Client:
    def __init__(self, budget, key, limits, events, *, fail_close=False):
        self.budget, self.key, self.events = budget, key, events
        self.fail_close = fail_close
        budget.register(key, limits)
    @property
    def usage_complete(self):
        return True
    def complete(self, request):
        self.budget.questions[self.key]["turns"] += 1
        self.budget.questions[self.key]["known_tokens"] += 5
        return "{}"
    def chat(self, messages, **kwargs):
        return self.complete(SimpleNamespace())
    def close(self):
        self.events.append("close:" + self.key)
        if self.fail_close:
            raise RuntimeError("controlled cleanup failure")


class Adapter:
    def __init__(self, path, **kwargs):
        self.path = path
        self.last_indexing_summary = None
    def open(self):
        return self
    def close(self):
        EVENTS.append("adapter:" + self.path.parent.name)


EVENTS = []


def run(tmp_path, monkeypatch, *, fail_key=None, canary_pass=True):
    EVENTS.clear()
    def factory(key, limits, budget):
        EVENTS.append("create:" + key)
        return Client(budget, key, limits, EVENTS, fail_close=key == fail_key)
    def canary(*args, **kwargs):
        EVENTS.append("canary")
        return {"passed": canary_pass}
    def evaluate(reader, judge, adapter, question, **kwargs):
        EVENTS.append("evaluate:" + adapter.path.parent.name)
        reader.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
        indexing = {"outcome": "success", "healthy": True, "summary_healthy": True,
                    "final_status": {"pending_chunks": 0}}
        adapter.last_indexing_summary = indexing
        return {"indexing": indexing, "benchmark_failure": None,
                "judge_error": False, "judge_parse_valid": True, "correct": True}
    lme = SimpleNamespace(HyMemAdapter=Adapter, evaluate_question=evaluate,
        IndexingConvergenceError=ValueError, DEFAULT_MAX_INPUT_TOKENS=100,
        DEFAULT_MAX_INPUT_BYTES=1000)
    monkeypatch.setattr(runner.old, "experimental_canary", canary)
    monkeypatch.setattr(runner.old, "make_adapter_class", lambda lme, memory: Adapter)
    monkeypatch.setattr(runner, "make_memory_client", lambda client, **kwargs: client)
    monkeypatch.setattr(runner, "terminalize_owned_dream_run", lambda *a, **kw:
        {"open_runs_before": 0, "terminalized": 0, "open_runs_after": 0,
         "active_leases_after": 0})
    limits = SimpleNamespace(seconds=10000)
    return runner.run_campaign(
        concurrent=SimpleNamespace(SharedBudget=FakeBudget,
            ConcurrentStop=type("ConcurrentStop", (BaseException,), {})),
        request_type=lambda **x: SimpleNamespace(**x), canary=object(), chunk=object(),
        lme=lme, protocol=SimpleNamespace(_validate_versioned_indexing=lambda *a, **kw: True),
        binary="/no-binary", questions=[{"index": 0}], output=tmp_path,
        campaign_limits=SimpleNamespace(seconds=20000), question_limits=limits,
        canary_limits=limits, indexing_timeout_s=3600, workers=1,
        client_factory=factory)


def test_canary_and_worker_close_before_terminal_result(tmp_path, monkeypatch):
    result = run(tmp_path, monkeypatch)
    assert result["campaign_stop"] is None
    assert result["questions"][0]["question_completed"]
    assert EVENTS.index("close:canary") < EVENTS.index("create:q-0000")
    assert EVENTS.index("adapter:q-0000") < EVENTS.index("close:q-0000")
    assert result["active_invocations"] == 0


def test_worker_close_failure_invalidates_score(tmp_path, monkeypatch):
    result = run(tmp_path, monkeypatch, fail_key="q-0000")
    assert result["campaign_stop"] == "warm_client_cleanup_failure"
    assert result["questions"][0]["question_completed"] is False
    assert result["questions"][0]["correct"] is None
    assert result["questions"][0]["cleanup_ok"] is False


def test_failed_canary_still_closes_before_return(tmp_path, monkeypatch):
    result = run(tmp_path, monkeypatch, canary_pass=False)
    assert result["campaign_stop"] == "canary_failed"
    assert "close:canary" in EVENTS
    assert "create:q-0000" not in EVENTS
