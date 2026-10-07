"""Offline control for the frozen pilot's question-side extraction chain."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


BUNDLE = Path("/private/tmp/hymem-lme-instrumented-IjmZdT/bundle")


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="frozen source bundle unavailable")
def test_successful_transport_can_become_chunk_call_failure_after_return():
    child = r'''
import importlib.util, json, sys
from pathlib import Path

bundle = Path(sys.argv[1])
sys.path[:0] = [str(bundle / "candidate"), str(bundle / "code")]
from hymem.extraction import chunk
from hymem.dreaming.runner import _CountingPhase1LLM, _HeartbeatLLMClient
from benchmarks.codex_subscription_warm_v9 import SharedBudget, BudgetLimits

spec = importlib.util.spec_from_file_location(
    "frozen_runner", bundle / "code/tools/diagnostics/luna_lme_diagnostic_v8.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
limits = BudgetLimits(turns=100, known_tokens=1000, seconds=1000)

def exercise(fail_after):
    budget = SharedBudget(limits, max_in_flight=1)
    budget.register("q-0001", limits)

    class Dual:
        key = "q-0001"

        def __init__(self):
            self.budget = budget
            self.calls = 0

        def complete(self, request):
            budget.reserve(self.key)
            budget.before_turn(self.key, {
                "auth": "chatgpt", "model": "gpt-6-luna",
                "config_isolation_admitted": True, "inference_enabled": False,
                "quota_windows": [{"remaining_percent": 100}],
            })
            self.calls += 1
            budget.settle(self.key, used=10, turn_started=True)
            return '{"triples":[],"markers":[],"complete":true}'

    dual = Dual()
    accounted = runner.AccountedClient(dual, bundle / "candidate")
    memory = runner._memory_client({}, accounted)
    beats = [0]

    def heartbeat():
        beats[0] += 1
        if fail_after and beats[0] == 2:
            raise RuntimeError("synthetic_after_return")

    counting = _CountingPhase1LLM(_HeartbeatLLMClient(memory, heartbeat))
    result = chunk.extract_chunk(
        counting, "Alice went to Paris.", completion_call_limit=2)
    return {
        "failed": result.failed, "reason": result.failure_reason,
        "calls": dual.calls, "beats": beats[0],
        "accounting": accounted.counts, "reconcile": accounted.reconcile(),
        "usage_complete": budget.snapshot()["usage_complete"],
        "stop": budget.stop_code,
    }

print(json.dumps({"healthy": exercise(False), "post_return": exercise(True)}))
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", child, str(BUNDLE)],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    import json

    state = json.loads(result.stdout.strip())
    healthy = state["healthy"]
    assert healthy["failed"] is False and healthy["calls"] == 2
    assert healthy["reconcile"] and healthy["usage_complete"]
    assert healthy["stop"] is None
    fault = state["post_return"]
    assert fault["failed"] is True and fault["reason"] == "call_failure"
    assert fault["calls"] == 1 and fault["beats"] == 2
    assert fault["reconcile"] and fault["usage_complete"]
    assert fault["stop"] is None
    assert fault["accounting"]["extraction"] == {
        "attempts": 1, "returned": 1, "turns": 1, "known_tokens": 10,
    }
