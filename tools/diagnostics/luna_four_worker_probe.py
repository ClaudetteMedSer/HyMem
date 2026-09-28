"""Four invented-text calls only; independent live transport/cleanup check."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import sys

PIN = "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0"
BASE_PIN = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
BINARY = "/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest for name, digest in (
            ("codex_subscription_concurrent_v2.py", PIN), ("codex_subscription.py", BASE_PIN))):
        raise RuntimeError("pin_invalid")
    sys.path.insert(0, args.candidate)
    spec = importlib.util.spec_from_file_location("four_worker_subject", root / "codex_subscription_concurrent_v2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from hymem.extraction.llm import LLMRequest
    sessions = []
    intervals = []
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=30)
    real_turn = module.base._run_turn
    def turn(*args):
        barrier.wait()
        started = time.monotonic()
        try:
            return real_turn(*args)
        finally:
            with lock:
                intervals.append((started, time.monotonic()))
    module.base._run_turn = turn
    def factory(*args, **kwargs):
        session = module.base.StdioSession(*args, **kwargs)
        with lock:
            sessions.append(session)
        return session
    budget = module.SharedBudget(module.BudgetLimits(4, 40000, 240), max_in_flight=4)
    clients = [module.ConcurrentSubscriptionClient(BINARY, budget, str(i),
        module.BudgetLimits(1, 10000, 230), session_factory=factory) for i in range(4)]
    started = time.monotonic()
    outcomes = []
    def call(index):
        expected = "ORCHID_TEST_" + str(index)
        value = clients[index].complete(LLMRequest(
            "Reply with only the exact text requested, without commentary.",
            "Reply exactly: " + expected, max_tokens=32))
        return value.strip() == expected
    try:
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(call, index) for index in range(4)]
            for future in futures:
                try:
                    outcomes.append(future.result())
                except BaseException:
                    outcomes.append(False)
    finally:
        cleanup_ok = True
        for session in sessions:
            try:
                session.close()
                os.killpg(session.process.pid, 0)
                cleanup_ok = False
            except ProcessLookupError:
                pass
            except BaseException:
                cleanup_ok = False
    overlap = len(intervals) == 4 and max(i[0] for i in intervals) < min(i[1] for i in intervals)
    state = budget.snapshot()
    ok = (outcomes == [True] * 4 and overlap and cleanup_ok and len(sessions) == 4
          and state["turns"] == 4 and state["usage_complete"] and not state["stopped"]
          and state["in_flight"] == state["reserved"] == 0)
    print(json.dumps({"ok": ok, "exact_responses": outcomes, "overlap_four": overlap,
        "cleanup_ok": cleanup_ok, "session_count": len(sessions),
        "wall_seconds": time.monotonic() - started, "budget": state,
        "benchmark": False, "internal_http_attempts": None}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(json.dumps({"ok": False, "code": "probe_failure"}))
        raise SystemExit(1)
