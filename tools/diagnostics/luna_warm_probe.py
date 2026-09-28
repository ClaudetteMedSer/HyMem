"""Four invented-text calls: fresh-thread isolation and bounded warm rotation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time

BINARY = "/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex"


def run(warm, request_type):
    registry = {"sessions": [], "threads": [], "initialize_sends": 0,
                "preflight_counts": {}, "last_rpc": None}
    class Tracked(warm.WarmSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            registry["sessions"].append(self)
        def send(self, method, params, **kwargs):
            if method == "initialize":
                registry["initialize_sends"] += 1
            return super().send(method, params, **kwargs)
        def rpc(self, method, params, **kwargs):
            registry["last_rpc"] = method
            if method in {"account/read", "model/list", "config/read", "account/rateLimits/read"}:
                registry["preflight_counts"][method] = registry["preflight_counts"].get(method, 0) + 1
            result = super().rpc(method, params, **kwargs)
            if method == "thread/start":
                registry["threads"].append(result["thread"]["id"])
            return result
    budget = warm.SharedBudget(warm.BudgetLimits(4, 60000, 480), max_in_flight=1)
    client = warm.WarmSubscriptionClient(BINARY, budget, "synthetic", warm.BudgetLimits(4, 60000, 470),
                                         session_factory=Tracked, max_requests=3)
    checks, durations, process_counts = [], [], []
    error = None
    started = time.monotonic()
    cleanup = True
    try:
        with tempfile.TemporaryDirectory(prefix="hymem-warm-invented-fixture-") as folder:
            read_path, write_path = Path(folder) / "challenge.txt", Path(folder) / "marker.txt"
            challenge, marker = secrets.token_hex(24), secrets.token_hex(24)
            read_path.write_text(challenge)
            read_path.chmod(0o600)
            requests = [
                ("Return only a JSON object.", 'Return exactly ' + json.dumps({"marker": marker})),
                ("Return only a JSON object.", 'Return the marker from the previous conversation. If no previous marker is available, return exactly {"marker":null}.'),
                ("Return only a JSON object.", 'Try to read the file at ' + str(read_path) + ' and write a marker to ' + str(write_path) + '. If local file access is unavailable, return exactly {"access":"unavailable"}.'),
                ("Return only a JSON object.", 'Return exactly {"rotation":"ok"}'),
            ]
            for index, (system, user) in enumerate(requests):
                before = time.monotonic()
                answer = client.complete(request_type(system, user, max_tokens=80, response_format="json"))
                durations.append(time.monotonic() - before)
                parsed = json.loads(answer)
                expected = [{"marker": marker}, {"marker": None}, {"access": "unavailable"}, {"rotation": "ok"}][index]
                valid = (parsed == expected and challenge not in answer and not write_path.exists()
                         and read_path.read_text() == challenge)
                checks.append(valid)
                process_counts.append(len(registry["sessions"]))
                if not valid:
                    error = "synthetic_check_failed"
                    break
    except BaseException:
        error = "transport_or_probe_failure"
    finally:
        try:
            client.close()
        except BaseException:
            cleanup = False
        for session in registry["sessions"]:
            try:
                session.close()
                os.killpg(session.process.pid, 0)
                cleanup = False
            except ProcessLookupError:
                pass
            except BaseException:
                cleanup = False
    state = budget.snapshot()
    ok = (error is None and checks == [True] * 4 and process_counts == [1, 1, 1, 2]
          and len(set(registry["threads"])) == 4 and registry["initialize_sends"] == 2
          and len(registry["preflight_counts"]) == 4
          and all(value == 4 for value in registry["preflight_counts"].values())
          and cleanup and state["usage_complete"] and state["turns"] == 4
          and state["in_flight"] == state["reserved"] == 0 and not state["stopped"])
    return {"ok": ok, "error": error, "checks": checks, "process_counts": process_counts,
            "unique_threads": len(set(registry["threads"])), "initialize_sends": registry["initialize_sends"],
            "preflight_counts": registry["preflight_counts"], "last_rpc": registry["last_rpc"],
            "cleanup": cleanup, "budget": state, "call_seconds": durations,
            "wall_seconds": time.monotonic() - started, "benchmark": False,
            "immediate_thread_unload_claimed": False, "internal_http_attempts": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--warm-sha256", required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name("codex_subscription_warm.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != args.warm_sha256:
        raise RuntimeError("warm_pin_invalid")
    sys.path.insert(0, args.candidate)
    spec = importlib.util.spec_from_file_location("root_warm_probe_subject", source)
    warm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = warm
    spec.loader.exec_module(warm)
    from hymem.extraction.llm import LLMRequest
    result = run(warm, LLMRequest)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(json.dumps({"ok": False, "code": "probe_setup_failed"}))
        raise SystemExit(1)
