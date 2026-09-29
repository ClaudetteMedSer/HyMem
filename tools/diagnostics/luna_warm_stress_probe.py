"""Finite four-worker invented-text stress probe for a SHA-pinned warm transport.

This module does not launch anything on import. The operator supplies a previously
verified frozen candidate, binary, exact warm-v2 digest, and private output root.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import threading
import time
import types

BASE_SHA256 = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
CONCURRENT_SHA256 = "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0"
CALLS_PER_WORKER = 40
WORKERS = 4
RESULT_NAME = "luna-warm-stress-result.json"
RUN_MARKER = "luna-warm-stress-started.json"
PREFLIGHT = frozenset({"account/read", "model/list", "config/read", "account/rateLimits/read"})
RPC_METHODS = frozenset({"initialize", "config/read", "model/list", "account/read",
    "account/rateLimits/read", "thread/start", "turn/start", "thread/unsubscribe",
    "startup", "turn/events"})
PHASES = frozenset({"startup", "rotation_cleanup", "preflight", "run",
    "unsubscribe", "cleanup", "rpc", "turn_events"})
NUMERIC_FIELDS = frozenset({"turns", "known_tokens", "in_flight", "reserved",
    "process_index", "request_index", "preflight_seconds", "model_seconds",
    "cleanup_seconds", "startup_seconds", "unsubscribe_seconds",
    "rotation_cleanup_seconds", "final_cleanup_seconds", "event_count",
    "warning_count", "notification_count", "attempt_count",
    "process_age_seconds", "retired_count", "queue_count"})
BOOL_FIELDS = frozenset({"turn_admitted", "known_usage", "usage_complete"})


def _regular_absolute(path: Path) -> bool:
    return path.is_absolute() and not path.is_symlink() and path.is_file() and stat.S_ISREG(path.stat().st_mode)


def load_verified(candidate: Path, binary: Path, warm_path: Path, warm_sha256: str):
    """Verify the flat transport bundle before importing the frozen candidate."""
    if (not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir()
            or not _regular_absolute(binary) or not _regular_absolute(warm_path)
            or warm_path.name != "codex_subscription_warm_v2.py"
            or len(warm_sha256) != 64 or any(c not in "0123456789abcdef" for c in warm_sha256)):
        raise ValueError("input_invalid")
    base_path = warm_path.with_name("codex_subscription.py")
    concurrent_path = warm_path.with_name("codex_subscription_concurrent_v2.py")
    for path, expected in ((base_path, BASE_SHA256), (concurrent_path, CONCURRENT_SHA256),
                           (warm_path, warm_sha256)):
        if not _regular_absolute(path) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("transport_pin_invalid")
    for name, module in tuple(sys.modules.items()):
        if name == "hymem" or name.startswith("hymem."):
            source = getattr(module, "__file__", None)
            if source is not None and not Path(source).resolve().is_relative_to(candidate.resolve()):
                raise ValueError("cached_source_mismatch")
    sys.path.insert(0, str(candidate))
    from hymem.extraction.llm import LLMRequest
    if not Path(sys.modules[LLMRequest.__module__].__file__).resolve().is_relative_to(candidate.resolve()):
        raise ValueError("loaded_source_mismatch")
    # Read and verify again at execution, closing the check/use gap for this file.
    source = warm_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != warm_sha256:
        raise ValueError("warm_pin_invalid")
    warm = types.ModuleType("pinned_luna_warm_stress_subject")
    warm.__file__ = str(warm_path)
    sys.modules[warm.__name__] = warm
    exec(compile(source, str(warm_path), "exec"), warm.__dict__)
    return warm, LLMRequest


def _safe_failure(failure, warm=None):
    """Keep only transport-defined bounded attribution fields and numeric values."""
    if not isinstance(failure, dict):
        return None
    result = {}
    code = failure.get("code")
    if type(code) is str and warm is not None and hasattr(warm, "_safe_code"):
        result["code"] = warm._safe_code(Exception(code))
    for key in ("phase", "route_phase"):
        value = failure.get(key)
        if type(value) is str and value in PHASES:
            result[key] = value
    for key in ("last_rpc", "rpc"):
        value = failure.get(key)
        if type(value) is str and value in RPC_METHODS:
            result[key] = value
    for key, value in failure.items():
        if key in NUMERIC_FIELDS and type(value) in (int, float) and math.isfinite(value):
            result[key] = value
        elif key in BOOL_FIELDS and type(value) is bool:
            result[key] = value
    return result or None


def _public_budget(state, warm):
    stop = state.get("stop_code")
    if type(stop) is str and hasattr(warm, "_safe_code"):
        stop = warm._safe_code(Exception(stop))
    elif stop is not None:
        stop = "fixed_other"
    return {"turns": state["turns"], "known_tokens": state["known_tokens"],
            "usage_complete": state["usage_complete"], "reserved": state["reserved"],
            "in_flight": state["in_flight"], "stopped": state["stopped"],
            "stop_code": stop, "timings": state["timings"],
            "first_failure": _safe_failure(state.get("first_failure"), warm),
            "workers": [{"turns": state["questions"][f"worker-{i}"]["turns"],
                         "known_tokens": state["questions"][f"worker-{i}"]["known_tokens"],
                         "usage_complete": state["questions"][f"worker-{i}"]["usage_complete"]}
                        for i in range(WORKERS)]}


def _peak_overlap(intervals):
    points = sorted([(start, 1) for start, _ in intervals] +
                    [(end, -1) for _, end in intervals])
    active = peak = 0
    for _, delta in points:
        active += delta
        peak = max(peak, active)
    return peak


def _event_metadata(event, session, warm):
    """Bounded classification of one event, with no copied event content or IDs."""
    if not isinstance(event, dict):
        return {"event_method": "invalid_event"}
    method = event.get("method")
    allowed = getattr(warm, "_EVENT_METHODS", frozenset())
    result = {"event_method": method if type(method) is str and method in allowed else
              ("rpc_response" if "id" in event else "other")}
    params = event.get("params")
    params = params if isinstance(params, dict) else {}
    tid = params.get("threadId")
    if method == "thread/started" and isinstance(params.get("thread"), dict):
        tid = params["thread"].get("id")
    result["thread_relation"] = ("missing" if not isinstance(tid, str) else
        "active" if tid == getattr(session, "active_thread", None) else
        "retired" if tid in getattr(session, "retired_threads", set()) else
        "new_unbound" if method == "thread/started" else "other")
    status = params.get("status")
    status = status.get("type") if isinstance(status, dict) else status
    result["lifecycle_status"] = status if type(status) is str and status in {"idle", "notLoaded", "active", "disabled"} else "other"
    if method == "warning":
        message = params.get("message")
        result["warning_kind"] = "other"
        if isinstance(message, str) and hasattr(warm, "base"):
            try:
                warm.base._validate_warning(event, None)
                result["warning_kind"] = ("code_mode_disabled_notice"
                    if message == warm.base._CODE_MODE_DISABLED_NOTICE
                    else "unstable_feature_notice")
            except BaseException:
                pass
    error = event.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    if type(code) is int and -1_000_000 <= code <= 1_000_000:
        result["rpc_error_code"] = code
    return result


def run(warm, request_type, binary: str):
    started = time.monotonic()
    budget = warm.SharedBudget(warm.BudgetLimits(160, 1_200_000, 900), max_in_flight=4)
    lock = threading.Lock()
    abort = threading.Event()
    barrier = threading.Barrier(WORKERS)
    records = []
    intervals = []
    all_threads = set()

    def worker(index):
        registry = {"sessions": [], "threads": set(), "thread_starts": 0,
                    "initialize_sends": 0, "preflight": {method: 0 for method in PREFLIGHT},
                    "last_rpc": None, "fault_event": None, "fault_phase": None}

        class Tracked(warm.WarmSession):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                registry["sessions"].append(self)

            def send(self, method, params, **kwargs):
                if method == "initialize":
                    registry["initialize_sends"] += 1
                return super().send(method, params, **kwargs)

            def receive(self):
                event = super().receive()
                self.last_event_metadata = _event_metadata(event, self, warm)
                return event

            def rpc(self, method, params, **kwargs):
                registry["last_rpc"] = method if method in PREFLIGHT | {"thread/start", "turn/start", "thread/unsubscribe"} else "other"
                if method in PREFLIGHT:
                    registry["preflight"][method] += 1
                try:
                    response = super().rpc(method, params, **kwargs)
                except BaseException:
                    if registry["fault_event"] is None:
                        registry["fault_event"] = getattr(self, "last_event_metadata", None)
                        registry["fault_phase"] = "rpc"
                    raise
                if method == "thread/start":
                    registry["thread_starts"] += 1
                    thread = response.get("thread") if isinstance(response, dict) else None
                    identity = thread.get("id") if isinstance(thread, dict) else None
                    if isinstance(identity, str):
                        registry["threads"].add(identity)
                        with lock:
                            all_threads.add(identity)
                return response

            def next_event(self):
                try:
                    event = super().next_event()
                    self.last_event_metadata = _event_metadata(event, self, warm)
                    return event
                except BaseException:
                    if registry["fault_event"] is None:
                        registry["fault_event"] = getattr(self, "last_event_metadata", None)
                        registry["fault_phase"] = "turn_events"
                    raise

        client = warm.WarmSubscriptionClient(binary, budget, f"worker-{index}",
                    warm.BudgetLimits(40, 300_000, 880), session_factory=Tracked,
                    max_requests=16, max_age_seconds=300)
        marker = secrets.token_hex(12)
        completed = 0
        shape_failure = False
        transport_failure = False
        cleanup = True
        try:
            barrier.wait(timeout=20)
            for call in range(CALLS_PER_WORKER):
                if abort.is_set() or budget.snapshot()["stopped"]:
                    break
                expected = {"marker": marker} if call % 2 == 0 else {"marker": None}
                prompt = ("Return exactly " + json.dumps(expected, separators=(",", ":")) + "."
                          if call % 2 == 0 else
                          'Return the marker from the previous conversation. If no previous marker is available, return exactly {"marker":null}.')
                before = time.monotonic()
                try:
                    answer = client.complete(request_type("Return only a JSON object.", prompt,
                                                           max_tokens=80, response_format="json"))
                except BaseException:
                    transport_failure = True
                    abort.set()
                    break
                finally:
                    after = time.monotonic()
                    with lock:
                        intervals.append((before - started, after - started))
                completed += 1
                try:
                    valid = json.loads(answer) == expected
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    shape_failure = True
                    budget.halt("synthetic_shape_failure")
                    abort.set()
                    break
        except BaseException:
            transport_failure = True
            budget.halt("probe_worker_failure")
            abort.set()
        finally:
            try:
                client.close()
            except BaseException:
                cleanup = False
                budget.halt("cleanup_failure")
                abort.set()
            for session in registry["sessions"]:
                if session.process.poll() is None:
                    cleanup = False
                try:
                    os.killpg(session.process.pid, 0)
                    cleanup = False
                except ProcessLookupError:
                    pass
                except BaseException:
                    cleanup = False
            with lock:
                records.append({"worker": index, "completed": completed,
                    "shape_failure": shape_failure, "transport_failure": transport_failure,
                    "cleanup": cleanup, "processes_started": client.processes_started,
                    "rotations": client.rotations, "sessions": len(registry["sessions"]),
                    "initialize_sends": registry["initialize_sends"],
                    "thread_starts": registry["thread_starts"],
                    "unique_threads": len(registry["threads"]),
                    "preflight_counts": registry["preflight"],
                    "last_rpc": registry["last_rpc"],
                    "fault_event": registry["fault_event"],
                    "fault_phase": registry["fault_phase"],
                    "startup_seconds": client.startup_seconds,
                    "unsubscribe_seconds": client.unsubscribe_seconds,
                    "rotation_cleanup_seconds": client.rotation_cleanup_seconds,
                    "final_cleanup_seconds": client.final_cleanup_seconds})

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(worker, index) for index in range(WORKERS)]
        for future in futures:
            future.result()
    records.sort(key=lambda row: row["worker"])
    state = budget.snapshot()
    overlap = _peak_overlap(intervals)
    ok = (len(records) == WORKERS and all(row["completed"] == CALLS_PER_WORKER
          and not row["shape_failure"] and not row["transport_failure"] and row["cleanup"]
          and row["processes_started"] >= 3 and row["sessions"] == row["processes_started"]
          and row["initialize_sends"] == row["sessions"]
          and row["thread_starts"] == row["unique_threads"] == CALLS_PER_WORKER
          and all(n == CALLS_PER_WORKER for n in row["preflight_counts"].values())
          for row in records) and len(all_threads) == 160
          and overlap == WORKERS and state["turns"] == 160
          and state["reserved"] == state["in_flight"] == 0
          and state["usage_complete"] and not state["stopped"])
    return {"schema": "luna-warm-stress-v1", "ok": ok, "benchmark": False,
            "inference_enabled": True, "workers": records, "peak_overlap": overlap,
            "invocations": len(intervals), "unique_threads": len(all_threads),
            "budget": _public_budget(state, warm),
            "wall_seconds": time.monotonic() - started,
            "internal_http_attempts": None, "immediate_thread_unload_claimed": False}


def atomic_private(root: Path, value):
    if (not root.is_absolute() or root.is_symlink() or not root.is_dir()
            or root.stat().st_mode & 0o077):
        raise ValueError("private_root_invalid")
    target = root / RESULT_NAME
    if target.exists() or target.is_symlink():
        raise ValueError("result_exists")
    pending = root / (RESULT_NAME + ".pending")
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, target)
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def claim_once(root: Path, warm_sha256: str):
    """Durably mark the private root before the first paid request."""
    marker = root / RUN_MARKER
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"schema": "luna-warm-stress-start-v1", "warm_sha256": warm_sha256}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--warm-path", type=Path, required=True)
    parser.add_argument("--warm-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if (not args.output_root.is_absolute() or args.output_root.is_symlink()
                or not args.output_root.is_dir() or args.output_root.stat().st_mode & 0o077
                or (args.output_root / RESULT_NAME).exists()
                or (args.output_root / RUN_MARKER).exists()):
            raise ValueError("private_root_invalid")
        warm, request_type = load_verified(args.candidate, args.binary, args.warm_path,
                                           args.warm_sha256)
        claim_once(args.output_root, args.warm_sha256)
        result = run(warm, request_type, str(args.binary))
        atomic_private(args.output_root, result)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["ok"] else 1
    except BaseException:
        print(json.dumps({"schema": "luna-warm-stress-v1", "ok": False,
                          "code": "probe_setup_or_output_failure"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
