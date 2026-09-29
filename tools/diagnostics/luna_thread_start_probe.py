"""One-shot, no-inference diagnostic for the pinned warm thread/start path.

Run only inside a bounded Afrodite systemd unit. Every attempt executes the
complete pinned preflight; successful threads are unsubscribed. No turn starts.
The public result contains classifications only. Bounded raw RPC error JSON is
retained in a mode-0600 file under the operator's private remote root.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import threading
import time
import types

BASE_SHA256 = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
CONCURRENT_SHA256 = "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0"
WARM_SHA256 = "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593"
MAX_WORKERS = 4
MAX_STARTS = 128
MAX_PER_PROCESS = 16
MAX_PROCESS_AGE = 300
MAX_WALL_SECONDS = 300
MAX_INVOCATION_SECONDS = 120
RESULT_NAME = "luna-thread-start-public.json"
PRIVATE_NAME = "luna-thread-start-private.json"
MARKER_NAME = "luna-thread-start-started.json"
PREFLIGHT_METHODS = ("account/read", "model/list", "account/rateLimits/read", "config/read")
RPC_METHODS = frozenset((*PREFLIGHT_METHODS, "initialize", "thread/start", "thread/unsubscribe"))
ERROR_KEYS = frozenset({"code", "message", "data"})
ERROR_CLASSES = (
    ("invalid_request", ("invalid params", "invalid request", "invalid type", "missing field")),
    ("resource_limit", ("resource exhausted", "too many open files", "too many threads", "task limit",
                        "resource temporarily unavailable", "os error 11")),
    ("spawn_failure", ("failed to spawn", "failed to create thread")),
    ("memory_allocation", ("cannot allocate memory", "memory allocation failed", "out of memory")),
    ("database_lock", ("database is locked", "database table is locked")),
    ("cwd_missing", ("no such file or directory", "working directory does not exist")),
    ("rate_limit", ("rate limit", "too many requests")),
    ("auth", ("unauthorized", "authentication required", "forbidden")),
    ("internal", ("internal error", "internal server error")),
)


def _regular_absolute(path: Path) -> bool:
    return (path.is_absolute() and not path.is_symlink() and path.is_file()
            and stat.S_ISREG(path.stat().st_mode))


def load_verified(candidate: Path, binary: Path, warm_path: Path):
    if (not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir()
            or not _regular_absolute(binary) or not _regular_absolute(warm_path)
            or warm_path.name != "codex_subscription_warm_v2.py"):
        raise ValueError("input_invalid")
    for path, digest in ((warm_path, WARM_SHA256),
                         (warm_path.with_name("codex_subscription.py"), BASE_SHA256),
                         (warm_path.with_name("codex_subscription_concurrent_v2.py"), CONCURRENT_SHA256)):
        if not _regular_absolute(path) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("transport_pin_invalid")
    for name, module in tuple(sys.modules.items()):
        if name == "hymem" or name.startswith("hymem."):
            source = getattr(module, "__file__", None)
            if source is not None and not Path(source).resolve().is_relative_to(candidate.resolve()):
                raise ValueError("cached_source_mismatch")
    sys.path.insert(0, str(candidate))
    # Match the frozen LME runner's Python import footprint before subprocesses
    # are opened. These imports perform no inference.
    from benchmarks import extraction_canary, longmemeval_adapter, lme_protocol
    from hymem.extraction import chunk, prompts
    from hymem.extraction.llm import LLMRequest
    loaded = (extraction_canary, longmemeval_adapter, lme_protocol, chunk, prompts,
              sys.modules[LLMRequest.__module__])
    if any(not Path(module.__file__).resolve().is_relative_to(candidate.resolve())
           for module in loaded):
        raise ValueError("loaded_source_mismatch")
    source = warm_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != WARM_SHA256:
        raise ValueError("warm_pin_invalid")
    warm = types.ModuleType("pinned_luna_thread_start_subject")
    warm.__file__ = str(warm_path)
    sys.modules[warm.__name__] = warm
    exec(compile(source, str(warm_path), "exec"), warm.__dict__)
    variants = {
        "empty": "",
        "minimal": "Return only a JSON object.",
        "chunk_empty_verifier": prompts.build_chunk_empty_verification_system(),
        "chunk_extraction": prompts.build_chunk_extraction_system(),
        "chunk_omission_verifier": prompts.build_chunk_omission_verification_system(),
        "session_digest": prompts.SESSION_DIGEST_SYSTEM,
        "session_digest_granular": prompts.SESSION_DIGEST_GRANULAR_SYSTEM,
    }
    if any(type(value) is not str for value in variants.values()):
        raise ValueError("variant_invalid")
    return warm, variants


def _bounded_error(event):
    """Private raw response fragment, bounded recursively and never printed."""
    error = event.get("error") if isinstance(event, dict) else None
    if not isinstance(error, dict):
        return None
    result = {}
    for key in ERROR_KEYS:
        value = error.get(key)
        if key == "code" and type(value) is int and -1_000_000 <= value <= 1_000_000:
            result[key] = value
        elif key == "message" and type(value) is str:
            result[key] = value[:2048]
        elif key == "data" and value is not None:
            result[key] = json.dumps(value, ensure_ascii=False, default=str)[:4096]
    return result


def _public_error(event):
    if not isinstance(event, dict):
        return {"response_shape": "invalid"}
    error = event.get("error")
    result = {"response_shape": ("error" if isinstance(error, dict) else
                                 "result" if isinstance(event.get("result"), dict) else "other")}
    if not isinstance(error, dict):
        return result
    result["error_keys"] = sorted(key for key in error if key in ERROR_KEYS)
    code = error.get("code")
    if type(code) is int and -1_000_000 <= code <= 1_000_000:
        result["rpc_error_code"] = code
    message = error.get("message")
    result["message_bytes"] = min(len(message.encode("utf-8")), 1_000_000) if type(message) is str else None
    data = error.get("data")
    result["data_bytes"] = min(len(json.dumps(data, default=str).encode("utf-8")), 1_000_000) if data is not None else None
    lowered = message.lower() if type(message) is str else ""
    result["message_class"] = next((name for name, patterns in ERROR_CLASSES
                                    if any(pattern in lowered for pattern in patterns)), "other")
    return result


def _cgroup_pids():
    """Read only numeric values from this process's cgroup v2 files."""
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(line.split(":", 2)[2] for line in entries if line.startswith("0::"))
        root = Path("/sys/fs/cgroup").joinpath(relative.lstrip("/"))
        if not root.resolve().is_relative_to(Path("/sys/fs/cgroup")):
            return None
        answer = {}
        for name in ("pids.current", "pids.peak", "pids.max"):
            raw = (root / name).read_text().strip()
            answer[name.replace(".", "_")] = int(raw) if raw.isdecimal() else None
        events = (root / "pids.events").read_text().splitlines()
        answer["pids_events_max"] = next(int(line.split()[1]) for line in events if line.startswith("max "))
        status = Path("/proc/self/status").read_text().splitlines()
        answer["controller_threads"] = next(int(line.split()[1]) for line in status
                                            if line.startswith("Threads:"))
        return answer
    except (OSError, ValueError, StopIteration):
        return None


def _private_root(root: Path):
    parent = Path("/home/atta")
    if (socket.gethostname().split(".", 1)[0].casefold() != "afrodite"
            or not root.is_absolute() or root.parent != parent
            or not root.name.startswith(".hymem-luna-thread-start-")
            or not root.resolve().is_relative_to(parent)
            or root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o077):
        raise ValueError("private_root_invalid")
    if root.stat().st_uid != 1000:
        raise ValueError("private_root_owner_invalid")
    if any((root / name).exists() or (root / name).is_symlink()
           for name in (RESULT_NAME, PRIVATE_NAME, MARKER_NAME)):
        raise ValueError("output_exists")


def _write_once(root: Path, name: str, value):
    target = root / name
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


SETUP_CODES = frozenset({"private_root_invalid", "private_root_owner_invalid",
    "output_exists", "input_invalid", "transport_pin_invalid", "cached_source_mismatch",
    "loaded_source_mismatch", "warm_pin_invalid", "variant_invalid", "limits_invalid"})


def _setup_code(exc: BaseException) -> str:
    """Expose only literal probe-defined errors; never copy exception text."""
    if type(exc) is ValueError and len(exc.args) == 1 and type(exc.args[0]) is str:
        if exc.args[0] in SETUP_CODES:
            return exc.args[0]
    if isinstance(exc, FileExistsError):
        return "output_exists"
    if isinstance(exc, OSError):
        return "output_io_failure"
    return "probe_setup_or_output_failure"


def run(warm, variants: dict[str, str], binary: str, *, workers: int = 4,
        starts: int = 128, wall_seconds: int = 300):
    if (type(workers) is not int or not 1 <= workers <= MAX_WORKERS
            or type(starts) is not int or not workers <= starts <= MAX_STARTS
            or type(wall_seconds) is not int or not 1 <= wall_seconds <= MAX_WALL_SECONDS
            or not variants or len(variants) > 8 or any(
                type(key) is not str or type(value) is not str or len(value.encode("utf-8")) > 100_000
                for key, value in variants.items())):
        raise ValueError("limits_invalid")
    begun = time.monotonic()
    deadline = begun + wall_seconds
    lock = threading.Lock()
    abort = threading.Event()
    slots = list(variants.items())
    rows, private = [], []
    next_index = 0

    def worker(index):
        nonlocal next_index
        session = None
        directory = None
        process_index = 0
        on_process = 0
        records = []
        cleanup_ok = True

        class Tracked(warm.WarmSession):
            def send(self, method, params, **kwargs):
                if method == "turn/start":
                    raise AssertionError("turn_start_forbidden")
                return super().send(method, params, **kwargs)

            def receive(self):
                event = super().receive()
                self.last_response = event if "id" in event else None
                return event

        try:
            while not abort.is_set() and time.monotonic() < deadline:
                with lock:
                    if next_index >= starts:
                        break
                    ordinal = next_index
                    next_index += 1
                variant_id, system = slots[ordinal % len(slots)]
                started = time.monotonic()
                stage = "startup"
                response = None
                code = None
                pids_before = _cgroup_pids()
                if session is not None and (on_process >= MAX_PER_PROCESS
                        or time.monotonic() - session.created_at >= MAX_PROCESS_AGE):
                    try:
                        session.close()
                    finally:
                        session = None
                        directory.cleanup()
                        directory = None
                    on_process = 0
                try:
                    if session is None:
                        directory = tempfile.TemporaryDirectory(prefix="hymem-luna-thread-start-")
                        session = Tracked(binary, directory.name,
                                          timeout=max(0.001, min(deadline, started + MAX_INVOCATION_SECONDS) - time.monotonic()))
                        process_index += 1
                    session.set_deadline(min(deadline, started + MAX_INVOCATION_SECONDS,
                                             session.created_at + MAX_PROCESS_AGE + MAX_INVOCATION_SECONDS))
                    session.last_response = None
                    stage = "preflight"
                    admission = warm.base.inspect_preflight(session, base_instructions=system)
                    thread_id = admission.get("_thread_id")
                    if not isinstance(thread_id, str) or not thread_id:
                        warm.base._fail("thread_id_missing")
                    stage = "unsubscribe"
                    session.unsubscribe(thread_id)
                    on_process += 1
                    stage = "complete"
                except BaseException as exc:
                    response = getattr(session, "last_response", None) if session is not None else None
                    code = warm._safe_code(exc) if isinstance(exc, Exception) else "fixed_other"
                    if type(exc) is AssertionError and exc.args == ("turn_start_forbidden",):
                        code = "turn_start_forbidden"
                    # If admission failed after a valid thread/start, retire
                    # that exact active thread before process cleanup.
                    if session is not None and session.active_thread is not None:
                        try:
                            session.unsubscribe(session.active_thread)
                        except BaseException:
                            pass
                    abort.set()
                pids_after = _cgroup_pids()
                record = {"ordinal": ordinal, "worker": index, "variant": variant_id,
                          "system_bytes": len(system.encode("utf-8")), "stage": stage,
                          "code": code, "process_index": process_index,
                          "request_index": on_process + (0 if stage == "complete" else 1),
                          "retired_count": len(session.retired_threads) if session is not None else 0,
                          "elapsed_seconds": round(time.monotonic() - started, 3),
                          "pids_before": pids_before, "pids_after": pids_after,
                          "rpc": session.stage if session is not None and session.stage in RPC_METHODS else None,
                          "response": _public_error(response) if code is not None else None}
                with lock:
                    records.append(record)
                    if code is not None:
                        private.append({"ordinal": ordinal, "worker": index,
                                        "error": _bounded_error(response)})
                if code is not None:
                    break
        finally:
            if session is not None:
                try:
                    session.close()
                    if session.process.poll() is None:
                        cleanup_ok = False
                except BaseException:
                    cleanup_ok = False
            if directory is not None:
                directory.cleanup()
            with lock:
                rows.append({"worker": index, "cleanup": cleanup_ok,
                             "processes_started": process_index,
                             "attempts": sorted(records, key=lambda item: item["ordinal"])})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, index) for index in range(workers)]
        for future in futures:
            future.result()
    rows.sort(key=lambda item: item["worker"])
    attempts = sorted((item for row in rows for item in row["attempts"]), key=lambda item: item["ordinal"])
    public = {"schema": "luna-thread-start-v1", "ok": len(attempts) == starts and
              all(item["code"] is None for item in attempts) and all(row["cleanup"] for row in rows),
              "inference_enabled": False, "turn_starts": 0, "benchmark": False,
              "can_rule_out_prior_turn_effects": False,
              "attempts": attempts, "workers": [{key: row[key] for key in ("worker", "cleanup", "processes_started")}
                                            for row in rows],
              "wall_seconds": round(time.monotonic() - begun, 3), "pids_final": _cgroup_pids()}
    return public, {"schema": "luna-thread-start-private-v1", "errors": private}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--warm-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--starts", type=int, default=128)
    parser.add_argument("--wall-seconds", type=int, default=300)
    parser.add_argument("--variants", default="empty,minimal,chunk_empty_verifier")
    args = parser.parse_args()
    try:
        _private_root(args.output_root)
        warm, available = load_verified(args.candidate, args.binary, args.warm_path)
        selected = args.variants.split(",")
        if len(selected) != len(set(selected)) or any(key not in available for key in selected):
            raise ValueError("variant_invalid")
        _write_once(args.output_root, MARKER_NAME, {"schema": "luna-thread-start-start-v1",
            "warm_sha256": WARM_SHA256, "variants": selected, "starts": args.starts})
        public, private = run(warm, {key: available[key] for key in selected}, str(args.binary),
                              workers=args.workers, starts=args.starts,
                              wall_seconds=args.wall_seconds)
        _write_once(args.output_root, PRIVATE_NAME, private)
        _write_once(args.output_root, RESULT_NAME, public)
        print(json.dumps(public, sort_keys=True, allow_nan=False))
        return 0 if public["ok"] else 1
    except BaseException as exc:
        print(json.dumps({"schema": "luna-thread-start-v1", "ok": False,
                          "code": _setup_code(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
