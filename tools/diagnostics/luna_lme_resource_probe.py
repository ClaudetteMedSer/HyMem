"""One-shot, bounded four-worker LME resource diagnostic on the pinned bundle.

The accepted profiled runner owns prompts, admission, isolation, quota checks,
accounting, and cleanup. This observer only samples RPC responses and cgroup
task counters; the first failed response is preserved privately before cleanup.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time


SCHEMA = "luna-lme-resource-probe-v1"
BUNDLE = Path("/home/atta/.hymem-luna-lme-attributed-q8mvj98i")
HOST = Path("/home/atta")
ROOT_PREFIX = ".hymem-luna-lme-resource-"
PINS = {
    "luna_subscription_lme_profiled_v2.py": "53628ac7e9c6107bb68d1cd4ebdf42d5a129b96c4d96360d4e7a6525be7bc739",
    "luna_subscription_lme_warm_v2.py": "3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567",
    "luna_stage_accounting.py": "800ef9baedc9d68093b3160cc324ef17528dd0b89f7a734f220217b07323fba2",
    "codex_subscription_warm_v2.py": "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593",
    "codex_subscription_concurrent_v2.py": "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0",
    "luna_subscription_pilot.py": "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0",
    "codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
    "headless-source-map.json": "852758cfa63902048669f78f2377db4354b2e196a16ccbb98a3e6cadca2590eb",
}
DATASET = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json")
DATASET_SHA = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
CANDIDATE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-r9-full-suite-v1/candidate")
BINARY = Path("/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex")
LIMITS = {
    "campaign-turns": 176, "campaign-known-tokens": 1_500_000,
    "campaign-seconds": 900, "question-turns": 80,
    "question-known-tokens": 600_000, "question-seconds": 800,
    "canary-turns": 12, "canary-known-tokens": 160_000,
    "canary-seconds": 600, "indexing-seconds": 750,
    "questions": 4, "workers": 4,
    "warm-max-requests": 16, "warm-max-age-seconds": 300,
}
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


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def public_error(event):
    if not isinstance(event, dict):
        return {"response_shape": "invalid"}
    error = event.get("error")
    result = {"response_shape": ("error" if isinstance(error, dict) else
                                 "result" if isinstance(event.get("result"), dict) else "other")}
    if not isinstance(error, dict):
        return result
    result["error_keys"] = sorted(key for key in error if key in {"code", "message", "data"})
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


def private_error(event):
    """Bound the first raw failure; no raw text enters public files or stdout."""
    if not isinstance(event, dict):
        return {"response_type": type(event).__name__[:40]}
    error = event.get("error")
    if not isinstance(error, dict):
        result = event.get("result")
        return {"response_shape": public_error(event)["response_shape"],
                "raw_result": json.dumps(result, ensure_ascii=False, default=str)[:4096]}
    value = {}
    for key in ("code", "message", "data"):
        item = error.get(key)
        if key == "code" and type(item) is int and -1_000_000 <= item <= 1_000_000:
            value[key] = item
        elif key == "message" and type(item) is str:
            value[key] = item[:2048]
        elif key == "data" and item is not None:
            value[key] = json.dumps(item, ensure_ascii=False, default=str)[:4096]
    return value


def cgroup_tasks():
    try:
        rows = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(row.split(":", 2)[2] for row in rows if row.startswith("0::"))
        root = Path("/sys/fs/cgroup") / relative.lstrip("/")
        if not root.resolve().is_relative_to(Path("/sys/fs/cgroup")):
            return None
        result = {}
        for name in ("pids.current", "pids.peak", "pids.max"):
            raw = (root / name).read_text().strip()
            result[name.replace(".", "_")] = int(raw) if raw.isdecimal() else None
        rows = (root / "pids.events").read_text().splitlines()
        result["pids_events_max"] = next(int(row.split()[1]) for row in rows if row.startswith("max "))
        rows = Path("/proc/self/status").read_text().splitlines()
        result["controller_threads"] = next(int(row.split()[1]) for row in rows if row.startswith("Threads:"))
        return result
    except (OSError, ValueError, StopIteration):
        return None


def write_once(path: Path, value):
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def verify_root(root: Path, bundle: Path):
    if (sys.platform != "linux" or os.getuid() != 1000 or
            socket.gethostname().split(".", 1)[0].casefold() != "afrodite"):
        raise ValueError("host_invalid")
    if (bundle != BUNDLE or not bundle.is_dir() or bundle.is_symlink() or
            not root.is_absolute() or root.parent != HOST or
            not root.name.startswith(ROOT_PREFIX) or root.is_symlink() or
            not root.is_dir() or root.stat().st_uid != os.getuid() or
            root.stat().st_mode & 0o077 or any(root.iterdir())):
        raise ValueError("root_invalid")
    for name, expected in PINS.items():
        path = bundle / name
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError("source_pin_invalid")
    if not DATASET.is_file() or digest(DATASET) != DATASET_SHA or not BINARY.is_file():
        raise ValueError("input_pin_invalid")
    receipt = json.loads((bundle / "launch-receipt.json").read_text())
    if (receipt.get("source_sha256") != PINS or receipt.get("dataset_sha256") != DATASET_SHA
            or receipt.get("binary_sha256") != digest(BINARY)
            or receipt.get("candidate") != str(CANDIDATE)):
        raise ValueError("receipt_invalid")


class ResourceObserver:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.responses = 0
        self.peak = {}
        self.first = None
        self.started = time.monotonic()

    def observe(self, session, event):
        if not isinstance(event, dict) or "id" not in event:
            return
        counts = cgroup_tasks()
        failed = "error" in event or not isinstance(event.get("result"), dict)
        with self.lock:
            self.responses += 1
            if counts:
                for key, value in counts.items():
                    if type(value) is int:
                        self.peak[key] = max(value, self.peak.get(key, value))
            if failed and self.first is None:
                method = session.stage if session.stage in {
                    "initialize", "config/read", "model/list", "account/read",
                    "account/rateLimits/read", "thread/start", "turn/start", "thread/unsubscribe"} else "other"
                self.first = {"method": method, "response": public_error(event),
                              "tasks": counts, "rpc_responses_before_failure": self.responses,
                              "elapsed_seconds": round(time.monotonic() - self.started, 3),
                              "process_age_seconds": round(time.monotonic() - session.created_at, 3)}
                write_once(self.root / "private-first-rpc-error.json", {
                    "schema": SCHEMA, "method": method, "error": private_error(event)})
                write_once(self.root / "public-first-rpc-error.json", {
                    "schema": SCHEMA, **self.first})

    def report(self):
        with self.lock:
            return {"schema": SCHEMA, "rpc_responses": self.responses,
                    "max_observed": dict(self.peak), "first_rpc_failure": self.first,
                    "final_tasks": cgroup_tasks()}


def instrument(warm, observer):
    original = warm.WarmSession.receive
    def receive(session):
        event = original(session)
        observer.observe(session, event)
        return event
    warm.WarmSession.receive = receive


def runner_args(bundle: Path, root: Path):
    options = {"binary": BINARY, "base-transport": bundle / "codex_subscription.py",
        "concurrent-transport": bundle / "codex_subscription_concurrent_v2.py",
        "warm-transport": bundle / "codex_subscription_warm_v2.py",
        "candidate": CANDIDATE, "inventory-stamp": bundle / "headless-source-map.json",
        "inventory-sha256": PINS["headless-source-map.json"], "dataset": DATASET,
        "dataset-sha256": DATASET_SHA, "output-dir": root / "run", **LIMITS}
    return [part for key, value in options.items() for part in ("--" + key, str(value))]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verify_root(args.output_root, args.bundle_root)
        wrapper_sha = digest(Path(__file__))
        write_once(args.output_root / "probe-started.json", {"schema": SCHEMA,
            "wrapper_sha256": wrapper_sha, "bundle_root": str(args.bundle_root),
            "limits": LIMITS, "one_shot": True})
        spec = importlib.util.spec_from_file_location("pinned_resource_profiled", args.bundle_root /
            "luna_subscription_lme_profiled_v2.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        observer = ResourceObserver(args.output_root)
        original_load = module.warm_runner.load_verified
        def load_verified(**kwargs):
            loaded = original_load(**kwargs)
            if loaded[0] != 508:
                raise ValueError("inventory_incomplete")
            instrument(loaded[-1], observer)
            return loaded
        module.warm_runner.load_verified = load_verified
        module.warm_runner.RUNNER_SHA256 = wrapper_sha
        module.warm_runner.SCHEMA = SCHEMA
        status = module.main(runner_args(args.bundle_root, args.output_root))
        write_once(args.output_root / "public-resource-summary.json", {
            **observer.report(), "wrapper_sha256": wrapper_sha,
            "accepted_profiled_sha256": PINS["luna_subscription_lme_profiled_v2.py"],
            "runner_exit_code": status, "diagnostic_budget_expected": True})
        return status
    except BaseException as exc:
        code = exc.args[0] if type(exc) is ValueError and exc.args and type(exc.args[0]) is str and exc.args[0] in {
            "host_invalid", "root_invalid", "source_pin_invalid", "input_pin_invalid",
            "receipt_invalid", "inventory_incomplete"} else "probe_setup_or_runtime_failure"
        print(json.dumps({"schema": SCHEMA, "ok": False, "setup_code": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
