"""One fresh subscription canary; no dataset, LME question, or API-key route."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback


PILOT_SHA256 = "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0"
TRANSPORT_SHA256 = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
MAX_CALLS = 24
WALL_SECONDS = 590
HEX = re.compile(r"[0-9a-f]{64}\Z")


class CanaryStop(RuntimeError):
    pass


def _fail(code: str) -> None:
    raise CanaryStop(code)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _private_json(path: Path, value: object) -> None:
    """Replace one 0600 JSON snapshot durably within a private directory."""
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _journal(path: Path, value: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True).encode() + b"\n"
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _group_absent(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


class SessionTracker:
    """Record only App Server instances this runner actually creates."""
    def __init__(self, inner, journal_path: Path, records: list[dict]):
        self.inner = inner
        self.journal_path = journal_path
        self.records = records
        self.process = inner.process
        self.pid = self.process.pid
        self.record = {"pid": self.pid, "pgid": self.pid, "closed": False,
                       "group_absent": False}
        records.append(self.record)
        try:
            _journal(journal_path, {"event": "app_server_started", "pid": self.pid, "pgid": self.pid})
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            self.inner.close()
        finally:
            self.record["closed"] = self.process.poll() is not None
            self.record["group_absent"] = _group_absent(self.pid)
            _journal(self.journal_path, {"event": "app_server_closed", **self.record})

    def __getattr__(self, name):
        return getattr(self.inner, name)


class BoundedClient:
    """Expose delegate telemetry while journaling each attempted completion."""
    def __init__(self, delegate, output: Path, deadline: float, progress):
        self.delegate = delegate
        self.output = output
        self.deadline = deadline
        self.progress = progress
        self.attempts = 0
        self.in_flight = False

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def complete(self, request):
        if self.attempts >= MAX_CALLS:
            _fail("call_cap")
        if time.monotonic() >= self.deadline:
            _fail("wall_limit")
        self.attempts += 1
        self.in_flight = True
        self.progress("canary", True, self)
        _journal(self.output / "private-invocations.jsonl", {
            "event": "request", "call": self.attempts,
            "request": {"system": request.system, "user": request.user,
                        "response_format": request.response_format,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature}})
        try:
            reply = self.delegate.complete(request)
            _journal(self.output / "private-invocations.jsonl", {
                "event": "response", "call": self.attempts, "text": reply,
                "observed_turns": self.delegate.observed_turns,
                "known_tokens": self.delegate.observed_tokens,
                "usage_complete": self.delegate.usage_complete})
            self.in_flight = False
            return reply
        finally:
            self.progress("canary", self.in_flight, self)


def run_canary(*, pilot, transport, canary, chunk, binary: str,
               output: Path, deadline: float, report: dict) -> dict:
    sessions: list[dict] = []

    def session_factory(path, cwd, timeout=120):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _fail("wall_limit")
        inner = transport.StdioSession(path, cwd, timeout=min(timeout, 120, remaining))
        try:
            return SessionTracker(inner, output / "private-sessions.jsonl", sessions)
        except BaseException:
            inner.close()
            raise

    delegate = transport.CodexSubscriptionClient(binary, session_factory=session_factory)
    def progress(phase, in_flight, client):
        _private_json(output / "private-progress.json", {
            "phase": phase, "in_flight": in_flight,
            "attempted_calls": client.attempts, "observed_turns": delegate.observed_turns,
            "known_tokens": delegate.observed_tokens,
            "usage_complete": delegate.usage_complete,
            "app_server_sessions": len(sessions),
            "app_server_cleanup_complete": all(s["closed"] and s["group_absent"] for s in sessions),
        })
    client = BoundedClient(delegate, output, deadline, progress)
    try:
        progress("preflight", False, client)
        admission = delegate.preflight()
        if (admission.get("config_isolation_admitted") is not True
                or admission.get("auth") != "chatgpt" or admission.get("model") != "gpt-6-luna"
                or admission.get("inference_enabled") is not False):
            _fail("preflight_rejected")
        progress("preflight_passed", False, client)
        delegate.inference_accepted = True
        result = pilot.experimental_canary(canary, chunk, client,
            evidence=lambda value: _private_json(output / "private-canary-evidence.json", value))
        report["canary"] = result
        if result.get("schema") != "luna-experimental-canary-v2" or result.get("passed") is not True:
            _fail("canary_failed")
        if not delegate.usage_complete or delegate.observed_tokens is None:
            _fail("usage_incomplete")
        if client.attempts > MAX_CALLS or delegate.observed_turns != client.attempts:
            _fail("turn_accounting_invalid")
        if time.monotonic() >= deadline:
            _fail("wall_limit")
        report["ok"] = True
        return report
    finally:
        report["ok"] = False if client.in_flight else report["ok"]
        report.update({"attempted_calls": client.attempts,
                       "observed_turns": delegate.observed_turns,
                       "known_tokens": delegate.observed_tokens,
                       "usage_complete": delegate.usage_complete,
                       "in_flight": client.in_flight,
                       "app_server_sessions": len(sessions),
                       "app_server_cleanup_complete": all(
                           s["closed"] and s["group_absent"] for s in sessions)})
        if not report["app_server_cleanup_complete"]:
            report["ok"] = False
            report["stop_code"] = "cleanup_unverified"
        try:
            progress("finished", client.in_flight, client)
        except BaseException:
            report["ok"] = False
            raise


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail("source_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for key in ("binary", "transport", "pilot", "candidate", "inventory-stamp",
                "inventory-sha256", "output-dir"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    report = {"ok": False, "stop_code": None, "schema": "luna-subscription-canary-v2",
              "question_executed": False, "dataset_opened": False,
              "internal_http_attempts": None}
    output = Path(args.output_dir)
    try:
        binary, transport_path, pilot_path, candidate, stamp = map(
            Path, (args.binary, args.transport, args.pilot, args.candidate,
                   args.inventory_stamp))
        if (not all(path.is_absolute() for path in (binary, transport_path, pilot_path,
                                                   candidate, stamp, output))
                or not all(path.is_file() and not path.is_symlink()
                           for path in (binary, transport_path, pilot_path, stamp))
                or not candidate.is_dir() or candidate.is_symlink()
                or not HEX.fullmatch(args.inventory_sha256)
                or digest(transport_path) != TRANSPORT_SHA256
                or digest(pilot_path) != PILOT_SHA256):
            _fail("input_or_pin_invalid")
        if output.exists() or not output.parent.is_dir():
            _fail("output_not_fresh")
        os.umask(0o077)
        output.mkdir(mode=0o700)
        log_fd = os.open(output / "private-run.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(log_fd, "w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    pilot = _load_module("pinned_luna_canary_gate", pilot_path)
                    if pilot.verify_inventory(candidate, stamp, args.inventory_sha256) <= 0:
                        _fail("inventory_invalid")
                    sys.path.insert(0, str(candidate))
                    transport = _load_module("pinned_luna_canary_transport", transport_path)
                    from hymem.extraction import chunk
                    from benchmarks import extraction_canary
                    report["source_pins_verified"] = True
                    report["transport_sha256"] = TRANSPORT_SHA256
                    report["pilot_sha256"] = PILOT_SHA256
                    run_canary(pilot=pilot, transport=transport,
                        canary=extraction_canary, chunk=chunk, binary=str(binary),
                        output=output, deadline=started + WALL_SECONDS, report=report)
                except BaseException:
                    traceback.print_exc(file=log)
                    raise
        if report["ok"] is not True:
            _fail(report.get("stop_code") or "canary_failed")
    except CanaryStop as exc:
        report["ok"] = False
        report["stop_code"] = str(exc)
    except Exception:
        report["ok"] = False
        report["stop_code"] = "canary_exception"
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
