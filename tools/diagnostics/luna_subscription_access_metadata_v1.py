"""Finite, zero-inference account metadata for the stopped x70ca5d2 run.

Execute on Afrodite with Python 3.11+ via stdin. No benchmark package import,
credential file read, login, refresh, thread, turn, or model RPC is performed.
All exceptions and app-server output are deliberately suppressed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any


SCHEMA = "luna-subscription-access-metadata-v1"
ROOT = Path("/home/atta/.hymem-lme-diagnostic-preflight-x70ca5d2")
RECEIPT_SHA256 = "bd689672eefb5a0de3e993be4af05194fc6b5837c0fc573b1c03ab7d45dec24d"
READER_SHA256 = "32caa15f7cdf3748ee4531c85be131c16229649f47618a70bd55df3734fe5207"
BASE_SHA256 = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
BINARY_SHA256 = "167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9"
BINARY = Path("/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex")
PLANS = frozenset({"plus", "pro", "prolite", "free", "go", "team"})
RPC_ALLOWLIST = {
    "initialize": {"clientInfo": {"name": "hymem_luna_pilot", "version": "0.1"},
                   "capabilities": {"experimentalApi": True}},
    "initialized": {},
    "account/read": {"refreshToken": False},
    "account/rateLimits/read": {},
}
SAFE_FAILURES = frozenset({"unknown_quota", "invalid_quota", "quota_exhausted",
    "quota_floor", "credit_balance_present", "subscription_plan_unverified"})
EMBEDDED_READER_SOURCE = None  # Replaced only by the local, hash-checked SSH launcher.


class DiagnosticFailure(Exception):
    pass


class PartialStartupFailure(DiagnosticFailure):
    def __init__(self, cleanup_verified: bool):
        self.cleanup_verified = cleanup_verified
        super().__init__("startup_unverified")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_file(path: Path, digest: str, *, executable: bool = False) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 8_000_000 and not executable:
        raise DiagnosticFailure("source_unverified")
    if _sha(path) != digest:
        raise DiagnosticFailure("source_unverified")
    return b"" if executable else path.read_bytes()


def _load_reader(source: bytes) -> Any:
    # Only the exact accepted stdlib-only reader is evaluated. Its __main__
    # entry point cannot run under this namespace.
    namespace = {"__name__": "_pinned_progress_reader", "__file__": "<pinned>"}
    exec(compile(source, "<pinned>", "exec"), namespace)
    return namespace["inspect"]


def _load_transport(source: bytes) -> tuple[Any, Any, type[Exception]]:
    # Remote Python lacks hymem. Select only the exact pinned stdlib transport
    # definitions. The original session's strict overrides, notification checks,
    # deadlines and process-group cleanup remain in force.
    tree = ast.parse(source, filename="<pinned-base>")
    constants = {"MODEL", "VERSION", "MAX_BENIGN_WARNINGS",
        "_CODE_MODE_DISABLED_NOTICE", "SUBSCRIPTION_PLANS", "DISABLED_FEATURES",
        "OVERRIDES"}
    functions = {"_fail", "_safe_method", "_validate_warning",
        "sanitized_environment", "_number", "quota_metadata",
        "_validate_account_notification"}
    classes = {"SubscriptionTransportError", "StdioSession"}
    chosen = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions | classes:
            chosen.append(node)
            found.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & constants:
                chosen.append(node)
                found.update(names & constants)
    if found != constants | functions | classes:
        raise DiagnosticFailure("source_unverified")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"Any": Any, "math": math, "os": os, "json": json,
                 "queue": queue, "re": re, "signal": signal,
                 "subprocess": subprocess, "threading": threading,
                 "time": time, "deque": deque}
    module = ast.fix_missing_locations(ast.Module(body=[future, *chosen], type_ignores=[]))
    exec(compile(module, "<pinned-base>", "exec"), namespace)
    if namespace["SUBSCRIPTION_PLANS"] != PLANS:
        raise DiagnosticFailure("source_unverified")
    return namespace["StdioSession"], namespace["quota_metadata"], namespace["SubscriptionTransportError"]


def _bound_run(inspect: Any) -> None:
    result = inspect(ROOT, RECEIPT_SHA256)
    fault = result.get("first_failure")
    app = fault.get("app_server_error") if type(fault) is dict else None
    if not (result.get("status") == "terminal_incomplete_or_unclean"
            and result.get("runtime_cleanup_verified") is True
            and result.get("selected_denominator") == 4
            and result.get("scored_count") == 0
            and result.get("failed_count") == 4
            and result.get("usage_complete") is False
            and type(fault) is dict and fault.get("phase") == "run"
            and fault.get("rpc") == "turn/events"
            and type(app) is dict and app.get("identity") == "matched"
            and app.get("error_class") == "responseStreamDisconnected"
            and app.get("http_status_code") == 403
            and app.get("will_retry") is True):
        raise DiagnosticFailure("terminal_unverified")


class AllowlistedSession:
    """Fence the exact pinned StdioSession to four metadata-only messages."""
    def __init__(self, session_class: Any, binary: Path):
        # Retain the object if the accepted constructor raises after Popen.
        self.inner = session_class.__new__(session_class)
        try:
            session_class.__init__(self.inner, str(binary), str(ROOT), timeout=20)
        except BaseException:
            if getattr(self.inner, "process", None) is not None:
                raise PartialStartupFailure(self.close()) from None
            raise

    def rpc(self, method: str) -> dict[str, Any]:
        if method not in RPC_ALLOWLIST:
            raise DiagnosticFailure("rpc_forbidden")
        if method == "initialized":
            self.inner.send(method, {}, notification=True)
            return {}
        return self.inner.rpc(method, RPC_ALLOWLIST[method])

    def close(self) -> bool:
        process = getattr(self.inner, "process", None)
        if process is None:
            return True
        try:
            self.inner.close()
        except Exception:
            pass
        # The accepted close already signals the group. Repeat a final bounded
        # reap if that routine raised before its SIGKILL path.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            return False
        try:
            process.wait(timeout=2)
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        except (OSError, subprocess.SubprocessError):
            return False
        return False


def _account_projection(response: dict[str, Any]) -> tuple[str, str]:
    account = response.get("account")
    if type(account) is not dict:
        return "unknown", "unknown"
    auth = "chatgpt" if account.get("type") == "chatgpt" else "other"
    plan = account.get("planType")
    return auth, plan if type(plan) is str and plan in PLANS else "unknown"


def _quota_projection(response: dict[str, Any], quota_metadata: Any,
                      gate_error: type[Exception]) -> tuple[str, list[dict[str, Any]]]:
    try:
        windows = quota_metadata(response)
    except gate_error as exc:
        code = str(exc)
        return (code if code in SAFE_FAILURES else "quota_unverified"), []
    return "quota_above_floor", [{
        "remaining_percent": window["remaining_percent"],
        "window_minutes": window["window_minutes"],
        "resets_at": window["resets_at"]} for window in windows[:16]]


def inspect_access(inspect: Any, session_class: Any, quota_metadata: Any,
                   gate_error: type[Exception], session_factory: Any = AllowlistedSession) -> dict[str, Any]:
    _bound_run(inspect)
    _pinned_file(BINARY, BINARY_SHA256, executable=True)
    session = None
    result: dict[str, Any] = {"schema": SCHEMA, "status": "rpc_unverified",
        "run_verified": True, "owned_process_cleanup_verified": False,
        "auth": "unknown", "plan": "unknown", "quota_status": "unknown_quota",
        "quota_windows": []}
    try:
        session = session_factory(session_class, BINARY)
        session.rpc("initialize")
        session.rpc("initialized")
        account = session.rpc("account/read")
        result["auth"], result["plan"] = _account_projection(account)
        if result["auth"] != "chatgpt" or result["plan"] == "unknown":
            result["status"] = "account_unverified"
        else:
            quota = session.rpc("account/rateLimits/read")
            result["quota_status"], result["quota_windows"] = _quota_projection(
                quota, quota_metadata, gate_error)
            result["status"] = "metadata_read"
    except PartialStartupFailure as exc:
        result["owned_process_cleanup_verified"] = exc.cleanup_verified
        result["status"] = "rpc_unverified"
    except gate_error as exc:
        code = str(exc)
        result["quota_status"] = code if code in SAFE_FAILURES else "quota_unverified"
        result["status"] = "rpc_unverified"
    except (DiagnosticFailure, OSError, ValueError, subprocess.SubprocessError):
        result["status"] = "rpc_unverified"
    finally:
        if session is not None:
            try:
                result["owned_process_cleanup_verified"] = bool(session.close())
            except Exception:
                result["owned_process_cleanup_verified"] = False
    return result


def _safe_remote_report(value: Any) -> dict[str, Any]:
    fallback = {"schema": SCHEMA, "status": "ssh_unverified", "run_verified": False,
                "owned_process_cleanup_verified": None}
    if type(value) is not dict or value.get("schema") != SCHEMA:
        return fallback
    if (value.get("status") == "source_unverified" and set(value) == set(fallback)
            and value.get("run_verified") is False
            and value.get("owned_process_cleanup_verified") is None):
        return {**value}
    if set(value) != {"schema", "status", "run_verified", "owned_process_cleanup_verified",
                      "auth", "plan", "quota_status", "quota_windows"}:
        return fallback
    if (value["status"] not in {"rpc_unverified", "account_unverified", "metadata_read"}
            or value["run_verified"] is not True
            or type(value["owned_process_cleanup_verified"]) is not bool
            or value["auth"] not in {"chatgpt", "other", "unknown"}
            or value["plan"] not in PLANS | {"unknown"}
            or value["quota_status"] not in SAFE_FAILURES | {"quota_above_floor", "quota_unverified"}
            or type(value["quota_windows"]) is not list
            or len(value["quota_windows"]) > 16):
        return fallback
    for window in value["quota_windows"]:
        if type(window) is not dict or set(window) != {"remaining_percent", "window_minutes", "resets_at"}:
            return fallback
        for key in window:
            number = window[key]
            if number is None and key != "remaining_percent":
                continue
            if type(number) not in (float, int) or not math.isfinite(number):
                return fallback
        if not 0 <= window["remaining_percent"] <= 100:
            return fallback
    return value


def _local_ssh() -> int:
    """One local invocation; SSH input contains the exact pinned reader bytes."""
    fallback = {"schema": SCHEMA, "status": "ssh_unverified", "run_verified": False,
                "owned_process_cleanup_verified": None}
    try:
        reader_path = Path(__file__).with_name("luna_lme_diagnostic_progress_v5.py")
        reader = _pinned_file(reader_path, READER_SHA256)
        lines = Path(__file__).read_text().splitlines(keepends=True)
        matches = [index for index, line in enumerate(lines)
                   if line.startswith("EMBEDDED_READER_SOURCE = None  #")]
        if len(matches) != 1:
            raise DiagnosticFailure("source_unverified")
        lines[matches[0]] = "EMBEDDED_READER_SOURCE = bytes.fromhex(" + repr(reader.hex()) + ")\n"
        payload = "".join(lines).encode()
        completed = subprocess.run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ConnectionAttempts=1", "afrodite", "/usr/bin/python3",
            "-I", "-B", "-", "--root", str(ROOT),
            "--receipt-sha256", RECEIPT_SHA256],
            input=payload, capture_output=True, timeout=45, check=False)
        if len(completed.stdout) > 16_384:
            raise DiagnosticFailure("ssh_unverified")
        parsed = json.loads(completed.stdout)
        fallback = _safe_remote_report(parsed)
        if completed.returncode != 0 and fallback.get("status") == "metadata_read":
            fallback = {"schema": SCHEMA, "status": "ssh_unverified",
                        "run_verified": False, "owned_process_cleanup_verified": None}
    except Exception:
        pass
    print(json.dumps(fallback, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if fallback.get("status") == "metadata_read" and fallback.get(
        "owned_process_cleanup_verified") is True else 1


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--local-ssh"]:
        return _local_ssh()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"schema": SCHEMA, "status": "source_unverified",
        "run_verified": False, "owned_process_cleanup_verified": None}
    try:
        if args.root != str(ROOT) or args.receipt_sha256 != RECEIPT_SHA256:
            raise DiagnosticFailure("identity_unverified")
        reader = EMBEDDED_READER_SOURCE
        if type(reader) is not bytes or hashlib.sha256(reader).hexdigest() != READER_SHA256:
            raise DiagnosticFailure("source_unverified")
        base = _pinned_file(ROOT / "code/benchmarks/codex_subscription.py", BASE_SHA256)
        inspect = _load_reader(reader)
        session_class, quota_metadata, gate_error = _load_transport(base)
        report = inspect_access(inspect, session_class, quota_metadata, gate_error)
    except Exception:
        # No exception message, path, RPC payload, or stderr may leave the host.
        pass
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if report.get("status") == "metadata_read" and report.get(
        "owned_process_cleanup_verified") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
