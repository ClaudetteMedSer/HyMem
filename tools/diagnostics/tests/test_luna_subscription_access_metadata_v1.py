"""Offline controls for the stopped-run, zero-inference access reader."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "luna_subscription_access_metadata_v1.py"
SPEC = importlib.util.spec_from_file_location("access_metadata_v1", SOURCE)
assert SPEC and SPEC.loader
access = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(access)
BASE = Path(__file__).resolve().parents[3] / "benchmarks/codex_subscription.py"
SESSION, QUOTA, GATE_ERROR = access._load_transport(BASE.read_bytes())


def stopped_run() -> dict:
    return {"status": "terminal_incomplete_or_unclean", "runtime_cleanup_verified": True,
            "selected_denominator": 4, "scored_count": 0, "failed_count": 4,
            "usage_complete": False, "first_failure": {"phase": "run", "rpc": "turn/events",
                "app_server_error": {"identity": "matched",
                    "error_class": "responseStreamDisconnected",
                    "http_status_code": 403, "will_retry": True}}}


class FakeSession:
    calls: list[str]
    closed: bool
    fail_on: str | None
    account: dict
    quota: dict

    def __init__(self, _session_class, _binary: Path):
        self.calls = []
        self.closed = False
        self.fail_on = None
        self.account = {"account": {"type": "chatgpt", "planType": "plus",
                                    "email": "secret@example.invalid", "id": "SECRET_ACCOUNT"}}
        self.quota = {"rateLimits": {"primary": {"usedPercent": 40,
                                                  "windowDurationMins": 300,
                                                  "resetsAt": 1_800_000_000},
                                     "credits": {"hasCredits": False},
                                     "privateName": "SECRET_QUOTA"}}

    def rpc(self, method: str) -> dict:
        self.calls.append(method)
        if method not in access.RPC_ALLOWLIST:
            raise AssertionError("forbidden RPC")
        if method == self.fail_on:
            raise access.DiagnosticFailure("secret@example.invalid")
        return {"account/read": self.account,
                "account/rateLimits/read": self.quota}.get(method, {})

    def close(self) -> bool:
        self.closed = True
        return True


class AccessMetadataTests(unittest.TestCase):
    def test_rpc_allowlist_and_privacy(self):
        instance = FakeSession(SESSION, access.BINARY)
        with patch.object(access, "_pinned_file", return_value=b""):
            result = access.inspect_access(lambda *_: stopped_run(), SESSION, QUOTA, GATE_ERROR,
                                           session_factory=lambda *_: instance)
        self.assertEqual(instance.calls, ["initialize", "initialized",
                                          "account/read", "account/rateLimits/read"])
        self.assertTrue(instance.closed)
        self.assertEqual(result["auth"], "chatgpt")
        self.assertEqual(result["plan"], "plus")
        self.assertEqual(result["quota_status"], "quota_above_floor")
        self.assertEqual(result["quota_windows"][0]["remaining_percent"], 60)
        self.assertNotIn("SECRET", repr(result))
        self.assertNotIn("example.invalid", repr(result))
        self.assertEqual(set(access.RPC_ALLOWLIST), {"initialize", "initialized",
            "account/read", "account/rateLimits/read"})
        self.assertIs(access.RPC_ALLOWLIST["account/read"]["refreshToken"], False)

    def test_terminal_failure_before_start(self):
        for changed in ({"runtime_cleanup_verified": False}, {"scored_count": 1},
                        {"status": "checkpoint_running"}):
            run = stopped_run()
            run.update(changed)
            with self.assertRaises(access.DiagnosticFailure):
                access._bound_run(lambda *_: run)
        run = stopped_run()
        run["first_failure"]["app_server_error"]["will_retry"] = False
        with self.assertRaises(access.DiagnosticFailure):
            access._bound_run(lambda *_: run)

    def test_source_pin_failure_before_start(self):
        with patch.object(access, "_pinned_file", side_effect=access.DiagnosticFailure("source_unverified")):
            with self.assertRaises(access.DiagnosticFailure):
                access.inspect_access(lambda *_: stopped_run(), SESSION, QUOTA, GATE_ERROR,
                                      session_factory=lambda *_: self.fail("started"))

    def test_unknown_malformed_and_floor_gate(self):
        for response, status in [({}, "unknown_quota"),
                                 ({"rateLimits": {"primary": {"usedPercent": "private"}}}, "invalid_quota"),
                                 ({"rateLimits": {"primary": {"usedPercent": 90}}}, "quota_floor"),
                                 ({"rateLimits": {"spendControlReached": True}}, "quota_exhausted")]:
            code, windows = access._quota_projection(response, QUOTA, GATE_ERROR)
            self.assertEqual((code, windows), (status, []))

    def test_rpc_failures_close_and_do_not_leak(self):
        for method in ("initialize", "account/read", "account/rateLimits/read"):
            instance = FakeSession(SESSION, access.BINARY)
            instance.fail_on = method
            with patch.object(access, "_pinned_file", return_value=b""):
                result = access.inspect_access(lambda *_: stopped_run(), SESSION, QUOTA, GATE_ERROR,
                                               session_factory=lambda *_: instance)
            self.assertEqual(result["status"], "rpc_unverified")
            self.assertTrue(result["owned_process_cleanup_verified"])
            self.assertTrue(instance.closed)
            self.assertNotIn("example.invalid", repr(result))

    def test_account_unknown_never_exports_ids(self):
        self.assertEqual(access._account_projection({"account": {"type": "api_key",
            "planType": "enterprise", "email": "secret@example.invalid"}}), ("other", "unknown"))

    def test_forbidden_rpc_never_sent(self):
        session = object.__new__(access.AllowlistedSession)
        session.inner = object()
        with self.assertRaises(access.DiagnosticFailure):
            session.rpc("thread/start")
        with self.assertRaises(access.DiagnosticFailure):
            session.rpc("turn/start")

    def test_cleanup_addresses_owned_group_even_after_parent_exit(self):
        class Process:
            pid = 54321
            def wait(self, timeout):
                return 0
        class Inner:
            process = Process()
            def close(self):
                calls.append("close")
        calls = []
        def killpg(pid, sig):
            calls.append((pid, sig))
            raise ProcessLookupError()
        session = object.__new__(access.AllowlistedSession)
        session.inner = Inner()
        with patch.object(access.os, "killpg", side_effect=killpg):
            self.assertTrue(session.close())
        self.assertEqual(calls, ["close", (54321, access.signal.SIGKILL),
                                 (54321, 0)])

    def test_constructor_partial_start_closes_group(self):
        calls = []
        class Partial:
            def __init__(self, *_args, **_kwargs):
                self.process = type("Process", (), {"pid": 54322,
                    "wait": lambda self, timeout: 0})()
                raise RuntimeError("private failure")
            def close(self):
                calls.append("close")
        with patch.object(access.os, "killpg", side_effect=ProcessLookupError):
            with self.assertRaises(access.PartialStartupFailure) as raised:
                access.AllowlistedSession(Partial, access.BINARY)
        self.assertEqual(calls, ["close"])
        self.assertTrue(raised.exception.cleanup_verified)

    def test_partial_start_cleanup_failure_is_reported(self):
        with patch.object(access, "_pinned_file", return_value=b""):
            result = access.inspect_access(lambda *_: stopped_run(), SESSION, QUOTA, GATE_ERROR,
                session_factory=lambda *_: (_ for _ in ()).throw(
                    access.PartialStartupFailure(False)))
        self.assertEqual(result["status"], "rpc_unverified")
        self.assertIs(result["owned_process_cleanup_verified"], False)

    def test_account_unverified_stops_before_quota(self):
        instance = FakeSession(SESSION, access.BINARY)
        instance.account = {"account": {"type": "api_key", "planType": "enterprise"}}
        with patch.object(access, "_pinned_file", return_value=b""):
            result = access.inspect_access(lambda *_: stopped_run(), SESSION, QUOTA, GATE_ERROR,
                                           session_factory=lambda *_: instance)
        self.assertEqual(result["status"], "account_unverified")
        self.assertNotIn("account/rateLimits/read", instance.calls)

    def test_notification_quota_gate_remains_finite(self):
        instance = FakeSession(SESSION, access.BINARY)
        def gated(method):
            if method == "account/rateLimits/read":
                raise GATE_ERROR("quota_floor")
            return FakeSession.rpc(instance, method)
        instance.rpc = gated
        with patch.object(access, "_pinned_file", return_value=b""):
            result = access.inspect_access(lambda *_: stopped_run(), SESSION, QUOTA, GATE_ERROR,
                                           session_factory=lambda *_: instance)
        self.assertEqual(result["quota_status"], "quota_floor")
        self.assertEqual(result["status"], "rpc_unverified")

    def test_local_payload_pins_reader_and_bounds_ssh(self):
        captured = {}
        def fake_run(command, **kwargs):
            captured.update(command=command, **kwargs)
            return subprocess.CompletedProcess(command, 1,
                stdout=json.dumps({"schema": access.SCHEMA, "status": "source_unverified",
                    "run_verified": False, "owned_process_cleanup_verified": None}).encode(), stderr=b"PRIVATE")
        with (patch.object(access.subprocess, "run", side_effect=fake_run),
              patch("builtins.print") as printed):
            self.assertEqual(access._local_ssh(), 1)
        self.assertEqual(captured["timeout"], 45)
        self.assertEqual(captured["command"][:8], ["ssh", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1", "afrodite"])
        payload = captured["input"].decode()
        self.assertIn("EMBEDDED_READER_SOURCE = bytes.fromhex(", payload)
        compile(payload, "<remote-payload>", "exec")
        self.assertNotIn("PRIVATE", str(printed.call_args))

    def test_remote_projection_rejects_private_fields(self):
        self.assertEqual(access._safe_remote_report({"schema": access.SCHEMA,
            "status": "metadata_read", "email": "secret@example.invalid"})["status"],
            "ssh_unverified")


if __name__ == "__main__":
    unittest.main()
