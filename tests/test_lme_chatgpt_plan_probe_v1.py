"""Offline controls for the one-shot SIWC probe; all prompts and tokens invented."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.diagnostics import lme_chatgpt_plan_probe_v1 as probe


HOST_ID = "urn:uuid:00000000-0000-4000-8000-000000000001"
CLIENT_ID = "oaiapp_12345678"


class FakeCatalog:
    def __init__(self):
        self.credential_reads = 0
        self.lock_enters = 0

    def _read_private_json(self, _fd, name):
        if name == "host.json":
            return {"ext_agent_host_id": HOST_ID}
        if name == "registration.json":
            return {"host_id": HOST_ID, "client_id": CLIENT_ID}
        raise AssertionError("prepare attempted credential read")

    def _load_signin_module(self):
        catalog = self
        @contextmanager
        def flow_lock(_state):
            catalog.lock_enters += 1
            yield
        return types.SimpleNamespace(_flow_lock=flow_lock)

    def _validated_access_token(self, _fd, _signin, _now):
        self.credential_reads += 1
        return "invented-token"


class FakeTransport:
    class TransportError(Exception):
        def __init__(self, code, http_status=None, body_shape=None):
            self.code, self.http_status, self.body_shape = code, http_status, body_shape

    class Credentials:
        def __init__(self, token):
            if token != "invented-token":
                raise AssertionError("wrong fake token")

    def __init__(self, replies=None):
        self.calls = []
        self.replies = list(replies or [])

    def complete(self, _credential, system, user, schema, *, timeout):
        self.calls.append((system, user, schema, timeout))
        response = self.replies.pop(0)
        if isinstance(response, BaseException):
            raise response
        return types.SimpleNamespace(text=response[0], input_tokens=1,
            output_tokens=response[1] - 1, total_tokens=response[1],
            cached_input_tokens=0, reasoning_output_tokens=0)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "probe"
        self.state = Path(self.temp.name) / "state"
        self.root.mkdir(mode=0o700)
        self.state.mkdir(mode=0o700)
        self.catalog = FakeCatalog()
        self.transport = FakeTransport()
        self.modules_patch = mock.patch.object(probe, "_modules", return_value=(self.catalog, self.transport))
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)

    def prepared(self):
        return probe.prepare(self.root, self.state)["receipt_sha256"]

    def test_prepare_is_private_source_bound_and_zero_credential_reads(self):
        sha = self.prepared()
        self.assertEqual(self.catalog.credential_reads, 0)
        raw = (self.root / "receipt.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
        self.assertEqual((self.root / "receipt.json").stat().st_mode & 0o777, 0o600)
        self.assertNotIn(b"invented-token", raw)
        self.assertEqual(probe.status(self.root, sha)["status"], "prepared")

    def test_two_successes_then_replay_refused(self):
        sha = self.prepared()
        self.transport.replies = [("blue paper kite is ready", 12), ('{"ready":true}', 15)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual((result["status"], result["model_calls"], result["known_tokens"]), ("passed", 2, 27))
        self.assertTrue(result["usage_complete"])
        self.assertEqual([x[3] for x in self.transport.calls], [120, 120])
        self.assertEqual(self.catalog.credential_reads, 1)
        self.assertEqual(probe.status(self.root, sha), result)
        with self.assertRaisesRegex(probe.ProbeError, "already_exists"):
            probe.run(self.root, self.state, sha)
        self.assertEqual(len(self.transport.calls), 2)

    def test_first_denial_stops_second_and_unknown_usage(self):
        sha = self.prepared()
        self.transport.replies = [FakeTransport.TransportError("quota_failure", 429, "error_object")]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["known_tokens"], 0)
        self.assertFalse(result["usage_complete"])
        self.assertEqual(result["first_fault"], {"code": "quota_failure", "phase": "call_1",
                                               "http_status": 429, "body_shape": "error_object"})
        self.assertEqual(len(self.transport.calls), 1)

    def test_missing_usage_stops_second(self):
        sha = self.prepared()
        self.transport.replies = [FakeTransport.TransportError("missing_usage")]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual((result["model_calls"], result["first_fault"]["code"]), (1, "missing_usage"))
        self.assertFalse(result["usage_complete"])

    def test_output_mismatch_and_token_cap(self):
        sha = self.prepared()
        self.transport.replies = [("wrong", 6)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["first_fault"]["code"], "plain_output_mismatch")
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(result["known_tokens"], 6)
        self.assertTrue(result["usage_complete"])

        other = Path(self.temp.name) / "other"
        other.mkdir(mode=0o700)
        self.root = other
        sha = self.prepared()
        self.transport = FakeTransport([("blue paper kite is ready", probe.MAX_TOKENS)])
        with mock.patch.object(probe, "_modules", return_value=(self.catalog, self.transport)):
            result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["first_fault"]["code"], "token_cap_reached")
        self.assertEqual(result["model_calls"], 1)

    def test_expired_credential_preflight_does_not_consume(self):
        sha = self.prepared()
        with mock.patch.object(self.catalog, "_validated_access_token", side_effect=RuntimeError("expired")):
            with self.assertRaises(RuntimeError):
                probe.run(self.root, self.state, sha)
        self.assertFalse((self.root / "attempt.json").exists())

    def test_receipt_expiry_state_binding_and_source_drift(self):
        sha = self.prepared()
        with self.assertRaisesRegex(probe.ProbeError, "state_binding_mismatch"):
            probe.run(self.root, Path(self.temp.name) / "wrong", sha)
        self.assertFalse((self.root / "attempt.json").exists())
        with mock.patch.object(probe, "_source_hashes", return_value={"drift": "x"}):
            with self.assertRaisesRegex(probe.ProbeError, "receipt_invalid"):
                probe.run(self.root, self.state, sha)
        with mock.patch.object(probe.time, "time", return_value=time.time() + 700):
            with self.assertRaisesRegex(probe.ProbeError, "receipt_expired"):
                probe.run(self.root, self.state, sha)

    def test_bad_receipt_sha_refused(self):
        sha = self.prepared()
        with self.assertRaisesRegex(probe.ProbeError, "receipt_mismatch"):
            probe.run(self.root, self.state, "0" * 64)
        self.assertFalse((self.root / "attempt.json").exists())
        self.assertEqual(len(sha), 64)

    def test_structured_integer_one_is_rejected(self):
        sha = self.prepared()
        self.transport.replies = [("blue paper kite is ready", 8), ('{"ready":1}', 9)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["first_fault"]["code"], "structured_output_mismatch")
        self.assertTrue(result["usage_complete"])
        self.assertEqual(result["known_tokens"], 17)

    def test_status_rejects_untrusted_extra_text(self):
        sha = self.prepared()
        self.transport.replies = [("blue paper kite is ready", 8), ('{"ready":true}', 9)]
        probe.run(self.root, self.state, sha)
        path = self.root / "result.json"
        value = json.loads(path.read_text())
        value["leaked_text"] = "invented but should not echo"
        path.write_text(json.dumps(value))
        path.chmod(0o600)
        with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
            probe.status(self.root, sha)

    def test_status_rejects_success_with_unknown_second_usage(self):
        sha = self.prepared()
        self.transport.replies = [("blue paper kite is ready", 8), ('{"ready":true}', 9)]
        probe.run(self.root, self.state, sha)
        path = self.root / "result.json"
        value = json.loads(path.read_text())
        value["known_usage"] = value["known_usage"][:1]
        value["known_tokens"] = 8
        value["usage_complete"] = False
        path.write_text(json.dumps(value))
        path.chmod(0o600)
        with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
            probe.status(self.root, sha)

    def test_actual_transport_spawn_import_under_isolated_python(self):
        # _child fails before connection.request: invalid test-only credentials
        # make attribute access fail while composing headers. No network call.
        script = ("import sys,multiprocessing;sys.path.insert(0," + repr(str(ROOT)) + ");"
                  "from tools.diagnostics import lme_chatgpt_plan_probe_v1 as p;"
                  "c,t=p._modules();"
                  "\ntry:t._run_child(t._child,(None,None),3)"
                  "\nexcept t.TransportError as e: assert e.code=='transport_failure'"
                  "\nelse:raise AssertionError('expected failure')"
                  "\nassert not multiprocessing.active_children()")
        # Spawn from -c can import the benchmark module because multiprocessing
        # carries the explicit sys.path even with interpreter isolation.
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
