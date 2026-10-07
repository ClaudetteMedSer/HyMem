"""Offline, invented-data controls for the structured-only SIWC probe."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
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
from tools.diagnostics import lme_chatgpt_plan_probe_v7 as probe
from benchmarks import chatgpt_plan_responses_v6 as wire

HOST_ID = "urn:uuid:00000000-0000-4000-8000-000000000001"
CLIENT_ID = "oaiapp_invented"


class FakeCatalog:
    def __init__(self):
        self.credential_reads = 0
        self.lock_enters = 0

    def _read_private_json(self, _fd, name):
        if name == "host.json":
            return {"ext_agent_host_id": HOST_ID}
        if name == "registration.json":
            return {"host_id": HOST_ID, "client_id": CLIENT_ID}
        raise AssertionError("unexpected private file")

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
    TransportError = wire.TransportError
    _v3 = wire._v3
    _sanitize_stream_observation = staticmethod(wire._sanitize_stream_observation)

    class Credentials:
        def __init__(self, token):
            if token != "invented-token":
                raise AssertionError("wrong invented credential")

    def __init__(self):
        self.calls = []
        self.replies = []

    def complete(self, _credentials, system, user, schema, *, timeout):
        self.calls.append((system, user, schema, timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        text, total = reply
        return types.SimpleNamespace(text=text, input_tokens=1,
            output_tokens=total - 1, total_tokens=total,
            cached_input_tokens=0, reasoning_output_tokens=0)


class StructuredProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "probe"
        self.state = Path(self.temp.name) / "state"
        self.root.mkdir(mode=0o700)
        self.state.mkdir(mode=0o700)
        self.catalog = FakeCatalog()
        self.transport = FakeTransport()
        patch = mock.patch.object(probe, "_modules",
                                  return_value=(self.catalog, self.transport))
        patch.start()
        self.addCleanup(patch.stop)

    def prepared(self):
        return probe.prepare(self.root, self.state)["receipt_sha256"]

    def test_source_closure_and_real_request_contract(self):
        hashes = probe._source_hashes()
        self.assertEqual(hashes["probe_v6"], probe.BASE_SHA256)
        self.assertEqual(hashes["transport_v6"],
                         "811bff13ebc4b24ebd22cad16542c3b58597dc04538a1d7deb5085df7190a28f")
        self.assertEqual(hashes["probe"], hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest())
        request = wire.build_request(probe.STRUCTURED_SYSTEM, probe.STRUCTURED_USER,
                                     probe.SCHEMA)
        self.assertEqual((request["model"], request["reasoning"], request["store"],
                          request["stream"]),
                         (probe.MODEL, {"effort": "low"}, False, True))
        self.assertEqual(request["text"], {"format": {"type": "json_schema",
            "name": "response", "strict": True, "schema": probe.SCHEMA}})
        self.assertNotIn("plain_system", probe._fixtures())
        script = ("import sys,multiprocessing;sys.path.insert(0," + repr(str(ROOT)) + ");"
                  "from tools.diagnostics import lme_chatgpt_plan_probe_v7 as p;"
                  "c,t=p._modules();"
                  "\ntry:t.complete(None,'invented system','invented user',p.SCHEMA,timeout=3)"
                  "\nexcept t.TransportError as e: assert e.code=='invalid_credentials'"
                  "\nelse:raise AssertionError('expected invalid credentials')"
                  "\nassert not multiprocessing.active_children()")
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_prepare_is_private_and_does_not_read_credentials(self):
        sha = self.prepared()
        raw = (self.root / "receipt.json").read_bytes()
        receipt = json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
        self.assertEqual((receipt["version"], receipt["purpose"]), (7, probe.PURPOSE))
        self.assertEqual(receipt["fixtures"], probe._fixtures())
        self.assertEqual(receipt["limits"], {"calls": 1, "observed_tokens": 160000,
            "campaign_seconds": 300, "child_seconds": 120})
        self.assertEqual((self.root / "receipt.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.catalog.credential_reads, 0)
        self.assertNotIn(b"invented-token", raw)
        self.assertEqual(probe.status(self.root, sha), {"status": "prepared", "model_calls": 0})

    def test_one_structured_success_then_replay_refusal(self):
        sha = self.prepared()
        self.transport.replies = [(' {"ready": true} ', 17)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual((result["status"], result["model_calls"], result["known_tokens"]),
                         ("passed", 1, 17))
        self.assertEqual(self.transport.calls,
                         [(probe.STRUCTURED_SYSTEM, probe.STRUCTURED_USER,
                           probe.SCHEMA, probe.CALL_SECONDS)])
        self.assertTrue(result["usage_complete"])
        self.assertEqual((self.catalog.lock_enters, self.catalog.credential_reads), (1, 1))
        self.assertEqual(probe.status(self.root, sha), result)
        with self.assertRaisesRegex(probe._base().ProbeError, "already_exists"):
            probe.run(self.root, self.state, sha)
        self.assertEqual(len(self.transport.calls), 1)

    def test_invalid_json_and_schema_mismatch_keep_known_usage(self):
        cases = (("not json", "structured_output_invalid"),
                 ('{"ready":true,"ready":true}', "structured_output_invalid"),
                 ('{"ready":1}', "structured_output_mismatch"),
                 ('{"ready":false}', "structured_output_mismatch"),
                 ('{"ready":true,"extra":1}', "structured_output_mismatch"),
                 ('[true]', "structured_output_mismatch"))
        for index, (answer, code) in enumerate(cases):
            with self.subTest(answer=answer):
                self.root = Path(self.temp.name) / f"invalid_{index}"
                self.root.mkdir(mode=0o700)
                sha = self.prepared()
                self.transport.replies = [(answer, 11)]
                result = probe.run(self.root, self.state, sha)
                self.assertEqual((result["status"], result["first_fault"]["code"]),
                                 ("failed", code))
                self.assertEqual((result["known_tokens"], result["usage_complete"]), (11, True))
                self.assertEqual(probe.status(self.root, sha), result)

    def test_failed_call_usage_unknown_and_finite_error(self):
        sha = self.prepared()
        self.transport.replies = [wire.TransportError("quota_failure", 429, "error_object", "json")]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual((result["status"], result["model_calls"], result["known_tokens"]),
                         ("failed", 1, 0))
        self.assertEqual((result["known_usage"], result["usage_complete"]), ([], False))
        self.assertEqual(result["first_fault"]["code"], "quota_failure")
        self.assertEqual(probe.status(self.root, sha), result)
        self.assertNotIn("invented-token", (self.root / "result.json").read_text())

    def test_token_cap_and_cleanup_failure(self):
        sha = self.prepared()
        self.transport.replies = [(' {"ready":true}', probe.MAX_TOKENS + 1)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["first_fault"]["code"], "token_cap_exceeded")
        self.root = Path(self.temp.name) / "cleanup"
        self.root.mkdir(mode=0o700)
        sha = self.prepared()
        self.transport.replies = [(' {"ready":true}', 17)]
        with mock.patch.object(probe.multiprocessing, "active_children", return_value=[object()]):
            result = probe.run(self.root, self.state, sha)
        self.assertEqual((result["status"], result["child_cleanup"],
                          result["first_fault"]["code"]),
                         ("failed", "unverified", "cleanup_failure"))

    def test_expiry_preflight_and_tampered_result(self):
        sha = self.prepared()
        with mock.patch.object(self.catalog, "_validated_access_token", side_effect=RuntimeError("expired")):
            with self.assertRaises(RuntimeError):
                probe.run(self.root, self.state, sha)
        self.assertFalse((self.root / "attempt.json").exists())
        with mock.patch.object(probe.time, "time", return_value=time.time() + 700):
            with self.assertRaisesRegex(probe.ProbeError, "receipt_expired"):
                probe.run(self.root, self.state, sha)
        self.transport.replies = [(' {"ready":true}', 17)]
        probe.run(self.root, self.state, sha)
        path = self.root / "result.json"
        valid = json.loads(path.read_text())
        variants = []
        variant = dict(valid); variant["usage_complete"] = False; variants.append(variant)
        variant = dict(valid); variant["model_calls"] = 0; variants.append(variant)
        variant = dict(valid); variant["first_fault"] = {"code": []}; variants.append(variant)
        for variant in variants:
            with self.subTest(variant=variant):
                path.write_text(json.dumps(variant))
                path.chmod(0o600)
                with self.assertRaises(Exception):
                    probe.status(self.root, sha)

    def test_receipt_rejects_equal_value_wrong_types(self):
        sha = self.prepared()
        original = json.loads((self.root / "receipt.json").read_text())
        variants = []
        value = json.loads(json.dumps(original)); value["limits"]["calls"] = True; variants.append(value)
        value = json.loads(json.dumps(original)); value["fixtures"]["schema"]["additionalProperties"] = 0; variants.append(value)
        value = json.loads(json.dumps(original)); value["fixtures"]["schema"]["properties"]["ready"]["type"] = ["boolean"]; variants.append(value)
        value = json.loads(json.dumps(original)); value["purpose"] = "ordinary"; variants.append(value)
        for value in variants:
            with self.subTest(value=value):
                with mock.patch.object(probe._base(), "_read", return_value=(value, sha)):
                    with self.assertRaisesRegex(probe.ProbeError, "receipt_invalid"):
                        probe.status(self.root, sha)

    def test_result_rejects_old_plain_mismatch_code(self):
        sha = self.prepared()
        self.transport.replies = [(' {"ready":false}', 17)]
        probe.run(self.root, self.state, sha)
        path = self.root / "result.json"
        result = json.loads(path.read_text())
        result["first_fault"]["code"] = "plain_output_mismatch"
        path.write_text(json.dumps(result))
        path.chmod(0o600)
        with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
            probe.status(self.root, sha)


if __name__ == "__main__":
    unittest.main()
