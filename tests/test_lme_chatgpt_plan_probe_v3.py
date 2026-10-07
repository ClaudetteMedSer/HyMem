"""Offline controls for the one-call SIWC diagnostic; fixtures are invented."""
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
from tools.diagnostics import lme_chatgpt_plan_probe_v3 as probe
from benchmarks import chatgpt_plan_responses_v3 as wire

HOST_ID = "urn:uuid:00000000-0000-4000-8000-000000000001"
CLIENT_ID = "oaiapp_invented"
OBSERVATION = {
    "header_defect": "none", "parsed_header_count": 1,
    "transfer_encoding": "missing", "content_encoding": "missing",
    "body_prefix": "json_prefix", "body_bytes": 2, "body_truncated": False,
    "sse_validation": "not_checked",
}


class FakeCatalog:
    def __init__(self):
        self.credential_reads = 0
        self.lock_enters = 0

    def _read_private_json(self, _fd, name):
        if name == "host.json":
            return {"ext_agent_host_id": HOST_ID}
        if name == "registration.json":
            return {"host_id": HOST_ID, "client_id": CLIENT_ID}
        raise AssertionError("preparation read unexpected private state")

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
        def __init__(self, code, http_status=None, body_shape=None,
                     media_type_class=None, wire_observation=None):
            self.code, self.http_status = code, http_status
            self.body_shape, self.media_type_class = body_shape, media_type_class
            self.wire_observation = wire_observation

    _sanitize_observation = staticmethod(wire._sanitize_observation)

    class Credentials:
        def __init__(self, token):
            if token != "invented-token":
                raise AssertionError("wrong invented token")

    def __init__(self, replies=None):
        self.calls = []
        self.replies = list(replies or [])

    def complete(self, _credential, system, user, schema, *, timeout):
        self.calls.append((system, user, schema, timeout))
        response = self.replies.pop(0)
        if isinstance(response, BaseException):
            raise response
        text, total = response
        return types.SimpleNamespace(text=text, input_tokens=1,
            output_tokens=total - 1, total_tokens=total,
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
        patch = mock.patch.object(probe, "_modules", return_value=(self.catalog, self.transport))
        patch.start()
        self.addCleanup(patch.stop)

    def prepared(self):
        return probe.prepare(self.root, self.state)["receipt_sha256"]

    def test_pins_all_transports_and_origin_under_isolated_python(self):
        hashes = probe._source_hashes()
        self.assertEqual(set(hashes), {"transport_v1", "transport_v2", "transport_v3", "catalog", "signin", "probe"})
        for name, (_, expected) in probe.SOURCES.items():
            self.assertEqual(hashes[name], expected)
        script = ("import sys,multiprocessing;sys.path.insert(0," + repr(str(ROOT)) + ");"
                  "from tools.diagnostics import lme_chatgpt_plan_probe_v3 as p;"
                  "c,t=p._modules();"
                  "\ntry:t._run_child(t._child,(None,None),3)"
                  "\nexcept t.TransportError as e: assert e.code=='transport_failure'"
                  "\nelse:raise AssertionError('expected failure')"
                  "\nassert not multiprocessing.active_children()")
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                                   capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_prepare_private_exact_receipt_without_credential_read(self):
        sha = self.prepared()
        raw = (self.root / "receipt.json").read_bytes()
        receipt = json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
        self.assertEqual(receipt["version"], 3)
        self.assertEqual(receipt["purpose"], probe.PURPOSE)
        self.assertEqual(receipt["limits"]["calls"], 1)
        self.assertEqual(receipt["fixtures"], probe._fixtures())
        self.assertEqual((self.root / "receipt.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.catalog.credential_reads, 0)
        self.assertNotIn(b"invented-token", raw)
        self.assertEqual(probe.status(self.root, sha), {"status": "prepared", "model_calls": 0})

    def test_one_success_then_replay_refused(self):
        sha = self.prepared()
        self.transport.replies = [("blue paper kite is ready", 17)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual((result["status"], result["model_calls"], result["known_tokens"]),
                         ("passed", 1, 17))
        self.assertTrue(result["usage_complete"])
        self.assertEqual(self.transport.calls,
                         [(probe.PLAIN_SYSTEM, probe.PLAIN_USER, None, probe.CALL_SECONDS)])
        self.assertEqual(self.catalog.lock_enters, 1)
        self.assertEqual(self.catalog.credential_reads, 1)
        self.assertEqual(probe.status(self.root, sha), result)
        with self.assertRaisesRegex(probe.ProbeError, "already_exists"):
            probe.run(self.root, self.state, sha)
        self.assertEqual(len(self.transport.calls), 1)

    def test_http_200_non_sse_attribution_unknown_usage_stops(self):
        sha = self.prepared()
        self.transport.replies = [FakeTransport.TransportError(
            "invalid_content_type", 200, "other_json", "json")]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["known_usage"], [])
        self.assertEqual(result["known_tokens"], 0)
        self.assertFalse(result["usage_complete"])
        self.assertEqual(result["first_fault"], {
            "code": "invalid_content_type", "phase": "call_1", "http_status": 200,
            "body_shape": "other_json", "media_type_class": "json",
            "wire_observation": None})
        self.assertEqual(probe.status(self.root, sha), result)
        self.assertEqual(len(self.transport.calls), 1)

    def test_wire_observation_is_copied_and_kept_finite(self):
        sha = self.prepared()
        supplied = dict(OBSERVATION)
        self.transport.replies = [FakeTransport.TransportError(
            "invalid_content_type", 200, "other_json", "json", supplied)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["first_fault"]["wire_observation"], OBSERVATION)
        supplied["body_prefix"] = "PRIVATE_SENTINEL"
        self.assertEqual(probe.status(self.root, sha)["first_fault"]["wire_observation"], OBSERVATION)
        self.assertEqual(len(self.transport.calls), 1)

    def test_malformed_nonnull_observation_fails_before_result_export(self):
        sha = self.prepared()
        self.transport.replies = [FakeTransport.TransportError(
            "invalid_content_type", 200, "other_json", "json",
            {**OBSERVATION, "raw_body": "PRIVATE_SENTINEL"})]
        with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
            probe.run(self.root, self.state, sha)
        self.assertTrue((self.root / "attempt.json").exists())
        self.assertFalse((self.root / "result.json").exists())
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(probe.status(self.root, sha),
                         {"status": "attempted_no_result", "usage_complete": False})

    def test_parsed_provider_error_can_have_text_lexical_prefix(self):
        body = b" " * 300 + json.dumps({"error": {
            "code": "subscription_sharing_user_not_eligible"}}).encode()
        self.assertEqual(wire._body_prefix(body), "text")
        self.assertEqual(wire._non_sse_classification(body),
                         ("subscription_sharing_user_not_eligible", "error_object"))
        sha = self.prepared()
        observation = {**OBSERVATION, "body_prefix": "text", "body_bytes": len(body)}
        self.transport.replies = [FakeTransport.TransportError(
            "subscription_sharing_user_not_eligible", 200, "error_object", "json",
            observation)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["first_fault"]["wire_observation"], observation)
        self.assertEqual(probe.status(self.root, sha), result)

    def test_saved_observation_contradictions_rejected(self):
        sha = self.prepared()
        self.transport.replies = [FakeTransport.TransportError(
            "invalid_content_type", 200, "other_json", "json", OBSERVATION)]
        probe.run(self.root, self.state, sha)
        path = self.root / "result.json"
        valid = json.loads(path.read_text())
        variants = []
        for field, value in (("http_status", 403), ("media_type_class", "sse"),
                             ("code", "timeout"), ("body_shape", "empty"),
                             ("wire_observation", {**OBSERVATION, "raw_body": "PRIVATE_SENTINEL"}),
                             ("wire_observation", {**OBSERVATION, "body_bytes": []}),
                             ("wire_observation", {**OBSERVATION, "body_prefix": "empty"})):
            variant = json.loads(json.dumps(valid))
            variant["first_fault"][field] = value
            variants.append(variant)
        for variant in variants:
            with self.subTest(fault=variant["first_fault"]):
                path.write_text(json.dumps(variant))
                path.chmod(0o600)
                with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
                    probe.status(self.root, sha)

    def test_source_pin_drift_fails_before_preparation(self):
        source = probe.SOURCES["transport_v3"]
        with mock.patch.dict(probe.SOURCES, {"transport_v3": (source[0], "0" * 64)}):
            with self.assertRaisesRegex(probe.ProbeError, "source_mismatch"):
                probe.prepare(self.root, self.state)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_output_mismatch_keeps_validated_usage_and_token_cap(self):
        sha = self.prepared()
        self.transport.replies = [("wrong", 11)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["first_fault"]["code"], "plain_output_mismatch")
        self.assertTrue(result["usage_complete"])
        self.assertEqual(result["known_tokens"], 11)
        self.assertEqual(len(self.transport.calls), 1)
        other = Path(self.temp.name) / "other"
        other.mkdir(mode=0o700)
        self.root = other
        sha = self.prepared()
        self.transport.replies = [("blue paper kite is ready", probe.MAX_TOKENS + 1)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["first_fault"]["code"], "token_cap_exceeded")
        self.assertEqual(result["model_calls"], 1)

    def test_cleanup_failure_after_completed_call_is_recorded(self):
        sha = self.prepared()
        self.transport.replies = [("blue paper kite is ready", 17)]
        with mock.patch.object(probe.multiprocessing, "active_children", return_value=[object()]):
            result = probe.run(self.root, self.state, sha)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["first_fault"]["code"], "cleanup_failure")
        self.assertEqual(result["child_cleanup"], "unverified")
        self.assertTrue(result["usage_complete"])
        self.assertEqual(probe.status(self.root, sha), result)

    def test_expiry_preflight_and_receipt_expiry_leave_attempt_unconsumed(self):
        sha = self.prepared()
        with mock.patch.object(self.catalog, "_validated_access_token", side_effect=RuntimeError("expired")):
            with self.assertRaises(RuntimeError):
                probe.run(self.root, self.state, sha)
        self.assertFalse((self.root / "attempt.json").exists())
        with mock.patch.object(probe.time, "time", return_value=time.time() + 700):
            with self.assertRaisesRegex(probe.ProbeError, "receipt_expired"):
                probe.run(self.root, self.state, sha)
        self.assertFalse((self.root / "attempt.json").exists())

    def test_old_receipt_and_unknown_receipt_keys_rejected(self):
        sha = self.prepared()
        for change in ({"version": 1}, {"purpose": "other"}, {"extra": "invented"}):
            receipt = json.loads((self.root / "receipt.json").read_text())
            receipt.update(change)
            with self.assertRaisesRegex(probe.ProbeError, "receipt_invalid"):
                with mock.patch.object(probe, "_read", return_value=(receipt, sha)):
                    probe.status(self.root, sha)
        self.assertFalse((self.root / "attempt.json").exists())

    def test_status_rejects_unhashable_unknown_and_inconsistent_success(self):
        sha = self.prepared()
        self.transport.replies = [("blue paper kite is ready", 17)]
        probe.run(self.root, self.state, sha)
        path = self.root / "result.json"
        valid = json.loads(path.read_text())
        variants = []
        value = dict(valid); value["extra"] = "invented"; variants.append(value)
        value = dict(valid); value["first_fault"] = {"code": []}; variants.append(value)
        value = dict(valid); value["usage_complete"] = False; variants.append(value)
        value = dict(valid); value["known_usage"] = []; value["known_tokens"] = 0; variants.append(value)
        value = dict(valid); value["model_calls"] = 0; variants.append(value)
        for variant in variants:
            with self.subTest(variant=variant):
                path.write_text(json.dumps(variant))
                path.chmod(0o600)
                with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
                    probe.status(self.root, sha)

    def test_status_rejects_unhashable_fault_metadata(self):
        sha = self.prepared()
        self.transport.replies = [FakeTransport.TransportError("invalid_content_type", 200, "empty", "json")]
        probe.run(self.root, self.state, sha)
        path = self.root / "result.json"
        valid = json.loads(path.read_text())
        for key in ("code", "phase", "body_shape", "media_type_class"):
            variant = json.loads(json.dumps(valid))
            variant["first_fault"][key] = []
            path.write_text(json.dumps(variant))
            path.chmod(0o600)
            with self.subTest(key=key), self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
                probe.status(self.root, sha)


if __name__ == "__main__":
    unittest.main()
