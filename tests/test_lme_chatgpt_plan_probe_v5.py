"""Offline controls for the one-call finalized-output probe; fixtures are invented."""
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
from tools.diagnostics import lme_chatgpt_plan_probe_v5 as probe
from benchmarks import chatgpt_plan_responses_v5 as wire

HOST_ID = "urn:uuid:00000000-0000-4000-8000-000000000001"
CLIENT_ID = "oaiapp_invented"
OBSERVATION = {
    "header_defect": "none", "parsed_header_count": 1,
    "transfer_encoding": "missing", "content_encoding": "missing",
    "body_prefix": "json_prefix", "body_bytes": 2, "body_truncated": False,
    "sse_validation": "not_checked",
}
STREAM_OBSERVATION = {
    "event_type": "response.completed", "terminal_status": "completed",
    "terminal_model_matches": True, "terminal_output_kind": "message",
    "terminal_channel": "final", "terminal_content_kind": "output_text",
    "terminal_output_state": "missing", "finalized_item_count": 1,
    "output_reconstructed": True,
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
                     media_type_class=None, wire_observation=None,
                     stream_observation=None):
            self.code, self.http_status = code, http_status
            self.body_shape, self.media_type_class = body_shape, media_type_class
            self.wire_observation = wire_observation
            self.stream_observation = stream_observation

    _v3 = wire._v3
    _sanitize_stream_observation = staticmethod(wire._sanitize_stream_observation)

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
        self.assertEqual(set(hashes), {"transport_v1", "transport_v2", "transport_v3",
                                       "transport_v4", "transport_v5", "catalog", "signin", "probe"})
        for name, (_, expected) in probe.SOURCES.items():
            self.assertEqual(hashes[name], expected)
        script = ("import sys,multiprocessing;sys.path.insert(0," + repr(str(ROOT)) + ");"
                  "from tools.diagnostics import lme_chatgpt_plan_probe_v5 as p;"
                  "c,t=p._modules();"
                  "\ntry:t.complete(None,'invented system','invented user',timeout=3)"
                  "\nexcept t.TransportError as e: assert e.code=='invalid_credentials'"
                  "\nelse:raise AssertionError('expected invalid credentials')"
                  "\ntry:t._run_child(t._child,(None,{}),3)"
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
        self.assertEqual(receipt["version"], 5)
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
            "wire_observation": None, "stream_observation": None})
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

    def test_missing_mime_stream_observation_is_finite_and_usage_unknown(self):
        sha = self.prepared()
        supplied = dict(STREAM_OBSERVATION)
        self.transport.replies = [FakeTransport.TransportError(
            "model_mismatch", 200, "sse_event", "missing", None, supplied)]
        result = probe.run(self.root, self.state, sha)
        fault = result["first_fault"]
        self.assertEqual(fault["stream_observation"], STREAM_OBSERVATION)
        self.assertIsNone(fault["wire_observation"])
        self.assertEqual((result["known_usage"], result["known_tokens"], result["usage_complete"]),
                         ([], 0, False))
        supplied["terminal_channel"] = "PRIVATE_SENTINEL"
        self.assertEqual(probe.status(self.root, sha), result)
        self.assertNotIn("PRIVATE_SENTINEL", (self.root / "result.json").read_text())

    def test_explicit_sse_stream_observation_is_accepted(self):
        sha = self.prepared()
        self.transport.replies = [FakeTransport.TransportError(
            "missing_usage", 200, "sse_event", "sse", None, STREAM_OBSERVATION)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(probe.status(self.root, sha), result)

    def test_bounded_finalized_output_observations_are_exported(self):
        for index, changes in enumerate((
                {"terminal_output_state": "missing", "finalized_item_count": 1,
                 "output_reconstructed": True},
                {"terminal_output_state": "null", "finalized_item_count": 2,
                 "output_reconstructed": True},
                {"terminal_output_state": "list", "finalized_item_count": 0,
                 "output_reconstructed": False})):
            with self.subTest(changes=changes):
                self.root = Path(self.temp.name) / f"valid_{index}"
                self.root.mkdir(mode=0o700)
                sha = self.prepared()
                supplied = {**STREAM_OBSERVATION, **changes}
                self.transport.replies = [FakeTransport.TransportError(
                    "invalid_output", 200, "sse_event", "sse", None, supplied)]
                result = probe.run(self.root, self.state, sha)
                self.assertEqual(result["first_fault"]["stream_observation"], supplied)
                supplied["terminal_output_state"] = "PRIVATE_SENTINEL"
                self.assertEqual(probe.status(self.root, sha), result)
                self.assertNotIn("PRIVATE_SENTINEL", (self.root / "result.json").read_text())

    def test_invalid_finalized_output_fields_fail_before_export(self):
        changes = (
            {"terminal_output_state": "PRIVATE_SENTINEL"},
            {"terminal_output_state": []},
            {"finalized_item_count": -1},
            {"finalized_item_count": probe.MAX_MEANINGFUL_EVENTS + 1},
            {"finalized_item_count": True},
            {"output_reconstructed": 1},
            {"output_reconstructed": True, "terminal_output_state": "list"},
            {"output_reconstructed": True, "finalized_item_count": 0},
            {"extra": "PRIVATE_SENTINEL"},
        )
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                self.root = Path(self.temp.name) / f"invalid_{index}"
                self.root.mkdir(mode=0o700)
                sha = self.prepared()
                self.transport.replies = [FakeTransport.TransportError(
                    "invalid_output", 200, "sse_event", "sse", None,
                    {**STREAM_OBSERVATION, **change})]
                with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
                    probe.run(self.root, self.state, sha)
                self.assertFalse((self.root / "result.json").exists())
                self.assertEqual(probe.status(self.root, sha),
                                 {"status": "attempted_no_result", "usage_complete": False})

    def test_status_rejects_tampered_finalized_output_fields(self):
        sha = self.prepared()
        self.transport.replies = [FakeTransport.TransportError(
            "invalid_output", 200, "sse_event", "sse", None, STREAM_OBSERVATION)]
        probe.run(self.root, self.state, sha)
        path = self.root / "result.json"
        valid = json.loads(path.read_text())
        changes = (
            {"terminal_output_state": "empty"},
            {"terminal_output_state": "unseen"},
            {"finalized_item_count": 0},
            {"finalized_item_count": probe.MAX_MEANINGFUL_EVENTS + 1},
            {"finalized_item_count": True},
            {"output_reconstructed": "true"},
            {"terminal_output_state": "null", "finalized_item_count": -1},
        )
        for change in changes:
            with self.subTest(change=change):
                variant = json.loads(json.dumps(valid))
                variant["first_fault"]["stream_observation"].update(change)
                path.write_text(json.dumps(variant))
                path.chmod(0o600)
                with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
                    probe.status(self.root, sha)

    def test_malformed_stream_observation_is_never_exported(self):
        sha = self.prepared()
        self.transport.replies = [FakeTransport.TransportError(
            "model_mismatch", 200, "sse_event", "missing", None,
            {**STREAM_OBSERVATION, "raw_text": "PRIVATE_SENTINEL"})]
        with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
            probe.run(self.root, self.state, sha)
        self.assertFalse((self.root / "result.json").exists())
        self.assertEqual(probe.status(self.root, sha),
                         {"status": "attempted_no_result", "usage_complete": False})

    def test_saved_stream_observation_contradictions_rejected(self):
        sha = self.prepared()
        self.transport.replies = [FakeTransport.TransportError(
            "model_mismatch", 200, "sse_event", "missing", None, STREAM_OBSERVATION)]
        probe.run(self.root, self.state, sha)
        path = self.root / "result.json"
        valid = json.loads(path.read_text())
        variants = []
        for key, value in (("stream_observation", {**STREAM_OBSERVATION,
                                      "raw_text": "PRIVATE_SENTINEL"}),
                           ("stream_observation", {**STREAM_OBSERVATION,
                                      "terminal_channel": "PRIVATE_SENTINEL"}),
                           ("stream_observation", {**STREAM_OBSERVATION,
                                      "terminal_model_matches": 1}),
                           ("wire_observation", OBSERVATION),
                           ("http_status", 403), ("body_shape", "other_json"),
                           ("media_type_class", "json"),
                           ("code", "plain_output_mismatch")):
            variant = json.loads(json.dumps(valid))
            variant["first_fault"][key] = value
            variants.append(variant)
        variant = json.loads(json.dumps(valid)); variant["usage_complete"] = True
        variant["known_usage"] = [{"call": 1, "input_tokens": 1, "output_tokens": 1,
                                   "total_tokens": 2, "cached_input_tokens": 0,
                                   "reasoning_output_tokens": 0}]
        variant["known_tokens"] = 2
        variants.append(variant)
        for variant in variants:
            with self.subTest(fault=variant["first_fault"]):
                path.write_text(json.dumps(variant))
                path.chmod(0o600)
                with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
                    probe.status(self.root, sha)

    def test_wrong_mime_branch_allows_sse_media_only_with_bad_headers(self):
        sha = self.prepared()
        malformed = {**OBSERVATION, "header_defect": "missing_separator"}
        self.transport.replies = [FakeTransport.TransportError(
            "invalid_content_type", 200, "other_json", "sse", malformed)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(probe.status(self.root, sha), result)
        path = self.root / "result.json"
        valid = json.loads(path.read_text())
        valid["first_fault"]["wire_observation"] = OBSERVATION
        path.write_text(json.dumps(valid))
        path.chmod(0o600)
        with self.assertRaisesRegex(probe.ProbeError, "result_invalid"):
            probe.status(self.root, sha)

    def test_missing_mime_wrong_encoding_can_have_wire_observation(self):
        sha = self.prepared()
        encoded = {**OBSERVATION, "content_encoding": "gzip"}
        self.transport.replies = [FakeTransport.TransportError(
            "invalid_content_type", 200, "other_json", "missing", encoded)]
        result = probe.run(self.root, self.state, sha)
        self.assertEqual(probe.status(self.root, sha), result)

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
        self.assertEqual(wire._v3._body_prefix(body), "text")
        self.assertEqual(wire._v3._non_sse_classification(body),
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
        source = probe.SOURCES["transport_v5"]
        with mock.patch.dict(probe.SOURCES, {"transport_v5": (source[0], "0" * 64)}):
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
