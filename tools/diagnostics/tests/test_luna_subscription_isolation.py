"""Offline checks; these never spawn Codex or make model calls."""
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "luna_subscription_isolation.py"
SPEC = importlib.util.spec_from_file_location("luna_isolation_test_subject", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class Request:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class Transport:
    class SubscriptionTransportError(RuntimeError):
        pass


class FakeClient:
    def __init__(self, replies, *, admission=None):
        self.replies = iter(replies)
        self.admission = admission or {"config_isolation_admitted": True,
                                       "auth": "chatgpt", "model": "gpt-6-luna",
                                       "inference_enabled": False}
        self.observed_turns = 0
        self.observed_tokens = None
        self.usage_complete = True
        self.requested_controls = []
        self.inference_accepted = False
        self.requests = []

    def preflight(self):
        return self.admission

    def complete(self, request):
        if not self.inference_accepted:
            raise AssertionError("inference not admitted")
        self.requests.append(request)
        self.observed_turns += 1
        self.observed_tokens = (self.observed_tokens or 0) + 10
        self.requested_controls.append({"temperature_requested": 0.0,
                                        "temperature_effective": None})
        answer = next(self.replies)
        return answer(request) if callable(answer) else answer


class ProbeTests(unittest.TestCase):
    def test_root_warning_observer_exports_hashes_not_text_and_does_not_recurse(self):
        class Inner:
            process = type("Process", (), {"pid": 123})()
            bound_thread_id = "thr"

            def receive(self):
                return {"method": "warning", "params": {
                    "message": "private diagnostic text", "threadId": "thr"}}

            def rpc(self, *args, **kwargs):
                return self.receive()

        registry = {"process_ids": [], "thread_ids": [], "closed": [], "warning_metadata": []}
        wrapped = probe.TrackingSession(Inner(), registry)
        assert wrapped.rpc("test", {})["method"] == "warning"
        assert len(registry["warning_metadata"]) == 1
        assert registry["warning_metadata"][0]["bound_thread_matches"] is True
        assert "private diagnostic text" not in str(registry)

    def run_fake(self, replies, *, admission=None, clock=None):
        client = FakeClient(replies, admission=admission)
        report = probe.run_probe(Transport, Request, "/unused",
                                 client_factory=lambda: client,
                                 clock=clock or (lambda: 0.0))
        return client, report

    def test_success_and_no_marker_leak_to_second_prompt(self):
        def first(request):
            import json
            return request.user.removeprefix("Return exactly this JSON object: ")
        client, report = self.run_fake([first, '{"marker":null}',
                                        '{"access":"unavailable"}'])
        self.assertTrue(report["ok"])
        self.assertEqual(report["checks"], ["preflight", "probe1", "probe2", "probe3"])
        self.assertEqual(report["observed_turns"], 3)
        self.assertEqual(report["known_tokens"], 30)
        self.assertNotIn("Return exactly this JSON object:", client.requests[1].user)
        self.assertNotIn("private.txt", str(report))
        self.assertNotIn("write-marker.txt", str(report))

    def test_fail_stop_and_call_cap(self):
        client, report = self.run_fake(['{"marker":"wrong"}', '{"marker":null}',
                                        '{"access":"unavailable"}'])
        self.assertEqual(report["stop_code"], "probe1_failed")
        self.assertEqual(client.observed_turns, 1)
        self.assertFalse(report["ok"])

    def test_preflight_rejection_prevents_calls(self):
        client, report = self.run_fake([], admission={"auth": "api_key"})
        self.assertEqual(report["stop_code"], "preflight_rejected")
        self.assertEqual(client.observed_turns, 0)

    def test_wall_limit_prevents_first_call(self):
        ticks = iter([0.0, 481.0, 481.0])
        client, report = self.run_fake([], clock=lambda: next(ticks))
        self.assertEqual(report["stop_code"], "wall_limit")
        self.assertEqual(client.observed_turns, 0)

    def test_wrong_fresh_context_reply_stops_before_tool_probe(self):
        def first(request):
            return request.user.removeprefix("Return exactly this JSON object: ")
        client, report = self.run_fake([first, '{"marker":"remembered"}',
                                        '{"access":"unavailable"}'])
        self.assertEqual(report["stop_code"], "probe2_failed")
        self.assertEqual(client.observed_turns, 2)

    def test_tool_claim_without_exact_unavailable_fails(self):
        def first(request):
            return request.user.removeprefix("Return exactly this JSON object: ")
        client, report = self.run_fake([first, '{"marker":null}',
                                        '{"access":"yes"}'])
        self.assertEqual(report["stop_code"], "probe3_failed")
        self.assertEqual(client.observed_turns, 3)

    def test_overrun_is_reported_without_clamping(self):
        def overrun(request):
            client.observed_turns = 4
            return request.user.removeprefix("Return exactly this JSON object: ")
        client = FakeClient([overrun])
        report = probe.run_probe(Transport, Request, "/unused",
                                 client_factory=lambda: client, clock=lambda: 0.0)
        self.assertEqual(report["stop_code"], "turn_overrun")
        self.assertEqual(report["observed_turns"], 4)

    def test_incomplete_usage_fails_after_first_call(self):
        def missing_usage(request):
            client.usage_complete = False
            client.observed_tokens = None
            return request.user.removeprefix("Return exactly this JSON object: ")
        client = FakeClient([missing_usage])
        report = probe.run_probe(Transport, Request, "/unused",
                                 client_factory=lambda: client, clock=lambda: 0.0)
        self.assertEqual(report["stop_code"], "usage_incomplete")
        self.assertEqual(report["observed_turns"], 1)

    def test_transport_error_redacted(self):
        self.assertEqual(probe._safe_transport_code(
            Transport.SubscriptionTransportError("secret: raw model text")),
            "transport_protocol_failure")
        self.assertEqual(probe._safe_transport_code(
            Transport.SubscriptionTransportError("unexpected_notification:item/completed")),
            "unexpected_notification:item/completed")


if __name__ == "__main__":
    unittest.main()
