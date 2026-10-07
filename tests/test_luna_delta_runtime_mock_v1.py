"""Offline controls for the standalone localhost delta probe."""
import ast
from http.client import HTTPConnection
import importlib.util
import json
from pathlib import Path
import threading
import time

import pytest


SOURCE = Path(__file__).parents[1] / "tools/diagnostics/luna_delta_runtime_mock_v1.py"
spec = importlib.util.spec_from_file_location("luna_delta_runtime_mock_v1", SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def events_from_wire():
    return [json.loads(block.split(b"data: ", 1)[1])
            for block in probe.response_events().split(b"\n\n") if block]


def test_synthetic_stream_has_5000_fragments_one_final_and_positive_usage():
    events = events_from_wire()
    assert len(events) == probe.DELTA_COUNT + 7
    deltas = [event["delta"] for event in events
              if event["type"] == "response.output_text.delta"]
    assert len(deltas) == 5000 and set(deltas) == {"x"}
    assert events[-1]["type"] == "response.completed"
    completed = events[-1]["response"]
    assert completed["output"][0]["phase"] == "final_answer"
    assert completed["output"][0]["content"][0]["text"] == "".join(deltas)
    assert completed["usage"]["total_tokens"] > 0


def test_frozen_binary_target_config_and_isolation_match_existing_mock():
    prior = Path(__file__).parents[1] / "tools/diagnostics/luna_retry_runtime_mock_v3.py"
    prior_tree = ast.parse(prior.read_text())
    current_tree = ast.parse(SOURCE.read_text())

    def literal(tree, name):
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == name
                                  for target in node.targets))
        return ast.literal_eval(assignment.value)

    assert literal(current_tree, "BINARY_SHA256") == literal(prior_tree, "BINARY_SHA256")
    assert literal(current_tree, "DISABLED_FEATURES") == literal(prior_tree, "DISABLED_FEATURES")
    assert probe.BINARY == probe.Path(
        "/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex")
    settings = probe.overrides(32777)
    assert settings["model_providers.local_mock.base_url"] == "http://127.0.0.1:32777/v1"
    assert settings["model_providers.local_mock.requires_openai_auth"] is False
    assert all(settings[f"features.{feature}"] is False for feature in probe.DISABLED_FEATURES)
    assert not any("chatgpt_base_url" in key or "openai_base_url" in key for key in settings)
    assert probe.CASES == ("baseline", "opt_out", "error_control")
    assert probe.MAX_HTTP_TOTAL == 3 and probe.EVENT_LIMIT == 6000
    assert probe.CASE_SECONDS == 32


def test_mock_accepts_exactly_one_request_per_case():
    body = json.dumps({"model": probe.MODEL}).encode()
    for case in probe.CASES:
        state = probe.MockState(case)
        assert state.accept("/v1/responses", body)
        assert not state.accept("/v1/responses", body)
        assert state.attempts == 1 and state.invalid
    for path, bad_body in (("/v1/chat/completions", body),
                           ("/v1/responses", b"not json"),
                           ("/v1/responses", b'{"model":"other"}')):
        state = probe.MockState("baseline")
        assert not state.accept(path, bad_body)
        assert state.attempts == 0 and state.invalid


def test_local_http_success_and_error_are_bounded():
    body = json.dumps({"model": probe.MODEL})
    wire = probe.response_events()
    for case, expected in (("baseline", 200), ("error_control", 400)):
        state = probe.MockState(case)
        try:
            server = probe.ThreadingHTTPServer(("127.0.0.1", 0), probe.handler_for(state, wire))
        except PermissionError:
            pytest.skip("local sandbox prohibits loopback bind")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("POST", "/v1/responses", body,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            assert response.status == expected
            received = response.read()
            assert received == (wire if case == "baseline" else
                                b'{"error":{"message":"invented bad request","type":"invalid_request_error"}}')
            connection.close()
            assert state.attempts == 1 and not state.invalid
        finally:
            server.shutdown()
            server.server_close()


def app_events(include_delta, include_error=False):
    tid, turn = "thread_mock", "turn_mock"
    common = {"threadId": tid, "turnId": turn}
    result = [{"method": "turn/started", "params": {"threadId": tid,
        "turn": {"id": turn, "status": "inProgress"}}}]
    if include_error:
        result.append({"method": "error", "params": {**common,
            "error": {"message": "private raw error", "codexErrorInfo": "unauthorized"},
            "willRetry": False}})
        return result
    result.append({"method": "item/started", "params": {**common,
        "item": {"type": "agentMessage"}}})
    if include_delta:
        result.extend({"method": probe.OPTOUT_METHOD,
                       "params": {**common, "delta": "x"}}
                      for _ in range(probe.DELTA_COUNT))
    result.extend([
        {"method": "item/completed", "params": {**common,
            "item": {"type": "agentMessage", "phase": "final_answer",
                     "text": probe.TEXT}}},
        {"method": "thread/tokenUsage/updated", "params": {**common,
            "tokenUsage": {"total": {"inputTokens": 8, "outputTokens": 5000,
                                     "totalTokens": 5008},
                           "last": {"inputTokens": 8, "outputTokens": 5000,
                                    "totalTokens": 5008}}}},
        {"method": "turn/completed", "params": {"threadId": tid,
            "turn": {"id": turn, "status": "completed"}}},
    ])
    return result


def test_observer_survives_baseline_flood_and_optout_preserves_lifecycle():
    rows = []
    for case, deltas in (("baseline", True), ("opt_out", False)):
        row = probe.empty_result(case)
        probe.consume(None, app_events(deltas), time.monotonic() + 1,
                      row, "thread_mock", "turn_mock")
        row.update(cleanup_verified=True, mock_boundary_valid=True, http_requests=1)
        rows.append(row)
    assert rows[0]["delta_notifications"] == 5000
    assert rows[1]["delta_notifications"] == 0
    assert all(row["status"] == "completed" and row["final_digest_matches"]
               and row["usage_positive"] for row in rows)
    assert probe.success_pair(rows)
    assert rows[0]["usage_digest"] == rows[1]["usage_digest"]


def test_optout_does_not_filter_error_and_identity_is_checked():
    row = probe.empty_result("error_control")
    probe.consume(None, app_events(False, include_error=True), time.monotonic() + 1, row,
                  "thread_mock", "turn_mock")
    assert row["status"] == "error_observed"
    assert row["error_notifications"] == 1
    assert row["error_classes"] == ["unauthorized"]
    assert "private raw error" not in json.dumps(row)
    wrong = app_events(False)
    wrong[1]["params"]["turnId"] = "wrong"
    with pytest.raises(probe.ProbeFailure, match="identity"):
        probe.consume(None, wrong, time.monotonic() + 1, probe.empty_result("opt_out"),
                      "thread_mock", "turn_mock")


def test_pending_events_cannot_extend_absolute_deadline():
    with pytest.raises(probe.ProbeFailure, match="deadline"):
        probe.consume(None, app_events(False), time.monotonic() - 1,
                      probe.empty_result("opt_out"), "thread_mock", "turn_mock")


def test_only_one_turn_rpc_and_exact_optout_method_in_initialize():
    source = SOURCE.read_text()
    assert source.count('rpc("turn/start"') == 1
    assert 'capabilities["optOutNotificationMethods"] = [OPTOUT_METHOD]' in source
    assert 'OPTOUT_METHOD = "item/agentMessage/delta"' in source
    assert 'rpc("account/read"' not in source
    assert 'rpc("account/rateLimits/read"' not in source
    assert "authorization" not in source.lower()


def test_binary_verification_rejects_any_other_path(tmp_path):
    fake = tmp_path / "codex"
    fake.write_bytes(b"fake")
    with pytest.raises(probe.ProbeFailure, match="binary_path"):
        probe.verified_binary(fake)
