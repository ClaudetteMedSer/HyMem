"""Offline controls; these never start Codex or contact an external service."""
import ast
from http.client import HTTPConnection
import importlib.util
import json
from pathlib import Path
import threading

import pytest


SOURCE = Path(__file__).parents[1] / "tools/diagnostics/luna_retry_runtime_mock_v2.py"
spec = importlib.util.spec_from_file_location("luna_retry_runtime_mock_v2", SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_exact_disabled_feature_set_matches_frozen_transport():
    base = Path(__file__).parents[1] / "benchmarks/codex_subscription.py"
    tree = ast.parse(base.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "DISABLED_FEATURES"
                              for target in node.targets))
    assert probe.DISABLED_FEATURES == ast.literal_eval(assignment.value)
    settings = probe.overrides(32777)
    assert settings["model"] == "gpt-6-luna"
    assert settings["model_provider"] == "local_mock"
    assert settings["model_providers.local_mock.base_url"] == "http://127.0.0.1:32777/v1"
    assert settings["model_providers.local_mock.requires_openai_auth"] is False
    assert all(settings[f"features.{name}"] is False for name in probe.DISABLED_FEATURES)
    assert not any("chatgpt_base_url" in key or "openai_base_url" in key for key in settings)


def test_method_labels_are_exact_pinned_schema_enums():
    schema = Path("/private/tmp/hymem-luna-0158-schema-cgpdzs/ServerNotification.json")
    if not schema.is_file():
        pytest.skip("pinned generated schema is not present on this host")
    document = json.loads(schema.read_text())
    methods = {variant["properties"]["method"]["enum"][0]
               for variant in document["oneOf"] if "method" in variant.get("properties", {})}
    assert probe.SCHEMA_METHODS == methods
    assert probe.EVENTS <= methods
    assert probe.schema_method_label("model/verification") == "model/verification"
    assert probe.schema_method_label("thread/queue/changed") == "thread/queue/changed"
    assert probe.schema_method_label("private/secret") == "unknown"
    assert probe.schema_method_label(8) == "unknown"


def test_v2_result_contains_finite_failure_attribution():
    row = probe.empty_result("completed")
    assert row["failure_code"] is None and row["failure_method"] is None
    assert "unexpected_event" in probe.FAILURE_CODES
    assert "internal_unverified" in probe.FAILURE_CODES


@pytest.mark.parametrize("bad", [0, 65536, -1, "8000", True])
def test_port_bounds(bad):
    with pytest.raises(probe.ProbeFailure, match="port"):
        probe.overrides(bad)


def test_mock_scenarios_and_request_bounds():
    expected = {
        "completed": ["success", "success"],
        "disconnect_recover": ["disconnect", "success"],
        "http403_recover": ["http403", "success"],
        "disconnect_repeat": ["disconnect", "disconnect"],
    }
    body = json.dumps({"model": probe.MODEL}).encode()
    for case, actions in expected.items():
        state = probe.MockState(case)
        assert [state.next_action("/v1/responses", body) for _ in range(2)] == actions
        for _ in range(probe.MAX_HTTP - 2):
            state.next_action("/v1/responses", body)
        assert state.next_action("/v1/responses", body) == "reject"
        assert state.invalid is True


def test_mock_rejects_unexpected_path_model_and_body_without_retaining_it():
    state = probe.MockState("completed")
    assert state.next_action("/v1/chat/completions", b"secret") == "reject"
    assert state.next_action("/v1/responses", b'{"model":"other"}') == "reject"
    assert state.next_action("/v1/responses", b"not json") == "reject"
    assert state.attempts == 0 and state.invalid is True
    assert "secret" not in vars(state)


def test_local_http_fault_then_success_wire():
    state = probe.MockState("http403_recover")
    try:
        server = probe.ThreadingHTTPServer(("127.0.0.1", 0), probe.handler_for(state))
    except PermissionError:
        pytest.skip("local sandbox prohibits even loopback bind")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for expected in (403, 200):
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("POST", "/v1/responses", json.dumps({"model": probe.MODEL}),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            assert response.status == expected
            response.read()
            connection.close()
        assert state.attempts == 2 and state.invalid is False
    finally:
        server.shutdown()
        server.server_close()


def test_sse_contains_one_completed_response_with_positive_usage_and_final_item():
    wire = probe.response_events()
    events = [json.loads(block.split(b"data: ", 1)[1])
              for block in wire.split(b"\n\n") if block]
    assert events[-1]["type"] == "response.completed"
    complete = events[-1]["response"]
    assert complete["usage"]["total_tokens"] > 0
    assert complete["output"][0]["status"] == "completed"
    assert complete["output"][0]["role"] == "assistant"
    assert any(event["type"] == "response.output_text.delta" for event in events)


def test_error_projection_is_finite_and_ignores_message():
    raw = {"willRetry": True, "error": {"message": "private raw text",
        "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 403}}}}
    projected = probe.classify_error(raw)
    assert projected == {"class": "responseStreamDisconnected", "will_retry": True,
                         "http_status": 403}
    assert "private raw text" not in json.dumps(projected)
    assert probe.classify_error({"willRetry": "yes", "error": {"codexErrorInfo": "private"}}) == {
        "class": "unknown", "will_retry": None, "http_status": None}


def test_binary_target_and_hash_are_pinned_before_execution(tmp_path):
    fake = tmp_path / "codex"
    fake.write_bytes(b"fake")
    with pytest.raises(probe.ProbeFailure, match="binary_path"):
        probe.verified_binary(fake)


def test_one_turn_start_send_site_and_no_real_account_rpcs():
    tree = ast.parse(SOURCE.read_text())
    source = ast.get_source_segment(SOURCE.read_text(),
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "observe"))
    assert source.count('rpc("turn/start"') == 1
    assert 'rpc("account/read"' not in source
    assert 'rpc("account/rateLimits/read"' not in source
    assert "authorization" not in source.lower()
