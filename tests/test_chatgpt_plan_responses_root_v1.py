"""Root-owned SIWC wire controls using invented data; no account/network use."""
import json
import multiprocessing
import os
import struct
import time

import pytest

from benchmarks import chatgpt_plan_responses_v1 as wire


def terminal():
    return {"type": "response.completed", "response": {
        "model": "gpt-5.6-luna", "status": "completed",
        "usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13,
                  "input_tokens_details": {"cached_tokens": 3},
                  "output_tokens_details": {"reasoning_tokens": 2}},
        "output": [{"type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": " invented\n"}]}]}}


def test_public_route_and_exact_request_contract():
    assert (wire.HOST, wire.PATH, wire.MODEL) == ("api.openai.com", "/v1/responses", "gpt-5.6-luna")
    request = wire.build_request(" system\n", " user\n")
    assert request == {"model": "gpt-5.6-luna", "reasoning": {"effort": "low"},
        "store": False, "stream": True, "instructions": " system\n",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": " user\n"}]}]}


def test_schema_and_output_are_preserved_without_mutable_aliases():
    schema = {"type": "object", "properties": {"flag": {"type": "boolean"}},
              "required": ["flag"], "additionalProperties": False}
    request = wire.build_request("s", "u", schema)
    schema["properties"]["flag"]["type"] = "string"
    assert request["text"]["format"]["schema"]["properties"]["flag"]["type"] == "boolean"
    reply = wire.parse_stream_events([terminal()])
    assert reply.text == " invented\n"
    assert reply.total_tokens == 13
    assert "invented" not in repr(reply)
    assert "invented-secret" not in repr(wire.Credentials("invented-secret"))


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(model="gpt-6-luna"),
    lambda r: r.update(status="incomplete"),
    lambda r: r.update(usage=None),
    lambda r: r["usage"].update(total_tokens=True),
    lambda r: r["usage"].update(total_tokens=14),
    lambda r: r["usage"].update(input_tokens_details={"cached_tokens": 10}),
    lambda r: r["usage"].update(output_tokens_details={"reasoning_tokens": 5}),
    lambda r: r["output"][0]["content"][0].update(type="refusal"),
    lambda r: r["output"][0].update(type="function_call"),
])
def test_terminal_faults_never_accepted(mutate):
    event = terminal()
    mutate(event["response"])
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events([event])


@pytest.mark.parametrize("code", ["subscription_sharing_usage_limit_exceeded",
    "subscription_sharing_usage_unavailable", "subscription_sharing_user_not_eligible",
    "subscription_sharing_route_not_supported", "subscription_sharing_invalid_user",
    "chatpass_v2_scope_not_authorized", "chatpass_v2_invalid_authorization_context"])
def test_server_denial_after_text_is_terminal_and_private(code):
    events = [{"type": "response.output_text.delta", "delta": "private-output"},
        {"type": "response.failed", "response": {"error": {"code": code, "message": "private-account-text"}}}]
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events(events)
    assert caught.value.code == code
    assert "private" not in str(caught.value)
    assert "private" not in repr(caught.value)


def partial_ipc(send, timeout):
    os.write(send.fileno(), struct.pack("!i", 10000) + b"partial")
    time.sleep(10)


def large_result(send, timeout):
    send.send(("ok", wire.Completed("x" * 900000, 9, 4, 13, 3, 2)))
    send.close()


def test_partial_ipc_deadline_and_recursive_direct_child_cleanup():
    before = {p.pid for p in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(wire.TransportError):
        wire._run_child(partial_ipc, (), 0.7)
    assert time.monotonic() - started < 3
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_large_result_pipe_does_not_deadlock():
    result = wire._run_child(large_result, (), 5)
    assert len(result.text) == 900000


class Stream:
    def __init__(self, lines): self.lines = iter(lines)
    def readline(self, limit): return next(self.lines, b"")


def test_bounded_stream_even_with_redundant_deltas(monkeypatch):
    monkeypatch.setattr(wire, "MAX_WIRE_BYTES", 200)
    event = json.dumps({"type": "response.output_text.delta", "delta": "x" * 20}).encode()
    stream = Stream([row for _ in range(20) for row in (b"data: " + event + b"\n", b"\n")])
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events(wire._sse_events(stream))


def test_completed_then_denial_is_not_success():
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events([terminal(), {"type": "response.failed", "response": {
            "error": {"code": "subscription_sharing_usage_limit_exceeded"}}}])


def test_error_event_top_level_code_is_preserved_without_message():
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([{"type": "error", "code": "subscription_sharing_usage_unavailable",
                                  "message": "do-not-export", "param": "private"}])
    assert caught.value.code == "subscription_sharing_usage_unavailable"
    assert "do-not-export" not in repr(caught.value)


def test_error_metadata_cannot_raise_or_export_arbitrary_values():
    error = wire.TransportError("private-arbitrary-message", True, ["private"])
    assert error.code == "transport_failure"
    assert error.http_status is None
    assert error.body_shape is None
    assert "private" not in repr(error)


def test_empty_final_output_is_not_completion_success():
    event = terminal()
    event["response"]["output"][0]["content"][0]["text"] = ""
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events([event])


def test_premature_sse_done_cannot_be_followed_by_completion():
    encoded = json.dumps(terminal()).encode()
    stream = Stream([b"data: [DONE]\n", b"\n", b"data: " + encoded + b"\n", b"\n"])
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events(wire._sse_events(stream))
