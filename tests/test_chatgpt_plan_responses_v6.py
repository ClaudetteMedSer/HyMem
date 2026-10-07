"""Offline HTTPResponse, finalized-item, privacy, and child controls for v6."""
from __future__ import annotations

import http.client
import io
import json
import multiprocessing
import time
import copy

import pytest

from benchmarks import chatgpt_plan_responses_v6 as wire
from tests.test_chatgpt_plan_responses_root_v1 import terminal


SECRET = "PRIVATE_STREAM_SENTINEL_v6"


def _sse(*events):
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)


def _response(headers: bytes, body: bytes, status: int = 200):
    class Socket:
        def makefile(self, *_args):
            return io.BytesIO(f"HTTP/1.1 {status} Reply\r\n".encode() + headers + b"\r\n" + body)
    response = http.client.HTTPResponse(Socket())
    response.begin()
    return response


def _request(monkeypatch, reply):
    calls = []

    class Connection:
        def __init__(self, host, port, **_kwargs):
            assert (host, port) == ("api.openai.com", 443)

        def request(self, method, path, body, headers):
            calls.append((method, path, body, headers))

        def getresponse(self):
            return reply

        def close(self):
            pass

    monkeypatch.setattr(wire.http.client, "HTTPSConnection", Connection)
    request = wire.build_request("system", "user")
    return lambda: wire._request_once(wire.Credentials("invented-token"), request, 1), calls


def _failure(call, code):
    with pytest.raises(wire.TransportError) as caught:
        call()
    assert caught.value.code == code
    assert SECRET not in repr(caught.value) + str(caught.value)
    return caught.value


def test_actual_http_response_missing_mime_valid_stream(monkeypatch):
    body = (b"\xef\xbb\xbf: invented comment\r\nid: private-id\r\nretry: 1000\r\n\r\n"
            b"event: response.reasoning_text.done\r\ndata: " + json.dumps({
                "type": "response.reasoning_text.done", "text": SECRET}).encode() + b"\r\n\r\n"
            b"event: response.completed\r\ndata: " + json.dumps(terminal()).encode() + b"\r\n\r\n"
            b"data: [DONE]\r\n\r\n")
    call, calls = _request(monkeypatch, _response(b"X-Invented: yes\r\n", body))
    result = call()
    assert result.total_tokens == 13 and result.text == " invented\n"
    assert len(calls) == 1
    method, path, encoded, headers = calls[0]
    assert (method, path) == ("POST", "/v1/responses")
    assert json.loads(encoded) == wire.build_request("system", "user")
    assert headers == {"Authorization": "Bearer invented-token",
                       "Content-Type": "application/json", "Accept": "text/event-stream"}


@pytest.mark.parametrize("state", ["missing", "null", "empty"])
def test_http_finalized_done_recovers_absent_null_or_empty_output(monkeypatch, state):
    end = terminal()
    item = end["response"].pop("output")[0]
    item["id"] = "invented-item"
    if state != "missing":
        end["response"]["output"] = None if state == "null" else []
    done = {"type": "response.output_item.done", "output_index": 3, "item": item}
    for headers in (b"Content-Type: text/event-stream\r\n", b""):
        call, _ = _request(monkeypatch, _response(headers, _sse(done, end)))
        result = call()
        assert result.text == " invented\n"
        assert (result.input_tokens, result.output_tokens, result.total_tokens,
                result.cached_input_tokens, result.reasoning_output_tokens) == (9, 4, 13, 3, 2)


def _empty_terminal_and_done():
    end = terminal()
    item = end["response"].pop("output")[0]
    end["response"]["output"] = []
    done = {"type": "response.output_item.done", "output_index": 2,
            "item": {**item, "id": "invented-final-item"}}
    return end, done


def test_empty_reconstruction_has_finite_observation_on_terminal_failure():
    end, done = _empty_terminal_and_done()
    end["response"]["model"] = "invented-other-model"
    exc = _failure(lambda: wire.parse_stream_events([done, end]), "model_mismatch")
    assert exc.stream_observation == {
        "event_type": "response.completed", "terminal_status": "completed",
        "terminal_model_matches": False, "terminal_output_kind": "missing",
        "terminal_channel": "missing", "terminal_content_kind": "missing",
        "terminal_output_state": "empty", "finalized_item_count": 1,
        "output_reconstructed": True}


@pytest.mark.parametrize("events,code", [
    (lambda end, done: [end], "invalid_output"),
    (lambda end, done: [{"type": "response.output_text.delta", "delta": SECRET}, end],
     "invalid_output"),
    (lambda end, done: [{"type": "response.output_item.added", "output_index": 1,
                         "item": {"type": "message", "id": "unfinished"}}, done, end],
     "incomplete_response"),
    (lambda end, done: [done, done, end], "invalid_event"),
])
def test_empty_requires_unique_completed_final_items(events, code):
    end, done = _empty_terminal_and_done()
    exc = _failure(lambda: wire.parse_stream_events(events(end, done)), code)
    assert exc.stream_observation["output_reconstructed"] is False


@pytest.mark.parametrize("mutate,code", [
    (lambda end, done: end["response"].update(status="incomplete"), "incomplete_response"),
    (lambda end, done: end["response"].update(model="invented-other-model"), "model_mismatch"),
    (lambda end, done: end["response"].pop("usage"), "missing_usage"),
    (lambda end, done: end["response"]["usage"].update(total_tokens=14), "invalid_usage"),
    (lambda end, done: end["response"].update(error={"code": "private"}), "response_failure"),
    (lambda end, done: done["item"]["content"][0].update(type="refusal", refusal=SECRET),
     "unsupported_output"),
    (lambda end, done: done["item"].update(status="incomplete"), "incomplete_response"),
])
def test_empty_keeps_final_item_and_terminal_validation(mutate, code):
    end, done = _empty_terminal_and_done()
    mutate(end, done)
    _failure(lambda: wire.parse_stream_events([done, end]), code)


@pytest.mark.parametrize("replacement", [{"private": SECRET}, "", 0, False])
def test_malformed_terminal_output_is_never_replaced(replacement):
    end, done = _empty_terminal_and_done()
    end["response"]["output"] = replacement
    exc = _failure(lambda: wire.parse_stream_events([done, end]), "invalid_output")
    assert exc.stream_observation["terminal_output_state"] == "invalid"
    assert exc.stream_observation["output_reconstructed"] is False


def test_nonempty_terminal_list_is_not_replaced():
    end, done = _empty_terminal_and_done()
    end["response"]["output"] = [{"type": "message", "role": "assistant",
                                  "content": [{"type": "refusal", "refusal": SECRET}]}]
    exc = _failure(lambda: wire.parse_stream_events([done, end]), "unsupported_output")
    assert exc.stream_observation["terminal_output_state"] == "list"
    assert exc.stream_observation["output_reconstructed"] is False


def test_reconstruction_observation_and_added_identity_conflict():
    end = terminal()
    item = end["response"].pop("output")[0]
    item["id"] = "invented-item"
    done = {"type": "response.output_item.done", "output_index": 9, "item": item}
    tracker = wire._Observation()
    end["response"]["model"] = "invented-other-model"
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([done, end], tracker)
    assert caught.value.code == "model_mismatch"
    assert caught.value.stream_observation == {
        "event_type": "response.completed", "terminal_status": "completed",
        "terminal_model_matches": False, "terminal_output_kind": "missing",
        "terminal_channel": "missing", "terminal_content_kind": "missing",
        "terminal_output_state": "missing", "finalized_item_count": 1,
        "output_reconstructed": True}
    added = {"type": "response.output_item.added", "output_index": 9,
             "item": {"type": "message", "id": "different-item"}}
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([added, done, end])
    assert caught.value.code == "invalid_event"
    missing_id = copy.deepcopy(done)
    missing_id["item"].pop("id")
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([added, missing_id, end])
    assert caught.value.code == "invalid_event"


def test_reconstruction_rejects_unfinished_added_item():
    end = terminal()
    item = end["response"].pop("output")[0]
    done = {"type": "response.output_item.done", "output_index": 8,
            "item": {**item, "id": "done-item"}}
    added = {"type": "response.output_item.added", "output_index": 2,
             "item": {"type": "message", "id": "unfinished-item"}}
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([added, done, end])
    assert caught.value.code == "incomplete_response"
    assert caught.value.stream_observation["finalized_item_count"] == 1
    assert caught.value.stream_observation["output_reconstructed"] is False


def test_added_after_done_and_second_terminal_stay_finite():
    end = terminal()
    item = end["response"].pop("output")[0]
    done = {"type": "response.output_item.done", "output_index": 2,
            "item": {**item, "id": "done-item"}}
    late_added = {"type": "response.output_item.added", "output_index": 2,
                  "item": {"type": "message", "id": "other-item"}}
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([done, late_added, end])
    assert caught.value.code == "invalid_event"
    assert caught.value.stream_observation["finalized_item_count"] == 1
    full_terminal = terminal()
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([done, end, full_terminal])
    assert caught.value.code == "event_after_completion"
    assert caught.value.stream_observation["terminal_output_state"] == "list"
    assert caught.value.stream_observation["output_reconstructed"] is False


def test_observation_reconstruction_invariant():
    normal = {"event_type": "response.completed", "terminal_status": "completed",
              "terminal_model_matches": True, "terminal_output_kind": "missing",
              "terminal_channel": "missing", "terminal_content_kind": "missing",
              "terminal_output_state": "missing", "finalized_item_count": 1,
              "output_reconstructed": True}
    assert wire.TransportError("invalid_event", stream_observation=normal).stream_observation == normal
    empty = {**normal, "terminal_output_state": "empty"}
    assert wire.TransportError("invalid_event", stream_observation=empty).stream_observation == empty
    for change in ({"terminal_output_state": "list"}, {"finalized_item_count": 0},
                   {"private_text": SECRET}):
        assert wire.TransportError("invalid_event", stream_observation={
            **normal, **change}).stream_observation is None


def test_done_items_are_bounded_and_copied(monkeypatch):
    end = terminal()
    item = end["response"].pop("output")[0]
    done = {"type": "response.output_item.done", "output_index": 4, "item": item}
    def events():
        yield done
        done["item"]["content"][0]["text"] = SECRET
        yield end
    assert wire.parse_stream_events(events()).text == " invented\n"
    monkeypatch.setattr(wire, "MAX_OUTPUT_CHARS", 3)
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([copy.deepcopy(done), end])
    assert caught.value.code == "output_limit"


@pytest.mark.parametrize("body,code", [
    (b"<!doctype html><html>" + SECRET.encode(), "truncated_stream"),
    (json.dumps(terminal()).encode(), "truncated_stream"),
    (b"garbage: body\n\n" + _sse(terminal()), "invalid_event"),
    (b"event: response.failed\ndata: " + json.dumps(terminal()).encode() + b"\n\n",
     "invalid_event"),
    (b"data: [DONE]\n\n" + _sse(terminal()), "missing_completion"),
])
def test_missing_mime_requires_strict_framing(monkeypatch, body, code):
    call, _ = _request(monkeypatch, _response(b"", body))
    exc = _failure(call, code)
    assert exc.http_status == 200 and exc.media_type_class == "missing"
    assert exc.body_shape == "sse_event"


@pytest.mark.parametrize("headers", [
    b"Content-Type: application/json\r\n",
    b"Content-Type: text/html\r\n",
    b"Content-Type: text/plain\r\n",
    b"Content-Type: text/event-stream\r\nContent-Type: text/html\r\n",
    b"Content-Encoding: gzip\r\n",
    b"BadHeader\r\n",
])
def test_explicit_wrong_mime_or_bad_headers_stay_closed(monkeypatch, headers):
    call, _ = _request(monkeypatch, _response(headers, _sse(terminal())))
    exc = _failure(call, "invalid_content_type")
    assert exc.wire_observation is not None
    assert exc.wire_observation["body_prefix"] == "sse_prefix"


@pytest.mark.parametrize("change,code,model", [
    (lambda e: e["response"].update(model=SECRET), "model_mismatch", False),
    (lambda e: e["response"].pop("usage"), "missing_usage", True),
    (lambda e: e["response"].update(status="incomplete"), "incomplete_response", True),
    (lambda e: e["response"]["output"][0]["content"][0].update(type="refusal", refusal=SECRET),
     "unsupported_output", True),
])
def test_terminal_semantics_and_finite_observation(monkeypatch, change, code, model):
    event = terminal()
    change(event)
    call, _ = _request(monkeypatch, _response(b"", _sse(event)))
    exc = _failure(call, code)
    obs = exc.stream_observation
    assert obs["event_type"] == "response.completed"
    assert obs["terminal_model_matches"] is model
    assert obs["terminal_status"] in {"completed", "incomplete"}
    assert SECRET not in json.dumps(obs)


def test_provider_denial_and_reasoning_done_after_terminal(monkeypatch):
    denial = {"type": "response.failed", "response": {"status": "failed", "error": {
        "code": "subscription_sharing_usage_limit_exceeded", "message": SECRET}}}
    call, _ = _request(monkeypatch, _response(b"", _sse(denial)))
    exc = _failure(call, "subscription_sharing_usage_limit_exceeded")
    assert exc.stream_observation["event_type"] == "response.failed"
    assert exc.stream_observation["terminal_status"] == "failed"
    call, _ = _request(monkeypatch, _response(b"", _sse(terminal(), {
        "type": "response.reasoning_text.done", "text": SECRET})))
    exc = _failure(call, "event_after_completion")
    assert exc.stream_observation["event_type"] == "response.reasoning_text.done"


def test_mismatched_label_observes_only_decoded_dict(monkeypatch):
    body = b"event: response.failed\ndata: " + json.dumps(terminal()).encode() + b"\n\n"
    call, _ = _request(monkeypatch, _response(b"", body))
    exc = _failure(call, "invalid_event")
    assert exc.stream_observation["event_type"] == "response.completed"
    assert exc.stream_observation["terminal_status"] == "completed"


def test_lone_surrogate_event_type_is_finite_invalid_event(monkeypatch):
    body = b'event: response.completed\ndata: {"type":"\\ud800"}\n\n'
    call, _ = _request(monkeypatch, _response(b"", body))
    exc = _failure(call, "invalid_event")
    assert exc.stream_observation["event_type"] == "unknown"


@pytest.mark.parametrize("mutate,expected", [
    (lambda response: response["output"][0].update(type={"private": SECRET}),
     ("other", "missing", "output_text")),
    (lambda response: response["output"][0].update(channel=[SECRET]),
     ("message", "other", "output_text")),
    (lambda response: response["output"][0]["content"][0].update(type={"private": SECRET}),
     ("message", "missing", "other")),
])
def test_malformed_terminal_categories_do_not_raise_or_export(monkeypatch, mutate, expected):
    event = terminal()
    mutate(event["response"])
    call, _ = _request(monkeypatch, _response(b"", _sse(event)))
    exc = _failure(call, "unsupported_output")
    obs = exc.stream_observation
    assert (obs["terminal_output_kind"], obs["terminal_channel"],
            obs["terminal_content_kind"]) == expected
    assert SECRET not in json.dumps(obs)


def test_stream_observation_copies_and_rejects_private_fields():
    observation = {"event_type": "response.completed", "terminal_status": "completed",
                   "terminal_model_matches": True, "terminal_output_kind": "message",
                   "terminal_channel": "missing", "terminal_content_kind": "output_text",
                   "terminal_output_state": "list", "finalized_item_count": 0,
                   "output_reconstructed": False}
    exc = wire.TransportError("invalid_event", stream_observation=observation)
    observation["terminal_channel"] = SECRET
    assert exc.stream_observation["terminal_channel"] == "missing"
    exc.stream_observation["terminal_channel"] = SECRET
    assert SECRET not in repr(exc)
    assert wire.TransportError("invalid_event", stream_observation={
        **observation, "raw_text": SECRET}).stream_observation is None


def test_reasoning_done_counts_toward_event_limit(monkeypatch):
    monkeypatch.setattr(wire, "MAX_MEANINGFUL_EVENTS", 2)
    _failure(lambda: wire.parse_stream_events([
        {"type": "response.reasoning_text.done", "text": SECRET},
        {"type": "response.reasoning_text.done", "text": SECRET},
        terminal()]), "event_limit")


def test_actual_source_wire_and_line_caps(monkeypatch):
    monkeypatch.setattr(wire, "MAX_WIRE_BYTES", 50)
    call, _ = _request(monkeypatch, _response(b"", b": " + b"x" * 49 + b"\n"))
    _failure(call, "wire_limit")
    monkeypatch.setattr(wire, "MAX_WIRE_BYTES", 1_000)
    monkeypatch.setattr(wire, "MAX_LINE_BYTES", 20)
    call, _ = _request(monkeypatch, _response(b"", b"data: " + b"x" * 20 + b"\n"))
    _failure(call, "wire_limit")


def _ipc_error(send, observation, timeout):
    send.send(("error", "invalid_event", 200, "sse_event", "missing", None, observation))
    send.close()


def _hanging_child(send, timeout):
    time.sleep(timeout + 2)


def test_spawn_ipc_sanitization_and_cleanup():
    before = {child.pid for child in multiprocessing.active_children()}
    observation = {"event_type": "response.completed", "terminal_status": "completed",
                   "terminal_model_matches": True, "terminal_output_kind": "message",
                   "terminal_channel": "missing", "terminal_content_kind": "output_text",
                   "terminal_output_state": "list", "finalized_item_count": 0,
                   "output_reconstructed": False}
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_ipc_error, (observation,), 5)
    assert caught.value.stream_observation == observation
    observation["terminal_channel"] = SECRET
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_ipc_error, (observation,), 5)
    assert caught.value.stream_observation is None
    assert SECRET not in repr(caught.value)
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_hanging_child, (), .2)
    assert caught.value.code == "timeout"
    assert {child.pid for child in multiprocessing.active_children()} == before
