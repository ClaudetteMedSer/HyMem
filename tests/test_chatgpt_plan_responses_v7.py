"""Offline-only progress, parity, privacy, and spawned timeout controls."""
from __future__ import annotations

import http.client
import io
import json
import multiprocessing
import time

import pytest

from benchmarks import chatgpt_plan_responses_v6 as v6
from benchmarks import chatgpt_plan_responses_v7 as wire
from tests.test_chatgpt_plan_responses_root_v1 import terminal


SECRET = "PRIVATE_TIMEOUT_SENTINEL"


def _response(body: bytes, status: int = 200):
    class Socket:
        def makefile(self, *_args):
            return io.BytesIO(f"HTTP/1.1 {status} Reply\r\n".encode() +
                              b"Content-Type: text/event-stream\r\n\r\n" + body)
    reply = http.client.HTTPResponse(Socket())
    reply.begin()
    return reply


def _call(monkeypatch, module, body):
    calls = []

    class Connection:
        def __init__(self, host, port, **kwargs):
            assert (host, port) == ("api.openai.com", 443)
            calls.append(("connect", kwargs["timeout"]))

        def request(self, method, path, body, headers):
            calls.append((method, path, json.loads(body), headers))

        def getresponse(self):
            return _response(body)

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    request = module.build_request("invented system", "invented user")
    return lambda: module._request_once(module.Credentials("invented-token"), request, 1), calls


@pytest.mark.parametrize("event", [terminal(),
    {"type": "response.completed", "response": {**terminal()["response"], "model": SECRET}},
    {"type": "response.failed", "response": {"status": "failed", "error": {
        "code": "subscription_sharing_usage_limit_exceeded", "message": SECRET}}}])
def test_request_parser_and_error_parity(monkeypatch, event):
    body = b"data: " + json.dumps(event).encode() + b"\n\n"
    old_call, old_calls = _call(monkeypatch, v6, body)
    try:
        old = ("ok", old_call())
    except v6.TransportError as exc:
        old = ("error", exc.code, exc.http_status, exc.body_shape,
               exc.media_type_class, exc.wire_observation, exc.stream_observation)
    new_call, new_calls = _call(monkeypatch, wire, body)
    try:
        new = ("ok", new_call())
    except wire.TransportError as exc:
        new = ("error", exc.code, exc.http_status, exc.body_shape,
               exc.media_type_class, exc.wire_observation, exc.stream_observation)
    assert new == old
    assert new_calls == old_calls


def test_actual_request_progress_counts_wire_and_completion(monkeypatch):
    body = b"data: " + json.dumps(terminal()).encode() + b"\n\n"
    _call(monkeypatch, wire, body)
    slots = multiprocessing.get_context("spawn").RawArray("q", wire._SLOTS)
    progress = wire._Progress(slots, 1)
    progress.mark("child_entry")
    request = wire.build_request("invented system", "invented user")
    result = wire._request_once(wire.Credentials("invented-token"), request, 1, progress)
    assert result.text == " invented\n"
    observation = wire._snapshot(slots, 1, "none", time.monotonic())
    assert observation["snapshot_valid"] is True
    assert observation["child_phase"] == "response_close"
    assert observation["wire_bytes"] == len(body)
    assert observation["event_count"] == 1
    assert observation["completion_seen"] is True
    assert observation["result_ready"] is False


def _stall_at_phase(send, slots, phase, timeout):
    progress = wire._Progress(slots, timeout)
    progress.mark("child_entry")
    progress.mark(phase, wire=17, event=True, completed=True,
                  ready=phase == "result_ipc", ipc=phase == "result_ipc")
    time.sleep(timeout + 2)


@pytest.mark.parametrize("phase", ["request_send", "headers_wait", "response_check", "stream_read",
                                   "parse", "response_close", "result_ipc"])
def test_spawned_stall_survives_watchdog_and_cleans_child(phase):
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_stall_at_phase, (phase,), .35)
    exc = caught.value
    assert exc.code == "timeout"
    observation = exc.timeout_observation
    assert observation["child_phase"] == phase
    assert observation["snapshot_valid"] is True
    assert observation["parent_timeout_site"] == "result_wait"
    assert observation["wire_bytes"] == 17
    assert observation["event_count"] == 1
    assert observation["completion_seen"] is True
    assert observation["result_ready"] is (phase == "result_ipc")
    assert observation["result_ipc_started"] is (phase == "result_ipc")
    assert 0 <= observation["last_progress_elapsed_ms"] <= 1000
    assert 0 <= observation["parent_elapsed_ms"] <= 1000
    assert observation["elapsed_saturated"] is False
    assert {child.pid for child in multiprocessing.active_children()} == before


def _stall_in_non_sse_body(send, slots, timeout):
    class Reply:
        status = 503

        def getheader(self, name, default=None):
            return "application/json" if name == "Content-Type" else default

        def read(self, _size):
            time.sleep(timeout + 2)
            return b""

    class Connection:
        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            return Reply()

        def close(self):
            pass

    wire.http.client.HTTPSConnection = Connection
    progress = wire._Progress(slots, timeout)
    progress.mark("child_entry")
    wire._request_once(wire.Credentials("invented-token"),
                       wire.build_request("system", "user"), timeout, progress)


def test_spawned_non_sse_error_body_stall_reports_response_check():
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_stall_in_non_sse_body, (), .35)
    assert caught.value.code == "timeout"
    observation = caught.value.timeout_observation
    assert observation["child_phase"] == "response_check"
    assert observation["event_count"] == 0
    assert observation["completion_seen"] is False
    assert {child.pid for child in multiprocessing.active_children()} == before


def _stall_before_entry(send, slots, timeout):
    time.sleep(timeout + 2)


def _stall_during_snapshot(send, slots, timeout):
    slots[0] = 1
    slots[1] = 4
    time.sleep(timeout + 2)


@pytest.mark.parametrize("target", [_stall_before_entry, _stall_during_snapshot])
def test_unknown_snapshot_does_not_invent_child_progress(target):
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(target, (), .3)
    observation = caught.value.timeout_observation
    assert observation["snapshot_valid"] is False
    assert observation["child_phase"] == "unknown"
    for key in ("last_progress_elapsed_ms", "wire_bytes", "event_count",
                "completion_seen", "result_ready", "result_ipc_started"):
        assert observation[key] is None


def _ipc_error(send, slots, observation, code, timeout):
    send.send(("error", code, 200, "sse_event", "sse", None,
               None, observation))


def test_strict_ipc_observation_and_repr_privacy():
    invented = {"child_phase": "parse", "parent_timeout_site": "none",
                "last_progress_elapsed_ms": 1, "parent_elapsed_ms": 2,
                "elapsed_saturated": False, "snapshot_valid": True,
                "timeout_allowance_ms": 1000, "wire_bytes": 17,
                "event_count": 1, "completion_seen": True, "result_ready": False,
                "result_ipc_started": False, "child_alive_when_sampled": None}
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_ipc_error, (invented, "timeout"), 2)
    assert caught.value.timeout_observation == invented
    invented["child_phase"] = SECRET
    assert SECRET not in repr(caught.value)
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_ipc_error, (invented, "timeout"), 2)
    assert caught.value.timeout_observation is None
    assert SECRET not in repr(caught.value)
    assert wire.sanitize_timeout_observation({**invented, "private": SECRET}) is None
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_ipc_error, ({**invented, "child_phase": "parse"}, "invalid_event"), 2)
    assert caught.value.timeout_observation is None


def test_snapshot_sanitizer_rejects_bools_as_counts_and_bad_fields():
    slots = multiprocessing.get_context("spawn").RawArray("q", wire._SLOTS)
    progress = wire._Progress(slots, 1)
    progress.mark("child_entry")
    valid = wire._snapshot(slots, 1, "result_wait", time.monotonic(), True)
    assert wire.sanitize_timeout_observation(valid) == valid
    for change in ({"event_count": True}, {"wire_bytes": -1},
                   {"child_phase": SECRET}, {"snapshot_valid": False},
                   {"parent_timeout_site": SECRET}, {"parent_elapsed_ms": float("nan")},
                   {"last_progress_elapsed_ms": None}, {"result_ready": None},
                   {"timeout_allowance_ms": 120001},
                   {"completion_seen": True, "event_count": 0},
                   {"child_phase": "result_ipc", "result_ipc_started": False},
                   {"result_ipc_started": True, "result_ready": False},
                   {"private": SECRET}):
        assert wire.sanitize_timeout_observation({**valid, **change}) is None


def test_no_retry_and_original_timeout_validation():
    assert wire.MAX_WALL_SECONDS == v6.MAX_WALL_SECONDS == 120
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_stall_before_entry, (), 120.1)
    assert caught.value.code == "invalid_timeout"


def test_public_v7_error_type_and_constants():
    assert (wire.MAX_LINE_BYTES, wire.MAX_OUTPUT_CHARS, wire.PROVIDER_CODES) == (
        v6.MAX_LINE_BYTES, v6.MAX_OUTPUT_CHARS, v6.PROVIDER_CODES)
    assert wire._sanitize_stream_observation is v6._sanitize_stream_observation
    with pytest.raises(wire.TransportError) as caught:
        wire.build_request("system", "\ud800")
    assert caught.value.code == "invalid_request"
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([])
    assert caught.value.code == "missing_completion"
