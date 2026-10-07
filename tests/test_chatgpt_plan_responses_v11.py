"""Offline-only controls for the 600-second, bucketed Responses transport."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import multiprocessing
import os
import threading
import time

import pytest

from benchmarks import chatgpt_plan_responses_v6 as v6
from benchmarks import chatgpt_plan_responses_v11 as wire
from tests.test_chatgpt_plan_responses_root_v1 import terminal
from tests.test_chatgpt_plan_responses_v7 import _call


def _no_network():
    import sys
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(
        AssertionError("network_forbidden")) if event in (
            "socket.connect", "socket.getaddrinfo") else None)


def _successful_child(send, slots, timeout):
    _no_network()
    progress = wire._Progress(slots, timeout)
    progress.mark("child_entry")
    result = wire.parse_stream_events([terminal()])
    progress.mark("result_ipc", ready=True, ipc=True)
    send.send(("ok", result))
    send.close()


def _stalled_child(send, slots, timeout):
    _no_network()
    wire._Progress(slots, timeout).mark("child_entry")
    time.sleep(3)


def _early_sentinel_child(send, slots, writer_fd, timeout):
    _no_network()
    deadline = time.monotonic() + 3
    while writer_fd.value == -1 and time.monotonic() < deadline:
        time.sleep(.001)
    assert writer_fd.value >= 0
    os.close(writer_fd.value)
    _successful_child(send, slots, timeout)
    time.sleep(3)


def _overlap_child(send, slots, index, stamps, timeout):
    _no_network()
    stamps[index * 2] = time.monotonic()
    time.sleep(.3)
    stamps[index * 2 + 1] = time.monotonic()
    _successful_child(send, slots, timeout)


def test_direct_v6_interface_and_unchanged_request_parser_limits():
    assert wire._v6 is v6
    assert wire._v1 is v6._v1 and wire._v2 is v6._v2 and wire._v3 is v6._v3
    assert wire.MAX_WALL_SECONDS == 600
    for name in ("MAX_WIRE_BYTES", "MAX_LINE_BYTES", "MAX_MEANINGFUL_EVENTS",
                 "MAX_OUTPUT_CHARS", "MAX_REQUEST_BYTES", "PROVIDER_CODES"):
        assert getattr(wire, name) == getattr(v6, name)
    request = wire.build_request("invented system", "invented user")
    assert request == v6.build_request("invented system", "invented user")
    assert wire.parse_stream_events([terminal()]) == v6.parse_stream_events([terminal()])
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([])
    assert caught.value.code == "missing_completion"
    with pytest.raises(wire.TransportError) as caught:
        wire.build_request("system", "\ud800")
    assert caught.value.code == "invalid_request"


def test_fake_http_request_and_sse_completion_match_v6(monkeypatch):
    body = b"data: " + json.dumps(terminal()).encode() + b"\n\n"
    old_call, old_calls = _call(monkeypatch, v6, body)
    old = old_call()
    new_call, new_calls = _call(monkeypatch, wire, body)
    assert new_call() == old
    assert new_calls == old_calls


def test_virtual_over_300_observation_and_600_bound():
    slots = multiprocessing.get_context("spawn").RawArray("q", wire._SLOTS)
    progress = wire._Progress(slots, 600)
    progress.started = time.monotonic() - 599.5
    progress.mark("stream_read", wire=17, event=True)
    observation = wire._snapshot(slots, 600, "result_recv",
                                 time.monotonic() - 600.5, False)
    assert observation["snapshot_valid"] is True
    assert observation["child_phase"] == "stream_read"
    assert 599_000 <= observation["last_progress_elapsed_ms"] <= 601_000
    assert 600_000 <= observation["parent_elapsed_ms"] <= 601_000
    assert observation["timeout_allowance_ms"] == 600_000
    assert observation["elapsed_saturated"] is False
    assert observation["event_buckets"]["other"] == 1
    assert wire.sanitize_timeout_observation(observation) == observation
    for key, value in (("parent_elapsed_ms", 601_001),
                       ("timeout_allowance_ms", 600_001)):
        assert wire.sanitize_timeout_observation({**observation, key: value}) is None
    saturated = wire._snapshot(slots, 600, "result_recv",
                               time.monotonic() - 602, False)
    assert saturated["parent_elapsed_ms"] == 601_000
    assert saturated["elapsed_saturated"] is True
    assert saturated["event_buckets"] is None
    progress.started = time.monotonic() - 602
    progress.mark("stream_read")
    child_saturated = wire._snapshot(slots, 600, "result_recv",
                                     time.monotonic() - 1, False)
    assert child_saturated["last_progress_elapsed_ms"] == 601_000
    assert child_saturated["elapsed_saturated"] is True
    assert child_saturated["event_buckets"] is None


def test_above_old_limit_accepted_and_above_new_limit_fails_closed():
    before = {child.pid for child in multiprocessing.active_children()}
    assert wire._run_child(_successful_child, (), 180).text == " invented\n"
    assert wire._run_child(_successful_child, (), 600).text == " invented\n"
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_successful_child, (), 600.001)
    assert caught.value.code == "invalid_timeout"
    assert wire.complete.__kwdefaults__["timeout"] == 600
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_real_short_deadline_reports_timeout_and_reaps_child():
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_stalled_child, (), .35)
    assert caught.value.code == "timeout"
    observation = caught.value.timeout_observation
    assert observation["snapshot_valid"] is True
    assert observation["child_phase"] == "child_entry"
    assert observation["timeout_allowance_ms"] == 350
    assert observation["completion_seen"] is False
    assert observation["event_buckets"] == dict.fromkeys(wire._BUCKETS, 0)
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_early_readable_sentinel_keeps_positive_join_allowance(monkeypatch):
    context = multiprocessing.get_context("spawn")
    writer_fd = context.RawValue("i", -1)
    original_start = wire._OwnedSpawnProcess.start
    processes = []

    def start(process):
        original_start(process)
        processes.append(process)
        writer_fd.value = process._popen._fds[-1]

    monkeypatch.setattr(wire._OwnedSpawnProcess, "start", start)
    try:
        result = wire._run_child(_early_sentinel_child, (writer_fd,), 5)
        assert result.text == " invented\n"
    finally:
        for process in processes:
            deadline = time.monotonic() + 3
            while process.is_alive() and time.monotonic() < deadline:
                time.sleep(.01)
            if process.is_alive():
                process.kill()
                deadline = time.monotonic() + 3
                while process.is_alive() and time.monotonic() < deadline:
                    time.sleep(.01)
            assert not process.is_alive()
            process.join(0)
            process.close()


def test_four_workers_still_overlap_without_child_leak():
    before = {child.pid for child in multiprocessing.active_children()}
    stamps = multiprocessing.get_context("spawn").RawArray("d", 8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda index: wire._run_child(
            _overlap_child, (index, stamps), 5), range(4)))
    assert all(result.text == " invented\n" for result in results)
    assert max(stamps[0::2]) < min(stamps[1::2])
    assert {child.pid for child in multiprocessing.active_children()} == before


@pytest.mark.parametrize("operation", ["poll", "signal"])
def test_status_lock_contention_remains_finite_and_strict(monkeypatch, operation):
    popen = object.__new__(wire._OwnedPopen)
    popen._status_owner = threading.current_thread()
    popen._status_lock = threading.Lock()
    popen.returncode, popen.pid = None, 123456789

    def forbidden(*args):
        raise AssertionError("unverified_status_must_not_reap_or_signal")

    monkeypatch.setattr(os, "waitpid", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    popen._status_lock.acquire()
    started = time.monotonic()
    try:
        with pytest.raises(wire.TransportError) as caught:
            popen.poll() if operation == "poll" else popen._send_signal(9)
    finally:
        popen._status_lock.release()
    assert caught.value.code == "cleanup_failure"
    assert time.monotonic() - started < .5


def test_fixed_disjoint_buckets_from_invented_sse_and_no_content_egress():
    assert wire._LIFECYCLE == v6._ALLOWED - {"response.completed"}
    kinds = ("response.output_text.delta", "response.reasoning_text.delta",
             "response.reasoning_summary_text.delta", "response.created")
    events = [{"type": kind, "delta": "PRIVATE_SENTINEL"} for kind in kinds]
    events.append(terminal())
    body = b": PRIVATE_COMMENT\n\n" + b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    slots = multiprocessing.get_context("spawn").RawArray("q", wire._SLOTS)
    progress = wire._Progress(slots, 600)
    tracker = v6._Observation()
    result = wire.parse_stream_events(wire._events(io.BytesIO(body), tracker, progress), tracker)
    assert result.text == " invented\n"
    observation = wire._snapshot(slots, 600, "result_recv", time.monotonic(), False)
    assert observation["snapshot_valid"] and observation["event_count"] == 5
    assert observation["completion_seen"] is True
    assert observation["wire_bytes"] == len(body)
    assert observation["event_buckets"] == {
        "output_text_delta": 1, "reasoning_text_delta": 1,
        "reasoning_summary_text_delta": 1, "lifecycle": 1,
        "completion": 1, "failure": 0, "other": 0}
    encoded = json.dumps(observation)
    assert "PRIVATE_SENTINEL" not in encoded and "PRIVATE_COMMENT" not in encoded
    assert "response.created" not in encoded


def test_failure_and_unknown_types_have_only_fixed_buckets():
    for kind in ("response.failed", "response.incomplete", "error"):
        assert wire._BUCKETS[wire._event_bucket({"type": kind})] == "failure"
    for kind in ("PRIVATE_ARBITRARY_EVENT", ["PRIVATE"], None):
        assert wire._BUCKETS[wire._event_bucket({"type": kind})] == "other"
    assert wire._BUCKETS[wire._event_bucket({"type": "response.completed"})] == "completion"


def test_bucket_schema_reconciliation_ranges_copy_and_privacy():
    slots = multiprocessing.get_context("spawn").RawArray("q", wire._SLOTS)
    progress = wire._Progress(slots, 600)
    progress.mark("parse", event=True, bucket_index=0)
    progress.mark("parse", event=True, bucket_index=wire._COMPLETION_BUCKET)
    observation = wire._snapshot(slots, 600, "result_recv", time.monotonic(), False)
    assert observation["event_count"] == 2 and observation["completion_seen"] is True
    assert wire.sanitize_timeout_observation(observation) == observation
    copied = wire.sanitize_timeout_observation(observation)
    assert copied is not None
    copied["event_buckets"]["output_text_delta"] = 99
    assert observation["event_buckets"]["output_text_delta"] == 1
    invalid = (
        {**observation, "event_buckets": None},
        {**observation, "event_buckets": {**observation["event_buckets"], "other": 1}},
        {**observation, "event_buckets": {**observation["event_buckets"], "other": True}},
        {**observation, "event_buckets": {**observation["event_buckets"], "other": -1}},
        {**observation, "event_buckets": {**observation["event_buckets"], "other": wire._MAX_EVENTS + 1}},
        {**observation, "event_buckets": {**observation["event_buckets"], "PRIVATE_KEY": 1}},
        {**observation, "event_buckets": {k: v for k, v in observation["event_buckets"].items()
                                          if k != "other"}},
        {**observation, "completion_seen": False},
    )
    assert all(wire.sanitize_timeout_observation(value) is None for value in invalid)


def test_torn_and_saturated_buckets_are_unknown_not_zero():
    slots = multiprocessing.get_context("spawn").RawArray("q", wire._SLOTS)
    progress = wire._Progress(slots, 600)
    progress.mark("parse", event=True, bucket_index=0)
    slots[0] += 1  # Invent a torn, odd seqlock sequence.
    torn = wire._snapshot(slots, 600, "result_recv", time.monotonic(), False)
    assert torn["snapshot_valid"] is False
    assert torn["event_count"] is None and torn["event_buckets"] is None
    assert wire.sanitize_timeout_observation(torn) == torn
    slots[0] += 1
    progress.events = wire._MAX_EVENTS - 1
    progress.buckets[0] = wire._MAX_EVENTS - 1
    progress.mark("parse", event=True, bucket_index=0)
    saturated = wire._snapshot(slots, 600, "result_recv", time.monotonic(), False)
    assert saturated["snapshot_valid"] is True
    assert saturated["event_count"] == wire._MAX_EVENTS
    assert saturated["event_buckets"] is None
    assert wire.sanitize_timeout_observation(saturated) == saturated
    progress.events = 1
    progress.buckets[0] = 1
    progress.mark("parse", wire=wire.MAX_WIRE_BYTES)
    wire_saturated = wire._snapshot(slots, 600, "result_recv", time.monotonic(), False)
    assert wire_saturated["event_buckets"] is None
