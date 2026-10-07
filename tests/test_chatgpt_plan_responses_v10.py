"""Offline-only controls for the standalone 300-second Responses transport."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
import threading
import time

import pytest

from benchmarks import chatgpt_plan_responses_v6 as v6
from benchmarks import chatgpt_plan_responses_v10 as wire
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
    assert wire.MAX_WALL_SECONDS == 300
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


def test_virtual_over_120_observation_and_300_bound():
    slots = multiprocessing.get_context("spawn").RawArray("q", wire._SLOTS)
    progress = wire._Progress(slots, 300)
    progress.started = time.monotonic() - 299.5
    progress.mark("stream_read", wire=17, event=True)
    observation = wire._snapshot(slots, 300, "result_recv",
                                 time.monotonic() - 300.5, False)
    assert observation["snapshot_valid"] is True
    assert observation["child_phase"] == "stream_read"
    assert 299_000 <= observation["last_progress_elapsed_ms"] <= 301_000
    assert 300_000 <= observation["parent_elapsed_ms"] <= 301_000
    assert observation["timeout_allowance_ms"] == 300_000
    assert observation["elapsed_saturated"] is False
    assert wire.sanitize_timeout_observation(observation) == observation
    for key, value in (("parent_elapsed_ms", 301_001),
                       ("timeout_allowance_ms", 300_001)):
        assert wire.sanitize_timeout_observation({**observation, key: value}) is None
    saturated = wire._snapshot(slots, 300, "result_recv",
                               time.monotonic() - 302, False)
    assert saturated["parent_elapsed_ms"] == 301_000
    assert saturated["elapsed_saturated"] is True
    progress.started = time.monotonic() - 302
    progress.mark("stream_read")
    child_saturated = wire._snapshot(slots, 300, "result_recv",
                                     time.monotonic() - 1, False)
    assert child_saturated["last_progress_elapsed_ms"] == 301_000
    assert child_saturated["elapsed_saturated"] is True


def test_above_old_limit_accepted_and_above_new_limit_fails_closed():
    before = {child.pid for child in multiprocessing.active_children()}
    assert wire._run_child(_successful_child, (), 180).text == " invented\n"
    assert wire._run_child(_successful_child, (), 300).text == " invented\n"
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_successful_child, (), 300.001)
    assert caught.value.code == "invalid_timeout"
    assert wire.complete.__kwdefaults__["timeout"] == 300
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
