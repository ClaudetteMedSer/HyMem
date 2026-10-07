"""Offline real-child checks for v9's early-sentinel wait contract."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing.connection import wait
import os
import time

import pytest

from benchmarks import chatgpt_plan_responses_v7 as v7
from benchmarks import chatgpt_plan_responses_v9 as v9
from tests.test_chatgpt_plan_responses_root_v1 import terminal
from tests.test_siwc_ready_sentinel_root_v1 import _early_sentinel_child


def _active_pids():
    return {child.pid for child in multiprocessing.active_children()}


def _no_network():
    import sys
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(
        AssertionError("network_forbidden")) if event in (
            "socket.connect", "socket.getaddrinfo") else None)


def _early_ipc_child(send, slots, writer_fd, mode, timeout):
    _no_network()
    deadline = time.monotonic() + 3
    while writer_fd.value == -1 and time.monotonic() < deadline:
        time.sleep(.001)
    assert writer_fd.value >= 0
    os.close(writer_fd.value)
    if mode == "success":
        send.send(("ok", v7.parse_stream_events([terminal()])))
    elif mode == "error":
        send.send(("error", "transport_failure", None, None, None, None, None, None))
    elif mode != "timeout":
        raise AssertionError("invented_mode_invalid")
    if mode != "timeout":
        send.close()
    time.sleep(3)


def _overlap_child(send, slots, index, stamps, timeout):
    _no_network()
    stamps[2 * index] = time.monotonic()
    time.sleep(.25)
    stamps[2 * index + 1] = time.monotonic()
    send.send(("ok", v7.parse_stream_events([terminal()])))
    send.close()


def test_positive_wait_reaps_after_early_sentinel_without_spinning():
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = v9._OwnedSpawnProcess(target=_early_sentinel_child, args=(send,))
    before = _active_pids()
    try:
        process.start()
        send.close()
        assert receive.poll(5) and receive.recv() == 1
        assert wait([process.sentinel], timeout=1)
        assert process.is_alive()
        started = time.monotonic()
        assert process._popen.wait(.8) == 0
        elapsed = time.monotonic() - started
        assert .1 <= elapsed < .8
        assert not process.is_alive()
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join(2)
            process.close()
    assert _active_pids() == before


@pytest.mark.parametrize("mode", ["success", "error", "timeout"])
def test_early_sentinel_ipc_and_timeout_keep_original_outcome(monkeypatch, mode):
    context = multiprocessing.get_context("spawn")
    writer_fd = context.RawValue("i", -1)
    original_start = v9._OwnedSpawnProcess.start
    processes = []
    before = _active_pids()

    def start(process):
        original_start(process)
        processes.append(process)
        writer_fd.value = process._popen._fds[-1]

    monkeypatch.setattr(v9._OwnedSpawnProcess, "start", start)
    try:
        if mode == "success":
            assert v9._run_child(_early_ipc_child, (writer_fd, mode), 5).text == " invented\n"
        else:
            with pytest.raises(v7.TransportError) as caught:
                v9._run_child(_early_ipc_child, (writer_fd, mode), .4 if mode == "timeout" else 5)
            assert caught.value.code == ("timeout" if mode == "timeout" else "transport_failure")
            if mode == "timeout":
                assert caught.value.timeout_observation["parent_timeout_site"] == "result_wait"
        assert all(not process.is_alive() for process in processes)
    finally:
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join(2)
                process.close()
    assert _active_pids() == before


def test_zero_wait_is_immediate_and_foreign_wait_reads_cache_only():
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = v9._OwnedSpawnProcess(target=_early_sentinel_child, args=(send,))
    try:
        process.start()
        send.close()
        assert receive.poll(5) and receive.recv() == 1
        assert wait([process.sentinel], timeout=1) and process.is_alive()
        started = time.monotonic()
        assert process._popen.wait(0) is None
        assert time.monotonic() - started < .1
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(process._popen.wait, .5).result(timeout=1) is None
        assert process._popen.wait(.8) == 0
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join(2)
            process.close()


def test_four_concurrent_children_remain_overlapped_and_reaped():
    context = multiprocessing.get_context("spawn")
    stamps = context.RawArray("d", 8)
    before = _active_pids()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda index: v9._run_child(
            _overlap_child, (index, stamps), 5), range(4)))
    assert all(result.text == " invented\n" for result in results)
    assert max(stamps[0::2]) < min(stamps[1::2])
    assert _active_pids() == before


def test_unreaped_child_still_fails_strict_cleanup(monkeypatch):
    class Unstoppable:
        pid = 123456789

        def __init__(self, *args, **kwargs):
            self.joins = []

        def start(self):
            pass

        def join(self, timeout=None):
            self.joins.append(timeout)

        def is_alive(self):
            return True

        def terminate(self):
            pass

        def kill(self):
            pass

    created = []

    def factory(*args, **kwargs):
        process = Unstoppable(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(v9, "_OwnedSpawnProcess", factory)
    with pytest.raises(v7.TransportError) as caught:
        v9._run_child(_early_ipc_child, (), .05)
    assert caught.value.code == "cleanup_failure"
    assert created[0].joins == [0, .1, 1]


def test_v7_wire_parser_and_wall_limit_are_unchanged():
    assert v9.build_request is v7.build_request
    assert v9.parse_stream_events is v7.parse_stream_events
    assert v9._request_once is v7._request_once
    assert v9.TransportError is v7.TransportError
    assert v9.MAX_WALL_SECONDS == 120
    with pytest.raises(v7.TransportError) as caught:
        v9._run_child(_early_ipc_child, (), 120.1)
    assert caught.value.code == "invalid_timeout"
