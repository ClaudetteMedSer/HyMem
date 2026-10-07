"""Finite invented-process checks for the v8 per-child reaping repair."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import multiprocessing
from multiprocessing.connection import wait
import os
import threading
import time

import pytest

from benchmarks import chatgpt_plan_responses_v7 as v7
from benchmarks import chatgpt_plan_responses_v8 as v8
from tests.test_chatgpt_plan_responses_root_v1 import terminal


def _no_network():
    import sys
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(
        AssertionError("network_forbidden")) if event in (
            "socket.connect", "socket.getaddrinfo") else None)


def _successful_child(send, slots, timeout):
    _no_network()
    progress = v7._Progress(slots, timeout)
    progress.mark("child_entry")
    result = v7.parse_stream_events([terminal()])
    progress.mark("result_ipc", ready=True, ipc=True)
    send.send(("ok", result))
    send.close()


def _overlap_child(send, slots, index, stamps, timeout):
    _no_network()
    stamps[index * 2] = time.monotonic()
    time.sleep(.35)
    stamps[index * 2 + 1] = time.monotonic()
    _successful_child(send, slots, timeout)


def _error_child(send, slots, timeout):
    _no_network()
    send.send(("error", "transport_failure", None, None, None, None, None, None))
    send.close()


def _stalled_child(send, slots, timeout):
    _no_network()
    v7._Progress(slots, timeout).mark("child_entry")
    time.sleep(3)


def _other_child():
    return


def _active_pids():
    return {child.pid for child in multiprocessing.active_children()}


def test_foreign_worker_start_cannot_reap_request_child(monkeypatch):
    """Actual BaseProcess.start cleanup runs in a foreign thread after exit."""
    context = multiprocessing.get_context("spawn")
    original_pipe = context.Pipe
    original_init = v8._OwnedSpawnProcess.__init__
    original_waitpid = os.waitpid
    request_exited, other_started = threading.Event(), threading.Event()
    active, outcomes, errors, foreign_waits = {}, {}, [], []
    before = _active_pids()

    def process_init(process, *args, **kwargs):
        original_init(process, *args, **kwargs)
        if threading.current_thread().name == "request-owner":
            active["request"] = process

    class Receive:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def recv(self):
            result = self.connection.recv()
            assert wait([active["request"].sentinel], timeout=5)
            request_exited.set()
            assert other_started.wait(5)
            return result

    def pipe_factory(*args, **kwargs):
        receive, send = original_pipe(*args, **kwargs)
        return Receive(receive), send

    def audited_waitpid(pid, flags):
        if (threading.current_thread().name == "foreign-starter" and
                pid == active["request"].pid):
            foreign_waits.append((pid, flags))
        return original_waitpid(pid, flags)

    def request():
        try:
            outcomes["result"] = v8._run_child(_successful_child, (), 10)
        except BaseException as exc:
            errors.append(exc)

    def foreign_start():
        try:
            assert request_exited.wait(5)
            other = context.Process(target=_other_child)
            active["other"] = other
            other.start()  # Calls multiprocessing.process._cleanup on request.
            other_started.set()
            other.join(2)
            assert not other.is_alive()
        except BaseException as exc:
            errors.append(exc)
            other_started.set()

    monkeypatch.setattr(v8._OwnedSpawnProcess, "__init__", process_init)
    monkeypatch.setattr(context, "Pipe", pipe_factory)
    monkeypatch.setattr(os, "waitpid", audited_waitpid)
    owner = threading.Thread(target=request, name="request-owner")
    foreign = threading.Thread(target=foreign_start, name="foreign-starter")
    try:
        foreign.start()
        owner.start()
        owner.join(6)
        foreign.join(6)
        assert not owner.is_alive() and not foreign.is_alive()
        assert not errors, errors
        assert not foreign_waits
        assert outcomes["result"].text == " invented\n"
    finally:
        other_started.set()
        owner.join(6)
        foreign.join(6)
        for process in active.values():
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join(2)
                process.close()
    assert _active_pids() == before


def test_four_concurrent_workers_and_no_fd_or_child_regression():
    # Prime multiprocessing's once-per-interpreter resource tracker first.
    assert v8._run_child(_successful_child, (), 3).text == " invented\n"
    before = _active_pids()
    fd_before = len(os.listdir("/dev/fd"))
    context = multiprocessing.get_context("spawn")
    stamps = context.RawArray("d", 8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda index: v8._run_child(
            _overlap_child, (index, stamps), 10), range(4)))
    assert all(result.text == " invented\n" for result in results)
    # All four children performed actual work during a common interval.
    assert max(stamps[0::2]) < min(stamps[1::2])
    gc.collect()
    assert _active_pids() == before
    assert len(os.listdir("/dev/fd")) <= fd_before + 2


def test_concurrent_watchdog_signal_cannot_target_reaped_child(monkeypatch):
    context = multiprocessing.get_context("spawn")
    original_pipe = context.Pipe
    original_init = v8._OwnedSpawnProcess.__init__
    original_waitpid = os.waitpid
    original_kill = os.kill
    original_send_signal = v8._OwnedPopen._send_signal
    active, outcomes, errors, signals = {}, {}, [], []
    reaped, attempted, release = (threading.Event() for _ in range(3))
    before = _active_pids()

    def process_init(process, *args, **kwargs):
        original_init(process, *args, **kwargs)
        if threading.current_thread().name == "request-owner":
            active["request"] = process

    class Receive:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def recv(self):
            result = self.connection.recv()
            assert wait([active["request"].sentinel], timeout=2)
            return result

    def pipe_factory(*args, **kwargs):
        receive, send = original_pipe(*args, **kwargs)
        return Receive(receive), send

    def delayed_waitpid(pid, flags):
        answer = original_waitpid(pid, flags)
        if (threading.current_thread().name == "request-owner" and
                pid == active["request"].pid and answer[0] == pid):
            reaped.set()
            assert release.wait(2)
        return answer

    def audited_kill(pid, sig):
        if pid == active["request"].pid:
            if reaped.is_set():
                signals.append(sig)
                return
        return original_kill(pid, sig)

    def send_signal(popen, sig):
        try:
            return original_send_signal(popen, sig)
        finally:
            if threading.current_thread().name != "request-owner":
                attempted.set()

    def request():
        try:
            outcomes["result"] = v8._run_child(_successful_child, (), 3)
        except BaseException as exc:
            errors.append(exc)

    def watchdog_signal():
        try:
            assert reaped.wait(2)
            active["request"].kill()
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(v8._OwnedSpawnProcess, "__init__", process_init)
    monkeypatch.setattr(context, "Pipe", pipe_factory)
    monkeypatch.setattr(os, "waitpid", delayed_waitpid)
    monkeypatch.setattr(os, "kill", audited_kill)
    monkeypatch.setattr(v8._OwnedPopen, "_send_signal", send_signal)
    owner = threading.Thread(target=request, name="request-owner")
    watchdog = threading.Thread(target=watchdog_signal, name="watchdog-signal")
    try:
        watchdog.start()
        owner.start()
        assert reaped.wait(3)
        assert attempted.wait(2)
    finally:
        release.set()
        owner.join(3)
        watchdog.join(3)
        process = active.get("request")
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join(2)
            if not process.is_alive():
                process.close()
    assert not owner.is_alive() and not watchdog.is_alive()
    assert not errors, errors
    assert not signals
    assert outcomes["result"].text == " invented\n"
    assert _active_pids() == before


def test_error_and_timeout_preserve_v7_failure_semantics_and_cleanup():
    before = _active_pids()
    with pytest.raises(v7.TransportError) as error:
        v8._run_child(_error_child, (), 3)
    assert error.value.code == "transport_failure"
    with pytest.raises(v7.TransportError) as timeout:
        v8._run_child(_stalled_child, (), .3)
    assert timeout.value.code == "timeout"
    assert timeout.value.timeout_observation["parent_timeout_site"] == "result_wait"
    assert timeout.value.timeout_observation["child_phase"] == "child_entry"
    assert _active_pids() == before


def test_start_failure_closes_both_pipe_ends_without_child_or_fd_leak(monkeypatch):
    context = multiprocessing.get_context("spawn")
    original_pipe = context.Pipe
    connections = []
    before = _active_pids()
    fd_before = len(os.listdir("/dev/fd"))

    def pipe_factory(*args, **kwargs):
        pair = original_pipe(*args, **kwargs)
        connections.extend(pair)
        return pair

    def fail_start(self):
        raise OSError("invented_start_failure")

    monkeypatch.setattr(context, "Pipe", pipe_factory)
    monkeypatch.setattr(v8._OwnedSpawnProcess, "start", fail_start)
    with pytest.raises(v7.TransportError) as caught:
        v8._run_child(_successful_child, (), .2)
    assert caught.value.code == "transport_failure"
    assert len(connections) == 2 and all(connection.closed for connection in connections)
    gc.collect()
    assert _active_pids() == before
    assert len(os.listdir("/dev/fd")) <= fd_before + 2


def test_genuinely_unreaped_child_still_fails_strict_cleanup(monkeypatch):
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

    monkeypatch.setattr(v8, "_OwnedSpawnProcess", factory)
    with pytest.raises(v7.TransportError) as caught:
        v8._run_child(_successful_child, (), .05)
    assert caught.value.code == "cleanup_failure"
    assert created[0].joins == [0, .1, 1]


def test_v8_preserves_request_parser_and_timeout_limit():
    assert v8.build_request is v7.build_request
    assert v8.parse_stream_events is v7.parse_stream_events
    assert v8._request_once is v7._request_once
    assert v8.TransportError is v7.TransportError
    assert v8.MAX_WALL_SECONDS == 120
    with pytest.raises(v7.TransportError) as caught:
        v8._run_child(_successful_child, (), 120.1)
    assert caught.value.code == "invalid_timeout"
