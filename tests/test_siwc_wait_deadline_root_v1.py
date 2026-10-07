"""Independent bounded-wait repair controls; no model or credential access."""
import multiprocessing
from multiprocessing.connection import wait
import threading
import time

import pytest

from benchmarks import chatgpt_plan_responses_v9 as repaired
from tests.test_siwc_ready_sentinel_root_v1 import _early_sentinel_child


@pytest.mark.parametrize('allowance,finished', [(0, False), (.04, False), (.8, True)])
def test_actual_early_sentinel_honors_positive_deadline(allowance, finished):
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = repaired._OwnedSpawnProcess(target=_early_sentinel_child, args=(send,))
    before = {child.pid for child in multiprocessing.active_children()}
    try:
        process.start()
        send.close()
        assert receive.poll(5) and receive.recv() == 1
        assert wait([process.sentinel], timeout=1) and process.is_alive()
        started = time.monotonic()
        status = process._popen.wait(allowance)
        elapsed = time.monotonic() - started
        assert elapsed < allowance + .2
        if finished:
            assert status == 0 and not process.is_alive() and elapsed >= .15
        else:
            assert status is None and process.is_alive()
            assert elapsed >= allowance * .8
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join(1)
            assert not process.is_alive()
            process.close()
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_foreign_wait_never_reaps_or_blocks(monkeypatch):
    instance = object.__new__(repaired._OwnedPopen)
    instance._status_owner = threading.current_thread()
    instance.returncode = None
    def forbidden(*args, **kwargs):
        raise AssertionError('foreign_wait_must_not_read_sentinel_or_poll')
    instance.poll = forbidden
    monkeypatch.setattr(repaired, 'wait_connections', forbidden)
    outputs = []
    thread = threading.Thread(target=lambda: outputs.append(instance.wait(.8)))
    thread.start()
    thread.join(.3)
    assert not thread.is_alive() and outputs == [None]


@pytest.mark.parametrize('mode,phase,count,completed', [
    ('startup', 'unknown', None, None),
    ('request', 'request_send', 0, False),
    ('headers', 'headers_wait', 0, False),
    ('stream', 'stream_read', 0, False),
    ('parse', 'parse', 0, False),
    ('close', 'response_close', 0, False),
    ('event_then_stall', 'stream_read', 1, False),
    ('completion_then_stall', 'stream_read', 1, True),
])
def test_timeout_phase_and_cleanup_unchanged(monkeypatch, mode, phase, count, completed):
    from tests import test_siwc_timeout_root_v1 as controls
    monkeypatch.setattr(controls, 'wire', repaired)
    controls.test_actual_child_timeout_phase_without_network(mode, phase, count, completed)


def test_partial_ipc_deadline_unchanged(monkeypatch):
    from tests import test_siwc_timeout_root_v1 as controls
    monkeypatch.setattr(controls, 'wire', repaired)
    controls.test_partial_result_delivery_remains_bounded_and_attributed()


def test_owner_only_reaping_still_excludes_foreign_worker(monkeypatch):
    from tests.test_siwc_cleanup_race_root_v1 import exercise_cross_worker_reaping
    result = exercise_cross_worker_reaping(monkeypatch, repaired, single_reaper=True)
    assert result['child_absent_at_kernel'] and not result['foreign_thread_reaped_request']
    assert result['result'].text == ' invented\n' and 'failure' not in result


@pytest.mark.parametrize('operation', ['poll', 'signal'])
def test_held_status_lock_remains_finite_fail_closed(monkeypatch, operation):
    import os
    instance = object.__new__(repaired._OwnedPopen)
    instance._status_owner = threading.current_thread()
    instance._status_lock = threading.Lock()
    instance.returncode, instance.pid = None, 123456789
    def forbidden(*args):
        raise AssertionError('unverified_status_must_not_reap_or_signal')
    monkeypatch.setattr(os, 'waitpid', forbidden)
    monkeypatch.setattr(os, 'kill', forbidden)
    instance._status_lock.acquire()
    started = time.monotonic()
    try:
        with pytest.raises(repaired.TransportError) as caught:
            instance.poll() if operation == 'poll' else instance._send_signal(9)
    finally:
        instance._status_lock.release()
    assert caught.value.code == 'cleanup_failure' and time.monotonic() - started < .5
