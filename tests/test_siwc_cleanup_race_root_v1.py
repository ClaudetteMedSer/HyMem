"""Root reproduction: real spawned workers, invented result, no provider I/O."""
import multiprocessing
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
import os
import threading
import time

import pytest

from benchmarks import chatgpt_plan_responses_v7 as wire
from tests.test_chatgpt_plan_responses_root_v1 import terminal


def _successful_child(send, slots, timeout):
    import sys
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(
        AssertionError('network_forbidden')) if event in (
            'socket.connect', 'socket.getaddrinfo') else None)
    progress = wire._Progress(slots, timeout)
    progress.mark('child_entry')
    result = wire.parse_stream_events([terminal()])
    progress.mark('result_ipc', ready=True, ipc=True)
    send.send(('ok', result))
    send.close()


def _other_child():
    return


def exercise_cross_worker_reaping(monkeypatch, module=wire, release_after=None,
                                 single_reaper=False):
    """Delay exit-status publication, not child termination or provider work."""
    ctx = multiprocessing.get_context('spawn')
    original_start, original_pipe = BaseProcess.start, ctx.Pipe
    original_waitpid = os.waitpid
    begin_reaping, reaped, release_cache = (threading.Event() for _ in range(3))
    foreign_started = threading.Event()
    active, outcomes, exceptions = {}, {}, []

    def start(process, *args, **kwargs):
        if threading.current_thread().name == 'root-request':
            active['request'] = process
        return original_start(process, *args, **kwargs)

    class Receive:
        def __init__(self, connection):
            self.connection = connection
        def __getattr__(self, name):
            return getattr(self.connection, name)
        def recv(self):
            result = self.connection.recv()
            assert wait([active['request'].sentinel], timeout=5)
            begin_reaping.set()
            assert (foreign_started if single_reaper else reaped).wait(5)
            return result

    def pipe_factory(*args, **kwargs):
        receive, send = original_pipe(*args, **kwargs)
        if threading.current_thread().name == 'root-request':
            receive = Receive(receive)
        return receive, send

    def delayed_waitpid(pid, flags):
        answer = original_waitpid(pid, flags)
        if (threading.current_thread().name == 'root-reaper'
                and pid == active['request'].pid and answer[0] == pid):
            reaped.set()
            if not release_cache.wait(5):
                raise AssertionError('cache_gate_timeout')
        return answer

    def request():
        try:
            outcomes['result'] = module._run_child(_successful_child, (), 10)
        except module.TransportError as exc:
            outcomes['failure'] = exc.code
        except BaseException as exc:
            exceptions.append(type(exc).__name__)

    def start_other_worker():
        try:
            assert begin_reaping.wait(5)
            other = ctx.Process(target=_other_child)
            active['other'] = other
            other.start()  # Actual BaseProcess.start -> _cleanup -> other child poll.
            foreign_started.set()
            other.join(2)
            assert not other.is_alive()
        except BaseException as exc:
            exceptions.append(type(exc).__name__)

    monkeypatch.setattr(BaseProcess, 'start', start)
    monkeypatch.setattr(ctx, 'Pipe', pipe_factory)
    monkeypatch.setattr(os, 'waitpid', delayed_waitpid)
    request_thread = threading.Thread(target=request, name='root-request')
    reaper_thread = threading.Thread(target=start_other_worker, name='root-reaper')
    publication_timer = None
    try:
        reaper_thread.start()
        request_thread.start()
        assert (foreign_started if single_reaper else reaped).wait(5)
        if release_after is not None:
            publication_timer = threading.Timer(release_after, release_cache.set)
            publication_timer.start()
        # v7's complete cleanup path returns within this interval despite its
        # process already having been reaped. A repaired path may wait for the
        # in-progress status publication, so release on a short finite bound.
        request_thread.join(1.5)
        outcomes['finished_before_cache_publication'] = (
            not request_thread.is_alive() and not release_cache.is_set())
        with pytest.raises(ProcessLookupError):
            os.kill(active['request'].pid, 0)
        outcomes['child_absent_at_kernel'] = True
        outcomes['foreign_thread_reaped_request'] = reaped.is_set()
    finally:
        release_cache.set()
        if publication_timer is not None:
            publication_timer.cancel()
            publication_timer.join(1)
        request_thread.join(5)
        reaper_thread.join(5)
        for process in active.values():
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join(2)
                assert not process.is_alive()
                process.close()
    assert not request_thread.is_alive() and not reaper_thread.is_alive()
    assert not exceptions, exceptions
    return outcomes


def test_v7_real_spawn_false_cleanup_failure(monkeypatch):
    result = exercise_cross_worker_reaping(monkeypatch)
    assert result['child_absent_at_kernel'] is True
    assert result['finished_before_cache_publication'] is True
    assert result['failure'] == 'cleanup_failure'
    assert 'result' not in result


def test_v7_false_cleanup_failure_with_short_publication_delay(monkeypatch):
    result = exercise_cross_worker_reaping(monkeypatch, release_after=0.05)
    assert result['child_absent_at_kernel'] is True
    assert result['failure'] == 'cleanup_failure'


def test_v8_real_spawn_owner_success_despite_foreign_start(monkeypatch):
    from benchmarks import chatgpt_plan_responses_v8 as repaired
    result = exercise_cross_worker_reaping(monkeypatch, repaired, single_reaper=True)
    assert result['child_absent_at_kernel'] is True
    assert result['foreign_thread_reaped_request'] is False
    assert result['result'].text == ' invented\n'
    assert 'failure' not in result


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
def test_v8_retains_actual_timeout_observation_and_cleanup(
        monkeypatch, mode, phase, count, completed):
    from benchmarks import chatgpt_plan_responses_v8 as repaired
    from tests import test_siwc_timeout_root_v1 as controls
    monkeypatch.setattr(controls, 'wire', repaired)
    controls.test_actual_child_timeout_phase_without_network(mode, phase, count, completed)


def test_v8_retains_partial_ipc_deadline_and_no_children(monkeypatch):
    from benchmarks import chatgpt_plan_responses_v8 as repaired
    from tests import test_siwc_timeout_root_v1 as controls
    monkeypatch.setattr(controls, 'wire', repaired)
    controls.test_partial_result_delivery_remains_bounded_and_attributed()


@pytest.mark.parametrize('operation', ['poll', 'signal'])
def test_v8_owner_status_lock_contention_is_finite_and_fail_closed(monkeypatch, operation):
    from benchmarks import chatgpt_plan_responses_v8 as repaired
    popen = object.__new__(repaired._OwnedPopen)
    popen._status_owner = threading.current_thread()
    popen._status_lock = threading.Lock()
    popen.returncode, popen.pid = None, 123456789
    def forbidden(*args):
        raise AssertionError('unverified_status_must_not_reap_or_signal')
    monkeypatch.setattr(os, 'waitpid', forbidden)
    monkeypatch.setattr(os, 'kill', forbidden)
    popen._status_lock.acquire()
    started = time.monotonic()
    try:
        with pytest.raises(repaired.TransportError) as caught:
            popen.poll() if operation == 'poll' else popen._send_signal(9)
    finally:
        popen._status_lock.release()
    assert caught.value.code == 'cleanup_failure'
    assert time.monotonic() - started < .5
