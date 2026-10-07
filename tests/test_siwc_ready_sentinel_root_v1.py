"""Root-owned finite proof of early-sentinel versus exit-status distinction."""
import fcntl
import multiprocessing
from multiprocessing.connection import wait
from multiprocessing import resource_tracker
import os
import stat
import time

from benchmarks import chatgpt_plan_responses_v8 as wire


def _early_sentinel_child(send):
    """Deliberately separate pipe closure and OS exit, without provider I/O."""
    import sys
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(
        AssertionError('network_forbidden')) if event in (
            'socket.connect', 'socket.getaddrinfo') else None)
    excluded = {send.fileno(), resource_tracker._resource_tracker._fd}
    closed = 0
    for fd in range(3, 256):
        if fd in excluded:
            continue
        try:
            if (stat.S_ISFIFO(os.fstat(fd).st_mode)
                    and fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_WRONLY):
                os.close(fd)
                closed += 1
        except OSError:
            pass
    send.send(closed)
    send.close()
    time.sleep(.3)


def test_v8_positive_wait_returns_before_exit_when_sentinel_ready():
    """A positive wait must use its timeout, not treat pipe EOF as reap-ready."""
    ctx = multiprocessing.get_context('spawn')
    receive, send = ctx.Pipe(duplex=False)
    process = wire._OwnedSpawnProcess(target=_early_sentinel_child, args=(send,))
    try:
        process.start()
        send.close()
        assert receive.poll(5)
        assert receive.recv() == 1
        assert wait([process.sentinel], timeout=1)
        assert process.is_alive()
        started = time.monotonic()
        status = process._popen.wait(.8)
        elapsed = time.monotonic() - started
        # Characterize the defect in immutable v8, not desired fixed behavior.
        assert status is None and elapsed < .1
        assert process.is_alive()
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            deadline = time.monotonic() + 2
            while process.is_alive() and time.monotonic() < deadline:
                time.sleep(.01)
            if process.is_alive():
                process.kill()
                deadline = time.monotonic() + 2
                while process.is_alive() and time.monotonic() < deadline:
                    time.sleep(.01)
            assert not process.is_alive()
            process.join(0)
            process.close()
