"""Versioned v7 transport with one status reaper per spawned request child.

``BaseProcess.start`` polls all registered children from the starting thread.
For a v7 request child, that thread could reap the child while its request
thread simultaneously polled it, leaving a brief ECHILD/cache-publication gap.
Only the thread that starts a v9 child may reap it. Other threads read the
published return code without calling waitpid. The request thread owns all
bounded joins; the watchdog may still kill a live child at its deadline. A
readable sentinel can precede a waitable exit status, so positive joins keep
their remaining allowance until the owner reaps status or the deadline expires.
"""
from __future__ import annotations

import math
import multiprocessing
from multiprocessing.connection import wait as wait_connections
from multiprocessing import popen_spawn_posix
import os
import threading
import time
from typing import Any

from benchmarks import chatgpt_plan_responses_v7 as _v7

_STATUS_LOCK_SECONDS = .05


def __getattr__(name: str) -> Any:
    # The request, parser, progress, error type, and telemetry are v7 exactly.
    return getattr(_v7, name)


class _OwnedPopen(popen_spawn_posix.Popen):
    def __init__(self, process_obj):
        self._status_owner = threading.current_thread()
        self._status_lock = threading.Lock()
        super().__init__(process_obj)

    def poll(self, flag=None):
        if threading.current_thread() is not self._status_owner:
            return self.returncode
        # Never hold the lock across blocking waitpid. Even after sentinel
        # readiness, WNOHANG is sufficient. A stalled signal holder must not
        # stall request cleanup or let an unverified child count as stopped.
        if not self._status_lock.acquire(timeout=_STATUS_LOCK_SECONDS):
            _v7._fail("cleanup_failure")
        try:
            return super().poll(os.WNOHANG)
        finally:
            self._status_lock.release()

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        if threading.current_thread() is not self._status_owner:
            return self.returncode
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        sentinel_ready = False
        while True:
            status = self.poll()
            if status is not None:
                return status
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None
            if sentinel_ready:
                # EOF may be published before the kernel offers exit status.
                # Keep the join's allowance without polling a ready FD tightly.
                time.sleep(.01 if remaining is None else min(.01, remaining))
            else:
                sentinel_ready = bool(wait_connections([self.sentinel], remaining))

    def _send_signal(self, sig):
        # A concurrent watchdog must not signal a PID already reaped but not
        # yet published by the owner. Its bounded attempt can be retried by
        # the request thread's mandatory cleanup sequence. Owner cleanup
        # cannot contend with its own WNOHANG poll, which has returned before
        # cleanup signals. Owner contention fails closed within the same
        # finite bound; foreign watchdog contention defers to owner cleanup.
        owner = threading.current_thread() is self._status_owner
        if not self._status_lock.acquire(timeout=_STATUS_LOCK_SECONDS):
            if owner:
                _v7._fail("cleanup_failure")
            return
        try:
            if self.returncode is None:
                try:
                    os.kill(self.pid, sig)
                except ProcessLookupError:
                    pass
        finally:
            self._status_lock.release()


class _OwnedSpawnProcess(multiprocessing.get_context("spawn").Process):
    @staticmethod
    def _Popen(process_obj):
        return _OwnedPopen(process_obj)


def _run_child(target, args: tuple[Any, ...], timeout: float):
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or
            not 0 < timeout <= _v7.MAX_WALL_SECONDS):
        _v7._fail("invalid_timeout")
    context = multiprocessing.get_context("spawn")
    slots = context.RawArray("q", _v7._SLOTS)
    receive, send = context.Pipe(duplex=False)
    process = _OwnedSpawnProcess(target=target, args=(send, slots, *args, timeout), daemon=True)
    parent_started = time.monotonic()
    deadline = parent_started + timeout
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        try:
            if process.pid is not None and process.is_alive():
                process.kill()
        except (OSError, ValueError):
            pass

    def timeout_error(site: str) -> None:
        try:
            alive = process.is_alive() if process.pid is not None else False
        except (OSError, ValueError):
            alive = None
        _v7._fail("timeout", timeout_observation=_v7._snapshot(
            slots, timeout, site, parent_started, alive))

    watchdog = threading.Timer(timeout, expire)
    watchdog.daemon = True
    try:
        watchdog.start()
        process.start()
        send.close()
        if not receive.poll(max(0, deadline - time.monotonic())):
            timeout_error("result_wait")
        try:
            message = receive.recv()
        except (EOFError, OSError, ValueError):
            if expired.is_set() or time.monotonic() >= deadline:
                timeout_error("result_recv")
            _v7._fail("transport_failure")
        if expired.is_set() or time.monotonic() >= deadline:
            timeout_error("deadline_after_recv")
        if type(message) is not tuple:
            _v7._fail("transport_failure")
        if len(message) == 2 and message[0] == "ok" and type(message[1]) is _v7.Completed:
            return message[1]
        if len(message) == 8 and message[0] == "error":
            _v7._fail(*message[1:])
        _v7._fail("transport_failure")
    except _v7.TransportError:
        raise
    except (OSError, ValueError, RuntimeError):
        _v7._fail("transport_failure")
    finally:
        watchdog.cancel()
        receive.close()
        send.close()
        if process.pid is not None:
            process.join(timeout=0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            if process.is_alive():
                _v7._fail("cleanup_failure")


def complete(credentials, system: str, user: str, json_schema=None,
             *, timeout: float = _v7.MAX_WALL_SECONDS):
    if type(credentials) is not _v7.Credentials:
        _v7._fail("invalid_credentials")
    try:
        request = _v7.build_request(system, user, json_schema)
    except _v7.TransportError as exc:
        _v7._fail(exc.code, exc.http_status, exc.body_shape, exc.media_type_class,
                  exc.wire_observation, exc.stream_observation)
    return _run_child(_v7._child, (credentials, request), timeout)
