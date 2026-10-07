"""Offline real-child reproduction of v8's early-readable sentinel cleanup path."""
from __future__ import annotations

import multiprocessing
import os
import time

from benchmarks import chatgpt_plan_responses_v7 as v7
from benchmarks import chatgpt_plan_responses_v8 as v8
from tests.test_chatgpt_plan_responses_root_v1 import terminal


def _early_sentinel_child(send, slots, writer_fd, sent_at, timeout):
    import sys
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(
        AssertionError("network_forbidden")) if event in (
            "socket.connect", "socket.getaddrinfo") else None)
    deadline = time.monotonic() + 3
    while writer_fd.value == -1 and time.monotonic() < deadline:
        time.sleep(.001)
    assert writer_fd.value >= 0
    os.close(writer_fd.value)
    result = v7.parse_stream_events([terminal()])
    sent_at.value = time.monotonic()
    send.send(("ok", result))
    send.close()
    # The process is intentionally alive after the sentinel becomes readable.
    time.sleep(3)


def test_v8_early_readable_sentinel_skips_positive_cleanup_waits(monkeypatch):
    context = multiprocessing.get_context("spawn")
    writer_fd = context.RawValue("i", -1)
    sent_at = context.RawValue("d", 0)
    original_start = v8._OwnedSpawnProcess.start
    processes = []

    def start(process):
        original_start(process)
        processes.append(process)
        # POSIX spawn Popen passes child_r and child_w at the end of _fds.
        writer_fd.value = process._popen._fds[-1]

    monkeypatch.setattr(v8._OwnedSpawnProcess, "start", start)
    try:
        try:
            v8._run_child(_early_sentinel_child, (writer_fd, sent_at), 5)
        except v7.TransportError as exc:
            code = exc.code
        else:
            code = "ok"
        elapsed = time.monotonic() - sent_at.value
        assert code == "cleanup_failure"
        # The claimed .1s + 1s termination allowance was not actually spent.
        assert sent_at.value > 0 and elapsed < .8, elapsed
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
