"""Real stdio/liveness and single-owner execution regressions for MCP tools."""
from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import inspect
import json
import os
from pathlib import Path
import sys
import threading

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import hymem.server as server


_CHILD = r'''
import os
from pathlib import Path
import sqlite3
import threading
import time
import hymem.server as server

root = Path(os.environ["HYMEM_MCP_TEST_ROOT"])
conn = None
owner = None

def bootstrap():
    global conn, owner
    owner = threading.get_ident()
    # Deliberately retain SQLite's strict default thread-affinity check.
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE events (name TEXT)")
    return object()

def dream():
    assert threading.get_ident() == owner
    conn.execute("INSERT INTO events VALUES ('dream:start')")
    (root / "started").touch()
    deadline = time.monotonic() + 10
    while not (root / "release").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("synthetic release missing")
        time.sleep(0.01)
    conn.execute("INSERT INTO events VALUES ('dream:end')")
    return "dream finished"

def profile():
    assert threading.get_ident() == owner
    conn.execute("INSERT INTO events VALUES ('profile')")
    return str(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

def shutdown():
    assert threading.get_ident() == owner
    conn.close()
    (root / "closed").touch()

server._get_hy = bootstrap
server._shutdown_hy = shutdown
server._do_dream = dream
server._do_profile = profile
server.main()
'''


async def _wait_path(path: Path) -> None:
    async with asyncio.timeout(8):
        while not path.exists():
            await asyncio.sleep(0.01)


def _params(tmp_path: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", _CHILD],
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ, "HYMEM_MCP_TEST_ROOT": str(tmp_path)},
    )


def _text(result) -> str:
    assert result.isError is False
    return result.content[0].text


def test_stdio_ping_remains_live_during_slow_dream(tmp_path):
    async def exercise():
        responsive = False
        async with stdio_client(_params(tmp_path)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                dream = asyncio.create_task(session.call_tool("hymem_dream", {}))
                await _wait_path(tmp_path / "started")
                try:
                    # The real MCP ping must complete BEFORE the synchronous
                    # dream releases. A timeout reproduces the old registration.
                    await asyncio.wait_for(session.send_ping(), timeout=0.75)
                    assert not dream.done()
                    responsive = True
                except TimeoutError:
                    pass
                finally:
                    (tmp_path / "release").touch()
                assert _text(await dream) == "dream finished"
                assert _text(await session.call_tool("hymem_profile", {})) == "3"
        await _wait_path(tmp_path / "closed")
        assert responsive, "the synchronous dream blocked a real MCP ping"

    asyncio.run(exercise())


def test_stdio_disconnect_waits_for_running_work_before_store_close(tmp_path):
    async def exercise():
        params = _params(tmp_path)
        process = await asyncio.create_subprocess_exec(
            params.command,
            *params.args,
            env=params.env,
            cwd=params.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert process.stdin is not None and process.stdout is not None

        async def send(payload):
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()

        try:
            await send({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "worker-test", "version": "1"}},
            })
            initialized = json.loads(
                await asyncio.wait_for(process.stdout.readline(), timeout=8)
            )
            assert initialized["id"] == 1 and "result" in initialized
            await send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            await send({
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "hymem_dream", "arguments": {}},
            })
            await _wait_path(tmp_path / "started")
            process.stdin.close()
            await process.stdin.wait_closed()
            await asyncio.sleep(0.1)
            assert process.returncode is None
            assert not (tmp_path / "closed").exists()
        finally:
            (tmp_path / "release").touch()
            await asyncio.wait_for(process.wait(), timeout=8)
        assert process.returncode == 0
        assert (tmp_path / "closed").exists()

    asyncio.run(exercise())


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(server, "_shutdown_hy", lambda: None)
    result = server._MCPToolWorker()
    try:
        yield result
    finally:
        result.close()


def test_all_twelve_tool_schemas_and_direct_signatures_are_unchanged(worker):
    names = (
        "hymem_capture", "hymem_log", "hymem_dream", "hymem_augment",
        "hymem_ask", "hymem_profile", "hymem_digest", "hymem_alias",
        "hymem_retract", "hymem_add_rule", "hymem_list_rules",
        "hymem_suggest_rules",
    )
    original, adapted = server._get_mcp(), server._get_mcp()
    for name in names:
        function = getattr(server, name)
        wrapper = worker.tool(function)
        assert not inspect.iscoroutinefunction(function)
        assert inspect.iscoroutinefunction(wrapper)
        assert inspect.signature(wrapper) == inspect.signature(function)
        original.tool()(function)
        adapted.tool()(wrapper)

    async def schemas(mcp):
        return [tool.model_dump() for tool in await mcp.list_tools()]

    assert asyncio.run(schemas(original)) == asyncio.run(schemas(adapted))


def test_tool_errors_propagate_without_poisoning_the_worker(worker):
    primary = ValueError("synthetic failure")

    def fail():
        raise primary

    async def exercise():
        with pytest.raises(ValueError) as caught:
            await worker.tool(fail)()
        assert caught.value is primary
        assert await worker.tool(lambda: 42)() == 42
        assert not worker._pending

    asyncio.run(exercise())


def test_request_context_is_copied_but_not_leaked_between_calls(worker):
    marker = ContextVar("mcp_worker_test", default="outside")

    def read_and_change():
        value = marker.get()
        marker.set("worker-only")
        return value

    async def exercise():
        token = marker.set("first request")
        try:
            assert await worker.tool(read_and_change)() == "first request"
            assert marker.get() == "first request"
        finally:
            marker.reset(token)
        assert await worker.tool(read_and_change)() == "outside"
        assert marker.get() == "outside"

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_running", [False, True])
def test_cancellation_never_allows_overlapping_store_calls(worker, cancel_running):
    started, release = threading.Event(), threading.Event()
    events = []
    owners = []

    def slow():
        owners.append(threading.get_ident())
        events.append("slow:start")
        started.set()
        assert release.wait(timeout=8)
        events.append("slow:end")
        return "done"

    def next_call():
        owners.append(threading.get_ident())
        events.append("next")
        return "next"

    async def exercise():
        running = asyncio.create_task(worker.tool(slow)())
        async with asyncio.timeout(3):
            while not started.is_set():
                await asyncio.sleep(0.01)
        queued = asyncio.create_task(worker.tool(next_call)())
        await asyncio.sleep(0)  # submit the second request behind the active one
        cancelled = running if cancel_running else queued
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await asyncio.sleep(0)
        assert events == ["slow:start"]
        assert not (queued if cancel_running else running).done()
        release.set()
        if cancel_running:
            assert await queued == "next"
        else:
            assert await running == "done"
            assert await worker.tool(next_call)() == "next"
        assert events == ["slow:start", "slow:end", "next"]
        assert len(set(owners)) == 1
        assert owners[0] != threading.get_ident()

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_shutdown_cancels_queued_work_and_joins_running_work_on_owner(monkeypatch):
    started, release = threading.Event(), threading.Event()
    closed = threading.Event()
    events = []
    owners = []
    worker = server._MCPToolWorker()

    def slow():
        owners.append(threading.get_ident())
        started.set()
        assert release.wait(timeout=8)
        events.append("completed")

    def shutdown():
        owners.append(threading.get_ident())
        events.append("closed")
        closed.set()

    monkeypatch.setattr(server, "_shutdown_hy", shutdown)
    active = worker.submit(slow)
    assert started.wait(timeout=3)
    queued = worker.submit(lambda: events.append("must not run"))
    with ThreadPoolExecutor(max_workers=1) as closer:
        closing = closer.submit(worker.close)
        try:
            # Cancellation is synchronous in close, before it queues teardown.
            with pytest.raises(FutureCancelledError):
                queued.result(timeout=3)
            assert not closed.is_set() and not closing.done()
            with pytest.raises(RuntimeError, match="shutting down"):
                worker.submit(lambda: None)
        finally:
            release.set()
        active.result(timeout=3)
        closing.result(timeout=3)
    worker.close()  # no repeated store/client close
    assert events == ["completed", "closed"]
    assert len(set(owners)) == 1
    assert not any(thread.is_alive() for thread in worker._executor._threads)


def test_shutdown_failure_still_joins_worker_and_is_not_replaced(monkeypatch):
    primary = RuntimeError("synthetic close failure")
    calls = []

    def shutdown():
        calls.append(threading.get_ident())
        raise primary

    monkeypatch.setattr(server, "_shutdown_hy", shutdown)
    worker = server._MCPToolWorker()
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            worker.close()
        assert caught.value is primary
    assert len(calls) == 1
    assert not any(thread.is_alive() for thread in worker._executor._threads)


def test_interrupted_shutdown_wait_does_not_cancel_queued_cleanup(monkeypatch):
    worker = server._MCPToolWorker()
    started, release = threading.Event(), threading.Event()
    events = []
    primary = KeyboardInterrupt("synthetic shutdown interruption")

    def slow():
        started.set()
        assert release.wait(timeout=8)
        events.append("completed")

    def shutdown():
        events.append("closed")

    monkeypatch.setattr(server, "_shutdown_hy", shutdown)
    worker.submit(slow)
    assert started.wait(timeout=3)
    real_submit = worker._executor.submit

    def submit(function, *args, **kwargs):
        future = real_submit(function, *args, **kwargs)
        if function is shutdown:
            def interrupted_wait(*_args, **_kwargs):
                raise primary
            monkeypatch.setattr(future, "result", interrupted_wait)
        return future

    monkeypatch.setattr(worker._executor, "submit", submit)
    releaser = threading.Timer(0.05, release.set)
    releaser.start()
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            worker.close()
        assert caught.value is primary
    finally:
        release.set()
        releaser.join(timeout=3)
    assert events == ["completed", "closed"]
    assert not any(thread.is_alive() for thread in worker._executor._threads)
