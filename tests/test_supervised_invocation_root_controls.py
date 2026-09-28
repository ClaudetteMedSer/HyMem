"""Independent root controls for actual owned subprocess cancellation.

All children are local synthetic programs. No provider or database is accessed.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from benchmarks.supervised_invocation import supervise_invocation


def run_child(tmp_path: Path, name: str, program: str, **kwargs):
    return supervise_invocation(
        [sys.executable, "-B", "-c", program],
        cwd=tmp_path.resolve(), env={"PATH": os.defpath},
        output_dir=(tmp_path / name).resolve(),
        timeout_seconds=kwargs.pop("timeout_seconds", 0.3),
        cleanup_seconds=kwargs.pop("cleanup_seconds", 1.0), **kwargs,
    )


def test_real_trickle_cannot_extend_wall_clock_or_publish_late(tmp_path):
    marker = tmp_path / "must-not-be-published"
    program = (
        "import sys,time; from pathlib import Path; "
        "[(sys.stdout.write('keepalive\\n'),sys.stdout.flush(),time.sleep(0.02)) "
        "for _ in range(150)]; "
        f"Path({str(marker)!r}).write_text('late')"
    )
    started = time.monotonic()
    result = run_child(tmp_path, "trickle", program)
    elapsed = time.monotonic() - started
    assert result.status == "timeout"
    assert result.child_reaped and result.cleanup_complete
    assert result.group_absent_after_reap
    assert elapsed < 2.0
    assert not marker.exists()


def test_absolute_expiry_is_not_reset_at_spawn(tmp_path):
    expires = time.monotonic() + 0.2
    time.sleep(0.08)
    started = time.monotonic()
    result = run_child(
        tmp_path, "absolute", "import time; time.sleep(3)",
        timeout_seconds=5.0, deadline_expires_at=expires,
    )
    assert result.status == "timeout"
    assert result.child_reaped and result.cleanup_complete
    assert time.monotonic() - started < 1.5


def test_noncooperative_child_is_killed_and_reaped(tmp_path):
    program = (
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "print('ready',flush=True); time.sleep(30)"
    )
    result = run_child(tmp_path, "ignore-term", program, timeout_seconds=0.4)
    assert result.status == "timeout"
    assert result.returncode == -signal.SIGKILL
    assert result.child_reaped and result.cleanup_complete
    assert result.group_absent_after_reap


def test_parallel_groups_do_not_cancel_each_other(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool:
        timeout = pool.submit(run_child, tmp_path, "parallel-timeout", "import time; time.sleep(3)")
        normal = pool.submit(run_child, tmp_path, "parallel-success", "print('complete')", timeout_seconds=2.0)
        failed, passed = timeout.result(timeout=5), normal.result(timeout=5)
    assert failed.status == "timeout" and passed.status == "completed"
    assert failed.pid != passed.pid
    assert failed.child_reaped and passed.child_reaped
    assert failed.cleanup_complete and passed.cleanup_complete


def test_secret_output_stays_out_of_parent_metadata(tmp_path):
    sentinel = "private-dummy-sentinel-never-a-real-key"
    result = run_child(tmp_path, "private-output", f"import sys; print({sentinel!r}); print({sentinel!r},file=sys.stderr)", timeout_seconds=2.0)
    assert result.status == "completed"
    assert sentinel not in json.dumps(asdict(result), default=str)
    receipts = list((tmp_path / "private-output").glob("*.json"))
    assert receipts
    assert all(sentinel not in path.read_text() for path in receipts)


@pytest.mark.parametrize("mode", ["finite", "keepalive", "idle"])
def test_actual_maintained_sdk_transport_is_owned_and_bounded(tmp_path, mode):
    """Real SDK, dummy loopback only; no provider or monkeypatched transport."""
    import http.server
    import threading

    # A cold maintained-SDK import/constructor can exceed two seconds in the
    # constrained target container. This is still ONE finite startup-inclusive
    # deadline; the stream fixture must outlive it, not win by finishing first.
    invocation_seconds, cleanup_seconds = 10.0, 1.0
    stall_seconds = invocation_seconds + cleanup_seconds + 5.0
    stop = threading.Event()
    counts = {"requests": 0, "newlines": 0}
    failures = []
    body = json.dumps({
        "id": "local-fixture", "object": "chat.completion", "created": 1,
        "model": "dummy-loopback-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                self.connection.settimeout(1)
                counts["requests"] += 1
                assert self.path == "/v1/chat/completions"
                assert self.headers.get("Authorization") == "Bearer dummy-loopback-only"
                size = int(self.headers["Content-Length"])
                assert 0 < size < 4096
                request = json.loads(self.rfile.read(size))
                assert request["model"] == "dummy-loopback-model"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body) + 4 if mode == "finite" else 1_000_000))
                self.end_headers()
                self.wfile.flush()
                if mode == "idle":
                    stop.wait(stall_seconds)
                else:
                    for _ in range(4 if mode == "finite" else int(stall_seconds / 0.02)):
                        if stop.wait(0.02):
                            break
                        self.wfile.write(b"\n")
                        self.wfile.flush()
                        counts["newlines"] += 1
                    if mode == "finite":
                        self.wfile.write(body)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BaseException as exc:
                failures.append(type(exc).__name__)
            finally:
                self.close_connection = True

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=False)
    thread.start()
    expiry = time.monotonic() + invocation_seconds
    payload = json.dumps({
        "endpoint": f"http://127.0.0.1:{server.server_port}/v1",
        "key": "dummy-loopback-only", "expiry": expiry,
    }).encode()
    program = """
import json, sys, time
from pathlib import Path
payload = json.load(sys.stdin)
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.deadline import MonotonicDeadline, use_deadline
from hymem.extraction.llm import LLMRequest
deadline = MonotonicDeadline(payload['expiry'])
client = OpenAICompatibleClient(api_key=payload['key'], base_url=payload['endpoint'],
    model='dummy-loopback-model', thinking='off',
    deployment_revision='offline-supervision', deployment_tenant='offline-supervision')
Path('worker-deadline.json').write_text(json.dumps({
    'expires_at': deadline.expires_at, 'before_complete': time.monotonic()}))
with use_deadline(deadline):
    value = client.complete(LLMRequest(system='Offline dummy control.', user='Return JSON {}.'))
client.close()
print(json.dumps({'accepted': value}))
"""
    try:
        outcome = supervise_invocation(
            [sys.executable, "-B", "-c", program], cwd=tmp_path.resolve(),
            env={"PATH": os.defpath, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            output_dir=(tmp_path / f"sdk-{mode}").resolve(), timeout_seconds=invocation_seconds,
            deadline_expires_at=expiry, cleanup_seconds=cleanup_seconds, stdin_bytes=payload,
        )
    finally:
        stop.set()
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert not thread.is_alive()
    assert not failures
    assert counts["requests"] == 1
    assert outcome.child_reaped and outcome.group_absent_after_reap and outcome.cleanup_complete
    assert outcome.expires_at == expiry
    assert outcome.finished_at < expiry + cleanup_seconds + 0.5
    worker_deadline = json.loads((tmp_path / "worker-deadline.json").read_text())
    assert worker_deadline["expires_at"] == expiry
    assert outcome.started_at <= worker_deadline["before_complete"] < expiry
    for filename in ("launched.json", "terminal.json"):
        assert json.loads((tmp_path / f"sdk-{mode}" / filename).read_text())["expires_at"] == expiry
    if mode == "finite":
        assert outcome.status == "completed"
        assert json.loads((tmp_path / f"sdk-{mode}" / "stdout.bin").read_bytes()) == {"accepted": "{}"}
    else:
        assert outcome.status == "timeout"
        assert outcome.finished_at >= expiry
        assert not (tmp_path / f"sdk-{mode}" / "stdout.bin").read_bytes()
        if mode == "keepalive":
            assert counts["newlines"] > 5
