"""Bounded, invented localhost probe of app-server delta notification opt-out.

Run under the reviewed network-none host container. This module never loads an
account, uses a provider, or prints protocol text/identifiers. Both completion
cases serve identical synthetic Responses bytes to fresh isolated app servers.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import stat
import subprocess
import tempfile
import threading
import time


SCHEMA = "luna-delta-runtime-mock-v1"
BINARY = Path("/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex")
BINARY_SHA256 = "167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9"
MODEL = "gpt-6-luna"
DISABLED_FEATURES = (
    "apps", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
    "code_mode", "code_mode_host", "computer_use", "goals", "hooks",
    "image_generation", "memories", "multi_agent", "multi_agent_v2", "plugins",
    "remote_plugin", "shell_snapshot", "shell_tool", "skill_mcp_dependency_install",
    "skill_search", "sleep_tool", "tool_suggest", "unified_exec", "view_image",
    "workspace_dependencies", "unbounded_connection_retries",
)
DELTA_COUNT = 5000
TEXT = "x" * DELTA_COUNT
TEXT_DIGEST = hashlib.sha256(TEXT.encode()).hexdigest()
OPTOUT_METHOD = "item/agentMessage/delta"
CASES = ("baseline", "opt_out", "error_control")
MAX_HTTP_TOTAL = 3
CASE_SECONDS = 32
EVENT_LIMIT = 6000
MAX_REQUEST_BYTES = 64_000
ALLOWED_METHODS = frozenset({"error", "turn/started", "turn/completed",
    "item/started", "item/completed", "item/agentMessage/delta",
    "thread/tokenUsage/updated", "thread/started", "thread/status/changed",
    "account/rateLimits/updated", "warning", "remoteControl/status/changed"})
ERROR_CLASSES = frozenset({"httpConnectionFailed", "responseStreamConnectionFailed",
    "responseStreamDisconnected", "responseTooManyFailedAttempts",
    "rateLimitExceeded", "serverOverloaded", "unauthorized", "badRequest", "other"})
FAILURE_CODES = frozenset({"binary", "deadline", "process_exit", "rpc", "rpc_identity",
    "event_limit", "thread", "turn", "unexpected_event", "identity",
    "final_shape", "usage_shape", "internal_unverified"})


class ProbeFailure(Exception):
    pass


def verified_binary(path: Path = BINARY) -> None:
    if path != BINARY:
        raise ProbeFailure("binary_path")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not os.access(path, os.X_OK):
        raise ProbeFailure("binary_type")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != BINARY_SHA256:
        raise ProbeFailure("binary_hash")


def overrides(port: int) -> dict[str, object]:
    if type(port) is not int or not 0 < port < 65536:
        raise ProbeFailure("port")
    return {
        "model": MODEL, "model_provider": "local_mock", "model_reasoning_effort": "low",
        "web_search": "disabled", "project_doc_max_bytes": 0,
        "memories.use_memories": False, "memories.generate_memories": False,
        "developer_instructions": "", "service_tier": "default",
        "features.skip_host_skill_discovery": True,
        **{f"features.{name}": False for name in DISABLED_FEATURES},
        "model_providers.local_mock.name": "Local delta mock",
        "model_providers.local_mock.base_url": f"http://127.0.0.1:{port}/v1",
        "model_providers.local_mock.env_key": "LUNA_MOCK_KEY",
        "model_providers.local_mock.wire_api": "responses",
        "model_providers.local_mock.requires_openai_auth": False,
    }


def error_class(params: object) -> str:
    error = params.get("error") if type(params) is dict else None
    info = error.get("codexErrorInfo") if type(error) is dict else None
    if type(info) is dict and len(info) == 1:
        info = next(iter(info))
    return info if type(info) is str and info in ERROR_CLASSES else "unknown"


class AppServer:
    def __init__(self, binary: Path, cwd: Path, port: int):
        argv = [str(binary), "--strict-config"]
        for key, value in overrides(port).items():
            argv += ["-c", f"{key}={json.dumps(value)}"]
        argv += ["app-server", "--listen", "stdio://"]
        env = {"HOME": str(cwd), "CODEX_HOME": str(cwd / ".codex"),
               "PATH": "/usr/bin:/bin", "TMPDIR": str(cwd),
               "LUNA_MOCK_KEY": "mock-only-key"}
        (cwd / ".codex").mkdir(mode=0o700)
        self.process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            start_new_session=True, bufsize=1)
        self.lines: queue.Queue[object] = queue.Queue(maxsize=512)
        self.next_id = 0
        try:
            threading.Thread(target=self._reader, daemon=True).start()
        except BaseException:
            self.close()
            raise

    def _reader(self):
        try:
            while line := self.process.stdout.readline(1_000_001):
                if len(line) > 1_000_000:
                    break
                try:
                    self.lines.put(json.loads(line), timeout=1)
                except (ValueError, queue.Full):
                    break
        finally:
            try:
                self.lines.put(None, timeout=1)
            except queue.Full:
                pass

    def send(self, method: str, params: dict, notification: bool = False) -> int:
        self.next_id += 1
        message = {"method": method, "params": params}
        if not notification:
            message["id"] = self.next_id
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        return self.next_id

    def receive(self, deadline: float) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProbeFailure("deadline")
        try:
            event = self.lines.get(timeout=remaining)
        except queue.Empty:
            raise ProbeFailure("deadline") from None
        if type(event) is not dict:
            raise ProbeFailure("process_exit")
        return event

    def close(self) -> bool:
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=2)
            os.killpg(self.process.pid, 0)
        except ProcessLookupError:
            return True
        except (OSError, subprocess.SubprocessError):
            return False
        return False


def response_events() -> bytes:
    """One final message fragmented into 5000 provider SSE deltas."""
    rid, iid = "resp_delta_mock", "msg_delta_mock"
    item = {"id": iid, "type": "message", "role": "assistant",
            "phase": "final_answer", "status": "completed",
            "content": [{"type": "output_text", "text": TEXT, "annotations": []}]}
    events = [
        {"type": "response.created", "response": {"id": rid, "object": "response",
            "status": "in_progress", "model": MODEL, "output": []}},
        {"type": "response.output_item.added", "output_index": 0,
            "item": {**item, "status": "in_progress", "content": []}},
        {"type": "response.content_part.added", "item_id": iid, "output_index": 0,
            "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}},
    ]
    events.extend({"type": "response.output_text.delta", "item_id": iid,
                   "output_index": 0, "content_index": 0, "delta": "x"}
                  for _ in range(DELTA_COUNT))
    events.extend([
        {"type": "response.output_text.done", "item_id": iid, "output_index": 0,
            "content_index": 0, "text": TEXT},
        {"type": "response.content_part.done", "item_id": iid, "output_index": 0,
            "content_index": 0, "part": item["content"][0]},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": {"id": rid, "object": "response",
            "status": "completed", "model": MODEL, "output": [item],
            "usage": {"input_tokens": 8, "output_tokens": DELTA_COUNT,
                      "total_tokens": DELTA_COUNT + 8}}},
    ])
    return b"".join((f"event: {event['type']}\ndata: "
                     + json.dumps(event, separators=(",", ":")) + "\n\n").encode()
                    for event in events)


class MockState:
    def __init__(self, case: str):
        self.case = case
        self.attempts = 0
        self.invalid = False
        self.lock = threading.Lock()

    def accept(self, path: str, body: bytes) -> bool:
        with self.lock:
            if path != "/v1/responses" or not body or len(body) > MAX_REQUEST_BYTES:
                self.invalid = True
                return False
            try:
                request = json.loads(body)
            except ValueError:
                self.invalid = True
                return False
            if (type(request) is not dict or request.get("model") != MODEL
                    or self.attempts != 0):
                self.invalid = True
                return False
            self.attempts = 1
            return True


def handler_for(state: MockState, wire: bytes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            size = self.headers.get("Content-Length", "")
            if not size.isdecimal() or int(size) > MAX_REQUEST_BYTES:
                state.invalid = True
                self.send_error(400)
                return
            accepted = state.accept(self.path, self.rfile.read(int(size)))
            if not accepted or state.case == "error_control":
                payload = b'{"error":{"message":"invented bad request","type":"invalid_request_error"}}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
            else:
                payload = wire
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            state.invalid = True
            self.send_error(400)

    return Handler


def empty_result(case: str) -> dict:
    return {"case": case, "status": "unverified", "failure_code": None,
            "failure_method": None, "http_requests": 0,
            "turn_starts": 0, "turn_completed": 0, "same_turn_identity": True,
            "delta_notifications": 0, "item_started": 0, "item_completed": 0,
            "final_items": 0, "final_digest_matches": False,
            "usage_updates": 0, "usage_positive": False, "usage_digest": None,
            "error_notifications": 0, "error_classes": [],
            "cleanup_verified": False, "mock_boundary_valid": False}


def _usage_projection(params: object) -> dict:
    token_usage = params.get("tokenUsage") if type(params) is dict else None
    if type(token_usage) is not dict:
        raise ProbeFailure("usage_shape")
    for section in ("total", "last"):
        values = token_usage.get(section)
        if type(values) is not dict:
            raise ProbeFailure("usage_shape")
        for name in ("inputTokens", "outputTokens", "totalTokens"):
            value = values.get(name)
            if type(value) is not int or value < 0:
                raise ProbeFailure("usage_shape")
    return token_usage


def consume(app, pending: list[dict], deadline: float, row: dict,
            thread_id: str, turn_id: str) -> None:
    for index in range(EVENT_LIMIT):
        if time.monotonic() >= deadline:
            raise ProbeFailure("deadline")
        event = pending[index] if index < len(pending) else app.receive(deadline)
        method, params = event.get("method"), event.get("params")
        if "id" in event or method not in ALLOWED_METHODS:
            row["failure_method"] = method if type(method) is str and method in ALLOWED_METHODS else "unknown"
            raise ProbeFailure("unexpected_event")
        if method in {"turn/started", "turn/completed", "item/started",
                      "item/completed", OPTOUT_METHOD, "thread/tokenUsage/updated", "error"}:
            if type(params) is not dict or params.get("threadId") != thread_id:
                row["same_turn_identity"] = False
                raise ProbeFailure("identity")
            if method.startswith("turn/"):
                turn = params.get("turn")
                valid = type(turn) is dict and turn.get("id") == turn_id
            else:
                valid = params.get("turnId") == turn_id
            if not valid:
                row["same_turn_identity"] = False
                raise ProbeFailure("identity")
        if method == "turn/started":
            row["turn_starts"] += 1
        elif method == "item/started":
            row["item_started"] += 1
        elif method == OPTOUT_METHOD:
            row["delta_notifications"] += 1
        elif method == "item/completed":
            row["item_completed"] += 1
            item = params.get("item")
            if type(item) is dict and item.get("type") == "agentMessage" and item.get("phase") == "final_answer":
                row["final_items"] += 1
                final_text = item.get("text")
                if type(final_text) is not str:
                    raise ProbeFailure("final_shape")
                row["final_digest_matches"] = (
                    hashlib.sha256(final_text.encode()).hexdigest() == TEXT_DIGEST)
        elif method == "thread/tokenUsage/updated":
            projection = _usage_projection(params)
            row["usage_updates"] += 1
            row["usage_positive"] = projection["total"]["totalTokens"] > 0
            row["usage_digest"] = hashlib.sha256(json.dumps(projection,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        elif method == "error":
            row["error_notifications"] += 1
            if len(row["error_classes"]) < 2:
                row["error_classes"].append(error_class(params))
            if row["case"] == "error_control":
                row["status"] = "error_observed"
                return
        elif method == "turn/completed":
            row["turn_completed"] += 1
            turn = params.get("turn")
            row["status"] = "completed" if turn.get("status") == "completed" else "terminal_other"
            return
    raise ProbeFailure("event_limit")


def observe(case: str, wire: bytes, binary: Path = BINARY) -> dict:
    if case not in CASES:
        raise ProbeFailure("case")
    verified_binary(binary)
    row = empty_result(case)
    state = MockState(case)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(state, wire))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    app = None
    tempdir = tempfile.TemporaryDirectory(prefix="luna-local-delta-")
    try:
        app = AppServer.__new__(AppServer)
        try:
            AppServer.__init__(app, binary, Path(tempdir.name), server.server_port)
        except BaseException:
            if getattr(app, "process", None) is not None:
                row["cleanup_verified"] = app.close()
            raise
        deadline = time.monotonic() + CASE_SECONDS
        pending = []

        def rpc(method: str, params: dict, preserve: bool = False) -> dict:
            request_id = app.send(method, params)
            for _ in range(EVENT_LIMIT):
                event = app.receive(deadline)
                if event.get("id") == request_id:
                    response = event.get("result")
                    if type(response) is not dict:
                        raise ProbeFailure("rpc")
                    return response
                if "id" in event:
                    raise ProbeFailure("rpc_identity")
                if preserve:
                    pending.append(event)
            raise ProbeFailure("event_limit")

        capabilities = {"experimentalApi": True}
        if case != "baseline":
            capabilities["optOutNotificationMethods"] = [OPTOUT_METHOD]
        rpc("initialize", {"clientInfo": {"name": "luna_delta_mock", "version": "1"},
            "capabilities": capabilities})
        app.send("initialized", {}, notification=True)
        thread = rpc("thread/start", {"model": MODEL, "modelProvider": "local_mock",
            "allowProviderModelFallback": False, "ephemeral": True, "environments": [],
            "runtimeWorkspaceRoots": [], "selectedCapabilityRoots": [], "dynamicTools": [],
            "baseInstructions": "", "developerInstructions": "",
            "approvalPolicy": "never", "sandbox": "read-only"})
        thread_id = thread.get("thread", {}).get("id")
        if type(thread_id) is not str or not thread_id:
            raise ProbeFailure("thread")
        turn = rpc("turn/start", {"threadId": thread_id,
            "input": [{"type": "text", "text": "Say only the mock response."}],
            "model": MODEL, "effort": "low", "environments": [],
            "runtimeWorkspaceRoots": [], "approvalPolicy": "never",
            "serviceTierForTurn": "default",
            "sandboxPolicy": {"type": "readOnly", "networkAccess": False}}, preserve=True)
        turn_id = turn.get("turn", {}).get("id")
        if type(turn_id) is not str or not turn_id or turn.get("turn", {}).get("status") != "inProgress":
            raise ProbeFailure("turn")
        consume(app, pending, deadline, row, thread_id, turn_id)
    except Exception as exc:
        code = str(exc) if isinstance(exc, ProbeFailure) else "internal_unverified"
        row["failure_code"] = code if code in FAILURE_CODES else "internal_unverified"
    finally:
        row["http_requests"] = state.attempts
        row["mock_boundary_valid"] = not state.invalid
        if app is not None and getattr(app, "process", None) is not None:
            row["cleanup_verified"] = app.close()
        tempdir.cleanup()
        server.shutdown()
        server.server_close()
    return row


def success_pair(rows: list[dict]) -> bool:
    if len(rows) != 2:
        return False
    first, second = rows
    return (all(row["status"] == "completed" and row["failure_code"] is None
                and row["cleanup_verified"] and row["mock_boundary_valid"]
                and row["http_requests"] == 1 and row["turn_starts"] == 1
                and row["turn_completed"] == 1 and row["same_turn_identity"]
                and row["item_started"] >= 1 and row["item_completed"] >= 1
                and row["final_items"] == 1 and row["final_digest_matches"]
                and row["usage_updates"] >= 1 and row["usage_positive"]
                and row["error_notifications"] == 0 for row in rows)
            and first["delta_notifications"] > 4096
            and second["delta_notifications"] == 0
            and first["usage_digest"] == second["usage_digest"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    try:
        verified_binary()
        wire = response_events()
        rows = []
        for case in CASES:
            if case == "error_control" and not success_pair(rows):
                break
            rows.append(observe(case, wire))
            if sum(row["http_requests"] for row in rows) > MAX_HTTP_TOTAL:
                break
            if not rows[-1]["cleanup_verified"] or not rows[-1]["mock_boundary_valid"]:
                break
        verified = (success_pair(rows[:2]) and len(rows) == 3
            and rows[2]["status"] == "error_observed"
            and rows[2]["error_notifications"] >= 1
            and rows[2]["failure_code"] is None
            and rows[2]["cleanup_verified"] and rows[2]["mock_boundary_valid"]
            and rows[2]["http_requests"] == 1
            and rows[2]["same_turn_identity"] and rows[2]["turn_starts"] == 1
            and rows[2]["error_notifications"] == 1
            and sum(row["http_requests"] for row in rows) == MAX_HTTP_TOTAL)
        print(json.dumps({"schema": SCHEMA, "verified": verified, "results": rows},
                         sort_keys=True))
    except (ProbeFailure, OSError):
        print(json.dumps({"schema": SCHEMA, "verified": False,
                          "status": "binary_unverified"}, sort_keys=True))


if __name__ == "__main__":
    main()
