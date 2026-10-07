"""Bounded localhost-only retry probe for the pinned Codex 0.158.0 app server.

Run only inside the reviewed host-owned network-restricted transient unit. This
uses invented text and a fresh HOME/CODEX_HOME; it never reads account files.
Output is a finite metadata projection, never protocol payloads or identifiers.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time

SCHEMA = "luna-retry-runtime-mock-v1"
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
CASES = ("completed", "disconnect_recover", "http403_recover", "disconnect_repeat")
EVENTS = frozenset({"error", "turn/started", "turn/completed", "turn/failed",
    "item/started", "item/completed", "item/agentMessage/delta",
    "thread/tokenUsage/updated", "thread/started", "thread/status/changed"})
ERROR_CLASSES = frozenset({"httpConnectionFailed", "responseStreamConnectionFailed",
    "responseStreamDisconnected", "responseTooManyFailedAttempts",
    "rateLimitExceeded", "serverOverloaded", "unauthorized", "badRequest", "other"})
MAX_HTTP = 4
CASE_SECONDS = 28


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
        "model_providers.local_mock.name": "Local retry mock",
        "model_providers.local_mock.base_url": f"http://127.0.0.1:{port}/v1",
        "model_providers.local_mock.env_key": "LUNA_MOCK_KEY",
        "model_providers.local_mock.wire_api": "responses",
        "model_providers.local_mock.requires_openai_auth": False,
    }


def response_events() -> bytes:
    # Invented provider data. A positive usage count is essential to distinguish
    # a completed turn from one that only emitted a final item.
    rid, iid, content = "resp_mock_1", "msg_mock_1", "mock complete"
    events = [
        {"type": "response.created", "response": {"id": rid, "object": "response", "status": "in_progress", "model": MODEL, "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"id": iid, "type": "message", "role": "assistant", "phase": "final_answer", "status": "in_progress", "content": []}},
        {"type": "response.content_part.added", "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}},
        {"type": "response.output_text.delta", "item_id": iid, "output_index": 0, "content_index": 0, "delta": content},
        {"type": "response.output_text.done", "item_id": iid, "output_index": 0, "content_index": 0, "text": content},
        {"type": "response.content_part.done", "item_id": iid, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": content, "annotations": []}},
        {"type": "response.output_item.done", "output_index": 0, "item": {"id": iid, "type": "message", "role": "assistant", "phase": "final_answer", "status": "completed", "content": [{"type": "output_text", "text": content, "annotations": []}]}},
        {"type": "response.completed", "response": {"id": rid, "object": "response", "status": "completed", "model": MODEL, "output": [{"id": iid, "type": "message", "role": "assistant", "phase": "final_answer", "status": "completed", "content": [{"type": "output_text", "text": content, "annotations": []}]}], "usage": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}}},
    ]
    return b"".join((f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n").encode() for event in events)


class MockState:
    def __init__(self, case: str):
        self.case = case
        self.attempts = 0
        self.invalid = False
        self.capped = False
        self.lock = threading.Lock()

    def next_action(self, path: str, body: bytes) -> str:
        with self.lock:
            # The only allowed outbound HTTP is a Responses POST to our local
            # endpoint. Never retain the request or echo it into output.
            if path != "/v1/responses" or len(body) > 64_000 or not body:
                self.invalid = True
                return "reject"
            try:
                request = json.loads(body)
            except ValueError:
                self.invalid = True
                return "reject"
            if type(request) is not dict or request.get("model") != MODEL:
                self.invalid = True
                return "reject"
            if self.attempts >= MAX_HTTP:
                self.invalid = True
                self.capped = True
                return "reject"
            self.attempts += 1
            if self.case == "completed":
                return "success"
            if self.case == "http403_recover":
                return "http403" if self.attempts == 1 else "success"
            if self.case == "disconnect_recover":
                return "disconnect" if self.attempts == 1 else "success"
            return "disconnect"


def handler_for(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            size = self.headers.get("Content-Length", "")
            if not size.isdecimal() or int(size) > 64_000:
                state.invalid = True
                self.send_error(400)
                return
            action = state.next_action(self.path, self.rfile.read(int(size)))
            if action in {"reject", "http403"}:
                payload = b'{"error":{"message":"mock forbidden","type":"mock_error"}}'
                self.send_response(403 if action == "http403" else 400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                if state.attempts >= MAX_HTTP:
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            payload = response_events()
            if action == "disconnect":
                payload = payload.split(b"event: response.output_text.delta", 1)[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            if action == "disconnect":
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()
            if state.attempts >= MAX_HTTP:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def do_GET(self):
            state.invalid = True
            self.send_error(400)

    return Handler


class AppServer:
    def __init__(self, binary: Path, cwd: Path, port: int):
        config = overrides(port)
        argv = [str(binary), "--strict-config"]
        for key, value in config.items():
            argv += ["-c", f"{key}={json.dumps(value)}"]
        argv += ["app-server", "--listen", "stdio://"]
        env = {"HOME": str(cwd), "CODEX_HOME": str(cwd / ".codex"),
               "PATH": "/usr/bin:/bin", "TMPDIR": str(cwd), "LUNA_MOCK_KEY": "mock-only-key"}
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
        try:
            event = self.lines.get(timeout=max(0.001, deadline - time.monotonic()))
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


def empty_result(case: str) -> dict:
    return {"case": case, "status": "unverified", "http_requests": 0,
        "single_turn_start": False, "same_turn_identity": True,
        "event_counts": {name: 0 for name in sorted(EVENTS)},
        "errors": [], "final_seen": False, "usage": "absent",
        "cleanup_verified": False, "mock_boundary_valid": False}


def classify_error(params: object) -> dict:
    if type(params) is not dict:
        return {"class": "unknown", "will_retry": None, "http_status": None}
    error = params.get("error")
    info = error.get("codexErrorInfo") if type(error) is dict else None
    status = None
    if type(info) is str:
        kind = info if info in ERROR_CLASSES else "unknown"
    elif type(info) is dict and len(info) == 1:
        kind, detail = next(iter(info.items()))
        if kind not in ERROR_CLASSES or type(detail) is not dict:
            kind = "unknown"
        else:
            status = detail.get("httpStatusCode")
    else:
        kind = "unknown"
    if type(status) is not int or not 100 <= status <= 599:
        status = None
    return {"class": kind, "will_retry": params.get("willRetry") if type(params.get("willRetry")) is bool else None,
            "http_status": status}


def observe(case: str, binary: Path = BINARY) -> dict:
    if case not in CASES:
        raise ProbeFailure("case")
    verified_binary(binary)
    result = empty_result(case)
    state = MockState(case)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    app = None
    tempdir = tempfile.TemporaryDirectory(prefix="luna-local-retry-")
    try:
        # The temporary HOME stays alive through process-group cleanup below.
        with nullcontext(tempdir.name) as directory:
            app = AppServer.__new__(AppServer)
            try:
                AppServer.__init__(app, binary, Path(directory), server.server_port)
            except BaseException:
                if getattr(app, "process", None) is not None:
                    result["cleanup_verified"] = app.close()
                raise
            deadline = time.monotonic() + CASE_SECONDS
            pending = []
            def rpc(method, params, preserve=False):
                request_id = app.send(method, params)
                for _ in range(512):
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

            rpc("initialize", {"clientInfo": {"name": "luna_retry_mock", "version": "1"},
                "capabilities": {"experimentalApi": True}})
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
                "input": [{"type": "text", "text": "Say mock complete."}],
                "model": MODEL, "effort": "low", "environments": [],
                "runtimeWorkspaceRoots": [], "approvalPolicy": "never",
                "serviceTierForTurn": "default",
                "sandboxPolicy": {"type": "readOnly", "networkAccess": False}}, preserve=True)
            result["single_turn_start"] = True
            turn_id = turn.get("turn", {}).get("id")
            if type(turn_id) is not str or not turn_id or turn.get("turn", {}).get("status") != "inProgress":
                raise ProbeFailure("turn")
            last_usage = None
            for index in range(512):
                event = pending[index] if index < len(pending) else app.receive(deadline)
                method = event.get("method")
                params = event.get("params")
                if "id" in event or method not in EVENTS | {"warning", "remoteControl/status/changed"}:
                    raise ProbeFailure("unexpected_event")
                if method in EVENTS:
                    result["event_counts"][method] += 1
                if method == "error":
                    if type(params) is not dict or params.get("threadId") != thread_id or params.get("turnId") != turn_id:
                        result["same_turn_identity"] = False
                    if len(result["errors"]) < 8:
                        result["errors"].append(classify_error(params))
                if method in {"turn/started", "turn/completed", "turn/failed", "item/started", "item/completed", "item/agentMessage/delta", "thread/tokenUsage/updated", "thread/status/changed"}:
                    if type(params) is not dict or params.get("threadId") != thread_id:
                        result["same_turn_identity"] = False
                    elif method.startswith("turn/"):
                        nested_turn = params.get("turn")
                        if type(nested_turn) is not dict or nested_turn.get("id") != turn_id:
                            result["same_turn_identity"] = False
                    elif method != "thread/status/changed" and params.get("turnId") != turn_id:
                        result["same_turn_identity"] = False
                if method == "item/completed" and type(params) is dict:
                    item = params.get("item")
                    if type(item) is dict and item.get("type") == "agentMessage" and item.get("phase") == "final_answer":
                        result["final_seen"] = True
                if method == "thread/tokenUsage/updated" and type(params) is dict:
                    token_usage = params.get("tokenUsage")
                    total = token_usage.get("total") if type(token_usage) is dict else None
                    usage = total.get("totalTokens") if type(total) is dict else None
                    if type(usage) is not int or usage < 0:
                        result["usage"] = "invalid"
                    elif last_usage is not None and usage < last_usage:
                        result["usage"] = "regressed"
                    elif result["usage"] not in {"invalid", "regressed"}:
                        result["usage"] = "positive" if usage > 0 else "zero"
                    if type(usage) is int and usage >= 0:
                        last_usage = usage
                if method == "turn/completed":
                    nested_turn = params.get("turn") if type(params) is dict else None
                    result["status"] = "completed" if type(nested_turn) is dict and nested_turn.get("status") == "completed" else "terminal_other"
                    break
                if method == "turn/failed":
                    result["status"] = "failed"
                    break
            else:
                raise ProbeFailure("event_limit")
            result["cleanup_verified"] = app.close()
            app = None
    except Exception as exc:
        result["status"] = str(exc) if isinstance(exc, ProbeFailure) and str(exc) in {
            "deadline", "process_exit", "rpc", "rpc_identity", "event_limit", "thread", "turn"} else "unverified"
    finally:
        result["http_requests"] = state.attempts
        result["mock_boundary_valid"] = not state.invalid
        if app is not None:
            result["cleanup_verified"] = app.close()
        tempdir.cleanup()
        server.shutdown()
        server.server_close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    args = parser.parse_args()
    if not args.run:
        return
    try:
        verified_binary()
        rows = []
        for case in CASES:
            row = observe(case)
            rows.append(row)
            if (not row["cleanup_verified"] or not row["mock_boundary_valid"]
                    or not row["same_turn_identity"] or not row["single_turn_start"]):
                break
            if case == "completed" and (row["status"] != "completed"
                    or not row["final_seen"] or row["usage"] != "positive"
                    or row["http_requests"] != 1 or row["errors"]):
                break
        print(json.dumps({"schema": SCHEMA, "results": rows}, sort_keys=True))
    except (ProbeFailure, OSError):
        print(json.dumps({"schema": SCHEMA, "status": "binary_unverified"}))


if __name__ == "__main__":
    main()
