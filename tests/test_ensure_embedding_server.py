from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hymem import api


def _free_port() -> int:
    """Reserve and release an ephemeral port, returning its number."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib naming
        status = 200 if self.path == "/health" else 404
        self.send_response(status)
        self.end_headers()

    def log_message(self, *args):  # silence per-request logging
        pass


def _serve_health(port: int) -> HTTPServer:
    """Start a localhost /health server in a daemon thread; caller shuts it down."""
    httpd = HTTPServer(("127.0.0.1", port), _HealthHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_skip_when_not_configured(monkeypatch):
    monkeypatch.delenv("HYMEM_EMBEDDING_BASE_URL", raising=False)
    assert api._ensure_embedding_server() is True


def test_skip_remote_provider(monkeypatch):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "https://api.deepseek.com")
    # A remote host must never trigger a subprocess.
    monkeypatch.setattr(
        api.subprocess, "Popen",
        lambda *a, **k: pytest.fail("must not restart a remote server"),
    )
    assert api._ensure_embedding_server() is True


def test_fast_path_when_already_healthy(monkeypatch):
    port = _free_port()
    httpd = _serve_health(port)
    try:
        monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", f"http://127.0.0.1:{port}")
        monkeypatch.setattr(
            api.subprocess, "Popen",
            lambda *a, **k: pytest.fail("healthy server must not be restarted"),
        )
        assert api._ensure_embedding_server() is True
    finally:
        httpd.shutdown()


def test_health_probed_at_origin_not_v1_path(monkeypatch):
    # base_url carries the OpenAI-style /v1 path for /v1/embeddings, but /health
    # lives at the server root. The probe must hit /health, not /v1/health —
    # otherwise a healthy server reads as down and gets spuriously restarted.
    port = _free_port()
    httpd = _serve_health(port)  # only answers 200 at exactly /health
    try:
        monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setattr(
            api.subprocess, "Popen",
            lambda *a, **k: pytest.fail("healthy server must not be restarted"),
        )
        assert api._ensure_embedding_server() is True
    finally:
        httpd.shutdown()


def test_local_down_without_command_returns_false(monkeypatch):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", f"http://127.0.0.1:{_free_port()}")
    monkeypatch.delenv("HYMEM_EMBEDDING_SERVER_CMD", raising=False)
    assert api._ensure_embedding_server() is False


def test_restart_local_when_down(monkeypatch):
    port = _free_port()
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("HYMEM_EMBEDDING_SERVER_CMD", "noop-launcher --serve")

    servers: list[HTTPServer] = []

    def fake_popen(cmd, *args, **kwargs):
        # Stand in for the real launcher: the operator's command is parsed and
        # would have started the server — here we bring up the health endpoint.
        assert cmd == ["noop-launcher", "--serve"]
        servers.append(_serve_health(port))
        return object()

    monkeypatch.setattr(api.subprocess, "Popen", fake_popen)
    try:
        assert api._ensure_embedding_server(timeout=10.0) is True
        assert len(servers) == 1  # restarted exactly once
    finally:
        for s in servers:
            s.shutdown()


def test_restart_fails_when_command_unparseable(monkeypatch):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", f"http://127.0.0.1:{_free_port()}")
    # An unbalanced quote makes shlex.split raise ValueError -> graceful False.
    monkeypatch.setenv("HYMEM_EMBEDDING_SERVER_CMD", 'serve --flag "unterminated')
    assert api._ensure_embedding_server(timeout=2.0) is False
