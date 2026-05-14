"""Contract tests: drive the *real* honcho-ai SDK against a live HyMem server.

`test_honcho_server.py` checks raw endpoint JSON. These tests check the thing
that actually matters in production: can the pinned honcho-ai SDK *parse* every
response without a validation error? This is what catches Pydantic shape
mismatches before Hermes does.

The honcho-ai SDK is sync and only speaks real HTTP, so the app runs in a
uvicorn server on an ephemeral port for the duration of each test.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("honcho")
uvicorn = pytest.importorskip("uvicorn")

from honcho import Honcho

import hymem.honcho.app as hsrv
from tests.conftest import make_routed_llm

WORKSPACE = "hermes"


class _ThreadedServer(uvicorn.Server):
    """uvicorn server that runs in a daemon thread without signal handlers."""

    def install_signal_handlers(self) -> None:  # pragma: no cover - thread context
        pass


@pytest.fixture
def honcho(hy_with_embed):
    """A real honcho-ai SDK client wired to a live in-process HyMem server."""
    hsrv.set_hy(hy_with_embed)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(hsrv.app, host="127.0.0.1", port=port, log_level="warning")
    server = _ThreadedServer(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn test server failed to start"

    client = Honcho(
        api_key="test-key",
        base_url=f"http://127.0.0.1:{port}",
        workspace_id=WORKSPACE,
    )
    yield client

    server.should_exit = True
    thread.join(timeout=5.0)


def test_peer_get_or_create(honcho):
    peer = honcho.peer("user-1", metadata={"name": "Alice"})
    assert peer.id == "user-1"


def test_session_get_or_create(honcho):
    session = honcho.session("sess-1", metadata={"topic": "demo"})
    assert session.id == "sess-1"


def test_add_messages_returns_parseable_messages(honcho):
    session = honcho.session("sess-msg")
    user = honcho.peer("user-1")
    agent = honcho.peer("agent-main")
    messages = session.add_messages([
        user.message("we use uv and system python for local dev"),
        agent.message("noted — no docker for the dev environment"),
    ])
    assert len(messages) == 2
    assert all(m.session_id == "sess-msg" for m in messages)
    assert all(m.token_count >= 1 for m in messages)


def test_list_messages_paginates(honcho):
    session = honcho.session("sess-list")
    peer = honcho.peer("user-1")
    session.add_messages([peer.message(f"message number {i}") for i in range(5)])
    page = session.messages(page=1, size=2)
    items = list(page)
    assert len(items) >= 1


def test_session_search_parses(honcho, hy_with_embed):
    session = honcho.session("sess-search")
    peer = honcho.peer("user-1")
    session.add_messages([
        peer.message("No, we use uv and system Python. Don't suggest Docker."),
    ])
    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
        {"subject": "local_dev", "predicate": "uses", "object": "Docker", "polarity": -1},
    ]
    hy_with_embed.set_llm(make_routed_llm(triples, []))
    hy_with_embed.dream()

    results = session.search("should we use docker?")
    # The SDK parses each result into a Message — success is "no exception".
    assert isinstance(results, list)


def test_session_context_parses(honcho):
    session = honcho.session("sess-ctx")
    peer = honcho.peer("user-1")
    session.add_messages([
        peer.message("first message in the session"),
        peer.message("second message in the session"),
    ])
    ctx = session.context()
    assert ctx is not None


def test_session_add_peers(honcho):
    session = honcho.session("sess-peers")
    session.add_peers(["user-alice", "agent-bob"])


def test_peer_chat_parses(honcho):
    peer = honcho.peer("user-1")
    answer = peer.chat("what tooling do I prefer?")
    # chat returns `str | None`; either is a valid parse.
    assert answer is None or isinstance(answer, str)


def test_peer_card_parses(honcho, hy_with_embed):
    hy_with_embed.config.user_md_path.write_text(
        "# Behavioral Profile\n\n- prefers uv\n", encoding="utf-8"
    )
    peer = honcho.peer("user-1")
    card = peer.get_card()
    assert card is None or isinstance(card, list)
