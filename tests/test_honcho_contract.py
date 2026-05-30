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


def _seed_docker_graph(honcho, hy_with_embed):
    """Log a turn and dream a 2-edge graph the docker query reliably retrieves."""
    session = honcho.session("sess-seed")
    user = honcho.peer("user-1")
    session.add_messages([
        user.message("No, we use uv and system Python. Don't suggest Docker."),
    ])
    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
        {"subject": "local_dev", "predicate": "uses", "object": "Docker", "polarity": -1},
    ]
    hy_with_embed.set_llm(make_routed_llm(triples, []))
    hy_with_embed.dream()


def test_peer_chat_returns_answer_when_graph_has_facts(honcho, hy_with_embed):
    """Regression: peer.chat() reads the answer from `content`. HyMem used to
    return only `response`, so the SDK returned None and honcho_reasoning came
    back empty. Through the real SDK we now get a non-None answer.
    """
    _seed_docker_graph(honcho, hy_with_embed)
    answer = honcho.peer("user-1").chat("should we use docker for dev?")
    assert isinstance(answer, str) and answer
    assert "docker" in answer.lower()


def test_peer_search_parses_and_returns_facts(honcho, hy_with_embed):
    """Regression: peer.search() POSTs to .../peers/{id}/search, which HyMem
    didn't implement — the call 404'd and honcho_search came back empty. The
    SDK must now parse a non-empty list of Messages.
    """
    _seed_docker_graph(honcho, hy_with_embed)
    results = honcho.peer("user-1").search("should we use docker for dev?")
    assert isinstance(results, list)
    assert results, "peer.search must surface the dreamed graph fact"
    assert any("docker" in m.content.lower() for m in results)


def test_all_supported_sdk_methods_round_trip(honcho, hy_with_embed):
    """Drive every Honcho SDK call HyMem backs end-to-end through the real SDK.

    `test_honcho_server.py::test_every_supported_sdk_route_is_registered`
    proves the routes *exist*; this proves the SDK can *call them and parse the
    result* without raising — the integration `test_peer_chat_parses` was too
    lenient to catch (it accepted None).
    """
    _seed_docker_graph(honcho, hy_with_embed)

    # workspace / peer / session get-or-create
    user = honcho.peer("user-1", metadata={"name": "Alice"})
    agent = honcho.peer("agent-main")
    session = honcho.session("sess-roundtrip", metadata={"topic": "demo"})
    assert user.id == "user-1" and session.id == "sess-roundtrip"

    # messages: add + paginated list
    msgs = session.add_messages([
        user.message("we use uv for local dev"),
        agent.message("understood"),
    ])
    assert len(msgs) == 2
    assert isinstance(list(session.messages(page=1, size=10)), list)

    # session peers + per-peer config
    session.add_peers(["user-1", "agent-main"])
    assert session.get_peer_configuration("user-1") is not None

    # session context + search
    assert session.context() is not None
    assert isinstance(session.search("docker"), list)

    # peer card / context / search / chat
    assert (user.get_card() is None) or isinstance(user.get_card(), list)
    assert isinstance(user.search("docker"), list)
    chat = user.chat("should we use docker for dev?")
    assert chat is None or isinstance(chat, str)
