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
    session = honcho.session("sess-chat")
    session.add_peers([peer])
    session.add_messages([peer.message("preferred tooling is uv")])
    answer = peer.chat("what tooling do I prefer?")
    assert isinstance(answer, str) and answer
    assert "uv" in answer.lower()


def test_peer_card_parses(honcho, hy_with_embed):
    peer = honcho.peer("user-1")
    session = honcho.session("sess-card")
    session.add_peers([peer])
    session.add_messages([peer.message("peer-owned card sentinel: uv")])
    hy_with_embed.config.user_md_path.write_text(
        "GLOBAL PROFILE SECRET", encoding="utf-8"
    )
    card = peer.get_card()
    assert isinstance(card, list) and card
    assert "peer-owned card sentinel: uv" in card[0]
    assert "GLOBAL PROFILE SECRET" not in card[0]


def test_peer_context_representation_reaches_sdk(honcho, hy_with_embed):
    """Regression: the peer context endpoint returned only
    `peer_representation`, but the SDK's PeerContextResponse declares the field
    as `representation` with no alias — Pydantic silently dropped the value, so
    SDK consumers (e.g. the Hermes harness prefetch path) saw an empty
    representation on this route every time. The route now sends both names;
    the parsed model must carry only workspace/peer-authorized evidence.
    """
    observer = honcho.peer("agent-main")
    target = honcho.peer("user-1")
    session = honcho.session("sess-peer-context")
    session.add_peers([observer, target])
    session.add_messages([target.message("SCOPED CONTEXT SENTINEL: uses uv")])
    hy_with_embed.config.user_md_path.write_text(
        "GLOBAL USER SECRET", encoding="utf-8"
    )
    hy_with_embed.config.memory_md_path.write_text(
        "GLOBAL MEMORY SECRET", encoding="utf-8"
    )

    ctx = observer.context(target=target)
    assert ctx.representation, "SDK dropped the representation field"
    assert "SCOPED CONTEXT SENTINEL" in ctx.representation
    assert "GLOBAL USER SECRET" not in ctx.representation
    assert "GLOBAL MEMORY SECRET" not in ctx.representation


def test_sdk_peer_and_session_representations_are_directional(honcho):
    observer_a = honcho.peer("observer-a")
    observer_b = honcho.peer("observer-b")
    target = honcho.peer("target")
    session_a = honcho.session("direction-a")
    session_b = honcho.session("direction-b")
    session_a.add_peers([observer_a, target])
    session_b.add_peers([observer_b, target])
    session_a.add_messages([
        target.message("shared-a sentinel; favorite editor is helix"),
    ])
    session_b.add_messages([target.message("private-b sentinel")])

    a = observer_a.representation(target=target)
    b = observer_b.representation(target=target)
    own = target.representation()
    local = session_a.representation(observer_a, target=target)
    directional_card = observer_a.get_card(target=target)
    untargeted_context = session_a.context(tokens=10_000)
    directional_context = session_a.context(
        tokens=10_000,
        peer_target=target.id,
        peer_perspective=observer_a.id,
    )
    local_omniscient_context = session_a.context(
        tokens=10_000,
        peer_target=target.id,
        limit_to_session=True,
    )

    assert isinstance(a, str) and "shared-a sentinel" in a
    assert "private-b sentinel" not in a
    assert isinstance(b, str) and "private-b sentinel" in b
    assert "shared-a sentinel" not in b
    assert "shared-a sentinel" in own and "private-b sentinel" in own
    assert "shared-a sentinel" in local and "private-b sentinel" not in local
    assert directional_card and "shared-a sentinel" in directional_card[0]
    assert "private-b sentinel" not in directional_card[0]
    assert untargeted_context.peer_representation is None
    assert untargeted_context.peer_card is None
    assert directional_context.peer_representation
    assert "shared-a sentinel" in directional_context.peer_representation
    assert "private-b sentinel" not in directional_context.peer_representation
    assert directional_context.peer_card
    assert "shared-a sentinel" in directional_context.peer_card[0]
    assert "private-b sentinel" not in directional_context.peer_card[0]
    assert local_omniscient_context.peer_representation
    assert "shared-a sentinel" in local_omniscient_context.peer_representation
    assert "private-b sentinel" not in local_omniscient_context.peer_representation
    assert local_omniscient_context.peer_card
    assert "shared-a sentinel" in local_omniscient_context.peer_card[0]
    assert "private-b sentinel" in local_omniscient_context.peer_card[0]


def test_sdk_context_card_is_independent_of_representation_filters(honcho):
    observer = honcho.peer("card-observer")
    target = honcho.peer("card-target")
    current = honcho.session("card-current")
    other = honcho.session("card-other")
    current.add_peers([observer, target])
    other.add_peers([observer, target])
    current.add_messages([
        target.message(f"current card item {index}") for index in range(30)
    ])
    other.add_messages([target.message("outside-session card sentinel")])

    card = observer.get_card(target=target)
    context = current.context(
        tokens=100_000,
        peer_target=target.id,
        peer_perspective=observer.id,
        limit_to_session=True,
        search_query="current card item 29",
        search_top_k=1,
        max_conclusions=1,
    )
    peer_context = observer.context(
        target=target,
        search_query="current card item 29",
        search_top_k=1,
        max_conclusions=1,
    )

    assert card and context.peer_card == card
    assert "outside-session card sentinel" in card[0]
    assert context.peer_representation
    assert "outside-session card sentinel" not in context.peer_representation
    assert context.peer_representation.count("message=msg_") == 1
    assert context.peer_representation != card[0]
    assert peer_context.peer_card == card
    assert peer_context.representation
    assert peer_context.representation.count("message=msg_") == 1
    assert peer_context.representation != card[0]


def test_sdk_chat_stream_consumes_sse_for_directional_session(honcho):
    observer = honcho.peer("observer-stream")
    target = honcho.peer("target-stream")
    session = honcho.session("stream-session")
    session.add_peers([observer, target])
    session.add_messages([target.message("favorite editor is helix")])

    stream = observer.chat_stream(
        "favorite editor?",
        target=target,
        session=session,
        reasoning_level="minimal",
    )
    chunks = list(stream)
    assert chunks and all(isinstance(chunk, str) for chunk in chunks)
    content = "".join(chunks)
    assert "helix" in content.lower()
    assert stream.is_complete
    assert stream.get_final_response() == {"content": content}


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
    peer_ctx = user.context()
    assert peer_ctx.peer_id == "user-1"
    assert isinstance(user.search("docker"), list)
    assert isinstance(user.representation(), str)
    assert isinstance(session.representation(user), str)
    chat = user.chat("should we use docker for dev?")
    assert isinstance(chat, str) and chat
    stream = user.chat_stream(
        "should we use docker for dev?", session=session,
        reasoning_level="minimal",
    )
    assert "".join(stream)
