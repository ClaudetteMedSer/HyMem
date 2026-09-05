from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import hymem.honcho.app as hsrv
import hymem.server as mcp_server
from hymem.honcho.adapters import infer_role
from hymem.query.augment import GraphFact, MessageHit
from hymem.query.fusion import (
    FusedEvidence,
    RetrievalProvenance,
    SourceOccurrence,
    estimate_tokens,
)
from hymem.query.graph_state import GraphEvidenceCitation
from tests.conftest import make_routed_llm, seed_edge


@pytest.fixture
def client(hy_with_embed):
    hsrv.set_hy(hy_with_embed)
    # Stop any scheduler leaked from a prior test before TestClient triggers
    # lifespan startup (which creates a new one).
    if hsrv._scheduler is not None:
        hsrv._scheduler.stop()
        hsrv.set_scheduler(None)
    with TestClient(hsrv.app) as c:
        for peer_id in ("user-1", "agent-main"):
            assert c.post(
                "/v3/workspaces/hermes/peers", json={"id": peer_id}
            ).status_code == 201
        yield c


def _open_external_session(hy, session_id: str) -> None:
    hy.open_session(session_id, source_workspace_id="hermes")


def _log_external(
    hy,
    session_id: str,
    role: str,
    content: str,
    *,
    peer_id: str | None = None,
) -> int:
    exact_peer = peer_id or ("agent-main" if role == "assistant" else "user-alice")
    return hy.log_message(
        session_id,
        role,
        content,
        source_peer_id=exact_peer,
        source_workspace_id="hermes",
    )


def _citation(**changes) -> GraphEvidenceCitation:
    citation = GraphEvidenceCitation(
        evidence_id=11,
        evidence_kind="triple_extraction",
        source_role="user",
        source_session_id="real-session",
        source_message_id=101,
        source_event_at="2024-02-01T09:00:00.000Z",
        source_created_at="2024-02-01T09:00:00Z",
        temporal_scope=None,
        recorded_at="2024-02-02T00:00:00Z",
        coverage_chunk_id="coverage:real-session:101",
        coverage_version="v1",
        extraction_chunk_id="extract-1",
        currently_authoritative=True,
        authoritative_at_recorded_time=True,
        provenance_status="canonical",
    )
    return replace(citation, **changes)


def test_graph_fact_message_never_turns_a_role_into_a_peer_identity():
    citations = [
        _citation(),
        _citation(
            evidence_id=12,
            source_role="assistant",
            source_session_id="second-session",
            source_message_id=202,
            source_event_at="2024-03-01T10:00:00.000Z",
            source_created_at="2024-03-01T10:00:00Z",
        ),
    ]
    fact = GraphFact(
        "app", "uses", "redis", 0.9, 3, 0,
        edge_id=77,
        valid_at="2024-02-01T09:00:00.000Z",
        citations=citations,
    )
    shaped = hsrv._graph_fact_message(fact, "workspace")
    assert shaped is None
    assert hsrv._graph_fact_message(
        GraphFact("manual", "uses", "sqlite", 1.0, 1, 0, edge_id=78),
        "workspace",
    ) is None


def test_graph_fact_message_exposes_only_individually_validated_citations(monkeypatch):
    valid = _citation(
        evidence_id=1, source_peer_id="alice", source_workspace_id="workspace",
    )
    forged = _citation(
        evidence_id=2, source_peer_id="alice", source_workspace_id="workspace",
        source_message_id=202,
    )
    proof = SimpleNamespace(
        message_id=101, content="exact authored content", source_peer_id="alice",
        session_id="real-session", source_created_at="2024-02-01T09:00:00Z",
    )
    monkeypatch.setattr(
        hsrv,
        "_validated_message_occurrence",
        lambda **kwargs: proof if kwargs["message_id"] == 101 else None,
    )
    shaped = hsrv._graph_fact_message(
        GraphFact(
            "app", "uses", "redis", .9, 2, 0, edge_id=7,
            citations=[valid, forged],
        ),
        "workspace",
        peer_id="alice",
    )
    assert shaped is not None
    assert shaped["content"] == "exact authored content"
    assert [item["evidence_id"] for item in shaped["metadata"]["citations"]] == [1]


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "backend": "hymem"}


def test_dream_status_endpoint(client):
    r = client.get("/dream-status")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) >= {
        "pending_chunks",
        "total_chunks",
        "prompt_version",
        "in_progress",
        "last_run",
    }
    assert isinstance(body["pending_chunks"], int)
    assert isinstance(body["total_chunks"], int)
    assert isinstance(body["in_progress"], bool)


def test_add_messages_logs_and_returns_message_objects(client, hy_with_embed):
    r = client.post(
        "/v3/workspaces/hermes/sessions/sess-1/messages",
        json={
            "messages": [
                {"content": "hello there", "peer_id": "user-123"},
                {"content": "hi back", "peer_id": "agent-main"},
            ]
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert len(body) == 2
    for msg in body:
        assert msg["session_id"] == "sess-1"
        assert msg["workspace_id"] == "hermes"
        assert msg["id"].startswith("msg_")
        assert "created_at" in msg
        assert "token_count" in msg

    rows = hy_with_embed.conn.execute(
        "SELECT role, content FROM messages WHERE session_id='sess-1' ORDER BY id"
    ).fetchall()
    assert [(r["role"], r["content"]) for r in rows] == [
        ("user", "hello there"),
        ("assistant", "hi back"),
    ]


def test_search_returns_full_exact_message_and_does_not_apply_token_excerpts(client):
    content = ("long authored prefix " * 100) + "rare-search-tail"
    created = client.post(
        "/v3/workspaces/hermes/sessions/exact-search/messages",
        json={"messages": [{"content": content, "peer_id": "user-1"}]},
    )
    assert created.status_code == 201
    response = client.post(
        "/v3/workspaces/hermes/sessions/exact-search/search",
        # ``tokens`` is a future/unknown SDK extra. Search is item-limited and
        # must never change exact Message identity/content based on it.
        json={"query": "rare-search-tail", "limit": 1, "tokens": 1},
    )
    assert response.status_code == 200
    assert response.json()[0]["id"] == created.json()[0]["id"]
    assert response.json()[0]["content"] == content


@pytest.mark.parametrize("payload", [
    {"query": "", "limit": 1},
    {"query": "needle", "limit": 101},
])
def test_search_validates_query_and_limit(client, payload):
    response = client.post(
        "/v3/workspaces/hermes/sessions/validation/search", json=payload
    )
    # Session ownership is created before request body use only on message
    # routes, so make it explicit then re-run the actual validation probe.
    if response.status_code == 404:
        client.post(
            "/v3/workspaces/hermes/sessions/validation/messages",
            json={"messages": [{"content": "needle", "peer_id": "user-1"}]},
        )
        response = client.post(
            "/v3/workspaces/hermes/sessions/validation/search", json=payload
        )
    assert response.status_code == 422


def test_search_rejects_nonempty_filters(client):
    client.post(
        "/v3/workspaces/hermes/sessions/filter-search/messages",
        json={"messages": [{"content": "needle", "peer_id": "user-1"}]},
    )
    response = client.post(
        "/v3/workspaces/hermes/sessions/filter-search/search",
        json={"query": "needle", "filters": {"peer_id": "user-1"}},
    )
    assert response.status_code == 422


def test_add_messages_persists_supplied_created_at(client, hy_with_embed):
    # A caller-supplied event time is preserved by instant and returned in the
    # same canonical UTC-millisecond form used by chronological retrieval.
    r = client.post(
        "/v3/workspaces/hermes/sessions/sess-ts/messages",
        json={
            "messages": [
                {"content": "earlier", "peer_id": "user-1", "created_at": "2024-02-15T10:00:00.0009+01:00"},
                {"content": "later", "peer_id": "user-1", "created_at": "2024-03-01T09:00:00.9995Z"},
                {"content": "no timestamp", "peer_id": "user-1"},
            ]
        },
    )
    assert r.status_code == 201

    rows = hy_with_embed.conn.execute(
        "SELECT content, created_at FROM messages WHERE session_id='sess-ts' ORDER BY id"
    ).fetchall()
    by_content = {r["content"]: r["created_at"] for r in rows}
    assert by_content["earlier"] == "2024-02-15T09:00:00.000Z"
    assert by_content["later"] == "2024-03-01T09:00:00.999Z"
    # Omitted created_at falls back to the DB default, not a blank string.
    assert by_content["no timestamp"]
    posted = {item["content"]: item["created_at"] for item in r.json()}
    assert posted == by_content
    listed_response = client.post(
        "/v3/workspaces/hermes/sessions/sess-ts/messages/list",
        json={"page": 1, "size": 10},
    )
    assert listed_response.status_code == 200
    listed = {
        item["content"]: item["created_at"]
        for item in listed_response.json()["items"]
    }
    assert listed == by_content


def test_add_messages_rejects_future_time_and_rolls_back_batch(
    client, hy_with_embed
):
    response = client.post(
        "/v3/workspaces/hermes/sessions/future-batch/messages",
        json={
            "messages": [
                {
                    "content": "valid first",
                    "peer_id": "user-alice",
                    "created_at": "2024-01-01T00:00:00Z",
                },
                {
                    "content": "poison second",
                    "peer_id": "user-alice",
                    "created_at": "2100-01-01T00:00:00Z",
                },
            ]
        },
    )
    assert response.status_code == 422
    assert "message source" in response.json()["detail"]
    assert hy_with_embed.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='future-batch'"
    ).fetchone()[0] == 0
    assert hy_with_embed.conn.execute(
        "SELECT COUNT(*) FROM message_retention_coverage coverage "
        "JOIN messages message ON message.id=coverage.message_id "
        "WHERE message.session_id='future-batch'"
    ).fetchone()[0] == 0


def test_upload_rejects_future_time_without_writing_source(client, hy_with_embed):
    response = client.post(
        "/v3/workspaces/hermes/sessions/future-upload/messages/upload",
        data={"peer_id": "user-alice", "created_at": "2100-01-01T00:00:00Z"},
        files={"file": ("memory.txt", b"future poison", "text/plain")},
    )
    assert response.status_code == 422
    assert hy_with_embed.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='future-upload'"
    ).fetchone()[0] == 0


def test_search_returns_empty_for_unknown_query(client):
    assert client.post(
        "/v3/workspaces/hermes/sessions", json={"id": "s"}
    ).status_code == 201
    r = client.post(
        "/v3/workspaces/hermes/sessions/s/search",
        json={"query": "totally unknown topic xyz"},
    )
    assert r.status_code == 200
    assert r.json() == []


def test_uncited_fact_stays_native_but_is_omitted_from_message_search(
    client, hy_with_embed
):
    seed_edge(hy_with_embed.conn, "manual_app", "uses", "manual_db")
    native = hy_with_embed.augment("manual app uses manual db")
    manual = next(
        fact for fact in native.graph_facts
        if fact.subject == "manual_app" and fact.object == "manual_db"
    )
    assert manual.citations == []

    assert client.post(
        "/v3/workspaces/hermes/sessions", json={"id": "any"}
    ).status_code == 201
    response = client.post(
        "/v3/workspaces/hermes/sessions/any/search",
        json={"query": "manual app uses manual db"},
    )
    assert response.status_code == 200
    assert not any(
        item["metadata"].get("type") == "graph_fact"
        for item in response.json()
    )


def test_search_does_not_fabricate_graph_fact_peer_identity(client, hy_with_embed):
    sid = "s-search"
    _open_external_session(hy_with_embed, sid)
    _log_external(hy_with_embed, sid, "assistant", "We could try Docker for the local dev environment.")
    _log_external(
        hy_with_embed,
        sid, "user",
        "No, we use uv and system Python for local dev. Don't suggest Docker.",
    )
    hy_with_embed.close_session(sid)
    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
        {"subject": "local_dev", "predicate": "uses", "object": "Docker", "polarity": -1},
    ]
    hy_with_embed.set_llm(make_routed_llm(triples, []))
    hy_with_embed.dream()

    r = client.post(
        "/v3/workspaces/hermes/sessions/s-search/search",
        json={"query": "should we use docker for dev?"},
    )
    assert r.status_code == 200
    body = r.json()
    kg = [item for item in body if item["metadata"].get("type") == "graph_fact"]
    assert kg
    assert all(item["peer_id"] in {"user-alice", "agent-main"} for item in kg)
    assert not any(item["peer_id"] == "hymem-kg" for item in body)
    assert any("docker" in item["content"].lower() for item in body)


def test_peer_search_returns_empty_for_unknown_query(client):
    # peer.search() POSTs to .../peers/{id}/search — the route must exist (it
    # didn't, which made honcho_search come back empty) and return [] cleanly.
    r = client.post(
        "/v3/workspaces/hermes/peers/user-1/search",
        json={"query": "totally unknown topic xyz"},
    )
    assert r.status_code == 200
    assert r.json() == []


def test_peer_search_does_not_fabricate_graph_fact_peer_identity(
    client, hy_with_embed
):
    sid = "s-peersearch"
    _open_external_session(hy_with_embed, sid)
    _log_external(hy_with_embed, sid, "assistant", "We could try Docker for the local dev environment.")
    _log_external(
        hy_with_embed,
        sid, "user",
        "No, we use uv and system Python for local dev. Don't suggest Docker.",
    )
    hy_with_embed.close_session(sid)
    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
        {"subject": "local_dev", "predicate": "uses", "object": "Docker", "polarity": -1},
    ]
    hy_with_embed.set_llm(make_routed_llm(triples, []))
    hy_with_embed.dream()

    r = client.post(
        "/v3/workspaces/hermes/peers/agent-main/search",
        json={"query": "should we use docker for dev?"},
    )
    assert r.status_code == 200
    body = r.json()
    kg = [item for item in body if item["metadata"].get("type") == "graph_fact"]
    assert kg == []
    assert not any(item["peer_id"] == "hymem-kg" for item in body)
    assert any("docker" in item["content"].lower() for item in body)


def test_context_returns_summary_messages_peers(client, hy_with_embed):
    sid = "s-ctx"
    _open_external_session(hy_with_embed, sid)
    _log_external(hy_with_embed, sid, "user", "first message")
    _log_external(hy_with_embed, sid, "assistant", "second message")
    hy_with_embed.close_session(sid)

    r = client.get(f"/v3/workspaces/hermes/sessions/{sid}/context")
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body
    assert "messages" in body
    assert "peers" in body
    assert len(body["messages"]) == 2
    assert body["messages"][0]["content"] == "first message"
    assert body["messages"][1]["content"] == "second message"


def test_add_peers_persists_role_mapping(client, hy_with_embed):
    r = client.post(
        "/v3/workspaces/hermes/sessions/sess/peers",
        json={
            "peers": [
                {"id": "user-alice"},
                {"id": "agent-bob", "metadata": {"version": "1"}},
            ]
        },
    )
    assert r.status_code == 201
    rows = hy_with_embed.conn.execute(
        "SELECT id, role FROM peers WHERE workspace_id='hermes' ORDER BY id"
    ).fetchall()
    by_id = {r["id"]: r["role"] for r in rows}
    assert by_id["user-alice"] == "user"
    assert by_id["agent-bob"] == "assistant"


def test_peer_card_does_not_return_unowned_user_md_content(client, hy_with_embed):
    hy_with_embed.config.user_md_path.write_text(
        "# Behavioral Profile\n\n- prefers uv\n", encoding="utf-8"
    )
    r = client.get("/v3/workspaces/hermes/peers/user-1/card")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "user-1"
    assert body["content"] == ""


def _seed_root_digest(hy):
    """Insert a deliberately unowned native digest isolation sentinel."""
    hy.conn.execute(
        "INSERT INTO aggregation_nodes "
        "(id, title, summary, member_episode_ids, session_ids, "
        " n_members, n_sessions, level, is_root) "
        "VALUES ('root-test', 'User digest', 'Works on HyMem and Hermes.', "
        " '[]', '[]', 3, 3, 1, 1)"
    )
    hy.conn.commit()


def test_peer_card_omits_unowned_digest_and_profile(client, hy_with_embed):
    hy_with_embed.config.user_md_path.write_text(
        "# Behavioral Profile\n\n- prefers uv\n", encoding="utf-8"
    )
    _seed_root_digest(hy_with_embed)
    r = client.get("/v3/workspaces/hermes/peers/user-1/card")
    assert r.status_code == 200
    content = r.json()["content"]
    assert content == ""


def test_peer_context_representation_omits_unowned_digest(client, hy_with_embed):
    hy_with_embed.config.user_md_path.write_text(
        "# Behavioral Profile\n\n- prefers uv\n", encoding="utf-8"
    )
    _seed_root_digest(hy_with_embed)
    r = client.get("/v3/workspaces/hermes/peers/user-1/context")
    assert r.status_code == 200
    rep = r.json()["peer_representation"]
    assert rep == ""
    # honcho-ai's PeerContextResponse expects `representation` (it has no
    # alias for `peer_representation`), so the route sends both names.
    assert r.json()["representation"] == rep


def test_untargeted_session_context_has_no_cross_session_representation(
    client, hy_with_embed
):
    hy_with_embed.config.user_md_path.write_text(
        "# Behavioral Profile\n\n- prefers uv\n", encoding="utf-8"
    )
    _seed_root_digest(hy_with_embed)
    sid = "s-ctx-digest"
    _open_external_session(hy_with_embed, sid)
    _log_external(hy_with_embed, sid, "user", "hello")
    hy_with_embed.close_session(sid)

    r = client.get(f"/v3/workspaces/hermes/sessions/{sid}/context")
    assert r.status_code == 200
    rep = r.json()["peer_representation"]
    assert rep == ""


def test_peer_representation_never_uses_process_global_user_md(client, hy_with_embed):
    hy_with_embed.config.user_md_path.write_text(
        "# Behavioral Profile\n\n- prefers uv\n", encoding="utf-8"
    )
    r = client.get("/v3/workspaces/hermes/peers/user-1/context")
    assert r.status_code == 200
    rep = r.json()["peer_representation"]
    assert rep == ""


def test_peer_chat_returns_response_for_query(client, hy_with_embed):
    r = client.post(
        "/v3/workspaces/hermes/peers/user-1/chat",
        json={"queries": ["what tooling do I prefer?"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert "response" in body
    assert "queries" in body
    assert body["queries"] == ["what tooling do I prefer?"]


def test_role_inference_from_peer_id():
    assert infer_role("user-123") == "user"
    assert infer_role("agent-main") == "assistant"
    assert infer_role("hermes") == "assistant"
    assert infer_role("telegram-12345") == "user"
    assert infer_role("ai-bot") == "assistant"


def _install_scheduler(hy, cooldown: float):
    from hymem.dreaming.scheduler import DreamScheduler
    if hsrv._scheduler is not None:
        hsrv._scheduler.stop()
    sched = DreamScheduler(hy, cooldown=cooldown)
    sched.start()
    hsrv.set_scheduler(sched)
    return sched


def test_dream_cooldown_throttles_back_to_back_calls(client, hy_with_embed):
    sched = _install_scheduler(hy_with_embed, cooldown=60.0)

    payload = {
        "messages": [
            {"content": "we use uv and system python for local dev", "peer_id": "user-1"},
            {"content": "noted, no docker for the dev environment", "peer_id": "agent-main"},
        ]
    }
    r1 = client.post("/v3/workspaces/hermes/sessions/cool-1/messages", json=payload)
    r2 = client.post("/v3/workspaces/hermes/sessions/cool-1/messages", json=payload)
    assert r1.status_code == 201
    assert r2.status_code == 201

    # First kick runs immediately; second is gated by cooldown.
    assert sched.wait_for_cycle(1, timeout=5.0)

    count = hy_with_embed.conn.execute(
        "SELECT COUNT(*) FROM dream_runs"
    ).fetchone()[0]
    assert count == 1


def test_dream_cooldown_allows_after_window(client, hy_with_embed):
    sched = _install_scheduler(hy_with_embed, cooldown=0.0)

    payload = {
        "messages": [
            {"content": "we use uv and system python for local dev", "peer_id": "user-1"},
            {"content": "noted, no docker for the dev environment", "peer_id": "agent-main"},
        ]
    }
    r1 = client.post("/v3/workspaces/hermes/sessions/cool-2/messages", json=payload)
    r2 = client.post("/v3/workspaces/hermes/sessions/cool-2/messages", json=payload)
    assert r1.status_code == 201
    assert r2.status_code == 201

    assert sched.wait_for_cycle(2, timeout=5.0)

    count = hy_with_embed.conn.execute(
        "SELECT COUNT(*) FROM dream_runs"
    ).fetchone()[0]
    assert count == 2


def test_resolve_role_uses_peers_table_when_present(client, hy_with_embed):
    hy_with_embed.conn.execute(
        "INSERT INTO peers(id, workspace_id, role, metadata) "
        "VALUES ('ambiguous-id', 'hermes', 'assistant', '{}')"
    )
    assert hsrv._resolve_role("hermes", "ambiguous-id") == "assistant"
    # Falls back to inference for unknown ids.
    assert hsrv._resolve_role("hermes", "user-fresh") == "user"


# ── get-or-create lifecycle endpoints ────────────────────────────────────────


def test_create_workspace_echoes_id_and_metadata(client):
    r = client.post("/v3/workspaces", json={"id": "hermes", "metadata": {"k": "v"}})
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "hermes"
    assert body["metadata"] == {"k": "v"}
    assert "created_at" in body


def test_create_workspace_is_idempotent(client):
    r1 = client.post("/v3/workspaces", json={"id": "hermes"})
    r2 = client.post("/v3/workspaces", json={"id": "hermes"})
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["id"] == r2.json()["id"] == "hermes"


def test_get_workspace_returns_stateless_echo(client):
    r = client.get("/v3/workspaces/hermes")
    assert r.status_code == 200
    assert r.json()["id"] == "hermes"


def test_create_peer_inserts_row_and_is_idempotent(client, hy_with_embed):
    r1 = client.post(
        "/v3/workspaces/hermes/peers",
        json={"id": "user-42", "metadata": {"name": "Alice"}},
    )
    r2 = client.post("/v3/workspaces/hermes/peers", json={"id": "user-42"})
    assert r1.status_code == 201
    assert r2.status_code == 201
    rows = hy_with_embed.conn.execute(
        "SELECT id, workspace_id, role FROM peers WHERE id = 'user-42'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert rows[0]["workspace_id"] == "hermes"


def test_get_peer_round_trip_and_404(client):
    client.post("/v3/workspaces/hermes/peers", json={"id": "user-7", "metadata": {"n": 1}})
    r = client.get("/v3/workspaces/hermes/peers/user-7")
    assert r.status_code == 200
    assert r.json()["id"] == "user-7"
    assert r.json()["metadata"] == {"n": 1}

    miss = client.get("/v3/workspaces/hermes/peers/nobody")
    assert miss.status_code == 404


def test_create_session_opens_session_and_links_peers(client, hy_with_embed):
    r = client.post(
        "/v3/workspaces/hermes/sessions",
        json={
            "id": "sess-A",
            "metadata": {"topic": "demo"},
            "peer_names": {"user-1": {}, "agent-main": {}},
        },
    )
    assert r.status_code == 201
    assert r.json()["id"] == "sess-A"

    sess_row = hy_with_embed.conn.execute(
        "SELECT id FROM sessions WHERE id = 'sess-A'"
    ).fetchone()
    assert sess_row is not None

    peer_ids = {
        row["id"]
        for row in hy_with_embed.conn.execute(
            "SELECT id FROM peers WHERE workspace_id = 'hermes'"
        )
    }
    assert {"user-1", "agent-main"} <= peer_ids


def test_create_session_is_idempotent(client):
    r1 = client.post("/v3/workspaces/hermes/sessions", json={"id": "sess-B"})
    r2 = client.post("/v3/workspaces/hermes/sessions", json={"id": "sess-B"})
    assert r1.status_code == 201
    assert r2.status_code == 201


def test_get_session_round_trip_and_404(client, hy_with_embed):
    client.post("/v3/workspaces/hermes/sessions", json={"id": "sess-C"})
    r = client.get("/v3/workspaces/hermes/sessions/sess-C")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "sess-C"
    assert body["is_active"] is True

    miss = client.get("/v3/workspaces/hermes/sessions/missing")
    assert miss.status_code == 404


# ── messages list (pagination) ───────────────────────────────────────────────


def test_list_messages_paginates(client, hy_with_embed):
    sid = "sess-list"
    _open_external_session(hy_with_embed, sid)
    for i in range(5):
        _log_external(hy_with_embed, sid, "user", f"msg-{i}")

    r = client.post(
        f"/v3/workspaces/hermes/sessions/{sid}/messages/list",
        json={"page": 1, "size": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 5
    assert body["size"] == 2
    assert body["pages"] == 3
    assert len(body["items"]) == 2
    assert body["items"][0]["content"] == "msg-0"

    r2 = client.post(
        f"/v3/workspaces/hermes/sessions/{sid}/messages/list",
        json={"page": 3, "size": 2},
    )
    assert r2.json()["items"][0]["content"] == "msg-4"
    assert len(r2.json()["items"]) == 1


def test_list_messages_empty_session(client, hy_with_embed):
    _open_external_session(hy_with_embed, "empty-sid")
    r = client.post(
        "/v3/workspaces/hermes/sessions/empty-sid/messages/list",
        json={},
    )
    assert r.status_code == 200
    assert r.json() == {"items": [], "total": 0, "page": 1, "size": 50, "pages": 0}


def test_list_messages_rejects_zero_size(client, hy_with_embed):
    # size=0 would divide-by-zero in page-count math — must be a clean 422.
    _open_external_session(hy_with_embed, "zero-sid")
    r = client.post(
        "/v3/workspaces/hermes/sessions/zero-sid/messages/list",
        json={"size": 0},
    )
    assert r.status_code == 422


# ── why_retrieved surfaced as metadata.why ───────────────────────────────────


def _seed_dreamed_graph(hy_with_embed):
    sid = "s-why"
    _open_external_session(hy_with_embed, sid)
    _log_external(
        hy_with_embed,
        sid, "user", "We use fast_api for the backend service."
    )
    hy_with_embed.close_session(sid)
    triples = [
        {"subject": "backend", "predicate": "uses", "object": "fast_api", "polarity": 1},
    ]
    hy_with_embed.set_llm(make_routed_llm(triples, []))
    hy_with_embed.dream()


def test_registered_user_alice_is_never_collapsed_to_role_peer_id(
    client, hy_with_embed
):
    registered = client.post(
        "/v3/workspaces/hermes/peers", json={"id": "user-alice"}
    )
    assert registered.status_code == 201
    _seed_dreamed_graph(hy_with_embed)
    native = hy_with_embed.augment("what technologies does the backend use")
    fact = native.graph_facts[0]
    assert fact.citations[0].source_role == "user"
    assert fact.citations[0].source_peer_id == "user-alice"
    assert hsrv._graph_fact_message(fact, "hermes")["peer_id"] == "user-alice"

    result = client.post(
        "/v3/workspaces/hermes/sessions/s-why/search",
        json={"query": "what technologies does the backend use"},
    ).json()
    graph_messages = [
        item for item in result
        if item["metadata"].get("type") == "graph_fact"
    ]
    assert graph_messages
    assert {item["peer_id"] for item in graph_messages} == {"user-alice"}


def test_search_preserves_cited_graph_fact_exact_peer_id(client, hy_with_embed):
    _seed_dreamed_graph(hy_with_embed)
    native = hy_with_embed.augment("what technologies does the backend use")
    assert native.graph_facts
    assert native.graph_facts[0].why_retrieved
    assert native.graph_facts[0].citations
    r = client.post(
        "/v3/workspaces/hermes/sessions/s-why/search",
        json={"query": "what technologies does the backend use"},
    )
    assert r.status_code == 200
    body = r.json()
    kg_items = [m for m in body if m["metadata"].get("type") == "graph_fact"]
    assert kg_items
    assert {item["peer_id"] for item in kg_items} == {"user-alice"}


def test_peer_context_omits_graph_fact_without_exact_peer_id(client, hy_with_embed):
    _seed_dreamed_graph(hy_with_embed)
    r = client.get(
        "/v3/workspaces/hermes/peers/agent-main/context",
        params={"search_query": "what technologies does the backend use"},
    )
    assert r.status_code == 200
    body = r.json()
    kg_items = [
        m for m in body["messages"]
        if m["metadata"].get("type") == "graph_fact"
    ]
    assert kg_items == []


def test_peer_chat_returns_exact_peer_scoped_structured_facts(client, hy_with_embed):
    _seed_dreamed_graph(hy_with_embed)
    r = client.post(
        "/v3/workspaces/hermes/peers/user-alice/chat",
        json={"query": "what technologies does the backend use"},
    )
    assert r.status_code == 200
    body = r.json()
    # Prose keeps retrieval reason codes out but carries exact source labels.
    assert "fallback:" not in body["response"]
    assert "semantic_" not in body["response"]
    assert "[edge " in body["response"]
    assert "role=user, session=s-why, message=" in body["response"]
    # Scoped output carries only the exact citations, never global confidence
    # or evidence totals influenced by another workspace/peer.
    assert "facts" in body and body["facts"], "facts should accompany prose response"
    for fact in body["facts"]:
        assert {"subject", "predicate", "object", "edge_id", "citations"} <= fact.keys()
        assert "confidence" not in fact and "why" not in fact
        assert fact["edge_id"] is not None
        assert fact["citations"]
        assert {c["source_peer_id"] for c in fact["citations"]} == {"user-alice"}


def test_mcp_augment_renders_stable_edge_and_exact_sources(client, hy_with_embed):
    _seed_dreamed_graph(hy_with_embed)
    mcp_server.set_hy(hy_with_embed)
    rendered = mcp_server._do_augment("what technologies does the backend use")
    assert "[edge " in rendered
    assert "role=user, session=s-why, message=" in rendered
    assert "event=" in rendered
    assert "hymem-kg" not in rendered


def test_peer_chat_exposes_content_field_for_sdk(client, hy_with_embed):
    # The honcho-ai SDK reads peer.chat() answers from `content`
    # (data.get("content")); `response` is only a HyMem-native alias. If
    # `content` is missing or empty the SDK returns None and honcho_reasoning
    # silently comes back empty — the bug this guards against.
    _seed_dreamed_graph(hy_with_embed)
    r = client.post(
        "/v3/workspaces/hermes/peers/user-alice/chat",
        json={"query": "what technologies does the backend use"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "content" in body, "SDK reads `content`; it must be present"
    assert body["content"] == body["response"]
    assert body["content"], "content must be non-empty when the graph has facts"
    assert "fast_api" in body["content"].lower()


# ── /conflicts endpoint ──────────────────────────────────────────────────────


def test_workspace_conflicts_explicitly_rejects_unpartitioned_graph(client):
    r = client.get("/v3/workspaces/hermes/conflicts")
    assert r.status_code == 501
    assert "workspace-scoped" in r.json()["detail"]


def test_workspace_conflicts_never_relabels_global_competing_objects(
    client, hy_with_embed
):
    conn = hy_with_embed.conn
    # `runs_on` is functional; multiple runtimes for one service is a true conflict.
    seed_edge(conn, "service_a", "runs_on", "python3")
    seed_edge(conn, "service_a", "runs_on", "python2")

    r = client.get("/v3/workspaces/hermes/conflicts")
    assert r.status_code == 501
    assert "service_a" not in r.text


def test_workspace_conflicts_never_relabels_global_opposing_edges(
    client, hy_with_embed
):
    conn = hy_with_embed.conn
    seed_edge(conn, "team", "prefers", "docker")
    seed_edge(conn, "team", "rejects", "docker")

    r = client.get("/v3/workspaces/hermes/conflicts")
    assert r.status_code == 501
    assert "docker" not in r.text


# ── route-registration contract ──────────────────────────────────────────────


def test_every_supported_sdk_route_is_registered():
    """Each Honcho SDK route HyMem backs must be registered on the app with a
    matching HTTP method, so a future SDK call can't silently 404 the way
    peer.search() did (its empty result was the honcho_search bug).

    Paths and verbs are taken from the *pinned SDK's own route table* and verb
    usage, so this test also breaks if an SDK upgrade renames a path HyMem must
    serve — turning a would-be production 404 into a local failure.
    """
    routes = pytest.importorskip("honcho.http.routes")
    from starlette.routing import Match

    WS, PID, SID = "ws", "pid", "sid"
    # (HTTP verb the SDK uses, concrete path the SDK builds). Curated to the
    # subset HyMem implements — list/clone/summaries/workspace-search are
    # deliberately out of scope and intentionally absent.
    supported = [
        ("POST", routes.workspaces()),
        ("GET", routes.workspace(WS)),
        ("POST", routes.peers(WS)),
        ("GET", routes.peer(WS, PID)),
        ("POST", routes.peer_chat(WS, PID)),
        ("POST", routes.peer_search(WS, PID)),
        ("GET", routes.peer_card(WS, PID)),
        ("GET", routes.peer_context(WS, PID)),
        ("POST", routes.peer_representation(WS, PID)),
        ("POST", routes.sessions(WS)),
        ("GET", routes.session(WS, SID)),
        ("POST", routes.session_search(WS, SID)),
        ("GET", routes.session_context(WS, SID)),
        ("POST", routes.session_peers(WS, SID)),
        ("GET", routes.session_peer_config(WS, SID, PID)),
        ("POST", routes.messages(WS, SID)),
        ("POST", routes.messages_list(WS, SID)),
        ("POST", routes.messages_upload(WS, SID)),
    ]

    def _full_match(method: str, path: str) -> bool:
        scope = {"type": "http", "method": method, "path": path}
        return any(
            route.matches(scope)[0] == Match.FULL for route in hsrv.app.routes
        )

    missing = [(m, p) for m, p in supported if not _full_match(m, p)]
    assert not missing, f"SDK routes not registered (path+method): {missing}"


def test_context_does_not_expose_unowned_global_rules(client, hy_with_embed):
    hy_with_embed.add_rule("never suggest docker")
    client.post(
        "/v3/workspaces/hermes/sessions/s-rules/messages",
        json={"messages": [{"content": "hi", "peer_id": "user-1"}]},
    )
    r = client.get("/v3/workspaces/hermes/sessions/s-rules/context")
    assert r.status_code == 200
    summary = r.json()["summary"]
    assert summary is None


def test_context_summary_false_does_not_expose_unowned_rules(client, hy_with_embed):
    hy_with_embed.add_rule("always preserve the complete rule")
    client.post(
        "/v3/workspaces/hermes/sessions/rules-no-summary/messages",
        json={"messages": [{"content": "hello", "peer_id": "user-1"}]},
    )
    enough = client.get(
        "/v3/workspaces/hermes/sessions/rules-no-summary/context"
        "?summary=false&tokens=500"
    )
    assert enough.status_code == 200
    assert enough.json()["peer_representation"] == ""
    too_small = client.get(
        "/v3/workspaces/hermes/sessions/rules-no-summary/context"
        "?summary=false&tokens=1"
    )
    assert too_small.status_code == 200
    assert too_small.json()["peer_representation"] == ""


def test_context_budget_changes_output_skips_oversized_and_reads_beyond_20(
    client,
):
    compact = [f"compact history {index}" for index in range(35)]
    response = client.post(
        "/v3/workspaces/hermes/sessions/token-history/messages",
        json={"messages": [
            *({"content": content, "peer_id": "user-1"} for content in compact),
            {"content": "oversized " * 1000, "peer_id": "user-1"},
        ]},
    )
    assert response.status_code == 201
    wide = client.get(
        "/v3/workspaces/hermes/sessions/token-history/context"
        "?summary=false&tokens=5000"
    )
    tight = client.get(
        "/v3/workspaces/hermes/sessions/token-history/context"
        "?summary=false&tokens=250"
    )
    assert wide.status_code == tight.status_code == 200
    assert [item["content"] for item in wide.json()["messages"]] == compact
    assert len(tight.json()["messages"]) < len(wide.json()["messages"])
    for result, budget in ((wide.json(), 5000), (tight.json(), 250)):
        assert result["context_token_count"] <= budget
        assert all(item["content"] != "oversized " * 1000 for item in result["messages"])
        assert result["context_truncated"]


def test_context_counts_long_peer_name_and_actual_openai_wrapper(client):
    peer_id = "user-" + ("x" * 600)
    assert client.post(
        "/v3/workspaces/hermes/peers", json={"id": peer_id}
    ).status_code == 201
    assert client.post(
        "/v3/workspaces/hermes/sessions/long-peer-budget/messages",
        json={"messages": [{"content": "資料庫🙂", "peer_id": peer_id}]},
    ).status_code == 201
    tight = client.get(
        "/v3/workspaces/hermes/sessions/long-peer-budget/context"
        "?summary=false&tokens=200"
    )
    wide = client.get(
        "/v3/workspaces/hermes/sessions/long-peer-budget/context"
        "?summary=false&tokens=1000"
    )
    assert tight.status_code == wide.status_code == 200
    assert tight.json()["messages"] == []
    assert len(wide.json()["messages"]) == 1

    # Build the SDK object and inspect its real to_openai() shape. The server's
    # conservative accounting must bound every role/name/content/framing token.
    from honcho.api_types import MessageResponse
    from honcho.message import Message as HonchoMessage
    from honcho.session_context import SessionContext

    data = wide.json()
    sdk_context = SessionContext(
        session_id="long-peer-budget",
        messages=[
            HonchoMessage.from_api_response(MessageResponse.model_validate(item))
            for item in data["messages"]
        ],
        summary=None,
        peer_representation=data.get("peer_representation") or None,
    )
    openai_messages = sdk_context.to_openai(assistant="agent-main")
    visible_cost = 0
    for message in openai_messages:
        serialized = f"role:{message['role']}\n"
        if "name" in message:
            serialized += f"name:{message['name']}\n"
        serialized += f"content:{message['content']}"
        visible_cost += estimate_tokens(serialized) + 4
    assert visible_cost <= 1000
    assert data["context_token_count"] >= visible_cost


@pytest.mark.parametrize("graph_first", [False, True])
def test_search_occurrence_dedup_merges_later_provenance_after_limit(
    monkeypatch, graph_first
):
    occurrence = SourceOccurrence("s", 1, "p", "w")
    raw_hit = MessageHit(
        1, "s", "user", "exact", -10, source_peer_id="p",
        source_workspace_id="w", source_occurrences=(occurrence,),
    )
    raw = FusedEvidence(
        "message:1", "message", raw_hit, .8, False, (occurrence,),
        (occurrence,),
        (RetrievalProvenance("message", "1", 1, -10, "coverage_lexical"),),
        ("message",),
    )
    graph = FusedEvidence(
        "graph:7", "graph", object(), .7, False, (occurrence,), (),
        (RetrievalProvenance("graph", "7", 2, .4, "rrf", ("graph",)),),
        ("graph",),
    )

    class FakeHy:
        def augment(self, *args, **kwargs):
            return SimpleNamespace(
                fused_evidence=[graph, raw] if graph_first else [raw, graph]
            )

    monkeypatch.setattr(hsrv, "_get_hy", lambda: FakeHy())
    monkeypatch.setattr(
        hsrv,
        "_graph_fact_message",
        lambda *args, **kwargs: {
            "id": "msg_1", "content": "exact", "peer_id": "p",
            "session_id": "s", "workspace_id": "w", "created_at": "now",
            "token_count": 1,
            "metadata": {
                "type": "graph_fact", "edge_id": 7,
                "graph_claim": {"subject": "a", "predicate": "uses", "object": "b"},
                "citations": [{"evidence_id": 4, "source_session_id": "s",
                               "source_message_id": 1, "source_peer_id": "p",
                               "source_workspace_id": "w"}],
            },
        },
    )
    result = hsrv._augment_messages("q", 1, "w", session_id="s")[0]
    assert set(result["metadata"]["source_tiers"]) == {"message", "graph"}
    assert {item["tier"] for item in result["metadata"]["retrieval_provenance"]} == {
        "message", "graph",
    }
    assert result["metadata"]["graph_claims"] == [
        {"subject": "a", "predicate": "uses", "object": "b"}
    ]


def test_peer_card_does_not_expose_unowned_rules_or_profile(client, hy_with_embed):
    hy_with_embed.config.user_md_path.write_text(
        "# Profile\n\n- prefers uv\n", encoding="utf-8"
    )
    hy_with_embed.add_rule("always run tests before pushing")
    r = client.get("/v3/workspaces/hermes/peers/user-1/card")
    assert r.status_code == 200
    content = r.json()["content"]
    assert content == ""
