from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import hymem.honcho_server as hsrv
from tests.conftest import make_routed_llm


@pytest.fixture
def client(hy_with_embed):
    hsrv.set_hy(hy_with_embed)
    return TestClient(hsrv.app)


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "backend": "hymem"}


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


def test_search_returns_empty_for_unknown_query(client):
    r = client.post(
        "/v3/workspaces/hermes/sessions/s/search",
        json={"query": "totally unknown topic xyz"},
    )
    assert r.status_code == 200
    assert r.json() == []


def test_search_returns_graph_facts_after_dream(client, hy_with_embed):
    sid = "s-search"
    hy_with_embed.open_session(sid)
    hy_with_embed.log_message(sid, "assistant", "We could try Docker for the local dev environment.")
    hy_with_embed.log_message(
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
    assert any(item["peer_id"] == "hymem-kg" for item in body)
    assert any("docker" in item["content"].lower() for item in body)


def test_context_returns_summary_messages_peers(client, hy_with_embed):
    sid = "s-ctx"
    hy_with_embed.open_session(sid)
    hy_with_embed.log_message(sid, "user", "first message")
    hy_with_embed.log_message(sid, "assistant", "second message")
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


def test_peer_card_returns_user_md_content(client, hy_with_embed):
    hy_with_embed.config.user_md_path.write_text(
        "# Behavioral Profile\n\n- prefers uv\n", encoding="utf-8"
    )
    r = client.get("/v3/workspaces/hermes/peers/user-1/card")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "user-1"
    assert "Behavioral Profile" in body["content"]
    assert "prefers uv" in body["content"]


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
    assert hsrv._infer_role("user-123") == "user"
    assert hsrv._infer_role("agent-main") == "assistant"
    assert hsrv._infer_role("hermes") == "assistant"
    assert hsrv._infer_role("telegram-12345") == "user"
    assert hsrv._infer_role("ai-bot") == "assistant"


def test_dream_cooldown_throttles_back_to_back_calls(client, hy_with_embed):
    hsrv._last_dream_kick = 0.0
    hsrv._DREAM_COOLDOWN_SECONDS = 60.0

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

    count = hy_with_embed.conn.execute(
        "SELECT COUNT(*) FROM dream_runs"
    ).fetchone()[0]
    assert count == 1


def test_dream_cooldown_allows_after_window(client, hy_with_embed):
    hsrv._last_dream_kick = 0.0
    hsrv._DREAM_COOLDOWN_SECONDS = 0.0

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
    hy_with_embed.open_session(sid)
    for i in range(5):
        hy_with_embed.log_message(sid, "user", f"msg-{i}")

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
    hy_with_embed.open_session("empty-sid")
    r = client.post(
        "/v3/workspaces/hermes/sessions/empty-sid/messages/list",
        json={},
    )
    assert r.status_code == 200
    assert r.json() == {"items": [], "total": 0, "page": 1, "size": 50, "pages": 0}
