from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from hymem import HyMem, HyMemConfig
from hymem import portability
from hymem import session as session_log
from hymem.core import db as core_db
from hymem.core.message_records import (
    MESSAGE_CONTENT_HASH_VERSION,
    MESSAGE_PROVENANCE_HASH_VERSION,
    MESSAGE_PROVENANCE_RECORD_VERSION,
    MESSAGE_RECORD_VERSION,
    canonical_message_record,
    encode_message_record,
    message_content_hash,
)
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import (
    coverage_chunk_id,
    validate_message_coverage_artifact,
)
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.dreaming.retention import prune_messages
from hymem.extraction.triples import Triple
from hymem.query.augment import _graph_lookup
import hymem.honcho.app as hsrv


class _NoopScheduler:
    def kick(self) -> bool:
        return False

    def stop(self) -> None:
        return None


class _ScopedMappingEmbedder:
    model = "scoped-mapping-v1"
    dim = 2

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def provenance_client(hy):
    hsrv.set_hy(hy)
    if hsrv._scheduler is not None:
        hsrv._scheduler.stop()
    hsrv.set_scheduler(_NoopScheduler())
    with TestClient(hsrv.app) as client:
        yield client


def _publish_claim(
    hy: HyMem,
    *,
    session_id: str,
    peer_id: str,
    content: str,
    chunk_id: str,
    workspace_id: str = "w",
    polarity: int = 1,
    subject: str = "shared_service",
    predicate: str = "uses",
    object_: str = "redis",
    created_at: str | None = None,
) -> int:
    hy.open_session(session_id, source_workspace_id=workspace_id)
    message_id = hy.log_message(
        session_id,
        "user",
        content,
        created_at=created_at,
        source_peer_id=peer_id,
        source_workspace_id=workspace_id,
    )
    chunk = Chunk(
        id=chunk_id,
        session_id=session_id,
        start_message_id=message_id,
        end_message_id=message_id,
        salience_reason="peer-provenance-test",
        text=f"user: {content}",
        source_message_ids=(message_id,),
    )
    with core_db.transaction(hy.conn):
        persist_chunks(hy.conn, [chunk])
    sources = phase1._claim_sources_for_chunk(hy.conn, chunk)
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn,
            chunk,
            ChunkExtraction(
                triples=[Triple(
                    subject,
                    predicate,
                    object_,
                    polarity,
                    source_message_id=message_id,
                )],
                markers=[],
                claim_sources={source.message_id: source for source in sources},
                source_validated=True,
            ),
            prompt_version="v13",
            cfg=hy.config,
        )
    return int(message_id)


def _rewrite_wire(path: Path, mutate) -> None:
    objects = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    body, end = objects[:-1], objects[-1]
    mutate(body)
    encoded = [
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        for item in body
    ]
    end["counts"] = {
        kind: sum(item.get("type") == kind for item in body)
        for kind in end["counts"]
    }
    end["sha256"] = hashlib.sha256("".join(encoded).encode()).hexdigest()
    path.write_text(
        "".join(encoded)
        + json.dumps(end, ensure_ascii=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _unsafe_v2_record(
    *,
    message_id: int,
    session_id: str,
    content: str,
    source_created_at: object,
) -> tuple[str, str]:
    """Build adversarial v2 bytes without the production encoder's guards."""
    record = json.dumps(
        {
            "content": content,
            "id": message_id,
            "record_version": MESSAGE_PROVENANCE_RECORD_VERSION,
            "role": "user",
            "session_id": session_id,
            "source_created_at": source_created_at,
            "source_peer_id": "alice",
            "source_workspace_id": "w",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return record, hashlib.sha256(record.encode("utf-8")).hexdigest()


def _project_wire_to_v7(path: Path) -> None:
    def project(body: list[dict]) -> None:
        chunks = {
            item["record"]["id"]: item["record"]
            for item in body if item["type"] == "chunk"
        }
        # A real v7 stream predates attributed v2 records. Explicitly perform
        # the complete downgrade rather than teaching the frozen v7 reader to
        # accept a record version that did not exist in that format.
        for item in body:
            if item["type"] != "message_retention_coverage":
                continue
            proof = item["record"]
            chunk = chunks[proof["chunk_id"]]
            payload = json.loads(chunk["text"])
            record = encode_message_record(
                message_id=proof["message_id"],
                role=proof["source_role"],
                content=payload["content"],
            )
            chunk["text"] = record
            proof["message_content_hash"] = message_content_hash(
                proof["source_role"], payload["content"]
            )
            proof["hash_version"] = MESSAGE_CONTENT_HASH_VERSION
            proof["record_version"] = MESSAGE_RECORD_VERSION
        projected: list[dict] = []
        for item in body:
            if item["type"] == "_meta":
                item["version"] = 7
                item["schema_version"] = 42
                projected.append(item)
                continue
            kind = item["type"]
            if kind not in portability._V7_COLS_BY_KIND:
                continue
            item["record"] = {
                column: item["record"][column]
                for column in portability._V7_COLS_BY_KIND[kind]
            }
            projected.append(item)
        body[:] = projected

    _rewrite_wire(path, project)
    lines = path.read_text(encoding="utf-8").splitlines()
    end = json.loads(lines[-1])
    body = [json.loads(line) for line in lines[:-1]]
    end["counts"] = {
        kind: sum(item.get("type") == kind for item in body)
        for kind in portability._V7_TABLE_BY_KIND
    }
    path.write_text(
        "\n".join(lines[:-1]) + "\n"
        + json.dumps(end, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_sdk_context_to_openai_preserves_exact_peer_ids(
    provenance_client,
):
    from honcho import Message, SessionContext
    from honcho.api_types import MessageResponse

    assert provenance_client.post(
        "/v3/workspaces/w/sessions", json={"id": "sdk-session"}
    ).status_code == 201
    response = provenance_client.post(
        "/v3/workspaces/w/sessions/sdk-session/messages",
        json={"messages": [
            {"content": "hello", "peer_id": "user-alice"},
            {"content": "hi", "peer_id": "agent-main"},
        ]},
    )
    assert response.status_code == 201
    context_json = provenance_client.get(
        "/v3/workspaces/w/sessions/sdk-session/context"
    ).json()
    messages = [
        Message.from_api_response(MessageResponse.model_validate(item))
        for item in context_json["messages"]
    ]
    context = SessionContext(session_id="sdk-session", messages=messages)
    assert context.to_openai(assistant="agent-main") == [
        {"role": "user", "name": "user-alice", "content": "hello"},
        {"role": "assistant", "name": "agent-main", "content": "hi"},
    ]
    listed = provenance_client.post(
        "/v3/workspaces/w/sessions/sdk-session/messages/list", json={}
    ).json()["items"]
    assert [item["peer_id"] for item in listed] == ["user-alice", "agent-main"]
    assert context_json["peers"] == [
        {"id": "agent-main"}, {"id": "user-alice"}
    ]


def test_session_create_accepts_sdk_peers_and_keeps_configuration_per_session(
    provenance_client, hy
):
    response = provenance_client.post(
        "/v3/workspaces/w/sessions",
        json={
            "id": "configured",
            "peers": {
                "user-alice": {"observe_me": False},
                "agent-main": {"observe_others": False},
            },
        },
    )
    assert response.status_code == 201
    assert hy.conn.execute(
        "SELECT source_workspace_id FROM sessions WHERE id='configured'"
    ).fetchone()[0] == "w"
    configurations = {
        row["peer_id"]: json.loads(row["configuration"])
        for row in hy.conn.execute(
            "SELECT peer_id,configuration FROM session_peers "
            "WHERE session_id='configured' ORDER BY peer_id"
        )
    }
    assert configurations == {
        "agent-main": {"observe_others": False},
        "user-alice": {"observe_me": False},
    }
    assert all(
        json.loads(row["metadata"]) == {}
        for row in hy.conn.execute("SELECT metadata FROM peers")
    )


def test_unknown_message_peer_is_atomically_registered_with_exact_identity(
    provenance_client, hy
):
    response = provenance_client.post(
        "/v3/workspaces/w/sessions/implicit/messages",
        json={"messages": [{
            "content": "created with my message", "peer_id": "person-42"
        }]},
    )
    assert response.status_code == 201
    assert response.json()[0]["peer_id"] == "person-42"
    assert tuple(hy.conn.execute(
        "SELECT source_peer_id,source_workspace_id,role FROM messages"
    ).fetchone()) == ("person-42", "w", "user")
    assert hy.conn.execute(
        "SELECT 1 FROM session_peers WHERE session_id='implicit' "
        "AND workspace_id='w' AND peer_id='person-42'"
    ).fetchone() is not None


def test_upload_uses_existing_registered_role_and_exact_peer(
    provenance_client, hy
):
    assert provenance_client.post(
        "/v3/workspaces/w/peers", json={"id": "ambiguous"}
    ).status_code == 201
    hy.conn.execute(
        "UPDATE peers SET role='assistant' "
        "WHERE id='ambiguous' AND workspace_id='w'"
    )
    response = provenance_client.post(
        "/v3/workspaces/w/sessions/upload/messages/upload",
        data={"peer_id": "ambiguous"},
        files={"file": ("memory.txt", b"uploaded", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()[0]["peer_id"] == "ambiguous"
    assert tuple(hy.conn.execute(
        "SELECT role,source_peer_id,source_workspace_id FROM messages"
    ).fetchone()) == ("assistant", "ambiguous", "w")


def test_wrong_workspace_session_matrix_fails_without_leak_or_mutation(
    provenance_client, hy
):
    assert provenance_client.post(
        "/v3/workspaces/w1/sessions", json={"id": "shared"}
    ).status_code == 201
    assert provenance_client.post(
        "/v3/workspaces/w1/sessions/shared/messages",
        json={"messages": [{"content": "private legacy history", "peer_id": "alice"}]},
    ).status_code == 201
    before = tuple(hy.conn.execute(
        "SELECT (SELECT COUNT(*) FROM messages),"
        "(SELECT COUNT(*) FROM peers),"
        "(SELECT COUNT(*) FROM session_peers)"
    ).fetchone())
    checks = [
        provenance_client.post(
            "/v3/workspaces/w2/sessions", json={"id": "shared"}
        ),
        provenance_client.get("/v3/workspaces/w2/sessions/shared"),
        provenance_client.get("/v3/workspaces/w2/sessions/shared/context"),
        provenance_client.post(
            "/v3/workspaces/w2/sessions/shared/search", json={"query": "legacy"}
        ),
        provenance_client.post(
            "/v3/workspaces/w2/sessions/shared/messages/list", json={}
        ),
        provenance_client.post(
            "/v3/workspaces/w2/sessions/shared/messages",
            json={"messages": [{"content": "attack", "peer_id": "mallory"}]},
        ),
        provenance_client.post(
            "/v3/workspaces/w2/sessions/shared/messages/upload",
            data={"peer_id": "mallory"},
            files={"file": ("attack.txt", b"attack", "text/plain")},
        ),
        provenance_client.post(
            "/v3/workspaces/w2/sessions/shared/peers",
            json={"peers": [{"id": "mallory"}]},
        ),
        provenance_client.get(
            "/v3/workspaces/w2/sessions/shared/peers/alice/config"
        ),
    ]
    assert [response.status_code for response in checks] == [
        409, 404, 404, 404, 404, 409, 409, 409, 404
    ]
    assert all("private legacy history" not in response.text for response in checks)
    after = tuple(hy.conn.execute(
        "SELECT (SELECT COUNT(*) FROM messages),"
        "(SELECT COUNT(*) FROM peers),"
        "(SELECT COUNT(*) FROM session_peers)"
    ).fetchone())
    assert after == before
    assert hy.conn.execute(
        "SELECT 1 FROM peers WHERE workspace_id='w2'"
    ).fetchone() is None


def test_raw_peer_and_session_scope_is_applied_before_limit(
    provenance_client,
):
    for index in range(8):
        assert provenance_client.post(
            "/v3/workspaces/w/sessions/other/messages",
            json={"messages": [{
                "content": f"needle needle needle bob-{index}",
                "peer_id": "bob",
            }]},
        ).status_code == 201
    assert provenance_client.post(
        "/v3/workspaces/w/sessions/target/messages",
        json={"messages": [{"content": "needle alice", "peer_id": "alice"}]},
    ).status_code == 201

    alice = provenance_client.post(
        "/v3/workspaces/w/peers/alice/search",
        json={"query": "needle", "limit": 1},
    )
    assert alice.status_code == 200
    assert [(item["peer_id"], item["session_id"]) for item in alice.json()] == [
        ("alice", "target")
    ]
    target = provenance_client.post(
        "/v3/workspaces/w/sessions/target/search",
        json={"query": "needle", "limit": 1},
    )
    assert target.status_code == 200
    assert [(item["peer_id"], item["session_id"]) for item in target.json()] == [
        ("alice", "target")
    ]


def test_mixed_author_chunk_is_never_emitted_as_peer_message(
    provenance_client, hy
):
    response = provenance_client.post(
        "/v3/workspaces/w/sessions/mixed/messages",
        json={"messages": [
            {"content": "needle alice private", "peer_id": "alice"},
            {"content": "needle bob private", "peer_id": "bob"},
        ]},
    )
    message_ids = [int(item["id"].removeprefix("msg_")) for item in response.json()]
    chunk = Chunk(
        id="mixed-authors",
        session_id="mixed",
        start_message_id=message_ids[0],
        end_message_id=message_ids[1],
        salience_reason="mixed",
        text="user: needle alice private\nuser: needle bob private",
        source_message_ids=tuple(message_ids),
    )
    with core_db.transaction(hy.conn):
        persist_chunks(hy.conn, [chunk])
    for peer_id, own, foreign in (
        ("alice", "alice private", "bob private"),
        ("bob", "bob private", "alice private"),
    ):
        result = provenance_client.post(
            f"/v3/workspaces/w/peers/{peer_id}/search",
            json={"query": "needle", "limit": 1},
        ).json()
        assert len(result) == 1
        assert result[0]["peer_id"] == peer_id
        assert own in result[0]["content"]
        assert foreign not in result[0]["content"]
        assert result[0]["metadata"]["type"] == "message_hit"


def test_multi_citation_graph_search_selects_only_requested_peer_and_session(
    provenance_client, hy
):
    _publish_claim(
        hy,
        session_id="alice-session",
        peer_id="alice",
        content="alice says shared service uses redis",
        chunk_id="alice-claim",
    )
    _publish_claim(
        hy,
        session_id="bob-session",
        peer_id="bob",
        content="bob says shared service uses redis",
        chunk_id="bob-claim",
    )
    for peer_id, session_id, excluded in (
        ("alice", "alice-session", "bob"),
        ("bob", "bob-session", "alice"),
    ):
        response = provenance_client.post(
            f"/v3/workspaces/w/peers/{peer_id}/search",
            json={"query": "shared service redis", "limit": 1},
        )
        assert response.status_code == 200
        item = response.json()[0]
        assert item["metadata"]["type"] == "graph_fact"
        assert item["peer_id"] == peer_id
        assert item["session_id"] == session_id
        assert excluded not in item["content"]
        assert {c["source_peer_id"] for c in item["metadata"]["citations"]} == {
            peer_id
        }
        assert "confidence" not in item["metadata"]
        assert "+" not in item["metadata"] and "-" not in item["metadata"]

        by_session = provenance_client.post(
            f"/v3/workspaces/w/sessions/{session_id}/search",
            json={"query": "shared service redis", "limit": 1},
        ).json()[0]
        assert by_session["peer_id"] == peer_id
        assert by_session["session_id"] == session_id


def test_bound_message_coverage_and_registry_reject_tampering(hy):
    hy.open_session("guarded", source_workspace_id="w")
    message_id = hy.log_message(
        "guarded", "user", "exact", source_peer_id="alice",
        source_workspace_id="w",
    )
    session_log.register_session_peer(
        hy.conn, "guarded", "w", "bob", "user"
    )
    coverage = hy.conn.execute(
        "SELECT chunk_id,coverage_version FROM message_retention_coverage "
        "WHERE message_id=?", (message_id,)
    ).fetchone()
    invalid_message_sql = (
        "INSERT INTO messages(session_id,role,source_peer_id,"
        "source_workspace_id,content) VALUES (?,?,?,?,?)"
    )
    for values in (
        ("guarded", "user", None, None, "missing exact author"),
        ("guarded", "assistant", "alice", "w", "role mismatch"),
        ("guarded", "user", "alice", None, "partial pair"),
    ):
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute(invalid_message_sql, values)
    with pytest.raises(sqlite3.IntegrityError):
        hy.conn.execute(
            "UPDATE message_retention_coverage SET source_peer_id='bob' "
            "WHERE message_id=?", (message_id,)
        )

    # Read validation remains fail-closed if the write trigger was removed.
    hy.conn.execute("DROP TRIGGER message_coverage_peer_update_guard")
    hy.conn.execute("DROP TRIGGER message_retention_coverage_update_guard")
    hy.conn.execute("DROP TRIGGER message_lossless_stream_update_guard")
    hy.conn.execute(
        "UPDATE message_retention_coverage SET source_peer_id='bob' "
        "WHERE message_id=?", (message_id,)
    )
    with pytest.raises(RuntimeError, match="coverage proof mismatch"):
        validate_message_coverage_artifact(
            hy.conn,
            message_id=message_id,
            chunk_id=coverage["chunk_id"],
            coverage_version=coverage["coverage_version"],
        )


def test_peer_registry_remains_valid_after_raw_retention(hy, tmp_path):
    hy.open_session("retained", source_workspace_id="w")
    message_id = hy.log_message(
        "retained",
        "user",
        "old exact source",
        created_at="2020-01-01T00:00:00Z",
        source_peer_id="alice",
        source_workspace_id="w",
    )
    hy.close_session("retained")
    assert prune_messages(
        hy.conn, replace(hy.config, message_retention_days=1)
    ) == 1
    assert hy.conn.execute(
        "SELECT 1 FROM messages WHERE id=?", (message_id,)
    ).fetchone() is None
    coverage = hy.conn.execute(
        "SELECT chunk_id,coverage_version FROM message_retention_coverage "
        "WHERE message_id=?", (message_id,)
    ).fetchone()
    proof = validate_message_coverage_artifact(
        hy.conn,
        message_id=message_id,
        chunk_id=coverage["chunk_id"],
        coverage_version=coverage["coverage_version"],
    )
    assert (proof.source_peer_id, proof.source_workspace_id) == ("alice", "w")
    for sql in (
        "DELETE FROM session_peers WHERE session_id='retained' AND peer_id='alice'",
        "UPDATE peers SET role='assistant' WHERE id='alice' AND workspace_id='w'",
        "DELETE FROM peers WHERE id='alice' AND workspace_id='w'",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute(sql)
    wire = tmp_path / "retained-v8.jsonl"
    counts = hy.export(wire)
    assert counts["peer"] == 1 and counts["session_peer"] == 1


@pytest.mark.parametrize("artifact", ["summary", "episode", "procedure"])
def test_nonpristine_native_session_cannot_be_claimed_by_workspace(hy, artifact):
    session_id = f"native-{artifact}"
    hy.open_session(session_id)
    if artifact == "summary":
        hy.conn.execute(
            "UPDATE sessions SET summary='private legacy history' WHERE id=?",
            (session_id,),
        )
    elif artifact == "episode":
        hy.conn.execute(
            "INSERT INTO episodes(id,session_id,title,summary) VALUES (?,?,?,?)",
            ("episode-1", session_id, "private", "legacy history"),
        )
    else:
        hy.conn.execute(
            "INSERT INTO procedures(id,session_id,name,steps) VALUES (?,?,?,?)",
            ("procedure-1", session_id, "private", "[]"),
        )
    with pytest.raises(ValueError, match="cannot infer workspace ownership"):
        session_log.open_session(
            hy.conn, session_id, source_workspace_id="attacker"
        )
    with pytest.raises(sqlite3.IntegrityError, match="binding is immutable"):
        hy.conn.execute(
            "UPDATE sessions SET source_workspace_id='attacker' WHERE id=?",
            (session_id,),
        )


def test_session_peer_cannot_attach_to_unbound_native_session(hy):
    hy.open_session("native")
    hy.conn.execute(
        "INSERT INTO peers(id,workspace_id,role,metadata) "
        "VALUES ('alice','w','user','{}')"
    )
    with pytest.raises(sqlite3.IntegrityError, match="does not match session"):
        hy.conn.execute(
            "INSERT INTO session_peers(session_id,workspace_id,peer_id) "
            "VALUES ('native','w','alice')"
        )


def test_v8_roundtrip_preserves_peer_after_raw_is_absent_and_replays_exactly(
    tmp_path,
):
    src = HyMem(HyMemConfig(root=tmp_path / "src"))
    try:
        _publish_claim(
            src,
            session_id="portable",
            peer_id="alice",
            content="shared service uses redis",
            chunk_id="portable-claim",
        )
        wire = tmp_path / "peer-v8.jsonl"
        src.export(wire)
    finally:
        src.close()
    dst = HyMem(HyMemConfig(root=tmp_path / "dst"))
    try:
        imported = dst.import_(wire)
        assert imported["peer"] == 1 and imported["session_peer"] == 1
        assert dst.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert tuple(dst.conn.execute(
            "SELECT source_peer_id,source_workspace_id "
            "FROM message_retention_coverage"
        ).fetchone()) == ("alice", "w")
        assert tuple(dst.conn.execute(
            "SELECT source_peer_id,source_workspace_id FROM kg_evidence"
        ).fetchone()) == ("alice", "w")
        assert sum(dst.import_(wire).values()) == 0
    finally:
        dst.close()


def test_v7_projection_imports_as_explicitly_unattributed(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "v7-src"))
    try:
        _publish_claim(
            src,
            session_id="legacy",
            peer_id="alice",
            content="shared service uses redis",
            chunk_id="legacy-claim",
        )
        wire = tmp_path / "projected-v7.jsonl"
        src.export(wire)
        _project_wire_to_v7(wire)
    finally:
        src.close()
    dst = HyMem(HyMemConfig(root=tmp_path / "v7-dst"))
    try:
        dst.import_(wire)
        assert dst.conn.execute(
            "SELECT source_workspace_id FROM sessions WHERE id='legacy'"
        ).fetchone()[0] is None
        assert tuple(dst.conn.execute(
            "SELECT source_peer_id,source_workspace_id "
            "FROM message_retention_coverage"
        ).fetchone()) == (None, None)
        assert tuple(dst.conn.execute(
            "SELECT source_peer_id,source_workspace_id FROM kg_evidence"
        ).fetchone()) == (None, None)
        assert dst.conn.execute("SELECT COUNT(*) FROM session_peers").fetchone()[0] == 0
    finally:
        dst.close()


@pytest.mark.parametrize("tamper", ["partial_coverage", "mismatched_evidence"])
def test_v8_peer_tampering_rejects_before_any_destination_write(tmp_path, tamper):
    src = HyMem(HyMemConfig(root=tmp_path / f"tamper-src-{tamper}"))
    try:
        _publish_claim(
            src,
            session_id="tampered",
            peer_id="alice",
            content="shared service uses redis",
            chunk_id="tampered-claim",
        )
        wire = tmp_path / f"{tamper}.jsonl"
        src.export(wire)
    finally:
        src.close()

    def mutate(body: list[dict]) -> None:
        kind = (
            "message_retention_coverage"
            if tamper == "partial_coverage" else "edge_evidence"
        )
        record = next(item["record"] for item in body if item["type"] == kind)
        if tamper == "partial_coverage":
            record["source_workspace_id"] = None
        else:
            record["source_peer_id"] = "mallory"

    _rewrite_wire(wire, mutate)
    dst = HyMem(HyMemConfig(root=tmp_path / f"tamper-dst-{tamper}"))
    try:
        with pytest.raises(ValueError, match="portable"):
            dst.import_(wire)
        assert dst.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert dst.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert dst.conn.execute("SELECT COUNT(*) FROM peers").fetchone()[0] == 0
    finally:
        dst.close()


def test_v8_session_peer_configuration_collision_is_fail_closed(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "config-src"))
    try:
        src.open_session("configured", source_workspace_id="w")
        with core_db.transaction(src.conn):
            session_log.register_session_peer(
                src.conn,
                "configured",
                "w",
                "alice",
                "user",
                configuration={"observe_me": False},
            )
        wire = tmp_path / "config-v8.jsonl"
        src.export(wire)
    finally:
        src.close()
    dst = HyMem(HyMemConfig(root=tmp_path / "config-dst"))
    try:
        dst.import_(wire)
        assert sum(dst.import_(wire).values()) == 0
        dst.conn.execute(
            "UPDATE session_peers SET configuration=? WHERE session_id='configured'",
            (json.dumps({"observe_me": True}),),
        )
        before = tuple(dst.conn.execute(
            "SELECT COUNT(*),MIN(configuration) FROM session_peers"
        ).fetchone())
        with pytest.raises(ValueError, match="session_peer collides"):
            dst.import_(wire)
        assert tuple(dst.conn.execute(
            "SELECT COUNT(*),MIN(configuration) FROM session_peers"
        ).fetchone()) == before
    finally:
        dst.close()


def test_add_peers_keeps_metadata_and_exact_session_configuration(
    provenance_client, hy
):
    response = provenance_client.post(
        "/v3/workspaces/w/sessions/separate/peers",
        json={"peers": [{
            "id": "alice",
            "metadata": {"display_name": "Alice"},
            "configuration": {"observe_me": False},
        }]},
    )
    assert response.status_code == 201
    assert json.loads(hy.conn.execute(
        "SELECT metadata FROM peers WHERE id='alice' AND workspace_id='w'"
    ).fetchone()[0]) == {"display_name": "Alice"}
    assert json.loads(hy.conn.execute(
        "SELECT configuration FROM session_peers "
        "WHERE session_id='separate' AND peer_id='alice'"
    ).fetchone()[0]) == {"observe_me": False}

    # Updating membership configuration never overwrites global registry data.
    response = provenance_client.post(
        "/v3/workspaces/w/sessions/separate/peers",
        json={"peers": [{
            "id": "alice",
            "metadata": {"display_name": "Mallory"},
            "configuration": {"observe_others": False},
        }]},
    )
    assert response.status_code == 201
    assert json.loads(hy.conn.execute(
        "SELECT metadata FROM peers WHERE id='alice' AND workspace_id='w'"
    ).fetchone()[0]) == {"display_name": "Alice"}
    assert json.loads(hy.conn.execute(
        "SELECT configuration FROM session_peers "
        "WHERE session_id='separate' AND peer_id='alice'"
    ).fetchone()[0]) == {"observe_others": False}

    bare = provenance_client.post(
        "/v3/workspaces/w/sessions/separate/peers",
        json={"bob": {"observe_me": False}},
    )
    assert bare.status_code == 201
    assert json.loads(hy.conn.execute(
        "SELECT configuration FROM session_peers "
        "WHERE session_id='separate' AND peer_id='bob'"
    ).fetchone()[0]) == {"observe_me": False}


@pytest.mark.parametrize("field,value", [
    ("metadata", []),
    ("metadata", False),
    ("configuration", ""),
    ("configuration", []),
])
def test_add_peers_rejects_falsey_non_object_fields(
    provenance_client, field, value
):
    response = provenance_client.post(
        "/v3/workspaces/w/sessions/invalid-peers/peers",
        json={"peers": [{"id": "alice", field: value}]},
    )
    assert response.status_code == 422


def test_session_create_rejects_non_object_peer_configuration(
    provenance_client,
):
    response = provenance_client.post(
        "/v3/workspaces/w/sessions",
        json={"id": "bad-config", "peers": {"alice": []}},
    )
    assert response.status_code == 422


def test_peer_card_matches_pinned_sdk_wire(provenance_client, hy):
    from honcho.api_types import PeerCardResponse

    assert provenance_client.post(
        "/v3/workspaces/w/peers", json={"id": "alice"}
    ).status_code == 201
    assert provenance_client.post(
        "/v3/workspaces/w/sessions",
        json={"id": "card-session", "peers": {"alice": {}}},
    ).status_code == 201
    assert provenance_client.post(
        "/v3/workspaces/w/sessions/card-session/messages",
        json={"messages": [{
            "peer_id": "alice", "content": "peer-owned preference: uv",
        }]},
    ).status_code == 201
    hy.config.user_md_path.write_text("GLOBAL PROFILE SECRET", encoding="utf-8")
    body = provenance_client.get(
        "/v3/workspaces/w/peers/alice/card"
    ).json()
    assert provenance_client.get(
        "/v3/workspaces/w/peers/alice/card"
    ).json() == body
    parsed = PeerCardResponse.model_validate(body)
    assert parsed.peer_card == [body["content"]]
    assert "peer-owned preference: uv" in parsed.peer_card[0]
    assert "GLOBAL PROFILE SECRET" not in parsed.peer_card[0]


def test_honcho_write_responses_use_persisted_redacted_and_truncated_content(
    tmp_path,
):
    local = HyMem(HyMemConfig(
        root=tmp_path / "normalized-responses",
        max_message_chars=50,
        redact_secrets=True,
    ))
    hsrv.set_hy(local)
    hsrv.set_scheduler(_NoopScheduler())
    try:
        with TestClient(hsrv.app) as client:
            response = client.post(
                "/v3/workspaces/w/sessions/normalized/messages",
                json={"messages": [
                    {"peer_id": "alice", "content": "email alice@example.com"},
                    {"peer_id": "alice", "content": "x" * 80},
                ]},
            )
            assert response.status_code == 201
            returned = [item["content"] for item in response.json()]
            stored = [row["content"] for row in local.conn.execute(
                "SELECT content FROM messages ORDER BY id"
            )]
            assert returned == stored
            assert returned[0] == "email [REDACTED-EMAIL]"
            assert returned[1] == "x" * 50 + "\n[TRUNCATED]"

            uploaded = client.post(
                "/v3/workspaces/w/sessions/normalized/messages/upload",
                data={"peer_id": "alice"},
                files={
                    "file": (
                        "secret.txt", b"email bob@example.com", "text/plain"
                    )
                },
            )
            assert uploaded.status_code == 200
            assert uploaded.json()[0]["content"] == "email [REDACTED-EMAIL]"
            for row in local.conn.execute(
                "SELECT message_id,chunk_id,coverage_version "
                "FROM message_retention_coverage ORDER BY message_id"
            ):
                proof = validate_message_coverage_artifact(
                    local.conn,
                    message_id=row["message_id"],
                    chunk_id=row["chunk_id"],
                    coverage_version=row["coverage_version"],
                )
                assert "example.com" not in proof.content
    finally:
        local.close()


def test_list_and_context_read_exact_coverage_before_and_after_raw_pruning(
    provenance_client, hy
):
    response = provenance_client.post(
        "/v3/workspaces/w/sessions/retained-sdk/messages",
        json={"messages": [
            {
                "peer_id": "alice",
                "content": f"retained turn {index}",
                "created_at": f"2020-01-0{index + 1}T00:00:00Z",
            }
            for index in range(3)
        ]},
    )
    assert response.status_code == 201

    def read() -> tuple[list[str], list[str]]:
        first = provenance_client.post(
            "/v3/workspaces/w/sessions/retained-sdk/messages/list",
            json={"page": 1, "size": 2},
        ).json()
        second = provenance_client.post(
            "/v3/workspaces/w/sessions/retained-sdk/messages/list",
            json={"page": 2, "size": 2},
        ).json()
        assert (first["total"], first["pages"], second["total"]) == (3, 2, 3)
        listed = [item["content"] for item in first["items"] + second["items"]]
        context = provenance_client.get(
            "/v3/workspaces/w/sessions/retained-sdk/context"
        ).json()
        return listed, [item["content"] for item in context["messages"]]

    expected = [f"retained turn {index}" for index in range(3)]
    assert read() == (expected, expected)
    hy.close_session("retained-sdk")
    assert prune_messages(
        hy.conn, replace(hy.config, message_retention_days=1)
    ) == 3
    assert hy.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='retained-sdk'"
    ).fetchone()[0] == 0
    assert read() == (expected, expected)


def test_list_and_context_fail_closed_on_raw_or_proof_tampering(
    provenance_client, hy
):
    response = provenance_client.post(
        "/v3/workspaces/w/sessions/tampered-sdk/messages",
        json={"messages": [{"peer_id": "alice", "content": "original secret"}]},
    )
    message_id = int(response.json()[0]["id"].removeprefix("msg_"))
    hy.conn.execute("DROP TRIGGER message_lossless_source_update_guard")
    hy.conn.execute(
        "UPDATE messages SET content='forged raw leak' WHERE id=?",
        (message_id,),
    )
    listed = provenance_client.post(
        "/v3/workspaces/w/sessions/tampered-sdk/messages/list", json={}
    ).json()
    context = provenance_client.get(
        "/v3/workspaces/w/sessions/tampered-sdk/context"
    ).json()
    assert listed["items"] == [] and listed["total"] == 0
    assert context["messages"] == []
    assert "forged raw leak" not in json.dumps((listed, context))


def test_v2_identity_binding_survives_guard_loss_and_reopen(tmp_path):
    root = tmp_path / "v2-tamper"
    hy = HyMem(HyMemConfig(root=root))
    try:
        message_id = hy.log_message(
            "session", "user", "identity-bound needle",
            source_peer_id="alice", source_workspace_id="w",
        )
        session_log.register_session_peer(
            hy.conn, "session", "w", "bob", "user"
        )
        coverage = hy.conn.execute(
            "SELECT chunk_id,coverage_version,hash_version,record_version "
            "FROM message_retention_coverage WHERE message_id=?",
            (message_id,),
        ).fetchone()
        assert (coverage["hash_version"], coverage["record_version"]) == (
            MESSAGE_PROVENANCE_HASH_VERSION,
            MESSAGE_PROVENANCE_RECORD_VERSION,
        )
        for trigger in (
            "message_lossless_source_update_guard",
            "message_retention_coverage_update_guard",
            "message_lossless_stream_update_guard",
            "message_coverage_peer_update_guard",
        ):
            hy.conn.execute(f"DROP TRIGGER {trigger}")
        hy.conn.execute(
            "UPDATE messages SET source_peer_id='bob' WHERE id=?",
            (message_id,),
        )
        hy.conn.execute(
            "UPDATE message_retention_coverage SET source_peer_id='bob' "
            "WHERE message_id=?",
            (message_id,),
        )
    finally:
        hy.close()

    reopened = HyMem(HyMemConfig(root=root))
    try:
        with pytest.raises(RuntimeError, match="coverage proof mismatch"):
            validate_message_coverage_artifact(
                reopened.conn,
                message_id=message_id,
                chunk_id=coverage["chunk_id"],
                coverage_version=coverage["coverage_version"],
            )
        for peer_id in ("alice", "bob"):
            ctx = reopened.augment(
                "identity-bound needle",
                source_peer_id=peer_id,
                source_workspace_id="w",
            )
            assert ctx.message_hits == [] and ctx.graph_facts == []
    finally:
        reopened.close()


def test_durable_search_survives_pruning_metadata_noise_and_workspace_flood(
    tmp_path,
):
    root = tmp_path / "durable-search"
    hy = HyMem(HyMemConfig(root=root))
    try:
        noise = [
            ("user", f"unrelated filler {index}", "2020-01-01T00:00:00Z")
            for index in range(270)
        ]
        turns = [*noise, (
            "user", "my role is cartographer", "2020-01-01T00:00:00Z"
        )]
        hy.log_messages(
            "a-session", turns,
            source_peer_ids=["alice"] * len(turns),
            source_workspace_id="workspace-a",
        )
        before = hy.augment(
            "what is my role",
            source_peer_id="alice",
            source_workspace_id="workspace-a",
        ).message_hits
        assert before and before[0].text == "my role is cartographer"
        assert hy.augment(
            "source_workspace_id",
            source_peer_id="alice",
            source_workspace_id="workspace-a",
        ).message_hits == []

        hy.log_messages(
            "b-session",
            [("user", f"what is my role flood {index}") for index in range(300)],
            source_peer_ids=["bob"] * 300,
            source_workspace_id="workspace-b",
        )
        after = hy.augment(
            "what is my role",
            source_peer_id="alice",
            source_workspace_id="workspace-a",
        ).message_hits
        assert [(hit.message_id, hit.score) for hit in after] == [
            (hit.message_id, hit.score) for hit in before
        ]

        hy.close_session("a-session")
        assert prune_messages(
            hy.conn, replace(hy.config, message_retention_days=1)
        ) == len(turns)
        retained = hy.augment(
            "what is my role",
            source_peer_id="alice",
            source_workspace_id="workspace-a",
        ).message_hits
        assert retained and retained[0].text == "my role is cartographer"
    finally:
        hy.close()


def test_durable_search_pages_past_more_than_256_invalid_proofs(hy):
    turns = [("user", "needle needle needle") for _ in range(270)]
    turns.append(("user", "needle valid target"))
    ids = hy.log_messages(
        "corrupt-crowd", turns,
        source_peer_ids=["alice"] * len(turns),
        source_workspace_id="w",
    )
    hy.conn.execute("DROP TRIGGER message_lossless_stream_update_guard")
    hy.conn.execute("DROP TRIGGER message_coverage_peer_update_guard")
    hy.conn.execute(
        "UPDATE message_retention_coverage SET message_content_hash=? "
        "WHERE message_id < ?",
        ("0" * 64, ids[-1]),
    )
    hits = hy.augment(
        "needle",
        source_peer_id="alice",
        source_workspace_id="w",
    ).message_hits
    assert hits and hits[0].message_id == ids[-1]
    assert hits[0].text == "needle valid target"


def test_coverage_fts_shape_triggers_reopen_and_vacuum_are_healed(tmp_path):
    root = tmp_path / "fts-heal"
    hy = HyMem(HyMemConfig(root=root))
    try:
        hy.log_message(
            "s", "user", "first durable term",
            source_peer_id="alice", source_workspace_id="w",
        )
        for trigger in (
            "message_coverage_fts_insert",
            "message_coverage_fts_delete",
            "message_coverage_fts_update_delete",
            "message_coverage_fts_update_insert",
        ):
            hy.conn.execute(f"DROP TRIGGER {trigger}")
        hy.log_message(
            "s", "user", "missing posting restored",
            source_peer_id="alice", source_workspace_id="w",
        )
    finally:
        hy.close()

    raw = sqlite3.connect(root / "hymem.sqlite")
    try:
        raw.execute("DROP TABLE message_coverage_fts")
        raw.execute(
            "CREATE VIRTUAL TABLE message_coverage_fts "
            "USING fts5(wrong, content='')"
        )
        raw.commit()
    finally:
        raw.close()

    reopened = HyMem(HyMemConfig(root=root))
    try:
        assert [row["name"] for row in reopened.conn.execute(
            "PRAGMA table_info(message_coverage_fts)"
        )] == ["content"]
        hits = reopened.augment(
            "missing posting restored",
            source_peer_id="alice",
            source_workspace_id="w",
        ).message_hits
        assert hits and hits[0].text == "missing posting restored"
        reopened.conn.execute("VACUUM")
        core_db.resync_rowid_shadows(reopened.conn)
        hits = reopened.augment(
            "first durable term",
            source_peer_id="alice",
            source_workspace_id="w",
        ).message_hits
        assert hits and hits[0].text == "first durable term"
    finally:
        reopened.close()


def test_scoped_graph_uses_only_local_authoritative_evidence(hy):
    _publish_claim(
        hy, session_id="alice-positive", peer_id="alice",
        content="shared service uses redis", chunk_id="alice-positive",
    )
    baseline = hy.augment(
        "shared service redis", source_peer_id="alice", source_workspace_id="w"
    ).graph_facts
    assert len(baseline) == 1
    baseline_metrics = (
        baseline[0].pos_evidence, baseline[0].neg_evidence,
        baseline[0].confidence,
    )
    _publish_claim(
        hy, session_id="bob-negative", peer_id="bob",
        content="shared service does not use redis", chunk_id="bob-negative",
        polarity=-1,
    )
    for index in range(4):
        _publish_claim(
            hy, session_id=f"bob-positive-{index}", peer_id="bob",
            content="shared service uses redis", chunk_id=f"bob-positive-{index}",
        )
    isolated = hy.augment(
        "shared service redis", source_peer_id="alice", source_workspace_id="w"
    ).graph_facts
    assert len(isolated) == 1
    assert (
        isolated[0].pos_evidence, isolated[0].neg_evidence,
        isolated[0].confidence,
    ) == baseline_metrics
    assert isolated[0].score == pytest.approx(baseline[0].score, rel=1e-6)


def test_global_manual_retraction_closes_scoped_graph_view(hy):
    _publish_claim(
        hy, session_id="alice-retracted", peer_id="alice",
        content="shared service uses redis", chunk_id="alice-retracted",
    )
    assert hy.augment(
        "shared service redis", source_peer_id="alice", source_workspace_id="w"
    ).graph_facts
    assert hy.retract_edge("shared_service", "uses", "redis") is True
    assert hy.augment(
        "shared service redis", source_peer_id="alice", source_workspace_id="w"
    ).graph_facts == []


def test_malformed_out_of_scope_lifecycle_event_cannot_suppress_peer(hy):
    _publish_claim(
        hy, session_id="alice-valid", peer_id="alice",
        content="shared service uses redis", chunk_id="alice-valid",
    )
    _publish_claim(
        hy, session_id="bob-corrupt", peer_id="bob",
        content="shared service uses redis", chunk_id="bob-corrupt",
    )
    bob = hy.conn.execute(
        "SELECT id,edge_id FROM kg_evidence WHERE source_peer_id='bob'"
    ).fetchone()
    hy.conn.execute("DROP TRIGGER kg_edge_lifecycle_insert_guard")
    hy.conn.execute(
        "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,direction,"
        "event_at,source_evidence_id,dependency_count) "
        "VALUES (?,?,?,?,?,?,0)",
        (bob["edge_id"], "malformed-bob", "claim_assertion", 1,
         "not-a-time", bob["id"]),
    )
    alice = hy.augment(
        "shared service redis", source_peer_id="alice", source_workspace_id="w"
    ).graph_facts
    assert len(alice) == 1
    assert {citation.source_peer_id for citation in alice[0].citations} == {"alice"}


def test_v8_redaction_rewrites_external_v2_proof_atomically(tmp_path):
    src = HyMem(HyMemConfig(
        root=tmp_path / "redact-src", redact_secrets=False
    ))
    try:
        message_id = src.log_message(
            "redacted", "user", "email alice@example.com",
            source_peer_id="alice", source_workspace_id="w",
        )
        wire = tmp_path / "redacted-v8.jsonl"
        src.export(wire)
    finally:
        src.close()
    assert "alice@example.com" in wire.read_text(encoding="utf-8")
    dst = HyMem(HyMemConfig(root=tmp_path / "redact-dst"))
    try:
        portability.import_jsonl(
            dst.conn, wire, redact_values=True, config=dst.config
        )
        row = dst.conn.execute(
            "SELECT chunk_id,coverage_version,hash_version,record_version "
            "FROM message_retention_coverage WHERE message_id=?",
            (message_id,),
        ).fetchone()
        assert (row["hash_version"], row["record_version"]) == (
            MESSAGE_PROVENANCE_HASH_VERSION,
            MESSAGE_PROVENANCE_RECORD_VERSION,
        )
        proof = validate_message_coverage_artifact(
            dst.conn, message_id=message_id, chunk_id=row["chunk_id"],
            coverage_version=row["coverage_version"],
        )
        assert proof.content == "email [REDACTED-EMAIL]"
        assert (proof.source_peer_id, proof.source_workspace_id) == ("alice", "w")
    finally:
        dst.close()


@pytest.mark.parametrize("malformed_time", [None, 2459999, "not-a-time"])
def test_v8_import_rejects_self_consistent_external_bad_timestamp_atomically(
    tmp_path, malformed_time
):
    label = type(malformed_time).__name__
    src = HyMem(HyMemConfig(root=tmp_path / f"bad-time-src-{label}"))
    try:
        src.log_message(
            "portable-time", "user", "portable timestamp source",
            source_peer_id="alice", source_workspace_id="w",
        )
        wire = tmp_path / f"bad-time-{label}.jsonl"
        src.export(wire)
    finally:
        src.close()

    def corrupt(body: list[dict]) -> None:
        proof = next(
            item["record"] for item in body
            if item["type"] == "message_retention_coverage"
        )
        chunk = next(
            item["record"] for item in body
            if item["type"] == "chunk"
            and item["record"]["id"] == proof["chunk_id"]
        )
        payload = json.loads(chunk["text"])
        payload["source_created_at"] = malformed_time
        record = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        chunk["text"] = record
        proof["source_created_at"] = malformed_time
        proof["message_content_hash"] = hashlib.sha256(
            record.encode("utf-8")
        ).hexdigest()

    _rewrite_wire(wire, corrupt)
    dst = HyMem(HyMemConfig(root=tmp_path / f"bad-time-dst-{label}"))
    try:
        with pytest.raises(ValueError):
            dst.import_(wire)
        assert dst.conn.execute(
            "SELECT COUNT(*) FROM message_retention_coverage"
        ).fetchone()[0] == 0
        assert dst.conn.execute(
            "SELECT COUNT(*) FROM peers"
        ).fetchone()[0] == 0
    finally:
        dst.close()


def test_sql_rejects_v2_record_with_different_relational_peer_after_prune(hy):
    message_id = hy.log_message(
        "sql-bound", "user", "alice durable source",
        created_at="2020-01-01T00:00:00Z",
        source_peer_id="alice", source_workspace_id="w",
    )
    session_log.register_session_peer(
        hy.conn, "sql-bound", "w", "bob", "user"
    )
    hy.close_session("sql-bound")
    assert prune_messages(
        hy.conn, replace(hy.config, message_retention_days=1)
    ) == 1
    original = hy.conn.execute(
        "SELECT * FROM message_retention_coverage WHERE message_id=?",
        (message_id,),
    ).fetchone()
    forged_chunk = "forged-peer-proof"
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
        (
            forged_chunk, "sql-bound", message_id, message_id,
            "forged", hy.conn.execute(
                "SELECT text FROM chunks WHERE id=?", (original["chunk_id"],)
            ).fetchone()[0], "coverage",
        ),
    )
    with pytest.raises(sqlite3.IntegrityError, match="coverage provenance"):
        hy.conn.execute(
            "INSERT INTO message_retention_coverage("
            "message_id,source_session_id,source_role,source_peer_id,"
            "source_workspace_id,source_created_at,chunk_id,"
            "message_content_hash,hash_version,record_version,coverage_version) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                message_id, "sql-bound", "user", "bob", "w",
                original["source_created_at"], forged_chunk,
                original["message_content_hash"], original["hash_version"],
                original["record_version"], "forged-v2",
            ),
        )


def test_malformed_noninteger_coverage_id_never_crashes_sdk_reads(
    provenance_client, hy
):
    assert provenance_client.post(
        "/v3/workspaces/w/sessions/malformed/messages",
        json={"messages": [{"peer_id": "alice", "content": "valid history"}]},
    ).status_code == 201
    hy.conn.execute("DROP TRIGGER message_coverage_peer_insert_guard")
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) "
        "VALUES ('malformed-proof','malformed',1,1,'bad',"
        "'{\"id\":\"bad\",\"content\":\"should not leak\"}','coverage')"
    )
    hy.conn.execute(
        "INSERT INTO message_retention_coverage("
        "message_id,source_session_id,source_role,source_peer_id,"
        "source_workspace_id,chunk_id,message_content_hash,hash_version,"
        "record_version,coverage_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "bad", "malformed", "user", "alice", "w",
            "malformed-proof", "0" * 64, MESSAGE_PROVENANCE_HASH_VERSION,
            MESSAGE_PROVENANCE_RECORD_VERSION, "malformed",
        ),
    )
    listed = provenance_client.post(
        "/v3/workspaces/w/sessions/malformed/messages/list", json={}
    )
    context = provenance_client.get(
        "/v3/workspaces/w/sessions/malformed/context"
    )
    assert listed.status_code == context.status_code == 200
    assert [item["content"] for item in listed.json()["items"]] == [
        "valid history"
    ]
    assert [item["content"] for item in context.json()["messages"]] == [
        "valid history"
    ]


def test_context_tail_exhausts_corrupt_duplicates_at_batch_boundary(
    provenance_client, hy
):
    contents = [f"ordered history {index}" for index in range(25)]
    response = provenance_client.post(
        "/v3/workspaces/w/sessions/duplicate-boundary/messages",
        json={"messages": [
            {"peer_id": "alice", "content": content}
            for content in contents
        ]},
    )
    assert response.status_code == 201
    message_ids = [
        int(item["id"].removeprefix("msg_")) for item in response.json()
    ]
    duplicate_id = message_ids[-10]
    source = hy.conn.execute(
        "SELECT mc.*,c.text FROM message_retention_coverage mc "
        "JOIN chunks c ON c.id=mc.chunk_id WHERE mc.message_id=?",
        (duplicate_id,),
    ).fetchone()
    hy.conn.execute("DROP TRIGGER message_coverage_peer_insert_guard")
    for index in range(45):
        chunk_id = f"corrupt-duplicate-{index:03d}"
        hy.conn.execute(
            "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
            "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
            (
                chunk_id, "duplicate-boundary", duplicate_id, duplicate_id,
                "corrupt duplicate", source["text"], "coverage",
            ),
        )
        hy.conn.execute(
            "INSERT INTO message_retention_coverage("
            "message_id,source_session_id,source_role,source_peer_id,"
            "source_workspace_id,source_created_at,chunk_id,"
            "message_content_hash,hash_version,record_version,"
            "coverage_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                duplicate_id, "duplicate-boundary", "user", "alice", "w",
                source["source_created_at"], chunk_id, "0" * 64,
                source["hash_version"], source["record_version"],
                LOSSLESS_COVERAGE_VERSION,
            ),
        )

    context = provenance_client.get(
        "/v3/workspaces/w/sessions/duplicate-boundary/context?tokens=10000"
    )
    assert context.status_code == 200
    assert [item["content"] for item in context.json()["messages"]] == contents


def test_large_sdk_history_reads_do_not_validate_with_n_plus_one_queries(
    provenance_client, hy
):
    assert provenance_client.post(
        "/v3/workspaces/w/sessions/large/messages",
        json={"messages": [
            {"peer_id": "alice", "content": f"history {index}"}
            for index in range(100)
        ]},
    ).status_code == 201
    for route, payload, maximum in (
        ("/v3/workspaces/w/sessions/large/messages/list", {"size": 100}, 4),
        ("/v3/workspaces/w/sessions/large/context", None, 10),
    ):
        statements: list[str] = []
        hy.conn.set_trace_callback(statements.append)
        try:
            response = (
                provenance_client.post(route, json=payload)
                if payload is not None else provenance_client.get(route)
            )
        finally:
            hy.conn.set_trace_callback(None)
        assert response.status_code == 200
        selects = [
            statement for statement in statements
            if statement.lstrip().upper().startswith("SELECT")
        ]
        assert len(selects) <= maximum


def test_scoped_graph_isolates_same_peer_id_across_workspaces(hy):
    _publish_claim(
        hy, session_id="workspace-a-positive", peer_id="shared-peer",
        workspace_id="workspace-a", content="shared service uses redis",
        chunk_id="workspace-a-positive",
    )
    baseline = hy.augment(
        "shared service redis", source_peer_id="shared-peer",
        source_workspace_id="workspace-a",
    ).graph_facts
    assert len(baseline) == 1
    _publish_claim(
        hy, session_id="workspace-b-negative", peer_id="shared-peer",
        workspace_id="workspace-b",
        content="shared service does not use redis",
        chunk_id="workspace-b-negative", polarity=-1,
    )
    isolated = hy.augment(
        "shared service redis", source_peer_id="shared-peer",
        source_workspace_id="workspace-a",
    ).graph_facts
    assert len(isolated) == 1
    assert isolated[0].pos_evidence == baseline[0].pos_evidence
    assert isolated[0].neg_evidence == 0
    assert {
        citation.source_workspace_id for citation in isolated[0].citations
    } == {"workspace-a"}


def test_scoped_graph_excludes_evidence_without_current_publication_authority(hy):
    _publish_claim(
        hy, session_id="publication", peer_id="alice",
        content="shared service uses redis", chunk_id="publication",
    )
    evidence_id = hy.conn.execute(
        "SELECT id FROM kg_evidence WHERE source_peer_id='alice'"
    ).fetchone()[0]
    assert hy.augment(
        "shared service redis", source_peer_id="alice", source_workspace_id="w"
    ).graph_facts
    with core_db.evidence_history_mutation(hy.conn):
        hy.conn.execute(
            "DELETE FROM kg_claim_observations WHERE evidence_id=?",
            (evidence_id,),
        )
    assert hy.augment(
        "shared service redis", source_peer_id="alice", source_workspace_id="w"
    ).graph_facts == []


def test_scoped_graph_top_k_tie_break_is_independent_of_edge_ids(tmp_path):
    winners = []
    for label, subjects in (
        ("forward", ("alpha_service", "beta_service")),
        ("reverse", ("beta_service", "alpha_service")),
    ):
        local = HyMem(replace(
            HyMemConfig(root=tmp_path / label), graph_top_k=1
        ))
        try:
            for index, subject in enumerate(subjects):
                _publish_claim(
                    local,
                    session_id=f"{label}-{index}",
                    peer_id="alice",
                    content=f"{subject} uses redis",
                    chunk_id=f"{label}-chunk-{index}",
                    subject=subject,
                    created_at="2025-01-01T00:00:00Z",
                )
            facts = local.augment(
                "redis", source_peer_id="alice", source_workspace_id="w"
            ).graph_facts
            assert len(facts) == 1
            winners.append(facts[0].subject)
        finally:
            local.close()
    assert winners == ["alpha_service", "alpha_service"]


def test_foreign_entity_anchor_cannot_change_scoped_fallback(hy):
    _publish_claim(
        hy, session_id="local-fallback", peer_id="alice", workspace_id="a",
        content="local service uses redis", chunk_id="local-fallback",
        subject="local_service",
    )
    before_ctx = hy.augment(
        "zephyr", source_peer_id="alice", source_workspace_id="a"
    )
    before = [
        (fact.subject, fact.predicate, fact.object, fact.score)
        for fact in before_ctx.graph_facts
    ]
    assert before and before_ctx.matched_entities == []

    _publish_claim(
        hy, session_id="foreign-anchor", peer_id="bob", workspace_id="b",
        content="zephyr uses postgres", chunk_id="foreign-anchor",
        subject="zephyr", object_="postgres",
    )
    after_ctx = hy.augment(
        "zephyr", source_peer_id="alice", source_workspace_id="a"
    )
    after = [
        (fact.subject, fact.predicate, fact.object, fact.score)
        for fact in after_ctx.graph_facts
    ]
    assert [item[:3] for item in after] == [item[:3] for item in before]
    assert [item[3] for item in after] == pytest.approx(
        [item[3] for item in before], rel=1e-6
    )
    assert after_ctx.matched_entities == []


def test_scoped_entity_match_uses_in_scope_source_surface_form(hy):
    hy.register_alias("Blue Comet", "local_service")
    _publish_claim(
        hy, session_id="surface-local", peer_id="alice", workspace_id="a",
        content="Blue Comet uses redis", chunk_id="surface-local",
        subject="Blue Comet",
    )
    # Retrieval must be justified by the retained in-scope evidence surface,
    # not continued trust in the global alias table.
    hy.conn.execute("DELETE FROM entity_aliases WHERE alias='blue_comet'")
    ctx = hy.augment(
        "tell me about Blue Comet",
        source_peer_id="alice", source_workspace_id="a",
    )
    assert ctx.matched_entities == ["local_service"]
    fact = next(fact for fact in ctx.graph_facts if fact.subject == "local_service")
    assert "entity_match" in fact.why_retrieved


def test_scoped_entity_match_preserves_non_latin_source_surface(hy):
    # The historical global canonicalizer is ASCII-only, so register the
    # extraction-time mapping once. Retrieval must subsequently prove the
    # Unicode surface from the retained scoped evidence, not the alias table.
    hy.register_alias("東京", "tokyo_city")
    _publish_claim(
        hy, session_id="surface-unicode", peer_id="alice", workspace_id="a",
        content="東京 uses redis", chunk_id="surface-unicode", subject="東京",
    )
    hy.conn.execute("DELETE FROM entity_aliases WHERE canonical='tokyo_city'")

    ctx = hy.augment(
        "tell me about 東京", source_peer_id="alice", source_workspace_id="a"
    )
    assert ctx.matched_entities == ["tokyo_city"]
    fact = next(fact for fact in ctx.graph_facts if fact.subject == "tokyo_city")
    assert "entity_match" in fact.why_retrieved


@pytest.mark.parametrize("invalid_limit", [True, float("nan"), float("inf")])
def test_invalid_graph_top_k_disables_scoped_lookup(hy, invalid_limit):
    _publish_claim(
        hy, session_id="invalid-cap", peer_id="alice", workspace_id="a",
        content="local service uses redis", chunk_id="invalid-cap",
        subject="local_service",
    )
    cfg = replace(hy.config, graph_top_k=invalid_limit)
    assert _graph_lookup(
        hy.conn, cfg, "local service", ["local_service"], {}, frozenset(),
        source_peer_id="alice", source_workspace_id="a",
    ) == []


def test_scoped_semantic_recall_beats_more_than_top_k_recency_distractors(hy):
    _publish_claim(
        hy, session_id="semantic-target", peer_id="alice", workspace_id="a",
        content="the archival constellation choice", chunk_id="semantic-target",
        subject="semantic_target", object_="answer",
        created_at="2020-01-01T00:00:00Z",
    )
    for index in range(12):
        _publish_claim(
            hy, session_id=f"recent-{index}", peer_id="alice", workspace_id="a",
            content=f"recent unrelated item {index}", chunk_id=f"recent-{index}",
            subject=f"recent_{index}", object_=f"noise_{index}",
            created_at="2025-01-01T00:00:00Z",
        )
    edge = hy.conn.execute(
        "SELECT id,subject_canonical,predicate,object_canonical "
        "FROM knowledge_graph WHERE subject_canonical='semantic_target'"
    ).fetchone()
    edge_text = f"{edge['subject_canonical']} {edge['predicate']} {edge['object_canonical']}"
    hy.conn.execute(
        "INSERT INTO edge_embeddings(edge_text,vector_json,model,dim) "
        "VALUES (?,?,?,2)",
        (edge_text, json.dumps([1.0, 0.0]), "scoped-mapping-v1"),
    )
    hy.set_embedding_client(_ScopedMappingEmbedder())
    ctx = hy.augment(
        "constellation memory", source_peer_id="alice", source_workspace_id="a"
    )
    assert ctx.graph_facts[0].subject == "semantic_target"
    assert "fallback:semantic" in ctx.graph_facts[0].why_retrieved


def test_scoped_multihop_ignores_foreign_topology(hy):
    _publish_claim(
        hy, session_id="chain-one", peer_id="alice", workspace_id="a",
        content="anchor is part of middle", chunk_id="chain-one",
        subject="anchor", predicate="part_of", object_="middle",
    )
    _publish_claim(
        hy, session_id="chain-two", peer_id="alice", workspace_id="a",
        content="middle deploys to target", chunk_id="chain-two",
        subject="middle", predicate="deploys_to", object_="target",
    )
    graph_cfg = replace(
        hy.config,
        graph_multihop_enabled=True,
        graph_multihop_min_score=0.01,
    )

    def scoped():
        return _graph_lookup(
            hy.conn, graph_cfg, "tell me about anchor", ["anchor"], {},
            frozenset(), overlap_info={}, source_peer_id="alice",
            source_workspace_id="a",
        )

    before = {
        (fact.subject, fact.predicate, fact.object): fact.score
        for fact in scoped()
    }
    assert ("middle", "deploys_to", "target") in before
    _publish_claim(
        hy, session_id="foreign-chain", peer_id="bob", workspace_id="b",
        content="middle deploys to foreign noise", chunk_id="foreign-chain",
        subject="middle", predicate="deploys_to", object_="foreign_noise",
    )
    after = {
        (fact.subject, fact.predicate, fact.object): fact.score
        for fact in scoped()
    }
    assert set(after) == set(before)
    for triple in before:
        assert after[triple] == pytest.approx(before[triple], rel=1e-6)


@pytest.mark.parametrize("mismatch", ["content", "created_at"])
def test_sql_coverage_insert_binds_live_raw_content_and_time(hy, mismatch):
    session_id = f"raw-insert-{mismatch}"
    session_log.register_session_peer(
        hy.conn, session_id, "w", "alice", "user"
    )
    source_created_at = "2025-01-01T00:00:00.000Z"
    message_id = session_log.append_message(
        hy.conn,
        session_id,
        "user",
        "authoritative content",
        created_at=source_created_at,
        source_peer_id="alice",
        source_workspace_id="w",
    )
    artifact_content = (
        "forged content" if mismatch == "content" else "authoritative content"
    )
    artifact_created_at = (
        "2025-01-02T00:00:00.000Z"
        if mismatch == "created_at" else source_created_at
    )
    record, digest, hash_version, record_version = canonical_message_record(
        message_id=message_id,
        session_id=session_id,
        role="user",
        content=artifact_content,
        source_created_at=artifact_created_at,
        source_peer_id="alice",
        source_workspace_id="w",
    )
    chunk_id = coverage_chunk_id(session_id, message_id)
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
        (
            chunk_id, session_id, message_id, message_id,
            "forged ordered proof", record, "coverage",
        ),
    )
    with pytest.raises(sqlite3.IntegrityError, match="coverage provenance"):
        hy.conn.execute(
            "INSERT INTO message_retention_coverage("
            "message_id,source_session_id,source_role,source_peer_id,"
            "source_workspace_id,source_created_at,chunk_id,"
            "message_content_hash,hash_version,record_version,coverage_version) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                message_id, session_id, "user", "alice", "w",
                artifact_created_at, chunk_id, digest, hash_version,
                record_version, LOSSLESS_COVERAGE_VERSION,
            ),
        )


@pytest.mark.parametrize("mismatch", ["content", "created_at"])
def test_sql_coverage_update_binds_live_raw_content_and_time(hy, mismatch):
    session_id = f"raw-update-{mismatch}"
    source_created_at = "2025-01-01T00:00:00.000Z"
    message_id = hy.log_message(
        session_id,
        "user",
        "authoritative content",
        created_at=source_created_at,
        source_peer_id="alice",
        source_workspace_id="w",
    )
    original = hy.conn.execute(
        "SELECT * FROM message_retention_coverage WHERE message_id=?",
        (message_id,),
    ).fetchone()
    source_text = hy.conn.execute(
        "SELECT text FROM chunks WHERE id=?", (original["chunk_id"],)
    ).fetchone()[0]
    valid_chunk_id = f"generic-valid-{mismatch}"
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
        (
            valid_chunk_id, session_id, message_id, message_id,
            "valid generic proof", source_text, "coverage",
        ),
    )
    hy.conn.execute(
        "INSERT INTO message_retention_coverage("
        "message_id,source_session_id,source_role,source_peer_id,"
        "source_workspace_id,source_created_at,chunk_id,"
        "message_content_hash,hash_version,record_version,coverage_version) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            message_id, session_id, "user", "alice", "w",
            source_created_at, valid_chunk_id,
            original["message_content_hash"], original["hash_version"],
            original["record_version"], f"generic-{mismatch}",
        ),
    )
    artifact_content = (
        "forged content" if mismatch == "content" else "authoritative content"
    )
    artifact_created_at = (
        "2025-01-02T00:00:00.000Z"
        if mismatch == "created_at" else source_created_at
    )
    record, digest, hash_version, record_version = canonical_message_record(
        message_id=message_id,
        session_id=session_id,
        role="user",
        content=artifact_content,
        source_created_at=artifact_created_at,
        source_peer_id="alice",
        source_workspace_id="w",
    )
    forged_chunk_id = f"generic-forged-{mismatch}"
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
        (
            forged_chunk_id, session_id, message_id, message_id,
            "forged generic proof", record, "coverage",
        ),
    )
    with pytest.raises(sqlite3.IntegrityError, match="coverage provenance"):
        hy.conn.execute(
            "UPDATE message_retention_coverage SET chunk_id=?,"
            "source_created_at=?,message_content_hash=?,hash_version=?,"
            "record_version=? WHERE message_id=? AND chunk_id=? "
            "AND coverage_version=?",
            (
                forged_chunk_id, artifact_created_at, digest, hash_version,
                record_version, message_id, valid_chunk_id,
                f"generic-{mismatch}",
            ),
        )


def test_workspace_binding_rejects_orphaned_evidence_dependency(hy):
    hy.open_session("binding-victim")
    decoy_message_id = hy.log_message("binding-decoy", "user", "decoy")
    decoy_chunk_id = hy.conn.execute(
        "SELECT chunk_id FROM message_retention_coverage WHERE message_id=?",
        (decoy_message_id,),
    ).fetchone()[0]
    edge_id = hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical,predicate,"
        "object_canonical) VALUES ('binding','uses','guard')"
    ).lastrowid

    # Simulate a stale/damaged historical store whose old evidence guard was
    # absent: the source session has no transitive chunk or coverage row.
    hy.conn.execute("DROP TRIGGER kg_evidence_v40_insert_guard")
    hy.conn.execute(
        "INSERT INTO kg_evidence(edge_id,chunk_id,polarity,source_session_id,"
        "provenance_status) VALUES (?,?,?,?,?)",
        (
            edge_id, decoy_chunk_id, 1, "binding-victim",
            "legacy_unattributed",
        ),
    )
    with pytest.raises(ValueError, match="cannot infer workspace ownership"):
        session_log.open_session(
            hy.conn, "binding-victim", source_workspace_id="w"
        )
    with pytest.raises(sqlite3.IntegrityError, match="binding is immutable"):
        hy.conn.execute(
            "UPDATE sessions SET source_workspace_id='w' "
            "WHERE id='binding-victim'"
        )


def test_reopen_heals_live_raw_coverage_tuple_guard(tmp_path):
    root = tmp_path / "raw-tuple-heal"
    first = HyMem(HyMemConfig(root=root))
    source_created_at = "2025-01-01T00:00:00.000Z"
    try:
        session_log.register_session_peer(
            first.conn, "healed", "w", "alice", "user"
        )
        message_id = session_log.append_message(
            first.conn,
            "healed",
            "user",
            "authoritative content",
            created_at=source_created_at,
            source_peer_id="alice",
            source_workspace_id="w",
        )
        first.conn.execute("DROP TRIGGER message_coverage_peer_insert_guard")
    finally:
        first.close()

    reopened = HyMem(HyMemConfig(root=root))
    try:
        record, digest, hash_version, record_version = canonical_message_record(
            message_id=message_id,
            session_id="healed",
            role="user",
            content="forged after reopen",
            source_created_at=source_created_at,
            source_peer_id="alice",
            source_workspace_id="w",
        )
        chunk_id = coverage_chunk_id("healed", message_id)
        reopened.conn.execute(
            "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
            "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
            (
                chunk_id, "healed", message_id, message_id,
                "forged after reopen", record, "coverage",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="coverage provenance"):
            reopened.conn.execute(
                "INSERT INTO message_retention_coverage("
                "message_id,source_session_id,source_role,source_peer_id,"
                "source_workspace_id,source_created_at,chunk_id,"
                "message_content_hash,hash_version,record_version,"
                "coverage_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    message_id, "healed", "user", "alice", "w",
                    source_created_at, chunk_id, digest, hash_version,
                    record_version, LOSSLESS_COVERAGE_VERSION,
                ),
            )
    finally:
        reopened.close()


@pytest.mark.parametrize("range_field", ["start_message_id", "end_message_id"])
@pytest.mark.parametrize("malformed", [float("inf"), "not-an-integer"])
def test_malformed_coverage_ranges_fail_closed_on_all_session_routes(
    provenance_client, hy, range_field, malformed
):
    session_id = f"bad-range-{range_field}-{type(malformed).__name__}"
    response = provenance_client.post(
        f"/v3/workspaces/w/sessions/{session_id}/messages",
        json={"messages": [{
            "peer_id": "alice", "content": "range integrity needle",
        }]},
    )
    assert response.status_code == 201
    message_id = int(response.json()[0]["id"].removeprefix("msg_"))
    chunk_id = hy.conn.execute(
        "SELECT chunk_id FROM message_retention_coverage WHERE message_id=?",
        (message_id,),
    ).fetchone()[0]
    hy.conn.execute("DROP TRIGGER message_retention_covered_chunk_update_guard")
    hy.conn.execute(
        f"UPDATE chunks SET {range_field}=? WHERE id=?",
        (malformed, chunk_id),
    )
    stored = hy.conn.execute(
        f"SELECT {range_field} FROM chunks WHERE id=?", (chunk_id,)
    ).fetchone()[0]
    if isinstance(malformed, float):
        assert stored == float("inf"), "SQLite must preserve this corruption"
    else:
        assert stored == malformed

    listed = provenance_client.post(
        f"/v3/workspaces/w/sessions/{session_id}/messages/list", json={}
    )
    context = provenance_client.get(
        f"/v3/workspaces/w/sessions/{session_id}/context"
    )
    searched = provenance_client.post(
        f"/v3/workspaces/w/sessions/{session_id}/search",
        json={"query": "range integrity needle", "limit": 10},
    )
    assert listed.status_code == context.status_code == searched.status_code == 200
    assert listed.json()["items"] == []
    assert context.json()["messages"] == []
    assert searched.json() == []


@pytest.mark.parametrize("malformed_time", [2459999, "not-a-time"])
def test_malformed_v2_source_time_fails_closed_on_all_session_routes(
    provenance_client, hy, malformed_time
):
    session_id = f"bad-v2-time-{type(malformed_time).__name__}"
    session_log.register_session_peer(
        hy.conn, session_id, "w", "alice", "user"
    )
    message_id = 5000
    record, digest = _unsafe_v2_record(
        message_id=message_id,
        session_id=session_id,
        content="timestamp integrity needle",
        source_created_at=malformed_time,
    )
    chunk_id = coverage_chunk_id(session_id, message_id)
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
        (
            chunk_id, session_id, message_id, message_id,
            "malformed v2 time", record, "coverage",
        ),
    )
    hy.conn.execute("DROP TRIGGER message_coverage_peer_insert_guard")
    hy.conn.execute(
        "INSERT INTO message_retention_coverage("
        "message_id,source_session_id,source_role,source_peer_id,"
        "source_workspace_id,source_created_at,chunk_id,message_content_hash,"
        "hash_version,record_version,coverage_version) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            message_id, session_id, "user", "alice", "w", malformed_time,
            chunk_id, digest, MESSAGE_PROVENANCE_HASH_VERSION,
            MESSAGE_PROVENANCE_RECORD_VERSION,
            LOSSLESS_COVERAGE_VERSION,
        ),
    )

    listed = provenance_client.post(
        f"/v3/workspaces/w/sessions/{session_id}/messages/list", json={}
    )
    context = provenance_client.get(
        f"/v3/workspaces/w/sessions/{session_id}/context"
    )
    searched = provenance_client.post(
        f"/v3/workspaces/w/sessions/{session_id}/search",
        json={"query": "timestamp integrity needle", "limit": 10},
    )
    assert listed.status_code == context.status_code == searched.status_code == 200
    assert listed.json()["items"] == []
    assert context.json()["messages"] == []
    assert searched.json() == []


@pytest.mark.parametrize("malformed_time", [None, 2459999, "not-a-time"])
def test_sql_guards_reject_malformed_v2_timestamp_insert_and_update(
    hy, malformed_time
):
    session_id = f"v2-time-guard-{type(malformed_time).__name__}"
    session_log.register_session_peer(
        hy.conn, session_id, "w", "alice", "user"
    )
    valid_time = "2025-01-01T00:00:00.000Z"

    # Seed one proof-only generic row so the UPDATE guard is exercised without
    # relying on a mutable raw message.
    valid_id = 6000
    valid_record, valid_hash, hash_version, record_version = (
        canonical_message_record(
            message_id=valid_id,
            session_id=session_id,
            role="user",
            content="valid proof-only source",
            source_created_at=valid_time,
            source_peer_id="alice",
            source_workspace_id="w",
        )
    )
    valid_chunk = f"valid-v2-time-{type(malformed_time).__name__}"
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
        (
            valid_chunk, session_id, valid_id, valid_id,
            "valid proof-only source", valid_record, "coverage",
        ),
    )
    hy.conn.execute(
        "INSERT INTO message_retention_coverage("
        "message_id,source_session_id,source_role,source_peer_id,"
        "source_workspace_id,source_created_at,chunk_id,message_content_hash,"
        "hash_version,record_version,coverage_version) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            valid_id, session_id, "user", "alice", "w", valid_time,
            valid_chunk, valid_hash, hash_version, record_version,
            "generic-v2-time",
        ),
    )

    insert_id = 6001
    insert_record, insert_hash = _unsafe_v2_record(
        message_id=insert_id,
        session_id=session_id,
        content="malformed insert source",
        source_created_at=malformed_time,
    )
    insert_chunk = coverage_chunk_id(session_id, insert_id)
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
        (
            insert_chunk, session_id, insert_id, insert_id,
            "malformed timestamp insert", insert_record, "coverage",
        ),
    )
    assert hy.conn.execute(
        "SELECT hymem_message_record_proof_valid(?,?,?,?)",
        (
            insert_record, insert_hash, MESSAGE_PROVENANCE_HASH_VERSION,
            MESSAGE_PROVENANCE_RECORD_VERSION,
        ),
    ).fetchone()[0] == 0
    assert hy.conn.execute(
        "SELECT hymem_message_record_matches_source(?,?,?,?,?,?,?,?,?,?)",
        (
            insert_record, insert_id, session_id, "user", malformed_time,
            "alice", "w", insert_hash, MESSAGE_PROVENANCE_HASH_VERSION,
            MESSAGE_PROVENANCE_RECORD_VERSION,
        ),
    ).fetchone()[0] == 0
    with pytest.raises(sqlite3.IntegrityError, match="coverage provenance"):
        hy.conn.execute(
            "INSERT INTO message_retention_coverage("
            "message_id,source_session_id,source_role,source_peer_id,"
            "source_workspace_id,source_created_at,chunk_id,"
            "message_content_hash,hash_version,record_version,"
            "coverage_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                insert_id, session_id, "user", "alice", "w",
                malformed_time, insert_chunk, insert_hash,
                MESSAGE_PROVENANCE_HASH_VERSION,
                MESSAGE_PROVENANCE_RECORD_VERSION,
                LOSSLESS_COVERAGE_VERSION,
            ),
        )

    update_record, update_hash = _unsafe_v2_record(
        message_id=valid_id,
        session_id=session_id,
        content="malformed update source",
        source_created_at=malformed_time,
    )
    update_chunk = f"invalid-v2-time-{type(malformed_time).__name__}"
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
        (
            update_chunk, session_id, valid_id, valid_id,
            "malformed timestamp update", update_record, "coverage",
        ),
    )
    # Raw absence activates the older lifecycle guard first. Remove only that
    # guard to prove the v43 tuple guard rejects the malformed UPDATE itself.
    hy.conn.execute("DROP TRIGGER message_retention_coverage_update_guard")
    with pytest.raises(sqlite3.IntegrityError, match="coverage provenance"):
        hy.conn.execute(
            "UPDATE message_retention_coverage SET chunk_id=?,"
            "source_created_at=?,message_content_hash=?,hash_version=?,"
            "record_version=? WHERE message_id=? AND chunk_id=? "
            "AND coverage_version='generic-v2-time'",
            (
                update_chunk, malformed_time, update_hash,
                MESSAGE_PROVENANCE_HASH_VERSION,
                MESSAGE_PROVENANCE_RECORD_VERSION, valid_id, valid_chunk,
            ),
        )


def test_reopen_heals_v2_timestamp_write_guard(tmp_path):
    root = tmp_path / "v2-time-heal"
    first = HyMem(HyMemConfig(root=root))
    try:
        session_log.register_session_peer(
            first.conn, "healed-v2-time", "w", "alice", "user"
        )
        first.conn.execute("DROP TRIGGER message_coverage_peer_insert_guard")
    finally:
        first.close()

    reopened = HyMem(HyMemConfig(root=root))
    try:
        message_id = 7000
        record, digest = _unsafe_v2_record(
            message_id=message_id,
            session_id="healed-v2-time",
            content="malformed healed timestamp",
            source_created_at="not-a-time",
        )
        chunk_id = coverage_chunk_id("healed-v2-time", message_id)
        reopened.conn.execute(
            "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
            "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,?)",
            (
                chunk_id, "healed-v2-time", message_id, message_id,
                "malformed healed timestamp", record, "coverage",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="coverage provenance"):
            reopened.conn.execute(
                "INSERT INTO message_retention_coverage("
                "message_id,source_session_id,source_role,source_peer_id,"
                "source_workspace_id,source_created_at,chunk_id,"
                "message_content_hash,hash_version,record_version,"
                "coverage_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    message_id, "healed-v2-time", "user", "alice", "w",
                    "not-a-time", chunk_id, digest,
                    MESSAGE_PROVENANCE_HASH_VERSION,
                    MESSAGE_PROVENANCE_RECORD_VERSION,
                    LOSSLESS_COVERAGE_VERSION,
                ),
            )
    finally:
        reopened.close()
