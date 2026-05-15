"""FastAPI routes for the Honcho v3-compatible HTTP server.

Hermes uses the honcho-ai SDK to call add_messages, search, and context on
every conversational turn. Pointing the SDK at this server via HONCHO_BASE_URL
captures all messages and retrieves structured context with zero LLM
discretion.

Endpoint mapping (high level):
  POST .../sessions/{sid}/messages    → log turns + background dream
  POST .../sessions/{sid}/search      → hy.augment() as Message objects
  GET  .../sessions/{sid}/context     → MEMORY.md + USER.md + recent turns
  POST .../sessions/{sid}/peers       → register peer → role mapping
  GET  .../peers/{pid}/card           → USER.md behavioral profile
  POST .../peers/{pid}/chat           → dialectic Q&A via hy.augment()

Configuration is entirely environment-driven — see hymem.bootstrap and
`hymem-doctor`. Server-specific variables:
  HYMEM_HONCHO_HOST            Bind address (default: 127.0.0.1)
  HYMEM_HONCHO_PORT            Port (default: 8765)
  HYMEM_DREAM_COOLDOWN_SECONDS Min seconds between background dream kicks (60)
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any

try:
    import uvicorn
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install 'hymem[server]'") from exc

# Startup, env-var resolution, and the shared singleton live in hymem.bootstrap.
from hymem.bootstrap import get_instance as _get_hy, set_instance as set_hy
from hymem.core import db as core_db
from hymem.honcho import adapters
from hymem.honcho.adapters import infer_role, msg, now
from hymem.honcho.models import (
    AddMessagesRequest,
    ChatRequest,
    MessageListRequest,
    PeerCreateRequest,
    RepresentationUpdateRequest,
    SearchRequest,
    SessionCreateRequest,
    WorkspaceCreateRequest,
)

log = logging.getLogger(__name__)

__all__ = ["app", "main", "set_hy"]


# ── role resolution ──────────────────────────────────────────────────────────

def _resolve_role(workspace_id: str, peer_id: str) -> str:
    """Role from the peers table if registered, else inferred from the id."""
    row = _get_hy().conn.execute(
        "SELECT role FROM peers WHERE id = ? AND workspace_id = ?",
        (peer_id, workspace_id),
    ).fetchone()
    return row["role"] if row else infer_role(peer_id)


# ── background dreaming ──────────────────────────────────────────────────────

def _background_dream() -> None:
    """Run a dream cycle on a forked HyMem instance (separate SQLite
    connection) to avoid write-transaction collisions with concurrent
    add_messages calls. Reuses the live instance's LLM/embedding clients."""
    try:
        start = time.monotonic()
        dream_hy = _get_hy().fork()
        try:
            dream_hy.dream()
        finally:
            dream_hy.close()
        log.info("background_dream completed in %.1fs", time.monotonic() - start)
    except Exception:
        log.exception("background dreaming failed")


_last_dream_kick: float = 0.0
_DREAM_COOLDOWN_SECONDS = float(os.environ.get("HYMEM_DREAM_COOLDOWN_SECONDS", "60"))


def _kick_dream_if_due() -> bool:
    """Return True at most once per cooldown window. Updates timestamp on True."""
    global _last_dream_kick
    current = time.monotonic()
    if current - _last_dream_kick >= _DREAM_COOLDOWN_SECONDS:
        _last_dream_kick = current
        return True
    return False


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="HyMem Honcho-compatible server", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": "hymem"}


# ── workspace (get-or-create) ────────────────────────────────────────────────

@app.post("/v3/workspaces", status_code=201)
def create_workspace(body: WorkspaceCreateRequest) -> dict:
    """Get-or-create a workspace. The SDK calls this once per client instance."""
    hy = _get_hy()
    with core_db.transaction(hy.conn):
        hy.conn.execute(
            "INSERT OR IGNORE INTO sessions(id, started_at) VALUES (?, ?)",
            (f"ws:{body.id}", now()),
        )
    return adapters.workspace_response(body.id, body.metadata)


@app.get("/v3/workspaces/{workspace_id}")
def get_workspace(workspace_id: str) -> dict:
    return adapters.workspace_response(workspace_id)


@app.get("/v3/workspaces/{workspace_id}/conflicts")
def list_conflicts(workspace_id: str) -> dict:
    """Surface knowledge-graph contradictions for the workspace.

    Wraps `hy.conflicts()` — pure SQL, no LLM call. Two kinds are reported:
    `competing_object` (same subject+predicate, different objects under a
    mutually-exclusive predicate) and `opposing_predicate` (same subject+object
    joined by an opposing predicate pair like prefers/rejects).
    """
    conflicts = _get_hy().conflicts()
    return {
        "workspace_id": workspace_id,
        "conflicts": [
            {
                "kind": c.kind,
                "subject": c.subject,
                "edge_a": list(c.edge_a),
                "edge_b": list(c.edge_b),
                "confidence_a": c.confidence_a,
                "confidence_b": c.confidence_b,
                "detail": c.detail,
            }
            for c in conflicts
        ],
    }


# ── peers (get-or-create) ────────────────────────────────────────────────────

@app.post("/v3/workspaces/{workspace_id}/peers", status_code=201)
def create_peer(workspace_id: str, body: PeerCreateRequest) -> dict:
    """Get-or-create a peer. Called by client.peer(id)."""
    hy = _get_hy()
    metadata = body.metadata or {}
    with core_db.transaction(hy.conn):
        hy.conn.execute(
            "INSERT OR IGNORE INTO peers(id, workspace_id, role, metadata, registered_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (body.id, workspace_id, infer_role(body.id), json.dumps(metadata), now()),
        )
    return adapters.peer_response(
        body.id, workspace_id, metadata=metadata, configuration=body.configuration or {}
    )


@app.get("/v3/workspaces/{workspace_id}/peers/{peer_id}")
def get_peer(workspace_id: str, peer_id: str) -> dict:
    hy = _get_hy()
    row = hy.conn.execute(
        "SELECT id, workspace_id, metadata FROM peers WHERE id = ? AND workspace_id = ?",
        (peer_id, workspace_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Peer not found")
    metadata: dict[str, Any] = {}
    if row["metadata"]:
        try:
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError:
            pass
    return adapters.peer_response(row["id"], row["workspace_id"], metadata=metadata)


# ── sessions (get-or-create) ─────────────────────────────────────────────────

@app.post("/v3/workspaces/{workspace_id}/sessions", status_code=201)
def create_session(workspace_id: str, body: SessionCreateRequest) -> dict:
    """Get-or-create a session. Called by client.session(id)."""
    hy = _get_hy()
    peer_names = body.peer_names
    with core_db.transaction(hy.conn):
        hy.conn.execute(
            "INSERT OR IGNORE INTO sessions(id, started_at) VALUES (?, ?)",
            (body.id, now()),
        )
        if peer_names and isinstance(peer_names, dict):
            for peer_name in peer_names:
                hy.conn.execute(
                    "INSERT OR IGNORE INTO peers(id, workspace_id, role, metadata, registered_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (peer_name, workspace_id, infer_role(peer_name),
                     json.dumps({}), now()),
                )
    return adapters.session_response(
        body.id, workspace_id,
        metadata=body.metadata or {}, configuration=body.configuration or {},
    )


@app.get("/v3/workspaces/{workspace_id}/sessions/{session_id}")
def get_session(workspace_id: str, session_id: str) -> dict:
    hy = _get_hy()
    row = hy.conn.execute(
        "SELECT started_at, ended_at FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return adapters.session_response(
        session_id, workspace_id,
        is_active=row["ended_at"] is None, created_at=row["started_at"],
    )


# ── messages ─────────────────────────────────────────────────────────────────

@app.post(
    "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages",
    status_code=201,
)
def add_messages(
    workspace_id: str,
    session_id: str,
    body: AddMessagesRequest,
    background_tasks: BackgroundTasks,
) -> list[dict]:
    hy = _get_hy()
    roles = [_resolve_role(workspace_id, m.peer_id) for m in body.messages]
    # One transaction for the whole batch (see HyMem.log_messages).
    msg_ids = hy.log_messages(
        session_id, [(role, m.content) for role, m in zip(roles, body.messages)]
    )
    responses = [
        msg(msg_id, m.content, m.peer_id, session_id, workspace_id,
            m.metadata, m.created_at)
        for msg_id, m in zip(msg_ids, body.messages)
    ]
    # Dreaming is non-blocking; the run-lock prevents concurrent cycles.
    if _kick_dream_if_due():
        background_tasks.add_task(_background_dream)
    return responses


@app.post("/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/upload")
async def upload_file(
    workspace_id: str,
    session_id: str,
    peer_id: str = Form(...),
    file: UploadFile = File(...),
) -> list[dict]:
    """Upload a file as a peer message. Used by migrate_memory_files()."""
    hy = _get_hy()
    hy.open_session(session_id)
    content = (await file.read()).decode("utf-8", errors="replace")
    msg_id = hy.log_message(session_id, infer_role(peer_id), content)
    return [msg(msg_id, content, peer_id, session_id, workspace_id)]


@app.post("/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/list")
def list_messages(
    workspace_id: str,
    session_id: str,
    body: MessageListRequest | None = None,
) -> dict:
    body = body or MessageListRequest()
    hy = _get_hy()
    page = max(1, body.page)
    size = body.size

    total_row = hy.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    total = total_row[0] if total_row else 0

    offset = (page - 1) * size
    rows = hy.conn.execute(
        "SELECT id, content, role, session_id, created_at FROM messages "
        "WHERE session_id = ? ORDER BY id LIMIT ? OFFSET ?",
        (session_id, size, offset),
    ).fetchall()

    items = [
        msg(row["id"], row["content"], row["role"], row["session_id"], workspace_id,
            created_at=row["created_at"])
        for row in rows
    ]
    pages = math.ceil(total / size) if total > 0 else 0
    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages,
    }


@app.post("/v3/workspaces/{workspace_id}/sessions/{session_id}/search")
def search_messages(workspace_id: str, session_id: str, body: SearchRequest) -> list[dict]:
    ctx = _get_hy().augment(body.query)
    results: list[dict] = []

    for fact in ctx.graph_facts:
        if len(results) >= body.limit:
            break
        content = (
            f"{fact.subject} {fact.predicate} {fact.object} "
            f"(confidence: {fact.confidence:.2f})"
        )
        results.append(msg(
            f"kg_{fact.subject}_{fact.predicate}_{fact.object}",
            content, "hymem-kg", session_id, workspace_id,
            {
                "type": "graph_fact",
                "+": fact.pos_evidence,
                "-": fact.neg_evidence,
                "why": list(fact.why_retrieved),
            },
        ))

    for hit in ctx.fts_hits:
        if len(results) >= body.limit:
            break
        results.append(msg(
            f"fts_{hit.chunk_id}",
            hit.text[:600], "hymem-fts", hit.session_id, workspace_id,
            {"type": "fts_hit"},
        ))

    return results


# ── context ──────────────────────────────────────────────────────────────────

@app.get("/v3/workspaces/{workspace_id}/sessions/{session_id}/context")
def get_context(
    workspace_id: str,
    session_id: str,
    summary: bool = True,
    tokens: int = 1000,
) -> dict:
    hy = _get_hy()
    cfg = hy.config
    memory_text = (
        cfg.memory_md_path.read_text(encoding="utf-8")
        if cfg.memory_md_path.exists() else ""
    )
    user_text = (
        cfg.user_md_path.read_text(encoding="utf-8")
        if cfg.user_md_path.exists() else ""
    )

    session_row = hy.conn.execute(
        "SELECT summary FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    session_summary = (
        session_row["summary"] if session_row and session_row["summary"] else ""
    )

    rows = hy.conn.execute(
        "SELECT id, role, content, created_at FROM messages "
        "WHERE session_id = ? ORDER BY id DESC LIMIT 20",
        (session_id,),
    ).fetchall()
    messages = [
        msg(r["id"], r["content"], r["role"], session_id, workspace_id,
            created_at=r["created_at"])
        for r in reversed(rows)
    ]

    summary_obj = None
    if summary and (session_summary or memory_text.strip()):
        if session_summary:
            summary_obj = adapters.summary_obj(session_summary, "session")
        else:
            summary_obj = adapters.summary_obj(memory_text, "memory")

    # HyMem models peers as roles (the messages table has no peer_id column),
    # so peer "ids" here are the distinct roles present in the session.
    peer_roles = list(dict.fromkeys(r["role"] for r in rows))
    return {
        "summary": summary_obj,
        "messages": messages,
        "peer_representation": user_text,
        "peers": [{"id": role} for role in peer_roles],
    }


# ── session peers ────────────────────────────────────────────────────────────

@app.post(
    "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers",
    status_code=201,
)
def add_peers(workspace_id: str, session_id: str, body: dict[str, Any]) -> list[dict]:
    """Add peers to a session. Accepts the SDK's two body shapes — see
    adapters.parse_add_peers."""
    hy = _get_hy()
    responses: list[dict] = []
    with core_db.transaction(hy.conn):
        for peer_id, metadata in adapters.parse_add_peers(body):
            hy.conn.execute(
                "INSERT OR REPLACE INTO peers(id, workspace_id, role, metadata) "
                "VALUES (?, ?, ?, ?)",
                (peer_id, workspace_id, infer_role(peer_id), json.dumps(metadata)),
            )
            responses.append({
                "id": peer_id,
                "workspace_id": workspace_id,
                "metadata": metadata,
            })
    return responses


@app.get("/v3/workspaces/{workspace_id}/sessions/{session_id}/peers/{peer_id}/config")
def get_peer_config(workspace_id: str, session_id: str, peer_id: str) -> dict:
    """Get per-session peer configuration."""
    hy = _get_hy()
    row = hy.conn.execute(
        "SELECT metadata FROM peers WHERE id = ? AND workspace_id = ?",
        (peer_id, workspace_id),
    ).fetchone()
    if row and row["metadata"]:
        try:
            meta = json.loads(row["metadata"])
            return {
                "observe_me": meta.get("observe_me", True),
                "observe_others": meta.get("observe_others", True),
            }
        except json.JSONDecodeError:
            pass
    return {"observe_me": True, "observe_others": True}


# ── peers (workspace-scoped) ─────────────────────────────────────────────────

@app.get("/v3/workspaces/{workspace_id}/peers/{peer_id}/card")
def get_peer_card(workspace_id: str, peer_id: str) -> dict:
    cfg = _get_hy().config
    content = (
        cfg.user_md_path.read_text(encoding="utf-8")
        if cfg.user_md_path.exists() else ""
    )
    return adapters.peer_card_response(peer_id, workspace_id, content)


@app.get("/v3/workspaces/{workspace_id}/peers/{peer_id}/context")
def get_peer_context(
    workspace_id: str,
    peer_id: str,
    target: str | None = None,
    search_query: str | None = None,
    limit_to_session: bool = False,
    summary: bool = True,
) -> dict:
    """Peer-scoped context with optional semantic search.

    Called by the Honcho SDK's search() and context() methods. When
    search_query is present, runs augment() and returns scored results.
    """
    hy = _get_hy()
    cfg = hy.config

    peer_representation = ""
    if cfg.user_md_path.exists():
        peer_representation = cfg.user_md_path.read_text(encoding="utf-8")

    messages = []
    if search_query:
        ctx = hy.augment(search_query)
        for hit in ctx.fts_hits:
            # chunk ids are text ("c0"); resolve role via the chunk's first
            # message rather than treating the chunk id as a message id.
            row = hy.conn.execute(
                "SELECT m.role FROM chunks c JOIN messages m "
                "ON m.id = c.start_message_id WHERE c.id = ?",
                (hit.chunk_id,),
            ).fetchone()
            role = row["role"] if row else "assistant"
            messages.append(msg(
                hit.chunk_id, hit.text[:600], role,
                hit.session_id, workspace_id,
                {"type": "fts_hit", "score": getattr(hit, "score", 0.0)},
            ))
        for fact in ctx.graph_facts:
            content = (
                f"{fact.subject} {fact.predicate} {fact.object} "
                f"(confidence: {fact.confidence:.2f})"
            )
            messages.append(msg(
                f"kg_{fact.subject}_{fact.predicate}_{fact.object}",
                content, "hymem-kg", "", workspace_id,
                {"type": "graph_fact", "why": list(fact.why_retrieved)},
            ))

    summary_obj = None
    if summary:
        memory_text = (
            cfg.memory_md_path.read_text(encoding="utf-8")
            if cfg.memory_md_path.exists() else ""
        )
        if memory_text.strip():
            summary_obj = adapters.summary_obj(memory_text, "memory")

    return {
        "summary": summary_obj,
        "messages": messages,
        "peer_representation": peer_representation,
    }


@app.post("/v3/workspaces/{workspace_id}/peers/{peer_id}/representation")
async def update_peer_representation(
    workspace_id: str,
    peer_id: str,
    body: RepresentationUpdateRequest,
) -> dict:
    """Persist a behavioral-profile update to USER.md."""
    cfg = _get_hy().config
    if body.content:
        cfg.user_md_path.write_text(body.content, encoding="utf-8")
    return {"status": "ok", "peer_id": peer_id}


# ── dialectic (honcho_reasoning) ─────────────────────────────────────────────

@app.post("/v3/workspaces/{workspace_id}/peers/{peer_id}/chat")
def peer_chat(workspace_id: str, peer_id: str, body: ChatRequest) -> dict:
    queries = body.queries or [body.query]
    responses: list[str] = []
    facts_per_query: list[list[dict]] = []

    for q in queries:
        if not q.strip():
            responses.append("")
            facts_per_query.append([])
            continue
        ctx = _get_hy().augment(q)
        parts: list[str] = []
        if ctx.graph_facts:
            lines = [
                f"- {f.subject} {f.predicate} {f.object} (conf {f.confidence:.2f})"
                for f in ctx.graph_facts
            ]
            parts.append("From knowledge graph:\n" + "\n".join(lines))
        if ctx.fts_hits:
            snippets = [f"- {h.text[:300]}" for h in ctx.fts_hits]
            parts.append("From conversation history:\n" + "\n".join(snippets))
        answer = "\n\n".join(parts) if parts else "No relevant information found in memory."
        responses.append(answer)
        facts_per_query.append([
            {
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
                "why": list(f.why_retrieved),
            }
            for f in ctx.graph_facts
        ])

    # `facts` is additive metadata for SDK consumers that want the structured
    # why_retrieved trail without parsing prose. Aligns with `response` (first
    # query); `facts_by_query` carries the full per-query breakdown.
    return {
        "response": responses[0],
        "queries": queries,
        "facts": facts_per_query[0] if facts_per_query else [],
        "facts_by_query": facts_per_query,
    }


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    host = os.environ.get("HYMEM_HONCHO_HOST", "127.0.0.1")
    port = int(os.environ.get("HYMEM_HONCHO_PORT", "8765"))
    uvicorn.run("hymem.honcho.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
