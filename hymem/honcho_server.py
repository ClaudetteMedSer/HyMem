"""Honcho v3-compatible HTTP server for HyMem.

Hermes Agent uses the honcho-ai SDK to call add_messages, search, and context
on every conversational turn. By pointing the SDK at this server via
HONCHO_BASE_URL, Hermes automatically captures all messages and retrieves
structured context — zero LLM discretion, no system-prompt instructions required.

Endpoint mapping:
  POST .../sessions/{sid}/messages    → log turns + background dream
  POST .../sessions/{sid}/search      → hy.augment() as Message objects
  GET  .../sessions/{sid}/context     → MEMORY.md + USER.md + recent turns
  POST .../sessions/{sid}/peers       → register peer → role mapping
  GET  .../peers/{pid}/card           → USER.md behavioral profile
  POST .../peers/{pid}/chat           → dialectic Q&A via hy.augment()

Configuration (environment variables):
  HYMEM_ROOT               Storage directory (default: ~/.hermes)
  HYMEM_LLM_API_KEY        Extraction LLM API key  (or DEEPSEEK_API_KEY)
  HYMEM_LLM_BASE_URL       Extraction LLM endpoint (default: https://api.deepseek.com)
  HYMEM_LLM_MODEL          Extraction model        (default: deepseek-chat)
  HYMEM_EMBEDDING_API_KEY  Embeddings API key (falls back to LLM key)
  HYMEM_EMBEDDING_BASE_URL Embedding endpoint (default: https://api.deepseek.com)
  HYMEM_EMBEDDING_MODEL    Embedding model (default: deepseek-embedding)
  HYMEM_HONCHO_HOST        Bind address (default: 127.0.0.1)
  HYMEM_HONCHO_PORT        Port (default: 8765)
  HYMEM_DREAM_COOLDOWN_SECONDS
                           Min seconds between background dream kicks (default: 60)

If the embedding client can't be constructed (e.g. no API key) the server logs
a warning and falls back to FTS-only retrieval. Everything else still works.

Hermes config:
  Set HONCHO_BASE_URL=http://127.0.0.1:8765  (or via honcho.json base_url)
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    import uvicorn
    from fastapi import BackgroundTasks, FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install 'hymem[server]'") from exc

from hymem import HyMem, HyMemConfig
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
from hymem.core import db as core_db

log = logging.getLogger(__name__)

# ── singleton ─────────────────────────────────────────────────────────────────

def _build() -> HyMem:
    root = Path(os.environ.get("HYMEM_ROOT", Path.home() / ".hermes"))

    embedder = None
    try:
        embedder = OpenAICompatibleEmbeddingClient()
    except Exception as e:
        log.info("embeddings disabled (FTS-only retrieval): %s", e)

    return HyMem(
        HyMemConfig(root=root),
        llm=OpenAICompatibleClient(),
        embedding_client=embedder,
    )


_hy_instance: HyMem | None = None


def _get_hy() -> HyMem:
    global _hy_instance
    if _hy_instance is None:
        _hy_instance = _build()
    return _hy_instance


def set_hy(instance: HyMem) -> None:
    """Test/integration helper: inject a pre-built HyMem instance."""
    global _hy_instance
    _hy_instance = instance

# ── Pydantic request / response models ───────────────────────────────────────

class MessageCreate(BaseModel):
    content: str
    peer_id: str
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    created_at: str | None = None


class AddMessagesRequest(BaseModel):
    messages: list[MessageCreate]


class SearchRequest(BaseModel):
    query: str
    filters: dict[str, Any] | None = None
    limit: int = 10


class PeerEntry(BaseModel):
    id: str
    metadata: dict[str, Any] | None = None


class AddPeersRequest(BaseModel):
    peers: list[PeerEntry]


class ChatRequest(BaseModel):
    queries: list[str]
    session_id: str | None = None
    stream: bool = False


# ── peer → role inference ─────────────────────────────────────────────────────

_USER_RE  = re.compile(r"(^user[-_]|human|client|telegram|discord|slack)", re.I)
_AGENT_RE = re.compile(r"(agent|hermes|assistant|ai[-_]|bot|llm)",         re.I)


def _infer_role(peer_id: str) -> str:
    if _USER_RE.search(peer_id):
        return "user"
    if _AGENT_RE.search(peer_id):
        return "assistant"
    return "user"


def _resolve_role(workspace_id: str, peer_id: str) -> str:
    row = _get_hy().conn.execute(
        "SELECT role FROM peers WHERE id = ? AND workspace_id = ?",
        (peer_id, workspace_id),
    ).fetchone()
    return row["role"] if row else _infer_role(peer_id)


# ── shared helpers ────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _msg(
    msg_id: int | str,
    content: str,
    peer_id: str,
    session_id: str,
    workspace_id: str,
    metadata: dict | None = None,
    created_at: str | None = None,
) -> dict:
    return {
        "id": f"msg_{msg_id}",
        "content": content,
        "peer_id": peer_id,
        "session_id": session_id,
        "workspace_id": workspace_id,
        "metadata": metadata or {},
        "created_at": created_at or _now(),
        "token_count": max(1, len(content.split())),
    }


def _background_dream() -> None:
    try:
        _get_hy().dream()
    except Exception:
        log.exception("background dreaming failed")


_last_dream_kick: float = 0.0
_DREAM_COOLDOWN_SECONDS = float(os.environ.get("HYMEM_DREAM_COOLDOWN_SECONDS", "60"))


def _kick_dream_if_due() -> bool:
    """Return True at most once per cooldown window. Updates timestamp on True."""
    global _last_dream_kick
    now = time.monotonic()
    if now - _last_dream_kick >= _DREAM_COOLDOWN_SECONDS:
        _last_dream_kick = now
        return True
    return False


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="HyMem Honcho-compatible server", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": "hymem"}


# ── messages ──────────────────────────────────────────────────────────────────

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
    hy.open_session(session_id)
    responses: list[dict] = []
    for m in body.messages:
        role   = _resolve_role(workspace_id, m.peer_id)
        msg_id = hy.log_message(session_id, role, m.content)
        responses.append(_msg(msg_id, m.content, m.peer_id, session_id, workspace_id,
                               m.metadata, m.created_at))
    # Dreaming is non-blocking; the run-lock prevents concurrent cycles.
    if _kick_dream_if_due():
        background_tasks.add_task(_background_dream)
    return responses


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
        results.append(_msg(
            f"kg_{fact.subject}_{fact.predicate}_{fact.object}",
            content, "hymem-kg", session_id, workspace_id,
            {"type": "graph_fact", "+": fact.pos_evidence, "-": fact.neg_evidence},
        ))

    for hit in ctx.fts_hits:
        if len(results) >= body.limit:
            break
        results.append(_msg(
            f"fts_{hit.chunk_id}",
            hit.text[:600], "hymem-fts", hit.session_id, workspace_id,
            {"type": "fts_hit"},
        ))

    return results


# ── context ───────────────────────────────────────────────────────────────────

@app.get("/v3/workspaces/{workspace_id}/sessions/{session_id}/context")
def get_context(
    workspace_id: str,
    session_id: str,
    summary: bool = True,
    tokens: int = 1000,
) -> dict:
    hy = _get_hy()
    cfg = hy.config
    memory_text = cfg.memory_md_path.read_text(encoding="utf-8") if cfg.memory_md_path.exists() else ""
    user_text   = cfg.user_md_path.read_text(encoding="utf-8")   if cfg.user_md_path.exists()   else ""

    rows = hy.conn.execute(
        "SELECT id, role, content, created_at FROM messages "
        "WHERE session_id = ? ORDER BY id DESC LIMIT 20",
        (session_id,),
    ).fetchall()
    recent = [
        _msg(r["id"], r["content"], r["role"], session_id, workspace_id,
             created_at=r["created_at"])
        for r in reversed(rows)
    ]

    return {
        "summary": memory_text,
        "messages": recent,
        "peers": [{"id": "user", "representation": user_text, "card": user_text}],
    }


# ── peers ─────────────────────────────────────────────────────────────────────

@app.post(
    "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers",
    status_code=201,
)
def add_peers(workspace_id: str, session_id: str, body: AddPeersRequest) -> list[dict]:
    hy = _get_hy()
    responses: list[dict] = []
    for peer in body.peers:
        role = _infer_role(peer.id)
        hy.conn.execute(
            "INSERT OR REPLACE INTO peers(id, workspace_id, role, metadata) VALUES (?, ?, ?, ?)",
            (peer.id, workspace_id, role, json.dumps(peer.metadata or {})),
        )
        responses.append({"id": peer.id, "workspace_id": workspace_id, "metadata": peer.metadata or {}})
    return responses


@app.get("/v3/workspaces/{workspace_id}/peers/{peer_id}/card")
def get_peer_card(workspace_id: str, peer_id: str) -> dict:
    cfg     = _get_hy().config
    content = cfg.user_md_path.read_text(encoding="utf-8") if cfg.user_md_path.exists() else ""
    return {"id": peer_id, "workspace_id": workspace_id, "content": content, "updated_at": _now()}


# ── dialectic (honcho_reasoning) ─────────────────────────────────────────────

@app.post("/v3/workspaces/{workspace_id}/peers/{peer_id}/chat")
def peer_chat(workspace_id: str, peer_id: str, body: ChatRequest) -> dict:
    query = " ".join(body.queries)
    ctx   = _get_hy().augment(query)
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
    return {"response": answer, "queries": body.queries}


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    host = os.environ.get("HYMEM_HONCHO_HOST", "127.0.0.1")
    port = int(os.environ.get("HYMEM_HONCHO_PORT", "8765"))
    uvicorn.run("hymem.honcho_server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
