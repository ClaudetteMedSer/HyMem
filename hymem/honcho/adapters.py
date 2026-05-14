"""Response shaping and request-shape normalization for the Honcho API.

This module is the single place that knows "what shape the honcho-ai SDK
expects." An SDK version bump should be a change here, not a grep across every
route handler.
"""
from __future__ import annotations

import datetime
import re
from typing import Any

# ── peer role inference ──────────────────────────────────────────────────────

_USER_RE = re.compile(r"(^user[-_]|human|client|telegram|discord|slack)", re.I)
_AGENT_RE = re.compile(r"(agent|hermes|assistant|ai[-_]|bot|llm)", re.I)


def infer_role(peer_id: str) -> str:
    """Best-effort user/assistant classification from a peer id string."""
    if _USER_RE.search(peer_id):
        return "user"
    if _AGENT_RE.search(peer_id):
        return "assistant"
    return "user"


# ── timestamps ───────────────────────────────────────────────────────────────

def now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── response builders ────────────────────────────────────────────────────────

def msg(
    msg_id: int | str,
    content: str,
    peer_id: str,
    session_id: str,
    workspace_id: str,
    metadata: dict | None = None,
    created_at: str | None = None,
) -> dict:
    """A MessageResponse-shaped dict."""
    return {
        "id": f"msg_{msg_id}",
        "content": content,
        "peer_id": peer_id,
        "session_id": session_id,
        "workspace_id": workspace_id,
        "metadata": metadata or {},
        "created_at": created_at or now(),
        "token_count": max(1, len(content.split())),
    }


def workspace_response(
    workspace_id: str,
    metadata: dict | None = None,
    created_at: str | None = None,
) -> dict:
    return {
        "id": workspace_id,
        "metadata": metadata or {},
        "configuration": {},
        "created_at": created_at or now(),
    }


def peer_response(
    peer_id: str,
    workspace_id: str,
    metadata: dict | None = None,
    configuration: dict | None = None,
    created_at: str | None = None,
) -> dict:
    return {
        "id": peer_id,
        "workspace_id": workspace_id,
        "created_at": created_at or now(),
        "metadata": metadata or {},
        "configuration": configuration or {},
    }


def session_response(
    session_id: str,
    workspace_id: str,
    *,
    is_active: bool = True,
    metadata: dict | None = None,
    configuration: dict | None = None,
    created_at: str | None = None,
) -> dict:
    return {
        "id": session_id,
        "is_active": is_active,
        "workspace_id": workspace_id,
        "metadata": metadata or {},
        "configuration": configuration or {},
        "created_at": created_at or now(),
    }


def peer_card_response(peer_id: str, workspace_id: str, content: str) -> dict:
    return {
        "id": peer_id,
        "workspace_id": workspace_id,
        "content": content,
        "updated_at": now(),
    }


def summary_obj(
    content: str, summary_type: str, message_id: str = "summary_mem"
) -> dict:
    """A SessionContext.summary-shaped dict."""
    return {
        "content": content,
        "message_id": message_id,
        "summary_type": summary_type,
        "created_at": now(),
        "token_count": max(1, len(content.split())),
    }


# ── request-shape normalization ──────────────────────────────────────────────

def parse_add_peers(body: dict[str, Any]) -> list[tuple[str, dict]]:
    """Normalize the two add_peers body shapes to ``(peer_id, metadata)`` pairs.

    Shape 1 (envelope):  {"peers": [{"id": "...", "metadata": {...}}, ...]}
    Shape 2 (bare map):  {peer_id: {"observe_me": bool, "observe_others": bool}}

    The returned metadata dict is exactly what gets persisted and echoed back.
    """
    pairs: list[tuple[str, dict]] = []
    if "peers" in body:
        for entry in body["peers"]:
            pairs.append((entry["id"], entry.get("metadata") or {}))
    else:
        for peer_id, config in body.items():
            cfg = config if isinstance(config, dict) else {}
            pairs.append((
                peer_id,
                {
                    "observe_me": cfg.get("observe_me", True),
                    "observe_others": cfg.get("observe_others", True),
                },
            ))
    return pairs
