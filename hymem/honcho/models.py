"""Typed Pydantic request models for the Honcho-compatible API.

Every endpoint that accepts a JSON body has a model here, so a shape mismatch
with the honcho-ai SDK surfaces as a clean 422 rather than an AttributeError
deep inside a handler. Models are permissive (``extra="allow"``): the SDK
evolves and unknown fields must not break ingestion.

The one deliberate exception is ``add_peers``: the SDK sends two mutually
exclusive top-level shapes — a ``{"peers": [...]}`` envelope, or a bare
``{peer_id: config}`` map. That polymorphism is normalized by
``adapters.parse_add_peers``, not expressed as a model.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Permissive(BaseModel):
    """Base model that tolerates unknown fields from newer SDK versions."""

    model_config = ConfigDict(extra="allow")


class MessageCreate(_Permissive):
    content: str
    peer_id: str
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    created_at: str | None = None


class AddMessagesRequest(_Permissive):
    messages: list[MessageCreate]


class SearchRequest(_Permissive):
    query: str
    filters: dict[str, Any] | None = None
    limit: int = 10


class WorkspaceCreateRequest(_Permissive):
    id: str
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None


class PeerCreateRequest(_Permissive):
    id: str
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None


class SessionCreateRequest(_Permissive):
    id: str
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    peer_names: dict[str, Any] | None = None


class MessageListRequest(_Permissive):
    # gt=0 on size: a 0 would cause a divide-by-zero in page-count math.
    page: int = Field(default=1, ge=1)
    size: int = Field(default=50, gt=0)


class RepresentationUpdateRequest(_Permissive):
    content: str = ""


class ChatRequest(_Permissive):
    query: str = ""
    queries: list[str] | None = None
    target: str | None = None
    session_id: str | None = None
    reasoning_level: str | None = None
    stream: bool = False
