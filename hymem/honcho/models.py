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

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class _Permissive(BaseModel):
    """Base model that tolerates unknown fields from newer SDK versions."""

    model_config = ConfigDict(extra="allow")


class MessageCreate(_Permissive):
    content: str
    peer_id: str = Field(min_length=1, max_length=4096)
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    created_at: str | None = None


class AddMessagesRequest(_Permissive):
    messages: list[MessageCreate] = Field(min_length=1, max_length=1000)


class SearchRequest(_Permissive):
    query: str = Field(min_length=1, max_length=10_000)
    filters: dict[str, Any] | None = None
    limit: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def _validate_query(self) -> "SearchRequest":
        if not self.query.strip():
            raise ValueError("query must be non-empty")
        return self


class WorkspaceCreateRequest(_Permissive):
    id: str
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None


class PeerConfigurationRequest(_Permissive):
    observe_me: StrictBool | None = None


class PeerCreateRequest(_Permissive):
    id: str = Field(min_length=1, max_length=4096)
    metadata: dict[str, Any] | None = None
    configuration: PeerConfigurationRequest | None = None

    @model_validator(mode="after")
    def _validate_id(self) -> "PeerCreateRequest":
        if not self.id.strip():
            raise ValueError("id must be non-empty")
        return self


class SessionCreateRequest(_Permissive):
    id: str = Field(min_length=1, max_length=4096)
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    # Current honcho-ai sends ``peers``; older clients used ``peer_names``.
    # Keep both explicit so permissive-extra handling cannot silently discard
    # the canonical form.
    peers: dict[str, dict[str, Any] | None] | None = None
    peer_names: dict[str, dict[str, Any] | None] | None = None

    @model_validator(mode="after")
    def _validate_id(self) -> "SessionCreateRequest":
        if not self.id.strip():
            raise ValueError("id must be non-empty")
        return self


class MessageListRequest(_Permissive):
    # gt=0 on size: a 0 would cause a divide-by-zero in page-count math.
    page: int = Field(default=1, ge=1)
    size: int = Field(default=50, gt=0)


class RepresentationRequest(_Permissive):
    """SDK representation *read* request.

    Despite using POST, ``Peer.representation()`` and
    ``Session.representation()`` do not mutate a peer.  Keeping this shape
    separate from chat prevents a future permissive-extra field from being
    mistaken for writable profile state again.
    """

    session_id: str | None = Field(default=None, max_length=4096)
    target: str | None = Field(default=None, max_length=4096)
    search_query: str | None = Field(default=None, max_length=10_000)
    search_top_k: int | None = Field(default=None, ge=1, le=100)
    search_max_distance: float | None = Field(default=None, ge=0.0, le=1.0)
    include_most_frequent: StrictBool | None = None
    max_conclusions: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def _validate_ids_and_query(self) -> "RepresentationRequest":
        for field_name in ("session_id", "target"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.search_query is not None and not self.search_query.strip():
            raise ValueError("search_query must be non-empty")
        return self


class ChatRequest(_Permissive):
    # ``query`` is the v2/v3 SDK form. ``queries`` is the v1 compatibility
    # envelope and remains supported, but both are validated as real queries.
    query: str | None = Field(default=None, max_length=10_000)
    queries: list[Annotated[str, Field(min_length=1, max_length=10_000)]] | None = Field(
        default=None, max_length=10
    )
    target: str | None = Field(default=None, max_length=4096)
    session_id: str | None = Field(default=None, max_length=4096)
    reasoning_level: Literal["minimal", "low", "medium", "high", "max"] | None = None
    stream: StrictBool = False

    @model_validator(mode="after")
    def _validate_chat(self) -> "ChatRequest":
        if self.query is not None and not self.query.strip():
            raise ValueError("query must be non-empty")
        if self.queries is not None:
            if not self.queries:
                raise ValueError("queries must not be empty")
            if any(not query.strip() for query in self.queries):
                raise ValueError("queries must contain non-empty strings")
        if self.query is None and self.queries is None:
            raise ValueError("query or queries is required")
        if self.query is not None and self.queries is not None:
            raise ValueError("query and queries are mutually exclusive")
        if self.stream and self.queries is not None and len(self.queries) != 1:
            raise ValueError("streaming accepts exactly one query")
        for field_name in ("target", "session_id"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        return self
