"""FastAPI routes for HyMem's pinned-SDK Honcho HTTP subset.

Hermes uses the honcho-ai SDK to call add_messages, search, and context on
every conversational turn. Pointing the SDK at this server via HONCHO_BASE_URL
captures all messages and retrieves structured context with zero LLM
discretion.

Endpoint mapping (high level):
  POST .../sessions/{sid}/messages    → log turns + background dream
  POST .../sessions/{sid}/search      → hy.augment() as Message objects
  GET  .../sessions/{sid}/context     → scoped summary + exact recent turns
  POST .../sessions/{sid}/peers       → register peer → role mapping
  GET  .../peers/{pid}/card           → peer-owned working representation
  POST .../peers/{pid}/search         → hy.augment() as Message objects (peer-scoped)
  POST .../peers/{pid}/chat           → scoped, bounded dialectic Q&A

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
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

try:
    import uvicorn
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import StreamingResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install 'hymem[server]'") from exc

# Startup, env-var resolution, and the shared singleton live in hymem.bootstrap.
from hymem.bootstrap import get_instance as _get_hy, set_instance as set_hy
from hymem.core import db as core_db
from hymem import session as session_log
from hymem.dreaming.lossless import (
    COVERAGE_VALIDATION_COLUMNS,
    COVERAGE_VALIDATION_JOINS,
    validate_message_coverage_artifact,
    validate_message_coverage_row,
)
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
from hymem.dreaming.scheduler import DreamScheduler
from hymem.honcho import adapters
from hymem.honcho.adapters import infer_role, msg, now
from hymem.honcho.models import (
    AddMessagesRequest,
    ChatRequest,
    MessageListRequest,
    PeerCreateRequest,
    RepresentationRequest,
    SearchRequest,
    SessionCreateRequest,
    WorkspaceCreateRequest,
)
from hymem.query.fusion import estimate_tokens
from hymem.honcho.reasoning import (
    GroundedEvidence,
    reason_iteratively,
    sanitize_evidence_text,
)

log = logging.getLogger(__name__)

__all__ = ["app", "main", "set_hy", "set_scheduler"]


# ── role resolution ──────────────────────────────────────────────────────────

_PEER_CONFIG_METADATA_KEY = "__hymem_honcho_peer_configuration__"
_MAX_OCCURRENCE_SESSIONS = 64
_MAX_SCOPED_OCCURRENCES = 2048
_MAX_SCOPED_AUGMENT_CALLS = 1
_MAX_CONTEXT_OCCURRENCES = 2048
_MAX_QUERY_TOKENS = 4096
_PEER_CARD_MAX_ITEMS = 40


def _decode_peer_state(raw: object) -> tuple[dict[str, Any], dict[str, bool]] | None:
    """Decode portable registry metadata and its private peer policy envelope."""
    if not isinstance(raw, str):
        return None
    try:
        metadata = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(metadata, dict):
        return None
    metadata = dict(metadata)
    raw_config = metadata.pop(_PEER_CONFIG_METADATA_KEY, {})
    if not isinstance(raw_config, dict):
        return None
    observe_me = raw_config.get("observe_me", True)
    if type(observe_me) is not bool:
        return None
    return metadata, {"observe_me": observe_me}


def _encode_peer_state(
    metadata: dict[str, Any], *, observe_me: bool | None = None
) -> str:
    if _PEER_CONFIG_METADATA_KEY in metadata:
        raise ValueError("peer metadata uses a reserved key")
    payload = dict(metadata)
    if observe_me is not None:
        payload[_PEER_CONFIG_METADATA_KEY] = {"observe_me": observe_me}
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _peer_observe_me(workspace_id: str, peer_id: str) -> bool:
    row = _get_hy().conn.execute(
        "SELECT metadata FROM peers WHERE id=? AND workspace_id=?",
        (peer_id, workspace_id),
    ).fetchone()
    state = _decode_peer_state(row["metadata"]) if row is not None else None
    return bool(state is not None and state[1]["observe_me"])

def _peer_row(workspace_id: str, peer_id: str):
    """Resolve one workspace-qualified peer or fail closed."""
    row = _get_hy().conn.execute(
        "SELECT id, workspace_id, role, metadata, registered_at FROM peers "
        "WHERE id = ? AND workspace_id = ?",
        (peer_id, workspace_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Peer not found")
    return row


def _resolve_role(workspace_id: str, peer_id: str) -> str:
    """Use a registered role, or the SDK-compatible default for a new peer.

    ``Session.add_messages`` implicitly adds an unknown author to the session.
    The caller-supplied peer id remains the exact identity; only its coarse
    behavioral role is defaulted, and HyMem registers both atomically with the
    message batch.
    """
    row = _get_hy().conn.execute(
        "SELECT role FROM peers WHERE id = ? AND workspace_id = ?",
        (peer_id, workspace_id),
    ).fetchone()
    return str(row["role"]) if row is not None else infer_role(peer_id)


def _session_row(
    workspace_id: str,
    session_id: str,
    *,
    creating: bool = False,
):
    """Resolve exact session ownership for every workspace-scoped route."""
    row = _get_hy().conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None:
        if creating:
            return None
        raise HTTPException(status_code=404, detail="Session not found")
    if row["source_workspace_id"] != workspace_id:
        raise HTTPException(
            status_code=409 if creating else 404,
            detail=(
                "Session id is already owned by another workspace"
                if creating else "Session not found"
            ),
        )
    return row


@dataclass(frozen=True)
class _DialecticalScope:
    """A resolved Honcho collection key and its authorized session slice.

    Honcho representations are directional ``(observer, observed)`` pairs.
    The target/observed peer owns every returned occurrence; the observer only
    determines which shared sessions may contribute it.
    """

    workspace_id: str
    observer_id: str
    target_id: str
    session_ids: tuple[str, ...]
    explicit_session_id: str | None = None


def _observation_flags(raw: object) -> tuple[bool, bool] | None:
    """Decode ``observe_me``/``observe_others`` without truthy coercion.

    A malformed configuration is ambiguous authorization state.  It is never
    repaired or defaulted on a read; callers either exclude that global slice
    or reject an explicitly requested session.
    """
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    observe_me = value.get("observe_me", True)
    observe_others = value.get("observe_others", True)
    if type(observe_me) is not bool or type(observe_others) is not bool:
        return None
    return observe_me, observe_others


def _validated_session_configuration(value: dict[str, Any] | None) -> dict[str, Any]:
    configuration = dict(value or {})
    for field_name in ("observe_me", "observe_others"):
        if field_name in configuration and type(configuration[field_name]) is not bool:
            raise ValueError(f"{field_name} must be a boolean")
    return configuration


def _membership_flags(
    workspace_id: str, session_id: str, peer_id: str
) -> tuple[bool, bool] | None:
    row = _get_hy().conn.execute(
        "SELECT configuration FROM session_peers WHERE session_id=? "
        "AND workspace_id=? AND peer_id=?",
        (session_id, workspace_id, peer_id),
    ).fetchone()
    return _observation_flags(row["configuration"]) if row is not None else None


def _session_allows_collection(
    workspace_id: str,
    session_id: str,
    observer_id: str,
    target_id: str,
) -> bool:
    if not _peer_observe_me(workspace_id, target_id):
        return False
    target_flags = _membership_flags(workspace_id, session_id, target_id)
    if target_flags is None:
        return False
    if observer_id == target_id:
        # Session observe_me controls whether *other peers* form a theory of
        # mind. It cannot hide a member's own/omniscient collection from them.
        return True
    if not target_flags[0]:
        return False
    observer_flags = _membership_flags(workspace_id, session_id, observer_id)
    return observer_flags is not None and observer_flags[1]


def _resolve_dialectical_scope(
    workspace_id: str,
    observer_id: str,
    *,
    target_id: str | None = None,
    session_id: str | None = None,
    required_session_id: str | None = None,
) -> _DialecticalScope:
    """Resolve one directional peer read or fail closed.

    ``required_session_id`` validates membership in a Session.context route
    even when its representation is intentionally cross-session.  ``session_id``
    is the narrower Honcho collection filter and therefore also required.
    """
    target_id = target_id or observer_id
    _peer_row(workspace_id, observer_id)
    if target_id != observer_id:
        _peer_row(workspace_id, target_id)
    target_observable = _peer_observe_me(workspace_id, target_id)

    required = required_session_id or session_id
    if required is not None:
        _session_row(workspace_id, required)
        if not _session_allows_collection(
            workspace_id, required, observer_id, target_id
        ):
            raise HTTPException(
                status_code=403,
                detail="Peer representation is not authorized in this session",
            )

    if session_id is not None:
        return _DialecticalScope(
            workspace_id, observer_id, target_id, (session_id,), session_id
        )

    # Only sessions with an unambiguous workspace binding and valid membership
    # configuration may contribute to a global directional collection.
    if observer_id == target_id:
        rows = _get_hy().conn.execute(
            "SELECT sp.session_id,sp.configuration FROM session_peers sp "
            "JOIN sessions s ON s.id=sp.session_id "
            "WHERE sp.workspace_id=? AND sp.peer_id=? "
            "AND s.source_workspace_id=? "
            "ORDER BY COALESCE((SELECT MAX(mc.source_created_at) "
            "FROM message_retention_coverage mc "
            "WHERE mc.source_session_id=sp.session_id "
            "AND mc.source_workspace_id=sp.workspace_id),s.started_at) DESC,"
            "sp.session_id DESC LIMIT 2048",
            (workspace_id, target_id, workspace_id),
        ).fetchall()
        session_ids = tuple(
            str(row["session_id"])
            for row in rows
            if (flags := _observation_flags(row["configuration"])) is not None
            and target_observable
        )[:512]
    else:
        rows = _get_hy().conn.execute(
            "SELECT observer.session_id,observer.configuration AS observer_config,"
            "target.configuration AS target_config FROM session_peers observer "
            "JOIN session_peers target ON target.session_id=observer.session_id "
            "AND target.workspace_id=observer.workspace_id "
            "JOIN sessions s ON s.id=observer.session_id "
            "WHERE observer.workspace_id=? AND observer.peer_id=? "
            "AND target.peer_id=? AND s.source_workspace_id=? "
            "ORDER BY COALESCE((SELECT MAX(mc.source_created_at) "
            "FROM message_retention_coverage mc "
            "WHERE mc.source_session_id=observer.session_id "
            "AND mc.source_workspace_id=observer.workspace_id),s.started_at) DESC,"
            "observer.session_id DESC LIMIT 2048",
            (workspace_id, observer_id, target_id, workspace_id),
        ).fetchall()
        allowed: list[str] = []
        for row in rows:
            observer_flags = _observation_flags(row["observer_config"])
            target_flags = _observation_flags(row["target_config"])
            if (
                observer_flags is not None
                and target_flags is not None
                and observer_flags[1]
                and target_flags[0]
                and target_observable
            ):
                allowed.append(str(row["session_id"]))
                if len(allowed) >= 512:
                    break
        session_ids = tuple(allowed)
    return _DialecticalScope(
        workspace_id, observer_id, target_id, session_ids, None
    )


# ── background dreaming ──────────────────────────────────────────────────────

_DREAM_COOLDOWN_SECONDS = float(os.environ.get("HYMEM_DREAM_COOLDOWN_SECONDS", "60"))
_scheduler: DreamScheduler | None = None


def _get_scheduler() -> DreamScheduler:
    """Lazy scheduler init. Lifespan startup primes this; the fallback covers
    test entry points (TestClient) that bypass startup events."""
    global _scheduler
    if _scheduler is None:
        _scheduler = DreamScheduler(_get_hy(), _DREAM_COOLDOWN_SECONDS)
        _scheduler.start()
    return _scheduler


def set_scheduler(scheduler: DreamScheduler | None) -> None:
    """Inject (or clear) the scheduler — used by tests that want to control
    dream timing without spinning up a real thread."""
    global _scheduler
    _scheduler = scheduler


# ── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    _get_scheduler()
    try:
        yield
    finally:
        global _scheduler
        if _scheduler is not None:
            _scheduler.stop()
            _scheduler = None


app = FastAPI(title="HyMem Honcho-compatible server", version="1.0.0", lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": "hymem"}


@app.get("/dream-status")
def dream_status() -> dict:
    """Operator visibility into the re-extraction backlog.

    Not workspace-scoped — dreaming is global to the HyMem instance, like
    `/health`. Wraps `hy.dream_status()` (pure SQL, no LLM) so an operator can
    see how many chunks are pending for the current prompt_version, whether a
    dream is in progress, and the last dream's outcome — making the surge after
    a prompt_version bump transparent rather than mysterious.
    """
    return _get_hy().dream_status()


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
    """Reject a legacy route whose native graph is not workspace partitioned.

    Relabeling ``HyMem.conflicts()`` with a caller-supplied workspace id leaked
    global graph state. A future implementation needs proof-valid per-workspace
    confidence/state projection; until then, failing explicitly is the only
    honest tenant-safe contract.
    """
    raise HTTPException(
        status_code=501,
        detail="workspace-scoped conflict projection is not supported",
    )


# ── peers (get-or-create) ────────────────────────────────────────────────────

@app.post("/v3/workspaces/{workspace_id}/peers", status_code=201)
def create_peer(workspace_id: str, body: PeerCreateRequest) -> dict:
    """Get-or-create a peer. Called by client.peer(id)."""
    hy = _get_hy()
    requested_config = (
        body.configuration.model_dump(exclude_none=True)
        if body.configuration is not None else {}
    )
    try:
        with core_db.transaction(hy.conn):
            existing = hy.conn.execute(
                "SELECT role,metadata,registered_at FROM peers "
                "WHERE id=? AND workspace_id=?",
                (body.id, workspace_id),
            ).fetchone()
            if existing is None:
                metadata = body.metadata or {}
                observe_me = requested_config.get("observe_me")
                registered_at = now()
                hy.conn.execute(
                    "INSERT INTO peers(id,workspace_id,role,metadata,registered_at) "
                    "VALUES (?,?,?,?,?)",
                    (
                        body.id, workspace_id, infer_role(body.id),
                        _encode_peer_state(metadata, observe_me=observe_me),
                        registered_at,
                    ),
                )
                configuration = {"observe_me": (
                    observe_me if observe_me is not None else True
                )}
            else:
                registered_at = existing["registered_at"]
                state = _decode_peer_state(existing["metadata"])
                if state is None:
                    raise ValueError("peer registry metadata is malformed")
                stored_metadata, stored_config = state
                metadata = body.metadata if body.metadata is not None else stored_metadata
                observe_me = requested_config.get(
                    "observe_me", stored_config["observe_me"]
                )
                if body.metadata is not None or body.configuration is not None:
                    hy.conn.execute(
                        "UPDATE peers SET metadata=? WHERE id=? AND workspace_id=?",
                        (
                            _encode_peer_state(metadata, observe_me=observe_me),
                            body.id, workspace_id,
                        ),
                    )
                configuration = {"observe_me": observe_me}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return adapters.peer_response(
        body.id,
        workspace_id,
        metadata=metadata,
        configuration=configuration,
        created_at=registered_at,
    )


@app.get("/v3/workspaces/{workspace_id}/peers/{peer_id}")
def get_peer(workspace_id: str, peer_id: str) -> dict:
    hy = _get_hy()
    row = hy.conn.execute(
        "SELECT id, workspace_id, metadata, registered_at FROM peers "
        "WHERE id = ? AND workspace_id = ?",
        (peer_id, workspace_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Peer not found")
    state = _decode_peer_state(row["metadata"])
    if state is None:
        raise HTTPException(status_code=409, detail="Peer registry is malformed")
    metadata, configuration = state
    return adapters.peer_response(
        row["id"], row["workspace_id"], metadata=metadata,
        configuration=configuration,
        created_at=row["registered_at"],
    )


# ── sessions (get-or-create) ─────────────────────────────────────────────────

@app.post("/v3/workspaces/{workspace_id}/sessions", status_code=201)
def create_session(workspace_id: str, body: SessionCreateRequest) -> dict:
    """Get-or-create a session. Called by client.session(id)."""
    hy = _get_hy()
    existing = _session_row(workspace_id, body.id, creating=True)
    peers: dict[str, Any] = {}
    for peer_map in (body.peer_names, body.peers):
        if isinstance(peer_map, dict):
            peers.update(peer_map)
    try:
        with core_db.transaction(hy.conn):
            if existing is None:
                session_log.open_session(
                    hy.conn, body.id, source_workspace_id=workspace_id
                )
            for peer_id, configuration in peers.items():
                if not isinstance(peer_id, str) or not peer_id.strip():
                    raise ValueError("session peer id must be non-empty")
                if configuration is not None and not isinstance(
                    configuration, dict
                ):
                    raise ValueError("session peer configuration must be an object")
                configuration = _validated_session_configuration(configuration)
                registered = hy.conn.execute(
                    "SELECT role FROM peers WHERE id = ? AND workspace_id = ?",
                    (peer_id, workspace_id),
                ).fetchone()
                role = registered["role"] if registered else infer_role(peer_id)
                session_log.register_session_peer(
                    hy.conn,
                    body.id,
                    workspace_id,
                    peer_id,
                    role,
                    configuration=configuration,
                )
    except (sqlite3.IntegrityError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    persisted_session = _session_row(workspace_id, body.id)
    return adapters.session_response(
        body.id, workspace_id,
        metadata=body.metadata or {}, configuration=body.configuration or {},
        created_at=persisted_session["started_at"],
    )


@app.get("/v3/workspaces/{workspace_id}/sessions/{session_id}")
def get_session(workspace_id: str, session_id: str) -> dict:
    row = _session_row(workspace_id, session_id)
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
) -> list[dict]:
    hy = _get_hy()
    _session_row(workspace_id, session_id, creating=True)
    roles = [_resolve_role(workspace_id, m.peer_id) for m in body.messages]
    # One transaction for the whole batch (see HyMem.log_messages). Pass each
    # message's caller-supplied created_at through so the persisted row carries
    # the real event time, not bulk-ingestion time — chronological/temporal
    # retrieval (ORDER BY created_at) depends on it.
    try:
        msg_ids = hy.log_messages(
            session_id,
            [(role, m.content, m.created_at) for role, m in zip(roles, body.messages)],
            source_peer_ids=[m.peer_id for m in body.messages],
            source_workspace_id=workspace_id,
        )
    except (sqlite3.IntegrityError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    persisted = {
        int(row["id"]): row
        for row in hy.conn.execute(
            "SELECT id,content,created_at FROM messages WHERE id IN ("
            + ",".join("?" for _ in msg_ids)
            + ")",
            msg_ids,
        ).fetchall()
    } if msg_ids else {}
    responses = [
        msg(
            msg_id, persisted[msg_id]["content"], m.peer_id,
            session_id, workspace_id, m.metadata,
            persisted[msg_id]["created_at"],
        )
        for msg_id, m in zip(msg_ids, body.messages)
    ]
    # Fire-and-forget; scheduler owns cooldown and concurrency.
    _get_scheduler().kick()
    return responses


@app.post("/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/upload")
async def upload_file(
    workspace_id: str,
    session_id: str,
    peer_id: str = Form(...),
    created_at: str | None = Form(None),
    file: UploadFile = File(...),
) -> list[dict]:
    """Upload a file as a peer message. Used by migrate_memory_files()."""
    hy = _get_hy()
    _session_row(workspace_id, session_id, creating=True)
    role = _resolve_role(workspace_id, peer_id)
    content = (await file.read()).decode("utf-8", errors="replace")
    try:
        msg_id = hy.log_message(
            session_id,
            role,
            content,
            created_at=created_at,
            source_peer_id=peer_id,
            source_workspace_id=workspace_id,
        )
    except (sqlite3.IntegrityError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    persisted = hy.conn.execute(
        "SELECT content,created_at FROM messages WHERE id=?", (msg_id,)
    ).fetchone()
    return [
        msg(
            msg_id,
            persisted["content"],
            peer_id,
            session_id,
            workspace_id,
            created_at=persisted["created_at"],
        )
    ]


def _validated_session_occurrences(
    workspace_id: str,
    session_id: str,
    *,
    tail_limit: int | None = None,
    source_peer_id: str | None = None,
) -> list:
    """Return the exact durable occurrence stream for one bound session.

    Honcho ingestion publishes this proof atomically with each raw message.
    Reading it as the source of truth preserves SDK history after raw pruning
    and prevents a temporarily unguarded raw-table edit from becoming output.
    """
    hy = _get_hy()
    occurrences = []
    seen: set[int] = set()
    offset = 0
    while True:
        params: list[object] = [
            session_id, workspace_id, LOSSLESS_COVERAGE_VERSION,
        ]
        peer_clause = ""
        if source_peer_id is not None:
            peer_clause = " AND mc.source_peer_id=?"
            params.append(source_peer_id)
        limit_clause = ""
        if tail_limit is not None:
            limit_clause = " LIMIT ? OFFSET ?"
            params.extend((max(40, tail_limit * 2), offset))
        order = "DESC" if tail_limit is not None else "ASC"
        rows = hy.conn.execute(
            f"SELECT {COVERAGE_VALIDATION_COLUMNS} "
            f"FROM message_retention_coverage mc {COVERAGE_VALIDATION_JOINS} "
            "WHERE mc.source_session_id=? AND mc.source_workspace_id=? "
            "AND mc.source_peer_id IS NOT NULL AND mc.coverage_version=?"
            f"{peer_clause}"
            " AND typeof(mc.message_id)='integer'"
            f" ORDER BY hymem_normalize_iso_timestamp(mc.source_created_at) {order}, "
            f"mc.message_id {order}, mc.created_at DESC, "
            "mc.chunk_id, mc.coverage_version"
            f"{limit_clause}",
            params,
        ).fetchall()
        if not rows:
            break
        for row in rows:
            try:
                proof = validate_message_coverage_row(row)
            except (RuntimeError, TypeError, ValueError):
                continue
            message_id = proof.message_id
            if message_id in seen:
                continue
            if (
                proof.session_id != session_id
                or proof.source_workspace_id != workspace_id
                or proof.source_peer_id is None
                or (
                    source_peer_id is not None
                    and proof.source_peer_id != source_peer_id
                )
            ):
                continue
            seen.add(message_id)
            occurrences.append(proof)
            if tail_limit is not None and len(occurrences) >= tail_limit:
                break
        if tail_limit is None or len(occurrences) >= tail_limit:
            break
        offset += len(rows)
        if len(rows) < max(40, tail_limit * 2):
            break
    if tail_limit is not None:
        occurrences.reverse()
    return occurrences


def _validated_message_occurrence(
    *,
    workspace_id: str,
    session_id: str,
    message_id: int,
    peer_id: str,
):
    """Resolve one exact external occurrence through its durable proof.

    Graph retrieval cites claims, not authored Message objects. Honcho search
    can expose such a hit only after resolving the citation back to the exact
    canonical source record; returning claim prose under the cited peer would
    falsely attribute HyMem's derived wording to that author.
    """
    rows = _get_hy().conn.execute(
        f"SELECT {COVERAGE_VALIDATION_COLUMNS} "
        f"FROM message_retention_coverage mc {COVERAGE_VALIDATION_JOINS} "
        "WHERE mc.source_session_id=? AND mc.message_id=? "
        "AND mc.source_workspace_id=? AND mc.source_peer_id=? "
        "AND mc.coverage_version=? "
        "ORDER BY mc.created_at DESC, mc.chunk_id, mc.coverage_version",
        (
            session_id,
            message_id,
            workspace_id,
            peer_id,
            LOSSLESS_COVERAGE_VERSION,
        ),
    ).fetchall()
    for row in rows:
        try:
            proof = validate_message_coverage_row(row)
        except (RuntimeError, TypeError, ValueError):
            continue
        if (
            proof.session_id == session_id
            and proof.message_id == message_id
            and proof.source_workspace_id == workspace_id
            and proof.source_peer_id == peer_id
        ):
            return proof
    return None


def _iter_validated_session_occurrences_desc(
    workspace_id: str,
    session_id: str,
    *,
    page_size: int = 128,
):
    """Yield exact session occurrences newest-first with bounded memory.

    Coverage duplicates are contiguous under the ordering. OFFSET is retained
    deliberately because a hostile run can contain more than one page of
    corrupt duplicates for a single message id; keying only on message id would
    skip the valid proof behind that run.
    """
    hy = _get_hy()
    offset = 0
    yielded_message_id: int | None = None
    while True:
        rows = hy.conn.execute(
            f"SELECT {COVERAGE_VALIDATION_COLUMNS} "
            f"FROM message_retention_coverage mc {COVERAGE_VALIDATION_JOINS} "
            "WHERE mc.source_session_id=? AND mc.source_workspace_id=? "
            "AND mc.source_peer_id IS NOT NULL AND mc.coverage_version=? "
            "AND typeof(mc.message_id)='integer' "
            "ORDER BY hymem_normalize_iso_timestamp(mc.source_created_at) DESC, "
            "mc.message_id DESC, mc.created_at DESC, "
            "mc.chunk_id, mc.coverage_version LIMIT ? OFFSET ?",
            (
                session_id, workspace_id, LOSSLESS_COVERAGE_VERSION,
                max(1, page_size), offset,
            ),
        ).fetchall()
        if not rows:
            return
        for row in rows:
            if row["message_id"] == yielded_message_id:
                continue
            try:
                proof = validate_message_coverage_row(row)
            except (RuntimeError, TypeError, ValueError):
                continue
            if (
                proof.session_id != session_id
                or proof.source_workspace_id != workspace_id
                or proof.source_peer_id is None
            ):
                continue
            yielded_message_id = proof.message_id
            yield proof
        offset += len(rows)
        if len(rows) < max(1, page_size):
            return


@app.post("/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/list")
def list_messages(
    workspace_id: str,
    session_id: str,
    body: MessageListRequest | None = None,
) -> dict:
    body = body or MessageListRequest()
    hy = _get_hy()
    _session_row(workspace_id, session_id)
    page = max(1, body.page)
    size = body.size

    occurrences = _validated_session_occurrences(workspace_id, session_id)
    total = len(occurrences)
    offset = (page - 1) * size
    rows = occurrences[offset:offset + size]

    items = [
        msg(row.message_id, row.content, row.source_peer_id, row.session_id,
            workspace_id, created_at=row.source_created_at)
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


def _augment_messages(
    query: str,
    limit: int,
    workspace_id: str,
    *,
    session_id: str | None = None,
    peer_id: str | None = None,
) -> list[dict]:
    """Shape only exact source occurrences from a pre-ranked scoped query.

    Composite extraction chunks may contain several authors. They remain a
    discovery tier for native HyMem callers, but are never represented as a
    peer-authored Honcho ``Message``. Raw turns and graph citations both carry
    exact workspace-qualified provenance.
    """
    if estimate_tokens(query) > _MAX_QUERY_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=f"query exceeds {_MAX_QUERY_TOKENS} estimated tokens",
        )
    ctx = _get_hy().augment(
        query,
        session_id=session_id,
        source_session_id=session_id,
        source_peer_id=peer_id,
        source_workspace_id=workspace_id,
    )
    results: list[dict] = []
    result_by_occurrence: dict[tuple[str, str, str, str], dict] = {}

    def merge_atomic_facts(records: list[object]) -> list[dict]:
        """Merge duplicates without separating a claim from its citations."""
        merged: dict[tuple[object, object, object, object], dict] = {}
        citation_maps: dict[
            tuple[object, object, object, object], dict[str, dict]
        ] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            key = (
                record.get("edge_id"), record.get("subject"),
                record.get("predicate"), record.get("object"),
            )
            try:
                hash(key)
            except TypeError:
                continue
            if key not in merged:
                merged[key] = {**record, "citations": []}
                citation_maps[key] = {}
            raw_citations = record.get("citations")
            if not isinstance(raw_citations, list):
                continue
            for citation in raw_citations:
                if not isinstance(citation, dict):
                    continue
                citation_key = json.dumps(
                    citation, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), default=str,
                )
                citation_maps[key].setdefault(citation_key, citation)
        for key, record in merged.items():
            record["citations"] = [
                citation_maps[key][citation_key]
                for citation_key in sorted(citation_maps[key])
            ]
        return [merged[key] for key in sorted(merged, key=repr)]

    for fused in ctx.fused_evidence:
        shaped = None
        if fused.tier == "graph":
            shaped = _graph_fact_message(
                fused.payload,
                workspace_id,
                session_id=session_id,
                peer_id=peer_id,
            )
        elif fused.tier in {"message", "count_message"}:
            hit = fused.payload
            if (
                hit.source_workspace_id != workspace_id
                or (peer_id is not None and hit.source_peer_id != peer_id)
                or (session_id is not None and hit.session_id != session_id)
                or hit.source_peer_id is None
            ):
                continue
            shaped = msg(
                hit.message_id, hit.text, hit.source_peer_id,
                hit.session_id, workspace_id,
                created_at=hit.created_at or None,
            )
        if shaped is None:
            continue

        occurrence_key = (
            str(shaped["workspace_id"]),
            str(shaped["session_id"]),
            str(shaped["id"]),
            str(shaped["peer_id"]),
        )

        native = next(
            (entry for entry in fused.provenance if entry.tier == fused.tier),
            fused.provenance[0] if fused.provenance else None,
        )
        retrieval_provenance = [
            {
                "tier": entry.tier,
                "artifact_id": entry.artifact_id,
                "rank": entry.rank,
                "score": entry.raw_score,
                "score_kind": entry.score_kind,
                "why_retrieved": list(entry.why_retrieved),
            }
            for entry in fused.provenance
        ]
        shaped["metadata"] = {
            **shaped.get("metadata", {}),
            "type": shaped.get("metadata", {}).get("type", f"{fused.tier}_hit"),
            "normalized_score": fused.normalized_score,
            "score": native.raw_score if native is not None else None,
            "score_kind": native.score_kind if native is not None else "unknown",
            "why_retrieved": (
                list(native.why_retrieved) if native is not None else []
            ),
            "source_tiers": list(fused.source_tiers),
            "retrieval_provenance": retrieval_provenance,
        }
        graph_claim = shaped["metadata"].pop("graph_claim", None)
        if graph_claim is not None:
            shaped["metadata"]["graph_claims"] = [graph_claim]
        graph_fact = shaped["metadata"].pop("graph_fact", None)
        if graph_fact is not None:
            shaped["metadata"]["graph_facts"] = [graph_fact]

        existing = result_by_occurrence.get(occurrence_key)
        if existing is not None:
            metadata = existing["metadata"]
            metadata["normalized_score"] = max(
                float(metadata.get("normalized_score", 0.0)),
                float(shaped["metadata"].get("normalized_score", 0.0)),
            )
            metadata["source_tiers"] = sorted({
                *metadata.get("source_tiers", []),
                *shaped["metadata"].get("source_tiers", []),
            })
            metadata["why_retrieved"] = sorted({
                *metadata.get("why_retrieved", []),
                *shaped["metadata"].get("why_retrieved", []),
            })
            provenance = {
                (
                    entry.get("tier"), entry.get("artifact_id"),
                    entry.get("rank"), entry.get("score_kind"),
                ): entry
                for entry in metadata.get("retrieval_provenance", [])
            }
            for entry in shaped["metadata"].get("retrieval_provenance", []):
                provenance.setdefault((
                    entry.get("tier"), entry.get("artifact_id"),
                    entry.get("rank"), entry.get("score_kind"),
                ), entry)
            metadata["retrieval_provenance"] = [
                provenance[key] for key in sorted(provenance, key=repr)
            ]
            graph_claims = [
                *metadata.get("graph_claims", []),
                *shaped["metadata"].get("graph_claims", []),
            ]
            if graph_claims:
                unique_claims = {
                    (
                        claim.get("subject"), claim.get("predicate"),
                        claim.get("object"),
                    ): claim
                    for claim in graph_claims
                }
                metadata["graph_claims"] = [
                    unique_claims[key] for key in sorted(unique_claims, key=repr)
                ]
                metadata["type"] = "graph_fact"
            graph_facts = [
                *metadata.get("graph_facts", []),
                *shaped["metadata"].get("graph_facts", []),
            ]
            if graph_facts:
                metadata["graph_facts"] = merge_atomic_facts(graph_facts)
                metadata["type"] = "graph_fact"
            citations = [
                *metadata.get("citations", []),
                *shaped["metadata"].get("citations", []),
            ]
            if citations:
                unique_citations = {
                    (
                        citation.get("evidence_id"),
                        citation.get("source_session_id"),
                        citation.get("source_message_id"),
                        citation.get("source_peer_id"),
                        citation.get("source_workspace_id"),
                    ): citation
                    for citation in citations
                }
                metadata["citations"] = [
                    unique_citations[key]
                    for key in sorted(unique_citations, key=repr)
                ]
            edge_ids = {
                value for value in (
                    metadata.get("edge_id"),
                    shaped["metadata"].get("edge_id"),
                    *metadata.get("edge_ids", []),
                    *shaped["metadata"].get("edge_ids", []),
                )
                if value is not None
            }
            if edge_ids:
                ordered_edge_ids = sorted(edge_ids, key=repr)
                metadata["edge_id"] = ordered_edge_ids[0]
                metadata["edge_ids"] = ordered_edge_ids
            continue

        if len(results) >= limit:
            # Continue scanning the bounded fused list so later alternate
            # representations of an already-selected occurrence can still add
            # provenance, but do not admit a new Message past the item cap.
            continue
        result_by_occurrence[occurrence_key] = shaped
        results.append(shaped)

    return results


def _graph_fact_message(
    fact,
    workspace_id: str,
    *,
    session_id: str | None = None,
    peer_id: str | None = None,
) -> dict | None:
    """Map a fact to the exact authored message behind a valid citation.

    A role is never accepted as an author identity. Citations are filtered to
    the verified request scope before the primary message is chosen, rendered,
    or copied into metadata; facts without such a citation fail closed.
    """
    if fact.edge_id is None or not fact.citations:
        return None
    citations = [
        citation for citation in fact.citations
        if citation.source_workspace_id == workspace_id
        and (peer_id is None or citation.source_peer_id == peer_id)
        and (session_id is None or citation.source_session_id == session_id)
    ]
    if not citations:
        return None
    validated = []
    for citation in citations:
        if (
            citation.source_peer_id is None
            or citation.source_session_id is None
            or citation.source_message_id is None
        ):
            continue
        proof = _validated_message_occurrence(
            workspace_id=workspace_id,
            session_id=citation.source_session_id,
            message_id=citation.source_message_id,
            peer_id=citation.source_peer_id,
        )
        if proof is not None:
            validated.append((citation, proof))
    if not validated:
        return None
    primary, proof = validated[0]
    metadata = {
        "type": "graph_fact",
        "edge_id": fact.edge_id,
        "source_message_id": proof.message_id,
        "source_event_at": primary.source_event_at,
        "graph_claim": {
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
        },
        # Keep edge, claim, and citations atomic. The legacy aliases above and
        # below are useful to direct HTTP consumers but cannot safely be zipped
        # after multiple representations of one occurrence are fused.
        "graph_fact": {
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
            "edge_id": fact.edge_id,
            "citations": [asdict(citation) for citation, _proof in validated],
        },
        "citations": [asdict(citation) for citation, _proof in validated],
    }
    return msg(
        proof.message_id,
        proof.content,
        proof.source_peer_id,
        proof.session_id,
        workspace_id,
        metadata,
        proof.source_created_at,
    )


def _validate_representation_controls(
    *,
    search_top_k: int | None,
    search_max_distance: float | None,
    include_most_frequent: bool | None,
    max_conclusions: int | None,
) -> int:
    """Return the effective item cap and reject controls we cannot honor."""
    if search_max_distance is not None:
        raise HTTPException(
            status_code=422,
            detail="search_max_distance is not supported by this backend",
        )
    if include_most_frequent is True:
        raise HTTPException(
            status_code=422,
            detail="include_most_frequent is not supported by this backend",
        )
    # Honcho's RepresentationParams defaults to 25 conclusions. An omitted
    # search_top_k is not a second implicit 25-item ceiling: callers may widen
    # max_conclusions for an unsearched representation.
    conclusion_limit = max_conclusions or 25
    return (
        min(search_top_k, conclusion_limit)
        if search_top_k is not None
        else conclusion_limit
    )


def _scoped_occurrences(
    scope: _DialecticalScope,
    *,
    per_session_limit: int = 100,
    total_limit: int = _MAX_SCOPED_OCCURRENCES,
) -> list:
    """Read a bounded exact target occurrence window from recent sessions."""
    if total_limit <= 0:
        return []
    occurrences = []
    for session_id in scope.session_ids[:_MAX_OCCURRENCE_SESSIONS]:
        for proof in _validated_session_occurrences(
            scope.workspace_id,
            session_id,
            tail_limit=per_session_limit,
            source_peer_id=scope.target_id,
        ):
            if proof.source_peer_id == scope.target_id:
                occurrences.append(proof)
                if len(occurrences) >= total_limit:
                    break
        if len(occurrences) >= total_limit:
            break
    occurrences.sort(key=lambda proof: (
        proof.source_created_at,
        proof.session_id,
        proof.message_id,
    ))
    return occurrences


def _message_occurrence_key(message: dict) -> tuple[str, str, str, str]:
    return (
        str(message.get("workspace_id", "")),
        str(message.get("session_id", "")),
        str(message.get("id", "")),
        str(message.get("peer_id", "")),
    )


def _validated_fact_citations(
    raw: object,
    *,
    workspace_id: str,
    target_id: str,
    allowed_sessions: set[str],
) -> list[dict]:
    """Keep every independently proof-valid citation in the read scope."""
    if not isinstance(raw, list):
        return []
    validated: dict[tuple[object, ...], dict] = {}
    for citation in raw:
        if not isinstance(citation, dict):
            continue
        source_session_id = citation.get("source_session_id")
        source_message_id = citation.get("source_message_id")
        evidence_id = citation.get("evidence_id")
        if (
            citation.get("source_workspace_id") != workspace_id
            or citation.get("source_peer_id") != target_id
            or not isinstance(source_session_id, str)
            or source_session_id not in allowed_sessions
            or isinstance(source_message_id, bool)
            or not isinstance(source_message_id, int)
            or source_message_id < 1
            or isinstance(evidence_id, bool)
            or not isinstance(evidence_id, int)
            or evidence_id < 1
        ):
            continue
        proof = _validated_message_occurrence(
            workspace_id=workspace_id,
            session_id=source_session_id,
            message_id=source_message_id,
            peer_id=target_id,
        )
        if proof is None:
            continue
        asserted_role = citation.get("source_role")
        asserted_created_at = citation.get("source_created_at")
        asserted_chunk = citation.get("coverage_chunk_id")
        if (
            (asserted_role is not None and asserted_role != proof.role)
            or (
                asserted_created_at is not None
                and asserted_created_at != proof.source_created_at
            )
            or (asserted_chunk is not None and asserted_chunk != proof.chunk_id)
        ):
            continue
        key = (
            evidence_id,
            source_session_id,
            source_message_id,
            target_id,
            workspace_id,
        )
        validated.setdefault(key, dict(citation))
    return [validated[key] for key in sorted(validated, key=repr)]


def _authorize_scoped_message(
    message: dict, scope: _DialecticalScope
) -> dict | None:
    """Re-resolve one retrieval result and strip out-of-scope fact sources."""
    allowed_sessions = set(scope.session_ids)
    if (
        not isinstance(message, dict)
        or message.get("workspace_id") != scope.workspace_id
        or message.get("peer_id") != scope.target_id
        or message.get("session_id") not in allowed_sessions
    ):
        return None
    wire_id = str(message.get("id", ""))
    if not wire_id.startswith("msg_") or not wire_id[4:].isdigit():
        return None
    proof = _validated_message_occurrence(
        workspace_id=scope.workspace_id,
        session_id=str(message["session_id"]),
        message_id=int(wire_id[4:]),
        peer_id=scope.target_id,
    )
    if (
        proof is None
        or message.get("content") != proof.content
        or message.get("created_at") != proof.source_created_at
    ):
        return None

    shaped = dict(message)
    metadata = message.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    raw_atomic = metadata.get("graph_facts")
    if isinstance(raw_atomic, list):
        retained = []
        for fact in raw_atomic:
            if (
                not isinstance(fact, dict)
                or _validated_fact_key(fact) is None
            ):
                continue
            citations = _validated_fact_citations(
                fact.get("citations"),
                workspace_id=scope.workspace_id,
                target_id=scope.target_id,
                allowed_sessions=allowed_sessions,
            )
            if citations:
                retained.append({**fact, "citations": citations})
        # Once atomic metadata exists, rebuild every legacy alias from the
        # retained atoms. This prevents a rejected citation from surviving in
        # an independently merged graph_claims/citations array.
        metadata["graph_facts"] = retained
        metadata["graph_claims"] = [
            {field: fact.get(field) for field in ("subject", "predicate", "object")}
            for fact in retained
        ]
        metadata["edge_ids"] = [fact.get("edge_id") for fact in retained]
        metadata["citations"] = [
            citation for fact in retained for citation in fact["citations"]
        ]
        if retained:
            metadata["edge_id"] = retained[0].get("edge_id")
        else:
            metadata.pop("edge_id", None)
    shaped["metadata"] = metadata
    return shaped


def _scoped_search_messages(
    scope: _DialecticalScope,
    query: str,
    *,
    limit: int,
) -> list[dict]:
    """Retrieve then re-authorize every result against the directional scope.

    Native retrieval runs a constant number of times for the whole peer scope,
    then every result is checked against the authorized session set. An exact
    bounded lossless lexical fallback covers fresh or raw-pruned history and
    prevents an unauthorized high-scoring turn from starving the allowed pool.
    """
    allowed_sessions = set(scope.session_ids)
    if not allowed_sessions:
        return []
    by_occurrence: dict[tuple[str, str, str, str], dict] = {}
    per_session_limit = min(100, max(10, limit * 2))
    # A global directional collection needs only one ranked retrieval. An
    # explicit local collection supplies its sole session to the same call.
    # Keep the loop/constant visible so a future edit cannot accidentally
    # restore one embedding/provider call per historical session.
    retrieval_slices = (scope.explicit_session_id,)
    assert len(retrieval_slices) <= _MAX_SCOPED_AUGMENT_CALLS
    for session_id in retrieval_slices:
        for message in _augment_messages(
            query,
            min(100, max(per_session_limit, limit * 4)),
            scope.workspace_id,
            session_id=session_id,
            peer_id=scope.target_id,
        ):
            authorized = _authorize_scoped_message(message, scope)
            if authorized is None:
                continue
            by_occurrence.setdefault(
                _message_occurrence_key(authorized), authorized
            )

    # Exact fallback/ranking. It is deliberately simple but deterministic; it
    # supplements rather than replaces the semantic/graph retrieval above.
    terms = {
        token.casefold()
        for token in __import__("re").findall(r"[\w'-]+", query)
        if token.strip()
    }
    for proof in _scoped_occurrences(
        scope,
        per_session_limit=per_session_limit,
        total_limit=min(_MAX_SCOPED_OCCURRENCES, max(128, limit * 16)),
    ):
        haystack = proof.content.casefold()
        matched = sum(term in haystack for term in terms)
        if terms and matched == 0:
            continue
        message = msg(
            proof.message_id,
            proof.content,
            proof.source_peer_id,
            proof.session_id,
            scope.workspace_id,
            created_at=proof.source_created_at,
        )
        message["metadata"] = {
            "type": "exact_occurrence",
            "normalized_score": matched / max(1, len(terms)),
            "source_tiers": ["message"],
            "retrieval_provenance": [{
                "tier": "message",
                "artifact_id": str(proof.message_id),
                "rank": None,
                "score": matched,
                "score_kind": "exact_lexical_overlap",
                "why_retrieved": ["lossless_coverage"],
            }],
        }
        occurrence_key = _message_occurrence_key(message)
        existing = by_occurrence.get(occurrence_key)
        if existing is None:
            by_occurrence[occurrence_key] = message
        else:
            existing_metadata = existing.get("metadata")
            if not isinstance(existing_metadata, dict):
                existing_metadata = {}
                existing["metadata"] = existing_metadata
            existing_metadata["normalized_score"] = max(
                float(existing_metadata.get("normalized_score") or 0.0),
                float(message["metadata"]["normalized_score"]),
            )
            existing_metadata["source_tiers"] = sorted({
                *existing_metadata.get("source_tiers", []), "message",
            })
            provenance = existing_metadata.get("retrieval_provenance")
            if not isinstance(provenance, list):
                provenance = []
            lexical_provenance = message["metadata"]["retrieval_provenance"][0]
            if lexical_provenance not in provenance:
                provenance.append(lexical_provenance)
            existing_metadata["retrieval_provenance"] = provenance

    ranked = list(by_occurrence.values())
    # Stable newest-first tie break followed by stable relevance sort.
    ranked.sort(key=lambda message: (
        str(message.get("created_at", "")),
        str(message.get("session_id", "")),
        str(message.get("id", "")),
    ), reverse=True)
    ranked.sort(
        key=lambda message: float(
            message.get("metadata", {}).get("normalized_score") or 0.0
        ),
        reverse=True,
    )
    return ranked[:limit]


def _representation_messages(
    scope: _DialecticalScope,
    *,
    search_query: str | None,
    limit: int,
) -> list[dict]:
    if search_query is not None:
        return _scoped_search_messages(scope, search_query, limit=limit)
    occurrences = _scoped_occurrences(
        scope,
        per_session_limit=max(1, limit),
        total_limit=min(_MAX_SCOPED_OCCURRENCES, max(128, limit * 4)),
    )
    # A representation is the newest bounded window presented in chronological
    # order, so event-time updates remain interpretable and deterministic.
    selected = occurrences[-limit:]
    return [
        msg(
            proof.message_id,
            proof.content,
            proof.source_peer_id,
            proof.session_id,
            scope.workspace_id,
            created_at=proof.source_created_at,
        )
        for proof in selected
    ]


def _render_representation(
    scope: _DialecticalScope,
    *,
    search_query: str | None = None,
    limit: int = 100,
) -> tuple[str, list[dict]]:
    messages = _representation_messages(
        scope, search_query=search_query, limit=limit
    )
    if not messages:
        return "", []
    lines = [
        "=== PEER REPRESENTATION "
        f"({scope.observer_id} -> {scope.target_id}) ==="
    ]
    for message in messages:
        lines.append(
            "- "
            f"[{message['created_at']} session={message['session_id']} "
            f"message={message['id']}] "
            f"{sanitize_evidence_text(message['content'])}"
        )
    return "\n".join(lines), messages


def _render_peer_card(
    workspace_id: str,
    observer_id: str,
    *,
    target_id: str | None = None,
    required_session_id: str | None = None,
) -> tuple[str, list[dict]]:
    """Render a directional card independently of query representation.

    Search and ``limit_to_session`` tune only the representation returned by
    ``Session.context``. The card always uses the same bounded global
    observer/target collection as ``Peer.get_card(target=...)``. An enclosing
    session may still be required so the request cannot bypass its participant
    and observation policy.
    """
    scope = _resolve_dialectical_scope(
        workspace_id,
        observer_id,
        target_id=target_id,
        required_session_id=required_session_id,
    )
    return _render_representation(scope, limit=_PEER_CARD_MAX_ITEMS)


def _fact_scope_for_message(
    messages: list[dict],
    message: dict,
    scope: _DialecticalScope | None,
) -> tuple[str, str, set[str]]:
    """Resolve the citation boundary for a shaped fact message."""
    if scope is not None:
        return scope.workspace_id, scope.target_id, set(scope.session_ids)
    workspace_id = str(message.get("workspace_id", ""))
    target_id = str(message.get("peer_id", ""))
    return workspace_id, target_id, {
        str(item.get("session_id"))
        for item in messages
        if item.get("workspace_id") == workspace_id
        and item.get("peer_id") == target_id
    }


def _merge_fact(
    facts: dict[tuple[object, object, object, object], dict],
    *,
    key: tuple[object, object, object, object],
    citations: list[dict],
) -> None:
    existing = facts.get(key)
    if existing is None:
        facts[key] = {
            "subject": key[1],
            "predicate": key[2],
            "object": key[3],
            "edge_id": key[0],
            "citations": citations,
        }
        return
    merged = {
        (
            citation.get("evidence_id"),
            citation.get("source_session_id"),
            citation.get("source_message_id"),
            citation.get("source_peer_id"),
            citation.get("source_workspace_id"),
        ): citation
        for citation in [*existing["citations"], *citations]
    }
    existing["citations"] = [merged[item] for item in sorted(merged, key=repr)]


def _validated_fact_key(fact: dict) -> tuple[int, str, str, str] | None:
    edge_id = fact.get("edge_id")
    values = tuple(fact.get(field) for field in ("subject", "predicate", "object"))
    if (
        isinstance(edge_id, bool)
        or not isinstance(edge_id, int)
        or edge_id < 1
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        return None
    return edge_id, values[0], values[1], values[2]


def _structured_facts(
    messages: list[dict], *, scope: _DialecticalScope | None = None
) -> list[dict]:
    """Return only atomically associated claims with in-scope citations."""

    facts: dict[tuple[object, object, object, object], dict] = {}
    for message in messages:
        metadata = message.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        raw_atomic = metadata.get("graph_facts")
        atomic = raw_atomic if isinstance(raw_atomic, list) else []
        if atomic:
            for fact in atomic:
                if not isinstance(fact, dict):
                    continue
                workspace_id, target_id, allowed_sessions = (
                    _fact_scope_for_message(messages, message, scope)
                )
                citations = _validated_fact_citations(
                    fact.get("citations"),
                    workspace_id=workspace_id,
                    target_id=target_id,
                    allowed_sessions=allowed_sessions,
                )
                if not citations:
                    continue
                key = _validated_fact_key(fact)
                if key is None:
                    continue
                _merge_fact(facts, key=key, citations=citations)
            continue
        # Legacy graph_claims/edge_ids/citations arrays have no atomic
        # association and therefore cannot establish which source supports
        # which claim. Keep the aliases on Message responses, but do not
        # project them into structured facts.
    return [facts[key] for key in sorted(facts, key=repr)]


def _reasoning_evidence(
    messages: list[dict], *, scope: _DialecticalScope | None = None
) -> list[GroundedEvidence]:
    evidence: list[GroundedEvidence] = []
    for message in messages:
        metadata = message.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        raw_atomic_facts = metadata.get("graph_facts")
        atomic_facts = (
            raw_atomic_facts if isinstance(raw_atomic_facts, list) else []
        )
        claim_lines: list[str] = []
        for fact in atomic_facts:
            if not isinstance(fact, dict):
                continue
            source_labels = []
            workspace_id, target_id, allowed_sessions = _fact_scope_for_message(
                messages, message, scope
            )
            citations = _validated_fact_citations(
                fact.get("citations"),
                workspace_id=workspace_id,
                target_id=target_id,
                allowed_sessions=allowed_sessions,
            )
            for citation in citations:
                source_labels.append(
                    f"role={citation.get('source_role', '')}, "
                    f"session={citation.get('source_session_id', '')}, "
                    f"message={citation.get('source_message_id', '')}, "
                    f"peer={citation.get('source_peer_id', '')}, "
                    f"workspace={citation.get('source_workspace_id', '')}"
                )
            if not source_labels:
                continue
            claim_lines.append(sanitize_evidence_text(
                f"[edge {fact.get('edge_id', 'unavailable')}] "
                f"{fact.get('subject', '')} {fact.get('predicate', '')} "
                f"{fact.get('object', '')} [sources: {'; '.join(source_labels)}]"
            ))
        # Pre-atomic graph_claims/edge_ids aliases are returned for legacy
        # clients but never become synthesis evidence: even a one-item array
        # cannot prove that a separately stored citation belongs to that claim.
        suffix = (
            " | derived claim: " + "; ".join(claim_lines)
            if claim_lines else ""
        )
        text = (
            f"- [{message.get('created_at', '')} "
            f"session={message.get('session_id', '')} "
            f"message={message.get('id', '')} peer={message.get('peer_id', '')}] "
            f"{sanitize_evidence_text(message.get('content', ''))}{suffix}"
        )
        evidence.append(GroundedEvidence(_message_occurrence_key(message), text))
    return evidence


def _deterministic_scoped_answer(
    messages: list[dict], *, scope: _DialecticalScope
) -> str:
    if not messages:
        return "No relevant information found in the authorized memory scope."
    return "From authorized memory:\n" + "\n".join(
        item.text for item in _reasoning_evidence(messages, scope=scope)
    )


@app.post("/v3/workspaces/{workspace_id}/sessions/{session_id}/search")
def search_messages(workspace_id: str, session_id: str, body: SearchRequest) -> list[dict]:
    _session_row(workspace_id, session_id)
    if body.filters:
        raise HTTPException(
            status_code=422,
            detail="search filters are not supported by this backend",
        )
    return _augment_messages(
        body.query,
        body.limit,
        workspace_id,
        session_id=session_id,
    )


# ── context ──────────────────────────────────────────────────────────────────

@app.get("/v3/workspaces/{workspace_id}/sessions/{session_id}/context")
def get_context(
    workspace_id: str,
    session_id: str,
    summary: bool = True,
    tokens: int = Query(default=1000, ge=1),
    peer_target: str | None = Query(default=None, min_length=1, max_length=4096),
    peer_perspective: str | None = Query(
        default=None, min_length=1, max_length=4096
    ),
    limit_to_session: bool = False,
    search_query: str | None = Query(
        default=None, min_length=1, max_length=10_000
    ),
    search_top_k: int | None = Query(default=None, ge=1, le=100),
    search_max_distance: float | None = Query(default=None, ge=0.0, le=1.0),
    include_most_frequent: bool | None = None,
    max_conclusions: int | None = Query(default=None, ge=1, le=100),
) -> dict:
    hy = _get_hy()
    _session_row(workspace_id, session_id)
    if peer_target is None and peer_perspective is not None:
        raise HTTPException(
            status_code=422,
            detail="peer_perspective requires peer_target",
        )
    if peer_target is None and search_query is not None:
        raise HTTPException(status_code=422, detail="search_query requires peer_target")
    if peer_target is None and any(value is not None for value in (
        search_top_k, search_max_distance, include_most_frequent,
        max_conclusions,
    )):
        raise HTTPException(
            status_code=422,
            detail="representation controls require peer_target",
        )

    session_row = hy.conn.execute(
        "SELECT summary, auto_summary, summary_source FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    from hymem.dreaming.summary import effective_session_summary
    session_summary = effective_session_summary(session_row)

    # Session context is local until a peer target is explicitly requested.
    # Process-global MEMORY.md, USER.md, digest/profile, and unowned rules have
    # no workspace+peer authority and therefore never enter this route.
    summary_text = session_summary or ""
    summary_type = "session"
    selected_summary = ""
    selected_messages: list[dict] = []
    selected_representation = ""
    selected_peer_card: list[str] | None = None
    omitted = 0

    def message_cost(message: dict) -> int:
        return estimate_tokens(
            "role:assistant\n"
            f"name:{message['peer_id']}\n"
            f"content:{message['content']}"
        ) + 4

    def context_cost(
        summary_content: str,
        representation: str,
        peer_card: list[str] | None,
        messages: list[dict],
    ) -> int:
        # Mirror SessionContext.to_openai(): its XML framing is part of the
        # model-visible context and therefore part of the hard total. The small
        # per-message allowance covers ChatML role/name framing when an exact
        # cached tokenizer (rather than the conservative byte fallback) is used.
        total = 0
        if representation:
            total += estimate_tokens(
                "role:system\ncontent:"
                f"<peer_representation>{representation}</peer_representation>"
            ) + 4
        if peer_card:
            # Exact framing used by honcho.session.SessionContext.to_openai().
            # The SDK interpolates the decoded Python list, including brackets
            # and quotes, rather than joining individual card strings.
            total += estimate_tokens(
                "role:system\ncontent:"
                f"<peer_card>{peer_card}</peer_card>"
            ) + 4
        if summary_content:
            total += estimate_tokens(
                f"role:system\ncontent:<summary>{summary_content}</summary>"
            ) + 4
        total += sum(message_cost(message) for message in messages)
        return total

    optional_representation = ""
    optional_peer_card: list[str] | None = None
    if peer_target is not None:
        observer_id = peer_perspective or peer_target
        scope = _resolve_dialectical_scope(
            workspace_id,
            observer_id,
            target_id=peer_target,
            session_id=session_id if limit_to_session else None,
            required_session_id=session_id,
        )
        representation_limit = _validate_representation_controls(
            search_top_k=search_top_k,
            search_max_distance=search_max_distance,
            include_most_frequent=include_most_frequent,
            max_conclusions=max_conclusions,
        )
        optional_representation, _ = _render_representation(
            scope, search_query=search_query, limit=representation_limit
        )
        card, _ = _render_peer_card(
            workspace_id,
            observer_id,
            target_id=peer_target,
            required_session_id=session_id,
        )
        optional_peer_card = [card] if card else []

    # A targeted representation/card pair is the sole cross-session signal.
    # Keep the two SDK-visible views atomic and count both exact wrappers.
    if peer_target is not None:
        trial_representation = optional_representation
        trial_peer_card = optional_peer_card
        if context_cost(
            selected_summary,
            trial_representation,
            trial_peer_card,
            selected_messages,
        ) <= tokens:
            selected_representation = trial_representation
            selected_peer_card = trial_peer_card
        else:
            omitted += int(bool(trial_representation)) + int(bool(trial_peer_card))

    # Long summaries cannot erase all exact recent turns: if the whole item
    # does not fit after the cross-session representation, skip it and continue.
    if summary and summary_text.strip():
        trial_summary = "\n\n".join(
            part for part in (selected_summary, summary_text) if part
        )
        if context_cost(
            trial_summary,
            selected_representation,
            selected_peer_card,
            selected_messages,
        ) <= tokens:
            selected_summary = trial_summary
        else:
            omitted += 1

    # Prefer the most recent exact occurrences, then restore chronology. Keep
    # scanning after an oversized turn so an older compact one can still fit.
    chosen_reverse: list[dict] = []
    used_tokens = context_cost(
        selected_summary,
        selected_representation,
        selected_peer_card,
        selected_messages,
    )
    for occurrence_index, occurrence in enumerate(
        _iter_validated_session_occurrences_desc(workspace_id, session_id)
    ):
        if occurrence_index >= _MAX_CONTEXT_OCCURRENCES:
            omitted += 1
            break
        message = msg(
            occurrence.message_id,
            occurrence.content,
            occurrence.source_peer_id,
            occurrence.session_id,
            workspace_id,
            created_at=occurrence.source_created_at,
        )
        cost = message_cost(message)
        if used_tokens + cost <= tokens:
            chosen_reverse.append(message)
            used_tokens += cost
        else:
            omitted += 1
    selected_messages = list(reversed(chosen_reverse))
    assert used_tokens == context_cost(
        selected_summary,
        selected_representation,
        selected_peer_card,
        selected_messages,
    )

    summary_obj = (
        adapters.summary_obj(selected_summary, summary_type)
        if selected_summary else None
    )

    peer_ids = [
        row["peer_id"]
        for row in hy.conn.execute(
            "SELECT peer_id FROM session_peers WHERE session_id = ? "
            "AND workspace_id = ? ORDER BY added_at, peer_id",
            (session_id, workspace_id),
        ).fetchall()
    ]
    return {
        "summary": summary_obj,
        "messages": selected_messages,
        "peer_representation": selected_representation,
        "peer_card": selected_peer_card,
        "peers": [{"id": peer_id} for peer_id in peer_ids],
        "context_token_budget": tokens,
        "context_token_count": used_tokens,
        "context_truncated": omitted > 0,
        "context_omitted_items": omitted,
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
    existing_session = _session_row(
        workspace_id, session_id, creating=True
    )
    responses: list[dict] = []
    try:
        with core_db.transaction(hy.conn):
            if existing_session is None:
                session_log.open_session(
                    hy.conn, session_id, source_workspace_id=workspace_id
                )
            for peer_id, metadata, configuration in adapters.parse_add_peers(body):
                configuration = _validated_session_configuration(configuration)
                existing = hy.conn.execute(
                    "SELECT role,metadata FROM peers "
                    "WHERE id = ? AND workspace_id = ?",
                    (peer_id, workspace_id),
                ).fetchone()
                role = existing["role"] if existing else infer_role(peer_id)
                if existing is None:
                    hy.conn.execute(
                        "INSERT INTO peers(id,workspace_id,role,metadata) "
                        "VALUES (?,?,?,?)",
                        (
                            peer_id, workspace_id, role,
                            _encode_peer_state(metadata),
                        ),
                    )
                    public_metadata = metadata
                else:
                    peer_state = _decode_peer_state(existing["metadata"])
                    if peer_state is None:
                        raise ValueError("peer registry metadata is malformed")
                    public_metadata = peer_state[0]
                session_log.register_session_peer(
                    hy.conn,
                    session_id,
                    workspace_id,
                    peer_id,
                    role,
                    configuration=configuration,
                )
                responses.append({
                    "id": peer_id,
                    "workspace_id": workspace_id,
                    "metadata": public_metadata,
                    "configuration": configuration,
                })
    except (KeyError, sqlite3.IntegrityError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return responses


@app.get("/v3/workspaces/{workspace_id}/sessions/{session_id}/peers/{peer_id}/config")
def get_peer_config(workspace_id: str, session_id: str, peer_id: str) -> dict:
    """Get per-session peer configuration."""
    hy = _get_hy()
    _session_row(workspace_id, session_id)
    row = hy.conn.execute(
        "SELECT configuration FROM session_peers WHERE session_id = ? "
        "AND workspace_id = ? AND peer_id = ?",
        (session_id, workspace_id, peer_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session peer not found")
    flags = _observation_flags(row["configuration"])
    if flags is None:
        raise HTTPException(
            status_code=409, detail="Session peer configuration is malformed"
        )
    return {"observe_me": flags[0], "observe_others": flags[1]}


# ── peers (workspace-scoped) ─────────────────────────────────────────────────

@app.post("/v3/workspaces/{workspace_id}/peers/{peer_id}/search")
def search_peer_messages(
    workspace_id: str, peer_id: str, body: SearchRequest
) -> list[dict]:
    """Search exact occurrences authored by this peer across its sessions."""
    _peer_row(workspace_id, peer_id)
    if body.filters:
        raise HTTPException(
            status_code=422,
            detail="search filters are not supported by this backend",
        )
    return _augment_messages(
        body.query,
        body.limit,
        workspace_id,
        peer_id=peer_id,
    )


@app.get("/v3/workspaces/{workspace_id}/peers/{peer_id}/card")
def get_peer_card(
    workspace_id: str,
    peer_id: str,
    target: str | None = Query(default=None, min_length=1, max_length=4096),
) -> dict:
    representation, messages = _render_peer_card(
        workspace_id,
        peer_id,
        target_id=target,
    )
    target_peer = _peer_row(workspace_id, target or peer_id)
    updated_at = (
        messages[-1]["created_at"]
        if messages
        else target_peer["registered_at"]
    )
    return adapters.peer_card_response(
        peer_id, workspace_id, representation, updated_at=updated_at
    )


@app.get("/v3/workspaces/{workspace_id}/peers/{peer_id}/context")
def get_peer_context(
    workspace_id: str,
    peer_id: str,
    target: str | None = Query(default=None, min_length=1, max_length=4096),
    search_query: str | None = Query(
        default=None, min_length=1, max_length=10_000
    ),
    limit_to_session: bool = False,
    summary: bool = True,
    search_top_k: int | None = Query(default=None, ge=1, le=100),
    search_max_distance: float | None = Query(default=None, ge=0.0, le=1.0),
    include_most_frequent: bool | None = None,
    max_conclusions: int | None = Query(default=None, ge=1, le=100),
) -> dict:
    """Read target's representation from path peer's perspective."""
    if limit_to_session:
        raise HTTPException(
            status_code=422,
            detail="limit_to_session requires the representation POST session_id",
        )
    limit = _validate_representation_controls(
        search_top_k=search_top_k,
        search_max_distance=search_max_distance,
        include_most_frequent=include_most_frequent,
        max_conclusions=max_conclusions,
    )
    scope = _resolve_dialectical_scope(
        workspace_id, peer_id, target_id=target
    )
    peer_representation, representation_messages = _render_representation(
        scope, search_query=search_query, limit=limit
    )
    card, _ = _render_peer_card(
        workspace_id,
        peer_id,
        target_id=target,
    )

    return {
        # honcho-ai's PeerContextResponse requires peer_id/target_id and
        # declares the representation field as `representation` with no alias
        # (SessionContext, by contrast, maps `peer_representation` correctly).
        # Without all three the SDK either raises a ValidationError or silently
        # drops the value — either way SDK consumers (e.g. the Hermes harness
        # prefetch path) got an empty representation from this route. Send the
        # required ids plus both representation names so scoped evidence
        # reaches the SDK without a client-side patch.
        "peer_id": peer_id,
        "target_id": scope.target_id,
        "representation": peer_representation,
        "peer_card": [card] if card else [],
        # Additive legacy aliases/direct-HTTP fields. They carry the same
        # already-scoped bytes, never a second global source.
        "summary": None,
        "messages": representation_messages if search_query else [],
        "peer_representation": peer_representation,
    }


@app.post("/v3/workspaces/{workspace_id}/peers/{peer_id}/representation")
def get_peer_representation(
    workspace_id: str,
    peer_id: str,
    body: RepresentationRequest,
) -> dict:
    """SDK-compatible POST read of a directional working representation."""
    limit = _validate_representation_controls(
        search_top_k=body.search_top_k,
        search_max_distance=body.search_max_distance,
        include_most_frequent=body.include_most_frequent,
        max_conclusions=body.max_conclusions,
    )
    scope = _resolve_dialectical_scope(
        workspace_id,
        peer_id,
        target_id=body.target,
        session_id=body.session_id,
    )
    representation, _ = _render_representation(
        scope, search_query=body.search_query, limit=limit
    )
    return {"representation": representation}


@app.get("/v3/workspaces/{workspace_id}/peers/{peer_id}/representation")
def get_peer_representation_legacy(
    workspace_id: str,
    peer_id: str,
    session_id: str | None = Query(default=None, min_length=1, max_length=4096),
    target: str | None = Query(default=None, min_length=1, max_length=4096),
    search_query: str | None = Query(
        default=None, min_length=1, max_length=10_000
    ),
    search_top_k: int | None = Query(default=None, ge=1, le=100),
    search_max_distance: float | None = Query(default=None, ge=0.0, le=1.0),
    include_most_frequent: bool | None = None,
    max_conclusions: int | None = Query(default=None, ge=1, le=100),
) -> dict:
    """Compatibility read for pre-SDK callers that used GET."""
    body = RepresentationRequest(
        session_id=session_id,
        target=target,
        search_query=search_query,
        search_top_k=search_top_k,
        search_max_distance=search_max_distance,
        include_most_frequent=include_most_frequent,
        max_conclusions=max_conclusions,
    )
    return get_peer_representation(workspace_id, peer_id, body)


# ── dialectic (honcho_reasoning) ─────────────────────────────────────────────

@app.post("/v3/workspaces/{workspace_id}/peers/{peer_id}/chat")
def peer_chat(workspace_id: str, peer_id: str, body: ChatRequest):
    scope = _resolve_dialectical_scope(
        workspace_id,
        peer_id,
        target_id=body.target,
        session_id=body.session_id,
    )
    queries = list(body.queries) if body.queries is not None else [body.query]
    responses: list[str] = []
    facts_per_query: list[list[dict]] = []

    for q in queries:
        # Pydantic proves this before the handler; the local assertion keeps
        # direct Python callers from widening a malformed request.
        assert isinstance(q, str) and q.strip()
        messages = _scoped_search_messages(scope, q, limit=40)
        deterministic = _deterministic_scoped_answer(messages, scope=scope)
        hy = _get_hy()
        answer = reason_iteratively(
            getattr(hy, "_llm", None),
            question=q,
            evidence=_reasoning_evidence(messages, scope=scope),
            deterministic_answer=deterministic,
            reasoning_level=body.reasoning_level or "low",
            max_tokens=hy.config.ask_max_tokens,
            max_context_chars=hy.config.ask_max_context_chars,
            max_context_tokens=hy.config.ask_max_context_tokens,
        )
        responses.append(answer)
        facts_per_query.append(_structured_facts(messages, scope=scope))

    # `content` is the field the honcho-ai SDK's peer.chat() actually reads
    # (`data.get("content")`); without it the SDK returns None and
    # honcho_reasoning comes back empty. `response` is kept as an alias for
    # consumers that read the HyMem-native shape. `facts` is additive metadata
    # for consumers that want the structured why_retrieved trail without parsing
    # prose — it aligns with the first query; `facts_by_query` carries the full
    # per-query breakdown.
    payload = {
        "content": responses[0],
        "response": responses[0],
        "queries": queries,
        "peer_id": peer_id,
        "target_id": scope.target_id,
        "session_id": scope.explicit_session_id,
        "reasoning_level": body.reasoning_level or "low",
        "facts": facts_per_query[0] if facts_per_query else [],
        "facts_by_query": facts_per_query,
    }
    if not body.stream:
        return payload

    def event_stream():
        # The SDK parser accepts any chunking. Fixed character chunks are
        # deterministic and JSON encoding keeps UTF-8 boundaries intact.
        content = payload["content"]
        for offset in range(0, len(content), 256):
            event = {"delta": {"content": content[offset:offset + 256]}}
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        yield "data: {\"done\":true}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    host = os.environ.get("HYMEM_HONCHO_HOST", "127.0.0.1")
    port = int(os.environ.get("HYMEM_HONCHO_PORT", "8765"))
    uvicorn.run("hymem.honcho.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
