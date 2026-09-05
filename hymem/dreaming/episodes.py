from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field

from hymem.dreaming.aggregation_provenance import (
    BoundSourceOccurrence,
    persist_episode_source_manifest,
    resolve_cited_episode_sources,
    unpublish_episode_source_manifest,
)
from hymem.extraction.jsonio import loads_lenient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import EPISODE_SYSTEM, EPISODE_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.episodes")

_VALID_OUTCOMES = frozenset({"resolved", "blocked", "deferred", "informational"})


@dataclass
class EpisodesExtraction:
    """Validated episode items ready to persist. Empty list = LLM returned
    nothing usable; None at the call boundary = nothing to extract."""
    items: list[dict] = field(default_factory=list)


def extract_episodes_for_session(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
) -> EpisodesExtraction | None:
    """Read the session's chunks and run the episode-extraction LLM call.
    Returns None when there is nothing to extract from. No write transaction
    held; persist via persist_episodes inside one.
    """
    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE session_id = ? "
        "AND chunk_kind = 'extraction' ORDER BY start_message_id",
        (session_id,),
    ).fetchall()
    if not rows:
        return None
    valid_chunk_ids = {r["id"] for r in rows}

    combined = "\n\n---\n\n".join(
        f"[chunk {r['id']}] {r['text']}" for r in rows
    )

    request = LLMRequest(
        system=EPISODE_SYSTEM,
        user=EPISODE_USER_TEMPLATE.format(text=combined),
        response_format="json",
    )
    raw = llm.complete(request)

    # EPISODE_SYSTEM asks for a bare JSON array; fences/prose around it are
    # tolerated (dream 1013 — json_object mode is a request, not a contract).
    data = loads_lenient(raw, expect="array")
    if data is None:
        log.warning("episodes.parse_failure session_id=%s raw_len=%d",
                    session_id, len(raw) if isinstance(raw, str) else -1)
        return EpisodesExtraction()
    if not isinstance(data, list):
        # validate_episode_items() returns [] for a non-list, which is the
        # behavior we want but not the silence — an empty extraction here is
        # indistinguishable from "the session held no episodes".
        log.warning("episodes.shape_failure session_id=%s type=%s",
                    session_id, type(data).__name__)
        return EpisodesExtraction()

    return EpisodesExtraction(items=validate_episode_items(data, valid_chunk_ids))


def validate_episode_items(
    data: object,
    valid_chunk_ids: set[str],
    *,
    max_items: int | None = None,
) -> list[dict]:
    """Validate raw LLM episode items into clean dicts ready to persist.

    Shared by the standalone episode call and the batched session digest. Drops
    items missing title/summary.  For positional compatibility the standalone
    result still exposes only recognized chunk ids, but a private marker records
    that a non-empty citation claim was altered. Persistence rejects that item
    transactionally; it must never turn a hallucinated proof into an
    unattributed episode and advance a digest cursor. Returns [] for any
    non-list ``data``.

    ``max_items`` (Plan C's ``dream_max_episodes_per_session``) truncates a
    runaway reply BEFORE any row is written — the profile/facts validator
    precedent. It defaults to None = unbounded, which is what every pre-Plan-C
    caller gets: the blob prompt asks for a handful of segments and has never
    been capped, and capping it here would be a silent default change.
    """
    if not isinstance(data, list):
        return []
    items: list[dict] = []
    for item in data:
        if max_items is not None and len(items) >= max_items:
            break
        if not isinstance(item, dict):
            continue
        title = item.get("title", "")
        summary = item.get("summary", "")
        if not isinstance(title, str) or not isinstance(summary, str):
            continue
        if not title.strip() or not summary.strip():
            continue
        raw_chunk_ids = item.get("chunk_ids", [])
        citations_invalid = bool(raw_chunk_ids) and not isinstance(
            raw_chunk_ids, list
        )
        if not isinstance(raw_chunk_ids, list):
            raw_chunk_ids = []
        # Drop anything the LLM hallucinated that isn't in the input.
        chunk_ids = [c for c in raw_chunk_ids if isinstance(c, str) and c in valid_chunk_ids]
        if raw_chunk_ids and len(chunk_ids) != len(raw_chunk_ids):
            citations_invalid = True
        clean = dict(item)
        clean["title"] = title.strip()
        clean["summary"] = summary.strip()
        clean["chunk_ids"] = chunk_ids
        if citations_invalid:
            clean["_source_citations_invalid"] = True
        items.append(clean)
    return items


def _resolve_message_range(
    conn: sqlite3.Connection, chunk_ids: list[str]
) -> tuple[int | None, int | None]:
    """Return (min start_message_id, max end_message_id) across the named chunks,
    or (None, None) if the list is empty or no chunks resolved."""
    if not chunk_ids:
        return None, None
    placeholders = ",".join("?" * len(chunk_ids))
    row = conn.execute(
        f"""
        SELECT MIN(start_message_id) AS s, MAX(end_message_id) AS e
        FROM chunks WHERE id IN ({placeholders})
        """,
        tuple(chunk_ids),
    ).fetchone()
    if row is None:
        return None, None
    return row["s"], row["e"]


def _resolve_participants(
    conn: sqlite3.Connection,
    session_id: str,
    start_msg: int | None,
    end_msg: int | None,
) -> list[str]:
    """Distinct speakers in the range, preferring exact external peer ids.

    Native/legacy rows retain their role label. A workspace-qualified author
    is never collapsed to that role, so two user peers remain two participants.
    """
    if start_msg is None or end_msg is None:
        rows = conn.execute(
            """
            SELECT COALESCE(source_peer_id, role) AS participant
            FROM messages WHERE session_id = ?
            UNION
            SELECT COALESCE(source_peer_id, source_role) AS participant
            FROM message_retention_coverage
            WHERE source_session_id = ?
            ORDER BY participant
            """,
            (session_id, session_id),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT COALESCE(source_peer_id, role) AS participant FROM messages
            WHERE session_id = ? AND id BETWEEN ? AND ?
            UNION
            SELECT COALESCE(source_peer_id, source_role) AS participant
            FROM message_retention_coverage
            WHERE source_session_id = ? AND message_id BETWEEN ? AND ?
            ORDER BY participant
            """,
            (session_id, start_msg, end_msg, session_id, start_msg, end_msg),
        ).fetchall()
    return [r["participant"] for r in rows]


def _participants_from_sources(
    occurrences: tuple[BoundSourceOccurrence, ...],
) -> list[str]:
    """Exact participants in stable lexical order, without range inference."""

    return sorted({item.source_peer_id or item.role for item in occurrences})


def _title_hash(session_id: str, title: str) -> str:
    return hashlib.sha1(
        f"{session_id}|{title.strip().lower()}".encode("utf-8")
    ).hexdigest()[:12]


def _episode_id(
    session_id: str,
    start_msg: int | None,
    end_msg: int | None,
    title: str,
    *,
    granular: bool = False,
    digest_slice_key: str | None = None,
    digest_item_index: int | None = None,
    digest_generation: str | None = None,
) -> str:
    """Stable id for an episode.

    Standalone extraction prefers the message range — same range across
    re-dreams = same id. Resumable digest extraction additionally keys on the
    active build generation, stable slice, and response ordinal.  Several valid
    blob episodes sharing one coverage range coexist, while retries inside one
    build still UPSERT.  A replacement build receives distinct ids so it cannot
    overwrite the last complete generation before reaching the tail. Granular
    episodes retain range+title identity inside that build. When the LLM didn't
    provide chunk_ids we use an equivalent slice/ordinal fallback.

    `granular` (Plan C) appends a title hash to the RANGE id, and the reason is
    the whole reason Plan C needs a persist change at all: at decision
    granularity several episodes of one session legitimately cite the same
    chunk — "chose fly.io" and "hit the 512MB memory limit" can both rest on
    chunk chk_7 — so they resolve to the SAME message range. Under the bare
    range id the second UPSERT would silently overwrite the first and a
    3-8-episode session would persist as one row, which reads downstream as the
    granularity change having done nothing. Same range + same title still means
    the same id, so re-dreams still UPSERT in place instead of duplicating.
    """
    generation_part = ""
    if digest_slice_key is not None and digest_generation is not None:
        generation_hash = hashlib.sha256(
            digest_generation.encode("utf-8")
        ).hexdigest()[:16]
        generation_part = f"~g{generation_hash}"
    if start_msg is not None and end_msg is not None:
        base = f"{session_id}@{start_msg}-{end_msg}{generation_part}"
        if digest_slice_key is not None:
            slice_hash = hashlib.sha1(digest_slice_key.encode("utf-8")).hexdigest()[:10]
            base = f"{base}~{slice_hash}"
        # Every resumable digest slice may legitimately emit several episodes
        # over the same one-message coverage range.  The historical blob path
        # collapsed those rows by range; adding title identity whenever a
        # slice key is present prevents one valid item silently overwriting the
        # next while preserving standalone/pre-v38 ids.
        if granular:
            return f"{base}#{_title_hash(session_id, title)}"
        if digest_slice_key is not None:
            # Blob episodes historically used range identity so a title edit
            # UPSERTed in place.  Preserve that stability while allowing more
            # than one episode from the same bounded window by adding its
            # deterministic response ordinal, not its mutable title.
            return f"{base}#i{int(digest_item_index or 0)}"
        return base
    base = f"{session_id}@h{_title_hash(session_id, title)}{generation_part}"
    if digest_slice_key is not None:
        slice_hash = hashlib.sha1(digest_slice_key.encode("utf-8")).hexdigest()[:10]
        if granular:
            return f"{base}~{slice_hash}"
        return (
            f"{session_id}@slice{generation_part}~{slice_hash}"
            f"#i{int(digest_item_index or 0)}"
        )
    return base


def persist_episodes(
    conn: sqlite3.Connection,
    session_id: str,
    extraction: EpisodesExtraction,
    *,
    granular: bool = False,
    supersede_window: tuple[int | None, int | None] | None = None,
    digest_slice_key: str | None = None,
    digest_generation: str | None = None,
) -> int:
    """Insert/UPSERT validated episodes. Caller wraps in core_db.transaction().

    Each episode gets a stable id derived from the message range (see
    ``_episode_id``). UPSERT-on-conflict refreshes title/summary/outcome
    /key_entities/participants for re-dreams without reshuffling rowids —
    important for FTS and vec_episodes alignment.

    ``granular`` (Plan C) selects the decision-grained range+title id shape;
    resumable blob calls use range+slice+ordinal. Which prompt an episode came
    from is recorded once per extraction call, on
    ``sessions.episodes_prompt_version`` (schema v35), because the call is what
    has a prompt version; a per-row copy would be derived state that can
    disagree with its source.

    ``supersede_window`` (start, end) turns the write into a REPLACE of the
    episodes inside that message window, and the runner passes it on EITHER side
    of a granularity change -- flipping on, and reverting off again. Without it
    a granularity change is silently additive: UPSERT refreshes a row only when
    the new episode resolves to the SAME id, so the rows written under the other
    id shape (different id, therefore no conflict) survive the re-extraction and
    the store ends up serving both granularities of the same conversation. It is
    NOT passed on a store that has only ever run the blob prompt, whose re-dreams
    stay additive exactly as before. Scoped deliberately:

      * only rows whose range lies wholly INSIDE the window this call re-read —
        a session's older, already-covered episodes are outside it and must not
        be deleted just because this dream read the tail;
      * only rows with a known range. A NULL range is unattributable to any
        window (the hash-id fallback), and NULL means unattributed, so those
        rows are left alone rather than guessed at;
      * never the ids just written.
    """
    written_ids: list[str] = []
    count = 0
    for item_index, item in enumerate(extraction.items):
        title = item.get("title", "").strip()
        summary = item.get("summary", "").strip()
        if not title or not summary:
            continue
        chunk_ids = item.get("chunk_ids", []) or []
        if item.get("_source_citations_invalid") is True:
            raise ValueError("episode contains an invalid non-empty source citation")
        # Resolve the exact citation set before writing any part of this item.
        # Empty citations are a supported legacy/unattributed shape.  A non-empty
        # but corrupt citation raises and rolls back the caller's episode + digest
        # cursor transaction; silently downgrading it would make lost provenance
        # permanent once the cursor advanced.
        source_occurrences = resolve_cited_episode_sources(
            conn, session_id, chunk_ids
        )
        start_msg, end_msg = _resolve_message_range(conn, chunk_ids)
        participants = (
            _participants_from_sources(source_occurrences)
            if source_occurrences
            else _resolve_participants(conn, session_id, start_msg, end_msg)
        )
        outcome = item.get("outcome")
        outcome = outcome if outcome in _VALID_OUTCOMES else None
        key_entities = json.dumps(item.get("key_entities", []))
        episode_id = _episode_id(
            session_id, start_msg, end_msg, title, granular=granular,
            digest_slice_key=digest_slice_key,
            digest_item_index=item_index,
            digest_generation=digest_generation,
        )

        # A published manifest binds title/summary as prompt inputs.  Unpublish
        # it before the UPSERT so a same-id replay or in-place rewrite cannot
        # momentarily pair old proof rows with new episode bytes.  The caller's
        # transaction makes the replacement atomic to other connections.
        unpublish_episode_source_manifest(conn, episode_id)
        conn.execute(
            """
            INSERT INTO episodes(
                id, session_id, title, summary, participants,
                start_message_id, end_message_id, outcome, key_entities,
                digest_slice_key, digest_generation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                participants = excluded.participants,
                start_message_id = excluded.start_message_id,
                end_message_id = excluded.end_message_id,
                outcome = excluded.outcome,
                key_entities = excluded.key_entities,
                digest_slice_key = excluded.digest_slice_key,
                digest_generation = excluded.digest_generation
            """,
            (
                episode_id, session_id, title, summary,
                json.dumps(participants),
                start_msg, end_msg, outcome, key_entities, digest_slice_key,
                digest_generation,
            ),
        )
        persist_episode_source_manifest(conn, episode_id, source_occurrences)
        written_ids.append(episode_id)
        count += 1

    if supersede_window is not None:
        _supersede_window(
            conn, session_id, supersede_window, written_ids,
            digest_slice_key=digest_slice_key,
        )

    if count:
        log.debug("episodes.persisted session_id=%s count=%d", session_id, count)
    return count


def _supersede_window(
    conn: sqlite3.Connection,
    session_id: str,
    window: tuple[int | None, int | None],
    keep_ids: list[str],
    *,
    digest_slice_key: str | None = None,
) -> int:
    """Delete this session's episodes that lie wholly inside ``window`` and were
    not just written. Returns the number deleted (0 on an unknown window).

    Deleting rather than leaving the row is the point: an episode is a retrieval
    surface, so a stale one is not inert — it competes for the same episode
    slots as the row that replaced it and it renders in ``ask()``. The FTS
    shadow is kept in sync by the ``episodes_fts_delete`` trigger and
    ``episode_embeddings`` cascades, the same path ``prune_episodes_and
    _procedures`` already takes; ``vec_episodes`` is repaired by the existing
    ``heal_rowid_shadows``/resync machinery, which is where every other episode
    delete in this codebase leaves it too.
    """
    start, end = window
    if start is None or end is None:
        # An unknown window would make the DELETE unbounded. Nothing is deleted
        # and the stale rows stay: over-keeping is recoverable, over-deleting is
        # not.
        return 0
    if not keep_ids:
        # The caller persisted nothing, so there is no replacement to supersede
        # anything with. Wiping the window here would turn an empty extraction —
        # a legitimate "this slice held nothing" — into deletion of episodes
        # that a previous, successful extraction produced.
        return 0
    placeholders = ",".join("?" * len(keep_ids))
    slice_clause = (
        "AND digest_slice_key IS NULL"
        if digest_slice_key is None
        else "AND digest_slice_key = ?"
    )
    params: tuple = (session_id, start, end, *keep_ids)
    if digest_slice_key is not None:
        params = (*params, digest_slice_key)
    cur = conn.execute(
        f"""
        DELETE FROM episodes
        WHERE session_id = ?
          AND start_message_id IS NOT NULL
          AND end_message_id IS NOT NULL
          AND start_message_id >= ?
          AND end_message_id <= ?
          AND id NOT IN ({placeholders})
          {slice_clause}
        """,
        params,
    )
    deleted = cur.rowcount if cur.rowcount > 0 else 0
    if deleted:
        log.info(
            "episodes.superseded session_id=%s window=%s-%s deleted=%d kept=%d",
            session_id, start, end, deleted, len(keep_ids),
        )
    return deleted
