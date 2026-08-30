from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field

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
        "SELECT id, text FROM chunks WHERE session_id = ? ORDER BY start_message_id",
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
    items missing title/summary and strips hallucinated chunk_ids not present in
    the input. Returns [] for any non-list ``data``.

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
        if not isinstance(raw_chunk_ids, list):
            raw_chunk_ids = []
        # Drop anything the LLM hallucinated that isn't in the input.
        chunk_ids = [c for c in raw_chunk_ids if isinstance(c, str) and c in valid_chunk_ids]
        clean = dict(item)
        clean["title"] = title.strip()
        clean["summary"] = summary.strip()
        clean["chunk_ids"] = chunk_ids
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
    """Distinct, sorted roles in the message range (e.g. ['assistant', 'user']).
    Falls back to all session roles if the range is unknown."""
    if start_msg is None or end_msg is None:
        rows = conn.execute(
            "SELECT DISTINCT role FROM messages WHERE session_id = ? ORDER BY role",
            (session_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT DISTINCT role FROM messages
            WHERE session_id = ? AND id BETWEEN ? AND ?
            ORDER BY role
            """,
            (session_id, start_msg, end_msg),
        ).fetchall()
    return [r["role"] for r in rows]


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
) -> str:
    """Stable id for an episode.

    Prefers the message range — same range across re-dreams = same id, so an
    UPSERT updates the title/summary in place. When the LLM didn't provide
    chunk_ids (so range is unknown) we fall back to a content hash so the row
    is still re-findable on the next run for the same content.

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
    if start_msg is not None and end_msg is not None:
        base = f"{session_id}@{start_msg}-{end_msg}"
        return f"{base}#{_title_hash(session_id, title)}" if granular else base
    return f"{session_id}@h{_title_hash(session_id, title)}"


def persist_episodes(
    conn: sqlite3.Connection,
    session_id: str,
    extraction: EpisodesExtraction,
    *,
    granular: bool = False,
    supersede_window: tuple[int | None, int | None] | None = None,
) -> int:
    """Insert/UPSERT validated episodes. Caller wraps in core_db.transaction().

    Each episode gets a stable id derived from the message range (see
    ``_episode_id``). UPSERT-on-conflict refreshes title/summary/outcome
    /key_entities/participants for re-dreams without reshuffling rowids —
    important for FTS and vec_episodes alignment.

    ``granular`` (Plan C) selects the id shape — range+title instead of the bare
    range — and nothing else about the row changes. Which prompt an episode came
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
    for item in extraction.items:
        title = item.get("title", "").strip()
        summary = item.get("summary", "").strip()
        if not title or not summary:
            continue
        chunk_ids = item.get("chunk_ids", []) or []
        start_msg, end_msg = _resolve_message_range(conn, chunk_ids)
        participants = _resolve_participants(conn, session_id, start_msg, end_msg)
        outcome = item.get("outcome")
        outcome = outcome if outcome in _VALID_OUTCOMES else None
        key_entities = json.dumps(item.get("key_entities", []))
        episode_id = _episode_id(
            session_id, start_msg, end_msg, title, granular=granular
        )

        conn.execute(
            """
            INSERT INTO episodes(
                id, session_id, title, summary, participants,
                start_message_id, end_message_id, outcome, key_entities
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                participants = excluded.participants,
                start_message_id = excluded.start_message_id,
                end_message_id = excluded.end_message_id,
                outcome = excluded.outcome,
                key_entities = excluded.key_entities
            """,
            (
                episode_id, session_id, title, summary,
                json.dumps(participants),
                start_msg, end_msg, outcome, key_entities,
            ),
        )
        written_ids.append(episode_id)
        count += 1

    if supersede_window is not None:
        _supersede_window(conn, session_id, supersede_window, written_ids)

    if count:
        log.debug("episodes.persisted session_id=%s count=%d", session_id, count)
    return count


def _supersede_window(
    conn: sqlite3.Connection,
    session_id: str,
    window: tuple[int | None, int | None],
    keep_ids: list[str],
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
    cur = conn.execute(
        f"""
        DELETE FROM episodes
        WHERE session_id = ?
          AND start_message_id IS NOT NULL
          AND end_message_id IS NOT NULL
          AND start_message_id >= ?
          AND end_message_id <= ?
          AND id NOT IN ({placeholders})
        """,
        (session_id, start, end, *keep_ids),
    )
    deleted = cur.rowcount if cur.rowcount > 0 else 0
    if deleted:
        log.info(
            "episodes.superseded session_id=%s window=%s-%s deleted=%d kept=%d",
            session_id, start, end, deleted, len(keep_ids),
        )
    return deleted
