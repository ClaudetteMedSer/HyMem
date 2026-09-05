from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass

# Trigger phrases that mark a turn as worth keeping. Cheap heuristic; the LLM
# extraction step is the real filter. We err on the side of "include" — it's
# better to extract zero triples from a chunk than to silently miss a correction.
_CORRECTION_PATTERNS = re.compile(
    # English
    r"\bno,"
    r"|\b(actually|not\s+(?:quite|really|exactly)|don'?t\s+use|"
    r"stopped\s+using|switched\s+(?:to|from)|replaced|moved\s+(?:off|away))\b"
    r"|\bthat'?s\s+wrong\b|\bwrong\b|\bincorrect\b|\bfix:"
    r"|\bthe\s+right\s+(?:answer|way)\b"
    r"|\binstead\s+of\b|\brather\s+than\b"
    r"|\buse\s+\S+\s+not\s+\S+"
    # Dutch
    r"|\bnee,"
    r"|\b(eigenlijk|gebruik\s+geen|niet\s+gebruiken|gestopt\s+met|"
    r"overgestapt(?:\s+(?:van|naar|op))?|vervangen)\b"
    r"|\b(?:dat\s+klopt\s+niet|niet\s+correct|verkeerd|fout)\b"
    r"|\bde\s+juiste\s+(?:manier|antwoord)\b"
    r"|\bin\s+plaats\s+van\b|\bliever\s+dan\b",
    re.IGNORECASE,
)
_PREFERENCE_PATTERNS = re.compile(
    # English
    r"\b(i\s+prefer|i\s+like|i\s+want|we\s+use|we\s+chose|let'?s\s+use|"
    r"we\s+rely\s+on|we\s+depend\s+on)\b"
    # Dutch
    r"|\b(ik\s+prefereer|ik\s+heb\s+(?:een\s+)?voorkeur\s+voor|ik\s+wil|"
    r"ik\s+gebruik\s+graag|(?:we|wij)\s+gebruiken|we\s+kozen(?:\s+voor)?|"
    r"we\s+hebben\s+gekozen|laten\s+we\s+\S+\s+gebruiken|"
    r"we\s+vertrouwen\s+op|we\s+zijn\s+afhankelijk\s+van)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    id: str
    session_id: str
    start_message_id: int
    end_message_id: int
    salience_reason: str
    text: str
    # Exact ordered source membership for claim extraction. Empty means the
    # legacy chunk has no proven manifest and must not authorize new claims.
    source_message_ids: tuple[int, ...] = ()


def extract_high_salience_chunks(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    min_chars: int,
) -> list[Chunk]:
    """Walk the session messages and produce chunks worth running extraction on.

    Strategy: a sliding pair of (preceding assistant turn, user turn). When the
    user turn matches a trigger or is long enough on its own, we mint a chunk
    spanning the pair so the LLM sees what was being corrected.
    """
    chunks: list[Chunk] = []
    last_assistant: sqlite3.Row | None = None

    for row in conn.execute(
        "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ):
        role = row["role"]
        content = row["content"] or ""
        if role == "assistant":
            last_assistant = row
            continue
        if role != "user":
            continue

        is_trigger = bool(
            _CORRECTION_PATTERNS.search(content) or _PREFERENCE_PATTERNS.search(content)
        )
        is_substantive = len(content) >= min_chars
        if not (is_trigger or is_substantive):
            continue

        start_id = last_assistant["id"] if last_assistant is not None else row["id"]
        end_id = row["id"]
        pieces = []
        if last_assistant is not None:
            pieces.append(f"assistant: {last_assistant['content']}")
        pieces.append(f"user: {content}")
        text = "\n".join(pieces)

        reason = "correction_or_preference_trigger" if is_trigger else "long_user_turn"
        chunk_id = _chunk_id(session_id, start_id, end_id)
        chunks.append(
            Chunk(
                id=chunk_id,
                session_id=session_id,
                start_message_id=start_id,
                end_message_id=end_id,
                salience_reason=reason,
                text=text,
                source_message_ids=tuple(
                    [int(last_assistant["id"])] if last_assistant is not None else []
                ) + (int(row["id"]),),
            )
        )

    return chunks


def extract_baseline_chunks(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    prompt_version: str,
    limit: int,
    min_chars: int,
    max_attempts: int = 0,
    exclude_ids: set[str] | None = None,
) -> list[Chunk]:
    """Build chunks from any user turn (no salience filter) that hasn't been
    processed under the current prompt_version. Newest first.

    Backstop for the high-salience tier: most chunks don't trip the
    correction/preference regexes, so without this they never reach the LLM
    and the graph can't reinforce existing edges via re-mention.

    Caller is expected to use this only when high-salience tier has unspent
    budget — that's enforced upstream in the runner, not here.
    """
    candidates: list[Chunk] = []
    last_assistant: sqlite3.Row | None = None

    for row in conn.execute(
        "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ):
        role = row["role"]
        content = row["content"] or ""
        if role == "assistant":
            last_assistant = row
            continue
        if role != "user":
            continue
        if len(content) < min_chars:
            continue

        start_id = last_assistant["id"] if last_assistant is not None else row["id"]
        end_id = row["id"]
        chunk_id = _chunk_id(session_id, start_id, end_id)

        pieces = []
        if last_assistant is not None:
            pieces.append(f"assistant: {last_assistant['content']}")
        pieces.append(f"user: {content}")
        text = "\n".join(pieces)
        is_trigger = bool(
            _CORRECTION_PATTERNS.search(content)
            or _PREFERENCE_PATTERNS.search(content)
        )

        candidates.append(
            Chunk(
                id=chunk_id,
                session_id=session_id,
                start_message_id=start_id,
                end_message_id=end_id,
                # Chunk identity is source-range based, so every producer must
                # materialize identical durable metadata for that range.  The
                # salience tier can have persisted this same chunk earlier in
                # the cycle (or in another cycle while ingestion continues).
                salience_reason=(
                    "correction_or_preference_trigger"
                    if is_trigger else "long_user_turn"
                ),
                text=text,
                source_message_ids=tuple(
                    [int(last_assistant["id"])] if last_assistant is not None else []
                ) + (int(row["id"]),),
            )
        )

    # Newest first, then drop already-processed and cap to limit.
    candidates.reverse()
    result: list[Chunk] = []
    for chunk in candidates:
        if chunk.id in (exclude_ids or ()):
            continue
        already = conn.execute(
            "SELECT 1 FROM processed_chunks WHERE chunk_id = ? AND prompt_version = ?",
            (chunk.id, prompt_version),
        ).fetchone()
        if already:
            continue
        if chunk_extraction_is_quarantined(
            conn,
            chunk.id,
            prompt_version=prompt_version,
            max_attempts=max_attempts,
        ):
            continue
        result.append(chunk)
        if len(result) >= limit:
            break
    return result


def chunk_extraction_is_quarantined(
    conn: sqlite3.Connection,
    chunk_id: str,
    *,
    prompt_version: str,
    max_attempts: int,
) -> bool:
    """Whether a failed chunk has exhausted the current retry policy.

    Quarantine is derived from the auditable attempt row instead of being
    represented by a false ``processed_chunks`` success. Changing the prompt
    version, raising the bound, or setting it to zero immediately makes the
    chunk eligible again without destructive bookkeeping.
    """
    if max_attempts <= 0:
        return False
    row = conn.execute(
        "SELECT attempts FROM chunk_extraction_attempts "
        "WHERE chunk_id = ? AND prompt_version = ?",
        (chunk_id, prompt_version),
    ).fetchone()
    return bool(row is not None and int(row["attempts"]) >= int(max_attempts))


def load_pending_persisted_chunks(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    prompt_version: str,
    limit: int,
    max_attempts: int = 0,
    exclude_ids: set[str] | None = None,
) -> list[Chunk]:
    """Load already-durable extraction chunks still owed this prompt salt.

    Rebuilding candidates exclusively from ``messages`` made a prompt bump
    impossible to replay after opt-in raw retention. This backlog reader uses
    the stored extraction artifact itself, excludes coverage storage and
    quarantined failures, and lets the runner bound work with its normal
    budget.
    """
    if limit <= 0:
        return []
    # Pre-v40/unmigrated stores have no exact extraction-input membership.
    # Chunk prose is not a trustworthy substitute for the published source
    # manifest, so fail closed instead of returning unverifiable backlog.
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='chunk_message_sources'"
    ).fetchone() is None:
        return []
    excluded = tuple(sorted(exclude_ids or ()))
    exclusion = ""
    params: list[object] = [session_id, prompt_version]
    if excluded:
        exclusion = f"AND c.id NOT IN ({','.join('?' * len(excluded))})"
        params.extend(excluded)
    quarantine = ""
    if max_attempts > 0:
        quarantine = (
            "AND NOT EXISTS ("
            " SELECT 1 FROM chunk_extraction_attempts a"
            " WHERE a.chunk_id = c.id AND a.prompt_version = ?"
            "   AND a.attempts >= ?"
            ")"
        )
        params.extend((prompt_version, int(max_attempts)))
    params.append(int(limit))
    rows = conn.execute(
        f"""
        SELECT c.id, c.session_id, c.start_message_id, c.end_message_id,
               c.salience_reason, c.text
        FROM chunks c
        WHERE c.session_id = ?
          AND c.chunk_kind = 'extraction'
          AND COALESCE(c.salience_reason, '') <> 'short_session_fallback'
          AND NOT EXISTS (
              SELECT 1 FROM processed_chunks pc
              WHERE pc.chunk_id = c.id AND pc.prompt_version = ?
          )
          {exclusion}
          {quarantine}
        ORDER BY c.created_at, c.id
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    result: list[Chunk] = []
    for row in rows:
        manifest = conn.execute(
            "SELECT source_message_id FROM chunk_message_sources "
            "WHERE chunk_id = ? ORDER BY ordinal",
            (row["id"],),
        ).fetchall()
        result.append(Chunk(
            id=row["id"],
            session_id=row["session_id"],
            start_message_id=(
                int(row["start_message_id"])
                if row["start_message_id"] is not None
                else -1
            ),
            end_message_id=(
                int(row["end_message_id"])
                if row["end_message_id"] is not None
                else -1
            ),
            salience_reason=row["salience_reason"] or "persisted_backlog",
            text=row["text"] or "",
            source_message_ids=tuple(int(item["source_message_id"]) for item in manifest),
        ))
    return result


def extract_fallback_chunk(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    max_chars: int,
) -> Chunk | None:
    """Mint ONE chunk spanning the whole session, for sessions both tiers missed.

    Closes the never-dreamed bug: the salience and baseline tiers only mint
    chunks from USER turns that clear ``min_chars`` or a trigger regex, so a
    session whose user turns are all short (test/WebSocket/diagnostic sessions)
    produces zero chunks in both tiers and the runner skips the per-session
    tail — the digest never runs and ``sessions.digested_prompt_version`` stays
    NULL forever. This fallback gives the digest one real chunk to read (and
    episodes a valid evidence id). Phase-1 triple extraction is deliberately
    never run on fallback chunks — the goal is digest/episode coverage, not
    graph growth from diagnostic noise.

    The chunk spans the first→last user/assistant message with non-empty
    content; its text is the ``role: content`` lines of all such turns,
    truncated to ``max_chars``. Returns None when no user/assistant message
    with non-empty content exists (truly empty sessions still skip the tail).
    """
    rows = [
        row
        for row in conn.execute(
            "SELECT id, role, content FROM messages "
            "WHERE session_id = ? AND role IN ('user', 'assistant') "
            "ORDER BY id",
            (session_id,),
        )
        if row["content"]
    ]
    if not rows:
        return None

    start_id = rows[0]["id"]
    end_id = rows[-1]["id"]
    pieces = [f"{row['role']}: {row['content']}" for row in rows]
    text = "\n".join(pieces)[:max_chars]

    return Chunk(
        id=_chunk_id(session_id, start_id, end_id),
        session_id=session_id,
        start_message_id=start_id,
        end_message_id=end_id,
        salience_reason="short_session_fallback",
        text=text,
        source_message_ids=tuple(int(row["id"]) for row in rows),
    )


def persist_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    from hymem.dreaming.lossless import validate_message_coverage_artifact
    from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION

    for c in chunks:
        conn.execute(
            """
            INSERT OR IGNORE INTO chunks(id, session_id, start_message_id, end_message_id, salience_reason, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (c.id, c.session_id, c.start_message_id, c.end_message_id, c.salience_reason, c.text),
        )
        stored_chunk = conn.execute(
            "SELECT session_id, start_message_id, end_message_id, "
            "text, chunk_kind FROM chunks WHERE id = ?",
            (c.id,),
        ).fetchone()
        if stored_chunk is None or tuple(stored_chunk) != (
            c.session_id, c.start_message_id, c.end_message_id,
            c.text, "extraction",
        ):
            raise RuntimeError("chunk identity collision")
        if not c.source_message_ids:
            continue
        if (
            len(set(c.source_message_ids)) != len(c.source_message_ids)
            or c.source_message_ids[0] != c.start_message_id
            or c.source_message_ids[-1] != c.end_message_id
        ):
            raise ValueError("chunk source manifest does not match its boundaries")
        expected: list[tuple[int, int, str, str, str]] = []
        for ordinal, message_id in enumerate(c.source_message_ids):
            row = conn.execute(
                """
                SELECT chunk_id FROM message_retention_coverage
                WHERE message_id = ? AND source_session_id = ?
                  AND coverage_version = ?
                """,
                (message_id, c.session_id, LOSSLESS_COVERAGE_VERSION),
            ).fetchone()
            if row is None:
                raise ValueError("chunk source lacks ordered coverage")
            frontier = conn.execute(
                "SELECT coverage_message_id FROM sessions WHERE id = ?",
                (c.session_id,),
            ).fetchone()
            if (
                frontier is None
                or frontier["coverage_message_id"] is None
                or message_id > int(frontier["coverage_message_id"])
            ):
                raise ValueError("chunk source exceeds the producer frontier")
            proof = validate_message_coverage_artifact(
                conn, message_id=message_id, chunk_id=row["chunk_id"],
                coverage_version=LOSSLESS_COVERAGE_VERSION,
            )
            if proof.session_id != c.session_id:
                raise ValueError("chunk source belongs to another session")
            expected.append((
                ordinal, message_id, c.session_id, proof.chunk_id,
                LOSSLESS_COVERAGE_VERSION,
            ))
        existing = conn.execute(
            """
            SELECT ordinal, source_message_id, source_session_id,
                   source_coverage_chunk_id, source_coverage_version
            FROM chunk_message_sources WHERE chunk_id = ? ORDER BY ordinal
            """,
            (c.id,),
        ).fetchall()
        expected_tuples = [tuple(item) for item in expected]
        if existing and [tuple(row) for row in existing] != expected_tuples:
            raise RuntimeError("chunk source manifest identity collision")
        if not existing:
            conn.executemany(
                """
                INSERT INTO chunk_message_sources(
                    ordinal, source_message_id, source_session_id,
                    source_coverage_chunk_id, source_coverage_version, chunk_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(*item, c.id) for item in expected],
            )
        conn.execute(
            "UPDATE chunks SET source_manifest_version = ?, "
            "source_manifest_count = ? WHERE id = ?",
            ("claim-source-manifest-v1", len(expected), c.id),
        )


def _chunk_id(session_id: str, start: int, end: int) -> str:
    h = hashlib.sha1(f"{session_id}:{start}:{end}".encode("utf-8")).hexdigest()
    return f"chk_{h}"
