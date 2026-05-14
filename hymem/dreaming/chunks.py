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
            )
        )

    return chunks


def persist_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    for c in chunks:
        conn.execute(
            """
            INSERT OR IGNORE INTO chunks(id, session_id, start_message_id, end_message_id, salience_reason, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (c.id, c.session_id, c.start_message_id, c.end_message_id, c.salience_reason, c.text),
        )


def _chunk_id(session_id: str, start: int, end: int) -> str:
    h = hashlib.sha1(f"{session_id}:{start}:{end}".encode("utf-8")).hexdigest()
    return f"chk_{h}"
