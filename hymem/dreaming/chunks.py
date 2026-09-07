from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass

from hymem.extraction.contract import extraction_cache_key

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


@dataclass(frozen=True)
class _SourceMessage:
    id: int
    role: str
    content: str


_UNBOUNDED_SOURCE_FRONTIER = object()


TERMINAL_SOURCE_LOSS_REASON = "source_manifest_unrecoverable"
BASELINE_SALIENCE_REASON = "baseline_user_turn"
# Durable acknowledgement written only after both current Phase-1 source
# builders have walked one exact lossless frontier and their candidates have
# been persisted.  Bump this identity whenever that combined producer contract
# changes; old acknowledgements then reopen without guessing from chunk rows.
SOURCE_MATERIALIZATION_POLICY_VERSION = "dream-source-materialization-v1"


def source_materialization_config_version(*, min_chars: int) -> str:
    """Bind a source-producer acknowledgement to all runtime classification.

    The two builders' union is intentionally lossless for non-blank USER turns,
    but ``min_chars`` changes high-vs-baseline ownership and therefore bounded
    scheduling.  Persist it even though chunk ids are source-coordinate based.
    The policy constant binds code-owned regex/framing/manifest behavior.
    """
    if isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 0:
        raise ValueError("source materialization min_chars must be nonnegative")
    return (
        f"{SOURCE_MATERIALIZATION_POLICY_VERSION}|"
        f"coverage=dream-lossless-message-v1|"
        f"manifest=claim-source-manifest-v1|min-chars={min_chars}"
    )


def _high_salience_flags(content: str, min_chars: int) -> tuple[bool, bool]:
    """Return trigger/length eligibility without treating blanks as content."""
    is_trigger = bool(
        _CORRECTION_PATTERNS.search(content)
        or _PREFERENCE_PATTERNS.search(content)
    )
    is_substantive = bool(content.strip()) and len(content) >= min_chars
    return is_trigger, is_substantive


def _session_source_messages(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    through_message_id: int | None | object = _UNBOUNDED_SOURCE_FRONTIER,
) -> Iterator[_SourceMessage]:
    """Yield the session stream from durable coverage when it is available.

    The dream runner completes the lossless producer walk before invoking any
    chunk builder. Reading that validated stream makes both scheduling classes
    exactly reconstructible after raw-message retention. The raw fallback is
    retained for direct builder callers and migration-era stores whose producer
    frontier has not been established yet; ``persist_chunks`` still refuses to
    publish a source manifest until exact coverage exists.  The dream runner
    passes its captured producer frontier explicitly.  That bound excludes a
    turn appended after the cycle's coverage walk even if a concurrent public
    ingest has already advanced the live coverage frontier.
    """
    bounded = through_message_id is not _UNBOUNDED_SOURCE_FRONTIER
    if bounded and through_message_id is not None and (
        isinstance(through_message_id, bool)
        or not isinstance(through_message_id, int)
        or through_message_id < 1
    ):
        raise ValueError("source frontier must be a positive message id or None")
    if bounded and through_message_id is None:
        return

    frontier = conn.execute(
        "SELECT coverage_message_id FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if frontier is not None and frontier["coverage_message_id"] is not None:
        from hymem.dreaming.lossless import covered_messages_after

        effective_frontier = int(frontier["coverage_message_id"])
        if bounded:
            effective_frontier = min(effective_frontier, int(through_message_id))
        cursor: int | None = None
        while True:
            page = covered_messages_after(
                conn,
                session_id,
                cursor,
                limit=256,
                through_message_id=effective_frontier,
            )
            if not page:
                break
            for row in page:
                yield _SourceMessage(
                    id=int(row.message_id),
                    role=row.role,
                    content=row.content or "",
                )
            cursor = int(page[-1].message_id)
            if len(page) < 256:
                break
        # Direct builder callers can append between producer walks. Preserve
        # their historical read behavior by adding only the raw tail beyond
        # the proven frontier. A bounded runner walk must not consume that tail:
        # it belongs to the next producer acknowledgement.
        if bounded:
            return
        for row in conn.execute(
            "SELECT id, role, content FROM messages "
            "WHERE session_id = ? AND id > ? ORDER BY id",
            (session_id, int(frontier["coverage_message_id"])),
        ):
            yield _SourceMessage(
                id=int(row["id"]),
                role=row["role"],
                content=row["content"] or "",
            )
        return

    for row in conn.execute(
        "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ):
        yield _SourceMessage(
            id=int(row["id"]),
            role=row["role"],
            content=row["content"] or "",
        )


def recover_legacy_chunk_source_manifests(
    conn: sqlite3.Connection,
    session_id: str,
) -> int:
    """Recover recognized legacy builder output from exact source artifacts.

    This is the runtime counterpart to migration v40.  It first covers every
    surviving raw message (including rows below an old/sparse frontier), then
    accepts only the historical one-user or assistant+user byte shape.  The
    normal ``persist_chunks`` writer performs the final proof validation and
    publication, so a numeric range or similar-looking prose is never enough.
    """
    from hymem.dreaming.lossless import (
        backfill_all_message_coverage,
        validate_message_coverage_artifact,
    )
    from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION

    pending = conn.execute(
        "SELECT id,session_id,start_message_id,end_message_id,salience_reason,text "
        "FROM chunks WHERE session_id=? AND chunk_kind='extraction' "
        "AND COALESCE(salience_reason, '') <> 'short_session_fallback' "
        "AND source_manifest_version IS NULL AND source_manifest_count IS NULL "
        "ORDER BY id",
        (session_id,),
    ).fetchall()
    if not pending:
        return 0

    backfill_all_message_coverage(conn, session_id)
    recovered = 0
    for stored in pending:
        start = stored["start_message_id"]
        end = stored["end_message_id"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 1
            or end < start
        ):
            continue
        source_ids = [int(start)]
        if int(end) != int(start):
            source_ids.append(int(end))
        proofs = []
        for message_id in source_ids:
            proof_row = conn.execute(
                "SELECT chunk_id FROM message_retention_coverage "
                "WHERE message_id=? AND source_session_id=? "
                "AND coverage_version=?",
                (message_id, session_id, LOSSLESS_COVERAGE_VERSION),
            ).fetchone()
            if proof_row is None:
                proofs = []
                break
            try:
                proof = validate_message_coverage_artifact(
                    conn,
                    message_id=message_id,
                    chunk_id=proof_row["chunk_id"],
                    coverage_version=LOSSLESS_COVERAGE_VERSION,
                )
            except (RuntimeError, TypeError, ValueError):
                proofs = []
                break
            proofs.append(proof)
        if len(proofs) == 1:
            roles_valid = proofs[0].role == "user"
            expected_text = f"user: {proofs[0].content}"
        elif len(proofs) == 2:
            roles_valid = proofs[0].role == "assistant" and proofs[1].role == "user"
            expected_text = (
                f"assistant: {proofs[0].content}\nuser: {proofs[1].content}"
            )
        else:
            continue
        if not roles_valid or stored["text"] != expected_text:
            continue
        persist_chunks(
            conn,
            [Chunk(
                id=stored["id"],
                session_id=stored["session_id"],
                start_message_id=int(start),
                end_message_id=int(end),
                salience_reason=stored["salience_reason"],
                text=stored["text"],
                source_message_ids=tuple(source_ids),
            )],
        )
        recovered += 1
    return recovered


def record_unrecoverable_chunk_losses(
    conn: sqlite3.Connection,
    session_id: str | None = None,
) -> int:
    """Durably classify extraction chunks that have no admissible source.

    Callers invoke this only after every available exact recovery path has run:
    migration v47 first materializes surviving raw-message coverage and retries
    the conservative v40 builder reconstruction; the dream runner first
    persists all chunks reproducible from the live session.  No chunk prose or
    numeric range is treated as provenance.  The resulting state is global to
    the chunk, so prompt changes cannot reopen impossible work.
    """
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='chunk_extraction_terminal_losses'"
    ).fetchone() is None:
        return 0
    scope = ""
    params: list[object] = [TERMINAL_SOURCE_LOSS_REASON]
    if session_id is not None:
        scope = "AND c.session_id = ?"
        params.append(session_id)
    cursor = conn.execute(
        f"""
        INSERT OR IGNORE INTO chunk_extraction_terminal_losses(chunk_id, reason)
        SELECT c.id, ?
        FROM chunks c
        WHERE c.chunk_kind = 'extraction'
          AND COALESCE(c.salience_reason, '') <> 'short_session_fallback'
          AND c.source_manifest_version IS NULL
          AND c.source_manifest_count IS NULL
          {scope}
        """,
        tuple(params),
    )
    return max(0, int(cursor.rowcount or 0))


def extract_high_salience_chunks(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    min_chars: int,
    through_message_id: int | None | object = _UNBOUNDED_SOURCE_FRONTIER,
) -> list[Chunk]:
    """Walk the session messages and produce chunks worth running extraction on.

    Strategy: a sliding pair of (preceding assistant turn, user turn). When the
    user turn matches a trigger or is long enough on its own, we mint a chunk
    spanning the pair so the LLM sees what was being corrected.
    """
    chunks: list[Chunk] = []
    last_assistant: _SourceMessage | None = None

    for row in _session_source_messages(
        conn, session_id, through_message_id=through_message_id
    ):
        role = row.role
        content = row.content
        if role == "assistant":
            last_assistant = row
            continue
        if role != "user":
            continue

        is_trigger, is_substantive = _high_salience_flags(content, min_chars)
        if not (is_trigger or is_substantive):
            continue

        start_id = last_assistant.id if last_assistant is not None else row.id
        end_id = row.id
        pieces = []
        if last_assistant is not None:
            pieces.append(f"assistant: {last_assistant.content}")
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
                    [int(last_assistant.id)] if last_assistant is not None else []
                ) + (int(row.id),),
            )
        )

    return chunks


def extract_baseline_chunks(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    prompt_version: str,
    limit: int | None,
    min_chars: int,
    max_attempts: int = 0,
    phase1_generation_key: str | None = None,
    exclude_ids: set[str] | None = None,
    through_message_id: int | None | object = _UNBOUNDED_SOURCE_FRONTIER,
) -> list[Chunk]:
    """Build the non-salient user-turn backstop, newest first.

    The high-salience producer already includes every trigger match and every
    user turn at least ``min_chars`` long.  Baseline must therefore select the
    *remaining* non-blank turns; applying the same minimum here would make it a
    strict subset of the high tier and permanently miss short facts.

    Input comes from the validated lossless stream, not the optional raw
    ``messages`` rows.  The runner materializes that stream before calling this
    function, so a budget-blocked baseline chunk can be rebuilt byte-for-byte
    after raw-message or extraction-chunk retention.  ``limit=None`` is used by
    the runner's local durability pass; actual LLM scheduling remains bounded
    separately by ``dream_baseline_budget`` and the shared extraction budgets.

    Whitespace-only content is the only safe noise exclusion.  Short replies
    such as "Yes." can confirm the preceding assistant statement and must stay
    eligible; language-specific acknowledgement stoplists would create silent
    memory holes.
    """
    prompt_version = extraction_cache_key(prompt_version)
    if limit is not None and limit <= 0:
        return []

    candidates: list[Chunk] = []
    last_assistant: _SourceMessage | None = None
    for row in _session_source_messages(
        conn, session_id, through_message_id=through_message_id
    ):
        role = row.role
        content = row.content
        if role == "assistant":
            last_assistant = row
            continue
        if role != "user":
            continue
        if not content.strip():
            continue

        is_trigger, is_substantive = _high_salience_flags(content, min_chars)
        if is_trigger or is_substantive:
            # This exact source range belongs to the high-priority tier.
            continue

        start_id = last_assistant.id if last_assistant is not None else row.id
        end_id = row.id
        chunk_id = _chunk_id(session_id, start_id, end_id)

        pieces = []
        if last_assistant is not None:
            pieces.append(f"assistant: {last_assistant.content}")
        pieces.append(f"user: {content}")
        candidates.append(
            Chunk(
                id=chunk_id,
                session_id=session_id,
                start_message_id=start_id,
                end_message_id=end_id,
                salience_reason=BASELINE_SALIENCE_REASON,
                text="\n".join(pieces),
                source_message_ids=tuple(
                    [int(last_assistant.id)] if last_assistant is not None else []
                ) + (int(row.id),),
            )
        )

    # Newest first, then drop already-processed and cap to limit.
    candidates.reverse()
    result: list[Chunk] = []
    for chunk in candidates:
        if chunk.id in (exclude_ids or ()):
            continue
        if phase1_generation_key is None:
            # Prompt-only/legacy rows cannot prove who produced them.  The
            # compatibility call shape remains accepted, but reuse fails
            # closed until the caller supplies an exact generation key.
            already = None
        else:
            already = conn.execute(
                "SELECT 1 FROM current_phase1_publications publication "
                "WHERE publication.chunk_id=? "
                "AND publication.prompt_version=? "
                "AND publication.phase1_generation_key=?",
                (chunk.id, prompt_version, phase1_generation_key),
            ).fetchone()
        if already:
            continue
        if chunk_extraction_is_quarantined(
            conn,
            chunk.id,
            prompt_version=prompt_version,
            max_attempts=max_attempts,
            phase1_generation_key=phase1_generation_key,
        ):
            continue
        result.append(chunk)
        if limit is not None and len(result) >= limit:
            break
    return result


def chunk_extraction_is_quarantined(
    conn: sqlite3.Connection,
    chunk_id: str,
    *,
    prompt_version: str,
    max_attempts: int,
    phase1_generation_key: str | None = None,
) -> bool:
    """Whether a failed chunk has exhausted the current retry policy.

    Quarantine is derived from the auditable attempt row instead of being
    represented by a false ``processed_chunks`` success. Changing the prompt
    version, raising the bound, or setting it to zero immediately makes the
    chunk eligible again without destructive bookkeeping.
    """
    prompt_version = extraction_cache_key(prompt_version)
    if max_attempts <= 0:
        return False
    if phase1_generation_key is None:
        return False
    else:
        row = conn.execute(
            "SELECT attempts FROM chunk_extraction_attempts "
            "WHERE chunk_id=? AND prompt_version=? "
            "AND phase1_generation_key=?",
            (chunk_id, prompt_version, phase1_generation_key),
        ).fetchone()
    return bool(row is not None and int(row["attempts"]) >= int(max_attempts))


def has_pending_persisted_chunks(
    conn: sqlite3.Connection,
    *,
    prompt_version: str,
    max_attempts: int = 0,
    phase1_generation_key: str | None = None,
    session_ids: tuple[str, ...] | None = None,
    included_chunk_ids: set[str] | frozenset[str] | None = None,
    excluded_chunk_ids: set[str] | frozenset[str] | None = None,
    included_salience_reasons: tuple[str, ...] = (),
    excluded_salience_reasons: tuple[str, ...] = (),
) -> bool:
    """Return whether one genuinely schedulable extraction chunk remains.

    This is the zero-model-call completion probe used after a Phase-1 budget
    reaches exactly zero. Its eligibility predicates intentionally mirror the
    durable backlog loader: successful, quarantined, terminal-source-loss,
    unverifiable-manifest, coverage-only, and short-session fallback rows do
    not make a completed budget look exhausted. Optional session/id/salience
    filters let the runner apply the exact invocation scope and its current
    (rather than historically stored) high-vs-baseline classification.
    """

    if included_salience_reasons and excluded_salience_reasons:
        raise ValueError(
            "included and excluded salience reasons are mutually exclusive"
        )
    prompt_version = extraction_cache_key(prompt_version)
    if phase1_generation_key is None:
        processed = "SELECT 0"
        params: list[object] = []
    else:
        processed = (
            "SELECT 1 FROM current_phase1_publications publication "
            "WHERE publication.chunk_id=c.id "
            "AND publication.prompt_version=? "
            "AND publication.phase1_generation_key=?"
        )
        params = [prompt_version, phase1_generation_key]

    quarantine = ""
    if max_attempts > 0 and phase1_generation_key is not None:
        quarantine = (
            "AND NOT EXISTS ("
            " SELECT 1 FROM chunk_extraction_attempts a"
            " WHERE a.chunk_id = c.id AND a.prompt_version = ?"
            + " AND a.phase1_generation_key = ?"
            + " AND a.attempts >= ?)"
        )
        params.append(prompt_version)
        params.append(phase1_generation_key)
        params.append(int(max_attempts))

    salience = ""
    if included_salience_reasons:
        reasons = tuple(sorted(set(included_salience_reasons)))
        salience = (
            f"AND COALESCE(c.salience_reason, '') IN "
            f"({','.join('?' * len(reasons))})"
        )
        params.extend(reasons)
    elif excluded_salience_reasons:
        reasons = tuple(sorted(set(excluded_salience_reasons)))
        salience = (
            f"AND COALESCE(c.salience_reason, '') NOT IN "
            f"({','.join('?' * len(reasons))})"
        )
        params.extend(reasons)

    terminal = ""
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='chunk_extraction_terminal_losses'"
    ).fetchone() is not None:
        terminal = (
            "AND NOT EXISTS ("
            " SELECT 1 FROM chunk_extraction_terminal_losses loss"
            " WHERE loss.chunk_id = c.id"
            ")"
        )

    base_sql = f"""
        SELECT c.id, c.session_id
        FROM chunks c
        WHERE c.chunk_kind = 'extraction'
          AND COALESCE(c.salience_reason, '') <> 'short_session_fallback'
          AND c.source_manifest_version = 'claim-source-manifest-v1'
          AND c.source_manifest_count > 0
          AND NOT EXISTS (
              {processed}
          )
          {quarantine}
          {salience}
          {terminal}
    """
    base_params = tuple(params)
    excluded_ids = frozenset(excluded_chunk_ids or ())
    scoped_sessions = (
        None if session_ids is None else frozenset(session_ids)
    )
    if scoped_sessions is not None and not scoped_sessions:
        return False

    # Direct-id probes are split below SQLite's conservative host-parameter
    # limit. Current high/baseline candidates already carry exact durable ids,
    # so this avoids scanning an unrelated backlog just to classify their tier.
    if included_chunk_ids is not None:
        included_ids = sorted(
            frozenset(included_chunk_ids) - excluded_ids
        )
        if not included_ids:
            return False
        batch_size = 400
        for offset in range(0, len(included_ids), batch_size):
            batch = included_ids[offset:offset + batch_size]
            rows = conn.execute(
                base_sql
                + f" AND c.id IN ({','.join('?' * len(batch))})",
                base_params + tuple(batch),
            ).fetchall()
            if any(
                scoped_sessions is None
                or row["session_id"] in scoped_sessions
                for row in rows
            ):
                return True
        return False

    # Exclusion sets cannot be safely split across independent NOT IN queries.
    # Walk matching ids in bounded keyset pages instead, returning on the first
    # row outside the current-tier exclusion set. Explicit replay/debug scopes
    # are queried one session at a time, avoiding both global false positives
    # and unbounded IN parameter lists.
    scopes: tuple[str | None, ...] = (
        (None,)
        if scoped_sessions is None
        else tuple(sorted(scoped_sessions))
    )
    batch_size = 256
    for session_id in scopes:
        after_id: str | None = None
        while True:
            scope_sql = ""
            scope_params: tuple[object, ...] = ()
            if session_id is not None:
                scope_sql += " AND c.session_id = ?"
                scope_params += (session_id,)
            if after_id is not None:
                scope_sql += " AND c.id > ?"
                scope_params += (after_id,)
            rows = conn.execute(
                base_sql + scope_sql + " ORDER BY c.id LIMIT ?",
                base_params + scope_params + (batch_size,),
            ).fetchall()
            if any(row["id"] not in excluded_ids for row in rows):
                return True
            if len(rows) < batch_size:
                break
            after_id = str(rows[-1]["id"])
    return False


def load_pending_persisted_chunks(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    prompt_version: str,
    limit: int,
    max_attempts: int = 0,
    phase1_generation_key: str | None = None,
    exclude_ids: set[str] | None = None,
    excluded_salience_reasons: tuple[str, ...] = (),
) -> list[Chunk]:
    """Load already-durable extraction chunks still owed this prompt salt.

    Rebuilding candidates exclusively from ``messages`` made a prompt bump
    impossible to replay after opt-in raw retention. This backlog reader uses
    the stored extraction artifact itself, excludes coverage storage and
    quarantined failures, and lets the runner bound work with its normal
    budget.
    """
    prompt_version = extraction_cache_key(prompt_version)
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
    if phase1_generation_key is None:
        processed = "SELECT 0"
        params: list[object] = [session_id]
    else:
        processed = (
            "SELECT 1 FROM current_phase1_publications publication "
            "WHERE publication.chunk_id=c.id "
            "AND publication.prompt_version=? "
            "AND publication.phase1_generation_key=?"
        )
        params = [session_id, prompt_version, phase1_generation_key]
    if excluded:
        exclusion = f"AND c.id NOT IN ({','.join('?' * len(excluded))})"
        params.extend(excluded)
    quarantine = ""
    if max_attempts > 0 and phase1_generation_key is not None:
        quarantine = (
            "AND NOT EXISTS ("
            " SELECT 1 FROM chunk_extraction_attempts a"
            " WHERE a.chunk_id = c.id AND a.prompt_version = ?"
            + " AND a.phase1_generation_key = ?"
            + " AND a.attempts >= ?)"
        )
        params.append(prompt_version)
        params.append(phase1_generation_key)
        params.append(int(max_attempts))
    salience = ""
    if excluded_salience_reasons:
        reasons = tuple(sorted(set(excluded_salience_reasons)))
        salience += (
            f"AND COALESCE(c.salience_reason, '') NOT IN "
            f"({','.join('?' * len(reasons))})\n"
        )
        params.extend(reasons)
    terminal = ""
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='chunk_extraction_terminal_losses'"
    ).fetchone() is not None:
        terminal = (
            "AND NOT EXISTS ("
            " SELECT 1 FROM chunk_extraction_terminal_losses loss"
            " WHERE loss.chunk_id = c.id"
            ")"
        )
    params.append(int(limit))
    rows = conn.execute(
        f"""
        SELECT c.id, c.session_id, c.start_message_id, c.end_message_id,
               c.salience_reason, c.text
        FROM chunks c
        WHERE c.session_id = ?
          AND c.chunk_kind = 'extraction'
          AND COALESCE(c.salience_reason, '') <> 'short_session_fallback'
          AND c.source_manifest_version = 'claim-source-manifest-v1'
          AND c.source_manifest_count > 0
          AND NOT EXISTS (
              {processed}
          )
          {exclusion}
          {quarantine}
          {salience}
          {terminal}
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
    """Mint one legacy whole-session compatibility chunk.

    The runner no longer uses this helper: the independent lossless stream
    drives session tails, while the baseline tier now covers short non-blank
    user turns. It remains available for old integrations and tests that need
    the historical ``short_session_fallback`` artifact. Phase-1 triple
    extraction deliberately excludes fallback chunks.

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
        # Chunk ids deliberately bind source coordinates rather than the
        # scheduling tier. A salience threshold/policy change can therefore
        # reproduce the same exact source artifact in the other tier. Publish
        # that current classification only after its identity and manifest
        # have both validated, so the producer acknowledgement cannot claim a
        # policy that the persisted scheduler metadata does not reflect.
        conn.execute(
            "UPDATE chunks SET salience_reason=? WHERE id=?",
            (c.salience_reason, c.id),
        )


def _chunk_id(session_id: str, start: int, end: int) -> str:
    h = hashlib.sha1(f"{session_id}:{start}:{end}".encode("utf-8")).hexdigest()
    return f"chk_{h}"
