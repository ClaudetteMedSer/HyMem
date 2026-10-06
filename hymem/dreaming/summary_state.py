"""Independent, source-relative health of published rolling summaries.

An item cursor is not a summary acknowledgement. These helpers never advance
item state, never infer coverage from a truthy string, and never publish a
partial or gapped summary as current. In-progress summary recovery belongs in
private staging; ``mark_summary_current`` accepts only the complete item tail.
"""
from __future__ import annotations

import re
import sqlite3

from hymem.dreaming.lossless import lossless_cursor_is_valid
from hymem.dreaming.summary import persist_auto_session_summary

SUMMARY_STATE_VERSION = "independent-summary-frontier-v1"
SUMMARY_FAILURE_REASONS = frozenset({
    "summary_output_cap", "summary_shape_failure", "summary_validation_failure",
    "parse_failure", "output_truncated", "shape_failure", "prior_summary_gap",
})
_REASON = re.compile(r"[a-z0-9_]{1,80}\Z")
_MAX_INTEGER = (1 << 63) - 1


def _generation_valid(value: object) -> bool:
    from hymem.dreaming.digest import digest_generation_is_recognized
    return digest_generation_is_recognized(value)


def _integer(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and minimum <= value <= _MAX_INTEGER


def _reason_valid(value: object) -> bool:
    return (isinstance(value, str) and value in SUMMARY_FAILURE_REASONS
            and _REASON.fullmatch(value) is not None)


def classify_summary_state(
    conn: sqlite3.Connection, session_id: str, *, require_source_tail: bool = True,
) -> dict[str, bool]:
    """Classify published state against exact lossless source and item frontier.

    ``missing`` means source exists but no automatic text has been published;
    an operator/legacy summary is preserved but is not automatic coverage.
    ``degraded`` includes a missing/stale automatic summary or a recorded
    failure. ``malformed`` is separate corruption, not ordinary model refusal.
    A genuinely source-empty session has no summary work and is healthy.

    Health readers use the default source-tail requirement. A digest consuming
    a normal newly appended tail may pass ``require_source_tail=False`` only
    to check continuity of its prior summary against the published ITEM base;
    pending new input is not itself a hole in that previously accepted base.
    """
    if type(require_source_tail) is not bool:
        raise ValueError("summary classification source-tail policy must be boolean")
    # The same session-scoped ordered stream used by coverage materialization
    # includes every role. Read its raw suffix in the SAME statement/snapshot
    # as the coverage marker; a raw append is stale even before its artifact
    # exists, whereas pruning already-covered raw rows must not undo proof.
    row = conn.execute(
        "SELECT s.*, EXISTS (SELECT 1 FROM messages m WHERE m.session_id=s.id "
        "AND m.id>COALESCE(s.coverage_message_id,-1)) AS _summary_raw_tail_pending "
        "FROM sessions s WHERE s.id=?", (session_id,),
    ).fetchone()
    if row is None:
        return dict(summary_healthy=False, degraded=False, missing=False, malformed=True)
    text = row["auto_summary"]
    summary_generation = row["auto_summary_generation"]
    published_generation = row["digest_published_generation"]
    published_id = row["digest_published_message_id"]
    before_id = row["auto_summary_message_id"]
    partial_id = row["auto_summary_partial_message_id"]
    offset = row["auto_summary_message_offset"]
    reason, count = row["summary_failure_reason"], row["summary_failure_count"]
    has_source = row["coverage_message_id"] is not None
    raw_tail_pending = bool(require_source_tail and row["_summary_raw_tail_pending"])
    summary_position = (before_id, partial_id, offset)

    malformed = bool(
        (text is not None and (not isinstance(text, str) or len(text) > 500))
        or (row["coverage_message_id"] is not None
            and not _integer(row["coverage_message_id"], minimum=1))
        or (summary_generation is not None and not _generation_valid(summary_generation))
        or (published_id is not None and not _integer(published_id, minimum=1))
        or (published_id is not None and not _generation_valid(published_generation))
        or not _integer(count)
        or (reason is not None and not _reason_valid(reason))
        or ((reason is None) != (count == 0))
        or (has_source and not lossless_cursor_is_valid(
            conn, session_id, row["coverage_message_id"], None, 0,
        ))
        or not lossless_cursor_is_valid(conn, session_id, *summary_position)
        or (published_id is not None and not lossless_cursor_is_valid(
            conn, session_id, published_id, None, 0,
        ))
        or (text is None and (summary_generation is not None
                            or summary_position != (None, None, 0)))
        or (summary_generation is not None and published_id is None)
    )
    if _integer(published_id, minimum=1) and (
        (_integer(before_id, minimum=1) and before_id > published_id)
        or (_integer(partial_id, minimum=1) and partial_id > published_id)
    ):
        malformed = True
    missing = bool(not malformed and (has_source or raw_tail_pending) and text is None)
    empty = bool(
        not has_source and not raw_tail_pending and text is None and published_id is None
        and summary_generation is None and published_generation is None
        and summary_position == (None, None, 0) and reason is None and count == 0
    )
    healthy = bool(not malformed and (empty or (
        text is not None and _generation_valid(summary_generation)
        and summary_generation == published_generation
        and published_id is not None
        and not raw_tail_pending
        and (not require_source_tail or published_id == row["coverage_message_id"])
        and summary_position == (published_id, None, 0)
        and reason is None and count == 0
    )))
    return dict(summary_healthy=healthy, degraded=bool(not malformed and not healthy),
                missing=missing, malformed=malformed)


def record_summary_failure(conn: sqlite3.Connection, session_id: str, reason: str) -> None:
    """Record bounded metadata, preserving the accepted text and frontier.

    Call inside the same transaction that publishes/stages the corresponding
    validated items. Reasons are machine labels, never raw model/error text.
    The first unresolved reason remains visible until genuine recovery.
    """
    if not conn.in_transaction or not _reason_valid(reason):
        raise RuntimeError("summary failure requires a transaction and bounded reason")
    row = conn.execute(
        "SELECT summary_failure_count,summary_failure_reason FROM sessions WHERE id=?", (session_id,),
    ).fetchone()
    if (row is None or not _integer(row["summary_failure_count"])
            or (row["summary_failure_reason"] is not None
                and not _reason_valid(row["summary_failure_reason"]))
            or ((row["summary_failure_reason"] is None) != (row["summary_failure_count"] == 0))):
        raise RuntimeError("summary failure state is missing or malformed")
    conn.execute(
        "UPDATE sessions SET summary_failure_count=?,"
        "summary_failure_reason=COALESCE(summary_failure_reason,?) WHERE id=?",
        (min(_MAX_INTEGER, row["summary_failure_count"] + 1), reason, session_id),
    )


def mark_summary_current(
    conn: sqlite3.Connection, session_id: str, summary: str, *,
    generation: str, covered_message_id: int,
) -> None:
    """Publish an already-validated COMPLETE summary in the owning transaction.

    The caller must have proved its contiguous recovery chain. This helper
    additionally fences the exact currently published item generation/frontier
    and lossless source; it cannot bless a future/partial/foreign cursor.
    """
    if (not conn.in_transaction or not isinstance(summary, str) or len(summary) > 500
            or not _generation_valid(generation)
            or not _integer(covered_message_id, minimum=1)):
        raise RuntimeError("summary publication requires a bounded complete fenced result")
    row = conn.execute(
        "SELECT digest_published_generation,digest_published_message_id FROM sessions WHERE id=?",
        (session_id,),
    ).fetchone()
    if (row is None or row["digest_published_generation"] != generation
            or row["digest_published_message_id"] != covered_message_id
            or not lossless_cursor_is_valid(conn, session_id, covered_message_id, None, 0)):
        raise RuntimeError("summary publication does not match source-proved item publication")
    persist_auto_session_summary(
        conn, session_id, summary, covered_message_id=covered_message_id,
    )
    conn.execute(
        "UPDATE sessions SET auto_summary_generation=?,summary_failure_reason=NULL,"
        "summary_failure_count=0 WHERE id=?", (generation, session_id),
    )
