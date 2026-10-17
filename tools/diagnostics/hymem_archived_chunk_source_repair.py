"""Restore provable legacy chunk sources from read-only v31/v33 archives.

This is an explicit repair for chunks stamped source_manifest_unrecoverable
after their raw messages were retained away. It requires the original message
IDs, session ownership, roles, and bytes from a database archive. It does not
infer episode citations, extract claims, or call a provider.
Sessions with any existing coverage, cursor, publication stamp, or staged
consumer work are excluded: inserting older sources behind committed state
can invalidate authority. Current Hermes1 archives have exact sources for
some chunks, but no session passes this conservative application guard.

Preview (no writes):
    python3 tools/diagnostics/hymem_archived_chunk_source_repair.py \
        --db CURRENT.sqlite --archive pre-invalid.sqlite --archive v33.sqlite

Apply to a private clone after reviewing the preview:
    python3 tools/diagnostics/hymem_archived_chunk_source_repair.py \
        --db CLONE.sqlite --archive pre-invalid.sqlite --archive v33.sqlite \
        --apply --expect-candidates EXPECTED_FROM_PREVIEW

Output contains counts and SHA-256 session identifiers only. The archive and
target contain private memory. This tool refuses writes to the known live path.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hymem.core import db
from hymem.config import HyMemConfig
from hymem.dreaming.aggregation_provenance import resolve_cited_episode_sources
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import (
    backfill_all_message_coverage,
    validate_message_coverage_artifact,
)
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
from hymem.dreaming.status import durable_dream_work_status


_MALFORMED_FIELDS = (
    "malformed_source_materialization", "malformed_digests",
    "malformed_profiles", "malformed_facts", "malformed_summaries",
    "quarantined_facts_malformed",
)

# A NULL cursor ID alone does not mean that a session has no committed work.
# In particular, zero-call profile/fact acknowledgements leave prompt stamps,
# and legacy digest migration leaves compatibility watermarks and generation.
_CURSOR_IDS = (
    "coverage_message_id", "source_materialized_message_id",
    "digest_cursor_message_id", "digest_cursor_partial_message_id",
    "profile_cursor_message_id", "profile_cursor_partial_message_id",
    "facts_cursor_message_id", "facts_cursor_partial_message_id",
)
_CURSOR_OFFSETS = (
    "digest_cursor_offset", "profile_cursor_offset", "facts_cursor_offset",
    "auto_summary_message_offset",
)
_PUBLICATION_STAMPS = (
    "source_materialization_config_version", "digested_prompt_version",
    "digested_message_id", "episodes_prompt_version",
    "digest_cursor_prompt_version", "digest_published_generation",
    "digest_published_message_id", "profile_prompt_version",
    "profile_cursor_prompt_version", "profile_published_generation",
    "facts_message_id", "facts_cursor_prompt_version",
    "auto_summary_message_id", "auto_summary_partial_message_id",
    "auto_summary_generation", "auto_summary", "summary_failure_reason",
    "summary", "summary_source",
)
_RETRY_FIELDS = (
    "digest_retry_count", "digest_retry_config_version", "digest_quarantined",
    "profile_retry_count", "profile_retry_config_version", "profile_quarantined",
    "facts_retry_count", "facts_retry_config_version", "facts_quarantined",
    "summary_failure_count",
)
_SESSION_ROW_GUARDS = (
    ("coverage_rows", "message_retention_coverage", "source_session_id"),
    ("digest_staging", "digest_staging", "session_id"),
    ("profile_staging", "profile_staging", "session_id"),
    ("fact_outcomes", "fact_extraction_outcomes", "session_id"),
    ("summary_recovery", "summary_recovery", "session_id"),
    ("profile_publication", "user_profile", "source_session_id"),
    ("legacy_episodes", "episodes", "session_id"),
    ("legacy_procedures", "procedures", "session_id"),
    ("legacy_facts", "narrative_facts", "session_id"),
    ("live_messages", "messages", "session_id"),
)


@dataclass(frozen=True)
class ArchivedMessage:
    id: int
    session_id: str
    role: str
    content: str
    created_at: str | None


@dataclass(frozen=True)
class RecoveryPlan:
    chunks: tuple[Chunk, ...]
    messages: tuple[ArchivedMessage, ...]
    exact_chunks_with_archive_row: int
    candidate_chunks_with_archive_row: int
    exact_chunks: int
    excluded_chunks_with_existing_state: int
    exact_cohort_archived_messages: int
    session_counts: tuple[dict[str, object], ...]

    def counts(self) -> dict[str, object]:
        return {
            "exact_chunks": self.exact_chunks,
            "candidate_chunks": len(self.chunks),
            "excluded_chunks_with_existing_state": self.excluded_chunks_with_existing_state,
            "exact_chunks_with_archive_row": self.exact_chunks_with_archive_row,
            "candidate_chunks_with_archive_row": self.candidate_chunks_with_archive_row,
            "archived_messages_to_cover": len(self.messages),
            "exact_cohort_archived_messages": self.exact_cohort_archived_messages,
            "exact_sessions": len(self.session_counts),
            "candidate_sessions": len({chunk.session_id for chunk in self.chunks}),
            "excluded_sessions": sum(
                bool(row["excluded_existing_state"]) for row in self.session_counts
            ),
            "session_counts": self.session_counts,
        }


def _open_archive(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _plan(conn: sqlite3.Connection,
          archives: tuple[sqlite3.Connection, ...]) -> RecoveryPlan:
    current_schema = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    if current_schema is None or int(current_schema[0]) != 64:
        raise ValueError("unexpected database schema")

    archived_messages: dict[int, ArchivedMessage] = {}
    archived_chunks: dict[str, tuple[object, ...]] = {}
    for archive in archives:
        archive_schema = archive.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        if archive_schema is None or int(archive_schema[0]) not in (31, 33):
            raise ValueError("unexpected archive schema")
        for row in archive.execute(
            "SELECT id,session_id,role,content,created_at FROM messages"
        ):
            message = ArchivedMessage(
                id=int(row["id"]), session_id=str(row["session_id"]),
                role=str(row["role"]), content=str(row["content"]),
                created_at=row["created_at"],
            )
            previous = archived_messages.setdefault(message.id, message)
            if previous != message:
                raise ValueError("archived message identity conflict")
        for row in archive.execute(
            "SELECT id,session_id,start_message_id,end_message_id,text FROM chunks"
        ):
            identity = tuple(row[key] for key in (
                "session_id", "start_message_id", "end_message_id", "text"
            ))
            previous = archived_chunks.setdefault(str(row["id"]), identity)
            if previous != identity:
                raise ValueError("archived chunk identity conflict")

    exact: list[Chunk] = []
    for row in conn.execute(
        "SELECT c.id,c.session_id,c.start_message_id,c.end_message_id,"
        "c.salience_reason,c.text,c.source_manifest_version,"
        "c.source_manifest_count "
        "FROM chunks c JOIN chunk_extraction_terminal_losses loss "
        "ON loss.chunk_id=c.id WHERE c.chunk_kind='extraction' "
        "ORDER BY c.id"
    ):
        if row["source_manifest_version"] is not None or row["source_manifest_count"] is not None:
            raise ValueError("terminal chunk already has a source manifest")
        start, end = row["start_message_id"], row["end_message_id"]
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            continue
        ids = (start,) if start == end else (start, end)
        messages = tuple(archived_messages.get(message_id) for message_id in ids)
        if any(message is None or message.session_id != row["session_id"] for message in messages):
            continue
        if not (
            (len(messages) == 1 and messages[0].role == "user")
            or (len(messages) == 2 and messages[0].role == "assistant"
                and messages[1].role == "user")
        ):
            continue
        if row["text"] != "\n".join(
            f"{message.role}: {message.content}" for message in messages
        ):
            continue
        archived_chunk = archived_chunks.get(str(row["id"]))
        if archived_chunk is not None:
            if archived_chunk != tuple(row[key] for key in (
                "session_id", "start_message_id", "end_message_id", "text"
            )):
                raise ValueError("archived chunk identity differs")
        exact.append(Chunk(
            id=str(row["id"]), session_id=str(row["session_id"]),
            start_message_id=start, end_message_id=end,
            salience_reason=str(row["salience_reason"]), text=str(row["text"]),
            source_message_ids=ids,
        ))

    exact_by_session: dict[str, list[Chunk]] = {}
    for chunk in exact:
        exact_by_session.setdefault(chunk.session_id, []).append(chunk)
    excluded_sessions: set[str] = set()
    session_counts: list[dict[str, object]] = []
    for session_id, group in sorted(exact_by_session.items()):
        state = conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,),
        ).fetchone()
        if state is None:
            raise ValueError("archived session missing in target")
        reasons = []
        if any(state[column] is not None for column in _CURSOR_IDS):
            reasons.append("cursor_id")
        if any(state[column] != 0 for column in _CURSOR_OFFSETS):
            reasons.append("cursor_offset")
        if any(state[column] is not None for column in _PUBLICATION_STAMPS):
            reasons.append("publication_stamp")
        if any(state[column] not in (None, 0) for column in _RETRY_FIELDS):
            reasons.append("retry_state")
        for label, table, column in _SESSION_ROW_GUARDS:
            if conn.execute(
                f"SELECT 1 FROM {table} WHERE {column}=? LIMIT 1",
                (session_id,),
            ).fetchone() is not None:
                reasons.append(label)
        if conn.execute(
            "SELECT 1 FROM procedure_digest_publications p "
            "JOIN procedures r ON r.id=p.procedure_id "
            "WHERE r.session_id=? LIMIT 1", (session_id,),
        ).fetchone() is not None:
            reasons.append("procedure_publication")
        excluded = bool(reasons)
        if excluded:
            excluded_sessions.add(session_id)
        session_counts.append({
            "session_sha256": hashlib.sha256(session_id.encode()).hexdigest(),
            "exact_chunks": len(group),
            "archived_messages": sum(
                message.session_id == session_id
                for message in archived_messages.values()
            ),
            "excluded_existing_state": excluded,
            "exclusion_reasons": reasons,
        })
    candidates = tuple(
        chunk for chunk in exact if chunk.session_id not in excluded_sessions
    )
    sessions = {chunk.session_id for chunk in candidates}
    # Cover all surviving archived turns in each selected session, so the
    # restored producer stream does not omit source rows known to this archive.
    messages = tuple(sorted(
        (message for message in archived_messages.values()
         if message.session_id in sessions),
        key=lambda message: message.id,
    ))
    for message in messages:
        if conn.execute("SELECT 1 FROM messages WHERE id=?", (message.id,)).fetchone():
            raise ValueError("archived message ID already exists in target")
        if conn.execute(
            "SELECT 1 FROM message_retention_coverage WHERE message_id=?",
            (message.id,),
        ).fetchone():
            raise ValueError("archived message ID already has coverage")
    if sessions:
        existing = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE id IN ("
            + ",".join("?" for _ in sessions) + ")", tuple(sessions),
        ).fetchone()[0]
        if existing != len(sessions):
            raise ValueError("archived session missing in target")
    return RecoveryPlan(
        candidates, messages,
        sum(chunk.id in archived_chunks for chunk in exact),
        sum(chunk.id in archived_chunks for chunk in candidates),
        len(exact), len(exact) - len(candidates),
        sum(message.session_id in exact_by_session
            for message in archived_messages.values()),
        tuple(session_counts),
    )


def apply_repair(conn: sqlite3.Connection,
                 archives: tuple[sqlite3.Connection, ...],
                 expected_candidates: int) -> dict[str, object]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        plan = _plan(conn, archives)
        if len(plan.chunks) != expected_candidates or not plan.chunks:
            raise ValueError("candidate count changed")
        # Default-config telemetry is a regression check, not a live-config
        # readiness verdict. Session guards above are the authority boundary.
        status_cfg = HyMemConfig(root=Path("/tmp"))
        health_before = durable_dream_work_status(conn, status_cfg)
        evidence_before = conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0]
        episodes_before = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        for message in plan.messages:
            conn.execute(
                "INSERT INTO messages(id,session_id,role,content,created_at) "
                "VALUES (?,?,?,?,?)",
                (message.id, message.session_id, message.role,
                 message.content, message.created_at),
            )
        for session_id in sorted({chunk.session_id for chunk in plan.chunks}):
            backfill_all_message_coverage(conn, session_id)
        for message in plan.messages:
            coverage = conn.execute(
                "SELECT chunk_id FROM message_retention_coverage "
                "WHERE message_id=? AND source_session_id=? AND coverage_version=?",
                (message.id, message.session_id, LOSSLESS_COVERAGE_VERSION),
            ).fetchone()
            if coverage is None:
                raise RuntimeError("archive coverage was not materialized")
            proof = validate_message_coverage_artifact(
                conn, message_id=message.id, chunk_id=coverage["chunk_id"],
                coverage_version=LOSSLESS_COVERAGE_VERSION,
            )
            if proof.content != message.content or proof.role != message.role:
                raise RuntimeError("archive coverage proof differs")
        for chunk in plan.chunks:
            if conn.execute(
                "DELETE FROM chunk_extraction_terminal_losses WHERE chunk_id=?",
                (chunk.id,),
            ).rowcount != 1:
                raise RuntimeError("terminal marker changed")
            persist_chunks(conn, [chunk])
            proof = resolve_cited_episode_sources(conn, chunk.session_id, [chunk.id])
            if proof is None or tuple(item.message_id for item in proof) != chunk.source_message_ids:
                raise RuntimeError("published chunk source proof differs")
        # Retain the canonical lossless coverage, not a second copy of old raw
        # message content. The whole operation is one transaction.
        for message in plan.messages:
            if conn.execute("DELETE FROM messages WHERE id=?", (message.id,)).rowcount != 1:
                raise RuntimeError("temporary archive message was not removed")
        for chunk in plan.chunks:
            if resolve_cited_episode_sources(conn, chunk.session_id, [chunk.id]) is None:
                raise RuntimeError("post-retention chunk proof differs")
        health_after = durable_dream_work_status(conn, status_cfg)
        if any(health_after[key] > health_before[key] for key in _MALFORMED_FIELDS):
            raise RuntimeError("repair would increase malformed work")
        if (
            conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0]
            != evidence_before
            or conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            != episodes_before
        ):
            raise RuntimeError("repair changed derived knowledge counts")
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("repair would leave foreign-key violations")
        conn.execute("COMMIT")
        return plan.counts()
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True, action="append")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expect-candidates", type=int)
    args = parser.parse_args()
    if any(
        args.db.resolve() == path.resolve()
        or (args.db.exists() and path.exists() and args.db.samefile(path))
        for path in args.archive
    ):
        parser.error("target and archive must differ")
    if args.apply and (args.expect_candidates is None or args.expect_candidates < 1):
        parser.error("--apply requires a positive --expect-candidates")
    live_path = Path("/opt/stacks/hermes/instance1/home/.hermes/hymem.sqlite")
    # resolve() catches symlinks; samefile() also catches hardlinks to the live
    # inode. Never open a live alias writable, even for a guard-only rehearsal.
    aliases_live = args.db.resolve() == live_path.resolve()
    if args.db.exists() and live_path.exists():
        aliases_live = aliases_live or args.db.samefile(live_path)
    if args.apply and aliases_live:
        parser.error("this tool is private-clone-only; live target is forbidden")
    try:
        with ExitStack() as stack:
            archives = tuple(
                stack.enter_context(closing(_open_archive(path)))
                for path in args.archive
            )
            if args.apply:
                with closing(db.connect(args.db)) as conn:
                    result = apply_repair(conn, archives, args.expect_candidates)
            else:
                with db.read_snapshot(args.db) as conn:
                    result = _plan(conn, archives).counts()
        print(json.dumps({"status": "applied" if args.apply else "preview", **result}, sort_keys=True))
    except (sqlite3.Error, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"archived source repair failed: {type(exc).__name__}") from None


if __name__ == "__main__":
    main()
