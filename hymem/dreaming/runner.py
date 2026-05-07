from __future__ import annotations

import contextlib
import os
import socket
import sqlite3
from dataclasses import dataclass

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import phase1, phase2, phase3
from hymem.dreaming.chunks import extract_high_salience_chunks, persist_chunks
from hymem.extraction.llm import LLMClient


@dataclass
class DreamReport:
    sessions_processed: int = 0
    chunks_seen: int = 0
    chunks_processed: int = 0
    triples_extracted: int = 0
    markers_extracted: int = 0
    skipped_locked: bool = False


def run_dreaming(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    llm: LLMClient,
    *,
    session_ids: list[str] | None = None,
) -> DreamReport:
    """Run all three dreaming phases. Holds an advisory lock so concurrent runs
    bail out instead of double-processing.
    """
    report = DreamReport()
    holder = f"{socket.gethostname()}:{os.getpid()}"

    if not _acquire_lock(conn, holder):
        report.skipped_locked = True
        return report

    try:
        target_sessions = session_ids or _all_sessions(conn)

        for session_id in target_sessions:
            report.sessions_processed += 1
            chunks = extract_high_salience_chunks(
                conn, session_id, min_chars=cfg.salience_min_chars
            )
            report.chunks_seen += len(chunks)
            if not chunks:
                continue

            with core_db.transaction(conn):
                persist_chunks(conn, chunks)

            for chunk in chunks:
                with core_db.transaction(conn):
                    triples, markers = phase1.process_chunk(
                        conn, chunk, llm, prompt_version=cfg.prompt_version
                    )
                    if triples or markers:
                        report.chunks_processed += 1
                        report.triples_extracted += len(triples)
                        report.markers_extracted += len(markers)

        with core_db.transaction(conn):
            phase2.consolidate_profile(conn, cfg)
            phase2.consolidate_insights(conn, cfg)

        with core_db.transaction(conn):
            phase3.decay(conn, cfg)
            phase2.consolidate_insights(conn, cfg)  # refresh after decay

        return report
    finally:
        _release_lock(conn, holder)


_LOCK_TTL_SECONDS = 300


def _acquire_lock(conn: sqlite3.Connection, holder: str) -> bool:
    try:
        conn.execute(
            "INSERT INTO run_lock(name, acquired_at, holder) VALUES ('dreaming', CURRENT_TIMESTAMP, ?)",
            (holder,),
        )
        return True
    except sqlite3.IntegrityError:
        pass

    # Lock exists — check whether it's stale (holder crashed without releasing).
    stale = conn.execute(
        "SELECT 1 FROM run_lock WHERE name = 'dreaming'"
        " AND acquired_at < datetime('now', ?)",
        (f"-{_LOCK_TTL_SECONDS} seconds",),
    ).fetchone()
    if not stale:
        return False

    conn.execute("DELETE FROM run_lock WHERE name = 'dreaming'")
    try:
        conn.execute(
            "INSERT INTO run_lock(name, acquired_at, holder) VALUES ('dreaming', CURRENT_TIMESTAMP, ?)",
            (holder,),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _release_lock(conn: sqlite3.Connection, holder: str) -> None:
    with contextlib.suppress(sqlite3.Error):
        conn.execute(
            "DELETE FROM run_lock WHERE name = 'dreaming' AND holder = ?",
            (holder,),
        )


def _all_sessions(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT id FROM sessions ORDER BY started_at").fetchall()
    return [r["id"] for r in rows]
