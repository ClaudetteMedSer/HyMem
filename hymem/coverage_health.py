"""Bounded, read-only audit of retained lossless coverage, without repair.

Unlike a successful schema open, this checks every retained proof, including
proofs that no derived artifact currently cites. The canonical lossless reader
owns proof semantics. Missing raw messages are legitimate after retention;
missing backing chunks, sessions, ownership, or membership are not.

The scan does not establish the existence of already-destroyed raw/proof pairs,
nor does it certify unmaterialized raw messages above the producer frontier.
Caller-defined producer versions may carry valid independent exact proofs;
these are counted separately and never accepted as ordered-stream membership.
Recorded stream failures remain actionable until a maintained full
stream walk clears them: valid retained artifacts alone do not prove that an
earlier materialization failure has recovered. No ledger rows are cleared here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import sqlite3

# Reuse the scanner-owned deadline/progress guard. It uses BaseException so a
# best-effort source validator cannot accidentally turn exhaustion into success.
from hymem.embedding_source_health import _AuditLimit, _Budget


@dataclass(frozen=True)
class CoverageHealth:
    total: int | None = None
    checked: int = 0
    valid_raw_present: int = 0
    valid_raw_pruned: int = 0
    invalid: int = 0
    independent_proofs: int = 0
    missing_proofs: int | None = None
    invalid_frontiers: int | None = None
    recorded_failure_sessions: int | None = None
    complete: bool = False
    error_code: str | None = None

    @property
    def status(self) -> str:
        if not self.complete or self.error_code:
            return "unavailable"
        if (self.invalid or self.missing_proofs
                or self.invalid_frontiers or self.recorded_failure_sessions):
            return "invalid"
        return "valid"


def scan_coverage_health(
    path: Path, *, max_rows: int = 100_000, max_seconds: float = 30.0,
    max_sql_steps: int = 100_000_000, max_bytes: int = 256 * 1024 * 1024,
) -> CoverageHealth:
    """Scan one current-schema snapshot with bounded SQL/Python work.

    Owns a mode=ro connection, transaction and progress handler. No initialize,
    authority mutation, provider calls, or repairs occur. Rows are streamed;
    SQLite values are capped at 8 MiB and accumulated text/byte work is bounded.
    Deadline checks are cooperative around each capped canonical validation.
    Incomplete counts are explicitly marked unavailable, never partial green.
    """
    report = CoverageHealth()
    if (type(max_rows) is not int or not 1 <= max_rows <= 1_000_000
            or type(max_seconds) not in (int, float)
            or not math.isfinite(max_seconds) or not 0 < max_seconds <= 300
            or type(max_sql_steps) is not int or not 1 <= max_sql_steps <= 1_000_000_000
            or type(max_bytes) is not int or not 1 <= max_bytes <= 1024 * 1024 * 1024):
        return replace(report, error_code="invalid_bounds")
    budget = _Budget(max_seconds, max_sql_steps)
    conn = None
    try:
        from hymem.core import db
        from hymem.dreaming.lossless import (
            COVERAGE_VALIDATION_COLUMNS, COVERAGE_VALIDATION_JOINS,
            validate_message_coverage_row,
        )
        from hymem.dreaming.message_coverage import LOSSLESS_READ_VERSIONS

        budget.check()
        conn = sqlite3.connect(
            Path(path).absolute().as_uri() + "?mode=ro", uri=True,
            isolation_level=None, timeout=min(1.0, max_seconds),
        )
        conn.row_factory = sqlite3.Row
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 8 * 1024 * 1024)
        conn.set_progress_handler(budget.progress, budget.interval)
        conn.execute("BEGIN")
        if db.schema_version(conn) != db.EXPECTED_SCHEMA_VERSION:
            return replace(report, error_code="current_schema_required")

        def count(sql: str, parameters=()) -> int:
            budget.check()
            result = conn.execute(sql, parameters).fetchone()[0]
            budget.check()
            return result

        report = replace(report, total=count("SELECT count(*) FROM message_retention_coverage"))
        report = replace(report, recorded_failure_sessions=count(
            "SELECT count(*) FROM coverage_integrity_failures"
        ))
        # A malformed frontier must not make a missing-proof witness disappear
        # through SQLite's permissive numeric comparisons.
        report = replace(report, invalid_frontiers=count(
            "SELECT count(*) FROM sessions WHERE coverage_message_id IS NOT NULL "
            "AND (typeof(coverage_message_id) != 'integer' OR coverage_message_id < 0)"
        ))
        placeholders = ",".join("?" for _ in LOSSLESS_READ_VERSIONS)
        report = replace(report, missing_proofs=count(
            "SELECT count(*) FROM messages raw JOIN sessions session ON session.id=raw.session_id "
            "WHERE typeof(session.coverage_message_id)='integer' "
            "AND raw.id <= session.coverage_message_id AND NOT EXISTS ("
            "SELECT 1 FROM message_retention_coverage mc "
            "WHERE mc.message_id=raw.id AND mc.source_session_id=raw.session_id "
            f"AND mc.coverage_version IN ({placeholders}))", LOSSLESS_READ_VERSIONS,
        ))

        # Keep the maintained projection and validator; only the audit joins
        # differ. Missing chunks must survive to validation, and a surviving
        # raw row with a *different* session must not masquerade as pruned.
        joins = COVERAGE_VALIDATION_JOINS.replace(
            "JOIN chunks c", "LEFT JOIN chunks c"
        ).replace("     AND raw.session_id = mc.source_session_id", "")
        cursor = conn.execute(
            f"SELECT {COVERAGE_VALIDATION_COLUMNS} "
            f"FROM message_retention_coverage mc {joins}"
        )
        used_bytes = 0
        for row in cursor:
            budget.check()
            if report.checked >= max_rows:
                raise _AuditLimit("coverage_row_budget_exhausted")
            # UTF-8 is at most four bytes per character; avoid allocating an
            # encoding of source text solely for diagnostic accounting.
            used_bytes += sum(
                len(value) * (4 if isinstance(value, str) else 1)
                for value in row if isinstance(value, (str, bytes))
            )
            if used_bytes > max_bytes:
                raise _AuditLimit("coverage_byte_budget_exhausted")
            report = replace(report, checked=report.checked + 1)
            try:
                if (not isinstance(row["coverage_version"], str)
                        or not row["coverage_version"].strip()):
                    raise ValueError("invalid coverage version")
                validate_message_coverage_row(row)
            except (RuntimeError, ValueError, TypeError, KeyError, IndexError, UnicodeError, OverflowError):
                # Do not export exception strings, IDs, text or table values.
                report = replace(report, invalid=report.invalid + 1)
            else:
                if row["coverage_version"] not in LOSSLESS_READ_VERSIONS:
                    report = replace(report, independent_proofs=report.independent_proofs + 1)
                if row["raw_message_id"] is None:
                    report = replace(report, valid_raw_pruned=report.valid_raw_pruned + 1)
                else:
                    report = replace(report, valid_raw_present=report.valid_raw_present + 1)
            budget.check()
        if report.checked != report.total:
            return replace(report, error_code="coverage_join_count_mismatch")
        budget.check()
        return replace(report, complete=True)
    except _AuditLimit as exc:
        return replace(report, error_code=exc.code)
    except Exception:  # Diagnostic output must never contain arbitrary exception text.
        return replace(report, error_code=(
            "diagnostic_budget_exhausted" if budget.exhausted else "coverage_schema_or_read_failure"
        ))
    finally:
        if conn is not None:
            conn.set_progress_handler(None, 0)
            if conn.in_transaction:
                conn.rollback()
            conn.close()
