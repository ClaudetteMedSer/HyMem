"""Dependency-neutral source-time versus transaction-time validation."""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, date, datetime, time, timedelta


# Producer and database clocks may differ slightly. Keep this fixed and shared
# so ingestion, lifecycle persistence, and portable replay accept the same skew.
EVENT_CLOCK_SKEW_SECONDS = 300
_SUPPORTED_ISO_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"(?:[T ][0-9]{2}:[0-9]{2}"
    r"(?::[0-9]{2}(?:\.[0-9]{1,6})?)?"
    r"(?:Z|[+-][0-9]{2}:[0-9]{2})?"
    r")?\Z"
)
_OFFSET_SUFFIX = re.compile(r"[+-]([0-9]{2}):([0-9]{2})\Z")


def register_sqlite_time_functions(conn: sqlite3.Connection) -> None:
    """Install the shared temporal parser/comparison functions on ``conn``.

    Most application connections are created by :mod:`hymem.core.db`, but
    read-only probes intentionally open SQLite directly.  Keeping registration
    next to the implementations prevents those consumers from either failing
    with a missing UDF or silently falling back to SQLite's divergent date
    grammar.
    """
    conn.create_function(
        "hymem_is_iso_timestamp", 1, is_iso_timestamp, deterministic=True
    )
    conn.create_function(
        "hymem_normalize_iso_timestamp",
        1,
        normalize_iso_timestamp_or_none,
        deterministic=True,
    )
    conn.create_function(
        "hymem_timestamp_at_or_before",
        2,
        timestamp_at_or_before,
        deterministic=True,
    )
    conn.create_function(
        "hymem_timestamp_gap_within",
        3,
        timestamp_gap_within,
        deterministic=True,
    )
    conn.create_function(
        "hymem_event_clock_is_valid",
        2,
        event_clock_is_valid,
        deterministic=True,
    )


def is_iso_timestamp(value: object) -> int:
    """Return ``1`` only for timestamps accepted by the public clock parser.

    SQLite's date functions deliberately accept inputs such as Julian-day
    numbers, year zero, invalid calendar days, and ``24:00``.  Public graph
    reads use this function as a deterministic SQL guard so corrupt stored
    values cannot acquire a different meaning on the read path.
    """
    try:
        _parse_iso_timestamp(value, context="stored")
    except ValueError:
        return 0
    return 1


def normalize_iso_timestamp_or_none(value: object) -> str | None:
    """SQLite UDF adapter: canonicalize valid input and fail closed otherwise."""
    try:
        return normalize_iso_timestamp(value, context="stored")
    except ValueError:
        return None


def timestamp_at_or_before(value: object, cutoff: object) -> int:
    """SQLite-safe comparison using the exact public timestamp grammar.

    Both operands are normalized by :func:`normalize_iso_timestamp`; malformed
    stored values therefore fail closed instead of inheriting SQLite's much
    broader Julian-date grammar or its fractional-second rounding behavior.
    """
    try:
        left = normalize_iso_timestamp(value, context="stored")
        right = normalize_iso_timestamp(cutoff, context="cutoff")
    except ValueError:
        return 0
    return int(left <= right)


def timestamp_gap_within(
    earlier_at: object, later_at: object, maximum_gap_seconds: object
) -> int:
    """SQLite-safe bounded causal-order check using the shared parser."""
    try:
        if (
            isinstance(maximum_gap_seconds, bool)
            or not isinstance(maximum_gap_seconds, int)
            or maximum_gap_seconds < 0
        ):
            return 0
        earlier = _parse_iso_timestamp(earlier_at, context="stored")
        later = _parse_iso_timestamp(later_at, context="stored")
        return int(
            later >= earlier
            and later - earlier <= timedelta(seconds=maximum_gap_seconds)
        )
    except (OverflowError, ValueError):
        return 0


def event_clock_is_valid(event_at: object, recorded_at: object) -> int:
    """UDF-safe form of the shared valid-time/transaction-time guard."""
    try:
        event = _parse_iso_timestamp(event_at, context="stored event")
        recorded = _parse_iso_timestamp(recorded_at, context="stored event")
    except ValueError:
        return 0
    return int(
        event <= recorded
        or event - recorded <= timedelta(seconds=EVENT_CLOCK_SKEW_SECONDS)
    )


def normalize_iso_timestamp(value: object, *, context: str) -> str:
    """Return one UTC-millisecond spelling for the supported wire grammar."""
    parsed = _parse_iso_timestamp(value, context=context)
    # Spell the year explicitly: platform strftime implementations are not
    # consistent about zero-padding years below 1000. Fractions are truncated,
    # never rounded, so e.g. .9999 remains inside the same millisecond.
    return (
        f"{parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d}T"
        f"{parsed.hour:02d}:{parsed.minute:02d}:{parsed.second:02d}."
        f"{parsed.microsecond // 1000:03d}Z"
    )


def validate_event_clock(
    conn: sqlite3.Connection,
    event_at: str | None,
    recorded_at: str | None,
    *,
    context: str = "event",
) -> None:
    """Require parseable event/recorded clocks and bound future source skew.

    The comparison is against the causal row's persisted transaction timestamp,
    not Python wall time. Imports and replays are therefore deterministic.
    """
    # Keep ``conn`` in the signature because every caller's comparison is tied
    # to SQLite transaction metadata; parsing itself is deliberately stricter
    # than SQLite's permissive julianday() (which accepts bare Julian numbers).
    del conn
    event = _parse_iso_timestamp(event_at, context=context)
    recorded = _parse_iso_timestamp(recorded_at, context=context)
    # Subtraction is safe at datetime.min/max; adding the allowance to a
    # year-9999 transaction coordinate can itself overflow.
    if event > recorded and not event_clock_is_valid(event_at, recorded_at):
        raise ValueError(
            f"{context} valid time cannot be later than its recorded time "
            f"by more than {EVENT_CLOCK_SKEW_SECONDS} seconds"
        )


def validate_timestamp_order(
    earlier_at: object,
    later_at: object,
    *,
    context: str,
    maximum_gap_seconds: int | None = None,
) -> None:
    """Require a non-negative causal interval, optionally with a gap bound."""
    earlier = _parse_iso_timestamp(earlier_at, context=context)
    later = _parse_iso_timestamp(later_at, context=context)
    if later < earlier:
        raise ValueError(f"{context} timestamps are causally inverted")
    if (
        maximum_gap_seconds is not None
        and later - earlier > timedelta(seconds=maximum_gap_seconds)
    ):
        raise ValueError(
            f"{context} timestamps differ by more than "
            f"{maximum_gap_seconds} seconds"
        )


def earliest_timestamp_spelling(*values: object) -> str:
    """Choose the earliest instant, breaking equivalent-instant ties portably."""
    candidates = [value for value in values if isinstance(value, str)]
    if not candidates:
        raise ValueError("timestamp minimum requires at least one string")
    return min(
        candidates,
        key=lambda value: (
            normalize_iso_timestamp(value, context="timestamp minimum"),
            value,
        ),
    )


def latest_timestamp_spelling(*values: object) -> str:
    """Choose the latest instant, breaking equivalent-instant ties portably."""
    candidates = [value for value in values if isinstance(value, str)]
    if not candidates:
        raise ValueError("timestamp maximum requires at least one string")
    return max(
        candidates,
        key=lambda value: (
            normalize_iso_timestamp(value, context="timestamp maximum"),
            value,
        ),
    )


def _parse_iso_timestamp(value: object, *, context: str) -> datetime:
    if (
        not isinstance(value, str)
        or not value
        or _SUPPORTED_ISO_TIMESTAMP.fullmatch(value) is None
    ):
        raise ValueError(f"{context} timestamps must be valid ISO-8601 strings")
    raw = value
    try:
        offset = _OFFSET_SUFFIX.search(raw)
        if offset is not None:
            hours, minutes = (int(part) for part in offset.groups())
            # Keep offsets inside the civil-time range used by ISO profiles.
            # SQL never reparses the raw value; its UDF calls this same parser.
            if hours > 14 or minutes > 59 or (hours == 14 and minutes != 0):
                raise ValueError
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            parsed = datetime.combine(date.fromisoformat(raw), time(), tzinfo=UTC)
        else:
            parsed = datetime.fromisoformat(
                raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            )
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            else:
                parsed = parsed.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{context} timestamps must be valid ISO-8601 strings"
        ) from exc
    return parsed
