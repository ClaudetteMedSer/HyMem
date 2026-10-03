"""Stdlib-only date extraction for the temporal-reasoning (TR) retrieval path.

Why this exists
---------------
BEAM Temporal-Reasoning questions ("how long between X and Y?", "what happened
first?") need a *date-ordered* view of events. HyMem already stores real event
time on ``messages.created_at`` and a ``temporal_scope`` on graph edges, but it
never extracts the dates a user *writes in prose* ("we shipped on Feb 15", "de
release was 3 maart"). This module finds those explicit dates and normalizes
them to ISO ``YYYY-MM-DD`` so the dream cycle can index them into
``temporal_mentions`` and the TR augment path can return events already sorted.

Scope & limits (deliberately narrow)
------------------------------------
- **Stdlib only.** ``dateparser`` is explicitly deferred, so this is a set of
  hand-rolled regexes, not a general natural-language date parser. It targets
  the *explicit, written* date forms that show up in chat, not relative
  expressions ("yesterday", "two weeks ago") — those are recency, not the
  absolute timeline TR needs, and resolving them requires an anchor date.
- **Latin-script, English + Dutch** month names only (the project's documented
  language scope), plus language-agnostic numeric/ISO forms.
- **Day/month ambiguity.** Bare numeric forms like ``2/3`` are inherently
  ambiguous (US vs. EU ordering). We commit to **day-month** ordering when a
  token is clearly > 12 (``15/02`` can only be 15 Feb) and otherwise assume
  **month-day** for slash-separated and **day-month** for hyphen/dot-separated
  forms — the most common conventions for each separator. A wrong guess on a
  genuinely ambiguous bare date is a known, bounded failure mode; month-name
  forms ("Feb 15", "15 maart") are unambiguous and always preferred.
- **Year inference.** A date with no year cannot be normalized to a full ISO
  string on its own, so it is returned with ``normalized_date=None`` but its
  raw text is still captured — the caller (and the host LLM) can still use the
  month/day ordering for "what happened first" reasoning even without a year.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# English + Dutch month names → month number. Lowercased keys; both the full
# name and the common 3-letter abbreviation are registered. Dutch and English
# share several spellings (april, augustus≈august via "aug"), which is fine —
# the value is identical, so a collision just maps to the same month.
_MONTHS: dict[str, int] = {
    # English full
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    # English abbreviations
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # Dutch full (those that differ from English)
    "januari": 1, "februari": 2, "maart": 3, "mei": 5, "juni": 6,
    "juli": 7, "augustus": 8, "oktober": 10,
    # Dutch abbreviations that differ from English
    "mrt": 3, "okt": 10,
}

# Alternation of every month token, longest first so "september" wins over
# "sep" during matching. Word-boundary anchored at the use sites.
_MONTH_ALT = "|".join(
    sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True)
)

# Full ISO date: 2024-02-15. Anchored so it is not a substring of a longer
# token; the surrounding boundaries keep it from matching inside, e.g., a UUID.
_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")

# Month-name + day, optionally with year and an ordinal suffix:
#   "Feb 15", "March 1st", "February 15, 2024", "Jan 3 2023"
_MONTH_DAY = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)

# Day + month-name, optionally with year (the Dutch/European ordering):
#   "15 maart", "15 March 2024", "1st of April" → ("of" tolerated)
_DAY_MONTH = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})\.?(?:\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)

# Numeric slash form: 2/15 or 02/15/2024. Ambiguous; resolved in _numeric_order.
_NUM_SLASH = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?!\d)")

# Numeric hyphen/dot form: 15-02, 15.02.2024. Treated as day-first (EU style),
# the dominant convention for these separators in the project's locales. The
# 4-digit-first case is already handled by _ISO, so leading 4-digit groups are
# excluded here to avoid double-matching a year.
_NUM_DMY = re.compile(
    r"(?<!\d)(\d{1,2})[.\-](\d{1,2})(?:[.\-](\d{2,4}))?(?!\d)"
)


@dataclass(frozen=True)
class DateMention:
    """One explicit date found in a span of text.

    ``normalized_date`` is the ISO ``YYYY-MM-DD`` rendering when a full date
    (incl. year) could be resolved, else ``None`` (year-less dates still carry
    their ``raw_text`` so month/day ordering remains usable). ``raw_text`` is
    the exact matched substring, kept verbatim for evidence/debugging.
    """

    normalized_date: str | None
    raw_text: str


def _two_digit_year(year: int) -> int:
    """Expand a 2-digit year the way chat logs usually mean it: 00–69 → 2000s,
    70–99 → 1900s. A coarse, deterministic heuristic; full 4-digit years (the
    common case in these forms) skip it entirely."""
    if year >= 100:
        return year
    return 2000 + year if year <= 69 else 1900 + year


def _valid_md(month: int, day: int) -> bool:
    """Cheap calendar sanity check: month 1–12, day 1–31. Deliberately does not
    validate day-per-month (no Feb-30 rejection) — over-strict validation would
    silently drop a usable, if imperfect, user-written date, and the host LLM is
    the final arbiter of correctness."""
    return 1 <= month <= 12 and 1 <= day <= 31


def _iso(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def _numeric_order(a: int, b: int, *, day_first: bool) -> tuple[int, int] | None:
    """Resolve a two-number date into (month, day).

    If one value is > 12 it can only be the day — that disambiguates regardless
    of the assumed convention. Otherwise fall back to the separator-implied
    ``day_first`` ordering (hyphen/dot → day-first, slash → month-first).
    Returns ``None`` when neither ordering yields a valid month/day."""
    if a > 12 and b <= 12:
        month, day = b, a
    elif b > 12 and a <= 12:
        month, day = a, b
    elif day_first:
        month, day = b, a
    else:
        month, day = a, b
    return (month, day) if _valid_md(month, day) else None


def extract_dates(text: str) -> list[DateMention]:
    """Return the explicit dates written in ``text``, de-duplicated by raw span.

    Ordering of detection is intentional: unambiguous forms (ISO, then
    month-name) are matched first and their character spans recorded, so the
    ambiguous numeric fallbacks never re-match a slice already claimed by a
    clearer form (e.g. the ``2024-02`` inside an ISO date won't also fire the
    numeric matcher). Returns mentions in first-appearance order."""
    if not text:
        return []

    claimed: list[tuple[int, int]] = []
    results: list[DateMention] = []
    seen_raw: set[str] = set()

    def _overlaps(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in claimed)

    def _add(start: int, end: int, normalized: str | None, raw: str) -> None:
        if _overlaps(start, end):
            return
        claimed.append((start, end))
        if raw not in seen_raw:
            seen_raw.add(raw)
            results.append(DateMention(normalized_date=normalized, raw_text=raw))

    # 1. ISO — fully unambiguous, highest priority.
    for m in _ISO.finditer(text):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        norm = _iso(year, month, day) if _valid_md(month, day) else None
        _add(m.start(), m.end(), norm, m.group(0))

    # 2. Month-name forms (both orderings) — unambiguous month, optional year.
    for matcher, day_grp, mon_grp in (
        (_MONTH_DAY, 2, 1),
        (_DAY_MONTH, 1, 2),
    ):
        for m in matcher.finditer(text):
            month = _MONTHS.get(m.group(mon_grp).lower())
            if month is None:
                continue
            day = int(m.group(day_grp))
            if not _valid_md(month, day):
                continue
            year_str = m.group(3)
            norm = (
                _iso(_two_digit_year(int(year_str)), month, day)
                if year_str
                else None
            )
            _add(m.start(), m.end(), norm, m.group(0))

    # 3. Numeric fallbacks — ambiguous, lowest priority.
    for matcher, day_first in ((_NUM_SLASH, False), (_NUM_DMY, True)):
        for m in matcher.finditer(text):
            a, b = int(m.group(1)), int(m.group(2))
            order = _numeric_order(a, b, day_first=day_first)
            if order is None:
                continue
            month, day = order
            year_str = m.group(3)
            norm = (
                _iso(_two_digit_year(int(year_str)), month, day)
                if year_str
                else None
            )
            _add(m.start(), m.end(), norm, m.group(0))

    return results
