#!/usr/bin/env python3
"""E4 front-run — does a temporal-range boost ever FIRE, and is it right?

The pre-build gate for Campaign E, Step 8 / E4 (`additional_planning.md`). E4
boosts items whose dates fall inside a range resolved **from the query** ("what
did we decide last week?"). Before writing `hymem/query/reldates.py` and the
augment wiring, this probe answers the only question that can kill the feature
outright: *do real queries carry a resolvable relative date at all?*

That question has a precedent. Track A's multi-hop traversal was mechanism-
validated and then turned out **inert** on LME — the store's star topology meant
~0 genuine bridges existed, so a correct feature bridged nothing. E4 has the same
failure shape: a correct range boost that no query ever triggers is dead code
with a config flag. Measure first.

── The three populations, and why the split is the finding ──────────────────────
This probe deliberately measures relative-date language in two different places,
because they answer two different questions:

  • **query-side** (`locomo-questions`, `lme-questions`) — what E4 actually
    boosts on, and the ONLY population the gate reads. A scored A/B can never
    see more than this.
  • **content-side** (`locomo-turns`) — relative dates written *inside stored
    conversation*. NON-GATING, reported for context: it is the population a
    different feature (normalizing relative mentions at ingest, the gap
    `dreaming/dates.py` documents) would serve. A high content-side rate with a
    dead query-side rate means "there is something here, but E4 as specified is
    not the way to get it" — which is a materially different verdict from
    "relative dates do not occur".

Benchmark questions are written by annotators as self-contained lookups, so a
low query-side rate is partly an artifact of the corpus, not of user behaviour —
the same reason E5's value is invisible to LME. That argument is only admissible
if the content-side number supports it, which is exactly why it is measured here
rather than asserted.

── What "resolvable" means (and what is deliberately NOT) ───────────────────────
Only expressions that yield a concrete [start, end] against an anchor date count
as a fire: yesterday, N days/weeks/months ago, last/this/next week/month/year,
"in the past N days", plus the Dutch equivalents. **Vague markers are counted
SEPARATELY and never as fires** — "recently", "lately", "a while back", "the
other day", "onlangs" carry no arithmetic, and resolving them to an invented
window is the failure mode this probe exists to avoid, not a feature. The split
is reported because a corpus whose relative language is ~all vague is one E4
cannot serve no matter how good the resolver is.

`resolve_range()` here is a PROTOTYPE. If G-E4a passes it is promoted to
`hymem/query/reldates.py`, hardened, and given the pattern-matrix test the plan
specifies; the probe then imports it instead. This mirrors `fact_probe.py`
carrying `FACTS_PROMPT_V1` before `facts.py` existed.

── Pre-registered gate G-E4a (decides build/no-build; score-free) ───────────────
BUILD iff ALL THREE hold on the query-side populations:
  1. **fire rate ≥ 5%** — below that the boost cannot move a scored benchmark
     even when it is perfectly correct (Track A's lesson).
  2. **range precision ≥ 90%** on fired questions with a dated gold session —
     the resolved range must contain the gold date. A wrong range boosts the
     wrong items; boost-not-filter caps that damage but does not remove it.
  3. **no-harm control: ZERO** fires on questions carrying no temporal marker.
     Same shape as the E5 gate, whose control was the load-bearing half.

Criterion 2 reads gold labels. That is correct for a probe and forbidden in the
product: gold is applied strictly AFTER resolution, and the resolver never sees
it (the `fact_probe.py` provenance posture).

── Usage (from benchmarks/) ─────────────────────────────────────────────────────
  python reldate_probe.py --locomo data/locomo10.json
  python reldate_probe.py --locomo data/locomo10.json --verbose
  python reldate_probe.py --dataset ~/.hermes/data/lme_s.json          # LME
  python reldate_probe.py --locomo data/locomo10.json --json
No LLM, no embeddings, no store — stdlib + the repo. Runs in seconds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pre-registered gate thresholds (see module docstring). These ARE the gate, so
# they live here as named constants — not as CLI defaults a run could bend.
_MIN_FIRE_RATE = 0.05
_MIN_PRECISION = 0.90
_MAX_CONTROL_FIRES = 0
# Below this many fired-with-gold questions, precision is reported but NOT read
# as a gate criterion: a 1-in-8 miss is not distinguishable from a 1-in-8 fluke.
_MIN_N_FOR_PRECISION = 10

_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")

# ── number words ────────────────────────────────────────────────────────────────
# EN + NL, 1-12 plus the articles that mean "one" ("a week ago", "een week
# geleden"). Bare digits are handled by the same regex branch.
_NUMBERS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "couple": 2, "few": 3,
    "een": 1, "één": 1, "twee": 2, "drie": 3, "vier": 4, "vijf": 5, "zes": 6,
    "zeven": 7, "acht": 8, "negen": 9, "tien": 10, "elf": 11, "twaalf": 12,
    "paar": 2,
}
_NUM_ALT = "|".join(sorted(_NUMBERS, key=len, reverse=True))

# Unit → (days per unit, half-window tolerance in days). The tolerance is what
# turns a POINT ("two weeks ago") into a retrieval RANGE: people do not mean the
# 14th day exactly, they mean that week. Generous is correct here because E4
# boosts rather than filters — a slightly wide window costs ranking, a narrow one
# costs the hit outright.
_UNITS: dict[str, tuple[int, int]] = {
    "day": (1, 1), "days": (1, 1), "dag": (1, 1), "dagen": (1, 1),
    "week": (7, 3), "weeks": (7, 3), "weken": (7, 3),
    "month": (30, 7), "months": (30, 7), "maand": (30, 7), "maanden": (30, 7),
    "year": (365, 30), "years": (365, 30), "jaar": (365, 30), "jaren": (365, 30),
}
_UNIT_ALT = "|".join(sorted(_UNITS, key=len, reverse=True))

# ── patterns ────────────────────────────────────────────────────────────────────
# "3 weeks ago", "een paar dagen geleden", "two months back", "a couple of
# months back". The leading article is OPTIONAL and the quantity is REQUIRED, so
# "a week ago" still resolves (the article backtracks into the quantity slot as
# one) while a bare "years ago" — a quantity-free plural, i.e. vague — does not.
_AGO = re.compile(
    rf"\b(?:(?:a|an|een)\s+)?(?:(\d{{1,3}})|({_NUM_ALT}))\s+"
    rf"(?:of\s+|van\s+)?({_UNIT_ALT})\s+(?:ago|back|geleden)\b")
# "in the last 10 days", "de afgelopen twee weken", "over the past month"
_WITHIN = re.compile(
    rf"\b(?:in\s+the\s+(?:last|past)|over\s+the\s+(?:last|past)|within\s+the\s+"
    rf"(?:last|past)|de\s+afgelopen|afgelopen|in\s+de\s+afgelopen)\s+"
    rf"(?:(\d{{1,3}})|({_NUM_ALT}))?\s*({_UNIT_ALT})\b")
# "last week", "vorige maand", "this year", "volgende week"
_CALENDAR = re.compile(
    rf"\b(last|this|next|past|vorige|afgelopen|deze|dit|volgende|komende)\s+"
    rf"({_UNIT_ALT})\b")
_MONTHS_EN_NAMES = (
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "januari", "februari", "maart", "mei", "juni", "juli", "augustus",
    "oktober",
)
_MONTH_NUM: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "januari": 1, "februari": 2, "maart": 3, "mei": 5, "juni": 6, "juli": 7,
    "augustus": 8, "oktober": 10,
}
_MONTH_RE_ALT = "|".join(sorted(_MONTH_NUM, key=len, reverse=True))
_ANCHOR_ISO = re.compile(
    r"(?<!\d)(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})(?!\d)")
# "6 November, 2023" and "November 6, 2023" — both orders, comma optional.
_PROSE_DATE = re.compile(
    rf"\b(?:(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<mon1>{_MONTH_RE_ALT})|"
    rf"(?P<mon2>{_MONTH_RE_ALT})\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?)"
    rf",?\s+(?P<y2>\d{{4}})\b", re.I)


def _prose_to_date(m: re.Match) -> date | None:
    g = m.groupdict()
    try:
        if g.get("mo"):
            return date(int(g["y"]), int(g["mo"]), int(g["d"]))
        mon = _MONTH_NUM[(g.get("mon1") or g.get("mon2")).lower()]
        return date(int(g["y2"]), mon, int(g.get("d1") or g.get("d2")))
    except (ValueError, KeyError, AttributeError):
        return None


# ABSOLUTE constructions that reuse the relative vocabulary: "the last week OF
# August 2023", "de eerste week van maart" — anchored to a named period, not to
# now. Left unguarded these fire the relative branch and resolve to a window
# that can be a year off, which is a FALSE FIRE dressed as a low-precision hit.
# Matched on the text FOLLOWING a candidate span, so the guard is local.
_ABSOLUTE_OF = re.compile(
    rf"^\s*(?:of|in|van)\s+(?:{'|'.join(_MONTHS_EN_NAMES)}|\d{{4}})\b", re.I)
# Single-token day references.
_DAY_WORDS: dict[str, int] = {
    "yesterday": -1, "gisteren": -1, "last night": -1, "gisteravond": -1,
    "gisternacht": -1, "today": 0, "vandaag": 0, "tonight": 0, "vanavond": 0,
    "tomorrow": 1, "morgen": 1,
}
_DAY_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(_DAY_WORDS, key=len, reverse=True)) + r")\b")
# "between 2023-01-05 and 2023-02-01" — explicit ISO only; prose dates are
# `dreaming/dates.py`'s job and are not what E4 adds.
_BETWEEN = re.compile(
    r"\bbetween\s+(\d{4}-\d{2}-\d{2})\s+(?:and|to|en)\s+(\d{4}-\d{2}-\d{2})\b")

# Vague relative language: real temporal intent, NO resolvable arithmetic. Never
# a fire; counted separately because the ratio vague:resolvable is the finding.
_VAGUE = (
    "recently", "lately", "a while back", "a while ago", "some time ago",
    "sometime ago", "the other day", "earlier", "previously", "in the past",
    "back then", "at some point", "these days", "nowadays", "of late",
    "onlangs", "laatst", "pas geleden", "kortgeleden", "eerder", "vroeger",
    "destijds", "ooit", "tegenwoordig", "een tijdje terug", "laatste tijd",
)


@dataclass(frozen=True)
class RangeHit:
    """A resolved [start, end] window plus the rule and surface that produced it.
    `rule` is what the why-code would carry in production, so a probe row is
    directly readable as the boost's justification.

    `prospective` marks a window that lies ahead of the anchor ("tonight", "next
    week"). It is resolved but must NOT be boosted on: stored content is past, so
    the window can never match, and the fire is pure cost. The resolver reports
    it rather than returning None so it can be counted as its own category — a
    prospective question is neither a resolvable retrieval range nor a
    marker-free control, and silently sorting it into either one is how the
    2026-07-30 LME control read as clean while two prospective questions sat in
    the fired bucket.
    """
    start: str
    end: str
    rule: str
    surface: str
    prospective: bool = False


def _win(anchor: date, *, offset_days: int, tol: int) -> tuple[date, date]:
    p = anchor + timedelta(days=offset_days)
    return p - timedelta(days=tol), p + timedelta(days=tol)


def _month_span(d: date, delta_months: int) -> tuple[date, date]:
    m = d.month - 1 + delta_months
    y, m = d.year + m // 12, m % 12 + 1
    start = date(y, m, 1)
    end = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
    return start, end


def _calendar_span(anchor: date, unit: str, direction: int) -> tuple[date, date]:
    """Calendar-aligned window for the last/this/next <unit> family. Users mean
    the calendar unit ("last week" = that Mon-Sun block), not a rolling window,
    so this aligns to ISO weeks / month bounds / calendar years."""
    days, _ = _UNITS[unit]
    if days == 7:
        monday = anchor - timedelta(days=anchor.weekday()) + timedelta(weeks=direction)
        return monday, monday + timedelta(days=6)
    if days == 30:
        return _month_span(anchor, direction)
    if days == 365:
        y = anchor.year + direction
        return date(y, 1, 1), date(y, 12, 31)
    return _win(anchor, offset_days=direction, tol=0)


def _num(digits: str | None, word: str | None) -> int:
    if digits:
        return max(1, min(999, int(digits)))
    return _NUMBERS.get((word or "").lower(), 1)


# Day words that point AHEAD even though they share the anchor's date. "Tonight"
# is the evening to come; "today" is genuinely ambiguous ("what did I log
# today?") and is deliberately left retrospective.
_PROSPECTIVE_DAY_WORDS = frozenset({"tonight", "vanavond", "tomorrow", "morgen"})

# Directional qualifiers turn a POINT expression into a HALF-OPEN interval.
# "before today" does not mean today; "since three weeks ago" does not mean that
# week. Ignoring the qualifier emits a point window where the text denotes an
# interval — the defect class found in the 2026-07-30 LME misses.
_BACKWARD_CUES = ("before", "prior to", "up to", "up until", "until", "till",
                  "earlier than", "voor ", "vóór ", "tot ")
# NOTE: bare "from" is deliberately absent. "the plan from last month" means
# *of* last month, not *since* it — including it turned three ordinary lookups
# into open-ended ranges, one of which then covered its gold by accident and
# inflated precision. A cue that widens a window can only ever help the score,
# so it has to be held to a higher bar than one that narrows it.
_FORWARD_CUES = ("since", "ever since", "after", "sinds", "vanaf", "na ")
# An open-backward window still needs a concrete lower bound for a string
# comparison; nothing in a conversation store predates this.
_OPEN_START = "0001-01-01"
# How far back of the matched span to look for a qualifier. "since I started
# writing again three weeks ago" puts 30 characters between the two.
_CUE_WINDOW = 40


def _directional(low: str, match: re.Match) -> str:
    """The directional qualifier governing the expression at `match`, if any.

    Looks only at the text BEFORE the span, within a bounded window, so a cue
    belonging to a later clause cannot capture this one.
    """
    prefix = low[max(0, match.start() - _CUE_WINDOW):match.start()]
    back = max((prefix.rfind(c) for c in _BACKWARD_CUES), default=-1)
    fwd = max((prefix.rfind(c) for c in _FORWARD_CUES), default=-1)
    if back < 0 and fwd < 0:
        return ""
    # Whichever cue sits closest to the expression governs it.
    return "before" if back > fwd else "since"


def _apply_direction(hit: RangeHit, anchor: date, direction: str) -> RangeHit:
    if direction == "before":
        # Everything up to the window's end. A prospective base becomes
        # retrospective ("before tomorrow" is a past-facing range).
        return RangeHit(_OPEN_START, hit.end, f"before_{hit.rule}", hit.surface)
    if direction == "since":
        end = max(hit.end, anchor.isoformat())
        return RangeHit(hit.start, end, f"since_{hit.rule}", hit.surface)
    return hit


def _absolute_context(low: str, match: re.Match) -> bool:
    """True when a relative-looking span is actually part of an ABSOLUTE
    construction ("the last week of August 2023")."""
    return bool(_ABSOLUTE_OF.match(low[match.end():]))


def resolve_range(text: str, anchor: date) -> RangeHit | None:
    """Resolve the first resolvable relative-date expression in `text`.

    PROTOTYPE for `hymem/query/reldates.py` (see module docstring). Returns None
    when nothing resolves — including when the text carries only vague markers,
    which is a deliberate non-answer, not a miss.

    Precedence is most-specific-first: an explicit ISO range beats "N units ago"
    beats a bounded "past N units" beats a bare calendar unit beats a day word.
    """
    low = text.lower()

    m = _BETWEEN.search(low)
    if m:
        a, b = sorted((m.group(1), m.group(2)))
        return RangeHit(a, b, "between", m.group(0))

    m = _AGO.search(low)
    if m:
        n = _num(m.group(1), m.group(2))
        days, tol = _UNITS[m.group(3)]
        s, e = _win(anchor, offset_days=-n * days, tol=tol)
        return _finish(RangeHit(s.isoformat(), e.isoformat(), "n_units_ago",
                                m.group(0)), anchor, low, m)

    m = _WITHIN.search(low)
    if m and not _absolute_context(low, m):
        n = _num(m.group(1), m.group(2))
        days, _tol = _UNITS[m.group(3)]
        return _finish(RangeHit((anchor - timedelta(days=n * days)).isoformat(),
                                anchor.isoformat(), "within_last_n",
                                m.group(0)), anchor, low, m)

    m = _CALENDAR.search(low)
    if m and not _absolute_context(low, m):
        word, unit = m.group(1), m.group(2)
        direction = 1 if word in ("next", "volgende", "komende") else (
            0 if word in ("this", "deze", "dit") else -1)
        s, e = _calendar_span(anchor, unit, direction)
        return _finish(RangeHit(
            s.isoformat(), e.isoformat(),
            f"calendar_{'next' if direction > 0 else 'this' if direction == 0 else 'last'}",
            m.group(0)), anchor, low, m)

    m = _DAY_WORD_RE.search(low)
    if m:
        word = m.group(1)
        s, e = _win(anchor, offset_days=_DAY_WORDS[word], tol=0)
        return _finish(RangeHit(s.isoformat(), e.isoformat(), "day_word", word),
                       anchor, low, m, prospective_word=word in _PROSPECTIVE_DAY_WORDS)
    return None


def _finish(hit: RangeHit, anchor: date, low: str, match: re.Match,
            *, prospective_word: bool = False) -> RangeHit:
    """Apply the directional qualifier, then classify the result as prospective.

    Order matters: the qualifier is applied FIRST, because it can flip a
    prospective base into a retrospective range ("before tomorrow" asks about
    the past). Classifying first would suppress a question the qualifier makes
    perfectly answerable.
    """
    direction = _directional(low, match)
    hit = _apply_direction(hit, anchor, direction)
    prospective = (hit.start > anchor.isoformat()
                   or (prospective_word and not direction))
    return RangeHit(hit.start, hit.end, hit.rule, hit.surface, prospective)


def effective_anchor(text: str, default: date) -> tuple[date, bool]:
    """The date a relative expression in `text` should be measured FROM.

    A relative expression is deictic: it means "relative to when this was said".
    When the text itself states that reference point — LoCoMo annotators write
    "…, as mentioned on November 6, 2023", production passes `question_date` —
    that stated date is the anchor, and the caller's default is wrong by a year
    or more. Returns (anchor, overridden).

    This is not a nicety: without it the probe reports the *anchor's* error as
    the *resolver's* imprecision, which is exactly the class of instrument
    defect that made the G-F1 date reading unusable.
    """
    m = _ANCHOR_ISO.search(text) or _PROSE_DATE.search(text)
    if not m:
        return default, False
    d = _prose_to_date(m)
    return (d, True) if d else (default, False)


def vague_markers(text: str) -> list[str]:
    """Vague temporal language present in `text`. Reported, never resolved."""
    low = text.lower()
    return [v for v in _VAGUE if v in low]


def has_temporal_language(text: str) -> bool:
    """True when ANY temporal marker is present — resolvable or vague. Its
    negation defines the no-harm control population, so a marker missing here
    silently promotes a real temporal question into the control and makes a true
    fire look like a false one. Kept deliberately wide for that reason."""
    return bool(vague_markers(text)) or resolve_range(text, date(2000, 1, 1)) is not None


# ── LoCoMo loading ──────────────────────────────────────────────────────────────
_LOCOMO_DATE = re.compile(
    r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")
_MONTHS_EN = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}


def parse_locomo_date(raw: str) -> date | None:
    """LoCoMo session stamps read '1:56 pm on 8 May, 2023'. `dreaming/dates.py`
    drops the year on that form (the comma), and rather than loosen a production
    parser for one benchmark's format, the adapter owns it — the
    `locomo_adapter.py` posture."""
    m = _LOCOMO_DATE.search(raw or "")
    if not m:
        return None
    mon = _MONTHS_EN.get(m.group(2).lower())
    if not mon:
        return None
    try:
        return date(int(m.group(3)), mon, int(m.group(1)))
    except ValueError:
        return None


def _locomo_sessions(conv: dict) -> dict[int, date]:
    out: dict[int, date] = {}
    for key, val in conv.items():
        m = re.fullmatch(r"session_(\d+)_date_time", key)
        if m:
            d = parse_locomo_date(str(val))
            if d:
                out[int(m.group(1))] = d
    return out


def load_locomo(path: Path) -> tuple[list[dict], list[dict]]:
    """Returns (question rows, turn rows).

    Question anchor is the LAST session date: LoCoMo questions are posed after
    the conversation ends, so "last week" in a question means the week before
    that. Turn anchor is the turn's OWN session date.
    """
    data = json.loads(path.read_text())
    questions: list[dict] = []
    turns: list[dict] = []
    for si, sample in enumerate(data):
        conv = sample.get("conversation", {}) or {}
        dates = _locomo_sessions(conv)
        if not dates:
            continue
        last = max(dates.values())
        for qi, qa in enumerate(sample.get("qa", []) or []):
            q = str(qa.get("question", "")).strip()
            if not q:
                continue
            gold: list[str] = []
            for ev in (qa.get("evidence") or []):
                m = re.match(r"[A-Za-z](\d+):", str(ev))
                if m and int(m.group(1)) in dates:
                    gold.append(dates[int(m.group(1))].isoformat())
            questions.append({
                "id": f"{sample.get('sample_id', si)}-q{qi}", "text": q,
                "anchor": last.isoformat(), "gold_dates": sorted(set(gold)),
                "category": qa.get("category"),
            })
        for key, val in conv.items():
            m = re.fullmatch(r"session_(\d+)", key)
            if not m or not isinstance(val, list):
                continue
            anchor = dates.get(int(m.group(1)))
            if not anchor:
                continue
            for ti, turn in enumerate(val):
                text = str((turn or {}).get("text", "")).strip()
                if text:
                    turns.append({
                        "id": f"{sample.get('sample_id', si)}-s{m.group(1)}t{ti}",
                        "text": text, "anchor": anchor.isoformat(),
                        "gold_dates": [],
                    })
    return questions, turns


def normalize_date(raw: object) -> str:
    """LongMemEval stamps read `2023/05/20 (Sat) 02:21` — slash-separated, with
    a weekday and a time. Slicing the first 10 chars and demanding ISO drops
    EVERY date silently, which turns the precision criterion into a confident
    'n/a' rather than a loud failure. Returns '' when nothing parses."""
    s = str(raw or "").strip()
    m = re.match(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)),
                        int(m.group(3))).isoformat()
        except ValueError:
            return ""
    d = _prose_to_date(_PROSE_DATE.search(s)) if _PROSE_DATE.search(s) else None
    return d.isoformat() if d else ""


def load_lme(path: Path) -> list[dict]:
    """LME questions with `question_date` as anchor and the gold sessions' dates
    (via `answer_session_ids` → the `haystack_session_ids`/`haystack_dates`
    index alignment `fact_probe.py` relies on) as the precision ground truth."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("questions") or data.get("data") or []
    rows: list[dict] = []
    for q in data:
        text = str(q.get("question", "")).strip()
        anchor = normalize_date(q.get("question_date"))
        if not text or not anchor:
            continue
        sids = [str(s) for s in (q.get("haystack_session_ids") or [])]
        dates = [normalize_date(d) for d in (q.get("haystack_dates") or [])]
        gold_ids = {str(s) for s in (q.get("answer_session_ids") or [])}
        gold = sorted({d for s, d in zip(sids, dates) if s in gold_ids and d})
        rows.append({"id": str(q.get("question_id", len(rows))), "text": text,
                     "anchor": anchor, "gold_dates": gold,
                     "category": q.get("question_type")})
    return rows


# ── measurement ─────────────────────────────────────────────────────────────────
def measure(rows: list[dict], *, name: str, gating: bool) -> dict:
    """Score one population. Gold is applied strictly AFTER resolution."""
    fired: list[dict] = []
    vague_only: list[dict] = []
    prospective: list[dict] = []
    control_fires: list[dict] = []
    n_control = 0
    hits = misses = 0
    n_reanchored = 0
    rules: dict[str, int] = {}
    sides: dict[str, int] = {}
    # Fire rate by question category. A rate carried entirely by one
    # annotator-designed category ("temporal-reasoning") is a property of the
    # benchmark's category mix, not of how people ask questions — and E4 would
    # then be a one-category feature, which is a different claim.
    by_cat: dict[str, dict] = {}

    for row in rows:
        anchor, overridden = effective_anchor(row["text"],
                                              date.fromisoformat(row["anchor"]))
        hit = resolve_range(row["text"], anchor)
        cat = str(row.get("category") or "?")
        slot = by_cat.setdefault(cat, {"n": 0, "fired": 0})
        slot["n"] += 1
        slot["fired"] += hit is not None
        vague = vague_markers(row["text"])
        if hit is not None and hit.prospective:
            # Resolved, but forward-facing: no stored past item can fall in the
            # window, so boosting on it is cost with no upside. Counted as its
            # OWN category — not a fire (it would only ever be a miss) and not a
            # control row (the question does carry temporal language).
            prospective.append({**row, "start": hit.start, "end": hit.end,
                                "rule": hit.rule, "surface": hit.surface})
            continue
        if hit is None:
            if vague:
                vague_only.append({**row, "markers": vague})
            else:
                n_control += 1
            continue
        rules[hit.rule] = rules.get(hit.rule, 0) + 1
        n_reanchored += overridden
        rec = {**row, "start": hit.start, "end": hit.end, "rule": hit.rule,
               "surface": hit.surface, "anchor_used": anchor.isoformat(),
               "reanchored": overridden}
        # No temporal marker, yet the resolver fired → a false fire by
        # construction. `has_temporal_language` includes every resolvable form,
        # so this can only trip when the two disagree.
        if not vague and not has_temporal_language(row["text"]):
            control_fires.append(rec)
        if row["gold_dates"]:
            covered = any(hit.start <= g <= hit.end for g in row["gold_dates"])
            rec["gold_covered"] = covered
            hits += covered
            misses += not covered
            if not covered:
                # WHICH SIDE the gold falls on is the diagnosis, not a detail.
                # A range boost matches an item's stored date, which is
                # SPEECH time (when it was said). A relative expression in a
                # question is usually about EVENT time (when it happened).
                # "Where did Caroline move from 4 years ago?" resolves to 2020
                # correctly — and the turn where she says it is dated 2023.
                # Gold consistently AFTER the range is that axis mismatch;
                # gold scattered on both sides would instead be a resolver
                # accuracy problem. The two have opposite remedies, so the
                # split is measured rather than argued.
                after = sum(g > hit.end for g in row["gold_dates"])
                before = sum(g < hit.start for g in row["gold_dates"])
                rec["gold_side"] = "after" if after >= before else "before"
                sides[rec["gold_side"]] = sides.get(rec["gold_side"], 0) + 1
        fired.append(rec)

    n = len(rows) or 1
    scored = hits + misses
    # Per-rule precision. An aggregate can pass while the SINGLE most common
    # construction is broken and rarely-firing rules carry the average — and a
    # boost that is wrong on the expression people actually use is worse than
    # no boost, however good the mean looks.
    by_rule: dict[str, dict[str, int]] = {}
    for rec in fired:
        slot = by_rule.setdefault(rec["rule"], {"fired": 0, "scored": 0, "hits": 0})
        slot["fired"] += 1
        if "gold_covered" in rec:
            slot["scored"] += 1
            slot["hits"] += bool(rec["gold_covered"])
    return {
        "name": name, "gating": gating, "n": len(rows),
        "fired": len(fired), "fire_rate": 100.0 * len(fired) / n,
        "vague_only": len(vague_only),
        "vague_rate": 100.0 * len(vague_only) / n,
        "prospective": len(prospective),
        "prospective_rate": 100.0 * len(prospective) / n,
        "any_temporal": len(fired) + len(vague_only) + len(prospective),
        "n_control": n_control, "control_fires": len(control_fires),
        "reanchored": n_reanchored,
        "miss_sides": sides,
        # How many rows carry gold AT ALL — separates "this corpus has no dated
        # gold" from "the loader dropped it", which look identical downstream.
        "rows_with_gold": sum(1 for r in rows if r.get("gold_dates")),
        "by_category": by_cat,
        "by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1]["fired"])),
        "precision_n": scored,
        "precision": (100.0 * hits / scored) if scored else None,
        "rules": dict(sorted(rules.items(), key=lambda kv: -kv[1])),
        "rows_fired": fired, "rows_vague": vague_only,
        "rows_prospective": prospective,
        "rows_control_fires": control_fires,
    }


def summarize(pops: list[dict]) -> dict:
    gating = [p for p in pops if p["gating"] and p["n"]]
    n = sum(p["n"] for p in gating)
    fired = sum(p["fired"] for p in gating)
    ctl = sum(p["control_fires"] for p in gating)
    pn = sum(p["precision_n"] for p in gating)
    ph = sum(round((p["precision"] or 0) / 100.0 * p["precision_n"])
             for p in gating)
    fire_rate = (fired / n) if n else 0.0
    precision = (ph / pn) if pn else None
    # Precision is UNREAD, not failed, below the discrimination floor: with a
    # handful of scored questions the number is noise either way, and a probe
    # that fails a build on noise is worse than one that says "cannot tell".
    precision_read = pn >= _MIN_N_FOR_PRECISION
    # An UNMEASURED criterion is not a satisfied one. A corpus with no dated
    # gold cannot say whether the resolved ranges are right, and 2-of-3 is not
    # a pass — `fact_probe.py` reports INCOMPLETE for exactly this reason and
    # this probe reporting PASS instead was a defect, not a difference.
    n_with_gold = sum(1 for p in gating for _ in range(p["precision_n"]))
    gate = {
        "fire_rate_ok": fire_rate >= _MIN_FIRE_RATE,
        "precision_ok": (precision is not None and precision_read
                         and precision >= _MIN_PRECISION),
        "no_harm_ok": ctl <= _MAX_CONTROL_FIRES,
    }
    measured = {k: v for k, v in gate.items() if k != "precision_ok"}
    unmeasured = precision is None or not precision_read
    verdict = ("FAIL" if not all(measured.values())
               else "INCOMPLETE" if unmeasured
               else "PASS" if gate["precision_ok"] else "FAIL")
    return {
        "n_gating": n, "fired": fired, "fire_rate": 100.0 * fire_rate,
        "vague_only": sum(p["vague_only"] for p in gating),
        "precision_n": pn, "n_with_gold": n_with_gold,
        "precision": (100.0 * precision) if precision is not None else None,
        "precision_read": precision_read,
        "control_fires": ctl,
        "gate": gate, "verdict": verdict, "pass": verdict == "PASS",
        "populations": [{k: v for k, v in p.items()
                         if not k.startswith("rows_")} for p in pops],
    }


def report(pops: list[dict], summary: dict, *, verbose: bool,
           limit: int = 40) -> bool:
    print("\n=== E4 front-run — relative-date resolution ===\n")
    print(f"{'population':<20}{'n':>7}{'fired':>8}{'rate':>8}"
          f"{'vague':>8}{'prosp':>7}{'prec':>8}{'ctl':>6}")
    for p in pops:
        if not p["n"]:
            continue
        prec = f"{p['precision']:.0f}%" if p["precision"] is not None else "—"
        tag = "" if p["gating"] else "  (non-gating)"
        print(f"{p['name']:<20}{p['n']:>7}{p['fired']:>8}"
              f"{p['fire_rate']:>7.1f}%{p['vague_only']:>8}"
              f"{p.get('prospective', 0):>7}{prec:>8}"
              f"{p['control_fires']:>6}{tag}")

    for p in pops:
        if p["rules"]:
            top = ", ".join(f"{k}={v}" for k, v in list(p["rules"].items())[:6])
            print(f"\n  {p['name']} rules: {top}")
            print(f"  {p['name']} anchors: {p['reanchored']} re-anchored to a "
                  f"date stated in the text")
            scored_rules = [(k, v) for k, v in (p.get("by_rule") or {}).items()
                            if v["scored"]]
            if scored_rules:
                line = "  ".join(
                    f"{k}={100.0 * v['hits'] / v['scored']:.0f}% ({v['hits']}/"
                    f"{v['scored']})" for k, v in scored_rules)
                print(f"  {p['name']} precision by rule: {line}")
                worst = min(scored_rules, key=lambda kv: kv[1]["hits"] / kv[1]["scored"])
                w_name, w = worst
                w_prec = w["hits"] / w["scored"]
                share = w["scored"] / max(p["precision_n"], 1)
                # An aggregate pass carried by rules that rarely fire is not a
                # pass for the boost anyone would actually hit.
                if w_prec < 0.75 and share >= 0.2:
                    print(f"    ⚠ '{w_name}' is {share:.0%} of the scored fires "
                          f"at {w_prec:.0%} precision. The aggregate is an\n"
                          f"      average over a broken dominant rule — read the "
                          f"per-rule row, not the headline.")
            cats = {k: v for k, v in (p.get("by_category") or {}).items()
                    if v["n"] >= 10 and k != "?"}
            if len(cats) > 1:
                ranked = sorted(cats.items(),
                                key=lambda kv: -kv[1]["fired"] / kv[1]["n"])
                head = "  ".join(f"{k}={100.0 * v['fired'] / v['n']:.1f}%"
                                 for k, v in ranked[:5])
                print(f"  {p['name']} by category: {head}")
                top_name, top = ranked[0]
                fire_share = top["fired"] / p["fired"] if p["fired"] else 0.0
                size_share = top["n"] / p["n"] if p["n"] else 0.0
                # Concentration means the category punches ABOVE its weight —
                # the largest category naturally carries the most fires without
                # that meaning anything. Requiring both a majority of fires and
                # a 2× over-representation keeps this from firing on size alone.
                if fire_share >= 0.6 and fire_share >= 2 * size_share:
                    print(f"    ⚠ '{top_name}' is {size_share:.0%} of the "
                          f"questions but {fire_share:.0%} of the fires. The "
                          f"rate is\n      the benchmark's category mix, not "
                          f"how people ask — E4 would be a one-category "
                          f"feature.")
            sides = p.get("miss_sides") or {}
            if sides:
                after, before = sides.get("after", 0), sides.get("before", 0)
                print(f"  {p['name']} misses: gold AFTER the range {after}, "
                      f"BEFORE {before}")
                if after >= 3 * max(before, 1):
                    print("    ⚠ misses are one-directional. A range boost "
                          "matches an item's SPEECH time (when it was said); a\n"
                          "      relative expression in a question is usually "
                          "about EVENT time (when it happened). Correct\n"
                          "      arithmetic, wrong axis — that is a bi-temporal "
                          "gap, not a resolver bug, and no amount of\n"
                          "      resolver work closes it.")
                    # A lopsided split on a handful of misses is a coin flip.
                    # Six one-sided misses is p<0.05; four is p=0.125, which
                    # is a prior worth carrying, not a finding.
                    if after + before < 6:
                        print(f"      (n={after + before} misses — suggestive, "
                              f"not decisive on its own: a {after}/{before} "
                              f"split\n       arises by chance about "
                              f"{2 ** -(after + before) * 2:.0%} of the time. "
                              f"Read it against the other corpora.)")
                elif after + before:
                    # Said explicitly, because the ABSENCE of the warning above
                    # is not self-interpreting: a reader who has seen a
                    # one-directional corpus will carry that diagnosis over to
                    # a balanced one unless the balanced case names itself.
                    print("    → misses are BALANCED, so this is NOT the "
                          "speech-time/event-time axis mismatch. Scattered "
                          "misses mean\n      the resolver is inaccurate on "
                          "specific expressions — a different failure with a "
                          "different (and\n      fixable) remedy. Read them: "
                          "--verbose.")

    if verbose:
        for p in pops:
            if not p["rows_fired"]:
                continue
            print(f"\n── {p['name']}: fired ({len(p['rows_fired'])}) ──")
            # MISSES ARE NEVER TRUNCATED. They are the evidence the revision
            # decision is made from, and a silently capped table hides some of
            # them behind a row count that still reports the full total — the
            # 2026-07-30 LME run displayed 8 of 9 misses because `fired` (47)
            # overran a flat [:40] cap, and the gap was read as a counting bug.
            missed = [r for r in p["rows_fired"] if r.get("gold_covered") is False]
            rest = [r for r in p["rows_fired"] if r.get("gold_covered") is not False]
            shown = missed + rest[:max(0, limit - len(missed))]
            hidden = len(p["rows_fired"]) - len(shown)
            if hidden:
                print(f"  ({hidden} covered/unscored rows hidden; all "
                      f"{len(missed)} misses shown — raise --limit for the rest)")
            for r in shown:
                mark = "" if "gold_covered" not in r else (
                    " ✓" if r["gold_covered"] else " ✗")
                print(f"  [{r['rule']:<16}] {r['start']}..{r['end']}{mark}"
                      f"  «{r['surface']}»")
                print(f"      {r['text'][:110]}")
            if p["rows_vague"]:
                print(f"\n── {p['name']}: vague, unresolvable "
                      f"({len(p['rows_vague'])}) ──")
                for r in p["rows_vague"][:limit]:
                    print(f"  {','.join(r['markers']):<24} {r['text'][:90]}")
        for p in pops:
            for r in p["rows_control_fires"]:
                print(f"\n  ✗ CONTROL FIRE [{p['name']}] {r['rule']}: "
                      f"{r['text'][:100]}")

    s = summary
    prec = f"{s['precision']:.1f}%" if s["precision"] is not None else "n/a"
    print("\n── gating populations (query-side only) ──")
    print(f"  n={s['n_gating']}  fired={s['fired']} ({s['fire_rate']:.1f}%)  "
          f"vague-only={s['vague_only']}  precision={prec} "
          f"(n={s['precision_n']})  control fires={s['control_fires']}")

    if not s["precision_n"]:
        # A silent "n/a" is how a gate gets reported as passed on two criteria.
        # The usual cause is not "this corpus has no gold" but gold plumbing
        # that failed to attach — an unparsed date format drops every gold and
        # returns exactly this confident constant.
        gold_rows = sum(p.get("rows_with_gold", 0) for p in pops if p["gating"])
        print(f"  ⚠ precision UNMEASURED — no fired question carried a dated "
              f"gold session ({gold_rows} of {s['n_gating']} questions in the "
              f"gating populations have gold dates AT ALL).")
        if not gold_rows:
            print("    ↳ ZERO questions have gold dates: this is a LOADER "
                  "failure, not a corpus property. Check the date format in\n"
                  "      `haystack_dates`/`answer_session_ids` — LongMemEval "
                  "ships `2023/05/20 (Sat) 02:21`, not ISO.")
        print("    ↳ criterion 2 is the one LoCoMo FAILED at 20.8% on a "
              "mechanism (speech-time vs event-time) that is corpus-\n"
              "      independent. Leaving it unmeasured here is not neutral: "
              "it is the criterion most likely to fail.")
    elif not s["precision_read"]:
        print(f"  ⚠ precision UNREAD: {s['precision_n']} scored questions is "
              f"below the {_MIN_N_FOR_PRECISION}-question floor — reported, "
              f"not gated.")
    prosp = sum(p.get("prospective", 0) for p in pops if p["gating"])
    if prosp:
        print(f"  → {prosp} PROSPECTIVE questions resolved a forward-facing "
              f"window ('tonight', 'next week') and were NOT\n    fired on: no "
              f"stored past item can fall in them. They are excluded from the "
              f"fire rate AND from\n    the control, because they are neither "
              f"a retrieval range nor a marker-free question.")
    if s["vague_only"] > s["fired"]:
        print(f"  ⚠ vague markers ({s['vague_only']}) OUTNUMBER resolvable "
              f"ones ({s['fired']}): most temporal intent in this corpus "
              f"carries no arithmetic, so no resolver improvement reaches it.")
    content = [p for p in pops if not p["gating"] and p["n"]]
    if content and s["fire_rate"] < 100 * _MIN_FIRE_RATE:
        best = max(content, key=lambda p: p["fire_rate"])
        if best["fire_rate"] >= 100 * _MIN_FIRE_RATE:
            print(f"  ⚠ content-side ({best['name']}) fires at "
                  f"{best['fire_rate']:.1f}% while the query side does not: "
                  f"relative dates EXIST in this corpus but not in its "
                  f"questions. That points at ingest-side normalization, NOT "
                  f"at E4's query-side boost.")

    checks = [
        (s["gate"]["fire_rate_ok"],
         f"fire rate ≥ {_MIN_FIRE_RATE:.0%} ({s['fire_rate']:.1f}%)"),
        (s["gate"]["precision_ok"],
         f"range precision ≥ {_MIN_PRECISION:.0%} ({prec}"
         + ("" if s["precision_n"] and s["precision_read"]
            else ", unmeasured" if not s["precision_n"] else ", unread") + ")"),
        (s["gate"]["no_harm_ok"],
         f"control fires ≤ {_MAX_CONTROL_FIRES} ({s['control_fires']})"),
    ]
    print(f"\n── G-E4a: {s['verdict']} ──")
    for ok, label in checks:
        mark = "✓" if ok else ("?" if label.endswith("unmeasured)") else "✗")
        print(f"  [{mark}] {label}")
    print("  (fire rate is the Track A criterion: a correct boost no query "
          "triggers\n   is dead code with a config flag.)")
    if s["verdict"] == "INCOMPLETE":
        print("  INCOMPLETE is not a pass: 2 of 3 criteria hold and the third "
              "was not measured.\n   Supply a corpus with dated gold, or read "
              "this as the fire-rate criterion alone.")
    return s["pass"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locomo", type=Path, help="locomo10.json")
    ap.add_argument("--dataset", type=Path, help="LongMemEval questions JSON")
    ap.add_argument("--no-turns", action="store_true",
                    help="skip the non-gating content-side population")
    ap.add_argument("--verbose", action="store_true",
                    help="per-row fired/vague/control tables")
    ap.add_argument("--limit", type=int, default=40,
                    help="cap on per-row table length (misses are never capped)")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable summary instead of the report")
    ap.add_argument("--out", type=Path, help="write the full summary JSON")
    args = ap.parse_args()

    if not (args.locomo or args.dataset):
        ap.error("give at least one source: --locomo and/or --dataset")

    pops: list[dict] = []
    if args.locomo:
        questions, turns = load_locomo(args.locomo)
        if not questions:
            ap.error(f"{args.locomo}: no questions with a parseable session date")
        pops.append(measure(questions, name="locomo-questions", gating=True))
        if not args.no_turns:
            pops.append(measure(turns, name="locomo-turns", gating=False))
    if args.dataset:
        rows = load_lme(args.dataset)
        if not rows:
            ap.error(f"{args.dataset}: no questions with a `question_date`")
        pops.append(measure(rows, name="lme-questions", gating=True))

    summary = summarize(pops)
    if args.json:
        print(json.dumps(summary))
    else:
        report(pops, summary, verbose=args.verbose, limit=args.limit)
    if args.out:
        args.out.write_text(json.dumps(
            {"summary": summary,
             "fired": {p["name"]: p["rows_fired"] for p in pops},
             "vague": {p["name"]: p["rows_vague"] for p in pops},
             "control_fires": {p["name"]: p["rows_control_fires"]
                               for p in pops}}, indent=1))
        print(f"\n[out] {args.out}")
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
