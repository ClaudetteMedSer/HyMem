"""Cheap, dependency-free intent auto-detection for `augment()`.

WHY this exists: `augment()` shapes retrieval per *ability* (MR counting,
TR temporal reasoning) only when a host passes an explicit `ability` hint. In
the BEAM benchmark the harness supplies the ground-truth question type, but the
real Hermes host does NOT — so any benchmark gain that depends on the label
would never fire in production. That gap is benchmark-gaming, which we want to
avoid. This module closes it: when the host gives no ability, we infer a likely
one from the query text using regex/keyword heuristics (stdlib `re` only, no new
deps), so the MR/TR shaping fires automatically in real conversations and
benchmark behaviour converges with production behaviour.

Scope is deliberately narrow. Only MR ("how many X") and TR ("how long between X
and Y", "what happened first") have wired-in shaping today, so those are the
only intents detected; everything else returns None and takes the default
un-shaped path. The project is English + Dutch, Latin-script (Dutch prioritized),
mirroring `_AGG_STOPWORDS` / `_enumerates_items` in augment.py, so each English
pattern has a Dutch counterpart.

Precision over recall: a false positive routes an ordinary question down a
shaping path that still degrades gracefully (MR aggregation just returns a count
+ evidence turns; TR just builds an empty/short timeline), but we still prefer to
miss a borderline case than to mis-shape a normal turn. Patterns that would fire
on conversational filler ("how are you", "how about") are deliberately excluded
below.

MR-vs-TR precedence: the two intents overlap because "how many" also opens TR
questions ("how many days between X and Y?"). The rule is: **TR wins whenever a
counting phrase is paired with a temporal unit (day/week/month/...) AND a
relational anchor (between / after / before / since / until)** — that is a
duration question, not an item count. We therefore test TR FIRST and only fall
through to MR when the TR signal is absent. A bare "how many cards did I add?"
has no temporal unit + anchor, so it stays MR.
"""

from __future__ import annotations

import re

# --- TR (temporal reasoning) ------------------------------------------------
#
# Five independent TR signals, any one of which is sufficient. They are kept
# separate so the duration-counting form (#1) can be checked before MR without
# also loosening the span/ordering/recency forms (#2/#3/#5). The distance form
# (#4) is the one exception — it is checked AFTER MR (see detect_ability), so a
# count question carrying a timeframe stays a count, not a timeline.

# Temporal units (English + Dutch) used to recognise a *duration* counting
# question ("how many days between …") as TR rather than MR. "time"/"times" is
# deliberately EXCLUDED: "how many times before the release did I deploy?" is an
# occurrence *count* (MR), not a duration, so admitting "times" here would
# mis-route it to TR. A genuine "how much time before X" is left as MR — it
# degrades gracefully and is rarer than the count reading.
_TEMPORAL_UNIT = (
    r"(?:day|days|week|weeks|month|months|year|years|hour|hours|"
    r"dag|dagen|week|weken|maand|maanden|jaar|jaren|uur|uren)"
)

# Relational anchors that turn a span into a *between-two-events* question.
# English + Dutch. "since"/"until" included; their Dutch forms "sinds"/"tot".
# Kept as a bare alternation BODY so it can be reused inside larger patterns
# without re-typing (the duration-end set below extends it).
_TR_ANCHOR_BODY = (
    r"between|after|before|since|until|"
    r"tussen|na|voor|voordat|nadat|sinds|tot"
)
_TR_ANCHOR = r"(?:" + _TR_ANCHOR_BODY + r")"

# Perfect/copular auxiliaries (EN + NL) that frame an ongoing duration-to-now:
# "how long HAVE I BEEN …", "how many weeks HAVE I BEEN …". Admitting these as a
# duration end-marker (alongside anchors and "ago") catches the duration reading
# that carries no two-event anchor.
_TR_PERFECT_AUX = (
    r"have|has|had|been|"
    r"heb|hebt|heeft|hebben|ben|bent|geweest"
)

# A *duration end*: a two-event anchor, a deictic "ago"/"geleden" that closes a
# span against now, OR a perfect auxiliary signalling duration-to-now. "how many
# months ago" / "how many weeks have I been …" are duration questions with no
# anchor, so without these the count opener falls through to MR and gets
# mis-shaped — admitting them keeps the temporal reading.
_TR_DUR_END_BODY = _TR_ANCHOR_BODY + r"|ago|geleden|" + _TR_PERFECT_AUX
_TR_DUR_END = r"(?:" + _TR_DUR_END_BODY + r")"

# 1) Duration counting: "how many days between X and Y", "how many months ago",
#    "how many weeks have I been …", "hoeveel dagen tussen". A counting opener + a
#    temporal unit (the COUNTED noun, so it must follow the opener near-immediately
#    — up to two filler words for "how many MORE days") + a duration end. Requiring
#    the unit to be the counted object keeps a count about something ELSE that
#    merely mentions a timeframe ("how many emails did I send a week ago") out of
#    TR — there the unit belongs to the deictic distance, not the count. This is
#    the form that overlaps MR, so it must be recognised as TR and tested first.
_TR_DURATION = re.compile(
    r"\b(?:how\s+(?:many|much|long)|hoe(?:\s+(?:veel|lang))?|hoeveel)\b"
    r"\s+(?:\w+\s+){0,2}?" + _TEMPORAL_UNIT + r"\b"
    r"[\s\S]*?\b" + _TR_DUR_END + r"\b",
    re.IGNORECASE,
)

# 2) Explicit span phrasing without a count: "how long between/after/before",
#    "how long ago", AND the duration-to-now form "how long have I been …" /
#    "how long has it been". The trailing set is a duration end. A bare degree
#    question ("how long is the rope") carries none of these and correctly stays
#    unmatched.
_TR_HOWLONG = re.compile(
    r"\b(?:how\s+long|hoe\s+lang)\b"
    r"[\s\S]*?\b(?:" + _TR_DUR_END_BODY + r")\b",
    re.IGNORECASE,
)

# Ordinal / comparison tokens that mark a *sequence* question, EN + NL. These are
# the tail of an ordering question ("happened FIRST", "graduated FIRST, SECOND").
# English ordinals are guarded by a determiner lookbehind below (English uses the
# SAME word adverbially and adjectivally — "happened first" vs "the first thing").
_TR_ORDINAL_EN = (
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"last|earliest|latest|earlier|later|sooner|most\s+recent(?:ly)?"
)
# Dutch splits cleanly by inflection: the UNINFLECTED forms ("eerst", "laatst")
# are adverbial ("kwam eerst", "kreeg ik het eerst") and need no guard, while the
# INFLECTED forms ("eerste", "laatste", "tweede") are adjectival before a noun
# ("het eerste boek") and DO need the determiner guard.
_TR_ORDINAL_NL_ADV = r"eerst|laatst|eerder|later|vroegst|recentst"
_TR_ORDINAL_NL_ADJ = r"eerste|tweede|derde|vierde|vijfde|laatste|recentste"

# Determiner guards: an ordinal in ADJECTIVAL position ("the FIRST thing", "my
# LAST order", "het EERSTE boek") is NOT a sequence question — the ordinal modifies
# a noun, not a verb/event. Stacked fixed-width negative lookbehinds reject that
# reading while admitting the adverbial one ("happened FIRST", "graduated FIRST").
# Each is `\b`-anchored so it matches only a STANDALONE determiner word — without
# it a verb ending in the determiner's letters collides (Dutch "gebeur-DE eerst"
# would trip a bare `(?<!de )`; English "pizz-A first" a bare `(?<!a )`).
_DET_EN = (
    r"(?<!\bthe )(?<!\ba )(?<!\ban )(?<!\bmy )(?<!\bour )(?<!\byour )(?<!\bhis )"
    r"(?<!\bher )(?<!\btheir )(?<!\bthis )(?<!\bthat )(?<!\bthese )(?<!\bthose )"
    r"(?<!\beach )(?<!\bany )(?<!\bevery )(?<!\bsome )(?<!\bno )"
)
_DET_NL = (
    r"(?<!\bde )(?<!\bhet )(?<!\been )(?<!\bmijn )(?<!\bonze )(?<!\bjouw )"
    r"(?<!\bdeze )(?<!\bdie )(?<!\bdit )(?<!\bdat )(?<!\bzijn )(?<!\bhaar )(?<!\bgeen )"
)

# 3) Ordering / sequence questions: "which event happened first", "who graduated
#    first, second, third", "which device did I get first", "in what order", + NL.
#    A WH-opener (which/who/what + NL welk[e]) followed — across an intervening
#    subject+verb — by an ordinal in adverbial position. The legacy pattern
#    required the verb to sit DIRECTLY after "which" (`which happened first`), so
#    every real question with a noun between ("which EVENT happened first") missed;
#    the `{0,60}` gap closes that, and the determiner lookbehind keeps "the first
#    thing" out.
_TR_ORDER = re.compile(
    r"(?:"
    r"\b(?:which|who|what)\b[\s\S]{0,60}?" + _DET_EN + r"\b(?:" + _TR_ORDINAL_EN + r")\b|"
    r"\bin\s+what\s+order\b|\bwhat(?:'s| is| was)?\s+the\s+order\b|"
    # Dutch — uninflected ordinal needs no guard, inflected does.
    r"\bin\s+welke\s+volgorde\b|\bwat\s+is\s+de\s+volgorde\b|"
    r"\b(?:welke|welk|wie|wat)\b[\s\S]{0,60}?\b(?:" + _TR_ORDINAL_NL_ADV + r")\b|"
    r"\b(?:welke|welk|wie|wat)\b[\s\S]{0,60}?" + _DET_NL + r"\b(?:" + _TR_ORDINAL_NL_ADJ + r")\b"
    r")",
    re.IGNORECASE,
)

# 4) Distance / deictic time reference: "a week ago", "two weeks ago", "last
#    Saturday", "last month", NL "vorige week" / "afgelopen maandag". Pure
#    temporal anchors against now — a question carrying one is asking to recall
#    around that point in time. Checked AFTER the MR count opener (see
#    detect_ability) so "how many times did I X last week" stays a count, not a
#    timeline. Holidays ("Valentine's day") need a gazetteer and are out of scope.
_DAYS_EN = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_DAYS_NL = r"maandag|dinsdag|woensdag|donderdag|vrijdag|zaterdag|zondag"
_TR_DISTANCE = re.compile(
    r"\b(?:"
    r"\w+\s+" + _TEMPORAL_UNIT + r"\s+(?:ago|geleden)|"   # "a week ago", "two months ago"
    + _TEMPORAL_UNIT + r"\s+(?:ago|geleden)|"             # "weeks ago", "days geleden"
    r"(?:last|past)\s+(?:week|month|year|weekend|" + _DAYS_EN + r")|"
    r"(?:vorige?|afgelopen)\s+(?:week|maand|jaar|weekend|" + _DAYS_NL + r")"
    r")\b",
    re.IGNORECASE,
)

# 5) Recency / first-occurrence: "when did I last …", "when was the last time",
#    "when did I first try …", "when did I start using …", and Dutch forms. These
#    pin a single event in time (the most/least recent occurrence) — squarely
#    temporal, and a large slice of BEAM's TR questions that none of the span /
#    ordering forms above catch. Kept high-precision: each branch requires an
#    explicit recency token ("last"/"first"/"start"/"begin"/"voor het laatst"…)
#    next to a "when did/was" or "the … time" frame, so plain item counts ("how
#    many first-edition books") and bare facts never match.
_TR_RECENCY = re.compile(
    r"\b(?:"
    r"when\s+(?:was|is|were|'s)\s+the\s+(?:last|first|most\s+recent)\s+time|"
    r"when\s+did\s+\w+(?:\s+\w+)?\s+(?:last|first|start|begin)\b|"
    r"the\s+(?:last|first|most\s+recent)\s+time\s+(?:i|we|you|he|she|they)\b|"
    # Dutch
    r"wanneer\s+was\s+de\s+(?:laatste|eerste)\s+keer|"
    r"voor\s+het\s+(?:laatst|eerst)|"
    r"de\s+(?:laatste|eerste)\s+keer\s+dat|"
    r"wanneer\s+(?:begon|startte|ben\s+ik\s+begonnen)"
    r")",
    re.IGNORECASE,
)

# --- MR (counting) ----------------------------------------------------------
#
# Plain item-count openers, English + Dutch. Anchored as whole phrases so they
# don't fire on conversational filler. Deliberately NOT included (false-positive
# risk on ordinary chat): a bare "how about", "how are", "how come"; "how much"
# in a cost/degree sense is accepted because MR aggregation degrades gracefully
# and "how much did I spend" is genuinely count-shaped.
_MR_COUNT = re.compile(
    r"\b(?:"
    r"how\s+(?:many|much|often)|"
    r"number\s+of|"
    r"count\s+of|"
    r"total\s+(?:number\s+of|count\s+of|amount\s+of)|"
    # Dutch
    r"hoeveel|"
    r"hoe\s+vaak|"
    r"aantal\b"  # "aantal" = "number of"
    r")\b",
    re.IGNORECASE,
)


def detect_ability(query: str) -> str | None:
    """Infer a likely ability hint from raw query text, or None if nothing fits.

    Only MR (counting) and TR (temporal reasoning) are inferred — they are the
    only intents `augment()` shapes for today. Everything else returns None and
    takes the default un-shaped retrieval path.

    Precedence (see module docstring for the WHY): TR is tested FIRST so that a
    duration-counting question ("how many days between X and Y") is recognised as
    temporal reasoning, not as a plain item count. Only when no TR signal fires
    do we fall through to the MR counting check. This resolves the deliberate
    "how many" overlap in favour of the more specific (temporal) reading.

    Conservative by design: a borderline phrase that matches nothing here simply
    gets the default path. Returns the bare ability code ("MR"/"TR") — the caller
    re-validates it through the same `_ABILITIES` gate as a host-supplied hint.
    """
    if not query:
        return None

    # TR first: the STRONG temporal signals (duration-count, how-long span,
    # ordering, recency) win the overlap with MR.
    if (
        _TR_DURATION.search(query)
        or _TR_HOWLONG.search(query)
        or _TR_ORDER.search(query)
        or _TR_RECENCY.search(query)
    ):
        return "TR"

    # MR: a plain counting opener with no temporal framing. Checked BEFORE the
    # weak distance signal so "how many times did I X last week" stays a count —
    # the timeframe is incidental to a count question, not its subject.
    if _MR_COUNT.search(query):
        return "MR"

    # TR (weak): a bare distance/deictic reference ("a week ago", "last Saturday")
    # with no count opener — a recall-around-a-time question, so shape it temporal.
    if _TR_DISTANCE.search(query):
        return "TR"

    return None
