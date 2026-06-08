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
import unicodedata
from dataclasses import dataclass

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

# Activity-duration verbs (EN + NL): "how long did it TAKE to finish", "how many
# days did I SPEND on the trip". These close a single-activity span the same way an
# anchor does — the answer is end-minus-start of ONE activity, a TR timeline — so
# admitting them as a duration end catches the "spend/take" duration forms that
# carry no between/after/ago anchor (router-eval TR misses: camping-trip days,
# book-finishing days). "last"/"lasted" is deliberately EXCLUDED here: it collides
# with the "last week/month" distance adjective; the gain isn't worth that risk.
_TR_DUR_VERB = (
    r"spend|spent|spending|takes?|took|taking|"
    r"besteed|besteedde|besteden|duurt|duurde|duren|kost|kostte"
)

# A *duration end*: a two-event anchor, a deictic "ago"/"geleden" that closes a
# span against now, a perfect auxiliary signalling duration-to-now, OR an
# activity-duration verb (spend/take). "how many months ago" / "how many weeks
# have I been …" / "how long did it take …" are duration questions with no
# between-anchor, so without these the count opener falls through to MR and gets
# mis-shaped — admitting them keeps the temporal reading.
_TR_DUR_END_BODY = _TR_ANCHOR_BODY + r"|ago|geleden|" + _TR_PERFECT_AUX + r"|" + _TR_DUR_VERB
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

# 6) Age-at-event: "how old was I when I moved to the US", "hoe oud was ik toen
#    …". The answer is an age computed from a birth date and a dated event — a
#    two-point temporal calculation, so the TR timeline (birth + event) is the
#    right shaping. Kept tight: requires "how old + copula" THEN a "when/toen"
#    clause, so a bare "how old is my laptop" (no event anchor) never matches.
_TR_AGE = re.compile(
    r"\bhow\s+old\s+(?:was|were|am|is|are)\b[\s\S]{0,40}?\bwhen\b|"
    r"\bhoe\s+oud\s+(?:was|ben|is|waren)\b[\s\S]{0,40}?\b(?:toen|wanneer)\b",
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

# Aggregation framings that carry NO "how many / how much / hoeveel" opener — a
# sum / total / average / percentage question over the user's history ("what's
# the total amount I spent on luxury items", "what percentage of my books are
# fiction", "on average how late do I go to bed"). These take the SAME MR
# aggregate path (count + evidence turns; the host LLM computes the sum/avg/pct
# from the matching turns), but `_MR_COUNT` misses them because it keys on the
# count opener and requires a literal "amount OF" ("total amount I spent" has no
# "of"). Kept a SEPARATE pattern + rule ("mr_aggregate") so its recall recovery
# and false-positive cost are measurable independently of `_MR_COUNT` in
# router_eval. Precision posture is the same as MR overall: a false MR route is
# near-harmless under additive-MR (`mr_aggregate_additive`, default True) — the
# count merely layers on relevance retrieval — so we admit genuinely
# aggregation-shaped phrasings without demanding they also be count-shaped.
_MR_AGGREGATE = re.compile(
    r"\b(?:"
    r"total\s+(?:number|count|amount|sum|spend|spent|cost)|"  # "total amount I spent", "total spent"
    r"total\s+(?:i|we|you)\s+(?:spent|spend|paid|earned|saved)|"  # "the total I spent"
    r"in\s+total|altogether|"
    r"on\s+average|"
    r"what(?:'s|\s+is|\s+was)?\s+the\s+average\b|"
    r"average\s+(?:number|amount|cost|price|spend|spending|time|rating|score)\b|"
    r"what\s+percent(?:age)?\b|percentage\s+of|proportion\s+of|"
    # Dutch
    r"in\s+totaal|totaal\s+(?:aantal|bedrag)|"
    r"gemiddeld\b|gemiddelde\s+(?:aantal|bedrag|tijd|prijs)|"
    r"hoeveel\s+procent|welk\s+percentage|percentage\s+van"
    r")\b",
    re.IGNORECASE,
)

# Explicit "sum across events" cue (EN + NL). Paired with a count opener it marks
# AGGREGATION, not a single duration: "how many hours have I spent playing games
# IN TOTAL" sums per-session durations (MR), whereas "how many days did I spend on
# my camping trip" is ONE span (TR). The activity-duration verbs added to
# `_TR_DUR_END` would otherwise route the former to TR via tr_duration, so this
# cue is checked FIRST (see `detect_ability_signal`) and forces MR when present.
# Deliberately narrow — only the unambiguous total phrasings — so it never steals
# a plain duration ("in total" must be explicit).
_MR_TOTAL_CUE = re.compile(
    r"\b(?:in\s+total|altogether|in\s+all|all\s+up|"
    r"in\s+totaal|bij\s+elkaar|in\s+het\s+totaal)\b",
    re.IGNORECASE,
)


# --- Production hardening ----------------------------------------------------
#
# detect_ability sits on the hot path of EVERY label-free augment() call, run on
# raw host-supplied text — the user's latest turn, which in real Hermes can be a
# multi-kilobyte paste, malformed, or (on a host bug) not a string at all. Two
# guards keep that from degrading or crashing the router:
#
# (1) Bounded scan. The intent opener ("how many…", "when did…", "how long…",
#     "which … first") ALWAYS sits at the start of a question. So we classify a
#     bounded prefix only — which both reflects where the signal lives AND caps
#     the cost of the lazy `[\s\S]*?` / `[\s\S]{0,60}?` bridges, which on
#     unbounded input would let a near-miss (an opener+unit with no following
#     anchor) rescan to end-of-string at every start position. 4096 chars is far
#     beyond any real opener-to-anchor span (~600 words) yet bounds worst-case
#     regex work regardless of input size.
# (2) Type/empty tolerance. A non-str (None, bytes) or blank turn is "no
#     detectable intent", returned as an abstain rather than raised — the router
#     must never be the thing that crashes the host on a malformed turn.
_MAX_SCAN_CHARS = 4096


def _prepare(query: object) -> str | None:
    """Normalise + bound host input for classification; None when there is
    nothing to classify (non-str or blank). NFC-normalises so composed vs
    decomposed Latin diacritics (Dutch ë, ï, …) match a single way, then clips to
    `_MAX_SCAN_CHARS` so the regex cost is bounded no matter how long the turn."""
    if not isinstance(query, str):
        return None
    q = unicodedata.normalize("NFC", query)
    if not q.strip():
        return None
    return q[:_MAX_SCAN_CHARS]


@dataclass(frozen=True)
class AbilitySignal:
    """The router's decision plus WHY it fired — production observability.

    `ability` is the wired shaping target ("MR"/"TR") or None. `rule` names the
    branch that produced it so a misroute in real Hermes is diagnosable without
    re-deriving which pattern matched: the firing rules ("mr_total",
    "tr_duration", "tr_howlong", "tr_order", "tr_recency", "tr_age", "mr_count",
    "mr_aggregate", "tr_distance"), or an abstain reason ("none" = matched
    nothing, "empty" = blank string, "non_str" = host passed a non-string). It
    mirrors the `why_retrieved` chips' role: make a routing decision auditable."""

    ability: str | None
    rule: str


def detect_ability_signal(query: object) -> AbilitySignal:
    """Classify `query` AND report which rule decided — the observable core that
    `detect_ability` wraps. Precedence is unchanged (see module docstring): the
    STRONG TR signals (duration-count, how-long span, ordering, recency) are
    tested first and win the "how many" overlap with MR; a plain counting opener
    is MR; a bare distance/deictic anchor with no count opener is weak TR. Input
    is hardened via `_prepare` (bounded scan, type/empty tolerance)."""
    q = _prepare(query)
    if q is None:
        return AbilitySignal(None, "empty" if isinstance(query, str) else "non_str")

    # Aggregation guard FIRST: an explicit "in total" sum cue on a counting opener
    # is summing across events (MR), not one duration — it must win before the
    # activity-duration verbs in `_TR_DUR_END` route "how many hours have I spent
    # … in total" to TR. Without the count opener it is left to `_MR_AGGREGATE`.
    if _MR_TOTAL_CUE.search(q) and _MR_COUNT.search(q):
        return AbilitySignal("MR", "mr_total")

    # TR first: the STRONG temporal signals win the overlap with MR. Split per
    # rule (vs one boolean OR) purely so the firing branch is nameable.
    if _TR_DURATION.search(q):
        return AbilitySignal("TR", "tr_duration")
    if _TR_HOWLONG.search(q):
        return AbilitySignal("TR", "tr_howlong")
    if _TR_ORDER.search(q):
        return AbilitySignal("TR", "tr_order")
    if _TR_RECENCY.search(q):
        return AbilitySignal("TR", "tr_recency")
    if _TR_AGE.search(q):
        return AbilitySignal("TR", "tr_age")

    # MR: a plain counting opener with no temporal framing. Checked BEFORE the
    # weak distance signal so "how many times did I X last week" stays a count —
    # the timeframe is incidental to a count question, not its subject.
    if _MR_COUNT.search(q):
        return AbilitySignal("MR", "mr_count")

    # MR (aggregation): a sum/total/average/percentage question with no count
    # opener ("total amount I spent", "what percentage of …", "on average …").
    # Same MR aggregate path; checked after the count opener so a question with
    # both ("how many in total") reports the stronger `mr_count` rule.
    if _MR_AGGREGATE.search(q):
        return AbilitySignal("MR", "mr_aggregate")

    # TR (weak): a bare distance/deictic reference ("a week ago", "last Saturday")
    # with no count opener — a recall-around-a-time question, so shape it temporal.
    if _TR_DISTANCE.search(q):
        return AbilitySignal("TR", "tr_distance")

    return AbilitySignal(None, "none")


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
    Thin wrapper over `detect_ability_signal`; use that when you also want the
    firing rule for logging/observability.
    """
    return detect_ability_signal(query).ability
