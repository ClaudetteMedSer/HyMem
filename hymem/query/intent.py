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
# Three independent TR signals, any one of which is sufficient. They are kept
# separate so the duration-counting form (#1) can be checked before MR without
# also loosening the ordering forms (#2/#3).

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
_TR_ANCHOR = (
    r"(?:between|after|before|since|until|"
    r"tussen|na|voor|voordat|nadat|sinds|tot)"
)

# 1) Duration counting: "how many days between X and Y", "hoeveel dagen tussen".
#    A counting opener + a temporal unit + a relational anchor. This is the form
#    that overlaps MR, so it must be recognised as TR and tested first.
_TR_DURATION = re.compile(
    r"\b(?:how\s+(?:many|much|long)|hoe(?:\s+(?:veel|lang))?|hoeveel)\b"
    r"[\s\S]*?\b" + _TEMPORAL_UNIT + r"\b"
    r"[\s\S]*?\b" + _TR_ANCHOR + r"\b",
    re.IGNORECASE,
)

# 2) Explicit span phrasing without a count: "how long between/after/before",
#    "how long ago", "hoe lang tussen / na / voor / geleden".
_TR_HOWLONG = re.compile(
    r"\b(?:how\s+long|hoe\s+lang)\b"
    r"[\s\S]*?\b(?:" + _TR_ANCHOR.strip("()?:") + r"|ago|geleden)\b",
    re.IGNORECASE,
)

# 3) Ordering / sequence questions: "what happened first/before/after",
#    "which came first", "in what order", and Dutch equivalents. These ask for a
#    chronology directly. "in what order" / "in welke volgorde" are strong,
#    unambiguous order cues.
_TR_ORDER = re.compile(
    r"\b(?:"
    r"what\s+(?:happened|came|did\s+\w+\s+do)\s+(?:first|before|after)|"
    r"which\s+(?:one\s+)?came\s+first|"
    r"which\s+(?:happened|was)\s+(?:first|earlier|later)|"
    r"in\s+what\s+order|"
    r"what(?:'s| is| was)?\s+the\s+order|"
    # Dutch
    r"wat\s+gebeurde\s+(?:er\s+)?(?:eerst|als\s+eerste|eerder|later)|"
    r"wat\s+(?:kwam|was)\s+(?:er\s+)?eerst|"
    r"in\s+welke\s+volgorde|"
    r"welke\s+(?:kwam|gebeurde|was)\s+(?:er\s+)?(?:eerst|eerder|later)"
    r")\b",
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

    # TR first: any of the three temporal signals (duration-count, how-long span,
    # or ordering) wins the overlap with MR.
    if (
        _TR_DURATION.search(query)
        or _TR_HOWLONG.search(query)
        or _TR_ORDER.search(query)
    ):
        return "TR"

    # MR: a plain counting opener with no temporal framing.
    if _MR_COUNT.search(query):
        return "MR"

    return None
