"""E5 — anaphora / ellipsis resolution for follow-up queries (Campaign E, Step 2).

WHY this exists: real conversations ask follow-ups that carry no content of their
own — "what did she say about that?", "en de prijs?", "how is that project
going?". Every retrieval tier in `augment()` is lexical-or-vector over the query
STRING, so all of them miss simultaneously: BM25 has nothing but stopwords and a
pronoun to match on, the embedding of a pronoun-only query points nowhere in
particular, `match_known_entities` finds no entity, and `route_predicates`
routes nothing. The referent is sitting one turn up in the session and no tier
can see it.

This module resolves the referent against the session's recent turns BEFORE any
tier runs, so all of them benefit from one fix.

── Two invariants, both deliberate ─────────────────────────────────────────────

1. **APPEND, never replace.** The rewrite is `original + " (context: X, Y)"`, so
   retrieval keeps every original token AND gains the resolved ones. This is the
   additive invariant (`additional_planning.md`: "never suppress-filter on a
   routed signal") applied at the *query* level: a wrong referent can only add
   noise to a candidate pool, it can never delete the tokens that would have
   found the right answer. A full replacement rewrite ("what did Sarah say about
   the MedFlow migration?") is strictly more dangerous for the same upside.

2. **Fire only on a resolvable referent.** A trigger that resolves to nothing
   returns the query byte-identical. The no-harm control the gate reads (zero
   rewrites on self-contained queries) is a property of the trigger set; this
   second condition makes the failure mode "no rewrite", never "rewrite with
   junk".

LME is blind to this by construction — its questions are self-contained
single-shot lookups with no conversational antecedent — so this ships production
value with an expected LME delta of exactly zero. It is the campaign's hedge: it
holds even if the G-F1 fact gate kills E1.

Scope is EN + NL (the project's Latin-script, Dutch-prioritized scope), stdlib
only, no LLM call on the default path. The optional Stage-2 LLM fallback
(`cfg.coref_llm_enabled`, default False) is `LLMClient` Protocol only — HyMem
never ships a backend.

The stopword/pronoun tables are this module's own rather than imports from
`augment._AGG_STOPWORDS`: `augment` imports THIS module (so the dependency can
only run one way), and the two lists serve different jobs — the aggregate one
strips noise from an FTS query, this one decides whether a query says anything
at all.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field

from hymem.config import HyMemConfig
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.query.entities import match_known_entities
from hymem.session import Message

log = logging.getLogger("hymem.query.coref")

# Bound the regex/scan cost regardless of how long a host turn is (mirrors
# `intent._MAX_SCAN_CHARS` — same hardening contract for the same reason).
_MAX_SCAN_CHARS = 2000

# Max referents appended to one query. The clause is evidence for retrieval, not
# a summary: three canonical names is already enough to pull the antecedent's
# turns into every tier, and more only dilutes the BM25 OR-query.
_MAX_REFERENTS = 3

# Min length for a salient-token referent. Two-letter residue ("ok", "ja") is
# noise, and the FTS tokenizers already drop len<2.
_MIN_REFERENT_CHARS = 3

# Time expressions are demoted (never dropped) in salient-token ranking: "I
# migrated the billing service yesterday" → the antecedent of a following "it" is
# the service, not the day. They stay eligible so a window that says nothing else
# still resolves. EN + NL.
_TIME_WORDS = frozenset({
    "today", "yesterday", "tomorrow", "tonight", "morning", "afternoon",
    "evening", "week", "weeks", "month", "months", "year", "years", "day",
    "days", "hour", "hours", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "weekend", "later", "earlier", "soon",
    "vandaag", "gisteren", "morgen", "vanavond", "ochtend", "middag", "avond",
    "weekend", "maand", "maanden", "jaar", "jaren", "dag", "dagen", "uur",
    "uren", "maandag", "dinsdag", "woensdag", "donderdag", "vrijdag",
    "zaterdag", "zondag", "straks", "eerder",
})

# Word-ish token: first char a Unicode letter so accented Dutch words tokenize
# whole (mirrors `entities._TOKEN`), body letters/digits/_/-/.
_TOKEN = re.compile(r"[^\W\d_][\w\-.]{0,40}")

# Function words that carry no referential content. English + Dutch. Used ONLY
# to count how much a query actually says — a query whose content tokens are all
# in here (plus a pronoun) is an anaphoric follow-up, not a lookup.
_STOPWORDS = frozenset({
    # English — determiners, auxiliaries, prepositions, wh-words, fillers
    "a", "an", "the", "and", "or", "but", "so", "then", "than", "as", "if",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "about",
    "into", "over", "under", "again", "also", "just", "still", "yet",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "there", "here", "not", "no", "yes", "ok", "okay", "please", "thanks",
    "any", "some", "else", "more", "much", "many", "very", "really",
    # Interlocutor references. These are NOT anaphora — the speaker and the
    # assistant are never the missing antecedent — but they ARE content-free for
    # the "does this query say anything?" test, so they belong here rather than in
    # `_PRONOUNS`. Leaving them out inflated the content-token count and let
    # ordinary questions ("wat hebben we afgesproken over X?") reach a trigger.
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    # Dutch
    "de", "het", "een", "en", "of", "maar", "dus", "dan", "als", "dat", "dit",
    "is", "zijn", "was", "waren", "wees", "ben", "bent",
    "doe", "doet", "deed", "gedaan", "doen",
    "heb", "hebt", "heeft", "had", "hadden", "hebben",
    "kan", "kun", "kunt", "kunnen", "zal", "zou", "zullen", "moet", "moeten",
    "mag", "mogen", "wil", "wilt", "willen",
    "van", "in", "op", "aan", "te", "voor", "met", "uit", "door", "over",
    "naar", "bij", "om", "ook", "nog", "wel", "niet", "geen", "ja", "nee",
    "wat", "welke", "wie", "wiens", "wanneer", "waar", "waarom", "hoe",
    "daar", "hier", "graag", "bedankt", "al", "iets", "nog", "meer", "erg",
    "ik", "mij", "mijn", "wij", "we", "ons", "onze", "je", "jij", "jou",
    "jouw", "jullie", "u", "uw",
})

# Referential pronouns (EN + NL). A query leaning on one of these has its subject
# somewhere else — in the conversation. "we"/"I"/"you"/"wij"/"ik"/"je" are
# deliberately EXCLUDED: they refer to the interlocutors, who are never the
# missing antecedent, and admitting them would fire on every ordinary question
# ("what did we decide about the migration?").
_PRONOUNS = frozenset({
    # English
    "it", "its", "that", "this", "those", "these", "they", "them", "their",
    "she", "her", "hers", "he", "him", "his", "one", "ones",
    # Dutch. "zijn" is deliberately omitted: the possessive ("zijn laptop") is
    # indistinguishable here from the copula ("wat zijn de kosten?"), so it would
    # mislabel ordinary questions as pronoun-triggered. Those cases are already
    # caught by the ellipsis/demonstrative triggers.
    "hij", "hem", "ze", "zij", "haar", "hen", "hun",
    "dat", "dit", "die", "deze", "diegene", "datzelfde",
})

# Demonstrative determiners: "that project", "die tool" — the noun is generic and
# the *identity* lives in the prior turn. Separate from the bare-pronoun trigger
# because the query here DOES carry a content noun, so the token-count test below
# would never fire on it.
_DEMONSTRATIVES = frozenset({
    "that", "this", "those", "these",       # English
    "die", "dat", "deze", "dit", "dezelfde", "datzelfde",  # Dutch
})

# Content-token ceilings per trigger — the precision knobs. Each is set from the
# eval set's OBSERVED distribution (`benchmarks/coref_eval_set.json`), tightened
# to the largest genuine follow-up rather than left generous: the E5 gate's
# load-bearing half is the no-harm control (zero rewrites on self-contained
# queries), and the first run of the gate showed a generous ceiling firing on two
# ordinary lookups ("what were the three options for the analytics warehouse?" —
# four content tokens, nothing anaphoric about it). Tightening cost no resolution
# (still 30/30) because real follow-ups are genuinely tiny.
#
# Pronoun: a pronoun-bearing query may carry at most this many OTHER content
# tokens ("what did she say about that?" carries one — "say").
_MAX_CONTENT_TOKENS_PRONOUN = 3
# Demonstrative: "does that tool support notebooks?" is the widest real case (3).
_MAX_CONTENT_TOKENS_DEMONSTRATIVE = 3
# Ellipsis is the weakest signal (no pronoun, no demonstrative — just a short
# query), so it gets the tightest ceiling: "and the rollback plan?" carries two.
_MAX_CONTENT_TOKENS_ELLIPSIS = 2
# ...plus a ceiling on TOTAL tokens, because content count alone is not enough:
# "wat hebben we afgesproken over het prijsexperiment?" has only two content
# tokens (the rest are function words) yet is a complete, self-contained question.
# What makes an utterance elliptical is that it is TRUNCATED — the widest genuine
# case in the eval set is four tokens ("and the rollback plan?").
_MAX_TOKENS_ELLIPSIS = 5


@dataclass(frozen=True)
class QueryRewrite:
    """The rewrite decision plus WHY it fired — the same observability contract as
    `AbilitySignal`/`detected_rule`: a production misfire ("why did this query
    gain a MedFlow clause?") must be diagnosable from the result object alone,
    without re-running the patterns.

    `rewritten` is ALWAYS a usable query string: when `changed` is False it is
    the original, byte-identical (callers can use it unconditionally).
    `rule` names the firing trigger ("pronoun", "demonstrative", "ellipsis",
    "llm") or the abstain reason ("disabled", "empty", "non_str", "no_turns",
    "self_contained", "no_referent", "llm_empty").
    `resolved` maps each trigger surface form to the referent chosen for it, so
    the appended clause is traceable to the token that asked for it.
    """

    rewritten: str
    changed: bool
    rule: str
    resolved: dict[str, str] = field(default_factory=dict)


def _prepare(query: object) -> str | None:
    """Normalise + bound host input; None when there is nothing to rewrite.
    NFC so composed vs decomposed Dutch diacritics compare one way, then clipped
    to `_MAX_SCAN_CHARS`. Mirrors `intent._prepare`."""
    if not isinstance(query, str):
        return None
    q = unicodedata.normalize("NFC", query)
    if not q.strip():
        return None
    return q[:_MAX_SCAN_CHARS]


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens, diacritics preserved (the tables above are NFC)."""
    return [m.group(0).lower() for m in _TOKEN.finditer(text)]


def _content_tokens(tokens: list[str]) -> list[str]:
    """Tokens that actually say something: not a stopword, not a pronoun."""
    return [t for t in tokens if t not in _STOPWORDS and t not in _PRONOUNS]


def _demonstrative_pairs(tokens: list[str]) -> list[tuple[str, str]]:
    """(determiner, noun) pairs where a demonstrative precedes a content token —
    "that project", "die tool". The noun is returned so the caller can check
    whether it is itself a known entity (in which case nothing is missing)."""
    pairs: list[tuple[str, str]] = []
    for det, nxt in zip(tokens, tokens[1:]):
        if det in _DEMONSTRATIVES and nxt not in _STOPWORDS and nxt not in _PRONOUNS:
            pairs.append((det, nxt))
    return pairs


def _entity_referents(
    conn: sqlite3.Connection | None, window: list[Message]
) -> list[str]:
    """Canonical entities named in the recent turns, most-recent turn first.

    The graph is its own dictionary (`match_known_entities`), so this is a cheap
    deterministic lookup with no LLM call — and it returns the CANONICAL name,
    which is exactly the token the graph/entity tiers index on. Turns are probed
    newest-first because the antecedent of "that" is nearly always the thing just
    discussed, not the thing five turns ago.
    """
    if conn is None:
        return []
    out: list[str] = []
    for msg in reversed(window):
        try:
            matched = match_known_entities(conn, msg.content or "")
        except sqlite3.Error:
            # A pre-graph / partially-migrated store must degrade to the salient-
            # token path, never fail a query.
            return out
        for canon in matched:
            if canon and canon not in out:
                out.append(canon)
            if len(out) >= _MAX_REFERENTS:
                return out
    return out


def _salient_referents(window: list[Message]) -> list[str]:
    """Fallback referents: salient content tokens of the most recent USER turn
    (any turn if there is no user turn in the window), longest first so the
    distinctive noun beats a short verb. Used when the graph knows no entity in
    the window yet — a brand-new session has no edges, and that is exactly when
    a follow-up is most likely."""
    if not window:
        return []
    users = [m for m in window if (m.role or "") == "user"]
    source = (users or window)[-1]
    seen: list[str] = []
    for tok in _content_tokens(_tokens(source.content or "")):
        if len(tok) >= _MIN_REFERENT_CHARS and tok not in seen:
            seen.append(tok)
    # Salience proxy: non-time-words first, then longest-first. Crude, but stable,
    # explainable, and it gets the noun ahead of the date in the common
    # "I shipped X yesterday" → "how did it go?" shape. `sort` is stable, so ties
    # keep the order they were said in.
    seen.sort(key=lambda t: (t in _TIME_WORDS, -len(t)))
    return seen[:_MAX_REFERENTS]


def _append_clause(query: str, referents: list[str]) -> str:
    """The one rewrite form: append, never replace (see module docstring).

    The parenthesised clause is stripped to bare tokens by `_FTS_SAFE` on every
    FTS path, so the referents enter the BM25 OR-query as ordinary terms while
    remaining readable to an LLM reranker and to a human reading a log line."""
    return f"{query} (context: {', '.join(referents)})"


# Stage 2 (optional, off by default). Kept deliberately tiny: one short call, text
# out, and its output is still only ever APPENDED — we harvest the terms the model
# introduced and add those, so even a hallucinated rewrite cannot delete a token
# the original query would have matched on.
COREF_SYSTEM_PROMPT = (
    "You resolve pronouns and ellipsis in a follow-up question using the "
    "conversation that precedes it. Rewrite the question as a standalone "
    "question with every pronoun replaced by the thing it refers to. Keep it "
    "short. Use only names and terms that appear in the conversation. Output the "
    "rewritten question and nothing else."
)


def _llm_referents(
    query: str, window: list[Message], llm: LLMClient
) -> list[str]:
    """Terms the LLM's standalone rewrite introduces that the query lacked.

    Returning the DIFF (not the rewrite) is what keeps Stage 2 inside invariant
    #1: the caller appends these to the original query, so the LLM can only add
    retrieval signal, never remove it."""
    turns = "\n".join(
        f"{(m.role or 'user')}: {(m.content or '')[:400]}" for m in window
    )
    try:
        raw = llm.complete(
            LLMRequest(
                system=COREF_SYSTEM_PROMPT,
                user=f"Conversation:\n{turns}\n\nFollow-up question: {query}\n\n"
                     f"Standalone question:",
                response_format="text",
                max_tokens=120,
                temperature=0.0,
            )
        )
    except Exception as exc:  # a coref miss must never fail the query
        log.debug("coref.llm_failed error=%s", exc)
        return []
    have = set(_tokens(query))
    out: list[str] = []
    for tok in _content_tokens(_tokens(raw or "")):
        if tok not in have and len(tok) >= _MIN_REFERENT_CHARS and tok not in out:
            out.append(tok)
        if len(out) >= _MAX_REFERENTS:
            break
    return out


def rewrite_query(
    query: object,
    recent_turns: list[Message] | None,
    *,
    cfg: HyMemConfig,
    conn: sqlite3.Connection | None = None,
    llm: LLMClient | None = None,
) -> QueryRewrite:
    """Resolve anaphora/ellipsis in `query` against `recent_turns`, additively.

    Returns a `QueryRewrite` whose `rewritten` is always safe to hand to
    retrieval: unchanged (byte-identical) unless a trigger fired AND a referent
    resolved. `conn` is optional — with it, referents are canonical graph
    entities (the strong path); without it the salient-token fallback is used.

    `conn` is not in the campaign-plan signature; it was added because
    `match_known_entities` (the plan's own resolution rule) is a graph lookup.
    Callers with no store still get the heuristic.
    """
    q = _prepare(query)
    if q is None:
        original = query if isinstance(query, str) else ""
        return QueryRewrite(original, False, "empty" if isinstance(query, str) else "non_str")
    if not cfg.coref_enabled:
        return QueryRewrite(q, False, "disabled")

    if cfg.coref_max_turns <= 0:
        return QueryRewrite(q, False, "no_turns")
    window = list(recent_turns or [])[-cfg.coref_max_turns:]
    # A host that logs the incoming turn BEFORE calling augment() hands us the
    # query itself as the newest "recent turn" — resolving against it would
    # append the query's own tokens back onto it. Drop those.
    stripped = q.strip()
    window = [m for m in window if (m.content or "").strip() != stripped]
    if not window:
        return QueryRewrite(q, False, "no_turns")

    tokens = _tokens(q)
    content = _content_tokens(tokens)
    pronouns = [t for t in tokens if t in _PRONOUNS]
    demos = _demonstrative_pairs(tokens)

    # A query that already names a known entity is self-contained for retrieval
    # purposes — every tier has something to bite on — so it is left alone even
    # if it also contains a pronoun ("is it faster than postgres?").
    query_entities: list[str] = []
    if conn is not None:
        try:
            query_entities = match_known_entities(conn, q)
        except sqlite3.Error:
            query_entities = []

    # Every trigger requires `not query_entities`: a query naming a canonical the
    # graph already knows is self-contained for retrieval purposes (some tier has
    # something to bite on), so it is left alone even when it also carries a
    # pronoun ("is it faster than postgres?").
    #
    # Precedence is by SPECIFICITY, most specific first: a demonstrative paired
    # with a generic noun ("that project") is a narrower signal than a bare
    # pronoun, and the two overlap because "that"/"die"/"dit" are in both tables.
    # Ellipsis is last — it is the weakest signal (short query, nothing else).
    trigger: str | None = None
    surface = ""
    if query_entities:
        trigger = None
    elif demos and len(content) <= _MAX_CONTENT_TOKENS_DEMONSTRATIVE:
        trigger, surface = "demonstrative", " ".join(demos[0])
    elif pronouns and len(content) <= _MAX_CONTENT_TOKENS_PRONOUN:
        trigger, surface = "pronoun", pronouns[0]
    elif (
        not pronouns
        and 0 < len(content) <= _MAX_CONTENT_TOKENS_ELLIPSIS
        and len(tokens) <= _MAX_TOKENS_ELLIPSIS
    ):
        trigger, surface = "ellipsis", " ".join(content)

    if trigger is None:
        return QueryRewrite(q, False, "self_contained")

    referents = _entity_referents(conn, window)
    rule = trigger
    if not referents:
        # Low confidence: the graph knows no entity in the window. Stage 2 (when
        # enabled) gets the one call; otherwise fall back to salient tokens.
        if cfg.coref_llm_enabled and llm is not None:
            referents = _llm_referents(q, window, llm)
            rule = "llm" if referents else "llm_empty"
        if not referents:
            referents = _salient_referents(window)
            if rule == "llm_empty" and referents:
                rule = trigger

    if not referents:
        return QueryRewrite(q, False, "no_referent")

    rewritten = _append_clause(q, referents)
    log.debug("coref.rewrite rule=%s surface=%r referents=%s", rule, surface, referents)
    return QueryRewrite(rewritten, True, rule, {surface: ", ".join(referents)})
