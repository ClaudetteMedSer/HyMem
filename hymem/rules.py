"""`always_on` Rules — a first-class standing-imperative tier (Idea B, schema v23).

HyMem's "always loaded" layer was scattered across the MEMORY.md / USER.md
auto-sections, `profile_entries`, and the closed-vocabulary `user_profile` slots —
none of which is a clean standing imperative ("always run the tests before
pushing", "never suggest Docker"). Those are *rules*, not facts/preferences, and
they competed for the capped insight/profile budgets. This module gives them a
dedicated home (BrainDB's `always_on` rule node), injected into every context call.

Two scopes:
  - ``always_on``   — loaded unconditionally into every ``augment()`` ctx.
  - ``contextual``  — loaded only when a ``trigger_entities`` member overlaps the
                      call's ``matched_entities`` (so a redis-specific rule stays
                      dormant until redis is in play).

Rows are bi-temporal like ``knowledge_graph`` / ``user_profile`` (v15/v18): a rule
is never overwritten. A contradicting rule closes the prior's validity interval
(``invalid_at`` + ``status='retracted'``) so history is auditable. Re-asserting an
identical rule reinforces (``pos_evidence++``) rather than duplicating. Text is
redaction-scrubbed at this persist chokepoint, mirroring the ``user_profile`` and
``log_message`` chokepoints, so no consumer re-scrubs.

Purely ADDITIVE: ``load_rules`` is a standalone SELECT feeding its own ``ctx.rules``
tier — it never consumes a slot from message/chunk/episode/graph retrieval, so a
populated rule set cannot crowd gold turns out of any other tier. Degrades to ``[]``
on a pre-v23 store. Deliberately NOT wired into the RAPTOR digest's ``_anchor_facts``
(additional_planning.md §0): coupling rule edits to digest regeneration is undesired.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field

from hymem import redaction
from hymem.dreaming.canonicalize import normalize

log = logging.getLogger("hymem.rules")

RULE_SCOPES = frozenset({"always_on", "contextual"})
RULE_SOURCES = frozenset({"user", "agent_inferred"})

# ── marker → rule routing (Idea B write-side) ───────────────────────────────
# Which behavioral_marker kinds can become a standing rule. `preference` is
# excluded on purpose: a preference is a taste ("likes dark mode"), not an
# imperative the agent must obey on every turn — it belongs in profile_entries,
# not in a rule injected into every context.
_RULE_KINDS = frozenset({"rejection", "style", "correction"})


def is_rule_eligible_kind(kind: str) -> bool:
    """Whether a behavioral_marker kind may become a standing rule AT ALL — the
    single eligibility gate BOTH routing paths must share.

    This is ORTHOGONAL to durability. `preference` is excluded on purpose: a
    preference is a taste that belongs in `profile_entries` (the profile tier
    already surfaces it), so minting it ALSO as an always_on rule is redundant and
    inflates every context. The LLM durability tagger answers a DIFFERENT question
    ("standing vs one-off"), and a *durable* preference ("prefers dark mode") is
    still not a rule — so the kind gate must run BEFORE the tagger, not be left to
    it. The lexical path gets this for free via `rule_scope_for_marker`; the LLM
    path (`rules_extract.route_decisions`) MUST apply it explicitly, or preference
    markers flood the tagger and mint hundreds of false rules (box full run:
    1,768 preferences → 1,476 FPs, precision 5.8%)."""
    return kind in _RULE_KINDS


# High-precision imperative test. A rule injects into EVERY call, so a false
# positive is costly. The cue vocabulary is the standing-directive words a
# durable rule uses ("never use X", "avoid Y", "prefer Z").
#
# `rejects?`/`refuses?` were REMOVED (2026-07-27, real-marker precision run): the
# extractor writes EVERY rejection marker as "User rejects X" in the present
# tense, one-off or standing alike, so those cues fired on 100% of rejection
# markers — zero discriminative power, just re-detecting the kind. That let
# one-offs ("User rejects the LOWER() patch", "…rejects LoCoMo as a benchmark")
# mint rules. The earlier present-vs-past-tense heuristic assumed the extractor
# varies tense with durability; it does not. A rejection now routes ONLY on a
# genuine imperative modal (never/always/must/should/…), which a durable
# avoidance carries and a one-off does not.
_DIRECTIVE_RE = re.compile(
    r"\b(always|never|do\s*not|don'?t|must(?:\s*not)?|avoids?|"
    r"forbids?|requires?|prefers?|instead|only\s+ever|no\s+longer|stop|ensure|"
    r"make\s+sure|should\s+(?:always|never))\b",
    re.IGNORECASE,
)


def rule_scope_for_marker(kind: str, statement: str) -> str | None:
    """Classify a behavioral marker: return the rule scope it should become, or
    None if it is not a standing imperative. Deterministic, LLM-free — this is
    the whole routing policy, and the metric the extraction probe gates on.

    Policy (precision-first, tuned against `rule_extraction_probe.py`):
      - `style` markers are inherently durable directives → routed on kind alone.
      - `rejection` / `correction` must carry an imperative modal cue
        (`_DIRECTIVE_RE`), which separates a standing avoidance ("never use
        Mongo") from a one-off event ("User rejects the LOWER() patch", "the
        deadline is March 3") that must NOT become a rule injected into every
        call. The cue is the modal itself, NOT the word "rejects" — see
        `_DIRECTIVE_RE`'s note on why that token was removed.
      - `preference` is a taste, never a rule (excluded from `_RULE_KINDS`).

    Everything routed is `always_on`: an agent-inferred rule has no explicit
    trigger entities (the told-path `add_rule` carries those), so it cannot be
    `contextual`."""
    if kind not in _RULE_KINDS:
        return None
    if kind == "style":
        return "always_on"
    return "always_on" if _DIRECTIVE_RE.search(statement or "") else None


def route_markers_to_rules(conn: sqlite3.Connection, cfg, llm=None) -> int:
    """Promote the durable sub-slice of UNCONSOLIDATED behavioral markers into
    `agent_inferred` rules. Returns the number of rules minted/reinforced.

    Runs during Phase-2 consolidation, BEFORE `consolidate_profile` stamps the
    markers consolidated (both read `consolidated_at IS NULL`). The routing
    instrument is `cfg.rules_extraction_mode` (`lexical` = deterministic, no LLM;
    `llm`/`llm_fastpath` = one batched durability call via `rules_extract`). The
    LLM arms rewrite each kept marker to a CANONICAL imperative, so `add_rule`'s
    text-UPSERT collapses paraphrases and accumulates `pos_evidence` — the
    recurrence signal repetition-gated promotion will read. Idempotent; degrades
    to 0 on a pre-v23 store; a bad/duplicate statement never blocks the rest."""
    try:
        rows = conn.execute(
            "SELECT kind, statement FROM behavioral_markers "
            "WHERE consolidated_at IS NULL ORDER BY id"
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    if not rows:
        return 0

    from hymem import rules_extract

    markers = [(r["kind"], r["statement"]) for r in rows]
    decisions = rules_extract.route_decisions(
        markers,
        mode=getattr(cfg, "rules_extraction_mode", "lexical"),
        llm=llm,
        confidence_min=getattr(cfg, "rules_extraction_confidence_min", 0.75),
        batch_size=getattr(cfg, "rules_extraction_batch_size", 20),
    )
    minted = 0
    for d in decisions:
        if not d.route:
            continue
        try:
            add_rule(conn, d.text, scope=d.scope, source="agent_inferred")
            minted += 1
        except (ValueError, sqlite3.OperationalError):
            continue  # a bad/duplicate statement never blocks the rest
    if minted:
        log.info("rules.extracted count=%d mode=%s (agent_inferred from markers)",
                 minted, getattr(cfg, "rules_extraction_mode", "lexical"))
    return minted


@dataclass
class RuleCandidate:
    """A standing rule the durability tagger INFERRED from behavioral markers,
    offered for human/agent confirmation via `add_rule` — NEVER auto-persisted.

    This is the candidate-suggestion surface: write-side auto-injection didn't
    clear the precision gate on real markers (the tagger is a strong high-RECALL
    detector but over-fires on one-offs — "rejects LoCoMo" reads standing without
    the recurrence a live store would show), so the human confirming is the
    precision gate. Each candidate carries the corroboration a reviewer needs:
    ``marker_count`` supporting markers over ``session_count`` distinct sessions
    (more sessions = more durable), the ``kinds`` that fed it, the tagger's max
    ``confidence``, the raw ``supporting_statements`` to eyeball, and
    ``already_active`` (its canonical text already matches an active rule, i.e.
    confirming would reinforce, not add)."""

    text: str
    scope: str
    confidence: float
    kinds: list[str]
    marker_count: int
    session_count: int
    supporting_statements: list[str] = field(default_factory=list)
    already_active: bool = False


def suggest_rules_from_markers(
    conn: sqlite3.Connection,
    cfg,
    llm,
    *,
    limit: int | None = None,
    mode: str = "llm",
    confidence_min: float | None = None,
) -> list["RuleCandidate"]:
    """Propose standing rules from UNCONSOLIDATED behavioral markers — the manual,
    NO-WRITE twin of `route_markers_to_rules`. Reads the same marker set (plus a
    session join for recurrence), runs the durability tagger, collapses
    paraphrases by canonical text, and returns ranked candidates for confirmation.
    Nothing is persisted.

    Intended flow: log a session, call this to review the standing rules it
    implies, adopt the good ones via `add_rule`, THEN dream (which consolidates
    the markers). Because it reads `consolidated_at IS NULL`, it operates on the
    fresh signal since the last dream — bounded cost, natural cadence. Requires an
    LLM (the tagger); `[]` on a pre-v23 store or when no marker clears the tagger.

    Ranking surfaces the most adoptable first: NOVEL before already-active, then
    most-corroborated (distinct sessions, then markers), then most confident.
    De-dups against active rules via `already_active` so re-review doesn't
    re-propose what's already adopted (a *declined* one-off can still recur — a
    dismissed-candidate store is the natural v2, gated behind real usage)."""
    if confidence_min is None:
        confidence_min = getattr(cfg, "rules_extraction_confidence_min", 0.75)
    try:
        rows = conn.execute(
            "SELECT bm.kind AS kind, bm.statement AS statement, c.session_id AS session_id "
            "FROM behavioral_markers bm LEFT JOIN chunks c ON bm.chunk_id = c.id "
            "WHERE bm.consolidated_at IS NULL ORDER BY bm.id"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    if not rows:
        return []

    from hymem import rules_extract

    markers = [(r["kind"], r["statement"]) for r in rows]
    sessions = [r["session_id"] for r in rows]
    decisions = rules_extract.route_decisions(
        markers,
        mode=mode,
        llm=llm,
        confidence_min=confidence_min,
        batch_size=getattr(cfg, "rules_extraction_batch_size", 20),
    )

    # Normalized text of every active rule → flag candidates that would reinforce
    # rather than add (so a reviewer isn't offered rules already in force).
    active_norm = {normalize(r.text) for r in list_rules(conn)}

    groups: dict[str, dict] = {}
    for (kind, stmt), sess, d in zip(markers, sessions, decisions):
        if not d.route:
            continue
        key = normalize(d.text)
        g = groups.setdefault(key, {
            "text": d.text, "scope": d.scope, "confidence": 0.0,
            "kinds": set(), "statements": [], "sessions": set(),
        })
        g["confidence"] = max(g["confidence"], d.confidence)
        g["kinds"].add(kind)
        g["statements"].append(stmt)
        if sess:
            g["sessions"].add(sess)

    candidates = [
        RuleCandidate(
            text=g["text"], scope=g["scope"], confidence=g["confidence"],
            kinds=sorted(g["kinds"]), marker_count=len(g["statements"]),
            session_count=len(g["sessions"]),
            supporting_statements=g["statements"],
            already_active=key in active_norm,
        )
        for key, g in groups.items()
    ]
    # Novel first, then most-corroborated (sessions, markers), then confident.
    candidates.sort(key=lambda c: (c.already_active, -c.session_count,
                                   -c.marker_count, -c.confidence, c.text))
    return candidates[:limit] if limit is not None else candidates


def list_rules(conn: sqlite3.Connection) -> list["Rule"]:
    """Every ACTIVE rule (always_on AND contextual), ignoring triggers — the
    "show me the rulebook" reader behind `HyMem.rules()` / the MCP tool. Unlike
    `load_rules` (a per-call, trigger-gated retrieval), this lists the whole
    active set. Read-only; `[]` on a pre-v23 store."""
    try:
        rows = conn.execute(
            f"SELECT {_RULE_COLS} FROM rules "
            "WHERE status = 'active' AND invalid_at IS NULL ORDER BY scope, id"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = [_row_to_rule(r) for r in rows]
    out.sort(key=lambda x: (x.scope != "always_on", x.id))
    return out


@dataclass
class Rule:
    """One standing imperative. ``trigger_entities`` is the parsed JSON list
    (canonicalized), empty for ``always_on`` rules."""

    id: int
    text: str
    scope: str
    trigger_entities: list[str] = field(default_factory=list)
    source: str = "user"
    pos_evidence: int = 1
    neg_evidence: int = 0
    valid_at: str = ""


_RULE_COLS = (
    "id, text, scope, trigger_entities, source, "
    "pos_evidence, neg_evidence, valid_at"
)


def _parse_triggers(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(t) for t in parsed] if isinstance(parsed, list) else []


def _row_to_rule(r: sqlite3.Row) -> "Rule":
    return Rule(
        id=int(r["id"]),
        text=r["text"],
        scope=r["scope"],
        trigger_entities=_parse_triggers(r["trigger_entities"]),
        source=r["source"],
        pos_evidence=int(r["pos_evidence"]),
        neg_evidence=int(r["neg_evidence"]),
        valid_at=r["valid_at"] or "",
    )


def load_rules(
    conn: sqlite3.Connection,
    matched_entities: list[str],
    cap: int | None = None,
) -> list[Rule]:
    """ACTIVE rules for this call, ``always_on`` first then by id (stable insert
    order). ``contextual`` rules are included only when a trigger overlaps
    ``matched_entities``. Read-only; returns ``[]`` on a pre-v23 store."""
    try:
        rows = conn.execute(
            f"SELECT {_RULE_COLS} FROM rules "
            "WHERE status = 'active' AND invalid_at IS NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    matched = set(matched_entities)
    out: list[Rule] = []
    for r in rows:
        if r["scope"] == "contextual":
            triggers = _parse_triggers(r["trigger_entities"])
            if not matched or matched.isdisjoint(triggers):
                continue
        out.append(_row_to_rule(r))
    # always_on before contextual, then stable by id (insertion order).
    out.sort(key=lambda x: (x.scope != "always_on", x.id))
    return out[:cap] if cap is not None else out


def add_rule(
    conn: sqlite3.Connection,
    text: str,
    *,
    scope: str = "always_on",
    trigger_entities: list[str] | None = None,
    source: str = "user",
    supersedes: int | None = None,
) -> int:
    """Insert (or reinforce) a rule; return its id.

    - ``text`` is stripped and redaction-scrubbed here (the persist chokepoint).
    - ``supersedes`` closes that prior rule's validity interval first, so a
      contradicting instruction supersedes rather than overwrites (bi-temporal).
    - Re-asserting an identical (post-scrub) text reinforces ``pos_evidence`` and
      revives the row if it was retracted, never creating a duplicate (``text`` is
      UNIQUE, driving the UPSERT).
    - ``contextual`` triggers are canonicalized so they compare against the
      already-canonical ``matched_entities`` at load time.
    """
    text = redaction.redact(text.strip())
    if not text:
        raise ValueError("rule text must be non-empty")
    if scope not in RULE_SCOPES:
        raise ValueError(f"unknown rule scope {scope!r} (expected one of {sorted(RULE_SCOPES)})")
    if source not in RULE_SOURCES:
        raise ValueError(f"unknown rule source {source!r} (expected one of {sorted(RULE_SOURCES)})")

    triggers = sorted({normalize(t) for t in (trigger_entities or []) if t and t.strip()})
    triggers_json = json.dumps(triggers)

    if supersedes is not None:
        retract_rule(conn, supersedes)

    conn.execute(
        """
        INSERT INTO rules(text, scope, trigger_entities, source, valid_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(text) DO UPDATE SET
            pos_evidence     = pos_evidence + 1,
            status           = 'active',
            invalid_at       = NULL,
            scope            = excluded.scope,
            trigger_entities = excluded.trigger_entities,
            source           = excluded.source
        """,
        (text, scope, triggers_json, source),
    )
    row = conn.execute("SELECT id FROM rules WHERE text = ?", (text,)).fetchone()
    return int(row["id"])


def retract_rule(conn: sqlite3.Connection, rule_id: int) -> None:
    """Close an active rule's validity interval (``status='retracted'`` +
    ``invalid_at``). Idempotent: an already-retracted rule keeps its close date."""
    conn.execute(
        """
        UPDATE rules
        SET status = 'retracted',
            invalid_at = COALESCE(invalid_at, CURRENT_TIMESTAMP)
        WHERE id = ? AND status = 'active'
        """,
        (rule_id,),
    )
