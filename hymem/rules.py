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
import sqlite3
from dataclasses import dataclass, field

from hymem import redaction
from hymem.dreaming.canonicalize import normalize

log = logging.getLogger("hymem.rules")

RULE_SCOPES = frozenset({"always_on", "contextual"})
RULE_SOURCES = frozenset({"user", "agent_inferred"})


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


def _parse_triggers(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(t) for t in parsed] if isinstance(parsed, list) else []


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
            """
            SELECT id, text, scope, trigger_entities, source,
                   pos_evidence, neg_evidence, valid_at
            FROM rules
            WHERE status = 'active' AND invalid_at IS NULL
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    matched = set(matched_entities)
    out: list[Rule] = []
    for r in rows:
        triggers = _parse_triggers(r["trigger_entities"])
        if r["scope"] == "contextual":
            if not matched or matched.isdisjoint(triggers):
                continue
        out.append(
            Rule(
                id=int(r["id"]),
                text=r["text"],
                scope=r["scope"],
                trigger_entities=triggers,
                source=r["source"],
                pos_evidence=int(r["pos_evidence"]),
                neg_evidence=int(r["neg_evidence"]),
                valid_at=r["valid_at"] or "",
            )
        )
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
