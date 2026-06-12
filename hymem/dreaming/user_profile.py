"""Typed user-profile extraction and consumption (Stage 1 / P4, schema v18).

The knowledge graph's 18-predicate vocabulary is tech-domain, so durable
personal facts ("user is a bedrijfsarts in Amsterdam named Atta") never become
edges — the digest stays honest but identity-thin, and incidental personal
facts are unreachable at query time. This module closes that gap with a CLOSED
slot vocabulary extracted from USER turns only: the schema-constrained, safe
version of the shelved open-ended incidental extraction. Unknown slots are
rejected twice — at :func:`validate_profile_items` and by the ``user_profile``
table's CHECK constraint — so the LLM can never invent a slot.

Rows are bi-temporal like the knowledge graph (valid_at / invalid_at, the v15
semantics in :mod:`hymem.dreaming.bitemporal`):

  - ``valid_at``   opens when a fact is asserted — the evidence message's
                   ``created_at`` world date, falling back to insert time
                   (mirrors ``stamp_validity``).
  - ``invalid_at`` closes when a conflicting value supersedes the row — the
                   new evidence's world date, falling back to the flip time
                   (mirrors ``stamp_invalidation``).

Supersession policy: single-valued slots (name, role, employer, location,
age_birthday) hold one active value — a new conflicting value closes the old
row. ``relationship`` is multi-valued but keyed per person (slot_key), so a
new value for the SAME person supersedes while different people accumulate.
The remaining multi-valued slots (language, possession, health_condition,
recurring_activity) accumulate. Re-asserting an already-active value never
duplicates the row — it reinforces confidence and is otherwise a no-op.

Redaction: values pass through :func:`hymem.redaction.redact` at THIS persist
chokepoint (mirroring the ``log_message`` ingest chokepoint that already
scrubbed the source turns), so every consumer — the digest's VERIFIED FACTS
anchor, augment()'s ``ctx.user_profile`` tier, and ``HyMem.profile()`` —
inherits scrubbed text without a second pass. This matters for sensitive
slots like ``health_condition``, whose values must never resurface embedded
secrets/PII the LLM lifted from context.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

from hymem import redaction
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import USER_PROFILE_SYSTEM, USER_PROFILE_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.user_profile")

# Pinned prompt version, following the salt convention in aggregate.py
# (cluster.v2 / root.v4): bump when USER_PROFILE_SYSTEM / the user template
# change materially so operators can attribute extraction shifts to the prompt.
# profile.v2: aboutness test + per-slot tightening (health_condition, employer,
# possession) after profile.v1 failed the on-box precision gate at ~8%.
PROFILE_PROMPT_VERSION = "profile.v2"

# The CLOSED slot vocabulary, in the priority order consumers render
# (identity first). Must stay in lockstep with the CHECK constraint in
# migration 018_user_profile.sql / schema.sql.
PROFILE_SLOTS: tuple[str, ...] = (
    "name",
    "role",
    "employer",
    "location",
    "age_birthday",
    "language",
    "relationship",
    "possession",
    "health_condition",
    "recurring_activity",
)
_SLOT_ORDER = {slot: i for i, slot in enumerate(PROFILE_SLOTS)}
_SLOTS_SET = frozenset(PROFILE_SLOTS)

# Slots that hold exactly one active value: a new conflicting value closes the
# old row's validity interval. 'relationship' behaves the same but PER slot_key
# (one active value per person); the remaining multi-valued slots accumulate.
SINGLE_VALUED_SLOTS = frozenset({"name", "role", "employer", "location", "age_birthday"})


@dataclass
class ProfileExtraction:
    """Validated profile items ready to persist. Empty list = LLM returned
    nothing usable; None at the call boundary = no user turns to extract from.
    Mirrors EpisodesExtraction."""

    items: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ProfileEntry:
    """One ACTIVE (invalid_at IS NULL) user-profile row, as returned by
    `HyMem.profile()` and the `ctx.user_profile` augment tier. `slot_key` is
    the parameterizing key for keyed slots (relationship → the other person);
    None for unkeyed slots. `valid_at` is the world date the fact became true
    (bi-temporal v15 semantics)."""

    slot: str
    slot_key: str | None
    value: str
    confidence: float
    evidence_message_id: int | None
    valid_at: str


def fetch_user_turns(conn: sqlite3.Connection, session_id: str) -> list[tuple[int, str]]:
    """The session's USER turns as (message_id, content), in conversation
    order. User turns ONLY — the profile must never assert a fact the
    assistant introduced (a confabulated identity would otherwise launder
    itself into ground truth via the anchor)."""
    rows = conn.execute(
        "SELECT id, content FROM messages "
        "WHERE session_id = ? AND role = 'user' ORDER BY id",
        (session_id,),
    ).fetchall()
    return [(int(r["id"]), r["content"]) for r in rows]


def build_profile_user_prompt(turns: list[tuple[int, str]], *, max_chars: int) -> str:
    """Render the EXACT extraction user prompt for a session's user turns.

    Shared with `benchmarks/profile_prompt_dump.py` so the manual front-run
    precision gate scores the very prompt production runs."""
    combined = "\n\n".join(f"[msg {mid}] {text}" for mid, text in turns)
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return USER_PROFILE_USER_TEMPLATE.format(text=combined)


def extract_user_profile(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
    *,
    max_chars: int,
    max_items: int,
) -> ProfileExtraction | None:
    """One LLM call extracting typed profile items from the session's USER
    turns. Returns None when the session has no user turns. No write
    transaction held; persist via `persist_user_profile` inside one."""
    turns = fetch_user_turns(conn, session_id)
    if not turns:
        return None
    valid_ids = {mid for mid, _ in turns}

    request = LLMRequest(
        system=USER_PROFILE_SYSTEM,
        user=build_profile_user_prompt(turns, max_chars=max_chars),
        response_format="json",
    )
    raw = llm.complete(request)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ProfileExtraction()
    return ProfileExtraction(
        items=validate_profile_items(data, valid_ids, max_items=max_items)
    )


def validate_profile_items(
    data: object, valid_message_ids: set[int], *, max_items: int
) -> list[dict]:
    """Strictly validate raw LLM items into clean dicts ready to persist.

    Drops: non-dict items, slots outside the CLOSED vocabulary (the LLM must
    never mint a slot — the DB CHECK is the second line of defense), empty
    values, evidence_message_id missing or not among the input turns
    (hallucinated provenance), and confidence missing or outside [0, 1].
    slot_key is required for 'relationship' (supersession is keyed on it) and
    stripped from every other slot so unkeyed supersession stays deterministic.
    Returns [] for any non-list payload; at most `max_items` items survive."""
    if not isinstance(data, list):
        return []
    items: list[dict] = []
    for item in data:
        if len(items) >= max_items:
            break
        if not isinstance(item, dict):
            continue
        slot = item.get("slot")
        if not isinstance(slot, str):
            continue
        slot = slot.strip().lower()
        if slot not in _SLOTS_SET:
            continue
        value = item.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        mid = item.get("evidence_message_id")
        if isinstance(mid, bool) or not isinstance(mid, int) or mid not in valid_message_ids:
            continue
        conf = item.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            continue
        if not 0.0 <= conf <= 1.0:
            continue
        key = item.get("slot_key")
        if key is not None and not isinstance(key, str):
            continue
        # Casefold the key so "Anna"/"anna" supersede each other rather than
        # accumulating as two people.
        key = key.strip().lower() if key else None
        if slot == "relationship":
            if not key:
                continue
        else:
            key = None
        items.append({
            "slot": slot,
            "slot_key": key,
            "value": value.strip(),
            "evidence_message_id": mid,
            "confidence": float(conf),
        })
    return items


def _same_value(a: str, b: str) -> bool:
    """Conservative re-assertion check: casefold + collapse whitespace, nothing
    else — so any content difference counts as a new (possibly superseding)
    value, mirroring the _dedup_key philosophy in augment.py."""
    return " ".join(a.casefold().split()) == " ".join(b.casefold().split())


def persist_user_profile(
    conn: sqlite3.Connection,
    extraction: ProfileExtraction,
    *,
    redact_values: bool = True,
) -> int:
    """Persist validated items with bi-temporal supersession. Caller wraps in
    core_db.transaction().

    Per item: an already-active identical value reinforces confidence (no new
    row); a conflicting value on a single-valued slot — or on 'relationship'
    for the same slot_key person — closes the old rows (invalid_at = the new
    evidence's world date, falling back to now: the stamp_invalidation
    contract) and inserts the new one (valid_at = the evidence message's
    created_at, falling back to now: the stamp_validity contract). Unkeyed
    multi-valued slots accumulate without supersession. `redact_values` runs
    each value through hymem.redaction at this single chokepoint (pass
    cfg.redact_secrets) so every consumer inherits the scrubbing. Returns the
    number of NEW rows inserted."""
    inserted = 0
    for item in extraction.items:
        slot, key, mid = item["slot"], item.get("slot_key"), item["evidence_message_id"]
        conf = item["confidence"]
        value = redaction.redact(item["value"]) if redact_values else item["value"]

        evidence = conn.execute(
            "SELECT created_at FROM messages WHERE id = ?", (mid,)
        ).fetchone()
        world_date = evidence["created_at"] if evidence else None

        # `slot_key IS ?` is SQLite's NULL-safe equality, so unkeyed slots
        # compare on slot alone and keyed slots on the exact (slot, key) pair.
        active = conn.execute(
            "SELECT id, value FROM user_profile "
            "WHERE slot = ? AND slot_key IS ? AND invalid_at IS NULL",
            (slot, key),
        ).fetchall()

        same = [r for r in active if _same_value(r["value"], value)]
        if same:
            conn.execute(
                "UPDATE user_profile SET confidence = MAX(confidence, ?) WHERE id = ?",
                (conf, same[0]["id"]),
            )
            continue

        if slot in SINGLE_VALUED_SLOTS or slot == "relationship":
            ids = [r["id"] for r in active]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE user_profile "
                    f"SET invalid_at = COALESCE(?, CURRENT_TIMESTAMP) "
                    f"WHERE id IN ({placeholders}) AND invalid_at IS NULL",
                    (world_date, *ids),
                )

        conn.execute(
            "INSERT INTO user_profile("
            "    slot, slot_key, value, evidence_message_id, confidence, valid_at"
            ") VALUES (?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
            (slot, key, value, mid, conf, world_date),
        )
        inserted += 1

    if inserted:
        log.debug("profile.persisted rows=%d", inserted)
    return inserted


def load_profile(conn: sqlite3.Connection, cap: int | None = None) -> list[ProfileEntry]:
    """All ACTIVE profile rows, identity-first (PROFILE_SLOTS order), then by
    slot_key / confidence for a stable rendering. Returns [] on a pre-v18 DB
    (no table) so every consumer degrades cleanly. Read-only."""
    try:
        rows = conn.execute(
            "SELECT slot, slot_key, value, confidence, evidence_message_id, valid_at "
            "FROM user_profile WHERE invalid_at IS NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    entries = [
        ProfileEntry(
            slot=r["slot"],
            slot_key=r["slot_key"],
            value=r["value"],
            confidence=float(r["confidence"]),
            evidence_message_id=(
                int(r["evidence_message_id"])
                if r["evidence_message_id"] is not None
                else None
            ),
            valid_at=r["valid_at"] or "",
        )
        for r in rows
    ]
    entries.sort(
        key=lambda e: (
            _SLOT_ORDER.get(e.slot, len(PROFILE_SLOTS)),
            e.slot_key or "",
            -e.confidence,
            e.value,
        )
    )
    return entries[:cap] if cap is not None else entries


def render_profile_fact(entry: ProfileEntry) -> str:
    """One-line rendering for the VERIFIED FACTS anchor block, shaped like the
    graph-edge lines it sits above ("user role bedrijfsarts",
    "user relationship(anna) sister")."""
    if entry.slot_key:
        return f"user {entry.slot}({entry.slot_key}) {entry.value}"
    return f"user {entry.slot} {entry.value}"
