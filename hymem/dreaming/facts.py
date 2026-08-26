"""Narrative-facts extraction and persistence (E1, schema v26).

The middle granularity between the atomic knowledge-graph triple and the
abstract episode summary: one self-contained sentence per thing that happened,
was decided, or was true — readable without the conversation around it.
Extracted at dream time (one LLM call per session tail), stored APPEND-ONLY in
``narrative_facts``, served as an additive retrieval tier (``ctx.facts``) and
as the lead evidence block in ``ask()``.

Built behind the G-F1b extraction-faithfulness gate (PASSED 2026-08-02,
123/123 strict on the healed full-source sample; instrument =
``benchmarks/fact_probe.py``). Three gate findings are load-bearing here:

  * **The input shape is the gated shape.** The gate scored extraction over the
    session's raw user/assistant turns rendered as ``role: content`` lines,
    char-capped — so that is what :func:`extract_facts` feeds the model. It
    deliberately does NOT read the dreamed ``chunks`` the digest reads: chunks
    carry only (assistant, user) pairs above a salience floor, duplicate the
    assistant prefix across chunks, and drop short turns — a different corpus
    from the one the faithfulness verdict was measured on.
  * **The prompt is the gated prompt, verbatim** (``FACTS_SYSTEM`` /
    ``FACTS_USER_TEMPLATE`` in the prompts module, tagged ``facts.v2``). Any
    rewording re-enters the gate before it ships.
  * **An undated fact stays undated.** ``fact_date`` holds explicit
    YYYY-MM-DD dates the conversation wrote; relative references stay NULL
    (resolving them is E4's job). Stamping the session date was a proven
    invention amplifier — both the v1 prompt and the probe's own validator had
    to lose it.

Coverage follows the v24 watermark pattern with its OWN column
(``sessions.facts_message_id``), so facts and digest advance independently:
extraction reads only messages ABOVE the watermark, truncates on whole-message
boundaries so the watermark names a real message id, advances only on a
successful parse (a failure holds it and the slice retries next dream), and a
prompt bump extracts FORWARD ONLY — covered ranges are never re-extracted, so
every stored fact stays attributable to the prompt that produced it.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field

from hymem.config import HyMemConfig
from hymem.dreaming.canonicalize import normalize
from hymem.extraction.jsonio import loads_lenient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import FACTS_SYSTEM, FACTS_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.facts")

# Pinned prompt version, following the PROFILE_PROMPT_VERSION convention.
# facts.v2 is the arm that cleared G-F1b; facts.v1 (the draft that failed
# G-F1) never shipped. Bumping this does NOT re-extract covered ranges —
# forward-only — but any material change to FACTS_SYSTEM re-enters the
# fact_probe gate before the bump lands.
FACTS_PROMPT_VERSION = "facts.v2"

# Validation bounds. Kept in lockstep with benchmarks/fact_probe.py's
# validator (_MAX_FACT_CHARS there) — the gate must reject exactly what the
# build rejects, or its density criterion reads a different pipeline.
_MAX_FACT_CHARS = 600
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class FactsExtraction:
    """Validated narrative facts from one extraction call over a session tail.

    ``start_message_id``/``covered_message_id`` bound the raw messages the LLM
    actually saw; every fact of this call persists under that range, and the
    runner advances ``sessions.facts_message_id`` to ``covered_message_id``.
    A parse failure sets ``parse_failed`` and claims no coverage, so the slice
    is retried next dream (the SessionDigest contract)."""

    items: list[dict] = field(default_factory=list)
    start_message_id: int | None = None
    covered_message_id: int | None = None
    parse_failed: bool = False


def extract_facts(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
    cfg: HyMemConfig,
    *,
    since_message_id: int | None = None,
) -> FactsExtraction | None:
    """One LLM call extracting narrative facts from the session's uncovered
    tail: user/assistant messages strictly above the facts watermark, rendered
    as ``role: content`` lines (the exact shape the G-F1b gate scored).

    Truncates on whole-message boundaries so ``covered_message_id`` names a
    message the model fully saw — a half-included turn would either be re-read
    forever or silently skipped, the starvation class migration 024 fixed for
    the digest. Returns None when there is nothing to read (fully covered tail
    or no content). No write transaction held; persist via
    :func:`persist_facts` inside one.
    """
    if since_message_id is None:
        where, params = "", (session_id,)
    else:
        where = " AND id > ?"
        params = (session_id, since_message_id)
    rows = conn.execute(
        "SELECT id, content, role FROM messages "
        f"WHERE session_id = ? AND role IN ('user', 'assistant'){where} "
        "ORDER BY id",
        params,
    ).fetchall()
    rows = [r for r in rows if (r["content"] or "").strip()]
    if not rows:
        return None

    max_chars = cfg.dream_digest_max_chars
    lines: list[str] = []
    start: int | None = None
    covered: int | None = None
    used = 0
    for r in rows:
        line = f"{r['role']}: {r['content'].strip()}"
        cost = len(line) + (1 if lines else 0)  # the "\n" join
        if lines and used + cost > max_chars:
            break
        lines.append(line)
        used += cost
        if start is None:
            start = int(r["id"])
        covered = int(r["id"])
    combined = "\n".join(lines)
    if len(combined) > max_chars:
        # ONE message longer than the whole cap (the loop always admits the
        # first line, whatever its size). Truncate it and still claim
        # coverage: truncation keeps the HEAD, so no later dream could read
        # more of this turn than this one just did — holding the watermark
        # would re-send it, and re-spend the call, on every dream forever
        # while storing nothing. Losing the tail of one oversized turn is the
        # lesser evil, and it is logged rather than silent.
        combined = combined[:max_chars]
        log.warning(
            "facts.oversized_turn session_id=%s message_id=%s chars=%d cap=%d "
            "(tail not read)",
            session_id, covered, len(lines[0]), max_chars,
        )

    request = LLMRequest(
        system=FACTS_SYSTEM,
        user=FACTS_USER_TEMPLATE.format(text=combined),
        response_format="json",
        max_tokens=cfg.dream_digest_max_tokens,
    )
    raw = llm.complete(request)

    items = validate_fact_items(raw, max_items=cfg.dream_max_facts_per_session)
    if items is None:
        # Unparseable reply: no coverage claimed, slice retried next dream.
        return FactsExtraction(parse_failed=True)
    return FactsExtraction(
        items=items, start_message_id=start, covered_message_id=covered
    )


def validate_fact_items(raw: object, *, max_items: int) -> list[dict] | None:
    """Coerce a model reply into validated fact dicts.

    Returns None for an UNPARSEABLE payload (no JSON array recoverable) —
    the caller must treat that as a failure that holds the watermark — and a
    list (possibly empty: a valid "nothing here") otherwise.

    Mirrors the fact_probe validator the gate ran: non-empty text capped at
    ``_MAX_FACT_CHARS``; a malformed date is dropped while the fact is kept
    (the date is metadata, the text is the evidence); NO session-date fallback
    ever (an undated fact stays undated — the G-F1 date lesson); at most
    ``max_items`` facts survive, truncated, so a runaway reply is bounded
    before any row is written. Entities additionally pass through
    ``canonicalize.normalize`` here — the store speaks canonical ids — which
    the probe (scoring verbatim faithfulness) deliberately did not do.
    """
    if isinstance(raw, str):
        # Tolerate a fenced or prose-wrapped array — the same leniency the
        # dreaming parsers apply, no more: one shared helper, so that stays true.
        raw = loads_lenient(raw, expect="array")
    if not isinstance(raw, list):
        return None

    out: list[dict] = []
    for item in raw:
        if len(out) >= max_items:
            break
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        date = item.get("date")
        date = str(date).strip() if date else ""
        if date and not _ISO_DATE.match(date):
            date = ""
        ents = item.get("entities")
        entities: list[str] = []
        if isinstance(ents, list):
            for e in ents:
                canonical = normalize(str(e)) if str(e).strip() else ""
                if canonical and canonical not in entities:
                    entities.append(canonical)
        out.append({
            "text": text[:_MAX_FACT_CHARS],
            "date": date or None,
            "entities": entities,
        })
    return out


def persist_facts(
    conn: sqlite3.Connection,
    session_id: str,
    extraction: FactsExtraction,
) -> int:
    """Append-only insert of validated facts. Caller wraps in
    core_db.transaction(). Returns the number of NEW rows.

    ``INSERT OR IGNORE`` on the UNIQUE (session_id, start_message_id, text)
    key is the whole idempotency story: a re-submitted range (retry after a
    partial failure, an operator replay) no-ops row by row, and there is no
    UPDATE path for text/entities — immutability is what lets a prompt bump
    leave covered ranges untouched.

    ``valid_at`` mirrors the knowledge-graph stamp_validity contract: the
    explicit ``fact_date`` when the conversation wrote one, else the world
    date the fact was asserted (the covered range's last message
    ``created_at``). That is speech time, not event time — the E4 finding —
    but it is the same proxy every other bi-temporal row in the store uses,
    and it is never surfaced as a content date (``fact_date`` is what
    renders)."""
    if not extraction.items:
        return 0
    if extraction.start_message_id is None or extraction.covered_message_id is None:
        # Defensive: every extraction that produced items also names its range
        # (the reader always admits at least one message). A range-less
        # extraction would key rows on nothing, so drop them rather than
        # write evidence that cannot be traced back to its turns.
        return 0

    asserted = conn.execute(
        "SELECT created_at FROM messages WHERE id = ?",
        (extraction.covered_message_id,),
    ).fetchone()
    asserted_at = asserted["created_at"] if asserted else None

    inserted = 0
    for item in extraction.items:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO narrative_facts(
                session_id, start_message_id, end_message_id,
                text, fact_date, entities, prompt_version, valid_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, ?, CURRENT_TIMESTAMP))
            """,
            (
                session_id,
                extraction.start_message_id,
                extraction.covered_message_id,
                item["text"],
                item["date"],
                json.dumps(item["entities"]),
                FACTS_PROMPT_VERSION,
                item["date"],
                asserted_at,
            ),
        )
        inserted += cur.rowcount if cur.rowcount > 0 else 0

    if inserted:
        log.debug("facts.persisted session_id=%s rows=%d", session_id, inserted)
    return inserted
