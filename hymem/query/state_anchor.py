"""State-anchor expansion — Plan D (borrowed from MindCache's decision-anchor
retrieval, adapted 2026-08-14, implemented 2026-08-25).

A retrieval-time expansion tier: seed a secondary lexical/vector expansion from
the CURRENTLY ACTIVE graph facts — the same selection the digest anchor uses
(`dreaming/aggregate.py:_anchor_facts`) — so that supporting-evidence rows
which share no lexical or vector overlap with the query, but overlap with the
*current state*, become reachable.

Three pieces, mirroring the plan's tasks:

1. ``select_anchor_edges`` — the exact ``_anchor_facts`` predicate
   (``status='active' AND derived=0 AND invalid_at IS NULL AND
   pos_evidence > neg_evidence``), ordered by evidence margin, bounded by
   ``cap``. Copied verbatim from ``aggregate.py``; the digest function is NOT
   refactored here (YAGNI — cross-module churn is a separate change if the
   probe justifies it).
2. ``seed_terms_from_edges`` — canonical subject + predicate + object, plus
   typed-value sub-terms mirroring ``value_supersession.py``'s v3 classes:
   a version's alpha prefix is the discriminative side (``python_3.12`` ->
   ``python``), a number's unit is (``65_percent`` -> ``percent``), a date's
   year is (``2024-03-01`` -> ``2024``). The value itself stays reachable via
   the canonical object term.
3. ``state_anchor_expand`` — run the seed terms through the existing
   ``_fts_search``/``_rrf_merge`` machinery (optional single ``_vec_search``
   call, C3: <=1 vector call per query), dedup by chunk id (the entry key),
   bounded by ``top_k``. Inert with no seed terms.

Pure query-side and read-only: no store writes, no prompt changes, no schema
change. The additive merge into ``augment()`` is Task 6 and only lands behind
the pre-registered gate (C1-C4) of the plan.

ONE DELIBERATE DEVIATION FROM ``_anchor_facts`` (banked in
``additional_planning.md`` Plan D correction 5 on 2026-08-25, before the shadow
probe ran). The digest budgets profile rows and edges from a SHARED cap and
returns early once profile fills it. That is correct there: it is sizing ONE
PROMPT BLOCK, where profile rows genuinely outrank edges. It is wrong for a
SEED SOURCE. Measured on the production box: 22 active profile rows against
``aggregation_digest_anchor_facts`` = 20 leaves an edge budget of ZERO against
8754 active edges; LoCoMo conv-26 leaves 4 slots for 55. Seeded that way the
tier is inert, C1 reads near zero, and Plan D closes FAIL-mechanism because of
the digest's block budget rather than because state anchors do not work.

``select_state_anchor`` therefore defaults to INDEPENDENT caps. The digest's
shared-cap behaviour stays reachable via ``shared_cap=`` so the leg already
measured on the box remains reproducible and labelled, rather than silently
disappearing under a changed default.
"""
from __future__ import annotations

import sqlite3
import re
from collections.abc import Iterable, Mapping
from typing import Any

from hymem.dreaming.value_supersession import _ISO_DATE, _classify_object

# The EXACT anchor predicate — copy of dreaming/aggregate.py:825-835. If the
# digest anchor ever changes, this module must change with it (the probe's
# whole point is measuring the CURRENT state anchor).
_ANCHOR_SQL = """
    SELECT subject_canonical, predicate, object_canonical
    FROM knowledge_graph
    WHERE status = 'active' AND derived = 0 AND invalid_at IS NULL
      AND pos_evidence > neg_evidence
    ORDER BY pos_evidence - neg_evidence DESC, last_seen DESC, id
    LIMIT ?
"""


def select_anchor_edges(conn: sqlite3.Connection, cap: int = 20) -> list[sqlite3.Row]:
    """The bitemporal active edge set, exactly as ``_anchor_facts`` selects it.

    ``cap`` bounds the list (house style like ``aggregation_digest_anchor_facts``,
    default 20); cap <= 0 returns [] (mirrors the digest's early return).
    Message: NO profile-row interplay here — the digest lets profile rows
    consume the cap first; this selection is the GRAPH side of the anchor and
    is what Task 6's expansion will seed from. If ``aggregate.py`` ever
    changes the profile-first ordering, this function must state its caveat.
    """
    if cap <= 0:
        return []
    return conn.execute(_ANCHOR_SQL, (cap,)).fetchall()


def select_anchor_profile_rows(conn: sqlite3.Connection, cap: int = 20) -> list:
    """The ACTIVE profile rows, exactly as the digest anchor's profile leg
    selects them (``dreaming/user_profile.py:load_profile`` — identity-first
    order, capped). ``cap <= 0`` -> [] (mirrors the digest's early return).
    """
    from hymem.dreaming.user_profile import load_profile

    if cap <= 0:
        return []
    return load_profile(conn, cap)


def seed_terms_from_profile(entries: Iterable[Any]) -> list[str]:
    """Seed terms from profile rows (the digest anchor's profile leg).

    A profile row renders as ``user {slot}({slot_key}) {value}``; the
    searchable vocabulary is the SUBJECT-matter content: the slot's words
    (``recurring_activity`` -> ``recurring``, ``activity``), the slot_key, the
    value, and the value's words (``running pottery`` -> ``running``,
    ``pottery``). The generic ``user`` prefix carries no lexical recall and is
    skipped. Dedup in first-seen order.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def add(t: str | None) -> None:
        if not t:
            return
        tt = t.strip().lower()
        if tt and tt not in seen:
            seen.add(tt)
            terms.append(tt)

    def add_with_words(t: str | None) -> None:
        if not t:
            return
        add(t)
        for w in re.findall(r"[a-z0-9]+", t.strip().lower()):
            if len(w) >= 2:
                add(w)

    for e in entries:
        add_with_words(getattr(e, "slot", None))
        add_with_words(getattr(e, "slot_key", None))
        add_with_words(getattr(e, "value", None))
    return terms


def select_state_anchor(
    conn: sqlite3.Connection,
    *,
    edge_cap: int = 20,
    profile_cap: int = 20,
    shared_cap: int | None = None,
) -> tuple[list, list]:
    """Both anchor legs. Independent caps by default; ``shared_cap`` reproduces
    the digest.

    Default (``shared_cap=None``): the two sources are selected independently,
    each with its own cap, and neither starves the other. This is the banked
    Plan D deviation — see the module docstring for why a shared cap is a
    gate-validity problem rather than a preference.

    ``shared_cap=N``: profile rows consume N first and edges fill the remainder,
    exactly as ``aggregate.py:_anchor_facts`` (801-835) budgets its prompt
    block. Kept so the shared-cap leg measured on the box 2026-08-25 stays
    reproducible; it is NOT the default and must be named explicitly.
    """
    if shared_cap is not None:
        profiles = select_anchor_profile_rows(conn, cap=shared_cap)
        edges = select_anchor_edges(conn, cap=max(0, shared_cap - len(profiles)))
        return profiles, edges
    return (
        select_anchor_profile_rows(conn, cap=profile_cap),
        select_anchor_edges(conn, cap=edge_cap),
    )


def _field(e: Mapping[str, Any] | Any, name: str) -> Any:
    """Field access that works for sqlite3.Row and plain dicts, and yields
    None for a missing/blank field (so an empty edge produces no terms)."""
    try:
        return e[name]
    except (KeyError, IndexError, TypeError):
        return None


def seed_terms_from_edges(edges: Iterable[Mapping[str, Any]]) -> list[str]:
    """Convert anchor edges into lexical seed terms.

    Every non-empty canonical field is a term; typed objects additionally
    contribute their discriminative sub-term per value_supersession v3:
    versions the alpha prefix, numbers the unit, dates the year. Terms are
    deduplicated in first-seen order. Edge with no usable field -> no terms.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str | None) -> None:
        if not term:
            return
        t = term.strip()
        if t and t not in seen:
            seen.add(t)
            terms.append(t)

    for e in edges:
        subj = _field(e, "subject_canonical")
        pred = _field(e, "predicate")
        obj = _field(e, "object_canonical")
        add(subj)
        add(pred)
        add(obj)
        cls = _classify_object(obj)
        if cls:
            kind, key = cls
            if kind == "ver" and key:
                add(key)  # alpha prefix: 'python' from 'python_3.12'
            elif kind == "num" and key:
                add(key)  # unit: 'percent' from '65_percent'
            elif kind == "date":
                m = _ISO_DATE.search((obj or "").strip())
                if m:
                    add(m.group(0)[:4])  # year: '2024' from '2024-03-01'
    return terms


def state_anchor_expand(
    conn: sqlite3.Connection,
    seed_terms: Iterable[str],
    *,
    top_k: int = 5,
    embedding_client: Any = None,
) -> list[dict]:
    """Run seed terms through the existing search machinery.

    FTS always; the optional vector leg fires at most once per call (C3) and
    only when ``embedding_client`` is provided. Results are RRF-merged and
    deduplicated by chunk id (entry key), capped at ``top_k``. Returns dicts
    {chunk_id, session_id, text, score, score_kind, why_retrieved} — the
    ``FtsHit`` shape (plain dict instead of the dataclass so the probe can
    JSON-serialize it). Empty/inert seed terms -> [] (zero cost).
    """
    from hymem.query.augment import _fts_search, _rrf_merge, _vec_search

    terms = [t for t in (t.strip() if t else "" for t in seed_terms) if len(t) >= 2]
    if not terms:
        return []

    # One combined FTS OR-query keeps the call count small: a query over all
    # seed terms at top_k. `_fts_search` fragments on _FTS_SAFE (dots split,
    # quotes stripped), so separator-heavy canonical terms (cuda_12.1,
    # 65_percent) are matched through their v3-class sub-terms — the version
    # prefix / unit / year the seed generator emits — while plain canonical
    # terms match directly.
    fts = _fts_search(conn, " ".join(terms), top_k=top_k)

    vec: list[Any] = []
    if embedding_client is not None:
        # Single vector call over the joined seed text (the C3 allowance).
        seed_text = " ".join(terms)
        vec = _vec_search(conn, embedding_client, seed_text, top_k=top_k)

    merged = _rrf_merge(fts, vec, top_k=top_k) if vec else fts
    return [
        {
            "chunk_id": h.chunk_id,
            "session_id": h.session_id,
            "text": h.text,
            "score": getattr(h, "score", 0.0),
            "score_kind": getattr(h, "score_kind", "rrf"),
            "why_retrieved": list(getattr(h, "why_retrieved", [])),
        }
        for h in merged
    ]


def expand_over_messages(
    conn: sqlite3.Connection,
    seed_terms: Iterable[str],
    *,
    top_k: int = 5,
) -> list[dict]:
    """Anchor expansion over the RAW-TURN corpus, in the shape of
    ``state_anchor_expand``.

    A second corpus, not an alternative implementation. D3/D5 require both to
    be probed SEPARATELY and at most one to ship: seeding both behind one flag
    lets a helping path and a hurting path cancel to null, and the plan closes
    for the wrong reason.

    FTS only — ``_VEC_TABLES`` (``core/db.py:156``) has no ``vec_messages``, so
    this arm inherits BM25's vocabulary brittleness. That is exactly the
    architectural residual the LoCoMo close-out named, which is why its number
    must never be pooled with the chunk arm's.
    """
    from hymem.query.augment import _message_fts_search

    terms = [t for t in (t.strip() if t else "" for t in seed_terms) if len(t) >= 2]
    if not terms:
        return []
    hits = _message_fts_search(conn, " ".join(terms), top_k=top_k)[:top_k]
    return [
        {
            "message_id": h.message_id,
            "session_id": h.session_id,
            "role": h.role,
            "text": h.text,
            "score": h.score,
            "created_at": h.created_at,
            "score_kind": h.score_kind,
            "why_retrieved": list(h.why_retrieved),
        }
        for h in hits
    ]
