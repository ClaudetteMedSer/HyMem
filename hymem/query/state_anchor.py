"""State-anchor selection and seed terms (Plan D).

Existing query-side expansion (`query/augment.py`) grows the search from
entities found in the QUERY. A state anchor grows it from the ANSWER STATE: the
bitemporal active edge set plus the typed profile, i.e. what the store currently
holds to be true. Supporting-evidence rows that share no lexical or vector
overlap with the question — but do overlap with its answer — become reachable
that way, and superseded values become reachable through their active successor.

The selection is the digest anchor's predicate (`dreaming/aggregate.py:826-834`)
copied verbatim, with ONE deliberate deviation.

    `_anchor_facts` budgets profile rows and edges from a SHARED cap and returns
    early once profile fills it. That is correct there: it is sizing one prompt
    block, and profile rows genuinely outrank graph edges inside it. It is wrong
    for a seed source. Measured on the production box 2026-08-25 (Grove E2
    Stage 0): 22 active profile rows against `aggregation_digest_anchor_facts`
    = 20 leaves an edge budget of ZERO against 8754 active edges, and LoCoMo
    conv-26 leaves 4 slots for 55. Seeded that way the tier would be inert, the
    shadow probe would read C1 near zero, and Plan D would close FAIL-mechanism
    because of the digest's block budget rather than because state anchors do
    not work.

So the two sources are selected independently, each with its own cap, and
neither starves the other. Everything else — the filter, the ordering, the
`0 disables` convention, the missing-table degradation — matches the digest.

Read-only: this module issues SELECTs only and is safe on the `query_only`
connection `HyMem.augment()` uses.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from hymem.dreaming.user_profile import ProfileEntry, load_profile

if TYPE_CHECKING:  # the search primitives import this module's callers, so the
    # concrete hit types are pulled in lazily inside the functions below.
    from hymem.query.augment import FtsHit, MessageHit

# `dreaming/aggregate.py:826-834` verbatim. The ORDER BY is load-bearing, not
# cosmetic: the cap turns it into a selection, so a different order seeds a
# different tier.
_ANCHOR_EDGES = """
    SELECT subject_canonical AS s, predicate AS p, object_canonical AS o
    FROM knowledge_graph
    WHERE status = 'active' AND derived = 0 AND invalid_at IS NULL
      AND pos_evidence > neg_evidence
    ORDER BY pos_evidence - neg_evidence DESC, last_seen DESC, id
    LIMIT ?
"""


def select_anchor_edges(
    conn: sqlite3.Connection, *, edge_cap: int = 20
) -> list[sqlite3.Row]:
    """The currently-true graph edges, strongest evidence first.

    `edge_cap <= 0` disables the source (house convention). Returns [] rather
    than raising on a pre-migration store, so an old DB degrades instead of
    breaking `augment()`.
    """
    if edge_cap <= 0:
        return []
    try:
        return list(conn.execute(_ANCHOR_EDGES, (edge_cap,)).fetchall())
    except sqlite3.OperationalError:
        return []


def select_anchor_profile(
    conn: sqlite3.Connection, *, profile_cap: int = 20
) -> list[ProfileEntry]:
    """The active typed profile — a SEPARATE seed source from the edges.

    Kept separate so the shadow probe can attribute an anchored-only hit to
    edges or to profile rather than reporting one undifferentiated number, and
    so neither source can consume the other's budget.
    """
    if profile_cap <= 0:
        return []
    return load_profile(conn, cap=profile_cap)


def seed_terms_from_edges(
    edges: Iterable[sqlite3.Row],
    profile: Sequence[ProfileEntry] = (),
) -> list[str]:
    """Flatten the anchor sources into deduped lexical seed terms.

    Anchor sets are subject-heavy — the same subject recurs across predicates —
    so dedup is not tidiness: without it the FTS query degenerates into one term
    repeated N times, and BM25 scores the repetition rather than the state.
    Order is preserved (edges first, strongest evidence first) so the term list
    is deterministic for a given store.

    Typed values are deliberately NOT read here. They live in `kg_evidence`
    (`value_text` / `value_numeric` / `value_unit`), not on the edge row, so
    including them costs a JOIN per query against the C3 cost gate. Revisit only
    if the shadow probe shows the canonicals under-seeding.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def _add(value: object) -> None:
        text = str(value).strip() if value is not None else ""
        if text and text not in seen:
            seen.add(text)
            terms.append(text)

    for edge in edges:
        _add(edge["s"])
        _add(edge["p"])
        _add(edge["o"])
    for entry in profile:
        _add(entry.value)
        _add(entry.slot_key)
    return terms


# ── expansion over the two corpora ──────────────────────────────────────────
# Probed separately and shipped at most one (D5). They are NOT symmetric: chunks
# carry an FTS and a vector arm, raw messages carry only FTS because
# `_VEC_TABLES` (core/db.py:156) has no `vec_messages` — the BM25-only-ness that
# the LoCoMo close-out named as the architectural residual. Whichever pays is a
# measurement; seeding both behind one flag would let a helping path and a
# hurting path cancel to null (D5).


def _seed_query(seed_terms: Sequence[str]) -> str:
    """Join the anchor set into ONE query string.

    One string means one embedding call for the whole anchor set however many
    edges it holds (C3: <=1 vector call per query). The FTS helpers do their own
    `_FTS_SAFE` cleaning and OR-joining, so no tokenisation happens here.
    """
    return " ".join(seed_terms)


def expand_over_chunks(
    conn: sqlite3.Connection,
    seed_terms: Sequence[str],
    *,
    top_k: int = 5,
    embedding_client: object | None = None,
    max_scan: int = 2000,
) -> list["FtsHit"]:
    """Anchor expansion over the chunk corpus: FTS, plus vector when available.

    Returns `FtsHit`, the type the chunk tier already uses, so nothing
    downstream needs a new dataclass or a new renderer branch.
    """
    if not seed_terms or top_k <= 0:
        return []
    from hymem.query.augment import _fts_search, _rrf_merge, _vector_search

    query = _seed_query(seed_terms)
    fts = _fts_search(conn, query, top_k=top_k)
    if embedding_client is None:
        return fts[:top_k]
    vec = _vector_search(conn, embedding_client, query,
                         top_k=top_k, max_scan=max_scan)
    if not vec:
        return fts[:top_k]
    if not fts:
        return vec[:top_k]
    return _rrf_merge(fts, vec, top_k=top_k)


def expand_over_messages(
    conn: sqlite3.Connection,
    seed_terms: Sequence[str],
    *,
    top_k: int = 5,
) -> list["MessageHit"]:
    """Anchor expansion over the raw-turn corpus. FTS only — there is no vector
    path over messages, so this arm inherits BM25's vocabulary brittleness and
    the probe must report it separately rather than pooling the two corpora."""
    if not seed_terms or top_k <= 0:
        return []
    from hymem.query.augment import _message_fts_search

    return _message_fts_search(conn, _seed_query(seed_terms), top_k=top_k)[:top_k]
