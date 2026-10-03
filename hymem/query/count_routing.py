"""Route an MR ("how many X …") query to an EXACT graph-native count.

The MR aggregate path (`augment._message_fts_aggregate`) gives a *candidate*
count over raw user turns — it over/under-counts because it never types or
dedups by entity. For the IN-DOMAIN slice of MR questions ("how many services
depend on redis?", "how many databases do we use?") the knowledge graph can
instead give an EXACT, dedup-correct count via `entities.count_relations`. This
module is the bridge: it inspects a query and, when (and only when) it maps
cleanly onto the in-vocab type/predicate/entity machinery, emits the arguments
for `count_relations`. When the mapping is ambiguous or out-of-vocab it returns
None, so the keyword candidate stands alone (the fallback for consumer-domain /
un-typed questions the graph simply can't answer).

Everything here is pure routing — no DB, no LLM. It reuses, never re-implements,
the same detectors `augment()` already relies on: `detect_query_types` (the
`_TYPE_QUERY_KEYWORDS` phrase map → `entity_types` label), `route_predicates`
(query phrasing → predicate vocabulary), and the caller-supplied entity matches
from `match_known_entities`. That shared provenance is the whole safety story:
the graph count can only fire on vocabulary the rest of the graph already speaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hymem.query.augment import detect_query_types
from hymem.query.predicate_routing import route_predicates


@dataclass
class CountPlan:
    """A resolved argument bundle for `entities.count_relations`.

    Mirrors the `count_relations` keyword surface 1:1 so the caller can splat it.
    `count` is the load-bearing side decision (subject vs object) — see
    `plan_count` for the heuristic. `why` records which signals (type/predicate/
    anchor) justified the plan, mirroring `GraphFact.why_retrieved`, so a
    misfiring exact count is debuggable from the result alone."""

    count: str  # "subject" | "object"
    predicates: list[str] | None = None
    subject: str | None = None
    object: str | None = None
    subject_type: str | None = None
    object_type: str | None = None
    why: list[str] = field(default_factory=list)


def plan_count(
    user_message: str,
    matched_entities: list[str],
) -> CountPlan | None:
    """Decide whether an MR query maps onto an in-vocab graph count, and how.

    Returns a `CountPlan` (ready to splat into `count_relations`) when the query
    clearly names an in-vocab **type**, optionally an in-vocab **predicate**, and
    optionally an **anchor entity**; returns None otherwise so the keyword
    aggregate path stands alone. The bar for emitting a plan is deliberately high
    — a graph count is presented to the host as EXACT, so a false-positive
    mapping is worse than no mapping at all.

    Gate (all required to emit a plan):
      - The query must name exactly ONE in-vocab type label (`detect_query_types`,
        the shared `_TYPE_QUERY_KEYWORDS` map). Zero types ⇒ nothing typed to
        count (let the keyword path answer). More than one type ⇒ ambiguous which
        column to constrain, so we abstain rather than guess.

    Side heuristic (which column to de-dup and tally — the one decision
    `count_relations` refuses to infer):
      - ANCHORED shape — the query names a type T AND an anchor entity E
        ("how many <T> depend on <redis>"): the unknown being counted is the set
        of distinct **subjects** of type T related to the known object E, so
        count="subject", subject_type=T, object=E. This matches the canonical
        "how many subjects relate to this object" framing `count_relations`
        documents as its default side.
      - UNANCHORED shape — the query names a type T with NO anchor entity
        ("how many <T> do we use"): there is no object to pivot on, so we count
        the distinct **objects** of type T that appear under the routed predicate,
        i.e. count="object", object_type=T. This is the "inventory of typed
        things we relate to" framing.

    Predicate: attached when `route_predicates` fires, narrowing the count to the
    intended relation; omitted (None) when no predicate routes, in which case the
    count spans all predicates for that type (still exact, just broader). We do
    NOT require a predicate — "how many databases do we have" carries a clear type
    but no predicate keyword, and a typed all-predicate count is still correct.

    Anchor selection: the first matched entity is used as the anchor for the
    anchored shape. `matched_entities` comes from the caller's
    `match_known_entities` (+ its type/overlap expansions). To avoid letting a
    TYPE-expansion canonical masquerade as a user-named anchor (it would silently
    pivot the count on an entity the user never mentioned), the caller passes the
    DIRECT entity matches only — see `augment()`'s call site.
    """
    types = detect_query_types(user_message)
    if len(types) != 1:
        # Zero ⇒ nothing typed to count; >1 ⇒ ambiguous target column. Either way
        # the keyword path is the safer answer than a guessed exact count.
        return None
    target_type = next(iter(types))

    routed = route_predicates(user_message)
    predicates = sorted(routed) if routed else None

    anchor = matched_entities[0] if matched_entities else None

    why: list[str] = [f"type:{target_type}"]
    if predicates:
        why.append("predicates:" + ",".join(predicates))

    if anchor is not None:
        # Anchored: count distinct subjects OF the named type related to the
        # anchor object. The user named the object (e.g. "redis"); the unknown is
        # the typed subjects depending on it.
        why.append(f"anchor:{anchor}")
        why.append("side:subject(anchored)")
        return CountPlan(
            count="subject",
            predicates=predicates,
            subject_type=target_type,
            object=anchor,
            why=why,
        )

    # Unanchored: no entity to pivot on, so count distinct typed objects (the
    # inventory of typed things we relate to).
    why.append("side:object(unanchored)")
    return CountPlan(
        count="object",
        predicates=predicates,
        object_type=target_type,
        why=why,
    )
