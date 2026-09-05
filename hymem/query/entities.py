from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, Literal

from hymem.dreaming.canonicalize import normalize, resolve
from hymem.core.graph import live_edge_predicate

# Cap on the evidence entities returned by count_relations. The `count` stays
# exact (it is COUNT(DISTINCT ...) in SQL); this only bounds the materialized
# list of distinct entities the caller gets back for verification, so a query
# matching thousands of subjects doesn't return thousands of strings.
_COUNT_EVIDENCE_CAP = 100

# First char must be a Unicode letter (so accented Latin words like "préfère"
# tokenize whole instead of being shredded at the accent); body allows letters,
# digits, underscore, hyphen, dot. normalize() then folds it consistently with
# how the entity was stored.
_TOKEN = re.compile(r"[^\W\d_][\w\-.]{1,40}")


def match_known_entities(conn: sqlite3.Connection, message: str) -> list[str]:
    """Return canonical ids that the user message references.

    Strategy: tokenize the message, normalize each token, and look it up against
    the alias table and the graph's existing canonical names. Cheap, deterministic,
    and the graph is its own dictionary — no LLM call needed at query time.
    """
    raw_tokens = {m.group(0) for m in _TOKEN.finditer(message)}
    candidates = {normalize(t) for t in raw_tokens if len(t) >= 2}

    # Also try multi-word phrases (up to 3-grams) to catch "local dev environment".
    words = [w for w in re.split(r"\s+", message.strip()) if w]
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            candidates.add(normalize(phrase))

    if not candidates:
        return []

    candidates_list = list(candidates)
    placeholders = ",".join("?" * len(candidates_list))
    # The object-canonical branch applies a shape filter: an object only counts
    # as a known entity if it looks entity-shaped — appears as a subject
    # somewhere, has an entity_types record, or shows up as an object in more
    # than one edge. This suppresses one-off LLM extractions where a gerund or
    # verb form ("working") landed as an object and would otherwise match any
    # query containing that word. The alias and subject branches pass through
    # unfiltered: an explicit alias registration or subject-position usage are
    # already strong entity-shape signals.
    live_alias = live_edge_predicate("alias_edge")
    live_subject = live_edge_predicate("subject_edge")
    live_object = live_edge_predicate("kg")
    live_shape_subject = live_edge_predicate("shape_subject")
    live_shape_object = live_edge_predicate("kg2")
    rows = conn.execute(
        f"""
        SELECT DISTINCT aliases.canonical
        FROM entity_aliases aliases
        WHERE aliases.alias IN ({placeholders})
          AND EXISTS (
              SELECT 1 FROM knowledge_graph alias_edge
              WHERE {live_alias}
                AND (alias_edge.subject_canonical = aliases.canonical
                     OR alias_edge.object_canonical = aliases.canonical)
          )
        UNION
        SELECT DISTINCT subject_edge.subject_canonical
        FROM knowledge_graph subject_edge
        WHERE subject_edge.subject_canonical IN ({placeholders})
          AND {live_subject}
        UNION
        SELECT DISTINCT object_canonical FROM knowledge_graph kg
        WHERE object_canonical IN ({placeholders})
          AND {live_object}
          AND (
            EXISTS (
              SELECT 1 FROM knowledge_graph shape_subject
              WHERE shape_subject.subject_canonical = kg.object_canonical
                AND {live_shape_subject}
            )
            OR EXISTS (
              SELECT 1 FROM entity_types
              WHERE entity_canonical = kg.object_canonical
            )
            OR EXISTS (
              SELECT 1 FROM knowledge_graph kg2
              WHERE kg2.object_canonical = kg.object_canonical AND kg2.id != kg.id
                AND {live_shape_object}
            )
          )
        """,
        candidates_list + candidates_list + candidates_list,
    ).fetchall()
    return [r[0] for r in rows]


@dataclass
class TimelineEntry:
    """The earliest currently-valid direct edge per predicate.

    ``first_seen`` remains the honest ingestion clock for compatibility;
    ``valid_at`` is the source/world clock used for selection and ordering.
    """
    predicate: str
    subject: str
    object: str
    first_seen: str
    status: str
    valid_at: str | None = None
    edge_id: int | None = None
    derived: bool = False


def timeline(
    conn: sqlite3.Connection,
    entity: str,
) -> list[TimelineEntry]:
    """Earliest source-valid edge per predicate for ``entity``, oldest first.

    Inferred closure is always excluded because it has no persisted derivation
    lineage or truthful source-valid interval.

    `entity` is resolved through the alias table, so a surface form
    ("Postgres") maps to its canonical id ("postgres").
    """
    canon = resolve(conn, entity)
    current = live_edge_predicate()
    rows = conn.execute(
        f"""
        SELECT id, predicate, subject_canonical, object_canonical,
               first_seen, normalized_valid_at AS valid_at, status, derived
        FROM (
            SELECT id, predicate, subject_canonical, object_canonical,
                   first_seen,
                   hymem_normalize_iso_timestamp(valid_at) AS normalized_valid_at,
                   status, derived,
                   ROW_NUMBER() OVER (
                       PARTITION BY predicate
                       ORDER BY hymem_normalize_iso_timestamp(valid_at) ASC, id ASC
                   ) AS rn
            FROM knowledge_graph
            WHERE {current}
              AND valid_at IS NOT NULL
              AND (subject_canonical = ? OR object_canonical = ?)
        )
        WHERE rn = 1
        ORDER BY normalized_valid_at ASC, predicate ASC, id ASC
        """,
        (canon, canon),
    ).fetchall()
    return [
        TimelineEntry(
            predicate=r["predicate"],
            subject=r["subject_canonical"],
            object=r["object_canonical"],
            first_seen=r["first_seen"],
            status=r["status"],
            valid_at=r["valid_at"],
            edge_id=int(r["id"]),
            derived=bool(r["derived"]),
        )
        for r in rows
    ]


@dataclass
class GraphCount:
    """Result of a graph-native count over `knowledge_graph` edges.

    `count` is the EXACT number of distinct entities (subjects or objects, per
    `counted`) that satisfy the filters — it is a `COUNT(DISTINCT ...)` in SQL
    and never truncated. `entities` is the same distinct set materialized as
    evidence so a caller can verify the number, but it is capped at
    `_COUNT_EVIDENCE_CAP` (so `count >= len(entities)` always; equal until the
    cap bites). `counted` records WHICH side was counted — the single most
    important fact about a count, since "how many services depend on redis"
    (distinct subjects) and "how many databases do we use" (distinct objects)
    look alike to a caller but count opposite columns. `filters` echoes the
    resolved/applied filters (surface forms already mapped to canonicals) so the
    contract behind the number is auditable, mirroring `GraphFact.why_retrieved`.
    """

    count: int
    counted: Literal["subject", "object"]
    entities: list[str] = field(default_factory=list)
    filters: dict[str, object] = field(default_factory=dict)


def count_relations(
    conn: sqlite3.Connection,
    *,
    count: Literal["subject", "object"] = "subject",
    predicates: Iterable[str] | None = None,
    subject: str | None = None,
    object: str | None = None,
    object_type: str | None = None,
    subject_type: str | None = None,
    include_derived: bool = False,
) -> GraphCount:
    """Count DISTINCT subjects or objects of `knowledge_graph` edges matching the
    given filters — the graph-native primitive behind in-domain "how many X …"
    questions ("how many services depend on redis?", "how many databases do we
    use?").

    Counting contract (the load-bearing decision): `count` names which column is
    de-duplicated and tallied, and it defaults to `"subject"`. The default is
    deliberate: the canonical in-domain question is "how many <subjects> relate
    to <object> via <predicate>" ("how many services depend_on redis"), where the
    anchor is a known *object* (redis) and the unknown being counted is the set
    of distinct *subjects*. To count the other side — "how many <objects> of type
    T do we <predicate>" ("how many databases do we use", anchored on the subject
    "we"/an app, counting distinct typed objects) — pass `count="object"`. The
    function NEVER guesses the side from the filters; the caller states it, so the
    number is unambiguous.

    Filter semantics (all ANDed; an omitted filter is not constrained):
      - the full live-current predicate is ALWAYS enforced: active, open valid
        interval, and positive evidence strictly outweighing negative evidence.
        Direct observations are counted unless ``include_derived=True``.
      - `subject`/`object` are surface forms resolved through the alias table via
        `resolve()` (the same helper `timeline()` uses), so "Postgres" → "postgres"
        before filtering. Filtering an anchor to its canonical is what makes the
        count stable across surface variants.
      - `subject_type`/`object_type` filter the respective column to canonicals
        carrying that label in `entity_types` (an EXISTS subquery, so an entity
        with multiple type rows still counts once).
      - `predicates` restricts to those predicate labels; omitted ⇒ count across
        all predicates. (Not validated against the predicate vocabulary here — an
        unknown predicate simply matches nothing.)

    Degrades gracefully: a missing `knowledge_graph`/`entity_types` table (an old
    DB) yields a zero/empty `GraphCount` rather than raising, mirroring
    `_message_fts_search`'s `sqlite3.OperationalError` tolerance.
    """
    counted_col = (
        "subject_canonical" if count == "subject" else "object_canonical"
    )

    # Resolve surface anchors to canonicals up front so both the WHERE clause and
    # the echoed `filters` carry the post-resolution values.
    subject_canon = resolve(conn, subject) if subject is not None else None
    object_canon = resolve(conn, object) if object is not None else None
    predicate_list = (
        [p for p in (normalize_predicate(p) for p in predicates) if p]
        if predicates is not None
        else None
    )

    where: list[str] = [live_edge_predicate(include_derived=include_derived)]
    params: list[object] = []
    if subject_canon is not None:
        where.append("subject_canonical = ?")
        params.append(subject_canon)
    if object_canon is not None:
        where.append("object_canonical = ?")
        params.append(object_canon)
    if predicate_list is not None:
        if not predicate_list:
            # An explicit but empty predicate set matches nothing — short-circuit
            # to an empty result instead of building `IN ()` (a SQL syntax error).
            return GraphCount(
                count=0,
                counted=count,
                entities=[],
                filters=_count_filters(
                    count, predicate_list, subject_canon, object_canon,
                    subject_type, object_type, include_derived,
                ),
            )
        placeholders = ",".join("?" * len(predicate_list))
        where.append(f"predicate IN ({placeholders})")
        params.extend(predicate_list)
    if subject_type is not None:
        where.append(
            "EXISTS (SELECT 1 FROM entity_types et "
            "WHERE et.entity_canonical = knowledge_graph.subject_canonical "
            "AND et.type = ?)"
        )
        params.append(subject_type)
    if object_type is not None:
        where.append(
            "EXISTS (SELECT 1 FROM entity_types et "
            "WHERE et.entity_canonical = knowledge_graph.object_canonical "
            "AND et.type = ?)"
        )
        params.append(object_type)

    where_sql = " AND ".join(where)
    filters = _count_filters(
        count, predicate_list, subject_canon, object_canon,
        subject_type, object_type, include_derived,
    )

    try:
        # Exact count over the DISTINCT counted column — independent of the
        # evidence cap below, so the number is right even when the list is short.
        total = conn.execute(
            f"SELECT COUNT(DISTINCT {counted_col}) FROM knowledge_graph "
            f"WHERE {where_sql}",
            params,
        ).fetchone()[0]
        rows = conn.execute(
            f"SELECT DISTINCT {counted_col} FROM knowledge_graph "
            f"WHERE {where_sql} ORDER BY {counted_col} LIMIT ?",
            params + [_COUNT_EVIDENCE_CAP],
        ).fetchall()
    except sqlite3.OperationalError:
        return GraphCount(count=0, counted=count, entities=[], filters=filters)

    return GraphCount(
        count=int(total or 0),
        counted=count,
        entities=[r[0] for r in rows],
        filters=filters,
    )


def normalize_predicate(predicate: str) -> str:
    """Fold a caller-supplied predicate label to its stored form (strip + lower).

    Predicates are stored verbatim from the CHECK-constrained vocabulary
    (`uses`, `depends_on`, …), all lowercase with underscores, so a caller
    passing "Depends_On" or " uses " still matches. Empty/whitespace yields ""
    so `count_relations` can drop it."""
    return predicate.strip().lower()


def _count_filters(
    count: str,
    predicates: list[str] | None,
    subject: str | None,
    object: str | None,
    subject_type: str | None,
    object_type: str | None,
    include_derived: bool,
) -> dict[str, object]:
    """Assemble the `GraphCount.filters` audit dict, omitting unset filters so the
    echo shows only what actually constrained the count."""
    filters: dict[str, object] = {
        "counted": count,
        "include_derived": include_derived,
    }
    if predicates is not None:
        filters["predicates"] = list(predicates)
    if subject is not None:
        filters["subject"] = subject
    if object is not None:
        filters["object"] = object
    if subject_type is not None:
        filters["subject_type"] = subject_type
    if object_type is not None:
        filters["object_type"] = object_type
    return filters
