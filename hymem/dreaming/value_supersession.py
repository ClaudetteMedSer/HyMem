"""Single-assertion supersession for typed-value knowledge-graph edges (v15).

The phase-3 retract rule (``neg_evidence >= 2*pos_evidence + zombie_neg_threshold``,
or confidence decay) is *evidence-accumulation*: it needs repeated contradiction
to close an edge. A knowledge UPDATE — "the coverage target is now 78%" after an
earlier "65%" — is a single authoritative event that emits only POSITIVE evidence
for the new value and nothing against the old, so accumulation never fires and
both values stay active (see ``tests/test_ku_update_supersession_repro.py``).

This module closes that gap where it is *safe*: two active edges that share
subject + predicate but point at different **typed-value** objects (a numeric
quantity / percentage / count, recognised via the ``kg_evidence.value_numeric``
column, with ``value_unit`` as the compatibility key). For such a competing pair
the newer-``valid_at`` value supersedes the older — the older edge is retracted
and its validity interval closed at the newer edge's ``valid_at`` (the world date
the new value took over).

Typed-value scoping is the correctness guard, and the reason this is keyed on the
object's value type rather than a predicate allow-list (as ``query/conflicts.py``
does with its functional ``_EXCLUSIVE_PREDICATES``): multi-valued relations — a
project that uses many tools, a person with several preferences — carry no numeric
value on their objects, so they are never seen as competing and never collapsed,
while value attributes on *any* predicate ("coverage configured_with 65 -> 78")
are reached. Unit compatibility (``percent`` vs ``percent``, never ``percent`` vs
``MB``) keeps two unrelated numeric attributes that happen to share a subject +
predicate from colliding.

Opt-in via ``cfg.value_supersession_enabled`` (default off until the LME guard
clears). Idempotent: it only flips ``active`` -> ``retracted`` and write-once
``invalid_at`` via COALESCE, so re-running a dream cycle is a no-op.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

from hymem.config import HyMemConfig


def _norm_unit(unit: str | None) -> str | None:
    """Normalise a value_unit for compatibility comparison. Bare numbers (no
    unit) normalise to None and are only compatible with other bare numbers."""
    return unit.strip().lower() if unit else None


def _units_compatible(a: str | None, b: str | None) -> bool:
    """Two typed values may compete only when their units match exactly (both
    None, i.e. bare counts, or the same normalised unit). A unit on one side but
    not the other is treated as incompatible — conservative, so a unit-less count
    is never superseded by a percentage that happens to share subject+predicate."""
    return _norm_unit(a) == _norm_unit(b)


def supersede_competing_values(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    """Retract the older of each competing typed-value edge pair. Returns the
    number of edges retracted.

    For every group of active, non-derived edges sharing subject + predicate and
    carrying a positive numeric value, the edge with the **latest** ``valid_at``
    is the current value; every other edge in the group with a *different* object,
    a *compatible* unit, and a strictly *earlier* ``valid_at`` is retracted and its
    ``invalid_at`` closed at the winner's ``valid_at``. Ties on ``valid_at`` are
    left untouched (no temporal basis to order them).
    """
    rows = conn.execute(
        """
        SELECT kg.id AS id,
               kg.subject_canonical AS subj,
               kg.predicate AS pred,
               kg.object_canonical AS obj,
               kg.valid_at AS valid_at,
               MIN(ev.value_unit) AS unit,
               MAX(CASE WHEN ev.value_numeric IS NOT NULL THEN 1 ELSE 0 END) AS has_numeric
        FROM knowledge_graph kg
        JOIN kg_evidence ev ON ev.edge_id = kg.id AND ev.polarity = 1
        WHERE kg.status = 'active' AND kg.derived = 0 AND kg.valid_at IS NOT NULL
        GROUP BY kg.id
        HAVING has_numeric = 1
        """
    ).fetchall()

    groups: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        groups[(r["subj"], r["pred"])].append(r)

    to_retract: list[tuple[int, str]] = []  # (older_edge_id, invalid_at)
    for edges in groups.values():
        if len(edges) < 2:
            continue
        # The current value is the latest-valid edge; everything strictly older
        # with a different value and a compatible unit is superseded by it.
        winner = max(edges, key=lambda r: r["valid_at"])
        for e in edges:
            if e["id"] == winner["id"]:
                continue
            if e["obj"] == winner["obj"]:
                continue  # same value, just reinforced at an earlier date
            if e["valid_at"] >= winner["valid_at"]:
                continue  # tie / not strictly older — no temporal basis to order
            if not _units_compatible(e["unit"], winner["unit"]):
                continue  # unrelated numeric attribute sharing subject+predicate
            to_retract.append((e["id"], winner["valid_at"]))

    for edge_id, invalid_at in to_retract:
        # Retract removes the stale value from `status='active'` retrieval; the
        # COALESCE keeps any existing invalid_at (idempotent) and otherwise closes
        # the interval at the world date the new value took over. Not routed
        # through bitemporal.stamp_invalidation: there is no negative evidence
        # here, so the supersession date is the winner's valid_at, not a flip time.
        conn.execute(
            """
            UPDATE knowledge_graph
            SET status = 'retracted',
                invalid_at = COALESCE(invalid_at, ?)
            WHERE id = ? AND status = 'active'
            """,
            (invalid_at, edge_id),
        )

    return len(to_retract)
