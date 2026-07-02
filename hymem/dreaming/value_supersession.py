"""Single-assertion supersession for typed-value knowledge-graph edges (v15).

The phase-3 retract rule (``neg_evidence >= 2*pos_evidence + zombie_neg_threshold``,
or confidence decay) is *evidence-accumulation*: it needs repeated contradiction
to close an edge. A knowledge UPDATE — "the coverage target is now 78%" after an
earlier "65%" — is a single authoritative event that emits only POSITIVE evidence
for the new value and nothing against the old, so accumulation never fires and
both values stay active (see ``tests/test_ku_update_supersession_repro.py``).

This module closes that gap where it is *safe*: two active edges that share
subject + predicate but point at different **typed-value** objects, where the
newer-``valid_at`` value supersedes the older — the older edge is retracted and
its validity interval closed at the newer edge's ``valid_at`` (the world date the
new value took over).

Discriminator (v3). A value is "typed" when it is a NUMBER (count / percentage /
quantity / currency), a DATE, or a VERSION. The type is read from the ``object_canonical``
string itself — ``"165"``, ``"65_percent"``, ``"april_5_2024"`` — and NOT from the
``kg_evidence.value_numeric`` metadata column. v1 keyed on that column and never
fired against a real extractor: production LLMs capture the number in the object
string but almost never populate ``value_numeric`` (a box run measured 1 of 207
evidence rows). The string parse is AUTHORITATIVE for the class: a free-text
object never competes, even when an extractor tagged its evidence numeric — a
year lifted from ``vintage_omega_seamaster_watch`` must not route the possession
into the numeric pool (v3.1 fix; the earlier fast-path that trusted the column
over the parse was the v1 mistake resurfacing). The column survives only as a
unit REFINEMENT: it fills the missing unit on an object that already parsed as a
bare number, so ``"165"`` with evidence unit ``percent`` competes with
``"78_percent"``. Two edges compete only when they parse to the **same class**
and (for numbers) a **compatible unit**; a string object like
``adidas_black_sneakers`` parses to ``None`` and never competes — so
multi-valued relations (a project that uses many tools, a person with several
possessions) are never collapsed. This is the correctness guard, in place of a
predicate allow-list (cf. the functional ``_EXCLUSIVE_PREDICATES`` in
``query/conflicts.py``).

A VERSION (v3) is a dotted numeric core that v2 left as free text: a bare
``2.3.1`` (three or more components) or an alpha-prefixed ``python_3.12`` /
``api_v2.3`` (canonicalization flattens dots to underscores, so ``2_3_1`` and
``python_3_12`` are the shapes that actually mint and are recognised too). The
alpha prefix is the compatibility key — versions of *different* things must
never compete, so ``python_3.12`` can only ever supersede another ``python_*``
version, never ``node_20``. Undotted single-number names (``sprint_3``,
``endpoint_v2``, ``node_20``) are NOT versions: those are typically distinct
coexisting entities, and collapsing them would destroy multi-valued facts.

Residual risk: a genuinely *multi-valued numeric* attribute on one subject +
predicate (two exam scores, say) would be seen as competing. That is why every
supersession is logged at INFO as ``bitemporal.supersede ...`` — the log is a free
collateral audit across a whole dream run, independent of any downstream scoring.

Default ON via ``cfg.value_supersession_enabled`` since 2026-07-02: the LME
guard A/B (500q) came back score-neutral (+0.8pp overall) with zero false
positives across all three firings — too small a footprint for the benchmark to
size, so the flip is a correctness call (a superseded edge left active is wrong
context wherever graph facts are consumed) and the INFO audit line above is the
production tripwire. Set the flag False to disable. Idempotent: it only flips
``active`` -> ``retracted`` and write-once ``invalid_at`` via COALESCE, so
re-running a dream cycle is a no-op.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict

from hymem.config import HyMemConfig

log = logging.getLogger(__name__)

_MONTHS = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
    }
)

# An ISO-8601 calendar date anywhere in the string.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# A whole-string number with optional leading currency symbol and trailing unit:
#   "165", "65_percent", "78%", "$120", "120_usd", "-3.5_kg"
_NUMERIC = re.compile(
    r"([$€£]?)\s*([+-]?\d+(?:\.\d+)?)\s*[_\s]*([a-z%][a-z_%]*)?"
)
# A whole-string version: an optional alpha prefix (the compatibility key), an
# optional literal ``v``, then a dotted numeric core with at least two
# components: "2.3.1", "python_3.12", "api_v2.3", "v2.3". Canonicalization
# flattens dots to underscores ("2.3.1" mints as "2_3_1"), so "_" is accepted
# as a core separator too. The core REQUIRES a separator, so single-number
# names ("sprint_3", "endpoint_v2", "node_20") never match. Checked only after
# the number check, so bare two-part decimals ("3.12") keep classifying as
# numbers.
_VERSION = re.compile(
    r"(?:([a-z][a-z_]*?)[_\s]+)?(v?)(\d+(?:[._]\d+)+)"
)


def _norm_unit(unit: str | None) -> str | None:
    """Normalise a unit for compatibility comparison. Bare numbers (no unit)
    normalise to None and are only compatible with other bare numbers."""
    return unit.strip().lower() if unit and unit.strip() else None


def _classify_object(obj: str | None) -> tuple[str, str | None] | None:
    """Classify a canonical object string as a typed value.

    Returns ``("date", None)`` for a calendar date, ``("num", unit)`` for a
    number (``unit`` normalised, ``None`` for a bare count), ``("ver", key)``
    for a version (``key`` is the alpha prefix, ``None`` for a bare/``v``-only
    core), or ``None`` for a free-text object that must never be treated as a
    single-valued quantity. Dates are checked first so an embedded year is not
    mistaken for a count; versions are checked last so only strings that would
    otherwise be free text can become versions (a bare decimal stays a number).
    """
    if not obj:
        return None
    s = obj.strip().lower()
    if not s:
        return None

    # Date: an ISO date, or a month name alongside a number (a day or year).
    if _ISO_DATE.search(s):
        return ("date", None)
    tokens = re.split(r"[_\s,]+", s)
    if any(t in _MONTHS for t in tokens) and any(re.search(r"\d", t) for t in tokens):
        return ("date", None)

    # Number: the whole string must be a (currency?) number with an optional unit.
    m = _NUMERIC.fullmatch(s)
    if m:
        currency, _, suffix = m.group(1), m.group(2), m.group(3)
        unit = suffix or (currency or None)
        return ("num", _norm_unit(unit))

    # Version: only reached by strings the number check rejected, so nothing
    # previously classified can change class. A prefixed core needs >=2
    # components; a bare core needs >=3 (or an explicit leading "v") — a bare
    # two-component core is a decimal and was already caught above.
    m = _VERSION.fullmatch(s)
    if m:
        prefix, vee, core = m.group(1), m.group(2), m.group(3)
        if prefix:
            return ("ver", prefix)
        if vee or len(re.split(r"[._]", core)) >= 3:
            return ("ver", None)

    return None


def supersede_competing_values(conn: sqlite3.Connection, cfg: HyMemConfig) -> int:
    """Retract the older of each competing typed-value edge pair. Returns the
    number of edges retracted.

    For every group of active, non-derived edges sharing subject + predicate, a
    value class and (for numbers) a compatible unit, the edge with the **latest**
    ``valid_at`` is the current value; every other edge in the group with a
    *different* object and a strictly *earlier* ``valid_at`` is retracted and its
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
               MIN(ev.value_unit) AS ev_unit,
               MAX(CASE WHEN ev.value_numeric IS NOT NULL THEN 1 ELSE 0 END) AS has_numeric
        FROM knowledge_graph kg
        JOIN kg_evidence ev ON ev.edge_id = kg.id AND ev.polarity = 1
        WHERE kg.status = 'active' AND kg.derived = 0 AND kg.valid_at IS NOT NULL
        GROUP BY kg.id
        """
    ).fetchall()

    # Sub-group by (subject, predicate, value-class, unit) so only same-class,
    # same-unit values ever compete. The object-string parse is AUTHORITATIVE
    # for the class: a free-text object never competes, even when an extractor
    # tagged its evidence with value_numeric (a year embedded in a possession
    # name, say) — trusting the column over the parse routed free text into the
    # numeric pool and collapsed multi-valued facts. Class-first also makes a
    # version-shaped object keep its prefix key when value_numeric is tagged,
    # without a special case. has_numeric survives only as a refinement: it
    # fills the MISSING unit on an object that already parsed as a bare number
    # ("165" + evidence unit "percent" competes with "78_percent"); a unit the
    # string itself asserts ("65_percent") is never overridden by metadata.
    groups: dict[tuple[str, str, str, str | None], list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        cls = _classify_object(r["obj"])
        if cls is None:
            continue  # free-text object — never a single-valued quantity
        kind, unit = cls
        if kind == "num" and unit is None and r["has_numeric"]:
            unit = _norm_unit(r["ev_unit"])
        groups[(r["subj"], r["pred"], kind, unit)].append(r)

    to_retract: list[tuple[int, str]] = []  # (older_edge_id, invalid_at)
    for (subj, pred, kind, unit), edges in groups.items():
        if len(edges) < 2:
            continue
        # The current value is the latest-valid edge; everything strictly older
        # with a different value is superseded by it.
        winner = max(edges, key=lambda r: r["valid_at"])
        for e in edges:
            if e["id"] == winner["id"]:
                continue
            if e["obj"] == winner["obj"]:
                continue  # same value, just reinforced at an earlier date
            if e["valid_at"] >= winner["valid_at"]:
                continue  # tie / not strictly older — no temporal basis to order
            to_retract.append((e["id"], winner["valid_at"]))
            # Per-supersession audit line: a free collateral check across a whole
            # dream run, independent of downstream scoring. A clean log is all
            # value updates; a `prefers`/multi-valued row here flags a false hit.
            log.info(
                "bitemporal.supersede subj=%s pred=%s old=%s new=%s "
                "old_valid=%s new_valid=%s kind=%s unit=%s",
                subj, pred, e["obj"], winner["obj"],
                e["valid_at"], winner["valid_at"], kind, unit,
            )

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
