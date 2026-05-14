"""Keyword → predicate routing for knowledge-graph queries.

Pure functions, no DB or LLM. Maps natural-language cues in a user query to the
typed predicates they imply, so the graph ranker can boost matching edges and
pull in predicate-routed candidates. Routing only ever *adds* signal — it never
filters edges out.
"""

from __future__ import annotations

import re

from hymem.extraction.prompts import ALLOWED_PREDICATES

# Phrase keywords are matched as substrings; single-word keywords are matched on
# word boundaries. Each maps to the set of predicates it implies.
_KEYWORD_PREDICATES: dict[str, frozenset[str]] = {
    # uses / runs_on
    "technologies": frozenset({"uses", "runs_on"}),
    "tech stack": frozenset({"uses", "runs_on"}),
    "built with": frozenset({"uses", "runs_on", "implements"}),
    "uses": frozenset({"uses"}),
    "using": frozenset({"uses"}),
    "utilizes": frozenset({"uses"}),
    "works with": frozenset({"uses"}),
    "runs on": frozenset({"runs_on"}),
    "running on": frozenset({"runs_on"}),
    "executes on": frozenset({"runs_on"}),
    # dependencies
    "depends on": frozenset({"depends_on"}),
    "depend on": frozenset({"depends_on"}),
    "dependency": frozenset({"depends_on"}),
    "dependencies": frozenset({"depends_on"}),
    "requires": frozenset({"depends_on", "requires_version"}),
    "needs": frozenset({"depends_on"}),
    "version": frozenset({"requires_version"}),
    # preferences / rejections
    "prefers": frozenset({"prefers"}),
    "prefer": frozenset({"prefers"}),
    "preferred": frozenset({"prefers"}),
    "favors": frozenset({"prefers"}),
    "rejects": frozenset({"rejects"}),
    "rejected": frozenset({"rejects"}),
    "refuses": frozenset({"rejects"}),
    "avoids": frozenset({"avoids", "rejects"}),
    "avoid": frozenset({"avoids", "rejects"}),
    "steers clear": frozenset({"avoids"}),
    # replacement / conflict
    "replaces": frozenset({"replaces"}),
    "replaced": frozenset({"replaces"}),
    "supersedes": frozenset({"replaces"}),
    "instead of": frozenset({"replaces"}),
    "conflicts": frozenset({"conflicts_with"}),
    "conflict": frozenset({"conflicts_with"}),
    "incompatible": frozenset({"conflicts_with"}),
    # deployment
    "deploys to": frozenset({"deploys_to"}),
    "deployed": frozenset({"deploys_to"}),
    "deployment": frozenset({"deploys_to"}),
    "released to": frozenset({"deploys_to"}),
    # structure
    "part of": frozenset({"part_of", "contains"}),
    "component of": frozenset({"part_of"}),
    "belongs to": frozenset({"part_of"}),
    "contains": frozenset({"contains"}),
    "includes": frozenset({"contains"}),
    "made up of": frozenset({"contains"}),
    "equivalent": frozenset({"equivalent_to"}),
    "same as": frozenset({"equivalent_to"}),
    "synonymous": frozenset({"equivalent_to"}),
    # implementation / config
    "implements": frozenset({"implements"}),
    "implementation": frozenset({"implements"}),
    "fulfills": frozenset({"implements"}),
    "configured with": frozenset({"configured_with"}),
    "configuration": frozenset({"configured_with"}),
    "configured": frozenset({"configured_with"}),
    "set up with": frozenset({"configured_with"}),
    # connectivity
    "connects to": frozenset({"connects_to"}),
    "connected to": frozenset({"connects_to"}),
    "talks to": frozenset({"connects_to"}),
    "integrates with": frozenset({"connects_to"}),
    # generation / testing
    "generates": frozenset({"generates"}),
    "produces": frozenset({"generates"}),
    "outputs": frozenset({"generates"}),
    "tested by": frozenset({"tested_by"}),
    "tested with": frozenset({"tested_by"}),
    "testing": frozenset({"tested_by"}),
    "tests": frozenset({"tested_by"}),
}

# Fail fast if a mapped predicate ever drifts out of the locked vocabulary.
_allowed = set(ALLOWED_PREDICATES)
for _kw, _preds in _KEYWORD_PREDICATES.items():
    _unknown = _preds - _allowed
    assert not _unknown, f"predicate routing maps unknown predicate(s) {_unknown} for '{_kw}'"


def route_predicates(query: str) -> frozenset[str]:
    """Return the set of predicates implied by keywords in `query`.

    Phrase keywords (containing a space) match as substrings; single-word
    keywords match on word boundaries. The empty set means no routing applies.
    """
    lowered = query.lower()
    matched: set[str] = set()
    for keyword, predicates in _KEYWORD_PREDICATES.items():
        if " " in keyword:
            if keyword in lowered:
                matched |= predicates
        elif re.search(rf"\b{re.escape(keyword)}\b", lowered):
            matched |= predicates
    return frozenset(matched)
