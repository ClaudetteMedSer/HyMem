"""v9 extraction extension: personal-life value facts reach the graph.

The KU coverage gap (only 14% of knowledge-update gold values minted as edges)
was that possessions, residence, activities, and personal metrics had no
expressible predicate, so the value never entered the knowledge graph. v9 adds
`owns`, `located_in`, `participates_in`, `has_attribute` to ALLOWED_PREDICATES and
de-techs the extraction prompt. This test pins the mechanical guarantee: those
predicates now survive validation (hymem/extraction/triples.py drops anything not
in ALLOWED_PREDICATES) and mint edges through the real dreaming path.
"""

from __future__ import annotations

import json

import pytest

from hymem.extraction.prompts import ALLOWED_PREDICATES
from hymem.extraction.llm import StubLLMClient


def _chunk(triples: list[dict]) -> str:
    return json.dumps({"triples": triples, "markers": []})


@pytest.mark.parametrize("predicate", ["owns", "located_in", "participates_in", "has_attribute"])
def test_personal_predicate_is_allowed(predicate):
    assert predicate in ALLOWED_PREDICATES


def test_personal_value_facts_mint_edges(hy):
    """A possession, a residence, an activity, and a personal metric — the four
    KU-zero classes — all mint active edges through dreaming."""
    sid = "s_personal"
    hy.open_session(sid)
    hy.log_message(
        sid, "user",
        "I drive a Ford F-150, we just moved to Austin, I play tennis every "
        "Tuesday, and my resting heart rate is 60 bpm.",
        created_at="2024-03-01 09:00:00",
    )
    hy.close_session(sid)

    hy.set_llm(StubLLMClient(
        fixtures={"Ford F-150": _chunk([
            {"subject": "user", "predicate": "owns", "object": "ford_f_150", "polarity": 1},
            {"subject": "user", "predicate": "located_in", "object": "austin", "polarity": 1},
            {"subject": "user", "predicate": "participates_in", "object": "tennis",
             "polarity": 1, "temporal_scope": "every Tuesday"},
            {"subject": "user", "predicate": "has_attribute", "object": "60_bpm",
             "polarity": 1, "value_numeric": 60, "value_unit": "bpm"},
        ])},
        default="[]",
    ))
    report = hy.dream()
    assert report.chunks_processed >= 1

    minted = {
        (r["predicate"], r["object_canonical"])
        for r in hy.conn.execute(
            "SELECT predicate, object_canonical FROM knowledge_graph "
            "WHERE subject_canonical = 'user' AND status = 'active'"
        ).fetchall()
    }
    for expected in [
        ("owns", "ford_f_150"),
        ("located_in", "austin"),
        ("participates_in", "tennis"),
        ("has_attribute", "60_bpm"),
    ]:
        assert expected in minted, f"{expected} not minted; got {sorted(minted)}"
