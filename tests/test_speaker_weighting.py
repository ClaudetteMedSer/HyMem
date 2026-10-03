"""Tests for exact-source speaker-weighted evidence (improv item D).

Evidence weight follows the role of the message cited by the extracted claim,
not the first message in a mixed-role chunk.  Thus a USER assertion still gets
USER weight when an ASSISTANT context turn precedes it.  The cited role is also
recorded in ``kg_evidence.source_role``.
"""

from __future__ import annotations

import dataclasses

from hymem import HyMem
from tests.conftest import make_routed_llm


def _dream_single_use_triple(cfg, turns, obj):
    triples = [{"subject": "app", "predicate": "uses", "object": obj, "polarity": 1}]
    hy = HyMem(cfg, llm=make_routed_llm(triples, []))
    sid = "s_weight"
    hy.open_session(sid)
    for role, content in turns:
        hy.log_message(sid, role, content)
    hy.close_session(sid)
    hy.dream()
    return hy


def _edge(hy, obj):
    pos = hy.conn.execute(
        "SELECT pos_evidence FROM knowledge_graph WHERE object_canonical = ?", (obj,)
    ).fetchone()
    role = hy.conn.execute(
        "SELECT e.source_role FROM kg_evidence e "
        "JOIN knowledge_graph kg ON kg.id = e.edge_id "
        "WHERE kg.object_canonical = ?",
        (obj,),
    ).fetchone()
    return pos, role


def test_user_opened_chunk_doubles_pos_evidence(cfg):
    """A user turn with no preceding assistant turn opens the chunk → weight 2."""
    hy = _dream_single_use_triple(
        cfg,
        [("user", "We rely on postgres for all production data storage these days.")],
        "postgres",
    )
    try:
        pos, role = _edge(hy, "postgres")
        assert pos["pos_evidence"] == 2
        assert role["source_role"] == "user"
    finally:
        hy.close()


def test_assistant_context_does_not_steal_user_claim_weight(cfg):
    """A preceding assistant turn cannot become the user's claim speaker."""
    hy = _dream_single_use_triple(
        cfg,
        [
            ("assistant", "What's your main datastore for the service?"),
            ("user", "We rely on redis for caching across the production environment."),
        ],
        "redis",
    )
    try:
        pos, role = _edge(hy, "redis")
        assert pos["pos_evidence"] == 2
        assert role["source_role"] == "user"
    finally:
        hy.close()


def test_empty_role_weights_disables_weighting(cfg):
    """With an empty evidence_role_weights map, even a user-opened chunk
    contributes weight 1 — proving the weighting is config-driven."""
    flat_cfg = dataclasses.replace(cfg, evidence_role_weights={})
    hy = _dream_single_use_triple(
        flat_cfg,
        [("user", "We rely on kafka for the event backbone in production today.")],
        "kafka",
    )
    try:
        pos, role = _edge(hy, "kafka")
        assert pos["pos_evidence"] == 1
        # The role is still recorded even when its weight is 1.
        assert role["source_role"] == "user"
    finally:
        hy.close()
