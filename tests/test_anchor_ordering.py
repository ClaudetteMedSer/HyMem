"""One portable total anchor order, with each consumer's authority gate intact."""
from __future__ import annotations

import pytest

from hymem import HyMem
from hymem.core import db as core_db, graph
from hymem.dreaming.aggregate import _anchor_facts
from hymem.dreaming.aggregation_generation import aggregation_generation_binding
from hymem.dreaming.aggregation_provenance import load_knowledge_graph_anchor_inputs
from hymem.dreaming.bitemporal import stamp_validity
from hymem.extraction.llm import StubLLMClient
from hymem.query.state_anchor import select_anchor_edges
from tests.test_digest_squeeze_probe import _seed_exact_kg_claims


_CLAIMS = [
    ("e", "uses", "sqlite"), ("a", "uses", "redis"),
    ("a", "uses", "postgres"), ("a", "prefers", "mysql"),
    ("d", "uses", "redis"), ("b", "uses", "mysql"),
]


def _render(rows):
    return [" ".join(row[key] for key in (
        "subject_canonical", "predicate", "object_canonical",
    )) for row in rows]


def _seed(conn, cfg, *, reverse=False):
    _seed_exact_kg_claims(conn, cfg, list(reversed(_CLAIMS)) if reverse else _CLAIMS,
                          source_tag="shared-anchor-order")
    stamp_validity(conn)
    conn.execute("UPDATE knowledge_graph SET last_seen='2024-01-01 00:00:00'")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("clock_case", ["equal", "newer", "equivalent", "malformed", "future", "margin"])
def test_exact_anchor_order_is_insertion_independent_and_bounded(cfg, reverse, clock_case):
    hy = HyMem(cfg)
    try:
        with core_db.transaction(hy.conn):
            _seed(hy.conn, cfg, reverse=reverse)
            if clock_case == "newer":
                hy.conn.execute("UPDATE knowledge_graph SET last_seen='2024-01-01T00:00:01Z' WHERE subject_canonical='e'")
            elif clock_case == "equivalent":
                hy.conn.execute("UPDATE knowledge_graph SET last_seen='2024-01-01T02:00:00+02:00' WHERE subject_canonical='a'")
                hy.conn.execute("UPDATE knowledge_graph SET last_seen='2024-01-01T00:00:00Z' WHERE subject_canonical='e'")
            elif clock_case in {"malformed", "future"}:
                hy.conn.execute("UPDATE knowledge_graph SET last_seen=? WHERE subject_canonical='a' AND predicate='prefers'",
                                ("not-a-clock" if clock_case == "malformed" else "9999-01-01T00:00:00Z",))
            elif clock_case == "margin":
                # Recency only breaks equal margins, not the other way round.
                hy.conn.execute("UPDATE knowledge_graph SET last_seen='2024-01-01T00:00:01Z' WHERE subject_canonical='a'")
                hy.conn.execute("UPDATE knowledge_graph SET pos_evidence=pos_evidence+1 WHERE subject_canonical='e'")
        expected = sorted(" ".join(claim) for claim in _CLAIMS)
        if clock_case in {"newer", "margin"}:
            expected.remove("e uses sqlite")
            expected.insert(0, "e uses sqlite")
        elif clock_case in {"malformed", "future"}:
            expected.remove("a prefers mysql")
            expected.append("a prefers mysql")
        before = hy.conn.total_changes
        for cap in (-1, 0, 1, 2, 4, 6, 50):
            wanted = expected[:max(0, cap)]
            assert _render(select_anchor_edges(hy.conn, cap)) == wanted
            assert [proof.rendered_text for proof in load_knowledge_graph_anchor_inputs(hy.conn, cap)] == wanted
            assert _anchor_facts(hy.conn, cap) == wanted
        assert hy.conn.total_changes == before
    finally:
        hy.close()


def test_digest_proof_filter_still_precedes_shared_order_cap(cfg):
    hy = HyMem(cfg)
    try:
        with core_db.transaction(hy.conn):
            _seed(hy.conn, cfg)
            # High-ranking historical rows are still live retrieval candidates,
            # but cannot consume any of the digest's exact-proof budget. More
            # than one fetch batch catches an accidental SQL LIMIT before proof.
            for index in range(65):
                hy.conn.execute(
                    "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
                    "pos_evidence,neg_evidence,last_seen,status,derived) "
                    "VALUES (?,'uses','legacy',99,0,'2024-01-01 00:00:00','active',0)",
                    (f"unproven{index:02d}",),
                )
        before = hy.conn.total_changes
        assert _render(select_anchor_edges(hy.conn, 1)) == ["unproven00 uses legacy"]
        assert _anchor_facts(hy.conn, 1) == ["a prefers mysql"]
        assert hy.conn.total_changes == before
    finally:
        hy.close()


@pytest.mark.parametrize("alias", ["kg", "edge_2", None])
def test_shared_order_alias_is_exact(alias):
    prefix = f"{alias}." if alias else ""
    order = graph.anchor_edge_order_sql(alias)
    assert order.startswith(f"{prefix}pos_evidence - {prefix}neg_evidence DESC, ")
    assert graph.graph_clock_order_sql(prefix + "last_seen") in order
    assert order.endswith(f"{prefix}subject_canonical, {prefix}predicate, {prefix}object_canonical, {prefix}id")


@pytest.mark.parametrize("alias", ["", "kg; DROP TABLE knowledge_graph", "kg.foo", "kg --", 1])
def test_shared_order_rejects_non_identifier_aliases(alias):
    with pytest.raises(ValueError, match="alias"):
        graph.anchor_edge_order_sql(alias)


def test_rebound_shared_order_is_consumed_and_changes_aggregation_generation(cfg, monkeypatch):
    hy = HyMem(cfg)
    client = StubLLMClient(default="[]")
    try:
        with core_db.transaction(hy.conn):
            _seed(hy.conn, cfg)
        before = aggregation_generation_binding(cfg, client)
        assert aggregation_generation_binding(cfg, client) == before

        def reversed_order(alias=None):
            prefix = f"{alias}." if alias else ""
            return f"{prefix}subject_canonical DESC, {prefix}predicate DESC, {prefix}object_canonical DESC, {prefix}id"

        monkeypatch.setattr(graph, "anchor_edge_order_sql", reversed_order)
        assert _render(select_anchor_edges(hy.conn, 1)) == ["e uses sqlite"]
        assert _anchor_facts(hy.conn, 1) == ["e uses sqlite"]
        after = aggregation_generation_binding(cfg, client)
        assert after["contract"]["implementation_sha256"] != before["contract"]["implementation_sha256"]
        assert after["generation_key"] != before["generation_key"]
        assert client.calls == []
    finally:
        hy.close()


@pytest.mark.parametrize("surface", ["graph_clock_order_sql", "bounded_graph_clock_sql", "EVENT_CLOCK_SKEW_SECONDS"])
def test_aggregation_identity_binds_loaded_anchor_clock_helpers_and_policy(cfg, monkeypatch, surface):
    client = StubLLMClient(default="[]")
    before = aggregation_generation_binding(cfg, client)
    original = getattr(graph, surface)
    if callable(original):
        def replaced_clock(*args, **kwargs):
            return original(*args, **kwargs) + " "
        monkeypatch.setattr(graph, surface, replaced_clock)
    else:
        monkeypatch.setattr(graph, surface, original + 1)
    after = aggregation_generation_binding(cfg, client)
    assert after["contract"]["implementation_sha256"] != before["contract"]["implementation_sha256"]
    assert after["generation_key"] != before["generation_key"]
    assert client.calls == []
