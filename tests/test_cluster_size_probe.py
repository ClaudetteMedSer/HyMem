"""Offline (Mac, StubLLM, no box) tests for the Stage-3a chaining-guard probe
(benchmarks/cluster_size_probe.py).

The probe re-exports the PRODUCTION loader + clusterer from
hymem.dreaming.aggregate, so what's pinned here is the probe's own measurement
layer: read-only store access, the cluster-size histogram, and the verdict
logic that decides whether the chaining guard (raptor_digest_plan.md, 3a) must
be built before `aggregation_nodes_enabled` flips on in prod. Episodes are
seeded with direct SQL inserts (the test_aggregate.py pattern) — no dream pass,
no LLM, no embeddings client beyond schema creation stubs.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from cluster_size_probe import (  # noqa: E402
    DEFAULT_CAP,
    DEFAULT_EMB_THRESHOLD,
    DEFAULT_ENT_THRESHOLD,
    VERDICT_GUARD,
    VERDICT_SKIP,
    main,
    open_store_readonly,
    probe_cluster_sizes,
)

from hymem import HyMem, HyMemConfig, StubEmbeddingClient  # noqa: E402
from hymem.core import db as core_db  # noqa: E402
from hymem.core.vectors import encode_vector  # noqa: E402
from hymem.extraction.llm import StubLLMClient  # noqa: E402


def _make_store(tmp_path: Path, episodes: list[tuple]) -> Path:
    """Create a real hymem store (full schema via HyMem + StubLLM) and seed
    `episodes` = [(eid, sid, entities, vector|None), ...] with direct SQL —
    the same shape a dream pass would have left behind. Returns the db path."""
    cfg = HyMemConfig(root=tmp_path)
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    try:
        with core_db.transaction(hy.conn):
            for eid, sid, entities, vector in episodes:
                hy.conn.execute(
                    "INSERT OR IGNORE INTO sessions(id) VALUES (?)", (sid,))
                hy.conn.execute(
                    """INSERT INTO episodes(id, session_id, title, summary,
                                            participants, start_message_id,
                                            end_message_id, key_entities)
                       VALUES (?, ?, ?, ?, '[]', 1, 2, ?)""",
                    (eid, sid, f"title {eid}", f"summary {eid}",
                     json.dumps(entities)),
                )
                if vector is not None:
                    hy.conn.execute(
                        """INSERT INTO episode_embeddings(episode_id, vector_json,
                                                          model, dim, text_hash)
                           VALUES (?, ?, ?, ?, ?)""",
                        (eid, encode_vector(vector), "stub", len(vector),
                         f"hash-{eid}"),
                    )
    finally:
        hy.close()
    return cfg.db_path


def _chain(n: int) -> list[tuple]:
    """n episodes where ONLY consecutive pairs entity-link at jaccard exactly
    0.50 (= the production ent threshold): {t0}, {t0,t1}, {t1}, {t1,t2}, ...
    Non-adjacent pairs overlap at most 1/3 — so a single big cluster can form
    ONLY through union-find transitivity, the exact failure mode the chaining
    guard exists for."""
    eps = []
    for k in range(n):
        j = k // 2
        ents = [f"t{j}"] if k % 2 == 0 else [f"t{j}", f"t{j + 1}"]
        eps.append((f"e{k:02d}", f"s{k:02d}", ents, None))
    return eps


# ── production-default parity ────────────────────────────────────────────────

def test_probe_defaults_are_the_production_thresholds(tmp_path):
    # The probe must measure at the SAME thresholds build_aggregation_nodes
    # runs with, or its verdict gates a different clusterer than prod ships.
    cfg = HyMemConfig(root=tmp_path)
    assert DEFAULT_EMB_THRESHOLD == cfg.aggregation_emb_threshold == 0.55
    assert DEFAULT_ENT_THRESHOLD == cfg.aggregation_ent_threshold == 0.50
    assert DEFAULT_CAP == 15                       # the plan's mega-cluster line


# ── verdict: clearly separated clusters → skip the guard ─────────────────────

def test_separated_clusters_verdict_skip(tmp_path):
    db = _make_store(tmp_path, [
        # entity-linked pair (jaccard 1.0)
        ("e1", "s1", ["postgres", "billing"], None),
        ("e2", "s2", ["postgres", "billing"], None),
        # second, disjoint entity-linked pair
        ("e3", "s3", ["cycling", "weekend"], None),
        ("e4", "s4", ["cycling", "weekend"], None),
        # embedding-linked pair (identical vectors, disjoint entities)
        ("e5", "s5", ["kafka"], [1.0, 0.0]),
        ("e6", "s6", ["duckdb"], [1.0, 0.0]),
        # singleton
        ("e7", "s7", ["cooking"], [0.0, 1.0]),
    ])
    conn = open_store_readonly(db)
    try:
        rep = probe_cluster_sizes(conn)            # production defaults
    finally:
        conn.close()

    assert rep["n_episodes"] == 7
    assert rep["n_clusters"] == 4                  # three pairs + one singleton
    assert rep["histogram"] == {"1": 1, "2-4": 3, "5-9": 0, "10-14": 0, "15+": 0}
    assert rep["max_cluster_size"] == 2
    assert rep["mean_cluster_size"] == pytest.approx(7 / 4)
    assert rep["guard_needed"] is False
    assert rep["verdict"] == VERDICT_SKIP


# ── verdict: transitive chain → mega-cluster → guard needed ──────────────────

def test_transitive_chain_forms_single_cluster_and_demands_guard(tmp_path):
    # A~B, B~C, C~D each link at exactly jaccard 0.5; A shares NOTHING with D
    # (and only 1/3 with C) — the cluster of 4 exists purely by chaining.
    db = _make_store(tmp_path, [
        ("eA", "s1", ["a"], None),
        ("eB", "s2", ["a", "b"], None),
        ("eC", "s3", ["b"], None),
        ("eD", "s4", ["b", "c"], None),
    ])
    conn = open_store_readonly(db)
    try:
        rep = probe_cluster_sizes(conn, cap=4)     # mega line lowered to the chain
        rep_default_cap = probe_cluster_sizes(conn)
    finally:
        conn.close()

    assert rep["n_episodes"] == 4
    assert rep["n_clusters"] == 1                  # transitivity merged them all
    assert rep["histogram"] == {"1": 0, "2-4": 1, "5-9": 0, "10-14": 0, "15+": 0}
    assert rep["max_cluster_size"] == 4
    # max == cap counts as mega ("max < cap → skip; otherwise guard").
    assert rep["guard_needed"] is True
    assert rep["verdict"] == VERDICT_GUARD
    # The mega-cluster is inspectable: its member session ids are reported.
    assert rep["largest_cluster"]["size"] == 4
    assert rep["largest_cluster"]["session_ids"] == ["s1", "s2", "s3", "s4"]
    assert set(rep["largest_cluster"]["episode_ids"]) == {"eA", "eB", "eC", "eD"}

    # Same store, the plan's default cap of 15: a 4-chain is NOT a mega-cluster.
    assert rep_default_cap["guard_needed"] is False
    assert rep_default_cap["verdict"] == VERDICT_SKIP


def test_chain_of_fifteen_trips_the_default_cap(tmp_path):
    db = _make_store(tmp_path, _chain(15))
    conn = open_store_readonly(db)
    try:
        rep = probe_cluster_sizes(conn)            # default cap = 15
    finally:
        conn.close()

    assert rep["n_episodes"] == 15
    assert rep["n_clusters"] == 1
    assert rep["max_cluster_size"] == 15
    assert rep["histogram"]["15+"] == 1
    assert rep["guard_needed"] is True
    assert rep["verdict"] == VERDICT_GUARD


# ── read-only contract / edge cases ──────────────────────────────────────────

def test_open_store_readonly_rejects_writes(tmp_path):
    db = _make_store(tmp_path, [("e1", "s1", ["postgres"], None)])
    conn = open_store_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO sessions(id) VALUES ('attempted-write')")
    finally:
        conn.close()


def test_open_store_readonly_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_store_readonly(tmp_path / "nope.sqlite")


def test_empty_store_reports_skip_without_crashing(tmp_path):
    db = _make_store(tmp_path, [])
    conn = open_store_readonly(db)
    try:
        rep = probe_cluster_sizes(conn)
    finally:
        conn.close()
    assert rep["n_episodes"] == 0
    assert rep["n_clusters"] == 0
    assert rep["max_cluster_size"] == 0
    assert rep["largest_cluster"] is None
    assert rep["guard_needed"] is False
    assert rep["verdict"] == VERDICT_SKIP


# ── thin CLI ─────────────────────────────────────────────────────────────────

def test_cli_prints_verdict_and_dumps_json(tmp_path, capsys):
    db = _make_store(tmp_path, [
        ("eA", "s1", ["a"], None),
        ("eB", "s2", ["a", "b"], None),
        ("eC", "s3", ["b"], None),
        ("eD", "s4", ["b", "c"], None),
    ])
    out_json = tmp_path / "sizes.json"
    main([str(db), "--cap", "4", "--grid", "0.55:0.50,0.55:0.90",
          "--json", str(out_json)])

    out = capsys.readouterr().out
    assert "VERDICT:" in out
    assert VERDICT_GUARD in out                    # 0.50 point chains all four
    assert VERDICT_SKIP in out                     # ent 0.90 breaks every link

    payload = json.loads(out_json.read_text())
    assert len(payload["grid"]) == 2
    chained, broken = payload["grid"]
    assert chained["max_cluster_size"] == 4 and chained["guard_needed"] is True
    assert broken["max_cluster_size"] == 1 and broken["guard_needed"] is False
    # Per-cluster dump present for mega-cluster inspection.
    assert chained["clusters"][0]["session_ids"] == ["s1", "s2", "s3", "s4"]
