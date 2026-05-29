"""Dry-run report for retroactive behavioral-edge dedup.

Behavioral edges (`prefers` / `avoids` / `rejects`) minted before same-wave
collapse existed don't retroactively merge. `behavioral_duplicate_report` is a
read-only report of which ones *would* merge on semantic similarity alone (the
lexical gate is intentionally dropped, since behavioral objects are paraphrases
rather than tool-name siblings). These tests pin that contract.
"""
from __future__ import annotations

import json

from hymem import HyMem, HyMemConfig


def _edge(conn, subj, pred, obj, vec, *, pos=1, neg=0):
    cur = conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, status, derived) VALUES (?, ?, ?, ?, ?, 'active', 0)",
        (subj, pred, obj, pos, neg),
    )
    conn.execute(
        "INSERT INTO edge_embeddings(edge_text, vector_json, model, dim) "
        "VALUES (?, ?, 'fake', 4)",
        (f"{subj} {pred} {obj}", json.dumps(vec)),
    )
    return cur.lastrowid


def test_semantically_close_behavioral_objects_merge(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0], pos=5)
    _edge(conn, "atta", "prefers", "concise_responses", [0.99, 0.01, 0.0, 0.0], pos=2)

    report = hy.behavioral_duplicate_report(cosine_threshold=0.9)

    assert report["clusters"] == 1
    assert report["edges_collapsed"] == 1
    merge = report["merges"][0]
    assert merge["subject"] == "atta"
    assert merge["predicate"] == "prefers"
    # Highest-evidence edge is the proposed survivor.
    assert merge["survivor"]["object"] == "concise"
    assert [m["object"] for m in merge["members"]] == ["concise_responses"]
    hy.close()


def test_lexical_gate_is_dropped(tmp_path):
    # `concise` and `brevity` share no token and aren't substrings — normal
    # dedup's lexical gate would block them — but they're semantically close, so
    # the behavioral report merges them. This is the whole point of the sweep.
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0], pos=3)
    _edge(conn, "atta", "prefers", "brevity", [0.98, 0.02, 0.0, 0.0], pos=1)

    report = hy.behavioral_duplicate_report(cosine_threshold=0.9)
    assert report["clusters"] == 1
    assert {m["object"] for m in report["merges"][0]["members"]} == {"brevity"}
    hy.close()


def test_semantically_distinct_objects_do_not_merge(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0])
    _edge(conn, "atta", "prefers", "verbose", [0.0, 1.0, 0.0, 0.0])  # orthogonal

    report = hy.behavioral_duplicate_report(cosine_threshold=0.9)
    assert report["clusters"] == 0
    assert report["edges_collapsed"] == 0
    hy.close()


def test_different_subjects_are_separate(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0])
    _edge(conn, "sara", "prefers", "concise", [1.0, 0.0, 0.0, 0.0])

    report = hy.behavioral_duplicate_report(cosine_threshold=0.9)
    assert report["clusters"] == 0  # same object, but different subjects
    hy.close()


def test_non_behavioral_predicates_excluded(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    # `uses` is not behavioral — even with near-identical vectors it's ignored.
    _edge(conn, "app", "uses", "redis", [1.0, 0.0, 0.0, 0.0])
    _edge(conn, "app", "uses", "redash", [0.99, 0.01, 0.0, 0.0])

    report = hy.behavioral_duplicate_report(cosine_threshold=0.9)
    assert report["clusters"] == 0
    hy.close()


def test_report_is_read_only(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0])
    _edge(conn, "atta", "prefers", "concise_mode", [0.99, 0.01, 0.0, 0.0])
    conn.commit()
    before = conn.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0]

    hy.behavioral_duplicate_report(cosine_threshold=0.9)

    after = conn.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0]
    assert after == before  # nothing merged or deleted
    hy.close()


def test_threshold_controls_aggressiveness(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0])
    # cosine ≈ 0.928 to the survivor — merges at 0.9, not at 0.97.
    _edge(conn, "atta", "prefers", "terse", [0.8, 0.3, 0.0, 0.0])

    assert hy.behavioral_duplicate_report(cosine_threshold=0.90)["clusters"] == 1
    assert hy.behavioral_duplicate_report(cosine_threshold=0.97)["clusters"] == 0
    hy.close()
