"""Report and apply paths for retroactive behavioral-edge dedup.

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


# ── apply tests ───────────────────────────────────────────────────────────


def test_apply_merges_evidence_and_retracts_members(tmp_path):
    """The survivor gets summed evidence and collapsed members are removed."""
    from hymem.core import db as core_db
    from hymem.dreaming.behavioral_dedup import (
        find_behavioral_duplicates,
        apply_behavioral_merges,
    )

    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    s_id = _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0], pos=5, neg=1)
    m_id = _edge(conn, "atta", "prefers", "concise_mode", [0.99, 0.01, 0.0, 0.0], pos=2, neg=0)

    proposals = find_behavioral_duplicates(conn, cosine_threshold=0.9)
    assert len(proposals) == 1

    with core_db.transaction(conn):
        result = apply_behavioral_merges(conn, proposals)

    assert result["clusters_merged"] == 1
    assert result["edges_retracted"] == 1
    assert result["survivors_updated"] == 1

    # Survivor evidence summed.
    survivor = conn.execute(
        "SELECT pos_evidence, neg_evidence, status FROM knowledge_graph WHERE id = ?",
        (s_id,),
    ).fetchone()
    assert survivor["pos_evidence"] == 7  # 5 + 2
    assert survivor["neg_evidence"] == 1  # 1 + 0
    assert survivor["status"] == "active"

    # The alias preserves resolution; an authority-free direct tombstone would
    # be unportable and is therefore removed.
    member = conn.execute(
        "SELECT status FROM knowledge_graph WHERE id = ?", (m_id,)
    ).fetchone()
    assert member is None
    assert conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias='concise_mode'"
    ).fetchone()[0] == "concise"

    hy.close()


def test_apply_is_idempotent(tmp_path):
    """Second apply with same proposals is no-op — idempotent."""
    from hymem.core import db as core_db
    from hymem.dreaming.behavioral_dedup import (
        find_behavioral_duplicates,
        apply_behavioral_merges,
    )

    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0], pos=5)
    _edge(conn, "atta", "prefers", "concise_mode", [0.99, 0.01, 0.0, 0.0], pos=2)

    proposals = find_behavioral_duplicates(conn, cosine_threshold=0.9)

    with core_db.transaction(conn):
        r1 = apply_behavioral_merges(conn, proposals)
    assert r1["edges_retracted"] == 1

    with core_db.transaction(conn):
        r2 = apply_behavioral_merges(conn, proposals)
    assert r2["edges_retracted"] == 0  # already retracted
    assert r2["clusters_merged"] == 0

    # Evidence unchanged by second call.
    pos = conn.execute(
        "SELECT pos_evidence FROM knowledge_graph WHERE object_canonical = 'concise'"
    ).fetchone()["pos_evidence"]
    assert pos == 7

    hy.close()


def test_apply_registers_object_alias(tmp_path):
    """Member object canonical is aliased to survivor object."""
    from hymem.core import db as core_db
    from hymem.dreaming.behavioral_dedup import (
        find_behavioral_duplicates,
        apply_behavioral_merges,
    )

    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0], pos=5)
    _edge(conn, "atta", "prefers", "concise_mode", [0.99, 0.01, 0.0, 0.0], pos=2)

    proposals = find_behavioral_duplicates(conn, cosine_threshold=0.9)

    with core_db.transaction(conn):
        apply_behavioral_merges(conn, proposals)

    alias = conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias = 'concise_mode'"
    ).fetchone()
    assert alias is not None
    assert alias["canonical"] == "concise"

    hy.close()


def test_apply_reassigns_kg_evidence(tmp_path):
    """kg_evidence rows from members move to survivor."""
    from hymem.core import db as core_db
    from hymem.dreaming.behavioral_dedup import (
        find_behavioral_duplicates,
        apply_behavioral_merges,
    )

    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s1')")
    cid = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s1', 'user', 'x')"
    ).lastrowid
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES ('c1', 's1', ?, ?, 'test', 'x')",
        (cid, cid),
    )

    m_id = _edge(conn, "atta", "prefers", "concise_mode", [0.99, 0.01, 0.0, 0.0], pos=2)
    with core_db.evidence_mutation(conn):
        conn.execute(
            "INSERT INTO kg_evidence(edge_id, chunk_id, polarity) "
            "VALUES (?, 'c1', 1)",
            (m_id,),
        )
    s_id = _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0], pos=5)

    proposals = find_behavioral_duplicates(conn, cosine_threshold=0.9)

    with core_db.transaction(conn):
        apply_behavioral_merges(conn, proposals)

    # Evidence should now be on survivor.
    evidence = conn.execute(
        "SELECT edge_id FROM kg_evidence WHERE chunk_id = 'c1' AND polarity = 1"
    ).fetchone()
    assert evidence["edge_id"] == s_id

    hy.close()


def test_apply_with_no_proposals_returns_zeros(tmp_path):
    from hymem.core import db as core_db
    from hymem.dreaming.behavioral_dedup import apply_behavioral_merges

    hy = HyMem(HyMemConfig(root=tmp_path))
    with core_db.transaction(hy.conn):
        result = apply_behavioral_merges(hy.conn, [])
    assert result == {"clusters_merged": 0, "edges_retracted": 0, "survivors_updated": 0}
    hy.close()


def test_apply_via_hy_api(tmp_path):
    """End-to-end through HyMem.apply_behavioral_merges()."""
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0], pos=5)
    _edge(conn, "atta", "prefers", "concise_mode", [0.99, 0.01, 0.0, 0.0], pos=2)

    result = hy.apply_behavioral_merges(cosine_threshold=0.9)

    assert result["proposals_found"] == 1
    assert result["clusters_merged"] == 1
    assert result["edges_retracted"] == 1
    assert result["survivors_updated"] == 1

    # Verify the collapsed identity is represented only by its alias.
    assert conn.execute(
        "SELECT 1 FROM knowledge_graph WHERE object_canonical = 'concise_mode'"
    ).fetchone() is None
    assert conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias='concise_mode'"
    ).fetchone()[0] == "concise"

    hy.close()


def test_apply_records_extraction_feedback(tmp_path):
    """Retracted edges produce extraction_feedback rows when evidence exists."""
    from hymem.core import db as core_db
    from hymem.dreaming.behavioral_dedup import (
        find_behavioral_duplicates,
        apply_behavioral_merges,
    )

    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    # Setup a real chunk so feedback FK is satisfied.
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s1')")
    cid = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s1', 'user', 'I prefer concise')"
    ).lastrowid
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES ('c1', 's1', ?, ?, 'test', 'I prefer concise mode')",
        (cid, cid),
    )

    s_id = _edge(conn, "atta", "prefers", "concise", [1.0, 0.0, 0.0, 0.0], pos=5)
    m_id = _edge(conn, "atta", "prefers", "concise_mode", [0.99, 0.01, 0.0, 0.0], pos=2)
    # Attach evidence to the member edge.
    with core_db.evidence_mutation(conn):
        conn.execute(
            "INSERT INTO kg_evidence(edge_id, chunk_id, polarity) "
            "VALUES (?, 'c1', 1)",
            (m_id,),
        )

    proposals = find_behavioral_duplicates(conn, cosine_threshold=0.9)

    with core_db.transaction(conn):
        apply_behavioral_merges(conn, proposals)

    fb_count = conn.execute(
        "SELECT COUNT(*) AS c FROM extraction_feedback WHERE feedback_type = 'retracted'"
    ).fetchone()["c"]
    assert fb_count >= 1

    hy.close()
