"""Tests for hy.timeline(entity) (improv item F): first-seen active edge per
predicate for an entity, oldest first, read from knowledge_graph.first_seen.
"""

from __future__ import annotations


def _seed(hy, subj, pred, obj, days_ago, *, status="active"):
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, first_seen, status) "
        "VALUES (?, ?, ?, 1, datetime('now', ?), ?)",
        (subj, pred, obj, f"-{days_ago} days", status),
    )


def test_timeline_returns_first_seen_per_predicate(hy):
    # Two 'uses' edges for app — the older one wins for that predicate.
    _seed(hy, "app", "uses", "postgres", days_ago=100)
    _seed(hy, "app", "uses", "redis", days_ago=10)
    _seed(hy, "app", "depends_on", "kafka", days_ago=50)

    entries = hy.timeline("app")

    by_pred = {e.predicate: e for e in entries}
    assert set(by_pred) == {"uses", "depends_on"}
    # Oldest 'uses' edge is the postgres one (100 days ago).
    assert by_pred["uses"].object == "postgres"
    assert by_pred["depends_on"].object == "kafka"
    # Ordered oldest-first overall: uses(100d) before depends_on(50d).
    assert [e.predicate for e in entries] == ["uses", "depends_on"]


def test_timeline_matches_object_position_and_resolves_aliases(hy):
    hy.register_alias("Postgres", "postgres")
    _seed(hy, "app", "uses", "postgres", days_ago=30)

    # Surface form resolves; entity appears as the object here.
    entries = hy.timeline("Postgres")
    assert len(entries) == 1
    assert entries[0].subject == "app"
    assert entries[0].object == "postgres"
    assert entries[0].predicate == "uses"


def test_timeline_excludes_retracted_edges(hy):
    _seed(hy, "app", "uses", "mysql", days_ago=200, status="retracted")
    _seed(hy, "app", "uses", "postgres", days_ago=20)

    entries = hy.timeline("app")
    objs = {e.object for e in entries}
    assert objs == {"postgres"}  # retracted mysql excluded


def test_timeline_unknown_entity_is_empty(hy):
    assert hy.timeline("nonexistent_entity") == []
