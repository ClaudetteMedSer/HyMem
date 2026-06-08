"""Tests for hy.count_relations(...): the graph-native counting primitive over
active knowledge_graph edges, for in-domain "how many X …" questions.

Covers the load-bearing subject-vs-object contract, alias/type resolution,
active-only filtering, predicate-agnostic counting, and graceful empties.
"""

from __future__ import annotations


def _seed(hy, subj, pred, obj, *, status="active"):
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, status) VALUES (?, ?, ?, 1, ?)",
        (subj, pred, obj, status),
    )


def _type(hy, entity, type_label):
    hy.conn.execute(
        "INSERT INTO entity_types(entity_canonical, type) VALUES (?, ?)",
        (entity, type_label),
    )


def test_count_distinct_subjects_for_predicate_and_object(hy):
    # "how many services depend on redis?" -> distinct subjects.
    _seed(hy, "billing", "depends_on", "redis")
    _seed(hy, "auth", "depends_on", "redis")
    _seed(hy, "gateway", "depends_on", "redis")
    # Same subject twice via a different predicate must not double-count.
    _seed(hy, "billing", "uses", "redis")
    # An unrelated object must not leak in.
    _seed(hy, "search", "depends_on", "elasticsearch")

    result = hy.count_relations(
        count="subject", predicates=["depends_on"], object="redis"
    )

    assert result.count == 3
    assert result.counted == "subject"
    assert set(result.entities) == {"billing", "auth", "gateway"}
    assert result.filters["object"] == "redis"
    assert result.filters["predicates"] == ["depends_on"]


def test_count_distinct_objects_of_a_type(hy):
    # "how many databases do we use?" -> distinct objects of type 'database'.
    _seed(hy, "app", "uses", "postgres")
    _seed(hy, "app", "uses", "redis")
    _seed(hy, "app", "uses", "react")  # not a database
    _type(hy, "postgres", "database")
    _type(hy, "redis", "database")
    _type(hy, "react", "framework")

    result = hy.count_relations(
        count="object", predicates=["uses"], object_type="database"
    )

    assert result.count == 2
    assert result.counted == "object"
    assert set(result.entities) == {"postgres", "redis"}


def test_object_with_multiple_type_rows_counted_once(hy):
    # An entity tagged with several types must still count once for its type.
    _seed(hy, "app", "uses", "postgres")
    _type(hy, "postgres", "database")
    _type(hy, "postgres", "service")

    result = hy.count_relations(count="object", object_type="database")
    assert result.count == 1
    assert result.entities == ["postgres"]


def test_alias_resolution_of_object_surface_form(hy):
    hy.register_alias("Postgres", "postgres")
    _seed(hy, "app", "depends_on", "postgres")

    # Surface form "Postgres" resolves to canonical before filtering.
    result = hy.count_relations(
        count="subject", predicates=["depends_on"], object="Postgres"
    )
    assert result.count == 1
    assert result.entities == ["app"]
    # Echoed filter carries the resolved canonical, not the surface form.
    assert result.filters["object"] == "postgres"


def test_excludes_stale_and_retracted_edges(hy):
    _seed(hy, "billing", "depends_on", "redis")  # active
    _seed(hy, "auth", "depends_on", "redis", status="stale")
    _seed(hy, "gateway", "depends_on", "redis", status="retracted")

    result = hy.count_relations(
        count="subject", predicates=["depends_on"], object="redis"
    )
    assert result.count == 1
    assert result.entities == ["billing"]


def test_predicate_agnostic_count(hy):
    # Omitting predicates counts across all predicates.
    _seed(hy, "billing", "depends_on", "redis")
    _seed(hy, "auth", "uses", "redis")
    _seed(hy, "billing", "connects_to", "redis")  # dup subject, different pred

    result = hy.count_relations(count="subject", object="redis")
    assert result.count == 2
    assert set(result.entities) == {"billing", "auth"}
    # No predicates filter echoed when none supplied.
    assert "predicates" not in result.filters


def test_subject_type_filter(hy):
    _seed(hy, "billing", "depends_on", "redis")
    _seed(hy, "cronjob", "depends_on", "redis")
    _type(hy, "billing", "service")
    # cronjob is not typed as a service, so it's excluded.

    result = hy.count_relations(
        count="subject", predicates=["depends_on"], object="redis",
        subject_type="service",
    )
    assert result.count == 1
    assert result.entities == ["billing"]


def test_empty_result_when_nothing_matches(hy):
    _seed(hy, "app", "uses", "postgres")

    result = hy.count_relations(
        count="subject", predicates=["depends_on"], object="nonexistent"
    )
    assert result.count == 0
    assert result.entities == []
    assert result.counted == "subject"


def test_explicit_empty_predicate_set_matches_nothing(hy):
    _seed(hy, "app", "uses", "postgres")

    result = hy.count_relations(count="object", predicates=[])
    assert result.count == 0
    assert result.entities == []


def test_predicate_normalized_before_match(hy):
    _seed(hy, "billing", "depends_on", "redis")

    # Caller passes mixed-case / padded predicate; it still matches stored form.
    result = hy.count_relations(
        count="subject", predicates=[" Depends_On "], object="redis"
    )
    assert result.count == 1
    assert result.entities == ["billing"]
