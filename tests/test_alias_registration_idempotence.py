"""An existing alias remains usable when old canonical rows still name its key."""

from __future__ import annotations

import sqlite3

import pytest

from hymem.dreaming import canonicalize, phase1
from hymem.extraction.triples import Triple
from tests.test_bitemporal import _seed_chunk, _seed_message, _seed_session


SCALAR_OWNERS = (
    ("entity_aliases", "canonical"),
    ("entity_types", "entity_canonical"),
    ("entity_properties", "entity_canonical"),
    ("entity_type_observations", "entity_canonical"),
    ("entity_property_observations", "entity_canonical"),
    ("entity_mention_observations", "entity_canonical"),
    ("entity_mentions", "entity_canonical"),
    ("knowledge_graph", "subject_canonical"),
    ("knowledge_graph", "object_canonical"),
)


def _minimal_store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE entity_aliases(alias TEXT PRIMARY KEY, canonical TEXT)")
    conn.execute(
        "CREATE TABLE knowledge_graph("
        "subject_canonical TEXT, object_canonical TEXT)"
    )
    for table, column in SCALAR_OWNERS:
        if table in {"entity_aliases", "knowledge_graph"}:
            continue
        conn.execute(f"CREATE TABLE {table}({column} TEXT)")
    conn.execute("CREATE TABLE rules(scope TEXT, trigger_entities TEXT)")
    return conn


@pytest.mark.parametrize("table,column", SCALAR_OWNERS + (("rules", "trigger_entities"),))
def test_exact_registration_preserves_historical_owner_rows(table: str, column: str):
    conn = _minimal_store()
    try:
        conn.execute(
            "INSERT INTO entity_aliases(alias, canonical) VALUES (?, ?)",
            ("redis_cache", "redis"),
        )
        if table == "entity_aliases":
            conn.execute(
                "INSERT INTO entity_aliases(alias, canonical) VALUES (?, ?)",
                ("legacy_name", "redis_cache"),
            )
        elif table == "rules":
            conn.execute(
                "INSERT INTO rules(scope, trigger_entities) VALUES (?, ?)",
                ("contextual", '["redis_cache"]'),
            )
        else:
            conn.execute(f"INSERT INTO {table}({column}) VALUES (?)", ("redis_cache",))

        before = "\n".join(conn.iterdump())
        assert canonicalize.resolve(conn, "Redis Cache") == "redis"
        canonicalize.register_alias(conn, "Redis Cache", "redis")
        assert "\n".join(conn.iterdump()) == before
        assert canonicalize.resolve(conn, "Redis Cache") == "redis"
    finally:
        conn.close()


def test_exact_registration_still_validates_target_and_new_writes():
    conn = _minimal_store()
    try:
        conn.execute(
            "INSERT INTO entity_aliases(alias, canonical) VALUES (?, ?)",
            ("redis_cache", "redis"),
        )
        conn.execute(
            "INSERT INTO entity_types(entity_canonical) VALUES (?)", ("redis_cache",),
        )
        conn.execute(
            "INSERT INTO entity_types(entity_canonical) VALUES (?)", ("legacy_owner",),
        )
        before = "\n".join(conn.iterdump())
        for surface, target, message in (
            ("Redis Cache", "other", "already maps"),
            ("Redis Cache", "Not Normal", "normalized canonical"),
            ("Another Cache", "redis_cache", "must not itself be an alias"),
        ):
            with pytest.raises(ValueError, match=message):
                canonicalize.register_alias(conn, surface, target)
        assert "\n".join(conn.iterdump()) == before

        # A new mapping of a historical owner is still rejected.
        with pytest.raises(ValueError, match="already owns state"):
            canonicalize.register_alias(conn, "Legacy Owner", "redis")
    finally:
        conn.close()


def test_phase1_triple_persists_through_existing_alias_with_old_owner(hy):
    conn = hy.conn
    _seed_session(conn)
    _seed_message(conn, 10, "2024-03-15 09:00:00")
    _seed_chunk(conn, "c1", 10)
    hy.set_entity_type("Redis Cache", "database")
    conn.execute(
        "INSERT INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        ("redis_cache", "redis"),
    )
    before_type = conn.execute(
        "SELECT * FROM entity_types WHERE entity_canonical=?", ("redis_cache",)
    ).fetchall()

    phase1._upsert_triple(conn, "c1", Triple("app", "uses", "Redis Cache", 1))

    assert conn.execute(
        "SELECT COUNT(*) FROM knowledge_graph WHERE "
        "subject_canonical='app' AND predicate='uses' AND object_canonical='redis'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT * FROM entity_types WHERE entity_canonical=?", ("redis_cache",)
    ).fetchall() == before_type
