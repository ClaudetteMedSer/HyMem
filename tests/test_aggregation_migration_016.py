"""The shipped 016 CREATE + real ALTER lineage, not schema.sql lookalikes."""

from __future__ import annotations

from contextlib import closing
from importlib.resources import files
import json
import re
import sqlite3

import pytest

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming.aggregation_provenance import (
    load_aggregation_node_proof,
    load_current_aggregation_publication,
)
from hymem.dreaming.aggregation_generation import aggregation_generation_binding
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import StubLLMClient
from tests.test_aggregation_provenance_v55 import _seed_tree


_HEADER_NAMES = (
    "aggregation_source_header_insert_guard",
    "aggregation_source_header_update_guard",
)


def _migration_sql(name):
    return files("hymem.core.migrations").joinpath(name).read_text(encoding="utf-8")


def _header_sql(name):
    return next(
        statement for statement in core_db._split_sql_statements(
            _migration_sql("055_aggregation_typed_provenance.sql")
        )
        if statement.startswith(f"CREATE TRIGGER {name}\n")
    )


def _install_original_v55_headers(conn):
    """Precisely undo this repair's three changes to the shipped v55 guards."""
    for name in _HEADER_NAMES:
        sql = _header_sql(name)
        sql = sql.replace("WHEN (\n", "WHEN NOT (\n", 1)
        sql = sql.replace(") IS NOT 1 BEGIN", ") BEGIN", 1)
        if name.endswith("update_guard"):
            sql = sql.replace(
                "input_manifest_hash, input_manifest_complete, input_fingerprint",
                "input_manifest_hash, input_manifest_complete",
                1,
            )
            sql = sql.replace(
                "AND new.input_manifest_version IS NULL\n"
                "     AND (new.input_fingerprint IS NULL OR (\n"
                "          length(new.input_fingerprint) = 71\n"
                "          AND new.input_fingerprint GLOB 'sha256:*')))",
                "AND new.input_manifest_version IS NULL)",
                1,
            )
            assert "input_fingerprint" not in sql.split("ON aggregation_nodes")[0]
            assert "input_fingerprint" not in sql.split("    OR\n")[0]
        assert "WHEN NOT (" in sql and "IS NOT 1 BEGIN" not in sql
        conn.execute(f"DROP TRIGGER {name}")
        conn.execute(sql)


def _historical_v46(path, monkeypatch, *, weaken_check=False):
    """Bootstrap other domains normally, but create this table from 016.

    Limit the public runner to v46 so 017/045 really ALTER that old table.
    The other domains are current bootstrap fixtures, not an archived whole
    v46 database; no source or shape for aggregation_nodes comes from schema.sql.
    """
    conn = core_db.connect(path)
    original = next(
        statement for statement in core_db._split_sql_statements(
            _migration_sql("016_aggregation_nodes.sql")
        )
        if statement.startswith("CREATE TABLE IF NOT EXISTS aggregation_nodes (")
    )
    conn.execute(original)
    entries = [(v, p) for v, p in core_db._discover_migrations() if v <= 46]
    with monkeypatch.context() as scoped:
        scoped.setattr(core_db, "_discover_migrations", lambda: entries)
        if weaken_check:
            # Same PRAGMA columns, but not any shipped migration lineage.
            apply = core_db._apply_migration_sql

            def weakened(conn, script):
                return apply(conn, script.replace(
                    "ALTER TABLE aggregation_nodes ADD COLUMN source_manifest_count "
                    "INTEGER NOT NULL DEFAULT 0\n    CHECK (source_manifest_count >= 0)",
                    "ALTER TABLE aggregation_nodes ADD COLUMN source_manifest_count "
                    "INTEGER NOT NULL DEFAULT 0",
                ))

            scoped.setattr(core_db, "_apply_migration_sql", weakened)
        core_db.initialize(conn)
    assert core_db.schema_version(conn) == 46
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='aggregation_nodes'"
    ).fetchone()[0]
    assert "CHECK (\n        (source_manifest_complete" not in sql
    assert ("CHECK (source_manifest_count >= 0)" in sql) != weaken_check
    return conn


def _seed_legacy_node(conn):
    with core_db.transaction(conn):
        conn.execute("INSERT INTO sessions(id) VALUES ('legacy-source')")
        conn.execute(
            "INSERT INTO messages(session_id,role,content) "
            "VALUES ('legacy-source','user','Exact historical source')"
        )
        materialize_message_coverage(conn, "legacy-source")
        conn.execute(
            "INSERT INTO aggregation_nodes(rowid,id,title,summary) "
            "VALUES (77,'legacy-016','Legacy Redis Cluster','Historical bytes')"
        )
        conn.execute(
            "INSERT INTO aggregation_node_source_occurrences "
            "SELECT 'legacy-016',0,message_id,source_session_id,source_role,"
            "source_peer_id,source_workspace_id,source_created_at,chunk_id,"
            "coverage_version,message_content_hash "
            "FROM message_retention_coverage WHERE source_session_id='legacy-source'"
        )
        conn.execute(
            "INSERT INTO aggregation_node_embeddings"
            "(node_id,vector_json,model,dim,text_hash) "
            "VALUES ('legacy-016','[1.0,0.0]','legacy-local',2,'legacy-hash')"
        )
    return tuple(conn.execute(
        "SELECT * FROM aggregation_node_source_occurrences WHERE node_id='legacy-016'"
    ).fetchone())


def _assert_legacy_preserved(conn, source):
    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert core_db._v57_material_bindings_present(conn)
    row = conn.execute(
        "SELECT rowid,* FROM aggregation_nodes WHERE id='legacy-016'"
    ).fetchone()
    assert row["rowid"] == 77 and row["title"] == "Legacy Redis Cluster"
    assert row["summary"] == "Historical bytes"
    assert row["source_manifest_complete"] == row["input_manifest_complete"] == 0
    assert row["aggregation_generation_key"] is None
    assert load_current_aggregation_publication(conn) is None
    assert tuple(conn.execute(
        "SELECT * FROM aggregation_node_source_occurrences WHERE node_id='legacy-016'"
    ).fetchone()) == source
    assert conn.execute(
        "SELECT rowid FROM aggregation_nodes_fts WHERE aggregation_nodes_fts MATCH 'Redis'"
    ).fetchone()[0] == 77
    # Existing v57 policy deliberately removes unbound pre-v57 vector caches.
    assert conn.execute("SELECT count(*) FROM aggregation_node_embeddings").fetchone()[0] == 0
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    return tuple(row)


def _interrupt_migration(conn, monkeypatch, version):
    apply = core_db._apply_migration_sql
    target = _migration_sql({
        56: "056_aggregation_generation_identity.sql",
        57: "057_aggregation_material_epoch.sql",
    }[version])

    def interrupted(conn, script):
        apply(conn, script)
        if script == target:
            raise RuntimeError("injected migration interruption")

    with monkeypatch.context() as scoped:
        scoped.setattr(core_db, "_apply_migration_sql", interrupted)
        with pytest.raises(RuntimeError, match="injected migration interruption"):
            core_db.initialize(conn)
    assert not conn.in_transaction
    assert core_db.schema_version(conn) == version - 1


@pytest.mark.parametrize("interrupt_at", [None, 56, 57])
def test_original_016_public_upgrade_and_interrupted_retry_preserve_rows(
    tmp_path, monkeypatch, interrupt_at,
):
    path = tmp_path / "historical.sqlite"
    conn = _historical_v46(path, monkeypatch)
    source = _seed_legacy_node(conn)
    if interrupt_at is not None:
        _interrupt_migration(conn, monkeypatch, interrupt_at)
        if interrupt_at == 56:
            assert core_db._v56_domain_footprint_present(conn)
            assert "aggregation_generation_key" not in {
                r[1] for r in conn.execute("PRAGMA table_info(aggregation_nodes)")
            }
        _install_original_v55_headers(conn)
    conn.close()
    conn = core_db.connect(path)
    try:
        core_db.initialize(conn)
        before = _assert_legacy_preserved(conn, source)
        core_db.initialize(conn)
        assert _assert_legacy_preserved(conn, source) == before
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM message_retention_coverage")
        with pytest.raises(sqlite3.IntegrityError, match="invalid aggregation proof"):
            conn.execute(
                "UPDATE aggregation_nodes SET input_fingerprint='malformed' "
                "WHERE id='legacy-016'"
            )
    finally:
        conn.close()


@pytest.mark.parametrize("damage", [
    "nonempty_registry", "nonempty_publication", "marker",
    "registry_check", "publication_check", "registry_column", "publication_column",
])
def test_mixed_v55_bootstrap_is_empty_exact_storage_only(cfg, monkeypatch, damage):
    conn = _historical_v46(cfg.db_path, monkeypatch)
    try:
        _interrupt_migration(conn, monkeypatch, 56)
        assert core_db._v55_aggregation_storage_present(conn, allow_v56_bootstrap=True)
        assert not core_db._v55_aggregation_bindings_present(conn)
        if damage == "marker":
            conn.execute(
                "INSERT INTO schema_meta(key,value) VALUES ('aggregation_generation_schema','56')"
            )
        elif damage == "nonempty_registry":
            # Even an internally valid producer declaration is not an empty
            # bootstrap. Admission cannot manufacture a generation authority.
            binding = aggregation_generation_binding(cfg, StubLLMClient(default="{}"))
            producer = binding["producer"]
            conn.execute(
                "INSERT INTO aggregation_generations(generation_key,material_config_version,"
                "producer_identity_sha256,identity_exact,reuse_scope,binding_json) "
                "VALUES (?,?,?,?,?,?)",
                (binding["generation_key"], binding["contract"]["material_config_version"],
                 producer["identity_sha256"], int(producer["identity_exact"]),
                 producer["reuse_scope"], json.dumps(binding, sort_keys=True, separators=(",", ":"))),
            )
        elif damage == "nonempty_publication":
            conn.execute(
                "INSERT INTO aggregation_publication_state(id,publication_id,config_version,"
                "cluster_min_members,cluster_min_sessions,anchor_fact_cap,node_count,node_set_hash) "
                "VALUES (1,?,?,1,1,0,0,?)",
                ("sha256:" + "a" * 64, "aggregation-build-config-v1:" + "b" * 64,
                 "sha256:" + "c" * 64),
            )
        else:
            table = "aggregation_generations" if damage.startswith("registry") else "aggregation_publication_state"
            if damage.endswith("column"):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN unexpected TEXT")
            else:
                sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name=?", (table,),
                ).fetchone()[0]
                target = (
                    "CHECK (identity_exact IN (0,1))" if damage.startswith("registry")
                    else "CHECK (cluster_min_members >= 1)"
                )
                assert target in sql
                conn.execute(f"DROP TABLE {table}")
                conn.execute(sql.replace(target, ""))
        assert not core_db._v55_aggregation_storage_present(conn, allow_v56_bootstrap=True)
        with pytest.raises(RuntimeError, match="v55 aggregation provenance domain is incomplete"):
            core_db.initialize(conn)
        assert core_db.schema_version(conn) == 55
    finally:
        conn.close()


def test_original_lineage_does_not_accept_missing_column_check(tmp_path, monkeypatch):
    conn = _historical_v46(tmp_path / "malformed.sqlite", monkeypatch, weaken_check=True)
    try:
        with pytest.raises(RuntimeError, match="aggregation.*malformed"):
            core_db.initialize(conn)
        assert core_db.schema_version(conn) == 55
        with pytest.raises(RuntimeError, match="aggregation.*incomplete"):
            core_db.initialize(conn)
        assert core_db.schema_version(conn) == 55
    finally:
        conn.close()


@pytest.mark.parametrize("damage", ["missing", "forged", "original"])
def test_historical_current_store_requires_and_heals_exact_guards(
    cfg, monkeypatch, damage,
):
    path = cfg.db_path
    conn = _historical_v46(path, monkeypatch)
    core_db.initialize(conn)
    conn.close()
    hy = HyMem(cfg)
    _enabled, publication = _seed_tree(hy, cfg)
    conn = hy.conn
    assert load_current_aggregation_publication(conn) is not None
    if damage == "original":
        _install_original_v55_headers(conn)
    else:
        for name in _HEADER_NAMES:
            conn.execute(f"DROP TRIGGER {name}")
            if damage == "forged":
                conn.execute(
                    f"CREATE TRIGGER {name} AFTER INSERT ON aggregation_nodes "
                    "BEGIN SELECT 1; END"
                )
    assert core_db._v57_material_bindings_present(conn, validate_triggers=False)
    assert not core_db._v57_material_bindings_present(conn)
    # The reader independently checks complete row proofs, not installed DDL.
    # Unchanged valid contents remain readable; a damaged schema is never an
    # excuse to accept corrupt contents. Also remove the payload guard only
    # within this savepoint to exercise that independent reader boundary.
    assert load_current_aggregation_publication(conn) is not None
    node_id = next(iter(publication.nodes))
    conn.execute("SAVEPOINT corrupt_proof")
    try:
        conn.execute("DROP TRIGGER aggregation_source_bound_update_guard")
        conn.execute(
            "UPDATE aggregation_nodes SET input_fingerprint=NULL WHERE id=?", (node_id,),
        )
        assert conn.execute("SELECT count(*) FROM aggregation_publication_state").fetchone()[0] == 1
        assert load_aggregation_node_proof(conn, node_id) is None
        assert load_current_aggregation_publication(conn) is None
    finally:
        conn.execute("ROLLBACK TO corrupt_proof")
        conn.execute("RELEASE corrupt_proof")
    assert load_current_aggregation_publication(conn) is not None
    hy.close()
    conn = core_db.connect(path)
    try:
        core_db.initialize(conn)
        assert core_db._v57_material_bindings_present(conn)
        assert load_current_aggregation_publication(conn) is not None
        for name in _HEADER_NAMES:
            actual = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (name,)
            ).fetchone()[0]
            assert re.sub(r"\s+", " ", actual).rstrip(";") == re.sub(
                r"\s+", " ", _header_sql(name)
            ).rstrip(";")
    finally:
        conn.close()


def test_aggregation_guard_refresh_does_not_commit_callers_transaction(tmp_path):
    conn = core_db.connect(tmp_path / "atomic.sqlite")
    try:
        core_db.initialize(conn)
        _install_original_v55_headers(conn)
        before = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (_HEADER_NAMES[1],)
        ).fetchone()[0]
        with pytest.raises(RuntimeError, match="rollback"):
            with core_db.transaction(conn):
                core_db._install_aggregation_source_guards(conn)
                assert conn.in_transaction
                conn.execute("INSERT INTO sessions(id) VALUES ('uncommitted')")
                raise RuntimeError("rollback")
        assert conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (_HEADER_NAMES[1],)
        ).fetchone()[0] == before
        assert conn.execute("SELECT 1 FROM sessions WHERE id='uncommitted'").fetchone() is None
    finally:
        conn.close()


@pytest.mark.parametrize("historical", [False, True])
def test_header_fingerprint_and_null_publication_controls(cfg, monkeypatch, historical):
    if historical:
        conn = _historical_v46(cfg.db_path, monkeypatch)
        core_db.initialize(conn)
        conn.close()
    with closing(HyMem(cfg)) as hy:
        hy.conn.execute(
            "INSERT INTO aggregation_nodes(id,title,summary) VALUES ('empty','Empty','Empty')"
        )
        for valid in (None, "sha256:" + "a" * 64):
            hy.conn.execute(
                "UPDATE aggregation_nodes SET input_fingerprint=? WHERE id='empty'", (valid,)
            )
        for malformed in ("", "malformed", "sha256:" + "a" * 63):
            with pytest.raises(sqlite3.IntegrityError):
                hy.conn.execute(
                    "UPDATE aggregation_nodes SET input_fingerprint=? WHERE id='empty'",
                    (malformed,),
                )
            with pytest.raises(sqlite3.IntegrityError):
                hy.conn.execute(
                    "INSERT INTO aggregation_nodes(id,title,summary,input_fingerprint) "
                    "VALUES ('invalid','Invalid','Invalid',?)", (malformed,),
                )
        _enabled, publication = _seed_tree(hy, cfg)
        node = next(iter(publication.nodes.values())).row
        fields = (
            "source_manifest_version", "source_manifest_count", "source_manifest_hash",
            "source_manifest_complete", "input_manifest_version", "input_manifest_count",
            "input_manifest_hash", "input_manifest_complete", "input_fingerprint",
            "node_kind", "output_hash", "publication_id", "build_config_version",
        )
        for missing in fields:
            hy.conn.execute("SAVEPOINT null_header")
            try:
                hy.conn.execute(
                    "UPDATE aggregation_nodes SET source_manifest_version=NULL,"
                    "source_manifest_count=0,source_manifest_hash=NULL,source_manifest_complete=0,"
                    "input_manifest_version=NULL,input_manifest_count=0,"
                    "input_manifest_hash=NULL,input_manifest_complete=0 WHERE id=?", (node["id"],),
                )
                params = [None if field == missing else node[field] for field in fields]
                with pytest.raises(sqlite3.IntegrityError):
                    hy.conn.execute(
                        "UPDATE aggregation_nodes SET " + ",".join(f"{f}=?" for f in fields)
                        + " WHERE id=?", (*params, node["id"]),
                    )
            finally:
                hy.conn.execute("ROLLBACK TO null_header")
                hy.conn.execute("RELEASE null_header")
        assert load_current_aggregation_publication(hy.conn) is not None
