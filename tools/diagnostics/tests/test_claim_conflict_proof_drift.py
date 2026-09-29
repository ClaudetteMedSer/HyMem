"""Network-free controls for the diagnostic census, without changing its gate."""
import importlib.util
from pathlib import Path
import shutil
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_proof_drift as audit


def test_physical_census_excludes_temp_and_detects_main_data_changes():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO sample VALUES (1,'original')")
    before = audit.physical_rows(conn)
    conn.execute("CREATE TEMP TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO temp.sample VALUES (1,'private temporary')")
    assert audit.compare_rows(before, audit.physical_rows(conn))["ordered_equal"] is True
    conn.execute("UPDATE main.sample SET value='changed' WHERE id=1")
    compared = audit.compare_rows(before, audit.physical_rows(conn))
    assert compared["changed"] == 1
    assert compared["unordered_equal"] is False
    assert compared["changed_table_ids"] == [audit.table_id("sample")]
    conn.close()


def test_physical_census_includes_shadow_data_and_detects_its_mutation():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE documents USING fts5(content)")
    conn.execute("INSERT INTO documents VALUES ('original')")
    before = audit.physical_rows(conn)
    shadows = {row[1] for row in audit.table_inventory(conn) if row[2] == "shadow"}
    assert shadows
    assert shadows <= before.keys()
    conn.execute("INSERT INTO documents VALUES ('changed')")
    compared = audit.compare_rows(before, audit.physical_rows(conn))
    assert compared["changed"] > 0
    assert compared["ordered_equal"] is False
    assert compared["unordered_equal"] is False
    conn.close()


def test_only_actual_claim_proof_column_is_excluded_from_physical_rows():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE kg_claim_extraction_outcomes (id TEXT)")
    conn.execute("INSERT INTO kg_claim_extraction_outcomes VALUES ('one')")
    conn.execute("CREATE TABLE unrelated (local_replay_proof TEXT)")
    conn.execute("INSERT INTO unrelated VALUES ('before')")
    before = audit.physical_rows(conn)
    conn.execute("ALTER TABLE kg_claim_extraction_outcomes ADD COLUMN local_replay_proof TEXT")
    after = audit.physical_rows(conn)
    assert audit.compare_rows(before, after)["ordered_equal"] is True
    assert before["kg_claim_extraction_outcomes"]["column_names_sha256"] != after["kg_claim_extraction_outcomes"]["column_names_sha256"]
    conn.execute("UPDATE unrelated SET local_replay_proof='after'")
    assert audit.compare_rows(before, audit.physical_rows(conn))["unordered_equal"] is False
    conn.close()


def test_census_distinguishes_order_only_changes_from_row_content_changes():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sample (value TEXT)")
    conn.execute("INSERT INTO sample VALUES ('b'),('a')")
    before = audit.physical_rows(conn)
    conn.execute("DELETE FROM sample")
    conn.execute("INSERT INTO sample VALUES ('a'),('b')")
    compared = audit.compare_rows(before, audit.physical_rows(conn))
    assert compared["ordered_equal"] is False
    assert compared["unordered_equal"] is True
    conn.close()


def test_inventory_exports_only_explicitly_approved_table_names():
    before = [("main", "vec_chunks_rowids", "table", 4, 0, 0),
              ("temp", "sensitive arbitrary name", "table", 1, 0, 0)]
    after = [("main", "vec_chunks_rowids", "shadow", 4, 0, 0)]
    changes = audit.inventory_changes(before, after)
    assert changes[0]["table_id"] == "vec_chunks_rowids"
    assert changes[1]["table_id"].startswith("sha256:")
    assert "sensitive" not in str(changes)


@pytest.mark.parametrize("instrumented", [False, True])
def test_exact_arm_does_not_scan_rows_before_failed_call_sequence(monkeypatch, tmp_path, instrumented):
    from hymem.core import db
    calls = []
    source = tmp_path / "source.sqlite"
    conn = sqlite3.connect(source)
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY,value TEXT)")
    conn.execute("INSERT INTO schema_meta VALUES ('schema_version','63')")
    conn.execute("CREATE TABLE kg_claim_extraction_outcomes (id TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(audit, "WORK", tmp_path)
    monkeypatch.setattr(audit, "SNAPSHOT", source)
    original_rows = audit.physical_rows
    def physical_rows(conn):
        calls.append("rows")
        return original_rows(conn)
    def initialize(conn):
        calls.append("initialize")
        conn.execute("ALTER TABLE kg_claim_extraction_outcomes ADD COLUMN local_replay_proof TEXT")
        conn.execute("UPDATE schema_meta SET value='64'")
    def semantic(_conn):
        calls.append("digest")
        return "a" * 64
    monkeypatch.setattr(audit, "physical_rows", physical_rows)
    monkeypatch.setattr(db, "initialize", initialize)
    def open_clone(path):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn
    replay = SimpleNamespace(clone=shutil.copyfile, open_clone=open_clone, integrity=lambda _: {"clean": True})
    v1 = SimpleNamespace(semantic_digest=semantic, require_clean=lambda *_: None)
    metadata = {}
    result = audit.one_arm(v1, replay, instrumented=instrumented, metadata=metadata)
    assert result["schema_before"] == 63 and result["schema_after"] == 64
    assert result["proof_nonnull"] == 0
    assert calls == (["digest", "rows", "digest", "initialize", "digest", "digest", "rows"]
                     if instrumented else ["digest", "initialize", "digest", "digest", "rows"])
    assert len(metadata["instrumented" if instrumented else "exact"]["inventories"]) >= 7
    if instrumented:
        assert result["row_comparison"]["ordered_equal"] is True


def test_actual_vec_lazy_schema_classification_reproduces_without_row_changes(tmp_path):
    """Optional native local reproduction; no downloads or providers."""
    if importlib.util.find_spec("sqlite_vec") is None:
        pytest.skip("sqlite-vec not installed")
    import sqlite_vec
    candidates = ([Path(shutil.which("sqlite3"))] if shutil.which("sqlite3") else [])
    candidates.extend(Path("/opt/homebrew/Cellar/sqlite").glob("*/bin/sqlite3"))
    if not candidates:
        pytest.skip("extension-capable sqlite CLI not installed")
    source = tmp_path / "synthetic.sqlite"
    extension = sqlite_vec.loadable_path()
    create = (f'.load "{extension}"\n'
              "CREATE TABLE ordinary (id TEXT);\n"
              "CREATE VIRTUAL TABLE vec_probe USING vec0(id INTEGER PRIMARY KEY, embedding float[3]);\n"
              "INSERT INTO vec_probe(id,embedding) VALUES (1,'[1,2,3]');\n")
    executable = None
    for candidate in candidates:
        result = subprocess.run([str(candidate), str(source)], input=create, text=True, capture_output=True, timeout=10)
        if result.returncode == 0:
            executable = str(candidate)
            break
        if source.exists():
            source.unlink()
    if executable is None:
        pytest.skip("installed sqlite CLI cannot load local sqlite-vec binary")
    script = ("SELECT count(*) FROM sqlite_master;\n"
              f'.load "{extension}"\n'
              "SELECT 'before',name,type FROM pragma_table_list WHERE name IN ('vec_probe_info','vec_probe_rowids','vec_probe_chunks') ORDER BY name;\n"
              "PRAGMA integrity_check;\n"
              "SELECT 'row-before',quote(rowid),hex(vectors) FROM vec_probe_vector_chunks00;\n"
              "ALTER TABLE ordinary ADD COLUMN nullable TEXT;\n"
              "SELECT 'after',name,type FROM pragma_table_list WHERE name IN ('vec_probe_info','vec_probe_rowids','vec_probe_chunks') ORDER BY name;\n"
              "SELECT 'row-after',quote(rowid),hex(vectors) FROM vec_probe_vector_chunks00;\n")
    result = subprocess.run([executable, str(source)], input=script, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    before = [line.removeprefix("before|") for line in lines if line.startswith("before|")]
    after = [line.removeprefix("after|") for line in lines if line.startswith("after|")]
    assert len(before) == len(after) == 3
    assert all(line.endswith("|table") for line in before)
    assert all(line.endswith("|shadow") for line in after)
    assert [line.removeprefix("row-before|") for line in lines if line.startswith("row-before|")] == [
        line.removeprefix("row-after|") for line in lines if line.startswith("row-after|")]
