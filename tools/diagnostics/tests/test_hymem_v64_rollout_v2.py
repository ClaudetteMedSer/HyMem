"""Local real sqlite-vec regression; no Docker or production operations."""
import json
from pathlib import Path
import sqlite3
import struct
import sys
from types import ModuleType

import pytest
from hymem.core import db as candidate_db

from tools.diagnostics import hymem_v64_rollout_v2 as adapter


def fixture(tmp_path, monkeypatch, defect=None):
    path = tmp_path / "vec.sqlite"
    c = sqlite3.connect(path)
    assert candidate_db._load_vec_extension(c), "real sqlite-vec required"
    c.executescript("CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT);"
                    "INSERT INTO schema_meta VALUES('schema_version','63');"
                    "CREATE TABLE kg_claim_extraction_outcomes(id INTEGER PRIMARY KEY,value TEXT);"
                    "INSERT INTO kg_claim_extraction_outcomes VALUES(1,'durable');"
                    "CREATE VIRTUAL TABLE vectors USING vec0(embedding float[2]);")
    c.execute("INSERT INTO vectors(rowid,embedding) VALUES(?,?)", (7, struct.pack("ff", 1, 2)))
    c.commit()
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name!='schema_meta'")}
    assert "vectors" in tables and any(t.startswith("vectors_") for t in tables)
    c.close()
    fake = ModuleType("hymem.core.db")
    fake.connect = sqlite3.connect
    fake.schema_version = lambda c: int(c.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0])
    calls = []

    def load(c):
        calls.append(c)
        return False if defect == "missing" or (defect == "missing_reopen" and len(calls) == 2) else candidate_db._load_vec_extension(c)

    fake._load_vec_extension = load

    def initialize(c):
        # Model initialize's maintained loader, including on the original path.
        assert candidate_db._load_vec_extension(c)
        if fake.schema_version(c) == 63:
            c.execute("ALTER TABLE kg_claim_extraction_outcomes ADD COLUMN local_replay_proof TEXT")
            c.execute("UPDATE schema_meta SET value='64' WHERE key='schema_version'")
            if defect == "vector":
                c.execute("UPDATE vectors SET embedding=? WHERE rowid=7", (struct.pack("ff", 3, 4),))
            c.commit()
        elif defect == "reopen":
            c.execute("UPDATE vectors SET embedding=? WHERE rowid=7", (struct.pack("ff", 5, 6),))
            c.commit()

    fake.initialize = initialize
    package = ModuleType("hymem.core")
    package.db = fake
    monkeypatch.setitem(sys.modules, "hymem.core", package)
    return path, tables, calls


def execute(path, script):
    namespace = {}
    exec(compile("DBPATH=" + repr(str(path)) + "\n" + script, "<migration>", "exec"), namespace)
    return namespace


def test_original_reproduces_vec_module_failure(tmp_path, monkeypatch):
    path, _, calls = fixture(tmp_path, monkeypatch)
    original = Path(adapter.__file__).with_name("hymem_v64_rollout.py")
    namespace = {"__name__": "original"}
    exec(compile(original.read_bytes(), str(original), "exec"), namespace)
    with pytest.raises(sqlite3.OperationalError, match="no such module: vec0"):
        execute(path, namespace["MIGRATION"])
    assert calls == []


def test_real_vec_virtual_and_shadow_rows_preserved_and_reopened(tmp_path, monkeypatch, capsys):
    path, tables, calls = fixture(tmp_path, monkeypatch)
    result = execute(path, adapter.MIGRATION)
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["preserved_tables"] == len(tables)
    assert set(result["old"]) == tables
    assert result["old"] == result["new"]
    assert result["all_after"] == result["reopened"]
    assert result["old"]["vectors"]["count"] == 1
    assert any(result["old"][t]["count"] > 0 for t in tables if t.startswith("vectors_"))
    assert len(calls) == 2


@pytest.mark.parametrize("defect,error", [("missing", "vec_extension_unavailable"),
                                         ("missing_reopen", "vec_extension_unavailable"),
                                         ("vector", "durable_rows_changed"),
                                         ("reopen", "reopen_rows_changed")])
def test_vec_failures_fail_closed(tmp_path, monkeypatch, defect, error):
    path, _, _ = fixture(tmp_path, monkeypatch, defect)
    with pytest.raises(RuntimeError, match=error):
        execute(path, adapter.MIGRATION)
    if defect == "missing":
        c = sqlite3.connect(path)
        assert c.execute("SELECT value FROM schema_meta").fetchone()[0] == "63"
        c.close()


def test_original_seal_and_exact_two_line_change(tmp_path):
    path = Path(adapter.__file__).with_name("hymem_v64_rollout.py")
    namespace = {"__name__": "original"}
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    assert adapter.MIGRATION.replace("    need(db._load_vec_extension(c),'vec_extension_unavailable')\n", "") == namespace["MIGRATION"]
    drift = tmp_path / "drift.py"
    drift.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="original_helper_pin_mismatch"):
        adapter.load_original(drift)
    assert adapter.original.Rollout.offline.__globals__["MIGRATION"] == adapter.MIGRATION
