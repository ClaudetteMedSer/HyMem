"""Independent root controls: SQLite authority and frozen source identity."""
import ast
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_instrumented_held_census_v1 as census


def authority():
    scope = {}
    exec(census.PROJECTION, scope)
    return scope["_authorize"]


@pytest.mark.parametrize("statement", [
    "UPDATE chunks SET id='changed'", "DELETE FROM chunks",
    "INSERT INTO chunks VALUES ('changed')", "DROP TABLE chunks",
    "CREATE TABLE added(x)", "ATTACH DATABASE ':memory:' AS extra",
    "PRAGMA user_version=3", "SELECT * FROM forbidden",
    "SELECT randomblob(10)",
])
def test_real_sqlite_authorizer_denies_unrequested_work(statement):
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript("CREATE TABLE chunks(id TEXT); INSERT INTO chunks VALUES ('original');"
                           "CREATE TABLE forbidden(text TEXT);")
        conn.set_authorizer(authority())
        assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (1,)
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(statement).fetchall()
        assert conn.execute("SELECT id FROM chunks").fetchone() == ("original",)
    finally:
        conn.close()


def test_source_and_helper_pins_remain_exact():
    for path, expected in ((Path(census.v1.__file__), census.V1_SHA),
                           (Path(census.v2.__file__), census.V2_SHA),
                           (Path("benchmarks/lme_diagnostic.py"), census.HELPER_SHA)):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_exact_frozen_retry_and_db_path_contract():
    path = Path("/private/tmp/hymem-lme-instrumented-IjmZdT/bundle/candidate/hymem/config.py")
    tree = ast.parse(path.read_text())
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "HyMemConfig")
    attempt = next(node for node in config.body if isinstance(node, ast.AnnAssign)
                   and isinstance(node.target, ast.Name)
                   and node.target.id == "chunk_extraction_max_attempts")
    assert ast.literal_eval(attempt.value) == census.RETRY_BOUND == 3
    db = next(node for node in config.body if isinstance(node, ast.FunctionDef)
              and node.name == "db_path")
    assert 'self.root / "hymem.sqlite"' in ast.get_source_segment(path.read_text(), db)


def test_closed_reasons_match_frozen_producer_ast():
    path = Path("/private/tmp/hymem-lme-instrumented-IjmZdT/bundle/candidate/hymem/extraction/chunk.py")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_FAILURE_REASONS"
                        for t in n.targets))
    assert set(census.REASON_CODES) == ast.literal_eval(node.value.args[0]) | {"other"}


@pytest.mark.parametrize("payload", [
    {"private": "PRIVATE-SENTINEL"},
    {"schema": census.SCHEMA, "reason_counts": {"PRIVATE-SENTINEL": 79}},
    {"schema": census.SCHEMA, "status": "PRIVATE-SENTINEL"},
    None, [], "PRIVATE-SENTINEL",
])
def test_main_never_exports_unvalidated_stdout(monkeypatch, capsys, payload):
    monkeypatch.setattr(census.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="PRIVATE-STDERR"))
    assert census.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema": census.SCHEMA, "status": "metadata_unavailable"}


def test_remote_source_gate_precedes_private_inspection():
    scope = {"SOURCE": "from pathlib import Path\ndef inspect(*args):\n    raise ValueError('rejected')",
             "ROOT": census.v1.ROOT, "RECEIPT_SHA": census.v1.RECEIPT_SHA,
             "V1_PROJECTION": "raise AssertionError('projection reached')"}
    with pytest.raises(ValueError, match="rejected"):
        exec(census.REMOTE, scope)
