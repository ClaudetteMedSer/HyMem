"""Independent offline controls for the stable physical-main replay audit."""
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_proof_drift as census
from tools.diagnostics import claim_conflict_proof_replay as old
from tools.diagnostics import claim_conflict_proof_replay_v3 as worker


def fixture_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    for name in worker.REQUIRED_TABLES:
        conn.execute('CREATE TABLE "' + name + '" (id INTEGER PRIMARY KEY, value TEXT)')
    conn.execute("INSERT INTO sessions VALUES (1,'original')")
    return conn


def digest(conn):
    return worker.stable_semantic_digest(conn, census)


def test_required_domain_matches_sealed_original_audit():
    assert worker.REQUIRED_TABLES == frozenset(old.SEMANTIC_TABLES)


@pytest.mark.parametrize("table", sorted(worker.REQUIRED_TABLES))
def test_missing_required_table_fails_closed(table):
    conn = fixture_conn()
    conn.execute('DROP TABLE "' + table + '"')
    with pytest.raises(RuntimeError, match="semantic_audit_table_missing"):
        digest(conn)
    conn.close()


def test_table_or_shadow_reclassification_cannot_change_physical_digest(monkeypatch):
    conn = fixture_conn()
    conn.execute("CREATE TABLE vec_chunks_info (key TEXT, value TEXT)")
    conn.execute("INSERT INTO vec_chunks_info VALUES ('metadata','unchanged')")
    before = digest(conn)
    original = census.table_inventory
    monkeypatch.setattr(census, "table_inventory", lambda db: [
        (schema, name, "shadow" if name == "vec_chunks_info" else kind, ncol, wr, strict)
        for schema, name, kind, ncol, wr, strict in original(db)])
    assert digest(conn) == before
    conn.execute("UPDATE vec_chunks_info SET value='changed'")
    assert digest(conn) != before
    conn.close()


def test_regular_data_and_fts_shadow_data_mutations_change_digest():
    conn = fixture_conn()
    conn.execute("CREATE VIRTUAL TABLE documents USING fts5(content)")
    conn.execute("INSERT INTO documents VALUES ('original')")
    initial = digest(conn)
    conn.execute("UPDATE sessions SET value='changed' WHERE id=1")
    ordinary_changed = digest(conn)
    assert ordinary_changed != initial
    conn.execute("INSERT INTO documents VALUES ('new document')")
    assert digest(conn) != ordinary_changed
    conn.close()


def test_temp_objects_cannot_hide_main_rows_or_change_digest():
    conn = fixture_conn()
    initial = digest(conn)
    conn.execute("CREATE TEMP TABLE sessions (private_content TEXT)")
    conn.execute("INSERT INTO temp.sessions VALUES ('not persisted main data')")
    assert digest(conn) == initial
    conn.execute("UPDATE main.sessions SET value='changed' WHERE id=1")
    assert digest(conn) != initial
    conn.close()


def test_only_schema_meta_and_specific_outcome_proof_are_excluded():
    conn = fixture_conn()
    conn.execute("CREATE TABLE schema_meta (key TEXT,value TEXT)")
    conn.execute("INSERT INTO schema_meta VALUES ('schema_version','63')")
    initial = digest(conn)
    conn.execute("ALTER TABLE kg_claim_extraction_outcomes ADD COLUMN local_replay_proof TEXT")
    conn.execute("UPDATE schema_meta SET value='64'")
    assert digest(conn) == initial
    conn.execute("ALTER TABLE messages ADD COLUMN local_replay_proof TEXT")
    assert digest(conn) != initial
    conn.close()


def test_adapter_changes_only_the_loaded_diagnostic_digest(monkeypatch):
    sentinel = object()
    v1 = SimpleNamespace(SEMANTIC_TABLES=old.SEMANTIC_TABLES,
                         semantic_digest=lambda _: "old", one_arm=sentinel)
    v2 = SimpleNamespace(load_v1=lambda: v1, one_arm=sentinel, main=sentinel)
    monkeypatch.setattr(worker, "load_pinned", lambda path, *_: v2 if path == worker.V2_WORKER else census)
    configured, _ = worker.configured_v2()
    conn = fixture_conn()
    assert configured is v2
    assert configured.one_arm is sentinel and configured.main is sentinel
    assert configured.load_v1().one_arm is sentinel
    assert v1.semantic_digest(conn) == digest(conn)
    conn.close()


def test_exact_native_lazy_shadow_transition_keeps_stable_digest(tmp_path):
    """Exercise the exact captured failure mechanism with installed native vec0."""
    python = Path("/opt/anaconda3/bin/python")
    if not python.is_file():
        pytest.skip("local extension-capable Python unavailable")
    script = r'''
import json, pathlib, sqlite3, sys
sys.path.insert(0, sys.argv[1])
import sqlite_vec
from tools.diagnostics import claim_conflict_proof_drift as census
from tools.diagnostics import claim_conflict_proof_replay as old
from tools.diagnostics import claim_conflict_proof_replay_v3 as worker
source = pathlib.Path(sys.argv[2])
c = sqlite3.connect(source)
c.enable_load_extension(True); sqlite_vec.load(c)
for name in worker.REQUIRED_TABLES:
    c.execute('CREATE TABLE "'+name+'" (id INTEGER PRIMARY KEY, value TEXT)')
c.execute("INSERT INTO sessions VALUES (1,'unchanged')")
c.execute('CREATE VIRTUAL TABLE vec_probe USING vec0(id INTEGER PRIMARY KEY, embedding float[3])')
c.execute("INSERT INTO vec_probe(id,embedding) VALUES(1,'[1,2,3]')")
c.commit();c.close()
c = sqlite3.connect(source)
c.row_factory = sqlite3.Row
c.execute('SELECT count(*) FROM sqlite_master').fetchone()
c.enable_load_extension(True);sqlite_vec.load(c)
assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
types_before={r[1]:r[2] for r in c.execute('PRAGMA table_list')}
old_before=old.semantic_digest(c)
stable_before=worker.stable_semantic_digest(c,census)
c.execute('ALTER TABLE kg_claim_extraction_outcomes ADD COLUMN local_replay_proof TEXT')
types_after={r[1]:r[2] for r in c.execute('PRAGMA table_list')}
old_after=old.semantic_digest(c)
stable_after=worker.stable_semantic_digest(c,census)
assert all(types_before['vec_probe_'+suffix]=='table' and types_after['vec_probe_'+suffix]=='shadow' for suffix in ('info','chunks','rowids'))
assert old_before!=old_after
assert stable_before==stable_after
c.execute("UPDATE sessions SET value='changed'")
assert worker.stable_semantic_digest(c,census)!=stable_after
c.close()
print(json.dumps({'exact_transition':True,'old_digest_changed':True,'stable_digest_equal':True,'mutation_detected':True}))
'''
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run([str(python), "-I", "-B", "-c", script, str(repo), str(tmp_path / "native.sqlite")],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '"stable_digest_equal": true' in result.stdout
