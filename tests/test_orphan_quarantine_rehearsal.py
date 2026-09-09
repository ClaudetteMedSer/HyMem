"""Synthetic v59 clone-only quarantine; no deployed stores or providers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from hymem.core import db
from hymem.core.vectors import encode_vector
from hymem.deadline import MonotonicDeadline
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.extraction.embeddings import MappedStubEmbeddingClient, embedding_text_hash


_SCRIPT = Path(__file__).resolve().parents[1] / "tools/deployment/rehearse_orphan_quarantine.py"


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location("quarantine_rehearsal", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.sqlite"
    conn = db.connect(path)
    db.initialize(conn)
    client = MappedStubEmbeddingClient({
        "zebraproof": [1.0, 0, 0],
        "zebraproof healthy control": [0.5, 0.5, 0],
        "zebraproof private orphan payload": [1.0, 0, 0],
    }, model="synthetic-quarantine", dim=3)
    model, dim = embedding_storage_identity(client)
    conn.execute("INSERT INTO sessions(id) VALUES ('healthy-control')")
    conn.execute("INSERT INTO chunks(rowid,id,session_id,start_message_id,end_message_id,salience_reason,text) VALUES (7,'control','healthy-control',1,1,'test','zebraproof healthy control')")
    # Reconstruct pre-existing corruption in a synthetic fixture only. The
    # helper never changes FK enforcement on any connection.
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("INSERT INTO chunks(rowid,id,session_id,start_message_id,end_message_id,salience_reason,text) VALUES (1058,'orphan-id','missing-parent',900001,900001,'legacy','zebraproof private orphan payload')")
    conn.execute("PRAGMA foreign_keys=ON")
    for index in range(4):
        conn.execute("INSERT INTO entity_mentions(chunk_id,entity_canonical) VALUES ('orphan-id',?)", (f"canonical-{index}",))
    with db.transaction(conn), db.embedding_mutation(conn):
        conn.execute("INSERT INTO chunk_embeddings(rowid,chunk_id,vector_json,model,dim,text_hash) VALUES (27,'control',?, ?,3,?)", (encode_vector([0.5, 0.5, 0]), model, embedding_text_hash("zebraproof healthy control")))
        # SQLite BLOB stored in the historical TEXT-affinity vector column.
        conn.execute("INSERT INTO chunk_embeddings(rowid,chunk_id,vector_json,model,dim,text_hash) VALUES (1059,'orphan-id',?,?,3,?)", (encode_vector([1.0, 0, 0]).encode("ascii"), model, embedding_text_hash("zebraproof private orphan payload")))
        db.ensure_vec_table(conn, dim, model=model)
    conn.close()
    yield path, client


def _reference(helper, path):
    conn = helper._readonly(path)
    try:
        return helper.reference_fingerprint(conn)
    finally:
        conn.close()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logical(helper, path):
    conn = helper._readonly(path)
    try:
        helper._load_vectors(conn)
        return helper._schema(conn), helper._snapshot(conn, MonotonicDeadline.after(30))
    finally:
        conn.close()


def test_success_preserves_source_archive_native_rows_and_full_baseline_restore(helper, source, tmp_path):
    path, client = source
    before_file, before = _digest(path), _logical(helper, path)
    reference = _reference(helper, path)
    from hymem.query.augment import _query_embedding_with_status, _vector_search
    # The maintained offline stub and real query-binding path establish the
    # exact test producer; never relabel arbitrary raw query vectors.
    query_vector, status = _query_embedding_with_status(client, "zebraproof")
    assert query_vector is not None
    original_search = db.connect(path)
    try:
        assert [hit.chunk_id for hit in _vector_search(original_search, client, "zebraproof", top_k=10, max_scan=100, query_vector=query_vector)] == ["orphan-id", "control"]
        assert len(client.calls) == 1
        assert {row[0] for row in original_search.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'zebraproof'")} == {7, 1058}
    finally:
        original_search.close()
    destination = tmp_path / "rehearsal"
    result = helper.rehearse(path, destination, reference)
    assert result["status"] == "verified_clone_only"
    assert result["quarantined_chunks"] == result["quarantined_durable_vectors"] == 1
    assert result["quarantined_entity_mentions"] == 4
    assert result["archive_native_roundtrip_verified"] and result["full_baseline_restore_verified"]
    assert result["archive_only_runtime_reinsert_tested"] is False
    assert _digest(path) == before_file and _logical(helper, path) == before
    assert _logical(helper, destination / "baseline.sqlite") == before
    assert _logical(helper, destination / "baseline-restored.sqlite") == before
    assert destination.stat().st_mode & 0o777 == 0o700
    for filename in ("baseline.sqlite", "working.sqlite", "quarantine.sqlite", "baseline-restored.sqlite"):
        assert (destination / filename).stat().st_mode & 0o777 == 0o600
    archive = helper._readonly(destination / "quarantine.sqlite")
    try:
        assert archive.execute("SELECT rowid,_rowid_ FROM chunks").fetchone()[:] == (1058, 1058)
        assert archive.execute("SELECT rowid,_rowid_,typeof(vector_json) FROM chunk_embeddings").fetchone()[:] == (1059, 1059, "blob")
        original = helper._readonly(path)
        try:
            assert tuple(archive.execute("SELECT * FROM chunk_embeddings").fetchone()) == tuple(original.execute("SELECT rowid,* FROM chunk_embeddings WHERE rowid=1059").fetchone())
            assert [tuple(row) for row in archive.execute("SELECT * FROM entity_mentions ORDER BY rowid")] == [tuple(row) for row in original.execute("SELECT rowid,* FROM entity_mentions ORDER BY rowid")]
        finally:
            original.close()
    finally:
        archive.close()
    working = db.connect(destination / "working.sqlite")
    try:
        assert working.execute("PRAGMA foreign_key_check").fetchall() == []
        assert working.execute("SELECT id FROM chunks").fetchone()[0] == "control"
        assert [row[0] for row in working.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'zebraproof'")] == [7]
        hits = _vector_search(working, client, "zebraproof", top_k=10, max_scan=100, query_vector=query_vector)
        assert [hit.chunk_id for hit in hits] == ["control"]
        assert len(client.calls) == 1
        if db.has_vec_table(working, table="vec_chunks"):
            helper._load_vectors(working)
            assert [row[0] for row in working.execute("SELECT rowid FROM vec_chunks")] == [7]
    finally:
        working.close()


def test_reusing_destination_or_already_quarantined_source_refuses(helper, source, tmp_path):
    path, _ = source
    destination = tmp_path / "rehearsal"
    reference = _reference(helper, path)
    helper.rehearse(path, destination, reference)
    before = {file.name: _digest(file) for file in destination.glob("*.sqlite")}
    with pytest.raises(FileExistsError):
        helper.rehearse(path, destination, reference)
    assert {file.name: _digest(file) for file in destination.glob("*.sqlite")} == before
    with pytest.raises(helper.Refused, match="target_absent"):
        helper.rehearse(destination / "working.sqlite", tmp_path / "repeat", reference)
    assert _digest(destination / "working.sqlite") == before["working.sqlite"]


@pytest.mark.parametrize("dependent", ["declared", "physical", "aggregate_json", "raw", "parent"])
def test_proven_or_referenced_targets_refuse_without_cascading(helper, source, tmp_path, dependent):
    path, _ = source
    conn = db.connect(path)
    if dependent == "declared":
        conn.execute("INSERT INTO processed_chunks(chunk_id,prompt_version) VALUES ('orphan-id','reviewed-old-version')")
    elif dependent == "physical":
        conn.execute("CREATE TABLE custom_source_reference(value TEXT)")
        conn.execute("INSERT INTO custom_source_reference VALUES ('orphan-id')")
    elif dependent == "aggregate_json":
        conn.execute("INSERT INTO aggregation_nodes(id,title,summary,member_episode_ids,session_ids) VALUES ('derived','Private','Proof dependency','[]','[\"missing-parent\"]')")
    elif dependent == "raw":
        conn.execute("INSERT INTO messages(id,session_id,role,content) VALUES (900001,'healthy-control','user','Recovered possible source')")
    else:
        conn.execute("INSERT INTO sessions(id) VALUES ('missing-parent')")
    conn.close()
    reference, before = _reference(helper, path), _digest(path)
    destination = tmp_path / "blocked"
    with pytest.raises(helper.Refused):
        helper.rehearse(path, destination, reference)
    assert _digest(path) == before
    assert not (destination / "quarantine.sqlite").exists()
    conn = helper._readonly(destination / "working.sqlite")
    try:
        assert conn.execute("SELECT COUNT(*) FROM chunks WHERE rowid=1058").fetchone()[0] == 1
    finally:
        conn.close()


def test_exact_reference_rejects_changed_bytes_even_same_rowid_and_id(helper, source, tmp_path):
    path, _ = source
    reference = _reference(helper, path)
    conn = db.connect(path)
    conn.execute("UPDATE chunks SET text='changed private bytes' WHERE rowid=1058")
    conn.close()
    before = _digest(path)
    with pytest.raises(helper.Refused, match="reference_mismatch"):
        helper.rehearse(path, tmp_path / "changed", reference)
    assert _digest(path) == before


@pytest.mark.parametrize("fault", ["archive", "precommit", "cancel"])
def test_archive_failure_and_precommit_fault_preserve_working_and_source(helper, source, tmp_path, monkeypatch, fault):
    path, _ = source
    reference, before_file, before = _reference(helper, path), _digest(path), _logical(helper, path)
    destination = tmp_path / "failed"
    if fault == "archive":
        def fail(*args):
            raise helper.Refused("archive_roundtrip_failed")
        monkeypatch.setattr(helper, "_archive", fail)
    else:
        original = helper._integrity
        def fail(conn):
            original(conn)
            if conn.in_transaction:
                if fault == "cancel":
                    raise KeyboardInterrupt()
                raise helper.Refused("injected_precommit_failure")
        monkeypatch.setattr(helper, "_integrity", fail)
    with pytest.raises((helper.Refused, KeyboardInterrupt)):
        helper.rehearse(path, destination, reference)
    assert _digest(path) == before_file and _logical(helper, path) == before
    assert _logical(helper, destination / "working.sqlite") == before
    if fault != "archive":
        assert (destination / "quarantine.sqlite").exists()
        conn = helper._readonly(destination / "quarantine.sqlite")
        try:
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
        finally:
            conn.close()


def test_unexpected_trigger_delta_rolls_back_without_rebuilding(helper, source, tmp_path):
    path, _ = source
    conn = db.connect(path)
    conn.execute("CREATE TRIGGER unexpected_quarantine_delta AFTER DELETE ON chunks BEGIN DELETE FROM chunk_embeddings WHERE chunk_id='control'; END")
    conn.close()
    before = _logical(helper, path)
    with pytest.raises(helper.Refused, match="unexpected_logical_delta"):
        helper.rehearse(path, tmp_path / "trigger", _reference(helper, path))
    assert _logical(helper, tmp_path / "trigger" / "working.sqlite") == before


def test_absent_fts_delete_trigger_cannot_leave_hidden_postings(helper, source, tmp_path):
    path, _ = source
    conn = db.connect(path)
    conn.execute("DROP TRIGGER chunks_fts_delete")
    conn.close()
    before = _logical(helper, path)
    with pytest.raises(helper.Refused, match="unexpected_logical_delta"):
        helper.rehearse(path, tmp_path / "fts", _reference(helper, path))
    assert _logical(helper, tmp_path / "fts" / "working.sqlite") == before


def test_cli_output_never_contains_paths_identifiers_or_source_text(helper, source, tmp_path):
    path, _ = source
    args = [sys.executable, str(_SCRIPT), "--source", str(path), "--rehearsal-dir", str(tmp_path / "cli"),
            "--reference-sha256", _reference(helper, path)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)["status"] == "verified_clone_only"
    assert result.stderr == ""
    for private in (str(path), "orphan-id", "missing-parent", "private orphan payload"):
        assert private not in result.stdout + result.stderr
    bad = subprocess.run(args + ["--private-unknown-option", "credential-value"], capture_output=True, text=True, timeout=10)
    assert bad.returncode == 1 and bad.stderr == "" and "credential-value" not in bad.stdout
    assert json.loads(bad.stdout)["reason"] == "invalid_arguments"


def test_existing_vector_shadows_require_extension(helper, source, tmp_path, monkeypatch):
    path, _ = source
    conn = helper._readonly(path)
    try:
        if not db.has_vec_table(conn, table="vec_chunks"):
            pytest.skip("sqlite-vec unavailable in fixture")
    finally:
        conn.close()
    monkeypatch.setattr(db, "_load_vec_extension", lambda conn: False)
    before = _digest(path)
    with pytest.raises(helper.Refused, match="vector_extension_unavailable"):
        helper.rehearse(path, tmp_path / "no-vec", _reference(helper, path))
    assert _digest(path) == before


def test_orphan_without_entity_mentions_is_also_recoverable(helper, source, tmp_path):
    path, _ = source
    conn = db.connect(path)
    conn.execute("DELETE FROM entity_mentions")
    conn.close()
    result = helper.rehearse(path, tmp_path / "no-mentions", _reference(helper, path))
    assert result["quarantined_entity_mentions"] == 0
