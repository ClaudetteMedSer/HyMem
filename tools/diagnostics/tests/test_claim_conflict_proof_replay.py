"""Offline controls for v61→v62 ordered-proof replay diagnostics."""
from pathlib import Path
import hashlib
import sqlite3
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_proof_replay as proof


def test_semantic_digest_ignores_only_new_proof_column(tmp_path):
    conn = sqlite3.connect(tmp_path / "synthetic.sqlite")
    try:
        conn.execute("CREATE TABLE kg_claim_extraction_outcomes (chunk_id TEXT, result_hash TEXT)")
        conn.execute("INSERT INTO kg_claim_extraction_outcomes VALUES ('private', 'hash')")
        with pytest.raises(RuntimeError, match="semantic_audit_table_missing"):
            proof.semantic_digest(conn)
        original = proof.SEMANTIC_TABLES
        try:
            proof.SEMANTIC_TABLES = ("kg_claim_extraction_outcomes",)
            before = proof.semantic_digest(conn)
            conn.execute("ALTER TABLE kg_claim_extraction_outcomes ADD COLUMN local_replay_proof TEXT")
            assert proof.semantic_digest(conn) == before
            conn.execute("UPDATE kg_claim_extraction_outcomes SET local_replay_proof='sha256:"
                         + "a" * 64 + "'")
            assert proof.semantic_digest(conn) == before
            conn.execute("UPDATE kg_claim_extraction_outcomes SET result_hash='changed'")
            assert proof.semantic_digest(conn) != before
        finally:
            proof.SEMANTIC_TABLES = original
    finally:
        conn.close()


def test_integrity_gate_rejects_any_findings():
    clean = {"integrity_ok": True, "foreign_key_findings": 0,
             "canonical_drift_findings": 0, "ledger_count_mismatches": 0,
             "same_generation_disagreeing_groups": 0}
    proof.require_clean(clean, "bad")
    for key in clean:
        bad = {**clean, key: False if key == "integrity_ok" else 1}
        with pytest.raises(RuntimeError, match="bad"):
            proof.require_clean(bad, "bad")


def test_private_reference_modes_and_no_symlinks(tmp_path):
    path = tmp_path / "source.sqlite"
    path.write_bytes(b"private")
    for mode in (0o400, 0o600):
        path.chmod(mode)
        proof.private_input(path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private_input_mode_invalid"):
        proof.private_input(path)
    link = tmp_path / "link.sqlite"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="private_input_mode_invalid"):
        proof.private_input(link)


def test_proof_lookup_and_shape_are_private(tmp_path):
    conn = sqlite3.connect(tmp_path / "proof.sqlite")
    try:
        conn.execute("CREATE TABLE kg_claim_extraction_outcomes ("
                     "chunk_id TEXT,prompt_version TEXT,phase1_generation_key TEXT,"
                     "local_replay_proof TEXT)")
        value = "sha256:" + "a" * 64
        conn.execute("INSERT INTO kg_claim_extraction_outcomes VALUES (?,?,?,?)",
                     ("private chunk", "derived", "gen", value))
        assert proof.proof_row(conn, "private chunk", "derived", "gen") == value
        assert proof.PROOF.fullmatch(value)
        assert proof.proof_row(conn, "private chunk", "raw", "gen") is None
    finally:
        conn.close()


@pytest.mark.parametrize("erase_on_reopen", [False, True])
def test_reopen_initialize_cannot_erase_proof_unobserved(tmp_path, monkeypatch,
                                                           erase_on_reopen):
    from hymem.core import db

    clean = {"integrity_ok": True, "foreign_key_findings": 0,
             "canonical_drift_findings": 0, "ledger_count_mismatches": 0,
             "same_generation_disagreeing_groups": 0}
    calls = {"initialize": 0, "attempt": 0}

    def clone(_source, target):
        conn = sqlite3.connect(target)
        conn.execute("PRAGMA user_version=61")
        conn.execute("CREATE TABLE phase1_generations (generation_key TEXT,extraction_cache_key TEXT)")
        conn.execute("INSERT INTO phase1_generations VALUES ('gen','derived')")
        conn.execute("CREATE TABLE kg_claim_extraction_outcomes ("
                     "chunk_id TEXT,prompt_version TEXT,phase1_generation_key TEXT,"
                     "local_replay_proof TEXT)")
        conn.commit()
        conn.close()

    def open_clone(target):
        conn = sqlite3.connect(target)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(conn):
        calls["initialize"] += 1
        conn.execute("PRAGMA user_version=62")
        if erase_on_reopen and calls["initialize"] == 3:
            conn.execute("UPDATE kg_claim_extraction_outcomes SET local_replay_proof=NULL")
            conn.commit()

    def attempt(conn, _raw, **_kwargs):
        calls["attempt"] += 1
        before = logical(conn)
        if calls["attempt"] == 1:
            conn.execute("INSERT INTO kg_claim_extraction_outcomes VALUES (?,?,?,?)",
                         ("chunk", "derived", "gen", "sha256:" + "a" * 64))
            conn.commit()
        after = logical(conn)
        return {"status": "persisted", "logical_digest_before": before,
                "logical_digest_after": after}

    def logical(conn):
        return hashlib.sha256("\n".join(conn.iterdump()).encode()).hexdigest()

    replay = SimpleNamespace(
        GENERATION="gen", captured_cache_key=lambda _raw: "derived",
        clone=clone, open_clone=open_clone, integrity=lambda _conn: clean,
        logical_digest=logical,
        reconstruct=lambda *_args, **_kwargs: (
            SimpleNamespace(id="chunk"),
            SimpleNamespace(phase1_generation={"generation_key": "gen"}),
            None, None, None),
        publication_count=lambda conn, *_args: int(conn.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes").fetchone()[0]),
        attempt=attempt,
    )
    monkeypatch.setattr(db, "initialize", initialize)
    monkeypatch.setattr(db, "schema_version", lambda conn: conn.execute(
        "PRAGMA user_version").fetchone()[0])
    monkeypatch.setattr(proof, "semantic_digest", lambda _conn: "s" * 64)
    monkeypatch.setattr(proof, "WORK", tmp_path)
    raw = {"extraction": {"phase1_generation": {
        "generation_key": "gen", "extraction_cache_key": "derived"}}}
    if erase_on_reopen:
        with pytest.raises(RuntimeError, match="proof_or_state_changed_on_reopen"):
            proof.one_arm(replay, raw, tmp_path / "source.sqlite",
                          dedup=True, label="dedup-on")
        assert calls["attempt"] == 1
    else:
        report = proof.one_arm(replay, raw, tmp_path / "source.sqlite",
                               dedup=True, label="dedup-on")
        assert report["status"] == "completed"
        assert report["published_after_first"] == report["published_after_repeat"] == 1
        assert calls["attempt"] == 2
