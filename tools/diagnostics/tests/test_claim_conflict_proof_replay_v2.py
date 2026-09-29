"""Offline schema-boundary controls for the R7 v63→v64 proof replay."""
import sqlite3
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_proof_replay_v2 as worker


def test_fails_closed_on_preexisting_proof_column(tmp_path, monkeypatch):
    from hymem.core import db

    def clone(_source, target):
        conn = sqlite3.connect(target)
        conn.execute("CREATE TABLE kg_claim_extraction_outcomes (local_replay_proof TEXT)")
        conn.close()

    replay = SimpleNamespace(
        GENERATION="gen", captured_cache_key=lambda _raw: "derived",
        clone=clone, open_clone=lambda target: sqlite3.connect(target),
    )
    monkeypatch.setattr(worker, "WORK", tmp_path)
    monkeypatch.setattr(db, "schema_version", lambda _conn: 63)
    with pytest.raises(RuntimeError, match="preupgrade_proof_column_present"):
        worker.one_arm(None, replay, {"extraction": {"phase1_generation": {
            "generation_key": "gen", "extraction_cache_key": "derived"}}},
            tmp_path / "source.sqlite", dedup=True, label="already-stamped")


def test_fails_closed_on_wrong_source_schema(tmp_path, monkeypatch):
    from hymem.core import db

    def clone(_source, target):
        sqlite3.connect(target).close()

    replay = SimpleNamespace(
        GENERATION="gen", captured_cache_key=lambda _raw: "derived",
        clone=clone, open_clone=lambda target: sqlite3.connect(target),
    )
    monkeypatch.setattr(worker, "WORK", tmp_path)
    monkeypatch.setattr(db, "schema_version", lambda _conn: 61)
    with pytest.raises(RuntimeError, match="snapshot_schema_not_v63"):
        worker.one_arm(None, replay, {"extraction": {"phase1_generation": {
            "generation_key": "gen", "extraction_cache_key": "derived"}}},
            tmp_path / "source.sqlite", dedup=True, label="wrong-schema")
