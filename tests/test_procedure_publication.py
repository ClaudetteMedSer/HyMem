"""Complete digest publications reconcile only procedures they own."""
import json
import hashlib
import sqlite3
import uuid
from dataclasses import replace

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.dreaming.procedures import ProceduresExtraction, publish_digest_procedures
from tests.test_digest_publication import _finish, _initial


def _procedures(hy, sid="x"):
    return [tuple(row) for row in hy.conn.execute(
        "SELECT id,name,steps,status,confidence FROM procedures "
        "WHERE session_id=? ORDER BY id", (sid,),
    )]


def test_complete_empty_rebuild_retires_old_procedures_and_fts(cfg):
    hy, llm = _initial(cfg, long=False)
    try:
        old = _procedures(hy)
        assert old
        llm.emit_slice_artifacts = False
        hy.conn.execute("UPDATE sessions SET digested_prompt_version=NULL WHERE id='x'")
        _finish(hy)
        assert all(row[3] == "stale" for row in _procedures(hy))
        assert not hy.augment("Handle slice").procedures
    finally:
        hy.close()


def _item(name, action="do this"):
    return dict(name=name, description="test workflow",
                steps=[dict(order=1, action=action, tool=None)],
                triggers=[], entities_involved=[])


def _publish(hy, slices, *, replacing=True):
    old = hy.conn.execute("SELECT digest_published_generation FROM sessions WHERE id='x'").fetchone()[0]
    generation = old.rsplit("|walk=", 1)[0] + "|walk=" + uuid.uuid4().hex if replacing else old
    with db.transaction(hy.conn):
        publish_digest_procedures(hy.conn, "x", generation,
                                 [ProceduresExtraction(items=items) for items in slices],
                                 replacing=replacing)
        hy.conn.execute("UPDATE sessions SET digest_published_generation=? WHERE id='x'", (generation,))


def _active(hy):
    return {row[1]: row for row in _procedures(hy) if row[3] == "active"}


def test_completed_union_replaces_omissions_but_tail_is_additive(cfg):
    hy, _ = _initial(cfg, long=False)
    try:
        _publish(hy, [[_item("first")], [], [_item("second")]])
        assert set(_active(hy)) == {"first", "second"}
        _publish(hy, [[]], replacing=False)
        assert set(_active(hy)) == {"first", "second"}
        _publish(hy, [[_item("tail")]], replacing=False)
        assert set(_active(hy)) == {"first", "second", "tail"}
        _publish(hy, [[_item("only replacement")], []])
        assert set(_active(hy)) == {"only replacement"}
    finally:
        hy.close()


def test_name_duplicates_have_deterministic_last_source_wins_policy(cfg):
    hy, _ = _initial(cfg, long=False)
    try:
        _publish(hy, [[_item("STRASSE", "first"), _item("strasse", "second")],
                      [_item("Straße", "third")]])
        rows = list(_active(hy).values())
        assert len(rows) == 1 and "third" in rows[0][2]
        pid = rows[0][0]
        _publish(hy, [[_item("STRASSE", "tail")]], replacing=False)
        assert len(_active(hy)) == 1
        assert next(iter(_active(hy).values()))[0] == pid
    finally:
        hy.close()


def test_unknown_manual_same_name_other_session_and_identity_collision_survive(cfg):
    hy, _ = _initial(cfg, long=False)
    try:
        hy.conn.execute("INSERT INTO sessions(id) VALUES ('other')")
        key = hashlib.sha256(b"x\0manual").hexdigest()
        for pid, sid, name in (("manual", "x", "manual"),
                               (f"x@digest_proc_{key}", "x", "id occupied"),
                               ("other-proc", "other", "manual")):
            hy.conn.execute("INSERT INTO procedures(id,session_id,name,steps) VALUES (?,?,?,?)",
                            (pid, sid, name, '[{"order":1,"action":"curated"}]'))
        before = [tuple(r) for r in hy.conn.execute("SELECT * FROM procedures WHERE id IN ('manual','other-proc')")]
        _publish(hy, [[_item("manual", "inferred")]])
        assert hy.conn.execute("SELECT COUNT(*) FROM procedures WHERE name='manual' AND session_id='x'").fetchone()[0] == 2
        _publish(hy, [[]])
        after = [tuple(r) for r in hy.conn.execute("SELECT * FROM procedures WHERE id IN ('manual','other-proc')")]
        assert before == after
        assert set(_active(hy)) == {"manual", "id occupied"}
    finally:
        hy.close()


def test_manual_content_edit_detaches_ownership_without_losing_feedback(cfg):
    hy, _ = _initial(cfg, long=False)
    try:
        _publish(hy, [[_item("edited"), _item("feedback")]])
        edited, feedback = _active(hy)["edited"][0], _active(hy)["feedback"][0]
        hy.conn.execute("UPDATE procedures SET description='operator override' WHERE id=?", (edited,))
        assert hy.mark_procedure_stale(feedback)
        penalty = hy.conn.execute("SELECT confidence FROM procedures WHERE id=?", (feedback,)).fetchone()[0]
        _publish(hy, [[]])
        assert "edited" in _active(hy)
        assert hy.conn.execute("SELECT 1 FROM procedure_digest_publications WHERE procedure_id=?", (edited,)).fetchone() is None
        _publish(hy, [[_item("feedback")]])
        assert _active(hy)["feedback"][0] == feedback
        assert _active(hy)["feedback"][4] == penalty < 1
        assert "edited" in _active(hy)
    finally:
        hy.close()


def test_publication_failure_and_restart_preserve_old_then_retire(cfg):
    hy, llm = _initial(cfg)
    try:
        old = _procedures(hy)
        llm.emit_slice_artifacts = False
        hy.conn.execute("UPDATE sessions SET digested_prompt_version=NULL WHERE id='x'")
        hy.dream()
        assert _procedures(hy) == old
        config = hy.config
        hy.close()
        hy = HyMem(config, llm=llm)
        assert _procedures(hy) == old
        hy.conn.execute("CREATE TEMP TRIGGER publication_fail BEFORE UPDATE OF digest_published_generation "
                        "ON sessions BEGIN SELECT RAISE(ABORT,'publication failed'); END")
        with pytest.raises(sqlite3.IntegrityError, match="publication failed"):
            _finish(hy)
        assert _procedures(hy) == old
        hy.conn.execute("DROP TRIGGER publication_fail")
        _finish(hy)
        assert not _active(hy)
    finally:
        hy.close()


def test_v58_upgrade_does_not_infer_legacy_ownership(cfg):
    hy, llm = _initial(cfg, long=False)
    try:
        old = _procedures(hy)
        hy.conn.execute("DROP TABLE procedure_digest_publications")
        hy.conn.execute("UPDATE schema_meta SET value='58' WHERE key='schema_version'")
        config = hy.config
        hy.close()
        hy = HyMem(config, llm=llm)
        assert db.schema_version(hy.conn) == db.EXPECTED_SCHEMA_VERSION
        assert hy.conn.execute("SELECT COUNT(*) FROM procedure_digest_publications").fetchone()[0] == 0
        llm.emit_slice_artifacts = False
        hy.conn.execute("UPDATE sessions SET digested_prompt_version=NULL WHERE id='x'")
        _finish(hy)
        assert _procedures(hy) == old
    finally:
        hy.close()


def _rewrite_export(path, change):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    change(records)
    counts = {key: 0 for key in records[-1]["counts"]}
    for row in records[1:-1]:
        counts[row["type"]] += 1
    body = "".join(json.dumps(row) + "\n" for row in records[:-1])
    records[-1]["counts"] = counts
    records[-1]["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    path.write_text(body + json.dumps(records[-1]) + "\n")


@pytest.mark.parametrize("legacy", [False, True])
def test_portable_roundtrip_retains_explicit_but_not_inferred_ownership(cfg, tmp_path, legacy):
    hy, llm = _initial(cfg, long=False)
    try:
        path = tmp_path / "procedures.jsonl"
        hy.export(path)
        if legacy:
            def downgrade(records):
                records[0]["version"] = 15
                newer_kinds = {"procedure_digest_publication", "edge_evidence_extraction_audit"}
                records[:] = [row for row in records if row["type"] not in newer_kinds]
                for kind in newer_kinds:
                    del records[-1]["counts"][kind]
            _rewrite_export(path, downgrade)
        restored = HyMem(replace(hy.config, root=tmp_path / "restored"), llm=llm)
        try:
            restored.import_(path)
            restored.import_(path)  # exact ownership re-import is idempotent
            assert len(_active(restored)) == 1
            llm.emit_slice_artifacts = False
            restored.conn.execute("UPDATE sessions SET digested_prompt_version=NULL WHERE id='x'")
            _finish(restored)
            assert bool(_active(restored)) is legacy
        finally:
            restored.close()
    finally:
        hy.close()


@pytest.mark.parametrize("mutation", ["payload", "generation", "orphan", "duplicate", "retired"])
def test_import_rejects_unbound_ownership_atomically(cfg, tmp_path, mutation):
    hy, llm = _initial(cfg, long=False)
    try:
        path = tmp_path / "bad.jsonl"
        hy.export(path)
        def damage(records):
            envelope = next(row for row in records if row["type"] == "procedure_digest_publication")
            owner = envelope["record"]
            if mutation == "payload":
                owner["payload_sha256"] = "0" * 64
            elif mutation == "generation":
                owner["generation"] = owner["generation"][:-32] + "0" * 32
            elif mutation == "orphan":
                owner["procedure_id"] = "not-a-procedure"
            elif mutation == "retired":
                owner["retired"] = 1
            else:
                records.insert(-1, envelope)
        _rewrite_export(path, damage)
        restored = HyMem(replace(hy.config, root=tmp_path / "restored"), llm=llm)
        try:
            with pytest.raises(ValueError, match="procedure ownership"):
                restored.import_(path)
            assert restored.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        finally:
            restored.close()
    finally:
        hy.close()


def test_import_cannot_take_retirement_authority_over_identical_manual_target(cfg, tmp_path):
    hy, llm = _initial(cfg, long=False)
    try:
        path = tmp_path / "owned.jsonl"
        hy.export(path)
        hy.conn.execute("DELETE FROM procedure_digest_publications")
        old = _procedures(hy)
        with pytest.raises(ValueError, match="unknown/manual target"):
            hy.import_(path)
        assert _procedures(hy) == old
        assert hy.conn.execute("SELECT COUNT(*) FROM procedure_digest_publications").fetchone()[0] == 0
    finally:
        hy.close()


def test_ownership_is_material_and_cascades_with_procedure_retention(cfg):
    from benchmarks.store_attestation import material_store_state
    from hymem.dreaming.retention import prune_episodes_and_procedures
    hy, _ = _initial(cfg, long=False)
    try:
        before = material_store_state(hy.config.db_path)
        hy.conn.execute("UPDATE procedure_digest_publications SET retired=1")
        after = material_store_state(hy.config.db_path)
        changed = {key for key in before["tables"] if before["tables"][key] != after["tables"][key]}
        assert changed == {"procedure_digest_publications"}
        assert before["sha256"] != after["sha256"]
        hy.conn.execute("UPDATE procedures SET status='stale',created_at='2000-01-01'")
        with db.transaction(hy.conn):
            prune_episodes_and_procedures(hy.conn, hy.config)
        assert hy.conn.execute("SELECT COUNT(*) FROM procedure_digest_publications").fetchone()[0] == 0
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        hy.close()


def test_redacted_portable_ownership_binds_redacted_payload(cfg, tmp_path):
    from hymem.dreaming.procedures import procedure_payload_sha256
    hy, llm = _initial(cfg, long=False)
    try:
        _publish(hy, [[_item("Contact alice.private@example.com", "email alice.private@example.com")]])
        path = tmp_path / "private.jsonl"
        hy.export(path)
        restored = HyMem(replace(hy.config, root=tmp_path / "restored"), llm=llm)
        try:
            restored.import_(path)
            rows = restored.conn.execute("SELECT p.*,o.payload_sha256 FROM procedures p JOIN "
                                         "procedure_digest_publications o ON o.procedure_id=p.id").fetchall()
            assert rows
            assert all(row["payload_sha256"] == procedure_payload_sha256(row) for row in rows)
            assert all("alice.private@example.com" not in row["name"] + row["steps"] for row in rows)
            llm.emit_slice_artifacts = False
            restored.conn.execute("UPDATE sessions SET digested_prompt_version=NULL WHERE id='x'")
            _finish(restored)
            assert not _active(restored)
        finally:
            restored.close()
    finally:
        hy.close()


def test_export_does_not_confer_ownership_after_manual_content_edit(cfg, tmp_path):
    hy, _ = _initial(cfg, long=False)
    try:
        hy.conn.execute("UPDATE procedures SET description='manual override'")
        path = tmp_path / "edited.jsonl"
        hy.export(path)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert any(row["type"] == "procedure" for row in records)
        assert not any(row["type"] == "procedure_digest_publication" for row in records)
    finally:
        hy.close()
