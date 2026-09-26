"""The portable public summary frontier is explicit; local drafts are not."""
from __future__ import annotations

import hashlib
import json
from contextlib import closing

import pytest

from hymem import HyMem, HyMemConfig, portability
from hymem.core import db
from hymem.dreaming.digest import digest_config_version
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.summary_state import (
    classify_summary_state, mark_summary_current, record_summary_failure,
)


def _generation(char="a"):
    return digest_config_version(
        prompt_version="v1", episode_prompt_version=None, max_chars=8000,
        max_tokens=3072, max_episodes=None,
    ) + "|walk=" + char * 32


def _seed(hy, state="healthy"):
    first = hy.log_message("x", "user", "The first source says the system uses SQLite.")
    last = hy.log_message("x", "assistant", "The second source describes a nightly backup.")
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, "x")
    hy.conn.execute(
        "UPDATE sessions SET digest_published_generation=?,digest_published_message_id=?,"
        "digest_cursor_prompt_version=?,digest_cursor_message_id=? WHERE id='x'",
        (_generation(), last, _generation(), last),
    )
    if state != "missing":
        with db.transaction(hy.conn):
            mark_summary_current(hy.conn, "x", "SQLite is backed up every night.",
                                 generation=_generation(), covered_message_id=last)
    if state in {"degraded", "missing"}:
        with db.transaction(hy.conn):
            record_summary_failure(hy.conn, "x", "summary_validation_failure")
    # Operator-facing text is not an automatic acknowledgement.
    hy.conn.execute("UPDATE sessions SET summary=?,summary_source='legacy' WHERE id='x'",
                    ("  Operator’s untouched note.\nSecond line.  ",))
    return first, last


def _rewrite(path, mutate):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    body, end = rows[:-1], rows[-1]
    mutate(body)
    end["counts"] = {
        kind: sum(row["type"] == kind for row in body) for kind in end["counts"]
    }
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in body)
    end["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    path.write_text(encoded + json.dumps(end) + "\n")


def _record(body):
    return next(row["record"] for row in body if row["type"] == "session" and row["record"]["id"] == "x")


def _downgrade(path):
    def mutate(body):
        body[0]["version"] = 17
        for row in body:
            if row["type"] == "session":
                for field in portability._V18_SUMMARY_FIELDS:
                    row["record"].pop(field)
    _rewrite(path, mutate)


@pytest.mark.parametrize("state", ["healthy", "degraded", "missing"])
@pytest.mark.parametrize("redact", [False, True])
def test_public_summary_state_roundtrips_without_automatic_promotion(tmp_path, state, redact):
    with closing(HyMem(HyMemConfig(root=tmp_path / "src", redact_secrets=False))) as src:
        _seed(src, state)
        before = dict(src.conn.execute("SELECT * FROM sessions WHERE id='x'").fetchone())
        health = classify_summary_state(src.conn, "x")
        path = tmp_path / "memory.jsonl"
        src.export(path)
    assert json.loads(path.read_text().splitlines()[0])["version"] == 18
    with closing(HyMem(HyMemConfig(root=tmp_path / "dst", redact_secrets=redact))) as dst:
        dst.import_(path)
        after = dict(dst.conn.execute("SELECT * FROM sessions WHERE id='x'").fetchone())
        for field in (*portability._V18_SUMMARY_FIELDS, "summary", "summary_source",
                      "auto_summary", "auto_summary_message_id"):
            assert after[field] == before[field]
        assert classify_summary_state(dst.conn, "x") == health
        assert dst.conn.execute("SELECT COUNT(*) FROM summary_recovery").fetchone()[0] == 0
        # Restoring the same publication is idempotent, including degradation.
        dst.import_(path)
        assert classify_summary_state(dst.conn, "x") == health


@pytest.mark.parametrize("mode", ["healthy", "active_ahead", "unacknowledged"])
def test_v17_upgrade_uses_only_old_source_proved_summary_publication(tmp_path, mode):
    with closing(HyMem(HyMemConfig(root=tmp_path / "src"))) as src:
        first, last = _seed(src, "missing" if mode == "unacknowledged" else "healthy")
        if mode == "active_ahead":
            # The old automatic publication was the first message, while the
            # digest cursor is further ahead. Only the former is an old ack.
            src.conn.execute("UPDATE sessions SET auto_summary_message_id=?,"
                             "digest_published_message_id=? WHERE id='x'", (first, first))
        path = tmp_path / "legacy.jsonl"
        src.export(path)
    _downgrade(path)
    with closing(HyMem(HyMemConfig(root=tmp_path / "dst"))) as dst:
        dst.import_(path)
        row = dst.conn.execute("SELECT * FROM sessions WHERE id='x'").fetchone()
        expected = None if mode == "unacknowledged" else first if mode == "active_ahead" else last
        assert row["digest_published_message_id"] == expected
        assert row["auto_summary_generation"] == (None if expected is None else _generation())
        assert row["summary_failure_count"] == 0
        assert row["summary_failure_reason"] is None
        assert classify_summary_state(dst.conn, "x")["summary_healthy"] is (mode == "healthy")
        assert row["summary"] == "  Operator’s untouched note.\nSecond line.  "


@pytest.mark.parametrize("field,value", [
    ("digest_published_message_id", True), ("digest_published_message_id", 0),
    ("digest_published_message_id", -1), ("digest_published_message_id", 1.5),
    ("digest_published_message_id", 1 << 63), ("digest_published_message_id", 99999),
    ("auto_summary_generation", "invented"), ("auto_summary_generation", []),
    ("summary_failure_reason", []), ("summary_failure_reason", "private exception text"),
    ("summary_failure_reason", "summary_validation_failure"),
    ("summary_failure_count", True), ("summary_failure_count", -1),
    ("summary_failure_count", 1), ("summary_failure_count", 1 << 63),
    ("auto_summary_message_id", 99999),
])
def test_v18_forged_summary_state_rejects_and_rolls_back(tmp_path, field, value):
    with closing(HyMem(HyMemConfig(root=tmp_path / "src"))) as src:
        _seed(src)
        path = tmp_path / "forged.jsonl"
        src.export(path)
    _rewrite(path, lambda body: _record(body).__setitem__(field, value))
    with closing(HyMem(HyMemConfig(root=tmp_path / "dst"))) as dst:
        dst.log_message("sentinel", "user", "Keep existing destination content.")
        before = list(dst.conn.iterdump())
        with pytest.raises(ValueError):
            dst.import_(path)
        assert list(dst.conn.iterdump()) == before
        assert not dst.conn.in_transaction


def test_v18_foreign_source_frontier_is_not_an_acknowledgement(tmp_path):
    with closing(HyMem(HyMemConfig(root=tmp_path / "src"))) as src:
        _seed(src)
        other = src.log_message("other", "user", "A valid occurrence owned by another session.")
        with db.transaction(src.conn):
            materialize_message_coverage(src.conn, "other")
        path = tmp_path / "foreign.jsonl"
        src.export(path)
    _rewrite(path, lambda body: _record(body).__setitem__("digest_published_message_id", other))
    with closing(HyMem(HyMemConfig(root=tmp_path / "dst"))) as dst:
        with pytest.raises(ValueError, match="exact source proof"):
            dst.import_(path)
        assert dst.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert dst.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


def test_private_summary_recovery_draft_is_not_exported_or_import_authority(tmp_path):
    with closing(HyMem(HyMemConfig(root=tmp_path / "src"))) as src:
        _, last = _seed(src, "missing")
        src.conn.execute(
            "INSERT INTO summary_recovery(session_id,config_version,walk_id,target_generation,"
            "target_message_id,base_sha256,target_source_sha256,draft,source_sha256,"
            "attempt_limit,state_sha256) VALUES ('x','private',?,?,?,?,?,?,?,3,?)",
            ("a" * 32, _generation(), last, "a" * 64, "b" * 64,
             "PRIVATE DRAFT IS NOT PUBLISHED", "c" * 64, "d" * 64),
        )
        path = tmp_path / "private.jsonl"
        src.export(path)
    assert "PRIVATE DRAFT" not in path.read_text()
    assert '"summary_recovery"' not in path.read_text()
    with closing(HyMem(HyMemConfig(root=tmp_path / "dst"))) as dst:
        dst.import_(path)
        assert classify_summary_state(dst.conn, "x")["missing"]
        assert dst.conn.execute("SELECT COUNT(*) FROM summary_recovery").fetchone()[0] == 0


def test_current_summary_field_collision_cannot_overwrite_destination(tmp_path):
    with closing(HyMem(HyMemConfig(root=tmp_path / "src"))) as src:
        _seed(src)
        path = tmp_path / "collision.jsonl"
        src.export(path)
    with closing(HyMem(HyMemConfig(root=tmp_path / "dst"))) as dst:
        dst.import_(path)
        with db.transaction(dst.conn):
            record_summary_failure(dst.conn, "x", "summary_validation_failure")
        before = list(dst.conn.iterdump())
        with pytest.raises(ValueError, match="collides"):
            dst.import_(path)
        assert list(dst.conn.iterdump()) == before


def test_redaction_of_unsafe_generation_discards_all_new_authority_markers():
    record = dict(id="x", digest_published_generation=_generation(),
                  auto_summary_generation="unsafe-generation", digest_published_message_id=4,
                  summary_failure_reason="summary_validation_failure", summary_failure_count=2,
                  auto_summary="Text survives as an unacknowledged summary.",
                  auto_summary_message_id=4, auto_summary_partial_message_id=None,
                  auto_summary_message_offset=0, summary="Operator text", summary_source="legacy")
    grouped = {"session": [record]}
    portability._redact_portable_records(grouped)
    assert record["digest_published_message_id"] is None
    assert record["auto_summary_generation"] is None
    assert record["summary_failure_reason"] is None
    assert record["summary_failure_count"] == 0
    assert record["auto_summary"] == "Text survives as an unacknowledged summary."
    assert record["summary"] == "Operator text"


def test_pre_v6_import_redaction_initializes_new_summary_state(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({"type": "_meta", "format": "hymem-jsonl", "version": 1}) + "\n" +
                    json.dumps({"type": "session", "record": {"id": "old", "summary": "Old operator text"}}) + "\n")
    with closing(HyMem(HyMemConfig(root=tmp_path / "dst", redact_secrets=True))) as dst:
        dst.import_(path)
        row = dst.conn.execute("SELECT * FROM sessions WHERE id='old'").fetchone()
        assert row["summary"] == "Old operator text"
        assert row["digest_published_message_id"] is None
        assert row["auto_summary_generation"] is None
        assert row["summary_failure_count"] == 0


@pytest.mark.parametrize("mode", ["item_behind_summary", "missing_ack_field", "private_authority"])
def test_v18_rejects_cross_field_and_private_authority_forgery(tmp_path, mode):
    with closing(HyMem(HyMemConfig(root=tmp_path / "src"))) as src:
        first, _ = _seed(src)
        path = tmp_path / "forged-contract.jsonl"
        src.export(path)

    def mutate(body):
        if mode == "item_behind_summary":
            _record(body)["digest_published_message_id"] = first
        elif mode == "missing_ack_field":
            _record(body).pop("auto_summary_generation")
        else:
            body.append({"type": "summary_recovery", "record": {"session_id": "x", "draft": "trust me"}})

    _rewrite(path, mutate)
    with closing(HyMem(HyMemConfig(root=tmp_path / "dst"))) as dst:
        with pytest.raises(ValueError):
            dst.import_(path)
        assert dst.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_v18_stale_valid_summary_generation_stays_degraded(tmp_path):
    with closing(HyMem(HyMemConfig(root=tmp_path / "src"))) as src:
        _seed(src)
        src.conn.execute("UPDATE sessions SET auto_summary_generation=? WHERE id='x'", (_generation("b"),))
        path = tmp_path / "stale.jsonl"
        src.export(path)
    with closing(HyMem(HyMemConfig(root=tmp_path / "dst"))) as dst:
        dst.import_(path)
        assert classify_summary_state(dst.conn, "x") == dict(
            summary_healthy=False, degraded=True, missing=False, malformed=False,
        )
        assert dst.conn.execute("SELECT auto_summary_generation FROM sessions WHERE id='x'").fetchone()[0] == _generation("b")


@pytest.mark.parametrize("mutation", ["unproved_frontier", "invalid_generation"])
def test_export_does_not_publish_malformed_summary_or_replace_backup(tmp_path, mutation):
    with closing(HyMem(HyMemConfig(root=tmp_path / "src"))) as src:
        _seed(src)
        if mutation == "unproved_frontier":
            src.conn.execute("UPDATE sessions SET digest_published_message_id=99999 WHERE id='x'")
        else:
            src.conn.execute("UPDATE sessions SET auto_summary_generation='unrecognized' WHERE id='x'")
        path = tmp_path / "existing-backup.jsonl"
        path.write_text("KEEP EXISTING BACKUP")
        with pytest.raises(ValueError):
            src.export(path)
        assert path.read_text() == "KEEP EXISTING BACKUP"
