"""Offline, synthetic-database controls for the stopped retry census."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest


SOURCE = Path(__file__).parents[1] / "tools/diagnostics/siwc_lme_retry_reasons_v1.py"
spec = importlib.util.spec_from_file_location("siwc_retry_reasons", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _database(path: Path, rows=()):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE chunk_extraction_attempts("
                     "chunk_id TEXT, attempts, last_failure_reason TEXT COLLATE NOCASE, "
                     "last_failure_details TEXT)")
        conn.executemany("INSERT INTO chunk_extraction_attempts VALUES(?,?,?,?)", rows)


def _fixture(tmp_path, rows=()):
    root = tmp_path / "stopped"
    for index in range(4):
        directory = root / "run" / f"q-{index:04d}"
        directory.mkdir(parents=True)
        _database(directory / "hymem.sqlite", rows if index == 0 else ())
    return root


def _remote(root, *, gate=True):
    # Only the pinned-gate result is replaced; the actual census code and
    # four actual SQLite files execute unchanged.
    reader = f"ROOT_UID={os.getuid()}\n"
    projection = ("def _project(root, reader):\n"
                  + ("    return {'ok': True}\n" if gate else "    raise ValueError('gate_failed')\n")
                  + "def _validated(value):\n    return value if value == {'ok': True} else None\n")
    metadata = "PROJECTION = " + repr(projection)
    scope = {"READER_SOURCE": reader, "METADATA_SOURCE": metadata,
             "ROOT": str(root), "SCHEMA": module.SCHEMA,
             "REASONS": module.REASONS, "MAX_ROWS": module.MAX_ROWS,
             "__name__": "test_remote"}
    exec(module.REMOTE, scope)


def test_histogram_is_finite_and_database_unchanged(tmp_path, capsys):
    rows = [("private-id", 1, "parse_failure", "secret details"),
            ("private-id-2", 2, "call_failure", "secret details"),
            ("private-id-3", 3, "unexpected private reason", "secret details"),
            ("private-id-4", 1, None, "secret details"),
            ("private-id-5", 2, "PARSE_FAILURE", "secret details"),
            ("private-id-6", 2, b"parse_failure", "secret details")]
    root = _fixture(tmp_path, rows)
    database = root / "run/q-0000/hymem.sqlite"
    before = database.read_bytes()
    _remote(root)
    output = capsys.readouterr().out
    assert "private-id" not in output and "secret" not in output
    assert "unexpected private reason" not in output
    result = json.loads(output)
    assert module._validated(result) == result
    bucket = result["questions"]["q-0000"]
    assert bucket["rows"] == 6
    assert bucket["by_attempt"]["1"]["parse_failure"] == 1
    assert bucket["by_attempt"]["1"]["other"] == 1
    assert bucket["by_attempt"]["2"]["call_failure"] == 1
    assert bucket["by_attempt"]["2"]["parse_failure"] == 0
    assert bucket["by_attempt"]["2"]["other"] == 2
    assert bucket["by_attempt"]["3"]["other"] == 1
    assert database.read_bytes() == before
    assert not Path(str(database) + "-wal").exists()


@pytest.mark.parametrize("attempt", [0, 4, "1", None, 1.0, b"1"])
def test_invalid_attempt_fails_closed(tmp_path, attempt):
    root = _fixture(tmp_path, [("private", attempt, "parse_failure", "secret")])
    with pytest.raises(ValueError, match="invalid_retry_bucket"):
        _remote(root)


@pytest.mark.parametrize("suffix", ["-wal", "-journal", "-shm"])
def test_nonempty_sidecar_fails_closed(tmp_path, suffix):
    root = _fixture(tmp_path)
    sidecar = root / "run/q-0000" / ("hymem.sqlite" + suffix)
    sidecar.write_bytes(b"not safe to ignore")
    with pytest.raises(ValueError, match="sqlite_sidecar_nonempty"):
        _remote(root)


def test_empty_sidecar_is_allowed(tmp_path, capsys):
    root = _fixture(tmp_path)
    (root / "run/q-0000/hymem.sqlite-wal").touch()
    _remote(root)
    assert module._validated(json.loads(capsys.readouterr().out)) is not None


def test_symlink_and_extra_link_fail_closed(tmp_path):
    root = _fixture(tmp_path)
    database = root / "run/q-0000/hymem.sqlite"
    link = tmp_path / "extra-link"
    os.link(database, link)
    with pytest.raises(ValueError, match="unsafe_path"):
        _remote(root)
    link.unlink()
    database.rename(tmp_path / "real-db")
    database.symlink_to(tmp_path / "real-db")
    with pytest.raises(ValueError, match="unsafe_path"):
        _remote(root)


def test_gate_precedes_database_access(tmp_path):
    root = _fixture(tmp_path)
    (root / "run/q-0000/hymem.sqlite").unlink()
    with pytest.raises(ValueError, match="gate_failed"):
        _remote(root, gate=False)


def test_actual_pinned_bootstrap_exposes_projector_and_terminal_gate(tmp_path):
    reader_source = module._pinned(module.READER, module.READER_SHA)
    metadata_source = module._pinned(module.METADATA, module.METADATA_SHA)
    reader = {"__name__": "pinned_progress", "__file__": "<pinned-progress>"}
    metadata = {"__name__": "pinned_metadata", "__file__": "<pinned-metadata>"}
    exec(compile(reader_source, "<pinned-progress>", "exec"), reader)
    exec(compile(metadata_source, "<pinned-metadata>", "exec"), metadata)
    assert "_project" not in metadata
    exec(compile(metadata["PROJECTION"], "<pinned-metadata-projection>", "exec"), metadata)
    assert callable(metadata["_project"])
    reader["inspect"] = lambda *args: {"status": "not_terminal"}
    with pytest.raises(ValueError, match="terminal_gate_invalid"):
        metadata["_project"](tmp_path, reader)


def test_missing_database_or_column_fails_closed(tmp_path):
    root = _fixture(tmp_path)
    (root / "run/q-0000/hymem.sqlite").unlink()
    with pytest.raises(FileNotFoundError):
        _remote(root)
    database = root / "run/q-0000/hymem.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE chunk_extraction_attempts(attempts INTEGER)")
    with pytest.raises(sqlite3.DatabaseError):
        _remote(root)


def test_duplicate_json_and_privacy_validation():
    with pytest.raises(ValueError, match="duplicate_projection_field"):
        json.loads('{"schema":"a","schema":"b"}', object_pairs_hook=module._unique_fields)
    assert module._validated({"schema": module.SCHEMA, "questions": {},
                              "source_receipt_terminal_cleanup_verified": True}) is None
    good = {"schema": module.SCHEMA,
            "source_receipt_terminal_cleanup_verified": True,
            "questions": {f"q-{i:04d}": {"rows": 0, "by_attempt": {
                str(attempt): {reason: 0 for reason in module.REASONS}
                for attempt in (1, 2, 3)}} for i in range(4)}}
    assert module._validated(good) is not None
    good["questions"]["q-0000"]["private_id"] = "leak"
    assert module._validated(good) is None


def test_local_source_pins_and_ssh_options():
    payload = module._payload()
    assert "mode=ro&immutable=1" in payload
    assert module.ROOT in payload
    assert "last_failure_details" not in module.REMOTE
    assert "BatchMode=yes" in SOURCE.read_text()
    assert "ConnectTimeout=10" in SOURCE.read_text()
    assert "ConnectionAttempts=1" in SOURCE.read_text()


def test_source_pin_drift_fails_before_ssh(tmp_path, monkeypatch):
    altered = tmp_path / "altered.py"
    altered.write_text("pass\n")
    monkeypatch.setattr(module, "METADATA", altered)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("SSH attempted"))
    with pytest.raises(ValueError, match="source_pin_mismatch"):
        module._payload()


@pytest.mark.parametrize("remote", [
    '{"schema":"leak","private_id":"secret"}',
    '{"schema":"a","schema":"b"}',
    'NaN',
    'not json',
])
def test_main_rejects_untrusted_stdout_without_echo(remote, monkeypatch, capsys):
    monkeypatch.setattr(module, "_payload", lambda: "safe stub")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, remote, "private stderr secret"))
    assert module.main() == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {"schema": module.SCHEMA, "status": "reasons_unavailable"}
    assert "secret" not in output and "private" not in output


def test_row_bound_fails_closed(tmp_path, monkeypatch):
    root = _fixture(tmp_path, [("private-a", 1, "parse_failure", "secret"),
                               ("private-b", 1, "parse_failure", "secret")])
    monkeypatch.setattr(module, "MAX_ROWS", 1)
    with pytest.raises(ValueError, match="invalid_retry_bucket"):
        _remote(root)
