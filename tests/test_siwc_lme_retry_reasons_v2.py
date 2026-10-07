"""Synthetic SQLite and privacy controls for the stopped-run v2 census."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest


SOURCE = Path(__file__).parents[1] / "tools/diagnostics/siwc_lme_retry_reasons_v2.py"
spec = importlib.util.spec_from_file_location("siwc_retry_reasons_v2", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _fixture(tmp_path):
    root = tmp_path / "stopped"
    for label, triples in module.EXPECTED.items():
        directory = root / "run" / label
        directory.mkdir(parents=True)
        path = directory / "hymem.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE chunk_extraction_attempts("
                               "chunk_id TEXT, attempts, last_failure_reason TEXT COLLATE NOCASE, "
                               "last_failure_details TEXT, private_payload TEXT)")
            rows = []
            for attempt, (grounding, branch) in enumerate(triples, 1):
                for index in range(grounding):
                    rows.append((f"private-{label}-{attempt}-g-{index}", attempt,
                                 "grounding_failure", json.dumps([
                                     "grounding:verdict_unsupported"]), "private payload"))
                for index in range(branch):
                    rows.append((f"private-{label}-{attempt}-b-{index}", attempt,
                                 "branch_incomplete", json.dumps([
                                     "left:grounding_failure",
                                     "left.grounding:contract_response_shape"]), "private payload"))
            connection.executemany("INSERT INTO chunk_extraction_attempts VALUES(?,?,?,?,?)", rows)
    return root


def _remote(root, *, gate=True, expected=None):
    # Stub only the pinned metadata gate; run the real v2 SQLite code.
    reader = f"ROOT_UID={os.getuid()}\n"
    projection = ("def _project(root, reader):\n"
                  + ("    return {'ok': True}\n" if gate else
                     "    raise ValueError('gate_failed')\n")
                  + "def _validated(value):\n    return value if value == {'ok': True} else None\n")
    metadata = "PROJECTION = " + repr(projection)
    scope = {"READER_SOURCE": reader, "METADATA_SOURCE": metadata,
             "ROOT": str(root), "SCHEMA": module.SCHEMA,
             "REASONS": module.REASONS, "MAX_ROWS": module.MAX_ROWS,
             "MAX_DETAILS_CHARS": module.MAX_DETAILS_CHARS,
             "MAX_DETAILS_ITEMS": module.MAX_DETAILS_ITEMS,
             "MAX_DETAIL_CHARS": module.MAX_DETAIL_CHARS,
             "MAX_OUTPUT_BYTES": module.MAX_OUTPUT_BYTES,
             "EXPECTED": module.EXPECTED if expected is None else expected,
             "CODES": module.CODES, "CATEGORIES": module.CATEGORIES,
             "__name__": "test_remote"}
    exec(module.REMOTE, scope)


def _change(root, statement, values):
    with sqlite3.connect(root / "run/q-0000/hymem.sqlite") as connection:
        connection.execute(statement, values)


def test_exact_census_branch_wrapping_and_database_unchanged(tmp_path, capsys):
    root = _fixture(tmp_path)
    db = root / "run/q-0000/hymem.sqlite"
    before = db.read_bytes()
    _remote(root)
    output = capsys.readouterr().out
    assert "private" not in output and "left." not in output
    result = json.loads(output)
    assert module._validated(result) == result
    assert result["v1_histogram_reconciled"] is True
    assert result["questions"]["q-0000"] == {
        "rows": 30,
        "categories": {"grounding:verdict_unsupported": 29,
                       "grounding:contract_response_shape": 1}}
    assert result["questions"]["q-0001"]["categories"] == {
        "grounding:verdict_unsupported": 21,
        "grounding:contract_response_shape": 3}
    assert db.read_bytes() == before
    assert not Path(str(db) + "-wal").exists()


def test_distinct_codes_per_row_and_unknown_never_exported(tmp_path, capsys):
    root = _fixture(tmp_path)
    _change(root, "UPDATE chunk_extraction_attempts SET last_failure_details=? WHERE chunk_id=?",
            (json.dumps(["grounding:verdict_uncertain", "grounding:verdict_uncertain",
                         "grounding:contract_evidence_quote_missing",
                         "grounding:secret_private_unknown"]),
             "private-q-0000-1-g-0"))
    _remote(root)
    output = capsys.readouterr().out
    assert "secret_private_unknown" not in output
    counts = json.loads(output)["questions"]["q-0000"]["categories"]
    assert counts["grounding:verdict_uncertain"] == 1
    assert counts["grounding:contract_evidence_quote_missing"] == 1
    assert counts["other"] == 1
    assert counts["grounding:verdict_unsupported"] == 28


@pytest.mark.parametrize("detail", [
    "not JSON", "{}", "[1]", json.dumps(["x" * 161]),
    json.dumps(["x"] * 33), "x" * 8193,
    "\x00" + "x" * 8193,
    sqlite3.Binary(b'["grounding:verdict_uncertain"]'),
])
def test_invalid_detail_shapes_are_fixed_category(tmp_path, capsys, detail):
    root = _fixture(tmp_path)
    _change(root, "UPDATE chunk_extraction_attempts SET last_failure_details=? WHERE chunk_id=?",
            (detail, "private-q-0000-1-g-0"))
    _remote(root)
    output = capsys.readouterr().out
    if type(detail) is str:
        assert detail[:30] not in output
    counts = json.loads(output)["questions"]["q-0000"]["categories"]
    assert counts["invalid_details"] == 1
    assert counts["grounding:verdict_unsupported"] == 28


def test_nested_branch_and_truncation(tmp_path, capsys):
    root = _fixture(tmp_path)
    _change(root, "UPDATE chunk_extraction_attempts SET last_failure_details=? WHERE chunk_id=?",
            (json.dumps(["right.left.grounding:verdict_uncertain",
                         "diagnostics:truncated"]), "private-q-0000-3-b-0"))
    _remote(root)
    counts = json.loads(capsys.readouterr().out)["questions"]["q-0000"]["categories"]
    assert counts["grounding:verdict_uncertain"] == 1
    assert counts["truncated_details"] == 1
    assert "grounding:contract_response_shape" not in counts


def test_v1_histogram_mismatch_precedes_detail_read(tmp_path):
    root = _fixture(tmp_path)
    _change(root, "UPDATE chunk_extraction_attempts SET last_failure_reason=? WHERE chunk_id=?",
            ("call_failure", "private-q-0000-1-g-0"))
    with pytest.raises(ValueError, match="v1_histogram_changed"):
        _remote(root)


def test_wrong_attempt_and_sidecar_fail_closed(tmp_path):
    root = _fixture(tmp_path)
    _change(root, "UPDATE chunk_extraction_attempts SET attempts=? WHERE chunk_id=?",
            (4, "private-q-0000-1-g-0"))
    with pytest.raises(ValueError, match="invalid_retry_bucket"):
        _remote(root)
    root = _fixture(tmp_path / "another")
    sidecar = root / "run/q-0000/hymem.sqlite-wal"
    sidecar.write_bytes(b"nonempty")
    with pytest.raises(ValueError, match="sqlite_sidecar_nonempty"):
        _remote(root)


def test_gate_precedes_database_and_symlink_is_rejected(tmp_path):
    root = _fixture(tmp_path)
    database = root / "run/q-0000/hymem.sqlite"
    database.unlink()
    with pytest.raises(ValueError, match="gate_failed"):
        _remote(root, gate=False)
    root = _fixture(tmp_path / "another")
    database = root / "run/q-0000/hymem.sqlite"
    database.rename(tmp_path / "outside-db")
    database.symlink_to(tmp_path / "outside-db")
    with pytest.raises(ValueError, match="unsafe_path"):
        _remote(root)


def test_pinned_bootstrap_and_source_pins(tmp_path):
    payload = module._payload()
    assert "mode=ro&immutable=1" in payload
    assert "last_failure_details" in payload
    assert "SELECT *" not in payload
    assert "private_payload" not in payload
    assert module._pinned(module.V1, module.V1_SHA)
    reader_source = module._pinned(module.READER, module.READER_SHA)
    metadata_source = module._pinned(module.METADATA, module.METADATA_SHA)
    reader = {"__name__": "pinned_progress", "__file__": "<pinned-progress>"}
    metadata = {"__name__": "pinned_metadata", "__file__": "<pinned-metadata>"}
    exec(compile(reader_source, "<pinned-progress>", "exec"), reader)
    exec(compile(metadata_source, "<pinned-metadata>", "exec"), metadata)
    exec(compile(metadata["PROJECTION"], "<pinned-metadata-projection>", "exec"), metadata)
    reader["inspect"] = lambda *args: {"status": "not_terminal"}
    with pytest.raises(ValueError, match="terminal_gate_invalid"):
        metadata["_project"](tmp_path, reader)


@pytest.mark.parametrize("remote", [
    '{"schema":"leak","private_id":"secret"}',
    '{"schema":"a","schema":"b"}', "NaN", "not json",
])
def test_main_never_echoes_untrusted_remote_data(remote, monkeypatch, capsys):
    monkeypatch.setattr(module, "_payload", lambda: "safe stub")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, remote, "private stderr secret"))
    assert module.main() == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {"schema": module.SCHEMA, "status": "reasons_unavailable"}
    assert "secret" not in output and "private" not in output


def test_output_validator_rejects_free_text_and_wrong_counts():
    good = {"schema": module.SCHEMA,
            "source_receipt_terminal_cleanup_verified": True,
            "v1_histogram_reconciled": True,
            "questions": {label: {"rows": sum(sum(pair) for pair in expected),
                                  "categories": {"no_grounding_code": sum(sum(pair) for pair in expected)}}
                          for label, expected in module.EXPECTED.items()}}
    assert module._validated(good) == good
    good["questions"]["q-0000"]["categories"]["private-free-text"] = 1
    assert module._validated(good) is None
    del good["questions"]["q-0000"]["categories"]["private-free-text"]
    good["questions"]["q-0000"]["rows"] += 1
    assert module._validated(good) is None


def test_literal_taxonomy_covers_five_source_modules():
    extraction = SOURCE.parents[2] / "hymem/extraction"
    for name in ("grounding_staged_gate_v1", "grounding_staged_v1",
                 "grounding_classification_v4", "grounding_v2", "grounding_gate"):
        tree = ast.parse((extraction / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
                    or node.func.id != "_fail" or not node.args
                    or not isinstance(node.args[0], ast.Constant)
                    or type(node.args[0].value) is not str
                    or ":" not in node.args[0].value):
                continue
            code = node.args[0].value.replace(":", "_")
            if name not in {"grounding_staged_gate_v1", "grounding_gate"}:
                code = "contract_" + code
            assert "grounding:" + code in module.CODES
