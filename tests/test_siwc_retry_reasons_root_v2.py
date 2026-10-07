"""Independent offline controls for the finite stopped-run v2 projection."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = Path(__file__).parents[1]
SUPPORT = _load("root_support_retry_v2", BASE / "tests/test_siwc_lme_retry_reasons_v2.py")
M = SUPPORT.module
V1 = _load("root_retry_v1", BASE / "tools/diagnostics/siwc_lme_retry_reasons_v1.py")


def _definitions():
    tree = ast.parse(M.REMOTE)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    scope = {"json": json, "CODES": M.CODES, "MAX_DETAIL_CHARS": M.MAX_DETAIL_CHARS,
             "MAX_DETAILS_CHARS": M.MAX_DETAILS_CHARS, "MAX_DETAILS_ITEMS": M.MAX_DETAILS_ITEMS}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "<root-offline-functions>", "exec"), scope)
    return scope


def test_all_categories_exact_mapping_and_closed_fallbacks():
    classify = _definitions()["_categories"]
    assert len(M.CATEGORIES) == len(set(M.CATEGORIES))
    for code in M.CODES:
        for prefix in ("", "left.", "right.left.", "left.right.left.right."):
            assert classify(json.dumps([prefix + code, prefix + code])) == {code}
        for bad in (code.upper(), code + "_SECRET", code + "\x00SECRET",
                    "left." * 5 + code, "SECRET" + code):
            result = classify(json.dumps([bad]))
            assert result <= set(M.FALLBACKS)
            assert "SECRET" not in json.dumps(sorted(result))
    for raw in (None, b"[]", 1, True, "null", "true", "{}", "[[]]", "[NaN]",
                '"grounding:verdict_unsupported"', "[" * 4000, "[]SECRET"):
        assert classify(raw) == {"invalid_details"}
    assert classify("[]") == {"no_grounding_code"}
    assert classify(json.dumps(["left:grounding_failure"])) == {"no_grounding_code"}
    assert classify(json.dumps(["grounding:verdict_uncertain", "diagnostic:invalid"])) == {
        "grounding:verdict_uncertain", "invalid_details"}


def test_preserves_v1_filesystem_and_digest_controls():
    def functions(source):
        return {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    old, new = functions(V1.REMOTE), functions(M.REMOTE)
    for name in ("_stat", "_identity", "_digest", "_sidecars"):
        assert old[name] == new[name]
    assert M._pinned(M.V1, M.V1_SHA)
    assert M._pinned(M.METADATA, M.METADATA_SHA)
    assert M._pinned(M.READER, M.READER_SHA)


def test_authorizer_restricts_columns_and_writes(tmp_path, monkeypatch, capsys):
    root = SUPPORT._fixture(tmp_path)
    authorizers = []
    connect = sqlite3.connect

    class Observed(sqlite3.Connection):
        def set_authorizer(self, callback):
            authorizers.append(callback)
            return super().set_authorizer(callback)

    def observed_connect(address, *args, **kwargs):
        assert address.endswith("?mode=ro&immutable=1")
        assert kwargs["uri"] is True
        return connect(address, *args, factory=Observed, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", observed_connect)
    SUPPORT._remote(root)
    value = json.loads(capsys.readouterr().out)
    assert M._validated(value)
    assert len(authorizers) == 4
    for callback in authorizers:
        for column in ("attempts", "last_failure_reason", "last_failure_details"):
            assert callback(sqlite3.SQLITE_READ, "chunk_extraction_attempts", column, "main", None) == sqlite3.SQLITE_OK
        for action, arg1, arg2, database in (
            (sqlite3.SQLITE_READ, "chunk_extraction_attempts", "chunk_id", "main"),
            (sqlite3.SQLITE_READ, "chunk_extraction_attempts", "private_payload", "main"),
            (sqlite3.SQLITE_READ, "messages", "content", "main"),
            (sqlite3.SQLITE_READ, "chunk_extraction_attempts", "attempts", "other"),
            (sqlite3.SQLITE_UPDATE, "chunk_extraction_attempts", "attempts", "main"),
            (sqlite3.SQLITE_DELETE, "chunk_extraction_attempts", None, "main"),
            (sqlite3.SQLITE_ATTACH, "secret", None, None),
            (sqlite3.SQLITE_PRAGMA, "journal_mode", "WAL", None),
            (sqlite3.SQLITE_FUNCTION, None, "load_extension", None),
        ):
            assert callback(action, arg1, arg2, database, None) == sqlite3.SQLITE_DENY


def test_detail_sql_guard_covers_nul_unicode_and_blobs(tmp_path, capsys):
    root = SUPPORT._fixture(tmp_path)
    db = root / "run/q-0000/hymem.sqlite"
    query = _definitions()["_details_query"]()
    for text in ("[]\x00" + "s" * 9000, "ø" * 5000, b"[]", None):
        with sqlite3.connect(db) as connection:
            connection.execute("UPDATE chunk_extraction_attempts SET last_failure_details=?", (text,))
            assert all(row == (None,) for row in connection.execute(query))
        SUPPORT._remote(root)
        result = json.loads(capsys.readouterr().out)
        assert result["questions"]["q-0000"]["categories"] == {"invalid_details": 30}


def test_external_projection_cannot_introduce_labels_or_counts(tmp_path, capsys):
    SUPPORT._remote(SUPPORT._fixture(tmp_path))
    good = json.loads(capsys.readouterr().out)
    bads = []
    for key in ("schema", "source_receipt_terminal_cleanup_verified", "v1_histogram_reconciled"):
        bad = copy.deepcopy(good)
        bad[key] = "SECRET"
        bads.append(bad)
    for count in (True, 0, -1, 31, 1.0, "1", None):
        bad = copy.deepcopy(good)
        bad["questions"]["q-0000"]["categories"]["grounding:verdict_unsupported"] = count
        bads.append(bad)
    for key in ("SECRET", "grounding:verdict_unsupported_SECRET", "left.grounding:verdict_uncertain"):
        bad = copy.deepcopy(good)
        bad["questions"]["q-0000"]["categories"][key] = 1
        bads.append(bad)
    bad = copy.deepcopy(good)
    bad["questions"]["q-0000"]["categories"] = {}
    bads.append(bad)
    for bad in bads:
        assert M._validated(bad) is None


def test_actual_bootstrap_rejects_before_sqlite_on_invalid_gate(tmp_path, monkeypatch):
    scope = {"__name__": "pinned_root_gate", "__file__": "<pinned>"}
    exec(compile(M._pinned(M.METADATA, M.METADATA_SHA), "<metadata>", "exec"), scope)
    exec(compile(scope["PROJECTION"], "<projection>", "exec"), scope)
    reader = {"__name__": "pinned_reader", "__file__": "<pinned-reader>"}
    exec(compile(M._pinned(M.READER, M.READER_SHA), "<reader>", "exec"), reader)
    reader["inspect"] = lambda *args: {"status": "not_terminal"}
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("database opened before gate"))
    with pytest.raises(ValueError, match="terminal_gate_invalid"):
        scope["_project"](tmp_path, reader)
