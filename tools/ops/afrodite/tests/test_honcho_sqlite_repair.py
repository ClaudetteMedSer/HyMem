"""Local-only checks for the exact remote source transformer."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "honcho_sqlite_repair.py"
_SPEC = importlib.util.spec_from_file_location("honcho_sqlite_repair", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
repair = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(repair)


def _remote_transform(original: str):
    """Execute the actual remote pure transform functions, without its CLI."""
    names = {"need", "sha_bytes", "definition", "replace_one", "transform"}
    nodes = [
        node for node in ast.parse(repair.REMOTE).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    namespace = {"ast": ast, "hashlib": hashlib,
                 "EXPECTED": hashlib.sha256(original.encode()).hexdigest()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<remote-transform>", "exec"), namespace)
    return namespace["transform"]


def _head_db() -> str:
    return subprocess.run(
        ["git", "show", f"HEAD:{repair.DB_REL}"], cwd=repair.REPO,
        text=True, capture_output=True, check=True,
    ).stdout


def test_remote_transform_changes_only_three_pinned_spans() -> None:
    old = _head_db()
    payload = repair.candidate_payload("/home/node/.hermes/hymem.sqlite")
    result = _remote_transform(old)(old, payload)
    expected = old.replace(
        payload["original_transaction"], payload["candidate_transaction"], 1,
    ).replace(
        "from hymem.deadline import check_current_deadline\n",
        "from hymem.deadline import check_current_deadline\n"
        "from hymem.core.serialized_sqlite import SerializedConnection, operation_scope\n",
        1,
    ).replace(
        "        cached_statements=0 if sys.version_info >= (3, 12) else 128,\n",
        "        cached_statements=0 if sys.version_info >= (3, 12) else 128,\n"
        "        factory=SerializedConnection,\n",
        1,
    )
    assert result == expected
    assert repair.definition(result, "transaction") == payload["candidate_transaction"]


def test_remote_transform_rejects_changed_function_and_duplicate_anchor() -> None:
    old = _head_db()
    payload = repair.candidate_payload("/home/node/.hermes/hymem.sqlite")
    transform = _remote_transform(old)
    changed = dict(payload, original_transaction=payload["original_transaction"] + "# drift\n")
    with pytest.raises(RuntimeError, match="transaction_source_changed"):
        transform(old, changed)
    duplicated = old.replace(
        "from hymem.deadline import check_current_deadline\n",
        "from hymem.deadline import check_current_deadline\n" * 2,
        1,
    )
    duplicate_transform = _remote_transform(duplicated)
    with pytest.raises(RuntimeError, match="import_match_count"):
        duplicate_transform(duplicated, payload)
