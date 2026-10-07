"""Local-only structural parsing checks; no source values in metadata."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re


MODULE = Path(__file__).resolve().parents[1] / "honcho_recovery_launch_structure.py"
SPEC = importlib.util.spec_from_file_location("honcho_recovery_launch_structure", MODULE)
assert SPEC is not None and SPEC.loader is not None
structure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(structure)


def _parser():
    nodes = [node for node in ast.parse(structure.REMOTE).body
             if isinstance(node, ast.FunctionDef)
             and node.name in ("classify_value", "classify_line")]
    assert len(nodes) == 2
    namespace = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<remote-structure>", "exec"), namespace)
    return namespace["classify_line"]


def test_export_assignment_retains_assignment_kind_and_hides_value() -> None:
    classify = _parser()
    item = classify('export HYMEM_LLM_API_KEY="secret-$DEEPSEEK_API_KEY"', 77)
    assert item["kind"] == "export_assignment"
    assert item["value_kind"] == "variable_reference"
    assert item["name"] == "HYMEM_LLM_API_KEY"
    assert item["references"] == ["DEEPSEEK_API_KEY"]
    assert "secret" not in str(item)


def test_nohup_line_returns_structure_only() -> None:
    classify = _parser()
    item = classify('nohup "$VENV/bin/hymem-honcho" > "$PRIVATE_LOG" 2>&1 &', 92)
    assert item == {"line": 92, "kind": "nohup",
                    "references": ["PRIVATE_LOG", "VENV"],
                    "has_redirection": True, "background": True,
                    "has_command_substitution": False}
    assert "hymem-honcho" not in str(item)
