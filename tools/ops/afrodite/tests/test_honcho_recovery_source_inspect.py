"""Local-only tests of stopped-Honcho source metadata inspection."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shlex


MODULE = Path(__file__).resolve().parents[1] / "honcho_recovery_source_inspect.py"
SPEC = importlib.util.spec_from_file_location("honcho_recovery_source_inspect", MODULE)
assert SPEC is not None and SPEC.loader is not None
inspect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspect)


def _functions(*names):
    selected = set(names) | {"sha"}
    nodes = [node for node in ast.parse(inspect.REMOTE).body
             if isinstance(node, ast.FunctionDef) and node.name in selected]
    assert {node.name for node in nodes} == selected
    namespace = {"hashlib": hashlib, "os": os, "re": re, "shlex": shlex,
                 "REQUIRED": ("HYMEM_LLM_API_KEY", "HYMEM_LLM_MODEL")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<remote-metadata>", "exec"), namespace)
    return namespace


def test_recorded_command_hash_has_one_exact_whitelisted_match() -> None:
    match = _functions("argv_match")["argv_match"]
    found = match("664846cdaa6d7ee5e16e2403a0a18461a004c27b676796b44ef340237f1c7429")
    assert found == [{"shape": "console:/home/node/hymem-env/bin/python3",
                      "argc": 2, "trailing_nul": True}]
    assert match("0" * 64) == []


def test_static_env_parser_reports_structure_without_values() -> None:
    parse = _functions("parse_static_env")["parse_static_env"]
    source = "# comment\nexport HYMEM_LLM_MODEL='deepseek-flash'\nHYMEM_LLM_API_KEY='private'\nDYNAMIC=$HOME\nBAD line\n"
    values, metadata = parse(source)
    assert values["HYMEM_LLM_API_KEY"] == "private"
    assert metadata["required_present"] == {
        "HYMEM_LLM_API_KEY": True, "HYMEM_LLM_MODEL": True,
    }
    assert metadata["dynamic_lines"] == 1
    assert metadata["unparsed_lines"] == 1
    assert "private" not in str(metadata)
