"""Keep public runtime-default guidance tied to executable configuration.

Historical ledgers deliberately preserve the defaults and model aliases used
by their dated experiments.  This test covers only current operator surfaces;
it avoids matching whole prose blocks so ordinary copy edits stay cheap.
"""

from __future__ import annotations

import re
from pathlib import Path

from hymem.bootstrap import DEFAULT_LLM_MODEL, resolve_env
from hymem.config import HyMemConfig


_ROOT = Path(__file__).resolve().parents[1]


def _markdown_default(document: str, setting: str) -> str:
    """Return the Default cell for one exact Markdown configuration row."""

    prefix = f"| `{setting}` |"
    rows = [line for line in document.splitlines() if line.startswith(prefix)]
    assert len(rows) == 1, f"expected one public row for {setting}, got {len(rows)}"
    return rows[0].split("|")[2].strip()


def _clear_runtime_env(monkeypatch) -> None:
    for name in (
        "HYMEM_ROOT",
        "HYMEM_LLM_API_KEY",
        "HYMEM_LLM_BASE_URL",
        "HYMEM_LLM_MODEL",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "HYMEM_EMBEDDING_API_KEY",
        "HYMEM_EMBEDDING_BASE_URL",
        "HYMEM_EMBEDDING_MODEL",
        "HYMEM_EMBEDDING_DIM",
        "HYMEM_AGGREGATION_NODES_ENABLED",
        "HYMEM_AGGREGATION_DIGEST_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


def test_readme_defaults_follow_code(tmp_path):
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    config = HyMemConfig(root=tmp_path)
    master_word = "on" if config.aggregation_nodes_enabled else "off"
    digest_word = "on" if config.aggregation_digest_enabled else "off"

    assert _markdown_default(readme, "HYMEM_LLM_MODEL") == (
        f"`{DEFAULT_LLM_MODEL}`"
    )
    assert _markdown_default(readme, "HYMEM_AGGREGATION_NODES_ENABLED") == (
        f"unset (config default: {master_word})"
    )
    assert _markdown_default(readme, "HYMEM_AGGREGATION_DIGEST_ENABLED") == (
        f"unset (config default: {digest_word})"
    )
    assert _markdown_default(readme, "aggregation_nodes_enabled") == (
        f"`{config.aggregation_nodes_enabled}`"
    )
    assert _markdown_default(readme, "aggregation_digest_enabled") == (
        f"`{config.aggregation_digest_enabled}`"
    )


def test_unset_env_defers_to_dataclass_and_server_help_matches(
    monkeypatch, tmp_path,
):
    _clear_runtime_env(monkeypatch)
    env_config = resolve_env()
    code_config = HyMemConfig(root=tmp_path)
    server_source = (_ROOT / "hymem" / "server.py").read_text(encoding="utf-8")
    bootstrap_source = (_ROOT / "hymem" / "bootstrap.py").read_text(
        encoding="utf-8"
    )

    # Unset env values remain None until build_from_env applies only explicit
    # overrides; the dataclass is the single executable source of truth.
    assert env_config.llm_model == DEFAULT_LLM_MODEL
    assert env_config.aggregation_nodes_enabled is None
    assert env_config.aggregation_digest_enabled is None
    assert "HyMemConfig dataclass as the authoritative default" in bootstrap_source

    master_word = "on" if code_config.aggregation_nodes_enabled else "off"
    assert f"Model name (default: {DEFAULT_LLM_MODEL})" in server_source
    assert f"time (default: {master_word}). Set false to opt out." in server_source
    assert "aggregation layer enabled via env" not in bootstrap_source


def test_current_operator_surfaces_only_name_retired_aliases_as_rejected():
    for relative in ("README.md", "hymem/server.py"):
        text = (_ROOT / relative).read_text(encoding="utf-8")
        for match in re.finditer(r"deepseek-(?:chat|reasoner)", text):
            neighborhood = text[max(0, match.start() - 120):match.end() + 160].lower()
            assert "retired" in neighborhood
            assert "reject" in neighborhood

    migration = (
        _ROOT / "references" / "deepseek-model-migration.md"
    ).read_text(encoding="utf-8")
    assert "lower-tier screening" in migration
    assert "require_active_model" in migration
    assert "explicit, immutable model" in migration
