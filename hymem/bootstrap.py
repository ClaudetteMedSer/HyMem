"""Zero-config startup: build a HyMem instance from environment variables.

This is the single source of truth for environment-variable resolution. Both
entry points (`hymem-server`, `hymem-honcho`) and `hymem-doctor` build on it,
so configuration behaviour stays consistent across every surface.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from hymem.api import HyMem
from hymem.config import HyMemConfig

log = logging.getLogger("hymem.bootstrap")

DEFAULT_ROOT = Path.home() / ".hermes"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-chat"
DEFAULT_EMBEDDING_MODEL = "deepseek-embedding"


@dataclass(frozen=True)
class EnvConfig:
    """Environment-resolved configuration. ``*_api_key`` is None when absent."""

    root: Path
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    embedding_api_key: str | None
    embedding_base_url: str
    embedding_model: str

    @property
    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def has_embedding_key(self) -> bool:
        return bool(self.embedding_api_key)


def resolve_env() -> EnvConfig:
    """Resolve all HyMem configuration from the environment.

    Never raises and never constructs network clients — safe for the doctor
    to call to report what *would* be used.
    """
    env = os.environ.get
    llm_key = env("HYMEM_LLM_API_KEY") or env("DEEPSEEK_API_KEY") or env("OPENAI_API_KEY")
    embedding_key = (
        env("HYMEM_EMBEDDING_API_KEY")
        or env("HYMEM_LLM_API_KEY")
        or env("DEEPSEEK_API_KEY")
        or env("OPENAI_API_KEY")
    )
    return EnvConfig(
        root=Path(env("HYMEM_ROOT", str(DEFAULT_ROOT))),
        llm_api_key=llm_key,
        llm_base_url=env("HYMEM_LLM_BASE_URL", DEFAULT_BASE_URL),
        llm_model=env("HYMEM_LLM_MODEL", DEFAULT_LLM_MODEL),
        embedding_api_key=embedding_key,
        embedding_base_url=env("HYMEM_EMBEDDING_BASE_URL", DEFAULT_BASE_URL),
        embedding_model=env("HYMEM_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )


def build_from_env() -> HyMem:
    """Construct a HyMem instance from environment variables.

    Fails fast with a clear, actionable error if the extraction LLM key is
    missing — instead of raising deep inside the first dream cycle. The
    embedding client is optional: when its key is absent the server logs a
    warning and falls back to FTS-only retrieval.
    """
    from hymem.contrib.openai_client import OpenAICompatibleClient
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient

    cfg = resolve_env()

    if not cfg.has_llm_key:
        raise RuntimeError(
            "HyMem cannot start: no extraction LLM API key found.\n"
            "Set HYMEM_LLM_API_KEY (or DEEPSEEK_API_KEY / OPENAI_API_KEY) "
            "before launching the server.\n"
            "Run `hymem-doctor` to diagnose your configuration."
        )

    llm = OpenAICompatibleClient(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
    )

    embedder = None
    if cfg.has_embedding_key:
        try:
            from hymem.extraction.embeddings import CachedEmbeddingClient
            embedder = CachedEmbeddingClient(
                OpenAICompatibleEmbeddingClient(
                    api_key=cfg.embedding_api_key,
                    base_url=cfg.embedding_base_url,
                    model=cfg.embedding_model,
                )
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            log.warning("embeddings disabled (FTS-only retrieval): %s", exc)
    else:
        log.warning(
            "no embedding API key found — FTS-only retrieval "
            "(set HYMEM_EMBEDDING_API_KEY to enable vector search)"
        )

    return HyMem(HyMemConfig(root=cfg.root), llm=llm, embedding_client=embedder)


# ── shared singleton ─────────────────────────────────────────────────────────
# Both server entry points and the test/integration harness go through these,
# so there is exactly one HyMem instance per process unless explicitly injected.

_instance: HyMem | None = None


def get_instance() -> HyMem:
    # Not locked: both server entry points initialize this during
    # single-threaded startup, before any request thread exists.
    global _instance
    if _instance is None:
        _instance = build_from_env()
    return _instance


def set_instance(instance: HyMem) -> None:
    """Inject a pre-built HyMem (used by tests and integration harnesses)."""
    global _instance
    _instance = instance
