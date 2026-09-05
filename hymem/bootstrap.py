"""Zero-config startup: build a HyMem instance from environment variables.

This is the single source of truth for environment-variable resolution. Both
entry points (`hymem-server`, `hymem-honcho`) and `hymem-doctor` build on it,
so configuration behaviour stays consistent across every surface.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from hymem.api import HyMem
from hymem.config import HyMemConfig

log = logging.getLogger("hymem.bootstrap")

DEFAULT_ROOT = Path.home() / ".hermes"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"  # deepseek-chat hard-deprecated 2026-07-24
DEFAULT_EMBEDDING_BASE_URL = "local://feature-hash"
DEFAULT_EMBEDDING_MODEL = "hymem-local-feature-hash-v1"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_REMOTE_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
DEFAULT_REMOTE_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_REMOTE_EMBEDDING_DIM = 1536


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
    embedding_dim: int
    embedding_backend: str
    embedding_fallback_reason: str | None
    # None = env var unset → fall back to the HyMemConfig dataclass default
    # (don't hard-code it here, so a future default change stays authoritative).
    aggregation_nodes_enabled: bool | None
    aggregation_digest_enabled: bool | None

    @property
    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def has_embedding_key(self) -> bool:
        return bool(self.embedding_api_key)

    @property
    def has_embedding_client(self) -> bool:
        return self.embedding_backend == "local_feature_hash" or self.has_embedding_key

    @property
    def embedding_identity(self) -> str:
        """Exact durable vector-space identity expected from this config."""
        if self.embedding_backend != "openai_compatible":
            return self.embedding_model
        from hymem.contrib.openai_embedding_client import (
            openai_compatible_embedding_identity,
        )
        return openai_compatible_embedding_identity(
            self.embedding_base_url, self.embedding_model
        )


def _env_flag(name: str) -> bool | None:
    """Parse a boolean env var, or None when unset.

    None lets the caller defer to the dataclass default instead of forcing a
    value. Truthy set: 1/true/yes/on (case-insensitive); anything else is False.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def resolve_env() -> EnvConfig:
    """Resolve all HyMem configuration from the environment.

    Never raises and never constructs network clients — safe for the doctor
    to call to report what *would* be used.
    """
    env = os.environ.get
    llm_key = env("HYMEM_LLM_API_KEY") or env("DEEPSEEK_API_KEY") or env("OPENAI_API_KEY")
    explicit_embedding = any(
        env(name)
        for name in (
            "HYMEM_EMBEDDING_API_KEY", "HYMEM_EMBEDDING_BASE_URL",
            "HYMEM_EMBEDDING_MODEL", "HYMEM_EMBEDDING_DIM",
        )
    )
    embedding_key: str | None = None
    embedding_fallback_reason: str | None = None
    if explicit_embedding:
        from hymem.contrib.openai_embedding_client import (
            is_loopback_embedding_url,
            is_official_openai_embedding_url,
            validate_embedding_base_url,
        )

        requested_base = env(
            "HYMEM_EMBEDDING_BASE_URL", DEFAULT_REMOTE_EMBEDDING_BASE_URL
        )
        requested_model = env(
            "HYMEM_EMBEDDING_MODEL", DEFAULT_REMOTE_EMBEDDING_MODEL
        )
        requested_dim = _env_positive_int(
            "HYMEM_EMBEDDING_DIM", DEFAULT_REMOTE_EMBEDDING_DIM
        )
        try:
            validate_embedding_base_url(requested_base)
        except (TypeError, ValueError):
            endpoint_valid = False
            embedding_fallback_reason = "remote_embedding_endpoint_rejected"
        else:
            endpoint_valid = True
        embedding_key = env("HYMEM_EMBEDDING_API_KEY")
        if (
            not embedding_key
            and endpoint_valid
            and is_official_openai_embedding_url(requested_base)
        ):
            embedding_key = env("OPENAI_API_KEY")
        if (
            not embedding_key
            and endpoint_valid
            and is_loopback_embedding_url(requested_base)
        ):
            embedding_key = "local"
        if embedding_key and endpoint_valid:
            embedding_base_url = requested_base
            embedding_model = requested_model
            embedding_dim = requested_dim
            embedding_backend = "openai_compatible"
        else:
            # An incomplete remote configuration must not instantiate a client
            # that is guaranteed to fail. Keep vector tiers alive in a separate,
            # explicitly lower-quality local vector space.
            embedding_base_url = DEFAULT_EMBEDDING_BASE_URL
            embedding_model = DEFAULT_EMBEDDING_MODEL
            embedding_dim = DEFAULT_EMBEDDING_DIM
            embedding_backend = "local_feature_hash"
            if embedding_fallback_reason is None:
                embedding_fallback_reason = "remote_embedding_credentials_missing"
    else:
        # In particular, a DeepSeek-only LLM environment lands here. DeepSeek
        # does not expose the old fictional `deepseek-embedding` model, so its
        # key is never silently reused for embeddings.
        embedding_base_url = DEFAULT_EMBEDDING_BASE_URL
        embedding_model = DEFAULT_EMBEDDING_MODEL
        embedding_dim = DEFAULT_EMBEDDING_DIM
        embedding_backend = "local_feature_hash"
    return EnvConfig(
        root=Path(env("HYMEM_ROOT", str(DEFAULT_ROOT))),
        llm_api_key=llm_key,
        llm_base_url=env("HYMEM_LLM_BASE_URL", DEFAULT_BASE_URL),
        llm_model=env("HYMEM_LLM_MODEL", DEFAULT_LLM_MODEL),
        embedding_api_key=embedding_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_backend=embedding_backend,
        embedding_fallback_reason=embedding_fallback_reason,
        aggregation_nodes_enabled=_env_flag("HYMEM_AGGREGATION_NODES_ENABLED"),
        aggregation_digest_enabled=_env_flag("HYMEM_AGGREGATION_DIGEST_ENABLED"),
    )


def build_from_env() -> HyMem:
    """Construct a HyMem instance from environment variables.

    Fails fast with a clear, actionable error if the extraction LLM key is
    missing — instead of raising deep inside the first dream cycle. The
    Embeddings default to a deterministic dependency-free local feature hash.
    An OpenAI-compatible endpoint is used only when explicitly configured.
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

    from hymem.extraction.embeddings import (
        CachedEmbeddingClient,
        LocalHashEmbeddingClient,
    )

    if cfg.embedding_backend == "openai_compatible":
        try:
            embedder = CachedEmbeddingClient(
                OpenAICompatibleEmbeddingClient(
                    api_key=cfg.embedding_api_key,
                    base_url=cfg.embedding_base_url,
                    model=cfg.embedding_model,
                    dim=cfg.embedding_dim,
                )
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            from hymem.contrib.openai_embedding_client import (
                safe_embedding_base_url,
            )
            log.warning(
                "configured embedding client unavailable at %s (%s); using "
                "deterministic local lexical fallback",
                safe_embedding_base_url(cfg.embedding_base_url),
                type(exc).__name__,
            )
            embedder = CachedEmbeddingClient(LocalHashEmbeddingClient(
                fallback_reason="remote_embedding_client_unavailable",
            ))
    else:
        embedder = CachedEmbeddingClient(LocalHashEmbeddingClient(
            dim_value=cfg.embedding_dim,
            model_name=cfg.embedding_model,
            fallback_reason=cfg.embedding_fallback_reason,
        ))
        log.info(
            "embeddings backend=%s model=%s dim=%d quality=lexical network=none%s",
            cfg.embedding_backend, cfg.embedding_model, cfg.embedding_dim,
            (
                f" fallback_reason={cfg.embedding_fallback_reason}"
                if cfg.embedding_fallback_reason else ""
            ),
        )

    # Only env vars that were actually set override the dataclass defaults, so
    # the shipped default (aggregation off) stays in force until the 3c flip.
    overrides: dict[str, object] = {}
    if cfg.aggregation_nodes_enabled is not None:
        overrides["aggregation_nodes_enabled"] = cfg.aggregation_nodes_enabled
    if cfg.aggregation_digest_enabled is not None:
        overrides["aggregation_digest_enabled"] = cfg.aggregation_digest_enabled

    mem_cfg = HyMemConfig(root=cfg.root, **overrides)
    if mem_cfg.aggregation_nodes_enabled:
        log.info(
            "aggregation layer enabled via env (digest=%s) — dream will report "
            "aggregation_nodes_built/reused; watch the dream.end log line",
            mem_cfg.aggregation_digest_enabled,
        )

    instance = HyMem(mem_cfg, llm=llm, embedding_client=embedder)
    _clear_orphaned_dream_lock(instance.conn)
    return instance


def _clear_orphaned_dream_lock(conn: sqlite3.Connection) -> None:
    """Clear the dreaming run_lock left by a previous unclean shutdown.

    run_dreaming releases its lock in a ``finally`` block, but SIGKILL,
    container OOM, or a hard crash skip that. Without this, the next dream
    cycles return ``skipped_locked=True`` until the 5-minute TTL fallback
    in ``_acquire_lock`` kicks in. Safe under the single-writer assumption
    enforced by ``get_instance``.
    """
    with contextlib.suppress(sqlite3.Error):
        cur = conn.execute("DELETE FROM run_lock WHERE name = 'dreaming'")
        if cur.rowcount > 0:
            conn.execute(
                "UPDATE dream_runs SET ended_at = CURRENT_TIMESTAMP, "
                "error = 'cleared at startup' WHERE ended_at IS NULL"
            )
            log.warning("dream.cleared_orphaned_lock from previous unclean shutdown")


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
