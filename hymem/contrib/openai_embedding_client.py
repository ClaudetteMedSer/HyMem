from __future__ import annotations

import os
from typing import Sequence


class OpenAICompatibleEmbeddingClient:
    """EmbeddingClient backed by any OpenAI-compatible HTTP endpoint.

    Works with DeepSeek, OpenAI, Together, local vLLM, etc.
    All constructor arguments fall back to environment variables so the server
    can be configured entirely via the shell environment.

    Environment variables (all optional if arguments are passed directly):
        HYMEM_EMBEDDING_API_KEY   — API key (falls back to HYMEM_LLM_API_KEY,
                                    DEEPSEEK_API_KEY, OPENAI_API_KEY)
        HYMEM_EMBEDDING_BASE_URL  — base URL (default: https://api.deepseek.com)
        HYMEM_EMBEDDING_MODEL     — model name (default: deepseek-embedding)
        HYMEM_EMBEDDING_DIM       — declared dim (default: 1024); the actual
                                    dim is read from the API response on the
                                    first call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required: pip install 'hymem[server]'"
            ) from e

        resolved_key = (
            api_key
            or os.environ.get("HYMEM_EMBEDDING_API_KEY")
            or os.environ.get("HYMEM_LLM_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not resolved_key:
            raise EnvironmentError(
                "No API key found. Set HYMEM_EMBEDDING_API_KEY (or HYMEM_LLM_API_KEY)."
            )

        resolved_base = (
            base_url
            or os.environ.get("HYMEM_EMBEDDING_BASE_URL")
            or "https://api.deepseek.com"
        )
        self._model = (
            model
            or os.environ.get("HYMEM_EMBEDDING_MODEL")
            or "deepseek-embedding"
        )
        env_dim = os.environ.get("HYMEM_EMBEDDING_DIM")
        self._dim = dim if dim is not None else (int(env_dim) if env_dim else 1024)
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        vectors = [list(d.embedding) for d in resp.data]
        if vectors:
            self._dim = len(vectors[0])
        return vectors
