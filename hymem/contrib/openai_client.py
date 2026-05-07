from __future__ import annotations

import os

from hymem.extraction.llm import LLMClient, LLMRequest


class OpenAICompatibleClient:
    """LLMClient backed by any OpenAI-compatible HTTP endpoint.

    Works with DeepSeek, OpenAI, Together, local vLLM, etc.
    All constructor arguments fall back to environment variables so the server
    can be configured entirely via the shell environment.

    Environment variables (all optional if arguments are passed directly):
        HYMEM_LLM_API_KEY   — API key (falls back to OPENAI_API_KEY)
        HYMEM_LLM_BASE_URL  — base URL (default: https://api.deepseek.com)
        HYMEM_LLM_MODEL     — model name (default: deepseek-chat)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required: pip install 'hymem[server]'"
            ) from e

        resolved_key = (
            api_key
            or os.environ.get("HYMEM_LLM_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not resolved_key:
            raise EnvironmentError(
                "No API key found. Set HYMEM_LLM_API_KEY (or DEEPSEEK_API_KEY)."
            )

        resolved_base = (
            base_url
            or os.environ.get("HYMEM_LLM_BASE_URL")
            or "https://api.deepseek.com"
        )
        self.model = model or os.environ.get("HYMEM_LLM_MODEL") or "deepseek-chat"
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base)

    def complete(self, request: LLMRequest) -> str:
        kwargs: dict = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user",   "content": request.user},
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        if request.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
