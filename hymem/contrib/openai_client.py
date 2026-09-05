from __future__ import annotations

import os
import math
import threading
import time
from typing import Callable
from urllib.parse import urlsplit

from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.retry import with_retry

# Accepted values for the thinking override (env var / constructor argument).
# "disabled" force-sends the vendor body key, "off"/"enabled" force-omit it.
_THINKING_MODES = {"auto", "disabled", "off", "enabled"}


class OpenAICompatibleClient:
    """LLMClient backed by any OpenAI-compatible HTTP endpoint.

    Works with DeepSeek, OpenAI, Together, local vLLM, etc.
    All constructor arguments fall back to environment variables so the server
    can be configured entirely via the shell environment.

    Environment variables (all optional if arguments are passed directly):
        HYMEM_LLM_API_KEY   — API key (falls back to OPENAI_API_KEY)
        HYMEM_LLM_BASE_URL  — base URL (default: https://api.deepseek.com)
        HYMEM_LLM_MODEL     — model name (default: deepseek-v4-flash)
        HYMEM_LLM_THINKING  — whether to send DeepSeek's `thinking` body key:
                              "auto" (default, send only on DeepSeek endpoints),
                              "disabled" (always send it, i.e. force reasoning
                              off), "off"/"enabled" (never send it).

    ``token_counter`` is an optional trusted tokenizer for the configured
    model. When supplied it is exposed as ``count_tokens`` so prompt packing
    and reported context usage share the exact same accounting. When omitted,
    the query layer uses its conservative offline fallback.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        thinking: str | None = None,
        token_counter: Callable[[str], int] | None = None,
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
        self.model = model or os.environ.get("HYMEM_LLM_MODEL") or "deepseek-v4-flash"
        self.base_url = resolved_base.rstrip("/")
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token_counter must be callable or None")
        self.count_tokens = token_counter

        # `thinking` is a DeepSeek-specific body key: deepseek-v4-flash otherwise
        # spends its whole token budget reasoning and returns empty content. But
        # the OpenAI API rejects unknown body params with a 400, and vLLM's
        # tolerance varies by version — and because every call goes through
        # with_retry(), an unconditional send turns "wrong vendor" into three
        # doomed attempts with sleeps in between rather than one clean failure.
        # So resolve the decision once, here, from the endpoint we actually
        # resolved above, and let an operator override it either way.
        mode = (
            thinking
            or os.environ.get("HYMEM_LLM_THINKING")
            or "auto"
        ).strip().lower()
        if mode not in _THINKING_MODES:
            raise ValueError(
                f"Invalid HYMEM_LLM_THINKING value {mode!r}; "
                f"expected one of {sorted(_THINKING_MODES)}."
            )
        if mode == "auto":
            # Substring match, not equality: DeepSeek is also reached via
            # regional hosts and reverse proxies that keep the vendor name in
            # the host or the model id.
            host = (urlsplit(resolved_base).hostname or "").lower()
            self._send_thinking = "deepseek" in host or "deepseek" in self.model.lower()
        else:
            self._send_thinking = mode == "disabled"
        self.thinking_mode = mode
        self.effective_extra_body = (
            {"thinking": {"type": "disabled"}} if self._send_thinking else {}
        )

        # Provider accounting is cumulative and thread-safe because benchmark
        # workers can share this client. Any failed attempt or response without
        # a complete usage block makes token totals unavailable rather than
        # falsely exact.
        self.call_count = 0
        self.request_attempts = 0
        self.successful_responses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.cost_usd = None
        self.token_usage_available = False
        self._usage_complete = True
        self._usage_lock = threading.Lock()

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
        # Omit the key entirely rather than passing an empty/None extra_body:
        # some servers reject a body they were not expecting at all.
        if self._send_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if request.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        def _attempt():
            with self._usage_lock:
                self.request_attempts += 1
            started = time.monotonic()
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception:
                with self._usage_lock:
                    self._usage_complete = False
                    self.token_usage_available = False
                raise
            finally:
                with self._usage_lock:
                    self.total_latency_s += time.monotonic() - started

        resp = with_retry(
            _attempt,
            label=f"LLM completion ({self.model})",
        )
        content = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        values = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        valid = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
            for value in values.values()
        )
        with self._usage_lock:
            self.call_count += 1
            self.successful_responses += 1
            if valid:
                self.prompt_tokens += values["prompt_tokens"]
                self.completion_tokens += values["completion_tokens"]
                self.total_tokens += values["total_tokens"]
            else:
                self._usage_complete = False
            self.token_usage_available = (
                self.successful_responses > 0 and self._usage_complete
            )
        return content
